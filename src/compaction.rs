use crate::backend::LlmBackend;
use crate::session::Session;
use crate::types::{AgentEvent, ContentBlock, Message, RequestConfig, Role, StreamEvent};
use anyhow::Result;
use futures::StreamExt;
use futures::channel::mpsc;
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as TokioMutex;

/// Compaction strategy: LLM summarization or simple truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CompactionStrategy {
    /// Use an LLM call to summarize the removed messages.
    #[default]
    Summarize,
    /// Simply drop the oldest messages with a stub placeholder.
    Truncate,
}

impl std::fmt::Display for CompactionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Summarize => write!(f, "summarize"),
            Self::Truncate => write!(f, "truncate"),
        }
    }
}

const SUMMARIZE_SYSTEM_PROMPT: &str = "You are a conversation summarizer. Your job is to produce a concise but informationally dense summary of a conversation that is being compacted to free up context window space.\n\nYour summary MUST preserve:\n- Key decisions made and their rationale\n- Important facts, findings, or conclusions established\n- Tool calls that were made and their significant results (especially file edits, search results, or bash outputs that changed state)\n- Errors encountered and how they were resolved\n- Any ongoing work or open questions that the assistant should be aware of\n\nDo NOT include:\n- Greetings, pleasantries, or filler\n- Verbatim copies of large outputs (summarize them instead)\n- Information that was superseded or corrected later\n\nFormat the summary as a structured note that the assistant can use as context. Start with [CONTEXT SUMMARY] and be as specific as possible about what was done and why. Keep it under 500 words.";

const SUMMARIZE_USER_PROMPT: &str = "Summarize the following conversation history. This summary will replace the original messages in the conversation, so preserve all important information that the assistant would need to continue the conversation coherently.";

/// Format a slice of messages into a human-readable conversation transcript
/// suitable for the LLM summarization prompt.
pub fn format_messages_for_summary(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
        };
        for block in &msg.content {
            match block {
                ContentBlock::Text(t) => {
                    out.push_str(&format!("{role}: {t}\n\n"));
                }
                ContentBlock::ToolUse { id, name, input } => {
                    out.push_str(&format!(
                        "{role}: [Called tool '{name}' (id={id}) with input: {input}]\n\n"
                    ));
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let label = if *is_error { "ERROR" } else { "RESULT" };
                    out.push_str(&format!(
                        "{role}: [Tool result for {tool_use_id} ({label}): {content}]\n\n"
                    ));
                }
            }
        }
    }
    out
}

/// Call the LLM backend to produce a summary of the given messages.
///
/// Returns the summary text on success, or an error if the backend call fails.
pub async fn summarize_messages(
    backend: &Arc<dyn LlmBackend>,
    config: &RequestConfig,
    messages: &[Message],
) -> Result<String> {
    let transcript = format_messages_for_summary(messages);

    let summarization_messages = vec![
        Message::text(Role::User, SUMMARIZE_SYSTEM_PROMPT.to_string()),
        Message::text(Role::Assistant, "I will summarize the conversation history you provide, preserving all key decisions, tool results, and important context.".to_string()),
        Message::text(Role::User, SUMMARIZE_USER_PROMPT.to_string()),
        Message::text(Role::Assistant, "Please provide the conversation history and I will produce a concise summary.".to_string()),
        Message::text(Role::User, transcript),
    ];

    let mut stream = backend
        .send_message(&summarization_messages, config)
        .await?;

    let mut summary = String::new();
    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::TextDelta(text)) => {
                summary.push_str(&text);
            }
            Ok(StreamEvent::Done) => break,
            Ok(StreamEvent::Usage { .. }) => {}
            Ok(StreamEvent::ToolUseStart { .. }) => {}
            Ok(StreamEvent::ToolUseDelta(_)) => {}
            Ok(StreamEvent::ToolUseDone) => {}
            Err(e) => {
                anyhow::bail!("LLM summarization call failed: {e}");
            }
        }
    }

    if summary.trim().is_empty() {
        anyhow::bail!("LLM summarization returned empty response");
    }

    Ok(summary.trim().to_string())
}

/// Compute which messages to compact (identifying IDs and the cut point).
///
/// Returns `None` if the history is too short to compact.
///
/// # Arguments
/// * `history` — Full conversation history including context prefix.
/// * `context_prefix_len` — Number of messages at the front that are the
///   non-persisted context prefix. These are never compacted.
/// * `keep_recent` — Number of messages at the tail to keep verbatim.
///
/// # Invariants
/// * Context prefix is never touched.
/// * Tool-use pairs (ToolUse + matching ToolResult) are never split across
///   the compaction boundary.
pub fn compute_compaction_cut(
    history: &[(i64, Message)],
    context_prefix_len: usize,
    keep_recent: usize,
) -> Option<(Vec<i64>, usize)> {
    let persisted_len = history.len().saturating_sub(context_prefix_len);
    if persisted_len <= keep_recent {
        return None;
    }

    let max_compactable = persisted_len.saturating_sub(keep_recent);
    if max_compactable == 0 {
        return None;
    }

    let cut = find_safe_cut(history, context_prefix_len, max_compactable)?;

    let ids_to_deactivate: Vec<i64> = history[context_prefix_len..cut]
        .iter()
        .map(|(id, _)| *id)
        .collect();

    if ids_to_deactivate.is_empty() {
        return None;
    }

    let messages_kept = persisted_len - ids_to_deactivate.len();

    Some((ids_to_deactivate, messages_kept))
}

/// Build a truncation-style summary message.
pub fn truncation_summary_message(messages_removed: usize, messages_kept: usize) -> Message {
    Message::text(
        Role::User,
        format!(
            "[CONTEXT SUMMARY] Truncation: {messages_removed} older messages \
             removed, {messages_kept} recent messages retained."
        ),
    )
}

/// Build a summarization-style summary message wrapping LLM-generated text.
pub fn summarize_summary_message(
    llm_summary: &str,
    messages_removed: usize,
    messages_kept: usize,
) -> Message {
    Message::text(
        Role::User,
        format!(
            "[CONTEXT SUMMARY] {messages_removed} older messages were compacted into this summary. \
             {messages_kept} recent messages are retained verbatim below.\n\n{llm_summary}"
        ),
    )
}

/// Find a safe index to cut the history so that no tool-use pair is split.
///
/// A tool-use pair spans two messages: an assistant message with ToolUse block(s)
/// followed by a user message with ToolResult block(s). We must not cut between
/// these two messages.
///
/// The cut is between messages `i-1` and `i`. It is unsafe if:
/// - Message `i-1` contains ToolUse blocks (the matching ToolResult is in message `i`), OR
/// - Message `i` contains ToolResult blocks (matching ToolUse in message `i-1`).
fn find_safe_cut(
    history: &[(i64, Message)],
    context_prefix_len: usize,
    max_cut: usize,
) -> Option<usize> {
    let cut_end = context_prefix_len + max_cut;

    // Walk backward from cut_end looking for a safe boundary.
    for i in (context_prefix_len + 1..=cut_end).rev() {
        if i >= history.len() {
            continue;
        }
        let prev_msg = &history[i - 1].1;
        let next_msg = if i < history.len() {
            &history[i].1
        } else {
            break;
        };

        let prev_has_tool_use = prev_msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }));
        let next_has_tool_result = next_msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }));

        if !prev_has_tool_use && !next_has_tool_result {
            return Some(i);
        }
    }

    // Fallback: cut at context_prefix_len (compact everything possible).
    // This only happens if the entire persisted range is one giant tool-use chain.
    if cut_end > context_prefix_len {
        Some(context_prefix_len)
    } else {
        None
    }
}

/// Context needed for LLM-based summarization during compaction.
pub struct SummarizeContext {
    pub backend: Arc<dyn LlmBackend>,
    pub config: RequestConfig,
}

/// Execute compaction against the session DB and refresh in-memory history.
///
/// When `strategy` is `Summarize`, an LLM call is made to produce a concise
/// summary of the removed messages. If the LLM call fails, we fall back to
/// truncation.
///
/// Returns the number of messages removed, kept, and the strategy actually used
/// (may differ from requested if fallback occurred).
pub async fn execute_compaction(
    history_arc: &Arc<Mutex<Vec<Message>>>,
    session: &Arc<TokioMutex<Session>>,
    context_prefix_len: &Arc<Mutex<usize>>,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    keep_recent: usize,
    strategy: CompactionStrategy,
    summarize_ctx: Option<&SummarizeContext>,
) -> Result<(usize, usize, CompactionStrategy)> {
    let prefix_len = *context_prefix_len.lock().unwrap_or_else(|e| e.into_inner());

    // Load active history with DB row IDs.
    let persisted: Vec<(i64, Message)> = {
        let sess = session.lock().await;
        sess.conversation().load_history_with_ids().await?
    };

    let (ids_to_deactivate, messages_kept) =
        match compute_compaction_cut(&persisted, 0, keep_recent) {
            Some(cut) => cut,
            None => return Ok((0, 0, strategy)),
        };

    let messages_removed = ids_to_deactivate.len();

    let (summary_message, actual_strategy) = if strategy == CompactionStrategy::Summarize {
        // Extract the messages being removed so we can summarize them.
        let messages_to_summarize: Vec<Message> = persisted
            .iter()
            .filter(|(id, _)| ids_to_deactivate.contains(id))
            .map(|(_, msg)| msg.clone())
            .collect();

        match try_summarize(summarize_ctx, &messages_to_summarize, event_tx).await {
            Ok(llm_summary) => (
                summarize_summary_message(&llm_summary, messages_removed, messages_kept),
                CompactionStrategy::Summarize,
            ),
            Err(e) => {
                let _ = event_tx.unbounded_send(AgentEvent::Warn(format!(
                    "LLM summarization failed, falling back to truncation: {e}"
                )));
                (
                    truncation_summary_message(messages_removed, messages_kept),
                    CompactionStrategy::Truncate,
                )
            }
        }
    } else {
        (
            truncation_summary_message(messages_removed, messages_kept),
            CompactionStrategy::Truncate,
        )
    };

    // Execute against DB.
    {
        let sess = session.lock().await;
        sess.conversation()
            .deactivate_messages(&ids_to_deactivate)
            .await?;
        sess.conversation().insert_message(&summary_message).await?;
    }

    // Refresh in-memory history: preserve prefix, reload from DB.
    {
        let new_db_history = {
            let sess = session.lock().await;
            sess.conversation().load_history().await?
        };
        let mut history = lock(history_arc);
        let take = prefix_len.min(history.len());
        let prefix: Vec<Message> = history.drain(..take).collect();
        *history = prefix;
        history.extend(new_db_history);
    }

    let _ = event_tx.unbounded_send(AgentEvent::Warn(format!(
        "compaction complete ({actual_strategy}): {messages_removed} removed, {messages_kept} kept"
    )));

    Ok((messages_removed, messages_kept, actual_strategy))
}

/// Attempt LLM summarization with the available backend.
async fn try_summarize(
    summarize_ctx: Option<&SummarizeContext>,
    messages: &[Message],
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
) -> Result<String> {
    let ctx =
        summarize_ctx.ok_or_else(|| anyhow::anyhow!("no backend available for summarization"))?;

    let _ = event_tx.unbounded_send(AgentEvent::Warn(
        "requesting LLM summary of compacted messages...".to_string(),
    ));

    summarize_messages(&ctx.backend, &ctx.config, messages).await
}

/// Recover from a poisoned mutex: a thread panicked while holding the lock, leaving
/// history in an unknown state. Panicking here would crash the app; accepting partial
/// corruption is the lesser evil for a long-running interactive process.
fn lock(m: &Mutex<Vec<Message>>) -> std::sync::MutexGuard<'_, Vec<Message>> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_history(messages: Vec<(String, Vec<ContentBlock>)>) -> Vec<(i64, Message)> {
        messages
            .into_iter()
            .enumerate()
            .map(|(i, (role_str, content))| {
                let role = if role_str == "user" {
                    Role::User
                } else {
                    Role::Assistant
                };
                ((i + 1) as i64, Message { role, content })
            })
            .collect()
    }

    fn text_msg(role: &str, text: &str) -> (String, Vec<ContentBlock>) {
        (role.to_string(), vec![ContentBlock::Text(text.to_string())])
    }

    fn tool_use_msg(id: &str, name: &str) -> (String, Vec<ContentBlock>) {
        (
            "assistant".to_string(),
            vec![
                ContentBlock::Text("using tool".to_string()),
                ContentBlock::ToolUse {
                    id: id.to_string(),
                    name: name.to_string(),
                    input: serde_json::json!({}),
                },
            ],
        )
    }

    fn tool_result_msg(id: &str, content: &str) -> (String, Vec<ContentBlock>) {
        (
            "user".to_string(),
            vec![ContentBlock::ToolResult {
                tool_use_id: id.to_string(),
                content: content.to_string(),
                is_error: false,
            }],
        )
    }

    #[test]
    fn compact_empty_history_returns_none() {
        let history: Vec<(i64, Message)> = vec![];
        assert!(compute_compaction_cut(&history, 0, 6).is_none());
    }

    #[test]
    fn compact_below_keep_recent_returns_none() {
        let history = make_history(vec![text_msg("user", "hello"), text_msg("assistant", "hi")]);
        assert!(compute_compaction_cut(&history, 0, 2).is_none());
    }

    #[test]
    fn compact_with_context_prefix_preserves_prefix() {
        let history = make_history(vec![
            text_msg("user", "context prefix 1"),
            text_msg("user", "context prefix 2"),
            text_msg("user", "hello"),
            text_msg("assistant", "hi"),
            text_msg("user", "how are you"),
            text_msg("assistant", "fine"),
        ]);

        let (ids, kept) = compute_compaction_cut(&history, 2, 2).expect("should compact");
        assert!(ids.contains(&3));
        assert!(ids.contains(&4));
        assert!(!ids.contains(&1));
        assert!(!ids.contains(&2));
        assert_eq!(kept, 2);
    }

    #[test]
    fn compact_preserves_recent_messages() {
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            text_msg("user", "msg3"),
            text_msg("assistant", "msg4"),
            text_msg("user", "msg5"),
            text_msg("assistant", "msg6"),
        ]);

        let (ids, kept) = compute_compaction_cut(&history, 0, 4).expect("should compact");
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert_eq!(kept, 4);
        assert!(!ids.contains(&3));
        assert!(!ids.contains(&4));
        assert!(!ids.contains(&5));
        assert!(!ids.contains(&6));
    }

    #[test]
    fn compact_maintains_tool_use_pairs() {
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            tool_use_msg("t1", "bash"),
            tool_result_msg("t1", "output"),
            text_msg("user", "msg5"),
            text_msg("assistant", "msg6"),
        ]);

        let (ids, _) = compute_compaction_cut(&history, 0, 2).expect("should compact");
        let has_3 = ids.contains(&3);
        let has_4 = ids.contains(&4);
        assert_eq!(
            has_3, has_4,
            "tool-use pair (ids 3,4) must not be split: ids={ids:?}"
        );
    }

    #[test]
    fn truncation_summary_message_format() {
        let msg = truncation_summary_message(5, 3);
        match &msg.content[0] {
            ContentBlock::Text(t) => {
                assert!(t.starts_with("[CONTEXT SUMMARY]"));
                assert!(t.contains("5 older messages"));
                assert!(t.contains("3 recent messages"));
            }
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn summarize_summary_message_includes_llm_text() {
        let msg = summarize_summary_message("The user asked about Rust.", 4, 2);
        match &msg.content[0] {
            ContentBlock::Text(t) => {
                assert!(t.starts_with("[CONTEXT SUMMARY]"));
                assert!(t.contains("The user asked about Rust."));
                assert!(t.contains("4 older messages"));
                assert!(t.contains("2 recent messages"));
            }
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn compact_configurable_keep_recent() {
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            text_msg("user", "msg3"),
            text_msg("assistant", "msg4"),
            text_msg("user", "msg5"),
            text_msg("assistant", "msg6"),
        ]);

        let (ids, kept) = compute_compaction_cut(&history, 0, 2).expect("should compact");
        assert_eq!(ids.len(), 4);
        assert_eq!(kept, 2);

        let (ids2, kept2) = compute_compaction_cut(&history, 0, 4).expect("should compact");
        assert_eq!(ids2.len(), 2);
        assert_eq!(kept2, 4);
    }

    #[test]
    fn compact_context_prefix_only_returns_none() {
        let history = make_history(vec![
            text_msg("user", "prefix1"),
            text_msg("user", "prefix2"),
        ]);
        assert!(compute_compaction_cut(&history, 2, 0).is_none());
    }

    #[test]
    fn truncation_basic() {
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            text_msg("user", "msg3"),
            text_msg("assistant", "msg4"),
        ]);

        let (ids, kept) = compute_compaction_cut(&history, 0, 2).expect("should compute cut");
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert_eq!(kept, 2);
    }

    #[test]
    fn truncation_respects_tool_pairs() {
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            tool_use_msg("t1", "bash"),
            tool_result_msg("t1", "output"),
            text_msg("user", "msg5"),
            text_msg("assistant", "msg6"),
        ]);

        let (ids, _) = compute_compaction_cut(&history, 0, 2).expect("should compute cut");
        let has_3 = ids.contains(&3);
        let has_4 = ids.contains(&4);
        assert_eq!(
            has_3, has_4,
            "tool-use pair must not be split in truncation"
        );
    }

    #[test]
    fn compact_with_consecutive_tool_pairs() {
        let history = make_history(vec![
            text_msg("user", "msg1"),
            tool_use_msg("t1", "bash"),
            tool_result_msg("t1", "out1"),
            tool_use_msg("t2", "bash"),
            tool_result_msg("t2", "out2"),
            text_msg("user", "msg6"),
        ]);

        let (ids, _) = compute_compaction_cut(&history, 0, 2).expect("should compact");
        let has_2 = ids.contains(&2);
        let has_3 = ids.contains(&3);
        let has_4 = ids.contains(&4);
        let has_5 = ids.contains(&5);
        assert_eq!(has_2, has_3, "pair t1 must be together");
        assert_eq!(has_4, has_5, "pair t2 must be together");
    }

    #[test]
    fn compact_keeps_at_least_keep_recent_even_with_tool_pairs() {
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            tool_use_msg("t1", "bash"),
            tool_result_msg("t1", "output"),
            text_msg("user", "msg5"),
            text_msg("assistant", "msg6"),
        ]);

        let (ids, _) = compute_compaction_cut(&history, 0, 3).expect("should compact");
        assert!(!ids.contains(&3));
        assert!(!ids.contains(&4));
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn format_messages_for_summary_text_only() {
        let messages = vec![
            Message::text(Role::User, "hello".to_string()),
            Message::text(Role::Assistant, "hi there".to_string()),
        ];
        let formatted = format_messages_for_summary(&messages);
        assert!(formatted.contains("User: hello"));
        assert!(formatted.contains("Assistant: hi there"));
    }

    #[test]
    fn format_messages_for_summary_includes_tool_use() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text("let me check".to_string()),
                ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "ls"}),
                },
            ],
        }];
        let formatted = format_messages_for_summary(&messages);
        assert!(formatted.contains("let me check"));
        assert!(formatted.contains("Called tool 'bash'"));
        assert!(formatted.contains("id=t1"));
    }

    #[test]
    fn format_messages_for_summary_includes_tool_result() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".to_string(),
                content: "file.txt".to_string(),
                is_error: false,
            }],
        }];
        let formatted = format_messages_for_summary(&messages);
        assert!(formatted.contains("Tool result for t1"));
        assert!(formatted.contains("RESULT"));
        assert!(formatted.contains("file.txt"));
    }

    #[test]
    fn format_messages_for_summary_error_result() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".to_string(),
                content: "command not found".to_string(),
                is_error: true,
            }],
        }];
        let formatted = format_messages_for_summary(&messages);
        assert!(formatted.contains("ERROR"));
    }

    #[test]
    fn compaction_strategy_default_is_summarize() {
        assert_eq!(CompactionStrategy::default(), CompactionStrategy::Summarize);
    }

    #[test]
    fn compaction_strategy_display() {
        assert_eq!(format!("{}", CompactionStrategy::Summarize), "summarize");
        assert_eq!(format!("{}", CompactionStrategy::Truncate), "truncate");
    }

    #[test]
    fn compaction_strategy_serde_roundtrip() {
        let json = serde_json::to_string(&CompactionStrategy::Summarize).expect("serialize");
        assert_eq!(json, "\"summarize\"");
        let deserialized: CompactionStrategy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, CompactionStrategy::Summarize);

        let json = serde_json::to_string(&CompactionStrategy::Truncate).expect("serialize");
        assert_eq!(json, "\"truncate\"");
        let deserialized: CompactionStrategy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, CompactionStrategy::Truncate);
    }

    #[tokio::test]
    async fn summarize_messages_calls_backend_and_returns_summary() {
        use async_trait::async_trait;
        use futures::stream;

        struct SummaryBackend;

        #[async_trait]
        impl LlmBackend for SummaryBackend {
            async fn send_message(
                &self,
                _messages: &[Message],
                _config: &RequestConfig,
            ) -> anyhow::Result<crate::types::BoxStream<Result<StreamEvent>>> {
                Ok(Box::pin(stream::iter(vec![
                    Ok(StreamEvent::TextDelta("[CONTEXT SUMMARY] ".to_string())),
                    Ok(StreamEvent::TextDelta(
                        "User asked about Rust. Decision: use tokio.".to_string(),
                    )),
                    Ok(StreamEvent::Done),
                ])))
            }
        }

        let backend: Arc<dyn LlmBackend> = Arc::new(SummaryBackend);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 1024,
            tools: vec![],
        };

        let messages = vec![
            Message::text(Role::User, "Tell me about Rust".to_string()),
            Message::text(
                Role::Assistant,
                "Rust is great. Let's use tokio.".to_string(),
            ),
        ];

        let result = summarize_messages(&backend, &config, &messages).await;
        let summary = result.expect("summarization should succeed");
        assert!(summary.contains("[CONTEXT SUMMARY]"));
        assert!(summary.contains("tokio"));
    }

    #[tokio::test]
    async fn summarize_messages_empty_response_fails() {
        use async_trait::async_trait;
        use futures::stream;

        struct EmptyBackend;

        #[async_trait]
        impl LlmBackend for EmptyBackend {
            async fn send_message(
                &self,
                _messages: &[Message],
                _config: &RequestConfig,
            ) -> anyhow::Result<crate::types::BoxStream<Result<StreamEvent>>> {
                Ok(Box::pin(stream::iter(vec![
                    Ok(StreamEvent::TextDelta("   ".to_string())),
                    Ok(StreamEvent::Done),
                ])))
            }
        }

        let backend: Arc<dyn LlmBackend> = Arc::new(EmptyBackend);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 1024,
            tools: vec![],
        };

        let messages = vec![Message::text(Role::User, "hello".to_string())];
        let result = summarize_messages(&backend, &config, &messages).await;
        assert!(result.is_err(), "empty summary should fail");
    }

    #[tokio::test]
    async fn summarize_messages_stream_error_fails() {
        use async_trait::async_trait;
        use futures::stream;

        struct ErrorBackend;

        #[async_trait]
        impl LlmBackend for ErrorBackend {
            async fn send_message(
                &self,
                _messages: &[Message],
                _config: &RequestConfig,
            ) -> anyhow::Result<crate::types::BoxStream<Result<StreamEvent>>> {
                Ok(Box::pin(stream::iter(vec![Err(anyhow::anyhow!(
                    "backend exploded"
                ))])))
            }
        }

        let backend: Arc<dyn LlmBackend> = Arc::new(ErrorBackend);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 1024,
            tools: vec![],
        };

        let messages = vec![Message::text(Role::User, "hello".to_string())];
        let result = summarize_messages(&backend, &config, &messages).await;
        assert!(result.is_err(), "stream error should cause failure");
    }
}
