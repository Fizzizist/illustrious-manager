use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use futures::channel::mpsc;
use tokio::sync::Mutex as TokioMutex;

use crate::session::Session;
use crate::types::{AgentEvent, ContentBlock, Message};

use super::{Agent, AgentSpawner, lock};

/// Check whether auto-compaction should be triggered based on peak input
/// tokens and the configured threshold. Emits `AutoCompactTriggered` or
/// `Warn` as appropriate, and updates the consecutive-compaction guard flag.
pub(crate) fn check_auto_compact(
    peak_input_tokens: u32,
    max_context_window_len: u32,
    last_auto_compacted: &AtomicBool,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
) {
    if max_context_window_len == 0 {
        return;
    }

    if peak_input_tokens > max_context_window_len {
        if last_auto_compacted.load(Ordering::SeqCst) {
            let _ = event_tx.unbounded_send(AgentEvent::Warn(format!(
                "Context still exceeds threshold ({} > {}) after auto-compaction. \
                 Manual /compact or starting a new session is recommended.",
                peak_input_tokens, max_context_window_len
            )));
        } else {
            let _ = event_tx.unbounded_send(AgentEvent::AutoCompactTriggered {
                current_tokens: peak_input_tokens,
                threshold: max_context_window_len,
            });
            last_auto_compacted.store(true, Ordering::SeqCst);
        }
    } else {
        last_auto_compacted.store(false, Ordering::SeqCst);
    }
}

/// Count the number of turns in a message slice.
/// A "turn" starts with a user message and includes all immediately following
/// assistant messages. Orphan assistant messages at the start form their own turn.
fn count_turns(messages: &[Message]) -> usize {
    if messages.is_empty() {
        return 0;
    }
    let mut turns = 0;
    let mut i = 0;
    while i < messages.len() {
        turns += 1;
        i += 1;
        while i < messages.len() && messages[i].role == crate::types::Role::Assistant {
            i += 1;
        }
    }
    turns
}

/// Find the start index of the last `n` turns in the message slice.
/// Returns `messages.len()` if there are fewer than `n` turns (i.e., all
/// messages are retained).
fn retained_start_index(messages: &[Message], n: usize) -> usize {
    if n == 0 || messages.is_empty() {
        return 0;
    }
    let total = count_turns(messages);
    let skip = total.saturating_sub(n);
    let mut turn_start = 0;
    let mut turns_seen = 0;
    let mut i = 0;
    while i < messages.len() && turns_seen < skip {
        turn_start = i + 1;
        turns_seen += 1;
        i += 1;
        while i < messages.len() && messages[i].role == crate::types::Role::Assistant {
            turn_start = i + 1;
            i += 1;
        }
    }
    turn_start
}

/// Strip non-text blocks from a message, returning `None` if the result is empty.
fn strip_non_text(msg: &Message) -> Option<Message> {
    let stripped: Vec<ContentBlock> = msg
        .content
        .iter()
        .filter(|b| matches!(b, ContentBlock::Text(_)))
        .cloned()
        .collect();
    if stripped.is_empty() {
        None
    } else {
        Some(Message {
            role: msg.role.clone(),
            content: stripped,
        })
    }
}

/// Extracted core compaction logic so it can be called from `Agent::compact()`
/// without needing `&self`.
pub async fn compact_with(
    spawner: &Arc<AgentSpawner>,
    history: &Arc<Mutex<Vec<Message>>>,
    session: &Arc<TokioMutex<Session>>,
    context_prefix_len: &Arc<Mutex<usize>>,
    num_retained_turns: u32,
) -> Result<String, String> {
    let compaction_role = spawner.app_config.compaction.role.clone();

    let role = if spawner.app_config.models.contains_key(&compaction_role) {
        compaction_role
    } else {
        "default".to_string()
    };

    let (prompt, retained_messages) = {
        let hist = lock(history);
        let prefix_len = *context_prefix_len.lock().unwrap_or_else(|e| e.into_inner());
        let persisted = &hist[prefix_len.min(hist.len())..];

        if num_retained_turns == 0 {
            let mut parts = Vec::new();
            for msg in persisted {
                let role_label = match msg.role {
                    crate::types::Role::User => "User",
                    crate::types::Role::Assistant => "Assistant",
                };
                for block in &msg.content {
                    match block {
                        ContentBlock::Text(text) => {
                            parts.push(format!("{role_label}: {text}"));
                        }
                        ContentBlock::ToolUse { name, input, .. } => {
                            parts.push(format!(
                                "{role_label}: [Called tool {name} with input {input}]"
                            ));
                        }
                        ContentBlock::ToolResult {
                            content, is_error, ..
                        } => {
                            let label = if *is_error { "error" } else { "result" };
                            parts.push(format!("{role_label}: [Tool {label}: {content}]"));
                        }
                        ContentBlock::Thinking { text, .. } => {
                            parts.push(format!("{role_label}: [Thinking: {text}]"));
                        }
                        ContentBlock::RedactedThinking { .. } => {
                            parts.push(format!("{role_label}: [Redacted thinking]"));
                        }
                    }
                }
            }
            if parts.is_empty() {
                return Ok("Nothing to compact: the conversation is empty.".to_string());
            }
            let prompt = format!(
                "Summarize the following conversation concisely, preserving key facts, decisions, and context that would be needed to continue the conversation. Do not include meta-commentary — output only the summary.\n\n{}",
                parts.join("\n\n")
            );
            (prompt, Vec::<Message>::new())
        } else {
            let retained_start = retained_start_index(persisted, num_retained_turns as usize);

            let compacted = &persisted[..retained_start];
            let retained = &persisted[retained_start..];

            if compacted.is_empty() {
                return Ok("Nothing to compact: all turns are retained.".to_string());
            }

            let mut parts = Vec::new();
            for msg in compacted {
                let role_label = match msg.role {
                    crate::types::Role::User => "User",
                    crate::types::Role::Assistant => "Assistant",
                };
                for block in &msg.content {
                    match block {
                        ContentBlock::Text(text) => {
                            parts.push(format!("{role_label}: {text}"));
                        }
                        ContentBlock::ToolUse { name, input, .. } => {
                            parts.push(format!(
                                "{role_label}: [Called tool {name} with input {input}]"
                            ));
                        }
                        ContentBlock::ToolResult {
                            content, is_error, ..
                        } => {
                            let label = if *is_error { "error" } else { "result" };
                            parts.push(format!("{role_label}: [Tool {label}: {content}]"));
                        }
                        ContentBlock::Thinking { text, .. } => {
                            parts.push(format!("{role_label}: [Thinking: {text}]"));
                        }
                        ContentBlock::RedactedThinking { .. } => {
                            parts.push(format!("{role_label}: [Redacted thinking]"));
                        }
                    }
                }
            }
            if parts.is_empty() {
                return Ok("Nothing to compact: the conversation is empty.".to_string());
            }
            let prompt = format!(
                "Summarize the following conversation concisely, preserving key facts, decisions, and context that would be needed to continue the conversation. Do not include meta-commentary — output only the summary.\n\n{}",
                parts.join("\n\n")
            );
            let stripped_retained: Vec<Message> =
                retained.iter().filter_map(strip_non_text).collect();
            (prompt, stripped_retained)
        }
    };

    let outcome = spawner
        .spawn(
            &role,
            crate::config::ConfirmationMode::Never,
            Some(&[]),
            prompt,
        )
        .await;

    if outcome.is_error {
        let msg = outcome
            .error_message
            .unwrap_or_else(|| "Compaction failed with an unknown error.".to_string());
        return Err(msg);
    }

    let summary = outcome.text;
    if summary.trim().is_empty() {
        return Err("Compaction produced an empty summary.".to_string());
    }

    let summary_msg = Message::text(crate::types::Role::User, format!("[Compacted] {summary}"));

    let sess = session.lock().await;

    if num_retained_turns == 0 {
        if let Err(e) = sess.conversation().compact(&summary_msg).await {
            return Err(format!("Compaction failed: {e}"));
        }
    } else {
        let result = sess
            .conversation()
            .compact_retaining(&summary_msg, num_retained_turns)
            .await
            .map_err(|e| format!("Compaction failed: {e}"))?;
        if !result {
            drop(sess);
            return Ok("Nothing to compact: all turns are retained.".to_string());
        }
    }

    drop(sess);

    let prefix_len_val = *context_prefix_len.lock().unwrap_or_else(|e| e.into_inner());
    let prefix: Vec<Message> = {
        let hist = lock(history);
        let take = prefix_len_val.min(hist.len());
        hist[..take].to_vec()
    };
    lock(history).clear();
    lock(history).extend(prefix);
    lock(history).push(summary_msg.clone());
    lock(history).extend(retained_messages);

    Ok(summary)
}

impl Agent {
    /// Run context compaction: summarise the active conversation history using a
    /// headless sub-agent, then replace it with the summary while preserving the
    /// context prefix (skills, CLAUDE.md, etc.).
    ///
    /// When `num_retained_turns > 0`, the last N turns are preserved after the
    /// summary with tool-use blocks stripped, maintaining conversational continuity.
    ///
    /// Returns the summary text on success, or an error message on failure.
    pub async fn compact(&self) -> Result<String, String> {
        let spawner = match &self.compaction_spawner {
            Some(s) => Arc::clone(s),
            None => {
                return Err("Compaction not available: no spawner configured.".to_string());
            }
        };
        compact_with(
            &spawner,
            &self.history,
            &self.session,
            &self.context_prefix_len,
            self.num_retained_turns,
        )
        .await
    }

    /// Reset the consecutive auto-compaction guard flag.
    ///
    /// Called by the TUI when compaction fails, so subsequent turns can
    /// retry auto-compaction instead of being permanently blocked by the
    /// guard.
    pub fn reset_auto_compact_flag(&self) {
        self.last_auto_compacted
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentBlock, Role};

    #[test]
    fn count_turns_empty() {
        assert_eq!(count_turns(&[]), 0);
    }

    #[test]
    fn count_turns_single_user() {
        let msgs = vec![Message::text(Role::User, "hello".to_string())];
        assert_eq!(count_turns(&msgs), 1);
    }

    #[test]
    fn count_turns_single_turn() {
        let msgs = vec![
            Message::text(Role::User, "hello".to_string()),
            Message::text(Role::Assistant, "hi".to_string()),
        ];
        assert_eq!(count_turns(&msgs), 1);
    }

    #[test]
    fn count_turns_multiple_turns() {
        let msgs = vec![
            Message::text(Role::User, "hello".to_string()),
            Message::text(Role::Assistant, "hi".to_string()),
            Message::text(Role::User, "how are you".to_string()),
            Message::text(Role::Assistant, "fine".to_string()),
            Message::text(Role::User, "goodbye".to_string()),
            Message::text(Role::Assistant, "bye".to_string()),
        ];
        assert_eq!(count_turns(&msgs), 3);
    }

    #[test]
    fn count_turns_partial_turn_at_end() {
        let msgs = vec![
            Message::text(Role::User, "hello".to_string()),
            Message::text(Role::Assistant, "hi".to_string()),
            Message::text(Role::User, "waiting".to_string()),
        ];
        assert_eq!(count_turns(&msgs), 2);
    }

    #[test]
    fn count_turns_orphan_assistant_at_start() {
        let msgs = vec![
            Message::text(Role::Assistant, "orphan".to_string()),
            Message::text(Role::User, "hello".to_string()),
            Message::text(Role::Assistant, "hi".to_string()),
        ];
        assert_eq!(count_turns(&msgs), 2);
    }

    #[test]
    fn count_turns_multiple_assistants_after_user() {
        let msgs = vec![
            Message::text(Role::User, "hello".to_string()),
            Message::text(Role::Assistant, "hi".to_string()),
            Message::text(Role::Assistant, "there".to_string()),
        ];
        assert_eq!(count_turns(&msgs), 1);
    }

    #[test]
    fn retained_start_index_retains_all() {
        let msgs = vec![
            Message::text(Role::User, "a".to_string()),
            Message::text(Role::Assistant, "b".to_string()),
            Message::text(Role::User, "c".to_string()),
            Message::text(Role::Assistant, "d".to_string()),
        ];
        // retain 2 turns (all) → start from 0
        assert_eq!(retained_start_index(&msgs, 2), 0);
    }

    #[test]
    fn retained_start_index_retains_one_of_two() {
        let msgs = vec![
            Message::text(Role::User, "a".to_string()),
            Message::text(Role::Assistant, "b".to_string()),
            Message::text(Role::User, "c".to_string()),
            Message::text(Role::Assistant, "d".to_string()),
        ];
        // retain 1 of 2 turns → skip first turn (indices 0-1), start from 2
        assert_eq!(retained_start_index(&msgs, 1), 2);
    }

    #[test]
    fn retained_start_index_zero() {
        let msgs = vec![
            Message::text(Role::User, "a".to_string()),
            Message::text(Role::Assistant, "b".to_string()),
        ];
        // retain 0 turns → start from 0 (compact everything)
        assert_eq!(retained_start_index(&msgs, 0), 0);
    }

    #[test]
    fn strip_non_text_preserves_text_only() {
        let msg = Message::text(Role::User, "hello".to_string());
        let result = strip_non_text(&msg);
        assert!(result.is_some());
        let stripped = result.expect("checked");
        assert_eq!(stripped.content.len(), 1);
        assert!(matches!(&stripped.content[0], ContentBlock::Text(t) if t == "hello"));
    }

    #[test]
    fn strip_non_text_removes_tool_use() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({}),
                },
                ContentBlock::Text("I ran it".to_string()),
            ],
        };
        let result = strip_non_text(&msg);
        assert!(result.is_some());
        let stripped = result.expect("checked");
        assert_eq!(stripped.content.len(), 1);
        assert!(matches!(&stripped.content[0], ContentBlock::Text(t) if t == "I ran it"));
    }

    #[test]
    fn strip_non_text_drops_empty_message() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "t1".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({}),
            }],
        };
        let result = strip_non_text(&msg);
        assert!(result.is_none());
    }

    #[test]
    fn strip_non_text_removes_thinking_and_tool_result() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    text: "deep".to_string(),
                    signature: "sig".to_string(),
                },
                ContentBlock::RedactedThinking {
                    data: "redacted".to_string(),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    content: "output".to_string(),
                    is_error: false,
                },
                ContentBlock::Text("visible".to_string()),
            ],
        };
        let result = strip_non_text(&msg);
        assert!(result.is_some());
        let stripped = result.expect("checked");
        assert_eq!(stripped.content.len(), 1);
        assert!(matches!(&stripped.content[0], ContentBlock::Text(t) if t == "visible"));
    }

    #[test]
    fn retained_start_index_with_partial_turn() {
        let msgs = vec![
            Message::text(Role::User, "a".to_string()),
            Message::text(Role::Assistant, "b".to_string()),
            Message::text(Role::User, "c".to_string()),
        ];
        // 2 turns: turn 1 = [a, b], turn 2 = [c] (partial)
        // retain 1 → skip turn 1, start from index 2
        assert_eq!(retained_start_index(&msgs, 1), 2);
    }
}
