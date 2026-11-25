use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use futures::channel::mpsc;
use tokio::sync::Mutex as TokioMutex;

use crate::session::Session;
use crate::types::{AgentEvent, Message};

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

/// Extracted core compaction logic so it can be called from `Agent::compact()`
/// without needing `&self`.
pub async fn compact_with(
    spawner: &Arc<AgentSpawner>,
    history: &Arc<Mutex<Vec<Message>>>,
    session: &Arc<TokioMutex<Session>>,
    context_prefix_len: &Arc<Mutex<usize>>,
) -> Result<String, String> {
    let compaction_role = spawner.app_config.compaction.role.clone();

    let role = if spawner.app_config.models.contains_key(&compaction_role) {
        compaction_role
    } else {
        "default".to_string()
    };

    let prompt = {
        let hist = lock(history);
        let prefix_len = *context_prefix_len.lock().unwrap_or_else(|e| e.into_inner());
        let persisted = &hist[prefix_len.min(hist.len())..];
        let mut parts = Vec::new();
        for msg in persisted {
            let role_label = match msg.role {
                crate::types::Role::User => "User",
                crate::types::Role::Assistant => "Assistant",
            };
            for block in &msg.content {
                match block {
                    crate::types::ContentBlock::Text(text) => {
                        parts.push(format!("{role_label}: {text}"));
                    }
                    crate::types::ContentBlock::ToolUse { name, input, .. } => {
                        parts.push(format!(
                            "{role_label}: [Called tool {name} with input {input}]"
                        ));
                    }
                    crate::types::ContentBlock::ToolResult {
                        content, is_error, ..
                    } => {
                        let label = if *is_error { "error" } else { "result" };
                        let text: String = content
                            .iter()
                            .filter_map(|b| {
                                if let crate::types::ContentBlock::Text(s) = b {
                                    Some(s.as_str())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        parts.push(format!("{role_label}: [Tool {label}: {text}]"));
                    }
                    crate::types::ContentBlock::Thinking { text, .. } => {
                        parts.push(format!("{role_label}: [Thinking: {text}]"));
                    }
                    crate::types::ContentBlock::RedactedThinking { .. } => {
                        parts.push(format!("{role_label}: [Redacted thinking]"));
                    }
                    crate::types::ContentBlock::Image { media_type, .. } => {
                        parts.push(format!("{role_label}: [Image: {media_type}]"));
                    }
                }
            }
        }
        if parts.is_empty() {
            return Ok("Nothing to compact: the conversation is empty.".to_string());
        }
        format!(
            "Summarize the following conversation concisely, preserving key facts, decisions, and context that would be needed to continue the conversation. Do not include meta-commentary — output only the summary.\n\n{}",
            parts.join("\n\n")
        )
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

    if let Err(e) = sess.conversation().compact(&summary_msg).await {
        return Err(format!("Compaction failed: {e}"));
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

    Ok(summary)
}

impl Agent {
    /// Run context compaction: summarise the active conversation history using a
    /// headless sub-agent, then replace it with the summary while preserving the
    /// context prefix (skills, CLAUDE.md, etc.).
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
    use std::collections::{BTreeMap, HashMap};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use crate::backend::BackendFactory;
    use crate::config::{
        AppConfig, CompactionConfig, ConfirmationMode, OpenAiCompatConfigToml,
        ReasoningStyleConfig, RetryConfig, ToolsConfig, VertexConfig,
    };
    use crate::timestamp::now_timestamp;
    use crate::tools::ToolRegistry;
    use crate::types::{ChatMode, ContentBlock, Role};

    async fn serve_openai_sse_once(summary: &str) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let port = listener.local_addr().expect("local addr").port();

        let body = format!(
            "data: {}\n\ndata: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":10,\"completion_tokens\":5}}}}\n\ndata: [DONE]\n\n",
            serde_json::json!({
                "choices": [{
                    "delta": { "content": summary },
                    "finish_reason": null
                }]
            })
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );

        let handle = tokio::spawn(async move {
            let (mut socket, _) =
                tokio::time::timeout(std::time::Duration::from_secs(5), listener.accept())
                    .await
                    .expect("accept should not time out")
                    .expect("accept connection");

            let mut received = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n =
                    tokio::time::timeout(std::time::Duration::from_secs(5), socket.read(&mut buf))
                        .await
                        .expect("read should not time out")
                        .expect("read request");
                if n == 0 {
                    break;
                }
                received.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&received);
                if let Some(head_end) = text.find("\r\n\r\n") {
                    let content_length = text[..head_end]
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|v| v.trim().parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    if received.len() >= head_end + 4 + content_length {
                        break;
                    }
                }
            }

            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
            let _ = socket.shutdown().await;
        });

        (port, handle)
    }

    #[tokio::test]
    async fn compact_with_persists_compacted_summary_with_current_timestamp() {
        let (port, server) = serve_openai_sse_once("Key decisions and context.").await;

        let sessions_dir = tempfile::TempDir::new().expect("temp dir");
        let mut config = AppConfig {
            backend: "openai_compat".to_string(),
            sessions_dir: sessions_dir.path().to_path_buf(),
            vertex: VertexConfig {
                project: String::new(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                max_tokens: None,
            },
            zai: None,
            ollama: None,
            openai_compat: Some(OpenAiCompatConfigToml {
                base_url: format!("http://127.0.0.1:{port}/v1"),
                api_key: None,
                model: "test-model".to_string(),
                max_tokens: None,
                reasoning: ReasoningStyleConfig::None,
            }),
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        config.normalize_back_compat();

        let spawner = Arc::new(AgentSpawner {
            factory: Arc::new(BackendFactory::new(config.clone())),
            app_config: Arc::new(config),
            registry_builder: Box::new(|_| Ok(ToolRegistry::new())),
            parent_confirmation: ConfirmationMode::Never,
            skills: HashMap::new(),
            chat_mode: ChatMode::new(false),
        });

        let session = Arc::new(TokioMutex::new(
            Session::new(None, sessions_dir.path().to_path_buf())
                .await
                .expect("test session"),
        ));
        let history = Arc::new(Mutex::new(vec![
            Message::text(Role::User, "What should we build?".to_string()),
            Message::text(Role::Assistant, "Let's build a widget.".to_string()),
        ]));
        let context_prefix_len = Arc::new(Mutex::new(0usize));

        let before = now_timestamp();
        let summary = compact_with(&spawner, &history, &session, &context_prefix_len)
            .await
            .expect("compact_with should succeed");
        let after = now_timestamp();

        server.await.expect("mock SSE server task should finish");

        assert_eq!(summary, "Key decisions and context.");

        let persisted = session
            .lock()
            .await
            .conversation()
            .load_history()
            .await
            .expect("load compacted history");
        assert_eq!(
            persisted.len(),
            1,
            "compacted session should contain only the summary message"
        );
        let summary_msg = &persisted[0];
        assert_eq!(summary_msg.role, Role::User);
        assert!(
            summary_msg.content.iter().any(|b| {
                matches!(
                    b,
                    ContentBlock::Text(t) if t == "[Compacted] Key decisions and context."
                )
            }),
            "persisted message must be the [Compacted] summary; got {:?}",
            summary_msg.content
        );
        assert!(
            summary_msg.created_at > 0.0,
            "expected positive created_at for persisted [Compacted] summary, got {}",
            summary_msg.created_at
        );
        assert!(
            before <= summary_msg.created_at && summary_msg.created_at <= after,
            "expected persisted [Compacted] created_at {} within [{}, {}]",
            summary_msg.created_at,
            before,
            after
        );
    }
}
