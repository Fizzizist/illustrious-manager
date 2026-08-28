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
