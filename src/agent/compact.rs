use std::sync::Arc;

use crate::config::ConfirmationMode;
use crate::types::{ContentBlock, Message, Role};

use super::{Agent, lock};

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

        let compaction_role = spawner.app_config.compaction_role.clone();

        let role = if spawner.app_config.models.contains_key(&compaction_role) {
            compaction_role
        } else {
            "default".to_string()
        };

        let prompt = {
            let history = lock(&self.history);
            let prefix_len = *self
                .context_prefix_len
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let persisted = &history[prefix_len.min(history.len())..];
            let mut parts = Vec::new();
            for msg in persisted {
                let role_label = match msg.role {
                    Role::User => "User",
                    Role::Assistant => "Assistant",
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
            format!(
                "Summarize the following conversation concisely, preserving key facts, decisions, and context that would be needed to continue the conversation. Do not include meta-commentary — output only the summary.\n\n{}",
                parts.join("\n\n")
            )
        };

        let outcome = spawner
            .spawn(&role, ConfirmationMode::Never, Some(&[]), prompt)
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

        let summary_msg = Message::text(Role::User, format!("[Compacted] {summary}"));

        let session = self.session.lock().await;

        if let Err(e) = session.conversation().compact(&summary_msg).await {
            return Err(format!("Compaction failed: {e}"));
        }

        drop(session);

        let prefix_len = *self
            .context_prefix_len
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prefix: Vec<Message> = {
            let history = lock(&self.history);
            let take = prefix_len.min(history.len());
            history[..take].to_vec()
        };
        lock(&self.history).clear();
        lock(&self.history).extend(prefix);
        lock(&self.history).push(summary_msg.clone());

        Ok(summary)
    }
}
