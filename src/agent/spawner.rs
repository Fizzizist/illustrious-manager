use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::Mutex as TokioMutex;

use crate::backend::BackendFactory;
use crate::config::{ConfirmationMode, ToolsConfig};
use crate::session::Session;
use crate::tools::ToolRegistry;
use crate::types::{AgentEvent, RequestConfig};

use super::{Agent, DEFAULT_MAX_TOKENS};

/// Outcome of a headless sub-agent run.
pub struct HeadlessOutcome {
    pub text: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub is_error: bool,
    pub error_message: Option<String>,
}

/// Drive an `Agent` to completion without a human in the loop.
///
/// Any `ToolConfirmationRequired` event causes an immediate error — sub-agents
/// must be configured with a confirmation mode that does not require human input.
pub async fn run_headless(agent: &Agent, prompt: String) -> HeadlessOutcome {
    let stream = match agent.send(prompt, None, None).await {
        Ok(s) => s,
        Err(e) => {
            return HeadlessOutcome {
                text: String::new(),
                input_tokens: 0,
                output_tokens: 0,
                is_error: true,
                error_message: Some(e.to_string()),
            };
        }
    };

    let mut stream = stream;
    let mut text = String::new();
    let mut input_tokens: u32 = 0;
    let mut output_tokens: u32 = 0;
    let mut is_error = false;
    let mut error_message: Option<String> = None;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TokenReceived(t) => text.push_str(&t),
            AgentEvent::ResponseComplete(_) => {}
            AgentEvent::ToolUseReceived { .. } => {}
            AgentEvent::ToolResult { .. } => {}
            AgentEvent::ToolConfirmationRequired { name, .. } => {
                is_error = true;
                error_message = Some(format!(
                    "Sub-agent required confirmation for tool '{name}' but no human is present. \
                     Set a less restrictive confirmation mode for the sub-agent."
                ));
                break;
            }
            AgentEvent::Error(msg) => {
                is_error = true;
                error_message = Some(msg);
                break;
            }
            AgentEvent::Usage {
                input_tokens: it,
                output_tokens: ot,
                ..
            } => {
                input_tokens = input_tokens.max(it);
                output_tokens = output_tokens.saturating_add(ot);
            }
            AgentEvent::SubAgentUsage { .. } => {}
            AgentEvent::Interrupted { .. } => {}
            AgentEvent::Warn(_) => {}
            AgentEvent::ThinkingReceived(_) => {}
            AgentEvent::CompactionComplete { .. } => {}
            AgentEvent::AutoCompactTriggered { .. } => {}
            AgentEvent::BashCommandComplete => {}
        }
    }

    HeadlessOutcome {
        text,
        input_tokens,
        output_tokens,
        is_error,
        error_message,
    }
}

/// Returns the stricter of two confirmation modes.
///
/// Strictness ordering: `Always` > `WriteOnly` > `Never`.
/// The sub-agent can never be more permissive than the parent.
pub fn clamp_confirmation(
    parent: &ConfirmationMode,
    requested: Option<&ConfirmationMode>,
) -> ConfirmationMode {
    let requested = match requested {
        Some(r) => r,
        None => return parent.clone(),
    };

    match (parent, requested) {
        (ConfirmationMode::Always, _) => ConfirmationMode::Always,
        (ConfirmationMode::WriteOnly, ConfirmationMode::Always) => ConfirmationMode::Always,
        (ConfirmationMode::WriteOnly, _) => ConfirmationMode::WriteOnly,
        (ConfirmationMode::Never, ConfirmationMode::Always) => ConfirmationMode::Always,
        (ConfirmationMode::Never, ConfirmationMode::WriteOnly) => ConfirmationMode::WriteOnly,
        (ConfirmationMode::Never, ConfirmationMode::Never) => ConfirmationMode::Never,
    }
}

/// Builds a fresh `ToolRegistry` for a sub-agent.
///
/// The closure receives the sub-agent's session so that session-bound tools
/// (e.g. task tools) are wired to the sub-agent rather than the parent.
pub type RegistryBuilder =
    Box<dyn Fn(Arc<tokio::sync::Mutex<Session>>) -> anyhow::Result<ToolRegistry> + Send + Sync>;

/// Factory used by `AgentTool` to spawn independent sub-agents.
pub struct AgentSpawner {
    pub factory: Arc<BackendFactory>,
    pub app_config: Arc<crate::config::AppConfig>,
    pub registry_builder: RegistryBuilder,
    pub parent_confirmation: ConfirmationMode,
    pub skills: std::collections::HashMap<String, std::path::PathBuf>,
    pub chat_mode: crate::types::ChatMode,
}

impl AgentSpawner {
    pub async fn spawn(
        &self,
        role: &str,
        confirmation: ConfirmationMode,
        tool_allowlist: Option<&[String]>,
        prompt: String,
    ) -> HeadlessOutcome {
        let session = match Session::new(None, self.app_config.sessions_dir.clone()).await {
            Ok(s) => Arc::new(tokio::sync::Mutex::new(s)),
            Err(e) => {
                return HeadlessOutcome {
                    text: String::new(),
                    input_tokens: 0,
                    output_tokens: 0,
                    is_error: true,
                    error_message: Some(format!("Failed to create sub-agent session: {e}")),
                };
            }
        };

        let registry = match (self.registry_builder)(Arc::clone(&session)) {
            Ok(r) => r,
            Err(e) => {
                return HeadlessOutcome {
                    text: String::new(),
                    input_tokens: 0,
                    output_tokens: 0,
                    is_error: true,
                    error_message: Some(format!("Failed to build sub-agent registry: {e}")),
                };
            }
        };

        let registry = if self.chat_mode.is_on() {
            registry.into_chat_compatible()
        } else if let Some(allowlist) = tool_allowlist {
            registry.into_filtered(allowlist)
        } else {
            registry
        };

        let tool_config = ToolsConfig {
            confirmation,
            ..self.app_config.tools.clone()
        };

        let selection = match self.factory.for_role(role).await {
            Ok(s) => s,
            Err(e) => {
                return HeadlessOutcome {
                    text: String::new(),
                    input_tokens: 0,
                    output_tokens: 0,
                    is_error: true,
                    error_message: Some(format!("Failed to resolve role '{role}': {e}")),
                };
            }
        };

        let agent =
            match spawn_agent_with_selection(selection, &tool_config, session, registry).await {
                Ok(a) => a.with_thinking(self.app_config.thinking.clone()),
                Err(e) => {
                    return HeadlessOutcome {
                        text: String::new(),
                        input_tokens: 0,
                        output_tokens: 0,
                        is_error: true,
                        error_message: Some(format!("Failed to spawn sub-agent: {e}")),
                    };
                }
            };

        let agent = match agent.with_context_files() {
            Ok(a) => a,
            Err(e) => {
                return HeadlessOutcome {
                    text: String::new(),
                    input_tokens: 0,
                    output_tokens: 0,
                    is_error: true,
                    error_message: Some(format!("Failed to load context files: {e}")),
                };
            }
        };

        let agent = agent.with_skills(&self.skills);
        run_headless(&agent, prompt).await
    }
}

/// Spawn a fresh `Agent` for the given named role using the shared `BackendFactory`.
///
/// The caller supplies a pre-built `ToolRegistry` and a `Session`. The agent's
/// model is taken from the role definition; `tool_config` controls iteration
/// limits and confirmation behaviour.
pub async fn spawn_agent(
    factory: &BackendFactory,
    role: &str,
    tool_config: &ToolsConfig,
    session: Arc<TokioMutex<Session>>,
    tools: ToolRegistry,
) -> anyhow::Result<Agent> {
    let selection = factory.for_role(role).await?;
    spawn_agent_with_selection(selection, tool_config, session, tools).await
}

/// Core of `spawn_agent` — constructs an `Agent` from an already-resolved
/// `BackendSelection`. Separated out so tests can inject a fake backend without
/// going through real auth.
pub async fn spawn_agent_with_selection(
    selection: crate::backend::BackendSelection,
    tool_config: &ToolsConfig,
    session: Arc<TokioMutex<Session>>,
    tools: ToolRegistry,
) -> anyhow::Result<Agent> {
    let request_config = RequestConfig {
        model: selection.model,
        max_tokens: DEFAULT_MAX_TOKENS,
        tools: tools.definitions(),
        thinking: None,
        cancel_token: None,
    };
    Ok(Agent::new(selection.backend, request_config, session)
        .await
        .with_tools(tools)
        .with_tool_config(tool_config)
        .with_compaction_config(&crate::config::CompactionConfig::default())
        .with_retry_config(&crate::config::RetryConfig::default()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMode;

    #[test]
    fn spawner_chat_mode_filters_write_tools_from_registry() {
        let chat_mode = ChatMode::new(true);
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(crate::tools::bash::BashTool::new(
                vec![],
                vec![],
                std::path::PathBuf::from("/tmp"),
                crate::config::ConfirmationMode::Never,
                Box::new(|_| true),
                chat_mode.clone(),
            )))
            .expect("register bash");
        registry
            .register(Box::new(crate::tools::edit_file::EditFile::new(
                crate::tools::sandbox::SandboxPolicy::new(std::path::Path::new("/tmp")),
            )))
            .expect("register edit_file");
        registry
            .register(Box::new(crate::tools::write_file::WriteFileTool::new(
                crate::tools::sandbox::SandboxPolicy::new(std::path::Path::new("/tmp")),
            )))
            .expect("register write_file");
        registry
            .register(Box::new(crate::tools::search::SearchTool::new(
                std::path::PathBuf::from("/tmp"),
            )))
            .expect("register search");
        registry
            .register(Box::new(crate::tools::skill::SkillTool::new(
                &std::collections::HashMap::new(),
            )))
            .expect("register skill");

        let all_names = registry.tool_names();
        assert!(all_names.contains(&"bash".to_string()));
        assert!(all_names.contains(&"edit_file".to_string()));
        assert!(all_names.contains(&"write_file".to_string()));
        assert!(all_names.contains(&"search".to_string()));

        let registry = registry.into_chat_compatible();

        let remaining = registry.tool_names();
        assert!(
            remaining.contains(&"bash".to_string()),
            "bash should remain available in chat mode (as restricted read-only)"
        );
        assert!(
            !remaining.contains(&"edit_file".to_string()),
            "edit_file should be filtered out in chat mode"
        );
        assert!(
            !remaining.contains(&"write_file".to_string()),
            "write_file should be filtered out in chat mode"
        );
        assert!(
            remaining.contains(&"search".to_string()),
            "search should remain in chat mode"
        );
        assert!(
            remaining.contains(&"skill".to_string()),
            "skill should remain in chat mode"
        );
        assert!(chat_mode.is_on());
    }

    #[test]
    fn spawner_chat_mode_off_keeps_all_tools() {
        let chat_mode = ChatMode::new(false);
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(crate::tools::bash::BashTool::new(
                vec![],
                vec![],
                std::path::PathBuf::from("/tmp"),
                crate::config::ConfirmationMode::Never,
                Box::new(|_| true),
                chat_mode.clone(),
            )))
            .expect("register bash");

        assert!(!chat_mode.is_on());
        let names = registry.tool_names();
        assert!(
            names.contains(&"bash".to_string()),
            "bash should be present when chat mode is off"
        );
    }
}
