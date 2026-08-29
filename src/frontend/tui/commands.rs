use crate::agent::Agent;
use crate::backend::BackendFactory;
use crate::config::{AppConfig, generate_config_message};
use crate::frontend::tui::tasks_picker::{TasksPicker, sort_tasks};
use crate::frontend::tui::tui_app::{App, AppState};
use crate::frontend::tui::{ConversationEntry, ConversationRole, SessionPicker};
use crate::session::{enumerate_sessions, hydrate_next_page};
use crate::types::AgentEvent;
use std::sync::Arc;
use tokio::sync::mpsc;

pub const SESSION_PAGE_SIZE: usize = 100;

/// Result of dispatching a slash command.
#[derive(Debug, PartialEq, Eq)]
pub enum DispatchResult {
    /// The command was recognised and handled.
    Handled,
    /// The input was not a known command; treat it as a normal user message.
    Passthrough,
}

/// A parsed slash command with its name and trailing arguments.
pub struct ParsedCommand<'a> {
    pub name: &'a str,
    pub args: &'a str,
}

/// Parse a user input string into a `ParsedCommand`, or return `None` if the
/// input does not start with `/`.
pub fn parse_command(input: &str) -> Option<ParsedCommand<'_>> {
    let trimmed = input.trim();
    let after_slash = trimmed.strip_prefix('/')?;
    let (name, args) = after_slash.split_once(' ').unwrap_or((after_slash, ""));
    Some(ParsedCommand {
        name,
        args: args.trim(),
    })
}

/// Context passed to every slash-command handler.
pub struct CommandContext<'a> {
    pub app: &'a mut App,
    pub agent: Arc<Agent>,
    pub config: &'a AppConfig,
    pub event_tx: &'a mpsc::Sender<AgentEvent>,
    pub backend_factory: Arc<BackendFactory>,
}

/// Trait implemented by every registered slash command.
pub trait SlashCommand: Send + Sync {
    fn name(&self) -> &str;
    /// Execute the command. Returns `Handled` on success or `Passthrough` if
    /// the command decides to delegate to the LLM.
    fn execute<'a>(
        &self,
        args: &str,
        ctx: &'a mut CommandContext<'_>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<DispatchResult>> + 'a>>;
}

/// Registry of named slash commands.
#[derive(Default)]
pub struct CommandRegistry {
    commands: Vec<Box<dyn SlashCommand>>,
}

impl CommandRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, cmd: Box<dyn SlashCommand>) {
        self.commands.push(cmd);
    }

    /// Dispatch a user input string. Returns `Passthrough` for non-slash input
    /// and for unrecognised slash commands, so the caller can forward to the LLM.
    pub async fn dispatch(
        &self,
        input: &str,
        ctx: &mut CommandContext<'_>,
    ) -> anyhow::Result<DispatchResult> {
        let parsed = match parse_command(input) {
            Some(p) => p,
            None => return Ok(DispatchResult::Passthrough),
        };
        for cmd in &self.commands {
            if cmd.name() == parsed.name {
                return cmd.execute(parsed.args, ctx).await;
            }
        }
        // Unknown slash command — pass through to the LLM.
        Ok(DispatchResult::Passthrough)
    }
}

/// Built-in `/sessions` command — opens the session picker overlay.
pub struct SessionsCommand;

impl SlashCommand for SessionsCommand {
    fn name(&self) -> &str {
        "sessions"
    }

    fn execute<'a>(
        &self,
        _args: &str,
        ctx: &'a mut CommandContext<'_>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<DispatchResult>> + 'a>>
    {
        Box::pin(async move {
            ctx.app.input.clear();
            match enumerate_sessions(&ctx.config.sessions_dir).await {
                Ok(mut refs) => {
                    let (page, has_more) = hydrate_next_page(&mut refs, SESSION_PAGE_SIZE).await;
                    ctx.app.pending_session_refs = refs;
                    ctx.app.session_picker = Some(SessionPicker::new(page, has_more));
                    ctx.app.set_state(AppState::SessionPicker);
                }
                Err(e) => {
                    ctx.app.conversation.push(ConversationEntry::new(
                        ConversationRole::Error,
                        format!("Failed to list sessions: {e}"),
                        crate::timestamp::format_now_timestamp(),
                    ));
                }
            }
            Ok(DispatchResult::Handled)
        })
    }
}

/// Built-in `/model <name>` command — switches the active model.
pub struct ModelCommand;

impl SlashCommand for ModelCommand {
    fn name(&self) -> &str {
        "model"
    }

    fn execute<'a>(
        &self,
        args: &str,
        ctx: &'a mut CommandContext<'_>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<DispatchResult>> + 'a>>
    {
        let model = args.to_string();
        Box::pin(async move {
            ctx.app.input.clear();
            if model.is_empty() {
                ctx.app.conversation.push(ConversationEntry::new(
                    ConversationRole::Error,
                    "Usage: /model <model-name>".to_string(),
                    crate::timestamp::format_now_timestamp(),
                ));
            } else {
                ctx.agent.set_model(model.clone());
                ctx.app.model = model.clone();
                ctx.app.conversation.push(ConversationEntry::new(
                    ConversationRole::Info,
                    format!("Model switched to `{model}`"),
                    crate::timestamp::format_now_timestamp(),
                ));
            }
            Ok(DispatchResult::Handled)
        })
    }
}

/// Built-in `/tasks` command — opens the tasks picker overlay.
pub struct TasksCommand;

/// Built-in `/compact` command — triggers context compaction.
pub struct CompactCommand;

impl SlashCommand for CompactCommand {
    fn name(&self) -> &str {
        "compact"
    }

    fn execute<'a>(
        &self,
        _args: &str,
        ctx: &'a mut CommandContext<'_>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<DispatchResult>> + 'a>>
    {
        let event_tx = ctx.event_tx.clone();
        Box::pin(async move {
            ctx.app.input.clear();
            ctx.app.set_state(AppState::Compacting);
            let agent = Arc::clone(&ctx.agent);
            let handle = tokio::spawn(async move {
                let result = agent.compact().await;
                let (summary, is_error) = match result {
                    Ok(s) => (s, false),
                    Err(e) => (e, true),
                };
                let _ = event_tx
                    .send(AgentEvent::CompactionComplete { summary, is_error })
                    .await;
            });
            ctx.app.compaction_task = Some(handle);
            Ok(DispatchResult::Handled)
        })
    }
}

impl SlashCommand for TasksCommand {
    fn name(&self) -> &str {
        "tasks"
    }

    fn execute<'a>(
        &self,
        _args: &str,
        ctx: &'a mut CommandContext<'_>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<DispatchResult>> + 'a>>
    {
        Box::pin(async move {
            ctx.app.input.clear();
            match ctx.agent.tasks_snapshot().await {
                Ok(mut tasks) => {
                    sort_tasks(&mut tasks);
                    ctx.app.tasks_picker = Some(TasksPicker::new(tasks));
                    ctx.app.set_state(AppState::TasksPicker);
                }
                Err(e) => {
                    ctx.app.conversation.push(ConversationEntry::new(
                        ConversationRole::Error,
                        format!("Failed to load tasks: {e}"),
                        crate::timestamp::format_now_timestamp(),
                    ));
                }
            }
            Ok(DispatchResult::Handled)
        })
    }
}

/// Built-in `/new` command — starts a fresh session.
pub struct NewCommand;

impl SlashCommand for NewCommand {
    fn name(&self) -> &str {
        "new"
    }

    fn execute<'a>(
        &self,
        _args: &str,
        ctx: &'a mut CommandContext<'_>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<DispatchResult>> + 'a>>
    {
        Box::pin(async move {
            ctx.app.input.clear();
            // Checkpoint current session WAL
            if let Err(e) = ctx.agent.checkpoint_session().await {
                ctx.app.conversation.push(ConversationEntry::new(
                    ConversationRole::Error,
                    format!("Failed to checkpoint session: {e}"),
                    crate::timestamp::format_now_timestamp(),
                ));
            }
            // Clean up current session if empty
            if let Err(e) = ctx.agent.cleanup_empty_session().await {
                ctx.app.conversation.push(ConversationEntry::new(
                    ConversationRole::Error,
                    format!("Failed to clean up empty session: {e}"),
                    crate::timestamp::format_now_timestamp(),
                ));
            }
            // Create a new session
            let new_session = crate::session::Session::new(None, ctx.config.sessions_dir.clone())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create new session: {e}"))?;
            let new_history = new_session
                .conversation()
                .load_history()
                .await
                .unwrap_or_default();
            // Reload context files and skills into the new session
            ctx.agent.load_session(new_session).await;
            // Preserve the current model
            let model = ctx.agent.model();
            ctx.app.conversation.clear();
            ctx.app.current_response.clear();
            ctx.app.current_thinking.clear();
            ctx.app.scroll_offset = 0;
            ctx.app.reset_for_session_switch();
            ctx.app.load_history(&new_history);
            ctx.app.model = model;
            ctx.app.conversation.push(ConversationEntry::new(
                ConversationRole::Info,
                "Started new session.".to_string(),
                crate::timestamp::format_now_timestamp(),
            ));
            Ok(DispatchResult::Handled)
        })
    }
}

/// Built-in `/role <name>` command — switches backend and model by named role.
/// With no arguments, displays current backend/model configuration.
pub struct RoleCommand;

impl SlashCommand for RoleCommand {
    fn name(&self) -> &str {
        "role"
    }

    fn execute<'a>(
        &self,
        args: &str,
        ctx: &'a mut CommandContext<'_>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<DispatchResult>> + 'a>>
    {
        let role_name = args.to_string();
        let factory = Arc::clone(&ctx.backend_factory);
        Box::pin(async move {
            ctx.app.input.clear();
            if role_name.is_empty() {
                let model = ctx.agent.model();
                ctx.app.conversation.push(ConversationEntry::new(
                    ConversationRole::Info,
                    format!("Current model: {model}"),
                    crate::timestamp::format_now_timestamp(),
                ));
                return Ok(DispatchResult::Handled);
            }
            match factory.for_role(&role_name).await {
                Ok(selection) => {
                    let new_model = selection.model.clone();
                    let max_tokens = selection.max_tokens;
                    let backend: Arc<dyn crate::backend::LlmBackend> = Arc::from(selection.backend);
                    ctx.agent
                        .set_backend(backend, new_model.clone(), max_tokens);
                    ctx.app.model = new_model.clone();
                    ctx.app.conversation.push(ConversationEntry::new(
                        ConversationRole::Info,
                        format!("Switched to role '{role_name}' (model: {new_model})"),
                        crate::timestamp::format_now_timestamp(),
                    ));
                }
                Err(e) => {
                    ctx.app.conversation.push(ConversationEntry::new(
                        ConversationRole::Error,
                        format!("Unknown role '{role_name}': {e}"),
                        crate::timestamp::format_now_timestamp(),
                    ));
                }
            }
            Ok(DispatchResult::Handled)
        })
    }
}

/// Drop guard that ensures `BashCommandComplete` is always emitted from the
/// spawned bash task, even if the task body panics.
struct BashStateGuard {
    event_tx: tokio::sync::mpsc::Sender<AgentEvent>,
    disarmed: bool,
}

impl BashStateGuard {
    fn new(event_tx: tokio::sync::mpsc::Sender<AgentEvent>) -> Self {
        Self {
            event_tx,
            disarmed: false,
        }
    }

    fn disarm(&mut self) {
        self.disarmed = true;
    }
}

impl Drop for BashStateGuard {
    fn drop(&mut self) {
        if !self.disarmed {
            let _ = self.event_tx.try_send(AgentEvent::BashCommandComplete);
        }
    }
}

/// Built-in `/bash <command>` command — executes a shell command directly,
/// records the tool call and result in conversation history, and renders it
/// exactly like an LLM-driven bash call. Bypasses all policy checks.
pub struct BashCommand;

impl SlashCommand for BashCommand {
    fn name(&self) -> &str {
        "bash"
    }

    fn execute<'a>(
        &self,
        args: &str,
        ctx: &'a mut CommandContext<'_>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<DispatchResult>> + 'a>>
    {
        let command = args.to_string();
        let sandbox_root = std::path::PathBuf::from(&ctx.config.tools.sandbox_root);
        Box::pin(async move {
            ctx.app.input.clear();
            if command.is_empty() {
                ctx.app.conversation.push(ConversationEntry::new(
                    ConversationRole::Info,
                    "Usage: /bash <command>".to_string(),
                    crate::timestamp::format_now_timestamp(),
                ));
                return Ok(DispatchResult::Handled);
            }

            let tool_use_id = format!("user-bash-{}", uuid::Uuid::now_v7());
            let index = 1usize;

            ctx.app.set_state(AppState::RunningBash);
            let cancel = tokio_util::sync::CancellationToken::new();
            ctx.app.cancel_token = Some(cancel.clone());

            let input_json = serde_json::json!({ "command": command });
            let _ = ctx
                .event_tx
                .send(AgentEvent::ToolUseReceived {
                    id: tool_use_id.clone(),
                    name: "bash".to_string(),
                    input: input_json,
                    index,
                })
                .await;

            let agent = Arc::clone(&ctx.agent);
            let event_tx = ctx.event_tx.clone();
            let cmd_clone = command.clone();
            let id_clone = tool_use_id.clone();

            tokio::spawn(async move {
                let mut guard = BashStateGuard::new(event_tx.clone());

                let (raw_content, is_error) =
                    crate::tools::bash::execute_raw(&cmd_clone, &sandbox_root, Some(cancel)).await;

                if let Err(e) = agent
                    .record_synthetic_tool_call(
                        id_clone.clone(),
                        cmd_clone,
                        raw_content.clone(),
                        is_error,
                    )
                    .await
                {
                    let _ = event_tx
                        .send(AgentEvent::Error(format!(
                            "Failed to record /bash result: {e}"
                        )))
                        .await;
                    return;
                }

                let _ = event_tx
                    .send(AgentEvent::ToolResult {
                        name: "bash".to_string(),
                        content: raw_content,
                        is_error,
                        index,
                    })
                    .await;

                guard.disarm();
                let _ = event_tx.send(AgentEvent::BashCommandComplete).await;
            });

            Ok(DispatchResult::Handled)
        })
    }
}

/// Built-in `/chat` command — toggles chat mode on/off.
pub struct ChatCommand;

impl SlashCommand for ChatCommand {
    fn name(&self) -> &str {
        "chat"
    }

    fn execute<'a>(
        &self,
        _args: &str,
        ctx: &'a mut CommandContext<'_>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<DispatchResult>> + 'a>>
    {
        Box::pin(async move {
            ctx.app.input.clear();
            let new_state = ctx.app.chat_mode.toggle();
            ctx.agent.set_chat_mode(new_state);
            if new_state {
                ctx.app.conversation.push(ConversationEntry::new(
                    ConversationRole::Info,
                    "Chat mode ON — write tools hidden, bash restricted to read-only commands"
                        .to_string(),
                    crate::timestamp::format_now_timestamp(),
                ));
            } else {
                ctx.app.conversation.push(ConversationEntry::new(
                    ConversationRole::Info,
                    "Chat mode OFF — all tools available".to_string(),
                    crate::timestamp::format_now_timestamp(),
                ));
            }
            Ok(DispatchResult::Handled)
        })
    }
}

pub struct ConfigCommand;

impl SlashCommand for ConfigCommand {
    fn name(&self) -> &str {
        "config"
    }

    fn execute<'a>(
        &self,
        _args: &str,
        ctx: &'a mut CommandContext<'_>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<DispatchResult>> + 'a>>
    {
        Box::pin(async move {
            ctx.app.input.clear();
            ctx.app.conversation.push(ConversationEntry::new(
                ConversationRole::Info,
                generate_config_message(ctx.config),
                crate::timestamp::format_now_timestamp(),
            ));
            Ok(DispatchResult::Handled)
        })
    }
}

/// Build the default `CommandRegistry` with all built-in commands registered.
pub fn default_registry() -> CommandRegistry {
    let mut registry = CommandRegistry::new();
    registry.register(Box::new(SessionsCommand));
    registry.register(Box::new(ModelCommand));
    registry.register(Box::new(TasksCommand));
    registry.register(Box::new(CompactCommand));
    registry.register(Box::new(NewCommand));
    registry.register(Box::new(RoleCommand));
    registry.register(Box::new(ChatCommand));
    registry.register(Box::new(BashCommand));
    registry.register(Box::new(ConfigCommand));
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CompactionConfig;
    use crate::config::{AppConfig, RetryConfig, ToolsConfig, VertexConfig};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn make_config() -> AppConfig {
        AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        }
    }

    fn make_factory() -> Arc<BackendFactory> {
        Arc::new(BackendFactory::new(make_config()))
    }

    #[test]
    fn parse_command_returns_name_and_args_for_slash_input() {
        let parsed = parse_command("/model claude-haiku").expect("should parse");
        assert_eq!(parsed.name, "model");
        assert_eq!(parsed.args, "claude-haiku");
    }

    #[test]
    fn parse_command_returns_none_for_non_slash_input() {
        assert!(parse_command("hello world").is_none());
        assert!(parse_command("").is_none());
        assert!(parse_command("model claude").is_none());
    }

    #[test]
    fn parse_command_handles_no_args() {
        let parsed = parse_command("/sessions").expect("should parse");
        assert_eq!(parsed.name, "sessions");
        assert_eq!(parsed.args, "");
    }

    #[test]
    fn parse_command_trims_leading_whitespace() {
        let parsed = parse_command("  /sessions  ").expect("should parse");
        assert_eq!(parsed.name, "sessions");
    }

    #[test]
    fn command_registry_returns_passthrough_for_unknown_slash_input() {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let registry = default_registry();
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));

        rt.block_on(async {
            let dir = tempfile::TempDir::new().expect("temp dir");
            let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
                .await
                .expect("session");
            let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
            let agent = Arc::new(
                crate::agent::Agent::new(
                    Box::new(FakeBackend),
                    crate::types::RequestConfig {
                        model: "test".to_string(),
                        max_tokens: 1024,
                        tools: vec![],
                        thinking: None,
                        cancel_token: None,
                    },
                    session,
                )
                .await,
            );
            let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
            let mut ctx = CommandContext {
                app: &mut app,
                agent,
                config: &config,
                event_tx: &event_tx,
                backend_factory: make_factory(),
            };
            let result = registry
                .dispatch("/code-review 42", &mut ctx)
                .await
                .expect("dispatch");
            assert_eq!(result, DispatchResult::Passthrough);
        });
    }

    #[test]
    fn command_registry_returns_passthrough_for_non_slash_input() {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let registry = default_registry();
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));

        rt.block_on(async {
            let dir = tempfile::TempDir::new().expect("temp dir");
            let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
                .await
                .expect("session");
            let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
            let agent = Arc::new(
                crate::agent::Agent::new(
                    Box::new(FakeBackend),
                    crate::types::RequestConfig {
                        model: "test".to_string(),
                        max_tokens: 1024,
                        tools: vec![],
                        thinking: None,
                        cancel_token: None,
                    },
                    session,
                )
                .await,
            );
            let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
            let mut ctx = CommandContext {
                app: &mut app,
                agent,
                config: &config,
                event_tx: &event_tx,
                backend_factory: make_factory(),
            };
            let result = registry
                .dispatch("hello there", &mut ctx)
                .await
                .expect("dispatch");
            assert_eq!(result, DispatchResult::Passthrough);
        });
    }

    #[test]
    fn sessions_command_opens_session_picker() {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));

        rt.block_on(async {
            let dir = tempfile::TempDir::new().expect("temp dir");
            let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
                .await
                .expect("session");
            let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
            let agent = Arc::new(
                crate::agent::Agent::new(
                    Box::new(FakeBackend),
                    crate::types::RequestConfig {
                        model: "test".to_string(),
                        max_tokens: 1024,
                        tools: vec![],
                        thinking: None,
                        cancel_token: None,
                    },
                    session,
                )
                .await,
            );
            let cmd = SessionsCommand;
            let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
            let mut ctx = CommandContext {
                app: &mut app,
                agent,
                config: &config,
                event_tx: &event_tx,
                backend_factory: make_factory(),
            };
            let result = cmd.execute("", &mut ctx).await.expect("execute");
            assert_eq!(result, DispatchResult::Handled);
            assert_eq!(ctx.app.state, AppState::SessionPicker);
            assert!(ctx.app.session_picker.is_some());
        });
    }

    #[test]
    fn model_command_with_name_updates_agent_and_app_model() {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));

        rt.block_on(async {
            let dir = tempfile::TempDir::new().expect("temp dir");
            let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
                .await
                .expect("session");
            let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
            let agent = Arc::new(
                crate::agent::Agent::new(
                    Box::new(FakeBackend),
                    crate::types::RequestConfig {
                        model: "claude-original".to_string(),
                        max_tokens: 1024,
                        tools: vec![],
                        thinking: None,
                        cancel_token: None,
                    },
                    session,
                )
                .await,
            );
            let cmd = ModelCommand;
            let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
            let mut ctx = CommandContext {
                app: &mut app,
                agent: Arc::clone(&agent),
                config: &config,
                event_tx: &event_tx,
                backend_factory: make_factory(),
            };
            let result = cmd.execute("claude-new", &mut ctx).await.expect("execute");
            assert_eq!(result, DispatchResult::Handled);
            assert_eq!(agent.model(), "claude-new");
            assert_eq!(ctx.app.model, "claude-new");
        });
    }

    #[test]
    fn model_command_without_name_pushes_usage_error_to_conversation() {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));

        rt.block_on(async {
            let dir = tempfile::TempDir::new().expect("temp dir");
            let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
                .await
                .expect("session");
            let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
            let agent = Arc::new(
                crate::agent::Agent::new(
                    Box::new(FakeBackend),
                    crate::types::RequestConfig {
                        model: "claude-original".to_string(),
                        max_tokens: 1024,
                        tools: vec![],
                        thinking: None,
                        cancel_token: None,
                    },
                    session,
                )
                .await,
            );
            let cmd = ModelCommand;
            let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
            let mut ctx = CommandContext {
                app: &mut app,
                agent: Arc::clone(&agent),
                config: &config,
                event_tx: &event_tx,
                backend_factory: make_factory(),
            };
            let result = cmd.execute("", &mut ctx).await.expect("execute");
            assert_eq!(result, DispatchResult::Handled);
            // model should be unchanged
            assert_eq!(agent.model(), "claude-original");
            // an error entry should have been pushed
            assert!(
                ctx.app
                    .conversation
                    .iter()
                    .any(|e| e.role == ConversationRole::Error
                        && e.content.contains("Usage: /model")),
                "should have usage error in conversation"
            );
        });
    }

    struct FakeBackend;

    #[async_trait::async_trait]
    impl crate::backend::LlmBackend for FakeBackend {
        async fn send_message(
            &self,
            _: &[crate::types::Message],
            _: &crate::types::RequestConfig,
        ) -> anyhow::Result<crate::types::BoxStream<anyhow::Result<crate::types::StreamEvent>>>
        {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[test]
    fn tasks_command_opens_picker_and_sets_state() {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));

        rt.block_on(async {
            let dir = tempfile::TempDir::new().expect("temp dir");
            let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
                .await
                .expect("session");
            let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
            let agent = Arc::new(
                crate::agent::Agent::new(
                    Box::new(FakeBackend),
                    crate::types::RequestConfig {
                        model: "test".to_string(),
                        max_tokens: 1024,
                        tools: vec![],
                        thinking: None,
                        cancel_token: None,
                    },
                    session,
                )
                .await,
            );
            let cmd = TasksCommand;
            let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
            let mut ctx = CommandContext {
                app: &mut app,
                agent,
                config: &config,
                event_tx: &event_tx,
                backend_factory: make_factory(),
            };
            let result = cmd.execute("", &mut ctx).await.expect("execute");
            assert_eq!(result, DispatchResult::Handled);
            assert_eq!(ctx.app.state, AppState::TasksPicker);
            assert!(ctx.app.tasks_picker.is_some());
        });
    }

    #[tokio::test]
    async fn tasks_command_pulls_current_session_tasks() {
        use crate::session::Session;

        let dir = tempfile::TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");

        // Pre-populate two tasks
        session
            .tasks()
            .create("task alpha", None)
            .await
            .expect("create alpha");
        session
            .tasks()
            .create("task beta", None)
            .await
            .expect("create beta");

        let session_arc = std::sync::Arc::new(tokio::sync::Mutex::new(session));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "test".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                    thinking: None,
                    cancel_token: None,
                },
                session_arc,
            )
            .await,
        );

        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        let cmd = TasksCommand;
        let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
            event_tx: &event_tx,
            backend_factory: make_factory(),
        };
        let result = cmd.execute("", &mut ctx).await.expect("execute");
        assert_eq!(result, DispatchResult::Handled);
        let picker = ctx.app.tasks_picker.as_ref().expect("picker present");
        assert_eq!(picker.tasks().len(), 2);
        // both pending with the same status, ordered by stable sort then DB insertion order (alpha inserted first)
        assert_eq!(picker.tasks()[0].title, "task alpha");
        assert_eq!(picker.tasks()[1].title, "task beta");
    }

    #[tokio::test]
    async fn compact_command_transitions_to_compacting_state() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "test".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                    thinking: None,
                    cancel_token: None,
                },
                session,
            )
            .await,
        );

        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        let cmd = CompactCommand;
        let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
            event_tx: &event_tx,
            backend_factory: make_factory(),
        };
        let result = cmd.execute("", &mut ctx).await.expect("execute");
        assert_eq!(result, DispatchResult::Handled);
        assert_eq!(ctx.app.state, AppState::Compacting);
    }

    #[test]
    fn parse_command_recognizes_compact() {
        let parsed = parse_command("/compact").expect("should parse");
        assert_eq!(parsed.name, "compact");
        assert_eq!(parsed.args, "");
    }

    #[test]
    fn parse_command_compact_with_args() {
        let parsed = parse_command("/compact ").expect("should parse");
        assert_eq!(parsed.name, "compact");
    }

    #[test]
    fn parse_command_recognizes_new() {
        let parsed = parse_command("/new").expect("should parse");
        assert_eq!(parsed.name, "new");
        assert_eq!(parsed.args, "");
    }

    #[test]
    fn parse_command_recognizes_role() {
        let parsed = parse_command("/role fast").expect("should parse");
        assert_eq!(parsed.name, "role");
        assert_eq!(parsed.args, "fast");
    }

    #[test]
    fn parse_command_role_with_no_args() {
        let parsed = parse_command("/role").expect("should parse");
        assert_eq!(parsed.name, "role");
        assert_eq!(parsed.args, "");
    }

    #[tokio::test]
    async fn role_command_with_no_args_shows_current_model() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "claude-sonnet-4-20250514".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                    thinking: None,
                    cancel_token: None,
                },
                session,
            )
            .await,
        );
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        let cmd = RoleCommand;
        let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
            event_tx: &event_tx,
            backend_factory: make_factory(),
        };
        let result = cmd.execute("", &mut ctx).await.expect("execute");
        assert_eq!(result, DispatchResult::Handled);
        assert!(
            ctx.app
                .conversation
                .iter()
                .any(|e| e.role == ConversationRole::Info
                    && e.content.contains("claude-sonnet-4-20250514")),
            "should show current model in info message"
        );
    }

    #[tokio::test]
    async fn role_command_with_unknown_role_shows_error() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "test".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                    thinking: None,
                    cancel_token: None,
                },
                session,
            )
            .await,
        );
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        let cmd = RoleCommand;
        let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
            event_tx: &event_tx,
            backend_factory: make_factory(),
        };
        let result = cmd
            .execute("nonexistent_role", &mut ctx)
            .await
            .expect("execute");
        assert_eq!(result, DispatchResult::Handled);
        assert!(
            ctx.app.conversation.iter().any(
                |e| e.role == ConversationRole::Error && e.content.contains("nonexistent_role")
            ),
            "should show error for unknown role"
        );
    }

    #[tokio::test]
    async fn new_command_creates_new_session_and_clears_conversation() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "test-model".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                    thinking: None,
                    cancel_token: None,
                },
                session,
            )
            .await,
        );
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        // Pre-populate conversation
        app.conversation.push(ConversationEntry::new(
            ConversationRole::User,
            "old message".to_string(),
            String::new(),
        ));
        app.model = "test-model".to_string();
        // Pre-populate App state that should be reset
        app.current_response = "partial response".to_string();
        app.current_thinking = "partial thinking".to_string();
        app.scroll_offset = 42;
        app.usage = crate::frontend::tui::TokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            is_estimated: false,
        };
        // Capture original session ID
        let original_session_id = agent.session_id().await;
        let cmd = NewCommand;
        let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
            event_tx: &event_tx,
            backend_factory: make_factory(),
        };
        let result = cmd.execute("", &mut ctx).await.expect("execute");
        assert_eq!(result, DispatchResult::Handled);
        // Model should be preserved
        assert_eq!(ctx.app.model, "test-model");
        // Should have info message about new session
        assert!(
            ctx.app
                .conversation
                .iter()
                .any(|e| e.role == ConversationRole::Info && e.content.contains("new session")),
            "should have info message about new session"
        );
        // Old message should be gone
        assert!(
            !ctx.app
                .conversation
                .iter()
                .any(|e| e.content == "old message"),
            "old conversation should be cleared"
        );
        // Session ID should have changed
        let new_session_id = ctx.agent.session_id().await;
        assert_ne!(
            original_session_id, new_session_id,
            "session ID should change after /new"
        );
        // Agent history should reflect the new empty session
        let history = ctx.agent.history();
        assert!(
            history.is_empty(),
            "new session should have empty history, got {} messages",
            history.len()
        );
        // Full App state should be reset
        assert!(
            ctx.app.current_response.is_empty(),
            "current_response should be cleared after /new"
        );
        assert!(
            ctx.app.current_thinking.is_empty(),
            "current_thinking should be cleared after /new"
        );
        assert_eq!(
            ctx.app.scroll_offset, 0,
            "scroll_offset should be reset after /new"
        );
        assert_eq!(
            ctx.app.usage.input_tokens, 0,
            "usage should be reset after /new"
        );
        assert_eq!(
            ctx.app.usage.output_tokens, 0,
            "usage should be reset after /new"
        );
    }

    #[tokio::test]
    async fn role_command_with_valid_role_switches_backend_and_model() {
        use crate::config::{ModelRole, OpenAiCompatConfigToml, ReasoningStyleConfig};

        let dir = tempfile::TempDir::new().expect("temp dir");
        let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "original-model".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                    thinking: None,
                    cancel_token: None,
                },
                session,
            )
            .await,
        );

        let mut config = make_config();
        config.openai_compat = Some(OpenAiCompatConfigToml {
            base_url: "https://example.com/v1".to_string(),
            api_key: None,
            model: "fast-model".to_string(),
            max_tokens: None,
            reasoning: ReasoningStyleConfig::None,
        });
        config.models.insert(
            "fast".to_string(),
            ModelRole {
                backend: "openai_compat".to_string(),
                model: "fast-model".to_string(),
            },
        );
        let factory = Arc::new(BackendFactory::new(config.clone()));

        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        app.model = "original-model".to_string();
        let cmd = RoleCommand;
        let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
            event_tx: &event_tx,
            backend_factory: factory,
        };
        let result = cmd.execute("fast", &mut ctx).await.expect("execute");
        assert_eq!(result, DispatchResult::Handled);
        assert_eq!(
            ctx.app.model, "fast-model",
            "app.model should be updated to the role's model"
        );
        assert_eq!(
            ctx.agent.model(),
            "fast-model",
            "agent model should be updated to the role's model"
        );
        assert_eq!(
            ctx.agent.max_tokens(),
            16384,
            "agent max_tokens should be updated to the openai_compat default (16384); got {}",
            ctx.agent.max_tokens()
        );
        assert!(
            ctx.app
                .conversation
                .iter()
                .any(|e| e.role == ConversationRole::Info
                    && e.content.contains("Switched to role 'fast'")),
            "should show success message for valid role switch"
        );
    }

    #[tokio::test]
    async fn bash_command_empty_args_shows_usage_hint() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "test".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                    thinking: None,
                    cancel_token: None,
                },
                session,
            )
            .await,
        );
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        let cmd = BashCommand;
        let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
            event_tx: &event_tx,
            backend_factory: make_factory(),
        };
        let result = cmd.execute("", &mut ctx).await.expect("execute");
        assert_eq!(result, DispatchResult::Handled);
        assert_eq!(
            ctx.app.state,
            AppState::Input,
            "empty /bash should not change state"
        );
        assert!(
            ctx.app
                .conversation
                .iter()
                .any(|e| e.role == ConversationRole::Info && e.content.contains("Usage: /bash")),
            "should show usage hint"
        );
    }

    #[tokio::test]
    async fn bash_command_with_args_sets_running_bash_state() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "test".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                    thinking: None,
                    cancel_token: None,
                },
                session,
            )
            .await,
        );
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        let cmd = BashCommand;
        let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
            event_tx: &event_tx,
            backend_factory: make_factory(),
        };
        let result = cmd.execute("echo hi", &mut ctx).await.expect("execute");
        assert_eq!(result, DispatchResult::Handled);
        assert_eq!(
            ctx.app.state,
            AppState::RunningBash,
            "/bash with args should set RunningBash state"
        );
        assert!(
            ctx.app.cancel_token.is_some(),
            "cancel_token should be set during bash execution"
        );
        let event = event_rx
            .recv()
            .await
            .expect("should receive ToolUseReceived");
        assert!(
            matches!(event, AgentEvent::ToolUseReceived { name, .. } if name == "bash"),
            "should send ToolUseReceived with name=bash"
        );
    }

    #[tokio::test]
    async fn new_command_session_creation_failure_returns_error() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "test-model".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                    thinking: None,
                    cancel_token: None,
                },
                session,
            )
            .await,
        );
        let mut config = make_config();
        config.sessions_dir = std::path::PathBuf::from("/nonexistent/path/that/does/not/exist");
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        let cmd = NewCommand;
        let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
            event_tx: &event_tx,
            backend_factory: make_factory(),
        };
        let result = cmd.execute("", &mut ctx).await;
        assert!(
            result.is_err(),
            "/new with invalid sessions_dir should return Err"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to create new session"),
            "error should mention session creation failure, got: {err_msg}"
        );
    }

    #[tokio::test]
    async fn bash_command_esc_cancels_and_records_cancellation_result() {
        use crate::frontend::tui::tui_app::{KeyDisposition, handle_agent_event};
        use crate::types::ContentBlock;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let dir = tempfile::TempDir::new().expect("temp dir");
        let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "test".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                    thinking: None,
                    cancel_token: None,
                },
                session,
            )
            .await,
        );
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        let cmd = BashCommand;
        let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
        let mut ctx = CommandContext {
            app: &mut app,
            agent: Arc::clone(&agent),
            config: &config,
            event_tx: &event_tx,
            backend_factory: make_factory(),
        };

        // Dispatch /bash sleep 30 (long-running)
        let result = cmd.execute("sleep 30", &mut ctx).await.expect("execute");
        assert_eq!(result, DispatchResult::Handled);
        assert_eq!(ctx.app.state, AppState::RunningBash);

        // Simulate Esc — cancels the token
        let esc_key = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        let disposition = ctx.app.application_command(&esc_key);
        assert_eq!(
            disposition,
            KeyDisposition::Consumed,
            "Esc in RunningBash should be Consumed by application_command"
        );

        // Drop ctx so app is accessible
        drop(ctx);

        // Drain events until BashCommandComplete (with timeout)
        let timeout = std::time::Duration::from_secs(5);
        let start = std::time::Instant::now();
        let mut got_complete = false;
        while start.elapsed() < timeout {
            match tokio::time::timeout(std::time::Duration::from_millis(200), event_rx.recv()).await
            {
                Ok(Some(event)) => {
                    if matches!(event, AgentEvent::BashCommandComplete) {
                        handle_agent_event(&mut app, event, None).expect("handle");
                        got_complete = true;
                        break;
                    } else {
                        let _ = handle_agent_event(&mut app, event, None);
                    }
                }
                Ok(None) => break,
                Err(_) => continue,
            }
        }

        assert!(
            got_complete,
            "BashCommandComplete should be received after Esc"
        );
        assert_eq!(
            app.state,
            AppState::Input,
            "state should return to Input after BashCommandComplete"
        );
        assert!(app.cancel_token.is_none(), "cancel_token should be cleared");

        // History should contain the synthetic pair with cancellation content
        let history = agent.history();
        let has_cancelled = history.iter().any(|m| {
            m.content.iter().any(|b| {
                matches!(
                    b,
                    ContentBlock::ToolResult {
                        content,
                        is_error,
                        ..
                    } if content.iter().any(|cb| matches!(cb, ContentBlock::Text(t) if t.contains("cancelled"))) && *is_error
                )
            })
        });
        assert!(
            has_cancelled,
            "history should contain ToolResult with 'cancelled' content and is_error=true"
        );
    }

    #[test]
    fn parse_command_recognizes_chat() {
        let parsed = parse_command("/chat").expect("should parse");
        assert_eq!(parsed.name, "chat");
        assert_eq!(parsed.args, "");
    }

    #[tokio::test]
    async fn chat_command_toggles_mode_on() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "test".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                    thinking: None,
                    cancel_token: None,
                },
                session,
            )
            .await,
        );
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        let cmd = ChatCommand;
        let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
            event_tx: &event_tx,
            backend_factory: make_factory(),
        };
        let result = cmd.execute("", &mut ctx).await.expect("execute");
        assert_eq!(result, DispatchResult::Handled);
        assert!(ctx.app.chat_mode.is_on());
        assert!(ctx.agent.is_chat_mode());
        assert!(
            ctx.app
                .conversation
                .iter()
                .any(|e| e.role == ConversationRole::Info && e.content.contains("Chat mode ON")),
            "should show ON message"
        );
    }

    #[tokio::test]
    async fn chat_command_toggles_mode_off() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let session_inner = crate::session::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(session_inner));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "test".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                    thinking: None,
                    cancel_token: None,
                },
                session,
            )
            .await,
        );
        agent.set_chat_mode(true);
        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        app.chat_mode.set(true);
        let cmd = ChatCommand;
        let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
            event_tx: &event_tx,
            backend_factory: make_factory(),
        };
        let result = cmd.execute("", &mut ctx).await.expect("execute");
        assert_eq!(result, DispatchResult::Handled);
        assert!(!ctx.app.chat_mode.is_on());
        assert!(!ctx.agent.is_chat_mode());
        assert!(
            ctx.app
                .conversation
                .iter()
                .any(|e| e.role == ConversationRole::Info && e.content.contains("Chat mode OFF")),
            "should show OFF message"
        );
    }
}
