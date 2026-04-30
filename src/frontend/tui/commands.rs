use crate::agent::Agent;
use crate::config::AppConfig;
use crate::frontend::tui::tasks_picker::{TasksPicker, sort_tasks};
use crate::frontend::tui::tui_app::{App, AppState};
use crate::frontend::tui::{ConversationEntry, ConversationRole, SessionPicker};
use crate::session::list_sessions;
use std::sync::Arc;

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
            match list_sessions(&ctx.config.sessions_dir).await {
                Ok(sessions) => {
                    ctx.app.session_picker = Some(SessionPicker::new(sessions));
                    ctx.app.set_state(AppState::SessionPicker);
                }
                Err(e) => {
                    ctx.app.conversation.push(ConversationEntry::new(
                        ConversationRole::Error,
                        format!("Failed to list sessions: {e}"),
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
                ));
            } else {
                ctx.agent.set_model(model.clone());
                ctx.app.model = model.clone();
                ctx.app.conversation.push(ConversationEntry::new(
                    ConversationRole::Info,
                    format!("Model switched to `{model}`"),
                ));
            }
            Ok(DispatchResult::Handled)
        })
    }
}

/// Built-in `/tasks` command — opens the tasks picker overlay.
pub struct TasksCommand;

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
                    ));
                }
            }
            Ok(DispatchResult::Handled)
        })
    }
}

/// Built-in `/compact` command — manually triggers context compaction.
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
        Box::pin(async move {
            ctx.app.input.clear();
            match ctx.agent.compact().await {
                Ok((removed, kept)) => {
                    ctx.app.conversation.push(ConversationEntry::new(
                        ConversationRole::Info,
                        format!(
                            "Context compacted: {removed} messages removed, \
                             {kept} recent messages retained."
                        ),
                    ));
                }
                Err(e) => {
                    ctx.app.conversation.push(ConversationEntry::new(
                        ConversationRole::Error,
                        format!("Compaction failed: {e}"),
                    ));
                }
            }
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
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AppConfig, ToolsConfig, VertexConfig};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn make_config() -> AppConfig {
        AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        }
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
                    },
                    session,
                )
                .await,
            );
            let mut ctx = CommandContext {
                app: &mut app,
                agent,
                config: &config,
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
                    },
                    session,
                )
                .await,
            );
            let mut ctx = CommandContext {
                app: &mut app,
                agent,
                config: &config,
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
                    },
                    session,
                )
                .await,
            );
            let cmd = SessionsCommand;
            let mut ctx = CommandContext {
                app: &mut app,
                agent,
                config: &config,
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
                    },
                    session,
                )
                .await,
            );
            let cmd = ModelCommand;
            let mut ctx = CommandContext {
                app: &mut app,
                agent: Arc::clone(&agent),
                config: &config,
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
                    },
                    session,
                )
                .await,
            );
            let cmd = ModelCommand;
            let mut ctx = CommandContext {
                app: &mut app,
                agent: Arc::clone(&agent),
                config: &config,
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
                    },
                    session,
                )
                .await,
            );
            let cmd = TasksCommand;
            let mut ctx = CommandContext {
                app: &mut app,
                agent,
                config: &config,
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
                },
                session_arc,
            )
            .await,
        );

        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        let cmd = TasksCommand;
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
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
    async fn compact_command_without_config_shows_error() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let session = crate::session::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("session");
        let session_arc = std::sync::Arc::new(tokio::sync::Mutex::new(session));
        let agent = Arc::new(
            crate::agent::Agent::new(
                Box::new(FakeBackend),
                crate::types::RequestConfig {
                    model: "test".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                },
                session_arc,
            )
            .await,
        );

        let config = make_config();
        let tools = Arc::new(crate::tools::ToolRegistry::new());
        let mut app = App::new(Arc::clone(&tools));
        let cmd = CompactCommand;
        let mut ctx = CommandContext {
            app: &mut app,
            agent,
            config: &config,
        };
        let result = cmd.execute("", &mut ctx).await.expect("execute");
        assert_eq!(result, DispatchResult::Handled);
        assert!(
            ctx.app
                .conversation
                .iter()
                .any(|e| e.role == ConversationRole::Error && e.content.contains("not configured")),
            "should show not-configured error"
        );
    }
}
