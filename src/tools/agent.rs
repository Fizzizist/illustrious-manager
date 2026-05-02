use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::{AgentSpawner, HeadlessOutcome, clamp_confirmation};
use crate::config::ConfirmationMode;
use crate::tools::{Tool, ToolError, ToolResult};
use crate::types::ContentBlock;

/// Trait allowing `AgentTool` to be tested with a fake spawner.
#[async_trait]
pub trait SubAgentSpawner: Send + Sync {
    async fn spawn(
        &self,
        role: &str,
        confirmation: ConfirmationMode,
        tool_allowlist: Option<&[String]>,
        prompt: String,
    ) -> HeadlessOutcome;

    fn parent_confirmation(&self) -> &ConfirmationMode;
}

#[async_trait]
impl SubAgentSpawner for AgentSpawner {
    async fn spawn(
        &self,
        role: &str,
        confirmation: ConfirmationMode,
        tool_allowlist: Option<&[String]>,
        prompt: String,
    ) -> HeadlessOutcome {
        AgentSpawner::spawn(self, role, confirmation, tool_allowlist, prompt).await
    }

    fn parent_confirmation(&self) -> &ConfirmationMode {
        &self.parent_confirmation
    }
}

pub struct AgentTool {
    spawner: Arc<dyn SubAgentSpawner>,
    available_roles: Vec<String>,
    schema: Value,
    description_text: String,
}

impl AgentTool {
    pub fn new(spawner: Arc<AgentSpawner>, available_roles: Vec<String>) -> Self {
        Self::with_spawner(spawner, available_roles)
    }

    pub fn with_spawner(spawner: Arc<dyn SubAgentSpawner>, available_roles: Vec<String>) -> Self {
        let schema = build_schema(&available_roles);
        let description_text = build_description(&available_roles);
        Self {
            spawner,
            available_roles,
            schema,
            description_text,
        }
    }
}

fn build_schema(available_roles: &[String]) -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The task or question for the sub-agent to handle."
            },
            "role": {
                "type": "string",
                "description": "Named model role from [models] config. Defaults to 'default'.",
                "enum": available_roles
            },
            "confirmation": {
                "type": "string",
                "enum": ["Always", "WriteOnly", "Never"],
                "description": "Confirmation mode for the sub-agent. Clamped to be no more permissive than the parent."
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit allowlist of tool names for the sub-agent. If omitted, inherits the parent's tool set."
            }
        },
        "required": ["prompt"]
    })
}

fn build_description(available_roles: &[String]) -> String {
    let role_list = available_roles.join(", ");
    format!(
        "Spawn an independent sub-agent to run a focused task in its own session. \
         The sub-agent has its own conversation history and tool set. \
         Returns the sub-agent's final response text.\n\n\
         NOTE: Sub-agents inherit the parent's confirmation mode by default. \
         In `WriteOnly` or `Always` mode the sub-agent cannot perform write tool calls \
         without aborting — pass `confirmation: \"Never\"` only when the parent also \
         runs in `Never` mode (clamping prevents elevation beyond the parent).\n\n\
         Available roles: {role_list}"
    )
}

#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &str {
        "agent"
    }

    fn description(&self) -> &str {
        &self.description_text
    }

    fn input_schema(&self) -> &Value {
        &self.schema
    }

    fn is_write_tool(&self) -> bool {
        true
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let prompt = input["prompt"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput {
                message: "Missing required field 'prompt'".to_string(),
            })?
            .to_string();

        let role = input["role"].as_str().unwrap_or("default").to_string();

        if !self.available_roles.contains(&role) {
            let valid = self.available_roles.join(", ");
            return Err(ToolError::InvalidInput {
                message: format!("Invalid role '{role}'. Available roles: {valid}"),
            });
        }

        let requested_confirmation = input["confirmation"].as_str().and_then(|s| match s {
            "Always" => Some(ConfirmationMode::Always),
            "WriteOnly" => Some(ConfirmationMode::WriteOnly),
            "Never" => Some(ConfirmationMode::Never),
            _ => None,
        });

        let confirmation = clamp_confirmation(
            self.spawner.parent_confirmation(),
            requested_confirmation.as_ref(),
        );

        let tool_allowlist: Option<Vec<String>> = input["tools"].as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        });

        let outcome = self
            .spawner
            .spawn(&role, confirmation, tool_allowlist.as_deref(), prompt)
            .await;

        if outcome.is_error {
            let msg = outcome
                .error_message
                .unwrap_or_else(|| "Sub-agent failed".to_string());
            Ok(ToolResult {
                content: vec![ContentBlock::Text(msg)],
                is_error: true,
                agent_events: vec![],
            })
        } else {
            let usage_event = crate::types::AgentEvent::SubAgentUsage {
                input_tokens: outcome.input_tokens,
                output_tokens: outcome.output_tokens,
                role: role.clone(),
            };
            Ok(ToolResult {
                content: vec![ContentBlock::Text(outcome.text)],
                is_error: false,
                agent_events: vec![usage_event],
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::HeadlessOutcome;
    use crate::config::ConfirmationMode;
    use crate::types::ContentBlock;
    use std::sync::Arc;

    // ── Fake spawner ──────────────────────────────────────────────────────

    struct FakeSpawner {
        outcome: HeadlessOutcome,
        parent: ConfirmationMode,
        captured_role: std::sync::Mutex<Option<String>>,
        captured_confirmation: std::sync::Mutex<Option<ConfirmationMode>>,
        captured_allowlist: std::sync::Mutex<Option<Option<Vec<String>>>>,
    }

    impl FakeSpawner {
        fn new(outcome: HeadlessOutcome, parent: ConfirmationMode) -> Arc<Self> {
            Arc::new(Self {
                outcome,
                parent,
                captured_role: std::sync::Mutex::new(None),
                captured_confirmation: std::sync::Mutex::new(None),
                captured_allowlist: std::sync::Mutex::new(None),
            })
        }
    }

    #[async_trait]
    impl SubAgentSpawner for FakeSpawner {
        async fn spawn(
            &self,
            role: &str,
            confirmation: ConfirmationMode,
            tool_allowlist: Option<&[String]>,
            _prompt: String,
        ) -> HeadlessOutcome {
            *self.captured_role.lock().expect("lock") = Some(role.to_string());
            *self.captured_confirmation.lock().expect("lock") = Some(confirmation);
            *self.captured_allowlist.lock().expect("lock") =
                Some(tool_allowlist.map(|l| l.to_vec()));

            HeadlessOutcome {
                text: self.outcome.text.clone(),
                input_tokens: self.outcome.input_tokens,
                output_tokens: self.outcome.output_tokens,
                is_error: self.outcome.is_error,
                error_message: self.outcome.error_message.clone(),
            }
        }

        fn parent_confirmation(&self) -> &ConfirmationMode {
            &self.parent
        }
    }

    fn ok_outcome(text: &str) -> HeadlessOutcome {
        HeadlessOutcome {
            text: text.to_string(),
            input_tokens: 10,
            output_tokens: 5,
            is_error: false,
            error_message: None,
        }
    }

    fn err_outcome(msg: &str) -> HeadlessOutcome {
        HeadlessOutcome {
            text: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            is_error: true,
            error_message: Some(msg.to_string()),
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────

    fn agent_tool_with_spawner(
        outcome: HeadlessOutcome,
        parent: ConfirmationMode,
    ) -> (AgentTool, Arc<FakeSpawner>) {
        let spawner = FakeSpawner::new(outcome, parent);
        let tool = AgentTool::with_spawner(
            Arc::clone(&spawner) as Arc<dyn SubAgentSpawner>,
            vec!["default".to_string()],
        );
        (tool, spawner)
    }

    #[tokio::test]
    async fn agent_tool_returns_final_text_from_subagent() {
        let (tool, spawner) =
            agent_tool_with_spawner(ok_outcome("sub-agent result"), ConfirmationMode::Never);
        let input = serde_json::json!({"prompt": "do something"});
        let result = tool.execute(input).await.expect("execute should succeed");
        assert!(!result.is_error);
        assert!(matches!(&result.content[0], ContentBlock::Text(t) if t == "sub-agent result"));
        let captured = spawner
            .captured_role
            .lock()
            .expect("lock")
            .clone()
            .expect("should be set");
        assert_eq!(captured, "default");
    }

    #[tokio::test]
    async fn agent_tool_surfaces_backend_error_as_error_result() {
        let (tool, _) =
            agent_tool_with_spawner(err_outcome("backend exploded"), ConfirmationMode::Never);
        let input = serde_json::json!({"prompt": "do something"});
        let result = tool.execute(input).await.expect("execute should succeed");
        assert!(result.is_error);
        assert!(
            matches!(&result.content[0], ContentBlock::Text(t) if t.contains("backend exploded"))
        );
    }

    #[tokio::test]
    async fn agent_tool_clamps_confirmation_mode_never_requested_always_parent() {
        let (tool, spawner) = agent_tool_with_spawner(ok_outcome("ok"), ConfirmationMode::Always);
        let input = serde_json::json!({"prompt": "do", "confirmation": "Never"});
        tool.execute(input).await.expect("ok");
        let captured = spawner
            .captured_confirmation
            .lock()
            .expect("lock")
            .clone()
            .expect("should be set");
        assert_eq!(
            captured,
            ConfirmationMode::Always,
            "Never requested but parent is Always → should be clamped to Always"
        );
    }

    #[tokio::test]
    async fn agent_tool_restricts_tools_when_allowlist_provided() {
        let (tool, spawner) = agent_tool_with_spawner(ok_outcome("ok"), ConfirmationMode::Never);
        let input = serde_json::json!({"prompt": "do", "tools": ["bash", "search"]});
        tool.execute(input).await.expect("ok");
        let captured = spawner
            .captured_allowlist
            .lock()
            .expect("lock")
            .clone()
            .expect("should be set");
        let list = captured.expect("allowlist should be Some");
        assert_eq!(list, vec!["bash".to_string(), "search".to_string()]);
    }

    #[tokio::test]
    async fn agent_tool_no_tools_field_means_no_allowlist() {
        let (tool, spawner) = agent_tool_with_spawner(ok_outcome("ok"), ConfirmationMode::Never);
        let input = serde_json::json!({"prompt": "do"});
        tool.execute(input).await.expect("ok");
        let captured = spawner
            .captured_allowlist
            .lock()
            .expect("lock")
            .clone()
            .expect("should be set");
        assert!(
            captured.is_none(),
            "no 'tools' field → allowlist should be None"
        );
    }

    // ── Integration test: AgentTool creates a new session DB ─────────────
    //
    // Uses a real AgentSpawner wired to a fake LLM backend via
    // `spawn_agent_with_selection`, bypassing BackendFactory auth.

    #[tokio::test]
    async fn agent_tool_creates_new_session_db() {
        use crate::agent::spawn_agent_with_selection;
        use crate::backend::BackendSelection;
        use crate::config::{AppConfig, CompactionConfig, ToolsConfig, VertexConfig};
        use crate::session::Session;
        use crate::tools::ToolRegistry;
        use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};
        use anyhow::Result;
        use async_trait::async_trait;
        use futures::stream;
        use std::collections::BTreeMap;
        use std::sync::Arc;
        use tokio::sync::Mutex as TokioMutex;

        struct ImmediateTextBackend;

        #[async_trait]
        impl crate::backend::LlmBackend for ImmediateTextBackend {
            async fn send_message(
                &self,
                _: &[Message],
                _: &RequestConfig,
            ) -> Result<BoxStream<Result<StreamEvent>>> {
                Ok(Box::pin(stream::iter(vec![
                    Ok(StreamEvent::TextDelta("sub-agent done".to_string())),
                    Ok(StreamEvent::Done),
                ])))
            }
        }

        let dir = tempfile::TempDir::new().expect("temp dir");
        let dir_path = dir.path().to_path_buf();

        let app_config = Arc::new(AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "test".to_string(),
                region: "us-east5".to_string(),
                model: "claude-test".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig {
                confirmation: ConfirmationMode::Never,
                ..Default::default()
            },
            sessions_dir: dir_path.clone(),
            models: BTreeMap::new(),
            compaction: CompactionConfig::default(),
            thinking: None,
        });

        // Spawner that wires ImmediateTextBackend bypassing real auth.
        let dir_path_clone = dir_path.clone();
        let app_config_clone = Arc::clone(&app_config);
        let spawner: Arc<dyn SubAgentSpawner> = Arc::new(DirectSpawner {
            dir_path: dir_path_clone,
            app_config: app_config_clone,
        });

        struct DirectSpawner {
            dir_path: std::path::PathBuf,
            app_config: Arc<AppConfig>,
        }

        #[async_trait]
        impl SubAgentSpawner for DirectSpawner {
            async fn spawn(
                &self,
                _role: &str,
                confirmation: ConfirmationMode,
                _tool_allowlist: Option<&[String]>,
                prompt: String,
            ) -> HeadlessOutcome {
                use crate::agent::run_headless;

                let session = match Session::new(None, self.dir_path.clone()).await {
                    Ok(s) => Arc::new(TokioMutex::new(s)),
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

                let tool_config = crate::config::ToolsConfig {
                    confirmation,
                    ..self.app_config.tools.clone()
                };

                let selection = BackendSelection {
                    backend: Box::new(ImmediateTextBackend),
                    model: "claude-test".to_string(),
                };

                let agent = match spawn_agent_with_selection(
                    selection,
                    &tool_config,
                    session,
                    ToolRegistry::new(),
                )
                .await
                {
                    Ok(a) => a,
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

                run_headless(&agent, prompt).await
            }

            fn parent_confirmation(&self) -> &ConfirmationMode {
                &self.app_config.tools.confirmation
            }
        }

        let tool = AgentTool::with_spawner(spawner, vec!["default".to_string()]);

        let count_before = std::fs::read_dir(&dir_path)
            .expect("read dir")
            .filter(|e| {
                e.as_ref()
                    .ok()
                    .and_then(|e| e.path().extension().map(|x| x == "db"))
                    .unwrap_or(false)
            })
            .count();

        let input = serde_json::json!({"prompt": "do something"});
        let result = tool.execute(input).await.expect("execute should succeed");

        assert!(!result.is_error, "sub-agent should complete without error");
        assert!(
            matches!(&result.content[0], ContentBlock::Text(t) if t == "sub-agent done"),
            "result should contain sub-agent response text"
        );

        let count_after = std::fs::read_dir(&dir_path)
            .expect("read dir")
            .filter(|e| {
                e.as_ref()
                    .ok()
                    .and_then(|e| e.path().extension().map(|x| x == "db"))
                    .unwrap_or(false)
            })
            .count();

        assert_eq!(
            count_after,
            count_before + 1,
            "AgentTool::execute must create exactly one new session DB file"
        );
    }

    #[test]
    fn agent_tool_schema_lists_available_roles() {
        let roles = vec![
            "default".to_string(),
            "implement".to_string(),
            "thinking".to_string(),
        ];
        let spawner = FakeSpawner::new(ok_outcome("ok"), ConfirmationMode::Never);
        let tool = AgentTool::with_spawner(Arc::clone(&spawner) as Arc<dyn SubAgentSpawner>, roles);
        let schema = tool.input_schema();
        let role_enum = schema["properties"]["role"]["enum"]
            .as_array()
            .expect("enum array");
        assert_eq!(role_enum.len(), 3);
        assert_eq!(role_enum[0].as_str(), Some("default"));
        assert_eq!(role_enum[1].as_str(), Some("implement"));
        assert_eq!(role_enum[2].as_str(), Some("thinking"));
    }

    #[test]
    fn agent_tool_description_lists_available_roles() {
        let roles = vec!["default".to_string(), "implement".to_string()];
        let spawner = FakeSpawner::new(ok_outcome("ok"), ConfirmationMode::Never);
        let tool = AgentTool::with_spawner(Arc::clone(&spawner) as Arc<dyn SubAgentSpawner>, roles);
        let desc = tool.description();
        assert!(
            desc.contains("Available roles: default, implement"),
            "description should list roles, got: {desc}"
        );
    }

    #[tokio::test]
    async fn agent_tool_rejects_invalid_role_with_error() {
        let spawner = FakeSpawner::new(ok_outcome("ok"), ConfirmationMode::Never);
        let tool = AgentTool::with_spawner(
            Arc::clone(&spawner) as Arc<dyn SubAgentSpawner>,
            vec!["default".to_string(), "implement".to_string()],
        );
        let input = serde_json::json!({"prompt": "do", "role": "nonexistent"});
        let result = tool.execute(input).await;
        assert!(result.is_err(), "should reject invalid role");
        let msg = match result {
            Err(ToolError::InvalidInput { message }) => message,
            _ => panic!("expected InvalidInput error"),
        };
        assert!(
            msg.contains("Invalid role 'nonexistent'"),
            "error should mention the invalid role, got: {msg}"
        );
        assert!(
            msg.contains("Available roles: default, implement"),
            "error should list valid roles, got: {msg}"
        );
    }

    #[tokio::test]
    async fn agent_tool_forwards_explicit_valid_role_to_spawner() {
        let spawner = FakeSpawner::new(ok_outcome("ok"), ConfirmationMode::Never);
        let tool = AgentTool::with_spawner(
            Arc::clone(&spawner) as Arc<dyn SubAgentSpawner>,
            vec!["default".to_string(), "implement".to_string()],
        );
        let input = serde_json::json!({"prompt": "refactor this", "role": "implement"});
        let result = tool.execute(input).await.expect("execute should succeed");
        assert!(!result.is_error);
        let captured = spawner
            .captured_role
            .lock()
            .expect("lock")
            .clone()
            .expect("should be set");
        assert_eq!(
            captured, "implement",
            "explicit valid role should be forwarded to spawner"
        );
    }
}
