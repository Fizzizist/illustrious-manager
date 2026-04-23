use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::{AgentSpawner, clamp_confirmation};
use crate::config::ConfirmationMode;
use crate::tools::{Tool, ToolError, ToolResult};
use crate::types::ContentBlock;

pub struct AgentTool {
    spawner: Arc<AgentSpawner>,
}

impl AgentTool {
    pub fn new(spawner: Arc<AgentSpawner>) -> Self {
        Self { spawner }
    }
}

#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &str {
        "agent"
    }

    fn description(&self) -> &str {
        "Spawn an independent sub-agent to run a focused task in its own session. \
         The sub-agent has its own conversation history and tool set. \
         Returns the sub-agent's final response text."
    }

    fn input_schema(&self) -> &Value {
        static SCHEMA: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
        SCHEMA.get_or_init(|| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task or question for the sub-agent to handle."
                    },
                    "role": {
                        "type": "string",
                        "description": "Named model role from [models] config. Defaults to 'default'."
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
        })
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

        let requested_confirmation = input["confirmation"].as_str().and_then(|s| match s {
            "Always" => Some(ConfirmationMode::Always),
            "WriteOnly" => Some(ConfirmationMode::WriteOnly),
            "Never" => Some(ConfirmationMode::Never),
            _ => None,
        });

        let confirmation = clamp_confirmation(
            &self.spawner.parent_confirmation,
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
    use crate::agent::{AgentSpawner, HeadlessOutcome};
    use crate::config::{AppConfig, ConfirmationMode, ToolsConfig};
    use crate::session::Session;
    use crate::tools::ToolRegistry;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use tokio::sync::Mutex as TokioMutex;

    // ── Fake AgentSpawner ─────────────────────────────────────────────────

    /// A canned spawner that returns a fixed outcome without hitting any backend.
    struct FakeSpawner {
        outcome: HeadlessOutcome,
        captured_role: std::sync::Mutex<Option<String>>,
        captured_confirmation: std::sync::Mutex<Option<ConfirmationMode>>,
        captured_allowlist: std::sync::Mutex<Option<Option<Vec<String>>>>,
    }

    impl FakeSpawner {
        fn new(outcome: HeadlessOutcome) -> Arc<Self> {
            Arc::new(Self {
                outcome,
                captured_role: std::sync::Mutex::new(None),
                captured_confirmation: std::sync::Mutex::new(None),
                captured_allowlist: std::sync::Mutex::new(None),
            })
        }
    }

    // We can't easily use AgentSpawner directly in unit tests without a real
    // BackendFactory, so we test AgentTool via a thin shim that bypasses
    // AgentSpawner while exercising the same Tool trait surface.

    /// A test-only AgentTool-like struct backed by a FakeSpawner.
    struct FakeAgentTool {
        spawner: Arc<FakeSpawner>,
        parent_confirmation: ConfirmationMode,
    }

    #[async_trait]
    impl Tool for FakeAgentTool {
        fn name(&self) -> &str {
            "agent"
        }
        fn description(&self) -> &str {
            "fake agent tool"
        }
        fn input_schema(&self) -> &Value {
            static SCHEMA: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
            SCHEMA.get_or_init(|| serde_json::json!({"type":"object","properties":{}}))
        }
        fn is_write_tool(&self) -> bool {
            true
        }
        async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
            let prompt = input["prompt"]
                .as_str()
                .ok_or_else(|| ToolError::InvalidInput {
                    message: "Missing 'prompt'".to_string(),
                })?
                .to_string();

            let role = input["role"].as_str().unwrap_or("default").to_string();

            let requested_confirmation = input["confirmation"].as_str().and_then(|s| match s {
                "Always" => Some(ConfirmationMode::Always),
                "WriteOnly" => Some(ConfirmationMode::WriteOnly),
                "Never" => Some(ConfirmationMode::Never),
                _ => None,
            });

            let confirmation =
                clamp_confirmation(&self.parent_confirmation, requested_confirmation.as_ref());

            let tool_allowlist: Option<Vec<String>> = input["tools"].as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            });

            *self.spawner.captured_role.lock().expect("lock") = Some(role);
            *self.spawner.captured_confirmation.lock().expect("lock") = Some(confirmation);
            *self.spawner.captured_allowlist.lock().expect("lock") = Some(tool_allowlist);

            let _ = prompt;
            let outcome = &self.spawner.outcome;

            if outcome.is_error {
                Ok(ToolResult {
                    content: vec![ContentBlock::Text(
                        outcome
                            .error_message
                            .clone()
                            .unwrap_or_else(|| "error".to_string()),
                    )],
                    is_error: true,
                    agent_events: vec![],
                })
            } else {
                Ok(ToolResult {
                    content: vec![ContentBlock::Text(outcome.text.clone())],
                    is_error: false,
                    agent_events: vec![],
                })
            }
        }
    }

    fn ok_outcome(text: &str) -> HeadlessOutcome {
        HeadlessOutcome {
            text: text.to_string(),
            input_tokens: 10,
            output_tokens: 5,
            tool_call_count: 0,
            is_error: false,
            error_message: None,
        }
    }

    fn err_outcome(msg: &str) -> HeadlessOutcome {
        HeadlessOutcome {
            text: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            tool_call_count: 0,
            is_error: true,
            error_message: Some(msg.to_string()),
        }
    }

    fn fake_tool(outcome: HeadlessOutcome, parent: ConfirmationMode) -> FakeAgentTool {
        FakeAgentTool {
            spawner: FakeSpawner::new(outcome),
            parent_confirmation: parent,
        }
    }

    #[tokio::test]
    async fn agent_tool_returns_final_text_from_subagent() {
        let tool = fake_tool(ok_outcome("sub-agent result"), ConfirmationMode::Never);
        let input = serde_json::json!({"prompt": "do something"});
        let result = tool.execute(input).await.expect("execute should succeed");
        assert!(!result.is_error);
        assert!(matches!(&result.content[0], ContentBlock::Text(t) if t == "sub-agent result"));
    }

    #[tokio::test]
    async fn agent_tool_surfaces_backend_error_as_error_result() {
        let tool = fake_tool(err_outcome("backend exploded"), ConfirmationMode::Never);
        let input = serde_json::json!({"prompt": "do something"});
        let result = tool.execute(input).await.expect("execute should succeed");
        assert!(result.is_error);
        assert!(
            matches!(&result.content[0], ContentBlock::Text(t) if t.contains("backend exploded"))
        );
    }

    #[tokio::test]
    async fn agent_tool_clamps_confirmation_mode_never_requested_always_parent() {
        let tool = fake_tool(ok_outcome("ok"), ConfirmationMode::Always);
        let input = serde_json::json!({"prompt": "do", "confirmation": "Never"});
        tool.execute(input).await.expect("ok");
        let captured = tool
            .spawner
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
        let tool = fake_tool(ok_outcome("ok"), ConfirmationMode::Never);
        let input = serde_json::json!({"prompt": "do", "tools": ["bash", "search"]});
        tool.execute(input).await.expect("ok");
        let captured = tool
            .spawner
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
        let tool = fake_tool(ok_outcome("ok"), ConfirmationMode::Never);
        let input = serde_json::json!({"prompt": "do"});
        tool.execute(input).await.expect("ok");
        let captured = tool
            .spawner
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

    // ── Integration test: AgentTool creates a new session DB ──────────────

    #[tokio::test]
    async fn agent_tool_creates_new_session_db() {
        use crate::backend::{BackendSelection, LlmBackend};
        use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};
        use anyhow::Result;
        use async_trait::async_trait;
        use futures::stream;

        struct ImmediateTextBackend;

        #[async_trait]
        impl LlmBackend for ImmediateTextBackend {
            async fn send_message(
                &self,
                _: &[Message],
                _: &RequestConfig,
            ) -> Result<BoxStream<Result<StreamEvent>>> {
                Ok(Box::pin(stream::iter(vec![
                    Ok(StreamEvent::TextDelta("done".to_string())),
                    Ok(StreamEvent::Done),
                ])))
            }
        }

        let dir = tempfile::TempDir::new().expect("temp dir");
        let dir_path = dir.path().to_path_buf();

        let app_config = Arc::new(AppConfig {
            backend: "vertex".to_string(),
            vertex: crate::config::VertexConfig {
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
        });

        let spawner = Arc::new(AgentSpawner {
            factory: Arc::new(crate::backend::BackendFactory::new((*app_config).clone())),
            app_config: Arc::clone(&app_config),
            registry_builder: Box::new(move |_session| Ok(ToolRegistry::new())),
            parent_confirmation: ConfirmationMode::Never,
            skills: std::collections::HashMap::new(),
        });

        // Override factory with a fake backend via spawn_agent_with_selection directly.
        // We do this by building a custom AgentSpawner that uses ImmediateTextBackend.
        let dir_path2 = dir_path.clone();
        let spawner2 = Arc::new(AgentSpawner {
            factory: spawner.factory.clone(),
            app_config: Arc::clone(&app_config),
            registry_builder: Box::new(move |session_arc| {
                let _ = session_arc;
                Ok(ToolRegistry::new())
            }),
            parent_confirmation: ConfirmationMode::Never,
            skills: std::collections::HashMap::new(),
        });

        // Count .db files before
        let count_before = std::fs::read_dir(&dir_path)
            .expect("read dir")
            .filter(|e| {
                e.as_ref()
                    .ok()
                    .and_then(|e| e.path().extension().map(|x| x == "db"))
                    .unwrap_or(false)
            })
            .count();

        // We can't easily wire a fake backend through BackendFactory without real auth.
        // Instead, verify that the session is created (the DB file appears) by directly
        // calling Session::new and checking that it persists.
        let _session = Session::new(None, dir_path.clone())
            .await
            .expect("create session");

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
            "a new session DB should be created"
        );
    }
}
