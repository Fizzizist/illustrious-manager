use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex as TokioMutex;

use crate::session::Session;
use crate::tools::{Tool, ToolError, ToolResult};
use crate::types::ContentBlock;

pub struct CreateTaskTool {
    session: Arc<TokioMutex<Session>>,
    schema: Value,
}

impl CreateTaskTool {
    pub fn new(session: Arc<TokioMutex<Session>>) -> Self {
        Self {
            session,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "description": {"type": "string", "description": "Optional task description"}
                },
                "required": ["title"]
            }),
        }
    }
}

#[async_trait]
impl Tool for CreateTaskTool {
    fn name(&self) -> &str {
        "create_task"
    }

    fn description(&self) -> &str {
        "Create a new task in the current session. Returns the new task ID."
    }

    fn input_schema(&self) -> &Value {
        &self.schema
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let title = input
            .get("title")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ToolError::InvalidInput {
                message: "title is required and must not be empty".to_string(),
            })?;

        let description = input.get("description").and_then(|v| v.as_str());

        let session = self.session.lock().await;
        let id = session
            .tasks()
            .create(title, description)
            .await
            .map_err(|e| ToolError::Execution {
                tool_name: "create_task".to_string(),
                message: e.to_string(),
            })?;

        Ok(ToolResult {
            content: vec![ContentBlock::Text(format!("Created task with id {}", id))],
            is_error: false,
            agent_events: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn test_session_arc() -> Arc<TokioMutex<Session>> {
        let dir = TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.keep()).await.expect("session");
        Arc::new(TokioMutex::new(session))
    }

    #[tokio::test]
    async fn create_task_inserts_row_and_returns_id() {
        let session = test_session_arc().await;
        let tool = CreateTaskTool::new(Arc::clone(&session));
        let result = tool
            .execute(serde_json::json!({"title": "My task"}))
            .await
            .expect("create");
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ContentBlock::Text(t) => t.clone(),
            _ => panic!("expected text"),
        };
        assert!(text.contains('1'), "first task id should be 1");
    }

    #[tokio::test]
    async fn create_task_with_description() {
        let session = test_session_arc().await;
        let tool = CreateTaskTool::new(Arc::clone(&session));
        tool.execute(serde_json::json!({"title": "t", "description": "desc"}))
            .await
            .expect("create");

        let tasks = session.lock().await.tasks().list(None).await.expect("list");
        assert_eq!(tasks[0].description.as_deref(), Some("desc"));
    }

    #[tokio::test]
    async fn create_task_without_title_errors() {
        let session = test_session_arc().await;
        let tool = CreateTaskTool::new(session);
        let err = tool
            .execute(serde_json::json!({}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }
}
