use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex as TokioMutex;

use crate::session::Session;
use crate::tools::{Tool, ToolError, ToolResult};
use crate::types::ContentBlock;

pub struct DeleteTaskTool {
    session: Arc<TokioMutex<Session>>,
    schema: Value,
}

impl DeleteTaskTool {
    pub fn new(session: Arc<TokioMutex<Session>>) -> Self {
        Self {
            session,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Task ID to delete"}
                },
                "required": ["id"]
            }),
        }
    }
}

#[async_trait]
impl Tool for DeleteTaskTool {
    fn name(&self) -> &str {
        "delete_task"
    }

    fn description(&self) -> &str {
        "Delete a task by ID from the current session."
    }

    fn input_schema(&self) -> &Value {
        &self.schema
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let id =
            input
                .get("id")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| ToolError::InvalidInput {
                    message: "id is required and must be an integer".to_string(),
                })?;

        let session = self.session.lock().await;
        let deleted = session
            .tasks()
            .delete(id)
            .await
            .map_err(|e| ToolError::Execution {
                tool_name: "delete_task".to_string(),
                message: e.to_string(),
            })?;

        if !deleted {
            return Err(ToolError::Execution {
                tool_name: "delete_task".to_string(),
                message: format!("Task with id {} does not exist", id),
            });
        }

        Ok(ToolResult {
            content: vec![ContentBlock::Text(format!("Deleted task {}", id))],
            is_error: false,
            agent_events: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::task::create::CreateTaskTool;
    use tempfile::TempDir;

    async fn test_session_arc() -> Arc<TokioMutex<Session>> {
        let dir = TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.keep()).await.expect("session");
        Arc::new(TokioMutex::new(session))
    }

    #[tokio::test]
    async fn delete_task_removes_row() {
        let session = test_session_arc().await;
        CreateTaskTool::new(Arc::clone(&session))
            .execute(serde_json::json!({"title": "to delete"}))
            .await
            .expect("create");

        DeleteTaskTool::new(Arc::clone(&session))
            .execute(serde_json::json!({"id": 1}))
            .await
            .expect("delete");

        let tasks = session.lock().await.tasks().list(None).await.expect("list");
        assert!(tasks.is_empty());
    }

    #[tokio::test]
    async fn delete_task_missing_id_errors() {
        let session = test_session_arc().await;
        let err = DeleteTaskTool::new(session)
            .execute(serde_json::json!({"id": 999}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::Execution { .. }));
    }
}
