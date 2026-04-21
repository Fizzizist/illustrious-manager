use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex as TokioMutex;
use turso::Value as DbValue;

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

        // Check existence first
        let mut rows = session
            .conn
            .query(
                "SELECT COUNT(*) FROM task WHERE id = ?1",
                [DbValue::Integer(id)],
            )
            .await
            .map_err(|e| ToolError::Execution {
                tool_name: "delete_task".to_string(),
                message: e.to_string(),
            })?;

        let count = if let Some(row) = rows.next().await.map_err(|e| ToolError::Execution {
            tool_name: "delete_task".to_string(),
            message: e.to_string(),
        })? {
            match row.get_value(0).map_err(|e| ToolError::Execution {
                tool_name: "delete_task".to_string(),
                message: e.to_string(),
            })? {
                DbValue::Integer(n) => n,
                _ => 0,
            }
        } else {
            0
        };

        if count == 0 {
            return Err(ToolError::Execution {
                tool_name: "delete_task".to_string(),
                message: format!("Task with id {} does not exist", id),
            });
        }

        session
            .conn
            .execute("DELETE FROM task WHERE id = ?1", [DbValue::Integer(id)])
            .await
            .map_err(|e| ToolError::Execution {
                tool_name: "delete_task".to_string(),
                message: e.to_string(),
            })?;

        Ok(ToolResult {
            content: vec![ContentBlock::Text(format!("Deleted task {}", id))],
            is_error: false,
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
        let create = CreateTaskTool::new(Arc::clone(&session));
        create
            .execute(serde_json::json!({"title": "to delete"}))
            .await
            .expect("create");

        let delete = DeleteTaskTool::new(Arc::clone(&session));
        delete
            .execute(serde_json::json!({"id": 1}))
            .await
            .expect("delete");

        let sess = session.lock().await;
        let mut rows = sess
            .conn
            .query("SELECT COUNT(*) FROM task", ())
            .await
            .expect("q");
        let row = rows.next().await.expect("n").expect("r");
        let count = match row.get_value(0).expect("v") {
            DbValue::Integer(n) => n,
            _ => panic!(),
        };
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn delete_task_missing_id_errors() {
        let session = test_session_arc().await;
        let tool = DeleteTaskTool::new(session);
        let err = tool
            .execute(serde_json::json!({"id": 999}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::Execution { .. }));
    }

    #[tokio::test]
    async fn is_write_tool_returns_false() {
        let session = test_session_arc().await;
        let tool = DeleteTaskTool::new(session);
        assert!(!tool.is_write_tool());
    }
}
