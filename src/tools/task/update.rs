use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex as TokioMutex;

use crate::session::{Session, TaskStatus};
use crate::tools::{Tool, ToolError, ToolResult};
use crate::types::ContentBlock;

pub struct UpdateTaskTool {
    session: Arc<TokioMutex<Session>>,
    schema: Value,
}

impl UpdateTaskTool {
    pub fn new(session: Arc<TokioMutex<Session>>) -> Self {
        Self {
            session,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Task ID to update"},
                    "title": {"type": "string", "description": "New title"},
                    "description": {"type": "string", "description": "New description"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "New status"
                    }
                },
                "required": ["id"]
            }),
        }
    }
}

#[async_trait]
impl Tool for UpdateTaskTool {
    fn name(&self) -> &str {
        "update_task"
    }

    fn description(&self) -> &str {
        "Update a task's title, description, or status. At least one of title, description, or status must be provided."
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

        let new_title = input.get("title").and_then(|v| v.as_str());
        let new_description = input.get("description").and_then(|v| v.as_str());
        let new_status = input
            .get("status")
            .and_then(|v| v.as_str())
            .map(|s| {
                TaskStatus::from_str(s).map_err(|e| ToolError::InvalidInput {
                    message: e.to_string(),
                })
            })
            .transpose()?;

        if new_title.is_none() && new_description.is_none() && new_status.is_none() {
            return Err(ToolError::InvalidInput {
                message: "At least one of title, description, or status must be provided"
                    .to_string(),
            });
        }

        if let Some(t) = new_title
            && t.is_empty()
        {
            return Err(ToolError::InvalidInput {
                message: "title must not be empty".to_string(),
            });
        }

        let session = self.session.lock().await;
        let updated = session
            .tasks()
            .update(id, new_title, new_description, new_status)
            .await
            .map_err(|e| ToolError::Execution {
                tool_name: "update_task".to_string(),
                message: e.to_string(),
            })?;

        if !updated {
            return Err(ToolError::Execution {
                tool_name: "update_task".to_string(),
                message: format!("Task with id {} does not exist", id),
            });
        }

        Ok(ToolResult {
            content: vec![ContentBlock::Text(format!("Updated task {}", id))],
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

    async fn create_task(session: &Arc<TokioMutex<Session>>, title: &str) {
        CreateTaskTool::new(Arc::clone(session))
            .execute(serde_json::json!({"title": title}))
            .await
            .expect("create");
    }

    #[tokio::test]
    async fn update_task_partial_title_only() {
        let session = test_session_arc().await;
        create_task(&session, "original").await;

        let before = session.lock().await.tasks().list(None).await.expect("list")[0].updated_at;

        UpdateTaskTool::new(Arc::clone(&session))
            .execute(serde_json::json!({"id": 1, "title": "updated"}))
            .await
            .expect("update");

        let tasks = session.lock().await.tasks().list(None).await.expect("list");
        assert_eq!(tasks[0].title, "updated");
        assert_eq!(tasks[0].status, TaskStatus::Pending);
        assert!(
            tasks[0].updated_at >= before,
            "updated_at should not regress"
        );
    }

    #[tokio::test]
    async fn update_task_status_transition() {
        let session = test_session_arc().await;
        create_task(&session, "task").await;

        let tool = UpdateTaskTool::new(Arc::clone(&session));
        tool.execute(serde_json::json!({"id": 1, "status": "in_progress"}))
            .await
            .expect("u1");
        tool.execute(serde_json::json!({"id": 1, "status": "completed"}))
            .await
            .expect("u2");

        let tasks = session.lock().await.tasks().list(None).await.expect("list");
        assert_eq!(tasks[0].status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn update_task_invalid_status_errors() {
        let session = test_session_arc().await;
        create_task(&session, "task").await;
        let err = UpdateTaskTool::new(Arc::clone(&session))
            .execute(serde_json::json!({"id": 1, "status": "bogus"}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn update_task_with_no_mutable_fields_errors() {
        let session = test_session_arc().await;
        create_task(&session, "task").await;
        let err = UpdateTaskTool::new(Arc::clone(&session))
            .execute(serde_json::json!({"id": 1}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn update_task_empty_title_errors() {
        let session = test_session_arc().await;
        create_task(&session, "task").await;
        let err = UpdateTaskTool::new(Arc::clone(&session))
            .execute(serde_json::json!({"id": 1, "title": ""}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn update_task_missing_id_errors() {
        let session = test_session_arc().await;
        let err = UpdateTaskTool::new(Arc::clone(&session))
            .execute(serde_json::json!({"id": 999, "title": "x"}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::Execution { .. }));
    }
}
