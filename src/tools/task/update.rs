use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex as TokioMutex;
use turso::Value as DbValue;

use crate::session::Session;
use crate::tools::{Tool, ToolError, ToolResult};
use crate::types::ContentBlock;

use super::now_epoch_secs;
use super::status::TaskStatus;

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

        let new_title = input
            .get("title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let new_description = input
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let new_status = if let Some(s) = input.get("status").and_then(|v| v.as_str()) {
            Some(TaskStatus::from_str(s)?)
        } else {
            None
        };

        if new_title.is_none() && new_description.is_none() && new_status.is_none() {
            return Err(ToolError::InvalidInput {
                message: "At least one of title, description, or status must be provided"
                    .to_string(),
            });
        }

        if let Some(ref t) = new_title
            && t.is_empty()
        {
            return Err(ToolError::InvalidInput {
                message: "title must not be empty".to_string(),
            });
        }

        let now = now_epoch_secs();
        let session = self.session.lock().await;

        // Build SET clauses dynamically
        let mut set_parts: Vec<String> = Vec::new();
        let mut params: Vec<DbValue> = Vec::new();
        let mut idx = 1usize;

        if let Some(ref t) = new_title {
            set_parts.push(format!("title = ?{}", idx));
            params.push(DbValue::Text(t.clone()));
            idx += 1;
        }
        if let Some(ref d) = new_description {
            set_parts.push(format!("description = ?{}", idx));
            params.push(DbValue::Text(d.clone()));
            idx += 1;
        }
        if let Some(s) = new_status {
            set_parts.push(format!("status = ?{}", idx));
            params.push(DbValue::Text(s.as_str().to_string()));
            idx += 1;
        }
        set_parts.push(format!("updated_at = ?{}", idx));
        params.push(DbValue::Integer(now));
        idx += 1;

        params.push(DbValue::Integer(id));
        let sql = format!(
            "UPDATE task SET {} WHERE id = ?{}",
            set_parts.join(", "),
            idx
        );

        let rows_affected =
            session
                .conn
                .execute(&sql, params)
                .await
                .map_err(|e| ToolError::Execution {
                    tool_name: "update_task".to_string(),
                    message: e.to_string(),
                })?;

        if rows_affected == 0 {
            return Err(ToolError::Execution {
                tool_name: "update_task".to_string(),
                message: format!("Task with id {} does not exist", id),
            });
        }

        Ok(ToolResult {
            content: vec![ContentBlock::Text(format!("Updated task {}", id))],
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

    async fn create_task(session: &Arc<TokioMutex<Session>>, title: &str) {
        let tool = CreateTaskTool::new(Arc::clone(session));
        tool.execute(serde_json::json!({"title": title}))
            .await
            .expect("create");
    }

    #[tokio::test]
    async fn update_task_partial_title_only() {
        let session = test_session_arc().await;
        create_task(&session, "original").await;

        let before = {
            let sess = session.lock().await;
            let mut rows = sess
                .conn
                .query("SELECT updated_at FROM task WHERE id = 1", ())
                .await
                .expect("q");
            let row = rows.next().await.expect("n").expect("r");
            match row.get_value(0).expect("v") {
                DbValue::Integer(n) => n,
                _ => panic!(),
            }
        };

        let tool = UpdateTaskTool::new(Arc::clone(&session));
        tool.execute(serde_json::json!({"id": 1, "title": "updated"}))
            .await
            .expect("update");

        let sess = session.lock().await;
        let mut rows = sess
            .conn
            .query(
                "SELECT title, status, updated_at FROM task WHERE id = 1",
                (),
            )
            .await
            .expect("query");
        let row = rows.next().await.expect("next").expect("row");
        let title = match row.get_value(0).expect("v") {
            DbValue::Text(s) => s,
            _ => panic!(),
        };
        let status = match row.get_value(1).expect("v") {
            DbValue::Text(s) => s,
            _ => panic!(),
        };
        let updated_at = match row.get_value(2).expect("v") {
            DbValue::Integer(n) => n,
            _ => panic!(),
        };
        assert_eq!(title, "updated");
        assert_eq!(status, "pending");
        assert!(updated_at >= before, "updated_at should not regress");
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

        let sess = session.lock().await;
        let mut rows = sess
            .conn
            .query("SELECT status FROM task WHERE id = 1", ())
            .await
            .expect("q");
        let row = rows.next().await.expect("n").expect("r");
        let status = match row.get_value(0).expect("v") {
            DbValue::Text(s) => s,
            _ => panic!(),
        };
        assert_eq!(status, "completed");
    }

    #[tokio::test]
    async fn update_task_invalid_status_errors() {
        let session = test_session_arc().await;
        create_task(&session, "task").await;
        let tool = UpdateTaskTool::new(Arc::clone(&session));
        let err = tool
            .execute(serde_json::json!({"id": 1, "status": "bogus"}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn update_task_with_no_mutable_fields_errors() {
        let session = test_session_arc().await;
        create_task(&session, "task").await;
        let tool = UpdateTaskTool::new(Arc::clone(&session));
        let err = tool
            .execute(serde_json::json!({"id": 1}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn update_task_empty_title_errors() {
        let session = test_session_arc().await;
        create_task(&session, "task").await;
        let tool = UpdateTaskTool::new(Arc::clone(&session));
        let err = tool
            .execute(serde_json::json!({"id": 1, "title": ""}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn update_task_missing_id_errors() {
        let session = test_session_arc().await;
        let tool = UpdateTaskTool::new(Arc::clone(&session));
        let err = tool
            .execute(serde_json::json!({"id": 999, "title": "x"}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::Execution { .. }));
    }
}
