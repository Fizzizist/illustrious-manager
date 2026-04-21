use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex as TokioMutex;
use turso::Value as DbValue;

use crate::session::Session;
use crate::tools::{Tool, ToolError, ToolResult};
use crate::types::ContentBlock;

fn now_epoch_secs() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system time is before UNIX epoch")
        .as_secs() as i64
}

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
                message: "title is required".to_string(),
            })?
            .to_string();

        let description = input
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let now = now_epoch_secs();
        let session = self.session.lock().await;

        let mut rows = session
            .conn
            .query(
                "INSERT INTO task (title, description, status, created_at, updated_at) VALUES (?1, ?2, 'pending', ?3, ?4) RETURNING id",
                [
                    DbValue::Text(title),
                    description.map(DbValue::Text).unwrap_or(DbValue::Null),
                    DbValue::Integer(now),
                    DbValue::Integer(now),
                ],
            )
            .await
            .map_err(|e| ToolError::Execution {
                tool_name: "create_task".to_string(),
                message: e.to_string(),
            })?;

        let id = if let Some(row) = rows.next().await.map_err(|e| ToolError::Execution {
            tool_name: "create_task".to_string(),
            message: e.to_string(),
        })? {
            match row.get_value(0).map_err(|e| ToolError::Execution {
                tool_name: "create_task".to_string(),
                message: e.to_string(),
            })? {
                DbValue::Integer(id) => id,
                other => {
                    return Err(ToolError::Execution {
                        tool_name: "create_task".to_string(),
                        message: format!("Unexpected id type: {:?}", other),
                    });
                }
            }
        } else {
            return Err(ToolError::Execution {
                tool_name: "create_task".to_string(),
                message: "INSERT returned no rows".to_string(),
            });
        };

        Ok(ToolResult {
            content: vec![ContentBlock::Text(format!("Created task with id {}", id))],
            is_error: false,
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
        let result = tool
            .execute(serde_json::json!({"title": "t", "description": "desc"}))
            .await
            .expect("create");
        assert!(!result.is_error);

        let sess = session.lock().await;
        let mut rows = sess
            .conn
            .query("SELECT description FROM task WHERE id = 1", ())
            .await
            .expect("query");
        let row = rows.next().await.expect("next").expect("row");
        match row.get_value(0).expect("val") {
            DbValue::Text(d) => assert_eq!(d, "desc"),
            other => panic!("unexpected: {:?}", other),
        }
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

    #[tokio::test]
    async fn is_write_tool_returns_false() {
        let session = test_session_arc().await;
        let tool = CreateTaskTool::new(session);
        assert!(!tool.is_write_tool());
    }
}
