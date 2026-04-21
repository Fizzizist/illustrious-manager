use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex as TokioMutex;
use turso::Value as DbValue;

use crate::session::Session;
use crate::tools::{Tool, ToolError, ToolResult};
use crate::types::ContentBlock;

use super::status::TaskStatus;

pub struct ListTasksTool {
    session: Arc<TokioMutex<Session>>,
    schema: Value,
}

impl ListTasksTool {
    pub fn new(session: Arc<TokioMutex<Session>>) -> Self {
        Self {
            session,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "Optional status filter"
                    }
                }
            }),
        }
    }
}

#[async_trait]
impl Tool for ListTasksTool {
    fn name(&self) -> &str {
        "list_tasks"
    }

    fn description(&self) -> &str {
        "List tasks in the current session, ordered by id. Optionally filter by status."
    }

    fn input_schema(&self) -> &Value {
        &self.schema
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let status_filter = if let Some(s) = input.get("status").and_then(|v| v.as_str()) {
            Some(TaskStatus::from_str(s)?)
        } else {
            None
        };

        let session = self.session.lock().await;

        let mut rows = if let Some(status) = status_filter {
            session
                .conn
                .query(
                    "SELECT id, title, description, status, created_at, updated_at FROM task WHERE status = ?1 ORDER BY id ASC",
                    [DbValue::Text(status.as_str().to_string())],
                )
                .await
                .map_err(|e| ToolError::Execution {
                    tool_name: "list_tasks".to_string(),
                    message: e.to_string(),
                })?
        } else {
            session
                .conn
                .query(
                    "SELECT id, title, description, status, created_at, updated_at FROM task ORDER BY id ASC",
                    (),
                )
                .await
                .map_err(|e| ToolError::Execution {
                    tool_name: "list_tasks".to_string(),
                    message: e.to_string(),
                })?
        };

        let mut tasks: Vec<serde_json::Value> = Vec::new();
        while let Some(row) = rows.next().await.map_err(|e| ToolError::Execution {
            tool_name: "list_tasks".to_string(),
            message: e.to_string(),
        })? {
            let id = match row.get_value(0).map_err(|e| ToolError::Execution {
                tool_name: "list_tasks".to_string(),
                message: e.to_string(),
            })? {
                DbValue::Integer(n) => n,
                other => {
                    return Err(ToolError::Execution {
                        tool_name: "list_tasks".to_string(),
                        message: format!("unexpected id type: {:?}", other),
                    });
                }
            };
            let title = match row.get_value(1).map_err(|e| ToolError::Execution {
                tool_name: "list_tasks".to_string(),
                message: e.to_string(),
            })? {
                DbValue::Text(s) => s,
                other => {
                    return Err(ToolError::Execution {
                        tool_name: "list_tasks".to_string(),
                        message: format!("unexpected title type: {:?}", other),
                    });
                }
            };
            let description = match row.get_value(2).map_err(|e| ToolError::Execution {
                tool_name: "list_tasks".to_string(),
                message: e.to_string(),
            })? {
                DbValue::Text(s) => Some(s),
                DbValue::Null => None,
                other => {
                    return Err(ToolError::Execution {
                        tool_name: "list_tasks".to_string(),
                        message: format!("unexpected description type: {:?}", other),
                    });
                }
            };
            let status = match row.get_value(3).map_err(|e| ToolError::Execution {
                tool_name: "list_tasks".to_string(),
                message: e.to_string(),
            })? {
                DbValue::Text(s) => s,
                other => {
                    return Err(ToolError::Execution {
                        tool_name: "list_tasks".to_string(),
                        message: format!("unexpected status type: {:?}", other),
                    });
                }
            };
            let created_at = match row.get_value(4).map_err(|e| ToolError::Execution {
                tool_name: "list_tasks".to_string(),
                message: e.to_string(),
            })? {
                DbValue::Integer(n) => n,
                other => {
                    return Err(ToolError::Execution {
                        tool_name: "list_tasks".to_string(),
                        message: format!("unexpected created_at type: {:?}", other),
                    });
                }
            };
            let updated_at = match row.get_value(5).map_err(|e| ToolError::Execution {
                tool_name: "list_tasks".to_string(),
                message: e.to_string(),
            })? {
                DbValue::Integer(n) => n,
                other => {
                    return Err(ToolError::Execution {
                        tool_name: "list_tasks".to_string(),
                        message: format!("unexpected updated_at type: {:?}", other),
                    });
                }
            };

            tasks.push(serde_json::json!({
                "id": id,
                "title": title,
                "description": description,
                "status": status,
                "created_at": created_at,
                "updated_at": updated_at,
            }));
        }

        let output = serde_json::to_string_pretty(&tasks).map_err(|e| ToolError::Execution {
            tool_name: "list_tasks".to_string(),
            message: e.to_string(),
        })?;

        Ok(ToolResult {
            content: vec![ContentBlock::Text(output)],
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
    async fn list_tasks_returns_empty_when_none() {
        let session = test_session_arc().await;
        let tool = ListTasksTool::new(session);
        let result = tool.execute(serde_json::json!({})).await.expect("list");
        let text = match &result.content[0] {
            ContentBlock::Text(t) => t.clone(),
            _ => panic!("expected text"),
        };
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("json");
        assert_eq!(parsed.as_array().expect("array").len(), 0);
    }

    #[tokio::test]
    async fn list_tasks_returns_all_in_id_order() {
        let session = test_session_arc().await;
        let create = CreateTaskTool::new(Arc::clone(&session));
        create
            .execute(serde_json::json!({"title": "a"}))
            .await
            .expect("c1");
        create
            .execute(serde_json::json!({"title": "b"}))
            .await
            .expect("c2");
        create
            .execute(serde_json::json!({"title": "c"}))
            .await
            .expect("c3");

        let list = ListTasksTool::new(Arc::clone(&session));
        let result = list.execute(serde_json::json!({})).await.expect("list");
        let text = match &result.content[0] {
            ContentBlock::Text(t) => t.clone(),
            _ => panic!("expected text"),
        };
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&text).expect("json");
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0]["title"], "a");
        assert_eq!(parsed[1]["title"], "b");
        assert_eq!(parsed[2]["title"], "c");
    }

    #[tokio::test]
    async fn list_tasks_filters_by_status() {
        let session = test_session_arc().await;
        let create = CreateTaskTool::new(Arc::clone(&session));
        create
            .execute(serde_json::json!({"title": "pending-task"}))
            .await
            .expect("c1");
        create
            .execute(serde_json::json!({"title": "another"}))
            .await
            .expect("c2");

        // Manually update one to in_progress
        {
            let sess = session.lock().await;
            sess.conn
                .execute("UPDATE task SET status = 'in_progress' WHERE id = 2", ())
                .await
                .expect("update");
        }

        let list = ListTasksTool::new(Arc::clone(&session));
        let result = list
            .execute(serde_json::json!({"status": "in_progress"}))
            .await
            .expect("list");
        let text = match &result.content[0] {
            ContentBlock::Text(t) => t.clone(),
            _ => panic!("expected text"),
        };
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&text).expect("json");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["title"], "another");
    }

    #[tokio::test]
    async fn list_tasks_invalid_status_filter_errors() {
        let session = test_session_arc().await;
        let tool = ListTasksTool::new(session);
        let err = tool
            .execute(serde_json::json!({"status": "bogus"}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }
}
