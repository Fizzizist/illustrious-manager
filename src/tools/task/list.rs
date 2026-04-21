use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex as TokioMutex;

use crate::session::{Session, TaskStatus};
use crate::tools::{Tool, ToolError, ToolResult};
use crate::types::ContentBlock;

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
        let status_filter = input
            .get("status")
            .and_then(|v| v.as_str())
            .map(|s| {
                TaskStatus::from_str(s).map_err(|e| ToolError::InvalidInput {
                    message: e.to_string(),
                })
            })
            .transpose()?;

        let session = self.session.lock().await;
        let tasks =
            session
                .tasks()
                .list(status_filter)
                .await
                .map_err(|e| ToolError::Execution {
                    tool_name: "list_tasks".to_string(),
                    message: e.to_string(),
                })?;

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

    fn titles_from_result(result: &ToolResult) -> Vec<String> {
        let text = match &result.content[0] {
            ContentBlock::Text(t) => t.clone(),
            _ => panic!("expected text"),
        };
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&text).expect("json");
        parsed
            .iter()
            .map(|t| t["title"].as_str().expect("title").to_string())
            .collect()
    }

    #[tokio::test]
    async fn list_tasks_returns_empty_when_none() {
        let session = test_session_arc().await;
        let result = ListTasksTool::new(session)
            .execute(serde_json::json!({}))
            .await
            .expect("list");
        assert!(titles_from_result(&result).is_empty());
    }

    #[tokio::test]
    async fn list_tasks_returns_all_in_id_order() {
        let session = test_session_arc().await;
        for title in ["a", "b", "c"] {
            CreateTaskTool::new(Arc::clone(&session))
                .execute(serde_json::json!({"title": title}))
                .await
                .expect("create");
        }
        let result = ListTasksTool::new(Arc::clone(&session))
            .execute(serde_json::json!({}))
            .await
            .expect("list");
        assert_eq!(titles_from_result(&result), vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn list_tasks_filters_by_status() {
        let session = test_session_arc().await;
        CreateTaskTool::new(Arc::clone(&session))
            .execute(serde_json::json!({"title": "pending-task"}))
            .await
            .expect("c1");
        CreateTaskTool::new(Arc::clone(&session))
            .execute(serde_json::json!({"title": "active"}))
            .await
            .expect("c2");

        session
            .lock()
            .await
            .tasks()
            .update(2, None, None, Some(TaskStatus::InProgress))
            .await
            .expect("update");

        let result = ListTasksTool::new(Arc::clone(&session))
            .execute(serde_json::json!({"status": "in_progress"}))
            .await
            .expect("list");
        assert_eq!(titles_from_result(&result), vec!["active"]);
    }

    #[tokio::test]
    async fn list_tasks_invalid_status_filter_errors() {
        let session = test_session_arc().await;
        let err = ListTasksTool::new(session)
            .execute(serde_json::json!({"status": "bogus"}))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }
}
