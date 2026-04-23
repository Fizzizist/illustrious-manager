pub mod create;
pub mod delete;
pub mod list;
pub mod update;

pub use create::CreateTaskTool;
pub use delete::DeleteTaskTool;
pub use list::ListTasksTool;
pub use update::UpdateTaskTool;

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::sync::Mutex as TokioMutex;

    use crate::session::Session;
    use crate::tools::Tool;
    use crate::types::ContentBlock;

    use super::*;

    async fn test_session_arc_in(dir: &std::path::Path) -> Arc<TokioMutex<Session>> {
        Arc::new(TokioMutex::new(
            Session::new(None, dir.to_path_buf())
                .await
                .expect("session"),
        ))
    }

    async fn test_session_arc() -> Arc<TokioMutex<Session>> {
        let dir = TempDir::new().expect("temp dir");
        test_session_arc_in(&dir.keep()).await
    }

    async fn create_task(session: &Arc<TokioMutex<Session>>, title: &str) {
        CreateTaskTool::new(Arc::clone(session))
            .execute(serde_json::json!({"title": title}))
            .await
            .expect("create");
    }

    fn task_titles_from_result(result: &crate::tools::ToolResult) -> Vec<String> {
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
    async fn tasks_persist_across_session_reopen() {
        let dir = TempDir::new().expect("temp dir");
        let dir_path = dir.keep();

        let session_id = {
            let session = Session::new(None, dir_path.clone()).await.expect("create");
            let id = session.id.clone();
            session
                .tasks()
                .create("persisted task", None)
                .await
                .expect("create");
            id
        };

        let session2 = Session::new(Some(session_id), dir_path)
            .await
            .expect("reopen");
        let tasks = session2.tasks().list(None).await.expect("list");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].title, "persisted task");
    }

    #[tokio::test]
    async fn tasks_are_scoped_per_session() {
        let dir = TempDir::new().expect("temp dir");
        let dir_path = dir.keep();

        let session_a = test_session_arc_in(&dir_path).await;
        create_task(&session_a, "session a task").await;

        let session_b = test_session_arc_in(&dir_path).await;
        let tasks = session_b
            .lock()
            .await
            .tasks()
            .list(None)
            .await
            .expect("list");
        assert!(tasks.is_empty(), "session B should have no tasks");
    }

    #[tokio::test]
    async fn session_swap_routes_subsequent_calls_to_new_session_db() {
        let dir = TempDir::new().expect("temp dir");
        let dir_path = dir.keep();

        let session_a = Session::new(None, dir_path.clone()).await.expect("a");
        let session_b = Session::new(None, dir_path.clone()).await.expect("b");

        let arc = Arc::new(TokioMutex::new(session_a));
        create_task(&arc, "task in A").await;

        *arc.lock().await = session_b;

        let result = ListTasksTool::new(Arc::clone(&arc))
            .execute(serde_json::json!({}))
            .await
            .expect("list after swap");
        assert!(
            task_titles_from_result(&result).is_empty(),
            "after swap to B, list should be empty"
        );

        create_task(&arc, "task in B").await;
        let result2 = ListTasksTool::new(Arc::clone(&arc))
            .execute(serde_json::json!({}))
            .await
            .expect("list after create in B");
        assert_eq!(task_titles_from_result(&result2), vec!["task in B"]);
    }

    #[tokio::test]
    async fn is_write_tool_returns_false_for_all_four() {
        let session = test_session_arc().await;
        assert!(!CreateTaskTool::new(Arc::clone(&session)).is_write_tool());
        assert!(!ListTasksTool::new(Arc::clone(&session)).is_write_tool());
        assert!(!UpdateTaskTool::new(Arc::clone(&session)).is_write_tool());
        assert!(!DeleteTaskTool::new(Arc::clone(&session)).is_write_tool());
    }
}
