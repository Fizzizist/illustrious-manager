use std::str::FromStr;
use std::time::SystemTime;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use turso::Value as DbValue;

use super::Session;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    #[default]
    Pending,
    InProgress,
    Completed,
}

impl TaskStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
        }
    }
}

impl FromStr for TaskStatus {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "pending" => Ok(Self::Pending),
            "in_progress" => Ok(Self::InProgress),
            "completed" => Ok(Self::Completed),
            other => bail!(
                "Invalid status '{}'. Must be one of: pending, in_progress, completed",
                other
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecord {
    pub id: i64,
    pub title: String,
    pub description: Option<String>,
    pub status: TaskStatus,
    pub created_at: i64,
    pub updated_at: i64,
}

fn now_epoch_secs() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system time is before UNIX epoch")
        .as_secs() as i64
}

pub struct TaskRepo<'a> {
    session: &'a Session,
}

impl<'a> TaskRepo<'a> {
    pub(super) fn new(session: &'a Session) -> Self {
        Self { session }
    }

    pub async fn create(&self, title: &str, description: Option<&str>) -> Result<i64> {
        let now = now_epoch_secs();
        let mut rows = self
            .session
            .conn
            .query(
                "INSERT INTO task (title, description, status, created_at, updated_at) \
                 VALUES (?1, ?2, 'pending', ?3, ?4) RETURNING id",
                [
                    DbValue::Text(title.to_string()),
                    description
                        .map(|s| DbValue::Text(s.to_string()))
                        .unwrap_or(DbValue::Null),
                    DbValue::Integer(now),
                    DbValue::Integer(now),
                ],
            )
            .await
            .context("Failed to insert task")?;

        let row = rows
            .next()
            .await?
            .context("INSERT RETURNING returned no rows")?;

        match row.get_value(0)? {
            DbValue::Integer(id) => Ok(id),
            other => bail!("Unexpected id type from INSERT RETURNING: {:?}", other),
        }
    }

    pub async fn list(&self, status_filter: Option<TaskStatus>) -> Result<Vec<TaskRecord>> {
        let mut rows = if let Some(status) = status_filter {
            self.session
                .conn
                .query(
                    "SELECT id, title, description, status, created_at, updated_at \
                     FROM task WHERE status = ?1 ORDER BY id ASC",
                    [DbValue::Text(status.as_str().to_string())],
                )
                .await
                .context("Failed to list tasks by status")?
        } else {
            self.session
                .conn
                .query(
                    "SELECT id, title, description, status, created_at, updated_at \
                     FROM task ORDER BY id ASC",
                    (),
                )
                .await
                .context("Failed to list tasks")?
        };

        let mut tasks = Vec::new();
        while let Some(row) = rows.next().await? {
            let id = match row.get_value(0)? {
                DbValue::Integer(n) => n,
                other => bail!("Unexpected id type: {:?}", other),
            };
            let title = match row.get_value(1)? {
                DbValue::Text(s) => s,
                other => bail!("Unexpected title type: {:?}", other),
            };
            let description = match row.get_value(2)? {
                DbValue::Text(s) => Some(s),
                DbValue::Null => None,
                other => bail!("Unexpected description type: {:?}", other),
            };
            let status_str = match row.get_value(3)? {
                DbValue::Text(s) => s,
                other => bail!("Unexpected status type: {:?}", other),
            };
            let status = TaskStatus::from_str(&status_str)?;
            let created_at = match row.get_value(4)? {
                DbValue::Integer(n) => n,
                other => bail!("Unexpected created_at type: {:?}", other),
            };
            let updated_at = match row.get_value(5)? {
                DbValue::Integer(n) => n,
                other => bail!("Unexpected updated_at type: {:?}", other),
            };
            tasks.push(TaskRecord {
                id,
                title,
                description,
                status,
                created_at,
                updated_at,
            });
        }

        Ok(tasks)
    }

    pub async fn update(
        &self,
        id: i64,
        title: Option<&str>,
        description: Option<&str>,
        status: Option<TaskStatus>,
    ) -> Result<bool> {
        let now = now_epoch_secs();
        let mut set_parts: Vec<String> = Vec::new();
        let mut params: Vec<DbValue> = Vec::new();
        let mut idx = 1usize;

        if let Some(t) = title {
            set_parts.push(format!("title = ?{}", idx));
            params.push(DbValue::Text(t.to_string()));
            idx += 1;
        }
        if let Some(d) = description {
            set_parts.push(format!("description = ?{}", idx));
            params.push(DbValue::Text(d.to_string()));
            idx += 1;
        }
        if let Some(s) = status {
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

        let rows_affected = self
            .session
            .conn
            .execute(&sql, params)
            .await
            .context("Failed to update task")?;

        Ok(rows_affected > 0)
    }

    pub async fn delete(&self, id: i64) -> Result<bool> {
        let rows_affected = self
            .session
            .conn
            .execute("DELETE FROM task WHERE id = ?1", [DbValue::Integer(id)])
            .await
            .context("Failed to delete task")?;

        Ok(rows_affected > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Session;
    use tempfile::TempDir;

    async fn new_session() -> Session {
        let dir = TempDir::new().expect("temp dir");
        Session::new(None, dir.keep()).await.expect("session")
    }

    #[test]
    fn task_status_round_trips_through_string() {
        for (variant, s) in [
            (TaskStatus::Pending, "pending"),
            (TaskStatus::InProgress, "in_progress"),
            (TaskStatus::Completed, "completed"),
        ] {
            assert_eq!(variant.as_str(), s);
            assert_eq!(TaskStatus::from_str(s).expect("parse"), variant);
        }
    }

    #[test]
    fn task_status_rejects_invalid_string() {
        assert!(TaskStatus::from_str("done").is_err());
    }

    #[test]
    fn task_status_default_is_pending() {
        assert_eq!(TaskStatus::default(), TaskStatus::Pending);
    }

    #[tokio::test]
    async fn create_returns_id() {
        let session = new_session().await;
        let id = session
            .tasks()
            .create("my task", None)
            .await
            .expect("create");
        assert_eq!(id, 1);
    }

    #[tokio::test]
    async fn create_with_description_persists() {
        let session = new_session().await;
        session
            .tasks()
            .create("t", Some("desc"))
            .await
            .expect("create");
        let tasks = session.tasks().list(None).await.expect("list");
        assert_eq!(tasks[0].description.as_deref(), Some("desc"));
    }

    #[tokio::test]
    async fn list_returns_empty_when_none() {
        let session = new_session().await;
        let tasks = session.tasks().list(None).await.expect("list");
        assert!(tasks.is_empty());
    }

    #[tokio::test]
    async fn list_returns_all_in_id_order() {
        let session = new_session().await;
        let repo = session.tasks();
        repo.create("a", None).await.expect("c1");
        repo.create("b", None).await.expect("c2");
        repo.create("c", None).await.expect("c3");

        let tasks = repo.list(None).await.expect("list");
        assert_eq!(tasks.len(), 3);
        assert_eq!(tasks[0].title, "a");
        assert_eq!(tasks[1].title, "b");
        assert_eq!(tasks[2].title, "c");
    }

    #[tokio::test]
    async fn list_filters_by_status() {
        let session = new_session().await;
        let repo = session.tasks();
        repo.create("pending", None).await.expect("c1");
        repo.create("active", None).await.expect("c2");
        repo.update(2, None, None, Some(TaskStatus::InProgress))
            .await
            .expect("update");

        let tasks = repo.list(Some(TaskStatus::InProgress)).await.expect("list");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].title, "active");
    }

    #[tokio::test]
    async fn update_returns_true_on_success() {
        let session = new_session().await;
        let repo = session.tasks();
        repo.create("task", None).await.expect("create");
        let updated = repo
            .update(1, Some("new title"), None, None)
            .await
            .expect("update");
        assert!(updated);
        let tasks = repo.list(None).await.expect("list");
        assert_eq!(tasks[0].title, "new title");
    }

    #[tokio::test]
    async fn update_returns_false_for_missing_id() {
        let session = new_session().await;
        let updated = session
            .tasks()
            .update(999, Some("x"), None, None)
            .await
            .expect("update");
        assert!(!updated);
    }

    #[tokio::test]
    async fn update_advances_updated_at() {
        let session = new_session().await;
        let repo = session.tasks();
        repo.create("task", None).await.expect("create");
        let before = repo.list(None).await.expect("list")[0].updated_at;
        repo.update(1, Some("new"), None, None)
            .await
            .expect("update");
        let after = repo.list(None).await.expect("list")[0].updated_at;
        assert!(after >= before);
    }

    #[tokio::test]
    async fn delete_returns_true_on_success() {
        let session = new_session().await;
        let repo = session.tasks();
        repo.create("task", None).await.expect("create");
        let deleted = repo.delete(1).await.expect("delete");
        assert!(deleted);
        assert!(repo.list(None).await.expect("list").is_empty());
    }

    #[tokio::test]
    async fn delete_returns_false_for_missing_id() {
        let session = new_session().await;
        let deleted = session.tasks().delete(999).await.expect("delete");
        assert!(!deleted);
    }
}
