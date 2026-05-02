use anyhow::{Context, Result, bail};

use crate::types::{ContentBlock, Message, Role};

use super::Session;

pub struct ConversationRepo<'a> {
    session: &'a Session,
}

impl<'a> ConversationRepo<'a> {
    pub(super) fn new(session: &'a Session) -> Self {
        Self { session }
    }

    pub async fn insert_message(&self, message: &Message) -> Result<()> {
        let content_json = serde_json::to_string(&message.content)
            .context("Failed to serialize message content")?;
        let role_str = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        self.session
            .conn
            .execute(
                "INSERT INTO conversation (role, content) VALUES (?1, ?2)",
                [
                    turso::Value::Text(role_str.to_string()),
                    turso::Value::Text(content_json),
                ],
            )
            .await
            .context("Failed to insert message into session")?;
        Ok(())
    }

    pub async fn is_empty(&self) -> Result<bool> {
        let mut rows = self
            .session
            .conn
            .query("SELECT COUNT(*) FROM conversation WHERE active = 1", ())
            .await
            .context("Failed to count conversation messages")?;

        if let Some(row) = rows.next().await? {
            let count = match row.get_value(0)? {
                turso::Value::Integer(n) => n,
                _ => return Ok(true),
            };
            return Ok(count == 0);
        }

        Ok(true)
    }

    pub async fn load_history(&self) -> Result<Vec<Message>> {
        let mut rows = self
            .session
            .conn
            .query(
                "SELECT role, content FROM conversation WHERE active = 1 ORDER BY id ASC",
                (),
            )
            .await
            .context("Failed to query conversation history")?;

        let mut messages = Vec::new();
        while let Some(row) = rows.next().await? {
            let role_str = match row.get_value(0)? {
                turso::Value::Text(s) => s,
                other => bail!("Unexpected role type in DB: {:?}", other),
            };
            let content_str = match row.get_value(1)? {
                turso::Value::Text(s) => s,
                other => bail!("Unexpected content type in DB: {:?}", other),
            };

            let role = match role_str.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                other => bail!("Unknown role in DB: {}", other),
            };

            let content: Vec<ContentBlock> =
                serde_json::from_str(&content_str).context("Failed to deserialize content")?;

            messages.push(Message { role, content });
        }

        Ok(messages)
    }

    pub async fn load_active_ids(&self) -> Result<Vec<i64>> {
        let mut rows = self
            .session
            .conn
            .query(
                "SELECT id FROM conversation WHERE active = 1 ORDER BY id ASC",
                (),
            )
            .await
            .context("Failed to query active conversation IDs")?;

        let mut ids = Vec::new();
        while let Some(row) = rows.next().await? {
            if let turso::Value::Integer(n) = row.get_value(0)? {
                ids.push(n);
            }
        }

        Ok(ids)
    }

    pub async fn deactivate_entries(&self, ids: &[i64]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "UPDATE conversation SET active = 0 WHERE id IN ({})",
            placeholders.join(",")
        );
        let params: Vec<turso::Value> = ids.iter().map(|id| turso::Value::Integer(*id)).collect();
        self.session
            .conn
            .execute(&sql, params)
            .await
            .context("Failed to deactivate conversation entries")?;
        Ok(())
    }

    pub async fn read_first_user_message(&self) -> Result<String> {
        let mut rows = self
            .session
            .conn
            .query(
                "SELECT content FROM conversation WHERE role = 'user' AND active = 1 ORDER BY id ASC LIMIT 1",
                (),
            )
            .await
            .context("Failed to query first user message")?;

        if let Some(row) = rows.next().await? {
            let content_str = match row.get_value(0)? {
                turso::Value::Text(s) => s,
                _ => return Ok(String::new()),
            };
            let blocks: Vec<ContentBlock> = serde_json::from_str(&content_str)
                .context("Failed to deserialize content blocks")?;
            for block in blocks {
                if let ContentBlock::Text(text) = block {
                    return Ok(text);
                }
            }
        }

        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentBlock, Message, Role};
    use tempfile::TempDir;

    async fn test_session() -> Session {
        let dir = TempDir::new().expect("temp dir");
        Session::new(None, dir.keep()).await.expect("test session")
    }

    #[tokio::test]
    async fn load_history_returns_only_active_entries() {
        let session = test_session().await;
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "first".to_string()))
            .await
            .expect("insert");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "second".to_string()))
            .await
            .expect("insert");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "third".to_string()))
            .await
            .expect("insert");

        let ids = session.conversation().load_active_ids().await.expect("ids");
        assert_eq!(ids.len(), 3);

        session
            .conversation()
            .deactivate_entries(&ids[..2])
            .await
            .expect("deactivate");

        let history = session.conversation().load_history().await.expect("load");
        assert_eq!(history.len(), 1);
        match &history[0].content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "third"),
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn deactivate_entries_marks_rows_inactive() {
        let session = test_session().await;
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "a".to_string()))
            .await
            .expect("insert");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "b".to_string()))
            .await
            .expect("insert");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "c".to_string()))
            .await
            .expect("insert");

        let ids = session.conversation().load_active_ids().await.expect("ids");
        assert_eq!(ids.len(), 3);

        session
            .conversation()
            .deactivate_entries(&ids[..2])
            .await
            .expect("deactivate");

        let active_ids = session.conversation().load_active_ids().await.expect("ids");
        assert_eq!(active_ids.len(), 1);
        assert_eq!(active_ids[0], ids[2]);
    }

    #[tokio::test]
    async fn deactivate_entries_with_empty_ids_is_noop() {
        let session = test_session().await;
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "msg".to_string()))
            .await
            .expect("insert");

        let result = session.conversation().deactivate_entries(&[]).await;
        assert!(result.is_ok());

        let history = session.conversation().load_history().await.expect("load");
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn is_empty_ignores_inactive_entries() {
        let session = test_session().await;
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "msg".to_string()))
            .await
            .expect("insert");

        assert!(!session.conversation().is_empty().await.expect("is_empty"));

        let ids = session.conversation().load_active_ids().await.expect("ids");
        session
            .conversation()
            .deactivate_entries(&ids)
            .await
            .expect("deactivate");

        assert!(session.conversation().is_empty().await.expect("is_empty"));
    }

    #[tokio::test]
    async fn read_first_user_message_ignores_inactive_entries() {
        let session = test_session().await;
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "first".to_string()))
            .await
            .expect("insert");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "second".to_string()))
            .await
            .expect("insert");

        let ids = session.conversation().load_active_ids().await.expect("ids");
        session
            .conversation()
            .deactivate_entries(&ids[..1])
            .await
            .expect("deactivate");

        let first = session
            .conversation()
            .read_first_user_message()
            .await
            .expect("read");
        assert_eq!(first, "second");
    }

    #[tokio::test]
    async fn schema_migration_adds_active_column_idempotently() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.keep();

        // Create a session with the current schema (which includes active column)
        let session = Session::new(None, path.clone()).await.expect("session");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "test".to_string()))
            .await
            .expect("insert");

        // Reopening the session should succeed (migration is idempotent)
        let session2 = Session::new(Some(session.id.clone()), path)
            .await
            .expect("reopen");
        let history = session2.conversation().load_history().await.expect("load");
        assert_eq!(history.len(), 1);
    }
}
