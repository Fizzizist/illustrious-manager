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
                "INSERT INTO conversation (role, content, active) VALUES (?1, ?2, 1)",
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
            .query("SELECT COUNT(*) FROM conversation", ())
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

    pub async fn read_first_user_message(&self) -> Result<String> {
        let mut rows = self
            .session
            .conn
            .query(
                "SELECT content FROM conversation WHERE role = 'user' ORDER BY id ASC LIMIT 1",
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
    use crate::types::{Message, Role};
    use tempfile::TempDir;

    async fn create_test_session() -> (TempDir, super::super::Session) {
        let dir = TempDir::new().expect("temp dir");
        let session = super::super::Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create session");
        (dir, session)
    }

    #[tokio::test]
    async fn load_history_filters_out_inactive_entries() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "active one".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "inactive".to_string()))
            .await
            .expect("insert 2");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "active two".to_string()))
            .await
            .expect("insert 3");

        session
            .conn
            .execute("UPDATE conversation SET active = 0 WHERE id = 2", ())
            .await
            .expect("deactivate row 2");

        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");
        assert_eq!(history.len(), 2, "should only return active entries");
    }

    #[tokio::test]
    async fn load_history_returns_all_when_all_active() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "a".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "b".to_string()))
            .await
            .expect("insert 2");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "c".to_string()))
            .await
            .expect("insert 3");

        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");
        assert_eq!(history.len(), 3);
    }

    #[tokio::test]
    async fn insert_message_sets_active_to_1() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "check active".to_string()))
            .await
            .expect("insert");

        let mut rows = session
            .conn
            .query(
                "SELECT active FROM conversation ORDER BY id DESC LIMIT 1",
                (),
            )
            .await
            .expect("query active column");
        let row = rows
            .next()
            .await
            .expect("get row")
            .expect("should have a row");
        match row.get_value(0).expect("get active value") {
            turso::Value::Integer(n) => assert_eq!(n, 1, "active column should be 1"),
            other => panic!("expected integer, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn is_empty_counts_inactive_entries() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "will deactivate".to_string()))
            .await
            .expect("insert");

        session
            .conn
            .execute("UPDATE conversation SET active = 0 WHERE id = 1", ())
            .await
            .expect("deactivate");

        let empty = session.conversation().is_empty().await.expect("is_empty");
        assert!(
            !empty,
            "is_empty should return false because the row still exists regardless of active"
        );
    }

    #[tokio::test]
    async fn read_first_user_message_ignores_active_filter() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "first".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "second".to_string()))
            .await
            .expect("insert 2");

        session
            .conn
            .execute("UPDATE conversation SET active = 0 WHERE id = 1", ())
            .await
            .expect("deactivate first");

        let msg = session
            .conversation()
            .read_first_user_message()
            .await
            .expect("read first user message");
        assert_eq!(
            msg, "first",
            "should return the absolute first user message, not filtered by active"
        );
    }
}
