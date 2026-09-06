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
                "INSERT INTO conversation (role, content, active, created_at) VALUES (?1, ?2, 1, ?3)",
                [
                    turso::Value::Text(role_str.to_string()),
                    turso::Value::Text(content_json),
                    turso::Value::Real(message.created_at),
                ],
            )
            .await
            .context("Failed to insert message into session")?;
        Ok(())
    }

    /// Insert multiple messages atomically: a crash partway through leaves none
    /// of them behind, keeping paired rows (e.g. `tool_use`/`tool_result`)
    /// from dangling independently in the database.
    pub async fn insert_messages(&self, messages: &[Message]) -> Result<()> {
        self.session
            .conn
            .execute("BEGIN TRANSACTION", ())
            .await
            .context("Failed to begin message-insert transaction")?;

        for msg in messages {
            if let Err(e) = self.insert_message(msg).await {
                let _ = self.session.conn.execute("ROLLBACK", ()).await;
                return Err(e);
            }
        }

        self.session
            .conn
            .execute("COMMIT", ())
            .await
            .context("Failed to commit message-insert transaction")?;

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
                "SELECT role, content, created_at FROM conversation WHERE active = 1 ORDER BY id ASC",
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
            let created_at = match row.get_value(2)? {
                turso::Value::Real(f) => f,
                turso::Value::Integer(i) => i as f64,
                _ => 0.0,
            };

            let role = match role_str.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                other => bail!("Unknown role in DB: {}", other),
            };

            let content: Vec<ContentBlock> =
                serde_json::from_str(&content_str).context("Failed to deserialize content")?;

            messages.push(Message {
                role,
                content,
                created_at,
            });
        }

        Ok(messages)
    }

    pub async fn deactivate_all(&self) -> Result<()> {
        self.session
            .conn
            .execute("UPDATE conversation SET active = 0 WHERE active = 1", ())
            .await
            .context("Failed to deactivate all conversation entries")?;
        Ok(())
    }

    /// Atomically deactivate all active entries and insert a summary message.
    /// Wrapped in a transaction so that a crash between the two operations
    /// cannot leave the database with zero active rows and no summary.
    pub async fn compact(&self, summary: &Message) -> Result<()> {
        self.session
            .conn
            .execute("BEGIN TRANSACTION", ())
            .await
            .context("Failed to begin compaction transaction")?;

        if let Err(e) = self.deactivate_all().await {
            let _ = self.session.conn.execute("ROLLBACK", ()).await;
            return Err(e);
        }

        if let Err(e) = self.insert_message(summary).await {
            let _ = self.session.conn.execute("ROLLBACK", ()).await;
            return Err(e);
        }

        self.session
            .conn
            .execute("COMMIT", ())
            .await
            .context("Failed to commit compaction transaction")?;

        Ok(())
    }

    /// Atomically deactivate all active rows and re-insert the given messages.
    pub async fn replace_all(&self, messages: &[Message]) -> Result<()> {
        self.session
            .conn
            .execute("BEGIN TRANSACTION", ())
            .await
            .context("Failed to begin replace_all transaction")?;

        if let Err(e) = self.deactivate_all().await {
            let _ = self.session.conn.execute("ROLLBACK", ()).await;
            return Err(e);
        }

        for msg in messages {
            if let Err(e) = self.insert_message(msg).await {
                let _ = self.session.conn.execute("ROLLBACK", ()).await;
                return Err(e);
            }
        }

        self.session
            .conn
            .execute("COMMIT", ())
            .await
            .context("Failed to commit replace_all transaction")?;

        Ok(())
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
    use crate::types::{ContentBlock, Message, Role};
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
    async fn insert_messages_persists_all_in_order() {
        let (_dir, session) = create_test_session().await;

        let messages = vec![
            Message::text(Role::Assistant, "assistant half".to_string()),
            Message::text(Role::User, "user half".to_string()),
        ];
        session
            .conversation()
            .insert_messages(&messages)
            .await
            .expect("insert messages");

        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, Role::Assistant);
        assert_eq!(history[1].role, Role::User);
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
    async fn deactivate_all_sets_all_active_rows_to_zero() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "one".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "two".to_string()))
            .await
            .expect("insert 2");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "three".to_string()))
            .await
            .expect("insert 3");

        session
            .conversation()
            .deactivate_all()
            .await
            .expect("deactivate_all");

        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");
        assert!(
            history.is_empty(),
            "all entries should be inactive after deactivate_all"
        );
    }

    #[tokio::test]
    async fn deactivate_all_noops_on_empty_table() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .deactivate_all()
            .await
            .expect("deactivate_all on empty should not error");
    }

    #[tokio::test]
    async fn deactivate_all_does_not_affect_already_inactive_rows() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "active".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(
                Role::Assistant,
                "will deactivate".to_string(),
            ))
            .await
            .expect("insert 2");
        session
            .conn
            .execute("UPDATE conversation SET active = 0 WHERE id = 2", ())
            .await
            .expect("manual deactivate");

        session
            .conversation()
            .deactivate_all()
            .await
            .expect("deactivate_all");

        let mut rows = session
            .conn
            .query("SELECT active FROM conversation ORDER BY id ASC", ())
            .await
            .expect("query");
        let row1 = rows.next().await.expect("row 1").expect("row 1 present");
        match row1.get_value(0).expect("val 1") {
            turso::Value::Integer(n) => assert_eq!(n, 0, "row 1 should now be inactive"),
            other => panic!("expected integer, got {:?}", other),
        }
        let row2 = rows.next().await.expect("row 2").expect("row 2 present");
        match row2.get_value(0).expect("val 2") {
            turso::Value::Integer(n) => assert_eq!(n, 0, "row 2 should remain inactive"),
            other => panic!("expected integer, got {:?}", other),
        }
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

    #[tokio::test]
    async fn compact_deactivates_all_and_inserts_summary_atomically() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "message one".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "response one".to_string()))
            .await
            .expect("insert 2");

        let summary = Message::text(
            Role::User,
            "[Compacted] Summary of conversation".to_string(),
        );
        session
            .conversation()
            .compact(&summary)
            .await
            .expect("compact should succeed");

        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");
        assert_eq!(history.len(), 1, "only the summary should be active");
        assert_eq!(history[0].role, Role::User);
        match &history[0].content[0] {
            ContentBlock::Text(t) => assert!(t.contains("[Compacted]")),
            other => panic!("expected text block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn compact_on_empty_succeeds_with_just_summary() {
        let (_dir, session) = create_test_session().await;

        let summary = Message::text(Role::User, "[Compacted] Empty".to_string());
        session
            .conversation()
            .compact(&summary)
            .await
            .expect("compact on empty should succeed");

        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");
        assert_eq!(history.len(), 1, "summary should be the only active entry");
    }

    #[tokio::test]
    async fn created_at_roundtrips_through_db() {
        let (_dir, session) = create_test_session().await;

        let msg =
            Message::text(Role::User, "timestamped".to_string()).with_created_at(1704348000.0);
        session
            .conversation()
            .insert_message(&msg)
            .await
            .expect("insert");

        let history = session.conversation().load_history().await.expect("load");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].created_at, 1704348000.0);
    }

    #[tokio::test]
    async fn created_at_defaults_to_zero_when_absent() {
        let (_dir, session) = create_test_session().await;

        let msg = Message::text(Role::User, "default timestamp".to_string());
        assert_eq!(msg.created_at, 0.0);
        session
            .conversation()
            .insert_message(&msg)
            .await
            .expect("insert");

        let history = session.conversation().load_history().await.expect("load");
        assert_eq!(history[0].created_at, 0.0);
    }
}
