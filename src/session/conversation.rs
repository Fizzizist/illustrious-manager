use anyhow::{Context, Result, bail};

use crate::agent::compact_helpers::{count_turns, retained_turns};
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

    pub async fn deactivate_all(&self) -> Result<()> {
        self.session
            .conn
            .execute("UPDATE conversation SET active = 0 WHERE active = 1", ())
            .await
            .context("Failed to deactivate all conversation entries")?;
        Ok(())
    }

    async fn begin_transaction(&self) -> Result<()> {
        self.session
            .conn
            .execute("BEGIN TRANSACTION", ())
            .await
            .context("Failed to begin transaction")?;
        Ok(())
    }

    async fn commit_transaction(&self) -> Result<()> {
        self.session
            .conn
            .execute("COMMIT", ())
            .await
            .context("Failed to commit transaction")?;
        Ok(())
    }

    async fn rollback_transaction(&self) {
        let _ = self.session.conn.execute("ROLLBACK", ()).await;
    }

    pub async fn compact(&self, summary: &Message) -> Result<()> {
        self.begin_transaction().await?;
        if let Err(e) = self.deactivate_all().await {
            self.rollback_transaction().await;
            return Err(e);
        }
        if let Err(e) = self.insert_message(summary).await {
            self.rollback_transaction().await;
            return Err(e);
        }
        self.commit_transaction().await?;
        Ok(())
    }

    pub async fn compact_retaining(&self, summary: &Message, retain_count: u32) -> Result<bool> {
        let history = self.load_history().await?;
        let total_turns = count_turns(&history);
        if retain_count as usize >= total_turns {
            return Ok(false);
        }

        let retained_messages = if retain_count == 0 {
            Vec::new()
        } else {
            retained_turns(&history, retain_count as usize)
        };

        self.begin_transaction().await?;
        if let Err(e) = self.deactivate_all().await {
            self.rollback_transaction().await;
            return Err(e);
        }
        if let Err(e) = self.insert_message(summary).await {
            self.rollback_transaction().await;
            return Err(e);
        }
        for msg in &retained_messages {
            if let Err(e) = self.insert_message(msg).await {
                self.rollback_transaction().await;
                return Err(e);
            }
        }
        self.commit_transaction().await?;
        Ok(true)
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
    async fn compact_retaining_deactivates_old_and_keeps_last_n_turns() {
        let (_dir, session) = create_test_session().await;

        // 3 turns:
        // turn 1: user "hello" + assistant "hi there"
        // turn 2: user "how are you" + assistant "fine"
        // turn 3: user "goodbye" + assistant "see you"
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "hello".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "hi there".to_string()))
            .await
            .expect("insert 2");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "how are you".to_string()))
            .await
            .expect("insert 3");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "fine".to_string()))
            .await
            .expect("insert 4");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "goodbye".to_string()))
            .await
            .expect("insert 5");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "see you".to_string()))
            .await
            .expect("insert 6");

        let summary = Message::text(Role::User, "[Compacted] Summary".to_string());
        let result = session
            .conversation()
            .compact_retaining(&summary, 2)
            .await
            .expect("compact_retaining should succeed");
        assert!(
            result,
            "compact_retaining should return true when it performs compaction"
        );

        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");

        // Expected: summary + turn 2 messages + turn 3 messages = 1 + 4 = 5
        // But turn 2 and turn 3 only have text blocks, so they're all kept.
        assert_eq!(
            history.len(),
            5,
            "should have summary + 4 retained text messages"
        );

        assert_eq!(history[0].role, Role::User);
        assert!(
            matches!(&history[0].content[0], ContentBlock::Text(t) if t.contains("[Compacted]"))
        );
        assert_eq!(history[1].role, Role::User);
        assert!(matches!(&history[1].content[0], ContentBlock::Text(t) if t == "how are you"));
        assert_eq!(history[2].role, Role::Assistant);
        assert!(matches!(&history[2].content[0], ContentBlock::Text(t) if t == "fine"));
        assert_eq!(history[3].role, Role::User);
        assert!(matches!(&history[3].content[0], ContentBlock::Text(t) if t == "goodbye"));
        assert_eq!(history[4].role, Role::Assistant);
        assert!(matches!(&history[4].content[0], ContentBlock::Text(t) if t == "see you"));
    }

    #[tokio::test]
    async fn compact_retaining_with_zero_is_equivalent_to_compact() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "hello".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "hi there".to_string()))
            .await
            .expect("insert 2");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "how are you".to_string()))
            .await
            .expect("insert 3");

        let summary = Message::text(Role::User, "[Compacted] Summary".to_string());
        let result = session
            .conversation()
            .compact_retaining(&summary, 0)
            .await
            .expect("compact_retaining with 0 should succeed");
        assert!(
            result,
            "compact_retaining with 0 should return true (there are turns to compact)"
        );

        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");
        assert_eq!(
            history.len(),
            1,
            "with 0 retained turns, only summary should remain"
        );
        assert!(
            matches!(&history[0].content[0], ContentBlock::Text(t) if t.contains("[Compacted]"))
        );
    }

    #[tokio::test]
    async fn compact_retaining_with_count_exceeding_total_is_noop() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "hello".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "hi".to_string()))
            .await
            .expect("insert 2");

        let summary = Message::text(Role::User, "[Compacted] Summary".to_string());
        let result = session
            .conversation()
            .compact_retaining(&summary, 5)
            .await
            .expect("compact_retaining should succeed");
        assert!(
            !result,
            "compact_retaining with retain_count >= total turns should be a no-op"
        );

        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");
        assert_eq!(history.len(), 2, "history should be unchanged after no-op");
    }

    #[tokio::test]
    async fn compact_retaining_on_single_turn_is_noop() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "hello".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "hi".to_string()))
            .await
            .expect("insert 2");

        let summary = Message::text(Role::User, "[Compacted] Summary".to_string());
        let result = session
            .conversation()
            .compact_retaining(&summary, 1)
            .await
            .expect("compact_retaining should succeed");
        assert!(
            !result,
            "compact_retaining with retain_count == total turns should be a no-op"
        );
    }

    #[tokio::test]
    async fn compact_retaining_strips_tool_use_blocks() {
        let (_dir, session) = create_test_session().await;

        // 2 turns; retain 1 turn
        // turn 1: user text + assistant text (will be compacted away)
        // turn 2: user text + assistant with tool_use and text
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "hello".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "hi there".to_string()))
            .await
            .expect("insert 2");
        session
            .conversation()
            .insert_message(&Message {
                role: Role::User,
                content: vec![ContentBlock::Text("run something".to_string())],
            })
            .await
            .expect("insert 3");
        session
            .conversation()
            .insert_message(&Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::ToolUse {
                        id: "tool-1".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "ls"}),
                    },
                    ContentBlock::Text("I ran ls".to_string()),
                ],
            })
            .await
            .expect("insert 4");

        let summary = Message::text(Role::User, "[Compacted] Summary".to_string());
        let result = session
            .conversation()
            .compact_retaining(&summary, 1)
            .await
            .expect("compact_retaining should succeed");
        assert!(result);

        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");

        // summary + user "run something" + assistant "I ran ls" (tool-use stripped)
        assert_eq!(history.len(), 3);

        assert!(
            matches!(&history[0].content[0], ContentBlock::Text(t) if t.contains("[Compacted]"))
        );

        assert_eq!(history[1].role, Role::User);
        assert!(matches!(&history[1].content[0], ContentBlock::Text(t) if t == "run something"));

        assert_eq!(history[2].role, Role::Assistant);
        assert_eq!(
            history[2].content.len(),
            1,
            "tool-use block should be stripped"
        );
        assert!(matches!(&history[2].content[0], ContentBlock::Text(t) if t == "I ran ls"));
    }

    #[tokio::test]
    async fn compact_retaining_strips_thinking_and_tool_result_blocks() {
        let (_dir, session) = create_test_session().await;

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "hello".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "hi".to_string()))
            .await
            .expect("insert 2");
        session
            .conversation()
            .insert_message(&Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        text: "deep thoughts".to_string(),
                        signature: "sig123".to_string(),
                    },
                    ContentBlock::RedactedThinking {
                        data: "redacted_data".to_string(),
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "tool-1".to_string(),
                        content: "output".to_string(),
                        is_error: false,
                    },
                    ContentBlock::Text("visible response".to_string()),
                ],
            })
            .await
            .expect("insert 3");

        // 2 turns total, retain 1 (the one with the assistant message that has mixed blocks)
        // But wait — this is only 1 full turn (user + assistant) plus an orphan assistant.
        // Let's start over with a proper 2-turn setup using a new session.
        drop(session);
        let dir2 = TempDir::new().expect("temp dir 2");
        let session2 = super::super::Session::new(None, dir2.path().to_path_buf())
            .await
            .expect("create session 2");

        // turn 1: user + assistant
        session2
            .conversation()
            .insert_message(&Message::text(Role::User, "hello".to_string()))
            .await
            .expect("insert 1");
        session2
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "hi".to_string()))
            .await
            .expect("insert 2");

        // turn 2: user + assistant with thinking, redacted, tool_result, text
        session2
            .conversation()
            .insert_message(&Message::text(Role::User, "question".to_string()))
            .await
            .expect("insert 3");
        session2
            .conversation()
            .insert_message(&Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        text: "deep thoughts".to_string(),
                        signature: "sig123".to_string(),
                    },
                    ContentBlock::RedactedThinking {
                        data: "redacted_data".to_string(),
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "tool-1".to_string(),
                        content: "output".to_string(),
                        is_error: false,
                    },
                    ContentBlock::Text("visible response".to_string()),
                ],
            })
            .await
            .expect("insert 4");

        let summary = Message::text(Role::User, "[Compacted] Summary".to_string());
        let result = session2
            .conversation()
            .compact_retaining(&summary, 1)
            .await
            .expect("compact_retaining should succeed");
        assert!(result);

        let history = session2
            .conversation()
            .load_history()
            .await
            .expect("load history");

        // summary + user "question" + assistant "visible response" (thinking, redacted, tool_result stripped)
        assert_eq!(history.len(), 3);

        assert_eq!(history[1].role, Role::User);
        assert!(matches!(&history[1].content[0], ContentBlock::Text(t) if t == "question"));

        assert_eq!(history[2].role, Role::Assistant);
        assert_eq!(history[2].content.len(), 1, "only text block should remain");
        assert!(matches!(&history[2].content[0], ContentBlock::Text(t) if t == "visible response"));
    }

    #[tokio::test]
    async fn compact_retaining_drops_empty_messages_after_stripping() {
        let (_dir, session) = create_test_session().await;

        // turn 1: user + assistant (will be compacted)
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "hello".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "hi".to_string()))
            .await
            .expect("insert 2");

        // turn 2: user text + assistant with only tool_use (no text → entire assistant message dropped)
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "run it".to_string()))
            .await
            .expect("insert 3");
        session
            .conversation()
            .insert_message(&Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "tool-1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "ls"}),
                }],
            })
            .await
            .expect("insert 4");

        let summary = Message::text(Role::User, "[Compacted] Summary".to_string());
        let result = session
            .conversation()
            .compact_retaining(&summary, 1)
            .await
            .expect("compact_retaining should succeed");
        assert!(result);

        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");

        // Turn 2 has:
        //   user "run it" (text only → kept)
        //   assistant with only ToolUse (stripped → empty → dropped)
        // So: summary + user "run it" = 2 active messages
        assert_eq!(history.len(), 2);
        assert!(
            matches!(&history[0].content[0], ContentBlock::Text(t) if t.contains("[Compacted]"))
        );
        assert_eq!(history[1].role, Role::User);
        assert!(matches!(&history[1].content[0], ContentBlock::Text(t) if t == "run it"));
    }
}
