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

    /// Load active conversation rows with their DB row IDs.
    pub async fn load_active_history_with_ids(&self) -> Result<Vec<(i64, Message)>> {
        let mut rows = self
            .session
            .conn
            .query(
                "SELECT id, role, content FROM conversation WHERE active = 1 ORDER BY id ASC",
                (),
            )
            .await
            .context("Failed to query conversation history")?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            let id = match row.get_value(0)? {
                turso::Value::Integer(n) => n,
                other => bail!("Unexpected id type in DB: {:?}", other),
            };
            let role_str = match row.get_value(1)? {
                turso::Value::Text(s) => s,
                other => bail!("Unexpected role type in DB: {:?}", other),
            };
            let content_str = match row.get_value(2)? {
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

            results.push((id, Message { role, content }));
        }

        Ok(results)
    }

    pub async fn load_history(&self) -> Result<Vec<Message>> {
        Ok(self
            .load_active_history_with_ids()
            .await?
            .into_iter()
            .map(|(_, msg)| msg)
            .collect())
    }

    /// Load all conversation rows including inactive (compacted) ones.
    pub async fn load_full_history(&self) -> Result<Vec<(i64, Message, bool)>> {
        let mut rows = self
            .session
            .conn
            .query(
                "SELECT id, role, content, active FROM conversation ORDER BY id ASC",
                (),
            )
            .await
            .context("Failed to query full conversation history")?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            let id = match row.get_value(0)? {
                turso::Value::Integer(n) => n,
                other => bail!("Unexpected id type in DB: {:?}", other),
            };
            let role_str = match row.get_value(1)? {
                turso::Value::Text(s) => s,
                other => bail!("Unexpected role type in DB: {:?}", other),
            };
            let content_str = match row.get_value(2)? {
                turso::Value::Text(s) => s,
                other => bail!("Unexpected content type in DB: {:?}", other),
            };
            let active = match row.get_value(3)? {
                turso::Value::Integer(n) => n != 0,
                other => bail!("Unexpected active type in DB: {:?}", other),
            };

            let role = match role_str.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                other => bail!("Unknown role in DB: {}", other),
            };

            let content: Vec<ContentBlock> =
                serde_json::from_str(&content_str).context("Failed to deserialize content")?;

            results.push((id, Message { role, content }, active));
        }

        Ok(results)
    }

    /// Soft-delete messages by marking them as inactive.
    pub async fn deactivate_messages(&self, ids: &[i64]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "UPDATE conversation SET active = 0 WHERE id IN ({})",
            placeholders.join(",")
        );
        let params: Vec<turso::Value> = ids.iter().map(|&id| turso::Value::Integer(id)).collect();
        self.session
            .conn
            .execute(&sql, turso::params_from_iter(params))
            .await
            .context("Failed to deactivate messages")?;
        Ok(())
    }

    /// Insert a message and return its row id.
    pub async fn insert_message_returning_id(&self, message: &Message) -> Result<i64> {
        let content_json = serde_json::to_string(&message.content)
            .context("Failed to serialize message content")?;
        let role_str = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        let mut rows = self
            .session
            .conn
            .query(
                "INSERT INTO conversation (role, content) VALUES (?1, ?2) RETURNING id",
                [
                    turso::Value::Text(role_str.to_string()),
                    turso::Value::Text(content_json),
                ],
            )
            .await
            .context("Failed to insert message into session")?;
        if let Some(row) = rows.next().await? {
            match row.get_value(0)? {
                turso::Value::Integer(id) => return Ok(id),
                other => bail!("Unexpected returning id type: {:?}", other),
            }
        }
        bail!("INSERT RETURNING did not return a row")
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
