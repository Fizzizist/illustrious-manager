use anyhow::{Context, Result, bail};
use turso::Value;

use crate::types::{ContentBlock, Message, Role};

use super::Session;

pub async fn insert_message(session: &Session, message: &Message) -> Result<()> {
    let content_json =
        serde_json::to_string(&message.content).context("Failed to serialize message content")?;
    let role_str = match message.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    session
        .conn
        .execute(
            "INSERT INTO conversation (role, content) VALUES (?1, ?2)",
            [Value::Text(role_str.to_string()), Value::Text(content_json)],
        )
        .await
        .context("Failed to insert message into session")?;
    Ok(())
}

pub async fn is_empty(session: &Session) -> Result<bool> {
    let mut rows = session
        .conn
        .query("SELECT COUNT(*) FROM conversation", ())
        .await
        .context("Failed to count conversation messages")?;

    if let Some(row) = rows.next().await? {
        let count = match row.get_value(0)? {
            Value::Integer(n) => n,
            _ => return Ok(true),
        };
        return Ok(count == 0);
    }

    Ok(true)
}

pub async fn load_history(session: &Session) -> Result<Vec<Message>> {
    let mut rows = session
        .conn
        .query("SELECT role, content FROM conversation ORDER BY id ASC", ())
        .await
        .context("Failed to query conversation history")?;

    let mut messages = Vec::new();
    while let Some(row) = rows.next().await? {
        let role_str = match row.get_value(0)? {
            Value::Text(s) => s,
            other => bail!("Unexpected role type in DB: {:?}", other),
        };
        let content_str = match row.get_value(1)? {
            Value::Text(s) => s,
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

pub async fn read_first_user_message(session: &Session) -> Result<String> {
    let mut rows = session
        .conn
        .query(
            "SELECT content FROM conversation WHERE role = 'user' ORDER BY id ASC LIMIT 1",
            (),
        )
        .await
        .context("Failed to query first user message")?;

    if let Some(row) = rows.next().await? {
        let content_str = match row.get_value(0)? {
            Value::Text(s) => s,
            _ => return Ok(String::new()),
        };
        let blocks: Vec<ContentBlock> =
            serde_json::from_str(&content_str).context("Failed to deserialize content blocks")?;
        for block in blocks {
            if let ContentBlock::Text(text) = block {
                return Ok(text);
            }
        }
    }

    Ok(String::new())
}
