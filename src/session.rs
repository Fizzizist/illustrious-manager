use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use turso::{Builder, Value};

use crate::types::{ContentBlock, Message, Role};

const CREATE_TABLE_SQL: &str = "CREATE TABLE IF NOT EXISTS conversation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);";

const INSERT_SQL: &str = "INSERT INTO conversation (role, content) VALUES (?, ?);";

const SELECT_ALL_SQL: &str = "SELECT role, content FROM conversation ORDER BY id ASC;";

pub fn sessions_dir() -> Result<PathBuf> {
    let config_dir =
        dirs::config_dir().context("Could not determine config directory for sessions")?;
    let dir = config_dir.join("illustrious-manager").join("sessions");
    Ok(dir)
}

pub fn session_db_path(session_id: &str) -> Result<PathBuf> {
    validate_session_id(session_id)?;
    let dir = sessions_dir()?;
    Ok(dir.join(format!("{session_id}.db")))
}

pub fn validate_session_id(id: &str) -> Result<()> {
    if id.is_empty() {
        bail!("session ID must not be empty");
    }

    id.parse::<uuid7::Uuid>()
        .map(|_| ())
        .with_context(|| format!("session ID '{id}' is not a valid UUID"))
}

pub fn generate_session_id() -> String {
    uuid7::uuid7().to_string()
}

pub struct Session {
    db: turso::Database,
}

impl Session {
    pub async fn create(session_id: &str) -> Result<Self> {
        validate_session_id(session_id)?;

        let db_path = session_db_path(session_id)?;
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create sessions directory: {}", parent.display())
            })?;
        }

        if db_path.exists() {
            bail!("session DB file already exists: {}", db_path.display());
        }

        let db = Builder::new_local(db_path.to_string_lossy().as_ref())
            .build()
            .await
            .with_context(|| format!("Failed to create session DB at {}", db_path.display()))?;

        let conn = db
            .connect()
            .context("Failed to connect to new session DB")?;
        conn.execute(CREATE_TABLE_SQL, ())
            .await
            .context("Failed to create conversation table")?;

        Ok(Self { db })
    }

    pub async fn open(session_id: &str) -> Result<Self> {
        validate_session_id(session_id)?;

        let db_path = session_db_path(session_id)?;
        if !db_path.exists() {
            bail!(
                "session DB file not found: {}. Use --session-id with a new UUIDv7 to create a new session.",
                db_path.display()
            );
        }

        let db = Builder::new_local(db_path.to_string_lossy().as_ref())
            .build()
            .await
            .with_context(|| format!("Failed to open session DB at {}", db_path.display()))?;

        Ok(Self { db })
    }

    pub async fn open_or_create(session_id: &str) -> Result<Self> {
        validate_session_id(session_id)?;

        let db_path = session_db_path(session_id)?;
        if db_path.exists() {
            Self::open(session_id).await
        } else {
            Self::create(session_id).await
        }
    }

    #[cfg(test)]
    async fn create_in_dir(session_id: &str, dir: &std::path::Path) -> Result<Self> {
        validate_session_id(session_id)?;

        let db_path = dir.join(format!("{session_id}.db"));

        if db_path.exists() {
            bail!("session DB file already exists: {}", db_path.display());
        }

        let db = Builder::new_local(db_path.to_string_lossy().as_ref())
            .build()
            .await
            .with_context(|| format!("Failed to create session DB at {}", db_path.display()))?;

        let conn = db
            .connect()
            .context("Failed to connect to new session DB")?;
        conn.execute(CREATE_TABLE_SQL, ())
            .await
            .context("Failed to create conversation table")?;

        Ok(Self { db })
    }

    #[cfg(test)]
    async fn open_in_dir(session_id: &str, dir: &std::path::Path) -> Result<Self> {
        validate_session_id(session_id)?;

        let db_path = dir.join(format!("{session_id}.db"));
        if !db_path.exists() {
            bail!("session DB file not found: {}", db_path.display());
        }

        let db = Builder::new_local(db_path.to_string_lossy().as_ref())
            .build()
            .await
            .with_context(|| format!("Failed to open session DB at {}", db_path.display()))?;

        Ok(Self { db })
    }

    #[cfg(test)]
    async fn open_or_create_in_dir(session_id: &str, dir: &std::path::Path) -> Result<Self> {
        let db_path = dir.join(format!("{session_id}.db"));
        if db_path.exists() {
            Self::open_in_dir(session_id, dir).await
        } else {
            Self::create_in_dir(session_id, dir).await
        }
    }

    pub async fn load_history(&self) -> Result<Vec<Message>> {
        let conn = self
            .db
            .connect()
            .context("Failed to connect to session DB")?;

        let mut rows = conn
            .query(SELECT_ALL_SQL, ())
            .await
            .context("Failed to query conversation history")?;

        let mut messages = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .context("Failed to fetch next row from conversation history")?
        {
            let role_val = row.get_value(0).context("Failed to read role column")?;
            let content_val = row.get_value(1).context("Failed to read content column")?;

            let role_str = match role_val {
                Value::Text(s) => s,
                _ => bail!("Expected TEXT for role column"),
            };

            let content_str = match content_val {
                Value::Text(s) => s,
                _ => bail!("Expected TEXT for content column"),
            };

            let role = match role_str.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                other => bail!("Unknown role: {other}"),
            };

            let content: Vec<ContentBlock> =
                serde_json::from_str(&content_str).context("Failed to deserialize content")?;

            messages.push(Message { role, content });
        }

        Ok(messages)
    }

    pub async fn save_history(&self, messages: &[Message]) -> Result<()> {
        let mut conn = self
            .db
            .connect()
            .context("Failed to connect to session DB")?;

        let tx = conn
            .transaction()
            .await
            .context("Failed to begin transaction")?;

        tx.execute("DELETE FROM conversation", ())
            .await
            .context("Failed to clear conversation history")?;

        for message in messages {
            let role_str = match message.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            let content_json = serde_json::to_string(&message.content)
                .context("Failed to serialize message content")?;

            tx.execute(
                INSERT_SQL,
                turso::params::Params::Positional(vec![
                    Value::Text(role_str.to_string()),
                    Value::Text(content_json),
                ]),
            )
            .await
            .context("Failed to insert message into conversation")?;
        }

        tx.commit()
            .await
            .context("Failed to commit conversation history")?;

        conn.cacheflush()
            .context("Failed to flush session DB to disk")?;

        Ok(())
    }
}

pub async fn session_exists(session_id: &str) -> Result<bool> {
    let path = session_db_path(session_id)?;
    Ok(path.exists())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ContentBlock;
    use tempfile::TempDir;

    fn test_session_id() -> String {
        uuid7::uuid7().to_string()
    }

    async fn create_test_session(tmp: &TempDir, session_id: &str) -> Session {
        let db_path = tmp.path().join(format!("{session_id}.db"));
        let db = Builder::new_local(db_path.to_string_lossy().as_ref())
            .build()
            .await
            .expect("Failed to create test DB");

        let conn = db.connect().expect("Failed to connect");
        conn.execute(CREATE_TABLE_SQL, ())
            .await
            .expect("Failed to create table");

        Session { db }
    }

    async fn open_test_session(tmp: &TempDir, session_id: &str) -> Session {
        let db_path = tmp.path().join(format!("{session_id}.db"));
        let db = Builder::new_local(db_path.to_string_lossy().as_ref())
            .build()
            .await
            .expect("Failed to open test DB");
        Session { db }
    }

    #[test]
    fn validate_session_id_accepts_valid_uuidv7() {
        let id = uuid7::uuid7().to_string();
        assert!(validate_session_id(&id).is_ok());
    }

    #[test]
    fn validate_session_id_rejects_empty_string() {
        assert!(validate_session_id("").is_err());
    }

    #[test]
    fn validate_session_id_rejects_too_short() {
        assert!(validate_session_id("abc").is_err());
    }

    #[test]
    fn validate_session_id_rejects_non_hex() {
        let id = "01944ab8-7a67-7zzz-9219-566f82fff672";
        assert!(validate_session_id(id).is_err());
    }

    #[test]
    fn validate_session_id_rejects_misplaced_hyphens() {
        let bad = "0194-4ab8-7a67-7000-9219-566f82fff672";
        assert!(validate_session_id(bad).is_err());
    }

    #[test]
    fn validate_session_id_accepts_no_hyphen_format() {
        let id = "01944ab87a6770009219566f82fff672";
        assert!(validate_session_id(id).is_ok());
    }

    #[test]
    fn generate_session_id_is_valid() {
        let id = generate_session_id();
        assert!(validate_session_id(&id).is_ok());
    }

    #[tokio::test]
    async fn save_and_load_single_text_message() {
        let tmp = TempDir::new().expect("temp dir");
        let session_id = test_session_id();
        let session = create_test_session(&tmp, &session_id).await;

        let msg = Message::text(Role::User, "Hello, world!".to_string());
        session
            .save_history(&[msg])
            .await
            .expect("save should succeed");

        let loaded = session.load_history().await.expect("load should succeed");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].role, Role::User);
        assert_eq!(
            loaded[0].content,
            vec![ContentBlock::Text("Hello, world!".to_string())]
        );
    }

    #[tokio::test]
    async fn save_and_load_multiple_messages() {
        let tmp = TempDir::new().expect("temp dir");
        let session_id = test_session_id();
        let session = create_test_session(&tmp, &session_id).await;

        let messages = vec![
            Message::text(Role::User, "What is Rust?".to_string()),
            Message::text(
                Role::Assistant,
                "Rust is a systems programming language.".to_string(),
            ),
            Message::text(Role::User, "Tell me more.".to_string()),
        ];

        session
            .save_history(&messages)
            .await
            .expect("save should succeed");

        let loaded = session.load_history().await.expect("load should succeed");
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].role, Role::User);
        assert_eq!(loaded[1].role, Role::Assistant);
        assert_eq!(loaded[2].role, Role::User);
    }

    #[tokio::test]
    async fn save_and_load_tool_use_message() {
        let tmp = TempDir::new().expect("temp dir");
        let session_id = test_session_id();
        let session = create_test_session(&tmp, &session_id).await;

        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text("Running ls".to_string()),
                ContentBlock::ToolUse {
                    id: "tool-1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "ls -la"}),
                },
            ],
        }];

        session
            .save_history(&messages)
            .await
            .expect("save should succeed");
        let loaded = session.load_history().await.expect("load should succeed");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].role, Role::Assistant);
        assert_eq!(loaded[0].content.len(), 2);
        assert_eq!(
            loaded[0].content[0],
            ContentBlock::Text("Running ls".to_string())
        );

        if let ContentBlock::ToolUse { id, name, input } = &loaded[0].content[1] {
            assert_eq!(id, "tool-1");
            assert_eq!(name, "bash");
            assert_eq!(input, &serde_json::json!({"command": "ls -la"}));
        } else {
            panic!("Expected ToolUse content block");
        }
    }

    #[tokio::test]
    async fn save_and_load_tool_result_message() {
        let tmp = TempDir::new().expect("temp dir");
        let session_id = test_session_id();
        let session = create_test_session(&tmp, &session_id).await;

        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tool-1".to_string(),
                content: "file1.txt\nfile2.txt".to_string(),
                is_error: false,
            }],
        }];

        session
            .save_history(&messages)
            .await
            .expect("save should succeed");
        let loaded = session.load_history().await.expect("load should succeed");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].role, Role::User);

        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } = &loaded[0].content[0]
        {
            assert_eq!(tool_use_id, "tool-1");
            assert_eq!(content, "file1.txt\nfile2.txt");
            assert!(!is_error);
        } else {
            panic!("Expected ToolResult content block");
        }
    }

    #[tokio::test]
    async fn load_history_from_empty_session_returns_empty() {
        let tmp = TempDir::new().expect("temp dir");
        let session_id = test_session_id();
        let session = create_test_session(&tmp, &session_id).await;

        let loaded = session.load_history().await.expect("load should succeed");
        assert!(loaded.is_empty());
    }

    #[tokio::test]
    async fn session_persists_across_reopens() {
        let tmp = TempDir::new().expect("temp dir");
        let session_id = test_session_id();

        {
            let session = create_test_session(&tmp, &session_id).await;
            let messages = vec![Message::text(Role::User, "persistent message".to_string())];
            session
                .save_history(&messages)
                .await
                .expect("save should succeed");
        }

        {
            let session = open_test_session(&tmp, &session_id).await;
            let loaded = session.load_history().await.expect("load should succeed");
            assert_eq!(loaded.len(), 1);
            assert_eq!(loaded[0].role, Role::User);
            assert_eq!(
                loaded[0].content,
                vec![ContentBlock::Text("persistent message".to_string())]
            );
        }
    }

    #[tokio::test]
    async fn save_history_overwrites_existing_messages() {
        let tmp = TempDir::new().expect("temp dir");
        let session_id = test_session_id();
        let session = create_test_session(&tmp, &session_id).await;

        let msg1 = vec![Message::text(Role::User, "first message".to_string())];
        session.save_history(&msg1).await.expect("save msg1");

        let new_messages = vec![
            Message::text(Role::User, "replaced user".to_string()),
            Message::text(Role::Assistant, "replaced assistant".to_string()),
        ];

        session
            .save_history(&new_messages)
            .await
            .expect("save_history should succeed");

        let loaded = session.load_history().await.expect("load should succeed");
        assert_eq!(loaded.len(), 2);
        assert_eq!(
            loaded[0].content,
            vec![ContentBlock::Text("replaced user".to_string())]
        );
        assert_eq!(
            loaded[1].content,
            vec![ContentBlock::Text("replaced assistant".to_string())]
        );
    }

    #[tokio::test]
    async fn save_history_with_empty_vec_clears_all() {
        let tmp = TempDir::new().expect("temp dir");
        let session_id = test_session_id();
        let session = create_test_session(&tmp, &session_id).await;

        let msg = vec![Message::text(Role::User, "to be cleared".to_string())];
        session.save_history(&msg).await.expect("save");

        session
            .save_history(&[])
            .await
            .expect("save_history with empty vec");

        let loaded = session.load_history().await.expect("load");
        assert!(loaded.is_empty());
    }

    #[tokio::test]
    async fn save_and_load_full_conversation_roundtrip() {
        let tmp = TempDir::new().expect("temp dir");
        let session_id = test_session_id();

        {
            let session = create_test_session(&tmp, &session_id).await;
            let messages = vec![
                Message::text(Role::User, "List files".to_string()),
                Message {
                    role: Role::Assistant,
                    content: vec![
                        ContentBlock::Text("I'll list the files.".to_string()),
                        ContentBlock::ToolUse {
                            id: "t1".to_string(),
                            name: "bash".to_string(),
                            input: serde_json::json!({"command": "ls"}),
                        },
                    ],
                },
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "t1".to_string(),
                        content: "file1.txt".to_string(),
                        is_error: false,
                    }],
                },
                Message::text(Role::Assistant, "There is one file: file1.txt".to_string()),
            ];

            session
                .save_history(&messages)
                .await
                .expect("save should succeed");
        }

        {
            let session = open_test_session(&tmp, &session_id).await;
            let loaded = session.load_history().await.expect("load should succeed");
            assert_eq!(loaded.len(), 4);
            assert_eq!(loaded[0].role, Role::User);
            assert_eq!(loaded[1].role, Role::Assistant);
            assert_eq!(loaded[2].role, Role::User);
            assert_eq!(loaded[3].role, Role::Assistant);

            assert_eq!(loaded[1].content.len(), 2);
            assert!(
                matches!(&loaded[1].content[0], ContentBlock::Text(s) if s == "I'll list the files.")
            );
            assert!(
                matches!(&loaded[1].content[1], ContentBlock::ToolUse { name, .. } if name == "bash")
            );

            assert!(
                matches!(&loaded[2].content[0], ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "t1")
            );
        }
    }

    #[tokio::test]
    async fn open_or_create_creates_new_session_when_absent() {
        let tmp = TempDir::new().expect("temp dir");
        let session_id = test_session_id();

        let db_path = tmp.path().join(format!("{session_id}.db"));
        assert!(!db_path.exists());

        let session = Session::open_or_create_in_dir(&session_id, tmp.path())
            .await
            .expect("open_or_create");

        let messages = vec![Message::text(Role::User, "hello".to_string())];
        session
            .save_history(&messages)
            .await
            .expect("save should succeed");

        assert!(db_path.exists());
    }

    #[tokio::test]
    async fn open_or_create_opens_existing_session() {
        let tmp = TempDir::new().expect("temp dir");
        let session_id = test_session_id();

        {
            let session = create_test_session(&tmp, &session_id).await;
            let messages = vec![Message::text(Role::User, "existing data".to_string())];
            session
                .save_history(&messages)
                .await
                .expect("save should succeed");
        }

        let session = Session::open_or_create_in_dir(&session_id, tmp.path())
            .await
            .expect("open_or_create");

        let loaded = session.load_history().await.expect("load should succeed");
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded[0].content,
            vec![ContentBlock::Text("existing data".to_string())]
        );
    }
}
