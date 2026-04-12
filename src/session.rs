use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use turso::{Builder, Connection, Value};

use crate::types::{ContentBlock, Message, Role};

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS conversation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);";

pub struct Session {
    pub id: String,
    pub conn: Connection,
}

impl Session {
    pub async fn new(id: Option<String>, session_dir: PathBuf) -> Result<Self> {
        let sess_id = match id {
            Some(session_id) => {
                validate_uuidv7(&session_id)?;
                session_id
            }
            None => generate_uuidv7(),
        };

        let db_path = session_dir.join(format!("{}.db", &sess_id));
        let needs_migration = !db_path.exists();
        let db = Builder::new_local(db_path.to_string_lossy().as_ref())
            .build()
            .await
            .with_context(|| format!("Failed to open session DB at {}", db_path.display()))?;
        let conn = db.connect()?;
        if !needs_migration {
            return Ok(Self { id: sess_id, conn });
        }
        conn.execute(SCHEMA, ())
            .await
            .context("Failed to create conversation table")?;

        Ok(Self { id: sess_id, conn })
    }

    pub async fn insert_message(&self, message: &Message) -> Result<()> {
        //insert_message_with_conn(&self.conn, message).await
        let content_json = serde_json::to_string(&message.content)
            .context("Failed to serialize message content")?;
        let role_str = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        self.conn
            .execute(
                "INSERT INTO conversation (role, content) VALUES (?1, ?2)",
                [Value::Text(role_str.to_string()), Value::Text(content_json)],
            )
            .await
            .context("Failed to insert message into session")?;
        Ok(())
    }

    pub async fn load_history(&self) -> Result<Vec<Message>> {
        //load_history_from_conn(&self.conn).await
        let mut rows = self
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
}

pub async fn load_history_from_conn(conn: &Connection) -> Result<Vec<Message>> {
    let mut rows = conn
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

fn lock_session(m: &Mutex<Session>) -> std::sync::MutexGuard<'_, Session> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub async fn persist_to_session(session: &Mutex<Session>, message: &Message) {
    let owned_conn = {
        let guard = lock_session(session);
        guard.conn.clone()
    };
    let _ = insert_message_with_conn(&owned_conn, message).await;
}

async fn insert_message_with_conn(conn: &Connection, message: &Message) -> Result<()> {
    let content_json =
        serde_json::to_string(&message.content).context("Failed to serialize message content")?;
    let role_str = match message.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    conn.execute(
        "INSERT INTO conversation (role, content) VALUES (?1, ?2)",
        [Value::Text(role_str.to_string()), Value::Text(content_json)],
    )
    .await
    .context("Failed to insert message into session")?;
    Ok(())
}

#[cfg(test)]
thread_local! {
    static TEST_SESSIONS_DIR: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

fn generate_uuidv7() -> String {
    uuid::Uuid::now_v7().to_string()
}

fn validate_uuidv7(id: &str) -> Result<()> {
    let parsed = uuid::Uuid::parse_str(id).with_context(|| format!("Invalid UUID: '{id}'"))?;
    let version = parsed.get_version();
    match version {
        Some(uuid::Version::SortRand) | Some(uuid::Version::SortMac) => Ok(()),
        _ => bail!(
            "Session ID must be a valid UUIDv7, got version {:?}",
            version
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn set_test_sessions_dir(dir: &Path) {
        TEST_SESSIONS_DIR.with(|d| *d.borrow_mut() = Some(dir.to_path_buf()));
    }

    fn clear_test_sessions_dir() {
        TEST_SESSIONS_DIR.with(|d| *d.borrow_mut() = None);
    }

    #[test]
    fn validate_uuidv7_accepts_valid_uuidv7() {
        let id = uuid::Uuid::now_v7().to_string();
        assert!(validate_uuidv7(&id).is_ok());
    }

    #[test]
    fn validate_uuidv7_rejects_non_uuid() {
        let result = validate_uuidv7("not-a-uuid");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid UUID"));
    }

    #[test]
    fn validate_uuidv7_rejects_uuidv4() {
        let v4 = uuid::Uuid::new_v4().to_string();
        let result = validate_uuidv7(&v4);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("UUIDv7"),
            "should reject non-v7 UUID"
        );
    }

    #[tokio::test]
    async fn create_session_initializes_db_with_schema() {
        let dir = TempDir::new().expect("temp dir");
        set_test_sessions_dir(dir.path());

        let session = Session::new(None).await.expect("create session");

        let db_path = dir.path().join(format!("{}.db", session.id));
        assert!(db_path.exists(), "DB file should be created");

        let msg = Message::text(Role::User, "hello".to_string());
        session.insert_message(&msg).await.expect("insert");
        let history = session.load_history().await.expect("load");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::User);

        clear_test_sessions_dir();
    }

    #[tokio::test]
    async fn create_session_with_custom_id_uses_that_id() {
        let dir = TempDir::new().expect("temp dir");
        set_test_sessions_dir(dir.path());

        let id = uuid::Uuid::now_v7().to_string();
        let session = Session::new(Some(id.clone()))
            .await
            .expect("create session");

        assert_eq!(session.id, id);
        assert!(dir.path().join(format!("{id}.db")).exists());

        clear_test_sessions_dir();
    }

    #[tokio::test]
    async fn create_session_rejects_invalid_id() {
        let dir = TempDir::new().expect("temp dir");
        set_test_sessions_dir(dir.path());

        let result = Session::new(Some("garbage".to_string())).await;
        assert!(result.is_err());

        clear_test_sessions_dir();
    }

    #[tokio::test]
    async fn load_session_reads_existing_db() {
        let dir = TempDir::new().expect("temp dir");
        set_test_sessions_dir(dir.path());

        let id = uuid::Uuid::now_v7().to_string();

        {
            let session = Session::new(Some(id.clone())).await.expect("create");
            let msg = Message::text(Role::User, "saved message".to_string());
            session.insert_message(&msg).await.expect("insert");
        }

        let session = Session::load(dir.path(), &id).await.expect("load");
        let history = session.load_history().await.expect("load history");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::User);
        match &history[0].content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "saved message"),
            _ => panic!("expected text block"),
        }

        clear_test_sessions_dir();
    }

    #[tokio::test]
    async fn load_session_errors_on_missing_file() {
        let dir = TempDir::new().expect("temp dir");
        let id = uuid::Uuid::now_v7().to_string();
        let result = Session::load(dir.path(), &id).await;
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn create_or_load_loads_existing() {
        let dir = TempDir::new().expect("temp dir");
        set_test_sessions_dir(dir.path());

        let id = uuid::Uuid::now_v7().to_string();

        {
            let session = Session::new(Some(id.clone())).await.expect("create");
            session
                .insert_message(&Message::text(Role::User, "exists".to_string()))
                .await
                .expect("insert");
        }

        let session = Session::create_or_load(id).await.expect("create_or_load");
        let history = session.load_history().await.expect("history");
        assert_eq!(history.len(), 1);

        clear_test_sessions_dir();
    }

    #[tokio::test]
    async fn create_or_load_creates_when_missing() {
        let dir = TempDir::new().expect("temp dir");
        set_test_sessions_dir(dir.path());

        let id = uuid::Uuid::now_v7().to_string();

        let session = Session::create_or_load(id.clone())
            .await
            .expect("create_or_load");
        assert_eq!(session.id, id);
        let history = session.load_history().await.expect("history");
        assert!(history.is_empty());

        clear_test_sessions_dir();
    }

    #[tokio::test]
    async fn insert_and_load_multiple_messages() {
        let dir = TempDir::new().expect("temp dir");
        set_test_sessions_dir(dir.path());

        let session = Session::new(None).await.expect("create");

        session
            .insert_message(&Message::text(Role::User, "first".to_string()))
            .await
            .expect("insert 1");
        session
            .insert_message(&Message::text(Role::Assistant, "second".to_string()))
            .await
            .expect("insert 2");
        session
            .insert_message(&Message::text(Role::User, "third".to_string()))
            .await
            .expect("insert 3");

        let history = session.load_history().await.expect("load");
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].role, Role::User);
        assert_eq!(history[1].role, Role::Assistant);
        assert_eq!(history[2].role, Role::User);

        clear_test_sessions_dir();
    }

    #[tokio::test]
    async fn insert_message_with_tool_use_content() {
        let dir = TempDir::new().expect("temp dir");
        set_test_sessions_dir(dir.path());

        let session = Session::new(None).await.expect("create");

        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text("let me check".to_string()),
                ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "ls"}),
                },
            ],
        };
        session.insert_message(&msg).await.expect("insert");

        let history = session.load_history().await.expect("load");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content.len(), 2);
        match &history[0].content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "let me check"),
            _ => panic!("expected text"),
        }
        match &history[0].content[1] {
            ContentBlock::ToolUse { name, .. } => assert_eq!(name, "bash"),
            _ => panic!("expected tool_use"),
        }

        clear_test_sessions_dir();
    }

    #[tokio::test]
    async fn insert_message_with_tool_result_content() {
        let dir = TempDir::new().expect("temp dir");
        set_test_sessions_dir(dir.path());

        let session = Session::new(None).await.expect("create");

        let msg = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".to_string(),
                content: "file.txt".to_string(),
                is_error: false,
            }],
        };
        session.insert_message(&msg).await.expect("insert");

        let history = session.load_history().await.expect("load");
        assert_eq!(history.len(), 1);
        match &history[0].content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "t1");
                assert_eq!(content, "file.txt");
                assert!(!is_error);
            }
            _ => panic!("expected tool_result"),
        }

        clear_test_sessions_dir();
    }

    #[tokio::test]
    async fn session_id_is_uuidv7() {
        let dir = TempDir::new().expect("temp dir");
        set_test_sessions_dir(dir.path());

        let session = Session::new(None).await.expect("create");
        assert!(
            validate_uuidv7(&session.id).is_ok(),
            "auto-generated session ID should be valid UUIDv7"
        );

        clear_test_sessions_dir();
    }

    #[test]
    fn sessions_dir_returns_expected_path() {
        let dir = sessions_dir().expect("should resolve");
        assert!(dir.to_string_lossy().contains("illustrious-manager"));
        assert!(dir.to_string_lossy().contains("sessions"));
    }
}
