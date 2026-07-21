pub mod conversation;
pub mod task;

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::SystemTime;
use turso::{Builder, Connection};

pub use conversation::ConversationRepo;
pub use task::{TaskRecord, TaskRepo, TaskStatus};

/// Filesystem reference to a session DB, without any DB content loaded.
#[derive(Debug, Clone)]
pub struct SessionRef {
    pub id: String,
    pub path: PathBuf,
    pub modified: SystemTime,
}

/// Summary of a session, used for the session picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSummary {
    pub id: String,
    pub first_user_message: String,
    pub modified: SystemTime,
}

pub async fn enumerate_sessions(session_dir: &std::path::Path) -> Result<Vec<SessionRef>> {
    let mut refs = Vec::new();

    let read_dir = match std::fs::read_dir(session_dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(e).with_context(|| {
                format!(
                    "Failed to read sessions directory: {}",
                    session_dir.display()
                )
            });
        }
    };

    for entry in read_dir {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        if path.extension().and_then(|e| e.to_str()) != Some("db") {
            continue;
        }

        let file_stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };

        if validate_uuidv7(&file_stem).is_err() {
            continue;
        }

        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        refs.push(SessionRef {
            id: file_stem,
            path,
            modified,
        });
    }

    refs.sort_by_key(|b| std::cmp::Reverse(b.modified));

    Ok(refs)
}

pub async fn hydrate_sessions(refs: &[SessionRef]) -> Vec<SessionSummary> {
    let mut summaries = Vec::with_capacity(refs.len());
    for r in refs {
        let first_user_message = read_first_user_message_from_path(&r.path)
            .await
            .unwrap_or_default();
        summaries.push(SessionSummary {
            id: r.id.clone(),
            first_user_message,
            modified: r.modified,
        });
    }
    summaries
}

pub async fn hydrate_next_page(
    pending: &mut Vec<SessionRef>,
    page_size: usize,
) -> (Vec<SessionSummary>, bool) {
    let page_len = pending.len().min(page_size);
    let page = hydrate_sessions(&pending[..page_len]).await;
    let has_more = pending.len() > page_size;
    pending.drain(..page_len);
    (page, has_more)
}

pub async fn list_sessions(session_dir: &std::path::Path) -> Result<Vec<SessionSummary>> {
    let refs = enumerate_sessions(session_dir).await?;
    Ok(hydrate_sessions(&refs).await)
}

async fn read_first_user_message_from_path(db_path: &std::path::Path) -> Result<String> {
    let db = Builder::new_local(db_path.to_string_lossy().as_ref())
        .build()
        .await
        .with_context(|| format!("Failed to open session DB at {}", db_path.display()))?;
    let conn = db.connect()?;
    let session = Session {
        id: String::new(),
        conn,
        db_path: db_path.to_path_buf(),
    };
    session.conversation().read_first_user_message().await
}

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS conversation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS task (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);";

pub struct Session {
    pub id: String,
    pub conn: Connection,
    db_path: PathBuf,
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

        let db_path = session_dir.join(format!("{}.db", sess_id));
        let db = Builder::new_local(db_path.to_string_lossy().as_ref())
            .build()
            .await
            .with_context(|| format!("Failed to open session DB at {}", db_path.display()))?;
        let conn = db.connect()?;
        conn.execute_batch(SCHEMA)
            .await
            .context("Failed to run schema DDL")?;

        migrate_active_column(&conn).await?;
        migrate_created_at_column(&conn).await?;

        Ok(Self {
            id: sess_id,
            conn,
            db_path,
        })
    }

    pub fn conversation(&self) -> ConversationRepo<'_> {
        ConversationRepo::new(self)
    }

    pub fn tasks(&self) -> TaskRepo<'_> {
        TaskRepo::new(self)
    }

    /// Checkpoint the WAL into the main DB file and truncate the WAL sidecar.
    ///
    /// Must be called before dropping the session when a clean, self-contained `.db`
    /// is required (e.g. on graceful exit or session switch).
    pub async fn checkpoint(&self) -> Result<()> {
        let mut rows = self
            .conn
            .query("PRAGMA wal_checkpoint(TRUNCATE)", ())
            .await
            .context("Failed to execute wal_checkpoint pragma")?;
        while rows.next().await?.is_some() {}
        Ok(())
    }

    pub fn delete_db(&self) -> Result<()> {
        for suffix in ["", "-wal", "-shm"] {
            let path = if suffix.is_empty() {
                self.db_path.clone()
            } else {
                self.db_path.with_extension(format!("db{}", suffix))
            };
            if path.exists() {
                std::fs::remove_file(&path)
                    .with_context(|| format!("Failed to remove {}", path.display()))?;
            }
        }
        Ok(())
    }
}

async fn migrate_active_column(conn: &Connection) -> Result<()> {
    let mut rows = conn
        .query("PRAGMA table_info(conversation)", ())
        .await
        .context("Failed to query conversation table info")?;
    let mut has_active = false;
    while let Some(row) = rows.next().await? {
        if let turso::Value::Text(name) = row.get_value(1)?
            && name == "active"
        {
            has_active = true;
            break;
        }
    }
    if !has_active {
        conn.execute_batch("ALTER TABLE conversation ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
            .await
            .context("Failed to add active column to conversation table")?;
    }
    Ok(())
}

async fn migrate_created_at_column(conn: &Connection) -> Result<()> {
    let mut rows = conn
        .query("PRAGMA table_info(conversation)", ())
        .await
        .context("Failed to query conversation table info")?;
    let mut has_created_at = false;
    while let Some(row) = rows.next().await? {
        if let turso::Value::Text(name) = row.get_value(1)?
            && name == "created_at"
        {
            has_created_at = true;
            break;
        }
    }
    if !has_created_at {
        conn.execute_batch(
            "ALTER TABLE conversation ADD COLUMN created_at REAL NOT NULL DEFAULT 0.0",
        )
        .await
        .context("Failed to add created_at column to conversation table")?;
    }
    Ok(())
}

fn generate_uuidv7() -> String {
    uuid::Uuid::now_v7().to_string()
}

fn validate_uuidv7(id: &str) -> Result<()> {
    let parsed = uuid::Uuid::parse_str(id).with_context(|| format!("Invalid UUID: '{id}'"))?;
    let version = parsed.get_version();
    match version {
        Some(uuid::Version::SortRand) | Some(uuid::Version::SortMac) => Ok(()),
        _ => anyhow::bail!(
            "Session ID must be a valid UUIDv7, got version {:?}",
            version
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentBlock, Message, Role};
    use tempfile::TempDir;

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

        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create session");

        let db_path = dir.path().join(format!("{}.db", session.id));
        assert!(db_path.exists(), "DB file should be created");

        let msg = Message::text(Role::User, "hello".to_string());
        session
            .conversation()
            .insert_message(&msg)
            .await
            .expect("insert");
        let history = session.conversation().load_history().await.expect("load");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::User);
    }

    #[tokio::test]
    async fn create_session_with_custom_id_uses_that_id() {
        let dir = TempDir::new().expect("temp dir");

        let id = uuid::Uuid::now_v7().to_string();
        let session = Session::new(Some(id.clone()), dir.path().to_path_buf())
            .await
            .expect("create session");

        assert_eq!(session.id, id);
        assert!(dir.path().join(format!("{id}.db")).exists());
    }

    #[tokio::test]
    async fn create_session_rejects_invalid_id() {
        let dir = TempDir::new().expect("temp dir");

        let result = Session::new(Some("garbage".to_string()), dir.path().to_path_buf()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn reopen_session_reads_existing_db() {
        let dir = TempDir::new().expect("temp dir");

        let id = uuid::Uuid::now_v7().to_string();

        {
            let session = Session::new(Some(id.clone()), dir.path().to_path_buf())
                .await
                .expect("create");
            let msg = Message::text(Role::User, "saved message".to_string());
            session
                .conversation()
                .insert_message(&msg)
                .await
                .expect("insert");
        }

        let session = Session::new(Some(id), dir.path().to_path_buf())
            .await
            .expect("reopen");
        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::User);
        match &history[0].content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "saved message"),
            _ => panic!("expected text block"),
        }
    }

    #[tokio::test]
    async fn insert_and_load_multiple_messages() {
        let dir = TempDir::new().expect("temp dir");

        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");

        session
            .conversation()
            .insert_message(&Message::text(Role::User, "first".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "second".to_string()))
            .await
            .expect("insert 2");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "third".to_string()))
            .await
            .expect("insert 3");

        let history = session.conversation().load_history().await.expect("load");
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].role, Role::User);
        assert_eq!(history[1].role, Role::Assistant);
        assert_eq!(history[2].role, Role::User);
    }

    #[tokio::test]
    async fn insert_message_with_tool_use_content() {
        let dir = TempDir::new().expect("temp dir");

        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");

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
            created_at: 0.0,
        };
        session
            .conversation()
            .insert_message(&msg)
            .await
            .expect("insert");

        let history = session.conversation().load_history().await.expect("load");
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
    }

    #[tokio::test]
    async fn insert_message_with_tool_result_content() {
        let dir = TempDir::new().expect("temp dir");

        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");

        let msg = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".to_string(),
                content: "file.txt".to_string(),
                is_error: false,
            }],
            created_at: 0.0,
        };
        session
            .conversation()
            .insert_message(&msg)
            .await
            .expect("insert");

        let history = session.conversation().load_history().await.expect("load");
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
    }

    #[tokio::test]
    async fn session_id_is_uuidv7() {
        let dir = TempDir::new().expect("temp dir");

        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");
        assert!(
            validate_uuidv7(&session.id).is_ok(),
            "auto-generated session ID should be valid UUIDv7"
        );
    }

    #[tokio::test]
    async fn list_sessions_returns_empty_for_nonexistent_dir() {
        let dir = TempDir::new().expect("temp dir");
        let missing = dir.path().join("nonexistent");
        let summaries = list_sessions(&missing).await.expect("should not error");
        assert!(summaries.is_empty());
    }

    #[tokio::test]
    async fn list_sessions_returns_empty_for_empty_dir() {
        let dir = TempDir::new().expect("temp dir");
        let summaries = list_sessions(dir.path()).await.expect("list sessions");
        assert!(summaries.is_empty());
    }

    #[tokio::test]
    async fn list_sessions_returns_summaries_with_first_user_message() {
        let dir = TempDir::new().expect("temp dir");

        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "hello world".to_string()))
            .await
            .expect("insert");

        let summaries = list_sessions(dir.path()).await.expect("list sessions");
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].id, session.id);
        assert_eq!(summaries[0].first_user_message, "hello world");
    }

    #[tokio::test]
    async fn list_sessions_ignores_non_db_files() {
        let dir = TempDir::new().expect("temp dir");

        std::fs::write(dir.path().join("notes.txt"), b"hello").expect("write file");
        std::fs::write(dir.path().join("config.toml"), b"[foo]").expect("write file");

        Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");

        let summaries = list_sessions(dir.path()).await.expect("list sessions");
        assert_eq!(summaries.len(), 1, "only the .db file should be listed");
    }

    #[tokio::test]
    async fn list_sessions_ignores_db_files_with_non_uuidv7_names() {
        let dir = TempDir::new().expect("temp dir");

        std::fs::write(dir.path().join("not-a-uuid.db"), b"garbage").expect("write file");

        Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");

        let summaries = list_sessions(dir.path()).await.expect("list sessions");
        assert_eq!(
            summaries.len(),
            1,
            "only the valid UUID-named .db file should appear"
        );
    }

    #[tokio::test]
    async fn list_sessions_orders_by_most_recently_modified_first() {
        let dir = TempDir::new().expect("temp dir");

        let session_a = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create a");
        session_a
            .conversation()
            .insert_message(&Message::text(Role::User, "first session".to_string()))
            .await
            .expect("insert a");

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let session_b = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create b");
        session_b
            .conversation()
            .insert_message(&Message::text(Role::User, "second session".to_string()))
            .await
            .expect("insert b");

        let summaries = list_sessions(dir.path()).await.expect("list sessions");
        assert_eq!(summaries.len(), 2);
        assert_eq!(
            summaries[0].id, session_b.id,
            "most recently modified should be first"
        );
        assert_eq!(summaries[1].id, session_a.id);
    }

    #[tokio::test]
    async fn list_sessions_empty_message_for_session_with_no_messages() {
        let dir = TempDir::new().expect("temp dir");

        Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");

        let summaries = list_sessions(dir.path()).await.expect("list sessions");
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].first_user_message, "");
    }

    #[tokio::test]
    async fn list_sessions_returns_only_first_user_message_text() {
        let dir = TempDir::new().expect("temp dir");

        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "first message".to_string()))
            .await
            .expect("insert 1");
        session
            .conversation()
            .insert_message(&Message::text(Role::Assistant, "response".to_string()))
            .await
            .expect("insert 2");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "second message".to_string()))
            .await
            .expect("insert 3");

        let summaries = list_sessions(dir.path()).await.expect("list sessions");
        assert_eq!(summaries[0].first_user_message, "first message");
    }

    #[tokio::test]
    async fn is_empty_returns_true_for_new_session() {
        let dir = TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");
        assert!(session.conversation().is_empty().await.expect("is_empty"));
    }

    #[tokio::test]
    async fn is_empty_returns_false_after_inserting_message() {
        let dir = TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "hello".to_string()))
            .await
            .expect("insert");
        assert!(!session.conversation().is_empty().await.expect("is_empty"));
    }

    #[tokio::test]
    async fn checkpoint_succeeds_on_empty_session() {
        let dir = TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");
        session.checkpoint().await.expect("checkpoint");
    }

    #[tokio::test]
    async fn checkpoint_consolidates_wal_into_main_db_file() {
        let dir = TempDir::new().expect("temp dir");
        let id = uuid::Uuid::now_v7().to_string();

        {
            let session = Session::new(Some(id.clone()), dir.path().to_path_buf())
                .await
                .expect("create");
            session
                .conversation()
                .insert_message(&Message::text(Role::User, "persisted via wal".to_string()))
                .await
                .expect("insert");
            session.checkpoint().await.expect("checkpoint");
        }

        // Remove sidecar files so the next open sees only the main .db.
        let wal = dir.path().join(format!("{id}.db-wal"));
        let shm = dir.path().join(format!("{id}.db-shm"));
        if wal.exists() {
            std::fs::remove_file(&wal).expect("remove wal");
        }
        if shm.exists() {
            std::fs::remove_file(&shm).expect("remove shm");
        }

        let session = Session::new(Some(id), dir.path().to_path_buf())
            .await
            .expect("reopen");
        let history = session.conversation().load_history().await.expect("load");
        assert_eq!(history.len(), 1);
        match &history[0].content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "persisted via wal"),
            _ => panic!("expected text block"),
        }
    }

    #[tokio::test]
    async fn checkpoint_leaves_wal_file_zero_length_or_absent() {
        let dir = TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "hello".to_string()))
            .await
            .expect("insert");
        session.checkpoint().await.expect("checkpoint");

        let wal_path = dir.path().join(format!("{}.db-wal", session.id));
        if wal_path.exists() {
            assert_eq!(
                std::fs::metadata(&wal_path).expect("metadata").len(),
                0,
                "WAL should be zero-length after checkpoint"
            );
        }
    }

    #[tokio::test]
    async fn delete_db_removes_db_file() {
        let dir = TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");
        let db_path = dir.path().join(format!("{}.db", session.id));
        assert!(db_path.exists());

        session.delete_db().expect("delete_db");
        assert!(!db_path.exists());
    }

    #[tokio::test]
    async fn delete_db_removes_wal_and_shm_sidecars() {
        let dir = TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");

        let wal_path = dir.path().join(format!("{}.db-wal", session.id));
        let shm_path = dir.path().join(format!("{}.db-shm", session.id));
        std::fs::write(&wal_path, b"fake wal").expect("write wal");
        std::fs::write(&shm_path, b"fake shm").expect("write shm");

        session.delete_db().expect("delete_db");
        assert!(!wal_path.exists());
        assert!(!shm_path.exists());
    }

    #[tokio::test]
    async fn delete_db_succeeds_when_no_sidecars_exist() {
        let dir = TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create");
        session.delete_db().expect("delete_db");
    }

    #[tokio::test]
    async fn task_table_is_created_on_new_session() {
        let dir = TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create session");

        let id = session.tasks().create("t", None).await.expect("create");
        assert_eq!(id, 1);
    }

    #[tokio::test]
    async fn task_table_is_added_on_reopen_of_existing_db() {
        let dir = TempDir::new().expect("temp dir");
        let id = uuid::Uuid::now_v7().to_string();

        {
            let session = Session::new(Some(id.clone()), dir.path().to_path_buf())
                .await
                .expect("create");
            session
                .conversation()
                .insert_message(&Message::text(Role::User, "hello".to_string()))
                .await
                .expect("insert");
        }

        let session = Session::new(Some(id), dir.path().to_path_buf())
            .await
            .expect("reopen");
        let task_id = session.tasks().create("t", None).await.expect("create");
        assert_eq!(task_id, 1);
    }

    #[tokio::test]
    async fn active_column_is_added_on_reopen_of_existing_db() {
        let dir = TempDir::new().expect("temp dir");
        let id = uuid::Uuid::now_v7().to_string();
        let db_path = dir.path().join(format!("{id}.db"));

        {
            let db = Builder::new_local(db_path.to_string_lossy().as_ref())
                .build()
                .await
                .expect("build");
            let conn = db.connect().expect("connect");
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS conversation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL
                )",
            )
            .await
            .expect("create old schema");
            conn.execute(
                "INSERT INTO conversation (role, content) VALUES (?1, ?2)",
                [
                    turso::Value::Text("user".to_string()),
                    turso::Value::Text("[{\"type\":\"text\",\"text\":\"hello\"}]".to_string()),
                ],
            )
            .await
            .expect("insert");
        }

        let session = Session::new(Some(id), dir.path().to_path_buf())
            .await
            .expect("reopen with migration");
        let history = session.conversation().load_history().await.expect("load");
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn created_at_column_is_added_on_reopen_of_existing_db() {
        let dir = TempDir::new().expect("temp dir");
        let id = uuid::Uuid::now_v7().to_string();
        let db_path = dir.path().join(format!("{id}.db"));

        {
            let db = Builder::new_local(db_path.to_string_lossy().as_ref())
                .build()
                .await
                .expect("build");
            let conn = db.connect().expect("connect");
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS conversation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                )",
            )
            .await
            .expect("create schema without created_at");
            conn.execute(
                "INSERT INTO conversation (role, content, active) VALUES (?1, ?2, 1)",
                [
                    turso::Value::Text("user".to_string()),
                    turso::Value::Text("[{\"type\":\"text\",\"text\":\"hello\"}]".to_string()),
                ],
            )
            .await
            .expect("insert");
        }

        let session = Session::new(Some(id), dir.path().to_path_buf())
            .await
            .expect("reopen with migration");
        let history = session.conversation().load_history().await.expect("load");
        assert_eq!(history.len(), 1);
        assert_eq!(
            history[0].created_at, 0.0,
            "migrated rows should default to 0.0"
        );
    }

    #[tokio::test]
    async fn enumerate_sessions_returns_sorted_refs_without_hydration() {
        let dir = TempDir::new().expect("temp dir");

        let session_a = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create a");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let session_b = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create b");

        let refs = enumerate_sessions(dir.path()).await.expect("enumerate");
        assert_eq!(refs.len(), 2);
        // Most recently modified should be first
        assert_eq!(refs[0].id, session_b.id);
        assert_eq!(refs[1].id, session_a.id);
        // Paths should point to the DB files
        assert!(refs[0].path.ends_with(format!("{}.db", session_b.id)));
        assert!(refs[1].path.ends_with(format!("{}.db", session_a.id)));
    }

    #[tokio::test]
    async fn hydrate_sessions_hydrates_only_given_refs() {
        let dir = TempDir::new().expect("temp dir");

        let mut sessions = Vec::new();
        for i in 0..5 {
            let s = Session::new(None, dir.path().to_path_buf())
                .await
                .expect("create");
            s.conversation()
                .insert_message(&Message::text(Role::User, format!("msg {i}")))
                .await
                .expect("insert");
            sessions.push(s);
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        let refs = enumerate_sessions(dir.path()).await.expect("enumerate");
        // Hydrate only first 2 refs
        let summaries = hydrate_sessions(&refs[..2]).await;
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].id, refs[0].id);
        assert_eq!(summaries[1].id, refs[1].id);
        // first_user_message should be hydrated
        assert_eq!(summaries[0].first_user_message, "msg 4");
        assert_eq!(summaries[1].first_user_message, "msg 3");
    }

    #[tokio::test]
    async fn paginate_enumerate_two_pages_ordered_disjoint_complete() {
        let page_size = 2;
        let dir = TempDir::new().expect("temp dir");

        for i in 0..5 {
            let s = Session::new(None, dir.path().to_path_buf())
                .await
                .expect("create");
            s.conversation()
                .insert_message(&Message::text(Role::User, format!("msg {i}")))
                .await
                .expect("insert");
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        let refs = enumerate_sessions(dir.path()).await.expect("enumerate");
        assert_eq!(refs.len(), 5);

        let page1 = hydrate_sessions(&refs[..page_size]).await;
        let page2 = hydrate_sessions(&refs[page_size..page_size * 2]).await;

        let page1_ids: std::collections::HashSet<&str> =
            page1.iter().map(|s| s.id.as_str()).collect();
        let page2_ids: std::collections::HashSet<&str> =
            page2.iter().map(|s| s.id.as_str()).collect();
        assert!(page1_ids.is_disjoint(&page2_ids), "pages must be disjoint");
        assert_eq!(
            page1.len() + page2.len(),
            4,
            "two pages of size 2 should cover 4 of 5 refs"
        );
    }
}
