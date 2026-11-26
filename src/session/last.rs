use anyhow::{Context, Result};
use std::path::Path;

const LAST_DIR_NAME: &str = ".last";

#[cfg(unix)]
pub fn current_terminal_session_id() -> Result<u32> {
    let sid = unsafe { libc::getsid(0) };
    if sid <= 0 {
        return Err(std::io::Error::last_os_error())
            .context("Failed to determine current terminal session ID");
    }
    u32::try_from(sid).context("Terminal session ID out of range")
}

#[cfg(not(unix))]
pub fn current_terminal_session_id() -> Result<u32> {
    anyhow::bail!("--continue is unsupported on non-Unix platforms")
}

#[cfg(unix)]
pub fn is_terminal_session_alive(sid: u32) -> bool {
    if sid == 0 {
        return false;
    }
    // Keep our own terminal session alive even when its original session leader exited.
    if current_terminal_session_id().ok() == Some(sid) {
        return true;
    }
    let Ok(pid) = libc::pid_t::try_from(sid) else {
        return false;
    };
    let res = unsafe { libc::getsid(pid) };
    if res == pid {
        return true;
    }
    if res == -1 {
        return std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
    }
    false
}

#[cfg(not(unix))]
pub fn is_terminal_session_alive(_sid: u32) -> bool {
    false
}

pub fn write_last_session(sessions_dir: &Path, terminal_sid: u32, session_id: &str) -> Result<()> {
    super::validate_uuidv7(session_id)?;
    let last_dir = sessions_dir.join(LAST_DIR_NAME);
    std::fs::create_dir_all(&last_dir)
        .with_context(|| format!("Failed to create {}", last_dir.display()))?;
    let path = last_dir.join(terminal_sid.to_string());
    std::fs::write(&path, session_id)
        .with_context(|| format!("Failed to write last session pointer at {}", path.display()))?;
    Ok(())
}

pub fn resolve_last_session(sessions_dir: &Path, terminal_sid: u32) -> Result<String> {
    let path = sessions_dir
        .join(LAST_DIR_NAME)
        .join(terminal_sid.to_string());
    let raw = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            anyhow::bail!(
                "No previous session found for this terminal (expected {})",
                path.display()
            );
        }
        Err(e) => {
            return Err(e).with_context(|| {
                format!("Failed to read last session pointer at {}", path.display())
            });
        }
    };

    let session_id = raw.trim().to_string();
    super::validate_uuidv7(&session_id).with_context(|| {
        format!(
            "Invalid session ID in last session pointer at {}",
            path.display()
        )
    })?;

    let db_path = super::session_db_path(sessions_dir, &session_id);
    if !db_path.is_file() {
        anyhow::bail!(
            "Previous session '{}' no longer exists at {}",
            session_id,
            db_path.display()
        );
    }

    Ok(session_id)
}

pub fn prune_last_sessions_with<F>(sessions_dir: &Path, mut is_alive: F) -> Result<usize>
where
    F: FnMut(u32) -> bool,
{
    let last_dir = sessions_dir.join(LAST_DIR_NAME);
    let read_dir = match std::fs::read_dir(&last_dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => {
            return Err(e).with_context(|| format!("Failed to read {}", last_dir.display()));
        }
    };

    let mut removed = 0;
    for entry in read_dir {
        let entry = entry.context("Failed to read .last directory entry")?;
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if file_type.is_dir() {
            continue;
        }

        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let Ok(sid) = name.parse::<u32>() else {
            continue;
        };
        if sid == 0 || is_alive(sid) {
            continue;
        }

        std::fs::remove_file(entry.path()).with_context(|| {
            format!(
                "Failed to remove stale last-session pointer {}",
                entry.path().display()
            )
        })?;
        removed += 1;
    }

    Ok(removed)
}

pub fn prune_dead_last_sessions(sessions_dir: &Path) -> Result<usize> {
    prune_last_sessions_with(sessions_dir, is_terminal_session_alive)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Session, enumerate_sessions};
    use tempfile::TempDir;

    #[tokio::test]
    async fn write_and_resolve_round_trip() {
        let dir = TempDir::new().expect("temp dir");
        let session1 = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create session1");
        let session2 = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create session2");

        write_last_session(dir.path(), 4242, &session1.id).expect("write session1");
        let resolved1 = resolve_last_session(dir.path(), 4242).expect("resolve session1");
        assert_eq!(resolved1, session1.id);

        write_last_session(dir.path(), 4242, &session2.id).expect("overwrite with session2");
        let resolved2 = resolve_last_session(dir.path(), 4242).expect("resolve session2");
        assert_eq!(resolved2, session2.id);

        let last_file = dir.path().join(".last").join("4242");
        std::fs::write(&last_file, format!("  {}\n", session1.id))
            .expect("write whitespace-padded session1 id");
        let resolved_trimmed =
            resolve_last_session(dir.path(), 4242).expect("resolve trimmed session1");
        assert_eq!(resolved_trimmed, session1.id);
    }

    #[test]
    fn write_rejects_invalid_session_id() {
        let dir = TempDir::new().expect("temp dir");
        let v4 = "550e8400-e29b-41d4-a716-446655440000";

        assert!(write_last_session(dir.path(), 1111, "not-a-uuid").is_err());
        assert!(!dir.path().join(".last").join("1111").exists());

        assert!(write_last_session(dir.path(), 2222, v4).is_err());
        assert!(!dir.path().join(".last").join("2222").exists());
    }

    #[test]
    fn resolve_missing_file_errors_informatively() {
        let dir = TempDir::new().expect("temp dir");
        let err = resolve_last_session(dir.path(), 9999)
            .expect_err("resolving missing last session should fail");
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("session") && (msg.contains("previous") || msg.contains("terminal")),
            "error message should mention no previous/terminal session, got: {msg}"
        );
    }

    #[test]
    fn resolve_corrupt_content_errors_informatively() {
        let dir = TempDir::new().expect("temp dir");
        let last_dir = dir.path().join(".last");
        std::fs::create_dir_all(&last_dir).expect("create .last dir");

        let cases = [
            ("101", ""),
            ("102", "   \n"),
            ("103", "not-a-uuid"),
            ("104", "550e8400-e29b-41d4-a716-446655440000"),
        ];

        for (sid_str, content) in cases {
            std::fs::write(last_dir.join(sid_str), content).expect("write corrupt last file");
            let sid: u32 = sid_str.parse().expect("valid u32 sid");
            assert!(
                resolve_last_session(dir.path(), sid).is_err(),
                "expected error for corrupt content {content:?}"
            );
        }
    }

    #[tokio::test]
    async fn resolve_missing_session_db_errors_informatively() {
        let dir = TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create session");
        let id = session.id.clone();

        write_last_session(dir.path(), 5555, &id).expect("write last session");
        session.delete_db().expect("delete session db");

        let err = resolve_last_session(dir.path(), 5555)
            .expect_err("resolving deleted session db should fail");
        assert!(
            err.to_string().contains(&id),
            "error message should mention missing session id {id}, got: {err}"
        );
        assert!(
            !dir.path().join(format!("{id}.db")).exists(),
            "session db should still not exist"
        );
    }

    #[test]
    fn prune_removes_dead_and_preserves_alive_and_non_pid_entries() {
        let dir = TempDir::new().expect("temp dir");

        let empty_count = prune_last_sessions_with(dir.path(), |_| false)
            .expect("pruning with no .last dir should succeed");
        assert_eq!(empty_count, 0);

        let last_dir = dir.path().join(".last");
        std::fs::create_dir_all(&last_dir).expect("create .last dir");

        std::fs::write(last_dir.join("1001"), "dead-1").expect("write 1001");
        std::fs::write(last_dir.join("1002"), "dead-2").expect("write 1002");
        std::fs::write(last_dir.join("2001"), "alive").expect("write 2001");
        std::fs::write(last_dir.join("0"), "zero").expect("write 0");
        std::fs::write(last_dir.join("not-a-pid"), "non-numeric").expect("write not-a-pid");
        std::fs::create_dir(last_dir.join("subdir")).expect("create subdir");
        std::fs::create_dir(last_dir.join("3003")).expect("create numeric subdir");

        let removed =
            prune_last_sessions_with(dir.path(), |sid| sid == 2001).expect("prune dead entries");
        assert_eq!(removed, 2);

        assert!(!last_dir.join("1001").exists());
        assert!(!last_dir.join("1002").exists());
        assert!(last_dir.join("2001").exists());
        assert!(last_dir.join("0").exists());
        assert!(last_dir.join("not-a-pid").exists());
        assert!(last_dir.join("subdir").is_dir());
        assert!(last_dir.join("3003").is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn unix_current_sid_is_positive_and_alive() {
        let sid = current_terminal_session_id().expect("current sid");
        assert!(sid > 0);
        assert!(is_terminal_session_alive(sid));
        assert!(!is_terminal_session_alive(0));
        assert!(!is_terminal_session_alive(u32::MAX));

        let dir = TempDir::new().expect("temp dir");
        let valid_id = uuid::Uuid::now_v7().to_string();
        write_last_session(dir.path(), sid, &valid_id).expect("write alive last session");

        let dead_sid_file = dir.path().join(".last").join(u32::MAX.to_string());
        std::fs::write(&dead_sid_file, &valid_id).expect("write dead sid file");

        let pruned = prune_dead_last_sessions(dir.path()).expect("prune dead last sessions");
        assert_eq!(pruned, 1);
        assert!(dir.path().join(".last").join(sid.to_string()).exists());
        assert!(!dead_sid_file.exists());
    }

    #[tokio::test]
    async fn enumerate_sessions_ignores_last_directory() {
        let dir = TempDir::new().expect("temp dir");
        let session = Session::new(None, dir.path().to_path_buf())
            .await
            .expect("create session");

        write_last_session(dir.path(), 1234, &session.id).expect("write last session");
        let dummy_uuid = uuid::Uuid::now_v7().to_string();
        std::fs::write(
            dir.path().join(".last").join(format!("{dummy_uuid}.db")),
            b"",
        )
        .expect("write dummy db inside .last");

        let refs = enumerate_sessions(dir.path())
            .await
            .expect("enumerate sessions");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].id, session.id);
    }
}
