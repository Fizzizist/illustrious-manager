#[cfg(unix)]
use std::io::{Read, Write};
#[cfg(unix)]
use std::net::TcpListener;
use std::process::Command;

struct Sandbox {
    _dir: tempfile::TempDir,
    config_path: std::path::PathBuf,
    sessions_dir: std::path::PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let sessions_dir = dir.path().join("sessions");
        let config_path = dir.path().join("config.toml");
        let config = format!(
            "sessions_dir = {:?}\n\n[vertex]\nproject = \"test-project\"\nregion = \"us-east5\"\nmodel = \"claude-sonnet-4-20250514\"\n",
            sessions_dir.to_string_lossy(),
        );
        std::fs::write(&config_path, config).expect("write config");
        Self {
            _dir: dir,
            config_path,
            sessions_dir,
        }
    }

    #[cfg(unix)]
    fn with_openai_compat(port: u16) -> Self {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let sessions_dir = dir.path().join("sessions");
        let config_path = dir.path().join("config.toml");
        let config = format!(
            "backend = \"openai_compat\"\nsessions_dir = {:?}\n\n[vertex]\nproject = \"\"\n\n[openai_compat]\nbase_url = \"http://127.0.0.1:{port}/v1\"\nmodel = \"test-model\"\nreasoning = \"none\"\n",
            sessions_dir.to_string_lossy(),
        );
        std::fs::write(&config_path, config).expect("write config");
        Self {
            _dir: dir,
            config_path,
            sessions_dir,
        }
    }

    fn run(&self, args: &[&str]) -> std::process::Output {
        Command::new(env!("CARGO_BIN_EXE_illustrious-manager"))
            .arg("--config")
            .arg(&self.config_path)
            .args(args)
            .current_dir(self._dir.path())
            .env_remove("HOME")
            .output()
            .expect("spawn binary")
    }
}

/// Acceptance: a typo'd role must fail fast — before any session DB is
/// created — with an error that names the role and lists the alternatives.
#[test]
fn unknown_role_fails_without_creating_session_db() {
    let sandbox = Sandbox::new();
    let output = sandbox.run(&[
        "--role",
        "nonexistent",
        "--session-id",
        "01923456-7890-7abc-def0-123456789abc",
    ]);

    assert!(
        !output.status.success(),
        "unknown role must exit non-zero: stderr={:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("nonexistent"),
        "stderr should name the unknown role: {stderr}"
    );
    assert!(
        stderr.contains("Available roles: default"),
        "stderr should list available roles: {stderr}"
    );
    assert!(
        !sandbox
            .sessions_dir
            .join("01923456-7890-7abc-def0-123456789abc.db")
            .exists(),
        "no session DB may be created for a rejected role"
    );
}

#[test]
fn continue_and_session_id_conflict_exits_nonzero() {
    let sandbox = Sandbox::new();
    let output = sandbox.run(&[
        "--continue",
        "--session-id",
        "01923456-7890-7abc-def0-123456789abc",
    ]);

    assert!(
        !output.status.success(),
        "conflicting --continue and --session-id must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--continue"),
        "stderr should mention --continue: {stderr}"
    );
    assert!(
        stderr.contains("--session-id"),
        "stderr should mention --session-id: {stderr}"
    );
}

#[test]
fn continue_without_last_file_fails_with_informative_error_and_no_db() {
    let sandbox = Sandbox::new();
    let output = sandbox.run(&["--continue"]);

    assert!(
        !output.status.success(),
        "--continue without a prior session must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stderr_lower = stderr.to_lowercase();
    assert!(!stderr.trim().is_empty(), "stderr should not be empty");
    assert!(
        stderr_lower.contains("session")
            || stderr_lower.contains("previous")
            || stderr_lower.contains("unsupported"),
        "stderr should explain missing previous session: {stderr}"
    );

    if sandbox.sessions_dir.exists() {
        let db_count = std::fs::read_dir(&sandbox.sessions_dir)
            .expect("read sessions_dir")
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().and_then(|e| e.to_str()) == Some("db"))
            .count();
        assert_eq!(
            db_count, 0,
            "no session DB should be created on --continue failure"
        );
    }
}

#[cfg(unix)]
#[test]
fn continue_with_deleted_linked_session_fails_without_recreating_db() {
    let sandbox = Sandbox::new();
    std::fs::create_dir_all(&sandbox.sessions_dir).expect("create sessions_dir");
    let sid = illustrious_manager::session::last::current_terminal_session_id().expect("unix sid");
    let missing_id = uuid::Uuid::now_v7().to_string();
    illustrious_manager::session::last::write_last_session(&sandbox.sessions_dir, sid, &missing_id)
        .expect("write last");

    let output = sandbox.run(&["--continue"]);
    assert!(
        !output.status.success(),
        "--continue with missing linked session must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&missing_id),
        "stderr should mention missing session id {missing_id}: {stderr}"
    );
    assert!(
        !sandbox
            .sessions_dir
            .join(format!("{missing_id}.db"))
            .exists(),
        "missing session DB must not be recreated"
    );
}

#[cfg(unix)]
fn sse_reply_response(reply: &str) -> String {
    let body = format!(
        "data: {}\n\ndata: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":10,\"completion_tokens\":5}}}}\n\ndata: [DONE]\n\n",
        serde_json::json!({
            "choices": [{
                "delta": { "content": reply },
                "finish_reason": null
            }]
        })
    );
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
}

#[cfg(unix)]
fn read_http_request(stream: &mut std::net::TcpStream) {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .expect("set read timeout");
    let mut received = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        let n = stream.read(&mut buf).expect("read request");
        if n == 0 {
            break;
        }
        received.extend_from_slice(&buf[..n]);
        let text = String::from_utf8_lossy(&received);
        if let Some(head_end) = text.find("\r\n\r\n") {
            let content_length = text[..head_end]
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .and_then(|v| v.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            if received.len() >= head_end + 4 + content_length {
                break;
            }
        }
    }
}

#[cfg(unix)]
fn extract_session_id(stderr: &str) -> String {
    stderr
        .lines()
        .find_map(|line| line.strip_prefix("Session ID: "))
        .map(str::trim)
        .map(str::to_string)
        .unwrap_or_else(|| panic!("expected 'Session ID: ...' line in stderr: {stderr}"))
}

#[cfg(unix)]
fn message_contains_text(message: &illustrious_manager::types::Message, needle: &str) -> bool {
    message.content.iter().any(|block| match block {
        illustrious_manager::types::ContentBlock::Text(text) => text.contains(needle),
        _ => false,
    })
}

#[cfg(unix)]
#[tokio::test]
async fn single_shot_writes_last_and_continue_resumes_same_session() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();

    let server_handle = std::thread::spawn(move || {
        for reply in ["First reply", "Second reply"] {
            let (mut stream, _) = listener.accept().expect("accept connection");
            read_http_request(&mut stream);
            let response = sse_reply_response(reply);
            stream
                .write_all(response.as_bytes())
                .expect("write response");
            stream.flush().expect("flush response");
            let _ = stream.shutdown(std::net::Shutdown::Both);
        }
    });

    let sandbox = Sandbox::with_openai_compat(port);
    let sid = illustrious_manager::session::last::current_terminal_session_id().expect("unix sid");
    let last_dir = sandbox.sessions_dir.join(".last");
    std::fs::create_dir_all(&last_dir).expect("create .last dir");
    let dead_path = last_dir.join(u32::MAX.to_string());
    std::fs::write(&dead_path, "dead-session").expect("seed dead file");

    let output1 = sandbox.run(&["--single-shot", "hello one"]);
    let stderr1 = String::from_utf8_lossy(&output1.stderr);
    assert!(output1.status.success(), "run 1 failed: stderr={stderr1:?}");
    let session_id_1 = extract_session_id(&stderr1);
    assert_eq!(
        illustrious_manager::session::last::resolve_last_session(&sandbox.sessions_dir, sid)
            .expect("resolve after run 1"),
        session_id_1
    );
    assert!(!dead_path.exists(), "dead .last entry should be pruned");

    let output2 = sandbox.run(&["--continue", "--single-shot", "hello two"]);
    let stderr2 = String::from_utf8_lossy(&output2.stderr);
    assert!(output2.status.success(), "run 2 failed: stderr={stderr2:?}");
    let session_id_2 = extract_session_id(&stderr2);
    assert_eq!(session_id_2, session_id_1);
    assert_eq!(
        illustrious_manager::session::last::resolve_last_session(&sandbox.sessions_dir, sid)
            .expect("resolve after run 2"),
        session_id_1
    );

    server_handle.join().expect("mock server thread");

    let session = illustrious_manager::session::Session::new(
        Some(session_id_1),
        sandbox.sessions_dir.clone(),
    )
    .await
    .expect("open resumed session");
    let history = session
        .conversation()
        .load_history()
        .await
        .expect("load history");
    assert_eq!(history.len(), 4, "expected 4 messages, got: {history:?}");
    assert!(message_contains_text(&history[0], "hello one"));
    assert!(message_contains_text(&history[1], "First reply"));
    assert!(message_contains_text(&history[2], "hello two"));
    assert!(message_contains_text(&history[3], "Second reply"));
}
