use crate::config::ConfirmationMode;
use crate::types::ContentBlock;
use async_trait::async_trait;
use nix::pty::openpty;
use serde_json::Value;
use std::collections::HashSet;
use std::os::fd::{AsRawFd, FromRawFd};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::{Tool, ToolError, ToolResult};

#[cfg(target_os = "linux")]
const TIOCSCTTY: u64 = 0x540E;

pub struct BashTool {
    allowlist: HashSet<String>,
    denylist: HashSet<String>,
    sandbox_root: PathBuf,
    confirmation: ConfirmationMode,
    confirm_fn: Box<dyn Fn(&str) -> bool + Send + Sync>,
    bash_timeout_secs: Option<u64>,
    schema: Value,
}

impl BashTool {
    pub fn new(
        allowlist: Vec<String>,
        denylist: Vec<String>,
        sandbox_root: PathBuf,
        confirmation: ConfirmationMode,
        confirm_fn: Box<dyn Fn(&str) -> bool + Send + Sync>,
        bash_timeout_secs: Option<u64>,
    ) -> Self {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        });
        Self {
            allowlist: allowlist.into_iter().collect(),
            denylist: denylist.into_iter().collect(),
            sandbox_root,
            confirmation,
            confirm_fn,
            bash_timeout_secs,
            schema,
        }
    }

    fn shell_command_tokens(command: &str) -> Vec<&str> {
        command
            .split(['|', '&', ';', '\n', '(', '`'])
            .filter_map(|segment| segment.split_whitespace().next())
            .collect()
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute shell commands in the sandbox directory"
    }

    fn input_schema(&self) -> &Value {
        &self.schema
    }

    fn markdown_input(&self, input: &Value) -> String {
        let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
        format!("```sh\n{}\n```", command)
    }

    fn markdown_output(&self, result: &ToolResult) -> String {
        let text = result
            .content
            .iter()
            .filter_map(|b| {
                if let crate::types::ContentBlock::Text(s) = b {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("```\n{}\n```", text)
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput {
                message: "missing required field 'command'".to_string(),
            })?;

        let tokens = Self::shell_command_tokens(command);

        for token in &tokens {
            if self.denylist.contains(*token) {
                return Ok(ToolResult {
                    content: vec![ContentBlock::Text(format!(
                        "Command rejected: '{}' is not allowed",
                        token
                    ))],
                    is_error: true,
                    agent_events: vec![],
                });
            }
        }

        let all_allowlisted = tokens.iter().all(|token| self.allowlist.contains(*token));

        if !all_allowlisted {
            let needs_confirmation = matches!(
                self.confirmation,
                ConfirmationMode::Always | ConfirmationMode::WriteOnly
            );

            if needs_confirmation && !(self.confirm_fn)(command) {
                return Ok(ToolResult {
                    content: vec![ContentBlock::Text(
                        "Command rejected: user denied confirmation".to_string(),
                    )],
                    is_error: true,
                    agent_events: vec![],
                });
            }
        }

        run_pty_command(
            command,
            &self.sandbox_root,
            self.bash_timeout_secs
                .and_then(|s| if s == 0 { None } else { Some(s) }),
        )
        .await
    }
}

async fn run_pty_command(
    command: &str,
    cwd: &PathBuf,
    timeout_secs: Option<u64>,
) -> Result<ToolResult, ToolError> {
    let pty = openpty(None, None).map_err(|e| ToolError::Execution {
        tool_name: "bash".to_string(),
        message: format!("Failed to open PTY: {}", e),
    })?;

    let slave_for_preexec = pty.slave.as_raw_fd();

    let slave_stdout = nix::unistd::dup(&pty.slave).map_err(|e| ToolError::Execution {
        tool_name: "bash".to_string(),
        message: format!("Failed to dup slave fd for stdout: {}", e),
    })?;
    let slave_stderr = nix::unistd::dup(&pty.slave).map_err(|e| ToolError::Execution {
        tool_name: "bash".to_string(),
        message: format!("Failed to dup slave fd for stderr: {}", e),
    })?;

    let master_raw = pty.master.as_raw_fd();
    // Prevent OwnedFd::drop from closing master fd — tokio::fs::File will own it.
    std::mem::forget(pty.master);

    let mut child = {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(command)
            .current_dir(cwd)
            .kill_on_drop(true)
            .stdin(Stdio::from(pty.slave))
            .stdout(Stdio::from(slave_stdout))
            .stderr(Stdio::from(slave_stderr));

        // Safety: pre_exec runs in the child between fork and exec.
        // setsid() creates a new session and process group; TIOCSCTTY sets
        // the controlling terminal for the PTY.
        unsafe {
            cmd.pre_exec(move || {
                nix::unistd::setsid()
                    .map_err(|e| std::io::Error::other(format!("setsid failed: {e}")))?;
                #[cfg(target_os = "linux")]
                {
                    let ret = nix::libc::ioctl(slave_for_preexec, TIOCSCTTY as _, 0);
                    if ret < 0 {
                        return Err(std::io::Error::other("TIOCSCTTY ioctl failed"));
                    }
                }
                Ok(())
            });
        }

        cmd.spawn().map_err(|e| ToolError::Execution {
            tool_name: "bash".to_string(),
            message: format!("Failed to spawn command: {}", e),
        })?
    };

    let output_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let output_buf_clone = Arc::clone(&output_buf);

    let master_file = tokio::fs::File::from_std(unsafe { std::fs::File::from_raw_fd(master_raw) });

    let read_future = read_master_into_buffer(master_file, output_buf_clone);

    let output_result = match timeout_secs {
        Some(secs) => tokio::time::timeout(Duration::from_secs(secs), read_future).await,
        None => {
            read_future.await;
            Ok(())
        }
    };

    match output_result {
        Ok(()) => {
            let status = child.wait().await;
            let is_error = status.as_ref().is_ok_and(|s| !s.success());
            // master fd is closed when master_file is dropped inside read_master_into_buffer
            let output = String::from_utf8_lossy(&output_buf.lock().expect("lock")).into_owned();
            let content = if output.trim().is_empty() {
                "(no output)".to_string()
            } else {
                output
            };
            Ok(ToolResult {
                content: vec![ContentBlock::Text(content)],
                is_error,
                agent_events: vec![],
            })
        }
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            // master fd is closed when master_file is dropped on timeout
            let partial = String::from_utf8_lossy(&output_buf.lock().expect("lock")).into_owned();
            let timeout_msg = format!(
                "bash timed out after {} seconds. Partial output:\n{}",
                timeout_secs.expect("timeout path requires Some"),
                if partial.trim().is_empty() {
                    "(no output)".to_string()
                } else {
                    partial
                }
            );
            Ok(ToolResult {
                content: vec![ContentBlock::Text(timeout_msg)],
                is_error: true,
                agent_events: vec![],
            })
        }
    }
}

async fn read_master_into_buffer(mut file: tokio::fs::File, buf: Arc<Mutex<Vec<u8>>>) {
    use tokio::io::AsyncReadExt;
    let mut tmp = [0u8; 4096];
    loop {
        match file.read(&mut tmp).await {
            Ok(0) => break,
            Ok(n) => {
                buf.lock().expect("lock").extend_from_slice(&tmp[..n]);
            }
            Err(_) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_tool(
        confirmation: ConfirmationMode,
        sandbox_root: PathBuf,
        confirm_fn: Box<dyn Fn(&str) -> bool + Send + Sync>,
    ) -> BashTool {
        make_tool_with_timeout(confirmation, sandbox_root, confirm_fn, None)
    }

    fn make_tool_with_timeout(
        confirmation: ConfirmationMode,
        sandbox_root: PathBuf,
        confirm_fn: Box<dyn Fn(&str) -> bool + Send + Sync>,
        bash_timeout_secs: Option<u64>,
    ) -> BashTool {
        BashTool::new(
            vec!["echo", "ls", "cat"]
                .into_iter()
                .map(String::from)
                .collect(),
            vec!["rm", "sudo"].into_iter().map(String::from).collect(),
            sandbox_root,
            confirmation,
            confirm_fn,
            bash_timeout_secs,
        )
    }

    #[tokio::test]
    async fn allowlisted_command_executes_without_confirmation() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Always,
            temp_dir.path().to_path_buf(),
            Box::new(|_| panic!("confirm_fn must not be called for allowlisted commands")),
        );
        let result = tool
            .execute(serde_json::json!({"command": "echo hello"}))
            .await
            .expect("should succeed");
        assert!(!result.is_error);
        match &result.content[0] {
            ContentBlock::Text(text) => assert!(text.contains("hello")),
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn denylisted_command_is_rejected() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"command": "rm -rf /"}))
            .await
            .expect("execute must not err");
        assert!(result.is_error);
        match &result.content[0] {
            ContentBlock::Text(text) => assert!(text.contains("rm")),
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn unlisted_command_with_never_policy_executes() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| panic!("confirm_fn must not be called with Never policy")),
        );
        let result = tool
            .execute(serde_json::json!({"command": "pwd"}))
            .await
            .expect("should succeed");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn unlisted_command_with_always_policy_and_approved_executes() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Always,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"command": "pwd"}))
            .await
            .expect("should succeed");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn unlisted_command_with_always_policy_and_denied_returns_error() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Always,
            temp_dir.path().to_path_buf(),
            Box::new(|_| false),
        );
        let result = tool
            .execute(serde_json::json!({"command": "pwd"}))
            .await
            .expect("execute must not err");
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn unlisted_command_with_write_only_policy_and_denied_returns_error() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::WriteOnly,
            temp_dir.path().to_path_buf(),
            Box::new(|_| false),
        );
        let result = tool
            .execute(serde_json::json!({"command": "pwd"}))
            .await
            .expect("execute must not err");
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn piped_command_with_denylisted_segment_is_rejected() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"command": "cat file.txt | rm -rf /"}))
            .await
            .expect("execute must not err");
        assert!(result.is_error);
        match &result.content[0] {
            ContentBlock::Text(text) => assert!(text.contains("rm")),
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn piped_command_with_all_allowlisted_segments_executes_without_confirmation() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Always,
            temp_dir.path().to_path_buf(),
            Box::new(|_| panic!("confirm_fn must not be called when all segments allowlisted")),
        );
        let result = tool
            .execute(serde_json::json!({"command": "echo hello | cat"}))
            .await
            .expect("should succeed");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn command_runs_with_sandbox_root_as_cwd() {
        let temp_dir = TempDir::new().expect("temp dir");
        let sandbox = temp_dir.path().canonicalize().expect("canonicalize");
        let tool = make_tool(ConfirmationMode::Never, sandbox.clone(), Box::new(|_| true));
        let result = tool
            .execute(serde_json::json!({"command": "pwd"}))
            .await
            .expect("should succeed");
        assert!(!result.is_error);
        match &result.content[0] {
            ContentBlock::Text(text) => {
                assert_eq!(
                    text.trim(),
                    sandbox.to_string_lossy(),
                    "cwd should be sandbox root"
                );
            }
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn command_failure_returns_is_error_true_with_stderr() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"command": "echo 'err msg' >&2; exit 1"}))
            .await
            .expect("execute must not err");
        assert!(result.is_error);
        match &result.content[0] {
            ContentBlock::Text(text) => assert!(text.contains("err msg")),
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn missing_command_field_returns_invalid_input_error() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"not_command": "echo hello"}))
            .await;
        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    #[tokio::test]
    async fn and_operator_denylist_bypass_is_blocked() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"command": "echo hello && rm -rf /"}))
            .await
            .expect("execute must not err");
        assert!(
            result.is_error,
            "&&-separated denylist command must be blocked"
        );
    }

    #[tokio::test]
    async fn semicolon_denylist_bypass_is_blocked() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"command": "echo hello; rm -rf /"}))
            .await
            .expect("execute must not err");
        assert!(
            result.is_error,
            "semicolon-separated denylist command must be blocked"
        );
    }

    #[tokio::test]
    async fn newline_denylist_bypass_is_blocked() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"command": "echo hello\nrm -rf /"}))
            .await
            .expect("execute must not err");
        assert!(
            result.is_error,
            "newline-separated denylist command must be blocked"
        );
    }

    #[tokio::test]
    async fn subshell_denylist_bypass_is_blocked() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"command": "echo $(rm -rf /)"}))
            .await
            .expect("execute must not err");
        assert!(
            result.is_error,
            "$() subshell denylist command must be blocked"
        );
    }

    #[tokio::test]
    async fn backtick_denylist_bypass_is_blocked() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"command": "echo `rm -rf /`"}))
            .await
            .expect("execute must not err");
        assert!(
            result.is_error,
            "backtick subshell denylist command must be blocked"
        );
    }

    #[tokio::test]
    async fn silent_command_produces_no_output_sentinel() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"command": "true"}))
            .await
            .expect("should succeed");
        assert!(!result.is_error);
        match &result.content[0] {
            ContentBlock::Text(text) => assert!(
                !text.is_empty(),
                "silent command should produce sentinel, not empty string"
            ),
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn markdown_input_wraps_command_in_sh_code_block() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let md = tool.markdown_input(&serde_json::json!({"command": "ls -la"}));
        assert!(md.contains("```sh"), "should wrap in sh code block");
        assert!(md.contains("ls -la"), "should include command");
    }

    #[tokio::test]
    async fn markdown_output_wraps_result_in_code_block() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"command": "echo hello"}))
            .await
            .expect("should succeed");
        let md = tool.markdown_output(&result);
        assert!(md.contains("```"), "should wrap in code block");
        assert!(md.contains("hello"), "should include output");
    }

    #[tokio::test]
    async fn dropping_execute_future_kills_long_running_subprocess() {
        let temp_dir = TempDir::new().expect("temp dir");
        let sandbox = temp_dir.path().to_path_buf();
        let marker = sandbox.join("ran_to_completion.marker");
        let marker_str = marker.to_string_lossy().to_string();

        let tool = make_tool(ConfirmationMode::Never, sandbox.clone(), Box::new(|_| true));

        let command = format!("sleep 30 && touch {}", marker_str);
        let exec_future = tool.execute(serde_json::json!({"command": command}));

        let outcome =
            tokio::time::timeout(std::time::Duration::from_millis(100), exec_future).await;
        assert!(
            outcome.is_err(),
            "test setup error: bash should not have completed in 100ms"
        );

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        assert!(
            !marker.exists(),
            "marker file at {marker_str} exists — subprocess was NOT killed when future was dropped"
        );
    }

    // ── PTY and timeout tests ────────────────────────────────────────────

    #[tokio::test]
    async fn command_exceeding_timeout_returns_timeout_error() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool_with_timeout(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
            Some(1),
        );
        let result = tool
            .execute(serde_json::json!({"command": "sleep 300"}))
            .await
            .expect("should succeed");
        assert!(result.is_error);
        match &result.content[0] {
            ContentBlock::Text(text) => assert!(
                text.to_lowercase().contains("timed out"),
                "expected timeout message, got: {text}"
            ),
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn timeout_captures_partial_output() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool_with_timeout(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
            Some(2),
        );
        let result = tool
            .execute(serde_json::json!({"command": "echo 'partial output here'; sleep 300"}))
            .await
            .expect("should succeed");
        assert!(result.is_error);
        match &result.content[0] {
            ContentBlock::Text(text) => {
                assert!(
                    text.contains("partial output here"),
                    "timeout result should contain partial output, got: {text}"
                );
                assert!(
                    text.to_lowercase().contains("timed out"),
                    "expected timeout message, got: {text}"
                );
            }
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn null_timeout_allows_long_running_command() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool_with_timeout(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
            None,
        );
        let result = tool
            .execute(serde_json::json!({"command": "sleep 2"}))
            .await
            .expect("should succeed");
        assert!(
            !result.is_error,
            "long-running command with no timeout should succeed"
        );
    }

    #[tokio::test]
    async fn successful_command_within_timeout_works_normally() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool_with_timeout(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
            Some(30),
        );
        let result = tool
            .execute(serde_json::json!({"command": "echo hello"}))
            .await
            .expect("should succeed");
        assert!(!result.is_error);
        match &result.content[0] {
            ContentBlock::Text(text) => assert!(text.contains("hello")),
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn pty_provides_tty_to_child() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
        );
        let result = tool
            .execute(serde_json::json!({"command": "tty"}))
            .await
            .expect("should succeed");
        assert!(!result.is_error, "tty command should succeed under PTY");
        match &result.content[0] {
            ContentBlock::Text(text) => assert!(
                text.contains("/dev/pts/"),
                "tty output should be a PTY path, got: {text}"
            ),
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn interactive_command_via_pty_does_not_hang() {
        let temp_dir = TempDir::new().expect("temp dir");
        let tool = make_tool_with_timeout(
            ConfirmationMode::Never,
            temp_dir.path().to_path_buf(),
            Box::new(|_| true),
            Some(5),
        );
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tool.execute(serde_json::json!({"command": "test -t 0 && echo YES || echo NO"})),
        )
        .await
        .expect("command should not hang")
        .expect("should succeed");
        assert!(
            !result.is_error,
            "interactive command via PTY should not error"
        );
        match &result.content[0] {
            ContentBlock::Text(text) => assert!(
                text.contains("YES"),
                "stdin should be a TTY under PTY, got: {text}"
            ),
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn kill_on_drop_still_works_with_pty() {
        let temp_dir = TempDir::new().expect("temp dir");
        let sandbox = temp_dir.path().to_path_buf();
        let marker = sandbox.join("pty_kill_test.marker");
        let marker_str = marker.to_string_lossy().to_string();

        let tool = make_tool(ConfirmationMode::Never, sandbox.clone(), Box::new(|_| true));

        let command = format!("sleep 30 && touch {}", marker_str);
        let exec_future = tool.execute(serde_json::json!({"command": command}));

        let outcome =
            tokio::time::timeout(std::time::Duration::from_millis(100), exec_future).await;
        assert!(
            outcome.is_err(),
            "test setup error: bash should not have completed in 100ms"
        );

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        assert!(
            !marker.exists(),
            "marker file exists — PTY subprocess was NOT killed when future was dropped"
        );
    }
}
