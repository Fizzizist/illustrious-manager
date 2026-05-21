use crate::config::ConfirmationMode;
use crate::types::ContentBlock;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;

use super::{Tool, ToolError, ToolResult};

/// Executes shell commands in the sandbox directory.
///
/// # Confirmation behavior
/// - Allowlisted commands execute without confirmation regardless of `ConfirmationMode`.
/// - Denylisted commands are always rejected.
/// - All other commands follow `ConfirmationMode`. Because bash commands cannot be reliably
///   classified as read-only or write operations, `WriteOnly` is treated identically to `Always`.
///
/// # Denylist limitations
/// Segment detection splits on common shell operators (`|`, `&`, `;`, newline, `(`, backtick)
/// to catch obvious bypass patterns. It is best-effort: complex quoting, heredocs, variable
/// indirection, and other shell features can still evade detection. Confirmation policy is
/// the primary security gate; the denylist is a convenience filter.
pub struct BashTool {
    allowlist: HashSet<String>,
    denylist: HashSet<String>,
    sandbox_root: PathBuf,
    confirmation: ConfirmationMode,
    confirm_fn: Box<dyn Fn(&str) -> bool + Send + Sync>,
    schema: Value,
}

impl BashTool {
    pub fn new(
        allowlist: Vec<String>,
        denylist: Vec<String>,
        sandbox_root: PathBuf,
        confirmation: ConfirmationMode,
        confirm_fn: Box<dyn Fn(&str) -> bool + Send + Sync>,
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
            schema,
        }
    }

    /// Extracts the first token (command name) from each shell segment.
    ///
    /// Splits on `|`, `&`, `;`, newline, `(`, and backtick to catch common shell operator
    /// bypass patterns. Best-effort only — see struct-level docs.
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

        let (content, is_error) = execute_raw(command, &self.sandbox_root, None).await;

        Ok(ToolResult {
            content: vec![ContentBlock::Text(content)],
            is_error,
            agent_events: vec![],
        })
    }
}

/// Execute a shell command directly, bypassing all policy checks.
///
/// Returns `(combined_output, is_error)`. If `cancel` is cancelled while the
/// command is running, the child process is killed (via `kill_on_drop`) and
/// `("[cancelled by user]", true)` is returned.
pub async fn execute_raw(
    command: &str,
    sandbox_root: &std::path::Path,
    cancel: Option<CancellationToken>,
) -> (String, bool) {
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(sandbox_root)
        .kill_on_drop(true)
        .output();

    let outcome = if let Some(token) = cancel {
        tokio::select! {
            biased;
            _ = token.cancelled() => None,
            result = child => Some(result),
        }
    } else {
        Some(child.await)
    };

    match outcome {
        None => ("[cancelled by user]".to_string(), true),
        Some(Err(e)) => (format!("Failed to spawn command: {e}"), true),
        Some(Ok(output)) => {
            let is_error = !output.status.success();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let content = match (stdout.is_empty(), stderr.is_empty()) {
                (false, false) => format!("{}\n{}", stdout, stderr),
                (true, false) => stderr.into_owned(),
                (false, true) => stdout.into_owned(),
                (true, true) => "(no output)".to_string(),
            };
            (content, is_error)
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
        BashTool::new(
            vec!["echo", "ls", "cat"]
                .into_iter()
                .map(String::from)
                .collect(),
            vec!["rm", "sudo"].into_iter().map(String::from).collect(),
            sandbox_root,
            confirmation,
            confirm_fn,
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
        // confirm_fn panics to prove it is not called
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
        // Regression: prior to migrating to tokio::process::Command with kill_on_drop,
        // dropping the execute() future on cancellation left bash subprocesses running
        // for the full sleep duration, freezing the TUI from the user's perspective.
        //
        // Strategy: spawn `sleep 30 && touch <marker>`. Race against a 100ms timeout.
        // After timeout fires (dropping the future), wait 500ms and assert the marker
        // file does NOT exist — proving the subprocess was killed before it could run
        // the `touch`.
        let temp_dir = TempDir::new().expect("temp dir");
        let sandbox = temp_dir.path().to_path_buf();
        let marker = sandbox.join("ran_to_completion.marker");
        let marker_str = marker.to_string_lossy().to_string();

        let tool = make_tool(ConfirmationMode::Never, sandbox.clone(), Box::new(|_| true));

        let command = format!("sleep 30 && touch {}", marker_str);
        let exec_future = tool.execute(serde_json::json!({"command": command}));

        // Race the execution against a short timeout; on timeout the future is dropped.
        let outcome =
            tokio::time::timeout(std::time::Duration::from_millis(100), exec_future).await;
        assert!(
            outcome.is_err(),
            "test setup error: bash should not have completed in 100ms"
        );

        // Give the OS a moment to reap the killed child before checking the filesystem.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        assert!(
            !marker.exists(),
            "marker file at {marker_str} exists — subprocess was NOT killed when future was dropped"
        );
    }

    #[tokio::test]
    async fn execute_raw_returns_stdout_for_successful_command() {
        let temp_dir = TempDir::new().expect("temp dir");
        let (output, is_error) = execute_raw("echo hello", temp_dir.path(), None).await;
        assert!(!is_error);
        assert!(output.contains("hello"));
    }

    #[tokio::test]
    async fn execute_raw_reports_nonzero_exit_as_error() {
        let temp_dir = TempDir::new().expect("temp dir");
        let (output, is_error) = execute_raw("exit 1", temp_dir.path(), None).await;
        assert!(is_error, "non-zero exit should be is_error=true");
        let _ = output; // content may vary
    }

    #[tokio::test]
    async fn execute_raw_combines_stdout_and_stderr() {
        let temp_dir = TempDir::new().expect("temp dir");
        let (output, _) = execute_raw("echo out; echo err >&2", temp_dir.path(), None).await;
        assert!(output.contains("out"), "should contain stdout");
        assert!(output.contains("err"), "should contain stderr");
    }

    #[tokio::test]
    async fn execute_raw_cancellation_kills_child() {
        use tokio_util::sync::CancellationToken;
        let temp_dir = TempDir::new().expect("temp dir");
        let token = CancellationToken::new();
        let cancel_clone = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });
        let (output, is_error) = execute_raw("sleep 30", temp_dir.path(), Some(token)).await;
        assert!(is_error, "cancelled command should be is_error=true");
        assert!(
            output.contains("cancelled"),
            "cancelled command should mention cancellation, got: {output}"
        );
    }
}
