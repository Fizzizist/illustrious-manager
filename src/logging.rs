use crate::types::AgentEvent;
use anyhow::Result;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static GLOBAL_LOGGER: OnceLock<Mutex<Logger>> = OnceLock::new();

/// Initialise the process-global logger. Call exactly once from `main`.
///
/// When `log_path` is `None` the global is left uninitialised and all free
/// functions remain silent no-ops. A second call is silently ignored (the
/// first call wins) so this never panics — duplicate-init is a programming
/// error but the policy forbids `.unwrap()`/panic.
pub fn init_global(log_path: Option<PathBuf>) -> Result<()> {
    let Some(path) = log_path else {
        return Ok(());
    };
    let logger = Logger::new(Some(path))?;
    let _ = GLOBAL_LOGGER.set(Mutex::new(logger));
    Ok(())
}

/// Returns `true` when the global logger has been initialised with a real
/// log file (i.e. `--debug` was passed). Used by callers that need to gate
/// optional output (e.g. streaming thinking text to stdout) on debug mode
/// without re-threading the `cli.debug` flag through every layer.
pub fn is_enabled() -> bool {
    GLOBAL_LOGGER.get().is_some()
}

// ── Free functions — all silent no-ops when the global is uninitialised ──────

pub fn log_user_input(input: &str) {
    with_global(|l| l.log_user_input(input));
}

pub fn log_tool_use(name: &str, input: &serde_json::Value) {
    with_global(|l| l.log_tool_use(name, input));
}

pub fn log_tool_result(name: &str, content: &str, is_error: bool) {
    with_global(|l| l.log_tool_result(name, content, is_error));
}

pub fn log_assistant_response(response: &str) {
    with_global(|l| l.log_assistant_response(response));
}

pub fn log_usage(input_tokens: u32, output_tokens: u32, stop_reason: &str) {
    with_global(|l| l.log_usage(input_tokens, output_tokens, stop_reason));
}

pub fn log_config(config: &dyn std::fmt::Debug) {
    with_global(|l| l.log_config(config));
}

pub fn log_event(event: &AgentEvent) {
    with_global(|l| l.log_event(event));
}

pub fn log_warn(message: &str) {
    with_global(|l| l.log_warn(message));
}

pub fn log_error(message: &str) {
    with_global(|l| l.log_error(message));
}

pub fn log_info(message: &str) {
    with_global(|l| l.log_info(message));
}

pub fn flush() {
    with_global(|l| l.flush());
}

/// Forward `f` to the locked global logger.
///
/// Silently absorbs three failure modes by design:
/// 1. Global uninitialised (no `--debug`) → no-op.
/// 2. Mutex poisoned by a panicking thread → no-op for all subsequent calls.
/// 3. The closure's `Result` (typically a `writeln!` `io::Error`) is dropped.
///
/// This is intentional: the entire purpose of routing diagnostics through
/// this module is to prevent stray stdout/stderr writes from corrupting the
/// TUI's alternate-screen render. Propagating an IO error from a logging
/// call would force every callsite to either swallow it or terminate the
/// app, both of which are worse than silently dropping a debug log line.
fn with_global(f: impl FnOnce(&mut Logger) -> Result<()>) {
    if let Some(mutex) = GLOBAL_LOGGER.get()
        && let Ok(mut guard) = mutex.lock()
    {
        let _ = f(&mut guard);
    }
}

// ── Logger struct (stays constructible for tests) ─────────────────────────────

pub struct Logger {
    log_file: Option<BufWriter<File>>,
}

impl Logger {
    pub fn new(log_path: Option<PathBuf>) -> Result<Self> {
        let log_file = match log_path {
            Some(path) => {
                let file = File::create(&path)?;
                Some(BufWriter::new(file))
            }
            None => None,
        };
        Ok(Self { log_file })
    }

    pub fn flush(&mut self) -> Result<()> {
        if let Some(ref mut writer) = self.log_file {
            writer.flush()?;
        }
        Ok(())
    }

    pub fn log_user_input(&mut self, input: &str) -> Result<()> {
        if let Some(ref mut writer) = self.log_file {
            writeln!(writer, "[USER INPUT]")?;
            writeln!(writer, "{}", input)?;
            writeln!(writer)?;
        }
        Ok(())
    }

    pub fn log_tool_use(&mut self, name: &str, input: &serde_json::Value) -> Result<()> {
        if let Some(ref mut writer) = self.log_file {
            writeln!(writer, "[TOOL CALL]")?;
            writeln!(writer, "name: {}", name)?;
            writeln!(writer, "input: {}", serde_json::to_string(input)?)?;
            writeln!(writer)?;
        }
        Ok(())
    }

    pub fn log_tool_result(&mut self, name: &str, content: &str, is_error: bool) -> Result<()> {
        if let Some(ref mut writer) = self.log_file {
            writeln!(writer, "[TOOL RESULT]")?;
            writeln!(writer, "name: {}", name)?;
            writeln!(writer, "content: {}", content)?;
            writeln!(writer, "is_error: {}", is_error)?;
            writeln!(writer)?;
        }
        Ok(())
    }

    pub fn log_assistant_response(&mut self, response: &str) -> Result<()> {
        if let Some(ref mut writer) = self.log_file {
            writeln!(writer, "[ASSISTANT RESPONSE]")?;
            writeln!(writer, "{}", response)?;
            writeln!(writer)?;
        }
        Ok(())
    }

    pub fn log_usage(
        &mut self,
        input_tokens: u32,
        output_tokens: u32,
        stop_reason: &str,
    ) -> Result<()> {
        if let Some(ref mut writer) = self.log_file {
            writeln!(writer, "[USAGE]")?;
            writeln!(writer, "input_tokens: {}", input_tokens)?;
            writeln!(writer, "output_tokens: {}", output_tokens)?;
            writeln!(writer, "stop_reason: {}", stop_reason)?;
            writeln!(writer)?;
        }
        Ok(())
    }

    pub fn log_config(&mut self, config: &dyn std::fmt::Debug) -> Result<()> {
        if let Some(ref mut writer) = self.log_file {
            writeln!(writer, "[CONFIG]")?;
            writeln!(writer, "{:#?}", config)?;
            writeln!(writer)?;
        }
        Ok(())
    }

    pub fn log_error(&mut self, message: &str) -> Result<()> {
        if let Some(ref mut writer) = self.log_file {
            writeln!(writer, "[ERROR]")?;
            writeln!(writer, "{}", message)?;
            writeln!(writer)?;
        }
        Ok(())
    }

    pub fn log_warn(&mut self, message: &str) -> Result<()> {
        if let Some(ref mut writer) = self.log_file {
            writeln!(writer, "[WARN]")?;
            writeln!(writer, "{}", message)?;
            writeln!(writer)?;
        }
        Ok(())
    }

    pub fn log_info(&mut self, message: &str) -> Result<()> {
        if let Some(ref mut writer) = self.log_file {
            writeln!(writer, "[INFO]")?;
            writeln!(writer, "{}", message)?;
            writeln!(writer)?;
        }
        Ok(())
    }

    pub fn log_event(&mut self, event: &AgentEvent) -> Result<()> {
        match event {
            AgentEvent::ToolUseReceived { name, input, .. } => {
                self.log_tool_use(name, input)?;
            }
            AgentEvent::ToolResult {
                name,
                content,
                is_error,
                ..
            } => {
                self.log_tool_result(name, content, *is_error)?;
            }
            AgentEvent::ResponseComplete(response) => {
                self.log_assistant_response(response)?;
            }
            AgentEvent::Error(message) => {
                self.log_error(message)?;
            }
            AgentEvent::Warn(message) => {
                self.log_warn(message)?;
            }
            AgentEvent::Usage {
                input_tokens,
                output_tokens,
                stop_reason,
            } => {
                self.log_usage(*input_tokens, *output_tokens, stop_reason)?;
            }
            _ => {}
        }
        Ok(())
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        if let Some(ref mut writer) = self.log_file {
            let _ = writer.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentEvent;
    use tempfile::TempDir;

    #[test]
    fn logger_with_none_path_creates_disabled_logger() {
        let mut logger = Logger::new(None).expect("logger creation should succeed");
        // Logger with None should not panic on any operations
        logger
            .log_user_input("test")
            .expect("log_user_input should not error");
        logger
            .log_tool_use("bash", &serde_json::json!({}))
            .expect("log_tool_use should not error");
        logger
            .log_assistant_response("response")
            .expect("log_assistant_response should not error");
    }

    #[test]
    fn log_user_input_writes_to_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");

        {
            let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");
            logger
                .log_user_input("hello world")
                .expect("Failed to log user input");
        }

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[USER INPUT]"));
        assert!(content.contains("hello world"));
    }

    #[test]
    fn log_tool_use_writes_name_and_input_to_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");

        {
            let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");
            let input = serde_json::json!({"command": "ls -la"});
            logger
                .log_tool_use("bash", &input)
                .expect("Failed to log tool use");
        }

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[TOOL CALL]"));
        assert!(content.contains("name: bash"));
        assert!(content.contains(r#""command":"ls -la""#));
    }

    #[test]
    fn log_tool_result_writes_name_content_and_error_status_to_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");

        {
            let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");
            logger
                .log_tool_result("bash", "file1.txt\nfile2.txt", false)
                .expect("Failed to log tool result");
        }

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[TOOL RESULT]"));
        assert!(content.contains("name: bash"));
        assert!(content.contains("content: file1.txt\nfile2.txt"));
        assert!(content.contains("is_error: false"));
    }

    #[test]
    fn log_assistant_response_writes_response_to_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");

        {
            let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");
            logger
                .log_assistant_response("Hello, how can I help you?")
                .expect("Failed to log assistant response");
        }

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[ASSISTANT RESPONSE]"));
        assert!(content.contains("Hello, how can I help you?"));
    }

    #[test]
    fn log_event_tool_use_received_calls_log_tool_use() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");

        {
            let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");
            let event = AgentEvent::ToolUseReceived {
                id: "tool-1".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"command": "ls"}),
                index: 1,
            };
            logger.log_event(&event).expect("Failed to log event");
        }

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[TOOL CALL]"));
        assert!(content.contains("name: bash"));
    }

    #[test]
    fn log_event_tool_result_calls_log_tool_result() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");

        {
            let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");
            let event = AgentEvent::ToolResult {
                name: "bash".to_string(),
                content: "output".to_string(),
                is_error: false,
                index: 1,
            };
            logger.log_event(&event).expect("Failed to log event");
        }

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[TOOL RESULT]"));
        assert!(content.contains("name: bash"));
        assert!(content.contains("content: output"));
    }

    #[test]
    fn log_event_response_complete_calls_log_assistant_response() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");

        {
            let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");
            let event = AgentEvent::ResponseComplete("Hello world".to_string());
            logger.log_event(&event).expect("Failed to log event");
        }

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[ASSISTANT RESPONSE]"));
        assert!(content.contains("Hello world"));
    }

    #[test]
    fn multiple_logs_append_to_same_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");

        {
            let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");
            logger.log_user_input("input1").expect("Failed to log");
            logger
                .log_assistant_response("response1")
                .expect("Failed to log");
            logger.log_user_input("input2").expect("Failed to log");
            logger
                .log_assistant_response("response2")
                .expect("Failed to log");
        }

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("input1"));
        assert!(content.contains("response1"));
        assert!(content.contains("input2"));
        assert!(content.contains("response2"));
    }

    #[test]
    fn logger_flushes_on_drop() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");

        {
            let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");
            logger.log_user_input("test input").expect("Failed to log");
        }

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[USER INPUT]"));
        assert!(content.contains("test input"));
    }

    #[test]
    fn log_config_writes_pretty_debug_representation() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");

        {
            let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");
            let config = vec!["backend", "vertex"];
            logger.log_config(&config).expect("Failed to log config");
        }

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[CONFIG]"));
        assert!(content.contains(r#""backend""#));
        assert!(content.contains(r#""vertex""#));
    }

    #[test]
    fn log_config_does_nothing_when_no_log_file() {
        let mut logger = Logger::new(None).expect("logger creation should succeed");
        let config = vec!["backend", "vertex"];
        logger
            .log_config(&config)
            .expect("log_config should not error");
    }

    #[test]
    fn explicit_flush_method_writes_buffered_content() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");
        let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");

        logger
            .log_user_input("before flush")
            .expect("Failed to log");
        logger.flush().expect("Failed to flush");

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("before flush"));
    }

    /// Exercises `Logger` instance methods (`log_warn`/`log_info`/`log_error`)
    /// directly. Does NOT touch the process-global; for global+free-function
    /// coverage see `init_global_then_free_functions_write_to_log_file` below.
    #[test]
    fn logger_instance_warn_info_error_methods_write_to_file() {
        let temp_dir = TempDir::new().expect("temp dir");
        let log_path = temp_dir.path().join("instance.log");
        let mut logger = Logger::new(Some(log_path.clone())).expect("logger");

        logger.log_warn("test warning").expect("log_warn");
        logger.log_info("test info").expect("log_info");
        logger.log_error("test error").expect("log_error");
        logger.flush().expect("flush");

        let content = std::fs::read_to_string(&log_path).expect("read file");
        assert!(content.contains("[WARN]"));
        assert!(content.contains("test warning"));
        assert!(content.contains("[INFO]"));
        assert!(content.contains("test info"));
        assert!(content.contains("[ERROR]"));
        assert!(content.contains("test error"));
    }

    /// End-to-end test of the global logger path. `OnceLock` permits exactly
    /// one initialisation per process, so this is the *only* test in the lib
    /// test binary that calls `init_global`. It exercises:
    ///   - `init_global(Some(_))` actually sets the global
    ///   - `is_enabled()` reports `true` after initialisation
    ///   - `init_global` is idempotent on second call (silent no-op, no panic)
    ///   - free functions (`log_warn`, `log_event`) reach the underlying file
    ///   - `flush()` forces buffered output to disk
    ///
    /// The TempDir is leaked so the open file descriptor in `GLOBAL_LOGGER`
    /// outlives the test (other tests that route through `with_global` would
    /// otherwise fail with EBADF on cleanup).
    #[test]
    fn init_global_then_free_functions_write_to_log_file() {
        // Use a leaked path so the file outlives this test — the global keeps
        // the BufWriter for the rest of the process.
        let dir = TempDir::new().expect("temp dir").keep();
        let log_path = dir.join("global.log");

        init_global(Some(log_path.clone())).expect("first init_global");
        assert!(is_enabled(), "is_enabled() must be true after init_global");

        // Idempotent: second call must not panic and must not change the
        // existing logger (still writes to the original log_path).
        init_global(Some(dir.join("ignored.log")))
            .expect("second init_global must be a silent no-op");

        log_warn("global warn message");

        let event = AgentEvent::ToolUseReceived {
            id: "tool-1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
            index: 1,
        };
        log_event(&event);

        flush();

        let content = std::fs::read_to_string(&log_path).expect("read file");
        assert!(
            content.contains("[WARN]") && content.contains("global warn message"),
            "free fn log_warn must reach file; got: {content}"
        );
        assert!(
            content.contains("[TOOL CALL]") && content.contains("name: bash"),
            "free fn log_event must reach file; got: {content}"
        );
        assert!(
            !content.contains("ignored.log"),
            "second init_global must not have replaced the logger"
        );
    }
}
