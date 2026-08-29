use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::types::AgentEvent;
use anyhow::{Context, Result};

static GLOBAL_WRITER: OnceLock<Mutex<Option<BufWriter<File>>>> = OnceLock::new();

/// Initialise the global debug-log writer once.
pub fn init_global(log_path: Option<PathBuf>) -> Result<()> {
    let writer = match log_path {
        Some(path) => Some(BufWriter::new(File::create(&path).with_context(|| {
            format!("Failed to create debug log file at {}", path.display())
        })?)),
        None => None,
    };
    let _ = GLOBAL_WRITER.set(Mutex::new(writer));
    Ok(())
}

// ── Free functions ──────────────────────────────────────────────────────────

fn with_global_writer(f: impl FnOnce(&mut BufWriter<File>)) {
    let guard = match GLOBAL_WRITER.get() {
        Some(g) => g,
        None => return,
    };
    let mut lock = match guard.lock() {
        Ok(l) => l,
        Err(_) => return,
    };
    if let Some(ref mut writer) = *lock {
        f(writer);
    }
}

pub fn log_warn(message: &str) {
    with_global_writer(|w| {
        let _ = writeln!(w, "[WARN]\n{}\n", message);
    });
}

pub fn log_error(message: &str) {
    with_global_writer(|w| {
        let _ = writeln!(w, "[ERROR]\n{}\n", message);
    });
}

pub fn log_info(message: &str) {
    with_global_writer(|w| {
        let _ = writeln!(w, "[INFO]\n{}\n", message);
    });
}

pub fn log_user_input(input: &str) {
    with_global_writer(|w| {
        let _ = writeln!(w, "[USER INPUT]\n{}\n", input);
    });
}

pub fn log_tool_use(name: &str, input: &serde_json::Value) {
    with_global_writer(|w| {
        let _ = writeln!(w, "[TOOL CALL]\nname: {}\ninput: {}\n", name, input);
    });
}

pub fn log_tool_result(name: &str, content: &str, is_error: bool) {
    let max_log = 8_192;
    let display = if content.len() > max_log {
        format!(
            "{}\n[... log output truncated: {} bytes elided ...]",
            &content[..content.floor_char_boundary(max_log)],
            content.len()
        )
    } else {
        content.to_string()
    };
    with_global_writer(|w| {
        let _ = writeln!(
            w,
            "[TOOL RESULT]\nname: {}\ncontent: {}\nis_error: {}\n",
            name, display, is_error
        );
    });
}

pub fn log_assistant_response(response: &str) {
    with_global_writer(|w| {
        let _ = writeln!(w, "[ASSISTANT RESPONSE]\n{}\n", response);
    });
}

pub fn log_usage(input_tokens: u32, output_tokens: u32, stop_reason: &str) {
    with_global_writer(|w| {
        let _ = writeln!(
            w,
            "[USAGE]\ninput_tokens: {}\noutput_tokens: {}\nstop_reason: {}\n",
            input_tokens, output_tokens, stop_reason
        );
    });
}

pub fn log_config(config: &dyn std::fmt::Debug) {
    with_global_writer(|w| {
        let _ = writeln!(w, "[CONFIG]\n{:#?}\n", config);
    });
}

pub fn log_event(event: &AgentEvent) {
    match event {
        AgentEvent::TokenReceived(_text) => {}
        AgentEvent::ThinkingReceived(text) => {
            with_global_writer(|w| {
                let _ = writeln!(w, "[THINKING]\n{}\n", text);
            });
        }
        AgentEvent::ToolUseReceived { name, input, .. } => {
            log_tool_use(name, input);
        }
        AgentEvent::ToolResult {
            name,
            content,
            is_error,
            ..
        } => {
            log_tool_result(name, content, *is_error);
        }
        AgentEvent::ToolConfirmationRequired { name, input, .. } => {
            with_global_writer(|w| {
                let _ = writeln!(w, "[TOOL CONFIRMATION]\nname: {}\ninput: {}\n", name, input);
            });
        }
        AgentEvent::ResponseComplete(text) => {
            log_assistant_response(text);
        }
        AgentEvent::Error(msg) => {
            log_error(msg);
        }
        AgentEvent::Retrying(msg) => {
            log_warn(&format!("Retrying: {msg}"));
        }
        AgentEvent::Usage {
            input_tokens,
            output_tokens,
            stop_reason,
        } => {
            log_usage(*input_tokens, *output_tokens, stop_reason);
        }
        AgentEvent::SubAgentUsage {
            input_tokens,
            output_tokens,
            role,
        } => {
            with_global_writer(|w| {
                let _ = writeln!(
                    w,
                    "[SUB-AGENT USAGE]\nrole: {}\ninput_tokens: {}\noutput_tokens: {}\n",
                    role, input_tokens, output_tokens
                );
            });
        }
        AgentEvent::Interrupted { partial_text } => {
            with_global_writer(|w| {
                let _ = writeln!(w, "[INTERRUPTED]\n{}\n", partial_text);
            });
        }
        AgentEvent::Warn(msg) => {
            log_warn(msg);
        }
        AgentEvent::CompactionComplete { summary, is_error } => {
            with_global_writer(|w| {
                let _ = writeln!(
                    w,
                    "[COMPACTION]\nsummary: {}\nis_error: {}\n",
                    summary, is_error
                );
            });
        }
        AgentEvent::AutoCompactTriggered {
            current_tokens,
            threshold,
        } => {
            with_global_writer(|w| {
                let _ = writeln!(
                    w,
                    "[AUTO-COMPACT]\ncurrent_tokens: {}\nthreshold: {}\n",
                    current_tokens, threshold
                );
            });
        }
        AgentEvent::BashCommandComplete => {}
    }
}

pub fn flush() {
    with_global_writer(|w| {
        let _ = w.flush();
    });
}

// ── Logger (instance API — kept for backward compat / tests) ────────────────

pub struct Logger {
    log_file: Option<BufWriter<File>>,
}

impl Logger {
    pub fn new(path: Option<PathBuf>) -> Result<Self> {
        let log_file = match path {
            Some(p) => {
                let f = File::create(&p)?;
                Some(BufWriter::new(f))
            }
            None => None,
        };
        Ok(Self { log_file })
    }

    fn write_or_delegate(
        &mut self,
        file_fn: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
        free_fn: impl FnOnce(),
    ) -> Result<()> {
        if let Some(ref mut w) = self.log_file {
            file_fn(w)
        } else {
            free_fn();
            Ok(())
        }
    }

    pub fn log_warn(&mut self, message: &str) -> Result<()> {
        self.write_or_delegate(
            |w| writeln!(w, "[WARN]\n{}\n", message).map_err(Into::into),
            || log_warn(message),
        )
    }

    pub fn log_error(&mut self, message: &str) -> Result<()> {
        self.write_or_delegate(
            |w| writeln!(w, "[ERROR]\n{}\n", message).map_err(Into::into),
            || log_error(message),
        )
    }

    pub fn log_info(&mut self, message: &str) -> Result<()> {
        self.write_or_delegate(
            |w| writeln!(w, "[INFO]\n{}\n", message).map_err(Into::into),
            || log_info(message),
        )
    }

    pub fn log_user_input(&mut self, input: &str) -> Result<()> {
        self.write_or_delegate(
            |w| writeln!(w, "[USER INPUT]\n{}\n", input).map_err(Into::into),
            || log_user_input(input),
        )
    }

    pub fn log_tool_use(&mut self, name: &str, input: &serde_json::Value) -> Result<()> {
        self.write_or_delegate(
            |w| writeln!(w, "[TOOL CALL]\nname: {}\ninput: {}\n", name, input).map_err(Into::into),
            || log_tool_use(name, input),
        )
    }

    pub fn log_tool_result(&mut self, name: &str, content: &str, is_error: bool) -> Result<()> {
        self.write_or_delegate(
            |w| {
                writeln!(
                    w,
                    "[TOOL RESULT]\nname: {}\ncontent: {}\nis_error: {}\n",
                    name, content, is_error
                )
                .map_err(Into::into)
            },
            || log_tool_result(name, content, is_error),
        )
    }

    pub fn log_assistant_response(&mut self, response: &str) -> Result<()> {
        self.write_or_delegate(
            |w| writeln!(w, "[ASSISTANT RESPONSE]\n{}\n", response).map_err(Into::into),
            || log_assistant_response(response),
        )
    }

    pub fn log_usage(
        &mut self,
        input_tokens: u32,
        output_tokens: u32,
        stop_reason: &str,
    ) -> Result<()> {
        self.write_or_delegate(
            |w| {
                writeln!(
                    w,
                    "[USAGE]\ninput_tokens: {}\noutput_tokens: {}\nstop_reason: {}\n",
                    input_tokens, output_tokens, stop_reason
                )
                .map_err(Into::into)
            },
            || log_usage(input_tokens, output_tokens, stop_reason),
        )
    }

    pub fn log_config(&mut self, config: &dyn std::fmt::Debug) -> Result<()> {
        self.write_or_delegate(
            |w| writeln!(w, "[CONFIG]\n{:#?}\n", config).map_err(Into::into),
            || log_config(config),
        )
    }

    pub fn log_event(&mut self, event: &AgentEvent) -> Result<()> {
        if let Some(ref mut w) = self.log_file {
            Self::write_event(w, event)
        } else {
            log_event(event);
            Ok(())
        }
    }

    fn write_event(w: &mut BufWriter<File>, event: &AgentEvent) -> Result<()> {
        match event {
            AgentEvent::TokenReceived(_) => Ok(()),
            AgentEvent::ThinkingReceived(text) => {
                writeln!(w, "[THINKING]\n{}\n", text).map_err(Into::into)
            }
            AgentEvent::ToolUseReceived { name, input, .. } => {
                writeln!(w, "[TOOL CALL]\nname: {}\ninput: {}\n", name, input).map_err(Into::into)
            }
            AgentEvent::ToolResult {
                name,
                content,
                is_error,
                ..
            } => writeln!(
                w,
                "[TOOL RESULT]\nname: {}\ncontent: {}\nis_error: {}\n",
                name, content, is_error
            )
            .map_err(Into::into),
            AgentEvent::ToolConfirmationRequired { name, input, .. } => {
                writeln!(w, "[TOOL CONFIRMATION]\nname: {}\ninput: {}\n", name, input)
                    .map_err(Into::into)
            }
            AgentEvent::ResponseComplete(text) => {
                writeln!(w, "[ASSISTANT RESPONSE]\n{}\n", text).map_err(Into::into)
            }
            AgentEvent::Error(msg) => writeln!(w, "[ERROR]\n{}\n", msg).map_err(Into::into),
            AgentEvent::Retrying(msg) => writeln!(w, "[RETRYING]\n{}\n", msg).map_err(Into::into),
            AgentEvent::Usage {
                input_tokens,
                output_tokens,
                stop_reason,
            } => writeln!(
                w,
                "[USAGE]\ninput_tokens: {}\noutput_tokens: {}\nstop_reason: {}\n",
                input_tokens, output_tokens, stop_reason
            )
            .map_err(Into::into),
            AgentEvent::SubAgentUsage {
                input_tokens,
                output_tokens,
                role,
            } => writeln!(
                w,
                "[SUB-AGENT USAGE]\nrole: {}\ninput_tokens: {}\noutput_tokens: {}\n",
                role, input_tokens, output_tokens
            )
            .map_err(Into::into),
            AgentEvent::Interrupted { partial_text } => {
                writeln!(w, "[INTERRUPTED]\n{}\n", partial_text).map_err(Into::into)
            }
            AgentEvent::Warn(msg) => writeln!(w, "[WARN]\n{}\n", msg).map_err(Into::into),
            AgentEvent::CompactionComplete { summary, is_error } => writeln!(
                w,
                "[COMPACTION]\nsummary: {}\nis_error: {}\n",
                summary, is_error
            )
            .map_err(Into::into),
            AgentEvent::AutoCompactTriggered {
                current_tokens,
                threshold,
            } => writeln!(
                w,
                "[AUTO-COMPACT]\ncurrent_tokens: {}\nthreshold: {}\n",
                current_tokens, threshold
            )
            .map_err(Into::into),
            AgentEvent::BashCommandComplete => Ok(()),
        }
    }

    pub fn flush(&mut self) -> Result<()> {
        if let Some(ref mut w) = self.log_file {
            w.flush().map_err(Into::into)
        } else {
            flush();
            Ok(())
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Read;

    fn temp_log_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("illustrious-manager-logging-tests");
        let _ = fs::create_dir_all(&dir);
        dir.join(format!("{}-{}.log", name, std::process::id()))
    }

    fn read_file(path: &PathBuf) -> String {
        let mut contents = String::new();
        File::open(path)
            .and_then(|mut f| f.read_to_string(&mut contents))
            .expect("Should be able to read temp log file");
        contents
    }

    #[test]
    fn logger_creates_file_on_init() {
        let path = temp_log_path("creates_file");
        let _logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        assert!(path.exists(), "Log file should be created");
    }

    #[test]
    fn logger_warn_writes_to_file() {
        let path = temp_log_path("warn");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_warn("warning message")
            .expect("Failed to log warn");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[WARN]"));
        assert!(contents.contains("warning message"));
    }

    #[test]
    fn logger_error_writes_to_file() {
        let path = temp_log_path("error");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_error("error message")
            .expect("Failed to log error");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[ERROR]"));
        assert!(contents.contains("error message"));
    }

    #[test]
    fn logger_info_writes_to_file() {
        let path = temp_log_path("info");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger.log_info("info message").expect("Failed to log info");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[INFO]"));
        assert!(contents.contains("info message"));
    }

    #[test]
    fn logger_user_input_writes_to_file() {
        let path = temp_log_path("user_input");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_user_input("ls -la")
            .expect("Failed to log user input");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[USER INPUT]"));
        assert!(contents.contains("ls -la"));
    }

    #[test]
    fn logger_tool_use_writes_to_file() {
        let path = temp_log_path("tool_use");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        let input = serde_json::json!({"command": "ls"});
        logger
            .log_tool_use("bash", &input)
            .expect("Failed to log tool use");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[TOOL CALL]"));
        assert!(contents.contains("name: bash"));
        assert!(contents.contains("command"));
    }

    #[test]
    fn logger_tool_result_writes_to_file() {
        let path = temp_log_path("tool_result");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_tool_result("bash", "file1\nfile2", false)
            .expect("Failed to log tool result");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[TOOL RESULT]"));
        assert!(contents.contains("name: bash"));
        assert!(contents.contains("file1"));
        assert!(contents.contains("is_error: false"));
    }

    #[test]
    fn logger_assistant_response_writes_to_file() {
        let path = temp_log_path("assistant_response");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_assistant_response("Hello, world!")
            .expect("Failed to log assistant response");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[ASSISTANT RESPONSE]"));
        assert!(contents.contains("Hello, world!"));
    }

    #[test]
    fn logger_usage_writes_to_file() {
        let path = temp_log_path("usage");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_usage(100, 200, "end_turn")
            .expect("Failed to log usage");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[USAGE]"));
        assert!(contents.contains("input_tokens: 100"));
        assert!(contents.contains("output_tokens: 200"));
        assert!(contents.contains("stop_reason: end_turn"));
    }

    #[test]
    fn logger_config_writes_to_file() {
        let path = temp_log_path("config");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        let config = serde_json::json!({"model": "claude-sonnet-4"});
        logger.log_config(&config).expect("Failed to log config");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[CONFIG]"));
        assert!(contents.contains("claude-sonnet-4"));
    }

    #[test]
    fn logger_event_token_received_does_not_write() {
        let path = temp_log_path("event_token");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::TokenReceived("test".into()))
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.is_empty());
    }

    #[test]
    fn logger_event_thinking_received_writes() {
        let path = temp_log_path("event_thinking");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::ThinkingReceived("thinking...".into()))
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[THINKING]"));
        assert!(contents.contains("thinking..."));
    }

    #[test]
    fn logger_event_tool_use_received_writes() {
        let path = temp_log_path("event_tool_use");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::ToolUseReceived {
                id: "1".into(),
                name: "bash".into(),
                input: serde_json::json!({"command": "ls"}),
                index: 1,
            })
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[TOOL CALL]"));
        assert!(contents.contains("name: bash"));
    }

    #[test]
    fn logger_event_tool_result_writes() {
        let path = temp_log_path("event_tool_result");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::ToolResult {
                name: "bash".into(),
                content: "output".into(),
                is_error: false,
                index: 1,
            })
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[TOOL RESULT]"));
        assert!(contents.contains("name: bash"));
    }

    #[test]
    fn logger_event_response_complete_writes() {
        let path = temp_log_path("event_response_complete");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::ResponseComplete("Done".into()))
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[ASSISTANT RESPONSE]"));
        assert!(contents.contains("Done"));
    }

    #[test]
    fn logger_event_error_writes() {
        let path = temp_log_path("event_error");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::Error("something failed".into()))
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[ERROR]"));
        assert!(contents.contains("something failed"));
    }

    #[test]
    fn logger_event_retrying_writes() {
        let path = temp_log_path("event_retrying");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::Retrying("max tokens exceeded".into()))
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[RETRYING]"));
        assert!(contents.contains("max tokens exceeded"));
    }

    #[test]
    fn logger_event_usage_writes() {
        let path = temp_log_path("event_usage");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::Usage {
                input_tokens: 50,
                output_tokens: 100,
                stop_reason: "max_tokens".into(),
            })
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[USAGE]"));
        assert!(contents.contains("input_tokens: 50"));
        assert!(contents.contains("output_tokens: 100"));
    }

    #[test]
    fn logger_event_subagent_usage_writes() {
        let path = temp_log_path("event_subagent_usage");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::SubAgentUsage {
                input_tokens: 200,
                output_tokens: 300,
                role: "default".into(),
            })
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[SUB-AGENT USAGE]"));
        assert!(contents.contains("role: default"));
        assert!(contents.contains("input_tokens: 200"));
        assert!(contents.contains("output_tokens: 300"));
    }

    #[test]
    fn logger_event_interrupted_writes() {
        let path = temp_log_path("event_interrupted");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::Interrupted {
                partial_text: "partial".into(),
            })
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[INTERRUPTED]"));
        assert!(contents.contains("partial"));
    }

    #[test]
    fn logger_event_warn_writes() {
        let path = temp_log_path("event_warn");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::Warn("warning from agent".into()))
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[WARN]"));
        assert!(contents.contains("warning from agent"));
    }

    #[test]
    fn logger_event_compaction_complete_writes() {
        let path = temp_log_path("event_compaction");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::CompactionComplete {
                summary: "summarised".into(),
                is_error: false,
            })
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[COMPACTION]"));
        assert!(contents.contains("summary: summarised"));
        assert!(contents.contains("is_error: false"));
    }

    #[test]
    fn logger_event_auto_compact_triggered_writes() {
        let path = temp_log_path("event_auto_compact");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::AutoCompactTriggered {
                current_tokens: 50000,
                threshold: 48000,
            })
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[AUTO-COMPACT]"));
        assert!(contents.contains("current_tokens: 50000"));
        assert!(contents.contains("threshold: 48000"));
    }

    #[test]
    fn logger_multiple_messages_append() {
        let path = temp_log_path("multiple");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger.log_info("first").expect("Failed to log info");
        logger.log_warn("second").expect("Failed to log warn");
        logger.log_info("third").expect("Failed to log info");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        let info_count = contents.matches("[INFO]").count();
        assert_eq!(info_count, 2, "Should have two [INFO] entries");
        assert!(contents.contains("first"));
        assert!(contents.contains("second"));
        assert!(contents.contains("third"));
    }

    #[test]
    fn logger_tool_error_flag_is_true() {
        let path = temp_log_path("tool_error");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_tool_result("bash", "error output", true)
            .expect("Failed to log tool result");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("is_error: true"));
    }

    #[test]
    fn logger_tool_confirmation_required_writes() {
        let path = temp_log_path("tool_confirmation");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger
            .log_event(&AgentEvent::ToolConfirmationRequired {
                id: "tool-1".into(),
                name: "write_file".into(),
                input: serde_json::json!({"path": "test.txt"}),
                index: 1,
            })
            .expect("Failed to log event");
        logger.flush().expect("Failed to flush");

        let contents = read_file(&path);
        assert!(contents.contains("[TOOL CONFIRMATION]"));
        assert!(contents.contains("name: write_file"));
    }

    #[test]
    fn logger_flush_succeeds() {
        let path = temp_log_path("flush");
        let mut logger = Logger::new(Some(path.clone())).expect("Failed to create logger");
        logger.log_info("flush test").expect("Failed to log info");
        logger.flush().expect("Failed to flush");
        let first = read_file(&path);
        logger.log_info("flush test 2").expect("Failed to log info");
        logger.flush().expect("Failed to flush");
        let second = read_file(&path);
        assert!(
            second.len() > first.len(),
            "Second flush should have more data"
        );
    }

    #[test]
    fn log_free_functions_are_noop_when_global_not_used() {
        log_warn("x");
        log_error("x");
        log_info("x");
        // No panic = success.
    }

    #[test]
    fn logger_none_delegates_to_free_functions() {
        let mut logger = Logger::new(None).expect("Failed to create logger");
        logger.log_warn("x").expect("Failed to log warn");
        logger.log_error("x").expect("Failed to log error");
        logger.log_info("x").expect("Failed to log info");
        // No panic = success.
    }

    #[test]
    fn global_writer_path_and_free_log_event() {
        // GLOBAL_WRITER is a process-global OnceLock. This is the only test that calls
        // init_global; a single test function avoids races with other test threads.
        // If the OnceLock was already set (e.g. by a prior call in the same process),
        // init_global silently no-ops — the "no panic" guarantee still holds.
        use crate::types::AgentEvent;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("global-test.log");

        init_global(Some(path.clone())).expect("init_global should succeed");

        log_warn("global warn");
        log_error("global error");
        log_event(&AgentEvent::ResponseComplete("assistant text".to_string()));
        log_event(&AgentEvent::Error("some error".to_string()));
        log_event(&AgentEvent::Retrying("max tokens exceeded".to_string()));
        flush();

        // Only assert file contents when this process's init_global call won the OnceLock.
        if path.exists() {
            let content = std::fs::read_to_string(&path).expect("read log");
            assert!(content.contains("[WARN]"), "expected [WARN] tag");
            assert!(content.contains("global warn"), "expected warn message");
            assert!(content.contains("[ERROR]"), "expected [ERROR] tag");
            assert!(content.contains("global error"), "expected error message");
            assert!(
                content.contains("Retrying: max tokens exceeded"),
                "expected retrying message in warn format"
            );
            assert!(
                content.contains("[ASSISTANT RESPONSE]"),
                "expected [ASSISTANT RESPONSE] tag"
            );
            assert!(
                content.contains("assistant text"),
                "expected assistant text"
            );
        }
    }
}
