use crate::types::AgentEvent;
use anyhow::Result;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

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
}
