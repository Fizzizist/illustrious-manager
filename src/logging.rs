use crate::types::AgentEvent;
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

pub struct Logger {
    log_file: Option<File>,
}

impl Logger {
    pub fn new(log_path: Option<PathBuf>) -> Result<Self> {
        let log_file = match log_path {
            Some(path) => {
                let file = File::create(&path)?;
                Some(file)
            }
            None => None,
        };
        Ok(Self { log_file })
    }

    pub fn log_user_input(&mut self, input: &str) -> Result<()> {
        if let Some(ref mut file) = self.log_file {
            writeln!(file, "[USER INPUT]")?;
            writeln!(file, "{}", input)?;
            writeln!(file)?;
            file.flush()?;
        }
        Ok(())
    }

    pub fn log_tool_use(&mut self, name: &str, input: &serde_json::Value) -> Result<()> {
        if let Some(ref mut file) = self.log_file {
            writeln!(file, "[TOOL CALL]")?;
            writeln!(file, "name: {}", name)?;
            writeln!(file, "input: {}", serde_json::to_string(input)?)?;
            writeln!(file)?;
            file.flush()?;
        }
        Ok(())
    }

    pub fn log_tool_result(&mut self, name: &str, content: &str, is_error: bool) -> Result<()> {
        if let Some(ref mut file) = self.log_file {
            writeln!(file, "[TOOL RESULT]")?;
            writeln!(file, "name: {}", name)?;
            writeln!(file, "content: {}", content)?;
            writeln!(file, "is_error: {}", is_error)?;
            writeln!(file)?;
            file.flush()?;
        }
        Ok(())
    }

    pub fn log_assistant_response(&mut self, response: &str) -> Result<()> {
        if let Some(ref mut file) = self.log_file {
            writeln!(file, "[ASSISTANT RESPONSE]")?;
            writeln!(file, "{}", response)?;
            writeln!(file)?;
            file.flush()?;
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
            } => {
                self.log_tool_result(name, content, *is_error)?;
            }
            AgentEvent::ResponseComplete(response) => {
                self.log_assistant_response(response)?;
            }
            _ => {}
        }
        Ok(())
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
        let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");

        logger
            .log_user_input("hello world")
            .expect("Failed to log user input");

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[USER INPUT]"));
        assert!(content.contains("hello world"));
    }

    #[test]
    fn log_tool_use_writes_name_and_input_to_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");
        let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");

        let input = serde_json::json!({"command": "ls -la"});
        logger
            .log_tool_use("bash", &input)
            .expect("Failed to log tool use");

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[TOOL CALL]"));
        assert!(content.contains("name: bash"));
        assert!(content.contains(r#""command":"ls -la""#));
    }

    #[test]
    fn log_tool_result_writes_name_content_and_error_status_to_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");
        let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");

        logger
            .log_tool_result("bash", "file1.txt\nfile2.txt", false)
            .expect("Failed to log tool result");

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
        let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");

        logger
            .log_assistant_response("Hello, how can I help you?")
            .expect("Failed to log assistant response");

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[ASSISTANT RESPONSE]"));
        assert!(content.contains("Hello, how can I help you?"));
    }

    #[test]
    fn log_event_tool_use_received_calls_log_tool_use() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");
        let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");

        let event = AgentEvent::ToolUseReceived {
            id: "tool-1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
        };
        logger.log_event(&event).expect("Failed to log event");

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[TOOL CALL]"));
        assert!(content.contains("name: bash"));
    }

    #[test]
    fn log_event_tool_result_calls_log_tool_result() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");
        let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");

        let event = AgentEvent::ToolResult {
            name: "bash".to_string(),
            content: "output".to_string(),
            is_error: false,
        };
        logger.log_event(&event).expect("Failed to log event");

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[TOOL RESULT]"));
        assert!(content.contains("name: bash"));
        assert!(content.contains("content: output"));
    }

    #[test]
    fn log_event_response_complete_calls_log_assistant_response() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");
        let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");

        let event = AgentEvent::ResponseComplete("Hello world".to_string());
        logger.log_event(&event).expect("Failed to log event");

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("[ASSISTANT RESPONSE]"));
        assert!(content.contains("Hello world"));
    }

    #[test]
    fn multiple_logs_append_to_same_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let log_path = temp_dir.path().join("test.log");
        let mut logger = Logger::new(Some(log_path.clone())).expect("Failed to create logger");

        logger.log_user_input("input1").expect("Failed to log");
        logger
            .log_assistant_response("response1")
            .expect("Failed to log");
        logger.log_user_input("input2").expect("Failed to log");
        logger
            .log_assistant_response("response2")
            .expect("Failed to log");

        let content = std::fs::read_to_string(&log_path).expect("Failed to read log file");
        assert!(content.contains("input1"));
        assert!(content.contains("response1"));
        assert!(content.contains("input2"));
        assert!(content.contains("response2"));
    }
}
