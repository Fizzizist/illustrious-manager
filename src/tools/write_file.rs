use std::path::Path;

use serde_json::Value;

use super::{Tool, ToolError, ToolResult};
use crate::tools::sandbox::SandboxPolicy;
use crate::types::ContentBlock;

pub struct WriteFileTool {
    sandbox: SandboxPolicy,
    schema: Value,
}

impl WriteFileTool {
    pub fn new(sandbox: SandboxPolicy) -> Self {
        Self {
            sandbox,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write (relative to sandbox root or absolute within sandbox)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }
}

impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Create or overwrite a file within the sandbox with the given content"
    }

    fn input_schema(&self) -> &Value {
        &self.schema
    }

    fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let path_str = input["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput {
                message: "Missing required field 'path'".to_string(),
            })?;

        let content = input["content"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput {
                message: "Missing required field 'content'".to_string(),
            })?;

        let path = Path::new(path_str);

        let validated =
            self.sandbox
                .validate_write_path(path)
                .map_err(|e| ToolError::Execution {
                    tool_name: self.name().to_string(),
                    message: e.to_string(),
                })?;

        if let Some(parent) = validated.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ToolError::Execution {
                tool_name: self.name().to_string(),
                message: format!("Failed to create parent directories: {}", e),
            })?;
        }

        // Re-validate after directory creation to mitigate TOCTOU: a symlink could
        // have been inserted into the path between initial validation and create_dir_all.
        let validated =
            self.sandbox
                .validate_write_path(path)
                .map_err(|e| ToolError::Execution {
                    tool_name: self.name().to_string(),
                    message: e.to_string(),
                })?;

        std::fs::write(&validated, content).map_err(|e| ToolError::Execution {
            tool_name: self.name().to_string(),
            message: format!("Failed to write file: {}", e),
        })?;

        let bytes = content.len();
        Ok(ToolResult {
            content: vec![ContentBlock::Text(format!(
                "Wrote {} bytes to {:?}",
                bytes, validated
            ))],
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::sandbox::SandboxPolicy;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn write_file_creates_new_file_with_correct_content() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let tool = WriteFileTool::new(sandbox);

        let file_path = temp_dir.path().join("hello.txt");
        let input = serde_json::json!({
            "path": file_path.to_str().expect("temp path should be valid UTF-8"),
            "content": "Hello, world!"
        });

        let result = tool.execute(input).expect("Write should succeed");
        assert!(!result.is_error);

        let written = fs::read_to_string(&file_path).expect("File should exist");
        assert_eq!(written, "Hello, world!");
    }

    #[test]
    fn write_file_overwrites_existing_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("existing.txt");
        fs::write(&file_path, "old content").expect("Failed to create file");

        let sandbox = SandboxPolicy::new(temp_dir.path());
        let tool = WriteFileTool::new(sandbox);

        let input = serde_json::json!({
            "path": file_path.to_str().expect("temp path should be valid UTF-8"),
            "content": "new content"
        });

        tool.execute(input).expect("Write should succeed");

        let written = fs::read_to_string(&file_path).expect("File should exist");
        assert_eq!(written, "new content");
    }

    #[test]
    fn write_file_creates_parent_directories() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let tool = WriteFileTool::new(sandbox);

        let file_path = temp_dir
            .path()
            .join("a")
            .join("b")
            .join("c")
            .join("file.txt");
        let input = serde_json::json!({
            "path": file_path.to_str().expect("temp path should be valid UTF-8"),
            "content": "nested content"
        });

        let result = tool.execute(input).expect("Write should succeed");
        assert!(!result.is_error);

        let written = fs::read_to_string(&file_path).expect("File should exist");
        assert_eq!(written, "nested content");
    }

    #[test]
    fn write_file_outside_sandbox_is_rejected() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let tool = WriteFileTool::new(sandbox);

        let parent = temp_dir
            .path()
            .parent()
            .expect("Temp dir should have parent");
        let outside_path = parent.join("escape.txt");
        let input = serde_json::json!({
            "path": outside_path.to_str().expect("temp path should be valid UTF-8"),
            "content": "should not be written"
        });

        let result = tool.execute(input);
        match result {
            Err(ToolError::Execution { .. }) => {}
            Ok(_) => panic!("Write outside sandbox should fail"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn write_file_with_symlink_escaping_sandbox_is_rejected() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let outside_dir = TempDir::new().expect("Failed to create outside dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());

        let symlink_dir = temp_dir.path().join("escape_dir");

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outside_dir.path(), &symlink_dir)
                .expect("Failed to create symlink");
        }
        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_dir(outside_dir.path(), &symlink_dir)
                .expect("Failed to create symlink");
        }

        let tool = WriteFileTool::new(sandbox);
        let target = symlink_dir.join("escaped.txt");
        let input = serde_json::json!({
            "path": target.to_str().expect("temp path should be valid UTF-8"),
            "content": "should not be written"
        });

        let result = tool.execute(input);
        match result {
            Err(ToolError::Execution { .. }) => {}
            Ok(_) => panic!("Symlink escaping sandbox should be rejected"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn write_file_result_reports_bytes_written() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let tool = WriteFileTool::new(sandbox);

        let file_path = temp_dir.path().join("count.txt");
        let content = "12345";
        let input = serde_json::json!({
            "path": file_path.to_str().expect("temp path should be valid UTF-8"),
            "content": content
        });

        let result = tool.execute(input).expect("Write should succeed");
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlock::Text(msg) => {
                assert!(msg.contains("5"), "Should report 5 bytes written");
            }
            _ => panic!("Expected text content block"),
        }
    }
}
