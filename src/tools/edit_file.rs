use crate::tools::sandbox::SandboxPolicy;
use crate::tools::{Tool, ToolError, ToolResult};
use crate::types::ContentBlock;
use async_trait::async_trait;
use serde_json::Value;
use std::path::Path;

/// EditFile tool for performing exact string replacements in files
///
/// Reads a file, finds an exact string match, and replaces it with new content.
/// Fails if the old string is not found or if it matches multiple locations.
#[derive(Debug)]
pub struct EditFile {
    sandbox: SandboxPolicy,
}

impl EditFile {
    pub fn new(sandbox: SandboxPolicy) -> Self {
        Self { sandbox }
    }
}

#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Perform exact string replacement in a file within the sandbox"
    }

    fn input_schema(&self) -> &Value {
        use std::sync::LazyLock;
        static SCHEMA: LazyLock<Value> = LazyLock::new(|| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit (relative to sandbox root)"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact string to search for and replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "String to replace the old_string with"
                    }
                },
                "required": ["path", "old_string", "new_string"]
            })
        });
        &SCHEMA
    }

    fn markdown_input(&self, input: &Value) -> String {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        let old = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let new = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        format!("**Edit:** `{}`\n```diff\n- {}\n+ {}\n```", path, old, new)
    }

    fn markdown_output(&self, result: &ToolResult) -> String {
        result
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
            .join("\n")
    }

    fn is_write_tool(&self) -> bool {
        true
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        use std::fs;

        let path_str =
            input
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidInput {
                    message: "Missing 'path' field".to_string(),
                })?;

        let old_string = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput {
                message: "Missing 'old_string' field".to_string(),
            })?;

        let new_string = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput {
                message: "Missing 'new_string' field".to_string(),
            })?;

        if old_string.is_empty() {
            return Err(ToolError::InvalidInput {
                message: "'old_string' cannot be empty".to_string(),
            });
        }

        let path = Path::new(path_str);
        let validated_path =
            self.sandbox
                .validate_path(path)
                .map_err(|e| ToolError::Execution {
                    tool_name: self.name().to_string(),
                    message: format!("Path validation failed: {}", e),
                })?;

        let content = fs::read_to_string(&validated_path).map_err(|e| ToolError::Execution {
            tool_name: self.name().to_string(),
            message: format!("Failed to read file: {}", e),
        })?;

        if !content.contains(old_string) {
            return Err(ToolError::Execution {
                tool_name: self.name().to_string(),
                message: "'old_string' not found in file".to_string(),
            });
        }

        let matches = content.matches(old_string).count();
        if matches > 1 {
            return Err(ToolError::Execution {
                tool_name: self.name().to_string(),
                message: format!("'old_string' matches {} locations, must be unique", matches),
            });
        }

        let new_content = content.replacen(old_string, new_string, 1);

        fs::write(&validated_path, new_content).map_err(|e| ToolError::Execution {
            tool_name: self.name().to_string(),
            message: format!("Failed to write file: {}", e),
        })?;

        Ok(ToolResult {
            content: vec![ContentBlock::Text(format!(
                "Successfully replaced string in {:?}",
                validated_path
            ))],
            is_error: false,
            agent_events: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_file(content: &str) -> (TempDir, PathBuf) {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, content).expect("Failed to write test file");
        (temp_dir, file_path)
    }

    #[tokio::test]
    async fn successful_replacement_writes_correct_content() {
        let (_temp_dir, file_path) = create_test_file("hello world\nfoo bar\n");
        let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
        let tool = EditFile::new(sandbox);

        let input = serde_json::json!({
            "path": file_path.file_name().unwrap().to_str().unwrap(),
            "old_string": "hello world",
            "new_string": "goodbye world"
        });

        let result = tool.execute(input).await.expect("Execution should succeed");
        assert!(!result.is_error, "Result should not be an error");

        let content = fs::read_to_string(&file_path).expect("Failed to read file");
        assert_eq!(content, "goodbye world\nfoo bar\n");
    }

    #[tokio::test]
    async fn old_string_not_found_returns_error() {
        let (_temp_dir, file_path) = create_test_file("hello world\n");
        let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
        let tool = EditFile::new(sandbox);

        let input = serde_json::json!({
            "path": file_path.file_name().unwrap().to_str().unwrap(),
            "old_string": "nonexistent",
            "new_string": "replacement"
        });

        let result = tool.execute(input).await;
        assert!(
            result.is_err(),
            "Should return error when old_string not found"
        );
    }

    #[tokio::test]
    async fn old_string_matches_multiple_locations_returns_error() {
        let (_temp_dir, file_path) = create_test_file("hello world\nhello there\n");
        let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
        let tool = EditFile::new(sandbox);

        let input = serde_json::json!({
            "path": file_path.file_name().unwrap().to_str().unwrap(),
            "old_string": "hello",
            "new_string": "goodbye"
        });

        let result = tool.execute(input).await;
        assert!(
            result.is_err(),
            "Should return error when old_string matches multiple locations"
        );
    }

    #[tokio::test]
    async fn path_outside_sandbox_returns_error() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());

        let outside_file = temp_dir.path().parent().unwrap().join("outside.txt");
        fs::write(&outside_file, "content").expect("Failed to write outside file");

        let tool = EditFile::new(sandbox);

        let input = serde_json::json!({
            "path": outside_file.to_str().unwrap(),
            "old_string": "content",
            "new_string": "replacement"
        });

        let result = tool.execute(input).await;
        assert!(
            result.is_err(),
            "Should return error for path outside sandbox"
        );
    }

    #[tokio::test]
    async fn file_doesnt_exist_returns_error() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let tool = EditFile::new(sandbox);

        let input = serde_json::json!({
            "path": "nonexistent.txt",
            "old_string": "old",
            "new_string": "new"
        });

        let result = tool.execute(input).await;
        assert!(
            result.is_err(),
            "Should return error when file doesn't exist"
        );
    }

    #[tokio::test]
    async fn empty_old_string_returns_error() {
        let (_temp_dir, file_path) = create_test_file("content\n");
        let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
        let tool = EditFile::new(sandbox);

        let input = serde_json::json!({
            "path": file_path.file_name().unwrap().to_str().unwrap(),
            "old_string": "",
            "new_string": "replacement"
        });

        let result = tool.execute(input).await;
        assert!(result.is_err(), "Should return error for empty old_string");
    }

    #[tokio::test]
    async fn new_string_can_be_empty() {
        let (_temp_dir, file_path) = create_test_file("hello world\nfoo bar\n");
        let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
        let tool = EditFile::new(sandbox);

        let input = serde_json::json!({
            "path": file_path.file_name().unwrap().to_str().unwrap(),
            "old_string": "hello world\n",
            "new_string": ""
        });

        let result = tool.execute(input).await.expect("Execution should succeed");
        assert!(!result.is_error, "Result should not be an error");

        let content = fs::read_to_string(&file_path).expect("Failed to read file");
        assert_eq!(content, "foo bar\n");
    }

    #[tokio::test]
    async fn markdown_input_formats_as_diff() {
        let (_temp_dir, file_path) = create_test_file("hello world\n");
        let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
        let tool = EditFile::new(sandbox);

        let input = serde_json::json!({
            "path": "test.txt",
            "old_string": "hello world",
            "new_string": "goodbye world"
        });
        let md = tool.markdown_input(&input);
        assert!(md.contains("```diff"), "should format as diff code block");
        assert!(md.contains("- hello world"), "should show removed line");
        assert!(md.contains("+ goodbye world"), "should show added line");
        assert!(md.contains("`test.txt`"), "should show file path");
    }

    #[tokio::test]
    async fn markdown_output_returns_result_text() {
        let (_temp_dir, file_path) = create_test_file("hello world\n");
        let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
        let tool = EditFile::new(sandbox);

        let input = serde_json::json!({
            "path": file_path.file_name().unwrap().to_str().unwrap(),
            "old_string": "hello world",
            "new_string": "goodbye"
        });
        let result = tool.execute(input).await.expect("should succeed");
        let md = tool.markdown_output(&result);
        assert!(
            md.contains("Successfully replaced"),
            "should contain success message"
        );
    }
}
