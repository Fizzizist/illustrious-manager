use crate::tools::path_lock::PathLocks;
use crate::tools::sandbox::SandboxPolicy;
use crate::tools::{Tool, ToolError, ToolResult, run_blocking};
use crate::types::ContentBlock;
use async_trait::async_trait;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

const TOOL: &str = "edit_file";

/// EditFile tool for performing exact string replacements in files
///
/// Reads a file, finds an exact string match, and replaces it with new content.
/// Fails if the old string is not found or if it matches multiple locations.
#[derive(Debug)]
pub struct EditFile {
    sandbox: SandboxPolicy,
    locks: Arc<PathLocks>,
}

impl EditFile {
    pub fn new(sandbox: SandboxPolicy, locks: Arc<PathLocks>) -> Self {
        Self { sandbox, locks }
    }
}

#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> &str {
        TOOL
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

        let lock_key = self
            .sandbox
            .validate_write_path(Path::new(path_str))
            .map_err(|e| ToolError::Execution {
                tool_name: self.name().to_string(),
                message: format!("Path validation failed: {}", e),
            })?;
        let guard = self.locks.lock(&lock_key).await;

        let path = path_str.to_string();
        let sandbox = self.sandbox.clone();
        let old_string = old_string.to_string();
        let new_string = new_string.to_string();

        run_blocking(TOOL, move || -> Result<ToolResult, ToolError> {
            let _guard = guard;
            let validated_path =
                sandbox
                    .validate_path(Path::new(&path))
                    .map_err(|e| ToolError::Execution {
                        tool_name: TOOL.to_string(),
                        message: format!("Path validation failed: {}", e),
                    })?;

            let content =
                fs::read_to_string(&validated_path).map_err(|e| ToolError::Execution {
                    tool_name: TOOL.to_string(),
                    message: format!("Failed to read file: {}", e),
                })?;

            if !content.contains(&old_string) {
                return Err(ToolError::Execution {
                    tool_name: TOOL.to_string(),
                    message: "'old_string' not found in file".to_string(),
                });
            }

            let matches = content.matches(&old_string).count();
            if matches > 1 {
                return Err(ToolError::Execution {
                    tool_name: TOOL.to_string(),
                    message: format!("'old_string' matches {} locations, must be unique", matches),
                });
            }

            let new_content = content.replacen(&old_string, &new_string, 1);

            fs::write(&validated_path, new_content).map_err(|e| ToolError::Execution {
                tool_name: TOOL.to_string(),
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
        })
        .await
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
        let tool = EditFile::new(sandbox, std::sync::Arc::default());

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
        let tool = EditFile::new(sandbox, std::sync::Arc::default());

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
        let tool = EditFile::new(sandbox, std::sync::Arc::default());

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

        let tool = EditFile::new(sandbox, std::sync::Arc::default());

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
        let tool = EditFile::new(sandbox, std::sync::Arc::default());

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
        let tool = EditFile::new(sandbox, std::sync::Arc::default());

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
        let tool = EditFile::new(sandbox, std::sync::Arc::default());

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
        let tool = EditFile::new(sandbox, std::sync::Arc::default());

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

    #[cfg(unix)]
    #[tokio::test]
    async fn edit_file_edits_existing_file_under_tmp_extra_root() {
        let sandbox_dir = TempDir::new().expect("Failed to create sandbox dir");
        let extra_dir = tempfile::tempdir_in("/tmp").expect("Failed to create extra root dir");
        let file_path = extra_dir.path().join("tmp_edit.txt");
        fs::write(&file_path, "hello tmp\n").expect("Failed to write test file");

        let sandbox = SandboxPolicy::new(sandbox_dir.path()).with_extra_root(extra_dir.path());
        let tool = EditFile::new(sandbox, std::sync::Arc::default());

        let input = serde_json::json!({
            "path": file_path.to_str().expect("temp path should be valid UTF-8"),
            "old_string": "hello tmp",
            "new_string": "goodbye tmp"
        });

        let result = tool.execute(input).await.expect("Execution should succeed");
        assert!(!result.is_error, "Result should not be an error");

        let content = fs::read_to_string(&file_path).expect("Failed to read file");
        assert_eq!(content, "goodbye tmp\n");
    }

    #[tokio::test]
    async fn markdown_output_returns_result_text() {
        let (_temp_dir, file_path) = create_test_file("hello world\n");
        let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
        let tool = EditFile::new(sandbox, std::sync::Arc::default());

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

    const CONCURRENT_EDITS: usize = 24;

    fn numbered_lines(n: usize) -> String {
        (0..n).map(|i| format!("line{i}=old\n")).collect()
    }

    fn edit_line_input(path: &str, i: usize) -> Value {
        serde_json::json!({
            "path": path,
            "old_string": format!("line{i}=old"),
            "new_string": format!("line{i}=new"),
        })
    }

    fn assert_all_lines_edited(file_path: &Path, n: usize) {
        let content = fs::read_to_string(file_path).expect("Failed to read file");
        let expected: String = (0..n).map(|i| format!("line{i}=new\n")).collect();
        assert_eq!(content, expected, "every concurrent edit must be applied");
    }

    #[tokio::test]
    async fn concurrent_edits_to_same_file_all_apply() {
        let (_temp_dir, file_path) = create_test_file(&numbered_lines(CONCURRENT_EDITS));
        let tool = EditFile::new(
            SandboxPolicy::new(file_path.parent().expect("file has parent")),
            std::sync::Arc::default(),
        );

        let results = futures::future::join_all(
            (0..CONCURRENT_EDITS).map(|i| tool.execute(edit_line_input("test.txt", i))),
        )
        .await;

        for (i, result) in results.iter().enumerate() {
            let result = result
                .as_ref()
                .unwrap_or_else(|e| panic!("edit {i} failed: {e}"));
            assert!(!result.is_error, "edit {i} reported an error");
        }
        assert_all_lines_edited(&file_path, CONCURRENT_EDITS);
    }

    #[tokio::test]
    async fn edits_via_aliased_paths_serialise() {
        let (temp_dir, file_path) = create_test_file(&numbered_lines(CONCURRENT_EDITS));
        let absolute = file_path
            .to_str()
            .expect("temp path should be valid UTF-8")
            .to_string();
        #[allow(unused_mut)]
        let mut aliases = vec!["test.txt".to_string(), absolute];
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&file_path, temp_dir.path().join("link.txt"))
                .expect("Failed to create symlink");
            aliases.push("link.txt".to_string());
        }
        let tool = EditFile::new(
            SandboxPolicy::new(temp_dir.path()),
            std::sync::Arc::default(),
        );

        let results = futures::future::join_all(
            (0..CONCURRENT_EDITS)
                .map(|i| tool.execute(edit_line_input(&aliases[i % aliases.len()], i))),
        )
        .await;

        for (i, result) in results.iter().enumerate() {
            assert!(
                result.as_ref().is_ok_and(|r| !r.is_error),
                "edit {i} via {} failed: {result:?}",
                aliases[i % aliases.len()]
            );
        }
        assert_all_lines_edited(&file_path, CONCURRENT_EDITS);
    }

    #[tokio::test]
    async fn separate_tool_instances_sharing_locks_serialise() {
        let (temp_dir, file_path) = create_test_file(&numbered_lines(CONCURRENT_EDITS));
        let locks = std::sync::Arc::new(crate::tools::path_lock::PathLocks::new());
        let parent = EditFile::new(SandboxPolicy::new(temp_dir.path()), locks.clone());
        let sub_agent = EditFile::new(SandboxPolicy::new(temp_dir.path()), locks);

        let results = futures::future::join_all((0..CONCURRENT_EDITS).map(|i| {
            let tool = if i % 2 == 0 { &parent } else { &sub_agent };
            tool.execute(edit_line_input("test.txt", i))
        }))
        .await;

        for (i, result) in results.iter().enumerate() {
            assert!(
                result.as_ref().is_ok_and(|r| !r.is_error),
                "edit {i} failed: {result:?}"
            );
        }
        assert_all_lines_edited(&file_path, CONCURRENT_EDITS);
    }

    #[tokio::test]
    async fn edit_waits_while_path_lock_held() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let file_a = temp_dir.path().join("a.txt");
        let file_b = temp_dir.path().join("b.txt");
        fs::write(&file_a, "alpha\n").expect("Failed to write a.txt");
        fs::write(&file_b, "bravo\n").expect("Failed to write b.txt");
        let locks = std::sync::Arc::new(crate::tools::path_lock::PathLocks::new());
        let tool = EditFile::new(SandboxPolicy::new(temp_dir.path()), locks.clone());

        let held = locks
            .lock(&fs::canonicalize(&file_a).expect("canonicalize a.txt"))
            .await;

        let edit_a = tool.execute(serde_json::json!({
            "path": "a.txt",
            "old_string": "alpha",
            "new_string": "ALPHA",
        }));
        tokio::pin!(edit_a);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(100), &mut edit_a)
                .await
                .is_err(),
            "edit to a locked file must wait for the lock"
        );
        assert_eq!(
            fs::read_to_string(&file_a).expect("read a.txt"),
            "alpha\n",
            "locked file must be untouched while the lock is held"
        );

        let edit_b = tool.execute(serde_json::json!({
            "path": "b.txt",
            "old_string": "bravo",
            "new_string": "BRAVO",
        }));
        tokio::time::timeout(std::time::Duration::from_secs(5), edit_b)
            .await
            .expect("edit to an unlocked file must not wait")
            .expect("edit to b.txt should succeed");
        assert_eq!(fs::read_to_string(&file_b).expect("read b.txt"), "BRAVO\n");

        drop(held);
        tokio::time::timeout(std::time::Duration::from_secs(5), edit_a)
            .await
            .expect("edit must finish once the lock is released")
            .expect("edit to a.txt should succeed");
        assert_eq!(fs::read_to_string(&file_a).expect("read a.txt"), "ALPHA\n");
    }

    #[tokio::test]
    async fn dropped_edit_future_does_not_lose_other_edits() {
        let (temp_dir, file_path) = create_test_file(&numbered_lines(CONCURRENT_EDITS));
        let locks = std::sync::Arc::new(crate::tools::path_lock::PathLocks::new());
        let tool = EditFile::new(SandboxPolicy::new(temp_dir.path()), locks.clone());

        let outcomes = futures::future::join_all((0..CONCURRENT_EDITS).map(|i| {
            let exec = tool.execute(edit_line_input("test.txt", i));
            async move {
                if i % 3 == 0 {
                    tokio::time::timeout(std::time::Duration::ZERO, exec)
                        .await
                        .ok()
                } else {
                    Some(exec.await)
                }
            }
        }))
        .await;

        let _quiesced = locks
            .lock(&fs::canonicalize(&file_path).expect("canonicalize test.txt"))
            .await;
        let content = fs::read_to_string(&file_path).expect("Failed to read file");
        let mut reported_success = 0;
        for (i, outcome) in outcomes.iter().enumerate() {
            if let Some(Ok(result)) = outcome
                && !result.is_error
            {
                reported_success += 1;
                assert!(
                    content.contains(&format!("line{i}=new\n")),
                    "edit {i} reported success but its change was lost"
                );
            }
        }
        assert!(
            reported_success >= CONCURRENT_EDITS * 2 / 3,
            "every edit that was not dropped must succeed; got {reported_success}"
        );
    }
}
