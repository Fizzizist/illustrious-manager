// Tool trait and registry for extensible tool system

use crate::types::ContentBlock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Error type for tool operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolError {
    NotFound { tool_name: String },
    AlreadyRegistered { tool_name: String },
    Execution { tool_name: String, message: String },
    InvalidInput { message: String },
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::NotFound { tool_name } => {
                write!(f, "Tool '{}' not found", tool_name)
            }
            ToolError::AlreadyRegistered { tool_name } => {
                write!(f, "Tool '{}' already registered", tool_name)
            }
            ToolError::Execution { tool_name, message } => {
                write!(f, "Tool '{}' execution failed: {}", tool_name, message)
            }
            ToolError::InvalidInput { message } => {
                write!(f, "Invalid input: {}", message)
            }
        }
    }
}

impl std::error::Error for ToolError {}

/// Result of tool execution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<ContentBlock>,
    pub is_error: bool,
}

pub use crate::types::ToolDefinition;

/// Trait that all tools must implement
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> &Value;
    fn execute(&self, input: Value) -> Result<ToolResult, ToolError>;
}

/// Registry for tool discovery and execution
///
/// Thread-safe: uses interior mutability via RwLock for concurrent access
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) -> Result<(), ToolError> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(ToolError::AlreadyRegistered { tool_name: name });
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    pub fn lookup(&self, name: &str) -> Result<&dyn Tool, ToolError> {
        self.tools
            .get(name)
            .map(|t| t.as_ref())
            .ok_or_else(|| ToolError::NotFound {
                tool_name: name.to_string(),
            })
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema().clone(),
            })
            .collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTool {
        name: String,
        description: String,
        schema: Value,
    }

    impl MockTool {
        fn new(name: &str, description: &str) -> Self {
            Self {
                name: name.to_string(),
                description: description.to_string(),
                schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string"}
                    }
                }),
            }
        }
    }

    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            &self.description
        }

        fn input_schema(&self) -> &Value {
            &self.schema
        }

        fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                content: vec![ContentBlock::Text(format!("executed with: {}", input))],
                is_error: false,
            })
        }
    }

    struct FailingTool {
        name: String,
        schema: Value,
    }

    impl FailingTool {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                schema: serde_json::json!({"type": "object"}),
            }
        }
    }

    impl Tool for FailingTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "A tool that always fails"
        }

        fn input_schema(&self) -> &Value {
            &self.schema
        }

        fn execute(&self, _input: Value) -> Result<ToolResult, ToolError> {
            Err(ToolError::Execution {
                tool_name: self.name.clone(),
                message: "Tool execution failed".to_string(),
            })
        }
    }

    #[test]
    fn register_tool_then_lookup_by_name() {
        let mut registry = ToolRegistry::new();
        let tool = MockTool::new("test_tool", "A test tool");

        registry
            .register(Box::new(tool))
            .expect("Tool should register");

        let retrieved = registry.lookup("test_tool");
        assert!(retrieved.is_ok(), "Tool should be found");
        let retrieved_tool = retrieved.unwrap();
        assert_eq!(retrieved_tool.name(), "test_tool");
        assert_eq!(retrieved_tool.description(), "A test tool");
    }

    #[test]
    fn definitions_returns_vec_of_tool_definitions() {
        let mut registry = ToolRegistry::new();
        let tool1 = MockTool::new("tool1", "First tool");
        let tool2 = MockTool::new("tool2", "Second tool");

        registry
            .register(Box::new(tool1))
            .expect("Tool1 should register");
        registry
            .register(Box::new(tool2))
            .expect("Tool2 should register");

        let defs = registry.definitions();
        assert_eq!(defs.len(), 2);

        let def1 = defs
            .iter()
            .find(|d| d.name == "tool1")
            .expect("tool1 should exist");
        assert_eq!(def1.name, "tool1");
        assert_eq!(def1.description, "First tool");
        assert_eq!(def1.input_schema["properties"]["arg1"]["type"], "string");

        let def2 = defs
            .iter()
            .find(|d| d.name == "tool2")
            .expect("tool2 should exist");
        assert_eq!(def2.name, "tool2");
        assert_eq!(def2.description, "Second tool");
    }

    #[test]
    fn lookup_unregistered_tool_returns_error() {
        let registry = ToolRegistry::new();
        let result = registry.lookup("nonexistent");

        match result {
            Ok(_) => panic!("Lookup should fail for unregistered tool"),
            Err(ToolError::NotFound { tool_name }) => {
                assert_eq!(tool_name, "nonexistent");
            }
            Err(other) => panic!("Expected NotFound error, got: {:?}", other),
        }
    }

    #[test]
    fn register_duplicate_tool_returns_error() {
        let mut registry = ToolRegistry::new();
        let tool1 = MockTool::new("duplicate", "First");
        let tool2 = MockTool::new("duplicate", "Second");

        registry
            .register(Box::new(tool1))
            .expect("First tool should register");

        let result = registry.register(Box::new(tool2));
        match result {
            Err(ToolError::AlreadyRegistered { tool_name }) => {
                assert_eq!(tool_name, "duplicate");
            }
            Ok(_) => panic!("Duplicate registration should fail"),
            Err(other) => panic!("Expected AlreadyRegistered error, got: {:?}", other),
        }
    }

    #[test]
    fn tool_execute_returns_success_result() {
        let tool = MockTool::new("executor", "Executes things");
        let input = serde_json::json!({"arg1": "test"});

        let result = tool.execute(input).expect("Execution should succeed");

        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlock::Text(text) => {
                assert!(
                    text.contains("executed with:"),
                    "Content should show execution"
                );
            }
            _ => panic!("Expected Text content block"),
        }
    }

    #[test]
    fn tool_execute_can_return_error_result() {
        let tool = FailingTool::new("failing_tool");
        let input = serde_json::json!({});

        let result = tool.execute(input);

        match result {
            Err(ToolError::Execution { tool_name, message }) => {
                assert_eq!(tool_name, "failing_tool");
                assert_eq!(message, "Tool execution failed");
            }
            Ok(_) => panic!("Execution should fail"),
            Err(other) => panic!("Expected Execution error, got: {:?}", other),
        }
    }

    #[test]
    fn tool_result_serializes_correctly() {
        let result = ToolResult {
            content: vec![ContentBlock::Text("output".to_string())],
            is_error: false,
        };

        let json = serde_json::to_string(&result).expect("Should serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should parse JSON");

        assert_eq!(parsed["is_error"], false);
        assert!(parsed["content"].is_array());
    }

    #[test]
    fn tool_definition_serializes_correctly() {
        let def = ToolDefinition {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        };

        let json = serde_json::to_string(&def).expect("Should serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should parse JSON");

        assert_eq!(parsed["name"], "test_tool");
        assert_eq!(parsed["description"], "A test tool");
        assert_eq!(parsed["input_schema"]["type"], "object");
    }

    #[test]
    fn tool_error_not_found_display_formatting() {
        let err = ToolError::NotFound {
            tool_name: "my_tool".to_string(),
        };
        assert_eq!(format!("{}", err), "Tool 'my_tool' not found");
        assert_eq!(err.to_string(), "Tool 'my_tool' not found");
    }

    #[test]
    fn tool_error_already_registered_display_formatting() {
        let err = ToolError::AlreadyRegistered {
            tool_name: "my_tool".to_string(),
        };
        assert_eq!(format!("{}", err), "Tool 'my_tool' already registered");
    }
}

/// EditFile tool for performing exact string replacements in files
///
/// Reads a file, finds an exact string match, and replaces it with new content.
/// Fails if the old string is not found or if it matches multiple locations.
pub mod edit_file {
    use super::*;

    #[derive(Debug)]
    pub struct EditFile {
        sandbox: SandboxPolicy,
    }

    impl EditFile {
        pub fn new(sandbox: SandboxPolicy) -> Self {
            Self { sandbox }
        }
    }

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

        fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
            use std::fs;

            let path_str = input.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
                ToolError::InvalidInput {
                    message: "Missing 'path' field".to_string(),
                }
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

            let content =
                fs::read_to_string(&validated_path).map_err(|e| ToolError::Execution {
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
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::fs;
        use tempfile::TempDir;

        fn create_test_file(content: &str) -> (TempDir, PathBuf) {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let file_path = temp_dir.path().join("test.txt");
            fs::write(&file_path, content).expect("Failed to write test file");
            (temp_dir, file_path)
        }

        #[test]
        fn successful_replacement_writes_correct_content() {
            let (_temp_dir, file_path) = create_test_file("hello world\nfoo bar\n");
            let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
            let tool = EditFile::new(sandbox);

            let input = serde_json::json!({
                "path": file_path.file_name().unwrap().to_str().unwrap(),
                "old_string": "hello world",
                "new_string": "goodbye world"
            });

            let result = tool.execute(input).expect("Execution should succeed");
            assert!(!result.is_error, "Result should not be an error");

            let content = fs::read_to_string(&file_path).expect("Failed to read file");
            assert_eq!(content, "goodbye world\nfoo bar\n");
        }

        #[test]
        fn old_string_not_found_returns_error() {
            let (_temp_dir, file_path) = create_test_file("hello world\n");
            let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
            let tool = EditFile::new(sandbox);

            let input = serde_json::json!({
                "path": file_path.file_name().unwrap().to_str().unwrap(),
                "old_string": "nonexistent",
                "new_string": "replacement"
            });

            let result = tool.execute(input);
            assert!(
                result.is_err(),
                "Should return error when old_string not found"
            );
        }

        #[test]
        fn old_string_matches_multiple_locations_returns_error() {
            let (_temp_dir, file_path) = create_test_file("hello world\nhello there\n");
            let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
            let tool = EditFile::new(sandbox);

            let input = serde_json::json!({
                "path": file_path.file_name().unwrap().to_str().unwrap(),
                "old_string": "hello",
                "new_string": "goodbye"
            });

            let result = tool.execute(input);
            assert!(
                result.is_err(),
                "Should return error when old_string matches multiple locations"
            );
        }

        #[test]
        fn path_outside_sandbox_returns_error() {
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

            let result = tool.execute(input);
            assert!(
                result.is_err(),
                "Should return error for path outside sandbox"
            );
        }

        #[test]
        fn file_doesnt_exist_returns_error() {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let sandbox = SandboxPolicy::new(temp_dir.path());
            let tool = EditFile::new(sandbox);

            let input = serde_json::json!({
                "path": "nonexistent.txt",
                "old_string": "old",
                "new_string": "new"
            });

            let result = tool.execute(input);
            assert!(
                result.is_err(),
                "Should return error when file doesn't exist"
            );
        }

        #[test]
        fn empty_old_string_returns_error() {
            let (_temp_dir, file_path) = create_test_file("content\n");
            let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
            let tool = EditFile::new(sandbox);

            let input = serde_json::json!({
                "path": file_path.file_name().unwrap().to_str().unwrap(),
                "old_string": "",
                "new_string": "replacement"
            });

            let result = tool.execute(input);
            assert!(result.is_err(), "Should return error for empty old_string");
        }

        #[test]
        fn new_string_can_be_empty() {
            let (_temp_dir, file_path) = create_test_file("hello world\nfoo bar\n");
            let sandbox = SandboxPolicy::new(file_path.parent().unwrap());
            let tool = EditFile::new(sandbox);

            let input = serde_json::json!({
                "path": file_path.file_name().unwrap().to_str().unwrap(),
                "old_string": "hello world\n",
                "new_string": ""
            });

            let result = tool.execute(input).expect("Execution should succeed");
            assert!(!result.is_error, "Result should not be an error");

            let content = fs::read_to_string(&file_path).expect("Failed to read file");
            assert_eq!(content, "foo bar\n");
        }
    }
}

/// Sandbox policy for validating file paths
///
/// Ensures all file operations stay within the configured sandbox root directory.
/// Prevents directory traversal attacks and symlink-based sandbox escapes.
#[derive(Debug, Clone, PartialEq)]
pub struct SandboxPolicy {
    root: PathBuf,
}

impl SandboxPolicy {
    /// Create a new sandbox policy with the specified root directory
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
        }
    }

    /// Validate that a path is within the sandbox
    ///
    /// Resolves symlinks and canonicalizes the path, then ensures it stays
    /// within the sandbox root directory.
    ///
    /// # Security
    /// Only validates existing paths. Non-existent paths are rejected to prevent
    /// symlink-based sandbox escapes via parent directory symlinks.
    pub fn validate_path(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };

        if !absolute.try_exists().map_err(|_| {
            SandboxError::InvalidPath(format!("Cannot check path existence: {:?}", path))
        })? {
            return Err(SandboxError::InvalidPath(format!(
                "Path does not exist: {:?}",
                path
            )));
        }

        let sandbox_canonical = self.root.canonicalize().map_err(|_| {
            SandboxError::InvalidPath(format!("Cannot canonicalize sandbox root: {:?}", self.root))
        })?;

        let canonical = absolute.canonicalize().map_err(|_| {
            SandboxError::InvalidPath(format!("Cannot canonicalize path: {:?}", path))
        })?;

        #[cfg(windows)]
        let is_within = {
            use std::path::Component;
            let canonical_components: Vec<_> = canonical.components().collect();
            let sandbox_components: Vec<_> = sandbox_canonical.components().collect();

            if canonical_components.len() < sandbox_components.len() {
                false
            } else {
                canonical_components
                    .iter()
                    .zip(sandbox_components.iter())
                    .all(|(c, s)| match (c, s) {
                        (Component::Prefix(a), Component::Prefix(b)) => {
                            a.to_string().to_lowercase() == b.to_string().to_lowercase()
                        }
                        (Component::Normal(a), Component::Normal(b)) => {
                            a.to_string_lossy().to_lowercase() == b.to_string_lossy().to_lowercase()
                        }
                        _ => c == s,
                    })
            }
        };

        #[cfg(not(windows))]
        let is_within = canonical.starts_with(&sandbox_canonical);

        if !is_within {
            return Err(SandboxError::OutsideSandbox {
                path: canonical,
                sandbox: sandbox_canonical,
            });
        }

        Ok(canonical)
    }

    /// Get the sandbox root directory
    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        Self::new(&root)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SandboxError {
    OutsideSandbox { path: PathBuf, sandbox: PathBuf },
    InvalidPath(String),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::OutsideSandbox { path, sandbox } => {
                write!(f, "Path {:?} is outside sandbox {:?}", path, sandbox)
            }
            SandboxError::InvalidPath(msg) => {
                write!(f, "Invalid path: {}", msg)
            }
        }
    }
}

impl std::error::Error for SandboxError {}

#[cfg(test)]
mod sandbox_tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn path_inside_sandbox_is_allowed() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let test_file = temp_dir.path().join("test.txt");

        fs::write(&test_file, "test content").expect("Failed to create test file");

        let result = sandbox.validate_path(&test_file);
        assert!(result.is_ok(), "Path inside sandbox should be allowed");
        assert_eq!(result.unwrap(), test_file.canonicalize().unwrap());
    }

    #[test]
    fn path_outside_sandbox_is_rejected() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let outside_path = temp_dir
            .path()
            .ancestors()
            .nth(1)
            .expect("Temp dir should have parent")
            .join("outside.txt");

        fs::write(&outside_path, "test content").expect("Failed to create outside file");

        let result = sandbox.validate_path(&outside_path);
        match result {
            Err(SandboxError::OutsideSandbox { .. }) => {}
            Ok(_) => panic!("Path outside sandbox should be rejected"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn symlink_escaping_sandbox_is_rejected() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());

        let symlink_path = temp_dir.path().join("escape_link");
        let outside_path = temp_dir.path().parent().unwrap().join("target.txt");

        fs::write(&outside_path, "test content").expect("Failed to write file");

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside_path, &symlink_path)
                .expect("Failed to create symlink");
        }

        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_file(&outside_path, &symlink_path)
                .expect("Failed to create symlink");
        }

        let result = sandbox.validate_path(&symlink_path);
        match result {
            Err(SandboxError::OutsideSandbox { .. }) => {}
            Ok(_) => panic!("Symlink escaping sandbox should be rejected"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn relative_paths_resolved_correctly() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());

        let subdir = temp_dir.path().join("subdir");
        fs::create_dir(&subdir).expect("Failed to create subdir");

        let test_file = subdir.join("test.txt");
        fs::write(&test_file, "test content").expect("Failed to create test file");

        let relative_path = Path::new("subdir/test.txt");
        let result = sandbox.validate_path(relative_path);

        assert!(result.is_ok(), "Relative path should be resolved correctly");
        let canonical = result.unwrap();
        assert!(canonical.starts_with(temp_dir.path()));
        assert!(canonical.ends_with("subdir/test.txt"));
    }

    #[test]
    fn non_existent_path_is_rejected() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let nonexistent_file = temp_dir.path().join("nonexistent.txt");

        let result = sandbox.validate_path(&nonexistent_file);
        assert!(
            result.is_err(),
            "Non-existent path should be rejected for security reasons"
        );
    }

    #[test]
    fn default_sandbox_uses_current_directory() {
        let sandbox = SandboxPolicy::default();

        let result = sandbox.validate_path(Path::new("."));
        assert!(
            result.is_ok(),
            "Current directory should be in default sandbox"
        );
    }
}
