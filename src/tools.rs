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

    /// Validate a path for writing (file may not yet exist)
    ///
    /// Walks up the path to find the nearest existing ancestor, canonicalizes it,
    /// and verifies the intended write location stays within the sandbox.
    ///
    /// # Security
    /// Rejects paths containing `..` components in the non-existing portion.
    /// Symlinks in existing ancestors are resolved and checked against the sandbox.
    pub fn validate_write_path(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };

        let sandbox_canonical = self.root.canonicalize().map_err(|_| {
            SandboxError::InvalidPath(format!("Cannot canonicalize sandbox root: {:?}", self.root))
        })?;

        let mut existing_ancestor = absolute.clone();
        let mut pending: Vec<std::ffi::OsString> = vec![];

        loop {
            match existing_ancestor.try_exists() {
                Ok(true) => break,
                Ok(false) => {}
                Err(_) => {
                    return Err(SandboxError::InvalidPath(format!(
                        "Cannot check path existence: {:?}",
                        existing_ancestor
                    )));
                }
            }

            let component = existing_ancestor
                .file_name()
                .ok_or_else(|| {
                    SandboxError::InvalidPath(format!("Cannot resolve write path: {:?}", path))
                })?
                .to_os_string();

            if component == ".." {
                return Err(SandboxError::InvalidPath(
                    "Path traversal via '..' is not allowed".to_string(),
                ));
            }

            pending.push(component);

            existing_ancestor = existing_ancestor
                .parent()
                .ok_or_else(|| {
                    SandboxError::InvalidPath(format!("Cannot resolve write path: {:?}", path))
                })?
                .to_path_buf();
        }

        let canonical_ancestor = existing_ancestor.canonicalize().map_err(|_| {
            SandboxError::InvalidPath(format!(
                "Cannot canonicalize ancestor: {:?}",
                existing_ancestor
            ))
        })?;

        if !canonical_ancestor.starts_with(&sandbox_canonical) {
            return Err(SandboxError::OutsideSandbox {
                path: canonical_ancestor,
                sandbox: sandbox_canonical,
            });
        }

        let mut result = canonical_ancestor;
        for component in pending.iter().rev() {
            result = result.join(component);
            if !result.starts_with(&sandbox_canonical) {
                return Err(SandboxError::OutsideSandbox {
                    path: result,
                    sandbox: sandbox_canonical,
                });
            }
        }

        Ok(result)
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

    #[test]
    fn validate_write_path_allows_nonexistent_file_inside_sandbox() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let new_file = temp_dir.path().join("new_file.txt");

        let result = sandbox.validate_write_path(&new_file);
        assert!(
            result.is_ok(),
            "Non-existent path inside sandbox should be allowed for writing"
        );
        assert_eq!(result.unwrap(), new_file);
    }

    #[test]
    fn validate_write_path_rejects_nonexistent_file_outside_sandbox() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let parent = temp_dir
            .path()
            .parent()
            .expect("Temp dir should have parent");
        let outside_file = parent.join("outside_new.txt");

        let result = sandbox.validate_write_path(&outside_file);
        match result {
            Err(SandboxError::OutsideSandbox { .. }) => {}
            Ok(_) => panic!("Path outside sandbox should be rejected"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn validate_write_path_rejects_symlink_parent_escaping_sandbox() {
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

        let target = symlink_dir.join("file.txt");
        let result = sandbox.validate_write_path(&target);
        match result {
            Err(SandboxError::OutsideSandbox { .. }) => {}
            Ok(_) => panic!("Symlink escaping sandbox should be rejected"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn validate_write_path_rejects_dotdot_traversal() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());

        let traversal = temp_dir
            .path()
            .join("subdir")
            .join("..")
            .join("..")
            .join("escape.txt");
        let result = sandbox.validate_write_path(&traversal);
        assert!(result.is_err(), "Path traversal should be rejected");
    }
}

/// Tool that creates or overwrites a file within the sandbox
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

        std::fs::write(&validated, content).map_err(|e| ToolError::Execution {
            tool_name: self.name().to_string(),
            message: format!("Failed to write file: {}", e),
        })?;

        let bytes = content.len();
        Ok(ToolResult {
            content: vec![crate::types::ContentBlock::Text(format!(
                "Wrote {} bytes to {:?}",
                bytes, validated
            ))],
            is_error: false,
        })
    }
}

#[cfg(test)]
mod write_file_tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn write_file_creates_new_file_with_correct_content() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let tool = WriteFileTool::new(sandbox);

        let file_path = temp_dir.path().join("hello.txt");
        let input = serde_json::json!({
            "path": file_path.to_str().unwrap(),
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
            "path": file_path.to_str().unwrap(),
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
            "path": file_path.to_str().unwrap(),
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
            "path": outside_path.to_str().unwrap(),
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
            "path": target.to_str().unwrap(),
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
            "path": file_path.to_str().unwrap(),
            "content": content
        });

        let result = tool.execute(input).expect("Write should succeed");
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            crate::types::ContentBlock::Text(msg) => {
                assert!(msg.contains("5"), "Should report 5 bytes written");
            }
            _ => panic!("Expected text content block"),
        }
    }
}
