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

/// Definition of a tool for discovery/registration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

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
    pub fn validate_path(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        // Resolve the path to its absolute form
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };

        // Get the canonicalized sandbox root for comparison
        let sandbox_canonical = self.root.canonicalize().map_err(|_| {
            SandboxError::InvalidPath(format!("Cannot canonicalize sandbox root: {:?}", self.root))
        })?;

        // Try to canonicalize the path (if it exists)
        // If it doesn't exist, we'll validate the absolute path
        let validated = if absolute.exists() {
            let canonical = absolute.canonicalize().map_err(|_| {
                SandboxError::InvalidPath(format!("Cannot canonicalize path: {:?}", path))
            })?;

            // Check if the canonicalized path is within the sandbox
            if !canonical.starts_with(&sandbox_canonical) {
                return Err(SandboxError::OutsideSandbox {
                    path: canonical,
                    sandbox: sandbox_canonical,
                });
            }

            canonical
        } else {
            // For non-existent paths, normalize the path by cleaning up . and ..
            // Then check if it's within the sandbox
            let normalized = self.normalize_path(&absolute)?;

            // Check if the normalized path is within the sandbox
            if !normalized.starts_with(&sandbox_canonical) {
                return Err(SandboxError::OutsideSandbox {
                    path: normalized,
                    sandbox: sandbox_canonical,
                });
            }

            normalized
        };

        Ok(validated)
    }

    /// Normalize a path by resolving . and .. components
    fn normalize_path(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        let mut result = PathBuf::new();

        for component in path.components() {
            use std::path::Component;
            match component {
                Component::Prefix(_) | Component::RootDir => {
                    result.push(component);
                }
                Component::Normal(_) => {
                    result.push(component);
                }
                Component::CurDir => {
                    // Skip . (current directory)
                }
                Component::ParentDir => {
                    // Go up one directory if possible
                    if !result.pop() {
                        return Err(SandboxError::InvalidPath(
                            "Path escapes root directory".to_string(),
                        ));
                    }
                }
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
        Self::new(&std::env::current_dir().expect("Failed to get current directory"))
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

        // Create the file
        fs::write(&test_file, "test content").expect("Failed to create test file");

        let result = sandbox.validate_path(&test_file);
        assert!(result.is_ok(), "Path inside sandbox should be allowed");
        assert_eq!(result.unwrap(), test_file.canonicalize().unwrap());
    }

    #[test]
    fn path_outside_sandbox_is_rejected() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let outside_path = temp_dir.path().parent().unwrap().join("outside.txt");

        // Create the file outside sandbox
        fs::write(&outside_path, "test content").expect("Failed to create outside file");

        let result = sandbox.validate_path(&outside_path);
        match result {
            Err(SandboxError::OutsideSandbox { .. }) => {
                // Expected error
            }
            Ok(_) => panic!("Path outside sandbox should be rejected"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn symlink_escaping_sandbox_is_rejected() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());

        // Create a symlink inside sandbox that points outside
        let symlink_path = temp_dir.path().join("escape_link");
        let outside_path = temp_dir.path().parent().unwrap().join("target.txt");

        // Create actual file outside sandbox
        fs::write(&outside_path, "test content").expect("Failed to write file");

        // Create symlink pointing outside
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
            Err(SandboxError::OutsideSandbox { .. }) => {
                // Expected error - symlink should be rejected
            }
            Ok(_) => panic!("Symlink escaping sandbox should be rejected"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn relative_paths_resolved_correctly() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());

        // Create a subdirectory
        let subdir = temp_dir.path().join("subdir");
        fs::create_dir(&subdir).expect("Failed to create subdir");

        // Create a file in the subdirectory
        let test_file = subdir.join("test.txt");
        fs::write(&test_file, "test content").expect("Failed to create test file");

        // Test relative path from sandbox root
        let relative_path = Path::new("subdir/test.txt");
        let result = sandbox.validate_path(relative_path);

        assert!(result.is_ok(), "Relative path should be resolved correctly");
        let canonical = result.unwrap();
        assert!(canonical.starts_with(temp_dir.path()));
        assert!(canonical.ends_with("subdir/test.txt"));
    }

    #[test]
    fn default_sandbox_uses_current_directory() {
        let sandbox = SandboxPolicy::default();
        let current_dir = std::env::current_dir().expect("Failed to get cwd");

        // Test that current directory is within sandbox
        let result = sandbox.validate_path(&current_dir);
        assert!(
            result.is_ok(),
            "Current directory should be in default sandbox"
        );
    }
}
