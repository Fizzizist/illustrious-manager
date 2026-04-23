// Tool trait and registry for extensible tool system

pub mod agent;
pub mod bash;
pub mod edit_file;
pub mod sandbox;
pub mod search;
pub mod skill;
pub mod task;
pub mod write_file;

use crate::types::ContentBlock;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<ContentBlock>,
    pub is_error: bool,
    /// Additional agent-level events to forward to the event stream (e.g. SubAgentUsage).
    #[serde(skip)]
    pub agent_events: Vec<crate::types::AgentEvent>,
}

pub use crate::types::ToolDefinition;

/// Trait that all tools must implement
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> &Value;
    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError>;
    fn is_write_tool(&self) -> bool {
        false
    }

    fn markdown_input(&self, input: &Value) -> String {
        format!(
            "```json\n{}\n```",
            serde_json::to_string_pretty(input).unwrap_or_else(|_| "{}".to_string())
        )
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

    /// Consume this registry and return a new one containing only the tools
    /// whose names appear in `allowlist`. Names not found in the registry are
    /// silently skipped; a warning is printed if the result is empty.
    pub fn into_filtered(mut self, allowlist: &[String]) -> Self {
        let allowed: std::collections::HashSet<&str> =
            allowlist.iter().map(String::as_str).collect();
        self.tools.retain(|name, _| allowed.contains(name.as_str()));
        if self.tools.is_empty() && !allowlist.is_empty() {
            eprintln!(
                "WARNING: agent tool allowlist [{list}] matched no registered tools; \
                 sub-agent will run with an empty tool set",
                list = allowlist.join(", ")
            );
        }
        self
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

    #[async_trait]
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

        async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                content: vec![ContentBlock::Text(format!("executed with: {}", input))],
                is_error: false,
                agent_events: vec![],
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

    #[async_trait]
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

        async fn execute(&self, _input: Value) -> Result<ToolResult, ToolError> {
            Err(ToolError::Execution {
                tool_name: self.name.clone(),
                message: "Tool execution failed".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn register_tool_then_lookup_by_name() {
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

    #[tokio::test]
    async fn definitions_returns_vec_of_tool_definitions() {
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

    #[tokio::test]
    async fn lookup_unregistered_tool_returns_error() {
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

    #[tokio::test]
    async fn register_duplicate_tool_returns_error() {
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

    #[tokio::test]
    async fn tool_execute_returns_success_result() {
        let tool = MockTool::new("executor", "Executes things");
        let input = serde_json::json!({"arg1": "test"});

        let result = tool.execute(input).await.expect("Execution should succeed");

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

    #[tokio::test]
    async fn tool_execute_can_return_error_result() {
        let tool = FailingTool::new("failing_tool");
        let input = serde_json::json!({});

        let result = tool.execute(input).await;

        match result {
            Err(ToolError::Execution { tool_name, message }) => {
                assert_eq!(tool_name, "failing_tool");
                assert_eq!(message, "Tool execution failed");
            }
            Ok(_) => panic!("Execution should fail"),
            Err(other) => panic!("Expected Execution error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn tool_result_serializes_correctly() {
        let result = ToolResult {
            content: vec![ContentBlock::Text("output".to_string())],
            is_error: false,
            agent_events: vec![],
        };

        let json = serde_json::to_string(&result).expect("Should serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should parse JSON");

        assert_eq!(parsed["is_error"], false);
        assert!(parsed["content"].is_array());
    }

    #[tokio::test]
    async fn tool_definition_serializes_correctly() {
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

    #[tokio::test]
    async fn tool_error_not_found_display_formatting() {
        let err = ToolError::NotFound {
            tool_name: "my_tool".to_string(),
        };
        assert_eq!(format!("{}", err), "Tool 'my_tool' not found");
        assert_eq!(err.to_string(), "Tool 'my_tool' not found");
    }

    #[tokio::test]
    async fn tool_error_already_registered_display_formatting() {
        let err = ToolError::AlreadyRegistered {
            tool_name: "my_tool".to_string(),
        };
        assert_eq!(format!("{}", err), "Tool 'my_tool' already registered");
    }

    #[tokio::test]
    async fn default_markdown_input_formats_as_json_code_block() {
        let tool = MockTool::new("test", "A test tool");
        let input = serde_json::json!({"arg1": "value"});
        let md = tool.markdown_input(&input);
        assert!(md.contains("```json"), "should wrap in json code block");
        assert!(md.contains("arg1"), "should contain the field name");
    }

    #[tokio::test]
    async fn default_markdown_output_returns_text_content() {
        let tool = MockTool::new("test", "A test tool");
        let result = tool
            .execute(serde_json::json!({"arg1": "hello"}))
            .await
            .expect("should succeed");
        let md = tool.markdown_output(&result);
        assert!(
            md.contains("executed with:"),
            "should contain execution result text"
        );
    }
}
