// Tool trait and registry for extensible tool system

use serde_json::Value;
use std::collections::HashMap;

/// Result of tool execution
#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

/// Definition of a tool for discovery/registration
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Trait that all tools must implement
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> Value;
    fn execute(&self, input: Value) -> Result<ToolResult, String>;
}

/// Registry for tool discovery and execution
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) -> Result<(), String> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(format!("Tool '{}' already registered", name));
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    pub fn lookup(&self, name: &str) -> Result<&dyn Tool, String> {
        self.tools
            .get(name)
            .map(|t| t.as_ref())
            .ok_or_else(|| format!("Tool '{}' not found", name))
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
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
    }

    impl MockTool {
        fn new(name: &str, description: &str) -> Self {
            Self {
                name: name.to_string(),
                description: description.to_string(),
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

        fn input_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"}
                }
            })
        }

        fn execute(&self, input: Value) -> Result<ToolResult, String> {
            Ok(ToolResult {
                content: format!("executed with: {}", input),
                is_error: false,
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
            Err(err) => {
                assert!(
                    err.contains("nonexistent"),
                    "Error should mention tool name"
                );
                assert!(err.contains("not found"), "Error should say tool not found");
            }
        }
    }
}
