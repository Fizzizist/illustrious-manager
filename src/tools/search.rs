use crate::types::ContentBlock;
use async_trait::async_trait;
use search_semantically::SearchEngine;
use serde_json::Value;
use std::path::{Path, PathBuf};

use super::{Tool, ToolError, ToolResult, run_blocking};

const TOOL: &str = "search";

pub struct SearchTool {
    sandbox_root: PathBuf,
    schema: Value,
}

impl SearchTool {
    pub fn new(sandbox_root: PathBuf) -> Self {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — natural language, identifier name, or file path pattern"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 20)",
                    "default": 20
                },
                "restrictToDir": {
                    "type": "string",
                    "description": "Restrict search to files under this directory (relative to project root)"
                },
                "rebuild": {
                    "type": "boolean",
                    "description": "Force rebuild of the search index (default: false)",
                    "default": false
                }
            },
            "required": ["query"]
        });
        Self {
            sandbox_root,
            schema,
        }
    }

    fn delete_index_db(root: &Path) {
        for suffix in &["search.db", "search.db-wal", "search.db-shm"] {
            let path = root.join(".search-index").join(suffix);
            let _ = std::fs::remove_file(&path);
        }
    }
}

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str {
        TOOL
    }

    fn description(&self) -> &str {
        "Search the codebase using semantic code search. Supports natural language queries, identifier names, and file path patterns. Returns ranked code chunks with relevance scores."
    }

    fn input_schema(&self) -> &Value {
        &self.schema
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let query = input["query"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput {
                message: "Missing required 'query' field".to_string(),
            })?;

        let limit = input["limit"].as_u64().unwrap_or(20) as usize;
        let restrict_to_dir = input["restrictToDir"].as_str().map(String::from);
        let rebuild = input["rebuild"].as_bool().unwrap_or(false);
        let query = query.to_string();
        let sandbox_root = self.sandbox_root.clone();

        let output = run_blocking(TOOL, move || {
            if rebuild {
                Self::delete_index_db(&sandbox_root);
            }

            let engine = SearchEngine::new(sandbox_root);
            engine
                .search(&query, limit, restrict_to_dir.as_deref())
                .map_err(|e| ToolError::Execution {
                    tool_name: TOOL.to_string(),
                    message: e.to_string(),
                })
        })
        .await?;

        Ok(ToolResult {
            content: vec![ContentBlock::Text(output)],
            is_error: false,
            agent_events: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn search_tool_has_correct_name() {
        let temp = TempDir::new().expect("temp dir");
        let tool = SearchTool::new(temp.path().to_path_buf());
        assert_eq!(tool.name(), "search");
    }

    #[tokio::test]
    async fn search_tool_has_description() {
        let temp = TempDir::new().expect("temp dir");
        let tool = SearchTool::new(temp.path().to_path_buf());
        assert!(!tool.description().is_empty());
    }

    #[tokio::test]
    async fn search_tool_input_schema_has_query() {
        let temp = TempDir::new().expect("temp dir");
        let tool = SearchTool::new(temp.path().to_path_buf());
        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"]
            .as_array()
            .expect("required should be array");
        assert!(required.iter().any(|r| r == "query"));
    }

    #[tokio::test]
    async fn search_tool_is_not_a_write_tool() {
        let temp = TempDir::new().expect("temp dir");
        let tool = SearchTool::new(temp.path().to_path_buf());
        assert!(!tool.is_write_tool());
    }

    #[tokio::test]
    async fn search_tool_requires_query_field() {
        let temp = TempDir::new().expect("temp dir");
        let tool = SearchTool::new(temp.path().to_path_buf());
        let result = tool.execute(serde_json::json!({})).await;
        assert!(result.is_err());
        match result {
            Err(ToolError::InvalidInput { message }) => {
                assert!(message.contains("query"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn search_tool_registered_in_registry() {
        let temp = TempDir::new().expect("temp dir");
        let mut registry = crate::tools::ToolRegistry::new();
        let tool = SearchTool::new(temp.path().to_path_buf());
        registry.register(Box::new(tool)).expect("should register");
        let def = registry.lookup("search").expect("should find search tool");
        assert_eq!(def.name(), "search");
    }

    #[tokio::test]
    async fn rebuild_deletes_index_db() {
        let temp = TempDir::new().expect("temp dir");
        let index_dir = temp.path().join(".search-index");
        std::fs::create_dir_all(&index_dir).expect("dir");
        std::fs::write(index_dir.join("search.db"), "fake db content").expect("write");
        std::fs::write(index_dir.join("search.db-wal"), "wal").expect("write");

        SearchTool::delete_index_db(temp.path());

        assert!(!index_dir.join("search.db").exists());
        assert!(!index_dir.join("search.db-wal").exists());
    }

    #[tokio::test]
    async fn execute_returns_results_for_populated_project() {
        let temp = TempDir::new().expect("temp dir");
        std::fs::write(
            temp.path().join("main.rs"),
            "fn calculate_total(prices: &[f64]) -> f64 {\n    prices.iter().sum()\n}\n",
        )
        .expect("write");

        let tool = SearchTool::new(temp.path().to_path_buf());
        let result = tool
            .execute(serde_json::json!({
                "query": "calculate_total"
            }))
            .await
            .expect("execute should succeed");

        assert!(!result.is_error);
        assert!(
            result.content.len() == 1,
            "expected exactly one content block"
        );
        match &result.content[0] {
            ContentBlock::Text(text) => {
                assert!(
                    text.contains("main.rs"),
                    "result should reference main.rs, got: {text}"
                );
            }
            _ => panic!("expected Text content block"),
        }
    }

    #[tokio::test]
    async fn execute_with_rebuild_fresh_search() {
        let temp = TempDir::new().expect("temp dir");
        std::fs::write(
            temp.path().join("lib.rs"),
            "pub fn process_data(input: &str) -> String {\n    input.to_uppercase()\n}\n",
        )
        .expect("write");

        let tool = SearchTool::new(temp.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "query": "process_data",
                "rebuild": true
            }))
            .await
            .expect("execute with rebuild should succeed");

        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn execute_empty_project_returns_no_results() {
        let temp = TempDir::new().expect("temp dir");

        let tool = SearchTool::new(temp.path().to_path_buf());
        let result = tool
            .execute(serde_json::json!({
                "query": "anything"
            }))
            .await
            .expect("execute should succeed");

        assert!(!result.is_error);
        match &result.content[0] {
            ContentBlock::Text(text) => {
                assert_eq!(text, "No results found.");
            }
            _ => panic!("expected Text content block"),
        }
    }
}
