use crate::types::ContentBlock;
use search_semantically::SearchEngine;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::{Tool, ToolError, ToolResult};

pub struct SearchTool {
    sandbox_root: PathBuf,
    engine: Arc<Mutex<Option<SearchEngine>>>,
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
            engine: Arc::new(Mutex::new(None)),
            schema,
        }
    }

    fn get_or_create_engine(&self) -> SearchEngine {
        let mut guard = self.engine.lock().expect("SearchTool engine lock poisoned");
        if guard.is_none() {
            *guard = Some(SearchEngine::new(self.sandbox_root.clone()));
        }
        guard
            .clone()
            .expect("engine should be Some after initialization")
    }

    fn delete_index_db(&self) {
        let db_path = self.sandbox_root.join(".search-index").join("search.db");
        let _ = std::fs::remove_file(&db_path);
        let wal_path = self
            .sandbox_root
            .join(".search-index")
            .join("search.db-wal");
        let _ = std::fs::remove_file(&wal_path);
        let shm_path = self
            .sandbox_root
            .join(".search-index")
            .join("search.db-shm");
        let _ = std::fs::remove_file(&shm_path);
    }
}

impl Tool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Search the codebase using semantic code search. Supports natural language queries, identifier names, and file path patterns. Returns ranked code chunks with relevance scores."
    }

    fn input_schema(&self) -> &Value {
        &self.schema
    }

    fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let query = input["query"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput {
                message: "Missing required 'query' field".to_string(),
            })?;

        let limit = input["limit"].as_u64().unwrap_or(20) as usize;
        let restrict_to_dir = input["restrictToDir"].as_str().map(String::from);
        let rebuild = input["rebuild"].as_bool().unwrap_or(false);

        if rebuild {
            self.delete_index_db();
            let mut guard = self.engine.lock().expect("SearchTool engine lock poisoned");
            *guard = None;
        }

        let eng = self.get_or_create_engine();
        let result = eng.search(query, limit, restrict_to_dir.as_deref());

        match result {
            Ok(output) => Ok(ToolResult {
                content: vec![ContentBlock::Text(output)],
                is_error: false,
            }),
            Err(e) => Err(ToolError::Execution {
                tool_name: "search".to_string(),
                message: e.to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn search_tool_has_correct_name() {
        let temp = TempDir::new().expect("temp dir");
        let tool = SearchTool::new(temp.path().to_path_buf());
        assert_eq!(tool.name(), "search");
    }

    #[test]
    fn search_tool_has_description() {
        let temp = TempDir::new().expect("temp dir");
        let tool = SearchTool::new(temp.path().to_path_buf());
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn search_tool_input_schema_has_query() {
        let temp = TempDir::new().expect("temp dir");
        let tool = SearchTool::new(temp.path().to_path_buf());
        let schema = tool.input_schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"]
            .as_array()
            .expect("required should be array");
        assert!(required.iter().any(|r| r == "query"));
    }

    #[test]
    fn search_tool_is_not_a_write_tool() {
        let temp = TempDir::new().expect("temp dir");
        let tool = SearchTool::new(temp.path().to_path_buf());
        assert!(!tool.is_write_tool());
    }

    #[test]
    fn search_tool_requires_query_field() {
        let temp = TempDir::new().expect("temp dir");
        let tool = SearchTool::new(temp.path().to_path_buf());
        let result = tool.execute(serde_json::json!({}));
        assert!(result.is_err());
        match result {
            Err(ToolError::InvalidInput { message }) => {
                assert!(message.contains("query"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn search_tool_registered_in_registry() {
        let temp = TempDir::new().expect("temp dir");
        let mut registry = crate::tools::ToolRegistry::new();
        let tool = SearchTool::new(temp.path().to_path_buf());
        registry.register(Box::new(tool)).expect("should register");
        let def = registry.lookup("search").expect("should find search tool");
        assert_eq!(def.name(), "search");
    }

    #[test]
    fn rebuild_deletes_index_db() {
        let temp = TempDir::new().expect("temp dir");
        let index_dir = temp.path().join(".search-index");
        std::fs::create_dir_all(&index_dir).expect("dir");
        std::fs::write(index_dir.join("search.db"), "fake db content").expect("write");
        std::fs::write(index_dir.join("search.db-wal"), "wal").expect("write");

        let tool = SearchTool::new(temp.path().to_path_buf());
        tool.delete_index_db();

        assert!(!index_dir.join("search.db").exists());
        assert!(!index_dir.join("search.db-wal").exists());
    }
}
