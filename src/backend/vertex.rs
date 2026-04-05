use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;

use super::LlmBackend;
use super::sse::create_sse_event_stream;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

/// Required protocol version string for Vertex AI's Anthropic API.
const ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

/// Stateful SSE parser that tracks which block indices are tool_use blocks.
pub struct VertexSseParser {
    tool_use_indices: HashSet<u64>,
}

impl VertexSseParser {
    pub fn new() -> Self {
        Self {
            tool_use_indices: HashSet::new(),
        }
    }

    pub fn parse(&mut self, data: &str) -> Result<Option<StreamEvent>> {
        let json: serde_json::Value = serde_json::from_str(data)
            .with_context(|| format!("Failed to parse SSE data: {}", data))?;

        let event_type = json["type"].as_str().unwrap_or("");

        match event_type {
            "content_block_start" => {
                let block_type = json["content_block"]["type"].as_str().unwrap_or("");
                if block_type == "tool_use" {
                    let index = json["index"].as_u64().unwrap_or(0);
                    self.tool_use_indices.insert(index);
                    let id = json["content_block"]["id"]
                        .as_str()
                        .ok_or_else(|| {
                            anyhow::anyhow!("tool_use content_block_start missing 'id'")
                        })?
                        .to_string();
                    let name = json["content_block"]["name"]
                        .as_str()
                        .ok_or_else(|| {
                            anyhow::anyhow!("tool_use content_block_start missing 'name'")
                        })?
                        .to_string();
                    Ok(Some(StreamEvent::ToolUseStart { id, name }))
                } else {
                    Ok(None)
                }
            }
            "content_block_delta" => {
                let delta_type = json["delta"]["type"].as_str().unwrap_or("");
                if delta_type == "input_json_delta" {
                    let chunk = json["delta"]["partial_json"]
                        .as_str()
                        .ok_or_else(|| anyhow::anyhow!("input_json_delta missing 'partial_json'"))?
                        .to_string();
                    Ok(Some(StreamEvent::ToolUseDelta(chunk)))
                } else {
                    let text = json["delta"]["text"].as_str().unwrap_or("").to_string();
                    Ok(Some(StreamEvent::TextDelta(text)))
                }
            }
            "content_block_stop" => {
                let index = json["index"].as_u64().unwrap_or(0);
                if self.tool_use_indices.remove(&index) {
                    Ok(Some(StreamEvent::ToolUseDone))
                } else {
                    Ok(None)
                }
            }
            "message_stop" => Ok(Some(StreamEvent::Done)),
            _ => Ok(None),
        }
    }
}

/// Vertex AI backend for Claude models.
pub struct VertexBackend {
    client: Client,
    project: String,
    region: String,
    auth_manager: Arc<dyn gcp_auth::TokenProvider>,
}

impl VertexBackend {
    /// Create a new VertexBackend using Application Default Credentials.
    pub async fn new(project: String, region: String) -> Result<Self> {
        let auth_manager = gcp_auth::provider().await.context(
            "Failed to initialize GCP authentication. Run: gcloud auth application-default login",
        )?;
        Ok(Self {
            client: Client::new(),
            project,
            region,
            auth_manager,
        })
    }

    fn endpoint(&self, model: &str) -> String {
        format!(
            "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:streamRawPredict",
            region = self.region,
            project = self.project,
            model = model,
        )
    }
}

#[async_trait]
impl LlmBackend for VertexBackend {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        let scopes = &["https://www.googleapis.com/auth/cloud-platform"];
        let token = self
            .auth_manager
            .token(scopes)
            .await
            .context("Failed to get GCP auth token")?;
        let token_str = token.as_str().to_owned();

        let url = self.endpoint(&config.model);
        let body = build_request_body(messages, config)?;

        let response = self
            .client
            .post(&url)
            .bearer_auth(&token_str)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Vertex AI")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<failed to read response body>"));
            anyhow::bail!("Vertex AI returned {}: {}", status, body);
        }

        let byte_stream = response.bytes_stream();
        let mut sse_parser = VertexSseParser::new();
        let event_stream = create_sse_event_stream(byte_stream, move |data| sse_parser.parse(data));
        Ok(event_stream)
    }
}

fn build_request_body(messages: &[Message], config: &RequestConfig) -> Result<serde_json::Value> {
    let messages_json: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            serde_json::json!({
                "role": m.role,
                "content": m.content,
            })
        })
        .collect();

    let mut body = serde_json::json!({
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": config.max_tokens,
        "stream": true,
        "messages": messages_json,
    });

    if !config.tools.is_empty() {
        body["tools"] =
            serde_json::to_value(&config.tools).context("Failed to serialize tool definitions")?;
    }

    Ok(body)
}

/// Convenience wrapper for single-event parsing without state.
/// For testing individual events in isolation; use VertexSseParser for sequences.
pub fn parse_sse_data(data: &str) -> Result<Option<StreamEvent>> {
    VertexSseParser::new().parse(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolDefinition;

    #[test]
    fn build_request_body_uses_max_tokens_and_model_from_config() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 32768,
            tools: vec![],
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert_eq!(body["max_tokens"], 32768);
        assert_eq!(body["anthropic_version"], ANTHROPIC_VERSION);
    }

    #[test]
    fn build_request_body_omits_model_field_for_vertex_ai() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert!(
            body.get("model").is_none() || body["model"].is_null(),
            "model must not be in the request body; Vertex AI embeds it in the URL"
        );
    }

    #[test]
    fn build_request_body_includes_tools_when_non_empty() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![ToolDefinition {
                name: "bash".to_string(),
                description: "Run a bash command".to_string(),
                input_schema: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            }],
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        let tools = body["tools"].as_array().expect("tools should be an array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "bash");
        assert_eq!(tools[0]["description"], "Run a bash command");
    }

    #[test]
    fn build_request_body_omits_tools_key_when_empty() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert!(
            body.get("tools").is_none() || body["tools"].is_null(),
            "tools must not be in the request body when empty"
        );
    }

    #[test]
    fn parser_content_block_start_tool_use_returns_tool_use_start() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"bash"}}"#;
        let result = parser.parse(data).expect("should parse successfully");
        match result {
            Some(StreamEvent::ToolUseStart { id, name }) => {
                assert_eq!(id, "toolu_01");
                assert_eq!(name, "bash");
            }
            other => panic!("expected ToolUseStart, got {:?}", other),
        }
    }

    #[test]
    fn parser_input_json_delta_returns_tool_use_delta() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"command\":"}}"#;
        let result = parser.parse(data).expect("should parse successfully");
        match result {
            Some(StreamEvent::ToolUseDelta(chunk)) => {
                assert_eq!(chunk, "{\"command\":");
            }
            other => panic!("expected ToolUseDelta, got {:?}", other),
        }
    }

    #[test]
    fn parser_content_block_stop_after_tool_use_returns_tool_use_done() {
        let mut parser = VertexSseParser::new();
        parser
            .parse(r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"bash"}}"#)
            .expect("should parse block_start");
        let result = parser
            .parse(r#"{"type":"content_block_stop","index":1}"#)
            .expect("should parse block_stop");
        assert!(
            matches!(result, Some(StreamEvent::ToolUseDone)),
            "expected ToolUseDone, got {:?}",
            result
        );
    }

    #[test]
    fn parser_content_block_stop_after_text_block_returns_none() {
        let mut parser = VertexSseParser::new();
        parser
            .parse(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#)
            .expect("should parse block_start");
        let result = parser
            .parse(r#"{"type":"content_block_stop","index":0}"#)
            .expect("should parse block_stop");
        assert!(
            result.is_none(),
            "stopping a text block should return None, got {:?}",
            result
        );
    }

    #[test]
    fn parser_mixed_text_and_tool_use_content_block_stop_fires_only_for_tool_use() {
        let mut parser = VertexSseParser::new();
        parser
            .parse(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#)
            .expect("text block_start");
        parser
            .parse(r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"bash"}}"#)
            .expect("tool_use block_start");

        let text_stop = parser
            .parse(r#"{"type":"content_block_stop","index":0}"#)
            .expect("text block_stop");
        assert!(
            text_stop.is_none(),
            "text block_stop should be None, got {:?}",
            text_stop
        );

        let tool_stop = parser
            .parse(r#"{"type":"content_block_stop","index":1}"#)
            .expect("tool_use block_stop");
        assert!(
            matches!(tool_stop, Some(StreamEvent::ToolUseDone)),
            "tool_use block_stop should be ToolUseDone, got {:?}",
            tool_stop
        );
    }

    #[test]
    fn parser_content_block_start_tool_use_missing_id_returns_error() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","name":"bash"}}"#;
        assert!(
            parser.parse(data).is_err(),
            "missing 'id' should return error"
        );
    }

    #[test]
    fn parser_content_block_start_tool_use_missing_name_returns_error() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01"}}"#;
        assert!(
            parser.parse(data).is_err(),
            "missing 'name' should return error"
        );
    }

    #[test]
    fn parser_input_json_delta_missing_partial_json_returns_error() {
        let mut parser = VertexSseParser::new();
        let data =
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta"}}"#;
        assert!(
            parser.parse(data).is_err(),
            "missing 'partial_json' should return error"
        );
    }

    #[test]
    fn parser_text_delta_returns_text_delta() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let result = parser.parse(data).expect("should parse successfully");
        match result {
            Some(StreamEvent::TextDelta(text)) => assert_eq!(text, "Hello"),
            other => panic!("expected TextDelta, got {:?}", other),
        }
    }

    #[test]
    fn parser_content_block_start_text_type_returns_none() {
        let mut parser = VertexSseParser::new();
        let data =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let result = parser.parse(data).expect("should parse successfully");
        assert!(
            result.is_none(),
            "text content_block_start should return None"
        );
    }
}
