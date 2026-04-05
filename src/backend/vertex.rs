use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;

use super::LlmBackend;
use super::sse::create_sse_event_stream;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

/// Required protocol version string for Vertex AI's Anthropic API.
const ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

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
        let body = build_request_body(messages, config);

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
        let event_stream = create_sse_event_stream(byte_stream, parse_sse_data);
        Ok(event_stream)
    }
}

fn build_request_body(messages: &[Message], config: &RequestConfig) -> serde_json::Value {
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
        body["tools"] = serde_json::to_value(&config.tools)
            .expect("ToolDefinition must serialize to valid JSON");
    }

    body
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
        let body = build_request_body(&[], &config);
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
        let body = build_request_body(&[], &config);
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
        let body = build_request_body(&[], &config);
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
        let body = build_request_body(&[], &config);
        assert!(
            body.get("tools").is_none() || body["tools"].is_null(),
            "tools must not be in the request body when empty"
        );
    }

    #[test]
    fn parse_sse_data_content_block_start_tool_use_returns_tool_use_start() {
        let data = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"bash"}}"#;
        let result = parse_sse_data(data).expect("should parse successfully");
        match result {
            Some(StreamEvent::ToolUseStart { id, name }) => {
                assert_eq!(id, "toolu_01");
                assert_eq!(name, "bash");
            }
            other => panic!("expected ToolUseStart, got {:?}", other),
        }
    }

    #[test]
    fn parse_sse_data_input_json_delta_returns_tool_use_delta() {
        let data = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"command\":"}}"#;
        let result = parse_sse_data(data).expect("should parse successfully");
        match result {
            Some(StreamEvent::ToolUseDelta(chunk)) => {
                assert_eq!(chunk, "{\"command\":");
            }
            other => panic!("expected ToolUseDelta, got {:?}", other),
        }
    }

    #[test]
    fn parse_sse_data_content_block_stop_returns_tool_use_done() {
        let data = r#"{"type":"content_block_stop","index":1}"#;
        let result = parse_sse_data(data).expect("should parse successfully");
        assert!(
            matches!(result, Some(StreamEvent::ToolUseDone)),
            "expected ToolUseDone, got {:?}",
            result
        );
    }

    #[test]
    fn parse_sse_data_text_delta_still_returns_text_delta() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let result = parse_sse_data(data).expect("should parse successfully");
        match result {
            Some(StreamEvent::TextDelta(text)) => assert_eq!(text, "Hello"),
            other => panic!("expected TextDelta, got {:?}", other),
        }
    }

    #[test]
    fn parse_sse_data_content_block_start_text_type_returns_none() {
        let data =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let result = parse_sse_data(data).expect("should parse successfully");
        assert!(
            result.is_none(),
            "text content_block_start should return None"
        );
    }
}

/// Parse an SSE data payload JSON into a StreamEvent.
/// Returns None for event types we intentionally ignore.
pub fn parse_sse_data(data: &str) -> Result<Option<StreamEvent>> {
    let json: serde_json::Value = serde_json::from_str(data)
        .with_context(|| format!("Failed to parse SSE data: {}", data))?;

    let event_type = json["type"].as_str().unwrap_or("");

    match event_type {
        "content_block_start" => {
            let block_type = json["content_block"]["type"].as_str().unwrap_or("");
            if block_type == "tool_use" {
                let id = json["content_block"]["id"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                let name = json["content_block"]["name"]
                    .as_str()
                    .unwrap_or("")
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
                    .unwrap_or("")
                    .to_string();
                Ok(Some(StreamEvent::ToolUseDelta(chunk)))
            } else {
                let text = json["delta"]["text"].as_str().unwrap_or("").to_string();
                Ok(Some(StreamEvent::TextDelta(text)))
            }
        }
        "content_block_stop" => Ok(Some(StreamEvent::ToolUseDone)),
        "message_stop" => Ok(Some(StreamEvent::Done)),
        _ => Ok(None),
    }
}
