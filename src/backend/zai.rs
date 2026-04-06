use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use std::collections::HashMap;

use super::LlmBackend;
use super::sse::create_sse_event_stream;
use crate::types::{BoxStream, ContentBlock, Message, RequestConfig, StreamEvent};

const ENDPOINT: &str = "https://api.z.ai/api/coding/paas/v4/chat/completions";

/// Stateful SSE parser that tracks tool calls by index for OpenAI-compatible streaming.
#[derive(Default)]
pub struct ZaiSseParser {
    /// Track tool calls by their index to generate stable IDs
    tool_calls_by_index: HashMap<u64, String>,
}

impl ZaiSseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parse(&mut self, data: &str) -> Result<Option<StreamEvent>> {
        if data == "[DONE]" {
            return Ok(Some(StreamEvent::Done));
        }

        let json: serde_json::Value = serde_json::from_str(data)
            .with_context(|| format!("Failed to parse SSE data: {}", data))?;

        // Check for finish_reason indicating tool calls are complete
        if let Some(finish_reason) = json["choices"][0]["finish_reason"].as_str()
            && finish_reason == "tool_calls"
        {
            // Clear all tracked tool calls and emit ToolUseDone
            self.tool_calls_by_index.clear();
            return Ok(Some(StreamEvent::ToolUseDone));
        }

        // Check for text content in delta
        if let Some(content) = json["choices"][0]["delta"]["content"].as_str()
            && !content.is_empty()
        {
            return Ok(Some(StreamEvent::TextDelta(content.to_string())));
        }

        // Check for tool_calls in delta (OpenAI-compatible format)
        if let Some(tool_calls) = json["choices"][0]["delta"]["tool_calls"].as_array() {
            for tool_call in tool_calls {
                if let Some(index) = tool_call["index"].as_u64()
                    && let Some(function) = tool_call["function"].as_object()
                {
                    // Check if this is a new tool call with a name
                    if let Some(name) = function.get("name").and_then(|v| v.as_str()) {
                        // Generate a stable ID for this tool call
                        let id = format!("tool_{}", index);
                        self.tool_calls_by_index.insert(index, id.clone());
                        return Ok(Some(StreamEvent::ToolUseStart {
                            id,
                            name: name.to_string(),
                        }));
                    }

                    // Check if this has arguments (delta)
                    if let Some(arguments) = function.get("arguments").and_then(|v| v.as_str())
                        && !arguments.is_empty()
                    {
                        return Ok(Some(StreamEvent::ToolUseDelta(arguments.to_string())));
                    }
                }
            }
        }

        Ok(None)
    }
}

#[derive(Debug)]
pub struct ZaiBackend {
    client: Client,
    api_key: String,
}

impl ZaiBackend {
    pub fn new(api_key: String) -> Result<Self> {
        if api_key.is_empty() {
            anyhow::bail!("API key cannot be empty");
        }
        Ok(Self {
            client: Client::new(),
            api_key,
        })
    }

    fn build_request_body(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> serde_json::Value {
        let messages_json: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let content: Vec<serde_json::Value> = m
                    .content
                    .iter()
                    .map(|block| match block {
                        ContentBlock::Text(text) => {
                            serde_json::json!({"type": "text", "text": text})
                        }
                        other => serde_json::to_value(other).unwrap_or(serde_json::Value::Null),
                    })
                    .collect();
                serde_json::json!({"role": m.role, "content": content})
            })
            .collect();

        let mut body = serde_json::json!({
            "model": config.model,
            "messages": messages_json,
            "stream": true,
            "max_tokens": config.max_tokens,
        });

        // Add tools in OpenAI-compatible format if present
        if !config.tools.is_empty() {
            let tools_json: Vec<serde_json::Value> = config
                .tools
                .iter()
                .map(|tool| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools_json);
        }

        body
    }
}

#[async_trait]
impl LlmBackend for ZaiBackend {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        let body = self.build_request_body(messages, config);

        let response = self
            .client
            .post(ENDPOINT)
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to z.ai")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<failed to read response body>"));
            anyhow::bail!("z.ai returned {}: {}", status, body);
        }

        let byte_stream = response.bytes_stream();
        let mut parser = ZaiSseParser::new();
        let event_stream = create_sse_event_stream(byte_stream, move |data| parser.parse(data));
        Ok(event_stream)
    }
}

pub fn parse_sse_data(data: &str) -> Result<Option<StreamEvent>> {
    if data == "[DONE]" {
        return Ok(Some(StreamEvent::Done));
    }

    let json: serde_json::Value = serde_json::from_str(data)
        .with_context(|| format!("Failed to parse SSE data: {}", data))?;

    let content = json["choices"][0]["delta"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    if content.is_empty() {
        Ok(None)
    } else {
        Ok(Some(StreamEvent::TextDelta(content)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Role;

    #[test]
    fn new_rejects_empty_api_key() {
        let result = ZaiBackend::new("".to_string());
        assert!(result.is_err(), "Should reject empty API key");
        assert!(result.unwrap_err().to_string().contains("API key"));
    }

    #[test]
    fn new_accepts_valid_api_key() {
        let result = ZaiBackend::new("test-key".to_string());
        assert!(result.is_ok(), "Should accept valid API key");
        let backend = result.unwrap();
        assert_eq!(backend.api_key, "test-key");
    }

    #[test]
    fn build_request_body_includes_model_and_stream() {
        let backend = ZaiBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "glm-5-turbo".to_string(),
            max_tokens: 4096,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![crate::types::ContentBlock::Text("Hello".to_string())],
        }];

        let body = backend.build_request_body(&messages, &config);

        assert_eq!(body["model"], "glm-5-turbo");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 4096);
    }

    #[test]
    fn build_request_body_includes_messages_with_correct_roles() {
        let backend = ZaiBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "glm-5-turbo".to_string(),
            max_tokens: 4096,
            tools: vec![],
        };
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![crate::types::ContentBlock::Text("Hello".to_string())],
            },
            Message {
                role: Role::Assistant,
                content: vec![crate::types::ContentBlock::Text("Hi there!".to_string())],
            },
        ];

        let body = backend.build_request_body(&messages, &config);

        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(
            body["messages"][0]["content"],
            serde_json::json!([{"type": "text", "text": "Hello"}])
        );
        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(
            body["messages"][1]["content"],
            serde_json::json!([{"type": "text", "text": "Hi there!"}])
        );
    }

    #[test]
    fn build_request_body_formats_text_blocks_as_typed_objects() {
        let backend = ZaiBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "glm-5-turbo".to_string(),
            max_tokens: 4096,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![crate::types::ContentBlock::Text(
                "read the test.txt file".to_string(),
            )],
        }];

        let body = backend.build_request_body(&messages, &config);

        let content0 = &body["messages"][0]["content"][0];
        assert_eq!(content0["type"], "text", "text block must have type='text'");
        assert_eq!(content0["text"], "read the test.txt file");
    }

    #[test]
    fn parse_sse_data_extracts_text_delta() {
        let mut parser = ZaiSseParser::new();
        let data = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
        let result = parser.parse(data);
        assert!(result.is_ok(), "Should successfully parse valid SSE data");
        let event = result.unwrap();
        assert!(event.is_some(), "Should return Some(event)");
        assert!(
            matches!(event, Some(StreamEvent::TextDelta(_))),
            "Should be TextDelta"
        );
        if let Some(StreamEvent::TextDelta(text)) = event {
            assert_eq!(text, "Hello");
        }
    }

    #[test]
    fn parse_sse_data_returns_done_for_done_event() {
        let mut parser = ZaiSseParser::new();
        let data = "[DONE]";
        let result = parser.parse(data);
        assert!(result.is_ok(), "Should successfully parse [DONE]");
        let event = result.unwrap();
        assert!(event.is_some(), "Should return Some(event) for [DONE]");
        assert!(matches!(event, Some(StreamEvent::Done)), "Should be Done");
    }

    #[test]
    fn parse_sse_data_returns_error_for_invalid_json() {
        let mut parser = ZaiSseParser::new();
        let data = "not valid json";
        let result = parser.parse(data);
        assert!(result.is_err(), "Should return error for invalid JSON");
    }

    #[test]
    fn parse_sse_data_returns_none_for_empty_delta() {
        let mut parser = ZaiSseParser::new();
        let data = r#"{"choices":[{"delta":{}}]}"#;
        let result = parser.parse(data);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn build_request_body_includes_tools_in_openai_compatible_format() {
        let backend = ZaiBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "glm-5-turbo".to_string(),
            max_tokens: 4096,
            tools: vec![
                crate::types::ToolDefinition {
                    name: "bash".to_string(),
                    description: "Execute bash commands".to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute"
                            }
                        },
                        "required": ["command"]
                    }),
                },
                crate::types::ToolDefinition {
                    name: "read_file".to_string(),
                    description: "Read a file".to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The file path"
                            }
                        },
                        "required": ["path"]
                    }),
                },
            ],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![crate::types::ContentBlock::Text("Hello".to_string())],
        }];

        let body = backend.build_request_body(&messages, &config);

        // Verify tools array exists
        assert!(
            body.get("tools").is_some(),
            "body should contain 'tools' field"
        );

        let tools = body["tools"].as_array().expect("tools should be an array");
        assert_eq!(tools.len(), 2, "should have 2 tools");

        // Verify first tool structure (OpenAI-compatible format)
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "bash");
        assert_eq!(tools[0]["function"]["description"], "Execute bash commands");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
        assert_eq!(
            tools[0]["function"]["parameters"]["properties"]["command"]["type"],
            "string"
        );
        assert_eq!(
            tools[0]["function"]["parameters"]["required"],
            serde_json::json!(["command"])
        );

        // Verify second tool structure
        assert_eq!(tools[1]["type"], "function");
        assert_eq!(tools[1]["function"]["name"], "read_file");
        assert_eq!(tools[1]["function"]["description"], "Read a file");
    }

    #[test]
    fn build_request_body_with_empty_tools_does_not_include_tools_field() {
        let backend = ZaiBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "glm-5-turbo".to_string(),
            max_tokens: 4096,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![crate::types::ContentBlock::Text("Hello".to_string())],
        }];

        let body = backend.build_request_body(&messages, &config);

        // Tools field should either not exist or be null/empty when no tools provided
        assert!(
            body.get("tools").is_none()
                || body["tools"].as_array().map_or(true, |arr| arr.is_empty()),
            "body should not include tools field or it should be empty when no tools provided"
        );
    }

    #[test]
    fn parse_sse_data_extracts_tool_use_start() {
        // OpenAI-compatible SSE with tool call start
        let mut parser = ZaiSseParser::new();
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"bash","arguments":""}}]}}]}"#;
        let result = parser.parse(data);
        assert!(
            result.is_ok(),
            "Should successfully parse tool use start SSE data"
        );
        let event = result.unwrap();
        assert!(event.is_some(), "Should return Some(event)");
        assert!(
            matches!(event, Some(StreamEvent::ToolUseStart { .. })),
            "Should be ToolUseStart, got: {:?}",
            event
        );
        if let Some(StreamEvent::ToolUseStart { id, name }) = event {
            assert_eq!(name, "bash");
            // For OpenAI-compatible format, we generate our own ID
            assert!(!id.is_empty(), "ID should not be empty");
        }
    }

    #[test]
    fn parse_sse_data_extracts_tool_use_delta() {
        // OpenAI-compatible SSE with tool call arguments delta
        let mut parser = ZaiSseParser::new();
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"comm"}}]}}]}"#;
        let result = parser.parse(data);
        assert!(
            result.is_ok(),
            "Should successfully parse tool use delta SSE data"
        );
        let event = result.unwrap();
        assert!(event.is_some(), "Should return Some(event)");
        assert!(
            matches!(event, Some(StreamEvent::ToolUseDelta(_))),
            "Should be ToolUseDelta, got: {:?}",
            event
        );
        if let Some(StreamEvent::ToolUseDelta(delta)) = event {
            assert_eq!(delta, "{\"comm");
        }
    }

    #[test]
    fn parse_sse_data_handles_tool_use_done() {
        // OpenAI-compatible SSE with finish_reason indicates tool call is done
        let mut parser = ZaiSseParser::new();
        let data = r#"{"choices":[{"finish_reason":"tool_calls"}]}"#;
        let result = parser.parse(data);
        assert!(
            result.is_ok(),
            "Should successfully parse tool use done SSE data"
        );
        let event = result.unwrap();
        assert!(event.is_some(), "Should return Some(event)");
        assert!(
            matches!(event, Some(StreamEvent::ToolUseDone)),
            "Should be ToolUseDone, got: {:?}",
            event
        );
    }

    #[test]
    fn parse_sse_data_handles_multiple_tool_calls_with_index() {
        // Verify we can track multiple concurrent tool calls by index
        let mut parser = ZaiSseParser::new();
        let data1 = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"bash","arguments":""}}]}}]}"#;
        let data2 = r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"name":"read_file","arguments":""}}]}}]}"#;

        let result1 = parser.parse(data1);
        let result2 = parser.parse(data2);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let event1 = result1.unwrap();
        let event2 = result2.unwrap();

        assert!(event1.is_some());
        assert!(event2.is_some());

        // Both should be ToolUseStart events with different names
        if let Some(StreamEvent::ToolUseStart { name: name1, .. }) = event1 {
            assert_eq!(name1, "bash");
        } else {
            panic!("First event should be ToolUseStart");
        }

        if let Some(StreamEvent::ToolUseStart { name: name2, .. }) = event2 {
            assert_eq!(name2, "read_file");
        } else {
            panic!("Second event should be ToolUseStart");
        }
    }

    #[test]
    fn parse_sse_data_returns_none_for_empty_tool_calls() {
        // Some SSE chunks may have empty tool_calls array
        let mut parser = ZaiSseParser::new();
        let data = r#"{"choices":[{"delta":{"tool_calls":[]}}]}"#;
        let result = parser.parse(data);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }
}
