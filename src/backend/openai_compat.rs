use std::collections::HashMap;

use anyhow::{Context, Result};
use reqwest::Client;

use super::sse::create_sse_event_stream;
use crate::types::{BoxStream, ContentBlock, Message, RequestConfig, Role, StreamEvent};

#[derive(Default)]
pub struct OpenAiSseParser {
    tool_calls_by_index: HashMap<u64, String>,
    pub event_buffer: Vec<StreamEvent>,
}

impl OpenAiSseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parse(&mut self, data: &str) -> Result<Option<StreamEvent>> {
        if !data.is_empty() {
            self.fill_buffer(data)?;
        }

        if !self.event_buffer.is_empty() {
            return Ok(Some(self.event_buffer.remove(0)));
        }

        Ok(None)
    }

    pub fn fill_buffer(&mut self, data: &str) -> Result<()> {
        if data == "[DONE]" {
            self.event_buffer.push(StreamEvent::Done);
            return Ok(());
        }

        let json: serde_json::Value = serde_json::from_str(data)
            .with_context(|| format!("Failed to parse SSE data: {}", data))?;

        if let Some(finish_reason) = json["choices"][0]["finish_reason"].as_str() {
            let input_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
            let output_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
            match finish_reason {
                "tool_calls" => {
                    self.tool_calls_by_index.clear();
                    self.event_buffer.push(StreamEvent::ToolUseDone);
                    if input_tokens > 0 || output_tokens > 0 {
                        self.event_buffer.push(StreamEvent::Usage {
                            input_tokens,
                            output_tokens,
                            stop_reason: finish_reason.to_string(),
                        });
                    }
                    return Ok(());
                }
                "length" => {
                    return Err(anyhow::anyhow!(
                        "Response truncated: max_tokens limit reached (input_tokens={}, output_tokens={}). Increase max_tokens in your config.",
                        input_tokens,
                        output_tokens,
                    ));
                }
                _ => {
                    if input_tokens > 0 || output_tokens > 0 {
                        self.event_buffer.push(StreamEvent::Usage {
                            input_tokens,
                            output_tokens,
                            stop_reason: finish_reason.to_string(),
                        });
                    }
                }
            }
        }

        if let Some(content) = json["choices"][0]["delta"]["content"].as_str()
            && !content.is_empty()
        {
            self.event_buffer
                .push(StreamEvent::TextDelta(content.to_string()));
            return Ok(());
        }

        if let Some(tool_calls) = json["choices"][0]["delta"]["tool_calls"].as_array() {
            for tool_call in tool_calls {
                if let Some(index) = tool_call["index"].as_u64()
                    && let Some(function) = tool_call["function"].as_object()
                {
                    if let Some(name) = function.get("name").and_then(|v| v.as_str()) {
                        let id = format!("tool_{}", index);
                        self.tool_calls_by_index.insert(index, id.clone());
                        self.event_buffer.push(StreamEvent::ToolUseStart {
                            id,
                            name: name.to_string(),
                        });
                    }

                    if let Some(arguments) = function.get("arguments").and_then(|v| v.as_str())
                        && !arguments.is_empty()
                    {
                        self.event_buffer
                            .push(StreamEvent::ToolUseDelta(arguments.to_string()));
                    }
                }
            }
        }

        Ok(())
    }
}

pub fn build_request_body(
    messages: &[Message],
    config: &RequestConfig,
) -> Result<serde_json::Value> {
    let mut messages_json: Vec<serde_json::Value> = Vec::new();

    for m in messages {
        match m.role {
            Role::User => {
                let has_tool_results = m
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolResult { .. }));

                if has_tool_results {
                    for block in &m.content {
                        match block {
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                ..
                            } => {
                                messages_json.push(serde_json::json!({
                                    "role": "tool",
                                    "tool_call_id": tool_use_id,
                                    "content": content
                                }));
                            }
                            ContentBlock::Text(text) => {
                                messages_json.push(serde_json::json!({
                                    "role": "user",
                                    "content": [{"type": "text", "text": text}]
                                }));
                            }
                            other => {
                                return Err(anyhow::anyhow!(
                                    "Unexpected content block in user message: {}",
                                    serde_json::to_string(other)
                                        .unwrap_or_else(|_| "<unserializable>".to_string())
                                ));
                            }
                        }
                    }
                } else {
                    let mut content_json: Vec<serde_json::Value> = Vec::new();
                    for block in &m.content {
                        match block {
                            ContentBlock::Text(text) => {
                                content_json
                                    .push(serde_json::json!({"type": "text", "text": text}));
                            }
                            other => {
                                return Err(anyhow::anyhow!(
                                    "Unsupported content block type in user message: {}",
                                    serde_json::to_string(other)
                                        .unwrap_or_else(|_| "<unserializable>".to_string())
                                ));
                            }
                        }
                    }
                    messages_json
                        .push(serde_json::json!({"role": "user", "content": content_json}));
                }
            }
            Role::Assistant => {
                let has_tool_use = m
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolUse { .. }));

                if has_tool_use {
                    let mut text_parts: Vec<String> = Vec::new();
                    let mut tool_calls: Vec<serde_json::Value> = Vec::new();

                    for block in &m.content {
                        match block {
                            ContentBlock::Text(text) => text_parts.push(text.clone()),
                            ContentBlock::ToolUse { id, name, input } => {
                                let arguments = serde_json::to_string(input)
                                    .context("Failed to serialize tool input")?;
                                tool_calls.push(serde_json::json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": arguments
                                    }
                                }));
                            }
                            other => {
                                return Err(anyhow::anyhow!(
                                    "Unexpected content block in assistant message: {}",
                                    serde_json::to_string(other)
                                        .unwrap_or_else(|_| "<unserializable>".to_string())
                                ));
                            }
                        }
                    }

                    let mut msg = serde_json::json!({"role": "assistant"});
                    if !text_parts.is_empty() {
                        msg["content"] = serde_json::json!(text_parts.join(""));
                    }
                    msg["tool_calls"] = serde_json::json!(tool_calls);
                    messages_json.push(msg);
                } else {
                    let mut content_json: Vec<serde_json::Value> = Vec::new();
                    for block in &m.content {
                        match block {
                            ContentBlock::Text(text) => {
                                content_json
                                    .push(serde_json::json!({"type": "text", "text": text}));
                            }
                            other => {
                                return Err(anyhow::anyhow!(
                                    "Unsupported content block type in assistant message: {}",
                                    serde_json::to_string(other)
                                        .unwrap_or_else(|_| "<unserializable>".to_string())
                                ));
                            }
                        }
                    }
                    messages_json
                        .push(serde_json::json!({"role": "assistant", "content": content_json}));
                }
            }
        }
    }

    let mut body = serde_json::json!({
        "model": config.model,
        "messages": messages_json,
        "stream": true,
        "max_tokens": config.max_tokens,
    });

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

    Ok(body)
}

pub async fn send_request(
    client: &Client,
    api_key: &str,
    endpoint: &str,
    messages: &[Message],
    config: &RequestConfig,
    provider_name: &str,
) -> Result<BoxStream<Result<StreamEvent>>> {
    let body = build_request_body(messages, config)?;

    let response = client
        .post(endpoint)
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .with_context(|| format!("Failed to send request to {}", provider_name))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| String::from("<failed to read response body>"));
        anyhow::bail!("{} returned {}: {}", provider_name, status, body);
    }

    let byte_stream = response.bytes_stream();
    let mut parser = OpenAiSseParser::new();
    let event_stream = create_sse_event_stream(byte_stream, move |data| {
        parser.fill_buffer(data)?;
        let events = std::mem::take(&mut parser.event_buffer);
        Ok(events)
    });
    Ok(event_stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Role;

    #[test]
    fn parse_sse_data_extracts_text_delta() {
        let mut parser = OpenAiSseParser::new();
        let data = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
        let result = parser.parse(data);
        assert!(result.is_ok());
        let event = result.unwrap();
        assert!(event.is_some());
        assert!(matches!(event, Some(StreamEvent::TextDelta(_))));
        if let Some(StreamEvent::TextDelta(text)) = event {
            assert_eq!(text, "Hello");
        }
    }

    #[test]
    fn parse_sse_data_returns_done_for_done_event() {
        let mut parser = OpenAiSseParser::new();
        let result = parser.parse("[DONE]");
        assert!(result.is_ok());
        let event = result.unwrap();
        assert!(event.is_some());
        assert!(matches!(event, Some(StreamEvent::Done)));
    }

    #[test]
    fn parse_sse_data_returns_error_for_invalid_json() {
        let mut parser = OpenAiSseParser::new();
        let result = parser.parse("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn parse_sse_data_returns_none_for_empty_delta() {
        let mut parser = OpenAiSseParser::new();
        let data = r#"{"choices":[{"delta":{}}]}"#;
        let result = parser.parse(data);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn parse_sse_data_extracts_tool_use_start() {
        let mut parser = OpenAiSseParser::new();
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"bash","arguments":""}}]}}]}"#;
        let result = parser.parse(data);
        assert!(result.is_ok());
        let event = result.unwrap();
        assert!(event.is_some());
        assert!(matches!(event, Some(StreamEvent::ToolUseStart { .. })));
        if let Some(StreamEvent::ToolUseStart { id, name }) = event {
            assert_eq!(name, "bash");
            assert!(!id.is_empty());
        }
    }

    #[test]
    fn parse_sse_data_extracts_tool_use_delta() {
        let mut parser = OpenAiSseParser::new();
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"comm"}}]}}]}"#;
        let result = parser.parse(data);
        assert!(result.is_ok());
        let event = result.unwrap();
        assert!(event.is_some());
        assert!(matches!(event, Some(StreamEvent::ToolUseDelta(_))));
        if let Some(StreamEvent::ToolUseDelta(delta)) = event {
            assert_eq!(delta, "{\"comm");
        }
    }

    #[test]
    fn parse_sse_data_handles_tool_use_done() {
        let mut parser = OpenAiSseParser::new();
        let data = r#"{"choices":[{"finish_reason":"tool_calls"}]}"#;
        let result = parser.parse(data);
        assert!(result.is_ok());
        let event = result.unwrap();
        assert!(event.is_some());
        assert!(matches!(event, Some(StreamEvent::ToolUseDone)));
    }

    #[test]
    fn parse_sse_data_handles_multiple_tool_calls_in_single_chunk() {
        let mut parser = OpenAiSseParser::new();
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"bash","arguments":""}},{"index":1,"function":{"name":"read_file","arguments":""}}]}}]}"#;

        let result1 = parser.parse(data);
        assert!(result1.is_ok());
        if let Some(StreamEvent::ToolUseStart { name, .. }) = result1.unwrap() {
            assert_eq!(name, "bash");
        } else {
            panic!("expected ToolUseStart for bash");
        }

        let result2 = parser.parse("");
        assert!(result2.is_ok());
        if let Some(StreamEvent::ToolUseStart { name, .. }) = result2.unwrap() {
            assert_eq!(name, "read_file");
        } else {
            panic!("expected ToolUseStart for read_file");
        }

        let result3 = parser.parse("");
        assert!(result3.is_ok());
        assert!(result3.unwrap().is_none());
    }

    #[test]
    fn build_request_body_includes_model_stream_and_max_tokens() {
        let config = RequestConfig {
            model: "llama3.2".to_string(),
            max_tokens: 4096,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
        }];

        let body = build_request_body(&messages, &config).unwrap();

        assert_eq!(body["model"], "llama3.2");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 4096);
    }

    #[test]
    fn build_request_body_formats_user_message_as_typed_content() {
        let config = RequestConfig {
            model: "test-model".to_string(),
            max_tokens: 1024,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
        }];

        let body = build_request_body(&messages, &config).unwrap();

        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(
            body["messages"][0]["content"][0],
            serde_json::json!({"type": "text", "text": "Hello"})
        );
    }

    #[test]
    fn build_request_body_converts_tool_result_to_tool_role_message() {
        let config = RequestConfig {
            model: "test-model".to_string(),
            max_tokens: 1024,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tool_0".to_string(),
                content: "output".to_string(),
                is_error: false,
            }],
        }];

        let body = build_request_body(&messages, &config).unwrap();
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "tool_0");
        assert_eq!(msgs[0]["content"], "output");
    }

    #[test]
    fn build_request_body_converts_assistant_tool_use_to_tool_calls_format() {
        let config = RequestConfig {
            model: "test-model".to_string(),
            max_tokens: 1024,
            tools: vec![],
        };
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text("run ls".to_string())],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "tool_0".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "ls"}),
                }],
            },
        ];

        let body = build_request_body(&messages, &config).unwrap();
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["role"], "assistant");
        let tool_calls = msgs[1]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls[0]["function"]["name"], "bash");
        assert_eq!(tool_calls[0]["id"], "tool_0");
        let args: serde_json::Value =
            serde_json::from_str(tool_calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["command"], "ls");
    }

    #[test]
    fn build_request_body_converts_tool_results_to_tool_role_messages() {
        let config = RequestConfig {
            model: "test-model".to_string(),
            max_tokens: 1024,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tool_0".to_string(),
                content: "file1.txt\nfile2.txt".to_string(),
                is_error: false,
            }],
        }];

        let body = build_request_body(&messages, &config).unwrap();
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "tool_0");
        assert_eq!(msgs[0]["content"], "file1.txt\nfile2.txt");
    }

    #[test]
    fn build_request_body_includes_tools_in_openai_format() {
        let config = RequestConfig {
            model: "test-model".to_string(),
            max_tokens: 1024,
            tools: vec![crate::types::ToolDefinition {
                name: "bash".to_string(),
                description: "Execute bash commands".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }),
            }],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
        }];

        let body = build_request_body(&messages, &config).unwrap();
        let tools = body["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "bash");
        assert_eq!(tools[0]["function"]["description"], "Execute bash commands");
    }

    #[test]
    fn build_request_body_omits_tools_field_when_empty() {
        let config = RequestConfig {
            model: "test-model".to_string(),
            max_tokens: 1024,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
        }];

        let body = build_request_body(&messages, &config).unwrap();
        assert!(
            body.get("tools").is_none() || body["tools"].as_array().map_or(true, |a| a.is_empty()),
        );
    }

    #[test]
    fn build_request_body_full_tool_use_round_trip() {
        let config = RequestConfig {
            model: "test-model".to_string(),
            max_tokens: 1024,
            tools: vec![],
        };
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text("run ls".to_string())],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "tool_0".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "ls"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "tool_0".to_string(),
                    content: "file1.txt".to_string(),
                    is_error: false,
                }],
            },
        ];

        let body = build_request_body(&messages, &config).unwrap();
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert!(msgs[1]["tool_calls"].as_array().is_some());
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "tool_0");
        assert_eq!(msgs[2]["content"], "file1.txt");
    }
}
