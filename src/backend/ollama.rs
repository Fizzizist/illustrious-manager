use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;

use super::LlmBackend;
use super::sse::create_sse_event_stream;
use super::zai::ZaiSseParser;
use crate::types::{BoxStream, ContentBlock, Message, RequestConfig, Role, StreamEvent};

const ENDPOINT: &str = "https://api.ollama.com/v1/chat/completions";

#[derive(Debug)]
pub struct OllamaBackend {
    client: Client,
    api_key: String,
}

impl OllamaBackend {
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
                                _ => {}
                            }
                        }
                    } else {
                        let content: Vec<serde_json::Value> =
                            m.content
                                .iter()
                                .map(|block| match block {
                                    ContentBlock::Text(text) => {
                                        serde_json::json!({"type": "text", "text": text})
                                    }
                                    other => serde_json::to_value(other)
                                        .unwrap_or(serde_json::Value::Null),
                                })
                                .collect();
                        messages_json.push(serde_json::json!({"role": "user", "content": content}));
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
                                        .unwrap_or_else(|_| "{}".to_string());
                                    tool_calls.push(serde_json::json!({
                                        "id": id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": arguments
                                        }
                                    }));
                                }
                                _ => {}
                            }
                        }

                        let mut msg = serde_json::json!({"role": "assistant"});
                        if !text_parts.is_empty() {
                            msg["content"] = serde_json::json!(text_parts.join(""));
                        }
                        msg["tool_calls"] = serde_json::json!(tool_calls);
                        messages_json.push(msg);
                    } else {
                        let content: Vec<serde_json::Value> =
                            m.content
                                .iter()
                                .map(|block| match block {
                                    ContentBlock::Text(text) => {
                                        serde_json::json!({"type": "text", "text": text})
                                    }
                                    other => serde_json::to_value(other)
                                        .unwrap_or(serde_json::Value::Null),
                                })
                                .collect();
                        messages_json
                            .push(serde_json::json!({"role": "assistant", "content": content}));
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

        body
    }
}

#[async_trait]
impl LlmBackend for OllamaBackend {
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
            .context("Failed to send request to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<failed to read response body>"));
            anyhow::bail!("Ollama returned {}: {}", status, body);
        }

        let byte_stream = response.bytes_stream();
        let mut parser = ZaiSseParser::new();
        let event_stream = create_sse_event_stream(byte_stream, move |data| {
            parser.fill_buffer(data)?;
            let events = std::mem::take(&mut parser.event_buffer);
            Ok(events)
        });
        Ok(event_stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Role;

    #[test]
    fn new_rejects_empty_api_key() {
        let result = OllamaBackend::new("".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key"));
    }

    #[test]
    fn new_accepts_valid_api_key() {
        let result = OllamaBackend::new("test-key".to_string());
        assert!(result.is_ok());
        let backend = result.unwrap();
        assert_eq!(backend.api_key, "test-key");
    }

    #[test]
    fn build_request_body_includes_model_stream_and_max_tokens() {
        let backend = OllamaBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "llama3.2".to_string(),
            max_tokens: 4096,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
        }];

        let body = backend.build_request_body(&messages, &config);

        assert_eq!(body["model"], "llama3.2");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 4096);
    }

    #[test]
    fn build_request_body_formats_user_message_with_typed_content() {
        let backend = OllamaBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "llama3.2".to_string(),
            max_tokens: 4096,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
        }];

        let body = backend.build_request_body(&messages, &config);

        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(
            body["messages"][0]["content"][0],
            serde_json::json!({"type": "text", "text": "Hello"})
        );
    }

    #[test]
    fn build_request_body_includes_tools_in_openai_format() {
        let backend = OllamaBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "llama3.2".to_string(),
            max_tokens: 4096,
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

        let body = backend.build_request_body(&messages, &config);

        let tools = body["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "bash");
    }

    #[test]
    fn build_request_body_with_no_tools_omits_tools_field() {
        let backend = OllamaBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "llama3.2".to_string(),
            max_tokens: 4096,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
        }];

        let body = backend.build_request_body(&messages, &config);

        assert!(
            body.get("tools").is_none() || body["tools"].as_array().map_or(true, |a| a.is_empty()),
        );
    }

    #[test]
    fn build_request_body_converts_tool_result_to_tool_role_message() {
        let backend = OllamaBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "llama3.2".to_string(),
            max_tokens: 4096,
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

        let body = backend.build_request_body(&messages, &config);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "tool_0");
        assert_eq!(msgs[0]["content"], "output");
    }

    #[test]
    fn build_request_body_converts_assistant_tool_use_to_tool_calls_format() {
        let backend = OllamaBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "llama3.2".to_string(),
            max_tokens: 4096,
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

        let body = backend.build_request_body(&messages, &config);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["role"], "assistant");
        let tool_calls = msgs[1]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls[0]["function"]["name"], "bash");
        assert_eq!(tool_calls[0]["id"], "tool_0");
    }
}
