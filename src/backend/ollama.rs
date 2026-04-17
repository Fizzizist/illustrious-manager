use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;

use super::LlmBackend;
use super::openai_compat::OpenAiSseParser;
use super::sse::create_sse_event_stream;
use crate::types::{BoxStream, ContentBlock, Message, RequestConfig, Role, StreamEvent};

const ENDPOINT: &str = "https://ollama.com/api/chat";

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
}

#[async_trait]
impl LlmBackend for OllamaBackend {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        let body = build_request_body(messages, config)?;

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
        let mut parser = OpenAiSseParser::new();
        let event_stream = create_sse_event_stream(byte_stream, move |data| {
            parser.fill_buffer(data)?;
            let events = std::mem::take(&mut parser.event_buffer);
            Ok(events)
        });
        Ok(event_stream)
    }
}

fn build_request_body(messages: &[Message], config: &RequestConfig) -> Result<serde_json::Value> {
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
                                    "content": text
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
                    let text = m
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text(t) => Some(t.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    messages_json.push(serde_json::json!({
                        "role": "user",
                        "content": text
                    }));
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
                    let text = m
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text(t) => Some(t.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    messages_json.push(serde_json::json!({
                        "role": "assistant",
                        "content": text
                    }));
                }
            }
        }
    }

    let mut body = serde_json::json!({
        "model": config.model,
        "messages": messages_json,
        "stream": true,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolDefinition;

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
    fn build_request_body_sends_user_content_as_string() {
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

        assert_eq!(body["messages"][0]["role"], "user");
        assert!(
            body["messages"][0]["content"].is_string(),
            "content should be a plain string, got: {}",
            body["messages"][0]["content"]
        );
        assert_eq!(body["messages"][0]["content"], "Hello");
    }

    #[test]
    fn build_request_body_sends_assistant_content_as_string() {
        let config = RequestConfig {
            model: "llama3.2".to_string(),
            max_tokens: 4096,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text("Hi there".to_string())],
        }];

        let body = build_request_body(&messages, &config).unwrap();

        assert_eq!(body["messages"][0]["role"], "assistant");
        assert!(
            body["messages"][0]["content"].is_string(),
            "content should be a plain string, got: {}",
            body["messages"][0]["content"]
        );
        assert_eq!(body["messages"][0]["content"], "Hi there");
    }

    #[test]
    fn build_request_body_tool_result_uses_tool_role() {
        let config = RequestConfig {
            model: "llama3.2".to_string(),
            max_tokens: 4096,
            tools: vec![],
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tool_0".to_string(),
                content: "file1.txt".to_string(),
                is_error: false,
            }],
        }];

        let body = build_request_body(&messages, &config).unwrap();
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "tool_0");
        assert!(msgs[0]["content"].is_string());
        assert_eq!(msgs[0]["content"], "file1.txt");
    }

    #[test]
    fn build_request_body_includes_tools() {
        let config = RequestConfig {
            model: "llama3.2".to_string(),
            max_tokens: 4096,
            tools: vec![ToolDefinition {
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
            content: vec![ContentBlock::Text("run ls".to_string())],
        }];

        let body = build_request_body(&messages, &config).unwrap();
        let tools = body["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "bash");
    }

    #[test]
    fn build_request_body_omits_tools_when_empty() {
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
        assert!(
            body.get("tools").is_none() || body["tools"].as_array().map_or(true, |a| a.is_empty()),
        );
    }

    #[test]
    fn build_request_body_assistant_with_tool_use() {
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

        let body = build_request_body(&messages, &config).unwrap();
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["role"], "assistant");
        let tool_calls = msgs[1]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls[0]["function"]["name"], "bash");
        assert_eq!(tool_calls[0]["id"], "tool_0");
    }

    #[test]
    fn build_request_body_does_not_include_max_tokens() {
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
        assert!(
            body.get("max_tokens").is_none(),
            "Ollama does not use max_tokens in its chat API"
        );
    }
}
