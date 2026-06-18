use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;

use super::LlmBackend;
use super::error::BackendError;
use super::ndjson::create_ndjson_event_stream;
use crate::config::OllamaConfig;
use crate::types::{BoxStream, ContentBlock, Message, RequestConfig, Role, StreamEvent};

const DEFAULT_CLOUD_ENDPOINT: &str = "https://ollama.com/api/chat";

#[derive(Default)]
pub struct OllamaParser;

impl OllamaParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse_chunk(&mut self, data: &str) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();

        let json: serde_json::Value = serde_json::from_str(data)
            .with_context(|| format!("Failed to parse NDJSON line: {}", data))?;

        if let Some(error_msg) = json.get("error").and_then(|v| v.as_str()) {
            return Err(BackendError::Other(format!("Ollama error: {}", error_msg)).into());
        }

        if let Some(content) = json["message"]["content"].as_str()
            && !content.is_empty()
        {
            events.push(StreamEvent::TextDelta(content.to_string()));
        }

        if let Some(thinking) = json["message"]["thinking"].as_str()
            && !thinking.is_empty()
        {
            events.push(StreamEvent::ThinkingDelta(thinking.to_string()));
        }

        if let Some(tool_calls) = json["message"]["tool_calls"].as_array() {
            for (idx, tool_call) in tool_calls.iter().enumerate() {
                let function = &tool_call["function"];
                let name = match function["name"].as_str() {
                    Some(n) if !n.is_empty() => n.to_string(),
                    _ => {
                        return Err(BackendError::Other(format!(
                            "Malformed tool call at index {}: missing or empty function name",
                            idx
                        ))
                        .into());
                    }
                };
                let id = match tool_call
                    .get("id")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                {
                    Some(s) => s.to_string(),
                    None => format!("tool_{idx}"),
                };

                let arguments = if function["arguments"].is_object() {
                    serde_json::to_string(&function["arguments"])
                        .unwrap_or_else(|_| "{}".to_string())
                } else if let Some(args_str) = function["arguments"].as_str() {
                    args_str.to_string()
                } else {
                    "{}".to_string()
                };

                events.push(StreamEvent::ToolUseStart { id, name });
                events.push(StreamEvent::ToolUseDelta(arguments));
                events.push(StreamEvent::ToolUseDone);
            }
        }

        if json["done"].as_bool() == Some(true) {
            let input_tokens = json["prompt_eval_count"].as_u64().unwrap_or(0) as u32;
            let output_tokens = json["eval_count"].as_u64().unwrap_or(0) as u32;
            let stop_reason = json["done_reason"].as_str().unwrap_or("stop").to_string();

            if input_tokens > 0 || output_tokens > 0 {
                events.push(StreamEvent::Usage {
                    input_tokens,
                    output_tokens,
                    stop_reason: stop_reason.clone(),
                });
            }
            events.push(StreamEvent::Done);
        }

        Ok(events)
    }
}

#[derive(Debug)]
pub struct OllamaBackend {
    client: Client,
    api_key: String,
    endpoint: String,
}

impl OllamaBackend {
    pub fn new(config: &OllamaConfig) -> Result<Self> {
        if config.base_url == DEFAULT_CLOUD_ENDPOINT && config.api_key.is_empty() {
            return Err(BackendError::Other(String::from(
                "API key is required for Ollama Cloud. Set it in your config, or set base_url to your self-hosted endpoint."
            ))
            .into());
        }
        Ok(Self {
            client: Client::new(),
            api_key: config.api_key.clone(),
            endpoint: config.base_url.clone(),
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
                                        "content": text
                                    }));
                                }
                                _ => {}
                            }
                        }
                    } else {
                        let content: String = m
                            .content
                            .iter()
                            .filter_map(|block| match block {
                                ContentBlock::Text(text) => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("");
                        messages_json.push(serde_json::json!({
                            "role": "user",
                            "content": content
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
                                    tool_calls.push(serde_json::json!({
                                        "id": id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": input
                                        }
                                    }));
                                }
                                ContentBlock::Thinking { .. }
                                | ContentBlock::RedactedThinking { .. } => {}
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
                        let content: String = m
                            .content
                            .iter()
                            .filter_map(|block| match block {
                                ContentBlock::Text(text) => Some(text.as_str()),
                                ContentBlock::Thinking { .. }
                                | ContentBlock::RedactedThinking { .. } => None,
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("");
                        messages_json.push(serde_json::json!({
                            "role": "assistant",
                            "content": content
                        }));
                    }
                }
            }
        }

        let mut body = serde_json::json!({
            "model": config.model,
            "messages": messages_json,
            "stream": true,
            "options": {
                "num_predict": config.max_tokens
            }
        });

        if let Some(ref thinking_config) = config.thinking
            && thinking_config.enabled
        {
            body["options"]["thinking"] = serde_json::json!(true);
        }

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

        let mut request = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json");

        if !self.api_key.is_empty() {
            request = request.bearer_auth(&self.api_key);
        }

        let response = request
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
            return Err(super::error::BackendError::HttpStatus {
                code: status.as_u16(),
                body,
            }
            .into());
        }

        let byte_stream = response.bytes_stream();
        let mut parser = OllamaParser::new();
        let event_stream =
            create_ndjson_event_stream(byte_stream, move |line| parser.parse_chunk(line));
        Ok(event_stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_text_delta() {
        let mut parser = OllamaParser::new();
        let events = parser
            .parse_chunk(r#"{"message":{"content":"Hello"}}"#)
            .unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Hello"));
    }

    #[test]
    fn parser_empty_content_produces_no_text_delta() {
        let mut parser = OllamaParser::new();
        let events = parser.parse_chunk(r#"{"message":{"content":""}}"#).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn parser_single_tool_call() {
        let mut parser = OllamaParser::new();
        let chunk = r#"{"message":{"tool_calls":[{"id":"call_1","function":{"name":"bash","arguments":{"command":"ls"}}}]}}"#;
        let events = parser.parse_chunk(chunk).unwrap();
        assert_eq!(events.len(), 3);
        assert!(
            matches!(&events[0], StreamEvent::ToolUseStart { id, name } if id == "call_1" && name == "bash")
        );
        assert!(matches!(&events[1], StreamEvent::ToolUseDelta(args) if args.contains("ls")));
        assert!(matches!(&events[2], StreamEvent::ToolUseDone));
    }

    #[test]
    fn parser_multiple_tool_calls_in_one_chunk() {
        let mut parser = OllamaParser::new();
        let chunk = r#"{"message":{"tool_calls":[{"id":"c1","function":{"name":"bash","arguments":{"command":"ls"}}},{"id":"c2","function":{"name":"read_file","arguments":{"path":"foo.rs"}}}]}}"#;
        let events = parser.parse_chunk(chunk).unwrap();
        assert_eq!(events.len(), 6);
        assert!(matches!(&events[0], StreamEvent::ToolUseStart { name, .. } if name == "bash"));
        assert!(matches!(&events[1], StreamEvent::ToolUseDelta(_)));
        assert!(matches!(&events[2], StreamEvent::ToolUseDone));
        assert!(
            matches!(&events[3], StreamEvent::ToolUseStart { name, .. } if name == "read_file")
        );
        assert!(matches!(&events[4], StreamEvent::ToolUseDelta(_)));
        assert!(matches!(&events[5], StreamEvent::ToolUseDone));
    }

    #[test]
    fn parser_done_chunk_with_usage() {
        let mut parser = OllamaParser::new();
        let chunk = r#"{"done":true,"prompt_eval_count":10,"eval_count":20,"done_reason":"stop"}"#;
        let events = parser.parse_chunk(chunk).unwrap();
        assert_eq!(events.len(), 2);
        assert!(
            matches!(&events[0], StreamEvent::Usage { input_tokens: 10, output_tokens: 20, stop_reason } if stop_reason == "stop")
        );
        assert!(matches!(&events[1], StreamEvent::Done));
    }

    #[test]
    fn parser_done_chunk_without_usage_still_emits_done() {
        let mut parser = OllamaParser::new();
        let chunk = r#"{"done":true}"#;
        let events = parser.parse_chunk(chunk).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::Done));
    }

    #[test]
    fn parser_error_chunk() {
        let mut parser = OllamaParser::new();
        let chunk = r#"{"error":"model not found"}"#;
        let result = parser.parse_chunk(chunk);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("model not found"));
    }

    #[test]
    fn parser_done_reason_propagated_as_stop_reason() {
        let mut parser = OllamaParser::new();
        let chunk = r#"{"done":true,"prompt_eval_count":5,"eval_count":3,"done_reason":"length"}"#;
        let events = parser.parse_chunk(chunk).unwrap();
        assert!(
            matches!(&events[0], StreamEvent::Usage { stop_reason, .. } if stop_reason == "length")
        );
    }

    #[test]
    fn parser_invalid_json_returns_error() {
        let mut parser = OllamaParser::new();
        let result = parser.parse_chunk("not json");
        assert!(result.is_err());
    }

    #[test]
    fn request_body_has_model_stream_and_options() {
        let backend = OllamaBackend::new(&OllamaConfig {
            api_key: "test-key".to_string(),
            model: "gpt-oss:120b".to_string(),
            base_url: "https://ollama.com/api/chat".to_string(),
        })
        .unwrap();
        let config = RequestConfig {
            model: "gpt-oss:120b".to_string(),
            max_tokens: 4096,
            tools: vec![],
            thinking: None,
            cancel_token: None,
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
            created_at: 0.0,
        }];

        let body = backend.build_request_body(&messages, &config);

        assert_eq!(body["model"], "gpt-oss:120b");
        assert_eq!(body["stream"], true);
        assert_eq!(body["options"]["num_predict"], 4096);
    }

    #[test]
    fn request_body_includes_tools_array() {
        let backend = OllamaBackend::new(&OllamaConfig {
            api_key: "test-key".to_string(),
            model: "gpt-oss:120b".to_string(),
            base_url: "https://ollama.com/api/chat".to_string(),
        })
        .unwrap();
        let config = RequestConfig {
            model: "gpt-oss:120b".to_string(),
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
            thinking: None,
            cancel_token: None,
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
            created_at: 0.0,
        }];

        let body = backend.build_request_body(&messages, &config);

        let tools = body["tools"].as_array().expect("tools should be array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "bash");
        assert_eq!(tools[0]["function"]["description"], "Execute bash commands");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn request_body_no_tools_field_when_empty() {
        let backend = OllamaBackend::new(&OllamaConfig {
            api_key: "test-key".to_string(),
            model: "gpt-oss:120b".to_string(),
            base_url: "https://ollama.com/api/chat".to_string(),
        })
        .unwrap();
        let config = RequestConfig {
            model: "gpt-oss:120b".to_string(),
            max_tokens: 4096,
            tools: vec![],
            thinking: None,
            cancel_token: None,
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
            created_at: 0.0,
        }];

        let body = backend.build_request_body(&messages, &config);
        assert!(
            body.get("tools").is_none(),
            "should not include tools field when empty"
        );
    }

    #[test]
    fn ollama_request_body_does_not_include_parallel_tool_calls_field() {
        // Ollama's /api/chat does not support a parallel_tool_calls flag.
        // This test pins the body shape to ensure no such field is ever added.
        let backend = OllamaBackend::new(&OllamaConfig {
            api_key: "test-key".to_string(),
            model: "gpt-oss:120b".to_string(),
            base_url: "https://ollama.com/api/chat".to_string(),
        })
        .unwrap();
        let config = RequestConfig {
            model: "gpt-oss:120b".to_string(),
            max_tokens: 4096,
            tools: vec![crate::types::ToolDefinition {
                name: "bash".to_string(),
                description: "Run bash".to_string(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            }],
            thinking: None,
            cancel_token: None,
        };
        let body = backend.build_request_body(&[], &config);
        assert!(
            body.get("parallel_tool_calls").is_none(),
            "Ollama /api/chat has no parallel_tool_calls field; got: {body}"
        );
    }

    #[test]
    fn request_body_assistant_tool_calls_arguments_as_object() {
        let backend = OllamaBackend::new(&OllamaConfig {
            api_key: "test-key".to_string(),
            model: "gpt-oss:120b".to_string(),
            base_url: "https://ollama.com/api/chat".to_string(),
        })
        .unwrap();
        let config = RequestConfig {
            model: "gpt-oss:120b".to_string(),
            max_tokens: 4096,
            tools: vec![],
            thinking: None,
            cancel_token: None,
        };
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "tool_0".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"command": "ls"}),
            }],
            created_at: 0.0,
        }];

        let body = backend.build_request_body(&messages, &config);
        let msgs = body["messages"].as_array().expect("messages array");
        assert_eq!(msgs[0]["role"], "assistant");

        let tool_calls = msgs[0]["tool_calls"].as_array().expect("tool_calls array");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["function"]["name"], "bash");

        let args = &tool_calls[0]["function"]["arguments"];
        assert!(
            args.is_object(),
            "arguments must be a JSON object, not a string, got: {}",
            args
        );
        assert_eq!(args["command"], "ls");
    }

    #[test]
    fn request_body_tool_result_maps_to_role_tool() {
        let backend = OllamaBackend::new(&OllamaConfig {
            api_key: "test-key".to_string(),
            model: "gpt-oss:120b".to_string(),
            base_url: "https://ollama.com/api/chat".to_string(),
        })
        .unwrap();
        let config = RequestConfig {
            model: "gpt-oss:120b".to_string(),
            max_tokens: 4096,
            tools: vec![],
            thinking: None,
            cancel_token: None,
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tool_0".to_string(),
                content: "file1.txt\nfile2.txt".to_string(),
                is_error: false,
            }],
            created_at: 0.0,
        }];

        let body = backend.build_request_body(&messages, &config);
        let msgs = body["messages"].as_array().expect("messages array");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "tool_0");
        assert_eq!(msgs[0]["content"], "file1.txt\nfile2.txt");
    }

    #[test]
    fn request_body_full_tool_use_round_trip() {
        let backend = OllamaBackend::new(&OllamaConfig {
            api_key: "test-key".to_string(),
            model: "gpt-oss:120b".to_string(),
            base_url: "https://ollama.com/api/chat".to_string(),
        })
        .unwrap();
        let config = RequestConfig {
            model: "gpt-oss:120b".to_string(),
            max_tokens: 4096,
            tools: vec![],
            thinking: None,
            cancel_token: None,
        };
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text("run ls".to_string())],
                created_at: 0.0,
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "tool_0".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "ls"}),
                }],
                created_at: 0.0,
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "tool_0".to_string(),
                    content: "file1.txt".to_string(),
                    is_error: false,
                }],
                created_at: 0.0,
            },
        ];

        let body = backend.build_request_body(&messages, &config);
        let msgs = body["messages"].as_array().expect("messages array");
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert!(msgs[1]["tool_calls"].as_array().is_some());
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "tool_0");
        assert_eq!(msgs[2]["content"], "file1.txt");
    }

    #[test]
    fn new_rejects_empty_api_key_for_cloud_endpoint() {
        let result = OllamaBackend::new(&OllamaConfig {
            api_key: "".to_string(),
            model: "gpt-oss:120b".to_string(),
            base_url: "https://ollama.com/api/chat".to_string(),
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key"));
    }

    #[test]
    fn new_allows_empty_api_key_for_self_hosted() {
        let result = OllamaBackend::new(&OllamaConfig {
            api_key: "".to_string(),
            model: "gpt-oss:120b".to_string(),
            base_url: "http://localhost:11434/api/chat".to_string(),
        });
        assert!(result.is_ok(), "should accept empty key for self-hosted");
    }

    #[test]
    fn new_accepts_valid_config() {
        let backend = OllamaBackend::new(&OllamaConfig {
            api_key: "my-key".to_string(),
            model: "gpt-oss:120b".to_string(),
            base_url: "https://ollama.com/api/chat".to_string(),
        });
        assert!(backend.is_ok());
        let b = backend.unwrap();
        assert_eq!(b.endpoint, "https://ollama.com/api/chat");
    }

    #[test]
    fn parser_missing_tool_name_bails() {
        let mut parser = OllamaParser::new();
        let chunk = r#"{"message":{"tool_calls":[{"id":"c1","function":{"arguments":{}}}]}}"#;
        let result = parser.parse_chunk(chunk);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("missing or empty function name")
        );
    }

    #[test]
    fn parser_empty_tool_name_bails() {
        let mut parser = OllamaParser::new();
        let chunk =
            r#"{"message":{"tool_calls":[{"id":"c1","function":{"name":"","arguments":{}}}]}}"#;
        let result = parser.parse_chunk(chunk);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("missing or empty function name")
        );
    }

    #[test]
    fn parser_missing_id_generates_fallback() {
        let mut parser = OllamaParser::new();
        let chunk = r#"{"message":{"tool_calls":[{"function":{"name":"bash","arguments":{"command":"ls"}}}]}}"#;
        let events = parser.parse_chunk(chunk).unwrap();
        assert_eq!(events.len(), 3);
        if let StreamEvent::ToolUseStart { id, name } = &events[0] {
            assert_eq!(name, "bash");
            assert_eq!(id, "tool_0", "should generate fallback ID with index");
        } else {
            panic!("expected ToolUseStart");
        }
    }

    #[test]
    fn parser_chunk_with_both_text_and_tool_calls() {
        let mut parser = OllamaParser::new();
        let chunk = r#"{"message":{"content":"Thinking...","tool_calls":[{"id":"c1","function":{"name":"bash","arguments":{"command":"ls"}}}]}}"#;
        let events = parser.parse_chunk(chunk).unwrap();
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Thinking..."));
        assert!(matches!(&events[1], StreamEvent::ToolUseStart { .. }));
        assert!(matches!(&events[2], StreamEvent::ToolUseDelta(_)));
        assert!(matches!(&events[3], StreamEvent::ToolUseDone));
    }

    #[test]
    fn parser_thinking_field_produces_thinking_delta() {
        let mut parser = OllamaParser::new();
        let chunk = r#"{"message":{"content":"","thinking":"Let me reason about this..."}}"#;
        let events = parser.parse_chunk(chunk).unwrap();
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], StreamEvent::ThinkingDelta(t) if t == "Let me reason about this...")
        );
    }

    #[test]
    fn parser_thinking_field_empty_produces_no_thinking_delta() {
        let mut parser = OllamaParser::new();
        let chunk = r#"{"message":{"content":"","thinking":""}}"#;
        let events = parser.parse_chunk(chunk).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn request_body_with_thinking_adds_thinking_option() {
        let backend = OllamaBackend::new(&OllamaConfig {
            api_key: "test-key".to_string(),
            model: "gpt-oss:120b".to_string(),
            base_url: "https://ollama.com/api/chat".to_string(),
        })
        .unwrap();
        let config = RequestConfig {
            model: "gpt-oss:120b".to_string(),
            max_tokens: 4096,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig::default()),
            cancel_token: None,
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
            created_at: 0.0,
        }];

        let body = backend.build_request_body(&messages, &config);
        assert_eq!(body["options"]["thinking"], true);
    }

    #[test]
    fn request_body_without_thinking_has_no_thinking_option() {
        let backend = OllamaBackend::new(&OllamaConfig {
            api_key: "test-key".to_string(),
            model: "gpt-oss:120b".to_string(),
            base_url: "https://ollama.com/api/chat".to_string(),
        })
        .unwrap();
        let config = RequestConfig {
            model: "gpt-oss:120b".to_string(),
            max_tokens: 4096,
            tools: vec![],
            thinking: None,
            cancel_token: None,
        };
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
            created_at: 0.0,
        }];

        let body = backend.build_request_body(&messages, &config);
        assert!(
            body["options"].get("thinking").is_none(),
            "should not include thinking option when config.thinking is None"
        );
    }

    #[test]
    fn parser_chunk_with_both_thinking_and_content() {
        let mut parser = OllamaParser::new();
        let chunk = r#"{"message":{"content":"Here is the answer","thinking":"Let me work through this step by step..."}}"#;
        let events = parser.parse_chunk(chunk).unwrap();
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Here is the answer"));
        assert!(
            matches!(&events[1], StreamEvent::ThinkingDelta(t) if t == "Let me work through this step by step...")
        );
    }
}
