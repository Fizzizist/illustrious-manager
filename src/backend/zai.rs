use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;

use super::LlmBackend;
use super::sse::create_sse_event_stream;
use crate::types::{BoxStream, ContentBlock, Message, RequestConfig, StreamEvent};

const ENDPOINT: &str = "https://api.z.ai/api/coding/paas/v4/chat/completions";

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

        serde_json::json!({
            "model": config.model,
            "messages": messages_json,
            "stream": true,
            "max_tokens": config.max_tokens,
        })
    }
}

#[async_trait]
impl LlmBackend for ZaiBackend {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        if !config.tools.is_empty() {
            anyhow::bail!("z.ai backend does not support tool use");
        }

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
        let event_stream = create_sse_event_stream(byte_stream, parse_sse_data);
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
        let data = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
        let result = parse_sse_data(data);
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
        let data = "[DONE]";
        let result = parse_sse_data(data);
        assert!(result.is_ok(), "Should successfully parse [DONE]");
        let event = result.unwrap();
        assert!(event.is_some(), "Should return Some(event) for [DONE]");
        assert!(matches!(event, Some(StreamEvent::Done)), "Should be Done");
    }

    #[test]
    fn parse_sse_data_returns_error_for_invalid_json() {
        let data = "not valid json";
        let result = parse_sse_data(data);
        assert!(result.is_err(), "Should return error for invalid JSON");
    }

    #[test]
    fn parse_sse_data_returns_none_for_empty_delta() {
        let data = r#"{"choices":[{"delta":{}}]}"#;
        let result = parse_sse_data(data);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn send_message_returns_error_when_tools_are_present() {
        let backend = ZaiBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "glm-5-turbo".to_string(),
            max_tokens: 4096,
            tools: vec![crate::types::ToolDefinition {
                name: "bash".to_string(),
                description: "Run bash".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
        };
        match backend.send_message(&[], &config).await {
            Err(e) => assert!(
                e.to_string().contains("tool"),
                "error message should mention tools, got: {}",
                e
            ),
            Ok(_) => panic!("should reject requests with tools"),
        }
    }
}
