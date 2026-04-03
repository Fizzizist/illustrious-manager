use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;

use super::LlmBackend;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

/// Z.ai backend for LLM models.
#[derive(Debug)]
pub struct ZaiBackend {
    client: Client,
    api_key: String,
}

impl ZaiBackend {
    /// Create a new ZaiBackend using an API key.
    pub fn new(api_key: String) -> Result<Self> {
        if api_key.is_empty() {
            anyhow::bail!("API key cannot be empty");
        }
        Ok(Self {
            client: Client::new(),
            api_key,
        })
    }

    fn endpoint(&self) -> String {
        "https://api.z.ai/v1/chat/completions".to_string()
    }

    fn build_request_body(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> serde_json::Value {
        let messages_json: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
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
        let url = self.endpoint();
        let body = self.build_request_body(messages, config);

        let response = self
            .client
            .post(&url)
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

        let event_stream = futures::stream::unfold(
            (byte_stream, String::new()),
            |(mut byte_stream, mut buffer)| async move {
                loop {
                    if let Some(pos) = buffer.find("\n\n") {
                        let event_text = buffer[..pos].to_string();
                        buffer = buffer[pos + 2..].to_string();

                        if let Some(data) = extract_sse_data(&event_text) {
                            match parse_sse_data(data) {
                                Ok(Some(event)) => return Some((Ok(event), (byte_stream, buffer))),
                                Ok(None) => continue,
                                Err(e) => return Some((Err(e), (byte_stream, buffer))),
                            }
                        }
                        continue;
                    }

                    match byte_stream.next().await {
                        Some(Ok(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Some(Err(e)) => {
                            return Some((
                                Err(anyhow::anyhow!("Stream read error: {}", e)),
                                (byte_stream, buffer),
                            ));
                        }
                        None => {
                            if !buffer.trim().is_empty()
                                && let Some(data) = extract_sse_data(&buffer).map(str::to_owned)
                            {
                                buffer.clear();
                                match parse_sse_data(&data) {
                                    Ok(Some(event)) => {
                                        return Some((Ok(event), (byte_stream, buffer)));
                                    }
                                    Ok(None) => return None,
                                    Err(e) => return Some((Err(e), (byte_stream, buffer))),
                                }
                            }
                            return None;
                        }
                    }
                }
            },
        );

        Ok(Box::pin(event_stream))
    }
}

/// Extract the data payload from an SSE event block.
fn extract_sse_data(event_text: &str) -> Option<&str> {
    for line in event_text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            return Some(data);
        }
    }
    None
}

/// Parse an SSE data payload JSON into a StreamEvent.
/// Returns None for event types we intentionally ignore.
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
    fn endpoint_returns_correct_url() {
        let backend = ZaiBackend::new("test-key".to_string()).unwrap();
        assert_eq!(backend.endpoint(), "https://api.z.ai/v1/chat/completions");
    }

    #[test]
    fn build_request_body_includes_model_and_stream() {
        let backend = ZaiBackend::new("test-key".to_string()).unwrap();
        let config = RequestConfig {
            model: "glm-5-turbo".to_string(),
            max_tokens: 4096,
        };
        let messages = vec![Message {
            role: Role::User,
            content: "Hello".to_string(),
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
        };
        let messages = vec![
            Message {
                role: Role::User,
                content: "Hello".to_string(),
            },
            Message {
                role: Role::Assistant,
                content: "Hi there!".to_string(),
            },
        ];

        let body = backend.build_request_body(&messages, &config);

        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "Hello");
        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["messages"][1]["content"], "Hi there!");
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
}
