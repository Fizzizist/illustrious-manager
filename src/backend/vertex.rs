use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;

use super::LlmBackend;
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

    serde_json::json!({
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": config.max_tokens,
        "stream": true,
        "messages": messages_json,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_request_body_uses_max_tokens_and_model_from_config() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 32768,
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
        };
        let body = build_request_body(&[], &config);
        assert!(
            body.get("model").is_none() || body["model"].is_null(),
            "model must not be in the request body; Vertex AI embeds it in the URL"
        );
    }

    #[test]
    fn extract_sse_data_returns_json_after_data_prefix() {
        let event = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}";
        assert_eq!(
            extract_sse_data(event),
            Some("{\"type\":\"content_block_delta\"}")
        );
    }

    #[test]
    fn extract_sse_data_returns_none_when_no_data_line() {
        let event = "event: content_block_delta\n";
        assert!(extract_sse_data(event).is_none());
    }

    #[test]
    fn extract_sse_data_returns_first_data_line_when_multiple_present() {
        let event = "data: first\ndata: second";
        assert_eq!(extract_sse_data(event), Some("first"));
    }
}

/// Parse an SSE data payload JSON into a StreamEvent.
/// Returns None for event types we intentionally ignore.
pub fn parse_sse_data(data: &str) -> Result<Option<StreamEvent>> {
    let json: serde_json::Value = serde_json::from_str(data)
        .with_context(|| format!("Failed to parse SSE data: {}", data))?;

    let event_type = json["type"].as_str().unwrap_or("");

    match event_type {
        "content_block_delta" => {
            let text = json["delta"]["text"].as_str().unwrap_or("").to_string();
            Ok(Some(StreamEvent::TextDelta(text)))
        }
        "message_stop" => Ok(Some(StreamEvent::Done)),
        _ => Ok(None),
    }
}
