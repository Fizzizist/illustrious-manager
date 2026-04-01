use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;

use super::LlmBackend;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

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

    fn build_request_body(&self, messages: &[Message], model: &str) -> serde_json::Value {
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
            "anthropic_version": "vertex-2023-10-16",
            "model": model,
            "max_tokens": 8192,
            "stream": true,
            "messages": messages_json,
        })
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
        let body = self.build_request_body(messages, &config.model);

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
            let body = response.text().await.unwrap_or_default();
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
