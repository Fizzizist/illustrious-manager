use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;

use super::LlmBackend;
use super::openai_compat;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

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
}

#[async_trait]
impl LlmBackend for OllamaBackend {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        openai_compat::send_request(
            &self.client,
            &self.api_key,
            ENDPOINT,
            messages,
            config,
            "Ollama",
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
