use anyhow::Result;
use async_trait::async_trait;

use super::CancellationToken;
use super::LlmBackend;
use super::openai_compat::{OpenAiCompatBackend, OpenAiCompatConfig, ReasoningStyle};
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

/// Back-compat re-export: integration tests that reference `ZaiSseParser` continue to compile.
pub use super::openai_compat::OpenAiCompatSseParser as ZaiSseParser;

const BASE_URL: &str = "https://api.z.ai/api/coding/paas/v4";

/// Thin shim over [`OpenAiCompatBackend`] for the z.ai provider.
#[derive(Debug)]
pub struct ZaiBackend(OpenAiCompatBackend);

impl ZaiBackend {
    pub fn new(api_key: String) -> Result<Self> {
        if api_key.is_empty() {
            anyhow::bail!("API key cannot be empty");
        }
        Ok(Self(OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: BASE_URL.to_string(),
            api_key: Some(api_key),
            model: String::new(),
            max_tokens: None,
            reasoning: ReasoningStyle::ZaiEnableThinking,
        })?))
    }
}

#[async_trait]
impl LlmBackend for ZaiBackend {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
        cancel_token: Option<CancellationToken>,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        self.0.send_message(messages, config, cancel_token).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zai_backend_new_rejects_empty_api_key() {
        let err = ZaiBackend::new(String::new()).unwrap_err();
        assert!(err.to_string().contains("API key"));
    }

    #[test]
    fn zai_backend_delegates_send_message_to_openai_compat() {
        let backend = ZaiBackend::new("test-key".to_string()).expect("valid key");
        assert_eq!(
            backend.0.endpoint(),
            "https://api.z.ai/api/coding/paas/v4/chat/completions"
        );
    }
}
