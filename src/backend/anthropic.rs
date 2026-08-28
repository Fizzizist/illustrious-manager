use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;

use super::LlmBackend;
use super::anthropic_compat::{AnthropicCompatBackend, AnthropicCompatConfig, AuthStyle};
use crate::config::AnthropicConfig;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Thin shim over [`AnthropicCompatBackend`] for the direct Anthropic
/// Messages API (`POST https://api.anthropic.com/v1/messages`).
///
/// Uses `x-api-key` authentication, sends `anthropic-version` as an HTTP
/// header (not a body field), includes `model` in the request body, and
/// omits `anthropic-beta` — adaptive thinking on Opus 4.8 auto-enables
/// interleaved thinking without a beta header.
#[derive(Debug)]
pub struct AnthropicBackend(AnthropicCompatBackend);

impl AnthropicBackend {
    pub fn new(config: &AnthropicConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            anyhow::bail!("API key cannot be empty for anthropic backend");
        }

        let base_url = config.base_url.trim_end_matches('/').to_string();

        if base_url.ends_with("/messages") {
            anyhow::bail!(
                "base_url must not include '/messages' — provide the base URL only \
                 (e.g. '{}') and the backend will append the path automatically.",
                crate::config::ANTHROPIC_DEFAULT_BASE_URL,
            );
        }

        let endpoint = format!("{base_url}/messages");
        let compat_config = AnthropicCompatConfig {
            endpoint,
            auth_token: Some(config.api_key.clone()),
            auth_style: AuthStyle::XApiKey,
            anthropic_version: ANTHROPIC_VERSION.to_string(),
            include_model_in_body: true,
            anthropic_beta: None,
            max_tokens_override: config.max_tokens,
        };

        Ok(Self(AnthropicCompatBackend::new(
            Client::new(),
            compat_config,
        )))
    }
}

#[async_trait]
impl LlmBackend for AnthropicBackend {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        self.0.send_message(messages, config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AnthropicConfig {
        AnthropicConfig {
            api_key: "test-key".to_string(),
            base_url: crate::config::ANTHROPIC_DEFAULT_BASE_URL.to_string(),
            model: "claude-opus-4-8".to_string(),
            max_tokens: None,
            vision: false,
        }
    }

    #[test]
    fn new_rejects_empty_api_key() {
        let mut config = test_config();
        config.api_key = String::new();
        let result = AnthropicBackend::new(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key"));
    }

    #[test]
    fn new_rejects_base_url_with_messages_suffix() {
        let mut config = test_config();
        config.base_url = "https://api.anthropic.com/v1/messages".to_string();
        let result = AnthropicBackend::new(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("/messages"));
    }

    #[test]
    fn new_strips_trailing_slash() {
        let mut config = test_config();
        config.base_url = "https://api.anthropic.com/v1/".to_string();
        let backend = AnthropicBackend::new(&config).expect("valid config");
        assert_eq!(
            backend.0.config().endpoint,
            "https://api.anthropic.com/v1/messages",
        );
    }

    #[test]
    fn new_builds_correct_compat_config() {
        let config = test_config();
        let backend = AnthropicBackend::new(&config).expect("valid config");
        let compat = backend.0.config();
        assert_eq!(compat.auth_style, AuthStyle::XApiKey);
        assert!(compat.include_model_in_body);
        assert_eq!(compat.anthropic_version, "2023-06-01");
        assert!(compat.anthropic_beta.is_none());
        assert!(compat.endpoint.ends_with("/messages"));
        assert_eq!(compat.auth_token.as_deref(), Some("test-key"));
    }

    #[test]
    fn new_passes_through_max_tokens_override() {
        let mut config = test_config();
        config.max_tokens = Some(65536);
        let backend = AnthropicBackend::new(&config).expect("valid config");
        let compat = backend.0.config();
        assert_eq!(compat.max_tokens_override, Some(65536));
    }

    #[test]
    fn new_omits_max_tokens_override_when_absent() {
        let config = test_config();
        let backend = AnthropicBackend::new(&config).expect("valid config");
        let compat = backend.0.config();
        assert!(compat.max_tokens_override.is_none());
    }
}
