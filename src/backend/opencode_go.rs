use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;

use super::LlmBackend;
use super::anthropic_compat::{AnthropicCompatBackend, AnthropicCompatConfig, AuthStyle};
use super::openai_compat::{OpenAiCompatBackend, OpenAiCompatConfig, ReasoningStyle};
use crate::config::{OPENCODE_GO_DEFAULT_BASE_URL, OpenCodeGoConfig, Protocol};
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const ANTHROPIC_BETA: &str = "interleaved-thinking-2025-05-14";

#[derive(Debug)]
enum ProtocolBackend {
    OpenAi(OpenAiCompatBackend),
    Anthropic(AnthropicCompatBackend),
}

/// Dual-protocol backend for OpenCode Go.
///
/// Triages the model name against config-driven lists to route requests to
/// either the OpenAI Chat Completions or Anthropic Messages protocol.
#[derive(Debug)]
pub struct OpenCodeGoBackend {
    backend: ProtocolBackend,
}

impl OpenCodeGoBackend {
    pub fn new(config: &OpenCodeGoConfig, resolved_model: String) -> Result<Self> {
        if config.api_key.is_empty() {
            anyhow::bail!("API key cannot be empty for opencode_go backend");
        }

        let base_url = config.base_url.trim_end_matches('/').to_string();

        if base_url.ends_with("/chat/completions") {
            anyhow::bail!(
                "base_url must not include '/chat/completions' — provide the base URL only \
                 (e.g. '{OPENCODE_GO_DEFAULT_BASE_URL}') and the backend will append the path automatically."
            );
        }

        if let Some(duplicate) = config.find_duplicate_model() {
            anyhow::bail!(
                "Model '{}' appears in both openai_models and anthropic_models",
                duplicate
            );
        }

        let reasoning = ReasoningStyle::from(&config.reasoning);
        let protocol = config.protocol_for(&resolved_model)?;

        let backend = match protocol {
            Protocol::Anthropic => {
                let endpoint = format!("{base_url}/messages");
                let compat_config = AnthropicCompatConfig {
                    endpoint,
                    auth_token: Some(config.api_key.clone()),
                    auth_style: AuthStyle::XApiKey,
                    anthropic_version: ANTHROPIC_VERSION.to_string(),
                    include_model_in_body: true,
                    anthropic_beta: Some(ANTHROPIC_BETA.to_string()),
                    max_tokens_override: config.max_tokens,
                };
                ProtocolBackend::Anthropic(AnthropicCompatBackend::new(
                    Client::new(),
                    compat_config,
                ))
            }
            Protocol::OpenAi => {
                let oc_config = OpenAiCompatConfig {
                    base_url,
                    api_key: Some(config.api_key.clone()),
                    model: String::new(),
                    max_tokens: config.max_tokens,
                    reasoning,
                };
                ProtocolBackend::OpenAi(OpenAiCompatBackend::new(oc_config)?)
            }
        };

        Ok(Self { backend })
    }
}

#[async_trait]
impl LlmBackend for OpenCodeGoBackend {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        match &self.backend {
            ProtocolBackend::OpenAi(b) => b.send_message(messages, config).await,
            ProtocolBackend::Anthropic(b) => b.send_message(messages, config).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReasoningStyleConfig;

    fn test_config() -> OpenCodeGoConfig {
        OpenCodeGoConfig {
            api_key: "test-key".to_string(),
            base_url: OPENCODE_GO_DEFAULT_BASE_URL.to_string(),
            model: "grok-code".to_string(),
            openai_models: vec!["grok-code".to_string(), "glm-code".to_string()],
            anthropic_models: vec!["minimax-m1".to_string()],
            max_tokens: None,
            reasoning: ReasoningStyleConfig::Default,
            vision: false,
        }
    }

    #[test]
    fn new_rejects_empty_api_key() {
        let mut config = test_config();
        config.api_key = String::new();
        let result = OpenCodeGoBackend::new(&config, "grok-code".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key"));
    }

    #[test]
    fn new_rejects_model_in_both_lists() {
        let mut config = test_config();
        config.anthropic_models.push("grok-code".to_string());
        let result = OpenCodeGoBackend::new(&config, "grok-code".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("both"));
    }

    #[test]
    fn new_rejects_unknown_model() {
        let config = test_config();
        let result = OpenCodeGoBackend::new(&config, "unknown-model".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not in either"));
    }

    #[test]
    fn new_rejects_base_url_with_chat_completions_suffix() {
        let mut config = test_config();
        config.base_url = "https://opencode.ai/zen/go/v1/chat/completions".to_string();
        let result = OpenCodeGoBackend::new(&config, "grok-code".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("chat/completions"));
    }

    #[test]
    fn new_construes_anthropic_backend_for_anthropic_model() {
        let mut config = test_config();
        config.model = "minimax-m1".to_string();
        let backend =
            OpenCodeGoBackend::new(&config, "minimax-m1".to_string()).expect("valid config");
        assert!(matches!(backend.backend, ProtocolBackend::Anthropic(_)));
    }

    #[test]
    fn new_construes_openai_backend_for_openai_model() {
        let config = test_config();
        let backend =
            OpenCodeGoBackend::new(&config, "grok-code".to_string()).expect("valid config");
        assert!(matches!(backend.backend, ProtocolBackend::OpenAi(_)));
    }
}
