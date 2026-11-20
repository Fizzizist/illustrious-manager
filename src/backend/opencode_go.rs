use anyhow::Result;
use async_trait::async_trait;

use super::LlmBackend;
use super::anthropic_compat::{AnthropicCompatBackend, AnthropicCompatConfig, AuthStyle};
use super::openai_compat::{OpenAiCompatBackend, OpenAiCompatConfig, ReasoningStyle};
use crate::config::{OPENCODE_GO_DEFAULT_BASE_URL, OpenCodeGoConfig, Protocol};
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const ANTHROPIC_BETA: &str = "interleaved-thinking-2025-05-14";
const SESSION_HEADER: &str = "x-opencode-session";

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
                    session_header_name: Some(SESSION_HEADER.to_string()),
                };
                ProtocolBackend::Anthropic(AnthropicCompatBackend::new(
                    super::build_http_client()?,
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
                    session_header_name: Some(SESSION_HEADER.to_string()),
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

    fn wire_config(port: u16, model: &str) -> OpenCodeGoConfig {
        OpenCodeGoConfig {
            api_key: "test-key".to_string(),
            base_url: format!("http://127.0.0.1:{port}"),
            model: model.to_string(),
            openai_models: vec!["grok-code".to_string()],
            anthropic_models: vec!["minimax-m1".to_string()],
            max_tokens: None,
            reasoning: ReasoningStyleConfig::Default,
        }
    }

    fn session_stamped_config(model: &str, session_id: &str) -> RequestConfig {
        RequestConfig {
            model: model.to_string(),
            max_tokens: 1024,
            session_id: session_id.to_string(),
            ..Default::default()
        }
    }

    /// Accept one connection and read until the end of the request head
    /// (`\r\n\r\n`), bounded so a malformed capture cannot hang the test.
    /// Returns the captured request head.
    async fn capture_request_head(listener: tokio::net::TcpListener) -> String {
        let (mut socket, _) = listener
            .accept()
            .await
            .expect("listener should accept a connection");

        let mut head = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                tokio::io::AsyncReadExt::read(&mut socket, &mut buf),
            )
            .await
            .expect("bounded read must not hang")
            .expect("read should succeed");
            if n == 0 {
                break;
            }
            head.extend_from_slice(&buf[..n]);
            if head.windows(4).any(|w| w == b"\r\n\r\n") || head.len() >= 64 * 1024 {
                break;
            }
        }
        String::from_utf8_lossy(&head).to_string()
    }

    async fn assert_wire_headers(model: &str, session_id: &str) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let port = listener.local_addr().expect("local addr").port();

        let backend = OpenCodeGoBackend::new(&wire_config(port, model), model.to_string())
            .expect("backend should construct");

        let capture = tokio::spawn(capture_request_head(listener));

        // The capture server drops the socket without responding, so a send
        // error is expected and ignored (cancel_blackhole precedent).
        let _ = backend
            .send_message(&[], &session_stamped_config(model, session_id))
            .await;

        let head = capture.await.expect("capture task should not panic");

        let lower = head.to_lowercase();
        assert!(
            lower.contains(&format!("x-opencode-session: {session_id}")),
            "opencode_go ({model}) must send the session header; captured head:\n{head}"
        );
        assert!(
            lower.contains("user-agent: illustrious-manager/"),
            "opencode_go ({model}) must send the User-Agent; captured head:\n{head}"
        );
    }

    #[tokio::test]
    async fn openai_protocol_wire_request_carries_session_and_user_agent() {
        assert_wire_headers("grok-code", "sess-openai-42").await;
    }

    #[tokio::test]
    async fn anthropic_protocol_wire_request_carries_session_and_user_agent() {
        assert_wire_headers("minimax-m1", "sess-anthropic-42").await;
    }
}
