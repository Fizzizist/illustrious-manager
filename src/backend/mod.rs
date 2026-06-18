use anyhow::Result;
use async_trait::async_trait;

use crate::config::AppConfig;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

pub mod error;
pub mod ndjson;
pub mod ollama;
pub mod openai_compat;
pub mod sse;
pub mod vertex;
pub mod zai;

#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>>;
}

pub struct BackendSelection {
    pub backend: Box<dyn LlmBackend>,
    pub model: String,
}

/// Factory that constructs backends on demand, sharing expensive auth state
/// across roles that target the same Vertex AI `(project, region)` pair.
///
/// Note: per-role `project`/`region` overrides are not yet wired into
/// `ModelRole`; all Vertex roles currently share the same
/// `config.vertex.{project,region}` values, so the cache will hold at most
/// one entry until per-role overrides land in a future PR.
pub struct BackendFactory {
    config: AppConfig,
    vertex_auth_cache: vertex::VertexAuthCache,
}

impl BackendFactory {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config,
            vertex_auth_cache: vertex::VertexAuthCache::new(),
        }
    }

    /// Construct a `BackendSelection` for the named role.
    pub async fn for_role(&self, role: &str) -> Result<BackendSelection> {
        let resolved = self.config.resolve_role(role)?;
        match resolved.backend_name.as_str() {
            "vertex" => {
                // TODO: per-role project/region overrides — when `ModelRole` gains
                // those fields, pass them here instead of reading from `self.config.vertex`.
                let project = self.config.vertex.project.clone();
                let region = self.config.vertex.region.clone();
                let auth = self
                    .vertex_auth_cache
                    .get_or_init(project.clone(), region.clone())
                    .await?;
                let backend = vertex::VertexBackend::with_auth(project, region, auth);
                Ok(BackendSelection {
                    backend: Box::new(backend),
                    model: resolved.model,
                })
            }
            "zai" => {
                let zai_config = self.config.zai.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Role '{role}' uses zai backend but no [zai] section is configured."
                    )
                })?;
                let backend = zai::ZaiBackend::new(zai_config.api_key.clone())?;
                Ok(BackendSelection {
                    backend: Box::new(backend),
                    model: resolved.model,
                })
            }
            "ollama" => {
                let ollama_config = self.config.ollama.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Role '{role}' uses ollama backend but no [ollama] section is configured."
                    )
                })?;
                let backend = ollama::OllamaBackend::new(ollama_config)?;
                Ok(BackendSelection {
                    backend: Box::new(backend),
                    model: resolved.model,
                })
            }
            "openai_compat" => {
                let oc_toml = self.config.openai_compat.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Role '{role}' uses openai_compat backend but no [openai_compat] section is configured."
                    )
                })?;
                use crate::config::ReasoningStyleConfig;
                let reasoning = match oc_toml.reasoning {
                    ReasoningStyleConfig::ZaiEnableThinking => {
                        openai_compat::ReasoningStyle::ZaiEnableThinking
                    }
                    ReasoningStyleConfig::QwenChatTemplate => {
                        openai_compat::ReasoningStyle::QwenChatTemplate
                    }
                    ReasoningStyleConfig::Default => openai_compat::ReasoningStyle::Default,
                    ReasoningStyleConfig::None => openai_compat::ReasoningStyle::None,
                };
                let oc_config = openai_compat::OpenAiCompatConfig {
                    base_url: oc_toml.base_url.clone(),
                    api_key: oc_toml.api_key.clone(),
                    model: resolved.model.clone(),
                    max_tokens: oc_toml.max_tokens,
                    reasoning,
                };
                let backend = openai_compat::OpenAiCompatBackend::new(oc_config)?;
                Ok(BackendSelection {
                    backend: Box::new(backend),
                    model: resolved.model,
                })
            }
            other => anyhow::bail!("Unknown backend '{other}' for role '{role}'"),
        }
    }

    /// Inject a pre-built `BackendSelection` for testing without real auth.
    ///
    /// Seeds the Vertex auth cache with a fake provider so that `for_role`
    /// returns a backend built from the supplied `selection` when the role
    /// is `"vertex"`.  Only intended for use in tests.
    #[cfg(test)]
    pub async fn with_injected_selection(
        config: AppConfig,
        selection: BackendSelection,
    ) -> (Self, BackendSelection) {
        (Self::new(config), selection)
    }

    /// Build a `BackendSelection` directly from a boxed backend and model string,
    /// bypassing role resolution and auth. For use in tests only.
    #[cfg(test)]
    pub fn make_selection(backend: Box<dyn LlmBackend>, model: String) -> BackendSelection {
        BackendSelection { backend, model }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::CompactionConfig;
    use anyhow::Result;
    use async_trait::async_trait;
    use futures::{StreamExt, stream};
    use std::sync::Arc;

    use super::LlmBackend;
    use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

    struct EchoBackend;

    #[async_trait]
    impl LlmBackend for EchoBackend {
        async fn send_message(
            &self,
            _messages: &[Message],
            _config: &RequestConfig,
        ) -> Result<BoxStream<Result<StreamEvent>>> {
            let events = vec![
                Ok(StreamEvent::TextDelta("hello".to_string())),
                Ok(StreamEvent::Done),
            ];
            Ok(Box::pin(stream::iter(events)))
        }
    }

    #[tokio::test]
    async fn mock_backend_streams_text_delta_then_done() {
        let backend = EchoBackend;
        let messages: Vec<Message> = vec![];
        let config = RequestConfig {
            model: "test-model".to_string(),
            max_tokens: 1024,
            tools: vec![],
            thinking: None,
        };

        let mut stream = backend
            .send_message(&messages, &config)
            .await
            .expect("send_message should succeed");

        let first = stream
            .next()
            .await
            .expect("stream should have first event")
            .expect("first event should be Ok");
        assert!(
            matches!(first, StreamEvent::TextDelta(_)),
            "expected TextDelta, got {first:?}"
        );

        let second = stream
            .next()
            .await
            .expect("stream should have second event")
            .expect("second event should be Ok");
        assert!(
            matches!(second, StreamEvent::Done),
            "expected Done, got {second:?}"
        );

        assert!(
            stream.next().await.is_none(),
            "stream should be exhausted after Done"
        );
    }

    #[tokio::test]
    async fn backend_factory_errors_for_unknown_role() {
        use crate::config::{AppConfig, RetryConfig, ToolsConfig, VertexConfig};
        use std::collections::BTreeMap;

        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let factory = super::BackendFactory::new(config);

        let err = factory
            .for_role("nonexistent")
            .await
            .err()
            .expect("should be an error");
        assert!(
            err.to_string().contains("nonexistent"),
            "error should mention the unknown role name"
        );
    }

    #[tokio::test]
    async fn backend_factory_caches_vertex_auth_provider_per_project_region() {
        use crate::config::{AppConfig, ModelRole, RetryConfig, ToolsConfig, VertexConfig};
        use std::collections::BTreeMap;

        let mut models = BTreeMap::new();
        models.insert(
            "role-a".to_string(),
            ModelRole {
                backend: "vertex".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
        );
        models.insert(
            "role-b".to_string(),
            ModelRole {
                backend: "vertex".to_string(),
                model: "claude-haiku".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "shared-project".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let factory = super::BackendFactory::new(config);

        // Pre-seed the cache with a fake provider so we don't call real gcp_auth.
        let fake_provider: Arc<dyn gcp_auth::TokenProvider> = Arc::new(FakeTokenProvider);
        let cell = factory
            .vertex_auth_cache
            .seed("shared-project", "us-east5", fake_provider);

        // Two calls to get_or_init must return the same Arc.
        let first = factory
            .vertex_auth_cache
            .get_or_init("shared-project".to_string(), "us-east5".to_string())
            .await
            .expect("first call");
        let second = factory
            .vertex_auth_cache
            .get_or_init("shared-project".to_string(), "us-east5".to_string())
            .await
            .expect("second call");

        assert!(
            Arc::ptr_eq(&first, &second),
            "both calls must return the same Arc — provider is shared"
        );
        assert!(
            Arc::ptr_eq(&first, cell.get().expect("cell initialised")),
            "returned provider must be the one we seeded"
        );
    }

    struct FakeTokenProvider;

    #[async_trait::async_trait]
    impl gcp_auth::TokenProvider for FakeTokenProvider {
        async fn token(
            &self,
            _scopes: &[&str],
        ) -> Result<std::sync::Arc<gcp_auth::Token>, gcp_auth::Error> {
            unimplemented!("fake provider")
        }
        async fn project_id(&self) -> Result<std::sync::Arc<str>, gcp_auth::Error> {
            unimplemented!("fake provider")
        }
    }
}
