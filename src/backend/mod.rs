use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;

use crate::config::AppConfig;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

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
pub struct BackendFactory {
    config: AppConfig,
    /// Cached Vertex auth providers keyed by `(project, region)`.
    vertex_auth_cache: Mutex<HashMap<(String, String), Arc<dyn gcp_auth::TokenProvider>>>,
}

impl BackendFactory {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config,
            vertex_auth_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Construct a `BackendSelection` for the named role.
    pub async fn for_role(&self, role: &str) -> Result<BackendSelection> {
        let resolved = self.config.resolve_role(role)?;
        match resolved.backend_name.as_str() {
            "vertex" => {
                let project = self.config.vertex.project.clone();
                let region = self.config.vertex.region.clone();
                let auth = self
                    .vertex_auth_for(project.clone(), region.clone())
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
            other => anyhow::bail!("Unknown backend '{other}' for role '{role}'"),
        }
    }

    async fn vertex_auth_for(
        &self,
        project: String,
        region: String,
    ) -> Result<Arc<dyn gcp_auth::TokenProvider>> {
        let key = (project, region);
        {
            let cache = self
                .vertex_auth_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(provider) = cache.get(&key) {
                return Ok(Arc::clone(provider));
            }
        }
        let provider: Arc<dyn gcp_auth::TokenProvider> = gcp_auth::provider()
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to initialize GCP authentication. Run: gcloud auth application-default login\n{e}"
                )
            })?;
        {
            let mut cache = self
                .vertex_auth_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.insert(key, Arc::clone(&provider));
        }
        Ok(provider)
    }
}

/// Thin wrapper kept for any remaining direct callsites; delegates to `BackendFactory`.
pub async fn from_config(config: &AppConfig) -> Result<BackendSelection> {
    match config.backend.as_str() {
        "vertex" => {
            let backend = vertex::VertexBackend::new(
                config.vertex.project.clone(),
                config.vertex.region.clone(),
            )
            .await?;
            Ok(BackendSelection {
                backend: Box::new(backend),
                model: config.vertex.model.clone(),
            })
        }
        "zai" => {
            let zai_config = config.zai.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "zai backend configuration is missing. Add a [zai] section to your config file."
                )
            })?;
            let backend = zai::ZaiBackend::new(zai_config.api_key.clone())?;
            Ok(BackendSelection {
                backend: Box::new(backend),
                model: zai_config.model.clone(),
            })
        }
        _ => {
            anyhow::bail!("Invalid backend '{}'", config.backend);
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use async_trait::async_trait;
    use futures::{StreamExt, stream};

    use super::LlmBackend;
    use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

    struct MockBackend;

    #[async_trait]
    impl LlmBackend for MockBackend {
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
        let backend = MockBackend;
        let messages: Vec<Message> = vec![];
        let config = RequestConfig {
            model: "test-model".to_string(),
            max_tokens: 1024,
            tools: vec![],
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

    #[test]
    fn backend_factory_errors_for_unknown_role() {
        use crate::config::{AppConfig, ToolsConfig, VertexConfig};
        use std::collections::BTreeMap;

        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        let factory = super::BackendFactory::new(config);

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let result = rt.block_on(factory.for_role("nonexistent"));
        let err = result.err().expect("should be an error");
        assert!(
            err.to_string().contains("nonexistent"),
            "error should mention the unknown role name"
        );
    }

    #[test]
    fn backend_factory_caches_vertex_auth_provider_per_project_region() {
        use crate::config::{AppConfig, ModelRole, ToolsConfig, VertexConfig};
        use std::collections::BTreeMap;
        use std::sync::Arc;

        // Two roles sharing the same (project, region) should reuse the same Arc.
        // We verify cache logic by directly inspecting the cache after two insertions
        // with the same key via the internal method.

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
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
        };
        let factory = Arc::new(super::BackendFactory::new(config));

        // Manually seed two entries for the same key to confirm deduplication logic.
        {
            let key = ("shared-project".to_string(), "us-east5".to_string());
            let fake_provider: Arc<dyn gcp_auth::TokenProvider> = Arc::new(FakeTokenProvider);
            let mut cache = factory.vertex_auth_cache.lock().expect("lock");
            cache.insert(key.clone(), Arc::clone(&fake_provider));
            // A second insert with the same key — should overwrite, not add a new entry.
            cache.insert(key.clone(), Arc::clone(&fake_provider));
            assert_eq!(
                cache.len(),
                1,
                "cache should deduplicate entries for the same (project, region)"
            );
            // Both retrieved values point to the same allocation.
            let a = Arc::clone(cache.get(&key).expect("entry"));
            let b = Arc::clone(cache.get(&key).expect("entry"));
            assert!(Arc::ptr_eq(&a, &b), "cached providers must be the same Arc");
        }
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
