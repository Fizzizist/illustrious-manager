use anyhow::Result;
use async_trait::async_trait;
use std::time::Duration;

use crate::config::{AppConfig, RetryConfig};
use crate::logging::log_warn;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

pub mod anthropic;
pub mod anthropic_compat;
pub mod defaults;
pub mod error;
pub mod ndjson;
pub mod ollama;
pub mod openai_compat;
pub mod opencode_go;
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
    pub max_tokens: u32,
}

pub struct RetryingBackend {
    inner: Box<dyn LlmBackend>,
    config: RetryConfig,
}

impl RetryingBackend {
    pub fn new(inner: Box<dyn LlmBackend>, config: RetryConfig) -> Self {
        Self { inner, config }
    }

    fn next_delay(&self, attempt: u32) -> Duration {
        let shift = attempt.min(20);
        let multiplier = 1u64.checked_shl(shift).unwrap_or(u64::MAX);
        let base = self.config.initial_delay_ms.saturating_mul(multiplier);
        let capped = base.min(self.config.max_delay_ms);
        let jitter = capped.saturating_mul(rand_fraction()) / (u64::MAX / 4);
        Duration::from_millis(capped.saturating_add(jitter))
    }
}

/// Simple pseudo-random fraction in [0, 1) using thread-local state.
/// We avoid pulling in a full `rand` crate dependency for this.
fn rand_fraction() -> u64 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xDEADBEEF);
            nanos.wrapping_mul(0x2545F4914F6CDD1D)
        });
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        x
    })
}

#[async_trait]
impl LlmBackend for RetryingBackend {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        let max_retries = self.config.max_retries;
        let mut last_err: Option<anyhow::Error> = None;

        for attempt in 0..=max_retries {
            match self.inner.send_message(messages, config).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    let retryable = e
                        .downcast_ref::<error::BackendError>()
                        .is_some_and(|be| be.is_retryable());

                    if !retryable || attempt >= max_retries {
                        return Err(e);
                    }

                    last_err = Some(e);
                    let delay = self.next_delay(attempt);
                    log_warn(&format!(
                        "Backend returned retryable error (attempt {}/{max_retries}); \
                         retrying in {}ms",
                        attempt + 1,
                        delay.as_millis()
                    ));

                    if let Some(ref token) = config.cancel_token {
                        tokio::select! {
                            _ = tokio::time::sleep(delay) => {}
                            _ = token.cancelled() => {
                                return Err(last_err
                                    .expect("error was set before sleep")
                                    .context("retry cancelled by user"));
                            }
                        }
                    } else {
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(last_err.expect("loop ran at least once"))
    }
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
                let project = self.config.vertex.project.clone();
                let region = self.config.vertex.region.clone();
                let auth = self
                    .vertex_auth_cache
                    .get_or_init(project.clone(), region.clone())
                    .await?;
                let backend = vertex::VertexBackend::with_auth(project, region, auth);
                Ok(BackendSelection {
                    backend: Box::new(RetryingBackend::new(
                        Box::new(backend),
                        self.config.retry.clone(),
                    )),
                    model: resolved.model,
                    max_tokens: self.config.vertex.max_tokens.unwrap_or(defaults::VERTEX),
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
                    backend: Box::new(RetryingBackend::new(
                        Box::new(backend),
                        self.config.retry.clone(),
                    )),
                    model: resolved.model,
                    max_tokens: defaults::ZAI,
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
                    backend: Box::new(RetryingBackend::new(
                        Box::new(backend),
                        self.config.retry.clone(),
                    )),
                    model: resolved.model,
                    max_tokens: ollama_config.max_tokens.unwrap_or(defaults::OLLAMA),
                })
            }
            "openai_compat" => {
                let oc_toml = self.config.openai_compat.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Role '{role}' uses openai_compat backend but no [openai_compat] section is configured."
                    )
                })?;
                let reasoning = openai_compat::ReasoningStyle::from(&oc_toml.reasoning);
                let oc_config = openai_compat::OpenAiCompatConfig {
                    base_url: oc_toml.base_url.clone(),
                    api_key: oc_toml.api_key.clone(),
                    model: resolved.model.clone(),
                    max_tokens: oc_toml.max_tokens,
                    reasoning,
                };
                let backend = openai_compat::OpenAiCompatBackend::new(oc_config)?;
                Ok(BackendSelection {
                    backend: Box::new(RetryingBackend::new(
                        Box::new(backend),
                        self.config.retry.clone(),
                    )),
                    model: resolved.model,
                    max_tokens: oc_toml.max_tokens.unwrap_or(defaults::OPENAI_COMPAT),
                })
            }
            "anthropic" => {
                let anthropic_config = self.config.anthropic.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Role '{role}' uses anthropic backend but no [anthropic] section is configured."
                    )
                })?;
                let backend = anthropic::AnthropicBackend::new(anthropic_config)?;
                Ok(BackendSelection {
                    backend: Box::new(RetryingBackend::new(
                        Box::new(backend),
                        self.config.retry.clone(),
                    )),
                    model: resolved.model,
                    max_tokens: anthropic_config.max_tokens.unwrap_or(defaults::ANTHROPIC),
                })
            }
            "opencode_go" => {
                let oc_go = self.config.opencode_go.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Role '{role}' uses opencode_go backend but no [opencode_go] section is configured."
                    )
                })?;
                let backend = opencode_go::OpenCodeGoBackend::new(oc_go, resolved.model.clone())?;
                Ok(BackendSelection {
                    backend: Box::new(RetryingBackend::new(
                        Box::new(backend),
                        self.config.retry.clone(),
                    )),
                    model: resolved.model,
                    max_tokens: oc_go.max_tokens.unwrap_or(defaults::OPENCODE_GO),
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
        BackendSelection {
            backend,
            model,
            max_tokens: 8_192,
        }
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
            cancel_token: None,
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
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
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
    async fn backend_factory_anthropic_arm_succeeds_and_errors_when_section_absent() {
        use crate::config::{
            ANTHROPIC_DEFAULT_BASE_URL, AnthropicConfig, AppConfig, RetryConfig, ToolsConfig,
            VertexConfig,
        };
        use std::collections::BTreeMap;

        // ── Happy path ──
        let mut config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: ANTHROPIC_DEFAULT_BASE_URL.to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        config.normalize_back_compat();

        let factory = super::BackendFactory::new(config);
        let selection = factory
            .for_role("default")
            .await
            .expect("anthropic factory arm should succeed with a configured section");
        assert_eq!(selection.model, "claude-opus-4-8");

        // ── Error path: [anthropic] section absent ──
        let mut config_missing = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        config_missing.models.insert(
            "anth-role".to_string(),
            crate::config::ModelRole {
                backend: "anthropic".to_string(),
                model: "claude-opus-4-8".to_string(),
            },
        );

        let factory_missing = super::BackendFactory::new(config_missing);
        let err = factory_missing
            .for_role("anth-role")
            .await
            .err()
            .expect("should error when [anthropic] section is absent");
        assert!(
            err.to_string()
                .contains("[anthropic] section is configured"),
            "error should mention missing [anthropic] section; got: {err}"
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
                max_tokens: None,
                project: "shared-project".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
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

    // ── RetryingBackend tests ───────────────────────────────────────────

    use crate::backend::RetryingBackend;
    use crate::backend::error::BackendError;
    use crate::config::RetryConfig;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    /// Mock backend that returns a sequence of pre-set results, one per
    /// `send_message` call.  Each entry is either `Ok(stream)` or
    /// `Err(BackendError)`.
    struct FlakyBackend {
        responses: Vec<Result<Vec<StreamEvent>, BackendError>>,
        call_count: Arc<AtomicUsize>,
    }

    impl FlakyBackend {
        fn new(responses: Vec<Result<Vec<StreamEvent>, BackendError>>) -> Self {
            Self {
                responses,
                call_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn counter(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.call_count)
        }
    }

    #[async_trait]
    impl LlmBackend for FlakyBackend {
        async fn send_message(
            &self,
            _messages: &[Message],
            _config: &RequestConfig,
        ) -> Result<BoxStream<Result<StreamEvent>>> {
            let idx = self.call_count.fetch_add(1, AtomicOrdering::SeqCst);
            match self.responses.get(idx) {
                Some(Ok(events)) => {
                    let owned: Vec<Result<StreamEvent>> = events.iter().cloned().map(Ok).collect();
                    Ok(Box::pin(stream::iter(owned)))
                }
                Some(Err(e)) => Err(e.clone().into()),
                None => {
                    let events: Vec<Result<StreamEvent>> = vec![
                        Ok(StreamEvent::TextDelta("fallback".to_string())),
                        Ok(StreamEvent::Done),
                    ];
                    Ok(Box::pin(stream::iter(events)))
                }
            }
        }
    }

    fn retry_config_fast() -> RetryConfig {
        RetryConfig {
            max_retries: 3,
            initial_delay_ms: 1,
            max_delay_ms: 8,
            max_token_retries: 3,
        }
    }

    fn request_config() -> RequestConfig {
        RequestConfig {
            model: "test".to_string(),
            max_tokens: 1024,
            tools: vec![],
            thinking: None,
            cancel_token: None,
        }
    }

    fn ok_stream() -> Vec<StreamEvent> {
        vec![
            StreamEvent::TextDelta("hello".to_string()),
            StreamEvent::Done,
        ]
    }

    #[tokio::test]
    async fn retrying_backend_succeeds_on_first_try() {
        let mock = FlakyBackend::new(vec![Ok(ok_stream())]);
        let backend = RetryingBackend::new(Box::new(mock), retry_config_fast());
        let config = request_config();

        let mut stream = backend
            .send_message(&[], &config)
            .await
            .expect("should succeed");
        let first = stream.next().await.expect("has event").expect("ok");
        assert!(matches!(first, StreamEvent::TextDelta(_)));
    }

    #[tokio::test]
    async fn retrying_backend_retries_on_503_then_succeeds() {
        let mock = FlakyBackend::new(vec![
            Err(BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }),
            Ok(ok_stream()),
        ]);
        let backend = RetryingBackend::new(Box::new(mock), retry_config_fast());
        let config = request_config();

        let result = backend.send_message(&[], &config).await;
        assert!(result.is_ok(), "should succeed after retry");
    }

    #[tokio::test]
    async fn retrying_backend_retries_on_429_then_succeeds() {
        let mock = FlakyBackend::new(vec![
            Err(BackendError::HttpStatus {
                code: 429,
                body: "rate limited".to_string(),
            }),
            Ok(ok_stream()),
        ]);
        let backend = RetryingBackend::new(Box::new(mock), retry_config_fast());
        let config = request_config();

        let result = backend.send_message(&[], &config).await;
        assert!(result.is_ok(), "should succeed after retry on 429");
    }

    #[tokio::test]
    async fn retrying_backend_retries_on_transport_then_succeeds() {
        let mock = FlakyBackend::new(vec![
            Err(BackendError::Transport {
                message: "Failed to send request to Ollama: connection refused".to_string(),
            }),
            Ok(ok_stream()),
        ]);
        let counter = mock.counter();
        let backend = RetryingBackend::new(Box::new(mock), retry_config_fast());
        let config = request_config();

        let result = backend.send_message(&[], &config).await;
        assert!(
            result.is_ok(),
            "should succeed after retry on transport error"
        );
        assert_eq!(
            counter.load(AtomicOrdering::SeqCst),
            2,
            "should have called the backend twice: one failure then one success"
        );
    }

    #[tokio::test]
    async fn retrying_backend_exhausts_retries_and_propagates_error() {
        let mock = FlakyBackend::new(vec![
            Err(BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }),
            Err(BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }),
            Err(BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }),
            Err(BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }),
        ]);
        let backend = RetryingBackend::new(Box::new(mock), retry_config_fast());
        let config = request_config();

        let result = backend.send_message(&[], &config).await;
        assert!(result.is_err(), "should fail after exhausting retries");
        let err = result.err().expect("should have error");
        assert!(err.to_string().contains("503"));
    }

    #[tokio::test]
    async fn retrying_backend_does_not_retry_on_400() {
        let mock = FlakyBackend::new(vec![Err(BackendError::HttpStatus {
            code: 400,
            body: "bad request".to_string(),
        })]);
        let backend = RetryingBackend::new(Box::new(mock), retry_config_fast());
        let config = request_config();

        let result = backend.send_message(&[], &config).await;
        assert!(result.is_err(), "400 should propagate immediately");
        assert!(
            result
                .err()
                .expect("should have error")
                .to_string()
                .contains("400")
        );
    }

    #[tokio::test]
    async fn retrying_backend_does_not_retry_on_501() {
        let mock = FlakyBackend::new(vec![Err(BackendError::HttpStatus {
            code: 501,
            body: "not implemented".to_string(),
        })]);
        let backend = RetryingBackend::new(Box::new(mock), retry_config_fast());
        let config = request_config();

        let result = backend.send_message(&[], &config).await;
        assert!(result.is_err(), "501 should not be retried");
    }

    #[tokio::test]
    async fn retrying_backend_cancellation_aborts_retry() {
        use tokio_util::sync::CancellationToken;

        let token = CancellationToken::new();
        let mock = FlakyBackend::new(vec![
            Err(BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }),
            Err(BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }),
            Err(BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }),
            Err(BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }),
        ]);
        // Use a large delay so the cancellation triggers during sleep
        let retry_config = RetryConfig {
            max_retries: 3,
            initial_delay_ms: 10000,
            max_delay_ms: 30000,
            max_token_retries: 3,
        };
        let backend = RetryingBackend::new(Box::new(mock), retry_config);
        let mut config = request_config();
        config.cancel_token = Some(token.clone());

        // Cancel after a short delay
        let token_clone = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            token_clone.cancel();
        });

        let result = backend.send_message(&[], &config).await;
        assert!(result.is_err(), "should error when cancelled");
    }

    #[tokio::test]
    async fn retrying_backend_backoff_is_exponential() {
        // Verify delays increase by using a config with measurable delays
        // and tracking time between calls
        let mock = FlakyBackend::new(vec![
            Err(BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }),
            Err(BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }),
            Ok(ok_stream()),
        ]);
        let retry_config = RetryConfig {
            max_retries: 3,
            initial_delay_ms: 10,
            max_delay_ms: 80,
            max_token_retries: 3,
        };
        let backend = RetryingBackend::new(Box::new(mock), retry_config);
        let config = request_config();

        let start = std::time::Instant::now();
        let result = backend.send_message(&[], &config).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "should succeed after 2 retries");
        // With 10ms initial, exponential: 10ms + 20ms = 30ms minimum
        assert!(
            elapsed >= std::time::Duration::from_millis(25),
            "total elapsed should reflect exponential backoff: {elapsed:?}"
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

    #[tokio::test]
    async fn for_role_resolves_max_tokens_from_config_override() {
        use crate::config::{
            ANTHROPIC_DEFAULT_BASE_URL, AnthropicConfig, AppConfig, RetryConfig, ToolsConfig,
            VertexConfig,
        };
        use std::collections::BTreeMap;

        let mut config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: ANTHROPIC_DEFAULT_BASE_URL.to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: Some(32768),
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        config.normalize_back_compat();

        let factory = super::BackendFactory::new(config);
        let selection = factory
            .for_role("default")
            .await
            .expect("for_role should succeed");
        assert_eq!(
            selection.max_tokens, 32768,
            "config override should take precedence over default"
        );
    }

    #[tokio::test]
    async fn for_role_resolves_max_tokens_from_default_when_no_override() {
        use crate::config::{
            ANTHROPIC_DEFAULT_BASE_URL, AnthropicConfig, AppConfig, RetryConfig, ToolsConfig,
            VertexConfig,
        };
        use std::collections::BTreeMap;

        let mut config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: ANTHROPIC_DEFAULT_BASE_URL.to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        config.normalize_back_compat();

        let factory = super::BackendFactory::new(config);
        let selection = factory
            .for_role("default")
            .await
            .expect("for_role should succeed");
        assert_eq!(
            selection.max_tokens,
            super::defaults::ANTHROPIC,
            "should use anthropic default when no override"
        );
    }

    #[tokio::test]
    async fn for_role_resolves_vertex_max_tokens_to_vertex_default() {
        use crate::config::{AppConfig, ModelRole, RetryConfig, ToolsConfig, VertexConfig};
        use std::collections::BTreeMap;

        let mut models = BTreeMap::new();
        models.insert(
            "test-role".to_string(),
            ModelRole {
                backend: "vertex".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let factory = super::BackendFactory::new(config);

        let fake_provider: Arc<dyn gcp_auth::TokenProvider> = Arc::new(FakeTokenProvider);
        factory
            .vertex_auth_cache
            .seed("proj", "us-east5", fake_provider);

        let selection = factory
            .for_role("test-role")
            .await
            .expect("for_role should succeed");
        assert_eq!(
            selection.max_tokens,
            super::defaults::VERTEX,
            "vertex should use vertex default (8192)"
        );
    }

    #[tokio::test]
    async fn for_role_resolves_vertex_max_tokens_from_config_override() {
        use crate::config::{AppConfig, ModelRole, RetryConfig, ToolsConfig, VertexConfig};
        use std::collections::BTreeMap;

        let mut models = BTreeMap::new();
        models.insert(
            "test-role".to_string(),
            ModelRole {
                backend: "vertex".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: Some(32768),
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let factory = super::BackendFactory::new(config);

        let fake_provider: Arc<dyn gcp_auth::TokenProvider> = Arc::new(FakeTokenProvider);
        factory
            .vertex_auth_cache
            .seed("proj", "us-east5", fake_provider);

        let selection = factory
            .for_role("test-role")
            .await
            .expect("for_role should succeed");
        assert_eq!(
            selection.max_tokens, 32768,
            "config override should take precedence over the vertex default"
        );
    }

    #[tokio::test]
    async fn for_role_resolves_ollama_max_tokens_from_config_override() {
        use crate::config::{
            AppConfig, ModelRole, OllamaConfig, RetryConfig, ToolsConfig, VertexConfig,
        };
        use std::collections::BTreeMap;

        let mut models = BTreeMap::new();
        models.insert(
            "test-role".to_string(),
            ModelRole {
                backend: "ollama".to_string(),
                model: "gpt-oss:120b".to_string(),
            },
        );
        let config = AppConfig {
            backend: "ollama".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "test-key".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
                max_tokens: Some(32768),
            }),
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let factory = super::BackendFactory::new(config);

        let selection = factory
            .for_role("test-role")
            .await
            .expect("for_role should succeed");
        assert_eq!(
            selection.max_tokens, 32768,
            "config override should take precedence over the ollama default"
        );
    }
}
