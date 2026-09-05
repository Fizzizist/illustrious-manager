use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use tokio::sync::OnceCell;

use super::LlmBackend;
use super::anthropic_compat::{AnthropicCompatBackend, AnthropicCompatConfig};
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

pub use super::anthropic_compat::AnthropicCompatSseParser as VertexSseParser;
pub use super::anthropic_compat::parse_sse_data;

type AuthProvider = Arc<dyn gcp_auth::TokenProvider>;
type AuthCell = Arc<OnceCell<AuthProvider>>;

/// Cache that shares expensive GCP auth state across Vertex AI backends that
/// target the same `(project, region)` pair.
///
/// Each key maps to a `OnceCell` initialised at most once, even under
/// concurrent callers.  This avoids the TOCTOU race of the
/// "read-lock / drop / await / write-lock" pattern.
pub struct VertexAuthCache {
    cells: Mutex<HashMap<(String, String), AuthCell>>,
}

impl Default for VertexAuthCache {
    fn default() -> Self {
        Self::new()
    }
}

impl VertexAuthCache {
    pub fn new() -> Self {
        Self {
            cells: Mutex::new(HashMap::new()),
        }
    }

    pub async fn get_or_init(&self, project: String, region: String) -> Result<AuthProvider> {
        let key = (project, region);
        let cell: AuthCell = {
            let mut cache = self.cells.lock().unwrap_or_else(|e| e.into_inner());
            Arc::clone(
                cache
                    .entry(key)
                    .or_insert_with(|| Arc::new(OnceCell::new())),
            )
        };
        let provider = cell
            .get_or_try_init(|| async {
                gcp_auth::provider().await.map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to initialize GCP authentication. \
                         Run: gcloud auth application-default login\n{e}"
                    )
                })
            })
            .await?;
        Ok(Arc::clone(provider))
    }

    #[cfg(test)]
    pub fn seed(
        &self,
        project: &str,
        region: &str,
        provider: AuthProvider,
    ) -> Arc<OnceCell<AuthProvider>> {
        let key = (project.to_string(), region.to_string());
        let mut cache = self.cells.lock().expect("lock");
        let cell = Arc::clone(
            cache
                .entry(key)
                .or_insert_with(|| Arc::new(OnceCell::new())),
        );
        cell.set(provider)
            .ok()
            .expect("cell should not have been set already");
        cell
    }
}

/// Vertex AI backend for Claude models.
///
/// Partial shim: resolves GCP auth tokens and constructs the Vertex-specific
/// endpoint, then delegates HTTP request sending and SSE parsing to
/// [`AnthropicCompatBackend`].
pub struct VertexBackend {
    client: Client,
    project: String,
    region: String,
    auth_manager: Arc<dyn gcp_auth::TokenProvider>,
}

impl VertexBackend {
    pub async fn new(project: String, region: String) -> Result<Self> {
        let auth_manager = gcp_auth::provider().await.context(
            "Failed to initialize GCP authentication. Run: gcloud auth application-default login",
        )?;
        Ok(Self {
            client: super::build_http_client()?,
            project,
            region,
            auth_manager,
        })
    }

    pub fn with_auth(
        project: String,
        region: String,
        auth_manager: Arc<dyn gcp_auth::TokenProvider>,
    ) -> Result<Self> {
        Ok(Self {
            client: super::build_http_client()?,
            project,
            region,
            auth_manager,
        })
    }

    fn endpoint(&self, model: &str) -> String {
        let host = if self.region == "global" {
            "aiplatform.googleapis.com".to_string()
        } else {
            format!("{}-aiplatform.googleapis.com", self.region)
        };
        format!(
            "https://{host}/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:streamRawPredict",
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

        let endpoint = self.endpoint(&config.model);
        let thinking_enabled = config.thinking.as_ref().is_some_and(|tc| tc.enabled);
        let anthropic_beta = if thinking_enabled {
            Some("interleaved-thinking-2025-05-14".to_string())
        } else {
            None
        };

        let compat_config = AnthropicCompatConfig {
            endpoint,
            auth_token: Some(token_str),
            auth_style: super::anthropic_compat::AuthStyle::Bearer,
            anthropic_version: "vertex-2023-10-16".to_string(),
            include_model_in_body: false,
            anthropic_beta,
            max_tokens_override: None,
        };

        let backend = AnthropicCompatBackend::new(self.client.clone(), compat_config);
        backend.send_message(messages, config).await
    }
}
