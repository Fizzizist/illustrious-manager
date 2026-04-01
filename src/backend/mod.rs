pub mod vertex;

use anyhow::Result;
use async_trait::async_trait;

use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

/// Trait for LLM provider backends.
///
/// Implementations handle authentication, request formatting, and
/// response streaming for a specific LLM provider.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>>;
}
