use anyhow::Result;
use async_trait::async_trait;

use crate::config::AppConfig;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

pub mod retry;
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

pub async fn from_config(config: &AppConfig) -> Result<BackendSelection> {
    match config.backend.as_str() {
        "vertex" => {
            let backend = vertex::VertexBackend::new(
                config.vertex.project.clone(),
                config.vertex.region.clone(),
            )
            .await?;
            Ok(BackendSelection {
                backend: Box::new(retry::RetryBackend::new(backend)),
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
                backend: Box::new(retry::RetryBackend::new(backend)),
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
}
