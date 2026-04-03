pub mod vertex;
pub mod zai;

use anyhow::Result;
use async_trait::async_trait;

use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>>;
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
