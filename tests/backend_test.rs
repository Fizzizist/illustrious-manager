use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;

use illustrious_manager::backend::LlmBackend;
use illustrious_manager::types::{BoxStream, Message, RequestConfig, Role, StreamEvent};

/// A mock backend implementation to verify the trait compiles correctly
struct MockBackend;

#[async_trait]
impl LlmBackend for MockBackend {
    async fn send_message(
        &self,
        _messages: &[Message],
        _config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        let events = vec![
            Ok(StreamEvent::TextDelta("Hello".to_string())),
            Ok(StreamEvent::Done),
        ];
        let stream = futures::stream::iter(events);
        Ok(Box::pin(stream))
    }
}

#[tokio::test]
async fn test_trait_is_object_safe() {
    // Verify we can create a boxed trait object
    let backend: Box<dyn LlmBackend> = Box::new(MockBackend);

    let messages = vec![Message {
        role: Role::User,
        content: "Test".to_string(),
    }];

    let config = RequestConfig {
        model: "test-model".to_string(),
    };

    let mut stream = backend.send_message(&messages, &config).await.unwrap();

    // Collect events
    let mut events = Vec::new();
    while let Some(result) = stream.next().await {
        events.push(result.unwrap());
    }

    assert_eq!(events.len(), 2);
    match &events[0] {
        StreamEvent::TextDelta(text) => assert_eq!(text, "Hello"),
        _ => panic!("Expected TextDelta"),
    }
    match &events[1] {
        StreamEvent::Done => {}
        _ => panic!("Expected Done"),
    }
}

#[test]
fn test_trait_is_send_sync() {
    // This is a compile-time check. If the trait is not Send + Sync,
    // this function will not compile.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Box<dyn LlmBackend>>();
}
