use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;

use illustrious_manager::agent::Agent;
use illustrious_manager::backend::LlmBackend;
use illustrious_manager::types::*;

/// A mock backend that returns a fixed sequence of StreamEvents.
struct MockBackend {
    responses: Vec<Vec<StreamEvent>>,
    call_count: std::sync::atomic::AtomicUsize,
}

impl MockBackend {
    fn new(responses: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            responses,
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl LlmBackend for MockBackend {
    async fn send_message(
        &self,
        _messages: &[Message],
        _config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        let idx = self
            .call_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let events = self.responses.get(idx).cloned().unwrap_or_default();
        let stream = futures::stream::iter(events.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }
}

#[tokio::test]
async fn test_agent_single_message() {
    let backend = MockBackend::new(vec![vec![
        StreamEvent::TextDelta("Hello ".to_string()),
        StreamEvent::TextDelta("world!".to_string()),
        StreamEvent::Done,
    ]]);

    let config = RequestConfig {
        model: "test-model".to_string(),
        max_tokens: 1024,
        tools: vec![],
    };
    let mut agent = Agent::new(Box::new(backend), config);

    let mut stream = agent.send("Hi".to_string()).await.unwrap();

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Should get: TokenReceived("Hello "), TokenReceived("world!"), ResponseComplete("Hello world!")
    assert_eq!(events.len(), 3);
    match &events[0] {
        AgentEvent::TokenReceived(t) => assert_eq!(t, "Hello "),
        other => panic!("Expected TokenReceived, got {:?}", other),
    }
    match &events[1] {
        AgentEvent::TokenReceived(t) => assert_eq!(t, "world!"),
        other => panic!("Expected TokenReceived, got {:?}", other),
    }
    match &events[2] {
        AgentEvent::ResponseComplete(full) => assert_eq!(full, "Hello world!"),
        other => panic!("Expected ResponseComplete, got {:?}", other),
    }
}

#[tokio::test]
async fn test_agent_history_accumulates() {
    let backend = MockBackend::new(vec![
        vec![
            StreamEvent::TextDelta("First response".to_string()),
            StreamEvent::Done,
        ],
        vec![
            StreamEvent::TextDelta("Second response".to_string()),
            StreamEvent::Done,
        ],
    ]);

    let config = RequestConfig {
        model: "test-model".to_string(),
        max_tokens: 1024,
        tools: vec![],
    };
    let mut agent = Agent::new(Box::new(backend), config);

    // First message
    let stream = agent.send("Hello".to_string()).await.unwrap();
    let _: Vec<_> = stream.collect().await;

    // Second message
    let stream = agent.send("Again".to_string()).await.unwrap();
    let _: Vec<_> = stream.collect().await;

    // History should have 4 messages: user, assistant, user, assistant
    assert_eq!(agent.history().len(), 4);
}

#[tokio::test]
async fn test_agent_backend_error_emits_error_event() {
    // Backend that returns an error in the stream
    struct ErrorBackend;

    #[async_trait]
    impl LlmBackend for ErrorBackend {
        async fn send_message(
            &self,
            _messages: &[Message],
            _config: &RequestConfig,
        ) -> Result<BoxStream<Result<StreamEvent>>> {
            let stream = futures::stream::iter(vec![
                Ok(StreamEvent::TextDelta("partial".to_string())),
                Err(anyhow::anyhow!("connection lost")),
            ]);
            Ok(Box::pin(stream))
        }
    }

    let config = RequestConfig {
        model: "test-model".to_string(),
        max_tokens: 1024,
        tools: vec![],
    };
    let mut agent = Agent::new(Box::new(ErrorBackend), config);

    let mut stream = agent.send("Hi".to_string()).await.unwrap();

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    assert_eq!(events.len(), 2);
    match &events[0] {
        AgentEvent::TokenReceived(t) => assert_eq!(t, "partial"),
        other => panic!("Expected TokenReceived, got {:?}", other),
    }
    match &events[1] {
        AgentEvent::Error(msg) => assert!(msg.contains("connection lost")),
        other => panic!("Expected Error, got {:?}", other),
    }
}
