use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

use illustrious_manager::agent::Agent;
use illustrious_manager::backend::LlmBackend;
use illustrious_manager::session::Session;
use illustrious_manager::types::*;

async fn test_session_arc() -> Arc<TokioMutex<Session>> {
    let dir = tempfile::TempDir::new().expect("temp dir");
    Arc::new(TokioMutex::new(
        Session::new(None, dir.keep()).await.expect("test session"),
    ))
}

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
    let agent = Agent::new(Box::new(backend), config, test_session_arc().await).await;

    let mut stream = agent.send("Hi".to_string(), None).await.unwrap();

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
    let agent = Agent::new(Box::new(backend), config, test_session_arc().await).await;

    // First message
    let stream = agent.send("Hello".to_string(), None).await.unwrap();
    let _: Vec<_> = stream.collect().await;

    // Second message
    let stream = agent.send("Again".to_string(), None).await.unwrap();
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
    let agent = Agent::new(Box::new(ErrorBackend), config, test_session_arc().await).await;

    let mut stream = agent.send("Hi".to_string(), None).await.unwrap();

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

#[tokio::test]
async fn cleanup_empty_session_integration_deletes_db_on_exit_with_no_messages() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let dir_path = dir.keep();

    let session = Session::new(None, dir_path.clone()).await.expect("session");
    let db_path = dir_path.join(format!("{}.db", session.id));
    assert!(db_path.exists());

    let config = RequestConfig {
        model: "test".to_string(),
        max_tokens: 100,
        tools: vec![],
    };
    let agent = Agent::new(
        Box::new(MockBackend::new(vec![])),
        config,
        Arc::new(TokioMutex::new(session)),
    )
    .await;

    agent.cleanup_empty_session().await.expect("cleanup");
    assert!(!db_path.exists());
}

#[tokio::test]
async fn cleanup_empty_session_integration_retains_db_when_messages_exist() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let dir_path = dir.keep();

    let session = Session::new(None, dir_path.clone()).await.expect("session");
    session
        .conversation()
        .insert_message(&Message::text(Role::User, "hello".to_string()))
        .await
        .expect("insert");
    let db_path = dir_path.join(format!("{}.db", session.id));
    assert!(db_path.exists());

    let config = RequestConfig {
        model: "test".to_string(),
        max_tokens: 100,
        tools: vec![],
    };
    let agent = Agent::new(
        Box::new(MockBackend::new(vec![])),
        config,
        Arc::new(TokioMutex::new(session)),
    )
    .await;

    agent.cleanup_empty_session().await.expect("cleanup");
    assert!(db_path.exists());
}

#[tokio::test]
async fn agent_checkpoint_session_persists_messages_to_main_db() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let dir_path = dir.keep();

    let session = Session::new(None, dir_path.clone()).await.expect("session");
    let session_id = session.id.clone();
    session
        .conversation()
        .insert_message(&Message::text(Role::User, "checkpoint test".to_string()))
        .await
        .expect("insert");

    let config = RequestConfig {
        model: "test".to_string(),
        max_tokens: 100,
        tools: vec![],
    };
    let agent = Agent::new(
        Box::new(MockBackend::new(vec![])),
        config,
        Arc::new(TokioMutex::new(session)),
    )
    .await;

    agent.checkpoint_session().await.expect("checkpoint");
    drop(agent);

    // Remove WAL/SHM sidecars to prove messages are in the main .db file.
    let _ = std::fs::remove_file(dir_path.join(format!("{session_id}.db-wal")));
    let _ = std::fs::remove_file(dir_path.join(format!("{session_id}.db-shm")));

    let session = Session::new(Some(session_id), dir_path.clone())
        .await
        .expect("reopen");
    let history = session.conversation().load_history().await.expect("load");
    assert_eq!(history.len(), 1);
    match &history[0].content[0] {
        illustrious_manager::types::ContentBlock::Text(t) => {
            assert_eq!(t, "checkpoint test")
        }
        _ => panic!("expected text block"),
    }
}
