// Tests for loading context files into agent history

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

use illustrious_manager::agent::Agent;
use illustrious_manager::backend::LlmBackend;
use illustrious_manager::context_files::ContextFile;
use illustrious_manager::session::Session;
use illustrious_manager::types::*;

async fn test_session_arc() -> Arc<TokioMutex<Session>> {
    let dir = tempfile::TempDir::new().expect("temp dir");
    Arc::new(TokioMutex::new(
        Session::new(None, dir.keep()).await.expect("test session"),
    ))
}

struct NullBackend;

impl NullBackend {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LlmBackend for NullBackend {
    async fn send_message(
        &self,
        _messages: &[Message],
        _config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        let stream = futures::stream::iter(vec![Ok(StreamEvent::Done)]);
        Ok(Box::pin(stream))
    }
}

#[tokio::test]
async fn load_context_files_adds_messages_to_history() {
    let backend = Box::new(NullBackend::new());
    let config = RequestConfig {
        model: "test".to_string(),
        max_tokens: 100,
        tools: vec![],
        thinking: None,
        cancel_token: None,
    };
    let agent = Agent::new(backend, config, test_session_arc().await).await;

    let context_files = vec![
        ContextFile {
            path: "/path/to/CLAUDE.md".into(),
            content: "# Project instructions\nBe helpful.".to_string(),
        },
        ContextFile {
            path: "/path/to/AGENTS.md".into(),
            content: "# Agent config\nUse tools carefully.".to_string(),
        },
    ];

    agent.load_context_files(context_files);

    let history = agent.history();
    assert_eq!(history.len(), 1, "should add one message with all context");

    let msg = &history[0];
    assert_eq!(
        msg.role,
        Role::User,
        "context should be added as user message"
    );
    assert_eq!(msg.content.len(), 1, "should have single content block");

    let text = if let ContentBlock::Text(t) = &msg.content[0] {
        t
    } else {
        panic!("expected Text content block")
    };

    assert!(text.contains("CLAUDE.md"), "should mention CLAUDE.md");
    assert!(
        text.contains("Be helpful"),
        "should include CLAUDE.md content"
    );
    assert!(text.contains("AGENTS.md"), "should mention AGENTS.md");
    assert!(
        text.contains("Use tools carefully"),
        "should include AGENTS.md content"
    );
}

#[tokio::test]
async fn load_context_files_with_empty_vec_does_not_modify_history() {
    let backend = Box::new(NullBackend::new());
    let config = RequestConfig {
        model: "test".to_string(),
        max_tokens: 100,
        tools: vec![],
        thinking: None,
        cancel_token: None,
    };
    let agent = Agent::new(backend, config, test_session_arc().await).await;

    agent.load_context_files(vec![]);

    let history = agent.history();
    assert!(
        history.is_empty(),
        "should not add message for empty context files"
    );
}
