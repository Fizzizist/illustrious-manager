use insta::assert_snapshot;
use ratatui::{Terminal, backend::TestBackend};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

use illustrious_manager::agent::Agent;
use illustrious_manager::backend::LlmBackend;
use illustrious_manager::frontend::tui::{
    App, AppState, ConversationEntry, ConversationRole, render_app, submit_message,
};
use illustrious_manager::types::*;

/// A mock backend that introduces a delay before streaming responses.
/// Used to test that UI updates immediately without waiting for backend.
struct DelayedBackend {
    delay_ms: u64,
    response: String,
}

#[async_trait::async_trait]
impl LlmBackend for DelayedBackend {
    async fn send_message(
        &self,
        _messages: &[Message],
        _config: &RequestConfig,
    ) -> anyhow::Result<BoxStream<Result<StreamEvent, anyhow::Error>>> {
        tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;

        let events: Vec<Result<StreamEvent, anyhow::Error>> = vec![
            Ok(StreamEvent::TextDelta(self.response.clone())),
            Ok(StreamEvent::Done),
        ];
        let stream = futures::stream::iter(events);
        Ok(Box::pin(stream))
    }
}

#[test]
fn test_tui_initial_state() {
    let app = App::new();
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");

    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");
    assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_with_conversation() {
    let mut app = App::new();
    app.conversation.push(ConversationEntry {
        role: ConversationRole::User,
        content: "Hello!".to_string(),
    });
    app.conversation.push(ConversationEntry {
        role: ConversationRole::Assistant,
        content: "Hi there! How can I help you?".to_string(),
    });

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");

    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");
    assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_streaming_state() {
    let mut app = App::new();
    app.conversation.push(ConversationEntry {
        role: ConversationRole::User,
        content: "Tell me a story".to_string(),
    });
    app.current_response = "Once upon a time".to_string();
    app.state = AppState::Streaming;

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");

    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");
    assert_snapshot!(terminal.backend());
}

/// Regression test for issue #21: Sluggish input submission.
///
/// This test verifies that when a user submits a message, the UI immediately
/// displays the user's message and shows a "thinking" indicator, without waiting
/// for the backend to start streaming the response.
#[tokio::test]
async fn regression_test_issue_21_user_message_appears_immediately() {
    // Create a backend that delays before responding
    let backend = DelayedBackend {
        delay_ms: 500, // 500ms delay
        response: "Hello!".to_string(),
    };

    let config = RequestConfig {
        model: "test-model".to_string(),
        max_tokens: 1024,
    };
    let agent = Arc::new(Agent::new(Box::new(backend), config));

    // Create an app and event channel
    let mut app = App::new();
    let (event_tx, _event_rx) = mpsc::channel::<AgentEvent>(100);

    // Set the input message
    app.input = "Test message".to_string();

    // Call submit_message - this should:
    // 1. Immediately add the user message to the conversation
    // 2. Change state to Streaming
    // 3. Spawn a background task for the backend call
    // 4. Return immediately (NOT wait for the backend)
    let _task = submit_message(&mut app, agent.clone(), &event_tx);

    // IMMEDIATELY after calling submit_message (before backend responds):
    // 1. The user's message should be in the conversation
    // 2. The state should be Streaming (shows "thinking...")
    // 3. The UI should reflect this immediately

    assert_eq!(
        app.conversation.len(),
        1,
        "User message should appear immediately"
    );
    assert_eq!(app.conversation[0].content, "Test message");
    assert_eq!(app.conversation[0].role, ConversationRole::User);
    assert_eq!(
        app.state,
        AppState::Streaming,
        "Should be in streaming state"
    );

    // Render the UI to verify it shows the user's message and "Streaming..." indicator
    let test_backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(test_backend).expect("terminal creation must succeed");

    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");

    // The snapshot should show:
    // - User message in conversation
    // - "Streaming..." in input title (the "thinking" indicator)
    assert_snapshot!(terminal.backend());
}
