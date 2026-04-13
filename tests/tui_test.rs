use insta::assert_snapshot;
use ratatui::{Terminal, backend::TestBackend};

use illustrious_manager::frontend::tui::{
    App, AppState, ConversationEntry, ConversationRole, render_app,
};

#[test]
fn test_tui_tool_call_renders_inline() {
    let mut app = App::new(std::sync::Arc::new(
        illustrious_manager::tools::ToolRegistry::new(),
    ));
    app.conversation.push(ConversationEntry::new(
        ConversationRole::User,
        "Run ls".to_string(),
    ));
    app.conversation.push(ConversationEntry::new(
        ConversationRole::ToolUse,
        "bash\n  {\"command\":\"ls\"}".to_string(),
    ));

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");
    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");
    assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_tool_result_renders_below_invocation() {
    let mut app = App::new(std::sync::Arc::new(
        illustrious_manager::tools::ToolRegistry::new(),
    ));
    app.conversation.push(ConversationEntry::new(
        ConversationRole::ToolUse,
        "bash\n  {\"command\":\"ls\"}".to_string(),
    ));
    app.conversation.push(ConversationEntry::new(
        ConversationRole::ToolResult,
        "file1.txt\nfile2.txt".to_string(),
    ));

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");
    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");
    assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_long_tool_result_is_truncated() {
    let mut app = App::new(std::sync::Arc::new(
        illustrious_manager::tools::ToolRegistry::new(),
    ));
    let long_output = "x".repeat(500);
    app.conversation.push(ConversationEntry::new(
        ConversationRole::ToolResult,
        long_output,
    ));

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");
    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");
    assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_confirmation_prompt_state_renders() {
    let mut app = App::new(std::sync::Arc::new(
        illustrious_manager::tools::ToolRegistry::new(),
    ));
    app.conversation.push(ConversationEntry::new(
        ConversationRole::User,
        "Write a file".to_string(),
    ));
    app.set_state(AppState::ToolConfirmation {
        name: "write_file".to_string(),
        input: serde_json::json!({"path": "/tmp/test.txt", "content": "hello"}),
    });

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");
    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");
    assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_initial_state() {
    let app = App::new(std::sync::Arc::new(
        illustrious_manager::tools::ToolRegistry::new(),
    ));
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");

    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");
    assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_with_conversation() {
    let mut app = App::new(std::sync::Arc::new(
        illustrious_manager::tools::ToolRegistry::new(),
    ));
    app.conversation.push(ConversationEntry::new(
        ConversationRole::User,
        "Hello!".to_string(),
    ));
    app.conversation.push(ConversationEntry::new(
        ConversationRole::Assistant,
        "Hi there! How can I help you?".to_string(),
    ));

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");

    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");
    assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_streaming_state() {
    let mut app = App::new(std::sync::Arc::new(
        illustrious_manager::tools::ToolRegistry::new(),
    ));
    app.conversation.push(ConversationEntry::new(
        ConversationRole::User,
        "Tell me a story".to_string(),
    ));
    app.current_response = "Once upon a time".to_string();
    app.set_state(AppState::Streaming);

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");

    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");
    assert_snapshot!(terminal.backend());
}

#[tokio::test]
async fn user_message_appears_immediately() {
    // This test verifies that submit_message completes quickly without
    // waiting for the backend response.
    //
    // The bug in issue #21 was that submit_message would await the backend
    // response, blocking the UI event loop. The fix is to spawn the backend
    // request in a background task and return immediately.

    use anyhow::Result;
    use async_trait::async_trait;
    use illustrious_manager::agent::Agent;
    use illustrious_manager::backend::LlmBackend;
    use illustrious_manager::session::Session;
    use illustrious_manager::types::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::mpsc;
    use tokio::time::Duration;

    // A mock backend that blocks for a long time before responding
    struct SlowBackend {
        delay_ms: u64,
        call_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl LlmBackend for SlowBackend {
        async fn send_message(
            &self,
            _messages: &[Message],
            _config: &RequestConfig,
        ) -> Result<BoxStream<Result<StreamEvent>>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            // Simulate a slow backend that takes time to start responding
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            let stream = futures::stream::empty::<Result<StreamEvent>>();
            Ok(Box::pin(stream))
        }
    }

    let config = RequestConfig {
        model: "test-model".to_string(),
        max_tokens: 1024,
        tools: vec![],
    };

    // Create a backend that will delay for 5 seconds
    let call_count = Arc::new(AtomicUsize::new(0));
    let backend = Box::new(SlowBackend {
        delay_ms: 5000,
        call_count: call_count.clone(),
    });
    let dir = tempfile::TempDir::new().expect("temp dir");
    let session = Session::new(None, dir.keep()).await.expect("test session");
    let agent = Arc::new(Agent::new(backend, config, session).await);

    // Create app with user input
    let mut app = App::new(std::sync::Arc::new(
        illustrious_manager::tools::ToolRegistry::new(),
    ));
    app.set_input("Hello, world!");

    // Submit the message - this should return QUICKLY (< 100ms) without
    // waiting for the backend, because the backend request is spawned
    // in a background task
    let (event_tx, _) = mpsc::channel(100);

    let start = std::time::Instant::now();
    let _task =
        illustrious_manager::frontend::tui::submit_message(&mut app, agent.clone(), &event_tx)
            .await
            .expect("submit_message must succeed");
    let elapsed = start.elapsed();

    // Verify submit_message completed quickly (should be < 100ms)
    // If it takes longer, the code is awaiting the backend response (BUG)
    assert!(
        elapsed < Duration::from_millis(100),
        "submit_message took {:?}, should complete in < 100ms (backend call is slow)",
        elapsed
    );

    // Verify the user message is in the conversation
    assert_eq!(app.conversation.len(), 1);
    assert_eq!(app.conversation[0].role, ConversationRole::User);
    assert_eq!(app.conversation[0].content, "Hello, world!");

    // Verify the state is Streaming
    assert_eq!(app.state, AppState::Streaming);

    // Verify input was cleared
    assert!(app.input_text().is_empty());

    // Give the background task a moment to start and call the backend
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Verify the backend call was actually made (background task started)
    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

#[test]
fn test_tui_auto_scroll_shows_bottom_with_wrapping_content() {
    let mut app = App::new(std::sync::Arc::new(
        illustrious_manager::tools::ToolRegistry::new(),
    ));
    // Each assistant response is 100 chars, which wraps at 78 chars (80 wide - 2 borders)
    let long_response = "x".repeat(100);
    for i in 0..10 {
        app.conversation.push(ConversationEntry::new(
            ConversationRole::User,
            format!("Message {i}"),
        ));
        app.conversation.push(ConversationEntry::new(
            ConversationRole::Assistant,
            long_response.clone(),
        ));
    }
    // scroll_offset=0 means auto-scroll to bottom — the last entry must be visible
    app.scroll_offset = 0;

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");
    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");

    let rendered = format!("{:?}", terminal.backend());
    assert!(
        rendered.contains("Message 9"),
        "last user message should be visible when auto-scrolled to bottom, got:\n{rendered}"
    );
}

#[test]
fn test_tui_scrolled_up_with_wrapping_content_shows_correct_messages() {
    // Regression test for logical-vs-visual line space mismatch during scroll-up.
    // Each entry wraps across multiple visual lines; scroll_offset is in visual-line
    // space. The viewport must be fully filled with real content, no blank lines.
    let mut app = App::new(std::sync::Arc::new(
        illustrious_manager::tools::ToolRegistry::new(),
    ));
    app.viewport_height = 22;
    app.viewport_width = 78;
    for i in 0..15 {
        app.conversation.push(ConversationEntry::new(
            ConversationRole::Assistant,
            format!(
                "WRAP_MSG_{i} this line is long enough to word-wrap at eighty columns wide terminal"
            ),
        ));
    }
    // Scroll up 10 visual lines from the bottom.
    app.scroll_offset = 10;

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");
    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");

    let buffer = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol().to_string())
        .collect::<String>();
    // With wrapping, the viewport must be fully filled — no two consecutive blank
    // lines (the border + a blank interior row would suggest the window fell short).
    assert!(
        !buffer.contains("│                                                                              │\n│                                                                              │"),
        "two consecutive blank interior lines indicate the viewport was not filled:\n{buffer}"
    );
    // Scrolled up 10 visual lines: the last visible message must not be the very
    // last entry (that would mean scroll-up had no effect).
    assert!(
        !buffer.contains("WRAP_MSG_14"),
        "with scroll_offset=10 the last entry should be scrolled off-screen:\n{buffer}"
    );
}

#[test]
fn test_tui_scrolled_up_shows_earlier_content() {
    let mut app = App::new(std::sync::Arc::new(
        illustrious_manager::tools::ToolRegistry::new(),
    ));
    for i in 0..20 {
        app.conversation.push(ConversationEntry::new(
            ConversationRole::User,
            format!("Message {i}"),
        ));
    }
    app.scroll_offset = 10;

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal creation must succeed");
    terminal
        .draw(|frame| render_app(&app, frame))
        .expect("draw must succeed");
    assert_snapshot!(terminal.backend());
}
