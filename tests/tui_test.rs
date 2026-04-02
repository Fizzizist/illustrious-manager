use insta::assert_snapshot;
use ratatui::{Terminal, backend::TestBackend};

use illustrious_manager::frontend::tui::{App, AppState, ConversationEntry, render_app};

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
        role: "You".to_string(),
        content: "Hello!".to_string(),
    });
    app.conversation.push(ConversationEntry {
        role: "Assistant".to_string(),
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
        role: "You".to_string(),
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
