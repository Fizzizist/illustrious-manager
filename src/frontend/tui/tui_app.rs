use anyhow::Result;
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use futures::channel::mpsc as fmpsc;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use std::io;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::conversation_area::ConversationArea;
use super::input_area::{InputArea, InputMode};
use crate::agent::Agent;
use crate::logging::Logger;
use crate::types::{AgentEvent, ConfirmationResponse};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppState {
    Input,
    Streaming,
    ToolConfirmation {
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationRole {
    User,
    Assistant,
    Error,
    ToolUse,
    ToolResult,
}

pub struct ConversationEntry {
    pub role: ConversationRole,
    pub content: String,
}

pub struct App {
    pub input: InputArea<'static>,
    pub conversation: Vec<ConversationEntry>,
    pub current_response: String,
    pub state: AppState,
    pub confirmation_tx: Option<fmpsc::UnboundedSender<ConfirmationResponse>>,
    pub conversation_area: ConversationArea<'static>,
}

impl App {
    pub fn set_state(&mut self, state: AppState) {
        self.state = state;
        match &self.state {
            AppState::Input => self.input.set_mode(InputMode::Input),
            AppState::Streaming => self.input.set_mode(InputMode::Streaming),
            AppState::ToolConfirmation { name, input } => {
                self.input.set_mode(InputMode::ToolConfirmation {
                    name: name.clone(),
                    input: input.clone(),
                });
            }
        }
    }

    pub fn new() -> Self {
        Self {
            input: InputArea::new(),
            conversation: Vec::new(),
            current_response: String::new(),
            state: AppState::Input,
            confirmation_tx: None,
            conversation_area: ConversationArea::new(),
        }
    }

    pub fn input_text(&self) -> String {
        self.input.text()
    }

    pub fn set_input(&mut self, text: &str) {
        self.input.clear();
        if !text.is_empty() {
            self.input.set_text(text);
        }
    }

    pub fn handle_scroll_key(&mut self, key: &KeyEvent) -> bool {
        match key {
            KeyEvent {
                code: KeyCode::Char('u'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.conversation_area.scroll_up_half();
                true
            }
            KeyEvent {
                code: KeyCode::Char('d'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.conversation_area.scroll_down_half();
                true
            }
            _ => false,
        }
    }

    pub fn sync_conversation_area(&mut self) {
        self.conversation_area
            .update(&self.conversation, &self.current_response);
    }
}

fn tool_use_display_content(name: &str, input: &serde_json::Value) -> String {
    format!(
        "{}\n  {}",
        name,
        serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string())
    )
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

pub fn render_app(app: &App, frame: &mut ratatui::Frame) {
    let input_height = app
        .input
        .height_for_width(frame.area().width, frame.area().height);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(input_height)])
        .split(frame.area());

    app.conversation_area.render(frame, chunks[0]);
    app.input.render(frame, chunks[1]);
}

pub fn handle_agent_event(
    app: &mut App,
    event: AgentEvent,
    logger: Option<&mut Logger>,
) -> Result<()> {
    if let Some(log) = logger {
        log.log_event(&event)?;
        log.flush()?;
    }
    match event {
        AgentEvent::TokenReceived(text) => {
            app.current_response.push_str(&text);
            app.sync_conversation_area();
        }
        AgentEvent::ResponseComplete(full) => {
            app.conversation.push(ConversationEntry {
                role: ConversationRole::Assistant,
                content: full,
            });
            app.current_response.clear();
            app.confirmation_tx = None;
            app.set_state(AppState::Input);
            app.sync_conversation_area();
        }
        AgentEvent::Error(msg) => {
            app.conversation.push(ConversationEntry {
                role: ConversationRole::Error,
                content: msg,
            });
            app.current_response.clear();
            app.confirmation_tx = None;
            app.set_state(AppState::Input);
            app.sync_conversation_area();
        }
        AgentEvent::ToolUseReceived { name, input, .. } => {
            app.current_response.clear();
            app.conversation.push(ConversationEntry {
                role: ConversationRole::ToolUse,
                content: tool_use_display_content(&name, &input),
            });
            app.sync_conversation_area();
        }
        AgentEvent::ToolResult {
            content, is_error, ..
        } => {
            let role = if is_error {
                ConversationRole::Error
            } else {
                ConversationRole::ToolResult
            };
            app.conversation.push(ConversationEntry { role, content });
            app.sync_conversation_area();
        }
        AgentEvent::ToolConfirmationRequired { name, input, .. } => {
            app.current_response.clear();
            app.set_state(AppState::ToolConfirmation { name, input });
        }
        AgentEvent::Usage { .. } => {}
    }
    Ok(())
}

/// Run the TUI REPL. If `initial_prompt` is provided, it's sent immediately.
pub async fn run(
    agent: Arc<Agent>,
    initial_prompt: Option<String>,
    logger: Option<Logger>,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal, agent, initial_prompt, logger).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    agent: Arc<Agent>,
    initial_prompt: Option<String>,
    mut logger: Option<Logger>,
) -> Result<()> {
    let mut app = App::new();
    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
    let mut stream_task: Option<JoinHandle<()>> = None;

    if let Some(prompt) = initial_prompt {
        if let Some(ref mut log) = logger {
            log.log_user_input(&prompt)?;
        }
        app.set_input(&prompt);
        stream_task = Some(submit_message(&mut app, agent.clone(), &event_tx).await?);
    }

    let mut terminal_events = EventStream::new();

    loop {
        terminal.draw(|frame| render_app(&app, frame))?;
        tokio::select! {
            Some(agent_event) = event_rx.recv() => {
                handle_agent_event(&mut app, agent_event, logger.as_mut())?;
            }
            Some(Ok(terminal_event)) = terminal_events.next() => {
                if let Event::Key(key) = terminal_event && !app.handle_scroll_key(&key) {
                    match app.state {
                        AppState::Input => {
                            match key {
                                KeyEvent {
                                    code: KeyCode::Char('c'),
                                    modifiers: KeyModifiers::CONTROL,
                                    ..
                                }
                                | KeyEvent {
                                    code: KeyCode::Esc, ..
                                } => break,
                                KeyEvent {
                                    code: KeyCode::Enter,
                                    modifiers: KeyModifiers::NONE,
                                    ..
                                } => {
                                    let text = app.input_text();
                                    if !text.trim().is_empty() {
                                        if let Some(ref mut log) = logger {
                                            log.log_user_input(&text)?;
                                        }
                                        stream_task =
                                            Some(submit_message(&mut app, agent.clone(), &event_tx).await?);
                                    }
                                }
                                _ => {
                                    app.input.input(key);
                                }
                            }
                        },
                        AppState::ToolConfirmation { .. } => {
                            let response = match key {
                                KeyEvent {
                                    code: KeyCode::Char('y') | KeyCode::Char('Y'),
                                    ..
                                } => Some(ConfirmationResponse::Approved),
                                KeyEvent {
                                    code: KeyCode::Char('n') | KeyCode::Char('N'),
                                    ..
                                } => Some(ConfirmationResponse::Rejected),
                                KeyEvent {
                                    code: KeyCode::Char('c'),
                                    modifiers: KeyModifiers::CONTROL,
                                    ..
                                } => break,
                                _ => None,
                            };
                            if let Some(response) = response {
                                let (name, input) = match &app.state {
                                    AppState::ToolConfirmation { name, input } => (name.clone(), input.clone()),
                                    _ => unreachable!(),
                                };
                                app.conversation.push(ConversationEntry {
                                    role: ConversationRole::ToolUse,
                                    content: tool_use_display_content(&name, &input),
                                });
                                let sent = app
                                    .confirmation_tx
                                    .as_ref()
                                    .is_some_and(|tx| tx.unbounded_send(response).is_ok());
                                if sent {
                                    app.set_state(AppState::Streaming);
                                } else {
                                    app.conversation.push(ConversationEntry {
                                        role: ConversationRole::Error,
                                        content: "Confirmation channel closed unexpectedly.".to_string(),
                                    });
                                    app.confirmation_tx = None;
                                    app.set_state(AppState::Input);
                                }
                                app.sync_conversation_area();
                            }
                        },
                        _ => {
                            if let
                                KeyEvent {
                                    code: KeyCode::Char('c'),
                                    modifiers: KeyModifiers::CONTROL,
                                    ..
                                } = key {
                                    break;
                                }
                        }
                    }
                }
            }
        }
    }

    if let Some(handle) = stream_task {
        handle.abort();
    }

    Ok(())
}

pub async fn submit_message(
    app: &mut App,
    agent: Arc<Agent>,
    event_tx: &mpsc::Sender<AgentEvent>,
) -> Result<JoinHandle<()>> {
    let input = app.input_text();
    app.input.clear();

    app.conversation.push(ConversationEntry {
        role: ConversationRole::User,
        content: input.clone(),
    });

    app.set_state(AppState::Streaming);
    app.sync_conversation_area();

    let (confirm_tx, confirm_rx) = fmpsc::unbounded::<ConfirmationResponse>();
    app.confirmation_tx = Some(confirm_tx);

    let tx = event_tx.clone();

    let handle = tokio::spawn(async move {
        match agent.send(input, Some(confirm_rx)).await {
            Ok(mut stream) => {
                while let Some(event) = stream.next().await {
                    if tx.send(event).await.is_err() {
                        break;
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(AgentEvent::Error(e.to_string())).await;
            }
        }
    });

    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_with_content() -> App {
        let mut app = App::new();
        for i in 0..40 {
            app.conversation.push(ConversationEntry {
                role: ConversationRole::User,
                content: format!("line {i}"),
            });
        }
        app.sync_conversation_area();
        app
    }

    #[test]
    fn new_app_conversation_is_empty() {
        let app = App::new();
        assert!(app.conversation.is_empty());
    }

    #[test]
    fn handle_scroll_key_ctrl_u_returns_true() {
        let mut app = app_with_content();
        let key = KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL);
        assert!(app.handle_scroll_key(&key));
    }

    #[test]
    fn handle_scroll_key_ctrl_d_returns_true() {
        let mut app = app_with_content();
        let key = KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL);
        assert!(app.handle_scroll_key(&key));
    }

    #[test]
    fn handle_scroll_key_other_returns_false() {
        let mut app = App::new();
        let key = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);
        assert!(!app.handle_scroll_key(&key));
    }

    #[tokio::test]
    async fn submit_message_adds_user_entry_and_clears_input() {
        use crate::agent::Agent;
        use crate::backend::LlmBackend;
        use crate::types::*;
        use async_trait::async_trait;
        use std::sync::Arc;
        use tokio::sync::mpsc;

        struct StubBackend;

        #[async_trait]
        impl LlmBackend for StubBackend {
            async fn send_message(
                &self,
                _messages: &[Message],
                _config: &RequestConfig,
            ) -> anyhow::Result<BoxStream<anyhow::Result<StreamEvent>>> {
                Ok(Box::pin(futures::stream::empty()))
            }
        }

        let agent = Arc::new(Agent::new(
            Box::new(StubBackend),
            RequestConfig {
                model: "test".to_string(),
                max_tokens: 1024,
                tools: vec![],
            },
        ));

        let mut app = App::new();
        app.set_input("hello");

        let (tx, _rx) = mpsc::channel(10);
        submit_message(&mut app, agent, &tx)
            .await
            .expect("submit must succeed");

        assert!(app.input.is_empty(), "input should be cleared after submit");
        assert_eq!(app.conversation.len(), 1);
        assert_eq!(app.conversation[0].content, "hello");
        assert_eq!(app.state, AppState::Streaming);
    }

    #[test]
    fn handle_agent_event_logs_tool_use_to_logger() {
        use crate::logging::Logger;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("temp dir");
        let log_path = temp_dir.path().join("test.log");
        let mut logger = Logger::new(Some(log_path.clone())).expect("logger");
        let mut app = App::new();

        let event = AgentEvent::ToolUseReceived {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
        };

        handle_agent_event(&mut app, event, Some(&mut logger)).expect("handle event");
        drop(logger);

        let content = std::fs::read_to_string(&log_path).expect("read log");
        assert!(content.contains("[TOOL CALL]"), "should log tool call");
        assert!(content.contains("name: bash"), "should log tool name");
    }

    #[test]
    fn handle_agent_event_logs_response_complete_to_logger() {
        use crate::logging::Logger;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("temp dir");
        let log_path = temp_dir.path().join("test.log");
        let mut logger = Logger::new(Some(log_path.clone())).expect("logger");
        let mut app = App::new();

        let event = AgentEvent::ResponseComplete("hello world".to_string());

        handle_agent_event(&mut app, event, Some(&mut logger)).expect("handle event");
        drop(logger);

        let content = std::fs::read_to_string(&log_path).expect("read log");
        assert!(
            content.contains("[ASSISTANT RESPONSE]"),
            "should log response"
        );
        assert!(
            content.contains("hello world"),
            "should log response content"
        );
    }

    #[test]
    fn handle_agent_event_with_no_logger_does_not_panic() {
        let mut app = App::new();
        let event = AgentEvent::ResponseComplete("test".to_string());
        handle_agent_event(&mut app, event, None).expect("should not error without logger");
    }

    #[test]
    fn handle_agent_event_token_received_appends_to_current_response() {
        let mut app = App::new();
        handle_agent_event(
            &mut app,
            AgentEvent::TokenReceived("hello".to_string()),
            None,
        )
        .expect("ok");
        handle_agent_event(
            &mut app,
            AgentEvent::TokenReceived(" world".to_string()),
            None,
        )
        .expect("ok");
        assert_eq!(app.current_response, "hello world");
    }

    #[test]
    fn handle_agent_event_response_complete_moves_to_conversation() {
        let mut app = App::new();
        app.current_response = "partial".to_string();
        handle_agent_event(
            &mut app,
            AgentEvent::ResponseComplete("full response".to_string()),
            None,
        )
        .expect("ok");
        assert!(app.current_response.is_empty());
        assert_eq!(app.conversation.len(), 1);
        assert_eq!(app.conversation[0].content, "full response");
        assert!(matches!(
            app.conversation[0].role,
            ConversationRole::Assistant
        ));
    }
}
