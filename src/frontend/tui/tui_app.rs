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
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::conversation_area::{
    ConversationArea, ConversationEntry, ConversationRole, tool_use_display_content,
};
use super::input_area::{InputArea, InputMode};
use crate::agent::Agent;
use crate::logging::Logger;
use crate::types::{AgentEvent, ConfirmationResponse};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppState {
    Input,
    Streaming,
    ToolConfirmation {
        name: String,
        input: serde_json::Value,
    },
}

pub struct App {
    pub input: InputArea<'static>,
    pub conversation: Vec<ConversationEntry>,
    pub current_response: String,
    pub state: AppState,
    pub confirmation_tx: Option<fmpsc::UnboundedSender<ConfirmationResponse>>,
    pub scroll_offset: u16,
    pub viewport_height: u16,
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
            scroll_offset: 0,
            viewport_height: 0,
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

    pub fn scroll_up(&mut self, amount: u16) {
        let max = self.max_scroll();
        self.scroll_offset = self.scroll_offset.saturating_add(amount).min(max);
    }

    pub fn scroll_down(&mut self, amount: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
    }

    fn half_page(&self) -> u16 {
        (self.viewport_height / 2).max(1)
    }

    pub fn handle_scroll_key(&mut self, key: &KeyEvent) -> bool {
        match key {
            KeyEvent {
                code: KeyCode::Char('u'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                let amount = self.half_page();
                self.scroll_up(amount);
                true
            }
            KeyEvent {
                code: KeyCode::Char('d'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                let amount = self.half_page();
                self.scroll_down(amount);
                true
            }
            _ => false,
        }
    }

    fn max_scroll(&self) -> u16 {
        let text_width = 0;
        let conv_area = ConversationArea::new(
            &self.conversation,
            &self.current_response,
            0,
            self.viewport_height,
        );
        conv_area.max_scroll(text_width)
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

/// Render the app to a frame. Includes scroll and cursor positioning.
pub fn render_app(app: &App, frame: &mut ratatui::Frame) {
    let input_height = app
        .input
        .height_for_width(frame.area().width, frame.area().height);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(input_height)])
        .split(frame.area());

    let text_width = chunks[0].width.saturating_sub(2);
    let conv_area = ConversationArea::new(
        &app.conversation,
        &app.current_response,
        app.scroll_offset,
        chunks[0].height.saturating_sub(2),
    );
    conv_area.render(frame, chunks[0], text_width);

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
            app.scroll_offset = 0;
        }
        AgentEvent::ResponseComplete(full) => {
            app.conversation.push(ConversationEntry {
                role: ConversationRole::Assistant,
                content: full,
            });
            app.current_response.clear();
            app.confirmation_tx = None;
            app.set_state(AppState::Input);
            app.scroll_offset = 0;
        }
        AgentEvent::Error(msg) => {
            app.conversation.push(ConversationEntry {
                role: ConversationRole::Error,
                content: msg,
            });
            app.current_response.clear();
            app.confirmation_tx = None;
            app.set_state(AppState::Input);
            app.scroll_offset = 0;
        }
        AgentEvent::ToolUseReceived { name, input, .. } => {
            if !app.current_response.is_empty() {
                app.conversation.push(ConversationEntry {
                    role: ConversationRole::Assistant,
                    content: std::mem::take(&mut app.current_response),
                });
            }
            app.conversation.push(ConversationEntry {
                role: ConversationRole::ToolUse,
                content: tool_use_display_content(&name, &input),
            });
            app.scroll_offset = 0;
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
            app.scroll_offset = 0;
        }
        AgentEvent::ToolConfirmationRequired { name, input, .. } => {
            if !app.current_response.is_empty() {
                app.conversation.push(ConversationEntry {
                    role: ConversationRole::Assistant,
                    content: std::mem::take(&mut app.current_response),
                });
            }
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
    session: crate::session::Session,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal, agent, initial_prompt, logger, session).await;

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
    session: crate::session::Session,
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
        app.viewport_height = terminal.size()?.height.saturating_sub(5);
        terminal.draw(|frame| render_app(&app, frame))?;
        tokio::select! {
            Some(agent_event) = event_rx.recv() => {
                let is_response_complete = matches!(&agent_event, AgentEvent::ResponseComplete(_));
                handle_agent_event(&mut app, agent_event, logger.as_mut())?;
                if is_response_complete {
                    agent.save_history_to_session(&session).await?;
                }
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

    app.scroll_offset = 0;
    app.set_state(AppState::Streaming);

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

    #[test]
    fn scroll_offset_starts_at_zero() {
        let app = App::new();
        assert_eq!(app.scroll_offset, 0);
    }

    fn app_with_content(viewport_height: u16) -> App {
        let mut app = App::new();
        app.viewport_height = viewport_height;
        for i in 0..40 {
            app.conversation.push(ConversationEntry {
                role: ConversationRole::User,
                content: format!("line {i}"),
            });
        }
        app
    }

    #[test]
    fn scroll_up_increases_offset() {
        let mut app = app_with_content(10);
        app.scroll_up(10);
        assert_eq!(app.scroll_offset, 10);
        app.scroll_up(5);
        assert_eq!(app.scroll_offset, 15);
    }

    #[test]
    fn scroll_up_clamps_at_max_scroll() {
        let mut app = app_with_content(10);
        let max = app.max_scroll();
        app.scroll_up(max + 50);
        assert_eq!(app.scroll_offset, max);
    }

    #[test]
    fn scroll_down_decreases_offset() {
        let mut app = app_with_content(10);
        app.scroll_offset = 20;
        app.scroll_down(10);
        assert_eq!(app.scroll_offset, 10);
    }

    #[test]
    fn scroll_down_does_not_underflow() {
        let mut app = App::new();
        app.scroll_offset = 5;
        app.scroll_down(20);
        assert_eq!(app.scroll_offset, 0);
    }

    #[tokio::test]
    async fn submit_message_resets_scroll_offset() {
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

        let mut app = app_with_content(10);
        app.scroll_offset = 15;
        app.set_input("hello");

        let (tx, _rx) = mpsc::channel(10);
        submit_message(&mut app, agent, &tx)
            .await
            .expect("submit must succeed");

        assert_eq!(app.scroll_offset, 0);
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
    fn tool_use_received_preserves_accumulated_text_as_assistant_entry() {
        let mut app = App::new();
        app.current_response = "Let me look into that.".to_string();

        let event = AgentEvent::ToolUseReceived {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
        };
        handle_agent_event(&mut app, event, None).expect("handle event");

        let assistant_entries: Vec<_> = app
            .conversation
            .iter()
            .filter(|e| e.role == ConversationRole::Assistant)
            .collect();
        assert_eq!(
            assistant_entries.len(),
            1,
            "should have saved assistant text"
        );
        assert_eq!(
            assistant_entries[0].content, "Let me look into that.",
            "saved text should match streamed text"
        );
        assert!(
            app.current_response.is_empty(),
            "current_response should be cleared after saving"
        );

        let tool_entries: Vec<_> = app
            .conversation
            .iter()
            .filter(|e| e.role == ConversationRole::ToolUse)
            .collect();
        assert_eq!(tool_entries.len(), 1, "should also have the tool use entry");
    }

    #[test]
    fn tool_use_received_with_no_accumulated_text_does_not_add_empty_assistant() {
        let mut app = App::new();

        let event = AgentEvent::ToolUseReceived {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
        };
        handle_agent_event(&mut app, event, None).expect("handle event");

        let assistant_entries: Vec<_> = app
            .conversation
            .iter()
            .filter(|e| e.role == ConversationRole::Assistant)
            .collect();
        assert!(
            assistant_entries.is_empty(),
            "should not add empty assistant entry"
        );
    }

    #[test]
    fn tool_confirmation_required_preserves_accumulated_text_as_assistant_entry() {
        let mut app = App::new();
        app.current_response = "I need to edit the file.".to_string();

        let event = AgentEvent::ToolConfirmationRequired {
            id: "t1".to_string(),
            name: "edit_file".to_string(),
            input: serde_json::json!({"path": "/tmp/test.txt"}),
        };
        handle_agent_event(&mut app, event, None).expect("handle event");

        let assistant_entries: Vec<_> = app
            .conversation
            .iter()
            .filter(|e| e.role == ConversationRole::Assistant)
            .collect();
        assert_eq!(
            assistant_entries.len(),
            1,
            "should have saved assistant text"
        );
        assert_eq!(
            assistant_entries[0].content, "I need to edit the file.",
            "saved text should match streamed text"
        );
        assert!(
            app.current_response.is_empty(),
            "current_response should be cleared after saving"
        );
    }
}
