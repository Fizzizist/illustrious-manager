use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use futures::channel::mpsc as fmpsc;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::borrow::Cow;
use std::io;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::agent::Agent;
use crate::types::{AgentEvent, ConfirmationResponse};
use std::sync::Arc;

const TOOL_RESULT_TRUNCATE_CHARS: usize = 200;

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

impl ConversationRole {
    fn display_label(&self) -> &'static str {
        match self {
            ConversationRole::User => "You",
            ConversationRole::Assistant => "Assistant",
            ConversationRole::Error => "Error",
            ConversationRole::ToolUse => "[Tool]",
            ConversationRole::ToolResult => "[Result]",
        }
    }

    fn color(&self) -> Color {
        match self {
            ConversationRole::User => Color::Green,
            ConversationRole::Assistant => Color::Blue,
            ConversationRole::Error => Color::Red,
            ConversationRole::ToolUse => Color::Cyan,
            ConversationRole::ToolResult => Color::Yellow,
        }
    }
}

pub struct ConversationEntry {
    pub role: ConversationRole,
    pub content: String,
}

pub struct App {
    pub input: String,
    pub conversation: Vec<ConversationEntry>,
    pub current_response: String,
    pub state: AppState,
    pub confirmation_tx: Option<fmpsc::UnboundedSender<ConfirmationResponse>>,
    pub scroll_offset: u16,
    pub viewport_height: u16,
}

impl App {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            conversation: Vec::new(),
            current_response: String::new(),
            state: AppState::Input,
            confirmation_tx: None,
            scroll_offset: 0,
            viewport_height: 0,
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
        let total = self.conversation_lines().len() as u16;
        total.saturating_sub(self.viewport_height)
    }

    fn conversation_lines(&self) -> Vec<Line<'_>> {
        let mut lines = Vec::new();
        for entry in &self.conversation {
            lines.push(Line::from(Span::styled(
                format!("{}:", entry.role.display_label()),
                Style::default().fg(entry.role.color()),
            )));
            let display_content = maybe_truncate(&entry.content, entry.role);
            for line in display_content.lines() {
                lines.push(Line::from(format!("  {line}")));
            }
            lines.push(Line::from(""));
        }

        if !self.current_response.is_empty() {
            lines.push(Line::from(Span::styled(
                "Assistant:",
                Style::default().fg(Color::Blue),
            )));
            for line in self.current_response.lines() {
                lines.push(Line::from(format!("  {line}")));
            }
        }

        lines
    }
}

fn maybe_truncate(content: &str, role: ConversationRole) -> Cow<'_, str> {
    if role != ConversationRole::ToolResult {
        return Cow::Borrowed(content);
    }
    let mut chars = content.chars();
    let head: String = (&mut chars).take(TOOL_RESULT_TRUNCATE_CHARS).collect();
    if chars.next().is_some() {
        Cow::Owned(format!("{head}...[truncated]"))
    } else {
        Cow::Borrowed(content)
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

/// Render the app to a frame. Includes scroll and cursor positioning.
pub fn render_app(app: &App, frame: &mut ratatui::Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(frame.area());

    let visible_height = chunks[0].height.saturating_sub(2);
    let conv_lines = app.conversation_lines();
    let total_lines = conv_lines.len() as u16;
    let auto_scroll = total_lines.saturating_sub(visible_height);
    let scroll_row = auto_scroll.saturating_sub(app.scroll_offset.min(auto_scroll));

    let conversation = Paragraph::new(conv_lines)
        .block(Block::default().borders(Borders::ALL).title("Conversation"))
        .wrap(Wrap { trim: false })
        .scroll((scroll_row, 0));
    frame.render_widget(conversation, chunks[0]);

    let input_title = match &app.state {
        AppState::Input => "Input (Enter to send, Ctrl+C to quit)".to_string(),
        AppState::Streaming => "Streaming...".to_string(),
        AppState::ToolConfirmation { name, .. } => format!("Allow '{name}'? [y/n]"),
    };

    let input_content = match &app.state {
        AppState::ToolConfirmation { name, input } => {
            format!("Allow '{}' with input {}?", name, input)
        }
        _ => app.input.clone(),
    };

    let input = Paragraph::new(input_content.as_str())
        .block(Block::default().borders(Borders::ALL).title(input_title));
    frame.render_widget(input, chunks[1]);

    if matches!(app.state, AppState::Input) {
        let cursor_x = chunks[1].x + u16::try_from(app.input.len()).unwrap_or(u16::MAX) + 1;
        frame.set_cursor_position((cursor_x, chunks[1].y + 1));
    }
}

/// Run the TUI REPL. If `initial_prompt` is provided, it's sent immediately.
pub async fn run(agent: Arc<Agent>, initial_prompt: Option<String>) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal, agent, initial_prompt).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    agent: Arc<Agent>,
    initial_prompt: Option<String>,
) -> Result<()> {
    let mut app = App::new();
    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
    let mut stream_task: Option<JoinHandle<()>> = None;

    if let Some(prompt) = initial_prompt {
        app.input = prompt;
        stream_task = Some(submit_message(&mut app, agent.clone(), &event_tx).await?);
    }

    loop {
        app.viewport_height = terminal.size()?.height.saturating_sub(5);
        terminal.draw(|frame| render_app(&app, frame))?;

        if matches!(app.state, AppState::Input) {
            if event::poll(std::time::Duration::from_millis(50))?
                && let Event::Key(key) = event::read()?
                && !app.handle_scroll_key(&key)
            {
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
                        ..
                    } => {
                        if !app.input.trim().is_empty() {
                            stream_task =
                                Some(submit_message(&mut app, agent.clone(), &event_tx).await?);
                        }
                    }
                    KeyEvent {
                        code: KeyCode::Char(c),
                        ..
                    } => {
                        app.input.push(c);
                    }
                    KeyEvent {
                        code: KeyCode::Backspace,
                        ..
                    } => {
                        app.input.pop();
                    }
                    _ => {}
                }
            }
        } else if matches!(app.state, AppState::ToolConfirmation { .. }) {
            if event::poll(std::time::Duration::from_millis(50))?
                && let Event::Key(key) = event::read()?
            {
                let response = if app.handle_scroll_key(&key) {
                    None
                } else {
                    match key {
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
                    }
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
                        app.state = AppState::Streaming;
                    } else {
                        app.conversation.push(ConversationEntry {
                            role: ConversationRole::Error,
                            content: "Confirmation channel closed unexpectedly.".to_string(),
                        });
                        app.confirmation_tx = None;
                        app.state = AppState::Input;
                    }
                }
            }
        } else {
            tokio::select! {
                Some(agent_event) = event_rx.recv() => {
                    match agent_event {
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
                            app.state = AppState::Input;
                            app.scroll_offset = 0;
                        }
                        AgentEvent::Error(msg) => {
                            app.conversation.push(ConversationEntry {
                                role: ConversationRole::Error,
                                content: msg,
                            });
                            app.current_response.clear();
                            app.confirmation_tx = None;
                            app.state = AppState::Input;
                            app.scroll_offset = 0;
                        }
                        AgentEvent::ToolUseReceived { name, input, .. } => {
                            app.current_response.clear();
                            app.conversation.push(ConversationEntry {
                                role: ConversationRole::ToolUse,
                                content: tool_use_display_content(&name, &input),
                            });
                            app.scroll_offset = 0;
                        }
                        AgentEvent::ToolResult { content, is_error, .. } => {
                            let role = if is_error {
                                ConversationRole::Error
                            } else {
                                ConversationRole::ToolResult
                            };
                            app.conversation.push(ConversationEntry {
                                role,
                                content,
                            });
                            app.scroll_offset = 0;
                        }
                        AgentEvent::ToolConfirmationRequired { name, input, .. } => {
                            app.current_response.clear();
                            app.state = AppState::ToolConfirmation { name, input };
                        }
                    }
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(16)) => {
                    if event::poll(std::time::Duration::from_millis(0))?
                        && let Event::Key(key) = event::read()?
                        && !app.handle_scroll_key(&key)
                            && let KeyEvent {
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
    let input = app.input.drain(..).collect::<String>();

    app.conversation.push(ConversationEntry {
        role: ConversationRole::User,
        content: input.clone(),
    });

    app.state = AppState::Streaming;

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

    #[test]
    fn maybe_truncate_borrows_short_tool_result() {
        let content = "short output";
        let result = maybe_truncate(content, ConversationRole::ToolResult);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, content);
    }

    #[test]
    fn maybe_truncate_borrows_non_tool_result_roles() {
        let content = "x".repeat(500);
        for role in [
            ConversationRole::User,
            ConversationRole::Assistant,
            ConversationRole::Error,
            ConversationRole::ToolUse,
        ] {
            let result = maybe_truncate(&content, role);
            assert!(
                matches!(result, Cow::Borrowed(_)),
                "expected borrow for {role:?}"
            );
        }
    }

    #[test]
    fn maybe_truncate_handles_multibyte_utf8() {
        let emoji = "🦀".repeat(300);
        let result = maybe_truncate(&emoji, ConversationRole::ToolResult);
        assert!(result.ends_with("...[truncated]"));
        let char_count = result
            .strip_suffix("...[truncated]")
            .unwrap()
            .chars()
            .count();
        assert_eq!(char_count, TOOL_RESULT_TRUNCATE_CHARS);
    }

    #[test]
    fn maybe_truncate_does_not_split_multibyte_char() {
        let content = "é".repeat(300);
        let result = maybe_truncate(&content, ConversationRole::ToolResult);
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
    }
}
