use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::io;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::agent::Agent;
use crate::types::AgentEvent;

pub enum AppState {
    Input,
    Streaming,
}

pub enum ConversationRole {
    User,
    Assistant,
    Error,
}

impl ConversationRole {
    fn display_label(&self) -> &'static str {
        match self {
            ConversationRole::User => "You",
            ConversationRole::Assistant => "Assistant",
            ConversationRole::Error => "Error",
        }
    }

    fn color(&self) -> Color {
        match self {
            ConversationRole::User => Color::Green,
            ConversationRole::Assistant => Color::Blue,
            ConversationRole::Error => Color::Red,
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
}

impl App {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            conversation: Vec::new(),
            current_response: String::new(),
            state: AppState::Input,
        }
    }

    fn conversation_lines(&self) -> Vec<Line<'_>> {
        let mut lines = Vec::new();
        for entry in &self.conversation {
            lines.push(Line::from(Span::styled(
                format!("{}:", entry.role.display_label()),
                Style::default().fg(entry.role.color()),
            )));
            for line in entry.content.lines() {
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

    let conv_lines = app.conversation_lines();
    let total_lines = conv_lines.len() as u16;
    let visible_height = chunks[0].height.saturating_sub(2);
    let scroll = total_lines.saturating_sub(visible_height);

    let conversation = Paragraph::new(conv_lines)
        .block(Block::default().borders(Borders::ALL).title("Conversation"))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(conversation, chunks[0]);

    let input_title = match app.state {
        AppState::Input => "Input (Enter to send, Ctrl+C to quit)",
        AppState::Streaming => "Streaming...",
    };
    let input = Paragraph::new(app.input.as_str())
        .block(Block::default().borders(Borders::ALL).title(input_title));
    frame.render_widget(input, chunks[1]);

    if matches!(app.state, AppState::Input) {
        let cursor_x = chunks[1].x + u16::try_from(app.input.len()).unwrap_or(u16::MAX) + 1;
        frame.set_cursor_position((cursor_x, chunks[1].y + 1));
    }
}

/// Run the TUI REPL. If `initial_prompt` is provided, it's sent immediately.
pub async fn run(agent: &mut Agent, initial_prompt: Option<String>) -> Result<()> {
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
    agent: &mut Agent,
    initial_prompt: Option<String>,
) -> Result<()> {
    let mut app = App::new();
    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
    let mut stream_task: Option<JoinHandle<()>> = None;

    if let Some(prompt) = initial_prompt {
        app.input = prompt;
        stream_task = Some(submit_message(&mut app, agent, &event_tx).await?);
    }

    loop {
        terminal.draw(|frame| render_app(&app, frame))?;

        match app.state {
            AppState::Input => {
                if event::poll(std::time::Duration::from_millis(50))?
                    && let Event::Key(key) = event::read()?
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
                                    Some(submit_message(&mut app, agent, &event_tx).await?);
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
            }
            AppState::Streaming => {
                tokio::select! {
                    Some(agent_event) = event_rx.recv() => {
                        match agent_event {
                            AgentEvent::TokenReceived(text) => {
                                app.current_response.push_str(&text);
                            }
                            AgentEvent::ResponseComplete(full) => {
                                app.conversation.push(ConversationEntry {
                                    role: ConversationRole::Assistant,
                                    content: full,
                                });
                                app.current_response.clear();
                                app.state = AppState::Input;
                            }
                            AgentEvent::Error(msg) => {
                                app.conversation.push(ConversationEntry {
                                    role: ConversationRole::Error,
                                    content: msg,
                                });
                                app.current_response.clear();
                                app.state = AppState::Input;
                            }
                        }
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(16)) => {
                        if event::poll(std::time::Duration::from_millis(0))?
                            && let Event::Key(KeyEvent {
                                code: KeyCode::Char('c'),
                                modifiers: KeyModifiers::CONTROL,
                                ..
                            }) = event::read()?
                        {
                            break;
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

async fn submit_message(
    app: &mut App,
    agent: &mut Agent,
    event_tx: &mpsc::Sender<AgentEvent>,
) -> Result<JoinHandle<()>> {
    let input = app.input.drain(..).collect::<String>();

    app.conversation.push(ConversationEntry {
        role: ConversationRole::User,
        content: input.clone(),
    });

    app.state = AppState::Streaming;

    let mut stream = agent.send(input).await?;
    let tx = event_tx.clone();

    let handle = tokio::spawn(async move {
        while let Some(event) = stream.next().await {
            if tx.send(event).await.is_err() {
                break;
            }
        }
    });

    Ok(handle)
}
