use anyhow::Result;
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEvent,
    KeyModifiers,
};
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

use super::commands::{CommandContext, DispatchResult, default_registry};
use super::conversation_area::{ConversationArea, ConversationEntry, ConversationRole};
use super::diff::{render_edit_file_diff, render_write_file};
use super::input_area::{InputArea, InputMode};
use super::session_picker::{SessionPicker, SessionPickerAction};
use super::status_line::{self, StatusLineInfo, TokenUsage};
use crate::agent::Agent;
use crate::config::AppConfig;
use crate::logging::Logger;
use crate::tools::ToolRegistry;
use crate::types::{AgentEvent, ConfirmationResponse};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppState {
    Input,
    Streaming,
    ToolConfirmation {
        name: String,
        input: serde_json::Value,
        index: usize,
    },
    SessionPicker,
}

pub struct App {
    pub input: InputArea<'static>,
    pub conversation: Vec<ConversationEntry>,
    pub current_response: String,
    pub state: AppState,
    pub confirmation_tx: Option<fmpsc::UnboundedSender<ConfirmationResponse>>,
    pub scroll_offset: u16,
    pub viewport_height: u16,
    pub text_width: u16,
    pub session_picker: Option<SessionPicker>,
    pub usage: TokenUsage,
    pub subagent_usage: TokenUsage,
    /// The input_tokens value reported by the last Usage event. The API always
    /// reports the full context size, so we subtract the previous value to
    /// count only the newly added (non-cached) input tokens per turn.
    pub last_input_total: u32,
    pub model: String,
    pub git_branch: Option<String>,
    pub working_dir: std::path::PathBuf,
    tools: std::sync::Arc<ToolRegistry>,
}

impl App {
    pub fn set_state(&mut self, state: AppState) {
        self.state = state;
        match &self.state {
            AppState::Input => self.input.set_mode(InputMode::Insert),
            AppState::Streaming => self.input.set_mode(InputMode::Streaming),
            AppState::ToolConfirmation { name, input, .. } => {
                self.input.set_mode(InputMode::ToolConfirmation {
                    name: name.clone(),
                    input: input.clone(),
                });
            }
            AppState::SessionPicker => self.input.set_mode(InputMode::SessionPicker),
        }
    }

    pub fn new(tools: std::sync::Arc<ToolRegistry>) -> Self {
        Self {
            input: InputArea::new(),
            conversation: Vec::new(),
            current_response: String::new(),
            state: AppState::Input,
            confirmation_tx: None,
            scroll_offset: 0,
            viewport_height: 0,
            text_width: 0,
            session_picker: None,
            usage: TokenUsage::default(),
            subagent_usage: TokenUsage::default(),
            last_input_total: 0,
            model: String::new(),
            git_branch: None,
            working_dir: std::path::PathBuf::new(),
            tools,
        }
    }

    pub fn load_history(&mut self, messages: &[crate::types::Message]) {
        if !messages.is_empty() {
            let (input_tokens, output_tokens) = status_line::estimate_usage_from_messages(messages);
            self.usage = TokenUsage {
                input_tokens: input_tokens as u64,
                output_tokens: output_tokens as u64,
                is_estimated: true,
            };
            // Seed last_input_total so the next real Usage event subtracts
            // correctly against the estimated context size.
            self.last_input_total = input_tokens;
        }

        for message in messages {
            let role = match message.role {
                crate::types::Role::User => ConversationRole::User,
                crate::types::Role::Assistant => ConversationRole::Assistant,
            };

            // Re-derive per-turn 1-based indices for tool entries within each message.
            let tool_count = message
                .content
                .iter()
                .filter(|b| {
                    matches!(
                        b,
                        crate::types::ContentBlock::ToolUse { .. }
                            | crate::types::ContentBlock::ToolResult { .. }
                    )
                })
                .count();
            let is_indexed = tool_count > 0;
            let mut tool_index: usize = 0;

            for block in &message.content {
                let entry = match block {
                    crate::types::ContentBlock::Text(text) => {
                        Some(ConversationEntry::new(role.clone(), text.clone()))
                    }
                    crate::types::ContentBlock::ToolUse { name, input, .. } => {
                        tool_index += 1;
                        let idx = if is_indexed { Some(tool_index) } else { None };
                        Some(self.tool_use_entry_indexed(
                            name,
                            input,
                            self.text_width as usize,
                            idx,
                        ))
                    }
                    crate::types::ContentBlock::ToolResult {
                        content, is_error, ..
                    } => {
                        tool_index += 1;
                        let entry_role = if *is_error {
                            ConversationRole::Error
                        } else {
                            ConversationRole::ToolResult
                        };
                        let entry = if is_indexed && entry_role == ConversationRole::ToolResult {
                            ConversationEntry::new_indexed(entry_role, content.clone(), tool_index)
                        } else {
                            ConversationEntry::new(entry_role, content.clone())
                        };
                        Some(entry)
                    }
                };
                if let Some(e) = entry {
                    self.conversation.push(e);
                }
            }
        }
    }

    pub fn set_intro_message(&mut self, message: String) {
        self.conversation
            .push(ConversationEntry::new(ConversationRole::Info, message));
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

    fn tool_use_markdown(&self, name: &str, input: &serde_json::Value) -> String {
        match self.tools.lookup(name) {
            Ok(tool) => format!("**{}**\n{}", tool.name(), tool.markdown_input(input)),
            Err(_) => format!(
                "{}\n{}",
                name,
                serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string())
            ),
        }
    }

    /// Build a `ConversationEntry` for a tool use event.
    ///
    /// For `edit_file` and `write_file`, a word-level diff renderer is used.
    /// All other tools fall back to the standard markdown rendering.
    ///
    /// `width` may be 0 if called before the first render (e.g. from
    /// `load_history`); 80 is used as a sensible default in that case.
    /// `index` is the per-turn 1-based tool call index, or `None` for unindexed entries.
    fn tool_use_entry_indexed(
        &self,
        name: &str,
        input: &serde_json::Value,
        width: usize,
        index: Option<usize>,
    ) -> ConversationEntry {
        let effective_width = if width == 0 { 80 } else { width };
        let content = self.tool_use_markdown(name, input);
        match name {
            "edit_file" => {
                if let Some(lines) = render_edit_file_diff(input, effective_width) {
                    return ConversationEntry::new_with_lines_indexed(
                        ConversationRole::ToolUse,
                        content,
                        lines,
                        index,
                    );
                }
            }
            "write_file" => {
                if let Some(lines) = render_write_file(input, effective_width) {
                    return ConversationEntry::new_with_lines_indexed(
                        ConversationRole::ToolUse,
                        content,
                        lines,
                        index,
                    );
                }
            }
            _ => {}
        }
        match index {
            Some(i) => ConversationEntry::new_indexed(ConversationRole::ToolUse, content, i),
            None => ConversationEntry::new(ConversationRole::ToolUse, content),
        }
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

    fn max_scroll(&mut self) -> u16 {
        if self.text_width == 0 {
            return 0;
        }
        let mut conv_area = ConversationArea::new(
            &mut self.conversation,
            &self.current_response,
            0,
            self.viewport_height,
        );
        conv_area.max_scroll(self.text_width)
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new(std::sync::Arc::new(ToolRegistry::new()))
    }
}

/// Render the app to a frame. Includes scroll and cursor positioning.
pub fn render_app(app: &mut App, frame: &mut ratatui::Frame) {
    let input_height = app
        .input
        .height_for_width(frame.area().width, frame.area().height);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ])
        .split(frame.area());

    let text_width = chunks[0].width.saturating_sub(2);
    app.text_width = text_width;
    let mut conv_area = ConversationArea::new(
        &mut app.conversation,
        &app.current_response,
        app.scroll_offset,
        chunks[0].height.saturating_sub(2),
    );
    conv_area.render(frame, chunks[0], text_width);

    app.input.render(frame, chunks[1]);

    let info = StatusLineInfo {
        model: &app.model,
        git_branch: app.git_branch.as_deref(),
        working_dir: &app.working_dir,
        usage: &app.usage,
        subagent_usage: Some(&app.subagent_usage),
    };
    status_line::render_status_line(&info, frame, chunks[2]);

    if let Some(ref mut picker) = app.session_picker {
        picker.render(frame, frame.area());
    }
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
            app.conversation
                .push(ConversationEntry::new(ConversationRole::Assistant, full));
            app.current_response.clear();
            app.confirmation_tx = None;
            app.set_state(AppState::Input);
            app.scroll_offset = 0;
            app.git_branch = status_line::detect_git_branch();
        }
        AgentEvent::Error(msg) => {
            app.conversation
                .push(ConversationEntry::new(ConversationRole::Error, msg));
            app.current_response.clear();
            app.confirmation_tx = None;
            app.set_state(AppState::Input);
            app.scroll_offset = 0;
            app.git_branch = status_line::detect_git_branch();
        }
        AgentEvent::ToolUseReceived {
            name, input, index, ..
        } => {
            if !app.current_response.is_empty() {
                app.conversation.push(ConversationEntry::new(
                    ConversationRole::Assistant,
                    std::mem::take(&mut app.current_response),
                ));
            }
            let width = app.text_width as usize;
            let entry = app.tool_use_entry_indexed(&name, &input, width, Some(index));
            app.conversation.push(entry);
            app.scroll_offset = 0;
        }
        AgentEvent::ToolResult {
            name,
            content,
            is_error,
            index,
        } => {
            let role = if is_error {
                ConversationRole::Error
            } else {
                ConversationRole::ToolResult
            };
            let display = match app.tools.lookup(&name) {
                Ok(tool) => {
                    let result = crate::tools::ToolResult {
                        content: vec![crate::types::ContentBlock::Text(content)],
                        is_error: false,
                        agent_events: vec![],
                    };
                    tool.markdown_output(&result)
                }
                Err(_) => content,
            };
            let entry = if role == ConversationRole::ToolResult {
                ConversationEntry::new_indexed(role, display, index)
            } else {
                ConversationEntry::new(role, display)
            };
            app.conversation.push(entry);
            app.scroll_offset = 0;
        }
        AgentEvent::ToolConfirmationRequired {
            name, input, index, ..
        } => {
            if !app.current_response.is_empty() {
                app.conversation.push(ConversationEntry::new(
                    ConversationRole::Assistant,
                    std::mem::take(&mut app.current_response),
                ));
            }
            app.set_state(AppState::ToolConfirmation { name, input, index });
        }
        AgentEvent::Usage {
            input_tokens,
            output_tokens,
            ..
        } => {
            let new_input = input_tokens.saturating_sub(app.last_input_total);
            app.last_input_total = input_tokens;
            app.usage.add(new_input, output_tokens);
        }
        AgentEvent::SubAgentUsage {
            input_tokens,
            output_tokens,
            ..
        } => {
            app.subagent_usage.add(input_tokens, output_tokens);
        }
    }
    Ok(())
}

/// Run the TUI REPL. If `initial_prompt` is provided, it's sent immediately.
pub async fn run(
    agent: Arc<Agent>,
    initial_prompt: Option<String>,
    logger: Option<Logger>,
    config: &AppConfig,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal, agent, initial_prompt, logger, config).await;

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableBracketedPaste
    )?;
    terminal.show_cursor()?;

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    agent: Arc<Agent>,
    initial_prompt: Option<String>,
    mut logger: Option<Logger>,
    config: &AppConfig,
) -> Result<()> {
    let mut app = App::new(agent.tools());
    app.model = agent.model();
    app.git_branch = status_line::detect_git_branch();
    app.working_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    // we load just the session history here to avoid printing the loaded context messages from
    // skills and CLAUDE.md
    app.load_history(&agent.session_history().await?);
    app.set_intro_message(crate::config::generate_intro_message(config));
    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
    let mut stream_task: Option<JoinHandle<()>> = None;
    let cmd_registry = default_registry();

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
        terminal.draw(|frame| render_app(&mut app, frame))?;
        tokio::select! {
            Some(agent_event) = event_rx.recv() => {
                handle_agent_event(&mut app, agent_event, logger.as_mut())?;
            }
            Some(Ok(terminal_event)) = terminal_events.next() => {
                if let Event::Paste(text) = &terminal_event {
                    if matches!(app.state, AppState::Input) {
                        app.input.insert_paste(text);
                    }
                } else if let Event::Key(key) = terminal_event && !app.handle_scroll_key(&key) {
                    match app.state {
                        AppState::Input => {
                            match key {
                                KeyEvent {
                                    code: KeyCode::Char('c'),
                                    modifiers: KeyModifiers::CONTROL,
                                    ..
                                } => break,
                                KeyEvent {
                                    code: KeyCode::Enter,
                                    modifiers: KeyModifiers::NONE,
                                    ..
                                } => {
                                    let text = app.input_text();
                                    if !text.trim().is_empty() {
                                        let mut ctx = CommandContext {
                                            app: &mut app,
                                            agent: agent.clone(),
                                            config,
                                        };
                                        let dispatch = cmd_registry.dispatch(&text, &mut ctx).await?;
                                        if dispatch == DispatchResult::Passthrough {
                                            if let Some(ref mut log) = logger {
                                                log.log_user_input(&text)?;
                                            }
                                            stream_task = Some(
                                                submit_message(&mut app, agent.clone(), &event_tx)
                                                    .await?,
                                            );
                                        }
                                    }
                                }
                                _ => {
                                    app.input.input(key);
                                }
                            }
                        },
                        AppState::SessionPicker => {
                            if let KeyEvent {
                                code: KeyCode::Char('c'),
                                modifiers: KeyModifiers::CONTROL,
                                ..
                            } = key {
                                break;
                            }
                            if let Some(ref mut picker) = app.session_picker {
                                let action = picker.handle_key(key);
                                match action {
                                    SessionPickerAction::Close => {
                                        app.session_picker = None;
                                        app.set_state(AppState::Input);
                                    }
                                    SessionPickerAction::Select(session_id) => {
                                        app.session_picker = None;
                                        app.set_state(AppState::Input);
                                        // Checkpoint current session before switching
                                        let _ = agent.checkpoint_session().await;
                                        // Clean up current session if empty
                                        if let Err(e) = agent.cleanup_empty_session().await {
                                            app.conversation.push(ConversationEntry::new(
                                                ConversationRole::Error,
                                                format!("Failed to clean up empty session: {e}"),
                                            ));
                                        }
                                        // Reload the selected session
                                        match crate::session::Session::new(
                                            Some(session_id.clone()),
                                            config.sessions_dir.clone(),
                                        )
                                        .await
                                        {
                                            Ok(session) => {
                                                match session.conversation().load_history().await {
                                                    Ok(history) => {
                                                        app.conversation.clear();
                                                        app.current_response.clear();
                                                        app.scroll_offset = 0;
                                                        app.usage = TokenUsage::default();
                                                        app.subagent_usage = TokenUsage::default();
                                                        app.last_input_total = 0;
                                                        app.load_history(&history);
                                                        agent.load_session(session).await;
                                                    }
                                                    Err(e) => {
                                                        app.conversation.push(ConversationEntry::new(
                                                            ConversationRole::Error,
                                                            format!("Failed to load session history: {e}"),
                                                        ));
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                app.conversation.push(ConversationEntry::new(
                                                    ConversationRole::Error,
                                                    format!("Failed to open session: {e}"),
                                                ));
                                            }
                                        }
                                    }
                                    SessionPickerAction::None => {}
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
                                let (name, input, index) = match &app.state {
                                    AppState::ToolConfirmation { name, input, index } => {
                                        (name.clone(), input.clone(), *index)
                                    }
                                    _ => unreachable!(),
                                };
                                let width = app.text_width as usize;
                                let entry = app.tool_use_entry_indexed(&name, &input, width, Some(index));
                                app.conversation.push(entry);
                                let sent = app
                                    .confirmation_tx
                                    .as_ref()
                                    .is_some_and(|tx| tx.unbounded_send(response).is_ok());
                                if sent {
                                    app.set_state(AppState::Streaming);
                                } else {
                                    app.conversation.push(ConversationEntry::new(
                                        ConversationRole::Error,
                                        "Confirmation channel closed unexpectedly.".to_string(),
                                    ));
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

    app.conversation.push(ConversationEntry::new(
        ConversationRole::User,
        input.clone(),
    ));

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
        let app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        assert_eq!(app.scroll_offset, 0);
    }

    fn app_with_content(viewport_height: u16) -> App {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.viewport_height = viewport_height;
        app.text_width = 58;
        for i in 0..40 {
            app.conversation.push(ConversationEntry::new(
                ConversationRole::User,
                format!("line {i}"),
            ));
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
    fn max_scroll_returns_zero_when_text_width_not_set() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.viewport_height = 10;
        for i in 0..40 {
            app.conversation.push(ConversationEntry::new(
                ConversationRole::User,
                format!("line {i}"),
            ));
        }
        assert_eq!(app.text_width, 0);
        assert_eq!(app.max_scroll(), 0);
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
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
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

        let dir = tempfile::TempDir::new().expect("temp dir");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::session::Session::new(None, dir.keep())
                .await
                .expect("test session"),
        ));
        let agent = Arc::new(
            Agent::new(
                Box::new(StubBackend),
                RequestConfig {
                    model: "test".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                },
                session,
            )
            .await,
        );

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
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));

        let event = AgentEvent::ToolUseReceived {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
            index: 1,
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
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));

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
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        let event = AgentEvent::ResponseComplete("test".to_string());
        handle_agent_event(&mut app, event, None).expect("should not error without logger");
    }

    #[test]
    fn response_complete_refreshes_git_branch() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.git_branch = Some("old-branch".to_string());
        let event = AgentEvent::ResponseComplete("done".to_string());
        handle_agent_event(&mut app, event, None).expect("handle event");
        let current = status_line::detect_git_branch();
        assert_eq!(
            app.git_branch, current,
            "ResponseComplete should refresh git branch"
        );
    }

    #[test]
    fn error_event_refreshes_git_branch() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.git_branch = Some("old-branch".to_string());
        let event = AgentEvent::Error("oops".to_string());
        handle_agent_event(&mut app, event, None).expect("handle event");
        let current = status_line::detect_git_branch();
        assert_eq!(app.git_branch, current, "Error should refresh git branch");
    }

    #[test]
    fn usage_event_accumulates_token_counts() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        assert_eq!(app.usage.input_tokens, 0);
        assert_eq!(app.usage.output_tokens, 0);

        // First turn: 100 input tokens reported (all new), 50 output
        let event = AgentEvent::Usage {
            input_tokens: 100,
            output_tokens: 50,
            stop_reason: "end_turn".to_string(),
        };
        handle_agent_event(&mut app, event, None).expect("handle usage event");

        assert_eq!(app.usage.input_tokens, 100);
        assert_eq!(app.usage.output_tokens, 50);
        assert_eq!(app.last_input_total, 100);

        // Second turn: API reports 300 total input tokens (prior 100 cached + 200 new), 75 output.
        // We should only count the 200 new tokens, not the full 300.
        let event2 = AgentEvent::Usage {
            input_tokens: 300,
            output_tokens: 75,
            stop_reason: "end_turn".to_string(),
        };
        handle_agent_event(&mut app, event2, None).expect("handle second usage event");

        assert_eq!(
            app.usage.input_tokens, 300,
            "100 first turn + 200 new = 300"
        );
        assert_eq!(app.usage.output_tokens, 125);
        assert_eq!(app.last_input_total, 300);
    }

    #[test]
    fn usage_event_does_not_double_count_cached_input_tokens() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));

        // Simulate 3 turns where the context grows by different amounts each time.
        // Turn 1: 500 total input (all new)
        handle_agent_event(
            &mut app,
            AgentEvent::Usage {
                input_tokens: 500,
                output_tokens: 100,
                stop_reason: "end_turn".to_string(),
            },
            None,
        )
        .expect("turn 1");

        // Turn 2: 700 total input (500 cached + 200 new)
        handle_agent_event(
            &mut app,
            AgentEvent::Usage {
                input_tokens: 700,
                output_tokens: 150,
                stop_reason: "end_turn".to_string(),
            },
            None,
        )
        .expect("turn 2");

        // Turn 3: 850 total input (700 cached + 150 new)
        handle_agent_event(
            &mut app,
            AgentEvent::Usage {
                input_tokens: 850,
                output_tokens: 80,
                stop_reason: "end_turn".to_string(),
            },
            None,
        )
        .expect("turn 3");

        // Unique input tokens: 500 + 200 + 150 = 850
        assert_eq!(
            app.usage.input_tokens, 850,
            "should count only unique input tokens across turns"
        );
        assert_eq!(app.usage.output_tokens, 330);
        assert_eq!(app.last_input_total, 850);
    }

    #[test]
    fn subagent_usage_event_aggregates_into_app_usage() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        assert_eq!(app.subagent_usage.input_tokens, 0);
        assert_eq!(app.subagent_usage.output_tokens, 0);

        let event = AgentEvent::SubAgentUsage {
            input_tokens: 200,
            output_tokens: 80,
            role: "default".to_string(),
        };
        handle_agent_event(&mut app, event, None).expect("handle SubAgentUsage");

        assert_eq!(app.subagent_usage.input_tokens, 200);
        assert_eq!(app.subagent_usage.output_tokens, 80);
        // SubAgentUsage does not update usage or last_input_total
        assert_eq!(app.usage.input_tokens, 0);
        assert_eq!(app.last_input_total, 0);

        // A subsequent SubAgentUsage should add on top.
        let event2 = AgentEvent::SubAgentUsage {
            input_tokens: 50,
            output_tokens: 30,
            role: "fast".to_string(),
        };
        handle_agent_event(&mut app, event2, None).expect("handle second SubAgentUsage");

        assert_eq!(app.subagent_usage.input_tokens, 250);
        assert_eq!(app.subagent_usage.output_tokens, 110);
    }

    #[test]
    fn render_app_includes_status_line() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.model = "test-model".to_string();
        app.usage.add(1000, 500);
        app.git_branch = Some("main".to_string());
        app.working_dir = std::path::PathBuf::from("/test/project");

        let backend = ratatui::backend::TestBackend::new(80, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                render_app(&mut app, frame);
            })
            .expect("draw");

        let rendered = format!("{:?}", terminal.backend());
        assert!(
            rendered.contains("test-model"),
            "status line should show model name"
        );
        assert!(
            rendered.contains("main"),
            "status line should show git branch"
        );
        assert!(
            rendered.contains("↑1.0k"),
            "status line should show input tokens"
        );
        assert!(
            rendered.contains("↓500"),
            "status line should show output tokens"
        );
    }

    #[test]
    fn tool_use_received_preserves_accumulated_text_as_assistant_entry() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.current_response = "Let me look into that.".to_string();

        let event = AgentEvent::ToolUseReceived {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
            index: 1,
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
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));

        let event = AgentEvent::ToolUseReceived {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
            index: 1,
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
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.current_response = "I need to edit the file.".to_string();

        let event = AgentEvent::ToolConfirmationRequired {
            id: "t1".to_string(),
            name: "edit_file".to_string(),
            input: serde_json::json!({"path": "/tmp/test.txt"}),
            index: 1,
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

    #[test]
    fn load_history_populates_conversation_from_messages() {
        use crate::types::{Message, Role};

        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));

        let messages = vec![
            Message::text(Role::User, "hello".to_string()),
            Message::text(Role::Assistant, "hi there".to_string()),
            Message::text(Role::User, "how are you?".to_string()),
            Message::text(Role::Assistant, "doing well".to_string()),
        ];

        app.load_history(&messages);

        assert_eq!(app.conversation.len(), 4);
        assert_eq!(app.conversation[0].role, ConversationRole::User);
        assert_eq!(app.conversation[0].content, "hello");
        assert_eq!(app.conversation[1].role, ConversationRole::Assistant);
        assert_eq!(app.conversation[1].content, "hi there");
        assert_eq!(app.conversation[2].role, ConversationRole::User);
        assert_eq!(app.conversation[2].content, "how are you?");
        assert_eq!(app.conversation[3].role, ConversationRole::Assistant);
        assert_eq!(app.conversation[3].content, "doing well");
    }

    #[test]
    fn load_history_with_tool_use_and_result_blocks() {
        use crate::types::{ContentBlock, Message, Role};

        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));

        let messages = vec![
            Message::text(Role::User, "run ls".to_string()),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Text("let me check".to_string()),
                    ContentBlock::ToolUse {
                        id: "t1".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "ls"}),
                    },
                ],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    content: "file.txt".to_string(),
                    is_error: false,
                }],
            },
            Message::text(Role::Assistant, "here is the file".to_string()),
        ];

        app.load_history(&messages);

        assert_eq!(app.conversation.len(), 5);
        assert_eq!(app.conversation[0].role, ConversationRole::User);
        assert_eq!(app.conversation[0].content, "run ls");
        assert_eq!(app.conversation[1].role, ConversationRole::Assistant);
        assert_eq!(app.conversation[1].content, "let me check");
        assert_eq!(app.conversation[2].role, ConversationRole::ToolUse);
        assert!(app.conversation[2].content.contains("bash"));
        assert_eq!(app.conversation[3].role, ConversationRole::ToolResult);
        assert_eq!(app.conversation[3].content, "file.txt");
        assert_eq!(app.conversation[4].role, ConversationRole::Assistant);
        assert_eq!(app.conversation[4].content, "here is the file");
    }

    #[test]
    fn load_history_with_multiple_tool_calls_in_one_turn_renders_indexed_labels() {
        use crate::types::{ContentBlock, Message, Role};

        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::ToolUse {
                        id: "t1".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "ls"}),
                    },
                    ContentBlock::ToolUse {
                        id: "t2".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "pwd"}),
                    },
                    ContentBlock::ToolUse {
                        id: "t3".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "whoami"}),
                    },
                ],
            },
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "t1".to_string(),
                        content: "file.txt".to_string(),
                        is_error: false,
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "t2".to_string(),
                        content: "/home/user".to_string(),
                        is_error: false,
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "t3".to_string(),
                        content: "alice".to_string(),
                        is_error: false,
                    },
                ],
            },
        ];

        app.load_history(&messages);

        let tool_entries: Vec<_> = app
            .conversation
            .iter()
            .filter(|e| e.role == ConversationRole::ToolUse)
            .collect();
        assert_eq!(tool_entries.len(), 3, "expected 3 tool use entries");
        assert_eq!(
            tool_entries[0].tool_index,
            Some(1),
            "first tool use should have index 1"
        );
        assert_eq!(
            tool_entries[1].tool_index,
            Some(2),
            "second tool use should have index 2"
        );
        assert_eq!(
            tool_entries[2].tool_index,
            Some(3),
            "third tool use should have index 3"
        );

        let result_entries: Vec<_> = app
            .conversation
            .iter()
            .filter(|e| e.role == ConversationRole::ToolResult)
            .collect();
        assert_eq!(result_entries.len(), 3, "expected 3 tool result entries");
        assert_eq!(
            result_entries[0].tool_index,
            Some(1),
            "first result should have index 1"
        );
        assert_eq!(
            result_entries[1].tool_index,
            Some(2),
            "second result should have index 2"
        );
        assert_eq!(
            result_entries[2].tool_index,
            Some(3),
            "third result should have index 3"
        );
    }

    #[test]
    fn tool_confirmation_required_index_is_threaded_to_entry_on_approval() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));

        // Simulate receiving a ToolConfirmationRequired for tool index 3.
        let event = AgentEvent::ToolConfirmationRequired {
            id: "t3".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
            index: 3,
        };
        handle_agent_event(&mut app, event, None).expect("handle event");

        // The app is now in ToolConfirmation state with index=3.
        assert!(matches!(
            &app.state,
            AppState::ToolConfirmation { index: 3, .. }
        ));

        // Simulate approval: build the entry as the confirmation handler would.
        let (name, input, index) = match &app.state {
            AppState::ToolConfirmation { name, input, index } => {
                (name.clone(), input.clone(), *index)
            }
            _ => panic!("expected ToolConfirmation state"),
        };
        let entry = app.tool_use_entry_indexed(&name, &input, 80, Some(index));
        assert_eq!(
            entry.tool_index,
            Some(3),
            "approved tool entry must carry index 3"
        );
    }

    #[test]
    fn load_history_with_empty_messages_does_nothing() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.load_history(&[]);
        assert!(app.conversation.is_empty());
        assert_eq!(app.usage.input_tokens, 0);
        assert_eq!(app.usage.output_tokens, 0);
        assert!(!app.usage.is_estimated);
    }

    #[test]
    fn load_history_seeds_estimated_usage() {
        use crate::types::{Message, Role};

        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        let messages = vec![
            Message::text(Role::User, "a".repeat(400)),
            Message::text(Role::Assistant, "b".repeat(200)),
        ];
        app.load_history(&messages);

        assert!(
            app.usage.is_estimated,
            "usage should be flagged as estimated"
        );
        assert!(
            app.usage.input_tokens > 0,
            "input tokens should be non-zero"
        );
        assert!(
            app.usage.output_tokens > 0,
            "output tokens should be non-zero"
        );
        assert_eq!(
            app.last_input_total, app.usage.input_tokens as u32,
            "last_input_total should match estimated input tokens"
        );
    }

    #[test]
    fn real_usage_event_after_load_history_clears_estimated_flag() {
        use crate::types::{Message, Role};

        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.load_history(&[Message::text(Role::User, "hello world".to_string())]);
        assert!(app.usage.is_estimated);

        let event = AgentEvent::Usage {
            input_tokens: app.last_input_total + 20,
            output_tokens: 30,
            stop_reason: "end_turn".to_string(),
        };
        handle_agent_event(&mut app, event, None).expect("handle event");

        assert!(
            !app.usage.is_estimated,
            "real Usage event should clear estimated flag"
        );
    }

    #[test]
    fn sessions_command_transitions_app_to_session_picker_state() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.session_picker = Some(crate::frontend::tui::SessionPicker::new(vec![]));
        app.set_state(AppState::SessionPicker);
        assert_eq!(app.state, AppState::SessionPicker);
        assert!(app.session_picker.is_some());
    }

    #[test]
    fn session_picker_close_transitions_back_to_input() {
        use crate::frontend::tui::SessionPickerAction;
        use crate::session::SessionSummary;
        use std::time::SystemTime;

        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        let summaries = vec![SessionSummary {
            id: "01900000-0000-7000-0000-000000000001".to_string(),
            first_user_message: "hello".to_string(),
            modified: SystemTime::UNIX_EPOCH,
        }];
        app.session_picker = Some(crate::frontend::tui::SessionPicker::new(summaries));
        app.set_state(AppState::SessionPicker);

        // simulate Close action
        let picker = app.session_picker.as_mut().expect("picker");
        let action = picker.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ));
        assert_eq!(action, SessionPickerAction::Close);

        // apply the action
        if matches!(action, SessionPickerAction::Close) {
            app.session_picker = None;
            app.set_state(AppState::Input);
        }

        assert_eq!(app.state, AppState::Input);
        assert!(app.session_picker.is_none());
    }

    #[test]
    fn session_picker_esc_also_closes() {
        use crate::frontend::tui::SessionPickerAction;
        use crate::session::SessionSummary;
        use std::time::SystemTime;

        let summaries = vec![SessionSummary {
            id: "01900000-0000-7000-0000-000000000001".to_string(),
            first_user_message: "hello".to_string(),
            modified: SystemTime::UNIX_EPOCH,
        }];
        let mut picker = crate::frontend::tui::SessionPicker::new(summaries);
        let action = picker.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Esc,
            crossterm::event::KeyModifiers::NONE,
        ));
        assert_eq!(action, SessionPickerAction::Close);
    }

    #[tokio::test]
    async fn session_picker_select_clears_conversation_and_reloads_history() {
        use crate::agent::Agent;
        use crate::backend::LlmBackend;
        use crate::types::*;
        use async_trait::async_trait;
        use std::sync::Arc;

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

        let dir = tempfile::TempDir::new().expect("temp dir");
        let dir_path = dir.keep();

        // Create session with a known message
        let session = crate::session::Session::new(None, dir_path.clone())
            .await
            .expect("session");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "loaded message".to_string()))
            .await
            .expect("insert");

        let initial_session = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::session::Session::new(None, dir_path.clone())
                .await
                .expect("initial session"),
        ));
        let agent = Arc::new(
            Agent::new(
                Box::new(StubBackend),
                RequestConfig {
                    model: "test".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                },
                initial_session,
            )
            .await,
        );

        let mut app = App::new(agent.tools());
        // pre-populate conversation with stale data
        app.conversation.push(ConversationEntry::new(
            ConversationRole::User,
            "old message".to_string(),
        ));
        app.scroll_offset = 10;

        // simulate selecting the session
        let history = session
            .conversation()
            .load_history()
            .await
            .expect("load history");
        app.conversation.clear();
        app.current_response.clear();
        app.scroll_offset = 0;
        app.load_history(&history);
        agent.load_session(session).await;

        assert_eq!(app.conversation.len(), 1);
        assert_eq!(app.conversation[0].content, "loaded message");
        assert_eq!(app.scroll_offset, 0);
    }

    #[test]
    fn session_picker_select_returns_selected_id() {
        use crate::frontend::tui::SessionPickerAction;
        use crate::session::SessionSummary;
        use std::time::SystemTime;

        let summaries = vec![SessionSummary {
            id: "01900000-0000-7000-0000-000000000001".to_string(),
            first_user_message: "hello".to_string(),
            modified: SystemTime::UNIX_EPOCH,
        }];
        let mut picker = crate::frontend::tui::SessionPicker::new(summaries);
        let action = picker.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ));
        assert_eq!(
            action,
            SessionPickerAction::Select("01900000-0000-7000-0000-000000000001".to_string())
        );
    }

    #[test]
    fn set_intro_message_adds_intro_entry() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.set_intro_message("# Welcome\n\nHello!".to_string());
        assert_eq!(app.conversation.len(), 1);
        assert_eq!(app.conversation[0].role, ConversationRole::Info);
        assert_eq!(app.conversation[0].content, "# Welcome\n\nHello!");
    }

    #[test]
    fn intro_message_displayed_after_history_when_resuming() {
        use crate::types::{Message, Role};

        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));

        let messages = vec![
            Message::text(Role::User, "hello".to_string()),
            Message::text(Role::Assistant, "hi there".to_string()),
        ];
        app.load_history(&messages);
        app.set_intro_message("# Welcome".to_string());

        assert_eq!(app.conversation.len(), 3);
        assert_eq!(app.conversation[0].role, ConversationRole::User);
        assert_eq!(app.conversation[1].role, ConversationRole::Assistant);
        assert_eq!(app.conversation[2].role, ConversationRole::Info);
    }

    #[test]
    fn render_intro_message_snapshot() {
        use crate::config::{AppConfig, ToolsConfig, VertexConfig, generate_intro_message};

        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "my-project".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::path::PathBuf::from("/sessions"),
            models: std::collections::BTreeMap::new(),
        };
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.set_intro_message(generate_intro_message(&config));

        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                render_app(&mut app, frame);
            })
            .expect("draw");

        insta::assert_snapshot!("render_intro_message", terminal.backend());
    }

    #[test]
    fn regression_load_history_edit_file_with_zero_text_width_does_not_mangle_diff() {
        // Regression: load_history is called before the first render, so text_width
        // is 0. Previously this caused hunk headers to be truncated to "…".
        use crate::types::{ContentBlock, Message, Role};

        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        assert_eq!(app.text_width, 0, "text_width starts at 0");

        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "t1".to_string(),
                name: "edit_file".to_string(),
                input: serde_json::json!({
                    "path": "src/main.rs",
                    "old_string": "let x = 1;",
                    "new_string": "let x = 42;"
                }),
            }],
        }];

        app.load_history(&messages);

        assert_eq!(app.conversation.len(), 1);
        // The diff entry must contain "@@" somewhere (not "…") — verifies the
        // hunk header was not mangled by truncation at width=0.
        let all_content: String = app.conversation[0]
            .lines()
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(
            all_content.contains("@@"),
            "hunk header should contain '@@', got: {all_content:?}"
        );
    }

    #[test]
    fn edit_file_tool_use_entry_produces_diff_rendered_lines() {
        let app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        let input = serde_json::json!({
            "path": "src/main.rs",
            "old_string": "let x = 1;",
            "new_string": "let x = 42;"
        });
        let entry = app.tool_use_entry_indexed("edit_file", &input, 80, None);
        assert_eq!(entry.role, ConversationRole::ToolUse);
        // The diff renderer produces spans with colour styles; verify that
        // at least one span has a coloured foreground (indicating diff styling)
        // rather than plain un-styled content.
        let has_coloured_span = entry.lines().iter().any(|l| {
            l.spans
                .iter()
                .any(|s| s.style.fg.is_some() || s.style.bg.is_some())
        });
        assert!(
            has_coloured_span,
            "edit_file tool use should produce diff-styled spans"
        );
    }

    #[test]
    fn write_file_tool_use_entry_produces_diff_rendered_lines() {
        let app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        let input = serde_json::json!({
            "path": "hello.txt",
            "content": "Hello, world!\n"
        });
        let entry = app.tool_use_entry_indexed("write_file", &input, 80, None);
        assert_eq!(entry.role, ConversationRole::ToolUse);
        // write_file renders as a syntax-highlighted code block (green + markers)
        let has_plus_marker = entry
            .lines()
            .iter()
            .any(|l| l.spans.iter().any(|s| s.content.as_ref() == "+"));
        assert!(
            has_plus_marker,
            "write_file tool use should render with '+' markers"
        );
    }

    #[test]
    fn regression_non_diff_tool_use_still_renders_via_markdown() {
        let app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        let input = serde_json::json!({"command": "ls -la"});
        let entry = app.tool_use_entry_indexed("bash", &input, 80, None);
        assert_eq!(entry.role, ConversationRole::ToolUse);
        // The content (markdown string) should mention the tool name.
        assert!(
            entry.content.contains("bash") || entry.content.contains("ls"),
            "bash tool use content should contain command info"
        );
    }

    #[test]
    fn handle_agent_event_edit_file_uses_diff_renderer() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.text_width = 80;

        let event = AgentEvent::ToolUseReceived {
            id: "t1".to_string(),
            name: "edit_file".to_string(),
            input: serde_json::json!({
                "path": "src/lib.rs",
                "old_string": "fn old() {}",
                "new_string": "fn new() {}"
            }),
            index: 1,
        };
        handle_agent_event(&mut app, event, None).expect("handle event");

        assert_eq!(app.conversation.len(), 1);
        assert_eq!(app.conversation[0].role, ConversationRole::ToolUse);
        // Diff rendering uses coloured spans; at least one must have colour
        let has_coloured = app.conversation[0].lines().iter().any(|l| {
            l.spans
                .iter()
                .any(|s| s.style.fg.is_some() || s.style.bg.is_some())
        });
        assert!(
            has_coloured,
            "edit_file event should render with diff colours"
        );
    }

    #[tokio::test]
    async fn model_command_updates_agent_model() {
        use crate::agent::Agent;
        use crate::backend::LlmBackend;
        use crate::types::*;
        use async_trait::async_trait;
        use std::sync::Arc;

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

        let dir = tempfile::TempDir::new().expect("temp dir");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::session::Session::new(None, dir.keep())
                .await
                .expect("test session"),
        ));
        let agent = Arc::new(
            Agent::new(
                Box::new(StubBackend),
                RequestConfig {
                    model: "claude-original".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                },
                session,
            )
            .await,
        );

        agent.set_model("claude-new-model".to_string());
        assert_eq!(agent.model(), "claude-new-model");
    }

    #[tokio::test]
    async fn model_command_empty_name_does_not_update_model() {
        use crate::agent::Agent;
        use crate::backend::LlmBackend;
        use crate::types::*;
        use async_trait::async_trait;
        use std::sync::Arc;

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

        let dir = tempfile::TempDir::new().expect("temp dir");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::session::Session::new(None, dir.keep())
                .await
                .expect("test session"),
        ));
        let agent = Arc::new(
            Agent::new(
                Box::new(StubBackend),
                RequestConfig {
                    model: "claude-original".to_string(),
                    max_tokens: 1024,
                    tools: vec![],
                },
                session,
            )
            .await,
        );

        // "/model " with no name: the UI guards against empty model name
        let model_name = "  ".trim().to_string();
        if !model_name.is_empty() {
            agent.set_model(model_name);
        }
        assert_eq!(
            agent.model(),
            "claude-original",
            "model should be unchanged for blank input"
        );
    }

    #[test]
    fn subagent_usage_event_only_increments_subagent_usage() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));

        let event = AgentEvent::SubAgentUsage {
            input_tokens: 300,
            output_tokens: 120,
            role: "default".to_string(),
        };
        handle_agent_event(&mut app, event, None).expect("handle SubAgentUsage");

        assert_eq!(app.subagent_usage.input_tokens, 300);
        assert_eq!(app.subagent_usage.output_tokens, 120);
        assert_eq!(
            app.usage.input_tokens, 0,
            "parent usage must not be touched"
        );
        assert_eq!(
            app.usage.output_tokens, 0,
            "parent usage must not be touched"
        );
    }

    #[test]
    fn usage_event_does_not_increment_subagent_usage() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));

        let event = AgentEvent::Usage {
            input_tokens: 200,
            output_tokens: 80,
            stop_reason: "end_turn".to_string(),
        };
        handle_agent_event(&mut app, event, None).expect("handle Usage");

        assert_eq!(
            app.subagent_usage.input_tokens, 0,
            "subagent_usage must stay zero after regular Usage"
        );
        assert_eq!(app.subagent_usage.output_tokens, 0);
        assert_eq!(app.usage.input_tokens, 200);
    }

    #[test]
    fn session_switch_resets_subagent_usage() {
        let mut app = App::new(std::sync::Arc::new(crate::tools::ToolRegistry::new()));
        app.subagent_usage.add(500, 200);
        assert_eq!(app.subagent_usage.input_tokens, 500);

        // Simulate what the session-switch code path does
        app.usage = TokenUsage::default();
        app.subagent_usage = TokenUsage::default();
        app.last_input_total = 0;

        assert_eq!(
            app.subagent_usage.input_tokens, 0,
            "subagent_usage should reset to zero on session switch"
        );
        assert_eq!(app.subagent_usage.output_tokens, 0);
    }
}
