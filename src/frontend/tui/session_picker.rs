use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState};

use crate::session::SessionSummary;

pub struct SessionPicker {
    pub sessions: Vec<SessionSummary>,
    state: ListState,
}

impl SessionPicker {
    pub fn new(sessions: Vec<SessionSummary>) -> Self {
        let mut state = ListState::default();
        if !sessions.is_empty() {
            state.select(Some(0));
        }
        Self { sessions, state }
    }

    /// Move selection down (vim `j`).
    pub fn move_down(&mut self) {
        if self.sessions.is_empty() {
            return;
        }
        let next = match self.state.selected() {
            Some(i) => (i + 1).min(self.sessions.len() - 1),
            None => 0,
        };
        self.state.select(Some(next));
    }

    /// Move selection up (vim `k`).
    pub fn move_up(&mut self) {
        if self.sessions.is_empty() {
            return;
        }
        let prev = match self.state.selected() {
            Some(i) => i.saturating_sub(1),
            None => 0,
        };
        self.state.select(Some(prev));
    }

    /// Return the currently selected session ID, if any.
    pub fn selected_id(&self) -> Option<&str> {
        let idx = self.state.selected()?;
        self.sessions.get(idx).map(|s| s.id.as_str())
    }

    /// Handle a key event. Returns a `SessionPickerAction` indicating what happened.
    pub fn handle_key(&mut self, key: KeyEvent) -> SessionPickerAction {
        match key.code {
            KeyCode::Char('j') => {
                self.move_down();
                SessionPickerAction::None
            }
            KeyCode::Char('k') => {
                self.move_up();
                SessionPickerAction::None
            }
            KeyCode::Enter => match self.selected_id() {
                Some(id) => SessionPickerAction::Select(id.to_string()),
                None => SessionPickerAction::None,
            },
            KeyCode::Char('q') | KeyCode::Esc => SessionPickerAction::Close,
            _ => SessionPickerAction::None,
        }
    }

    /// Render the session picker as an overlay centred in `area`.
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let popup = super::centered_rect(80, 70, area);
        frame.render_widget(Clear, popup);

        let items: Vec<ListItem> = self
            .sessions
            .iter()
            .map(|s| {
                let preview = if s.first_user_message.is_empty() {
                    "(no messages)".to_string()
                } else {
                    let truncated: String =
                        s.first_user_message.chars().take(60).collect::<String>();
                    if s.first_user_message.chars().count() > 60 {
                        format!("{}…", truncated)
                    } else {
                        truncated
                    }
                };
                ListItem::new(format!("{}  {}", s.id.get(..8).unwrap_or(&s.id), preview))
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Sessions (j/k to navigate, Enter to open, q to close) "),
            )
            .highlight_style(
                Style::default()
                    .bg(Color::Blue)
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("> ");

        frame.render_stateful_widget(list, popup, &mut self.state);
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum SessionPickerAction {
    None,
    Select(String),
    Close,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use std::time::SystemTime;

    fn make_session(id: &str, first_msg: &str) -> SessionSummary {
        SessionSummary {
            id: id.to_string(),
            first_user_message: first_msg.to_string(),
            modified: SystemTime::UNIX_EPOCH,
        }
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn new_picker_selects_first_item() {
        let sessions = vec![
            make_session("01900000-0000-7000-0000-000000000001", "hello"),
            make_session("01900000-0000-7000-0000-000000000002", "world"),
        ];
        let picker = SessionPicker::new(sessions);
        assert_eq!(
            picker.selected_id(),
            Some("01900000-0000-7000-0000-000000000001")
        );
    }

    #[test]
    fn new_picker_with_empty_list_has_no_selection() {
        let picker = SessionPicker::new(vec![]);
        assert_eq!(picker.selected_id(), None);
    }

    #[test]
    fn j_moves_selection_down() {
        let sessions = vec![
            make_session("01900000-0000-7000-0000-000000000001", "a"),
            make_session("01900000-0000-7000-0000-000000000002", "b"),
            make_session("01900000-0000-7000-0000-000000000003", "c"),
        ];
        let mut picker = SessionPicker::new(sessions);
        picker.handle_key(key(KeyCode::Char('j')));
        assert_eq!(
            picker.selected_id(),
            Some("01900000-0000-7000-0000-000000000002")
        );
        picker.handle_key(key(KeyCode::Char('j')));
        assert_eq!(
            picker.selected_id(),
            Some("01900000-0000-7000-0000-000000000003")
        );
    }

    #[test]
    fn j_at_bottom_clamps() {
        let sessions = vec![
            make_session("01900000-0000-7000-0000-000000000001", "a"),
            make_session("01900000-0000-7000-0000-000000000002", "b"),
        ];
        let mut picker = SessionPicker::new(sessions);
        picker.handle_key(key(KeyCode::Char('j')));
        picker.handle_key(key(KeyCode::Char('j')));
        picker.handle_key(key(KeyCode::Char('j')));
        assert_eq!(
            picker.selected_id(),
            Some("01900000-0000-7000-0000-000000000002")
        );
    }

    #[test]
    fn k_moves_selection_up() {
        let sessions = vec![
            make_session("01900000-0000-7000-0000-000000000001", "a"),
            make_session("01900000-0000-7000-0000-000000000002", "b"),
        ];
        let mut picker = SessionPicker::new(sessions);
        picker.handle_key(key(KeyCode::Char('j')));
        picker.handle_key(key(KeyCode::Char('k')));
        assert_eq!(
            picker.selected_id(),
            Some("01900000-0000-7000-0000-000000000001")
        );
    }

    #[test]
    fn k_at_top_clamps() {
        let sessions = vec![
            make_session("01900000-0000-7000-0000-000000000001", "a"),
            make_session("01900000-0000-7000-0000-000000000002", "b"),
        ];
        let mut picker = SessionPicker::new(sessions);
        picker.handle_key(key(KeyCode::Char('k')));
        picker.handle_key(key(KeyCode::Char('k')));
        assert_eq!(
            picker.selected_id(),
            Some("01900000-0000-7000-0000-000000000001")
        );
    }

    #[test]
    fn enter_returns_select_action_with_id() {
        let sessions = vec![make_session(
            "01900000-0000-7000-0000-000000000001",
            "hello",
        )];
        let mut picker = SessionPicker::new(sessions);
        let action = picker.handle_key(key(KeyCode::Enter));
        assert_eq!(
            action,
            SessionPickerAction::Select("01900000-0000-7000-0000-000000000001".to_string())
        );
    }

    #[test]
    fn enter_on_empty_list_returns_none() {
        let mut picker = SessionPicker::new(vec![]);
        let action = picker.handle_key(key(KeyCode::Enter));
        assert_eq!(action, SessionPickerAction::None);
    }

    #[test]
    fn q_returns_close_action() {
        let sessions = vec![make_session(
            "01900000-0000-7000-0000-000000000001",
            "hello",
        )];
        let mut picker = SessionPicker::new(sessions);
        let action = picker.handle_key(key(KeyCode::Char('q')));
        assert_eq!(action, SessionPickerAction::Close);
    }

    #[test]
    fn esc_returns_close_action() {
        let sessions = vec![make_session(
            "01900000-0000-7000-0000-000000000001",
            "hello",
        )];
        let mut picker = SessionPicker::new(sessions);
        let action = picker.handle_key(key(KeyCode::Esc));
        assert_eq!(action, SessionPickerAction::Close);
    }

    #[test]
    fn unknown_key_returns_none() {
        let sessions = vec![make_session(
            "01900000-0000-7000-0000-000000000001",
            "hello",
        )];
        let mut picker = SessionPicker::new(sessions);
        let action = picker.handle_key(key(KeyCode::Char('x')));
        assert_eq!(action, SessionPickerAction::None);
    }

    #[test]
    fn render_session_picker_empty() {
        let mut picker = SessionPicker::new(vec![]);

        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                let area = frame.area();
                picker.render(frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("session_picker_empty", terminal.backend());
    }

    #[test]
    fn render_session_picker_with_sessions() {
        let sessions = vec![
            SessionSummary {
                id: "01900000-0000-7000-0000-000000000001".to_string(),
                first_user_message: "Hello, can you help me write a Rust function?".to_string(),
                modified: SystemTime::UNIX_EPOCH,
            },
            SessionSummary {
                id: "01900000-0000-7000-0000-000000000002".to_string(),
                first_user_message: "Explain the borrow checker".to_string(),
                modified: SystemTime::UNIX_EPOCH,
            },
            SessionSummary {
                id: "01900000-0000-7000-0000-000000000003".to_string(),
                first_user_message: "".to_string(),
                modified: SystemTime::UNIX_EPOCH,
            },
        ];
        let mut picker = SessionPicker::new(sessions);

        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                let area = frame.area();
                picker.render(frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("session_picker_with_sessions", terminal.backend());
    }

    #[test]
    fn render_session_picker_second_item_selected() {
        let sessions = vec![
            SessionSummary {
                id: "01900000-0000-7000-0000-000000000001".to_string(),
                first_user_message: "First session message".to_string(),
                modified: SystemTime::UNIX_EPOCH,
            },
            SessionSummary {
                id: "01900000-0000-7000-0000-000000000002".to_string(),
                first_user_message: "Second session message".to_string(),
                modified: SystemTime::UNIX_EPOCH,
            },
        ];
        let mut picker = SessionPicker::new(sessions);
        picker.move_down();

        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                let area = frame.area();
                picker.render(frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("session_picker_second_selected", terminal.backend());
    }
}
