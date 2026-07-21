use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::widgets::ListItem;

use crate::session::SessionSummary;

use super::list_picker::{ListPicker, PickerAction};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionPickerRow {
    Session(SessionSummary),
    LoadMore,
}

pub struct SessionPicker {
    inner: ListPicker<SessionPickerRow>,
}

impl SessionPicker {
    pub fn new(sessions: Vec<SessionSummary>, has_more: bool) -> Self {
        let mut rows: Vec<SessionPickerRow> = sessions
            .into_iter()
            .map(SessionPickerRow::Session)
            .collect();
        if has_more {
            rows.push(SessionPickerRow::LoadMore);
        }
        Self {
            inner: ListPicker::new(rows),
        }
    }

    pub fn extend(&mut self, sessions: Vec<SessionSummary>, has_more: bool) {
        if matches!(self.inner.items().last(), Some(SessionPickerRow::LoadMore)) {
            self.inner.pop_last();
        }
        let rows: Vec<SessionPickerRow> = sessions
            .into_iter()
            .map(SessionPickerRow::Session)
            .collect();
        self.inner.extend(rows);
        if has_more {
            self.inner.extend(vec![SessionPickerRow::LoadMore]);
        }
    }

    pub fn move_down(&mut self) {
        self.inner.move_down();
    }

    pub fn move_up(&mut self) {
        self.inner.move_up();
    }

    pub fn selected_id(&self) -> Option<&str> {
        match self.inner.selected_item() {
            Some(SessionPickerRow::Session(s)) => Some(s.id.as_str()),
            _ => None,
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> SessionPickerAction {
        match self.inner.handle_key(key, true) {
            PickerAction::Select(idx) => match self.inner.items().get(idx) {
                Some(SessionPickerRow::Session(s)) => SessionPickerAction::Select(s.id.clone()),
                Some(SessionPickerRow::LoadMore) => SessionPickerAction::LoadMore,
                None => SessionPickerAction::None,
            },
            PickerAction::Close => SessionPickerAction::Close,
            PickerAction::None => SessionPickerAction::None,
        }
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        self.inner.render(
            frame,
            area,
            " Sessions (j/k to navigate, Enter to open, q to close) ",
            |row| match row {
                SessionPickerRow::Session(s) => {
                    let preview = if s.first_user_message.is_empty() {
                        "(no messages)".to_string()
                    } else {
                        let mut chars = s.first_user_message.chars();
                        let truncated: String = chars.by_ref().take(60).collect();
                        if chars.next().is_some() {
                            format!("{}…", truncated)
                        } else {
                            truncated
                        }
                    };
                    ListItem::new(format!("{}  {}", s.id.get(..8).unwrap_or(&s.id), preview))
                }
                SessionPickerRow::LoadMore => ListItem::new("... (load more sessions)"),
            },
            None,
        );
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum SessionPickerAction {
    None,
    Select(String),
    LoadMore,
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
        let picker = SessionPicker::new(sessions, false);
        assert_eq!(
            picker.selected_id(),
            Some("01900000-0000-7000-0000-000000000001")
        );
    }

    #[test]
    fn new_picker_with_empty_list_has_no_selection() {
        let picker = SessionPicker::new(vec![], false);
        assert_eq!(picker.selected_id(), None);
    }

    #[test]
    fn j_moves_selection_down() {
        let sessions = vec![
            make_session("01900000-0000-7000-0000-000000000001", "a"),
            make_session("01900000-0000-7000-0000-000000000002", "b"),
            make_session("01900000-0000-7000-0000-000000000003", "c"),
        ];
        let mut picker = SessionPicker::new(sessions, false);
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
        let mut picker = SessionPicker::new(sessions, false);
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
        let mut picker = SessionPicker::new(sessions, false);
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
        let mut picker = SessionPicker::new(sessions, false);
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
        let mut picker = SessionPicker::new(sessions, false);
        let action = picker.handle_key(key(KeyCode::Enter));
        assert_eq!(
            action,
            SessionPickerAction::Select("01900000-0000-7000-0000-000000000001".to_string())
        );
    }

    #[test]
    fn enter_on_empty_list_returns_none() {
        let mut picker = SessionPicker::new(vec![], false);
        let action = picker.handle_key(key(KeyCode::Enter));
        assert_eq!(action, SessionPickerAction::None);
    }

    #[test]
    fn q_returns_close_action() {
        let sessions = vec![make_session(
            "01900000-0000-7000-0000-000000000001",
            "hello",
        )];
        let mut picker = SessionPicker::new(sessions, false);
        let action = picker.handle_key(key(KeyCode::Char('q')));
        assert_eq!(action, SessionPickerAction::Close);
    }

    #[test]
    fn esc_returns_close_action() {
        let sessions = vec![make_session(
            "01900000-0000-7000-0000-000000000001",
            "hello",
        )];
        let mut picker = SessionPicker::new(sessions, false);
        let action = picker.handle_key(key(KeyCode::Esc));
        assert_eq!(action, SessionPickerAction::Close);
    }

    #[test]
    fn unknown_key_returns_none() {
        let sessions = vec![make_session(
            "01900000-0000-7000-0000-000000000001",
            "hello",
        )];
        let mut picker = SessionPicker::new(sessions, false);
        let action = picker.handle_key(key(KeyCode::Char('x')));
        assert_eq!(action, SessionPickerAction::None);
    }

    #[test]
    fn load_more_sentinel_appears_when_has_more_true() {
        let sessions = vec![
            make_session("01900000-0000-7000-0000-000000000001", "a"),
            make_session("01900000-0000-7000-0000-000000000002", "b"),
        ];
        let picker = SessionPicker::new(sessions, true);
        assert!(matches!(
            picker.inner.items().last(),
            Some(SessionPickerRow::LoadMore)
        ));
    }

    #[test]
    fn enter_on_sentinel_returns_load_more_action() {
        let sessions = vec![make_session("01900000-0000-7000-0000-000000000001", "a")];
        let mut picker = SessionPicker::new(sessions, true);
        picker.move_down();
        let action = picker.handle_key(key(KeyCode::Enter));
        assert_eq!(action, SessionPickerAction::LoadMore);
    }

    #[test]
    fn extend_replaces_sentinel_and_appends() {
        let sessions = vec![make_session("01900000-0000-7000-0000-000000000001", "a")];
        let mut picker = SessionPicker::new(sessions, true);
        assert!(matches!(
            picker.inner.items().last(),
            Some(SessionPickerRow::LoadMore)
        ));

        let more = vec![
            make_session("01900000-0000-7000-0000-000000000002", "b"),
            make_session("01900000-0000-7000-0000-000000000003", "c"),
        ];
        picker.extend(more, false);

        assert_eq!(picker.inner.items().len(), 3);
        assert!(!matches!(
            picker.inner.items().last(),
            Some(SessionPickerRow::LoadMore)
        ));
        assert!(matches!(
            picker.inner.items().last(),
            Some(SessionPickerRow::Session(_))
        ));
    }

    #[test]
    fn extend_preserves_sentinel_when_still_has_more() {
        let sessions = vec![make_session("01900000-0000-7000-0000-000000000001", "a")];
        let mut picker = SessionPicker::new(sessions, true);

        let more = vec![make_session("01900000-0000-7000-0000-000000000002", "b")];
        picker.extend(more, true);

        assert_eq!(picker.inner.items().len(), 3);
        assert!(matches!(
            picker.inner.items().last(),
            Some(SessionPickerRow::LoadMore)
        ));
    }

    #[test]
    fn render_session_picker_empty() {
        let mut picker = SessionPicker::new(vec![], false);
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
        let mut picker = SessionPicker::new(sessions, false);
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
        let mut picker = SessionPicker::new(sessions, false);
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

    #[test]
    fn render_session_picker_load_more_sentinel() {
        let sessions = vec![
            SessionSummary {
                id: "01900000-0000-7000-0000-000000000001".to_string(),
                first_user_message: "Hello world".to_string(),
                modified: SystemTime::UNIX_EPOCH,
            },
            SessionSummary {
                id: "01900000-0000-7000-0000-000000000002".to_string(),
                first_user_message: "Testing pagination".to_string(),
                modified: SystemTime::UNIX_EPOCH,
            },
        ];
        let mut picker = SessionPicker::new(sessions, true);
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                let area = frame.area();
                picker.render(frame, area);
            })
            .expect("draw");
        insta::assert_snapshot!("session_picker_load_more_sentinel", terminal.backend());
    }

    #[test]
    fn preview_exactly_60_chars_renders_without_ellipsis() {
        let msg = "a".repeat(60);
        let sessions = vec![make_session("01900000-0000-7000-0000-000000000001", &msg)];
        let mut picker = SessionPicker::new(sessions, false);
        let backend = ratatui::backend::TestBackend::new(160, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| picker.render(frame, frame.area()))
            .expect("draw");
        let rendered = format!("{:?}", terminal.backend());
        assert!(
            !rendered.contains('…'),
            "exactly-60-char preview should not be truncated"
        );
    }

    #[test]
    fn preview_61_chars_renders_with_ellipsis() {
        let msg = "b".repeat(61);
        let sessions = vec![make_session("01900000-0000-7000-0000-000000000001", &msg)];
        let mut picker = SessionPicker::new(sessions, false);
        let backend = ratatui::backend::TestBackend::new(160, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| picker.render(frame, frame.area()))
            .expect("draw");
        let rendered = format!("{:?}", terminal.backend());
        assert!(
            rendered.contains('…'),
            "61-char preview should be truncated with ellipsis"
        );
    }

    #[test]
    fn preview_multibyte_chars_truncate_on_char_boundary() {
        let msg = "→".repeat(61);
        let sessions = vec![make_session("01900000-0000-7000-0000-000000000001", &msg)];
        let mut picker = SessionPicker::new(sessions, false);
        let backend = ratatui::backend::TestBackend::new(160, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| picker.render(frame, frame.area()))
            .expect("draw");
        let rendered = format!("{:?}", terminal.backend());
        assert!(
            rendered.contains('…'),
            "multibyte 61-char preview should be truncated with ellipsis"
        );
    }
}
