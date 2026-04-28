use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph};

pub struct ListPicker<T> {
    pub items: Vec<T>,
    state: ListState,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PickerAction {
    None,
    Select(usize),
    Close,
}

impl<T> ListPicker<T> {
    pub fn new(items: Vec<T>) -> Self {
        let mut state = ListState::default();
        if !items.is_empty() {
            state.select(Some(0));
        }
        Self { items, state }
    }

    pub fn move_down(&mut self) {
        if self.items.is_empty() {
            return;
        }
        let next = match self.state.selected() {
            Some(i) => (i + 1).min(self.items.len() - 1),
            None => 0,
        };
        self.state.select(Some(next));
    }

    pub fn move_up(&mut self) {
        if self.items.is_empty() {
            return;
        }
        let prev = match self.state.selected() {
            Some(i) => i.saturating_sub(1),
            None => 0,
        };
        self.state.select(Some(prev));
    }

    pub fn selected_index(&self) -> Option<usize> {
        self.state.selected()
    }

    pub fn handle_key(&mut self, key: KeyEvent, enter_selects: bool) -> PickerAction {
        match key.code {
            KeyCode::Char('j') => {
                self.move_down();
                PickerAction::None
            }
            KeyCode::Char('k') => {
                self.move_up();
                PickerAction::None
            }
            KeyCode::Enter => {
                if enter_selects {
                    match self.state.selected() {
                        Some(idx) => PickerAction::Select(idx),
                        None => PickerAction::None,
                    }
                } else {
                    PickerAction::None
                }
            }
            KeyCode::Char('q') | KeyCode::Esc => PickerAction::Close,
            _ => PickerAction::None,
        }
    }

    pub fn render<F>(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        title: &str,
        render_item: F,
        empty_message: Option<&str>,
    ) where
        F: Fn(&T) -> ListItem<'static>,
    {
        let popup = super::centered_rect(80, 70, area);
        frame.render_widget(Clear, popup);

        if self.items.is_empty() {
            let block = Block::default()
                .borders(Borders::ALL)
                .title(title.to_string());
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            if let Some(msg) = empty_message {
                let para = Paragraph::new(msg.to_string())
                    .alignment(Alignment::Center)
                    .style(Style::default().fg(Color::DarkGray));
                let y_offset = inner.height / 2;
                let msg_area = Rect {
                    y: inner.y + y_offset,
                    height: 1,
                    ..inner
                };
                frame.render_widget(para, msg_area);
            }
            return;
        }

        let items: Vec<ListItem> = self.items.iter().map(render_item).collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title.to_string()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn make_picker(n: usize) -> ListPicker<String> {
        let items: Vec<String> = (0..n).map(|i| format!("item {i}")).collect();
        ListPicker::new(items)
    }

    #[test]
    fn new_selects_first_when_non_empty() {
        let picker = make_picker(3);
        assert_eq!(picker.selected_index(), Some(0));
    }

    #[test]
    fn new_with_empty_list_has_no_selection() {
        let picker: ListPicker<String> = ListPicker::new(vec![]);
        assert_eq!(picker.selected_index(), None);
    }

    #[test]
    fn move_down_advances_and_clamps() {
        let mut picker = make_picker(2);
        picker.move_down();
        assert_eq!(picker.selected_index(), Some(1));
        picker.move_down();
        picker.move_down();
        assert_eq!(picker.selected_index(), Some(1));
    }

    #[test]
    fn move_up_retreats_and_clamps() {
        let mut picker = make_picker(2);
        picker.move_down();
        picker.move_up();
        assert_eq!(picker.selected_index(), Some(0));
        picker.move_up();
        picker.move_up();
        assert_eq!(picker.selected_index(), Some(0));
    }

    #[test]
    fn enter_selects_when_enter_selects_true() {
        let mut picker = make_picker(2);
        picker.move_down();
        let action = picker.handle_key(key(KeyCode::Enter), true);
        assert_eq!(action, PickerAction::Select(1));
    }

    #[test]
    fn enter_is_noop_when_enter_selects_false() {
        let mut picker = make_picker(2);
        let action = picker.handle_key(key(KeyCode::Enter), false);
        assert_eq!(action, PickerAction::None);
    }

    #[test]
    fn enter_on_empty_list_returns_none_even_when_enter_selects_true() {
        let mut picker: ListPicker<String> = ListPicker::new(vec![]);
        let action = picker.handle_key(key(KeyCode::Enter), true);
        assert_eq!(action, PickerAction::None);
    }

    #[test]
    fn q_returns_close() {
        let mut picker = make_picker(1);
        assert_eq!(
            picker.handle_key(key(KeyCode::Char('q')), false),
            PickerAction::Close
        );
    }

    #[test]
    fn esc_returns_close() {
        let mut picker = make_picker(1);
        assert_eq!(
            picker.handle_key(key(KeyCode::Esc), false),
            PickerAction::Close
        );
    }

    #[test]
    fn unknown_key_returns_none() {
        let mut picker = make_picker(1);
        assert_eq!(
            picker.handle_key(key(KeyCode::Char('x')), false),
            PickerAction::None
        );
    }
}
