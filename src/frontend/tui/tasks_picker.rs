use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph};

use crate::session::{TaskRecord, TaskStatus};

pub struct TasksPicker {
    pub tasks: Vec<TaskRecord>,
    state: ListState,
}

impl TasksPicker {
    pub fn new(tasks: Vec<TaskRecord>) -> Self {
        let mut state = ListState::default();
        if !tasks.is_empty() {
            state.select(Some(0));
        }
        Self { tasks, state }
    }

    pub fn move_down(&mut self) {
        if self.tasks.is_empty() {
            return;
        }
        let next = match self.state.selected() {
            Some(i) => (i + 1).min(self.tasks.len() - 1),
            None => 0,
        };
        self.state.select(Some(next));
    }

    pub fn move_up(&mut self) {
        if self.tasks.is_empty() {
            return;
        }
        let prev = match self.state.selected() {
            Some(i) => i.saturating_sub(1),
            None => 0,
        };
        self.state.select(Some(prev));
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> TasksPickerAction {
        match key.code {
            KeyCode::Char('j') => {
                self.move_down();
                TasksPickerAction::None
            }
            KeyCode::Char('k') => {
                self.move_up();
                TasksPickerAction::None
            }
            KeyCode::Enter => TasksPickerAction::None,
            KeyCode::Char('q') | KeyCode::Esc => TasksPickerAction::Close,
            _ => TasksPickerAction::None,
        }
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let popup = super::centered_rect(80, 70, area);
        frame.render_widget(Clear, popup);

        if self.tasks.is_empty() {
            let block = Block::default()
                .borders(Borders::ALL)
                .title(" Tasks (j/k to navigate, q to close) ");
            let inner = block.inner(popup);
            frame.render_widget(block, popup);
            let msg = Paragraph::new("No tasks yet.")
                .alignment(Alignment::Center)
                .style(Style::default().fg(Color::DarkGray));
            // centre vertically
            let y_offset = inner.height / 2;
            let msg_area = Rect {
                y: inner.y + y_offset,
                height: 1,
                ..inner
            };
            frame.render_widget(msg, msg_area);
            return;
        }

        let items: Vec<ListItem> = self
            .tasks
            .iter()
            .map(|t| {
                let (glyph, color) = match t.status {
                    TaskStatus::InProgress => ('●', Color::Yellow),
                    TaskStatus::Pending => ('○', Color::DarkGray),
                    TaskStatus::Completed => ('✓', Color::Green),
                };
                let desc_suffix = match &t.description {
                    Some(d) if !d.is_empty() => {
                        let mut chars = d.chars();
                        let truncated: String = chars.by_ref().take(40).collect();
                        if chars.next().is_some() {
                            format!(" — {}…", truncated)
                        } else {
                            format!(" — {}", truncated)
                        }
                    }
                    _ => String::new(),
                };
                let label = format!("{} {}{}", glyph, t.title, desc_suffix);
                ListItem::new(label).style(Style::default().fg(color))
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Tasks (j/k to navigate, q to close) "),
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
pub enum TasksPickerAction {
    None,
    Close,
}

/// Sort tasks: in-progress first, then pending, then completed; stable by created_at within group.
pub fn sort_tasks(tasks: &mut [TaskRecord]) {
    tasks.sort_by_key(|t| {
        let order = match t.status {
            TaskStatus::InProgress => 0,
            TaskStatus::Pending => 1,
            TaskStatus::Completed => 2,
        };
        (order, t.created_at)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn make_task(id: i64, title: &str, status: TaskStatus, created_at: i64) -> TaskRecord {
        TaskRecord {
            id,
            title: title.to_string(),
            description: None,
            status,
            created_at,
            updated_at: created_at,
        }
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn new_picker_selects_first_when_non_empty() {
        let tasks = vec![
            make_task(1, "task a", TaskStatus::Pending, 1000),
            make_task(2, "task b", TaskStatus::Completed, 2000),
        ];
        let picker = TasksPicker::new(tasks);
        assert!(picker.state.selected() == Some(0));
    }

    #[test]
    fn new_picker_with_empty_list_has_no_selection() {
        let picker = TasksPicker::new(vec![]);
        assert!(picker.state.selected().is_none());
    }

    #[test]
    fn j_moves_down_and_clamps_at_bottom() {
        let tasks = vec![
            make_task(1, "a", TaskStatus::Pending, 1000),
            make_task(2, "b", TaskStatus::Pending, 2000),
        ];
        let mut picker = TasksPicker::new(tasks);
        picker.handle_key(key(KeyCode::Char('j')));
        assert_eq!(picker.state.selected(), Some(1));
        picker.handle_key(key(KeyCode::Char('j')));
        picker.handle_key(key(KeyCode::Char('j')));
        assert_eq!(picker.state.selected(), Some(1));
    }

    #[test]
    fn k_moves_up_and_clamps_at_top() {
        let tasks = vec![
            make_task(1, "a", TaskStatus::Pending, 1000),
            make_task(2, "b", TaskStatus::Pending, 2000),
        ];
        let mut picker = TasksPicker::new(tasks);
        picker.handle_key(key(KeyCode::Char('j')));
        picker.handle_key(key(KeyCode::Char('k')));
        assert_eq!(picker.state.selected(), Some(0));
        picker.handle_key(key(KeyCode::Char('k')));
        assert_eq!(picker.state.selected(), Some(0));
    }

    #[test]
    fn enter_is_noop() {
        let tasks = vec![make_task(1, "a", TaskStatus::Pending, 1000)];
        let mut picker = TasksPicker::new(tasks);
        let action = picker.handle_key(key(KeyCode::Enter));
        assert_eq!(action, TasksPickerAction::None);
    }

    #[test]
    fn q_returns_close_action() {
        let tasks = vec![make_task(1, "a", TaskStatus::Pending, 1000)];
        let mut picker = TasksPicker::new(tasks);
        let action = picker.handle_key(key(KeyCode::Char('q')));
        assert_eq!(action, TasksPickerAction::Close);
    }

    #[test]
    fn esc_returns_close_action() {
        let tasks = vec![make_task(1, "a", TaskStatus::Pending, 1000)];
        let mut picker = TasksPicker::new(tasks);
        let action = picker.handle_key(key(KeyCode::Esc));
        assert_eq!(action, TasksPickerAction::Close);
    }

    #[test]
    fn unknown_key_returns_none() {
        let tasks = vec![make_task(1, "a", TaskStatus::Pending, 1000)];
        let mut picker = TasksPicker::new(tasks);
        let action = picker.handle_key(key(KeyCode::Char('x')));
        assert_eq!(action, TasksPickerAction::None);
    }

    #[test]
    fn sort_orders_in_progress_then_pending_then_completed_by_created_at() {
        let mut tasks = vec![
            make_task(1, "completed-old", TaskStatus::Completed, 100),
            make_task(2, "pending-new", TaskStatus::Pending, 300),
            make_task(3, "in-progress-new", TaskStatus::InProgress, 400),
            make_task(4, "pending-old", TaskStatus::Pending, 200),
            make_task(5, "in-progress-old", TaskStatus::InProgress, 50),
        ];
        sort_tasks(&mut tasks);
        assert_eq!(tasks[0].title, "in-progress-old");
        assert_eq!(tasks[1].title, "in-progress-new");
        assert_eq!(tasks[2].title, "pending-old");
        assert_eq!(tasks[3].title, "pending-new");
        assert_eq!(tasks[4].title, "completed-old");
    }

    #[test]
    fn tasks_picker_empty_snapshot() {
        let mut picker = TasksPicker::new(vec![]);
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                let area = frame.area();
                picker.render(frame, area);
            })
            .expect("draw");
        insta::assert_snapshot!("tasks_picker_empty", terminal.backend());
    }

    #[test]
    fn tasks_picker_with_tasks_snapshot() {
        let mut tasks = vec![
            make_task(1, "implement feature", TaskStatus::InProgress, 1000),
            make_task(2, "write tests", TaskStatus::Pending, 2000),
            make_task(3, "deploy", TaskStatus::Completed, 3000),
        ];
        sort_tasks(&mut tasks);
        let mut picker = TasksPicker::new(tasks);
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                let area = frame.area();
                picker.render(frame, area);
            })
            .expect("draw");
        insta::assert_snapshot!("tasks_picker_with_tasks", terminal.backend());
    }

    #[test]
    fn tasks_picker_second_selected_snapshot() {
        let mut tasks = vec![
            make_task(1, "first task", TaskStatus::InProgress, 1000),
            make_task(2, "second task", TaskStatus::Pending, 2000),
        ];
        sort_tasks(&mut tasks);
        let mut picker = TasksPicker::new(tasks);
        picker.move_down();
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                let area = frame.area();
                picker.render(frame, area);
            })
            .expect("draw");
        insta::assert_snapshot!("tasks_picker_second_selected", terminal.backend());
    }

    fn make_task_with_desc(
        id: i64,
        title: &str,
        status: TaskStatus,
        created_at: i64,
        desc: &str,
    ) -> TaskRecord {
        TaskRecord {
            id,
            title: title.to_string(),
            description: Some(desc.to_string()),
            status,
            created_at,
            updated_at: created_at,
        }
    }

    #[test]
    fn description_short_renders_without_ellipsis() {
        let task = make_task_with_desc(1, "my task", TaskStatus::Pending, 1000, "short desc");
        let mut picker = TasksPicker::new(vec![task]);
        // exercise the render path by drawing to a backend
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| picker.render(frame, frame.area()))
            .expect("draw");
        let rendered = format!("{:?}", terminal.backend());
        assert!(
            rendered.contains("short desc"),
            "short description should appear without ellipsis"
        );
        assert!(!rendered.contains('…'), "no ellipsis for short description");
    }

    #[test]
    fn description_long_renders_with_ellipsis() {
        // 41 chars — one over the limit
        let long_desc = "a".repeat(41);
        let task = make_task_with_desc(1, "my task", TaskStatus::Pending, 1000, &long_desc);
        let mut picker = TasksPicker::new(vec![task]);
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| picker.render(frame, frame.area()))
            .expect("draw");
        let rendered = format!("{:?}", terminal.backend());
        assert!(
            rendered.contains('…'),
            "long description should be truncated with ellipsis"
        );
    }

    #[test]
    fn description_exactly_40_chars_renders_without_ellipsis() {
        let desc = "b".repeat(40);
        let task = make_task_with_desc(1, "my task", TaskStatus::Pending, 1000, &desc);
        let mut picker = TasksPicker::new(vec![task]);
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| picker.render(frame, frame.area()))
            .expect("draw");
        let rendered = format!("{:?}", terminal.backend());
        assert!(
            !rendered.contains('…'),
            "exactly-40-char description should not be truncated"
        );
    }

    #[test]
    fn tasks_picker_with_description_snapshot() {
        let tasks = vec![make_task_with_desc(
            1,
            "implement feature",
            TaskStatus::InProgress,
            1000,
            "this is a task with a description",
        )];
        let mut picker = TasksPicker::new(tasks);
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                let area = frame.area();
                picker.render(frame, area);
            })
            .expect("draw");
        insta::assert_snapshot!("tasks_picker_with_description", terminal.backend());
    }
}
