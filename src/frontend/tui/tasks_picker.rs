use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::ListItem;

use crate::session::{TaskRecord, TaskStatus};

use super::list_picker::{ListPicker, PickerAction};

pub struct TasksPicker {
    inner: ListPicker<TaskRecord>,
}

impl TasksPicker {
    pub fn new(tasks: Vec<TaskRecord>) -> Self {
        Self {
            inner: ListPicker::new(tasks),
        }
    }

    pub fn move_down(&mut self) {
        self.inner.move_down();
    }

    pub fn move_up(&mut self) {
        self.inner.move_up();
    }

    pub fn tasks(&self) -> &[TaskRecord] {
        self.inner.items()
    }

    pub fn selected_index(&self) -> Option<usize> {
        self.inner.selected_index()
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> TasksPickerAction {
        match self.inner.handle_key(key, false) {
            PickerAction::Close => TasksPickerAction::Close,
            _ => TasksPickerAction::None,
        }
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        self.inner.render(
            frame,
            area,
            " Tasks (j/k to navigate, q to close) ",
            |t| {
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
            },
            Some("No tasks yet."),
        );
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
        assert!(picker.selected_index() == Some(0));
    }

    #[test]
    fn new_picker_with_empty_list_has_no_selection() {
        let picker = TasksPicker::new(vec![]);
        assert!(picker.selected_index().is_none());
    }

    #[test]
    fn j_moves_down_and_clamps_at_bottom() {
        let tasks = vec![
            make_task(1, "a", TaskStatus::Pending, 1000),
            make_task(2, "b", TaskStatus::Pending, 2000),
        ];
        let mut picker = TasksPicker::new(tasks);
        picker.handle_key(key(KeyCode::Char('j')));
        assert_eq!(picker.selected_index(), Some(1));
        picker.handle_key(key(KeyCode::Char('j')));
        picker.handle_key(key(KeyCode::Char('j')));
        assert_eq!(picker.selected_index(), Some(1));
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
        assert_eq!(picker.selected_index(), Some(0));
        picker.handle_key(key(KeyCode::Char('k')));
        assert_eq!(picker.selected_index(), Some(0));
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

    #[test]
    fn description_short_renders_without_ellipsis() {
        let task = make_task_with_desc(1, "my task", TaskStatus::Pending, 1000, "short desc");
        let mut picker = TasksPicker::new(vec![task]);
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
