use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::widgets::{Block, Borders};
#[cfg(test)]
use ratatui_textarea::DataCursor;
use ratatui_textarea::{CursorMove, TextArea, WrapMode};

const INSERT_TITLE: &str = " -- INSERT -- ";
const NORMAL_TITLE: &str = " -- NORMAL -- ";
const VISUAL_TITLE: &str = " -- VISUAL -- ";
const STREAMING_TITLE: &str = "Streaming... (Esc to interrupt)";
const SESSIONS_TITLE: &str = " Sessions ";
const TASKS_TITLE: &str = " Tasks ";
const MIN_HEIGHT: u16 = 3;
const MAX_INPUT_RATIO: u16 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputMode {
    Insert,
    Normal,
    Visual,
    Streaming,
    SessionPicker,
    TasksPicker,
    ToolConfirmation {
        name: String,
        input: serde_json::Value,
    },
}

pub struct InputArea<'a> {
    textarea: TextArea<'a>,
    mode: InputMode,
}

impl<'a> InputArea<'a> {
    pub fn new() -> Self {
        let textarea = TextArea::default();
        let mut input = Self {
            textarea,
            mode: InputMode::Insert,
        };
        input.textarea.set_wrap_mode(WrapMode::WordOrGlyph);
        input
            .textarea
            .set_cursor_line_style(Style::default().add_modifier(Modifier::empty()));
        input.apply_block();
        input
    }

    pub fn input(&mut self, event: crossterm::event::KeyEvent) -> bool {
        match self.mode {
            InputMode::Insert => match event {
                KeyEvent {
                    code: KeyCode::Esc, ..
                } => {
                    self.set_mode(InputMode::Normal);
                    true
                }
                // todo get past working. Figure out how `ratatui-textarea` is getting data from clipboard
                _ => self.textarea.input(event),
            },
            InputMode::Normal => match event {
                KeyEvent {
                    code: KeyCode::Char('i'),
                    ..
                } => {
                    self.set_mode(InputMode::Insert);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('o'),
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::End);
                    self.textarea.insert_newline();
                    self.set_mode(InputMode::Insert);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('O'),
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::Head);
                    self.textarea.insert_newline();
                    self.textarea.move_cursor(CursorMove::Up);
                    self.set_mode(InputMode::Insert);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('w'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::WordForward);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('b'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::WordBack);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('e'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::WordEnd);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('0'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::Head);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('$'),
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::End);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('v'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.start_selection();
                    self.set_mode(InputMode::Visual);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('p'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.paste();
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('h'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::Back);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('l'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::Forward);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('j'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::Down);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('k'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::Up);
                    true
                }
                _ => false,
            },
            InputMode::Visual => match event {
                KeyEvent {
                    code: KeyCode::Char('h'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::Back);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('l'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::Forward);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('w'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::WordForward);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('b'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::WordBack);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('e'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::WordEnd);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('0'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::Head);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('$'),
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::End);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('j'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::Down);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('k'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.move_cursor(CursorMove::Up);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('d'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.cut();
                    self.set_mode(InputMode::Normal);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('c'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    self.textarea.cut();
                    self.set_mode(InputMode::Insert);
                    true
                }
                KeyEvent {
                    code: KeyCode::Char('p'),
                    modifiers: KeyModifiers::NONE,
                    ..
                } => {
                    let saved = self.textarea.yank_text();
                    self.textarea.cut();
                    self.textarea.set_yank_text(saved);
                    self.textarea.paste();
                    self.set_mode(InputMode::Normal);
                    true
                }
                KeyEvent {
                    code: KeyCode::Esc, ..
                } => {
                    self.set_mode(InputMode::Normal);
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn lines(&self) -> &[String] {
        self.textarea.lines()
    }

    pub fn is_empty(&self) -> bool {
        self.textarea.is_empty()
    }

    pub fn text(&self) -> String {
        self.textarea.lines().join("\n")
    }

    pub fn set_text(&mut self, text: &str) {
        let lines: Vec<String> = if text.is_empty() {
            vec![String::new()]
        } else {
            text.lines().map(String::from).collect()
        };
        self.textarea = TextArea::new(lines);
        self.textarea.set_wrap_mode(WrapMode::WordOrGlyph);
        self.textarea
            .set_cursor_line_style(Style::default().add_modifier(Modifier::empty()));
        self.apply_block();
    }

    pub fn clear(&mut self) {
        self.textarea.clear();
    }

    pub fn insert_paste(&mut self, text: &str) {
        self.textarea.insert_str(text);
    }

    pub fn set_mode(&mut self, mode: InputMode) {
        if self.mode == InputMode::Visual {
            self.textarea.cancel_selection();
        }
        self.mode = mode;
        self.apply_block();
    }

    pub fn mode(&self) -> &InputMode {
        &self.mode
    }

    fn apply_block(&mut self) {
        let block = match &self.mode {
            InputMode::Insert => Block::default().borders(Borders::ALL).title(INSERT_TITLE),
            InputMode::Normal => Block::default().borders(Borders::ALL).title(NORMAL_TITLE),
            InputMode::Visual => Block::default().borders(Borders::ALL).title(VISUAL_TITLE),
            InputMode::Streaming => Block::default()
                .borders(Borders::ALL)
                .title(STREAMING_TITLE),
            InputMode::SessionPicker => {
                Block::default().borders(Borders::ALL).title(SESSIONS_TITLE)
            }
            InputMode::TasksPicker => Block::default().borders(Borders::ALL).title(TASKS_TITLE),
            InputMode::ToolConfirmation { name, .. } => Block::default()
                .borders(Borders::ALL)
                .title(format!("Allow '{name}'? [y/n]")),
        };
        self.textarea.set_block(block);
    }

    pub fn height_for_width(&self, width: u16, available_height: u16) -> u16 {
        let max_height = (available_height / MAX_INPUT_RATIO).max(MIN_HEIGHT);
        match &self.mode {
            InputMode::ToolConfirmation { name, input } => {
                let confirmation_text = format!("Allow '{}' with input {}?", name, input);
                let lines_needed = confirmation_text.lines().count() as u16;
                lines_needed
                    .saturating_add(2)
                    .max(MIN_HEIGHT)
                    .min(max_height)
            }
            InputMode::Streaming | InputMode::SessionPicker | InputMode::TasksPicker => MIN_HEIGHT,
            InputMode::Insert | InputMode::Normal | InputMode::Visual => {
                self.text_height_for_width(width, max_height)
            }
        }
    }

    fn text_height_for_width(&self, width: u16, max_height: u16) -> u16 {
        if width == 0 {
            return MIN_HEIGHT;
        }
        let inner_width = width.saturating_sub(2) as usize;
        if inner_width == 0 {
            return MIN_HEIGHT;
        }
        let lines = self.textarea.lines();
        if lines.is_empty() || lines.iter().all(|l| l.is_empty()) {
            return MIN_HEIGHT;
        }
        let total_visual_lines: usize = lines
            .iter()
            .map(|line| {
                if line.is_empty() {
                    1
                } else {
                    let width_val = unicode_width::UnicodeWidthStr::width(line.as_str());
                    width_val.div_ceil(inner_width)
                }
            })
            .sum();
        let needed = (total_visual_lines as u16).saturating_add(2);
        needed.clamp(MIN_HEIGHT, max_height)
    }

    pub fn render(&self, frame: &mut ratatui::Frame, area: Rect) {
        frame.render_widget(&self.textarea, area);
    }

    #[cfg(test)]
    pub(super) fn cursor(&self) -> DataCursor {
        self.textarea.cursor()
    }
}

impl Default for InputArea<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn char_key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    #[test]
    fn new_input_area_is_empty() {
        let input = InputArea::new();
        assert!(input.is_empty());
    }

    #[test]
    fn typing_characters_adds_to_content() {
        let mut input = InputArea::new();
        input.input(char_key('h'));
        input.input(char_key('i'));
        assert_eq!(input.text(), "hi");
        assert!(!input.is_empty());
    }

    #[test]
    fn clear_empties_content() {
        let mut input = InputArea::new();
        input.input(char_key('a'));
        input.input(char_key('b'));
        input.input(char_key('c'));
        assert_eq!(input.text(), "abc");
        input.clear();
        assert!(input.is_empty());
    }

    #[test]
    fn backspace_removes_last_character() {
        let mut input = InputArea::new();
        input.input(char_key('x'));
        input.input(char_key('y'));
        input.input(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(input.text(), "x");
    }

    #[test]
    fn enter_key_inserts_newline() {
        let mut input = InputArea::new();
        input.input(char_key('a'));
        input.input(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        input.input(char_key('b'));
        assert_eq!(input.text(), "a\nb");
        assert_eq!(input.lines().len(), 2);
    }

    #[test]
    fn set_text_replaces_content() {
        let mut input = InputArea::new();
        input.input(char_key('o'));
        input.input(char_key('l'));
        input.input(char_key('d'));
        input.set_text("new text");
        assert_eq!(input.text(), "new text");
    }

    #[test]
    fn set_text_with_multiline() {
        let mut input = InputArea::new();
        input.set_text("line1\nline2\nline3");
        assert_eq!(input.text(), "line1\nline2\nline3");
        assert_eq!(input.lines().len(), 3);
    }

    #[test]
    fn set_text_empty_clears() {
        let mut input = InputArea::new();
        input.input(char_key('a'));
        input.set_text("");
        assert!(input.is_empty());
    }

    #[test]
    fn height_for_width_minimum_when_empty() {
        let input = InputArea::new();
        assert_eq!(input.height_for_width(60, 24), MIN_HEIGHT);
    }

    #[test]
    fn height_for_width_minimum_when_short_text() {
        let mut input = InputArea::new();
        input.input(char_key('h'));
        input.input(char_key('i'));
        assert_eq!(input.height_for_width(60, 24), MIN_HEIGHT);
    }

    #[test]
    fn height_for_width_grows_with_wrapping() {
        let mut input = InputArea::new();
        for c in "abcdefghijklmnopqrstuvwxyz".chars() {
            input.input(char_key(c));
        }
        let wide = input.height_for_width(60, 24);
        let narrow = input.height_for_width(10, 24);
        assert!(
            narrow > wide,
            "narrow={narrow} should be > wide={wide} for wrapping text"
        );
    }

    #[test]
    fn height_for_width_respects_available_height() {
        let mut input = InputArea::new();
        for c in "abcdefghijklmnopqrstuvwxyz".chars() {
            input.input(char_key(c));
        }
        let available = 24u16;
        let height = input.height_for_width(4, available);
        assert!(
            height <= available / MAX_INPUT_RATIO,
            "height {height} should be <= available/{MAX_INPUT_RATIO} = {}",
            available / MAX_INPUT_RATIO
        );
    }

    #[test]
    fn height_for_width_respects_min() {
        let input = InputArea::new();
        let height = input.height_for_width(4, 24);
        assert!(
            height >= MIN_HEIGHT,
            "height {height} should be >= {MIN_HEIGHT}"
        );
    }

    #[test]
    fn height_for_width_zero_width_returns_min() {
        let input = InputArea::new();
        assert_eq!(input.height_for_width(0, 24), MIN_HEIGHT);
    }

    #[test]
    fn text_returns_joined_lines() {
        let mut input = InputArea::new();
        input.input(char_key('a'));
        input.input(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        input.input(char_key('b'));
        assert_eq!(input.text(), "a\nb");
    }

    #[test]
    fn render_produces_widget_output() {
        let mut input = InputArea::new();
        input.input(char_key('h'));
        input.input(char_key('e'));
        input.input(char_key('l'));
        input.input(char_key('l'));
        input.input(char_key('o'));

        let backend = ratatui::backend::TestBackend::new(60, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 60, MIN_HEIGHT);
                input.render(frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("render_hello", terminal.backend());
    }

    #[test]
    fn render_wrapped_text_grows_height() {
        let mut input = InputArea::new();
        for c in "abcdefghijklmnopqrstuvwxyz0123456789".chars() {
            input.input(char_key(c));
        }

        let backend = ratatui::backend::TestBackend::new(20, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let height = input.height_for_width(20, 24);
        assert!(height > MIN_HEIGHT, "should need more than min height");

        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 20, height);
                input.render(frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("render_wrapped_text", terminal.backend());
    }

    #[test]
    fn render_displays_border_and_title() {
        let mut input = InputArea::new();
        input.input(char_key('h'));
        input.input(char_key('i'));

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("render_with_block", terminal.backend());
    }

    #[test]
    fn height_for_width_grows_with_multiline_input() {
        let mut input = InputArea::new();
        input.set_text("line1\nline2\nline3\nline4\nline5");
        let height = input.height_for_width(60, 24);
        assert!(
            height > MIN_HEIGHT,
            "multiline input should need more than {MIN_HEIGHT} height, got {height}"
        );
    }

    #[test]
    fn height_for_width_grows_with_multiline_and_wrapping() {
        let mut input = InputArea::new();
        input.set_text("a very long line that needs to wrap\nanother long line that wraps");
        let narrow_height = input.height_for_width(20, 24);
        let wide_height = input.height_for_width(80, 24);
        assert!(
            narrow_height > wide_height,
            "narrow={narrow_height} should be > wide={wide_height}"
        );
    }

    #[test]
    fn set_mode_streaming_updates_title() {
        let mut input = InputArea::new();
        input.set_mode(InputMode::Streaming);

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("render_streaming", terminal.backend());
    }

    #[test]
    fn set_mode_tool_confirmation_updates_title() {
        let mut input = InputArea::new();
        input.set_mode(InputMode::ToolConfirmation {
            name: "write_file".to_string(),
            input: serde_json::json!({"path": "/tmp/test.txt"}),
        });

        let backend = ratatui::backend::TestBackend::new(60, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let height = input.height_for_width(60, 24);
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 60, height);
                input.render(frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("render_tool_confirmation", terminal.backend());
    }

    #[test]
    fn set_mode_back_to_input_restores_input_title() {
        let mut input = InputArea::new();
        input.input(char_key('h'));
        input.set_mode(InputMode::Streaming);
        input.set_mode(InputMode::Insert);

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("render_restored_input", terminal.backend());
    }

    #[test]
    fn esc_switches_to_normal_mode() {
        let mut input = InputArea::new();
        input.input(char_key('h'));
        input.input(char_key('i'));
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(input.mode(), &InputMode::Normal);
        assert_eq!(input.text(), "hi");
    }

    #[test]
    fn i_returns_to_insert_mode() {
        let mut input = InputArea::new();
        input.set_mode(InputMode::Normal);
        input.input(char_key('i'));
        assert_eq!(input.mode(), &InputMode::Insert);
    }

    #[test]
    fn normal_mode_ignores_typing() {
        let mut input = InputArea::new();
        input.input(char_key('a'));
        input.set_mode(InputMode::Normal);
        input.input(char_key('x'));
        input.input(char_key('y'));
        assert_eq!(input.text(), "a");
    }

    #[test]
    fn o_opens_line_below_and_enters_insert() {
        let mut input = InputArea::new();
        input.set_text("first");
        input.set_mode(InputMode::Normal);
        input.input(char_key('o'));
        assert_eq!(input.mode(), &InputMode::Insert);
        assert_eq!(input.lines().len(), 2);
        assert_eq!(input.lines()[0], "first");
        assert_eq!(input.lines()[1], "");
    }

    #[test]
    fn upper_o_opens_line_above_and_enters_insert() {
        let mut input = InputArea::new();
        input.set_text("first");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Char('O'), KeyModifiers::SHIFT));
        assert_eq!(input.mode(), &InputMode::Insert);
        assert_eq!(input.lines().len(), 2);
        assert_eq!(input.lines()[0], "");
        assert_eq!(input.lines()[1], "first");
    }

    #[test]
    fn render_normal_mode_shows_title() {
        let mut input = InputArea::new();
        input.input(char_key('h'));
        input.input(char_key('i'));
        input.set_mode(InputMode::Normal);

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("render_normal_mode", terminal.backend());
    }

    #[test]
    fn mode_defaults_to_input() {
        let input = InputArea::new();
        assert_eq!(input.mode(), &InputMode::Insert);
    }

    #[test]
    fn height_for_width_streaming_returns_min() {
        let mut input = InputArea::new();
        input.set_mode(InputMode::Streaming);
        assert_eq!(input.height_for_width(60, 24), MIN_HEIGHT);
    }

    #[test]
    fn height_for_width_tool_confirmation_scales_with_content() {
        let mut input = InputArea::new();
        input.set_mode(InputMode::ToolConfirmation {
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls -la /some/long/path"}),
        });
        let height = input.height_for_width(60, 24);
        assert!(height >= MIN_HEIGHT);
    }

    #[test]
    fn normal_mode_w_moves_cursor_word_forward() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        let consumed = input.input(char_key('w'));
        assert!(consumed, "w should be consumed in Normal mode");
        assert_eq!(
            input.cursor().1,
            6,
            "w from col 0 on 'hello world' should land at col 6 (start of 'world')"
        );
    }

    #[test]
    fn normal_mode_b_moves_cursor_word_back() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('w')); // now at col 6
        let consumed = input.input(char_key('b'));
        assert!(consumed, "b should be consumed in Normal mode");
        assert_eq!(
            input.cursor().1,
            0,
            "b from col 6 on 'hello world' should return to col 0"
        );
    }

    #[test]
    fn normal_mode_e_moves_cursor_word_end() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        let consumed = input.input(char_key('e'));
        assert!(consumed, "e should be consumed in Normal mode");
        assert_eq!(
            input.cursor().1,
            4,
            "e from col 0 on 'hello world' should land at col 4 (end of 'hello')"
        );
    }

    #[test]
    fn insert_mode_w_inserts_literal() {
        let mut input = InputArea::new();
        assert_eq!(input.mode(), &InputMode::Insert);
        input.input(char_key('w'));
        assert_eq!(
            input.text(),
            "w",
            "w in Insert mode should insert literal 'w'"
        );
    }

    #[test]
    fn insert_mode_b_inserts_literal() {
        let mut input = InputArea::new();
        input.input(char_key('b'));
        assert_eq!(
            input.text(),
            "b",
            "b in Insert mode should insert literal 'b'"
        );
    }

    #[test]
    fn insert_mode_e_inserts_literal() {
        let mut input = InputArea::new();
        input.input(char_key('e'));
        assert_eq!(
            input.text(),
            "e",
            "e in Insert mode should insert literal 'e'"
        );
    }

    // --- Visual mode and new Normal keybinding tests ---

    #[test]
    fn normal_mode_0_moves_to_line_start() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(char_key('w'));
        let consumed = input.input(char_key('0'));
        assert!(consumed, "0 should be consumed in Normal mode");
        assert_eq!(input.cursor().1, 0, "0 should move cursor to line start");
    }

    #[test]
    fn normal_mode_dollar_moves_to_line_end() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        let consumed = input.input(KeyEvent::new(KeyCode::Char('$'), KeyModifiers::NONE));
        assert!(consumed, "$ should be consumed in Normal mode");
        assert!(
            input.cursor().1 >= 10,
            "$ should move cursor near line end, got col {}",
            input.cursor().1
        );
    }

    #[test]
    fn normal_mode_v_enters_visual() {
        let mut input = InputArea::new();
        input.set_text("hello");
        input.set_mode(InputMode::Normal);
        let consumed = input.input(char_key('v'));
        assert!(consumed, "v should be consumed in Normal mode");
        assert_eq!(
            input.mode(),
            &InputMode::Visual,
            "v should enter Visual mode"
        );
    }

    #[test]
    fn normal_mode_p_pastes_register() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(char_key('v'));
        input.input(char_key('l'));
        input.input(char_key('d'));
        assert_eq!(input.mode(), &InputMode::Normal);
        let consumed = input.input(char_key('p'));
        assert!(consumed, "p should be consumed in Normal mode");
        assert!(
            input.text().contains("h"),
            "p should paste from register, got '{}'",
            input.text()
        );
    }

    #[test]
    fn normal_mode_h_moves_left() {
        let mut input = InputArea::new();
        input.set_text("hello");
        input.set_mode(InputMode::Normal);
        input.input(char_key('l'));
        input.input(char_key('l'));
        assert_eq!(
            input.cursor().1,
            2,
            "should start at col 2 after two l presses"
        );
        let consumed = input.input(char_key('h'));
        assert!(consumed, "h should be consumed in Normal mode");
        assert_eq!(input.cursor().1, 1, "h should move cursor left by one");
    }

    #[test]
    fn normal_mode_l_moves_right() {
        let mut input = InputArea::new();
        input.set_text("hello");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        let consumed = input.input(char_key('l'));
        assert!(consumed, "l should be consumed in Normal mode");
        assert_eq!(input.cursor().1, 1, "l should move cursor right by one");
    }

    #[test]
    fn normal_mode_j_moves_down() {
        let mut input = InputArea::new();
        input.set_text("line1\nline2");
        input.set_mode(InputMode::Normal);
        let start_row = input.cursor().0;
        let consumed = input.input(char_key('j'));
        assert!(consumed, "j should be consumed in Normal mode");
        assert_eq!(
            input.cursor().0,
            start_row + 1,
            "j should move cursor down one line"
        );
    }

    #[test]
    fn normal_mode_k_moves_up() {
        let mut input = InputArea::new();
        input.set_text("line1\nline2");
        input.set_mode(InputMode::Normal);
        input.input(char_key('j'));
        let row_after_j = input.cursor().0;
        assert!(row_after_j > 0, "should be on row > 0 after j");
        let consumed = input.input(char_key('k'));
        assert!(consumed, "k should be consumed in Normal mode");
        assert_eq!(
            input.cursor().0,
            row_after_j - 1,
            "k should move cursor up one line"
        );
    }

    #[test]
    fn visual_mode_motions_extend_selection() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        assert_eq!(input.mode(), &InputMode::Visual);
        let consumed = input.input(char_key('l'));
        assert!(consumed, "l should be consumed in Visual mode");
        assert_eq!(input.cursor().1, 1, "l should move cursor to col 1");
        let consumed = input.input(char_key('l'));
        assert!(consumed, "second l should be consumed in Visual mode");
        assert_eq!(input.cursor().1, 2, "second l should move cursor to col 2");
    }

    #[test]
    fn visual_mode_d_deletes_and_returns_normal() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        input.input(char_key('l'));
        input.input(char_key('l'));
        let consumed = input.input(char_key('d'));
        assert!(consumed, "d should be consumed in Visual mode");
        assert_eq!(
            input.mode(),
            &InputMode::Normal,
            "d should return to Normal mode"
        );
        assert_eq!(input.text(), "llo world", "d should delete selected text");
    }

    #[test]
    fn visual_mode_c_deletes_and_returns_insert() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        input.input(char_key('l'));
        input.input(char_key('l'));
        let consumed = input.input(char_key('c'));
        assert!(consumed, "c should be consumed in Visual mode");
        assert_eq!(
            input.mode(),
            &InputMode::Insert,
            "c should enter Insert mode"
        );
        assert_eq!(input.text(), "llo world", "c should delete selected text");
    }

    #[test]
    fn visual_mode_p_replaces_selection() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        input.input(char_key('l'));
        input.input(char_key('l'));
        input.input(char_key('d'));
        assert_eq!(input.mode(), &InputMode::Normal);
        assert_eq!(input.text(), "llo world");

        input.set_mode(InputMode::Normal);
        input.input(char_key('$'));
        input.input(char_key('v'));
        input.input(char_key('l'));
        let consumed = input.input(char_key('p'));
        assert!(consumed, "p should be consumed in Visual mode");
        assert_eq!(
            input.mode(),
            &InputMode::Normal,
            "p should return to Normal mode"
        );
        assert!(
            input.text().contains("he"),
            "p should paste register content, got '{}'",
            input.text()
        );
    }

    #[test]
    fn visual_mode_esc_cancels_and_returns_normal() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        input.input(char_key('l'));
        assert_eq!(input.mode(), &InputMode::Visual);
        let consumed = input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(consumed, "Esc should be consumed in Visual mode");
        assert_eq!(
            input.mode(),
            &InputMode::Normal,
            "Esc should return to Normal mode"
        );
        assert_eq!(input.text(), "hello world", "Esc should not modify text");
    }

    #[test]
    fn set_mode_from_visual_cancels_selection() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        assert_eq!(input.mode(), &InputMode::Visual);
        input.set_mode(InputMode::Insert);
        assert_eq!(input.mode(), &InputMode::Insert);
        assert_eq!(
            input.text(),
            "hello world",
            "transitioning out of Visual should not modify text"
        );
    }

    #[test]
    fn render_visual_mode() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        input.input(char_key('l'));

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("render_visual_mode", terminal.backend());
    }

    #[test]
    fn visual_mode_w_moves_word_forward() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        let consumed = input.input(char_key('w'));
        assert!(consumed, "w should be consumed in Visual mode");
        assert_eq!(
            input.cursor().1,
            6,
            "w in Visual should move to start of 'world'"
        );
    }

    #[test]
    fn visual_mode_b_moves_word_back() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        input.input(char_key('w'));
        assert_eq!(input.cursor().1, 6);
        let consumed = input.input(char_key('b'));
        assert!(consumed, "b should be consumed in Visual mode");
        assert_eq!(input.cursor().1, 0, "b in Visual should move back to start");
    }

    #[test]
    fn visual_mode_e_moves_word_end() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        let consumed = input.input(char_key('e'));
        assert!(consumed, "e should be consumed in Visual mode");
        assert_eq!(
            input.cursor().1,
            4,
            "e in Visual should move to end of 'hello'"
        );
    }

    #[test]
    fn visual_mode_0_moves_to_line_start() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        input.input(char_key('l'));
        assert!(input.cursor().1 > 0);
        let consumed = input.input(char_key('0'));
        assert!(consumed, "0 should be consumed in Visual mode");
        assert_eq!(
            input.cursor().1,
            0,
            "0 in Visual should move cursor to line start"
        );
    }

    #[test]
    fn visual_mode_dollar_moves_to_line_end() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        let consumed = input.input(KeyEvent::new(KeyCode::Char('$'), KeyModifiers::NONE));
        assert!(consumed, "$ should be consumed in Visual mode");
        assert!(
            input.cursor().1 >= 10,
            "$ in Visual should move cursor near line end, got col {}",
            input.cursor().1
        );
    }

    #[test]
    fn visual_mode_j_moves_down() {
        let mut input = InputArea::new();
        input.set_text("line1\nline2");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        let start_row = input.cursor().0;
        let consumed = input.input(char_key('j'));
        assert!(consumed, "j should be consumed in Visual mode");
        assert_eq!(
            input.cursor().0,
            start_row + 1,
            "j in Visual should move cursor down one line"
        );
    }

    #[test]
    fn visual_mode_k_moves_up() {
        let mut input = InputArea::new();
        input.set_text("line1\nline2");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        input.input(char_key('v'));
        input.input(char_key('j'));
        let row_after_j = input.cursor().0;
        assert!(row_after_j > 0);
        let consumed = input.input(char_key('k'));
        assert!(consumed, "k should be consumed in Visual mode");
        assert_eq!(
            input.cursor().0,
            row_after_j - 1,
            "k in Visual should move cursor up one line"
        );
    }

    #[test]
    fn visual_mode_unrecognized_key_not_consumed() {
        let mut input = InputArea::new();
        input.set_text("hello");
        input.set_mode(InputMode::Normal);
        input.input(char_key('v'));
        let consumed = input.input(char_key('x'));
        assert!(
            !consumed,
            "unrecognized key in Visual mode should not be consumed"
        );
    }

    #[test]
    fn normal_mode_p_empty_register_noop() {
        let mut input = InputArea::new();
        input.set_text("hello");
        input.set_mode(InputMode::Normal);
        input.input(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        let consumed = input.input(char_key('p'));
        assert!(
            consumed,
            "p should be consumed in Normal mode even with empty register"
        );
        assert_eq!(
            input.text(),
            "hello",
            "p with empty register should not change text"
        );
    }
}
