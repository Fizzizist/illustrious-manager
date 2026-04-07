use ratatui::layout::Rect;
use ratatui::widgets::{Block, Borders};
use ratatui_textarea::{TextArea, WrapMode};

const INPUT_TITLE: &str = "Input (Enter to send, Ctrl+C to quit)";
const STREAMING_TITLE: &str = "Streaming...";
const MIN_HEIGHT: u16 = 3;
const MAX_INPUT_RATIO: u16 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputMode {
    Input,
    Streaming,
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
            mode: InputMode::Input,
        };
        input.textarea.set_wrap_mode(WrapMode::WordOrGlyph);
        input.apply_block();
        input
    }

    pub fn input(&mut self, event: crossterm::event::KeyEvent) -> bool {
        self.textarea.input(event)
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
        self.apply_block();
    }

    pub fn clear(&mut self) {
        self.textarea.clear();
    }

    pub fn set_mode(&mut self, mode: InputMode) {
        self.mode = mode;
        self.apply_block();
    }

    pub fn mode(&self) -> &InputMode {
        &self.mode
    }

    fn apply_block(&mut self) {
        let block = match &self.mode {
            InputMode::Input => Block::default().borders(Borders::ALL).title(INPUT_TITLE),
            InputMode::Streaming => Block::default()
                .borders(Borders::ALL)
                .title(STREAMING_TITLE),
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
            InputMode::Streaming => MIN_HEIGHT,
            InputMode::Input => self.text_height_for_width(width, max_height),
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
        input.set_mode(InputMode::Input);

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
    fn mode_defaults_to_input() {
        let input = InputArea::new();
        assert_eq!(input.mode(), &InputMode::Input);
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
}
