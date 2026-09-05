use crossterm::event::KeyEvent;
use hjkl_buffer::{
    Viewport, Wrap,
    wrap::{segment_for_col, wrap_segments},
};
use hjkl_editor_tui::crossterm_key_event_to_input;
use hjkl_engine::{Host, decode_planned_input};
use hjkl_form::{TextFieldEditor, VimMode};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use unicode_width::UnicodeWidthStr;

const INSERT_TITLE: &str = " -- INSERT -- ";
const NORMAL_TITLE: &str = " -- NORMAL -- ";
const VISUAL_TITLE: &str = " -- VISUAL -- ";
const VISUAL_LINE_TITLE: &str = " -- VISUAL LINE -- ";
const VISUAL_BLOCK_TITLE: &str = " -- VISUAL BLOCK -- ";
const SESSIONS_TITLE: &str = " Sessions ";
const TASKS_TITLE: &str = " Tasks ";
const MIN_HEIGHT: u16 = 3;
const MAX_INPUT_RATIO: u16 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppMode {
    Editing,
    Streaming,
    SessionPicker,
    TasksPicker,
    ToolConfirmation {
        name: String,
        input: serde_json::Value,
    },
    Compacting,
    RunningBash,
}

pub struct InputArea {
    editor: TextFieldEditor,
    mode: AppMode,
    elapsed: Option<std::time::Duration>,
}

impl InputArea {
    pub fn new() -> Self {
        let editor = TextFieldEditor::new(false);
        let mut input = Self {
            editor,
            mode: AppMode::Editing,
            elapsed: None,
        };
        input.editor.enter_insert_at_end();
        input
    }

    pub fn input(&mut self, event: KeyEvent) -> bool {
        match self.mode {
            AppMode::Editing => {}
            _ => return false,
        }

        let planned = crossterm_key_event_to_input(event);
        if let Some(input) = decode_planned_input(planned) {
            self.editor.handle_input(input);
            return true;
        }
        false
    }

    pub fn is_empty(&self) -> bool {
        self.editor.text().is_empty()
    }

    pub fn text(&self) -> String {
        self.editor.text()
    }

    pub fn set_text(&mut self, text: &str) {
        self.editor.set_text(text);
    }

    pub fn clear(&mut self) {
        self.editor.set_text("");
    }

    pub fn insert_paste(&mut self, text: &str) {
        // TextFieldEditor exposes no paste method; insert_str is the
        // engine's public insertion API and calls mark_content_dirty().
        self.editor.editor.insert_str(text);
    }

    pub fn set_mode(&mut self, mode: AppMode) {
        self.mode = mode;
    }

    pub fn reset_to_insert(&mut self) {
        self.editor.enter_insert_at_end();
    }

    pub fn mode(&self) -> &AppMode {
        &self.mode
    }

    pub fn vim_mode(&self) -> VimMode {
        self.editor.vim_mode()
    }

    pub fn is_normal(&self) -> bool {
        self.editor.vim_mode() == VimMode::Normal
    }

    pub fn set_elapsed_title(&mut self, duration: std::time::Duration) {
        self.elapsed = Some(duration);
    }

    pub fn height_for_width(&self, width: u16, available_height: u16) -> u16 {
        let max_height = (available_height / MAX_INPUT_RATIO).max(MIN_HEIGHT);
        match &self.mode {
            AppMode::ToolConfirmation { name, input } => {
                let confirmation_text = format!("Allow '{}' with input {}?", name, input);
                let lines_needed = confirmation_text.lines().count() as u16;
                lines_needed
                    .saturating_add(2)
                    .max(MIN_HEIGHT)
                    .min(max_height)
            }
            AppMode::Streaming
            | AppMode::SessionPicker
            | AppMode::TasksPicker
            | AppMode::Compacting
            | AppMode::RunningBash => MIN_HEIGHT,
            AppMode::Editing => self.text_height_for_width(width, max_height),
        }
    }

    fn text_height_for_width(&self, width: u16, max_height: u16) -> u16 {
        if width == 0 {
            return MIN_HEIGHT;
        }
        let text_width = width.saturating_sub(2);
        if text_width == 0 {
            return MIN_HEIGHT;
        }
        let viewport = Viewport {
            wrap: Wrap::Word,
            text_width,
            ..Viewport::default()
        };
        let buffer = self.editor.buffer();
        let row_count = buffer.row_count();
        if row_count == 0 {
            return MIN_HEIGHT;
        }
        let screen_rows = buffer.screen_rows_between(&viewport, 0, row_count.saturating_sub(1));
        let content_lines = screen_rows
            .min(u16::MAX as usize)
            .min(max_height.saturating_sub(2) as usize) as u16;
        content_lines
            .saturating_add(2)
            .clamp(MIN_HEIGHT, max_height)
    }

    pub fn render(
        &mut self,
        frame: &mut ratatui::Frame,
        area: Rect,
        disabled: bool,
        interrupt_requested: bool,
    ) {
        let block = self.build_block(area, disabled, interrupt_requested);
        let inner = block.inner(area);

        if inner.width == 0 || inner.height == 0 {
            frame.render_widget(block, area);
            return;
        }

        let text_width = inner.width;
        {
            let v = self.editor.editor.host_mut().viewport_mut();
            v.wrap = Wrap::Word;
            v.text_width = text_width;
            v.width = inner.width;
            v.height = inner.height;
        }
        self.editor.editor.set_viewport_height(inner.height);
        self.editor.editor.ensure_cursor_in_scrolloff();
        let mut viewport = *self.editor.editor.host().viewport();
        self.editor
            .editor
            .buffer_mut()
            .ensure_cursor_visible(&mut viewport);
        *self.editor.editor.host_mut().viewport_mut() = viewport;

        let buffer = self.editor.buffer();
        let text = self.editor.text();
        let row_count = buffer.row_count();

        let lines: Vec<Line> = if row_count == 0 {
            vec![Line::from("")]
        } else {
            let selection = self.editor.editor.selection_highlight();
            let line_texts: Vec<&str> = text.split('\n').collect();

            (0..row_count)
                .flat_map(|row| {
                    let line_text = line_texts.get(row).copied().unwrap_or("");
                    let segments = wrap_segments(line_text, text_width, Wrap::Word);
                    match &selection {
                        Some(sel) => {
                            let sel_start = sel.range.start.line as usize;
                            let sel_end = sel.range.end.line as usize;
                            if row >= sel_start && row <= sel_end {
                                let col_start = if row == sel_start {
                                    sel.range.start.col as usize
                                } else {
                                    0
                                };
                                let col_end = if row == sel_end {
                                    (sel.range.end.col as usize).saturating_add(1)
                                } else {
                                    line_text.chars().count()
                                };
                                make_selection_lines(line_text, &segments, col_start, col_end)
                            } else {
                                lines_from_segments(line_text, &segments)
                            }
                        }
                        None => lines_from_segments(line_text, &segments),
                    }
                })
                .collect()
        };

        let scroll_y = if viewport.top_row == 0 {
            0
        } else {
            buffer.screen_rows_between(&viewport, 0, viewport.top_row.saturating_sub(1)) as u16
        };
        let paragraph = Paragraph::new(lines).block(block).scroll((scroll_y, 0));
        frame.render_widget(paragraph, area);

        if let Some((x, y)) = cursor_xy(&self.editor, inner, text_width, scroll_y) {
            frame.set_cursor_position(ratatui::layout::Position { x, y });
        }
    }

    fn build_block(
        &self,
        _area: Rect,
        disabled: bool,
        interrupt_requested: bool,
    ) -> Block<'static> {
        let title = match &self.mode {
            AppMode::Editing => match self.editor.vim_mode() {
                VimMode::Normal => NORMAL_TITLE,
                VimMode::Insert => INSERT_TITLE,
                VimMode::Visual => VISUAL_TITLE,
                VimMode::VisualLine => VISUAL_LINE_TITLE,
                VimMode::VisualBlock => VISUAL_BLOCK_TITLE,
            }
            .to_string(),
            AppMode::Streaming => elapsed_title(
                self.elapsed,
                if interrupt_requested {
                    "Cancelling... (Esc to interrupt)"
                } else {
                    "Streaming... (Esc to interrupt)"
                },
            ),
            AppMode::Compacting => elapsed_title(self.elapsed, "Compacting..."),
            AppMode::RunningBash => elapsed_title(
                self.elapsed,
                if interrupt_requested {
                    "Cancelling... (Esc to cancel)"
                } else {
                    "Running bash... (Esc to cancel)"
                },
            ),
            AppMode::SessionPicker => SESSIONS_TITLE.to_string(),
            AppMode::TasksPicker => TASKS_TITLE.to_string(),
            AppMode::ToolConfirmation { name, .. } => {
                format!(" Allow '{}'? [y/n] ", name)
            }
        };

        let border_style = if disabled {
            Style::default().fg(ratatui::style::Color::DarkGray)
        } else {
            Style::default()
        };
        Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .title(title)
    }
}

impl Default for InputArea {
    fn default() -> Self {
        Self::new()
    }
}

fn elapsed_title(elapsed: Option<std::time::Duration>, label: &str) -> String {
    match elapsed {
        Some(d) => format!("{} {}", crate::timestamp::format_elapsed(d), label),
        None => format!(" {} ", label),
    }
}

fn char_slice(line_text: &str, start: usize, end: usize) -> String {
    line_text
        .chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

fn make_selection_lines(
    line_text: &str,
    segments: &[(usize, usize)],
    col_start: usize,
    col_end: usize,
) -> Vec<Line<'static>> {
    segments
        .iter()
        .map(|&(seg_start, seg_end)| {
            let seg_len = seg_end.saturating_sub(seg_start);
            let local_sel_start = col_start.saturating_sub(seg_start).min(seg_len);
            let local_sel_end = col_end.saturating_sub(seg_start).min(seg_len);

            if local_sel_start == 0 && local_sel_end >= seg_len {
                Line::from(Span::styled(
                    char_slice(line_text, seg_start, seg_end),
                    Style::default().add_modifier(Modifier::REVERSED),
                ))
            } else if local_sel_end <= local_sel_start {
                Line::from(Span::raw(char_slice(line_text, seg_start, seg_end)))
            } else {
                let mut spans: Vec<Span<'static>> = Vec::new();
                let before = char_slice(line_text, seg_start, seg_start + local_sel_start);
                if !before.is_empty() {
                    spans.push(Span::raw(before));
                }
                let selected = char_slice(
                    line_text,
                    seg_start + local_sel_start,
                    seg_start + local_sel_end,
                );
                if !selected.is_empty() {
                    spans.push(Span::styled(
                        selected,
                        Style::default().add_modifier(Modifier::REVERSED),
                    ));
                }
                let after = char_slice(line_text, seg_start + local_sel_end, seg_end);
                if !after.is_empty() {
                    spans.push(Span::raw(after));
                }
                Line::from(spans)
            }
        })
        .collect()
}

fn lines_from_segments(line_text: &str, segments: &[(usize, usize)]) -> Vec<Line<'static>> {
    if segments.is_empty() {
        vec![Line::from("")]
    } else {
        segments
            .iter()
            .map(|&(start, end)| Line::from(Span::raw(char_slice(line_text, start, end))))
            .collect()
    }
}

fn cursor_xy(
    editor: &TextFieldEditor,
    rect: Rect,
    text_width: u16,
    scroll_y: u16,
) -> Option<(u16, u16)> {
    let viewport = editor.editor.host().viewport();
    let buffer = editor.buffer();
    let (row, col) = editor.cursor();

    let text = editor.text();
    let line_text = text.split('\n').nth(row)?;
    let segments = wrap_segments(line_text, text_width, Wrap::Word);
    let seg_idx = segment_for_col(&segments, col);
    let &(seg_start, _seg_end) = segments.get(seg_idx)?;

    let prefix: String = line_text
        .chars()
        .skip(seg_start)
        .take(col.saturating_sub(seg_start))
        .collect();
    let dx = UnicodeWidthStr::width(prefix.as_str()) as u16;

    let dy = if row == 0 {
        seg_idx as u16
    } else {
        let rows_before = buffer.screen_rows_between(viewport, 0, row.saturating_sub(1));
        (rows_before.min(u16::MAX as usize) as u16).saturating_add(seg_idx as u16)
    };

    // take scroll offset into account
    let terminal_dy = dy.saturating_sub(scroll_y);
    if terminal_dy >= rect.height {
        return None;
    }

    Some((
        rect.x.saturating_add(dx),
        rect.y.saturating_add(terminal_dy),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyModifiers};

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
    fn enter_in_insert_mode_inserts_newline() {
        let mut input = InputArea::new();
        input.input(char_key('a'));
        input.input(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        input.input(char_key('b'));
        assert_eq!(input.text(), "a\nb");
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
                input.render(frame, area, true, false);
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
                input.render(frame, area, true, false);
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
                input.render(frame, area, true, false);
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
        input.set_mode(AppMode::Streaming);

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        insta::assert_snapshot!("render_streaming", terminal.backend());
    }

    #[test]
    fn render_streaming_with_elapsed() {
        let mut input = InputArea::new();
        input.set_mode(AppMode::Streaming);
        input.set_elapsed_title(std::time::Duration::from_secs(55));

        let backend = ratatui::backend::TestBackend::new(60, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 60, MIN_HEIGHT);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        insta::assert_snapshot!("render_streaming_with_elapsed", terminal.backend());
    }

    #[test]
    fn render_cancelling() {
        let mut input = InputArea::new();
        input.set_mode(AppMode::Streaming);
        input.set_elapsed_title(std::time::Duration::from_secs(55));

        let backend = ratatui::backend::TestBackend::new(60, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 60, MIN_HEIGHT);
                input.render(frame, area, true, true);
            })
            .expect("draw");

        insta::assert_snapshot!("render_cancelling", terminal.backend());
    }

    #[test]
    fn render_compacting_with_elapsed() {
        let mut input = InputArea::new();
        input.set_mode(AppMode::Compacting);
        input.set_elapsed_title(std::time::Duration::from_secs(55));

        let backend = ratatui::backend::TestBackend::new(60, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 60, MIN_HEIGHT);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        insta::assert_snapshot!("render_compacting_with_elapsed", terminal.backend());
    }

    #[test]
    fn render_running_bash_with_elapsed() {
        let mut input = InputArea::new();
        input.set_mode(AppMode::RunningBash);
        input.set_elapsed_title(std::time::Duration::from_secs(55));

        let backend = ratatui::backend::TestBackend::new(60, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 60, MIN_HEIGHT);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        insta::assert_snapshot!("render_running_bash_with_elapsed", terminal.backend());
    }

    #[test]
    fn set_mode_tool_confirmation_updates_title() {
        let mut input = InputArea::new();
        input.set_mode(AppMode::ToolConfirmation {
            name: "write_file".to_string(),
            input: serde_json::json!({"path": "/tmp/test.txt"}),
        });

        let backend = ratatui::backend::TestBackend::new(60, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let height = input.height_for_width(60, 24);
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 60, height);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        insta::assert_snapshot!("render_tool_confirmation", terminal.backend());
    }

    #[test]
    fn set_mode_back_to_input_restores_input_title() {
        let mut input = InputArea::new();
        input.input(char_key('h'));
        input.set_mode(AppMode::Streaming);
        input.set_mode(AppMode::Editing);

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, true, false);
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
        assert!(input.is_normal());
        assert_eq!(input.text(), "hi");
    }

    #[test]
    fn i_returns_to_insert_mode() {
        let mut input = InputArea::new();
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        input.input(char_key('i'));
        assert_eq!(input.vim_mode(), VimMode::Insert);
    }

    #[test]
    fn normal_mode_x_deletes_char() {
        let mut input = InputArea::new();
        input.input(char_key('a'));
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(input.text(), "a");
        input.input(char_key('x'));
        assert_eq!(input.text(), "");
    }

    #[test]
    fn normal_mode_motion_keys_do_not_insert() {
        let mut input = InputArea::new();
        input.input(char_key('a'));
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        input.input(char_key('h'));
        input.input(char_key('l'));
        assert_eq!(input.text(), "a");
    }

    #[test]
    fn render_normal_mode_shows_title() {
        let mut input = InputArea::new();
        input.input(char_key('h'));
        input.input(char_key('i'));
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        insta::assert_snapshot!("render_normal_mode", terminal.backend());
    }

    #[test]
    fn mode_defaults_to_editing() {
        let input = InputArea::new();
        assert_eq!(input.mode(), &AppMode::Editing);
    }

    #[test]
    fn height_for_width_streaming_returns_min() {
        let mut input = InputArea::new();
        input.set_mode(AppMode::Streaming);
        assert_eq!(input.height_for_width(60, 24), MIN_HEIGHT);
    }

    #[test]
    fn height_for_width_tool_confirmation_scales_with_content() {
        let mut input = InputArea::new();
        input.set_mode(AppMode::ToolConfirmation {
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls -la /some/long/path"}),
        });
        let height = input.height_for_width(60, 24);
        assert!(height >= MIN_HEIGHT);
    }

    #[test]
    fn normal_mode_v_enters_visual() {
        let mut input = InputArea::new();
        input.set_text("hello");
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        let consumed = input.input(char_key('v'));
        assert!(consumed, "v should be consumed in Normal mode");
        assert_eq!(
            input.vim_mode(),
            VimMode::Visual,
            "v should enter Visual mode"
        );
    }

    #[test]
    fn visual_mode_esc_returns_to_normal() {
        let mut input = InputArea::new();
        input.set_text("hello");
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        input.input(char_key('v'));
        assert_eq!(input.vim_mode(), VimMode::Visual);
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(input.vim_mode(), VimMode::Normal);
    }

    #[test]
    fn is_normal_checks_editor_state() {
        let mut input = InputArea::new();
        assert!(!input.is_normal());
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(input.is_normal());
        input.input(char_key('i'));
        assert!(!input.is_normal());
    }

    #[test]
    fn insert_paste_adds_text() {
        let mut input = InputArea::new();
        input.insert_paste("pasted text");
        assert_eq!(input.text(), "pasted text");
    }

    #[test]
    fn render_visual_mode() {
        let mut input = InputArea::new();
        input.set_text("hello world");
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        input.input(char_key('v'));
        input.input(char_key('l'));

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        insta::assert_snapshot!("render_visual_mode", terminal.backend());
    }

    #[test]
    fn compacting_mode_shows_compacting_title() {
        let mut input = InputArea::new();
        input.set_mode(AppMode::Compacting);

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        insta::assert_snapshot!("render_compacting", terminal.backend());
    }

    #[test]
    fn compacting_mode_height_is_min() {
        let mut input = InputArea::new();
        input.set_mode(AppMode::Compacting);
        assert_eq!(input.height_for_width(60, 24), MIN_HEIGHT);
    }

    #[test]
    fn make_selection_lines_full_segment_selected() {
        let segments = [(0, 5), (5, 10)];
        let lines = make_selection_lines("0123456789", &segments, 0, 10);
        assert_eq!(lines.len(), 2, "should produce 2 lines");
        for line in &lines {
            assert!(
                line.spans
                    .iter()
                    .any(|s| s.style.add_modifier.contains(Modifier::REVERSED)),
                "each line should have reversed span when fully selected"
            );
        }
    }

    #[test]
    fn make_selection_lines_partial_selection() {
        let segments = [(0, 5), (5, 10)];
        let lines = make_selection_lines("0123456789", &segments, 2, 8);
        assert_eq!(lines.len(), 2, "should produce 2 lines");
    }

    #[test]
    fn make_selection_lines_spanning_segments() {
        let segments = [(0, 3), (3, 6), (6, 9)];
        let lines = make_selection_lines("012345678", &segments, 2, 7);
        assert_eq!(lines.len(), 3, "should produce 3 lines");
    }

    #[test]
    fn make_selection_lines_empty_selection_range() {
        let segments = [(0, 5)];
        let lines = make_selection_lines("hello", &segments, 3, 3);
        assert_eq!(lines.len(), 1, "should produce 1 line");
        let line = &lines[0];
        assert!(
            !line
                .spans
                .iter()
                .any(|s| s.style.add_modifier.contains(Modifier::REVERSED)),
            "empty selection should produce raw (non-reversed) line"
        );
    }

    #[test]
    fn lines_from_segments_empty() {
        let lines = lines_from_segments("hello", &[]);
        assert_eq!(lines.len(), 1, "empty segments should produce 1 line");
    }

    #[test]
    fn lines_from_segments_nonempty() {
        let lines = lines_from_segments("hello", &[(0, 5)]);
        assert_eq!(lines.len(), 1, "should produce 1 line");
        assert_eq!(lines[0].to_string(), "hello");
    }

    #[test]
    fn render_visual_line_mode() {
        let mut input = InputArea::new();
        input.set_text("hello");
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        input.input(KeyEvent::new(KeyCode::Char('V'), KeyModifiers::SHIFT));

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        insta::assert_snapshot!("render_visual_line_mode", terminal.backend());
    }

    #[test]
    fn render_visual_block_mode() {
        let mut input = InputArea::new();
        input.set_text("hello");
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        input.input(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL));

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        insta::assert_snapshot!("render_visual_block_mode", terminal.backend());
    }

    #[test]
    fn render_session_picker() {
        let mut input = InputArea::new();
        input.set_mode(AppMode::SessionPicker);

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        insta::assert_snapshot!("render_session_picker", terminal.backend());
    }

    #[test]
    fn reset_to_insert_clears_pending_operator() {
        let mut input = InputArea::new();
        input.input(char_key('h'));
        input.input(char_key('e'));
        input.input(char_key('l'));
        input.input(char_key('l'));
        input.input(char_key('o'));
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(input.vim_mode(), VimMode::Normal);
        input.input(char_key('d'));
        assert_eq!(
            input.vim_mode(),
            VimMode::Normal,
            "d should leave editor in Normal with pending operator"
        );

        input.reset_to_insert();
        assert_eq!(
            input.vim_mode(),
            VimMode::Insert,
            "reset_to_insert should enter Insert mode"
        );

        input.input(char_key('x'));
        assert_eq!(
            input.text(),
            "hellox",
            "text should be hellox, not corrupted by pending d"
        );
    }

    #[test]
    fn reset_to_insert_from_visual_mode() {
        let mut input = InputArea::new();
        input.set_text("hello");
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        input.input(char_key('v'));
        assert_eq!(input.vim_mode(), VimMode::Visual);

        input.reset_to_insert();
        assert_eq!(
            input.vim_mode(),
            VimMode::Insert,
            "reset_to_insert from Visual should enter Insert mode"
        );
    }

    #[test]
    fn reset_to_insert_preserves_text() {
        let mut input = InputArea::new();
        input.set_text("important text");
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(input.vim_mode(), VimMode::Normal);

        input.reset_to_insert();
        assert_eq!(
            input.text(),
            "important text",
            "reset_to_insert should not destroy text content"
        );
        assert_eq!(input.vim_mode(), VimMode::Insert);
    }

    #[test]
    fn operator_pending_survives_mode_cycle_regression() {
        let mut input = InputArea::new();
        input.input(char_key('h'));
        input.input(char_key('e'));
        input.input(char_key('l'));
        input.input(char_key('l'));
        input.input(char_key('o'));
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        input.input(char_key('d'));

        input.set_mode(AppMode::Streaming);
        input.set_mode(AppMode::Editing);

        input.input(char_key('i'));
        assert_eq!(
            input.vim_mode(),
            VimMode::Normal,
            "BUG: without reset_to_insert, i is consumed as operand of pending d"
        );

        let mut input2 = InputArea::new();
        input2.input(char_key('h'));
        input2.input(char_key('e'));
        input2.input(char_key('l'));
        input2.input(char_key('l'));
        input2.input(char_key('o'));
        input2.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        input2.input(char_key('d'));

        input2.set_mode(AppMode::Streaming);
        input2.reset_to_insert();
        input2.set_mode(AppMode::Editing);

        input2.input(char_key('i'));
        assert_eq!(
            input2.vim_mode(),
            VimMode::Insert,
            "WITH reset_to_insert, i should enter Insert mode"
        );
    }

    #[test]
    fn render_tasks_picker() {
        let mut input = InputArea::new();
        input.set_mode(AppMode::TasksPicker);

        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        insta::assert_snapshot!("render_tasks_picker", terminal.backend());
    }
}
