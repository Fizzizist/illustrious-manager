use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hjkl_buffer::{
    Viewport, Wrap,
    wrap::{segment_for_col, wrap_segments},
};
use hjkl_editor_tui::crossterm_key_event_to_input;
use hjkl_engine::{Host, decode_planned_input};
use hjkl_form::{TextFieldEditor, VimMode};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
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
    suggestion: Option<String>,
}

impl InputArea {
    pub fn new() -> Self {
        let editor = TextFieldEditor::new(false);
        let mut input = Self {
            editor,
            mode: AppMode::Editing,
            elapsed: None,
            suggestion: None,
        };
        input.editor.enter_insert_at_end();
        input
    }

    pub fn set_suggestion(&mut self, suggestion: Option<String>) {
        self.suggestion = suggestion.filter(|s| !s.is_empty());
    }

    pub fn suggestion(&self) -> Option<&str> {
        self.suggestion.as_deref()
    }

    pub fn cursor(&self) -> (usize, usize) {
        self.editor.cursor()
    }

    pub fn accept_suggestion(&mut self) -> bool {
        let Some(suffix) = self.suggestion.take() else {
            return false;
        };
        self.editor.editor.insert_str(&suffix);
        true
    }

    pub fn input(&mut self, event: KeyEvent) -> bool {
        match self.mode {
            AppMode::Editing => {}
            _ => return false,
        }

        if event.code == KeyCode::Right
            && event.modifiers == KeyModifiers::NONE
            && self.editor.vim_mode() == VimMode::Insert
            && self.suggestion.is_some()
        {
            return self.accept_suggestion();
        }

        let was_insert = self.editor.vim_mode() == VimMode::Insert;
        let planned = crossterm_key_event_to_input(event);
        if let Some(input) = decode_planned_input(planned) {
            self.editor.handle_input(input);
            let is_insert = self.editor.vim_mode() == VimMode::Insert;
            if !was_insert || !is_insert {
                self.suggestion = None;
            } else if let KeyCode::Char(c) = event.code
                && (event.modifiers == KeyModifiers::NONE || event.modifiers == KeyModifiers::SHIFT)
            {
                self.suggestion = self
                    .suggestion
                    .take()
                    .and_then(|s| s.strip_prefix(c).map(str::to_string))
                    .filter(|s| !s.is_empty());
            } else {
                self.suggestion = None;
            }
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
        self.suggestion = None;
        self.editor.set_text(text);
    }

    pub fn clear(&mut self) {
        self.suggestion = None;
        self.editor.set_text("");
    }

    pub fn insert_paste(&mut self, text: &str) {
        self.suggestion = None;
        // TextFieldEditor exposes no paste method; insert_str is the
        // engine's public insertion API and calls mark_content_dirty().
        self.editor.editor.insert_str(text);
    }

    pub fn set_mode(&mut self, mode: AppMode) {
        if mode != AppMode::Editing {
            self.suggestion = None;
        }
        self.mode = mode;
    }

    pub fn reset_to_insert(&mut self) {
        self.suggestion = None;
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

    fn active_ghost(&self, disabled: bool) -> Option<&str> {
        if disabled || self.mode != AppMode::Editing || self.editor.vim_mode() != VimMode::Insert {
            return None;
        }
        self.suggestion.as_deref()
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
        let screen_rows = match self.active_ghost(false) {
            Some(ghost) => {
                let (cursor_row, cursor_col) = self.editor.cursor();
                let text = self.editor.text();
                let line_texts: Vec<&str> = text.split('\n').collect();
                (0..row_count)
                    .map(|row| {
                        let line_text = line_texts.get(row).copied().unwrap_or("");
                        if row == cursor_row {
                            let spliced = splice_ghost(line_text, cursor_col, ghost);
                            wrap_segments(&spliced, text_width, Wrap::Word).len()
                        } else {
                            wrap_segments(line_text, text_width, Wrap::Word).len()
                        }
                    })
                    .sum()
            }
            None => buffer.screen_rows_between(&viewport, 0, row_count.saturating_sub(1)),
        };
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
        let (cursor_row, cursor_col) = self.editor.cursor();
        let ghost = self.active_ghost(disabled).map(str::to_string);

        let lines: Vec<Line> = if row_count == 0 {
            vec![Line::from("")]
        } else {
            let selection = self.editor.editor.selection_highlight();
            let line_texts: Vec<&str> = text.split('\n').collect();

            (0..row_count)
                .flat_map(|row| {
                    let line_text = line_texts.get(row).copied().unwrap_or("");
                    if row == cursor_row
                        && let Some(ref ghost_text) = ghost
                    {
                        return make_ghost_lines(line_text, cursor_col, ghost_text, text_width);
                    }
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

        let mut scroll_y = if viewport.top_row == 0 {
            0
        } else {
            buffer.screen_rows_between(&viewport, 0, viewport.top_row.saturating_sub(1)) as u16
        };
        let cursor_pos = cursor_pos_in_doc(&self.editor, text_width, ghost.as_deref());
        if let Some((_, dy)) = cursor_pos
            && ghost.is_some()
            && dy >= scroll_y.saturating_add(inner.height)
        {
            scroll_y = dy.saturating_sub(inner.height.saturating_sub(1));
        }
        let paragraph = Paragraph::new(lines).block(block).scroll((scroll_y, 0));
        frame.render_widget(paragraph, area);

        if let Some((dx, dy)) = cursor_pos {
            let terminal_dy = dy.saturating_sub(scroll_y);
            if terminal_dy < inner.height {
                frame.set_cursor_position(ratatui::layout::Position {
                    x: inner.x.saturating_add(dx),
                    y: inner.y.saturating_add(terminal_dy),
                });
            }
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

fn splice_ghost(line_text: &str, cursor_col: usize, ghost: &str) -> String {
    let mut chars = line_text.chars();
    let mut out = String::with_capacity(line_text.len() + ghost.len());
    for _ in 0..cursor_col {
        if let Some(c) = chars.next() {
            out.push(c);
        } else {
            break;
        }
    }
    out.push_str(ghost);
    out.extend(chars);
    out
}

fn make_ghost_lines(
    line_text: &str,
    cursor_col: usize,
    ghost: &str,
    text_width: u16,
) -> Vec<Line<'static>> {
    let col = cursor_col.min(line_text.chars().count());
    let ghost_start = col;
    let ghost_end = col + ghost.chars().count();
    let spliced = splice_ghost(line_text, col, ghost);
    let segments = wrap_segments(&spliced, text_width, Wrap::Word);
    let ghost_style = Style::default().fg(Color::DarkGray);

    segments
        .iter()
        .map(|&(seg_start, seg_end)| {
            let mut spans = Vec::new();
            let b_start = seg_start.min(ghost_start);
            let b_end = seg_end.min(ghost_start);
            if b_start < b_end {
                spans.push(Span::raw(char_slice(&spliced, b_start, b_end)));
            }
            let g_start = seg_start.max(ghost_start).min(ghost_end);
            let g_end = seg_end.max(ghost_start).min(ghost_end);
            if g_start < g_end {
                spans.push(Span::styled(
                    char_slice(&spliced, g_start, g_end),
                    ghost_style,
                ));
            }
            let a_start = seg_start.max(ghost_end);
            let a_end = seg_end.max(ghost_end);
            if a_start < a_end {
                spans.push(Span::raw(char_slice(&spliced, a_start, a_end)));
            }
            if spans.is_empty() {
                Line::from("")
            } else {
                Line::from(spans)
            }
        })
        .collect()
}

fn cursor_pos_in_doc(
    editor: &TextFieldEditor,
    text_width: u16,
    ghost: Option<&str>,
) -> Option<(u16, u16)> {
    let viewport = editor.editor.host().viewport();
    let buffer = editor.buffer();
    let (row, col) = editor.cursor();

    let text = editor.text();
    let line_text = text.split('\n').nth(row)?;
    let display_line = match ghost {
        Some(g) => splice_ghost(line_text, col, g),
        None => line_text.to_string(),
    };
    let segments = wrap_segments(&display_line, text_width, Wrap::Word);
    let seg_idx = segment_for_col(&segments, col);
    let &(seg_start, _seg_end) = segments.get(seg_idx)?;

    let prefix: String = display_line
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

    Some((dx, dy))
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

    fn type_str(input: &mut InputArea, s: &str) {
        for c in s.chars() {
            input.input(char_key(c));
        }
    }

    fn row_text(
        backend: &ratatui::backend::TestBackend,
        y: u16,
        x_start: u16,
        x_end: u16,
    ) -> String {
        let buf = backend.buffer();
        let mut out = String::new();
        for x in x_start..x_end {
            if let Some(cell) = buf.cell((x, y)) {
                out.push_str(cell.symbol());
            }
        }
        out
    }

    #[test]
    fn right_arrow_accepts_suggestion_in_insert_mode() {
        let mut input = InputArea::new();
        type_str(&mut input, "src/f");
        input.set_suggestion(Some("rontend/".to_string()));
        assert_eq!(input.suggestion(), Some("rontend/"));

        let consumed = input.input(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        assert!(consumed);
        assert_eq!(input.text(), "src/frontend/");
        assert_eq!(input.cursor(), (0, 13));
        assert_eq!(input.suggestion(), None);
        assert!(!input.accept_suggestion());

        input.set_suggestion(Some(String::new()));
        assert_eq!(input.suggestion(), None);
    }

    #[test]
    fn right_arrow_moves_cursor_normally_when_no_suggestion_or_in_normal_mode() {
        let mut input = InputArea::new();
        type_str(&mut input, "abcd");
        assert_eq!(input.cursor(), (0, 4));

        input.input(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        input.input(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        assert_eq!(input.cursor(), (0, 2));

        let consumed = input.input(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        assert!(consumed);
        assert_eq!(input.cursor(), (0, 3));
        assert_eq!(input.text(), "abcd");

        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(input.vim_mode(), VimMode::Normal);
        input.input(char_key('h'));
        let normal_col = input.cursor().1;

        input.set_suggestion(Some("xyz".to_string()));
        input.input(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        assert_eq!(input.text(), "abcd");
        assert_eq!(input.cursor(), (0, normal_col + 1));
    }

    #[test]
    fn typing_matching_char_trims_suggestion_and_mismatch_clears_it() {
        let mut input = InputArea::new();
        type_str(&mut input, "src/f");
        input.set_suggestion(Some("rontend/".to_string()));

        input.input(char_key('r'));
        assert_eq!(input.text(), "src/fr");
        assert_eq!(input.suggestion(), Some("ontend/"));

        input.input(char_key('x'));
        assert_eq!(input.text(), "src/frx");
        assert_eq!(input.suggestion(), None);

        input.set_suggestion(Some("z".to_string()));
        input.input(char_key('z'));
        assert_eq!(input.suggestion(), None);

        input.set_suggestion(Some("abc".to_string()));
        input.input(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(input.suggestion(), None);
    }

    #[test]
    fn suggestion_clears_on_escape_clear_set_text_paste_and_mode_change() {
        let mut input = InputArea::new();
        type_str(&mut input, "src/");
        input.set_suggestion(Some("main.rs".to_string()));
        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(input.suggestion(), None);

        input.reset_to_insert();
        input.set_suggestion(Some("main.rs".to_string()));
        input.clear();
        assert_eq!(input.suggestion(), None);

        input.set_suggestion(Some("main.rs".to_string()));
        input.set_text("other");
        assert_eq!(input.suggestion(), None);

        input.set_suggestion(Some("main.rs".to_string()));
        input.insert_paste(" pasted");
        assert_eq!(input.suggestion(), None);

        input.set_suggestion(Some("main.rs".to_string()));
        input.set_mode(AppMode::Streaming);
        assert_eq!(input.suggestion(), None);

        input.set_mode(AppMode::Editing);
        input.set_suggestion(Some("main.rs".to_string()));
        input.reset_to_insert();
        assert_eq!(input.suggestion(), None);
    }

    #[test]
    fn ghost_text_hidden_when_disabled_or_not_insert_mode() {
        let mut input = InputArea::new();
        type_str(&mut input, "src/f");
        input.set_suggestion(Some("rontend/".to_string()));

        let backend = ratatui::backend::TestBackend::new(40, MIN_HEIGHT);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, true, false);
            })
            .expect("draw");

        let rendered = row_text(terminal.backend(), 1, 1, 39);
        assert_eq!(rendered.trim_end(), "src/f");
        for x in 1..39 {
            let cell = terminal
                .backend()
                .buffer()
                .cell((x, 1))
                .expect("interior cell");
            assert_ne!(cell.fg, ratatui::style::Color::DarkGray);
        }

        input.input(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        input.set_suggestion(Some("rontend/".to_string()));
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, false, false);
            })
            .expect("draw");

        let rendered_normal = row_text(terminal.backend(), 1, 1, 39);
        assert_eq!(rendered_normal.trim_end(), "src/f");
        for x in 1..39 {
            let cell = terminal
                .backend()
                .buffer()
                .cell((x, 1))
                .expect("interior cell");
            assert_ne!(cell.fg, ratatui::style::Color::DarkGray);
        }
    }

    #[test]
    fn snapshot_ghost_text_inline_at_eol() {
        let mut input = InputArea::new();
        type_str(&mut input, "src/f");
        input.set_suggestion(Some("rontend/".to_string()));

        let backend = ratatui::backend::TestBackend::new(40, MIN_HEIGHT);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, false, false);
            })
            .expect("draw");

        for x in 1..6 {
            let cell = terminal
                .backend()
                .buffer()
                .cell((x, 1))
                .expect("typed cell");
            assert_ne!(cell.fg, ratatui::style::Color::DarkGray);
        }
        assert_eq!(row_text(terminal.backend(), 1, 6, 14), "rontend/");
        for x in 6..14 {
            let cell = terminal
                .backend()
                .buffer()
                .cell((x, 1))
                .expect("ghost cell");
            assert_eq!(cell.fg, ratatui::style::Color::DarkGray);
        }
        assert_eq!(
            terminal.get_cursor_position().expect("cursor position"),
            ratatui::layout::Position { x: 1 + 5, y: 1 }
        );

        insta::assert_snapshot!("ghost_text_inline_at_eol", terminal.backend());
    }

    #[test]
    fn snapshot_ghost_text_mid_line_before_whitespace() {
        let mut input = InputArea::new();
        type_str(&mut input, "edit src/f please");
        for _ in 0..7 {
            input.input(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        }
        input.set_suggestion(Some("rontend/".to_string()));

        let backend = ratatui::backend::TestBackend::new(40, MIN_HEIGHT);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, false, false);
            })
            .expect("draw");

        assert_eq!(row_text(terminal.backend(), 1, 1, 11), "edit src/f");
        assert_eq!(row_text(terminal.backend(), 1, 11, 19), "rontend/");
        assert_eq!(row_text(terminal.backend(), 1, 19, 26), " please");
        for x in 11..19 {
            let cell = terminal
                .backend()
                .buffer()
                .cell((x, 1))
                .expect("ghost cell");
            assert_eq!(cell.fg, ratatui::style::Color::DarkGray);
        }
        for x in 19..26 {
            let cell = terminal
                .backend()
                .buffer()
                .cell((x, 1))
                .expect("trailing cell");
            assert_ne!(cell.fg, ratatui::style::Color::DarkGray);
        }
        assert_eq!(
            terminal.get_cursor_position().expect("cursor position"),
            ratatui::layout::Position { x: 1 + 10, y: 1 }
        );

        insta::assert_snapshot!("ghost_text_mid_line_before_whitespace", terminal.backend());
    }

    #[test]
    fn snapshot_ghost_text_wrapping_updates_height_and_cursor() {
        let mut input = InputArea::new();
        type_str(&mut input, "edit src/front");
        input.set_suggestion(Some("end/tui/completion.rs".to_string()));

        let height = input.height_for_width(20, 24);
        assert!(
            height > MIN_HEIGHT,
            "wrapped ghost text should grow height > {MIN_HEIGHT}, got {height}"
        );

        let backend = ratatui::backend::TestBackend::new(20, height);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 20, height);
                input.render(frame, area, false, false);
            })
            .expect("draw");

        let cursor = terminal.get_cursor_position().expect("cursor position");
        assert!(cursor.y >= 1 && cursor.y < height - 1);
        let first_ghost = terminal
            .backend()
            .buffer()
            .cell((cursor.x, cursor.y))
            .expect("cursor cell");
        assert_eq!(first_ghost.symbol(), "e");
        assert_eq!(first_ghost.fg, ratatui::style::Color::DarkGray);
        if cursor.x > 1 {
            let before_cursor = terminal
                .backend()
                .buffer()
                .cell((cursor.x - 1, cursor.y))
                .expect("before cursor cell");
            assert_eq!(before_cursor.symbol(), "t");
            assert_ne!(before_cursor.fg, ratatui::style::Color::DarkGray);
        }

        insta::assert_snapshot!(
            "ghost_text_wrapping_updates_height_and_cursor",
            terminal.backend()
        );
    }

    #[test]
    fn snapshot_ghost_text_with_unicode_prefix() {
        let mut input = InputArea::new();
        type_str(&mut input, "日本語 src/f");
        input.set_suggestion(Some("rontend/".to_string()));

        let backend = ratatui::backend::TestBackend::new(40, MIN_HEIGHT);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, MIN_HEIGHT);
                input.render(frame, area, false, false);
            })
            .expect("draw");

        assert_eq!(
            terminal.get_cursor_position().expect("cursor position"),
            ratatui::layout::Position { x: 1 + 12, y: 1 }
        );
        assert_eq!(row_text(terminal.backend(), 1, 13, 21), "rontend/");
        for x in 13..21 {
            let cell = terminal
                .backend()
                .buffer()
                .cell((x, 1))
                .expect("ghost cell");
            assert_eq!(cell.fg, ratatui::style::Color::DarkGray);
        }

        insta::assert_snapshot!("ghost_text_with_unicode_prefix", terminal.backend());
    }

    #[test]
    fn snapshot_ghost_text_multiline_with_wrapped_earlier_line() {
        let mut input = InputArea::new();
        type_str(&mut input, "first line wraps here");
        input.input(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        type_str(&mut input, "edit src/front");

        let height_without_ghost = input.height_for_width(20, 24);
        assert_eq!(height_without_ghost, 5);

        input.set_suggestion(Some("end/tui/completion.rs".to_string()));
        let height_with_ghost = input.height_for_width(20, 24);
        assert!(
            height_with_ghost > height_without_ghost,
            "multiline ghost wrapping should grow height from {height_without_ghost}, got {height_with_ghost}"
        );
        assert_eq!(height_with_ghost, 7);

        let backend = ratatui::backend::TestBackend::new(20, height_with_ghost);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 20, height_with_ghost);
                input.render(frame, area, false, false);
            })
            .expect("draw");

        let cursor = terminal.get_cursor_position().expect("cursor position");
        assert_eq!(cursor, ratatui::layout::Position { x: 10, y: 4 });
        let first_ghost = terminal
            .backend()
            .buffer()
            .cell((cursor.x, cursor.y))
            .expect("cursor cell");
        assert_eq!(first_ghost.symbol(), "e");
        assert_eq!(first_ghost.fg, ratatui::style::Color::DarkGray);
        let before_cursor = terminal
            .backend()
            .buffer()
            .cell((cursor.x - 1, cursor.y))
            .expect("before cursor cell");
        assert_eq!(before_cursor.symbol(), "t");
        assert_ne!(before_cursor.fg, ratatui::style::Color::DarkGray);

        insta::assert_snapshot!(
            "ghost_text_multiline_with_wrapped_earlier_line",
            terminal.backend()
        );
    }

    #[test]
    fn snapshot_ghost_text_scroll_overflow_keeps_cursor_visible() {
        let mut input = InputArea::new();
        type_str(&mut input, "first line");
        input.input(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        type_str(&mut input, "edit src/front");
        input.set_suggestion(Some("end/tui/completion.rs".to_string()));

        let constrained_height = 4;
        let backend = ratatui::backend::TestBackend::new(20, constrained_height);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 20, constrained_height);
                input.render(frame, area, false, false);
            })
            .expect("draw");

        let cursor = terminal.get_cursor_position().expect("cursor position");
        assert_eq!(cursor, ratatui::layout::Position { x: 10, y: 2 });
        let first_ghost = terminal
            .backend()
            .buffer()
            .cell((cursor.x, cursor.y))
            .expect("cursor cell");
        assert_eq!(first_ghost.symbol(), "e");
        assert_eq!(first_ghost.fg, ratatui::style::Color::DarkGray);
        let before_cursor = terminal
            .backend()
            .buffer()
            .cell((cursor.x - 1, cursor.y))
            .expect("before cursor cell");
        assert_eq!(before_cursor.symbol(), "t");
        assert_ne!(before_cursor.fg, ratatui::style::Color::DarkGray);

        insta::assert_snapshot!(
            "ghost_text_scroll_overflow_keeps_cursor_visible",
            terminal.backend()
        );
    }
}
