use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::borrow::Cow;
use tui_markdown::from_str as markdown_to_text;

use super::markdown_tables::preprocess_tables;

const TOOL_RESULT_TRUNCATE_CHARS: usize = 200;

const CONVERSATION_TITLE: &str = "Conversation";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationRole {
    User,
    Assistant,
    Error,
    ToolUse,
    ToolResult,
    Info,
    Thinking,
}

impl ConversationRole {
    pub fn display_label(&self) -> &'static str {
        match self {
            ConversationRole::User => "You",
            ConversationRole::Assistant => "Assistant",
            ConversationRole::Error => "Error",
            ConversationRole::ToolUse => "[Tool]",
            ConversationRole::ToolResult => "[Result]",
            ConversationRole::Info => "Info",
            ConversationRole::Thinking => "[Thinking]",
        }
    }

    pub fn color(&self) -> Color {
        match self {
            ConversationRole::User => Color::Green,
            ConversationRole::Assistant => Color::Blue,
            ConversationRole::Error => Color::Red,
            ConversationRole::ToolUse => Color::Cyan,
            ConversationRole::ToolResult => Color::Yellow,
            ConversationRole::Info => Color::Magenta,
            ConversationRole::Thinking => Color::DarkGray,
        }
    }
}

pub struct ConversationEntry {
    pub role: ConversationRole,
    pub content: String,
    /// Per-turn 1-based index for tool use / result entries. `None` for other roles.
    pub tool_index: Option<usize>,
    cached_lines: Vec<Line<'static>>,
    cached_lines_width: u16,
    cached_wrapped_count: u16,
    cached_width: u16,
}

impl ConversationEntry {
    pub fn new(role: ConversationRole, content: String) -> Self {
        let cached_lines = render_entry_lines(&role, None, &content, 0);
        Self {
            role,
            content,
            tool_index: None,
            cached_lines,
            cached_lines_width: 0,
            cached_wrapped_count: 0,
            cached_width: 0,
        }
    }

    pub fn new_indexed(role: ConversationRole, content: String, index: usize) -> Self {
        let cached_lines = render_entry_lines(&role, Some(index), &content, 0);
        Self {
            role,
            content,
            tool_index: Some(index),
            cached_lines,
            cached_lines_width: 0,
            cached_wrapped_count: 0,
            cached_width: 0,
        }
    }

    /// Create an entry whose display lines are provided directly, bypassing markdown rendering.
    ///
    /// Used for diff output from `edit_file` and `write_file` tool calls.
    pub fn new_with_lines(
        role: ConversationRole,
        content: String,
        lines: Vec<Line<'static>>,
    ) -> Self {
        Self::new_with_lines_indexed(role, content, lines, None)
    }

    pub fn new_with_lines_indexed(
        role: ConversationRole,
        content: String,
        lines: Vec<Line<'static>>,
        index: Option<usize>,
    ) -> Self {
        let label = role_label(&role, index);
        let mut all_lines = Vec::new();
        all_lines.push(Line::from(Span::styled(
            format!("{label}:"),
            Style::default().fg(role.color()),
        )));
        all_lines.extend(lines.into_iter().map(|mut l| {
            l.spans.insert(0, Span::raw("  "));
            l
        }));
        all_lines.push(Line::from(""));

        Self {
            role,
            content,
            tool_index: index,
            cached_lines: all_lines,
            cached_lines_width: u16::MAX,
            cached_wrapped_count: 0,
            cached_width: 0,
        }
    }

    pub fn lines(&self) -> &[Line<'static>] {
        &self.cached_lines
    }

    pub fn wrapped_line_count(&mut self, text_width: u16) -> u16 {
        if self.cached_lines_width != u16::MAX && self.cached_lines_width != text_width {
            self.cached_lines =
                render_entry_lines(&self.role, self.tool_index, &self.content, text_width);
            self.cached_lines_width = text_width;
            self.cached_width = 0;
        }
        if text_width == self.cached_width && self.cached_width > 0 {
            return self.cached_wrapped_count;
        }
        let paragraph = Paragraph::new(self.cached_lines.clone()).wrap(Wrap { trim: false });
        let count = paragraph.line_count(text_width) as u16;
        self.cached_wrapped_count = count;
        self.cached_width = text_width;
        count
    }
}

fn role_label(role: &ConversationRole, index: Option<usize>) -> String {
    match (role, index) {
        (ConversationRole::ToolUse, Some(n)) => format!("[Tool({})]", n),
        (ConversationRole::ToolResult, Some(n)) => format!("[Result({})]", n),
        _ => role.display_label().to_string(),
    }
}

fn render_role_lines(
    role: &ConversationRole,
    index: Option<usize>,
    content: &str,
    trailing_blank: bool,
    text_width: u16,
) -> Vec<Line<'static>> {
    let label = role_label(role, index);
    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        format!("{label}:"),
        Style::default().fg(role.color()),
    )));
    let display_content = maybe_truncate(content, role);
    let budget = usize::from(text_width).saturating_sub(2);
    let preprocessed = preprocess_tables(&display_content, budget);
    let rendered = markdown_to_text(&preprocessed);
    for line in rendered.lines {
        let mut prefixed = Line::from(Span::raw("  "));
        prefixed.spans.extend(
            line.spans
                .into_iter()
                .map(|span| Span::styled(span.content.into_owned(), span.style)),
        );
        lines.push(prefixed);
    }
    if trailing_blank {
        lines.push(Line::from(""));
    }
    lines
}

fn render_entry_lines(
    role: &ConversationRole,
    index: Option<usize>,
    content: &str,
    text_width: u16,
) -> Vec<Line<'static>> {
    render_role_lines(role, index, content, true, text_width)
}

fn render_current_response_lines(current_response: &str, text_width: u16) -> Vec<Line<'static>> {
    render_role_lines(
        &ConversationRole::Assistant,
        None,
        current_response,
        false,
        text_width,
    )
}

fn thinking_indicator_lines() -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled(
            "[Thinking]:",
            Style::default().fg(ConversationRole::Thinking.color()),
        )),
        Line::from(""),
    ]
}

fn estimate_wrapped_count(lines: &[Line<'_>], text_width: u16) -> u16 {
    if text_width == 0 {
        return lines.len() as u16;
    }
    lines
        .iter()
        .map(|line| {
            let w = line.width() as u16;
            if w == 0 { 1u16 } else { w.div_ceil(text_width) }
        })
        .sum()
}

pub struct ConversationArea<'a> {
    entries: &'a mut [ConversationEntry],
    is_thinking: bool,
    current_response: &'a str,
    scroll_offset: u16,
    viewport_height: u16,
}

impl<'a> ConversationArea<'a> {
    pub fn new(
        entries: &'a mut [ConversationEntry],
        is_thinking: bool,
        current_response: &'a str,
        scroll_offset: u16,
        viewport_height: u16,
    ) -> Self {
        Self {
            entries,
            is_thinking,
            current_response,
            scroll_offset,
            viewport_height,
        }
    }

    fn compute_entry_counts(&mut self, text_width: u16) -> Vec<u16> {
        self.entries
            .iter_mut()
            .map(|entry| entry.wrapped_line_count(text_width))
            .collect()
    }

    pub fn max_scroll(&mut self, text_width: u16) -> u16 {
        let entry_counts = self.compute_entry_counts(text_width);
        let total_entries: u16 = entry_counts.iter().sum();

        let thinking_count = if self.is_thinking { 2u16 } else { 0u16 };

        let response_count = if self.current_response.is_empty() {
            0u16
        } else {
            let response_lines = render_current_response_lines(self.current_response, text_width);
            estimate_wrapped_count(&response_lines, text_width)
        };

        let total_visual = total_entries
            .saturating_add(thinking_count)
            .saturating_add(response_count);
        total_visual.saturating_sub(self.viewport_height)
    }

    pub fn render(&mut self, frame: &mut ratatui::Frame, area: Rect, text_width: u16) {
        let visible_height = area.height.saturating_sub(2);

        let entry_counts = self.compute_entry_counts(text_width);
        let total_entries: u16 = entry_counts.iter().sum();

        let thinking_lines = if self.is_thinking {
            thinking_indicator_lines()
        } else {
            Vec::new()
        };
        let thinking_count = if self.is_thinking { 2u16 } else { 0u16 };

        let response_lines = if self.current_response.is_empty() {
            Vec::new()
        } else {
            render_current_response_lines(self.current_response, text_width)
        };
        let response_count = if response_lines.is_empty() {
            0u16
        } else {
            estimate_wrapped_count(&response_lines, text_width)
        };

        let total_visual = total_entries
            .saturating_add(thinking_count)
            .saturating_add(response_count);
        let auto_scroll = total_visual.saturating_sub(visible_height);
        let scroll_row = auto_scroll.saturating_sub(self.scroll_offset.min(auto_scroll));

        // Windowed approach: find which entries are visible and only collect those lines,
        // plus compute how many wrapped lines to skip at the top of the window.
        let (window_lines, lines_to_skip) = self.collect_window_lines(
            &entry_counts,
            &thinking_lines,
            &response_lines,
            scroll_row,
            visible_height,
            text_width,
        );

        let conversation = Paragraph::new(window_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(CONVERSATION_TITLE),
            )
            .wrap(Wrap { trim: false })
            .scroll((lines_to_skip, 0));
        frame.render_widget(conversation, area);
    }

    fn collect_window_lines(
        &self,
        entry_counts: &[u16],
        thinking_lines: &[Line<'static>],
        response_lines: &[Line<'static>],
        scroll_row: u16,
        visible_height: u16,
        text_width: u16,
    ) -> (Vec<Line<'static>>, u16) {
        let window_end = scroll_row.saturating_add(visible_height);

        let mut result = Vec::new();
        let mut cumulative: u16 = 0;
        // Track wrapped lines from entries fully above the viewport that were
        // skipped — the difference between scroll_row and the start of the
        // first included entry tells us how many wrapped lines to skip via
        // Paragraph::scroll.
        let mut first_included_start: Option<u16> = None;

        for (i, &count) in entry_counts.iter().enumerate() {
            let entry_start = cumulative;
            let entry_end = cumulative.saturating_add(count);

            if entry_end <= scroll_row {
                cumulative = entry_end;
                continue;
            }
            if entry_start >= window_end {
                break;
            }

            if first_included_start.is_none() {
                first_included_start = Some(entry_start);
            }

            result.extend(self.entries[i].lines().iter().cloned());

            cumulative = entry_end;
        }

        // Add current thinking lines if they fall within the window
        if !thinking_lines.is_empty() {
            let thinking_count = estimate_wrapped_count(thinking_lines, text_width);
            let thinking_start = cumulative;
            let thinking_end = cumulative.saturating_add(thinking_count);

            if thinking_end > scroll_row && thinking_start < window_end {
                if first_included_start.is_none() {
                    first_included_start = Some(thinking_start);
                }
                result.extend(thinking_lines.iter().cloned());
            }

            cumulative = thinking_end;
        }

        // Add current response lines if they fall within the window
        if !response_lines.is_empty() {
            let response_count = estimate_wrapped_count(response_lines, text_width);
            let response_start = cumulative;
            let response_end = cumulative.saturating_add(response_count);

            if response_end > scroll_row && response_start < window_end {
                if first_included_start.is_none() {
                    first_included_start = Some(response_start);
                }
                result.extend(response_lines.iter().cloned());
            }
        }

        let lines_to_skip = scroll_row.saturating_sub(first_included_start.unwrap_or(scroll_row));

        (result, lines_to_skip)
    }
}

fn maybe_truncate<'a>(content: &'a str, role: &ConversationRole) -> Cow<'a, str> {
    if role != &ConversationRole::ToolResult {
        return Cow::Borrowed(content);
    }
    let mut chars = content.chars();
    let head: String = (&mut chars).take(TOOL_RESULT_TRUNCATE_CHARS).collect();
    if chars.next().is_some() {
        Cow::Owned(format!("{head}...[truncated]"))
    } else {
        Cow::Borrowed(content)
    }
}

pub fn tool_use_display_content(name: &str, input: &serde_json::Value) -> String {
    format!(
        "{}\n  {}",
        name,
        serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string())
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_role_labels() {
        assert_eq!(ConversationRole::User.display_label(), "You");
        assert_eq!(ConversationRole::Assistant.display_label(), "Assistant");
        assert_eq!(ConversationRole::Error.display_label(), "Error");
        assert_eq!(ConversationRole::ToolUse.display_label(), "[Tool]");
        assert_eq!(ConversationRole::ToolResult.display_label(), "[Result]");
        assert_eq!(ConversationRole::Info.display_label(), "Info");
    }

    #[test]
    fn role_label_without_index_uses_display_label() {
        assert_eq!(role_label(&ConversationRole::ToolUse, None), "[Tool]");
        assert_eq!(role_label(&ConversationRole::ToolResult, None), "[Result]");
        assert_eq!(role_label(&ConversationRole::User, None), "You");
    }

    #[test]
    fn role_label_with_index_formats_tool_labels() {
        assert_eq!(role_label(&ConversationRole::ToolUse, Some(1)), "[Tool(1)]");
        assert_eq!(
            role_label(&ConversationRole::ToolResult, Some(2)),
            "[Result(2)]"
        );
        assert_eq!(role_label(&ConversationRole::User, Some(3)), "You");
    }

    #[test]
    fn conversation_role_colors() {
        assert_eq!(ConversationRole::User.color(), Color::Green);
        assert_eq!(ConversationRole::Assistant.color(), Color::Blue);
        assert_eq!(ConversationRole::Error.color(), Color::Red);
        assert_eq!(ConversationRole::ToolUse.color(), Color::Cyan);
        assert_eq!(ConversationRole::ToolResult.color(), Color::Yellow);
        assert_eq!(ConversationRole::Info.color(), Color::Magenta);
    }

    #[test]
    fn entry_lines_empty_conversation() {
        let entries: &mut [ConversationEntry] = &mut [];
        let area = ConversationArea::new(entries, false, "", 0, 10);
        assert!(area.entries.is_empty());
    }

    #[test]
    fn entry_lines_with_entries() {
        let mut entries = vec![
            ConversationEntry::new(ConversationRole::User, "hello".to_string()),
            ConversationEntry::new(ConversationRole::Assistant, "world".to_string()),
        ];
        let total_lines: usize = entries.iter().map(|e| e.lines().len()).sum();
        assert_eq!(total_lines, 6);
        let _ = ConversationArea::new(&mut entries, false, "", 0, 10);
    }

    #[test]
    fn entry_lines_with_current_response() {
        let mut entries: Vec<ConversationEntry> = Vec::new();
        let area = ConversationArea::new(&mut entries, false, "streaming text", 0, 10);
        assert!(!area.current_response.is_empty());
    }

    #[test]
    fn entry_lines_with_current_thinking_and_response() {
        let mut entries: Vec<ConversationEntry> = Vec::new();
        let area = ConversationArea::new(&mut entries, true, "streaming text", 0, 10);
        assert!(area.is_thinking);
        assert!(!area.current_response.is_empty());
    }

    #[test]
    fn render_current_thinking_appears_above_current_response() {
        let mut entries: Vec<ConversationEntry> = Vec::new();
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 58, 18);
                let mut area = ConversationArea::new(&mut entries, true, "the answer", 0, 18);
                area.render(frame, rect, 56);
            })
            .expect("draw");

        let rendered = format!("{:?}", terminal.backend());
        // Thinking label must appear before Assistant label
        let thinking_pos = rendered
            .find("[Thinking]")
            .expect("should contain [Thinking] label");
        let assistant_pos = rendered
            .find("Assistant:")
            .expect("should contain Assistant label");
        assert!(
            thinking_pos < assistant_pos,
            "thinking should appear above assistant response in rendered output"
        );
    }

    #[test]
    fn streaming_thinking_shows_compact_indicator_not_full_text() {
        let mut entries: Vec<ConversationEntry> = Vec::new();
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 58, 18);
                let mut area = ConversationArea::new(&mut entries, true, "", 0, 18);
                area.render(frame, rect, 56);
            })
            .expect("draw");

        let rendered = format!("{:?}", terminal.backend());
        assert!(
            rendered.contains("[Thinking]:"),
            "should contain [Thinking]: label"
        );
        // The compact indicator should not render any thinking content text
        // (the old behavior rendered the full markdown text of the thinking)
        // With the compact indicator, only the label line + blank line appear
        assert!(
            !rendered.contains("  reasoning"),
            "streaming thinking should not render full thinking content text, got:\n{rendered}"
        );
    }

    #[test]
    fn committed_thinking_entry_renders_full_text() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Thinking,
            "reasoning step 1 reasoning step 2".to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        let rendered = format!("{:?}", terminal.backend());
        assert!(
            rendered.contains("[Thinking]:"),
            "should contain [Thinking]: label"
        );
        assert!(
            rendered.contains("reasoning step 1"),
            "committed thinking entry should render full text content"
        );
    }

    #[test]
    fn maybe_truncate_borrows_short_tool_result() {
        let content = "short output";
        let result = maybe_truncate(content, &ConversationRole::ToolResult);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, content);
    }

    #[test]
    fn maybe_truncate_borrows_non_tool_result_roles() {
        let content = "x".repeat(500);
        for role in [
            ConversationRole::User,
            ConversationRole::Assistant,
            ConversationRole::Error,
            ConversationRole::ToolUse,
        ] {
            let result = maybe_truncate(&content, &role);
            assert!(
                matches!(result, Cow::Borrowed(_)),
                "expected borrow for {role:?}"
            );
        }
    }

    #[test]
    fn maybe_truncate_handles_multibyte_utf8() {
        let emoji = "🦀".repeat(300);
        let result = maybe_truncate(&emoji, &ConversationRole::ToolResult);
        assert!(result.ends_with("...[truncated]"));
        let char_count = result
            .strip_suffix("...[truncated]")
            .expect("suffix should exist")
            .chars()
            .count();
        assert_eq!(char_count, TOOL_RESULT_TRUNCATE_CHARS);
    }

    #[test]
    fn maybe_truncate_does_not_split_multibyte_char() {
        let content = "é".repeat(300);
        let result = maybe_truncate(&content, &ConversationRole::ToolResult);
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
    }

    #[test]
    fn render_empty_conversation() {
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let mut entries: Vec<ConversationEntry> = Vec::new();
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_empty_conversation", terminal.backend());
    }

    #[test]
    fn render_with_entries() {
        let mut entries = vec![
            ConversationEntry::new(ConversationRole::User, "Hello".to_string()),
            ConversationEntry::new(ConversationRole::Assistant, "Hi there!".to_string()),
        ];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_with_entries", terminal.backend());
    }

    #[test]
    fn render_with_current_response() {
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let mut entries: Vec<ConversationEntry> = Vec::new();
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget =
                    ConversationArea::new(&mut entries, false, "Streaming response...", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_with_current_response", terminal.backend());
    }

    #[test]
    fn max_scroll_with_many_entries() {
        let mut entries: Vec<ConversationEntry> = (0..40)
            .map(|i| ConversationEntry::new(ConversationRole::User, format!("line {i}")))
            .collect();
        let mut area = ConversationArea::new(&mut entries, false, "", 0, 10);
        let max = area.max_scroll(58);
        assert!(max > 0, "should have scrollable content");
    }

    #[test]
    fn max_scroll_zero_when_content_fits() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::User,
            "short".to_string(),
        )];
        let mut area = ConversationArea::new(&mut entries, false, "", 0, 100);
        let max = area.max_scroll(58);
        assert_eq!(max, 0);
    }

    #[test]
    fn tool_use_display_content_formats() {
        let result = tool_use_display_content("bash", &serde_json::json!({"command": "ls"}));
        assert!(result.contains("bash"));
        assert!(result.contains("command"));
    }

    #[test]
    fn render_markdown_bold_text() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Assistant,
            "This is **bold** text".to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_markdown_bold_text", terminal.backend());
    }

    #[test]
    fn render_markdown_code_block() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Assistant,
            "```rust\nfn main() {}\n```".to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_markdown_code_block", terminal.backend());
    }

    #[test]
    fn render_markdown_heading() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Assistant,
            "# Hello World\nSome content".to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_markdown_heading", terminal.backend());
    }

    #[test]
    fn render_markdown_list() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Assistant,
            "- item one\n- item two\n- item three".to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_markdown_list", terminal.backend());
    }

    #[test]
    fn render_markdown_table() {
        let table = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |";
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Assistant,
            table.to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_markdown_table", terminal.backend());
    }

    #[test]
    fn render_markdown_table_with_alignment() {
        let table = "| Left | Center | Right |\n|:-----|:------:|------:|\n| a | b | c |\n| longer | mid | r |";
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Assistant,
            table.to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_markdown_table_with_alignment", terminal.backend());
    }

    #[test]
    fn render_markdown_table_in_assistant_with_surrounding_text() {
        let content = "# Results\n\n| File | Lines |\n|------|-------|\n| main.rs | 100 |\n| lib.rs | 200 |\n\nSee above.";
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Assistant,
            content.to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!(
            "render_markdown_table_in_assistant_with_surrounding_text",
            terminal.backend()
        );
    }

    #[test]
    fn render_markdown_table_in_streaming_response() {
        // Finding 10: verify preprocess_tables runs on the current_response (streaming) path.
        let table = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |";
        let mut entries: Vec<ConversationEntry> = Vec::new();
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, table, 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!(
            "render_markdown_table_in_streaming_response",
            terminal.backend()
        );
    }

    #[test]
    fn render_markdown_table_with_cjk_and_emoji() {
        // Finding 11: snapshot pins exact column widths for wide Unicode characters.
        let table = "| 言語 | 記号 |\n|------|------|\n| 🦀 Rust | ✓ |\n| Python | ✗ |";
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Assistant,
            table.to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!(
            "render_markdown_table_with_cjk_and_emoji",
            terminal.backend()
        );
    }

    #[test]
    fn render_user_markdown_italic() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::User,
            "This is *italic* text".to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_user_markdown_italic", terminal.backend());
    }

    #[test]
    fn render_intro_entry() {
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
            openai_compat: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::path::PathBuf::from("/sessions"),
            models: std::collections::BTreeMap::new(),
            thinking: None,
            compaction_role: "compaction".to_string(),
        };
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Info,
            generate_intro_message(&config),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_intro_entry", terminal.backend());
    }

    #[test]
    fn render_tool_entry_not_rendered_as_markdown() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::ToolUse,
            "bash\n  {\"command\": \"ls\"}".to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!(
            "render_tool_entry_not_rendered_as_markdown",
            terminal.backend()
        );
    }

    #[test]
    fn render_indexed_tool_entries_show_numbered_labels() {
        let mut entries = vec![
            ConversationEntry::new_indexed(
                ConversationRole::ToolUse,
                "bash\n  {\"command\": \"ls\"}".to_string(),
                1,
            ),
            ConversationEntry::new_indexed(
                ConversationRole::ToolResult,
                "file1.txt\nfile2.txt".to_string(),
                1,
            ),
            ConversationEntry::new_indexed(
                ConversationRole::ToolUse,
                "bash\n  {\"command\": \"pwd\"}".to_string(),
                2,
            ),
            ConversationEntry::new_indexed(
                ConversationRole::ToolResult,
                "/home/user".to_string(),
                2,
            ),
        ];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        let rendered = format!("{:?}", terminal.backend());
        assert!(rendered.contains("[Tool(1)]"), "should show Tool(1) label");
        assert!(
            rendered.contains("[Result(1)]"),
            "should show Result(1) label"
        );
        assert!(rendered.contains("[Tool(2)]"), "should show Tool(2) label");
        assert!(
            rendered.contains("[Result(2)]"),
            "should show Result(2) label"
        );

        insta::assert_snapshot!("render_indexed_tool_entries", terminal.backend());
    }

    #[test]
    fn cached_line_count_matches_paragraph_line_count() {
        let mut entry = ConversationEntry::new(ConversationRole::User, "hello world".to_string());
        let count = entry.wrapped_line_count(58);
        let paragraph = Paragraph::new(entry.lines().to_vec()).wrap(Wrap { trim: false });
        let expected = paragraph.line_count(58) as u16;
        assert_eq!(count, expected);
    }

    #[test]
    fn cached_line_count_is_reused_for_same_width() {
        let mut entry = ConversationEntry::new(ConversationRole::User, "hello world".to_string());
        let count1 = entry.wrapped_line_count(58);
        let count2 = entry.wrapped_line_count(58);
        assert_eq!(count1, count2);
        assert_eq!(entry.cached_width, 58);
    }

    #[test]
    fn cached_line_count_recomputes_for_different_width() {
        let mut entry = ConversationEntry::new(
            ConversationRole::User,
            "a longer message that wraps differently at different widths".to_string(),
        );
        let wide = entry.wrapped_line_count(80);
        let narrow = entry.wrapped_line_count(10);
        assert!(narrow >= wide, "narrow={narrow} should be >= wide={wide}");
    }

    #[test]
    fn windowed_render_skips_entries_above_viewport() {
        let mut entries: Vec<ConversationEntry> = (0..50)
            .map(|i| ConversationEntry::new(ConversationRole::User, format!("message {i}")))
            .collect();
        let backend = ratatui::backend::TestBackend::new(60, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 10);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 8);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("windowed_render_many_entries", terminal.backend());
    }

    #[test]
    fn windowed_render_with_scroll_offset() {
        let mut entries: Vec<ConversationEntry> = (0..20)
            .map(|i| ConversationEntry::new(ConversationRole::User, format!("message {i}")))
            .collect();
        let backend = ratatui::backend::TestBackend::new(60, 12);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 12);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 20, 10);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("windowed_render_with_scroll_offset", terminal.backend());
    }

    #[test]
    fn max_scroll_accurate_with_wrapping_text() {
        let long_text = "word ".repeat(100);
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Assistant,
            long_text,
        )];
        let viewport = 10u16;
        let text_width = 30u16;
        let mut area = ConversationArea::new(&mut entries, false, "", 0, viewport);
        let max = area.max_scroll(text_width);

        let paragraph = Paragraph::new(entries[0].lines().to_vec()).wrap(Wrap { trim: false });
        let actual_lines = paragraph.line_count(text_width) as u16;
        let expected_max = actual_lines.saturating_sub(viewport);
        assert_eq!(max, expected_max);
    }

    #[test]
    fn regression_conversation_sync_after_many_rounds() {
        let text_width = 58u16;
        let viewport = 18u16;
        let mut entries: Vec<ConversationEntry> = Vec::new();
        for i in 0..30 {
            entries.push(ConversationEntry::new(
                ConversationRole::User,
                format!("User message {i} with some extra text to test wrapping behavior"),
            ));
            entries.push(ConversationEntry::new(
                ConversationRole::Assistant,
                format!(
                    "Assistant response {i}: {}",
                    "detailed explanation ".repeat(5)
                ),
            ));
        }

        let mut area = ConversationArea::new(&mut entries, false, "", 0, viewport);
        let max = area.max_scroll(text_width);

        let all_lines: Vec<Line<'static>> =
            entries.iter().flat_map(|e| e.lines().to_vec()).collect();
        let full_paragraph = Paragraph::new(all_lines).wrap(Wrap { trim: false });
        let total_visual = full_paragraph.line_count(text_width) as u16;
        let expected_max = total_visual.saturating_sub(viewport);
        assert_eq!(
            max, expected_max,
            "max_scroll should match total visual lines minus viewport"
        );
    }

    #[test]
    fn regression_scroll_to_top_shows_first_message() {
        let text_width = 58u16;
        let viewport = 8u16;
        let mut entries: Vec<ConversationEntry> = Vec::new();
        for i in 0..20 {
            entries.push(ConversationEntry::new(
                ConversationRole::User,
                format!("message {i}"),
            ));
        }

        let mut area = ConversationArea::new(&mut entries, false, "", 0, viewport);
        let max = area.max_scroll(text_width);

        let backend = ratatui::backend::TestBackend::new(60, viewport + 2);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, viewport + 2);
                let mut scrolled_area =
                    ConversationArea::new(&mut entries, false, "", max, viewport);
                scrolled_area.render(frame, rect, text_width);
            })
            .expect("draw");

        let rendered = format!("{:?}", terminal.backend());
        assert!(
            rendered.contains("message 0"),
            "first message should be visible when scrolled to top, got:\n{rendered}"
        );
    }

    #[test]
    fn regression_auto_scroll_shows_latest_after_streaming() {
        let text_width = 58u16;
        let viewport = 8u16;
        let mut entries: Vec<ConversationEntry> = Vec::new();
        for i in 0..20 {
            entries.push(ConversationEntry::new(
                ConversationRole::User,
                format!("message {i}"),
            ));
        }

        let backend = ratatui::backend::TestBackend::new(60, viewport + 2);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, viewport + 2);
                let mut area = ConversationArea::new(&mut entries, false, "", 0, viewport);
                area.render(frame, rect, text_width);
            })
            .expect("draw");

        let rendered = format!("{:?}", terminal.backend());
        assert!(
            rendered.contains("message 19"),
            "latest message should be visible with scroll_offset=0, got:\n{rendered}"
        );
    }

    #[test]
    fn regression_large_entry_scrollable_through_middle() {
        let text_width = 30u16;
        let viewport = 6u16;
        let long_text = (0..20)
            .map(|i| format!("Line number {i} of the response"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Assistant,
            long_text,
        )];

        let mut area = ConversationArea::new(&mut entries, false, "", 0, viewport);
        let max = area.max_scroll(text_width);
        assert!(max > 0, "should have scrollable content");

        // Render at different scroll positions and collect what's visible
        let mut all_visible_content: Vec<String> = Vec::new();
        let mut scroll = max;
        loop {
            let backend = ratatui::backend::TestBackend::new(text_width + 2, viewport + 2);
            let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
            terminal
                .draw(|frame| {
                    let rect = ratatui::layout::Rect::new(0, 0, text_width + 2, viewport + 2);
                    let mut area = ConversationArea::new(&mut entries, false, "", scroll, viewport);
                    area.render(frame, rect, text_width);
                })
                .expect("draw");
            let rendered = format!("{:?}", terminal.backend());
            all_visible_content.push(rendered);

            if scroll == 0 {
                break;
            }
            scroll = scroll.saturating_sub(viewport);
        }

        // Every line of the entry should appear in at least one scroll position
        for i in 0..20 {
            let needle = format!("Line number {i} ");
            let found = all_visible_content
                .iter()
                .any(|content| content.contains(&needle));
            assert!(
                found,
                "Line {i} should be visible at some scroll position but was not found"
            );
        }
    }

    #[test]
    fn regression_viewport_boundary_between_entries_and_streaming_response() {
        let text_width = 58u16;
        let viewport = 10u16;

        // Create entries whose total lines *almost* fill the viewport, so the
        // streaming response straddles the boundary.
        let mut entries: Vec<ConversationEntry> = (0..3)
            .map(|i| ConversationEntry::new(ConversationRole::User, format!("message {i}")))
            .collect();

        // Verify entries exist so the boundary scenario is meaningful.
        let mut area = ConversationArea::new(&mut entries, false, "", 0, viewport);
        let entry_only_max = area.max_scroll(text_width);
        assert!(
            entry_only_max == 0,
            "entries should fit within viewport for this test"
        );

        // The current response should start at total_entry_lines.
        // Use a streaming response long enough to span beyond the viewport.
        let streaming = (0..30)
            .map(|i| format!("streaming line {i} with enough text to avoid wrapping at width 58"))
            .collect::<Vec<_>>()
            .join("\n");

        // Render with scroll_offset=0 (auto-scroll to bottom) and verify the
        // window includes lines from both the last entry and the streaming response.
        let backend = ratatui::backend::TestBackend::new(text_width + 2, viewport + 2);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, text_width + 2, viewport + 2);
                let mut area = ConversationArea::new(&mut entries, false, &streaming, 0, viewport);
                area.render(frame, rect, text_width);
            })
            .expect("draw");

        let rendered = format!("{:?}", terminal.backend());

        // The streaming response should be visible at the bottom.
        assert!(
            rendered.contains("streaming line 29"),
            "last streaming line should be visible, got:\n{rendered}"
        );

        // Now scroll up one viewport so the window overlaps the entry/streaming boundary.
        let scroll_up_amount = viewport;
        let backend2 = ratatui::backend::TestBackend::new(text_width + 2, viewport + 2);
        let mut terminal2 = ratatui::Terminal::new(backend2).expect("terminal creation");
        terminal2
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, text_width + 2, viewport + 2);
                let mut area = ConversationArea::new(
                    &mut entries,
                    false,
                    &streaming,
                    scroll_up_amount,
                    viewport,
                );
                area.render(frame, rect, text_width);
            })
            .expect("draw");

        let rendered2 = format!("{:?}", terminal2.backend());

        // When scrolled up by one viewport, we should see earlier streaming lines
        // but NOT the very last one.
        assert!(
            !rendered2.contains("streaming line 29"),
            "scrolled-up view should not show last streaming line, got:\n{rendered2}"
        );
        assert!(
            rendered2.contains("streaming line"),
            "scrolled-up view should still show some streaming lines, got:\n{rendered2}"
        );
    }

    #[test]
    fn regression_adjacent_scroll_positions_differ() {
        let text_width = 58u16;
        let viewport = 8u16;
        let long_text = "word ".repeat(200);
        let mut entries = vec![
            ConversationEntry::new(ConversationRole::Assistant, long_text),
            ConversationEntry::new(ConversationRole::User, "final message".to_string()),
        ];

        let mut area = ConversationArea::new(&mut entries, false, "", 0, viewport);
        let max = area.max_scroll(text_width);
        assert!(
            max > viewport,
            "need enough content to scroll multiple pages"
        );

        // Render at two adjacent half-page scroll positions
        let scroll_a = max;
        let scroll_b = max.saturating_sub(viewport / 2);

        let render_at = |entries: &mut Vec<ConversationEntry>, scroll: u16| -> String {
            let backend = ratatui::backend::TestBackend::new(60, viewport + 2);
            let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
            terminal
                .draw(|frame| {
                    let rect = ratatui::layout::Rect::new(0, 0, 60, viewport + 2);
                    let mut area = ConversationArea::new(entries, false, "", scroll, viewport);
                    area.render(frame, rect, text_width);
                })
                .expect("draw");
            format!("{:?}", terminal.backend())
        };

        let content_a = render_at(&mut entries, scroll_a);
        let content_b = render_at(&mut entries, scroll_b);
        assert_ne!(
            content_a, content_b,
            "adjacent scroll positions should show different content"
        );
    }

    #[test]
    fn render_markdown_table_overflow_wraps_cells() {
        let table = "| Column One | Column Two | Column Three | Column Four |\n\
                     |------------|------------|--------------|-------------|\n\
                     | Long value here | Another long value | Yet another long value | Final long value |";
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::Assistant,
            table.to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(40, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 40, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, "", 0, 20);
                area_widget.render(frame, rect, 38);
            })
            .expect("draw");

        insta::assert_snapshot!(
            "render_markdown_table_overflow_wraps_cells",
            terminal.backend()
        );
    }

    #[test]
    fn render_markdown_table_reflows_on_resize() {
        use unicode_width::UnicodeWidthStr;
        let table = "| Column One | Column Two | Column Three | Column Four |\n\
                     |------------|------------|--------------|-------------|\n\
                     | Long value here | Another long value | Yet another long value | Final long value |";
        let mut entry = ConversationEntry::new(ConversationRole::Assistant, table.to_string());
        let wide_count = entry.wrapped_line_count(80);
        let wide_render: String = entry
            .lines()
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        let narrow_count = entry.wrapped_line_count(40);
        let narrow_render: String = entry
            .lines()
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            narrow_count >= wide_count,
            "narrow count {narrow_count} should be >= wide count {wide_count}"
        );
        assert_ne!(
            wide_render, narrow_render,
            "rendered content must differ between widths 80 and 40"
        );
        let wide_max = wide_render.lines().map(|l| l.width()).max().unwrap_or(0);
        let narrow_max = narrow_render.lines().map(|l| l.width()).max().unwrap_or(0);
        assert!(
            narrow_max < wide_max,
            "narrow max line width {narrow_max} should be < wide max {wide_max}"
        );
    }

    #[test]
    fn cached_lines_rebuild_on_width_change() {
        let table = "| Column One | Column Two | Column Three | Column Four |\n\
                     |------------|------------|--------------|-------------|\n\
                     | Long value here | Another long value | Yet another long value | Final long value |";
        let mut entry = ConversationEntry::new(ConversationRole::Assistant, table.to_string());
        let wide_count = entry.wrapped_line_count(80);
        let lines_at_80 = entry.lines().len();
        let render_at_80: String = entry
            .lines()
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");

        let narrow_count = entry.wrapped_line_count(40);
        let lines_at_40 = entry.lines().len();
        let render_at_40: String = entry
            .lines()
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            narrow_count >= wide_count,
            "narrow wrapped count {narrow_count} should be >= wide {wide_count}"
        );
        assert!(
            lines_at_40 >= lines_at_80,
            "lines_at_40={lines_at_40} should be >= lines_at_80={lines_at_80}"
        );
        // Content must actually differ — a bug producing more lines at the same
        // width would not change the rendered text.
        assert_ne!(
            render_at_80, render_at_40,
            "cached_lines content must change after width-driven rebuild"
        );
        // Re-rendering at width 80 again should match the original wide render
        // (cache is rebuilt cleanly, not corrupted).
        let _ = entry.wrapped_line_count(80);
        let render_at_80_again: String = entry
            .lines()
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            render_at_80, render_at_80_again,
            "rebuilding back to width 80 should produce identical content"
        );
    }
}
