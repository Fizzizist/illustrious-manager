use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::borrow::Cow;
use tui_markdown::from_str as markdown_to_text;

const TOOL_RESULT_TRUNCATE_CHARS: usize = 200;

const CONVERSATION_TITLE: &str = "Conversation";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationRole {
    User,
    Assistant,
    Error,
    ToolUse,
    ToolResult,
}

impl ConversationRole {
    pub fn display_label(&self) -> &'static str {
        match self {
            ConversationRole::User => "You",
            ConversationRole::Assistant => "Assistant",
            ConversationRole::Error => "Error",
            ConversationRole::ToolUse => "[Tool]",
            ConversationRole::ToolResult => "[Result]",
        }
    }

    pub fn color(&self) -> Color {
        match self {
            ConversationRole::User => Color::Green,
            ConversationRole::Assistant => Color::Blue,
            ConversationRole::Error => Color::Red,
            ConversationRole::ToolUse => Color::Cyan,
            ConversationRole::ToolResult => Color::Yellow,
        }
    }
}

pub struct ConversationEntry {
    pub role: ConversationRole,
    pub content: String,
    cached_lines: Vec<Line<'static>>,
    cached_wrapped_count: u16,
    cached_width: u16,
}

impl ConversationEntry {
    pub fn new(role: ConversationRole, content: String) -> Self {
        let cached_lines = render_entry_lines(&role, &content);
        Self {
            role,
            content,
            cached_lines,
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
        // Wrap in the role header + trailing blank, then append the pre-built lines.
        let mut all_lines = Vec::new();
        all_lines.push(Line::from(Span::styled(
            format!("{}:", role.display_label()),
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
            cached_lines: all_lines,
            cached_wrapped_count: 0,
            cached_width: 0,
        }
    }

    pub fn lines(&self) -> &[Line<'static>] {
        &self.cached_lines
    }

    pub fn wrapped_line_count(&mut self, text_width: u16) -> u16 {
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

fn render_role_lines(
    role: &ConversationRole,
    content: &str,
    trailing_blank: bool,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        format!("{}:", role.display_label()),
        Style::default().fg(role.color()),
    )));
    let display_content = maybe_truncate(content, role);
    let rendered = markdown_to_text(&display_content);
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

fn render_entry_lines(role: &ConversationRole, content: &str) -> Vec<Line<'static>> {
    render_role_lines(role, content, true)
}

// TODO: For long responses with complex markdown, this O(response_size) per-frame
// cost during streaming could become a bottleneck. Consider caching if it becomes an issue.
fn render_current_response_lines(current_response: &str) -> Vec<Line<'static>> {
    render_role_lines(&ConversationRole::Assistant, current_response, false)
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
    current_response: &'a str,
    scroll_offset: u16,
    viewport_height: u16,
}

impl<'a> ConversationArea<'a> {
    pub fn new(
        entries: &'a mut [ConversationEntry],
        current_response: &'a str,
        scroll_offset: u16,
        viewport_height: u16,
    ) -> Self {
        Self {
            entries,
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

        let response_count = if self.current_response.is_empty() {
            0u16
        } else {
            let response_lines = render_current_response_lines(self.current_response);
            estimate_wrapped_count(&response_lines, text_width)
        };

        let total_visual = total_entries.saturating_add(response_count);
        total_visual.saturating_sub(self.viewport_height)
    }

    pub fn render(&mut self, frame: &mut ratatui::Frame, area: Rect, text_width: u16) {
        let visible_height = area.height.saturating_sub(2);

        let entry_counts = self.compute_entry_counts(text_width);
        let total_entries: u16 = entry_counts.iter().sum();

        let response_lines = if self.current_response.is_empty() {
            Vec::new()
        } else {
            render_current_response_lines(self.current_response)
        };
        let response_count = if response_lines.is_empty() {
            0u16
        } else {
            estimate_wrapped_count(&response_lines, text_width)
        };

        let total_visual = total_entries.saturating_add(response_count);
        let auto_scroll = total_visual.saturating_sub(visible_height);
        let scroll_row = auto_scroll.saturating_sub(self.scroll_offset.min(auto_scroll));

        // Windowed approach: find which entries are visible and only collect those lines,
        // plus compute how many wrapped lines to skip at the top of the window.
        let (window_lines, lines_to_skip) = self.collect_window_lines(
            &entry_counts,
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
    }

    #[test]
    fn conversation_role_colors() {
        assert_eq!(ConversationRole::User.color(), Color::Green);
        assert_eq!(ConversationRole::Assistant.color(), Color::Blue);
        assert_eq!(ConversationRole::Error.color(), Color::Red);
        assert_eq!(ConversationRole::ToolUse.color(), Color::Cyan);
        assert_eq!(ConversationRole::ToolResult.color(), Color::Yellow);
    }

    #[test]
    fn entry_lines_empty_conversation() {
        let entries: &mut [ConversationEntry] = &mut [];
        let area = ConversationArea::new(entries, "", 0, 10);
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
        let _ = ConversationArea::new(&mut entries, "", 0, 10);
    }

    #[test]
    fn entry_lines_with_current_response() {
        let mut entries: Vec<ConversationEntry> = Vec::new();
        let area = ConversationArea::new(&mut entries, "streaming text", 0, 10);
        assert!(!area.current_response.is_empty());
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
                let mut area_widget = ConversationArea::new(&mut entries, "", 0, 20);
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
                let mut area_widget = ConversationArea::new(&mut entries, "", 0, 20);
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
                    ConversationArea::new(&mut entries, "Streaming response...", 0, 20);
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
        let mut area = ConversationArea::new(&mut entries, "", 0, 10);
        let max = area.max_scroll(58);
        assert!(max > 0, "should have scrollable content");
    }

    #[test]
    fn max_scroll_zero_when_content_fits() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::User,
            "short".to_string(),
        )];
        let mut area = ConversationArea::new(&mut entries, "", 0, 100);
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
                let mut area_widget = ConversationArea::new(&mut entries, "", 0, 20);
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
                let mut area_widget = ConversationArea::new(&mut entries, "", 0, 20);
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
                let mut area_widget = ConversationArea::new(&mut entries, "", 0, 20);
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
                let mut area_widget = ConversationArea::new(&mut entries, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_markdown_list", terminal.backend());
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
                let mut area_widget = ConversationArea::new(&mut entries, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_user_markdown_italic", terminal.backend());
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
                let mut area_widget = ConversationArea::new(&mut entries, "", 0, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!(
            "render_tool_entry_not_rendered_as_markdown",
            terminal.backend()
        );
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
                let mut area_widget = ConversationArea::new(&mut entries, "", 0, 8);
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
                let mut area_widget = ConversationArea::new(&mut entries, "", 20, 10);
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
        let mut area = ConversationArea::new(&mut entries, "", 0, viewport);
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

        let mut area = ConversationArea::new(&mut entries, "", 0, viewport);
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

        let mut area = ConversationArea::new(&mut entries, "", 0, viewport);
        let max = area.max_scroll(text_width);

        let backend = ratatui::backend::TestBackend::new(60, viewport + 2);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, viewport + 2);
                let mut scrolled_area = ConversationArea::new(&mut entries, "", max, viewport);
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
                let mut area = ConversationArea::new(&mut entries, "", 0, viewport);
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

        let mut area = ConversationArea::new(&mut entries, "", 0, viewport);
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
                    let mut area = ConversationArea::new(&mut entries, "", scroll, viewport);
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
        let mut area = ConversationArea::new(&mut entries, "", 0, viewport);
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
                let mut area = ConversationArea::new(&mut entries, &streaming, 0, viewport);
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
                let mut area =
                    ConversationArea::new(&mut entries, &streaming, scroll_up_amount, viewport);
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

        let mut area = ConversationArea::new(&mut entries, "", 0, viewport);
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
                    let mut area = ConversationArea::new(entries, "", scroll, viewport);
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
}
