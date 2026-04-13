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
}

/// Caches the rendered lines and accurate wrapped line count for finalized
/// conversation entries.
///
/// `Paragraph::line_count()` uses ratatui's internal `WordWrapper` to get the
/// exact number of visual lines after wrapping, but it is expensive to call.
/// Markdown parsing via `entry_lines()` is also expensive for large
/// conversations. This cache is only invalidated when entries are added or the
/// text width changes — never on every render or every streaming token.
pub struct LineCountCache {
    entry_count: usize,
    text_width: u16,
    cached_total: u16,
    cached_lines: Vec<Line<'static>>,
}

impl Default for LineCountCache {
    fn default() -> Self {
        Self::new()
    }
}

impl LineCountCache {
    pub fn new() -> Self {
        Self {
            entry_count: 0,
            text_width: 0,
            cached_total: 0,
            cached_lines: Vec::new(),
        }
    }

    pub fn get(&mut self, entries: &[ConversationEntry], text_width: u16) -> u16 {
        if entries.len() != self.entry_count {
            self.rebuild_lines(entries);
        }
        if text_width != self.text_width {
            self.recount(text_width);
        }
        self.cached_total
    }

    pub fn lines(&self) -> &[Line<'static>] {
        &self.cached_lines
    }

    fn rebuild_lines(&mut self, entries: &[ConversationEntry]) {
        self.cached_lines = entry_lines_owned(entries);
        self.entry_count = entries.len();
        self.text_width = u16::MAX;
        self.cached_total = 0;
    }

    fn recount(&mut self, text_width: u16) {
        self.cached_total = paragraph_line_count(&self.cached_lines, text_width);
        self.text_width = text_width;
    }
}

fn paragraph_line_count(lines: &[Line<'_>], text_width: u16) -> u16 {
    if text_width == 0 {
        return lines.len() as u16;
    }
    let paragraph = Paragraph::new(lines.to_vec()).wrap(Wrap { trim: false });
    paragraph.line_count(text_width) as u16
}

fn estimate_line_count(lines: &[Line<'_>], text_width: u16) -> u16 {
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

fn entry_lines_owned(entries: &[ConversationEntry]) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for entry in entries {
        lines.push(Line::from(Span::styled(
            format!("{}:", entry.role.display_label()),
            Style::default().fg(entry.role.color()),
        )));
        let display_content = maybe_truncate(&entry.content, &entry.role);
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
        lines.push(Line::from(""));
    }
    lines
}

fn current_response_lines<'a>(current_response: &'a str) -> Vec<Line<'a>> {
    let mut lines = Vec::new();
    if !current_response.is_empty() {
        lines.push(Line::from(Span::styled(
            "Assistant:",
            Style::default().fg(Color::Blue),
        )));
        let rendered = markdown_to_text(current_response);
        for line in rendered.lines {
            let mut prefixed = Line::from(Span::raw("  "));
            prefixed.spans.extend(line.spans);
            lines.push(prefixed);
        }
    }
    lines
}

pub struct ConversationArea<'a> {
    cached_entry_lines: &'a [Line<'static>],
    current_response: &'a str,
    scroll_offset: u16,
    viewport_height: u16,
    cached_entry_line_count: u16,
}

impl<'a> ConversationArea<'a> {
    pub fn new(
        cached_entry_lines: &'a [Line<'static>],
        current_response: &'a str,
        scroll_offset: u16,
        viewport_height: u16,
        cached_entry_line_count: u16,
    ) -> Self {
        Self {
            cached_entry_lines,
            current_response,
            scroll_offset,
            viewport_height,
            cached_entry_line_count,
        }
    }

    pub fn lines(&self) -> Vec<Line<'_>> {
        let mut lines: Vec<Line<'_>> = self.cached_entry_lines.to_vec();
        lines.extend(current_response_lines(self.current_response));
        lines
    }

    fn total_visual_lines(&self, text_width: u16) -> u16 {
        let response_lines = current_response_lines(self.current_response);
        let response_count = estimate_line_count(&response_lines, text_width);
        self.cached_entry_line_count.saturating_add(response_count)
    }

    pub fn max_scroll(&self, text_width: u16) -> u16 {
        let total_visual = self.total_visual_lines(text_width);
        total_visual.saturating_sub(self.viewport_height)
    }

    pub fn render(&self, frame: &mut ratatui::Frame, area: Rect, text_width: u16) {
        let visible_height = area.height.saturating_sub(2);
        let total_visual = self.total_visual_lines(text_width);
        let auto_scroll = total_visual.saturating_sub(visible_height);
        let scroll_row = auto_scroll.saturating_sub(self.scroll_offset.min(auto_scroll));

        let mut all_lines: Vec<Line<'_>> = self.cached_entry_lines.to_vec();
        all_lines.extend(current_response_lines(self.current_response));

        let conversation = Paragraph::new(all_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(CONVERSATION_TITLE),
            )
            .wrap(Wrap { trim: false })
            .scroll((scroll_row, 0));
        frame.render_widget(conversation, area);
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

    fn cached_lines(entries: &[ConversationEntry], text_width: u16) -> (u16, Vec<Line<'static>>) {
        let mut cache = LineCountCache::new();
        let count = cache.get(entries, text_width);
        (count, cache.lines().to_vec())
    }

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
    fn lines_empty_conversation() {
        let area = ConversationArea::new(&[], "", 0, 10, 0);
        let lines = area.lines();
        assert!(lines.is_empty());
    }

    #[test]
    fn lines_with_entries() {
        let entries = vec![
            ConversationEntry {
                role: ConversationRole::User,
                content: "hello".to_string(),
            },
            ConversationEntry {
                role: ConversationRole::Assistant,
                content: "world".to_string(),
            },
        ];
        let (_, lines) = cached_lines(&entries, 58);
        let area = ConversationArea::new(&lines, "", 0, 10, 0);
        let result = area.lines();
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn lines_with_current_response() {
        let area = ConversationArea::new(&[], "streaming text", 0, 10, 0);
        let lines = area.lines();
        assert!(!lines.is_empty());
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
            .unwrap()
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
        let area_widget = ConversationArea::new(&[], "", 0, 20, 0);
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_empty_conversation", terminal.backend());
    }

    #[test]
    fn render_with_entries() {
        let entries = vec![
            ConversationEntry {
                role: ConversationRole::User,
                content: "Hello".to_string(),
            },
            ConversationEntry {
                role: ConversationRole::Assistant,
                content: "Hi there!".to_string(),
            },
        ];
        let (count, lines) = cached_lines(&entries, 58);
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let area_widget = ConversationArea::new(&lines, "", 0, 20, count);
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_with_entries", terminal.backend());
    }

    #[test]
    fn render_with_current_response() {
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let area_widget = ConversationArea::new(&[], "Streaming response...", 0, 20, 0);
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_with_current_response", terminal.backend());
    }

    #[test]
    fn max_scroll_with_many_entries() {
        let entries: Vec<ConversationEntry> = (0..40)
            .map(|i| ConversationEntry {
                role: ConversationRole::User,
                content: format!("line {i}"),
            })
            .collect();
        let (count, lines) = cached_lines(&entries, 58);
        let area = ConversationArea::new(&lines, "", 0, 10, count);
        let max = area.max_scroll(58);
        assert!(max > 0, "should have scrollable content");
    }

    #[test]
    fn max_scroll_zero_when_content_fits() {
        let entries = vec![ConversationEntry {
            role: ConversationRole::User,
            content: "short".to_string(),
        }];
        let (count, lines) = cached_lines(&entries, 58);
        let area = ConversationArea::new(&lines, "", 0, 100, count);
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
        let entries = vec![ConversationEntry {
            role: ConversationRole::Assistant,
            content: "This is **bold** text".to_string(),
        }];
        let (count, lines) = cached_lines(&entries, 58);
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let area_widget = ConversationArea::new(&lines, "", 0, 20, count);
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_markdown_bold_text", terminal.backend());
    }

    #[test]
    fn render_markdown_code_block() {
        let entries = vec![ConversationEntry {
            role: ConversationRole::Assistant,
            content: "```rust\nfn main() {}\n```".to_string(),
        }];
        let (count, lines) = cached_lines(&entries, 58);
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let area_widget = ConversationArea::new(&lines, "", 0, 20, count);
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_markdown_code_block", terminal.backend());
    }

    #[test]
    fn render_markdown_heading() {
        let entries = vec![ConversationEntry {
            role: ConversationRole::Assistant,
            content: "# Hello World\nSome content".to_string(),
        }];
        let (count, lines) = cached_lines(&entries, 58);
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let area_widget = ConversationArea::new(&lines, "", 0, 20, count);
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_markdown_heading", terminal.backend());
    }

    #[test]
    fn render_markdown_list() {
        let entries = vec![ConversationEntry {
            role: ConversationRole::Assistant,
            content: "- item one\n- item two\n- item three".to_string(),
        }];
        let (count, lines) = cached_lines(&entries, 58);
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let area_widget = ConversationArea::new(&lines, "", 0, 20, count);
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_markdown_list", terminal.backend());
    }

    #[test]
    fn render_user_markdown_italic() {
        let entries = vec![ConversationEntry {
            role: ConversationRole::User,
            content: "This is *italic* text".to_string(),
        }];
        let (count, lines) = cached_lines(&entries, 58);
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let area_widget = ConversationArea::new(&lines, "", 0, 20, count);
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!("render_user_markdown_italic", terminal.backend());
    }

    #[test]
    fn render_tool_entry_not_rendered_as_markdown() {
        let entries = vec![ConversationEntry {
            role: ConversationRole::ToolUse,
            content: "bash\n  {\"command\": \"ls\"}".to_string(),
        }];
        let (count, lines) = cached_lines(&entries, 58);
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let area_widget = ConversationArea::new(&lines, "", 0, 20, count);
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        insta::assert_snapshot!(
            "render_tool_entry_not_rendered_as_markdown",
            terminal.backend()
        );
    }

    #[test]
    fn line_count_cache_returns_same_value_on_repeated_calls() {
        let entries = vec![
            ConversationEntry {
                role: ConversationRole::User,
                content: "hello world".to_string(),
            },
            ConversationEntry {
                role: ConversationRole::Assistant,
                content: "hi there".to_string(),
            },
        ];
        let mut cache = LineCountCache::new();
        let first = cache.get(&entries, 58);
        let second = cache.get(&entries, 58);
        assert_eq!(first, second);
    }

    #[test]
    fn line_count_cache_invalidates_on_new_entry() {
        let mut entries = vec![ConversationEntry {
            role: ConversationRole::User,
            content: "hello".to_string(),
        }];
        let mut cache = LineCountCache::new();
        let before = cache.get(&entries, 58);
        entries.push(ConversationEntry {
            role: ConversationRole::Assistant,
            content: "world".to_string(),
        });
        let after = cache.get(&entries, 58);
        assert!(after > before, "adding entry should increase line count");
    }

    #[test]
    fn line_count_cache_invalidates_on_width_change() {
        let entries = vec![ConversationEntry {
            role: ConversationRole::User,
            content: "a long message that should wrap at narrow widths but not at wide ones"
                .to_string(),
        }];
        let mut cache = LineCountCache::new();
        let wide = cache.get(&entries, 80);
        let narrow = cache.get(&entries, 20);
        assert!(
            narrow > wide,
            "narrow width should produce more lines: narrow={narrow}, wide={wide}"
        );
    }

    #[test]
    fn max_scroll_accounts_for_wrapping() {
        let entries = vec![ConversationEntry {
            role: ConversationRole::User,
            content: "a ".repeat(100),
        }];
        let (count_wide, lines_wide) = cached_lines(&entries, 80);
        let (count_narrow, lines_narrow) = cached_lines(&entries, 20);
        let wide = ConversationArea::new(&lines_wide, "", 0, 5, count_wide);
        let narrow = ConversationArea::new(&lines_narrow, "", 0, 5, count_narrow);
        assert!(
            narrow.max_scroll(20) > wide.max_scroll(80),
            "narrow viewport should need more scroll"
        );
    }

    #[test]
    fn render_long_content_scrolls_to_bottom_showing_last_entry() {
        let mut entries: Vec<ConversationEntry> = (0..20)
            .map(|i| ConversationEntry {
                role: ConversationRole::User,
                content: format!("message {i}"),
            })
            .collect();
        entries.push(ConversationEntry {
            role: ConversationRole::Assistant,
            content: "final answer".to_string(),
        });

        let backend = ratatui::backend::TestBackend::new(60, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let (count, lines) = cached_lines(&entries, 58);
        let area_widget = ConversationArea::new(&lines, "", 0, 8, count);
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 10);
                area_widget.render(frame, rect, 58);
            })
            .expect("draw");

        let buffer = terminal.backend().buffer().clone();
        let mut found_final = false;
        for y in 0..buffer.area.height {
            let mut row_text = String::new();
            for x in 0..buffer.area.width {
                let cell = &buffer[(x, y)];
                row_text.push_str(cell.symbol());
            }
            if row_text.contains("final answer") {
                found_final = true;
                break;
            }
        }
        assert!(
            found_final,
            "the last entry should be visible when auto-scrolled to bottom"
        );
    }
}
