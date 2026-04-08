use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};

const CONVERSATION_TITLE: &str = "Conversation";
const TOOL_RESULT_TRUNCATE_CHARS: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

fn conversation_block() -> Block<'static> {
    Block::bordered().title(CONVERSATION_TITLE)
}

pub struct ConversationArea<'a> {
    paragraph: Paragraph<'a>,
    raw_lines: Vec<Line<'a>>,
    scroll_offset: u16,
}

impl<'a> ConversationArea<'a> {
    pub fn new() -> Self {
        let lines = vec![Line::raw("")];
        let paragraph = Paragraph::new(lines.clone())
            .block(conversation_block())
            .wrap(Wrap { trim: false });
        Self {
            paragraph,
            raw_lines: lines,
            scroll_offset: 0,
        }
    }

    pub fn update_content(&mut self, conversation: &[ConversationEntry], current_response: &str) {
        let mut lines: Vec<Line<'a>> = Vec::new();

        for entry in conversation {
            let label_style = Style::default().fg(entry.role.color());
            lines.push(Line::from(Span::styled(
                format!("{}:", entry.role.display_label()),
                label_style,
            )));
            let display_content = maybe_truncate(&entry.content, entry.role);
            let content_style = Style::default().fg(entry.role.color());
            for line in display_content.lines() {
                lines.push(Line::from(Span::styled(format!("  {line}"), content_style)));
            }
            lines.push(Line::raw(""));
        }

        if !current_response.is_empty() {
            let style = Style::default().fg(ConversationRole::Assistant.color());
            lines.push(Line::from(Span::styled("Assistant:", style)));
            for line in current_response.lines() {
                lines.push(Line::from(Span::styled(format!("  {line}"), style)));
            }
        }

        if lines.is_empty() {
            lines.push(Line::raw(""));
        }

        self.raw_lines = lines.clone();
        self.paragraph = Paragraph::new(lines)
            .block(conversation_block())
            .wrap(Wrap { trim: false });
        self.scroll_offset = u16::MAX;
    }

    pub fn scroll_half_page_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(half_page_size());
    }

    pub fn scroll_half_page_down(&mut self) {
        let half = half_page_size();
        self.scroll_offset = self.scroll_offset.saturating_add(half);
    }

    pub fn render(&mut self, frame: &mut ratatui::Frame, area: Rect) {
        let inner = conversation_block().inner(area);
        let total_visual = visual_line_count(&self.raw_lines, inner.width);
        let max_scroll = max_scroll_for(total_visual, inner.height);
        self.scroll_offset = std::cmp::min(self.scroll_offset, max_scroll);
        let scrolled = self.paragraph.clone().scroll((self.scroll_offset, 0));
        frame.render_widget(scrolled, area);
    }

    pub fn content_text(&self) -> String {
        self.raw_lines
            .iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for ConversationArea<'_> {
    fn default() -> Self {
        Self::new()
    }
}

fn visual_line_count(lines: &[Line<'_>], width: u16) -> usize {
    let inner_width = width as usize;
    if inner_width == 0 {
        return lines.len();
    }
    lines
        .iter()
        .map(|line| {
            let w = line.width();
            if w == 0 { 1 } else { w.div_ceil(inner_width) }
        })
        .sum()
}

fn half_page_size() -> u16 {
    11u16
}

fn max_scroll_for(total_visual_lines: usize, viewport_height: u16) -> u16 {
    let total: u16 = total_visual_lines.try_into().unwrap_or(u16::MAX);
    total.saturating_sub(viewport_height).saturating_sub(1)
}

fn maybe_truncate(content: &str, role: ConversationRole) -> std::borrow::Cow<'_, str> {
    if role != ConversationRole::ToolResult {
        return std::borrow::Cow::Borrowed(content);
    }
    let mut chars = content.chars();
    let head: String = (&mut chars).take(TOOL_RESULT_TRUNCATE_CHARS).collect();
    if chars.next().is_some() {
        std::borrow::Cow::Owned(format!("{head}...[truncated]"))
    } else {
        std::borrow::Cow::Borrowed(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_entry(content: &str) -> ConversationEntry {
        ConversationEntry {
            role: ConversationRole::User,
            content: content.to_string(),
        }
    }

    fn assistant_entry(content: &str) -> ConversationEntry {
        ConversationEntry {
            role: ConversationRole::Assistant,
            content: content.to_string(),
        }
    }

    fn tool_result_entry(content: &str) -> ConversationEntry {
        ConversationEntry {
            role: ConversationRole::ToolResult,
            content: content.to_string(),
        }
    }

    #[test]
    fn new_conversation_area_is_empty() {
        let area = ConversationArea::new();
        assert_eq!(area.scroll_offset, 0);
    }

    #[test]
    fn update_content_with_single_entry() {
        let mut area = ConversationArea::new();
        let entries = vec![user_entry("Hello!")];
        area.update_content(&entries, "");
        assert_eq!(area.scroll_offset, u16::MAX);
    }

    #[test]
    fn update_content_with_multiple_entries() {
        let mut area = ConversationArea::new();
        let entries = vec![
            user_entry("Hi"),
            assistant_entry("Hello! How can I help you?"),
        ];
        area.update_content(&entries, "");
    }

    #[test]
    fn update_content_includes_current_response() {
        let mut area = ConversationArea::new();
        let entries = vec![user_entry("Tell me a story")];
        area.update_content(&entries, "Once upon a time");
    }

    #[test]
    fn update_content_without_entries_but_current_response() {
        let mut area = ConversationArea::new();
        area.update_content(&[], "Streaming text");
    }

    #[test]
    fn update_content_empty_clears_to_default() {
        let mut area = ConversationArea::new();
        let entries = vec![user_entry("Hello")];
        area.update_content(&entries, "");
        area.update_content(&[], "");
    }

    #[test]
    fn render_empty_conversation() {
        let mut area = ConversationArea::new();
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 80, 24);
                area.render(frame, rect);
            })
            .expect("draw");
        insta::assert_snapshot!("conversation_empty", terminal.backend());
    }

    #[test]
    fn render_with_user_and_assistant() {
        let mut area = ConversationArea::new();
        let entries = vec![
            user_entry("Hello!"),
            assistant_entry("Hi there! How can I help you?"),
        ];
        area.update_content(&entries, "");
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 80, 24);
                area.render(frame, rect);
            })
            .expect("draw");
        insta::assert_snapshot!("conversation_user_assistant", terminal.backend());
    }

    #[test]
    fn render_with_streaming_response() {
        let mut area = ConversationArea::new();
        let entries = vec![user_entry("Tell me a story")];
        area.update_content(&entries, "Once upon a time");
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 80, 24);
                area.render(frame, rect);
            })
            .expect("draw");
        insta::assert_snapshot!("conversation_streaming", terminal.backend());
    }

    #[test]
    fn render_with_tool_result_truncation() {
        let mut area = ConversationArea::new();
        let long_output = "x".repeat(500);
        let entries = vec![tool_result_entry(&long_output)];
        area.update_content(&entries, "");
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 80, 24);
                area.render(frame, rect);
            })
            .expect("draw");
        insta::assert_snapshot!("conversation_tool_result_truncated", terminal.backend());
    }

    #[test]
    fn maybe_truncate_borrows_short_tool_result() {
        let content = "short output";
        let result = maybe_truncate(content, ConversationRole::ToolResult);
        assert!(matches!(result, std::borrow::Cow::Borrowed(_)));
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
            let result = maybe_truncate(&content, role);
            assert!(
                matches!(result, std::borrow::Cow::Borrowed(_)),
                "expected borrow for {role:?}"
            );
        }
    }

    #[test]
    fn maybe_truncate_handles_multibyte_utf8() {
        let emoji = "🦀".repeat(300);
        let result = maybe_truncate(&emoji, ConversationRole::ToolResult);
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
        let result = maybe_truncate(&content, ConversationRole::ToolResult);
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
    }

    #[test]
    fn scroll_half_page_up_does_not_underflow() {
        let mut area = ConversationArea::new();
        assert_eq!(area.scroll_offset, 0);
        area.scroll_half_page_up();
        assert_eq!(area.scroll_offset, 0);
    }

    #[test]
    fn scroll_half_page_down_increases_offset() {
        let mut area = ConversationArea::new();
        let entries: Vec<ConversationEntry> =
            (0..100).map(|i| user_entry(&format!("line {i}"))).collect();
        area.update_content(&entries, "");
        area.scroll_half_page_down();
        assert!(area.scroll_offset > 0);
    }

    #[test]
    fn render_with_colored_roles() {
        let mut area = ConversationArea::new();
        let entries = vec![user_entry("Hello!"), assistant_entry("Hi there!")];
        area.update_content(&entries, "");
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 80, 24);
                area.render(frame, rect);
            })
            .expect("draw");
        insta::assert_snapshot!("conversation_colored_roles", terminal.backend());
    }

    #[test]
    fn visual_line_count_single_line_fits() {
        let lines = vec![Line::raw("hello")];
        assert_eq!(visual_line_count(&lines, 80), 1);
    }

    #[test]
    fn visual_line_count_wraps_long_line() {
        let lines = vec![Line::raw("abcdefghijklmnopqrstuvwxyz")];
        assert_eq!(visual_line_count(&lines, 10), 3);
    }

    #[test]
    fn visual_line_count_empty_line_counts_as_one() {
        let lines = vec![Line::raw("")];
        assert_eq!(visual_line_count(&lines, 80), 1);
    }

    #[test]
    fn max_scroll_for_large_content() {
        assert_eq!(max_scroll_for(100, 22), 77);
    }

    #[test]
    fn max_scroll_for_small_content() {
        assert_eq!(max_scroll_for(10, 22), 0);
    }
}
