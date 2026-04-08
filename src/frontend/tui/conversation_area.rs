use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders};
use ratatui_textarea::{CursorMove, Scrolling, TextArea, WrapMode};
use std::borrow::Cow;

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
    pub fn color(self) -> Color {
        match self {
            ConversationRole::User => Color::Green,
            ConversationRole::Assistant => Color::Blue,
            ConversationRole::Error => Color::Red,
            ConversationRole::ToolUse => Color::Cyan,
            ConversationRole::ToolResult => Color::Yellow,
        }
    }

    fn label(self) -> &'static str {
        match self {
            ConversationRole::User => "You",
            ConversationRole::Assistant => "Assistant",
            ConversationRole::Error => "Error",
            ConversationRole::ToolUse => "[Tool]",
            ConversationRole::ToolResult => "[Result]",
        }
    }
}

pub struct ConversationEntry {
    pub role: ConversationRole,
    pub content: String,
}

pub struct ConversationArea<'a> {
    textarea: TextArea<'a>,
}

impl<'a> ConversationArea<'a> {
    pub fn new() -> Self {
        let mut textarea = TextArea::default();
        textarea.set_wrap_mode(WrapMode::WordOrGlyph);
        textarea.set_cursor_style(Style::default());
        textarea.set_cursor_line_style(Style::default());
        textarea.set_block(Block::default().borders(Borders::ALL).title("Conversation"));
        Self { textarea }
    }

    pub fn update_content(&mut self, conversation: &[ConversationEntry], current_response: &str) {
        let lines = build_lines(conversation, current_response);
        self.textarea = TextArea::new(lines);
        self.textarea.set_wrap_mode(WrapMode::WordOrGlyph);
        self.textarea.set_cursor_style(Style::default());
        self.textarea.set_cursor_line_style(Style::default());
        self.textarea
            .set_block(Block::default().borders(Borders::ALL).title("Conversation"));
        self.textarea.move_cursor(CursorMove::Bottom);
    }

    pub fn scroll_half_page_up(&mut self) {
        self.textarea.scroll(Scrolling::HalfPageUp);
    }

    pub fn scroll_half_page_down(&mut self) {
        self.textarea.scroll(Scrolling::HalfPageDown);
    }

    pub fn render(&self, frame: &mut ratatui::Frame, area: Rect) {
        frame.render_widget(&self.textarea, area);
    }

    pub fn content_text(&self) -> String {
        self.textarea.lines().join("\n")
    }
}

impl Default for ConversationArea<'_> {
    fn default() -> Self {
        Self::new()
    }
}

fn build_lines(conversation: &[ConversationEntry], current_response: &str) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();

    for entry in conversation {
        lines.push(format!("{}:", entry.role.label()));
        let display = maybe_truncate(&entry.content, entry.role);
        for line in display.lines() {
            lines.push(format!("  {line}"));
        }
        lines.push(String::new());
    }

    if !current_response.is_empty() {
        lines.push("Assistant:".to_string());
        for line in current_response.lines() {
            lines.push(format!("  {line}"));
        }
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

fn maybe_truncate(content: &str, role: ConversationRole) -> Cow<'_, str> {
    if role != ConversationRole::ToolResult {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(role: ConversationRole, content: &str) -> ConversationEntry {
        ConversationEntry {
            role,
            content: content.to_string(),
        }
    }

    fn user_entry(content: &str) -> ConversationEntry {
        entry(ConversationRole::User, content)
    }

    fn assistant_entry(content: &str) -> ConversationEntry {
        entry(ConversationRole::Assistant, content)
    }

    fn tool_result_entry(content: &str) -> ConversationEntry {
        entry(ConversationRole::ToolResult, content)
    }

    #[test]
    fn new_conversation_area_has_empty_content() {
        let area = ConversationArea::new();
        let lines = area.textarea.lines();
        assert!(
            lines.is_empty() || lines.iter().all(|l| l.is_empty()),
            "expected empty content, got: {lines:?}"
        );
    }

    #[test]
    fn update_content_with_user_entry_includes_label() {
        let mut area = ConversationArea::new();
        area.update_content(&[user_entry("hello")], "");
        let content = area.content_text();
        assert!(content.contains("You:"), "missing 'You:' label");
        assert!(content.contains("hello"), "missing message content");
    }

    #[test]
    fn update_content_with_assistant_entry_includes_label() {
        let mut area = ConversationArea::new();
        area.update_content(&[assistant_entry("world")], "");
        let content = area.content_text();
        assert!(content.contains("Assistant:"), "missing 'Assistant:' label");
        assert!(content.contains("world"), "missing message content");
    }

    #[test]
    fn update_content_with_current_response_appends_streaming() {
        let mut area = ConversationArea::new();
        area.update_content(&[], "streaming token");
        let content = area.content_text();
        assert!(content.contains("Assistant:"), "missing streaming label");
        assert!(content.contains("streaming token"), "missing content");
    }

    #[test]
    fn update_content_empty_clears_to_blank() {
        let mut area = ConversationArea::new();
        area.update_content(&[], "");
        let lines = area.textarea.lines();
        assert!(
            lines.is_empty() || lines.iter().all(|l| l.is_empty()),
            "expected empty content, got: {lines:?}"
        );
    }

    #[test]
    fn update_content_truncates_long_tool_result() {
        let long_content = "x".repeat(500);
        let mut area = ConversationArea::new();
        area.update_content(&[tool_result_entry(&long_content)], "");
        assert!(
            area.content_text().contains("...[truncated]"),
            "long tool result should be truncated"
        );
    }

    #[test]
    fn update_content_does_not_truncate_user_role() {
        let long_content = "x".repeat(500);
        let mut area = ConversationArea::new();
        area.update_content(&[user_entry(&long_content)], "");
        assert!(
            !area.content_text().contains("...[truncated]"),
            "user content should not be truncated"
        );
    }

    #[test]
    fn update_content_preserves_entry_order() {
        let mut area = ConversationArea::new();
        area.update_content(&[user_entry("first"), assistant_entry("second")], "");
        let content = area.content_text();
        let user_pos = content.find("You:").expect("missing You:");
        let asst_pos = content.find("Assistant:").expect("missing Assistant:");
        assert!(
            user_pos < asst_pos,
            "User entry should come before Assistant"
        );
    }

    #[test]
    fn scroll_half_page_up_does_not_panic_when_empty() {
        let mut area = ConversationArea::new();
        area.scroll_half_page_up();
    }

    #[test]
    fn scroll_half_page_down_does_not_panic_when_empty() {
        let mut area = ConversationArea::new();
        area.scroll_half_page_down();
    }

    #[test]
    fn maybe_truncate_borrows_short_tool_result() {
        let content = "short output";
        let result = maybe_truncate(content, ConversationRole::ToolResult);
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
            let result = maybe_truncate(&content, role);
            assert!(
                matches!(result, Cow::Borrowed(_)),
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
            .expect("has truncated suffix")
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
    fn render_produces_output() {
        let mut area = ConversationArea::new();
        area.update_content(&[user_entry("hello world")], "");

        let backend = ratatui::backend::TestBackend::new(60, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                let rect = Rect::new(0, 0, 60, 10);
                area.render(frame, rect);
            })
            .expect("draw");

        insta::assert_snapshot!("conversation_area_render", terminal.backend());
    }

    #[test]
    fn render_with_tool_roles_produces_output() {
        let mut area = ConversationArea::new();
        area.update_content(
            &[
                entry(ConversationRole::ToolUse, "bash\n  {\"command\": \"ls\"}"),
                tool_result_entry("file1.txt\nfile2.txt"),
            ],
            "",
        );

        let backend = ratatui::backend::TestBackend::new(60, 12);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                let rect = Rect::new(0, 0, 60, 12);
                area.render(frame, rect);
            })
            .expect("draw");

        insta::assert_snapshot!("conversation_area_tool_render", terminal.backend());
    }
}
