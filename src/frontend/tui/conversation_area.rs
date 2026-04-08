use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::{Block, Borders};
use ratatui_textarea::{CursorMove, Scrolling, TextArea, WrapMode};
use std::borrow::Cow;

use super::tui_app::{ConversationEntry, ConversationRole};

const TOOL_RESULT_TRUNCATE_CHARS: usize = 200;

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

    pub fn update(&mut self, conversation: &[ConversationEntry], current_response: &str) {
        let lines = build_lines(conversation, current_response);
        self.textarea = TextArea::new(lines);
        self.textarea.set_wrap_mode(WrapMode::WordOrGlyph);
        self.textarea.set_cursor_style(Style::default());
        self.textarea.set_cursor_line_style(Style::default());
        self.textarea
            .set_block(Block::default().borders(Borders::ALL).title("Conversation"));
        self.textarea.move_cursor(CursorMove::Bottom);
    }

    pub fn scroll_to_bottom(&mut self) {
        self.textarea.move_cursor(CursorMove::Bottom);
    }

    pub fn scroll_up_half(&mut self) {
        self.textarea.scroll(Scrolling::HalfPageUp);
    }

    pub fn scroll_down_half(&mut self) {
        self.textarea.scroll(Scrolling::HalfPageDown);
    }

    pub fn render(&self, frame: &mut ratatui::Frame, area: Rect) {
        frame.render_widget(&self.textarea, area);
    }
}

impl Default for ConversationArea<'_> {
    fn default() -> Self {
        Self::new()
    }
}

fn role_label(role: ConversationRole) -> &'static str {
    match role {
        ConversationRole::User => "You",
        ConversationRole::Assistant => "Assistant",
        ConversationRole::Error => "Error",
        ConversationRole::ToolUse => "[Tool]",
        ConversationRole::ToolResult => "[Result]",
    }
}

fn build_lines(conversation: &[ConversationEntry], current_response: &str) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();

    for entry in conversation {
        lines.push(format!("{}:", role_label(entry.role)));
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
    use crate::frontend::tui::tui_app::{ConversationEntry, ConversationRole};

    fn entry(role: ConversationRole, content: &str) -> ConversationEntry {
        ConversationEntry {
            role,
            content: content.to_string(),
        }
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
    fn update_includes_user_entry_label() {
        let mut area = ConversationArea::new();
        area.update(&[entry(ConversationRole::User, "hello")], "");
        let content = area.textarea.lines().join("\n");
        assert!(content.contains("You:"), "missing 'You:' label");
        assert!(content.contains("hello"), "missing message content");
    }

    #[test]
    fn update_includes_assistant_entry_label() {
        let mut area = ConversationArea::new();
        area.update(&[entry(ConversationRole::Assistant, "world")], "");
        let content = area.textarea.lines().join("\n");
        assert!(content.contains("Assistant:"), "missing 'Assistant:' label");
        assert!(content.contains("world"), "missing message content");
    }

    #[test]
    fn update_with_current_response_appends_streaming_content() {
        let mut area = ConversationArea::new();
        area.update(&[], "streaming token");
        let content = area.textarea.lines().join("\n");
        assert!(content.contains("Assistant:"), "missing streaming label");
        assert!(
            content.contains("streaming token"),
            "missing streaming content"
        );
    }

    #[test]
    fn update_with_empty_shows_empty_content() {
        let mut area = ConversationArea::new();
        area.update(&[], "");
        let lines = area.textarea.lines();
        assert!(
            lines.is_empty() || lines.iter().all(|l| l.is_empty()),
            "expected empty content, got: {lines:?}"
        );
    }

    #[test]
    fn update_truncates_long_tool_result() {
        let long_content = "x".repeat(500);
        let mut area = ConversationArea::new();
        area.update(&[entry(ConversationRole::ToolResult, &long_content)], "");
        let content = area.textarea.lines().join("\n");
        assert!(
            content.contains("...[truncated]"),
            "long tool result should be truncated"
        );
    }

    #[test]
    fn update_does_not_truncate_non_tool_result_roles() {
        let long_content = "x".repeat(500);
        let mut area = ConversationArea::new();
        area.update(&[entry(ConversationRole::User, &long_content)], "");
        let content = area.textarea.lines().join("\n");
        assert!(
            !content.contains("...[truncated]"),
            "user content should not be truncated"
        );
    }

    #[test]
    fn update_includes_multiple_entries_in_order() {
        let mut area = ConversationArea::new();
        area.update(
            &[
                entry(ConversationRole::User, "first"),
                entry(ConversationRole::Assistant, "second"),
            ],
            "",
        );
        let content = area.textarea.lines().join("\n");
        let user_pos = content.find("You:").expect("missing You:");
        let asst_pos = content.find("Assistant:").expect("missing Assistant:");
        assert!(
            user_pos < asst_pos,
            "User entry should come before Assistant"
        );
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
        area.update(&[entry(ConversationRole::User, "hello world")], "");

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
        area.update(
            &[
                entry(ConversationRole::ToolUse, "bash\n  {\"command\": \"ls\"}"),
                entry(ConversationRole::ToolResult, "file1.txt\nfile2.txt"),
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
