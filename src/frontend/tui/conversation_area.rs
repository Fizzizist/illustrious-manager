use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::{Block, Borders};
use ratatui_textarea::{Scrolling, TextArea, WrapMode};

use super::tui_app::{ConversationEntry, ConversationRole};

const CONVERSATION_TITLE: &str = "Conversation";
const TOOL_RESULT_TRUNCATE_CHARS: usize = 200;

pub struct ConversationArea<'a> {
    pub(crate) textarea: TextArea<'a>,
}

impl<'a> ConversationArea<'a> {
    pub fn new() -> Self {
        let mut textarea = TextArea::new(vec![String::new()]);
        textarea.set_wrap_mode(WrapMode::WordOrGlyph);
        textarea.set_block(
            Block::default()
                .borders(Borders::ALL)
                .title(CONVERSATION_TITLE),
        );
        textarea.set_cursor_line_style(Style::default());
        textarea.set_cursor_style(Style::default());
        Self { textarea }
    }

    pub fn update_content(&mut self, conversation: &[ConversationEntry], current_response: &str) {
        let mut lines: Vec<String> = Vec::new();

        for entry in conversation {
            lines.push(format!("{}:", entry.role.display_label()));
            let display_content = maybe_truncate(&entry.content, entry.role);
            for line in display_content.lines() {
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

        self.textarea = TextArea::new(lines);
        self.textarea.set_wrap_mode(WrapMode::WordOrGlyph);
        self.textarea.set_block(
            Block::default()
                .borders(Borders::ALL)
                .title(CONVERSATION_TITLE),
        );
        self.textarea.set_cursor_line_style(Style::default());
        self.textarea.set_cursor_style(Style::default());
        self.textarea
            .move_cursor(ratatui_textarea::CursorMove::Bottom);
        self.textarea.move_cursor(ratatui_textarea::CursorMove::End);
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
}

impl Default for ConversationArea<'_> {
    fn default() -> Self {
        Self::new()
    }
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
        assert_eq!(area.textarea.lines().len(), 1);
    }

    #[test]
    fn update_content_with_single_entry() {
        let mut area = ConversationArea::new();
        let entries = vec![user_entry("Hello!")];
        area.update_content(&entries, "");
        let lines = area.textarea.lines();
        assert!(lines.iter().any(|l| l.contains("You:")));
        assert!(lines.iter().any(|l| l.contains("Hello!")));
    }

    #[test]
    fn update_content_with_multiple_entries() {
        let mut area = ConversationArea::new();
        let entries = vec![
            user_entry("Hi"),
            assistant_entry("Hello! How can I help you?"),
        ];
        area.update_content(&entries, "");
        let lines = area.textarea.lines();
        assert!(lines.iter().any(|l| l.contains("You:")));
        assert!(lines.iter().any(|l| l.contains("Assistant:")));
    }

    #[test]
    fn update_content_includes_current_response() {
        let mut area = ConversationArea::new();
        let entries = vec![user_entry("Tell me a story")];
        area.update_content(&entries, "Once upon a time");
        let lines = area.textarea.lines();
        assert!(lines.iter().any(|l| l.contains("Once upon a time")));
    }

    #[test]
    fn update_content_without_entries_but_current_response() {
        let mut area = ConversationArea::new();
        area.update_content(&[], "Streaming text");
        let lines = area.textarea.lines();
        assert!(lines.iter().any(|l| l.contains("Streaming text")));
    }

    #[test]
    fn update_content_empty_clears_to_default() {
        let mut area = ConversationArea::new();
        let entries = vec![user_entry("Hello")];
        area.update_content(&entries, "");
        area.update_content(&[], "");
        let lines = area.textarea.lines();
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn render_empty_conversation() {
        let area = ConversationArea::new();
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
}
