use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::borrow::Cow;

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

pub struct ConversationArea<'a> {
    entries: &'a [ConversationEntry],
    current_response: &'a str,
    scroll_offset: u16,
    viewport_height: u16,
}

impl<'a> ConversationArea<'a> {
    pub fn new(
        entries: &'a [ConversationEntry],
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

    pub fn lines(&self) -> Vec<Line<'_>> {
        let mut lines = Vec::new();
        for entry in self.entries {
            lines.push(Line::from(Span::styled(
                format!("{}:", entry.role.display_label()),
                Style::default().fg(entry.role.color()),
            )));
            let display_content = maybe_truncate(&entry.content, &entry.role);
            for line in display_content.lines() {
                lines.push(Line::from(format!("  {line}")));
            }
            lines.push(Line::from(""));
        }

        if !self.current_response.is_empty() {
            lines.push(Line::from(Span::styled(
                "Assistant:",
                Style::default().fg(Color::Blue),
            )));
            for line in self.current_response.lines() {
                lines.push(Line::from(format!("  {line}")));
            }
        }

        lines
    }

    pub fn max_scroll(&self, text_width: u16) -> u16 {
        let conv_lines = self.lines();
        let total_visual: u16 = conv_lines
            .iter()
            .map(|line| {
                if text_width == 0 {
                    1u16
                } else {
                    let w = line.width() as u16;
                    if w == 0 { 1u16 } else { w.div_ceil(text_width) }
                }
            })
            .sum();
        total_visual.saturating_sub(self.viewport_height)
    }

    pub fn render(&self, frame: &mut ratatui::Frame, area: Rect, text_width: u16) {
        let conv_lines = self.lines();
        let visible_height = area.height.saturating_sub(2);
        let total_visual: u16 = conv_lines
            .iter()
            .map(|line| {
                if text_width == 0 {
                    1u16
                } else {
                    let w = line.width() as u16;
                    if w == 0 { 1u16 } else { w.div_ceil(text_width) }
                }
            })
            .sum();
        let auto_scroll = total_visual.saturating_sub(visible_height);
        let scroll_row = auto_scroll.saturating_sub(self.scroll_offset.min(auto_scroll));

        let conversation = Paragraph::new(conv_lines)
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
        let area = ConversationArea::new(&[], "", 0, 10);
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
        let area = ConversationArea::new(&entries, "", 0, 10);
        let lines = area.lines();
        assert_eq!(lines.len(), 6);
    }

    #[test]
    fn lines_with_current_response() {
        let area = ConversationArea::new(&[], "streaming text", 0, 10);
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
        let area_widget = ConversationArea::new(&[], "", 0, 20);
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
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        let area_widget = ConversationArea::new(&entries, "", 0, 20);
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
        let area_widget = ConversationArea::new(&[], "Streaming response...", 0, 20);
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
        let area = ConversationArea::new(&entries, "", 0, 10);
        let max = area.max_scroll(58);
        assert!(max > 0, "should have scrollable content");
    }

    #[test]
    fn max_scroll_zero_when_content_fits() {
        let entries = vec![ConversationEntry {
            role: ConversationRole::User,
            content: "short".to_string(),
        }];
        let area = ConversationArea::new(&entries, "", 0, 100);
        let max = area.max_scroll(58);
        assert_eq!(max, 0);
    }

    #[test]
    fn tool_use_display_content_formats() {
        let result = tool_use_display_content("bash", &serde_json::json!({"command": "ls"}));
        assert!(result.contains("bash"));
        assert!(result.contains("command"));
    }
}
