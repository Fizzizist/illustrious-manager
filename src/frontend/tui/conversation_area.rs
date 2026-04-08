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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolName {
    EditFile,
    WriteFile,
    Bash,
    Skill,
    Other(String),
}

impl ToolName {
    pub fn from_tool_name(name: &str) -> Self {
        match name {
            "edit_file" => ToolName::EditFile,
            "write_file" => ToolName::WriteFile,
            "bash" => ToolName::Bash,
            "skill" => ToolName::Skill,
            other => ToolName::Other(other.to_string()),
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            ToolName::EditFile => "edit_file",
            ToolName::WriteFile => "write_file",
            ToolName::Bash => "bash",
            ToolName::Skill => "skill",
            ToolName::Other(s) => s.as_str(),
        }
    }
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
    pub tool_name: Option<ToolName>,
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
            match entry.role {
                ConversationRole::Assistant => {
                    lines.push(Line::from(Span::styled(
                        format!("{}:", entry.role.display_label()),
                        Style::default().fg(entry.role.color()),
                    )));
                    let md_text = markdown_to_text(&entry.content);
                    for md_line in md_text.lines {
                        let mut prefixed = Line::from(Span::raw("  "));
                        prefixed.spans.extend(md_line.spans);
                        lines.push(prefixed);
                    }
                    lines.push(Line::from(""));
                }
                ConversationRole::ToolUse => {
                    lines.push(Line::from(Span::styled(
                        format!("{}:", entry.role.display_label()),
                        Style::default().fg(entry.role.color()),
                    )));
                    match entry.tool_name.as_ref() {
                        Some(ToolName::EditFile) => {
                            let input_str = extract_tool_input(&entry.content);
                            for dl in format_edit_file_diff(input_str) {
                                lines.push(dl);
                            }
                        }
                        Some(ToolName::WriteFile) => {
                            let input_str = extract_tool_input(&entry.content);
                            for dl in format_write_file_diff(input_str) {
                                lines.push(dl);
                            }
                        }
                        _ => {
                            let display_content = maybe_truncate(&entry.content, &entry.role);
                            for line in display_content.lines() {
                                lines.push(Line::from(format!("  {line}")));
                            }
                        }
                    }
                    lines.push(Line::from(""));
                }
                _ => {
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
            }
        }

        if !self.current_response.is_empty() {
            lines.push(Line::from(Span::styled(
                "Assistant:",
                Style::default().fg(Color::Blue),
            )));
            let md_text = markdown_to_text(self.current_response);
            for md_line in md_text.lines {
                let mut prefixed = Line::from(Span::raw("  "));
                prefixed.spans.extend(md_line.spans);
                lines.push(prefixed);
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

fn extract_tool_input(content: &str) -> &str {
    content
        .split_once('\n')
        .map(|(_, rest)| rest.trim())
        .unwrap_or("")
}

fn format_diff_block(header: &str, removals: &[&str], additions: &[&str]) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        format!("  {header}"),
        Style::default().fg(Color::Cyan),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::raw("  ```diff")));
    for line in removals.iter().flat_map(|s| s.lines()) {
        lines.push(Line::from(Span::styled(
            format!("  -{line}"),
            Style::default().fg(Color::Red),
        )));
    }
    for line in additions.iter().flat_map(|s| s.lines()) {
        lines.push(Line::from(Span::styled(
            format!("  +{line}"),
            Style::default().fg(Color::Green),
        )));
    }
    lines.push(Line::from(Span::raw("  ```")));
    lines
}

fn format_edit_file_diff(input_str: &str) -> Vec<Line<'static>> {
    let input: serde_json::Value =
        serde_json::from_str(input_str).unwrap_or(serde_json::Value::Null);

    let path = input
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let old_string = input
        .get("old_string")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let new_string = input
        .get("new_string")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    format_diff_block(&format!("edit_file: {path}"), &[old_string], &[new_string])
}

fn format_write_file_diff(input_str: &str) -> Vec<Line<'static>> {
    let input: serde_json::Value =
        serde_json::from_str(input_str).unwrap_or(serde_json::Value::Null);

    let path = input
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");

    format_diff_block(&format!("write_file: {path}"), &[], &[content])
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
                tool_name: None,
                content: "hello".to_string(),
            },
            ConversationEntry {
                role: ConversationRole::Assistant,
                tool_name: None,
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
                tool_name: None,
                content: "Hello".to_string(),
            },
            ConversationEntry {
                role: ConversationRole::Assistant,
                tool_name: None,
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
                tool_name: None,
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
            tool_name: None,
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

    #[test]
    fn format_tool_use_edit_file_produces_diff() {
        let content = "edit_file\n  {\"path\":\"src/main.rs\",\"old_string\":\"old\\n\",\"new_string\":\"new\\n\"}";
        let entries = vec![ConversationEntry {
            role: ConversationRole::ToolUse,
            tool_name: Some(ToolName::EditFile),
            content: content.to_string(),
        }];
        let area = ConversationArea::new(&entries, "", 0, 10);
        let text: String = area
            .lines()
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("-old"),
            "diff should contain removed line: {text}"
        );
        assert!(
            text.contains("+new"),
            "diff should contain added line: {text}"
        );
        assert!(
            text.contains("edit_file: src/main.rs"),
            "should show tool name and path: {text}"
        );
    }

    #[test]
    fn format_tool_use_write_file_produces_diff() {
        let content = "write_file\n  {\"path\":\"src/lib.rs\",\"content\":\"fn main() {}\"}";
        let entries = vec![ConversationEntry {
            role: ConversationRole::ToolUse,
            tool_name: Some(ToolName::WriteFile),
            content: content.to_string(),
        }];
        let area = ConversationArea::new(&entries, "", 0, 10);
        let text: String = area
            .lines()
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("+fn main() {}"),
            "diff should contain added line: {text}"
        );
        assert!(
            text.contains("write_file: src/lib.rs"),
            "should show tool name and path: {text}"
        );
    }

    #[test]
    fn format_tool_use_unknown_tool_renders_plain() {
        let content = "bash\n  {\"command\":\"ls\"}";
        let entries = vec![ConversationEntry {
            role: ConversationRole::ToolUse,
            tool_name: Some(ToolName::Bash),
            content: content.to_string(),
        }];
        let area = ConversationArea::new(&entries, "", 0, 10);
        let text: String = area
            .lines()
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("bash"),
            "plain render should show tool name: {text}"
        );
        assert!(
            text.contains("command"),
            "plain render should show input: {text}"
        );
    }

    #[test]
    fn format_tool_use_edit_file_malformed_json() {
        let content = "edit_file\n  {{{not valid json}}}";
        let entries = vec![ConversationEntry {
            role: ConversationRole::ToolUse,
            tool_name: Some(ToolName::EditFile),
            content: content.to_string(),
        }];
        let area = ConversationArea::new(&entries, "", 0, 10);
        let text: String = area
            .lines()
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("edit_file: unknown"),
            "malformed JSON should fall back to unknown path: {text}"
        );
    }

    #[test]
    fn format_tool_use_write_file_malformed_json() {
        let content = "write_file\n  {{{garbage}}}";
        let entries = vec![ConversationEntry {
            role: ConversationRole::ToolUse,
            tool_name: Some(ToolName::WriteFile),
            content: content.to_string(),
        }];
        let area = ConversationArea::new(&entries, "", 0, 10);
        let text: String = area
            .lines()
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("write_file: unknown"),
            "malformed JSON should fall back to unknown path: {text}"
        );
    }

    #[test]
    fn format_tool_use_edit_file_empty_old_string() {
        let content = r#"edit_file
  {"path":"src/new.rs","old_string":"","new_string":"fn main() {}"}"#;
        let entries = vec![ConversationEntry {
            role: ConversationRole::ToolUse,
            tool_name: Some(ToolName::EditFile),
            content: content.to_string(),
        }];
        let area = ConversationArea::new(&entries, "", 0, 10);
        let text: String = area
            .lines()
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("+fn main() {}"),
            "should contain added line: {text}"
        );
        assert!(!text.contains("-"), "should have no removal lines: {text}");
    }

    #[test]
    fn format_tool_use_write_file_empty_content() {
        let content = r#"write_file
  {"path":"src/empty.rs","content":""}"#;
        let entries = vec![ConversationEntry {
            role: ConversationRole::ToolUse,
            tool_name: Some(ToolName::WriteFile),
            content: content.to_string(),
        }];
        let area = ConversationArea::new(&entries, "", 0, 10);
        let text: String = area
            .lines()
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("write_file: src/empty.rs"),
            "should show tool name and path: {text}"
        );
        assert!(
            !text.contains("+"),
            "should have no addition lines for empty content: {text}"
        );
    }

    #[test]
    fn lines_current_response_renders_markdown() {
        let area = ConversationArea::new(&[], "**bold** and *italic*", 0, 10);
        let lines = area.lines();
        let text: String = lines
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !text.contains("**"),
            "streaming response should render markdown, not raw syntax: {text}"
        );
        assert!(
            !text.contains("*italic*"),
            "streaming response should render italic markdown: {text}"
        );
    }

    #[test]
    fn render_markdown_assistant_bold() {
        let entries = vec![ConversationEntry {
            role: ConversationRole::Assistant,
            tool_name: None,
            content: "This is **bold** text.".to_string(),
        }];
        let area = ConversationArea::new(&entries, "", 0, 10);
        let lines = area.lines();
        let text: String = lines
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !text.contains("**"),
            "markdown syntax should be rendered, not raw: {text}"
        );
    }
}
