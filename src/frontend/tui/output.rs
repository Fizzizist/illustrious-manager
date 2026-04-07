use ratatui::style::Color;
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

pub fn maybe_truncate(content: &str, role: ConversationRole) -> Cow<'_, str> {
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

pub fn tool_use_display_content(name: &str, input: &serde_json::Value) -> String {
    format!(
        "{}\n  {}",
        name,
        serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string())
    )
}

pub fn build_conversation_lines<'a>(
    conversation: &'a [ConversationEntry],
    current_response: &'a str,
) -> Vec<ratatui::text::Line<'a>> {
    use ratatui::style::Style;
    use ratatui::text::{Line, Span};

    let mut lines = Vec::new();
    for entry in conversation {
        lines.push(Line::from(Span::styled(
            format!("{}:", entry.role.display_label()),
            Style::default().fg(entry.role.color()),
        )));
        let display_content = maybe_truncate(&entry.content, entry.role);
        for line in display_content.lines() {
            lines.push(Line::from(format!("  {line}")));
        }
        lines.push(Line::from(""));
    }

    if !current_response.is_empty() {
        lines.push(Line::from(Span::styled(
            "Assistant:",
            Style::default().fg(Color::Blue),
        )));
        for line in current_response.lines() {
            lines.push(Line::from(format!("  {line}")));
        }
    }

    lines
}
