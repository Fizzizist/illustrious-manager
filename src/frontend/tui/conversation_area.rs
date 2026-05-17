use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::borrow::Cow;
use tui_markdown::from_str as markdown_to_text;

use super::markdown_tables::preprocess_tables;
use crate::frontend::tui::tui_app::SearchMatch;

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

fn truncate_to_width(s: &str, max_width: usize) -> String {
    use unicode_width::UnicodeWidthStr;

    if max_width == 0 {
        return String::new();
    }

    let total_width = UnicodeWidthStr::width(s);
    if total_width <= max_width {
        return s.to_string();
    }

    let prefix_width = 3;
    let available = max_width.saturating_sub(prefix_width);

    if available == 0 {
        return "...".to_string();
    }

    let chars: Vec<char> = s.chars().collect();
    let mut tail_width = 0usize;
    let mut tail_start = chars.len();

    for (i, &ch) in chars.iter().enumerate().rev() {
        let ch_width = UnicodeWidthStr::width(ch.to_string().as_str());
        if tail_width + ch_width > available {
            break;
        }
        tail_width += ch_width;
        tail_start = i;
    }

    let tail: String = chars[tail_start..].iter().collect();
    format!("...{tail}")
}

fn thinking_indicator_lines(preview: Option<&str>, text_width: u16) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        "[Thinking]:",
        Style::default().fg(ConversationRole::Thinking.color()),
    ))];

    if let Some(preview_text) = preview
        && !preview_text.trim().is_empty()
    {
        let max_width = text_width.saturating_sub(2) as usize;
        let truncated = truncate_to_width(preview_text, max_width);
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(truncated, Style::default().fg(Color::DarkGray)),
        ]));
    }

    lines.push(Line::from(""));
    lines
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
    thinking_preview: Option<String>,
    current_response: &'a str,
    scroll_offset: u16,
    viewport_height: u16,
}

impl<'a> ConversationArea<'a> {
    pub fn new(
        entries: &'a mut [ConversationEntry],
        is_thinking: bool,
        thinking_preview: Option<&str>,
        current_response: &'a str,
        scroll_offset: u16,
        viewport_height: u16,
    ) -> Self {
        Self {
            entries,
            is_thinking,
            thinking_preview: thinking_preview.map(|s| s.to_string()),
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

        let thinking_count = if self.is_thinking {
            let has_preview = self
                .thinking_preview
                .as_ref()
                .is_some_and(|s| !s.trim().is_empty());
            if has_preview { 3u16 } else { 2u16 }
        } else {
            0u16
        };

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

    pub fn render(
        &mut self,
        frame: &mut ratatui::Frame,
        area: Rect,
        text_width: u16,
        search_matches: Option<(&[SearchMatch], usize)>,
        search_input: Option<&str>,
        search_pattern: Option<&str>,
    ) {
        let visible_height = area.height.saturating_sub(2);

        let entry_counts = self.compute_entry_counts(text_width);
        let total_entries: u16 = entry_counts.iter().sum();

        let thinking_lines = if self.is_thinking {
            thinking_indicator_lines(self.thinking_preview.as_deref(), text_width)
        } else {
            Vec::new()
        };
        let thinking_count = if self.is_thinking {
            let has_preview = self
                .thinking_preview
                .as_ref()
                .is_some_and(|s| !s.trim().is_empty());
            if has_preview { 3u16 } else { 2u16 }
        } else {
            0u16
        };

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

        let (window_lines, lines_to_skip) = self.collect_window_lines(
            &entry_counts,
            &thinking_lines,
            &response_lines,
            scroll_row,
            visible_height,
            text_width,
        );

        let highlighted_lines = if let Some((matches, _current_idx)) = search_matches {
            let info = SearchHighlightInfo {
                matches,
                entries: self.entries,
                scroll_row,
                visible_height,
                entry_counts: &entry_counts,
            };
            apply_search_highlights(&window_lines, &info)
        } else {
            window_lines
        };

        let search_active = search_input.is_some() || search_pattern.is_some();
        let conv_height = if search_active {
            area.height.saturating_sub(1)
        } else {
            area.height
        };
        let conv_area = Rect {
            height: conv_height,
            ..area
        };

        let conversation = Paragraph::new(highlighted_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(CONVERSATION_TITLE),
            )
            .wrap(Wrap { trim: false })
            .scroll((lines_to_skip, 0));
        frame.render_widget(conversation, conv_area);

        if search_active {
            let search_bar_area = Rect {
                x: area.x,
                y: area.y + area.height - 1,
                width: area.width,
                height: 1,
            };
            let search_text = if let Some(input) = search_input {
                format!("/{}", input)
            } else if let Some(pattern) = search_pattern {
                let matches_ref = search_matches.expect("active search should have matches");
                let total = matches_ref.0.len();
                let current = matches_ref.1 + 1;
                format!("/{} [{}/{}]", pattern, current, total)
            } else {
                String::new()
            };
            let search_line = Line::from(Span::styled(
                search_text,
                Style::default().fg(Color::Yellow),
            ));
            let search_bar = Paragraph::new(search_line);
            frame.render_widget(search_bar, search_bar_area);
        }
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

/// Apply search highlights to rendered lines.
/// Scans each line's text for case-insensitive occurrences of the pattern,
/// and wraps matched substrings with highlight style (yellow bg, black fg).
struct SearchHighlightInfo<'a> {
    matches: &'a [SearchMatch],
    entries: &'a [ConversationEntry],
    scroll_row: u16,
    visible_height: u16,
    entry_counts: &'a [u16],
}

fn apply_search_highlights(
    lines: &[Line<'static>],
    info: &SearchHighlightInfo<'_>,
) -> Vec<Line<'static>> {
    if info.matches.is_empty() {
        return lines.to_vec();
    }

    let pattern = &info.entries[info.matches[0].entry_index].content
        [info.matches[0].byte_start..info.matches[0].byte_end];
    let pattern_lower = pattern.to_lowercase();

    let highlight_style = Style::default().fg(Color::Black).bg(Color::Yellow);

    // Build entry visual line ranges to determine which entry each window line belongs to
    let mut entry_ranges: Vec<(usize, usize)> = Vec::new();
    let mut cum: usize = 0;
    for &count in info.entry_counts.iter() {
        let start = cum;
        cum += count as usize;
        entry_ranges.push((start, cum));
    }

    // Determine which entries are visible in the window
    let window_start = info.scroll_row as usize;
    let _window_end = (info.scroll_row as usize).saturating_add(info.visible_height as usize);

    // For each line in the window, determine which entry it belongs to and
    // which content line within that entry, then check if any match falls on that line.
    let mut result_lines = Vec::with_capacity(lines.len());
    for (line_in_window, line) in lines.iter().enumerate() {
        let global_line = window_start + line_in_window;

        // Find which entry this global line belongs to
        let mut entry_idx: Option<usize> = None;
        for (idx, &(start, end)) in entry_ranges.iter().enumerate() {
            if global_line >= start && global_line < end {
                entry_idx = Some(idx);
                break;
            }
        }

        // Check if any match falls in this entry
        let has_match_in_entry =
            entry_idx.is_some_and(|idx| info.matches.iter().any(|m| m.entry_index == idx));

        if has_match_in_entry {
            result_lines.push(highlight_line(line, &pattern_lower, highlight_style));
        } else {
            result_lines.push(line.clone());
        }
    }

    result_lines
}

/// Highlight occurrences of `pattern_lower` (already lowercased) in a line's spans.
fn highlight_line(
    line: &Line<'static>,
    pattern_lower: &str,
    highlight_style: Style,
) -> Line<'static> {
    let full_text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    let full_text_lower = full_text.to_lowercase();
    let pattern_len = pattern_lower.len();

    // Find all case-insensitive occurrences in the concatenated text
    let mut occurrences: Vec<(usize, usize)> = Vec::new();
    let mut search_start = 0;
    while let Some(pos) = full_text_lower[search_start..].find(pattern_lower) {
        let abs_start = search_start + pos;
        let abs_end = abs_start + pattern_len;
        if full_text.is_char_boundary(abs_start) && full_text.is_char_boundary(abs_end) {
            occurrences.push((abs_start, abs_end));
        }
        search_start = abs_start + 1;
    }

    if occurrences.is_empty() {
        return line.clone();
    }

    // Split spans at occurrence boundaries
    let mut result_spans: Vec<Span<'static>> = Vec::new();
    let mut global_pos: usize = 0;

    for span in &line.spans {
        let span_text = span.content.as_ref();
        let span_start = global_pos;
        let span_end = global_pos + span_text.len();

        // Find occurrences that overlap this span
        let mut last_end: usize = 0;

        for &(occ_start, occ_end) in &occurrences {
            if occ_end <= span_start || occ_start >= span_end {
                continue;
            }

            // Convert to local positions within this span
            let local_start = occ_start.saturating_sub(span_start);
            let local_end = (occ_end).min(span_end) - span_start;

            // Ensure local positions are valid character boundaries in span_text
            if local_start > span_text.len() || local_end > span_text.len() {
                continue;
            }
            if !span_text.is_char_boundary(local_start) || !span_text.is_char_boundary(local_end) {
                continue;
            }

            if local_start > last_end {
                let before = &span_text[last_end..local_start];
                if !before.is_empty() {
                    result_spans.push(Span::styled(before.to_string(), span.style));
                }
            }

            let highlight_start = local_start.max(last_end);
            let highlighted = &span_text[highlight_start..local_end];
            if !highlighted.is_empty() {
                result_spans.push(Span::styled(highlighted.to_string(), highlight_style));
            }
            last_end = local_end;
        }

        if last_end < span_text.len() {
            result_spans.push(Span::styled(span_text[last_end..].to_string(), span.style));
        }

        global_pos += span_text.len();
    }

    Line::from(result_spans)
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
    use crate::config::CompactionConfig;

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
        let area = ConversationArea::new(entries, false, None, "", 0, 10);
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
        let _ = ConversationArea::new(&mut entries, false, None, "", 0, 10);
    }

    #[test]
    fn entry_lines_with_current_response() {
        let mut entries: Vec<ConversationEntry> = Vec::new();
        let area = ConversationArea::new(&mut entries, false, None, "streaming text", 0, 10);
        assert!(!area.current_response.is_empty());
    }

    #[test]
    fn entry_lines_with_current_thinking_and_response() {
        let mut entries: Vec<ConversationEntry> = Vec::new();
        let area = ConversationArea::new(&mut entries, true, None, "streaming text", 0, 10);
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
                let mut area = ConversationArea::new(&mut entries, true, None, "the answer", 0, 18);
                area.render(frame, rect, 56, None, None, None);
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
                let mut area = ConversationArea::new(&mut entries, true, None, "", 0, 18);
                area.render(frame, rect, 56, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(
                    &mut entries,
                    false,
                    None,
                    "Streaming response...",
                    0,
                    20,
                );
                area_widget.render(frame, rect, 58, None, None, None);
            })
            .expect("draw");

        insta::assert_snapshot!("render_with_current_response", terminal.backend());
    }

    #[test]
    fn max_scroll_with_many_entries() {
        let mut entries: Vec<ConversationEntry> = (0..40)
            .map(|i| ConversationEntry::new(ConversationRole::User, format!("line {i}")))
            .collect();
        let mut area = ConversationArea::new(&mut entries, false, None, "", 0, 10);
        let max = area.max_scroll(58);
        assert!(max > 0, "should have scrollable content");
    }

    #[test]
    fn max_scroll_zero_when_content_fits() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::User,
            "short".to_string(),
        )];
        let mut area = ConversationArea::new(&mut entries, false, None, "", 0, 100);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget =
                    ConversationArea::new(&mut entries, false, None, table, 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
            compaction: CompactionConfig::default(),
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 8);
                area_widget.render(frame, rect, 58, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 20, 10);
                area_widget.render(frame, rect, 58, None, None, None);
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
        let mut area = ConversationArea::new(&mut entries, false, None, "", 0, viewport);
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

        let mut area = ConversationArea::new(&mut entries, false, None, "", 0, viewport);
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

        let mut area = ConversationArea::new(&mut entries, false, None, "", 0, viewport);
        let max = area.max_scroll(text_width);

        let backend = ratatui::backend::TestBackend::new(60, viewport + 2);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, viewport + 2);
                let mut scrolled_area =
                    ConversationArea::new(&mut entries, false, None, "", max, viewport);
                scrolled_area.render(frame, rect, text_width, None, None, None);
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
                let mut area = ConversationArea::new(&mut entries, false, None, "", 0, viewport);
                area.render(frame, rect, text_width, None, None, None);
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

        let mut area = ConversationArea::new(&mut entries, false, None, "", 0, viewport);
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
                    let mut area =
                        ConversationArea::new(&mut entries, false, None, "", scroll, viewport);
                    area.render(frame, rect, text_width, None, None, None);
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
        let mut area = ConversationArea::new(&mut entries, false, None, "", 0, viewport);
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
                let mut area =
                    ConversationArea::new(&mut entries, false, None, &streaming, 0, viewport);
                area.render(frame, rect, text_width, None, None, None);
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
                    None,
                    &streaming,
                    scroll_up_amount,
                    viewport,
                );
                area.render(frame, rect, text_width, None, None, None);
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

        let mut area = ConversationArea::new(&mut entries, false, None, "", 0, viewport);
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
                    let mut area =
                        ConversationArea::new(entries, false, None, "", scroll, viewport);
                    area.render(frame, rect, text_width, None, None, None);
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
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 38, None, None, None);
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

    #[test]
    fn truncate_to_width_short_string() {
        let result = truncate_to_width("hello", 10);
        assert_eq!(result, "hello");
    }

    #[test]
    fn truncate_to_width_long_string() {
        let result = truncate_to_width("the user is asking me to tell a joke", 20);
        assert!(
            result.starts_with("..."),
            "should start with '...': {result}"
        );
        use unicode_width::UnicodeWidthStr;
        let width = UnicodeWidthStr::width(result.as_str());
        assert!(width <= 20, "display width {width} should be <= 20");
        assert!(
            result.contains("joke"),
            "tail should be preserved, got: {result}"
        );
    }

    #[test]
    fn truncate_to_width_multibyte() {
        let result = truncate_to_width("hello 🦀🦀🦀 world", 12);
        assert!(
            std::str::from_utf8(result.as_bytes()).is_ok(),
            "result should be valid UTF-8"
        );
        use unicode_width::UnicodeWidthStr;
        let width = UnicodeWidthStr::width(result.as_str());
        assert!(width <= 12, "display width {width} should be <= 12");
    }

    #[test]
    fn truncate_to_width_zero_max_width() {
        let result = truncate_to_width("hello", 0);
        assert_eq!(result, "");
    }

    #[test]
    fn truncate_to_width_exact_fit() {
        let result = truncate_to_width("hello", 5);
        assert_eq!(result, "hello");
    }

    #[test]
    fn thinking_indicator_lines_with_preview() {
        let lines = thinking_indicator_lines(Some("preview text"), 80);
        assert_eq!(lines.len(), 3, "should have 3 lines with preview");
        let label_line = format!("{:?}", lines[0]);
        assert!(
            label_line.contains("[Thinking]:"),
            "first line should contain [Thinking]: label, got: {label_line}"
        );
        let preview_line = &lines[1];
        let preview_text: String = preview_line
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert!(
            preview_text.starts_with("  "),
            "preview line should start with '  ' indent, got: {preview_text}"
        );
        assert!(
            preview_text.contains("preview text"),
            "preview line should contain the preview text, got: {preview_text}"
        );
        let has_dark_gray = preview_line
            .spans
            .iter()
            .any(|s| s.style.fg == Some(Color::DarkGray));
        assert!(has_dark_gray, "preview line should have DarkGray style");
        assert!(
            lines[2].spans.is_empty() || lines[2].spans.iter().all(|s| s.content.is_empty()),
            "last line should be blank"
        );
    }

    #[test]
    fn thinking_indicator_lines_without_preview() {
        let lines = thinking_indicator_lines(None, 80);
        assert_eq!(lines.len(), 2, "should have 2 lines without preview");
        let label_line = format!("{:?}", lines[0]);
        assert!(
            label_line.contains("[Thinking]:"),
            "first line should contain [Thinking]: label, got: {label_line}"
        );
        assert!(
            lines[1].spans.is_empty() || lines[1].spans.iter().all(|s| s.content.is_empty()),
            "second line should be blank"
        );
    }

    #[test]
    fn thinking_indicator_lines_empty_preview() {
        let lines = thinking_indicator_lines(Some(""), 80);
        assert_eq!(lines.len(), 2, "empty preview should fall back to 2 lines");

        let lines = thinking_indicator_lines(Some("   "), 80);
        assert_eq!(
            lines.len(),
            2,
            "whitespace-only preview should fall back to 2 lines"
        );
    }

    #[test]
    fn render_with_thinking_preview() {
        let mut entries: Vec<ConversationEntry> = Vec::new();
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 58, 18);
                let mut area = ConversationArea::new(
                    &mut entries,
                    true,
                    Some("analyzing the request"),
                    "",
                    0,
                    18,
                );
                area.render(frame, rect, 56, None, None, None);
            })
            .expect("draw");

        insta::assert_snapshot!("render_with_thinking_preview", terminal.backend());
    }

    #[test]
    fn max_scroll_includes_thinking_preview_line() {
        let mut entries: Vec<ConversationEntry> = (0..6)
            .map(|i| ConversationEntry::new(ConversationRole::User, format!("message {i}")))
            .collect();

        let mut area_no_preview = ConversationArea::new(&mut entries, true, None, "", 0, 10);
        let max_no_preview = area_no_preview.max_scroll(58);

        let mut area_with_preview =
            ConversationArea::new(&mut entries, true, Some("test preview"), "", 0, 10);
        let max_with_preview = area_with_preview.max_scroll(58);

        assert_eq!(
            max_with_preview,
            max_no_preview + 1,
            "thinking preview should add exactly 1 to max_scroll"
        );
    }

    #[test]
    fn render_search_highlights_in_conversation() {
        let mut entries = vec![
            ConversationEntry::new(ConversationRole::User, "hello world".to_string()),
            ConversationEntry::new(ConversationRole::Assistant, "Hello there".to_string()),
        ];
        let matches = vec![
            SearchMatch {
                entry_index: 0,
                byte_start: 0,
                byte_end: 5,
            },
            SearchMatch {
                entry_index: 1,
                byte_start: 0,
                byte_end: 5,
            },
        ];
        let entry_counts: Vec<u16> = entries
            .iter_mut()
            .map(|e| e.wrapped_line_count(58))
            .collect();

        let _info = SearchHighlightInfo {
            matches: &matches,
            entries: &entries,
            scroll_row: 0,
            visible_height: 20,
            entry_counts: &entry_counts,
        };

        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, Some((&matches, 0)), None, Some("hello"));
            })
            .expect("draw");

        insta::assert_snapshot!("render_search_highlights", terminal.backend());
    }

    #[test]
    fn render_search_bar_during_input() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::User,
            "test content".to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, Some("search term"), None);
            })
            .expect("draw");

        insta::assert_snapshot!("render_search_bar_input", terminal.backend());
    }

    #[test]
    fn render_search_bar_with_active_results() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::User,
            "hello world".to_string(),
        )];
        let matches = vec![SearchMatch {
            entry_index: 0,
            byte_start: 0,
            byte_end: 5,
        }];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, Some((&matches, 0)), None, Some("hello"));
            })
            .expect("draw");

        insta::assert_snapshot!("render_search_bar_active", terminal.backend());
    }

    #[test]
    fn render_no_search_when_none() {
        let mut entries = vec![ConversationEntry::new(
            ConversationRole::User,
            "hello world".to_string(),
        )];
        let backend = ratatui::backend::TestBackend::new(60, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let rect = ratatui::layout::Rect::new(0, 0, 60, 20);
                let mut area_widget = ConversationArea::new(&mut entries, false, None, "", 0, 20);
                area_widget.render(frame, rect, 58, None, None, None);
            })
            .expect("draw");

        insta::assert_snapshot!("render_no_search", terminal.backend());
    }

    #[test]
    fn highlight_line_basic() {
        use ratatui::style::Color;

        let line = Line::from(vec![Span::raw("hello world")]);
        let result = highlight_line(
            &line,
            "hello",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        );

        let highlighted_count = result
            .spans
            .iter()
            .filter(|s| s.style.bg == Some(Color::Yellow))
            .count();
        assert_eq!(highlighted_count, 1, "should have one highlighted span");
        let highlighted_text: String = result
            .spans
            .iter()
            .filter(|s| s.style.bg == Some(Color::Yellow))
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(highlighted_text, "hello");
    }

    #[test]
    fn highlight_line_case_insensitive() {
        use ratatui::style::Color;

        let line = Line::from(vec![Span::raw("Hello World")]);
        let result = highlight_line(
            &line,
            "hello",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        );

        let highlighted_text: String = result
            .spans
            .iter()
            .filter(|s| s.style.bg == Some(Color::Yellow))
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(highlighted_text, "Hello");
    }

    #[test]
    fn highlight_line_multiple_occurrences() {
        use ratatui::style::Color;

        let line = Line::from(vec![Span::raw("abc abc abc")]);
        let result = highlight_line(
            &line,
            "abc",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        );

        let highlighted_count = result
            .spans
            .iter()
            .filter(|s| s.style.bg == Some(Color::Yellow))
            .count();
        assert_eq!(highlighted_count, 3, "should highlight all occurrences");
    }

    #[test]
    fn highlight_line_no_match() {
        let line = Line::from(vec![Span::raw("hello world")]);
        let result = highlight_line(
            &line,
            "xyz",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        );

        assert_eq!(
            result.spans.len(),
            1,
            "unmatched line should remain unchanged"
        );
        assert_eq!(result.spans[0].content.as_ref(), "hello world");
    }

    #[test]
    fn highlight_line_cross_span_boundary() {
        use ratatui::style::Color;

        // Two spans where the match straddles the boundary: "hel" | "lo world"
        // Searching for "hello" should highlight across both spans
        let line = Line::from(vec![Span::raw("hel"), Span::raw("lo world")]);
        let result = highlight_line(
            &line,
            "hello",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        );

        let highlighted: String = result
            .spans
            .iter()
            .filter(|s| s.style.bg == Some(Color::Yellow))
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(
            highlighted, "hello",
            "match crossing span boundary should be highlighted, got: {highlighted:?}"
        );

        let full: String = result.spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(
            full, "hello world",
            "full text should be preserved, got: {full:?}"
        );
    }

    #[test]
    fn highlight_line_match_within_second_span() {
        use ratatui::style::Color;

        // Role prefix span + content span where match is in the content
        let line = Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::raw("find me here"),
        ]);
        let result = highlight_line(
            &line,
            "find",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        );

        let highlighted: String = result
            .spans
            .iter()
            .filter(|s| s.style.bg == Some(Color::Yellow))
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(
            highlighted, "find",
            "match within second span should be highlighted"
        );
    }

    #[test]
    fn highlight_line_multiple_spans_multiple_matches() {
        use ratatui::style::Color;

        // "abc | def abc | ghi" with match "abc" appearing in first and second spans
        let line = Line::from(vec![Span::raw("abc de"), Span::raw("f abc gh")]);
        let result = highlight_line(
            &line,
            "abc",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        );

        let highlighted_count = result
            .spans
            .iter()
            .filter(|s| s.style.bg == Some(Color::Yellow))
            .count();
        assert_eq!(
            highlighted_count, 2,
            "should highlight both occurrences of abc"
        );
    }

    #[test]
    fn highlight_line_only_highlights_matched_substring() {
        use ratatui::style::Color;

        // Regression: searching for "test" in "You are a test" should only
        // highlight "test", not the entire string
        let line = Line::from(vec![Span::raw("  "), Span::raw("You are a test")]);
        let result = highlight_line(
            &line,
            "test",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        );

        let highlighted_text: String = result
            .spans
            .iter()
            .filter(|s| s.style.bg == Some(Color::Yellow))
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(
            highlighted_text, "test",
            "only 'test' should be highlighted, got: {highlighted_text:?}"
        );

        let full_text: String = result.spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(
            full_text, "  You are a test",
            "full text should be preserved"
        );
    }

    #[test]
    fn highlight_line_with_indent_prefix() {
        use ratatui::style::Color;

        // Content lines have a "  " prefix. Searching for a word in the
        // content should only highlight that word, not the prefix or surrounding text.
        let line = Line::from(vec![
            Span::styled("You:", Style::default().fg(Color::Green)),
            Span::raw("  "),
            Span::raw("some test text"),
        ]);
        let result = highlight_line(
            &line,
            "test",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        );

        let highlighted_text: String = result
            .spans
            .iter()
            .filter(|s| s.style.bg == Some(Color::Yellow))
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(
            highlighted_text, "test",
            "only 'test' should be highlighted in role-prefixed line, got: {highlighted_text:?}"
        );
    }
}
