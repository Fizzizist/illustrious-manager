use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

const STATUS_BG: Color = Color::Rgb(30, 30, 30);

#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// True when counts were estimated from character length rather than
    /// reported directly by the API.
    pub is_estimated: bool,
}

impl TokenUsage {
    pub fn add(&mut self, input: u32, output: u32) {
        self.input_tokens += input as u64;
        self.output_tokens += output as u64;
        self.is_estimated = false;
    }
}

pub struct StatusLineInfo<'a> {
    pub model: &'a str,
    pub git_branch: Option<&'a str>,
    pub working_dir: &'a std::path::Path,
    pub usage: &'a TokenUsage,
}

/// Estimate token counts from a slice of conversation messages.
///
/// The Anthropic tokenizer averages ~4 characters per token for English prose.
/// This is an approximation; the returned counts are flagged as estimated so
/// the UI can display a `~` prefix.
pub fn estimate_usage_from_messages(messages: &[crate::types::Message]) -> (u32, u32) {
    let mut input_chars: usize = 0;
    let mut output_chars: usize = 0;

    for message in messages {
        let chars: usize = message
            .content
            .iter()
            .map(|block| match block {
                crate::types::ContentBlock::Text(t) => t.len(),
                crate::types::ContentBlock::ToolUse { name, input, .. } => {
                    name.len() + input.to_string().len()
                }
                crate::types::ContentBlock::ToolResult { content, .. } => content.len(),
            })
            .sum();

        match message.role {
            crate::types::Role::User => input_chars += chars,
            crate::types::Role::Assistant => output_chars += chars,
        }
    }

    let input_tokens = input_chars.div_ceil(4) as u32;
    let output_tokens = output_chars.div_ceil(4) as u32;
    (input_tokens, output_tokens)
}

fn format_token_count(count: u64) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}k", count as f64 / 1_000.0)
    } else {
        format!("{count}")
    }
}

fn format_token_usage(usage: &TokenUsage) -> String {
    let prefix = if usage.is_estimated { "~" } else { "" };
    format!(
        "{}↑{} ↓{}",
        prefix,
        format_token_count(usage.input_tokens),
        format_token_count(usage.output_tokens)
    )
}

fn compact_path(path: &std::path::Path) -> String {
    if let Some(home) = dirs::home_dir()
        && let Ok(stripped) = path.strip_prefix(&home)
    {
        return format!("~/{}", stripped.display());
    }
    path.display().to_string()
}

pub fn render_status_line(info: &StatusLineInfo<'_>, frame: &mut ratatui::Frame, area: Rect) {
    let line = build_status_line(info, area.width);
    let paragraph = Paragraph::new(line);
    frame.render_widget(paragraph, area);
}

pub fn build_status_line(info: &StatusLineInfo<'_>, width: u16) -> Line<'static> {
    let usage_str = format_token_usage(info.usage);
    let branch_str = info.git_branch.unwrap_or("no git");
    let dir_str = compact_path(info.working_dir);

    let left_spans = vec![
        Span::styled(
            format!(" {usage_str} "),
            Style::default().fg(Color::White).bg(STATUS_BG),
        ),
        Span::styled(
            format!("  {branch_str} "),
            Style::default().fg(Color::Cyan).bg(STATUS_BG),
        ),
    ];

    let right_spans = vec![
        Span::styled(
            format!(" {} ", info.model),
            Style::default().fg(Color::Yellow).bg(STATUS_BG),
        ),
        Span::styled(
            format!(" {dir_str} "),
            Style::default().fg(Color::Magenta).bg(STATUS_BG),
        ),
    ];

    let left_width: usize = left_spans.iter().map(|s| s.width()).sum();
    let right_width: usize = right_spans.iter().map(|s| s.width()).sum();
    let total_width = width as usize;
    let gap = total_width.saturating_sub(left_width + right_width);

    let mut spans = left_spans;
    spans.push(Span::styled(
        " ".repeat(gap),
        Style::default().bg(STATUS_BG),
    ));
    spans.extend(right_spans);

    Line::from(spans)
}

pub fn detect_git_branch() -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let branch = String::from_utf8(output.stdout).ok()?;
    let trimmed = branch.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_info() -> (String, Option<String>, PathBuf, TokenUsage) {
        (
            "claude-sonnet-4-20250514".to_string(),
            Some("main".to_string()),
            PathBuf::from("/home/user/project"),
            TokenUsage::default(),
        )
    }

    fn make_info<'a>(
        model: &'a str,
        git_branch: Option<&'a str>,
        working_dir: &'a std::path::Path,
        usage: &'a TokenUsage,
    ) -> StatusLineInfo<'a> {
        StatusLineInfo {
            model,
            git_branch,
            working_dir,
            usage,
        }
    }

    #[test]
    fn token_usage_add_accumulates() {
        let mut usage = TokenUsage::default();
        usage.add(100, 50);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        usage.add(200, 100);
        assert_eq!(usage.input_tokens, 300);
        assert_eq!(usage.output_tokens, 150);
    }

    #[test]
    fn token_usage_default_is_zero() {
        let usage = TokenUsage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
    }

    #[test]
    fn format_token_count_plain() {
        assert_eq!(format_token_count(500), "500");
    }

    #[test]
    fn format_token_count_thousands() {
        assert_eq!(format_token_count(5_000), "5.0k");
    }

    #[test]
    fn format_token_count_thousands_with_remainder() {
        assert_eq!(format_token_count(15_300), "15.3k");
    }

    #[test]
    fn format_token_count_millions() {
        assert_eq!(format_token_count(1_500_000), "1.5M");
    }

    #[test]
    fn format_token_usage_zero() {
        let usage = TokenUsage::default();
        assert_eq!(format_token_usage(&usage), "↑0 ↓0");
    }

    #[test]
    fn format_token_usage_mixed_scale() {
        let usage = TokenUsage {
            input_tokens: 15_000,
            output_tokens: 500,
            ..Default::default()
        };
        assert_eq!(format_token_usage(&usage), "↑15.0k ↓500");
    }

    #[test]
    fn format_token_usage_large() {
        let usage = TokenUsage {
            input_tokens: 1_200_000,
            output_tokens: 350_000,
            ..Default::default()
        };
        assert_eq!(format_token_usage(&usage), "↑1.2M ↓350.0k");
    }

    #[test]
    fn compact_path_non_home_directory() {
        let path = PathBuf::from("/tmp/some/project");
        assert_eq!(compact_path(&path), "/tmp/some/project");
    }

    #[test]
    fn build_status_line_contains_all_sections() {
        let (model, branch, dir, usage) = test_info();
        let info = make_info(&model, branch.as_deref(), &dir, &usage);
        let line = build_status_line(&info, 120);
        let text: String = line.spans.iter().map(|s| s.content.to_string()).collect();
        assert!(text.contains("main"), "should contain git branch");
        assert!(
            text.contains("claude-sonnet-4-20250514"),
            "should contain model"
        );
        assert!(
            text.contains("/home/user/project"),
            "should contain directory"
        );
        assert!(text.contains('↑'), "should contain input token indicator");
        assert!(text.contains('↓'), "should contain output token indicator");
    }

    #[test]
    fn build_status_line_no_git_branch() {
        let (model, _, dir, usage) = test_info();
        let info = make_info(&model, None, &dir, &usage);
        let line = build_status_line(&info, 120);
        let text: String = line.spans.iter().map(|s| s.content.to_string()).collect();
        assert!(text.contains("no git"), "should show 'no git' fallback");
    }

    #[test]
    fn build_status_line_with_token_usage() {
        let (model, branch, dir, mut usage) = test_info();
        usage.add(5000, 1000);
        let info = make_info(&model, branch.as_deref(), &dir, &usage);
        let line = build_status_line(&info, 120);
        let text: String = line.spans.iter().map(|s| s.content.to_string()).collect();
        assert!(text.contains("↑5.0k"), "should contain input token count");
        assert!(text.contains("↓1.0k"), "should contain output token count");
    }

    #[test]
    fn build_status_line_sections_have_distinct_colors() {
        let (model, branch, dir, usage) = test_info();
        let info = make_info(&model, branch.as_deref(), &dir, &usage);
        let line = build_status_line(&info, 120);
        let colors: Vec<Option<Color>> = line
            .spans
            .iter()
            .filter(|s| !s.content.trim().is_empty())
            .map(|s| s.style.fg)
            .collect();
        let unique: std::collections::HashSet<_> = colors.iter().collect();
        assert!(
            unique.len() >= 3,
            "should have at least 3 distinct colors, got {unique:?}"
        );
    }

    #[test]
    fn build_status_line_all_spans_have_dark_gray_background() {
        let (model, branch, dir, usage) = test_info();
        let info = make_info(&model, branch.as_deref(), &dir, &usage);
        let line = build_status_line(&info, 120);
        for span in &line.spans {
            assert_eq!(
                span.style.bg,
                Some(STATUS_BG),
                "all spans should have DarkGray background, span content: {:?}",
                span.content
            );
        }
    }

    #[test]
    fn render_status_line_snapshot() {
        let model = "claude-sonnet-4-20250514".to_string();
        let branch = Some("main".to_string());
        let dir = PathBuf::from("/home/user/project");
        let usage = TokenUsage {
            input_tokens: 12500,
            output_tokens: 3200,
            ..Default::default()
        };
        let info = make_info(&model, branch.as_deref(), &dir, &usage);

        let backend = ratatui::backend::TestBackend::new(80, 1);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 80, 1);
                render_status_line(&info, frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("render_status_line", terminal.backend());
    }

    #[test]
    fn render_status_line_narrow_terminal() {
        let model = "claude-sonnet-4-20250514".to_string();
        let branch = Some("feature/long-branch-name".to_string());
        let dir = PathBuf::from("/home/user/project");
        let usage = TokenUsage::default();
        let info = make_info(&model, branch.as_deref(), &dir, &usage);

        let backend = ratatui::backend::TestBackend::new(40, 1);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal creation");
        terminal
            .draw(|frame| {
                let area = ratatui::layout::Rect::new(0, 0, 40, 1);
                render_status_line(&info, frame, area);
            })
            .expect("draw");

        insta::assert_snapshot!("render_status_line_narrow", terminal.backend());
    }

    #[test]
    fn detect_git_branch_returns_some_in_git_repo() {
        let branch = detect_git_branch();
        assert!(branch.is_some(), "should detect a git branch");
        let name = branch.expect("branch should exist");
        assert!(!name.is_empty(), "branch name should not be empty");
    }

    #[test]
    fn format_token_usage_estimated_adds_tilde_prefix() {
        let usage = TokenUsage {
            input_tokens: 500,
            output_tokens: 100,
            is_estimated: true,
        };
        let s = format_token_usage(&usage);
        assert!(
            s.starts_with('~'),
            "estimated usage should start with '~', got: {s}"
        );
    }

    #[test]
    fn format_token_usage_exact_has_no_tilde() {
        let usage = TokenUsage {
            input_tokens: 500,
            output_tokens: 100,
            is_estimated: false,
        };
        let s = format_token_usage(&usage);
        assert!(
            !s.starts_with('~'),
            "exact usage should not start with '~', got: {s}"
        );
    }

    #[test]
    fn token_usage_add_clears_estimated_flag() {
        let mut usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            is_estimated: true,
        };
        usage.add(10, 5);
        assert!(!usage.is_estimated, "add() should clear the estimated flag");
        assert_eq!(usage.input_tokens, 110);
        assert_eq!(usage.output_tokens, 55);
    }

    #[test]
    fn estimate_usage_from_empty_messages_returns_zeros() {
        let (input, output) = estimate_usage_from_messages(&[]);
        assert_eq!(input, 0);
        assert_eq!(output, 0);
    }

    #[test]
    fn estimate_usage_splits_by_role() {
        use crate::types::{Message, Role};
        let messages = vec![
            Message::text(Role::User, "a".repeat(400)),
            Message::text(Role::Assistant, "b".repeat(200)),
        ];
        let (input, output) = estimate_usage_from_messages(&messages);
        assert_eq!(input, 100, "400 chars / 4 = 100 input tokens");
        assert_eq!(output, 50, "200 chars / 4 = 50 output tokens");
    }

    #[test]
    fn estimate_usage_counts_tool_use_as_input() {
        use crate::types::{ContentBlock, Message, Role};
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "t1".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"command": "ls"}),
            }],
        }];
        let (input, output) = estimate_usage_from_messages(&messages);
        // tool use on Assistant message counts as output
        assert_eq!(input, 0);
        assert!(output > 0, "tool use chars should contribute to output");
    }

    #[test]
    fn estimate_usage_counts_tool_result_as_input() {
        use crate::types::{ContentBlock, Message, Role};
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".to_string(),
                content: "a".repeat(40),
                is_error: false,
            }],
        }];
        let (input, output) = estimate_usage_from_messages(&messages);
        assert_eq!(input, 10, "40 chars / 4 = 10 input tokens");
        assert_eq!(output, 0);
    }
}
