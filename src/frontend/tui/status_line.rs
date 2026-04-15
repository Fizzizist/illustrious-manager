use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl TokenUsage {
    pub fn add(&mut self, input: u32, output: u32) {
        self.input_tokens += input as u64;
        self.output_tokens += output as u64;
    }
}

pub struct StatusLineInfo {
    pub model: String,
    pub git_branch: Option<String>,
    pub working_dir: PathBuf,
    pub usage: TokenUsage,
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
    format!(
        "↑{} ↓{}",
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

pub fn render_status_line(info: &StatusLineInfo, frame: &mut ratatui::Frame, area: Rect) {
    let line = build_status_line(info, area.width);
    let paragraph = Paragraph::new(line);
    frame.render_widget(paragraph, area);
}

pub fn build_status_line(info: &StatusLineInfo, width: u16) -> Line<'static> {
    let usage_str = format_token_usage(&info.usage);
    let branch_str = info.git_branch.as_deref().unwrap_or("no git").to_string();
    let dir_str = compact_path(&info.working_dir);

    let left_spans = vec![
        Span::styled(
            format!(" {usage_str} "),
            Style::default().fg(Color::White).bg(Color::DarkGray),
        ),
        Span::styled(
            format!("  {branch_str} "),
            Style::default().fg(Color::Cyan).bg(Color::DarkGray),
        ),
    ];

    let right_spans = vec![
        Span::styled(
            format!(" {} ", info.model),
            Style::default().fg(Color::Yellow).bg(Color::DarkGray),
        ),
        Span::styled(
            format!(" {dir_str} "),
            Style::default().fg(Color::Magenta).bg(Color::DarkGray),
        ),
    ];

    let left_width: usize = left_spans.iter().map(|s| s.width()).sum();
    let right_width: usize = right_spans.iter().map(|s| s.width()).sum();
    let total_width = width as usize;
    let gap = total_width.saturating_sub(left_width + right_width);

    let mut spans = left_spans;
    spans.push(Span::styled(
        " ".repeat(gap),
        Style::default().bg(Color::DarkGray),
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

    fn test_info() -> StatusLineInfo {
        StatusLineInfo {
            model: "claude-sonnet-4-20250514".to_string(),
            git_branch: Some("main".to_string()),
            working_dir: PathBuf::from("/home/user/project"),
            usage: TokenUsage::default(),
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
        };
        assert_eq!(format_token_usage(&usage), "↑15.0k ↓500");
    }

    #[test]
    fn format_token_usage_large() {
        let usage = TokenUsage {
            input_tokens: 1_200_000,
            output_tokens: 350_000,
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
        let info = test_info();
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
        let mut info = test_info();
        info.git_branch = None;
        let line = build_status_line(&info, 120);
        let text: String = line.spans.iter().map(|s| s.content.to_string()).collect();
        assert!(text.contains("no git"), "should show 'no git' fallback");
    }

    #[test]
    fn build_status_line_with_token_usage() {
        let mut info = test_info();
        info.usage.add(5000, 1000);
        let line = build_status_line(&info, 120);
        let text: String = line.spans.iter().map(|s| s.content.to_string()).collect();
        assert!(text.contains("↑5.0k"), "should contain input token count");
        assert!(text.contains("↓1.0k"), "should contain output token count");
    }

    #[test]
    fn build_status_line_sections_have_distinct_colors() {
        let info = test_info();
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
        let info = test_info();
        let line = build_status_line(&info, 120);
        for span in &line.spans {
            assert_eq!(
                span.style.bg,
                Some(Color::DarkGray),
                "all spans should have DarkGray background, span content: {:?}",
                span.content
            );
        }
    }

    #[test]
    fn render_status_line_snapshot() {
        let info = StatusLineInfo {
            model: "claude-sonnet-4-20250514".to_string(),
            git_branch: Some("main".to_string()),
            working_dir: PathBuf::from("/home/user/project"),
            usage: TokenUsage {
                input_tokens: 12500,
                output_tokens: 3200,
            },
        };

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
        let info = StatusLineInfo {
            model: "claude-sonnet-4-20250514".to_string(),
            git_branch: Some("feature/long-branch-name".to_string()),
            working_dir: PathBuf::from("/home/user/project"),
            usage: TokenUsage::default(),
        };

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
}
