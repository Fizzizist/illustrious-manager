use ratatui::style::Color;
use ratatui::text::{Line, Span};
use std::sync::LazyLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::Theme;
use syntect::parsing::SyntaxSet;
use syntect_assets::assets::HighlightingAssets;

use super::markdown_theme::monokai_comment;

pub static SYNTAX_SET: LazyLock<SyntaxSet> = LazyLock::new(SyntaxSet::load_defaults_newlines);
pub static MONOKAI_EXTENDED: LazyLock<Theme> = LazyLock::new(|| {
    let assets = HighlightingAssets::from_binary();
    assets.get_theme("Monokai Extended Origin").clone()
});

pub fn syntect_to_ratatui_color(c: syntect::highlighting::Color) -> Color {
    Color::Rgb(c.r, c.g, c.b)
}

pub fn highlight_code_block(lang: &str, content: &str) -> Vec<Line<'static>> {
    let syntax = if lang.is_empty() {
        SYNTAX_SET.find_syntax_plain_text()
    } else {
        SYNTAX_SET
            .find_syntax_by_token(lang)
            .unwrap_or_else(|| SYNTAX_SET.find_syntax_plain_text())
    };

    let mut highlighter = HighlightLines::new(syntax, &MONOKAI_EXTENDED);

    let mut lines: Vec<Line<'static>> = Vec::new();

    if !lang.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("[{lang}]"),
            ratatui::style::Style::new()
                .fg(monokai_comment())
                .add_modifier(ratatui::style::Modifier::ITALIC),
        )));
    }

    for raw_line in content.split('\n') {
        let line_with_newline = format!("{raw_line}\n");
        let ranges = highlighter
            .highlight_line(&line_with_newline, &SYNTAX_SET)
            .unwrap_or_default();

        let spans: Vec<Span<'static>> = ranges
            .into_iter()
            .filter_map(|(style, text)| {
                let content = text.trim_end_matches('\n').to_string();
                if content.is_empty() {
                    return None;
                }
                let fg = syntect_to_ratatui_color(style.foreground);
                Some(Span::styled(content, ratatui::style::Style::new().fg(fg)))
            })
            .collect();

        if spans.is_empty() {
            lines.push(Line::raw(String::new()));
        } else {
            lines.push(Line::from(spans));
        }
    }

    if lines.last().is_some_and(|l| l.spans.is_empty()) {
        lines.pop();
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highlight_rust_code_block() {
        let content = "fn main() {\n    println!(\"hello\");\n}\n";
        let lines = highlight_code_block("rust", content);
        assert!(lines.len() >= 3, "should have at least 3 lines");
        let first_text: String = lines[1].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            first_text.contains("fn"),
            "first code line should contain 'fn': {first_text}"
        );
    }

    #[test]
    fn highlight_plain_text_when_no_language() {
        let content = "just some text\n";
        let lines = highlight_code_block("", content);
        assert!(!lines.is_empty(), "should have at least 1 line");
        assert!(
            lines[0].spans.iter().any(|s| s.content.contains("just")),
            "should contain the text"
        );
    }

    #[test]
    fn unknown_language_falls_back_to_plain_text() {
        let content = "hello world\n";
        let lines = highlight_code_block("nonexistent_lang", content);
        assert!(!lines.is_empty());
        let text: String = lines[1].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("hello"), "should contain the text: {text}");
    }

    #[test]
    fn language_label_rendered() {
        let content = "code\n";
        let lines = highlight_code_block("python", content);
        let label: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            label.contains("[python]"),
            "should have language label: {label}"
        );
    }

    #[test]
    fn no_language_label_when_empty() {
        let content = "code\n";
        let lines = highlight_code_block("", content);
        let first: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            !first.contains('['),
            "should not have language label: {first}"
        );
    }

    #[test]
    fn trailing_newline_does_not_produce_extra_blank_line() {
        let content = "line1\nline2\n";
        let lines = highlight_code_block("", content);
        assert_eq!(lines.len(), 2, "should have exactly 2 lines");
    }

    #[test]
    fn different_tokens_get_different_colors() {
        let content = "fn main() {\n    let x = 42;\n}\n";
        let lines = highlight_code_block("rust", content);
        let code_lines: Vec<_> = lines
            .iter()
            .skip_while(|l| l.spans.iter().any(|s| s.content.starts_with('[')))
            .collect();
        let first_spans = &code_lines[0].spans;
        let colors: Vec<Option<Color>> = first_spans.iter().map(|s| s.style.fg).collect();
        let unique_colors: std::collections::HashSet<Option<Color>> = colors.into_iter().collect();
        assert!(
            unique_colors.len() > 1,
            "syntax-highlighted Rust code should have more than one color"
        );
    }

    #[test]
    fn highlight_empty_content() {
        let lines = highlight_code_block("rust", "");
        assert!(
            lines.len() <= 1,
            "empty content should produce at most 1 line, got {} lines",
            lines.len()
        );
    }

    #[test]
    fn highlight_blank_only_content() {
        let lines = highlight_code_block("rust", "   \n   \n");
        assert!(
            !lines.is_empty(),
            "blank-only content should produce at least 1 line"
        );
    }

    #[test]
    fn highlight_blank_lines_between_code() {
        let content = "line1\n\nline3\n";
        let lines = highlight_code_block("rust", content);
        assert!(
            lines.len() >= 3,
            "should have at least 3 lines (label + 3 code lines), got {}",
            lines.len()
        );
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("");
        assert!(
            text.contains("line1") && text.contains("line3"),
            "should contain line1 and line3, got: {text}"
        );
    }
}
