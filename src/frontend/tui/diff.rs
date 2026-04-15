use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use similar::{ChangeTag, TextDiff};
use std::sync::LazyLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;

static SYNTAX_SET: LazyLock<SyntaxSet> = LazyLock::new(SyntaxSet::load_defaults_newlines);
static THEME_SET: LazyLock<ThemeSet> = LazyLock::new(ThemeSet::load_defaults);

const GUTTER_WIDTH: usize = 6;
const REMOVED_BG: Color = Color::Rgb(80, 20, 20);
const ADDED_BG: Color = Color::Rgb(20, 60, 20);
const CHANGED_WORD_REMOVED_BG: Color = Color::Rgb(140, 40, 40);
const CHANGED_WORD_ADDED_BG: Color = Color::Rgb(40, 120, 40);
const HEADER_FG: Color = Color::Cyan;

/// The kind of a line in a diff hunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffLineKind {
    Header,
    Context,
    Added,
    Removed,
}

/// A single rendered line in a diff view.
#[derive(Debug, Clone)]
pub struct DiffLine {
    pub kind: DiffLineKind,
    pub content: String,
    pub old_lineno: Option<usize>,
    pub new_lineno: Option<usize>,
}

/// A group of contiguous diff lines with context around changes.
#[derive(Debug, Clone)]
pub struct DiffHunk {
    pub lines: Vec<DiffLine>,
}

/// A complete file diff with path, stats, and hunks.
#[derive(Debug, Clone)]
pub struct FileDiff {
    pub path: String,
    pub added: usize,
    pub removed: usize,
    pub is_new_file: bool,
    pub hunks: Vec<DiffHunk>,
}

/// Build a `FileDiff` from old and new text snapshots.
pub fn build_file_diff_from_snapshots(path: &str, before: &str, after: &str) -> FileDiff {
    let is_new_file = before.is_empty();
    let diff = TextDiff::from_lines(before, after);
    let mut hunks = Vec::new();
    let mut added = 0usize;
    let mut removed = 0usize;

    for group in diff.grouped_ops(3) {
        let mut lines = Vec::new();

        // Compute hunk header ranges
        let first = group.first().expect("group must be non-empty");
        let old_start = first.old_range().start + 1;
        let new_start = first.new_range().start + 1;
        let old_len: usize = group.iter().map(|op| op.old_range().len()).sum();
        let new_len: usize = group.iter().map(|op| op.new_range().len()).sum();

        let header = format!(
            "@@ -{},{} +{},{} @@",
            old_start, old_len, new_start, new_len
        );
        lines.push(DiffLine {
            kind: DiffLineKind::Header,
            content: header,
            old_lineno: None,
            new_lineno: None,
        });

        let mut old_lineno = old_start;
        let mut new_lineno = new_start;

        for op in &group {
            for change in diff.iter_changes(op) {
                match change.tag() {
                    ChangeTag::Equal => {
                        lines.push(DiffLine {
                            kind: DiffLineKind::Context,
                            content: change.value().trim_end_matches('\n').to_string(),
                            old_lineno: Some(old_lineno),
                            new_lineno: Some(new_lineno),
                        });
                        old_lineno += 1;
                        new_lineno += 1;
                    }
                    ChangeTag::Delete => {
                        lines.push(DiffLine {
                            kind: DiffLineKind::Removed,
                            content: change.value().trim_end_matches('\n').to_string(),
                            old_lineno: Some(old_lineno),
                            new_lineno: None,
                        });
                        old_lineno += 1;
                        removed += 1;
                    }
                    ChangeTag::Insert => {
                        lines.push(DiffLine {
                            kind: DiffLineKind::Added,
                            content: change.value().trim_end_matches('\n').to_string(),
                            old_lineno: None,
                            new_lineno: Some(new_lineno),
                        });
                        new_lineno += 1;
                        added += 1;
                    }
                }
            }
        }

        hunks.push(DiffHunk { lines });
    }

    FileDiff {
        path: path.to_string(),
        added,
        removed,
        is_new_file,
        hunks,
    }
}

/// Compute word-level inline diff spans for a removed/added line pair.
/// Returns `(removed_spans, added_spans)`.
pub fn build_inline_word_diff(old: &str, new: &str) -> (Vec<Span<'static>>, Vec<Span<'static>>) {
    let diff = TextDiff::from_words(old, new);
    let mut removed_spans = Vec::new();
    let mut added_spans = Vec::new();

    for change in diff.iter_all_changes() {
        let word = change.value().to_string();
        match change.tag() {
            ChangeTag::Equal => {
                removed_spans.push(Span::styled(
                    word.clone(),
                    Style::default().fg(Color::DarkGray),
                ));
                added_spans.push(Span::styled(word, Style::default().fg(Color::DarkGray)));
            }
            ChangeTag::Delete => {
                removed_spans.push(Span::styled(
                    word,
                    Style::default()
                        .fg(Color::White)
                        .bg(CHANGED_WORD_REMOVED_BG)
                        .add_modifier(Modifier::BOLD),
                ));
            }
            ChangeTag::Insert => {
                added_spans.push(Span::styled(
                    word,
                    Style::default()
                        .fg(Color::White)
                        .bg(CHANGED_WORD_ADDED_BG)
                        .add_modifier(Modifier::BOLD),
                ));
            }
        }
    }

    (removed_spans, added_spans)
}

fn syntect_color_to_ratatui(color: syntect::highlighting::Color) -> Color {
    Color::Rgb(color.r, color.g, color.b)
}

/// Apply syntect syntax highlighting to a single line of code.
/// Falls back to plain style for unrecognized file extensions.
pub fn highlight_code_line(line: &str, path: &str, base_style: Style) -> Vec<Span<'static>> {
    let syntax = SYNTAX_SET
        .find_syntax_for_file(path)
        .ok()
        .flatten()
        .unwrap_or_else(|| SYNTAX_SET.find_syntax_plain_text());

    let theme = THEME_SET
        .themes
        .get("base16-ocean.dark")
        .or_else(|| THEME_SET.themes.values().next())
        .expect("at least one theme must be available in the bundled theme set");

    let mut highlighter = HighlightLines::new(syntax, theme);
    let ranges = highlighter
        .highlight_line(line, &SYNTAX_SET)
        .unwrap_or_default();

    if ranges.is_empty() {
        return vec![Span::styled(line.to_string(), base_style)];
    }

    ranges
        .into_iter()
        .map(|(style, text)| {
            let fg = syntect_color_to_ratatui(style.foreground);
            let mut ratatui_style = base_style.fg(fg);
            if let Some(bg) = base_style.bg {
                ratatui_style = ratatui_style.bg(bg);
            }
            Span::styled(text.to_string(), ratatui_style)
        })
        .collect()
}

fn gutter(old: Option<usize>, new: Option<usize>, marker: char) -> String {
    let old_s = old
        .map(|n| format!("{:>3}", n))
        .unwrap_or_else(|| "   ".to_string());
    let new_s = new
        .map(|n| format!("{:>3}", n))
        .unwrap_or_else(|| "   ".to_string());
    format!("{}{} {}", old_s, new_s, marker)
}

/// Render a `FileDiff` into ratatui `Line`s suitable for display.
///
/// Adjacent Removed/Added pairs get word-level inline diff highlights.
/// All code lines get syntax highlighting based on the file path.
pub fn build_diff_lines(file: &FileDiff, width: u16) -> Vec<Line<'static>> {
    let max_content_width = (width as usize).saturating_sub(GUTTER_WIDTH + 1);
    let mut lines: Vec<Line<'static>> = Vec::new();

    // File header
    let header_style = Style::default().fg(HEADER_FG).add_modifier(Modifier::BOLD);
    let verb = if file.is_new_file {
        "new file"
    } else {
        "modified"
    };
    lines.push(Line::from(Span::styled(
        format!(
            "--- {} ({}, +{} -{})",
            file.path, verb, file.added, file.removed
        ),
        header_style,
    )));

    for hunk in &file.hunks {
        let hunk_lines = &hunk.lines;
        let mut i = 0;

        while i < hunk_lines.len() {
            let dl = &hunk_lines[i];

            match dl.kind {
                DiffLineKind::Header => {
                    lines.push(Line::from(Span::styled(
                        format!(
                            "{:>width$}",
                            &dl.content,
                            width = GUTTER_WIDTH + 1 + dl.content.len().min(max_content_width)
                        ),
                        Style::default().fg(HEADER_FG),
                    )));
                    i += 1;
                }
                DiffLineKind::Context => {
                    let gutter_str = gutter(dl.old_lineno, dl.new_lineno, ' ');
                    let base_style = Style::default().fg(Color::Gray);
                    let mut spans = vec![Span::styled(
                        gutter_str,
                        Style::default().fg(Color::DarkGray),
                    )];
                    let content = truncate_str(&dl.content, max_content_width);
                    spans.extend(highlight_code_line(&content, &file.path, base_style));
                    lines.push(Line::from(spans));
                    i += 1;
                }
                DiffLineKind::Removed => {
                    // Peek ahead: if next line is Added, do inline word diff
                    if i + 1 < hunk_lines.len() && hunk_lines[i + 1].kind == DiffLineKind::Added {
                        let added = &hunk_lines[i + 1];
                        let (removed_word_spans, added_word_spans) =
                            build_inline_word_diff(&dl.content, &added.content);

                        // Removed line
                        let rm_gutter = gutter(dl.old_lineno, None, '-');
                        let mut rm_spans =
                            vec![Span::styled(rm_gutter, Style::default().fg(Color::Red))];
                        for span in removed_word_spans {
                            // Ensure background is set when span has no per-word highlight
                            let s = if span.style.bg.is_none() {
                                Span::styled(span.content, span.style.bg(REMOVED_BG))
                            } else {
                                span
                            };
                            rm_spans.push(s);
                        }
                        lines.push(Line::from(rm_spans));

                        // Added line
                        let add_gutter = gutter(None, added.new_lineno, '+');
                        let mut add_spans =
                            vec![Span::styled(add_gutter, Style::default().fg(Color::Green))];
                        for span in added_word_spans {
                            let s = if span.style.bg.is_none() {
                                Span::styled(span.content, span.style.bg(ADDED_BG))
                            } else {
                                span
                            };
                            add_spans.push(s);
                        }
                        lines.push(Line::from(add_spans));

                        i += 2;
                    } else {
                        // Plain removed line with syntax highlighting
                        let rm_gutter = gutter(dl.old_lineno, None, '-');
                        let base_style = Style::default().fg(Color::Red).bg(REMOVED_BG);
                        let mut spans =
                            vec![Span::styled(rm_gutter, Style::default().fg(Color::Red))];
                        let content = truncate_str(&dl.content, max_content_width);
                        spans.extend(highlight_code_line(&content, &file.path, base_style));
                        lines.push(Line::from(spans));
                        i += 1;
                    }
                }
                DiffLineKind::Added => {
                    // Plain added line (only reached when not part of a Removed/Added pair)
                    let add_gutter = gutter(None, dl.new_lineno, '+');
                    let base_style = Style::default().fg(Color::Green).bg(ADDED_BG);
                    let mut spans =
                        vec![Span::styled(add_gutter, Style::default().fg(Color::Green))];
                    let content = truncate_str(&dl.content, max_content_width);
                    spans.extend(highlight_code_line(&content, &file.path, base_style));
                    lines.push(Line::from(spans));
                    i += 1;
                }
            }
        }
    }

    lines
}

fn truncate_str(s: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut chars = s.chars();
    let head: String = (&mut chars).take(max_chars).collect();
    if chars.next().is_some() {
        format!("{}…", head)
    } else {
        head
    }
}

/// Detect whether a string looks like a unified diff produced by `edit_file`.
/// We look for the sentinel prefix we embed in edit_file output.
pub fn is_diff_output(content: &str) -> bool {
    content.starts_with("DIFF:")
}

/// Parse a diff payload embedded in tool result content.
///
/// Format: `"DIFF:<path>\n<before>\n---BEFORE/AFTER---\n<after>"`
pub fn parse_diff_payload(content: &str) -> Option<FileDiff> {
    let rest = content.strip_prefix("DIFF:")?;
    let newline = rest.find('\n')?;
    let path = &rest[..newline];
    let body = &rest[newline + 1..];
    let sep = "\n---BEFORE/AFTER---\n";
    let sep_pos = body.find(sep)?;
    let before = &body[..sep_pos];
    let after = &body[sep_pos + sep.len()..];
    Some(build_file_diff_from_snapshots(path, before, after))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_inline_word_diff_equal_content_produces_no_highlights() {
        let (removed, added) = build_inline_word_diff("hello world", "hello world");
        assert!(!removed.is_empty());
        assert!(!added.is_empty());
        for span in &removed {
            assert!(
                span.style.bg.is_none(),
                "equal words should have no background highlight"
            );
        }
        for span in &added {
            assert!(
                span.style.bg.is_none(),
                "equal words should have no background highlight"
            );
        }
    }

    #[test]
    fn build_inline_word_diff_single_word_change_highlights_only_changed() {
        let (removed, added) = build_inline_word_diff("hello world", "hello rust");
        // "hello " is equal — no highlight; "world" is removed — highlighted; "rust" is added — highlighted
        let has_highlighted_removed = removed.iter().any(|s| s.style.bg.is_some());
        let has_highlighted_added = added.iter().any(|s| s.style.bg.is_some());
        assert!(
            has_highlighted_removed,
            "removed changed word should be highlighted"
        );
        assert!(
            has_highlighted_added,
            "added changed word should be highlighted"
        );

        // "hello" should appear without highlight in both
        let equal_spans_removed: Vec<_> = removed.iter().filter(|s| s.style.bg.is_none()).collect();
        assert!(
            !equal_spans_removed.is_empty(),
            "equal part should have no background"
        );
    }

    #[test]
    fn build_inline_word_diff_multi_word_change() {
        let (removed, added) = build_inline_word_diff("foo bar baz", "foo qux quux");
        let removed_highlighted: Vec<_> = removed.iter().filter(|s| s.style.bg.is_some()).collect();
        let added_highlighted: Vec<_> = added.iter().filter(|s| s.style.bg.is_some()).collect();
        assert!(!removed_highlighted.is_empty());
        assert!(!added_highlighted.is_empty());
    }

    #[test]
    fn build_file_diff_from_snapshots_new_file() {
        let diff = build_file_diff_from_snapshots("src/main.rs", "", "fn main() {}\n");
        assert!(diff.is_new_file);
        assert!(diff.added > 0);
        assert_eq!(diff.removed, 0);
        assert_eq!(diff.path, "src/main.rs");
        assert!(!diff.hunks.is_empty());
    }

    #[test]
    fn build_file_diff_from_snapshots_modified_file() {
        let before = "fn main() {\n    println!(\"hello\");\n}\n";
        let after = "fn main() {\n    println!(\"world\");\n}\n";
        let diff = build_file_diff_from_snapshots("src/main.rs", before, after);
        assert!(!diff.is_new_file);
        assert!(diff.added > 0);
        assert!(diff.removed > 0);
        assert!(!diff.hunks.is_empty());

        // There should be a removed line containing "hello" and an added line with "world"
        let all_lines: Vec<_> = diff.hunks.iter().flat_map(|h| h.lines.iter()).collect();
        let has_removed_hello = all_lines
            .iter()
            .any(|l| l.kind == DiffLineKind::Removed && l.content.contains("hello"));
        let has_added_world = all_lines
            .iter()
            .any(|l| l.kind == DiffLineKind::Added && l.content.contains("world"));
        assert!(has_removed_hello);
        assert!(has_added_world);
    }

    #[test]
    fn build_file_diff_from_snapshots_no_changes() {
        let text = "fn main() {}\n";
        let diff = build_file_diff_from_snapshots("src/main.rs", text, text);
        assert!(diff.hunks.is_empty(), "no hunks when files are identical");
        assert_eq!(diff.added, 0);
        assert_eq!(diff.removed, 0);
    }

    #[test]
    fn is_diff_output_detects_sentinel() {
        assert!(is_diff_output(
            "DIFF:src/main.rs\nold\n---BEFORE/AFTER---\nnew"
        ));
        assert!(!is_diff_output("Successfully replaced string in"));
        assert!(!is_diff_output(""));
        assert!(!is_diff_output("Wrote 42 bytes to"));
    }

    #[test]
    fn parse_diff_payload_roundtrip() {
        let before = "fn old() {}\n";
        let after = "fn new() {}\n";
        let payload = format!("DIFF:src/lib.rs\n{before}---BEFORE/AFTER---\n{after}");
        let diff = parse_diff_payload(&payload).expect("should parse");
        assert_eq!(diff.path, "src/lib.rs");
        assert!(diff.added > 0);
        assert!(diff.removed > 0);
    }

    #[test]
    fn parse_diff_payload_returns_none_for_invalid_input() {
        assert!(parse_diff_payload("not a diff").is_none());
        assert!(parse_diff_payload("DIFF:").is_none());
        assert!(parse_diff_payload("DIFF:path\nno separator").is_none());
    }

    #[test]
    fn build_diff_lines_produces_ratatui_lines() {
        let before = "fn main() {\n    println!(\"hello\");\n}\n";
        let after = "fn main() {\n    println!(\"world\");\n}\n";
        let diff = build_file_diff_from_snapshots("src/main.rs", before, after);
        let lines = build_diff_lines(&diff, 80);
        assert!(!lines.is_empty(), "should produce at least one line");

        // The first line should be the file header
        let first_line_content: String =
            lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            first_line_content.contains("src/main.rs"),
            "first line should be file header: {first_line_content}"
        );
    }

    #[test]
    fn build_diff_lines_new_file_shows_new_file_label() {
        let diff = build_file_diff_from_snapshots("new.rs", "", "fn foo() {}\n");
        let lines = build_diff_lines(&diff, 80);
        let header: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(header.contains("new file"), "header: {header}");
    }

    #[test]
    fn truncate_str_truncates_long_strings() {
        let s = "a".repeat(200);
        let result = truncate_str(&s, 10);
        assert!(
            result.len() <= 15,
            "should be truncated: len={}",
            result.len()
        );
        assert!(result.ends_with('…'), "should end with ellipsis");
    }

    #[test]
    fn truncate_str_leaves_short_strings_unchanged() {
        let s = "short";
        assert_eq!(truncate_str(s, 80), "short");
    }

    #[test]
    fn truncate_str_zero_width_returns_empty() {
        assert_eq!(truncate_str("anything", 0), "");
    }

    #[test]
    fn build_diff_lines_snapshot() {
        let before = "fn greet(name: &str) {\n    println!(\"Hello, {}!\", name);\n}\n";
        let after = "fn greet(name: &str) {\n    println!(\"Hi, {}!\", name);\n}\n";
        let diff = build_file_diff_from_snapshots("src/greet.rs", before, after);
        let lines = build_diff_lines(&diff, 80);

        // Render lines to plain text for snapshot
        let rendered: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();

        insta::assert_snapshot!("diff_lines_single_word_change", rendered.join("\n"));
    }

    #[test]
    fn build_diff_lines_new_file_snapshot() {
        let diff = build_file_diff_from_snapshots(
            "src/new.rs",
            "",
            "pub fn hello() -> &'static str {\n    \"hello\"\n}\n",
        );
        let lines = build_diff_lines(&diff, 80);
        let rendered: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        insta::assert_snapshot!("diff_lines_new_file", rendered.join("\n"));
    }
}
