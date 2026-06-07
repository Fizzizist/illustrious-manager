use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use similar::{ChangeTag, TextDiff};
use std::path::Path;
use syntect::easy::HighlightLines;

use super::syntect_highlight::{MONOKAI_EXTENDED, SYNTAX_SET};

/// Minimum digits reserved for each line-number column in the gutter.
const MIN_LINE_NO_DIGITS: usize = 1;

/// Kind of a line in a diff.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffLineKind {
    Context,
    Added,
    Removed,
    Header,
}

/// A single line in a rendered diff, with metadata.
#[derive(Debug, Clone)]
pub struct DiffLine {
    pub kind: DiffLineKind,
    pub content: String,
    pub old_line_no: Option<usize>,
    pub new_line_no: Option<usize>,
}

/// A group of contiguous diff lines (hunk).
#[derive(Debug, Clone)]
pub struct DiffHunk {
    pub old_start: usize,
    pub new_start: usize,
    pub lines: Vec<DiffLine>,
}

/// The full computed diff for one file.
#[derive(Debug, Clone)]
pub struct FileDiff {
    pub path: String,
    pub added: usize,
    pub removed: usize,
    pub is_new_file: bool,
    pub hunks: Vec<DiffHunk>,
}

/// Build a `FileDiff` from old and new file content, using line-level diffing.
///
/// Context lines shown around each hunk: 3 lines.
pub fn build_file_diff_from_snapshots(path: &str, before: &str, after: &str) -> FileDiff {
    let is_new_file = before.is_empty();
    let diff = TextDiff::from_lines(before, after);

    let mut added = 0usize;
    let mut removed = 0usize;

    let mut hunks: Vec<DiffHunk> = Vec::new();

    for group in diff.grouped_ops(3) {
        let first = match group.first() {
            Some(op) => op,
            None => continue,
        };
        let last = match group.last() {
            Some(op) => op,
            None => continue,
        };

        let old_start = first.old_range().start;
        let new_start = first.new_range().start;

        let header_content = format!(
            "@@ -{},{} +{},{} @@",
            old_start + 1,
            last.old_range().end - old_start,
            new_start + 1,
            last.new_range().end - new_start,
        );

        let mut hunk_lines = vec![DiffLine {
            kind: DiffLineKind::Header,
            content: header_content,
            old_line_no: None,
            new_line_no: None,
        }];

        for op in &group {
            for change in diff.iter_changes(op) {
                let content = change.value().trim_end_matches('\n').to_string();
                let (kind, old_no, new_no) = match change.tag() {
                    ChangeTag::Equal => (
                        DiffLineKind::Context,
                        Some(change.old_index().map(|i| i + 1).unwrap_or(0)),
                        Some(change.new_index().map(|i| i + 1).unwrap_or(0)),
                    ),
                    ChangeTag::Delete => {
                        removed += 1;
                        (
                            DiffLineKind::Removed,
                            Some(change.old_index().map(|i| i + 1).unwrap_or(0)),
                            None,
                        )
                    }
                    ChangeTag::Insert => {
                        added += 1;
                        (
                            DiffLineKind::Added,
                            None,
                            Some(change.new_index().map(|i| i + 1).unwrap_or(0)),
                        )
                    }
                };
                hunk_lines.push(DiffLine {
                    kind,
                    content,
                    old_line_no: old_no,
                    new_line_no: new_no,
                });
            }
        }

        hunks.push(DiffHunk {
            old_start,
            new_start,
            lines: hunk_lines,
        });
    }

    FileDiff {
        path: path.to_string(),
        added,
        removed,
        is_new_file,
        hunks,
    }
}

fn default_theme() -> &'static syntect::highlighting::Theme {
    &MONOKAI_EXTENDED
}

/// Convert a syntect `Color` to a ratatui `Color`.
fn syntect_to_ratatui_color(c: syntect::highlighting::Color) -> Color {
    Color::Rgb(c.r, c.g, c.b)
}

/// Highlight a single line of code using syntect, returning ratatui `Span`s with the
/// base_style's background preserved and foreground colours from the theme.
///
/// Falls back to a single unstyled span if the extension is unknown.
///
/// Note: a fresh `HighlightLines` is created per call, which is correct for
/// single-line use (e.g. diff context lines) but loses multi-line state.
/// `render_write_file` maintains its own `HighlightLines` across all lines
/// for correct stateful highlighting of a complete file.
fn highlight_code_line(line: &str, path: &str, base_style: Style) -> Vec<Span<'static>> {
    let syntax = SYNTAX_SET
        .find_syntax_for_file(Path::new(path))
        .ok()
        .flatten()
        .unwrap_or_else(|| SYNTAX_SET.find_syntax_plain_text());

    let mut highlighter = HighlightLines::new(syntax, default_theme());

    let line_with_newline = format!("{line}\n");
    let ranges = highlighter
        .highlight_line(&line_with_newline, &SYNTAX_SET)
        .unwrap_or_default();

    if ranges.is_empty() {
        return vec![Span::styled(line.to_string(), base_style)];
    }

    ranges
        .into_iter()
        .filter_map(|(style, text)| {
            let content = text.trim_end_matches('\n').to_string();
            if content.is_empty() {
                return None;
            }
            let fg = syntect_to_ratatui_color(style.foreground);
            let span_style = base_style.fg(fg);
            Some(Span::styled(content, span_style))
        })
        .collect()
}

/// Compute word-level inline diff spans for a pair of removed/added lines.
///
/// Returns `(removed_spans, added_spans)` where changed words have a coloured
/// background to draw the reader's eye to the exact modification.
pub fn build_inline_diff_spans(
    old_line: &str,
    new_line: &str,
) -> (Vec<Span<'static>>, Vec<Span<'static>>) {
    let diff = TextDiff::from_words(old_line, new_line);

    let removed_bg = Color::Rgb(100, 20, 20);
    let added_bg = Color::Rgb(20, 80, 20);
    let removed_fg = Color::Rgb(255, 140, 140);
    let added_fg = Color::Rgb(140, 255, 140);

    let mut old_spans: Vec<Span<'static>> = Vec::new();
    let mut new_spans: Vec<Span<'static>> = Vec::new();

    for change in diff.iter_all_changes() {
        let word = change.value().to_string();
        match change.tag() {
            ChangeTag::Equal => {
                let style = Style::default().fg(Color::Gray);
                old_spans.push(Span::styled(word.clone(), style));
                new_spans.push(Span::styled(word, style));
            }
            ChangeTag::Delete => {
                let style = Style::default()
                    .fg(removed_fg)
                    .bg(removed_bg)
                    .add_modifier(Modifier::BOLD);
                old_spans.push(Span::styled(word, style));
            }
            ChangeTag::Insert => {
                let style = Style::default()
                    .fg(added_fg)
                    .bg(added_bg)
                    .add_modifier(Modifier::BOLD);
                new_spans.push(Span::styled(word, style));
            }
        }
    }

    (old_spans, new_spans)
}

/// Render a `FileDiff` into ratatui `Line`s suitable for display.
///
/// `width` is the available terminal width; content is truncated to fit.
pub fn build_diff_lines(file: &FileDiff, width: usize) -> Vec<Line<'static>> {
    // Compute the number of digits needed for line numbers from the max line
    // number present in this diff, so the gutter is as narrow as possible.
    let max_line_no = file
        .hunks
        .iter()
        .flat_map(|h| &h.lines)
        .flat_map(|dl| [dl.old_line_no, dl.new_line_no])
        .flatten()
        .max()
        .unwrap_or(1);
    let digits = max_line_no.to_string().len().max(MIN_LINE_NO_DIGITS);
    // gutter = <digits> + space + <digits> + trailing space
    let gutter_width = digits + 1 + digits + 1;

    let content_width = width.saturating_sub(gutter_width + 2);

    let header_style = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let added_style = Style::default().fg(Color::Green);
    let removed_style = Style::default().fg(Color::Red);
    let context_style = Style::default().fg(Color::DarkGray);

    let file_label = if file.is_new_file {
        format!("New file: {}", file.path)
    } else {
        format!("Edit: {} (+{} -{})", file.path, file.added, file.removed)
    };

    let mut lines: Vec<Line<'static>> = vec![Line::from(Span::styled(
        file_label,
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    ))];

    for hunk in &file.hunks {
        // Collect runs of consecutive Removed/Added lines so we can pair them
        // for inline word-level diffs.
        let hunk_lines = &hunk.lines;
        let mut i = 0;
        while i < hunk_lines.len() {
            let dl = &hunk_lines[i];
            match dl.kind {
                DiffLineKind::Header => {
                    let truncated = truncate_str(&dl.content, width);
                    lines.push(Line::from(Span::styled(truncated, header_style)));
                    i += 1;
                }
                DiffLineKind::Context => {
                    let gutter = format_gutter(dl.old_line_no, dl.new_line_no, digits);
                    let mut spans = vec![Span::styled(gutter, context_style)];
                    let code_spans = highlight_code_line(&dl.content, &file.path, context_style);
                    spans.extend(code_spans);
                    truncate_spans(&mut spans, content_width);
                    lines.push(Line::from(spans));
                    i += 1;
                }
                DiffLineKind::Removed => {
                    // Look ahead: is the very next line an insertion?
                    // This pairs single adjacent Removed→Added lines for word-level
                    // highlighting, which covers the most common case (single-line edits).
                    // For multi-line replacements (N removed + N added), `similar`
                    // emits all deletions before all insertions, so only the first
                    // removed/added pair is matched here; the rest fall through to
                    // standalone rendering. A more sophisticated approach would
                    // collect all consecutive Removed lines, then all consecutive Added
                    // lines, and pair them positionally.
                    let next_is_added = hunk_lines
                        .get(i + 1)
                        .is_some_and(|n| n.kind == DiffLineKind::Added);

                    if next_is_added {
                        let added_dl = &hunk_lines[i + 1];
                        let (old_word_spans, new_word_spans) =
                            build_inline_diff_spans(&dl.content, &added_dl.content);

                        // Removed line with inline word highlights
                        let gutter = format_gutter(dl.old_line_no, None, digits);
                        let mut removed_spans = vec![
                            Span::styled(gutter, removed_style),
                            Span::styled("-", removed_style),
                        ];
                        removed_spans.extend(old_word_spans);
                        truncate_spans(&mut removed_spans, content_width);
                        lines.push(Line::from(removed_spans));

                        // Added line with inline word highlights
                        let gutter = format_gutter(None, added_dl.new_line_no, digits);
                        let mut added_spans = vec![
                            Span::styled(gutter, added_style),
                            Span::styled("+", added_style),
                        ];
                        added_spans.extend(new_word_spans);
                        truncate_spans(&mut added_spans, content_width);
                        lines.push(Line::from(added_spans));

                        i += 2;
                    } else {
                        // Standalone removed line — syntax-highlighted with red base
                        let gutter = format_gutter(dl.old_line_no, None, digits);
                        let mut spans = vec![
                            Span::styled(gutter, removed_style),
                            Span::styled("-", removed_style),
                        ];
                        let code_spans =
                            highlight_code_line(&dl.content, &file.path, removed_style);
                        spans.extend(code_spans);
                        truncate_spans(&mut spans, content_width);
                        lines.push(Line::from(spans));
                        i += 1;
                    }
                }
                DiffLineKind::Added => {
                    let gutter = format_gutter(None, dl.new_line_no, digits);
                    let mut spans = vec![
                        Span::styled(gutter, added_style),
                        Span::styled("+", added_style),
                    ];
                    let code_spans = highlight_code_line(&dl.content, &file.path, added_style);
                    spans.extend(code_spans);
                    truncate_spans(&mut spans, content_width);
                    lines.push(Line::from(spans));
                    i += 1;
                }
            }
        }
    }

    lines
}

fn format_gutter(old: Option<usize>, new: Option<usize>, digits: usize) -> String {
    match (old, new) {
        (Some(o), Some(n)) => format!("{o:>digits$} {n:>digits$} "),
        (Some(o), None) => format!("{o:>digits$} {:>digits$} ", ""),
        (None, Some(n)) => format!("{:>digits$} {n:>digits$} ", ""),
        (None, None) => " ".repeat(digits + 1 + digits + 1),
    }
}

fn truncate_str(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let head: String = (&mut chars).take(max_chars).collect();
    if chars.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}

fn truncate_spans(spans: &mut Vec<Span<'static>>, max_content_chars: usize) {
    if max_content_chars == 0 {
        return;
    }
    let mut total = 0usize;
    let mut cut_at = spans.len();
    for (idx, span) in spans.iter().enumerate() {
        let span_len = span.content.chars().count();
        if total + span_len > max_content_chars {
            cut_at = idx;
            break;
        }
        total += span_len;
    }
    spans.truncate(cut_at);
}

/// Render an `edit_file` tool call (old_string → new_string) as diff `Line`s.
pub fn render_edit_file_diff(
    input: &serde_json::Value,
    width: usize,
) -> Option<Vec<Line<'static>>> {
    let path = input.get("path").and_then(|v| v.as_str())?;
    let old = input.get("old_string").and_then(|v| v.as_str())?;
    let new = input.get("new_string").and_then(|v| v.as_str())?;
    let file_diff = build_file_diff_from_snapshots(path, old, new);
    Some(build_diff_lines(&file_diff, width))
}

/// Render a `write_file` tool call as a syntax-highlighted code block.
///
/// Syntax is detected from the file extension via syntect's built-in mappings
/// (the same strategy used by `highlight_code_line` for diff lines).
/// A persistent `HighlightLines` is used across all lines to preserve
/// multi-line highlighter state — this is why `highlight_code_line` is not
/// reused here, which resets state on every call.
///
/// No diff gutter is shown — there is no previous version to compare against.
pub fn render_write_file(input: &serde_json::Value, width: usize) -> Option<Vec<Line<'static>>> {
    let path = input.get("path").and_then(|v| v.as_str())?;
    let content = input.get("content").and_then(|v| v.as_str())?;

    let added_style = Style::default().fg(Color::Green);
    let label_style = Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD);

    let file_label = format!("New file: {path}");
    let mut lines: Vec<Line<'static>> = vec![Line::from(Span::styled(file_label, label_style))];

    let syntax = SYNTAX_SET
        .find_syntax_for_file(Path::new(path))
        .ok()
        .flatten()
        .unwrap_or_else(|| SYNTAX_SET.find_syntax_plain_text());

    let mut highlighter = HighlightLines::new(syntax, default_theme());

    let max_line_no = content.lines().count().max(1);
    let digits = max_line_no.to_string().len().max(MIN_LINE_NO_DIGITS);
    // gutter: line number + trailing space (no old-column — this is a new file)
    let gutter_width = digits + 1;
    let content_width = width.saturating_sub(gutter_width + 1); // +1 for "+" marker

    for (idx, source_line) in content.lines().enumerate() {
        let line_no = idx + 1;
        let gutter = format!("{line_no:>digits$} ");

        let line_with_newline = format!("{source_line}\n");
        let ranges = highlighter
            .highlight_line(&line_with_newline, &SYNTAX_SET)
            .unwrap_or_default();

        let mut spans: Vec<Span<'static>> = vec![
            Span::styled(gutter, added_style),
            Span::styled("+", added_style),
        ];

        if ranges.is_empty() {
            spans.push(Span::styled(source_line.to_string(), added_style));
        } else {
            for (hl_style, text) in ranges {
                let content_part = text.trim_end_matches('\n').to_string();
                if content_part.is_empty() {
                    continue;
                }
                let fg = syntect_to_ratatui_color(hl_style.foreground);
                spans.push(Span::styled(content_part, added_style.fg(fg)));
            }
        }

        truncate_spans(&mut spans, content_width);
        lines.push(Line::from(spans));
    }

    Some(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── build_inline_diff_spans ──────────────────────────────────────────────

    #[test]
    fn inline_diff_equal_content_produces_no_highlighted_spans() {
        let (old, new) = build_inline_diff_spans("hello world", "hello world");
        // All words should have equal (gray) style — no background set
        for span in old.iter().chain(new.iter()) {
            assert_eq!(
                span.style.bg, None,
                "equal words should not have background highlight"
            );
        }
    }

    #[test]
    fn inline_diff_single_word_change_highlights_only_changed_word() {
        let (old_spans, new_spans) = build_inline_diff_spans("hello world", "hello earth");

        // "hello" and the space should be equal (gray, no bg)
        // "world" should be highlighted (has bg) in old
        let old_highlighted: Vec<_> = old_spans.iter().filter(|s| s.style.bg.is_some()).collect();
        assert_eq!(old_highlighted.len(), 1);
        assert!(old_highlighted[0].content.contains("world"));

        // "earth" should be highlighted in new
        let new_highlighted: Vec<_> = new_spans.iter().filter(|s| s.style.bg.is_some()).collect();
        assert_eq!(new_highlighted.len(), 1);
        assert!(new_highlighted[0].content.contains("earth"));
    }

    #[test]
    fn inline_diff_multi_word_change_highlights_all_changed_words() {
        let (old_spans, new_spans) = build_inline_diff_spans("foo bar baz", "foo qux quux");

        let old_highlighted: Vec<_> = old_spans.iter().filter(|s| s.style.bg.is_some()).collect();
        let new_highlighted: Vec<_> = new_spans.iter().filter(|s| s.style.bg.is_some()).collect();

        assert!(
            !old_highlighted.is_empty(),
            "should have highlighted old words"
        );
        assert!(
            !new_highlighted.is_empty(),
            "should have highlighted new words"
        );
    }

    #[test]
    fn inline_diff_empty_old_all_new_words_highlighted() {
        let (old_spans, new_spans) = build_inline_diff_spans("", "brand new line");
        assert!(old_spans.is_empty(), "no old spans for empty old");
        let highlighted: Vec<_> = new_spans.iter().filter(|s| s.style.bg.is_some()).collect();
        assert!(!highlighted.is_empty(), "new words should be highlighted");
    }

    // ── build_file_diff_from_snapshots ───────────────────────────────────────

    #[test]
    fn file_diff_new_file_sets_is_new_file_flag() {
        let diff = build_file_diff_from_snapshots("new.rs", "", "fn main() {}\n");
        assert!(diff.is_new_file, "empty before → is_new_file");
        assert!(diff.added > 0);
        assert_eq!(diff.removed, 0);
    }

    #[test]
    fn file_diff_modified_file_counts_added_and_removed() {
        let before = "line one\nline two\nline three\n";
        let after = "line one\nline TWO\nline three\n";
        let diff = build_file_diff_from_snapshots("file.txt", before, after);
        assert!(!diff.is_new_file);
        assert_eq!(diff.added, 1);
        assert_eq!(diff.removed, 1);
    }

    #[test]
    fn file_diff_no_changes_produces_no_hunks() {
        let content = "unchanged\n";
        let diff = build_file_diff_from_snapshots("same.txt", content, content);
        assert_eq!(diff.added, 0);
        assert_eq!(diff.removed, 0);
        assert!(diff.hunks.is_empty(), "identical files produce no hunks");
    }

    #[test]
    fn file_diff_hunks_contain_header_lines() {
        let before = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n";
        let after = "a\nb\nc\nd\nX\nf\ng\nh\ni\nj\n";
        let diff = build_file_diff_from_snapshots("f.txt", before, after);
        assert!(!diff.hunks.is_empty());
        let first_hunk = &diff.hunks[0];
        assert!(
            first_hunk.lines[0].kind == DiffLineKind::Header,
            "first line of hunk should be Header"
        );
        assert!(
            first_hunk.lines[0].content.starts_with("@@"),
            "header should start with @@"
        );
    }

    #[test]
    fn file_diff_path_is_preserved() {
        let diff = build_file_diff_from_snapshots("src/foo.rs", "old\n", "new\n");
        assert_eq!(diff.path, "src/foo.rs");
    }

    // ── build_diff_lines ─────────────────────────────────────────────────────

    #[test]
    fn build_diff_lines_returns_at_least_file_header() {
        let diff = build_file_diff_from_snapshots("test.txt", "old\n", "new\n");
        let lines = build_diff_lines(&diff, 80);
        assert!(!lines.is_empty(), "should produce at least one line");
        let first_text: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            first_text.contains("test.txt"),
            "first line should mention the file path"
        );
    }

    #[test]
    fn build_diff_lines_new_file_label_says_new_file() {
        let diff = build_file_diff_from_snapshots("fresh.rs", "", "fn main() {}\n");
        let lines = build_diff_lines(&diff, 80);
        let first_text: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            first_text.contains("New file"),
            "should say 'New file' for brand-new files, got: {first_text}"
        );
    }

    #[test]
    fn build_diff_lines_produces_added_lines_with_plus_marker() {
        let diff = build_file_diff_from_snapshots("a.txt", "", "hello\n");
        let lines = build_diff_lines(&diff, 80);
        let has_plus = lines.iter().any(|l| {
            l.spans
                .iter()
                .any(|s| s.content.as_ref() == "+" && s.style.fg == Some(Color::Green))
        });
        assert!(has_plus, "should have green '+' marker for added lines");
    }

    #[test]
    fn build_diff_lines_produces_removed_lines_with_minus_marker() {
        let diff = build_file_diff_from_snapshots("a.txt", "hello\n", "");
        let lines = build_diff_lines(&diff, 80);
        let has_minus = lines.iter().any(|l| {
            l.spans
                .iter()
                .any(|s| s.content.as_ref() == "-" && s.style.fg == Some(Color::Red))
        });
        assert!(has_minus, "should have red '-' marker for removed lines");
    }

    #[test]
    fn build_diff_lines_inline_word_diff_for_adjacent_changed_lines() {
        let before = "let x = 1;\n";
        let after = "let x = 42;\n";
        let diff = build_file_diff_from_snapshots("main.rs", before, after);
        let lines = build_diff_lines(&diff, 80);
        // Both the removed and added lines should have spans with a background colour
        // (the word-level highlights).
        let highlighted_lines: Vec<_> = lines
            .iter()
            .filter(|l| l.spans.iter().any(|s| s.style.bg.is_some()))
            .collect();
        assert!(
            !highlighted_lines.is_empty(),
            "adjacent changed lines should produce word-level highlights"
        );
    }

    // ── render_edit_file_diff / render_write_file ────────────────────────────

    #[test]
    fn render_edit_file_diff_returns_none_for_missing_fields() {
        let input = serde_json::json!({"path": "f.txt"});
        assert!(render_edit_file_diff(&input, 80).is_none());
    }

    #[test]
    fn render_write_file_returns_none_for_missing_fields() {
        let input = serde_json::json!({"path": "f.txt"});
        assert!(render_write_file(&input, 80).is_none());
    }

    #[test]
    fn render_edit_file_diff_returns_lines_for_valid_input() {
        let input = serde_json::json!({
            "path": "src/main.rs",
            "old_string": "let x = 1;",
            "new_string": "let x = 42;"
        });
        let result = render_edit_file_diff(&input, 80);
        assert!(result.is_some());
        assert!(!result.expect("some").is_empty());
    }

    #[test]
    fn render_write_file_returns_lines_for_valid_input() {
        let input = serde_json::json!({
            "path": "hello.txt",
            "content": "Hello, world!\n"
        });
        let result = render_write_file(&input, 80);
        assert!(result.is_some());
        assert!(!result.expect("some").is_empty());
    }

    #[test]
    fn render_write_file_shows_file_label_with_path() {
        let input = serde_json::json!({
            "path": "src/foo.py",
            "content": "print('hello')\n"
        });
        let lines = render_write_file(&input, 80).expect("should render");
        let first: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            first.contains("src/foo.py"),
            "first line should contain the file path, got: {first}"
        );
        assert!(
            first.contains("New file"),
            "first line should say 'New file', got: {first}"
        );
    }

    #[test]
    fn render_write_file_each_content_line_has_plus_marker() {
        let input = serde_json::json!({
            "path": "hello.rs",
            "content": "fn main() {}\n"
        });
        let lines = render_write_file(&input, 80).expect("should render");
        // Skip header line; every content line should have a "+" span
        for line in lines.iter().skip(1) {
            let has_plus = line.spans.iter().any(|s| s.content.as_ref() == "+");
            assert!(has_plus, "content line should have '+' marker: {line:?}");
        }
    }

    #[test]
    fn render_write_file_no_diff_gutter_two_columns() {
        // write_file has no old-column, just a single line-number column.
        // Verify the gutter doesn't have the old "     N " two-column format.
        let input = serde_json::json!({
            "path": "a.txt",
            "content": "line one\nline two\n"
        });
        let lines = render_write_file(&input, 80).expect("should render");
        // Content lines (skip header): gutter should be just "N " (one number)
        for line in lines.iter().skip(1) {
            let gutter_text: String = line
                .spans
                .iter()
                .take(1)
                .map(|s| s.content.as_ref())
                .collect();
            // Should be a short number + space, not two numbers
            assert!(
                !gutter_text.contains("  "),
                "gutter should not have two-column padding, got: {gutter_text:?}"
            );
        }
    }

    // ── insta snapshot tests ─────────────────────────────────────────────────

    #[test]
    fn snapshot_diff_lines_edit_file() {
        let before = "fn greet(name: &str) {\n    println!(\"Hello {name}\");\n}\n";
        let after = "fn greet(name: &str) {\n    println!(\"Hi {name}!\");\n}\n";
        let diff = build_file_diff_from_snapshots("src/greet.rs", before, after);
        let lines = build_diff_lines(&diff, 80);

        let rendered: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();

        insta::assert_debug_snapshot!("diff_lines_edit_file", rendered);
    }

    #[test]
    fn snapshot_diff_lines_new_file() {
        let input = serde_json::json!({
            "path": "src/main.rs",
            "content": "fn main() {\n    println!(\"Hello!\");\n}\n"
        });
        let lines = render_write_file(&input, 80).expect("should render");

        let rendered: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();

        insta::assert_debug_snapshot!("diff_lines_new_file", rendered);
    }
}
