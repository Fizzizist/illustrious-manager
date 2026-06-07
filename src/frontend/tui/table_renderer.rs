use ratatui::style::Style;
use ratatui::text::{Line, Span};
use the_other_tui_markdown::Theme;
use unicode_width::UnicodeWidthStr;

const MIN_COL_WIDTH: usize = 3;

const TABLE_SEP: &str = " │ ";
const SEP_INNER: &str = "─┼─";
const SEP_DASH: char = '─';

pub fn render_table(
    header: &[String],
    rows: &[Vec<String>],
    theme: &Theme,
    max_width: usize,
) -> Vec<Line<'static>> {
    let ncols = header
        .len()
        .max(rows.iter().map(|r| r.len()).max().unwrap_or(0));
    if ncols == 0 {
        return Vec::new();
    }

    let col_widths = compute_col_widths(header, rows, ncols, max_width);

    let mut out = Vec::new();

    // Header row (may wrap if header text exceeds column width)
    for line in wrap_row(header, ncols, &col_widths, theme.table_header) {
        out.push(line);
    }

    // Separator row
    out.push(make_separator(&col_widths, theme.table_separator));

    // Body rows
    for row in rows {
        for line in wrap_row(row, ncols, &col_widths, theme.table_cell) {
            out.push(line);
        }
    }

    out
}

fn compute_col_widths(
    header: &[String],
    rows: &[Vec<String>],
    ncols: usize,
    max_width: usize,
) -> Vec<usize> {
    let sep_overhead = TABLE_SEP.len() * (ncols.saturating_sub(1));

    let natural_widths: Vec<usize> = (0..ncols)
        .map(|i| {
            let hw = header
                .get(i)
                .map(|h| UnicodeWidthStr::width(h.as_str()))
                .unwrap_or(0);
            let rw = rows
                .iter()
                .map(|r| {
                    r.get(i)
                        .map(|c| UnicodeWidthStr::width(c.as_str()))
                        .unwrap_or(0)
                })
                .max()
                .unwrap_or(0);
            hw.max(rw)
        })
        .collect();

    let total_natural: usize = natural_widths.iter().sum::<usize>() + sep_overhead;

    if total_natural <= max_width {
        return natural_widths;
    }

    // Shrink proportionally
    let available = max_width.saturating_sub(sep_overhead);
    let min_total = MIN_COL_WIDTH * ncols;
    let available = available.max(min_total);

    let mut widths = Vec::with_capacity(ncols);
    let natural_total: usize = natural_widths.iter().sum();
    for w in natural_widths.iter() {
        let proportional = (*w * available)
            .checked_div(natural_total.max(1))
            .unwrap_or(MIN_COL_WIDTH)
            .max(MIN_COL_WIDTH);
        widths.push(proportional);
    }

    widths
}

fn make_separator(col_widths: &[usize], style: Style) -> Line<'static> {
    let mut spans = Vec::new();
    for (i, w) in col_widths.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(SEP_INNER.to_string(), style));
        }
        spans.push(Span::styled(SEP_DASH.to_string().repeat(*w), style));
    }
    Line::from(spans)
}

fn wrap_row(
    row: &[String],
    ncols: usize,
    col_widths: &[usize],
    style: Style,
) -> Vec<Line<'static>> {
    let wrapped_cells: Vec<Vec<String>> = (0..ncols)
        .map(|i| {
            let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
            wrap_cell_text(cell, col_widths[i])
        })
        .collect();

    let max_lines = wrapped_cells.iter().map(|c| c.len()).max().unwrap_or(1);

    let mut out = Vec::new();
    for line_idx in 0..max_lines {
        let mut spans = Vec::new();
        for (i, width) in col_widths.iter().enumerate().take(ncols) {
            if i > 0 {
                spans.push(Span::styled(TABLE_SEP.to_string(), style));
            }
            let text = wrapped_cells
                .get(i)
                .and_then(|lines| lines.get(line_idx))
                .map(|s| s.as_str())
                .unwrap_or("");
            let padded = pad_cell(text, *width);
            spans.push(Span::styled(padded, style));
        }
        out.push(Line::from(spans));
    }

    out
}

fn wrap_cell_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;

    for word in text.split(' ') {
        let word_width = UnicodeWidthStr::width(if word.is_empty() { " " } else { word });
        if current_width == 0 {
            current.push_str(word);
            current_width = word_width;
        } else if current_width + 1 + word_width <= width {
            current.push(' ');
            current.push_str(word);
            current_width += 1 + word_width;
        } else {
            if !current.is_empty() {
                lines.push(std::mem::take(&mut current));
            }
            current.push_str(word);
            current_width = word_width;
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    // Handle words wider than the column width
    lines
        .into_iter()
        .flat_map(|line| {
            if UnicodeWidthStr::width(line.as_str()) <= width {
                vec![line]
            } else {
                hard_wrap(&line, width)
            }
        })
        .collect()
}

fn hard_wrap(s: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;

    for ch in s.chars() {
        let ch_width = UnicodeWidthStr::width(ch.to_string().as_str());
        if current_width + ch_width > width && current_width > 0 {
            lines.push(std::mem::take(&mut current));
            current_width = 0;
        }
        current.push(ch);
        current_width += ch_width;
    }
    if !current.is_empty() {
        lines.push(current);
    }

    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn pad_cell(s: &str, width: usize) -> String {
    let display_w = UnicodeWidthStr::width(s);
    if display_w < width {
        format!("{}{}", s, " ".repeat(width - display_w))
    } else {
        s.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Color;

    fn test_theme() -> Theme {
        Theme::default()
    }

    #[test]
    fn simple_table_renders() {
        let header = vec!["Name".to_string(), "Age".to_string()];
        let rows = vec![vec!["Alice".to_string(), "30".to_string()]];
        let lines = render_table(&header, &rows, &test_theme(), 80);
        assert!(
            lines.len() >= 3,
            "should have header, separator, and body row"
        );
    }

    #[test]
    fn wide_table_wraps_cells() {
        let header = vec![
            "Column One".to_string(),
            "Column Two".to_string(),
            "Column Three".to_string(),
            "Column Four".to_string(),
        ];
        let rows = vec![vec![
            "Long value here".to_string(),
            "Another long value".to_string(),
            "Yet another long value".to_string(),
            "Final long value".to_string(),
        ]];
        let lines = render_table(&header, &rows, &test_theme(), 38);
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<&str>>()
            .join("");
        assert!(
            text.contains('│') || text.contains('|'),
            "should contain column separators"
        );
    }

    #[test]
    fn narrow_table_fits_without_wrapping() {
        let header = vec!["A".to_string(), "B".to_string()];
        let rows = vec![vec!["1".to_string(), "2".to_string()]];
        let lines = render_table(&header, &rows, &test_theme(), 80);
        assert!(
            lines.len() == 3,
            "narrow table should have exactly 3 lines (header + sep + row)"
        );
    }

    #[test]
    fn wrap_cell_text_simple() {
        let lines = wrap_cell_text("hello world", 20);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "hello world");
    }

    #[test]
    fn wrap_cell_text_wraps_long_text() {
        let lines = wrap_cell_text("Long value here Another long value", 15);
        assert!(lines.len() > 1, "should wrap to multiple lines");
    }

    #[test]
    fn pad_cell_pads_short_string() {
        let result = pad_cell("hi", 5);
        assert_eq!(result, "hi   ");
    }

    #[test]
    fn pad_cell_does_not_pad_exact_fit() {
        let result = pad_cell("hello", 5);
        assert_eq!(result, "hello");
    }

    #[test]
    fn render_table_separator_contains_separator_chars() {
        let header = vec!["A".to_string(), "B".to_string()];
        let rows = vec![vec!["1".to_string(), "2".to_string()]];
        let lines = render_table(&header, &rows, &test_theme(), 80);
        let sep_line = &lines[1];
        let text: String = sep_line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains('┼'), "separator should contain ┼");
    }
}
