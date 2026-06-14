use ratatui::style::Style;
use ratatui::text::{Line, Span};
use the_other_tui_markdown::Theme;
use unicode_width::UnicodeWidthStr;

const MIN_COL_WIDTH: usize = 3;

const TABLE_SEP: &str = " │ ";
const SEP_INNER: &str = "─┼─";
const SEP_DASH: char = '─';

pub fn render_table(
    header: &[Vec<Span<'static>>],
    rows: &[Vec<Vec<Span<'static>>>],
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

    for line in wrap_row(header, ncols, &col_widths, theme.table_header) {
        out.push(line);
    }

    out.push(make_separator(&col_widths, theme.table_separator));

    for row in rows {
        for line in wrap_row(row, ncols, &col_widths, theme.table_cell) {
            out.push(line);
        }
    }

    out
}

fn compute_col_widths(
    header: &[Vec<Span<'static>>],
    rows: &[Vec<Vec<Span<'static>>>],
    ncols: usize,
    max_width: usize,
) -> Vec<usize> {
    let sep_overhead = TABLE_SEP.len() * (ncols.saturating_sub(1));

    let natural_widths: Vec<usize> = (0..ncols)
        .map(|i| {
            let hw = header.get(i).map(|cell| span_cell_width(cell)).unwrap_or(0);
            let rw = rows
                .iter()
                .map(|r| r.get(i).map(|cell| span_cell_width(cell)).unwrap_or(0))
                .max()
                .unwrap_or(0);
            hw.max(rw)
        })
        .collect();

    let total_natural: usize = natural_widths.iter().sum::<usize>() + sep_overhead;

    if total_natural <= max_width {
        return natural_widths;
    }

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

fn span_cell_width(cell: &[Span<'static>]) -> usize {
    cell.iter()
        .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
        .sum()
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
    row: &[Vec<Span<'static>>],
    ncols: usize,
    col_widths: &[usize],
    base_style: Style,
) -> Vec<Line<'static>> {
    let wrapped_cells: Vec<Vec<Vec<Span<'static>>>> = (0..ncols)
        .map(|i| {
            let cell = row.get(i).map(Vec::as_slice).unwrap_or_default();
            wrap_cell_spans(cell, col_widths[i], base_style)
        })
        .collect();

    let max_lines = wrapped_cells.iter().map(|c| c.len()).max().unwrap_or(1);

    let mut out = Vec::new();
    for line_idx in 0..max_lines {
        let mut spans = Vec::new();
        for (i, width) in col_widths.iter().enumerate().take(ncols) {
            if i > 0 {
                spans.push(Span::styled(TABLE_SEP.to_string(), base_style));
            }
            let cell_lines = wrapped_cells
                .get(i)
                .expect("wrapped_cells should have ncols entries");
            if let Some(cell_line) = cell_lines.get(line_idx) {
                let padded = pad_cell_spans(cell_line, *width, base_style);
                spans.extend(padded);
            } else {
                spans.push(Span::styled(" ".repeat(*width), base_style));
            }
        }
        out.push(Line::from(spans));
    }

    out
}

fn wrap_cell_spans(
    cell: &[Span<'static>],
    width: usize,
    base_style: Style,
) -> Vec<Vec<Span<'static>>> {
    // Invariant: wrap_cell_text must not collapse or elide whitespace. src_pos
    // tracks position in the flat char vector, advancing only for characters
    // that exist in the original cell content. If wrap_cell_text were changed
    // to collapse whitespace, this mapping would desync.
    if width == 0 {
        return vec![vec![]];
    }

    let flat: Vec<(char, Style)> = cell
        .iter()
        .flat_map(|span| span.content.chars().map(move |ch| (ch, span.style)))
        .collect();

    if flat.is_empty() {
        return vec![pad_cell_spans(&[], width, base_style)];
    }

    let plain_text: String = flat.iter().map(|(ch, _)| *ch).collect();
    let wrapped_lines = wrap_cell_text(&plain_text, width);

    let mut result = Vec::with_capacity(wrapped_lines.len());
    let mut src_pos = 0usize;

    for line_text in &wrapped_lines {
        let mut line_spans: Vec<Span<'static>> = Vec::new();
        let mut current_content = String::new();
        let mut current_style = Style::default();
        let mut first_char = true;

        for ch in line_text.chars() {
            let style = if ch == ' ' && src_pos < flat.len() && flat[src_pos].0 != ' ' {
                base_style
            } else if src_pos < flat.len() {
                let s = flat[src_pos].1;
                src_pos += 1;
                s
            } else {
                base_style
            };

            if first_char {
                current_style = style;
                current_content.push(ch);
                first_char = false;
            } else if style == current_style {
                current_content.push(ch);
            } else {
                line_spans.push(Span::styled(
                    std::mem::take(&mut current_content),
                    current_style,
                ));
                current_content.push(ch);
                current_style = style;
            }
        }

        if !current_content.is_empty() {
            line_spans.push(Span::styled(current_content, current_style));
        }

        result.push(line_spans);
    }

    result
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

fn pad_cell_spans(spans: &[Span<'static>], width: usize, base_style: Style) -> Vec<Span<'static>> {
    let mut out: Vec<Span<'static>> = spans
        .iter()
        .map(|s| Span {
            content: s.content.clone(),
            style: base_style.patch(s.style),
        })
        .collect();
    let cell_w = out
        .iter()
        .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
        .sum::<usize>();
    if cell_w < width {
        out.push(Span::styled(" ".repeat(width - cell_w), base_style));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Color;

    fn test_theme() -> Theme {
        Theme::default()
    }

    fn plain_cell(s: &str) -> Vec<Span<'static>> {
        vec![Span::raw(s.to_string())]
    }

    fn plain_row(cells: &[&str]) -> Vec<Vec<Span<'static>>> {
        cells.iter().map(|c| plain_cell(c)).collect()
    }

    fn plain_header(cells: &[&str]) -> Vec<Vec<Span<'static>>> {
        cells.iter().map(|c| plain_cell(c)).collect()
    }

    #[test]
    fn simple_table_renders() {
        let header = plain_header(&["Name", "Age"]);
        let rows = vec![plain_row(&["Alice", "30"])];
        let lines = render_table(&header, &rows, &test_theme(), 80);
        assert!(
            lines.len() >= 3,
            "should have header, separator, and body row"
        );
    }

    #[test]
    fn wide_table_wraps_cells() {
        let header = plain_header(&["Column One", "Column Two", "Column Three", "Column Four"]);
        let rows = vec![plain_row(&[
            "Long value here",
            "Another long value",
            "Yet another long value",
            "Final long value",
        ])];
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
        let header = plain_header(&["A", "B"]);
        let rows = vec![plain_row(&["1", "2"])];
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
    fn render_table_separator_contains_separator_chars() {
        let header = plain_header(&["A", "B"]);
        let rows = vec![plain_row(&["1", "2"])];
        let lines = render_table(&header, &rows, &test_theme(), 80);
        let sep_line = &lines[1];
        let text: String = sep_line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains('┼'), "separator should contain ┼");
    }

    #[test]
    fn separator_and_data_alignment() {
        let header = plain_header(&["A", "B", "C"]);
        let rows = vec![plain_row(&["1", "2", "3"])];
        let lines = render_table(&header, &rows, &test_theme(), 80);
        let header_line: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        let sep_line: String = lines[1].spans.iter().map(|s| s.content.as_ref()).collect();
        let data_line: String = lines[2].spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(
            UnicodeWidthStr::width(header_line.as_str()),
            UnicodeWidthStr::width(sep_line.as_str()),
            "header and separator should have same width\nheader: {header_line}\nsep:    {sep_line}"
        );
        assert_eq!(
            UnicodeWidthStr::width(header_line.as_str()),
            UnicodeWidthStr::width(data_line.as_str()),
            "header and data should have same width\nheader: {header_line}\ndata:   {data_line}"
        );
    }

    #[test]
    fn styled_spans_survive_wrapping() {
        let cell = vec![
            Span::styled("hello".to_string(), Style::default().fg(Color::Red)),
            Span::styled(" world".to_string(), Style::default().fg(Color::Blue)),
        ];
        let wrapped = wrap_cell_spans(&cell, 20, Style::default());
        assert_eq!(wrapped.len(), 1, "should fit on one line at width 20");
        let line = &wrapped[0];
        let text: String = line
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<Vec<&str>>()
            .join("");
        assert_eq!(text.trim(), "hello world");
        let red_count = line
            .iter()
            .filter(|s| s.style.fg == Some(Color::Red))
            .count();
        let blue_count = line
            .iter()
            .filter(|s| s.style.fg == Some(Color::Blue))
            .count();
        assert!(
            red_count >= 1,
            "should have at least one red span, got: {line:?}"
        );
        assert!(
            blue_count >= 1,
            "should have at least one blue span, got: {line:?}"
        );
    }

    #[test]
    fn styled_spans_survive_wrapping_narrow() {
        let cell = vec![
            Span::styled("hello".to_string(), Style::default().fg(Color::Red)),
            Span::styled(" world".to_string(), Style::default().fg(Color::Blue)),
        ];
        let wrapped = wrap_cell_spans(&cell, 6, Style::default());
        assert!(
            wrapped.len() > 1,
            "should wrap to multiple lines at width 6"
        );
        let all_text: String = wrapped
            .iter()
            .flat_map(|line| line.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<&str>>()
            .join("");
        assert!(
            all_text.contains("hello"),
            "should contain 'hello' across lines"
        );
        assert!(
            all_text.contains("world"),
            "should contain 'world' across lines"
        );
        let has_red = wrapped
            .iter()
            .any(|line| line.iter().any(|s| s.style.fg == Some(Color::Red)));
        let has_blue = wrapped
            .iter()
            .any(|line| line.iter().any(|s| s.style.fg == Some(Color::Blue)));
        assert!(has_red, "should preserve red style across lines");
        assert!(has_blue, "should preserve blue style across lines");
    }

    #[test]
    fn base_style_merging() {
        let cell = vec![Span::styled(
            "hello".to_string(),
            Style::default().fg(Color::Red),
        )];
        let base = Style::default()
            .fg(Color::White)
            .add_modifier(ratatui::style::Modifier::BOLD);
        let padded = pad_cell_spans(&cell, 10, base);
        let text: String = padded
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<Vec<&str>>()
            .join("");
        assert_eq!(text, "hello     ");
        let styled_span = padded
            .iter()
            .find(|s| s.content.as_ref() == "hello")
            .expect("should find 'hello' span");
        assert_eq!(
            styled_span.style.fg,
            Some(Color::Red),
            "span's explicit fg should override base: {:?}",
            styled_span.style
        );
        assert!(
            styled_span
                .style
                .add_modifier
                .contains(ratatui::style::Modifier::BOLD),
            "base bold should be patched onto span: {:?}",
            styled_span.style
        );
    }

    #[test]
    fn empty_cell_renders_as_padding() {
        let cell: Vec<Span<'static>> = vec![];
        let wrapped = wrap_cell_spans(&cell, 10, Style::default());
        assert_eq!(wrapped.len(), 1, "empty cell should produce one line");
        let text: String = wrapped[0]
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<Vec<&str>>()
            .join("");
        assert_eq!(text, "          ", "empty cell should pad to column width");
    }

    #[test]
    fn pad_cell_spans_short_spans() {
        let cell = vec![Span::raw("hi".to_string())];
        let padded = pad_cell_spans(&cell, 5, Style::default());
        let text: String = padded
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<Vec<&str>>()
            .join("");
        assert_eq!(text, "hi   ");
    }

    #[test]
    fn pad_cell_spans_exact_fit() {
        let cell = vec![Span::raw("hello".to_string())];
        let padded = pad_cell_spans(&cell, 5, Style::default());
        let text: String = padded
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<Vec<&str>>()
            .join("");
        assert_eq!(text, "hello");
    }

    #[test]
    fn render_table_preserves_styled_spans() {
        let header = vec![
            vec![Span::styled(
                "Name".to_string(),
                Style::default().fg(Color::Cyan),
            )],
            vec![Span::styled(
                "Value".to_string(),
                Style::default().fg(Color::Yellow),
            )],
        ];
        let rows = vec![vec![
            vec![Span::styled(
                "alpha".to_string(),
                Style::default().fg(Color::Red),
            )],
            vec![Span::styled(
                "beta".to_string(),
                Style::default().fg(Color::Green),
            )],
        ]];
        let lines = render_table(&header, &rows, &test_theme(), 80);
        assert!(
            lines.len() >= 3,
            "should have header, separator, and body row"
        );

        let header_line = &lines[0];
        let header_text: String = header_line
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert!(
            header_text.contains("Name"),
            "header should contain 'Name': {header_text}"
        );
        assert!(
            header_text.contains("Value"),
            "header should contain 'Value': {header_text}"
        );

        let has_cyan_header = header_line
            .spans
            .iter()
            .any(|s| s.style.fg == Some(Color::Cyan));
        let has_yellow_header = header_line
            .spans
            .iter()
            .any(|s| s.style.fg == Some(Color::Yellow));
        assert!(
            has_cyan_header,
            "header should preserve cyan style on 'Name' span"
        );
        assert!(
            has_yellow_header,
            "header should preserve yellow style on 'Value' span"
        );

        let body_line = &lines[2];
        let body_text: String = body_line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            body_text.contains("alpha"),
            "body should contain 'alpha': {body_text}"
        );
        assert!(
            body_text.contains("beta"),
            "body should contain 'beta': {body_text}"
        );

        let has_red_body = body_line
            .spans
            .iter()
            .any(|s| s.style.fg == Some(Color::Red));
        let has_green_body = body_line
            .spans
            .iter()
            .any(|s| s.style.fg == Some(Color::Green));
        assert!(
            has_red_body,
            "body should preserve red style on 'alpha' span"
        );
        assert!(
            has_green_body,
            "body should preserve green style on 'beta' span"
        );
    }

    #[test]
    fn styled_spans_character_level_mapping() {
        let cell = vec![
            Span::styled("AB".to_string(), Style::default().fg(Color::Red)),
            Span::styled("CD".to_string(), Style::default().fg(Color::Blue)),
        ];
        let wrapped = wrap_cell_spans(&cell, 20, Style::default());
        assert_eq!(wrapped.len(), 1, "should fit on one line at width 20");

        let line = &wrapped[0];
        let ab_span = line
            .iter()
            .find(|s| s.content.contains('A'))
            .expect("should find span containing 'A'");
        assert_eq!(
            ab_span.style.fg,
            Some(Color::Red),
            "'AB' span should be Red"
        );

        let cd_span = line
            .iter()
            .find(|s| s.content.contains('C'))
            .expect("should find span containing 'C'");
        assert_eq!(
            cd_span.style.fg,
            Some(Color::Blue),
            "'CD' span should be Blue"
        );
    }

    #[test]
    fn styled_spans_wrapping_maps_characters_correctly() {
        let cell = vec![
            Span::styled("redtext".to_string(), Style::default().fg(Color::Red)),
            Span::styled(" bluepart".to_string(), Style::default().fg(Color::Blue)),
        ];
        let wrapped = wrap_cell_spans(&cell, 8, Style::default());
        assert!(wrapped.len() > 1, "should wrap at width 8");

        let first_line_text: String = wrapped[0]
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<Vec<&str>>()
            .join("");
        assert!(
            first_line_text.trim().starts_with("redtext"),
            "first line should start with 'redtext': {first_line_text}"
        );

        let first_line_has_red = wrapped[0]
            .iter()
            .any(|s| s.content.contains("red") && s.style.fg == Some(Color::Red));
        assert!(
            first_line_has_red,
            "first line should have 'red' characters with Red style"
        );

        let blue_spans: Vec<&ratatui::text::Span> = wrapped
            .iter()
            .flat_map(|line| line.iter())
            .filter(|s| s.style.fg == Some(Color::Blue))
            .collect();
        assert!(
            !blue_spans.is_empty(),
            "should have at least one Blue span across wrapped lines"
        );
        let blue_text: String = blue_spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<Vec<&str>>()
            .join("");
        assert!(
            blue_text.contains("blue"),
            "Blue spans should contain 'bluepart' text: {blue_text}"
        );
    }

    #[test]
    fn span_cell_width_with_wide_characters() {
        let cell = vec![Span::raw("你好世界".to_string())];
        let width = span_cell_width(&cell);
        assert_eq!(
            width, 8,
            "CJK characters should be 2 display width each: 4 chars × 2 = 8"
        );

        let cell_mixed = vec![Span::raw("🦀Rust".to_string())];
        let width_mixed = span_cell_width(&cell_mixed);
        assert_eq!(
            width_mixed, 6,
            "emoji is 2 display width + 4 ASCII chars = 6"
        );
    }

    #[test]
    fn wrap_cell_spans_with_wide_characters() {
        let cell = vec![Span::raw("你好世界测试".to_string())];
        let wrapped = wrap_cell_spans(&cell, 6, Style::default());
        assert!(wrapped.len() > 1, "wide text should wrap at narrow width");
        let total_text: String = wrapped
            .iter()
            .flat_map(|line| line.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<&str>>()
            .join("");
        assert!(
            total_text.contains("你"),
            "wrapped output should preserve all characters"
        );
    }
}
