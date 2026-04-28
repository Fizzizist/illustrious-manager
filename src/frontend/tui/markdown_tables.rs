use pulldown_cmark::{Alignment, Event, Options, Parser, Tag, TagEnd};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

const MIN_COL_WIDTH: usize = 3;

/// Pre-process GFM tables in `input` into monospaced fenced code blocks so
/// that `tui_markdown::from_str` renders them as verbatim text.
///
/// Non-table input is returned unchanged.  If the input contains no `|`
/// character the function returns immediately without invoking the parser.
///
/// Tables wider than `width` are proportionally shrunk with cell wrapping.
/// If even the minimum column widths exceed the budget, the table overflows
/// (documented fallback behavior).
pub fn preprocess_tables(input: &str, width: usize) -> String {
    if !input.contains('|') {
        return input.to_owned();
    }

    let options = Options::ENABLE_TABLES;

    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    let parser = Parser::new_ext(input, options).into_offset_iter();

    let mut in_table = false;
    let mut in_head = false;
    let mut table_start: usize = 0;
    let mut table_end: usize;
    let mut alignments: Vec<Alignment> = Vec::new();
    let mut header: Vec<String> = Vec::new();
    let mut body: Vec<Vec<String>> = Vec::new();
    let mut current_row: Vec<String> = Vec::new();
    let mut current_cell = String::new();

    for (event, range) in parser {
        match event {
            Event::Start(Tag::Table(aligns)) => {
                in_table = true;
                table_start = range.start;
                alignments = aligns;
                header.clear();
                body.clear();
                current_row.clear();
                current_cell.clear();
            }
            Event::End(TagEnd::Table) => {
                table_end = range.end;
                in_table = false;
                in_head = false;
                let rendered = render_table(&alignments, &header, &body, width);
                replacements.push((table_start, table_end, rendered));
            }
            Event::Start(Tag::TableHead) if in_table => {
                in_head = true;
            }
            Event::End(TagEnd::TableHead) if in_table => {
                in_head = false;
                header = std::mem::take(&mut current_row);
            }
            Event::Start(Tag::TableRow) if in_table => {
                current_row.clear();
            }
            Event::End(TagEnd::TableRow) if in_table && !in_head => {
                body.push(std::mem::take(&mut current_row));
            }
            Event::Start(Tag::TableCell) if in_table => {
                current_cell.clear();
            }
            Event::End(TagEnd::TableCell) if in_table => {
                current_row.push(current_cell.trim().to_owned());
                current_cell.clear();
            }
            Event::Text(t) if in_table => {
                current_cell.push_str(&t);
            }
            Event::Code(t) if in_table => {
                current_cell.push('`');
                current_cell.push_str(&t);
                current_cell.push('`');
            }
            Event::SoftBreak | Event::HardBreak if in_table => {
                current_cell.push(' ');
            }
            _ => {}
        }
    }

    if replacements.is_empty() {
        return input.to_owned();
    }

    let mut output = input.to_owned();
    for (start, end, replacement) in replacements.into_iter().rev() {
        output.replace_range(start..end, &replacement);
    }
    output
}

fn render_table(
    alignments: &[Alignment],
    header: &[String],
    body: &[Vec<String>],
    budget: usize,
) -> String {
    let num_cols = header.len().max(1);

    let natural_widths: Vec<usize> = (0..num_cols)
        .map(|col| {
            let header_w = header.get(col).map(|s| s.width()).unwrap_or(0);
            let body_w = body
                .iter()
                .map(|row| row.get(col).map(|s| s.width()).unwrap_or(0))
                .max()
                .unwrap_or(0);
            header_w.max(body_w).max(1)
        })
        .collect();

    let col_widths = allocate_widths(&natural_widths, budget);

    let wrapped_header: Vec<Vec<String>> = (0..num_cols)
        .map(|col| {
            let text = header.get(col).map(String::as_str).unwrap_or("");
            wrap_cell(text, col_widths[col])
        })
        .collect();

    let mut out = String::new();
    out.push_str("```\n");
    out.push_str(&border_line('┌', '─', '┬', '┐', &col_widths));
    out.push_str(&format_logical_row(
        &wrapped_header,
        &col_widths,
        alignments,
    ));
    out.push_str(&border_line('╞', '═', '╪', '╡', &col_widths));

    for (row_idx, row) in body.iter().enumerate() {
        let wrapped_row: Vec<Vec<String>> = (0..num_cols)
            .map(|col| {
                let text = row.get(col).map(String::as_str).unwrap_or("");
                wrap_cell(text, col_widths[col])
            })
            .collect();
        out.push_str(&format_logical_row(&wrapped_row, &col_widths, alignments));
        if row_idx + 1 < body.len() {
            out.push_str(&border_line('├', '─', '┼', '┤', &col_widths));
        }
    }

    out.push_str(&border_line('└', '─', '┴', '┘', &col_widths));
    out.push_str("```");
    out
}

pub fn allocate_widths(natural: &[usize], budget: usize) -> Vec<usize> {
    let num_cols = natural.len();
    if num_cols == 0 {
        return Vec::new();
    }

    let border_overhead = num_cols.saturating_mul(3).saturating_add(1);
    let natural_sum: usize = natural.iter().copied().sum();

    if natural_sum.saturating_add(border_overhead) <= budget {
        return natural.to_vec();
    }

    let min_total = num_cols
        .saturating_mul(MIN_COL_WIDTH)
        .saturating_add(border_overhead);
    if min_total > budget {
        return natural.to_vec();
    }

    let budget_for_content = budget.saturating_sub(border_overhead);

    natural
        .iter()
        .map(|&w| {
            let shrunk = (w as f64 * budget_for_content as f64 / natural_sum as f64) as usize;
            shrunk.max(MIN_COL_WIDTH)
        })
        .collect()
}

pub fn wrap_cell(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_owned()];
    }
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_width: usize = 0;

    for grapheme in text.graphemes(true) {
        let g_width = grapheme.width();
        if current_width.saturating_add(g_width) > width && !current.is_empty() {
            lines.push(current);
            current = String::new();
            current_width = 0;
        }
        current.push_str(grapheme);
        current_width = current_width.saturating_add(g_width);
    }

    if !current.is_empty() || lines.is_empty() {
        lines.push(current);
    }

    lines
}

fn format_logical_row(
    cells: &[Vec<String>],
    col_widths: &[usize],
    alignments: &[Alignment],
) -> String {
    let max_lines = cells.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut out = String::new();

    for line_idx in 0..max_lines {
        let mut line = String::from("│");
        for (col, &w) in col_widths.iter().enumerate() {
            let cell = cells
                .get(col)
                .and_then(|lines| lines.get(line_idx))
                .map(String::as_str)
                .unwrap_or("");
            let align = alignments.get(col).copied().unwrap_or(Alignment::None);
            let cell_w = cell.width();
            let padding = w.saturating_sub(cell_w);
            line.push(' ');
            match align {
                Alignment::Right => {
                    line.push_str(&" ".repeat(padding));
                    line.push_str(cell);
                }
                Alignment::Center => {
                    let left_pad = padding / 2;
                    let right_pad = padding - left_pad;
                    line.push_str(&" ".repeat(left_pad));
                    line.push_str(cell);
                    line.push_str(&" ".repeat(right_pad));
                }
                Alignment::Left | Alignment::None => {
                    line.push_str(cell);
                    line.push_str(&" ".repeat(padding));
                }
            }
            line.push(' ');
            line.push('│');
        }
        line.push('\n');
        out.push_str(&line);
    }

    out
}

fn border_line(left: char, fill: char, mid: char, right: char, col_widths: &[usize]) -> String {
    let mut s = String::from(left);
    let last = col_widths.len().saturating_sub(1);
    for (i, &w) in col_widths.iter().enumerate() {
        s.push_str(&fill.to_string().repeat(w + 2));
        if i < last {
            s.push(mid);
        }
    }
    s.push(right);
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_table() -> &'static str {
        "| Header 1 | Header 2 |\n| --- | --- |\n| cell 1 | cell 2 |"
    }

    fn make_aligned_table() -> &'static str {
        "| Left | Center | Right |\n| :--- | :---: | ---: |\n| l | c | r |"
    }

    #[test]
    fn preprocess_no_pipe_returns_unchanged() {
        let input = "no pipes here";
        assert_eq!(preprocess_tables(input, 200), input);
    }

    #[test]
    fn preprocess_simple_table_produces_code_block() {
        let result = preprocess_tables(make_simple_table(), 200);
        assert!(result.starts_with("```\n"));
        assert!(result.ends_with("```"));
        assert!(result.contains("Header 1"));
        assert!(result.contains("Header 2"));
        assert!(result.contains("cell 1"));
        assert!(result.contains("cell 2"));
    }

    #[test]
    fn preprocess_preserves_surrounding_text() {
        let input = "before\n\n| H1 | H2 |\n| --- | --- |\n| a | b |\n\nafter";
        let result = preprocess_tables(input, 200);
        assert!(result.contains("before"), "before text should be preserved");
        assert!(result.contains("after"), "after text should be preserved");
        assert!(result.contains("H1"), "table header should be present");
    }

    #[test]
    fn preprocess_aligned_table() {
        let result = preprocess_tables(make_aligned_table(), 200);
        assert!(result.contains("Left"));
        assert!(result.contains("Center"));
        assert!(result.contains("Right"));
    }

    #[test]
    fn preprocess_empty_cells() {
        let input = "| A | B |\n| --- | --- |\n| | empty |";
        let result = preprocess_tables(input, 200);
        assert!(result.contains("```"));
        assert!(result.contains("empty"));
    }

    #[test]
    fn preprocess_multirow_body() {
        let input = "| H |\n| --- |\n| row1 |\n| row2 |\n| row3 |";
        let result = preprocess_tables(input, 200);
        assert!(result.contains("row1"));
        assert!(result.contains("row2"));
        assert!(result.contains("row3"));
    }

    #[test]
    fn preprocess_single_column() {
        let input = "| Only |\n| --- |\n| value |";
        let result = preprocess_tables(input, 200);
        assert!(result.contains("Only"));
        assert!(result.contains("value"));
    }

    #[test]
    fn preprocess_contains_border_chars() {
        let result = preprocess_tables(make_simple_table(), 200);
        assert!(result.contains('│'));
        assert!(result.contains('┌'));
        assert!(result.contains('┘'));
    }

    #[test]
    fn preprocess_header_separator_uses_double_line() {
        let result = preprocess_tables(make_simple_table(), 200);
        assert!(result.contains('╞'));
        assert!(result.contains('╡'));
        assert!(result.contains('═'));
    }

    #[test]
    fn preprocess_body_rows_separated_by_single_line() {
        let input = "| H |\n| --- |\n| r1 |\n| r2 |";
        let result = preprocess_tables(input, 200);
        assert!(result.contains('├'));
        assert!(result.contains('┤'));
    }

    #[test]
    fn preprocess_code_in_cell() {
        let input = "| H |\n| --- |\n| `code` |";
        let result = preprocess_tables(input, 200);
        assert!(result.contains("`code`"));
    }

    #[test]
    fn preprocess_multiple_tables() {
        let input = "| A |\n| --- |\n| 1 |\n\ntext\n\n| B |\n| --- |\n| 2 |";
        let result = preprocess_tables(input, 200);
        let count = result.matches("```").count();
        assert_eq!(
            count, 4,
            "expected 4 fence markers (open+close) for 2 tables"
        );
    }

    #[test]
    fn preprocess_no_table_returns_unchanged() {
        let input = "just | some | pipe | text";
        let result = preprocess_tables(input, 200);
        assert_eq!(result, input);
    }

    #[test]
    fn preprocess_wide_cells_pad_correctly() {
        let input = "| Short | A very long header |\n| --- | --- |\n| x | y |";
        let result = preprocess_tables(input, 200);
        let lines: Vec<&str> = result.lines().collect();
        let data_lines: Vec<&&str> = lines.iter().filter(|l| l.starts_with('│')).collect();
        assert!(!data_lines.is_empty());
        let first_len = data_lines[0].chars().count();
        for line in &data_lines {
            assert_eq!(line.chars().count(), first_len);
        }
    }

    #[test]
    fn preprocess_right_alignment() {
        let input = "| Num |\n| ---: |\n| 42 |";
        let result = preprocess_tables(input, 200);
        assert!(result.contains("42"));
    }

    #[test]
    fn preprocess_center_alignment() {
        let input = "| Title |\n| :---: |\n| centered |";
        let result = preprocess_tables(input, 200);
        assert!(result.contains("centered"));
    }

    #[test]
    fn preprocess_soft_break_becomes_space() {
        let input = "| H |\n| --- |\n| line1 line2 |";
        let result = preprocess_tables(input, 200);
        assert!(result.contains("line1 line2"));
    }

    #[test]
    fn preprocess_wraps_wide_table_to_budget() {
        let header = "| LongHeader | LongHeader | LongHeader | LongHeader |";
        let sep = "| --- | --- | --- | --- |";
        let row1 = "| Long value 1 | Long value 2 | Long value 3 | Long value 4 |";
        let row2 = "| Long value 5 | Long value 6 | Long value 7 | Long value 8 |";
        let input = format!("{}\n{}\n{}\n{}", header, sep, row1, row2);

        let result = preprocess_tables(&input, 40);

        for line in result.lines() {
            let line_width = line.width();
            assert!(
                line_width <= 40 || line == "```",
                "line too wide ({line_width}): {line:?}"
            );
        }
    }

    #[test]
    fn allocate_widths_proportional_shrink() {
        let result = allocate_widths(&[20, 20, 20], 40);
        let border_overhead = 3 * 3 + 1;
        let sum: usize = result.iter().sum();
        assert!(sum <= 40usize.saturating_sub(border_overhead));
        for &w in &result {
            assert!(w >= MIN_COL_WIDTH);
        }
    }

    #[test]
    fn allocate_widths_no_shrink_when_fits() {
        let result = allocate_widths(&[5, 5], 100);
        assert_eq!(result, vec![5, 5]);
    }

    #[test]
    fn allocate_widths_falls_back_when_minimums_exceed_budget() {
        let natural = vec![10, 10, 10, 10, 10, 10, 10, 10];
        let result = allocate_widths(&natural, 20);
        assert_eq!(result, natural);
    }

    #[test]
    fn wrap_cell_grapheme_safe() {
        let lines = wrap_cell("hello world test", 5);
        for line in &lines {
            assert!(line.width() <= 5, "line too wide: {line:?}");
        }
        assert!(lines.len() > 1);
    }

    #[test]
    fn wrap_cell_empty() {
        assert_eq!(wrap_cell("", 10), vec![""]);
    }

    #[test]
    fn preprocess_wide_table_keeps_borders_aligned() {
        let header = "| Alpha | Beta | Gamma | Delta |";
        let sep = "| --- | --- | --- | --- |";
        let row1 = "| aaaaaaaaaa | bbbbbbbbbb | cccccccccc | dddddddddd |";
        let row2 = "| 1 | 2 | 3 | 4 |";
        let input = format!("{}\n{}\n{}\n{}", header, sep, row1, row2);

        let result = preprocess_tables(&input, 40);

        let pipe_counts: Vec<usize> = result
            .lines()
            .filter(|l| l.starts_with('│'))
            .map(|l| l.chars().filter(|&c| c == '│').count())
            .collect();

        assert!(!pipe_counts.is_empty());
        let first = pipe_counts[0];
        for count in &pipe_counts {
            assert_eq!(*count, first, "inconsistent │ count");
        }
    }
}
