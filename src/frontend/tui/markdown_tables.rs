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

    let mut allocated: Vec<usize> = natural
        .iter()
        .map(|&w| {
            let shrunk = w.saturating_mul(budget_for_content) / natural_sum;
            shrunk.max(MIN_COL_WIDTH)
        })
        .collect();

    let mut current_sum: usize = allocated.iter().sum();
    while current_sum > budget_for_content {
        let mut trimmed = false;
        let mut order: Vec<usize> = (0..num_cols).collect();
        order.sort_by(|&a, &b| allocated[b].cmp(&allocated[a]));
        for &i in &order {
            if allocated[i] > MIN_COL_WIDTH {
                allocated[i] -= 1;
                current_sum -= 1;
                trimmed = true;
                if current_sum <= budget_for_content {
                    break;
                }
            }
        }
        if !trimmed {
            break;
        }
    }

    allocated
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
        // Wide header forces shrink/wrap; short body cells fit per wrap line so
        // we can assert they survive intact.
        let header_wide =
            "| HeaderColumnOne | HeaderColumnTwo | HeaderColumnThree | HeaderColumnFour |";
        let sep = "| --- | --- | --- | --- |";
        let row1 = "| aaa1xx | bbb1xx | ccc1xx | ddd1xx |";
        let row2 = "| aaa2 | bbb2 | ccc2 | ddd2 |";
        let input = format!("{}\n{}\n{}\n{}", header_wide, sep, row1, row2);

        let result = preprocess_tables(&input, 40);

        for line in result.lines() {
            let line_width = line.width();
            assert!(
                line_width <= 40 || line == "```",
                "line too wide ({line_width}): {line:?}"
            );
        }

        for needle in &[
            "aaa1xx", "bbb1xx", "ccc1xx", "ddd1xx", "aaa2", "bbb2", "ccc2", "ddd2",
        ] {
            assert!(
                result.contains(needle),
                "missing cell value {needle:?} in:\n{result}"
            );
        }
    }

    #[test]
    fn preprocess_pathological_narrow_does_not_panic() {
        // Acceptance criterion 6: 8-column table at width 30 must not panic;
        // output must still be a valid fenced code block (overflow fallback).
        let header = "| A | B | C | D | E | F | G | H |";
        let sep = "|---|---|---|---|---|---|---|---|";
        let row = "| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |";
        let input = format!("{}\n{}\n{}", header, sep, row);

        let result = preprocess_tables(&input, 30);

        assert!(result.contains("```"), "should still be fenced: {result}");
        assert!(result.contains('│'), "should still have borders: {result}");
        // All cell contents survive (overflow is the documented fallback).
        for needle in &["A", "B", "H", "1", "8"] {
            assert!(result.contains(needle), "missing {needle:?}: {result}");
        }
    }

    #[test]
    fn preprocess_right_alignment_with_wrapping() {
        // Wide right-aligned cell forced to wrap — every non-empty sub-line
        // in the right column must be flush right (no trailing spaces between
        // content and the right border padding).
        let input =
            "| Left | Right |\n|:-----|------:|\n| short | a longer value that should wrap |";
        let result = preprocess_tables(input, 30);

        // Only inspect lines below the header separator (after ╞) — those are
        // body rows containing the wrapped right-aligned cell.
        let mut in_body = false;
        let mut checked_at_least_one = false;
        for line in result.lines() {
            if line.starts_with('╞') {
                in_body = true;
                continue;
            }
            if !in_body || !line.starts_with('│') {
                continue;
            }
            let segments: Vec<&str> = line.split('│').collect();
            if segments.len() != 4 {
                continue;
            }
            let right_seg = segments[2];
            // Strip the outer single-space pads.
            let inner = right_seg
                .strip_prefix(' ')
                .and_then(|s| s.strip_suffix(' '))
                .unwrap_or(right_seg);
            if inner.trim().is_empty() {
                continue;
            }
            checked_at_least_one = true;
            // Right-aligned: content flush right, so inner does not end with space.
            assert!(
                !inner.ends_with(' '),
                "right-aligned content not flush right: {line:?}"
            );
        }
        assert!(
            checked_at_least_one,
            "expected at least one wrapped right-aligned body sub-line in: {result}"
        );
    }

    #[test]
    fn preprocess_center_alignment_with_wrapping() {
        // Wide center-aligned cell forced to wrap — sub-lines should be
        // center-padded (left and right pad differ by at most 1).
        let input =
            "| Left | Center |\n|:-----|:------:|\n| short | a longer value that should wrap |";
        let result = preprocess_tables(input, 30);

        let mut in_body = false;
        let mut checked = false;
        for line in result.lines() {
            if line.starts_with('╞') {
                in_body = true;
                continue;
            }
            if !in_body || !line.starts_with('│') {
                continue;
            }
            let segments: Vec<&str> = line.split('│').collect();
            if segments.len() != 4 {
                continue;
            }
            let center_seg = segments[2];
            let inner = center_seg
                .strip_prefix(' ')
                .and_then(|s| s.strip_suffix(' '))
                .unwrap_or(center_seg);
            if inner.trim().is_empty() {
                continue;
            }
            checked = true;
            let leading = inner.len() - inner.trim_start().len();
            let trailing = inner.len() - inner.trim_end().len();
            let diff = leading.abs_diff(trailing);
            assert!(
                diff <= 1,
                "center-aligned padding asymmetric (leading={leading}, trailing={trailing}): {line:?}"
            );
        }
        assert!(
            checked,
            "expected at least one wrapped center-aligned body sub-line in: {result}"
        );
    }

    #[test]
    fn allocate_widths_proportional_shrink() {
        // Equal natural widths should produce equal allocated widths.
        let result = allocate_widths(&[20, 20, 20], 40);
        let border_overhead = 3 * 3 + 1;
        let budget_for_content = 40usize.saturating_sub(border_overhead);
        let sum: usize = result.iter().sum();
        assert!(
            sum <= budget_for_content,
            "sum {sum} exceeds content budget {budget_for_content}"
        );
        for &w in &result {
            assert!(w >= MIN_COL_WIDTH);
        }
        // Equal naturals → equal allocations.
        assert_eq!(result[0], result[1]);
        assert_eq!(result[1], result[2]);
    }

    #[test]
    fn allocate_widths_proportional_shrink_unequal_naturals() {
        // Unequal naturals should produce unequal allocations roughly proportional to inputs.
        let natural = vec![10, 30, 60]; // ratios 1:3:6
        let result = allocate_widths(&natural, 50);
        let border_overhead = 3 * 3 + 1;
        let budget_for_content = 50usize.saturating_sub(border_overhead);
        let sum: usize = result.iter().sum();
        assert!(
            sum <= budget_for_content,
            "sum {sum} exceeds content budget {budget_for_content}"
        );
        // Largest natural still gets largest allocation.
        assert!(result[2] >= result[1], "{result:?}");
        assert!(result[1] >= result[0], "{result:?}");
        // Distinct allocations (not collapsed to all-MIN).
        assert!(result[2] > result[0], "{result:?}");
        for &w in &result {
            assert!(w >= MIN_COL_WIDTH);
        }
    }

    #[test]
    fn allocate_widths_skewed_naturals_does_not_overshoot_budget() {
        // Regression: previously, clamping small columns up to MIN_COL_WIDTH
        // produced a sum > budget_for_content because surplus was not
        // redistributed.  See review finding 1.
        let natural = vec![1, 1, 100, 1, 1];
        let budget = 40;
        let result = allocate_widths(&natural, budget);
        let border_overhead = 5 * 3 + 1;
        let budget_for_content = budget.saturating_sub(border_overhead);
        let sum: usize = result.iter().sum();
        assert!(
            sum <= budget_for_content,
            "skewed allocation sum {sum} exceeds content budget {budget_for_content}: {result:?}"
        );
        for &w in &result {
            assert!(w >= MIN_COL_WIDTH, "column below floor: {result:?}");
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
    fn wrap_cell_wide_grapheme_at_boundary_fits() {
        // 🦀 is width 2.  "ab" (width 2) + 🦀 (width 2) = 4 fits at width 4.
        // Then "cd" wraps to next line.
        let lines = wrap_cell("ab🦀cd", 4);
        assert_eq!(lines, vec!["ab🦀".to_owned(), "cd".to_owned()]);
        for line in &lines {
            assert!(line.width() <= 4, "line too wide: {line:?}");
        }
    }

    #[test]
    fn wrap_cell_wide_grapheme_exceeds_remaining_wraps() {
        // "ab" (width 2) + 🦀 (width 2) = 4 > width 3, so 🦀 starts a new line.
        let lines = wrap_cell("ab🦀", 3);
        assert_eq!(lines, vec!["ab".to_owned(), "🦀".to_owned()]);
        for line in &lines {
            assert!(line.width() <= 3, "line too wide: {line:?}");
        }
    }

    #[test]
    fn wrap_cell_cjk_at_boundary() {
        // 言 and 語 are width 2 each.  Width 4 fits two CJK chars per line.
        let lines = wrap_cell("言語言語", 4);
        assert_eq!(lines, vec!["言語".to_owned(), "言語".to_owned()]);
        for line in &lines {
            assert!(line.width() <= 4, "line too wide: {line:?}");
        }
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
