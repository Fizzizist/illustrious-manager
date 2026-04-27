use pulldown_cmark::{Alignment, Event, Options, Parser, Tag, TagEnd};
use unicode_width::UnicodeWidthStr;

/// Pre-process GFM tables in `input` into monospaced fenced code blocks so
/// that `tui_markdown::from_str` renders them as verbatim text.
///
/// Non-table input is returned unchanged.  If the input contains no `|`
/// character the function returns immediately without invoking the parser.
///
/// Wide tables (wider than the terminal viewport) overflow into the code
/// block's wrap behavior — horizontal scroll is out of scope for this
/// implementation.
pub fn preprocess_tables(input: &str) -> String {
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
                let rendered = render_table(&alignments, &header, &body);
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

    // Apply replacements in reverse order so byte offsets remain valid.
    let mut output = input.to_owned();
    for (start, end, replacement) in replacements.into_iter().rev() {
        output.replace_range(start..end, &replacement);
    }
    output
}

fn render_table(alignments: &[Alignment], header: &[String], body: &[Vec<String>]) -> String {
    let num_cols = header.len().max(1);

    let col_widths: Vec<usize> = (0..num_cols)
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

    let mut out = String::new();
    out.push_str("```\n");
    out.push_str(&border_line('┌', '─', '┬', '┐', &col_widths));
    out.push_str(&format_row(header, &col_widths, alignments));
    out.push_str(&border_line('╞', '═', '╪', '╡', &col_widths));

    for (row_idx, row) in body.iter().enumerate() {
        out.push_str(&format_row(row, &col_widths, alignments));
        if row_idx + 1 < body.len() {
            out.push_str(&border_line('├', '─', '┼', '┤', &col_widths));
        }
    }

    out.push_str(&border_line('└', '─', '┴', '┘', &col_widths));
    out.push_str("```");
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

fn format_row(cells: &[String], col_widths: &[usize], alignments: &[Alignment]) -> String {
    let mut line = String::from("│");
    for (col, &w) in col_widths.iter().enumerate() {
        let cell = cells.get(col).map(String::as_str).unwrap_or("");
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
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_passes_through_non_table_input() {
        let input = "Hello world\nNo pipes here.";
        assert_eq!(preprocess_tables(input), input);
    }

    #[test]
    fn preprocess_passes_through_pipes_outside_tables() {
        let input = "Use `a | b` for pipes in code. Also some | prose.";
        let result = preprocess_tables(input);
        assert_eq!(result, input);
    }

    #[test]
    fn preprocess_formats_basic_table() {
        let input = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |";
        let result = preprocess_tables(input);
        assert!(result.starts_with("```\n"), "should be fenced: {result}");
        assert!(result.ends_with("```"), "should end with fence: {result}");
        assert!(result.contains("Name"), "header cell missing: {result}");
        assert!(result.contains("Alice"), "body cell missing: {result}");
        assert!(result.contains("Bob"), "body cell missing: {result}");
    }

    #[test]
    fn preprocess_respects_column_alignment() {
        // Left=5 chars, Center=6 chars, Right=5 chars headers
        // Body: "a"(1), "b"(1), "c"(1) — all narrower than headers, so padding applies
        let input = "| Left | Center | Right |\n|:-----|:------:|------:|\n| a | b | c |";
        let result = preprocess_tables(input);

        // Right-aligned "c" in a 5-char column → 4 spaces before "c"
        assert!(
            result.contains("│     c │"),
            "right-aligned 'c' should have 4 leading spaces: {result}"
        );
        // Center-aligned "b" in a 6-char column → 2 spaces left, 3 spaces right
        assert!(
            result.contains("│   b    │"),
            "center-aligned 'b' should have symmetric padding: {result}"
        );
        // Left-aligned "a" in a 4-char column → 3 trailing spaces
        assert!(
            result.contains("│ a    │"),
            "left-aligned 'a' should have trailing spaces: {result}"
        );
    }

    #[test]
    fn preprocess_handles_unicode_widths() {
        let input = "| Emoji | Value |\n|-------|-------|\n| 🦀 | 42 |";
        let result = preprocess_tables(input);
        assert!(result.contains('🦀'), "emoji should appear in output");
        for line in result.lines() {
            if line.starts_with('│') {
                let pipe_count = line.chars().filter(|&c| c == '│').count();
                assert_eq!(pipe_count, 3, "each data row should have 3 │ chars: {line}");
            }
        }
    }

    #[test]
    fn preprocess_handles_empty_cells() {
        let input = "| A | B |\n|---|---|\n| | filled |";
        let result = preprocess_tables(input);
        assert!(result.contains("filled"), "non-empty cell missing");
        assert!(result.contains('│'), "borders should be present");
    }

    #[test]
    fn preprocess_handles_multiple_tables() {
        let table1 = "| X | Y |\n|---|---|\n| 1 | 2 |";
        let table2 = "| P | Q |\n|---|---|\n| 3 | 4 |";
        let input = format!("{table1}\n\nSome text\n\n{table2}");
        let result = preprocess_tables(&input);
        assert!(
            result.contains("Some text"),
            "prose between tables should survive"
        );
        let fence_count = result.matches("```").count();
        assert_eq!(
            fence_count, 4,
            "expected 4 fence markers for 2 tables, got: {result}"
        );
    }

    #[test]
    fn preprocess_preserves_surrounding_markdown() {
        let input = "# Heading\n\n| Col |\n|-----|\n| val |\n\nParagraph after.";
        let result = preprocess_tables(input);
        assert!(result.contains("# Heading"), "heading should be preserved");
        assert!(
            result.contains("Paragraph after."),
            "paragraph should be preserved"
        );
        assert!(result.contains("val"), "table cell should be present");
    }

    #[test]
    fn preprocess_strips_inline_markup_preserving_text() {
        // Bold, italic, and links lose markup in a monospaced grid; inner text is kept.
        let input = "| **Bold** | [link](http://x.com) |\n|-----------|----------------------|\n| *italic* | plain |";
        let result = preprocess_tables(input);
        assert!(
            result.contains("Bold"),
            "bold text content should be present"
        );
        assert!(result.contains("link"), "link text should be present");
        assert!(result.contains("italic"), "italic text should be present");
        assert!(
            !result.contains("**"),
            "asterisks should be stripped from cell"
        );
        assert!(
            !result.contains("http://x.com"),
            "URL should not appear in cell"
        );
    }

    #[test]
    fn preprocess_inline_code_in_cell_preserves_backticks() {
        let input = "| Command | Result |\n|---------|--------|\n| `ls -la` | ok |";
        let result = preprocess_tables(input);
        assert!(
            result.contains("`ls -la`"),
            "inline code should keep backticks: {result}"
        );
    }

    #[test]
    fn preprocess_header_only_table_renders_without_body() {
        // Valid GFM with no body rows — produces top border, header, separator, bottom border.
        let input = "| A | B |\n|---|---|";
        let result = preprocess_tables(input);
        assert!(result.starts_with("```\n"), "should be fenced");
        assert!(result.contains("A"), "header cell A missing");
        assert!(result.contains("B"), "header cell B missing");
        // No body means no mid-row dividers — just top/header/separator/bottom.
        assert!(
            !result.contains('├'),
            "no mid-divider expected for header-only table"
        );
    }

    #[test]
    fn preprocess_single_column_table() {
        let input = "| Name |\n|------|\n| Alice |\n| Bob |";
        let result = preprocess_tables(input);
        assert!(result.contains("Alice"), "Alice missing");
        assert!(result.contains("Bob"), "Bob missing");
        // Single-column borders have no joiners.
        assert!(
            !result.contains('┬'),
            "no ┬ joiner expected for single-column table"
        );
        assert!(
            !result.contains('┴'),
            "no ┴ joiner expected for single-column table"
        );
        // Outer │ borders are present (one per side).
        for line in result.lines() {
            if line.starts_with('│') {
                let pipe_count = line.chars().filter(|&c| c == '│').count();
                assert_eq!(
                    pipe_count, 2,
                    "single-col row should have exactly 2 │: {line}"
                );
            }
        }
    }

    #[test]
    fn preprocess_body_row_with_fewer_columns_than_header_pads_with_blanks() {
        // Body row has only 1 cell; header has 2. Missing cell should render as blank.
        let input = "| A | B |\n|---|---|\n| only_a |";
        let result = preprocess_tables(input);
        assert!(result.contains("only_a"), "first cell should be present");
        // The second column should still have a │ boundary (blank padded cell).
        for line in result.lines() {
            if line.contains("only_a") {
                let pipe_count = line.chars().filter(|&c| c == '│').count();
                assert_eq!(
                    pipe_count, 3,
                    "row with missing col should still have 3 │ chars: {line}"
                );
            }
        }
    }

    #[test]
    fn preprocess_body_row_with_more_columns_than_header_truncates_extras() {
        // Body row has 3 cells; header has 2. Extra cell is silently dropped.
        let input = "| A | B |\n|---|---|\n| x | y | z_extra |";
        let result = preprocess_tables(input);
        assert!(result.contains('x'), "first body cell present");
        assert!(result.contains('y'), "second body cell present");
        assert!(
            !result.contains("z_extra"),
            "extra cell should be truncated: {result}"
        );
    }

    #[test]
    fn preprocess_header_only_partial_input_does_not_render_as_table() {
        // Incomplete separator means pulldown-cmark does not emit a Table event.
        let input = "| Name | Age |\n|---";
        let result = preprocess_tables(input);
        assert_eq!(result, input, "partial input should pass through unchanged");
    }
}
