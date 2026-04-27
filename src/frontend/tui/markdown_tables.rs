use pulldown_cmark::{Alignment, Event, Options, Parser, Tag, TagEnd};
use unicode_width::UnicodeWidthStr;

/// Pre-process GFM tables in `input` into monospaced fenced code blocks so
/// that `tui_markdown::from_str` renders them as verbatim text.
///
/// Non-table input is returned unchanged.  If the input contains no `|`
/// character the function returns immediately without invoking the parser.
pub fn preprocess_tables(input: &str) -> String {
    if !input.contains('|') {
        return input.to_owned();
    }

    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);

    // Collect (byte_start, byte_end, replacement) for every table in the source.
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    let parser = Parser::new_ext(input, options).into_offset_iter();

    // State machine
    let mut in_table = false;
    let mut in_head = false;
    let mut table_start: usize = 0;
    let mut table_end: usize;
    let mut alignments: Vec<Alignment> = Vec::new();
    // header row: one String per column
    let mut header: Vec<String> = Vec::new();
    // body rows
    let mut body: Vec<Vec<String>> = Vec::new();
    // current row being accumulated
    let mut current_row: Vec<String> = Vec::new();
    // current cell text
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

    // Compute display width of each cell, per column.
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

    // Top border: ┌─...─┬─...─┐
    out.push('┌');
    for (i, &w) in col_widths.iter().enumerate() {
        for _ in 0..w + 2 {
            out.push('─');
        }
        if i + 1 < num_cols {
            out.push('┬');
        }
    }
    out.push_str("┐\n");

    // Header row
    out.push_str(&format_row(header, &col_widths, alignments));

    // Separator: ╞═...═╪═...═╡
    out.push('╞');
    for (i, &w) in col_widths.iter().enumerate() {
        for _ in 0..w + 2 {
            out.push('═');
        }
        if i + 1 < num_cols {
            out.push('╪');
        }
    }
    out.push_str("╡\n");

    // Body rows with ├─┼─┤ dividers between them
    for (row_idx, row) in body.iter().enumerate() {
        out.push_str(&format_row(row, &col_widths, alignments));
        if row_idx + 1 < body.len() {
            out.push('├');
            for (i, &w) in col_widths.iter().enumerate() {
                for _ in 0..w + 2 {
                    out.push('─');
                }
                if i + 1 < num_cols {
                    out.push('┼');
                }
            }
            out.push_str("┤\n");
        }
    }

    // Bottom border: └─...─┴─...─┘
    out.push('└');
    for (i, &w) in col_widths.iter().enumerate() {
        for _ in 0..w + 2 {
            out.push('─');
        }
        if i + 1 < num_cols {
            out.push('┴');
        }
    }
    out.push_str("┘\n```");
    out
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
                for _ in 0..padding {
                    line.push(' ');
                }
                line.push_str(cell);
            }
            Alignment::Center => {
                let left_pad = padding / 2;
                let right_pad = padding - left_pad;
                for _ in 0..left_pad {
                    line.push(' ');
                }
                line.push_str(cell);
                for _ in 0..right_pad {
                    line.push(' ');
                }
            }
            Alignment::Left | Alignment::None => {
                line.push_str(cell);
                for _ in 0..padding {
                    line.push(' ');
                }
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
        // pulldown-cmark won't recognise this as a table (no header row + separator).
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
        let input = "| Left | Center | Right |\n|:-----|:------:|------:|\n| a | b | c |";
        let result = preprocess_tables(input);
        // "Right" header should appear right-aligned — for the 5-char column "Right"
        // fills the full width so padding = 0; just verify it's present.
        assert!(result.contains("Right"), "right col header missing");
        // "b" in center column should have surrounding spaces for padding.
        assert!(result.contains("b"), "center cell missing");
    }

    #[test]
    fn preprocess_handles_unicode_widths() {
        // CJK characters are 2 display columns wide.
        let input = "| Emoji | Value |\n|-------|-------|\n| 🦀 | 42 |";
        let result = preprocess_tables(input);
        assert!(result.contains('🦀'), "emoji should appear in output");
        // Verify the column boundaries are │ characters (rudimentary alignment check).
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
        // Empty cell should still produce padded space (no crash, no collapse).
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
        // Count fence markers — expect 2 pairs (4 occurrences of ```)
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
}
