use crate::tools::search::db::StoredChunk;
use crate::tools::search::ranker::MetricScores;

pub struct SearchResult {
    pub chunk: StoredChunk,
    pub scores: MetricScores,
    pub rank: usize,
}

pub fn format_results(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No results found.".to_string();
    }

    let mut lines = Vec::new();

    for (i, r) in results.iter().enumerate() {
        let line_range = if r.chunk.start_line == r.chunk.end_line {
            format!("L{}", r.chunk.start_line)
        } else {
            format!("L{}-{}", r.chunk.start_line, r.chunk.end_line)
        };

        let kind_label = match &r.chunk.name {
            Some(name) => format!("{} {}", r.chunk.kind, name),
            None => r.chunk.kind.clone(),
        };

        lines.push(format!(
            "{}. {}:{} ({})",
            i + 1,
            r.chunk.file_path,
            line_range,
            kind_label
        ));

        let score_pairs = [
            ("bm25", r.scores.bm25),
            ("cosine", r.scores.cosine),
            ("pathMatch", r.scores.path_match),
            ("symbolMatch", r.scores.symbol_match),
            ("importGraph", r.scores.import_graph),
            ("gitRecency", r.scores.git_recency),
        ];
        let top_scores: Vec<String> = score_pairs
            .iter()
            .filter(|(_, v)| *v > 0.01)
            .map(|(k, v)| format!("{}={:.2}", k, v))
            .collect();
        if !top_scores.is_empty() {
            lines.push(format!("   scores: {}", top_scores.join(" ")));
        }

        let content_lines: Vec<&str> = r.chunk.content.lines().collect();
        for line in content_lines.iter().take(3) {
            let trimmed = if line.len() > 120 {
                format!("{}...", &line[..117])
            } else {
                line.to_string()
            };
            lines.push(format!("   {trimmed}"));
        }
        if content_lines.len() > 3 {
            lines.push(format!("   ... ({} more lines)", content_lines.len() - 3));
        }

        if i < results.len() - 1 {
            lines.push(String::new());
        }
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(
        file_path: &str,
        start_line: i64,
        end_line: i64,
        kind: &str,
        name: Option<&str>,
        content: &str,
    ) -> SearchResult {
        SearchResult {
            chunk: StoredChunk {
                id: 1,
                file_id: 1,
                file_path: file_path.to_string(),
                start_line,
                end_line,
                kind: kind.to_string(),
                name: name.map(String::from),
                content: content.to_string(),
                file_type: "rust".to_string(),
            },
            scores: MetricScores {
                bm25: 0.8,
                cosine: 0.5,
                path_match: 0.0,
                symbol_match: 0.3,
                import_graph: 0.0,
                git_recency: 0.7,
            },
            rank: 0,
        }
    }

    #[test]
    fn empty_results_returns_no_results() {
        let output = format_results(&[]);
        assert_eq!(output, "No results found.");
    }

    #[test]
    fn single_result_formats_correctly() {
        let result = make_result(
            "src/main.rs",
            1,
            5,
            "function",
            Some("main"),
            "fn main() {}",
        );
        let output = format_results(&[result]);

        assert!(output.contains("src/main.rs:L1-5"));
        assert!(output.contains("function main"));
        assert!(output.contains("scores:"));
        assert!(output.contains("bm25="));
    }

    #[test]
    fn single_line_result_shows_l_prefix() {
        let result = make_result(
            "test.rs",
            42,
            42,
            "function",
            Some("helper"),
            "fn helper() {}",
        );
        let output = format_results(&[result]);
        assert!(output.contains("L42"));
        assert!(!output.contains("L42-"));
    }

    #[test]
    fn result_without_name_shows_kind_only() {
        let result = make_result("test.rs", 1, 10, "file", None, "some content");
        let output = format_results(&[result]);
        assert!(output.contains("(file)"));
    }

    #[test]
    fn long_content_preview_truncated() {
        let long_line: String = "x".repeat(200);
        let result = make_result("test.rs", 1, 1, "file", None, &long_line);
        let output = format_results(&[result]);
        assert!(output.contains("..."));
    }

    #[test]
    fn multi_line_content_shows_more_lines_indicator() {
        let content = "line1\nline2\nline3\nline4\nline5";
        let result = make_result("test.rs", 1, 5, "file", None, content);
        let output = format_results(&[result]);
        assert!(output.contains("2 more lines"));
    }

    #[test]
    fn zero_scores_are_omitted() {
        let mut result = make_result("test.rs", 1, 1, "file", None, "code");
        result.scores.path_match = 0.0;
        result.scores.import_graph = 0.0;
        let output = format_results(&[result]);
        assert!(!output.contains("pathMatch="));
        assert!(!output.contains("importGraph="));
    }
}
