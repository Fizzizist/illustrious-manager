mod tokenize;

pub use tokenize::tokenize;

use crate::tools::search::db::{SearchDb, StoredChunk};

pub fn compute_bm25_scores(
    db: &mut SearchDb,
    query: &str,
    limit: usize,
) -> std::collections::HashMap<i64, f64> {
    let mut scores = std::collections::HashMap::new();

    let sanitized = sanitize_fts_query(query);
    if sanitized.is_empty() {
        return scores;
    }

    let results = match db.fts_search(&sanitized, limit as i64) {
        Ok(r) => r,
        Err(_) => return scores,
    };

    if results.is_empty() {
        return scores;
    }

    let max_score = results.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max);

    if max_score > 0.0 {
        for (chunk_id, score) in results {
            scores.insert(chunk_id, score / max_score);
        }
    }

    scores
}

pub fn compute_path_match_scores(
    query: &str,
    chunks: &[StoredChunk],
) -> std::collections::HashMap<i64, f64> {
    let mut scores = std::collections::HashMap::new();
    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return scores;
    }

    let query_set: std::collections::HashSet<&str> =
        query_tokens.iter().map(|s| s.as_str()).collect();

    let mut path_cache: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();

    let mut max_score = 0.0_f64;

    for chunk in chunks {
        let path_tokens = path_cache
            .entry(chunk.file_path.clone())
            .or_insert_with(|| {
                tokenize(&chunk.file_path)
                    .into_iter()
                    .collect::<std::collections::HashSet<String>>()
            });

        let intersection = query_set
            .iter()
            .filter(|qt| path_tokens.contains(**qt))
            .count();

        let score = intersection as f64 / query_tokens.len() as f64;
        if score > 0.0 {
            scores.insert(chunk.id, score);
            if score > max_score {
                max_score = score;
            }
        }
    }

    if max_score > 0.0 && max_score != 1.0 {
        for score in scores.values_mut() {
            *score /= max_score;
        }
    }

    scores
}

pub fn compute_symbol_match_scores(
    query: &str,
    symbols: &[(i64, String)],
) -> std::collections::HashMap<i64, f64> {
    let mut scores = std::collections::HashMap::new();
    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return scores;
    }

    let query_lower = query.to_lowercase();
    let mut max_score = 0.0_f64;

    let mut by_chunk: std::collections::HashMap<i64, Vec<&str>> = std::collections::HashMap::new();
    for (chunk_id, name) in symbols {
        by_chunk.entry(*chunk_id).or_default().push(name);
    }

    for (chunk_id, symbol_names) in &by_chunk {
        let mut best_score = 0.0_f64;

        for symbol_name in symbol_names {
            let symbol_token_vec = tokenize(symbol_name);
            let symbol_tokens: std::collections::HashSet<&str> =
                symbol_token_vec.iter().map(|s| s.as_str()).collect();
            let symbol_lower = symbol_name.to_lowercase();

            let match_count = query_tokens
                .iter()
                .filter(|qt| symbol_tokens.contains(qt.as_str()))
                .count();

            let mut score = match_count as f64 / query_tokens.len() as f64;

            if symbol_lower.contains(&query_lower) {
                score = (score + 0.3).min(1.0);
            } else if query_lower.contains(&symbol_lower) && symbol_lower.len() >= 3 {
                score = (score + 0.2).min(1.0);
            }

            if score > best_score {
                best_score = score;
            }
        }

        if best_score > 0.0 {
            scores.insert(*chunk_id, best_score);
            if best_score > max_score {
                max_score = best_score;
            }
        }
    }

    if max_score > 0.0 && max_score != 1.0 {
        for score in scores.values_mut() {
            *score /= max_score;
        }
    }

    scores
}

pub fn compute_import_graph_scores(
    db: &mut SearchDb,
    seed_scores: &std::collections::HashMap<i64, f64>,
    file_id_to_chunk_ids: &std::collections::HashMap<i64, Vec<i64>>,
) -> std::collections::HashMap<i64, f64> {
    const PROPAGATION_FACTOR: f64 = 0.5;
    const SELF_BOOST_FACTOR: f64 = 0.25;

    let mut scores = std::collections::HashMap::new();
    if seed_scores.is_empty() {
        return scores;
    }

    let all_files = match db.get_all_files() {
        Ok(f) => f,
        Err(_) => return scores,
    };

    let mut path_to_file_id: std::collections::HashMap<String, i64> =
        std::collections::HashMap::new();

    for f in &all_files {
        path_to_file_id.insert(f.file_path.clone(), f.id);
    }

    let mut propagated: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();

    for (&file_id, &seed_score) in seed_scores {
        let prop_score = seed_score * PROPAGATION_FACTOR;
        if prop_score <= 0.0 {
            continue;
        }

        let mut connected_seed_count = 0usize;

        if let Ok(imported_paths) = db.get_imports_from(file_id) {
            for target_path in &imported_paths {
                if let Some(&target_file_id) = path_to_file_id.get(target_path) {
                    if seed_scores.contains_key(&target_file_id) {
                        connected_seed_count += 1;
                    } else {
                        *propagated.entry(target_file_id).or_insert(0.0) += prop_score;
                    }
                }
            }
        }

        let file_path = all_files
            .iter()
            .find(|f| f.id == file_id)
            .map(|f| f.file_path.clone());
        if let Some(file_path) = file_path
            && let Ok(importer_ids) = db.get_importers_of(&file_path)
        {
            for importer_id in importer_ids {
                if seed_scores.contains_key(&importer_id) {
                    connected_seed_count += 1;
                } else {
                    *propagated.entry(importer_id).or_insert(0.0) += prop_score;
                }
            }
        }

        if connected_seed_count > 0 {
            let self_boost =
                seed_score * SELF_BOOST_FACTOR * (connected_seed_count as f64 / 3.0).min(1.0);
            *propagated.entry(file_id).or_insert(0.0) += self_boost;
        }
    }

    if propagated.is_empty() {
        return scores;
    }

    let max_score = propagated.values().fold(0.0_f64, |a, &b| a.max(b));
    if max_score <= 0.0 {
        return scores;
    }

    for (file_id, file_score) in propagated {
        if let Some(chunk_ids) = file_id_to_chunk_ids.get(&file_id) {
            if chunk_ids.is_empty() {
                continue;
            }
            let per_chunk_score = file_score / max_score / chunk_ids.len() as f64;
            for &chunk_id in chunk_ids {
                scores.insert(chunk_id, per_chunk_score);
            }
        }
    }

    scores
}

pub fn compute_git_recency_scores(
    project_root: &std::path::Path,
    chunks: &[StoredChunk],
) -> std::collections::HashMap<i64, f64> {
    let mut scores = std::collections::HashMap::new();
    const NEUTRAL_SCORE: f64 = 0.5;

    if chunks.is_empty() {
        return scores;
    }

    let unique_paths: std::collections::HashSet<&str> =
        chunks.iter().map(|c| c.file_path.as_str()).collect();

    let file_timestamps = get_file_timestamps(project_root, &unique_paths);

    if file_timestamps.is_empty() {
        for chunk in chunks {
            scores.insert(chunk.id, NEUTRAL_SCORE);
        }
        return scores;
    }

    let oldest = file_timestamps
        .values()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let newest = file_timestamps
        .values()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = newest - oldest;

    for chunk in chunks {
        let ts = file_timestamps.get(chunk.file_path.as_str());
        let score = match ts {
            None => NEUTRAL_SCORE,
            Some(_) if range == 0.0 => 1.0,
            Some(&ts) => (ts - oldest) / range,
        };
        scores.insert(chunk.id, score);
    }

    scores
}

fn get_file_timestamps(
    project_root: &std::path::Path,
    target_paths: &std::collections::HashSet<&str>,
) -> std::collections::HashMap<String, f64> {
    let mut timestamps = std::collections::HashMap::new();

    let output = std::process::Command::new("git")
        .args([
            "log",
            "--max-count=10000",
            "--format=COMMIT %at",
            "--name-only",
            "--diff-filter=AMCR",
        ])
        .current_dir(project_root)
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return timestamps,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut current_timestamp = 0.0_f64;

    for line in stdout.lines() {
        if let Some(ts_str) = line.strip_prefix("COMMIT ") {
            current_timestamp = ts_str.parse().unwrap_or(0.0);
        } else if !line.trim().is_empty() && current_timestamp > 0.0 {
            let file_path = line.trim();
            if target_paths.contains(file_path) && !timestamps.contains_key(file_path) {
                timestamps.insert(file_path.to_string(), current_timestamp);
                if timestamps.len() == target_paths.len() {
                    break;
                }
            }
        }
    }

    timestamps
}

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has", "have",
    "he", "her", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not",
    "of", "on", "or", "our", "she", "so", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "up", "us", "was", "we", "what", "when", "where", "which",
    "who", "will", "with", "would", "you", "your",
];

fn sanitize_fts_query(query: &str) -> String {
    let cleaned: String = query
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect();
    let cleaned = cleaned.trim();

    let tokens: Vec<&str> = cleaned
        .split_whitespace()
        .map(|t| t.trim_start_matches('-'))
        .filter(|t| t.len() > 1 && !STOPWORDS.contains(t))
        .collect();

    if tokens.is_empty() {
        return String::new();
    }
    tokens.join(" OR ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_fts_removes_special_chars() {
        let result = sanitize_fts_query("find the search_engine");
        assert!(!result.contains("the"));
        assert!(result.contains("find"));
        assert!(result.contains("search_engine"));
    }

    #[test]
    fn sanitize_fts_joins_with_or() {
        let result = sanitize_fts_query("search engine tool");
        assert!(result.contains("OR"));
    }

    #[test]
    fn sanitize_fts_empty_query_returns_empty() {
        assert!(sanitize_fts_query("").is_empty());
    }

    #[test]
    fn sanitize_fts_stopwords_only_returns_empty() {
        assert!(sanitize_fts_query("the a an").is_empty());
    }

    #[test]
    fn path_match_scores_basic() {
        let chunks = vec![StoredChunk {
            id: 1,
            file_id: 1,
            file_path: "src/tools/search/mod.rs".to_string(),
            start_line: 1,
            end_line: 10,
            kind: "function".to_string(),
            name: Some("search".to_string()),
            content: "fn search() {}".to_string(),
            file_type: "rust".to_string(),
        }];

        let scores = compute_path_match_scores("search mod", &chunks);
        assert!(scores.contains_key(&1));
        assert!(*scores.get(&1).expect("score") > 0.0);
    }

    #[test]
    fn path_match_no_match_returns_empty() {
        let chunks = vec![StoredChunk {
            id: 1,
            file_id: 1,
            file_path: "src/main.rs".to_string(),
            start_line: 1,
            end_line: 1,
            kind: "file".to_string(),
            name: None,
            content: String::new(),
            file_type: "rust".to_string(),
        }];

        let scores = compute_path_match_scores("completely unrelated query", &chunks);
        assert!(scores.is_empty());
    }

    #[test]
    fn symbol_match_scores_basic() {
        let symbols = vec![(1_i64, "search_engine".to_string())];
        let scores = compute_symbol_match_scores("search", &symbols);
        assert!(scores.contains_key(&1));
    }

    #[test]
    fn symbol_match_exact_substring_bonus() {
        let symbols = vec![(1_i64, "search_engine".to_string())];
        let scores = compute_symbol_match_scores("search_engine", &symbols);
        assert!(scores[&1] > 0.3);
    }

    #[test]
    fn git_recency_no_git_returns_neutral() {
        let temp = tempfile::TempDir::new().expect("temp dir");
        let chunks = vec![StoredChunk {
            id: 1,
            file_id: 1,
            file_path: "test.rs".to_string(),
            start_line: 1,
            end_line: 1,
            kind: "file".to_string(),
            name: None,
            content: String::new(),
            file_type: "rust".to_string(),
        }];

        let scores = compute_git_recency_scores(temp.path(), &chunks);
        assert_eq!(scores[&1], 0.5);
    }
}
