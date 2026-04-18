use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};

use super::chunker;
use super::db::{SearchDb, StoredChunk};
use super::embedder::Embedder;
use super::format::{SearchResult, format_results};
use super::metrics;
use super::query_classifier::classify_query;
use super::ranker::{MetricScores, poem_rank};
use super::scanner;
use super::vector_store;

const MODEL_NAME: &str = "Xenova/all-MiniLM-L6-v2";
const METRIC_CANDIDATE_LIMIT: usize = 1000;

#[derive(Clone)]
pub struct SearchEngine {
    project_root: PathBuf,
    embedder_cache_dir: PathBuf,
}

impl SearchEngine {
    pub fn new(project_root: PathBuf) -> Self {
        let embedder_cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("illustrious-manager")
            .join("models");
        Self {
            project_root,
            embedder_cache_dir,
        }
    }

    fn get_embedder(&self) -> Embedder {
        Embedder::new(self.embedder_cache_dir.clone())
    }

    fn ensure_embedder(embedder: &mut Embedder) -> bool {
        if embedder.initialize().is_err() {
            return false;
        }
        true
    }

    pub fn search(
        &self,
        query: &str,
        limit: usize,
        restrict_to_dir: Option<&str>,
    ) -> Result<String> {
        let index_dir = self.project_root.join(".search-index");
        std::fs::create_dir_all(&index_dir)
            .with_context(|| format!("Creating index directory: {}", index_dir.display()))?;

        let db_path = index_dir.join("search.db");
        let mut db = SearchDb::open(&db_path)?;

        let mut embedder = self.get_embedder();

        self.build_index(&mut db, &mut embedder)?;

        let mut all_chunks = db.get_all_chunks()?;
        if let Some(dir) = restrict_to_dir {
            all_chunks.retain(|c| c.file_path.starts_with(dir));
        }

        if all_chunks.is_empty() {
            return Ok(format_results(&[]));
        }

        let query_type = classify_query(query);

        let bm25_scores = metrics::compute_bm25_scores(&mut db, query, METRIC_CANDIDATE_LIMIT);

        let cosine_scores =
            self.compute_vector_scores(&mut db, &mut embedder, query, METRIC_CANDIDATE_LIMIT)?;

        let path_scores = metrics::compute_path_match_scores(query, &all_chunks);

        let symbols = db.get_all_symbols()?;
        let symbol_scores = metrics::compute_symbol_match_scores(query, &symbols);

        let file_seed_scores =
            self.aggregate_file_scores(&all_chunks, &bm25_scores, &cosine_scores);
        let seed_threshold = self.compute_seed_threshold(&file_seed_scores);
        let filtered_seeds: HashMap<i64, f64> = file_seed_scores
            .into_iter()
            .filter(|(_, score)| *score >= seed_threshold)
            .collect();
        let file_id_to_chunk_ids = self.build_file_chunk_map(&all_chunks);
        let import_scores =
            metrics::compute_import_graph_scores(&mut db, &filtered_seeds, &file_id_to_chunk_ids);

        let recency_scores = metrics::compute_git_recency_scores(&self.project_root, &all_chunks);

        let candidate_ids = self.collect_candidate_ids(
            &bm25_scores,
            &cosine_scores,
            &path_scores,
            &symbol_scores,
            &import_scores,
            &recency_scores,
        );

        let mut candidates: HashMap<i64, MetricScores> = HashMap::new();
        for &id in &candidate_ids {
            candidates.insert(
                id,
                MetricScores {
                    bm25: bm25_scores.get(&id).copied().unwrap_or(0.0),
                    cosine: cosine_scores.get(&id).copied().unwrap_or(0.0),
                    path_match: path_scores.get(&id).copied().unwrap_or(0.0),
                    symbol_match: symbol_scores.get(&id).copied().unwrap_or(0.0),
                    import_graph: import_scores.get(&id).copied().unwrap_or(0.0),
                    git_recency: recency_scores.get(&id).copied().unwrap_or(0.0),
                },
            );
        }

        if candidates.is_empty() {
            return Ok(format_results(&[]));
        }

        let ranked = poem_rank(&candidates, &query_type, METRIC_CANDIDATE_LIMIT);

        let chunk_map: HashMap<i64, &StoredChunk> = all_chunks.iter().map(|c| (c.id, c)).collect();

        let results: Vec<SearchResult> = ranked
            .into_iter()
            .take(limit)
            .filter_map(|candidate| {
                let chunk = chunk_map.get(&candidate.id)?;
                Some(SearchResult {
                    chunk: (*chunk).clone(),
                    scores: candidate.scores,
                    rank: candidate.rank,
                })
            })
            .collect();

        Ok(format_results(&results))
    }

    fn build_index(&self, db: &mut SearchDb, embedder: &mut Embedder) -> Result<()> {
        let scanned_files = scanner::scan_project(&self.project_root);
        let existing_files = db.get_all_files()?;

        let existing_by_path: HashMap<String, _> = existing_files
            .iter()
            .map(|f| (f.file_path.clone(), f))
            .collect();

        let scanned_by_path: HashMap<String, _> = scanned_files
            .iter()
            .map(|f| (f.file_path.clone(), f))
            .collect();

        let mut to_add = Vec::new();
        let mut to_update = Vec::new();
        let mut to_remove = Vec::new();

        for scanned in &scanned_files {
            if let Some(existing) = existing_by_path.get(&scanned.file_path) {
                if (existing.mtime - scanned.mtime).abs() > 0.001 {
                    to_update.push(scanned);
                }
            } else {
                to_add.push(scanned);
            }
        }

        for existing in &existing_files {
            if !scanned_by_path.contains_key(&existing.file_path) {
                to_remove.push(existing);
            }
        }

        if to_add.is_empty() && to_update.is_empty() && to_remove.is_empty() {
            return Ok(());
        }

        for file in &to_remove {
            db.delete_file(file.id)?;
        }

        let files_to_process: Vec<_> = to_add
            .into_iter()
            .chain(to_update.iter().copied())
            .collect();
        let mut all_new_chunk_ids: Vec<i64> = Vec::new();

        for scanned in &files_to_process {
            let abs_path = self.project_root.join(&scanned.file_path);
            let content = match std::fs::read_to_string(&abs_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let chunks = chunker::chunk_file(&content, &scanned.file_path, &scanned.file_type);

            let file_id = db.upsert_file(
                &scanned.file_path,
                scanned.mtime,
                &scanned.file_type.to_string(),
            )?;

            if let Some(existing) = existing_by_path.get(&scanned.file_path) {
                db.delete_chunks_for_file(existing.id)?;
                let _ = db.delete_imports_for_file(existing.id);
            }

            for text_chunk in &chunks {
                let chunk_id = db.insert_chunk(
                    file_id,
                    &text_chunk.file_path,
                    text_chunk.start_line as i64,
                    text_chunk.end_line as i64,
                    &text_chunk.kind.to_string(),
                    text_chunk.name.as_deref(),
                    &text_chunk.content,
                    &scanned.file_type.to_string(),
                )?;
                all_new_chunk_ids.push(chunk_id);

                if let Some(name) = &text_chunk.name {
                    db.insert_symbol(chunk_id, name, &text_chunk.kind.to_string())?;
                }
            }

            let imports = extract_imports(&content, &scanned.file_type);
            for target_path in imports {
                let _ = db.insert_import(file_id, &target_path);
            }
        }

        if !all_new_chunk_ids.is_empty() {
            let _ = self.embed_chunks(db, embedder, &all_new_chunk_ids);
        }

        Ok(())
    }

    fn embed_chunks(
        &self,
        db: &mut SearchDb,
        embedder: &mut Embedder,
        chunk_ids: &[i64],
    ) -> Result<()> {
        if chunk_ids.is_empty() {
            return Ok(());
        }

        let chunks = db.get_chunks_by_ids(chunk_ids)?;
        if chunks.is_empty() {
            return Ok(());
        }

        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();

        if !Self::ensure_embedder(embedder) {
            return Ok(());
        }

        let vectors = embedder.embed(&texts)?;

        let items: Vec<(i64, String, Vec<u8>)> = chunks
            .iter()
            .zip(vectors.iter())
            .map(|(chunk, vector)| {
                let blob = vector_store::pack_vector(vector);
                (chunk.id, MODEL_NAME.to_string(), blob)
            })
            .collect();

        db.batch_upsert_embeddings(&items)?;
        Ok(())
    }

    fn compute_vector_scores(
        &self,
        db: &mut SearchDb,
        embedder: &mut Embedder,
        query: &str,
        limit: usize,
    ) -> Result<HashMap<i64, f64>> {
        if !Self::ensure_embedder(embedder) {
            return Ok(HashMap::new());
        }

        let query_vectors = embedder.embed(&[query])?;
        let query_vector = &query_vectors[0];

        let stored = db.get_all_embeddings(MODEL_NAME)?;
        if stored.is_empty() {
            return Ok(HashMap::new());
        }

        let vectors: Vec<(i64, Vec<f32>)> = stored
            .into_iter()
            .map(|(id, blob)| (id, vector_store::unpack_vector(&blob)))
            .collect();

        let top_k = vector_store::top_k_similar(query_vector, &vectors, limit);

        let scores: HashMap<i64, f64> = top_k
            .into_iter()
            .map(|(id, score)| (id, score.max(0.0) as f64))
            .collect();

        Ok(scores)
    }

    fn collect_candidate_ids(
        &self,
        bm25: &HashMap<i64, f64>,
        cosine: &HashMap<i64, f64>,
        path: &HashMap<i64, f64>,
        symbol: &HashMap<i64, f64>,
        import: &HashMap<i64, f64>,
        recency: &HashMap<i64, f64>,
    ) -> Vec<i64> {
        let mut ids = std::collections::HashSet::new();
        for map in &[bm25, cosine, path, symbol, import, recency] {
            for &id in map.keys() {
                ids.insert(id);
            }
        }
        ids.into_iter().collect()
    }

    fn aggregate_file_scores(
        &self,
        chunks: &[StoredChunk],
        bm25: &HashMap<i64, f64>,
        cosine: &HashMap<i64, f64>,
    ) -> HashMap<i64, f64> {
        let mut file_scores = HashMap::new();
        for chunk in chunks {
            let max_score = bm25
                .get(&chunk.id)
                .copied()
                .unwrap_or(0.0)
                .max(cosine.get(&chunk.id).copied().unwrap_or(0.0));
            if max_score > 0.0 {
                let entry = file_scores.entry(chunk.file_id).or_insert(0.0);
                if max_score > *entry {
                    *entry = max_score;
                }
            }
        }
        file_scores
    }

    fn compute_seed_threshold(&self, file_scores: &HashMap<i64, f64>) -> f64 {
        if file_scores.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = file_scores.values().copied().collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).expect("floats"));
        let median = sorted[sorted.len() / 2];
        median.max(0.1)
    }

    fn build_file_chunk_map(&self, chunks: &[StoredChunk]) -> HashMap<i64, Vec<i64>> {
        let mut map = HashMap::new();
        for chunk in chunks {
            map.entry(chunk.file_id)
                .or_insert_with(Vec::new)
                .push(chunk.id);
        }
        map
    }
}

fn extract_imports(content: &str, file_type: &super::scanner::FileType) -> Vec<String> {
    match file_type {
        super::scanner::FileType::Rust => extract_rust_imports(content),
        _ => Vec::new(),
    }
}

fn extract_rust_imports(content: &str) -> Vec<String> {
    let mut imports = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("use ") {
            continue;
        }
        let path = trimmed
            .strip_prefix("use ")
            .unwrap_or("")
            .trim()
            .trim_end_matches(';')
            .trim();

        // Handle grouped imports: use foo::{bar, baz}
        if let Some(brace_start) = path.find("::{") {
            let base = &path[..brace_start];
            let inner = &path[brace_start + 3..path.len().saturating_sub(1)];
            let base_normalized = base.replace("::", "/");
            for item in inner.split(',') {
                let full = format!("{}/{}", base_normalized, item.trim());
                let resolved = resolve_rust_path(&full);
                imports.push(resolved);
            }
        } else {
            let resolved = resolve_rust_path(&path.replace("::", "/"));
            imports.push(resolved);
        }
    }
    imports
}

fn resolve_rust_path(crate_path: &str) -> String {
    let parts: Vec<&str> = crate_path.split('/').collect();
    if parts.len() < 2 {
        return crate_path.to_string();
    }

    match parts[0] {
        "crate" => {
            let rest = &parts[1..];
            if rest.is_empty() {
                return crate_path.to_string();
            }
            match rest[0] {
                "super" | "self" => rest[1..].join("/"),
                _ => rest.join("/"),
            }
        }
        "super" | "self" => parts[1..].join("/"),
        "std" => format!("lib/std/{}", parts[1..].join("/")),
        "core" => format!("lib/core/{}", parts[1..].join("/")),
        "alloc" => format!("lib/alloc/{}", parts[1..].join("/")),
        _ => parts.join("/"),
    }
}

#[cfg(feature = "ts-typescript")]
fn extract_ts_imports(content: &str) -> Vec<String> {
    let mut imports = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        let import_path = if let Some(rest) = trimmed
            .strip_prefix("import ")
            .and_then(|s| s.strip_prefix("type "))
            .or_else(|| trimmed.strip_prefix("import "))
        {
            rest.split("from").last().map(|s| {
                s.trim()
                    .trim_end_matches(';')
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
            })
        } else if let Some(rest) = trimmed.strip_prefix("from ") {
            Some(
                rest.split("import")
                    .next()
                    .unwrap_or("")
                    .trim()
                    .trim_end_matches(';')
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\''),
            )
        } else if trimmed.starts_with("require(") {
            Some(
                trimmed
                    .trim_start_matches("require(")
                    .trim_end_matches(')')
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\''),
            )
        } else {
            None
        };

        if let Some(path) = import_path {
            if !path.is_empty() {
                imports.push(path.to_string());
            }
        }
    }
    imports
}

#[cfg(feature = "ts-python")]
fn extract_python_imports(content: &str) -> Vec<String> {
    let mut imports = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
            let module = if let Some(rest) = trimmed.strip_prefix("from ") {
                rest.split(" import").next().unwrap_or("")
            } else {
                trimmed.strip_prefix("import ").unwrap_or("")
            };
            let module = module.trim().split(" as ").next().unwrap_or(module).trim();
            if !module.is_empty() {
                imports.push(module.replace('.', "/"));
            }
        }
    }
    imports
}

#[cfg(feature = "ts-go")]
fn extract_go_imports(content: &str) -> Vec<String> {
    let mut imports = Vec::new();
    let mut in_import_block = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "import (" {
            in_import_block = true;
            continue;
        }
        if in_import_block && trimmed == ")" {
            in_import_block = false;
            continue;
        }
        if in_import_block {
            let path = trimmed
                .trim_matches('"')
                .split("//")
                .next()
                .unwrap_or("")
                .trim();
            if !path.is_empty() {
                imports.push(path.to_string());
            }
        } else if let Some(rest) = trimmed.strip_prefix("import ") {
            let path = rest
                .trim()
                .trim_matches('"')
                .split("//")
                .next()
                .unwrap_or("")
                .trim();
            if !path.is_empty() {
                imports.push(path.to_string());
            }
        }
    }
    imports
}

#[cfg(feature = "ts-java")]
fn extract_java_imports(content: &str) -> Vec<String> {
    let mut imports = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("import ") {
            let path = trimmed
                .strip_prefix("import ")
                .unwrap_or("")
                .trim()
                .trim_end_matches(';')
                .trim();
            if !path.is_empty() {
                imports.push(path.replace('.', "/"));
            }
        }
    }
    imports
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn search_empty_directory_returns_no_results() {
        let temp = TempDir::new().expect("temp dir");
        let engine = SearchEngine::new(temp.path().to_path_buf());
        let result = engine.search("test", 20, None).expect("search");
        assert_eq!(result, "No results found.");
    }

    #[test]
    fn search_finds_files_in_project() {
        let temp = TempDir::new().expect("temp dir");
        fs::write(
            temp.path().join("main.rs"),
            "fn search_engine() -> Vec<String> {\n    vec![\"hello\".to_string()]\n}\n",
        )
        .expect("write");

        let engine = SearchEngine::new(temp.path().to_path_buf());
        let result = engine.search("search_engine", 20, None).expect("search");
        assert_ne!(result, "No results found.");
        assert!(result.contains("main.rs"));
    }

    #[test]
    fn search_with_restrict_to_dir() {
        let temp = TempDir::new().expect("temp dir");
        let sub = temp.path().join("src");
        fs::create_dir_all(&sub).expect("dir");
        fs::write(sub.join("mod.rs"), "fn helper() {}").expect("write");
        fs::write(temp.path().join("main.rs"), "fn main() {}").expect("write");

        let engine = SearchEngine::new(temp.path().to_path_buf());
        let result = engine.search("main", 20, Some("src")).expect("search");
        assert!(!result.contains("main.rs") || result.contains("No results"));
    }

    #[test]
    fn extract_rust_imports_simple() {
        let code = "use std::collections::HashMap;\nuse crate::tools::search::db;\n";
        let imports = extract_rust_imports(code);
        assert!(imports.contains(&"lib/std/collections/HashMap".to_string()));
        assert!(imports.contains(&"tools/search/db".to_string()));
    }

    #[test]
    fn extract_rust_imports_grouped() {
        let code = "use std::collections::{HashMap, BTreeMap};\n";
        let imports = extract_rust_imports(code);
        assert!(imports.contains(&"lib/std/collections/HashMap".to_string()));
        assert!(imports.contains(&"lib/std/collections/BTreeMap".to_string()));
    }

    #[test]
    fn extract_rust_imports_ignores_non_use_lines() {
        let code = "fn main() {}\n// use something\nconst X: i32 = 1;\n";
        let imports = extract_rust_imports(code);
        assert!(imports.is_empty());
    }

    #[test]
    fn extract_rust_imports_super_path() {
        let code = "use super::engine::SearchEngine;\n";
        let imports = extract_rust_imports(code);
        assert!(imports.contains(&"engine/SearchEngine".to_string()));
    }

    #[test]
    fn extract_rust_imports_crate_self() {
        let code = "use crate::config::SearchConfig;\n";
        let imports = extract_rust_imports(code);
        assert!(imports.contains(&"config/SearchConfig".to_string()));
    }

    #[test]
    fn extract_rust_imports_empty() {
        let imports = extract_rust_imports("");
        assert!(imports.is_empty());
    }

    #[test]
    fn rebuild_creates_fresh_index() {
        let temp = TempDir::new().expect("temp dir");
        fs::write(temp.path().join("a.rs"), "fn first() {}").expect("write");

        let engine = SearchEngine::new(temp.path().to_path_buf());
        let result1 = engine.search("first", 20, None).expect("search");
        assert!(result1.contains("a.rs"));

        // Modify the file
        fs::write(temp.path().join("a.rs"), "fn renamed() {}").expect("write");

        // Add another file
        fs::write(temp.path().join("b.rs"), "fn second() {}").expect("write");

        // Delete the db to simulate rebuild
        let db_path = temp.path().join(".search-index").join("search.db");
        let _ = std::fs::remove_file(&db_path);

        let result2 = engine.search("second", 20, None).expect("search");
        assert!(result2.contains("b.rs"));
    }
}
