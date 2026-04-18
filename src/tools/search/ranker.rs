use crate::tools::search::query_classifier::QueryType;

#[derive(Debug, Clone)]
pub struct MetricScores {
    pub bm25: f64,
    pub cosine: f64,
    pub path_match: f64,
    pub symbol_match: f64,
    pub import_graph: f64,
    pub git_recency: f64,
}

#[derive(Debug, Clone)]
pub struct RankedCandidate {
    pub id: i64,
    pub scores: MetricScores,
    pub rank: usize,
}

const METRIC_NAMES: [&str; 6] = [
    "bm25",
    "cosine",
    "path_match",
    "symbol_match",
    "import_graph",
    "git_recency",
];

struct ColumnWeights {
    bm25: usize,
    cosine: usize,
    path_match: usize,
    symbol_match: usize,
    import_graph: usize,
    git_recency: usize,
}

fn column_weights(query_type: &QueryType) -> ColumnWeights {
    match query_type {
        QueryType::Identifier => ColumnWeights {
            bm25: 2,
            cosine: 1,
            path_match: 1,
            symbol_match: 2,
            import_graph: 1,
            git_recency: 1,
        },
        QueryType::NaturalLanguage => ColumnWeights {
            bm25: 1,
            cosine: 2,
            path_match: 1,
            symbol_match: 1,
            import_graph: 1,
            git_recency: 1,
        },
        QueryType::PathLike => ColumnWeights {
            bm25: 1,
            cosine: 1,
            path_match: 3,
            symbol_match: 1,
            import_graph: 1,
            git_recency: 1,
        },
    }
}

const EPSILON: f64 = 0.05;

fn get_score(scores: &MetricScores, metric: &str) -> f64 {
    match metric {
        "bm25" => scores.bm25,
        "cosine" => scores.cosine,
        "path_match" => scores.path_match,
        "symbol_match" => scores.symbol_match,
        "import_graph" => scores.import_graph,
        "git_recency" => scores.git_recency,
        _ => 0.0,
    }
}

fn get_weight(weights: &ColumnWeights, metric: &str) -> usize {
    match metric {
        "bm25" => weights.bm25,
        "cosine" => weights.cosine,
        "path_match" => weights.path_match,
        "symbol_match" => weights.symbol_match,
        "import_graph" => weights.import_graph,
        "git_recency" => weights.git_recency,
        _ => 0,
    }
}

pub fn poem_rank(
    candidates: &std::collections::HashMap<i64, MetricScores>,
    query_type: &QueryType,
    top_k: usize,
) -> Vec<RankedCandidate> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let surviving = prune_top_k(candidates, top_k);

    let ids: Vec<i64> = surviving.iter().map(|&(id, _)| id).collect();
    let scores: Vec<&MetricScores> = surviving.iter().map(|&(_, s)| s).collect();

    if ids.len() == 1 {
        return vec![RankedCandidate {
            id: ids[0],
            scores: scores[0].clone(),
            rank: 0,
        }];
    }

    let weights = column_weights(query_type);
    let n = ids.len();
    let total_weight = METRIC_NAMES
        .iter()
        .map(|m| get_weight(&weights, m))
        .sum::<usize>() as f64;

    let mut counts = vec![0u16; n * n];

    for metric in &METRIC_NAMES {
        let weight = get_weight(&weights, metric) as u16;
        if weight == 0 {
            continue;
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            let sa = get_score(scores[a], metric);
            let sb = get_score(scores[b], metric);
            sb.partial_cmp(&sa).expect("floats should be comparable")
        });

        let k = top_k.min(n);

        for ri in 0..k {
            let i = indices[ri];
            for rj in (ri + 1)..k {
                counts[i * n + indices[rj]] += weight;
            }
        }
    }

    let threshold = total_weight * 0.5;
    let mut fitness = vec![0.0_f64; n];

    for i in 0..n {
        let mut sum_dom = 0.0_f64;
        let mut num_dominating = 0usize;
        let mut num_submitting = 0usize;

        for j in 0..n {
            if i == j {
                continue;
            }
            let count = counts[i * n + j] as f64;
            sum_dom += count;
            if count > threshold {
                num_dominating += 1;
            }
            if count < threshold {
                num_submitting += 1;
            }
        }

        let mean_dom = if n > 1 {
            sum_dom / ((n - 1) as f64 * total_weight)
        } else {
            0.0
        };
        fitness[i] =
            mean_dom * (num_dominating as f64 + EPSILON) / (num_submitting as f64 + EPSILON);
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        fitness[b]
            .partial_cmp(&fitness[a])
            .expect("floats should be comparable")
    });

    order
        .into_iter()
        .enumerate()
        .map(|(rank, idx)| RankedCandidate {
            id: ids[idx],
            scores: scores[idx].clone(),
            rank,
        })
        .collect()
}

fn prune_top_k(
    candidates: &std::collections::HashMap<i64, MetricScores>,
    top_k: usize,
) -> Vec<(i64, &MetricScores)> {
    if candidates.len() <= top_k {
        return candidates.iter().map(|(&id, s)| (id, s)).collect();
    }

    let mut surviving = std::collections::HashSet::new();

    for metric in &METRIC_NAMES {
        let mut pairs: Vec<(i64, f64)> = candidates
            .iter()
            .map(|(&id, s)| (id, get_score(s, metric)))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("floats"));
        for (id, _) in pairs.into_iter().take(top_k) {
            surviving.insert(id);
        }
    }

    candidates
        .iter()
        .filter(|(id, _)| surviving.contains(id))
        .map(|(&id, s)| (id, s))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_candidates_returns_empty() {
        let candidates = std::collections::HashMap::new();
        let result = poem_rank(&candidates, &QueryType::Identifier, 1000);
        assert!(result.is_empty());
    }

    #[test]
    fn single_candidate_gets_rank_zero() {
        let mut candidates = std::collections::HashMap::new();
        candidates.insert(
            1,
            MetricScores {
                bm25: 0.5,
                cosine: 0.8,
                path_match: 0.0,
                symbol_match: 0.3,
                import_graph: 0.0,
                git_recency: 0.5,
            },
        );
        let result = poem_rank(&candidates, &QueryType::Identifier, 1000);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].rank, 0);
        assert_eq!(result[0].id, 1);
    }

    #[test]
    fn higher_scoring_candidate_ranks_better() {
        let mut candidates = std::collections::HashMap::new();
        candidates.insert(
            1,
            MetricScores {
                bm25: 0.9,
                cosine: 0.9,
                path_match: 0.9,
                symbol_match: 0.9,
                import_graph: 0.5,
                git_recency: 0.5,
            },
        );
        candidates.insert(
            2,
            MetricScores {
                bm25: 0.1,
                cosine: 0.1,
                path_match: 0.1,
                symbol_match: 0.1,
                import_graph: 0.5,
                git_recency: 0.5,
            },
        );
        let result = poem_rank(&candidates, &QueryType::NaturalLanguage, 1000);
        assert_eq!(
            result[0].id, 1,
            "Higher-scoring candidate should rank first"
        );
        assert_eq!(result[1].id, 2);
    }

    #[test]
    fn deterministic_ranking_for_same_inputs() {
        let mut candidates = std::collections::HashMap::new();
        candidates.insert(
            1,
            MetricScores {
                bm25: 0.5,
                cosine: 0.7,
                path_match: 0.3,
                symbol_match: 0.4,
                import_graph: 0.2,
                git_recency: 0.6,
            },
        );
        candidates.insert(
            2,
            MetricScores {
                bm25: 0.3,
                cosine: 0.5,
                path_match: 0.7,
                symbol_match: 0.2,
                import_graph: 0.8,
                git_recency: 0.4,
            },
        );
        candidates.insert(
            3,
            MetricScores {
                bm25: 0.7,
                cosine: 0.3,
                path_match: 0.5,
                symbol_match: 0.6,
                import_graph: 0.1,
                git_recency: 0.9,
            },
        );

        let result1 = poem_rank(&candidates, &QueryType::Identifier, 1000);
        let result2 = poem_rank(&candidates, &QueryType::Identifier, 1000);

        for (a, b) in result1.iter().zip(result2.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.rank, b.rank);
        }
    }
}
