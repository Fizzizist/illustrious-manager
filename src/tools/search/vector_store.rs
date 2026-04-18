pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn top_k_similar(query: &[f32], vectors: &[(i64, Vec<f32>)], k: usize) -> Vec<(i64, f32)> {
    if k == 0 || vectors.is_empty() {
        return Vec::new();
    }

    let mut scored: Vec<(i64, f32)> = vectors
        .iter()
        .map(|(id, vec)| (*id, cosine_similarity(query, vec)))
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("floats should be comparable"));

    scored.into_iter().take(k).collect()
}

pub fn pack_vector(vector: &[f32]) -> Vec<u8> {
    let byte_len = vector.len() * 4;
    let mut buf = Vec::with_capacity(byte_len);
    for &val in vector {
        buf.extend_from_slice(&val.to_le_bytes());
    }
    buf
}

pub fn unpack_vector(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_cosine_similarity_one() {
        let v = vec![1.0_f32, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_have_cosine_similarity_zero() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors_have_cosine_similarity_minus_one() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn top_k_returns_highest_scoring() {
        let query = vec![1.0_f32, 0.0];
        let vectors = vec![
            (1_i64, vec![0.5, 0.5]),
            (2_i64, vec![1.0, 0.0]),
            (3_i64, vec![0.0, 1.0]),
        ];
        let result = top_k_similar(&query, &vectors, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 2);
        assert!(result[0].1 > result[1].1);
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let query = vec![1.0_f32];
        let vectors = vec![(1_i64, vec![1.0])];
        assert!(top_k_similar(&query, &vectors, 0).is_empty());
    }

    #[test]
    fn top_k_empty_vectors_returns_empty() {
        let query = vec![1.0_f32];
        assert!(top_k_similar(&query, &[], 5).is_empty());
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let original = vec![0.1_f32, -0.2, 0.3, 0.0, 1.0];
        let packed = pack_vector(&original);
        let unpacked = unpack_vector(&packed);
        assert_eq!(original.len(), unpacked.len());
        for (a, b) in original.iter().zip(unpacked.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
