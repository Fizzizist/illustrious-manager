use anyhow::{Context, Result};
use ort::session::Session;
use std::path::{Path, PathBuf};
use std::str::FromStr;

const DEFAULT_MODEL_NAME: &str = "Xenova/all-MiniLM-L6-v2";
const DEFAULT_DIMENSION: usize = 384;

pub struct Embedder {
    model_cache_dir: PathBuf,
    model_name: String,
    session: Option<Session>,
    tokenizer: Option<tokenizers::Tokenizer>,
    dimension: usize,
}

impl Embedder {
    pub fn new(model_cache_dir: PathBuf) -> Self {
        Self {
            model_cache_dir,
            model_name: DEFAULT_MODEL_NAME.to_string(),
            session: None,
            tokenizer: None,
            dimension: DEFAULT_DIMENSION,
        }
    }

    pub fn initialize(&mut self) -> Result<()> {
        if self.session.is_some() {
            return Ok(());
        }

        let model_dir = self.model_cache_dir.join(&self.model_name);
        std::fs::create_dir_all(&model_dir)
            .with_context(|| format!("Creating model cache dir: {}", model_dir.display()))?;

        let onnx_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !onnx_path.exists() || !tokenizer_path.exists() {
            download_model(&self.model_name, &model_dir)?;
        }

        let session = Session::builder()
            .context("Creating ONNX session builder")?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("Loading ONNX model from {}", onnx_path.display()))?;

        let tokenizer_data = std::fs::read_to_string(&tokenizer_path)
            .with_context(|| format!("Reading tokenizer from {}", tokenizer_path.display()))?;
        let tokenizer = tokenizers::Tokenizer::from_str(&tokenizer_data)
            .map_err(|e| anyhow::anyhow!("Parsing tokenizer JSON: {e}"))?;

        self.dimension = detect_dimension(&session);
        self.session = Some(session);
        self.tokenizer = Some(tokenizer);

        Ok(())
    }

    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let tokenizer = self.tokenizer.as_ref().expect("Embedder not initialized");

        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let encoding = tokenizer
                .encode(*text, true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

            let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
            let attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&m| m as i64)
                .collect();
            let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

            let len = ids.len();
            let input_ids = ndarray::Array2::from_shape_vec((1, len), ids)
                .context("Creating input_ids array")?;
            let attn_mask = ndarray::Array2::from_shape_vec((1, len), attention_mask)
                .context("Creating attention_mask array")?;
            let token_types = ndarray::Array2::from_shape_vec((1, len), type_ids)
                .context("Creating token_type_ids array")?;

            let session = self.session.as_mut().expect("Embedder not session");

            let input_ids_val =
                ort::value::Tensor::from_array(input_ids).context("Creating input_ids tensor")?;
            let attn_mask_val = ort::value::Tensor::from_array(attn_mask)
                .context("Creating attention_mask tensor")?;
            let token_types_val = ort::value::Tensor::from_array(token_types)
                .context("Creating token_type_ids tensor")?;

            let outputs = session
                .run(ort::inputs! {
                    "input_ids" => input_ids_val,
                    "attention_mask" => attn_mask_val,
                    "token_type_ids" => token_types_val,
                })
                .context("Running ONNX inference")?;

            let output = outputs.iter().next().context("No output from model")?.1;

            let (_, data) = output
                .try_extract_tensor::<f32>()
                .context("Extracting tensor")?;

            let mask_f32: Vec<f32> = encoding
                .get_attention_mask()
                .iter()
                .map(|&m| m as f32)
                .collect();
            let embedding = mean_pool_normalize(data, len, self.dimension, &mask_f32);

            results.push(embedding);
        }

        Ok(results)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

fn detect_dimension(session: &Session) -> usize {
    session
        .outputs()
        .first()
        .and_then(|outlet| outlet.dtype().tensor_shape())
        .and_then(|shape| shape.last().copied())
        .filter(|&d| d > 0)
        .map(|d| d as usize)
        .unwrap_or(DEFAULT_DIMENSION)
}

fn mean_pool_normalize(data: &[f32], seq_len: usize, dim: usize, mask: &[f32]) -> Vec<f32> {
    let mut pooled = vec![0.0_f32; dim];
    let mut mask_sum = 0.0_f32;

    for i in 0..seq_len {
        let weight = mask[i];
        mask_sum += weight;
        for j in 0..dim {
            pooled[j] += data[i * dim + j] * weight;
        }
    }

    if mask_sum > 0.0 {
        for val in pooled.iter_mut() {
            *val /= mask_sum;
        }
    }

    let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in pooled.iter_mut() {
            *val /= norm;
        }
    }

    pooled
}

fn download_model(model_name: &str, target_dir: &Path) -> Result<()> {
    let base_url = format!("https://huggingface.co/{model_name}/resolve/main");

    let files = ["model.onnx", "tokenizer.json"];

    for file in &files {
        let url = format!("{base_url}/{file}");
        let dest = target_dir.join(file);

        eprintln!("Downloading {url}...");

        let response = reqwest::blocking::get(&url)
            .with_context(|| format!("HTTP request to {url}"))?
            .error_for_status()
            .context("HTTP request failed")?;
        let buf = response.bytes().context("Reading response body")?;
        std::fs::write(&dest, &buf).with_context(|| format!("Writing {}", dest.display()))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedder_new_has_no_session() {
        let embedder = Embedder::new(std::env::temp_dir());
        assert!(embedder.session.is_none());
        assert!(embedder.tokenizer.is_none());
    }

    #[test]
    fn embedder_default_dimension() {
        let embedder = Embedder::new(std::env::temp_dir());
        assert_eq!(embedder.dimension(), 384);
    }

    #[test]
    fn mean_pool_normalize_produces_unit_vector() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 tokens x 3 dim
        let mask = vec![1.0_f32, 1.0];
        let result = mean_pool_normalize(&data, 2, 3, &mask);

        let norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Should be unit vector, got norm {norm}"
        );
    }

    #[test]
    fn mean_pool_normalize_with_zero_mask() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let mask = vec![0.0_f32];
        let result = mean_pool_normalize(&data, 1, 3, &mask);
        assert!(result.iter().all(|&v| v == 0.0));
    }
}
