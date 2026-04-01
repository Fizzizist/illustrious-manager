use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

const DEFAULT_REGION: &str = "us-east5";
const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";

const CONFIG_TEMPLATE: &str = r#"[vertex]
# Required: your GCP project ID
project = ""
# Vertex AI region
region = "us-east5"
# Model to use
model = "claude-sonnet-4-20250514"
"#;

#[derive(Debug, serde::Deserialize)]
pub struct AppConfig {
    pub vertex: VertexConfig,
}

#[derive(Debug, serde::Deserialize)]
pub struct VertexConfig {
    pub project: String,
    #[serde(default = "default_region")]
    pub region: String,
    #[serde(default = "default_model")]
    pub model: String,
}

fn default_region() -> String {
    DEFAULT_REGION.to_string()
}

fn default_model() -> String {
    DEFAULT_MODEL.to_string()
}

/// Returns the default config file path: ~/.config/illustrious-manager/config.toml
pub fn default_config_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir().context("Could not determine config directory")?;
    Ok(config_dir.join("illustrious-manager").join("config.toml"))
}

/// Load config from a specific path.
pub fn load_config_from_path(path: &Path) -> Result<AppConfig> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;
    let config: AppConfig = toml::from_str(&content)
        .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
    Ok(config)
}

/// Load config from the default path, auto-creating if it doesn't exist.
pub fn load_config(custom_path: Option<&Path>) -> Result<AppConfig> {
    let path = match custom_path {
        Some(p) => p.to_path_buf(),
        None => default_config_path()?,
    };

    if !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create config directory: {}", parent.display())
            })?;
        }
        std::fs::write(&path, CONFIG_TEMPLATE)
            .with_context(|| format!("Failed to write default config: {}", path.display()))?;
        eprintln!("Created default config at: {}", path.display());
    }

    load_config_from_path(&path)
}

/// Apply CLI flag overrides to the config.
pub fn apply_overrides(
    config: &mut AppConfig,
    project: Option<&str>,
    region: Option<&str>,
    model: Option<&str>,
) {
    if let Some(p) = project {
        config.vertex.project = p.to_string();
    }
    if let Some(r) = region {
        config.vertex.region = r.to_string();
    }
    if let Some(m) = model {
        config.vertex.model = m.to_string();
    }
}

/// Validate that the config has all required fields.
///
/// `config_path`: The path that was actually used to load the config, for accurate error messages.
/// Pass `None` if the default path was used.
pub fn validate(config: &AppConfig, config_path: Option<&Path>) -> Result<()> {
    if config.vertex.project.is_empty() {
        let path = match config_path {
            Some(p) => p.display().to_string(),
            None => default_config_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "~/.config/illustrious-manager/config.toml".to_string()),
        };
        bail!(
            "GCP project ID is required. Set it in your config file at:\n  {}\n\nOr pass --project <PROJECT> on the command line.",
            path
        );
    }
    Ok(())
}
