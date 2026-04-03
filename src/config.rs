use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

const DEFAULT_REGION: &str = "us-east5";
const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";
const DEFAULT_BACKEND: &str = "vertex";

const CONFIG_TEMPLATE: &str = r#"# Which backend to use: "vertex" or "zai"
backend = "vertex"

[vertex]
# Required: your GCP project ID
project = ""
# Vertex AI region
region = "us-east5"
# Model to use
model = "claude-sonnet-4-20250514"

[zai]
# Required: your z.ai API key
api_key = ""
# Model to use
model = "glm-5.1"
"#;

#[derive(Debug, serde::Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_backend")]
    pub backend: String,
    pub vertex: VertexConfig,
    #[serde(default)]
    pub zai: Option<ZaiConfig>,
}

#[derive(Debug, serde::Deserialize)]
pub struct VertexConfig {
    pub project: String,
    #[serde(default = "default_region")]
    pub region: String,
    #[serde(default = "default_model")]
    pub model: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct ZaiConfig {
    pub api_key: String,
    #[serde(default = "default_zai_model")]
    pub model: String,
}

fn default_backend() -> String {
    DEFAULT_BACKEND.to_string()
}

fn default_zai_model() -> String {
    "glm-5.1".to_string()
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
    match config.backend.as_str() {
        "vertex" => {
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
        "zai" => {
            if let Some(m) = model
                && let Some(ref mut zai) = config.zai
            {
                zai.model = m.to_string();
            }
        }
        _ => {}
    }
}

/// Validate that the config has all required fields for the selected backend.
///
/// `config_path`: The path that was actually used to load the config, for accurate error messages.
/// Pass `None` if the default path was used.
pub fn validate(config: &AppConfig, config_path: Option<&Path>) -> Result<()> {
    match config.backend.as_str() {
        "vertex" => {
            if config.vertex.project.is_empty() {
                let path = match config_path {
                    Some(p) => p.display().to_string(),
                    None => default_config_path()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|_| {
                            "~/.config/illustrious-manager/config.toml".to_string()
                        }),
                };
                bail!(
                    "GCP project ID is required for vertex backend. Set it in your config file at:\n  {}\n\nOr pass --project <PROJECT> on the command line.",
                    path
                );
            }
        }
        "zai" => {
            if let Some(ref zai_config) = config.zai {
                if zai_config.api_key.is_empty() {
                    let path = match config_path {
                        Some(p) => p.display().to_string(),
                        None => default_config_path()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|_| {
                                "~/.config/illustrious-manager/config.toml".to_string()
                            }),
                    };
                    bail!(
                        "API key is required for zai backend. Set it in your config file at:\n  {}",
                        path
                    );
                }
            } else {
                bail!(
                    "zai backend configuration is missing. Add a [zai] section to your config file."
                );
            }
        }
        _ => {
            bail!(
                "Invalid backend '{}'. Supported backends are: vertex, zai",
                config.backend
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_vertex_backend_with_empty_project_errors() {
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("GCP project ID"));
    }

    #[test]
    fn validate_vertex_backend_with_valid_project_succeeds() {
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "my-project".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
        };
        let result = validate(&config, None);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_zai_backend_with_missing_config_errors() {
        let config = AppConfig {
            backend: "zai".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("[zai]"));
    }

    #[test]
    fn validate_zai_backend_with_empty_api_key_errors() {
        let config = AppConfig {
            backend: "zai".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: Some(ZaiConfig {
                api_key: "".to_string(),
                model: "glm-5.1".to_string(),
            }),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key"));
    }

    #[test]
    fn validate_zai_backend_with_valid_config_succeeds() {
        let config = AppConfig {
            backend: "zai".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: Some(ZaiConfig {
                api_key: "test-key".to_string(),
                model: "glm-5.1".to_string(),
            }),
        };
        let result = validate(&config, None);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_invalid_backend_errors() {
        let config = AppConfig {
            backend: "invalid".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid backend"));
    }
}
