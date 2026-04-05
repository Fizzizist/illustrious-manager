use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

#[derive(Debug, Default, PartialEq, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum ConfirmationMode {
    Always,
    #[default]
    WriteOnly,
    Never,
}

fn default_confirmation() -> ConfirmationMode {
    ConfirmationMode::default()
}

fn default_sandbox_root() -> String {
    ".".to_string()
}

fn default_max_tool_iterations() -> u32 {
    25
}

fn default_bash_allowlist() -> Vec<String> {
    ["cat", "ls", "grep", "find", "head", "tail", "wc", "tree"]
        .iter()
        .map(ToString::to_string)
        .collect()
}

fn default_bash_denylist() -> Vec<String> {
    ["rm", "wget", "sudo", "chmod", "chown"]
        .iter()
        .map(ToString::to_string)
        .collect()
}

#[derive(Debug, serde::Deserialize)]
pub struct ToolsConfig {
    #[serde(default = "default_confirmation")]
    pub confirmation: ConfirmationMode,
    #[serde(default = "default_sandbox_root")]
    pub sandbox_root: String,
    #[serde(default = "default_max_tool_iterations")]
    pub max_tool_iterations: u32,
    #[serde(default = "default_bash_allowlist")]
    pub bash_allowlist: Vec<String>,
    #[serde(default = "default_bash_denylist")]
    pub bash_denylist: Vec<String>,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        ToolsConfig {
            confirmation: default_confirmation(),
            sandbox_root: default_sandbox_root(),
            max_tool_iterations: default_max_tool_iterations(),
            bash_allowlist: default_bash_allowlist(),
            bash_denylist: default_bash_denylist(),
        }
    }
}

pub fn resolve_sandbox_root(path: &str) -> Result<PathBuf> {
    let p = PathBuf::from(path);
    let base = if p.is_absolute() {
        p
    } else {
        std::env::current_dir()
            .context("Failed to get current directory")?
            .join(p)
    };
    base.canonicalize().with_context(|| {
        format!(
            "sandbox_root '{}' does not exist or is not accessible",
            path
        )
    })
}

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

# [tools]
# When to prompt for confirmation before executing a tool: Always, WriteOnly, or Never
# confirmation = "WriteOnly"
# Directory tools are allowed to read/write (resolved to absolute path at startup)
# sandbox_root = "."
# Maximum number of tool-use iterations per agent turn
# max_tool_iterations = 25
# Shell commands that may be executed without a denylist match
# bash_allowlist = ["cat", "ls", "grep", "find", "head", "tail", "wc", "tree"]
# Shell commands that are always blocked
# bash_denylist = ["rm", "wget", "sudo", "chmod", "chown"]
"#;

#[derive(Debug, serde::Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_backend")]
    pub backend: String,
    pub vertex: VertexConfig,
    #[serde(default)]
    pub zai: Option<ZaiConfig>,
    #[serde(default)]
    pub tools: ToolsConfig,
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
    let mut config: AppConfig = toml::from_str(&content)
        .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
    config.tools.sandbox_root = resolve_sandbox_root(&config.tools.sandbox_root)?
        .to_string_lossy()
        .into_owned();
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
    fn default_tools_config_has_correct_defaults() {
        let config = ToolsConfig::default();
        assert_eq!(config.confirmation, ConfirmationMode::WriteOnly);
        assert_eq!(config.sandbox_root, ".");
        assert_eq!(config.max_tool_iterations, 25);
        assert_eq!(
            config.bash_allowlist,
            vec!["cat", "ls", "grep", "find", "head", "tail", "wc", "tree"]
        );
        assert_eq!(
            config.bash_denylist,
            vec!["rm", "wget", "sudo", "chmod", "chown"]
        );
    }

    #[test]
    fn custom_tools_config_overrides_all_fields() {
        let toml_str = r#"
            confirmation = "Always"
            sandbox_root = "/tmp/sandbox"
            max_tool_iterations = 10
            bash_allowlist = ["echo"]
            bash_denylist = ["curl"]
        "#;
        let config: ToolsConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.confirmation, ConfirmationMode::Always);
        assert_eq!(config.sandbox_root, "/tmp/sandbox");
        assert_eq!(config.max_tool_iterations, 10);
        assert_eq!(config.bash_allowlist, vec!["echo"]);
        assert_eq!(config.bash_denylist, vec!["curl"]);
    }

    #[test]
    fn missing_tools_section_uses_all_defaults() {
        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "my-project"
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.tools.confirmation, ConfirmationMode::WriteOnly);
        assert_eq!(config.tools.max_tool_iterations, 25);
        assert_eq!(config.tools.sandbox_root, ".");
    }

    #[test]
    fn sandbox_root_resolved_to_absolute_path() {
        let resolved = resolve_sandbox_root(".").expect("should resolve");
        assert!(resolved.is_absolute());
    }

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
            tools: ToolsConfig::default(),
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
            tools: ToolsConfig::default(),
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
            tools: ToolsConfig::default(),
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
            tools: ToolsConfig::default(),
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
            tools: ToolsConfig::default(),
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
            tools: ToolsConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid backend"));
    }
}
