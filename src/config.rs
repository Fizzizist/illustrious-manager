use dirs;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

#[derive(Debug, Default, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
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

fn default_sessions_dir() -> PathBuf {
    dirs::config_dir()
        .map(|path| path.join("illustrious-manager/sessions"))
        .unwrap_or_else(|| PathBuf::from("./illustrious-manager-sessions"))
}

fn default_compaction_threshold() -> f32 {
    0.80
}

fn default_compaction_keep_recent() -> usize {
    6
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
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
    /// Model context window size in tokens. When `None`, compaction is disabled.
    #[serde(default)]
    pub context_window_tokens: Option<u32>,
    /// Fraction of context window that triggers automatic compaction.
    #[serde(default = "default_compaction_threshold")]
    pub compaction_threshold: f32,
    /// Number of recent messages to keep verbatim during compaction.
    #[serde(default = "default_compaction_keep_recent")]
    pub compaction_keep_recent: usize,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        ToolsConfig {
            confirmation: default_confirmation(),
            sandbox_root: default_sandbox_root(),
            max_tool_iterations: default_max_tool_iterations(),
            bash_allowlist: default_bash_allowlist(),
            bash_denylist: default_bash_denylist(),
            context_window_tokens: None,
            compaction_threshold: default_compaction_threshold(),
            compaction_keep_recent: default_compaction_keep_recent(),
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

const CONFIG_TEMPLATE: &str = r#"# Which backend to use: "vertex", "zai", or "ollama"
backend = "vertex"
# where the session database files are stored. Defaults to $HOME/.config/illustrious-manager/sessions
# or a local `illustrious-manager-sessions` directory if $HOME is not found.
# sessions_dir = "/path/to/sessions"

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

[ollama]
# Required: your Ollama API key
api_key = ""
# Model to use
model = "gpt-oss:120b"
# Base URL for Ollama API (change for self-hosted)
# base_url = "https://ollama.com/api/chat"

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

# Named model roles for multi-agent workflows.
# When absent, a "default" role is synthesized from the top-level backend
# and the matching [vertex]/[zai]/[ollama] model field above.
# [models.thinking]
# backend = "vertex"
# model = "claude-3-5-thinking"
# [models.implement]
# backend = "vertex"
# model = "claude-sonnet-4-20250514"
"#;

/// A named model role binding a backend to a specific model string.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ModelRole {
    pub backend: String,
    pub model: String,
}

/// Concrete backend + model resolved from a `ModelRole`.
#[derive(Debug)]
pub struct ResolvedRole {
    pub backend_name: String,
    pub model: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct AppConfig {
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_sessions_dir")]
    pub sessions_dir: PathBuf,
    pub vertex: VertexConfig,
    #[serde(default)]
    pub zai: Option<ZaiConfig>,
    #[serde(default)]
    pub ollama: Option<OllamaConfig>,
    #[serde(default)]
    pub tools: ToolsConfig,
    /// Named model roles. When empty, a `default` role is synthesized from
    /// the top-level `backend` + `[vertex]`/`[zai]` blocks for back-compat.
    #[serde(default, rename = "models")]
    pub models: BTreeMap<String, ModelRole>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct VertexConfig {
    pub project: String,
    #[serde(default = "default_region")]
    pub region: String,
    #[serde(default = "default_model")]
    pub model: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct OllamaConfig {
    #[serde(skip_serializing)]
    pub api_key: String,
    #[serde(default = "default_ollama_model")]
    pub model: String,
    #[serde(default = "default_ollama_base_url")]
    pub base_url: String,
}

fn default_ollama_model() -> String {
    "gpt-oss:120b".to_string()
}

fn default_ollama_base_url() -> String {
    "https://ollama.com/api/chat".to_string()
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ZaiConfig {
    #[serde(skip_serializing)]
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
    fs::create_dir_all(&config.sessions_dir)?;
    config.normalize_back_compat();
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
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create config directory: {}", parent.display())
            })?;
        }
        fs::write(&path, CONFIG_TEMPLATE)
            .with_context(|| format!("Failed to write default config: {}", path.display()))?;
        eprintln!("Created default config at: {}", path.display());
    }

    load_config_from_path(&path)
}

impl AppConfig {
    /// Resolve a named role to its concrete backend and model names.
    pub fn resolve_role(&self, role: &str) -> Result<ResolvedRole> {
        let role_def = self
            .models
            .get(role)
            .ok_or_else(|| anyhow::anyhow!("Undefined model role '{role}'"))?;
        Ok(ResolvedRole {
            backend_name: role_def.backend.clone(),
            model: role_def.model.clone(),
        })
    }

    /// Ensure a `"default"` model role always exists, synthesized from the
    /// top-level `backend` + `[vertex]`/`[zai]`/`[ollama]` fields if the user
    /// hasn't explicitly defined one.  Other named roles are left untouched.
    ///
    /// Called by `load_config_from_path` after deserialization so that code
    /// that predates the model registry continues to work unchanged.
    pub fn normalize_back_compat(&mut self) {
        if self.models.contains_key("default") {
            return;
        }
        let model = match self.backend.as_str() {
            "zai" => self
                .zai
                .as_ref()
                .map(|z| z.model.clone())
                .unwrap_or_else(|| "glm-5.1".to_string()),
            "ollama" => self
                .ollama
                .as_ref()
                .map(|o| o.model.clone())
                .unwrap_or_else(|| "gpt-oss:120b".to_string()),
            _ => self.vertex.model.clone(),
        };
        self.models.insert(
            "default".to_string(),
            ModelRole {
                backend: self.backend.clone(),
                model,
            },
        );
    }
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
        "ollama" => {
            if let Some(m) = model
                && let Some(ref mut ollama) = config.ollama
            {
                ollama.model = m.to_string();
            }
        }
        _ => {}
    }

    if let Some(m) = model
        && let Some(default_role) = config.models.get_mut("default")
    {
        default_role.backend = config.backend.clone();
        default_role.model = m.to_string();
    }
}

/// Generate a markdown intro message displaying the currently loaded configuration.
///
/// Serializes the config to TOML to ensure all fields are captured automatically,
/// then formats each key-value pair as a markdown list item.
pub fn generate_intro_message(config: &AppConfig) -> String {
    let toml_value = toml::Value::try_from(config).unwrap_or(toml::Value::String(
        "(error serializing config)".to_string(),
    ));

    let mut msg = String::from("# Illustrious Manager\n\n");
    append_toml_as_list(&mut msg, &toml_value, "");
    msg
}

fn append_toml_as_list(out: &mut String, value: &toml::Value, prefix: &str) {
    if let toml::Value::Table(table) = value {
        for (key, val) in table {
            let full_key = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}.{key}")
            };
            match val {
                toml::Value::Table(_) => append_toml_as_list(out, val, &full_key),
                toml::Value::Array(arr) => {
                    let items: Vec<String> = arr.iter().map(toml_value_display).collect();
                    let _ = writeln!(out, "- **{full_key}:** {}", items.join(", "));
                }
                _ => {
                    let _ = writeln!(out, "- **{full_key}:** {}", toml_value_display(val));
                }
            }
        }
    }
}

fn toml_value_display(val: &toml::Value) -> String {
    match val {
        toml::Value::String(s) => s.clone(),
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::Float(f) => format_float(*f),
        toml::Value::Boolean(b) => b.to_string(),
        other => other.to_string(),
    }
}

fn format_float(f: f64) -> String {
    // Round-trip through f32 to match the precision of the original source value,
    // which is typically an f32 in Rust. This eliminates f64 representation
    // artifacts like 0.800000011920929.
    let as_f32 = f as f32;
    let as_f64 = as_f32 as f64;
    // Use enough decimal places to represent the f32 precisely, but trim
    // trailing zeros for cleanliness.
    let s = format!("{as_f64:.6}");
    let trimmed = s.trim_end_matches('0');
    let trimmed = trimmed.trim_end_matches('.');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        // Ensure at least one decimal place for float identification
        if !trimmed.contains('.') {
            format!("{trimmed}.0")
        } else {
            trimmed.to_string()
        }
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
        "ollama" => {
            if let Some(ref ollama_config) = config.ollama {
                if ollama_config.api_key.is_empty()
                    && ollama_config.base_url == default_ollama_base_url()
                {
                    let path = match config_path {
                        Some(p) => p.display().to_string(),
                        None => default_config_path()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|_| {
                                "~/.config/illustrious-manager/config.toml".to_string()
                            }),
                    };
                    bail!(
                        "API key is required for Ollama Cloud. Set it in your config file at:\n  {}\nOr set base_url to your self-hosted endpoint.",
                        path
                    );
                }
            } else {
                bail!(
                    "ollama backend configuration is missing. Add a [ollama] section to your config file."
                );
            }
        }
        _ => {
            bail!(
                "Invalid backend '{}'. Supported backends are: vertex, zai, ollama",
                config.backend
            );
        }
    }

    // Validate named roles: each role must reference a configured backend.
    for (name, role) in &config.models {
        match role.backend.as_str() {
            "vertex" => {
                if config.vertex.project.is_empty() {
                    bail!(
                        "Model role '{name}' uses backend 'vertex' but [vertex].project is not configured."
                    );
                }
            }
            "zai" => match &config.zai {
                None => {
                    bail!(
                        "Model role '{name}' uses backend 'zai' but no [zai] section is present."
                    );
                }
                Some(zai) if zai.api_key.is_empty() => {
                    bail!(
                        "Model role '{name}' uses backend 'zai' but [zai].api_key is not configured."
                    );
                }
                _ => {}
            },
            "ollama" => match &config.ollama {
                None => {
                    bail!(
                        "Model role '{name}' uses backend 'ollama' but no [ollama] section is present."
                    );
                }
                Some(ollama)
                    if ollama.api_key.is_empty()
                        && ollama.base_url == default_ollama_base_url() =>
                {
                    bail!(
                        "Model role '{name}' uses backend 'ollama' but [ollama].api_key is not configured."
                    );
                }
                _ => {}
            },
            other => {
                bail!(
                    "Model role '{name}' references unknown backend '{other}'. Supported: vertex, zai, ollama"
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

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
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
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
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
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
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
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
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
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
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
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
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid backend"));
    }

    #[test]
    fn generate_intro_message_contains_vertex_backend_settings() {
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "my-gcp-project".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        let msg = generate_intro_message(&config);
        assert!(msg.contains("vertex"), "should mention backend name");
        assert!(msg.contains("my-gcp-project"), "should mention project");
        assert!(msg.contains("us-east5"), "should mention region");
        assert!(
            msg.contains("claude-sonnet-4-20250514"),
            "should mention model"
        );
    }

    #[test]
    fn generate_intro_message_contains_zai_backend_settings() {
        let config = AppConfig {
            backend: "zai".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: Some(ZaiConfig {
                api_key: "secret-key".to_string(),
                model: "glm-5.1".to_string(),
            }),
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        let msg = generate_intro_message(&config);
        assert!(msg.contains("zai"), "should mention backend name");
        assert!(msg.contains("glm-5.1"), "should mention zai model");
        assert!(!msg.contains("secret-key"), "should not leak API key");
    }

    #[test]
    fn generate_intro_message_contains_tool_config() {
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig {
                confirmation: ConfirmationMode::Always,
                sandbox_root: "/tmp/sandbox".to_string(),
                max_tool_iterations: 10,
                ..Default::default()
            },
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        let msg = generate_intro_message(&config);
        assert!(msg.contains("Always"), "should mention confirmation mode");
        assert!(msg.contains("/tmp/sandbox"), "should mention sandbox root");
        assert!(msg.contains("10"), "should mention max tool iterations");
    }

    #[test]
    fn generate_intro_message_contains_sessions_dir() {
        let sessions_dir = std::env::temp_dir().join("my-sessions");
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: sessions_dir.clone(),
            models: BTreeMap::new(),
        };
        let msg = generate_intro_message(&config);
        assert!(
            msg.contains(&sessions_dir.display().to_string()),
            "should mention sessions directory"
        );
    }

    #[test]
    fn generate_intro_message_contains_markdown_heading() {
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        let msg = generate_intro_message(&config);
        assert!(msg.starts_with("# "), "should start with markdown heading");
    }

    #[test]
    fn legacy_config_without_models_section_synthesizes_default_role() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "my-project"
            region = "us-east5"
            model = "claude-sonnet-4-20250514"
        "#;
        let mut tmp = NamedTempFile::new().expect("temp file");
        write!(tmp, "{toml_str}").expect("write");

        let config = load_config_from_path(tmp.path()).expect("load config");

        assert!(
            config.models.contains_key("default"),
            "default role should be synthesized by load_config_from_path"
        );
        let default_role = &config.models["default"];
        assert_eq!(default_role.backend, "vertex");
        assert_eq!(default_role.model, "claude-sonnet-4-20250514");
    }

    #[test]
    fn normalize_back_compat_synthensizes_default_alongside_existing_roles() {
        let mut models = BTreeMap::new();
        models.insert(
            "thinking".to_string(),
            ModelRole {
                backend: "vertex".to_string(),
                model: "claude-haiku".to_string(),
            },
        );
        let mut config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
        };
        config.normalize_back_compat();
        assert!(
            config.models.contains_key("default"),
            "should synthesize default role from backend config even when other roles exist"
        );
        assert!(config.models.contains_key("thinking"));
        assert_eq!(
            config.models["default"].backend, "vertex",
            "default role backend should match top-level backend"
        );
        assert_eq!(
            config.models["default"].model, "claude-sonnet-4-20250514",
            "default role model should come from [vertex].model"
        );
    }

    #[test]
    fn normalize_back_compat_preserves_explicit_default() {
        let mut models = BTreeMap::new();
        models.insert(
            "default".to_string(),
            ModelRole {
                backend: "vertex".to_string(),
                model: "custom-default-model".to_string(),
            },
        );
        models.insert(
            "thinking".to_string(),
            ModelRole {
                backend: "vertex".to_string(),
                model: "claude-haiku".to_string(),
            },
        );
        let mut config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
        };
        config.normalize_back_compat();
        assert_eq!(
            config.models["default"].model, "custom-default-model",
            "should not overwrite an explicitly defined default role"
        );
        assert_eq!(config.models.len(), 2);
    }

    #[test]
    fn models_section_parses_multiple_roles() {
        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "proj"
            [models.thinking]
            backend = "vertex"
            model = "claude-3-5-thinking"
            [models.implement]
            backend = "vertex"
            model = "claude-sonnet-4-20250514"
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert!(config.models.contains_key("thinking"));
        assert!(config.models.contains_key("implement"));
        assert_eq!(config.models["thinking"].model, "claude-3-5-thinking");
        assert_eq!(config.models["implement"].model, "claude-sonnet-4-20250514");
    }

    #[test]
    fn resolve_role_returns_concrete_backend_and_model_for_named_role() {
        let mut models = BTreeMap::new();
        models.insert(
            "fast".to_string(),
            ModelRole {
                backend: "vertex".to_string(),
                model: "claude-haiku".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
        };
        let resolved = config.resolve_role("fast").expect("should resolve");
        assert_eq!(resolved.backend_name, "vertex");
        assert_eq!(resolved.model, "claude-haiku");
    }

    #[test]
    fn resolve_role_errors_for_undefined_role() {
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        let result = config.resolve_role("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("nonexistent"));
    }

    #[test]
    fn validate_errors_when_role_references_unconfigured_zai_backend() {
        let mut models = BTreeMap::new();
        models.insert(
            "my-role".to_string(),
            ModelRole {
                backend: "zai".to_string(),
                model: "glm-5.1".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("my-role") || msg.contains("zai"),
            "error should mention the role or backend"
        );
    }

    #[test]
    fn validate_errors_when_role_references_zai_with_empty_api_key() {
        let mut models = BTreeMap::new();
        models.insert(
            "my-role".to_string(),
            ModelRole {
                backend: "zai".to_string(),
                model: "glm-5.1".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: Some(ZaiConfig {
                api_key: "".to_string(),
                model: "glm-5.1".to_string(),
            }),
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("api_key") || msg.contains("my-role") || msg.contains("zai"),
            "error should mention key, role, or backend; got: {msg}"
        );
    }

    #[test]
    fn validate_ollama_backend_with_missing_config_errors() {
        let config = AppConfig {
            backend: "ollama".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("[ollama]"));
    }

    #[test]
    fn validate_ollama_backend_with_empty_api_key_for_cloud_errors() {
        let config = AppConfig {
            backend: "ollama".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key"));
    }

    #[test]
    fn validate_ollama_backend_with_empty_api_key_for_self_hosted_succeeds() {
        let config = AppConfig {
            backend: "ollama".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "http://localhost:11434/api/chat".to_string(),
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        let result = validate(&config, None);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_ollama_backend_with_valid_config_succeeds() {
        let config = AppConfig {
            backend: "ollama".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "test-key".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        let result = validate(&config, None);
        assert!(result.is_ok());
    }

    #[test]
    fn apply_overrides_ollama_model() {
        let mut config = AppConfig {
            backend: "ollama".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "test-key".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        apply_overrides(&mut config, None, None, Some("custom-model"));
        assert_eq!(
            config
                .ollama
                .as_ref()
                .expect("ollama config present in test fixture")
                .model,
            "custom-model",
            "model override should be applied to ollama config"
        );
    }

    /// Regression: `--model` flag must override the synthesized `default`
    /// model role, not just the per-backend `[vertex]/[zai]/[ollama]` table.
    /// The backend resolves the model via `resolve_role("default")`, so an
    /// override that only touches the per-backend table is silently ignored.
    #[test]
    fn apply_overrides_updates_default_model_role_for_vertex() {
        let mut config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "p".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        config.normalize_back_compat();

        apply_overrides(&mut config, None, None, Some("cli-model"));

        let resolved = config
            .resolve_role("default")
            .expect("default role exists after normalize_back_compat");
        assert_eq!(
            resolved.model, "cli-model",
            "--model flag must propagate into models[\"default\"]"
        );
    }

    #[test]
    fn apply_overrides_updates_default_model_role_for_zai() {
        let mut config = AppConfig {
            backend: "zai".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: Some(ZaiConfig {
                api_key: "k".to_string(),
                model: "glm-5.1".to_string(),
            }),
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        config.normalize_back_compat();

        apply_overrides(&mut config, None, None, Some("glm-9000"));

        let resolved = config
            .resolve_role("default")
            .expect("default role exists after normalize_back_compat");
        assert_eq!(resolved.model, "glm-9000");
    }

    #[test]
    fn apply_overrides_updates_default_model_role_for_ollama() {
        let mut config = AppConfig {
            backend: "ollama".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "k".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        config.normalize_back_compat();

        apply_overrides(&mut config, None, None, Some("llama-cli"));

        let resolved = config
            .resolve_role("default")
            .expect("default role exists after normalize_back_compat");
        assert_eq!(resolved.model, "llama-cli");
    }

    #[test]
    fn validate_errors_when_role_references_ollama_with_missing_config() {
        let mut models = BTreeMap::new();
        models.insert(
            "my-ollama".to_string(),
            ModelRole {
                backend: "ollama".to_string(),
                model: "gpt-oss:120b".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("my-ollama") || msg.contains("ollama"),
            "error should mention the role or backend; got: {msg}"
        );
    }

    #[test]
    fn validate_succeeds_when_role_references_configured_ollama_backend() {
        let mut models = BTreeMap::new();
        models.insert(
            "my-ollama".to_string(),
            ModelRole {
                backend: "ollama".to_string(),
                model: "gpt-oss:120b".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "test-key".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
        };
        let result = validate(&config, None);
        assert!(result.is_ok(), "ollama role with valid config should pass");
    }

    #[test]
    fn legacy_ollama_config_synthesizes_default_role_with_ollama_model() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let toml_str = r#"
            backend = "ollama"
            [vertex]
            project = ""
            region = "us-east5"
            model = "claude-sonnet-4-20250514"
            [ollama]
            api_key = "test-key"
            model = "gpt-oss:120b"
            base_url = "https://ollama.com/api/chat"
        "#;
        let mut tmp = NamedTempFile::new().expect("temp file");
        write!(tmp, "{toml_str}").expect("write");

        let config = load_config_from_path(tmp.path()).expect("load config");

        assert!(
            config.models.contains_key("default"),
            "default role should be synthesized for ollama backend"
        );
        let default_role = &config.models["default"];
        assert_eq!(default_role.backend, "ollama");
        assert_eq!(
            default_role.model, "gpt-oss:120b",
            "model must come from [ollama].model, not vertex"
        );
    }

    #[test]
    fn generate_intro_message_contains_ollama_backend_settings() {
        let config = AppConfig {
            backend: "ollama".to_string(),
            vertex: VertexConfig {
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "secret-key".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
        };
        let msg = generate_intro_message(&config);
        assert!(msg.contains("ollama"), "should mention backend name");
        assert!(msg.contains("gpt-oss:120b"), "should mention ollama model");
        assert!(msg.contains("ollama.com"), "should mention base_url");
        assert!(!msg.contains("secret-key"), "should not leak API key");
    }

    #[test]
    fn tools_config_default_values() {
        let config = ToolsConfig::default();
        assert_eq!(config.context_window_tokens, None);
        assert!(
            (config.compaction_threshold - 0.80).abs() < 0.01,
            "default compaction_threshold should be ~0.80"
        );
        assert_eq!(config.compaction_keep_recent, 6);
    }

    #[test]
    fn tools_config_deserialization_with_compaction() {
        let toml_str = r#"
            context_window_tokens = 200000
            compaction_threshold = 0.75
            compaction_keep_recent = 4
        "#;
        let config: ToolsConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.context_window_tokens, Some(200000));
        assert!(
            (config.compaction_threshold - 0.75).abs() < 0.01,
            "compaction_threshold should be 0.75"
        );
        assert_eq!(config.compaction_keep_recent, 4);
    }
}
