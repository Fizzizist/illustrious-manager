use dirs;
use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{self, Write};
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

fn default_max_tool_result_bytes() -> u64 {
    65536
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

fn default_max_context_window_len() -> u32 {
    0
}

fn default_compaction_role() -> String {
    "compaction".to_string()
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CompactionConfig {
    #[serde(default = "default_compaction_role")]
    pub role: String,
    #[serde(default = "default_max_context_window_len")]
    pub max_context_window_len: u32,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        CompactionConfig {
            role: default_compaction_role(),
            max_context_window_len: default_max_context_window_len(),
        }
    }
}

fn default_max_token_retries() -> u32 {
    3
}

fn default_retry_config() -> RetryConfig {
    RetryConfig {
        max_retries: 3,
        initial_delay_ms: 1000,
        max_delay_ms: 8000,
        max_token_retries: 3,
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq)]
#[serde(default = "default_retry_config")]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    #[serde(default = "default_max_token_retries")]
    pub max_token_retries: u32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        default_retry_config()
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ToolsConfig {
    #[serde(default = "default_confirmation")]
    pub confirmation: ConfirmationMode,
    #[serde(default = "default_sandbox_root")]
    pub sandbox_root: String,
    #[serde(default = "default_max_tool_iterations")]
    pub max_tool_iterations: u32,
    #[serde(default = "default_max_tool_result_bytes")]
    pub max_tool_result_bytes: u64,
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
            max_tool_result_bytes: default_max_tool_result_bytes(),
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

const CONFIG_TEMPLATE: &str = r#"# Which backend to use: "vertex", "zai", "ollama", "opencode_go", "anthropic", or "openai_compat"
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
# Optional: override max_tokens for this backend
# max_tokens = 8192
# Whether models on this backend support image input (default: true for Vertex/Claude).
# When true, the image_viewer tool is registered for this role.
# vision = true

[zai]
# Required: your z.ai API key
api_key = ""
# Model to use
model = "glm-5.1"
# Whether models on this backend support image input (default: false for z.ai).
# vision = false

# [ollama]
# Required: your Ollama API key
# api_key = ""
# Model to use
# model = "gpt-oss:120b"
# Base URL for Ollama API (change for self-hosted)
# base_url = "https://ollama.com/api/chat"
# Optional: override max_tokens for this backend
# max_tokens = 8192
# Whether models on this backend support image input (default: false for Ollama).
# vision = false

# [openai_compat]
# Required: base URL of the OpenAI-compatible endpoint (without /chat/completions)
# base_url = "https://vllm.k8s.dc.rxrx.io/v1"
# Optional: API key (omit or leave empty for unauthenticated endpoints)
# api_key = ""
# Model to use
# model = "Qwen/Qwen3-32B-FP8"
# Optional: override max_tokens for this backend
# max_tokens = 16384
# Reasoning style: "none", "zai_enable_thinking", "qwen_chat_template", "default"
# reasoning = "qwen_chat_template"
# Whether models on this endpoint support image input (default: false).
# vision = false

# [opencode_go]
# OpenCode Go — dual-protocol backend. Triages model name against config-driven
# lists to route to either OpenAI Chat Completions or Anthropic Messages.
# Required: your OpenCode Go API key
# api_key = ""
# Base URL (defaults to the OpenCode Go endpoint)
# base_url = "https://opencode.ai/zen/go/v1"
# Default model (used when synthesizing the "default" role)
# model = "kimi-k3"
# Models that use the OpenAI Chat Completions protocol
# Constraints: No model may appear in both lists; every model must be in exactly one list.
# openai_models = ["grok-code-fast", "grok-code", "glm-4.6-code", "kimi-k2-code", "deepseek-v3.2-code", "mimo-7b-code"]
# Models that use the Anthropic Messages protocol
# anthropic_models = ["minimax-m1", "qwen3-coder-plus"]
# Optional: override max_tokens for this backend
# max_tokens = 16384
# Reasoning style for OpenAI-protocol models: "none", "zai_enable_thinking", "qwen_chat_template", "default"
# reasoning = "default"
# Whether models on this backend support image input (default: true for OpenCode Go).
# vision = true

# [anthropic]
# Direct Anthropic Messages API backend.
# Required: your Anthropic API key
# api_key = ""
# Base URL (defaults to the Anthropic API endpoint)
# base_url = "https://api.anthropic.com/v1"
# Model to use (defaults to claude-opus-4-8)
# model = "claude-opus-4-8"
# Optional: override max_tokens for this backend (agent default is 8192)
# max_tokens = 65536
# Whether models on this backend support image input (default: true for Anthropic).
# vision = true

# [compaction]
# Role name used for compaction sub-agents. Defaults to "compaction".
# If the role isn't defined in [models], falls back to "default".
# role = "compaction"
# Context window length threshold for auto-compaction (0 = disabled).
# When input_tokens exceeds this value, the agent automatically compacts the conversation.
# max_context_window_len = 0

# [retry]
# Maximum number of retry attempts for transient HTTP errors (5xx, 429).
# max_retries = 3
# Initial delay in milliseconds before the first retry.
# initial_delay_ms = 1000
# Maximum delay in milliseconds for backoff (caps exponential growth).
# max_delay_ms = 8000
# Maximum number of times the agent retries when the model hits its max_tokens
# output limit. Each retry re-sends the full conversation history (including the
# truncated response and error message), so input token cost grows per retry.
# max_token_retries = 3

# [tools]
# When to prompt for confirmation before executing a tool: Always, WriteOnly, or Never
# confirmation = "WriteOnly"
# Directory tools are allowed to read/write (resolved to absolute path at startup)
# sandbox_root = "."
# Note: /tmp (on Unix) and the platform temp directory are always accessible
# to edit_file/write_file as additional roots, regardless of sandbox_root.
# This is not configurable; bash already has unrestricted /tmp access.
# Maximum number of tool-use iterations per agent turn
# max_tool_iterations = 25
# Maximum byte cap for tool results. Results exceeding this cap are truncated
# with head+tail preserved and a sentinel message. Set to 0 for unlimited.
# max_tool_result_bytes = 65536
# Shell commands that may be executed without a denylist match
# bash_allowlist = ["cat", "ls", "grep", "find", "head", "tail", "wc", "tree"]
# Shell commands that are always blocked
# bash_denylist = ["rm", "wget", "sudo", "chmod", "chown"]

# Extended thinking configuration for models that support it (Vertex AI / Anthropic).
# [thinking]
# Whether thinking is enabled (default: true)
# enabled = true
# Thinking mode: "budget" (fixed token count) or "adaptive" (model decides)
# For budget mode, set the token budget in the mode field:
#   mode = { type = "budget", tokens = 8192 }
# For adaptive mode:
#   mode = { type = "adaptive" }
# Whether the API emits thinking content: "summarized" (default) or "omitted".
# Vertex defaults to "summarized" for visibility on models that would otherwise
# suppress thinking_delta events (e.g. Opus 4.7).
# display = "summarized"

# Named model roles for multi-agent workflows.
# When absent, a "default" role is synthesized from the top-level backend
# and the matching [vertex]/[zai]/[ollama] model field above.
# [models.thinking]
# backend = "vertex"
# model = "claude-opus-4-7"
# [models.implement]
# backend = "vertex"
# model = "claude-sonnet-4-6"
# [models.compaction]
# backend = "vertex"
# model = "claude-haiku-4-5"
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
    pub openai_compat: Option<OpenAiCompatConfigToml>,
    #[serde(default)]
    pub opencode_go: Option<OpenCodeGoConfig>,
    #[serde(default)]
    pub anthropic: Option<AnthropicConfig>,
    #[serde(default)]
    pub tools: ToolsConfig,
    /// Named model roles. When empty, a `default` role is synthesized from
    /// the top-level `backend` + `[vertex]`/`[zai]` blocks for back-compat.
    #[serde(default, rename = "models")]
    pub models: BTreeMap<String, ModelRole>,
    #[serde(default)]
    pub thinking: Option<crate::types::ThinkingConfig>,
    #[serde(default)]
    pub compaction: CompactionConfig,
    #[serde(default)]
    pub retry: RetryConfig,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct VertexConfig {
    pub project: String,
    #[serde(default = "default_region")]
    pub region: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default = "default_vertex_vision")]
    pub vision: bool,
}

fn default_vertex_vision() -> bool {
    crate::backend::defaults::VISION_VERTEX
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct OllamaConfig {
    #[serde(skip_serializing)]
    pub api_key: String,
    #[serde(default = "default_ollama_model")]
    pub model: String,
    #[serde(default = "default_ollama_base_url")]
    pub base_url: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default = "default_ollama_vision")]
    pub vision: bool,
}

fn default_ollama_vision() -> bool {
    crate::backend::defaults::VISION_OLLAMA
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
    #[serde(default = "default_zai_vision")]
    pub vision: bool,
}

fn default_zai_vision() -> bool {
    crate::backend::defaults::VISION_ZAI
}

/// Reasoning style for OpenAI-compatible backends, mirroring `openai_compat::ReasoningStyle`.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningStyleConfig {
    #[default]
    None,
    ZaiEnableThinking,
    QwenChatTemplate,
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct OpenAiCompatConfigToml {
    pub base_url: String,
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
    #[serde(default = "default_openai_compat_model")]
    pub model: String,
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub reasoning: ReasoningStyleConfig,
    #[serde(default = "default_openai_compat_vision")]
    pub vision: bool,
}

fn default_openai_compat_vision() -> bool {
    crate::backend::defaults::VISION_OPENAI_COMPAT
}

fn default_openai_compat_model() -> String {
    String::new()
}

pub const OPENCODE_GO_DEFAULT_BASE_URL: &str = "https://opencode.ai/zen/go/v1";

pub const ANTHROPIC_DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

fn default_anthropic_base_url() -> String {
    ANTHROPIC_DEFAULT_BASE_URL.to_string()
}

fn default_anthropic_model() -> String {
    "claude-opus-4-8".to_string()
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct AnthropicConfig {
    #[serde(skip_serializing)]
    pub api_key: String,
    #[serde(default = "default_anthropic_base_url")]
    pub base_url: String,
    #[serde(default = "default_anthropic_model")]
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default = "default_anthropic_vision")]
    pub vision: bool,
}

fn default_anthropic_vision() -> bool {
    crate::backend::defaults::VISION_ANTHROPIC
}

fn default_opencode_go_base_url() -> String {
    OPENCODE_GO_DEFAULT_BASE_URL.to_string()
}

fn default_opencode_go_model() -> String {
    "kimi-k3".to_string()
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct OpenCodeGoConfig {
    #[serde(skip_serializing)]
    pub api_key: String,
    #[serde(default = "default_opencode_go_base_url")]
    pub base_url: String,
    #[serde(default = "default_opencode_go_model")]
    pub model: String,
    #[serde(default)]
    pub openai_models: Vec<String>,
    #[serde(default)]
    pub anthropic_models: Vec<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub reasoning: ReasoningStyleConfig,
    #[serde(default = "default_opencode_go_vision")]
    pub vision: bool,
}

fn default_opencode_go_vision() -> bool {
    crate::backend::defaults::VISION_OPENCODE_GO
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    OpenAi,
    Anthropic,
}

impl OpenCodeGoConfig {
    pub fn protocol_for(&self, model: &str) -> anyhow::Result<Protocol> {
        let in_openai = self.openai_models.iter().any(|m| m == model);
        let in_anthropic = self.anthropic_models.iter().any(|m| m == model);
        if in_openai {
            Ok(Protocol::OpenAi)
        } else if in_anthropic {
            Ok(Protocol::Anthropic)
        } else {
            anyhow::bail!(
                "Model '{model}' is not in either openai_models or anthropic_models. \
                 Add it to the appropriate list in your [opencode_go] config section.\n\
                 OpenAI-protocol models: {:?}\n\
                 Anthropic-protocol models: {:?}",
                self.openai_models,
                self.anthropic_models,
            )
        }
    }

    pub fn find_duplicate_model(&self) -> Option<String> {
        let openai_set: std::collections::HashSet<&str> =
            self.openai_models.iter().map(String::as_str).collect();
        self.anthropic_models
            .iter()
            .find(|m| openai_set.contains(m.as_str()))
            .cloned()
    }
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
        // Intentional stderr write: config creation notice runs before the TUI starts.
        writeln!(
            io::stderr(),
            "Created default config at: {}",
            path.display()
        )?;
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
            "openai_compat" => self
                .openai_compat
                .as_ref()
                .map(|o| o.model.clone())
                .unwrap_or_default(),
            "opencode_go" => self
                .opencode_go
                .as_ref()
                .map(|o| o.model.clone())
                .unwrap_or_else(default_opencode_go_model),
            "anthropic" => self
                .anthropic
                .as_ref()
                .map(|a| a.model.clone())
                .unwrap_or_else(default_anthropic_model),
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
        "openai_compat" => {
            if let Some(m) = model
                && let Some(ref mut oc) = config.openai_compat
            {
                oc.model = m.to_string();
            }
        }
        "opencode_go" => {
            if let Some(m) = model
                && let Some(ref mut oc) = config.opencode_go
            {
                oc.model = m.to_string();
            }
        }
        "anthropic" => {
            if let Some(m) = model
                && let Some(ref mut a) = config.anthropic
            {
                a.model = m.to_string();
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
pub fn generate_config_message(config: &AppConfig) -> String {
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
        toml::Value::Float(f) => f.to_string(),
        toml::Value::Boolean(b) => b.to_string(),
        other => other.to_string(),
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
        "openai_compat" => match &config.openai_compat {
            None => {
                bail!(
                    "openai_compat backend configuration is missing. Add a [openai_compat] section to your config file."
                );
            }
            Some(oc) if oc.base_url.is_empty() => {
                bail!(
                    "base_url is required for openai_compat backend. Set it in your [openai_compat] config section."
                );
            }
            Some(oc)
                if oc
                    .base_url
                    .trim_end_matches('/')
                    .ends_with("/chat/completions") =>
            {
                bail!(
                    "base_url must not include '/chat/completions' — provide the base URL only                      (e.g. 'https://example.com/v1')."
                );
            }
            Some(oc) if oc.base_url.contains('?') => {
                bail!(
                    "base_url must not contain a query string. Provide only the base URL                      (e.g. 'https://example.com/v1')."
                );
            }
            _ => {}
        },
        "opencode_go" => match &config.opencode_go {
            None => {
                bail!(
                    "opencode_go backend configuration is missing. Add a [opencode_go] section to your config file."
                );
            }
            Some(oc) if oc.api_key.is_empty() => {
                bail!(
                    "API key is required for opencode_go backend. Set it in your [opencode_go] config section."
                );
            }
            Some(oc)
                if oc
                    .base_url
                    .trim_end_matches('/')
                    .ends_with("/chat/completions") =>
            {
                bail!("base_url must not include '/chat/completions' — provide the base URL only.");
            }
            Some(oc) => {
                if let Some(duplicate) = oc.find_duplicate_model() {
                    bail!(
                        "Model '{}' appears in both openai_models and anthropic_models — \
                         each model must belong to exactly one protocol list.",
                        duplicate
                    );
                }
            }
        },
        "anthropic" => match &config.anthropic {
            None => {
                bail!(
                    "anthropic backend configuration is missing. Add a [anthropic] section to your config file."
                );
            }
            Some(a) if a.api_key.is_empty() => {
                bail!(
                    "API key is required for anthropic backend. Set it in your [anthropic] config section."
                );
            }
            Some(a) if a.base_url.trim().is_empty() => {
                bail!(
                    "base_url is required for anthropic backend. Set it in your [anthropic] config section \
                     (e.g. '{}').",
                    ANTHROPIC_DEFAULT_BASE_URL,
                );
            }
            Some(a) if a.base_url.trim_end_matches('/').ends_with("/messages") => {
                bail!(
                    "base_url must not include '/messages' — provide the base URL only \
                     (e.g. '{}') and the backend will append the path automatically.",
                    ANTHROPIC_DEFAULT_BASE_URL,
                );
            }
            Some(a) if a.base_url.contains('?') => {
                bail!(
                    "base_url must not contain a query string. Provide only the base URL \
                     (e.g. '{}').",
                    ANTHROPIC_DEFAULT_BASE_URL,
                );
            }
            _ => {}
        },
        _ => {
            bail!(
                "Invalid backend '{}'. Supported backends are: vertex, zai, ollama, openai_compat, anthropic, opencode_go",
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
            "openai_compat" => match &config.openai_compat {
                None => {
                    bail!(
                        "Model role '{name}' uses backend 'openai_compat' but no [openai_compat] section is present."
                    );
                }
                Some(oc) if oc.base_url.is_empty() => {
                    bail!(
                        "Model role '{name}' uses backend 'openai_compat' but [openai_compat].base_url is not configured."
                    );
                }
                _ => {}
            },
            "opencode_go" => match &config.opencode_go {
                None => {
                    bail!(
                        "Model role '{name}' uses backend 'opencode_go' but no [opencode_go] section is present."
                    );
                }
                Some(oc) if oc.api_key.is_empty() => {
                    bail!(
                        "Model role '{name}' uses backend 'opencode_go' but [opencode_go].api_key is not configured."
                    );
                }
                Some(oc) => {
                    if let Some(dup) = oc.find_duplicate_model() {
                        bail!(
                            "Model '{dup}' appears in both openai_models and anthropic_models — \
                             each model must belong to exactly one protocol list."
                        );
                    }
                    if let Err(e) = oc.protocol_for(&role.model) {
                        bail!("Model role '{name}' {e}");
                    }
                }
            },
            "anthropic" => match &config.anthropic {
                None => {
                    bail!(
                        "Model role '{name}' uses backend 'anthropic' but no [anthropic] section is present."
                    );
                }
                Some(a) if a.api_key.is_empty() => {
                    bail!(
                        "Model role '{name}' uses backend 'anthropic' but [anthropic].api_key is not configured."
                    );
                }
                Some(a) if a.base_url.trim().is_empty() => {
                    bail!(
                        "Model role '{name}' uses backend 'anthropic' but [anthropic].base_url is not configured."
                    );
                }
                Some(a) if a.base_url.trim_end_matches('/').ends_with("/messages") => {
                    bail!(
                        "Model role '{name}' uses backend 'anthropic' but [anthropic].base_url \
                         must not include '/messages' — the backend appends the path automatically."
                    );
                }
                Some(a) if a.base_url.contains('?') => {
                    bail!(
                        "Model role '{name}' uses backend 'anthropic' but [anthropic].base_url \
                         must not contain a query string."
                    );
                }
                _ => {}
            },
            other => {
                bail!(
                    "Model role '{name}' references unknown backend '{other}'. Supported: vertex, zai, ollama, openai_compat, anthropic, opencode_go"
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
            max_tool_result_bytes = 0
            bash_allowlist = ["echo"]
            bash_denylist = ["curl"]
        "#;
        let config: ToolsConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.confirmation, ConfirmationMode::Always);
        assert_eq!(config.sandbox_root, "/tmp/sandbox");
        assert_eq!(config.max_tool_iterations, 10);
        assert_eq!(config.max_tool_result_bytes, 0);
        assert_eq!(config.bash_allowlist, vec!["echo"]);
        assert_eq!(config.bash_denylist, vec!["curl"]);
    }

    #[test]
    fn default_tools_config_has_max_tool_result_bytes() {
        let config = ToolsConfig::default();
        assert_eq!(config.max_tool_result_bytes, 65536);
    }

    #[test]
    fn custom_max_tool_result_bytes_overrides_default() {
        let toml_str = r#"
            max_tool_result_bytes = 0
        "#;
        let config: ToolsConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.max_tool_result_bytes, 0);
    }

    #[test]
    fn default_compaction_config_includes_max_context_window_len() {
        let config = CompactionConfig::default();
        assert_eq!(config.max_context_window_len, 0);
    }

    #[test]
    fn max_context_window_len_deserializes_from_toml() {
        let toml_str = r#"
            max_context_window_len = 50000
        "#;
        let config: CompactionConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.max_context_window_len, 50000);
    }

    #[test]
    fn max_context_window_len_zero_by_default_in_toml() {
        let toml_str = r#"
            role = "fast"
        "#;
        let config: CompactionConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.max_context_window_len, 0);
    }

    #[test]
    fn compaction_config_role_defaults_to_compaction() {
        let config = CompactionConfig::default();
        assert_eq!(config.role, "compaction");
    }

    #[test]
    fn compaction_config_role_overrides_from_toml() {
        let toml_str = r#"
            role = "fast"
            max_context_window_len = 50000
        "#;
        let config: CompactionConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.role, "fast");
        assert_eq!(config.max_context_window_len, 50000);
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
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "my-project".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_zai_backend_with_missing_config_errors() {
        let config = AppConfig {
            backend: "zai".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: Some(ZaiConfig {
                api_key: "".to_string(),
                model: "glm-5.1".to_string(),
                vision: false,
            }),
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: Some(ZaiConfig {
                api_key: "test-key".to_string(),
                model: "glm-5.1".to_string(),
                vision: false,
            }),
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_invalid_backend_errors() {
        let config = AppConfig {
            backend: "invalid".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid backend"));
    }

    #[test]
    fn generate_config_message_contains_vertex_backend_settings() {
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "my-gcp-project".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let msg = generate_config_message(&config);
        assert!(msg.contains("vertex"), "should mention backend name");
        assert!(msg.contains("my-gcp-project"), "should mention project");
        assert!(msg.contains("us-east5"), "should mention region");
        assert!(
            msg.contains("claude-sonnet-4-20250514"),
            "should mention model"
        );
    }

    #[test]
    fn generate_config_message_contains_zai_backend_settings() {
        let config = AppConfig {
            backend: "zai".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: Some(ZaiConfig {
                api_key: "secret-key".to_string(),
                model: "glm-5.1".to_string(),
                vision: false,
            }),
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let msg = generate_config_message(&config);
        assert!(msg.contains("zai"), "should mention backend name");
        assert!(msg.contains("glm-5.1"), "should mention zai model");
        assert!(!msg.contains("secret-key"), "should not leak API key");
    }

    #[test]
    fn generate_config_message_contains_tool_config() {
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig {
                confirmation: ConfirmationMode::Always,
                sandbox_root: "/tmp/sandbox".to_string(),
                max_tool_iterations: 10,
                ..Default::default()
            },
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let msg = generate_config_message(&config);
        assert!(msg.contains("Always"), "should mention confirmation mode");
        assert!(msg.contains("/tmp/sandbox"), "should mention sandbox root");
        assert!(msg.contains("10"), "should mention max tool iterations");
    }

    #[test]
    fn generate_config_message_contains_sessions_dir() {
        let sessions_dir = std::env::temp_dir().join("my-sessions");
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: sessions_dir.clone(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let msg = generate_config_message(&config);
        assert!(
            msg.contains(&sessions_dir.display().to_string()),
            "should mention sessions directory"
        );
    }

    #[test]
    fn generate_config_message_contains_markdown_heading() {
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let msg = generate_config_message(&config);
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
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: Some(ZaiConfig {
                api_key: "".to_string(),
                model: "glm-5.1".to_string(),
                vision: false,
            }),
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
                max_tokens: None,
                vision: false,
            }),
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "http://localhost:11434/api/chat".to_string(),
                max_tokens: None,
                vision: false,
            }),
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_ollama_backend_with_valid_config_succeeds() {
        let config = AppConfig {
            backend: "ollama".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "test-key".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
                max_tokens: None,
                vision: false,
            }),
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_ok());
    }

    #[test]
    fn apply_overrides_ollama_model() {
        let mut config = AppConfig {
            backend: "ollama".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "test-key".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
                max_tokens: None,
                vision: false,
            }),
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "p".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: Some(ZaiConfig {
                api_key: "k".to_string(),
                model: "glm-5.1".to_string(),
                vision: false,
            }),
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "k".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
                max_tokens: None,
                vision: false,
            }),
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "test-key".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
                max_tokens: None,
                vision: false,
            }),
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
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
    fn generate_config_message_contains_ollama_backend_settings() {
        let config = AppConfig {
            backend: "ollama".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: Some(OllamaConfig {
                api_key: "secret-key".to_string(),
                model: "gpt-oss:120b".to_string(),
                base_url: "https://ollama.com/api/chat".to_string(),
                max_tokens: None,
                vision: false,
            }),
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let msg = generate_config_message(&config);
        assert!(msg.contains("ollama"), "should mention backend name");
        assert!(msg.contains("gpt-oss:120b"), "should mention ollama model");
        assert!(msg.contains("ollama.com"), "should mention base_url");
        assert!(!msg.contains("secret-key"), "should not leak API key");
    }

    // ── Thinking config tests ─────────────────────────────────────────────

    #[test]
    fn thinking_section_parses_from_toml() {
        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "my-project"
            [thinking]
            enabled = true
            mode = { type = "adaptive" }
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        let thinking = config.thinking.expect("thinking should be present");
        assert!(thinking.enabled);
        assert_eq!(thinking.mode, crate::types::ThinkingMode::Adaptive);
    }

    #[test]
    fn thinking_section_defaults_to_none() {
        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "my-project"
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert!(
            config.thinking.is_none(),
            "thinking should be None when not specified"
        );
    }

    #[test]
    fn thinking_budget_mode_parses_correctly() {
        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "my-project"
            [thinking]
            mode = { type = "budget", tokens = 16384 }
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        let thinking = config.thinking.expect("thinking should be present");
        assert_eq!(
            thinking.mode,
            crate::types::ThinkingMode::Budget { tokens: 16384 }
        );
    }

    #[test]
    fn thinking_adaptive_mode_parses_correctly() {
        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "my-project"
            [thinking]
            mode = { type = "adaptive" }
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        let thinking = config.thinking.expect("thinking should be present");
        assert_eq!(thinking.mode, crate::types::ThinkingMode::Adaptive);
    }

    #[test]
    fn thinking_section_with_display_summarized_parses() {
        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "my-project"
            [thinking]
            mode = { type = "adaptive" }
            display = "summarized"
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(
            config.thinking.expect("thinking present").display,
            Some(crate::types::ThinkingDisplay::Summarized)
        );
    }

    #[test]
    fn thinking_section_with_display_omitted_parses() {
        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "my-project"
            [thinking]
            mode = { type = "adaptive" }
            display = "omitted"
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(
            config.thinking.expect("thinking present").display,
            Some(crate::types::ThinkingDisplay::Omitted)
        );
    }

    #[test]
    fn thinking_section_without_display_defaults_to_none() {
        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "my-project"
            [thinking]
            mode = { type = "adaptive" }
            enabled = true
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert!(config.thinking.expect("thinking present").display.is_none());
    }

    #[test]
    fn thinking_section_with_unknown_display_value_fails_to_parse() {
        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "my-project"
            [thinking]
            mode = { type = "adaptive" }
            display = "hidden"
        "#;
        let result: Result<AppConfig, _> = toml::from_str(toml_str);
        assert!(
            result.is_err(),
            "unknown display value must fail deserialization, got {:?}",
            result
        );
    }

    #[test]
    fn compaction_role_defaults_to_compaction() {
        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "my-project"
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.compaction.role, "compaction");
    }

    #[test]
    fn compaction_role_overrides_from_toml() {
        let toml_str = r#"
            backend = "vertex"
            [compaction]
            role = "fast"
            [vertex]
            project = "my-project"
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.compaction.role, "fast");
    }

    // ── openai_compat config tests ────────────────────────────────────────

    #[test]
    fn validate_openai_compat_backend_with_missing_config_errors() {
        let config = AppConfig {
            backend: "openai_compat".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("[openai_compat]"),
            "error should mention [openai_compat]"
        );
    }

    #[test]
    fn validate_openai_compat_backend_with_empty_base_url_errors() {
        let config = AppConfig {
            backend: "openai_compat".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: Some(OpenAiCompatConfigToml {
                base_url: "".to_string(),
                api_key: None,
                model: "Qwen/Qwen3-32B".to_string(),
                max_tokens: None,
                reasoning: ReasoningStyleConfig::None,
                vision: false,
            }),
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("base_url"),
            "error should mention base_url"
        );
    }

    #[test]
    fn validate_openai_compat_backend_with_valid_config_succeeds() {
        let config = AppConfig {
            backend: "openai_compat".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: Some(OpenAiCompatConfigToml {
                base_url: "https://vllm.example.com/v1".to_string(),
                api_key: None,
                model: "Qwen/Qwen3-32B".to_string(),
                max_tokens: None,
                reasoning: ReasoningStyleConfig::None,
                vision: false,
            }),
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(
            result.is_ok(),
            "valid openai_compat config should pass: {:?}",
            result
        );
    }

    #[test]
    fn normalize_back_compat_synthesizes_default_role_for_openai_compat() {
        let mut config = AppConfig {
            backend: "openai_compat".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: Some(OpenAiCompatConfigToml {
                base_url: "https://example.com/v1".to_string(),
                api_key: None,
                model: "Qwen/Qwen3-32B".to_string(),
                max_tokens: None,
                reasoning: ReasoningStyleConfig::None,
                vision: false,
            }),
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        config.normalize_back_compat();
        assert!(
            config.models.contains_key("default"),
            "default role should be synthesized for openai_compat backend"
        );
        assert_eq!(config.models["default"].backend, "openai_compat");
        assert_eq!(config.models["default"].model, "Qwen/Qwen3-32B");
    }

    #[test]
    fn legacy_openai_compat_config_synthesizes_default_role() {
        let toml_str = r#"
            backend = "openai_compat"
            [vertex]
            project = ""
            [openai_compat]
            base_url = "https://example.com/v1"
            model = "Qwen/Qwen3-32B"
        "#;
        let mut config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        config.normalize_back_compat();
        assert!(
            config.models.contains_key("default"),
            "default role should be synthesized from [openai_compat] block"
        );
        let default_role = &config.models["default"];
        assert_eq!(default_role.backend, "openai_compat");
        assert_eq!(default_role.model, "Qwen/Qwen3-32B");
    }

    #[test]
    fn reasoning_style_config_defaults_to_none() {
        let toml_str = r#"
            base_url = "https://example.com/v1"
            model = "Qwen/Qwen3-32B"
        "#;
        let config: OpenAiCompatConfigToml = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.reasoning, ReasoningStyleConfig::None);
    }

    #[test]
    fn reasoning_style_config_parses_qwen_chat_template() {
        let toml_str = r#"
            base_url = "https://example.com/v1"
            model = "Qwen/Qwen3-32B"
            reasoning = "qwen_chat_template"
        "#;
        let config: OpenAiCompatConfigToml = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.reasoning, ReasoningStyleConfig::QwenChatTemplate);
    }

    #[test]
    fn apply_overrides_openai_compat_model() {
        // Finding 9d: apply_overrides for openai_compat must work analogously
        // to the vertex/zai/ollama variants.
        let mut config = AppConfig {
            backend: "openai_compat".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: Some(OpenAiCompatConfigToml {
                base_url: "https://example.com/v1".to_string(),
                api_key: None,
                model: "Qwen/Qwen3-32B".to_string(),
                max_tokens: None,
                reasoning: ReasoningStyleConfig::None,
                vision: false,
            }),
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        config.normalize_back_compat();
        apply_overrides(&mut config, None, None, Some("Qwen/Qwen3-235B"));
        assert_eq!(
            config
                .openai_compat
                .as_ref()
                .expect("openai_compat present")
                .model,
            "Qwen/Qwen3-235B",
            "model override should be applied to openai_compat config"
        );
        let resolved = config
            .resolve_role("default")
            .expect("default role exists after normalize_back_compat");
        assert_eq!(
            resolved.model, "Qwen/Qwen3-235B",
            "--model flag must propagate into models[\"default\"]"
        );
    }

    #[test]
    fn validate_openai_compat_backend_with_chat_completions_suffix_errors() {
        let config = AppConfig {
            backend: "openai_compat".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: Some(OpenAiCompatConfigToml {
                base_url: "https://example.com/v1/chat/completions".to_string(),
                api_key: None,
                model: "Qwen/Qwen3-32B".to_string(),
                max_tokens: None,
                reasoning: ReasoningStyleConfig::None,
                vision: false,
            }),
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("chat/completions"),
            "error should mention chat/completions; got: {msg}"
        );
    }

    #[test]
    fn validate_openai_compat_backend_with_query_string_errors() {
        let config = AppConfig {
            backend: "openai_compat".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: Some(OpenAiCompatConfigToml {
                base_url: "https://example.com/v1?token=secret".to_string(),
                api_key: None,
                model: "Qwen/Qwen3-32B".to_string(),
                max_tokens: None,
                reasoning: ReasoningStyleConfig::None,
                vision: false,
            }),
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("query string"),
            "error should mention query string; got: {msg}"
        );
    }

    // ── RetryConfig tests ────────────────────────────────────────────────

    #[test]
    fn retry_config_defaults() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_delay_ms, 1000);
        assert_eq!(config.max_delay_ms, 8000);
        assert_eq!(config.max_token_retries, 3);
    }

    #[test]
    fn retry_config_parses_from_toml() {
        let toml_str = r#"
            max_retries = 5
            initial_delay_ms = 500
            max_delay_ms = 20000
            max_token_retries = 7
        "#;
        let config: RetryConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_delay_ms, 500);
        assert_eq!(config.max_delay_ms, 20000);
        assert_eq!(config.max_token_retries, 7);
    }

    #[test]
    fn retry_config_missing_section_uses_defaults() {
        let toml_str = r#"
            backend = "vertex"
            [vertex]
            project = "my-project"
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.retry.max_retries, 3);
        assert_eq!(config.retry.initial_delay_ms, 1000);
        assert_eq!(config.retry.max_delay_ms, 8000);
        assert_eq!(config.retry.max_token_retries, 3);
    }

    #[test]
    fn retry_config_partial_overrides_keep_default_for_max_token_retries() {
        let toml_str = r#"
            max_retries = 10
        "#;
        let config: RetryConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.max_retries, 10);
        assert_eq!(
            config.max_token_retries, 3,
            "should default when not specified"
        );
    }

    // ── opencode_go config tests ──────────────────────────────────────────

    #[test]
    fn validate_opencode_go_backend_with_missing_config_errors() {
        let config = AppConfig {
            backend: "opencode_go".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("[opencode_go]"),
            "error should mention [opencode_go]"
        );
    }

    #[test]
    fn validate_opencode_go_backend_with_empty_api_key_errors() {
        let config = AppConfig {
            backend: "opencode_go".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: Some(OpenCodeGoConfig {
                api_key: "".to_string(),
                base_url: "https://opencode.ai/zen/go/v1".to_string(),
                model: "grok-code".to_string(),
                openai_models: vec!["grok-code".to_string()],
                anthropic_models: vec![],
                max_tokens: None,
                reasoning: ReasoningStyleConfig::Default,
                vision: false,
            }),
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key"));
    }

    #[test]
    fn validate_opencode_go_backend_with_valid_config_succeeds() {
        let config = AppConfig {
            backend: "opencode_go".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: Some(OpenCodeGoConfig {
                api_key: "test-key".to_string(),
                base_url: "https://opencode.ai/zen/go/v1".to_string(),
                model: "grok-code".to_string(),
                openai_models: vec!["grok-code".to_string()],
                anthropic_models: vec!["minimax-m1".to_string()],
                max_tokens: None,
                reasoning: ReasoningStyleConfig::Default,
                vision: false,
            }),
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(
            result.is_ok(),
            "valid opencode_go config should pass: {result:?}"
        );
    }

    #[test]
    fn validate_opencode_go_rejects_duplicate_models_across_lists() {
        let config = AppConfig {
            backend: "opencode_go".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: Some(OpenCodeGoConfig {
                api_key: "test-key".to_string(),
                base_url: "https://opencode.ai/zen/go/v1".to_string(),
                model: "grok-code".to_string(),
                openai_models: vec!["grok-code".to_string()],
                anthropic_models: vec!["grok-code".to_string()],
                max_tokens: None,
                reasoning: ReasoningStyleConfig::Default,
                vision: false,
            }),
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("grok-code") && msg.contains("both"),
            "error should mention the duplicate model and 'both'; got: {msg}"
        );
    }

    #[test]
    fn normalize_back_compat_synthesizes_default_role_from_opencode_go() {
        let mut config = AppConfig {
            backend: "opencode_go".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: Some(OpenCodeGoConfig {
                api_key: "test-key".to_string(),
                base_url: "https://opencode.ai/zen/go/v1".to_string(),
                model: "grok-code-fast".to_string(),
                openai_models: vec!["grok-code-fast".to_string()],
                anthropic_models: vec![],
                max_tokens: None,
                reasoning: ReasoningStyleConfig::Default,
                vision: false,
            }),
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        config.normalize_back_compat();
        assert!(
            config.models.contains_key("default"),
            "default role should be synthesized for opencode_go backend"
        );
        assert_eq!(config.models["default"].backend, "opencode_go");
        assert_eq!(config.models["default"].model, "grok-code-fast");
    }

    #[test]
    fn apply_overrides_opencode_go_model() {
        let mut config = AppConfig {
            backend: "opencode_go".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: Some(OpenCodeGoConfig {
                api_key: "test-key".to_string(),
                base_url: "https://opencode.ai/zen/go/v1".to_string(),
                model: "grok-code-fast".to_string(),
                openai_models: vec!["grok-code-fast".to_string(), "grok-code".to_string()],
                anthropic_models: vec![],
                max_tokens: None,
                reasoning: ReasoningStyleConfig::Default,
                vision: false,
            }),
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        config.normalize_back_compat();
        apply_overrides(&mut config, None, None, Some("grok-code"));
        assert_eq!(
            config
                .opencode_go
                .as_ref()
                .expect("opencode_go present")
                .model,
            "grok-code",
            "model override should be applied to opencode_go config"
        );
        let resolved = config
            .resolve_role("default")
            .expect("default role exists after normalize_back_compat");
        assert_eq!(
            resolved.model, "grok-code",
            "--model flag must propagate into models[\"default\"]"
        );
    }

    #[test]
    fn validate_opencode_go_role_rejects_model_not_in_either_list() {
        let mut models = BTreeMap::new();
        models.insert(
            "my-role".to_string(),
            ModelRole {
                backend: "opencode_go".to_string(),
                model: "unknown-model".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: Some(OpenCodeGoConfig {
                api_key: "test-key".to_string(),
                base_url: "https://opencode.ai/zen/go/v1".to_string(),
                model: "grok-code".to_string(),
                openai_models: vec!["grok-code".to_string()],
                anthropic_models: vec!["minimax-m1".to_string()],
                max_tokens: None,
                reasoning: ReasoningStyleConfig::Default,
                vision: false,
            }),
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("unknown-model"),
            "error should mention the unknown model; got: {msg}"
        );
    }

    #[test]
    fn validate_opencode_go_role_rejects_duplicate_model_in_both_lists() {
        let mut models = BTreeMap::new();
        models.insert(
            "my-role".to_string(),
            ModelRole {
                backend: "opencode_go".to_string(),
                model: "grok-code".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: Some(OpenCodeGoConfig {
                api_key: "test-key".to_string(),
                base_url: "https://opencode.ai/zen/go/v1".to_string(),
                model: "grok-code".to_string(),
                openai_models: vec!["grok-code".to_string()],
                anthropic_models: vec!["grok-code".to_string()],
                max_tokens: None,
                reasoning: ReasoningStyleConfig::Default,
                vision: false,
            }),
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("grok-code") && msg.contains("both"),
            "error should mention duplicate model and 'both'; got: {msg}"
        );
    }

    #[test]
    fn vertex_config_max_tokens_defaults_to_none() {
        let toml_str = r#"
            [vertex]
            project = "my-project"
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert!(config.vertex.max_tokens.is_none());
    }

    #[test]
    fn vertex_config_parses_max_tokens_override() {
        let toml_str = r#"
            [vertex]
            project = "my-project"
            max_tokens = 32768
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(config.vertex.max_tokens, Some(32768));
    }

    // ── ollama config tests ───────────────────────────────────────────────

    #[test]
    fn ollama_config_max_tokens_defaults_to_none() {
        let toml_str = r#"
            [vertex]
            project = "my-project"
            [ollama]
            api_key = "key"
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert!(config.ollama.is_some());
        assert!(
            config
                .ollama
                .as_ref()
                .expect("ollama config present")
                .max_tokens
                .is_none()
        );
    }

    #[test]
    fn ollama_config_parses_max_tokens_override() {
        let toml_str = r#"
            [vertex]
            project = "my-project"
            [ollama]
            api_key = "key"
            max_tokens = 32768
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        assert_eq!(
            config
                .ollama
                .as_ref()
                .expect("ollama config present")
                .max_tokens,
            Some(32768)
        );
    }

    // ── anthropic config tests ────────────────────────────────────────────

    #[test]
    fn anthropic_config_parses_with_defaults() {
        let toml_str = r#"
            backend = "anthropic"
            [vertex]
            project = ""
            [anthropic]
            api_key = "test-key"
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        let a = config
            .anthropic
            .expect("anthropic config should be present");
        assert_eq!(a.api_key, "test-key");
        assert_eq!(a.base_url, ANTHROPIC_DEFAULT_BASE_URL);
        assert_eq!(a.model, "claude-opus-4-8");
        assert!(a.max_tokens.is_none());
    }

    #[test]
    fn anthropic_config_parses_with_all_fields() {
        let toml_str = r#"
            backend = "anthropic"
            [vertex]
            project = ""
            [anthropic]
            api_key = "my-key"
            base_url = "https://custom.anthropic.com/v1"
            model = "claude-sonnet-4-20250514"
            max_tokens = 65536
        "#;
        let config: AppConfig = toml::from_str(toml_str).expect("valid toml");
        let a = config.anthropic.expect("anthropic config present");
        assert_eq!(a.api_key, "my-key");
        assert_eq!(a.base_url, "https://custom.anthropic.com/v1");
        assert_eq!(a.model, "claude-sonnet-4-20250514");
        assert_eq!(a.max_tokens, Some(65536));
    }

    #[test]
    fn validate_anthropic_backend_missing_section_errors() {
        let config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("[anthropic]"),
            "error should mention [anthropic]"
        );
    }

    #[test]
    fn validate_anthropic_backend_empty_api_key_errors() {
        let config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "".to_string(),
                base_url: ANTHROPIC_DEFAULT_BASE_URL.to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
                vision: false,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key"));
    }

    #[test]
    fn validate_anthropic_backend_with_messages_suffix_errors() {
        let config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: "https://api.anthropic.com/v1/messages".to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
                vision: false,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("/messages"),
            "error should mention /messages suffix"
        );
    }

    #[test]
    fn validate_anthropic_backend_valid_config_succeeds() {
        let config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: ANTHROPIC_DEFAULT_BASE_URL.to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: Some(65536),
                vision: false,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(
            result.is_ok(),
            "valid anthropic config should pass: {result:?}"
        );
    }

    #[test]
    fn validate_anthropic_backend_empty_base_url_errors() {
        let config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: String::new(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
                vision: false,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("base_url") && msg.contains("required"),
            "error should mention base_url is required; got: {msg}"
        );
    }

    #[test]
    fn validate_anthropic_backend_whitespace_base_url_errors() {
        let config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: "   ".to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
                vision: false,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err(), "whitespace-only base_url should error");
    }

    #[test]
    fn validate_anthropic_backend_query_string_base_url_errors() {
        let config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: "https://api.anthropic.com/v1?token=x".to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
                vision: false,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("query string"),
            "error should mention query string; got: {msg}"
        );
    }

    #[test]
    fn validate_anthropic_backend_messages_suffix_with_trailing_slash_errors() {
        let config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: "https://api.anthropic.com/v1/messages/".to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
                vision: false,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("/messages"),
            "error should catch /messages even after trailing-slash trim"
        );
    }

    #[test]
    fn normalize_back_compat_synthesizes_default_role_for_anthropic() {
        let mut config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: ANTHROPIC_DEFAULT_BASE_URL.to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
                vision: false,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        config.normalize_back_compat();
        assert!(
            config.models.contains_key("default"),
            "default role should be synthesized for anthropic backend"
        );
        assert_eq!(config.models["default"].backend, "anthropic");
        assert_eq!(config.models["default"].model, "claude-opus-4-8");
    }

    #[test]
    fn normalize_back_compat_synthesizes_default_model_for_anthropic_without_section() {
        let mut config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        config.normalize_back_compat();
        assert_eq!(config.models["default"].model, "claude-opus-4-8");
    }

    #[test]
    fn apply_overrides_anthropic_model() {
        let mut config = AppConfig {
            backend: "anthropic".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: ANTHROPIC_DEFAULT_BASE_URL.to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
                vision: false,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        config.normalize_back_compat();
        apply_overrides(&mut config, None, None, Some("claude-sonnet-4-20250514"));
        assert_eq!(
            config.anthropic.as_ref().expect("anthropic present").model,
            "claude-sonnet-4-20250514",
        );
        let resolved = config
            .resolve_role("default")
            .expect("default role exists after normalize_back_compat");
        assert_eq!(resolved.model, "claude-sonnet-4-20250514");
    }

    #[test]
    fn validate_anthropic_role_rejects_unconfigured_section() {
        let mut models = BTreeMap::new();
        models.insert(
            "my-role".to_string(),
            ModelRole {
                backend: "anthropic".to_string(),
                model: "claude-opus-4-8".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: None,
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("anthropic") && msg.contains("is present"),
            "error should mention missing anthropic section; got: {msg}",
        );
    }

    #[test]
    fn validate_anthropic_role_rejects_empty_api_key() {
        let mut models = BTreeMap::new();
        models.insert(
            "my-role".to_string(),
            ModelRole {
                backend: "anthropic".to_string(),
                model: "claude-opus-4-8".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "".to_string(),
                base_url: ANTHROPIC_DEFAULT_BASE_URL.to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
                vision: false,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("api_key") && msg.contains("not configured"),
            "error should mention missing api_key; got: {msg}"
        );
    }

    #[test]
    fn validate_anthropic_role_rejects_messages_suffix_base_url() {
        let mut models = BTreeMap::new();
        models.insert(
            "my-role".to_string(),
            ModelRole {
                backend: "anthropic".to_string(),
                model: "claude-opus-4-8".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: "https://api.anthropic.com/v1/messages".to_string(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
                vision: false,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("/messages") && msg.contains("my-role"),
            "error should mention /messages and role name; got: {msg}"
        );
    }

    #[test]
    fn validate_anthropic_role_rejects_empty_base_url() {
        let mut models = BTreeMap::new();
        models.insert(
            "my-role".to_string(),
            ModelRole {
                backend: "anthropic".to_string(),
                model: "claude-opus-4-8".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: VertexConfig {
                max_tokens: None,
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                vision: false,
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            opencode_go: None,
            anthropic: Some(AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: String::new(),
                model: "claude-opus-4-8".to_string(),
                max_tokens: None,
                vision: false,
            }),
            tools: ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
        };
        let result = validate(&config, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("base_url") && msg.contains("my-role"),
            "error should mention base_url and role name; got: {msg}"
        );
    }
}
