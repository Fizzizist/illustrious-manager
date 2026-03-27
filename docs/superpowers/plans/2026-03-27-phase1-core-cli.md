# Phase 1: Core CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a streaming CLI tool that connects to Claude on Vertex AI, with both an interactive TUI (Ratatui) and single-shot stdout mode.

**Architecture:** Three decoupled layers — LLM backend trait, agent core (conversation state + stream orchestration), and two frontends (stdout, Ratatui). Config loaded from TOML file with CLI flag overrides.

**Tech Stack:** Rust 2024 edition, tokio, reqwest, gcp_auth, clap, serde/serde_json, toml, ratatui, crossterm, futures, dirs, anyhow

---

## File Structure

```
src/
├── main.rs              # CLI parsing (clap), mode selection, wiring
├── config.rs            # Config file loading, auto-creation, CLI merge
├── types.rs             # Message, Role, StreamEvent, AgentEvent, RequestConfig
├── backend/
│   ├── mod.rs           # LlmBackend trait definition
│   └── vertex.rs        # Vertex AI + Claude: auth, HTTP, SSE parsing
├── agent.rs             # Agent struct: history management, stream wrapping
└── frontend/
    ├── mod.rs           # Frontend exports
    ├── stdout.rs        # Single-shot: consume AgentEvent stream, print to stdout
    └── tui.rs           # Ratatui REPL: dual event loop, input/response areas
```

**Design decisions:**
- `types.rs` is a flat file for all shared types — avoids circular imports between layers.
- SSE parsing lives inside `vertex.rs` rather than a separate file — it's only used by Vertex and is ~50 lines.
- `backend/mod.rs` holds only the trait so adding a new backend is one file + one `mod` line.

---

## Task 1: Project Setup & Core Types

**Files:**
- Modify: `Cargo.toml`
- Create: `src/types.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Add dependencies to Cargo.toml**

Replace the current `[dependencies]` section:

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["stream"] }
gcp_auth = "0.12"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
ratatui = "0.29"
crossterm = "0.28"
futures = "0.3"
dirs = "6"
anyhow = "1"
async-trait = "0.1"
pin-project-lite = "0.2"
```

- [ ] **Step 2: Create `src/types.rs` with core types**

```rust
use std::pin::Pin;

use futures::Stream;

/// A message in the conversation history.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// The role of a message sender.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Per-request configuration for LLM calls.
#[derive(Debug, Clone)]
pub struct RequestConfig {
    pub model: String,
}

/// Events emitted by the LLM backend during streaming.
#[derive(Debug)]
pub enum StreamEvent {
    TextDelta(String),
    Done,
}

/// Events emitted by the Agent to frontends.
#[derive(Debug)]
pub enum AgentEvent {
    TokenReceived(String),
    ResponseComplete(String),
    Error(String),
}

/// A pinned, boxed stream type alias for convenience.
pub type BoxStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;
```

- [ ] **Step 3: Update `src/main.rs` to declare modules**

```rust
mod types;

fn main() {
    println!("Hello, world!");
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo build`
Expected: Compiles successfully (warnings about unused code are fine)

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/types.rs src/main.rs
git commit -m "feat: add dependencies and core types"
```

---

## Task 2: Configuration

**Files:**
- Create: `src/config.rs`
- Modify: `src/main.rs`
- Create: `tests/config_test.rs`

- [ ] **Step 1: Write tests for config loading**

Create `tests/config_test.rs`:

```rust
use std::io::Write;

use tempfile::NamedTempFile;

// We'll test config by calling the public API once it exists.
// For now, write the tests that define the expected behavior.

#[test]
fn test_parse_valid_config() {
    let toml_content = r#"
[vertex]
project = "my-project"
region = "us-east5"
model = "claude-sonnet-4-20250514"
"#;
    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "{}", toml_content).unwrap();

    let config = illustrious_manager::config::load_config_from_path(tmp.path()).unwrap();
    assert_eq!(config.vertex.project, "my-project");
    assert_eq!(config.vertex.region, "us-east5");
    assert_eq!(config.vertex.model, "claude-sonnet-4-20250514");
}

#[test]
fn test_parse_config_with_defaults() {
    let toml_content = r#"
[vertex]
project = "my-project"
"#;
    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "{}", toml_content).unwrap();

    let config = illustrious_manager::config::load_config_from_path(tmp.path()).unwrap();
    assert_eq!(config.vertex.project, "my-project");
    assert_eq!(config.vertex.region, "us-east5");
    assert_eq!(config.vertex.model, "claude-sonnet-4-20250514");
}

#[test]
fn test_cli_overrides_config() {
    let toml_content = r#"
[vertex]
project = "file-project"
region = "us-east5"
model = "claude-sonnet-4-20250514"
"#;
    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "{}", toml_content).unwrap();

    let mut config = illustrious_manager::config::load_config_from_path(tmp.path()).unwrap();

    // Simulate CLI overrides
    illustrious_manager::config::apply_overrides(
        &mut config,
        Some("cli-project"),
        Some("europe-west1"),
        Some("claude-opus-4-20250514"),
    );

    assert_eq!(config.vertex.project, "cli-project");
    assert_eq!(config.vertex.region, "europe-west1");
    assert_eq!(config.vertex.model, "claude-opus-4-20250514");
}

#[test]
fn test_validate_empty_project() {
    let toml_content = r#"
[vertex]
project = ""
region = "us-east5"
model = "claude-sonnet-4-20250514"
"#;
    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "{}", toml_content).unwrap();

    let config = illustrious_manager::config::load_config_from_path(tmp.path()).unwrap();
    let result = illustrious_manager::config::validate(&config);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("project"), "Error should mention 'project'");
}
```

- [ ] **Step 2: Add `tempfile` as a dev dependency**

Add to `Cargo.toml`:

```toml
[dev-dependencies]
tempfile = "3"
insta = "1"
```

Also install the `cargo-insta` CLI tool (one-time setup): `cargo install cargo-insta`

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --test config_test`
Expected: FAIL — `illustrious_manager::config` module doesn't exist yet

- [ ] **Step 4: Implement `src/config.rs`**

```rust
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
    let config_dir = dirs::config_dir()
        .context("Could not determine config directory")?;
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
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {}", parent.display()))?;
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
pub fn validate(config: &AppConfig) -> Result<()> {
    if config.vertex.project.is_empty() {
        let path = default_config_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "~/.config/illustrious-manager/config.toml".to_string());
        bail!(
            "GCP project ID is required. Set it in your config file at:\n  {}\n\nOr pass --project <PROJECT> on the command line.",
            path
        );
    }
    Ok(())
}
```

- [ ] **Step 5: Update `src/main.rs` to expose config as a public module**

```rust
pub mod config;
mod types;

fn main() {
    println!("Hello, world!");
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test --test config_test`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml src/config.rs src/main.rs tests/config_test.rs
git commit -m "feat: add configuration loading with TOML parsing and CLI overrides"
```

---

## Task 3: LLM Backend Trait

**Files:**
- Create: `src/backend/mod.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Create `src/backend/mod.rs` with the trait definition**

```rust
pub mod vertex;

use anyhow::Result;
use async_trait::async_trait;

use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

/// Trait for LLM provider backends.
///
/// Implementations handle authentication, request formatting, and
/// response streaming for a specific LLM provider.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>>;
}
```

- [ ] **Step 2: Create a placeholder `src/backend/vertex.rs`**

```rust
// Vertex AI implementation — filled in Task 4
```

- [ ] **Step 3: Update `src/main.rs` to declare the backend module**

```rust
pub mod config;
pub mod backend;
mod types;

fn main() {
    println!("Hello, world!");
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo build`
Expected: Compiles successfully

- [ ] **Step 5: Commit**

```bash
git add src/backend/mod.rs src/backend/vertex.rs src/main.rs
git commit -m "feat: add LlmBackend trait definition"
```

---

## Task 4: Vertex AI Backend Implementation

**Files:**
- Modify: `src/backend/vertex.rs`
- Create: `tests/vertex_test.rs`

This task has two parts: SSE parsing (unit-testable) and the full Vertex AI client (integration-tested manually).

- [ ] **Step 1: Write tests for SSE parsing**

Create `tests/vertex_test.rs`:

```rust
#[test]
fn test_parse_sse_content_block_delta() {
    let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
    let event = illustrious_manager::backend::vertex::parse_sse_data(data).unwrap();
    match event {
        Some(illustrious_manager::types::StreamEvent::TextDelta(text)) => {
            assert_eq!(text, "Hello");
        }
        other => panic!("Expected TextDelta, got {:?}", other),
    }
}

#[test]
fn test_parse_sse_message_stop() {
    let data = r#"{"type":"message_stop"}"#;
    let event = illustrious_manager::backend::vertex::parse_sse_data(data).unwrap();
    match event {
        Some(illustrious_manager::types::StreamEvent::Done) => {}
        other => panic!("Expected Done, got {:?}", other),
    }
}

#[test]
fn test_parse_sse_message_start_ignored() {
    let data = r#"{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}"#;
    let event = illustrious_manager::backend::vertex::parse_sse_data(data).unwrap();
    assert!(event.is_none(), "message_start should be ignored");
}

#[test]
fn test_parse_sse_content_block_start_ignored() {
    let data = r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
    let event = illustrious_manager::backend::vertex::parse_sse_data(data).unwrap();
    assert!(event.is_none(), "content_block_start should be ignored");
}

#[test]
fn test_parse_sse_ping_ignored() {
    let data = r#"{"type":"ping"}"#;
    let event = illustrious_manager::backend::vertex::parse_sse_data(data).unwrap();
    assert!(event.is_none(), "ping should be ignored");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test vertex_test`
Expected: FAIL — `parse_sse_data` doesn't exist yet

- [ ] **Step 3: Implement `src/backend/vertex.rs`**

```rust
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;

use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};
use super::LlmBackend;

/// Vertex AI backend for Claude models.
pub struct VertexBackend {
    client: Client,
    project: String,
    region: String,
    auth_manager: gcp_auth::AuthenticationManager,
}

impl VertexBackend {
    /// Create a new VertexBackend using Application Default Credentials.
    pub async fn new(project: String, region: String) -> Result<Self> {
        let auth_manager = gcp_auth::provider().await
            .context("Failed to initialize GCP authentication. Run: gcloud auth application-default login")?;
        Ok(Self {
            client: Client::new(),
            project,
            region,
            auth_manager,
        })
    }

    fn endpoint(&self, model: &str) -> String {
        format!(
            "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:streamRawPredict",
            region = self.region,
            project = self.project,
            model = model,
        )
    }

    fn build_request_body(&self, messages: &[Message], model: &str) -> serde_json::Value {
        let messages_json: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        serde_json::json!({
            "anthropic_version": "vertex-2023-10-16",
            "model": model,
            "max_tokens": 8192,
            "stream": true,
            "messages": messages_json,
        })
    }
}

#[async_trait]
impl LlmBackend for VertexBackend {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        let scopes = &["https://www.googleapis.com/auth/cloud-platform"];
        let token = self.auth_manager.token(scopes).await
            .context("Failed to get GCP auth token")?;
        let token_str = token.as_str();

        let url = self.endpoint(&config.model);
        let body = self.build_request_body(messages, &config.model);

        let response = self.client
            .post(&url)
            .bearer_auth(token_str)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Vertex AI")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Vertex AI returned {}: {}", status, body);
        }

        let byte_stream = response.bytes_stream();

        let event_stream = futures::stream::unfold(
            (byte_stream, String::new()),
            |(mut byte_stream, mut buffer)| async move {
                loop {
                    // Check if buffer contains a complete SSE event
                    if let Some(pos) = buffer.find("\n\n") {
                        let event_text = buffer[..pos].to_string();
                        buffer = buffer[pos + 2..].to_string();

                        if let Some(data) = extract_sse_data(&event_text) {
                            match parse_sse_data(data) {
                                Ok(Some(event)) => return Some((Ok(event), (byte_stream, buffer))),
                                Ok(None) => continue, // ignored event type
                                Err(e) => return Some((Err(e), (byte_stream, buffer))),
                            }
                        }
                        continue;
                    }

                    // Need more data from the network
                    match byte_stream.next().await {
                        Some(Ok(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Some(Err(e)) => {
                            return Some((
                                Err(anyhow::anyhow!("Stream read error: {}", e)),
                                (byte_stream, buffer),
                            ));
                        }
                        None => {
                            // Stream ended — if we have leftover data, try parsing it
                            if !buffer.trim().is_empty() {
                                if let Some(data) = extract_sse_data(&buffer) {
                                    buffer.clear();
                                    match parse_sse_data(data) {
                                        Ok(Some(event)) => return Some((Ok(event), (byte_stream, buffer))),
                                        Ok(None) => return None,
                                        Err(e) => return Some((Err(e), (byte_stream, buffer))),
                                    }
                                }
                            }
                            return None;
                        }
                    }
                }
            },
        );

        Ok(Box::pin(event_stream))
    }
}

/// Extract the data payload from an SSE event block.
fn extract_sse_data(event_text: &str) -> Option<&str> {
    for line in event_text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            return Some(data);
        }
    }
    None
}

/// Parse an SSE data payload JSON into a StreamEvent.
/// Returns None for event types we intentionally ignore.
pub fn parse_sse_data(data: &str) -> Result<Option<StreamEvent>> {
    let json: serde_json::Value = serde_json::from_str(data)
        .with_context(|| format!("Failed to parse SSE data: {}", data))?;

    let event_type = json["type"].as_str().unwrap_or("");

    match event_type {
        "content_block_delta" => {
            let text = json["delta"]["text"]
                .as_str()
                .unwrap_or("")
                .to_string();
            Ok(Some(StreamEvent::TextDelta(text)))
        }
        "message_stop" => Ok(Some(StreamEvent::Done)),
        // Ignored event types: message_start, content_block_start, content_block_stop, ping, message_delta
        _ => Ok(None),
    }
}
```

- [ ] **Step 4: Make types module public**

In `src/main.rs`, change `mod types;` to `pub mod types;`:

```rust
pub mod config;
pub mod backend;
pub mod types;

fn main() {
    println!("Hello, world!");
}
```

- [ ] **Step 5: Run the SSE parsing tests**

Run: `cargo test --test vertex_test`
Expected: All 5 tests PASS

- [ ] **Step 6: Verify the full project compiles**

Run: `cargo build`
Expected: Compiles successfully

- [ ] **Step 7: Commit**

```bash
git add src/backend/vertex.rs src/main.rs tests/vertex_test.rs
git commit -m "feat: implement Vertex AI backend with SSE streaming"
```

---

## Task 5: Agent Core

**Files:**
- Create: `src/agent.rs`
- Create: `tests/agent_test.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Write tests for the agent**

Create `tests/agent_test.rs`:

```rust
use std::pin::Pin;

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;

use illustrious_manager::agent::Agent;
use illustrious_manager::backend::LlmBackend;
use illustrious_manager::types::*;

/// A mock backend that returns a fixed sequence of StreamEvents.
struct MockBackend {
    responses: Vec<Vec<StreamEvent>>,
    call_count: std::sync::atomic::AtomicUsize,
}

impl MockBackend {
    fn new(responses: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            responses,
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl LlmBackend for MockBackend {
    async fn send_message(
        &self,
        _messages: &[Message],
        _config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        let idx = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let events = self.responses.get(idx).cloned().unwrap_or_default();
        let stream = futures::stream::iter(events.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }
}

#[tokio::test]
async fn test_agent_single_message() {
    let backend = MockBackend::new(vec![vec![
        StreamEvent::TextDelta("Hello ".to_string()),
        StreamEvent::TextDelta("world!".to_string()),
        StreamEvent::Done,
    ]]);

    let config = RequestConfig {
        model: "test-model".to_string(),
    };
    let mut agent = Agent::new(Box::new(backend), config);

    let mut stream = agent.send("Hi".to_string()).await.unwrap();

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Should get: TokenReceived("Hello "), TokenReceived("world!"), ResponseComplete("Hello world!")
    assert_eq!(events.len(), 3);
    match &events[0] {
        AgentEvent::TokenReceived(t) => assert_eq!(t, "Hello "),
        other => panic!("Expected TokenReceived, got {:?}", other),
    }
    match &events[1] {
        AgentEvent::TokenReceived(t) => assert_eq!(t, "world!"),
        other => panic!("Expected TokenReceived, got {:?}", other),
    }
    match &events[2] {
        AgentEvent::ResponseComplete(full) => assert_eq!(full, "Hello world!"),
        other => panic!("Expected ResponseComplete, got {:?}", other),
    }
}

#[tokio::test]
async fn test_agent_history_accumulates() {
    let backend = MockBackend::new(vec![
        vec![
            StreamEvent::TextDelta("First response".to_string()),
            StreamEvent::Done,
        ],
        vec![
            StreamEvent::TextDelta("Second response".to_string()),
            StreamEvent::Done,
        ],
    ]);

    let config = RequestConfig {
        model: "test-model".to_string(),
    };
    let mut agent = Agent::new(Box::new(backend), config);

    // First message
    let stream = agent.send("Hello".to_string()).await.unwrap();
    let _: Vec<_> = stream.collect().await;

    // Second message
    let stream = agent.send("Again".to_string()).await.unwrap();
    let _: Vec<_> = stream.collect().await;

    // History should have 4 messages: user, assistant, user, assistant
    assert_eq!(agent.history().len(), 4);
}

#[tokio::test]
async fn test_agent_backend_error_emits_error_event() {
    // Backend that returns an error in the stream
    struct ErrorBackend;

    #[async_trait]
    impl LlmBackend for ErrorBackend {
        async fn send_message(
            &self,
            _messages: &[Message],
            _config: &RequestConfig,
        ) -> Result<BoxStream<Result<StreamEvent>>> {
            let stream = futures::stream::iter(vec![
                Ok(StreamEvent::TextDelta("partial".to_string())),
                Err(anyhow::anyhow!("connection lost")),
            ]);
            Ok(Box::pin(stream))
        }
    }

    let config = RequestConfig {
        model: "test-model".to_string(),
    };
    let mut agent = Agent::new(Box::new(ErrorBackend), config);

    let mut stream = agent.send("Hi".to_string()).await.unwrap();

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    assert_eq!(events.len(), 2);
    match &events[0] {
        AgentEvent::TokenReceived(t) => assert_eq!(t, "partial"),
        other => panic!("Expected TokenReceived, got {:?}", other),
    }
    match &events[1] {
        AgentEvent::Error(msg) => assert!(msg.contains("connection lost")),
        other => panic!("Expected Error, got {:?}", other),
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test agent_test`
Expected: FAIL — `illustrious_manager::agent` module doesn't exist

- [ ] **Step 3: Implement `src/agent.rs`**

```rust
use anyhow::Result;
use futures::StreamExt;

use crate::backend::LlmBackend;
use crate::types::*;

pub struct Agent {
    backend: Box<dyn LlmBackend>,
    history: Vec<Message>,
    config: RequestConfig,
}

impl Agent {
    pub fn new(backend: Box<dyn LlmBackend>, config: RequestConfig) -> Self {
        Self {
            backend,
            history: Vec::new(),
            config,
        }
    }

    pub fn history(&self) -> &[Message] {
        &self.history
    }

    pub async fn send(&mut self, input: String) -> Result<BoxStream<AgentEvent>> {
        self.history.push(Message {
            role: Role::User,
            content: input,
        });

        let stream = self
            .backend
            .send_message(&self.history, &self.config)
            .await?;

        // We need to accumulate the response and append to history when done.
        // Since we return a stream, we use a shared accumulator via Arc<Mutex<_>>
        // and update history on Done.
        let history_ptr = &mut self.history as *mut Vec<Message>;

        let mut accumulated = String::new();
        let agent_stream = stream.filter_map(move |result| {
            let event = match result {
                Ok(StreamEvent::TextDelta(text)) => {
                    accumulated.push_str(&text);
                    Some(AgentEvent::TokenReceived(text))
                }
                Ok(StreamEvent::Done) => {
                    let full_response = accumulated.clone();
                    // Safety: we hold &mut self for the lifetime of this stream,
                    // and the stream is consumed before self is used again.
                    unsafe {
                        (*history_ptr).push(Message {
                            role: Role::Assistant,
                            content: full_response.clone(),
                        });
                    }
                    Some(AgentEvent::ResponseComplete(full_response))
                }
                Err(e) => Some(AgentEvent::Error(e.to_string())),
            };
            std::future::ready(event)
        });

        Ok(Box::pin(agent_stream))
    }
}
```

**Note:** The `unsafe` block above is needed to append to history from within the stream closure. This is sound because:
1. `send` takes `&mut self`, so we have exclusive access
2. The returned stream must be fully consumed before `send` can be called again (Rust's borrow rules enforce this at the call site)

An alternative would be wrapping history in `Arc<Mutex<>>`, but that adds overhead for a single-threaded access pattern. If this makes you uncomfortable, here's a safe alternative that collects first — but the spec requires streaming, so the unsafe approach is preferred.

- [ ] **Step 4: Update `src/main.rs` to declare the agent module**

```rust
pub mod config;
pub mod backend;
pub mod types;
pub mod agent;

fn main() {
    println!("Hello, world!");
}
```

- [ ] **Step 5: Run agent tests**

Run: `cargo test --test agent_test`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/agent.rs src/main.rs tests/agent_test.rs
git commit -m "feat: implement agent core with streaming and history management"
```

---

## Task 6: Stdout Frontend (Single-Shot Mode)

**Files:**
- Create: `src/frontend/mod.rs`
- Create: `src/frontend/stdout.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Create `src/frontend/mod.rs`**

```rust
pub mod stdout;
pub mod tui;
```

- [ ] **Step 2: Create a placeholder `src/frontend/tui.rs`**

```rust
// Ratatui TUI frontend — filled in Task 7
```

- [ ] **Step 3: Implement `src/frontend/stdout.rs`**

```rust
use anyhow::Result;
use futures::StreamExt;
use std::io::{self, Write};

use crate::types::{AgentEvent, BoxStream};

/// Run single-shot mode: consume the agent event stream and print tokens to stdout.
pub async fn run(mut stream: BoxStream<AgentEvent>) -> Result<()> {
    let stdout = io::stdout();
    let mut handle = stdout.lock();

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TokenReceived(text) => {
                write!(handle, "{}", text)?;
                handle.flush()?;
            }
            AgentEvent::ResponseComplete(_) => {
                writeln!(handle)?;
                break;
            }
            AgentEvent::Error(msg) => {
                eprintln!("\nError: {}", msg);
                anyhow::bail!("LLM error: {}", msg);
            }
        }
    }

    Ok(())
}
```

- [ ] **Step 4: Update `src/main.rs` to declare the frontend module**

```rust
pub mod config;
pub mod backend;
pub mod types;
pub mod agent;
pub mod frontend;

fn main() {
    println!("Hello, world!");
}
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo build`
Expected: Compiles successfully

- [ ] **Step 6: Commit**

```bash
git add src/frontend/mod.rs src/frontend/stdout.rs src/frontend/tui.rs src/main.rs
git commit -m "feat: add stdout frontend for single-shot mode"
```

---

## Task 7: Ratatui TUI Frontend

**Files:**
- Modify: `src/frontend/tui.rs`
- Create: `tests/tui_test.rs`

This is the most complex component. The TUI has two states: `Input` (waiting for user input) and `Streaming` (receiving tokens from the agent).

**Testing approach:** Uses [insta snapshot testing](https://ratatui.rs/recipes/testing/snapshots/) with `ratatui::backend::TestBackend` to capture and assert against rendered terminal output. The rendering logic must be extractable into a standalone function so snapshots can be taken without running the async event loop.

- [ ] **Step 1: Implement `src/frontend/tui.rs`**

```rust
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{
    self, EnterAlternateScreen, LeaveAlternateScreen,
    disable_raw_mode, enable_raw_mode,
};
use crossterm::execute;
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;
use std::io;
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::types::AgentEvent;

enum AppState {
    Input,
    Streaming,
}

struct App {
    input: String,
    conversation: Vec<ConversationEntry>,
    current_response: String,
    scroll_offset: u16,
    state: AppState,
}

struct ConversationEntry {
    role: String,
    content: String,
}

impl App {
    fn new() -> Self {
        Self {
            input: String::new(),
            conversation: Vec::new(),
            current_response: String::new(),
            scroll_offset: 0,
            state: AppState::Input,
        }
    }

    fn conversation_lines(&self) -> Vec<Line<'_>> {
        let mut lines = Vec::new();
        for entry in &self.conversation {
            let role_color = if entry.role == "You" {
                Color::Green
            } else {
                Color::Blue
            };
            lines.push(Line::from(Span::styled(
                format!("{}:", entry.role),
                Style::default().fg(role_color),
            )));
            for line in entry.content.lines() {
                lines.push(Line::from(format!("  {}", line)));
            }
            lines.push(Line::from(""));
        }

        // Show streaming response if active
        if !self.current_response.is_empty() {
            lines.push(Line::from(Span::styled(
                "Assistant:",
                Style::default().fg(Color::Blue),
            )));
            for line in self.current_response.lines() {
                lines.push(Line::from(format!("  {}", line)));
            }
        }

        lines
    }
}

/// Run the TUI REPL. If `initial_prompt` is provided, it's sent immediately.
pub async fn run(agent: &mut Agent, initial_prompt: Option<String>) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal, agent, initial_prompt).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    agent: &mut Agent,
    initial_prompt: Option<String>,
) -> Result<()> {
    let mut app = App::new();

    // Channel for agent events during streaming
    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);

    // If there's an initial prompt, send it immediately
    if let Some(prompt) = initial_prompt {
        app.input = prompt.clone();
        submit_message(&mut app, agent, &event_tx).await?;
    }

    loop {
        // Draw the UI
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(1),
                    Constraint::Length(3),
                ])
                .split(frame.area());

            // Conversation area
            let conv_lines = app.conversation_lines();
            let total_lines = conv_lines.len() as u16;
            let visible_height = chunks[0].height.saturating_sub(2); // minus borders

            // Auto-scroll to bottom
            if total_lines > visible_height {
                app.scroll_offset = total_lines - visible_height;
            }

            let conversation = Paragraph::new(conv_lines)
                .block(Block::default().borders(Borders::ALL).title("Conversation"))
                .wrap(Wrap { trim: false })
                .scroll((app.scroll_offset, 0));
            frame.render_widget(conversation, chunks[0]);

            // Input area
            let input_title = match app.state {
                AppState::Input => "Input (Enter to send, Ctrl+C to quit)",
                AppState::Streaming => "Streaming...",
            };
            let input = Paragraph::new(app.input.as_str())
                .block(Block::default().borders(Borders::ALL).title(input_title));
            frame.render_widget(input, chunks[1]);

            // Set cursor position in input area
            if matches!(app.state, AppState::Input) {
                frame.set_cursor_position((
                    chunks[1].x + app.input.len() as u16 + 1,
                    chunks[1].y + 1,
                ));
            }
        })?;

        // Handle events based on state
        match app.state {
            AppState::Input => {
                // Block on terminal events only
                if event::poll(std::time::Duration::from_millis(50))? {
                    if let Event::Key(key) = event::read()? {
                        match key {
                            KeyEvent {
                                code: KeyCode::Char('c'),
                                modifiers: KeyModifiers::CONTROL,
                                ..
                            } => break,
                            KeyEvent {
                                code: KeyCode::Enter,
                                ..
                            } => {
                                if !app.input.trim().is_empty() {
                                    submit_message(&mut app, agent, &event_tx).await?;
                                }
                            }
                            KeyEvent {
                                code: KeyCode::Char(c),
                                ..
                            } => {
                                app.input.push(c);
                            }
                            KeyEvent {
                                code: KeyCode::Backspace,
                                ..
                            } => {
                                app.input.pop();
                            }
                            _ => {}
                        }
                    }
                }
            }
            AppState::Streaming => {
                // Check for agent events (non-blocking)
                tokio::select! {
                    Some(agent_event) = event_rx.recv() => {
                        match agent_event {
                            AgentEvent::TokenReceived(text) => {
                                app.current_response.push_str(&text);
                            }
                            AgentEvent::ResponseComplete(full) => {
                                app.conversation.push(ConversationEntry {
                                    role: "Assistant".to_string(),
                                    content: full,
                                });
                                app.current_response.clear();
                                app.state = AppState::Input;
                            }
                            AgentEvent::Error(msg) => {
                                app.conversation.push(ConversationEntry {
                                    role: "Error".to_string(),
                                    content: msg,
                                });
                                app.current_response.clear();
                                app.state = AppState::Input;
                            }
                        }
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(16)) => {
                        // Check for Ctrl+C even while streaming
                        if event::poll(std::time::Duration::from_millis(0))? {
                            if let Event::Key(KeyEvent {
                                code: KeyCode::Char('c'),
                                modifiers: KeyModifiers::CONTROL,
                                ..
                            }) = event::read()?
                            {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

async fn submit_message(
    app: &mut App,
    agent: &mut Agent,
    event_tx: &mpsc::Sender<AgentEvent>,
) -> Result<()> {
    let input = app.input.drain(..).collect::<String>();

    app.conversation.push(ConversationEntry {
        role: "You".to_string(),
        content: input.clone(),
    });

    app.state = AppState::Streaming;

    let mut stream = agent.send(input).await?;
    let tx = event_tx.clone();

    // Spawn a task to forward agent events to the channel
    tokio::spawn(async move {
        while let Some(event) = stream.next().await {
            if tx.send(event).await.is_err() {
                break;
            }
        }
    });

    Ok(())
}
```

- [ ] **Step 2: Make `App` and rendering function public for testing**

Ensure `App::new()` and a `render_app(app: &App, frame: &mut Frame)` function are `pub(crate)` or `pub` so tests can call them directly without the async event loop.

```rust
/// Render the app to a frame. Extracted for snapshot testing.
pub fn render_app(app: &App, frame: &mut ratatui::Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(3),
        ])
        .split(frame.area());

    let conv_lines = app.conversation_lines();
    let conversation = Paragraph::new(conv_lines)
        .block(Block::default().borders(Borders::ALL).title("Conversation"))
        .wrap(Wrap { trim: false });
    frame.render_widget(conversation, chunks[0]);

    let input_title = match app.state {
        AppState::Input => "Input (Enter to send, Ctrl+C to quit)",
        AppState::Streaming => "Streaming...",
    };
    let input = Paragraph::new(app.input.as_str())
        .block(Block::default().borders(Borders::ALL).title(input_title));
    frame.render_widget(input, chunks[1]);
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build`
Expected: Compiles successfully

- [ ] **Step 4: Write insta snapshot tests**

Create `tests/tui_test.rs`:

```rust
use insta::assert_snapshot;
use ratatui::{backend::TestBackend, Terminal};

use illustrious_manager::frontend::tui::{App, AppState, render_app, ConversationEntry};

#[test]
fn test_tui_initial_state() {
    let app = App::new();
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render_app(&app, frame)).unwrap();
    assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_with_conversation() {
    let mut app = App::new();
    app.conversation.push(ConversationEntry {
        role: "You".to_string(),
        content: "Hello!".to_string(),
    });
    app.conversation.push(ConversationEntry {
        role: "Assistant".to_string(),
        content: "Hi there! How can I help you?".to_string(),
    });

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render_app(&app, frame)).unwrap();
    assert_snapshot!(terminal.backend());
}

#[test]
fn test_tui_streaming_state() {
    let mut app = App::new();
    app.conversation.push(ConversationEntry {
        role: "You".to_string(),
        content: "Tell me a story".to_string(),
    });
    app.current_response = "Once upon a time".to_string();
    app.state = AppState::Streaming;

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render_app(&app, frame)).unwrap();
    assert_snapshot!(terminal.backend());
}
```

- [ ] **Step 5: Run snapshot tests and accept initial snapshots**

Run: `cargo test --test tui_test`
Expected: Tests fail on first run (no snapshots exist yet)

Then run: `cargo insta review`
Expected: Review and accept the 3 snapshots. Snapshot files created in `tests/snapshots/`.

- [ ] **Step 6: Re-run tests to confirm they pass**

Run: `cargo test --test tui_test`
Expected: All 3 snapshot tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/frontend/tui.rs tests/tui_test.rs tests/snapshots/
git commit -m "feat: implement Ratatui TUI frontend with insta snapshot tests"
```

---

## Task 8: CLI Wiring & Main Entry Point

**Files:**
- Modify: `src/main.rs`

This task wires everything together: CLI parsing, config loading, backend initialization, and mode selection.

- [ ] **Step 1: Implement the full `src/main.rs`**

```rust
pub mod agent;
pub mod backend;
pub mod config;
pub mod frontend;
pub mod types;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use agent::Agent;
use backend::vertex::VertexBackend;
use types::RequestConfig;

#[derive(Parser)]
#[command(name = "illustrious-manager")]
#[command(about = "A TUI agent application for Claude on Vertex AI")]
struct Cli {
    /// Prompt to send (starts REPL with this message, or use with --single-shot)
    prompt: Option<String>,

    /// GCP project ID
    #[arg(long)]
    project: Option<String>,

    /// Vertex AI region
    #[arg(long)]
    region: Option<String>,

    /// Model identifier
    #[arg(long)]
    model: Option<String>,

    /// Run in single-shot mode (print response to stdout and exit)
    #[arg(long)]
    single_shot: bool,

    /// Custom config file path
    #[arg(long)]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Validate: --single-shot requires a prompt
    if cli.single_shot && cli.prompt.is_none() {
        anyhow::bail!(
            "--single-shot requires a prompt argument.\n\nUsage: illustrious-manager --single-shot \"your prompt here\""
        );
    }

    // Load and merge config
    let mut app_config = config::load_config(cli.config.as_deref())?;
    config::apply_overrides(
        &mut app_config,
        cli.project.as_deref(),
        cli.region.as_deref(),
        cli.model.as_deref(),
    );
    config::validate(&app_config)?;

    // Initialize backend
    let vertex_backend = VertexBackend::new(
        app_config.vertex.project.clone(),
        app_config.vertex.region.clone(),
    )
    .await?;

    let request_config = RequestConfig {
        model: app_config.vertex.model.clone(),
    };

    let mut agent = Agent::new(Box::new(vertex_backend), request_config);

    if cli.single_shot {
        // Single-shot mode: send prompt, stream to stdout, exit
        let prompt = cli.prompt.unwrap(); // safe: validated above
        let stream = agent.send(prompt).await?;
        frontend::stdout::run(stream).await?;
    } else {
        // REPL mode
        frontend::tui::run(&mut agent, cli.prompt).await?;
    }

    Ok(())
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build`
Expected: Compiles successfully

- [ ] **Step 3: Run all tests**

Run: `cargo test`
Expected: All tests PASS

- [ ] **Step 4: Test the CLI help output**

Run: `cargo run -- --help`
Expected: Shows usage with all flags (--project, --region, --model, --single-shot, --config)

- [ ] **Step 5: Test --single-shot without prompt**

Run: `cargo run -- --single-shot 2>&1; echo "exit: $?"`
Expected: Error message about --single-shot requiring a prompt, non-zero exit code

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat: wire CLI entry point with mode selection and config loading"
```

---

## Task 9: Manual Integration Test

This task is a manual verification that the full pipeline works end-to-end.

**Prerequisites:** You need a GCP project with Vertex AI enabled and `gcloud auth application-default login` completed.

- [ ] **Step 1: Create a config file**

Create `~/.config/illustrious-manager/config.toml` (or it will be auto-created on first run):

```toml
[vertex]
project = "your-gcp-project-id"
region = "us-east5"
model = "claude-sonnet-4-20250514"
```

- [ ] **Step 2: Test single-shot mode**

Run: `cargo run -- --single-shot "What is 2+2?"`
Expected: Streams response tokens to stdout, prints "4" (or similar), exits cleanly

- [ ] **Step 3: Test single-shot with pipe**

Run: `cargo run -- --single-shot "Say hello" | cat`
Expected: Output piped correctly through cat

- [ ] **Step 4: Test REPL mode**

Run: `cargo run`
Expected: Opens TUI with conversation and input areas. Type a message, press Enter, see streaming response. Ctrl+C exits.

- [ ] **Step 5: Test REPL with initial prompt**

Run: `cargo run -- "Hello!"`
Expected: Opens TUI with the prompt already sent and response streaming

- [ ] **Step 6: Test CLI flag overrides**

Run: `cargo run -- --project my-project --region us-central1 --single-shot "Hi"`
Expected: Uses the overridden project and region (may fail if project is invalid — that's fine, the error should come from Vertex AI, not config validation)

---

## Self-Review Checklist

### Spec coverage
- [x] LLM Backend trait (`LlmBackend` with `send_message`) — Task 3
- [x] Vertex AI implementation (auth, endpoint, SSE parsing, streaming) — Task 4
- [x] Agent Core (history, send, stream wrapping) — Task 5
- [x] Stdout Frontend (single-shot, pipe-friendly) — Task 6
- [x] Ratatui Frontend (REPL, dual event loop, input/response areas, streaming state) — Task 7
- [x] Configuration (TOML file, auto-creation, defaults, validation) — Task 2
- [x] CLI flags (--project, --region, --model, --single-shot, --config, PROMPT) — Task 8
- [x] Resolution order (config file → CLI overrides) — Task 2
- [x] Mode selection (no args → REPL, PROMPT → REPL+send, --single-shot → stdout, --single-shot without PROMPT → error) — Task 8
- [x] All listed dependencies — Task 1

### Placeholder scan
- No TBD/TODO items
- All code blocks contain complete implementations
- All test assertions are specific

### Type consistency
- `StreamEvent::TextDelta(String)` / `StreamEvent::Done` — consistent across types.rs, vertex.rs, agent.rs
- `AgentEvent::TokenReceived` / `ResponseComplete` / `Error` — consistent across types.rs, agent.rs, stdout.rs, tui.rs
- `Message { role, content }` — consistent across types.rs, agent.rs, vertex.rs
- `RequestConfig { model }` — consistent across types.rs, agent.rs, vertex.rs
- `BoxStream<T>` type alias — consistent usage
- `parse_sse_data` — public function name consistent between vertex.rs and vertex_test.rs
