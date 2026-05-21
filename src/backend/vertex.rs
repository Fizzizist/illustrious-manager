use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use tokio::sync::OnceCell;

use super::LlmBackend;
use super::sse::create_sse_event_stream;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

/// Required protocol version string for Vertex AI's Anthropic API.
const ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

/// Stateful SSE parser that tracks which block indices are tool_use blocks.
#[derive(Default)]
pub struct VertexSseParser {
    tool_use_indices: HashSet<u64>,
    /// Tracks which indices are thinking blocks so we can emit ThinkingSignature at stop.
    thinking_signatures: HashMap<u64, String>,
    input_tokens: u32,
    /// Buffer for events when multiple events appear in a single SSE chunk
    pub(crate) event_buffer: Vec<StreamEvent>,
}

impl VertexSseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parse(&mut self, data: &str) -> Result<Option<StreamEvent>> {
        // Always process incoming data into the buffer first so no SSE events are dropped.
        // Empty string is used by tests to drain buffered events without consuming new data.
        if !data.is_empty() {
            self.fill_buffer(data)?;
        }

        if !self.event_buffer.is_empty() {
            return Ok(Some(self.event_buffer.remove(0)));
        }

        Ok(None)
    }

    fn fill_buffer(&mut self, data: &str) -> Result<()> {
        let json: serde_json::Value = serde_json::from_str(data)
            .with_context(|| format!("Failed to parse SSE data: {}", data))?;

        let event_type = json["type"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("SSE event missing 'type' field"))?;

        match event_type {
            "message_start" => {
                self.input_tokens = json["message"]["usage"]["input_tokens"]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("message_start missing 'input_tokens'"))?
                    as u32;
            }
            "content_block_start" => {
                let block_type = json["content_block"]["type"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("content_block_start missing 'type'"))?;
                match block_type {
                    "tool_use" => {
                        let index = json["index"].as_u64().ok_or_else(|| {
                            anyhow::anyhow!("content_block_start missing 'index'")
                        })?;
                        self.tool_use_indices.insert(index);
                        let id = json["content_block"]["id"]
                            .as_str()
                            .ok_or_else(|| {
                                anyhow::anyhow!("tool_use content_block_start missing 'id'")
                            })?
                            .to_string();
                        let name = json["content_block"]["name"]
                            .as_str()
                            .ok_or_else(|| {
                                anyhow::anyhow!("tool_use content_block_start missing 'name'")
                            })?
                            .to_string();
                        self.event_buffer
                            .push(StreamEvent::ToolUseStart { id, name });
                    }
                    "thinking" => {
                        let index = json["index"].as_u64().ok_or_else(|| {
                            anyhow::anyhow!("content_block_start missing 'index'")
                        })?;
                        self.thinking_signatures.insert(index, String::new());
                    }
                    "redacted_thinking" => {
                        let index = json["index"].as_u64().ok_or_else(|| {
                            anyhow::anyhow!("content_block_start missing 'index'")
                        })?;
                        // Redacted thinking doesn't need a signature — it carries opaque data
                        self.thinking_signatures.insert(index, String::new());
                        if let Some(data) = json["content_block"]["data"].as_str() {
                            self.event_buffer
                                .push(StreamEvent::ThinkingDelta(data.to_string()));
                        }
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let delta_type = json["delta"]["type"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("content_block_delta missing 'type'"))?;
                match delta_type {
                    "input_json_delta" => {
                        let chunk = json["delta"]["partial_json"]
                            .as_str()
                            .ok_or_else(|| {
                                anyhow::anyhow!("input_json_delta missing 'partial_json'")
                            })?
                            .to_string();
                        self.event_buffer.push(StreamEvent::ToolUseDelta(chunk));
                    }
                    "thinking_delta" => {
                        let text = json["delta"]["thinking"].as_str().unwrap_or("").to_string();
                        self.event_buffer.push(StreamEvent::ThinkingDelta(text));
                    }
                    "signature_delta" => {
                        let index = json["index"].as_u64().ok_or_else(|| {
                            anyhow::anyhow!("content_block_delta missing 'index'")
                        })?;
                        let chunk = json["delta"]["signature"]
                            .as_str()
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "signature_delta missing or invalid 'signature' field"
                                )
                            })?
                            .to_string();
                        self.thinking_signatures
                            .entry(index)
                            .or_default()
                            .push_str(&chunk);
                    }
                    _ => {
                        let text = json["delta"]["text"].as_str().unwrap_or("").to_string();
                        self.event_buffer.push(StreamEvent::TextDelta(text));
                    }
                }
            }
            "content_block_stop" => {
                let index = json["index"]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("content_block_stop missing 'index'"))?;
                if self.tool_use_indices.remove(&index) {
                    self.event_buffer.push(StreamEvent::ToolUseDone);
                }
                if let Some(sig) = self.thinking_signatures.remove(&index) {
                    self.event_buffer.push(StreamEvent::ThinkingSignature(sig));
                }
            }
            "message_delta" => {
                let stop_reason = json["delta"]["stop_reason"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("message_delta missing 'stop_reason'"))?
                    .to_string();
                let output_tokens = json["usage"]["output_tokens"]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("message_delta missing 'output_tokens'"))?
                    as u32;
                if stop_reason == "max_tokens" {
                    return Err(anyhow::anyhow!(
                        "Response truncated: max_tokens limit reached (input_tokens={}, output_tokens={}). Increase max_tokens in your config.",
                        self.input_tokens,
                        output_tokens,
                    ));
                } else {
                    self.event_buffer.push(StreamEvent::Usage {
                        input_tokens: self.input_tokens,
                        output_tokens,
                        stop_reason,
                    });
                }
            }
            "message_stop" => {
                self.event_buffer.push(StreamEvent::Done);
            }
            _ => {}
        }

        Ok(())
    }
}

type AuthProvider = Arc<dyn gcp_auth::TokenProvider>;
type AuthCell = Arc<OnceCell<AuthProvider>>;

/// Cache that shares expensive GCP auth state across Vertex AI backends that
/// target the same `(project, region)` pair.
///
/// Each key maps to a `OnceCell` initialised at most once, even under
/// concurrent callers.  This avoids the TOCTOU race of the
/// "read-lock / drop / await / write-lock" pattern.
pub struct VertexAuthCache {
    cells: Mutex<HashMap<(String, String), AuthCell>>,
}

impl Default for VertexAuthCache {
    fn default() -> Self {
        Self::new()
    }
}

impl VertexAuthCache {
    pub fn new() -> Self {
        Self {
            cells: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the auth provider for `(project, region)`, initialising it
    /// exactly once — safe under concurrent callers because
    /// `OnceCell::get_or_try_init` serialises initialisation.
    pub async fn get_or_init(&self, project: String, region: String) -> Result<AuthProvider> {
        let key = (project, region);
        let cell: AuthCell = {
            let mut cache = self.cells.lock().unwrap_or_else(|e| e.into_inner());
            Arc::clone(
                cache
                    .entry(key)
                    .or_insert_with(|| Arc::new(OnceCell::new())),
            )
        };
        let provider = cell
            .get_or_try_init(|| async {
                gcp_auth::provider().await.map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to initialize GCP authentication. \
                         Run: gcloud auth application-default login\n{e}"
                    )
                })
            })
            .await?;
        Ok(Arc::clone(provider))
    }

    /// Pre-seed the cache entry for `(project, region)` with a known provider.
    /// Intended for tests only.
    #[cfg(test)]
    pub fn seed(
        &self,
        project: &str,
        region: &str,
        provider: AuthProvider,
    ) -> Arc<OnceCell<AuthProvider>> {
        let key = (project.to_string(), region.to_string());
        let mut cache = self.cells.lock().expect("lock");
        let cell = Arc::clone(
            cache
                .entry(key)
                .or_insert_with(|| Arc::new(OnceCell::new())),
        );
        cell.set(provider)
            .ok()
            .expect("cell should not have been set already");
        cell
    }
}

/// Vertex AI backend for Claude models.
pub struct VertexBackend {
    client: Client,
    project: String,
    region: String,
    auth_manager: Arc<dyn gcp_auth::TokenProvider>,
}

impl VertexBackend {
    /// Create a new VertexBackend using Application Default Credentials.
    pub async fn new(project: String, region: String) -> Result<Self> {
        let auth_manager = gcp_auth::provider().await.context(
            "Failed to initialize GCP authentication. Run: gcloud auth application-default login",
        )?;
        Ok(Self {
            client: Client::new(),
            project,
            region,
            auth_manager,
        })
    }

    /// Create a `VertexBackend` with an already-initialized auth provider.
    /// Used by `BackendFactory` to share a single provider across roles that
    /// target the same `(project, region)`.
    pub fn with_auth(
        project: String,
        region: String,
        auth_manager: Arc<dyn gcp_auth::TokenProvider>,
    ) -> Self {
        Self {
            client: Client::new(),
            project,
            region,
            auth_manager,
        }
    }

    fn endpoint(&self, model: &str) -> String {
        let host = if self.region == "global" {
            "aiplatform.googleapis.com".to_string()
        } else {
            format!("{}-aiplatform.googleapis.com", self.region)
        };
        format!(
            "https://{host}/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:streamRawPredict",
            region = self.region,
            project = self.project,
            model = model,
        )
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
        let token = self
            .auth_manager
            .token(scopes)
            .await
            .context("Failed to get GCP auth token")?;
        let token_str = token.as_str().to_owned();

        let url = self.endpoint(&config.model);
        let body = build_request_body(messages, config)?;
        let thinking_enabled = config.thinking.as_ref().is_some_and(|tc| tc.enabled);

        let mut request = self
            .client
            .post(&url)
            .bearer_auth(&token_str)
            .header("Content-Type", "application/json");

        if thinking_enabled {
            request = request.header("anthropic-beta", "interleaved-thinking-2025-05-14");
        }

        let response = request
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Vertex AI")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<failed to read response body>"));
            anyhow::bail!("Vertex AI returned {}: {}", status, body);
        }

        let byte_stream = response.bytes_stream();
        let mut sse_parser = VertexSseParser::new();
        let event_stream = create_sse_event_stream(byte_stream, move |data| {
            sse_parser.fill_buffer(data)?;
            let events = std::mem::take(&mut sse_parser.event_buffer);
            Ok(events)
        });
        Ok(event_stream)
    }
}

fn filter_content_blocks<'a>(
    blocks: &'a [crate::types::ContentBlock],
    warnings: &mut Vec<String>,
) -> Vec<&'a crate::types::ContentBlock> {
    blocks
        .iter()
        .filter(|block| {
            if let crate::types::ContentBlock::Thinking { signature, .. } = block
                && signature.is_empty()
            {
                warnings.push(
                    "dropping thinking block with empty signature from history \
                     (corrupted session data); it cannot be replayed."
                        .to_string(),
                );
                return false;
            }
            true
        })
        .collect()
}

fn build_request_body(messages: &[Message], config: &RequestConfig) -> Result<serde_json::Value> {
    let messages_json: Vec<serde_json::Value> = messages
        .iter()
        .filter_map(|m| {
            let mut warnings = Vec::new();
            let content = filter_content_blocks(&m.content, &mut warnings);
            for w in warnings {
                crate::logging::log_warn(&w);
            }
            if content.is_empty() {
                return None;
            }
            Some(serde_json::json!({
                "role": m.role,
                "content": content,
            }))
        })
        .collect();

    let mut max_tokens = config.max_tokens as u64;

    let thinking_json = config.thinking.as_ref().and_then(|tc| {
        if !tc.enabled {
            return None;
        }
        let display = match tc.display {
            Some(crate::types::ThinkingDisplay::Omitted) => "omitted",
            _ => "summarized",
        };
        match &tc.mode {
            crate::types::ThinkingMode::Budget { tokens } => {
                let budget = *tokens as u64;
                max_tokens += budget;
                Some(serde_json::json!({
                    "type": "enabled",
                    "budget_tokens": budget,
                    "display": display
                }))
            }
            crate::types::ThinkingMode::Adaptive => Some(serde_json::json!({
                "type": "adaptive",
                "display": display
            })),
        }
    });

    let mut body = serde_json::json!({
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": max_tokens,
        "stream": true,
        "messages": messages_json,
    });

    if let Some(thinking) = thinking_json {
        body["thinking"] = thinking;
    }

    if !config.tools.is_empty() {
        body["tools"] =
            serde_json::to_value(&config.tools).context("Failed to serialize tool definitions")?;
        body["tool_choice"] = serde_json::json!({
            "type": "auto",
            "disable_parallel_tool_use": false
        });
    }

    Ok(body)
}

/// Convenience wrapper for single-event parsing without state.
/// For testing individual events in isolation; use VertexSseParser for sequences.
pub fn parse_sse_data(data: &str) -> Result<Option<StreamEvent>> {
    VertexSseParser::new().parse(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolDefinition;

    #[test]
    fn build_request_body_uses_max_tokens_and_model_from_config() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 32768,
            tools: vec![],
            thinking: None,
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert_eq!(body["max_tokens"], 32768);
        assert_eq!(body["anthropic_version"], ANTHROPIC_VERSION);
    }

    #[test]
    fn build_request_body_omits_model_field_for_vertex_ai() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: None,
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert!(
            body.get("model").is_none() || body["model"].is_null(),
            "model must not be in the request body; Vertex AI embeds it in the URL"
        );
    }

    #[test]
    fn build_request_body_includes_tools_when_non_empty() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![ToolDefinition {
                name: "bash".to_string(),
                description: "Run a bash command".to_string(),
                input_schema: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            }],
            thinking: None,
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        let tools = body["tools"].as_array().expect("tools should be an array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "bash");
        assert_eq!(tools[0]["description"], "Run a bash command");
    }

    #[test]
    fn build_request_body_includes_tool_choice_with_parallel_enabled_when_tools_present() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![ToolDefinition {
                name: "bash".to_string(),
                description: "Run a bash command".to_string(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            }],
            thinking: None,
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert_eq!(body["tool_choice"]["type"], "auto");
        assert_eq!(body["tool_choice"]["disable_parallel_tool_use"], false);
    }

    #[test]
    fn build_request_body_omits_tools_key_when_empty() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: None,
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert!(
            body.get("tools").is_none() || body["tools"].is_null(),
            "tools must not be in the request body when empty"
        );
        assert!(
            body.get("tool_choice").is_none() || body["tool_choice"].is_null(),
            "tool_choice must not be present when there are no tools"
        );
    }

    #[test]
    fn parser_content_block_start_tool_use_returns_tool_use_start() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"bash"}}"#;
        let result = parser.parse(data).expect("should parse successfully");
        match result {
            Some(StreamEvent::ToolUseStart { id, name }) => {
                assert_eq!(id, "toolu_01");
                assert_eq!(name, "bash");
            }
            other => panic!("expected ToolUseStart, got {:?}", other),
        }
    }

    #[test]
    fn parser_input_json_delta_returns_tool_use_delta() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"command\":"}}"#;
        let result = parser.parse(data).expect("should parse successfully");
        match result {
            Some(StreamEvent::ToolUseDelta(chunk)) => {
                assert_eq!(chunk, "{\"command\":");
            }
            other => panic!("expected ToolUseDelta, got {:?}", other),
        }
    }

    #[test]
    fn parser_content_block_stop_after_tool_use_returns_tool_use_done() {
        let mut parser = VertexSseParser::new();
        parser
            .parse(r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"bash"}}"#)
            .expect("should parse block_start");
        let result = parser
            .parse(r#"{"type":"content_block_stop","index":1}"#)
            .expect("should parse block_stop");
        assert!(
            matches!(result, Some(StreamEvent::ToolUseDone)),
            "expected ToolUseDone, got {:?}",
            result
        );
    }

    #[test]
    fn parser_content_block_stop_after_text_block_returns_none() {
        let mut parser = VertexSseParser::new();
        parser
            .parse(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#)
            .expect("should parse block_start");
        let result = parser
            .parse(r#"{"type":"content_block_stop","index":0}"#)
            .expect("should parse block_stop");
        assert!(
            result.is_none(),
            "stopping a text block should return None, got {:?}",
            result
        );
    }

    #[test]
    fn parser_mixed_text_and_tool_use_content_block_stop_fires_only_for_tool_use() {
        let mut parser = VertexSseParser::new();
        parser
            .parse(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#)
            .expect("text block_start");
        parser
            .parse(r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"bash"}}"#)
            .expect("tool_use block_start");

        let text_stop = parser
            .parse(r#"{"type":"content_block_stop","index":0}"#)
            .expect("text block_stop");
        assert!(
            text_stop.is_none(),
            "text block_stop should be None, got {:?}",
            text_stop
        );

        let tool_stop = parser
            .parse(r#"{"type":"content_block_stop","index":1}"#)
            .expect("tool_use block_stop");
        assert!(
            matches!(tool_stop, Some(StreamEvent::ToolUseDone)),
            "tool_use block_stop should be ToolUseDone, got {:?}",
            tool_stop
        );
    }

    #[test]
    fn parser_content_block_start_tool_use_missing_id_returns_error() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","name":"bash"}}"#;
        assert!(
            parser.parse(data).is_err(),
            "missing 'id' should return error"
        );
    }

    #[test]
    fn parser_content_block_start_tool_use_missing_name_returns_error() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01"}}"#;
        assert!(
            parser.parse(data).is_err(),
            "missing 'name' should return error"
        );
    }

    #[test]
    fn parser_input_json_delta_missing_partial_json_returns_error() {
        let mut parser = VertexSseParser::new();
        let data =
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta"}}"#;
        assert!(
            parser.parse(data).is_err(),
            "missing 'partial_json' should return error"
        );
    }

    #[test]
    fn parser_text_delta_returns_text_delta() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let result = parser.parse(data).expect("should parse successfully");
        match result {
            Some(StreamEvent::TextDelta(text)) => assert_eq!(text, "Hello"),
            other => panic!("expected TextDelta, got {:?}", other),
        }
    }

    #[test]
    fn parser_content_block_start_text_type_returns_none() {
        let mut parser = VertexSseParser::new();
        let data =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let result = parser.parse(data).expect("should parse successfully");
        assert!(
            result.is_none(),
            "text content_block_start should return None"
        );
    }

    #[test]
    fn parser_fill_buffer_populates_event_buffer() {
        let mut parser = VertexSseParser::new();
        parser
            .fill_buffer(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#)
            .expect("should fill buffer successfully");
        assert_eq!(
            parser.event_buffer.len(),
            1,
            "should have 1 event in buffer"
        );
    }

    #[test]
    fn parse_returns_buffered_events_one_at_a_time() {
        let mut parser = VertexSseParser::new();
        // Fill buffer with an event
        parser
            .fill_buffer(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#)
            .expect("should fill buffer successfully");

        // First parse should return the event
        let result1 = parser.parse("").expect("should parse successfully");
        assert!(result1.is_some(), "first parse should return Some(event)");

        // Second parse should return None (buffer is now empty)
        let result2 = parser.parse("").expect("should parse successfully");
        assert!(result2.is_none(), "second parse should return None");
    }

    #[test]
    fn fill_buffer_accumulates_multiple_events() {
        let mut parser = VertexSseParser::new();

        // Fill buffer with first event
        parser
            .fill_buffer(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#)
            .expect("should fill first event");

        // Fill buffer with second event
        parser
            .fill_buffer(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" World"}}"#)
            .expect("should fill second event");

        // Fill buffer with third event (message_stop)
        parser
            .fill_buffer(r#"{"type":"message_stop"}"#)
            .expect("should fill third event");

        // Buffer should have 3 events
        assert_eq!(
            parser.event_buffer.len(),
            3,
            "buffer should contain 3 events"
        );

        // Parse should return events one at a time
        let event1 = parser.parse("").expect("first parse");
        assert!(event1.is_some(), "should get first event");

        let event2 = parser.parse("").expect("second parse");
        assert!(event2.is_some(), "should get second event");

        let event3 = parser.parse("").expect("third parse");
        assert!(event3.is_some(), "should get third event");

        let event4 = parser.parse("").expect("fourth parse");
        assert!(event4.is_none(), "buffer should be empty after 3 events");
    }

    #[test]
    fn parser_thinking_delta_returns_thinking_delta() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"Let me reason about this"}}"#;
        let result = parser.parse(data).expect("should parse successfully");
        match result {
            Some(StreamEvent::ThinkingDelta(text)) => {
                assert_eq!(text, "Let me reason about this");
            }
            other => panic!("expected ThinkingDelta, got {:?}", other),
        }
    }

    #[test]
    fn parser_thinking_block_start_does_not_emit_event() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":""}}"#;
        let result = parser.parse(data).expect("should parse successfully");
        assert!(
            result.is_none(),
            "thinking content_block_start should return None, got {:?}",
            result
        );
    }

    #[test]
    fn parser_redacted_thinking_block_start_captures_data() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_start","index":2,"content_block":{"type":"redacted_thinking","data":"aGVsbG8gd29ybGQ="}}"#;
        let result = parser.parse(data).expect("should parse successfully");
        match result {
            Some(StreamEvent::ThinkingDelta(data)) => {
                assert_eq!(data, "aGVsbG8gd29ybGQ=");
            }
            other => panic!(
                "expected ThinkingDelta for redacted_thinking, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn parser_redacted_thinking_block_start_without_data_does_not_emit() {
        let mut parser = VertexSseParser::new();
        let data = r#"{"type":"content_block_start","index":2,"content_block":{"type":"redacted_thinking"}}"#;
        let result = parser.parse(data).expect("should parse successfully");
        assert!(
            result.is_none(),
            "redacted_thinking without data should return None, got {:?}",
            result
        );
    }

    #[test]
    fn parser_content_block_stop_after_thinking_emits_signature_via_signature_delta() {
        let mut parser = VertexSseParser::new();
        parser
            .parse(r#"{"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":""}}"#)
            .expect("should parse thinking block_start");
        parser
            .parse(r#"{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"sig_abc123"}}"#)
            .expect("should parse signature_delta");
        let result = parser
            .parse(r#"{"type":"content_block_stop","index":1}"#)
            .expect("should parse block_stop");
        match result {
            Some(StreamEvent::ThinkingSignature(sig)) => {
                assert_eq!(sig, "sig_abc123");
            }
            other => panic!("expected ThinkingSignature, got {:?}", other),
        }
    }

    #[test]
    fn parser_chunked_signature_delta_accumulates() {
        let mut parser = VertexSseParser::new();
        parser
            .parse(r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#)
            .expect("block_start");
        parser
            .parse(r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"chunk1_"}}"#)
            .expect("first signature_delta");
        parser
            .parse(r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"chunk2"}}"#)
            .expect("second signature_delta");
        let result = parser
            .parse(r#"{"type":"content_block_stop","index":0}"#)
            .expect("block_stop");
        match result {
            Some(StreamEvent::ThinkingSignature(sig)) => {
                assert_eq!(sig, "chunk1_chunk2");
            }
            other => panic!("expected ThinkingSignature, got {:?}", other),
        }
    }

    #[test]
    fn parser_round_trip_signature_survives_build_request_body() {
        let mut parser = VertexSseParser::new();
        parser
            .parse(r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#)
            .expect("block_start");
        parser
            .parse(r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think"}}"#)
            .expect("thinking_delta");
        parser
            .parse(r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"round_trip_sig"}}"#)
            .expect("signature_delta");
        // Drain ThinkingDelta from buffer
        parser.parse("").expect("drain");
        let stop_result = parser
            .parse(r#"{"type":"content_block_stop","index":0}"#)
            .expect("block_stop");
        let sig = match stop_result {
            Some(StreamEvent::ThinkingSignature(s)) => s,
            other => panic!("expected ThinkingSignature, got {:?}", other),
        };

        let message = Message {
            role: crate::types::Role::Assistant,
            content: vec![crate::types::ContentBlock::Thinking {
                text: "Let me think".to_string(),
                signature: sig,
            }],
        };
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: None,
        };
        let body = build_request_body(&[message], &config).expect("build");
        let content = &body["messages"][0]["content"];
        let block = &content[0];
        assert_eq!(block["type"], "thinking");
        assert_eq!(block["signature"], "round_trip_sig");
    }

    #[test]
    fn build_request_body_drops_empty_signature_thinking_blocks() {
        let messages = vec![Message {
            role: crate::types::Role::Assistant,
            content: vec![
                crate::types::ContentBlock::Thinking {
                    text: "some reasoning".to_string(),
                    signature: String::new(),
                },
                crate::types::ContentBlock::Text("visible text".to_string()),
            ],
        }];
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: None,
        };
        let body = build_request_body(&messages, &config).expect("build");
        let content = body["messages"][0]["content"]
            .as_array()
            .expect("content array");
        assert_eq!(
            content.len(),
            1,
            "empty-signature thinking block must be dropped"
        );
        assert_eq!(content[0]["type"], "text");
    }

    #[test]
    fn build_request_body_preserves_redacted_thinking_and_other_blocks() {
        let messages = vec![Message {
            role: crate::types::Role::Assistant,
            content: vec![
                crate::types::ContentBlock::RedactedThinking {
                    data: "opaque_data".to_string(),
                },
                crate::types::ContentBlock::Thinking {
                    text: "bad".to_string(),
                    signature: String::new(),
                },
                crate::types::ContentBlock::Text("answer".to_string()),
            ],
        }];
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: None,
        };
        let body = build_request_body(&messages, &config).expect("build");
        let content = body["messages"][0]["content"]
            .as_array()
            .expect("content array");
        assert_eq!(
            content.len(),
            2,
            "only empty-signature thinking block dropped"
        );
        assert_eq!(content[0]["type"], "redacted_thinking");
        assert_eq!(content[1]["type"], "text");
    }

    #[test]
    fn build_request_body_with_budget_thinking_includes_thinking_params() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                mode: crate::types::ThinkingMode::Budget { tokens: 16384 },
                enabled: true,
                display: None,
            }),
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 16384);
        assert_eq!(body["thinking"]["display"], "summarized");
    }

    #[test]
    fn build_request_body_with_adaptive_thinking_includes_thinking_params() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                mode: crate::types::ThinkingMode::Adaptive,
                enabled: true,
                display: None,
            }),
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["thinking"]["display"], "summarized");
    }

    #[test]
    fn build_request_body_with_thinking_adjusts_max_tokens() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                mode: crate::types::ThinkingMode::Budget { tokens: 16384 },
                enabled: true,
                display: None,
            }),
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert_eq!(body["max_tokens"], 24576);
    }

    #[test]
    fn build_request_body_without_thinking_has_no_thinking_field() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: None,
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert!(
            body.get("thinking").is_none(),
            "thinking must not be in the request body when None"
        );
    }

    #[test]
    fn build_request_body_with_disabled_thinking_has_no_thinking_field() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                mode: crate::types::ThinkingMode::Adaptive,
                enabled: false,
                display: None,
            }),
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert!(
            body.get("thinking").is_none(),
            "thinking must not be in the request body when disabled"
        );
    }

    #[test]
    fn build_request_body_adaptive_thinking_defaults_display_to_summarized() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                mode: crate::types::ThinkingMode::Adaptive,
                enabled: true,
                display: None,
            }),
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert_eq!(body["thinking"]["display"], "summarized");
    }

    #[test]
    fn build_request_body_budget_thinking_defaults_display_to_summarized() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                mode: crate::types::ThinkingMode::Budget { tokens: 8192 },
                enabled: true,
                display: None,
            }),
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert_eq!(body["thinking"]["display"], "summarized");
    }

    #[test]
    fn build_request_body_thinking_explicit_omitted_serializes_omitted() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                mode: crate::types::ThinkingMode::Adaptive,
                enabled: true,
                display: Some(crate::types::ThinkingDisplay::Omitted),
            }),
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert_eq!(body["thinking"]["display"], "omitted");
    }

    #[test]
    fn build_request_body_thinking_explicit_summarized_serializes_summarized() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                mode: crate::types::ThinkingMode::Adaptive,
                enabled: true,
                display: Some(crate::types::ThinkingDisplay::Summarized),
            }),
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert_eq!(body["thinking"]["display"], "summarized");
    }

    #[test]
    fn build_request_body_budget_thinking_explicit_omitted_serializes_omitted() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                mode: crate::types::ThinkingMode::Budget { tokens: 8192 },
                enabled: true,
                display: Some(crate::types::ThinkingDisplay::Omitted),
            }),
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 8192);
        assert_eq!(body["thinking"]["display"], "omitted");
    }

    #[test]
    fn build_request_body_budget_thinking_explicit_summarized_serializes_summarized() {
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                mode: crate::types::ThinkingMode::Budget { tokens: 8192 },
                enabled: true,
                display: Some(crate::types::ThinkingDisplay::Summarized),
            }),
        };
        let body = build_request_body(&[], &config).expect("should build successfully");
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 8192);
        assert_eq!(body["thinking"]["display"], "summarized");
    }

    #[test]
    fn parser_thinking_block_with_no_signature_delta_emits_empty_signature() {
        let mut parser = VertexSseParser::new();
        parser
            .parse(r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#)
            .expect("block_start");
        let result = parser
            .parse(r#"{"type":"content_block_stop","index":0}"#)
            .expect("block_stop");
        match result {
            Some(StreamEvent::ThinkingSignature(sig)) => {
                assert!(
                    sig.is_empty(),
                    "no signature_delta → empty signature emitted"
                );
            }
            other => panic!("expected ThinkingSignature, got {:?}", other),
        }
    }

    #[test]
    fn parser_signature_delta_missing_index_returns_error() {
        let mut parser = VertexSseParser::new();
        parser
            .parse(r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#)
            .expect("block_start");
        let result = parser.parse(
            r#"{"type":"content_block_delta","delta":{"type":"signature_delta","signature":"abc"}}"#,
        );
        assert!(
            result.is_err(),
            "signature_delta with missing index should return error"
        );
    }

    #[test]
    fn parser_signature_delta_non_string_signature_returns_error() {
        let mut parser = VertexSseParser::new();
        parser
            .parse(r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#)
            .expect("block_start");
        let result = parser.parse(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":42}}"#,
        );
        assert!(
            result.is_err(),
            "signature_delta with non-string signature should return error"
        );
    }

    #[test]
    fn build_request_body_drops_thinking_only_message_with_empty_signatures() {
        let messages = vec![
            Message {
                role: crate::types::Role::Assistant,
                content: vec![crate::types::ContentBlock::Thinking {
                    text: "reasoning".to_string(),
                    signature: String::new(),
                }],
            },
            Message {
                role: crate::types::Role::User,
                content: vec![crate::types::ContentBlock::Text("follow-up".to_string())],
            },
        ];
        let config = RequestConfig {
            model: "claude-test".to_string(),
            max_tokens: 8192,
            tools: vec![],
            thinking: None,
        };
        let body = build_request_body(&messages, &config).expect("build");
        let msgs = body["messages"].as_array().expect("messages array");
        assert_eq!(
            msgs.len(),
            1,
            "message with only empty-signature thinking blocks must be dropped"
        );
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn filter_content_blocks_drops_empty_signature_thinking_and_records_warning() {
        use crate::types::ContentBlock;
        let blocks = vec![
            ContentBlock::Thinking {
                text: "reasoning".to_string(),
                signature: String::new(), // empty → should be dropped
            },
            ContentBlock::Text("hello".to_string()),
        ];
        let mut warnings = Vec::new();
        let filtered = filter_content_blocks(&blocks, &mut warnings);
        assert_eq!(filtered.len(), 1, "only the Text block should remain");
        assert!(
            matches!(filtered[0], ContentBlock::Text(_)),
            "remaining block should be Text"
        );
        assert_eq!(warnings.len(), 1, "one warning should be recorded");
        assert!(
            warnings[0].contains("empty signature"),
            "warning should mention empty signature"
        );
    }
}
