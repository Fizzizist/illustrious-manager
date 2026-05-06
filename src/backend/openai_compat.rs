use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;

use super::LlmBackend;
use super::sse::create_sse_event_stream;
use crate::types::{BoxStream, ContentBlock, Message, RequestConfig, Role, StreamEvent};

/// Controls how the `enable_thinking` / reasoning field is serialised in the
/// request body.  Different OpenAI-compatible providers use different field
/// names or omit the field entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningStyle {
    /// z.ai: `"enable_thinking": true` when `config.thinking` is set and enabled.
    ZaiEnableThinking,
    /// vLLM/Qwen: `"chat_template_kwargs": {"enable_thinking": true}`.
    QwenChatTemplate,
    /// No reasoning field added (pass-through / model decides).
    Default,
    /// No reasoning field added (explicitly disabled).
    None,
}

/// Configuration for constructing an [`OpenAiCompatBackend`].
pub struct OpenAiCompatConfig {
    /// Base URL **without** a trailing slash and **without** `/chat/completions`.
    /// The backend appends `/chat/completions` automatically.
    pub base_url: String,
    /// Bearer token, or `None` if the endpoint does not require auth.
    pub api_key: Option<String>,
    /// Placeholder model string stored on the struct; the actual model is taken
    /// from [`RequestConfig::model`] at call time.
    pub model: String,
    /// Hard-coded `max_tokens` override.  When `None` the value from
    /// [`RequestConfig::max_tokens`] is used.
    pub max_tokens: Option<u32>,
    /// How (if at all) extended-thinking is communicated to the provider.
    pub reasoning: ReasoningStyle,
}

/// Stateful SSE parser for OpenAI-compatible streaming responses.
///
/// Buffers extra events that arise when a single SSE chunk contains multiple
/// tool calls so they can be drained one at a time.
#[derive(Default)]
pub struct OpenAiCompatSseParser {
    event_buffer: Vec<StreamEvent>,
}

impl OpenAiCompatSseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse one SSE `data:` payload and return the first buffered event, if
    /// any.  Pass an empty string to drain the internal buffer without
    /// consuming new data (useful in tests).
    pub fn parse(&mut self, data: &str) -> Result<Option<StreamEvent>> {
        if !data.is_empty() {
            self.fill_buffer(data)?;
        }
        if !self.event_buffer.is_empty() {
            return Ok(Some(self.event_buffer.remove(0)));
        }
        Ok(None)
    }

    pub fn fill_buffer(&mut self, data: &str) -> Result<()> {
        if data == "[DONE]" {
            self.event_buffer.push(StreamEvent::Done);
            return Ok(());
        }

        let json: serde_json::Value = serde_json::from_str(data)
            .with_context(|| format!("Failed to parse SSE data: {}", data))?;

        // Determine finish_reason and any co-located usage.
        let finish_reason = json["choices"][0]["finish_reason"].as_str();
        let input_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let output_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;

        // Terminal chunk for tool calls: ToolUseDone first, then usage.
        if finish_reason == Some("tool_calls") {
            self.event_buffer.push(StreamEvent::ToolUseDone);
            if input_tokens > 0 || output_tokens > 0 {
                self.event_buffer.push(StreamEvent::Usage {
                    input_tokens,
                    output_tokens,
                    stop_reason: "tool_calls".to_string(),
                });
            }
            return Ok(());
        }

        // Hard limit: propagate immediately as an error.
        if finish_reason == Some("length") {
            return Err(anyhow::anyhow!(
                "Response truncated: max_tokens limit reached \
                 (input_tokens={}, output_tokens={}). \
                 Increase max_tokens in your config.",
                input_tokens,
                output_tokens,
            ));
        }

        // Emit content deltas before any usage event so the TUI always
        // receives the last character before stats.

        // No early return after ThinkingDelta — fall through so a co-located
        // `content` field in the same chunk is also processed.
        if let Some(reasoning) = json["choices"][0]["delta"]["reasoning_content"].as_str()
            && !reasoning.is_empty()
        {
            self.event_buffer
                .push(StreamEvent::ThinkingDelta(reasoning.to_string()));
        }

        if let Some(content) = json["choices"][0]["delta"]["content"].as_str()
            && !content.is_empty()
        {
            self.event_buffer
                .push(StreamEvent::TextDelta(content.to_string()));
        }

        if let Some(tool_calls) = json["choices"][0]["delta"]["tool_calls"].as_array() {
            for tool_call in tool_calls {
                if let Some(index) = tool_call["index"].as_u64()
                    && let Some(function) = tool_call["function"].as_object()
                {
                    if let Some(name) = function.get("name").and_then(|v| v.as_str()) {
                        let id = format!("tool_{}", index);
                        self.event_buffer.push(StreamEvent::ToolUseStart {
                            id,
                            name: name.to_string(),
                        });
                    }

                    if let Some(arguments) = function.get("arguments").and_then(|v| v.as_str())
                        && !arguments.is_empty()
                    {
                        self.event_buffer
                            .push(StreamEvent::ToolUseDelta(arguments.to_string()));
                    }
                }
            }
        }

        // Emit usage after content deltas.  Also handles standalone usage-only
        // frames (no `choices` array) emitted by providers using stream_options.
        let choices_empty = json["choices"].as_array().is_none_or(|a| a.is_empty());
        let has_usage = input_tokens > 0 || output_tokens > 0;
        let normal_stop = finish_reason.is_some_and(|r| r != "tool_calls" && r != "length");
        if has_usage && (normal_stop || choices_empty) {
            self.event_buffer.push(StreamEvent::Usage {
                input_tokens,
                output_tokens,
                stop_reason: finish_reason.unwrap_or("stop").to_string(),
            });
        }

        Ok(())
    }
}

/// Generic OpenAI-compatible streaming backend.
///
/// Handles message serialisation, request dispatch, and SSE parsing.
/// Provider-specific quirks (auth header, reasoning field name, base URL)
/// are controlled via [`OpenAiCompatConfig`].
#[derive(Debug)]
pub struct OpenAiCompatBackend {
    client: Client,
    endpoint: String,
    api_key: Option<String>,
    max_tokens_override: Option<u32>,
    reasoning: ReasoningStyle,
}

impl OpenAiCompatBackend {
    pub fn new(config: OpenAiCompatConfig) -> Result<Self> {
        let base = config.base_url.trim_end_matches('/');
        if base.ends_with("/chat/completions") {
            anyhow::bail!(
                "base_url must not include '/chat/completions' — provide the base URL only \
                 (e.g. 'https://example.com/v1') and the backend will append the path automatically."
            );
        }
        if base.contains('?') {
            anyhow::bail!(
                "base_url must not contain a query string — provide only the base URL \
                 (e.g. 'https://example.com/v1')."
            );
        }
        let endpoint = format!("{}/chat/completions", base);
        Ok(Self {
            client: Client::new(),
            endpoint,
            api_key: config.api_key,
            max_tokens_override: config.max_tokens,
            reasoning: config.reasoning,
        })
    }

    /// Expose the resolved endpoint URL (used in structural tests).
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Build request headers.  The `Authorization` header is included only
    /// when an API key is present and non-empty.
    pub fn build_headers(&self) -> Result<reqwest::header::HeaderMap> {
        use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(ref key) = self.api_key
            && !key.is_empty()
        {
            let value = HeaderValue::from_str(&format!("Bearer {key}"))
                .context("Failed to build Authorization header value")?;
            headers.insert(AUTHORIZATION, value);
        }
        Ok(headers)
    }

    fn build_request_body(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> serde_json::Value {
        let mut messages_json: Vec<serde_json::Value> = Vec::new();

        for m in messages {
            match m.role {
                Role::User => {
                    let has_tool_results = m
                        .content
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolResult { .. }));

                    if has_tool_results {
                        for block in &m.content {
                            match block {
                                ContentBlock::ToolResult {
                                    tool_use_id,
                                    content,
                                    ..
                                } => {
                                    messages_json.push(serde_json::json!({
                                        "role": "tool",
                                        "tool_call_id": tool_use_id,
                                        "content": content
                                    }));
                                }
                                ContentBlock::Text(text) => {
                                    messages_json.push(serde_json::json!({
                                        "role": "user",
                                        "content": [{"type": "text", "text": text}]
                                    }));
                                }
                                _ => {}
                            }
                        }
                    } else {
                        let content: Vec<serde_json::Value> =
                            m.content
                                .iter()
                                .map(|block| match block {
                                    ContentBlock::Text(text) => {
                                        serde_json::json!({"type": "text", "text": text})
                                    }
                                    other => serde_json::to_value(other)
                                        .unwrap_or(serde_json::Value::Null),
                                })
                                .collect();
                        messages_json.push(serde_json::json!({"role": "user", "content": content}));
                    }
                }
                Role::Assistant => {
                    let has_tool_use = m
                        .content
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolUse { .. }));

                    if has_tool_use {
                        let mut text_parts: Vec<String> = Vec::new();
                        let mut tool_calls: Vec<serde_json::Value> = Vec::new();

                        for block in &m.content {
                            match block {
                                ContentBlock::Text(text) => text_parts.push(text.clone()),
                                ContentBlock::ToolUse { id, name, input } => {
                                    let arguments = serde_json::to_string(input)
                                        .unwrap_or_else(|_| "{}".to_string());
                                    tool_calls.push(serde_json::json!({
                                        "id": id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": arguments
                                        }
                                    }));
                                }
                                ContentBlock::Thinking { .. }
                                | ContentBlock::RedactedThinking { .. } => {}
                                _ => {}
                            }
                        }

                        let mut msg = serde_json::json!({"role": "assistant"});
                        if !text_parts.is_empty() {
                            msg["content"] = serde_json::json!(text_parts.join(""));
                        }
                        msg["tool_calls"] = serde_json::json!(tool_calls);
                        messages_json.push(msg);
                    } else {
                        let content: Vec<serde_json::Value> = m
                            .content
                            .iter()
                            .filter_map(|block| match block {
                                ContentBlock::Text(text) => {
                                    Some(serde_json::json!({"type": "text", "text": text}))
                                }
                                ContentBlock::Thinking { .. }
                                | ContentBlock::RedactedThinking { .. } => None,
                                other => serde_json::to_value(other).ok(),
                            })
                            .collect();
                        messages_json
                            .push(serde_json::json!({"role": "assistant", "content": content}));
                    }
                }
            }
        }

        let max_tokens = self.max_tokens_override.unwrap_or(config.max_tokens);

        let mut body = serde_json::json!({
            "model": config.model,
            "messages": messages_json,
            "stream": true,
            "max_tokens": max_tokens,
        });

        if !config.tools.is_empty() {
            let tools_json: Vec<serde_json::Value> = config
                .tools
                .iter()
                .map(|tool| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools_json);
            body["parallel_tool_calls"] = serde_json::json!(true);
        }

        match self.reasoning {
            ReasoningStyle::ZaiEnableThinking => {
                if let Some(ref thinking_config) = config.thinking
                    && thinking_config.enabled
                {
                    body["enable_thinking"] = serde_json::json!(true);
                }
            }
            ReasoningStyle::QwenChatTemplate => {
                if let Some(ref thinking_config) = config.thinking
                    && thinking_config.enabled
                {
                    body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": true});
                }
            }
            ReasoningStyle::Default | ReasoningStyle::None => {}
        }

        body
    }
}

#[async_trait]
impl LlmBackend for OpenAiCompatBackend {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        let body = self.build_request_body(messages, config);

        let headers = self.build_headers()?;
        let response = self
            .client
            .post(&self.endpoint)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("Failed to send request to {}", self.endpoint))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<failed to read response body>"));
            anyhow::bail!("{} returned {}: {}", self.endpoint, status, body);
        }

        let byte_stream = response.bytes_stream();
        let mut parser = OpenAiCompatSseParser::new();
        let event_stream = create_sse_event_stream(byte_stream, move |data| {
            parser.fill_buffer(data)?;
            let events = std::mem::take(&mut parser.event_buffer);
            Ok(events)
        });
        Ok(event_stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Role;

    fn make_backend(reasoning: ReasoningStyle) -> OpenAiCompatBackend {
        OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: "https://api.example.com/v1".to_string(),
            api_key: Some("test-key".to_string()),
            model: String::new(),
            max_tokens: None,
            reasoning,
        })
        .expect("test config is always valid")
    }

    fn simple_config() -> RequestConfig {
        RequestConfig {
            model: "test-model".to_string(),
            max_tokens: 4096,
            tools: vec![],
            thinking: None,
        }
    }

    fn user_text(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text(text.to_string())],
        }
    }

    // --- endpoint construction ---

    #[test]
    fn endpoint_appends_chat_completions() {
        let b = make_backend(ReasoningStyle::None);
        assert_eq!(b.endpoint(), "https://api.example.com/v1/chat/completions");
    }

    #[test]
    fn endpoint_strips_trailing_slash_before_appending() {
        let b = OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: "https://api.example.com/v1/".to_string(),
            api_key: None,
            model: String::new(),
            max_tokens: None,
            reasoning: ReasoningStyle::None,
        })
        .expect("valid config");
        assert_eq!(b.endpoint(), "https://api.example.com/v1/chat/completions");
    }

    // --- basic request body shape ---

    #[test]
    fn build_request_body_includes_model_stream_max_tokens() {
        let b = make_backend(ReasoningStyle::None);
        let body = b.build_request_body(&[user_text("hi")], &simple_config());
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 4096);
    }

    #[test]
    fn max_tokens_override_takes_precedence_over_request_config() {
        let b = OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: "https://api.example.com/v1".to_string(),
            api_key: None,
            model: String::new(),
            max_tokens: Some(8192),
            reasoning: ReasoningStyle::None,
        })
        .expect("valid config");
        let body = b.build_request_body(&[user_text("hi")], &simple_config());
        assert_eq!(body["max_tokens"], 8192);
    }

    // --- message serialisation ---

    #[test]
    fn user_text_block_serialised_as_typed_object() {
        let b = make_backend(ReasoningStyle::None);
        let body = b.build_request_body(&[user_text("Hello")], &simple_config());
        let content0 = &body["messages"][0]["content"][0];
        assert_eq!(content0["type"], "text");
        assert_eq!(content0["text"], "Hello");
    }

    #[test]
    fn tool_result_maps_to_role_tool_message() {
        let b = make_backend(ReasoningStyle::None);
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tool_0".to_string(),
                content: "output".to_string(),
                is_error: false,
            }],
        }];
        let body = b.build_request_body(&messages, &simple_config());
        let msgs = body["messages"].as_array().expect("messages");
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "tool_0");
        assert_eq!(msgs[0]["content"], "output");
    }

    #[test]
    fn assistant_tool_use_serialised_as_tool_calls() {
        let b = make_backend(ReasoningStyle::None);
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "tool_0".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"command": "ls"}),
            }],
        }];
        let body = b.build_request_body(&messages, &simple_config());
        let msgs = body["messages"].as_array().expect("messages");
        let tool_calls = msgs[0]["tool_calls"].as_array().expect("tool_calls");
        assert_eq!(tool_calls[0]["id"], "tool_0");
        assert_eq!(tool_calls[0]["type"], "function");
        assert_eq!(tool_calls[0]["function"]["name"], "bash");
        let args: serde_json::Value = serde_json::from_str(
            tool_calls[0]["function"]["arguments"]
                .as_str()
                .expect("args"),
        )
        .expect("args json");
        assert_eq!(args["command"], "ls");
    }

    // --- tools field ---

    #[test]
    fn tools_field_present_and_parallel_tool_calls_set_when_tools_provided() {
        let b = make_backend(ReasoningStyle::None);
        let config = RequestConfig {
            model: "m".to_string(),
            max_tokens: 1024,
            tools: vec![crate::types::ToolDefinition {
                name: "bash".to_string(),
                description: "run bash".to_string(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            }],
            thinking: None,
        };
        let body = b.build_request_body(&[], &config);
        assert!(body["tools"].as_array().is_some());
        assert_eq!(body["parallel_tool_calls"], true);
    }

    #[test]
    fn tools_field_absent_when_no_tools() {
        let b = make_backend(ReasoningStyle::None);
        let body = b.build_request_body(&[], &simple_config());
        assert!(body.get("tools").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
    }

    // --- reasoning styles ---

    #[test]
    fn zai_enable_thinking_adds_field_when_thinking_enabled() {
        let b = make_backend(ReasoningStyle::ZaiEnableThinking);
        let config = RequestConfig {
            model: "m".to_string(),
            max_tokens: 1024,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                enabled: true,
                mode: crate::types::ThinkingMode::Adaptive,
            }),
        };
        let body = b.build_request_body(&[], &config);
        assert_eq!(body["enable_thinking"], true);
    }

    #[test]
    fn zai_enable_thinking_absent_when_thinking_disabled() {
        let b = make_backend(ReasoningStyle::ZaiEnableThinking);
        let body = b.build_request_body(&[], &simple_config());
        assert!(body.get("enable_thinking").is_none());
    }

    #[test]
    fn reasoning_none_never_adds_enable_thinking() {
        let b = make_backend(ReasoningStyle::None);
        let config = RequestConfig {
            model: "m".to_string(),
            max_tokens: 1024,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                enabled: true,
                mode: crate::types::ThinkingMode::Adaptive,
            }),
        };
        let body = b.build_request_body(&[], &config);
        assert!(body.get("enable_thinking").is_none());
    }

    // --- SSE parser ---

    #[test]
    fn parser_text_delta() {
        let mut p = OpenAiCompatSseParser::new();
        let event = p
            .parse(r#"{"choices":[{"delta":{"content":"Hello"}}]}"#)
            .unwrap();
        assert!(matches!(event, Some(StreamEvent::TextDelta(t)) if t == "Hello"));
    }

    #[test]
    fn parser_done_sentinel() {
        let mut p = OpenAiCompatSseParser::new();
        assert!(matches!(
            p.parse("[DONE]").unwrap(),
            Some(StreamEvent::Done)
        ));
    }

    #[test]
    fn parser_invalid_json_errors() {
        let mut p = OpenAiCompatSseParser::new();
        assert!(p.parse("not json").is_err());
    }

    #[test]
    fn parser_empty_delta_returns_none() {
        let mut p = OpenAiCompatSseParser::new();
        assert!(p.parse(r#"{"choices":[{"delta":{}}]}"#).unwrap().is_none());
    }

    #[test]
    fn parser_tool_use_start() {
        let mut p = OpenAiCompatSseParser::new();
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"bash","arguments":""}}]}}]}"#;
        let event = p.parse(data).unwrap();
        assert!(matches!(event, Some(StreamEvent::ToolUseStart { name, .. }) if name == "bash"));
    }

    #[test]
    fn parser_tool_use_delta() {
        let mut p = OpenAiCompatSseParser::new();
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd"}}]}}]}"#;
        let event = p.parse(data).unwrap();
        assert!(matches!(event, Some(StreamEvent::ToolUseDelta(d)) if d == "{\"cmd"));
    }

    #[test]
    fn parser_finish_reason_tool_calls_emits_tool_use_done() {
        let mut p = OpenAiCompatSseParser::new();
        let event = p
            .parse(r#"{"choices":[{"finish_reason":"tool_calls"}]}"#)
            .unwrap();
        assert!(matches!(event, Some(StreamEvent::ToolUseDone)));
    }

    #[test]
    fn parser_reasoning_content_emits_thinking_delta() {
        let mut p = OpenAiCompatSseParser::new();
        let data = r#"{"choices":[{"delta":{"reasoning_content":"step 1"}}]}"#;
        let event = p.parse(data).unwrap();
        assert!(matches!(event, Some(StreamEvent::ThinkingDelta(t)) if t == "step 1"));
    }

    #[test]
    fn parser_multiple_tool_calls_in_single_chunk_buffered_correctly() {
        let mut p = OpenAiCompatSseParser::new();
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"bash","arguments":""}},{"index":1,"function":{"name":"read_file","arguments":""}}]}}]}"#;

        let e1 = p.parse(data).unwrap();
        assert!(matches!(e1, Some(StreamEvent::ToolUseStart { name, .. }) if name == "bash"));

        let e2 = p.parse("").unwrap();
        assert!(matches!(e2, Some(StreamEvent::ToolUseStart { name, .. }) if name == "read_file"));

        assert!(p.parse("").unwrap().is_none());
    }

    #[test]
    fn parser_finish_reason_length_returns_error() {
        let mut p = OpenAiCompatSseParser::new();
        let data = r#"{"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}"#;
        assert!(p.parse(data).is_err());
    }

    // --- constructor validation ---

    #[test]
    fn new_rejects_base_url_with_chat_completions_suffix() {
        let result = OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: "https://example.com/v1/chat/completions".to_string(),
            api_key: None,
            model: String::new(),
            max_tokens: None,
            reasoning: ReasoningStyle::None,
        });
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("chat/completions"),
            "error should mention chat/completions; got: {msg}"
        );
    }

    #[test]
    fn new_accepts_base_url_without_trailing_slash() {
        let b = OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: "https://example.com/v1".to_string(),
            api_key: None,
            model: String::new(),
            max_tokens: None,
            reasoning: ReasoningStyle::None,
        })
        .expect("valid config");
        assert_eq!(b.endpoint(), "https://example.com/v1/chat/completions");
    }

    #[test]
    fn new_with_no_api_key_succeeds() {
        let result = OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: "https://example.com/v1".to_string(),
            api_key: None,
            model: String::new(),
            max_tokens: None,
            reasoning: ReasoningStyle::None,
        });
        assert!(result.is_ok(), "None api_key should be accepted");
    }

    // --- build_headers ---

    #[test]
    fn build_headers_includes_auth_when_api_key_present() {
        let b = OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: "https://example.com/v1".to_string(),
            api_key: Some("my-key".to_string()),
            model: String::new(),
            max_tokens: None,
            reasoning: ReasoningStyle::None,
        })
        .expect("valid config");
        let headers = b.build_headers().expect("headers built");
        assert!(
            headers.contains_key(reqwest::header::AUTHORIZATION),
            "Authorization header must be present when api_key is Some"
        );
    }

    #[test]
    fn build_headers_omits_auth_when_api_key_none() {
        let b = OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: "https://example.com/v1".to_string(),
            api_key: None,
            model: String::new(),
            max_tokens: None,
            reasoning: ReasoningStyle::None,
        })
        .expect("valid config");
        let headers = b.build_headers().expect("headers built");
        assert!(
            !headers.contains_key(reqwest::header::AUTHORIZATION),
            "Authorization header must be absent when api_key is None"
        );
    }

    #[test]
    fn build_headers_omits_auth_when_api_key_empty() {
        let b = OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: "https://example.com/v1".to_string(),
            api_key: Some(String::new()),
            model: String::new(),
            max_tokens: None,
            reasoning: ReasoningStyle::None,
        })
        .expect("valid config");
        let headers = b.build_headers().expect("headers built");
        assert!(
            !headers.contains_key(reqwest::header::AUTHORIZATION),
            "Authorization header must be absent when api_key is empty"
        );
    }

    // --- max_tokens precedence ---

    #[test]
    fn build_request_body_max_tokens_config_overrides_request_config() {
        let b = OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: "https://example.com/v1".to_string(),
            api_key: None,
            model: String::new(),
            max_tokens: Some(16384),
            reasoning: ReasoningStyle::None,
        })
        .expect("valid config");
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 1024,
            tools: vec![],
            thinking: None,
        };
        let body = b.build_request_body(&[user_text("hi")], &config);
        assert_eq!(
            body["max_tokens"], 16384,
            "config max_tokens should override RequestConfig"
        );
    }

    #[test]
    fn build_request_body_max_tokens_falls_back_to_request_config_when_none() {
        let b = OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: "https://example.com/v1".to_string(),
            api_key: None,
            model: String::new(),
            max_tokens: None,
            reasoning: ReasoningStyle::None,
        })
        .expect("valid config");
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 1024,
            tools: vec![],
            thinking: None,
        };
        let body = b.build_request_body(&[user_text("hi")], &config);
        assert_eq!(
            body["max_tokens"], 1024,
            "should fall back to RequestConfig.max_tokens"
        );
    }

    // --- ReasoningStyle variants ---

    #[test]
    fn build_request_body_qwen_chat_template_writes_chat_template_kwargs() {
        let b = make_backend(ReasoningStyle::QwenChatTemplate);
        let config = RequestConfig {
            model: "qwen".to_string(),
            max_tokens: 4096,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                enabled: true,
                mode: crate::types::ThinkingMode::Adaptive,
            }),
        };
        let body = b.build_request_body(&[user_text("hi")], &config);
        assert_eq!(
            body["chat_template_kwargs"]["enable_thinking"], true,
            "chat_template_kwargs.enable_thinking must be true for QwenChatTemplate"
        );
        assert!(
            body.get("enable_thinking").is_none(),
            "top-level enable_thinking must not be set for QwenChatTemplate"
        );
    }

    #[test]
    fn build_request_body_zai_enable_thinking_writes_top_level_enable_thinking() {
        let b = make_backend(ReasoningStyle::ZaiEnableThinking);
        let config = RequestConfig {
            model: "zai".to_string(),
            max_tokens: 4096,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                enabled: true,
                mode: crate::types::ThinkingMode::Adaptive,
            }),
        };
        let body = b.build_request_body(&[user_text("hi")], &config);
        assert_eq!(
            body["enable_thinking"], true,
            "top-level enable_thinking must be true for ZaiEnableThinking"
        );
        assert!(
            body.get("chat_template_kwargs").is_none(),
            "chat_template_kwargs must not be set for ZaiEnableThinking"
        );
    }

    #[test]
    fn build_request_body_reasoning_none_writes_no_thinking_fields() {
        let b = make_backend(ReasoningStyle::None);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 4096,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                enabled: true,
                mode: crate::types::ThinkingMode::Adaptive,
            }),
        };
        let body = b.build_request_body(&[user_text("hi")], &config);
        assert!(
            body.get("enable_thinking").is_none(),
            "enable_thinking must not be set for None"
        );
        assert!(
            body.get("chat_template_kwargs").is_none(),
            "chat_template_kwargs must not be set for None"
        );
    }

    #[test]
    fn build_request_body_reasoning_default_writes_no_thinking_fields() {
        let b = make_backend(ReasoningStyle::Default);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 4096,
            tools: vec![],
            thinking: Some(crate::types::ThinkingConfig {
                enabled: true,
                mode: crate::types::ThinkingMode::Adaptive,
            }),
        };
        let body = b.build_request_body(&[user_text("hi")], &config);
        assert!(
            body.get("enable_thinking").is_none(),
            "enable_thinking must not be set for Default"
        );
        assert!(
            body.get("chat_template_kwargs").is_none(),
            "chat_template_kwargs must not be set for Default"
        );
    }

    // --- fill_buffer ordering and new behavior tests ---

    #[test]
    fn parser_tool_calls_finish_with_usage_emits_tool_use_done_then_usage() {
        // Finding 9b: tool_calls finish_reason + non-zero usage must drain
        // ToolUseDone before Usage.
        let mut p = OpenAiCompatSseParser::new();
        let data = r#"{"choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":50,"completion_tokens":20}}"#;
        p.fill_buffer(data).expect("parse ok");

        let e1 = p.event_buffer.remove(0);
        assert!(
            matches!(e1, StreamEvent::ToolUseDone),
            "first event must be ToolUseDone; got {e1:?}"
        );
        let e2 = p.event_buffer.remove(0);
        assert!(
            matches!(
                e2,
                StreamEvent::Usage {
                    input_tokens: 50,
                    output_tokens: 20,
                    ..
                }
            ),
            "second event must be Usage(50,20); got {e2:?}"
        );
        assert!(p.event_buffer.is_empty());
    }

    #[test]
    fn parser_stop_finish_reason_usage_comes_after_text_delta() {
        // Finding 4: when finish_reason="stop" co-located with content, Usage
        // must be buffered AFTER TextDelta.
        let mut p = OpenAiCompatSseParser::new();
        let data = r#"{"choices":[{"finish_reason":"stop","delta":{"content":"last"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        p.fill_buffer(data).expect("parse ok");

        let e1 = p.event_buffer.remove(0);
        assert!(
            matches!(e1, StreamEvent::TextDelta(ref t) if t == "last"),
            "first event must be TextDelta(\"last\"); got {e1:?}"
        );
        let e2 = p.event_buffer.remove(0);
        assert!(
            matches!(e2, StreamEvent::Usage { .. }),
            "second event must be Usage; got {e2:?}"
        );
        assert!(p.event_buffer.is_empty());
    }

    #[test]
    fn parser_reasoning_and_content_in_same_chunk_both_emitted() {
        // Finding 3: removing the early return after ThinkingDelta must allow
        // a co-located content field to also be emitted.
        let mut p = OpenAiCompatSseParser::new();
        let data = r#"{"choices":[{"delta":{"reasoning_content":"think","content":"answer"}}]}"#;
        p.fill_buffer(data).expect("parse ok");

        let e1 = p.event_buffer.remove(0);
        assert!(
            matches!(e1, StreamEvent::ThinkingDelta(ref t) if t == "think"),
            "first event must be ThinkingDelta; got {e1:?}"
        );
        let e2 = p.event_buffer.remove(0);
        assert!(
            matches!(e2, StreamEvent::TextDelta(ref t) if t == "answer"),
            "second event must be TextDelta; got {e2:?}"
        );
        assert!(p.event_buffer.is_empty());
    }

    #[test]
    fn parser_usage_only_frame_without_choices_emits_usage() {
        // Finding 5: standalone usage frame (no choices) must produce a Usage event.
        let mut p = OpenAiCompatSseParser::new();
        let data = r#"{"usage":{"prompt_tokens":100,"completion_tokens":42}}"#;
        p.fill_buffer(data).expect("parse ok");

        let e1 = p.event_buffer.remove(0);
        assert!(
            matches!(
                e1,
                StreamEvent::Usage {
                    input_tokens: 100,
                    output_tokens: 42,
                    ..
                }
            ),
            "must emit Usage(100,42) for standalone usage frame; got {e1:?}"
        );
        assert!(p.event_buffer.is_empty());
    }

    #[test]
    fn parser_usage_frame_with_empty_choices_emits_usage() {
        // Variant: vLLM sends {"choices": [], "usage": {...}}.
        let mut p = OpenAiCompatSseParser::new();
        let data = r#"{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}}"#;
        p.fill_buffer(data).expect("parse ok");

        let e1 = p.event_buffer.remove(0);
        assert!(
            matches!(
                e1,
                StreamEvent::Usage {
                    input_tokens: 5,
                    output_tokens: 3,
                    ..
                }
            ),
            "must emit Usage for empty choices frame; got {e1:?}"
        );
    }

    // --- construct validation tests ---

    #[test]
    fn new_rejects_base_url_with_query_string() {
        let result = OpenAiCompatBackend::new(OpenAiCompatConfig {
            base_url: "https://example.com/v1?token=secret".to_string(),
            api_key: None,
            model: String::new(),
            max_tokens: None,
            reasoning: ReasoningStyle::None,
        });
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("query string"),
            "error should mention query string; got: {msg}"
        );
    }

    // --- build_request_body: thinking/null regression ---

    #[test]
    fn build_request_body_assistant_thinking_blocks_not_serialised_as_null() {
        // Finding 2: Thinking blocks must be skipped, not emitted as null.
        let b = make_backend(ReasoningStyle::None);
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text("hi".to_string())],
            },
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        text: "internal".to_string(),
                        signature: "sig".to_string(),
                    },
                    ContentBlock::Text("answer".to_string()),
                ],
            },
        ];
        let body = b.build_request_body(&messages, &simple_config());
        let assistant_content = body["messages"][1]["content"]
            .as_array()
            .expect("content array");
        for item in assistant_content {
            assert!(
                !item.is_null(),
                "content array must not contain null; got {assistant_content:?}"
            );
        }
        assert_eq!(
            assistant_content.len(),
            1,
            "only the Text block should survive"
        );
        assert_eq!(assistant_content[0]["text"], "answer");
    }
}
