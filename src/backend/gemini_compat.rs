#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::error::BackendError;
    use crate::types::{
        ContentBlock, Role, ThinkingConfig, ThinkingDisplay, ThinkingMode, ToolDefinition,
    };
    use futures::StreamExt;
    use serde_json::{Value, json};

    const SKIP_SIGNATURE: &str = "skip_thought_signature_validator";

    fn backend_for(endpoint: &str) -> GeminiCompatBackend {
        GeminiCompatBackend::new(
            Client::new(),
            GeminiCompatConfig {
                endpoint: endpoint.to_string(),
                auth_token: None,
                session_header_name: None,
            },
        )
    }

    fn test_backend() -> GeminiCompatBackend {
        backend_for("https://test.example.com")
    }

    fn request_config() -> RequestConfig {
        RequestConfig {
            model: "gemini-test".to_string(),
            max_tokens: 4096,
            ..Default::default()
        }
    }

    fn msg(role: Role, content: Vec<ContentBlock>) -> Message {
        Message {
            role,
            content,
            created_at: 0.0,
        }
    }

    fn tool_use(id: &str, name: &str, input: Value, signature: Option<&str>) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input,
            signature: signature.map(str::to_string),
        }
    }

    fn tool_result(id: &str, content: Vec<ContentBlock>, is_error: bool) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id: id.to_string(),
            content,
            is_error,
        }
    }

    fn text(s: &str) -> ContentBlock {
        ContentBlock::Text(s.to_string())
    }

    fn build(messages: &[Message], config: &RequestConfig) -> Value {
        test_backend()
            .build_request_body(messages, config)
            .expect("request body should build")
    }

    fn contents_of(body: &Value) -> Vec<Value> {
        body["contents"]
            .as_array()
            .expect("body must contain a contents array")
            .clone()
    }

    fn thinking(
        mode: ThinkingMode,
        enabled: bool,
        display: Option<ThinkingDisplay>,
    ) -> RequestConfig {
        RequestConfig {
            thinking: Some(ThinkingConfig {
                mode,
                enabled,
                display,
            }),
            ..request_config()
        }
    }

    #[test]
    fn roles_map_to_user_and_model() {
        let body = build(
            &[
                msg(Role::User, vec![text("hello")]),
                msg(Role::Assistant, vec![text("hi")]),
            ],
            &request_config(),
        );
        assert_eq!(
            body["contents"],
            json!([
                {"role": "user", "parts": [{"text": "hello"}]},
                {"role": "model", "parts": [{"text": "hi"}]},
            ])
        );
    }

    #[test]
    fn image_block_becomes_inline_data() {
        let body = build(
            &[msg(
                Role::User,
                vec![
                    text("look"),
                    ContentBlock::Image {
                        media_type: "image/png".to_string(),
                        data: "QUJD".to_string(),
                    },
                ],
            )],
            &request_config(),
        );
        assert_eq!(
            body["contents"][0]["parts"],
            json!([
                {"text": "look"},
                {"inlineData": {"mimeType": "image/png", "data": "QUJD"}},
            ])
        );
    }

    #[test]
    fn thinking_blocks_are_dropped_and_empty_messages_vanish() {
        let body = build(
            &[
                msg(Role::User, vec![text("q")]),
                msg(
                    Role::Assistant,
                    vec![
                        ContentBlock::Thinking {
                            text: "hmm".to_string(),
                            signature: "anthropic-sig".to_string(),
                        },
                        ContentBlock::RedactedThinking {
                            data: "opaque".to_string(),
                        },
                    ],
                ),
                msg(
                    Role::Assistant,
                    vec![
                        ContentBlock::Thinking {
                            text: "more".to_string(),
                            signature: "anthropic-sig-2".to_string(),
                        },
                        text("answer"),
                    ],
                ),
            ],
            &request_config(),
        );
        assert_eq!(
            body["contents"],
            json!([
                {"role": "user", "parts": [{"text": "q"}]},
                {"role": "model", "parts": [{"text": "answer"}]},
            ])
        );
        let serialized = body.to_string();
        assert!(!serialized.contains("anthropic-sig"));
        assert!(!serialized.contains("opaque"));
    }

    #[test]
    fn consecutive_same_role_messages_merge_into_one_content() {
        let body = build(
            &[
                msg(Role::User, vec![text("one")]),
                msg(Role::User, vec![text("two")]),
                msg(Role::Assistant, vec![text("reply")]),
                msg(
                    Role::Assistant,
                    vec![ContentBlock::Thinking {
                        text: "t".to_string(),
                        signature: "s".to_string(),
                    }],
                ),
                msg(Role::Assistant, vec![text("reply2")]),
                msg(Role::User, vec![text("three")]),
            ],
            &request_config(),
        );
        assert_eq!(
            body["contents"],
            json!([
                {"role": "user", "parts": [{"text": "one"}, {"text": "two"}]},
                {"role": "model", "parts": [{"text": "reply"}, {"text": "reply2"}]},
                {"role": "user", "parts": [{"text": "three"}]},
            ])
        );
    }

    #[test]
    fn tool_use_becomes_function_call_with_sibling_signature() {
        let body = build(
            &[
                msg(Role::User, vec![text("go")]),
                msg(
                    Role::Assistant,
                    vec![
                        text("running"),
                        tool_use("call_1", "bash", json!({"command": "ls"}), Some("SIG==")),
                    ],
                ),
            ],
            &request_config(),
        );
        assert_eq!(
            body["contents"][1],
            json!({
                "role": "model",
                "parts": [
                    {"text": "running"},
                    {
                        "functionCall": {"name": "bash", "args": {"command": "ls"}, "id": "call_1"},
                        "thoughtSignature": "SIG==",
                    },
                ],
            })
        );
    }

    #[test]
    fn null_tool_input_becomes_empty_args_object() {
        let body = build(
            &[
                msg(Role::User, vec![text("go")]),
                msg(
                    Role::Assistant,
                    vec![tool_use("c1", "noargs", Value::Null, Some("S"))],
                ),
            ],
            &request_config(),
        );
        assert_eq!(
            body["contents"][1]["parts"][0]["functionCall"]["args"],
            json!({})
        );
    }

    #[test]
    fn parallel_calls_and_results_keep_call_order() {
        let body = build(
            &[
                msg(Role::User, vec![text("go")]),
                msg(
                    Role::Assistant,
                    vec![
                        tool_use("a", "bash", json!({"command": "ls"}), Some("SIG_A")),
                        tool_use("b", "search", json!({"query": "x"}), None),
                    ],
                ),
                msg(
                    Role::User,
                    vec![
                        tool_result("a", vec![text("out_a")], false),
                        tool_result("b", vec![text("out_b")], false),
                    ],
                ),
            ],
            &request_config(),
        );
        assert_eq!(
            body["contents"][1],
            json!({
                "role": "model",
                "parts": [
                    {
                        "functionCall": {"name": "bash", "args": {"command": "ls"}, "id": "a"},
                        "thoughtSignature": "SIG_A",
                    },
                    {"functionCall": {"name": "search", "args": {"query": "x"}, "id": "b"}},
                ],
            })
        );
        assert_eq!(
            body["contents"][2],
            json!({
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "bash", "id": "a", "response": {"output": "out_a"}}},
                    {"functionResponse": {"name": "search", "id": "b", "response": {"output": "out_b"}}},
                ],
            })
        );
        assert_eq!(contents_of(&body).len(), 3);
    }

    #[test]
    fn error_tool_result_uses_error_response_key() {
        let body = build(
            &[
                msg(Role::User, vec![text("go")]),
                msg(
                    Role::Assistant,
                    vec![tool_use("a", "bash", json!({}), Some("S"))],
                ),
                msg(Role::User, vec![tool_result("a", vec![text("boom")], true)]),
            ],
            &request_config(),
        );
        assert_eq!(
            body["contents"][2]["parts"][0]["functionResponse"],
            json!({"name": "bash", "id": "a", "response": {"error": "boom"}})
        );
    }

    #[test]
    fn tool_result_image_goes_into_function_response_parts() {
        let body = build(
            &[
                msg(Role::User, vec![text("go")]),
                msg(
                    Role::Assistant,
                    vec![tool_use(
                        "a",
                        "image_viewer",
                        json!({"path": "x.png"}),
                        Some("S"),
                    )],
                ),
                msg(
                    Role::User,
                    vec![tool_result(
                        "a",
                        vec![
                            ContentBlock::Image {
                                media_type: "image/png".to_string(),
                                data: "QUJD".to_string(),
                            },
                            text("x.png"),
                        ],
                        false,
                    )],
                ),
            ],
            &request_config(),
        );
        assert_eq!(
            body["contents"][2]["parts"][0]["functionResponse"],
            json!({
                "name": "image_viewer",
                "id": "a",
                "response": {"output": "x.png"},
                "parts": [{"inlineData": {"mimeType": "image/png", "data": "QUJD"}}],
            })
        );
    }

    #[test]
    fn unsigned_model_step_gets_skip_signature_on_first_call_only() {
        let body = build(
            &[
                msg(Role::User, vec![text("go")]),
                msg(
                    Role::Assistant,
                    vec![
                        tool_use("a", "bash", json!({}), None),
                        tool_use("b", "bash", json!({}), None),
                    ],
                ),
            ],
            &request_config(),
        );
        let parts = &body["contents"][1]["parts"];
        assert_eq!(parts[0]["thoughtSignature"], SKIP_SIGNATURE);
        assert!(parts[1].get("thoughtSignature").is_none());
    }

    #[test]
    fn signed_model_step_is_left_unchanged() {
        let body = build(
            &[
                msg(Role::User, vec![text("go")]),
                msg(
                    Role::Assistant,
                    vec![
                        tool_use("a", "bash", json!({}), None),
                        tool_use("b", "bash", json!({}), Some("REAL")),
                    ],
                ),
            ],
            &request_config(),
        );
        let parts = &body["contents"][1]["parts"];
        assert!(parts[0].get("thoughtSignature").is_none());
        assert_eq!(parts[1]["thoughtSignature"], "REAL");
        assert!(!body.to_string().contains(SKIP_SIGNATURE));
    }

    #[test]
    fn skip_signature_is_decided_per_model_step() {
        let body = build(
            &[
                msg(Role::User, vec![text("go")]),
                msg(
                    Role::Assistant,
                    vec![tool_use("a", "bash", json!({}), Some("REAL"))],
                ),
                msg(Role::User, vec![tool_result("a", vec![text("ok")], false)]),
                msg(
                    Role::Assistant,
                    vec![tool_use("b", "bash", json!({}), None)],
                ),
            ],
            &request_config(),
        );
        assert_eq!(body["contents"][1]["parts"][0]["thoughtSignature"], "REAL");
        assert_eq!(
            body["contents"][3]["parts"][0]["thoughtSignature"],
            SKIP_SIGNATURE
        );
    }

    #[test]
    fn tools_become_function_declarations_with_unchanged_schema() {
        let schema = json!({
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        });
        let config = RequestConfig {
            tools: vec![
                ToolDefinition {
                    name: "bash".to_string(),
                    description: "Run a command".to_string(),
                    input_schema: schema.clone(),
                },
                ToolDefinition {
                    name: "search".to_string(),
                    description: "Search".to_string(),
                    input_schema: json!({"type": "object"}),
                },
            ],
            ..request_config()
        };
        let body = build(&[], &config);
        assert_eq!(
            body["tools"],
            json!([{"functionDeclarations": [
                {"name": "bash", "description": "Run a command", "parametersJsonSchema": schema},
                {"name": "search", "description": "Search", "parametersJsonSchema": {"type": "object"}},
            ]}])
        );
    }

    #[test]
    fn tools_key_is_omitted_when_no_tools() {
        let body = build(&[], &request_config());
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 4096);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn max_tokens_maps_to_max_output_tokens() {
        let config = RequestConfig {
            max_tokens: 32768,
            ..request_config()
        };
        let body = build(&[], &config);
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 32768);
    }

    #[test]
    fn thinking_config_only_ever_sets_include_thoughts() {
        let cases = [
            (
                "adaptive",
                thinking(ThinkingMode::Adaptive, true, None),
                true,
            ),
            (
                "budget",
                thinking(ThinkingMode::Budget { tokens: 2048 }, true, None),
                true,
            ),
            (
                "budget summarized",
                thinking(
                    ThinkingMode::Budget { tokens: 2048 },
                    true,
                    Some(ThinkingDisplay::Summarized),
                ),
                true,
            ),
            (
                "adaptive summarized",
                thinking(
                    ThinkingMode::Adaptive,
                    true,
                    Some(ThinkingDisplay::Summarized),
                ),
                true,
            ),
            (
                "disabled",
                thinking(ThinkingMode::Budget { tokens: 2048 }, false, None),
                false,
            ),
            (
                "omitted",
                thinking(ThinkingMode::Adaptive, true, Some(ThinkingDisplay::Omitted)),
                false,
            ),
        ];
        for (label, config, expected) in cases {
            let body = build(&[], &config);
            let thinking_config = body["generationConfig"]["thinkingConfig"]
                .as_object()
                .unwrap_or_else(|| panic!("{label}: thinkingConfig must be an object"));
            assert_eq!(
                thinking_config.get("includeThoughts"),
                Some(&json!(expected)),
                "{label}"
            );
            assert_eq!(thinking_config.len(), 1, "{label}: only includeThoughts");
        }
    }

    #[test]
    fn no_thinking_config_emits_no_budget_or_level() {
        let body = build(&[], &request_config());
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 4096);
        let thinking_config = &body["generationConfig"]["thinkingConfig"];
        assert!(thinking_config.get("thinkingBudget").is_none());
        assert!(thinking_config.get("thinkingLevel").is_none());
        if !thinking_config.is_null() {
            assert_eq!(thinking_config["includeThoughts"], false);
        }
    }

    fn parse_chunks(chunks: &[Value]) -> Vec<StreamEvent> {
        let mut parser = GeminiSseParser::new();
        chunks
            .iter()
            .flat_map(|chunk| {
                parser
                    .parse(&chunk.to_string())
                    .expect("chunk should parse without error")
            })
            .collect()
    }

    fn parse_error(chunk: &Value) -> BackendError {
        let err = GeminiSseParser::new()
            .parse(&chunk.to_string())
            .expect_err("chunk should produce an error");
        err.downcast_ref::<BackendError>()
            .expect("error should wrap BackendError")
            .clone()
    }

    fn chunk_with_finish(reason: &str, message: Option<&str>) -> Value {
        let mut candidate = json!({
            "content": {"role": "model", "parts": [{"text": ""}]},
            "finishReason": reason,
        });
        if let Some(m) = message {
            candidate["finishMessage"] = json!(m);
        }
        json!({
            "candidates": [candidate],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 7,
                "totalTokenCount": 22,
            },
        })
    }

    fn function_call_chunk(call: Value, signature: Option<&str>) -> Value {
        let mut part = json!({"functionCall": call});
        if let Some(s) = signature {
            part["thoughtSignature"] = json!(s);
        }
        json!({
            "candidates": [{"content": {"role": "model", "parts": [part]}}],
            "usageMetadata": {"trafficType": "ON_DEMAND"},
        })
    }

    fn started_ids(events: &[StreamEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolUseStart { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn text_part_becomes_text_delta_and_thought_part_becomes_thinking_delta() {
        let events = parse_chunks(&[
            json!({"candidates": [{"content": {"role": "model", "parts": [
                {"text": "pondering", "thought": true}
            ]}}], "usageMetadata": {"trafficType": "ON_DEMAND"}}),
            json!({"candidates": [{"content": {"role": "model", "parts": [
                {"text": "Hi"}
            ]}}], "usageMetadata": {"trafficType": "ON_DEMAND"}}),
        ]);
        assert!(matches!(
            events.as_slice(),
            [StreamEvent::ThinkingDelta(t), StreamEvent::TextDelta(x)] if t == "pondering" && x == "Hi"
        ));
    }

    #[test]
    fn signed_function_call_emits_start_signature_delta_done_in_order() {
        let events = parse_chunks(&[function_call_chunk(
            json!({"name": "bash", "args": {"command": "ls"}, "id": "call_abc"}),
            Some("SIG"),
        )]);
        match events.as_slice() {
            [
                StreamEvent::ToolUseStart { id, name },
                StreamEvent::ToolUseSignature(sig),
                StreamEvent::ToolUseDelta(args),
                StreamEvent::ToolUseDone,
            ] => {
                assert_eq!(id, "call_abc");
                assert_eq!(name, "bash");
                assert_eq!(sig, "SIG");
                let parsed: Value = serde_json::from_str(args).expect("args should be JSON");
                assert_eq!(parsed, json!({"command": "ls"}));
            }
            other => panic!("unexpected events: {other:?}"),
        }
    }

    #[test]
    fn missing_call_id_gets_unique_non_index_id() {
        let call = json!({"name": "bash", "args": {}});
        let chunks = [
            function_call_chunk(call.clone(), Some("S")),
            function_call_chunk(call.clone(), None),
        ];
        let first = started_ids(&parse_chunks(&chunks));
        let second = started_ids(&parse_chunks(&chunks));
        assert_eq!(first.len(), 2);
        assert_eq!(second.len(), 2);

        let all: Vec<&String> = first.iter().chain(second.iter()).collect();
        for id in &all {
            assert!(!id.is_empty());
            let is_index_form = id
                .strip_prefix("tool_")
                .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
            assert!(!is_index_form, "id must not collide as tool_<n>: {id}");
        }
        let unique: std::collections::HashSet<&&String> = all.iter().collect();
        assert_eq!(
            unique.len(),
            4,
            "ids must be unique across calls and parsers"
        );
    }

    #[test]
    fn parallel_calls_in_separate_chunks_emit_one_signature() {
        let events = parse_chunks(&[
            function_call_chunk(
                json!({"name": "bash", "args": {"command": "ls"}, "id": "a"}),
                Some("SIG"),
            ),
            function_call_chunk(
                json!({"name": "search", "args": {"query": "x"}, "id": "b"}),
                None,
            ),
        ]);
        let signatures = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolUseSignature(_)))
            .count();
        let dones = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolUseDone))
            .count();
        assert_eq!(started_ids(&events), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(signatures, 1);
        assert_eq!(dones, 2);
    }

    #[test]
    fn stop_chunk_emits_usage_with_thoughts_then_done() {
        let events = parse_chunks(&[
            json!({"candidates": [{"content": {"role": "model", "parts": [
                {"text": "Hi"}
            ]}}], "usageMetadata": {"trafficType": "ON_DEMAND"}}),
            chunk_with_finish("STOP", None),
        ]);
        match events.as_slice() {
            [
                StreamEvent::TextDelta(t),
                StreamEvent::Usage {
                    input_tokens,
                    output_tokens,
                    ..
                },
                StreamEvent::Done,
            ] => {
                assert_eq!(t, "Hi");
                assert_eq!(*input_tokens, 10);
                assert_eq!(*output_tokens, 12);
            }
            other => panic!("unexpected events: {other:?}"),
        }
    }

    #[test]
    fn empty_signed_final_part_emits_no_text_and_no_error() {
        let events = parse_chunks(&[json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "", "thoughtSignature": "SIG2"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 7,
                "totalTokenCount": 22,
                "trafficType": "ON_DEMAND",
            },
        })]);
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StreamEvent::TextDelta(_)))
        );
        assert!(matches!(events.last(), Some(StreamEvent::Done)));
    }

    #[test]
    fn max_tokens_finish_reports_typed_error_with_token_counts() {
        match parse_error(&chunk_with_finish("MAX_TOKENS", None)) {
            BackendError::MaxTokensExceeded {
                input_tokens,
                output_tokens,
            } => {
                assert_eq!(input_tokens, 10);
                assert_eq!(output_tokens, 12);
            }
            other => panic!("expected MaxTokensExceeded, got {other:?}"),
        }
    }

    #[test]
    fn blocking_finish_reasons_map_to_refusal() {
        for reason in [
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "MODEL_ARMOR",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_RECITATION",
        ] {
            let err = parse_error(&chunk_with_finish(reason, Some("blocked")));
            assert!(
                err.is_refusal(),
                "{reason} should map to Refusal, got {err:?}"
            );
        }
    }

    #[test]
    fn prompt_feedback_block_without_candidates_is_refusal() {
        let err = parse_error(&json!({
            "promptFeedback": {"blockReason": "SAFETY"},
            "usageMetadata": {"promptTokenCount": 10, "totalTokenCount": 10},
        }));
        assert!(err.is_refusal(), "got {err:?}");
    }

    #[test]
    fn failure_finish_reasons_map_to_other_with_finish_message() {
        for reason in [
            "MALFORMED_FUNCTION_CALL",
            "UNEXPECTED_TOOL_CALL",
            "TOO_MANY_TOOL_CALLS",
            "MALFORMED_RESPONSE",
            "OTHER",
            "SOMETHING_NEW_FROM_GOOGLE",
        ] {
            let message = format!("detail for {reason}");
            match parse_error(&chunk_with_finish(reason, Some(&message))) {
                BackendError::Other(m) => {
                    assert!(m.contains(&message), "{reason}: message was {m:?}")
                }
                other => panic!("{reason}: expected Other, got {other:?}"),
            }
        }
    }

    async fn serve_once(response: String) -> u16 {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let port = listener.local_addr().expect("local addr").port();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept connection");
            let mut received = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = socket.read(&mut buf).await.expect("read request");
                if n == 0 {
                    break;
                }
                received.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&received).to_string();
                if let Some(head_end) = text.find("\r\n\r\n") {
                    let content_length = text[..head_end]
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|v| v.trim().parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    if received.len() >= head_end + 4 + content_length {
                        break;
                    }
                }
            }
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
            let _ = socket.shutdown().await;
        });
        port
    }

    fn endpoint(port: u16) -> String {
        format!("http://127.0.0.1:{port}/v1/models/gemini-test:streamGenerateContent?alt=sse")
    }

    async fn send_error(backend: &GeminiCompatBackend) -> BackendError {
        let result = backend
            .send_message(&[msg(Role::User, vec![text("hi")])], &request_config())
            .await;
        let err = result.err().expect("request should fail");
        err.downcast_ref::<BackendError>()
            .expect("error should wrap BackendError")
            .clone()
    }

    #[tokio::test]
    async fn http_400_maps_to_http_status_with_body() {
        let body = r#"{"error":{"message":"bad things"}}"#;
        let port = serve_once(format!(
            "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        ))
        .await;
        match send_error(&backend_for(&endpoint(port))).await {
            BackendError::HttpStatus { code, body } => {
                assert_eq!(code, 400);
                assert!(body.contains("bad things"), "body was {body:?}");
            }
            other => panic!("expected HttpStatus, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn connection_refused_maps_to_transport() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);
        let err = send_error(&backend_for(&endpoint(port))).await;
        assert!(
            matches!(err, BackendError::Transport { .. }),
            "expected Transport, got {err:?}"
        );
    }

    #[tokio::test]
    async fn streams_crlf_framed_sse_into_events() {
        let first = json!({"candidates": [{"content": {"role": "model", "parts": [
            {"text": "Hi"}
        ]}}], "usageMetadata": {"trafficType": "ON_DEMAND"}});
        let last = chunk_with_finish("STOP", None);
        let port = serve_once(format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: {first}\r\n\r\ndata: {last}\r\n\r\n"
        ))
        .await;
        let backend = backend_for(&endpoint(port));
        let stream = backend
            .send_message(&[msg(Role::User, vec![text("hi")])], &request_config())
            .await
            .expect("stream should start");
        let events: Vec<StreamEvent> = stream
            .map(|e| e.expect("event should not be an error"))
            .collect()
            .await;
        match events.as_slice() {
            [
                StreamEvent::TextDelta(t),
                StreamEvent::Usage {
                    input_tokens,
                    output_tokens,
                    ..
                },
                StreamEvent::Done,
            ] => {
                assert_eq!(t, "Hi");
                assert_eq!(*input_tokens, 10);
                assert_eq!(*output_tokens, 12);
            }
            other => panic!("unexpected events: {other:?}"),
        }
    }
}
