#[test]
fn test_parse_sse_text_delta() {
    let mut parser = illustrious_manager::backend::zai::ZaiSseParser::new(0);
    let data = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
    let event = parser.parse(data).unwrap();
    match event {
        Some(illustrious_manager::types::StreamEvent::TextDelta(text)) => {
            assert_eq!(text, "Hello");
        }
        other => panic!("Expected TextDelta, got {:?}", other),
    }
}

#[test]
fn test_parse_sse_done_event() {
    let mut parser = illustrious_manager::backend::zai::ZaiSseParser::new(0);
    let data = "[DONE]";
    let event = parser.parse(data).unwrap();
    match event {
        Some(illustrious_manager::types::StreamEvent::Done) => {}
        other => panic!("Expected Done, got {:?}", other),
    }
}

#[test]
fn test_parse_sse_empty_delta_returns_none() {
    let mut parser = illustrious_manager::backend::zai::ZaiSseParser::new(0);
    let data = r#"{"choices":[{"delta":{}}]}"#;
    let event = parser.parse(data).unwrap();
    assert!(
        event.is_none(),
        "choices with no delta content should be ignored"
    );
}

#[test]
fn test_parse_sse_empty_content_returns_none() {
    let mut parser = illustrious_manager::backend::zai::ZaiSseParser::new(0);
    let data = r#"{"choices":[{"delta":{"content":""}}]}"#;
    let event = parser.parse(data).unwrap();
    assert!(event.is_none(), "empty content string should be ignored");
}

#[test]
fn test_parse_sse_malformed_json_returns_err() {
    let mut parser = illustrious_manager::backend::zai::ZaiSseParser::new(0);
    let data = "this is not json";
    let result = parser.parse(data);
    assert!(result.is_err(), "malformed JSON should return Err");
}

#[test]
fn test_parse_sse_tool_use_start() {
    let mut parser = illustrious_manager::backend::zai::ZaiSseParser::new(0);
    let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"bash","arguments":""}}]}}]}"#;
    let event = parser.parse(data).unwrap();
    match event {
        Some(illustrious_manager::types::StreamEvent::ToolUseStart { id, name }) => {
            assert_eq!(name, "bash");
            assert_eq!(id, "tool_0");
        }
        other => panic!("Expected ToolUseStart, got {:?}", other),
    }
}

#[test]
fn test_parse_sse_tool_use_delta() {
    let mut parser = illustrious_manager::backend::zai::ZaiSseParser::new(0);
    let data =
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ls"}}]}}]}"#;
    let event = parser.parse(data).unwrap();
    match event {
        Some(illustrious_manager::types::StreamEvent::ToolUseDelta(delta)) => {
            assert_eq!(delta, "ls");
        }
        other => panic!("Expected ToolUseDelta, got {:?}", other),
    }
}

#[test]
fn test_parse_sse_tool_use_done() {
    let mut parser = illustrious_manager::backend::zai::ZaiSseParser::new(0);
    let data = r#"{"choices":[{"finish_reason":"tool_calls"}]}"#;
    let event = parser.parse(data).unwrap();
    match event {
        Some(illustrious_manager::types::StreamEvent::ToolUseDone) => {}
        other => panic!("Expected ToolUseDone, got {:?}", other),
    }
}

#[test]
fn test_parse_sse_multiple_tool_calls_in_single_chunk() {
    // Test the critical fix: multiple tool calls in a single chunk are buffered
    let mut parser = illustrious_manager::backend::zai::ZaiSseParser::new(0);
    let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"bash","arguments":""}},{"index":1,"function":{"name":"read_file","arguments":""}}]}}]}"#;

    // First call returns first tool call
    let event1 = parser.parse(data).unwrap();
    match event1 {
        Some(illustrious_manager::types::StreamEvent::ToolUseStart { name, .. }) => {
            assert_eq!(name, "bash");
        }
        other => panic!("Expected ToolUseStart for bash, got {:?}", other),
    }

    // Second call returns second tool call (from buffer)
    let event2 = parser.parse("").unwrap();
    match event2 {
        Some(illustrious_manager::types::StreamEvent::ToolUseStart { name, .. }) => {
            assert_eq!(name, "read_file");
        }
        other => panic!("Expected ToolUseStart for read_file, got {:?}", other),
    }

    // Third call returns None (buffer empty)
    let event3 = parser.parse("").unwrap();
    assert!(event3.is_none(), "Expected None when buffer is empty");
}

#[test]
fn test_zai_stealth_max_tokens_emits_error() {
    let mut parser = illustrious_manager::backend::zai::ZaiSseParser::new(4096);
    let data = r#"{"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4096}}"#;
    let result = parser.parse(data);
    assert!(
        result.is_err(),
        "should emit MaxTokensExceeded for stealth max-tokens via Zai"
    );
    let err = result.unwrap_err();
    let backend_err = err
        .downcast_ref::<illustrious_manager::backend::error::BackendError>()
        .expect("should downcast to BackendError");
    assert!(
        matches!(
            backend_err,
            illustrious_manager::backend::error::BackendError::MaxTokensExceeded {
                input_tokens: 10,
                output_tokens: 4096
            }
        ),
        "Zai should inherit stealth max-tokens detection; got: {backend_err:?}"
    );
}
