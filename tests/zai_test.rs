#[test]
fn test_parse_sse_text_delta() {
    let data = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
    let event = illustrious_manager::backend::zai::parse_sse_data(data).unwrap();
    match event {
        Some(illustrious_manager::types::StreamEvent::TextDelta(text)) => {
            assert_eq!(text, "Hello");
        }
        other => panic!("Expected TextDelta, got {:?}", other),
    }
}

#[test]
fn test_parse_sse_done_event() {
    let data = "[DONE]";
    let event = illustrious_manager::backend::zai::parse_sse_data(data).unwrap();
    match event {
        Some(illustrious_manager::types::StreamEvent::Done) => {}
        other => panic!("Expected Done, got {:?}", other),
    }
}

#[test]
fn test_parse_sse_empty_delta_returns_none() {
    let data = r#"{"choices":[{"delta":{}}]}"#;
    let event = illustrious_manager::backend::zai::parse_sse_data(data).unwrap();
    assert!(
        event.is_none(),
        "choices with no delta content should be ignored"
    );
}

#[test]
fn test_parse_sse_empty_content_returns_none() {
    let data = r#"{"choices":[{"delta":{"content":""}}]}"#;
    let event = illustrious_manager::backend::zai::parse_sse_data(data).unwrap();
    assert!(event.is_none(), "empty content string should be ignored");
}

#[test]
fn test_parse_sse_malformed_json_returns_err() {
    let data = "this is not json";
    let result = illustrious_manager::backend::zai::parse_sse_data(data);
    assert!(result.is_err(), "malformed JSON should return Err");
}
