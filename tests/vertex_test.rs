#[test]
fn test_parse_sse_content_block_delta() {
    let data =
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
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
    let data =
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
    let event = illustrious_manager::backend::vertex::parse_sse_data(data).unwrap();
    assert!(event.is_none(), "content_block_start should be ignored");
}

#[test]
fn test_parse_sse_ping_ignored() {
    let data = r#"{"type":"ping"}"#;
    let event = illustrious_manager::backend::vertex::parse_sse_data(data).unwrap();
    assert!(event.is_none(), "ping should be ignored");
}
