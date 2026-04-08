use illustrious_manager::session::{generate_session_id, validate_session_id};
use illustrious_manager::types::{ContentBlock, Message, Role};

#[test]
fn generated_session_ids_are_unique() {
    let ids: Vec<String> = (0..100).map(|_| generate_session_id()).collect();
    let unique_count = ids.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(
        unique_count, 100,
        "all generated session IDs should be unique"
    );
}

#[test]
fn session_id_validation_rejects_various_invalid_inputs() {
    let cases = vec![
        "",
        "not-a-uuid",
        "01944ab8-7a67-7000-9219",
        "01944ab87a6770009219566f82fff672extra",
        "01944ab8-7a67-7000-9219-566f82fff67",
        "01944ab8-7a67-7000-9219-566f82fff6721",
    ];
    for case in cases {
        assert!(
            validate_session_id(case).is_err(),
            "expected '{}' to be rejected",
            case
        );
    }
}

#[test]
fn session_id_validation_accepts_standard_uuidv7_format() {
    let id = "01944ab8-7a67-7000-9219-566f82fff672";
    assert!(validate_session_id(id).is_ok());
}

#[test]
fn session_id_validation_accepts_no_hyphen_format() {
    let id = "01944ab87a6770009219566f82fff672";
    assert!(validate_session_id(id).is_ok());
}

#[test]
fn generated_session_id_passes_validation() {
    for _ in 0..50 {
        let id = generate_session_id();
        assert!(
            validate_session_id(&id).is_ok(),
            "generated ID {id} should be valid"
        );
    }
}

#[test]
fn message_content_serialization_roundtrip_preserves_tool_use() {
    let msg = Message {
        role: Role::Assistant,
        content: vec![
            ContentBlock::Text("Running command".to_string()),
            ContentBlock::ToolUse {
                id: "t1".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"command": "ls -la"}),
            },
        ],
    };

    let json = serde_json::to_string(&msg.content).expect("serialize");
    let deserialized: Vec<ContentBlock> = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized, msg.content);
}
