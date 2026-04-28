// Core types for illustrious-manager

use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

/// Definition of a tool for discovery/registration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Role in a conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// A content block within a message
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentBlock {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
}

impl Serialize for ContentBlock {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ContentBlock::Text(s) => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", s)?;
                map.end()
            }
            ContentBlock::ToolUse { id, name, input } => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "tool_use")?;
                map.serialize_entry("id", id)?;
                map.serialize_entry("name", name)?;
                map.serialize_entry("input", input)?;
                map.end()
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "tool_result")?;
                map.serialize_entry("tool_use_id", tool_use_id)?;
                map.serialize_entry("content", content)?;
                map.serialize_entry("is_error", is_error)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ContentBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct ContentBlockVisitor;

        impl<'de> Visitor<'de> for ContentBlockVisitor {
            type Value = ContentBlock;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string (text block) or an object (tool_use or tool_result)")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(ContentBlock::Text(value.to_string()))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(ContentBlock::Text(value))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut type_field: Option<String> = None;
                let mut id = None;
                let mut name = None;
                let mut input = None;
                let mut tool_use_id = None;
                let mut content = None;
                let mut is_error = None;
                let mut text = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "type" => {
                            type_field = Some(map.next_value()?);
                        }
                        "text" => {
                            text = Some(map.next_value()?);
                        }
                        "id" => {
                            id = Some(map.next_value()?);
                        }
                        "name" => {
                            name = Some(map.next_value()?);
                        }
                        "input" => {
                            input = Some(map.next_value()?);
                        }
                        "tool_use_id" => {
                            tool_use_id = Some(map.next_value()?);
                        }
                        "content" => {
                            content = Some(map.next_value()?);
                        }
                        "is_error" => {
                            is_error = Some(map.next_value()?);
                        }
                        _ => {
                            map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                match type_field.as_deref() {
                    Some("text") => Ok(ContentBlock::Text(
                        text.ok_or_else(|| de::Error::missing_field("text"))?,
                    )),
                    Some("tool_use") => Ok(ContentBlock::ToolUse {
                        id: id.ok_or_else(|| de::Error::missing_field("id"))?,
                        name: name.ok_or_else(|| de::Error::missing_field("name"))?,
                        input: input.ok_or_else(|| de::Error::missing_field("input"))?,
                    }),
                    Some("tool_result") => Ok(ContentBlock::ToolResult {
                        tool_use_id: tool_use_id
                            .ok_or_else(|| de::Error::missing_field("tool_use_id"))?,
                        content: content.ok_or_else(|| de::Error::missing_field("content"))?,
                        is_error: is_error.ok_or_else(|| de::Error::missing_field("is_error"))?,
                    }),
                    Some(other) => Err(de::Error::unknown_variant(
                        other,
                        &["text", "tool_use", "tool_result"],
                    )),
                    None => Err(de::Error::missing_field("type")),
                }
            }
        }

        deserializer.deserialize_any(ContentBlockVisitor)
    }
}

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn text(role: Role, content: String) -> Self {
        Self {
            role,
            content: vec![ContentBlock::Text(content)],
        }
    }
}

/// Per-request configuration for LLM calls
#[derive(Debug, Clone)]
pub struct RequestConfig {
    pub model: String,
    pub max_tokens: u32,
    pub tools: Vec<ToolDefinition>,
}

/// Events emitted by the LLM backend during streaming
#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ToolUseStart {
        id: String,
        name: String,
    },
    ToolUseDelta(String),
    ToolUseDone,
    /// Token usage and stop reason reported by the backend at the end of a response.
    Usage {
        input_tokens: u32,
        output_tokens: u32,
        stop_reason: String,
    },
    Done,
}

/// Events emitted by the Agent to frontends
#[derive(Debug, Clone)]
pub enum AgentEvent {
    TokenReceived(String),
    ToolUseReceived {
        id: String,
        name: String,
        input: serde_json::Value,
        /// 1-based index within the current assistant turn; resets each turn.
        index: usize,
    },
    ToolResult {
        name: String,
        content: String,
        is_error: bool,
        /// 1-based index matching the corresponding `ToolUseReceived`.
        index: usize,
    },
    ToolConfirmationRequired {
        id: String,
        name: String,
        input: serde_json::Value,
        /// 1-based index within the current assistant turn.
        index: usize,
    },
    ResponseComplete(String),
    Error(String),
    Usage {
        input_tokens: u32,
        output_tokens: u32,
        stop_reason: String,
    },
    SubAgentUsage {
        input_tokens: u32,
        output_tokens: u32,
        role: String,
    },
    Interrupted {
        partial_text: String,
    },
}

/// A pinned, boxed stream type alias for convenience
pub type BoxStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;

/// Response from frontend to agent when confirmation is required
#[derive(Debug, Clone, PartialEq)]
pub enum ConfirmationResponse {
    Approved,
    Rejected,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_serializes_as_lowercase() {
        let user_role = Role::User;
        let assistant_role = Role::Assistant;

        let user_json =
            serde_json::to_string(&user_role).expect("Role::User should serialize to JSON");
        let assistant_json = serde_json::to_string(&assistant_role)
            .expect("Role::Assistant should serialize to JSON");

        assert_eq!(user_json, r#""user""#);
        assert_eq!(assistant_json, r#""assistant""#);
    }

    #[test]
    fn role_deserializes_from_lowercase() {
        let user_json = r#""user""#;
        let assistant_json = r#""assistant""#;

        let user_role: Role =
            serde_json::from_str(user_json).expect("'user' should deserialize to Role::User");
        let assistant_role: Role = serde_json::from_str(assistant_json)
            .expect("'assistant' should deserialize to Role::Assistant");

        assert_eq!(user_role, Role::User);
        assert_eq!(assistant_role, Role::Assistant);
    }

    #[test]
    fn message_roundtrips_through_serde() {
        let original = Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello, world!".to_string())],
        };

        let json = serde_json::to_string(&original).expect("Message should serialize to JSON");
        let deserialized: Message =
            serde_json::from_str(&json).expect("JSON should deserialize back to Message");

        assert_eq!(deserialized, original);
        assert_eq!(
            deserialized.content,
            vec![ContentBlock::Text("Hello, world!".to_string())]
        );
        assert_eq!(deserialized.role, Role::User);
    }

    #[test]
    fn content_block_text_variant_creates_text_block() {
        let block = ContentBlock::Text("Hello".to_string());
        assert!(matches!(block, ContentBlock::Text(_)));
        if let ContentBlock::Text(text) = block {
            assert_eq!(text, "Hello");
        }
    }

    #[test]
    fn content_block_tool_use_variant_contains_id_name_and_input() {
        let input = serde_json::json!({"command": "ls"});
        let block = ContentBlock::ToolUse {
            id: "tool-123".to_string(),
            name: "bash".to_string(),
            input: input.clone(),
        };
        assert!(matches!(block, ContentBlock::ToolUse { .. }));
        if let ContentBlock::ToolUse {
            id,
            name,
            input: block_input,
        } = block
        {
            assert_eq!(id, "tool-123");
            assert_eq!(name, "bash");
            assert_eq!(block_input, input);
        }
    }

    #[test]
    fn content_block_tool_result_variant_contains_tool_use_id_content_and_error_flag() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "tool-123".to_string(),
            content: "output".to_string(),
            is_error: false,
        };
        assert!(matches!(block, ContentBlock::ToolResult { .. }));
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } = block
        {
            assert_eq!(tool_use_id, "tool-123");
            assert_eq!(content, "output");
            assert!(!is_error);
        }
    }

    #[test]
    fn content_block_text_serializes_to_object() {
        let block = ContentBlock::Text("Hello".to_string());
        let json = serde_json::to_string(&block).expect("ContentBlock should serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should parse JSON");
        assert_eq!(parsed["type"], "text");
        assert_eq!(parsed["text"], "Hello");
    }

    #[test]
    fn content_block_text_deserializes_from_object() {
        let json = r#"{"type":"text","text":"Hello"}"#;
        let block: ContentBlock =
            serde_json::from_str(json).expect("Should deserialize to ContentBlock::Text");
        assert!(matches!(block, ContentBlock::Text(_)));
        if let ContentBlock::Text(text) = block {
            assert_eq!(text, "Hello");
        }
    }

    #[test]
    fn content_block_text_produces_object_not_string_for_api_compatibility() {
        // Regression test: Vertex AI / Anthropic API requires content blocks to be
        // dictionaries, not plain strings. ContentBlock::Text must serialize as
        // {"type":"text","text":"..."} rather than a bare string.
        let block = ContentBlock::Text("hello".to_string());
        let value = serde_json::to_value(&block).expect("should serialize");
        assert!(
            value.is_object(),
            "ContentBlock::Text must serialize as an object, got: {}",
            value
        );
    }

    #[test]
    fn content_block_tool_use_serializes_to_object() {
        let block = ContentBlock::ToolUse {
            id: "tool-123".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
        };
        let json = serde_json::to_string(&block).expect("ContentBlock should serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should parse JSON");
        assert_eq!(parsed["type"], "tool_use");
        assert_eq!(parsed["id"], "tool-123");
        assert_eq!(parsed["name"], "bash");
        assert_eq!(parsed["input"], serde_json::json!({"command": "ls"}));
    }

    #[test]
    fn content_block_tool_result_serializes_to_object() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "tool-123".to_string(),
            content: "output".to_string(),
            is_error: false,
        };
        let json = serde_json::to_string(&block).expect("ContentBlock should serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should parse JSON");
        assert_eq!(parsed["type"], "tool_result");
        assert_eq!(parsed["tool_use_id"], "tool-123");
        assert_eq!(parsed["content"], "output");
        assert_eq!(parsed["is_error"], false);
    }

    #[test]
    fn message_text_convenience_constructor_creates_message_with_single_text_block() {
        let msg = Message::text(Role::User, "Hello".to_string());
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.len(), 1);
        assert!(matches!(msg.content[0], ContentBlock::Text(_)));
        if let ContentBlock::Text(text) = &msg.content[0] {
            assert_eq!(text, "Hello");
        }
    }

    #[test]
    fn message_with_multiple_content_blocks_serializes_correctly() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text("Thinking...".to_string()),
                ContentBlock::ToolUse {
                    id: "tool-1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "ls"}),
                },
            ],
        };
        let json = serde_json::to_string(&msg).expect("Message should serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should parse JSON");
        assert_eq!(parsed["role"], "assistant");
        assert_eq!(parsed["content"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["content"][0]["type"], "text");
        assert_eq!(parsed["content"][0]["text"], "Thinking...");
        assert_eq!(parsed["content"][1]["type"], "tool_use");
    }

    #[test]
    fn stream_event_tool_use_start_variant_contains_id_and_name() {
        let event = StreamEvent::ToolUseStart {
            id: "tool-123".to_string(),
            name: "bash".to_string(),
        };
        assert!(matches!(event, StreamEvent::ToolUseStart { .. }));
        if let StreamEvent::ToolUseStart { id, name } = event {
            assert_eq!(id, "tool-123");
            assert_eq!(name, "bash");
        }
    }

    #[test]
    fn stream_event_tool_use_delta_variant_contains_string_delta() {
        let event = StreamEvent::ToolUseDelta("{".to_string());
        assert!(matches!(event, StreamEvent::ToolUseDelta(_)));
        if let StreamEvent::ToolUseDelta(delta) = event {
            assert_eq!(delta, "{");
        }
    }

    #[test]
    fn stream_event_tool_use_done_variant_exists() {
        let event = StreamEvent::ToolUseDone;
        assert!(matches!(event, StreamEvent::ToolUseDone));
    }

    #[test]
    fn agent_event_tool_use_received_variant_contains_id_name_and_input() {
        let input = serde_json::json!({"command": "ls"});
        let event = AgentEvent::ToolUseReceived {
            id: "tool-123".to_string(),
            name: "bash".to_string(),
            input: input.clone(),
            index: 1,
        };
        assert!(matches!(event, AgentEvent::ToolUseReceived { .. }));
        if let AgentEvent::ToolUseReceived {
            id,
            name,
            input: event_input,
            index,
        } = event
        {
            assert_eq!(id, "tool-123");
            assert_eq!(name, "bash");
            assert_eq!(event_input, input);
            assert_eq!(index, 1);
        }
    }

    #[test]
    fn agent_event_tool_result_variant_contains_name_content_and_error_flag() {
        let event = AgentEvent::ToolResult {
            name: "bash".to_string(),
            content: "file1.txt".to_string(),
            is_error: false,
            index: 1,
        };
        assert!(matches!(event, AgentEvent::ToolResult { .. }));
        if let AgentEvent::ToolResult {
            name,
            content,
            is_error,
            index,
        } = event
        {
            assert_eq!(name, "bash");
            assert_eq!(content, "file1.txt");
            assert!(!is_error);
            assert_eq!(index, 1);
        }
    }

    #[test]
    fn agent_event_tool_confirmation_required_variant_contains_id_name_and_input() {
        let input = serde_json::json!({"command": "rm -rf /"});
        let event = AgentEvent::ToolConfirmationRequired {
            id: "tool-123".to_string(),
            name: "bash".to_string(),
            input: input.clone(),
            index: 1,
        };
        assert!(matches!(event, AgentEvent::ToolConfirmationRequired { .. }));
        if let AgentEvent::ToolConfirmationRequired {
            id,
            name,
            input: event_input,
            index,
        } = event
        {
            assert_eq!(id, "tool-123");
            assert_eq!(name, "bash");
            assert_eq!(event_input, input);
            assert_eq!(index, 1);
        }
    }
}
