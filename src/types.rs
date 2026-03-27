// Core types for illustrious-manager

use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

/// Role in a conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Per-request configuration for LLM calls
#[derive(Debug, Clone)]
pub struct RequestConfig {
    pub model: String,
}

/// Events emitted by the LLM backend during streaming
#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    Done,
}

/// Events emitted by the Agent to frontends
#[derive(Debug, Clone)]
pub enum AgentEvent {
    TokenReceived(String),
    ResponseComplete(String),
    Error(String),
}

/// A pinned, boxed stream type alias for convenience
pub type BoxStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_serializes_as_lowercase() {
        let user_role = Role::User;
        let assistant_role = Role::Assistant;

        let user_json = serde_json::to_string(&user_role).unwrap();
        let assistant_json = serde_json::to_string(&assistant_role).unwrap();

        assert_eq!(user_json, r#""user""#);
        assert_eq!(assistant_json, r#""assistant""#);
    }

    #[test]
    fn role_deserializes_from_lowercase() {
        let user_json = r#""user""#;
        let assistant_json = r#""assistant""#;

        let user_role: Role = serde_json::from_str(user_json).unwrap();
        let assistant_role: Role = serde_json::from_str(assistant_json).unwrap();

        assert!(matches!(user_role, Role::User));
        assert!(matches!(assistant_role, Role::Assistant));
    }

    #[test]
    fn message_can_be_created() {
        let msg = Message {
            role: Role::User,
            content: "Hello".to_string(),
        };

        assert_eq!(msg.content, "Hello");
        assert!(matches!(msg.role, Role::User));
    }

    #[test]
    fn stream_event_has_text_delta_variant() {
        let event = StreamEvent::TextDelta("test".to_string());

        match event {
            StreamEvent::TextDelta(text) => assert_eq!(text, "test"),
            _ => panic!("Expected TextDelta variant"),
        }
    }

    #[test]
    fn stream_event_has_done_variant() {
        let event = StreamEvent::Done;

        assert!(matches!(event, StreamEvent::Done));
    }

    #[test]
    fn agent_event_has_token_received_variant() {
        let event = AgentEvent::TokenReceived("test".to_string());

        match event {
            AgentEvent::TokenReceived(text) => assert_eq!(text, "test"),
            _ => panic!("Expected TokenReceived variant"),
        }
    }

    #[test]
    fn agent_event_has_response_complete_variant() {
        let event = AgentEvent::ResponseComplete("final text".to_string());

        match event {
            AgentEvent::ResponseComplete(text) => assert_eq!(text, "final text"),
            _ => panic!("Expected ResponseComplete variant"),
        }
    }

    #[test]
    fn agent_event_has_error_variant() {
        let event = AgentEvent::Error("test error".to_string());

        match event {
            AgentEvent::Error(msg) => assert_eq!(msg, "test error"),
            _ => panic!("Expected Error variant"),
        }
    }

    #[test]
    fn request_config_can_be_created() {
        let config = RequestConfig {
            model: "claude-sonnet-4-20250514".to_string(),
        };

        assert_eq!(config.model, "claude-sonnet-4-20250514");
    }
}
