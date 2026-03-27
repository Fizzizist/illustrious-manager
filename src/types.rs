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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
            content: "Hello, world!".to_string(),
        };

        let json = serde_json::to_string(&original).expect("Message should serialize to JSON");
        let deserialized: Message =
            serde_json::from_str(&json).expect("JSON should deserialize back to Message");

        assert_eq!(deserialized, original);
        assert_eq!(deserialized.content, "Hello, world!");
        assert_eq!(deserialized.role, Role::User);
    }
}
