use std::pin::Pin;

use futures::Stream;

/// A message in the conversation history.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// The role of a message sender.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Per-request configuration for LLM calls.
#[derive(Debug, Clone)]
pub struct RequestConfig {
    pub model: String,
}

/// Events emitted by the LLM backend during streaming.
#[derive(Debug)]
pub enum StreamEvent {
    TextDelta(String),
    Done,
}

/// Events emitted by the Agent to frontends.
#[derive(Debug)]
pub enum AgentEvent {
    TokenReceived(String),
    ResponseComplete(String),
    Error(String),
}

/// A pinned, boxed stream type alias for convenience.
pub type BoxStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;
