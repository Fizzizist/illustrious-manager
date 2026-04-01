use anyhow::Result;
use futures::StreamExt;

use crate::backend::LlmBackend;
use crate::types::*;

pub struct Agent {
    backend: Box<dyn LlmBackend>,
    history: Vec<Message>,
    config: RequestConfig,
}

impl Agent {
    pub fn new(backend: Box<dyn LlmBackend>, config: RequestConfig) -> Self {
        Self {
            backend,
            history: Vec::new(),
            config,
        }
    }

    pub fn history(&self) -> &[Message] {
        &self.history
    }

    pub async fn send(&mut self, input: String) -> Result<BoxStream<AgentEvent>> {
        self.history.push(Message {
            role: Role::User,
            content: input,
        });

        let mut backend_stream = self
            .backend
            .send_message(&self.history, &self.config)
            .await?;

        let mut events = Vec::new();
        let mut accumulated = String::new();

        while let Some(result) = backend_stream.next().await {
            match result {
                Ok(StreamEvent::TextDelta(text)) => {
                    accumulated.push_str(&text);
                    events.push(AgentEvent::TokenReceived(text));
                }
                Ok(StreamEvent::Done) => {
                    self.history.push(Message {
                        role: Role::Assistant,
                        content: accumulated.clone(),
                    });
                    events.push(AgentEvent::ResponseComplete(accumulated));
                    break;
                }
                Err(e) => {
                    events.push(AgentEvent::Error(e.to_string()));
                    break;
                }
            }
        }

        Ok(Box::pin(futures::stream::iter(events)))
    }
}
