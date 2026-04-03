use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures::StreamExt;
use futures::channel::mpsc;

use crate::backend::LlmBackend;
use crate::types::{AgentEvent, BoxStream, Message, RequestConfig, Role, StreamEvent};

pub struct Agent {
    backend: Box<dyn LlmBackend>,
    history: Arc<Mutex<Vec<Message>>>,
    config: RequestConfig,
}

fn lock(m: &Mutex<Vec<Message>>) -> std::sync::MutexGuard<'_, Vec<Message>> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Agent {
    pub fn new(backend: Box<dyn LlmBackend>, config: RequestConfig) -> Self {
        Self {
            backend,
            history: Arc::new(Mutex::new(Vec::new())),
            config,
        }
    }

    pub fn history(&self) -> Vec<Message> {
        lock(&self.history).clone()
    }

    pub async fn send(&self, input: String) -> Result<BoxStream<AgentEvent>> {
        lock(&self.history).push(Message {
            role: Role::User,
            content: input,
        });

        let history_snapshot = lock(&self.history).clone();

        let backend_stream = match self
            .backend
            .send_message(&history_snapshot, &self.config)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                lock(&self.history).pop();
                return Err(e);
            }
        };

        let (event_tx, event_rx) = mpsc::unbounded::<AgentEvent>();
        let history_arc = Arc::clone(&self.history);

        tokio::spawn(async move {
            let mut accumulated = String::new();
            let mut backend_stream = backend_stream;

            while let Some(result) = backend_stream.next().await {
                match result {
                    Ok(StreamEvent::TextDelta(text)) => {
                        accumulated.push_str(&text);
                        let _ = event_tx.unbounded_send(AgentEvent::TokenReceived(text));
                    }
                    Ok(StreamEvent::Done) => {
                        lock(&history_arc).push(Message {
                            role: Role::Assistant,
                            content: accumulated.clone(),
                        });
                        let _ = event_tx.unbounded_send(AgentEvent::ResponseComplete(accumulated));
                        break;
                    }
                    Err(e) => {
                        lock(&history_arc).pop();
                        let _ = event_tx.unbounded_send(AgentEvent::Error(e.to_string()));
                        break;
                    }
                }
            }
        });

        Ok(Box::pin(event_rx))
    }
}
