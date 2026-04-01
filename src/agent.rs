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

impl Agent {
    pub fn new(backend: Box<dyn LlmBackend>, config: RequestConfig) -> Self {
        Self {
            backend,
            history: Arc::new(Mutex::new(Vec::new())),
            config,
        }
    }

    pub fn history(&self) -> Vec<Message> {
        self.history.lock().expect("history mutex poisoned").clone()
    }

    pub async fn send(&mut self, input: String) -> Result<BoxStream<AgentEvent>> {
        self.history
            .lock()
            .expect("history mutex poisoned")
            .push(Message {
                role: Role::User,
                content: input,
            });

        let history_snapshot = self.history.lock().expect("history mutex poisoned").clone();

        let backend_stream = match self
            .backend
            .send_message(&history_snapshot, &self.config)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                self.history.lock().expect("history mutex poisoned").pop();
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
                        history_arc
                            .lock()
                            .expect("history mutex poisoned")
                            .push(Message {
                                role: Role::Assistant,
                                content: accumulated.clone(),
                            });
                        let _ = event_tx.unbounded_send(AgentEvent::ResponseComplete(accumulated));
                        break;
                    }
                    Err(e) => {
                        history_arc.lock().expect("history mutex poisoned").pop();
                        let _ = event_tx.unbounded_send(AgentEvent::Error(e.to_string()));
                        break;
                    }
                }
            }
        });

        Ok(Box::pin(event_rx))
    }
}
