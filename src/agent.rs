use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures::StreamExt;
use futures::channel::mpsc;

use crate::backend::LlmBackend;
use crate::config::{ConfirmationMode, ToolsConfig};
use crate::tools::ToolRegistry;
use crate::types::{
    AgentEvent, BoxStream, ConfirmationResponse, ContentBlock, Message, RequestConfig, Role,
    StreamEvent,
};

pub struct Agent {
    backend: Arc<dyn LlmBackend>,
    history: Arc<Mutex<Vec<Message>>>,
    config: RequestConfig,
    tools: Arc<ToolRegistry>,
    max_tool_iterations: u32,
    confirmation_mode: ConfirmationMode,
    confirmation_rx: Option<Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<ConfirmationResponse>>>>,
}

fn lock(m: &Mutex<Vec<Message>>) -> std::sync::MutexGuard<'_, Vec<Message>> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Agent {
    pub fn new(backend: Box<dyn LlmBackend>, config: RequestConfig) -> Self {
        Self {
            backend: Arc::from(backend),
            history: Arc::new(Mutex::new(Vec::new())),
            config,
            tools: Arc::new(ToolRegistry::new()),
            max_tool_iterations: 25,
            confirmation_mode: ConfirmationMode::WriteOnly,
            confirmation_rx: None,
        }
    }

    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Arc::new(tools);
        self
    }

    pub fn with_tool_config(mut self, tool_config: &ToolsConfig) -> Self {
        self.max_tool_iterations = tool_config.max_tool_iterations;
        self.confirmation_mode = tool_config.confirmation.clone();
        self
    }

    pub fn with_confirmation_channel(
        mut self,
        rx: mpsc::UnboundedReceiver<ConfirmationResponse>,
    ) -> Self {
        self.confirmation_rx = Some(Arc::new(tokio::sync::Mutex::new(rx)));
        self
    }

    pub fn history(&self) -> Vec<Message> {
        lock(&self.history).clone()
    }

    pub async fn send(&self, input: String) -> Result<BoxStream<AgentEvent>> {
        lock(&self.history).push(Message::text(Role::User, input));

        let (event_tx, event_rx) = mpsc::unbounded::<AgentEvent>();
        let history_arc = Arc::clone(&self.history);
        let backend = Arc::clone(&self.backend);
        let tools = Arc::clone(&self.tools);
        let config = self.config.clone();
        let max_iterations = self.max_tool_iterations;
        let confirmation_mode = self.confirmation_mode.clone();
        let confirmation_rx = self.confirmation_rx.clone();

        tokio::spawn(async move {
            let mut iterations = 0u32;

            loop {
                if iterations >= max_iterations {
                    let _ = event_tx.unbounded_send(AgentEvent::Error(format!(
                        "Max tool iterations ({max_iterations}) exceeded"
                    )));
                    break;
                }
                iterations += 1;

                let history_snapshot = lock(&history_arc).clone();
                let backend_stream = match backend.send_message(&history_snapshot, &config).await {
                    Ok(s) => s,
                    Err(e) => {
                        if iterations == 1 {
                            lock(&history_arc).pop();
                        }
                        let _ = event_tx.unbounded_send(AgentEvent::Error(e.to_string()));
                        break;
                    }
                };

                let mut text_accumulated = String::new();
                let mut tool_calls: Vec<(String, String, String)> = vec![];
                let mut current_tool: Option<(String, String, String)> = None;
                let mut stream = backend_stream;
                let mut had_error = false;

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(StreamEvent::TextDelta(text)) => {
                            text_accumulated.push_str(&text);
                            let _ = event_tx.unbounded_send(AgentEvent::TokenReceived(text));
                        }
                        Ok(StreamEvent::ToolUseStart { id, name }) => {
                            current_tool = Some((id, name, String::new()));
                        }
                        Ok(StreamEvent::ToolUseDelta(chunk)) => {
                            if let Some((_, _, ref mut acc)) = current_tool {
                                acc.push_str(&chunk);
                            }
                        }
                        Ok(StreamEvent::ToolUseDone) => {
                            if let Some(tool) = current_tool.take() {
                                tool_calls.push(tool);
                            }
                        }
                        Ok(StreamEvent::Done) => break,
                        Err(e) => {
                            let _ = event_tx.unbounded_send(AgentEvent::Error(e.to_string()));
                            had_error = true;
                            break;
                        }
                    }
                }

                if had_error {
                    break;
                }

                if tool_calls.is_empty() {
                    let mut content = vec![];
                    if !text_accumulated.is_empty() {
                        content.push(ContentBlock::Text(text_accumulated.clone()));
                    }
                    lock(&history_arc).push(Message {
                        role: Role::Assistant,
                        content,
                    });
                    let _ = event_tx.unbounded_send(AgentEvent::ResponseComplete(text_accumulated));
                    break;
                }

                let mut assistant_content: Vec<ContentBlock> = vec![];
                if !text_accumulated.is_empty() {
                    assistant_content.push(ContentBlock::Text(text_accumulated));
                }
                let mut tool_result_blocks: Vec<ContentBlock> = vec![];

                for (id, name, input_json) in tool_calls {
                    let input: serde_json::Value =
                        serde_json::from_str(&input_json).unwrap_or(serde_json::Value::Null);

                    assistant_content.push(ContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    });
                    let _ = event_tx.unbounded_send(AgentEvent::ToolUseReceived {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    });

                    let needs_confirmation = match confirmation_mode {
                        ConfirmationMode::Always => true,
                        ConfirmationMode::Never => false,
                        ConfirmationMode::WriteOnly => {
                            tools.lookup(&name).is_ok_and(|t| t.is_write_tool())
                        }
                    };

                    let approved = if needs_confirmation {
                        let _ = event_tx.unbounded_send(AgentEvent::ToolConfirmationRequired {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        });
                        if let Some(ref rx_arc) = confirmation_rx {
                            let mut rx = rx_arc.lock().await;
                            matches!(rx.next().await, Some(ConfirmationResponse::Approved))
                        } else {
                            false
                        }
                    } else {
                        true
                    };

                    let (result_content, is_error) = if approved {
                        match tools.lookup(&name) {
                            Ok(tool) => match tool.execute(input) {
                                Ok(result) => {
                                    let content = result
                                        .content
                                        .iter()
                                        .filter_map(|b| {
                                            if let ContentBlock::Text(s) = b {
                                                Some(s.clone())
                                            } else {
                                                None
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n");
                                    (content, result.is_error)
                                }
                                Err(e) => (e.to_string(), true),
                            },
                            Err(e) => (e.to_string(), true),
                        }
                    } else {
                        ("User declined to execute this tool.".to_string(), true)
                    };

                    let _ = event_tx.unbounded_send(AgentEvent::ToolResult {
                        name: name.clone(),
                        content: result_content.clone(),
                        is_error,
                    });

                    tool_result_blocks.push(ContentBlock::ToolResult {
                        tool_use_id: id,
                        content: result_content,
                        is_error,
                    });
                }

                lock(&history_arc).push(Message {
                    role: Role::Assistant,
                    content: assistant_content,
                });
                lock(&history_arc).push(Message {
                    role: Role::User,
                    content: tool_result_blocks,
                });
            }
        });

        Ok(Box::pin(event_rx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ConfirmationMode, ToolsConfig};
    use crate::tools::{Tool, ToolError, ToolResult as ToolExecResult};
    use anyhow::Result;
    use async_trait::async_trait;
    use futures::{StreamExt, channel::mpsc, stream};

    struct SequencedBackend {
        responses: Arc<tokio::sync::Mutex<Vec<Vec<Result<StreamEvent>>>>>,
    }

    impl SequencedBackend {
        fn new(responses: Vec<Vec<Result<StreamEvent>>>) -> Self {
            Self {
                responses: Arc::new(tokio::sync::Mutex::new(responses)),
            }
        }
    }

    #[async_trait]
    impl LlmBackend for SequencedBackend {
        async fn send_message(
            &self,
            _: &[Message],
            _: &RequestConfig,
        ) -> Result<BoxStream<Result<StreamEvent>>> {
            let mut lock = self.responses.lock().await;
            let events = if lock.is_empty() {
                vec![Ok(StreamEvent::Done)]
            } else {
                lock.remove(0)
            };
            Ok(Box::pin(stream::iter(events)))
        }
    }

    fn text_response(text: &str) -> Vec<Result<StreamEvent>> {
        vec![
            Ok(StreamEvent::TextDelta(text.to_string())),
            Ok(StreamEvent::Done),
        ]
    }

    fn tool_call_response(id: &str, name: &str, input: &str) -> Vec<Result<StreamEvent>> {
        vec![
            Ok(StreamEvent::ToolUseStart {
                id: id.to_string(),
                name: name.to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta(input.to_string())),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ]
    }

    struct EchoTool {
        name: String,
        output: String,
        is_write: bool,
        schema: serde_json::Value,
    }

    impl EchoTool {
        fn new(name: &str, output: &str) -> Self {
            Self {
                name: name.to_string(),
                output: output.to_string(),
                is_write: false,
                schema: serde_json::json!({"type": "object", "properties": {}}),
            }
        }

        fn write_tool(name: &str, output: &str) -> Self {
            Self {
                is_write: true,
                ..Self::new(name, output)
            }
        }
    }

    impl Tool for EchoTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "Echo tool"
        }
        fn input_schema(&self) -> &serde_json::Value {
            &self.schema
        }
        fn is_write_tool(&self) -> bool {
            self.is_write
        }
        fn execute(&self, _input: serde_json::Value) -> Result<ToolExecResult, ToolError> {
            Ok(ToolExecResult {
                content: vec![ContentBlock::Text(self.output.clone())],
                is_error: false,
            })
        }
    }

    fn agent_with_mode(
        backend: impl LlmBackend + 'static,
        tool: Option<Box<dyn crate::tools::Tool>>,
        mode: ConfirmationMode,
    ) -> Agent {
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let mut registry = ToolRegistry::new();
        if let Some(t) = tool {
            registry.register(t).expect("register tool");
        }
        let tool_config = ToolsConfig {
            confirmation: mode,
            ..Default::default()
        };
        Agent::new(Box::new(backend), config)
            .with_tools(registry)
            .with_tool_config(&tool_config)
    }

    async fn collect_events(stream: BoxStream<AgentEvent>) -> Vec<AgentEvent> {
        let mut stream = stream;
        let mut events = vec![];
        while let Some(event) = stream.next().await {
            events.push(event);
        }
        events
    }

    #[tokio::test]
    async fn pure_text_response_emits_response_complete_without_tool_loop() {
        let backend = SequencedBackend::new(vec![text_response("hello world")]);
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never);

        let stream = agent
            .send("hi".to_string())
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::TokenReceived(t) if t == "hello world")),
            "expected TokenReceived"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "expected ResponseComplete"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolUseReceived { .. })),
            "must not have ToolUseReceived"
        );
    }

    #[tokio::test]
    async fn tool_use_response_with_never_confirmation_executes_tool_and_re_sends() {
        let backend = SequencedBackend::new(vec![
            tool_call_response("tool-1", "bash", r#"{}"#),
            text_response("done"),
        ]);
        let agent = agent_with_mode(
            backend,
            Some(Box::new(EchoTool::new("bash", "ls output"))),
            ConfirmationMode::Never,
        );

        let stream = agent
            .send("run ls".to_string())
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolUseReceived { name, .. } if name == "bash")),
            "expected ToolUseReceived"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                AgentEvent::ToolResult { name, content, is_error }
                    if name == "bash" && content == "ls output" && !is_error
            )),
            "expected ToolResult with ls output"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "expected ResponseComplete"
        );
    }

    #[tokio::test]
    async fn always_confirmation_emits_tool_confirmation_required() {
        let backend = SequencedBackend::new(vec![
            tool_call_response("tool-1", "bash", r#"{}"#),
            text_response("done"),
        ]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(EchoTool::new("bash", "output")))
            .expect("register");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Always,
            ..Default::default()
        };
        let (confirm_tx, confirm_rx) = mpsc::unbounded::<ConfirmationResponse>();
        confirm_tx
            .unbounded_send(ConfirmationResponse::Approved)
            .expect("send approval");

        let agent = Agent::new(Box::new(backend), config)
            .with_tools(registry)
            .with_tool_config(&tool_config)
            .with_confirmation_channel(confirm_rx);

        let stream = agent
            .send("run".to_string())
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events.iter().any(|e| matches!(
                e,
                AgentEvent::ToolConfirmationRequired { name, .. } if name == "bash"
            )),
            "expected ToolConfirmationRequired"
        );
    }

    #[tokio::test]
    async fn confirmation_approved_executes_tool() {
        let backend = SequencedBackend::new(vec![
            tool_call_response("tool-1", "bash", r#"{}"#),
            text_response("done"),
        ]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(EchoTool::new("bash", "the output")))
            .expect("register");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Always,
            ..Default::default()
        };
        let (confirm_tx, confirm_rx) = mpsc::unbounded::<ConfirmationResponse>();
        confirm_tx
            .unbounded_send(ConfirmationResponse::Approved)
            .expect("send approval");

        let agent = Agent::new(Box::new(backend), config)
            .with_tools(registry)
            .with_tool_config(&tool_config)
            .with_confirmation_channel(confirm_rx);

        let stream = agent
            .send("run".to_string())
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events.iter().any(|e| matches!(
                e,
                AgentEvent::ToolResult { content, is_error, .. }
                    if content == "the output" && !is_error
            )),
            "expected successful ToolResult"
        );
    }

    #[tokio::test]
    async fn confirmation_rejected_sends_error_result_to_model() {
        let backend = SequencedBackend::new(vec![
            tool_call_response("tool-1", "bash", r#"{}"#),
            text_response("ok"),
        ]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(EchoTool::new("bash", "output")))
            .expect("register");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Always,
            ..Default::default()
        };
        let (confirm_tx, confirm_rx) = mpsc::unbounded::<ConfirmationResponse>();
        confirm_tx
            .unbounded_send(ConfirmationResponse::Rejected)
            .expect("send rejection");

        let agent = Agent::new(Box::new(backend), config)
            .with_tools(registry)
            .with_tool_config(&tool_config)
            .with_confirmation_channel(confirm_rx);

        let stream = agent
            .send("run".to_string())
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolResult { is_error, .. } if *is_error)),
            "expected error ToolResult"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "expected ResponseComplete after rejected tool"
        );
    }

    #[tokio::test]
    async fn max_iterations_exceeded_emits_error() {
        let responses: Vec<Vec<Result<StreamEvent>>> = (0..30)
            .map(|_| tool_call_response("tool-1", "bash", r#"{}"#))
            .collect();
        let backend = SequencedBackend::new(responses);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(EchoTool::new("bash", "output")))
            .expect("register");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            max_tool_iterations: 3,
            ..Default::default()
        };

        let agent = Agent::new(Box::new(backend), config)
            .with_tools(registry)
            .with_tool_config(&tool_config);

        let stream = agent
            .send("run".to_string())
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events.iter().any(|e| matches!(e, AgentEvent::Error(_))),
            "expected Error event"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "must not have ResponseComplete"
        );
    }

    #[tokio::test]
    async fn multiple_tool_calls_in_single_response_all_executed() {
        let multi_tool_response: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::ToolUseStart {
                id: "tool-1".to_string(),
                name: "bash".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta(r#"{}"#.to_string())),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::ToolUseStart {
                id: "tool-2".to_string(),
                name: "bash".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta(r#"{}"#.to_string())),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ];
        let backend = SequencedBackend::new(vec![multi_tool_response, text_response("done")]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(EchoTool::new("bash", "output")))
            .expect("register");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };

        let agent = Agent::new(Box::new(backend), config)
            .with_tools(registry)
            .with_tool_config(&tool_config);

        let stream = agent
            .send("run".to_string())
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        let tool_results: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolResult { .. }))
            .collect();
        assert_eq!(tool_results.len(), 2, "expected two ToolResult events");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "expected ResponseComplete"
        );
    }

    #[tokio::test]
    async fn write_only_confirmation_requires_confirmation_for_write_tools_only() {
        let backend = SequencedBackend::new(vec![
            tool_call_response("tool-1", "write_file", r#"{}"#),
            text_response("done"),
        ]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(EchoTool::write_tool("write_file", "written")))
            .expect("register");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::WriteOnly,
            ..Default::default()
        };
        let (confirm_tx, confirm_rx) = mpsc::unbounded::<ConfirmationResponse>();
        confirm_tx
            .unbounded_send(ConfirmationResponse::Approved)
            .expect("send approval");

        let agent = Agent::new(Box::new(backend), config)
            .with_tools(registry)
            .with_tool_config(&tool_config)
            .with_confirmation_channel(confirm_rx);

        let stream = agent
            .send("write".to_string())
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolConfirmationRequired { .. })),
            "write tool should require confirmation in WriteOnly mode"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                AgentEvent::ToolResult { is_error, .. } if !is_error
            )),
            "tool should execute after approval"
        );
    }
}
