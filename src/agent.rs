use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures::StreamExt;
use futures::channel::mpsc;

use crate::backend::LlmBackend;
use crate::config::{ConfirmationMode, ToolsConfig};
use crate::context_files::{ContextFile, discover_context_files_from_env};
use crate::session::{Session, persist_to_session};
use crate::tools::ToolRegistry;
use crate::types::{
    AgentEvent, BoxStream, ConfirmationResponse, ContentBlock, Message, RequestConfig, Role,
    StreamEvent,
};

struct PendingToolCall {
    id: String,
    name: String,
    input_json: String,
}

pub struct Agent {
    backend: Arc<dyn LlmBackend>,
    history: Arc<Mutex<Vec<Message>>>,
    config: RequestConfig,
    tools: Arc<ToolRegistry>,
    max_tool_iterations: u32,
    confirmation_mode: ConfirmationMode,
    session: Arc<Mutex<Option<Session>>>,
}

// Recover from a poisoned mutex: a thread panicked while holding the lock, leaving
// history in an unknown state. Panicking here would crash the app; accepting partial
// corruption is the lesser evil for a long-running interactive process.
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
            session: Arc::new(Mutex::new(None)),
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

    pub async fn with_session(self, session: Session) -> Self {
        let existing_history: Vec<Message> = session.load_history().await.unwrap_or_default();
        if !existing_history.is_empty() {
            lock(&self.history).extend(existing_history);
        }
        *self.session.lock().unwrap_or_else(|e| e.into_inner()) = Some(session);
        self
    }

    pub fn with_context_files(self) -> Result<Self> {
        let files = discover_context_files_from_env()?;
        self.load_context_files(files);
        Ok(self)
    }

    pub fn with_skills(
        self,
        skills: &std::collections::HashMap<String, std::path::PathBuf>,
    ) -> Self {
        if skills.is_empty() {
            return self;
        }

        let mut names: Vec<&str> = skills.keys().map(String::as_str).collect();
        names.sort();

        let mut content =
            String::from("The following skills are available via the `skill` tool:\n\n");
        for name in names {
            let path = &skills[name];
            let desc = crate::tools::skill::skill_description(path)
                .unwrap_or_else(|| "(no description)".to_string());
            content.push_str(&format!("- {}: {}\n", name, desc));
        }

        let msg = Message::system(Role::User, content);
        lock(&self.history).push(msg.clone());
        self
    }

    pub fn tools(&self) -> Arc<ToolRegistry> {
        Arc::clone(&self.tools)
    }

    pub fn history(&self) -> Vec<Message> {
        lock(&self.history).clone()
    }

    pub fn load_context_files(&self, files: Vec<ContextFile>) {
        if files.is_empty() {
            return;
        }

        let mut content = String::from("The following context files were loaded:\n\n");
        for file in files {
            content.push_str(&format!(
                "## File: {}\n\n{}\n\n",
                file.path.display(),
                file.content
            ));
        }

        let msg = Message::system(Role::User, content);
        // prepend context files and don't persist them to the DB
        lock(&self.history).insert(0, msg.clone());
    }

    pub async fn send(
        &self,
        input: String,
        confirmation_rx: Option<mpsc::UnboundedReceiver<ConfirmationResponse>>,
    ) -> Result<BoxStream<AgentEvent>> {
        let pre_send_len = lock(&self.history).len();
        let user_msg = Message::text(Role::User, input);
        lock(&self.history).push(user_msg.clone());

        let (event_tx, event_rx) = mpsc::unbounded::<AgentEvent>();
        let history_arc = Arc::clone(&self.history);
        let backend = Arc::clone(&self.backend);
        let tools = Arc::clone(&self.tools);
        let config = self.config.clone();
        let max_iterations = self.max_tool_iterations;
        let confirmation_mode = self.confirmation_mode.clone();
        let session = Arc::clone(&self.session);

        persist_to_session(&session, &user_msg).await;

        tokio::spawn(async move {
            let mut iterations = 0u32;
            let mut confirmation_rx = confirmation_rx;

            'outer: loop {
                if iterations >= max_iterations {
                    lock(&history_arc).truncate(pre_send_len);
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
                        lock(&history_arc).truncate(pre_send_len);
                        let _ = event_tx.unbounded_send(AgentEvent::Error(e.to_string()));
                        break;
                    }
                };

                let mut text_accumulated = String::new();
                let mut tool_calls: Vec<PendingToolCall> = vec![];
                let mut current_tool: Option<PendingToolCall> = None;
                let mut stream = backend_stream;

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(StreamEvent::TextDelta(text)) => {
                            text_accumulated.push_str(&text);
                            let _ = event_tx.unbounded_send(AgentEvent::TokenReceived(text));
                        }
                        Ok(StreamEvent::ToolUseStart { id, name }) => {
                            current_tool = Some(PendingToolCall {
                                id,
                                name,
                                input_json: String::new(),
                            });
                        }
                        Ok(StreamEvent::ToolUseDelta(chunk)) => {
                            if let Some(ref mut t) = current_tool {
                                t.input_json.push_str(&chunk);
                            }
                        }
                        Ok(StreamEvent::ToolUseDone) => {
                            if let Some(t) = current_tool.take() {
                                tool_calls.push(t);
                            }
                        }
                        Ok(StreamEvent::Usage {
                            input_tokens,
                            output_tokens,
                            stop_reason,
                        }) => {
                            let _ = event_tx.unbounded_send(AgentEvent::Usage {
                                input_tokens,
                                output_tokens,
                                stop_reason,
                            });
                        }
                        Ok(StreamEvent::Done) => break,
                        Err(e) => {
                            lock(&history_arc).truncate(pre_send_len);
                            let _ = event_tx.unbounded_send(AgentEvent::Error(e.to_string()));
                            break 'outer;
                        }
                    }
                }

                if tool_calls.is_empty() {
                    let mut content = vec![];
                    if !text_accumulated.is_empty() {
                        content.push(ContentBlock::Text(text_accumulated.clone()));
                    }
                    let assistant_msg = Message {
                        role: Role::Assistant,
                        content,
                        hidden: false,
                    };
                    lock(&history_arc).push(assistant_msg.clone());
                    persist_to_session(&session, &assistant_msg).await;
                    let _ = event_tx.unbounded_send(AgentEvent::ResponseComplete(text_accumulated));
                    break;
                }

                let (assistant_content, tool_result_blocks) = execute_tool_calls(
                    tool_calls,
                    text_accumulated,
                    &tools,
                    &confirmation_mode,
                    &mut confirmation_rx,
                    &event_tx,
                )
                .await;

                let assistant_msg = Message {
                    role: Role::Assistant,
                    content: assistant_content,
                    hidden: false,
                };
                lock(&history_arc).push(assistant_msg.clone());
                persist_to_session(&session, &assistant_msg).await;

                let tool_result_msg = Message {
                    role: Role::User,
                    content: tool_result_blocks,
                    hidden: false,
                };
                lock(&history_arc).push(tool_result_msg.clone());
                persist_to_session(&session, &tool_result_msg).await;
            }
        });

        Ok(Box::pin(event_rx))
    }
}

async fn execute_tool_calls(
    tool_calls: Vec<PendingToolCall>,
    text_prefix: String,
    tools: &ToolRegistry,
    confirmation_mode: &ConfirmationMode,
    confirmation_rx: &mut Option<mpsc::UnboundedReceiver<ConfirmationResponse>>,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
) -> (Vec<ContentBlock>, Vec<ContentBlock>) {
    let mut assistant_content: Vec<ContentBlock> = vec![];
    if !text_prefix.is_empty() {
        assistant_content.push(ContentBlock::Text(text_prefix));
    }

    // TODO: tool calls within a single response are independent and could be executed concurrently.
    let mut tool_result_blocks: Vec<ContentBlock> = vec![];

    for call in tool_calls {
        let (input, parse_error) = match serde_json::from_str::<serde_json::Value>(&call.input_json)
        {
            Ok(v) => (v, None),
            Err(e) => (
                serde_json::Value::Null,
                Some(format!("Invalid tool input JSON: {e}")),
            ),
        };

        assistant_content.push(ContentBlock::ToolUse {
            id: call.id.clone(),
            name: call.name.clone(),
            input: input.clone(),
        });
        let _ = event_tx.unbounded_send(AgentEvent::ToolUseReceived {
            id: call.id.clone(),
            name: call.name.clone(),
            input: input.clone(),
        });

        if let Some(err_msg) = parse_error {
            let _ = event_tx.unbounded_send(AgentEvent::ToolResult {
                name: call.name.clone(),
                content: err_msg.clone(),
                is_error: true,
            });
            tool_result_blocks.push(ContentBlock::ToolResult {
                tool_use_id: call.id,
                content: err_msg,
                is_error: true,
            });
            continue;
        }

        let needs_confirmation = match confirmation_mode {
            ConfirmationMode::Always => true,
            ConfirmationMode::Never => false,
            ConfirmationMode::WriteOnly => {
                tools.lookup(&call.name).is_ok_and(|t| t.is_write_tool())
            }
        };

        let approved = if needs_confirmation {
            let _ = event_tx.unbounded_send(AgentEvent::ToolConfirmationRequired {
                id: call.id.clone(),
                name: call.name.clone(),
                input: input.clone(),
            });
            if let Some(rx) = confirmation_rx.as_mut() {
                matches!(rx.next().await, Some(ConfirmationResponse::Approved))
            } else {
                false
            }
        } else {
            true
        };

        let (result_content, is_error) = if approved {
            match tools.lookup(&call.name) {
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
            name: call.name.clone(),
            content: result_content.clone(),
            is_error,
        });

        tool_result_blocks.push(ContentBlock::ToolResult {
            tool_use_id: call.id,
            content: result_content,
            is_error,
        });
    }

    (assistant_content, tool_result_blocks)
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
            .send("hi".to_string(), None)
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
            .send("run ls".to_string(), None)
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
            .with_tool_config(&tool_config);

        let stream = agent
            .send("run".to_string(), Some(confirm_rx))
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
            .with_tool_config(&tool_config);

        let stream = agent
            .send("run".to_string(), Some(confirm_rx))
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
            .with_tool_config(&tool_config);

        let stream = agent
            .send("run".to_string(), Some(confirm_rx))
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
            .send("run".to_string(), None)
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
            .send("run".to_string(), None)
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
    async fn malformed_tool_input_json_sends_error_result_to_model() {
        let bad_json_response: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::ToolUseStart {
                id: "t1".to_string(),
                name: "bash".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta("not valid json {{{".to_string())),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ];
        let backend = SequencedBackend::new(vec![bad_json_response, text_response("ok")]);
        let agent = agent_with_mode(
            backend,
            Some(Box::new(EchoTool::new("bash", "output"))),
            ConfirmationMode::Never,
        );

        let stream = agent
            .send("run".to_string(), None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolResult { is_error, .. } if *is_error)),
            "malformed JSON should produce error ToolResult"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "loop should continue and complete after malformed input"
        );
    }

    #[tokio::test]
    async fn backend_error_on_second_iteration_clears_history() {
        let backend = SequencedBackend::new(vec![
            tool_call_response("t1", "bash", r#"{}"#),
            vec![Err(anyhow::anyhow!("backend failure on iteration 2"))],
        ]);
        let agent = agent_with_mode(
            backend,
            Some(Box::new(EchoTool::new("bash", "output"))),
            ConfirmationMode::Never,
        );

        let stream = agent
            .send("run".to_string(), None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events.iter().any(|e| matches!(e, AgentEvent::Error(_))),
            "expected Error event"
        );
        assert!(
            agent.history().is_empty(),
            "history must be fully cleared after mid-loop backend error"
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
            .with_tool_config(&tool_config);

        let stream = agent
            .send("write".to_string(), Some(confirm_rx))
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

    #[test]
    fn load_skills_adds_skill_names_to_history() {
        let backend = SequencedBackend::new(vec![]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let agent = Agent::new(Box::new(backend), config);

        let mut skills = std::collections::HashMap::new();
        skills.insert(
            "my-skill".to_string(),
            std::path::PathBuf::from("/fake/path"),
        );
        skills.insert(
            "another-skill".to_string(),
            std::path::PathBuf::from("/fake/path2"),
        );
        agent.load_skills(&skills);

        let history = agent.history();
        assert_eq!(history.len(), 1);
        assert!(history[0].hidden, "skill message should be hidden");
        match &history[0].content[0] {
            ContentBlock::Text(text) => {
                assert!(
                    text.contains("another-skill"),
                    "should mention another-skill"
                );
                assert!(text.contains("my-skill"), "should mention my-skill");
                assert!(text.contains("skill"), "should mention skill tool");
            }
            _ => panic!("expected Text content"),
        }
    }

    #[test]
    fn load_skills_with_empty_map_adds_nothing_to_history() {
        let backend = SequencedBackend::new(vec![]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let agent = Agent::new(Box::new(backend), config);

        agent.load_skills(&std::collections::HashMap::new());

        assert!(
            agent.history().is_empty(),
            "empty skills map should not add history entry"
        );
    }

    #[test]
    fn load_skills_skips_when_prefix_already_in_history() {
        let backend = SequencedBackend::new(vec![]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let agent = Agent::new(Box::new(backend), config);

        let mut skills = std::collections::HashMap::new();
        skills.insert(
            "my-skill".to_string(),
            std::path::PathBuf::from("/fake/path"),
        );
        agent.load_skills(&skills);
        assert_eq!(agent.history().len(), 1, "first call should add entry");

        agent.load_skills(&skills);
        assert_eq!(
            agent.history().len(),
            1,
            "second call should be deduplicated"
        );
    }

    #[test]
    fn load_context_files_skips_when_prefix_already_in_history() {
        let backend = SequencedBackend::new(vec![]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let agent = Agent::new(Box::new(backend), config);

        let files = vec![crate::context_files::ContextFile {
            path: std::path::PathBuf::from("/test.md"),
            content: "hello".to_string(),
        }];
        agent.load_context_files(files.clone());
        assert_eq!(agent.history().len(), 1, "first call should add entry");

        agent.load_context_files(files);
        assert_eq!(
            agent.history().len(),
            1,
            "second call should be deduplicated"
        );
    }

    #[test]
    fn load_context_files_creates_hidden_message() {
        let backend = SequencedBackend::new(vec![]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let agent = Agent::new(Box::new(backend), config);

        let files = vec![crate::context_files::ContextFile {
            path: std::path::PathBuf::from("/test.md"),
            content: "hello".to_string(),
        }];
        agent.load_context_files(files);

        let history = agent.history();
        assert_eq!(history.len(), 1);
        assert!(history[0].hidden, "context file message should be hidden");
    }

    #[tokio::test]
    async fn context_files_preserved_after_session_restore() {
        use crate::context_files::ContextFile;

        let dir = tempfile::TempDir::new().expect("temp dir");
        let session = Session::create(dir.path(), None)
            .await
            .expect("create session");
        session
            .insert_message(&Message::text(Role::User, "previous message".to_string()))
            .await
            .expect("insert");

        let backend = SequencedBackend::new(vec![]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };

        let agent = Agent::new(Box::new(backend), config)
            .with_session(session)
            .await;

        assert_eq!(
            agent.history().len(),
            1,
            "session history should be restored"
        );

        let files = vec![ContextFile {
            path: std::path::PathBuf::from("/test.md"),
            content: "hello".to_string(),
        }];
        agent.load_context_files(files);

        assert_eq!(
            agent.history().len(),
            2,
            "context files should be added on top of restored session history"
        );

        let context_msg = &agent.history()[1];
        match &context_msg.content[0] {
            ContentBlock::Text(t) => assert!(
                t.contains("The following context files were loaded"),
                "context file message should have expected prefix"
            ),
            _ => panic!("expected Text block"),
        }
    }
}
