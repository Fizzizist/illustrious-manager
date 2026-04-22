use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as TokioMutex;

use crate::backend::{BackendFactory, LlmBackend};
use crate::config::{ConfirmationMode, ToolsConfig};
use crate::context_files::{ContextFile, discover_context_files_from_env};
use crate::session::Session;
use crate::tools::ToolRegistry;
use crate::types::{
    AgentEvent, BoxStream, ConfirmationResponse, ContentBlock, Message, RequestConfig, Role,
    StreamEvent,
};
use anyhow::Result;
use futures::StreamExt;
use futures::channel::mpsc;

struct PendingToolCall {
    id: String,
    name: String,
    input_json: String,
}

pub struct Agent {
    backend: Arc<dyn LlmBackend>,
    history: Arc<Mutex<Vec<Message>>>,
    /// Number of messages prepended to `history` that are never persisted to the DB
    /// (context files, skill definitions). Preserved across session switches.
    context_prefix_len: Arc<Mutex<usize>>,
    config: Mutex<RequestConfig>,
    tools: Arc<ToolRegistry>,
    max_tool_iterations: u32,
    confirmation_mode: ConfirmationMode,
    session: Arc<TokioMutex<Session>>,
}

// Recover from a poisoned mutex: a thread panicked while holding the lock, leaving
// history in an unknown state. Panicking here would crash the app; accepting partial
// corruption is the lesser evil for a long-running interactive process.
fn lock(m: &Mutex<Vec<Message>>) -> std::sync::MutexGuard<'_, Vec<Message>> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Agent {
    pub async fn new(
        backend: Box<dyn LlmBackend>,
        config: RequestConfig,
        session: Arc<TokioMutex<Session>>,
    ) -> Self {
        let history: Vec<Message> = session
            .lock()
            .await
            .conversation()
            .load_history()
            .await
            .unwrap_or_default();

        Self {
            backend: Arc::from(backend),
            history: Arc::new(Mutex::new(history)),
            context_prefix_len: Arc::new(Mutex::new(0)),
            config: Mutex::new(config),
            tools: Arc::new(ToolRegistry::new()),
            max_tool_iterations: 25,
            confirmation_mode: ConfirmationMode::WriteOnly,
            session,
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

        let msg = Message::text(Role::User, content);
        lock(&self.history).insert(0, msg.clone());
        *self
            .context_prefix_len
            .lock()
            .unwrap_or_else(|e| e.into_inner()) += 1;
        self
    }

    pub fn tools(&self) -> Arc<ToolRegistry> {
        Arc::clone(&self.tools)
    }

    pub fn model(&self) -> String {
        self.config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .model
            .clone()
    }

    pub fn set_model(&self, model: String) {
        self.config.lock().unwrap_or_else(|e| e.into_inner()).model = model;
    }

    #[cfg(test)]
    pub fn max_tool_iterations_for_test(&self) -> u32 {
        self.max_tool_iterations
    }

    #[cfg(test)]
    pub fn confirmation_mode_for_test(&self) -> &ConfirmationMode {
        &self.confirmation_mode
    }

    pub fn history(&self) -> Vec<Message> {
        lock(&self.history).clone()
    }

    pub async fn session_id(&self) -> String {
        self.session.lock().await.id.clone()
    }

    pub async fn session_history(&self) -> Result<Vec<Message>, anyhow::Error> {
        self.session
            .lock()
            .await
            .conversation()
            .load_history()
            .await
    }

    /// If the current session has no messages, delete its DB file from disk.
    pub async fn cleanup_empty_session(&self) -> Result<()> {
        let session = self.session.lock().await;
        if session.conversation().is_empty().await? {
            session.delete_db()?;
        }
        Ok(())
    }

    /// Replace the current session with a new one and reload the conversation history.
    /// Non-persisted context messages (context files, skill definitions) are preserved
    /// at the front of history; only the persisted portion is replaced.
    pub async fn load_session(&self, session: Session) {
        let new_history = session
            .conversation()
            .load_history()
            .await
            .unwrap_or_default();
        {
            let prefix_len = *self
                .context_prefix_len
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut history = lock(&self.history);
            let take = prefix_len.min(history.len());
            let prefix: Vec<Message> = history.drain(..take).collect();
            *history = prefix;
            history.extend(new_history);
        }
        *self.session.lock().await = session;
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

        let msg = Message::text(Role::User, content);
        // prepend context files and don't persist them to the DB
        lock(&self.history).insert(0, msg.clone());
        *self
            .context_prefix_len
            .lock()
            .unwrap_or_else(|e| e.into_inner()) += 1;
    }

    pub async fn send(
        &self,
        input: String,
        confirmation_rx: Option<mpsc::UnboundedReceiver<ConfirmationResponse>>,
    ) -> Result<BoxStream<AgentEvent>> {
        let user_msg = Message::text(Role::User, input);
        lock(&self.history).push(user_msg.clone());

        let (event_tx, event_rx) = mpsc::unbounded::<AgentEvent>();
        let history_arc = Arc::clone(&self.history);
        let backend = Arc::clone(&self.backend);
        let tools = Arc::clone(&self.tools);
        let config = self
            .config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let max_iterations = self.max_tool_iterations;
        let confirmation_mode = self.confirmation_mode.clone();
        let session = Arc::clone(&self.session);

        self.session
            .lock()
            .await
            .conversation()
            .insert_message(&user_msg)
            .await?;

        tokio::spawn(async move {
            let mut iterations = 0u32;
            let mut confirmation_rx = confirmation_rx;

            'outer: loop {
                if iterations >= max_iterations {
                    let error_msg = format!("Max tool iterations ({max_iterations}) exceeded");
                    record_error(&error_msg, &history_arc, &session, &event_tx).await;
                    break;
                }
                iterations += 1;

                let history_snapshot = lock(&history_arc).clone();
                let backend_stream = match backend.send_message(&history_snapshot, &config).await {
                    Ok(s) => s,
                    Err(e) => {
                        record_error(&e.to_string(), &history_arc, &session, &event_tx).await;
                        break;
                    }
                };

                let mut text_accumulated = String::new();
                let mut tool_calls: Vec<PendingToolCall> = vec![];
                let mut current_tool: Option<PendingToolCall> = None;
                let mut pending_by_index: std::collections::HashMap<u64, PendingToolCall> =
                    std::collections::HashMap::new();
                let mut has_indexed_tools = false;
                let mut stream = backend_stream;

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(StreamEvent::TextDelta(text)) => {
                            text_accumulated.push_str(&text);
                            let _ = event_tx.unbounded_send(AgentEvent::TokenReceived(text));
                        }
                        Ok(StreamEvent::ToolUseStart { id, name, index }) => {
                            let tool = PendingToolCall {
                                id,
                                name,
                                input_json: String::new(),
                            };
                            match index {
                                Some(idx) => {
                                    has_indexed_tools = true;
                                    pending_by_index.insert(idx, tool);
                                }
                                None => {
                                    current_tool = Some(tool);
                                }
                            }
                        }
                        Ok(StreamEvent::ToolUseDelta { chunk, index }) => match index {
                            Some(idx) => {
                                if let Some(t) = pending_by_index.get_mut(&idx) {
                                    t.input_json.push_str(&chunk);
                                }
                            }
                            None => {
                                if let Some(ref mut t) = current_tool {
                                    t.input_json.push_str(&chunk);
                                }
                            }
                        },
                        Ok(StreamEvent::ToolUseDone) => {
                            if let Some(t) = current_tool.take() {
                                tool_calls.push(t);
                            }
                            // For indexed tools (OpenAI-compatible), ToolUseDone
                            // signals all tools are complete. Collect them in
                            // index order.
                            if has_indexed_tools && current_tool.is_none() {
                                let max_index = pending_by_index.keys().max().copied();
                                if let Some(max) = max_index {
                                    for i in 0..=max {
                                        if let Some(t) = pending_by_index.remove(&i) {
                                            tool_calls.push(t);
                                        }
                                    }
                                }
                                has_indexed_tools = false;
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
                            if !text_accumulated.is_empty() {
                                let partial_msg = Message {
                                    role: Role::Assistant,
                                    content: vec![ContentBlock::Text(text_accumulated.clone())],
                                };
                                lock(&history_arc).push(partial_msg.clone());
                                let _ = session
                                    .lock()
                                    .await
                                    .conversation()
                                    .insert_message(&partial_msg)
                                    .await;
                            }
                            record_error(&e.to_string(), &history_arc, &session, &event_tx).await;
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
                    };
                    lock(&history_arc).push(assistant_msg.clone());
                    let _ = session
                        .lock()
                        .await
                        .conversation()
                        .insert_message(&assistant_msg)
                        .await;
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
                };
                lock(&history_arc).push(assistant_msg.clone());
                let _ = session
                    .lock()
                    .await
                    .conversation()
                    .insert_message(&assistant_msg)
                    .await;

                let tool_result_msg = Message {
                    role: Role::User,
                    content: tool_result_blocks,
                };
                lock(&history_arc).push(tool_result_msg.clone());
                let _ = session
                    .lock()
                    .await
                    .conversation()
                    .insert_message(&tool_result_msg)
                    .await;
            }
        });

        Ok(Box::pin(event_rx))
    }
}

async fn record_error(
    error_msg: &str,
    history: &Arc<Mutex<Vec<Message>>>,
    session: &Arc<TokioMutex<Session>>,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
) {
    let error_user_msg = Message::text(Role::User, format!("[ERROR] {error_msg}"));
    lock(history).push(error_user_msg.clone());
    let _ = session
        .lock()
        .await
        .conversation()
        .insert_message(&error_user_msg)
        .await;
    let _ = event_tx.unbounded_send(AgentEvent::Error(error_msg.to_string()));
}

async fn execute_tool_calls(
    tool_calls: Vec<PendingToolCall>,
    text_prefix: String,
    tools: &Arc<ToolRegistry>,
    confirmation_mode: &ConfirmationMode,
    confirmation_rx: &mut Option<mpsc::UnboundedReceiver<ConfirmationResponse>>,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
) -> (Vec<ContentBlock>, Vec<ContentBlock>) {
    let mut assistant_content: Vec<ContentBlock> = vec![];
    if !text_prefix.is_empty() {
        assistant_content.push(ContentBlock::Text(text_prefix));
    }

    // Parse inputs and pre-build the assistant ToolUse blocks. Emit all
    // ToolUseReceived events synchronously in input order before any execution.
    struct ParsedCall {
        id: String,
        name: String,
        input: serde_json::Value,
        parse_error: Option<String>,
    }

    let parsed: Vec<ParsedCall> = tool_calls
        .into_iter()
        .map(|call| {
            let (input, parse_error) = if call.input_json.trim().is_empty() {
                (serde_json::Value::Object(serde_json::Map::new()), None)
            } else {
                match serde_json::from_str::<serde_json::Value>(&call.input_json) {
                    Ok(v) => (v, None),
                    Err(e) => (
                        serde_json::Value::Null,
                        Some(format!("Invalid tool input JSON: {e}")),
                    ),
                }
            };
            ParsedCall {
                id: call.id,
                name: call.name,
                input,
                parse_error,
            }
        })
        .collect();

    let batch_size = parsed.len();
    let show_index = batch_size > 1;

    for (i, call) in parsed.iter().enumerate() {
        assistant_content.push(ContentBlock::ToolUse {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.input.clone(),
        });
        let _ = event_tx.unbounded_send(AgentEvent::ToolUseReceived {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.input.clone(),
            index: if show_index { Some(i as u64 + 1) } else { None },
        });
    }

    // Confirmation gate: walk in input order, request confirmation for each
    // tool that needs it, collect approvals sequentially.
    let mut approvals: Vec<bool> = Vec::with_capacity(parsed.len());
    for call in &parsed {
        if call.parse_error.is_some() {
            approvals.push(false);
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
                input: call.input.clone(),
            });
            if let Some(rx) = confirmation_rx.as_mut() {
                matches!(rx.next().await, Some(ConfirmationResponse::Approved))
            } else {
                false
            }
        } else {
            true
        };
        approvals.push(approved);
    }

    // Dispatch all approved tools concurrently via FuturesUnordered so each
    // completion emits its ToolResult event immediately. Slot-vector preserves
    // input order for the final tool_result_blocks.
    let n = parsed.len();
    let mut slots: Vec<Option<(String, String, String, bool)>> = (0..n).map(|_| None).collect();

    let mut futures = futures::stream::FuturesUnordered::new();

    for (i, call) in parsed.iter().enumerate() {
        let display_index = if show_index { Some(i as u64 + 1) } else { None };
        let id = call.id.clone();
        let name = call.name.clone();
        let input = call.input.clone();
        let approved = approvals[i];
        let parse_error = call.parse_error.clone();

        // Handle non-dispatchable cases: parse errors, declined tools, and
        // unknown tools all produce synthetic error results without spawning.
        if let Some(err_msg) = parse_error {
            slots[i] = Some((id, name.clone(), err_msg, true));
            let _ = event_tx.unbounded_send(AgentEvent::ToolResult {
                id: call.id.clone(),
                name,
                content: slots[i].as_ref().expect("just-set").2.clone(),
                is_error: true,
                index: display_index,
            });
            continue;
        }
        if !approved {
            let msg = "User declined to execute this tool.".to_string();
            slots[i] = Some((id, name.clone(), msg.clone(), true));
            let _ = event_tx.unbounded_send(AgentEvent::ToolResult {
                id: call.id.clone(),
                name,
                content: msg,
                is_error: true,
                index: display_index,
            });
            continue;
        }
        let tool = match tools.lookup(&call.name) {
            Ok(t) => t,
            Err(e) => {
                let msg = e.to_string();
                slots[i] = Some((id, name.clone(), msg.clone(), true));
                let _ = event_tx.unbounded_send(AgentEvent::ToolResult {
                    id: call.id.clone(),
                    name,
                    content: msg,
                    is_error: true,
                    index: display_index,
                });
                continue;
            }
        };

        // The tool reference borrows &ToolRegistry which outlives the futures.
        let idx_for_closure = i;
        futures.push(async move {
            let (content, is_error) = match tool.execute(input).await {
                Ok(result) => {
                    let text = result
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
                    (text, result.is_error)
                }
                Err(e) => (e.to_string(), true),
            };
            (idx_for_closure, id, name, content, is_error)
        });
    }

    // Collect results as they complete. Each result emits a ToolResult event
    // immediately and fills its slot for ordered reassembly.
    while let Some(result) = futures.next().await {
        let (i, id, name, content, is_error) = result;
        let display_index = if show_index { Some(i as u64 + 1) } else { None };
        let _ = event_tx.unbounded_send(AgentEvent::ToolResult {
            id: id.clone(),
            name: name.clone(),
            content: content.clone(),
            is_error,
            index: display_index,
        });
        slots[i] = Some((id, name, content, is_error));
    }

    // Synthesize error results for any slot that was never filled to preserve
    // the Anthropic tool_use/tool_result count invariant.
    let mut tool_result_blocks: Vec<ContentBlock> = Vec::with_capacity(n);
    for (i, slot) in slots.into_iter().enumerate() {
        let (id, _name, content, is_error) = slot.unwrap_or_else(|| {
            let id = parsed[i].id.clone();
            let name = parsed[i].name.clone();
            let content = "Tool execution failed: result was unexpectedly lost".to_string();
            (id, name, content, true)
        });
        tool_result_blocks.push(ContentBlock::ToolResult {
            tool_use_id: id,
            content,
            is_error,
        });
    }

    (assistant_content, tool_result_blocks)
}

pub(crate) const DEFAULT_MAX_TOKENS: u32 = 8_192;

/// Spawn a fresh `Agent` for the given named role using the shared `BackendFactory`.
///
/// The caller supplies a pre-built `ToolRegistry` and a `Session`. The agent's
/// model is taken from the role definition; `tool_config` controls iteration
/// limits and confirmation behaviour.
pub async fn spawn_agent(
    factory: &BackendFactory,
    role: &str,
    tool_config: &ToolsConfig,
    session: Arc<TokioMutex<Session>>,
    tools: ToolRegistry,
) -> anyhow::Result<Agent> {
    let selection = factory.for_role(role).await?;
    spawn_agent_with_selection(selection, tool_config, session, tools).await
}

/// Core of `spawn_agent` — constructs an `Agent` from an already-resolved
/// `BackendSelection`. Separated out so tests can inject a fake backend without
/// going through real auth.
pub async fn spawn_agent_with_selection(
    selection: crate::backend::BackendSelection,
    tool_config: &ToolsConfig,
    session: Arc<TokioMutex<Session>>,
    tools: ToolRegistry,
) -> anyhow::Result<Agent> {
    let request_config = RequestConfig {
        model: selection.model,
        max_tokens: DEFAULT_MAX_TOKENS,
        tools: tools.definitions(),
    };
    Ok(Agent::new(selection.backend, request_config, session)
        .await
        .with_tools(tools)
        .with_tool_config(tool_config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ConfirmationMode, ToolsConfig};
    use crate::tools::{Tool, ToolError, ToolResult as ToolExecResult};
    use anyhow::Result;
    use async_trait::async_trait;
    use futures::{StreamExt, channel::mpsc, stream};

    async fn test_session() -> Session {
        let dir = tempfile::TempDir::new().expect("temp dir");
        Session::new(None, dir.keep()).await.expect("test session")
    }

    async fn test_session_arc() -> Arc<TokioMutex<Session>> {
        Arc::new(TokioMutex::new(test_session().await))
    }

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
                index: None,
            }),
            Ok(StreamEvent::ToolUseDelta {
                chunk: input.to_string(),
                index: None,
            }),
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

    #[async_trait]
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
        async fn execute(&self, _input: serde_json::Value) -> Result<ToolExecResult, ToolError> {
            Ok(ToolExecResult {
                content: vec![ContentBlock::Text(self.output.clone())],
                is_error: false,
            })
        }
    }

    async fn agent_with_mode(
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
        Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
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
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never).await;

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
        )
        .await;

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
                AgentEvent::ToolResult { name, content, is_error, .. }
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

        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
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

        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
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

        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
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

        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
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
                index: None,
            }),
            Ok(StreamEvent::ToolUseDelta {
                chunk: r#"{}"#.to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::ToolUseStart {
                id: "tool-2".to_string(),
                name: "bash".to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDelta {
                chunk: r#"{}"#.to_string(),
                index: None,
            }),
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

        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
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
                index: None,
            }),
            Ok(StreamEvent::ToolUseDelta {
                chunk: "not valid json {{{".to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ];
        let backend = SequencedBackend::new(vec![bad_json_response, text_response("ok")]);
        let agent = agent_with_mode(
            backend,
            Some(Box::new(EchoTool::new("bash", "output"))),
            ConfirmationMode::Never,
        )
        .await;

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
    async fn empty_tool_input_json_is_treated_as_empty_object() {
        // Regression: LLM sends no input for a no-arg tool (e.g. list_tasks).
        // The SSE stream delivers no ToolUseDelta events, leaving input_json = "".
        // This must not produce an error — it should be treated as {}.
        let empty_input_response: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::ToolUseStart {
                id: "t1".to_string(),
                name: "bash".to_string(),
                index: None,
            }),
            // No ToolUseDelta — input_json stays empty
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ];
        let backend = SequencedBackend::new(vec![empty_input_response, text_response("ok")]);
        let agent = agent_with_mode(
            backend,
            Some(Box::new(EchoTool::new("bash", "echo output"))),
            ConfirmationMode::Never,
        )
        .await;

        let stream = agent
            .send("list".to_string(), None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events.iter().any(|e| matches!(
                e,
                AgentEvent::ToolResult { content, is_error, .. }
                    if content == "echo output" && !is_error
            )),
            "empty input_json should execute successfully with empty object input"
        );
    }

    #[tokio::test]
    async fn backend_error_on_second_iteration_preserves_history() {
        let backend = SequencedBackend::new(vec![
            tool_call_response("t1", "bash", r#"{}"#),
            vec![Err(anyhow::anyhow!("backend failure on iteration 2"))],
        ]);
        let agent = agent_with_mode(
            backend,
            Some(Box::new(EchoTool::new("bash", "output"))),
            ConfirmationMode::Never,
        )
        .await;

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
            !agent.history().is_empty(),
            "history must be preserved after error so agent retains context"
        );
        let history = agent.history();
        assert!(
            history.iter().any(|m| m.role == Role::User
                && m.content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::Text(t) if t == "run"))),
            "user message must be retained in history"
        );
    }

    #[tokio::test]
    async fn stream_error_preserves_user_message_in_history() {
        let backend = SequencedBackend::new(vec![vec![Err(anyhow::anyhow!("connection refused"))]]);
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never).await;

        let stream = agent
            .send("hello".to_string(), None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events.iter().any(|e| matches!(e, AgentEvent::Error(_))),
            "expected Error event"
        );
        let history = agent.history();
        assert!(
            !history.is_empty(),
            "history must not be cleared on stream error"
        );
        assert!(
            history.iter().any(|m| m.role == Role::User
                && m.content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::Text(t) if t == "hello"))),
            "user message must be retained"
        );
    }

    struct FailingBackend {
        error_message: String,
    }

    #[async_trait]
    impl LlmBackend for FailingBackend {
        async fn send_message(
            &self,
            _: &[Message],
            _: &RequestConfig,
        ) -> Result<BoxStream<Result<StreamEvent>>> {
            Err(anyhow::anyhow!("{}", self.error_message))
        }
    }

    #[tokio::test]
    async fn backend_send_error_preserves_user_message_in_history() {
        let backend = FailingBackend {
            error_message: "connection refused".to_string(),
        };
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never).await;

        let stream = agent
            .send("hello".to_string(), None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::Error(msg) if msg.contains("connection refused"))),
            "expected Error event with connection refused message"
        );
        let history = agent.history();
        assert!(
            !history.is_empty(),
            "history must not be cleared on send_message error"
        );
        assert!(
            history.iter().any(|m| m.role == Role::User
                && m.content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::Text(t) if t == "hello"))),
            "user message must be retained"
        );
        assert!(
            history.iter().any(|m| m.role == Role::User
                && m.content.iter().any(
                    |b| matches!(b, ContentBlock::Text(t) if t.contains("[ERROR]") && t.contains("connection refused"))
                )),
            "error message with [ERROR] prefix must be in history"
        );
    }

    #[tokio::test]
    async fn stream_error_preserves_partial_text_in_history() {
        let backend = SequencedBackend::new(vec![vec![
            Ok(StreamEvent::TextDelta("partial response".to_string())),
            Err(anyhow::anyhow!("max_tokens reached")),
        ]]);
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never).await;

        let stream = agent
            .send("tell me a story".to_string(), None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::TokenReceived(t) if t == "partial response")),
            "expected partial text tokens before error"
        );
        assert!(
            events.iter().any(|e| matches!(e, AgentEvent::Error(_))),
            "expected Error event"
        );

        let history = agent.history();
        assert!(
            history.iter().any(|m| m.role == Role::Assistant
                && m.content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::Text(t) if t.contains("partial response")))),
            "partial assistant text must be saved in history"
        );
    }

    #[tokio::test]
    async fn error_message_is_added_to_history_as_user_message() {
        let backend = SequencedBackend::new(vec![vec![Err(anyhow::anyhow!("max_tokens reached"))]]);
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never).await;

        let stream = agent
            .send("hi".to_string(), None)
            .await
            .expect("send should succeed");
        let _events = collect_events(stream).await;

        let history = agent.history();
        let has_error_in_history = history.iter().any(|m| {
            m.role == Role::User
                && m.content.iter().any(
                    |b| matches!(b, ContentBlock::Text(t) if t == "[ERROR] max_tokens reached"),
                )
        });
        assert!(
            has_error_in_history,
            "error message with [ERROR] prefix must be added to history so the agent knows why a stoppage occurred"
        );
    }

    #[tokio::test]
    async fn max_iterations_error_preserves_history() {
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

        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tools(registry)
            .with_tool_config(&tool_config);

        let stream = agent
            .send("run".to_string(), None)
            .await
            .expect("send should succeed");
        let _events = collect_events(stream).await;

        let history = agent.history();
        assert!(
            !history.is_empty(),
            "history must be preserved after max iterations error"
        );
        assert!(
            history.iter().any(|m| m.role == Role::User
                && m.content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::Text(t) if t == "run"))),
            "original user message must be retained"
        );
        let tool_use_count = history
            .iter()
            .flat_map(|m| &m.content)
            .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
            .count();
        assert_eq!(
            tool_use_count, 3,
            "completed tool call iterations must be preserved in history"
        );
        let tool_result_count = history
            .iter()
            .flat_map(|m| &m.content)
            .filter(|b| matches!(b, ContentBlock::ToolResult { .. }))
            .count();
        assert_eq!(
            tool_result_count, 3,
            "completed tool result iterations must be preserved in history"
        );
        assert!(
            history.iter().any(|m| m.role == Role::User
                && m.content.iter().any(
                    |b| matches!(b, ContentBlock::Text(t) if t.contains("[ERROR]") && t.contains("Max tool iterations"))
                )),
            "error message with [ERROR] prefix must be in history"
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

        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
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

    #[tokio::test]
    async fn load_skills_adds_skill_names_to_history() {
        let backend = SequencedBackend::new(vec![]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };

        let mut skills = std::collections::HashMap::new();
        skills.insert(
            "my-skill".to_string(),
            std::path::PathBuf::from("/fake/path"),
        );
        skills.insert(
            "another-skill".to_string(),
            std::path::PathBuf::from("/fake/path2"),
        );
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_skills(&skills);

        let history = agent.history();
        assert_eq!(history.len(), 1);
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

    #[tokio::test]
    async fn load_skills_with_empty_map_adds_nothing_to_history() {
        let backend = SequencedBackend::new(vec![]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_skills(&std::collections::HashMap::new());

        assert!(
            agent.history().is_empty(),
            "empty skills map should not add history entry"
        );
    }

    #[tokio::test]
    async fn load_session_replaces_persisted_history() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let dir_path = dir.keep();

        let session_a = Session::new(None, dir_path.clone())
            .await
            .expect("session a");
        session_a
            .conversation()
            .insert_message(&Message::text(Role::User, "session a message".to_string()))
            .await
            .expect("insert");

        let session_b = Session::new(None, dir_path.clone())
            .await
            .expect("session b");
        session_b
            .conversation()
            .insert_message(&Message::text(Role::User, "session b message".to_string()))
            .await
            .expect("insert");

        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let agent = Agent::new(
            Box::new(SequencedBackend::new(vec![])),
            config,
            Arc::new(TokioMutex::new(session_a)),
        )
        .await;

        assert_eq!(agent.history().len(), 1);
        assert!(
            matches!(&agent.history()[0].content[0], crate::types::ContentBlock::Text(t) if t == "session a message")
        );

        agent.load_session(session_b).await;

        assert_eq!(agent.history().len(), 1);
        assert!(
            matches!(&agent.history()[0].content[0], crate::types::ContentBlock::Text(t) if t == "session b message")
        );
    }

    #[tokio::test]
    async fn load_session_preserves_context_prefix() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let dir_path = dir.keep();

        let session_a = Session::new(None, dir_path.clone())
            .await
            .expect("session a");
        session_a
            .conversation()
            .insert_message(&Message::text(Role::User, "session a message".to_string()))
            .await
            .expect("insert");

        let session_b = Session::new(None, dir_path.clone())
            .await
            .expect("session b");
        session_b
            .conversation()
            .insert_message(&Message::text(Role::User, "session b message".to_string()))
            .await
            .expect("insert");

        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };

        let mut skills = std::collections::HashMap::new();
        let tmp = tempfile::TempDir::new().expect("tmp");
        let skill_path = tmp.path().join("test-skill.md");
        std::fs::write(&skill_path, "---\ndescription: A test skill\n---\nContent").expect("write");
        skills.insert("test-skill".to_string(), skill_path);

        let agent = Agent::new(
            Box::new(SequencedBackend::new(vec![])),
            config,
            Arc::new(TokioMutex::new(session_a)),
        )
        .await
        .with_skills(&skills);

        // history should be: [skill_prefix, session_a_message]
        assert_eq!(agent.history().len(), 2);

        agent.load_session(session_b).await;

        // history should be: [skill_prefix, session_b_message] — prefix preserved
        let history = agent.history();
        assert_eq!(history.len(), 2, "context prefix should be preserved");
        assert!(
            matches!(&history[0].content[0], crate::types::ContentBlock::Text(t) if t.contains("test-skill")),
            "first entry should still be the skills prefix"
        );
        assert!(
            matches!(&history[1].content[0], crate::types::ContentBlock::Text(t) if t == "session b message"),
            "second entry should be new session's message"
        );
    }

    #[tokio::test]
    async fn load_session_updates_session_id() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let dir_path = dir.keep();

        let session_a = Session::new(None, dir_path.clone())
            .await
            .expect("session a");
        let id_a = session_a.id.clone();

        let session_b = Session::new(None, dir_path.clone())
            .await
            .expect("session b");
        let id_b = session_b.id.clone();

        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let agent = Agent::new(
            Box::new(SequencedBackend::new(vec![])),
            config,
            Arc::new(TokioMutex::new(session_a)),
        )
        .await;

        assert_eq!(agent.session_id().await, id_a);
        agent.load_session(session_b).await;
        assert_eq!(agent.session_id().await, id_b);
    }

    #[tokio::test]
    async fn cleanup_empty_session_deletes_db_when_no_messages() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let dir_path = dir.keep();

        let session = Session::new(None, dir_path.clone()).await.expect("session");
        let db_path = dir_path.join(format!("{}.db", session.id));
        assert!(db_path.exists());

        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let agent = Agent::new(
            Box::new(SequencedBackend::new(vec![])),
            config,
            Arc::new(TokioMutex::new(session)),
        )
        .await;

        agent.cleanup_empty_session().await.expect("cleanup");
        assert!(!db_path.exists());
    }

    #[tokio::test]
    async fn cleanup_empty_session_preserves_db_when_has_messages() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let dir_path = dir.keep();

        let session = Session::new(None, dir_path.clone()).await.expect("session");
        session
            .conversation()
            .insert_message(&Message::text(Role::User, "hello".to_string()))
            .await
            .expect("insert");
        let db_path = dir_path.join(format!("{}.db", session.id));

        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let agent = Agent::new(
            Box::new(SequencedBackend::new(vec![])),
            config,
            Arc::new(TokioMutex::new(session)),
        )
        .await;

        agent.cleanup_empty_session().await.expect("cleanup");
        assert!(db_path.exists());
    }

    #[tokio::test]
    async fn spawn_agent_produces_agent_with_role_model_and_tools_config() {
        use crate::backend::BackendSelection;
        use crate::config::ToolsConfig;
        use crate::tools::ToolRegistry;

        let dir = tempfile::TempDir::new().expect("temp dir");
        let session = Arc::new(TokioMutex::new(
            Session::new(None, dir.path().to_path_buf())
                .await
                .expect("session"),
        ));

        let selection = BackendSelection {
            backend: Box::new(SequencedBackend::new(vec![])),
            model: "claude-test-model".to_string(),
        };

        let mut tool_config = ToolsConfig::default();
        tool_config.max_tool_iterations = 7;
        tool_config.confirmation = ConfirmationMode::Never;

        let registry = ToolRegistry::new();

        let agent = super::spawn_agent_with_selection(selection, &tool_config, session, registry)
            .await
            .expect("spawn_agent_with_selection should succeed");

        assert_eq!(
            agent.model(),
            "claude-test-model",
            "agent model must match the role's model"
        );
        assert_eq!(
            agent.max_tool_iterations_for_test(),
            7,
            "agent max_tool_iterations must reflect tool_config"
        );
        assert_eq!(
            agent.confirmation_mode_for_test(),
            &ConfirmationMode::Never,
            "agent confirmation_mode must reflect tool_config"
        );
    }

    // --- Parallel execution tests ---

    struct SleepyTool {
        name: String,
        sleep_ms: u64,
        log: Arc<tokio::sync::Mutex<Vec<(String, std::time::Instant, std::time::Instant)>>>,
        schema: serde_json::Value,
    }

    impl SleepyTool {
        fn new(
            name: &str,
            sleep_ms: u64,
            log: Arc<tokio::sync::Mutex<Vec<(String, std::time::Instant, std::time::Instant)>>>,
        ) -> Self {
            Self {
                name: name.to_string(),
                sleep_ms,
                log,
                schema: serde_json::json!({"type": "object", "properties": {}}),
            }
        }
    }

    #[async_trait]
    impl Tool for SleepyTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "Sleepy tool"
        }
        fn input_schema(&self) -> &serde_json::Value {
            &self.schema
        }
        fn is_write_tool(&self) -> bool {
            false
        }
        async fn execute(&self, _input: serde_json::Value) -> Result<ToolExecResult, ToolError> {
            let start = std::time::Instant::now();
            tokio::time::sleep(tokio::time::Duration::from_millis(self.sleep_ms)).await;
            let end = std::time::Instant::now();
            self.log.lock().await.push((self.name.clone(), start, end));
            Ok(ToolExecResult {
                content: vec![ContentBlock::Text(format!("{}-output", self.name))],
                is_error: false,
            })
        }
    }

    fn two_tool_response(
        id1: &str,
        name1: &str,
        id2: &str,
        name2: &str,
    ) -> Vec<Result<StreamEvent>> {
        vec![
            Ok(StreamEvent::ToolUseStart {
                id: id1.to_string(),
                name: name1.to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDelta {
                chunk: r#"{}"#.to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::ToolUseStart {
                id: id2.to_string(),
                name: name2.to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDelta {
                chunk: r#"{}"#.to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ]
    }

    fn three_tool_response(
        id1: &str,
        name1: &str,
        id2: &str,
        name2: &str,
        id3: &str,
        name3: &str,
    ) -> Vec<Result<StreamEvent>> {
        vec![
            Ok(StreamEvent::ToolUseStart {
                id: id1.to_string(),
                name: name1.to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDelta {
                chunk: r#"{}"#.to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::ToolUseStart {
                id: id2.to_string(),
                name: name2.to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDelta {
                chunk: r#"{}"#.to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::ToolUseStart {
                id: id3.to_string(),
                name: name3.to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDelta {
                chunk: r#"{}"#.to_string(),
                index: None,
            }),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ]
    }

    #[tokio::test]
    async fn two_tool_calls_in_one_response_execute_concurrently() {
        let log = Arc::new(tokio::sync::Mutex::new(vec![]));
        let backend = SequencedBackend::new(vec![
            two_tool_response("t1", "slow", "t2", "fast"),
            text_response("done"),
        ]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(SleepyTool::new("slow", 200, Arc::clone(&log))))
            .expect("register slow");
        registry
            .register(Box::new(SleepyTool::new("fast", 50, Arc::clone(&log))))
            .expect("register fast");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tools(registry)
            .with_tool_config(&tool_config);

        let stream = agent
            .send("run".to_string(), None)
            .await
            .expect("send should succeed");
        let _events = collect_events(stream).await;

        let entries = log.lock().await;
        assert_eq!(entries.len(), 2, "both tools must have executed");

        // Assert true overlap: the fast tool must start before the slow tool ends.
        // This is the actual definition of "ran concurrently" and is immune to
        // wall-clock noise in CI environments.
        let slow_entry = entries.iter().find(|(n, _, _)| n == "slow").expect("slow");
        let fast_entry = entries.iter().find(|(n, _, _)| n == "fast").expect("fast");
        assert!(
            fast_entry.1 < slow_entry.2,
            "fast must start before slow ends: fast_start={:?} slow_end={:?}",
            fast_entry.1,
            slow_entry.2
        );
        assert!(
            slow_entry.1 < fast_entry.2,
            "slow must start before fast ends: slow_start={:?} fast_end={:?}",
            slow_entry.1,
            fast_entry.2
        );
    }

    #[tokio::test]
    async fn tool_results_preserve_input_order_under_concurrent_completion() {
        // Three tools with descending sleep durations — fastest finishes first.
        // The tool_result blocks in the User message must still appear in input order.
        let log = Arc::new(tokio::sync::Mutex::new(vec![]));
        let backend = SequencedBackend::new(vec![
            three_tool_response("t1", "slow", "t2", "medium", "t3", "fast"),
            text_response("done"),
        ]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(SleepyTool::new("slow", 150, Arc::clone(&log))))
            .expect("register slow");
        registry
            .register(Box::new(SleepyTool::new("medium", 75, Arc::clone(&log))))
            .expect("register medium");
        registry
            .register(Box::new(SleepyTool::new("fast", 10, Arc::clone(&log))))
            .expect("register fast");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tools(registry)
            .with_tool_config(&tool_config);

        let stream = agent
            .send("run".to_string(), None)
            .await
            .expect("send should succeed");
        let _events = collect_events(stream).await;

        // Inspect the User message that holds the tool_result blocks.
        let history = agent.history();
        let tool_result_msg = history
            .iter()
            .find(|m| {
                m.role == Role::User
                    && m.content
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
            })
            .expect("must have a tool result message");

        let ids: Vec<&str> = tool_result_msg
            .content
            .iter()
            .filter_map(|b| {
                if let ContentBlock::ToolResult { tool_use_id, .. } = b {
                    Some(tool_use_id.as_str())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(
            ids,
            vec!["t1", "t2", "t3"],
            "tool_result blocks must appear in input order"
        );
    }

    #[tokio::test]
    async fn always_confirmation_with_multiple_calls_prompts_in_order() {
        let backend = SequencedBackend::new(vec![
            three_tool_response("t1", "bash", "t2", "bash", "t3", "bash"),
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
        // Pre-load three approvals.
        confirm_tx
            .unbounded_send(ConfirmationResponse::Approved)
            .expect("send");
        confirm_tx
            .unbounded_send(ConfirmationResponse::Approved)
            .expect("send");
        confirm_tx
            .unbounded_send(ConfirmationResponse::Approved)
            .expect("send");

        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tools(registry)
            .with_tool_config(&tool_config);

        let stream = agent
            .send("run".to_string(), Some(confirm_rx))
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        let confirmations: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolConfirmationRequired { .. }))
            .collect();
        assert_eq!(
            confirmations.len(),
            3,
            "three confirmations must be required"
        );

        // Verify they arrived in input order (t1, t2, t3).
        let ids: Vec<&str> = confirmations
            .iter()
            .filter_map(|e| {
                if let AgentEvent::ToolConfirmationRequired { id, .. } = e {
                    Some(id.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            ids,
            vec!["t1", "t2", "t3"],
            "confirmations must arrive in input order"
        );

        // All three tools must have executed (three ToolResult events).
        let results: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolResult { .. }))
            .collect();
        assert_eq!(
            results.len(),
            3,
            "all three tools must execute after approval"
        );
    }

    #[tokio::test]
    async fn declined_tool_in_batch_does_not_block_others() {
        let backend = SequencedBackend::new(vec![
            three_tool_response("t1", "bash", "t2", "bash", "t3", "bash"),
            text_response("done"),
        ]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(EchoTool::new("bash", "real output")))
            .expect("register");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Always,
            ..Default::default()
        };
        let (confirm_tx, confirm_rx) = mpsc::unbounded::<ConfirmationResponse>();
        confirm_tx
            .unbounded_send(ConfirmationResponse::Approved)
            .expect("send");
        confirm_tx
            .unbounded_send(ConfirmationResponse::Rejected)
            .expect("send");
        confirm_tx
            .unbounded_send(ConfirmationResponse::Approved)
            .expect("send");

        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tools(registry)
            .with_tool_config(&tool_config);

        let stream = agent
            .send("run".to_string(), Some(confirm_rx))
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        let results: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolResult { .. }))
            .collect();
        assert_eq!(results.len(), 3, "three ToolResult events expected");

        // Middle slot (t2, second approval = Rejected) must contain decline message.
        let tool_result_ids: Vec<(&str, &str, bool)> = results
            .iter()
            .filter_map(|e| {
                if let AgentEvent::ToolResult {
                    id,
                    content,
                    is_error,
                    ..
                } = e
                {
                    Some((id.as_str(), content.as_str(), *is_error))
                } else {
                    None
                }
            })
            .collect();

        // Verify the declined tool has the canned message.
        let (_, declined_content, declined_error) = tool_result_ids
            .iter()
            .find(|(id, _, _)| *id == "t2")
            .expect("t2 result must exist");
        assert!(declined_error, "declined tool must be an error");
        assert!(
            declined_content.contains("declined"),
            "declined content must mention 'declined'"
        );

        // The other two must have the real tool output.
        for id in ["t1", "t3"] {
            let (_, content, is_error) = tool_result_ids
                .iter()
                .find(|(i, _, _)| *i == id)
                .unwrap_or_else(|| panic!("{id} result must exist"));
            assert!(!is_error, "{id} must not be an error");
            assert_eq!(*content, "real output", "{id} must have real output");
        }
    }

    #[tokio::test]
    async fn indexed_tool_calls_from_openai_backend_accumulate_correctly() {
        // Regression: OpenAI-compatible backends (zai, ollama) emit tool calls
        // with index fields and a single ToolUseDone at the end. The accumulation
        // loop must track each tool by index, not overwrite with the last one.
        let log = Arc::new(tokio::sync::Mutex::new(vec![]));
        let indexed_response: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::ToolUseStart {
                id: "tool_0".to_string(),
                name: "slow".to_string(),
                index: Some(0),
            }),
            Ok(StreamEvent::ToolUseDelta {
                chunk: r#"{}"#.to_string(),
                index: Some(0),
            }),
            Ok(StreamEvent::ToolUseStart {
                id: "tool_1".to_string(),
                name: "fast".to_string(),
                index: Some(1),
            }),
            Ok(StreamEvent::ToolUseDelta {
                chunk: r#"{}"#.to_string(),
                index: Some(1),
            }),
            // Single ToolUseDone for all indexed tools
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ];
        let backend = SequencedBackend::new(vec![indexed_response, text_response("done")]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(SleepyTool::new("slow", 50, Arc::clone(&log))))
            .expect("register slow");
        registry
            .register(Box::new(SleepyTool::new("fast", 10, Arc::clone(&log))))
            .expect("register fast");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tools(registry)
            .with_tool_config(&tool_config);

        let stream = agent
            .send("run".to_string(), None)
            .await
            .expect("send should succeed");
        let _events = collect_events(stream).await;

        // Both tools must have executed.
        let entries = log.lock().await;
        assert_eq!(entries.len(), 2, "both indexed tools must have executed");

        // Verify history has tool_result blocks in index order (tool_0, tool_1).
        let history = agent.history();
        let tool_result_msg = history
            .iter()
            .find(|m| {
                m.role == Role::User
                    && m.content
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
            })
            .expect("must have a tool result message");

        let ids: Vec<&str> = tool_result_msg
            .content
            .iter()
            .filter_map(|b| {
                if let ContentBlock::ToolResult { tool_use_id, .. } = b {
                    Some(tool_use_id.as_str())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(
            ids,
            vec!["tool_0", "tool_1"],
            "indexed tool_result blocks must appear in index order"
        );
    }
}
