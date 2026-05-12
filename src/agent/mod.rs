use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as TokioMutex;
use tokio_util::sync::CancellationToken;

use crate::backend::LlmBackend;
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
use futures::future::join_all;

mod compact;
mod spawner;

pub use spawner::{
    AgentSpawner, HeadlessOutcome, RegistryBuilder, clamp_confirmation, run_headless, spawn_agent,
    spawn_agent_with_selection,
};

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
    /// Spawner for creating compaction sub-agents. Set after construction via
    /// `with_compaction_spawner()`.
    compaction_spawner: Option<Arc<AgentSpawner>>,
    /// Maximum context window length in tokens. When the API's reported
    /// `input_tokens` exceeds this threshold after a complete assistant turn,
    /// auto-compaction is triggered. A value of 0 disables auto-compaction.
    max_context_window_len: u32,
    /// Whether the previous turn auto-compacted. Prevents consecutive compaction
    /// loops: if the previous turn already auto-compacted and the threshold is
    /// still exceeded, a warning is emitted instead.
    last_auto_compacted: Arc<AtomicBool>,
    /// Number of recent conversation turns to retain after compaction.
    /// When 0 (default), compaction replaces the entire history with a summary.
    /// When > 0, the last N turns are kept after the summary, with tool-use
    /// blocks stripped from retained messages.
    num_retained_turns: u32,
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
            compaction_spawner: None,
            max_context_window_len: 0,
            last_auto_compacted: Arc::new(AtomicBool::new(false)),
            num_retained_turns: 0,
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

    pub fn with_compaction_config(mut self, config: &crate::config::CompactionConfig) -> Self {
        self.max_context_window_len = config.max_context_window_len;
        self.num_retained_turns = config.num_retained_turns;
        self
    }

    pub fn with_thinking(self, thinking: Option<crate::types::ThinkingConfig>) -> Self {
        self.config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .thinking = thinking;
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

    pub fn with_compaction_spawner(self, spawner: Arc<AgentSpawner>) -> Self {
        Self {
            compaction_spawner: Some(spawner),
            ..self
        }
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
    pub fn max_context_window_len_for_test(&self) -> u32 {
        self.max_context_window_len
    }

    #[cfg(test)]
    pub fn num_retained_turns_for_test(&self) -> u32 {
        self.num_retained_turns
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

    /// Return a snapshot of all tasks in the current session.
    ///
    /// # Lock note
    /// The session `Mutex` is held for the duration of the DB query. `TaskRepo`
    /// borrows `&Session`, so the guard cannot be dropped before `list()` returns.
    /// This is acceptable here because `/tasks` is only dispatched from
    /// `AppState::Input` (no concurrent streaming), and `list()` is a fast
    /// read-only scan with no user-visible latency impact.
    pub async fn tasks_snapshot(&self) -> anyhow::Result<Vec<crate::session::TaskRecord>> {
        self.session.lock().await.tasks().list(None).await
    }

    /// Checkpoint the WAL of the current session into the main DB file.
    ///
    /// Errors are returned but callers are expected to log and swallow them so
    /// the frontend's `Result` is preserved.
    pub async fn checkpoint_session(&self) -> Result<()> {
        self.session.lock().await.checkpoint().await
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
        self.last_auto_compacted.store(false, Ordering::SeqCst);
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
        cancel_token: Option<CancellationToken>,
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
        let max_context_window_len = self.max_context_window_len;
        let last_auto_compacted = Arc::clone(&self.last_auto_compacted);

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
                // Check cancellation at the top of each iteration.
                if let Some(ref token) = cancel_token
                    && token.is_cancelled()
                {
                    persist_partial_and_interrupt("", &history_arc, &session, &event_tx).await;
                    break;
                }

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
                let mut thinking_accumulated = String::new();
                let mut thinking_signature = String::new();
                let mut tool_calls: Vec<PendingToolCall> = vec![];
                let mut current_tool: Option<PendingToolCall> = None;
                let mut peak_input_tokens: u32 = 0;
                let mut stream = backend_stream;

                loop {
                    let next_event = if let Some(ref token) = cancel_token {
                        tokio::select! {
                            biased;
                            _ = token.cancelled() => None,
                            item = stream.next() => item,
                        }
                    } else {
                        stream.next().await
                    };

                    match next_event {
                        None if cancel_token.as_ref().is_some_and(|t| t.is_cancelled()) => {
                            let _ = event_tx.unbounded_send(AgentEvent::Warn(
                                "stream interrupted by cancellation; any in-flight tool executions will be orphaned".to_string(),
                            ));
                            persist_partial_and_interrupt(
                                &text_accumulated,
                                &history_arc,
                                &session,
                                &event_tx,
                            )
                            .await;
                            if !thinking_accumulated.is_empty() {
                                let thinking_msg = Message {
                                    role: Role::Assistant,
                                    content: vec![ContentBlock::Thinking {
                                        text: thinking_accumulated.clone(),
                                        signature: thinking_signature.clone(),
                                    }],
                                };
                                lock(&history_arc).push(thinking_msg.clone());
                                let _ = session
                                    .lock()
                                    .await
                                    .conversation()
                                    .insert_message(&thinking_msg)
                                    .await;
                            }
                            break 'outer;
                        }
                        None => break,
                        Some(Ok(StreamEvent::TextDelta(text))) => {
                            text_accumulated.push_str(&text);
                            let _ = event_tx.unbounded_send(AgentEvent::TokenReceived(text));
                        }
                        Some(Ok(StreamEvent::ThinkingDelta(text))) => {
                            thinking_accumulated.push_str(&text);
                            let _ = event_tx.unbounded_send(AgentEvent::ThinkingReceived(text));
                        }
                        Some(Ok(StreamEvent::ThinkingSignature(sig))) => {
                            thinking_signature = sig;
                        }
                        Some(Ok(StreamEvent::ToolUseStart { id, name })) => {
                            current_tool = Some(PendingToolCall {
                                id,
                                name,
                                input_json: String::new(),
                            });
                        }
                        Some(Ok(StreamEvent::ToolUseDelta(chunk))) => {
                            if let Some(ref mut t) = current_tool {
                                t.input_json.push_str(&chunk);
                            }
                        }
                        Some(Ok(StreamEvent::ToolUseDone)) => {
                            if let Some(t) = current_tool.take() {
                                tool_calls.push(t);
                            }
                        }
                        Some(Ok(StreamEvent::Usage {
                            input_tokens,
                            output_tokens,
                            stop_reason,
                        })) => {
                            peak_input_tokens = peak_input_tokens.max(input_tokens);
                            let _ = event_tx.unbounded_send(AgentEvent::Usage {
                                input_tokens,
                                output_tokens,
                                stop_reason,
                            });
                        }
                        Some(Ok(StreamEvent::Done)) => break,
                        Some(Err(e)) => {
                            if !text_accumulated.is_empty() {
                                persist_partial(&text_accumulated, &history_arc, &session).await;
                            }
                            if !thinking_accumulated.is_empty() {
                                let thinking_msg = Message {
                                    role: Role::Assistant,
                                    content: vec![ContentBlock::Thinking {
                                        text: thinking_accumulated.clone(),
                                        signature: thinking_signature.clone(),
                                    }],
                                };
                                lock(&history_arc).push(thinking_msg.clone());
                                let _ = session
                                    .lock()
                                    .await
                                    .conversation()
                                    .insert_message(&thinking_msg)
                                    .await;
                            }
                            record_error(&e.to_string(), &history_arc, &session, &event_tx).await;
                            break 'outer;
                        }
                    }
                }

                if tool_calls.is_empty() {
                    let mut content = vec![];
                    if !thinking_accumulated.is_empty() {
                        content.push(ContentBlock::Thinking {
                            text: thinking_accumulated.clone(),
                            signature: thinking_signature.clone(),
                        });
                    }
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

                    // Auto-compact check: runs after every completed iteration,
                    // whether the response included tool calls or not.
                    compact::check_auto_compact(
                        peak_input_tokens,
                        max_context_window_len,
                        &last_auto_compacted,
                        &event_tx,
                    );

                    break;
                }

                // Check cancellation before executing tool calls.
                if let Some(ref token) = cancel_token
                    && token.is_cancelled()
                {
                    let _ = event_tx.unbounded_send(AgentEvent::Warn(
                        "cancellation requested before tool execution; tool calls will be skipped"
                            .to_string(),
                    ));
                    persist_partial_and_interrupt(
                        &text_accumulated,
                        &history_arc,
                        &session,
                        &event_tx,
                    )
                    .await;
                    break;
                }

                let text_for_cancel = text_accumulated.clone();
                let thinking_for_cancel = thinking_accumulated.clone();
                let signature_for_cancel = thinking_signature.clone();
                let (assistant_content, tool_result_blocks) = execute_tool_calls(
                    tool_calls,
                    text_accumulated,
                    thinking_accumulated,
                    thinking_signature.clone(),
                    &tools,
                    &confirmation_mode,
                    &mut confirmation_rx,
                    &event_tx,
                    cancel_token.clone(),
                )
                .await;

                // If cancellation was triggered during tool execution, persist what we have and stop.
                if let Some(ref token) = cancel_token
                    && token.is_cancelled()
                {
                    persist_partial_and_interrupt(
                        &text_for_cancel,
                        &history_arc,
                        &session,
                        &event_tx,
                    )
                    .await;
                    // Also persist thinking if any
                    if !thinking_for_cancel.is_empty() {
                        let thinking_msg = Message {
                            role: Role::Assistant,
                            content: vec![ContentBlock::Thinking {
                                text: thinking_for_cancel,
                                signature: signature_for_cancel,
                            }],
                        };
                        lock(&history_arc).push(thinking_msg.clone());
                        let _ = session
                            .lock()
                            .await
                            .conversation()
                            .insert_message(&thinking_msg)
                            .await;
                    }
                    break;
                }

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

                // Auto-compact check: runs after every completed iteration,
                // whether the response included tool calls or not.
                compact::check_auto_compact(
                    peak_input_tokens,
                    max_context_window_len,
                    &last_auto_compacted,
                    &event_tx,
                );
            }
        });

        Ok(Box::pin(event_rx))
    }
}

async fn persist_partial(
    text: &str,
    history: &Arc<Mutex<Vec<Message>>>,
    session: &Arc<TokioMutex<Session>>,
) {
    if text.is_empty() {
        return;
    }
    let msg = Message {
        role: Role::Assistant,
        content: vec![ContentBlock::Text(text.to_string())],
    };
    lock(history).push(msg.clone());
    let _ = session
        .lock()
        .await
        .conversation()
        .insert_message(&msg)
        .await;
}

async fn persist_partial_and_interrupt(
    text: &str,
    history: &Arc<Mutex<Vec<Message>>>,
    session: &Arc<TokioMutex<Session>>,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
) {
    persist_partial(text, history, session).await;
    let _ = event_tx.unbounded_send(AgentEvent::Interrupted {
        partial_text: text.to_string(),
    });
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

#[allow(clippy::too_many_arguments)]
async fn execute_tool_calls(
    tool_calls: Vec<PendingToolCall>,
    text_prefix: String,
    thinking_prefix: String,
    thinking_signature: String,
    tools: &ToolRegistry,
    confirmation_mode: &ConfirmationMode,
    confirmation_rx: &mut Option<mpsc::UnboundedReceiver<ConfirmationResponse>>,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    cancel_token: Option<CancellationToken>,
) -> (Vec<ContentBlock>, Vec<ContentBlock>) {
    let mut assistant_content: Vec<ContentBlock> = vec![];
    if !thinking_prefix.is_empty() {
        assistant_content.push(ContentBlock::Thinking {
            text: thinking_prefix,
            signature: thinking_signature,
        });
    }
    if !text_prefix.is_empty() {
        assistant_content.push(ContentBlock::Text(text_prefix));
    }

    enum ToolDecision {
        ParseError(String),
        Declined,
        Approved,
    }

    struct Resolved {
        id: String,
        name: String,
        input: serde_json::Value,
        decision: ToolDecision,
        index: usize,
    }

    // Parse inputs, emit ToolUseReceived, then gather confirmations sequentially.
    let mut resolved: Vec<Resolved> = Vec::with_capacity(tool_calls.len());
    for (i, call) in tool_calls.into_iter().enumerate() {
        let index = i + 1;
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

        assistant_content.push(ContentBlock::ToolUse {
            id: call.id.clone(),
            name: call.name.clone(),
            input: input.clone(),
        });
        let _ = event_tx.unbounded_send(AgentEvent::ToolUseReceived {
            id: call.id.clone(),
            name: call.name.clone(),
            input: input.clone(),
            index,
        });

        let decision = if let Some(err) = parse_error {
            ToolDecision::ParseError(err)
        } else {
            let needs_confirmation = match confirmation_mode {
                ConfirmationMode::Always => true,
                ConfirmationMode::Never => false,
                ConfirmationMode::WriteOnly => {
                    tools.lookup(&call.name).is_ok_and(|t| t.is_write_tool())
                }
            };
            if needs_confirmation {
                // If already cancelled, skip confirmation and decline immediately.
                if cancel_token.as_ref().is_some_and(|t| t.is_cancelled()) {
                    ToolDecision::Declined
                } else {
                    let _ = event_tx.unbounded_send(AgentEvent::ToolConfirmationRequired {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        input: input.clone(),
                        index,
                    });
                    let approved = if let Some(rx) = confirmation_rx.as_mut() {
                        // Race the confirmation against the cancel token so that Esc
                        // during a multi-tool confirmation sequence declines all
                        // remaining tools without requiring another keypress.
                        if let Some(ref token) = cancel_token {
                            tokio::select! {
                                biased;
                                _ = token.cancelled() => false,
                                response = rx.next() => {
                                    matches!(response, Some(ConfirmationResponse::Approved))
                                }
                            }
                        } else {
                            matches!(rx.next().await, Some(ConfirmationResponse::Approved))
                        }
                    } else {
                        false
                    };
                    if approved {
                        ToolDecision::Approved
                    } else {
                        ToolDecision::Declined
                    }
                }
            } else {
                ToolDecision::Approved
            }
        };

        resolved.push(Resolved {
            id: call.id,
            name: call.name,
            input,
            decision,
            index,
        });
    }

    // Execute approved tools concurrently; produce results in input order.
    // Each Approved execution races against the cancel token so that pressing
    // Esc mid-tool drops the in-flight future (which kills bash subprocesses
    // via tokio's kill_on_drop) and returns a cancelled-result block.
    let futures: Vec<_> = resolved
        .iter()
        .map(|r| {
            let cancel_token = cancel_token.clone();
            async move {
                match &r.decision {
                    ToolDecision::ParseError(err) => (
                        r.index,
                        r.id.clone(),
                        r.name.clone(),
                        err.clone(),
                        true,
                        vec![],
                    ),
                    ToolDecision::Declined => (
                        r.index,
                        r.id.clone(),
                        r.name.clone(),
                        "User declined to execute this tool.".to_string(),
                        true,
                        vec![],
                    ),
                    ToolDecision::Approved => match tools.lookup(&r.name) {
                        Ok(tool) => {
                            let exec = tool.execute(r.input.clone());
                            let outcome = if let Some(ref token) = cancel_token {
                                tokio::select! {
                                    biased;
                                    _ = token.cancelled() => None,
                                    result = exec => Some(result),
                                }
                            } else {
                                Some(exec.await)
                            };
                            match outcome {
                                None => (
                                    r.index,
                                    r.id.clone(),
                                    r.name.clone(),
                                    "Tool cancelled by user.".to_string(),
                                    true,
                                    vec![],
                                ),
                                Some(Ok(result)) => {
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
                                    (
                                        r.index,
                                        r.id.clone(),
                                        r.name.clone(),
                                        content,
                                        result.is_error,
                                        result.agent_events,
                                    )
                                }
                                Some(Err(e)) => (
                                    r.index,
                                    r.id.clone(),
                                    r.name.clone(),
                                    e.to_string(),
                                    true,
                                    vec![],
                                ),
                            }
                        }
                        Err(e) => (
                            r.index,
                            r.id.clone(),
                            r.name.clone(),
                            e.to_string(),
                            true,
                            vec![],
                        ),
                    },
                }
            }
        })
        .collect();

    let results = join_all(futures).await;

    let mut tool_result_blocks: Vec<ContentBlock> = Vec::with_capacity(results.len());
    for (index, id, name, content, is_error, agent_events) in results {
        for extra_event in agent_events {
            let _ = event_tx.unbounded_send(extra_event);
        }
        let _ = event_tx.unbounded_send(AgentEvent::ToolResult {
            name: name.clone(),
            content: content.clone(),
            is_error,
            index,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CompactionConfig;
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

    fn thinking_then_text_response(thinking: &str, text: &str) -> Vec<Result<StreamEvent>> {
        vec![
            Ok(StreamEvent::ThinkingDelta(thinking.to_string())),
            Ok(StreamEvent::ThinkingSignature("sig_abc123".to_string())),
            Ok(StreamEvent::TextDelta(text.to_string())),
            Ok(StreamEvent::Done),
        ]
    }

    fn thinking_only_response(thinking: &str) -> Vec<Result<StreamEvent>> {
        vec![
            Ok(StreamEvent::ThinkingDelta(thinking.to_string())),
            Ok(StreamEvent::ThinkingSignature("sig_def456".to_string())),
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
                agent_events: vec![],
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
            thinking: None,
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
            .send("hi".to_string(), None, None)
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
            .send("run ls".to_string(), None, None)
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
            thinking: None,
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
            .send("run".to_string(), Some(confirm_rx), None)
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
            thinking: None,
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
            .send("run".to_string(), Some(confirm_rx), None)
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
            thinking: None,
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
            .send("run".to_string(), Some(confirm_rx), None)
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
            thinking: None,
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
            .send("run".to_string(), None, None)
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
            thinking: None,
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
            .send("run".to_string(), None, None)
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
        )
        .await;

        let stream = agent
            .send("run".to_string(), None, None)
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
            .send("list".to_string(), None, None)
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
            .send("run".to_string(), None, None)
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
            .send("hello".to_string(), None, None)
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
            .send("hello".to_string(), None, None)
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
            .send("tell me a story".to_string(), None, None)
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
            .send("hi".to_string(), None, None)
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
            thinking: None,
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
            .send("run".to_string(), None, None)
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
            thinking: None,
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
            .send("write".to_string(), Some(confirm_rx), None)
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
            thinking: None,
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
            thinking: None,
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
            thinking: None,
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
            thinking: None,
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
            thinking: None,
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
            thinking: None,
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
            thinking: None,
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

        let tool_config = ToolsConfig {
            max_tool_iterations: 7,
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };

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

    struct SleepTool {
        duration_ms: u64,
    }

    #[async_trait]
    impl Tool for SleepTool {
        fn name(&self) -> &str {
            "sleep"
        }
        fn description(&self) -> &str {
            "Sleep for duration_ms milliseconds"
        }
        fn input_schema(&self) -> &serde_json::Value {
            static SCHEMA: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
            SCHEMA.get_or_init(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "duration_ms": {"type": "number"}
                    }
                })
            })
        }
        fn is_write_tool(&self) -> bool {
            false
        }
        async fn execute(&self, input: serde_json::Value) -> Result<ToolExecResult, ToolError> {
            let ms = input["duration_ms"].as_u64().unwrap_or(self.duration_ms);
            tokio::time::sleep(tokio::time::Duration::from_millis(ms)).await;
            Ok(ToolExecResult {
                content: vec![ContentBlock::Text(format!("slept {}ms", ms))],
                is_error: false,
                agent_events: vec![],
            })
        }
    }

    fn multi_sleep_response() -> Vec<Result<StreamEvent>> {
        vec![
            Ok(StreamEvent::ToolUseStart {
                id: "s1".to_string(),
                name: "sleep".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta(
                r#"{"duration_ms":300}"#.to_string(),
            )),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::ToolUseStart {
                id: "s2".to_string(),
                name: "sleep".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta(
                r#"{"duration_ms":100}"#.to_string(),
            )),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::ToolUseStart {
                id: "s3".to_string(),
                name: "sleep".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta(
                r#"{"duration_ms":200}"#.to_string(),
            )),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ]
    }

    async fn agent_with_sleep_tool() -> Agent {
        let backend = SequencedBackend::new(vec![multi_sleep_response(), text_response("done")]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(SleepTool { duration_ms: 300 }))
            .expect("register sleep");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tools(registry)
            .with_tool_config(&tool_config)
    }

    #[tokio::test]
    async fn parallel_tool_calls_complete_faster_than_sequential_bound() {
        let agent = agent_with_sleep_tool().await;
        let start = std::time::Instant::now();
        let stream = agent
            .send("sleep".to_string(), None, None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;
        let elapsed = start.elapsed();

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "expected ResponseComplete"
        );
        assert!(
            elapsed.as_millis() < 600,
            "three concurrent sleeps (300+100+200ms) should finish well under 600ms sequential bound; took {}ms",
            elapsed.as_millis()
        );
    }

    #[tokio::test]
    async fn parallel_tool_calls_preserve_input_order_in_results() {
        let agent = agent_with_sleep_tool().await;
        let stream = agent
            .send("sleep".to_string(), None, None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        let result_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolResult { .. }))
            .collect();
        assert_eq!(result_events.len(), 3, "expected 3 ToolResult events");

        // Results must be in input order (s1=300ms, s2=100ms, s3=200ms)
        if let AgentEvent::ToolResult { content, index, .. } = &result_events[0] {
            assert_eq!(*index, 1, "first result should have index 1");
            assert!(content.contains("300"), "first result should be s1 (300ms)");
        }
        if let AgentEvent::ToolResult { content, index, .. } = &result_events[1] {
            assert_eq!(*index, 2, "second result should have index 2");
            assert!(
                content.contains("100"),
                "second result should be s2 (100ms)"
            );
        }
        if let AgentEvent::ToolResult { content, index, .. } = &result_events[2] {
            assert_eq!(*index, 3, "third result should have index 3");
            assert!(content.contains("200"), "third result should be s3 (200ms)");
        }

        // The history tool_result_blocks must also be in input order.
        let history = agent.history();
        let result_blocks: Vec<_> = history
            .iter()
            .flat_map(|m| &m.content)
            .filter(|b| matches!(b, ContentBlock::ToolResult { .. }))
            .collect();
        assert_eq!(result_blocks.len(), 3);
        if let ContentBlock::ToolResult { tool_use_id, .. } = result_blocks[0] {
            assert_eq!(tool_use_id, "s1", "first block must correspond to s1");
        }
        if let ContentBlock::ToolResult { tool_use_id, .. } = result_blocks[1] {
            assert_eq!(tool_use_id, "s2", "second block must correspond to s2");
        }
        if let ContentBlock::ToolResult { tool_use_id, .. } = result_blocks[2] {
            assert_eq!(tool_use_id, "s3", "third block must correspond to s3");
        }
    }

    #[tokio::test]
    async fn parallel_tool_calls_failure_isolation_all_results_emitted() {
        struct FailOnSecond {
            call_count: Arc<tokio::sync::Mutex<u32>>,
        }

        #[async_trait]
        impl Tool for FailOnSecond {
            fn name(&self) -> &str {
                "maybe_fail"
            }
            fn description(&self) -> &str {
                "Fails on second call"
            }
            fn input_schema(&self) -> &serde_json::Value {
                static SCHEMA: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
                SCHEMA.get_or_init(|| serde_json::json!({"type": "object", "properties": {}}))
            }
            fn is_write_tool(&self) -> bool {
                false
            }
            async fn execute(
                &self,
                _input: serde_json::Value,
            ) -> Result<ToolExecResult, ToolError> {
                let mut count = self.call_count.lock().await;
                *count += 1;
                let n = *count;
                drop(count);
                if n == 2 {
                    Err(ToolError::Execution {
                        tool_name: "maybe_fail".to_string(),
                        message: "intentional failure".to_string(),
                    })
                } else {
                    Ok(ToolExecResult {
                        content: vec![ContentBlock::Text(format!("ok call {n}"))],
                        is_error: false,
                        agent_events: vec![],
                    })
                }
            }
        }

        let multi_call_response: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::ToolUseStart {
                id: "t1".to_string(),
                name: "maybe_fail".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta("{}".to_string())),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::ToolUseStart {
                id: "t2".to_string(),
                name: "maybe_fail".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta("{}".to_string())),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::ToolUseStart {
                id: "t3".to_string(),
                name: "maybe_fail".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta("{}".to_string())),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ];

        let backend = SequencedBackend::new(vec![multi_call_response, text_response("done")]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(FailOnSecond {
                call_count: Arc::new(tokio::sync::Mutex::new(0)),
            }))
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
            .send("run".to_string(), None, None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        let results: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolResult { .. }))
            .collect();
        assert_eq!(
            results.len(),
            3,
            "all three ToolResult events must be emitted even when one fails"
        );

        let error_results: Vec<_> = results
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolResult { is_error, .. } if *is_error))
            .collect();
        assert_eq!(error_results.len(), 1, "exactly one error result expected");

        let ok_results: Vec<_> = results
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolResult { is_error, .. } if !is_error))
            .collect();
        assert_eq!(ok_results.len(), 2, "two successful results expected");
    }

    #[tokio::test]
    async fn index_propagation_tool_use_received_and_result_carry_matching_indices() {
        let multi_tool_response: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::ToolUseStart {
                id: "t1".to_string(),
                name: "bash".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta("{}".to_string())),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::ToolUseStart {
                id: "t2".to_string(),
                name: "bash".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta("{}".to_string())),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::ToolUseStart {
                id: "t3".to_string(),
                name: "bash".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta("{}".to_string())),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ];

        let backend = SequencedBackend::new(vec![multi_tool_response, text_response("done")]);
        let agent = agent_with_mode(
            backend,
            Some(Box::new(EchoTool::new("bash", "output"))),
            ConfirmationMode::Never,
        )
        .await;

        let stream = agent
            .send("run".to_string(), None, None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        let use_indices: Vec<usize> = events
            .iter()
            .filter_map(|e| {
                if let AgentEvent::ToolUseReceived { index, .. } = e {
                    Some(*index)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            use_indices,
            vec![1, 2, 3],
            "ToolUseReceived indices must be 1,2,3"
        );

        let result_indices: Vec<usize> = events
            .iter()
            .filter_map(|e| {
                if let AgentEvent::ToolResult { index, .. } = e {
                    Some(*index)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            result_indices,
            vec![1, 2, 3],
            "ToolResult indices must be 1,2,3"
        );
    }

    #[tokio::test]
    async fn confirmation_then_parallel_all_confirmations_before_execution() {
        // With ConfirmationMode::Always, all N confirmations are sent before any
        // tool starts executing. A denied tool produces the "declined" result without
        // blocking the others.
        let multi_tool_response: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::ToolUseStart {
                id: "t1".to_string(),
                name: "bash".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta("{}".to_string())),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::ToolUseStart {
                id: "t2".to_string(),
                name: "bash".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta("{}".to_string())),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ];

        let backend = SequencedBackend::new(vec![multi_tool_response, text_response("done")]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(EchoTool::new("bash", "echo output")))
            .expect("register");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Always,
            ..Default::default()
        };

        let (confirm_tx, confirm_rx) = mpsc::unbounded::<ConfirmationResponse>();
        // Approve first, reject second
        confirm_tx
            .unbounded_send(ConfirmationResponse::Approved)
            .expect("send approval");
        confirm_tx
            .unbounded_send(ConfirmationResponse::Rejected)
            .expect("send rejection");

        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tools(registry)
            .with_tool_config(&tool_config);

        let stream = agent
            .send("run".to_string(), Some(confirm_rx), None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        // Both ToolConfirmationRequired events must appear before any ToolResult.
        let mut saw_confirmation = false;
        let mut all_confirmations_before_first_result = true;
        let mut first_result_seen = false;
        for event in &events {
            match event {
                AgentEvent::ToolConfirmationRequired { .. } => {
                    saw_confirmation = true;
                    if first_result_seen {
                        all_confirmations_before_first_result = false;
                    }
                }
                AgentEvent::ToolResult { .. } => {
                    first_result_seen = true;
                }
                _ => {}
            }
        }
        assert!(
            saw_confirmation,
            "expected at least one ToolConfirmationRequired"
        );
        assert!(
            all_confirmations_before_first_result,
            "all confirmations must occur before the first ToolResult"
        );

        let results: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolResult { .. }))
            .collect();
        assert_eq!(results.len(), 2, "two ToolResult events expected");

        // First approved → success, second rejected → error
        if let AgentEvent::ToolResult {
            is_error, index, ..
        } = results[0]
        {
            assert!(!is_error, "approved tool should succeed");
            assert_eq!(*index, 1);
        }
        if let AgentEvent::ToolResult {
            is_error, index, ..
        } = results[1]
        {
            assert!(is_error, "rejected tool should produce error result");
            assert_eq!(*index, 2);
        }
    }

    // ── clamp_confirmation ────────────────────────────────────────────────

    #[test]
    fn clamp_confirmation_picks_stricter() {
        use super::clamp_confirmation;
        use crate::config::ConfirmationMode::{Always, Never, WriteOnly};

        // parent=Always: always wins
        assert_eq!(clamp_confirmation(&Always, Some(&Always)), Always);
        assert_eq!(clamp_confirmation(&Always, Some(&WriteOnly)), Always);
        assert_eq!(clamp_confirmation(&Always, Some(&Never)), Always);

        // parent=WriteOnly
        assert_eq!(clamp_confirmation(&WriteOnly, Some(&Always)), Always);
        assert_eq!(clamp_confirmation(&WriteOnly, Some(&WriteOnly)), WriteOnly);
        assert_eq!(clamp_confirmation(&WriteOnly, Some(&Never)), WriteOnly);

        // parent=Never: requested takes precedence unless it's also Never
        assert_eq!(clamp_confirmation(&Never, Some(&Always)), Always);
        assert_eq!(clamp_confirmation(&Never, Some(&WriteOnly)), WriteOnly);
        assert_eq!(clamp_confirmation(&Never, Some(&Never)), Never);

        // None requested → fall back to parent
        assert_eq!(clamp_confirmation(&WriteOnly, None), WriteOnly);
        assert_eq!(clamp_confirmation(&Always, None), Always);
        assert_eq!(clamp_confirmation(&Never, None), Never);
    }

    // ── run_headless ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn run_headless_collects_text_and_usage() {
        use super::run_headless;

        let backend = SequencedBackend::new(vec![vec![
            Ok(StreamEvent::TextDelta("hello".to_string())),
            Ok(StreamEvent::TextDelta(" world".to_string())),
            Ok(StreamEvent::Usage {
                input_tokens: 10,
                output_tokens: 5,
                stop_reason: "end_turn".to_string(),
            }),
            Ok(StreamEvent::Done),
        ]]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await).await;
        let outcome = run_headless(&agent, "hi".to_string()).await;

        assert!(!outcome.is_error);
        assert_eq!(outcome.text, "hello world");
        assert_eq!(outcome.input_tokens, 10);
        assert_eq!(outcome.output_tokens, 5);
    }

    #[tokio::test]
    async fn run_headless_auto_rejects_confirmations() {
        use super::run_headless;

        // In Always mode the agent emits ToolConfirmationRequired before checking
        // the rx. run_headless has no rx and no user, so it treats the event as a
        // hard error: is_error = true with a descriptive error_message, and stops.
        let backend = SequencedBackend::new(vec![
            tool_call_response("t1", "write_file", r#"{}"#),
            text_response("done"),
        ]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(EchoTool::write_tool("write_file", "written")))
            .expect("register");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Always,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tools(registry)
            .with_tool_config(&tool_config);

        let outcome = run_headless(&agent, "write".to_string()).await;

        // ToolConfirmationRequired received with no human present → hard error.
        assert!(
            outcome.is_error,
            "confirmation required with no human present must set is_error"
        );
        let msg = outcome
            .error_message
            .expect("error_message must be set when is_error is true");
        assert!(
            msg.contains("write_file"),
            "error message should name the offending tool; got: {msg}"
        );
        assert!(
            msg.contains("confirmation"),
            "error message should mention confirmation; got: {msg}"
        );
    }

    #[tokio::test]
    async fn run_headless_backend_error_sets_is_error() {
        use super::run_headless;

        let backend = SequencedBackend::new(vec![vec![Err(anyhow::anyhow!("connection lost"))]]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await).await;
        let outcome = run_headless(&agent, "fail".to_string()).await;

        assert!(outcome.is_error);
        let msg = outcome.error_message.expect("should have error message");
        assert!(msg.contains("connection lost"));
    }

    // ── CancellationToken tests ───────────────────────────────────────────

    /// A backend that emits a configurable set of text deltas, then blocks
    /// indefinitely until the stream is polled after cancellation.
    struct CancellableBackend {
        initial_tokens: Vec<String>,
        token: CancellationToken,
    }

    impl CancellableBackend {
        fn new(initial_tokens: Vec<&str>, token: CancellationToken) -> Self {
            Self {
                initial_tokens: initial_tokens.into_iter().map(String::from).collect(),
                token,
            }
        }
    }

    #[async_trait]
    impl LlmBackend for CancellableBackend {
        async fn send_message(
            &self,
            _: &[Message],
            _: &RequestConfig,
        ) -> Result<BoxStream<Result<StreamEvent>>> {
            let tokens: Vec<Result<StreamEvent>> = self
                .initial_tokens
                .iter()
                .map(|t| Ok(StreamEvent::TextDelta(t.clone())))
                .collect();
            let token = self.token.clone();

            // Emit each token, then park until cancelled (simulating a slow stream).
            let s = stream::unfold(
                (tokens.into_iter(), token, false),
                |(mut iter, token, done)| async move {
                    if done {
                        return None;
                    }
                    if let Some(item) = iter.next() {
                        return Some((item, (iter, token, false)));
                    }
                    // No more tokens — block until cancelled, then end.
                    token.cancelled().await;
                    None
                },
            );
            Ok(Box::pin(s))
        }
    }

    #[tokio::test]
    async fn cancel_during_streaming_persists_partial_text() {
        let cancel = CancellationToken::new();
        let backend = CancellableBackend::new(vec!["hello", " world"], cancel.clone());
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never).await;

        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            cancel_clone.cancel();
        });

        let stream = agent
            .send("tell me a story".to_string(), None, Some(cancel))
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        let interrupted = events
            .iter()
            .find(|e| matches!(e, AgentEvent::Interrupted { .. }));
        assert!(
            interrupted.is_some(),
            "expected Interrupted event; got: {events:?}"
        );
        if let Some(AgentEvent::Interrupted { partial_text }) = interrupted {
            assert_eq!(partial_text, "hello world", "partial text mismatch");
        }

        // History should contain the partial assistant message.
        let history = lock(&agent.history).clone();
        assert!(
            history.iter().any(|m| {
                m.role == Role::Assistant
                    && m.content
                        .iter()
                        .any(|b| matches!(b, ContentBlock::Text(t) if t == "hello world"))
            }),
            "partial assistant message should be in history; history: {history:?}"
        );
    }

    #[tokio::test]
    async fn cancel_before_first_chunk_emits_empty_interrupted() {
        let cancel = CancellationToken::new();
        // Cancel immediately before the stream starts.
        cancel.cancel();

        let backend = SequencedBackend::new(vec![text_response("should not appear")]);
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never).await;

        let stream = agent
            .send("hi".to_string(), None, Some(cancel))
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        let interrupted = events
            .iter()
            .find(|e| matches!(e, AgentEvent::Interrupted { .. }));
        assert!(
            interrupted.is_some(),
            "expected Interrupted event; got: {events:?}"
        );
        if let Some(AgentEvent::Interrupted { partial_text }) = interrupted {
            assert!(
                partial_text.is_empty(),
                "partial text should be empty for pre-cancel; got: '{partial_text}'"
            );
        }
    }

    #[tokio::test]
    async fn cancel_token_none_preserves_legacy_behavior() {
        let backend = SequencedBackend::new(vec![text_response("hello")]);
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never).await;

        let stream = agent
            .send("hi".to_string(), None, None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "expected ResponseComplete without token; got: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::Interrupted { .. })),
            "must not have Interrupted without cancel; got: {events:?}"
        );
    }

    #[tokio::test]
    async fn cancel_between_tool_iterations_stops_before_next_backend_call() {
        let cancel = CancellationToken::new();
        let cancel_for_agent = cancel.clone();

        // First response: a tool call followed immediately by Done.
        // Second response: blocks until cancel fires, then emits nothing.
        // This simulates the agent completing one iteration then being
        // cancelled before the second backend response arrives.
        let second_cancel = cancel.clone();
        let backend = {
            struct TwoPhaseBackend {
                first_done: std::sync::atomic::AtomicBool,
                cancel: CancellationToken,
            }

            #[async_trait]
            impl LlmBackend for TwoPhaseBackend {
                async fn send_message(
                    &self,
                    _: &[Message],
                    _: &RequestConfig,
                ) -> Result<BoxStream<Result<StreamEvent>>> {
                    let already_called = self
                        .first_done
                        .swap(true, std::sync::atomic::Ordering::SeqCst);
                    if !already_called {
                        // First call: return a tool use.
                        Ok(Box::pin(futures::stream::iter(tool_call_response(
                            "t1", "bash", r#"{}"#,
                        ))))
                    } else {
                        // Second call: block until cancelled, return nothing.
                        let cancel = self.cancel.clone();
                        let s = stream::unfold(cancel, |token| async move {
                            token.cancelled().await;
                            None
                        });
                        Ok(Box::pin(s))
                    }
                }
            }

            TwoPhaseBackend {
                first_done: std::sync::atomic::AtomicBool::new(false),
                cancel: second_cancel,
            }
        };

        let agent = agent_with_mode(
            backend,
            Some(Box::new(EchoTool::new("bash", "done"))),
            ConfirmationMode::Never,
        )
        .await;

        // Cancel shortly after the agent starts the second (blocking) iteration.
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });

        let stream = agent
            .send("run".to_string(), None, Some(cancel_for_agent))
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        // Should have seen the tool use + result from the completed first iteration.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolUseReceived { .. })),
            "expected ToolUseReceived"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolResult { .. })),
            "expected ToolResult"
        );
        // Final text from second backend call must not appear.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "ResponseComplete must not appear — second iteration was cancelled"
        );
        // The stream should have ended with Interrupted.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::Interrupted { .. })),
            "expected Interrupted event; got: {events:?}"
        );
    }

    #[tokio::test]
    async fn cancel_during_tool_confirmation_emits_interrupted_without_second_confirmation() {
        // Scenario: LLM emits two write-tool calls in one turn (Always mode).
        // User presses Esc during the first confirmation dialog.
        // Expected: agent declines both tools and emits Interrupted — no second
        // ToolConfirmationRequired should appear, and the loop does not continue.
        let (confirm_tx, confirm_rx) = mpsc::unbounded::<ConfirmationResponse>();
        let cancel = CancellationToken::new();
        let cancel_for_agent = cancel.clone();

        let backend = SequencedBackend::new(vec![
            {
                // Two tool calls in a single response.
                let mut events = tool_call_response("t1", "bash", r#"{}"#);
                // Remove the trailing Done; append a second tool call then Done.
                events.pop(); // remove Done
                events.extend(vec![
                    Ok(StreamEvent::ToolUseStart {
                        id: "t2".to_string(),
                        name: "bash".to_string(),
                    }),
                    Ok(StreamEvent::ToolUseDelta(r#"{}"#.to_string())),
                    Ok(StreamEvent::ToolUseDone),
                    Ok(StreamEvent::Done),
                ]);
                events
            },
            text_response("should not be reached"),
        ]);

        let agent = agent_with_mode(
            backend,
            Some(Box::new(EchoTool::new("bash", "output"))),
            ConfirmationMode::Always,
        )
        .await;

        // Simulate Esc: send Rejected for the first dialog and cancel the token.
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            // Give the agent time to emit the first ToolConfirmationRequired.
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            let _ = confirm_tx.unbounded_send(ConfirmationResponse::Rejected);
            cancel_clone.cancel();
        });

        let stream = agent
            .send("run".to_string(), Some(confirm_rx), Some(cancel_for_agent))
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        // First ToolConfirmationRequired must appear (the user sees the dialog).
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolConfirmationRequired { index: 1, .. })),
            "expected first ToolConfirmationRequired; got: {events:?}"
        );
        // Second ToolConfirmationRequired must NOT appear (Esc cancelled it).
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolConfirmationRequired { index: 2, .. })),
            "second ToolConfirmationRequired must not appear after Esc; got: {events:?}"
        );
        // Stream must end with Interrupted, not ResponseComplete.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::Interrupted { .. })),
            "expected Interrupted after Esc during confirmation; got: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "ResponseComplete must not appear after cancellation; got: {events:?}"
        );
    }

    #[tokio::test]
    async fn cancel_during_tool_execution_returns_promptly_with_cancelled_result() {
        // Regression: prior to wiring cancel_token into execute_tool_calls, pressing
        // Esc while a long-running tool was executing left the agent blocked on
        // tool.execute().await for the full duration of the tool — freezing the TUI
        // until the tool finished. With the fix, the in-flight execute future is
        // dropped and a "Tool cancelled by user." result is produced.
        let cancel = CancellationToken::new();
        let cancel_for_agent = cancel.clone();

        // Backend: emit a single 5-second sleep tool call.
        let backend = SequencedBackend::new(vec![vec![
            Ok(StreamEvent::ToolUseStart {
                id: "s1".to_string(),
                name: "sleep".to_string(),
            }),
            Ok(StreamEvent::ToolUseDelta(
                r#"{"duration_ms":5000}"#.to_string(),
            )),
            Ok(StreamEvent::ToolUseDone),
            Ok(StreamEvent::Done),
        ]]);
        let agent = agent_with_mode(
            backend,
            Some(Box::new(SleepTool { duration_ms: 5000 })),
            ConfirmationMode::Never,
        )
        .await;

        // Cancel after 100ms — well before the 5s sleep would finish.
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            cancel_clone.cancel();
        });

        let start = std::time::Instant::now();
        let stream = agent
            .send("sleep".to_string(), None, Some(cancel_for_agent))
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;
        let elapsed = start.elapsed();

        // Must complete well under 5s (the tool's natural duration).
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "agent did not interrupt mid-tool; took {}ms (sleep would have taken 5000ms)",
            elapsed.as_millis()
        );

        // The cancelled tool result must appear (proof the future was dropped, not awaited).
        assert!(
            events.iter().any(|e| matches!(
                e,
                AgentEvent::ToolResult { content, is_error: true, .. }
                    if content.contains("cancelled")
            )),
            "expected ToolResult with 'cancelled' content; got: {events:?}"
        );

        // Stream must end with Interrupted.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::Interrupted { .. })),
            "expected Interrupted; got: {events:?}"
        );
    }

    // ── Thinking accumulation tests (Finding 6) ──────────────────────────

    #[tokio::test]
    async fn thinking_delta_emits_thinking_received_and_accumulates() {
        let backend = SequencedBackend::new(vec![thinking_then_text_response(
            "reasoning about the problem",
            "final answer",
        )]);
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never).await;

        let stream = agent
            .send("think".to_string(), None, None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ThinkingReceived(t) if t == "reasoning about the problem")),
            "expected ThinkingReceived with reasoning text"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::TokenReceived(t) if t == "final answer")),
            "expected TokenReceived with final answer"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "expected ResponseComplete"
        );
    }

    #[tokio::test]
    async fn thinking_persisted_in_history_with_signature() {
        let backend = SequencedBackend::new(vec![thinking_then_text_response(
            "my reasoning",
            "my answer",
        )]);
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never).await;

        let stream = agent
            .send("think".to_string(), None, None)
            .await
            .expect("send should succeed");
        let _events = collect_events(stream).await;

        let history = agent.history();
        let assistant_msg = history
            .iter()
            .find(|m| m.role == Role::Assistant)
            .expect("should have assistant message");

        let thinking_block = assistant_msg
            .content
            .iter()
            .find(|b| matches!(b, ContentBlock::Thinking { .. }))
            .expect("should have thinking block");
        if let ContentBlock::Thinking { text, signature } = thinking_block {
            assert_eq!(text, "my reasoning");
            assert_eq!(
                signature, "sig_abc123",
                "signature should be captured from stream"
            );
        }

        let text_block = assistant_msg
            .content
            .iter()
            .find(|b| matches!(b, ContentBlock::Text(_)))
            .expect("should have text block");
        if let ContentBlock::Text(text) = text_block {
            assert_eq!(text, "my answer");
        }
    }

    #[tokio::test]
    async fn thinking_only_response_persisted_in_history() {
        let backend = SequencedBackend::new(vec![thinking_only_response("just thinking")]);
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never).await;

        let stream = agent
            .send("think".to_string(), None, None)
            .await
            .expect("send should succeed");
        let _events = collect_events(stream).await;

        let history = agent.history();
        let assistant_msg = history
            .iter()
            .find(|m| m.role == Role::Assistant)
            .expect("should have assistant message");

        assert!(
            assistant_msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Thinking { .. })),
            "should have thinking block"
        );
    }

    #[tokio::test]
    async fn thinking_persisted_on_stream_error() {
        let backend = SequencedBackend::new(vec![vec![
            Ok(StreamEvent::ThinkingDelta("partial thinking".to_string())),
            Ok(StreamEvent::ThinkingSignature("sig_err".to_string())),
            Err(anyhow::anyhow!("stream error")),
        ]]);
        let agent = agent_with_mode(backend, None, ConfirmationMode::Never).await;

        let stream = agent
            .send("think".to_string(), None, None)
            .await
            .expect("send should succeed");
        let _events = collect_events(stream).await;

        let history = agent.history();
        let has_thinking = history.iter().any(|m| {
            m.role == Role::Assistant
                && m.content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::Thinking { text, signature } if text == "partial thinking" && signature == "sig_err"))
        });
        assert!(
            has_thinking,
            "thinking should be persisted even on stream error; history: {history:?}"
        );
    }

    #[tokio::test]
    async fn thinking_persisted_on_pre_tool_cancellation() {
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        // Backend emits thinking then blocks until cancelled
        struct ThinkCancelBackend {
            token: CancellationToken,
        }
        #[async_trait]
        impl LlmBackend for ThinkCancelBackend {
            async fn send_message(
                &self,
                _: &[Message],
                _: &RequestConfig,
            ) -> Result<BoxStream<Result<StreamEvent>>> {
                let token = self.token.clone();
                let events: Vec<Result<StreamEvent>> = vec![
                    Ok(StreamEvent::ThinkingDelta(
                        "thinking before cancel".to_string(),
                    )),
                    Ok(StreamEvent::ThinkingSignature("sig_cancel".to_string())),
                ];
                Ok(Box::pin(futures::stream::unfold(
                    (events.into_iter(), token, false),
                    |(mut iter, token, done)| async move {
                        if done {
                            return None;
                        }
                        if let Some(item) = iter.next() {
                            return Some((item, (iter, token, false)));
                        }
                        token.cancelled().await;
                        None
                    },
                )))
            }
        }

        let agent = agent_with_mode(
            ThinkCancelBackend {
                token: cancel_clone,
            },
            None,
            ConfirmationMode::Never,
        )
        .await;

        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            cancel.cancel();
        });

        let stream = agent
            .send("think".to_string(), None, Some(CancellationToken::new()))
            .await
            .expect("send should succeed");
        let _events = collect_events(stream).await;

        let history = agent.history();
        let has_thinking = history.iter().any(|m| {
            m.role == Role::Assistant
                && m.content.iter().any(|b| matches!(b, ContentBlock::Thinking { text, .. } if text == "thinking before cancel"))
        });
        assert!(
            has_thinking,
            "thinking should be persisted on pre-tool cancellation; history: {history:?}"
        );
    }

    #[tokio::test]
    async fn thinking_with_tool_call_includes_signature_in_assistant_message() {
        let backend = SequencedBackend::new(vec![
            vec![
                Ok(StreamEvent::ThinkingDelta("let me think".to_string())),
                Ok(StreamEvent::ThinkingSignature("sig_tool".to_string())),
                Ok(StreamEvent::ToolUseStart {
                    id: "t1".to_string(),
                    name: "bash".to_string(),
                }),
                Ok(StreamEvent::ToolUseDelta("{}".to_string())),
                Ok(StreamEvent::ToolUseDone),
                Ok(StreamEvent::Done),
            ],
            text_response("done"),
        ]);
        let agent = agent_with_mode(
            backend,
            Some(Box::new(EchoTool::new("bash", "output"))),
            ConfirmationMode::Never,
        )
        .await;

        let stream = agent
            .send("run".to_string(), None, None)
            .await
            .expect("send should succeed");
        let _events = collect_events(stream).await;

        let history = agent.history();
        let assistant_with_tool = history
            .iter()
            .find(|m| {
                m.role == Role::Assistant
                    && m.content
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
            })
            .expect("should have assistant message with tool use");

        let thinking = assistant_with_tool
            .content
            .iter()
            .find(|b| matches!(b, ContentBlock::Thinking { .. }));
        assert!(
            thinking.is_some(),
            "thinking block should be in assistant message with tool use"
        );
        if let ContentBlock::Thinking { text, signature } = thinking.expect("checked above") {
            assert_eq!(text, "let me think");
            assert_eq!(signature, "sig_tool");
        }
    }

    // ── Compaction tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn compact_without_spawner_returns_error() {
        let backend = SequencedBackend::new(vec![]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let session = test_session_arc().await;
        let agent = Agent::new(Box::new(backend), config, session).await;

        let result = agent.compact().await;
        assert!(
            result.is_err(),
            "compact without spawner should return error"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("no spawner"),
            "error message should mention missing spawner, got: {msg}"
        );
    }

    #[test]
    fn compaction_role_falls_back_to_default_when_not_in_models() {
        use crate::config::AppConfig;
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: crate::config::VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            tools: crate::config::ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models: std::collections::BTreeMap::new(),
            thinking: None,
            compaction: CompactionConfig::default(),
        };
        let role = if config.models.contains_key(&config.compaction.role) {
            config.compaction.role.clone()
        } else {
            "default".to_string()
        };
        assert_eq!(
            role, "default",
            "should fall back to default when compaction role not in models"
        );
    }

    #[test]
    fn compaction_role_uses_configured_role_when_in_models() {
        use crate::config::{AppConfig, ModelRole};
        let mut models = std::collections::BTreeMap::new();
        models.insert(
            "compaction".to_string(),
            ModelRole {
                backend: "vertex".to_string(),
                model: "claude-haiku".to_string(),
            },
        );
        let config = AppConfig {
            backend: "vertex".to_string(),
            vertex: crate::config::VertexConfig {
                project: "proj".to_string(),
                region: "us-east5".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            zai: None,
            ollama: None,
            openai_compat: None,
            tools: crate::config::ToolsConfig::default(),
            sessions_dir: std::env::temp_dir(),
            models,
            thinking: None,
            compaction: CompactionConfig::default(),
        };
        let role = if config.models.contains_key(&config.compaction.role) {
            config.compaction.role.clone()
        } else {
            "default".to_string()
        };
        assert_eq!(
            role, "compaction",
            "should use compaction role when defined in models"
        );
    }

    // ── Compaction with retained turns integration tests ────────────────

    #[tokio::test]
    async fn compact_with_retained_turns_rebuilds_history_correctly() {
        use crate::compact_helpers::{retained_start_index, retained_turns};

        let session = test_session_arc().await;

        // Insert 3 turns into the session
        let messages = vec![
            Message::text(Role::User, "hello".to_string()),
            Message::text(Role::Assistant, "hi there".to_string()),
            Message::text(Role::User, "how are you".to_string()),
            Message::text(Role::Assistant, "fine".to_string()),
            Message::text(Role::User, "goodbye".to_string()),
            Message::text(Role::Assistant, "see you".to_string()),
        ];
        for msg in &messages {
            session
                .lock()
                .await
                .conversation()
                .insert_message(msg)
                .await
                .expect("insert");
        }

        let summary_text = "[Compacted] Summary of conversation";
        let summary_msg = Message::text(Role::User, summary_text.to_string());

        // Compact with 1 retained turn via DB
        let result = session
            .lock()
            .await
            .conversation()
            .compact_retaining(&summary_msg, 1)
            .await
            .expect("compact_retaining should succeed");
        assert!(result, "compact_retaining should perform compaction");

        // Load DB history
        let db_history = session
            .lock()
            .await
            .conversation()
            .load_history()
            .await
            .expect("load DB history");

        // Compute expected in-memory history using shared helpers
        let retained_start = retained_start_index(&messages, 1);
        let expected_retained = retained_turns(&messages, 1);

        // Expected history: [summary, retained...]
        let mut expected: Vec<Message> = vec![summary_msg.clone()];
        expected.extend(expected_retained.clone());

        // Verify DB matches expected
        assert_eq!(
            db_history.len(),
            expected.len(),
            "DB history length mismatch"
        );
        assert_eq!(db_history[0].role, Role::User);
        assert!(
            matches!(&db_history[0].content[0], ContentBlock::Text(t) if t.contains("[Compacted]")),
            "first DB entry should be summary"
        );

        // Verify retained messages have only text blocks
        for (i, msg) in db_history.iter().skip(1).enumerate() {
            assert!(
                msg.content
                    .iter()
                    .all(|b| matches!(b, ContentBlock::Text(_))),
                "retained message {} should only have text blocks",
                i
            );
        }

        // Verify the compacted messages (before retained_start) are NOT in DB
        let compacted = &messages[..retained_start];
        for compacted_msg in compacted {
            let found = db_history.iter().any(|db_msg| {
                db_msg.content.iter().any(|db_b| {
                    compacted_msg.content.iter().any(|c_b| {
                        if let (ContentBlock::Text(a), ContentBlock::Text(b)) = (c_b, db_b) {
                            a == b
                        } else {
                            false
                        }
                    })
                })
            });
            assert!(!found, "compacted message should not appear in DB history");
        }
    }

    #[tokio::test]
    async fn compact_with_retained_turns_strips_tool_use_from_retained() {
        use crate::compact_helpers::retained_turns;

        let session = test_session_arc().await;

        // turn 1: user + assistant (text only, will be compacted)
        // turn 2: user text + assistant with tool_use and text
        let messages = vec![
            Message::text(Role::User, "hello".to_string()),
            Message::text(Role::Assistant, "hi".to_string()),
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text("run it".to_string())],
            },
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::ToolUse {
                        id: "t1".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({}),
                    },
                    ContentBlock::Text("done".to_string()),
                ],
            },
        ];
        for msg in &messages {
            session
                .lock()
                .await
                .conversation()
                .insert_message(msg)
                .await
                .expect("insert");
        }

        let summary_msg = Message::text(Role::User, "[Compacted] Summary".to_string());

        let result = session
            .lock()
            .await
            .conversation()
            .compact_retaining(&summary_msg, 1)
            .await
            .expect("compact_retaining should succeed");
        assert!(result);

        let db_history = session
            .lock()
            .await
            .conversation()
            .load_history()
            .await
            .expect("load history");

        // Expected: summary + user "run it" + assistant "done" (tool_use stripped)
        let expected_retained = retained_turns(&messages, 1);
        assert_eq!(
            expected_retained.len(),
            2,
            "should retain 2 messages after stripping"
        );

        assert_eq!(db_history.len(), 3, "summary + 2 retained = 3");
        assert!(
            matches!(&db_history[0].content[0], ContentBlock::Text(t) if t.contains("[Compacted]"))
        );
        assert!(matches!(&db_history[1].content[0], ContentBlock::Text(t) if t == "run it"));
        assert!(matches!(&db_history[2].content[0], ContentBlock::Text(t) if t == "done"));

        // Verify no tool-use blocks in retained messages
        for msg in db_history.iter().skip(1) {
            assert!(
                msg.content
                    .iter()
                    .all(|b| matches!(b, ContentBlock::Text(_))),
                "retained messages must only contain text blocks"
            );
        }
    }

    #[tokio::test]
    async fn compact_with_zero_retained_turns_behaves_as_before() {
        let session = test_session_arc().await;

        let messages = vec![
            Message::text(Role::User, "hello".to_string()),
            Message::text(Role::Assistant, "hi".to_string()),
            Message::text(Role::User, "how are you".to_string()),
            Message::text(Role::Assistant, "fine".to_string()),
        ];
        for msg in &messages {
            session
                .lock()
                .await
                .conversation()
                .insert_message(msg)
                .await
                .expect("insert");
        }

        let summary_msg = Message::text(Role::User, "[Compacted] Summary".to_string());

        let result = session
            .lock()
            .await
            .conversation()
            .compact_retaining(&summary_msg, 0)
            .await
            .expect("compact_retaining should succeed");
        assert!(result);

        let db_history = session
            .lock()
            .await
            .conversation()
            .load_history()
            .await
            .expect("load history");

        assert_eq!(
            db_history.len(),
            1,
            "with 0 retained turns, only summary should remain"
        );
        assert!(
            matches!(&db_history[0].content[0], ContentBlock::Text(t) if t.contains("[Compacted]"))
        );
    }

    // ── Auto-compact threshold tests ─────────────────────────────────────

    fn text_with_usage_response(
        text: &str,
        input_tokens: u32,
        output_tokens: u32,
    ) -> Vec<Result<StreamEvent>> {
        vec![
            Ok(StreamEvent::TextDelta(text.to_string())),
            Ok(StreamEvent::Usage {
                input_tokens,
                output_tokens,
                stop_reason: "end_turn".to_string(),
            }),
            Ok(StreamEvent::Done),
        ]
    }

    #[tokio::test]
    async fn auto_compact_not_triggered_when_threshold_zero() {
        let backend = SequencedBackend::new(vec![text_with_usage_response("hello", 100_000, 50)]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        let compaction_config = crate::config::CompactionConfig {
            max_context_window_len: 0,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tool_config(&tool_config)
            .with_compaction_config(&compaction_config);

        let stream = agent
            .send("hi".to_string(), None, None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "expected ResponseComplete"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::AutoCompactTriggered { .. })),
            "AutoCompactTriggered should not be emitted when threshold is 0"
        );
    }

    #[tokio::test]
    async fn auto_compact_not_triggered_when_tokens_below_threshold() {
        let backend = SequencedBackend::new(vec![text_with_usage_response("hello", 40_000, 50)]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        let compaction_config = crate::config::CompactionConfig {
            max_context_window_len: 50_000,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tool_config(&tool_config)
            .with_compaction_config(&compaction_config);

        let stream = agent
            .send("hi".to_string(), None, None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::AutoCompactTriggered { .. })),
            "AutoCompactTriggered should not be emitted when tokens below threshold"
        );
    }

    #[tokio::test]
    async fn auto_compact_triggered_when_tokens_exceed_threshold() {
        let backend = SequencedBackend::new(vec![text_with_usage_response("hello", 60_000, 50)]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        let compaction_config = crate::config::CompactionConfig {
            max_context_window_len: 50_000,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tool_config(&tool_config)
            .with_compaction_config(&compaction_config);

        let stream = agent
            .send("hi".to_string(), None, None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        let triggered = events
            .iter()
            .find(|e| matches!(e, AgentEvent::AutoCompactTriggered { .. }));
        assert!(
            triggered.is_some(),
            "expected AutoCompactTriggered event when tokens exceed threshold"
        );
        if let AgentEvent::AutoCompactTriggered {
            current_tokens,
            threshold,
        } = triggered.expect("checked above")
        {
            assert_eq!(
                *current_tokens, 60_000,
                "current_tokens should match input_tokens"
            );
            assert_eq!(*threshold, 50_000, "threshold should match config");
        }

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ResponseComplete(_))),
            "expected ResponseComplete"
        );
    }

    #[tokio::test]
    async fn auto_compact_consecutive_guard_emits_warning() {
        let combined = SequencedBackend::new(vec![
            text_with_usage_response("first", 60_000, 50),
            text_with_usage_response("second", 55_000, 50),
        ]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        let compaction_config = crate::config::CompactionConfig {
            max_context_window_len: 50_000,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(combined), config, test_session_arc().await)
            .await
            .with_tool_config(&tool_config)
            .with_compaction_config(&compaction_config);

        // First send: above threshold -> AutoCompactTriggered
        let stream1 = agent
            .send("hi".to_string(), None, None)
            .await
            .expect("first send should succeed");
        let events1 = collect_events(stream1).await;
        assert!(
            events1
                .iter()
                .any(|e| matches!(e, AgentEvent::AutoCompactTriggered { .. })),
            "first send should trigger AutoCompactTriggered"
        );

        // Second send: still above threshold -> Warn instead of AutoCompactTriggered
        let stream2 = agent
            .send("hi again".to_string(), None, None)
            .await
            .expect("second send should succeed");
        let events2 = collect_events(stream2).await;

        // Should NOT have AutoCompactTriggered
        assert!(
            !events2
                .iter()
                .any(|e| matches!(e, AgentEvent::AutoCompactTriggered { .. })),
            "second consecutive send above threshold should NOT trigger AutoCompactTriggered"
        );
        let warn_events: Vec<_> = events2
            .iter()
            .filter(|e| matches!(e, AgentEvent::Warn(_)))
            .collect();
        assert!(
            !warn_events.is_empty(),
            "second consecutive send above threshold should emit Warn"
        );
    }

    #[tokio::test]
    async fn auto_compact_guard_resets_when_tokens_drop_below_threshold() {
        let combined = SequencedBackend::new(vec![
            text_with_usage_response("above", 60_000, 50),
            text_with_usage_response("below", 30_000, 50),
            text_with_usage_response("above again", 60_000, 50),
        ]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        let compaction_config = crate::config::CompactionConfig {
            max_context_window_len: 50_000,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(combined), config, test_session_arc().await)
            .await
            .with_tool_config(&tool_config)
            .with_compaction_config(&compaction_config);

        // First send: above threshold -> AutoCompactTriggered
        let stream1 = agent
            .send("hi".to_string(), None, None)
            .await
            .expect("first send should succeed");
        let events1 = collect_events(stream1).await;
        assert!(
            events1
                .iter()
                .any(|e| matches!(e, AgentEvent::AutoCompactTriggered { .. })),
            "first send should trigger AutoCompactTriggered"
        );

        // Second send: below threshold -> guard resets
        let stream2 = agent
            .send("hello".to_string(), None, None)
            .await
            .expect("second send should succeed");
        let events2 = collect_events(stream2).await;
        assert!(
            !events2
                .iter()
                .any(|e| matches!(e, AgentEvent::AutoCompactTriggered { .. })),
            "second send below threshold should not trigger AutoCompactTriggered"
        );

        // Third send: above threshold again -> AutoCompactTriggered (guard was reset)
        let stream3 = agent
            .send("hi again".to_string(), None, None)
            .await
            .expect("third send should succeed");
        let events3 = collect_events(stream3).await;
        assert!(
            events3
                .iter()
                .any(|e| matches!(e, AgentEvent::AutoCompactTriggered { .. })),
            "third send above threshold should trigger AutoCompactTriggered again after guard reset"
        );
    }

    #[tokio::test]
    async fn with_compaction_config_sets_max_context_window_len() {
        let backend = SequencedBackend::new(vec![]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        let compaction_config = crate::config::CompactionConfig {
            max_context_window_len: 42_000,
            num_retained_turns: 3,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tool_config(&tool_config)
            .with_compaction_config(&compaction_config);
        assert_eq!(
            agent.max_context_window_len_for_test(),
            42_000,
            "max_context_window_len should be set from compaction_config"
        );
        assert_eq!(
            agent.num_retained_turns_for_test(),
            3,
            "num_retained_turns should be set from compaction_config"
        );
    }

    // Regression: auto-compact must trigger on tool-use turns, not just text-only turns.
    // When the LLM emits tool calls, the threshold check must still run after tool
    // results are persisted and the loop continues.
    #[tokio::test]
    async fn auto_compact_triggers_on_tool_use_turn() {
        // First turn: tool call with high input_tokens.
        // Second turn: text-only response (triggers after loop iteration).
        // The tool-use turn should emit AutoCompactTriggered before the loop
        // continues to the second iteration.
        let combined = SequencedBackend::new(vec![
            vec![
                Ok(StreamEvent::ToolUseStart {
                    id: "t1".to_string(),
                    name: "bash".to_string(),
                }),
                Ok(StreamEvent::ToolUseDelta("{}".to_string())),
                Ok(StreamEvent::ToolUseDone),
                Ok(StreamEvent::Usage {
                    input_tokens: 60_000,
                    output_tokens: 100,
                    stop_reason: "end_turn".to_string(),
                }),
                Ok(StreamEvent::Done),
            ],
            text_response("done"),
        ]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let mut registry = ToolRegistry::new();
        registry
            .register(Box::new(EchoTool::new("bash", "output")))
            .expect("register");
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        let compaction_config = crate::config::CompactionConfig {
            max_context_window_len: 50_000,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(combined), config, test_session_arc().await)
            .await
            .with_tools(registry)
            .with_tool_config(&tool_config)
            .with_compaction_config(&compaction_config);

        let stream = agent
            .send("run".to_string(), None, None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        // Must see AutoCompactTriggered even though the first turn had tool calls.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::AutoCompactTriggered { .. })),
            "AutoCompactTriggered should be emitted on tool-use turns"
        );
    }

    #[tokio::test]
    async fn load_session_resets_auto_compact_flag() {
        let backend = SequencedBackend::new(vec![text_with_usage_response("hi", 60_000, 50)]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        let compaction_config = crate::config::CompactionConfig {
            max_context_window_len: 50_000,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(backend), config, test_session_arc().await)
            .await
            .with_tool_config(&tool_config)
            .with_compaction_config(&compaction_config);

        // Trigger auto-compact to set the flag
        let stream = agent
            .send("hi".to_string(), None, None)
            .await
            .expect("send should succeed");
        let _events = collect_events(stream).await;
        assert!(
            agent.last_auto_compacted.load(Ordering::SeqCst),
            "flag should be set after AutoCompactTriggered"
        );

        // Load a new session — should reset the flag
        let dir = tempfile::TempDir::new().expect("temp dir");
        let new_session = Session::new(None, dir.keep()).await.expect("session");
        agent.load_session(new_session).await;
        assert!(
            !agent.last_auto_compacted.load(Ordering::SeqCst),
            "flag should be reset after load_session"
        );
    }

    // Regression: when multiple Usage events arrive in a single turn (e.g. tool-use
    // loops), the API reports the *total* context size per request, not incremental
    // tokens. We must track the peak, not the sum, so that a context of 12.8k across
    // two iterations doesn't falsely accumulate to 27.8k.
    #[tokio::test]
    async fn auto_compact_uses_peak_input_tokens_not_sum() {
        // Simulate two Usage events in one response (can happen with tool-use turns).
        // Both report the total context size (~12k), but the peak is 12k, not 24k.
        let combined = SequencedBackend::new(vec![vec![
            Ok(StreamEvent::TextDelta("hello".to_string())),
            Ok(StreamEvent::Usage {
                input_tokens: 12_000,
                output_tokens: 50,
                stop_reason: "end_turn".to_string(),
            }),
            // Second Usage event (some backends send a final summary)
            Ok(StreamEvent::Usage {
                input_tokens: 12_800,
                output_tokens: 50,
                stop_reason: "end_turn".to_string(),
            }),
            Ok(StreamEvent::Done),
        ]]);
        let config = RequestConfig {
            model: "test".to_string(),
            max_tokens: 100,
            tools: vec![],
            thinking: None,
        };
        let tool_config = ToolsConfig {
            confirmation: ConfirmationMode::Never,
            ..Default::default()
        };
        let compaction_config = crate::config::CompactionConfig {
            max_context_window_len: 15_000,
            ..Default::default()
        };
        let agent = Agent::new(Box::new(combined), config, test_session_arc().await)
            .await
            .with_tool_config(&tool_config)
            .with_compaction_config(&compaction_config);

        let stream = agent
            .send("hi".to_string(), None, None)
            .await
            .expect("send should succeed");
        let events = collect_events(stream).await;

        // Peak is 12800, which is below 15000, so auto-compact should NOT trigger.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::AutoCompactTriggered { .. })),
            "AutoCompactTriggered should not be emitted when peak (12800) is below threshold (15000)"
        );
    }
}
