use anyhow::Result;
use futures::StreamExt;
use futures::channel::mpsc;
use std::io::{self, BufRead, IsTerminal, Write};
use std::sync::Arc;

use crate::agent::Agent;
use crate::logging;
use crate::types::{AgentEvent, BoxStream, ConfirmationResponse};

#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

/// A compiled JSON Schema paired with its original source text.
///
/// The `validator` is used to validate LLM responses. The `raw` text is
/// included verbatim in reprompt messages so the model can see the schema.
pub struct JsonSchema {
    pub validator: jsonschema::Validator,
    pub raw: String,
}

impl std::fmt::Debug for JsonSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonSchema")
            .field("raw", &self.raw)
            .finish()
    }
}

pub async fn run(
    agent: Arc<Agent>,
    prompt: String,
    format: OutputFormat,
    json_schema: Option<JsonSchema>,
    max_schema_retries: u32,
) -> Result<()> {
    let is_tty = std::io::stdin().is_terminal();
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let mut stdin = io::BufReader::new(std::io::stdin());
    run_with_writer(
        agent,
        prompt,
        &mut handle,
        format,
        json_schema,
        max_schema_retries,
        is_tty,
        &mut stdin,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_with_writer<W: Write, R: BufRead>(
    agent: Arc<Agent>,
    prompt: String,
    writer: &mut W,
    format: OutputFormat,
    json_schema: Option<JsonSchema>,
    max_schema_retries: u32,
    is_tty: bool,
    stdin: &mut R,
) -> Result<()> {
    match format {
        OutputFormat::Text => {
            let (confirm_tx, confirm_rx) = mpsc::unbounded::<ConfirmationResponse>();
            let mut stream = agent.send(prompt, Some(confirm_rx), None).await?;
            run_text(&mut stream, writer, confirm_tx, is_tty, stdin).await
        }
        OutputFormat::Json => {
            run_json(
                agent,
                prompt,
                writer,
                json_schema,
                max_schema_retries,
                is_tty,
                stdin,
            )
            .await
        }
    }
}

async fn run_text<W: Write, R: BufRead>(
    stream: &mut BoxStream<AgentEvent>,
    writer: &mut W,
    confirm_tx: mpsc::UnboundedSender<ConfirmationResponse>,
    is_tty: bool,
    stdin: &mut R,
) -> Result<()> {
    while let Some(event) = stream.next().await {
        logging::log_event(&event);
        logging::flush();
        match event {
            AgentEvent::TokenReceived(text) => {
                write!(writer, "{}", text)?;
                writer.flush()?;
            }
            AgentEvent::ResponseComplete(_) => {
                writeln!(writer)?;
                break;
            }
            AgentEvent::Error(msg) => {
                eprintln!("\nError: {}", msg);
                anyhow::bail!("LLM error: {}", msg);
            }
            AgentEvent::ToolUseReceived {
                name, input, index, ..
            } => {
                writeln!(writer, "\n[tool({index}): {}] {}", name, input)?;
            }
            AgentEvent::ToolResult {
                name,
                content,
                is_error,
                index,
            } => {
                if is_error {
                    writeln!(writer, "[error({index}) from {}]: {}", name, content)?;
                } else {
                    writeln!(writer, "[result({index}) from {}]: {}", name, content)?;
                }
            }
            AgentEvent::Usage { .. } => {}
            AgentEvent::SubAgentUsage { .. } => {}
            AgentEvent::Warn(_) => {}
            AgentEvent::CompactionComplete { .. } => {}
            AgentEvent::AutoCompactTriggered { .. } => {}
            AgentEvent::ThinkingReceived(text) => {
                // Gated on --debug so single-shot stdout stays pipe-friendly by
                // default; thinking text would otherwise interleave with the
                // assistant response and break downstream JSON/jq parsers.
                if crate::logging::is_enabled() {
                    write!(writer, "// {}", text)?;
                    writer.flush()?;
                }
            }
            AgentEvent::Interrupted { .. } => {
                writeln!(writer, "\n*(interrupted)*")?;
                break;
            }
            AgentEvent::ToolConfirmationRequired { name, input, .. } => {
                handle_confirmation(confirm_tx.clone(), is_tty, stdin, &name, &input)?;
            }
        }
    }
    Ok(())
}

// Collects a single LLM response from the stream, returning the final text and
// whether an error occurred.
//
// Not the same as `agent::run_headless`: this function is interactive (prompts
// the user for tool confirmations, streams tokens to a logger, works with a
// confirm channel) and display-coupled (lives in the frontend layer). By
// contrast, `run_headless` is fully non-interactive (auto-rejects confirmations,
// accumulates usage totals, no logger) and lives in the agent layer.
async fn collect_response<R: BufRead>(
    stream: &mut BoxStream<AgentEvent>,
    confirm_tx: mpsc::UnboundedSender<ConfirmationResponse>,
    is_tty: bool,
    stdin: &mut R,
) -> Result<(String, bool)> {
    let mut result_text = String::new();
    let mut is_error = false;
    let mut in_tool_iteration = false;

    while let Some(event) = stream.next().await {
        logging::log_event(&event);
        logging::flush();
        match event {
            AgentEvent::TokenReceived(text) => {
                if !in_tool_iteration {
                    result_text.push_str(&text);
                }
            }
            AgentEvent::ToolUseReceived { .. } => {
                result_text.clear();
                in_tool_iteration = true;
            }
            AgentEvent::ToolResult { .. } => {
                in_tool_iteration = false;
            }
            AgentEvent::ResponseComplete(_) => {
                break;
            }
            AgentEvent::Error(msg) => {
                is_error = true;
                result_text = msg;
                break;
            }
            AgentEvent::Usage { .. } => {}
            AgentEvent::SubAgentUsage { .. } => {}
            AgentEvent::Warn(_) => {}
            AgentEvent::CompactionComplete { .. } => {}
            AgentEvent::AutoCompactTriggered { .. } => {}
            AgentEvent::ThinkingReceived(_) => {}
            AgentEvent::Interrupted { partial_text } => {
                is_error = true;
                result_text = partial_text;
                break;
            }
            AgentEvent::ToolConfirmationRequired { name, .. } if !is_tty => {
                let _ = confirm_tx.unbounded_send(ConfirmationResponse::Rejected);
                is_error = true;
                result_text = format!("tool confirmation required in non-TTY mode (tool: {name})");
                break;
            }
            AgentEvent::ToolConfirmationRequired { name, input, .. } => {
                handle_confirmation(confirm_tx.clone(), is_tty, stdin, &name, &input)?;
            }
        }
    }

    Ok((result_text, is_error))
}

// In JSON mode the `result` field contains only the final assistant turn — the
// text emitted after the last tool result. Intermediate "thinking" tokens that
// appear before a tool call are discarded so that scripts receive a clean,
// singular output rather than concatenated reasoning + final answer.
//
// When a JSON schema is provided, the response is validated against it. On
// success the `result` key contains the parsed JSON value directly (not a
// string). On failure a reprompt is sent through the agent, up to
// `max_schema_retries` times. On exhaustion `is_error: true` is emitted with
// the validation error as a string in `result`.
#[allow(clippy::too_many_arguments)]
async fn run_json<W: Write, R: BufRead>(
    agent: Arc<Agent>,
    initial_prompt: String,
    writer: &mut W,
    json_schema: Option<JsonSchema>,
    max_schema_retries: u32,
    is_tty: bool,
    stdin: &mut R,
) -> Result<()> {
    let (confirm_tx, confirm_rx) = mpsc::unbounded::<ConfirmationResponse>();
    let mut stream = agent.send(initial_prompt, Some(confirm_rx), None).await?;
    let initial = collect_response(&mut stream, confirm_tx, is_tty, stdin).await?;

    let (result, is_error) = apply_schema_retry(
        initial,
        json_schema.as_ref(),
        max_schema_retries,
        &agent,
        is_tty,
        stdin,
    )
    .await?;

    let output = serde_json::json!({
        "is_error": is_error,
        "result": result,
    });

    writeln!(writer, "{}", output)?;
    writer.flush()?;

    if is_error {
        anyhow::bail!("LLM error: {}", result);
    }

    Ok(())
}

// Drives the schema-validation + reprompt retry loop.
//
// `initial` is the `(text, is_error)` from the first LLM call. When no schema
// is provided, returns `(Value::String(text), is_error)`. When a schema is
// provided and validation succeeds, returns the parsed JSON value directly as
// `result`. On validation failure, reprompts up to `max_schema_retries` times.
// On exhaustion, returns `(Value::String(validation_err), true)`.
//
// Extracted so that tests can call it directly with a fake agent that returns
// canned responses, exercising the same production code path.
#[allow(clippy::too_many_arguments)]
async fn apply_schema_retry<R: BufRead>(
    initial: (String, bool),
    json_schema: Option<&JsonSchema>,
    max_schema_retries: u32,
    agent: &Agent,
    is_tty: bool,
    stdin: &mut R,
) -> Result<(serde_json::Value, bool)> {
    let (mut result_text, mut is_error) = initial;

    if !is_error && let Some(schema) = json_schema {
        let mut retries_left = max_schema_retries;

        loop {
            match validate_json_output(&result_text, &schema.validator) {
                Ok(parsed) => {
                    return Ok((parsed, false));
                }
                Err(validation_err) => {
                    if retries_left == 0 {
                        is_error = true;
                        result_text = validation_err;
                        break;
                    }
                    retries_left -= 1;
                    let reprompt = format!(
                        "Your previous output was invalid: {}. Output ONLY valid JSON conforming to this schema: {}",
                        validation_err, schema.raw
                    );
                    let (retry_tx, retry_rx) = mpsc::unbounded::<ConfirmationResponse>();
                    let mut retry_stream = agent.send(reprompt, Some(retry_rx), None).await?;
                    let (new_text, new_error) =
                        collect_response(&mut retry_stream, retry_tx, is_tty, stdin).await?;
                    result_text = new_text;
                    is_error = new_error;
                    if is_error {
                        break;
                    }
                }
            }
        }
    }

    Ok((serde_json::Value::String(result_text), is_error))
}

fn validate_json_output(
    text: &str,
    validator: &jsonschema::Validator,
) -> Result<serde_json::Value, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("output is not valid JSON: {}", e))?;
    let errors: Vec<String> = validator
        .iter_errors(&parsed)
        .map(|e| e.to_string())
        .collect();
    if errors.is_empty() {
        Ok(parsed)
    } else {
        Err(errors.join("; "))
    }
}

fn handle_confirmation<R: BufRead>(
    confirm_tx: mpsc::UnboundedSender<ConfirmationResponse>,
    is_tty: bool,
    stdin: &mut R,
    name: &str,
    input: &serde_json::Value,
) -> Result<()> {
    if !is_tty {
        eprintln!(
            "Error: tool confirmation required but stdin is not a TTY (tool: {})",
            name
        );
        let _ = confirm_tx.unbounded_send(ConfirmationResponse::Rejected);
        anyhow::bail!("tool confirmation required in non-TTY mode");
    }
    eprint!("Allow tool '{}' with input {}? [y/N] ", name, input);
    io::stderr().flush()?;
    let mut response = String::new();
    stdin.read_line(&mut response)?;
    let decision = if response.trim().eq_ignore_ascii_case("y") {
        ConfirmationResponse::Approved
    } else {
        ConfirmationResponse::Rejected
    };
    confirm_tx
        .unbounded_send(decision)
        .map_err(|_| anyhow::anyhow!("agent confirmation channel closed"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentEvent;
    use futures::channel::mpsc;
    use futures::stream;

    fn make_confirm_channel() -> (
        mpsc::UnboundedSender<ConfirmationResponse>,
        mpsc::UnboundedReceiver<ConfirmationResponse>,
    ) {
        mpsc::unbounded()
    }

    // Helper: runs run_with_writer using a fake stream-based agent stub.
    // For tests that don't need schema retry logic, we use the internal helpers directly.

    async fn run_text_events(events: Vec<AgentEvent>, is_tty: bool) -> (Result<()>, Vec<u8>) {
        let mut s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, _rx) = make_confirm_channel();
        let result = run_text(&mut s, &mut buf, tx, is_tty, &mut io::empty()).await;
        (result, buf)
    }

    async fn run_json_collect(events: Vec<AgentEvent>, is_tty: bool) -> (String, bool) {
        let mut s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let (tx, _rx) = make_confirm_channel();
        collect_response(&mut s, tx, is_tty, &mut io::empty())
            .await
            .expect("collect should not fail")
    }

    // --- Text format tests ---

    #[tokio::test]
    async fn token_received_writes_text_to_writer() {
        let events = vec![
            AgentEvent::TokenReceived("hello".to_string()),
            AgentEvent::TokenReceived(" world".to_string()),
            AgentEvent::ResponseComplete("hello world".to_string()),
        ];
        let (result, buf) = run_text_events(events, false).await;
        result.expect("stdout run should succeed");
        assert_eq!(
            String::from_utf8(buf).expect("valid UTF-8"),
            "hello world\n"
        );
    }

    #[tokio::test]
    async fn response_complete_writes_trailing_newline_and_stops() {
        let events = vec![
            AgentEvent::ResponseComplete("done".to_string()),
            AgentEvent::TokenReceived("should not appear".to_string()),
        ];
        let (result, buf) = run_text_events(events, false).await;
        result.expect("stdout run should succeed");
        assert_eq!(String::from_utf8(buf).expect("valid UTF-8"), "\n");
    }

    #[tokio::test]
    async fn error_returns_err() {
        let events = vec![AgentEvent::Error("something went wrong".to_string())];
        let (result, _buf) = run_text_events(events, false).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("something went wrong")
        );
    }

    #[tokio::test]
    async fn tool_use_received_prints_tool_name_and_input() {
        let events = vec![
            AgentEvent::ToolUseReceived {
                id: "t1".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"command": "ls"}),
                index: 1,
            },
            AgentEvent::ResponseComplete(String::new()),
        ];
        let (result, buf) = run_text_events(events, false).await;
        result.expect("should succeed");
        let output = String::from_utf8(buf).expect("valid UTF-8");
        assert!(
            output.contains("[tool(1): bash]"),
            "should contain indexed tool name"
        );
        assert!(
            output.contains(r#""command""#),
            "should contain input JSON key"
        );
    }

    #[tokio::test]
    async fn tool_result_success_prints_result_content() {
        let events = vec![
            AgentEvent::ToolResult {
                name: "bash".to_string(),
                content: "file1.txt".to_string(),
                is_error: false,
                index: 1,
            },
            AgentEvent::ResponseComplete(String::new()),
        ];
        let (result, buf) = run_text_events(events, false).await;
        result.expect("should succeed");
        let output = String::from_utf8(buf).expect("valid UTF-8");
        assert!(output.contains("[result(1) from bash]"));
        assert!(output.contains("file1.txt"));
    }

    #[tokio::test]
    async fn tool_result_error_prints_error_label() {
        let events = vec![
            AgentEvent::ToolResult {
                name: "bash".to_string(),
                content: "permission denied".to_string(),
                is_error: true,
                index: 1,
            },
            AgentEvent::ResponseComplete(String::new()),
        ];
        let (result, buf) = run_text_events(events, false).await;
        result.expect("should succeed");
        let output = String::from_utf8(buf).expect("valid UTF-8");
        assert!(output.contains("[error(1) from bash]"));
        assert!(output.contains("permission denied"));
    }

    #[tokio::test]
    async fn auto_compact_triggered_is_silently_absorbed_in_run_text() {
        let events = vec![
            AgentEvent::TokenReceived("hello".to_string()),
            AgentEvent::AutoCompactTriggered {
                current_tokens: 60000,
                threshold: 50000,
            },
            AgentEvent::TokenReceived(" world".to_string()),
            AgentEvent::ResponseComplete("hello world".to_string()),
        ];
        let (result, buf) = run_text_events(events, false).await;
        result.expect("stdout run should succeed");
        assert_eq!(
            String::from_utf8(buf).expect("valid UTF-8"),
            "hello world\n"
        );
    }

    #[tokio::test]
    async fn collect_response_auto_compact_triggered_is_silently_absorbed() {
        let events = vec![
            AgentEvent::TokenReceived("hello".to_string()),
            AgentEvent::AutoCompactTriggered {
                current_tokens: 60000,
                threshold: 50000,
            },
            AgentEvent::TokenReceived(" world".to_string()),
            AgentEvent::ResponseComplete("hello world".to_string()),
        ];
        let (text, is_error) = run_json_collect(events, false).await;
        assert!(!is_error);
        assert_eq!(text, "hello world");
    }

    #[tokio::test]
    async fn non_tty_confirmation_required_sends_rejection_and_errors() {
        let events = vec![AgentEvent::ToolConfirmationRequired {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "rm -rf /"}),
            index: 1,
        }];
        let mut s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, mut rx) = make_confirm_channel();
        let result = run_text(&mut s, &mut buf, tx, false, &mut io::empty()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-TTY"));
        let response = rx.try_recv().expect("channel should have a value");
        assert_eq!(response, ConfirmationResponse::Rejected);
    }

    #[tokio::test]
    async fn tty_confirmation_with_y_sends_approved() {
        let events = vec![
            AgentEvent::ToolConfirmationRequired {
                id: "t1".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"command": "ls"}),
                index: 1,
            },
            AgentEvent::ResponseComplete(String::new()),
        ];
        let mut s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, mut rx) = make_confirm_channel();
        let mut fake_stdin = io::Cursor::new(b"y\n".as_ref());
        run_text(&mut s, &mut buf, tx, true, &mut fake_stdin)
            .await
            .expect("should succeed in TTY mode");
        let response = rx.try_recv().expect("channel should have a value");
        assert_eq!(response, ConfirmationResponse::Approved);
    }

    #[tokio::test]
    async fn tty_confirmation_with_n_sends_rejected() {
        let events = vec![
            AgentEvent::ToolConfirmationRequired {
                id: "t1".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"command": "ls"}),
                index: 1,
            },
            AgentEvent::ResponseComplete(String::new()),
        ];
        let mut s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, mut rx) = make_confirm_channel();
        let mut fake_stdin = io::Cursor::new(b"n\n".as_ref());
        run_text(&mut s, &mut buf, tx, true, &mut fake_stdin)
            .await
            .expect("should succeed in TTY mode");
        let response = rx.try_recv().expect("channel should have a value");
        assert_eq!(response, ConfirmationResponse::Rejected);
    }

    // --- JSON collect tests (backing run_json) ---

    #[tokio::test]
    async fn json_format_success_emits_is_error_false_and_result() {
        let events = vec![
            AgentEvent::TokenReceived("hello".to_string()),
            AgentEvent::TokenReceived(" world".to_string()),
            AgentEvent::ResponseComplete("hello world".to_string()),
        ];
        let (text, is_error) = run_json_collect(events, false).await;
        assert!(!is_error);
        assert_eq!(text, "hello world");
    }

    #[tokio::test]
    async fn json_format_error_emits_is_error_true_and_message() {
        let events = vec![AgentEvent::Error("something failed".to_string())];
        let (text, is_error) = run_json_collect(events, false).await;
        assert!(is_error);
        assert_eq!(text, "something failed");
    }

    #[tokio::test]
    async fn json_format_tool_events_discarded_only_final_turn_returned() {
        let events = vec![
            AgentEvent::ToolUseReceived {
                id: "t1".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"command": "ls"}),
                index: 1,
            },
            AgentEvent::ToolResult {
                name: "bash".to_string(),
                content: "file.txt".to_string(),
                is_error: false,
                index: 1,
            },
            AgentEvent::TokenReceived("done".to_string()),
            AgentEvent::ResponseComplete("done".to_string()),
        ];
        let (text, is_error) = run_json_collect(events, false).await;
        assert!(!is_error);
        assert_eq!(text, "done");
    }

    #[tokio::test]
    async fn json_format_multi_iteration_result_contains_only_final_turn() {
        let events = vec![
            AgentEvent::TokenReceived("I will use bash.".to_string()),
            AgentEvent::ToolUseReceived {
                id: "t1".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"command": "ls"}),
                index: 1,
            },
            AgentEvent::ToolResult {
                name: "bash".to_string(),
                content: "file.txt".to_string(),
                is_error: false,
                index: 1,
            },
            AgentEvent::TokenReceived("The directory contains file.txt.".to_string()),
            AgentEvent::ResponseComplete("The directory contains file.txt.".to_string()),
        ];
        let (text, is_error) = run_json_collect(events, false).await;
        assert!(!is_error);
        assert_eq!(text, "The directory contains file.txt.");
        assert!(!text.contains("I will use bash"));
    }

    #[tokio::test]
    async fn json_format_non_tty_confirmation_emits_json_envelope_and_errors() {
        // Regression: non-TTY confirmation in JSON mode must still emit a parseable
        // JSON envelope so callers can always parse stdout on any exit path.
        //
        // We exercise collect_response (the inner loop of run_json) directly with a
        // real ToolConfirmationRequired event in non-TTY mode, then verify that the
        // resulting (text, is_error) tuple is what run_json would write as a JSON envelope.
        let events = vec![AgentEvent::ToolConfirmationRequired {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "rm -rf /"}),
            index: 1,
        }];
        let mut s: BoxStream<AgentEvent> = Box::pin(futures::stream::iter(events));
        let (tx, mut rx) = mpsc::unbounded::<ConfirmationResponse>();

        let (result_text, is_error) = collect_response(&mut s, tx, false, &mut io::empty())
            .await
            .expect("collect_response should not propagate error");

        // collect_response sets is_error=true and sends Rejected in non-TTY mode.
        assert!(is_error);
        assert!(result_text.contains("non-TTY"));
        let response = rx.try_recv().expect("rejection should be sent");
        assert_eq!(response, ConfirmationResponse::Rejected);

        // Verify that the JSON envelope run_json would emit is valid and correct.
        let envelope = serde_json::json!({
            "is_error": is_error,
            "result": result_text,
        });
        let reparsed: serde_json::Value =
            serde_json::from_str(&envelope.to_string()).expect("envelope must be valid JSON");
        assert_eq!(reparsed["is_error"], true);
        assert!(
            reparsed["result"]
                .as_str()
                .expect("result is a string")
                .contains("non-TTY")
        );
    }

    // --- validate_json_output unit tests ---

    fn make_validator(schema_json: &str) -> jsonschema::Validator {
        let schema: serde_json::Value =
            serde_json::from_str(schema_json).expect("test schema must be valid JSON");
        jsonschema::validator_for(&schema).expect("test schema must compile")
    }

    #[test]
    fn validate_json_output_valid_returns_parsed_value() {
        let validator = make_validator(
            r#"{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}"#,
        );
        let result = validate_json_output(r#"{"name": "Alice"}"#, &validator);
        assert!(result.is_ok());
        assert_eq!(result.expect("valid")["name"], "Alice");
    }

    #[test]
    fn validate_json_output_not_json_returns_err() {
        let validator = make_validator(r#"{"type": "object"}"#);
        let result = validate_json_output("not json at all", &validator);
        assert!(result.is_err());
        assert!(result.expect_err("invalid").contains("not valid JSON"));
    }

    #[test]
    fn validate_json_output_schema_invalid_returns_err() {
        let validator = make_validator(
            r#"{"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}"#,
        );
        let result = validate_json_output(r#"{"age": 42}"#, &validator);
        assert!(result.is_err());
    }

    // --- Retry orchestration behavioral tests ---
    //
    // These tests call `apply_schema_retry` — the production function — via a real
    // `Agent` backed by `SequencedBackend`, which returns canned `StreamEvent`
    // sequences. If the retry loop in `apply_schema_retry` changes, these break.

    const PERSON_SCHEMA: &str =
        r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#;

    /// Build an `Arc<Agent>` whose backend returns `responses` in order, one
    /// `Vec<AgentEvent>` per `agent.send()` call.
    async fn make_sequenced_agent(responses: Vec<Vec<AgentEvent>>) -> Arc<Agent> {
        use crate::backend::LlmBackend;
        use crate::session::Session;
        use crate::types::{RequestConfig, StreamEvent};
        use async_trait::async_trait;

        struct SequencedBackend {
            responses: Arc<tokio::sync::Mutex<Vec<Vec<AgentEvent>>>>,
        }

        #[async_trait]
        impl LlmBackend for SequencedBackend {
            async fn send_message(
                &self,
                _: &[crate::types::Message],
                _: &RequestConfig,
            ) -> Result<BoxStream<Result<StreamEvent>>> {
                let mut lock = self.responses.lock().await;
                let events: Vec<AgentEvent> = if lock.is_empty() {
                    vec![AgentEvent::ResponseComplete(String::new())]
                } else {
                    lock.remove(0)
                };
                // Convert AgentEvents to StreamEvents for the backend layer.
                let stream_events: Vec<Result<StreamEvent>> = events
                    .into_iter()
                    .flat_map(|e| match e {
                        AgentEvent::TokenReceived(t) => {
                            vec![Ok(StreamEvent::TextDelta(t)), Ok(StreamEvent::Done)]
                        }
                        AgentEvent::ToolConfirmationRequired {
                            name, input, id, ..
                        } => {
                            // Emit a tool use that will trigger confirmation in agent.
                            // For simplicity encode as a text delta so collect_response sees it.
                            // Actually: emit Done — the agent handles confirmation upstream.
                            // We rely on collect_response's non-TTY branch in our test.
                            let _ = (name, input, id);
                            vec![Ok(StreamEvent::Done)]
                        }
                        AgentEvent::Error(msg) => vec![Err(anyhow::anyhow!(msg))],
                        _ => vec![Ok(StreamEvent::Done)],
                    })
                    .collect();
                Ok(Box::pin(futures::stream::iter(stream_events)))
            }
        }

        let backend = SequencedBackend {
            responses: Arc::new(tokio::sync::Mutex::new(responses)),
        };
        let dir = tempfile::TempDir::new().expect("temp dir");
        let session = std::sync::Arc::new(tokio::sync::Mutex::new(
            Session::new(None, dir.keep()).await.expect("test session"),
        ));
        let config = crate::types::RequestConfig {
            model: "test".to_string(),
            max_tokens: 1024,
            tools: vec![],
            thinking: None,
        };
        Arc::new(Agent::new(Box::new(backend), config, session).await)
    }

    fn token_events(text: &str) -> Vec<AgentEvent> {
        vec![
            AgentEvent::TokenReceived(text.to_string()),
            AgentEvent::ResponseComplete(text.to_string()),
        ]
    }

    async fn run_apply_schema_retry(
        schema_json: &str,
        max_retries: u32,
        responses: Vec<Vec<AgentEvent>>,
    ) -> Result<(serde_json::Value, bool)> {
        let schema_val: serde_json::Value =
            serde_json::from_str(schema_json).expect("test schema valid JSON");
        let validator = jsonschema::validator_for(&schema_val).expect("test schema compiles");
        let json_schema = JsonSchema {
            validator,
            raw: schema_json.to_string(),
        };

        // The first response is used as the initial result (simulating the first agent.send).
        let mut all = responses;
        let first_events = if all.is_empty() {
            vec![AgentEvent::ResponseComplete(String::new())]
        } else {
            all.remove(0)
        };

        // Collect initial response from stream.
        let mut first_stream: BoxStream<AgentEvent> = Box::pin(futures::stream::iter(first_events));
        let (tx, _rx) = mpsc::unbounded::<ConfirmationResponse>();
        let initial = collect_response(&mut first_stream, tx, false, &mut io::empty()).await?;

        // Build agent with remaining responses for retries.
        let agent = make_sequenced_agent(all).await;

        apply_schema_retry(
            initial,
            Some(&json_schema),
            max_retries,
            &agent,
            false,
            &mut io::empty(),
        )
        .await
    }
    #[tokio::test]
    async fn schema_valid_json_on_first_attempt_returns_parsed_value_as_result() {
        let (result, is_error) =
            run_apply_schema_retry(PERSON_SCHEMA, 3, vec![token_events(r#"{"name":"Alice"}"#)])
                .await
                .expect("should succeed");

        assert!(!is_error);
        assert_eq!(result["name"], "Alice");
    }

    #[tokio::test]
    async fn schema_invalid_json_on_first_attempt_reprompts_and_succeeds_on_second() {
        let (result, is_error) = run_apply_schema_retry(
            PERSON_SCHEMA,
            3,
            vec![
                token_events("not json at all"),
                token_events(r#"{"name":"Bob"}"#),
            ],
        )
        .await
        .expect("should succeed");

        assert!(!is_error);
        assert_eq!(result["name"], "Bob");
    }

    #[tokio::test]
    async fn schema_invalid_json_exhausts_retries_and_sets_is_error() {
        let (result, is_error) = run_apply_schema_retry(
            PERSON_SCHEMA,
            2,
            vec![
                token_events("not json"),
                token_events("still not json"),
                token_events("never valid"),
            ],
        )
        .await
        .expect("orchestration should not itself propagate error");

        assert!(is_error, "is_error must be true after exhausting retries");
        assert!(
            result
                .as_str()
                .expect("result should be a string on error")
                .contains("not valid JSON"),
            "result should contain validation error, got: {result}"
        );
    }

    #[tokio::test]
    async fn schema_parseable_but_schema_invalid_json_triggers_retry() {
        let (result, is_error) = run_apply_schema_retry(
            PERSON_SCHEMA,
            3,
            vec![
                token_events(r#"{"age": 42}"#),
                token_events(r#"{"name": "Carol"}"#),
            ],
        )
        .await
        .expect("should succeed");

        assert!(!is_error);
        assert_eq!(result["name"], "Carol");
    }

    #[tokio::test]
    async fn schema_max_retries_zero_validates_once_and_fails() {
        let (result, is_error) =
            run_apply_schema_retry(PERSON_SCHEMA, 0, vec![token_events("not json")])
                .await
                .expect("orchestration should not propagate error");

        assert!(is_error);
        assert!(
            result
                .as_str()
                .expect("result should be a string on error")
                .contains("not valid JSON")
        );
    }

    #[tokio::test]
    async fn no_schema_returns_result_text_unchanged() {
        let events = vec![
            AgentEvent::TokenReceived("plain text".to_string()),
            AgentEvent::ResponseComplete("plain text".to_string()),
        ];
        let (text, is_error) = run_json_collect(events, false).await;
        assert!(!is_error);
        assert_eq!(text, "plain text");
    }

    #[tokio::test]
    async fn is_error_true_from_initial_collect_skips_schema_validation() {
        // If the initial LLM stream errors, apply_schema_retry must not attempt validation.
        let schema_val: serde_json::Value =
            serde_json::from_str(PERSON_SCHEMA).expect("valid schema");
        let json_schema = JsonSchema {
            validator: jsonschema::validator_for(&schema_val).expect("compiles"),
            raw: PERSON_SCHEMA.to_string(),
        };

        let initial = ("stream error".to_string(), true);
        let agent = make_sequenced_agent(vec![]).await;

        let (result, is_error) = apply_schema_retry(
            initial,
            Some(&json_schema),
            3,
            &agent,
            false,
            &mut io::empty(),
        )
        .await
        .expect("should not propagate");

        assert!(is_error);
        assert_eq!(result, "stream error");
    }

    // ── Interrupted event ──────────────────────────────────────────────────

    #[tokio::test]
    async fn run_text_interrupted_writes_marker_and_returns_ok() {
        let events = vec![
            AgentEvent::TokenReceived("partial".to_string()),
            AgentEvent::Interrupted {
                partial_text: "partial".to_string(),
            },
        ];
        let (result, buf) = run_text_events(events, false).await;
        result.expect("run_text should return Ok on Interrupted");
        let output = String::from_utf8(buf).expect("valid utf8");
        assert!(
            output.contains("*(interrupted)*"),
            "output should contain interrupted marker; got: {output:?}"
        );
    }

    #[tokio::test]
    async fn collect_response_interrupted_sets_is_error_and_returns_partial_text() {
        let events = vec![
            AgentEvent::TokenReceived("partial".to_string()),
            AgentEvent::Interrupted {
                partial_text: "partial".to_string(),
            },
        ];
        let (text, is_error) = run_json_collect(events, false).await;
        assert!(
            is_error,
            "collect_response should set is_error=true for Interrupted"
        );
        assert_eq!(
            text, "partial",
            "collect_response should return partial text for Interrupted"
        );
    }
}
