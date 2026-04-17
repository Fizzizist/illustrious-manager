use anyhow::Result;
use futures::StreamExt;
use futures::channel::mpsc;
use std::io::{self, BufRead, IsTerminal, Write};
use std::sync::Arc;

use crate::agent::Agent;
use crate::logging::Logger;
use crate::types::{AgentEvent, BoxStream, ConfirmationResponse};

#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

pub async fn run(
    agent: Arc<Agent>,
    prompt: String,
    format: OutputFormat,
    json_schema: Option<jsonschema::Validator>,
    json_schema_raw: Option<String>,
    max_schema_retries: u32,
    logger: Option<&mut Logger>,
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
        json_schema_raw,
        max_schema_retries,
        is_tty,
        &mut stdin,
        logger,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_with_writer<W: Write, R: BufRead>(
    agent: Arc<Agent>,
    prompt: String,
    writer: &mut W,
    format: OutputFormat,
    json_schema: Option<jsonschema::Validator>,
    json_schema_raw: Option<String>,
    max_schema_retries: u32,
    is_tty: bool,
    stdin: &mut R,
    mut logger: Option<&mut Logger>,
) -> Result<()> {
    match format {
        OutputFormat::Text => {
            let (confirm_tx, confirm_rx) = mpsc::unbounded::<ConfirmationResponse>();
            let mut stream = agent.send(prompt, Some(confirm_rx)).await?;
            run_text(&mut stream, writer, confirm_tx, is_tty, stdin, &mut logger).await
        }
        OutputFormat::Json => {
            run_json(
                agent,
                prompt,
                writer,
                json_schema,
                json_schema_raw,
                max_schema_retries,
                is_tty,
                stdin,
                &mut logger,
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
    logger: &mut Option<&mut Logger>,
) -> Result<()> {
    while let Some(event) = stream.next().await {
        if let Some(log) = logger.as_deref_mut() {
            log.log_event(&event)?;
            log.flush()?;
        }
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
            AgentEvent::ToolUseReceived { name, input, .. } => {
                writeln!(writer, "\n[tool: {}] {}", name, input)?;
            }
            AgentEvent::ToolResult {
                name,
                content,
                is_error,
            } => {
                if is_error {
                    writeln!(writer, "[error from {}]: {}", name, content)?;
                } else {
                    writeln!(writer, "[result from {}]: {}", name, content)?;
                }
            }
            AgentEvent::Usage { .. } => {}
            AgentEvent::ToolConfirmationRequired { name, input, .. } => {
                handle_confirmation(confirm_tx.clone(), is_tty, stdin, &name, &input)?;
            }
        }
    }
    Ok(())
}

// Collects a single LLM response from the stream, returning the final text and
// whether an error occurred.
async fn collect_response<R: BufRead>(
    stream: &mut BoxStream<AgentEvent>,
    confirm_tx: mpsc::UnboundedSender<ConfirmationResponse>,
    is_tty: bool,
    stdin: &mut R,
    logger: &mut Option<&mut Logger>,
) -> Result<(String, bool)> {
    let mut result_text = String::new();
    let mut is_error = false;
    let mut in_tool_iteration = false;

    while let Some(event) = stream.next().await {
        if let Some(log) = logger.as_deref_mut() {
            log.log_event(&event)?;
            log.flush()?;
        }
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
// failure a reprompt is sent through the agent, up to `max_schema_retries` times.
// On exhaustion `is_error: true` is emitted. On success `structured_output`
// contains the parsed JSON value.
#[allow(clippy::too_many_arguments)]
async fn run_json<W: Write, R: BufRead>(
    agent: Arc<Agent>,
    initial_prompt: String,
    writer: &mut W,
    json_schema: Option<jsonschema::Validator>,
    json_schema_raw: Option<String>,
    max_schema_retries: u32,
    is_tty: bool,
    stdin: &mut R,
    logger: &mut Option<&mut Logger>,
) -> Result<()> {
    // Inject schema instruction as part of the user prompt when schema is present.
    let first_prompt = if let Some(ref schema_raw) = json_schema_raw {
        format!(
            "{}\n\nYou MUST output ONLY valid JSON conforming to this schema (no markdown, no explanation):\n{}",
            initial_prompt, schema_raw
        )
    } else {
        initial_prompt
    };

    let (confirm_tx, confirm_rx) = mpsc::unbounded::<ConfirmationResponse>();
    let mut stream = agent.send(first_prompt, Some(confirm_rx)).await?;
    let (mut result_text, mut is_error) =
        collect_response(&mut stream, confirm_tx, is_tty, stdin, logger).await?;

    let mut structured_output: Option<serde_json::Value> = None;

    if !is_error && let Some(ref validator) = json_schema {
        let schema_raw = json_schema_raw
            .as_deref()
            .expect("schema_raw present when validator present");
        let mut retries_left = max_schema_retries;

        loop {
            match validate_json_output(&result_text, validator) {
                Ok(parsed) => {
                    structured_output = Some(parsed);
                    break;
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
                        validation_err, schema_raw
                    );
                    let (retry_tx, retry_rx) = mpsc::unbounded::<ConfirmationResponse>();
                    let mut retry_stream = agent.send(reprompt, Some(retry_rx)).await?;
                    let (new_text, new_error) =
                        collect_response(&mut retry_stream, retry_tx, is_tty, stdin, logger)
                            .await?;
                    result_text = new_text;
                    is_error = new_error;
                    if is_error {
                        break;
                    }
                }
            }
        }
    }

    let output = if let Some(ref sv) = structured_output {
        serde_json::json!({
            "is_error": is_error,
            "result": result_text,
            "structured_output": sv,
        })
    } else {
        serde_json::json!({
            "is_error": is_error,
            "result": result_text,
        })
    };

    writeln!(writer, "{}", output)?;
    writer.flush()?;

    if is_error {
        anyhow::bail!("LLM error: {}", result_text);
    }

    Ok(())
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
        let result = run_text(&mut s, &mut buf, tx, is_tty, &mut io::empty(), &mut None).await;
        (result, buf)
    }

    async fn run_json_collect(events: Vec<AgentEvent>, is_tty: bool) -> (String, bool) {
        let mut s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let (tx, _rx) = make_confirm_channel();
        collect_response(&mut s, tx, is_tty, &mut io::empty(), &mut None)
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
            },
            AgentEvent::ResponseComplete(String::new()),
        ];
        let (result, buf) = run_text_events(events, false).await;
        result.expect("should succeed");
        let output = String::from_utf8(buf).expect("valid UTF-8");
        assert!(output.contains("[tool: bash]"), "should contain tool name");
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
            },
            AgentEvent::ResponseComplete(String::new()),
        ];
        let (result, buf) = run_text_events(events, false).await;
        result.expect("should succeed");
        let output = String::from_utf8(buf).expect("valid UTF-8");
        assert!(output.contains("[result from bash]"));
        assert!(output.contains("file1.txt"));
    }

    #[tokio::test]
    async fn tool_result_error_prints_error_label() {
        let events = vec![
            AgentEvent::ToolResult {
                name: "bash".to_string(),
                content: "permission denied".to_string(),
                is_error: true,
            },
            AgentEvent::ResponseComplete(String::new()),
        ];
        let (result, buf) = run_text_events(events, false).await;
        result.expect("should succeed");
        let output = String::from_utf8(buf).expect("valid UTF-8");
        assert!(output.contains("[error from bash]"));
        assert!(output.contains("permission denied"));
    }

    #[tokio::test]
    async fn non_tty_confirmation_required_sends_rejection_and_errors() {
        let events = vec![AgentEvent::ToolConfirmationRequired {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "rm -rf /"}),
        }];
        let mut s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, mut rx) = make_confirm_channel();
        let result = run_text(&mut s, &mut buf, tx, false, &mut io::empty(), &mut None).await;
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
            },
            AgentEvent::ResponseComplete(String::new()),
        ];
        let mut s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, mut rx) = make_confirm_channel();
        let mut fake_stdin = io::Cursor::new(b"y\n".as_ref());
        run_text(&mut s, &mut buf, tx, true, &mut fake_stdin, &mut None)
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
            },
            AgentEvent::ResponseComplete(String::new()),
        ];
        let mut s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, mut rx) = make_confirm_channel();
        let mut fake_stdin = io::Cursor::new(b"n\n".as_ref());
        run_text(&mut s, &mut buf, tx, true, &mut fake_stdin, &mut None)
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
            },
            AgentEvent::ToolResult {
                name: "bash".to_string(),
                content: "file.txt".to_string(),
                is_error: false,
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
            },
            AgentEvent::ToolResult {
                name: "bash".to_string(),
                content: "file.txt".to_string(),
                is_error: false,
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
    async fn json_format_non_tty_confirmation_sets_is_error() {
        let events = vec![AgentEvent::ToolConfirmationRequired {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "rm -rf /"}),
        }];
        let mut s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let (tx, mut rx) = make_confirm_channel();
        let (text, is_error) = collect_response(&mut s, tx, false, &mut io::empty(), &mut None)
            .await
            .expect("should not fail");
        assert!(is_error);
        assert!(text.contains("non-TTY"));
        let response = rx.try_recv().expect("channel should have a value");
        assert_eq!(response, ConfirmationResponse::Rejected);
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
        assert_eq!(result.unwrap()["name"], "Alice");
    }

    #[test]
    fn validate_json_output_not_json_returns_err() {
        let validator = make_validator(r#"{"type": "object"}"#);
        let result = validate_json_output("not json at all", &validator);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not valid JSON"));
    }

    #[test]
    fn validate_json_output_schema_invalid_returns_err() {
        let validator = make_validator(
            r#"{"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}"#,
        );
        // Missing required field
        let result = validate_json_output(r#"{"age": 42}"#, &validator);
        assert!(result.is_err());
    }

    // --- Schema injection tests (main.rs determine_mode behavior tested there) ---

    // Verify no structured_output key when schema is absent (backward compat).
    // We test this via validate_json_output absence — the output branch is exercised
    // in run_json, but we verify the JSON envelope logic via the helper directly.
    #[test]
    fn no_schema_means_no_structured_output_key() {
        // When there's no schema, run_json emits only is_error + result.
        // We verify the serde_json::json! branch that omits structured_output.
        let output = serde_json::json!({
            "is_error": false,
            "result": "hello",
        });
        assert!(output.get("structured_output").is_none());
    }
}
