use anyhow::Result;
use futures::StreamExt;
use futures::channel::mpsc;
use std::io::{self, BufRead, IsTerminal, Write};

use crate::logging::Logger;
use crate::types::{AgentEvent, BoxStream, ConfirmationResponse};

#[derive(Debug, Clone, PartialEq, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

pub async fn run(
    stream: BoxStream<AgentEvent>,
    confirm_tx: mpsc::UnboundedSender<ConfirmationResponse>,
    format: OutputFormat,
    logger: Option<&mut Logger>,
) -> Result<()> {
    let is_tty = std::io::stdin().is_terminal();
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let mut stdin = io::BufReader::new(std::io::stdin());
    run_with_writer(
        stream,
        &mut handle,
        confirm_tx,
        format,
        is_tty,
        &mut stdin,
        logger,
    )
    .await
}

async fn run_with_writer<W: Write, R: BufRead>(
    mut stream: BoxStream<AgentEvent>,
    writer: &mut W,
    confirm_tx: mpsc::UnboundedSender<ConfirmationResponse>,
    format: OutputFormat,
    is_tty: bool,
    stdin: &mut R,
    mut logger: Option<&mut Logger>,
) -> Result<()> {
    match format {
        OutputFormat::Text => {
            run_text(&mut stream, writer, confirm_tx, is_tty, stdin, &mut logger).await
        }
        OutputFormat::Json => {
            run_json(&mut stream, writer, confirm_tx, is_tty, stdin, &mut logger).await
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
                handle_confirmation(writer, confirm_tx.clone(), is_tty, stdin, &name, &input)?;
            }
        }
    }
    Ok(())
}

async fn run_json<W: Write, R: BufRead>(
    stream: &mut BoxStream<AgentEvent>,
    writer: &mut W,
    confirm_tx: mpsc::UnboundedSender<ConfirmationResponse>,
    is_tty: bool,
    stdin: &mut R,
    logger: &mut Option<&mut Logger>,
) -> Result<()> {
    let mut result_text = String::new();
    let mut is_error = false;

    while let Some(event) = stream.next().await {
        if let Some(log) = logger.as_deref_mut() {
            log.log_event(&event)?;
            log.flush()?;
        }
        match event {
            AgentEvent::TokenReceived(text) => {
                result_text.push_str(&text);
            }
            AgentEvent::ResponseComplete(_) => {
                break;
            }
            AgentEvent::Error(msg) => {
                is_error = true;
                result_text = msg;
                break;
            }
            AgentEvent::ToolUseReceived { .. }
            | AgentEvent::ToolResult { .. }
            | AgentEvent::Usage { .. } => {}
            AgentEvent::ToolConfirmationRequired { name, input, .. } => {
                handle_confirmation(writer, confirm_tx.clone(), is_tty, stdin, &name, &input)?;
            }
        }
    }

    let output = serde_json::json!({
        "is_error": is_error,
        "result": result_text,
    });
    writeln!(writer, "{}", output)?;
    writer.flush()?;

    if is_error {
        anyhow::bail!("LLM error: {}", result_text);
    }

    Ok(())
}

fn handle_confirmation<W: Write, R: BufRead>(
    writer: &mut W,
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
    let _ = writer;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::channel::mpsc;
    use futures::stream;

    fn make_confirm_channel() -> (
        mpsc::UnboundedSender<ConfirmationResponse>,
        mpsc::UnboundedReceiver<ConfirmationResponse>,
    ) {
        mpsc::unbounded()
    }

    // --- Text format tests ---

    #[tokio::test]
    async fn token_received_writes_text_to_writer() {
        let events = vec![
            AgentEvent::TokenReceived("hello".to_string()),
            AgentEvent::TokenReceived(" world".to_string()),
            AgentEvent::ResponseComplete("hello world".to_string()),
        ];
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, _rx) = make_confirm_channel();

        run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Text,
            false,
            &mut io::empty(),
            None,
        )
        .await
        .expect("stdout run should succeed");

        assert_eq!(
            String::from_utf8(buf).expect("buffer should contain valid UTF-8"),
            "hello world\n"
        );
    }

    #[tokio::test]
    async fn response_complete_writes_trailing_newline_and_stops() {
        let events = vec![
            AgentEvent::ResponseComplete("done".to_string()),
            AgentEvent::TokenReceived("should not appear".to_string()),
        ];
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, _rx) = make_confirm_channel();

        run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Text,
            false,
            &mut io::empty(),
            None,
        )
        .await
        .expect("stdout run should succeed");

        assert_eq!(
            String::from_utf8(buf).expect("buffer should contain valid UTF-8"),
            "\n"
        );
    }

    #[tokio::test]
    async fn error_returns_err() {
        let events = vec![AgentEvent::Error("something went wrong".to_string())];
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, _rx) = make_confirm_channel();

        let result = run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Text,
            false,
            &mut io::empty(),
            None,
        )
        .await;

        assert!(result.is_err());
        assert!(
            result
                .expect_err("run should return an error on AgentEvent::Error")
                .to_string()
                .contains("something went wrong")
        );
    }

    #[tokio::test]
    async fn tool_use_received_prints_tool_name_and_input() {
        let input = serde_json::json!({"command": "ls"});
        let events = vec![
            AgentEvent::ToolUseReceived {
                id: "t1".to_string(),
                name: "bash".to_string(),
                input: input.clone(),
            },
            AgentEvent::ResponseComplete(String::new()),
        ];
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, _rx) = make_confirm_channel();

        run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Text,
            false,
            &mut io::empty(),
            None,
        )
        .await
        .expect("should succeed");

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
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, _rx) = make_confirm_channel();

        run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Text,
            false,
            &mut io::empty(),
            None,
        )
        .await
        .expect("should succeed");

        let output = String::from_utf8(buf).expect("valid UTF-8");
        assert!(
            output.contains("[result from bash]"),
            "should contain result label"
        );
        assert!(
            output.contains("file1.txt"),
            "should contain result content"
        );
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
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, _rx) = make_confirm_channel();

        run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Text,
            false,
            &mut io::empty(),
            None,
        )
        .await
        .expect("should succeed");

        let output = String::from_utf8(buf).expect("valid UTF-8");
        assert!(
            output.contains("[error from bash]"),
            "should contain error label"
        );
        assert!(
            output.contains("permission denied"),
            "should contain error content"
        );
    }

    #[tokio::test]
    async fn non_tty_confirmation_required_sends_rejection_and_errors() {
        let events = vec![AgentEvent::ToolConfirmationRequired {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "rm -rf /"}),
        }];
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, mut rx) = make_confirm_channel();

        let result = run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Text,
            false,
            &mut io::empty(),
            None,
        )
        .await;

        assert!(result.is_err(), "should error in non-TTY mode");
        assert!(
            result.unwrap_err().to_string().contains("non-TTY"),
            "error should mention non-TTY"
        );

        let response = rx.try_recv().expect("channel should have a value");
        assert_eq!(
            response,
            ConfirmationResponse::Rejected,
            "should send Rejected"
        );
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
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, mut rx) = make_confirm_channel();
        let mut fake_stdin = io::Cursor::new(b"y\n".as_ref());

        run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Text,
            true,
            &mut fake_stdin,
            None,
        )
        .await
        .expect("should succeed in TTY mode");

        let response = rx.try_recv().expect("channel should have a value");
        assert_eq!(
            response,
            ConfirmationResponse::Approved,
            "should send Approved for 'y'"
        );
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
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, mut rx) = make_confirm_channel();
        let mut fake_stdin = io::Cursor::new(b"n\n".as_ref());

        run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Text,
            true,
            &mut fake_stdin,
            None,
        )
        .await
        .expect("should succeed in TTY mode");

        let response = rx.try_recv().expect("channel should have a value");
        assert_eq!(
            response,
            ConfirmationResponse::Rejected,
            "should send Rejected for 'n'"
        );
    }

    // --- JSON format tests ---

    #[tokio::test]
    async fn json_format_success_emits_is_error_false_and_result() {
        let events = vec![
            AgentEvent::TokenReceived("hello".to_string()),
            AgentEvent::TokenReceived(" world".to_string()),
            AgentEvent::ResponseComplete("hello world".to_string()),
        ];
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, _rx) = make_confirm_channel();

        run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Json,
            false,
            &mut io::empty(),
            None,
        )
        .await
        .expect("json run should succeed");

        let output = String::from_utf8(buf).expect("valid UTF-8");
        let parsed: serde_json::Value = serde_json::from_str(output.trim()).expect("valid JSON");
        assert_eq!(parsed["is_error"], false);
        assert_eq!(parsed["result"], "hello world");
    }

    #[tokio::test]
    async fn json_format_error_emits_is_error_true_and_message() {
        let events = vec![AgentEvent::Error("something failed".to_string())];
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, _rx) = make_confirm_channel();

        let result = run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Json,
            false,
            &mut io::empty(),
            None,
        )
        .await;

        assert!(result.is_err(), "should return Err on error event");

        let output = String::from_utf8(buf).expect("valid UTF-8");
        let parsed: serde_json::Value = serde_json::from_str(output.trim()).expect("valid JSON");
        assert_eq!(parsed["is_error"], true);
        assert_eq!(parsed["result"], "something failed");
    }

    #[tokio::test]
    async fn json_format_tool_events_are_not_written_to_output() {
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
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, _rx) = make_confirm_channel();

        run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Json,
            false,
            &mut io::empty(),
            None,
        )
        .await
        .expect("should succeed");

        let output = String::from_utf8(buf).expect("valid UTF-8");
        let parsed: serde_json::Value = serde_json::from_str(output.trim()).expect("valid JSON");
        assert_eq!(parsed["is_error"], false);
        assert_eq!(parsed["result"], "done");
        assert!(
            !output.contains("[tool:"),
            "tool events should not appear in JSON output"
        );
    }

    #[tokio::test]
    async fn json_format_non_tty_confirmation_sends_rejection_and_errors() {
        let events = vec![AgentEvent::ToolConfirmationRequired {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "rm -rf /"}),
        }];
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();
        let (tx, mut rx) = make_confirm_channel();

        let result = run_with_writer(
            s,
            &mut buf,
            tx,
            OutputFormat::Json,
            false,
            &mut io::empty(),
            None,
        )
        .await;

        assert!(result.is_err(), "should error in non-TTY mode");
        let response = rx.try_recv().expect("channel should have a value");
        assert_eq!(response, ConfirmationResponse::Rejected);
    }
}
