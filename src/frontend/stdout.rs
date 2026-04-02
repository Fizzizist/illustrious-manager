use anyhow::Result;
use futures::StreamExt;
use std::io::{self, Write};

use crate::types::{AgentEvent, BoxStream};

pub async fn run(stream: BoxStream<AgentEvent>) -> Result<()> {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    run_with_writer(stream, &mut handle).await
}

async fn run_with_writer<W: Write>(
    mut stream: BoxStream<AgentEvent>,
    writer: &mut W,
) -> Result<()> {
    while let Some(event) = stream.next().await {
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
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[tokio::test]
    async fn token_received_writes_text_to_writer() {
        let events = vec![
            AgentEvent::TokenReceived("hello".to_string()),
            AgentEvent::TokenReceived(" world".to_string()),
            AgentEvent::ResponseComplete("hello world".to_string()),
        ];
        let s: BoxStream<AgentEvent> = Box::pin(stream::iter(events));
        let mut buf = Vec::new();

        run_with_writer(s, &mut buf)
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

        run_with_writer(s, &mut buf)
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

        let result = run_with_writer(s, &mut buf).await;

        assert!(result.is_err());
        assert!(
            result
                .expect_err("run should return an error on AgentEvent::Error")
                .to_string()
                .contains("something went wrong")
        );
    }
}
