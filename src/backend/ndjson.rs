use std::collections::VecDeque;

use anyhow::Result;
use futures::stream::unfold;
use futures::{Stream, StreamExt};

use crate::types::{BoxStream, StreamEvent};

pub fn create_ndjson_event_stream<S, B, E, P>(
    byte_stream: S,
    parser: P,
) -> BoxStream<Result<StreamEvent>>
where
    S: Stream<Item = Result<B, E>> + Unpin + Send + 'static,
    B: AsRef<[u8]>,
    E: std::fmt::Display + 'static,
    P: FnMut(&str) -> Result<Vec<StreamEvent>> + Send + 'static,
{
    let event_stream = unfold(
        (byte_stream, String::new(), parser, VecDeque::new()),
        move |(mut byte_stream, mut buffer, mut parser, mut event_queue)| async move {
            loop {
                if let Some(event) = event_queue.pop_front() {
                    return Some((Ok(event), (byte_stream, buffer, parser, event_queue)));
                }

                if let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if !line.is_empty() {
                        match parser(&line) {
                            Ok(events) => {
                                event_queue.extend(events);
                                continue;
                            }
                            Err(e) => {
                                return Some((Err(e), (byte_stream, buffer, parser, event_queue)));
                            }
                        }
                    }
                    continue;
                }

                match byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(bytes.as_ref()));
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(anyhow::anyhow!("Stream read error: {}", e)),
                            (byte_stream, buffer, parser, event_queue),
                        ));
                    }
                    None => {
                        if !buffer.trim().is_empty() {
                            let remaining = buffer.trim().to_string();
                            buffer.clear();
                            match parser(&remaining) {
                                Ok(events) => {
                                    event_queue.extend(events);
                                    continue;
                                }
                                Err(e) => {
                                    return Some((
                                        Err(e),
                                        (byte_stream, buffer, parser, event_queue),
                                    ));
                                }
                            }
                        }
                        return None;
                    }
                }
            }
        },
    );

    Box::pin(event_stream)
}

#[cfg(test)]
mod tests {
    use futures::stream;

    use super::*;

    fn text_event(text: &str) -> StreamEvent {
        StreamEvent::TextDelta(text.to_string())
    }

    #[tokio::test]
    async fn single_line_produces_single_event() {
        let chunks: Vec<Result<&[u8], std::io::Error>> = vec![Ok(b"{\"msg\":\"hi\"}\n")];
        let byte_stream = stream::iter(chunks);
        let events: Vec<_> =
            create_ndjson_event_stream(byte_stream, |line| Ok(vec![text_event(line)]))
                .collect::<Vec<_>>()
                .await;

        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].as_ref().unwrap(),
            StreamEvent::TextDelta(t) if t.contains("msg")
        ));
    }

    #[tokio::test]
    async fn multi_line_in_one_chunk_produces_multiple_events() {
        let data = "{\"a\":1}\n{\"b\":2}\n";
        let chunks: Vec<Result<&[u8], std::io::Error>> = vec![Ok(data.as_bytes())];
        let byte_stream = stream::iter(chunks);
        let events: Vec<_> =
            create_ndjson_event_stream(byte_stream, |line| Ok(vec![text_event(line)]))
                .collect::<Vec<_>>()
                .await;

        assert_eq!(events.len(), 2);
    }

    #[tokio::test]
    async fn partial_line_buffered_across_chunks() {
        let chunks: Vec<Result<&[u8], std::io::Error>> = vec![Ok(b"{\"part"), Ok(b"ial\":1}\n")];
        let byte_stream = stream::iter(chunks);
        let events: Vec<_> =
            create_ndjson_event_stream(byte_stream, |line| Ok(vec![text_event(line)]))
                .collect::<Vec<_>>()
                .await;

        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].as_ref().unwrap(),
            StreamEvent::TextDelta(t) if t == "{\"partial\":1}"
        ));
    }

    #[tokio::test]
    async fn trailing_line_without_newline_still_emits() {
        let chunks: Vec<Result<&[u8], std::io::Error>> = vec![Ok(b"{\"final\":true}")];
        let byte_stream = stream::iter(chunks);
        let events: Vec<_> =
            create_ndjson_event_stream(byte_stream, |line| Ok(vec![text_event(line)]))
                .collect::<Vec<_>>()
                .await;

        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].as_ref().unwrap(),
            StreamEvent::TextDelta(t) if t == "{\"final\":true}"
        ));
    }

    #[tokio::test]
    async fn parser_error_propagates() {
        let chunks: Vec<Result<&[u8], std::io::Error>> = vec![Ok(b"bad\n")];
        let byte_stream = stream::iter(chunks);
        let events: Vec<_> =
            create_ndjson_event_stream(byte_stream, |_line| Err(anyhow::anyhow!("parse failure")))
                .collect::<Vec<_>>()
                .await;

        assert_eq!(events.len(), 1);
        assert!(events[0].as_ref().is_err());
        assert!(
            events[0]
                .as_ref()
                .unwrap_err()
                .to_string()
                .contains("parse failure")
        );
    }
}
