use std::collections::VecDeque;

use anyhow::Result;
use futures::stream::unfold;
use futures::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use crate::types::{BoxStream, StreamEvent};

pub fn extract_sse_data(event_text: &str) -> Option<&str> {
    for line in event_text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            return Some(data);
        }
    }
    None
}

pub fn create_sse_event_stream<S, B, E, P>(
    byte_stream: S,
    cancel_token: Option<CancellationToken>,
    parser: P,
) -> BoxStream<Result<StreamEvent>>
where
    S: Stream<Item = Result<B, E>> + Unpin + Send + 'static,
    B: AsRef<[u8]>,
    E: std::fmt::Display + 'static,
    P: FnMut(&str) -> Result<Vec<StreamEvent>> + Send + 'static,
{
    let event_stream = unfold(
        (
            byte_stream,
            String::new(),
            parser,
            VecDeque::new(),
            cancel_token,
        ),
        move |(mut byte_stream, mut buffer, mut parser, mut event_queue, cancel_token)| async move {
            loop {
                // immediately yield queued events
                if let Some(event) = event_queue.pop_front() {
                    return Some((
                        Ok(event),
                        (
                            byte_stream,
                            buffer,
                            parser,
                            event_queue,
                            cancel_token.clone(),
                        ),
                    ));
                }

                // LF new SSE frames
                if let Some(pos) = buffer.find("\n\n") {
                    let event_text = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    if let Some(data) = extract_sse_data(&event_text) {
                        match parser(data) {
                            Ok(events) => {
                                event_queue.extend(events);
                                continue;
                            }
                            Err(e) => {
                                return Some((
                                    Err(e),
                                    (
                                        byte_stream,
                                        buffer,
                                        parser,
                                        event_queue,
                                        cancel_token.clone(),
                                    ),
                                ));
                            }
                        }
                    }
                    continue;
                }

                // no completed frames? read more bytes
                let next = if let Some(ref token) = cancel_token {
                    tokio::select! {
                        biased;
                        _ = token.cancelled() => {
                            return None;
                        }
                        item = byte_stream.next() => item,
                    }
                } else {
                    byte_stream.next().await
                };
                match next {
                    Some(Ok(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(bytes.as_ref()));
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(anyhow::anyhow!("Stream read error: {}", e)),
                            (
                                byte_stream,
                                buffer,
                                parser,
                                event_queue,
                                cancel_token.clone(),
                            ),
                        ));
                    }
                    None => {
                        if !buffer.trim().is_empty()
                            && let Some(data) = extract_sse_data(&buffer).map(str::to_owned)
                        {
                            buffer.clear();
                            match parser(&data) {
                                Ok(events) => {
                                    event_queue.extend(events);
                                    continue;
                                }
                                Err(e) => {
                                    return Some((
                                        Err(e),
                                        (
                                            byte_stream,
                                            buffer,
                                            parser,
                                            event_queue,
                                            cancel_token.clone(),
                                        ),
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
    use super::*;

    #[test]
    fn extract_sse_data_returns_json_after_data_prefix() {
        let event = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}";
        assert_eq!(
            extract_sse_data(event),
            Some("{\"type\":\"content_block_delta\"}")
        );
    }

    #[test]
    fn extract_sse_data_returns_none_when_no_data_line() {
        let event = "event: content_block_delta\n";
        assert!(extract_sse_data(event).is_none());
    }

    #[test]
    fn extract_sse_data_returns_first_data_line_when_multiple_present() {
        let event = "data: first\ndata: second";
        assert_eq!(extract_sse_data(event), Some("first"));
    }

    #[tokio::test]
    async fn sse_stream_terminates_on_cancellation() {
        use futures::channel::mpsc;

        let cancel = CancellationToken::new();
        let (tx, rx) = mpsc::unbounded::<Result<&[u8], std::io::Error>>();

        // Send one event, then leave the channel open.
        tx.unbounded_send(Ok(b"data: hello\n\n")).expect("send");

        cancel.cancel();

        let stream = create_sse_event_stream(rx, Some(cancel), |data| {
            Ok(vec![StreamEvent::TextDelta(data.to_string())])
        });

        let events: Vec<_> = stream.collect::<Vec<_>>().await;
        // When pre-cancelled, the stream aborts before reading more bytes.
        assert!(
            events.len() <= 1,
            "pre-cancelled stream should abort without draining buffered data; got {} events",
            events.len()
        );
    }

    #[tokio::test]
    async fn sse_stream_completes_normally_without_cancellation() {
        use futures::channel::mpsc;

        let (tx, rx) = mpsc::unbounded::<Result<&[u8], std::io::Error>>();

        tx.unbounded_send(Ok(b"data: hello\n\n")).expect("send");
        tx.unbounded_send(Ok(b"data: world\n\n")).expect("send");
        drop(tx);

        let stream = create_sse_event_stream(rx, None, |data| {
            Ok(vec![StreamEvent::TextDelta(data.to_string())])
        });

        let events: Vec<_> = stream.collect::<Vec<_>>().await;
        assert!(
            events.len() >= 2,
            "should receive both events; got {}",
            events.len()
        );
    }
}
