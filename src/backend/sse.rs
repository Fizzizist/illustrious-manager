use std::collections::VecDeque;

use anyhow::Result;
use futures::stream::unfold;
use futures::{Stream, StreamExt};

use crate::types::{BoxStream, StreamEvent};

pub fn extract_sse_data(event_text: &str) -> Option<&str> {
    for line in event_text.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            return Some(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    None
}

fn find_frame_end(buffer: &str) -> Option<(usize, usize)> {
    let lf = buffer.find("\n\n").map(|pos| (pos, 2));
    let crlf = buffer.find("\r\n\r\n").map(|pos| (pos, 4));
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (a, b) => a.or(b),
    }
}

pub fn create_sse_event_stream<S, B, E, P>(
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
                // immediately yield queued events
                if let Some(event) = event_queue.pop_front() {
                    return Some((Ok(event), (byte_stream, buffer, parser, event_queue)));
                }

                if let Some((pos, delimiter_len)) = find_frame_end(&buffer) {
                    let event_text = buffer[..pos].to_string();
                    buffer = buffer[pos + delimiter_len..].to_string();

                    if let Some(data) = extract_sse_data(&event_text) {
                        match parser(data) {
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

                // no completed frames? read more bytes
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
    fn extract_sse_data_returns_json_after_data_prefix_without_space() {
        let event = "event: content_block_delta\ndata:{\"type\":\"content_block_delta\"}";
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

    fn echo_stream(chunks: Vec<&'static str>) -> BoxStream<Result<StreamEvent>> {
        let byte_stream = futures::stream::iter(chunks.into_iter().map(Ok::<_, std::io::Error>));
        create_sse_event_stream(byte_stream, |data| {
            Ok(vec![StreamEvent::TextDelta(data.to_string())])
        })
    }

    async fn collect_texts(stream: BoxStream<Result<StreamEvent>>) -> Vec<String> {
        stream
            .map(|event| match event.expect("event should be Ok") {
                StreamEvent::TextDelta(t) => t,
                other => panic!("expected TextDelta, got {other:?}"),
            })
            .collect()
            .await
    }

    #[tokio::test]
    async fn crlf_delimited_frames_yield_one_event_each() {
        let stream = echo_stream(vec![
            "data: one\r\n\r\ndata: two\r\n\r\ndata: three\r\n\r\n",
        ]);
        let texts = collect_texts(stream).await;
        assert_eq!(texts, vec!["one", "two", "three"]);
    }

    #[tokio::test]
    async fn crlf_delimiter_split_across_chunks_still_parses() {
        let stream = echo_stream(vec!["data: one\r\n", "\r", "\ndata: two\r\n\r\n"]);
        let texts = collect_texts(stream).await;
        assert_eq!(texts, vec!["one", "two"]);
    }
}
