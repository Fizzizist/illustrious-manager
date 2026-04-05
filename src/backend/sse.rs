use anyhow::Result;
use futures::stream::unfold;
use futures::{Stream, StreamExt};

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
    parser: P,
) -> BoxStream<Result<StreamEvent>>
where
    S: Stream<Item = Result<B, E>> + Unpin + Send + 'static,
    B: AsRef<[u8]>,
    E: std::fmt::Display + 'static,
    P: FnMut(&str) -> Result<Option<StreamEvent>> + Send + 'static,
{
    let event_stream = unfold(
        (byte_stream, String::new(), parser),
        move |(mut byte_stream, mut buffer, mut parser)| async move {
            loop {
                if let Some(pos) = buffer.find("\n\n") {
                    let event_text = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    if let Some(data) = extract_sse_data(&event_text) {
                        match parser(data) {
                            Ok(Some(event)) => {
                                return Some((Ok(event), (byte_stream, buffer, parser)));
                            }
                            Ok(None) => continue,
                            Err(e) => return Some((Err(e), (byte_stream, buffer, parser))),
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
                            (byte_stream, buffer, parser),
                        ));
                    }
                    None => {
                        if !buffer.trim().is_empty()
                            && let Some(data) = extract_sse_data(&buffer).map(str::to_owned)
                        {
                            buffer.clear();
                            match parser(&data) {
                                Ok(Some(event)) => {
                                    return Some((Ok(event), (byte_stream, buffer, parser)));
                                }
                                Ok(None) => return None,
                                Err(e) => return Some((Err(e), (byte_stream, buffer, parser))),
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
}
