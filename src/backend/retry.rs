use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tokio_retry2::RetryError;
use tokio_retry2::strategy::{ExponentialBackoff, jitter};

use super::LlmBackend;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

/// Initial backoff delay for 429 retries.
const INITIAL_DELAY_MS: u64 = 1_000;
/// Maximum backoff delay cap.
const MAX_DELAY_MS: u64 = 60_000;
/// Maximum number of retry attempts after the initial try.
const MAX_RETRIES: usize = 5;

/// Returns true if the error message indicates an HTTP 429 Too Many Requests.
pub(crate) fn is_rate_limit_error(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    // Both Vertex and z.ai backends format errors as "<Provider> returned <STATUS>: <body>"
    msg.contains("429")
}

/// Wraps `inner` and retries `send_message` with exponential backoff on 429 errors.
pub struct RetryBackend<B: LlmBackend> {
    pub inner: B,
}

impl<B: LlmBackend> RetryBackend<B> {
    pub fn new(inner: B) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<B: LlmBackend + 'static> LlmBackend for RetryBackend<B> {
    async fn send_message(
        &self,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        let strategy = ExponentialBackoff::from_millis(INITIAL_DELAY_MS)
            .max_delay(Duration::from_millis(MAX_DELAY_MS))
            .map(jitter)
            .take(MAX_RETRIES);

        tokio_retry2::Retry::spawn(strategy, || async {
            self.inner.send_message(messages, config).await.map_err(
                |e| -> RetryError<anyhow::Error> {
                    if is_rate_limit_error(&e) {
                        RetryError::Transient {
                            err: e,
                            retry_after: None,
                        }
                    } else {
                        RetryError::Permanent(e)
                    }
                },
            )
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use anyhow::Result;
    use async_trait::async_trait;
    use futures::{StreamExt, stream};

    use super::*;
    use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

    struct FailNTimes {
        remaining: Arc<AtomicU32>,
        error: String,
    }

    impl FailNTimes {
        fn new(n: u32, error: &str) -> Self {
            Self {
                remaining: Arc::new(AtomicU32::new(n)),
                error: error.to_string(),
            }
        }
    }

    #[async_trait]
    impl LlmBackend for FailNTimes {
        async fn send_message(
            &self,
            _messages: &[Message],
            _config: &RequestConfig,
        ) -> Result<BoxStream<Result<StreamEvent>>> {
            if self.remaining.fetch_sub(1, Ordering::SeqCst) > 0 {
                anyhow::bail!("{}", self.error);
            }
            Ok(Box::pin(stream::iter(vec![
                Ok(StreamEvent::TextDelta("ok".to_string())),
                Ok(StreamEvent::Done),
            ])))
        }
    }

    fn config() -> RequestConfig {
        RequestConfig {
            model: "test".to_string(),
            max_tokens: 128,
            tools: vec![],
        }
    }

    #[test]
    fn is_rate_limit_error_detects_429_in_message() {
        let err = anyhow::anyhow!("Vertex AI returned 429: rate limit exceeded");
        assert!(is_rate_limit_error(&err));
    }

    #[test]
    fn is_rate_limit_error_rejects_other_status_codes() {
        let err = anyhow::anyhow!("Vertex AI returned 500: internal server error");
        assert!(!is_rate_limit_error(&err));
    }

    #[test]
    fn is_rate_limit_error_rejects_non_http_errors() {
        let err = anyhow::anyhow!("Failed to send request: connection refused");
        assert!(!is_rate_limit_error(&err));
    }

    #[tokio::test]
    async fn retries_once_on_429_then_succeeds() {
        // Backend fails once with 429, then succeeds on the second attempt.
        let inner = FailNTimes::new(1, "Vertex AI returned 429: rate limit");
        let backend = RetryBackend::new(inner);

        let mut stream = backend
            .send_message(&[], &config())
            .await
            .expect("should succeed after retry");

        let event = stream
            .next()
            .await
            .expect("stream must have first event")
            .expect("first event must be Ok");
        assert!(matches!(event, StreamEvent::TextDelta(_)));
    }

    #[tokio::test]
    async fn does_not_retry_on_non_429_error() {
        // Backend fails with a 500. Should not retry — error propagates immediately.
        let inner = FailNTimes::new(u32::MAX, "Vertex AI returned 500: internal error");
        let backend = RetryBackend::new(inner);

        let result = backend.send_message(&[], &config()).await;
        assert!(result.is_err());
        let err_msg = result.err().expect("must be an error").to_string();
        assert!(err_msg.contains("500"));
    }

    #[tokio::test]
    async fn succeeds_on_first_try_without_retrying() {
        let inner = FailNTimes::new(0, "");
        let backend = RetryBackend::new(inner);

        let mut stream = backend
            .send_message(&[], &config())
            .await
            .expect("should succeed on first try");

        let event = stream
            .next()
            .await
            .expect("stream must have first event")
            .expect("first event must be Ok");
        assert!(matches!(event, StreamEvent::TextDelta(_)));
    }

    #[tokio::test]
    async fn fails_after_exhausting_retries() {
        // Backend always fails with 429 — all retries exhausted, error is returned.
        let inner = FailNTimes::new(u32::MAX, "z.ai returned 429: quota exceeded");
        let backend = RetryBackend::new(inner);

        let result = backend.send_message(&[], &config()).await;
        assert!(result.is_err());
        let err_msg = result.err().expect("must be an error").to_string();
        assert!(err_msg.contains("429"));
    }
}
