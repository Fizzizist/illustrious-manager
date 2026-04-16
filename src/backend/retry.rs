use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_retry2::RetryError;
use tokio_retry2::strategy::{ExponentialFactorBackoff, jitter};

use super::LlmBackend;
use crate::types::{BoxStream, Message, RequestConfig, StreamEvent};

/// Initial backoff delay for 429 retries.
const INITIAL_DELAY_MS: u64 = 1_000;
/// Exponential growth factor applied to the delay on each retry.
const BACKOFF_FACTOR: f64 = 2.0;
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
/// Emits `StreamEvent::RateLimitRetry` events before each retry so frontends can surface them.
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
        let strategy = ExponentialFactorBackoff::from_millis(INITIAL_DELAY_MS, BACKOFF_FACTOR)
            .max_delay(Duration::from_millis(MAX_DELAY_MS))
            .map(jitter)
            .take(MAX_RETRIES);

        let (notify_tx, mut notify_rx) = mpsc::unbounded_channel::<StreamEvent>();
        let attempt_counter = Arc::new(AtomicU32::new(0));

        let notify_tx_clone = notify_tx.clone();
        let attempt_counter_clone = attempt_counter.clone();
        let notifier = move |_err: &anyhow::Error, duration: Duration| {
            let attempt = attempt_counter_clone.fetch_add(1, Ordering::SeqCst) + 1;
            let delay_ms = duration.as_millis() as u64;
            // Ignore send errors — receiver dropped means the stream was cancelled.
            let _ = notify_tx_clone.send(StreamEvent::RateLimitRetry { attempt, delay_ms });
        };

        let result = tokio_retry2::Retry::spawn_notify(
            strategy,
            || async {
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
            },
            notifier,
        )
        .await?;

        // Drain any retry notifications that were buffered and prepend them to the response stream.
        drop(notify_tx);
        notify_rx.close();
        let mut prefix_events: Vec<Result<StreamEvent>> = Vec::new();
        while let Some(event) = notify_rx.recv().await {
            prefix_events.push(Ok(event));
        }

        let combined = futures::stream::iter(prefix_events).chain(result);
        Ok(Box::pin(combined))
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

        // First event should be the retry notification.
        let first = stream
            .next()
            .await
            .expect("stream must have first event")
            .expect("first event must be Ok");
        assert!(
            matches!(first, StreamEvent::RateLimitRetry { attempt: 1, .. }),
            "expected RateLimitRetry(attempt=1), got {:?}",
            first
        );

        // Second event is the actual response.
        let second = stream
            .next()
            .await
            .expect("stream must have second event")
            .expect("second event must be Ok");
        assert!(
            matches!(second, StreamEvent::TextDelta(_)),
            "expected TextDelta, got {:?}",
            second
        );
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

        // No retry notification prefix — first event is the actual response.
        let event = stream
            .next()
            .await
            .expect("stream must have first event")
            .expect("first event must be Ok");
        assert!(
            matches!(event, StreamEvent::TextDelta(_)),
            "expected TextDelta with no retry prefix, got {:?}",
            event
        );
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

    #[tokio::test]
    async fn emits_retry_notification_before_each_retry() {
        // Backend fails twice with 429, then succeeds. Should emit 2 RateLimitRetry events.
        let inner = FailNTimes::new(2, "Vertex AI returned 429: rate limit");
        let backend = RetryBackend::new(inner);

        let mut stream = backend
            .send_message(&[], &config())
            .await
            .expect("should succeed after two retries");

        let events: Vec<StreamEvent> = stream
            .map(|r| r.expect("stream item must be Ok"))
            .collect()
            .await;

        let retry_events: Vec<&StreamEvent> = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::RateLimitRetry { .. }))
            .collect();

        assert_eq!(retry_events.len(), 2, "expected 2 retry notifications");

        assert!(
            matches!(
                retry_events[0],
                StreamEvent::RateLimitRetry { attempt: 1, .. }
            ),
            "first retry should be attempt 1"
        );
        assert!(
            matches!(
                retry_events[1],
                StreamEvent::RateLimitRetry { attempt: 2, .. }
            ),
            "second retry should be attempt 2"
        );
    }
}
