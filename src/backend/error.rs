use std::fmt;

#[derive(Debug, Clone)]
pub enum BackendError {
    HttpStatus {
        code: u16,
        body: String,
    },
    Transport {
        message: String,
    },
    Other(String),
    MaxTokensExceeded {
        input_tokens: u32,
        output_tokens: u32,
    },
    Refusal,
}

impl BackendError {
    pub fn is_retryable(&self) -> bool {
        match self {
            BackendError::HttpStatus { code, .. } => *code == 429 || (*code >= 500 && *code != 501),
            BackendError::Transport { .. } => true,
            BackendError::Other(_) => false,
            BackendError::MaxTokensExceeded { .. } => false,
            BackendError::Refusal => false,
        }
    }

    pub fn is_max_tokens(&self) -> bool {
        matches!(self, BackendError::MaxTokensExceeded { .. })
    }

    pub fn is_refusal(&self) -> bool {
        matches!(self, BackendError::Refusal)
    }

    /// Build a `Transport` error from `target` (a human description of the
    /// endpoint) and an error, flattening the error's `source()` chain into one
    /// truthful string so no cause is discarded.
    pub fn transport(target: &str, err: &dyn std::error::Error) -> Self {
        let mut message = format!("Failed to send request to {target}: {err}");
        let mut source = err.source();
        while let Some(cause) = source {
            message.push_str(": ");
            message.push_str(&cause.to_string());
            source = cause.source();
        }
        BackendError::Transport { message }
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::HttpStatus { code, body } => {
                write!(f, "HTTP {code}: {body}")
            }
            BackendError::Transport { message } => write!(f, "{message}"),
            BackendError::Other(msg) => write!(f, "{msg}"),
            BackendError::MaxTokensExceeded {
                input_tokens,
                output_tokens,
            } => write!(
                f,
                "Response truncated: max_tokens limit reached (input_tokens={input_tokens}, output_tokens={output_tokens}). Increase max_tokens in your config."
            ),
            BackendError::Refusal => write!(
                f,
                "The model refused to generate a response due to content policy. The conversation context may be unsafe to continue with."
            ),
        }
    }
}

impl std::error::Error for BackendError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_status_503_is_retryable() {
        let err = BackendError::HttpStatus {
            code: 503,
            body: "overloaded".to_string(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn http_status_429_is_retryable() {
        let err = BackendError::HttpStatus {
            code: 429,
            body: "rate limited".to_string(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn http_status_500_is_retryable() {
        let err = BackendError::HttpStatus {
            code: 500,
            body: "internal".to_string(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn http_status_502_is_retryable() {
        let err = BackendError::HttpStatus {
            code: 502,
            body: "bad gateway".to_string(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn http_status_504_is_retryable() {
        let err = BackendError::HttpStatus {
            code: 504,
            body: "timeout".to_string(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn http_status_505_is_retryable() {
        let err = BackendError::HttpStatus {
            code: 505,
            body: "http version not supported".to_string(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn http_status_599_is_retryable() {
        let err = BackendError::HttpStatus {
            code: 599,
            body: "network timeout".to_string(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn http_status_400_is_not_retryable() {
        let err = BackendError::HttpStatus {
            code: 400,
            body: "bad request".to_string(),
        };
        assert!(!err.is_retryable());
    }

    #[test]
    fn http_status_401_is_not_retryable() {
        let err = BackendError::HttpStatus {
            code: 401,
            body: "unauthorized".to_string(),
        };
        assert!(!err.is_retryable());
    }

    #[test]
    fn http_status_501_is_not_retryable() {
        let err = BackendError::HttpStatus {
            code: 501,
            body: "not implemented".to_string(),
        };
        assert!(!err.is_retryable());
    }

    #[test]
    fn other_is_not_retryable() {
        let err = BackendError::Other("transport error".to_string());
        assert!(!err.is_retryable());
    }

    #[test]
    fn transport_is_retryable() {
        let err = BackendError::Transport {
            message: "connection refused".to_string(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn display_format_for_transport() {
        let err = BackendError::Transport {
            message: "Failed to send request to Ollama: connection refused".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Failed to send request to Ollama: connection refused"
        );
    }

    #[test]
    fn transport_constructor_flattens_source_chain() {
        #[derive(Debug)]
        struct Layer {
            msg: &'static str,
            source: Option<Box<Layer>>,
        }

        impl fmt::Display for Layer {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.msg)
            }
        }

        impl std::error::Error for Layer {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                self.source
                    .as_ref()
                    .map(|b| b.as_ref() as &(dyn std::error::Error + 'static))
            }
        }

        let err = Layer {
            msg: "outer failure",
            source: Some(Box::new(Layer {
                msg: "middle failure",
                source: Some(Box::new(Layer {
                    msg: "connection refused",
                    source: None,
                })),
            })),
        };

        let backend_err = BackendError::transport("Ollama", &err);
        let text = backend_err.to_string();
        assert!(text.contains("Failed to send request to Ollama"));
        assert!(text.contains("outer failure"));
        assert!(text.contains("middle failure"));
        assert!(text.contains("connection refused"));
        assert!(backend_err.is_retryable());
    }

    #[test]
    fn display_format_for_http_status() {
        let err = BackendError::HttpStatus {
            code: 503,
            body: "overloaded".to_string(),
        };
        assert_eq!(err.to_string(), "HTTP 503: overloaded");
    }

    #[test]
    fn display_format_for_other() {
        let err = BackendError::Other("connection refused".to_string());
        assert_eq!(err.to_string(), "connection refused");
    }

    #[test]
    fn converts_to_anyhow_and_downcasts() {
        let original = BackendError::HttpStatus {
            code: 503,
            body: "overloaded".to_string(),
        };
        let err: anyhow::Error = original.into();
        assert!(err.to_string().contains("503"));
        let recovered = err
            .downcast_ref::<BackendError>()
            .expect("should downcast to BackendError");
        assert!(recovered.is_retryable());
    }

    #[test]
    fn max_tokens_exceeded_is_not_retryable() {
        let err = BackendError::MaxTokensExceeded {
            input_tokens: 100,
            output_tokens: 200,
        };
        assert!(!err.is_retryable());
    }

    #[test]
    fn max_tokens_exceeded_display_format() {
        let err = BackendError::MaxTokensExceeded {
            input_tokens: 42,
            output_tokens: 99,
        };
        let text = err.to_string();
        assert!(
            text.contains("Response truncated: max_tokens limit reached"),
            "display should contain the truncation message; got: {text}"
        );
        assert!(text.contains("input_tokens=42"));
        assert!(text.contains("output_tokens=99"));
        assert!(text.contains("Increase max_tokens"));
    }

    #[test]
    fn max_tokens_exceeded_downcasts_from_anyhow() {
        let original = BackendError::MaxTokensExceeded {
            input_tokens: 10,
            output_tokens: 20,
        };
        let err: anyhow::Error = original.into();
        let recovered = err
            .downcast_ref::<BackendError>()
            .expect("should downcast to BackendError");
        assert!(
            matches!(
                recovered,
                BackendError::MaxTokensExceeded {
                    input_tokens: 10,
                    output_tokens: 20
                }
            ),
            "should recover the MaxTokensExceeded variant with token counts"
        );
    }

    #[test]
    fn is_max_tokens_true_for_max_tokens_variant() {
        let err = BackendError::MaxTokensExceeded {
            input_tokens: 5,
            output_tokens: 10,
        };
        assert!(err.is_max_tokens());
    }

    #[test]
    fn is_max_tokens_false_for_other_variants() {
        assert!(
            !BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }
            .is_max_tokens()
        );
        assert!(
            !BackendError::Transport {
                message: "connection refused".to_string(),
            }
            .is_max_tokens()
        );
        assert!(!BackendError::Other("something".to_string()).is_max_tokens());
    }

    #[test]
    fn refusal_is_not_retryable() {
        let err = BackendError::Refusal;
        assert!(!err.is_retryable());
    }

    #[test]
    fn is_refusal_returns_true_for_refusal() {
        let err = BackendError::Refusal;
        assert!(err.is_refusal());
    }

    #[test]
    fn is_refusal_returns_false_for_other_variants() {
        assert!(
            !BackendError::HttpStatus {
                code: 503,
                body: "overloaded".to_string(),
            }
            .is_refusal()
        );
        assert!(
            !BackendError::MaxTokensExceeded {
                input_tokens: 100,
                output_tokens: 200,
            }
            .is_refusal()
        );
        assert!(
            !BackendError::Transport {
                message: "connection refused".to_string(),
            }
            .is_refusal()
        );
        assert!(!BackendError::Other("something".to_string()).is_refusal());
    }

    #[test]
    fn refusal_display_is_informative() {
        let err = BackendError::Refusal;
        let text = err.to_string();
        assert!(
            text.contains("refused"),
            "display should contain 'refused'; got: {text}"
        );
        assert!(
            text.contains("content policy"),
            "display should contain 'content policy'; got: {text}"
        );
    }

    #[test]
    fn refusal_downcasts_from_anyhow() {
        let original = BackendError::Refusal;
        let err: anyhow::Error = original.into();
        let recovered = err
            .downcast_ref::<BackendError>()
            .expect("should downcast to BackendError");
        assert!(
            matches!(recovered, BackendError::Refusal),
            "should recover the Refusal variant"
        );
    }
}
