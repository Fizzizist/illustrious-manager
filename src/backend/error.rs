use std::fmt;

/// Structured errors emitted by backends.
///
/// `HttpStatus` carries the HTTP status code and response body for
/// programmatic classification (retryable vs non-retryable). `Other` covers
/// transport failures, parse errors, and other non-HTTP failures.
#[derive(Debug, Clone)]
pub enum BackendError {
    HttpStatus { code: u16, body: String },
    Other(String),
}

impl BackendError {
    /// Returns `true` for status codes that are worth retrying: 429 (rate
    /// limit) and all 5xx codes except 501 (Not Implemented). Other 4xx
    /// codes are not retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            BackendError::HttpStatus { code, .. } => *code == 429 || (*code >= 500 && *code != 501),
            BackendError::Other(_) => false,
        }
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::HttpStatus { code, body } => {
                write!(f, "HTTP {code}: {body}")
            }
            BackendError::Other(msg) => write!(f, "{msg}"),
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
}
