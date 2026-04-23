use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::tools::ToolError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    #[default]
    Pending,
    InProgress,
    Completed,
}

impl TaskStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
        }
    }
}

impl FromStr for TaskStatus {
    type Err = ToolError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "in_progress" => Ok(Self::InProgress),
            "completed" => Ok(Self::Completed),
            other => Err(ToolError::InvalidInput {
                message: format!(
                    "Invalid status '{}'. Must be one of: pending, in_progress, completed",
                    other
                ),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_status_round_trips_through_string() {
        for (variant, s) in [
            (TaskStatus::Pending, "pending"),
            (TaskStatus::InProgress, "in_progress"),
            (TaskStatus::Completed, "completed"),
        ] {
            assert_eq!(variant.as_str(), s);
            assert_eq!(TaskStatus::from_str(s).expect("parse"), variant);
        }
    }

    #[test]
    fn task_status_rejects_invalid_string() {
        let err = TaskStatus::from_str("done").expect_err("should fail");
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[test]
    fn task_status_default_is_pending() {
        assert_eq!(TaskStatus::default(), TaskStatus::Pending);
    }
}
