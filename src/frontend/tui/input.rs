#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppState {
    Input,
    Streaming,
    ToolConfirmation {
        name: String,
        input: serde_json::Value,
    },
}
