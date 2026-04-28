pub mod commands;
pub mod conversation_area;
pub mod diff;
pub mod input_area;
pub mod list_picker;
pub mod markdown_tables;
pub mod session_picker;
pub mod status_line;
pub mod tasks_picker;
mod tui_app;

pub use conversation_area::{ConversationEntry, ConversationRole};
pub use session_picker::{SessionPicker, SessionPickerAction};
pub use status_line::{StatusLineInfo, TokenUsage};
pub use tasks_picker::{TasksPicker, TasksPickerAction};
pub use tui_app::*;

/// Compute a centred rectangle that is `percent_x`% wide and `percent_y`% tall of `r`.
pub(super) fn centered_rect(
    percent_x: u16,
    percent_y: u16,
    r: ratatui::layout::Rect,
) -> ratatui::layout::Rect {
    let w = r.width * percent_x / 100;
    let h = r.height * percent_y / 100;
    let x = r.x + (r.width.saturating_sub(w)) / 2;
    let y = r.y + (r.height.saturating_sub(h)) / 2;
    ratatui::layout::Rect {
        x,
        y,
        width: w,
        height: h,
    }
}
