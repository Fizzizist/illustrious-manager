/// Fixed character-count scale applied to `ContentBlock::Image` blocks in
/// token-usage estimation. Images have no text content, so the status-line
/// counter uses this constant to keep estimates sane (actual provider token
/// cost for images is heuristic anyway).
pub const IMAGE_PLACEHOLDER_TOKEN_SCALE: u64 = 200;

pub mod stdout;
pub mod tui;
