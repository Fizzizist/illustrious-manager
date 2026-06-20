//! Ratatui adapter for `hjkl-splash` — renders a [`hjkl_splash::StartScreen`]
//! into a ratatui [`Frame`].

use hjkl_splash::{
    CellKind, Layout, Rgb, Splash, default_trail_color, presets, start_screen::StartScreen,
};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::Widget,
};

/// The ASCII art block (5 rows × 32 cols).
pub const ART: &str = include_str!("../../../art/illust.txt");

/// Number of rows in the art block.
pub const ROWS: u16 = 10;

/// Number of columns in the art block.
pub const COLS: u16 = 35;

fn rgb_to_color(Rgb(r, g, b): Rgb) -> Color {
    Color::Rgb(r, g, b)
}

/// Render the start screen into a ratatui `Frame`.
///
/// Paints the hjkl art animation centred in `area`, followed by a dim version
/// line and a hint line below it.
pub fn render(frame: &mut Frame, area: Rect, screen: &StartScreen) {
    let splash = Splash::new(ART, &[(0 as u8, 0 as u8, '█')]);
    let layout = Layout::centered(area.width, area.height, ROWS, COLS);

    let buf = frame.buffer_mut();

    // Paint art + animation cells.
    for cell in splash.cells(layout) {
        let x = area.x + cell.x;
        let y = area.y + cell.y;
        if x >= area.x + area.width || y >= area.y + area.height {
            continue;
        }
        let buf_cell = buf.cell_mut((x, y)).unwrap_or_else(|| {
            // Safety: we bounds-checked above; if the cell is somehow missing
            // just skip this iteration via a panic-free path.
            panic!("start_screen: cell ({x},{y}) out of buffer bounds");
        });
        buf_cell.set_char(cell.ch);
        let style = match cell.kind {
            CellKind::Art => Style::default().fg(rgb_to_color(screen.palette.text_dim)),
            CellKind::Trail { age: _ } => Style::default(),
            CellKind::Cursor => Style::default(),
        };
        buf_cell.set_style(style);
    }

    let ver_y = area.y + layout.origin_y + ROWS + 1;
    if ver_y < area.y + area.height {
        let ver_line = Line::from(vec![Span::styled(
            format!("v{} — by Peter Vlasveld", screen.version),
            Style::default().fg(rgb_to_color(screen.palette.text_dim)),
        )]);
        let ver_area = Rect {
            x: area.x + layout.origin_x,
            y: ver_y,
            width: area.width.saturating_sub(layout.origin_x),
            height: 1,
        };
        ver_line.render(ver_area, buf);
    }
}
