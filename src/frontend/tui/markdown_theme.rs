use ratatui::style::{Color, Modifier, Style};
use the_other_tui_markdown::Theme;

fn monokai_yellow() -> Color {
    Color::Rgb(230, 219, 116)
}

fn monokai_green() -> Color {
    Color::Rgb(166, 226, 46)
}

fn monokai_blue() -> Color {
    Color::Rgb(102, 217, 239)
}

fn monokai_purple() -> Color {
    Color::Rgb(174, 129, 255)
}

fn monokai_pink() -> Color {
    Color::Rgb(249, 38, 114)
}

fn monokai_orange() -> Color {
    Color::Rgb(253, 151, 31)
}

fn monokai_comment() -> Color {
    Color::Rgb(117, 113, 94)
}

fn monokai_foreground() -> Color {
    Color::Rgb(248, 248, 242)
}

pub fn monokai_theme() -> Theme {
    Theme {
        base: Style::new().fg(monokai_foreground()),
        h1: Style::new()
            .fg(monokai_green())
            .add_modifier(Modifier::BOLD),
        h2: Style::new()
            .fg(monokai_green())
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        h3: Style::new().fg(monokai_blue()).add_modifier(Modifier::BOLD),
        h4: Style::new()
            .fg(monokai_blue())
            .add_modifier(Modifier::BOLD | Modifier::ITALIC),
        h5: Style::new()
            .fg(monokai_purple())
            .add_modifier(Modifier::BOLD),
        h6: Style::new()
            .fg(monokai_purple())
            .add_modifier(Modifier::BOLD | Modifier::ITALIC),
        strong: Style::new().fg(monokai_pink()).add_modifier(Modifier::BOLD),
        emphasis: Style::new()
            .fg(monokai_orange())
            .add_modifier(Modifier::ITALIC),
        strikethrough: Style::new().add_modifier(Modifier::CROSSED_OUT),
        superscript: Style::new().add_modifier(Modifier::DIM),
        subscript: Style::new().add_modifier(Modifier::DIM),
        inline_code: Style::new().fg(monokai_yellow()),
        link: Style::new()
            .fg(monokai_blue())
            .add_modifier(Modifier::UNDERLINED),
        image: Style::new()
            .fg(monokai_green())
            .add_modifier(Modifier::UNDERLINED),
        code_block: Style::new().fg(monokai_yellow()),
        code_block_lang: Style::new()
            .fg(monokai_comment())
            .add_modifier(Modifier::ITALIC),
        block_quote: Style::new()
            .fg(monokai_comment())
            .add_modifier(Modifier::ITALIC),
        block_quote_note: Style::new()
            .fg(monokai_blue())
            .add_modifier(Modifier::ITALIC),
        block_quote_tip: Style::new()
            .fg(monokai_green())
            .add_modifier(Modifier::ITALIC),
        block_quote_warning: Style::new()
            .fg(monokai_yellow())
            .add_modifier(Modifier::ITALIC),
        block_quote_caution: Style::new()
            .fg(monokai_pink())
            .add_modifier(Modifier::ITALIC),
        block_quote_important: Style::new()
            .fg(monokai_purple())
            .add_modifier(Modifier::ITALIC),
        list_marker: Style::new().fg(monokai_comment()),
        table_header: Style::new().fg(monokai_pink()).add_modifier(Modifier::BOLD),
        table_cell: Style::new().fg(monokai_foreground()),
        table_separator: Style::new().fg(monokai_comment()),
        rule: Style::new().fg(monokai_comment()),
        footnote_ref: Style::new()
            .fg(monokai_comment())
            .add_modifier(Modifier::DIM),
        footnote_def: Style::new().fg(monokai_comment()),
        math: Style::new().fg(monokai_yellow()),
        html: Style::new().fg(monokai_comment()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monokai_theme_has_non_default_base() {
        let theme = monokai_theme();
        let default_theme = Theme::default();
        assert_ne!(
            theme.base, default_theme.base,
            "monokai_theme base should differ from Theme::default()"
        );
        assert!(
            theme.base.fg.is_some(),
            "monokai_theme base should have a foreground color"
        );
    }

    #[test]
    fn monokai_theme_heading_colors() {
        let theme = monokai_theme();
        assert!(theme.h1.fg.is_some());
        assert!(theme.h2.fg.is_some());
        assert!(theme.h3.fg.is_some());
    }

    #[test]
    fn monokai_theme_table_styles() {
        let theme = monokai_theme();
        assert!(theme.table_header.fg.is_some());
        assert!(theme.table_header.add_modifier.contains(Modifier::BOLD));
    }
}
