use super::text_chunker::{ChunkKind, TextChunk};

use tree_sitter::Parser;

#[cfg(feature = "ts-rust")]
fn rust_target_types() -> Vec<&'static str> {
    vec![
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
    ]
}

#[cfg(feature = "ts-typescript")]
fn typescript_target_types() -> Vec<&'static str> {
    vec![
        "function_declaration",
        "method_definition",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "export_statement",
        "arrow_function",
    ]
}

#[cfg(feature = "ts-python")]
fn python_target_types() -> Vec<&'static str> {
    vec!["function_definition", "class_definition"]
}

#[cfg(feature = "ts-go")]
fn go_target_types() -> Vec<&'static str> {
    vec!["function_declaration", "method_declaration", "type_spec"]
}

#[cfg(feature = "ts-java")]
fn java_target_types() -> Vec<&'static str> {
    vec![
        "class_declaration",
        "method_declaration",
        "interface_declaration",
    ]
}

#[cfg(feature = "ts-c")]
fn c_target_types() -> Vec<&'static str> {
    vec!["function_definition", "struct_specifier"]
}

#[cfg(feature = "ts-cpp")]
fn cpp_target_types() -> Vec<&'static str> {
    vec!["function_definition", "struct_specifier", "class_specifier"]
}

struct ExtractedRegion {
    name: Option<String>,
    kind: ChunkKind,
    start_line: usize,
    end_line: usize,
}

#[cfg(feature = "ts-rust")]
fn get_rust_name(node: &tree_sitter::Node, source: &[u8]) -> (ChunkKind, Option<String>) {
    match node.kind() {
        "function_item" => (
            ChunkKind::Function,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "impl_item" => (
            ChunkKind::Impl,
            node.child_by_field_name("type")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "struct_item" => (
            ChunkKind::Struct,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "enum_item" => (
            ChunkKind::Enum,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "trait_item" => (
            ChunkKind::Interface,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        _ => (ChunkKind::File, None),
    }
}

#[cfg(feature = "ts-typescript")]
fn get_ts_name(node: &tree_sitter::Node, source: &[u8]) -> (ChunkKind, Option<String>) {
    match node.kind() {
        "function_declaration" | "arrow_function" => (
            ChunkKind::Function,
            node.child_by_field_name("name")
                .or_else(|| {
                    node.parent()
                        .filter(|p| p.kind() == "variable_declarator")
                        .and_then(|p| p.child_by_field_name("name"))
                })
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "method_definition" => (
            ChunkKind::Method,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "class_declaration" => (
            ChunkKind::Class,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "interface_declaration" => (
            ChunkKind::Interface,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "type_alias_declaration" => (
            ChunkKind::TypeAlias,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "export_statement" => (
            ChunkKind::Export,
            node.child_by_field_name("declaration")
                .and_then(|decl| decl.child_by_field_name("name"))
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        _ => (ChunkKind::File, None),
    }
}

#[cfg(feature = "ts-python")]
fn get_python_name(node: &tree_sitter::Node, source: &[u8]) -> (ChunkKind, Option<String>) {
    match node.kind() {
        "function_definition" => (
            ChunkKind::Function,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "class_definition" => (
            ChunkKind::Class,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        _ => (ChunkKind::File, None),
    }
}

#[cfg(feature = "ts-go")]
fn get_go_name(node: &tree_sitter::Node, source: &[u8]) -> (ChunkKind, Option<String>) {
    match node.kind() {
        "function_declaration" => (
            ChunkKind::Function,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "method_declaration" => (
            ChunkKind::Method,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "type_spec" => (
            ChunkKind::Struct,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        _ => (ChunkKind::File, None),
    }
}

#[cfg(feature = "ts-java")]
fn get_java_name(node: &tree_sitter::Node, source: &[u8]) -> (ChunkKind, Option<String>) {
    match node.kind() {
        "class_declaration" => (
            ChunkKind::Class,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "method_declaration" => (
            ChunkKind::Method,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "interface_declaration" => (
            ChunkKind::Interface,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        _ => (ChunkKind::File, None),
    }
}

#[cfg(feature = "ts-c")]
fn get_c_name(node: &tree_sitter::Node, source: &[u8]) -> (ChunkKind, Option<String>) {
    match node.kind() {
        "function_definition" => (
            ChunkKind::Function,
            node.child_by_field_name("declarator")
                .filter(|d| d.kind() == "function_declarator")
                .and_then(|d| d.child_by_field_name("declarator"))
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "struct_specifier" => (
            ChunkKind::Struct,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        _ => (ChunkKind::File, None),
    }
}

#[cfg(feature = "ts-cpp")]
fn get_cpp_name(node: &tree_sitter::Node, source: &[u8]) -> (ChunkKind, Option<String>) {
    match node.kind() {
        "function_definition" => (
            ChunkKind::Function,
            node.child_by_field_name("declarator")
                .filter(|d| d.kind() == "function_declarator")
                .and_then(|d| d.child_by_field_name("declarator"))
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "struct_specifier" => (
            ChunkKind::Struct,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        "class_specifier" => (
            ChunkKind::Class,
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(String::from),
        ),
        _ => (ChunkKind::File, None),
    }
}

fn collect_matching_nodes<'a>(
    node: &tree_sitter::Node<'a>,
    target_types: &[&str],
    results: &mut Vec<tree_sitter::Node<'a>>,
) {
    if target_types.contains(&node.kind()) {
        results.push(*node);
        return;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_matching_nodes(&child, target_types, results);
    }
}

const MIN_GAP_LINES: usize = 3;

fn collect_gaps(
    regions: &[ExtractedRegion],
    source_lines: &[&str],
    file_path: &str,
) -> Vec<TextChunk> {
    let mut gaps = Vec::new();
    let mut cursor: usize = 1;

    for region in regions {
        if region.start_line > cursor {
            let gap_lines = &source_lines[cursor - 1..region.start_line - 1];
            let non_blank = gap_lines.iter().filter(|l| !l.trim().is_empty()).count();
            if non_blank > MIN_GAP_LINES {
                gaps.push(TextChunk {
                    file_path: file_path.to_string(),
                    start_line: cursor,
                    end_line: region.start_line - 1,
                    kind: ChunkKind::File,
                    name: None,
                    content: gap_lines.join("\n"),
                });
            }
        }
        cursor = region.end_line + 1;
    }

    if cursor <= source_lines.len() {
        let gap_lines = &source_lines[cursor - 1..];
        let non_blank = gap_lines.iter().filter(|l| !l.trim().is_empty()).count();
        if non_blank > MIN_GAP_LINES {
            gaps.push(TextChunk {
                file_path: file_path.to_string(),
                start_line: cursor,
                end_line: source_lines.len(),
                kind: ChunkKind::File,
                name: None,
                content: gap_lines.join("\n"),
            });
        }
    }

    gaps
}

macro_rules! ts_chunk_impl {
    ($func_name:ident, $lang_func:expr, $target_types_func:expr, $get_name_func:expr) => {
        pub fn $func_name(content: &str, file_path: &str) -> Vec<TextChunk> {
            let mut parser = Parser::new();
            let lang = $lang_func;
            parser.set_language(&lang).expect("set language");

            let source = content.as_bytes();
            let tree = parser.parse(content, None);
            let Some(tree) = tree else {
                return vec![TextChunk {
                    file_path: file_path.to_string(),
                    start_line: 1,
                    end_line: content.lines().count().max(1),
                    kind: ChunkKind::File,
                    name: None,
                    content: content.to_string(),
                }];
            };

            let source_lines: Vec<&str> = content.lines().collect();
            let target_types = $target_types_func();

            let mut matching = Vec::new();
            collect_matching_nodes(&tree.root_node(), &target_types, &mut matching);

            let mut raw: Vec<ExtractedRegion> = matching
                .iter()
                .map(|node| {
                    let (kind, name) = $get_name_func(node, source);
                    ExtractedRegion {
                        name,
                        kind,
                        start_line: node.start_position().row + 1,
                        end_line: node.end_position().row + 1,
                    }
                })
                .collect();

            raw.sort_by(|a, b| {
                a.start_line
                    .cmp(&b.start_line)
                    .then(b.end_line.cmp(&a.end_line))
            });

            let mut regions = Vec::new();
            let mut last_end = 0;
            for region in raw {
                if region.start_line > last_end {
                    last_end = region.end_line;
                    regions.push(region);
                }
            }

            let mut chunks: Vec<TextChunk> = regions
                .iter()
                .map(|r| {
                    let start_byte = source_lines[..r.start_line - 1]
                        .iter()
                        .map(|l| l.len() + 1)
                        .sum::<usize>();
                    let end_byte = source_lines[..r.end_line]
                        .iter()
                        .map(|l| l.len() + 1)
                        .sum::<usize>();
                    let content_text =
                        String::from_utf8_lossy(&source[start_byte..end_byte.min(source.len())])
                            .to_string();

                    TextChunk {
                        file_path: file_path.to_string(),
                        start_line: r.start_line,
                        end_line: r.end_line,
                        kind: r.kind.clone(),
                        name: r.name.clone(),
                        content: content_text,
                    }
                })
                .collect();

            let gaps = collect_gaps(&regions, &source_lines, file_path);
            chunks.extend(gaps);
            chunks.sort_by_key(|c| c.start_line);

            if chunks.is_empty() {
                vec![TextChunk {
                    file_path: file_path.to_string(),
                    start_line: 1,
                    end_line: source_lines.len().max(1),
                    kind: ChunkKind::File,
                    name: None,
                    content: content.to_string(),
                }]
            } else {
                chunks
            }
        }
    };
}

#[cfg(feature = "ts-rust")]
ts_chunk_impl!(
    chunk_rust,
    tree_sitter_rust::LANGUAGE.into(),
    rust_target_types,
    get_rust_name
);

#[cfg(feature = "ts-typescript")]
ts_chunk_impl!(
    chunk_typescript,
    tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
    typescript_target_types,
    get_ts_name
);

#[cfg(feature = "ts-typescript")]
ts_chunk_impl!(
    chunk_tsx,
    tree_sitter_typescript::LANGUAGE_TSX.into(),
    typescript_target_types,
    get_ts_name
);

#[cfg(feature = "ts-python")]
ts_chunk_impl!(
    chunk_python,
    tree_sitter_python::LANGUAGE.into(),
    python_target_types,
    get_python_name
);

#[cfg(feature = "ts-go")]
ts_chunk_impl!(
    chunk_go,
    tree_sitter_go::LANGUAGE.into(),
    go_target_types,
    get_go_name
);

#[cfg(feature = "ts-java")]
ts_chunk_impl!(
    chunk_java,
    tree_sitter_java::LANGUAGE.into(),
    java_target_types,
    get_java_name
);

#[cfg(feature = "ts-c")]
ts_chunk_impl!(
    chunk_c,
    tree_sitter_c::LANGUAGE.into(),
    c_target_types,
    get_c_name
);

#[cfg(feature = "ts-cpp")]
ts_chunk_impl!(
    chunk_cpp,
    tree_sitter_cpp::LANGUAGE.into(),
    cpp_target_types,
    get_cpp_name
);

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ts-rust")]
    #[test]
    fn rust_function_extraction() {
        let code = r#"fn main() {
    println!("hello");
}

fn helper(x: i32) -> i32 {
    x + 1
}
"#;
        let chunks = chunk_rust(code, "main.rs");
        assert!(
            chunks.len() >= 2,
            "Should extract at least 2 functions, got: {:?}",
            chunks
        );

        let names: Vec<&str> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(names.contains(&"main"));
        assert!(names.contains(&"helper"));
    }

    #[cfg(feature = "ts-rust")]
    #[test]
    fn rust_struct_and_enum_extraction() {
        let code = r#"struct Config {
    port: u16,
    host: String,
}

enum Error {
    Io(std::io::Error),
    Parse(String),
}
"#;
        let chunks = chunk_rust(code, "types.rs");
        assert!(chunks.len() >= 2);

        let names: Vec<&str> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(names.contains(&"Config"));
        assert!(names.contains(&"Error"));
    }

    #[cfg(feature = "ts-rust")]
    #[test]
    fn rust_impl_extraction() {
        let code = r#"impl Config {
    fn new() -> Self {
        Self { port: 8080, host: "localhost".into() }
    }
}
"#;
        let chunks = chunk_rust(code, "config.rs");
        assert!(!chunks.is_empty());

        let names: Vec<&str> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(names.contains(&"Config"));
    }

    #[cfg(feature = "ts-rust")]
    #[test]
    fn empty_rust_file_returns_single_chunk() {
        let code = "";
        let chunks = chunk_rust(code, "empty.rs");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::File);
    }

    #[cfg(feature = "ts-rust")]
    #[test]
    fn rust_file_with_only_comments() {
        let code = "// just a comment\n// another comment\n";
        let chunks = chunk_rust(code, "comment.rs");
        assert!(!chunks.is_empty());
    }
}
