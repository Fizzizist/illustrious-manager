use super::scanner::FileType;
use super::text_chunker::{TextChunk, chunk_text_file};

pub fn chunk_file(content: &str, file_path: &str, file_type: &FileType) -> Vec<TextChunk> {
    let type_str = file_type.to_string();

    match file_type {
        FileType::Rust => {
            #[cfg(feature = "ts-rust")]
            {
                super::ts_chunker::chunk_rust(content, file_path)
            }
            #[cfg(not(feature = "ts-rust"))]
            chunk_text_file(content, file_path, "plaintext")
        }
        #[cfg(feature = "ts-typescript")]
        FileType::TypeScript => super::ts_chunker::chunk_typescript(content, file_path),
        #[cfg(feature = "ts-typescript")]
        FileType::Tsx => super::ts_chunker::chunk_tsx(content, file_path),
        #[cfg(feature = "ts-python")]
        FileType::Python => super::ts_chunker::chunk_python(content, file_path),
        #[cfg(feature = "ts-go")]
        FileType::Go => super::ts_chunker::chunk_go(content, file_path),
        #[cfg(feature = "ts-java")]
        FileType::Java => super::ts_chunker::chunk_java(content, file_path),
        #[cfg(feature = "ts-c")]
        FileType::C => super::ts_chunker::chunk_c(content, file_path),
        #[cfg(feature = "ts-cpp")]
        FileType::Cpp => super::ts_chunker::chunk_cpp(content, file_path),
        _ => chunk_text_file(content, file_path, &type_str),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_dispatches_to_text_chunker() {
        let content = "# Title\n\nSome content that is long enough to be split into chunks by the text chunker module";
        let chunks = chunk_file(content, "test.md", &FileType::Markdown);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn yaml_dispatches_to_text_chunker() {
        let content = "server:\n  port: 8080\n  host: localhost\ndatabase:\n  url: postgres://localhost\n  pool_size: 10";
        let chunks = chunk_file(content, "config.yaml", &FileType::Yaml);
        assert!(!chunks.is_empty());
    }

    #[cfg(feature = "ts-rust")]
    #[test]
    fn rust_dispatches_to_ts_chunker() {
        let code = "fn main() { println!(\"hello\"); }";
        let chunks = chunk_file(code, "main.rs", &FileType::Rust);
        assert!(!chunks.is_empty());
    }
}
