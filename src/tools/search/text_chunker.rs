use std::sync::LazyLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkKind {
    Function,
    Method,
    Class,
    Interface,
    Struct,
    Enum,
    Impl,
    Export,
    TypeAlias,
    Module,
    HeadingSection,
    TopLevelKey,
    Paragraph,
    File,
}

impl std::fmt::Display for ChunkKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChunkKind::Function => write!(f, "function"),
            ChunkKind::Method => write!(f, "method"),
            ChunkKind::Class => write!(f, "class"),
            ChunkKind::Interface => write!(f, "interface"),
            ChunkKind::Struct => write!(f, "struct"),
            ChunkKind::Enum => write!(f, "enum"),
            ChunkKind::Impl => write!(f, "impl"),
            ChunkKind::Export => write!(f, "export"),
            ChunkKind::TypeAlias => write!(f, "type_alias"),
            ChunkKind::Module => write!(f, "module"),
            ChunkKind::HeadingSection => write!(f, "heading_section"),
            ChunkKind::TopLevelKey => write!(f, "top_level_key"),
            ChunkKind::Paragraph => write!(f, "paragraph"),
            ChunkKind::File => write!(f, "file"),
        }
    }
}

impl std::str::FromStr for ChunkKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "function" => Ok(ChunkKind::Function),
            "method" => Ok(ChunkKind::Method),
            "class" => Ok(ChunkKind::Class),
            "interface" => Ok(ChunkKind::Interface),
            "struct" => Ok(ChunkKind::Struct),
            "enum" => Ok(ChunkKind::Enum),
            "impl" => Ok(ChunkKind::Impl),
            "export" => Ok(ChunkKind::Export),
            "type_alias" => Ok(ChunkKind::TypeAlias),
            "module" => Ok(ChunkKind::Module),
            "heading_section" => Ok(ChunkKind::HeadingSection),
            "top_level_key" => Ok(ChunkKind::TopLevelKey),
            "paragraph" => Ok(ChunkKind::Paragraph),
            "file" => Ok(ChunkKind::File),
            other => Err(format!("Unknown chunk kind: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextChunk {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub kind: ChunkKind,
    pub name: Option<String>,
    pub content: String,
}

const MIN_SPLIT_SIZE: usize = 500;
const MAX_CHUNK_SIZE: usize = 8000;
const MIN_PARAGRAPH_SIZE: usize = 200;

#[allow(clippy::type_complexity)]
static HEADING_RE: LazyLock<fn(&str) -> Option<(usize, &str)>> = LazyLock::new(|| {
    |line: &str| {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            let hash_count = trimmed.chars().take_while(|c| *c == '#').count();
            if hash_count <= 6 {
                let rest = trimmed[hash_count..].trim_start();
                if !rest.is_empty() {
                    return Some((hash_count, rest));
                }
            }
        }
        None
    }
});

static YAML_TOP_KEY_RE: LazyLock<fn(&str) -> Option<&str>> = LazyLock::new(|| {
    |line: &str| {
        let first_char = line.chars().next()?;
        if !first_char.is_alphabetic() && first_char != '_' {
            return None;
        }
        if let Some(colon_pos) = line.find(':') {
            let key = &line[..colon_pos];
            if key
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '.' || c == '-')
            {
                let after_colon = line[colon_pos + 1..].trim();
                if after_colon.is_empty()
                    || after_colon.starts_with(' ')
                    || after_colon.starts_with('\t')
                {
                    return Some(key);
                }
            }
        }
        None
    }
});

static TOML_SECTION_RE: LazyLock<fn(&str) -> Option<&str>> = LazyLock::new(|| {
    |line: &str| {
        let trimmed = line.trim();
        if (trimmed.starts_with('[') && trimmed.ends_with(']'))
            || (trimmed.starts_with("[[") && trimmed.ends_with("]]"))
        {
            let inner = trimmed.trim_start_matches('[').trim_end_matches(']');
            let name = inner.trim();
            if !name.is_empty() {
                return Some(name);
            }
        }
        None
    }
});

static TOML_KV_RE: LazyLock<fn(&str) -> Option<&str>> = LazyLock::new(|| {
    |line: &str| {
        let first_char = line.chars().next()?;
        if !first_char.is_alphabetic() && first_char != '_' {
            return None;
        }
        if let Some(eq_pos) = line.find('=') {
            let key = &line[..eq_pos];
            if key
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '.' || c == '-')
            {
                return Some(key.trim());
            }
        }
        None
    }
});

pub fn chunk_text_file(content: &str, file_path: &str, file_type: &str) -> Vec<TextChunk> {
    if content.len() < MIN_SPLIT_SIZE {
        return vec![whole_file_chunk(content, file_path)];
    }

    let chunks = match file_type {
        "markdown" => chunk_markdown(content, file_path),
        "yaml" => chunk_yaml(content, file_path),
        "json" => chunk_json(content, file_path),
        "toml" => chunk_toml(content, file_path),
        "plaintext" => chunk_plaintext(content, file_path),
        _ => Vec::new(),
    };

    if chunks.is_empty() {
        return vec![whole_file_chunk(content, file_path)];
    }

    chunks.into_iter().flat_map(enforce_max_size).collect()
}

fn whole_file_chunk(content: &str, file_path: &str) -> TextChunk {
    let line_count = content.lines().count().max(1);
    TextChunk {
        file_path: file_path.to_string(),
        start_line: 1,
        end_line: line_count,
        kind: ChunkKind::File,
        name: None,
        content: content.to_string(),
    }
}

fn chunk_markdown(content: &str, file_path: &str) -> Vec<TextChunk> {
    let lines: Vec<&str> = content.lines().collect();
    let mut headings: Vec<(usize, usize, String)> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        if let Some((level, text)) = HEADING_RE(line) {
            headings.push((i, level, text.to_string()));
        }
    }

    if headings.is_empty() {
        return chunk_plaintext(content, file_path);
    }

    let mut chunks = Vec::new();

    if headings[0].0 > 0 {
        let preamble: Vec<&str> = lines[..headings[0].0].to_vec();
        let preamble_content = preamble.join("\n");
        if !preamble_content.trim().is_empty() {
            chunks.push(TextChunk {
                file_path: file_path.to_string(),
                start_line: 1,
                end_line: headings[0].0,
                kind: ChunkKind::HeadingSection,
                name: None,
                content: preamble_content,
            });
        }
    }

    for i in 0..headings.len() {
        let start = headings[i].0;
        let end = {
            let mut next_same_or_higher = None;
            for j in (i + 1)..headings.len() {
                if headings[j].1 <= headings[i].1 {
                    next_same_or_higher = Some(j);
                    break;
                }
            }
            match next_same_or_higher {
                Some(j) => headings[j].0 - 1,
                None => lines.len() - 1,
            }
        };

        let section: Vec<&str> = lines[start..=end].to_vec();
        chunks.push(TextChunk {
            file_path: file_path.to_string(),
            start_line: start + 1,
            end_line: end + 1,
            kind: ChunkKind::HeadingSection,
            name: Some(headings[i].2.clone()),
            content: section.join("\n"),
        });
    }

    chunks
}

fn chunk_yaml(content: &str, file_path: &str) -> Vec<TextChunk> {
    let lines: Vec<&str> = content.lines().collect();
    let mut keys: Vec<(usize, String)> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('#')
            || trimmed.starts_with("---")
            || trimmed.starts_with("...")
            || trimmed.is_empty()
        {
            continue;
        }
        if let Some(key) = YAML_TOP_KEY_RE(line) {
            keys.push((i, key.to_string()));
        }
    }

    if keys.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    for i in 0..keys.len() {
        let start = if i == 0 { 0 } else { keys[i].0 };
        let end = if i < keys.len() - 1 {
            keys[i + 1].0 - 1
        } else {
            lines.len() - 1
        };

        let section: Vec<&str> = lines[start..=end].to_vec();
        chunks.push(TextChunk {
            file_path: file_path.to_string(),
            start_line: start + 1,
            end_line: end + 1,
            kind: ChunkKind::TopLevelKey,
            name: Some(keys[i].1.clone()),
            content: section.join("\n"),
        });
    }

    chunks
}

fn chunk_json(content: &str, file_path: &str) -> Vec<TextChunk> {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(content);
    let parsed = match parsed {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let obj = match parsed.as_object() {
        Some(o) => o,
        None => return Vec::new(),
    };

    let top_keys: Vec<&String> = obj.keys().collect();
    if top_keys.is_empty() {
        return Vec::new();
    }

    let lines: Vec<&str> = content.lines().collect();

    if lines.len() <= 1 {
        let mut chunks = Vec::new();
        for key in &top_keys {
            let val = &obj[*key];
            let serialized =
                serde_json::to_string_pretty(&serde_json::json!({ (*key).clone(): val.clone() }))
                    .expect("serialize json chunk");
            chunks.push(TextChunk {
                file_path: file_path.to_string(),
                start_line: 1,
                end_line: 1,
                kind: ChunkKind::TopLevelKey,
                name: Some(key.to_string()),
                content: serialized,
            });
        }
        return chunks;
    }

    let mut key_positions: Vec<(String, usize)> = Vec::new();
    let mut depth = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let chars: Vec<char> = line.chars().collect();
        let mut c = 0usize;
        while c < chars.len() {
            match chars[c] {
                '"' => {
                    c += 1;
                    while c < chars.len() && chars[c] != '"' {
                        if chars[c] == '\\' {
                            c += 1;
                        }
                        c += 1;
                    }
                    if depth == 1
                        && let Some(rest) = line.get(c + 1..)
                    {
                        let rest_trimmed = rest.trim_start();
                        if rest_trimmed.starts_with(':') {
                            let key_text = &line[line.find('"').expect("should have quote")
                                ..=line.rfind('"').expect("should have end quote")];
                            let key_clean = key_text.trim_matches('"');
                            if top_keys.iter().any(|k| k.as_str() == key_clean)
                                && !key_positions.iter().any(|(k, _)| k == key_clean)
                            {
                                key_positions.push((key_clean.to_string(), i));
                            }
                        }
                    }
                }
                '{' | '[' => depth += 1,
                '}' | ']' => depth = depth.saturating_sub(1),
                _ => {}
            }
            c += 1;
        }
    }

    let mut chunks = Vec::new();
    for i in 0..key_positions.len() {
        let start = key_positions[i].1;
        let end = if i < key_positions.len() - 1 {
            key_positions[i + 1].1 - 1
        } else {
            lines.len() - 1
        };

        let mut real_end = end;
        while real_end > start && lines[real_end].trim().is_empty() {
            real_end -= 1;
        }

        chunks.push(TextChunk {
            file_path: file_path.to_string(),
            start_line: start + 1,
            end_line: real_end + 1,
            kind: ChunkKind::TopLevelKey,
            name: Some(key_positions[i].0.clone()),
            content: lines[start..=real_end].join("\n"),
        });
    }

    chunks
}

fn chunk_toml(content: &str, file_path: &str) -> Vec<TextChunk> {
    let lines: Vec<&str> = content.lines().collect();

    let mut boundaries: Vec<(usize, String)> = Vec::new();
    let mut first_section_line = lines.len();

    for (i, line) in lines.iter().enumerate() {
        if let Some(name) = TOML_SECTION_RE(line) {
            if i < first_section_line {
                first_section_line = i;
            }
            boundaries.push((i, name.to_string()));
        }
    }

    for (i, &line) in lines.iter().enumerate().take(first_section_line) {
        if line.trim().is_empty() || line.trim().starts_with('#') {
            continue;
        }
        if let Some(key) = TOML_KV_RE(line) {
            boundaries.push((i, key.to_string()));
        }
    }

    boundaries.sort_by_key(|(line, _)| *line);

    if boundaries.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    for i in 0..boundaries.len() {
        let start = if i == 0 { 0 } else { boundaries[i].0 };
        let end = if i < boundaries.len() - 1 {
            boundaries[i + 1].0 - 1
        } else {
            lines.len() - 1
        };

        chunks.push(TextChunk {
            file_path: file_path.to_string(),
            start_line: start + 1,
            end_line: end + 1,
            kind: ChunkKind::TopLevelKey,
            name: Some(boundaries[i].1.clone()),
            content: lines[start..=end].join("\n"),
        });
    }

    chunks
}

fn chunk_plaintext(content: &str, file_path: &str) -> Vec<TextChunk> {
    let lines: Vec<&str> = content.lines().collect();

    let mut paragraphs: Vec<(usize, usize, String)> = Vec::new();
    let mut para_start: Option<usize> = None;
    let mut consecutive_blanks = 0usize;

    for i in 0..lines.len() {
        let is_blank = lines[i].trim().is_empty();

        if is_blank {
            consecutive_blanks += 1;
            if consecutive_blanks >= 2
                && let Some(start) = para_start
            {
                let para_end = i.saturating_sub(consecutive_blanks).max(start);
                paragraphs.push((start, para_end, lines[start..=para_end].join("\n")));
                para_start = None;
            }
        } else {
            if para_start.is_none() {
                para_start = Some(i);
            }
            consecutive_blanks = 0;
        }
    }

    if let Some(start) = para_start {
        let mut end = lines.len() - 1;
        while end > start && lines[end].trim().is_empty() {
            end -= 1;
        }
        paragraphs.push((start, end, lines[start..=end].join("\n")));
    }

    if paragraphs.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut group_start = paragraphs[0].0;
    let mut group_end = paragraphs[0].1;
    let mut group_content = paragraphs[0].2.clone();

    for para in paragraphs.iter().skip(1) {
        if group_content.len() < MIN_PARAGRAPH_SIZE {
            group_end = para.1;
            group_content = format!("{group_content}\n\n{}", para.2);
        } else {
            chunks.push(TextChunk {
                file_path: file_path.to_string(),
                start_line: group_start + 1,
                end_line: group_end + 1,
                kind: ChunkKind::Paragraph,
                name: extract_paragraph_name(&group_content),
                content: group_content,
            });
            group_start = para.0;
            group_end = para.1;
            group_content = para.2.clone();
        }
    }

    if group_content.len() < MIN_PARAGRAPH_SIZE && !chunks.is_empty() {
        let last = chunks.last_mut().expect("should have chunk");
        last.end_line = group_end + 1;
        last.content = format!("{}\n\n{group_content}", last.content);
    } else {
        chunks.push(TextChunk {
            file_path: file_path.to_string(),
            start_line: group_start + 1,
            end_line: group_end + 1,
            kind: ChunkKind::Paragraph,
            name: extract_paragraph_name(&group_content),
            content: group_content,
        });
    }

    chunks
}

fn extract_paragraph_name(content: &str) -> Option<String> {
    let first_line = content.lines().next()?.trim();
    if first_line.is_empty() {
        return None;
    }
    if first_line.len() <= 60 {
        Some(first_line.to_string())
    } else {
        Some(format!("{}...", &first_line[..57]))
    }
}

fn enforce_max_size(chunk: TextChunk) -> Vec<TextChunk> {
    if chunk.content.len() <= MAX_CHUNK_SIZE {
        return vec![chunk];
    }

    let lines: Vec<&str> = chunk.content.lines().collect();
    let mut sub_chunks: Vec<TextChunk> = Vec::new();
    let mut current_lines: Vec<&str> = Vec::new();
    let mut current_size: usize = 0;
    let mut chunk_start_line = chunk.start_line;
    let mut part_index = 0;

    for line in lines {
        let line_size = line.len() + 1;

        if current_size + line_size > MAX_CHUNK_SIZE && !current_lines.is_empty() {
            let mut split_at = current_lines.len();
            for j in (1..current_lines.len()).rev() {
                if current_lines[j].trim().is_empty() {
                    split_at = j;
                    break;
                }
            }

            let emit_lines: Vec<&str> = current_lines[..split_at].to_vec();
            let emit_content = emit_lines.join("\n");
            let emit_end_line = chunk_start_line + split_at - 1;

            sub_chunks.push(TextChunk {
                file_path: chunk.file_path.clone(),
                start_line: chunk_start_line,
                end_line: emit_end_line,
                kind: chunk.kind.clone(),
                name: if part_index == 0 {
                    chunk.name.clone()
                } else {
                    chunk.name.as_ref().map(|n| format!("{n} (cont.)"))
                },
                content: emit_content,
            });
            part_index += 1;

            let remaining: Vec<&str> = current_lines[split_at..].to_vec();
            current_lines = remaining;
            current_lines.push(line);
            chunk_start_line = emit_end_line + 1;
            current_size = current_lines.join("\n").len();
        } else {
            current_lines.push(line);
            current_size += line_size;
        }
    }

    if !current_lines.is_empty() {
        sub_chunks.push(TextChunk {
            file_path: chunk.file_path.clone(),
            start_line: chunk_start_line,
            end_line: chunk.end_line,
            kind: chunk.kind.clone(),
            name: if part_index == 0 {
                chunk.name.clone()
            } else {
                chunk.name.as_ref().map(|n| format!("{n} (cont.)"))
            },
            content: current_lines.join("\n"),
        });
    }

    if sub_chunks.is_empty() {
        vec![chunk]
    } else {
        sub_chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_file_returns_single_chunk() {
        let content = "short file";
        let chunks = chunk_text_file(content, "test.txt", "plaintext");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::File);
    }

    #[test]
    fn markdown_splits_by_headings() {
        let mut content = String::from("# Title\n\n");
        content.push_str("Some intro text that is long enough to make this file exceed the minimum split size threshold for our text chunker implementation. We need at least five hundred characters in total for the chunker to activate its format-specific splitting logic rather than returning the entire file as a single chunk.\n\n");
        content.push_str("## Section 1\n\n");
        content.push_str("Content 1 that is long enough to be significant and meaningful and contains enough text to meet size requirements for chunking.\n\n");
        content.push_str("## Section 2\n\n");
        content.push_str("Content 2 that is long enough to be significant and meaningful and contains enough text to meet size requirements for chunking purposes.\n\n");
        let chunks = chunk_text_file(&content, "test.md", "markdown");
        assert!(
            chunks.len() >= 2,
            "Should have at least 2 heading sections, got: {:?}",
            chunks
        );

        let names: Vec<&str> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(
            names.contains(&"Title"),
            "Should contain Title, got: {:?}",
            names
        );
        assert!(
            names.contains(&"Section 1"),
            "Should contain Section 1, got: {:?}",
            names
        );
        assert!(
            names.contains(&"Section 2"),
            "Should contain Section 2, got: {:?}",
            names
        );
    }

    #[test]
    fn markdown_no_headings_falls_back_to_plaintext() {
        let content =
            "Line 1\n\nLine 2\n\nLine 3\n\nLine 4 is longer and has more content to be meaningful";
        let chunks = chunk_text_file(content, "test.md", "markdown");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn yaml_splits_by_top_level_keys() {
        let mut content = String::new();
        content.push_str("server:\n  port: 8080\n  host: localhost\n  timeout: 30\n");
        content.push_str(
            "  max_connections: 100\n  enable_tls: true\n  cert_path: /etc/ssl/cert.pem\n",
        );
        content.push_str(
            "  key_path: /etc/ssl/key.pem\n  worker_threads: 4\n  max_body_size: 10485760\n",
        );
        content.push_str("  keep_alive_timeout: 75\n  client_header_timeout: 60\n");
        content.push_str("  client_body_timeout: 60\n  send_timeout: 30\n");
        content.push_str(
            "  access_log: /var/log/nginx/access.log\n  error_log: /var/log/nginx/error.log\n\n",
        );
        content.push_str("database:\n  url: postgres://localhost:5432/mydb\n  pool_size: 10\n");
        content
            .push_str("  timeout: 30\n  max_retries: 3\n  enable_ssl: true\n  ssl_mode: require\n");
        content
            .push_str("  connection_timeout: 5\n  statement_timeout: 30000\n  idle_timeout: 600\n");
        content.push_str("  max_lifetime: 1800\n");
        let chunks = chunk_text_file(&content, "config.yaml", "yaml");
        assert!(
            chunks.len() >= 2,
            "Should split on top-level keys, got: {:?}",
            chunks.len()
        );

        let names: Vec<&str> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(
            names.contains(&"server"),
            "Should contain server, got: {:?}",
            names
        );
        assert!(
            names.contains(&"database"),
            "Should contain database, got: {:?}",
            names
        );
    }

    #[test]
    fn json_splits_by_top_level_keys() {
        let content = r#"{
  "name": "test-project-with-a-long-name-for-testing",
  "version": "1.0.0",
  "description": "A test project with enough content to exceed the minimum split size threshold for our text chunker to activate format-specific splitting logic for JSON files in the test suite",
  "main": "src/main.rs",
  "license": "MIT",
  "repository": "https://github.com/example/test-project",
  "dependencies": {
    "serde": "1.0",
    "anyhow": "1.0",
    "tokio": "1.0"
  },
  "devDependencies": {
    "tempfile": "3.0",
    "insta": "1.0"
  }
}"#;
        let chunks = chunk_text_file(content, "package.json", "json");
        assert!(
            chunks.len() >= 2,
            "Should split on top-level keys, got: {:?}",
            chunks.len()
        );
    }

    #[test]
    fn toml_splits_by_sections_and_kv_pairs() {
        let mut content = String::new();
        content.push_str("name = \"test-project-with-long-name\"\n");
        content.push_str("version = \"1.0.0\"\n");
        content.push_str("edition = \"2024\"\n");
        content.push_str("description = \"A test project with enough content to exceed the minimum split size threshold for our text chunker implementation so that format-specific splitting is activated\"\n");
        content.push_str("license = \"MIT\"\n");
        content.push_str("repository = \"https://github.com/example/test-project\"\n");
        content.push_str("readme = \"README.md\"\n");
        content.push_str("keywords = [\"test\", \"project\", \"example\"]\n");
        content.push_str("categories = [\"development-tools\"]\n\n");
        content.push_str("[dependencies]\n");
        content.push_str("serde = \"1.0\"\n");
        content.push_str("anyhow = \"1.0\"\n");
        content.push_str("tokio = \"1.0\"\n");
        content.push_str("turso = \"0.5\"\n");
        content.push_str("reqwest = \"0.13\"\n");
        content.push_str("clap = \"4.6\"\n\n");
        content.push_str("[dev-dependencies]\n");
        content.push_str("tempfile = \"3.0\"\n");
        content.push_str("insta = \"1.0\"\n");
        let chunks = chunk_text_file(&content, "Cargo.toml", "toml");
        assert!(
            chunks.len() >= 2,
            "Should split on sections and KV pairs, got: {:?}",
            chunks.len()
        );
    }

    #[test]
    fn plaintext_splits_by_double_newlines() {
        let mut content = String::new();
        for i in 0..10 {
            content.push_str(&format!("This is paragraph {i} with enough text to be meaningful and exceed minimum size requirements for chunking in our system.\n\n"));
        }
        let chunks = chunk_text_file(&content, "test.txt", "plaintext");
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.content.trim().is_empty());
        }
    }

    #[test]
    fn chunk_kind_display_roundtrips() {
        let kinds = vec![
            ChunkKind::Function,
            ChunkKind::HeadingSection,
            ChunkKind::TopLevelKey,
            ChunkKind::File,
        ];
        for kind in kinds {
            let s = kind.to_string();
            let parsed: ChunkKind = s.parse().expect("should parse");
            assert_eq!(kind, parsed);
        }
    }

    #[test]
    fn oversized_chunk_is_split() {
        let mut content = String::new();
        for i in 0..500 {
            content.push_str(&format!("Line {i}: this is a somewhat long line of text\n"));
        }
        let chunk = TextChunk {
            file_path: "test.txt".to_string(),
            start_line: 1,
            end_line: 500,
            kind: ChunkKind::File,
            name: None,
            content: content.clone(),
        };
        let result = enforce_max_size(chunk);
        assert!(result.len() > 1, "Oversized chunk should be split");
        for sub in &result {
            assert!(sub.content.len() <= MAX_CHUNK_SIZE);
        }
    }

    #[test]
    fn invalid_json_returns_empty() {
        let content = "{not valid json at all";
        let chunks = chunk_json(content, "bad.json");
        assert!(chunks.is_empty());
    }

    #[test]
    fn json_array_returns_empty() {
        let content = "[1, 2, 3]";
        let chunks = chunk_json(content, "arr.json");
        assert!(chunks.is_empty());
    }
}
