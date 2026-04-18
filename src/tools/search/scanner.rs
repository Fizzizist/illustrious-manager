use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileType {
    Rust,
    #[cfg(feature = "ts-typescript")]
    TypeScript,
    #[cfg(feature = "ts-typescript")]
    Tsx,
    #[cfg(feature = "ts-python")]
    Python,
    #[cfg(feature = "ts-go")]
    Go,
    #[cfg(feature = "ts-java")]
    Java,
    #[cfg(feature = "ts-c")]
    C,
    #[cfg(feature = "ts-cpp")]
    Cpp,
    Markdown,
    Yaml,
    Json,
    Toml,
    Plaintext,
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileType::Rust => write!(f, "rust"),
            #[cfg(feature = "ts-typescript")]
            FileType::TypeScript => write!(f, "typescript"),
            #[cfg(feature = "ts-typescript")]
            FileType::Tsx => write!(f, "tsx"),
            #[cfg(feature = "ts-python")]
            FileType::Python => write!(f, "python"),
            #[cfg(feature = "ts-go")]
            FileType::Go => write!(f, "go"),
            #[cfg(feature = "ts-java")]
            FileType::Java => write!(f, "java"),
            #[cfg(feature = "ts-c")]
            FileType::C => write!(f, "c"),
            #[cfg(feature = "ts-cpp")]
            FileType::Cpp => write!(f, "cpp"),
            FileType::Markdown => write!(f, "markdown"),
            FileType::Yaml => write!(f, "yaml"),
            FileType::Json => write!(f, "json"),
            FileType::Toml => write!(f, "toml"),
            FileType::Plaintext => write!(f, "plaintext"),
        }
    }
}

impl std::str::FromStr for FileType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "rust" => Ok(FileType::Rust),
            #[cfg(feature = "ts-typescript")]
            "typescript" => Ok(FileType::TypeScript),
            #[cfg(feature = "ts-typescript")]
            "tsx" => Ok(FileType::Tsx),
            #[cfg(feature = "ts-python")]
            "python" => Ok(FileType::Python),
            #[cfg(feature = "ts-go")]
            "go" => Ok(FileType::Go),
            #[cfg(feature = "ts-java")]
            "java" => Ok(FileType::Java),
            #[cfg(feature = "ts-c")]
            "c" => Ok(FileType::C),
            #[cfg(feature = "ts-cpp")]
            "cpp" => Ok(FileType::Cpp),
            "markdown" => Ok(FileType::Markdown),
            "yaml" => Ok(FileType::Yaml),
            "json" => Ok(FileType::Json),
            "toml" => Ok(FileType::Toml),
            "plaintext" => Ok(FileType::Plaintext),
            other => Err(format!("Unknown file type: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScannedFile {
    pub file_path: String,
    pub file_type: FileType,
    pub mtime: f64,
}

const MAX_FILE_SIZE: u64 = 1024 * 1024;

fn extension_map() -> HashMap<&'static str, FileType> {
    let mut m = HashMap::new();
    m.insert("rs", FileType::Rust);
    #[cfg(feature = "ts-typescript")]
    {
        m.insert("ts", FileType::TypeScript);
        m.insert("tsx", FileType::Tsx);
    }
    #[cfg(feature = "ts-python")]
    {
        m.insert("py", FileType::Python);
    }
    #[cfg(feature = "ts-go")]
    {
        m.insert("go", FileType::Go);
    }
    #[cfg(feature = "ts-java")]
    {
        m.insert("java", FileType::Java);
    }
    #[cfg(feature = "ts-c")]
    {
        m.insert("c", FileType::C);
        m.insert("h", FileType::C);
    }
    #[cfg(feature = "ts-cpp")]
    {
        m.insert("cpp", FileType::Cpp);
        m.insert("hpp", FileType::Cpp);
        m.insert("cc", FileType::Cpp);
        m.insert("cxx", FileType::Cpp);
        m.insert("hh", FileType::Cpp);
        m.insert("hxx", FileType::Cpp);
    }
    m.insert("md", FileType::Markdown);
    m.insert("mdx", FileType::Markdown);
    m.insert("yml", FileType::Yaml);
    m.insert("yaml", FileType::Yaml);
    m.insert("json", FileType::Json);
    m.insert("toml", FileType::Toml);
    m.insert("txt", FileType::Plaintext);
    m.insert("cfg", FileType::Plaintext);
    m.insert("ini", FileType::Plaintext);
    m.insert("env", FileType::Plaintext);
    m.insert("conf", FileType::Plaintext);
    m
}

pub fn detect_file_type(file_path: &str) -> Option<FileType> {
    let path = Path::new(file_path);
    let ext = path.extension()?.to_str()?.to_lowercase();
    extension_map().get(&*ext).cloned()
}

pub fn scan_project(project_root: &Path) -> Vec<ScannedFile> {
    let mut results = Vec::new();
    let walker = ignore::WalkBuilder::new(project_root)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();

    for entry in walker.flatten() {
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        let path = entry.path();

        let file_size = match fs::metadata(path) {
            Ok(meta) => meta.len(),
            Err(_) => continue,
        };
        if file_size == 0 || file_size > MAX_FILE_SIZE {
            continue;
        }

        let relative = match path.strip_prefix(project_root) {
            Ok(r) => r.to_string_lossy().to_string(),
            Err(_) => continue,
        };

        let file_type = match detect_file_type(&relative) {
            Some(ft) => ft,
            None => continue,
        };

        let mtime = fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        results.push(ScannedFile {
            file_path: relative,
            file_type,
            mtime,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn detect_rust_file() {
        assert_eq!(detect_file_type("src/main.rs"), Some(FileType::Rust));
    }

    #[test]
    fn detect_markdown_file() {
        assert_eq!(detect_file_type("README.md"), Some(FileType::Markdown));
    }

    #[test]
    fn detect_yaml_file() {
        assert_eq!(detect_file_type("config.yaml"), Some(FileType::Yaml));
        assert_eq!(detect_file_type("config.yml"), Some(FileType::Yaml));
    }

    #[test]
    fn detect_json_file() {
        assert_eq!(detect_file_type("package.json"), Some(FileType::Json));
    }

    #[test]
    fn detect_toml_file() {
        assert_eq!(detect_file_type("Cargo.toml"), Some(FileType::Toml));
    }

    #[test]
    fn detect_unknown_extension_returns_none() {
        assert_eq!(detect_file_type("image.png"), None);
    }

    #[test]
    fn detect_no_extension_returns_none() {
        assert_eq!(detect_file_type("Makefile"), None);
    }

    #[test]
    fn scan_empty_directory() {
        let temp = TempDir::new().expect("temp dir");
        let results = scan_project(temp.path());
        assert!(results.is_empty());
    }

    #[test]
    fn scan_finds_files_with_known_extensions() {
        let temp = TempDir::new().expect("temp dir");
        fs::write(temp.path().join("main.rs"), "fn main() {}").expect("write");
        fs::write(temp.path().join("README.md"), "# Hello").expect("write");

        let results = scan_project(temp.path());
        assert_eq!(results.len(), 2);

        let paths: Vec<&str> = results.iter().map(|f| f.file_path.as_str()).collect();
        assert!(paths.contains(&"main.rs"));
        assert!(paths.contains(&"README.md"));
    }

    #[test]
    fn scan_skips_empty_files() {
        let temp = TempDir::new().expect("temp dir");
        fs::write(temp.path().join("empty.rs"), "").expect("write");
        fs::write(temp.path().join("main.rs"), "fn main() {}").expect("write");

        let results = scan_project(temp.path());
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn scan_skips_unknown_extensions() {
        let temp = TempDir::new().expect("temp dir");
        fs::write(temp.path().join("image.png"), "data").expect("write");

        let results = scan_project(temp.path());
        assert!(results.is_empty());
    }

    #[test]
    fn file_type_display_roundtrips() {
        let ft = FileType::Rust;
        let s = ft.to_string();
        let parsed: FileType = s.parse().expect("should parse");
        assert_eq!(ft, parsed);
    }
}
