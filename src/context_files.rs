use anyhow::Result;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct ContextFile {
    pub path: PathBuf,
    pub content: String,
}

pub fn discover_context_files_from_env() -> Result<Vec<ContextFile>> {
    let pwd = std::env::current_dir()?;
    let home = dirs::home_dir().unwrap_or_else(|| {
        eprintln!("Warning: Could not determine home directory, skipping home context files");
        pwd.clone()
    });
    discover_context_files(&pwd, &home)
}

pub fn discover_context_files(pwd: &Path, home: &Path) -> Result<Vec<ContextFile>> {
    let mut found = Vec::new();

    let locations = vec![
        pwd.join(".claude/CLAUDE.md"),
        pwd.join("CLAUDE.md"),
        pwd.join("AGENTS.md"),
        home.join(".claude/CLAUDE.md"),
        home.join("CLAUDE.md"),
        home.join("AGENTS.md"),
    ];

    for path in locations {
        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            found.push(ContextFile { path, content });
        }
    }

    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn discover_context_files_finds_all_files_in_pwd() {
        let temp_pwd = TempDir::new().expect("create temp pwd");
        let temp_home = TempDir::new().expect("create temp home");

        // Create files in pwd
        let claude_md = temp_pwd.path().join("CLAUDE.md");
        let agents_md = temp_pwd.path().join("AGENTS.md");
        let dot_claude = temp_pwd.path().join(".claude");
        fs::create_dir(&dot_claude).expect("create .claude dir");
        let dot_claude_md = dot_claude.join("CLAUDE.md");

        File::create(&claude_md)
            .unwrap()
            .write_all(b"PWD CLAUDE content")
            .unwrap();
        File::create(&agents_md)
            .unwrap()
            .write_all(b"PWD AGENTS content")
            .unwrap();
        File::create(&dot_claude_md)
            .unwrap()
            .write_all(b"PWD .claude CLAUDE content")
            .unwrap();

        let found = discover_context_files(temp_pwd.path(), temp_home.path())
            .expect("discover should succeed");

        assert_eq!(found.len(), 3, "should find all 3 files in pwd");

        let contents: Vec<_> = found.iter().map(|f| f.content.as_str()).collect();
        assert!(contents.contains(&"PWD CLAUDE content"));
        assert!(contents.contains(&"PWD AGENTS content"));
        assert!(contents.contains(&"PWD .claude CLAUDE content"));
    }

    #[test]
    fn discover_context_files_finds_files_in_home() {
        let temp_pwd = TempDir::new().expect("create temp pwd");
        let temp_home = TempDir::new().expect("create temp home");

        // Create files in home
        let claude_md = temp_home.path().join("CLAUDE.md");
        let agents_md = temp_home.path().join("AGENTS.md");
        let dot_claude = temp_home.path().join(".claude");
        fs::create_dir(&dot_claude).expect("create .claude dir");
        let dot_claude_md = dot_claude.join("CLAUDE.md");

        File::create(&claude_md)
            .unwrap()
            .write_all(b"HOME CLAUDE content")
            .unwrap();
        File::create(&agents_md)
            .unwrap()
            .write_all(b"HOME AGENTS content")
            .unwrap();
        File::create(&dot_claude_md)
            .unwrap()
            .write_all(b"HOME .claude CLAUDE content")
            .unwrap();

        let found = discover_context_files(temp_pwd.path(), temp_home.path())
            .expect("discover should succeed");

        assert_eq!(found.len(), 3, "should find all 3 files in home");

        let contents: Vec<_> = found.iter().map(|f| f.content.as_str()).collect();
        assert!(contents.contains(&"HOME CLAUDE content"));
        assert!(contents.contains(&"HOME AGENTS content"));
        assert!(contents.contains(&"HOME .claude CLAUDE content"));
    }

    #[test]
    fn discover_context_files_finds_files_from_both_locations() {
        let temp_pwd = TempDir::new().expect("create temp pwd");
        let temp_home = TempDir::new().expect("create temp home");

        // Create one file in each location
        let pwd_claude = temp_pwd.path().join("CLAUDE.md");
        let home_agents = temp_home.path().join("AGENTS.md");

        File::create(&pwd_claude)
            .unwrap()
            .write_all(b"PWD content")
            .unwrap();
        File::create(&home_agents)
            .unwrap()
            .write_all(b"HOME content")
            .unwrap();

        let found = discover_context_files(temp_pwd.path(), temp_home.path())
            .expect("discover should succeed");

        assert_eq!(found.len(), 2, "should find both files");

        let contents: Vec<_> = found.iter().map(|f| f.content.as_str()).collect();
        assert!(contents.contains(&"PWD content"));
        assert!(contents.contains(&"HOME content"));
    }

    #[test]
    fn discover_context_files_returns_empty_when_no_files_exist() {
        let temp_pwd = TempDir::new().expect("create temp pwd");
        let temp_home = TempDir::new().expect("create temp home");

        let found = discover_context_files(temp_pwd.path(), temp_home.path())
            .expect("discover should succeed");

        assert!(
            found.is_empty(),
            "should return empty vec when no files exist"
        );
    }

    #[test]
    fn discover_context_files_includes_correct_paths() {
        let temp_pwd = TempDir::new().expect("create temp pwd");
        let temp_home = TempDir::new().expect("create temp home");

        let pwd_claude = temp_pwd.path().join("CLAUDE.md");
        File::create(&pwd_claude)
            .unwrap()
            .write_all(b"content")
            .unwrap();

        let found = discover_context_files(temp_pwd.path(), temp_home.path())
            .expect("discover should succeed");

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].path, pwd_claude);
    }
}
