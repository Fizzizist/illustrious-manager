use crate::tools::{Tool, ToolError, ToolResult, run_blocking};
use crate::types::ContentBlock;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct SkillTool {
    skills: HashMap<String, PathBuf>,
}

impl SkillTool {
    pub fn new(skills: &HashMap<String, PathBuf>) -> Self {
        Self {
            skills: skills.clone(),
        }
    }
}

#[async_trait]
impl Tool for SkillTool {
    fn name(&self) -> &str {
        "skill"
    }

    fn description(&self) -> &str {
        "Load a skill by name and return its prompt content"
    }

    fn input_schema(&self) -> &Value {
        use std::sync::LazyLock;
        static SCHEMA: LazyLock<Value> = LazyLock::new(|| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to load"
                    }
                },
                "required": ["name"]
            })
        });
        &SCHEMA
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let name =
            input
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidInput {
                    message: "Missing 'name' field".to_string(),
                })?;

        let path = self.skills.get(name).ok_or_else(|| ToolError::Execution {
            tool_name: "skill".to_string(),
            message: format!("Skill '{}' not found", name),
        })?;

        let path_display = path.display().to_string();
        let path = path.clone();
        let content = run_blocking("skill", move || {
            std::fs::read_to_string(&path).map_err(|e| ToolError::Execution {
                tool_name: "skill".to_string(),
                message: format!("Failed to read skill file: {}", e),
            })
        })
        .await?;

        let output = format!("Skill path: {}\n\n{}", path_display, content);

        Ok(ToolResult {
            content: vec![ContentBlock::Text(output)],
            is_error: false,
            agent_events: vec![],
        })
    }
}

fn scan_skills_dir(dir: &Path, skills: &mut HashMap<String, PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let skill_file = path.join("SKILL.md");
                if skill_file.exists()
                    && let Some(name) = path.file_name().and_then(|n| n.to_str())
                {
                    skills.insert(name.to_string(), skill_file);
                }
            }
        }
    }
}

// home is scanned first so that pwd entries override home entries with the same name.
pub fn discover_skills(pwd: &Path, home: &Path) -> HashMap<String, PathBuf> {
    let mut skills = HashMap::new();
    scan_skills_dir(&home.join(".claude/skills"), &mut skills);
    scan_skills_dir(&pwd.join(".claude/skills"), &mut skills);
    skills
}

pub fn discover_skills_from_env() -> HashMap<String, PathBuf> {
    let mut skills = HashMap::new();
    if let Some(home) = dirs::home_dir() {
        scan_skills_dir(&home.join(".claude/skills"), &mut skills);
    }
    if let Ok(pwd) = std::env::current_dir() {
        scan_skills_dir(&pwd.join(".claude/skills"), &mut skills);
    }
    skills
}

/// Extracts the `description` field from YAML frontmatter at the top of a skill file.
///
/// Expects frontmatter delimited by `---` lines, e.g.:
/// ```markdown
/// ---
/// name: my-skill
/// description: what this skill does
/// ---
/// ```
pub fn skill_description(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut lines = content.lines();

    if lines.next()?.trim() != "---" {
        return None;
    }

    for line in lines {
        if line.trim() == "---" {
            break;
        }
        if let Some(rest) = line.strip_prefix("description:") {
            return Some(rest.trim().to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use tempfile::TempDir;

    fn make_skill(base: &Path, name: &str, content: &str) {
        let skill_dir = base.join(".claude/skills").join(name);
        fs::create_dir_all(&skill_dir).expect("create skill dir");
        let mut f = File::create(skill_dir.join("SKILL.md")).expect("create SKILL.md");
        f.write_all(content.as_bytes()).expect("write SKILL.md");
    }

    #[tokio::test]
    async fn discover_skills_finds_skills_in_home() {
        let pwd = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        make_skill(home.path(), "my-skill", "# My Skill\nContent here");

        let skills = discover_skills(pwd.path(), home.path());

        assert!(skills.contains_key("my-skill"), "should find home skill");
    }

    #[tokio::test]
    async fn discover_skills_finds_skills_in_pwd() {
        let pwd = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        make_skill(pwd.path(), "local-skill", "# Local Skill");

        let skills = discover_skills(pwd.path(), home.path());

        assert!(skills.contains_key("local-skill"), "should find pwd skill");
    }

    #[tokio::test]
    async fn discover_skills_finds_skills_from_both_locations() {
        let pwd = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        make_skill(home.path(), "home-skill", "# Home Skill");
        make_skill(pwd.path(), "pwd-skill", "# Pwd Skill");

        let skills = discover_skills(pwd.path(), home.path());

        assert!(skills.contains_key("home-skill"));
        assert!(skills.contains_key("pwd-skill"));
    }

    #[tokio::test]
    async fn discover_skills_ignores_dirs_without_skill_md() {
        let pwd = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let empty_dir = home.path().join(".claude/skills/empty-skill");
        fs::create_dir_all(&empty_dir).expect("create dir");

        let skills = discover_skills(pwd.path(), home.path());

        assert!(
            !skills.contains_key("empty-skill"),
            "should not include skill without SKILL.md"
        );
    }

    #[tokio::test]
    async fn discover_skills_returns_empty_when_no_skill_dirs_exist() {
        let pwd = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();

        let skills = discover_skills(pwd.path(), home.path());

        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn pwd_skill_overrides_home_skill_with_same_name() {
        let pwd = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        make_skill(home.path(), "shared-skill", "# Home version");
        make_skill(pwd.path(), "shared-skill", "# Pwd version");

        let skills = discover_skills(pwd.path(), home.path());

        assert_eq!(skills.len(), 1, "should have one entry for the skill");
        let content = fs::read_to_string(&skills["shared-skill"]).unwrap();
        assert_eq!(
            content, "# Pwd version",
            "pwd skill should override home skill"
        );
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn discover_skills_skips_dirs_with_non_utf8_names() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let pwd = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();

        let skills_dir = home.path().join(".claude/skills");
        fs::create_dir_all(&skills_dir).expect("create skills dir");

        let bad_name = OsStr::from_bytes(b"bad-\xff-skill");
        let bad_dir = skills_dir.join(bad_name);
        fs::create_dir_all(&bad_dir).expect("create non-utf8 dir");
        fs::write(bad_dir.join("SKILL.md"), "content").expect("write SKILL.md");

        let skills = discover_skills(pwd.path(), home.path());

        assert!(
            skills.is_empty(),
            "non-UTF8 skill dir names are silently skipped"
        );
    }

    #[tokio::test]
    async fn skill_tool_execute_returns_file_content_with_skill_path() {
        let dir = TempDir::new().unwrap();
        let skill_file = dir.path().join("my-skill.md");
        fs::write(&skill_file, "# Skill Content\nThis is the skill prompt").unwrap();

        let mut skills = HashMap::new();
        skills.insert("my-skill".to_string(), skill_file.clone());
        let tool = SkillTool::new(&skills);

        let result = tool
            .execute(serde_json::json!({"name": "my-skill"}))
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        match &result.content[0] {
            ContentBlock::Text(text) => {
                assert!(
                    text.contains(&*skill_file.to_string_lossy()),
                    "output should contain the SKILL.md path, got: {}",
                    text
                );
                assert!(
                    text.contains("# Skill Content"),
                    "output should still contain the file content"
                );
            }
            _ => panic!("expected Text content"),
        }
    }

    #[tokio::test]
    async fn skill_tool_execute_returns_file_content_without_path() {
        let dir = TempDir::new().unwrap();
        let skill_file = dir.path().join("my-skill.md");
        fs::write(&skill_file, "# Skill Content\nThis is the skill prompt").unwrap();

        let mut skills = HashMap::new();
        skills.insert("my-skill".to_string(), skill_file.clone());
        let tool = SkillTool::new(&skills);

        let result = tool
            .execute(serde_json::json!({"name": "my-skill"}))
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        let expected = format!(
            "Skill path: {}\n\n# Skill Content\nThis is the skill prompt",
            skill_file.display()
        );
        match &result.content[0] {
            ContentBlock::Text(text) => {
                assert_eq!(text, &expected);
            }
            _ => panic!("expected Text content"),
        }
    }

    #[tokio::test]
    async fn skill_tool_execute_unknown_name_returns_error() {
        let tool = SkillTool::new(&HashMap::new());

        let result = tool
            .execute(serde_json::json!({"name": "nonexistent"}))
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::Execution { message, .. } => {
                assert!(
                    message.contains("nonexistent"),
                    "error should name the skill"
                );
            }
            other => panic!("expected Execution error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn skill_tool_execute_missing_name_field_returns_invalid_input() {
        let tool = SkillTool::new(&HashMap::new());

        let result = tool.execute(serde_json::json!({})).await;

        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    #[tokio::test]
    async fn skill_tool_is_not_a_write_tool() {
        let tool = SkillTool::new(&HashMap::new());
        assert!(!tool.is_write_tool());
    }

    #[tokio::test]
    async fn skill_description_extracts_description_from_frontmatter() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("skill.md");
        fs::write(
            &path,
            "---\nname: my-skill\ndescription: does the thing\n---\n\n# Body",
        )
        .unwrap();

        assert_eq!(skill_description(&path).unwrap(), "does the thing");
    }

    #[tokio::test]
    async fn skill_description_returns_none_when_no_frontmatter() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("skill.md");
        fs::write(&path, "# My Skill\nNo frontmatter here").unwrap();

        assert!(skill_description(&path).is_none());
    }

    #[tokio::test]
    async fn skill_description_returns_none_when_description_field_missing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("skill.md");
        fs::write(&path, "---\nname: my-skill\n---\n\n# Body").unwrap();

        assert!(skill_description(&path).is_none());
    }

    #[tokio::test]
    async fn skill_description_returns_none_for_missing_file() {
        assert!(skill_description(Path::new("/nonexistent/path.md")).is_none());
    }
}
