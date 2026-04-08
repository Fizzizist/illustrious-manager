use crate::tools::{Tool, ToolError, ToolResult};
use crate::types::ContentBlock;
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

    fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
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

        let content = std::fs::read_to_string(path).map_err(|e| ToolError::Execution {
            tool_name: "skill".to_string(),
            message: format!("Failed to read skill file: {}", e),
        })?;

        Ok(ToolResult {
            content: vec![ContentBlock::Text(content)],
            is_error: false,
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

pub fn first_line(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()?
        .lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim_start_matches('#').trim().to_string())
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

    #[test]
    fn discover_skills_finds_skills_in_home() {
        let pwd = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        make_skill(home.path(), "my-skill", "# My Skill\nContent here");

        let skills = discover_skills(pwd.path(), home.path());

        assert!(skills.contains_key("my-skill"), "should find home skill");
    }

    #[test]
    fn discover_skills_finds_skills_in_pwd() {
        let pwd = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        make_skill(pwd.path(), "local-skill", "# Local Skill");

        let skills = discover_skills(pwd.path(), home.path());

        assert!(skills.contains_key("local-skill"), "should find pwd skill");
    }

    #[test]
    fn discover_skills_finds_skills_from_both_locations() {
        let pwd = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        make_skill(home.path(), "home-skill", "# Home Skill");
        make_skill(pwd.path(), "pwd-skill", "# Pwd Skill");

        let skills = discover_skills(pwd.path(), home.path());

        assert!(skills.contains_key("home-skill"));
        assert!(skills.contains_key("pwd-skill"));
    }

    #[test]
    fn discover_skills_ignores_dirs_without_skill_md() {
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

    #[test]
    fn discover_skills_returns_empty_when_no_skill_dirs_exist() {
        let pwd = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();

        let skills = discover_skills(pwd.path(), home.path());

        assert!(skills.is_empty());
    }

    #[test]
    fn pwd_skill_overrides_home_skill_with_same_name() {
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

    #[test]
    fn skill_tool_execute_returns_file_content() {
        let dir = TempDir::new().unwrap();
        let skill_file = dir.path().join("my-skill.md");
        fs::write(&skill_file, "# Skill Content\nThis is the skill prompt").unwrap();

        let mut skills = HashMap::new();
        skills.insert("my-skill".to_string(), skill_file);
        let tool = SkillTool::new(&skills);

        let result = tool
            .execute(serde_json::json!({"name": "my-skill"}))
            .expect("should succeed");

        assert!(!result.is_error);
        match &result.content[0] {
            ContentBlock::Text(text) => {
                assert_eq!(text, "# Skill Content\nThis is the skill prompt")
            }
            _ => panic!("expected Text content"),
        }
    }

    #[test]
    fn skill_tool_execute_unknown_name_returns_error() {
        let tool = SkillTool::new(&HashMap::new());

        let result = tool.execute(serde_json::json!({"name": "nonexistent"}));

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

    #[test]
    fn skill_tool_execute_missing_name_field_returns_invalid_input() {
        let tool = SkillTool::new(&HashMap::new());

        let result = tool.execute(serde_json::json!({}));

        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    #[test]
    fn skill_tool_is_not_a_write_tool() {
        let tool = SkillTool::new(&HashMap::new());
        assert!(!tool.is_write_tool());
    }

    #[test]
    fn first_line_strips_markdown_heading_marker() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("skill.md");
        fs::write(&path, "# My Skill Title\nsome body text").unwrap();

        assert_eq!(first_line(&path).unwrap(), "My Skill Title");
    }

    #[test]
    fn first_line_skips_leading_blank_lines() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("skill.md");
        fs::write(&path, "\n\n# Title").unwrap();

        assert_eq!(first_line(&path).unwrap(), "Title");
    }

    #[test]
    fn first_line_returns_none_for_empty_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("skill.md");
        fs::write(&path, "").unwrap();

        assert!(first_line(&path).is_none());
    }
}
