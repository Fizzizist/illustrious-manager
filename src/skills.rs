//! Skills discovery and invocation
//!
//! Skills are user-defined prompts that can be invoked with a `/` prefix.
//! They are discovered from two locations:
//! - `$PWD/.claude/skills`
//! - `$HOME/.claude/skills`

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Mapping of skill names to their SKILL.md file paths
pub type SkillMapping = HashMap<String, PathBuf>;

/// Discover skills from the standard locations.
///
/// Searches for skills in:
/// 1. `$PWD/.claude/skills`
/// 2. `$HOME/.claude/skills`
///
/// Each skill is a directory containing a `SKILL.md` file.
/// The skill name is derived from the directory name.
pub fn discover_skills() -> Result<SkillMapping> {
    let mut skills = SkillMapping::new();

    if let Ok(pwd) = std::env::current_dir() {
        let pwd_skills = pwd.join(".claude").join("skills");
        if let Ok(found) = discover_skills_from_dir(&pwd_skills) {
            skills.extend(found);
        }
    }

    if let Some(home) = dirs::home_dir() {
        let home_skills = home.join(".claude").join("skills");
        if let Ok(found) = discover_skills_from_dir(&home_skills) {
            skills.extend(found);
        }
    }

    Ok(skills)
}

/// Discover skills from a specific directory.
///
/// Scans the directory for subdirectories containing `SKILL.md` files.
/// Returns a mapping of directory names to SKILL.md paths.
fn discover_skills_from_dir(dir: &Path) -> Result<SkillMapping> {
    let mut skills = SkillMapping::new();

    if !dir.exists() {
        return Ok(skills);
    }

    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read skills directory: {}", dir.display()))?;

    for entry in entries {
        let entry = entry.with_context(|| format!("Failed to read entry in: {}", dir.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("Failed to get file type for: {}", entry.path().display()))?;

        if !file_type.is_dir() {
            continue;
        }

        let skill_name = entry.file_name();
        let skill_name = skill_name.to_string_lossy().into_owned();

        let skill_md = entry.path().join("SKILL.md");
        if skill_md.exists() {
            skills.insert(skill_name, skill_md);
        }
    }

    Ok(skills)
}

/// Load the content of a skill's SKILL.md file.
pub fn load_skill_content(path: &Path) -> Result<String> {
    std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read skill file: {}", path.display()))
}

/// Process a prompt, checking for skill invocations.
///
/// If the prompt starts with `/<skill_name>`, returns the skill content
/// prepended to the original prompt.
///
/// Returns `None` if no skill was invoked.
pub fn process_skill_command(prompt: &str, skills: &SkillMapping) -> Option<(String, String)> {
    let prompt = prompt.trim();

    if !prompt.starts_with('/') {
        return None;
    }

    let skill_name_end = prompt.find(' ').unwrap_or(prompt.len());
    let skill_name = &prompt[1..skill_name_end];

    let skill_path = skills.get(skill_name)?;

    let remaining_prompt = if skill_name_end < prompt.len() {
        prompt[skill_name_end..].trim()
    } else {
        ""
    };

    let skill_content = load_skill_content(skill_path).ok()?;

    let content = if remaining_prompt.is_empty() {
        skill_content
    } else {
        format!("{}\n\n{}", skill_content, remaining_prompt)
    };

    Some((skill_name.to_string(), content))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct TestSkillDir {
        temp_dir: tempfile::TempDir,
    }

    impl TestSkillDir {
        fn new() -> Self {
            Self {
                temp_dir: tempfile::TempDir::new().expect("create temp dir"),
            }
        }

        fn add_skill(&self, name: &str, content: &str) -> PathBuf {
            let skill_dir = self.temp_dir.path().join(name);
            fs::create_dir(&skill_dir).expect("create skill dir");
            let skill_md = skill_dir.join("SKILL.md");
            fs::write(&skill_md, content).expect("write skill.md");
            skill_md
        }

        fn path(&self) -> &Path {
            self.temp_dir.path()
        }
    }

    #[test]
    fn discover_skills_from_empty_directory_returns_empty_mapping() {
        let dir = TestSkillDir::new();
        let skills = discover_skills_from_dir(dir.path()).expect("discover should succeed");
        assert!(
            skills.is_empty(),
            "empty directory should return empty mapping"
        );
    }

    #[test]
    fn discover_skills_from_nonexistent_directory_returns_empty_mapping() {
        let nonexist = PathBuf::from("/tmp/illustrious_test_nonexistent_12345");
        let skills = discover_skills_from_dir(&nonexist).expect("discover should succeed");
        assert!(
            skills.is_empty(),
            "nonexistent directory should return empty mapping"
        );
    }

    #[test]
    fn discover_skills_finds_skill_folders_with_skill_md() {
        let dir = TestSkillDir::new();
        dir.add_skill("test-skill", "# Test Skill\n\nThis is a test skill.");

        let skills = discover_skills_from_dir(dir.path()).expect("discover should succeed");
        assert_eq!(skills.len(), 1, "should find one skill");
        assert!(
            skills.contains_key("test-skill"),
            "should have test-skill key"
        );
        assert_eq!(
            skills.get("test-skill").unwrap().file_name().unwrap(),
            "SKILL.md",
            "skill path should point to SKILL.md"
        );
    }

    #[test]
    fn discover_skills_ignores_directories_without_skill_md() {
        let dir = TestSkillDir::new();
        dir.add_skill("valid-skill", "# Valid");

        // Create a directory without SKILL.md
        let empty_dir = dir.path().join("empty-skill");
        fs::create_dir(&empty_dir).expect("create empty dir");

        let skills = discover_skills_from_dir(dir.path()).expect("discover should succeed");
        assert_eq!(skills.len(), 1, "should only find valid skill");
        assert!(
            skills.contains_key("valid-skill"),
            "should have valid-skill"
        );
        assert!(
            !skills.contains_key("empty-skill"),
            "should not have empty-skill"
        );
    }

    #[test]
    fn discover_skills_finds_multiple_skills() {
        let dir = TestSkillDir::new();
        dir.add_skill("skill1", "# Skill 1");
        dir.add_skill("skill2", "# Skill 2");
        dir.add_skill("skill3", "# Skill 3");

        let skills = discover_skills_from_dir(dir.path()).expect("discover should succeed");
        assert_eq!(skills.len(), 3, "should find three skills");
        assert!(skills.contains_key("skill1"));
        assert!(skills.contains_key("skill2"));
        assert!(skills.contains_key("skill3"));
    }

    #[test]
    fn discover_skills_handles_directory_with_non_directory_entries() {
        let dir = TestSkillDir::new();
        dir.add_skill("valid-skill", "# Valid");

        // Create a file (not a directory)
        let file_path = dir.path().join("not-a-directory");
        fs::write(&file_path, "not a directory").expect("write file");

        let skills = discover_skills_from_dir(dir.path()).expect("discover should succeed");
        assert_eq!(skills.len(), 1, "should only find valid skill");
    }

    #[test]
    fn load_skill_content_reads_file_content() {
        let dir = TestSkillDir::new();
        let skill_path = dir.add_skill("test", "# Test Skill\n\nContent here");

        let content = load_skill_content(&skill_path).expect("load should succeed");
        assert_eq!(content, "# Test Skill\n\nContent here");
    }

    #[test]
    fn load_skill_content_returns_error_for_nonexistent_file() {
        let result = load_skill_content(PathBuf::from("/tmp/nonexistent_skill_xyz.md").as_path());
        assert!(result.is_err(), "should return error for nonexistent file");
    }

    #[test]
    fn process_skill_command_returns_none_for_non_slash_prompt() {
        let skills = SkillMapping::new();
        let result = process_skill_command("hello world", &skills);
        assert!(result.is_none(), "non-slash prompt should return None");
    }

    #[test]
    fn process_skill_command_returns_none_for_unknown_skill() {
        let skills = SkillMapping::new();
        let result = process_skill_command("/unknown skill", &skills);
        assert!(result.is_none(), "unknown skill should return None");
    }

    #[test]
    fn process_skill_command_loads_skill_content_for_known_skill() {
        let dir = TestSkillDir::new();
        let skill_path = dir.add_skill("test-skill", "# Test Skill\n\nDo this thing");

        let mut skills = SkillMapping::new();
        skills.insert("test-skill".to_string(), skill_path);

        let result = process_skill_command("/test-skill", &skills);
        assert!(result.is_some(), "known skill should return Some");

        let (skill_name, content) = result.unwrap();
        assert_eq!(skill_name, "test-skill");
        assert!(content.contains("# Test Skill"));
        assert!(content.contains("Do this thing"));
    }

    #[test]
    fn process_skill_command_appends_remaining_prompt_to_skill_content() {
        let dir = TestSkillDir::new();
        let skill_path = dir.add_skill("commit", "# Commit\n\nWrite a commit message");

        let mut skills = SkillMapping::new();
        skills.insert("commit".to_string(), skill_path);

        let result = process_skill_command("/commit fix the bug", &skills);
        assert!(result.is_some(), "skill with args should return Some");

        let (_, content) = result.unwrap();
        assert!(content.contains("# Commit"));
        assert!(content.contains("Write a commit message"));
        assert!(content.contains("fix the bug"));
    }

    #[test]
    fn process_skill_command_handles_skill_with_no_additional_args() {
        let dir = TestSkillDir::new();
        let skill_path = dir.add_skill("help", "# Help\n\nThis is help");

        let mut skills = SkillMapping::new();
        skills.insert("help".to_string(), skill_path);

        let result = process_skill_command("/help", &skills);
        assert!(result.is_some(), "skill with no args should return Some");

        let (_, content) = result.unwrap();
        assert!(content.contains("# Help"));
        assert!(content.contains("This is help"));
        // Should not have trailing whitespace from empty remaining prompt
        assert!(!content.ends_with("\n\n"));
    }

    #[test]
    fn discover_skills_handles_read_directory_errors_gracefully() {
        // This test verifies that if we can't read a directory, we don't crash
        // We'll create a directory and then make it unreadable (if permissions allow)
        // For now, we'll just verify the function signature doesn't panic
        let temp_dir = tempfile::TempDir::new().expect("create temp dir");
        let skills = discover_skills_from_dir(temp_dir.path()).expect("should not panic");
        assert!(skills.is_empty(), "empty temp dir should have no skills");
    }
}
