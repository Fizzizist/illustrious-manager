use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub type SkillMapping = HashMap<String, PathBuf>;

pub fn discover_skills() -> Result<SkillMapping> {
    let mut skills = SkillMapping::new();

    if let Ok(pwd) = std::env::current_dir() {
        let pwd_skills = pwd.join(".claude").join("skills");
        match discover_skills_from_dir(&pwd_skills) {
            Ok(found) => skills.extend(found),
            Err(e) => eprintln!(
                "Warning: failed to discover skills from {}: {}",
                pwd_skills.display(),
                e
            ),
        }
    }

    if let Some(home) = dirs::home_dir() {
        let home_skills = home.join(".claude").join("skills");
        match discover_skills_from_dir(&home_skills) {
            Ok(found) => skills.extend(found),
            Err(e) => eprintln!(
                "Warning: failed to discover skills from {}: {}",
                home_skills.display(),
                e
            ),
        }
    }

    Ok(skills)
}

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
        let skill_name = skill_name.to_string_lossy().to_string();

        let skill_md = entry.path().join("SKILL.md");
        if skill_md.exists() {
            skills.insert(skill_name, skill_md);
        }
    }

    Ok(skills)
}

pub fn load_skill_content(path: &Path) -> Result<String> {
    std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read skill file: {}", path.display()))
}

/// Unknown skill names (those not found in the mapping) are passed through unchanged.
/// This allows forward compatibility with skills that may be added later and enables
/// collaborative workflows where users share prompts referencing skills others may not have.
pub fn process_prompt(input: String, skills: &SkillMapping) -> String {
    let prompt = input.trim();
    if let Some(skill_name_end) = prompt.find(' ').or_else(|| {
        if prompt.starts_with('/') {
            Some(prompt.len())
        } else {
            None
        }
    }) {
        if prompt.starts_with('/') {
            let skill_name = &prompt[1..skill_name_end];
            if let Some(skill_path) = skills.get(skill_name) {
                let remaining_prompt = if skill_name_end < prompt.len() {
                    prompt[skill_name_end..].trim()
                } else {
                    ""
                };

                match load_skill_content(skill_path) {
                    Ok(skill_content) => {
                        if remaining_prompt.is_empty() {
                            skill_content
                        } else {
                            format!("{}\n\n{}", skill_content, remaining_prompt)
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: failed to read skill file '{}': {}",
                            skill_path.display(),
                            e
                        );
                        input
                    }
                }
            } else {
                input
            }
        } else {
            input
        }
    } else if let Some(skill_name) = prompt.strip_prefix('/') {
        if let Some(skill_path) = skills.get(skill_name) {
            match load_skill_content(skill_path) {
                Ok(skill_content) => skill_content,
                Err(e) => {
                    eprintln!(
                        "Warning: failed to read skill file '{}': {}",
                        skill_path.display(),
                        e
                    );
                    input
                }
            }
        } else {
            input
        }
    } else {
        input
    }
}

pub fn format_skills_context(skills: &SkillMapping) -> Option<String> {
    if skills.is_empty() {
        return None;
    }

    let mut lines =
        vec!["The following skills are available for invocation with a `/` prefix:".to_string()];
    for (name, path) in skills {
        lines.push(format!("- /{} (from {})", name, path.display()));
    }
    Some(lines.join("\n"))
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
        assert!(skills.is_empty());
    }

    #[test]
    fn discover_skills_from_nonexistent_directory_returns_empty_mapping() {
        let nonexist = PathBuf::from("/tmp/illustrious_test_nonexistent_12345");
        let skills = discover_skills_from_dir(&nonexist).expect("discover should succeed");
        assert!(skills.is_empty());
    }

    #[test]
    fn discover_skills_finds_skill_folders_with_skill_md() {
        let dir = TestSkillDir::new();
        dir.add_skill("test-skill", "# Test Skill\n\nThis is a test skill.");

        let skills = discover_skills_from_dir(dir.path()).expect("discover should succeed");
        assert_eq!(skills.len(), 1);
        assert!(skills.contains_key("test-skill"));
        assert_eq!(
            skills.get("test-skill").unwrap().file_name().unwrap(),
            "SKILL.md"
        );
    }

    #[test]
    fn discover_skills_ignores_directories_without_skill_md() {
        let dir = TestSkillDir::new();
        dir.add_skill("valid-skill", "# Valid");

        let empty_dir = dir.path().join("empty-skill");
        fs::create_dir(&empty_dir).expect("create empty dir");

        let skills = discover_skills_from_dir(dir.path()).expect("discover should succeed");
        assert_eq!(skills.len(), 1);
        assert!(skills.contains_key("valid-skill"));
        assert!(!skills.contains_key("empty-skill"));
    }

    #[test]
    fn discover_skills_finds_multiple_skills() {
        let dir = TestSkillDir::new();
        dir.add_skill("skill1", "# Skill 1");
        dir.add_skill("skill2", "# Skill 2");
        dir.add_skill("skill3", "# Skill 3");

        let skills = discover_skills_from_dir(dir.path()).expect("discover should succeed");
        assert_eq!(skills.len(), 3);
        assert!(skills.contains_key("skill1"));
        assert!(skills.contains_key("skill2"));
        assert!(skills.contains_key("skill3"));
    }

    #[test]
    fn discover_skills_handles_directory_with_non_directory_entries() {
        let dir = TestSkillDir::new();
        dir.add_skill("valid-skill", "# Valid");

        let file_path = dir.path().join("not-a-directory");
        fs::write(&file_path, "not a directory").expect("write file");

        let skills = discover_skills_from_dir(dir.path()).expect("discover should succeed");
        assert_eq!(skills.len(), 1);
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
        assert!(result.is_err());
    }

    #[test]
    fn format_skills_context_returns_none_for_empty_mapping() {
        let skills = SkillMapping::new();
        assert!(format_skills_context(&skills).is_none());
    }

    #[test]
    fn format_skills_context_lists_skill_names_and_paths() {
        let dir = TestSkillDir::new();
        let path1 = dir.add_skill("commit", "# Commit");
        let path2 = dir.add_skill("review", "# Review");

        let mut skills = SkillMapping::new();
        skills.insert("commit".to_string(), path1);
        skills.insert("review".to_string(), path2);

        let ctx = format_skills_context(&skills).expect("should return Some");
        assert!(ctx.contains("/commit"));
        assert!(ctx.contains("/review"));
        assert!(ctx.contains("SKILL.md"));
    }
}
