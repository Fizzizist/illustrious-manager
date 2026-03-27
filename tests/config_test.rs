use std::io::Write;

use tempfile::NamedTempFile;

// We'll test config by calling the public API once it exists.
// For now, write the tests that define the expected behavior.

#[test]
fn test_parse_valid_config() {
    let toml_content = r#"
[vertex]
project = "my-project"
region = "us-east5"
model = "claude-sonnet-4-20250514"
"#;
    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "{}", toml_content).unwrap();

    let config = illustrious_manager::config::load_config_from_path(tmp.path()).unwrap();
    assert_eq!(config.vertex.project, "my-project");
    assert_eq!(config.vertex.region, "us-east5");
    assert_eq!(config.vertex.model, "claude-sonnet-4-20250514");
}

#[test]
fn test_parse_config_with_defaults() {
    let toml_content = r#"
[vertex]
project = "my-project"
"#;
    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "{}", toml_content).unwrap();

    let config = illustrious_manager::config::load_config_from_path(tmp.path()).unwrap();
    assert_eq!(config.vertex.project, "my-project");
    assert_eq!(config.vertex.region, "us-east5");
    assert_eq!(config.vertex.model, "claude-sonnet-4-20250514");
}

#[test]
fn test_cli_overrides_config() {
    let toml_content = r#"
[vertex]
project = "file-project"
region = "us-east5"
model = "claude-sonnet-4-20250514"
"#;
    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "{}", toml_content).unwrap();

    let mut config = illustrious_manager::config::load_config_from_path(tmp.path()).unwrap();

    // Simulate CLI overrides
    illustrious_manager::config::apply_overrides(
        &mut config,
        Some("cli-project"),
        Some("europe-west1"),
        Some("claude-opus-4-20250514"),
    );

    assert_eq!(config.vertex.project, "cli-project");
    assert_eq!(config.vertex.region, "europe-west1");
    assert_eq!(config.vertex.model, "claude-opus-4-20250514");
}

#[test]
fn test_validate_empty_project() {
    let toml_content = r#"
[vertex]
project = ""
region = "us-east5"
model = "claude-sonnet-4-20250514"
"#;
    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "{}", toml_content).unwrap();

    let config = illustrious_manager::config::load_config_from_path(tmp.path()).unwrap();
    let result = illustrious_manager::config::validate(&config);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("project"),
        "Error should mention 'project'"
    );
}
