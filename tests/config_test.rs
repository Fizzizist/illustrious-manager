use std::io::Write;

use tempfile::{NamedTempFile, TempDir};

#[test]
fn test_parse_valid_config() {
    let toml_content = r#"
[vertex]
project = "my-project"
region = "us-east5"
model = "claude-sonnet-4-20250514"
"#;
    let mut tmp = NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{}", toml_content).expect("Failed to write to temp file");

    let config = illustrious_manager::config::load_config_from_path(tmp.path())
        .expect("Failed to load config from temp file");
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
    let mut tmp = NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{}", toml_content).expect("Failed to write to temp file");

    let config = illustrious_manager::config::load_config_from_path(tmp.path())
        .expect("Failed to load config from temp file");
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
    let mut tmp = NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{}", toml_content).expect("Failed to write to temp file");

    let mut config = illustrious_manager::config::load_config_from_path(tmp.path())
        .expect("Failed to load config from temp file");

    // Simulate CLI overrides for all fields
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
fn test_partial_cli_overrides() {
    let toml_content = r#"
[vertex]
project = "file-project"
region = "us-central1"
model = "claude-haiku-4-20250514"
"#;
    let mut tmp = NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{}", toml_content).expect("Failed to write to temp file");

    let mut config = illustrious_manager::config::load_config_from_path(tmp.path())
        .expect("Failed to load config from temp file");

    // Override only the project, leaving region and model from config file
    illustrious_manager::config::apply_overrides(&mut config, Some("cli-project"), None, None);

    assert_eq!(config.vertex.project, "cli-project");
    assert_eq!(config.vertex.region, "us-central1"); // unchanged
    assert_eq!(config.vertex.model, "claude-haiku-4-20250514"); // unchanged
}

#[test]
fn test_validate_empty_project() {
    let toml_content = r#"
[vertex]
project = ""
region = "us-east5"
model = "claude-sonnet-4-20250514"
"#;
    let mut tmp = NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{}", toml_content).expect("Failed to write to temp file");

    let config = illustrious_manager::config::load_config_from_path(tmp.path())
        .expect("Failed to load config from temp file");
    let result = illustrious_manager::config::validate(&config, Some(tmp.path()));
    assert!(result.is_err());
    let err_msg = result.expect_err("Expected validation to fail").to_string();
    assert!(
        err_msg.contains("project"),
        "Error should mention 'project'"
    );
}

#[test]
fn test_validate_shows_correct_path_for_custom_config() {
    let toml_content = r#"
[vertex]
project = ""
"#;
    let mut tmp = NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{}", toml_content).expect("Failed to write to temp file");

    let config = illustrious_manager::config::load_config_from_path(tmp.path())
        .expect("Failed to load config from temp file");
    let result = illustrious_manager::config::validate(&config, Some(tmp.path()));

    assert!(result.is_err());
    let err_msg = result.expect_err("Expected validation to fail").to_string();
    // The error message should include the actual temp file path, not the default path
    assert!(
        err_msg.contains(&tmp.path().display().to_string()),
        "Error should mention the actual config path that was used"
    );
}

#[test]
fn test_load_config_auto_creates_file() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config_path = temp_dir.path().join("new_config.toml");

    // Verify the file doesn't exist yet
    assert!(!config_path.exists());

    // Load config, which should auto-create the file
    let config = illustrious_manager::config::load_config(Some(&config_path))
        .expect("Failed to auto-create and load config");

    // Verify the file was created
    assert!(config_path.exists());

    // Verify the config has default values and empty project
    assert_eq!(config.vertex.project, "");
    assert_eq!(config.vertex.region, "us-east5");
    assert_eq!(config.vertex.model, "claude-sonnet-4-20250514");

    // Verify the file content matches the template
    let file_content =
        std::fs::read_to_string(&config_path).expect("Failed to read created config file");
    assert!(file_content.contains("[vertex]"));
    assert!(file_content.contains("project = \"\""));
    assert!(file_content.contains("region = \"us-east5\""));
    assert!(file_content.contains("model = \"claude-sonnet-4-20250514\""));
    assert!(file_content.contains("# Required: your GCP project ID"));
}
