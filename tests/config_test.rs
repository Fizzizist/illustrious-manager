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

    illustrious_manager::config::apply_overrides(&mut config, Some("cli-project"), None, None);

    assert_eq!(config.vertex.project, "cli-project");
    assert_eq!(config.vertex.region, "us-central1");
    assert_eq!(config.vertex.model, "claude-haiku-4-20250514");
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
    assert!(
        err_msg.contains(&tmp.path().display().to_string()),
        "Error should mention the actual config path that was used"
    );
}

#[test]
fn test_load_config_auto_creates_file() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config_path = temp_dir.path().join("new_config.toml");

    assert!(!config_path.exists());

    let config = illustrious_manager::config::load_config(Some(&config_path))
        .expect("Failed to auto-create and load config");

    assert!(config_path.exists());

    assert_eq!(config.vertex.project, "");
    assert_eq!(config.vertex.region, "us-east5");
    assert_eq!(config.vertex.model, "claude-sonnet-4-20250514");

    let file_content =
        std::fs::read_to_string(&config_path).expect("Failed to read created config file");
    assert!(file_content.contains("[vertex]"));
    assert!(file_content.contains("project = \"\""));
    assert!(file_content.contains("region = \"us-east5\""));
    assert!(file_content.contains("model = \"claude-sonnet-4-20250514\""));
    assert!(file_content.contains("# Required: your GCP project ID"));
}

#[test]
fn test_parse_zai_config() {
    let toml_content = r#"
backend = "zai"

[vertex]
project = "my-project"
region = "us-east5"
model = "claude-sonnet-4-20250514"

[zai]
api_key = "test-api-key"
model = "glm-5.1"
"#;
    let mut tmp = NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{}", toml_content).expect("Failed to write to temp file");

    let config = illustrious_manager::config::load_config_from_path(tmp.path())
        .expect("Failed to load config from temp file");
    assert_eq!(config.backend, "zai");
    let zai = config.zai.as_ref().expect("zai config should be present");
    assert_eq!(zai.api_key, "test-api-key");
    assert_eq!(zai.model, "glm-5.1");
}

#[test]
fn test_backend_defaults_to_vertex() {
    let toml_content = r#"
[vertex]
project = "my-project"
"#;
    let mut tmp = NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{}", toml_content).expect("Failed to write to temp file");

    let config = illustrious_manager::config::load_config_from_path(tmp.path())
        .expect("Failed to load config from temp file");
    assert_eq!(config.backend, "vertex");
}

#[test]
fn test_zai_config_uses_default_model() {
    let toml_content = r#"
backend = "zai"

[vertex]
project = "my-project"

[zai]
api_key = "test-api-key"
"#;
    let mut tmp = NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{}", toml_content).expect("Failed to write to temp file");

    let config = illustrious_manager::config::load_config_from_path(tmp.path())
        .expect("Failed to load config from temp file");
    let zai = config.zai.as_ref().expect("zai config should be present");
    assert_eq!(zai.model, "glm-5.1");
}

#[test]
fn test_cli_model_overrides_zai_model() {
    let toml_content = r#"
backend = "zai"

[vertex]
project = "my-project"

[zai]
api_key = "test-api-key"
model = "glm-5.1"
"#;
    let mut tmp = NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{}", toml_content).expect("Failed to write to temp file");

    let mut config = illustrious_manager::config::load_config_from_path(tmp.path())
        .expect("Failed to load config from temp file");

    illustrious_manager::config::apply_overrides(&mut config, None, None, Some("glm-4"));

    let zai = config.zai.as_ref().expect("zai config should be present");
    assert_eq!(zai.model, "glm-4");
}

#[test]
fn test_cli_vertex_flags_ignored_for_zai_backend() {
    let toml_content = r#"
backend = "zai"

[vertex]
project = "original-project"
region = "us-east5"

[zai]
api_key = "test-api-key"
model = "glm-5.1"
"#;
    let mut tmp = NamedTempFile::new().expect("Failed to create temp file");
    write!(tmp, "{}", toml_content).expect("Failed to write to temp file");

    let mut config = illustrious_manager::config::load_config_from_path(tmp.path())
        .expect("Failed to load config from temp file");

    illustrious_manager::config::apply_overrides(
        &mut config,
        Some("cli-project"),
        Some("europe-west1"),
        None,
    );

    assert_eq!(
        config.vertex.project, "original-project",
        "project flag should not affect vertex config when backend is zai"
    );
}
