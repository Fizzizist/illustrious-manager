pub mod agent;
pub mod backend;
pub mod config;
pub mod context_files;
pub mod frontend;
pub mod logging;
pub mod session;
pub mod tools;
pub mod types;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use agent::Agent;
use logging::Logger;
use session::Session;
use tools::ToolRegistry;
use tools::bash::BashTool;
use tools::edit_file::EditFile;
use tools::sandbox::SandboxPolicy;
use tools::skill::{SkillTool, discover_skills_from_env};
use tools::write_file::WriteFileTool;
use types::RequestConfig;

const DEFAULT_MAX_SCHEMA_RETRIES: u32 = 3;

const DEFAULT_MAX_TOKENS: u32 = 8192;

#[derive(Parser)]
#[command(name = "illustrious-manager")]
#[command(about = "A TUI agent application for Claude on Vertex AI")]
struct Cli {
    prompt: Option<String>,

    #[arg(long)]
    project: Option<String>,

    #[arg(long)]
    region: Option<String>,

    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    single_shot: bool,

    #[arg(long, default_value = "text")]
    output_format: frontend::stdout::OutputFormat,

    #[arg(long)]
    json_schema: Option<String>,

    #[arg(long, default_value_t = DEFAULT_MAX_SCHEMA_RETRIES)]
    max_schema_retries: u32,

    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long)]
    debug: bool,

    #[arg(long)]
    session_id: Option<String>,
}

#[derive(Debug)]
enum Mode {
    Repl {
        initial_prompt: Option<String>,
    },
    SingleShot {
        prompt: String,
        json_schema: Option<jsonschema::Validator>,
        json_schema_raw: Option<String>,
        max_schema_retries: u32,
    },
}

fn determine_mode(cli: &Cli) -> Result<Mode> {
    if cli.single_shot {
        let prompt = cli.prompt.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "--single-shot requires a prompt argument.\n\nUsage: illustrious-manager --single-shot \"your prompt here\""
            )
        })?;

        let (json_schema, json_schema_raw) = if let Some(ref schema_str) = cli.json_schema {
            if cli.output_format != frontend::stdout::OutputFormat::Json {
                anyhow::bail!("--json-schema requires --output-format json");
            }
            let schema_value: serde_json::Value = serde_json::from_str(schema_str)
                .map_err(|e| anyhow::anyhow!("--json-schema is not valid JSON: {}", e))?;
            let validator = jsonschema::validator_for(&schema_value)
                .map_err(|e| anyhow::anyhow!("--json-schema failed to compile: {}", e))?;
            (Some(validator), Some(schema_str.clone()))
        } else {
            (None, None)
        };

        Ok(Mode::SingleShot {
            prompt,
            json_schema,
            json_schema_raw,
            max_schema_retries: cli.max_schema_retries,
        })
    } else {
        if cli.output_format != frontend::stdout::OutputFormat::Text {
            anyhow::bail!("--output-format requires --single-shot");
        }
        if cli.json_schema.is_some() {
            anyhow::bail!("--json-schema requires --single-shot");
        }
        Ok(Mode::Repl {
            initial_prompt: cli.prompt.clone(),
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let cli = Cli::parse();
    let mode = determine_mode(&cli)?;

    let mut app_config = config::load_config(cli.config.as_deref())?;
    config::apply_overrides(
        &mut app_config,
        cli.project.as_deref(),
        cli.region.as_deref(),
        cli.model.as_deref(),
    );
    config::validate(&app_config, cli.config.as_deref())?;

    let selection = backend::from_config(&app_config).await?;

    let mut registry = ToolRegistry::new();
    registry.register(Box::new(BashTool::new(
        app_config.tools.bash_allowlist.clone(),
        app_config.tools.bash_denylist.clone(),
        std::path::PathBuf::from(&app_config.tools.sandbox_root),
        app_config.tools.confirmation.clone(),
        Box::new(|_| true),
    )))?;

    let sandbox_policy = SandboxPolicy::new(std::path::Path::new(&app_config.tools.sandbox_root));
    registry.register(Box::new(EditFile::new(sandbox_policy.clone())))?;
    registry.register(Box::new(WriteFileTool::new(sandbox_policy)))?;

    let skills = discover_skills_from_env();
    registry.register(Box::new(SkillTool::new(&skills)))?;

    let request_config = RequestConfig {
        model: selection.model,
        max_tokens: DEFAULT_MAX_TOKENS,
        tools: registry.definitions(),
    };

    let session = Session::new(cli.session_id.clone(), app_config.sessions_dir.clone()).await?;

    let agent = Arc::new(
        Agent::new(selection.backend, request_config, session)
            .await
            .with_tools(registry)
            .with_tool_config(&app_config.tools)
            // This ordering is because both `with_skills` and `with_context_files` PREPEND to history.
            // because initial history is set from the input session
            .with_skills(&skills)
            .with_context_files()?,
    );

    let mut logger = if cli.debug {
        let log_path = create_log_path()?;
        Some(Logger::new(Some(log_path))?)
    } else {
        None
    };

    if let Some(ref mut log) = logger {
        log.log_config(&app_config)?;
        log.flush()?;
    }

    match mode {
        Mode::SingleShot {
            prompt,
            json_schema,
            json_schema_raw,
            max_schema_retries,
        } => {
            // Inject schema instruction before logging so the log reflects what is actually sent.
            let effective_prompt = if let Some(ref schema_raw) = json_schema_raw {
                format!(
                    "{}\n\nYou MUST output ONLY valid JSON conforming to this schema (no markdown, no explanation):\n{}",
                    prompt, schema_raw
                )
            } else {
                prompt
            };
            if let Some(ref mut log) = logger {
                log.log_user_input(&effective_prompt)?;
            }
            frontend::stdout::run(
                agent.clone(),
                effective_prompt,
                cli.output_format,
                json_schema,
                json_schema_raw,
                max_schema_retries,
                logger.as_mut(),
            )
            .await?;
        }
        Mode::Repl { initial_prompt } => {
            frontend::tui::run(agent.clone(), initial_prompt, logger, &app_config).await?;
        }
    }

    eprintln!("Session ID: {}", agent.session_id().await);
    Ok(())
}

fn create_log_path() -> Result<PathBuf> {
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("Time went backwards");
    let timestamp = duration.as_millis();
    Ok(PathBuf::from(format!(
        "illustrious-manager_{}.log",
        timestamp
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_shot_without_prompt_errors() {
        let cli = Cli::try_parse_from(["illustrious-manager", "--single-shot"]).unwrap();
        let err = determine_mode(&cli).unwrap_err().to_string();
        assert!(
            err.contains("--single-shot"),
            "Error should mention --single-shot"
        );
    }

    #[test]
    fn single_shot_with_prompt_gives_single_shot_mode() {
        let cli = Cli::try_parse_from(["illustrious-manager", "--single-shot", "hello"]).unwrap();
        match determine_mode(&cli).unwrap() {
            Mode::SingleShot { prompt, .. } => assert_eq!(prompt, "hello"),
            Mode::Repl { .. } => panic!("Expected SingleShot mode"),
        }
    }

    #[test]
    fn no_args_gives_repl_mode_with_no_initial_prompt() {
        let cli = Cli::try_parse_from(["illustrious-manager"]).unwrap();
        match determine_mode(&cli).unwrap() {
            Mode::Repl { initial_prompt } => assert!(initial_prompt.is_none()),
            Mode::SingleShot { .. } => panic!("Expected Repl mode"),
        }
    }

    #[test]
    fn prompt_without_single_shot_gives_repl_mode_with_initial_prompt() {
        let cli = Cli::try_parse_from(["illustrious-manager", "hello world"]).unwrap();
        match determine_mode(&cli).unwrap() {
            Mode::Repl { initial_prompt } => {
                assert_eq!(initial_prompt.as_deref(), Some("hello world"))
            }
            Mode::SingleShot { .. } => panic!("Expected Repl mode"),
        }
    }

    #[test]
    fn debug_flag_is_parsed_when_present() {
        let cli = Cli::try_parse_from(["illustrious-manager", "--debug"]).unwrap();
        assert!(cli.debug);
    }

    #[test]
    fn debug_flag_is_false_when_not_present() {
        let cli = Cli::try_parse_from(["illustrious-manager"]).unwrap();
        assert!(!cli.debug);
    }

    #[test]
    fn session_id_flag_is_parsed() {
        let cli = Cli::try_parse_from([
            "illustrious-manager",
            "--session-id",
            "01923456-7890-7abc-def0-123456789abc",
        ])
        .unwrap();
        assert_eq!(
            cli.session_id,
            Some("01923456-7890-7abc-def0-123456789abc".to_string())
        );
    }

    #[test]
    fn session_id_flag_defaults_to_none() {
        let cli = Cli::try_parse_from(["illustrious-manager"]).unwrap();
        assert!(cli.session_id.is_none());
    }

    #[test]
    fn output_format_defaults_to_text() {
        let cli = Cli::try_parse_from(["illustrious-manager"]).unwrap();
        assert_eq!(cli.output_format, frontend::stdout::OutputFormat::Text);
    }

    #[test]
    fn output_format_json_is_parsed() {
        let cli = Cli::try_parse_from([
            "illustrious-manager",
            "--single-shot",
            "--output-format",
            "json",
            "hello",
        ])
        .unwrap();
        assert_eq!(cli.output_format, frontend::stdout::OutputFormat::Json);
    }

    #[test]
    fn output_format_json_without_single_shot_errors() {
        let cli = Cli::try_parse_from(["illustrious-manager", "--output-format", "json", "hello"])
            .unwrap();
        let err = determine_mode(&cli).unwrap_err().to_string();
        assert!(
            err.contains("--output-format"),
            "Error should mention --output-format"
        );
    }

    #[test]
    fn json_schema_is_none_when_not_provided() {
        let cli = Cli::try_parse_from([
            "illustrious-manager",
            "--single-shot",
            "--output-format",
            "json",
            "hello",
        ])
        .unwrap();
        assert!(cli.json_schema.is_none());
    }

    #[test]
    fn max_schema_retries_defaults_to_three() {
        let cli = Cli::try_parse_from(["illustrious-manager"]).unwrap();
        assert_eq!(cli.max_schema_retries, DEFAULT_MAX_SCHEMA_RETRIES);
    }

    #[test]
    fn max_schema_retries_is_overridden_when_provided() {
        let cli = Cli::try_parse_from([
            "illustrious-manager",
            "--single-shot",
            "--output-format",
            "json",
            "--max-schema-retries",
            "5",
            "hello",
        ])
        .unwrap();
        assert_eq!(cli.max_schema_retries, 5);
    }

    #[test]
    fn json_schema_without_single_shot_errors() {
        let cli = Cli::try_parse_from([
            "illustrious-manager",
            "--json-schema",
            r#"{"type":"object"}"#,
            "hello",
        ])
        .unwrap();
        let err = determine_mode(&cli).unwrap_err().to_string();
        assert!(
            err.contains("--json-schema"),
            "error should mention --json-schema"
        );
    }

    #[test]
    fn json_schema_without_output_format_json_errors() {
        let cli = Cli::try_parse_from([
            "illustrious-manager",
            "--single-shot",
            "--json-schema",
            r#"{"type":"object"}"#,
            "hello",
        ])
        .unwrap();
        let err = determine_mode(&cli).unwrap_err().to_string();
        assert!(
            err.contains("--json-schema"),
            "error should mention --json-schema"
        );
    }

    #[test]
    fn json_schema_with_single_shot_and_json_format_succeeds() {
        let cli = Cli::try_parse_from([
            "illustrious-manager",
            "--single-shot",
            "--output-format",
            "json",
            "--json-schema",
            r#"{"type":"object"}"#,
            "hello",
        ])
        .unwrap();
        let mode = determine_mode(&cli).unwrap();
        match mode {
            Mode::SingleShot {
                prompt,
                json_schema,
                json_schema_raw,
                ..
            } => {
                assert_eq!(prompt, "hello");
                assert!(json_schema.is_some());
                assert_eq!(json_schema_raw.as_deref(), Some(r#"{"type":"object"}"#));
            }
            Mode::Repl { .. } => panic!("Expected SingleShot mode"),
        }
    }

    #[test]
    fn json_schema_non_json_value_errors() {
        let cli = Cli::try_parse_from([
            "illustrious-manager",
            "--single-shot",
            "--output-format",
            "json",
            "--json-schema",
            "not valid json",
            "hello",
        ])
        .unwrap();
        let err = determine_mode(&cli).unwrap_err().to_string();
        assert!(
            err.contains("not valid JSON"),
            "error should describe invalid JSON"
        );
    }

    #[test]
    fn json_schema_invalid_schema_errors() {
        // Valid JSON but not a valid JSON Schema (unknown keyword that causes compilation failure).
        // Use a type value that jsonschema rejects.
        let cli = Cli::try_parse_from([
            "illustrious-manager",
            "--single-shot",
            "--output-format",
            "json",
            "--json-schema",
            r#"{"type": "notavalidtype"}"#,
            "hello",
        ])
        .unwrap();
        let err = determine_mode(&cli).unwrap_err().to_string();
        assert!(
            err.contains("compile"),
            "error should mention compilation failure"
        );
    }
}
