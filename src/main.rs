#![deny(clippy::print_stderr, clippy::print_stdout, clippy::dbg_macro)]

pub mod agent;
pub mod backend;
pub mod config;
pub mod context_files;
pub mod frontend;
pub mod logging;
pub mod session;
pub mod timestamp;
pub mod tools;
pub mod types;

use anyhow::Result;
use clap::Parser;
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use agent::spawn_agent_with_selection;
use backend::BackendFactory;
use logging::Logger;
use session::Session;
use tools::ToolRegistry;
use tools::agent::AgentTool;
use tools::bash::BashTool;
use tools::edit_file::EditFile;
use tools::image_viewer::ImageViewer;
use tools::sandbox::SandboxPolicy;
use tools::search::SearchTool;
use tools::skill::{SkillTool, discover_skills_from_env};
use tools::task::{CreateTaskTool, DeleteTaskTool, ListTasksTool, UpdateTaskTool};
use tools::write_file::WriteFileTool;

const DEFAULT_MAX_SCHEMA_RETRIES: u32 = 3;

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

    /// Maximum number of reprompt attempts when schema validation fails.
    /// 0 means validate once and fail immediately with no reprompts.
    #[arg(long, default_value_t = DEFAULT_MAX_SCHEMA_RETRIES)]
    max_schema_retries: u32,

    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long)]
    debug: bool,

    #[arg(long)]
    session_id: Option<String>,

    #[arg(long)]
    chat: bool,
}

#[derive(Debug)]
enum Mode {
    Repl {
        initial_prompt: Option<String>,
    },
    SingleShot {
        prompt: String,
        json_schema: Option<frontend::stdout::JsonSchema>,
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

        let json_schema = if let Some(ref schema_str) = cli.json_schema {
            if cli.output_format != frontend::stdout::OutputFormat::Json {
                anyhow::bail!("--json-schema requires --output-format json");
            }
            let schema_value: serde_json::Value = serde_json::from_str(schema_str)
                .map_err(|e| anyhow::anyhow!("--json-schema is not valid JSON: {}", e))?;
            let validator = jsonschema::validator_for(&schema_value)
                .map_err(|e| anyhow::anyhow!("--json-schema failed to compile: {}", e))?;
            Some(frontend::stdout::JsonSchema {
                validator,
                raw: schema_str.clone(),
            })
        } else {
            None
        };

        Ok(Mode::SingleShot {
            prompt,
            json_schema,
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

    let factory = Arc::new(BackendFactory::new(app_config.clone()));
    let app_config_arc = Arc::new(app_config.clone());

    let skills = discover_skills_from_env();

    let session = Session::new(cli.session_id.clone(), app_config.sessions_dir.clone()).await?;
    let session_arc = Arc::new(tokio::sync::Mutex::new(session));

    let chat_mode = crate::types::ChatMode::new(cli.chat);

    let default_selection = factory.for_role("default").await?;

    let mut registry = build_tool_registry(
        Arc::clone(&session_arc),
        &app_config.tools,
        &skills,
        None,
        chat_mode.clone(),
    )?;

    let spawner = build_agent_spawner_and_register(
        &mut registry,
        &factory,
        &app_config_arc,
        &skills,
        chat_mode.clone(),
    )?;

    let agent = Arc::new(
        spawn_agent_with_selection(
            default_selection,
            &app_config.tools,
            &app_config.retry,
            session_arc,
            registry,
        )
        .await?
        .with_compaction_config(&app_config.compaction)
        .with_retry_config(&app_config.retry)
        .with_thinking(app_config.thinking.clone())
        // This ordering is because both `with_skills` and `with_context_files` PREPEND to history.
        // because initial history is set from the input session
        .with_skills(&skills)
        .with_context_files()?
        .with_compaction_spawner(spawner)
        .with_chat_mode(chat_mode),
    );

    if cli.debug {
        logging::init_global(Some(create_log_path()?))?;
    }
    let mut logger: Option<Logger> = if cli.debug {
        Some(Logger::new(None)?)
    } else {
        None
    };

    if let Some(ref mut log) = logger {
        log.log_config(&app_config)?;
        log.flush()?;
    }

    let frontend_result = match mode {
        Mode::SingleShot {
            prompt,
            json_schema,
            max_schema_retries,
        } => {
            // Inject schema instruction before logging so the log reflects what is actually sent.
            let effective_prompt = if let Some(ref schema) = json_schema {
                format!(
                    "{}\n\nYou MUST output ONLY valid JSON conforming to this schema (no markdown, no explanation):\n{}",
                    prompt, schema.raw
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
                max_schema_retries,
                logger.as_mut(),
            )
            .await
        }
        Mode::Repl { initial_prompt } => {
            frontend::tui::run(
                agent.clone(),
                initial_prompt,
                logger,
                &app_config,
                factory.clone(),
            )
            .await
        }
    };

    if let Err(e) = agent.checkpoint_session().await {
        logging::log_error(&format!("checkpoint_session failed: {e}"));
    }

    if let Err(e) = agent.cleanup_empty_session().await
        && cli.debug
    {
        logging::log_error(&format!("cleanup_empty_session failed: {e}"));
    }

    // Flush the debug log before printing the session-ID epilogue so no buffered
    // entries are lost (static OnceLock is never dropped by Rust).
    logging::flush();
    // Intentional stderr write: session ID epilogue is printed after TUI tears down, safe to write directly.
    writeln!(io::stderr(), "Session ID: {}", agent.session_id().await)?;
    frontend_result
}

/// Registers `/tmp` (on Unix) and the platform temp directory as extra
/// sandbox roots so file tools (`edit_file`, `write_file`) can always
/// read/write temp space regardless of `sandbox_root`. Both are needed on
/// macOS, where `std::env::temp_dir()` follows `$TMPDIR` to a per-user
/// `/var/folders/...` directory rather than `/tmp`. `bash` already has
/// unrestricted `/tmp` access, so this grants no new privilege beyond what
/// the agent already has.
fn extend_sandbox_with_tmp(policy: SandboxPolicy) -> SandboxPolicy {
    let policy = policy.with_extra_root(&std::env::temp_dir());
    #[cfg(unix)]
    let policy = policy.with_extra_root(Path::new("/tmp"));
    policy
}

/// Build a `ToolRegistry` with the standard tool set.
///
/// `agent_tool` is `Some` for sub-agent registries (enabling recursive spawning)
/// and `None` for the parent registry, where `build_agent_spawner_and_register`
/// adds `AgentTool` after constructing the spawner.
fn build_tool_registry(
    session: Arc<tokio::sync::Mutex<Session>>,
    tools_config: &config::ToolsConfig,
    skills: &HashMap<String, PathBuf>,
    agent_tool: Option<AgentTool>,
    chat_mode: crate::types::ChatMode,
) -> Result<ToolRegistry> {
    let sandbox_policy = SandboxPolicy::new(Path::new(&tools_config.sandbox_root));
    let sandbox_policy = extend_sandbox_with_tmp(sandbox_policy);
    let mut reg = ToolRegistry::new();
    reg.register(Box::new(BashTool::new(
        tools_config.bash_allowlist.clone(),
        tools_config.bash_denylist.clone(),
        PathBuf::from(&tools_config.sandbox_root),
        tools_config.confirmation.clone(),
        Box::new(|_| true),
        chat_mode.clone(),
    )))?;
    reg.register(Box::new(EditFile::new(sandbox_policy.clone())))?;
    reg.register(Box::new(WriteFileTool::new(sandbox_policy.clone())))?;
    reg.register(Box::new(SearchTool::new(PathBuf::from(
        &tools_config.sandbox_root,
    ))))?;
    reg.register(Box::new(SkillTool::new(skills)))?;
    reg.register(Box::new(CreateTaskTool::new(Arc::clone(&session))))?;
    reg.register(Box::new(ListTasksTool::new(Arc::clone(&session))))?;
    reg.register(Box::new(UpdateTaskTool::new(Arc::clone(&session))))?;
    reg.register(Box::new(DeleteTaskTool::new(session)))?;
    reg.register(Box::new(ImageViewer::new(sandbox_policy)))?;
    if let Some(tool) = agent_tool {
        reg.register(Box::new(tool))?;
    }
    Ok(reg)
}

/// Construct the `AgentSpawner` and register `AgentTool` into `registry`.
///
/// Uses a `OnceLock` to break the circular reference between the spawner and
/// the `registry_builder` closure that references it, enabling sub-agents to
/// spawn further sub-agents.
fn build_agent_spawner_and_register(
    registry: &mut ToolRegistry,
    factory: &Arc<BackendFactory>,
    app_config: &Arc<config::AppConfig>,
    skills: &HashMap<String, PathBuf>,
    chat_mode: crate::types::ChatMode,
) -> Result<Arc<agent::AgentSpawner>> {
    let factory_clone = Arc::clone(factory);
    let app_config_clone = Arc::clone(app_config);
    let skills_clone = skills.clone();
    let tools_config = app_config.tools.clone();
    let available_roles: Vec<String> = app_config.models.keys().cloned().collect();

    let spawner_cell: Arc<std::sync::OnceLock<Arc<agent::AgentSpawner>>> =
        Arc::new(std::sync::OnceLock::new());
    let spawner_cell_clone = Arc::clone(&spawner_cell);
    let roles_for_closure = available_roles.clone();

    let spawner = Arc::new(agent::AgentSpawner {
        factory: factory_clone,
        app_config: app_config_clone,
        registry_builder: Box::new({
            let chat_mode_clone = chat_mode.clone();
            move |sub_session| {
                let agent_tool = spawner_cell_clone
                    .get()
                    .map(|s| AgentTool::new(Arc::clone(s), roles_for_closure.clone()));
                build_tool_registry(
                    sub_session,
                    &tools_config,
                    &skills_clone,
                    agent_tool,
                    chat_mode_clone.clone(),
                )
            }
        }),
        parent_confirmation: app_config.tools.confirmation.clone(),
        skills: skills.clone(),
        chat_mode,
    });

    spawner_cell
        .set(Arc::clone(&spawner))
        .ok()
        .expect("spawner_cell set exactly once at startup");

    registry.register(Box::new(AgentTool::new(
        Arc::clone(&spawner),
        available_roles,
    )))?;
    Ok(spawner)
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

    // Regression: on macOS, std::env::temp_dir() follows $TMPDIR to
    // /var/folders/.../T/, so registering only temp_dir() left /tmp outside
    // the sandbox and writes to /tmp/... were rejected.
    #[cfg(unix)]
    #[test]
    fn extend_sandbox_with_tmp_allows_write_paths_under_slash_tmp() {
        let sandbox_dir = tempfile::TempDir::new().expect("Failed to create sandbox dir");
        let policy = extend_sandbox_with_tmp(SandboxPolicy::new(sandbox_dir.path()));

        let target = Path::new("/tmp/illustrious-manager-tmp-root-regression-test.txt");
        let result = policy.validate_write_path(target);
        assert!(
            result.is_ok(),
            "Write path under /tmp must validate even when $TMPDIR points elsewhere: {:?}",
            result
        );
    }

    #[cfg(unix)]
    #[test]
    fn extend_sandbox_with_tmp_allows_platform_temp_dir() {
        let sandbox_dir = tempfile::TempDir::new().expect("Failed to create sandbox dir");
        let policy = extend_sandbox_with_tmp(SandboxPolicy::new(sandbox_dir.path()));

        let target = std::env::temp_dir().join("illustrious-manager-tempdir-regression-test.txt");
        let result = policy.validate_write_path(&target);
        assert!(
            result.is_ok(),
            "Write path under the platform temp dir must validate: {:?}",
            result
        );
    }

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
                ..
            } => {
                assert_eq!(prompt, "hello");
                assert!(json_schema.is_some());
                assert_eq!(
                    json_schema.expect("some").raw.as_str(),
                    r#"{"type":"object"}"#
                );
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

    #[test]
    fn chat_flag_is_parsed_when_present() {
        let cli = Cli::try_parse_from(["illustrious-manager", "--chat"]).unwrap();
        assert!(cli.chat);
    }

    #[test]
    fn chat_flag_defaults_to_false() {
        let cli = Cli::try_parse_from(["illustrious-manager"]).unwrap();
        assert!(!cli.chat);
    }
}
