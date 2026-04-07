pub mod agent;
pub mod backend;
pub mod config;
pub mod context_files;
pub mod frontend;
pub mod logging;
pub mod tools;
pub mod types;

use anyhow::Result;
use clap::Parser;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use agent::Agent;
use context_files::discover_context_files;
use futures::channel::mpsc;
use logging::Logger;
use tools::ToolRegistry;
use tools::bash::BashTool;
use tools::edit_file::EditFile;
use tools::sandbox::SandboxPolicy;
use tools::write_file::WriteFileTool;
use types::{ConfirmationResponse, RequestConfig};

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

    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long)]
    debug: bool,
}

#[derive(Debug)]
enum Mode {
    Repl { initial_prompt: Option<String> },
    SingleShot { prompt: String },
}

fn determine_mode(cli: &Cli) -> Result<Mode> {
    if cli.single_shot {
        let prompt = cli.prompt.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "--single-shot requires a prompt argument.\n\nUsage: illustrious-manager --single-shot \"your prompt here\""
            )
        })?;
        Ok(Mode::SingleShot { prompt })
    } else {
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

    let request_config = RequestConfig {
        model: selection.model,
        max_tokens: DEFAULT_MAX_TOKENS,
        tools: registry.definitions(),
    };

    let agent = Arc::new(
        Agent::new(selection.backend, request_config)
            .with_tools(registry)
            .with_tool_config(&app_config.tools),
    );

    // Load context files from standard locations
    let pwd = env::current_dir()?;
    let home = env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/"));
    let context_files = discover_context_files(&pwd, &home)?;
    agent.load_context_files(context_files);

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
        Mode::SingleShot { prompt } => {
            if let Some(ref mut log) = logger {
                log.log_user_input(&prompt)?;
            }
            let (confirm_tx, confirm_rx) = mpsc::unbounded::<ConfirmationResponse>();
            let stream = agent.send(prompt, Some(confirm_rx)).await?;
            frontend::stdout::run(stream, confirm_tx, logger.as_mut()).await?;
        }
        Mode::Repl { initial_prompt } => {
            frontend::tui::run(agent.clone(), initial_prompt, logger).await?;
        }
    }

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
            Mode::SingleShot { prompt } => assert_eq!(prompt, "hello"),
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
        assert_eq!(cli.debug, true);
    }

    #[test]
    fn debug_flag_is_false_when_not_present() {
        let cli = Cli::try_parse_from(["illustrious-manager"]).unwrap();
        assert_eq!(cli.debug, false);
    }
}
