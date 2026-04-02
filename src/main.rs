pub mod agent;
pub mod backend;
pub mod config;
pub mod frontend;
pub mod types;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use agent::Agent;
use backend::vertex::VertexBackend;
use types::RequestConfig;

#[derive(Parser)]
#[command(name = "illustrious-manager")]
#[command(about = "A TUI agent application for Claude on Vertex AI")]
struct Cli {
    /// Prompt to send (starts REPL with this message, or use with --single-shot)
    prompt: Option<String>,

    /// GCP project ID
    #[arg(long)]
    project: Option<String>,

    /// Vertex AI region
    #[arg(long)]
    region: Option<String>,

    /// Model identifier
    #[arg(long)]
    model: Option<String>,

    /// Run in single-shot mode (print response to stdout and exit)
    #[arg(long)]
    single_shot: bool,

    /// Custom config file path
    #[arg(long)]
    config: Option<PathBuf>,
}

fn validate_cli(cli: &Cli) -> Result<()> {
    if cli.single_shot && cli.prompt.is_none() {
        anyhow::bail!(
            "--single-shot requires a prompt argument.\n\nUsage: illustrious-manager --single-shot \"your prompt here\""
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    validate_cli(&cli)?;

    // Load and merge config
    let mut app_config = config::load_config(cli.config.as_deref())?;
    config::apply_overrides(
        &mut app_config,
        cli.project.as_deref(),
        cli.region.as_deref(),
        cli.model.as_deref(),
    );
    config::validate(&app_config, cli.config.as_deref())?;

    // Initialize backend
    let vertex_backend = VertexBackend::new(
        app_config.vertex.project.clone(),
        app_config.vertex.region.clone(),
    )
    .await?;

    let request_config = RequestConfig {
        model: app_config.vertex.model.clone(),
        max_tokens: 8192,
    };

    let mut agent = Agent::new(Box::new(vertex_backend), request_config);

    if cli.single_shot {
        // Single-shot mode: send prompt, stream to stdout, exit
        let prompt = cli.prompt.expect("prompt is required in single-shot mode");
        let stream = agent.send(prompt).await?;
        frontend::stdout::run(stream).await?;
    } else {
        // REPL mode
        frontend::tui::run(&mut agent, cli.prompt).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_shot_without_prompt_fails_validation() {
        let cli = Cli::try_parse_from(["illustrious-manager", "--single-shot"]).unwrap();
        let err = validate_cli(&cli).unwrap_err().to_string();
        assert!(
            err.contains("--single-shot"),
            "Error should mention --single-shot"
        );
    }

    #[test]
    fn single_shot_with_prompt_passes_validation() {
        let cli = Cli::try_parse_from(["illustrious-manager", "--single-shot", "hello"]).unwrap();
        assert!(validate_cli(&cli).is_ok());
    }

    #[test]
    fn no_args_passes_validation() {
        let cli = Cli::try_parse_from(["illustrious-manager"]).unwrap();
        assert!(validate_cli(&cli).is_ok());
    }

    #[test]
    fn prompt_without_single_shot_passes_validation() {
        let cli = Cli::try_parse_from(["illustrious-manager", "hello world"]).unwrap();
        assert!(validate_cli(&cli).is_ok());
    }
}
