// Integration tests for context file loading

use anyhow::Result;
use async_trait::async_trait;
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

use illustrious_manager::agent::Agent;
use illustrious_manager::backend::LlmBackend;
use illustrious_manager::context_files;
use illustrious_manager::session::Session;
use illustrious_manager::types::*;

async fn test_session() -> Session {
    let dir = tempfile::TempDir::new().expect("temp dir");
    Session::new(None, dir.keep()).await.expect("test session")
}

struct NullBackend;

#[async_trait]
impl LlmBackend for NullBackend {
    async fn send_message(
        &self,
        _messages: &[Message],
        _config: &RequestConfig,
    ) -> Result<BoxStream<Result<StreamEvent>>> {
        let stream = futures::stream::iter(vec![Ok(StreamEvent::Done)]);
        Ok(Box::pin(stream))
    }
}

fn get_text_content(content_block: &ContentBlock) -> &str {
    match content_block {
        ContentBlock::Text(t) => t,
        _ => panic!("expected Text content block"),
    }
}

#[tokio::test]
async fn context_files_from_pwd_are_loaded_into_agent() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let home_dir = TempDir::new().expect("create home dir");

    // Create CLAUDE.md in pwd
    let claude_md = temp_dir.path().join("CLAUDE.md");
    File::create(&claude_md)
        .unwrap()
        .write_all(b"# Project Context\nThis is important.")
        .unwrap();

    // Discover files
    let files = context_files::discover_context_files(temp_dir.path(), home_dir.path())
        .expect("discover should succeed");

    // Load into agent
    let backend = Box::new(NullBackend);
    let config = RequestConfig {
        model: "test".to_string(),
        max_tokens: 100,
        tools: vec![],
    };
    let agent = Agent::new(backend, config, test_session().await).await;
    agent.load_context_files(files);

    let history = agent.history();
    assert_eq!(history.len(), 1);
    let text = get_text_content(&history[0].content[0]);
    assert!(text.contains("Project Context"));
    assert!(text.contains("This is important"));
}

#[tokio::test]
async fn context_files_from_home_are_loaded_into_agent() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let home_dir = TempDir::new().expect("create home dir");

    // Create AGENTS.md in home
    let agents_md = home_dir.path().join("AGENTS.md");
    File::create(&agents_md)
        .unwrap()
        .write_all(b"# Agent Instructions\nBe careful.")
        .unwrap();

    // Discover files
    let files = context_files::discover_context_files(temp_dir.path(), home_dir.path())
        .expect("discover should succeed");

    // Load into agent
    let backend = Box::new(NullBackend);
    let config = RequestConfig {
        model: "test".to_string(),
        max_tokens: 100,
        tools: vec![],
    };
    let agent = Agent::new(backend, config, test_session().await).await;
    agent.load_context_files(files);

    let history = agent.history();
    assert_eq!(history.len(), 1);
    let text = get_text_content(&history[0].content[0]);
    assert!(text.contains("Agent Instructions"));
    assert!(text.contains("Be careful"));
}

#[tokio::test]
async fn multiple_context_files_from_both_locations_are_all_loaded() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let home_dir = TempDir::new().expect("create home dir");

    // Create files in pwd
    let pwd_claude = temp_dir.path().join("CLAUDE.md");
    File::create(&pwd_claude)
        .unwrap()
        .write_all(b"PWD context")
        .unwrap();

    // Create files in home
    let home_agents = home_dir.path().join("AGENTS.md");
    File::create(&home_agents)
        .unwrap()
        .write_all(b"Home context")
        .unwrap();

    // Discover files
    let files = context_files::discover_context_files(temp_dir.path(), home_dir.path())
        .expect("discover should succeed");

    assert_eq!(files.len(), 2);

    // Load into agent
    let backend = Box::new(NullBackend);
    let config = RequestConfig {
        model: "test".to_string(),
        max_tokens: 100,
        tools: vec![],
    };
    let agent = Agent::new(backend, config, test_session().await).await;
    agent.load_context_files(files);

    let history = agent.history();
    assert_eq!(history.len(), 1);
    let text = get_text_content(&history[0].content[0]);
    assert!(text.contains("PWD context"));
    assert!(text.contains("Home context"));
}
