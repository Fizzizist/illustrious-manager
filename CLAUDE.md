# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Engineering Policies

These are _strict_ policies that must be followed by all engineers and developers in this project. MRs will be rejected if these policies are violated.

### Dependency Management

- All dependencies _must_ be added, removed and updated using `cargo` on the command line.
- Under no circumstances should the Cargo.toml be manually edited with regard to dependencies.

### Coding

- The use of `.unwrap()` is forbidden under _all_ circumstances. The program should _never_ panic.
- In a case where something needs to be unwrapped and it is _logically impossible_ for a panic to occur, the use of `.expect()` with an informative message is permitted.
- Always run `cargo fmt` before committing code.
- Always run `cargo clippy` before committing code.
- Keep code comments to a minimum. Only comment in cases where something is unable to be gleaned from the code itself.
- `eprintln!` and `println!` are forbidden in library and agent code (`src/backend/`, `src/tools/`, `src/agent/`). Return warnings as part of function return values and emit them via `tracing::warn!()` at the call site, or surface them as `AgentEvent::Warn` where an event channel is available.

### Testing

- Frontend REPL ratatui testing should be done with insta snapshots.
- It is expected that Test Driven Development will be the main way that code is implemented in this repo, so most code should have tests that test _behavior_.
- Any bug fix should include regression at least one regression test.

## Project Overview

Illustrious Manager is an experimental TUI agent application in Rust that connects to Claude on Vertex AI with streaming responses. It supports both an interactive REPL (Ratatui) and a single-shot stdout mode. Conversations are persisted to SQLite databases (one per session) and can be resumed.

## Build & Development Commands

```bash
cargo build                          # Build the project
cargo run                            # Run in REPL mode
cargo run -- --single-shot "prompt"  # Run in single-shot mode
cargo run -- --debug                 # Run with file logging (illustrious-manager_<ts>.log)
cargo run -- --config <path>         # Use a custom config file
cargo run -- --session-id <uuid>     # Resume a previous session by UUIDv7
cargo test                           # Run all tests
cargo test <test_name>               # Run a single test
cargo insta review                   # Review/accept snapshot test changes
cargo clippy                         # Lint
cargo fmt                            # Format
```

## Architecture

Four-layer decoupled design:

1. **Backend Layer** (`src/backend/`) — `LlmBackend` trait abstraction over LLM providers. Four implementations: `vertex` (Vertex AI + Claude via SSE, with `sse.rs` for SSE stream parsing), `zai` (z.ai — thin shim over `openai_compat`), `ollama` (Ollama Cloud/self-hosted via NDJSON, with `ndjson.rs` for NDJSON stream parsing), and `openai_compat` (generic OpenAI Chat Completions client — SSE, message serialisation, tool calls). `openai_compat` exposes `ReasoningStyle` (`None` | `ZaiEnableThinking` | `QwenChatTemplate` | `Default`) to control how extended-thinking is expressed per-provider. The `zai` backend is a back-compat shim that constructs an `OpenAiCompatBackend` with `ReasoningStyle::ZaiEnableThinking` and the z.ai base URL. The constructor strips trailing `/` from `base_url` and rejects URLs that already include `/chat/completions`. When `api_key` is `None` or empty, no `Authorization` header is sent. Emits `StreamEvent` (TextDelta | ToolUseStart/Delta/Done | Usage | Done).

2. **Agent Core** (`src/agent.rs`) — Owns conversation history, context files, and skills. Wraps backend streams into `AgentEvent` (TokenReceived | ToolUseReceived | ToolResult | ToolConfirmationRequired | ResponseComplete | Error | Usage). Display-agnostic. Drives agentic tool-use loops up to `max_tool_iterations`. Tool calls within a single assistant turn run concurrently via `join_all`; confirmations are gathered sequentially first, then approved calls execute in parallel. Tool results exceeding `max_tool_result_bytes` are truncated with head+tail preserved and a sentinel message before being added to history; the untruncated content is still sent to the TUI via `AgentEvent::ToolResult`. Supports session switching (`load_session`) while preserving non-persisted context prefix (context files, skill definitions).

3. **Tools Layer** (`src/tools/`) — `Tool` trait + `ToolRegistry`. Built-in tools: `bash` (allowlist/denylist enforced), `edit_file`, `write_file`, `skill` (loads skill prompts by name), `search` (semantic code search), `agent` (spawns sub-agents). File tools are sandboxed to `sandbox_root` via `SandboxPolicy` (`sandbox.rs`). `is_write_tool()` determines whether confirmation is required under `WriteOnly` mode.

The `agent` tool spawns an independent sub-agent with its own session, model, and tool set. The parent configures the backend role, confirmation mode (clamped so the sub-agent can never be more permissive than the parent), and an optional tool allowlist. Sub-agent token usage propagates to the TUI status line via `AgentEvent::SubAgentUsage`. Permission clamping is enforced by `clamp_confirmation` in `src/agent.rs` — strictness order: `Always` > `WriteOnly` > `Never`.

4. **Frontend Layer** (`src/frontend/`) — Two frontends consuming the same AgentEvent stream:
   - `stdout.rs`: Single-shot mode, streams tokens to stdout, pipe-friendly
   - `tui/`: Ratatui interactive REPL with vim-style input (`input_area.rs`), scrollable conversation display (`conversation_area.rs`), syntax-highlighted diffs for file tools (`diff.rs` using `syntect` + `similar`), and a session picker overlay (`session_picker.rs`)

**Semantic Search** (`src/tools/search.rs`) — Thin wrapper around the `search-semantically` crate. Exposes semantic code search as a tool with support for natural language queries, identifier names, and file path patterns. The `search-semantically` crate provides tree-sitter AST chunking, ONNX Runtime embeddings (`all-MiniLM-L6-v2`), 6-signal POEM ranking (BM25/FTS5, cosine similarity, path match, symbol match, import graph, git recency), and incremental indexing via SQLite at `<project_root>/.search-index/search.db`.

**Session persistence** (`src/session.rs`) — Each session is a SQLite database (via `turso`) identified by a UUIDv7. Conversations are persisted per-message. Sessions can be listed, resumed, and deleted. The session picker in the TUI allows browsing and switching sessions.

**Context files** (`src/context_files.rs`) — On startup, CLAUDE.md and AGENTS.md are auto-discovered from `~/.claude/`, `pwd/.claude/`, the current working directory, and the home directory, then injected into the conversation as initial context (not persisted to the session DB).

**Skills** (`src/tools/skill.rs`) — Skill directories are discovered from `~/.claude/skills/` and `pwd/.claude/skills/` (pwd overrides home). Each skill is a directory containing a `SKILL.md` file with optional YAML frontmatter (`description` field). Skills are registered as a tool and their names/descriptions are prepended to conversation history.

Key types live in `src/types.rs`. Configuration loading and CLI merge logic is in `src/config.rs`. Debug file logging is in `src/logging.rs`.

## Configuration

Config file at `~/.config/illustrious-manager/config.toml` (auto-created on first run):

```toml
backend = "vertex"                    # "vertex", "zai", "ollama", or "openai_compat"
# sessions_dir = "/path/to/sessions" # defaults to ~/.config/illustrious-manager/sessions

[vertex]
project = ""                          # GCP project ID (required)
region = "us-east5"
model = "claude-sonnet-4-20250514"

[zai]
api_key = ""                          # z.ai API key (required for zai backend)
model = "glm-5.1"

[ollama]
api_key = ""                          # Ollama API key (required for ollama backend)
model = "gpt-oss:120b"
# base_url = "https://ollama.com/api/chat"  # change for self-hosted Ollama

# [openai_compat]
# base_url = "https://vllm.k8s.dc.rxrx.io/v1"  # required, without /chat/completions
# api_key = ""                        # omit or leave empty for unauthenticated endpoints
# model = "Qwen/Qwen3-32B-FP8"
# max_tokens = 16384                  # optional override
# reasoning = "qwen_chat_template"    # none | zai_enable_thinking | qwen_chat_template | default

# [tools]
# confirmation = "WriteOnly"          # Always | WriteOnly | Never
# sandbox_root = "."                  # Directory tools are allowed to read/write
# max_tool_iterations = 25
# Maximum byte cap for tool results. Results exceeding this are truncated
# with head+tail preserved and a sentinel message. Set to 0 for unlimited.
# max_tool_result_bytes = 65536
# bash_allowlist = ["cat", "ls", "grep", "find", "head", "tail", "wc", "tree"]
# bash_denylist = ["rm", "wget", "sudo", "chmod", "chown"]
```

CLI flags (`--project`, `--region`, `--model`, `--session-id`) override config file values.

## Authentication

Uses GCP Application Default Credentials (ADC) via `gcp_auth`. Requires `gcloud auth application-default login` or equivalent.
