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

### Testing

- Frontend REPL ratatui testing should be done with insta snapshots.
- It is expected that Test Driven Development will be the main way that code is implemented in this repo, so most code should have tests that test _behavior_.
- Any bug fix should include regression at least one regression test.

## Project Overview

Illustrious Manager is an experimental TUI agent application in Rust that connects to Claude on Vertex AI with streaming responses. It supports both an interactive REPL (Ratatui) and a single-shot stdout mode.

## Build & Development Commands

```bash
cargo build                          # Build the project
cargo run                            # Run in REPL mode
cargo run -- --single-shot "prompt"  # Run in single-shot mode
cargo run -- --debug                 # Run with file logging (illustrious-manager_<ts>.log)
cargo run -- --config <path>         # Use a custom config file
cargo test                           # Run all tests
cargo test <test_name>               # Run a single test
cargo insta review                   # Review/accept snapshot test changes
cargo clippy                         # Lint
cargo fmt                            # Format
```

## Architecture

Four-layer decoupled design:

1. **Backend Layer** (`src/backend/`) — `LlmBackend` trait abstraction over LLM providers. Three implementations: `vertex` (Vertex AI + Claude via SSE), `zai` (z.ai), and `ollama` (Ollama Cloud). Both `zai` and `ollama` use OpenAI-compatible SSE; `ollama` reuses `ZaiSseParser`. Emits `StreamEvent` (TextDelta | ToolUseStart/Delta/Done | Usage | Done).

2. **Agent Core** (`src/agent.rs`) — Owns conversation history and context files. Wraps backend streams into `AgentEvent` (TokenReceived | ToolUseReceived | ToolResult | ToolConfirmationRequired | ResponseComplete | Error | Usage). Display-agnostic. Drives agentic tool-use loops up to `max_tool_iterations`.

3. **Tools Layer** (`src/tools/`) — `Tool` trait + `ToolRegistry`. Built-in tools: `bash` (allowlist/denylist enforced), `edit_file`, `write_file`. File tools are sandboxed to `sandbox_root` via `SandboxPolicy`. `is_write_tool()` determines whether confirmation is required under `WriteOnly` mode.

4. **Frontend Layer** (`src/frontend/`) — Two frontends consuming the same AgentEvent stream:
   - `stdout.rs`: Single-shot mode, streams tokens to stdout, pipe-friendly
   - `tui`: Ratatui interactive REPL with input/response areas

**Context files** (`src/context_files.rs`) — on startup, CLAUDE.md and AGENTS.md are auto-discovered from `~/.claude/` and the current working directory, then injected into the conversation as initial context.

Key types live in `src/types.rs`. Configuration loading and CLI merge logic is in `src/config.rs`.

## Configuration

Config file at `~/.config/illustrious-manager/config.toml` (auto-created on first run):

```toml
backend = "vertex"                    # "vertex", "zai", or "ollama"

[vertex]
project = ""                          # GCP project ID (required)
region = "us-east5"
model = "claude-sonnet-4-20250514"

[zai]
api_key = ""                          # z.ai API key (required for zai backend)
model = "glm-5.1"

# [ollama]
# api_key = ""                        # Ollama Cloud API key (required for ollama backend)
# model = "glm-5.1"

# [tools]
# confirmation = "WriteOnly"          # Always | WriteOnly | Never
# sandbox_root = "."                  # Directory tools are allowed to read/write
# max_tool_iterations = 25
# bash_allowlist = ["cat", "ls", "grep", "find", "head", "tail", "wc", "tree"]
# bash_denylist = ["rm", "wget", "sudo", "chmod", "chown"]
```

CLI flags (`--project`, `--region`, `--model`) override config file values.

## Authentication

Uses GCP Application Default Credentials (ADC) via `gcp_auth`. Requires `gcloud auth application-default login` or equivalent.

## Design Documents

- Design spec: `docs/superpowers/specs/2026-03-27-phase1-core-cli-design.md`
- Implementation plan: `docs/superpowers/plans/2026-03-27-phase1-core-cli.md`
