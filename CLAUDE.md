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
- The use of `pub(crate)` is forbidden. It is a leaky boundary that exposes implementation details to the rest of the crate without forcing a real API decision. If something needs to cross a module boundary, either make it fully `pub` with a thought-out interface, or restructure so the caller doesn't need access at all.
- Always run `cargo fmt` before committing code.
- Always run `cargo clippy` before committing code.
- Keep code comments to a minimum. Only comment in cases where something is unable to be gleaned from the code itself.
- The use of `eprintln!`, `println!`, `eprint!`, `print!`, and `dbg!` is forbidden in `src/`. All diagnostic output must go through the `logging` module. Direct stdout/stderr writes corrupt the TUI. Genuinely interactive prompts or end-of-process user-facing notices (e.g. single-shot TTY confirmation, session-id epilogue) may use `write!(io::stderr(), ...)` and must include a comment justifying why.

### Testing

- Frontend REPL ratatui testing should be done with insta snapshots.
- It is not necessary to unit test every line of code. More important is testing _behavior_. Do not clutter up the test suite with a bunch of test theater.
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

1. **Backend Layer** (`src/backend/`) — `LlmBackend` trait abstraction over LLM providers. Six implementations: `vertex` (Vertex AI + Claude — partial shim that resolves GCP auth and constructs the Vertex endpoint, then delegates to `anthropic_compat`), `anthropic_compat` (generic Anthropic Messages API protocol client — SSE parsing, request body building, content block filtering; owns `AnthropicCompatConfig` for endpoint/auth/version/beta/model-in-body configuration, `AuthStyle` enum (`Bearer` | `XApiKey`) for selecting between `Authorization: Bearer` and `x-api-key` headers, `AnthropicCompatSseParser` for SSE stream parsing, and `AnthropicCompatBackend` for HTTP sending + stream creation; `vertex.rs` re-exports `VertexSseParser` and `parse_sse_data` for back-compat), `zai` (z.ai — thin shim over `openai_compat`), `ollama` (Ollama Cloud/self-hosted via NDJSON, with `ndjson.rs` for NDJSON stream parsing), `openai_compat` (generic OpenAI Chat Completions client — SSE, message serialisation, tool calls), and `opencode_go` (OpenCode Go — dual-protocol backend that triages model name against config-driven lists to route to either `OpenAiCompatBackend` for OpenAI Chat Completions or `AnthropicCompatBackend` for Anthropic Messages; the Anthropic path uses `AuthStyle::XApiKey`, `anthropic-version: 2023-06-01`, `include_model_in_body: true`, and `anthropic-beta: interleaved-thinking-2025-05-14`). `openai_compat` exposes `ReasoningStyle` (`None` | `ZaiEnableThinking` | `QwenChatTemplate` | `Default`) to control how extended-thinking is expressed per-provider. The `zai` backend is a back-compat shim that constructs an `OpenAiCompatBackend` with `ReasoningStyle::ZaiEnableThinking` and the z.ai base URL. The constructor strips trailing `/` from `base_url` and rejects URLs that already include `/chat/completions`. When `api_key` is `None` or empty, no `Authorization` header is sent. Emits `StreamEvent` (TextDelta | ToolUseStart/Delta/Done | Usage | Done). `error.rs` defines `BackendError` (HttpStatus | Transport | Other | MaxTokensExceeded | Refusal) with `is_retryable()` returning true for 429 and all 5xx except 501, and for every `Transport` error. `Refusal` is emitted by all backends when the model returns a refusal or content-filter stop reason (Vertex/anthropic_compat: `stop_reason == "refusal"` or `"content_filter"`, OpenAI-compat: `finish_reason == "content_filter"` or `"refusal"`, Ollama: `done_reason == "content_filter"` or `"refusal"`); it is not retryable and the agent emits `AgentEvent::Error` without injecting a `[ERROR]` message into conversation history (unlike the normal error path). `MaxTokensExceeded` carries `input_tokens` and `output_tokens`, is not retryable by `RetryingBackend` (the retry happens at the agent loop level), and all three backends emit it when the model hits its `max_tokens` output limit (Vertex/anthropic_compat: `stop_reason == "max_tokens"`, OpenAI-compat: `finish_reason == "length"` or stealth detection when `completion_tokens >= max_tokens` with a non-`"length"` finish_reason, Ollama: `done_reason == "length"` or stealth detection when `eval_count >= max_tokens` with a non-`"length"` done_reason). The Ollama and OpenAI-compat parsers receive `max_tokens` at construction time to enable stealth detection; when `max_tokens == 0` the stealth check is skipped. All backends return `BackendError::HttpStatus` for non-2xx responses, and map request-send failures (connection refused, DNS, TLS, timeout) to `BackendError::Transport` via `BackendError::transport()`, which flattens the underlying error's `source()` chain into a single truthful message so no cause is discarded. `RetryingBackend` wraps every concrete backend (via `BackendFactory::for_role()`) and retries on retryable HTTP and transport errors with exponential backoff + jitter, up to `max_retries` (config: `[retry]` section). During backoff sleep, `RequestConfig.cancel_token` is polled so user cancellation aborts the retry loop immediately.

2. **Agent Core** (`src/agent.rs`) — Owns conversation history, context files, and skills. Wraps backend streams into `AgentEvent` (TokenReceived | ToolUseReceived | ToolResult | ToolConfirmationRequired | ResponseComplete | Error | Retrying | Usage | SubAgentUsage). Display-agnostic. Drives agentic tool-use loops up to `max_tool_iterations`. When a `BackendError::MaxTokensExceeded` error occurs in the stream, the agent retries (up to `max_token_retries` from `[retry]` config), injecting the error into conversation history each time so the model can self-correct; this counter is separate from `max_tool_iterations`. During these retries the agent emits `AgentEvent::Retrying` (not `AgentEvent::Error`) so frontends stay in their current state — only when retries are exhausted does the agent emit a terminal `AgentEvent::Error`. Additionally, when the stream ends normally with zero text, zero thinking, and zero tool calls but `output_tokens >= max_tokens`, the agent injects a `MaxTokensExceeded` error and triggers the same retry path (stealth max-tokens safety net). When a `BackendError::Refusal` error occurs, the agent emits `AgentEvent::Error` immediately and breaks the loop without injecting a `[ERROR]` message into history (context is poisoned; no retry). Tool calls within a single assistant turn run concurrently via `join_all`; confirmations are gathered sequentially first, then approved calls execute in parallel. Tool results exceeding `max_tool_result_bytes` are truncated with head+tail preserved and a sentinel message before being added to history; the untruncated content is still sent to the TUI via `AgentEvent::ToolResult`. Supports session switching (`load_session`) while preserving non-persisted context prefix (context files, skill definitions). Backend and model can be swapped at runtime via `set_backend()` (used by `/role`). Messages carry a `created_at` (f64 Unix seconds) timestamp set at creation time.

3. **Tools Layer** (`src/tools/`) — `Tool` trait + `ToolRegistry`. Built-in tools: `bash` (allowlist/denylist enforced), `edit_file`, `write_file`, `skill` (loads skill prompts by name), `search` (semantic code search), `agent` (spawns sub-agents). File tools are sandboxed to `sandbox_root` via `SandboxPolicy` (`sandbox.rs`). `SandboxPolicy` supports a primary root plus any number of `extra_roots`; a path is valid if its canonical form lies within any registered root. The platform temp directory (`std::env::temp_dir()`) and, on Unix, `/tmp` are always registered as extra roots for `edit_file`/`write_file` — both are needed on macOS, where `temp_dir()` follows `$TMPDIR` to `/var/folders/...` rather than `/tmp` — so file tools can read/write temp space regardless of `sandbox_root`; no config knob, not a net privilege gain since `bash` already has unrestricted `/tmp` access. Relative paths always resolve against the primary root only; `/tmp` access is absolute-path-only. `is_write_tool()` determines whether confirmation is required under `WriteOnly` mode.

The `agent` tool spawns an independent sub-agent with its own session, model, and tool set. The parent configures the backend role, confirmation mode (clamped so the sub-agent can never be more permissive than the parent), and an optional tool allowlist. Sub-agent token usage propagates to the TUI status line via `AgentEvent::SubAgentUsage`. Permission clamping is enforced by `clamp_confirmation` in `src/agent.rs` — strictness order: `Always` > `WriteOnly` > `Never`.

4. **Frontend Layer** (`src/frontend/`) — Two frontends consuming the same AgentEvent stream:
   - `stdout.rs`: Single-shot mode, streams tokens to stdout, pipe-friendly. Timestamps shown as `[YYYYMMDD-HH:MM]` before tool/result lines.
   - `tui/`: Ratatui interactive REPL with vim-style input (`input_area.rs` — powered by the hjkl crate stack: `hjkl-form`'s `TextFieldEditor` for vim modal editing, `hjkl-buffer` for word-wrapped rendering, `hjkl-engine` for vim FSM, `hjkl-editor-tui` for crossterm KeyEvent bridging; custom ratatui rendering with selection highlighting and cursor positioning), scrollable conversation display (`conversation_area.rs`), syntax-highlighted diffs for file tools (`diff.rs` using `syntect` + `similar`), syntax-highlighted code blocks in markdown (`syntect_highlight.rs` using `syntect` via the `the-other-tui-markdown` `RendererBuilder` `with_code_block` hook), a session picker overlay (`session_picker.rs`), and timestamp formatting (see `src/timestamp.rs` — converts Unix `f64` to local `[YYYYMMDD-HH:MM]`). Built-in slash commands: `/sessions`, `/model`, `/tasks`, `/compact`, `/new` (fresh session), `/role <name>` (switch backend+model by named role; `/role` alone shows current config).

**Semantic Search** (`src/tools/search.rs`) — Thin wrapper around the `search-semantically` crate. Exposes semantic code search as a tool with support for natural language queries, identifier names, and file path patterns. The `search-semantically` crate provides tree-sitter AST chunking, ONNX Runtime embeddings (`all-MiniLM-L6-v2`), 6-signal POEM ranking (BM25/FTS5, cosine similarity, path match, symbol match, import graph, git recency), and incremental indexing via SQLite at `<project_root>/.search-index/search.db`.

**Session persistence** (`src/session.rs`) — Each session is a SQLite database (via `turso`) identified by a UUIDv7. Conversations are persisted per-message with a `created_at` Unix timestamp (f64, seconds). Sessions can be listed, resumed, and deleted. The session picker in the TUI allows browsing and switching sessions.

**Context files** (`src/context_files.rs`) — On startup, CLAUDE.md and AGENTS.md are auto-discovered from `~/.claude/`, `pwd/.claude/`, the current working directory, and the home directory, then injected into the conversation as initial context (not persisted to the session DB).

**Skills** (`src/tools/skill.rs`) — Skill directories are discovered from `~/.claude/skills/` and `pwd/.claude/skills/` (pwd overrides home). Each skill is a directory containing a `SKILL.md` file with optional YAML frontmatter (`description` field). Skills are registered as a tool and their names/descriptions are prepended to conversation history.

Key types live in `src/types.rs`. Configuration loading and CLI merge logic is in `src/config.rs`. Debug file logging is in `src/logging.rs`.

## Configuration

Config file at `~/.config/illustrious-manager/config.toml` (auto-created on first run):

```toml
backend = "vertex"                    # "vertex", "zai", "ollama", "openai_compat", or "opencode_go"
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
# reasoning = "qwen_chat_template"

# [opencode_go]
# OpenCode Go — dual-protocol backend. Triages model name against config-driven
# lists to route to either OpenAI Chat Completions or Anthropic Messages.
# api_key = ""                        # required
# base_url = "https://opencode.ai/zen/go/v1"
# model = "grok-code-fast"             # default model for default role
# openai_models = ["grok-code-fast", "grok-code", "glm-4.6-code", "kimi-k2-code", "deepseek-v3.2-code", "mimo-7b-code"]
# anthropic_models = ["minimax-m1", "qwen3-coder-plus"]
# max_tokens = 16384                  # optional override
# reasoning = "default"               # none | zai_enable_thinking | qwen_chat_template | default    # none | zai_enable_thinking | qwen_chat_template | default

# [retry]
# max_retries = 3                     # retry attempts for transient HTTP (5xx, 429) and transport errors
# initial_delay_ms = 1000             # first retry delay
# max_delay_ms = 8000                 # cap for exponential backoff
# max_token_retries = 3               # retries when the model hits max_tokens output limit (agent loop level)

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
