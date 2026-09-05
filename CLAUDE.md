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

1. **Backend Layer** (`src/backend/`) — `LlmBackend` trait abstraction over LLM providers. Seven implementations: `vertex` (Vertex AI + Claude — partial shim that resolves GCP auth and constructs the Vertex endpoint, then delegates to `anthropic_compat`), `anthropic` (direct Anthropic Messages API — thin shim that delegates to `anthropic_compat` with `AuthStyle::XApiKey`, `include_model_in_body: true`, `anthropic-version: 2023-06-01`, no `anthropic-beta` header; omits beta entirely since adaptive thinking on Opus 4.8 auto-enables interleaved thinking), `anthropic_compat` (generic Anthropic Messages API protocol client — SSE parsing, request body building, content block filtering; owns `AnthropicCompatConfig` for endpoint/auth/version/beta/model-in-body configuration, `AuthStyle` enum (`Bearer` | `XApiKey`) for selecting between `Authorization: Bearer` and `x-api-key` headers, and the `include_model_in_body` flag which when `true` (direct Anthropic API) sends `anthropic-version` as an HTTP header and includes `model` in the request body, and when `false` (Vertex) puts `anthropic_version` in the request body and omits `model` (it is embedded in the URL); `AnthropicCompatSseParser` for SSE stream parsing, and `AnthropicCompatBackend` for HTTP sending + stream creation; `vertex.rs` re-exports `VertexSseParser` and `parse_sse_data` for back-compat), `zai` (z.ai — thin shim over `openai_compat`), `ollama` (Ollama Cloud/self-hosted via NDJSON, with `ndjson.rs` for NDJSON stream parsing), `openai_compat` (generic OpenAI Chat Completions client — SSE, message serialisation, tool calls), and `opencode_go` (OpenCode Go — dual-protocol backend that triages model name against config-driven lists to route to either `OpenAiCompatBackend` for OpenAI Chat Completions or `AnthropicCompatBackend` for Anthropic Messages; the Anthropic path uses `AuthStyle::XApiKey`, `anthropic-version: 2023-06-01`, `include_model_in_body: true`, and `anthropic-beta: interleaved-thinking-2025-05-14`). `openai_compat` exposes `ReasoningStyle` (`None` | `ZaiEnableThinking` | `QwenChatTemplate` | `Default`) to control how extended-thinking is expressed per-provider. The `zai` backend is a back-compat shim that constructs an `OpenAiCompatBackend` with `ReasoningStyle::ZaiEnableThinking` and the z.ai base URL. The constructor strips trailing `/` from `base_url` and rejects URLs that already include `/chat/completions`. The `anthropic` backend constructor strips trailing `/` from `base_url`, rejects a `/messages` suffix, and appends `/messages`. When `api_key` is `None` or empty, no `Authorization` header is sent (OpenAI-compat); the `anthropic` backend bails on empty `api_key` at construction. Emits `StreamEvent` (TextDelta | ToolUseStart/Delta/Done | Usage | Done). `error.rs` defines `BackendError` (HttpStatus | Transport | Other | MaxTokensExceeded | Refusal | Cancelled) with `is_retryable()` returning true for 429 and all 5xx except 501, and for every `Transport` error. `Refusal` is emitted by all backends when the model returns a refusal or content-filter stop reason (Vertex/anthropic_compat: `stop_reason == "refusal"` or `"content_filter"`, OpenAI-compat: `finish_reason == "content_filter"` or `"refusal"`, Ollama: `done_reason == "content_filter"` or `"refusal"`); it is not retryable and the agent emits `AgentEvent::Error` without injecting a `[ERROR]` message into conversation history (unlike the normal error path). `MaxTokensExceeded` carries `input_tokens` and `output_tokens`, is not retryable by `RetryingBackend` (the retry happens at the agent loop level), and all three backends emit it when the model hits its `max_tokens` output limit (Vertex/anthropic_compat: `stop_reason == "max_tokens"`, OpenAI-compat: `finish_reason == "length"` or stealth detection when `completion_tokens >= max_tokens` with a non-`"length"` finish_reason, Ollama: `done_reason == "length"` or stealth detection when `eval_count >= max_tokens` with a non-`"length"` done_reason). The Ollama and OpenAI-compat parsers receive `max_tokens` at construction time to enable stealth detection; when `max_tokens == 0` the stealth check is skipped. All backends return `BackendError::HttpStatus` for non-2xx responses, and map request-send failures (connection refused, DNS, TLS, timeout) to `BackendError::Transport` via `BackendError::transport()`, which flattens the underlying error's `source()` chain into a single truthful message so no cause is discarded. All production backends construct their `reqwest::Client` through the shared `build_http_client()` in `src/backend/mod.rs`, which sets a 30s connect timeout and deliberately no total/read timeout (long thinking streams must not be killed). `error.rs` also defines the `Cancelled` variant: not retryable, surfaced when `RetryingBackend` observes the cancel token. `RetryingBackend` wraps every concrete backend (via `BackendFactory::for_role()`) and retries on retryable HTTP and transport errors with exponential backoff + jitter, up to `max_retries` (config: `[retry]` section). It is fully cancel-aware: a pre-cancelled token skips the inner call, each send attempt is raced against `token.cancelled()` (biased token-first; dropping the losing reqwest future aborts the in-flight request, covering TCP connect/TLS/auth-fetch/pre-first-token stalls across all backends), and a cancel during backoff sleep — or during the attempt — yields the typed `BackendError::Cancelled`, never the last retryable error, so the agent never records a spurious `[ERROR]`. `BackendSelection` carries `max_tokens` resolved by `for_role()` from the backend's config override (e.g., `[vertex].max_tokens`, `[ollama].max_tokens`, `[anthropic].max_tokens`) or, when no override is set, from the per-backend default constants in `src/backend/defaults.rs` (`VERTEX`/`ZAI`/`OLLAMA`: 8192, `OPENAI_COMPAT`/`OPENCODE_GO`: 16384, `ANTHROPIC`: 65536). This single source of truth feeds the main agent, sub-agents (`spawn_agent`), and role-switched agents, ensuring no stale `max_tokens` after `/role`.

2. **Agent Core** (`src/agent.rs`) — Owns conversation history, context files, and skills. Wraps backend streams into `AgentEvent` (TokenReceived | ThinkingReceived | ToolUseReceived | ToolResult | ToolConfirmationRequired | ResponseComplete | Error | Retrying | Usage | SubAgentUsage | Interrupted | Warn | CompactionComplete | BashCommandComplete | AutoCompactTriggered). Display-agnostic. Drives agentic tool-use loops up to `max_tool_iterations`. When a `BackendError::MaxTokensExceeded` error occurs in the stream, the agent retries (up to `max_token_retries` from `[retry]` config), injecting the error into conversation history each time so the model can self-correct; this counter is separate from `max_tool_iterations`. When a `BackendError::HttpStatus(400)` ("bad request") error occurs — e.g. the model received content it cannot handle such as an image block on a backend that does not support image input — the agent strips all `ContentBlock::Image` blocks from history (replacing them with text placeholders, recursing into `ToolResult.content`) so the next request does not contain the offending content, injects the error message into history so the model can self-correct (e.g. stop calling `image_viewer`), emits `AgentEvent::Retrying` (not `AgentEvent::Error`), and continues the loop. The 400-retry path and the max-token retry path carry independent budgets (both bounded by `max_token_retries`): `bad_request_retries_used` for 400s and `max_token_retries_used` for max-tokens, so max-tokens hits cannot starve the image-stripping 400 path (and vice versa). Image-stripped history is re-persisted via `Conversation::replace_all` — a transactional deactivate-all + re-insert (crash-safe) restricted to the persisted suffix of history beyond the non-persisted context prefix, so context files are never written to the session DB. Both `send_message`-level errors and mid-stream errors are handled. During these retries the agent emits `AgentEvent::Retrying` (not `AgentEvent::Error`) so frontends stay in their current state — only when retries are exhausted does the agent emit a terminal `AgentEvent::Error`. Additionally, when the stream ends normally with zero text, zero tool calls, and either (`output_tokens >= max_tokens`) or (`output_tokens == 0` — a truly empty stream with no usage event), the agent injects an error and triggers the same retry path (stealth max-tokens safety net). The `output_tokens == 0` branch catches streams that emit `Done` without a preceding `Usage` event (e.g., `message_stop` with no `message_delta`), and the `thinking_accumulated.is_empty()` guard was removed so that thinking-only responses that consume the entire budget also trigger the retry. When the cause is definitively `MaxTokensExceeded` (output_tokens >= max_tokens), the typed `BackendError::MaxTokensExceeded` is injected; otherwise a descriptive error message is injected. When a `BackendError::Refusal` error occurs, the agent emits `AgentEvent::Error` immediately and breaks the loop without injecting a `[ERROR]` message into history (context is poisoned; no retry). When a `BackendError::Cancelled` error occurs — the typed signal that the cancel token fired during the request phase (attempt racing or retry backoff) — the triage in `backend_error_disposition`/`record_backend_error` (a `BackendErrorDisposition::Cancelled` arm, checked first so cancellation never consumes the bad-request or max-tokens retry budgets) calls `persist_partial_and_interrupt` and breaks the loop: the agent emits `AgentEvent::Interrupted` with no `Error` event and no `[ERROR]` history injection. Tool calls within a single assistant turn run concurrently via `join_all`; confirmations are gathered sequentially first, then approved calls execute in parallel (each execution raced against the cancel token with biased token-first, so an in-flight tool future is dropped on cancel). Tool results exceeding `max_tool_result_bytes` are truncated with head+tail preserved and a sentinel message before being added to history; the untruncated content is still sent to the TUI via `AgentEvent::ToolResult`. Supports session switching (`load_session`) while preserving non-persisted context prefix (context files, skill definitions). Backend, model, and `max_tokens` can be swapped at runtime via `set_backend()` (used by `/role`), which syncs `RequestConfig.max_tokens` from the new `BackendSelection`. Messages carry a `created_at` (f64 Unix seconds) timestamp set at creation time.

3. **Tools Layer** (`src/tools/`) — `Tool` trait + `ToolRegistry`. Built-in tools: `bash` (allowlist/denylist enforced), `edit_file`, `write_file`, `skill` (loads skill prompts by name), `search` (semantic code search), `agent` (spawns sub-agents), `image_viewer` (reads image files from disk and returns them as `ContentBlock::Image` content blocks; supported formats governed by the `ImageFormat` enum in `image_viewer.rs` — extensions, MIME types, magic-byte headers, and all schema/description/error prose derive from it — registered unconditionally: if the backend does not support image input, the model receives a 400 and the agent's 400-retry path strips image blocks from history so the model can self-correct). File tools are sandboxed to `sandbox_root` via `SandboxPolicy` (`sandbox.rs`). `SandboxPolicy` supports a primary root plus any number of `extra_roots`; a path is valid if its canonical form lies within any registered root. The platform temp directory (`std::env::temp_dir()`) and, on Unix, `/tmp` are always registered as extra roots for `edit_file`/`write_file` — both are needed on macOS, where `temp_dir()` follows `$TMPDIR` to `/var/folders/...` rather than `/tmp` — so file tools can read/write temp space regardless of `sandbox_root`; no config knob, not a net privilege gain since `bash` already has unrestricted `/tmp` access. Relative paths always resolve against the primary root only; `/tmp` access is absolute-path-only. `is_write_tool()` determines whether confirmation is required under `WriteOnly` mode.

The `agent` tool spawns an independent sub-agent with its own session, model, and tool set. The parent configures the backend role, confirmation mode (clamped so the sub-agent can never be more permissive than the parent), and an optional tool allowlist. Sub-agent token usage propagates to the TUI status line via `AgentEvent::SubAgentUsage`. Permission clamping is enforced by `clamp_confirmation` in `src/agent.rs` — strictness order: `Always` > `WriteOnly` > `Never`.

4. **Frontend Layer** (`src/frontend/`) — Two frontends consuming the same AgentEvent stream:
   - `stdout.rs`: Single-shot mode, streams tokens to stdout, pipe-friendly. Timestamps shown as `[YYYYMMDD-HH:MM]` before tool/result lines.
   - `tui/`: Ratatui interactive REPL with vim-style input (`input_area.rs` — powered by the hjkl crate stack: `hjkl-form`'s `TextFieldEditor` for vim modal editing, `hjkl-buffer` for word-wrapped rendering, `hjkl-engine` for vim FSM, `hjkl-editor-tui` for crossterm KeyEvent bridging; custom ratatui rendering with selection highlighting and cursor positioning), scrollable conversation display (`conversation_area.rs`), syntax-highlighted diffs for file tools (`diff.rs` using `syntect` + `similar`), syntax-highlighted code blocks in markdown (`syntect_highlight.rs` using `syntect` via the `the-other-tui-markdown` `RendererBuilder` `with_code_block` hook), a session picker overlay (`session_picker.rs` — paginated: hydrates the first 100 sessions on open; a `... (load more sessions)` sentinel row at the bottom loads the next 100 on Enter; backed by `ListPicker`'s `extend`/`pop_last` APIs), and timestamp formatting (see `src/timestamp.rs` — converts Unix `f64` to local `[YYYYMMDD-HH:MM]`). Built-in slash commands: `/sessions`, `/model`, `/tasks`, `/compact`, `/new` (fresh session), `/role <name>` (switch backend+model by named role; `/role` alone shows current config). Key dispatch follows a three-tier semantic priority in `tui_app.rs`: `application_command` (`ctrl+c` → `Quit`, `Esc` cancels active streams/bash and rejects tool confirmation in any focus) → `focus_command` (focus-switching `Ctrl+W` chord, conversation scroll under `AppFocus::Conversation`) → state machine (per-state key handling). The `KeyDisposition` enum (`Quit` | `Consumed` | `PassThrough`) expresses the result at each tier, enabling universal `ctrl+c` quit that cannot be swallowed by focus commands or the `pending_w` chord.

**Semantic Search** (`src/tools/search.rs`) — Thin wrapper around the `search-semantically` crate. Exposes semantic code search as a tool with support for natural language queries, identifier names, and file path patterns. The `search-semantically` crate provides tree-sitter AST chunking, ONNX Runtime embeddings (`all-MiniLM-L6-v2`), 6-signal POEM ranking (BM25/FTS5, cosine similarity, path match, symbol match, import graph, git recency), and incremental indexing via SQLite at `<project_root>/.search-index/search.db`.

**Session persistence** (`src/session.rs`) — Each session is a SQLite database (via `turso`) identified by a UUIDv7. Conversations are persisted per-message with a `created_at` Unix timestamp (f64, seconds). Sessions can be listed, resumed, and deleted. `enumerate_sessions` performs cheap directory listing + mtime sorting and returns `Vec<SessionRef>` without opening any DB; `hydrate_sessions` opens the DBs for a given slice of refs and extracts `first_user_message`. `hydrate_next_page` drains the next `page_size` refs from a pending vector and returns `(Vec<SessionSummary>, has_more)`. `list_sessions` remains as a convenience wrapper (enumerate + hydrate all). The session picker in the TUI is paginated (100 sessions per page). `Conversation::compact` (deactivate + insert summary) and `Conversation::replace_all` (deactivate + re-insert messages) both wrap their multi-step writes in a transaction so a crash cannot leave the DB with zero active rows.

**Context files** (`src/context_files.rs`) — On startup, CLAUDE.md and AGENTS.md are auto-discovered as bare `CLAUDE.md`/`AGENTS.md` in both the current working directory and the home directory, plus the subdirectory variants `.claude/CLAUDE.md` and `.agents/AGENTS.md` in both roots, then injected into the conversation as initial context (not persisted to the session DB).

**Skills** (`src/tools/skill.rs`) — Skill directories are discovered from `~/.claude/skills/`, `~/.agents/skills/`, `pwd/.claude/skills/`, and `pwd/.agents/skills/` (pwd overrides home). Each skill is a directory containing a `SKILL.md` file with optional YAML frontmatter (`description` field). Skills are registered as a tool and their names/descriptions are prepended to conversation history.

Key types live in `src/types.rs` — including `ContentBlock` which has variants `Text`, `Image { media_type, data }` (base64-encoded image content for multi-modal LLM input, serialized in the Anthropic wire format as `{"type":"image","source":{"type":"base64","media_type":"...","data":"..."}}`), `ToolUse`, `ToolResult` (whose `content: Vec<ContentBlock>` supports structured results — e.g. an image + text companion from `image_viewer`; a legacy `String` deserialization arm keeps old session databases readable), `Thinking`, and `RedactedThinking`. `ContentBlock::image_placeholder(media_type)` is the single source of the `[image: media_type]` display placeholder used by the agent's display path and the TUI history rebuild (the distinct `[image removed: ...]` error-context message in the 400-retry path is intentionally separate). Configuration loading and CLI merge logic is in `src/config.rs`. Debug file logging is in `src/logging.rs`.

## Configuration

Config file at `~/.config/illustrious-manager/config.toml` (auto-created on first run):

```toml
backend = "vertex"                    # "vertex", "zai", "ollama", "openai_compat", "anthropic", or "opencode_go"
# sessions_dir = "/path/to/sessions" # defaults to ~/.config/illustrious-manager/sessions

[vertex]
project = ""                          # GCP project ID (required)
region = "us-east5"
model = "claude-sonnet-4-20250514"
# max_tokens = 8192                   # optional override

[zai]
api_key = ""                          # z.ai API key (required for zai backend)
model = "glm-5.1"

[ollama]
api_key = ""                          # Ollama API key (required for ollama backend)
model = "gpt-oss:120b"
# base_url = "https://ollama.com/api/chat"  # change for self-hosted Ollama
# max_tokens = 8192                   # optional override

# [openai_compat]
# base_url = "https://vllm.k8s.dc.rxrx.io/v1"  # required, without /chat/completions
# api_key = ""                        # omit or leave empty for unauthenticated endpoints
# model = "Qwen/Qwen3-32B-FP8"
# max_tokens = 16384                  # optional override
# reasoning = "qwen_chat_template"    # none | zai_enable_thinking | qwen_chat_template | default

# [opencode_go]
# OpenCode Go — dual-protocol backend. Triages model name against config-driven
# lists to route to either OpenAI Chat Completions or Anthropic Messages.
# api_key = ""                        # required
# base_url = "https://opencode.ai/zen/go/v1"
# model = "grok-code-fast"             # default model for default role
# openai_models = ["grok-code-fast", "grok-code", "glm-4.6-code", "kimi-k2-code", "deepseek-v3.2-code", "mimo-7b-code"]
# anthropic_models = ["minimax-m1", "qwen3-coder-plus"]
# max_tokens = 16384                  # optional override
# reasoning = "default"               # none | zai_enable_thinking | qwen_chat_template | default

# [anthropic]
# Direct Anthropic Messages API backend.
# api_key = ""                        # required
# base_url = "https://api.anthropic.com/v1"
# model = "claude-opus-4-8"            # default model
# max_tokens = 65536                  # optional override (per-backend default is 65536)

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
