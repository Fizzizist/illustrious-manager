//! Per-backend `max_tokens` default constants.
//!
//! Each backend defines a single default here — the one value used by
//! `BackendFactory::for_role()` when no config override is present. This
//! ensures the main agent, sub-agents, and role-switched agents all resolve
//! to the same per-backend default.

/// Vertex AI — used when `[vertex].max_tokens` is absent.
pub const VERTEX: u32 = 8_192;

/// z.ai — no config override field; always 8192.
pub const ZAI: u32 = 8_192;

/// Ollama — used when `[ollama].max_tokens` is absent.
pub const OLLAMA: u32 = 8_192;

/// OpenAI-compatible — matches the config example in CLAUDE.md.
pub const OPENAI_COMPAT: u32 = 16_384;

/// Direct Anthropic Messages API — adaptive thinking requires large budgets.
pub const ANTHROPIC: u32 = 65_536;

/// OpenCode Go — matches the config example in CLAUDE.md.
pub const OPENCODE_GO: u32 = 16_384;

// ── Vision defaults ────────────────────────────────────────────────────
//
// Per-backend default for whether the backend's models support image input.
// Overridable via `[<backend>].vision` in config. Backends whose models are
// predominantly multi-modal (Anthropic, Vertex/Claude, OpenCode Go's
// Anthropic path) default to `true`; OpenAI-compat, Ollama, and z.ai
// default to `false` since model vision support varies by deployment.

pub const VISION_VERTEX: bool = true;
pub const VISION_ZAI: bool = false;
pub const VISION_OLLAMA: bool = false;
pub const VISION_OPENAI_COMPAT: bool = false;
pub const VISION_ANTHROPIC: bool = true;
pub const VISION_OPENCODE_GO: bool = true;
