//! Per-backend `max_tokens` default constants.
//!
//! Each backend defines a single default here — the one value used by
//! `BackendFactory::for_role()` when no config override is present. This
//! ensures the main agent, sub-agents, and role-switched agents all resolve
//! to the same per-backend default.

/// Vertex AI — no config override mechanism exists; always 8192.
pub const VERTEX: u32 = 8_192;

/// z.ai — no config override field; always 8192.
pub const ZAI: u32 = 8_192;

/// Ollama — no config override field; always 8192.
pub const OLLAMA: u32 = 8_192;

/// OpenAI-compatible — matches the config example in CLAUDE.md.
pub const OPENAI_COMPAT: u32 = 16_384;

/// Direct Anthropic Messages API — adaptive thinking requires large budgets.
pub const ANTHROPIC: u32 = 65_536;

/// OpenCode Go — matches the config example in CLAUDE.md.
pub const OPENCODE_GO: u32 = 16_384;
