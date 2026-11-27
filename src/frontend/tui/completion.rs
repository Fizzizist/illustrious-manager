use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};

const MAX_DIR_ENTRIES: usize = 10_000;
const COMPLETION_CHANNEL_CAP: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionContext {
    pub token: String,
    pub is_input_start: bool,
}

pub fn extract_completion_context(text: &str, cursor: (usize, usize)) -> Option<CompletionContext> {
    let (row, col) = cursor;
    let line = text.split('\n').nth(row)?;
    let chars: Vec<char> = line.chars().collect();
    if col == 0 || col > chars.len() {
        return None;
    }
    if let Some(&after) = chars.get(col)
        && !after.is_whitespace()
    {
        return None;
    }
    if chars[col - 1].is_whitespace() {
        return None;
    }
    let mut start = col;
    while start > 0 && !chars[start - 1].is_whitespace() {
        start -= 1;
    }
    let token: String = chars[start..col].iter().collect();
    if token.is_empty() {
        return None;
    }
    let is_input_start = row == 0 && chars[..start].iter().all(|c| c.is_whitespace());
    Some(CompletionContext {
        token,
        is_input_start,
    })
}

pub trait CompletionProvider: Send + Sync {
    fn matches_trigger(&self, ctx: &CompletionContext) -> bool;
    fn candidates(&self, ctx: &CompletionContext) -> Vec<String>;
}

pub struct SlashCommandProvider {
    commands: Vec<String>,
}

impl SlashCommandProvider {
    pub fn new(commands: Vec<String>) -> Self {
        Self { commands }
    }
}

impl CompletionProvider for SlashCommandProvider {
    fn matches_trigger(&self, ctx: &CompletionContext) -> bool {
        ctx.is_input_start && ctx.token.starts_with('/') && !ctx.token[1..].contains('/')
    }

    fn candidates(&self, ctx: &CompletionContext) -> Vec<String> {
        self.commands
            .iter()
            .map(|c| format!("/{}", c.trim_start_matches('/')))
            .filter(|c| c.starts_with(&ctx.token))
            .collect()
    }
}

pub struct SkillProvider {
    skills: Vec<String>,
}

impl SkillProvider {
    pub fn new(skills: Vec<String>) -> Self {
        Self { skills }
    }
}

impl CompletionProvider for SkillProvider {
    fn matches_trigger(&self, ctx: &CompletionContext) -> bool {
        ctx.token.starts_with('/') && !ctx.token[1..].contains('/')
    }

    fn candidates(&self, ctx: &CompletionContext) -> Vec<String> {
        self.skills
            .iter()
            .map(|s| format!("/{}", s.trim_start_matches('/')))
            .filter(|s| s.starts_with(&ctx.token))
            .collect()
    }
}

pub struct PathProvider {
    working_dir: PathBuf,
    home_dir: Option<PathBuf>,
}

impl PathProvider {
    pub fn new(working_dir: PathBuf, home_dir: Option<PathBuf>) -> Self {
        Self {
            working_dir,
            home_dir,
        }
    }

    fn resolve_dir(&self, dir_prefix: &str) -> Option<PathBuf> {
        if let Some(rest) = dir_prefix.strip_prefix("~/") {
            let home = self.home_dir.as_ref()?;
            let rest = rest.trim_start_matches('/');
            if rest.is_empty() {
                Some(home.clone())
            } else {
                Some(home.join(rest))
            }
        } else if dir_prefix.starts_with('/') {
            Some(PathBuf::from(dir_prefix))
        } else {
            Some(self.working_dir.join(dir_prefix))
        }
    }
}

impl CompletionProvider for PathProvider {
    fn matches_trigger(&self, ctx: &CompletionContext) -> bool {
        !ctx.token.is_empty()
            && !ctx.token.contains("://")
            && (ctx.token.starts_with("./")
                || ctx.token.starts_with("../")
                || ctx.token.starts_with('/')
                || ctx.token.starts_with("~/")
                || ctx.token.contains('/'))
    }

    fn candidates(&self, ctx: &CompletionContext) -> Vec<String> {
        let Some(last_slash) = ctx.token.rfind('/') else {
            return Vec::new();
        };
        let dir_prefix = &ctx.token[..=last_slash];
        let file_prefix = &ctx.token[last_slash + 1..];
        let Some(dir_path) = self.resolve_dir(dir_prefix) else {
            return Vec::new();
        };
        let Ok(entries) = std::fs::read_dir(&dir_path) else {
            return Vec::new();
        };

        let mut results = Vec::new();
        for entry in entries.take(MAX_DIR_ENTRIES) {
            let Ok(entry) = entry else {
                continue;
            };
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            if !name.starts_with(file_prefix) {
                continue;
            }
            let is_dir = match entry.file_type() {
                Ok(ft) if ft.is_dir() => true,
                Ok(ft) if ft.is_symlink() => {
                    std::fs::metadata(entry.path()).is_ok_and(|m| m.is_dir())
                }
                _ => false,
            };
            let mut candidate = String::with_capacity(dir_prefix.len() + name.len() + 1);
            candidate.push_str(dir_prefix);
            candidate.push_str(&name);
            if is_dir {
                candidate.push('/');
            }
            results.push(candidate);
        }
        results
    }
}

#[derive(Default)]
pub struct Completer {
    tiers: Vec<Vec<Box<dyn CompletionProvider>>>,
}

impl Completer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_defaults(
        working_dir: PathBuf,
        home_dir: Option<PathBuf>,
        commands: Vec<String>,
        skills: Vec<String>,
    ) -> Self {
        let mut completer = Self::new();
        completer.add_tier(vec![
            Box::new(SlashCommandProvider::new(commands)),
            Box::new(SkillProvider::new(skills)),
        ]);
        completer.add_tier(vec![Box::new(PathProvider::new(working_dir, home_dir))]);
        completer
    }

    pub fn add_tier(&mut self, providers: Vec<Box<dyn CompletionProvider>>) {
        self.tiers.push(providers);
    }

    pub fn register_provider(&mut self, tier: usize, provider: Box<dyn CompletionProvider>) {
        if tier >= self.tiers.len() {
            self.tiers.resize_with(tier + 1, Vec::new);
        }
        self.tiers[tier].push(provider);
    }

    pub fn needs_completion(&self, text: &str, cursor: (usize, usize)) -> bool {
        let Some(ctx) = extract_completion_context(text, cursor) else {
            return false;
        };
        self.tiers.iter().flatten().any(|p| p.matches_trigger(&ctx))
    }

    pub fn complete(&self, text: &str, cursor: (usize, usize)) -> Option<String> {
        let ctx = extract_completion_context(text, cursor)?;
        for tier in &self.tiers {
            let best = tier
                .iter()
                .filter(|p| p.matches_trigger(&ctx))
                .flat_map(|p| p.candidates(&ctx))
                .filter(|c| c.starts_with(&ctx.token) && c.len() > ctx.token.len())
                .min();
            if let Some(best) = best {
                return Some(best[ctx.token.len()..].to_string());
            }
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionResult {
    pub generation: u64,
    pub suggestion: Option<String>,
}

#[derive(Clone)]
struct CompletionRequest {
    generation: u64,
    text: String,
    cursor: (usize, usize),
}

pub struct CompletionWorker {
    completer: Arc<Completer>,
    tx: watch::Sender<Option<CompletionRequest>>,
}

impl CompletionWorker {
    pub fn spawn(completer: Completer) -> (Self, mpsc::Receiver<CompletionResult>) {
        let completer = Arc::new(completer);
        let (tx, mut rx) = watch::channel::<Option<CompletionRequest>>(None);
        let (result_tx, result_rx) = mpsc::channel(COMPLETION_CHANNEL_CAP);

        let worker_completer = Arc::clone(&completer);
        tokio::spawn(async move {
            while rx.changed().await.is_ok() {
                let req = rx.borrow_and_update().clone();
                let Some(req) = req else {
                    continue;
                };
                let c = Arc::clone(&worker_completer);
                let generation = req.generation;
                let suggestion =
                    tokio::task::spawn_blocking(move || c.complete(&req.text, req.cursor))
                        .await
                        .unwrap_or(None);
                if result_tx
                    .send(CompletionResult {
                        generation,
                        suggestion,
                    })
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });

        (Self { completer, tx }, result_rx)
    }

    pub fn request(&self, generation: u64, text: String, cursor: (usize, usize)) -> bool {
        if !self.completer.needs_completion(&text, cursor) {
            return false;
        }
        let _ = self.tx.send(Some(CompletionRequest {
            generation,
            text,
            cursor,
        }));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tempfile::TempDir;

    fn make_ctx(token: &str, is_input_start: bool) -> CompletionContext {
        CompletionContext {
            token: token.to_string(),
            is_input_start,
        }
    }

    #[test]
    fn extracts_token_at_eol_and_mid_line_before_whitespace() {
        let eol = extract_completion_context("read src/ma", (0, 11))
            .expect("should extract token at EOL");
        assert_eq!(eol.token, "src/ma");
        assert!(!eol.is_input_start);

        let mid = extract_completion_context("read src/ma please", (0, 11))
            .expect("should extract token before whitespace");
        assert_eq!(mid.token, "src/ma");
        assert!(!mid.is_input_start);

        let multiline_eol = extract_completion_context("first line\ncheck ./src", (1, 11))
            .expect("should extract token at multiline EOL");
        assert_eq!(multiline_eol.token, "./src");
        assert!(!multiline_eol.is_input_start);

        let multiline_mid = extract_completion_context("line1\n  /sk next", (1, 5))
            .expect("should extract token before space on subsequent line");
        assert_eq!(multiline_mid.token, "/sk");
        assert!(!multiline_mid.is_input_start);
    }

    #[test]
    fn returns_none_when_cursor_mid_word_or_on_whitespace() {
        assert!(extract_completion_context("src/foo", (0, 5)).is_none());
        assert!(extract_completion_context("src/foo ", (0, 8)).is_none());
        assert!(extract_completion_context("   ", (0, 2)).is_none());
        assert!(extract_completion_context("", (0, 0)).is_none());
        assert!(extract_completion_context("src/foo", (0, 0)).is_none());
        assert!(extract_completion_context("foo bar", (0, 4)).is_none());
    }

    #[test]
    fn handles_multibyte_unicode_cursor_columns() {
        let cjk = extract_completion_context("你好 ./src", (0, 8))
            .expect("should handle CJK characters before token");
        assert_eq!(cjk.token, "./src");
        assert!(!cjk.is_input_start);

        let emoji = extract_completion_context("café 🚀 src/li", (0, 13))
            .expect("should handle accented and emoji characters");
        assert_eq!(emoji.token, "src/li");
        assert!(!emoji.is_input_start);

        let mid_multibyte = extract_completion_context("🔥 /sk 日本語", (0, 5))
            .expect("should handle multibyte chars before and after token");
        assert_eq!(mid_multibyte.token, "/sk");
        assert!(!mid_multibyte.is_input_start);

        let multibyte_token = extract_completion_context("./café", (0, 6))
            .expect("should handle multibyte chars inside token");
        assert_eq!(multibyte_token.token, "./café");
        assert!(multibyte_token.is_input_start);
    }

    #[test]
    fn detects_is_input_start_correctly() {
        let start_col0 = extract_completion_context("/se", (0, 3)).expect("token at row 0 col 0");
        assert!(start_col0.is_input_start);

        let start_indented = extract_completion_context("   \t/se", (0, 7))
            .expect("token after leading whitespace on row 0");
        assert!(start_indented.is_input_start);

        let after_token =
            extract_completion_context("hi /se", (0, 6)).expect("token after prior word");
        assert!(!after_token.is_input_start);

        let row_one = extract_completion_context("\n/se", (1, 3)).expect("token on row 1");
        assert!(!row_one.is_input_start);

        let row_one_blank_above = extract_completion_context("   \n  /se", (1, 5))
            .expect("token on row 1 with blank row 0");
        assert!(!row_one_blank_above.is_input_start);
    }

    #[test]
    fn triggers_on_all_five_path_patterns() {
        let dir = TempDir::new().expect("create temp dir");
        let provider = PathProvider::new(dir.path().to_path_buf(), None);

        for pattern in [
            "./", "./foo", "../", "../foo", "/", "/usr", "~/", "~/doc", "src/", "src/main",
        ] {
            assert!(
                provider.matches_trigger(&make_ctx(pattern, false)),
                "expected PathProvider to trigger on {pattern:?}"
            );
        }

        for non_pattern in ["src", "hello", ".", "..", "~"] {
            assert!(
                !provider.matches_trigger(&make_ctx(non_pattern, true)),
                "expected PathProvider NOT to trigger on {non_pattern:?}"
            );
        }
    }

    #[test]
    fn returns_first_lexicographical_match_suffix() {
        let dir = TempDir::new().expect("create temp dir");
        for name in ["zebra.rs", "beta.rs", "alpine.rs", "alpha.rs"] {
            fs::write(dir.path().join(name), "").expect("create test file");
        }

        let completer = Completer::with_defaults(dir.path().to_path_buf(), None, vec![], vec![]);
        assert_eq!(
            completer.complete("./al", (0, 4)),
            Some("pha.rs".to_string())
        );
        assert_eq!(
            completer.complete("./", (0, 2)),
            Some("alpha.rs".to_string())
        );
    }

    #[test]
    fn appends_trailing_slash_for_directories_and_dir_symlinks() {
        let dir = TempDir::new().expect("create temp dir");
        let subdir = dir.path().join("dir_entry");
        let file = dir.path().join("file_entry");
        fs::create_dir(&subdir).expect("create subdir");
        fs::write(&file, "hello").expect("create file");

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&subdir, dir.path().join("link_dir"))
                .expect("create dir symlink");
            std::os::unix::fs::symlink(&file, dir.path().join("link_file"))
                .expect("create file symlink");
        }

        let completer = Completer::with_defaults(dir.path().to_path_buf(), None, vec![], vec![]);

        assert_eq!(
            completer.complete("./dir_", (0, 6)),
            Some("entry/".to_string())
        );
        assert_eq!(
            completer.complete("./file_", (0, 7)),
            Some("entry".to_string())
        );

        #[cfg(unix)]
        {
            assert_eq!(
                completer.complete("./link_d", (0, 8)),
                Some("ir/".to_string())
            );
            assert_eq!(
                completer.complete("./link_f", (0, 8)),
                Some("ile".to_string())
            );
        }
    }

    #[test]
    fn includes_dotfiles_like_normal_files() {
        let dir = TempDir::new().expect("create temp dir");
        fs::write(dir.path().join(".env"), "").expect("create .env");
        fs::create_dir(dir.path().join(".git")).expect("create .git dir");
        fs::write(dir.path().join("readme.md"), "").expect("create readme.md");

        let completer = Completer::with_defaults(dir.path().to_path_buf(), None, vec![], vec![]);

        assert_eq!(completer.complete("./.", (0, 3)), Some("env".to_string()));
        assert_eq!(completer.complete("./.g", (0, 4)), Some("it/".to_string()));
        assert_eq!(completer.complete("./", (0, 2)), Some(".env".to_string()));
    }

    #[test]
    fn resolves_relative_parent_tilde_and_absolute_paths() {
        let root = TempDir::new().expect("create root temp dir");
        let work = root.path().join("workspace");
        fs::create_dir(&work).expect("create workspace dir");
        fs::write(work.join("local.txt"), "").expect("create local.txt");
        fs::write(root.path().join("sibling.txt"), "").expect("create sibling.txt");

        let home = TempDir::new().expect("create home temp dir");
        fs::write(home.path().join("notes.md"), "").expect("create notes.md");

        let completer = Completer::with_defaults(
            work.clone(),
            Some(home.path().to_path_buf()),
            vec![],
            vec![],
        );

        assert_eq!(
            completer.complete("./loc", (0, 5)),
            Some("al.txt".to_string())
        );
        assert_eq!(
            completer.complete("../sib", (0, 6)),
            Some("ling.txt".to_string())
        );
        assert_eq!(
            completer.complete("~/not", (0, 5)),
            Some("es.md".to_string())
        );

        let abs_prefix = format!("{}/loc", work.display());
        let col = abs_prefix.chars().count();
        assert_eq!(
            completer.complete(&abs_prefix, (0, col)),
            Some("al.txt".to_string())
        );
    }

    #[test]
    fn returns_none_on_missing_or_unreadable_dir_and_exact_file_match() {
        let dir = TempDir::new().expect("create temp dir");
        fs::write(dir.path().join("exact.txt"), "").expect("create exact.txt");

        let completer = Completer::with_defaults(dir.path().to_path_buf(), None, vec![], vec![]);

        assert_eq!(completer.complete("./does_not_exist/foo", (0, 20)), None);
        assert_eq!(completer.complete("https://foo/bar", (0, 15)), None);
        assert_eq!(completer.complete("./exact.txt", (0, 11)), None);
    }

    #[test]
    fn completes_slash_commands_only_at_input_start() {
        let dir = TempDir::new().expect("create temp dir");
        let cmd_provider = SlashCommandProvider::new(vec!["sessions".to_string()]);
        assert!(cmd_provider.matches_trigger(&make_ctx("/se", true)));
        assert!(!cmd_provider.matches_trigger(&make_ctx("/se", false)));

        let completer = Completer::with_defaults(
            dir.path().to_path_buf(),
            None,
            vec!["sessions".to_string(), "compact".to_string()],
            vec![],
        );

        assert_eq!(
            completer.complete("/se", (0, 3)),
            Some("ssions".to_string())
        );
        assert_eq!(
            completer.complete("  /se", (0, 5)),
            Some("ssions".to_string())
        );
        assert_eq!(completer.complete("hello /se", (0, 9)), None);
    }

    #[test]
    fn completes_skills_at_start_and_mid_line() {
        let dir = TempDir::new().expect("create temp dir");
        let completer = Completer::with_defaults(
            dir.path().to_path_buf(),
            None,
            vec![],
            vec!["dg-test-plan".to_string()],
        );

        assert_eq!(
            completer.complete("/dg-t", (0, 5)),
            Some("est-plan".to_string())
        );
        assert_eq!(
            completer.complete("Good morning. /dg-t", (0, 19)),
            Some("est-plan".to_string())
        );
    }

    #[test]
    fn merges_commands_and_skills_lexicographically_in_tier_one() {
        let dir = TempDir::new().expect("create temp dir");
        let completer = Completer::with_defaults(
            dir.path().to_path_buf(),
            None,
            vec!["sessions".to_string(), "compact".to_string()],
            vec!["search-code".to_string(), "code-review".to_string()],
        );

        assert_eq!(
            completer.complete("/se", (0, 3)),
            Some("arch-code".to_string())
        );
        assert_eq!(
            completer.complete("/co", (0, 3)),
            Some("de-review".to_string())
        );
    }

    #[test]
    fn tier_one_takes_precedence_over_root_paths_and_falls_back() {
        let work = TempDir::new().expect("create work temp dir");
        let abs_dir = TempDir::new().expect("create abs temp dir");
        fs::write(abs_dir.path().join("fallback_target.txt"), "")
            .expect("create fallback_target.txt");

        let completer = Completer::with_defaults(
            work.path().to_path_buf(),
            None,
            vec!["sessions".to_string()],
            vec![],
        );

        assert_eq!(
            completer.complete("/s", (0, 2)),
            Some("essions".to_string())
        );

        let fallback_query = format!("{}/fallback_", abs_dir.path().display());
        let col = fallback_query.chars().count();
        assert_eq!(
            completer.complete(&fallback_query, (0, col)),
            Some("target.txt".to_string())
        );
    }

    struct FixedProvider {
        prefix: String,
        items: Vec<String>,
    }

    impl CompletionProvider for FixedProvider {
        fn matches_trigger(&self, ctx: &CompletionContext) -> bool {
            ctx.token.starts_with(&self.prefix)
        }

        fn candidates(&self, _ctx: &CompletionContext) -> Vec<String> {
            self.items.clone()
        }
    }

    #[test]
    fn custom_provider_can_be_registered() {
        let mut completer = Completer::new();
        completer.add_tier(vec![Box::new(FixedProvider {
            prefix: "@".to_string(),
            items: vec!["@zebra".to_string(), "@alpha".to_string()],
        })]);
        completer.add_tier(vec![Box::new(FixedProvider {
            prefix: "@".to_string(),
            items: vec!["@aaa_lower_tier".to_string()],
        })]);
        completer.register_provider(
            0,
            Box::new(FixedProvider {
                prefix: "@".to_string(),
                items: vec!["@aardvark".to_string()],
            }),
        );

        assert_eq!(
            completer.complete("ping @a", (0, 7)),
            Some("ardvark".to_string())
        );
        assert_eq!(
            completer.complete("ping @z", (0, 7)),
            Some("ebra".to_string())
        );
    }

    #[test]
    fn needs_completion_false_for_plain_prose() {
        let dir = TempDir::new().expect("create temp dir");
        let completer = Completer::with_defaults(
            dir.path().to_path_buf(),
            None,
            vec!["sessions".to_string()],
            vec!["dg-plan".to_string()],
        );

        assert!(!completer.needs_completion("hello world", (0, 11)));
        assert!(!completer.needs_completion("src", (0, 3)));
        assert!(!completer.needs_completion("", (0, 0)));
        assert!(!completer.needs_completion("hello ", (0, 6)));

        assert!(completer.needs_completion("/se", (0, 3)));
        assert!(completer.needs_completion("look ./foo", (0, 10)));
    }

    #[tokio::test]
    async fn worker_returns_matching_generation_and_suffix() {
        let dir = TempDir::new().expect("create temp dir");
        fs::write(dir.path().join("alpha.rs"), "").expect("create alpha.rs");

        let completer = Completer::with_defaults(dir.path().to_path_buf(), None, vec![], vec![]);
        let (worker, mut rx) = CompletionWorker::spawn(completer);

        assert!(!worker.request(1, "hello".to_string(), (0, 5)));
        assert!(worker.request(2, "./al".to_string(), (0, 4)));

        let result = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("worker response timed out")
            .expect("worker channel closed unexpectedly");

        assert_eq!(result.generation, 2);
        assert_eq!(result.suggestion, Some("pha.rs".to_string()));
    }

    struct GatedProvider {
        seen_tokens: Arc<Mutex<Vec<String>>>,
        started_tx: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
        release_rx: Mutex<Option<std::sync::mpsc::Receiver<()>>>,
    }

    impl CompletionProvider for GatedProvider {
        fn matches_trigger(&self, ctx: &CompletionContext) -> bool {
            ctx.token.starts_with('q')
        }

        fn candidates(&self, ctx: &CompletionContext) -> Vec<String> {
            self.seen_tokens
                .lock()
                .expect("lock seen_tokens")
                .push(ctx.token.clone());

            if let Some(tx) = self.started_tx.lock().expect("lock started_tx").take() {
                let _ = tx.send(());
            }

            if let Some(rx) = self.release_rx.lock().expect("lock release_rx").take() {
                let _ = rx.recv();
            }

            vec![format!("{}_done", ctx.token)]
        }
    }

    #[tokio::test]
    async fn worker_coalesces_rapid_requests_with_latest_generation() {
        let seen_tokens = Arc::new(Mutex::new(Vec::new()));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        let mut completer = Completer::new();
        completer.add_tier(vec![Box::new(GatedProvider {
            seen_tokens: Arc::clone(&seen_tokens),
            started_tx: Mutex::new(Some(started_tx)),
            release_rx: Mutex::new(Some(release_rx)),
        })]);

        let (worker, mut rx) = CompletionWorker::spawn(completer);

        assert!(worker.request(1, "q1".to_string(), (0, 2)));
        tokio::time::timeout(Duration::from_secs(5), started_rx)
            .await
            .expect("first request did not enter provider in time")
            .expect("started_tx dropped");

        assert!(worker.request(2, "q2".to_string(), (0, 2)));
        assert!(worker.request(3, "q3".to_string(), (0, 2)));
        assert!(worker.request(4, "q4".to_string(), (0, 2)));

        release_tx.send(()).expect("release gated provider");

        let mut received = Vec::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(Duration::from_millis(500), rx.recv()).await {
                Ok(Some(res)) => {
                    let generation = res.generation;
                    received.push(res);
                    if generation == 4 {
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => continue,
            }
        }

        let last = received
            .last()
            .expect("expected final completion result for generation 4");
        assert_eq!(last.generation, 4);
        assert_eq!(last.suggestion, Some("_done".to_string()));

        assert!(
            !received
                .iter()
                .any(|r| r.generation == 2 || r.generation == 3),
            "intermediate generations should be coalesced"
        );

        let tokens = seen_tokens.lock().expect("lock seen_tokens");
        assert!(
            !tokens.iter().any(|t| t == "q2" || t == "q3"),
            "intermediate generations should not execute: {tokens:?}"
        );
    }
}
