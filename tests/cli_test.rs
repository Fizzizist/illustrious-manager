use std::process::Command;

fn bin_path() -> std::path::PathBuf {
    let mut path = std::env::current_exe().expect("current_exe");
    assert!(path.pop(), "exe parent");
    if path.ends_with("deps") {
        assert!(path.pop(), "deps parent");
    }
    path.join("illustrious-manager")
}

struct Sandbox {
    _dir: tempfile::TempDir,
    config_path: std::path::PathBuf,
    sessions_dir: std::path::PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let config_path = dir.path().join("config.toml");
        std::fs::write(
            &config_path,
            "[vertex]\nproject = \"test-project\"\nregion = \"us-east5\"\nmodel = \"claude-sonnet-4-20250514\"\n",
        )
        .expect("write config");
        let sessions_dir = dir.path().join("sessions");
        Self {
            _dir: dir,
            config_path,
            sessions_dir,
        }
    }

    fn run(&self, args: &[&str]) -> std::process::Output {
        Command::new(bin_path())
            .arg("--config")
            .arg(&self.config_path)
            .args(args)
            .env_remove("HOME")
            .output()
            .expect("spawn binary")
    }
}

/// Acceptance: a typo'd role must fail fast — before any session DB is
/// created — with an error that names the role and lists the alternatives.
#[test]
fn unknown_role_fails_without_creating_session_db() {
    let sandbox = Sandbox::new();
    let output = sandbox.run(&[
        "--role",
        "nonexistent",
        "--session-id",
        "01923456-7890-7abc-def0-123456789abc",
    ]);

    assert!(
        !output.status.success(),
        "unknown role must exit non-zero: stderr={:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("nonexistent"),
        "stderr should name the unknown role: {stderr}"
    );
    assert!(
        stderr.contains("Available roles: default"),
        "stderr should list available roles: {stderr}"
    );
    assert!(
        !sandbox
            .sessions_dir
            .join("01923456-7890-7abc-def0-123456789abc.db")
            .exists(),
        "no session DB may be created for a rejected role"
    );
}
