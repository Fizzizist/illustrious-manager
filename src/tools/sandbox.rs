use std::path::{Path, PathBuf};

pub(super) fn path_is_within(path: &Path, sandbox: &Path) -> bool {
    #[cfg(windows)]
    {
        use std::path::Component;
        let path_components: Vec<_> = path.components().collect();
        let sandbox_components: Vec<_> = sandbox.components().collect();
        if path_components.len() < sandbox_components.len() {
            return false;
        }
        path_components
            .iter()
            .zip(sandbox_components.iter())
            .all(|(p, s)| match (p, s) {
                (Component::Prefix(a), Component::Prefix(b)) => {
                    a.to_string().to_lowercase() == b.to_string().to_lowercase()
                }
                (Component::Normal(a), Component::Normal(b)) => {
                    a.to_string_lossy().to_lowercase() == b.to_string_lossy().to_lowercase()
                }
                _ => p == s,
            })
    }
    #[cfg(not(windows))]
    {
        path.starts_with(sandbox)
    }
}

/// Sandbox policy for validating file paths
///
/// Ensures all file operations stay within the configured sandbox root directory,
/// or within any registered extra root (see [`SandboxPolicy::with_extra_root`]).
/// Prevents directory traversal attacks and symlink-based sandbox escapes.
#[derive(Debug, Clone, PartialEq)]
pub struct SandboxPolicy {
    root: PathBuf,
    extra_roots: Vec<PathBuf>,
}

impl SandboxPolicy {
    /// Create a new sandbox policy with the specified root directory
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            extra_roots: Vec::new(),
        }
    }

    /// Register an additional root directory that paths may resolve within.
    ///
    /// Only affects validation of *absolute* paths: relative paths always
    /// resolve against the primary root, never against extra roots.
    pub fn with_extra_root(mut self, root: &Path) -> Self {
        self.extra_roots.push(root.to_path_buf());
        self
    }

    /// Canonicalize the primary root (fatal on failure) and every extra root
    /// (skipped, not fatal, if canonicalization fails — e.g. a root that
    /// doesn't exist on the current platform).
    fn canonical_roots(&self) -> Result<(PathBuf, Vec<PathBuf>), SandboxError> {
        let primary = self.root.canonicalize().map_err(|_| {
            SandboxError::InvalidPath(format!("Cannot canonicalize sandbox root: {:?}", self.root))
        })?;

        let extras = self
            .extra_roots
            .iter()
            .filter_map(|r| r.canonicalize().ok())
            .collect();

        Ok((primary, extras))
    }

    /// Validate that a path is within the sandbox
    ///
    /// Resolves symlinks and canonicalizes the path, then ensures it stays
    /// within the sandbox root directory or one of the registered extra roots.
    ///
    /// # Security
    /// Only validates existing paths. Non-existent paths are rejected to prevent
    /// symlink-based sandbox escapes via parent directory symlinks.
    pub fn validate_path(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };

        if !absolute.try_exists().map_err(|_| {
            SandboxError::InvalidPath(format!("Cannot check path existence: {:?}", path))
        })? {
            return Err(SandboxError::InvalidPath(format!(
                "Path does not exist: {:?}",
                path
            )));
        }

        let (sandbox_canonical, extra_canonicals) = self.canonical_roots()?;

        let canonical = absolute.canonicalize().map_err(|_| {
            SandboxError::InvalidPath(format!("Cannot canonicalize path: {:?}", path))
        })?;

        if !path_is_within(&canonical, &sandbox_canonical)
            && !extra_canonicals
                .iter()
                .any(|extra| path_is_within(&canonical, extra))
        {
            return Err(SandboxError::OutsideSandbox {
                path: canonical,
                sandbox: sandbox_canonical,
            });
        }

        Ok(canonical)
    }

    /// Validate a path for writing (file may not yet exist).
    ///
    /// Walks up to the nearest existing ancestor, canonicalizes it, and verifies
    /// the intended write location stays within the sandbox root or one of the
    /// registered extra roots.
    pub fn validate_write_path(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };

        let (sandbox_canonical, extra_canonicals) = self.canonical_roots()?;

        let mut existing_ancestor = absolute.clone();
        let mut pending: Vec<std::ffi::OsString> = vec![];

        loop {
            match existing_ancestor.try_exists() {
                Ok(true) => break,
                Ok(false) => {}
                Err(_) => {
                    return Err(SandboxError::InvalidPath(format!(
                        "Cannot check path existence: {:?}",
                        existing_ancestor
                    )));
                }
            }

            let component = existing_ancestor
                .file_name()
                .ok_or_else(|| {
                    SandboxError::InvalidPath(format!("Cannot resolve write path: {:?}", path))
                })?
                .to_os_string();

            if component == ".." {
                return Err(SandboxError::InvalidPath(
                    "Path traversal via '..' is not allowed".to_string(),
                ));
            }

            pending.push(component);

            existing_ancestor = existing_ancestor
                .parent()
                .ok_or_else(|| {
                    SandboxError::InvalidPath(format!("Cannot resolve write path: {:?}", path))
                })?
                .to_path_buf();
        }

        let canonical_ancestor = existing_ancestor.canonicalize().map_err(|_| {
            SandboxError::InvalidPath(format!(
                "Cannot canonicalize ancestor: {:?}",
                existing_ancestor
            ))
        })?;

        if !path_is_within(&canonical_ancestor, &sandbox_canonical)
            && !extra_canonicals
                .iter()
                .any(|extra| path_is_within(&canonical_ancestor, extra))
        {
            return Err(SandboxError::OutsideSandbox {
                path: canonical_ancestor,
                sandbox: sandbox_canonical,
            });
        }

        let mut result = canonical_ancestor;
        for component in pending.iter().rev() {
            result = result.join(component);
        }

        Ok(result)
    }

    /// Get the sandbox root directory
    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        Self::new(&root)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SandboxError {
    OutsideSandbox { path: PathBuf, sandbox: PathBuf },
    InvalidPath(String),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::OutsideSandbox { path, sandbox } => {
                write!(f, "Path {:?} is outside sandbox {:?}", path, sandbox)
            }
            SandboxError::InvalidPath(msg) => {
                write!(f, "Invalid path: {}", msg)
            }
        }
    }
}

impl std::error::Error for SandboxError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn path_inside_sandbox_is_allowed() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let test_file = temp_dir.path().join("test.txt");

        fs::write(&test_file, "test content").expect("Failed to create test file");

        let result = sandbox.validate_path(&test_file);
        assert!(result.is_ok(), "Path inside sandbox should be allowed");
        assert_eq!(result.unwrap(), test_file.canonicalize().unwrap());
    }

    #[test]
    fn path_outside_sandbox_is_rejected() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let outside_path = temp_dir
            .path()
            .ancestors()
            .nth(1)
            .expect("Temp dir should have parent")
            .join("outside.txt");

        fs::write(&outside_path, "test content").expect("Failed to create outside file");

        let result = sandbox.validate_path(&outside_path);
        match result {
            Err(SandboxError::OutsideSandbox { .. }) => {}
            Ok(_) => panic!("Path outside sandbox should be rejected"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn symlink_escaping_sandbox_is_rejected() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());

        let symlink_path = temp_dir.path().join("escape_link");
        let outside_path = temp_dir.path().parent().unwrap().join("target.txt");

        fs::write(&outside_path, "test content").expect("Failed to write file");

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside_path, &symlink_path)
                .expect("Failed to create symlink");
        }

        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_file(&outside_path, &symlink_path)
                .expect("Failed to create symlink");
        }

        let result = sandbox.validate_path(&symlink_path);
        match result {
            Err(SandboxError::OutsideSandbox { .. }) => {}
            Ok(_) => panic!("Symlink escaping sandbox should be rejected"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn relative_paths_resolved_correctly() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());

        let subdir = temp_dir.path().join("subdir");
        fs::create_dir(&subdir).expect("Failed to create subdir");

        let test_file = subdir.join("test.txt");
        fs::write(&test_file, "test content").expect("Failed to create test file");

        let relative_path = Path::new("subdir/test.txt");
        let result = sandbox.validate_path(relative_path);

        assert!(result.is_ok(), "Relative path should be resolved correctly");
        let canonical = result.unwrap();
        let canonical_temp = temp_dir
            .path()
            .canonicalize()
            .expect("canonicalize temp dir");
        assert!(canonical.starts_with(&canonical_temp));
        assert!(canonical.ends_with("subdir/test.txt"));
    }

    #[test]
    fn non_existent_path_is_rejected() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let nonexistent_file = temp_dir.path().join("nonexistent.txt");

        let result = sandbox.validate_path(&nonexistent_file);
        assert!(
            result.is_err(),
            "Non-existent path should be rejected for security reasons"
        );
    }

    #[test]
    fn default_sandbox_uses_current_directory() {
        let sandbox = SandboxPolicy::default();

        let result = sandbox.validate_path(Path::new("."));
        assert!(
            result.is_ok(),
            "Current directory should be in default sandbox"
        );
    }

    #[test]
    fn validate_write_path_allows_nonexistent_file_inside_sandbox() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let new_file = temp_dir.path().join("new_file.txt");

        let result = sandbox.validate_write_path(&new_file);
        assert!(
            result.is_ok(),
            "Non-existent path inside sandbox should be allowed for writing"
        );
        let canonical_temp = temp_dir
            .path()
            .canonicalize()
            .expect("canonicalize temp dir");
        let expected = canonical_temp.join("new_file.txt");
        assert_eq!(
            result.expect("validate_write_path should succeed for path inside sandbox"),
            expected
        );
    }

    #[test]
    fn validate_write_path_rejects_nonexistent_file_outside_sandbox() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());
        let parent = temp_dir
            .path()
            .parent()
            .expect("Temp dir should have parent");
        let outside_file = parent.join("outside_new.txt");

        let result = sandbox.validate_write_path(&outside_file);
        match result {
            Err(SandboxError::OutsideSandbox { .. }) => {}
            Ok(_) => panic!("Path outside sandbox should be rejected"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn validate_write_path_rejects_symlink_parent_escaping_sandbox() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let outside_dir = TempDir::new().expect("Failed to create outside dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());

        let symlink_dir = temp_dir.path().join("escape_dir");

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outside_dir.path(), &symlink_dir)
                .expect("Failed to create symlink");
        }
        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_dir(outside_dir.path(), &symlink_dir)
                .expect("Failed to create symlink");
        }

        let target = symlink_dir.join("file.txt");
        let result = sandbox.validate_write_path(&target);
        match result {
            Err(SandboxError::OutsideSandbox { .. }) => {}
            Ok(_) => panic!("Symlink escaping sandbox should be rejected"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn validate_write_path_rejects_dotdot_traversal() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let sandbox = SandboxPolicy::new(temp_dir.path());

        let traversal = temp_dir
            .path()
            .join("subdir")
            .join("..")
            .join("..")
            .join("escape.txt");
        let result = sandbox.validate_write_path(&traversal);
        assert!(result.is_err(), "Path traversal should be rejected");
    }

    mod extra_root_tests {
        use super::*;

        #[test]
        fn existing_file_in_extra_root_passes_validate_path() {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let extra = TempDir::new().expect("Failed to create extra root dir");
            let sandbox = SandboxPolicy::new(temp_dir.path()).with_extra_root(extra.path());

            let file = extra.path().join("in_extra.txt");
            fs::write(&file, "hi").expect("Failed to write file");

            let result = sandbox.validate_path(&file);
            assert!(
                result.is_ok(),
                "Existing file inside extra root should be allowed"
            );
            let expected = file.canonicalize().expect("canonicalize file");
            assert_eq!(result.expect("validate_path should succeed"), expected);
        }

        #[test]
        fn nonexistent_file_in_extra_root_passes_validate_write_path() {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let extra = TempDir::new().expect("Failed to create extra root dir");
            let sandbox = SandboxPolicy::new(temp_dir.path()).with_extra_root(extra.path());

            let file = extra.path().join("new_in_extra.txt");
            let result = sandbox.validate_write_path(&file);
            assert!(
                result.is_ok(),
                "Non-existent file inside extra root should be allowed for writing"
            );

            let canonical_extra = extra
                .path()
                .canonicalize()
                .expect("canonicalize extra root");
            let expected = canonical_extra.join("new_in_extra.txt");
            assert_eq!(
                result.expect("validate_write_path should succeed"),
                expected
            );
        }

        // /tmp-specific: on macOS /tmp is a symlink to /private/tmp, so the
        // returned path must be the fully-resolved canonical form, not the
        // symlinked input. Requires a real /tmp, hence unix-gated.
        #[cfg(unix)]
        #[test]
        fn returned_path_from_extra_root_is_canonical() {
            let extra = tempfile::tempdir_in("/tmp").expect("Failed to create extra root dir");
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let sandbox = SandboxPolicy::new(temp_dir.path()).with_extra_root(Path::new("/tmp"));

            let file = extra.path().join("canon.txt");
            fs::write(&file, "hi").expect("Failed to write file");

            let result = sandbox
                .validate_path(&file)
                .expect("validate_path should succeed");
            assert_eq!(result, file.canonicalize().expect("canonicalize file"));
        }

        #[test]
        fn path_outside_all_roots_is_rejected() {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let extra = TempDir::new().expect("Failed to create extra root dir");
            let sandbox = SandboxPolicy::new(temp_dir.path()).with_extra_root(extra.path());

            let outside_dir = TempDir::new().expect("Failed to create outside dir");
            let outside_file = outside_dir.path().join("outside.txt");
            fs::write(&outside_file, "content").expect("Failed to write outside file");

            let result = sandbox.validate_path(&outside_file);
            match result {
                Err(SandboxError::OutsideSandbox { .. }) => {}
                Ok(_) => panic!("Path outside all roots should be rejected"),
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        #[cfg(unix)]
        #[test]
        fn symlink_in_extra_root_escaping_all_roots_is_rejected() {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let extra = TempDir::new().expect("Failed to create extra root dir");
            let sandbox = SandboxPolicy::new(temp_dir.path()).with_extra_root(extra.path());

            let outside_dir = TempDir::new().expect("Failed to create outside dir");
            let outside_file = outside_dir.path().join("target.txt");
            fs::write(&outside_file, "content").expect("Failed to write file");

            let symlink_path = extra.path().join("escape_link");
            std::os::unix::fs::symlink(&outside_file, &symlink_path)
                .expect("Failed to create symlink");

            let result = sandbox.validate_path(&symlink_path);
            match result {
                Err(SandboxError::OutsideSandbox { .. }) => {}
                Ok(_) => panic!("Symlink escaping all roots should be rejected"),
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        #[test]
        fn dotdot_traversal_via_extra_root_is_rejected() {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let extra = TempDir::new().expect("Failed to create extra root dir");
            let sandbox = SandboxPolicy::new(temp_dir.path()).with_extra_root(extra.path());

            let traversal = extra
                .path()
                .join("subdir")
                .join("..")
                .join("..")
                .join("escape.txt");
            let result = sandbox.validate_write_path(&traversal);
            assert!(
                result.is_err(),
                "Path traversal via extra root should be rejected"
            );
        }

        #[cfg(unix)]
        #[test]
        fn validate_write_path_rejects_symlinked_ancestor_dir_in_extra_root_escaping_all_roots() {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let extra = TempDir::new().expect("Failed to create extra root dir");
            let outside_dir = TempDir::new().expect("Failed to create outside dir");
            let sandbox = SandboxPolicy::new(temp_dir.path()).with_extra_root(extra.path());

            let symlink_dir = extra.path().join("escape_dir");
            std::os::unix::fs::symlink(outside_dir.path(), &symlink_dir)
                .expect("Failed to create symlink");

            let target = symlink_dir.join("file.txt");
            let result = sandbox.validate_write_path(&target);
            match result {
                Err(SandboxError::OutsideSandbox { .. }) => {}
                Ok(_) => panic!(
                    "Write path through a symlinked ancestor dir inside an extra root, escaping \
                     all roots, should be rejected"
                ),
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        #[test]
        fn relative_path_does_not_resolve_against_extra_root() {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let extra = TempDir::new().expect("Failed to create extra root dir");
            let sandbox = SandboxPolicy::new(temp_dir.path()).with_extra_root(extra.path());

            let file_name = "relative_only_in_extra.txt";
            fs::write(extra.path().join(file_name), "hi")
                .expect("Failed to write file in extra root");

            let result = sandbox.validate_path(Path::new(file_name));
            assert!(
                result.is_err(),
                "Relative path must not resolve against extra roots"
            );
        }

        #[test]
        fn absent_extra_root_is_skipped_not_fatal() {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let missing =
                std::env::temp_dir().join("im-sandbox-test-nonexistent-root-xyz-doesnotexist");
            let sandbox = SandboxPolicy::new(temp_dir.path()).with_extra_root(&missing);

            let file = temp_dir.path().join("inside.txt");
            fs::write(&file, "hi").expect("Failed to write file");

            let result = sandbox.validate_path(&file);
            assert!(
                result.is_ok(),
                "A missing extra root should be skipped, not cause a fatal error"
            );
        }
    }
}
