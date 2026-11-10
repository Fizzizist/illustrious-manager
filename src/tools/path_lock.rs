use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError};

use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

/// Per-path async mutexes so only one task at a time changes a given file.
///
/// Callers must pass a canonical key (see `SandboxPolicy::validate_write_path`).
#[derive(Debug, Default)]
pub struct PathLocks {
    locks: std::sync::Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>,
}

impl PathLocks {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn lock(&self, path: &Path) -> OwnedMutexGuard<()> {
        let entry = {
            let mut locks = self.locks.lock().unwrap_or_else(PoisonError::into_inner);
            Arc::clone(locks.entry(path.to_path_buf()).or_default())
        };
        entry.lock_owned().await
    }
}
