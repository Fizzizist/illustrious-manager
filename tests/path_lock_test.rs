use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use illustrious_manager::tools::path_lock::PathLocks;
use illustrious_manager::tools::run_blocking;

const BLOCKED: Duration = Duration::from_millis(50);
const GENEROUS: Duration = Duration::from_secs(5);

#[tokio::test]
async fn same_path_lock_waits_for_holder() {
    let locks = PathLocks::new();
    let path = Path::new("/sandbox/file.txt");

    let held = locks.lock(path).await;
    assert!(
        tokio::time::timeout(BLOCKED, locks.lock(path))
            .await
            .is_err(),
        "second lock on the same path must wait while the first guard is held"
    );

    drop(held);
    assert!(
        tokio::time::timeout(GENEROUS, locks.lock(path))
            .await
            .is_ok(),
        "lock must succeed once the holder drops its guard"
    );
}

#[tokio::test]
async fn different_paths_do_not_block() {
    let locks = PathLocks::new();

    let _held = locks.lock(Path::new("/sandbox/a.txt")).await;
    assert!(
        tokio::time::timeout(GENEROUS, locks.lock(Path::new("/sandbox/b.txt")))
            .await
            .is_ok(),
        "a lock on a different path must not wait for an unrelated holder"
    );
}

#[tokio::test]
async fn waiters_acquire_in_request_order() {
    let locks = PathLocks::new();
    let path = Path::new("/sandbox/file.txt");
    let order = std::sync::Mutex::new(Vec::new());

    let held = locks.lock(path).await;

    let waiters = futures::future::join_all((0..10).map(|i| {
        let locks = &locks;
        let order = &order;
        async move {
            let _guard = locks.lock(path).await;
            order.lock().expect("order mutex poisoned").push(i);
            tokio::task::yield_now().await;
        }
    }));
    let releaser = async move {
        tokio::task::yield_now().await;
        drop(held);
    };
    tokio::time::timeout(GENEROUS, async { tokio::join!(waiters, releaser) })
        .await
        .expect("waiters must all acquire after release");

    let order = order.into_inner().expect("order mutex poisoned");
    assert_eq!(order, (0..10).collect::<Vec<_>>());
}

#[tokio::test]
async fn guard_moved_into_blocking_task_holds_until_task_ends() {
    let locks = Arc::new(PathLocks::new());
    let path = Path::new("/sandbox/file.txt");
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();

    let guard = locks.lock(path).await;
    let blocking = run_blocking("test", move || {
        let _guard = guard;
        let _ = started_tx.send(());
        let _ = release_rx.recv();
        Ok(())
    });
    let _ = tokio::time::timeout(Duration::ZERO, blocking).await;
    tokio::time::timeout(GENEROUS, started_rx)
        .await
        .expect("blocking task should start")
        .expect("blocking task should signal start");

    assert!(
        tokio::time::timeout(BLOCKED, locks.lock(path))
            .await
            .is_err(),
        "dropping the awaiting future must not release a guard owned by the running blocking task"
    );

    release_tx
        .send(())
        .expect("blocking task should be waiting");
    assert!(
        tokio::time::timeout(GENEROUS, locks.lock(path))
            .await
            .is_ok(),
        "lock must succeed once the blocking task ends"
    );
}
