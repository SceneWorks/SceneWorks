//! App-owned lifetime management for background Dataset Catalog scans.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

type CatalogScanFuture = Pin<Box<dyn Future<Output = ()> + Send>>;
type CatalogScanTaskFn = Box<dyn FnOnce(CancellationToken) -> CatalogScanFuture + Send + 'static>;

struct CatalogScanTask {
    generation: u64,
    cancellation: CancellationToken,
    handle: JoinHandle<()>,
    restart: Option<CatalogScanTaskFn>,
}

#[derive(Default)]
struct SupervisorState {
    shutting_down: bool,
    tasks: HashMap<String, CatalogScanTask>,
}

/// Owns every catalog scan started by this API process.
///
/// Catalog ids are single-flight keys. Shutdown first closes the registry to
/// new work, then cooperatively cancels and drains all registered tasks.
struct CatalogScanSupervisorInner {
    state: Mutex<SupervisorState>,
    next_generation: AtomicU64,
    #[cfg(test)]
    wrapper_cleanup_barriers:
        parking_lot::Mutex<Option<(Arc<tokio::sync::Barrier>, Arc<tokio::sync::Barrier>)>>,
}

pub(crate) struct CatalogScanSupervisor {
    inner: Arc<CatalogScanSupervisorInner>,
}

impl Default for CatalogScanSupervisor {
    fn default() -> Self {
        Self {
            inner: Arc::new(CatalogScanSupervisorInner {
                state: Mutex::new(SupervisorState::default()),
                next_generation: AtomicU64::new(1),
                #[cfg(test)]
                wrapper_cleanup_barriers: parking_lot::Mutex::new(None),
            }),
        }
    }
}

impl CatalogScanSupervisor {
    pub(crate) async fn spawn<F, Fut>(&self, catalog_id: String, task: F) -> CatalogScanSpawn
    where
        F: FnOnce(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let task: CatalogScanTaskFn = Box::new(move |cancellation| Box::pin(task(cancellation)));
        let mut state = self.inner.state.lock().await;
        if state.shutting_down {
            return CatalogScanSpawn::ShuttingDown;
        }
        if let Some(entry) = state.tasks.get_mut(&catalog_id) {
            if !entry.handle.is_finished() {
                // A lifecycle request may commit desired=running while the
                // current generation is publishing a terminal paused/completed
                // state but before its wrapper has reaped itself. Retain one
                // successor under the same single-flight key so that request
                // cannot be lost in the teardown window. Repeated requests
                // coalesce to the newest equivalent restart.
                entry.restart = Some(task);
                return CatalogScanSpawn::RestartQueued;
            }
        }
        state.tasks.remove(&catalog_id);
        Self::spawn_boxed_locked(&self.inner, &mut state, catalog_id, task);
        CatalogScanSpawn::Started
    }

    fn spawn_boxed_locked(
        inner: &Arc<CatalogScanSupervisorInner>,
        state: &mut SupervisorState,
        catalog_id: String,
        task: CatalogScanTaskFn,
    ) {
        let generation = inner.next_generation.fetch_add(1, Ordering::Relaxed);
        let cancellation = CancellationToken::new();
        let task_cancellation = cancellation.clone();
        let weak_inner = Arc::downgrade(inner);
        let task_catalog_id = catalog_id.clone();
        let handle = tokio::spawn(async move {
            task(task_cancellation).await;
            let Some(inner) = weak_inner.upgrade() else {
                return;
            };
            #[cfg(test)]
            let cleanup_barriers = inner.wrapper_cleanup_barriers.lock().take();
            #[cfg(test)]
            if let Some((reached, release)) = cleanup_barriers {
                reached.wait().await;
                release.wait().await;
            }
            let mut state = inner.state.lock().await;
            let matching = state
                .tasks
                .get(&task_catalog_id)
                .is_some_and(|entry| entry.generation == generation);
            if !matching {
                return;
            }
            let completed = state
                .tasks
                .remove(&task_catalog_id)
                .expect("matching generation remains registered");
            if !state.shutting_down {
                if let Some(restart) = completed.restart {
                    Self::spawn_boxed_locked(&inner, &mut state, task_catalog_id, restart);
                }
            }
        });
        state.tasks.insert(
            catalog_id,
            CatalogScanTask {
                generation,
                cancellation,
                handle,
                restart: None,
            },
        );
    }

    /// Cancels every registered task and does not return until each async
    /// wrapper—and therefore every awaited `spawn_blocking` scan pass—has
    /// finished. Tokio cannot safely kill blocking work, so a timeout here would
    /// falsely report quiescence while a detached scanner still held its lease.
    pub(crate) async fn shutdown(&self) -> CatalogScanShutdown {
        let tasks = {
            let mut state = self.inner.state.lock().await;
            state.shutting_down = true;
            let tasks = state
                .tasks
                .drain()
                .map(|(_, task)| task)
                .collect::<Vec<_>>();
            for task in &tasks {
                task.cancellation.cancel();
            }
            tasks
        };
        let requested = tasks.len();
        let mut joined = 0;
        for task in tasks {
            if task.handle.await.is_ok() {
                joined += 1;
            }
        }
        CatalogScanShutdown { requested, joined }
    }

    #[cfg(test)]
    pub(crate) async fn active_count(&self) -> usize {
        self.inner
            .state
            .lock()
            .await
            .tasks
            .values()
            .filter(|entry| !entry.handle.is_finished())
            .count()
    }

    #[cfg(test)]
    pub(crate) async fn tracked_count(&self) -> usize {
        self.inner.state.lock().await.tasks.len()
    }

    #[cfg(test)]
    pub(crate) fn install_wrapper_cleanup_barriers(
        &self,
    ) -> (Arc<tokio::sync::Barrier>, Arc<tokio::sync::Barrier>) {
        let reached = Arc::new(tokio::sync::Barrier::new(2));
        let release = Arc::new(tokio::sync::Barrier::new(2));
        *self.inner.wrapper_cleanup_barriers.lock() =
            Some((Arc::clone(&reached), Arc::clone(&release)));
        (reached, release)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CatalogScanSpawn {
    Started,
    RestartQueued,
    ShuttingDown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CatalogScanShutdown {
    pub(crate) requested: usize,
    pub(crate) joined: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn catalog_ids_are_single_flight_and_shutdown_closes_the_registry() {
        let supervisor = Arc::new(CatalogScanSupervisor::default());
        let started = Arc::new(tokio::sync::Notify::new());
        let started_for_task = started.clone();
        assert_eq!(
            supervisor
                .spawn("catalog-a".to_owned(), move |cancellation| async move {
                    started_for_task.notify_one();
                    cancellation.cancelled().await;
                })
                .await,
            CatalogScanSpawn::Started
        );
        started.notified().await;
        assert_eq!(
            supervisor.spawn("catalog-a".to_owned(), |_| async {}).await,
            CatalogScanSpawn::RestartQueued
        );

        let result = supervisor.shutdown().await;
        assert_eq!(result.requested, 1);
        assert_eq!(result.joined, 1);
        assert_eq!(supervisor.active_count().await, 0);
        assert_eq!(
            supervisor.spawn("catalog-b".to_owned(), |_| async {}).await,
            CatalogScanSpawn::ShuttingDown
        );
    }

    #[tokio::test]
    async fn shutdown_waits_for_non_cooperative_blocking_work_to_finish() {
        let supervisor = Arc::new(CatalogScanSupervisor::default());
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let mutations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let task_mutations = mutations.clone();
        assert_eq!(
            supervisor
                .spawn("catalog-blocked".to_owned(), move |_| async move {
                    tokio::task::spawn_blocking(move || {
                        let _ = started_tx.send(());
                        release_rx.recv().expect("test releases blocking pass");
                        task_mutations.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    })
                    .await
                    .expect("blocking pass joins");
                })
                .await,
            CatalogScanSpawn::Started
        );
        started_rx.await.expect("blocking pass starts");

        let shutdown_supervisor = supervisor.clone();
        let mut shutdown = tokio::spawn(async move { shutdown_supervisor.shutdown().await });
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut shutdown)
                .await
                .is_err(),
            "shutdown must not detach a still-running blocking scanner"
        );
        release_tx.send(()).expect("blocking pass releases");
        let result = shutdown.await.expect("shutdown task joins");
        assert_eq!(result.requested, 1);
        assert_eq!(result.joined, 1);
        assert_eq!(mutations.load(std::sync::atomic::Ordering::SeqCst), 1);
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert_eq!(
            mutations.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "nothing mutates after shutdown returns"
        );
    }

    #[tokio::test]
    async fn distinct_completed_catalogs_are_reaped_from_the_registry() {
        let supervisor = CatalogScanSupervisor::default();
        for index in 0..200 {
            assert_eq!(
                supervisor
                    .spawn(format!("catalog-{index}"), |_| async {})
                    .await,
                CatalogScanSpawn::Started
            );
        }
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if supervisor.tracked_count().await == 0 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("completion reaping is prompt and bounded");
        assert_eq!(supervisor.active_count().await, 0);
    }

    #[tokio::test]
    async fn request_during_terminal_teardown_hands_off_to_one_successor() {
        let supervisor = Arc::new(CatalogScanSupervisor::default());
        let terminal = Arc::new(tokio::sync::Barrier::new(2));
        let release = Arc::new(tokio::sync::Barrier::new(2));
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        assert_eq!(
            supervisor
                .spawn("catalog-race".to_owned(), {
                    let terminal = terminal.clone();
                    let release = release.clone();
                    let runs = runs.clone();
                    move |_| async move {
                        runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        terminal.wait().await;
                        release.wait().await;
                    }
                })
                .await,
            CatalogScanSpawn::Started
        );
        terminal.wait().await;
        for _ in 0..3 {
            assert_eq!(
                supervisor
                    .spawn("catalog-race".to_owned(), {
                        let runs = runs.clone();
                        move |_| async move {
                            runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        }
                    })
                    .await,
                CatalogScanSpawn::RestartQueued
            );
        }
        release.wait().await;
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if supervisor.tracked_count().await == 0 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("successor completes");
        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "restart requests coalesce to exactly one successor"
        );
    }
}
