//! App-owned lifetime management for background Dataset Catalog scans.

use std::collections::HashMap;
use std::future::Future;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

struct CatalogScanTask {
    cancellation: CancellationToken,
    handle: JoinHandle<()>,
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
pub(crate) struct CatalogScanSupervisor {
    state: Mutex<SupervisorState>,
}

impl Default for CatalogScanSupervisor {
    fn default() -> Self {
        Self {
            state: Mutex::new(SupervisorState::default()),
        }
    }
}

impl CatalogScanSupervisor {
    pub(crate) async fn spawn<F, Fut>(&self, catalog_id: String, task: F) -> bool
    where
        F: FnOnce(CancellationToken) -> Fut,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let mut state = self.state.lock().await;
        if state.shutting_down {
            return false;
        }
        if state
            .tasks
            .get(&catalog_id)
            .is_some_and(|entry| !entry.handle.is_finished())
        {
            return false;
        }
        state.tasks.remove(&catalog_id);

        let cancellation = CancellationToken::new();
        let handle = tokio::spawn(task(cancellation.clone()));
        state.tasks.insert(
            catalog_id,
            CatalogScanTask {
                cancellation,
                handle,
            },
        );
        true
    }

    pub(crate) async fn shutdown(&self, grace: Duration) -> CatalogScanShutdown {
        let tasks = {
            let mut state = self.state.lock().await;
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
        let mut handles = tasks
            .into_iter()
            .map(|task| task.handle)
            .collect::<Vec<_>>();
        let drain = async {
            for handle in &mut handles {
                if handle.await.is_ok() {
                    joined += 1;
                }
            }
        };
        let timed_out = tokio::time::timeout(grace, drain).await.is_err();
        if timed_out {
            for handle in handles {
                handle.abort();
            }
        }
        CatalogScanShutdown {
            requested,
            joined,
            timed_out,
        }
    }

    #[cfg(test)]
    pub(crate) async fn active_count(&self) -> usize {
        self.state
            .lock()
            .await
            .tasks
            .values()
            .filter(|entry| !entry.handle.is_finished())
            .count()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CatalogScanShutdown {
    pub(crate) requested: usize,
    pub(crate) joined: usize,
    pub(crate) timed_out: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn catalog_ids_are_single_flight_and_shutdown_closes_the_registry() {
        let supervisor = Arc::new(CatalogScanSupervisor::default());
        let started = Arc::new(tokio::sync::Notify::new());
        let started_for_task = started.clone();
        assert!(
            supervisor
                .spawn("catalog-a".to_owned(), move |cancellation| async move {
                    started_for_task.notify_one();
                    cancellation.cancelled().await;
                })
                .await
        );
        started.notified().await;
        assert!(
            !supervisor.spawn("catalog-a".to_owned(), |_| async {}).await,
            "the same catalog must not receive a second live task"
        );

        let result = supervisor.shutdown(Duration::from_secs(1)).await;
        assert_eq!(result.requested, 1);
        assert_eq!(result.joined, 1);
        assert!(!result.timed_out);
        assert_eq!(supervisor.active_count().await, 0);
        assert!(
            !supervisor.spawn("catalog-b".to_owned(), |_| async {}).await,
            "shutdown permanently closes the supervisor to new work"
        );
    }
}
