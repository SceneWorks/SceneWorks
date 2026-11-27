//! Jobs database file upkeep owned by the API process: one-time startup compaction
//! and WAL truncation, and the periodic WAL checkpoint task.

use super::*;

/// Startup upkeep of the jobs DB file, after retention and recovery have released
/// whatever rows they release: compact a heavily fragmented file once
/// ([`CompactionPolicy::STARTUP`]), then truncate the WAL so the server starts from
/// an empty one. Both are physical-only and best effort — a failure is logged and
/// startup continues on the uncompacted file.
pub(crate) fn compact_and_checkpoint_jobs_db(jobs_store: &JobsStore) {
    match jobs_store.compact_if_fragmented(sceneworks_core::jobs_store::CompactionPolicy::STARTUP) {
        Ok(report) if report.vacuumed => tracing::info!(
            event = "jobs_db_compacted",
            page_size = report.page_size,
            page_count_before = report.page_count_before,
            freelist_count_before = report.freelist_count_before,
            page_count_after = report.page_count_after,
            elapsed_ms = report.elapsed.as_millis() as u64,
            "compacted the jobs database"
        ),
        Ok(_) => {}
        Err(error) => tracing::warn!(
            event = "jobs_db_compaction_failed",
            error = %error,
            "could not compact the jobs database; continuing with the existing file"
        ),
    }
    if let Err(error) = jobs_store.checkpoint_wal_truncate(Duration::from_secs(5)) {
        tracing::warn!(
            event = "jobs_db_checkpoint_failed",
            error = %error,
            "could not checkpoint the jobs database WAL at startup"
        );
    }
}

/// How often the running API truncates the jobs WAL. Checkpoints are cheap when
/// the WAL is small, and a blocked one (a long reader) simply retries next tick.
const JOBS_WAL_CHECKPOINT_INTERVAL: Duration = Duration::from_secs(60);
/// A periodic checkpoint never waits long for readers: it runs on its own
/// connection and would otherwise hold SQLite's writer lock while it waits.
const JOBS_WAL_CHECKPOINT_BUSY_TIMEOUT: Duration = Duration::from_millis(100);

/// Truncate the jobs WAL periodically so a burst of writes (a large claim
/// rewrite, a bulk cancel, a retention purge) does not leave a high-water WAL
/// behind for the life of the process.
pub(crate) fn spawn_jobs_wal_checkpoints(state: AppState) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticks = tokio::time::interval(JOBS_WAL_CHECKPOINT_INTERVAL);
        ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticks.tick().await;
        loop {
            ticks.tick().await;
            let result = store_call(state.clone(), |store, _timeout| {
                store.checkpoint_wal_truncate(JOBS_WAL_CHECKPOINT_BUSY_TIMEOUT)
            })
            .await;
            match result {
                Ok(checkpoint) if checkpoint.busy => tracing::debug!(
                    event = "jobs_db_checkpoint_busy",
                    log_frames = checkpoint.log_frames,
                    "jobs WAL checkpoint deferred by an active reader"
                ),
                Ok(_) => {}
                Err(error) => tracing::warn!(
                    event = "jobs_db_checkpoint_failed",
                    error = %error.detail,
                    "periodic jobs WAL checkpoint failed"
                ),
            }
        }
    })
}
