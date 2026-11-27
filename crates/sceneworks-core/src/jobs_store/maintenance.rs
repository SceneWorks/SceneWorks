//! Physical upkeep of the jobs database file: WAL checkpoints and one-shot
//! startup compaction. Nothing here changes a row; it only bounds how much disk
//! the file and its WAL sidecar keep and how much I/O every query pays for it.

use std::time::{Duration, Instant};

use super::{JobsStore, JobsStoreResult};

/// Result of a `wal_checkpoint(TRUNCATE)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalCheckpoint {
    /// A reader (or a competing writer) kept the checkpoint from completing; the
    /// WAL keeps its unflushed frames and the next attempt picks them up.
    pub busy: bool,
    /// Frames in the WAL when the checkpoint ran (-1 when the file is not in WAL mode).
    pub log_frames: i64,
    /// Frames copied back into the database file.
    pub checkpointed_frames: i64,
}

/// When startup compaction (`VACUUM`) is worth its one-time cost.
///
/// VACUUM rewrites every live page, so its cost tracks the LIVE size, not the file
/// size: on a copy of a real 930 MB jobs.db whose 598 MB of offline-worker claim
/// blobs had been released (62% of pages free), it took ~10 s and left a 339 MB
/// file. Requiring a quarter of the file AND 64 MiB to be free keeps it from
/// running on ordinary retention churn, and the live-size cap bounds the worst
/// case to roughly a minute of readiness-critical startup at that rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactionPolicy {
    pub min_free_fraction: f64,
    pub min_free_bytes: u64,
    pub max_live_bytes: u64,
}

impl CompactionPolicy {
    pub const STARTUP: Self = Self {
        min_free_fraction: 0.25,
        min_free_bytes: 64 * 1024 * 1024,
        max_live_bytes: 2 * 1024 * 1024 * 1024,
    };

    /// Whether a file of `page_count` pages of `page_size` bytes with
    /// `freelist_count` free pages should be compacted under this policy.
    pub fn compaction_due(&self, page_size: u64, page_count: u64, freelist_count: u64) -> bool {
        if page_count == 0 || freelist_count == 0 {
            return false;
        }
        let free_bytes = freelist_count.saturating_mul(page_size);
        let live_bytes = page_count
            .saturating_sub(freelist_count)
            .saturating_mul(page_size);
        free_bytes >= self.min_free_bytes
            && (freelist_count as f64) / (page_count as f64) >= self.min_free_fraction
            && live_bytes <= self.max_live_bytes
    }
}

/// What [`JobsStore::compact_if_fragmented`] found and did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactionReport {
    pub page_size: u64,
    pub page_count_before: u64,
    pub freelist_count_before: u64,
    pub page_count_after: u64,
    pub vacuumed: bool,
    pub elapsed: Duration,
}

impl JobsStore {
    /// Copy every committed WAL frame back into the database and truncate the WAL
    /// file to zero bytes.
    ///
    /// Runs on its own connection with `busy_timeout`, not on the store's write
    /// connection, so the periodic caller never holds the store's write mutex: a
    /// checkpoint blocked by a long reader gives up after `busy_timeout` and
    /// reports `busy` instead of stalling every queued writer behind it. While the
    /// checkpoint itself runs it holds SQLite's writer lock, which in-process
    /// writers wait out through their normal busy handler.
    pub fn checkpoint_wal_truncate(
        &self,
        busy_timeout: Duration,
    ) -> JobsStoreResult<WalCheckpoint> {
        let connection = self.open_connection()?;
        connection.busy_timeout(busy_timeout)?;
        let result = connection.query_row("pragma wal_checkpoint(truncate)", [], |row| {
            Ok(WalCheckpoint {
                busy: row.get::<_, i64>(0)? != 0,
                log_frames: row.get(1)?,
                checkpointed_frames: row.get(2)?,
            })
        });
        match result {
            Ok(checkpoint) => Ok(checkpoint),
            // A blocked TRUNCATE can also surface as SQLITE_BUSY rather than as
            // the row's busy flag; either way nothing was lost.
            Err(rusqlite::Error::SqliteFailure(error, _))
                if error.code == rusqlite::ErrorCode::DatabaseBusy =>
            {
                Ok(WalCheckpoint {
                    busy: true,
                    log_frames: -1,
                    checkpointed_frames: -1,
                })
            }
            Err(error) => Err(error.into()),
        }
    }

    /// One-shot `VACUUM` when `policy` says the freelist is worth reclaiming.
    ///
    /// Startup-only: the API is the sole process that opens jobs.db (workers talk to
    /// it over HTTP), and this runs before the router serves requests, so holding
    /// the write mutex for the rewrite blocks nothing. In WAL mode the rewrite goes
    /// through the WAL, so callers follow it with [`Self::checkpoint_wal_truncate`].
    pub fn compact_if_fragmented(
        &self,
        policy: CompactionPolicy,
    ) -> JobsStoreResult<CompactionReport> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let pragma_u64 = |name: &str| -> JobsStoreResult<u64> {
            let value: i64 = connection.pragma_query_value(None, name, |row| row.get(0))?;
            Ok(u64::try_from(value).unwrap_or(0))
        };
        let page_size = pragma_u64("page_size")?;
        let page_count_before = pragma_u64("page_count")?;
        let freelist_count_before = pragma_u64("freelist_count")?;
        let started = Instant::now();
        let vacuumed = policy.compaction_due(page_size, page_count_before, freelist_count_before);
        if vacuumed {
            connection.execute_batch("vacuum")?;
        }
        let page_count_after = pragma_u64("page_count")?;
        Ok(CompactionReport {
            page_size,
            page_count_before,
            freelist_count_before,
            page_count_after,
            vacuumed,
            elapsed: started.elapsed(),
        })
    }
}
