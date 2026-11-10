//! Per-item results of a job, stored beside the job row instead of inside its result (SW-7082).
//!
//! A job that produces one result object per input (the dataset batch describe job: six text
//! fields per image) cannot carry the whole list in its progress result. That result column is
//! replaced on every progress POST, the POST is capped by the API's JSON body limit, and every
//! accepted POST is fanned out over SSE as a `job.updated` event, so a cumulative list grows each
//! POST, eventually exceeds the body limit, and makes every tick cost O(items) (the hazard the
//! image lane hit with its accumulated `assets`, sc-8953). Instead the owning worker appends each
//! item here as it finishes (a bounded write), its progress POSTs carry only counts, and the UI
//! pages the ordered items back out with a byte budget per page.
//!
//! Rows are keyed `(job_id, item_id)` so a re-report of the same item replaces it, ordered by the
//! worker's `position`, and removed with their job by the retention purge.

use super::*;

/// Largest JSON encoding of one stored item. A worker's item is bounded well below this by
/// construction (the batch describe job caps each field); the store refuses anything larger, so
/// one item can never make an append or a page exceed the API's request limits.
pub const MAX_JOB_RESULT_ITEM_BYTES: usize = 1024 * 1024;

/// Most items one append may carry (with the per-item cap, an append stays under the API's
/// 10 MiB JSON body limit only when it carries few items; the batch describe worker sends one).
pub const MAX_JOB_RESULT_ITEMS_PER_APPEND: usize = 8;

/// Byte budget of one page read: a page stops before the item that would take it past this, but
/// always holds at least one item, so a page is at most this plus one item.
pub const JOB_RESULT_ITEMS_PAGE_BYTES: usize = 4 * 1024 * 1024;

/// Most items one page read returns.
pub const MAX_JOB_RESULT_ITEMS_PAGE_LIMIT: usize = 500;

/// One item a worker appends: its order among the job's items and its JSON object (which must
/// carry a non-empty string `itemId`).
#[derive(Debug, Clone, PartialEq)]
pub struct JobResultItemWrite {
    pub position: u64,
    pub item: Map<String, Value>,
}

/// One page of a job's items, in `position` order.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobResultItemsPage {
    pub items: Vec<Value>,
    /// How many items the job has stored in all.
    pub total: usize,
    /// The `offset` of the next page, or `None` when this page reached the end.
    pub next_offset: Option<usize>,
}

pub(super) const RESULT_ITEMS_SCHEMA: &str = "
    create table if not exists job_result_items (
      job_id text not null,
      item_id text not null,
      position integer not null,
      item_json text not null,
      updated_at text not null,
      primary key (job_id, item_id)
    );
    create index if not exists idx_job_result_items_order
      on job_result_items(job_id, position, item_id);
";

/// Why an append was refused: a request the caller can fix (400), not a store failure.
fn invalid_item(message: impl Into<String>) -> JobsStoreError {
    JobsStoreError::InvalidResultItem(message.into())
}

impl JobsStore {
    /// Append (or replace, by item id) items of a job from the worker that owns it. Refused when
    /// the job is terminal or owned by another worker (the same zombie-write guard as progress,
    /// sc-4172), when an item has no `itemId`, or when one item's JSON exceeds
    /// [`MAX_JOB_RESULT_ITEM_BYTES`]. Returns how many items the job now has.
    pub fn append_job_result_items(
        &self,
        job_id: &str,
        worker_id: Option<&str>,
        items: &[JobResultItemWrite],
    ) -> JobsStoreResult<usize> {
        if items.len() > MAX_JOB_RESULT_ITEMS_PER_APPEND {
            return Err(invalid_item(format!(
                "At most {MAX_JOB_RESULT_ITEMS_PER_APPEND} result items per request."
            )));
        }
        let mut rows = Vec::with_capacity(items.len());
        for write in items {
            let item_id = write
                .item
                .get("itemId")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| invalid_item("Every result item needs a string itemId."))?
                .to_owned();
            let encoded = serde_json::to_string(&write.item)?;
            if encoded.len() > MAX_JOB_RESULT_ITEM_BYTES {
                return Err(invalid_item(format!(
                    "Result item {item_id:?} is {} bytes; the limit is {MAX_JOB_RESULT_ITEM_BYTES}.",
                    encoded.len()
                )));
            }
            let position = i64::try_from(write.position)
                .map_err(|_| invalid_item("Result item position is out of range."))?;
            rows.push((item_id, position, encoded));
        }
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = self.get_job_on_connection(&transaction, job_id)?;
        if is_terminal_status(current.status.as_str()) {
            return Err(JobsStoreError::TerminalJobImmutable {
                job_id: job_id.to_owned(),
                status: current.status.as_str().to_owned(),
            });
        }
        match (worker_id, current.worker_id.as_deref()) {
            (Some(reporter), Some(owner)) if reporter == owner => {}
            _ => {
                return Err(JobsStoreError::NotJobOwner {
                    job_id: job_id.to_owned(),
                })
            }
        }
        let now = utc_now();
        for (item_id, position, encoded) in &rows {
            transaction.execute(
                "insert into job_result_items (job_id, item_id, position, item_json, updated_at)
                 values (?1, ?2, ?3, ?4, ?5)
                 on conflict(job_id, item_id) do update set
                   position = excluded.position,
                   item_json = excluded.item_json,
                   updated_at = excluded.updated_at",
                params![job_id, item_id, position, encoded, now],
            )?;
        }
        let total = count_result_items(&transaction, job_id)?;
        transaction.commit()?;
        Ok(total)
    }

    /// Read a page of a job's items in `position` order, starting at `offset`, holding at most
    /// `limit` (clamped to 1..=[`MAX_JOB_RESULT_ITEMS_PAGE_LIMIT`]) items and at most
    /// [`JOB_RESULT_ITEMS_PAGE_BYTES`] of item JSON beyond its first item. An unknown job is
    /// `NotFound`; a job with no items yet returns an empty page.
    pub fn list_job_result_items(
        &self,
        job_id: &str,
        offset: usize,
        limit: usize,
    ) -> JobsStoreResult<JobResultItemsPage> {
        let limit = limit.clamp(1, MAX_JOB_RESULT_ITEMS_PAGE_LIMIT);
        let connection = self.open_connection()?;
        // Existence check: a missing job is a 404, not an empty list.
        self.get_job_on_connection(&connection, job_id)?;
        let total = count_result_items(&connection, job_id)?;
        let mut statement = connection.prepare(
            "select item_json from job_result_items
              where job_id = ?1
              order by position, item_id
              limit ?2 offset ?3",
        )?;
        let offset_sql = i64::try_from(offset).unwrap_or(i64::MAX);
        let mut rows = statement.query(params![job_id, limit as i64, offset_sql])?;
        let mut items = Vec::new();
        let mut bytes = 0usize;
        while let Some(row) = rows.next()? {
            let encoded: String = row.get(0)?;
            if !items.is_empty() && bytes + encoded.len() > JOB_RESULT_ITEMS_PAGE_BYTES {
                break;
            }
            bytes += encoded.len();
            items.push(serde_json::from_str(&encoded)?);
        }
        let end = offset.saturating_add(items.len());
        Ok(JobResultItemsPage {
            items,
            total,
            next_offset: (end < total).then_some(end),
        })
    }
}

fn count_result_items(connection: &Connection, job_id: &str) -> JobsStoreResult<usize> {
    let count: i64 = connection.query_row(
        "select count(*) from job_result_items where job_id = ?1",
        params![job_id],
        |row| row.get(0),
    )?;
    Ok(usize::try_from(count).unwrap_or(0))
}
