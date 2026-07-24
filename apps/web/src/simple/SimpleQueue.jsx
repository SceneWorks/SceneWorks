import React, { useMemo } from "react";
import { useAppContext } from "../context/AppContext.js";
import { errorStatuses, terminalStatuses } from "../jobTypes.js";
import { percent } from "../formatting.js";

// Simple Queue (design handoff): three stat tiles (Running / Queued / Done) over a list
// of job rows — status dot, title, sub-line, status badge, and a progress bar.
//
// Same live `jobs` feed the advanced Queue reads (SSE-updated through useJobEvents), so
// the two views never disagree; what Simple drops is the per-worker resource panel and
// the retry/cancel/clear controls, which stay in the full workspace.

// Statuses the design's three buckets map onto. "Running" is anything a worker has
// claimed and not finished; "Queued" is the pending band; "Done" is terminal-and-clean.
export function bucketFor(job) {
  if (terminalStatuses.has(job.status)) {
    return errorStatuses.has(job.status) ? "failed" : "done";
  }
  return job.status === "queued" || job.status === "pending_caption" ? "queued" : "running";
}

const BADGE_LABEL = {
  running: "Running",
  queued: "Queued",
  done: "Done",
  failed: "Failed",
};

export function SimpleQueue() {
  const { jobs = [] } = useAppContext();

  const rows = useMemo(
    () =>
      jobs.map((job) => ({
        id: job.id,
        bucket: bucketFor(job),
        title: jobTitle(job),
        sub: jobSub(job),
        progress: Number(job.progress),
        status: job.status,
      })),
    [jobs],
  );

  const counts = useMemo(
    () => ({
      running: rows.filter((row) => row.bucket === "running").length,
      queued: rows.filter((row) => row.bucket === "queued").length,
      done: rows.filter((row) => row.bucket === "done").length,
    }),
    [rows],
  );

  return (
    <div className="su-screen su-screen--tight">
      <div className="su-stats">
        <div className="su-stat su-stat--running">
          <div className="su-stat-value">{counts.running}</div>
          <div className="su-stat-label">Running</div>
        </div>
        <div className="su-stat">
          <div className="su-stat-value">{counts.queued}</div>
          <div className="su-stat-label">Queued</div>
        </div>
        <div className="su-stat su-stat--done">
          <div className="su-stat-value">{counts.done}</div>
          <div className="su-stat-label">Done</div>
        </div>
      </div>

      {rows.length ? (
        rows.map((row) => {
          const showBar = row.bucket !== "queued";
          const width = row.bucket === "done" ? "100%" : Number.isFinite(row.progress) ? percent(row.progress) : "0%";
          return (
            <div className="su-job" key={row.id}>
              <div className="su-job-head">
                <span aria-hidden="true" className={`su-dot su-dot--${row.bucket}`} />
                <span className="su-job-text">
                  <span className="su-job-title">{row.title}</span>
                  <span className="su-job-sub">{row.sub}</span>
                </span>
                <span className={`su-job-badge su-job-badge--${row.bucket}`}>
                  {row.bucket === "running" && Number.isFinite(row.progress) && row.progress > 0
                    ? percent(row.progress)
                    : BADGE_LABEL[row.bucket]}
                </span>
              </div>
              {showBar ? (
                <div className={row.bucket === "done" ? "su-bar su-bar--done" : "su-bar"}>
                  <span style={{ width }} />
                </div>
              ) : null}
            </div>
          );
        })
      ) : (
        <p className="su-empty">Nothing in the queue. Runs you start in a studio show up here.</p>
      )}
    </div>
  );
}

// The job's headline: the model it runs, falling back to the job type. Mirrors what the
// advanced Queue card leads with, minus the worker/GPU detail.
function jobTitle(job) {
  const model = job.payload?.modelName ?? job.payload?.model ?? job.payload?.modelId;
  return model ? String(model) : formatType(job.type);
}

function jobSub(job) {
  const parts = [formatType(job.type)];
  const width = job.payload?.width;
  const height = job.payload?.height;
  if (width && height) {
    parts.push(`${width}×${height}`);
  }
  const count = job.payload?.count;
  if (Number.isFinite(count) && count > 1) {
    parts.push(`${count} variations`);
  }
  const duration = job.payload?.duration ?? job.payload?.targetDurationSecs;
  if (Number.isFinite(duration)) {
    parts.push(`${duration}s`);
  }
  return parts.join(" · ");
}

function formatType(type) {
  return String(type ?? "job").replaceAll("_", " ");
}
