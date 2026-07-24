import React, { useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { errorStatuses, pendingStatuses, terminalStatuses } from "../jobTypes.js";
import { percent } from "../formatting.js";

// The Simple UI's one job card. Rendered in TWO places from this single definition:
//   - the Queue screen, as the list rows
//   - each studio, directly under Generate, for that studio's newest run
// so a run's progress and its Cancel are in front of the user where they started it,
// not only on a screen they'd have to navigate to.
//
// Before this, Simple had no way to cancel a run at all and no progress readout
// anywhere — a submitted job was a spinner with no end and no exit.

// The three error statuses share one bucket (colour + placement) but NOT one label —
// a job the user canceled must not read as "Failed".
const STATUS_LABEL = {
  canceled: "Canceled",
  interrupted: "Interrupted",
};

const BUCKET_LABEL = {
  running: "Running",
  queued: "Queued",
  done: "Done",
  failed: "Failed",
};

/**
 * Which of the four visual buckets a job belongs to.
 * @param {object} job
 * @returns {"running"|"queued"|"done"|"failed"}
 */
export function bucketFor(job) {
  if (terminalStatuses.has(job.status)) {
    return errorStatuses.has(job.status) ? "failed" : "done";
  }
  return pendingStatuses.has(job.status) ? "queued" : "running";
}

export function badgeLabelFor(bucket, status) {
  return STATUS_LABEL[status] ?? BUCKET_LABEL[bucket];
}

/** A job can be canceled while it has not reached a terminal state. */
export function jobIsCancelable(job) {
  return Boolean(job) && !terminalStatuses.has(job.status);
}

// Whether a studio should stop showing this run's card. The QUEUE always keeps every row —
// that is its job — but a studio is a workspace, not a history, so a finished run should not
// sit between the Generate button and the results.
//
// Two outcomes clear themselves and two do not:
//   completed → the results grid below IS the outcome; a "Done" row is redundant chrome.
//   canceled  → the user did it deliberately, so they already know.
//   failed / interrupted → the card is the ONLY place the worker's reason is shown in a
//     studio. Dismissing it would put us back to a run that silently produced nothing.
export function jobClearsFromStudio(job) {
  return job?.status === "completed" || job?.status === "canceled";
}

// The job's headline: the model it runs, falling back to the job type — what the
// advanced Queue card leads with, minus the worker/GPU detail.
export function jobTitle(job) {
  const model = job.payload?.modelName ?? job.payload?.model ?? job.payload?.modelId;
  return model ? String(model) : formatType(job.type);
}

export function jobSub(job) {
  const parts = [formatType(job.type)];
  const { width, height, count } = job.payload ?? {};
  if (width && height) {
    parts.push(`${width}×${height}`);
  }
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

/**
 * @param {object} props
 * @param {object} props.job
 * @param {(job: object) => any} [props.onCancel] - omitted ⇒ no Cancel button.
 */
export function SimpleJobCard({ job, onCancel }) {
  // Local, so the button reports immediately: the job's status only flips once the
  // server responds and the SSE tick lands, which is long enough to look unresponsive.
  const [canceling, setCanceling] = useState(false);
  const bucket = bucketFor(job);
  const progress = Number(job.progress);
  const hasProgress = Number.isFinite(progress) && progress > 0;
  const cancelable = jobIsCancelable(job);
  const error = errorStatuses.has(job.status) ? String(job.error ?? job.message ?? "").trim() : "";
  // A running job with no progress reading yet gets an indeterminate bar rather than a
  // 0% one, so "working" is distinguishable from "stuck at zero" (mirrors the advanced
  // WorkerProgressCard).
  const indeterminate = bucket === "running" && !hasProgress;
  const showBar = bucket !== "queued" && !error;

  return (
    <div className="su-job">
      <div className="su-job-head">
        <span aria-hidden="true" className={`su-dot su-dot--${bucket}`} />
        <span className="su-job-text">
          <span className="su-job-title">{jobTitle(job)}</span>
          <span className="su-job-sub">{jobSub(job)}</span>
        </span>
        <span className={`su-job-badge su-job-badge--${bucket}`}>
          {bucket === "running" && hasProgress ? percent(progress) : badgeLabelFor(bucket, job.status)}
        </span>
        {cancelable && onCancel ? (
          <button
            className="su-job-cancel"
            disabled={canceling}
            onClick={async () => {
              setCanceling(true);
              try {
                await onCancel(job);
              } finally {
                setCanceling(false);
              }
            }}
            title="Cancel this run"
            type="button"
          >
            <Icon.Close size={14} />
            {canceling ? "Canceling…" : "Cancel"}
          </button>
        ) : null}
      </div>
      {error ? <p className="su-job-error">{error}</p> : null}
      {showBar ? (
        <div
          className={[
            "su-bar",
            bucket === "done" ? "su-bar--done" : "",
            indeterminate ? "su-bar--indeterminate" : "",
          ]
            .filter(Boolean)
            .join(" ")}
        >
          <span style={indeterminate ? undefined : { width: bucket === "done" ? "100%" : percent(progress) }} />
        </div>
      ) : null}
    </div>
  );
}
