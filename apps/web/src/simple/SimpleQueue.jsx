import React, { useMemo } from "react";
import { useAppContext } from "../context/AppContext.js";
import { SimpleJobCard, bucketFor } from "./SimpleJobCard.jsx";
import { useSimpleUi } from "./SimpleUiContext.js";

// Simple Queue (design handoff): three stat tiles (Running / Queued / Done) over a list
// of job rows. The rows are the shared SimpleJobCard, the same component the studios
// render under their Generate button — so progress, the failure reason and Cancel behave
// identically wherever a run is shown.
//
// Same live `jobs` feed the advanced Queue reads (SSE-updated through useJobEvents), so
// the two views never disagree; what Simple drops is the per-worker resource panel and
// the retry/clear controls, which stay in the full workspace.

export function SimpleQueue() {
  const { jobs = [], jobAction } = useAppContext();
  const { toast } = useSimpleUi();

  const counts = useMemo(() => {
    const tally = { running: 0, queued: 0, done: 0 };
    for (const job of jobs) {
      const bucket = bucketFor(job);
      if (bucket in tally) {
        tally[bucket] += 1;
      }
    }
    return tally;
  }, [jobs]);

  async function cancel(job) {
    await jobAction?.(job, "cancel");
    toast("Run canceled");
  }

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

      {jobs.length ? (
        jobs.map((job) => <SimpleJobCard job={job} key={job.id} onCancel={cancel} />)
      ) : (
        <p className="su-empty">Nothing in the queue. Runs you start in a studio show up here.</p>
      )}
    </div>
  );
}
