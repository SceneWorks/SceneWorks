import React from "react";
import { Yue2RunCard } from "../audioYue2Parts.jsx";

export function Yue2ReviewPanel({
  runs,
  assets,
  model,
  jobAction,
  readyToRun,
  tierProblem,
  submit,
  fetchVersionAbc,
  update
}) {
  return (
<div className="studio-results yue2-runs" data-testid="yue2-runs">
          <section className="review-panel">
            <div className="review-panel-head">
              <div className="review-panel-head-title">
                <h2>Song Lab runs</h2>
              </div>
            </div>
            {runs.length === 0 ? <div className="empty-panel">No YuE2 runs in this project yet</div> : null}
            {runs.map((job) => (
              <Yue2RunCard
                assets={assets}
                job={job}
                key={job.id}
                model={model}
                onCancel={(target) => jobAction?.(target, "cancel")}
                decodeBlocked={readyToRun ? null : tierProblem || "Install YuE2 first."}
                onDecodeAgain={(sourceJobId, decoder) => submit("decode", { sourceJobId, decoder })}
                onFetchVersionAbc={fetchVersionAbc}
                onOpenVersion={(versionId) => update({ tab: "scores", selectedVersionId: versionId })}
                onOpenTranscription={(transcriptionId) => update({ tab: "cover", transcriptionId })}
                onRestorePlan={(jobId) => update({ tab: "compose", planSource: "restore", restorePlanJobId: jobId })}
              />
            ))}
          </section>
        </div>
  );
}
