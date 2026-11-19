import { useState } from "react";

const reasonText = (reason) => ({
  receipt_missing: "This job has no complete execution receipt.",
  deterministic_replay_unavailable: "The provider has not verified exact replay for this run.",
  runtime_revision_changed: "The inference runtime changed since this run.",
  device_unavailable: "The original worker is unavailable.",
  device_changed: "The original device changed.",
  source_binding_missing: "The original source binding is unavailable.",
  source_route_missing: "The original source route is unavailable.",
  source_route_changed: "The original source route changed.",
  source_unavailable: "The original model source is unavailable.",
  route_unavailable: "The original worker no longer supports this route.",
  seed_draw_binding_changed: "The recorded random draws do not match the saved seed.",
  unsupported_draw_mapping: "This run used random draws that cannot be pinned for replay.",
  quality_rung_unavailable: "The recorded quality-affecting memory strategy is unavailable on this host.",
  quality_rung_changed: "The replay's quality-affecting memory strategy changed, so it stopped.",
  estimator_unavailable: "The recorded Low Res Start transition estimator is unavailable on this host.",
  estimator_changed: "The replay's Low Res Start transition estimator changed, so it stopped.",
  unsupported_low_res_recipe_version: "The recorded Low Res Start recipe version is unsupported by this build.",
  low_res_start_unavailable: "Low Res Start is unavailable for this model or backend on this host.",
  low_res_receipt_missing: "This run has no complete Low Res Start execution receipt.",
  low_res_recipe_changed: "The recorded Low Res Start recipe changed, so exact replay stopped.",
  backend_changed: "The executing backend differs from the recorded backend.",
  completed_execution_changed: "The replay's execution choices changed, so it stopped.",
  execution_failed: "The replay stopped before it could reproduce the recorded execution.",
}[reason] ?? "The recorded execution cannot be replayed exactly.");

export default function ExactReplayAction({ job, onExactReplay }) {
  const [refusal, setRefusal] = useState("");
  const canReplay = onExactReplay && job.status === "completed" &&
    ["image_generate", "image_edit"].includes(job.type);
  const reason = refusal || (job.result?.replayUnavailable?.code === "replay_unavailable"
    ? job.result.replayUnavailable.reason : "");
  return <>
    {reason ? <p role="alert" className="worker-progress-card__message error">{reasonText(reason)}</p> : null}
    {canReplay ? <button
      className="secondary-action"
      onClick={async () => {
        setRefusal("");
        const outcome = await onExactReplay(job);
        if (outcome?.code === "replay_unavailable") setRefusal(outcome.context?.reason ?? outcome.reason);
      }}
      type="button"
    >Exact replay</button> : null}
  </>;
}
