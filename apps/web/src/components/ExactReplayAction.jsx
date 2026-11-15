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
