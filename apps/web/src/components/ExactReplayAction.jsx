import { useState } from "react";
import { isLowResStartActive } from "../workflowShare.js";

const reasonText = (reason) => ({
  support_unknown: "Exact replay support is unavailable for this job. Use Duplicate to generate again.",
  receipt_missing: "This job has no complete execution receipt.",
  execution_unavailable: "This run did not record the execution details required for exact replay. Use Duplicate to generate again.",
  invalid_execution_identity: "This run's recorded execution details are incomplete.",
  receipt_changed: "This job's execution receipt no longer matches the original run.",
  composite_replay_unavailable: "Exact replay is unavailable for runs with multiple execution stages.",
  replay_scope_changed: "The provider cannot reproduce this run on the required device and runtime.",
  output_hashes_unavailable: "This run has no verified output fingerprints for exact replay.",
  source_not_complete: "This job has not finished saving its outputs. Try again after it completes.",
  replay_adapter_unavailable: "Exact replay is unavailable for this job type.",
  provider_route_unavailable: "This run has no verified provider route for exact replay.",
  provider_route_changed: "The original model provider route changed.",
  missing_randomness: "This run did not record the random draws required for exact replay.",
  options_not_allowed: "Exact replay requires the original options without changes.",
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

export function checkExactReplayHostSupport(payload, context = {}) {
  const receipt = context.receipt ?? null;
  const host = context.host ?? null;
  if (!receipt || !host) {
    return { ok: false, code: "replay_unavailable", reason: "receipt_missing", field: "executionReceipt" };
  }
  if (receipt.acceptedBackend && host.backend && receipt.acceptedBackend !== host.backend) {
    return { ok: false, code: "replay_unavailable", reason: "backend_changed", field: "backend" };
  }
  const identityRung = receipt.execution?.identity?.quality_policy?.value?.rung ?? null;
  if (identityRung && Array.isArray(host.availableQualityRungs) && !host.availableQualityRungs.includes(identityRung)) {
    return { ok: false, code: "replay_unavailable", reason: "quality_rung_unavailable", field: "execution.quality_policy" };
  }
  const lowResStart = payload?.advanced?.lowResStart;
  if (isLowResStartActive(lowResStart)) {
    if (lowResStart?.version !== 1) {
      return {
        ok: false,
        code: "replay_unavailable",
        reason: "unsupported_low_res_recipe_version",
        field: "advanced.lowResStart.version",
      };
    }
    if (host.supportsLowResStart === false) {
      return {
        ok: false,
        code: "replay_unavailable",
        reason: "low_res_start_unavailable",
        field: "advanced.lowResStart",
      };
    }
    const recorded = receipt.lowResStart;
    if (!recorded) {
      return {
        ok: false,
        code: "replay_unavailable",
        reason: "low_res_receipt_missing",
        field: "executionReceipt.lowResStart",
      };
    }
    if (recorded.version !== 1) {
      return {
        ok: false,
        code: "replay_unavailable",
        reason: "unsupported_low_res_recipe_version",
        field: "lowResStart.version",
      };
    }
    if (
      recorded.k !== lowResStart.k ||
      typeof lowResStart.scale !== "number" ||
      typeof recorded.scale !== "number" ||
      Math.fround(recorded.scale) !== Math.fround(lowResStart.scale)
    ) {
      return {
        ok: false,
        code: "replay_unavailable",
        reason: "low_res_recipe_changed",
        field: "executionReceipt.lowResStart",
      };
    }
    if (Array.isArray(host.availableEstimators) && !host.availableEstimators.includes(recorded.estimator)) {
      return {
        ok: false,
        code: "replay_unavailable",
        reason: "estimator_unavailable",
        field: "lowResStart.estimator",
      };
    }
    if (
      recorded.selectedRung &&
      Array.isArray(host.availableQualityRungs) &&
      !host.availableQualityRungs.includes(recorded.selectedRung)
    ) {
      return {
        ok: false,
        code: "replay_unavailable",
        reason: "quality_rung_unavailable",
        field: "lowResStart.selectedRung",
      };
    }
  } else if (receipt.lowResStart) {
    return {
      ok: false,
      code: "replay_unavailable",
      reason: "low_res_recipe_changed",
      field: "advanced.lowResStart",
    };
  }
  return { ok: true };
}

export default function ExactReplayAction({ job, onExactReplay }) {
  const [refusal, setRefusal] = useState("");
  const [pending, setPending] = useState(false);
  const canReplay = onExactReplay && job.status === "completed" &&
    ["image_generate", "image_edit"].includes(job.type);
  const reason = refusal || (job.result?.replayUnavailable?.code === "replay_unavailable"
    ? job.result.replayUnavailable.reason : "");
  // Older servers have no support summary. Fail closed instead of offering an
  // action whose recorded prerequisites are unknown; POST still checks live host facts.
  const unavailable = canReplay && job.exactReplay?.supported !== true
    ? job.exactReplay?.reason || "support_unknown" : "";
  return <>
    {reason ? <p role="alert" className="worker-progress-card__message error">{reasonText(reason)}</p> : null}
    {canReplay ? <button
      className="secondary-action"
      disabled={pending || Boolean(unavailable)}
      aria-description={unavailable ? reasonText(unavailable) : undefined}
      title={unavailable ? reasonText(unavailable) : "Replay the recorded execution on its original device and runtime"}
      onClick={async () => {
        if (pending || unavailable) return;
        setRefusal("");
        setPending(true);
        try {
          const outcome = await onExactReplay(job);
          if (outcome?.code === "replay_unavailable") setRefusal(outcome.context?.reason ?? outcome.reason ?? "unknown");
        } finally {
          setPending(false);
        }
      }}
      type="button"
    >{pending ? "Queuing replay…" : "Exact replay"}</button> : null}
  </>;
}
