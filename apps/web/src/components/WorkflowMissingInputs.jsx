import React from "react";

import {
  workflowInputSlots,
  workflowInstallable,
  workflowModelUnknown,
  workflowHasMissingInputs,
} from "../workflowShare.js";

// "Here is what is missing, and here is what to do about it" (sc-15952, epic 15945).
//
// The ACTION half of the resolution report. `WorkflowResolutionReport` states every requirement and
// repairs none of them, on purpose; this is the surface that offers the three repairs that exist —
// install what the catalog knows, choose a model when the catalog knows nothing, supply the images
// that were never in the file. One component, two callers: the dropped-image offer (sc-15951) and
// the same offer opened from an already-imported asset.
//
// # It renders NOTHING when there is nothing to do
//
// Not an empty panel, not a "Nothing missing" line, not a heading with an empty list under it. A
// fully-resolvable workflow's offer must look exactly as it did before this component existed —
// zero-state noise on a screen the user reached by dragging in an image is worse than no
// information, because it makes a working case look like it needs attention.
// [`workflowHasMissingInputs`] is the single gate, shared with the tests that assert the absence.
//
// # Nothing here is a default
//
// The substitute model picker starts EMPTY and stays empty until the user chooses. There is no
// "first installed model" fallback, no most-recently-used, no nearest match by family. A
// substituted model produces a different image, and one chosen for the user is indistinguishable
// on screen from a faithful replay — which is the failure this whole epic exists to prevent. The
// CTA is blocked, with the reason named, until the choice is made (`workflowReplayValidation`).

// `failed` is why "Queued" is not a one-way door. A download job that FAILS changes no catalog, so
// nothing re-resolves and the row would otherwise sit disabled saying "Queued" until the panel is
// closed and reopened — a dead button, and a user waiting for something that already stopped. When
// the job is known to have failed the button comes back and the row SAYS so, rather than quietly
// reverting to "Download" as though the press had never happened.
function InstallRow({ row, busy, done, failed, onInstall }) {
  return (
    <li className="workflow-fix-row">
      <span className="workflow-fix-head">
        <strong>
          {row.kind === "model" ? "Model" : "LoRA"} — {row.name}
        </strong>
        <button
          className="btn compact"
          disabled={busy || done}
          onClick={() => onInstall(row)}
          type="button"
        >
          {done ? "Queued" : busy ? "Starting…" : failed ? "Try again" : "Download"}
        </button>
      </span>
      <span className="workflow-fix-detail">
        {failed ? "That download failed — the queue has the reason. " : ""}
        {row.detail}
      </span>
    </li>
  );
}

export function WorkflowMissingInputs({
  report,
  // Installed models this studio can actually select — the SAME list the studio's own picker is
  // built from, so a substitute chosen here cannot be one the picker would drop on the next render.
  models = [],
  // Assets offered by the input pickers. The project's images; empty with no project open, which
  // the pickers say rather than rendering an empty select with no explanation.
  assets = [],
  installing = {},
  installed = {},
  // The rows whose download job failed. Kept apart from `installed` because they are different
  // answers: "not queued" is a button that was never pressed, "failed" is one that was.
  installFailed = {},
  onInstall,
  substituteModelId = "",
  onSubstituteModel,
  poseSubstituteRequired = false,
  poseSubstituteDetail = "",
  inputPicks = {},
  onPickInput,
}) {
  if (!workflowHasMissingInputs(report)) {
    return null;
  }
  const installable = workflowInstallable(report);
  const unknownModel = workflowModelUnknown(report);
  const slots = workflowInputSlots(report);

  return (
    <div className="workflow-fix">
      <h3 className="workflow-fix-title">What this needs from you</h3>

      {installable.length ? (
        <ul className="workflow-fix-list">
          {installable.map((row) => (
            <InstallRow
              busy={Boolean(installing[`${row.kind}:${row.id}`])}
              done={Boolean(installed[`${row.kind}:${row.id}`])}
              failed={Boolean(installFailed[`${row.kind}:${row.id}`])}
              key={`${row.kind}:${row.id}`}
              onInstall={onInstall}
              row={row}
            />
          ))}
        </ul>
      ) : null}

      {unknownModel ? (
        <div className="workflow-fix-substitute">
          <label className="workflow-fix-label" htmlFor="workflow-substitute-model">
            Model to use instead
          </label>
          <select
            disabled={poseSubstituteRequired && models.length === 0}
            id="workflow-substitute-model"
            onChange={(event) => onSubstituteModel?.(event.target.value)}
            value={substituteModelId}
          >
            <option value="">
              {poseSubstituteRequired && models.length === 0
                ? "No pose-capable model available"
                : "Choose a model…"}
            </option>
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name ?? model.id}
              </option>
            ))}
          </select>
          <p className="workflow-fix-detail">
            {report.model?.detail} Whichever you pick makes a different image from the one you
            were sent — nothing is chosen for you.
          </p>
          {poseSubstituteDetail ? (
            <p className="workflow-fix-detail">{poseSubstituteDetail}</p>
          ) : null}
        </div>
      ) : null}

      {slots.length ? (
        <div className="workflow-fix-inputs">
          {/* The images never travelled and were never meant to: the envelope records the SHAPE of
              each input and never an id, because an id from someone else's library resolves to
              nothing here. So this is the one section that is not a failure to report — it is the
              recipe asking for its ingredients. */}
          <p className="workflow-fix-detail">
            This recipe starts from images. They are not in the file — a shared recipe records what
            KIND of image it needs, never the image itself.
          </p>
          <ul className="workflow-fix-list">
            {slots.map((slot) => (
              <li className="workflow-fix-row" key={slot.key}>
                <span className="workflow-fix-head">
                  <strong>{slot.label}</strong>
                  {slot.pickable ? (
                    <select
                      aria-label={slot.label}
                      onChange={(event) => onPickInput?.(slot.key, event.target.value)}
                      value={inputPicks[slot.key] ?? ""}
                    >
                      <option value="">Choose an image…</option>
                      {assets.map((asset) => (
                        <option key={asset.id} value={asset.id}>
                          {asset.displayName ?? asset.id}
                        </option>
                      ))}
                    </select>
                  ) : null}
                </span>
                {slot.detail ? <span className="workflow-fix-detail">{slot.detail}</span> : null}
                {slot.elsewhere ? (
                  <span className="workflow-fix-detail">{slot.elsewhere}</span>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
