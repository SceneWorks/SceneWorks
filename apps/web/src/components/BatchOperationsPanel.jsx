import React, { useMemo, useState } from "react";
import { Modal } from "./Modal.jsx";
import { BATCH_OPS } from "../batchOps.js";
import { issue } from "../validation/issues.js";
import { useValidation } from "../validation/useValidation.js";
import { ValidationSummary } from "../validation/Validation.jsx";
import { DepthResolutionFields } from "./DepthResolutionFields.jsx";
import { DEFAULT_DEPTH_RESOLUTION, depthResolutionError } from "../depthJobs.js";
import { IrisRestoreControls, TrainedIrisModelSelect } from "./IrisRestoreControls.jsx";
import { IRIS_RESTORE_DEFAULTS, irisRestoreControls } from "../irisRestore.js";

// Batch operations panel (sc-6112): pick ONE op (upscale / detail / edit) + shared
// params and apply it across the selected image assets. Presentational — it owns the
// op/param form state and calls `onRun(op, params)`; the parent does the fan-out
// (decode dims for edit, POST one job per asset) and feeds back per-item `items` +
// the aggregate `progress` so this panel renders the live progress view.

const STATUS_LABEL = { queued: "Queued", running: "Running…", completed: "Done", failed: "Failed" };

// What gates Run, in the app-wide vocabulary (epic 10644, sc-10652). An empty selection
// is a silent requirement — the panel's header count shows it. A missing model/engine or
// a missing edit prompt used to only dim the button with no stated reason (the sc-10492
// defect); now they say why.
export function batchOperationValidation({ count, op, missingModel, missingPrompt, depthUnavailable, depthResolutionIssue, restorePlan = null } = {}) {
  const issues = [];
  if (!(count > 0)) {
    issues.push(issue.requirement("selection", "Select at least one image"));
  }
  if (missingModel) {
    issues.push(
      issue.error(null, op === "upscale" ? "No upscale engine is available — install one in Models." : "No compatible model is available for this operation."),
    );
  }
  if (missingPrompt) {
    issues.push(issue.error("prompt", "Enter a prompt for the edit."));
  }
  // Depth (sc-25682): the exact reason the action cannot run here (no native lane, not installed,
  // incomplete install), and an out-of-range max side.
  if (op === "depth" && depthUnavailable) {
    issues.push(issue.error(null, depthUnavailable));
  }
  if (op === "depth" && depthResolutionIssue) {
    issues.push(issue.error("resolution", depthResolutionIssue));
  }
  // Iris restoration (sc-25683): every selected image must plan before Run — a refused plan (or a
  // still-loading preview) is never submitted, and the request is never silently changed.
  if (op === "upscale" && restorePlan) {
    if (restorePlan.blocked) {
      issues.push(issue.error(null, restorePlan.error ? `Cannot restore: ${restorePlan.error}` : "The restoration cannot run as set."));
    } else if (!restorePlan.ready) {
      issues.push(issue.requirement("restorePlan", "Planning the restoration…"));
    }
  }
  return issues;
}

export function BatchOperationsPanel({
  assets,
  editModels = [],
  detailModels = [],
  upscaleEngines = [],
  depthAvailability = { available: false, reason: "Depth estimation is not available." },
  // The selected depth model's own availability (S3): a trained depth model needs no release install.
  // Absent → the `depthAvailability` above gates every selection.
  depthAvailabilityFor = null,
  defaultDepthModel = "",
  initialOp = "upscale",
  // For the Iris restoration preview (sc-25683): the selection's project and the API token.
  projectId = null,
  token = "",
  busy = false,
  items = null,
  progress = null,
  onRun,
  onClose,
}) {
  const [op, setOp] = useState(initialOp);
  // Upscale params.
  const [engine, setEngine] = useState(upscaleEngines[0]?.key ?? "real-esrgan");
  const [factor, setFactor] = useState(4);
  const [softness, setSoftness] = useState(0.5);
  const [irisRestore, setIrisRestore] = useState(IRIS_RESTORE_DEFAULTS);
  const [irisPlanStatus, setIrisPlanStatus] = useState({ ready: false, blocked: false, error: null });
  // Detail params.
  const [detailModel, setDetailModel] = useState(detailModels[0]?.id ?? "");
  const [strength, setStrength] = useState(0.55);
  const [cnScale, setCnScale] = useState(0.7);
  // Edit params.
  const [editModel, setEditModel] = useState(editModels[0]?.id ?? "");
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState("");
  // Depth params (sc-25682).
  const [depthResolution, setDepthResolution] = useState(DEFAULT_DEPTH_RESOLUTION);
  // `null` = untouched: follows `defaultDepthModel` (re-derived by the host from the live catalog).
  const [depthModelChoice, setDepthModel] = useState(null);
  const depthModel = depthModelChoice ?? defaultDepthModel;

  const count = assets.length;
  const running = items != null;

  // Resolve the active selections against the (async-loaded) option lists so a stale
  // or empty default falls back to the first available entry.
  const activeEngine = upscaleEngines.find((e) => e.key === engine) ?? upscaleEngines[0] ?? null;
  const factors = activeEngine?.factors ?? [2, 4];
  const effectiveFactor = factors.includes(factor) ? factor : factors[0];
  const activeDetailModel = detailModels.find((m) => m.id === detailModel)?.id ?? detailModels[0]?.id ?? "";
  const activeEditModel = editModels.find((m) => m.id === editModel)?.id ?? editModels[0]?.id ?? "";

  const missingModel =
    (op === "detail" && !activeDetailModel) || (op === "edit" && !activeEditModel) || (op === "upscale" && !activeEngine);
  const missingPrompt = op === "edit" && !prompt.trim();
  const selectedDepthAvailability = depthAvailabilityFor ? depthAvailabilityFor(depthModel) : depthAvailability;
  const depthUnavailable = selectedDepthAvailability?.available
    ? null
    : selectedDepthAvailability?.reason || "Depth estimation is not available.";
  const depthResolutionIssue = depthResolutionError(depthResolution);
  const restoring = op === "upscale" && Boolean(activeEngine?.restoration);
  const restorePlan = restoring ? irisPlanStatus : null;
  // One summary gates Run and carries its reasons (epic 10644); `busy` stays a plain gate.
  const runDraft = useMemo(
    () => ({ count, op, missingModel, missingPrompt, depthUnavailable, depthResolutionIssue, restorePlan }),
    [count, op, missingModel, missingPrompt, depthUnavailable, depthResolutionIssue, restorePlan],
  );
  const runValidity = useValidation(batchOperationValidation, runDraft, undefined);
  const canRun = runValidity.ready && !busy;

  function run() {
    if (!canRun) return;
    if (op === "upscale" && activeEngine.restoration) {
      onRun(op, { engine: activeEngine.key, restore: irisRestoreControls(irisRestore) });
    } else if (op === "upscale") {
      onRun(op, { engine: activeEngine.key, factor: effectiveFactor, softness });
    } else if (op === "detail") {
      onRun(op, { model: activeDetailModel, strength, cnScale });
    } else if (op === "depth") {
      onRun(op, { resolution: depthResolution, ...(depthModel ? { irisModel: depthModel } : {}) });
    } else {
      onRun(op, { model: activeEditModel, prompt: prompt.trim(), seed });
    }
  }

  return (
    <Modal className="batch-ops-modal" label="Batch operations" onClose={onClose}>
      <div className="batch-ops-head">
        <h3>Batch: {count} image{count === 1 ? "" : "s"}</h3>
        <button className="modal-close" onClick={onClose} title="Close" type="button">
          ✕
        </button>
      </div>

      {running ? (
        <div className="batch-ops-progress">
          <p className="batch-ops-summary" aria-live="polite">
            {progress?.done ?? 0} / {progress?.total ?? count} done
            {progress?.failed ? ` · ${progress.failed} failed` : ""}
          </p>
          <ul className="batch-ops-items">
            {items.map((item) => (
              <li className={`batch-ops-item batch-ops-item-${item.status}`} key={item.asset.id}>
                <span className="batch-ops-item-name">{item.asset.displayName ?? item.asset.id}</span>
                <span className="batch-ops-item-status">
                  {STATUS_LABEL[item.status] ?? item.status}
                  {item.error ? `: ${item.error}` : ""}
                </span>
              </li>
            ))}
          </ul>
          <div className="batch-ops-actions">
            <button onClick={onClose} type="button">
              {progress?.allDone ? "Done" : "Close"}
            </button>
          </div>
        </div>
      ) : (
        <div className="batch-ops-form">
          <div className="batch-ops-tabs" role="group" aria-label="Operation">
            {BATCH_OPS.map((entry) => (
              <button
                aria-pressed={op === entry.key}
                className={op === entry.key ? "active" : ""}
                key={entry.key}
                onClick={() => setOp(entry.key)}
                type="button"
              >
                {entry.label}
              </button>
            ))}
          </div>

          {op === "upscale" ? (
            <div className="batch-ops-params">
              <label>
                Engine
                <select onChange={(e) => setEngine(e.target.value)} value={activeEngine?.key ?? ""}>
                  {upscaleEngines.map((e) => (
                    <option key={e.key} value={e.key}>
                      {e.label}
                    </option>
                  ))}
                </select>
              </label>
              {activeEngine?.restoration ? (
                <IrisRestoreControls
                  idPrefix="batch-iris"
                  onChange={setIrisRestore}
                  onStatus={setIrisPlanStatus}
                  planBody={projectId && count ? { projectId, assetIds: assets.map((asset) => asset.id) } : null}
                  token={token}
                  value={irisRestore}
                />
              ) : (
              <label>
                Factor
                <select onChange={(e) => setFactor(Number(e.target.value))} value={effectiveFactor}>
                  {factors.map((f) => (
                    <option key={f} value={f}>
                      {f}×
                    </option>
                  ))}
                </select>
              </label>
              )}
              {activeEngine?.softness ? (
                <label>
                  Softness {softness.toFixed(2)}
                  <input
                    max="1"
                    min="0"
                    onChange={(e) => setSoftness(Number(e.target.value))}
                    step="0.05"
                    type="range"
                    value={softness}
                  />
                </label>
              ) : null}
            </div>
          ) : null}

          {op === "detail" ? (
            <div className="batch-ops-params">
              <label>
                Model
                {detailModels.length ? (
                  <select onChange={(e) => setDetailModel(e.target.value)} value={activeDetailModel}>
                    {detailModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label ?? m.name ?? m.id}
                      </option>
                    ))}
                  </select>
                ) : (
                  <span className="batch-ops-empty">No detail-capable model installed.</span>
                )}
              </label>
              <label>
                Strength {strength.toFixed(2)}
                <input max="1" min="0" onChange={(e) => setStrength(Number(e.target.value))} step="0.05" type="range" value={strength} />
              </label>
              <label>
                ControlNet {cnScale.toFixed(2)}
                <input max="1" min="0" onChange={(e) => setCnScale(Number(e.target.value))} step="0.05" type="range" value={cnScale} />
              </label>
            </div>
          ) : null}

          {op === "edit" ? (
            <div className="batch-ops-params">
              <label>
                Model
                {editModels.length ? (
                  <select onChange={(e) => setEditModel(e.target.value)} value={activeEditModel}>
                    {editModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label ?? m.name ?? m.id}
                      </option>
                    ))}
                  </select>
                ) : (
                  <span className="batch-ops-empty">No edit-capable model installed.</span>
                )}
              </label>
              <label>
                Prompt
                <textarea
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe the edit applied to every selected image…"
                  rows={2}
                  value={prompt}
                />
              </label>
              <label>
                Seed (optional)
                <input
                  onChange={(e) => setSeed(e.target.value)}
                  placeholder="random"
                  type="number"
                  value={seed}
                />
              </label>
              <p className="batch-ops-note">Each image is edited at its native size.</p>
            </div>
          ) : null}

          {op === "depth" ? (
            <div className="batch-ops-params">
              <TrainedIrisModelSelect
                onChange={setDepthModel}
                releaseLabel="Iris 3B Depth (release)"
                task="depth"
                value={depthModel}
              />
              <DepthResolutionFields idPrefix="batch-depth" onChange={setDepthResolution} value={depthResolution} />
              <p className="batch-ops-note">
                Each image gets its own depth map (raw .npy + preview) linked to it; the source images are not modified.
              </p>
            </div>
          ) : null}

          {/* The reasons Run is dead — a missing model/engine or edit prompt. An empty
              selection is a silent requirement (the header count shows it). sc-10652. */}
          <ValidationSummary issues={runValidity.surfaced} label="Batch operation errors" />
          <div className="batch-ops-actions">
            <button onClick={onClose} type="button">
              Cancel
            </button>
            <button className="primary" disabled={!canRun} onClick={run} type="button">
              {busy ? "Starting…" : `Run on ${count} image${count === 1 ? "" : "s"}`}
            </button>
          </div>
        </div>
      )}
    </Modal>
  );
}
