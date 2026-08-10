import React, { useEffect, useState } from "react";
import { WorkerProgressCard } from "./WorkerProgressCard.jsx";
import { terminalStatuses } from "../constants.js";

// Inline "this feature needs a model you don't have" notice with a download offer.
//
// The compact sibling of ModelAvailabilityGate (sc-5947). That one answers "no installed model can
// serve this SCREEN" and replaces the whole body; this one answers "one option on an otherwise
// usable screen needs a specific utility model", where blanking the screen would be wrong — the
// ControlNet Training Studio still trains LoRAs without DWPose, and a catalog still runs vision and
// embedding analysis without the person detector.
//
// It exists because that shape was already being open-coded per surface: RefinePromptControl and
// DatasetCaptionDialog each carry their own `downloadRequested` state, their own copy and their own
// error handling for the identical interaction. Extending the pattern to two more surfaces meant
// either a fourth and fifth copy or one component; those two are worth migrating onto this later.
//
// Renders NOTHING when `models` is empty, so callers can mount it unconditionally and let
// `missingRequiredModels` decide.
//
// Props:
//   models       — the missing catalog entries (modelEligibility.missingRequiredModels).
//   feature      — what needs them, as a sentence subject ("Pose ControlNet training").
//   detail       — optional extra sentence explaining the consequence.
//   downloadJobs — model_download jobs, for live progress. OPTIONAL: a screen that reads only the
//                  static app context (DatasetCatalogsScreen, deliberately — sc-8855 keeps it off
//                  the per-SSE-tick live context) passes nothing and gets the local "Queued" state
//                  below instead, which needs no jobs list.
function sizeText(model) {
  if (!model?.downloadSizeLabel) {
    return null;
  }
  return model.downloadSizeEstimated ? `~${model.downloadSizeLabel}` : model.downloadSizeLabel;
}

export function RequiredModelsNotice({
  models = [],
  feature,
  detail,
  downloadJobs = [],
  onDownload,
  onOpenModels,
  onOpenQueue,
  onCancelJob,
}) {
  // Ids the user has clicked Download for, for callers that pass no `downloadJobs`. Without it the
  // button would sit there looking unclicked until a catalog refetch lands.
  const [requested, setRequested] = useState(() => new Set());
  const [error, setError] = useState("");

  // Drop a model from the local "requested" set once it leaves `models` (i.e. it finished
  // installing and `missingRequiredModels` stopped returning it), so a later uninstall can be
  // re-requested rather than showing a permanently stale "Queued".
  //
  // Joined into a STRING so the effect depends on the ids BY VALUE: `models` is a fresh array on
  // every render, so depending on it directly would re-run this each tick. NUL is the separator for
  // the same reason TrainingStudio's `gpuOptions.join("\u0000")` uses it — no id can contain it, so
  // no two different id lists can collide into one key. Write it as the ESCAPE, never as a literal
  // NUL: `scripts/check-source-control-bytes.mjs` rejects raw C0 bytes in tracked source, and that
  // is a parity-lane failure rather than anything eslint or vitest would catch.
  const presentIds = models.map((model) => model.id).join("\u0000");
  useEffect(() => {
    const present = new Set(presentIds ? presentIds.split("\u0000") : []);
    setRequested((prev) => {
      const next = new Set([...prev].filter((id) => present.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [presentIds]);

  if (!models.length) {
    return null;
  }

  const activeJobFor = (model) =>
    downloadJobs.find(
      (job) => job.payload?.modelId === model.id && !terminalStatuses.has(job.status),
    );

  async function handleDownload(model) {
    if (typeof onDownload !== "function") return;
    setError("");
    try {
      const job = await onDownload(model);
      // A caller whose `onDownload` resolves to nothing still gets the optimistic state — the click
      // did enqueue something, and leaving the button live invites a duplicate download.
      setRequested((prev) => new Set(prev).add(model.id));
      return job;
    } catch (err) {
      setError(err?.message || "Could not start the model download.");
      return null;
    }
  }

  return (
    <div className="required-models-notice" role="status">
      <p className="inline-warning">
        {feature} needs {models.length === 1 ? "a model" : `${models.length} models`} that
        {models.length === 1 ? " isn’t" : " aren’t"} installed yet.
        {detail ? ` ${detail}` : ""}
      </p>
      <ul className="required-models-list">
        {models.map((model) => {
          const job = activeJobFor(model);
          const size = sizeText(model);
          const pending = Boolean(job) || requested.has(model.id);
          return (
            <li key={model.id}>
              <div className="required-models-row">
                <span>
                  <strong>{model.name ?? model.id}</strong>
                  {size ? <small>{size}</small> : null}
                </span>
                <button
                  className="secondary-action"
                  disabled={!onDownload || pending}
                  onClick={() => handleDownload(model)}
                  type="button"
                >
                  {job ? job.status : pending ? "Queued" : "Download"}
                </button>
              </div>
              {job ? (
                <WorkerProgressCard job={job} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
              ) : null}
            </li>
          );
        })}
      </ul>
      {error ? (
        <p className="inline-warning" role="alert">
          {error}
        </p>
      ) : null}
      {onOpenModels ? (
        <button className="link-button" onClick={onOpenModels} type="button">
          Open the Model Manager
        </button>
      ) : null}
    </div>
  );
}
