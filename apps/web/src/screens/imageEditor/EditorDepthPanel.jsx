import React, { useMemo, useState } from "react";
import { apiFetch } from "../../api.js";
import { terminalStatuses } from "../../constants.js";
import { useAppContext } from "../../context/AppContext.js";
import { AssetThumbnail } from "../../components/assetMedia.jsx";
import { DepthResolutionFields } from "../../components/DepthResolutionFields.jsx";
import { TrainedIrisModelSelect } from "../../components/IrisRestoreControls.jsx";
import {
  DEFAULT_DEPTH_RESOLUTION,
  DEPTH_ENDPOINT,
  buildDepthJobBody,
  defaultDepthModel,
  depthAvailability,
  depthModelFromAsset,
  depthResolutionError,
  depthSettingsFromAsset,
  isDepthAsset,
} from "../../depthJobs.js";

const STATUS_TEXT = { queued: "Queued", preparing: "Preparing", running: "Estimating", completed: "Done" };

// The Image Editor depth tool (Iris-3B, sc-25682). Depth is estimated from the SAVED Library image
// the document was opened from — never from an unsaved canvas — so the depth map's lineage always
// names a real asset. The result is a new Library asset (raw .npy + preview); the source is untouched
// and the canvas is not replaced. Earlier depth maps of the same source are listed with the settings
// they were made with, so a run can be reopened and repeated.
export function EditorDepthPanel({ working, dirty }) {
  const {
    activeProject,
    assets = [],
    jobs = [],
    jobAction,
    models = [],
    macCapabilities,
    requestedGpu = "auto",
    setPreviewAsset,
    token = "",
  } = useAppContext();
  const [resolution, setResolution] = useState(DEFAULT_DEPTH_RESOLUTION);
  // `null` = untouched: the default follows the catalog (it may load late, or a trained depth model
  // may be installed later) until the user picks a model or reopens one.
  const [irisModelChoice, setIrisModel] = useState(null);
  const irisModel = irisModelChoice ?? defaultDepthModel(models);
  const [jobId, setJobId] = useState(null);
  const [submitError, setSubmitError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const availability = depthAvailability(models, macCapabilities, irisModel);
  const sourceAssetId = working?.source?.assetId ?? null;
  const resolutionIssue = depthResolutionError(resolution);
  const job = jobId ? jobs.find((item) => item.id === jobId) ?? null : null;
  const running = Boolean(job && !terminalStatuses.has(job.status));
  const resultAsset = job?.status === "completed" ? job.result?.assets?.[0] ?? null : null;
  const previousResults = useMemo(
    () =>
      assets
        .filter(
          (asset) =>
            isDepthAsset(asset) && asset.lineage?.sourceAssetId === sourceAssetId && !asset.status?.trashed,
        )
        .sort((a, b) => String(b.createdAt ?? "").localeCompare(String(a.createdAt ?? ""))),
    [assets, sourceAssetId],
  );

  let blocked = null;
  if (!availability.available) blocked = availability.reason;
  else if (!sourceAssetId) blocked = "Save this image to the Library first — depth is estimated from a saved image.";
  else if (dirty) blocked = "Save your edits first — depth is estimated from the saved Library image, not the unsaved canvas.";
  else if (resolutionIssue) blocked = resolutionIssue;

  async function run() {
    if (blocked || running || submitting) return;
    setSubmitError(null);
    setSubmitting(true);
    try {
      const body = buildDepthJobBody({
        project: activeProject,
        requestedGpu,
        sourceAssetId,
        resolution,
        displayName: working?.source?.name ?? undefined,
        irisModel,
      });
      const created = await apiFetch(DEPTH_ENDPOINT, token, { method: "POST", body: JSON.stringify(body) });
      if (!created?.id) throw new Error("The job response did not include an id.");
      setJobId(created.id);
    } catch (error) {
      setSubmitError(error?.message || "Could not start depth estimation.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <>
      <div className="ie-section">
        <div className="ie-sec-title">Depth map</div>
        <p className="ie-note">
          Estimates relative depth (Iris-3B) from the saved image and saves it as a new Library asset: the raw float32
          map (.npy, not metric) plus a colorized preview. The image itself is not changed.
        </p>
        <TrainedIrisModelSelect
          models={models}
          onChange={setIrisModel}
          releaseLabel="Iris 3B Depth (release)"
          task="depth"
          value={irisModel}
        />
        <DepthResolutionFields
          idPrefix="editor-depth"
          onChange={setResolution}
          sourceSize={working ? { width: working.width, height: working.height } : null}
          value={resolution}
        />
      </div>
      <div className="ie-section">
        {blocked ? <p className="ie-note">{blocked}</p> : null}
        {submitError ? (
          <p className="field-error" role="alert">
            {submitError}
          </p>
        ) : null}
        <button
          className="ie-btn block primary"
          disabled={Boolean(blocked) || running || submitting}
          onClick={run}
          type="button"
        >
          {submitting ? "Starting…" : "Estimate depth"}
        </button>
        {job ? (
          <div className="depth-job-status" aria-live="polite">
            <p className="ie-note">
              {job.status === "failed" || job.status === "interrupted"
                ? `Depth estimation failed: ${job.error ?? job.message ?? "unknown error"}`
                : job.status === "canceled"
                  ? "Depth estimation canceled — no depth map was saved."
                  : `${STATUS_TEXT[job.status] ?? job.status}${job.message ? ` — ${job.message}` : ""}${
                      running && Number.isFinite(job.progress) ? ` (${Math.round(job.progress * 100)}%)` : ""
                    }`}
            </p>
            {running ? (
              <button className="ie-btn block" onClick={() => jobAction?.(job, "cancel")} type="button">
                Cancel depth estimation
              </button>
            ) : null}
            {resultAsset ? (
              <button className="ie-btn block" onClick={() => setPreviewAsset?.(resultAsset)} type="button">
                Open depth map
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
      {previousResults.length ? (
        <div className="ie-section">
          <div className="ie-sec-title">Depth maps of this image</div>
          <ul className="depth-results">
            {previousResults.map((asset) => {
              const settings = depthSettingsFromAsset(asset);
              return (
                <li key={asset.id}>
                  <button className="depth-result-thumb" onClick={() => setPreviewAsset?.(asset)} type="button">
                    <AssetThumbnail asset={asset} />
                  </button>
                  <span className="ie-note">
                    {settings.mode === "native" ? "Native" : `Max side ${settings.maxSide}px`}
                  </span>
                  <button
                    className="ie-btn"
                    onClick={() => {
                      setResolution(settings);
                      setIrisModel(depthModelFromAsset(asset));
                    }}
                    type="button"
                  >
                    Use these settings
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </>
  );
}
