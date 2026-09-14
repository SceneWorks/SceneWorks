import React from "react";
import { WorkerProgressCard } from "./WorkerProgressCard.jsx";
import { terminalStatuses } from "../constants.js";

// Per-Studio model-availability gate (sc-5947). When a Studio has no installed model that
// supports its functions, it renders this instead of its body: a short explanation plus the
// screen's recommended models with an inline Download. A completed download refreshes the
// catalog (App.jsx SSE handler), so `ready` flips and the Studio renders without a reload.
//
// Props:
//   ready          — when true, render `children` (the Studio body) unchanged.
//   initializing   — availability is not authoritative yet; render an indeterminate startup state
//                    instead of download offers derived from an empty or fallback catalog.
//   title/description — gate copy.
//   eyebrow        — the kicker above the title. Defaults to the Studio wording ("No supported
//                    model installed", i.e. none of a whole family qualifies); a screen gated on
//                    ONE named utility model (the Pose Library's DWPose detector) overrides it,
//                    because "no supported model" misdescribes a single required download.
//   offers         — models to offer for download (downloadOffersFor in modelEligibility.js).
//   downloadJobs   — model_download jobs, to show progress for an in-flight offer.
//   onDownload(model) / onOpenModels() / onOpenQueue() / onCancelJob(job) — wired from context.
function offerSizeText(model) {
  if (!model?.downloadSizeLabel) {
    return "Size unavailable";
  }
  return model.downloadSizeEstimated ? `~${model.downloadSizeLabel}` : model.downloadSizeLabel;
}

export function ModelAvailabilityGate({
  ready,
  initializing = false,
  title,
  description,
  eyebrow = "No supported model installed",
  offers = [],
  downloadJobs = [],
  onDownload,
  onOpenModels,
  onOpenQueue,
  onCancelJob,
  children,
}) {
  if (ready) {
    return children;
  }
  if (initializing) {
    return (
      <section className="model-availability-gate" aria-live="polite">
        <div className="model-availability-gate-card">
          <div className="section-heading">
            <p className="eyebrow">Starting local catalog</p>
            <h2>Initializing models…</h2>
          </div>
          <p>Checking which models are installed and available on this machine.</p>
          <div
            aria-label="Initializing models"
            className="catalog-progress catalog-progress--indeterminate"
            role="progressbar"
          >
            <span />
          </div>
        </div>
      </section>
    );
  }
  const activeJobFor = (model) =>
    downloadJobs.find((job) => job.payload?.modelId === model.id && !terminalStatuses.has(job.status));
  return (
    <section className="model-availability-gate">
      <div className="model-availability-gate-card">
        <div className="section-heading">
          <p className="eyebrow">{eyebrow}</p>
          <h2>{title}</h2>
        </div>
        {description ? <p>{description}</p> : null}
        {offers.length ? (
          <div className="model-availability-offers">
            {offers.map((model) => {
              const job = activeJobFor(model);
              return (
                <article className="model-availability-offer" key={model.id}>
                  <div className="model-availability-offer-head">
                    <span>
                      <strong>{model.name ?? model.id}</strong>
                      <small>{offerSizeText(model)}</small>
                    </span>
                    <button
                      disabled={!onDownload || Boolean(job)}
                      onClick={() => onDownload?.(model)}
                      type="button"
                    >
                      {job ? job.status : "Download"}
                    </button>
                  </div>
                  {job ? (
                    <WorkerProgressCard job={job} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
                  ) : null}
                </article>
              );
            })}
          </div>
        ) : (
          <p className="empty-panel compact-panel">No downloadable model in the catalog supports this screen yet.</p>
        )}
        {onOpenModels ? (
          <button className="model-availability-browse" onClick={onOpenModels} type="button">
            Browse all models
          </button>
        ) : null}
      </div>
    </section>
  );
}
