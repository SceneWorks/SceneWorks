import React, { useId, useState } from "react";

const VIEWS = [
  ["batch", "Latest batch"],
  ["upscale", "Upscale video"],
];

// Results area (studio layout simplification): a segmented switch between the latest batch
// (progress, recent assets, the Replace-person comparison) and the standalone SeedVR2 video
// upscaler, which used to be folded into Advanced. The upscaler is a post-process on an existing
// clip, independent of the compose form, so it lives with the results rather than the settings.
// Only the open view is mounted, the same contract the Advanced disclosure had for the upscaler.
export function VideoResultsArea({ batch, upscale, summary = null }) {
  const [view, setView] = useState("batch");
  const baseId = useId();
  return (
    <section aria-label="Results" className="studio-results video-results">
      <div className="video-results-head">
        <div aria-label="Results view" className="video-results-switch" role="tablist">
          {VIEWS.map(([id, label]) => (
            <button
              aria-controls={view === id ? `${baseId}-${id}` : undefined}
              aria-selected={view === id}
              className={view === id ? "video-results-tab on" : "video-results-tab"}
              id={`${baseId}-${id}-tab`}
              key={id}
              onClick={() => setView(id)}
              role="tab"
              type="button"
            >
              {label}
            </button>
          ))}
        </div>
        {summary ? <span className="video-results-summary">{summary}</span> : null}
      </div>
      <div
        aria-labelledby={`${baseId}-${view}-tab`}
        className="video-results-body"
        id={`${baseId}-${view}`}
        role="tabpanel"
      >
        {view === "batch" ? batch : upscale}
      </div>
    </section>
  );
}
