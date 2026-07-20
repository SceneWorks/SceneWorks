import React, { useEffect, useMemo, useRef, useState } from "react";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { Logo } from "../components/Logo.jsx";
import { terminalStatuses } from "../constants.js";
import { audioModelUsable } from "../modelEligibility.js";
import { isDesktop, tauriInvoke } from "../runtime.js";

// The curated "getting started" set is catalog-driven: a model is recommended when
// its manifest entry carries `recommended: true` (config/manifests/builtin.models.jsonc).
// Recommended models are pre-checked for download — unless the entry sets
// `autoDownload: false` (e.g. LTX-2.3, ~146 GB), which keeps it badged-but-unchecked so
// a new user opts into the big download deliberately instead of having it auto-queued.
function isRecommended(model) {
  return model.recommended === true;
}

function autoDownloadDisabled(model) {
  return model.autoDownload === false;
}

function isDownloadable(model) {
  return model.downloadable !== false && Boolean(model.downloads?.[0]?.repo ?? model.repo);
}

// Which catalog models onboarding actually OFFERS. Image/video/utility ride the plain
// downloadable check they always have. Audio is the new type (epic 13400 A3): an
// `audio`-type entry is only offered when it's genuinely usable — i.e. it serves ≥1 Audio
// Studio mode and isn't Mac-blocked — using the exact predicate the Audio Studio's own
// download offers run through (modelEligibility.js `audioModelUsable`), so onboarding never
// dangles a bare audio entry that no mode can drive. `audioModelUsable` already requires
// `type === "audio"`, so the guard is a no-op for every other type.
function isOfferable(model, caps) {
  return isDownloadable(model) && (model.type !== "audio" || audioModelUsable(model, caps));
}

function downloadSizeText(model) {
  if (!model.downloadSizeLabel) {
    return "Size unavailable";
  }
  return model.downloadSizeEstimated ? `~${model.downloadSizeLabel}` : model.downloadSizeLabel;
}

function defaultSelection(models, caps) {
  return new Set(
    models
      .filter(
        (model) =>
          isOfferable(model, caps) &&
          model.installState !== "installed" &&
          isRecommended(model) &&
          !autoDownloadDisabled(model),
      )
      .map((model) => model.id),
  );
}

const TYPE_LABELS = {
  image: "Image models",
  video: "Video models",
  audio: "Audio models",
  utility: "Utility models",
};

// Deterministic group order so audio always lands between video and utility regardless of
// catalog iteration order. Types not listed here (future / "other") render after these, in
// first-seen order.
const TYPE_ORDER = ["image", "video", "audio", "utility"];

function typeRank(type) {
  const index = TYPE_ORDER.indexOf(type);
  return index === -1 ? TYPE_ORDER.length : index;
}

export function SetupWizard({
  models,
  jobs,
  macCapabilities,
  onDownloadModel,
  onCreateProject,
  onComplete,
  onOpenQueue,
}) {
  const [step, setStep] = useState("models");
  const [selected, setSelected] = useState(() => defaultSelection(models, macCapabilities));
  const [started, setStarted] = useState(() => new Set());
  const [projectName, setProjectName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  // When the first project can't be created — almost always because the chosen
  // workspace folder rejects writes (issue #1435 / sc-11855) — the wizard is the
  // ONLY surface the user can see (it overlays the whole app, including Settings),
  // so it must offer its own recovery: repoint the workspace folder + restart.
  const [createFailed, setCreateFailed] = useState(false);
  const [workspaceNotice, setWorkspaceNotice] = useState("");
  const initializedRef = useRef(false);

  // The catalog may arrive a tick after mount; seed the recommended selection
  // once it does (without clobbering a choice the user already made).
  useEffect(() => {
    if (!initializedRef.current && models.length) {
      setSelected(defaultSelection(models, macCapabilities));
      initializedRef.current = true;
    }
  }, [models, macCapabilities]);

  const downloadable = useMemo(
    () => models.filter((model) => isOfferable(model, macCapabilities)),
    [models, macCapabilities],
  );
  const grouped = useMemo(() => {
    const byType = new Map();
    for (const model of downloadable) {
      const key = model.type ?? "other";
      if (!byType.has(key)) {
        byType.set(key, []);
      }
      byType.get(key).push(model);
    }
    // Render groups in the canonical image → video → audio → utility order (then any
    // other types, first-seen) so audio always slots in beside its peers.
    return new Map([...byType.entries()].sort((a, b) => typeRank(a[0]) - typeRank(b[0])));
  }, [downloadable]);

  const activeDownloadJobs = useMemo(
    () => jobs.filter((job) => job.type === "model_download" && !terminalStatuses.has(job.status)),
    [jobs],
  );

  const pendingSelection = useMemo(
    () =>
      downloadable.filter(
        (model) => selected.has(model.id) && model.installState !== "installed" && !started.has(model.id),
      ),
    [downloadable, selected, started],
  );

  function toggle(model) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(model.id)) {
        next.delete(model.id);
      } else {
        next.add(model.id);
      }
      return next;
    });
  }

  function downloadSelected() {
    pendingSelection.forEach((model) => onDownloadModel(model));
    setStarted((current) => new Set([...current, ...pendingSelection.map((model) => model.id)]));
  }

  async function finish(event) {
    event.preventDefault();
    const trimmed = projectName.trim();
    if (!trimmed || submitting) {
      return;
    }
    setSubmitting(true);
    setCreateFailed(false);
    try {
      const created = await onCreateProject(trimmed);
      if (created) {
        await onComplete();
      } else {
        // onCreateProject returns null on failure (the exact error is shown in the
        // app-level banner). Reveal the in-wizard recovery so the user isn't stuck.
        setCreateFailed(true);
      }
    } finally {
      setSubmitting(false);
    }
  }

  // Repoint the workspace folder from inside the wizard. Mirrors the Settings
  // data-directory control: pick a folder, persist it, then apply on restart
  // (the data dir is bound when the API sidecar spawns, so it can't move live).
  async function changeWorkspaceFolder() {
    setWorkspaceNotice("");
    try {
      const picked = await tauriInvoke("choose_data_dir");
      if (!picked) {
        return;
      }
      await tauriInvoke("set_data_dir", { path: picked });
      setWorkspaceNotice(
        `Workspace folder set to ${picked}. Quit and reopen SceneWorks to apply, then finish setup.`,
      );
    } catch (error) {
      setWorkspaceNotice(String(error?.message ?? error));
    }
  }

  return (
    <section className="setup-wizard">
      <div className="setup-wizard-card">
        <span className="setup-wizard-mark" aria-hidden="true">
          <Logo size={48} />
        </span>
        <ol className="setup-wizard-steps" aria-hidden="true">
          <li className={step === "models" ? "active" : "done"}>Models</li>
          <li className={step === "project" ? "active" : ""}>First project</li>
        </ol>

        {step === "models" ? (
          <>
            <h2>Download starter models</h2>
            <p className="setup-wizard-lede">
              Pick the models to download now. Recommended ones are pre-selected — you can add
              more later from Models. Downloads run in the background, so you can keep going.
            </p>

            <div className="setup-wizard-models">
              {downloadable.length === 0 ? (
                <p className="setup-wizard-empty">No downloadable models in the catalog yet.</p>
              ) : (
                [...grouped.entries()].map(([type, items]) => (
                  <div className="setup-wizard-group" key={type}>
                    <h3>{TYPE_LABELS[type] ?? type}</h3>
                    {items.map((model) => {
                      const installed = model.installState === "installed";
                      const downloading = started.has(model.id) && !installed;
                      const recommended = isRecommended(model);
                      return (
                        <label className={`setup-wizard-model${installed ? " installed" : ""}`} key={model.id}>
                          <input
                            type="checkbox"
                            checked={installed || selected.has(model.id)}
                            disabled={installed || downloading}
                            onChange={() => toggle(model)}
                          />
                          <span className="setup-wizard-model-main">
                            <span className="setup-wizard-model-name">
                              {model.name}
                              {recommended && !installed ? <span className="setup-wizard-tag">Recommended</span> : null}
                            </span>
                            <span className="setup-wizard-model-meta">
                              {installed ? "Already installed" : downloading ? "Download started" : downloadSizeText(model)}
                            </span>
                          </span>
                        </label>
                      );
                    })}
                  </div>
                ))
              )}
            </div>

            {activeDownloadJobs.length ? (
              <div className="setup-wizard-progress">
                {activeDownloadJobs.map((job) => (
                  <WorkerProgressCard job={job} key={job.id} onOpenQueue={onOpenQueue} />
                ))}
              </div>
            ) : null}

            <div className="setup-wizard-actions">
              <button
                className="setup-wizard-secondary"
                disabled={pendingSelection.length === 0}
                onClick={downloadSelected}
                type="button"
              >
                {pendingSelection.length ? `Download ${pendingSelection.length} selected` : "Download selected"}
              </button>
              <button className="setup-wizard-cta" onClick={() => setStep("project")} type="button">
                Continue
              </button>
            </div>
          </>
        ) : (
          <>
            <h2>Create your first project</h2>
            <p className="setup-wizard-lede">
              SceneWorks keeps your images, videos, characters, and timelines inside a project.
              Name your first one to jump into the studio.
            </p>
            <form className="setup-wizard-form" onSubmit={finish}>
              <input
                aria-label="Project name"
                autoFocus
                disabled={submitting}
                onChange={(event) => setProjectName(event.target.value)}
                placeholder="e.g. My First Project"
                value={projectName}
              />
              <button className="setup-wizard-cta" disabled={submitting || !projectName.trim()} type="submit">
                {submitting ? "Setting up…" : "Finish setup"}
              </button>
            </form>
            {createFailed ? (
              <div className="setup-wizard-recovery" role="alert">
                <p>
                  SceneWorks couldn&apos;t create the project in your current workspace folder — this
                  is almost always a permissions problem with that location. Choose a different
                  workspace folder, then restart to finish setup.
                </p>
                {isDesktop ? (
                  <button
                    className="setup-wizard-secondary"
                    onClick={changeWorkspaceFolder}
                    type="button"
                  >
                    Change workspace folder…
                  </button>
                ) : null}
                {workspaceNotice ? <p className="setup-wizard-notice">{workspaceNotice}</p> : null}
              </div>
            ) : null}
            <button className="setup-wizard-back" onClick={() => setStep("models")} type="button">
              ← Back to models
            </button>
          </>
        )}
      </div>
    </section>
  );
}
