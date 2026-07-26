import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { appConfirm } from "../appConfirm.jsx";
import { Icon } from "../components/Icons.jsx";
import { useAppStatic } from "../context/AppContext.js";
import { formatBytes } from "../formatting.js";
import { isDesktop, tauriInvoke } from "../runtime.js";
import { persistNavigationPreferences } from "../uiPreferences.js";

const EMPTY_CREATE = {
  name: "",
  path: "",
  sourceKind: "parquet",
  sourcePath: "",
};

const DEFAULT_ANALYZER_SETTINGS = {
  structuredAnalysisEnabled: true,
  visionAnalysisEnabled: false,
  semanticEmbeddingsEnabled: false,
  thresholds: {
    personMinConfidence: 0.25,
    faceMinConfidence: 0.65,
    poseMinKeypointConfidence: 0.3,
    prominentFrameFraction: 0.2,
    frameEdgeMargin: 0.01,
    minPoseCoverage: 0.72,
  },
};

const ANALYZER_THRESHOLD_FIELDS = [
  ["personMinConfidence", "Person confidence", 1],
  ["faceMinConfidence", "Face confidence", 1],
  ["poseMinKeypointConfidence", "Pose keypoint confidence", null],
  ["prominentFrameFraction", "Prominent frame fraction", 1],
  ["frameEdgeMargin", "Frame edge margin", 0.25],
  ["minPoseCoverage", "Minimum pose coverage", 1],
];

function analyzerSettings(value) {
  return {
    ...DEFAULT_ANALYZER_SETTINGS,
    ...(value ?? {}),
    thresholds: {
      ...DEFAULT_ANALYZER_SETTINGS.thresholds,
      ...(value?.thresholds ?? {}),
    },
  };
}

function safeCatalogError(error, action) {
  if (error?.code === "catalog_analyzer_config_conflict") {
    return "Analyzer settings changed in another session. Refresh and try again.";
  }
  const message = String(error?.message ?? "").toLowerCase();
  if (message.includes("not found")) return "That catalog is no longer attached. Refresh the list and try again.";
  if (message.includes("already")) return "That folder is already attached as a catalog.";
  if (message.includes("absolute")) return "Choose or enter an absolute folder path.";
  if (message.includes("incompatible")) return "This catalog was created by an incompatible SceneWorks version.";
  if (message.includes("corrupt") || message.includes("invalid")) {
    return "The catalog metadata is invalid or damaged. Its files were not changed.";
  }
  return `Could not ${action}. Try again or check the API logs for details.`;
}

function processingLabel(state) {
  return {
    idle: "Idle",
    running: "Running",
    paused: "Paused",
    completed: "Complete",
    failed: "Needs attention",
  }[state] ?? "Unknown";
}

function count(value) {
  return Number(value ?? 0).toLocaleString();
}

function CatalogListItem({ catalog, selected, onSelect }) {
  const processing = catalog.processing ?? {};
  return (
    <button
      aria-current={selected ? "true" : undefined}
      className={`catalog-list-item${selected ? " selected" : ""}`}
      onClick={() => onSelect(catalog.id)}
      type="button"
    >
      <span className="catalog-list-main">
        <strong>{catalog.name}</strong>
        <span className="catalog-path" title={String(catalog.path ?? "")}>{String(catalog.path ?? "Path unavailable")}</span>
      </span>
      <span className={`catalog-state catalog-state--${processing.state ?? "unknown"}`}>
        {catalog.availability === "available" ? processingLabel(processing.state) : "Unavailable"}
      </span>
      <span className="catalog-list-metrics">
        {count(catalog.counts?.recordCount)} rows
        <span aria-hidden="true"> · </span>
        {count(catalog.counts?.processedCount)} analyzed
        <span aria-hidden="true"> · </span>
        {formatBytes(catalog.storage?.totalBytes)}
      </span>
    </button>
  );
}

function Progress({ catalog }) {
  const processing = catalog.processing ?? {};
  const desiredState = catalog.processingControl?.desiredState ?? "running";
  const candidates = Number(processing.candidateCount ?? 0);
  const processed = Number(processing.processedCount ?? 0);
  const percent =
    processing.state === "completed"
      ? 100
      : candidates > processed
        ? Math.min(100, Math.round((processed / candidates) * 100))
        : processing.state === "idle" && processed === 0
          ? 0
          : null;
  const progressLabel =
    percent === null ? `${count(processed)} processed; total unknown` : `${percent}% processed`;
  return (
    <section className="catalog-detail-section" aria-labelledby="catalog-progress-title">
      <div className="catalog-section-heading">
        <h3 id="catalog-progress-title">Processing</h3>
        <span className={`catalog-state catalog-state--${processing.state ?? "unknown"}`}>
          {processingLabel(processing.state)}
        </span>
      </div>
      <div
        aria-label={progressLabel}
        aria-valuemax="100"
        aria-valuemin="0"
        aria-valuenow={percent ?? undefined}
        className={`catalog-progress${percent === null ? " catalog-progress--indeterminate" : ""}`}
        role="progressbar"
      >
        <span style={percent === null ? undefined : { width: `${percent}%` }} />
      </div>
      <div className="catalog-metric-grid">
        <span><strong>{count(processing.candidateCount)}</strong>Candidates</span>
        <span><strong>{count(processing.processedCount)}</strong>Processed</span>
        <span><strong>{count(processing.acceptedCount)}</strong>Accepted</span>
        <span><strong>{count(processing.rejectedCount)}</strong>Rejected</span>
        <span><strong>{count(processing.errorCount)}</strong>Errors</span>
      </div>
      {processing.message ? (
        <p className={processing.state === "failed" || processing.errorCount ? "notice error" : "muted"}>
          {processing.message}
        </p>
      ) : null}
      {processing.state === "running" && desiredState === "paused" ? (
        <p className="muted">Pause requested. The active processor will stop after its current bounded batch.</p>
      ) : null}
      {processing.state === "paused" && desiredState === "running" ? (
        <p className="muted">Resume requested. Actual processing remains paused until a processor picks it up.</p>
      ) : null}
      <p className="field-hint">
        Updated {processing.updatedAt ? new Date(processing.updatedAt).toLocaleString() : "—"}
      </p>
    </section>
  );
}

export function DatasetCatalogsScreen() {
  const { token = "" } = useAppStatic();
  const [catalogs, setCatalogs] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [pollEpoch, setPollEpoch] = useState(0);
  const [error, setError] = useState("");
  const [createDraft, setCreateDraft] = useState(EMPTY_CREATE);
  const [attachPath, setAttachPath] = useState("");
  const [analyzerDraft, setAnalyzerDraft] = useState(DEFAULT_ANALYZER_SETTINGS);
  const generationRef = useRef(0);
  const analyzerDraftVersionRef = useRef("");
  const pollAbortRef = useRef(null);
  const mutationAbortRef = useRef(null);
  const selected = catalogs.find((catalog) => catalog.id === selectedId) ?? null;

  useEffect(() => {
    const version = selected
      ? `${selected.id}:${selected.analyzerConfig?.revision ?? 0}`
      : "";
    if (analyzerDraftVersionRef.current === version) return;
    analyzerDraftVersionRef.current = version;
    setAnalyzerDraft(analyzerSettings(selected?.analyzerConfig?.settings));
  }, [selected]);

  const persistSelection = useCallback((id) => {
    generationRef.current += 1;
    pollAbortRef.current?.abort();
    mutationAbortRef.current?.abort();
    setBusy("");
    setSelectedId(id);
    persistNavigationPreferences({ selectedCatalogId: id }, token);
  }, [token]);

  const refresh = useCallback(async ({ quiet = false, signal, generation = generationRef.current } = {}) => {
    if (!quiet) setLoading(true);
    try {
      const next = await apiFetch("/api/v1/catalogs", token, { signal });
      if (signal?.aborted || generation !== generationRef.current) return next;
      setCatalogs(Array.isArray(next) ? next : []);
      setSelectedId((current) => {
        if (next.some((catalog) => catalog.id === current)) return current;
        return next[0]?.id ?? "";
      });
      setError("");
      return next;
    } catch (err) {
      if (!isAbortError(err) && generation === generationRef.current) {
        setError(safeCatalogError(err, "load catalogs"));
      }
      return [];
    } finally {
      if (!quiet && !signal?.aborted && generation === generationRef.current) setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    const generation = ++generationRef.current;
    const controller = new AbortController();
    Promise.all([
      apiFetch("/api/v1/ui-preferences", token, { signal: controller.signal }).catch(() => ({})),
      refresh({ signal: controller.signal, generation }),
    ]).then(([preferences, next]) => {
      if (controller.signal.aborted || generation !== generationRef.current) return;
      const preferred = preferences?.selectedCatalogId;
      if (preferred && next.some((catalog) => catalog.id === preferred)) setSelectedId(preferred);
    });
    return () => {
      generationRef.current += 1;
      controller.abort();
      pollAbortRef.current?.abort();
      mutationAbortRef.current?.abort();
    };
  }, [refresh, token]);

  useEffect(() => {
    if (!selectedId) return undefined;
    const generation = ++generationRef.current;
    let timer = null;
    let stopped = false;
    const poll = async () => {
      const controller = new AbortController();
      pollAbortRef.current?.abort();
      pollAbortRef.current = controller;
      try {
        const updated = await apiFetch(
          `/api/v1/catalogs/${encodeURIComponent(selectedId)}/status`,
          token,
          { signal: controller.signal },
        );
        if (stopped || controller.signal.aborted || generation !== generationRef.current) return;
        setCatalogs((current) => current.map((catalog) => (catalog.id === updated.id ? updated : catalog)));
        setError("");
      } catch (err) {
        if (isAbortError(err) || stopped || generation !== generationRef.current) return;
        if (err.status === 404) {
          setError("That catalog is no longer attached. The catalog list was refreshed.");
          await refresh({ quiet: true, generation });
        } else {
          setError(safeCatalogError(err, "refresh catalog status"));
        }
      } finally {
        if (!stopped && generation === generationRef.current) {
          timer = window.setTimeout(poll, 3000);
        }
      }
    };
    timer = window.setTimeout(poll, 3000);
    return () => {
      stopped = true;
      generationRef.current += 1;
      if (timer !== null) window.clearTimeout(timer);
      pollAbortRef.current?.abort();
    };
  }, [pollEpoch, refresh, selectedId, token]);

  async function chooseFolder(setter) {
    if (!isDesktop) return;
    try {
      const chosen = await tauriInvoke("choose_folder");
      if (chosen) setter(chosen);
    } catch (err) {
      setError(safeCatalogError(err, "open the folder picker"));
    }
  }

  async function mutate(action, request, success) {
    const generation = ++generationRef.current;
    pollAbortRef.current?.abort();
    mutationAbortRef.current?.abort();
    const controller = new AbortController();
    mutationAbortRef.current = controller;
    setBusy(action);
    setError("");
    try {
      const result = await request(controller.signal);
      if (controller.signal.aborted || generation !== generationRef.current) return;
      const next = await refresh({ quiet: true, signal: controller.signal, generation });
      if (controller.signal.aborted || generation !== generationRef.current) return;
      success?.(result, next);
    } catch (err) {
      if (!isAbortError(err) && generation === generationRef.current) {
        setError(safeCatalogError(err, action));
      }
    } finally {
      if (generation === generationRef.current) {
        setBusy("");
        setPollEpoch((current) => current + 1);
      }
    }
  }

  async function createCatalog(event) {
    event.preventDefault();
    const body = {
      name: createDraft.name.trim(),
      path: createDraft.path.trim(),
    };
    if (createDraft.sourcePath.trim()) {
      body.sourceConfig = {
        kind: createDraft.sourceKind,
        paths: [createDraft.sourcePath.trim()],
        options: {},
      };
    }
    await mutate(
      "create the catalog",
      (signal) => apiFetch("/api/v1/catalogs", token, {
        method: "POST",
        body: JSON.stringify(body),
        signal,
      }),
      (created) => {
        setCreateDraft(EMPTY_CREATE);
        persistSelection(created.id);
      },
    );
  }

  async function attachCatalog(event) {
    event.preventDefault();
    await mutate(
      "attach the catalog",
      (signal) => apiFetch("/api/v1/catalogs/attach", token, {
        method: "POST",
        body: JSON.stringify({ path: attachPath.trim() }),
        signal,
      }),
      (attached) => {
        setAttachPath("");
        persistSelection(attached.id);
      },
    );
  }

  async function changeProcessing(action) {
    if (!selected) return;
    await mutate(
      `${action} processing`,
      (signal) => apiFetch(`/api/v1/catalogs/${encodeURIComponent(selected.id)}/${action}`, token, {
        method: "POST",
        body: JSON.stringify({
          expectedRevision: selected.processingControl?.revision ?? 0,
        }),
        signal,
      }),
      (updated) => setCatalogs((current) => current.map((item) => item.id === updated.id ? updated : item)),
    );
  }

  async function saveAnalyzerConfig(event) {
    event.preventDefault();
    if (!selected) return;
    await mutate(
      "save analyzer settings",
      (signal) => apiFetch(
        `/api/v1/catalogs/${encodeURIComponent(selected.id)}/analyzer-config`,
        token,
        {
          method: "PUT",
          body: JSON.stringify({
            expectedRevision: selected.analyzerConfig?.revision ?? 0,
            settings: analyzerDraft,
          }),
          signal,
        },
      ),
      (updated) => setCatalogs((current) =>
        current.map((item) => item.id === updated.id ? updated : item)),
    );
  }

  async function detach() {
    if (!selected || !(await appConfirm({
      title: "Detach catalog?",
      message: `"${selected.name}" will disappear from SceneWorks, but every catalog file will remain on disk.`,
      confirmLabel: "Detach catalog",
    }))) return;
    await mutate(
      "detach the catalog",
      (signal) => apiFetch(`/api/v1/catalogs/${encodeURIComponent(selected.id)}`, token, {
        method: "DELETE",
        signal,
      }),
      (_detached, next) => persistSelection(next[0]?.id ?? ""),
    );
  }

  async function deleteOnDisk() {
    if (!selected || !(await appConfirm({
      title: "Delete catalog files?",
      message: `Permanently delete "${selected.name}" and its database, manifest, and generated artifacts from disk? Source images are not part of the catalog and are not deleted. This cannot be undone.`,
      confirmLabel: "Delete files permanently",
      tone: "danger",
    }))) return;
    await mutate(
      "delete the catalog files",
      (signal) => apiFetch(`/api/v1/catalogs/${encodeURIComponent(selected.id)}/on-disk`, token, {
        method: "DELETE",
        signal,
      }),
      (_deleted, next) => persistSelection(next[0]?.id ?? ""),
    );
  }

  const analyzerEntries = useMemo(() => Object.entries(selected?.analyzerVersions ?? {}), [selected]);
  const selectedSourceIsSchedulable = selected?.sourceConfig?.kind === "parquet"
    && selected.sourceConfig.paths?.length === 1;

  return (
    <div className="dataset-catalogs-screen">
      <section className="work-panel catalog-create-panel">
        <div className="work-panel-rule" />
        <div className="work-panel-head">
          <div className="work-panel-head-text">
            <span className="work-panel-eyebrow">Global data library</span>
            <h2>Create or attach a catalog</h2>
            <p className="work-panel-hint">Catalogs live outside projects and can index very large datasets in place.</p>
          </div>
          <button className="secondary-action" disabled={loading} onClick={() => refresh()} type="button">
            <Icon.Refresh /> Refresh
          </button>
        </div>
        <div className="catalog-forms">
          <form aria-label="Create catalog" onSubmit={createCatalog}>
            <h3>New catalog</h3>
            <label className="settings-field">
              <span>Name</span>
              <input
                onChange={(event) => setCreateDraft((draft) => ({ ...draft, name: event.target.value }))}
                placeholder="Product photography"
                required
                value={createDraft.name}
              />
            </label>
            <label className="settings-field">
              <span>Catalog folder</span>
              <span className="catalog-path-input">
                <input
                  aria-label="Catalog folder"
                  onChange={(event) => setCreateDraft((draft) => ({ ...draft, path: event.target.value }))}
                  placeholder="Absolute folder path"
                  required
                  value={createDraft.path}
                />
                {isDesktop ? (
                  <button aria-label="Choose catalog folder" onClick={() => chooseFolder((path) => setCreateDraft((draft) => ({ ...draft, path })))} type="button">
                    <Icon.Folder />
                  </button>
                ) : null}
              </span>
            </label>
            <label className="settings-field">
              <span>Source type</span>
              <select
                aria-describedby="catalog-source-support"
                onChange={(event) => setCreateDraft((draft) => ({ ...draft, sourceKind: event.target.value }))}
                value={createDraft.sourceKind}
              >
                <option value="parquet">Parquet dataset folder</option>
              </select>
              <small className="muted" id="catalog-source-support">Automatic catalog processing currently supports Parquet sources.</small>
            </label>
            <label className="settings-field">
              <span>Source folder <small>(optional)</small></span>
              <span className="catalog-path-input">
                <input
                  aria-label="Source folder"
                  onChange={(event) => setCreateDraft((draft) => ({ ...draft, sourcePath: event.target.value }))}
                  placeholder="Absolute source path"
                  value={createDraft.sourcePath}
                />
                {isDesktop ? (
                  <button aria-label="Choose source folder" onClick={() => chooseFolder((path) => setCreateDraft((draft) => ({ ...draft, sourcePath: path })))} type="button">
                    <Icon.Folder />
                  </button>
                ) : null}
              </span>
            </label>
            <button className="primary-action" disabled={Boolean(busy) || !createDraft.name.trim() || !createDraft.path.trim()} type="submit">
              Create catalog
            </button>
          </form>
          <form aria-label="Attach catalog" onSubmit={attachCatalog}>
            <h3>Existing catalog</h3>
            <p className="muted">Attach a SceneWorks catalog folder without moving or copying it.</p>
            <label className="settings-field">
              <span>Catalog folder</span>
              <span className="catalog-path-input">
                <input
                  aria-label="Existing catalog folder"
                  onChange={(event) => setAttachPath(event.target.value)}
                  placeholder="Absolute folder path"
                  required
                  value={attachPath}
                />
                {isDesktop ? (
                  <button aria-label="Choose existing catalog folder" onClick={() => chooseFolder(setAttachPath)} type="button">
                    <Icon.Folder />
                  </button>
                ) : null}
              </span>
            </label>
            <button className="secondary-action strong" disabled={Boolean(busy) || !attachPath.trim()} type="submit">
              Attach catalog
            </button>
          </form>
        </div>
      </section>

      {error ? <p className="notice error" role="alert">{error}</p> : null}

      <section className="catalog-workspace">
        <div className="catalog-list" aria-label="Dataset catalogs">
          <div className="catalog-list-heading">
            <h2>Catalogs</h2>
            <span>{catalogs.length}</span>
          </div>
          {loading ? <p className="muted">Loading catalogs…</p> : null}
          {!loading && catalogs.length === 0 ? (
            <div className="empty-state">
              <Icon.Folder size={28} />
              <h3>No catalogs attached</h3>
              <p>Create one above. No workspace or training dataset is required.</p>
            </div>
          ) : null}
          {catalogs.map((catalog) => (
            <CatalogListItem
              catalog={catalog}
              key={catalog.id}
              onSelect={persistSelection}
              selected={catalog.id === selectedId}
            />
          ))}
        </div>

        <div className="catalog-detail">
          {!selected ? (
            <div className="empty-state">
              <h3>Select a catalog</h3>
              <p>Choose a catalog to inspect its source, analyzers, progress, and storage.</p>
            </div>
          ) : (
            <>
              <header className="catalog-detail-header">
                <div>
                  <span className="work-panel-eyebrow">Dataset catalog</span>
                  <h2>{selected.name}</h2>
                  <p className="catalog-path" title={String(selected.path)}>{String(selected.path)}</p>
                </div>
                <div className="catalog-detail-actions">
                  {selected.processing?.state === "running" ? (
                    <button
                      className="secondary-action"
                      disabled={Boolean(busy) || selected.processingControl?.desiredState === "paused"}
                      onClick={() => changeProcessing("pause")}
                      type="button"
                    >
                      <Icon.Pause /> {selected.processingControl?.desiredState === "paused" ? "Pause requested" : "Pause"}
                    </button>
                  ) : selected.processing?.state === "paused" ? (
                    <button
                      className="primary-action"
                      disabled={Boolean(busy) || selected.availability !== "available" || selected.processingControl?.desiredState === "running"}
                      onClick={() => changeProcessing("resume")}
                      type="button"
                    >
                      <Icon.Play /> {selected.processingControl?.desiredState === "running" ? "Resume requested" : "Resume"}
                    </button>
                  ) : selected.processing?.state === "failed" ? (
                    <button
                      className="primary-action"
                      disabled={Boolean(busy) || selected.availability !== "available" || !selectedSourceIsSchedulable}
                      onClick={() => changeProcessing("resume")}
                      type="button"
                    >
                      <Icon.Play /> Restart
                    </button>
                  ) : null}
                </div>
              </header>

              {selected.availability !== "available" ? (
                <p className="notice error">This catalog is unavailable at its attached location. Detach it, then attach its new folder if it was moved.</p>
              ) : null}

              <Progress catalog={selected} />

              <section className="catalog-detail-section" aria-labelledby="catalog-source-title">
                <h3 id="catalog-source-title">Source configuration</h3>
                {selected.sourceConfig ? (
                  <dl className="catalog-definition-list">
                    <div><dt>Kind</dt><dd>{selected.sourceConfig.kind}</dd></div>
                    <div><dt>Paths</dt><dd>{selected.sourceConfig.paths?.map((path) => <code key={path}>{path}</code>)}</dd></div>
                    <div><dt>Options</dt><dd><code>{JSON.stringify(selected.sourceConfig.options ?? {})}</code></dd></div>
                  </dl>
                ) : <p className="muted">No source has been configured yet.</p>}
              </section>

              <section className="catalog-detail-section" aria-labelledby="catalog-analyzers-title">
                <h3 id="catalog-analyzers-title">Analyzer configuration</h3>
                <form className="catalog-analyzer-form" onSubmit={saveAnalyzerConfig}>
                  <div className="catalog-analyzer-toggles">
                    {[
                      ["structuredAnalysisEnabled", "Structured person, face, and pose analysis"],
                      ["visionAnalysisEnabled", "Vision classification and tags"],
                      ["semanticEmbeddingsEnabled", "Semantic embeddings"],
                    ].map(([field, label]) => (
                      <label key={field}>
                        <input
                          checked={Boolean(analyzerDraft[field])}
                          disabled={selected.availability !== "available"}
                          onChange={(event) => setAnalyzerDraft((current) => ({
                            ...current,
                            [field]: event.target.checked,
                          }))}
                          type="checkbox"
                        />
                        <span>{label}</span>
                      </label>
                    ))}
                  </div>
                  <div className="catalog-analyzer-thresholds">
                    {ANALYZER_THRESHOLD_FIELDS.map(([field, label, max]) => (
                      <label className="settings-field" key={field}>
                        <span>{label}</span>
                        <input
                          aria-label={label}
                          disabled={selected.availability !== "available"}
                          max={max ?? undefined}
                          min="0"
                          onChange={(event) => setAnalyzerDraft((current) => ({
                            ...current,
                            thresholds: {
                              ...current.thresholds,
                              [field]: Number(event.target.value),
                            },
                          }))}
                          required
                          step="0.01"
                          type="number"
                          value={analyzerDraft.thresholds[field]}
                        />
                      </label>
                    ))}
                  </div>
                  <p className="field-hint">
                    Saved per catalog for the next analysis run. Changing settings does not start processing.
                  </p>
                  <button
                    className="secondary-action strong"
                    disabled={Boolean(busy) || selected.availability !== "available"}
                    type="submit"
                  >
                    Save analyzer settings
                  </button>
                </form>
                <h4>Recorded analyzer versions</h4>
                {analyzerEntries.length ? (
                  <dl className="catalog-definition-list">
                    {analyzerEntries.map(([name, version]) => <div key={name}><dt>{name}</dt><dd><code>{version}</code></dd></div>)}
                  </dl>
                ) : <p className="muted">No analyzers have recorded a version yet.</p>}
              </section>

              <section className="catalog-detail-section">
                <h3>Storage</h3>
                <div className="catalog-metric-grid">
                  <span><strong>{formatBytes(selected.storage?.databaseBytes)}</strong>Database</span>
                  <span><strong>{formatBytes(selected.storage?.manifestBytes)}</strong>Manifest</span>
                  <span><strong>{formatBytes(selected.storage?.artifactBytes)}</strong>Artifacts</span>
                  <span><strong>{formatBytes(selected.storage?.totalBytes)}</strong>Total</span>
                </div>
              </section>

              <section className="catalog-danger-zone" aria-labelledby="catalog-danger-title">
                <div>
                  <h3 id="catalog-danger-title">Catalog lifecycle</h3>
                  <p>Detach keeps every file. Delete on disk permanently removes the catalog database and generated artifacts.</p>
                </div>
                <div className="catalog-detail-actions">
                  <button className="secondary-action" disabled={Boolean(busy)} onClick={detach} type="button">Detach</button>
                  <button className="danger-action" disabled={Boolean(busy)} onClick={deleteOnDisk} type="button">Delete on disk</button>
                </div>
              </section>
            </>
          )}
        </div>
      </section>
    </div>
  );
}
