import React, { useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { appConfirm } from "../appConfirm.jsx";
import { Icon } from "../components/Icons.jsx";
import { useAppStatic } from "../context/AppContext.js";
import { formatBytes } from "../formatting.js";
import { isDesktop, tauriInvoke } from "../runtime.js";

const EMPTY_CREATE = {
  name: "",
  path: "",
  sourceKind: "filesystem",
  sourcePath: "",
};

function safeCatalogError(error, action) {
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
  const candidates = Number(processing.candidateCount ?? 0);
  const processed = Number(processing.processedCount ?? 0);
  const percent = candidates > 0 ? Math.min(100, Math.round((processed / candidates) * 100)) : 0;
  return (
    <section className="catalog-detail-section" aria-labelledby="catalog-progress-title">
      <div className="catalog-section-heading">
        <h3 id="catalog-progress-title">Processing</h3>
        <span className={`catalog-state catalog-state--${processing.state ?? "unknown"}`}>
          {processingLabel(processing.state)}
        </span>
      </div>
      <div
        aria-label={`${percent}% processed`}
        aria-valuemax="100"
        aria-valuemin="0"
        aria-valuenow={percent}
        className="catalog-progress"
        role="progressbar"
      >
        <span style={{ width: `${percent}%` }} />
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
  const [error, setError] = useState("");
  const [createDraft, setCreateDraft] = useState(EMPTY_CREATE);
  const [attachPath, setAttachPath] = useState("");
  const selected = catalogs.find((catalog) => catalog.id === selectedId) ?? null;

  const persistSelection = useCallback((id) => {
    setSelectedId(id);
    apiFetch("/api/v1/ui-preferences", token, {
      method: "PUT",
      body: JSON.stringify({ selectedCatalogId: id }),
    }).catch(() => {});
  }, [token]);

  const refresh = useCallback(async ({ quiet = false, signal } = {}) => {
    if (!quiet) setLoading(true);
    try {
      const next = await apiFetch("/api/v1/catalogs", token, { signal });
      setCatalogs(Array.isArray(next) ? next : []);
      setSelectedId((current) => {
        if (next.some((catalog) => catalog.id === current)) return current;
        return next[0]?.id ?? "";
      });
      setError("");
      return next;
    } catch (err) {
      if (!isAbortError(err)) setError(safeCatalogError(err, "load catalogs"));
      return [];
    } finally {
      if (!quiet && !signal?.aborted) setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      apiFetch("/api/v1/ui-preferences", token, { signal: controller.signal }).catch(() => ({})),
      refresh({ signal: controller.signal }),
    ]).then(([preferences, next]) => {
      if (controller.signal.aborted) return;
      const preferred = preferences?.selectedCatalogId;
      if (preferred && next.some((catalog) => catalog.id === preferred)) setSelectedId(preferred);
    });
    return () => controller.abort();
  }, [refresh, token]);

  useEffect(() => {
    if (!selectedId) return undefined;
    const interval = window.setInterval(async () => {
      try {
        const updated = await apiFetch(`/api/v1/catalogs/${encodeURIComponent(selectedId)}/status`, token);
        setCatalogs((current) => current.map((catalog) => (catalog.id === updated.id ? updated : catalog)));
      } catch {
        // A background refresh is advisory. Explicit actions and manual reload surface errors.
      }
    }, 3000);
    return () => window.clearInterval(interval);
  }, [selectedId, token]);

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
    setBusy(action);
    setError("");
    try {
      const result = await request();
      const next = await refresh({ quiet: true });
      success?.(result, next);
    } catch (err) {
      setError(safeCatalogError(err, action));
    } finally {
      setBusy("");
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
      () => apiFetch("/api/v1/catalogs", token, { method: "POST", body: JSON.stringify(body) }),
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
      () => apiFetch("/api/v1/catalogs/attach", token, {
        method: "POST",
        body: JSON.stringify({ path: attachPath.trim() }),
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
      () => apiFetch(`/api/v1/catalogs/${encodeURIComponent(selected.id)}/${action}`, token, {
        method: "POST",
        body: JSON.stringify({}),
      }),
      (updated) => setCatalogs((current) => current.map((item) => item.id === updated.id ? updated : item)),
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
      () => apiFetch(`/api/v1/catalogs/${encodeURIComponent(selected.id)}`, token, { method: "DELETE" }),
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
      () => apiFetch(`/api/v1/catalogs/${encodeURIComponent(selected.id)}/on-disk`, token, { method: "DELETE" }),
      (_deleted, next) => persistSelection(next[0]?.id ?? ""),
    );
  }

  const analyzerEntries = useMemo(() => Object.entries(selected?.analyzerVersions ?? {}), [selected]);

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
                onChange={(event) => setCreateDraft((draft) => ({ ...draft, sourceKind: event.target.value }))}
                value={createDraft.sourceKind}
              >
                <option value="filesystem">Image folder</option>
                <option value="parquet">Parquet dataset folder</option>
              </select>
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
                    <button className="secondary-action" disabled={Boolean(busy)} onClick={() => changeProcessing("pause")} type="button">
                      <Icon.Pause /> Pause
                    </button>
                  ) : (
                    <button className="primary-action" disabled={Boolean(busy) || selected.availability !== "available"} onClick={() => changeProcessing("resume")} type="button">
                      <Icon.Play /> Resume
                    </button>
                  )}
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
