import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const navItems = ["Library", "Image", "Video", "Characters", "Editor", "Queue"];
const assetTypes = [
  ["", "All"],
  ["image", "Images"],
  ["video", "Videos"],
  ["upload", "Uploads"],
  ["frame", "Frames"],
  ["render", "Renders"],
  ["character_reference", "Characters"],
];
const sortOptions = [
  ["newest", "Newest"],
  ["rating", "Rating"],
  ["type", "Type"],
  ["name", "Name"],
];

const placeholders = {
  Image: "Prompt workflows will reuse selected Library assets here.",
  Video: "Generated shots will land in the Library and asset tray.",
  Characters: "Character references will be project assets with approved picks.",
  Editor: "Timeline items will reference assets by id, not by file path.",
  Queue: "Generation, imports, exports, and repair work will share job state.",
};

async function readError(response) {
  const detail = await response.json().catch(() => ({}));
  if (Array.isArray(detail.detail)) {
    return detail.detail.map((item) => item.msg).join("; ");
  }
  return detail.detail ?? `Request failed with ${response.status}`;
}

async function apiJson(path, token, options = {}) {
  const headers = new Headers(options.headers ?? {});
  headers.set("Content-Type", "application/json");
  if (token) {
    headers.set("X-SceneWorks-Token", token);
  }
  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.status === 204 ? null : response.json();
}

async function apiForm(path, token, formData) {
  const headers = new Headers();
  if (token) {
    headers.set("X-SceneWorks-Token", token);
  }
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers,
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

function StatusDot({ ok }) {
  return <span className={ok ? "status-dot ok" : "status-dot"} aria-hidden="true" />;
}

function assetUrl(asset) {
  return `${API_BASE_URL}${asset.previewUrl}`;
}

function AssetPreview({ asset, size = "normal" }) {
  if (!asset) {
    return <div className={`asset-preview ${size}`} />;
  }
  if (asset.type === "image" || asset.type === "frame" || asset.type === "render" || asset.type === "character_reference") {
    return <img className={`asset-preview ${size}`} src={assetUrl(asset)} alt={asset.displayName} />;
  }
  if (asset.type === "video") {
    return <video className={`asset-preview ${size}`} src={assetUrl(asset)} muted controls={size === "large"} />;
  }
  return (
    <div className={`asset-preview ${size} file-preview`}>
      <span>{asset.file.mimeType}</span>
    </div>
  );
}

function RatingControl({ value, onChange }) {
  return (
    <div className="segmented compact" aria-label="Rating">
      {[0, 1, 2, 3, 4, 5].map((rating) => (
        <button
          className={value === rating ? "active" : ""}
          key={rating}
          onClick={() => onChange(rating)}
          type="button"
        >
          {rating}
        </button>
      ))}
    </div>
  );
}

function App() {
  const [health, setHealth] = useState(null);
  const [access, setAccess] = useState({ authRequired: false });
  const [token, setToken] = useState(() => window.localStorage.getItem("sceneworks-token") ?? "");
  const [projects, setProjects] = useState([]);
  const [activeProject, setActiveProject] = useState(null);
  const [assets, setAssets] = useState([]);
  const [selectedAssetId, setSelectedAssetId] = useState(null);
  const [activeView, setActiveView] = useState("Library");
  const [projectName, setProjectName] = useState("");
  const [openPath, setOpenPath] = useState("");
  const [assetType, setAssetType] = useState("");
  const [sort, setSort] = useState("newest");
  const [search, setSearch] = useState("");
  const [showRejected, setShowRejected] = useState(false);
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [trayOpen, setTrayOpen] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const authenticated = useMemo(() => !access.authRequired || token.length > 0, [access, token]);
  const selectedAsset = useMemo(
    () => assets.find((asset) => asset.id === selectedAssetId) ?? assets[0] ?? null,
    [assets, selectedAssetId],
  );

  useEffect(() => {
    apiJson("/api/v1/health", "")
      .then(setHealth)
      .catch((err) => setError(err.message));
    apiJson("/api/v1/access", "")
      .then(setAccess)
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!authenticated) {
      return;
    }
    loadProjects();
  }, [authenticated, token]);

  useEffect(() => {
    if (!activeProject || !authenticated) {
      setAssets([]);
      return;
    }
    loadAssets(activeProject.id);
  }, [activeProject?.id, assetType, sort, showRejected, favoritesOnly, authenticated]);

  async function loadProjects() {
    try {
      const items = await apiJson("/api/v1/projects", token);
      setProjects(items);
      setActiveProject((current) => current ?? items[0] ?? null);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadAssets(projectId = activeProject?.id) {
    if (!projectId) {
      return;
    }
    const params = new URLSearchParams({
      sort,
      includeRejected: String(showRejected),
      favoritesOnly: String(favoritesOnly),
    });
    if (assetType) {
      params.set("type", assetType);
    }
    if (search.trim()) {
      params.set("search", search.trim());
    }
    try {
      const payload = await apiJson(`/api/v1/projects/${projectId}/assets?${params}`, token);
      setAssets(payload.assets);
      setSelectedAssetId((current) => (payload.assets.some((asset) => asset.id === current) ? current : payload.assets[0]?.id ?? null));
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  function saveToken(event) {
    event.preventDefault();
    window.localStorage.setItem("sceneworks-token", token);
    setError("");
    loadProjects();
  }

  async function createProject(event) {
    event.preventDefault();
    if (!projectName.trim()) {
      return;
    }
    setBusy(true);
    try {
      const created = await apiJson("/api/v1/projects", token, {
        method: "POST",
        body: JSON.stringify({ name: projectName }),
      });
      setProjects((items) => [created, ...items.filter((item) => item.id !== created.id)]);
      setActiveProject(created);
      setProjectName("");
      setActiveView("Library");
      setError("");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function openProject(event) {
    event.preventDefault();
    if (!openPath.trim()) {
      return;
    }
    setBusy(true);
    try {
      const opened = await apiJson("/api/v1/projects/open", token, {
        method: "POST",
        body: JSON.stringify({ path: openPath }),
      });
      setProjects((items) => [opened, ...items.filter((item) => item.id !== opened.id)]);
      setActiveProject(opened);
      setOpenPath("");
      setActiveView("Library");
      setError("");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function importAssets(event) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (!activeProject || files.length === 0) {
      return;
    }
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    setBusy(true);
    try {
      const payload = await apiForm(`/api/v1/projects/${activeProject.id}/assets/import`, token, formData);
      await loadAssets(activeProject.id);
      setSelectedAssetId(payload.assets[0]?.id ?? null);
      await loadProjects();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function patchAsset(asset, patch) {
    try {
      const updated = await apiJson(`/api/v1/projects/${activeProject.id}/assets/${asset.id}`, token, {
        method: "PATCH",
        body: JSON.stringify(patch),
      });
      setAssets((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function trashSelected() {
    if (!selectedAsset || !activeProject) {
      return;
    }
    setBusy(true);
    try {
      await apiJson(`/api/v1/projects/${activeProject.id}/assets/${selectedAsset.id}`, token, { method: "DELETE" });
      await loadAssets(activeProject.id);
      await loadProjects();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function reindexProject() {
    if (!activeProject) {
      return;
    }
    setBusy(true);
    try {
      const result = await apiJson(`/api/v1/projects/${activeProject.id}/reindex`, token, { method: "POST" });
      await loadAssets(activeProject.id);
      setError(result.errors.length ? result.errors.join("; ") : `Reindexed ${result.indexed} assets`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const workspaceClass = trayOpen ? "workspace with-tray" : "workspace";

  return (
    <main className="app">
      <aside className="sidebar" aria-label="Primary">
        <div className="brand">
          <span className="brand-mark">SW</span>
          <div>
            <h1>SceneWorks</h1>
            <p>Local creative studio</p>
          </div>
        </div>

        <nav className="nav-list">
          {navItems.map((item) => (
            <button
              className={activeView === item ? "nav-item active" : "nav-item"}
              key={item}
              onClick={() => setActiveView(item)}
              type="button"
            >
              {item}
            </button>
          ))}
        </nav>
      </aside>

      <section className={workspaceClass}>
        <header className="topbar">
          <div className="topbar-project">
            <p className="eyebrow">Project</p>
            <strong>{activeProject?.name ?? "No project open"}</strong>
            {activeProject ? <span>{activeProject.path}</span> : null}
          </div>
          <div className="topbar-status">
            <span>
              <StatusDot ok={health?.status === "ok"} />
              API
            </span>
            <span>{busy ? "Working" : "Idle"}</span>
            <button className="icon-button" onClick={() => setTrayOpen((value) => !value)} type="button">
              {trayOpen ? "Hide tray" : "Show tray"}
            </button>
          </div>
        </header>

        {error ? <p className={error.startsWith("Reindexed") ? "notice ok" : "notice error"}>{error}</p> : null}

        {access.authRequired && !window.localStorage.getItem("sceneworks-token") ? (
          <section className="auth-band">
            <form onSubmit={saveToken}>
              <label htmlFor="token">Pairing token</label>
              <div className="form-row">
                <input
                  id="token"
                  onChange={(event) => setToken(event.target.value)}
                  placeholder="Enter local token"
                  type="password"
                  value={token}
                />
                <button type="submit">Unlock</button>
              </div>
            </form>
          </section>
        ) : null}

        <section className="project-band">
          <div className="project-list">
            <div className="section-heading">
              <p className="eyebrow">Recent projects</p>
              <h2>Projects</h2>
            </div>
            <div className="project-buttons">
              {projects.length === 0 ? (
                <span className="empty-state">No recent projects</span>
              ) : (
                projects.map((project) => (
                  <button
                    className={activeProject?.id === project.id ? "project-pill active" : "project-pill"}
                    key={project.id}
                    onClick={() => {
                      setActiveProject(project);
                      setActiveView("Library");
                    }}
                    type="button"
                  >
                    <span>{project.name}</span>
                    <small>{project.assetCount} assets</small>
                  </button>
                ))
              )}
            </div>
          </div>

          <div className="project-forms">
            <form className="compact-form" onSubmit={createProject}>
              <label htmlFor="project-name">New project</label>
              <div className="form-row">
                <input
                  id="project-name"
                  onChange={(event) => setProjectName(event.target.value)}
                  placeholder="Noir Alley"
                  value={projectName}
                />
                <button disabled={!authenticated || busy} type="submit">
                  Create
                </button>
              </div>
            </form>
            <form className="compact-form" onSubmit={openProject}>
              <label htmlFor="project-path">Open folder</label>
              <div className="form-row">
                <input
                  id="project-path"
                  onChange={(event) => setOpenPath(event.target.value)}
                  placeholder="D:\\Projects\\Noir Alley.sceneworks"
                  value={openPath}
                />
                <button disabled={!authenticated || busy} type="submit">
                  Open
                </button>
              </div>
            </form>
          </div>
        </section>

        {activeView === "Library" ? (
          <section className="library-shell">
            <div className="library-main">
              <div className="library-toolbar">
                <div className="section-heading">
                  <p className="eyebrow">Library</p>
                  <h2>{assets.length} assets</h2>
                </div>
                <div className="toolbar-controls">
                  <label className="file-button">
                    Import
                    <input accept="image/*,video/*" disabled={!activeProject || busy} multiple onChange={importAssets} type="file" />
                  </label>
                  <button disabled={!activeProject || busy} onClick={reindexProject} type="button">
                    Reindex
                  </button>
                </div>
              </div>

              <div className="filters">
                <div className="segmented">
                  {assetTypes.map(([value, label]) => (
                    <button className={assetType === value ? "active" : ""} key={value || "all"} onClick={() => setAssetType(value)} type="button">
                      {label}
                    </button>
                  ))}
                </div>
                <input
                  aria-label="Search assets"
                  onChange={(event) => setSearch(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      loadAssets();
                    }
                  }}
                  placeholder="Search prompt, model, notes"
                  value={search}
                />
                <select aria-label="Sort assets" onChange={(event) => setSort(event.target.value)} value={sort}>
                  {sortOptions.map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
                <label className="toggle">
                  <input checked={showRejected} onChange={(event) => setShowRejected(event.target.checked)} type="checkbox" />
                  Rejected
                </label>
                <label className="toggle">
                  <input checked={favoritesOnly} onChange={(event) => setFavoritesOnly(event.target.checked)} type="checkbox" />
                  Favorites
                </label>
              </div>

              <div className="asset-grid" aria-label="Project assets">
                {assets.length === 0 ? (
                  <div className="empty-panel">No matching assets</div>
                ) : (
                  assets.map((asset) => (
                    <button
                      className={selectedAsset?.id === asset.id ? "asset-card active" : "asset-card"}
                      key={asset.id}
                      onClick={() => setSelectedAssetId(asset.id)}
                      type="button"
                    >
                      <AssetPreview asset={asset} />
                      <span>{asset.displayName}</span>
                      <small>
                        {asset.type} {asset.status.rating ? `${asset.status.rating}/5` : ""}
                      </small>
                    </button>
                  ))
                )}
              </div>
            </div>

            <aside className="detail-panel" aria-label="Asset detail">
              {selectedAsset ? (
                <>
                  <AssetPreview asset={selectedAsset} size="large" />
                  <div className="detail-heading">
                    <input
                      aria-label="Asset display name"
                      onBlur={(event) => {
                        if (event.target.value !== selectedAsset.displayName) {
                          patchAsset(selectedAsset, { displayName: event.target.value });
                        }
                      }}
                      defaultValue={selectedAsset.displayName}
                      key={selectedAsset.id}
                    />
                    <span>{selectedAsset.file.path}</span>
                  </div>
                  <div className="detail-actions">
                    <button
                      className={selectedAsset.status.favorite ? "active" : ""}
                      onClick={() => patchAsset(selectedAsset, { favorite: !selectedAsset.status.favorite })}
                      type="button"
                    >
                      Favorite
                    </button>
                    <button
                      className={selectedAsset.status.rejected ? "active danger" : ""}
                      onClick={() => patchAsset(selectedAsset, { rejected: !selectedAsset.status.rejected })}
                      type="button"
                    >
                      Reject
                    </button>
                    <button className="danger" disabled={busy} onClick={trashSelected} type="button">
                      Trash
                    </button>
                  </div>
                  <RatingControl value={selectedAsset.status.rating} onChange={(rating) => patchAsset(selectedAsset, { rating })} />
                  <textarea
                    aria-label="Asset notes"
                    defaultValue={selectedAsset.notes}
                    key={`${selectedAsset.id}-notes`}
                    onBlur={(event) => patchAsset(selectedAsset, { notes: event.target.value })}
                    placeholder="Notes"
                  />
                  <dl className="metadata-list">
                    <div>
                      <dt>Type</dt>
                      <dd>{selectedAsset.type}</dd>
                    </div>
                    <div>
                      <dt>Mime</dt>
                      <dd>{selectedAsset.file.mimeType}</dd>
                    </div>
                    <div>
                      <dt>Size</dt>
                      <dd>{Math.max(1, Math.round(selectedAsset.file.sizeBytes / 1024))} KB</dd>
                    </div>
                    <div>
                      <dt>Prompt</dt>
                      <dd>{selectedAsset.recipe?.prompt ?? "None"}</dd>
                    </div>
                    <div>
                      <dt>Model</dt>
                      <dd>{selectedAsset.recipe?.model ?? "None"}</dd>
                    </div>
                    <div>
                      <dt>Lineage</dt>
                      <dd>{selectedAsset.lineage.parents.length ? selectedAsset.lineage.parents.join(", ") : "None"}</dd>
                    </div>
                  </dl>
                  <div className="send-row">
                    <button type="button">Reuse</button>
                    <button type="button">Video</button>
                    <button type="button">Editor</button>
                  </div>
                </>
              ) : (
                <div className="empty-panel">Select an asset</div>
              )}
            </aside>
          </section>
        ) : (
          <section className="main-surface">
            <div className="section-heading">
              <p className="eyebrow">{activeView}</p>
              <h2>{activeView}</h2>
            </div>
            <p className="view-copy">{placeholders[activeView]}</p>
          </section>
        )}
      </section>

      {trayOpen ? (
        <aside className="asset-tray" aria-label="Asset tray">
          <div className="section-heading">
            <p className="eyebrow">Tray</p>
            <h2>Recent</h2>
          </div>
          <div className="tray-list">
            {assets.slice(0, 8).map((asset) => (
              <button
                className={selectedAsset?.id === asset.id ? "tray-item active" : "tray-item"}
                key={asset.id}
                onClick={() => {
                  setActiveView("Library");
                  setSelectedAssetId(asset.id);
                }}
                type="button"
              >
                <AssetPreview asset={asset} size="thumb" />
                <span>{asset.displayName}</span>
              </button>
            ))}
            {assets.length === 0 ? <span className="empty-state">No assets</span> : null}
          </div>
        </aside>
      ) : null}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
