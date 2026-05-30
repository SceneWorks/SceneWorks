import React, { useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { AssetDetail, AssetGrid, emptyTrash } from "../components/assetPanels.jsx";
import { useAppContext } from "../context/AppContext.js";
import { GLOBAL_POSES_PROJECT_ID } from "../poseLibrary.js";

// The Pose Library screen (epic 2282). Two tabs:
//  - "Poses": manage the global pose store — user-created type:"pose" assets in the
//    reserved project, as an image grid + viewer + Trashcan (reusing the shared asset
//    panels). Built-in poses stay bundled (read-only) and surface in the generation
//    pose pickers, not here.
//  - "Create": photo -> DWPose -> categorize -> save (sc-2287; placeholder for now).
// The reserved project is hidden from the project switcher, so these assets never
// appear in the Assets/Character views; we address it directly here.
const TABS = [
  ["poses", "Poses"],
  ["create", "Create"],
];

const UNCATEGORIZED = "uncategorized";

export function PoseLibraryScreen() {
  const { token } = useAppContext();
  const [activeTab, setActiveTab] = useState("poses");
  const [poses, setPoses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [assetMode, setAssetMode] = useState("assets");
  const [categoryFilter, setCategoryFilter] = useState("all");

  const refresh = useCallback(
    async (signal) => {
      try {
        setLoading(true);
        const items = await apiFetch(
          `/api/v1/projects/${GLOBAL_POSES_PROJECT_ID}/assets?includeRejected=true&includeTrashed=true`,
          token,
          signal ? { signal } : {},
        );
        setPoses((Array.isArray(items) ? items : []).filter((asset) => asset.type === "pose"));
        setError("");
      } catch (err) {
        if (isAbortError(err)) return;
        setError(String(err?.message ?? err));
      } finally {
        setLoading(false);
      }
    },
    [token],
  );

  useEffect(() => {
    const controller = new AbortController();
    refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  // Mutations target the reserved project (asset.projectId) and refetch locally — the
  // app-level asset mutators refresh the *active* project, not this one.
  const updateAssetStatus = useCallback(
    async (asset, changes) => {
      await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/status`, token, {
        method: "PATCH",
        body: JSON.stringify(changes),
      });
      await refresh();
    },
    [token, refresh],
  );
  const updateAssetTags = useCallback(
    async (asset, tags) => {
      await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/tags`, token, {
        method: "PATCH",
        body: JSON.stringify({ tags }),
      });
      await refresh();
    },
    [token, refresh],
  );
  const deleteAsset = useCallback(
    async (asset) => {
      await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}`, token, { method: "DELETE" });
      await refresh();
    },
    [token, refresh],
  );
  const purgeAsset = useCallback(
    async (asset) => {
      await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/purge`, token, { method: "DELETE" });
      await refresh();
    },
    [token, refresh],
  );

  const categoryOf = (asset) => asset.pose?.category || UNCATEGORIZED;
  const categories = useMemo(
    () => [...new Set(poses.map(categoryOf))].sort(),
    [poses],
  );
  const availableTags = useMemo(
    () => [...new Set(poses.flatMap((asset) => (Array.isArray(asset.tags) ? asset.tags : [])))].sort(),
    [poses],
  );

  const inFilter = (asset) => categoryFilter === "all" || categoryOf(asset) === categoryFilter;
  const inMode = (asset) => (assetMode === "trashcan" ? Boolean(asset.status?.trashed) : !asset.status?.trashed);
  const visible = poses.filter((asset) => inFilter(asset) && inMode(asset));
  const trashedInView = poses.filter((asset) => inFilter(asset) && asset.status?.trashed);
  const selected = poses.find((asset) => asset.id === selectedId) ?? null;
  const onPreview = (asset) => setSelectedId(asset.id);

  // Group the visible poses by category for the grid (category + tags shown per tile).
  const groups = useMemo(() => {
    const byCategory = new Map();
    for (const asset of visible) {
      const key = categoryOf(asset);
      if (!byCategory.has(key)) byCategory.set(key, []);
      byCategory.get(key).push(asset);
    }
    return [...byCategory.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [visible]);

  return (
    <section className="main-surface library-surface pose-library-surface">
      <div className="surface-header hero">
        <div className="section-heading">
          <p className="eyebrow">Pose Library</p>
          <h2>Poses</h2>
          <p className="hero-blurb">
            Manage your whole-body pose skeletons — discard, restore, tag, and categorize. Create new poses from photos in the Create tab.
          </p>
        </div>
        <div className="segmented-control" role="tablist" aria-label="Pose Library sections">
          {TABS.map(([id, label]) => (
            <button
              aria-controls={`pose-library-panel-${id}`}
              aria-selected={activeTab === id}
              className={activeTab === id ? "active" : ""}
              id={`pose-library-tab-${id}`}
              key={id}
              onClick={() => setActiveTab(id)}
              role="tab"
              type="button"
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div
        aria-labelledby="pose-library-tab-poses"
        hidden={activeTab !== "poses"}
        id="pose-library-panel-poses"
        role="tabpanel"
      >
        <div className="toolbar">
          <select
            aria-label="Pose category"
            onChange={(event) => setCategoryFilter(event.target.value)}
            value={categoryFilter}
          >
            <option value="all">All categories</option>
            {categories.map((category) => (
              <option key={category} value={category}>
                {category}
              </option>
            ))}
          </select>
          <div className="segmented-control" role="group" aria-label="Pose collection">
            <button className={assetMode === "assets" ? "active" : ""} onClick={() => setAssetMode("assets")} type="button">
              Poses
            </button>
            <button className={assetMode === "trashcan" ? "active" : ""} onClick={() => setAssetMode("trashcan")} type="button">
              Trashcan
            </button>
          </div>
          {assetMode === "trashcan" ? (
            <button
              className="danger-action empty-trash-button"
              disabled={!trashedInView.length}
              onClick={() => emptyTrash(trashedInView, purgeAsset)}
              type="button"
            >
              Empty Trash ({trashedInView.length})
            </button>
          ) : null}
        </div>

        {error ? <p className="inline-warning">Pose library unavailable: {error}</p> : null}

        <div className="library-layout">
          <div className="pose-library-grids">
            {loading && !poses.length ? (
              <div className="empty-panel">Loading poses…</div>
            ) : !visible.length ? (
              <div className="empty-panel">
                {assetMode === "trashcan"
                  ? "Trashcan is empty."
                  : "No saved poses yet — create some from photos in the Create tab."}
              </div>
            ) : (
              groups.map(([category, items]) => (
                <div className="pose-category" key={category}>
                  <p className="eyebrow">
                    {category} <span className="muted">({items.length})</span>
                  </p>
                  <AssetGrid
                    assets={items}
                    onPreview={onPreview}
                    selectedAsset={selected}
                    setSelectedAssetId={setSelectedId}
                  />
                </div>
              ))
            )}
          </div>
          <AssetDetail
            asset={selected}
            deleteAsset={deleteAsset}
            purgeAsset={purgeAsset}
            onPreview={onPreview}
            updateAssetStatus={updateAssetStatus}
            updateAssetTags={updateAssetTags}
            availableTags={availableTags}
          />
        </div>
      </div>

      <div
        aria-labelledby="pose-library-tab-create"
        hidden={activeTab !== "create"}
        id="pose-library-panel-create"
        role="tabpanel"
      >
        <div className="empty-panel">Pose creation from photos is coming soon.</div>
      </div>
    </section>
  );
}
