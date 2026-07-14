import React, { useState } from "react";
import { AssetBatchModal, AssetSelectionBar, useAssetBatch } from "../assetBatch.jsx";
import { foldUpscaledAssetVariants } from "../assetVariants.js";
import { AssetDetail, AssetGrid, emptyTrash } from "../components/assetPanels.jsx";
import { isLibraryAsset, terminalStatuses } from "../constants.js";
import { useAppContext } from "../context/AppContext.js";
import { WorkPanel } from "../components/WorkPanel.jsx";

// Free-text Library search across an asset's display name, prompt, and tags.
// Case-insensitive substring match; an empty needle matches everything so callers
// can pass the query unconditionally. `needle` is expected pre-lowercased/trimmed.
export function assetMatchesSearch(asset, needle) {
  if (!needle) {
    return true;
  }
  const displayName = asset.displayName ?? "";
  const prompt = asset.recipe?.prompt ?? "";
  const tags = Array.isArray(asset.tags) ? asset.tags.join(" ") : "";
  return `${displayName} ${prompt} ${tags}`.toLowerCase().includes(needle);
}

export function LibraryScreen() {
  const {
    activeProject,
    assets,
    jobs = [],
    imageModels = [],
    createVqaJob,
    deleteAsset,
    purgeAsset,
    characters = [],
    moveAssetToCharacter,
    importAsset,
    setPreviewAsset,
    sendAssetToImage,
    sendAssetToVideo,
    selectedAsset,
    setSelectedAssetId,
    setActiveView,
    updateAssetStatus,
    updateAssetTags,
  } = useAppContext();
  // Shared multi-select + batch toolbar (selection state, fan-out, Discard/Move).
  const batch = useAssetBatch();
  // Bind the fullscreen preview to the currently filtered library view so
  // navigation stays inside the same type/tag/trash scope the user is browsing.
  const onPreview = (asset) => setPreviewAsset(asset, visibleAssets);
  const onSendImage = (asset) => sendAssetToImage(asset);
  const onSendVideo = (asset) => sendAssetToVideo(asset);
  const onSendEditor = (asset) => {
    setSelectedAssetId(asset.id);
    setActiveView("Editor");
  };
  const vqaEnabled = Boolean(createVqaJob) && imageModels.some((model) => (model.capabilities ?? []).includes("vqa"));
  const [typeFilter, setTypeFilter] = useState("all");
  const [tagFilter, setTagFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [showRejected, setShowRejected] = useState(false);
  const [assetMode, setAssetMode] = useState("assets");
  const [isImporting, setIsImporting] = useState(false);
  // Free-text search over prompt + tags. Normalized once here; the matcher expects it
  // pre-lowercased/trimmed and treats an empty needle as "match everything".
  const searchNeedle = searchQuery.trim().toLowerCase();
  // Asset Library hygiene (sc-2024 / sc-8339): show only studio-generated and uploaded
  // media via a positive origin allow-list (isLibraryAsset). Character Studio outputs,
  // Pose/Key Point library assets, and any future/foreign origin stay out by default —
  // a single `!== "character_studio"` exclusion let everything else leak in. The backend
  // also exposes `?scope=library`; we filter on the authoritative `origin` field
  // client-side to reuse the shared asset feed instead of loading a second scoped copy.
  // Dataset images never reach the Library — they are stored outside the indexed folders.
  const libraryAssets = assets.filter(isLibraryAsset);
  const visibleAssets = foldUpscaledAssetVariants(libraryAssets.filter((asset) => {
    if (typeFilter !== "all" && asset.type !== typeFilter) {
      return false;
    }
    if (!showRejected && asset.status?.rejected) {
      return false;
    }
    if (assetMode === "trashcan" && !asset.status?.trashed) {
      return false;
    }
    if (assetMode === "assets" && asset.status?.trashed) {
      return false;
    }
    if (tagFilter !== "all" && !(asset.tags ?? []).includes(tagFilter)) {
      return false;
    }
    if (!assetMatchesSearch(asset, searchNeedle)) {
      return false;
    }
    return true;
  }));
  // Detail-sidebar selection, scoped to the Library (sc-8339). The global `selectedAsset`
  // falls back to the most-recent asset across ALL studios, so without scoping the panel
  // can show a freshly generated Character Studio image instead of a library asset. Keep
  // the global selection only when it is itself a library asset (even if the current
  // tag/type filter hides it); otherwise fall back to the most-recent visible asset —
  // never a non-library (e.g. character) asset.
  const librarySelectedAsset =
    libraryAssets.find((asset) => asset.id === selectedAsset?.id) ?? visibleAssets[0] ?? null;
  const assetVqaJobs = librarySelectedAsset
    ? jobs.filter((job) => job.type === "image_vqa" && job.payload?.sourceAssetId === librarySelectedAsset.id)
    : [];
  const vqaEntries = assetVqaJobs
    .filter((job) => job.status === "completed" && job.result?.answer)
    .map((job) => ({
      jobId: job.id,
      question: job.result?.question ?? job.payload?.question ?? "",
      answer: job.result.answer,
    }));
  const vqaPending = assetVqaJobs.some((job) => !terminalStatuses.has(job.status));
  // All discarded assets within the current type/tag filters — the exact scope
  // the "Empty Trash" button purges (unfolded, so folded variants go too).
  const trashedInView = libraryAssets.filter((asset) => {
    if (typeFilter !== "all" && asset.type !== typeFilter) {
      return false;
    }
    if (tagFilter !== "all" && !(asset.tags ?? []).includes(tagFilter)) {
      return false;
    }
    if (!assetMatchesSearch(asset, searchNeedle)) {
      return false;
    }
    return Boolean(asset.status?.trashed);
  });

  async function handleImport(event) {
    const [file] = event.target.files;
    if (!file) {
      return;
    }
    setIsImporting(true);
    await importAsset(file);
    setIsImporting(false);
    event.target.value = "";
  }

  const imageCount = libraryAssets.filter((asset) => asset.type === "image").length;
  const videoCount = libraryAssets.filter((asset) => asset.type === "video").length;
  const uploadCount = libraryAssets.filter((asset) => asset.type === "upload").length;
  const availableTags = [...new Set(libraryAssets.flatMap((asset) => (Array.isArray(asset.tags) ? asset.tags : [])))].sort();

  return (
    <section className="page-frame library-surface">
      <WorkPanel>
        <div className="toolbar">
          <label className="file-upload-button">
            <input accept="image/*,video/*" disabled={isImporting} onChange={handleImport} type="file" />
            {isImporting ? "Importing..." : "Import"}
          </label>
          <input
            type="search"
            aria-label="Search assets"
            placeholder="Search name, prompt, or tags…"
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
          />
          {searchNeedle ? (
            <button
              type="button"
              disabled={!visibleAssets.length}
              onClick={() => batch.selectAll(visibleAssets.map((asset) => asset.id))}
            >
              Select all ({visibleAssets.length})
            </button>
          ) : null}
          <select aria-label="Asset type" onChange={(event) => setTypeFilter(event.target.value)} value={typeFilter}>
            <option value="all">All media</option>
            <option value="image">Images</option>
            <option value="video">Videos</option>
            <option value="upload">Uploads</option>
            <option value="render">Renders</option>
          </select>
          <select aria-label="Asset tag" onChange={(event) => setTagFilter(event.target.value)} value={tagFilter}>
            <option value="all">All tags</option>
            {availableTags.map((tag) => (
              <option key={tag} value={tag}>
                {tag}
              </option>
            ))}
          </select>
          <label className="checkline">
            <input checked={showRejected} onChange={(event) => setShowRejected(event.target.checked)} type="checkbox" />
            Rejected
          </label>
          <div className="segmented-control" role="group" aria-label="Asset collection">
            <button className={assetMode === "assets" ? "active" : ""} onClick={() => setAssetMode("assets")} type="button">
              Assets
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
        <div className="stat-strip">
          <div className="stat-chip">
            <span className="stat-chip-label">Project</span>
            <span className="stat-chip-value">{activeProject?.name ?? "—"}</span>
          </div>
          <div className="stat-chip">
            <span className="stat-chip-label">Assets</span>
            <span className="stat-chip-value">{libraryAssets.length} total</span>
          </div>
          <div className="stat-chip">
            <span className="stat-chip-label">Images</span>
            <span className="stat-chip-value">{imageCount}</span>
          </div>
          <div className="stat-chip">
            <span className="stat-chip-label">Clips</span>
            <span className="stat-chip-value">{videoCount + uploadCount}</span>
          </div>
        </div>
      </WorkPanel>

      <AssetSelectionBar batch={batch} showDiscard={assetMode === "assets"} />

      <div className="library-layout">
        <AssetGrid
          assets={visibleAssets}
          onPreview={onPreview}
          selectedAsset={librarySelectedAsset}
          setSelectedAssetId={setSelectedAssetId}
          selectedIds={batch.selectedAssetIds}
          onToggleSelect={batch.toggleSelect}
        />
        <AssetDetail
          asset={librarySelectedAsset}
          deleteAsset={deleteAsset}
          purgeAsset={purgeAsset}
          onPreview={onPreview}
          onSendImage={onSendImage}
          onSendVideo={onSendVideo}
          onSendEditor={onSendEditor}
          characters={characters}
          onMoveToCharacter={moveAssetToCharacter ?? null}
          updateAssetStatus={updateAssetStatus}
          updateAssetTags={updateAssetTags}
          availableTags={availableTags}
          vqaEnabled={vqaEnabled}
          vqaEntries={vqaEntries}
          vqaPending={vqaPending}
          createVqaJob={createVqaJob}
        />
      </div>

      <AssetBatchModal batch={batch} />
    </section>
  );
}
