import React from "react";

import { AssetThumbnail } from "../../components/assetMedia.jsx";
import { CompactSelector } from "../../components/CompactSelector.jsx";
import { DatasetAddDialog } from "../../components/DatasetAddDialog.jsx";
import { DatasetCaptionDialog } from "../../components/DatasetCaptionDialog.jsx";
import { Icon } from "../../components/Icons.jsx";
import { WorkPanel } from "../../components/WorkPanel.jsx";
import {
  DatasetDoctorDistributions,
  DatasetDoctorReadout,
  NearDuplicateClusters,
  ReadinessBadge,
  ReadinessFlagDetails,
} from "./DatasetDoctor.jsx";
import { datasetItemCount, imageAssetName } from "../../training/datasetHelpers.js";
import { joyCaptionExtraOptions, joyCaptionLengths, joyCaptionTypes } from "../../training/joyCaptionPrompts.js";

// Human label for the detected caption source (sc-2025) — read-only on the card.
function captionSourceLabel(source) {
  if (source === "imported") {
    return "Imported";
  }
  if (source === "auto") {
    return "Auto";
  }
  return "Manual";
}

function DatasetHealth({ health, action }) {
  return (
    <div className="training-health-grid" aria-label="Dataset health">
      <div>
        <strong>{health.itemCount}</strong>
        <span>Items</span>
      </div>
      <div className={health.missingCaptions ? "needs-attention" : ""}>
        <strong>{health.missingCaptions}</strong>
        <span>Missing captions</span>
      </div>
      <div className={health.duplicateFilenames ? "needs-attention" : ""}>
        <strong>{health.duplicateFilenames}</strong>
        <span>Duplicate filenames</span>
      </div>
      {action ? <div className="training-health-action">{action}</div> : null}
    </div>
  );
}

// Dataset-editor panel (sc-4199): extracted verbatim from TrainingStudio so the
// dataset name/prefix fields, health summary, caption grid, and the add/caption
// dialogs live in their own component. All state and handlers are owned by the
// TrainingStudio screen and passed in as props — behavior is unchanged.
export function DatasetEditorPanel({
  active,
  loadingDatasets,
  onRefreshDatasets,
  busyDatasetId,
  datasetThumbAsset,
  datasets,
  startNewDataset,
  openDataset,
  activeDataset,
  selectedDatasetId,
  datasetsError,
  datasetError,
  datasetMessage,
  draftName,
  setDraftName,
  dirty,
  setAddDialogOpen,
  renamePrefix,
  setRenamePrefix,
  renaming,
  memberAssets,
  applyOrderedNames,
  setCaptionDialog,
  health,
  // sc-8942 (F-140): the Dataset Doctor's readout props (report/loading + the six
  // fix-action callbacks) are grouped into one `datasetDoctor` bundle instead of being
  // threaded individually. ConfigureJobPanel takes the same bundle, so the identical
  // set of props is no longer hand-mirrored across two panels. Shaped exactly like the
  // DatasetDoctorReadout signature so it can be spread straight onto it.
  datasetDoctor,
  readinessByKey,
  onToggleItemAck,
  canSave,
  saveDataset,
  savingDataset,
  unavailableAssetIds,
  removeUnavailableAsset,
  captionDraftById,
  onPreview,
  updateCaption,
  captioning,
  addDialogOpen,
  imageAssets,
  characters,
  importingAssets,
  selectedAssetIds,
  addAssets,
  handleImport,
  captionDialog,
  gpuOptions,
  updateCaptionSetting,
  runCaptionJob,
  toggleCaptionExtraOption,
  displayedCaptionPrompt,
  captionSettings,
  captionModelMissing = false,
  onDownloadCaptionModel,
  captionModelSizeLabel = "",
  captionModelName = "JoyCaption",
}) {
  // Local aliases for the doctor readout's report/loading, still referenced directly by
  // the distributions block and the per-card readiness badges below.
  const { report: readiness = null, loading: readinessLoading = false } = datasetDoctor ?? {};
  // sc-8564: resolve a near-dup cluster's item ids (server dataset-item ids — the readiness report's
  // `itemId` / flag `peers`) back to member assets, so the cluster view can render each sibling's
  // thumbnail. `activeDataset.items` carry both the server `id` and the `assetId` (the member-asset key).
  const assetByItemId = React.useMemo(() => {
    const byAssetId = new Map((memberAssets ?? []).map((asset) => [asset.id, asset]));
    const map = new Map();
    for (const item of activeDataset?.items ?? []) {
      const asset = item.assetId ? byAssetId.get(item.assetId) : null;
      if (asset) {
        map.set(item.id, asset);
      }
    }
    return map;
  }, [activeDataset, memberAssets]);
  const renderClusterThumbnail = (itemId) => {
    const asset = assetByItemId.get(itemId);
    return asset ? (
      <AssetThumbnail asset={asset} />
    ) : (
      <span className="dataset-doctor-nearup-thumb-missing" title={itemId} aria-hidden />
    );
  };
  return (
    <>
      <div className="training-panel-head">
        <div>
          <p className="eyebrow">Dataset</p>
          <h3>{active.title}</h3>
        </div>
        <div className="training-head-actions">
          <button className="secondary-action" disabled={loadingDatasets} onClick={onRefreshDatasets} type="button">
            <Icon.Search size={14} />
            {loadingDatasets ? "Refreshing" : "Refresh"}
          </button>
          <CompactSelector
            busyId={busyDatasetId}
            createLabel="New dataset"
            getSubtitle={(dataset) => {
              const count = datasetItemCount(dataset);
              return `${count} item${count === 1 ? "" : "s"}`;
            }}
            getThumbAsset={datasetThumbAsset}
            items={datasets}
            label="Select dataset"
            onCreate={startNewDataset}
            onSelect={(dataset) => openDataset(dataset.id)}
            placeholder={activeDataset ? activeDataset.name : "New dataset"}
            selectedId={selectedDatasetId}
          />
        </div>
      </div>
      {datasetsError ? <p className="inline-warning">{datasetsError}</p> : null}
      {datasetError ? <p className="inline-warning">{datasetError}</p> : null}
      {datasetMessage ? <p className="inline-success">{datasetMessage}</p> : null}
      <div className="training-dataset-workspace">
        {loadingDatasets ? <div className="empty-panel compact-panel">Loading training datasets</div> : null}
        {!loadingDatasets && datasets.length === 0 ? (
          <div className="empty-panel compact-panel">No training datasets yet — use “New” to start one.</div>
        ) : null}
        <div className="training-dataset-editor">
          <WorkPanel className="dataset-work-panel">
            <div className="dataset-primary-row">
              <label className="dataset-field">
                <span>Dataset name</span>
                <input
                  onChange={(event) => setDraftName(event.target.value)}
                  placeholder="Character portrait set"
                  value={draftName}
                />
              </label>
              <label className="dataset-field">
                <span>Name prefix</span>
                <input
                  onChange={(event) => setRenamePrefix(event.target.value)}
                  placeholder="item"
                  value={renamePrefix}
                />
              </label>
              <span className="dataset-version">
                {dirty ? "Unsaved changes" : activeDataset ? `Version ${activeDataset.version}` : "Draft"}
              </span>
              <button className="primary-action dataset-save" disabled={!canSave} onClick={saveDataset} type="button">
                {savingDataset ? "Saving" : activeDataset ? "Save dataset" : "Create dataset"}
              </button>
            </div>
            <div className="dataset-utility-row">
              <span className="dataset-count">
                {health.itemCount} image{health.itemCount === 1 ? "" : "s"} ·{" "}
                {Math.max(0, health.itemCount - health.missingCaptions)} captioned
              </span>
              <span className="dataset-utility-actions">
                <button className="secondary-action" onClick={() => setAddDialogOpen(true)} type="button">
                  <Icon.Plus size={14} />
                  Add images
                </button>
                <button
                  className="secondary-action"
                  disabled={!activeDataset?.id || renaming || !memberAssets.length}
                  onClick={applyOrderedNames}
                  type="button"
                >
                  <Icon.Sliders size={14} />
                  {renaming ? "Renaming…" : "Apply ordered names"}
                </button>
                <button
                  className="secondary-action"
                  disabled={!memberAssets.length}
                  onClick={() => setCaptionDialog({ type: "all" })}
                  type="button"
                >
                  <Icon.Sliders size={14} />
                  Caption all
                </button>
              </span>
            </div>
          </WorkPanel>
          <DatasetHealth health={health} />
          <div className="training-validity">
            <span className={health.valid ? "training-valid-dot valid" : "training-valid-dot"} />
            <span>{health.valid ? "Dataset is ready for downstream steps" : "Add image assets to build this dataset"}</span>
          </div>
          <DatasetDoctorReadout
            {...datasetDoctor}
            onRecaptionFlagged={(itemIds) => setCaptionDialog({ type: "flagged", itemIds })}
          />
          <NearDuplicateClusters
            report={readiness}
            renderThumbnail={renderClusterThumbnail}
            onRemoveDuplicates={datasetDoctor?.onRemoveDuplicates}
          />
          {readiness?.distributions ? (
            <details className="dataset-doctor-advanced">
              <summary>Metric distributions (advanced)</summary>
              <DatasetDoctorDistributions report={readiness} />
            </details>
          ) : null}
          {unavailableAssetIds.length ? (
            <div className="training-unavailable-list" aria-label="Unavailable dataset items">
              {unavailableAssetIds.map((assetId) => (
                <div className="training-unavailable-item" key={assetId}>
                  <div>
                    <strong>{assetId}</strong>
                    <span>Asset is no longer available</span>
                  </div>
                  <button className="secondary-action" onClick={() => removeUnavailableAsset(assetId)} type="button">
                    Remove
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          <div className="training-caption-grid" aria-label="Dataset images and captions">
            {memberAssets.length ? (
              memberAssets.map((asset) => {
                const disabled = asset.status?.rejected || asset.status?.trashed;
                const draft = captionDraftById[asset.id] ?? {};
                const source = draft.source ?? "manual";
                const name = asset.displayName ?? imageAssetName(asset);
                const readinessEntry = readinessByKey?.get(asset.id);
                return (
                  <article
                    className={["training-caption-card", disabled ? "disabled" : ""].filter(Boolean).join(" ")}
                    key={asset.id}
                  >
                    <div className="training-caption-card-head">
                      <button className="training-caption-card-thumb" onClick={() => onPreview(asset, memberAssets)} type="button">
                        <AssetThumbnail asset={asset} />
                        {/* Only flash pending on the first load — during an ack-triggered refetch the
                            prior report's badges hold steady rather than blinking to "·". */}
                        <ReadinessBadge entry={readinessEntry} loading={readinessLoading && !readiness} />
                      </button>
                      <div className="training-caption-card-meta">
                        <strong title={name}>{name}</strong>
                        <span className={`training-caption-source source-${source}`}>{captionSourceLabel(source)}</span>
                        {disabled ? (
                          <span className="training-asset-badge">{asset.status?.trashed ? "Trashed" : "Rejected"}</span>
                        ) : null}
                      </div>
                    </div>
                    <ReadinessFlagDetails
                      entry={readinessEntry}
                      onToggle={
                        readinessEntry && typeof onToggleItemAck === "function"
                          ? (check, dismissed) => onToggleItemAck(readinessEntry, check, dismissed)
                          : undefined
                      }
                    />
                    <textarea
                      aria-label={`Caption for ${name}`}
                      className="training-caption-card-text"
                      onChange={(event) => updateCaption(asset.id, event.target.value)}
                      placeholder="Describe this image…"
                      rows={3}
                      value={draft.text ?? ""}
                    />
                    <div className="training-caption-card-actions">
                      <button
                        aria-label={`Remove ${name}`}
                        className="secondary-action"
                        onClick={() => removeUnavailableAsset(asset.id)}
                        type="button"
                      >
                        Remove
                      </button>
                      <button
                        aria-label={`Re-caption ${name}`}
                        className="secondary-action"
                        disabled={captioning}
                        onClick={() => setCaptionDialog({ type: "item", member: asset })}
                        type="button"
                      >
                        Re-Caption
                      </button>
                    </div>
                  </article>
                );
              })
            ) : (
              <div className="empty-panel compact-panel">No images yet — use “Add images” to build this dataset.</div>
            )}
          </div>
          {addDialogOpen ? (
            <DatasetAddDialog
              assets={imageAssets}
              characters={characters}
              importing={importingAssets}
              memberIds={selectedAssetIds}
              onAdd={addAssets}
              onClose={() => setAddDialogOpen(false)}
              onImport={handleImport}
            />
          ) : null}
          {captionDialog ? (
            <DatasetCaptionDialog
              captionLengths={joyCaptionLengths}
              captionTypes={joyCaptionTypes}
              extraOptions={joyCaptionExtraOptions}
              gpuOptions={gpuOptions}
              onChange={updateCaptionSetting}
              onClose={() => setCaptionDialog(null)}
              onRun={runCaptionJob}
              onToggleExtra={toggleCaptionExtraOption}
              promptValue={displayedCaptionPrompt}
              running={captioning}
              modelMissing={captionModelMissing}
              onDownloadModel={onDownloadCaptionModel}
              modelSizeLabel={captionModelSizeLabel}
              modelName={captionModelName}
              scope={
                captionDialog.type === "item"
                  ? {
                      type: "item",
                      name: captionDialog.member.displayName ?? imageAssetName(captionDialog.member),
                    }
                  : captionDialog.type === "flagged"
                    ? { type: "flagged", count: captionDialog.itemIds?.length ?? 0 }
                    : { type: "all" }
              }
              settings={captionSettings}
            />
          ) : null}
        </div>
      </div>
    </>
  );
}
