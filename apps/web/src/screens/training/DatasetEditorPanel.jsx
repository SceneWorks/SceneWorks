import React from "react";

import { AssetThumbnail } from "../../components/assetMedia.jsx";
import { CompactSelector } from "../../components/CompactSelector.jsx";
import { DatasetAddDialog } from "../../components/DatasetAddDialog.jsx";
import { DatasetCaptionDialog } from "../../components/DatasetCaptionDialog.jsx";
import { Icon } from "../../components/Icons.jsx";
import { WorkPanel } from "../../components/WorkPanel.jsx";
import { ValidationSummary } from "../../validation/Validation.jsx";
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

// Character kinds arrive lowercase ("person" / "style" / "object"); the chip shows them capitalized.
function characterKindLabel(type) {
  return type ? type.charAt(0).toUpperCase() + type.slice(1) : "";
}

// A member has an unresolved finding when the readiness report flags it at warn/fatal
// and the user hasn't dismissed that flag. Same predicate the per-thumbnail badge reads,
// so the "Flagged" filter and the badges never disagree.
function hasActiveFlags(entry) {
  return (entry?.flags ?? []).some(
    (flag) => !flag.acknowledged && (flag.severity === "warn" || flag.severity === "fatal"),
  );
}

// Health rail (sc-10481): the three counts the design calls out, plus a wide tile carrying
// the readiness dot + sentence that used to live in a separate `.training-validity` row.
function DatasetHealth({ health }) {
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
      <div className="training-health-status">
        <span className={health.valid ? "training-valid-dot valid" : "training-valid-dot"} />
        <span className="training-health-status-text">
          {health.valid ? "Dataset is ready for downstream steps" : "Add image assets to build this dataset"}
        </span>
      </div>
    </div>
  );
}

// Dataset-editor panel (sc-4199): extracted verbatim from TrainingStudio so the
// dataset name/prefix fields, health summary, caption grid, and the add/caption
// dialogs live in their own component. All state and handlers are owned by the
// TrainingStudio screen and passed in as props.
//
// sc-10481 re-cuts it onto the page-frame standard: one WorkPanel is the Purpose
// zone (identity + tools + health + Dataset Doctor, separated by hairlines), and the
// caption grid becomes a bare Results zone beneath it. The topbar names the page, so
// the panel carries no title of its own.
export function DatasetEditorPanel({
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
  saveValidity,
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
  associatedCharacterId,
  setActiveView,
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
  // The pencil button next to a saved dataset's name is a rename affordance: it focuses
  // the (otherwise chrome-less) name input rather than opening anything.
  const nameInputRef = React.useRef(null);
  const [captionFilter, setCaptionFilter] = React.useState("all");
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

  // sc-2022: the dataset's owning character, shown as a chip beside the name so the set's
  // subject (and the readiness kind it drives) is visible without opening Characters.
  const associatedCharacter = associatedCharacterId
    ? (characters ?? []).find((character) => character.id === associatedCharacterId)
    : null;

  const captionedCount = Math.max(0, health.itemCount - health.missingCaptions);
  const isMissingCaption = (asset) => !String(captionDraftById[asset.id]?.text ?? "").trim();
  const isFlagged = (asset) => hasActiveFlags(readinessByKey?.get(asset.id));
  const visibleAssets = memberAssets.filter((asset) => {
    if (captionFilter === "missing") {
      return isMissingCaption(asset);
    }
    if (captionFilter === "flagged") {
      return isFlagged(asset);
    }
    return true;
  });
  // Only derivable once the report exists — an item with no entry is "not assessed yet",
  // never silently ready (sc-6534).
  const readyCount = readiness ? memberAssets.filter((asset) => !isFlagged(asset)).length : 0;

  const statusPill = dirty ? (
    <span className="dataset-status-pill unsaved">
      <span className="dataset-status-dot" aria-hidden="true" />
      Unsaved changes
    </span>
  ) : activeDataset ? (
    <span className="dataset-status-pill">Version {activeDataset.version}</span>
  ) : (
    <span className="dataset-status-pill">Draft</span>
  );

  return (
    <>
      {datasetsError ? <p className="inline-warning">{datasetsError}</p> : null}
      {datasetError ? <p className="inline-warning">{datasetError}</p> : null}
      {datasetMessage ? <p className="inline-success">{datasetMessage}</p> : null}

      <WorkPanel className="dataset-work-panel">
        <div className="dataset-identity-row">
          <div className="dataset-identity">
            <p className="eyebrow work-panel-eyebrow">Build &amp; caption</p>
            <div className="dataset-name-row">
              {activeDataset ? (
                <span className="dataset-name-inline">
                  <input
                    aria-label="Dataset name"
                    className="dataset-name-input"
                    onChange={(event) => setDraftName(event.target.value)}
                    placeholder="Character portrait set"
                    ref={nameInputRef}
                    value={draftName}
                  />
                  <button
                    className="dataset-rename-button"
                    onClick={() => nameInputRef.current?.focus()}
                    title="Rename dataset"
                    type="button"
                  >
                    <Icon.Pencil size={14} />
                  </button>
                </span>
              ) : (
                <input
                  aria-label="Dataset name"
                  className="dataset-name-input boxed"
                  onChange={(event) => setDraftName(event.target.value)}
                  placeholder="Name this dataset…"
                  value={draftName}
                />
              )}
              {associatedCharacter ? (
                <span className="dataset-character-chip">
                  {associatedCharacter.type ? `${characterKindLabel(associatedCharacter.type)} · ` : ""}
                  {associatedCharacter.name}
                </span>
              ) : null}
            </div>
            <p className="dataset-identity-copy">
              Curate the images and captions that teach this LoRA. Fix flags below, then train from the{" "}
              <button className="link-button" onClick={() => setActiveView?.("Train")} type="button">
                Training Studio
              </button>
              .
            </p>
          </div>
          <div className="dataset-identity-actions">
            <div className="dataset-identity-controls">
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
              <button className="secondary-action" disabled={loadingDatasets} onClick={onRefreshDatasets} type="button">
                <Icon.Refresh size={14} />
                {loadingDatasets ? "Refreshing" : "Refresh"}
              </button>
            </div>
            <button className="primary-action dataset-save" disabled={!canSave} onClick={saveDataset} type="button">
              {savingDataset ? "Saving" : activeDataset ? "Save dataset" : "Create dataset"}
            </button>
            {statusPill}
            {/* The one broken-value case: assets that went rejected/trashed/deleted after
                selection. Missing name / empty selection are requirements — silent, the
                empty field speaks for itself — and the caption/duplicate counts have their
                own home in the health grid, so this row is usually absent (sc-10648). */}
            <ValidationSummary issues={saveValidity?.surfaced} label="Dataset errors" />
          </div>
        </div>

        <div className="work-panel-divider" />

        <div className="dataset-tools-row">
          <label className="dataset-field dataset-prefix-field">
            <span>Name prefix</span>
            <input onChange={(event) => setRenamePrefix(event.target.value)} placeholder="item" value={renamePrefix} />
          </label>
          <span className="dataset-tools-actions">
            <button className="secondary-action strong" onClick={() => setAddDialogOpen(true)} type="button">
              <Icon.Plus size={14} />
              Add images
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
            <button
              className="secondary-action"
              disabled={!activeDataset?.id || renaming || !memberAssets.length}
              onClick={applyOrderedNames}
              type="button"
            >
              <Icon.Sliders size={14} />
              {renaming ? "Renaming…" : "Apply ordered names"}
            </button>
          </span>
        </div>

        <div className="work-panel-divider" />

        <DatasetHealth health={health} />

        {readiness || readinessLoading ? (
          <>
            <div className="work-panel-divider" />
            <div className="dataset-doctor-band">
              <div className="dataset-doctor-band-head">
                <span className="dataset-doctor-band-icon" aria-hidden="true">
                  <Icon.Stethoscope size={18} />
                </span>
                <strong>Dataset Doctor</strong>
                {readiness ? (
                  <span className="dataset-doctor-band-count">
                    {readyCount} of {health.itemCount} items ready
                  </span>
                ) : null}
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
            </div>
          </>
        ) : null}
      </WorkPanel>

      {loadingDatasets ? <div className="empty-panel compact-panel">Loading training datasets</div> : null}
      {!loadingDatasets && datasets.length === 0 ? (
        <div className="empty-panel compact-panel">
          No training datasets yet — open the dataset selector and pick “New dataset” to start one.
        </div>
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

      <section className="dataset-results">
        <div className="dataset-results-head">
          <div className="section-heading">
            <p className="eyebrow">Results</p>
            <h2>Images &amp; captions</h2>
          </div>
          <span className="dataset-count">
            {health.itemCount} image{health.itemCount === 1 ? "" : "s"} · {captionedCount} captioned
          </span>
          <div className="segmented-control" role="group" aria-label="Filter images">
            <button
              className={captionFilter === "all" ? "active" : ""}
              onClick={() => setCaptionFilter("all")}
              type="button"
            >
              All
            </button>
            <button
              className={captionFilter === "missing" ? "active" : ""}
              onClick={() => setCaptionFilter("missing")}
              type="button"
            >
              Missing
            </button>
            <button
              className={captionFilter === "flagged" ? "active" : ""}
              onClick={() => setCaptionFilter("flagged")}
              type="button"
            >
              Flagged
            </button>
          </div>
        </div>

        <div className="training-caption-grid" aria-label="Dataset images and captions">
          {!memberAssets.length ? (
            <div className="empty-panel compact-panel">No images yet — use “Add images” to build this dataset.</div>
          ) : !visibleAssets.length ? (
            <div className="empty-panel compact-panel">
              {captionFilter === "missing"
                ? "Every image has a caption."
                : "No images carry an unresolved quality finding."}
            </div>
          ) : (
            visibleAssets.map((asset) => {
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
                  <button className="training-caption-card-thumb" onClick={() => onPreview(asset, memberAssets)} type="button">
                    <AssetThumbnail asset={asset} />
                    {/* Only flash pending on the first load — during an ack-triggered refetch the
                        prior report's badges hold steady rather than blinking to "·". */}
                    <ReadinessBadge entry={readinessEntry} loading={readinessLoading && !readiness} />
                  </button>
                  <div className="training-caption-card-body">
                    <div className="training-caption-card-meta">
                      <strong title={name}>{name}</strong>
                      <span className={`training-caption-source source-${source}`}>{captionSourceLabel(source)}</span>
                      {disabled ? (
                        <span className="training-asset-badge">{asset.status?.trashed ? "Trashed" : "Rejected"}</span>
                      ) : null}
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
                      className={[
                        "training-caption-card-text",
                        isMissingCaption(asset) ? "is-empty" : "",
                      ]
                        .filter(Boolean)
                        .join(" ")}
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
                        className="secondary-action strong"
                        disabled={captioning}
                        onClick={() => setCaptionDialog({ type: "item", member: asset })}
                        type="button"
                      >
                        Re-Caption
                      </button>
                    </div>
                  </div>
                </article>
              );
            })
          )}
        </div>
      </section>

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
    </>
  );
}
