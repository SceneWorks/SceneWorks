import React from "react";
import { itemSplit } from "../../training/irisCurricula.js";

import { AssetThumbnail } from "../../components/assetMedia.jsx";
import { CompactSelector } from "../../components/CompactSelector.jsx";
import { DatasetAddDialog } from "../../components/DatasetAddDialog.jsx";
import { DatasetCaptionDialog } from "../../components/DatasetCaptionDialog.jsx";
import { DatasetParquetImportDialog } from "../../components/DatasetParquetImportDialog.jsx";
import { errorMessage } from "../../errorMessage.js";
import { Icon } from "../../components/Icons.jsx";
import { WorkPanel } from "../../components/WorkPanel.jsx";
import { ValidationSummary } from "../../validation/Validation.jsx";
import { IrisSidecarPanel } from "./IrisSidecarPanel.jsx";
import {
  DatasetDoctorDistributions,
  DatasetDoctorReadout,
  NearDuplicateClusters,
} from "./DatasetDoctor.jsx";
import { DatasetCaptionCard } from "./DatasetCaptionCard.jsx";
import { DatasetCaptionWindow } from "./DatasetCaptionWindow.jsx";
import { useLatestHandlers } from "../../training/useLatestHandlers.js";
import { datasetItemCount, datasetItemSelectionKey, imageAssetName } from "../../training/datasetHelpers.js";
import { joyCaptionExtraOptions, joyCaptionLengths, joyCaptionTypes } from "../../training/joyCaptionPrompts.js";
import {
  subjectMaskStatusByItemId,
  subjectMaskUrl,
} from "../../training/subjectMasks.js";

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
  datasetSession,
  captionSession,
  doctorSession,
  config,
  maskSession = {},
  // Instruction-edit pairs (sc-24161): per-item ordered reference rails. Absent (or a 0 cap — no
  // trainable edit target) leaves the editor exactly as it was.
  editPairSession = {},
}) {
  const {
    loadingDatasets, onRefreshDatasets, busyDatasetId, datasetThumbAsset, datasets,
    startNewDataset, openDataset, activeDataset, selectedDatasetId, datasetsError,
    datasetError, datasetMessage, draftName, setDraftName, dirty, discardDraft,
    setAddDialogOpen, renamePrefix, setRenamePrefix, renaming, memberAssets,
    applyOrderedNames, setCaptionDialog, health, canSave, saveValidity, saveDataset,
    savingDataset, unavailableAssetIds, removeUnavailableAsset,
    onImportParquet, onDeleteDataset, deletingDataset,
  } = datasetSession;
  const {
    captionDraftById, namedCaptionsById = {}, holdoutById = {}, onToggleHoldout, onPreview, updateCaption, captioning, addDialogOpen,
    selectedAssetIds, addAssets, handleImport, captionDialog, updateCaptionSetting,
    runCaptionJob, toggleCaptionExtraOption, displayedCaptionPrompt, captionSettings,
    captionModelMissing = false, onDownloadCaptionModel, captionModelSizeLabel = "",
    captionModelName = "JoyCaption",
  } = captionSession;
  const { datasetDoctor, readinessByKey, onToggleItemAck } = doctorSession;
  const {
    imageAssets, characters, associatedCharacterId, setActiveView, importingAssets, importProgress, onCancelImport,
    gpuOptions, onUploadPreparedBundle, onIrisSidecar,
  } = config;
  const {
    referenceCap = 0, referenceDraftById = {}, setItemReferences, addItemReferences,
    importItemReferences, editPairsMode = false, setEditPairsMode,
  } = editPairSession;
  // Only the CURRENT members count: drafts for removed items linger in the map (the save payload
  // ignores them) and must not lock the toggle on.
  const hasEditPairs = (memberAssets ?? []).some((asset) => referenceDraftById[asset.id]?.length);
  // The rails show once the user turns edit pairs on, or whenever the dataset already has some.
  const showEditPairs = referenceCap > 0 && (editPairsMode || hasEditPairs);
  const [referencePickerFor, setReferencePickerFor] = React.useState(null);
  const assetsById = React.useMemo(
    () => new Map((imageAssets ?? []).map((asset) => [asset.id, asset])),
    [imageAssets],
  );
  // Local aliases for the doctor readout's report/loading, still referenced directly by
  // the distributions block and the per-card readiness badges below.
  const { report: readiness = null, loading: readinessLoading = false } = datasetDoctor ?? {};
  // The pencil button next to a saved dataset's name is a rename affordance: it focuses
  // the (otherwise chrome-less) name input rather than opening anything.
  const nameInputRef = React.useRef(null);
  const [captionFocus, setCaptionFocus] = React.useState(null);
  const focusedCaptionId = captionFocus?.datasetId === activeDataset?.id ? captionFocus?.assetId : null;
  const [captionFilter, setCaptionFilter] = React.useState("all");
  const [parquetDialogOpen, setParquetDialogOpen] = React.useState(false);
  const [parquetImporting, setParquetImporting] = React.useState(false);
  const [parquetError, setParquetError] = React.useState("");
  const [preparedUploadingId, setPreparedUploadingId] = React.useState("");
  // Subject masks (sc-2126): overlay toggle, the card whose replacement mask is uploading, and the
  // per-card upload error (field-level: shown on the card that owns the upload).
  const { projectId: maskProjectId, report: maskReport, onGenerateMasks, onUploadMask } = maskSession;
  const [showMaskOverlay, setShowMaskOverlay] = React.useState(false);
  const [maskUploadingId, setMaskUploadingId] = React.useState("");
  const [maskUploadErrors, setMaskUploadErrors] = React.useState({});
  // Only trust a report for the dataset on screen — item ids repeat across datasets.
  const activeMaskReport = maskReport && maskReport.datasetId === activeDataset?.id ? maskReport : null;
  const maskStatusByItemId = React.useMemo(() => subjectMaskStatusByItemId(activeMaskReport), [activeMaskReport]);
  const parquetImportDisabled = !draftName.trim() || memberAssets.length > 0 || parquetImporting;

  async function runParquetImport(settings) {
    if (typeof onImportParquet !== "function" || parquetImporting) return;
    setParquetImporting(true);
    setParquetError("");
    try {
      await onImportParquet(settings);
      setParquetDialogOpen(false);
    } catch (error) {
      setParquetError(errorMessage(error, "Could not start the Parquet import."));
    } finally {
      setParquetImporting(false);
    }
  }
  // sc-8564: resolve a near-dup cluster's item ids (server dataset-item ids — the readiness report's
  // `itemId` / flag `peers`) back to member assets, so the cluster view can render each sibling's
  // thumbnail. Use the editor's selection key for both library and dataset-owned images.
  const assetByItemId = React.useMemo(() => {
    const byAssetId = new Map((memberAssets ?? []).map((asset) => [asset.id, asset]));
    const map = new Map();
    for (const [index, item] of (activeDataset?.items ?? []).entries()) {
      const asset = byAssetId.get(datasetItemSelectionKey(activeDataset, item, index));
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
      <span className="dataset-doctor-nearup-thumb-missing" role="img" aria-label="Image unavailable">Unavailable</span>
    );
  };

  // sc-2022: the dataset's owning character, shown as a chip beside the name so the set's
  // subject (and the readiness kind it drives) is visible without opening Characters.
  const associatedCharacter = associatedCharacterId
    ? (characters ?? []).find((character) => character.id === associatedCharacterId)
    : null;

  const captionedCount = Math.max(0, health.itemCount - health.missingCaptions);
  const visibleAssets = React.useMemo(() => captionFilter === "all" ? memberAssets : memberAssets.filter((asset) => {
    // Keep the active editor mounted until focus leaves, even when its first
    // keystroke resolves the Missing filter. Draft state remains owner-scoped.
    if (asset.id === focusedCaptionId) return true;
    if (captionFilter === "missing") {
      return !String(captionDraftById[asset.id]?.text ?? "").trim();
    }
    if (captionFilter === "flagged") {
      return hasActiveFlags(readinessByKey?.get(asset.id));
    }
    return true;
  }), [memberAssets, focusedCaptionId, captionFilter, captionDraftById, readinessByKey]);
  // Only derivable once the report exists — an item with no entry is "not assessed yet",
  // never silently ready (sc-6534).
  const readyCount = React.useMemo(
    () => readiness ? memberAssets.filter((asset) => !hasActiveFlags(readinessByKey?.get(asset.id))).length : 0,
    [readiness, memberAssets, readinessByKey],
  );
  const savedItemBySelection = React.useMemo(
    () => new Map((activeDataset?.items ?? []).map((item, index) => [datasetItemSelectionKey(activeDataset, item, index), item])),
    [activeDataset],
  );

  async function uploadMask(asset, file) {
    if (!file || typeof onUploadMask !== "function") return;
    setMaskUploadingId(asset.id);
    setMaskUploadErrors((errors) => ({ ...errors, [asset.id]: "" }));
    try {
      await onUploadMask(asset, file);
      setShowMaskOverlay(true);
    } catch (error) {
      setMaskUploadErrors((errors) => ({
        ...errors,
        [asset.id]: errorMessage(error, "Could not upload the subject mask."),
      }));
    } finally {
      setMaskUploadingId("");
    }
  }

  async function uploadPrepared(asset, file) {
    if (!file || typeof onUploadPreparedBundle !== "function") return;
    setPreparedUploadingId(asset.id);
    try {
      await onUploadPreparedBundle(asset, file);
    } finally {
      setPreparedUploadingId("");
    }
  }

  const cardHandlers = useLatestHandlers({
    preview: (asset) => onPreview(asset, memberAssets),
    toggleItemAck: onToggleItemAck,
    pickReferences: setReferencePickerFor,
    setItemReferences,
    updateCaption,
    toggleHoldout: onToggleHoldout,
    uploadPrepared,
    uploadMask,
    removeAsset: removeUnavailableAsset,
    recaption: (asset) => setCaptionDialog({ type: "item", member: asset }),
  });

  // The unsaved-state chip is now a real affordance (sc-11970): when there are unsaved
  // changes it pairs the "Unsaved changes" pill with a Discard action that reverts the
  // draft to the last saved state (Save lives in the actions row above). A saved, clean
  // dataset shows its version; an unstarted draft shows "Draft".
  const statusPill = dirty ? (
    <span className="dataset-status-row">
      <span className="dataset-status-pill unsaved">
        <span className="dataset-status-dot" aria-hidden="true" />
        Unsaved changes
      </span>
      {typeof discardDraft === "function" ? (
        <button
          className="link-button dataset-discard-button"
          onClick={discardDraft}
          type="button"
        >
          Discard
        </button>
      ) : null}
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
              {activeDataset ? (
                <button
                  aria-label={`Delete dataset ${activeDataset.name}`}
                  className="secondary-action danger"
                  disabled={deletingDataset}
                  onClick={onDeleteDataset}
                  title="Permanently delete this dataset and all its images and captions"
                  type="button"
                >
                  <Icon.Trash size={14} />
                  {deletingDataset ? "Deleting" : "Delete"}
                </button>
              ) : null}
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
              disabled={parquetImportDisabled}
              onClick={() => {
                setParquetError("");
                setParquetDialogOpen(true);
              }}
              title={
                !draftName.trim()
                  ? "Name the dataset before importing"
                  : memberAssets.length
                    ? "Parquet import currently requires an empty dataset"
                    : activeDataset?.id
                      ? "Import URL/caption rows from local Parquet shards"
                      : "Create this empty dataset and import URL/caption rows"
              }
              type="button"
            >
              <Icon.Folder size={14} />
              Import Parquet
            </button>
            {referenceCap > 0 ? (
              <button
                aria-pressed={showEditPairs}
                className={showEditPairs ? "secondary-action strong" : "secondary-action"}
                disabled={hasEditPairs}
                onClick={() => setEditPairsMode?.(!editPairsMode)}
                type="button"
              >
                <Icon.Sliders size={14} />
                Edit pairs
              </button>
            ) : null}
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
            {typeof onGenerateMasks === "function" ? (
              <button
                className="secondary-action"
                disabled={!activeDataset?.id || !memberAssets.length || savingDataset}
                onClick={onGenerateMasks}
                title="Segment the person in every image (SAM3) to build per-image subject masks"
                type="button"
              >
                <Icon.Sliders size={14} />
                Generate subject masks
              </button>
            ) : null}
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
                key={activeDataset?.id ?? "draft"}
                report={readiness}
                renderThumbnail={renderClusterThumbnail}
                itemLabel={(itemId) => assetByItemId.has(itemId) ? imageAssetName(assetByItemId.get(itemId)) : itemId}
                onRemoveDuplicates={datasetDoctor?.onRemoveDuplicates}
                onDeleteSelected={datasetDoctor?.onDeleteSelected}
                busy={savingDataset || deletingDataset}
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

      {activeDataset?.items?.length ? (
        <IrisSidecarPanel items={activeDataset.items} onSidecar={onIrisSidecar} />
      ) : null}

      <section className="dataset-results">
        <div className="dataset-results-head">
          <div className="section-heading">
            <p className="eyebrow">Results</p>
            <h2>Images &amp; captions</h2>
          </div>
          <span className="dataset-count">
            {health.itemCount} image{health.itemCount === 1 ? "" : "s"} · {captionedCount} captioned
            {activeMaskReport ? (
              <>
                {" · "}
                <span aria-label="Subject mask coverage">
                  {activeMaskReport.masked}/{activeMaskReport.total} masked
                </span>
                {activeMaskReport.empty ? (
                  <span className="dataset-mask-attention"> · {activeMaskReport.empty} with no subject</span>
                ) : null}
              </>
            ) : null}
          </span>
          {activeMaskReport?.masked ? (
            <button
              aria-pressed={showMaskOverlay}
              className={["secondary-action", showMaskOverlay ? "active" : ""].filter(Boolean).join(" ")}
              onClick={() => setShowMaskOverlay((shown) => !shown)}
              type="button"
            >
              {showMaskOverlay ? "Hide masks" : "Show masks"}
            </button>
          ) : null}
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

        <div
          className="training-caption-results"
          aria-label="Dataset images and captions"
          onFocusCapture={(event) => setCaptionFocus({ datasetId: activeDataset?.id, assetId: event.target.closest("[data-caption-id]")?.dataset.captionId })}
          onBlurCapture={(event) => {
            const nextId = event.relatedTarget?.closest?.("[data-caption-id]")?.dataset.captionId;
            if (!event.currentTarget.contains(event.relatedTarget) || !nextId) setCaptionFocus(null);
          }}
        >
          {!memberAssets.length ? (
            <div className="empty-panel compact-panel">
              No images yet — use “Add images” or “Import Parquet” to build this dataset.
            </div>
          ) : !visibleAssets.length ? (
            <div className="empty-panel compact-panel">
              {captionFilter === "missing"
                ? "Every image has a caption."
                : "No images carry an unresolved quality finding."}
            </div>
          ) : (
            <DatasetCaptionWindow key={activeDataset?.id ?? "draft"} assets={visibleAssets} renderItem={(asset) => {
              const savedItem = savedItemBySelection.get(asset.id);
              const maskStatus = savedItem ? maskStatusByItemId.get(savedItem.id) : undefined;
              return <DatasetCaptionCard
                key={asset.id}
                asset={asset}
                captionDraft={captionDraftById[asset.id]}
                namedCaptionDraft={namedCaptionsById[asset.id]}
                savedItem={savedItem}
                readinessEntry={readinessByKey?.get(asset.id)}
                readinessPending={readinessLoading && !readiness}
                maskStatus={maskStatus}
                maskSrc={showMaskOverlay && maskStatus?.hasMask ? subjectMaskUrl(maskProjectId, maskStatus) : ""}
                maskError={maskUploadErrors[asset.id]}
                showEditPairs={showEditPairs}
                assetsById={assetsById}
                referenceCap={referenceCap}
                referenceIds={referenceDraftById[asset.id]}
                holdout={itemSplit(holdoutById, asset.id, savedItem) === "holdout"}
                canToggleHoldout={typeof onToggleHoldout === "function"}
                canToggleItemAck={typeof onToggleItemAck === "function"}
                canUploadMask={typeof onUploadMask === "function"}
                preparedUploading={preparedUploadingId === asset.id}
                maskUploading={maskUploadingId === asset.id}
                captioning={captioning}
                handlers={cardHandlers}
              />;
            }} />
          )}
        </div>
      </section>

      {addDialogOpen ? (
        <DatasetAddDialog
          assets={imageAssets}
          characters={characters}
          importing={importingAssets}
          importProgress={importProgress}
          importError={datasetError}
          onCancelImport={onCancelImport}
          memberIds={selectedAssetIds}
          onAdd={addAssets}
          onClose={() => setAddDialogOpen(false)}
          onImport={handleImport}
          fileAccept="image/*,.txt,text/plain,.json,application/json"
          fileHint="Drag images with caption .txt or .json sidecars here, or"
        />
      ) : null}
      {referencePickerFor ? (
        <DatasetAddDialog
          assets={imageAssets}
          characters={characters}
          confirmLabel="Add"
          eyebrow="Reference"
          fileAccept="image/*"
          fileHint="Drop files, or"
          importing={importingAssets}
          importProgress={importProgress}
          importError={datasetError}
          onCancelImport={onCancelImport}
          // Never offer the item's own image or a reference it already has.
          memberIds={[referencePickerFor.selectionId, ...(referenceDraftById[referencePickerFor.selectionId] ?? [])]}
          onAdd={(ids) => {
            const result = addItemReferences?.(referencePickerFor.selectionId, ids);
            if (!result?.dropped) {
              if (importingAssets) onCancelImport?.();
              setReferencePickerFor(null);
            }
            return result;
          }}
          onClose={() => {
            if (importingAssets) onCancelImport?.();
            setReferencePickerFor(null);
          }}
          onImport={async (files) => {
            const picker = referencePickerFor;
            const result = await importItemReferences?.(picker.selectionId, files);
            if (result?.complete) setReferencePickerFor((current) => current === picker ? null : current);
            return result;
          }}
          title={`Add reference images to ${referencePickerFor.name}`}
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
      {parquetDialogOpen ? (
        <>
          {parquetError ? <p className="inline-warning">{parquetError}</p> : null}
          <DatasetParquetImportDialog
            onClose={() => !parquetImporting && setParquetDialogOpen(false)}
            onRun={runParquetImport}
            running={parquetImporting}
          />
        </>
      ) : null}
    </>
  );
}
