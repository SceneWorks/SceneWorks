import { useEffect, useMemo, useRef, useState } from "react";
import { assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { useValidation } from "../validation/useValidation.js";
import {
  captionDraftsFromDataset, datasetHealth, datasetItemSelectionKey,
  datasetOwnedAssets, datasetReferenceAssets, datasetSaveValidation,
  normalizeDatasetAssetIds, referenceDraftsDiffer, referenceDraftsFromDataset,
} from "./datasetHelpers.js";

// Owns the editable dataset and its saved baseline. Async loading/upload/job effects
// remain in the screen, which retains their project and generation ownership guards.
export function useDatasetDraft({ activeProject, assets }) {
  const [activeDataset, setActiveDataset] = useState(null);
  const [datasetError, setDatasetError] = useState("");
  const [datasetMessage, setDatasetMessage] = useState("");
  const [draftName, setDraftName] = useState("");
  const [busyDatasetId, setBusyDatasetId] = useState("");
  const [uploadedDatasetAssets, setUploadedDatasetAssets] = useState([]);
  const [savingDataset, setSavingDataset] = useState(false);
  const [deletingDataset, setDeletingDataset] = useState(false);
  const [selectedAssetIds, setSelectedAssetIds] = useState([]);
  // Owning character for this dataset (sc-2022). Seeded from the loaded dataset,
  // set when images are imported from the Character tab, threaded into the save
  // payload. "" = a general (unassociated) dataset.
  const [associatedCharacterId, setAssociatedCharacterId] = useState("");
  // Single source of truth for caption edits, keyed by selection id (sc-2025):
  // seeded from the saved dataset items, updated as the user edits an item's
  // caption or imports a .txt sidecar, and threaded into the dataset payload on
  // save. Replaces the old importedCaptions + rename-caption draft split.
  const [captionDraftById, setCaptionDraftById] = useState({});
  // Per-item named captions (`{ field: text }`) parsed from freshly imported `.json` sidecars,
  // keyed by selection id and threaded into the save payload (sc-25685). Saved items carry theirs
  // on the item itself.
  const [namedCaptionsById, setNamedCaptionsById] = useState({});
  // sc-25689: per-item holdout overrides (`{ [selectionId]: boolean }`) until the dataset is saved.
  const [holdoutById, setHoldoutById] = useState({});
  // Instruction-edit pairs (sc-24161): each item's ORDERED reference images, keyed by item
  // selection id -> [reference selection id, ...]. Seeded from the saved dataset like captions and
  // threaded into the save payload; the item image is the edit target and its caption the
  // instruction. `editPairsMode` reveals the per-item reference rails for a dataset with none yet.
  const [referenceDraftById, setReferenceDraftById] = useState({});
  const [editPairsMode, setEditPairsMode] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [renamePrefix, setRenamePrefix] = useState("");
  const datasetDraftRevisionRef = useRef(0);
  const datasetAssets = useMemo(
    () => [
      ...datasetOwnedAssets(activeDataset, activeProject?.id, assets),
      // Stored edit-pair references that are not library assets render (and re-save) through
      // synthetic dataset-owned assets, exactly like dataset-owned items (sc-24161).
      ...datasetReferenceAssets(activeDataset, activeProject?.id, assets),
    ],
    [activeDataset, activeProject?.id, assets],
  );
  const imageAssets = useMemo(() => {
    const merged = [
      ...assets.filter((asset) => assetCanRenderAsImage(asset)),
      ...uploadedDatasetAssets,
      ...datasetAssets,
    ];
    const seen = new Set();
    return merged.filter((asset) => {
      if (!asset?.id || seen.has(asset.id)) {
        return false;
      }
      seen.add(asset.id);
      return true;
    });
  }, [assets, datasetAssets, uploadedDatasetAssets]);
  const assetsById = useMemo(() => new Map(imageAssets.map((asset) => [asset.id, asset])), [imageAssets]);
  const unavailableAssetIds = useMemo(
    () => selectedAssetIds.filter((assetId) => !assetsById.has(assetId)),
    [assetsById, selectedAssetIds],
  );
  // Members shown in the editor body: the dataset's own items (available
  // assets), in selection order. Captions come from the saved dataset items or
  // freshly imported .txt sidecars, keyed by selection id.
  const memberAssets = useMemo(
    () => selectedAssetIds.map((id) => assetsById.get(id)).filter(Boolean),
    [assetsById, selectedAssetIds],
  );
  // Thumbnail for the compact dataset selector (sc-2025): the list summary's
  // server-resolved cover image (first item), falling back to the open dataset's
  // first loaded member.
  const datasetThumbAsset = (dataset) => {
    if (dataset?.coverPath) {
      return { projectId: activeProject?.id, type: "image", file: { path: dataset.coverPath } };
    }
    if (dataset?.id === selectedDatasetId && memberAssets[0]) {
      return memberAssets[0];
    }
    return null;
  };
  const health = useMemo(
    () => datasetHealth({ activeDataset, imageAssets, selectedAssetIds }),
    [activeDataset, imageAssets, selectedAssetIds],
  );
  const originalAssetIds = useMemo(() => normalizeDatasetAssetIds(activeDataset, assets), [activeDataset, assets]);
  // Caption edits also make the dataset dirty so the single Save persists them.
  const captionsDirty = useMemo(
    () =>
      Boolean(activeDataset) &&
      (activeDataset.items ?? []).some((item, index) => {
        const draft = captionDraftById[datasetItemSelectionKey(activeDataset, item, index)];
        return draft && (draft.text ?? "") !== (item.caption?.text ?? "");
      }),
    [activeDataset, captionDraftById],
  );
  // Reference edits (add / remove / reorder, sc-24161) also make the dataset dirty.
  const savedReferenceDrafts = useMemo(
    () => referenceDraftsFromDataset(activeDataset),
    [activeDataset],
  );
  const referencesDirty =
    Boolean(activeDataset) && referenceDraftsDiffer(referenceDraftById, savedReferenceDrafts, selectedAssetIds);
  const dirty =
    Boolean(activeDataset) &&
    (draftName.trim() !== activeDataset.name ||
      associatedCharacterId !== (activeDataset.characterId ?? "") ||
      selectedAssetIds.length !== originalAssetIds.length ||
      selectedAssetIds.some((id, index) => id !== originalAssetIds[index]) ||
      captionsDirty ||
      referencesDirty ||
      Object.keys(holdoutById).length > 0);
  // Refresh awaits the catalog request. Read the current draft state after that
  // await so a rename begun while refresh is in flight is never overwritten by
  // the freshly loaded dataset object.
  const dirtyRef = useRef(dirty);
  dirtyRef.current = dirty;
  const activeDatasetIdRef = useRef(activeDataset?.id ?? "");
  activeDatasetIdRef.current = activeDataset?.id ?? "";
  useEffect(() => {
    datasetDraftRevisionRef.current += 1;
  }, [activeDataset?.id, draftName, selectedAssetIds, associatedCharacterId, captionDraftById, referenceDraftById]);
  // Unsaved work the destructive-transition guards protect (sc-11970): an edited saved
  // dataset (dirty) OR a brand-new dataset the user has started building (name / members /
  // character) but not yet saved. A new draft isn't "dirty" (no baseline), but discarding
  // it is just as destructive, so the guard covers it too.
  const hasUnsavedDatasetDraft = activeDataset
    ? dirty
    : Boolean(draftName.trim() || selectedAssetIds.length || associatedCharacterId);
  // One summary gates Save and feeds the chip row, so they cannot disagree (epic 10644).
  // `savingDataset` and "nothing changed" are non-validation gates — like submittingJob on
  // the Train button — so they stay in the disabled expression rather than becoming issues.
  const saveDraft = useMemo(() => ({ name: draftName, selectedAssetIds }), [draftName, selectedAssetIds]);
  const saveContext = useMemo(() => ({ health }), [health]);
  const saveValidity = useValidation(datasetSaveValidation, saveDraft, saveContext);
  const canSave = saveValidity.ready && !savingDataset && (!activeDataset || dirty);
  // Drops a member (or an unavailable/orphaned id) from the dataset selection.
  function removeUnavailableAsset(assetId) {
    setDatasetMessage("");
    setSelectedAssetIds((current) => current.filter((id) => id !== assetId));
  }

  function clearDatasetDraft() {
    setActiveDataset(null);
    setDatasetError("");
    setDatasetMessage("");
    setDraftName("");
    setSelectedAssetIds([]);
    setAssociatedCharacterId("");
    setUploadedDatasetAssets([]);
    setSelectedDatasetId("");
    setReferenceDraftById({});
    setEditPairsMode(false);
  }

  // Revert the draft to the last saved state (sc-11970): the actionable half of the
  // "Unsaved changes" pill. A saved dataset restores from `activeDataset` in place (no
  // reload); an unstarted draft clears. `activeDataset` is unchanged, so the caption-seed
  // effect won't fire — reset the per-field drafts explicitly.
  function discardDraft() {
    if (!hasUnsavedDatasetDraft) {
      return;
    }
    if (!activeDataset) {
      clearDatasetDraft();
      return;
    }
    setDatasetError("");
    setDatasetMessage("");
    setDraftName(activeDataset.name ?? "");
    setSelectedAssetIds(normalizeDatasetAssetIds(activeDataset, assets));
    setAssociatedCharacterId(activeDataset.characterId ?? "");
    setCaptionDraftById(captionDraftsFromDataset(activeDataset));
    setReferenceDraftById(referenceDraftsFromDataset(activeDataset));
  }

  // Replace one item's ordered reference list (reorder / remove from the reference rail, sc-24161).
  function setItemReferences(selectionId, referenceIds) {
    setDatasetMessage("");
    setDatasetError("");
    setReferenceDraftById((current) => ({ ...current, [selectionId]: [...(referenceIds ?? [])] }));
  }

  // Editing a caption marks it as manually authored (sc-2025) — caption source
  // is detected, never picked by the user.
  function updateCaption(selectionId, text) {
    setDatasetMessage("");
    setCaptionDraftById((current) => ({
      ...current,
      [selectionId]: { text, source: "manual" },
    }));
  }

  return {
    activeDataset, setActiveDataset, datasetError, setDatasetError, datasetMessage, setDatasetMessage,
    draftName, setDraftName, busyDatasetId, setBusyDatasetId, setUploadedDatasetAssets, savingDataset,
    setSavingDataset, deletingDataset, setDeletingDataset, selectedAssetIds, setSelectedAssetIds,
    associatedCharacterId, setAssociatedCharacterId, captionDraftById, setCaptionDraftById, namedCaptionsById,
    setNamedCaptionsById, holdoutById, setHoldoutById, referenceDraftById, setReferenceDraftById,
    editPairsMode, setEditPairsMode, selectedDatasetId, setSelectedDatasetId, renamePrefix, setRenamePrefix,
    datasetDraftRevisionRef, imageAssets, assetsById, unavailableAssetIds, memberAssets, datasetThumbAsset,
    health, dirty, dirtyRef, activeDatasetIdRef, hasUnsavedDatasetDraft, saveValidity, canSave,
    removeUnavailableAsset, clearDatasetDraft, discardDraft, setItemReferences, updateCaption
  };
}
