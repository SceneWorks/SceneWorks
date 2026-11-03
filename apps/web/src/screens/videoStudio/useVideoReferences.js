import { VIDEO_PRESET_MODES } from "./videoTierHelpers.js";
import { useEffect, useRef } from "react";
import { pickClosestResolution } from "../../resolutionMatch.js";

// Extracted from VideoStudio along its existing concern boundary.
export function useVideoReferences({
  hasRestoredSource,
  selectedAssetId,
  selectedAsset,
  setSourceAssetOrigin,
  setSourceAssetId,
  setSourceClipAssetId,
  launchRequest,
  setMode,
  setModel,
  setSelectedPresetId,
  setCharacterId,
  setCharacterLookId,
  assets,
  setLastFrameAssetId,
  setBridgeRightClipAssetId,
  setReferenceClipAssetId,
  setReferenceAssetIds,
  setSourceClipAssetIds,
  personTracks,
  setPersonTrackId,
  selectedModel,
  setDuration,
  setResolution,
  setFps,
  sourceAssetId,
  lastFrameAssetId,
  mode
}) {
  // Sync the source from a genuine USER asset-selection TRANSITION after mount — but NEVER from
  // App's non-user auto-default (sc-11964). App derives `selectedAsset = assets.find(id ===
  // selectedAssetId) ?? assets[0]` (App.jsx:768) and `refreshAssets` auto-selects the newest asset
  // once the catalog lands at STARTUP (`setSelectedAssetId((current) => current ?? defaultAsset.id)`,
  // App.jsx:1270) — regardless of the active view. VideoStudio mounts LAZILY (keep-alive: it only
  // mounts when the user first navigates to it), so it almost always mounts AFTER that startup
  // auto-default has already fired, i.e. with `selectedAssetId` ALREADY set to the newest asset. A
  // plain "does selectedAssetId resolve" gate would then push that newest asset onto a source
  // restored from the snapshot and clobber it (empirically: restored "clip-old" -> "clip-new").
  //
  // Model: track the previously-synced selectedAssetId in a ref and sync ONLY on a real post-mount
  // change (selectedAssetId !== prevRef). When a restored source is present, the auto-default must
  // never count as that transition — whether it is ALREADY present at first mount (late mount, the
  // primary flow) OR arrives after mount while the studio was mounted during the restart window
  // (early mount, selectedAssetId still null at mount). So the ref seeds to a sentinel that absorbs
  // the FIRST resolved selection (the auto-default) exactly once, then tracks transitions normally.
  // With NO restored source the ref seeds to null, so the first resolved selection IS a transition
  // and the source defaults to the selected asset exactly as before. A launch (sendAssetToVideo)
  // sets the source directly in the effect below, independent of this sync.
  const AUTO_DEFAULT_PENDING = undefined;

  const prevSelectedAssetIdRef = useRef(hasRestoredSource ? AUTO_DEFAULT_PENDING : null);

  useEffect(() => {
    // Wait for the selection to resolve to a real asset before treating it as a transition; a
    // selectedAssetId whose asset the catalog hasn't landed yet is not yet a user-visible pick.
    if (!selectedAssetId || selectedAsset?.id !== selectedAssetId) {
      return;
    }
    const prevSelectedAssetId = prevSelectedAssetIdRef.current;
    prevSelectedAssetIdRef.current = selectedAssetId;
    // Absorb the first resolved selection (the restart auto-default) once when a restored source
    // is present, so navigating INTO Video Studio can't clobber it. Also a no-op when the value
    // hasn't actually changed since the last sync.
    if (prevSelectedAssetId === AUTO_DEFAULT_PENDING || selectedAssetId === prevSelectedAssetId) {
      return;
    }
    if (selectedAsset.type === "image" || selectedAsset.type === "frame") {
      // The app's selection, not a first-frame pick made in this studio: it stays AMBIENT, so a
      // last-only request never inherits it as a first frame the user never chose.
      setSourceAssetOrigin("ambient");
      setSourceAssetId(selectedAsset.id);
    }
    if (selectedAsset.type === "video") {
      setSourceClipAssetId(selectedAsset.id);
    }
    // Asset object refreshes must not replay a user-selection transition.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAssetId, selectedAsset?.id, selectedAsset?.type]);

  useEffect(() => {
    if (launchRequest?.view !== "Video") {
      return;
    }
    // A recipe launch is owned by the recipe effect below. It carries an `assetId` for selection
    // context but no `mode`, so without this it would fall through to the asset path and both
    // setMode(undefined) and adopt the replayed clip as its own source clip (sc-12324).
    if (launchRequest.recipe) {
      return;
    }
    // sc-10516: a preset launch. `availablePresets` filters on mode + model, so the
    // preset only resolves once both match — set them alongside the id, then let
    // useSavePreset's hydrate effect apply its `defaults`. Returns before the asset
    // paths below, which would otherwise fall through to setMode(undefined).
    if (launchRequest.presetId) {
      if (VIDEO_PRESET_MODES.includes(launchRequest.presetMode)) {
        setMode(launchRequest.presetMode);
      }
      if (launchRequest.presetModel) {
        setModel(launchRequest.presetModel);
      }
      setSelectedPresetId(launchRequest.presetId);
      return;
    }
    if (launchRequest.characterId) {
      setMode(launchRequest.mode ?? "text_to_video");
      setCharacterId(launchRequest.characterId);
      setCharacterLookId(launchRequest.lookId ?? "");
      return;
    }
    if (launchRequest.assetId !== selectedAsset?.id) {
      return;
    }
    setMode(launchRequest.mode);
    if (selectedAsset?.type === "video") {
      setSourceClipAssetId(selectedAsset.id);
    }
    if (selectedAsset?.type === "image" || selectedAsset?.type === "frame") {
      // The user sent THIS asset to the Video Studio: an explicit first frame.
      setSourceAssetOrigin("explicit");
      setSourceAssetId(selectedAsset.id);
    }
    // A persistent launch is applied once per launch/asset identity, not on catalog object refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [launchRequest?.id, selectedAsset?.id, selectedAsset?.type]);

  // Restore-time validation of the persisted asset selections (sc-11964). The snapshot seeds
  // sourceAssetId / referenceAssetIds / etc. at mount, but the asset catalog resolves
  // asynchronously after a restart. Once it first lands, drop any restored id that no longer
  // resolves to a real asset so we never carry a dangling reference to a deleted one. A ref
  // latches on the first non-empty catalog so this validates only the RESTORED values — it
  // won't fight a later user selection whose freshly generated asset the catalog hasn't caught
  // up to yet.
  const restoredAssetsValidatedRef = useRef(false);

  useEffect(() => {
    if (restoredAssetsValidatedRef.current || assets.length === 0) {
      return;
    }
    restoredAssetsValidatedRef.current = true;
    const assetExists = (id) => assets.some((asset) => asset.id === id);
    const dropMissing = (setter) => setter((current) => (current && assetExists(current) ? current : ""));
    dropMissing(setSourceAssetId);
    dropMissing(setLastFrameAssetId);
    dropMissing(setSourceClipAssetId);
    dropMissing(setBridgeRightClipAssetId);
    dropMissing(setReferenceClipAssetId);
    setReferenceAssetIds((current) => (current.some((id) => !assetExists(id)) ? current.filter(assetExists) : current));
    setSourceClipAssetIds((current) => (current.some((id) => !assetExists(id)) ? current.filter(assetExists) : current));
  }, [assets, setBridgeRightClipAssetId, setLastFrameAssetId, setReferenceAssetIds, setReferenceClipAssetId, setSourceAssetId, setSourceClipAssetId, setSourceClipAssetIds]);

  // Same restore-time validation for the restored person-track selection (sc-11964): once the
  // person-track catalog first lands, drop personTrackId if it no longer resolves to a real track.
  const restoredPersonTrackValidatedRef = useRef(false);

  useEffect(() => {
    if (restoredPersonTrackValidatedRef.current || personTracks.length === 0) {
      return;
    }
    restoredPersonTrackValidatedRef.current = true;
    setPersonTrackId((current) => (current && personTracks.some((track) => track.id === current) ? current : ""));
  }, [personTracks, setPersonTrackId]);

  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    setDuration((current) => {
      const options = selectedModel.limits?.durations ?? [4, 6, 8, 10];
      return options.includes(Number(current)) ? current : selectedModel.defaults?.duration ?? options[0];
    });
    setResolution((current) => {
      const options = selectedModel.limits?.resolutions ?? ["768x512"];
      return options.includes(current) ? current : selectedModel.defaults?.resolution ?? options[0];
    });
    setFps((current) => {
      const options = selectedModel.limits?.fps ?? [24, 25, 30];
      return options.includes(Number(current)) ? current : selectedModel.defaults?.fps ?? options[0];
    });
    // Reconcile defaults only when model identity changes, not on catalog object refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel?.id]);

  // I2V: when the user picks a source image (or first/last frame) after mount,
  // snap resolution to whichever option in the model's list best matches the
  // image's aspect ratio. The ref tracks the last-seen id so polling-driven
  // assets refreshes don't re-fire, and so the saved snapshot's resolution is
  // preserved when the asset id is just being restored on mount.
  const i2vSourceAssetId = sourceAssetId || lastFrameAssetId;

  const lastI2vAssetIdRef = useRef(i2vSourceAssetId);

  useEffect(() => {
    if (i2vSourceAssetId === lastI2vAssetIdRef.current) {
      return;
    }
    lastI2vAssetIdRef.current = i2vSourceAssetId;
    if (!i2vSourceAssetId) return;
    if (!["image_to_video", "first_last_frame"].includes(mode)) return;
    const asset = assets.find((item) => item.id === i2vSourceAssetId);
    const width = asset?.file?.width;
    const height = asset?.file?.height;
    if (!width || !height) return;
    const match = pickClosestResolution(width, height, selectedModel?.limits?.resolutions);
    if (match) setResolution(match);
    // Source/model identities drive this probe; an unrelated asset catalog refresh does not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [i2vSourceAssetId, mode, selectedModel?.id, assets]);
  return { lastI2vAssetIdRef };
}
