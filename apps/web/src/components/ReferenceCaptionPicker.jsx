// Shared reference-image → prompt picker (epic 8203, sc-8208).
//
// Factored out of StructuredPromptBuilder so BOTH reference-image flows share one
// component:
//   * Ideogram 4 (structuredPrompt): reference image → editable JSON caption (epic 8102).
//   * Every other text-to-image model: reference image → plain-text description (prose or
//     booru tags) that fills the prompt textarea (epic 8203).
//
// The component owns the picker UI, the selected-asset state, the busy/error state, the
// auto-preset-resolution probe (sc-8109), and the ModelAvailabilityGate (sc-8110) that
// offers the vision-captioner download when it is missing. It is format-agnostic: the
// parent supplies `onCaption(assetId)` (run the job + parse) and `onApply(result)` (apply
// the result to its own state). C1: the image is captioning-only — never sent to generation.

import React, { useEffect, useState } from "react";
import { AssetPickerField, ImageEditSourcePickerField } from "./AssetPicker.jsx";
import { assetUrl } from "./assetMedia.jsx";
import { ModelAvailabilityGate } from "./ModelAvailabilityGate.jsx";

export default function ReferenceCaptionPicker({
  // Run the vision job for the picked asset and resolve to a result (the parent parses:
  // a caption object for Ideogram, the raw prose/tags string for describe). A falsy result
  // means the reply was not usable.
  onCaption,
  // Apply a truthy result to the parent's state (inject the caption / fill the prompt).
  onApply,
  // sc-8109 seam: invoked with the uploaded image's natural (width, height) so the parent
  // can auto-preset the generation resolution to the nearest aspect.
  onReferenceImageLoaded,
  referenceAssets = [],
  referenceCharacters = [],
  importAsset,
  projectId,
  // Copy that differs between the JSON (Ideogram) and prose/tags surfaces.
  hint,
  buttonLabel = "✨ Generate from image",
  busyLabel = "Working…",
  emptyMessage = "The image didn’t produce a usable result. Try another reference.",
  errorFallback = "Could not process the image.",
  gateTitle = "Reference-image captioning needs a model",
  gateDescription = "Download the vision captioner to turn a reference image into a prompt. It runs locally on the native worker; the image is only used to write the prompt.",
  // Reference-image caption gate (sc-8110).
  visionCaptionReady = true,
  visionCaptionOffers = [],
  visionCaptionDownloadJobs = [],
  onDownloadModel,
  onOpenModels,
  onOpenQueue,
  onCancelJob,
  // Multi-image "mood board" (epic 8588, sc-8595): when true, an additional multi-select gallery lets the
  // user add MORE reference images beyond the primary. `onCaption` is then called with the FULL array of
  // ids ([primary, ...extras]) and the worker synthesizes ONE prompt/caption from the shared aesthetic. A
  // single reference (no extras) still calls `onCaption` with the plain id string — so describe/Ideogram
  // consumers that don't opt in are byte-unaffected. `moodBoardMax` bounds the board (server also caps).
  showMoodBoard = false,
  moodBoardMax = 6,
}) {
  const [referenceAssetId, setReferenceAssetId] = useState("");
  const [moodBoardIds, setMoodBoardIds] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // The extra mood-board picks, minus the primary (it is already the first image) and capped. This is
  // what actually augments the describe call; `referenceAssetId` stays the primary reference.
  const moodBoardExtras = moodBoardIds
    .filter((id) => id && id !== referenceAssetId)
    .slice(0, Math.max(0, moodBoardMax - 1));

  // Feed the uploaded image's natural dimensions to the sc-8109 auto-preset seam.
  function reportReferenceDimensions(asset) {
    if (typeof onReferenceImageLoaded !== "function" || !asset) return;
    const src = assetUrl(asset);
    if (!src || typeof Image === "undefined") return;
    const probe = new Image();
    probe.onload = () => {
      if (probe.naturalWidth && probe.naturalHeight) {
        onReferenceImageLoaded(probe.naturalWidth, probe.naturalHeight);
      }
    };
    probe.src = src;
  }

  function handleReferenceChange(assetId) {
    setReferenceAssetId(assetId);
    setError("");
  }

  // Run the auto-preset from an effect on the SELECTED id rather than the picker's onChange: a freshly
  // imported/dragged reference lands in `referenceAssets` a render AFTER its id is set, so the
  // onChange-time `find` missed it and the Aspect stayed at the 1:1 default (sc-8220). Keying on both
  // the id and the list re-runs once the asset resolves, covering select / import / drag / character
  // paths uniformly.
  useEffect(() => {
    if (!referenceAssetId) return;
    const asset = referenceAssets.find((item) => item.id === referenceAssetId);
    if (asset) reportReferenceDimensions(asset);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [referenceAssetId, referenceAssets]);

  async function handleCaption() {
    if (typeof onCaption !== "function" || !referenceAssetId || busy) return;
    setBusy(true);
    setError("");
    try {
      // A mood board sends the FULL ordered list ([primary, ...extras]); a lone reference keeps the
      // scalar id so consumers that never opted into `showMoodBoard` see the unchanged string contract.
      const arg = moodBoardExtras.length > 0
        ? [referenceAssetId, ...moodBoardExtras]
        : referenceAssetId;
      const result = await onCaption(arg);
      if (result) {
        onApply(result);
      } else {
        setError(emptyMessage);
      }
    } catch (e) {
      setError(e?.message || errorFallback);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="structured-reference">
      <ModelAvailabilityGate
        ready={visionCaptionReady}
        title={gateTitle}
        description={gateDescription}
        offers={visionCaptionOffers}
        downloadJobs={visionCaptionDownloadJobs}
        onDownload={onDownloadModel}
        onOpenModels={onOpenModels}
        onOpenQueue={onOpenQueue}
        onCancelJob={onCancelJob}
      >
        {hint ? <p className="structured-hint">{hint}</p> : null}
        <ImageEditSourcePickerField
          assets={referenceAssets}
          buttonLabel="Select reference image"
          changeLabel="Change reference"
          characters={referenceCharacters}
          emptyLabel="No reference image selected"
          importAsset={importAsset}
          label="Reference image"
          onChange={handleReferenceChange}
          projectId={projectId}
          value={referenceAssetId}
        />
        {/* Mood board (epic 8588, sc-8595): add MORE reference images beyond the primary; the model
            synthesizes ONE prompt/caption from the aesthetic they share. Only once a primary is chosen
            (it is the first image). The extras come from the project library (the primary picker keeps
            import/characters). Capped at moodBoardMax − 1 extras; the server enforces the same ceiling. */}
        {showMoodBoard && referenceAssetId ? (
          <AssetPickerField
            assets={referenceAssets.filter((asset) => asset.id !== referenceAssetId)}
            buttonLabel="Add mood-board images"
            changeLabel="Edit mood board"
            emptyLabel="Add more images to blend their style (optional)"
            label="Mood board (optional)"
            multiple
            onChange={setMoodBoardIds}
            values={moodBoardExtras}
          />
        ) : null}
        {/* Describe → prompt (epic 8203): the vision captioner turns the picked reference (or the whole
            mood board) into prompt text. img2img reference-guidance is a SEPARATE prompt-tool tile now
            (sc-10195), so this component is purely describe + mood board. */}
        {visionCaptionReady ? (
          <button
            type="button"
            className="secondary-action"
            disabled={!referenceAssetId || busy}
            onClick={handleCaption}
          >
            {busy ? busyLabel : buttonLabel}
          </button>
        ) : null}
        {error ? (
          <p className="structured-error" role="alert">
            {error}
          </p>
        ) : null}
      </ModelAvailabilityGate>
    </div>
  );
}
