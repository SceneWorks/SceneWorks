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
import { ImageEditSourcePickerField } from "./AssetPicker.jsx";
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
  // img2img double-duty (epic 8588 slice A, sc-8593): when the selected model supports img2img,
  // the SAME picked reference can drive generation (reference-guided / latent-init) via a strength
  // slider — in addition to (or instead of) the describe → prompt flow above. All additive + default
  // off, so the Ideogram-JSON + all-t2i describe consumers are byte-unaffected. `onReferenceAssetChange`
  // lifts the picked asset id so the parent can send it as `referenceAssetId`. `showImg2imgStrength`
  // also un-gates the picker without the vision captioner (img2img needs no captioner).
  onReferenceAssetChange,
  showImg2imgStrength = false,
  img2imgStrength = 0.5,
  onImg2imgStrengthChange,
  img2imgStrengthConfig,
}) {
  const [referenceAssetId, setReferenceAssetId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

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
    onReferenceAssetChange?.(assetId);
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
      const result = await onCaption(referenceAssetId);
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
        ready={visionCaptionReady || showImg2imgStrength}
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
        {/* Describe → prompt (epic 8203): only when the vision captioner is present. An img2img-only
            model (no captioner) still gets the picker + strength slider below via `showImg2imgStrength`. */}
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
        {/* img2img strength (epic 8588 slice A, sc-8593): the SAME picked reference guides generation
            (reference-guided / latent-init) at this strength. Shown only for img2img-capable models
            once a reference is chosen. */}
        {showImg2imgStrength && referenceAssetId ? (
          <label className="reference-strength img2img-strength">
            {img2imgStrengthConfig?.label ?? "Reference strength"}
            <input
              max={img2imgStrengthConfig?.max ?? 1}
              min={img2imgStrengthConfig?.min ?? 0}
              onChange={(event) => onImg2imgStrengthChange?.(Number(event.target.value))}
              step={img2imgStrengthConfig?.step ?? 0.05}
              type="range"
              value={img2imgStrength}
            />
            <span>{Number(img2imgStrength).toFixed(2)}</span>
          </label>
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
