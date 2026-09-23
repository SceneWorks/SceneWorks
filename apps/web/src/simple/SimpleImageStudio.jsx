import React, { useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { EditPromptTemplates } from "../components/EditPromptTemplates.jsx";
import { useAppContext } from "../context/AppContext.js";
import { imageModelServesMode } from "../modelEligibility.js";
import { findModelEditLora, loraIsInstalled } from "../presetUtils.js";
import { fitsResolutionOptions } from "../resolutionMemory.js";
import { useHostMemory } from "../hooks/useHostMemory.js";
import { hostMemoryGbForBackend } from "../hostMemory.js";
import { resultGridColumns } from "./breakpoint.js";
import { describeResolution, resolutionSummary } from "./aspect.js";
import { preferredResolution } from "./modelDefaults.js";
import {
  buildSimpleImageRequest,
  referenceStrengthFor,
  resolveSimpleTier,
  workerAdvertises,
} from "./simpleJobs.js";
import { useSimpleRefine } from "./useSimpleRefine.js";
// sc-24113 — Qwen-Image 2.1's controls in the Simple shell. Simple exposes the model, so it gets
// the same transparency toggle and the same rewrite affordance; what it does NOT get is a second,
// simplified version of either, which is why both reuse the advanced modules verbatim.
import { showTransparencyToggle, transparencyPromptSuggestion } from "../qwenAlpha.js";
import { maxReferencesForModel } from "../imageReferenceLimits.js";
import { OrderedReferenceList } from "../components/OrderedReferenceList.jsx";
import {
  dimensionConstraintMessage,
  evaluateModelDimensions,
  modelDimensionConstraints,
} from "../resolutionOverride.js";
import { minStepsForModel } from "../videoModelLimits.js";
import { QwenRewritePromptControl } from "../components/QwenRewritePromptControl.jsx";
import {
  QWEN_IMAGE_2_1_MODEL_ID,
  QWEN_REWRITE_I2I_MODEL_ID,
  QWEN_REWRITE_T2I_MODEL_ID,
} from "../constants.js";
import { useSimpleUi } from "./SimpleUiContext.js";
import { useStudioState } from "./useStudioState.js";
import { useSimpleLoras } from "./useSimpleLoras.js";
import { SimpleLoraField, promptWithKeyword } from "./SimpleLoraField.jsx";
import { SimpleLoraSheet } from "./SimpleLoraSheet.jsx";
import {
  Chips,
  ModelDescription,
  RefinePanel,
  ReferenceTile,
  SheetSelect,
  StudioResults,
  StudioRunStatus,
  StyleStrip,
  jobIsRunning,
  newestLocalJob,
} from "./studioParts.jsx";

// Simple Image Studio (design handoff): mode tabs → prompt → tool tiles → settings bar →
// style strip → Generate → results. Every control is backed by the real catalog and the
// real job queue; the reduced surface is which knobs are SHOWN, not what gets submitted
// (see simpleJobs.js — the payload goes through the same builder the full studio uses).

// The fallback variation ladder, for a model that publishes no `limits.count` (sc-24113 made the
// declared ladder win; before that this hardcoded list was the whole story and a model asking for
// 8 could not be given 8).
const DEFAULT_VARIATION_OPTIONS = [1, 2, 4, 6];
const asChipOptions = (values) => values.map((n) => ({ value: n, label: String(n) }));
const DEFAULT_RESOLUTIONS = ["1024x1024", "1344x768", "768x1344", "896x1152", "1152x896", "1216x832"];
const TIER_SCREEN = "image";

export function SimpleImageStudio() {
  const {
    imageModels = [],
    macCapabilities,
    createImageJob,
    rememberLocalGenerationJob,
    imageLocalJobs = [],
    assets = [],
    recentImageAssets = [],
    visibleWorkers = [],
    loras = [],
    jobs = [],
    createLoraDownloadJob,
    createModelDownloadJob,
    qwenRewritePrompt,
    activeProject,
  } = useAppContext();
  const { breakpoint, openSheet, closeSheet, openGuide, toast, referenceRequest, clearReferenceRequest } =
    useSimpleUi();
  const hostMemory = useHostMemory();
  const memoryBackend = macCapabilities?.macGatingActive ? "mlx" : "candle";
  const unifiedMemoryGb = hostMemoryGbForBackend(hostMemory, memoryBackend);
  const refine = useSimpleRefine("image");

  // Sticky across navigation (this studio unmounts when you leave it) — everything the user
  // set stays set. `submitting` stays local: it belongs to an in-flight submit, not to the
  // form, and a studio that remounted mid-request must not come back looking busy.
  const [mode, setMode] = useStudioState("image", "mode", "text_to_image");
  const [prompt, setPrompt] = useStudioState("image", "prompt", "");
  const [model, setModel] = useStudioState("image", "model", "");
  const [resolution, setResolution] = useStudioState("image", "resolution", "");
  const [variations, setVariations] = useStudioState("image", "variations", 1);
  const [styleId, setStyleId] = useStudioState("image", "styleId", null);
  const [referenceAssetId, setReferenceAssetId] = useStudioState("image", "referenceAssetId", null);
  const [refineOpen, setRefineOpen] = useStudioState("image", "refineOpen", false);
  // Transparency (sc-24113): sticky like every other Simple control, and only RENDERED for a model
  // that advertises four-channel decode — so a value carried over from Qwen 2.1 is inert elsewhere
  // (the payload builder re-checks the capability against the selected model).
  const [transparentBackground, setTransparentBackground] = useStudioState(
    "image",
    "transparentBackground",
    false,
  );
  const [qwenRewriteOpen, setQwenRewriteOpen] = useStudioState("image", "qwenRewriteOpen", false);
  // sc-24113 — the controls Simple did not expose. Every one is STICKY like the rest of this shell
  // and lives behind a disclosure that is COLLAPSED by default, so the reduced surface stays
  // reduced for someone who never opens it.
  const [advancedOpen, setAdvancedOpen] = useStudioState("image", "advancedOpen", false);
  const [steps, setSteps] = useStudioState("image", "steps", "");
  const [seed, setSeed] = useStudioState("image", "seed", "");
  const [negativePrompt, setNegativePrompt] = useStudioState("image", "negativePrompt", "");
  const [guidance, setGuidance] = useStudioState("image", "guidance", "");
  const [widthOverride, setWidthOverride] = useStudioState("image", "widthOverride", "");
  const [heightOverride, setHeightOverride] = useStudioState("image", "heightOverride", "");
  const [extraReferenceIds, setExtraReferenceIds] = useStudioState("image", "extraReferenceIds", []);
  const [submitting, setSubmitting] = useState(false);

  // Models that serve the active tab, under the same capability + Mac-gating predicate
  // the advanced studio uses — never a hardcoded list.
  const models = useMemo(
    () => imageModels.filter((entry) => imageModelServesMode(entry, mode, macCapabilities)),
    [imageModels, mode, macCapabilities],
  );
  const selectedModel = useMemo(
    () => models.find((entry) => entry.id === model) ?? null,
    [models, model],
  );
  const tier = useMemo(
    () =>
      resolveSimpleTier(selectedModel, TIER_SCREEN, {
        convRotEligible: workerAdvertises(visibleWorkers, "int8_convrot"),
        nvfp4Eligible: workerAdvertises(visibleWorkers, "nvfp4"),
        unifiedMemoryGb,
        backend: memoryBackend,
      }),
    [selectedModel, visibleWorkers, unifiedMemoryGb, memoryBackend],
  );

  // Keep the selection valid as the tab (and therefore the eligible set) changes.
  useEffect(() => {
    if (!models.length) {
      return;
    }
    if (!models.some((entry) => entry.id === model)) {
      setModel(models[0].id);
    }
  }, [models, model, setModel]);

  const resolutions = useMemo(() => {
    const declared = selectedModel?.limits?.resolutions?.length
      ? selectedModel.limits.resolutions
      : DEFAULT_RESOLUTIONS;
    const gated = fitsResolutionOptions(selectedModel, declared, unifiedMemoryGb, {
      backend: memoryBackend,
      tier: tier.quantTier,
    });
    return gated.length ? gated : declared;
  }, [selectedModel, unifiedMemoryGb, memoryBackend, tier.quantTier]);

  // Seed from the model's DECLARED default, not `limits.resolutions[0]` — for 16 shipped
  // image models those differ (z_image declares 1024², its list leads with 768²), so [0]
  // would silently start a Simple run at a lower resolution than the same model in the full
  // workspace.
  useEffect(() => {
    // Guard on a RESOLVED model (mirrors ImageStudio's sc-11962 guard). Before the catalog
    // lands, `resolutions` is the generic fallback list — seeding from it seeds 1024², and
    // because almost every model also allows 1024² the value then STICKS and the model's own
    // declared default is never applied. The guard is what makes the seed above actually bite.
    if (!selectedModel) {
      return;
    }
    if (resolutions.length && !resolutions.includes(resolution)) {
      setResolution(preferredResolution(selectedModel, resolutions));
    }
  }, [resolutions, resolution, selectedModel, setResolution]);

  // Resolved against the FULL catalog, not just the picker's recent-20 list: a reference
  // routed in from the Assets preview ("Use as reference") can be any library asset.
  const referenceAsset = useMemo(
    () => assets.find((asset) => asset.id === referenceAssetId) ?? null,
    [assets, referenceAssetId],
  );

  // A RESTORED reference can name an asset the user has since deleted — the id outlives the
  // library now that the studio's settings are durable. Drop it once, when the catalog has
  // actually loaded: `assets` is empty both before the fetch lands and when the library is
  // genuinely empty, and pruning during that window would strip a perfectly good reference
  // (the sc-11962 guard, in the shape VideoStudio's `restoredAssetsValidatedRef` uses).
  // Without this the tile reads "attach an image" while `needsSource` still sees an id, so
  // edit mode stays submittable and sends a dangling reference the job then fails on.
  const restoredReferenceValidated = useRef(false);
  useEffect(() => {
    if (restoredReferenceValidated.current || !assets.length) {
      return;
    }
    restoredReferenceValidated.current = true;
    setReferenceAssetId((current) =>
      current && !assets.some((asset) => asset.id === current) ? null : current,
    );
  }, [assets, setReferenceAssetId]);

  // "Use as reference" from an asset preview lands here. Keyed on the request token so a
  // repeat pick of the SAME asset still re-arms after the user cleared it.
  useEffect(() => {
    if (!referenceRequest) {
      return;
    }
    setReferenceAssetId(referenceRequest.id);
    clearReferenceRequest();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [referenceRequest?.token]);

  // Reference-guided generation on the TEXT tab is a per-model capability (`ui.img2img`):
  // Z-Image, Krea 2, SD3.5, SANA, Ideogram 4 and Boogu declare it, most models don't. On a
  // model without it the tile stays visible (the design's 2-up tile grid) but disabled and
  // says why, rather than accepting a reference the payload would then drop.
  const supportsImg2img = Boolean(selectedModel?.ui?.img2img);

  // The model's OWN variation ladder (sc-24113). `limits.count` was inert here — Simple offered a
  // hardcoded [1,2,4,6] to every model, so Qwen-Image 2.1, whose engine takes 8, could not be given
  // 8 and every model was offered a 6 that nothing declares. Absent ⇒ the historical ladder, so no
  // other model moves.
  const variationOptions = useMemo(() => {
    const declared = selectedModel?.limits?.count;
    const usable = Array.isArray(declared)
      ? declared.filter((value) => Number.isInteger(value) && value > 0)
      : [];
    return asChipOptions(usable.length ? usable : DEFAULT_VARIATION_OPTIONS);
  }, [selectedModel]);
  // Keep the selection legal as the ladder changes with the model, the same shape the resolution
  // seed above uses: an out-of-ladder value would render no chip as active.
  useEffect(() => {
    if (!variationOptions.some((option) => option.value === variations)) {
      setVariations(variationOptions[0].value);
    }
  }, [variationOptions, variations, setVariations]);

  // sc-24113 — Simple's ordered reference list. Simple held ONE `referenceAssetId`, which cannot
  // express the 1-10 ordered list 2.1 takes, so a model whose whole edit surface is that list was
  // reduced to a single image in this shell. The plural list is kept ALONGSIDE the existing single
  // tile rather than replacing it: every other edit model in Simple takes one source image, and the
  // tile is how the design surfaces it.
  const maxSimpleReferences = maxReferencesForModel(selectedModel, 1);
  const supportsOrderedReferences = maxSimpleReferences > 1 && Boolean(selectedModel?.ui?.multiReference);
  // Drop extras when the model cannot use them, so the rail never shows what will not be sent.
  useEffect(() => {
    if (!supportsOrderedReferences && extraReferenceIds.length) {
      setExtraReferenceIds([]);
      return;
    }
    if (extraReferenceIds.length > maxSimpleReferences - 1) {
      setExtraReferenceIds((current) => current.slice(0, maxSimpleReferences - 1));
    }
  }, [supportsOrderedReferences, maxSimpleReferences, extraReferenceIds.length, setExtraReferenceIds]);

  // sc-24113 — the sc-15299 generation axes, read the same way the full studio reads them: an
  // ABSENT `image` block means BOTH are supported, so a model that declares nothing keeps both
  // controls. Simple hid them from every model; that was a surface decision, not a capability one.
  const supportsGuidance = selectedModel?.image?.supportsGuidance !== false;
  const supportsNegativePrompt = selectedModel?.image?.supportsNegativePrompt !== false;

  // The model's own free-size envelope, shared with the full studio so the two shells cannot
  // disagree about what is legal. For a model that declares nothing this is the blanket
  // 256-4096 with no stride, exactly as before.
  const dimensionConstraints = modelDimensionConstraints(selectedModel);
  const dimensionEval = evaluateModelDimensions({
    model: selectedModel,
    resolution,
    widthOverride,
    heightOverride,
  });
  const dimensionError = dimensionConstraintMessage(dimensionEval);

  // The ORDERED list the render conditions on: the armed reference first (it is the one the tile
  // shows), then the extras in the order the user arranged them.
  const orderedReferenceIds = useMemo(
    () => (referenceAssetId ? [referenceAssetId, ...extraReferenceIds] : [...extraReferenceIds]),
    [referenceAssetId, extraReferenceIds],
  );

  // sc-24113 — which Qwen rewriter this request selects, and whether it is installed. Selected by
  // the REQUEST (a reference attached means the editing half) and never by a picker, exactly as in
  // the advanced studio. Simple arms at most one reference, so the list is 0 or 1 long.
  const qwenRewriteReferenceIds = useMemo(
    () => (referenceAssetId ? [referenceAssetId] : []),
    [referenceAssetId],
  );
  const qwenRewriteModel = useMemo(() => {
    if (selectedModel?.id !== QWEN_IMAGE_2_1_MODEL_ID) return null;
    const id = qwenRewriteReferenceIds.length
      ? QWEN_REWRITE_I2I_MODEL_ID
      : QWEN_REWRITE_T2I_MODEL_ID;
    return imageModels.find((entry) => entry.id === id) ?? null;
  }, [imageModels, selectedModel?.id, qwenRewriteReferenceIds.length]);
  const qwenRewriteAvailable =
    Boolean(qwenRewriteModel) &&
    qwenRewriteModel.installState !== "missing" &&
    typeof qwenRewritePrompt === "function";
  const referenceUsable = mode === "edit_image" || supportsImg2img;

  // Krea-style managed image-edit LoRA (epic 10871, sc-11069): Krea 2's edit lane requires an
  // `image_edit`-role LoRA and fails the job without one. Like the advanced studio we MANAGE it
  // rather than exposing a picker — auto-applied when installed, surfaced as a one-click download
  // when not. Null for edit models that need none (Qwen-Image-Edit, FLUX.2), so this is inert
  // for them.
  const editLora = useMemo(
    () => (mode === "edit_image" ? findModelEditLora(loras, selectedModel) : null),
    [mode, loras, selectedModel],
  );
  const editLoraInstalled = loraIsInstalled(editLora);
  const editLoraMissing = Boolean(editLora) && !editLoraInstalled;
  const [editLoraRequested, setEditLoraRequested] = useState(false);
  useEffect(() => {
    if (!editLoraMissing) {
      setEditLoraRequested(false);
    }
  }, [editLoraMissing]);

  // User-picked LoRAs (epic 15404). The managed edit LoRA is excluded from the picker — it is
  // applied automatically above, so offering it here would let the user double-add or toggle
  // off a LoRA the edit lane requires (the advanced studio hides it from its picker too).
  const lora = useSimpleLoras({
    loras,
    selectedModel,
    jobs,
    excludeLoraId: editLoraInstalled ? (editLora?.id ?? null) : null,
    scope: "image",
  });
  const openLoraPicker = () =>
    openSheet({
      title: "Add LoRA",
      body: (
        <SimpleLoraSheet
          atLimit={lora.atLimit}
          excludeIds={[...lora.selectedLoraIds, ...(editLoraInstalled && editLora ? [editLora.id] : [])]}
          onAdd={(picked) => {
            lora.addLora(picked);
            closeSheet();
          }}
          onImportQueued={lora.noteImportJob}
          selectedModel={selectedModel}
        />
      ),
    });

  const latestJob = newestLocalJob(imageLocalJobs);
  const busy = submitting || jobIsRunning(latestJob);
  const needsSource = mode === "edit_image" && !referenceAssetId;
  const canGenerate =
    Boolean(prompt.trim()) &&
    Boolean(model) &&
    Boolean(resolution) &&
    !busy &&
    !needsSource &&
    // sc-24113: a broken free-size override blocks Generate the same way it does in the full
    // studio. Without this the enqueue gate would 400 and the user would meet the model's own
    // envelope as a failed submit rather than as a message under the field they typed in.
    !dimensionEval.invalid &&
    // A required edit LoRA that isn't downloaded blocks the run HERE rather than letting the
    // worker reject it — the job would fail with an error the studio never surfaces.
    !editLoraMissing;

  async function generate() {
    if (!canGenerate) {
      return;
    }
    if (!activeProject) {
      toast("Create or open a workspace first");
      return;
    }
    setSubmitting(true);
    try {
      const request = buildSimpleImageRequest({
        prompt: prompt.trim(),
        mode,
        model,
        resolution,
        count: variations,
        styleId,
        // One armed reference; simpleJobs routes it by mode (edit source vs img2img
        // reference + advanced.strength) and drops it when the model can't use it.
        referenceAssetId,
        // sc-24113 — the ORDERED list, for a model that takes more than one. Passed only when the
        // model declares the surface, so every other model's payload is byte-identical.
        referenceAssetIds: supportsOrderedReferences ? orderedReferenceIds : [],
        // The advanced fold's knobs. Empty string means "the model default", exactly as the full
        // studio's overrides do, so an untouched control adds nothing to the payload.
        steps,
        seed,
        negativePrompt: supportsNegativePrompt ? negativePrompt : "",
        guidance: supportsGuidance ? guidance : "",
        width: dimensionEval.width,
        height: dimensionEval.height,
        supportsImg2img,
        img2imgStrength: referenceStrengthFor(selectedModel),
        // Auto-applied in edit mode; the worker's edit lane rejects the run without it.
        editLora: editLoraInstalled ? editLora : null,
        loras: lora.serializedLoras,
        // sc-24113 — the transparency request. `selectedModel` rides along as the capability
        // source so the builder can re-check it: the toggle is sticky, and a stale `true` must
        // never leak onto a model that would refuse it.
        selectedModel,
        transparentBackground,
        ...tier,
      });
      if (!request) {
        toast("That resolution isn’t valid");
        return;
      }
      const job = await createImageJob(request);
      if (job) {
        rememberLocalGenerationJob?.("image", job);
      }
    } finally {
      setSubmitting(false);
    }
  }

  async function runRefine() {
    const rewritten = await refine.run({
      prompt,
      modelId: model,
      guidePath: selectedModel?.ui?.promptGuide?.path ?? "/prompt-guides/generic-image.md",
    });
    if (rewritten) {
      setPrompt(rewritten);
      setRefineOpen(false);
    }
  }

  const resolutionText = resolution ? resolutionSummary(resolution) : "—";

  return (
    <div className="su-screen">
      <div className="su-tabs" role="tablist">
        <button
          aria-selected={mode === "text_to_image"}
          className={mode === "text_to_image" ? "su-tab active" : "su-tab"}
          onClick={() => setMode("text_to_image")}
          role="tab"
          type="button"
        >
          Text
        </button>
        <button
          aria-selected={mode === "edit_image"}
          className={mode === "edit_image" ? "su-tab active" : "su-tab"}
          onClick={() => setMode("edit_image")}
          role="tab"
          type="button"
        >
          Edit
        </button>
      </div>

      <div>
        <div className="su-prompt-head">
          <label className="su-field-label" htmlFor="su-image-prompt">
            Prompt
          </label>
          <button className="su-pill-btn" onClick={openGuide} type="button">
            <Icon.Book size={13} />
            Prompt guide
          </button>
        </div>
        <textarea
          className="su-textarea"
          id="su-image-prompt"
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Describe the image you want…"
          value={prompt}
        />
        {/* Edit tab only: the built-in edit recipes. Text mode has the Style strip for the
            same job; an instruction like "deblur this image" means nothing to text-to-image. */}
        {mode === "edit_image" ? <EditPromptTemplates onApply={setPrompt} variant="simple" /> : null}
        {/* sc-24113 — Qwen-Image 2.1's OFFICIAL rewriter in the Simple shell. It renders the SAME
            advanced control rather than a simplified twin, and that is deliberate: Simple's generic
            refine drops the review step and replaces the prompt outright, which is exactly what this
            story forbids. The rewrite lands in an editable box with Apply / Keep original beside it
            here too. Absent entirely unless the matching rewriter is already installed — direct
            prompting needs neither, with no download and no prompt to install. */}
        {qwenRewriteAvailable && qwenRewriteOpen ? (
          <QwenRewritePromptControl
            modelId={model}
            onApply={setPrompt}
            onApplyResolution={(value) => {
              if (resolutions.includes(value)) setResolution(value);
            }}
            onDownloadRewriteModel={
              qwenRewriteModel ? () => createModelDownloadJob(qwenRewriteModel) : undefined
            }
            projectId={activeProject?.id ?? ""}
            prompt={prompt}
            referenceAssetIds={qwenRewriteReferenceIds}
            rewriteModel={qwenRewriteModel}
            rewritePrompt={qwenRewritePrompt}
          />
        ) : null}
        {refineOpen ? (
          <RefinePanel
            blurb="Rewrite this prompt with richer detail using the Anubis-8B refiner."
            busy={refine.busy}
            error={refine.error}
            modelMissing={refine.modelMissing}
            onDownloadModel={refine.downloadModel}
            onRun={runRefine}
          />
        ) : null}
      </div>

      <div className="su-tiles">
        <ReferenceTile
          asset={referenceAsset}
          assets={recentImageAssets}
          disabled={!referenceUsable}
          hint={
            !referenceUsable
              ? `${selectedModel?.name ?? "This model"} can’t use a reference image — switch model to use one.`
              : mode === "edit_image"
                ? "Required — tap to pick the image to edit"
                : "Tap to attach an image"
          }
          label={mode === "edit_image" ? "Source image" : "Reference"}
          onChange={setReferenceAssetId}
          required={mode === "edit_image"}
        />
        {qwenRewriteAvailable ? (
          <button
            className={qwenRewriteOpen ? "su-tile active" : "su-tile"}
            onClick={() => {
              setRefineOpen(false);
              setQwenRewriteOpen((open) => !open);
            }}
            type="button"
          >
            <span className="su-tile-head">
              <Icon.Sparkle size={15} />
              Qwen rewriter
            </span>
            <span className="su-tile-sub">Qwen's own rewriter, plus an aspect suggestion</span>
          </button>
        ) : null}
        <button
          className={refineOpen ? "su-tile active" : "su-tile"}
          onClick={() => {
            refine.reset();
            setRefineOpen((open) => !open);
          }}
          type="button"
        >
          <span className="su-tile-head">
            <Icon.Sparkle size={15} />
            Refine
          </span>
          <span className="su-tile-hint">AI-rewrite your prompt</span>
        </button>
      </div>

      <div className="su-settings-bar">
        <div className="su-settings-row">
          <SheetSelect
            label="Model"
            onSelect={setModel}
            options={models.map((entry) => ({
              value: entry.id,
              label: entry.name ?? entry.id,
              active: entry.id === model,
            }))}
            value={selectedModel?.name ?? selectedModel?.id ?? "No image model installed"}
          />
          {/* sc-17162 — see the note in `studioParts.jsx`. `ui.description` had one reader in the
              whole app (the advanced Models card), so Simple identified a model by NAME ALONE. */}
          <ModelDescription model={selectedModel} />
          <SheetSelect
            kind="grid"
            label="Resolution"
            onSelect={setResolution}
            options={resolutions.map((option) => ({
              value: option,
              ...describeResolution(option),
              active: option === resolution,
            }))}
            title="Size & aspect"
            value={resolutionText}
          />
        </div>
        <Chips label="Variations" onChange={setVariations} options={variationOptions} value={variations} />
        {/* sc-24113 — native transparency. Simple exposes the model, so it exposes the toggle; it
            renders only for a model that advertises four-channel decode, so the reduced surface
            stays reduced for every other model in the catalog. */}
        {showTransparencyToggle(selectedModel) ? (
          <label className="su-checkline su-transparency-toggle">
            <input
              checked={transparentBackground}
              onChange={(event) => setTransparentBackground(event.target.checked)}
              type="checkbox"
            />
            <span>Transparent background (RGBA)</span>
          </label>
        ) : null}
        {/* The other half, and the non-obvious one: this model has no transparency MODE — the
            toggle only keeps the alpha channel, and whether it holds a cut-out is decided by the
            PROMPT. Offered as a button that edits the visible prompt, never applied silently; it
            disappears once the prompt already says it. */}
        {transparencyPromptSuggestion(prompt, transparentBackground) ? (
          <button
            className="su-pill-btn su-transparency-hint"
            onClick={() =>
              setPrompt(transparencyPromptSuggestion(prompt, transparentBackground))
            }
            type="button"
          >
            Ask for transparency in the prompt
          </button>
        ) : null}
        {/* Last child of the settings bar, so LoRAs read as a peer of Model / Resolution /
            Variations rather than a card of their own (sc-15370's call, applied to Simple). */}
        <SimpleLoraField
          atLimit={lora.atLimit}
          availableLoras={lora.availableLoras}
          effectiveLoraWeight={lora.effectiveLoraWeight}
          onAddKeyword={(keyword) => setPrompt((current) => promptWithKeyword(current, keyword))}
          onOpenPicker={openLoraPicker}
          onRemove={lora.removeLora}
          onWeightChange={lora.setLoraWeight}
          pendingImports={lora.pendingImports}
          selectedLoras={lora.selectedLoras}
          selectedModel={selectedModel}
        />
      </div>

      {/* sc-24113 — the controls Simple was missing, behind a disclosure that is COLLAPSED by
          default. Simple's contract is a reduced SURFACE, not a reduced payload (simpleJobs.js runs
          the same builder the full studio does), so the right shape for "this model declares these
          controls" is one fold rather than eight more rows in the settings bar. */}
      <details
        className="su-advanced"
        onToggle={(event) => setAdvancedOpen(event.currentTarget.open)}
        open={advancedOpen}
      >
        <summary className="su-advanced-summary">Advanced</summary>
        <div className="su-advanced-body">
          {supportsOrderedReferences ? (
            <div className="su-field">
              <label htmlFor="su-image-add-reference">
                References (up to {maxSimpleReferences}, in order)
              </label>
              {/* The ORDER is part of the request for this family — the template numbers the images
                  and each is visible only to what follows — so the rail shows it and edits it. The
                  reference armed on the tile above is always image 1. */}
              <OrderedReferenceList
                assetIds={orderedReferenceIds}
                labelFor={(id) => assets.find((asset) => asset.id === id)?.name ?? id}
                onChange={(next) => {
                  // The tile owns the first slot, so a reorder writes back through BOTH pieces of
                  // state rather than letting them drift apart.
                  const [first, ...rest] = next;
                  setReferenceAssetId(first ?? null);
                  setExtraReferenceIds(rest);
                }}
              />
              <SheetSelect
                kind="grid"
                label="Add a reference"
                onSelect={(id) =>
                  setExtraReferenceIds((current) =>
                    current.includes(id) || id === referenceAssetId
                      ? current
                      : [...current, id].slice(0, maxSimpleReferences - 1),
                  )
                }
                options={recentImageAssets
                  .filter(
                    (asset) =>
                      asset.id !== referenceAssetId && !extraReferenceIds.includes(asset.id),
                  )
                  .map((asset) => ({ value: asset.id, label: asset.name ?? asset.id }))}
                title="Add a reference"
                value=""
              />
              {extraReferenceIds.length ? (
                <button
                  className="su-pill-btn"
                  onClick={() => setExtraReferenceIds([])}
                  type="button"
                >
                  Clear extra references
                </button>
              ) : null}
            </div>
          ) : null}
          <div className="su-field">
            <label htmlFor="su-image-steps">Steps</label>
            <input
              className="su-input"
              id="su-image-steps"
              // The MODEL's floor, not a hardcoded 1 — the enqueue gate refuses below it.
              min={String(minStepsForModel(selectedModel))}
              max="80"
              onChange={(event) => setSteps(event.target.value)}
              placeholder={String(selectedModel?.defaults?.steps ?? "")}
              type="number"
              value={steps}
            />
          </div>
          <div className="su-field">
            <label htmlFor="su-image-seed">Seed</label>
            <input
              className="su-input"
              id="su-image-seed"
              onChange={(event) => setSeed(event.target.value)}
              placeholder="Random"
              type="number"
              value={seed}
            />
          </div>
          {supportsGuidance ? (
            <div className="su-field">
              <label htmlFor="su-image-guidance">Guidance</label>
              <input
                className="su-input"
                id="su-image-guidance"
                min="0"
                max="30"
                onChange={(event) => setGuidance(event.target.value)}
                placeholder={String(selectedModel?.defaults?.guidanceScale ?? "")}
                step="0.1"
                type="number"
                value={guidance}
              />
            </div>
          ) : null}
          {supportsNegativePrompt ? (
            <div className="su-field">
              <label htmlFor="su-image-negative">Negative prompt</label>
              <textarea
                className="su-textarea"
                id="su-image-negative"
                onChange={(event) => setNegativePrompt(event.target.value)}
                placeholder="What must NOT appear"
                value={negativePrompt}
              />
            </div>
          ) : null}
          <div className="su-field su-free-size">
            <label htmlFor="su-image-width">
              Custom size ({dimensionConstraints.min}–{dimensionConstraints.max} px
              {dimensionConstraints.step > 1 ? `, in steps of ${dimensionConstraints.step}` : ""})
            </label>
            <div className="su-free-size-row">
              <input
                aria-label="Custom width"
                className="su-input"
                id="su-image-width"
                max={dimensionConstraints.max}
                min={dimensionConstraints.min}
                onChange={(event) => setWidthOverride(event.target.value)}
                placeholder={String(dimensionEval.width || "")}
                step={dimensionConstraints.step}
                type="number"
                value={widthOverride}
              />
              <input
                aria-label="Custom height"
                className="su-input"
                max={dimensionConstraints.max}
                min={dimensionConstraints.min}
                onChange={(event) => setHeightOverride(event.target.value)}
                placeholder={String(dimensionEval.height || "")}
                step={dimensionConstraints.step}
                type="number"
                value={heightOverride}
              />
            </div>
            {dimensionError ? <p className="su-error">{dimensionError}</p> : null}
          </div>
        </div>
      </details>

      <StyleStrip onChange={setStyleId} value={styleId} />

      <button
        className={busy ? "su-generate busy" : "su-generate"}
        disabled={!canGenerate}
        onClick={generate}
        type="button"
      >
        {busy ? <span aria-hidden="true" className="su-spinner" /> : null}
        {busy ? "Generating…" : `Generate ${variations} image${variations > 1 ? "s" : ""}`}
      </button>
      {needsSource ? (
        <p className="su-empty">Pick the image you want to edit to start a run.</p>
      ) : null}
      {/* The managed edit LoRA isn't downloaded yet. The worker would reject the run outright
          ("the source-image conditioning is inert"), so offer the one-click fetch instead of
          letting the user discover it as a failed job. */}
      {editLoraMissing ? (
        <div className="su-notice" role="status">
          <Icon.Warning size={16} />
          <span>
            {selectedModel?.name ?? "This model"} needs the {editLora.name ?? "image-edit"} LoRA to
            edit.{" "}
            {editLoraRequested ? (
              "Downloading — track it in the Queue, then generate again."
            ) : (
              <button
                className="su-link"
                onClick={() => {
                  setEditLoraRequested(true);
                  createLoraDownloadJob?.(editLora);
                }}
                type="button"
              >
                Download it
              </button>
            )}
          </span>
        </div>
      ) : null}

      {/* Live run strip: progress + Cancel + the outcome, right under Generate. */}
      <StudioRunStatus job={latestJob} />

      <StudioResults
        assets={assets}
        columns={resultGridColumns(variations, breakpoint)}
        job={latestJob}
        meta={
          selectedModel
            ? `${selectedModel.name ?? selectedModel.id} · ${describeResolution(resolution).sub}`
            : null
        }
        pendingCount={variations}
        type="image"
      />
    </div>
  );
}
