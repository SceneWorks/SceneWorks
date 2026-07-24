import React, { useEffect, useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { useAppContext } from "../context/AppContext.js";
import { imageModelServesMode } from "../modelEligibility.js";
import { findModelEditLora, loraIsInstalled } from "../presetUtils.js";
import { fitsResolutionOptions } from "../resolutionMemory.js";
import { useUnifiedMemoryGb } from "../hooks/useUnifiedMemoryGb.js";
import { resultGridColumns } from "./breakpoint.js";
import { describeResolution, resolutionSummary } from "./aspect.js";
import { preferredResolution } from "./modelDefaults.js";
import { buildSimpleImageRequest, referenceStrengthFor, resolveSimpleTier } from "./simpleJobs.js";
import { useSimpleRefine } from "./useSimpleRefine.js";
import { useSimpleUi } from "./SimpleUiContext.js";
import {
  Chips,
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

const VARIATION_OPTIONS = [1, 2, 4, 6].map((n) => ({ value: n, label: String(n) }));
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
    createLoraDownloadJob,
    activeProject,
  } = useAppContext();
  const { breakpoint, openGuide, toast, referenceRequest, clearReferenceRequest } = useSimpleUi();
  const unifiedMemoryGb = useUnifiedMemoryGb();
  const refine = useSimpleRefine("image");

  const [mode, setMode] = useState("text_to_image");
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState("");
  const [resolution, setResolution] = useState("");
  const [variations, setVariations] = useState(1);
  const [styleId, setStyleId] = useState(null);
  const [referenceAssetId, setReferenceAssetId] = useState(null);
  const [refineOpen, setRefineOpen] = useState(false);
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

  // Keep the selection valid as the tab (and therefore the eligible set) changes.
  useEffect(() => {
    if (!models.length) {
      return;
    }
    if (!models.some((entry) => entry.id === model)) {
      setModel(models[0].id);
    }
  }, [models, model]);

  const resolutions = useMemo(() => {
    const declared = selectedModel?.limits?.resolutions?.length
      ? selectedModel.limits.resolutions
      : DEFAULT_RESOLUTIONS;
    const backend = macCapabilities?.macGatingActive ? "mlx" : "candle";
    const gated = fitsResolutionOptions(selectedModel, declared, unifiedMemoryGb, { backend });
    return gated.length ? gated : declared;
  }, [selectedModel, macCapabilities, unifiedMemoryGb]);

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
  }, [resolutions, resolution, selectedModel]);

  // Resolved against the FULL catalog, not just the picker's recent-20 list: a reference
  // routed in from the Assets preview ("Use as reference") can be any library asset.
  const referenceAsset = useMemo(
    () => assets.find((asset) => asset.id === referenceAssetId) ?? null,
    [assets, referenceAssetId],
  );

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

  const latestJob = newestLocalJob(imageLocalJobs);
  const busy = submitting || jobIsRunning(latestJob);
  const needsSource = mode === "edit_image" && !referenceAssetId;
  const canGenerate =
    Boolean(prompt.trim()) &&
    Boolean(model) &&
    Boolean(resolution) &&
    !busy &&
    !needsSource &&
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
      const tier = resolveSimpleTier(selectedModel, TIER_SCREEN, {
        convRotEligible: workerAdvertises(visibleWorkers, "int8_convrot"),
        nvfp4Eligible: workerAdvertises(visibleWorkers, "nvfp4"),
        unifiedMemoryGb,
      });
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
        supportsImg2img,
        img2imgStrength: referenceStrengthFor(selectedModel),
        // Auto-applied in edit mode; the worker's edit lane rejects the run without it.
        editLora: editLoraInstalled ? editLora : null,
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
        <Chips label="Variations" onChange={setVariations} options={VARIATION_OPTIONS} value={variations} />
      </div>

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

// Mirrors the advanced studios' candle-only tier gates: a tier is only offered when a
// live worker advertises the capability, so an MLX/pre-Ada host never resolves one.
export function workerAdvertises(workers, capability) {
  return (workers ?? []).some(
    (worker) =>
      worker?.status !== "offline" &&
      Array.isArray(worker?.capabilities) &&
      worker.capabilities.includes(capability),
  );
}
