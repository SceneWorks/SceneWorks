import React, { useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { useAppContext } from "../context/AppContext.js";
import { videoModelServesMode } from "../modelEligibility.js";
import { WAN_MOE_PAIRED_LORA_MODEL_IDS } from "../constants.js";
import { buildSimpleVideoRequest } from "./simpleJobs.js";
import { describeResolution, resolutionSummary } from "./aspect.js";
import { preferredDuration, preferredVideoResolution } from "./modelDefaults.js";
import { useSimpleRefine } from "./useSimpleRefine.js";
import { useSimpleUi } from "./SimpleUiContext.js";
import { useStudioState } from "./useStudioState.js";
import { useSimpleLoras } from "./useSimpleLoras.js";
import { SimpleLoraField, promptWithKeyword } from "./SimpleLoraField.jsx";
import { SimpleLoraSheet } from "./SimpleLoraSheet.jsx";
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

// Simple Video Studio (design handoff): the Image pattern with Duration in place of
// Variations, one full-width reference-image tile, and a single 16:9 result.
//
// Only the two headline modes are exposed — Text → Video and Image → Video. The other
// ten video modes the full studio offers (extend, bridge, replace-person, ads2v, …)
// stay in the advanced workspace by design; this is the "reduced surface", and the
// switch to Advanced is one click away in the sidebar.

const DEFAULT_DURATIONS = [3, 5, 8, 10];
const DEFAULT_RESOLUTIONS = ["1280x720", "720x1280", "720x720", "854x480"];
// Context, not a warning (design handoff): the pairing is how the model works, not something
// the user got wrong, so it reads as a hint line rather than a bordered notice.
const MOE_LORA_HINT =
  "Wan 2.2 A14B LoRAs are trained as a high/low-noise pair — selecting one loads both experts.";

export function SimpleVideoStudio() {
  const {
    videoModels = [],
    macCapabilities,
    createVideoJob,
    rememberLocalGenerationJob,
    videoLocalJobs = [],
    assets = [],
    recentImageAssets = [],
    loras = [],
    jobs = [],
    activeProject,
  } = useAppContext();
  const { openSheet, closeSheet, openGuide, toast } = useSimpleUi();
  const refine = useSimpleRefine("video");

  // Sticky across navigation — see the same block in SimpleImageStudio.
  const [mode, setMode] = useStudioState("video", "mode", "text_to_video");
  const [prompt, setPrompt] = useStudioState("video", "prompt", "");
  const [model, setModel] = useStudioState("video", "model", "");
  const [resolution, setResolution] = useStudioState("video", "resolution", "");
  const [duration, setDuration] = useStudioState("video", "duration", 5);
  const [styleId, setStyleId] = useStudioState("video", "styleId", null);
  const [sourceAssetId, setSourceAssetId] = useStudioState("video", "sourceAssetId", null);
  const [refineOpen, setRefineOpen] = useStudioState("video", "refineOpen", false);
  const [submitting, setSubmitting] = useState(false);

  const models = useMemo(
    () => videoModels.filter((entry) => videoModelServesMode(entry, mode, macCapabilities)),
    [videoModels, mode, macCapabilities],
  );
  const selectedModel = useMemo(
    () => models.find((entry) => entry.id === model) ?? null,
    [models, model],
  );

  useEffect(() => {
    if (models.length && !models.some((entry) => entry.id === model)) {
      setModel(models[0].id);
    }
  }, [models, model, setModel]);

  const resolutions = selectedModel?.limits?.resolutions?.length
    ? selectedModel.limits.resolutions
    : DEFAULT_RESOLUTIONS;
  const durations = selectedModel?.limits?.durations?.length
    ? selectedModel.limits.durations
    : DEFAULT_DURATIONS;

  // Both seeds read the model's DECLARED default rather than `limits.*[0]`, which is only
  // the first ALLOWED value. For nearly every shipped video model they differ — LTX-2.3
  // declares 6s but lists 4s first; Wan 2.2 A14B declares 1280×720 but lists 832×480 first —
  // so [0] silently started a Simple run shorter and smaller than the same model in the full
  // workspace.
  // Both effects guard on a RESOLVED model (mirrors ImageStudio's sc-11962 guard): before the
  // catalog lands these lists are the generic fallbacks, and seeding from them locks in a
  // value the model also happens to allow — after which the model's own declared default is
  // never applied.
  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    if (resolutions.length && !resolutions.includes(resolution)) {
      setResolution(preferredVideoResolution(selectedModel, resolutions));
    }
  }, [resolutions, resolution, selectedModel, setResolution]);

  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    if (durations.length && !durations.includes(duration)) {
      setDuration(preferredDuration(selectedModel, durations));
    }
  }, [durations, duration, selectedModel, setDuration]);

  // Resolved against the FULL catalog rather than the picker's recent-20 list. That list is
  // what the PICKER offers; a source restored from a previous session can have aged out of it
  // while the asset is still perfectly valid, and resolving against it would blank the tile
  // (and, below, wrongly prune the id).
  const sourceAsset = useMemo(
    () => assets.find((asset) => asset.id === sourceAssetId) ?? null,
    [assets, sourceAssetId],
  );

  // Same one-shot validation as the Image studio: a restored source can name an asset the
  // user has since deleted, and `needsSource` reads the id — so without this, Image→Video
  // stays submittable with a reference that no longer exists. Guarded on a LOADED catalog.
  const restoredSourceValidated = useRef(false);
  useEffect(() => {
    if (restoredSourceValidated.current || !assets.length) {
      return;
    }
    restoredSourceValidated.current = true;
    setSourceAssetId((current) =>
      current && !assets.some((asset) => asset.id === current) ? null : current,
    );
  }, [assets, setSourceAssetId]);

  // User-picked LoRAs (epic 15404) — same hook, same eligibility and cap as the Image studio;
  // the video lane just has no auto-applied managed LoRA to exclude.
  const lora = useSimpleLoras({ loras, selectedModel, jobs, scope: "video" });
  const openLoraPicker = () =>
    openSheet({
      title: "Add LoRA",
      body: (
        <SimpleLoraSheet
          atLimit={lora.atLimit}
          excludeIds={lora.selectedLoraIds}
          onAdd={(picked) => {
            lora.addLora(picked);
            closeSheet();
          }}
          onImportQueued={lora.noteImportJob}
          selectedModel={selectedModel}
        />
      ),
    });

  const latestJob = newestLocalJob(videoLocalJobs);
  const busy = submitting || jobIsRunning(latestJob);
  const needsSource = mode === "image_to_video" && !sourceAssetId;
  const canGenerate =
    Boolean(prompt.trim()) && Boolean(model) && Boolean(resolution) && !busy && !needsSource;

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
      const request = buildSimpleVideoRequest({
        prompt: prompt.trim(),
        mode,
        model,
        resolution,
        duration,
        styleId,
        sourceAssetId,
        loras: lora.serializedLoras,
      });
      if (!request) {
        toast("That resolution isn’t valid");
        return;
      }
      const job = await createVideoJob(request);
      if (job) {
        rememberLocalGenerationJob?.("video", job);
      }
    } finally {
      setSubmitting(false);
    }
  }

  async function runRefine() {
    const rewritten = await refine.run({
      prompt,
      modelId: model,
      guidePath: selectedModel?.ui?.promptGuide?.path ?? "/prompt-guides/generic-video.md",
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
          aria-selected={mode === "text_to_video"}
          className={mode === "text_to_video" ? "su-tab active" : "su-tab"}
          onClick={() => setMode("text_to_video")}
          role="tab"
          type="button"
        >
          Text&nbsp;→&nbsp;Video
        </button>
        <button
          aria-selected={mode === "image_to_video"}
          className={mode === "image_to_video" ? "su-tab active" : "su-tab"}
          onClick={() => setMode("image_to_video")}
          role="tab"
          type="button"
        >
          Image&nbsp;→&nbsp;Video
        </button>
      </div>

      <div>
        <div className="su-prompt-head">
          <label className="su-field-label" htmlFor="su-video-prompt">
            Prompt
          </label>
          {/* Video keeps ONE full-width Reference tile (design), so Refine rides here as a
              pill beside the guide rather than as a second tile. */}
          <span className="su-pill-row">
            <button
              aria-expanded={refineOpen}
              className={refineOpen ? "su-pill-btn active" : "su-pill-btn"}
              onClick={() => {
                refine.reset();
                setRefineOpen((open) => !open);
              }}
              type="button"
            >
              <Icon.Sparkle size={13} />
              Refine
            </button>
            <button className="su-pill-btn" onClick={openGuide} type="button">
              <Icon.Book size={13} />
              Prompt guide
            </button>
          </span>
        </div>
        <textarea
          className="su-textarea su-textarea--short"
          id="su-video-prompt"
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Describe the shot…"
          value={prompt}
        />
        {refineOpen ? (
          <RefinePanel
            blurb="Expand this into a detailed motion description."
            busy={refine.busy}
            error={refine.error}
            modelMissing={refine.modelMissing}
            onDownloadModel={refine.downloadModel}
            onRun={runRefine}
          />
        ) : null}
      </div>

      <div className="su-tiles su-tiles--single">
        <ReferenceTile
          asset={sourceAsset}
          assets={recentImageAssets}
          hint={mode === "image_to_video" ? "Required for Image → Video" : "Optional start frame"}
          label="Reference image"
          onChange={setSourceAssetId}
          required={mode === "image_to_video"}
        />
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
            value={selectedModel?.name ?? selectedModel?.id ?? "No video model installed"}
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
        <Chips
          label="Duration"
          onChange={setDuration}
          options={durations.map((value) => ({ value, label: `${value}s` }))}
          value={duration}
        />
        <SimpleLoraField
          atLimit={lora.atLimit}
          availableLoras={lora.availableLoras}
          effectiveLoraWeight={lora.effectiveLoraWeight}
          hint={WAN_MOE_PAIRED_LORA_MODEL_IDS.has(selectedModel?.id) ? MOE_LORA_HINT : null}
          onAddKeyword={(keyword) => setPrompt((current) => promptWithKeyword(current, keyword))}
          onOpenPicker={openLoraPicker}
          onRemove={lora.removeLora}
          onWeightChange={lora.setLoraWeight}
          pendingImports={lora.pendingImports}
          selectedLoras={lora.selectedLoras}
          selectedModel={selectedModel}
        />
      </div>

      <StyleStrip onChange={setStyleId} value={styleId} />

      <button
        className={busy ? "su-generate busy" : "su-generate"}
        disabled={!canGenerate}
        onClick={generate}
        type="button"
      >
        {busy ? <span aria-hidden="true" className="su-spinner" /> : null}
        {busy ? "Rendering…" : "Generate video"}
      </button>
      {needsSource ? <p className="su-empty">Attach a start frame to render an Image → Video clip.</p> : null}

      {/* Live run strip: progress + Cancel + the outcome, right under Generate. */}
      <StudioRunStatus job={latestJob} />

      <StudioResults
        assets={assets}
        columns={1}
        job={latestJob}
        meta={
          selectedModel
            ? `${selectedModel.name ?? selectedModel.id} · ${duration}s · ${describeResolution(resolution).sub}`
            : null
        }
        type="video"
        wide
      />
    </div>
  );
}
