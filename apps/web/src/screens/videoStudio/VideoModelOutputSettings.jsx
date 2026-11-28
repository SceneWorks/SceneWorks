import { ltx25VideoModelId } from "./videoTierHelpers.js";
import React from "react";
import { FlashAttentionToggle } from "../../components/FlashAttentionToggle.jsx";
import { GpuSelect } from "../../components/StudioFields.jsx";
import { StudioUpdateBadge, StudioUpdateNotice, updateOptionLabel } from "../../components/StudioUpdateNotice.jsx";
import { SourcePrecisionField, TierPickerField } from "../../components/generationStudio.jsx";
import { tierLabel } from "../../quantTier.js";
import { formatDurationOption } from "../../videoModelLimits.js";
import { ModelAttribution } from "../../components/ModelAttribution.jsx";
import { InfoTip } from "../../components/InfoTip.jsx";

// Settings tab › Model & output: what loads and what comes out — the model, the output shape
// (resolution, duration, frame rate), the one quantization slot, and the hardware knobs that used
// to sit in Advanced (GPU, text encoder, flash attention, prompt enhancement).
export function VideoModelOutputSettings({
  selectedModel,
  setUserModel,
  model,
  modelsForMode,
  mode,
  macVideoModels,
  createModelDownloadJob,
  recipeModelNotice,
  setResolution,
  resolution,
  resolutionOptions,
  setDuration,
  setAutoDuration,
  duration,
  durationOptions,
  durationHint,
  fps,
  setFps,
  fpsOptions,
  showTierPicker,
  quantTier,
  handleExecutionTierChange,
  tierPickerItems,
  tierSwitching,
  tierHasMemoryRisk,
  showTorchQuantization,
  setQuantization,
  quantization,
  quantVariants,
  setRequestedGpu,
  requestedGpu,
  gpuOptions,
  activeBackend,
  flashAttn,
  setFlashAttn,
  supportsPromptEnhancement,
  enhancePrompt,
  setEnhancePrompt,
  supportsStockPromptEnhancement,
  supportsTextEncoderSelection,
  setTextEncoderModel,
  selectedTextEncoderModel,
  textEncoderOptions,
  selectedTextEncoderAvailable,
  defaultTextEncoderId,
}) {
  return (
    <div className="studio-grid video-settings-model">
      <label className="settings-field-model span-2">
        <span className="video-field-label">
          Model
          <StudioUpdateBadge item={selectedModel} />
        </span>
        <select
          onChange={(event) => setUserModel(event.target.value)}
          value={model}
        >
          {/* Models gated on the selected tab (sc-5716): show only models that serve the
              active mode, falling back to the full available list if none do (a reduced
              catalog) so the picker is never empty. */}
          {(modelsForMode(mode).length ? modelsForMode(mode) : macVideoModels).map((item) => (
            <option key={item.id} value={item.id}>
              {updateOptionLabel(item)}
            </option>
          ))}
        </select>
        {/* Licence-required attribution (sc-17227 §IV.2, landed on the generation surfaces
            by sc-17161). The Models card alone is one screen a user may never revisit after
            installing; this is where the model is used. Reads the manifest field — never a
            second hard-coded copy of a string a licence specifies. */}
        <ModelAttribution model={selectedModel} className="studio-model-attribution" />
        {recipeModelNotice ? (
          <span className="field-hint" role="status">
            This clip was made with “{recipeModelNotice}”, which isn’t installed. Its
            settings were restored — pick a model to re-run it.
          </span>
        ) : null}
      </label>
      <StudioUpdateNotice item={selectedModel} onUpdate={createModelDownloadJob} />
      <label className="settings-field-aspect">
        Resolution
        <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
          {resolutionOptions.map((value) => (
            <option key={value} value={value}>
              {value.replace("x", " × ")}
            </option>
          ))}
        </select>
      </label>
      <label className="settings-field-count">
        <span className="field-label">
          Duration
          {durationHint ? <InfoTip label="About Duration">{durationHint}</InfoTip> : null}
        </span>
        {/* A MENU, never a range: `limits.durations` is the model's exact renderable set,
            and MiniMax-H3 is the case that makes the distinction visible — its supported
            `17n + 5` lattice rungs are the ONLY lengths the checkpoint renders, so a
            slider would emit durations between valid frame counts. The label
            carries the frame count because that is what the lattice is counting. */}
        <select
          onChange={(event) => {
            setDuration(Number(event.target.value));
            // A manual selection is an explicit duration request. Disable duration-head
            // selection immediately so the visible value and the submitted payload agree.
            if (model === ltx25VideoModelId) setAutoDuration(false);
          }}
          value={duration}
        >
          {durationOptions.map((value) => (
            <option key={value} value={value}>
              {formatDurationOption(value, fps)}
            </option>
          ))}
        </select>
      </label>
      <label>
        Frames
        <select onChange={(event) => setFps(Number(event.target.value))} value={fps}>
          {fpsOptions.map((value) => (
            <option key={value} value={value}>
              {value} fps
            </option>
          ))}
        </select>
      </label>
      {/* ONE quantization slot. The three are mutually exclusive (useVideoExecutionTier): a fixed
          source precision, else the native (MLX / candle) Quant tier picker, else the torch GGUF
          variant list. Quant tier, not the abstract Fast/Balanced/Best segment (studio-cleanup
          sc-15374): the slot names the concrete thing that will load; Quality is a Sampling knob. */}
      <SourcePrecisionField model={selectedModel} className="studio-field video-source-precision" />
      {showTierPicker ? (
        <TierPickerField
          value={quantTier}
          onChange={handleExecutionTierChange}
          items={tierPickerItems}
          tierSwitching={tierSwitching}
          tierLabel={tierLabel}
          title="Switch which installed MLX quant tier generates. Higher precision uses more memory; switching a heavy tier reloads it before the next generation."
          warning={tierHasMemoryRisk ? (
            <span className="field-hint quant-tier-memory-note">
              Higher MLX video tiers may run out of memory on long or high-resolution clips.
              Your pick is honored.
            </span>
          ) : null}
        />
      ) : null}
      {showTorchQuantization ? (
        <label>
          Quantization
          <select onChange={(event) => setQuantization(event.target.value)} value={quantization}>
            <option value="auto">Auto (per-platform default)</option>
            {quantVariants.map(([id, variant]) => (
              <option key={id} value={id}>
                {variant?.label ?? id}
              </option>
            ))}
            <option value="none">Full precision (unquantized)</option>
          </select>
        </label>
      ) : null}
      <GpuSelect options={gpuOptions} value={requestedGpu} onChange={setRequestedGpu} />
      {supportsPromptEnhancement && supportsTextEncoderSelection ? (
        <label>
          <span className="field-label">
            Text encoder model
            {!selectedTextEncoderAvailable && selectedTextEncoderModel !== defaultTextEncoderId ? (
              <InfoTip label="About Text encoder model" tone="warn">
                The recorded encoder is not staged on this worker. Choose the shipped default or stage the
                alternate before rendering.
              </InfoTip>
            ) : (
              <InfoTip label="About Text encoder model">
                {textEncoderOptions.length > 1
                  ? "Only complete encoders already staged for this worker are listed."
                  : "The shipped encoder is the default. Complete operator-staged alternates appear here after Models is refreshed."}
              </InfoTip>
            )}
          </span>
          <select
            aria-label="Text encoder model"
            disabled={!enhancePrompt}
            onChange={(event) => setTextEncoderModel(event.target.value)}
            value={selectedTextEncoderModel}
          >
            {textEncoderOptions.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
            {!selectedTextEncoderAvailable &&
            selectedTextEncoderModel !== defaultTextEncoderId ? (
              <option disabled value={selectedTextEncoderModel}>
                Previously selected encoder (not staged)
              </option>
            ) : null}
          </select>
        </label>
      ) : null}
      <FlashAttentionToggle backend={activeBackend} checked={flashAttn} onChange={setFlashAttn} />
      {supportsPromptEnhancement ? (
        <div className="studio-field span-2 video-checkline-field">
          <label className="checkline">
            <input
              checked={enhancePrompt}
              onChange={(event) => setEnhancePrompt(event.target.checked)}
              type="checkbox"
            />
            Enhance prompt before generation
            <InfoTip label="About Enhance prompt">
              {supportsStockPromptEnhancement
                ? "Rewrites the prompt with LTX-2.5's separately downloaded stock Gemma-4 enhancer."
                : "Rewrites the prompt with the selected text encoder before the model encodes it."}
            </InfoTip>
          </label>
        </div>
      ) : null}
    </div>
  );
}
