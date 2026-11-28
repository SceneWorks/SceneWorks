import React from "react";
import { ModelAttribution } from "../../components/ModelAttribution.jsx";
import { StudioUpdateBadge, StudioUpdateNotice, updateOptionLabel } from "../../components/StudioUpdateNotice.jsx";
import { SourcePrecisionField, TierPickerField } from "../../components/generationStudio.jsx";
import { GpuSelect } from "../../components/StudioFields.jsx";
import { FlashAttentionToggle } from "../../components/FlashAttentionToggle.jsx";
import { LossyAttentionControl } from "../../components/LossyAttentionControl.jsx";
import { tierLabel } from "../../quantTier.js";
import { mossAccelerationAvailable, mossAccelerationMode } from "../../mossImage.js";
import { AspectResolutionPicker } from "./AspectResolutionPicker.jsx";
import { InfoTip } from "../../components/InfoTip.jsx";

// Settings tab › Model & output: the model, its size and count, which build loads (source
// precision / quant tier), and the hardware/precision knobs that used to sit in Advanced (GPU,
// text encoder, flash attention, enhance prompt, full precision). Every control keeps the
// availability condition it had in the settings bar or Advanced.
export function ImageModelSettings({
  selectedModel,
  setModel,
  model,
  pickerModels,
  modeLabel,
  setActiveView,
  createModelDownloadJob,
  mossEdit,
  mossDmd,
  setMossField,
  mossComponents,
  mossDownload,
  mossDownloadProgress,
  downloadMossAcceleration,
  handleResolutionChange,
  resolution,
  sizeEnvelope,
  patchGridNotice,
  nativeResolutionNotice,
  setCount,
  count,
  showTierPicker,
  quantTier,
  handleTierChange,
  tierPickerItems,
  tierSwitching,
  tierSuggested,
  confirmTier,
  tierBelowFloor,
  qualityFloor,
  tierExplicit,
  // Hardware / precision (formerly Advanced).
  moss,
  setRequestedGpu,
  requestedGpu,
  gpuOptions,
  supportsTextEncoderSelection,
  setTextEncoderModel,
  selectedTextEncoderModel,
  textEncoderOptions,
  selectedTextEncoderAvailable,
  defaultTextEncoderId,
  activeBackend,
  flashAttn,
  setFlashAttn,
  lossyAttention,
  setLossyAttention,
  lossyAttentionError,
  promptEnhance,
  enhancePrompt,
  setEnhancePrompt,
  precisionToggle,
  bf16Precision,
  setRecipeTierRequest,
  setBf16Precision,
}) {
  return (
    <div className="studio-grid image-model-settings">
      <div className="span-full image-model-update">
        <StudioUpdateNotice item={selectedModel} onUpdate={createModelDownloadJob} />
      </div>
      <label className="settings-field settings-field-model span-2">
        <span className="studio-field-label">
          Model
          <StudioUpdateBadge item={selectedModel} />
        </span>
        <select onChange={(event) => setModel(event.target.value)} value={model}>
          {pickerModels.map((item) => (
            <option key={item.id} value={item.id}>
              {updateOptionLabel(item)}
            </option>
          ))}
        </select>
        {!pickerModels.length ? (
          <span className="field-hint" role="status">
            No installed model supports {modeLabel} generation.{" "}
            <button onClick={() => setActiveView("Models")} type="button">
              Browse models
            </button>
          </span>
        ) : null}
        {/* Licence-required attribution (MiniMax-H3-Image §IV.2, SW-7100): the generation
            surface, from the manifest field, as Video Studio does. */}
        <ModelAttribution model={selectedModel} className="studio-model-attribution" />
      </label>
      <AspectResolutionPicker
        envelope={sizeEnvelope}
        onChange={handleResolutionChange}
        resolution={resolution}
        // sc-25681: informational only — the size is never restricted.
        info={patchGridNotice || nativeResolutionNotice ? (
          <>
            {patchGridNotice ? <span role="note">{patchGridNotice}</span> : null}
            {nativeResolutionNotice ? <span role="note">{nativeResolutionNotice}</span> : null}
          </>
        ) : null}
      />
      <label className="settings-field settings-field-count">
        Variations
        <input min="1" max="8" onChange={(event) => setCount(Number(event.target.value))} type="number" value={count} />
      </label>
      {/* Quant tier belongs with the everyday model knobs, not buried behind a disclosure
          (studio-cleanup sc-15374): it names the build that will actually load. */}
      <SourcePrecisionField model={selectedModel} className="settings-field settings-field-tier studio-field" />
      {showTierPicker ? (
        <TierPickerField
          className="settings-field settings-field-tier"
          value={quantTier}
          onChange={handleTierChange}
          items={tierPickerItems}
          tierSwitching={tierSwitching}
          tierLabel={tierLabel}
          suggested={tierSuggested}
          onConfirm={confirmTier}
          title="Switch which installed quant tier generates, for A/B comparison. Higher precision = larger memory footprint; switching a heavy tier reloads it before the next generation. Tiers you haven't downloaded are shown but disabled — install them from the Models page to enable."
          warning={tierBelowFloor ? (
            <span className="field-hint quant-tier-floor-note">
              {tierLabel(quantTier)} is below the {tierLabel(qualityFloor)} recommended for{" "}
              {selectedModel?.name ?? "this model"} — it can look washed or lose fine detail
              here (quantization error is amplified under CFG).
              {tierExplicit ? " Your pick is honored." : ""}
            </span>
          ) : null}
        />
      ) : null}
      {mossAccelerationAvailable(selectedModel) && !mossEdit ? <label className="checkline">
        <input type="checkbox" checked={Boolean(mossDmd)} onChange={(event) => setMossField("acceleration", event.target.checked ? mossAccelerationMode : "off")} />
        8-step acceleration
      </label> : null}
      {mossDmd && !mossComponents.ready ? <div className="span-full moss-acceleration-download" role="status">
        <p>8-step package: {mossDownload.busy ? "queueing" : mossComponents.state}. {mossDownloadProgress || "Install the selected acceleration components before generating."}</p>
        <button type="button" disabled={mossDownload.busy} onClick={downloadMossAcceleration}>Download 8-step components</button>
      </div> : null}
      {mossDownload.error ? <p role="alert">{mossDownload.error}</p> : null}
      <GpuSelect options={gpuOptions} value={requestedGpu} onChange={setRequestedGpu} />
      {supportsTextEncoderSelection ? (
        <>
          <label className="span-2">
            <span className="field-label">
              Text encoder model
              <InfoTip
                label="About Text encoder model"
                tone={!selectedTextEncoderAvailable && selectedTextEncoderModel !== defaultTextEncoderId ? "warn" : undefined}
              >
                {!selectedTextEncoderAvailable &&
                selectedTextEncoderModel !== defaultTextEncoderId
                  ? "The recorded encoder is no longer staged. Choose the model default or restore that encoder before generating."
                  : textEncoderOptions.length > 1
                    ? "Only compatible installed encoders verified by this model's engine contract are listed."
                    : "The model encoder is the default. Compatible built-in, imported, and external choices appear after Models is refreshed."}
              </InfoTip>
            </span>
            <select
              aria-label="Text encoder model"
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
        </>
      ) : null}
      {!moss ? (
        <FlashAttentionToggle backend={activeBackend} checked={flashAttn} onChange={setFlashAttn} />
      ) : null}
      {setLossyAttention ? (
        <LossyAttentionControl
          backend={activeBackend}
          error={lossyAttentionError}
          model={selectedModel}
          onChange={setLossyAttention}
          value={lossyAttention}
        />
      ) : null}
      {promptEnhance ? (
        <label className="checkline prompt-enhance-toggle">
          <input
            checked={enhancePrompt}
            onChange={(event) => setEnhancePrompt(event.target.checked)}
            type="checkbox"
          />
          Enhance prompt
          <InfoTip label="About Enhance prompt">
            Have the model's native prompt enhancer rewrite your prompt before generating. FLUX 3 uses its separately installed optional upsampler. The saved recipe records the original and effective prompts, including any safe fallback. Distinct from Refine; off by default.
          </InfoTip>
        </label>
      ) : null}
      {precisionToggle && !showTierPicker ? (
        <label className="checkline boogu-precision-toggle">
          <input
            checked={bf16Precision}
            onChange={(event) => { setRecipeTierRequest(null); setBf16Precision(event.target.checked); }}
            type="checkbox"
          />
          Full precision (bf16)
          <InfoTip label="About Full precision">
            Use the full-precision bf16 build instead of the default Q8. Higher fidelity, but a much larger download (~38 GB per variant, fetched on demand) that needs a larger Mac (≈96 GB unified memory). Off = the Q8 default (~23 GB, 64 GB-class Mac).
          </InfoTip>
        </label>
      ) : null}
    </div>
  );
}
