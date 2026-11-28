import React from "react";
import { SamplerSchedulerFields, SamplingSeedField, SamplingStepsField } from "../../components/SamplerSchedulerFields.jsx";
import { StudioSettingsGroup } from "../../components/StudioSettingsTabs.jsx";
import { MultiPhaseEditor } from "../../components/MultiPhaseEditor.jsx";
import { MossOrdinaryControls } from "./MossOrdinaryControls.jsx";
import { minStepsForModel } from "../../videoModelLimits.js";
import {
  CAPTION_OVERFLOW_LABELS,
  H3_IMAGE_LCM_FIELDS,
  H3_IMAGE_SAMPLER_LABELS,
  LOW_RES_START_MAX_SCALE,
  LOW_RES_START_MIN_SCALE,
} from "./samplingControls.js";
import { GUIDANCE_METHOD_LABELS, fixedStepsForModel, guidanceDefaultFromModel, maxStepsForModel, stepsDefaultFromModel } from "../../samplerOptions.js";

// Settings tab › Sampling: the denoise knobs every model shares (cleared values → model
// default), then "For this model" — the blocks only one model family declares (MOSS-Image, PAG,
// MiniMax-H3-Image's sampler, Krea 2's multi-phase denoise). Each keeps the availability
// condition it had in Advanced; the group heading drops out when none apply.
export function ImageSamplingSettings({
  mossEdit,
  mossImageAdvanced,
  setSeed,
  seed,
  dimensionConstraints,
  setReferenceDimensionIntent,
  setWidthOverride,
  activeDimensionIntent,
  dropdownWidth,
  widthOverride,
  setHeightOverride,
  dropdownHeight,
  heightOverride,
  moss,
  mossDmd,
  mossImage,
  setMossField,
  setSampler,
  sampler,
  samplerOptions,
  setScheduler,
  scheduler,
  schedulerOptions,
  showSchedulerShift,
  setSchedulerShift,
  effectiveSchedulerShift,
  selectedModel,
  setMossDmdSteps,
  setStepsOverride,
  mossDmdSteps,
  stepsOverride,
  supportsGuidance,
  setGuidanceOverride,
  guidanceOverride,
  supportsCfgInterval,
  cfgIntervalDisabled,
  setCfgInterval,
  cfgInterval,
  cfgIntervalError,
  captionOverflowChoices,
  setCaptionOverflow,
  outgoingCaptionOverflow,
  showH3ImageSampler,
  h3ImageSampler,
  setH3ImageSampler,
  h3ImageSamplerError,
  pagCapability,
  pag,
  setPag,
  pagError,
  showLowResStart,
  lowResStart,
  setLowResStart,
  lowResStartResolved,
  supportsTextStyle,
  textStyleGainConfig,
  setTextStyleGain,
  textStyleGain,
  showGuidanceMethodPicker,
  setGuidanceMethod,
  guidanceMethod,
  guidanceMethodOptions,
  supportsNegativePrompt,
  negativePromptCfgOff,
  setNegativePrompt,
  negativePrompt,
  // Krea 2 multi-phase denoise (epic 13879 S5, sc-13885).
  multiPhaseAvailable,
  multiPhaseEnabled,
  applyTurboFinishPreset,
  setMultiPhasePhases,
  setMultiPhaseEnabled,
  multiPhasePhases,
  selectedLoras,
}) {
  const modelSpecific = Boolean(moss || showH3ImageSampler || pagCapability || multiPhaseAvailable);
  return (
    <div className="studio-grid image-sampling-settings">
      <SamplingSeedField
        value={seed}
        onChange={(value) => setSeed(value)}
        inputProps={{
          disabled: mossEdit && Boolean(mossImageAdvanced.noise),
          placeholder: mossEdit && mossImageAdvanced.noise ? "Supplied noise (seed null)" : "Random",
          type: "number"
        }}
      />
      {/* SW-7100: a model with exactly one legal step count (hardMinSteps === hardMaxSteps)
          has no Steps control; the job leaves steps to the model default. */}
      {mossDmd || fixedStepsForModel(selectedModel) == null ? <SamplingStepsField
        value={mossDmd ? mossDmdSteps : stepsOverride}
        onChange={(value) => mossDmd ? setMossDmdSteps(value) : setStepsOverride(value)}
        inputProps={{
          min: String(mossDmd ? 1 : minStepsForModel(selectedModel)),
          max: mossDmd ? "8" : moss ? undefined : String(maxStepsForModel(selectedModel)),
          placeholder: String(mossDmd ? 8 : stepsDefaultFromModel(selectedModel) ?? ""),
          type: "number"
        }}
      /> : null}
      {supportsGuidance ? (
        <label>
          Guidance
          <input
            disabled={mossDmd}
            onChange={(event) => setGuidanceOverride(event.target.value)}
            placeholder={(() => {
              const value = guidanceDefaultFromModel(selectedModel);
              return value == null ? "" : String(value);
            })()}
            step="any"
            type="number"
            value={guidanceOverride}
          />
        </label>
      ) : null}
      {showSchedulerShift ? (
        <label>
          Schedule shift
          <input
            disabled={mossDmd || selectedModel?.image?.supportsSchedulerShift === false}
            title={selectedModel?.image?.supportsSchedulerShift === false ? "This model has no native flow schedule shift." : undefined}
            min="0"
            onChange={(event) => setSchedulerShift(event.target.value)}
            step="any"
            placeholder={selectedModel?.defaults?.schedulerShift == null ? "Model default" : String(selectedModel.defaults.schedulerShift)}
            type="number"
            value={effectiveSchedulerShift}
          />
        </label>
      ) : null}
      <SamplerSchedulerFields
        sampler={sampler}
        scheduler={scheduler}
        samplerOptions={samplerOptions}
        schedulerOptions={schedulerOptions}
        onSamplerChange={setSampler}
        onSchedulerChange={setScheduler}
        disabled={mossDmd}
      />
      <label>
        Width override
        <input
          min={dimensionConstraints.min}
          max={dimensionConstraints.max}
          step={dimensionConstraints.step}
          onChange={(event) => {
            setReferenceDimensionIntent(null);
            setWidthOverride(event.target.value);
          }}
          placeholder={activeDimensionIntent?.width === null ? "Auto (reference canvas)" : String(dropdownWidth)}
          type="number"
          value={widthOverride}
        />
      </label>
      <label>
        Height override
        <input
          min={dimensionConstraints.min}
          max={dimensionConstraints.max}
          step={dimensionConstraints.step}
          onChange={(event) => {
            setReferenceDimensionIntent(null);
            setHeightOverride(event.target.value);
          }}
          placeholder={activeDimensionIntent?.height === null ? "Auto (reference canvas)" : String(dropdownHeight)}
          type="number"
          value={heightOverride}
        />
      </label>
      {/* sc-25681: the CFG interval — guidance runs only at model times inside
          [start, end] (1 = noise, 0 = clean). Disabled, and not sent, while guidance is
          1.0: the engine refuses an interval there because it would do nothing. */}
      {supportsCfgInterval ? (
        <label>
          Guidance interval
          <input
            disabled={cfgIntervalDisabled}
            onChange={(event) => setCfgInterval(event.target.value)}
            placeholder="0, 1"
            value={cfgInterval}
          />
          <span className="field-hint" role={cfgIntervalError ? "alert" : undefined}>
            {cfgIntervalDisabled
              ? "Needs guidance above 1."
              : cfgIntervalError || "start, end: guidance runs only between (1 = noise)."}
          </span>
        </label>
      ) : null}
      {showGuidanceMethodPicker ? (
        <label>
          Guidance method
          <select
            onChange={(event) => setGuidanceMethod(event.target.value)}
            value={guidanceMethod}
          >
            {guidanceMethodOptions.map((key) => (
              <option key={key} value={key}>
                {GUIDANCE_METHOD_LABELS[key] ?? key}
              </option>
            ))}
          </select>
          {guidanceMethod === "cfg_pp" ? (
            <span className="field-hint">
              CFG++ reparameterizes guidance — use a low CFG (~1.5–2.5); high
              values over-saturate.
            </span>
          ) : null}
        </label>
      ) : null}
      {captionOverflowChoices.length ? (
        <label>
          Long prompts
          <select
            aria-label="Caption overflow"
            onChange={(event) => setCaptionOverflow(event.target.value)}
            value={outgoingCaptionOverflow}
          >
            {["default", ...captionOverflowChoices].map((policy) => (
              <option key={policy} value={policy}>
                {CAPTION_OVERFLOW_LABELS[policy] ?? policy}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {supportsTextStyle ? (
        <label className="text-style-gain">
          {textStyleGainConfig?.label ?? "Text style"}
          <input
            max={textStyleGainConfig?.max ?? 1.75}
            min={textStyleGainConfig?.min ?? 0.25}
            onChange={(event) => setTextStyleGain(Number(event.target.value))}
            step={textStyleGainConfig?.step ?? 0.05}
            type="range"
            value={textStyleGain}
          />
          <span>{Number(textStyleGain).toFixed(2)}</span>
        </label>
      ) : null}
      {showLowResStart ? (
        <div className="low-res-start-controls span-full">
          <label
            className="checkline low-res-start-toggle"
            title="Denoise the first K steps on a reduced latent grid, then resize and re-noise onto the full target resolution for steps K..M."
          >
            <input
              aria-label="Low-res start"
              checked={Boolean(lowResStart?.enabled)}
              onChange={(event) =>
                setLowResStart((current) => ({ ...current, enabled: event.target.checked }))
              }
              type="checkbox"
            />
            Low-res start
          </label>
          {lowResStart?.enabled ? (
            <div className="low-res-start-fields">
              <label>
                Low-res steps (K)
                <input
                  aria-label="Low-res steps (K)"
                  max={Math.max(1, (lowResStartResolved?.executedSteps ?? 2) - 1)}
                  min={1}
                  onChange={(event) =>
                    setLowResStart((current) => ({
                      ...current,
                      k: event.target.value === "" ? "" : Number(event.target.value),
                    }))
                  }
                  step={1}
                  type="number"
                  value={
                    lowResStart?.k === "" || lowResStart?.k == null
                      ? (lowResStartResolved?.effectiveK ?? 1)
                      : lowResStart.k
                  }
                />
              </label>
              <label>
                {`Phase split (${lowResStartResolved?.effectiveK ?? 1} / ${lowResStartResolved?.executedSteps ?? 0})`}
                <input
                  aria-label="Low-res K / M slider"
                  disabled={(lowResStartResolved?.executedSteps ?? 0) < 2}
                  max={Math.max(1, (lowResStartResolved?.executedSteps ?? 2) - 1)}
                  min={1}
                  onChange={(event) =>
                    setLowResStart((current) => ({
                      ...current,
                      k: Number(event.target.value),
                    }))
                  }
                  step={1}
                  type="range"
                  value={Math.min(
                    Math.max(1, Number(lowResStartResolved?.effectiveK ?? 1)),
                    Math.max(1, (lowResStartResolved?.executedSteps ?? 2) - 1),
                  )}
                />
                <span>{`${lowResStartResolved?.effectiveK ?? 1} / ${lowResStartResolved?.executedSteps ?? 0}`}</span>
              </label>
              <label>
                Low-res scale
                <input
                  aria-label="Low-res scale"
                  max={LOW_RES_START_MAX_SCALE}
                  min={LOW_RES_START_MIN_SCALE}
                  onChange={(event) =>
                    setLowResStart((current) => ({
                      ...current,
                      scale: event.target.value === "" ? "" : Number(event.target.value),
                    }))
                  }
                  step={0.05}
                  type="number"
                  value={lowResStart?.scale ?? 0.5}
                />
              </label>
              <span
                className="field-hint low-res-start-dimensions"
                data-testid="low-res-start-dimensions"
                role={lowResStartResolved?.error ? "alert" : "status"}
              >
                {lowResStartResolved?.error ||
                  `Resolved grid: ${lowResStartResolved?.dimensions?.lowWidth ?? 0}×${lowResStartResolved?.dimensions?.lowHeight ?? 0} → ${lowResStartResolved?.dimensions?.fullWidth ?? 0}×${lowResStartResolved?.dimensions?.fullHeight ?? 0} (K=${lowResStartResolved?.effectiveK ?? 1} / M=${lowResStartResolved?.executedSteps ?? 0})`}
              </span>
            </div>
          ) : null}
        </div>
      ) : null}
      {supportsNegativePrompt ? (
        <label className="prompt-field span-full">
          Negative prompt
          <textarea
            disabled={negativePromptCfgOff || mossDmd}
            onChange={(event) => setNegativePrompt(event.target.value)}
            value={negativePrompt}
          />
          {negativePromptCfgOff ? (
            <span className="field-hint">
              The negative prompt applies when CFG is enabled for this model.
            </span>
          ) : null}
        </label>
      ) : null}
      {modelSpecific ? (
        <StudioSettingsGroup label="For this model">
          {moss ? (
            <>
              {mossDmd ? <>
                <label>Acceleration precision<select value={mossImage.acceleration} onChange={(event) => setMossField("acceleration", event.target.value)}>
                  <option value="unmerged_bf16">Standard precision · less memory</option>
                  <option value="full_fp32">Full precision · more memory</option>
                </select></label>
              </> : null}
              <label>Context limit<input type="number" min="1" max="4096" value={mossImage.max_sequence_length ?? 4096} onChange={(event) => setMossField("max_sequence_length", Number(event.target.value))} /></label>
              {mossEdit ? <label>Training shift<input type="number" min="0" step="any" value={mossImage.training_shift ?? 1} onChange={(event) => setMossField("training_shift", Number(event.target.value))} /></label> : null}
              {!mossEdit ? <MossOrdinaryControls options={mossImage} disabled={Boolean(mossDmd)} onChange={setMossField} /> : null}
            </>
          ) : null}
          {/* SW-7106: MiniMax-H3-Image's sampler. Euler is the released deterministic step;
              LCM re-noises each step's clean estimate with a noise scale ramping from start
              to end, optionally clipped. Shown only for a model that declares it. */}
          {showH3ImageSampler ? (
            <>
              <label>
                Sampler
                <select
                  aria-label="MiniMax-H3-Image sampler"
                  onChange={(event) => setH3ImageSampler((current) => ({ ...current, sampler: event.target.value }))}
                  value={h3ImageSampler.sampler}
                >
                  {Object.entries(H3_IMAGE_SAMPLER_LABELS).map(([value, label]) => (
                    <option key={value} value={value}>{label}</option>
                  ))}
                </select>
              </label>
              {h3ImageSampler.sampler === "lcm" ? (
                <>
                  {H3_IMAGE_LCM_FIELDS.map(({ key, label, min, max, step }) => (
                    <label key={key}>
                      {label}
                      <input
                        max={max}
                        min={min}
                        onChange={(event) => setH3ImageSampler((current) => ({ ...current, [key]: event.target.value }))}
                        step={step}
                        type="number"
                        value={h3ImageSampler[key]}
                      />
                    </label>
                  ))}
                  {h3ImageSamplerError ? (
                    <span className="field-hint span-full" role="alert">{h3ImageSamplerError}</span>
                  ) : null}
                </>
              ) : null}
            </>
          ) : null}
          {pagCapability ? (
            <div className="pag-controls span-full">
              <label>
                PAG scale
                <input
                  aria-label="PAG scale"
                  onChange={(event) => setPag((current) => ({ ...current, scale: event.target.value }))}
                  placeholder="0 (Off)"
                  step="any"
                  type="number"
                  value={pag.scale}
                />
                <span className="field-hint" role={pagError ? "alert" : undefined}>
                  {pagError ||
                    `Recommended: ${pagCapability.recommendedScale} · default blocks [${pagCapability.defaultLayers.join(", ")}] · window [${pagCapability.defaultStart}, ${pagCapability.defaultStop})`}
                </span>
              </label>
              <details className="pag-advanced">
                <summary>Advanced PAG overrides</summary>
                <div className="pag-advanced-grid">
                  <label>
                    PAG blocks
                    <input
                      aria-label="PAG blocks"
                      onChange={(event) => setPag((current) => ({ ...current, layers: event.target.value }))}
                      placeholder={`Default (${pagCapability.defaultLayers.join(", ")})`}
                      type="text"
                      value={pag.layers}
                    />
                    <span className="field-hint">
                      {`0..${pagCapability.numBlocks - 1} (${pagCapability.blockDomain})`}
                    </span>
                  </label>
                  <label>
                    PAG adaptive scale
                    <input
                      aria-label="PAG adaptive scale"
                      onChange={(event) => setPag((current) => ({ ...current, adaptiveScale: event.target.value }))}
                      placeholder="0"
                      step="any"
                      type="number"
                      value={pag.adaptiveScale}
                    />
                  </label>
                  <label>
                    PAG window start
                    <input
                      aria-label="PAG window start"
                      onChange={(event) => setPag((current) => ({ ...current, start: event.target.value }))}
                      placeholder={String(pagCapability.defaultStart)}
                      step="any"
                      type="number"
                      value={pag.start}
                    />
                  </label>
                  <label>
                    PAG window stop
                    <input
                      aria-label="PAG window stop"
                      onChange={(event) => setPag((current) => ({ ...current, stop: event.target.value }))}
                      placeholder={String(pagCapability.defaultStop)}
                      step="any"
                      type="number"
                      value={pag.stop}
                    />
                  </label>
                </div>
              </details>
            </div>
          ) : null}
          {/* Krea 2 multi-phase denoise (epic 13879 S5, sc-13885). EXPERIMENTAL, shown only for
              the model that advertises the acceleration LoRA compat (krea_2_raw) in text-to-image
              mode — matching the worker gate. The phase editor references the LoRAs picked in the
              LoRAs tab by index. */}
          {multiPhaseAvailable ? (
            <div className="multiphase-section span-full">
              <span className="studio-field-label">
                Multi-phase denoise <span className="badge">Experimental</span>
              </span>
              <span className="field-hint">Split the Raw denoise into phases.</span>
              <MultiPhaseEditor
                enabled={multiPhaseEnabled}
                onApplyTurboFinish={applyTurboFinishPreset}
                onChange={setMultiPhasePhases}
                onToggleEnabled={setMultiPhaseEnabled}
                phases={multiPhasePhases}
                selectedLoras={selectedLoras}
              />
            </div>
          ) : null}
        </StudioSettingsGroup>
      ) : null}
    </div>
  );
}
