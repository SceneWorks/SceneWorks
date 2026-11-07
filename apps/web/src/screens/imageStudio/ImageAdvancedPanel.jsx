import { GpuSelect } from "../../components/StudioFields.jsx";
import { SamplerSchedulerFields, SamplingSeedField, SamplingStepsField } from "../../components/SamplerSchedulerFields.jsx";
import { MossOrdinaryControls } from "./MossOrdinaryControls.jsx";
import React from "react";
import { AdvancedSection } from "../../components/AdvancedSection.jsx";
import { minStepsForModel } from "../../videoModelLimits.js";
import { showTransparencyToggle, transparencyPromptSuggestion, TRANSPARENCY_PROMPT_HINT } from "../../qwenAlpha.js";
import { CAPTION_OVERFLOW_LABELS } from "./samplingControls.js";
import { upscaleEngineHasSoftness, upscaleEngineIsRestoration, upscaleFactorsForEngine } from "../../upscaleEngines.js";
import { irisRestoreControls } from "../../irisRestore.js";
import { IrisRestoreControls } from "../../components/IrisRestoreControls.jsx";
import { GUIDANCE_METHOD_LABELS, guidanceDefaultFromModel, maxStepsForModel, stepsDefaultFromModel } from "../../samplerOptions.js";

export function ImageAdvancedPanel({
  toggleAdvanced,
  advancedOpen,
  setRequestedGpu,
  requestedGpu,
  gpuOptions,
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
  supportsTextStyle,
  textStyleGainConfig,
  setTextStyleGain,
  textStyleGain,
  supportsTextEncoderSelection,
  setTextEncoderModel,
  selectedTextEncoderModel,
  textEncoderOptions,
  selectedTextEncoderAvailable,
  defaultTextEncoderId,
  showGuidanceMethodPicker,
  setGuidanceMethod,
  guidanceMethod,
  guidanceMethodOptions,
  flashAttn,
  setFlashAttn,
  promptEnhance,
  enhancePrompt,
  setEnhancePrompt,
  precisionToggle,
  showTierPicker,
  bf16Precision,
  setRecipeTierRequest,
  setBf16Precision,
  transparentBackground,
  setTransparentBackground,
  prompt,
  setPromptFromUser,
  showPidToggle,
  usePid,
  upscaleEnabled,
  hiresFixEnabled,
  setUsePid,
  setDecoder,
  setPidTarget,
  pidTarget,
  pidDecodeNotice,
  showDecoderPicker,
  decoder,
  decoderOptions,
  setUpscaleEnabled,
  setHiresFixEnabled,
  upscaleEngine,
  setUpscaleFactor,
  upscaleFactor,
  handleUpscaleEngineChange,
  availableUpscaleEngines,
  setUpscaleRestore,
  setRestorePlanStatus,
  restorePlanBody,
  token,
  upscaleRestore,
  setUpscaleSoftness,
  upscaleSoftness,
  hiresFixDisabledReason,
  hiresFixDisabled,
  setHiresFixSteps,
  hiresFixSteps,
  setHiresFixDenoisingStrength,
  hiresFixDenoisingStrength,
  setHiresFixUpscaleBy,
  hiresFixUpscaleBy,
  hiresFixTargetInvalid,
  hiresFixTargetWidth,
  hiresFixTargetHeight,
  setHiresFixCfgScale,
  hiresFixCfgScale,
  supportsNegativePrompt,
  negativePromptCfgOff,
  setNegativePrompt,
  negativePrompt
}) {
  return (
<AdvancedSection
            hint="cleared values → model default"
            onToggle={toggleAdvanced}
            open={advancedOpen}
          >
            <div className="advanced-panel">
              <GpuSelect options={gpuOptions} value={requestedGpu} onChange={setRequestedGpu} />
              <SamplingSeedField
                value={seed}
                onChange={(value) => setSeed(value)}
                inputProps={{
                  disabled: mossEdit && Boolean(mossImageAdvanced.noise),
                  placeholder: mossEdit && mossImageAdvanced.noise ? "Supplied noise (seed null)" : "Random",
                  type: "number"
                }}
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
              <SamplerSchedulerFields
                sampler={sampler}
                scheduler={scheduler}
                samplerOptions={samplerOptions}
                schedulerOptions={schedulerOptions}
                onSamplerChange={setSampler}
                onSchedulerChange={setScheduler}
                disabled={mossDmd}
              />
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
              <SamplingStepsField
                value={mossDmd ? mossDmdSteps : stepsOverride}
                onChange={(value) => mossDmd ? setMossDmdSteps(value) : setStepsOverride(value)}
                inputProps={{
                  min: String(mossDmd ? 1 : minStepsForModel(selectedModel)),
                  max: mossDmd ? "8" : moss ? undefined : String(maxStepsForModel(selectedModel)),
                  placeholder: String(mossDmd ? 8 : stepsDefaultFromModel(selectedModel) ?? ""),
                  type: "number"
                }}
              />
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
              {supportsTextEncoderSelection ? (
                <>
                  <label>
                    Text encoder model
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
                  <p className="helper-copy">
                    {!selectedTextEncoderAvailable &&
                    selectedTextEncoderModel !== defaultTextEncoderId
                      ? "The recorded encoder is no longer staged. Choose the model default or restore that encoder before generating."
                      : textEncoderOptions.length > 1
                        ? "Only compatible installed encoders verified by this model's engine contract are listed."
                        : "The model encoder is the default. Compatible built-in, imported, and external choices appear after Models is refreshed."}
                  </p>
                </>
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
              {!moss ? <>
              <label
                className="checkline flash-attn-toggle"
                title="Fused flash-attention on the candle (Windows/CUDA) SDXL backend — faster and lower VRAM. Ignored on other backends."
              >
                <input
                  checked={flashAttn}
                  onChange={(event) => setFlashAttn(event.target.checked)}
                  type="checkbox"
                />
                Flash attention
              </label>
              </> : null}
              {promptEnhance ? (
                <label
                  className="checkline prompt-enhance-toggle"
                  title="Have the model's native prompt enhancer rewrite your prompt before generating. FLUX 3 uses its separately installed optional upsampler. The saved recipe records the original and effective prompts, including any safe fallback. Distinct from Refine; off by default."
                >
                  <input
                    checked={enhancePrompt}
                    onChange={(event) => setEnhancePrompt(event.target.checked)}
                    type="checkbox"
                  />
                  Enhance prompt
                </label>
              ) : null}
              {precisionToggle && !showTierPicker ? (
                <label
                  className="checkline boogu-precision-toggle"
                  title="Use the full-precision bf16 build instead of the default Q8. Higher fidelity, but a much larger download (~38 GB per variant, fetched on demand) that needs a larger Mac (≈96 GB unified memory). Off = the Q8 default (~23 GB, 64 GB-class Mac)."
                >
                  <input
                    checked={bf16Precision}
                    onChange={(event) => { setRecipeTierRequest(null); setBf16Precision(event.target.checked); }}
                    type="checkbox"
                  />
                  Full precision (bf16)
                </label>
              ) : null}
              {/* sc-24113 — native transparency. Rendered only when the selected model advertises
                  four-channel decode (`supportsAlphaOutput`), because a cut-out toggle on a model
                  that can only emit opaque RGB is a control that always fails. Hidden rather than
                  disabled: a control that can never apply here is noise, not information. */}
              {showTransparencyToggle(selectedModel) ? (
                <label
                  className="checkline transparency-toggle"
                  title="Keep the model's alpha channel instead of compositing the render onto white, and save the result as an RGBA PNG — so cut-outs, logos and stickers keep real transparency through editing, layering and export. The model has no transparency mode: ASK for a transparent background in the prompt, and use this to keep it."
                >
                  <input
                    checked={transparentBackground}
                    onChange={(event) => setTransparentBackground(event.target.checked)}
                    type="checkbox"
                  />
                  Transparent background (RGBA)
                </label>
              ) : null}
              {/* The other half of the control, and the non-obvious one. There is no transparency
                  MODE in this model: the toggle only keeps the alpha channel, and whether that
                  channel holds a cut-out is decided by the PROMPT. Without this the toggle hands
                  back a fully opaque RGBA PNG and reads as broken.

                  Offered, never applied: the wording goes into the prompt the user can see and
                  edit, on a button they press. `transparencyPromptSuggestion` returns null once the
                  prompt already says it, so re-running never accretes the sentence. */}
              {transparencyPromptSuggestion(prompt, transparentBackground) ? (
                <p className="transparency-prompt-hint">
                  This model has no transparency mode — ask for it in the prompt.{" "}
                  <button
                    className="hero-link"
                    onClick={() =>
                      setPromptFromUser(
                        transparencyPromptSuggestion(prompt, transparentBackground),
                      )
                    }
                    type="button"
                  >
                    Add “{TRANSPARENCY_PROMPT_HINT}”
                  </button>
                </p>
              ) : null}
              {showPidToggle ? (
                <>
                  <label
                    className="checkline pid-decoder-toggle"
                    title="Decode this generation through NVIDIA's PiD pixel-diffusion decoder instead of the model's VAE: it decodes and super-resolves in one pass to 2K or 4K (pick the tier at right — sharper detail, but slower and more memory). Non-commercial use only — PiD output is licensed for research/evaluation, unlike the rest of the pipeline. Off = the model's native VAE at the selected resolution."
                  >
                  <input
                    checked={usePid}
                    disabled={upscaleEnabled || hiresFixEnabled}
                    onChange={(event) => {
                      const enabled = event.target.checked;
                      setUsePid(enabled);
                      if (enabled) setDecoder("native");
                    }}
                      type="checkbox"
                    />
                    PiD decoder <span className="badge badge-nc">Non-Commercial</span>
                  </label>
                  {usePid ? (
                    <label
                      className="pid-target-select"
                      title="PiD super-resolves the base render 4×, so this sets the output size: 4K (~4096px, max detail) or 2K (~2048px, faster and less GPU memory). Both are super-resolved from the model's latent."
                    >
                      Output
                      <select
                        onChange={(event) => setPidTarget(event.target.value)}
                        value={pidTarget}
                      >
                        <option value="4k">4K · max detail</option>
                        <option value="2k">2K · faster</option>
                      </select>
                      {pidDecodeNotice ? (
                        <span className="field-hint pid-decode-hint" role="status">
                          {pidDecodeNotice.message}
                        </span>
                      ) : null}
                    </label>
                  ) : null}
                </>
              ) : null}
              {showDecoderPicker ? (
                <label className="alternate-decoder-select">
                  Decoder <span className="badge">Experimental</span>
                  <select
                    onChange={(event) => {
                      const next = event.target.value;
                      setDecoder(next);
                      if (next !== "native") setUsePid(false);
                    }}
                    value={decoder}
                  >
                    <option value="native">Native VAE (default)</option>
                    {decoderOptions.map((option) => {
                      const mib = option.estimatedSizeBytes
                        ? Math.round(option.estimatedSizeBytes / (1024 * 1024))
                        : null;
                      return (
                        <option disabled={!option.available} key={option.id} value={option.id}>
                          {option.label}{mib ? ` (+${mib} MiB)` : ""}{option.available ? "" : " — install component"}
                        </option>
                      );
                    })}
                  </select>
                  <span className="field-hint">
                    Alternate decode only; the model, text encoder, and native VAE remain installed.
                    Output licensing appends the selected decoder component.
                  </span>
                </label>
              ) : null}
              <label
                className="checkline upscale-toggle"
                title={usePid ? "Disabled while the PiD decoder is on — PiD already super-resolves to 2K/4K." : undefined}
              >
                <input
                  checked={upscaleEnabled}
                  disabled={usePid}
                  onChange={(event) => {
                    const enabled = event.target.checked;
                    setUpscaleEnabled(enabled);
                    if (enabled) {
                      setHiresFixEnabled(false);
                    }
                  }}
                  type="checkbox"
                />
                Upscale
              </label>
              {upscaleEngineIsRestoration(upscaleEngine) ? null : (
              <label>
                Scale
                <select disabled={!upscaleEnabled || usePid} onChange={(event) => setUpscaleFactor(Number(event.target.value))} value={upscaleFactor}>
                  {upscaleFactorsForEngine(upscaleEngine).map((factor) => (
                    <option key={factor} value={factor}>
                      {factor}x
                    </option>
                  ))}
                </select>
              </label>
              )}
              <label>
                Engine
                <select disabled={!upscaleEnabled || usePid} onChange={(event) => handleUpscaleEngineChange(event.target.value)} value={upscaleEngine}>
                  {availableUpscaleEngines.map((engine) => (
                    <option key={engine.key} value={engine.key}>
                      {engine.label}
                    </option>
                  ))}
                </select>
              </label>
              {upscaleEngineIsRestoration(upscaleEngine) && upscaleEnabled && !usePid ? (
                // Iris restoration (sc-25683): scale / input sizing / colour fix, with the engine
                // planner's processing + output dimensions for THIS generation's size shown before
                // submission (a budgeted result is labelled as such).
                <IrisRestoreControls
                  idPrefix="studio-iris"
                  onChange={(next) => setUpscaleRestore(irisRestoreControls(next))}
                  onStatus={setRestorePlanStatus}
                  planBody={restorePlanBody}
                  token={token}
                  value={upscaleRestore}
                />
              ) : null}
              {upscaleEngineHasSoftness(upscaleEngine) ? (
                <label title="Higher restores more detail from a degraded source; 0 keeps it faithful.">
                  Detail
                  <input
                    aria-label="SeedVR2 detail (softness)"
                    disabled={!upscaleEnabled || usePid}
                    max="1"
                    min="0"
                    onChange={(event) => setUpscaleSoftness(Number(event.target.value))}
                    step="0.05"
                    type="range"
                    value={upscaleSoftness}
                  />
                  <span>{upscaleSoftness.toFixed(2)}</span>
                </label>
              ) : null}
              <label
                className="checkline hires-fix-toggle"
                title={hiresFixDisabledReason}
              >
                <input
                  checked={hiresFixEnabled}
                  disabled={hiresFixDisabled}
                  onChange={(event) => {
                    const enabled = event.target.checked;
                    setHiresFixEnabled(enabled);
                    if (enabled) {
                      setUpscaleEnabled(false);
                    }
                  }}
                  type="checkbox"
                />
                Hires.fix
              </label>
              {hiresFixEnabled ? (
                <>
                  <label title="0 inherits the first-pass step count.">
                    Hires Steps
                    <input
                      aria-label="Hires Steps"
                      max="150"
                      min="0"
                      onChange={(event) => setHiresFixSteps(Number(event.target.value))}
                      step="1"
                      type="number"
                      value={hiresFixSteps}
                    />
                  </label>
                  <label title="How strongly the high-resolution pass may redraw the upscaled first pass.">
                    Denoising Strength
                    <input
                      aria-label="Hires Denoising Strength"
                      max="1"
                      min="0"
                      onChange={(event) =>
                        setHiresFixDenoisingStrength(Number(event.target.value))
                      }
                      step="0.05"
                      type="range"
                      value={hiresFixDenoisingStrength}
                    />
                    <span>{hiresFixDenoisingStrength.toFixed(2)}</span>
                  </label>
                  <label>
                    Upscale by
                    <select
                      aria-label="Hires Upscale by"
                      onChange={(event) => setHiresFixUpscaleBy(Number(event.target.value))}
                      value={hiresFixUpscaleBy}
                    >
                      {[1.25, 1.5, 1.75, 2, 2.5, 3, 4].map((factor) => (
                        <option key={factor} value={factor}>
                          {factor}x
                        </option>
                      ))}
                    </select>
                    <span
                      className="field-hint"
                      role={hiresFixTargetInvalid ? "alert" : undefined}
                    >
                      Output {hiresFixTargetWidth}×{hiresFixTargetHeight}
                      {hiresFixTargetInvalid ? " · exceeds 4096px limit" : ""}
                    </span>
                  </label>
                  <label
                    title={
                      supportsGuidance
                        ? "Blank inherits the first-pass CFG scale."
                        : "This model does not use CFG."
                    }
                  >
                    Hires CFG Scale
                    <input
                      aria-label="Hires CFG Scale"
                      disabled={!supportsGuidance}
                      max="60"
                      min="0"
                      onChange={(event) => setHiresFixCfgScale(event.target.value)}
                      placeholder="Inherit"
                      step="0.1"
                      type="number"
                      value={supportsGuidance ? hiresFixCfgScale : ""}
                    />
                  </label>
                </>
              ) : null}
              {supportsNegativePrompt ? (
                <label className="prompt-field">
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
            </div>
          </AdvancedSection>
  );
}
