import { sourcePrecisionLabel } from "../../quantTier.js";
import { humanizedNumberMenu, ltxVideoModelId, ltx25VideoModelId, LTX25_AUTO_DURATION_MIN_SECONDS, LTX25_AUTO_DURATION_MAX_SECONDS, clampLtx25AutoDurationSeconds, clampLtx25TemporalUpsampleRounds } from "./videoTierHelpers.js";
import { SamplerSchedulerFields, SamplingSeedField, SamplingStepsField } from "../../components/SamplerSchedulerFields.jsx";
import React from "react";
import { turboRecipeSummary } from "../../minimaxH3Turbo.js";
import { qualityChoices } from "../../jobTypes.js";
import { guidanceDefaultFromModel, stepsDefaultFromModel } from "../../samplerOptions.js";
import { VedaControls } from "../../components/VedaControls.jsx";
import { StudioSettingsGroup } from "../../components/StudioSettingsTabs.jsx";

// Whether the "For this model" group under Sampling has anything to show — the model-specific
// blocks (Wan Lightning, MiniMax-H3 Turbo + Veda, LTX-2.3, LTX-2.5). The group is omitted when
// none apply.
export function videoModelSpecificSettingsApply({ showLightning, showTurbo, veda, model }) {
  return Boolean(showLightning || showTurbo || veda?.enabled || model === ltxVideoModelId || model === ltx25VideoModelId);
}

// Settings tab › Sampling: the sampling overrides that lived in Advanced (seed, steps, guidance,
// sampler / scheduler / shift, negative prompt, clip strength, SVD quality, legacy LTX guidance),
// then the "For this model" group of engine-specific blocks. Cleared values → model default.
export function VideoSamplingSettings({
  showTurbo,
  veda,
  setVeda,
  vedaSteps,
  showLightning,
  lightning,
  setLightning,
  turboVariants,
  selectTurboVariant,
  activeTurboVariant,
  selectedModel,
  model,
  setLtxPipeline,
  ltxPipeline,
  setDistilledVariant,
  distilledVariant,
  setPrecision,
  precision,
  setTransformerVariant,
  transformerVariant,
  ltx25Dev,
  setVaeDecoder,
  vaeDecoder,
  autoDuration,
  setAutoDuration,
  setAutoDurationMinSeconds,
  setAutoDurationMaxSeconds,
  autoDurationMinSeconds,
  autoDurationMaxSeconds,
  setTemporalUpsampleRounds,
  temporalUpsampleRounds,
  showsLegacyLtxGuidanceControls,
  setLtxVideoCfg,
  ltxVideoCfg,
  setLtxVideoStg,
  ltxVideoStg,
  setLtxVideoRescale,
  ltxVideoRescale,
  showClipStrength,
  mode,
  setVideoConditioningStrength,
  videoConditioningStrength,
  setBridgeRightVideoConditioningStrength,
  bridgeRightVideoConditioningStrength,
  showQualitySegment,
  quality,
  setQuality,
  seed,
  setSeed,
  sampler,
  scheduler,
  samplerOptions,
  schedulerOptions,
  setSampler,
  setScheduler,
  setSchedulerShift,
  schedulerShift,
  stepsChoice,
  lightningActive,
  stepsOffMenu,
  stepsOverride,
  stepsPinned,
  setStepsOverride,
  minSteps,
  stepsPinnedValue,
  supportsGuidance,
  setGuidanceOverride,
  guidanceOverride,
  supportsNegativePrompt,
  setNegativePrompt,
  negativePrompt,
}) {
  const modelSpecific = videoModelSpecificSettingsApply({ showLightning, showTurbo, veda, model });
  return (
    <div className="studio-grid video-settings-sampling">
      <SamplingSeedField
        value={seed}
        onChange={(value) => setSeed(value)}
        inputProps={{
          placeholder: "Random",
          type: "number"
        }}
      />
      <SamplingStepsField
        value={(stepsChoice && !ltx25Dev) ? lightningActive || stepsOffMenu ? "" : stepsOverride : lightningActive || stepsPinned ? "" : stepsOverride}
        onChange={(value) => setStepsOverride(value)}
        menu={(stepsChoice && !ltx25Dev) ? stepsChoice : null}
        defaultLabel={stepsDefaultFromModel(selectedModel) == null
                ? "Model default"
                : `${stepsDefaultFromModel(selectedModel)} (model default)`}
        inputProps={{
          min: String(minSteps),
          max: "80",
          disabled: lightningActive || stepsPinned,
          placeholder: lightningActive
            ? "4 (Lightning)"
            : stepsPinned
            ? `${stepsPinnedValue} (fixed schedule)`
            : /* Turbo supplies a step count but does NOT seize the control, unlike
            Lightning above: upstream's own spec table lists the 8-step MiniMax-H3
            adapter as "8 / 4", so a caller who knows the checkpoint may run it
            shorter, and `minimax_h3_sampling` honours an explicit
            `advanced.steps` over the variant's default for exactly that reason.
            So the placeholder REPORTS the recipe's count while the box stays
            editable — a knob honoured rather than rejected. */
            activeTurboVariant
            ? `${activeTurboVariant.sampling.steps} (Turbo)`
            : String(stepsDefaultFromModel(selectedModel) ?? ""),
          title: lightningActive
            ? "Governed by Lightning (fast 4-step). Turn Lightning off to set steps."
            : stepsPinned
            ? `${selectedModel?.ui?.label ?? selectedModel?.name ?? "This model"} is distilled: it runs a fixed ${stepsPinnedValue}-step schedule baked into its weights and cannot render any other step count.`
            : activeTurboVariant
            ? `${activeTurboVariant.name} is distilled for ${activeTurboVariant.sampling.steps} steps, which is what runs when this is blank. You can still set your own count.`
            : minSteps > 1
            ? `${selectedModel?.name ?? "This model"} needs at least ${minSteps} steps.`
            : undefined,
          type: "number"
        }}
        selectProps={(stepsChoice && !ltx25Dev) ? { disabled: lightningActive, title: lightningActive
                ? "Governed by Lightning (fast 4-step). Turn Lightning off to set steps."
                : `${selectedModel?.ui?.label ?? selectedModel?.name ?? "This model"} is distilled: it renders at ${humanizedNumberMenu(stepsChoice)} steps only.` } : undefined}
      />
      {supportsGuidance ? (
        <label>
          Guidance
          <input
            onChange={(event) => setGuidanceOverride(event.target.value)}
            placeholder={lightningActive ? "1 (Lightning default)" : (() => {
              const value = guidanceDefaultFromModel(selectedModel);
              return value == null ? "" : String(value);
            })()}
            step="any"
            type="number"
            value={guidanceOverride}
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
      />
      <label>
        Schedule shift
        <input
          disabled={selectedModel?.video?.supportsSchedulerShift === false}
          title={selectedModel?.video?.supportsSchedulerShift === false ? "This model has no native flow schedule shift." : undefined}
          min="0"
          onChange={(event) => setSchedulerShift(event.target.value)}
          step="any"
          placeholder={selectedModel?.defaults?.schedulerShift == null ? "Model default" : String(selectedModel.defaults.schedulerShift)}
          type="number"
          value={schedulerShift ?? ""}
        />
      </label>
      {showClipStrength ? (
        <>
          <label>
            {mode === "video_bridge" ? "Left clip strength" : "Clip strength"}
            <input
              min="0"
              max="1"
              onChange={(event) => setVideoConditioningStrength(event.target.value)}
              placeholder="1.0"
              step="0.05"
              type="number"
              value={videoConditioningStrength}
            />
          </label>
          {mode === "video_bridge" ? (
            <label>
              Right clip strength
              <input
                min="0"
                max="1"
                onChange={(event) => setBridgeRightVideoConditioningStrength(event.target.value)}
                placeholder="1.0"
                step="0.05"
                type="number"
                value={bridgeRightVideoConditioningStrength}
              />
            </label>
          ) : null}
        </>
      ) : null}
      {showQualitySegment ? (
        <label
          className="video-quality-field span-2"
          title="Step-count preset for Stable Video Diffusion: Draft 15, Balanced 25, Final 30. An explicit Steps value overrides it."
        >
          Quality
          <div className="quality-segment" role="radiogroup" aria-label="Quality">
            {qualityChoices.map(([value, label]) => (
              <button
                aria-checked={quality === value}
                className={quality === value ? "active" : ""}
                key={value}
                onClick={() => setQuality(value)}
                role="radio"
                type="button"
              >
                {label}
              </button>
            ))}
          </div>
        </label>
      ) : null}
      {showsLegacyLtxGuidanceControls ? (
        <>
          <label>
            Video CFG
            <input
              min="0"
              max="30"
              onChange={(event) => setLtxVideoCfg(event.target.value)}
              placeholder="4.0"
              step="0.1"
              type="number"
              value={ltxVideoCfg}
            />
          </label>
          <label>
            Video STG
            <input
              min="0"
              onChange={(event) => setLtxVideoStg(event.target.value)}
              placeholder="0.0"
              step="0.1"
              type="number"
              value={ltxVideoStg}
            />
          </label>
          <label>
            Video rescale
            <input
              min="0"
              max="2"
              onChange={(event) => setLtxVideoRescale(event.target.value)}
              placeholder="0.7"
              step="0.05"
              type="number"
              value={ltxVideoRescale}
            />
          </label>
        </>
      ) : null}
      {supportsNegativePrompt ? (
        <label className="prompt-field span-full">
          Negative prompt
          <textarea onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} />
        </label>
      ) : null}

      {modelSpecific ? (
        <StudioSettingsGroup label="For this model">
          {showLightning ? (
            <div className="lightning-toggle studio-field span-2">
              <label className="checkline">
                <input
                  checked={lightning}
                  onChange={(event) => setLightning(event.target.checked)}
                  type="checkbox"
                />
                Lightning (fast 4-step)
              </label>
              <p className="helper-copy">
                {lightning
                  ? "On: ~10× faster, 4 steps, CFG off, small quality trade-off. Steps and guidance are governed by the recipe."
                  : "Off: full multi-step quality with CFG (slower). Use the Steps and Guidance controls above."}
              </p>
            </div>
          ) : null}
          {/* MiniMax-H3 turbo (sc-18727). A VARIANT selector, not a toggle: the three published
              fl2v adapters carry three different (NFE, video shift) pairs, so "on" is not one
              state. Writes the SAME `selectedLoraIds` the LoRA picker writes — the control and
              the picker are two views of one selection, which is why they cannot disagree.
              Default-on (seeded once per model): sc-18729 measured 2.42 h against 12.6 min at
              the model's default canvas. */}
          {showTurbo ? (
            <div className="lightning-toggle studio-field span-2">
              <label>
                Turbo (step-distilled)
                <select
                  aria-label="Turbo (step-distilled)"
                  disabled={!turboVariants.length}
                  onChange={(event) => selectTurboVariant(event.target.value)}
                  value={activeTurboVariant?.id ?? ""}
                >
                  <option value="">Off — {selectedModel?.defaults?.steps ?? 50} steps</option>
                  {turboVariants.map((variant) => (
                    <option key={variant.id} value={variant.id}>
                      {variant.name} — {variant.sampling.steps} steps
                    </option>
                  ))}
                </select>
              </label>
              <p className="helper-copy">
                {!turboVariants.length
                  ? "No turbo adapter installed for this model. Install one from the LoRA library to render in 4–8 steps instead of the full schedule."
                  : activeTurboVariant
                    ? `On: ${turboRecipeSummary(activeTurboVariant)}. Roughly 7–12× faster (measured 12.6 min against 2.42 h at 1344×768). The distilled checkpoints are trained at 544p/768p and upstream is still improving their detail, so this is a different sample rather than the same one faster — turn it off for the reference schedule.`
                    : "Off: the full schedule at the model's own sigma shift. Slow — a default-canvas clip measured 2.42 h."}
              </p>
            </div>
          ) : null}
          {showTurbo || veda.enabled ? (
            <div className="span-full video-veda-controls">
              <VedaControls value={veda} onChange={setVeda} steps={vedaSteps} />
            </div>
          ) : null}
          {model === ltxVideoModelId ? (
            <>
              <label>
                LTX pipeline
                <select onChange={(event) => setLtxPipeline(event.target.value)} value={ltxPipeline}>
                  <option value="auto">Auto (follow quality)</option>
                  <option value="distilled">Distilled (single-stage)</option>
                  <option value="two_stage">Two-stage (dev + upscaler)</option>
                </select>
              </label>
              <label>
                Distilled variant
                <select onChange={(event) => setDistilledVariant(event.target.value)} value={distilledVariant}>
                  <option value="1.1">1.1 (newer aesthetic + audio)</option>
                  <option value="1.0">1.0 (original)</option>
                </select>
              </label>
              {sourcePrecisionLabel(selectedModel) ? null : (
                <label>
                  Precision
                  <select onChange={(event) => setPrecision(event.target.value)} value={precision}>
                    <option value="fp8">FP8 (lower VRAM)</option>
                    <option value="bf16">BF16 (higher quality, CPU offload)</option>
                  </select>
                </label>
              )}
            </>
          ) : null}
          {model === ltx25VideoModelId ? (
            <>
              <label>
                Transformer
                <select
                  aria-label="LTX-2.5 transformer"
                  onChange={(event) => setTransformerVariant(event.target.value)}
                  value={transformerVariant}
                >
                  <option value="distilled">Distilled (default, 8 steps)</option>
                  <option value="dev">Dev (guided, 30 steps)</option>
                </select>
                <p className="helper-copy">
                  {ltx25Dev
                    ? "Dev uses the guided 30-step transformer and the bundled stage-two distilled refinement."
                    : "Distilled is the default packed transformer and does not apply the refinement adapter twice."}
                </p>
              </label>
              <label>
                VAE decoder
                <select
                  aria-label="LTX-2.5 VAE decoder"
                  onChange={(event) => setVaeDecoder(event.target.value)}
                  value={vaeDecoder}
                >
                  <option value="conv">Conv VAE (default, faster)</option>
                  <option value="diffusion">DiffVAE (higher fidelity, slower)</option>
                </select>
                <p className="helper-copy">
                  Conv is the fast default. DiffVAE performs a separate diffusion decode for
                  finer faces, textures, and on-screen detail.
                </p>
              </label>
              <div className="lightning-toggle studio-field span-2">
                <label className="checkline">
                  <input
                    aria-label="LTX-2.5 auto duration"
                    checked={autoDuration}
                    onChange={(event) => setAutoDuration(event.target.checked)}
                    type="checkbox"
                  />
                  Choose duration from the prompt
                </label>
                <p className="helper-copy">
                  Uses LTX-2.5's optional duration head. Choose a manual duration under Model &amp;
                  output to turn this off and send that explicit duration instead.
                </p>
              </div>
              <label>
                Auto duration minimum (seconds)
                <input
                  aria-label="LTX-2.5 auto duration minimum"
                  disabled={!autoDuration}
                  max={LTX25_AUTO_DURATION_MAX_SECONDS}
                  min={LTX25_AUTO_DURATION_MIN_SECONDS}
                  onChange={(event) => {
                    const next = clampLtx25AutoDurationSeconds(
                      event.target.value,
                      LTX25_AUTO_DURATION_MIN_SECONDS,
                    );
                    setAutoDurationMinSeconds(next);
                    setAutoDurationMaxSeconds((current) => Math.max(current, next));
                  }}
                  step="1"
                  type="number"
                  value={autoDurationMinSeconds}
                />
              </label>
              <label>
                Auto duration maximum (seconds)
                <input
                  aria-label="LTX-2.5 auto duration maximum"
                  disabled={!autoDuration}
                  max={LTX25_AUTO_DURATION_MAX_SECONDS}
                  min={LTX25_AUTO_DURATION_MIN_SECONDS}
                  onChange={(event) => {
                    const next = clampLtx25AutoDurationSeconds(
                      event.target.value,
                      LTX25_AUTO_DURATION_MAX_SECONDS,
                    );
                    setAutoDurationMaxSeconds(next);
                    setAutoDurationMinSeconds((current) => Math.min(current, next));
                  }}
                  step="1"
                  type="number"
                  value={autoDurationMaxSeconds}
                />
              </label>
              <label>
                Temporal upsampling
                <select
                  aria-label="LTX-2.5 temporal upsampling"
                  onChange={(event) =>
                    setTemporalUpsampleRounds(clampLtx25TemporalUpsampleRounds(event.target.value))
                  }
                  value={temporalUpsampleRounds}
                >
                  <option value={0}>Off (default)</option>
                  <option value={1}>1 round (2× frames)</option>
                  <option value={2}>2 rounds (4× frames)</option>
                </select>
              </label>
            </>
          ) : null}
        </StudioSettingsGroup>
      ) : null}
      <p className="studio-note">Cleared values → model default.</p>
    </div>
  );
}
