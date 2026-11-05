import { GpuSelect } from "../../components/StudioFields.jsx";
import { CharacterLookFields } from "../../components/StudioFields.jsx";
import { humanizedNumberMenu, ltxVideoModelId, ltx25VideoModelId, LTX25_AUTO_DURATION_MIN_SECONDS, LTX25_AUTO_DURATION_MAX_SECONDS, clampLtx25AutoDurationSeconds, clampLtx25TemporalUpsampleRounds } from "./videoTierHelpers.js";
import { SamplerSchedulerFields, SamplingSeedField, SamplingStepsField } from "../../components/SamplerSchedulerFields.jsx";
import React from "react";
import { VideoUpscalePanel } from "../VideoUpscalePanel.jsx";
import { turboRecipeSummary } from "../../minimaxH3Turbo.js";
import { qualityChoices } from "../../jobTypes.js";
import { guidanceDefaultFromModel, stepsDefaultFromModel } from "../../samplerOptions.js";
import { VedaControls } from "../../components/VedaControls.jsx";

export function VideoAdvancedPanel({
  showTurbo,
  veda,
  setVeda,
  vedaSteps,
  setFps,
  fps,
  fpsOptions,
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
  showTorchQuantization,
  setQuantization,
  quantization,
  quantVariants,
  setRequestedGpu,
  requestedGpu,
  gpuOptions,
  seed,
  setSeed,
  sampler,
  scheduler,
  samplerOptions,
  schedulerOptions,
  setSampler,
  setScheduler,
  showSchedulerPicker,
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
  setCharacterId,
  characterId,
  characters,
  setCharacterLookId,
  characterLookId,
  supportsNegativePrompt,
  setNegativePrompt,
  negativePrompt,
  createVideoUpscaleJob,
  importAsset,
  macCapabilities,
  onLocalJobCreated,
  onOpenQueue,
  selectedAsset,
  activeProject,
  videoAssets
}) {
  return (
<div className="advanced-panel">
              {showTurbo || veda.enabled ? <VedaControls value={veda} onChange={setVeda} steps={vedaSteps} /> : null}
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
              {showLightning ? (
                <div className="lightning-toggle">
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
                      : "Off: full multi-step quality with CFG (slower). Use the Steps and Guidance controls below."}
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
                <div className="lightning-toggle">
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
                  <label>
                    Precision
                    <select onChange={(event) => setPrecision(event.target.value)} value={precision}>
                      <option value="fp8">FP8 (lower VRAM)</option>
                      <option value="bf16">BF16 (higher quality, CPU offload)</option>
                    </select>
                  </label>
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
                  <div className="lightning-toggle">
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
                      Uses LTX-2.5's optional duration head. Choose a manual duration above to
                      turn this off and send that explicit duration instead.
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
              {supportsPromptEnhancement ? (
                <>
                  <div className="lightning-toggle">
                    <label className="checkline">
                      <input
                        checked={enhancePrompt}
                        onChange={(event) => setEnhancePrompt(event.target.checked)}
                        type="checkbox"
                      />
                      Enhance prompt before generation
                    </label>
                    <p className="helper-copy">
                      {supportsStockPromptEnhancement
                        ? "Rewrites the prompt with LTX-2.5's separately downloaded stock Gemma-4 enhancer."
                        : "Rewrites the prompt with the selected text encoder before the model encodes it."}
                    </p>
                  </div>
                  {supportsTextEncoderSelection ? (
                    <>
                      <label>
                        Text encoder model
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
                      <p className="helper-copy">
                        {!selectedTextEncoderAvailable &&
                        selectedTextEncoderModel !== defaultTextEncoderId
                          ? "The recorded encoder is not staged on this worker. Choose the shipped default or stage the alternate before rendering."
                          : textEncoderOptions.length > 1
                            ? "Only complete encoders already staged for this worker are listed."
                            : "The shipped encoder is the default. Complete operator-staged alternates appear here after Models is refreshed."}
                      </p>
                    </>
                  ) : null}
                </>
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
                      max="10"
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
                  className="video-quality-field"
                  title="Step-count preset for Stable Video Diffusion: Draft 15, Balanced 25, Final 30. An explicit Steps value below overrides it."
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
              <SamplingSeedField
                value={seed}
                onChange={(value) => setSeed(value)}
                inputProps={{
                  placeholder: "Random",
                  type: "number"
                }}
              />
              <SamplerSchedulerFields
                sampler={sampler}
                scheduler={scheduler}
                samplerOptions={samplerOptions}
                schedulerOptions={schedulerOptions}
                onSamplerChange={setSampler}
                onSchedulerChange={setScheduler}
              />
              {showSchedulerPicker && scheduler !== "default" ? (
                <label>
                  Schedule shift
                  <input
                    max="10"
                    min="0.1"
                    onChange={(event) => setSchedulerShift(Number(event.target.value))}
                    step="0.1"
                    type="number"
                    value={schedulerShift}
                  />
                </label>
              ) : null}
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
                    min="0"
                    max="30"
                    disabled={lightningActive}
                    onChange={(event) => setGuidanceOverride(event.target.value)}
                    placeholder={lightningActive ? "off (Lightning)" : (() => {
                      const value = guidanceDefaultFromModel(selectedModel);
                      return value == null ? "" : String(value);
                    })()}
                    step="0.1"
                    title={lightningActive ? "Governed by Lightning (fast 4-step). Turn Lightning off to set guidance." : undefined}
                    type="number"
                    value={lightningActive ? "" : guidanceOverride}
                  />
                </label>
              ) : null}
              <CharacterLookFields characters={characters} characterId={characterId} lookId={characterLookId}
                    onCharacterChange={setCharacterId} onLookChange={setCharacterLookId} />

              {supportsNegativePrompt ? (
                <label className="prompt-field">
                  Negative prompt
                  <textarea onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} />
                </label>
              ) : null}
              {characterId ? (
                <div className="guidance-strip">
                  <strong>Character reference</strong>
                  <span>
                    Character and look are saved with the recipe; LTX image conditioning uses IC-LoRA when the selected preset includes one.
                  </span>
                </div>
              ) : null}
              {/* Video upscale (super-resolve an existing clip) folds into Advanced — it
                  previously lived in the render rail this layout removes. It operates on a
                  selected existing asset, independent of the current generation payload. */}
              <VideoUpscalePanel
                characters={characters}
                createVideoUpscaleJob={createVideoUpscaleJob}
                importAsset={importAsset}
                macCapabilities={macCapabilities}
                onSubmitted={(job) => {
                  onLocalJobCreated(job);
                  onOpenQueue();
                }}
                selectedAsset={selectedAsset}
                projectId={activeProject?.id}
                videoAssets={videoAssets}
              />
            </div>
  );
}
