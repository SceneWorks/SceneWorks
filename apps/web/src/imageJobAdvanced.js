import { buildStructuredPromptRecipe } from "./ideogramCaption.js";
import { tierQuantize, isConvRotTier } from "./quantTier.js";

// sc-8854 (F-052): pure builder for the Image Studio job's `advanced` payload. Extracted
// verbatim from ImageStudio.submit() — the ~110-line object literal that assembled the
// payload from ~15 conditional spreads was the app's highest-drift surface (each new
// advanced knob threaded a new conditional through the middle of a 240-line async
// function). Pulling it into a pure state → payload function makes it independently
// unit-testable and keeps submit() focused on orchestration (prompt resolution, the API
// call, submitting-state). Behavior is preserved exactly: every conditional, guard, and
// omit-when-default rule (which keeps existing recipes byte-identical) is unchanged.
//
// The worker clones `advanced` verbatim into the asset's rawAdapterSettings, so omitting a
// key when it equals the engine default is load-bearing — it keeps saved recipes stable
// across releases. Do not "simplify" a spread into an always-present key.
export function buildImageJobAdvanced(state) {
  const {
    resolution,
    // Structured-prompt recipe round-trip (sc-6147).
    sendStructured,
    submitIntent,
    submitCaption,
    submitBackend,
    // Sampler / scheduler (epic 1753 / 7114).
    sampler,
    scheduler,
    schedulerShift,
    // Step / guidance overrides.
    stepsOverride,
    guidanceOverride,
    guidanceMethod,
    // Flash attention (sc-3674).
    flashAttn,
    // Caption upsampling (sc-6135).
    promptEnhance,
    enhancePrompt,
    // Boogu precision (sc-6568) + quant-tier A/B (sc-8515).
    precisionToggle,
    bf16Precision,
    showTierPicker,
    quantTier,
    // PiD decoder (epic 7840).
    showPidToggle,
    usePid,
    // Character-reference knobs.
    mode,
    referenceAssetId,
    hideReferenceStrength,
    ipAdapterScale,
    identityStructure,
    controlnetScale,
    variationStrength,
    trueCfgScale,
    viewAngles,
    viewAngle,
    // Pose library.
    posePayload,
    faceRestore,
    // Strict-control conditioning (epic 8236).
    controlActive,
    activeControlMode,
    controlPassthroughId,
    effectiveControlScale,
  } = state;

  return {
    resolution,
    // Structured-prompt recipe round-trip (sc-6147): persist the full caption +
    // original intent + magic-prompt backend alongside the job so "Use this recipe"
    // can rehydrate the builder rather than replay the serialized JSON as a plain
    // prompt. Rides in `advanced`, which the worker clones verbatim into the asset's
    // rawAdapterSettings — no backend change needed. Only for structured models.
    ...(sendStructured
      ? {
          structuredPrompt: buildStructuredPromptRecipe({
            intent: submitIntent,
            caption: submitCaption,
            magicPromptBackend: submitBackend,
            edited: !submitBackend,
          }),
        }
      : {}),
    // Configurable sampler / scheduler (epic 1753). Worker registry
    // falls back to model-native when given "default", so emitting the
    // values unconditionally is safe — invalid values are ignored.
    ...(sampler && sampler !== "default" ? { sampler } : {}),
    ...(scheduler && scheduler !== "default" ? { scheduler } : {}),
    // Guidance method (epic 7434). "cfg" is the engine-standard no-op, so it is
    // omitted — keeping existing recipes byte-identical; only a non-default
    // method (CFG++) rides the payload. The worker N3-falls an unadvertised
    // method back to the default, so an invalid value is harmless.
    ...(guidanceMethod && guidanceMethod !== "cfg" ? { guidanceMethod } : {}),
    // The schedule shift (time-shift mu) is only honored when a curated
    // (non-default) scheduler is active — it shapes that curated schedule;
    // the default scheduler keeps the engine's resolution-native shift, so
    // emitting it there would override the no-op default (epic 7114).
    ...(scheduler &&
    scheduler !== "default" &&
    Number.isFinite(Number(schedulerShift))
      ? { schedulerShift: Number(schedulerShift) }
      : {}),
    // Step / guidance overrides — empty string means "use the model
    // default", which the worker reads off MODEL_TARGETS.
    ...(stepsOverride !== "" && Number.isFinite(Number(stepsOverride))
      ? { steps: Number(stepsOverride) }
      : {}),
    ...(guidanceOverride !== "" && Number.isFinite(Number(guidanceOverride))
      ? { guidanceScale: Number(guidanceOverride) }
      : {}),
    // Flash attention (sc-3674): only emitted when toggled OFF — the worker defaults to ON
    // when `advanced.flashAttn` is absent, so the default-on case adds nothing to the payload.
    // Only the candle (Windows/CUDA) SDXL backend reads it; every other backend ignores it.
    ...(flashAttn ? {} : { flashAttn: false }),
    // FLUX.2-dev caption upsampling (sc-6135): emitted only when the model declares the
    // toggle AND it's on (off-by-default; the worker/engine ignore it for other models).
    ...(promptEnhance && enhancePrompt ? { enhancePrompt: true } : {}),
    // Boogu precision (sc-6568): emit mlxQuantize:0 (full-precision bf16) only when the model
    // exposes the precision toggle AND bf16 is selected; the default Q8 emits nothing (the
    // worker reads manifest mlx.quantize and fetches the `<variant>-bf16/` subfolder on demand).
    // `!showTierPicker` is a defensive guard: Boogu downloads via `base/`-style subfolder globs
    // (no `downloads[].variant` keys), so `hasVariantMatrix` — and therefore `showTierPicker` —
    // is always false for the precisionToggle set; the two controls are disjoint. The guard
    // makes that invariant load-bearing so a future manifest change can never emit both
    // mlxQuantize spreads for one model (the tier picker below would win the object-spread
    // race, but we never render/emit both).
    ...(precisionToggle && bf16Precision && !showTierPicker ? { mlxQuantize: 0 } : {}),
    // Quant-tier A/B (sc-8515): when the model has >1 tier installed and a tier is picked,
    // send that tier's mlxQuantize (bf16→0, q8→8, q4→4). The worker's resolve_quant +
    // generator cache route to it (reload-always). Emitted only when the picker is shown
    // AND the picked tier maps to a known quant value, so single-tier models and the
    // "default" pseudo-variant never leak an mlxQuantize into the payload. Disjoint from the
    // Boogu precisionToggle above (non-matrix models), enforced by its `!showTierPicker` guard.
    ...(showTierPicker && tierQuantize(quantTier) !== null
      ? { mlxQuantize: tierQuantize(quantTier) }
      : {}),
    // Krea 2 INT8-ConvRot tier (sc-9300, epic 9083): the online-rotation int8 DiT is NOT a bits-based
    // quant, so it can't ride `mlxQuantize` (its `tierQuantize` is null, so the spread above omits it).
    // Instead send a distinct `convRot: true` the worker maps to the ConvRot LoadSpec construction
    // (weights = Dir(bf16 snapshot) + text_encoder = File(convrot DiT)). Emitted only when the picker
    // is shown AND the picked tier is int8-convrot — disjoint from the mlxQuantize spread above.
    ...(showTierPicker && isConvRotTier(quantTier) ? { convRot: true } : {}),
    // PiD decoder (epic 7840, sc-7851): emit usePid:true only when the toggle is shown
    // (model PiD-eligible AND checkpoint installed) AND on. The worker swaps the native
    // VAE for the PiD decode + 2K/4K super-resolve pass; it rides `advanced` (opaque
    // pass-through, zero contract-snapshot drift) and is cloned into the asset's
    // rawAdapterSettings — that recorded `usePid:true` is the output's non-commercial
    // marker. The worker independently no-ops to the native VAE if the checkpoint is gone.
    ...(showPidToggle && usePid ? { usePid: true } : {}),
    // IP-Adapter / InstantID reference strength only applies when a character
    // reference is attached AND the model uses the IP-Adapter knob; Qwen's
    // edit pipeline ignores this scalar (hideReferenceStrength gates it out).
    ...(mode === "character_image" && referenceAssetId && !hideReferenceStrength
      ? { ipAdapterScale }
      : {}),
    // Identity structure (controlnetConditioningScale) is InstantID-only — sent
    // only when the model exposes the control and a reference is attached.
    ...(mode === "character_image" && referenceAssetId && identityStructure
      ? { controlnetConditioningScale: controlnetScale }
      : {}),
    // Variation knob (trueCfgScale) — FLUX uses it alongside ipAdapterScale,
    // Qwen uses it as the only variation lever. Sent only when the model
    // declares a variationStrength slider AND a reference is attached.
    ...(mode === "character_image" && referenceAssetId && variationStrength
      ? { trueCfgScale }
      : {}),
    // View angle (InstantID) — only when a specific angle is chosen and no pose is
    // selected (a library pose drives the whole body, superseding the head angle).
    ...(mode === "character_image" && referenceAssetId && viewAngles && viewAngle && !posePayload.length
      ? { viewAngle }
      : {}),
    // Pose library (InstantID) — one image per selected pose; faceRestore toggles
    // the full-body face-restoration pass.
    ...(posePayload.length ? { poses: posePayload, faceRestore } : {}),
    // Strict-control conditioning (epic 8236, sc-8245). The control type the worker's shared
    // strict-control driver reads (strict_control.rs `requested_control_kind`). Pose is the
    // engine default (omitted → byte-preserved), so only non-pose modes ride the payload. The
    // control-lock strength (`advanced.controlScale`) is sent whenever the panel is active.
    ...(controlActive && activeControlMode !== "pose" ? { controlMode: activeControlMode } : {}),
    // Use-as-is passthrough: a pre-made canny/depth map fed verbatim
    // (strict_control.rs `resolve_user_control_map`). Derive mode uses request.sourceAssetId.
    ...(controlPassthroughId ? { controlImage: controlPassthroughId } : {}),
    ...(controlActive ? { controlScale: effectiveControlScale } : {}),
  };
}
