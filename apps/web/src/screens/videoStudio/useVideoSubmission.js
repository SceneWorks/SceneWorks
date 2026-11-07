import { composeNegativePrompt } from "../../data/appearanceCatalog.js";
import { ltxVideoModelId, ltx25VideoModelId } from "./videoTierHelpers.js";
import { parseResolution } from "../../resolutionMatch.js";
import { effectiveFitMode } from "../../components/FitModeControl.jsx";
import { serializeLora } from "../../presetUtils.js";
import { replacementModeApplies } from "../../videoStudioValidation.js";
import { isMinimaxH3Entry } from "../../minimaxH3ReferenceResize.js";

// Extracted from VideoStudio along its existing concern boundary.
export function useVideoSubmission({
  submitting,
  executionTierBlockMessage,
  setSubmitting,
  stackActive,
  composedStack,
  createVideoJob,
  mode,
  styleApplied,
  composedStylePrompt,
  prompt,
  supportsNegativePrompt,
  negativePrompt,
  model,
  expectedImportedModelRef,
  ltx25AutoDurationEnabled,
  duration,
  fps,
  width,
  height,
  quality,
  seed,
  selectedPreset,
  characterId,
  characterLookId,
  importedLastOnly,
  sourceAssetOrigin,
  lastFrameAssetId,
  sourceAssetId,
  fitMode,
  acceptsLastOnly,
  sourceClipAssetId,
  outgoingSourceClipAssetIds,
  bridgeRightClipAssetId,
  outgoingReferenceAssetIds,
  outgoingReferenceAudioAssetIds,
  referenceClipAssetId,
  personTrackId,
  replacementMode,
  selectedLoras,
  effectiveLoraWeight,
  veda,
  resolution,
  durationHint,
  motion,
  selectedTrack,
  replacementModeLabels,
  styleId,
  stylePromptBase,
  appearanceSelection,
  appearanceActive,
  appearanceNegative,
  replayReferenceResize,
  selectedModel,
  ltxPipeline,
  distilledVariant,
  precision,
  transformerVariant,
  vaeDecoder,
  autoDuration,
  autoDurationMinSeconds,
  autoDurationMaxSeconds,
  temporalUpsampleRounds,
  supportsPromptEnhancement,
  enhancePrompt,
  supportsTextEncoderSelection,
  selectedTextEncoderModel,
  defaultTextEncoderId,
  showTorchQuantization,
  quantization,
  selectedTierQuantize,
  sampler,
  scheduler,
  schedulerShift,
  showLightning,
  lightning,
  lightningActive,
  stepsPinned,
  stepsOffMenu,
  stepsOverride,
  supportsGuidance,
  guidanceOverride,
  showsLegacyLtxGuidanceControls,
  ltxVideoCfg,
  ltxVideoStg,
  ltxVideoRescale,
  showClipStrength,
  videoConditioningStrength,
  bridgeRightVideoConditioningStrength,
  onLocalJobCreated
}) {
  async function submit(event) {
    event.preventDefault();
    // The button and validation summary already expose this refusal. Keep the submit boundary exact
    // as well so a direct/stale form submit cannot enqueue a SCAIL-2 Candle request that no product
    // tier is allowed to claim.
    if (submitting || executionTierBlockMessage) {
      return;
    }
    setSubmitting(true);
    try {
      // Fold the general-preset stack (epic 11949): send the composed prompt + negative and,
      // when a general sets aspect, the snapped resolution. The client is authoritative for the
      // composed prompt, so presetPromptResolvedClientSide tells the server to skip its fold.
      // (Video has no `count` field, so the stack's variations don't apply here.)
      const stackResolution = stackActive && composedStack.resolution ? parseResolution(composedStack.resolution) : null;
      const job = await createVideoJob({
        mode,
        // sc-13136: when a Style Catalog entry is active the composer has already wrapped the
        // (stack-folded) prompt LAST — send that composed string verbatim. Falls back to the plain
        // stack-folded / raw prompt when no style is applied.
        prompt: styleApplied ? composedStylePrompt : stackActive ? composedStack.prompt : prompt,
        // An engine with no negative-prompt axis gets an empty one rather than whatever a preset
        // stack composed — the field is hidden for it, so sending text the user cannot see or edit
        // (and the worker discards anyway) would be a ghost input on the recipe.
        negativePrompt: supportsNegativePrompt ? composeNegativePrompt(stackActive ? composedStack.negativePrompt : negativePrompt, appearanceNegative) : "",
        ...(appearanceActive ? { appearance: appearanceSelection } : {}),
        model,
        expectedImportedModelRef: expectedImportedModelRef ?? undefined,
        // The duration head chooses a legal duration only when the user explicitly opts in. Do not
        // send the visible manual-picker value alongside it: provider precedence is intentionally
        // represented by absence, so an explicit duration always wins by switching this toggle off.
        ...(ltx25AutoDurationEnabled ? {} : { duration: Number(duration) }),
        fps: Number(fps),
        width: stackResolution?.width ?? width,
        height: stackResolution?.height ?? height,
        quality,
        seed: seed === "" ? null : Number(seed),
        recipePresetId: selectedPreset?.id ?? null,
        // The studio seeds a selected preset's LoRAs into the visible `loras` (generationStudio's
        // preset-LoRA seed effect), so the client is authoritative for preset LoRAs — tell the
        // server to skip its own merge so edits/removals stick. Parity with the Image Studio.
        presetLorasResolvedClientSide: selectedPreset ? true : undefined,
        // The client composed the prompt (preset stack and/or Style Catalog wrap), so tell the
        // server to skip its own fold and send the composed string as-is (sc-13136).
        presetPromptResolvedClientSide: stackActive || styleApplied || undefined,
        characterId: characterId || null,
        characterLookId: characterLookId || null,
        // The authoritative guard (sc-26687): an IMPORTED H3 partition that accepts a last-only
        // request must never carry a first frame the user did not choose. The picker clears the
        // ambient one when a last frame is picked, but the ambient selection can also arrive
        // afterwards and the mode can change afterwards, so the request is decided HERE, where
        // the payload is built. A first frame the user chose here is theirs and is sent.
        //
        // The withdrawal needs a LAST frame to be about: without one this is an ordinary
        // image-to-video request, and withholding the first frame would leave it with no source
        // at all — which the API refuses ("Image to Video requires a source image"), while the
        // studio's own input check still saw the raw id and left Render enabled.
        sourceAssetId: ["image_to_video", "first_last_frame"].includes(mode)
          ? (importedLastOnly && mode === "image_to_video" && sourceAssetOrigin === "ambient" && lastFrameAssetId
            ? null
            : (sourceAssetId || null))
          : null,
        // Crop/Pad fit for the starting image (sc-6139) — only the image-conditioned
        // modes carry it; `effectiveFitMode(_, false)` coerces any stale outpaint back to
        // crop (video has no inpaint mask). Other modes omit it (DTO defaults to crop).
        fitMode: ["image_to_video", "first_last_frame"].includes(mode)
          ? effectiveFitMode(fitMode, false)
          : undefined,
        lastFrameAssetId: mode === "first_last_frame" || (acceptsLastOnly && mode === "image_to_video") ? lastFrameAssetId || null : null,
        sourceClipAssetId: [
          "extend_clip",
          "replace_person",
          "video_bridge",
          "video_to_video",
          "reference_video_to_video",
          "ads2v",
          // SCAIL-2 character animation (sc-5449): the driving video.
          "animate_character",
        ].includes(mode)
          ? sourceClipAssetId || null
          : null,
        // Bernini multi-source clips (sc-5425) — mv2v carries the array, and so does a
        // reference→video on a model that DECLARES a reference-clip cap (MiniMax-H3 Ref2VA takes
        // up to 3 clips alongside its images and audio, sc-17160).
        sourceClipAssetIds: outgoingSourceClipAssetIds,
        bridgeRightClipAssetId: mode === "video_bridge" ? bridgeRightClipAssetId || null : null,
        // Bernini subject references (sc-4703 / sc-5425) — the reference-driven modes + ads2v carry
        // them; SCAIL-2 character animation (sc-5449) carries the reference character image.
        referenceAssetIds: outgoingReferenceAssetIds,
        // Reference AUDIO clips (sc-17160). Gated on the MODEL's declared cap, not on a hardcoded
        // mode list like the two above: which modes take audio references is a model fact,
        // declared as `limits.maxReferenceAudioAssets` and enforced server-side, and a model that
        // takes none refuses a non-empty list at enqueue. Sending the raw state meant a selection
        // made on Ref2VA rode into a job for a model with no audio cap once the picker unmounted.
        referenceAudioAssetIds: outgoingReferenceAudioAssetIds,
        // Bernini ads2v reference video (sc-5425).
        referenceClipAssetId: mode === "ads2v" ? referenceClipAssetId || null : null,
        personTrackId: mode === "replace_person" ? personTrackId || null : null,
        // Gated on the MODEL as well as the mode (sc-20262), for the same reason
        // `referenceAudioAssetIds` above is: hiding a control does not unset the state behind it,
        // so a mode picked on a Wan-VACE engine would otherwise ride into a SCAIL-2 job once the
        // control unmounted — and SCAIL-2's engine refuses a non-default mode. `replacementMode`
        // is only meaningful where `replacementModeApplies`.
        replacementMode:
          mode === "replace_person" && replacementModeApplies(model) ? replacementMode : "face_only",
        loras: selectedLoras.map((lora) => serializeLora(lora, { weight: effectiveLoraWeight(lora) })),
        advanced: {
          // Keep the portable Veda object visible to workflow-share coverage. The helper call hid
          // its emitted key from the source scanner; this is the same enabled-only wire shape.
          ...(veda?.enabled ? { veda } : {}),
          resolution,
          durationHint,
          motion,
          selectedPersonTrack: selectedTrack ?? null,
          // The recipe's human-readable echo of the field above, so it follows the same gate — a
          // replayed SCAIL-2 recipe must not display a Replacement mode the job never carried.
          replacementModeLabel:
            replacementModeLabels[
              replacementModeApplies(model) ? replacementMode : "face_only"
            ],
          // Style Catalog round-trip (sc-13136, mirrors image sc-13132): record the picked style id
          // and the RAW pre-style prompt so replay re-selects the picker and recomposes the identical
          // prompt without double-wrapping. Rides advanced → rawAdapterSettings (cloned verbatim by
          // the worker; no backend change). Emitted only when a style is applied so styleless recipes
          // stay byte-identical.
          ...(styleApplied && styleId ? { styleId, stylePrompt: stylePromptBase } : {}),
          ...(appearanceActive ? { appearance: appearanceSelection, appearancePrompt: stylePromptBase, appearanceNegativePrompt: supportsNegativePrompt ? stackActive ? composedStack.negativePrompt : negativePrompt : "" } : {}),
          // sc-26691: a replayed H3 recipe's reference sizing, only while an H3 model is active
          // (any other family refuses the key at enqueue).
          ...(replayReferenceResize && isMinimaxH3Entry(selectedModel) ? replayReferenceResize : {}),
          ...(model === ltxVideoModelId ? { ltxPipeline, distilledVariant, precision } : {}),
          ...(model === ltx25VideoModelId
            ? {
                transformerVariant,
                vaeDecoder,
                autoDuration,
                autoDurationMinSeconds,
                autoDurationMaxSeconds,
                temporalUpsampleRounds,
              }
            : {}),
          ...(supportsPromptEnhancement && enhancePrompt
            ? { enhancePrompt: true }
            : {}),
          ...(supportsTextEncoderSelection &&
          enhancePrompt &&
          selectedTextEncoderModel !== defaultTextEncoderId
            ? { textEncoderModel: selectedTextEncoderModel }
            : {}),
          ...(showTorchQuantization && quantization !== "auto" ? { quantization } : {}),
          ...(selectedTierQuantize !== null ? { mlxQuantize: selectedTierQuantize } : {}),
          // Configurable sampler / scheduler (epic 1753). Sealed adapters
          // (LTX native, MLX) silently fall back to default; only the Wan
          // diffusers (torch) path actually applies these.
          ...(sampler && sampler !== "default" ? { sampler } : {}),
          ...(scheduler && scheduler !== "default" ? { scheduler } : {}),
          // An explicit Shift overrides the native grid independently of scheduler selection.
          ...(schedulerShift !== "" && schedulerShift != null &&
          Number.isFinite(Number(schedulerShift))
            ? { schedulerShift: Number(schedulerShift) }
            : {}),
          // Lightning fast-4-step toggle for Wan2.2 A14B MoE (sc-10048, epic 10043). Only the two
          // A14B engines honor it; emit the explicit bool for them (worker sc-10047 reads
          // `advanced.lightning`: absent → defaults on, false → off). When on the worker derives
          // the 4-step/CFG-off recipe, so we suppress the manual steps/guidance overrides below to
          // keep the payload consistent with the recipe the UI is reflecting.
          ...(showLightning ? { lightning } : {}),
          // `stepsPinned` suppresses the override for the same reason `lightningActive` does: the
          // count is not the caller's to set (sc-19502). Omitting `steps` entirely — rather than
          // sending the pinned value — is what "use the baked schedule" means to the engine, and it
          // also means a stale number left in the box by a previously-selected model can never leak
          // into the payload and 400.
          //
          // `stepsOffMenu` is the multi-entry half of the same suppression: the picker below shows
          // such a stale value as "model default", so emitting it anyway would send a count the UI
          // is not displaying AND that the enqueue gate refuses.
          ...(!lightningActive &&
          !stepsPinned &&
          !stepsOffMenu &&
          stepsOverride !== "" &&
          Number.isFinite(Number(stepsOverride))
            ? { steps: Number(stepsOverride) }
            : {}),
          ...(supportsGuidance &&
          guidanceOverride !== "" &&
          Number.isFinite(Number(guidanceOverride))
            ? { guidanceScale: Number(guidanceOverride) }
            : {}),
          // LTX native guidance knobs (epic 1753 sc-1769). Only emitted for
          // the LTX adapter — the worker would silently ignore them on other
          // adapters but keeping the payload tight avoids surprise overrides.
          ...(showsLegacyLtxGuidanceControls && ltxVideoCfg !== "" && Number.isFinite(Number(ltxVideoCfg))
            ? { videoCfgGuidanceScale: Number(ltxVideoCfg) }
            : {}),
          ...(showsLegacyLtxGuidanceControls && ltxVideoStg !== "" && Number.isFinite(Number(ltxVideoStg))
            ? { videoStgGuidanceScale: Number(ltxVideoStg) }
            : {}),
          ...(showsLegacyLtxGuidanceControls && ltxVideoRescale !== "" && Number.isFinite(Number(ltxVideoRescale))
            ? { videoRescaleScale: Number(ltxVideoRescale) }
            : {}),
          // Clip-conditioning strengths (sc-3522, sc-3755; sc-8445 for Krea Realtime v2v). The
          // worker reads these from `advanced`, defaulting to 1.0 when absent — extend uses the
          // source-clip strength, bridge uses both left and right, and Krea Realtime's v2v uses
          // the source-clip strength. `showClipStrength` is the same predicate that renders the
          // control, so the payload can never carry a strength the user was never offered.
          ...(showClipStrength &&
          videoConditioningStrength !== "" &&
          Number.isFinite(Number(videoConditioningStrength))
            ? { videoConditioningStrength: Number(videoConditioningStrength) }
            : {}),
          ...(mode === "video_bridge" &&
          bridgeRightVideoConditioningStrength !== "" &&
          Number.isFinite(Number(bridgeRightVideoConditioningStrength))
            ? { bridgeRightVideoConditioningStrength: Number(bridgeRightVideoConditioningStrength) }
            : {}),
        },
      });
      onLocalJobCreated?.(job);
    } finally {
      setSubmitting(false);
    }
  }
  return { submit };
}
