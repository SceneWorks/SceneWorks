import { ltx25VideoModelId, LTX25_AUTO_DURATION_MIN_SECONDS, LTX25_AUTO_DURATION_MAX_SECONDS, legacyDefaultTextEncoderId, clampLtx25AutoDurationSeconds, clampLtx25TemporalUpsampleRounds } from "./videoTierHelpers.js";
import { useState } from "react";

// Extracted from VideoStudio along its existing concern boundary.
export function useLtx25Controls({
  saved,
  model,
  selectedModel,
  macCapabilities
}) {
  const [ltxPipeline, setLtxPipeline] = useState(saved.ltxPipeline ?? "auto");

  const [distilledVariant, setDistilledVariant] = useState(saved.distilledVariant ?? "1.1");

  const [transformerVariant, setTransformerVariant] = useState(saved.transformerVariant ?? "distilled");

  const [vaeDecoder, setVaeDecoder] = useState(
    saved.vaeDecoder === "diffusion" ? "diffusion" : "conv",
  );

  const [autoDuration, setAutoDuration] = useState(saved.autoDuration === true);

  const [autoDurationMinSeconds, setAutoDurationMinSeconds] = useState(() =>
    clampLtx25AutoDurationSeconds(
      saved.autoDurationMinSeconds,
      LTX25_AUTO_DURATION_MIN_SECONDS,
    ),
  );

  const [autoDurationMaxSeconds, setAutoDurationMaxSeconds] = useState(() =>
    clampLtx25AutoDurationSeconds(
      saved.autoDurationMaxSeconds,
      LTX25_AUTO_DURATION_MAX_SECONDS,
    ),
  );

  const [temporalUpsampleRounds, setTemporalUpsampleRounds] = useState(() =>
    clampLtx25TemporalUpsampleRounds(saved.temporalUpsampleRounds),
  );

  const [precision, setPrecision] = useState(saved.precision ?? "fp8");

  const [enhancePrompt, setEnhancePrompt] = useState(saved.enhancePrompt ?? false);

  const [textEncoderSelection, setTextEncoderSelection] = useState({
    modelId: saved.model ?? null,
    id: saved.textEncoderModel ?? null,
  });

  const textEncoderModel =
    textEncoderSelection.modelId === model ? textEncoderSelection.id : null;

  const setTextEncoderModel = (next, modelId = model) =>
    setTextEncoderSelection((current) => {
      const currentId = current.modelId === modelId ? current.id : null;
      return {
        modelId,
        id: typeof next === "function" ? next(currentId) : next,
      };
    });

  const ltx25Dev = model === ltx25VideoModelId && transformerVariant === "dev";

  const ltx25AutoDurationEnabled = model === ltx25VideoModelId && autoDuration;

  // Runtime-curated selector surface (sc-13800). The API emits only complete encoders the worker can
  // resolve; Video Studio stays adapter-agnostic and future models can expose the same shape.
  const textEncoderOptions = selectedModel?.textEncoderOptions ?? [];

  // LTX-2.5's tier-local Gemma-4 generation encoder is not selectable. Runtime option discovery
  // is shared by the `ltx_video` adapter and can therefore carry LTX-2.3's Gemma-3/amoral choices
  // onto this catalog entry; exposing them would produce a request the 2.5 provider rejects.
  const supportsTextEncoderSelection =
    model !== ltx25VideoModelId && textEncoderOptions.length > 0;

  // The separately downloaded stock Gemma-4 enhancer is an MLX-only 2.5 capability. Keep the
  // simple opt-in checkbox while omitting the unrelated 2.3 encoder picker; off-Mac the Candle
  // provider has no prompt-enhancement input, so the control and payload both disappear. Require
  // a positive host identity: before capabilities load, showing a backend-specific control would
  // risk briefly offering it to an off-Mac client.
  const supportsStockPromptEnhancement =
    model === ltx25VideoModelId && ["macos", "darwin"].includes(macCapabilities?.platform);

  const supportsPromptEnhancement =
    supportsTextEncoderSelection || supportsStockPromptEnhancement;

  const defaultTextEncoderId =
    textEncoderOptions.find((option) => option.isDefault)?.id ??
    textEncoderOptions[0]?.id ??
    legacyDefaultTextEncoderId;

  const selectedTextEncoderModel =
    textEncoderModel == null ||
    (textEncoderModel === legacyDefaultTextEncoderId &&
      !textEncoderOptions.some((option) => option.id === legacyDefaultTextEncoderId))
      ? defaultTextEncoderId
      : textEncoderModel;

  const selectedTextEncoderAvailable = textEncoderOptions.some(
    (option) => option.id === selectedTextEncoderModel,
  );

  // LTX-2.3 native guidance knobs (epic 1753 sc-1769). The native ltx-core
  // path has no diffusers scheduler to swap — these three values (cfg + STG +
  // rescale) drive its sealed MultiModalGuiderParams instead.
  const [ltxVideoCfg, setLtxVideoCfg] = useState(saved.videoCfgGuidanceScale ?? "");

  const [ltxVideoStg, setLtxVideoStg] = useState(saved.videoStgGuidanceScale ?? "");

  const [ltxVideoRescale, setLtxVideoRescale] = useState(saved.videoRescaleScale ?? "");

  // Clip-conditioning strengths for the LTX IC-LoRA extend/bridge paths (sc-3522,
  // sc-3755) and for Krea Realtime's video_to_video (sc-8445). The worker reads these from
  // `advanced` (default 1.0 when absent): the source/left clip uses videoConditioningStrength,
  // the bridge right clip uses bridgeRightVideoConditioningStrength.
  const [videoConditioningStrength, setVideoConditioningStrength] = useState(saved.videoConditioningStrength ?? "");

  const [bridgeRightVideoConditioningStrength, setBridgeRightVideoConditioningStrength] = useState(
    saved.bridgeRightVideoConditioningStrength ?? "",
  );

  // The generic LTX-2.3 guider knobs map to MultiModalGuiderParams. LTX-2.5 has its own sealed
  // distilled/dev sampling contracts, so retaining or emitting these values there would be inert.
  const showsLegacyLtxGuidanceControls =
    selectedModel?.adapter === "ltx_video" && model !== ltx25VideoModelId;
  return { ltxPipeline, setLtxPipeline, distilledVariant, setDistilledVariant, transformerVariant, setTransformerVariant, vaeDecoder, setVaeDecoder, autoDuration, setAutoDuration, autoDurationMinSeconds, setAutoDurationMinSeconds, autoDurationMaxSeconds, setAutoDurationMaxSeconds, temporalUpsampleRounds, setTemporalUpsampleRounds, precision, setPrecision, enhancePrompt, setEnhancePrompt, textEncoderModel, setTextEncoderModel, ltx25Dev, ltx25AutoDurationEnabled, textEncoderOptions, supportsTextEncoderSelection, supportsStockPromptEnhancement, supportsPromptEnhancement, defaultTextEncoderId, selectedTextEncoderModel, selectedTextEncoderAvailable, ltxVideoCfg, setLtxVideoCfg, ltxVideoStg, setLtxVideoStg, ltxVideoRescale, setLtxVideoRescale, videoConditioningStrength, setVideoConditioningStrength, bridgeRightVideoConditioningStrength, setBridgeRightVideoConditioningStrength, showsLegacyLtxGuidanceControls };
}
