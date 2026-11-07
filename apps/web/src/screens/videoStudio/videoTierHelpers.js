import { resolveJobResultAssets } from "../../jobResultAssets.js";
import { SCAIL2_MODEL_ID } from "../../videoStudioValidation.js";
import { tierLabel } from "../../quantTier.js";

export const MOTIONS = [
  "static",
  "slow push-in",
  "pull out",
  "pan left",
  "pan right",
  "tilt up",
  "tilt down",
  "handheld",
];

// Named so the initial state and the recipe replay's "absent → default" reset can't drift apart.
export const DEFAULT_MOTION = "slow push-in";

// Resolve a video job's result assets against the live catalog so the
// WorkerProgressCard video-player variant can play the finished clip (sc-2089).
// Shares the unified resolver (sc-8853); the video lane keeps the generationSetId
// fallback in catalog order (no batch-slot sort — that is image-only).
export function jobVideoResultAssets(job, assets) {
  return resolveJobResultAssets(job, assets, { type: "video" });
}

// "8" / "4 or 8" / "4, 8, or 12" — the same phrasing `humanized_number_menu` gives the enqueue
// gate's own rejection (crates/sceneworks-core/src/video_request.rs, sc-19502), so the Steps
// picker's tooltip states the legal set the way a 400 from that gate would.
export function humanizedNumberMenu(menu) {
  if (menu.length <= 1) return String(menu[0] ?? "");
  if (menu.length === 2) return `${menu[0]} or ${menu[1]}`;
  return `${menu.slice(0, -1).join(", ")}, or ${menu[menu.length - 1]}`;
}

export const ltxVideoModelId = "ltx_2_3";

export const ltx25VideoModelId = "ltx_2_5";

export const LTX25_AUTO_DURATION_MIN_SECONDS = 4;

export const LTX25_AUTO_DURATION_MAX_SECONDS = 15;

export const LTX25_TEMPORAL_UPSAMPLE_MAX_ROUNDS = 2;

// sc-20969 terminal acceptance admits exactly SCAIL-2's shipped q4/q8/bf16 Candle packages. Keep
// this execution allowlist literal and local so an unrelated future catalog tier cannot become
// runnable merely by appearing in the manifest. MLX continues to use the complete installed tier set.
export const SCAIL2_CANDLE_PRODUCT_TIERS = Object.freeze(["q4", "q8", "bf16"]);

// LTX-2.5 ships its full q4/q8/bf16 ladder on Candle, symmetric with MLX. Keep this execution
// allowlist literal and local so an unrelated future catalog tier cannot become runnable merely by
// appearing in the manifest.
export const LTX25_CANDLE_PRODUCT_TIERS = Object.freeze(["q4", "q8", "bf16"]);

export const legacyDefaultTextEncoderId = "default";

export const amoralTextEncoderId = "ltx_amoral_gemma_3_12b";

export const TIER_SCREEN = "video";

export const MAX_SCAIL2_REFERENCE_CHARACTERS = 6;

export function videoExecutionTierModel(model, backend) {
  const acceptedTiers =
    model?.id === SCAIL2_MODEL_ID
      ? SCAIL2_CANDLE_PRODUCT_TIERS
      : model?.id === ltx25VideoModelId
        ? LTX25_CANDLE_PRODUCT_TIERS
        : null;
  if (backend !== "candle" || acceptedTiers === null) {
    return model;
  }
  const accepted = (tier) => acceptedTiers.includes(tier);
  return {
    ...model,
    variants: Array.isArray(model.variants)
      ? model.variants.filter((variant) => accepted(variant?.variant))
      : model.variants,
    runtimeQuantTiers: Array.isArray(model.runtimeQuantTiers)
      ? model.runtimeQuantTiers.filter(accepted)
      : model.runtimeQuantTiers,
    mlxTiers: Array.isArray(model.mlxTiers)
      ? model.mlxTiers.filter(accepted)
      : model.mlxTiers,
    mlxTierStates: Array.isArray(model.mlxTierStates)
      ? model.mlxTierStates.filter((state) => accepted(state?.tier))
      : model.mlxTierStates,
  };
}

export function clampLtx25AutoDurationSeconds(value, fallback) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return fallback;
  return Math.min(
    LTX25_AUTO_DURATION_MAX_SECONDS,
    Math.max(LTX25_AUTO_DURATION_MIN_SECONDS, Math.round(seconds)),
  );
}

export function clampLtx25TemporalUpsampleRounds(value) {
  const rounds = Number(value);
  if (!Number.isFinite(rounds)) return 0;
  return Math.min(LTX25_TEMPORAL_UPSAMPLE_MAX_ROUNDS, Math.max(0, Math.round(rounds)));
}

export function unavailableRecipeTierMessage(model, backend, tier) {
  const recorded = tierLabel(tier);
  if (
    backend === "candle" &&
    model?.id === SCAIL2_MODEL_ID &&
    !SCAIL2_CANDLE_PRODUCT_TIERS.includes(tier)
  ) {
    return `This recipe was recorded at ${recorded}, which is not enabled for SCAIL-2 Candle generation. Replay it on MLX, or install an admitted SCAIL-2 tier in Model Manager before starting a new generation.`;
  }
  return `This recipe was recorded at ${recorded}, but that tier is not installed and available for ${model?.name ?? "the selected model"} on ${backend === "mlx" ? "MLX" : "Candle"}. Install or repair that tier in Model Manager before replaying this recipe.`;
}

// Video sub-modes that map onto a recipe workflow. extend_clip / replace_person
// aren't recipe workflows, so "Save as Preset" is gated to these.
export const VIDEO_PRESET_MODES = ["image_to_video", "text_to_video", "first_last_frame"];
