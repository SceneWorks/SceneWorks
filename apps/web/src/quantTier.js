// Generation-time quant-tier selection (sc-8515, epic 8506). When a user has MORE THAN ONE
// quant tier of a model INSTALLED, the studio lets them toggle which tier generates, for A/B
// comparison. This module derives the picker's options from the model's per-variant install
// state (sc-8508: /models emits `hasVariantMatrix` + a `variants[]` array, each carrying a
// `variant` key and an `installState`), and maps the chosen tier to the worker control the
// generation already understands — `advanced.mlxQuantize`.
//
// The worker side is already done (GeneratorCacheKey includes `quantize`; `resolve_quant`
// honors `advanced.mlxQuantize`), so this is purely: which tiers are installed, and what
// mlxQuantize value does the picked tier send. Reload-always (epic decision 4): switching a
// heavy tier evicts + reloads on the worker; the studio surfaces a brief "loading" state and
// never attempts co-residence.

// Tier key → `advanced.mlxQuantize` value. bf16 → 0 (the worker's `<= 0` ⇒ dense/bf16 sentinel),
// q8 → 8, q4 → 4. A dense-TE model keeps its TE bf16 internally regardless (the worker forces
// that); the UI just sends the tier's nominal quant value and lets the worker handle the nuance.
const TIER_QUANTIZE = {
  bf16: 0,
  q8: 8,
  q4: 4,
};

// The candle-only Krea 2 INT8-ConvRot tier key (sc-9300, epic 9083). NOT a bits-based quant — the
// online-rotation int8 DiT can't be expressed as an `mlxQuantize` value — so it is DELIBERATELY absent
// from TIER_QUANTIZE and instead sends a distinct `advanced.convRot: true` signal (see
// imageJobAdvanced.js). It is a selectable tier (`isSelectableTier` accepts it) but `tierQuantize`
// returns null for it, so it never leaks an `mlxQuantize` into the payload.
export const INT8_CONVROT_TIER = "int8-convrot";

// Whether a tier key names the candle INT8-ConvRot tier.
export function isConvRotTier(tier) {
  return tier === INT8_CONVROT_TIER;
}

// Human labels for the picker, keyed by tier. Unknown/"default" tiers fall back to the raw key.
// "training" is NOT a quant tier: it's the flat-diffusers LoRA-training base some tiered models
// (lens, sc-8797) additionally host on macOS. It's absent from TIER_QUANTIZE, so the generation
// picker and RAM suggestion ignore it; only the Models download panel lists it (with this label).
const TIER_LABELS = {
  bf16: "Full precision (bf16)",
  q8: "Q8 (balanced)",
  q4: "Q4 (smallest)",
  training: "LoRA training base (bf16 diffusers)",
  // Candle-only (Windows/Linux, sm_89+). Online-rotation int8 DiT — closer to bf16 than Q4 (sc-9300).
  [INT8_CONVROT_TIER]: "INT8-ConvRot (candle, sm_89+)",
};

// Display order (smallest → largest); tiers not in this list sort after, alphabetically. int8-convrot
// sits between q4 and q8 by footprint/fidelity (int8 DiT, PSNR 34.4 dB — better than Q4's 22.7 dB).
const TIER_ORDER = ["q4", INT8_CONVROT_TIER, "q8", "bf16"];

// Map a tier key to its `advanced.mlxQuantize` value, or null when the key isn't a known quant
// tier (e.g. "default" on a single-variant model — such models never render the picker).
export function tierQuantize(tier) {
  return Object.prototype.hasOwnProperty.call(TIER_QUANTIZE, tier)
    ? TIER_QUANTIZE[tier]
    : null;
}

export function tierLabel(tier) {
  return TIER_LABELS[tier] ?? tier;
}

// Whether a tier key is a user-selectable generation tier: a known bits-based quant (bf16/q8/q4) OR
// the candle INT8-ConvRot tier. Excludes the "default" pseudo-variant of a single-variant model and
// non-generation pseudo-tiers like "training". Distinct from `tierQuantize` (which returns null for
// int8-convrot because it has no mlxQuantize value — the tier still selects, via `advanced.convRot`).
export function isSelectableTier(tier) {
  return tierQuantize(tier) !== null || isConvRotTier(tier);
}

// The installed, selectable quant tiers of a model, in display order. A tier is selectable when it is
// a known quant tier (bf16/q8/q4) OR the candle INT8-ConvRot tier (`isSelectableTier` — the "default"
// pseudo-variant of a single-variant model is excluded) AND its files are installed. Returns [] when
// the model has no variant matrix.
//
// `options.convRotEligible` (sc-9300, default true) gates the candle-only INT8-ConvRot tier: the
// caller passes `false` when NO live worker advertises the `int8_convrot` capability (macOS/MLX, or a
// pre-Ada NVIDIA GPU that fails the sm_89 compute-cap probe), so the tier is HIDDEN on an ineligible
// host even when its files happen to be present in the cache. Every other tier is unaffected. Default
// true keeps existing single-lane call sites (and tests) unchanged.
export function installedTiers(model, options = {}) {
  const { convRotEligible = true } = options;
  if (!model?.hasVariantMatrix || !Array.isArray(model.variants)) {
    return [];
  }
  return model.variants
    .filter(
      (variant) =>
        variant &&
        isSelectableTier(variant.variant) &&
        (convRotEligible || !isConvRotTier(variant.variant)) &&
        variant.installState === "installed",
    )
    .map((variant) => variant.variant)
    .sort((a, b) => {
      const ai = TIER_ORDER.indexOf(a);
      const bi = TIER_ORDER.indexOf(b);
      if (ai === -1 && bi === -1) {
        return a.localeCompare(b);
      }
      if (ai === -1) {
        return 1;
      }
      if (bi === -1) {
        return -1;
      }
      return ai - bi;
    });
}

// Whether the studio should render the tier picker: only when MORE THAN ONE quant tier is
// installed (a single installed tier — the common case — shows no toggle, per acceptance).
// `options` (sc-9300) forwards the `convRotEligible` gate so an ineligible host doesn't count the
// hidden INT8-ConvRot tier toward the >1 threshold.
export function shouldShowTierPicker(model, options = {}) {
  return installedTiers(model, options).length > 1;
}

// The tier that declares itself the default download (`variant.default === true`) IF it is
// installed and selectable, else null. Used to seed the picker when there's no last-used tier.
function defaultInstalledTier(model, tiers) {
  if (!model?.hasVariantMatrix || !Array.isArray(model.variants)) {
    return null;
  }
  const declared = model.variants.find(
    (variant) => variant && variant.default === true && tiers.includes(variant.variant),
  );
  return declared ? declared.variant : null;
}

// The tier the picker should start on for `model`. Preference order:
//   1. `lastUsed` — the per-model last-used tier, when it is still installed (persistence).
//   2. the model's declared default tier (`variant.default: true`), when installed.
//   3. q4 if installed (the catalog's smallest/default convention).
//   4. the first installed tier.
// Returns null when nothing is installed (no picker will render anyway). `options` (sc-9300) forwards
// the `convRotEligible` gate so a hidden INT8-ConvRot tier is never seeded as the selection.
export function defaultTierSelection(model, lastUsed, options = {}) {
  const tiers = installedTiers(model, options);
  if (tiers.length === 0) {
    return null;
  }
  if (lastUsed && tiers.includes(lastUsed)) {
    return lastUsed;
  }
  const declared = defaultInstalledTier(model, tiers);
  if (declared) {
    return declared;
  }
  if (tiers.includes("q4")) {
    return "q4";
  }
  return tiers[0];
}
