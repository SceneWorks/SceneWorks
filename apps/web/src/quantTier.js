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
//
// NOTE this map is BITS-VALUED, and that is why neither int8-convrot NOR nvfp4 appears in it: the
// worker parses `mlxQuantize` as an integer that NAMES a tier (`<= 0` ⇒ bf16, `<= 4` ⇒ q4, else q8),
// so a tier with no honest integer must not ride this key. `nvfp4` sending `4` here would select the
// int4-affine q4 tier — the exact aliasing epic 11037 SC#5 forbids. See NVFP4_TIER below.
const TIER_QUANTIZE = {
  bf16: 0,
  q8: 8,
  q4: 4,
};

// The three user-facing generation-quality tiers, in fidelity order — the vocabulary the global
// "default generation quality" setting (epic 10721 / sc-10728) ranges over. int8-convrot and nvfp4 are
// intentionally excluded: both are candle-only niche tiers, not sensible app-wide defaults. Excluding
// nvfp4 is also an epic 11037 SC#5 requirement — a tier reachable only by an explicit, per-generation
// pick can never become the silent default for anyone.
export const GENERATION_QUALITY_TIERS = ["bf16", "q8", "q4"];

// The app-wide base default generation tier used when no global setting, sticky, or manifest default
// applies (epic 10721 / sc-10726). Q8 matches the worker's generation default. `defaultTierSelection`
// uses it as the fallback base whenever `options.defaultQuality` is absent or invalid, so every legacy
// call site (and the worker) stays consistent on Q8.
export const DEFAULT_GENERATION_QUALITY = "q8";

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

// The candle-only NVFP4 tier key (sc-11042, epic 11037). NVFP4 = E2M1 4-bit elements over 16-element
// blocks + FP8-E4M3 micro-scales + an FP32 per-tensor scale (~4.5 effective bits/weight), served by the
// cuBLASLt FP4 GEMM on consumer Blackwell (sm_120).
//
// A DISTINCT, user-selectable tier — the sc-11042 "Option A" packaging decision. It is deliberately NOT
// a Blackwell execution backend auto-substituted for q4: NVFP4's numerics differ from int4-affine q4, so
// auto-selecting it on a Blackwell host would silently change what a picked q4 tier renders — the
// creative-choice violation epic 11037 SC#5 forbids. Picking NVFP4 is always an explicit user action.
//
// Like INT8_CONVROT_TIER, it is NOT a bits-based quant, so it is DELIBERATELY absent from TIER_QUANTIZE
// (`tierQuantize` returns null for it, so `mlxQuantize` stays out of the payload) and instead sends a
// distinct `advanced.quantTier: "nvfp4"` signal (see imageJobAdvanced.js) that the worker's tier-select
// reads.
export const NVFP4_TIER = "nvfp4";

// Whether a tier key names the candle NVFP4 tier.
export function isNvfp4Tier(tier) {
  return tier === NVFP4_TIER;
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
  // Candle-only (Windows/Linux, Blackwell sm_120+). FP4 tensor-core tier (sc-11042). Named distinctly
  // from Q4 on purpose — it is a different numeric regime, not a faster q4.
  [NVFP4_TIER]: "NVFP4 (candle, Blackwell sm_120+)",
};

// Display order (smallest → largest); tiers not in this list sort after, alphabetically. int8-convrot
// sits between q4 and q8 by footprint/fidelity (int8 DiT, PSNR 34.4 dB — better than Q4's 22.7 dB).
// nvfp4 sits just above q4: ~4.5 effective bits/weight vs q4's 4, with weight accuracy measured at
// roughly the int4 tier's (spike sc-11038: rel-RMS ~0.094) — so it is the nearest neighbour above q4
// and below the int8 tier by footprint.
const TIER_ORDER = ["q4", NVFP4_TIER, INT8_CONVROT_TIER, "q8", "bf16"];

// Map a tier key to its `advanced.mlxQuantize` value, or null when the key isn't a known quant
// tier (e.g. "default" on a single-variant model — such models never render the picker).
export function tierQuantize(tier) {
  return Object.prototype.hasOwnProperty.call(TIER_QUANTIZE, tier)
    ? TIER_QUANTIZE[tier]
    : null;
}

// Inverse of `tierQuantize`: the tier a recorded `advanced.mlxQuantize` was produced from, or null
// when the value matches no known tier (absent, or a non-bits tier that never emits an mlxQuantize).
//
// Recipe replay (sc-12324) needs this to put a re-run back on the tier the clip was generated at.
// The tier is an aesthetic choice, not just a perf knob — landing a replay on the default tier
// would silently reproduce a different look.
export function quantizeTier(quantize) {
  if (typeof quantize !== "number") {
    return null;
  }
  // `bf16` maps to 0, so compare explicitly rather than leaning on truthiness.
  return Object.keys(TIER_QUANTIZE).find((tier) => TIER_QUANTIZE[tier] === quantize) ?? null;
}

export function tierLabel(tier) {
  return TIER_LABELS[tier] ?? tier;
}

// Whether a tier key is a user-selectable generation tier: a known bits-based quant (bf16/q8/q4) OR one
// of the candle-only non-bits tiers (INT8-ConvRot, NVFP4). Excludes the "default" pseudo-variant of a
// single-variant model and non-generation pseudo-tiers like "training". Distinct from `tierQuantize`
// (which returns null for both non-bits tiers because they have no mlxQuantize value — they still
// select, via `advanced.convRot` / `advanced.quantTier`).
export function isSelectableTier(tier) {
  return tierQuantize(tier) !== null || isConvRotTier(tier) || isNvfp4Tier(tier);
}

// The installed, selectable quant tiers of a model, in display order. A tier is selectable when it is
// a known quant tier (bf16/q8/q4) OR one of the candle-only tiers (INT8-ConvRot, NVFP4)
// (`isSelectableTier` — the "default" pseudo-variant of a single-variant model is excluded) AND its
// files are installed. Returns [] when the model has no variant matrix.
//
// `options.convRotEligible` (sc-9300, default true) gates the candle-only INT8-ConvRot tier: the
// caller passes `false` when NO live worker advertises the `int8_convrot` capability (macOS/MLX, or a
// pre-Ada NVIDIA GPU that fails the sm_89 compute-cap probe), so the tier is HIDDEN on an ineligible
// host even when its files happen to be present in the cache. Every other tier is unaffected. Default
// true keeps existing single-lane call sites (and tests) unchanged.
//
// `options.nvfp4Eligible` (sc-11042, default true) gates the candle-only NVFP4 tier identically: the
// caller passes `false` when NO live worker advertises the `nvfp4` capability (macOS/MLX, or an NVIDIA
// GPU below the sm_120 Blackwell compute-cap floor). Hiding an unservable tier is the FIRST of the two
// Blackwell gates; the worker re-checks the cap at tier-select (`nvfp4_host_eligible` in
// image_jobs/base.rs) so a hand-crafted API call that bypasses this picker still falls back cleanly.
// Sort tier keys by display order (smallest → largest); unknown keys sort after, alphabetically.
function sortByTierOrder(a, b) {
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
}

export function installedTiers(model, options = {}) {
  const { convRotEligible = true, nvfp4Eligible = true } = options;
  // Whether a tier survives the per-host capability gates: the candle-only tiers are hidden when no live
  // worker can serve them; every bits-based tier is unaffected.
  const hostEligible = (tier) =>
    (convRotEligible || !isConvRotTier(tier)) && (nvfp4Eligible || !isNvfp4Tier(tier));
  // Download-matrix models (sc-8508): per-tier DOWNLOAD entries, install-tracked individually.
  if (model?.hasVariantMatrix && Array.isArray(model.variants)) {
    return model.variants
      .filter(
        (variant) =>
          variant &&
          isSelectableTier(variant.variant) &&
          hostEligible(variant.variant) &&
          variant.installState === "installed",
      )
      .map((variant) => variant.variant)
      .sort(sortByTierOrder);
  }
  // Convert-at-install models (sc-10730): tiers are convert OUTPUTS on disk, surfaced by the catalog as
  // `mlxTiers` — a plain array of installed tier keys, DECOUPLED from the download variant-matrix so the
  // Models download panel (`hasVariantMatrix`) is untouched. Anima (and other convert-at-install models)
  // get a Studio generation-time tier picker this way.
  if (Array.isArray(model?.mlxTiers) && model.mlxTiers.length > 0) {
    return model.mlxTiers.filter((tier) => isSelectableTier(tier) && hostEligible(tier)).sort(sortByTierOrder);
  }
  return [];
}

// Quality rank of a generation quant tier: higher = more faithful (bf16 > q8 > q4). Derived from
// GENERATION_QUALITY_TIERS (highest-fidelity first), so it stays in lockstep with that vocabulary.
// Unknown / non-quality tiers (e.g. int8-convrot, nvfp4, "training") rank 0 so they never take part in a
// floor comparison — a below-floor advisory or default clamp only ever fires between two known quality
// tiers. For nvfp4 that is deliberate as well as incidental: it is a distinct numeric regime, not a rung
// on the bf16/q8/q4 fidelity ladder, so ranking it against a floor would be a category error.
function tierQualityRank(tier) {
  const i = GENERATION_QUALITY_TIERS.indexOf(tier);
  return i === -1 ? 0 : GENERATION_QUALITY_TIERS.length - i;
}

// Whether tier `a` is LOWER fidelity than tier `b`. Both must be known quality tiers (else false).
function isTierBelow(a, b) {
  const ra = tierQualityRank(a);
  const rb = tierQualityRank(b);
  return ra > 0 && rb > 0 && ra < rb;
}

// The model's per-model quality FLOOR (sc-10731, epic 10721): the backend surfaces the manifest
// `mlx.minQualityTier` as a top-level `minQualityTier`. Returns the floor tier (bf16|q8|q4) when declared
// and valid, else null (default absent = no floor, q4-tolerant). A floored model's DEFAULT tier is
// clamped UP to this (see `defaultTierSelection`); an EXPLICIT picker pick below it is honored + flagged.
export function modelQualityFloor(model) {
  return GENERATION_QUALITY_TIERS.includes(model?.minQualityTier) ? model.minQualityTier : null;
}

// Whether `tier` is BELOW the model's quality floor (sc-10731): true only when the model declares a floor
// AND `tier` is a lower-fidelity quality tier than it. Drives the Studio's non-blocking advisory when a
// user EXPLICITLY picks a below-floor tier — the pick is honored (a quant tier is a deliberate creative
// choice) but flagged as a quality caution, never silently switched.
export function isBelowFloor(tier, model) {
  const floor = modelQualityFloor(model);
  return !!floor && isTierBelow(tier, floor);
}

// Whether the studio should render the tier picker: only when MORE THAN ONE quant tier is
// installed (a single installed tier — the common case — shows no toggle, per acceptance).
// `options` forwards the `convRotEligible` (sc-9300) / `nvfp4Eligible` (sc-11042) gates so an
// ineligible host doesn't count a hidden candle-only tier toward the >1 threshold.
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

// The tier the picker should start on for `model`. Preference order (epic-locked, epic 10721):
//   1. `lastUsed` — the per-(screen,model) sticky tier, when it is still installed (rung 2, sc-10727).
//   2. the model's declared default tier (`variant.default: true`), when installed. (Dead against the
//      real catalog — the backend never emits `default` — but kept so a manifest that does still wins.)
//   3. the global "default generation quality" setting (rung 3, sc-10728) — `options.defaultQuality`,
//      one of bf16|q8|q4. Absent/invalid falls back to `DEFAULT_GENERATION_QUALITY` (q8), matching the
//      worker's generation default. Clamped to installed: the base leads a clean-tier fallback so an
//      uninstalled base resolves to the nearest installed clean tier rather than null. Convert-at-install
//      models (mlxTiers, sc-10730) fall through bf16 before q4; download-matrix models fall q8 → q4.
//   4. the first installed tier.
// Rungs 2–4 (the derived DEFAULT) are then CLAMPED UP to the model's per-model quality floor
// (`minQualityTier`, sc-10731): a floored model (Anima base/aesthetic = q8) never lets a low global
// setting / fallback land the default below the floor. The sticky (rung 1) is NOT floored — it was a
// prior explicit pick, honored as-is. The clamp is capped by clamp-to-installed (a floor tier not on
// disk resolves to the nearest installed clean tier).
// Returns null when nothing is installed (no picker will render anyway). `options` also forwards the
// `convRotEligible` (sc-9300) / `nvfp4Eligible` (sc-11042) gates so a hidden candle-only tier is never
// seeded as the selection — and because `defaultQuality` can only ever be bf16|q8|q4, it never
// re-introduces a filtered tier.
//
// NVFP4 can only ever be reached here by rung 1 (the sticky — i.e. a PRIOR EXPLICIT pick by this user
// on this model), never by rungs 2–4: `variant.default` is dead against the real catalog, `base` is
// drawn from GENERATION_QUALITY_TIERS (which excludes nvfp4), and `cleanFallback` lists only q8/bf16/q4.
// Rung 4 (`tiers[0]`) cannot reach it either — TIER_ORDER puts q4 first, and nvfp4 is only present when
// an nvfp4 tier is installed, which by itself implies the user chose to install it. That is epic 11037
// SC#5 holding at the UI layer: NVFP4 never becomes anyone's default by accident.
export function defaultTierSelection(model, lastUsed, options = {}) {
  const tiers = installedTiers(model, options);
  if (tiers.length === 0) {
    return null;
  }
  if (lastUsed && tiers.includes(lastUsed)) {
    return lastUsed;
  }
  // The per-model quality FLOOR (sc-10731): the DEFAULT derivation below (rungs 2–4) is clamped UP to
  // this tier. A floored model (Anima base/aesthetic = q8) never lets a declared default, a low global
  // "default quality" setting, or the fallback land the default below the floor. NOT applied to the
  // sticky (rung 1) above — that was a prior EXPLICIT pick, honored as-is (the picker re-surfaces the
  // advisory). Capped by clamp-to-installed (the clean-tier fallback), so a floor tier not on disk
  // resolves to the nearest installed clean tier, never null.
  const floor = modelQualityFloor(model);
  // Rung 2: the model's declared default tier (`variant.default`), when installed. Dead against the real
  // catalog (no `default` emitted), kept so a manifest that does declare one still wins — subject to the
  // floor below (a declared q4 on a q8-floored model still starts at q8).
  const declared = defaultInstalledTier(model, tiers);
  // Rung 3: the global default-generation-quality setting is the app-wide base default. The caller
  // passes it as `options.defaultQuality`; an absent/invalid value falls back to Q8 (the historical
  // base + worker default) so legacy call sites are unchanged. The base leads a clean-tier fallback so
  // it is always clamped to what's installed — a base tier that isn't on disk resolves to the nearest
  // installed clean tier (never the washed q4 unless that's all that's left), never null.
  let base =
    declared ??
    (GENERATION_QUALITY_TIERS.includes(options.defaultQuality)
      ? options.defaultQuality
      : DEFAULT_GENERATION_QUALITY);
  // Clamp the default UP to the model's quality floor (raises only — a base at/above the floor is
  // untouched; the below fallback still caps it at what's installed).
  if (floor && isTierBelow(base, floor)) {
    base = floor;
  }
  const cleanFallback =
    !model?.hasVariantMatrix && Array.isArray(model?.mlxTiers)
      ? ["q8", "bf16", "q4"]
      : ["q8", "q4"];
  const preferred = [base, ...cleanFallback.filter((tier) => tier !== base)];
  for (const tier of preferred) {
    if (tiers.includes(tier)) {
      return tier;
    }
  }
  return tiers[0];
}
