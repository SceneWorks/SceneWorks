// RAM-based quant-tier download suggestion (sc-8509, epic 8506). The Models page lets a user pick
// which quant tier(s) of a model to DOWNLOAD, with a suggested default: the highest-fidelity tier
// that should fit the host's memory. This module is the pure, unit-testable core of that logic —
// deliberately separate from the React screen (like sc-8515's quantTier.js) so sc-8516 can later
// refine the thresholds/constants in ONE place once measured footprints land.
//
// SUGGEST, NEVER WITHHOLD (epic 8506 decision 1): every tier stays installable regardless of RAM.
// The suggestion only preselects/highlights the recommended tier; it never removes a tier from the
// installable set. `suggestTier` returning a smaller tier does not make bf16 un-installable — the
// screen keeps every tier's checkbox enabled.
//
// Consumes the sc-8508 per-variant catalog shape: each `model.variants[]` entry carries a `variant`
// key (bf16/q8/q4) and a `footprint` object. `footprint.diskSizeBytes` is always populated;
// `footprint.residentMemoryBytes` / `peakMemoryBytes` are the MEASURED memory fields sc-8516 populates
// (for the tiers it measured on-device; the rest stay null and fall back to the calibrated estimate).
//
// sc-8516 CALIBRATION (basis for the constants below): on-device measurement of steady-state resident
// + load+gen peak GPU memory (harness crates/sceneworks-worker/src/footprint_measure.rs — ONE tier per
// fresh process, resident sampled post-gen AFTER releasing the transient cache, peak = load+gen
// high-water — using the MLX counters mlx_rs::memory::{get_active_memory, get_peak_memory} the worker
// already publishes). Measured set: sdxl/q8, z_image/q4, z_image_turbo/q4, lens_turbo/q4. Two findings:
//   1. resident ≈ on-disk size (ratio 0.81–1.01, mean 0.94) — the old disk×1.5 resident estimate was
//      too high; packed weights sit resident at roughly their on-disk size.
//   2. peak − resident is a FIXED 14.04 GiB transient (1024² VAE decode + attention working set),
//      measured to within ~4 MiB across a 3.9→16.5 GiB resident spread over 3 different VAEs. It is
//      genuinely resolution-bound, NOT weight-bound (a real physical property of the 1024² decode, not
//      a "resident + constant" measurement artifact) — so peak = resident + a fixed addend, not ×N.
// The suggestion therefore budgets against PEAK (the real ceiling a generation must fit).

import { installedTiers, tierQuantize } from "./quantTier.js";

// Fidelity order, HIGHEST first. The suggestion walks this list and picks the first tier that both
// exists on the model AND fits memory; bf16 is preferred, then q8, then q4 (the smallest, always-fits
// fallback). Any declared tier not in this list is considered lowest-fidelity and only chosen last.
const TIER_FIDELITY = ["bf16", "q8", "q4"];

// Resident-memory estimate multiplier over on-disk size, used ONLY when a variant lacks any measured
// footprint. CALIBRATED by sc-8516 from on-device measurement (harness: crates/sceneworks-worker/src/
// footprint_measure.rs). Across sdxl-q8 / z-image-q4 / z-image-turbo-q4 / lens-turbo-q4 the measured
// steady-state RESIDENT/disk ratio was 0.81–1.01 (mean 0.94): packed weights sit resident at roughly
// their on-disk size, NOT the old 1.5× guess. We estimate resident ≈ disk × 1.0, then add the fixed
// transient below (which is where the real generation headroom lives).
export const DISK_TO_RESIDENT_MULTIPLIER = 1.0;

// Fixed transient working-set (bytes) a single 1024² generation needs ON TOP OF the resident weights —
// VAE decode buffers + attention activations/latents + framework scratch. sc-8516 measured this as the
// PEAK − RESIDENT gap and found it 14.04 GiB to within ~4 MiB across a 3.9→16.5 GiB resident spread
// (sdxl / z-image / lens, 3 different VAEs) — genuinely size-INDEPENDENT and resolution-bound, so it is
// modeled as a fixed addend, not a multiplier. 14 GiB ≈ the measured value, used directly. This is what
// makes the estimated budget (disk×MULT + transient) track the MEASURED peak the RAM suggestion must
// actually fit — the true install-time/run ceiling.
export const TRANSIENT_HEADROOM_BYTES = 14 * 1024 * 1024 * 1024;

// Fraction of detected unified/GPU memory a tier's peak footprint must fit UNDER to be suggested. The
// remainder is left for the OS, other apps, and margin. sc-8516 raised this from 0.8 → 0.9: because the
// budget is now peak-inclusive (weights + the fixed transient) rather than resident-only, the extra
// slack the old 0.8 baked in for un-modeled transient is already accounted for explicitly, and 0.9
// keeps the suggestion from needlessly under-picking on right-sized hardware. So on a 32 GB Mac a tier
// "fits" only if its peak is under 32 × 0.9 = 28.8 GB.
export const MEMORY_HEADROOM_FRACTION = 0.9;

const BYTES_PER_GB = 1024 * 1024 * 1024;

// A variant's PEAK memory requirement in bytes (the ceiling the suggestion must fit), or null when it
// can't be estimated. Basis, in priority order:
//   1. `footprint.peakMemoryBytes` — the MEASURED load+gen high-water mark, used verbatim (sc-8516).
//      This is the true ceiling: a tier whose peak OOMs during generation must not be suggested.
//   2. `footprint.residentMemoryBytes` + TRANSIENT_HEADROOM_BYTES — measured resident weights plus the
//      fixed transient working set, when resident was measured but peak was not.
//   3. `footprint.diskSizeBytes` × DISK_TO_RESIDENT_MULTIPLIER + TRANSIENT_HEADROOM_BYTES — the
//      estimate (the common case for un-measured tiers): weights ≈ on-disk size, plus the transient.
//   4. `downloadSizeBytes` × DISK_TO_RESIDENT_MULTIPLIER + TRANSIENT_HEADROOM_BYTES — last-resort when
//      the footprint object is absent but the catalog still knows the tier's download size.
// `measured` reports whether the value came from a measured field (peak or resident) vs the estimate.
export function variantFootprintBytes(variant) {
  const footprint = variant?.footprint;
  const peak = numberOrNull(footprint?.peakMemoryBytes);
  if (peak !== null && peak > 0) {
    return { bytes: peak, measured: true };
  }
  const resident = numberOrNull(footprint?.residentMemoryBytes);
  if (resident !== null && resident > 0) {
    return { bytes: resident + TRANSIENT_HEADROOM_BYTES, measured: true };
  }
  const disk = numberOrNull(footprint?.diskSizeBytes);
  if (disk !== null && disk > 0) {
    return {
      bytes: Math.round(disk * DISK_TO_RESIDENT_MULTIPLIER) + TRANSIENT_HEADROOM_BYTES,
      measured: false,
    };
  }
  const download = numberOrNull(variant?.downloadSizeBytes);
  if (download !== null && download > 0) {
    return {
      bytes: Math.round(download * DISK_TO_RESIDENT_MULTIPLIER) + TRANSIENT_HEADROOM_BYTES,
      measured: false,
    };
  }
  return null;
}

function numberOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

// ---------------------------------------------------------------------------------------------------
// Per-tier MEASURED memory floor (sc-15400), PER BACKEND
//
// `mlx.minMemoryGb` / `candle.minMemoryGb` is a SINGLE per-model integer, so it must admit the model's
// heaviest INSTALLABLE tier. That over-states the requirement for every user who only installed a
// lighter one: krea_realtime_14b is forced to 64 while its default q4 tier peaks at 27.90 GiB
// (`downloads[].footprint.peakMemoryBytes` 29,957,689,344 — with q8 at 34.44 and bf16 at 46.71 GiB).
// The manifest already carries the real per-tier ladder, so a consumer that knows WHICH tier is in play
// should read that tier's measured peak and use the blanket integer only as a fallback.
//
// THE TWO LANES CARRY INDEPENDENT EVIDENCE AND MUST NEVER BE CROSSED (sc-15613). The per-download
// `footprint` block is an MLX-ONLY measurement: the harness that produces it is
// `crates/sceneworks-worker/src/footprint_measure.rs`, sampling `mlx_rs::memory::get_peak_memory` — an
// Apple unified-memory high-water mark. A discrete CUDA card does not share system RAM, so that number
// does not describe the VRAM it must hold, and the schema says so outright at the `candle` block
// (model-manifest.schema.json): "the mlx.minMemoryGb ... does not describe the VRAM a discrete GPU must
// hold". The candle lane's own per-tier evidence is `candle.vramGbByTier` (populated on 33 catalog
// entries), with `candle.minMemoryGb` as its blanket.
//
// Crossing them is NOT merely imprecise, it can UNDER-state, which is the one direction that OOMs:
// `lens_turbo` measures 30.50 GiB on MLX but needs 37.3 GB of VRAM at the same q4 tier, and `qwen_image`
// declares `mlx.minMemoryGb` 50 against a `candle.minMemoryGb` of 56. So there is no "the MLX number is
// at least conservative" shortcut — every helper below takes the lane explicitly and reads only that
// lane's fields. `options.backend === "candle"` selects the candle lane; anything else is MLX (the
// convention every existing caller already uses: `macGatingActive ? "mlx" : "candle"`).
//
// These helpers are MEASURED-ONLY on purpose. Unlike `variantFootprintBytes` — which falls back to a
// resident/disk ESTIMATE so a download suggestion always has something to RANK — a memory FLOOR must
// never be synthesized from a guess: a hand-calibrated blanket `minMemoryGb` is strictly better
// evidence than a modelled estimate, so callers fall back to it instead of to the estimate.
// ---------------------------------------------------------------------------------------------------

// A single variant's MEASURED peak memory in GB (base 1024³, the same unit `variantFootprintBytes`
// budgets in), or null when this tier has no measured `footprint.peakMemoryBytes`. MLX LANE ONLY — see
// the harness note above.
function measuredVariantPeakGb(variant) {
  const peak = numberOrNull(variant?.footprint?.peakMemoryBytes);
  return peak !== null && peak > 0 ? peak / BYTES_PER_GB : null;
}

// Declared quant tier → the MLX measured peak (GB) for that tier, for the real bits-based tiers only
// (bf16/q8/q4). The "default" pseudo-variant of a single-variant model and the non-quant "training" base
// are excluded, so neither can ever be mistaken for a generation tier's footprint.
//
// DUPLICATE tier keys resolve to the MAX measured peak rather than last-write-wins. The manifest
// legitimately declares several downloads under one `variant` key — `mage_flow` carries each of q4/q8/bf16
// THREE times (main weights, plus a text_encoder and a vae component), and `wan_2_2_t2v_14b` carries each
// tier twice (a macOS/MLX entry with footprints and a windows/linux candle entry without).
//
// Three upstream filters currently make the emitted keys unique — the component rows are `coRequisite:
// true` (`is_co_requisite_download`), the per-platform rows are dropped by `retain_downloads_for_os`, and
// `model_variant_downloads` then de-dupes first-wins (sc-8508) — so no duplicate reaches this Map today.
// That is verified against the real catalog by memoryFloorCatalogParity.test.js rather than assumed here.
// A Map keyed blindly on `variant.variant` would nonetheless depend silently on all three: the LAST
// duplicate is a component row with no `footprint`, so last-write-wins would erase a real measurement and
// quietly fall back to the blanket. Taking the max over the duplicates that DO carry a measurement agrees
// with the first-wins projection today and errs toward over-stating if any of those filters ever change.
function mlxPeakGbByTier(model) {
  const byTier = new Map();
  for (const variant of model?.variants ?? []) {
    const tier = variant?.variant;
    if (tierQuantize(tier) === null) {
      continue;
    }
    const gb = measuredVariantPeakGb(variant);
    const prior = byTier.has(tier) ? byTier.get(tier) : null;
    if (gb === null) {
      // A duplicate with no footprint is a component subdir, not evidence that the tier is unmeasured.
      if (!byTier.has(tier)) {
        byTier.set(tier, null);
      }
      continue;
    }
    byTier.set(tier, prior === null ? gb : Math.max(prior, gb));
  }
  return byTier;
}

// Declared quant tier → the candle/CUDA measured peak VRAM (GB) for that tier, from
// `candle.vramGbByTier`. Keyed by the same q4/q8/bf16 tier vocabulary, so the two lanes are drop-in
// alternatives to each other; non-tier keys are dropped for the same reason as above.
function candlePeakGbByTier(model) {
  const byTier = new Map();
  const declared = model?.candle?.vramGbByTier;
  if (!declared || typeof declared !== "object") {
    return byTier;
  }
  for (const [tier, value] of Object.entries(declared)) {
    if (tierQuantize(tier) === null) {
      continue;
    }
    const gb = numberOrNull(value);
    byTier.set(tier, gb !== null && gb > 0 ? gb : null);
  }
  return byTier;
}

// Declared quant tier → that tier's measured peak (GB) ON `backend`. The single seam where the lane is
// chosen; everything below reads through it so no consumer can accidentally mix the two.
function peakGbByTier(model, backend) {
  return backend === "candle" ? candlePeakGbByTier(model) : mlxPeakGbByTier(model);
}

// The model's BLANKET declared floor (GB) on `backend`, or null when that lane declares none. The
// fallback for every helper below — and deliberately NOT cross-lane: a candle host with no candle
// evidence gets null (the caller shows nothing) rather than the MLX integer, because `qwen_image`'s
// mlx 50 / candle 56 proves the MLX blanket can sit BELOW the candle requirement.
export function blanketFloorGb(model, backend) {
  const block = backend === "candle" ? model?.candle : model?.mlx;
  const gb = numberOrNull(block?.minMemoryGb);
  return gb !== null && gb > 0 ? gb : null;
}

// The host size (GB) at which a `peakGb` requirement satisfies the app's OWN fit criterion — the same
// `peak <= host * MEMORY_HEADROOM_FRACTION` budget `tierFits` applies. This is what a USER-FACING
// "needs N GB" must report: the measured peak is a RAW high-water mark, so quoting it directly claims a
// host size at which the app itself would not run the tier (a 31 GB host fails lens_turbo's 30.50 GiB
// q4, because 31 × 0.9 = 27.9). The blanket `minMemoryGb` integers already bake in this headroom — the
// manifest says so at krea_realtime_14b: "~47 GiB active peak. 64 covers that with OS/app headroom" —
// so converting here is what makes the measured basis and the blanket basis mean the same thing.
export function hostGbForPeakGb(peakGb) {
  if (peakGb == null || !Number.isFinite(peakGb) || peakGb <= 0) {
    return null;
  }
  return Math.ceil(peakGb / MEMORY_HEADROOM_FRACTION);
}

// The MEASURED memory ceiling (GB) on `options.backend` for the tier(s) `model` will actually run at, or
// null when no such measurement exists (caller falls back to the blanket floor).
//
// The ceiling is the MAX over the INSTALLED tiers, because a studio tier picker can switch among them at
// will — and only when EVERY installed tier is measured. One unmeasured installed tier means the set has
// no known bound, so we return null and keep the curated blanket floor. This never UNDER-states the
// requirement, which matters: for a memory gate an under-estimate hides a warning about a model that
// then OOMs, whereas an over-estimate is merely conservative.
//
// `options` also forwards `installedTiers`' host-eligibility gates (convRotEligible / nvfp4Eligible), so
// a tier this host cannot serve never contributes to the ceiling.
export function installedTierPeakGb(model, options = {}) {
  const byTier = peakGbByTier(model, options.backend);
  const tiers = installedTiers(model, options);
  if (tiers.length === 0) {
    return null;
  }
  let ceiling = null;
  for (const tier of tiers) {
    const gb = byTier.get(tier) ?? null;
    if (gb === null) {
      // An installed tier we have no measurement for — the set is unbounded, so defer to the blanket.
      return null;
    }
    if (ceiling === null || gb > ceiling) {
      ceiling = gb;
    }
  }
  return ceiling;
}

// The CHEAPEST measured tier peak (GB) on `options.backend` among the model's declared quant tiers, or
// null when none is measured. This answers "what does it take to run this model AT ALL", which is the
// only honest per-tier statement available for a model that is NOT installed yet: the catalog does not
// emit the manifest's per-download `default` flag (see the `variant.default` note in quantTier.js
// `defaultTierSelection`), so the client cannot know which tier a download will fetch — and it must not
// assume the lightest, because 3 of the 53 shipped matrix models declare a heavier default (krea_2_raw /
// krea_2_turbo default to q8, instantid_realvisxl to bf16).
//
// Callers MUST therefore present this as a FLOOR ("from N GB"), never as a specific tier's requirement.
export function cheapestDeclaredTierPeakGb(model, options = {}) {
  let cheapest = null;
  for (const gb of peakGbByTier(model, options.backend).values()) {
    if (gb !== null && (cheapest === null || gb < cheapest)) {
      cheapest = gb;
    }
  }
  return cheapest;
}

// The model's real quant tiers (bf16/q8/q4), in fidelity order (highest first). Excludes the
// single-variant "default" pseudo-tier and any unknown key. Every declared tier is included whether
// or not it's installed — the suggestion is about what to DOWNLOAD, so uninstalled tiers count.
export function declaredTiers(model) {
  if (!model?.hasVariantMatrix || !Array.isArray(model.variants)) {
    return [];
  }
  const keys = model.variants
    .map((variant) => variant?.variant)
    .filter((key) => tierQuantize(key) !== null);
  const unique = [...new Set(keys)];
  return unique.sort((a, b) => fidelityRank(a) - fidelityRank(b));
}

// Rank a tier by fidelity, HIGHEST first (bf16=0). Unknown tiers sort last.
function fidelityRank(tier) {
  const index = TIER_FIDELITY.indexOf(tier);
  return index === -1 ? TIER_FIDELITY.length : index;
}

// Whether a variant's PEAK footprint fits within `unifiedMemoryGb` with headroom. When the memory
// signal is unknown (null) OR the tier has no estimable footprint, we treat it as fitting — we never
// withhold or block a tier on missing data; the worst case is suggesting a heavier tier.
export function tierFits(variant, unifiedMemoryGb) {
  if (unifiedMemoryGb == null || !Number.isFinite(unifiedMemoryGb)) {
    return true;
  }
  const footprint = variantFootprintBytes(variant);
  if (footprint === null) {
    return true;
  }
  const budgetBytes = unifiedMemoryGb * BYTES_PER_GB * MEMORY_HEADROOM_FRACTION;
  return footprint.bytes <= budgetBytes;
}

// The suggested default download tier for `model` given the host's `unifiedMemoryGb` (GPU VRAM off
// Mac). Picks the HIGHEST-FIDELITY tier that fits memory with headroom; if none fits (a tiny host,
// or every tier over budget) it falls back to the SMALLEST declared tier so there's always a
// suggestion. Returns null only when the model has no quant matrix.
//
// This never affects installability — it just picks which tier to pre-select/highlight. A 32 GB host
// lands on q4, a 512 GB Studio on bf16, and either can override.
export function suggestTier(model, unifiedMemoryGb) {
  const tiers = declaredTiers(model);
  if (tiers.length === 0) {
    return null;
  }
  const byKey = new Map(
    (model.variants ?? [])
      .filter((variant) => tierQuantize(variant?.variant) !== null)
      .map((variant) => [variant.variant, variant]),
  );
  // `tiers` is already highest-fidelity first; the first that fits wins.
  for (const tier of tiers) {
    const variant = byKey.get(tier);
    if (variant && tierFits(variant, unifiedMemoryGb)) {
      return tier;
    }
  }
  // Nothing fit (every tier over budget) — suggest the smallest so we still preselect something.
  return tiers[tiers.length - 1];
}
