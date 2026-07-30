// Resolution memory-gating (sc-13959, epic 13879). Krea 2 renders up to 2048² (the pinned engine's
// RES_MAX), and the manifest now advertises the full ÷16 ladder to that ceiling. 2048² is ~4× the
// pixels of 1024², so its single-pass VAE-decode + attention transient is far larger than the ≤1536²
// set the model's `minMemoryGb` visibility floor was calibrated for — a 48 GB Mac that runs 1536²
// fine OOMs at 2048². This module is the pure, unit-testable gate the studios use to hide a high-res
// bucket a host can't fit, MIRRORING the quant-tier gate (tierSuggestion.js): budget an estimated
// PEAK against the host's unified/GPU memory with headroom, and NEVER withhold on missing data.
//
// SCOPE — three deliberate no-ops keep this from touching anything the story doesn't:
//   1. Buckets at/below the historical 1536² ceiling are ALWAYS offered. They shipped before this
//      change and the model's own `minMemoryGb` visibility floor already gates them, so ≤1536²
//      behavior is byte-identical.
//   2. Unknown host memory (`null`, the probe hasn't resolved / no signal) ⇒ everything fits. We never
//      withhold a resolution on missing data — the worst case is offering a heavy one, which the
//      worker's own load-time fit gate + runtime decode still backstop (mirrors tierFits).
//   3. A model that declares NO memory floor (no `mlx.minMemoryGb` / `candle.minMemoryGb` — e.g.
//      SenseNova, which already ships 2048² buckets) is left entirely unchanged: with no floor there
//      is no basis to predict a peak, so the gate returns "fits". Only a model that BOTH declares a
//      floor AND advertises >1536² buckets is affected — today that is exactly Krea 2 Raw / Turbo.
//
// PER-TIER BASIS (sc-15400). The blanket `minMemoryGb` is a single per-model integer that has to admit
// the model's heaviest INSTALLABLE tier, so using it here made this a TIER-BLIND gate: a host that only
// installed q4 was measured against the bf16 worst case and could have a high-res bucket hidden that its
// installed tier renders fine. `floorGb` now prefers the tier's MEASURED peak
// (`downloads[].footprint.peakMemoryBytes`) — the selected `tier` when the studio passes one, else the
// ceiling over the INSTALLED tiers — and falls back to the blanket integer when the tier in play has no
// measurement. Note this is currently INERT for Krea 2 Raw / Turbo: they are the only shipped models with
// >1536² buckets AND a floor, and none of their tiers carry a measured footprint yet, so they keep the
// blanket 48/32 until epic 15448 fills those in. The preference is what makes filling them in take effect.

import { MEMORY_HEADROOM_FRACTION, installedTierPeakGb } from "./tierSuggestion.js";

// The historical resolution ceiling (pixels). Every bucket at/below this was shipped before sc-13959
// and stays unconditionally offered (SCOPE note 1). 1536×1536 = 2,359,296 px ≈ 2.36 MP.
export const BASELINE_PIXELS = 1536 * 1536;

// Transient working-set (GB) a single generation needs PER MEGAPIXEL of output ABOVE the baseline, on
// TOP of the model's `minMemoryGb` floor (which already covers the resident weights + the ≤1536²
// transient for the default tier). Anchored on the app's measured single-pass VAE-decode + attention
// transient: sc-8516 / mlx_fit_gate `HEADROOM_GB` measured ~14 GiB at 1024² (≈1.05 MP) ⇒ ~13 GB/MP.
// The plain text-to-image GENERATE lane still decodes SINGLE-PASS in the pinned Krea engine (the
// budget-gated tiled decode, sc-11747, is wired only on the control lane), so the gate budgets against
// that single-pass peak — deliberately conservative: over-estimating hides a borderline size (safe),
// under-estimating would offer one that OOMs. When generate-lane tiled decode lands upstream this
// coefficient can be relaxed toward the ~7 GB/MP tiled figure (qwen-image's tiled-VAE transient).
export const HIGHRES_TRANSIENT_GB_PER_MP = 13;

// Parse a "WxH" bucket to its pixel count, or null when malformed.
function pixelsOf(resolution) {
  if (typeof resolution !== "string") {
    return null;
  }
  const [width, height] = resolution.split("x").map((value) => Number(value));
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }
  return width * height;
}

// The model's memory floor (GB) for the tier in play on the active backend, or null when there is no
// basis to predict one.
//
// PREFERS the per-tier MEASURED peak (`downloads[].footprint.peakMemoryBytes`, sc-15400) for the tier
// this host will actually run — `tier` when the studio has an explicit pick, else the ceiling over the
// INSTALLED tiers. The blanket `mlx`/`candle.minMemoryGb` is the FALLBACK: it is a single per-model
// integer that must admit the heaviest INSTALLABLE tier, so it over-states the gate for a host that only
// installed a lighter one, and this module used it as a tier-blind visibility gate — hiding a high-res
// bucket a lighter installed tier would have rendered fine.
//
// Both bases are a PEAK compared under `MEMORY_HEADROOM_FRACTION`, so they are interchangeable here: the
// measured peak is a raw MLX-active high-water mark and the 0.9 fraction supplies the OS/app headroom,
// exactly as `tierSuggestion.tierFits` budgets it. (The blanket integers additionally bake in some of
// their own headroom, which only makes the fallback the more conservative of the two.)
function floorGb(model, backend, tier) {
  const measured = installedTierPeakGb(model, tier ? { tier } : {});
  if (measured !== null) {
    return measured;
  }
  const block = backend === "candle" ? model?.candle : model?.mlx;
  const gb = block?.minMemoryGb;
  return typeof gb === "number" && Number.isFinite(gb) && gb > 0 ? gb : null;
}

// Predicted PEAK memory (GB) a generation at `resolution` needs for `model` on `backend`, or null when
// it can't be predicted (malformed resolution, or the model declares no memory floor). At/below the
// baseline this is just the floor (already the calibrated ≤1536² peak); above it, the floor plus the
// per-megapixel transient for the extra pixels.
//
// `tier` (optional) is the quant tier the studio has selected, so the floor comes from THAT tier's
// measured footprint rather than the model's blanket worst-tier integer (sc-15400).
export function predictedResolutionPeakGb(model, resolution, backend, tier = null) {
  const pixels = pixelsOf(resolution);
  const floor = floorGb(model, backend, tier);
  if (pixels == null || floor == null) {
    return null;
  }
  if (pixels <= BASELINE_PIXELS) {
    return floor;
  }
  const extraMegapixels = (pixels - BASELINE_PIXELS) / 1_000_000;
  return floor + HIGHRES_TRANSIENT_GB_PER_MP * extraMegapixels;
}

// Whether `resolution` should be OFFERED for `model` on a host with `unifiedMemoryGb` of unified/GPU
// memory. See the SCOPE notes at the top for the three always-fit cases. Otherwise the predicted peak
// must fit under the same headroom fraction the quant-tier gate uses (0.9), leaving the remainder for
// the OS + other apps.
export function resolutionFitsMemory(model, resolution, unifiedMemoryGb, options = {}) {
  const { backend, tier = null } = options;
  const pixels = pixelsOf(resolution);
  // (1) The historical ≤1536² set is always offered.
  if (pixels == null || pixels <= BASELINE_PIXELS) {
    return true;
  }
  // (2) Never withhold on missing data.
  if (unifiedMemoryGb == null || !Number.isFinite(unifiedMemoryGb)) {
    return true;
  }
  const required = predictedResolutionPeakGb(model, resolution, backend, tier);
  // (3) No declared floor ⇒ no prediction ⇒ leave the model's buckets unchanged.
  if (required == null) {
    return true;
  }
  return required <= unifiedMemoryGb * MEMORY_HEADROOM_FRACTION;
}

// Filter a list of "WxH" buckets to those that fit `unifiedMemoryGb`, preserving order. The studio's
// effective resolution options: identical to the input for every ≤1536²-only model and for an unknown
// memory reading; trims only the over-budget high-res buckets of a floored model on a known-small host.
export function fitsResolutionOptions(model, resolutions, unifiedMemoryGb, options = {}) {
  if (!Array.isArray(resolutions)) {
    return [];
  }
  return resolutions.filter((resolution) =>
    resolutionFitsMemory(model, resolution, unifiedMemoryGb, options),
  );
}
