// PiD high-res decode heads-up (sc-10144, epic 7840). PiD super-resolves the base render by a FIXED
// 4× (worker `PID_SR_SCALE`, crates/sceneworks-worker/src/image_jobs/pid.rs), so the output image is
// always `base × 4`. At the default 4K tier the base is UNTOUCHED, so a large base means a
// multi-minute — and, above a 4096² output, auto-tiled (sc-10087) — super-res decode that can look
// hung when it is really just working. The 2K tier caps the base long side to 512 → ~2048² output,
// always fast, so it never warns. This pure helper decides when the Studio surfaces a proactive
// heads-up so the user knows a long decode is progressing, not stuck.

// PiD's fixed spatial super-resolution factor (mirrors the worker `PID_SR_SCALE`). Output = base × this.
export const PID_SR_SCALE = 4;

// Output long side (px) at/above which the decode runs multiple minutes and warrants the heads-up.
// Grounded in the sc-10087 timing evidence: a 4096² whole-image decode ≈ 3.7 min on Metal, a 6144²
// (tiled) decode ≈ 4.8 min, and smaller GPUs are slower. Below this (a ≤768 base → ≤3072² output) the
// decode is short enough that a heads-up would just be noise. 4096² == a 1024 base at the 4K tier.
export const PID_HEADS_UP_MIN_OUTPUT_LONG_SIDE = 4096;

// Output long side above which the pixel-space decode auto-tiles (sc-10087) instead of running a single
// whole-image forward. A 4096² output is the largest whole-image decode; anything larger tiles.
export const PID_TILING_OUTPUT_LONG_SIDE = 4096;

// Decide whether a PiD generation warrants a "this will take a while" heads-up, and describe it.
// Returns null when no heads-up is needed (PiD off, the fast 2K tier, invalid dims, or a small output),
// otherwise an object with the resolved output size, megapixels, whether it tiles, and a ready-to-render
// `message`. Pure + unit-tested so both Studio surfaces (Image Studio + Character) stay in lock-step.
export function pidDecodeHeadsUp({ usePid, pidTarget, width, height }) {
  if (!usePid) {
    return null;
  }
  // The 2K tier caps the base long side to 512 → ~2048² output, which is always fast: no heads-up.
  if (String(pidTarget ?? "").trim().toLowerCase() === "2k") {
    return null;
  }
  const baseWidth = Number(width);
  const baseHeight = Number(height);
  if (!Number.isFinite(baseWidth) || !Number.isFinite(baseHeight) || baseWidth <= 0 || baseHeight <= 0) {
    return null;
  }
  const outputWidth = baseWidth * PID_SR_SCALE;
  const outputHeight = baseHeight * PID_SR_SCALE;
  const longSide = Math.max(outputWidth, outputHeight);
  if (longSide < PID_HEADS_UP_MIN_OUTPUT_LONG_SIDE) {
    return null;
  }
  const megapixels = (outputWidth * outputHeight) / 1_000_000;
  const tiled = longSide > PID_TILING_OUTPUT_LONG_SIDE;
  return {
    outputWidth,
    outputHeight,
    megapixels,
    tiled,
    message: formatHeadsUp(outputWidth, outputHeight, megapixels, tiled),
  };
}

function formatHeadsUp(outputWidth, outputHeight, megapixels, tiled) {
  const size = `${outputWidth}×${outputHeight}`;
  const mp = `~${megapixels.toFixed(1)} MP`;
  return tiled
    ? `PiD super-resolves to ${size} (${mp}), decoded in tiles — this can take several minutes. It's working, not stuck.`
    : `PiD super-resolves to ${size} (${mp}) — this decode can take a few minutes. It's working, not stuck.`;
}
