// Advanced resolution override (Image Studio): a custom Width/Height lets the user
// experiment beyond a model's pre-declared Aspect options (e.g. Krea 2 up to 4K). A
// non-empty override wins for that axis; an empty ("") field falls back to the Aspect
// dropdown value, mirroring the Steps/Guidance overrides. Kept pure + unit-testable so
// the submit path stays focused on orchestration.

// Backend cap: rust-api validate_dimension enforces 256–4096 per side (lib.rs).
export const MIN_IMAGE_DIMENSION = 256;
export const MAX_IMAGE_DIMENSION = 4096;

// Resolve the effective width/height sent to the worker from the Aspect dropdown string
// ("1024x1024") and the optional per-axis overrides. Returns { width, height, invalid },
// where `invalid` is true when either effective side falls outside the backend range.
export function resolveEffectiveDimensions({ resolution, widthOverride, heightOverride }) {
  const [dropdownWidth, dropdownHeight] = String(resolution ?? "")
    .split("x")
    .map((value) => Number(value));
  const width = widthOverride !== "" && widthOverride != null ? Number(widthOverride) : dropdownWidth;
  const height = heightOverride !== "" && heightOverride != null ? Number(heightOverride) : dropdownHeight;
  const invalid = !(
    inRange(width) && inRange(height)
  );
  return { width, height, invalid };
}

function inRange(value) {
  return Number.isFinite(value) && value >= MIN_IMAGE_DIMENSION && value <= MAX_IMAGE_DIMENSION;
}
