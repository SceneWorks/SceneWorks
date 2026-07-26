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

// Per-model geometry constraints for the free Width/Height override (sc-14058). Most models keep the
// global backend envelope (256–4096, no stride) so their controls are untouched. A NATIVE-RESOLUTION
// model — one that declares a required dimension stride via `limits.requiresDimensionsMultipleOf`
// (Mage-Flow = 16) — is instead bounded by the extremes of its own published resolution ladder (Mage:
// 512–2048) and stepped by that stride. Deriving the range from the advisory ladder keeps it per-model
// and declarative: no extra manifest key, and a model that widens its ladder widens its free range in
// lockstep. Returns { min, max, step }; `step <= 1` means "no stride" (the common case).
export function modelDimensionConstraints(model) {
  const step = Number(model?.limits?.requiresDimensionsMultipleOf);
  if (!Number.isInteger(step) || step <= 1) {
    return { min: MIN_IMAGE_DIMENSION, max: MAX_IMAGE_DIMENSION, step: 1 };
  }
  const sides = Array.isArray(model?.limits?.resolutions)
    ? model.limits.resolutions
        .flatMap((entry) => String(entry).split("x").map((value) => Number(value)))
        .filter((value) => Number.isFinite(value) && value > 0)
    : [];
  const min = sides.length ? Math.max(MIN_IMAGE_DIMENSION, Math.min(...sides)) : MIN_IMAGE_DIMENSION;
  const max = sides.length ? Math.min(MAX_IMAGE_DIMENSION, Math.max(...sides)) : MAX_IMAGE_DIMENSION;
  return { min, max, step };
}

function sideInRange(value, { min, max }) {
  return Number.isFinite(value) && value >= min && value <= max;
}

function sideOnStride(value, step) {
  return step <= 1 || (Number.isFinite(value) && value % step === 0);
}

// Effective dimensions PLUS the selected model's native-resolution constraints (sc-14058). Layers the
// per-model range + stride on top of the pure `resolveEffectiveDimensions` primitive (left unchanged so
// its existing callers/tests stay byte-identical). Returns the resolved width/height, the constraints
// used, and WHY it is invalid split into `outOfRange` / `offStride` so the Studio can surface a precise,
// user-facing error (never a raw requirement string). `invalid` is the OR of the two — the single flag
// the Generate gate reads.
export function evaluateModelDimensions({ model, resolution, widthOverride, heightOverride }) {
  const constraints = modelDimensionConstraints(model);
  const { width, height } = resolveEffectiveDimensions({ resolution, widthOverride, heightOverride });
  const outOfRange = !(sideInRange(width, constraints) && sideInRange(height, constraints));
  const offStride =
    constraints.step > 1 && !(sideOnStride(width, constraints.step) && sideOnStride(height, constraints.step));
  return { width, height, constraints, outOfRange, offStride, invalid: outOfRange || offStride };
}

// The user-facing error message for a broken free Width/Height override, or null when the dimensions are
// valid. Kept here (pure) so the message and the gate cannot drift. An ERROR, not a requirement: the two
// number boxes hold a value the user actively broke, and nothing else on the form explains the dead CTA.
export function dimensionConstraintMessage(evaluation) {
  if (!evaluation || !evaluation.invalid) {
    return null;
  }
  const { constraints, outOfRange, offStride } = evaluation;
  const range = `between ${constraints.min} and ${constraints.max}`;
  if (outOfRange && offStride) {
    return `Width and height must be ${range} and a multiple of ${constraints.step}.`;
  }
  if (outOfRange) {
    return `Width and height must be ${range}.`;
  }
  return `Width and height must be a multiple of ${constraints.step} — this model renders at native resolution.`;
}
