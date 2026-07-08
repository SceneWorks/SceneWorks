// Resolve which tier of a model's catalog entry is its LoRA-TRAINING base (sc-8966). Pure +
// unit-testable, deliberately separate from TrainingStudio.jsx (mirrors tierSuggestion.js /
// quantTier.js) so the readiness gate + install offer share one source of truth.
//
// A quant-matrix base can ship MLX *inference* tiers (q4/q8/bf16) AND, independently, a base used for
// LoRA training. The training base is NOT always the same artifact a user installed for generation:
//   1. A dedicated flat-diffusers `training` variant (lens, sc-8797 — SceneWorks/Lens) — the LoRA
//      trainer loads THIS, not the packed MLX inference tiers. It tracks its own per-variant install
//      state (sc-8508), so a user can have lens q4 installed for gen while the training base is absent.
//   2. Otherwise the DENSE `bf16` tier (Krea 2 Raw quant-matrix re-host, epic 9992): generation may
//      have installed only the q8 default, but training needs full-precision weights.
//   3. Otherwise undefined — a non-matrix base (z-image / sdxl) trains on its single default tier.
export function trainingBaseTier(base) {
  const variants = base?.variants;
  if (!Array.isArray(variants)) {
    return undefined;
  }
  // `training` wins over `bf16`: a base with both (lens) trains on the flat-diffusers training base,
  // never the bf16 inference tier.
  if (variants.some((variant) => variant?.variant === "training")) {
    return "training";
  }
  if (variants.some((variant) => variant?.variant === "bf16")) {
    return "bf16";
  }
  return undefined;
}

// The install state of the tier LoRA training actually needs. For a matrix base whose training tier is
// a `training`/`bf16` variant, this is that tier's OWN per-variant `installState` (sc-8508) — so lens
// with q4 installed but the training base missing correctly reads "missing" rather than the top-level
// default-tier state (the sc-8966 bug: the Training Studio said "ready", then the submit failed
// server-side because `SceneWorks/Lens` wasn't cached). A non-matrix base falls back to the model-level
// `installState`.
export function trainingBaseState(base) {
  const tier = trainingBaseTier(base);
  if (tier) {
    return (
      base?.variants?.find((variant) => variant?.variant === tier)?.installState ?? "missing"
    );
  }
  return base?.installState;
}
