// Face-likeness band constants + classifier — the SINGLE SOURCE OF TRUTH for the
// identity-likeness UI (epic 4406). The frontend badge (sc-4413) imports `classifyLikeness`
// and `LIKENESS_BANDS` from here; do NOT duplicate the cut-points anywhere else.
//
// The backend scorer (crates/sceneworks-worker/src/face_likeness.rs) writes a per-asset
// `recipe.rawAdapterSettings.faceLikeness = { score, detected, method, sourceAssetId, reason? }`
// where `score` is an ArcFace antelopev2 cosine in [-1, 1] (effectively [0, 1] for face pairs).
// This module turns that block into one of four UI states.
//
// The metric, the frontal-only limitation, the N/A behaviour, the band rationale, and the
// antelopev2 non-commercial licensing note are all documented in docs/face-likeness.md.

// The recognition method string the backend stamps on every faceLikeness block
// (`LIKENESS_METHOD` in face_likeness.rs). Kept here so the frontend can assert/label the
// metric without hard-coding the literal at the call site.
export const LIKENESS_METHOD = "arcface_antelopev2";

// The band labels. `na` is NOT a likeness tier — it is the explicit "no detectable frontal
// face" outcome (detected:false, score:null), surfaced so the badge never renders a missing
// score as a low quality number. See docs/face-likeness.md § "Why profiles return N/A".
export const LIKENESS_BAND = Object.freeze({
  STRONG: "strong",
  MODERATE: "moderate",
  WEAK: "weak",
  NA: "na",
});

// The calibrated cosine cut-points (epic 4406, sc-4414). Anchored to recorded antelopev2
// ArcFace baselines — NOT invented numbers. Full rationale in docs/face-likeness.md.
//
//   strong   : score >= STRONG_MIN            (solid frontal identity)
//   moderate : MODERATE_MIN <= score < STRONG_MIN  (recognizable, with drift)
//   weak     : score < MODERATE_MIN           (clear drift / wrong person)
//
// STRONG_MIN = 0.80 — the floor that captures every solidly-recognizable anchor: InstantID
//   frontal ~0.876 and its 9-angle pack 0.81–0.89, PuLID-FLUX photoreal 0.8016, and the upper
//   end of the InstantID face-restore range (~0.83). It deliberately excludes the PuLID
//   id_weight=0.8 case (0.7422, which "visually drifts") — that should read as moderate, not
//   strong.
// MODERATE_MIN = 0.55 — the floor below which results are clear drift. It sits just below the
//   closest anchor above it, PuLID id_weight=0.6 (0.5689, "drifts further") — that borderline
//   case stays moderate — while putting the soft-identity backbones (FLUX.2-klein angle-set mean
//   ~0.52) and the InstantID landmark-disabled collapse (~0.15) into weak. Everything in
//   [0.55, 0.80) — the pose tier (~0.71), the lower face-restore edge (~0.74), Qwen-Edit
//   Lightning angle mean (~0.62) — is "recognizable but drifting", i.e. moderate.
export const STRONG_MIN = 0.8;
export const MODERATE_MIN = 0.55;

// Ordered descriptors for the bands (the frontend badge maps these to label/colour/tooltip).
// The order is highest-to-lowest confidence; `na` is appended as the non-tier outcome.
export const LIKENESS_BANDS = Object.freeze([
  {
    band: LIKENESS_BAND.STRONG,
    label: "Strong likeness",
    min: STRONG_MIN,
    max: 1,
    description: "Frontal identity solidly matches the reference.",
  },
  {
    band: LIKENESS_BAND.MODERATE,
    label: "Moderate likeness",
    min: MODERATE_MIN,
    max: STRONG_MIN,
    description: "Recognizable as the same person, with some identity drift.",
  },
  {
    band: LIKENESS_BAND.WEAK,
    label: "Weak likeness",
    min: -1,
    max: MODERATE_MIN,
    description: "Clear identity drift — likely not a confident match.",
  },
  {
    band: LIKENESS_BAND.NA,
    label: "No frontal face",
    min: null,
    max: null,
    description:
      "No detectable frontal face to score (e.g. a profile, extreme angle, or full-body pose). " +
      "This is not a low likeness — it is the absence of a measurable frontal identity signal.",
  },
]);

// Map a faceLikeness block (or a bare cosine) to a band. This is the EXACT mapping sc-4413's
// badge needs:
//   - detected:false / score:null  -> "na"  (the honest no-frontal-face outcome)
//   - score >= STRONG_MIN          -> "strong"
//   - score >= MODERATE_MIN        -> "moderate"
//   - otherwise                    -> "weak"
//
// Accepts either the persisted block `{ score, detected, ... }` or a raw number. A missing/
// null/non-finite score, or an explicit detected:false, is "na" — never silently coerced to a
// weak score. (Detection honesty is enforced in the backend scorer; this guard keeps the UI
// honest even if a block arrives without `detected`.)
export function classifyLikeness(input) {
  const score = typeof input === "number" ? input : input?.score;
  const detected = typeof input === "number" ? true : input?.detected;

  if (detected === false) {
    return LIKENESS_BAND.NA;
  }
  if (typeof score !== "number" || !Number.isFinite(score)) {
    return LIKENESS_BAND.NA;
  }
  if (score >= STRONG_MIN) {
    return LIKENESS_BAND.STRONG;
  }
  if (score >= MODERATE_MIN) {
    return LIKENESS_BAND.MODERATE;
  }
  return LIKENESS_BAND.WEAK;
}

// The descriptor (label/min/max/description) for a band id, for the badge to render.
export function likenessBand(band) {
  return LIKENESS_BANDS.find((entry) => entry.band === band) ?? null;
}
