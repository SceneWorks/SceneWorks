import { describe, expect, it } from "vitest";
import {
  LIKENESS_BAND,
  LIKENESS_BANDS,
  LIKENESS_METHOD,
  MODERATE_MIN,
  STRONG_MIN,
  classifyLikeness,
  likenessBand,
} from "./faceLikeness.js";

// Face-likeness band calibration + classifier (epic 4406, sc-4414). These pin the calibrated
// cut-points to the recorded antelopev2 ArcFace baselines and prove the classifier maps each
// outcome (incl. the explicit N/A) the way the sc-4413 badge consumes it.
describe("face-likeness band calibration (sc-4414)", () => {
  it("exposes the method string the backend stamps", () => {
    expect(LIKENESS_METHOD).toBe("arcface_antelopev2");
  });

  it("cut-points are the calibrated edges (strong>=0.80, moderate>=0.55)", () => {
    expect(STRONG_MIN).toBe(0.8);
    expect(MODERATE_MIN).toBe(0.55);
    expect(MODERATE_MIN).toBeLessThan(STRONG_MIN);
  });

  // -- the classifier mapping the badge (sc-4413) consumes ------------------------------
  describe("classifyLikeness", () => {
    it("classifies solid frontal identity as strong", () => {
      // Recorded anchors that MUST land in strong: InstantID frontal ~0.876, its angle pack
      // 0.81–0.89, PuLID-FLUX photoreal 0.8016, upper face-restore ~0.83.
      for (const score of [0.876, 0.81, 0.89, 0.8016, 0.83, STRONG_MIN, 1]) {
        expect(classifyLikeness(score), `${score} should be strong`).toBe(LIKENESS_BAND.STRONG);
      }
    });

    it("classifies recognizable-but-drifting as moderate", () => {
      // The [0.55, 0.80) band: pose tier ~0.71, lower face-restore ~0.74, PuLID iw0.8 drift
      // 0.7422, Qwen-Edit Lightning angle mean ~0.62, PuLID iw0.6 borderline drift 0.5689
      // (the closest anchor above the floor — 0.55 sits just below it), and the floor itself.
      for (const score of [0.71, 0.74, 0.7422, 0.62, 0.5689, MODERATE_MIN]) {
        expect(classifyLikeness(score), `${score} should be moderate`).toBe(LIKENESS_BAND.MODERATE);
      }
    });

    it("classifies clear drift / collapse as weak", () => {
      // Below the moderate floor: FLUX.2-klein angle mean ~0.52 (soft-identity backbone),
      // InstantID landmark-disabled collapse ~0.15, and outright non-matches.
      for (const score of [0.52, 0.15, 0, -0.3]) {
        expect(classifyLikeness(score), `${score} should be weak`).toBe(LIKENESS_BAND.WEAK);
      }
    });

    it("treats a detected:false block as N/A, never a low number", () => {
      // The honesty linchpin (mirrors the backend scorer): a profile / no-face generation is
      // detected:false / score:null -> N/A, NOT weak.
      expect(
        classifyLikeness({ score: null, detected: false, method: LIKENESS_METHOD, reason: "no_face" }),
      ).toBe(LIKENESS_BAND.NA);
      expect(classifyLikeness({ score: null, detected: false, reason: "low_confidence" })).toBe(
        LIKENESS_BAND.NA,
      );
    });

    it("treats a missing / non-finite score as N/A", () => {
      expect(classifyLikeness(undefined)).toBe(LIKENESS_BAND.NA);
      expect(classifyLikeness(null)).toBe(LIKENESS_BAND.NA);
      expect(classifyLikeness({})).toBe(LIKENESS_BAND.NA);
      expect(classifyLikeness(Number.NaN)).toBe(LIKENESS_BAND.NA);
      expect(classifyLikeness({ score: Number.NaN, detected: true })).toBe(LIKENESS_BAND.NA);
    });

    it("classifies a scored block (detected:true) by its cosine", () => {
      expect(classifyLikeness({ score: 0.876, detected: true })).toBe(LIKENESS_BAND.STRONG);
      expect(classifyLikeness({ score: 0.71, detected: true })).toBe(LIKENESS_BAND.MODERATE);
      expect(classifyLikeness({ score: 0.2, detected: true })).toBe(LIKENESS_BAND.WEAK);
    });
  });

  describe("band descriptors", () => {
    it("has a descriptor for every band incl. the N/A non-tier", () => {
      for (const band of Object.values(LIKENESS_BAND)) {
        const descriptor = likenessBand(band);
        expect(descriptor, `${band} must have a descriptor`).toBeTruthy();
        expect(descriptor.label).toBeTruthy();
        expect(descriptor.description).toBeTruthy();
      }
    });

    it("the scored bands tile [-1, 1] contiguously at the cut-points", () => {
      const strong = LIKENESS_BANDS.find((b) => b.band === LIKENESS_BAND.STRONG);
      const moderate = LIKENESS_BANDS.find((b) => b.band === LIKENESS_BAND.MODERATE);
      const weak = LIKENESS_BANDS.find((b) => b.band === LIKENESS_BAND.WEAK);
      expect(strong.min).toBe(STRONG_MIN);
      expect(moderate.max).toBe(STRONG_MIN);
      expect(moderate.min).toBe(MODERATE_MIN);
      expect(weak.max).toBe(MODERATE_MIN);
    });
  });
});
