import { describe, expect, it } from "vitest";

import { buildImageJobAdvanced } from "./imageJobAdvanced.js";

// sc-8854 (F-052): the `advanced` payload is the app's highest-drift surface — every
// conditional spread encodes an omit-when-default rule that keeps saved recipes
// byte-identical across releases. These tests pin those rules so the extraction from
// ImageStudio.submit() is provably behavior-preserving and future edits stay honest.

// A state shaped so every optional spread is OFF: the payload should carry only the
// always-present keys (resolution + flashAttn:false when flashAttn is off).
function offState(overrides = {}) {
  return {
    resolution: "1024x1024",
    sendStructured: false,
    submitIntent: "",
    submitCaption: null,
    submitBackend: null,
    sampler: "default",
    scheduler: "default",
    schedulerShift: "",
    stepsOverride: "",
    guidanceOverride: "",
    guidanceMethod: "cfg",
    flashAttn: true,
    promptEnhance: false,
    enhancePrompt: false,
    precisionToggle: false,
    bf16Precision: false,
    showTierPicker: false,
    quantTier: "default",
    showPidToggle: false,
    usePid: false,
    mode: "text_to_image",
    referenceAssetId: null,
    hideReferenceStrength: false,
    ipAdapterScale: 0.8,
    identityStructure: false,
    controlnetScale: 0.5,
    variationStrength: false,
    trueCfgScale: 4,
    viewAngles: false,
    viewAngle: "",
    posePayload: [],
    faceRestore: false,
    controlActive: false,
    activeControlMode: null,
    controlPassthroughId: null,
    effectiveControlScale: 0.7,
    ...overrides,
  };
}

describe("buildImageJobAdvanced", () => {
  it("emits only the always-present keys when every optional knob is at its default", () => {
    const advanced = buildImageJobAdvanced(offState());
    // flashAttn:true (default-on) is omitted; nothing else leaks.
    expect(advanced).toEqual({ resolution: "1024x1024" });
  });

  it("emits flashAttn:false only when flash attention is toggled off", () => {
    expect(buildImageJobAdvanced(offState({ flashAttn: true }))).not.toHaveProperty("flashAttn");
    expect(buildImageJobAdvanced(offState({ flashAttn: false })).flashAttn).toBe(false);
  });

  it("omits sampler/scheduler when default and includes them otherwise", () => {
    expect(buildImageJobAdvanced(offState())).not.toHaveProperty("sampler");
    const custom = buildImageJobAdvanced(offState({ sampler: "euler", scheduler: "karras" }));
    expect(custom.sampler).toBe("euler");
    expect(custom.scheduler).toBe("karras");
  });

  it("omits guidanceMethod for the engine no-op 'cfg' and rides a non-default method", () => {
    expect(buildImageJobAdvanced(offState({ guidanceMethod: "cfg" }))).not.toHaveProperty("guidanceMethod");
    expect(buildImageJobAdvanced(offState({ guidanceMethod: "cfg_pp" })).guidanceMethod).toBe("cfg_pp");
  });

  it("only rides schedulerShift when a curated (non-default) scheduler is active and the value is finite", () => {
    // Default scheduler → shift omitted even if provided.
    expect(
      buildImageJobAdvanced(offState({ scheduler: "default", schedulerShift: "3" })),
    ).not.toHaveProperty("schedulerShift");
    // Curated scheduler + finite shift → coerced to Number and sent.
    expect(
      buildImageJobAdvanced(offState({ scheduler: "karras", schedulerShift: "3" })).schedulerShift,
    ).toBe(3);
    // Curated scheduler but a non-finite shift (Number("abc") -> NaN) → omitted. Note the
    // empty-string case coerces to 0 (Number("") === 0), which IS finite, matching the
    // original submit() behavior, so it would be sent — we don't assert it away here.
    expect(
      buildImageJobAdvanced(offState({ scheduler: "karras", schedulerShift: "abc" })),
    ).not.toHaveProperty("schedulerShift");
  });

  it("coerces step/guidance overrides and omits empty strings", () => {
    expect(buildImageJobAdvanced(offState({ stepsOverride: "", guidanceOverride: "" }))).toEqual({
      resolution: "1024x1024",
    });
    const set = buildImageJobAdvanced(offState({ stepsOverride: "28", guidanceOverride: "3.5" }));
    expect(set.steps).toBe(28);
    expect(set.guidanceScale).toBe(3.5);
  });

  it("emits enhancePrompt only when the model declares the toggle and it is on", () => {
    expect(buildImageJobAdvanced(offState({ promptEnhance: true, enhancePrompt: false }))).not.toHaveProperty(
      "enhancePrompt",
    );
    expect(buildImageJobAdvanced(offState({ promptEnhance: false, enhancePrompt: true }))).not.toHaveProperty(
      "enhancePrompt",
    );
    expect(
      buildImageJobAdvanced(offState({ promptEnhance: true, enhancePrompt: true })).enhancePrompt,
    ).toBe(true);
  });

  it("keeps the Boogu precision and quant-tier mlxQuantize spreads disjoint", () => {
    // Boogu precision toggle: bf16 selected, no tier picker → mlxQuantize:0.
    expect(
      buildImageJobAdvanced(offState({ precisionToggle: true, bf16Precision: true, showTierPicker: false }))
        .mlxQuantize,
    ).toBe(0);
    // Its !showTierPicker guard suppresses the Boogu path when a tier picker is present;
    // the tier path then decides. With a "default" pseudo-tier the tier path also omits,
    // so neither spread leaks mlxQuantize.
    expect(
      buildImageJobAdvanced(
        offState({ precisionToggle: true, bf16Precision: true, showTierPicker: true, quantTier: "default" }),
      ),
    ).not.toHaveProperty("mlxQuantize");
    // With a real picked tier, that tier's quant value wins (q8 -> 8), never the Boogu 0.
    expect(
      buildImageJobAdvanced(
        offState({ precisionToggle: true, bf16Precision: true, showTierPicker: true, quantTier: "q8" }),
      ).mlxQuantize,
    ).toBe(8);
  });

  it("maps the picked quant tier to mlxQuantize (bf16->0, q8->8, q4->4) and omits the 'default' pseudo-tier", () => {
    expect(buildImageJobAdvanced(offState({ showTierPicker: true, quantTier: "q4" })).mlxQuantize).toBe(4);
    expect(buildImageJobAdvanced(offState({ showTierPicker: true, quantTier: "bf16" })).mlxQuantize).toBe(0);
    // "default" → tierQuantize null → omitted.
    expect(buildImageJobAdvanced(offState({ showTierPicker: true, quantTier: "default" }))).not.toHaveProperty(
      "mlxQuantize",
    );
  });

  it("emits usePid only when the PiD toggle is shown and on", () => {
    expect(buildImageJobAdvanced(offState({ showPidToggle: false, usePid: true }))).not.toHaveProperty("usePid");
    expect(buildImageJobAdvanced(offState({ showPidToggle: true, usePid: true })).usePid).toBe(true);
  });

  it("embeds a structured-prompt recipe only for structured submissions", () => {
    expect(buildImageJobAdvanced(offState({ sendStructured: false }))).not.toHaveProperty("structuredPrompt");
    const structured = buildImageJobAdvanced(
      offState({ sendStructured: true, submitIntent: "a cat", submitCaption: { subject: "cat" }, submitBackend: "llm" }),
    );
    expect(structured.structuredPrompt).toMatchObject({ intent: "a cat", magicPromptBackend: "llm", edited: false });
  });

  describe("character-reference gating", () => {
    const charRef = { mode: "character_image", referenceAssetId: "ref-1" };

    it("gates ipAdapterScale on a character reference that is not hidden", () => {
      expect(buildImageJobAdvanced(offState(charRef)).ipAdapterScale).toBe(0.8);
      expect(
        buildImageJobAdvanced(offState({ ...charRef, hideReferenceStrength: true })),
      ).not.toHaveProperty("ipAdapterScale");
      // No reference → omitted regardless.
      expect(buildImageJobAdvanced(offState({ mode: "character_image", referenceAssetId: null }))).not.toHaveProperty(
        "ipAdapterScale",
      );
    });

    it("gates controlnetConditioningScale on identityStructure + reference", () => {
      expect(
        buildImageJobAdvanced(offState({ ...charRef, identityStructure: true, controlnetScale: 0.6 }))
          .controlnetConditioningScale,
      ).toBe(0.6);
      expect(buildImageJobAdvanced(offState({ ...charRef, identityStructure: false }))).not.toHaveProperty(
        "controlnetConditioningScale",
      );
    });

    it("suppresses viewAngle when a pose is selected (pose supersedes head angle)", () => {
      const base = { ...charRef, viewAngles: true, viewAngle: "three_quarter" };
      expect(buildImageJobAdvanced(offState(base)).viewAngle).toBe("three_quarter");
      expect(
        buildImageJobAdvanced(offState({ ...base, posePayload: [{ id: "p1", keypoints: [] }] })),
      ).not.toHaveProperty("viewAngle");
    });
  });

  describe("pose + strict control", () => {
    it("emits poses + faceRestore when a pose payload is present", () => {
      const poses = [{ id: "p1", keypoints: [1, 2] }];
      const advanced = buildImageJobAdvanced(offState({ posePayload: poses, faceRestore: true }));
      expect(advanced.poses).toBe(poses);
      expect(advanced.faceRestore).toBe(true);
    });

    it("rides controlMode only for a non-pose active control mode, plus controlScale whenever active", () => {
      const pose = buildImageJobAdvanced(offState({ controlActive: true, activeControlMode: "pose" }));
      expect(pose).not.toHaveProperty("controlMode");
      expect(pose.controlScale).toBe(0.7);

      const canny = buildImageJobAdvanced(
        offState({ controlActive: true, activeControlMode: "canny", effectiveControlScale: 0.9 }),
      );
      expect(canny.controlMode).toBe("canny");
      expect(canny.controlScale).toBe(0.9);
    });

    it("rides controlImage only for a passthrough control map", () => {
      expect(buildImageJobAdvanced(offState({ controlPassthroughId: null }))).not.toHaveProperty("controlImage");
      expect(
        buildImageJobAdvanced(offState({ controlActive: true, controlPassthroughId: "map-1" })).controlImage,
      ).toBe("map-1");
    });
  });
});
