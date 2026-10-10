import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import JSON5 from "json5";

import {
  ADVANCED_TRANSPARENT_BACKGROUND,
  CHANNELS_RGB,
  CHANNELS_RGBA,
  MODEL_ALPHA_CAPABILITY_FLAG,
  OUTPUT_CHANNELS_RGB,
  OUTPUT_CHANNELS_RGBA,
  REQUEST_OUTPUT_CHANNELS,
  TRANSPARENCY_PROMPT_HINT,
  outputChannelsVariant,
  transparencyPromptSuggestion,
  modelSupportsAlphaOutput,
  resolveOutputChannels,
  showTransparencyToggle,
  transparencyAdvanced,
  transparencyRequested,
  imageBytesCarryAlpha,
  editTransparencyFor,
} from "./qwenAlpha.js";
import { buildEditJobBody } from "./imageJobs.js";
import {
  maxReferencesForModel,
  moveReference,
  referenceOrdinalLabel,
} from "./imageReferenceLimits.js";
import {
  dimensionConstraintMessage,
  evaluateModelDimensions,
  modelDimensionConstraints,
} from "./resolutionOverride.js";
import { minStepsForModel } from "./videoModelLimits.js";
import { fallbackModels, QWEN_IMAGE_2_1_MODEL_ID } from "./constants.js";
import { buildImageJobAdvanced } from "./imageJobAdvanced.js";
import { OrderedReferenceList } from "./components/OrderedReferenceList.jsx";

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST = resolve(HERE, "../../../config/manifests/builtin.models.jsonc");

function manifestEntry(id) {
  const manifest = JSON5.parse(readFileSync(MANIFEST, "utf8"));
  return manifest.models.find((model) => model.id === id);
}

const seed = fallbackModels.find((entry) => entry.id === QWEN_IMAGE_2_1_MODEL_ID);

// The seven presets from the S1 contract, in the engine's own order.
const PRESETS = [
  "2048x2048",
  "2400x1792",
  "1792x2400",
  "2528x1696",
  "1696x2528",
  "2752x1536",
  "1536x2752",
];

describe("Qwen Image 2.1 control surface (sc-24113)", () => {
  // The whole control surface is manifest-declared, so the catalog is the thing to assert. A
  // control that reads the right key from the wrong number is still the wrong control.
  it("declares every bound the studio controls read", () => {
    const entry = manifestEntry(QWEN_IMAGE_2_1_MODEL_ID);
    expect(entry).toBeTruthy();
    expect(entry.limits.resolutions).toEqual(PRESETS);
    expect(entry.defaults.resolution).toBe("2048x2048");
    expect(entry.defaults.steps).toBe(40);
    expect(entry.limits.hardMinSteps).toBe(2);
    expect(entry.limits.minDimension).toBe(32);
    expect(entry.limits.maxDimension).toBe(2752);
    expect(entry.limits.requiresDimensionsMultipleOf).toBe(32);
    expect(entry.limits.maxReferenceAssets).toBe(10);
    expect(Math.max(...entry.limits.count)).toBe(8);
    // Ordered references are an EDIT capability; there is no mask tensor in this family, so
    // `image_inpaint` must stay undeclared or the editor would offer a mask tool the engine
    // refuses by name.
    expect(entry.capabilities).toContain("image_to_image");
    expect(entry.capabilities).not.toContain("image_inpaint");
    expect(entry[MODEL_ALPHA_CAPABILITY_FLAG]).toBe(true);
  });

  // The seed in constants.js is what the studio renders BEFORE the catalog loads. It used to carry
  // no limits at all, so a pre-catalog render offered the blanket 768²/1024²/1280x720/720x1280 —
  // not one of which is a legal 2.1 bucket — plus a step floor of 1 against a real floor of 2.
  it("keeps the pre-catalog seed in step with the manifest", () => {
    const entry = manifestEntry(QWEN_IMAGE_2_1_MODEL_ID);
    expect(seed).toBeTruthy();
    expect(seed.limits.resolutions).toEqual(entry.limits.resolutions);
    expect(seed.limits.count).toEqual(entry.limits.count);
    expect(seed.limits.minDimension).toBe(entry.limits.minDimension);
    expect(seed.limits.maxDimension).toBe(entry.limits.maxDimension);
    expect(seed.limits.requiresDimensionsMultipleOf).toBe(
      entry.limits.requiresDimensionsMultipleOf,
    );
    expect(seed.limits.maxReferenceAssets).toBe(entry.limits.maxReferenceAssets);
    expect(seed.limits.hardMinSteps).toBe(entry.limits.hardMinSteps);
    expect(seed.defaults.resolution).toBe(entry.defaults.resolution);
    expect(seed.defaults.steps).toBe(entry.defaults.steps);
    expect(seed[MODEL_ALPHA_CAPABILITY_FLAG]).toBe(entry[MODEL_ALPHA_CAPABILITY_FLAG]);
    expect(seed.capabilities).toEqual(entry.capabilities);
  });

  it("reads the model's step floor instead of the hardcoded 1", () => {
    expect(minStepsForModel(seed)).toBe(2);
    // Every model that declares nothing keeps 1 — the control is byte-identical for them.
    expect(minStepsForModel({ limits: {} })).toBe(1);
    expect(minStepsForModel(undefined)).toBe(1);
  });
});

describe("free size on the declared 32-px grid (sc-24113)", () => {
  it("opens the full 32..2752 envelope rather than the preset extremes", () => {
    // Derived from the ladder this would be 1536..2752 — the smallest and largest PRESET sides —
    // which is not the engine's envelope and would refuse every legal small size.
    expect(modelDimensionConstraints(seed)).toEqual({ min: 32, max: 2752, step: 32 });
  });

  it("accepts every legal size on the grid and refuses the rest by reason", () => {
    const evaluate = (widthOverride, heightOverride) =>
      evaluateModelDimensions({
        model: seed,
        resolution: "2048x2048",
        widthOverride,
        heightOverride,
      });

    for (const size of [32, 64, 1024, 2048, 2752]) {
      const result = evaluate(String(size), String(size));
      expect(result.invalid, `${size} is legal`).toBe(false);
    }

    // Over the ceiling.
    const tooBig = evaluate("2784", "2048");
    expect(tooBig.outOfRange).toBe(true);
    expect(dimensionConstraintMessage(tooBig)).toContain("2752");

    // Below the floor.
    expect(evaluate("16", "2048").outOfRange).toBe(true);

    // Off the grid, inside the range — the case that used to pass every check in the app and die
    // inside the provider.
    const offGrid = evaluate("2050", "2048");
    expect(offGrid.outOfRange).toBe(false);
    expect(offGrid.offStride).toBe(true);
    expect(dimensionConstraintMessage(offGrid)).toContain("multiple of 32");
  });

  it("leaves a model that declares no envelope on the blanket bounds", () => {
    expect(modelDimensionConstraints({ limits: {} })).toEqual({ min: 256, max: 4096, step: 1 });
    expect(modelDimensionConstraints(undefined)).toEqual({ min: 256, max: 4096, step: 1 });
    // A stride-only model still derives its range from its ladder (the Mage-Flow shape), unchanged.
    const mageish = { limits: { requiresDimensionsMultipleOf: 16, resolutions: ["512x512", "2048x2048"] } };
    expect(modelDimensionConstraints(mageish)).toEqual({ min: 512, max: 2048, step: 16 });
  });

  it("every shipped preset sits inside the declared envelope and on the grid", () => {
    const { min, max, step } = modelDimensionConstraints(seed);
    for (const preset of PRESETS) {
      for (const side of preset.split("x").map(Number)) {
        expect(side, `${preset} within range`).toBeGreaterThanOrEqual(min);
        expect(side, `${preset} within range`).toBeLessThanOrEqual(max);
        expect(side % step, `${preset} on grid`).toBe(0);
      }
    }
  });
});

describe("ordered references (sc-24113)", () => {
  it("raises the cap to the model's declared ten without moving anyone else", () => {
    expect(maxReferencesForModel(seed, 4)).toBe(10);
    // Absent ⇒ the caller's own constant, so FLUX.2 / SenseNova / Krea are untouched.
    expect(maxReferencesForModel({ limits: {} }, 4)).toBe(4);
    expect(maxReferencesForModel(undefined, 4)).toBe(4);
    // A declared 0 is meaningful ("takes none") and must NOT collapse to the fallback.
    expect(maxReferencesForModel({ limits: { maxReferenceAssets: 0 } }, 4)).toBe(0);
  });

  // sc-24110: the picker has to be REACHABLE, not just correctly bounded.
  //
  // `maxReferencesForModel` above says how many references the rail accepts; this says whether the
  // rail is rendered at all. ImageEditor gates it on `ui.multiReference` and ImageStudio gates the
  // plural `referenceAssetIds` payload on the same flag, so without it the 1-10 ordered references
  // this model is built around were reachable only by driving the API directly, while the shipped
  // prompt guide documented them. Asserted against the SHIPPED manifest and the fallback catalog
  // together, because a flag in one and not the other is a pre-catalog UI that differs from the
  // post-catalog one.
  it("opens the reference picker for this model, on both catalogs, at the declared cap", () => {
    const entry = manifestEntry(QWEN_IMAGE_2_1_MODEL_ID);

    expect(entry.ui.multiReference, "manifest opens the picker").toBe(true);
    expect(seed.ui.multiReference, "fallback catalog opens it too").toBe(true);

    // The cap the rail uses comes from the same manifest key the enqueue gate and the worker read.
    expect(maxReferencesForModel(entry, 4)).toBe(10);
    expect(maxReferencesForModel(seed, 4)).toBe(10);

    // `img2img` stays UNDECLARED on both, and that is a decision rather than an omission: that flag
    // opens the Studio's single-source flow whose whole control is `advanced.strength`, and
    // upstream's condition images have NO strength - the engine refuses one and the API 400s it at
    // enqueue, so the slider would advertise a knob every render rejects. The single-reference case
    // is the ordered list with one entry, which is the same engine call.
    expect(entry.ui.img2img, "manifest must not offer a strength slider").toBeUndefined();
    expect(seed.ui.img2img, "nor may the fallback catalog").toBeUndefined();

    // And the capability the Editor gates its MASK tool on is absent, so no inpaint UI appears:
    // this model has no mask tensor, and a mask travels as an ordinary ordered reference.
    expect(entry.capabilities).toContain("edit_image");
    expect(entry.capabilities).not.toContain("image_inpaint");
    expect(seed.capabilities).not.toContain("image_inpaint");
  });

  it("shows the 1-based ordinal the engine's template uses", () => {
    expect(referenceOrdinalLabel(0)).toBe("Image 1");
    expect(referenceOrdinalLabel(1)).toBe("Image 2");
    expect(referenceOrdinalLabel(9)).toBe("Image 10");
  });

  it("reorders in place, and a reorder is a different list", () => {
    const ids = ["a", "b", "c"];
    expect(moveReference(ids, 0, 1)).toEqual(["b", "a", "c"]);
    expect(moveReference(ids, 2, 0)).toEqual(["c", "a", "b"]);
    // Pure: the input is never mutated, which is what lets the caller pass it straight to setState.
    expect(ids).toEqual(["a", "b", "c"]);
    // A reorder genuinely changes the request — this is the claim the whole ordered-reference
    // contract rests on.
    expect(moveReference(ids, 0, 1)).not.toEqual(ids);
  });

  it("is total: an out-of-range or malformed move is a no-op, never a throw", () => {
    const ids = ["a", "b"];
    for (const [from, to] of [
      [-1, 0],
      [0, -1],
      [5, 0],
      [0, 5],
      [Number.NaN, 0],
      [0, undefined],
    ]) {
      expect(moveReference(ids, from, to)).toEqual(ids);
    }
    expect(moveReference(null, 0, 1)).toEqual([]);
    expect(moveReference(undefined, 0, 1)).toEqual([]);
    // A no-op move still returns an EQUAL array, so a caller can setState unconditionally.
    expect(moveReference(ids, 1, 1)).toEqual(ids);
  });
});

describe("transparency / RGBA output (sc-24113)", () => {
  it("offers the toggle only where the model advertises alpha output", () => {
    expect(modelSupportsAlphaOutput(seed)).toBe(true);
    expect(showTransparencyToggle(seed)).toBe(true);
    // Strictly `=== true`. Unlike the sc-15299 generation axes, ABSENT MEANS FALSE here: a missing
    // flag would otherwise offer a cut-out toggle on every model in the catalog.
    for (const model of [undefined, {}, { supportsAlphaOutput: false }, { supportsAlphaOutput: "yes" }]) {
      expect(modelSupportsAlphaOutput(model)).toBe(false);
      expect(showTransparencyToggle(model)).toBe(false);
    }
  });

  it("puts nothing new on the wire for an opaque render", () => {
    expect(transparencyAdvanced(seed, false)).toEqual({});
    expect(resolveOutputChannels(seed, false)).toBe(CHANNELS_RGB);
    // Every pre-sc-24113 lane is byte-identical: a model with no flag emits nothing either way.
    expect(transparencyAdvanced({ id: "flux_dev" }, false)).toEqual({});
  });

  it("emits exactly one field for a genuine transparency request", () => {
    expect(transparencyAdvanced(seed, true)).toEqual({ [ADVANCED_TRANSPARENT_BACKGROUND]: true });
    expect(resolveOutputChannels(seed, true)).toBe(CHANNELS_RGBA);
  });

  it("cannot leak a sticky toggle onto a model that would refuse it", () => {
    // The toggle is a sticky pref, so it survives a model switch. The gate is re-evaluated against
    // the CURRENTLY selected model at payload-build time, which is what makes that safe: a `true`
    // carried over from Qwen 2.1 emits nothing on FLUX.2, where the worker would refuse it by name.
    expect(transparencyAdvanced({ id: "flux_dev" }, true)).toEqual({});
    expect(resolveOutputChannels({ id: "flux_dev" }, true)).toBe(CHANNELS_RGB);
  });

  it("round-trips through the advanced payload the studio actually builds", () => {
    const base = { resolution: "2048x2048", posePayload: [], selectedModel: seed };
    const advanced = buildImageJobAdvanced({ ...base, transparentBackground: true });
    expect(advanced[ADVANCED_TRANSPARENT_BACKGROUND]).toBe(true);
    // ... and reading it back out of a stored recipe re-arms the toggle.
    expect(transparencyRequested(advanced)).toBe(true);

    const opaque = buildImageJobAdvanced({ ...base, transparentBackground: false });
    expect(ADVANCED_TRANSPARENT_BACKGROUND in opaque).toBe(false);
    expect(transparencyRequested(opaque)).toBe(false);
  });

  it("reads a malformed stored toggle as off", () => {
    // An old recipe or a hand-edited payload must not silently change what the render emits.
    for (const value of ["true", 1, null, undefined, {}]) {
      expect(transparencyRequested({ [ADVANCED_TRANSPARENT_BACKGROUND]: value })).toBe(false);
    }
    expect(transparencyRequested(undefined)).toBe(false);
  });

  it("pins the S4 contract names the Rust half mirrors", () => {
    // Transcribed from the S4 RGBA contract (inference sc-24111, PR #1009), and asserted here so a
    // rename fails loudly in one place with the mirror named rather than drifting away from
    // `crates/sceneworks-worker/src/qwen_alpha.rs`.
    expect(MODEL_ALPHA_CAPABILITY_FLAG).toBe("supportsAlphaOutput");
    // An ENUM, not a channel count — `Rgb` is the contract's default.
    expect(REQUEST_OUTPUT_CHANNELS).toBe("output_channels");
    expect(OUTPUT_CHANNELS_RGB).toBe("rgb");
    expect(OUTPUT_CHANNELS_RGBA).toBe("rgba");
    expect(outputChannelsVariant(CHANNELS_RGBA)).toBe("rgba");
    expect(outputChannelsVariant(CHANNELS_RGB)).toBe("rgb");
    // This one is OURS — a SceneWorks request axis, not an engine field (the engine has no
    // transparency concept at all) — so it is stable.
    expect(ADVANCED_TRANSPARENT_BACKGROUND).toBe("transparentBackground");
  });

  // THE non-obvious half of this control. There is no transparency MODE upstream: `output_channels:
  // Rgba` only keeps the alpha channel through the decode, and whether that channel holds a cut-out
  // is decided by the PROMPT. A toggle with no wording behind it returns a fully opaque RGBA PNG
  // and reads as broken.
  it("pairs the toggle with the model card's prompt convention", () => {
    const suggestion = transparencyPromptSuggestion("a courier standing still", true);
    expect(suggestion).toContain("a courier standing still");
    expect(suggestion).toContain(TRANSPARENCY_PROMPT_HINT);
    // Punctuation is joined, not doubled.
    expect(suggestion).not.toContain("..");
    expect(transparencyPromptSuggestion("A courier.", true)).toBe(
      `A courier. ${TRANSPARENCY_PROMPT_HINT}`,
    );
    // An empty prompt gets the bare convention rather than a leading separator.
    expect(transparencyPromptSuggestion("", true)).toBe(TRANSPARENCY_PROMPT_HINT);
  });

  it("offers the wording once, and never when the toggle is off", () => {
    expect(transparencyPromptSuggestion("a courier", false)).toBeNull();
    // Already said — in the model card's words or the user's own. A near-duplicate sentence is
    // noise, and re-running a recipe must not accrete the hint once per run.
    for (const prompt of [
      "a courier on a transparent background",
      `A courier. ${TRANSPARENCY_PROMPT_HINT}`,
      "a courier, RGBA with transparency",
      "a courier on a TRANSPARENT backdrop",
    ]) {
      expect(transparencyPromptSuggestion(prompt, true), prompt).toBeNull();
    }
    // Total: a non-string prompt is not a crash.
    expect(transparencyPromptSuggestion(undefined, true)).toBe(TRANSPARENCY_PROMPT_HINT);
  });
});

describe("the ordered-reference rail (sc-24113)", () => {
  // The rail is the UI half of "reorder = a different request". The pure helpers are covered above;
  // this covers the component's own contract, which is what both shells depend on.
  it("renders nothing below two references", () => {
    // With one reference there is no order to show, and an empty rail is noise beside a picker that
    // already says "none selected".
    for (const ids of [[], ["a"], undefined]) {
      expect(
        OrderedReferenceList({ assetIds: ids, onChange: () => {} }),
        JSON.stringify(ids ?? null),
      ).toBeNull();
    }
    expect(OrderedReferenceList({ assetIds: ["a", "b"], onChange: () => {} })).not.toBeNull();
  });

  it("labels each reference with the ordinal the engine's template uses", () => {
    const rows = OrderedReferenceList({ assetIds: ["a", "b", "c"], onChange: () => {} }).props
      .children;
    expect(rows).toHaveLength(3);
    // 1-based, matching `<image1>` … — the numbering the prompt conventions for this family use
    // ("use the second image as a mask").
    expect(rows.map((row) => row.props.children[0].props.children)).toEqual([
      "Image 1",
      "Image 2",
      "Image 3",
    ]);
  });

  it("moves a reference and hands the caller a reordered list", () => {
    const onChange = vi.fn();
    const rows = OrderedReferenceList({ assetIds: ["a", "b", "c"], onChange }).props.children;
    const actionsFor = (index) => rows[index].props.children[2].props.children;

    // "Move later" on the first reference.
    actionsFor(0)[1].props.onClick();
    expect(onChange).toHaveBeenCalledWith(["b", "a", "c"]);

    // "Move earlier" on the last.
    actionsFor(2)[0].props.onClick();
    expect(onChange).toHaveBeenLastCalledWith(["a", "c", "b"]);

    // The ends cannot move past themselves — DISABLED rather than a silent no-op, so the control
    // says what it will do before it is pressed.
    expect(actionsFor(0)[0].props.disabled).toBe(true);
    expect(actionsFor(2)[1].props.disabled).toBe(true);
    expect(actionsFor(1)[0].props.disabled).toBe(false);
    expect(actionsFor(1)[1].props.disabled).toBe(false);
  });
});

// sc-24114 — transparency on the Image Editor's edit lane.
describe("editor transparency (sc-24114)", () => {
  const png = (colourType, chunks = []) => {
    const bytes = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    const chunk = (type, data) => {
      const length = data.length;
      bytes.push((length >>> 24) & 255, (length >>> 16) & 255, (length >>> 8) & 255, length & 255);
      for (const ch of type) bytes.push(ch.charCodeAt(0));
      bytes.push(...data, 0, 0, 0, 0);
    };
    chunk("IHDR", [0, 0, 0, 1, 0, 0, 0, 1, 8, colourType, 0, 0, 0]);
    for (const [type, data] of chunks) chunk(type, data);
    chunk("IDAT", [0]);
    chunk("IEND", []);
    return new Uint8Array(bytes);
  };

  // *Mutation that reds this:* dropping the colour-type 6 arm, or the tRNS scan.
  it("reads alpha off the PNG / WebP header", () => {
    expect(imageBytesCarryAlpha(png(6))).toBe(true);
    expect(imageBytesCarryAlpha(png(4))).toBe(true);
    expect(imageBytesCarryAlpha(png(2))).toBe(false);
    expect(imageBytesCarryAlpha(png(3, [["tRNS", [0]]]))).toBe(true);
    expect(imageBytesCarryAlpha(png(3))).toBe(false);
    const webp = (chunk, flagOffset, flag) => {
      const data = new Uint8Array(32);
      data.set([..."RIFF"].map((c) => c.charCodeAt(0)), 0);
      data.set([..."WEBP"].map((c) => c.charCodeAt(0)), 8);
      data.set([...chunk].map((c) => c.charCodeAt(0)), 12);
      data[flagOffset] = flag;
      return data;
    };
    expect(imageBytesCarryAlpha(webp("VP8X", 20, 0x10))).toBe(true);
    expect(imageBytesCarryAlpha(webp("VP8X", 20, 0))).toBe(false);
    expect(imageBytesCarryAlpha(new Uint8Array([0xff, 0xd8, 0xff]))).toBe(false);
  });

  // *Mutation that reds this:* defaulting to OFF regardless of the working image's alpha.
  it("defaults the edit's transparency to the working image's alpha until the user toggles it", () => {
    expect(editTransparencyFor(null, true)).toBe(true);
    expect(editTransparencyFor(null, false)).toBe(false);
    expect(editTransparencyFor(false, true)).toBe(false);
    expect(editTransparencyFor(true, false)).toBe(true);
  });

  // *Mutation that reds this:* dropping the transparentBackground arm from buildEditJobBody.
  it("buildEditJobBody carries transparentBackground only for an alpha-capable model", () => {
    const qwen = fallbackModels.find((model) => model.id === QWEN_IMAGE_2_1_MODEL_ID);
    const base = {
      project: { id: "p" },
      requestedGpu: null,
      sourceAssetId: "src",
      model: qwen.id,
      prompt: "cut out the subject",
      seed: null,
      width: 1024,
      height: 1024,
    };
    expect(
      buildEditJobBody({ ...base, transparentBackground: true, modelEntry: qwen }).advanced,
    ).toEqual({ transparentBackground: true });
    expect(buildEditJobBody({ ...base, transparentBackground: false, modelEntry: qwen }).advanced).toEqual({});
    expect(
      buildEditJobBody({
        ...base,
        transparentBackground: true,
        modelEntry: { id: "flux2_dev" },
      }).advanced,
    ).toEqual({});
  });
});
