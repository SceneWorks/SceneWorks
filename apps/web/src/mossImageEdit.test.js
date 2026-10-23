import { describe, expect, it } from "vitest";
import { buildImageJobRequest } from "./imageJobRequest.js";
import { buildSimpleImageRequest } from "./simple/simpleJobs.js";
import { buildEditJobBody } from "./imageJobs.js";
import { mossEditDefaults, mossComponentState, mossEditReferenceIds, mossEditWireOptions } from "./mossImage.js";
import { imageModelServesMode } from "./modelEligibility.js";

const state = (more = {}) => ({ model: "moss_image_sft", mode: "edit_image", promptToSend: "日本語: make image 2 blue", negativePrompt: "blur", count: 2, seed: "0", width: 512, height: 768, posePayload: [], referenceAssetIds: ["wide", "tall", "wide", "square"], loras: [{ id: "stale" }], mossImage: { acceleration: "full_fp32", slg: { layers: [22] }, uncond_skip_layers: [22] }, mossOrdinarySettings: { steps: 5 }, mossDmdSteps: 8, showPidToggle: true, usePid: true, supportsImg2img: true, img2imgReferenceAssetId: "stale", img2imgStrength: 0.8, hiresFixEnabled: true, sampler: "dpmpp_2m", scheduler: "custom", stepsOverride: "", guidanceOverride: "", schedulerShift: "", ...more });

describe("MOSS edit shared product boundary", () => {
  it("requires an explicit server edit capability rather than inferring support from OS or missing flags",()=>{
    for(const capabilities of [undefined,[],["text_to_image"]]) expect(imageModelServesMode({id:"moss_image_sft",capabilities,macSupport:{features:{edit:true}}},"edit_image",{platform:"macos"})).toBe(false);
    expect(imageModelServesMode({id:"moss_image_sft",capabilities:["text_to_image","edit_image"]},"edit_image",{platform:"linux"})).toBe(true);
    expect(imageModelServesMode({id:"krea_2",capabilities:["edit_image"]},"edit_image",null)).toBe(true);
  });
  it.each(["moss_image_base", "moss_image_sft"])("emits canonical slots and edit defaults without sticky ordinary/DMD controls for %s", (model) => {
    const body = buildImageJobRequest(state({ model }));
    expect(body.referenceAssetIds).toEqual(["wide", "tall", "wide", "square"]);
    expect(body.sourceAssetId).toBeUndefined(); expect(body.referenceAssetId).toBeUndefined(); expect(body.fitMode).toBeUndefined();
    expect(body.prompt).toBe("日本語: make image 2 blue"); expect(body.seed).toBe(0); expect(body.negativePrompt).toBe("blur");
    expect(body.loras).toEqual([]); expect(body.hiresFix).toBeUndefined();
    expect(body.advanced).toEqual({ resolution: undefined, mossImage: { acceleration: "off", max_sequence_length: 4096, training_shift: 1, uncond_skip_layers: [] }, mossImageEdit: mossEditDefaults(), steps: 30, guidanceScale: 4, schedulerShift: 5 });
    const batch = buildImageJobRequest(state({ model, promptToSend: "batch item" }));
    expect(batch.advanced).toEqual(body.advanced); expect(batch.referenceAssetIds).toEqual(body.referenceAssetIds);
  });
  it("shares nondefault controls across Full, Simple and Editor emitters", () => {
    const controls = { reference_max_edge: 1024, use_kv_cache: false };
    const full = buildImageJobRequest(state({ loras: [], stepsOverride: 41, guidanceOverride: 0, schedulerShift: 6, mossImageEdit: controls }));
    const simple = buildSimpleImageRequest({ prompt: "edit", mode: "edit_image", model: "moss_image_sft", resolution: "512x768", count: 2, seed: "0", referenceAssetIds: full.referenceAssetIds, mossImageEdit: controls, steps: 41, guidance: 0, schedulerShift: 6 });
    const editor = buildEditJobBody({ project: { id: "p" }, model: "moss_image_sft", prompt: "edit", referenceAssetIds: full.referenceAssetIds, mossImageEdit: controls, mossEditSettings: { width: 512, height: 768, steps: 41, guidanceScale: 0, schedulerShift: 6 } });
    for (const body of [simple, editor]) {
      expect(body.referenceAssetIds).toEqual(full.referenceAssetIds);
      for (const key of ["mossImageEdit", "steps", "guidanceScale", "schedulerShift"]) expect(body.advanced[key]).toEqual(full.advanced[key]);
    }
  });
  it("counts aliases once while preserving repeated vector slots and refusing conflicts", () => {
    expect(mossEditReferenceIds({ sourceAssetId: "a", referenceAssetId: "a", referenceAssetIds: ["a"] })).toEqual(["a"]);
    expect(mossEditReferenceIds({ sourceAssetId: "a", referenceAssetIds: ["a", "a", "b", "a"] })).toEqual(["a", "a", "b", "a"]);
    expect(() => mossEditReferenceIds({ sourceAssetId: "other", referenceAssetIds: ["a"] })).toThrow(/conflict/);
    for (const ids of [[], [""], [" a"], Array(5).fill("a")]) expect(() => mossEditReferenceIds({ referenceAssetIds: ids })).toThrow();
  });
  it("validates the exact edge/cache object rather than accepting freeform reports", () => {
    for (const options of [{ reference_max_edge: 15 }, { reference_max_edge: 4097 }, { reference_max_edge: 32.5 }, { use_kv_cache: "false" }, { executionReport: {} }]) expect(() => mossEditWireOptions(options)).toThrow();
    expect(mossEditWireOptions({ reference_max_edge: 16, use_kv_cache: false })).toEqual({ reference_max_edge: 16, use_kv_cache: false });
  });
  it("requires the edit package independently of installed DMD and respects corruption/blocked states", () => {
    const model = { conditionalPurposes: { moss_dmd_full_fp32: { installState: "installed" } } };
    expect(mossComponentState(model, { acceleration: "full_fp32" }, "edit_image").ready).toBe(false);
    for (const installState of ["missing", "incomplete", "corrupt", "installing"]) expect(mossComponentState({ conditionalPurposes: { moss_image_edit: { installState } } }, null, "edit_image").ready).toBe(false);
    expect(mossComponentState({ conditionalPurposes: { moss_image_edit: { installState: "installed", blocked: true } } }, null, "edit_image").ready).toBe(false);
    expect(mossComponentState({ conditionalPurposes: { moss_image_edit: { installState: "installed" } } }, null, "edit_image").ready).toBe(true);
  });
});
