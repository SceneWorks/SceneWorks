import { describe, expect, it } from "vitest";
import { buildImageJobRequest } from "./imageJobRequest.js";
import { mossDefaults, mossWireOptions, mossComponentState } from "./mossImage.js";
const state = (more = {}) => ({ model: "moss_image_sft", mode: "text_to_image", promptToSend: '日本語 <tag> {"items": [{"yxyx": [0,0,1,1]}]}', resolution: "1024x1024", width: 1024, height: 1024, count: 2, seed: "42", posePayload: [], referenceAssetIds: [], loras: [], sampler: "default", scheduler: "default", schedulerShift: 5, supportsSchedulerShift: true, guidanceOverride: 3, stepsOverride: 50, guidanceMethod: "cfg", negativePrompt: "", flashAttn: true, ...more });
describe("MOSS product transport", () => {
 it("preserves exact multilingual/tag/JSON prompt and explicit OFF for normal/batch", () => { const input=state(), normal=buildImageJobRequest(input); expect(buildImageJobRequest({...input})).toEqual(normal); expect(normal.prompt).toBe(input.promptToSend); expect(normal.advanced.mossImage).toEqual(mossDefaults()); expect(normal.advanced.steps).toBe(50); });
 it.each(["unmerged_bf16", "full_fp32"])("transports %s without incompatible hidden controls", (acceleration) => { const result=buildImageJobRequest(state({mossImage:{acceleration,uncond_skip_layers:[],training_shift:1,slg:{layers:[22],scale:1,start:0,stop:1}},mossDmdSteps:6})); expect(result.advanced.mossImage).toEqual({acceleration,max_sequence_length:4096}); expect(result.advanced.steps).toBe(6); for(const key of ["sampler","scheduler","guidanceScale","schedulerShift","guidanceMethod"]) expect(result.advanced).not.toHaveProperty(key); });
 it("preserves [] versus null and never derives acceleration from installed assets", () => { expect(mossWireOptions({uncond_skip_layers:[]}).uncond_skip_layers).toEqual([]); expect(mossWireOptions({uncond_skip_layers:null}).uncond_skip_layers).toBeNull(); expect(mossComponentState({}, {acceleration:"off"}).ready).toBe(true); expect(mossComponentState({}, {acceleration:"full_fp32"}).ready).toBe(false); });
 it("keeps CFG0 negative and exact empty/null layer selections in normal and batch payloads", () => {
  for(const layers of [[],null]) { const input=state({guidanceOverride:0,negativePrompt:"unconditional 日本語",mossImage:{acceleration:"off",uncond_skip_layers:layers,slg:{layers:[],scale:0.5,start:0,stop:1}}});
   const result=buildImageJobRequest(input); expect(result.negativePrompt).toBe(input.negativePrompt); expect(result.advanced.guidanceScale).toBe(0); expect(result.advanced.mossImage.uncond_skip_layers).toEqual(layers); expect(result.advanced.mossImage.slg.layers).toEqual([]); expect(buildImageJobRequest({...input,promptToSend:"batch row"}).advanced).toEqual(result.advanced);
  }
 });
 it("never emits stale MOSS options for another model", () => { expect(buildImageJobRequest(state({model:"iris_3b",mossImage:{acceleration:"full_fp32"}})).advanced).not.toHaveProperty("mossImage"); });
});
