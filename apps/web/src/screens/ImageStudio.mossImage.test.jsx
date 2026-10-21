import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, setSelect, unmountRoot } from "../testUtils/dom.js";

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async (path) =>
      path === "/api/v1/host-capabilities"
        ? { memoryGb: 128, memoryKind: "unified", platform: "macos" }
        : {},
    ),
  };
});

import { AppContext } from "../context/AppContext.js";
import { ImageStudio } from "./ImageStudio.jsx";
import { apiFetch } from "../api.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../../config/manifests/builtin.models.jsonc");
const manifestModels = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8")).models;
// Rust's real worker -> ProjectStore regression verifies this persisted recipe projection.
const BATCH_RECIPE = JSON.parse(readFileSync(resolve(HERE, "../../../../tests/fixtures/moss-batch-recipe.json"), "utf8"));
const installed = (id) => ({
  ...manifestModels.find((model) => model.id === id),
  installState: "installed",
  usable: true,
});

const TURBO = { ...installed("moss_image_sft"), conditionalPurposes: { moss_dmd_unmerged_bf16: { installState: "installed" }, moss_dmd_full_fp32: { installState: "missing" } } };
const BASE = installed("moss_image_base");
const REWRITE_T2I = installed("qwen_image_2_1_pe_t2i");
const REWRITE_I2I = installed("qwen_image_2_1_pe_i2i");
const REFS = Array.from({ length: 11 }, (_, index) => ({
  id: `ref-${index}`,
  projectId: "project_1",
  type: "image",
  displayName: `Ref ${index}`,
  status: {},
}));

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: REFS,
    characters: [],
    createImageJob: vi.fn(async () => ({ id: "job-1" })),
    createPreset: vi.fn(async (payload) => ({ id: payload.id })),
    refinePrompt: vi.fn(),
    qwenRewritePrompt: vi.fn(async () => ({ refinedPrompt: "a long descriptive prompt" })),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    imageModels: [TURBO, BASE],
    models: [TURBO, REWRITE_T2I, REWRITE_I2I],
    importAsset: vi.fn(),
    latestImageAssets: [],
    recentImageAssets: [],
    studioLaunch: null,
    imageLocalJobs: [],
    loras: [],
    jobAction: vi.fn(),
    rememberLocalGenerationJob: vi.fn(),
    setActiveView: vi.fn(),
    setPreviewAsset: vi.fn(),
    presets: [],
    requestedGpu: "",
    selectedAsset: null,
    setRequestedGpu: vi.fn(),
    updateAssetStatus: vi.fn(),
    ...overrides,
  };
}

const field = (container, labelText) => {
  const label = [...container.querySelectorAll("label")].find((node) =>
    node.textContent.trim().startsWith(labelText),
  );
  return label?.querySelector("input, select, textarea");
};
const buttonWithText = (text) =>
  [...document.body.querySelectorAll("button")].find((button) => button.textContent.includes(text));
const generateButton = () =>
  [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate");

describe("MOSS Image Studio", () => {
 let container, root;
 beforeEach(() => { global.IS_REACT_ACT_ENVIRONMENT = true; localStorage.clear(); ({root,container}=mountRoot()); });
 afterEach(async () => { await unmountRoot(root,container); vi.clearAllMocks(); });
 async function render(ctx=baseContext()) { await act(async () => root.render(<AppContext.Provider value={ctx}><ImageStudio /></AppContext.Provider>)); return ctx; }
 async function advanced() { if(!field(container,"Steps")) await click(buttonWithText("Advanced")); }
 it("exposes SFT-only default-OFF and resets on manual model switch",async()=>{
  await render(); expect(field(container,"8-step acceleration")?.checked).toBe(false);
  await click(field(container,"8-step acceleration")); expect(field(container,"8-step acceleration").checked).toBe(true);
  await act(async()=>setSelect(field(container,"Model"),BASE.id)); expect(field(container,"8-step acceleration")).toBeUndefined();
  await act(async()=>setSelect(field(container,"Model"),TURBO.id)); expect(field(container,"8-step acceleration").checked).toBe(false);
 });
 it("disables incompatible controls and restores ordinary settings on OFF",async()=>{
  await render(); await advanced(); await act(async()=>setInput(field(container,"Guidance"),"4.5"));
  await click(field(container,"8-step acceleration")); expect(field(container,"Guidance").disabled).toBe(true); expect(field(container,"Schedule shift").disabled).toBe(true); expect(field(container,"Steps").value).toBe("8");
  await click(field(container,"8-step acceleration")); expect(field(container,"Guidance").disabled).toBe(false); expect(field(container,"Guidance").value).toBe("4.5");
 });
 it("reopens a saved ON recipe, preserves ordinary choices, and treats old recipes as OFF",async()=>{
  const recipe={mode:"text_to_image",model:TURBO.id,prompt:"日本語 <tag> {\"items\":[]}",negativePrompt:"",loras:[],normalizedSettings:{width:1024,height:1024,count:1},rawAdapterSettings:{steps:6,mossImage:{acceleration:"unmerged_bf16",max_sequence_length:2048},mossOrdinarySettings:{options:{acceleration:"off",uncond_skip_layers:[]},steps:70,guidanceScale:4.5,negativePrompt:"ordinary negative",schedulerShift:6}}};
  const ctx=baseContext({studioLaunch:{id:"recipe-on",view:"Image",recipe}});await render(ctx);await advanced();expect(field(container,"8-step acceleration").checked).toBe(true);expect(field(container,"Steps").value).toBe("6");
  await click(field(container,"8-step acceleration"));expect(field(container,"Guidance").value).toBe("4.5");expect(field(container,"Steps").value).toBe("70");
  await render({...ctx,studioLaunch:{id:"old-recipe",view:"Image",recipe:{...recipe,rawAdapterSettings:{}}}});expect(field(container,"8-step acceleration").checked).toBe(false);
 });
 it("retains CFG0 negative input and transports empty layer lists through Generate and saved presets",async()=>{
  const ctx=await render(baseContext({studioLaunch:{id:"ordinary",view:"Image",recipe:{mode:"text_to_image",model:TURBO.id,prompt:"a red apple",loras:[],normalizedSettings:{},rawAdapterSettings:{}}}}));await advanced();await act(async()=>setInput(field(container,"Guidance"),"0"));
  const negative=field(container,"Negative prompt");expect(negative.disabled).toBe(false);
  await act(async()=>{const setter=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,"value").set;setter.call(negative,"unconditional 日本語");negative.dispatchEvent(new window.Event("input",{bubbles:true}));});
  await act(async()=>{const input=field(container,"Unconditional skip layers");setInput(input,"[]");input.dispatchEvent(new window.FocusEvent("focusout",{bubbles:true}));});
  await click(generateButton());expect(ctx.createImageJob).toHaveBeenCalledTimes(1);expect(ctx.createImageJob.mock.calls[0][0]).toMatchObject({negativePrompt:"unconditional 日本語",advanced:{guidanceScale:0,mossImage:{uncond_skip_layers:[]}}});
  await act(async()=>setInput(container.querySelector('input[aria-label="Preset name"]'),"Moss ordinary"));await click(buttonWithText("Save as preset"));await click([...container.querySelectorAll(".save-preset-scope-picker .compact-selector-item")].find(b=>b.textContent.includes("This project")));
  expect(ctx.createPreset.mock.calls[0][0].defaults.mossImage.uncond_skip_layers).toEqual([]);
 });
 it("submits arbitrary native CFG and SLG precision from a saved recipe without HTML step rounding",async()=>{
  const options={acceleration:"off",uncond_skip_layers:[],slg:{layers:[],scale:0.375,start:0.123,stop:0.777}};
  const ctx=await render(baseContext({studioLaunch:{id:"precision",view:"Image",recipe:{mode:"text_to_image",model:TURBO.id,prompt:"a cup",loras:[],normalizedSettings:{},rawAdapterSettings:{mossImage:options,guidanceScale:1.234}}}}));await advanced();
  expect(field(container,"Guidance").checkValidity()).toBe(true);expect(field(container,"SLG scale").checkValidity()).toBe(true);expect(field(container,"SLG start").checkValidity()).toBe(true);
  await click(generateButton());expect(ctx.createImageJob.mock.calls[0][0].advanced).toMatchObject({guidanceScale:1.234,mossImage:options});
 });
 it.each(["Unconditional skip layers", "SLG layers"])("Enter submits focused %s edits without requiring blur",async(label)=>{
  const options={acceleration:"off",uncond_skip_layers:[22],slg:{layers:[22],scale:1,start:0,stop:1}};
  const ctx=await render(baseContext({studioLaunch:{id:"focused-layers",view:"Image",recipe:{mode:"text_to_image",model:TURBO.id,prompt:"a cup",loras:[],normalizedSettings:{count:1},rawAdapterSettings:{mossImage:options}}}}));await advanced();
  const input=field(container,label);input.focus();
  await act(async()=>setInput(input,"["));expect(document.activeElement).toBe(input);expect(input.value).toBe("[");
  await act(async()=>setInput(input,"[]"));expect(document.activeElement).toBe(input);
  // jsdom does not implement the browser's implicit Enter submission; deliver
  // that default form action while the edited input stays focused, without blur.
  await act(async()=>{input.dispatchEvent(new window.KeyboardEvent("keydown",{key:"Enter",bubbles:true}));input.form.requestSubmit();});
  expect(document.activeElement).toBe(input);expect(ctx.createImageJob).toHaveBeenCalledTimes(1);
  const sent=ctx.createImageJob.mock.calls[0][0].advanced.mossImage;
  expect(label==="SLG layers"?sent.slg.layers:sent.uncond_skip_layers).toEqual([]);
 });
 it("persists acceleration choice across a studio remount",async()=>{
  await render();await click(field(container,"8-step acceleration"));await unmountRoot(root,container);({root,container}=mountRoot());await render();expect(field(container,"8-step acceleration").checked).toBe(true);
 });
 it("saves, selects and deselects a DMD preset without losing authored ordinary controls",async()=>{
  const options={acceleration:"off",max_sequence_length:1536,training_shift:2,uncond_skip_layers:[],slg:{layers:[],scale:0.375,start:0.123,stop:0.777}};
  const recipe={mode:"text_to_image",model:TURBO.id,prompt:"a cup",negativePrompt:"saved negative",loras:[],normalizedSettings:{count:2},rawAdapterSettings:{mossImage:options,steps:70,guidanceScale:4.5,schedulerShift:6}};
  const ctx=await render(baseContext({studioLaunch:{id:"saved-authored",view:"Image",recipe}}));await advanced();await click(field(container,"8-step acceleration"));
  await act(async()=>setInput(container.querySelector('input[aria-label="Preset name"]'),"Moss authored"));await click(buttonWithText("Save as preset"));await click([...container.querySelectorAll(".save-preset-scope-picker .compact-selector-item")].find(b=>b.textContent.includes("This project")));
  const preset=ctx.createPreset.mock.calls[0][0];
  await render({...ctx,presets:[preset]});await click(container.querySelector(".preset-chips button"));
  const previous={...recipe,negativePrompt:"previous negative",rawAdapterSettings:{mossImage:{acceleration:"off",max_sequence_length:512,uncond_skip_layers:[3]},steps:60,guidanceScale:2,schedulerShift:4}};
  await render({...ctx,presets:[preset],studioLaunch:{id:"previous-authored",view:"Image",recipe:previous}});
  await click(buttonWithText("Moss authored"));expect(field(container,"8-step acceleration").checked).toBe(true);expect(field(container,"Context limit").value).toBe("1536");expect(field(container,"Training shift").value).toBe("2");expect(field(container,"Unconditional skip layers").value).toBe("[]");expect(field(container,"SLG layers").value).toBe("[]");
  await click(container.querySelector(".preset-chips button"));expect(field(container,"8-step acceleration").checked).toBe(false);expect(field(container,"Context limit").value).toBe("512");expect(field(container,"Steps").value).toBe("60");expect(field(container,"Guidance").value).toBe("2");expect(field(container,"Schedule shift").value).toBe("4");expect(field(container,"Negative prompt").value).toBe("previous negative");expect(field(container,"Unconditional skip layers").value).toBe("[3]");
  await click(buttonWithText("Moss authored"));await click(field(container,"8-step acceleration"));expect(field(container,"Steps").value).toBe("70");expect(field(container,"Guidance").value).toBe("4.5");expect(field(container,"Schedule shift").value).toBe("6");expect(field(container,"Negative prompt").value).toBe("saved negative");
  await click(generateButton());expect(ctx.createImageJob.mock.calls.at(-1)[0].advanced.mossImage).toEqual(options);
  await act(async()=>setSelect(field(container,"Model"),BASE.id));
  await render({...ctx,presets:[preset],studioLaunch:{id:"cross-model-preset",view:"Image",presetId:preset.id,presetModel:TURBO.id,presetMode:"text_to_image"}});
  expect(field(container,"Model").value).toBe(TURBO.id);expect(field(container,"8-step acceleration").checked).toBe(true);expect(field(container,"Context limit").value).toBe("1536");
  await click(generateButton());expect(ctx.createImageJob.mock.calls.at(-1)[0].advanced).toMatchObject({steps:8,mossImage:{acceleration:"unmerged_bf16",max_sequence_length:1536}});
  await act(async()=>setSelect(field(container,"Model"),BASE.id));await act(async()=>setSelect(field(container,"Model"),TURBO.id));expect(field(container,"8-step acceleration").checked).toBe(false);
  await render({...ctx,presets:[preset],studioLaunch:{id:"recipe-after-preset",view:"Image",recipe:previous}});expect(field(container,"8-step acceleration").checked).toBe(false);expect(field(container,"Context limit").value).toBe("512");expect(field(container,"Steps").value).toBe("60");
 });
 it.each([2,6])("replays a persisted two-image-per-prompt recipe and splits %i prompts without multiplying variations",async(size)=>{
  const ctx=await render(baseContext({studioLaunch:{id:"worker-batch",view:"Image",recipe:BATCH_RECIPE}}));
  expect(field(container,"Variations").value).toBe("2");
  const prompts=size===2?BATCH_RECIPE.rawAdapterSettings.promptBatch:Array.from({length:size},(_,i)=>`prompt ${i}`);
  if(size>2) await act(async()=>{const input=container.querySelector('textarea[aria-label="Batch prompts"]');Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,"value").set.call(input,prompts.join("\n"));input.dispatchEvent(new window.Event("input",{bubbles:true}));});
  await click(buttonWithText("Run batch"));if(container.querySelector(".batch-confirm .prompt-cta")) await click(container.querySelector(".batch-confirm .prompt-cta"));
  await vi.waitFor(()=>expect(ctx.createImageJob).toHaveBeenCalledTimes(size===2?1:2));
  const jobs=ctx.createImageJob.mock.calls.map(([job])=>job);expect(jobs.map(job=>job.count)).toEqual(size===2?[2]:[2,2]);expect(jobs.flatMap(job=>job.prompts)).toEqual(prompts);expect(jobs.map(job=>job.count*job.prompts.length)).toEqual(size===2?[4]:[8,4]);
 });
 it("leaves a previously opened batch when replaying one selected image",async()=>{
  const ctx=await render(baseContext({studioLaunch:{id:"whole-batch",view:"Image",recipe:BATCH_RECIPE}}));
  expect(container.querySelector('textarea[aria-label="Batch prompts"]')).not.toBeNull();
  const recipe={...BATCH_RECIPE,prompt:"dog",normalizedSettings:{...BATCH_RECIPE.normalizedSettings,count:1},rawAdapterSettings:{mossImage:{acceleration:"off"}}};
  await render({...ctx,studioLaunch:{id:"selected-image",view:"Image",recipe,replaySeed:102}});
  expect(container.querySelector('textarea[aria-label="Batch prompts"]')).toBeNull();
  await click(generateButton());expect(ctx.createImageJob.mock.calls[0][0]).toMatchObject({prompt:"dog",count:1,seed:102});
  expect(ctx.createImageJob.mock.calls[0][0].prompts).toBeUndefined();
 });
 it("gates optional precision packages and reports download errors",async()=>{
  await render(); await click(field(container,"8-step acceleration")); await advanced(); await act(async()=>setSelect(field(container,"Acceleration precision"),"full_fp32"));
  expect(generateButton().disabled).toBe(true); apiFetch.mockRejectedValueOnce(new Error("component acquisition failed")); await click(buttonWithText("Download 8-step components")); expect(container.textContent).toContain("component acquisition failed");
  expect(apiFetch).toHaveBeenLastCalledWith(expect.stringContaining("moss_dmd_full_fp32/download"),"test-token",expect.objectContaining({method:"POST"}));
 });
});
