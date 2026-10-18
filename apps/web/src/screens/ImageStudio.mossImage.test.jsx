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
 it("persists acceleration choice across a studio remount",async()=>{
  await render();await click(field(container,"8-step acceleration"));await unmountRoot(root,container);({root,container}=mountRoot());await render();expect(field(container,"8-step acceleration").checked).toBe(true);
 });
 it("gates optional precision packages and reports download errors",async()=>{
  await render(); await click(field(container,"8-step acceleration")); await advanced(); await act(async()=>setSelect(field(container,"Acceleration precision"),"full_fp32"));
  expect(generateButton().disabled).toBe(true); apiFetch.mockRejectedValueOnce(new Error("component acquisition failed")); await click(buttonWithText("Download 8-step components")); expect(container.textContent).toContain("component acquisition failed");
  expect(apiFetch).toHaveBeenLastCalledWith(expect.stringContaining("moss_dmd_full_fp32/download"),"test-token",expect.objectContaining({method:"POST"}));
 });
});
