import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { mountRoot, unmountRoot } from "../../testUtils/dom.js";
import {
  ImageEditorEditPanel,
  ImageEditorToolPanel,
  setImageEditorToolPanelRenderObserverForTests,
  useStableImageEditorToolPanelScope,
} from "./ImageEditorToolPanel.jsx";
import { EDIT_PROMPT_TEMPLATES } from "../../data/editPromptTemplates.js";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import JSON5 from "json5";
import { maxReferencesForModel } from "../../imageReferenceLimits.js";
import { editCapableModels } from "../../imageJobs.js";

let container;
let root;
beforeEach(() => ({ container, root } = mountRoot()));
afterEach(async () => {
  setImageEditorToolPanelRenderObserverForTests(null);
  await unmountRoot(root, container);
});

describe("ImageEditorToolPanel memo boundary", () => {
  it("skips parent-only rerenders while a rendered handler sees its latest closure", async () => {
    const observer = vi.fn();
    const fitCalls = [];
    const working = { width: 640, height: 480 };
    setImageEditorToolPanelRenderObserverForTests(observer);
    function ImageEditorParentHarness() {
      const [stageWidth, setStageWidth] = React.useState(100);
      const scope = {
        working,
        layerCount: 1,
        fitToView: () => fitCalls.push(stageWidth),
        actualSize: () => {},
      };
      const stableScope = useStableImageEditorToolPanelScope(scope);
      return (
        <>
          <button data-action="parent-update" onClick={() => setStageWidth(200)} type="button">
            resize parent
          </button>
          <ImageEditorToolPanel panelKey="move" scope={stableScope} />
        </>
      );
    }

    await act(async () => root.render(<ImageEditorParentHarness />));
    await act(async () => {
      container.querySelector("[data-action='parent-update']")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(observer).toHaveBeenCalledTimes(1);
    await act(async () => {
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "Fit to view")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(fitCalls).toEqual([200]);
  });
});

describe("ImageEditorToolPanel color-key cutout", () => {
  it("offers connected/global selection, shared refinement, and one apply action", async () => {
    const setColorKeyGlobal = vi.fn();
    const applyMaskCutout = vi.fn();
    const refineMask = vi.fn();
    await act(async () => root.render(
      <ImageEditorToolPanel
        panelKey="cutout"
        scope={{
          colorKeyGlobal: false,
          colorKeySeed: { x: 7, y: 9 },
          colorKeySoftness: 8,
          colorKeyTolerance: 12,
          setColorKeyGlobal,
          setColorKeySeed: vi.fn(),
          setColorKeySoftness: vi.fn(),
          setColorKeyTolerance: vi.fn(),
          applyMaskCutout,
          clearMask: vi.fn(),
          cutoutKeepSelected: false,
          maskBaseImage: {},
          maskBrush: 40,
          maskErase: false,
          maskHasContent: () => false,
          maskLines: [],
          maskMode: true,
          maskRefineRadius: 6,
          maskSource: "colorKey",
          maskSubTool: "brush",
          refineMask,
          setTool: vi.fn(),
          setCutoutKeepSelected: vi.fn(),
          setMaskBrush: vi.fn(),
          setMaskErase: vi.fn(),
          setMaskMode: vi.fn(),
          setMaskRefineRadius: vi.fn(),
          setMaskSubTool: vi.fn(),
        }}
      />,
    ));
    expect(container.textContent).toContain("Click a background pixel on the active layer");
    expect(container.textContent).toContain("Connected preserves separate same-color subject areas");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Global")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Invert")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Apply cutout")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(setColorKeyGlobal).toHaveBeenCalledWith(true);
    expect(container.querySelector("input[aria-label='Color-key tolerance']").value).toBe("12");
    expect(container.querySelector("input[aria-label='Color-key softness']").value).toBe("8");
    expect(refineMask).toHaveBeenCalledWith("invert");
    expect(applyMaskCutout).toHaveBeenCalledTimes(1);
  });

  it("keeps color key actionable when SAM3 is unavailable", async () => {
    await act(async () => root.render(
      <ImageEditorToolPanel
        panelKey="cutout"
        scope={{
          colorKeyGlobal: false,
          colorKeySeed: { x: 7, y: 9 },
          colorKeySoftness: 8,
          colorKeyTolerance: 12,
          setColorKeyGlobal: vi.fn(),
          setColorKeySeed: vi.fn(),
          setColorKeySoftness: vi.fn(),
          setColorKeyTolerance: vi.fn(),
          applyMaskCutout: vi.fn(),
          clearMask: vi.fn(),
          smartSelectSupported: false,
          setTool: vi.fn(),
        }}
      />,
    ));
    expect(container.textContent).toContain("SAM3 object selection is unavailable on this worker");
    expect(container.textContent).toContain("Color key remains available above");
    expect([...container.querySelectorAll("button")].find((button) => button.textContent === "Global")).toBeTruthy();
    expect(container.querySelector("input[aria-label='Color-key tolerance']")).toBeTruthy();
    expect(container.textContent).not.toContain("Smart select");
  });

  it("offers an actionable SAM3 install when the worker supports segmentation but the model is missing", async () => {
    const requestSmartSelectDownload = vi.fn();
    await act(async () => root.render(
      <ImageEditorToolPanel
        panelKey="cutout"
        scope={{
          applyMaskCutout: vi.fn(),
          clearMask: vi.fn(),
          colorKeyGlobal: false,
          colorKeySeed: null,
          colorKeySoftness: 8,
          colorKeyTolerance: 12,
          requestSmartSelectDownload,
          setColorKeyGlobal: vi.fn(),
          setColorKeySeed: vi.fn(),
          setColorKeySoftness: vi.fn(),
          setColorKeyTolerance: vi.fn(),
          setTool: vi.fn(),
          smartSelectCapabilitySupported: true,
          smartSelectDownloadRequested: false,
          smartSelectModel: { id: "sam3_person_segment", installState: "missing" },
          smartSelectSupported: false,
        }}
      />,
    ));
    expect(container.textContent).toContain("supported on this worker but is not installed");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Install SAM3")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(requestSmartSelectDownload).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain("Color key");
  });

  it("offers SAM3 keep/remove selection cutouts independently of AI Edit", async () => {
    const setMaskMode = vi.fn();
    const setMaskSubTool = vi.fn();
    const setCutoutKeepSelected = vi.fn();
    const applyMaskCutout = vi.fn();
    await act(async () => root.render(
      <ImageEditorToolPanel
        panelKey="cutout"
        scope={{
          aiOp: null,
          applyMaskCutout,
          clearMask: vi.fn(),
          colorKeyGlobal: false,
          colorKeySeed: null,
          colorKeySoftness: 8,
          colorKeyTolerance: 12,
          cutoutKeepSelected: true,
          maskBaseImage: {},
          maskBrush: 40,
          maskErase: false,
          maskHasContent: () => false,
          maskLines: [],
          maskMode: true,
          maskRefineRadius: 6,
          maskSource: "smartSelect",
          maskSubTool: "select",
          refineMask: vi.fn(),
          setColorKeyGlobal: vi.fn(),
          setColorKeySeed: vi.fn(),
          setColorKeySoftness: vi.fn(),
          setColorKeyTolerance: vi.fn(),
          setCutoutKeepSelected,
          setMaskBrush: vi.fn(),
          setMaskErase: vi.fn(),
          setMaskMode,
          setMaskRefineRadius: vi.fn(),
          setMaskSubTool,
          smartSelectSupported: true,
        }}
      />,
    ));
    expect(container.textContent).toContain("works without choosing an AI Edit model");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Smart select")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Remove selected")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Apply cutout")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(setMaskMode).toHaveBeenCalledWith(true);
    expect(setMaskSubTool).toHaveBeenCalledWith("select");
    expect(setCutoutKeepSelected).toHaveBeenCalledWith(false);
    expect(applyMaskCutout).toHaveBeenCalledTimes(1);
  });
});

describe("ImageEditorEditPanel quick edit instructions", () => {
  const model = { id: "z_image_edit", name: "Z-Image-Edit" };

  function editScope(overrides) {
    return {
      EDIT_OUTPUT_ASPECTS: [{ key: "match", label: "Match" }],
      EditorLoraPanel: () => null,
      FitModeControl: () => null,
      MAX_EDIT_REFERENCES: 3,
      StudioUpdateBadge: () => null,
      StudioUpdateNotice: () => null,
      aiOp: null,
      assetUrl: () => "",
      canMask: false,
      clearMask: () => {},
      createLoraDownloadJob: () => {},
      createModelDownloadJob: () => {},
      editAspect: "match",
      editFitMode: "contain",
      editGuidance: "",
      editLora: null,
      editLoraDownloadRequested: false,
      editLoraInstalled: false,
      editLoraRequiredMissing: false,
      editLoraSelection: { selectedLoraIds: [], toggleLora: () => {}, weightFor: () => 1, setWeight: () => {} },
      editModel: model.id,
      editModels: [model],
      editPrompt: "",
      editSeed: "",
      editorPickerLoras: [],
      effectiveFitMode: (value) => value,
      guidanceDefaultFromModel: () => null,
      imageAssets: [],
      maskActive: false,
      maskBaseImage: null,
      maskBrush: 40,
      maskErase: false,
      maskHasContent: () => false,
      maskLines: [],
      maskMode: false,
      maskRefineRadius: 4,
      maskSubTool: "brush",
      multiRefCapable: false,
      refAssetIds: [],
      refineMask: () => {},
      requestEditLoraDownload: () => {},
      runEdit: () => {},
      selectedEditLoras: [],
      selectedEditModel: model,
      setEditAspect: () => {},
      setEditFitMode: () => {},
      setEditGuidance: () => {},
      setEditModel: () => {},
      setEditPrompt: () => {},
      setEditSeed: () => {},
      setMaskBrush: () => {},
      setMaskErase: () => {},
      setMaskMode: () => {},
      setMaskRefineRadius: () => {},
      setMaskSubTool: () => {},
      setRefAssetIds: () => {},
      setRefPickerOpen: () => {},
      setShowIncompatibleEditLoras: () => {},
      showIncompatibleEditLoras: false,
      smartSelectSupported: false,
      updateOptionLabel: (entry) => entry.name,
      ...overrides,
    };
  }

  it("offers every built-in recipe and writes its full instruction into the box", async () => {
    const setEditPrompt = vi.fn();
    await act(async () => root.render(<ImageEditorEditPanel scope={editScope({ setEditPrompt })} />));

    const pills = [...container.querySelectorAll(".ie-chip-row .ie-chip")];
    expect(pills.map((pill) => pill.textContent)).toEqual(EDIT_PROMPT_TEMPLATES.map((template) => template.label));

    await act(async () => {
      pills.find((pill) => pill.textContent === "Fix colors")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(setEditPrompt).toHaveBeenCalledWith("correct the colors of this photo");
  });
  it("renders all MOSS edit controls and repeated ordinal slots while gating a missing adapter",async()=>{
    const model={id:"moss_image_sft",name:"MOSS SFT",conditionalPurposes:{moss_image_edit:{installState:"missing"}}};
    const setMossImageEdit=vi.fn(),setMossEditSettings=vi.fn(),setRefAssetIds=vi.fn();
    const scope=editScope({editModel:model.id,editModels:[model],selectedEditModel:model,mossEdit:true,mossEditReady:false,mossEditError:"",mossImageEdit:{reference_max_edge:768,use_kv_cache:false},mossEditSettings:{width:512,height:768,steps:41,guidanceScale:4.5,schedulerShift:6,max_sequence_length:2048,negativePrompt:"blur"},setMossImageEdit,setMossEditSettings,setRefAssetIds,refAssetIds:["a","a","b"],imageAssets:[{id:"a",displayName:"Wide"},{id:"b",displayName:"Tall"}],editPrompt:"blue cup",multiRefCapable:true});
    await act(async()=>root.render(<ImageEditorEditPanel scope={scope}/>));
    expect(container.textContent).toContain("working image is Image 1");
    expect(container.textContent).toContain("Ordered MOSS references (2–4)");
    expect([...container.querySelectorAll("label")].find((label)=>label.textContent==="Reference maximum edge").querySelector("input").value).toBe("768");
    expect([...container.querySelectorAll("button")].find((button)=>button.textContent==="Generate edit").disabled).toBe(true);
    await act(async()=>container.querySelector('button[aria-label="Move Image 4 earlier"]').click());expect(setRefAssetIds).toHaveBeenCalledWith(["a","b","a"]);
    await act(async()=>root.render(<ImageEditorEditPanel scope={{...scope,mossEditReady:true,selectedEditModel:{...model,conditionalPurposes:{moss_image_edit:{installState:"installed"}}}}}/>));
    expect([...container.querySelectorAll("button")].find((button)=>button.textContent==="Generate edit").disabled).toBe(false);
  });

  // sc-24114: the editor exposes the Studio's transparency toggle for an alpha-capable model, and
  // offers the prompt wording when it is on. *Mutation that reds this:* removing the toggle.
  it("offers the transparency toggle and prompt hint for an alpha-capable edit model", async () => {
    const setEditTransparent = vi.fn();
    const setEditPrompt = vi.fn();
    const alphaModel = { id: "qwen_image_2_1", name: "Qwen Image 2.1", supportsAlphaOutput: true };
    await act(async () =>
      root.render(
        <ImageEditorEditPanel
          scope={editScope({
            editModel: alphaModel.id,
            editModels: [alphaModel],
            selectedEditModel: alphaModel,
            editPrompt: "remove the background",
            editTransparent: true,
            setEditTransparent,
            setEditPrompt,
          })}
        />,
      ),
    );
    const toggle = container.querySelector(".transparency-toggle input");
    expect(toggle?.checked).toBe(true);
    await act(async () => toggle.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(setEditTransparent).toHaveBeenCalledWith(false);
    const hint = container.querySelector(".transparency-prompt-hint button");
    await act(async () => hint.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(setEditPrompt.mock.calls.at(-1)[0]).toContain("The background is transparent.");

    const opaqueModel = { id: "flux2_dev", name: "FLUX.2" };
    await act(async () =>
      root.render(
        <ImageEditorEditPanel
          scope={editScope({ editModel: opaqueModel.id, editModels: [opaqueModel], selectedEditModel: opaqueModel })}
        />,
      ),
    );
    expect(container.querySelector(".transparency-toggle")).toBeNull();
  });

  // sc-24110: the SHIPPED Qwen Image 2.1 entry must open the reference rail, at its declared cap.
  // `multiRefCapable` / `maxEditReferences` are derived exactly as ImageEditor.jsx derives them
  // (`ui.multiReference`, `maxReferencesForModel(model, MAX_EDIT_REFERENCES)`), from the manifest
  // on disk — so dropping the flag, or the `limits.maxReferenceAssets` key, hides or re-caps it.
  it("opens the ordered reference rail for Qwen Image 2.1 with room for nine references beside the working image", async () => {
    const manifest = JSON5.parse(
      readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "../../../../../config/manifests/builtin.models.jsonc"), "utf8"),
    );
    const qwen = manifest.models.find((entry) => entry.id === "qwen_image_2_1");
    const scopeFor = (refAssetIds) =>
      editScope({
        editModel: qwen.id,
        editModels: [qwen],
        selectedEditModel: qwen,
        multiRefCapable: Boolean(qwen.ui?.multiReference),
        // The fallback (ImageEditor's MAX_EDIT_REFERENCES, 4) applies only to a model with no
        // declared cap — importing it here would pull the canvas stack into jsdom.
        maxEditReferences: maxReferencesForModel(qwen, 4),
        refAssetIds,
      });
    const addButton = () => container.querySelector(".ie-ref-add");

    const eight = Array.from({ length: 8 }, (_, index) => `ref_${index}`);
    await act(async () => root.render(<ImageEditorEditPanel scope={scopeFor(eight)} />));
    expect(addButton(), "the reference rail is rendered for this model").not.toBeNull();
    expect(addButton().disabled, "eight references + the working image is nine of ten").toBe(false);
    // sc-24114: the working image leads the ordered list the edit sends, so it is "Image 1" and
    // the first ATTACHED reference is "Image 2" — the numbering the engine's template uses.
    // *Mutation that reds this:* labelling attached references `referenceOrdinalLabel(index)`.
    const ordinals = [...container.querySelectorAll(".ie-ref-ordinal")].map((node) => node.textContent);
    expect(ordinals).toEqual(["Image 1", ...eight.map((_, index) => `Image ${index + 2}`)]);
    expect(container.querySelector(".ie-ref-working .ie-ref-ordinal")?.textContent).toBe("Image 1");

    const nine = Array.from({ length: 9 }, (_, index) => `ref_${index}`);
    await act(async () => root.render(<ImageEditorEditPanel scope={scopeFor(nine)} />));
    expect(addButton().disabled, "nine references + the working image is the cap of ten").toBe(true);
  });

  // sc-25908: Turbo serves the editor exactly as base does — it is an edit model, the rail opens
  // at ITS declared cap, and the transparency toggle follows ITS `supportsAlphaOutput`. Every value
  // is read from the shipped Turbo entry, derived the way ImageEditor.jsx derives it.
  // *Mutation that reds this:* dropping `edit_image`, `ui.multiReference` or `supportsAlphaOutput`
  // from the Turbo manifest entry.
  it("opens the reference rail and the transparency toggle for Qwen Image 2.1 Turbo", async () => {
    const manifest = JSON5.parse(
      readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "../../../../../config/manifests/builtin.models.jsonc"), "utf8"),
    );
    const turbo = manifest.models.find((entry) => entry.id === "qwen_image_2_1_turbo");
    expect(editCapableModels(manifest.models).map((entry) => entry.id)).toContain("qwen_image_2_1_turbo");
    const scopeFor = (refAssetIds) =>
      editScope({
        editModel: turbo.id,
        editModels: [turbo],
        selectedEditModel: turbo,
        multiRefCapable: Boolean(turbo.ui?.multiReference),
        maxEditReferences: maxReferencesForModel(turbo, 4),
        refAssetIds,
        editPrompt: "isolate the cyclist",
        editTransparent: true,
      });
    const nine = Array.from({ length: 9 }, (_, index) => `ref_${index}`);
    await act(async () => root.render(<ImageEditorEditPanel scope={scopeFor(nine.slice(0, 8))} />));
    expect(container.querySelector(".ie-ref-add")?.disabled).toBe(false);
    expect(container.querySelector(".transparency-toggle input")?.checked).toBe(true);
    expect(container.querySelector(".transparency-prompt-hint button")).not.toBeNull();
    await act(async () => root.render(<ImageEditorEditPanel scope={scopeFor(nine)} />));
    expect(container.querySelector(".ie-ref-add").disabled, "nine + the working image is ten").toBe(true);
  });
});

describe("ImageEditorToolPanel upscale tool with Iris restoration (sc-25683)", () => {
  afterEach(() => vi.restoreAllMocks());

  it("swaps the factor buttons for the Iris controls + planner preview and gates Restore on the plan", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ items: [{ assetId: null, plan: { summary: "4×: 640x480 → 2560x1920 (processed at 2560x1920 in 12 tiles)" }, error: null }] }),
    });
    const { IrisRestoreControls } = await import("../../components/IrisRestoreControls.jsx");
    const runUpscale = vi.fn();
    function Harness() {
      const [irisRestore, setIrisRestore] = React.useState({ scale: 4, inputSizing: "budgeted", colorFix: true });
      const [irisPlanStatus, setIrisPlanStatus] = React.useState({ ready: false, blocked: false, error: null });
      return (
        <ImageEditorToolPanel
          panelKey="upscale"
          scope={{
            IrisRestoreControls,
            UPSCALE_ENGINE_DESC: {},
            aiOp: null,
            availableUpscaleEngines: [
              { key: "real-esrgan", label: "Real-ESRGAN", factors: [2, 4] },
              { key: "iris", label: "Iris restore", factors: [4], restoration: true },
            ],
            irisPlanStatus,
            irisRestore,
            runUpscale,
            setIrisPlanStatus,
            setIrisRestore,
            setTool: vi.fn(),
            setUpscaleEngine: vi.fn(),
            setUpscaleFactor: vi.fn(),
            setUpscaleSoftness: vi.fn(),
            token: "t",
            upscaleEngine: "iris",
            upscaleEngineHasSoftness: () => false,
            upscaleEngineIsRestoration: (key) => key === "iris",
            upscaleFactor: 4,
            upscaleFactorsForEngine: () => [4],
            upscaleSoftness: 0,
            working: { width: 640, height: 480 },
          }}
        />
      );
    }
    await act(async () => root.render(<Harness />));
    const restore = () => Array.from(container.querySelectorAll("button")).find((b) => b.textContent === "Restore image");
    expect(container.textContent).not.toContain("Scale factor");
    expect(restore().disabled).toBe(true);
    await act(async () => new Promise((resolve) => setTimeout(resolve, 260)));
    expect(container.querySelector(".iris-restore-preview").textContent).toContain("4×: 640x480 → 2560x1920");
    expect(restore().disabled).toBe(false);
    await act(async () => restore().dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(runUpscale).toHaveBeenCalledTimes(1);
  });
});
