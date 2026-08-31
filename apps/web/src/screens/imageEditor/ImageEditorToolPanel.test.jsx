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
  it("offers connected/global selection, adjustable perceptual edge controls, and one apply action", async () => {
    const setColorKeyGlobal = vi.fn();
    const applyColorKey = vi.fn();
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
          applyColorKey,
          setTool: vi.fn(),
        }}
      />,
    ));
    expect(container.textContent).toContain("Click a background pixel on the active layer");
    expect(container.textContent).toContain("Connected preserves separate same-color subject areas");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Global")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Apply cutout")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(setColorKeyGlobal).toHaveBeenCalledWith(true);
    expect(container.querySelector("input[aria-label='Color-key tolerance']").value).toBe("12");
    expect(container.querySelector("input[aria-label='Color-key softness']").value).toBe("8");
    expect(applyColorKey).toHaveBeenCalledTimes(1);
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
});
