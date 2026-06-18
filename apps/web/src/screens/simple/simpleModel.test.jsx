import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  SIMPLE_IMAGE_MODEL_KEY,
  SIMPLE_VIDEO_MODEL_KEY,
  defaultImageModelId,
  defaultModelId,
  modelLabel,
  textToImageModels,
  useSimpleImageModel,
  useSimpleVideoModel,
} from "./simpleModel.js";
import { AppContext } from "../../context/AppContext.js";

const t2i = (id) => ({ id, capabilities: ["text_to_image"] });
const MODELS = [
  { id: "instantid_realvisxl", capabilities: ["character_image"] },
  { id: "z_image_edit", capabilities: ["edit_image"] },
  t2i("z_image_turbo"),
  t2i("realvisxl"),
  t2i("sdxl"),
];

describe("simpleModel helpers", () => {
  it("keeps only text-to-image models (drops edit/identity)", () => {
    expect(textToImageModels(MODELS).map((m) => m.id)).toEqual(["z_image_turbo", "realvisxl", "sdxl"]);
  });

  it("defaults to SDXL when installed, by preference order", () => {
    expect(defaultImageModelId(textToImageModels(MODELS))).toBe("sdxl");
  });

  it("falls back down the preference order, then to the first model", () => {
    expect(defaultImageModelId([t2i("realvisxl"), t2i("z_image_turbo")])).toBe("realvisxl");
    expect(defaultImageModelId([t2i("z_image_turbo")])).toBe("z_image_turbo");
    expect(defaultImageModelId([{ id: "mystery", capabilities: ["text_to_image"] }])).toBe("mystery");
  });

  it("labels prefer ui.label then name then id", () => {
    expect(modelLabel({ id: "sdxl", ui: { label: "Stable Diffusion XL" } })).toBe("Stable Diffusion XL");
    expect(modelLabel({ id: "x", name: "X Model" })).toBe("X Model");
    expect(modelLabel({ id: "raw" })).toBe("raw");
  });
});

let container;
let root;

beforeEach(() => {
  localStorage.clear();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  localStorage.clear();
});

function Harness() {
  const { model, modelId, select, makeDefault, isDefault } = useSimpleImageModel();
  return (
    <div>
      <span data-testid="id">{modelId ?? ""}</span>
      <span data-testid="label">{model ? modelLabel(model) : ""}</span>
      <span data-testid="isDefault">{String(isDefault)}</span>
      <button onClick={() => select("realvisxl")}>pick-realvis</button>
      <button onClick={makeDefault}>make-default</button>
    </div>
  );
}

function render() {
  return act(() => {
    root.render(
      <AppContext.Provider value={{ imageModels: MODELS }}>
        <Harness />
      </AppContext.Provider>,
    );
  });
}

const text = (id) => container.querySelector(`[data-testid="${id}"]`).textContent;
const click = (label) => act(() => [...container.querySelectorAll("button")].find((b) => b.textContent === label).dispatchEvent(new window.MouseEvent("click", { bubbles: true })));

describe("useSimpleImageModel", () => {
  it("starts on the SDXL default and selection is session-only until pinned", async () => {
    await render();
    expect(text("id")).toBe("sdxl");
    expect(text("isDefault")).toBe("false");

    await click("pick-realvis");
    expect(text("id")).toBe("realvisxl");
    // not persisted yet
    expect(localStorage.getItem(SIMPLE_IMAGE_MODEL_KEY)).toBe(null);

    await click("make-default");
    expect(localStorage.getItem(SIMPLE_IMAGE_MODEL_KEY)).toBe("realvisxl");
    expect(text("isDefault")).toBe("true");
  });

  it("rehydrates the saved default on mount", async () => {
    localStorage.setItem(SIMPLE_IMAGE_MODEL_KEY, "z_image_turbo");
    await render();
    expect(text("id")).toBe("z_image_turbo");
    expect(text("isDefault")).toBe("true");
  });
});

describe("defaultModelId (generic)", () => {
  it("honors the preference order, then falls back to the first", () => {
    const vids = [{ id: "other_video" }, { id: "ltx_2_3" }];
    expect(defaultModelId(vids, ["ltx_2_3"])).toBe("ltx_2_3");
    expect(defaultModelId([{ id: "other_video" }], ["ltx_2_3"])).toBe("other_video");
    expect(defaultModelId([], ["ltx_2_3"])).toBe(null);
  });
});

function VideoHarness() {
  const { modelId, makeDefault, isDefault } = useSimpleVideoModel();
  return (
    <div>
      <span data-testid="id">{modelId ?? ""}</span>
      <span data-testid="isDefault">{String(isDefault)}</span>
      <button onClick={makeDefault}>make-default</button>
    </div>
  );
}

describe("useSimpleVideoModel", () => {
  it("defaults to LTX and pins to its own storage key (no image filter)", async () => {
    await act(() => {
      root.render(
        <AppContext.Provider value={{ videoModels: [{ id: "ltx_2_3" }] }}>
          <VideoHarness />
        </AppContext.Provider>,
      );
    });
    expect(text("id")).toBe("ltx_2_3");
    expect(text("isDefault")).toBe("false");
    await click("make-default");
    expect(localStorage.getItem(SIMPLE_VIDEO_MODEL_KEY)).toBe("ltx_2_3");
    expect(localStorage.getItem(SIMPLE_IMAGE_MODEL_KEY)).toBe(null);
  });
});
