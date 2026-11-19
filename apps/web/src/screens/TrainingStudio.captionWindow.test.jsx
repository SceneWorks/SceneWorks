import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { TrainingDataSetsLibrary } from "./TrainingStudio.jsx";

const rendered = vi.hoisted(() => []);
vi.mock("../components/assetMedia.jsx", async (importOriginal) => ({
  ...await importOriginal(),
  AssetThumbnail: ({ asset }) => {
    const ref = React.useRef(null);
    React.useLayoutEffect(() => {
      if (ref.current?.closest(".training-caption-card")) rendered.push(asset.id);
    });
    return <span data-thumbnail={asset.id} ref={ref} />;
  },
}));
let root;
let container;
let context;
let assets;
const COUNT = 1000;
beforeEach(async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  assets = Array.from({ length: COUNT }, (_, index) => ({
    id: `asset-${index}`, type: "image", displayName: `image-${index}.png`, file: { path: `images/${index}.png` },
  }));
  const dataset = { id: "dataset", version: 1, name: "Large dataset", items: assets.map((asset, index) => ({
    id: `item-${index}`, assetId: asset.id, caption: { text: index ? `caption ${index}` : "", source: "manual" },
  })) };
  context = {
    activeProject: { id: "p" }, authenticated: true, assets, characters: [], jobs: [],
    trainingDatasets: [dataset], trainingDatasetsProjectId: "p", models: [], loras: [],
    trainingPresets: { presets: [] }, trainingTargets: { targets: [] },
    setPreviewAsset: vi.fn(), loadTrainingDataset: vi.fn(async () => dataset),
    studioLaunch: { id: "launch", view: "LibraryDataSets", datasetId: dataset.id },
  };
  await render(context);
  rendered.length = 0;
});
afterEach(() => { act(() => root.unmount()); container.remove(); });
async function render(value) {
  await act(async () => root.render(<AppContext.Provider value={value}><TrainingDataSetsLibrary /></AppContext.Provider>));
}
async function type(input, value) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
async function scroll(top) {
  await act(async () => {
    const viewport = container.querySelector(".training-caption-window");
    viewport.scrollTop = top;
    viewport.dispatchEvent(new Event("scroll", { bubbles: true }));
  });
}
function caption(index) { return container.querySelector(`textarea[aria-label="Caption for image-${index}.png"]`); }

it("bounds the real Studio DOM and renders only the edited card, with no card renders for unrelated SSE ticks", async () => {
  const mounted = [...container.querySelectorAll(".training-caption-card")];
  expect(mounted.length).toBeGreaterThan(1);
  expect(mounted.length).toBeLessThan(12);
  await type(caption(0), "edited caption");
  expect(caption(0).value).toBe("edited caption");
  expect(rendered.filter((id) => id !== "asset-0")).toEqual([]);
  expect(rendered.filter((id) => id === "asset-0").length).toBeGreaterThan(0);
  rendered.length = 0;
  const nextPreview = vi.fn();
  await render({ ...context, setPreviewAsset: nextPreview, jobs: [{ id: "unrelated", type: "image", status: "running" }] });
  expect(rendered).toEqual([]);
  await act(async () => container.querySelector(".training-caption-card-thumb").click());
  expect(nextPreview).toHaveBeenCalledWith(assets[0], assets);
  expect(nextPreview.mock.calls[0][1]).toHaveLength(COUNT);
});

it("keeps the focused caption mounted while scrolling and restores owner drafts after other rows unmount", async () => {
  const first = caption(0);
  await act(async () => first.focus());
  await type(first, "persist through windowing");
  await scroll(352 * 500);
  expect(document.activeElement).toBe(first);
  expect(first.isConnected).toBe(true);
  expect(caption(500)).not.toBeNull();
  expect(container.querySelectorAll(".training-caption-card").length).toBeLessThan(14);
  await act(async () => caption(500).focus());
  await type(caption(500), "far caption");
  await act(async () => caption(500).blur());
  await scroll(0);
  expect(caption(0).value).toBe("persist through windowing");
  await scroll(352 * 500);
  expect(caption(500).value).toBe("far caption");
});

it("reaches unmounted rows through keyboard boundaries and retains focus while resolving Missing", async () => {
  const last = [...container.querySelectorAll(".training-caption-card")].at(-1);
  const edge = [...last.querySelectorAll("button")].at(-1);
  const nextIndex = Number(last.dataset.captionId.split("-").at(-1)) + 1;
  await act(async () => {
    edge.focus();
    edge.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true }));
  });
  expect(document.activeElement.closest(".training-caption-card").dataset.captionId).toBe(`asset-${nextIndex}`);
  await scroll(0);
  const missing = [...container.querySelectorAll("button")].find((button) => button.textContent === "Missing");
  await act(async () => missing.click());
  const input = caption(0);
  await act(async () => input.focus());
  await type(input, "now has a caption");
  expect(document.activeElement).toBe(input);
  expect(input.isConnected).toBe(true);
  await act(async () => input.blur());
  expect(caption(0)).toBeNull();
  expect(container.textContent).toContain("Every image has a caption.");
});

it("submits the selected Loop caption tier and keeps training caption modes", async () => {
  const dataset = context.trainingDatasets[0];
  const createTrainingDatasetCaptionJob = vi.fn(async () => ({ id: "caption-job" }));
  const updateTrainingDataset = vi.fn(async () => dataset);
  await render({ ...context, updateTrainingDataset, createTrainingDatasetCaptionJob,
    models: [{ id: "vision_sensenova_loop_9b", name: "SenseNova-Loop-9B", variants: [
      { variant: "q4", installState: "installed" }, { variant: "q8", installState: "installed" },
    ] }],
  });
  const button = (text) => [...document.body.querySelectorAll("button")].find((node) => node.textContent.trim() === text);
  const select = async (label, value) => act(async () => {
    const control = [...document.body.querySelectorAll("label")]
      .find((node) => node.firstChild.textContent.trim() === label).querySelector("select");
    control.value = value;
    control.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await act(async () => button("Caption all").click());
  await select("Method", "sensenova_loop");
  await select("Tier", "q8");
  await select("Mode", "subjectOnly");
  await act(async () => button("Caption missing").click());
  expect(createTrainingDatasetCaptionJob).toHaveBeenCalledWith(dataset.id, expect.objectContaining({
    captioner: "sensenova_loop", modelNameOrPath: "sensenova/SenseNova-Loop-9B",
    variant: "q8", mode: "subjectOnly", options: expect.objectContaining({ captionPrompt: "" }),
  }));
});
