import React, { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleShell } from "./SimpleShell.jsx";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";

vi.mock("../api.js", async (importOriginal) => ({
  ...await importOriginal(), apiFetch: vi.fn(async () => ({})),
}));

let container;
let root;
beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
  ({ container, root } = mountRoot());
});
afterEach(async () => {
  await unmountRoot(root, container);
  vi.clearAllMocks();
});

it("submits the sole installed FP8R package from Simple without a picker or sticky selection", async () => {
  const model = { id: "flux3_klein", name: "FLUX3", type: "image", installState: "installed",
    usable: true, capabilities: ["text_to_image"], defaults: { resolution: "1024x1024" },
    limits: { resolutions: ["1024x1024"] }, hasVariantMatrix: true,
    variants: [{ variant: "fp8r", installState: "installed" }], ui: {} };
  const createImageJob = vi.fn(async () => ({ id: "job-fp8r-simple" }));
  const context = {
    activeProject: { id: "project-1", name: "Default" }, assets: [], recentImageAssets: [],
    recentVideoAssets: [], jobs: [], imageModels: [model], models: [model], videoModels: [],
    audioModels: [], loras: [], imageLocalJobs: [], videoLocalJobs: [], audioLocalJobs: [],
    visibleWorkers: [{ status: "online", capabilities: ["fp8r"] }], macCapabilities: null,
    createImageJob, rememberLocalGenerationJob: vi.fn(), setActiveView: vi.fn(),
  };
  await act(async () => root.render(<AppContext.Provider value={context}>
    <SimpleShell accent="teal" onAccentChange={() => {}} onModeChange={() => {}}
      onSimpleDefaultChange={() => {}} simpleDefault />
  </AppContext.Provider>));
  const prompt = container.querySelector("textarea");
  expect(prompt).toBeTruthy();
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set.call(prompt, "a fox");
    prompt.dispatchEvent(new Event("input", { bubbles: true }));
  });
  expect(container.querySelector(".quant-tier-picker")).toBeNull();
  const generate = [...container.querySelectorAll("button")].find((node) => node.textContent.includes("Generate"));
  expect(generate?.disabled).toBe(false);
  await act(async () => generate.click());
  expect(createImageJob).toHaveBeenCalledTimes(1);
  const payload = createImageJob.mock.calls[0][0];
  expect(payload.model).toBe("flux3_klein");
  expect(payload.advanced.quantTier).toBe("fp8r");
  expect(payload.advanced).not.toHaveProperty("mlxQuantize");
});
