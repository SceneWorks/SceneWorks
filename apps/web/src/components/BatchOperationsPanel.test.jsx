import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BatchOperationsPanel } from "./BatchOperationsPanel.jsx";

const assets = [
  { id: "a", displayName: "one.png" },
  { id: "b", displayName: "two.png" },
];
const upscaleEngines = [
  { key: "real-esrgan", label: "Real-ESRGAN", factors: [2, 4] },
  { key: "seedvr2", label: "SeedVR2", factors: [2, 4], softness: true },
];

// React controlled inputs ignore a direct `.value =`; drive them via the native setter.
function setValue(el, value) {
  const proto = el instanceof window.HTMLTextAreaElement ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, "value").set.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

describe("BatchOperationsPanel (sc-6112)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });

  const mount = (props) =>
    act(() => {
      root.render(
        <BatchOperationsPanel
          assets={assets}
          upscaleEngines={upscaleEngines}
          editModels={[{ id: "m1", label: "Edit Model" }]}
          detailModels={[{ id: "d1", label: "Detail Model" }]}
          onRun={vi.fn()}
          onClose={vi.fn()}
          {...props}
        />,
      );
    });

  const tab = (label) => [...document.body.querySelectorAll(".batch-ops-tabs button")].find((b) => b.textContent.trim() === label);
  const runButton = () => [...document.body.querySelectorAll(".batch-ops-actions button")].find((b) => /^Run on/.test(b.textContent));

  it("shows the op tabs + upscale params, and the asset count in the head", async () => {
    await mount();
    expect(document.body.querySelector(".batch-ops-head h3").textContent).toContain("2 images");
    expect([...document.body.querySelectorAll(".batch-ops-tabs button")].map((b) => b.textContent.trim())).toEqual([
      "Upscale",
      "Detail enhance",
      "AI edit",
      "Depth map",
    ]);
    // Default op is upscale → an Engine + Factor select.
    expect(document.body.querySelector(".batch-ops-params").textContent).toContain("Engine");
    expect(document.body.querySelector(".batch-ops-params").textContent).toContain("Factor");
  });

  it("fans the upscale op + params back to onRun", async () => {
    const onRun = vi.fn();
    await mount({ onRun });
    await act(async () => runButton().click());
    expect(onRun).toHaveBeenCalledTimes(1);
    const [op, params] = onRun.mock.calls[0];
    expect(op).toBe("upscale");
    expect(params).toMatchObject({ engine: "real-esrgan", factor: 4 });
  });

  it("runs Iris restoration with its own controls, gated on the planner preview of the selection (sc-25683)", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        items: [
          { assetId: "a", plan: { summary: "4×: 512x384 → 2048x1536 (processed at 2048x1536 in 6 tiles)" }, error: null },
          { assetId: "b", plan: { summary: "budgeted: 683x512 input → 2732x2048 (4× of the budgeted input; source 4032x3024, processed at 2736x2048 in 15 tiles)" }, error: null },
        ],
      }),
    });
    const onRun = vi.fn();
    await mount({
      onRun,
      projectId: "p1",
      token: "t",
      upscaleEngines: [...upscaleEngines, { key: "iris", label: "Iris restore", factors: [4], restoration: true }],
    });
    const engineSelect = document.body.querySelector(".batch-ops-params select");
    await act(async () => {
      engineSelect.value = "iris";
      engineSelect.dispatchEvent(new Event("change", { bubbles: true }));
    });
    // No factor select for a restoration engine; Run waits for the plan.
    expect(document.body.querySelector(".batch-ops-params").textContent).not.toContain("Factor");
    expect(runButton().disabled).toBe(true);
    await act(async () => new Promise((resolve) => setTimeout(resolve, 260)));
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body).toEqual({ projectId: "p1", assetIds: ["a", "b"], scale: 4, inputSizing: "budgeted", colorFix: true });
    expect(document.body.querySelector(".iris-restore-preview").textContent).toContain("budgeted: 683x512 input → 2732x2048");
    expect(runButton().disabled).toBe(false);
    await act(async () => runButton().click());
    expect(onRun).toHaveBeenCalledWith("upscale", {
      engine: "iris",
      restore: { scale: 4, inputSizing: "budgeted", colorFix: true },
    });
  });

  it("blocks an Iris batch when any selected image's plan is refused", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ items: [{ assetId: "a", plan: null, error: "over the 8192px-per-side limit" }] }),
    });
    await mount({
      projectId: "p1",
      upscaleEngines: [{ key: "iris", label: "Iris restore", factors: [4], restoration: true }],
    });
    await act(async () => new Promise((resolve) => setTimeout(resolve, 260)));
    expect(runButton().disabled).toBe(true);
    expect(document.body.textContent).toContain("Cannot restore: over the 8192px-per-side limit");
  });

  it("requires a prompt for the edit op, then passes the prompt + model", async () => {
    const onRun = vi.fn();
    await mount({ onRun });
    await act(async () => tab("AI edit").click());
    // Empty prompt → Run disabled.
    expect(runButton().disabled).toBe(true);
    const textarea = document.body.querySelector(".batch-ops-params textarea");
    await act(async () => setValue(textarea, "make it snow"));
    expect(runButton().disabled).toBe(false);
    await act(async () => runButton().click());
    expect(onRun).toHaveBeenCalledWith("edit", expect.objectContaining({ model: "m1", prompt: "make it snow" }));
  });

  it("renders the per-item progress view (with statuses) when items are supplied", async () => {
    await mount({
      items: [
        { asset: { id: "a", displayName: "one.png" }, status: "completed" },
        { asset: { id: "b", displayName: "two.png" }, status: "running" },
      ],
      progress: { total: 2, done: 1, failed: 0, completed: 1, running: 1, queued: 0, allDone: false },
    });
    // Progress view replaces the form (no Run button) and lists per-item status.
    expect(runButton()).toBeUndefined();
    expect(document.body.querySelector(".batch-ops-summary").textContent).toContain("1 / 2 done");
    const items = [...document.body.querySelectorAll(".batch-ops-item")];
    expect(items).toHaveLength(2);
    expect(items[0].textContent).toContain("Done");
    expect(items[1].textContent).toContain("Running");
  });

  it("surfaces the reason for each batch item that failed to submit", async () => {
    await mount({
      items: [
        {
          asset: { id: "a", displayName: "one.png" },
          status: "failed",
          error: "Models service rejected the request",
        },
      ],
      progress: { total: 1, done: 1, failed: 1, completed: 0, running: 0, queued: 0, allDone: true },
    });
    expect(document.body.querySelector(".batch-ops-item-status").textContent).toBe(
      "Failed: Models service rejected the request",
    );
  });

  it("offers depth with an explicit capped/native choice and states why it cannot run (sc-25682)", async () => {
    const onRun = vi.fn();
    await mount({ onRun, depthAvailability: { available: false, reason: "Install Iris 3B Depth in Models to estimate depth." } });
    await act(async () => tab("Depth map").click());
    expect(document.body.textContent).toContain("Install Iris 3B Depth in Models to estimate depth.");
    expect(runButton().disabled).toBe(true);

    await mount({ onRun, depthAvailability: { available: true, reason: null } });
    const maxSide = document.body.querySelector('input[aria-label="Depth max side"]');
    expect(maxSide.value).toBe("1024");
    await act(async () => setValue(maxSide, "40"));
    expect(document.body.textContent).toMatch(/between 64 and 8192/);
    expect(runButton().disabled).toBe(true);
    await act(async () => setValue(maxSide, "2048"));
    expect(runButton().disabled).toBe(false);
    await act(async () => runButton().click());
    expect(onRun).toHaveBeenLastCalledWith("depth", { resolution: { mode: "capped", maxSide: 2048 } });

    const native = [...document.body.querySelectorAll('input[type="radio"]')].find((input) =>
      input.parentElement.textContent.startsWith("Native"),
    );
    await act(async () => native.click());
    await act(async () => runButton().click());
    expect(onRun).toHaveBeenLastCalledWith("depth", { resolution: { mode: "native" } });
  });

  // S3: a trained depth model runs without the release install — Run follows the SELECTED model.
  it("gates depth Run on the selected model, so a trained depth model needs no release install", async () => {
    const onRun = vi.fn();
    const depthAvailabilityFor = (irisModel) =>
      irisModel === "my_depth"
        ? { available: true, reason: null }
        : { available: false, reason: "Install Iris 3B Depth in Models to estimate depth." };
    await mount({
      onRun,
      initialOp: "depth",
      depthAvailability: { available: true, reason: null },
      depthAvailabilityFor,
      defaultDepthModel: "my_depth",
    });
    expect(runButton().disabled).toBe(false);
    await act(async () => runButton().click());
    expect(onRun).toHaveBeenLastCalledWith("depth", {
      resolution: { mode: "capped", maxSide: 1024 },
      irisModel: "my_depth",
    });

    // A fresh panel (the picker's state starts from `defaultDepthModel`).
    await act(async () => root.unmount());
    root = createRoot(container);
    await mount({
      onRun,
      initialOp: "depth",
      depthAvailability: { available: true, reason: null },
      depthAvailabilityFor,
      defaultDepthModel: "",
    });
    expect(document.body.textContent).toContain("Install Iris 3B Depth in Models to estimate depth.");
    expect(runButton().disabled).toBe(true);
  });

  // Round 2 (4): the host re-derives `defaultDepthModel` from the live catalog; the untouched picker
  // follows it.
  it("follows a default depth model that arrives after mount", async () => {
    const onRun = vi.fn();
    const depthAvailabilityFor = (irisModel) =>
      irisModel === "my_depth"
        ? { available: true, reason: null }
        : { available: false, reason: "Install Iris 3B Depth in Models to estimate depth." };
    const props = { onRun, initialOp: "depth", depthAvailability: { available: true, reason: null }, depthAvailabilityFor };
    await mount({ ...props, defaultDepthModel: "" });
    expect(runButton().disabled).toBe(true);
    await mount({ ...props, defaultDepthModel: "my_depth" });
    expect(runButton().disabled).toBe(false);
    await act(async () => runButton().click());
    expect(onRun).toHaveBeenLastCalledWith("depth", { resolution: { mode: "capped", maxSide: 1024 }, irisModel: "my_depth" });
  });

  it("opens directly on the depth op for a single-asset action", async () => {
    await mount({ initialOp: "depth", depthAvailability: { available: true, reason: null } });
    expect(tab("Depth map").getAttribute("aria-pressed")).toBe("true");
  });
});

