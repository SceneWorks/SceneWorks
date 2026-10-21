import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { IrisRestoreControls } from "./IrisRestoreControls.jsx";

function setValue(el, value) {
  Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

const settle = () => act(async () => new Promise((resolve) => setTimeout(resolve, 260)));

function planResponse(items) {
  return { ok: true, status: 200, json: async () => ({ items }) };
}

describe("IrisRestoreControls (sc-25683)", () => {
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
    vi.restoreAllMocks();
  });

  function Harness({ initial = { scale: 4, inputSizing: "budgeted", colorFix: true }, planBody, onStatus }) {
    const [value, setValue_] = React.useState(initial);
    return <IrisRestoreControls onChange={setValue_} onStatus={onStatus} planBody={planBody} token="t" value={value} />;
  }

  it("shows the planner's processing + output dimensions before submission and reports ready", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockResolvedValue(
      planResponse([
        {
          assetId: null,
          plan: { summary: "budgeted: 683x512 input → 2732x2048 (4× of the budgeted input; source 4032x3024, processed at 2736x2048 in 15 tiles)" },
          error: null,
        },
      ]),
    );
    const onStatus = vi.fn();
    await act(async () => root.render(<Harness onStatus={onStatus} planBody={{ width: 4032, height: 3024 }} />));
    await settle();
    expect(container.querySelector(".iris-restore-preview").textContent).toContain(
      "budgeted: 683x512 input → 2732x2048",
    );
    expect(container.textContent).not.toMatch(/of the original/);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/api\/v1\/image-restoration\/plan$/);
    expect(JSON.parse(options.body)).toEqual({ width: 4032, height: 3024, scale: 4, inputSizing: "budgeted", colorFix: true });
    expect(onStatus).toHaveBeenLastCalledWith({ ready: true, blocked: false, error: null });
  });

  // S9b / S2: the selected trained model plans with its own export geometry, and a batch's distinct
  // render sizes plan together.
  it("plans with the selected trained model and every batch render size", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockResolvedValue(
      planResponse([
        { assetId: null, plan: { summary: "a" }, error: null },
        { assetId: null, plan: null, error: "output 6144x10752 exceeds 8192" },
      ]),
    );
    const onStatus = vi.fn();
    await act(async () =>
      root.render(
        <Harness
          initial={{ scale: 8, inputSizing: "original", colorFix: true, irisModel: "my_restore" }}
          onStatus={onStatus}
          planBody={{ sizes: [{ width: 512, height: 384 }, { width: 768, height: 1344 }] }}
        />,
      ),
    );
    await settle();
    expect(JSON.parse(fetchMock.mock.calls.at(-1)[1].body)).toEqual({
      sizes: [{ width: 512, height: 384 }, { width: 768, height: 1344 }],
      scale: 8,
      inputSizing: "original",
      colorFix: true,
      irisModel: "my_restore",
    });
    expect(onStatus).toHaveBeenLastCalledWith({ ready: false, blocked: true, error: "output 6144x10752 exceeds 8192" });
  });

  // Round 2 (5): with the release export absent, a fresh selection runs the first installed trained
  // restoration model — set on the value so the host's job carries it.
  it("defaults to the trained restoration model when the release export is absent", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockResolvedValue(planResponse([{ assetId: null, plan: { summary: "a" }, error: null }]));
    const trained = { id: "my_restore", name: "My restore", installState: "installed", trainedCheckpoint: { task: "restoration" } };
    let latest = null;
    function Host({ models }) {
      const [value, setValue_] = React.useState({ scale: 4, inputSizing: "budgeted", colorFix: true });
      latest = value;
      return <IrisRestoreControls models={models} onChange={setValue_} planBody={{ width: 512, height: 384 }} token="t" value={value} />;
    }
    await act(async () => root.render(<Host models={[{ id: "iris_3b_restore", installState: "missing" }]} />));
    expect(latest.irisModel).toBeUndefined();
    await act(async () => root.render(<Host models={[{ id: "iris_3b_restore", installState: "missing" }, trained]} />));
    await settle();
    expect(latest.irisModel).toBe("my_restore");
    expect(JSON.parse(fetchMock.mock.calls.at(-1)[1].body).irisModel).toBe("my_restore");
    // The release installed: nothing is substituted.
    await act(async () => root.unmount());
    root = createRoot(container);
    await act(async () => root.render(<Host models={[{ id: "iris_3b_restore", installState: "installed" }, trained]} />));
    expect(latest.irisModel).toBeUndefined();
  });

  it("re-plans when a control changes (1× restore-only, original input, colour fix off)", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockResolvedValue(
      planResponse([{ assetId: null, plan: { summary: "1× restore: 512x384 → 512x384 (processed at 1376x1024 in 2 tiles)" }, error: null }]),
    );
    await act(async () => root.render(<Harness planBody={{ width: 512, height: 384 }} />));
    await settle();
    const oneX = [...container.querySelectorAll("button")].find((b) => b.textContent === "1×");
    await act(async () => oneX.click());
    const original = [...container.querySelectorAll("label")].find((l) => l.textContent.includes("Original input")).querySelector("input");
    await act(async () => original.click());
    const colour = [...container.querySelectorAll("label")].find((l) => l.textContent.includes("Colour correction")).querySelector("input");
    await act(async () => colour.click());
    await settle();
    const last = JSON.parse(fetchMock.mock.calls.at(-1)[1].body);
    expect(last).toEqual({ width: 512, height: 384, scale: 1, inputSizing: "original", colorFix: false });
    expect(container.querySelector(".iris-restore-preview").textContent).toContain("1× restore: 512x384 → 512x384");
  });

  it("blocks submission on a refused plan and on an invalid scale (never substitutes one)", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockResolvedValue(
      planResponse([{ assetId: null, plan: null, error: "needs a 16128x12096 output, over the 8192px-per-side limit" }]),
    );
    const onStatus = vi.fn();
    await act(async () => root.render(<Harness onStatus={onStatus} planBody={{ width: 4032, height: 3024 }} />));
    await settle();
    expect(container.querySelector(".iris-restore-refused").textContent).toContain("Cannot restore");
    expect(onStatus).toHaveBeenLastCalledWith(expect.objectContaining({ ready: false, blocked: true }));

    const calls = fetchMock.mock.calls.length;
    await act(async () => setValue(container.querySelector("input[type=number]"), "0"));
    await settle();
    expect(fetchMock.mock.calls.length).toBe(calls);
    expect(container.textContent).toContain("Enter a scale greater than 0.");
    expect(onStatus).toHaveBeenLastCalledWith(expect.objectContaining({ ready: false, blocked: true }));
  });
});

describe("IrisRestoreControls trained model (sc-25688)", () => {
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
  });

  const release = [{ id: "iris_3b_restore", installState: "installed" }];
  const trained = { id: "mine", name: "My restorer", installState: "installed", trainedCheckpoint: { task: "restoration" } };

  it("offers a model select only when a trained restoration model is installed", async () => {
    const onChange = vi.fn();
    const value = { scale: 4, inputSizing: "budgeted", colorFix: true };
    await act(async () => root.render(<IrisRestoreControls models={release} onChange={onChange} value={value} />));
    expect(container.querySelector('select[aria-label="Iris restoration model"]')).toBeNull();
    const depthOnly = [...release, { ...trained, trainedCheckpoint: { task: "depth" } }];
    await act(async () => root.render(<IrisRestoreControls models={depthOnly} onChange={onChange} value={value} />));
    expect(container.querySelector('select[aria-label="Iris restoration model"]')).toBeNull();

    await act(async () => root.render(<IrisRestoreControls models={[...release, trained]} onChange={onChange} value={value} />));
    const select = container.querySelector('select[aria-label="Iris restoration model"]');
    expect([...select.options].map((option) => option.value)).toEqual(["", "mine"]);
    await act(async () => {
      select.value = "mine";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(onChange).toHaveBeenLastCalledWith({ ...value, irisModel: "mine" });
  });
});
