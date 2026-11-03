import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";

const { sidecar } = vi.hoisted(() => ({ sidecar: vi.fn() }));
vi.mock("../api.js", async (importOriginal) => ({ ...await importOriginal(), irisItemSidecar: sidecar }));
import { TrainingDataSetsLibrary } from "./TrainingStudio.jsx";

let root, container;
afterEach(() => { act(() => root?.unmount()); container?.remove(); vi.resetAllMocks(); });

it.each([
  ["unchanged owner", "same", true],
  ["dataset rejection", "dataset", false],
  ["dataset success", "dataset", true],
  ["project change", "project", false],
  ["token change", "token", false],
  ["unmount", "unmount", false],
])("stops an owned sidecar batch on %s", async (_label, change, succeed) => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  let resolveFirst, rejectFirst;
  sidecar.mockImplementationOnce(() => new Promise((resolve, reject) => { resolveFirst = resolve; rejectFirst = reject; }));
  const items = [{ id: "item_0000", displayName: "one.png" }, { id: "item_0001", displayName: "two.png" }];
  const a = { id: "a", name: "Dataset A", items };
  const b = { id: "b", name: "Dataset B", items };
  sidecar.mockResolvedValue(a);
  const context = {
    activeProject: { id: "p1" }, token: "token-a", authenticated: true, assets: [], characters: [], jobs: [],
    trainingDatasets: [a, b], trainingDatasetsProjectId: "p1", models: [], loras: [],
    trainingPresets: { presets: [] }, trainingTargets: { targets: [] }, setPreviewAsset: vi.fn(),
    loadTrainingDataset: vi.fn(async (id) => id === "a" ? a : b),
    studioLaunch: { id: "launch-a", view: "LibraryDataSets", datasetId: "a" },
  };
  const render = async (value) => act(async () => root.render(
    <AppContext.Provider value={value}><TrainingDataSetsLibrary /></AppContext.Provider>,
  ));
  await render(context);
  const details = container.querySelector("details.training-iris-options");
  details.open = true;
  await act(async () => details.dispatchEvent(new Event("toggle")));
  const input = container.querySelector('input[aria-label="Attach depth maps by filename"]');
  Object.defineProperty(input, "files", { configurable: true, value: [new File(["a"], "one.npy"), new File(["b"], "two.npy")] });
  await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));
  expect(sidecar).toHaveBeenCalledTimes(1);
  expect(sidecar.mock.calls[0][0]).toMatchObject({ projectId: "p1", datasetId: "a", itemId: "item_0000" });
  if (change === "dataset") {
    await render({ ...context, studioLaunch: { id: "launch-b", view: "LibraryDataSets", datasetId: "b" } });
    expect(container.textContent).toContain("Dataset B");
  } else if (change === "project") {
    await render({ ...context, activeProject: { id: "p2" }, trainingDatasetsProjectId: "p2" });
  } else if (change === "token") {
    await render({ ...context, token: "token-b" });
  } else if (change === "unmount") {
    await act(async () => root.unmount());
    root = null;
  }
  await act(async () => {
    if (succeed) resolveFirst(a); else rejectFirst(new Error("failed"));
    await Promise.resolve();
  });
  expect(sidecar).toHaveBeenCalledTimes(change === "same" ? 2 : 1);
  if (change === "same") expect(sidecar.mock.calls[1][0]).toMatchObject({ projectId: "p1", datasetId: "a", itemId: "item_0001" });
  if (change === "dataset") expect(container.textContent).toContain("Dataset B");
});
