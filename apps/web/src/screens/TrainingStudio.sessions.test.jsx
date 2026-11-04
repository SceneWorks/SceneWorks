import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { TrainingDataSetsLibrary } from "./TrainingStudio.jsx";

const capture = vi.hoisted(() => ({ current: null }));
vi.mock("./training/DatasetEditorPanel.jsx", () => ({
  DatasetEditorPanel: (props) => { capture.current = props; return null; },
}));
let root;
let container;
afterEach(() => { act(() => root?.unmount()); container?.remove(); });

it("retains session identity across queue ticks and invokes the current draft's handlers", async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  const onPreview = vi.fn();
  const context = {
    activeProject: { id: "p" }, authenticated: true, assets: [], characters: [], jobs: [],
    trainingDatasets: [], trainingDatasetsProjectId: "p", models: [], loras: [],
    trainingPresets: { presets: [] }, trainingTargets: { targets: [] },
    setPreviewAsset: onPreview,
  };
  const render = async (value) => act(async () => root.render(
    <AppContext.Provider value={value}><TrainingDataSetsLibrary /></AppContext.Provider>,
  ));
  await render(context);
  const initial = capture.current;
  const nextPreview = vi.fn();
  await render({ ...context, setPreviewAsset: nextPreview, jobs: [{ id: "other", type: "image", status: "running" }] });
  for (const key of Object.keys(initial)) expect(capture.current[key], key).toBe(initial[key]);
  // A stable consumer callback must see a replaced owner callback, not an old closure.
  initial.captionSession.onPreview("image");
  expect(nextPreview).toHaveBeenCalledWith("image");
  expect(onPreview).not.toHaveBeenCalled();
  await act(async () => initial.datasetSession.setDraftName("unsaved name"));
  expect(capture.current.datasetSession.draftName).toBe("unsaved name");
  await act(async () => initial.datasetSession.discardDraft());
  expect(capture.current.datasetSession.draftName).toBe("");
});
