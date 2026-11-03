import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { TrainingDataSetsLibrary } from "./TrainingStudio.jsx";

it("renders current dataset thumbnails when membership, dataset, or project changes", async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const empty = { id: "one", name: "Empty", items: [] };
  const covered = { id: "two", name: "Covered", coverPath: "cover.jpg", items: [] };
  const asset = { id: "image", type: "image", projectId: "p1", file: { path: "first.jpg" } };
  const context = {
    activeProject: { id: "p1" }, authenticated: true, assets: [asset], characters: [], jobs: [],
    trainingDatasets: [empty, covered], trainingDatasetsProjectId: "p1", models: [], loras: [],
    trainingPresets: { presets: [] }, trainingTargets: { targets: [] }, setPreviewAsset: vi.fn(),
    loadTrainingDataset: vi.fn(async (id) => id === "one" ? empty : covered),
    studioLaunch: { id: "launch-1", view: "LibraryDataSets", datasetId: "one" },
  };
  const render = async (value) => act(async () => root.render(
    <AppContext.Provider value={value}><TrainingDataSetsLibrary /></AppContext.Provider>,
  ));
  const click = async (element) => { expect(element).toBeTruthy(); await act(async () => element.click()); };
  const button = (text) => [...document.querySelectorAll("button")].find((el) => el.textContent.trim() === text);
  const thumb = () => container.querySelector(".compact-selector-pill .compact-selector-thumb img");
  try {
    await render(context);
    expect(thumb()).toBeNull();
    await click(button("Add images"));
    await click(button("Asset Library"));
    await click(document.querySelector(".dataset-add-card"));
    await click(button("Add 1"));
    expect(thumb()?.getAttribute("src")).toContain("first.jpg");
    await click(button("Done"));
    await click(container.querySelector('[aria-label="Remove first.jpg"]'));
    expect(thumb()).toBeNull();
    await click(container.querySelector(".compact-selector-pill"));
    await click([...container.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes("Covered")));
    expect(thumb()?.getAttribute("src")).toContain("/projects/p1/files/cover.jpg");
    await render({ ...context, activeProject: { id: "p2" }, trainingDatasetsProjectId: "p2", studioLaunch: { id: "launch-2", view: "LibraryDataSets", datasetId: "two" } });
    await click(container.querySelector(".compact-selector-pill"));
    const coveredOption = [...container.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes("Covered"));
    expect(coveredOption.querySelector("img")?.getAttribute("src")).toContain("/projects/p2/files/cover.jpg");
  } finally {
    act(() => root.unmount());
    container.remove();
  }
});
