import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// sc-11970 (S11): a dataset draft (name / membership / captions / character / config) must
// survive plain navigation (keep-alive, verified structurally below), and destructive
// transitions — opening another dataset or switching project while there are unsaved
// changes — must confirm via the desktop-safe appConfirm dialog (never window.confirm),
// before discarding. These tests spy appConfirm to assert the guard fires.
const appConfirmMock = vi.fn(() => Promise.resolve(true));
vi.mock("../appConfirm.jsx", () => ({
  appConfirm: (...args) => appConfirmMock(...args),
  useConfirm: () => appConfirmMock,
  ConfirmHost: () => null,
  normalizeConfirmOptions: (options) => options,
}));

import { AppContext } from "../context/AppContext.js";
import { TrainingDataSetsLibrary } from "./TrainingStudio.jsx";
import { KEEP_ALIVE_VIEWS } from "../App.jsx";

const datasetOne = { id: "dataset-1", name: "Mira Set", version: 2, characterId: "", items: [] };
const datasetTwo = { id: "dataset-2", name: "Other Set", version: 1, characterId: "", items: [] };

function deferred() {
  let resolve;
  const promise = new Promise((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project-a", name: "Project A" },
    authenticated: true,
    assets: [],
    characters: [],
    jobs: [],
    setPreviewAsset: () => {},
    trainingDatasets: [datasetOne, datasetTwo],
    trainingDatasetsProjectId: "project-a",
    loadingTrainingDatasets: false,
    refreshTrainingDatasets: () => {},
    loadTrainingDataset: vi.fn(async (id) => (id === "dataset-2" ? datasetTwo : datasetOne)),
    createTrainingDataset: vi.fn(),
    updateTrainingDataset: vi.fn(),
    // No targets → the Configure-job config effect early-returns, so the datasets panel
    // renders in isolation without a training catalog.
    trainingPresets: { presets: [] },
    trainingTargets: { targets: [] },
    setActiveView: () => {},
    models: [],
    createModelDownloadJob: () => {},
    // Auto-open dataset-1 on mount (Character Studio "Open" hand-off path), so the tests
    // don't have to drive the CompactSelector to get an active dataset.
    studioLaunch: { id: "launch-1", view: "LibraryDataSets", datasetId: "dataset-1" },
    ...overrides,
  };
}

async function settle() {
  await act(async () => {
    for (let index = 0; index < 6; index += 1) {
      await Promise.resolve();
    }
  });
}

function nameInput(container) {
  return container.querySelector('input[aria-label="Dataset name"]');
}

async function typeName(container, value) {
  const input = nameInput(container);
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(input.constructor.prototype, "value")?.set;
    setter?.call(input, value);
    input.dispatchEvent(new window.Event("input", { bubbles: true }));
  });
}

describe("TrainingStudio dataset draft guard (sc-11970)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    appConfirmMock.mockReset();
    appConfirmMock.mockResolvedValue(true);
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it("keeps the Train and Data Sets views mounted across plain nav (no prompt path)", () => {
    // Acceptance #1: keep-alive preserves the draft across plain navigation. Both views are
    // registered as keep-alive, so the screen is hidden — not unmounted — and no leave prompt
    // is wired for plain nav (the guard fires only on project switch / opening another dataset).
    expect(KEEP_ALIVE_VIEWS.has("Train")).toBe(true);
    expect(KEEP_ALIVE_VIEWS.has("LibraryDataSets")).toBe(true);
  });

  it("does NOT prompt when auto-opening a dataset with no unsaved draft", async () => {
    const context = baseContext();
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={context}>{<TrainingDataSetsLibrary />}</AppContext.Provider>);
    });
    await settle();
    // dataset-1 loaded, nothing edited → the initial open must not raise a discard prompt.
    expect(context.loadTrainingDataset).toHaveBeenCalledWith("dataset-1");
    expect(appConfirmMock).not.toHaveBeenCalled();
    expect(container.textContent).toContain("Version 2");
  });

  it("prompts via appConfirm before opening another dataset while dirty, and cancels on decline", async () => {
    appConfirmMock.mockResolvedValue(false);
    const context = baseContext();
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={context}>{<TrainingDataSetsLibrary />}</AppContext.Provider>);
    });
    await settle();
    expect(context.loadTrainingDataset).toHaveBeenCalledTimes(1);

    // Dirty the draft (rename), then request another dataset via a fresh studioLaunch.
    await typeName(container, "Mira Set edited");
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ ...context, studioLaunch: { id: "launch-2", view: "LibraryDataSets", datasetId: "dataset-2" } }}>
          {<TrainingDataSetsLibrary />}
        </AppContext.Provider>,
      );
    });
    await settle();

    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    expect(appConfirmMock.mock.calls[0][0]).toMatchObject({ tone: "danger", confirmLabel: "Discard" });
    // Declined → dataset-2 was never loaded; only the original dataset-1 open happened.
    expect(context.loadTrainingDataset).toHaveBeenCalledTimes(1);
    expect(context.loadTrainingDataset).not.toHaveBeenCalledWith("dataset-2");
  });

  it("clears the refresh owner when a different dataset open is declined", async () => {
    const reload = deferred();
    let loadCalls = 0;
    appConfirmMock.mockResolvedValue(false);
    const context = baseContext({
      refreshTrainingDatasets: vi.fn(async () => {}),
      loadTrainingDataset: vi.fn(() => {
        loadCalls += 1;
        return loadCalls === 1 ? Promise.resolve(datasetOne) : reload.promise;
      }),
    });
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={context}>{<TrainingDataSetsLibrary />}</AppContext.Provider>);
    });
    await settle();

    const refreshButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "Refresh");
    await act(async () => {
      refreshButton.click();
    });
    await vi.waitFor(() => expect(context.loadTrainingDataset).toHaveBeenCalledTimes(2));

    await typeName(container, "Mira Set edited while refresh reloads");
    await act(async () => {
      root.render(
        <AppContext.Provider value={{
          ...context,
          studioLaunch: { id: "launch-2", view: "LibraryDataSets", datasetId: "dataset-2" },
        }}>{<TrainingDataSetsLibrary />}</AppContext.Provider>,
      );
    });
    await vi.waitFor(() => expect(appConfirmMock).toHaveBeenCalledTimes(1));

    await act(async () => {
      reload.resolve(datasetOne);
      await Promise.resolve();
    });

    await act(async () => {
      container.querySelector(".compact-selector-pill").click();
    });
    const datasetOneOption = [...container.querySelectorAll(".compact-selector-item")]
      .find((button) => button.textContent.includes("Mira Set"));
    expect(datasetOneOption.disabled).toBe(false);
    expect(datasetOneOption.textContent).not.toContain("Opening…");
  });

  it("does not replace a rename started while dataset refresh is in flight", async () => {
    const refresh = deferred();
    const context = baseContext({ refreshTrainingDatasets: vi.fn(() => refresh.promise) });
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={context}>{<TrainingDataSetsLibrary />}</AppContext.Provider>);
    });
    await settle();

    const refreshButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "Refresh");
    await act(async () => {
      refreshButton.click();
    });
    await typeName(container, "Mira Set renamed");
    await act(async () => {
      refresh.resolve();
      await Promise.resolve();
    });

    expect(nameInput(container).value).toBe("Mira Set renamed");
    expect(context.loadTrainingDataset).toHaveBeenCalledTimes(1);
  });

  it("does not replace an edit made while the refresh reload is pending", async () => {
    const reload = deferred();
    let loadCalls = 0;
    const context = baseContext({
      refreshTrainingDatasets: vi.fn(async () => {}),
      loadTrainingDataset: vi.fn(() => {
        loadCalls += 1;
        return loadCalls === 1 ? Promise.resolve(datasetOne) : reload.promise;
      }),
    });
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={context}>{<TrainingDataSetsLibrary />}</AppContext.Provider>);
    });
    await settle();

    const refreshButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "Refresh");
    await act(async () => {
      refreshButton.click();
    });
    await vi.waitFor(() => expect(context.loadTrainingDataset).toHaveBeenCalledTimes(2));
    await typeName(container, "Mira Set edited during reload");
    await act(async () => {
      reload.resolve(datasetOne);
      await Promise.resolve();
    });

    expect(nameInput(container).value).toBe("Mira Set edited during reload");
  });

  it("does not replace a rename made while a completed Parquet import refreshes", async () => {
    const refresh = deferred();
    const refreshTrainingDatasets = vi.fn(() => refresh.promise);
    const context = baseContext({ refreshTrainingDatasets });
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={context}>{<TrainingDataSetsLibrary />}</AppContext.Provider>);
    });
    await settle();

    await act(async () => {
      root.render(
        <AppContext.Provider value={{
          ...context,
          jobs: [{
            id: "job-parquet",
            type: "dataset_parquet_import",
            status: "completed",
            payload: { datasetId: "dataset-1" },
            result: { importedItemCount: 1 },
          }],
        }}>{<TrainingDataSetsLibrary />}</AppContext.Provider>,
      );
    });
    await vi.waitFor(() => expect(refreshTrainingDatasets).toHaveBeenCalledTimes(1));

    await typeName(container, "Mira Set renamed during import refresh");
    await act(async () => {
      refresh.resolve();
      await Promise.resolve();
    });

    expect(nameInput(container).value).toBe("Mira Set renamed during import refresh");
    expect(context.loadTrainingDataset).toHaveBeenCalledTimes(1);
  });

  it("does not replace a rename made while a completed Parquet import reloads", async () => {
    const reload = deferred();
    let loadCalls = 0;
    const context = baseContext({
      refreshTrainingDatasets: vi.fn(async () => {}),
      loadTrainingDataset: vi.fn(() => {
        loadCalls += 1;
        return loadCalls === 1 ? Promise.resolve(datasetOne) : reload.promise;
      }),
    });
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={context}>{<TrainingDataSetsLibrary />}</AppContext.Provider>);
    });
    await settle();

    await act(async () => {
      root.render(
        <AppContext.Provider value={{
          ...context,
          jobs: [{
            id: "job-parquet",
            type: "dataset_parquet_import",
            status: "completed",
            payload: { datasetId: "dataset-1" },
            result: { importedItemCount: 1 },
          }],
        }}>{<TrainingDataSetsLibrary />}</AppContext.Provider>,
      );
    });
    await vi.waitFor(() => expect(context.loadTrainingDataset).toHaveBeenCalledTimes(2));

    await typeName(container, "Mira Set renamed during import reload");
    await act(async () => {
      reload.resolve(datasetOne);
      await Promise.resolve();
    });

    expect(nameInput(container).value).toBe("Mira Set renamed during import reload");
  });

  it("does not commit a completed Parquet import reload after the effect is superseded", async () => {
    const firstReload = deferred();
    const secondRefresh = deferred();
    let refreshCalls = 0;
    let loadCalls = 0;
    const reloadedDataset = {
      ...datasetOne,
      version: 3,
      items: [{ id: "parquet-item", path: "images/parquet.jpg", displayName: "parquet.jpg" }],
    };
    const context = baseContext({
      refreshTrainingDatasets: vi.fn(() => {
        refreshCalls += 1;
        return refreshCalls === 1 ? Promise.resolve() : secondRefresh.promise;
      }),
      loadTrainingDataset: vi.fn(() => {
        loadCalls += 1;
        return loadCalls === 1 ? Promise.resolve(datasetOne) : firstReload.promise;
      }),
    });
    const completedJob = (importedItemCount) => ({
      id: "job-parquet",
      type: "dataset_parquet_import",
      status: "completed",
      payload: { datasetId: "dataset-1" },
      result: { importedItemCount },
    });
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={context}>{<TrainingDataSetsLibrary />}</AppContext.Provider>);
    });
    await settle();

    await act(async () => {
      root.render(
        <AppContext.Provider value={{ ...context, jobs: [completedJob(1)] }}>
          {<TrainingDataSetsLibrary />}
        </AppContext.Provider>,
      );
    });
    await vi.waitFor(() => expect(context.loadTrainingDataset).toHaveBeenCalledTimes(2));

    // Updating the completion supersedes the first effect while its nested dataset load
    // is still pending. Keep the replacement effect in its refresh await, so only the
    // stale nested load could commit this fetched item.
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ ...context, jobs: [completedJob(2)] }}>
          {<TrainingDataSetsLibrary />}
        </AppContext.Provider>,
      );
    });
    await vi.waitFor(() => expect(context.refreshTrainingDatasets).toHaveBeenCalledTimes(2));
    await act(async () => {
      firstReload.resolve(reloadedDataset);
      await Promise.resolve();
    });

    expect(container.querySelectorAll(".training-caption-card")).toHaveLength(0);
    expect(container.textContent).not.toContain("Parquet import completed");

    // The replacement effect is intentionally still awaiting here. Unmount before
    // settling it so this regression leaves no background promise between tests.
    await act(async () => {
      root.unmount();
      root = null;
      secondRefresh.resolve();
      await Promise.resolve();
    });
  });

  it("opens another dataset when the discard prompt is confirmed", async () => {
    appConfirmMock.mockResolvedValue(true);
    const context = baseContext();
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={context}>{<TrainingDataSetsLibrary />}</AppContext.Provider>);
    });
    await settle();

    await typeName(container, "Mira Set edited");
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ ...context, studioLaunch: { id: "launch-2", view: "LibraryDataSets", datasetId: "dataset-2" } }}>
          {<TrainingDataSetsLibrary />}
        </AppContext.Provider>,
      );
    });
    await settle();

    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    expect(context.loadTrainingDataset).toHaveBeenCalledWith("dataset-2");
  });

  it("registers a project-switch guard that prompts only when the draft is dirty", async () => {
    let capturedGuard = null;
    const registerProjectSwitchGuard = vi.fn((guard) => {
      capturedGuard = guard;
      return () => {};
    });
    const context = baseContext({ registerProjectSwitchGuard });
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={context}>{<TrainingDataSetsLibrary />}</AppContext.Provider>);
    });
    await settle();

    expect(registerProjectSwitchGuard).toHaveBeenCalledTimes(1);
    expect(typeof capturedGuard).toBe("function");

    // Clean draft → the guard resolves true WITHOUT prompting (project switch proceeds).
    let cleanDecision;
    await act(async () => {
      cleanDecision = await capturedGuard({ id: "project-b" });
    });
    expect(cleanDecision).toBe(true);
    expect(appConfirmMock).not.toHaveBeenCalled();

    // Dirty the draft → the same guard now routes through appConfirm before allowing the switch.
    appConfirmMock.mockResolvedValue(false);
    await typeName(container, "Mira Set edited");
    let dirtyDecision;
    await act(async () => {
      dirtyDecision = await capturedGuard({ id: "project-b" });
    });
    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    expect(appConfirmMock.mock.calls[0][0]).toMatchObject({ tone: "danger" });
    expect(dirtyDecision).toBe(false);
  });

  it("promotes the unsaved pill: a Discard action reverts the draft to the saved state", async () => {
    const context = baseContext();
    root = createRoot(container);
    await act(async () => {
      root.render(<AppContext.Provider value={context}>{<TrainingDataSetsLibrary />}</AppContext.Provider>);
    });
    await settle();

    await typeName(container, "Mira Set edited");
    expect(container.textContent).toContain("Unsaved changes");
    const discard = [...container.querySelectorAll("button")].find((button) => button.textContent === "Discard");
    expect(discard).toBeTruthy();

    await act(async () => {
      discard.click();
    });
    await settle();
    // Reverted to the saved name → clean again, version pill returns.
    expect(nameInput(container).value).toBe("Mira Set");
    expect(container.textContent).toContain("Version 2");
    expect(container.textContent).not.toContain("Unsaved changes");
  });
});
