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
