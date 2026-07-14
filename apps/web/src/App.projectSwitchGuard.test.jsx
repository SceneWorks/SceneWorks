import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// sc-11970 (S11): creating a NEW workspace while a Data Sets draft is unsaved must route
// through the SAME project-switch guard as picking an EXISTING project — otherwise the switch
// silently resets the Training Studio (keyed on activeProject.id) and wipes the draft, while an
// existing-project pick would have prompted. These tests render the full App, dirty a Data Sets
// draft, then create a workspace via the ProjectSwitcher and assert the guard behaves the same:
// cancel keeps the current project + draft; confirm switches to the new workspace.
const appConfirmMock = vi.fn(() => Promise.resolve(true));
vi.mock("./appConfirm.jsx", () => ({
  appConfirm: (...args) => appConfirmMock(...args),
  useConfirm: () => appConfirmMock,
  ConfirmHost: () => null,
  normalizeConfirmOptions: (options) => options,
}));

import { App } from "./main.jsx";
import { FakeEventSource, response, settle, field, changeField } from "./main.testSupport.jsx";

function installFetch() {
  global.fetch = vi.fn((url, options = {}) => {
    const path = new URL(url).pathname;
    const method = options.method ?? "GET";
    if (path.endsWith("/health")) {
      return Promise.resolve(response({ status: "ok", authRequired: false }));
    }
    if (path.endsWith("/access")) {
      return Promise.resolve(response({ authRequired: false }));
    }
    if (path.endsWith("/jobs/events/ticket")) {
      return Promise.resolve(response({ ticket: "stream-ticket" }));
    }
    if (path.endsWith("/projects") && method === "POST") {
      return Promise.resolve(response({ id: "project-new", name: "Fresh Set Project" }));
    }
    if (path.endsWith("/projects")) {
      return Promise.resolve(response([{ id: "project-a", name: "Project A" }]));
    }
    // Everything else (training datasets, characters, assets, models, …) is empty — no saved
    // datasets means the Data Sets library opens on a fresh "New dataset" draft.
    return Promise.resolve(response([]));
  });
}

describe("createProject routes the workspace switch through the Data Sets guard (sc-11970)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    appConfirmMock.mockReset();
    appConfirmMock.mockResolvedValue(true);
    installFetch();
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  function projectPillText() {
    return container.querySelector(".project-pill")?.textContent ?? "";
  }

  // Boot the app on Project A and navigate to the Data Sets library (which registers the
  // project-switch guard). Returns once the datasets screen is mounted.
  async function bootIntoDataSets() {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    // Auto-selected the only workspace.
    expect(projectPillText()).toContain("Project A");

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Data Sets").click();
    });
    await settle();
    expect(field(container, "Dataset name")).toBeTruthy();
  }

  // Drive the ProjectSwitcher's "New workspace" flow to create + attempt to switch to a project.
  async function createWorkspace(name) {
    await act(async () => {
      container.querySelector(".project-pill").click();
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "New workspace").click();
    });
    await changeField(container.querySelector('input[aria-label="New workspace name"]'), name);
    await act(async () => {
      container.querySelector(".project-menu-create button[type='submit']").click();
    });
    await settle();
    await settle();
  }

  it("prompts and, on cancel, keeps the current workspace and the unsaved draft", async () => {
    appConfirmMock.mockResolvedValue(false);
    await bootIntoDataSets();

    // Dirty the draft — a brand-new dataset with a name is unsaved work the guard protects.
    await changeField(field(container, "Dataset name"), "Fresh Set");
    expect(appConfirmMock).not.toHaveBeenCalled();

    await createWorkspace("Fresh Set Project");

    // The guard fired with the desktop-safe danger dialog...
    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    expect(appConfirmMock.mock.calls[0][0]).toMatchObject({ tone: "danger", confirmLabel: "Discard" });
    // ...and cancelling kept us on Project A with the draft intact (the new project still
    // exists in the list — 2 workspaces — and can be opened later).
    expect(projectPillText()).toContain("Project A");
    expect(projectPillText()).not.toContain("Fresh Set Project");
    expect(field(container, "Dataset name").value).toBe("Fresh Set");
  });

  it("switches to the new workspace when the discard prompt is confirmed", async () => {
    appConfirmMock.mockResolvedValue(true);
    await bootIntoDataSets();

    await changeField(field(container, "Dataset name"), "Fresh Set");
    await createWorkspace("Fresh Set Project");

    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    // Confirming discarded the draft and switched to the freshly created workspace.
    expect(projectPillText()).toContain("Fresh Set Project");
  });

  it("does NOT prompt when creating a workspace with no unsaved Data Sets draft", async () => {
    await bootIntoDataSets();

    // No edits → nothing to lose → the guard resolves without ever raising appConfirm.
    await createWorkspace("Fresh Set Project");

    expect(appConfirmMock).not.toHaveBeenCalled();
    expect(projectPillText()).toContain("Fresh Set Project");
  });
});
