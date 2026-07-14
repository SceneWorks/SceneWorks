import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./main.jsx";
import { FakeEventSource, field, response, settle } from "./main.testSupport.jsx";

// sc-11969 (epic 11958, S10): the Preset Manager is a keep-alive screen (sc-11959), so an
// in-progress preset edit is held in component state and survives a plain nav round trip to
// another screen and back — untouched, and WITHOUT a discard prompt (leaving is
// non-destructive under keep-alive; only the editor's in-screen transitions guard). This
// mounts the FULL <App /> to prove the real mechanism end to end. The destructive-transition
// guards themselves are unit-tested in screens/PresetManagerScreen.test.jsx.
describe("Preset Manager keep-alive edit survival (sc-11969)", () => {
  let container;
  let root;

  const preset = {
    id: "cinematic",
    name: "Cinematic Portrait",
    scope: "global",
    workflow: "text_to_image",
    model: "z_image_turbo",
    defaults: { resolution: "1024x1024", count: 4 },
  };

  function mockFetch() {
    global.fetch = vi.fn((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) return Promise.resolve(response({ status: "ok", authRequired: false }));
      if (path.endsWith("/access")) return Promise.resolve(response({ authRequired: false }));
      if (path.endsWith("/jobs/events/ticket")) return Promise.resolve(response({ ticket: "stream-ticket" }));
      if (path.endsWith("/projects")) return Promise.resolve(response([{ id: "project-1", name: "Project One" }]));
      if (path.endsWith("/models")) {
        return Promise.resolve(
          response([{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image", capabilities: ["text_to_image"] }]),
        );
      }
      if (path.endsWith("/recipe-presets")) return Promise.resolve(response([preset]));
      return Promise.resolve(response([]));
    });
  }

  async function renderApp() {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
  }

  // Click a button by exact visible text. The sidebar renders before the workspace, so a
  // nav button wins over an identically-labelled inner button.
  async function clickButton(label) {
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent.trim() === label)?.click();
    });
    await settle();
  }

  const editorForm = () => document.body.querySelector(".preset-editor-form");
  const nameInput = () => field(editorForm(), "Name");
  // The desktop-safe discard confirm (appConfirm → ConfirmHost). A plain in-app nav under
  // keep-alive must never surface it — leaving is non-destructive (sc-11969).
  const confirmDialog = () => document.body.querySelector(".app-confirm-modal");

  async function openPresetForEditing() {
    await clickButton("Presets");
    await act(async () => {
      [...document.body.querySelectorAll(".preset-card")]
        .find((card) => card.textContent.includes("Cinematic Portrait"))
        .querySelector(".secondary-action")
        .click();
    });
    await settle();
  }

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    mockFetch();
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it("preserves an in-progress preset edit across a plain nav round trip, with no prompt", async () => {
    await renderApp();
    await openPresetForEditing();

    const editorBefore = editorForm();
    expect(editorBefore).not.toBeNull();

    // Make the edit dirty.
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
      setter.call(nameInput(), "Cinematic Portrait EDITED");
      nameInput().dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    await settle();
    expect(nameInput().value).toBe("Cinematic Portrait EDITED");
    expect(document.body.querySelector(".preset-status-pill").textContent).toContain("Unsaved changes");

    // Navigate to an OUT screen (Models unmounts on nav away) and back. A plain nav is
    // non-destructive under keep-alive, so it must NOT prompt a discard confirm.
    await clickButton("Models");
    expect(confirmDialog()).toBeNull();
    expect(document.body.querySelector(".models-surface")).not.toBeNull();
    // The Preset Manager stays resident (hidden) while another view is active.
    expect(editorForm()).not.toBeNull();

    await clickButton("Presets");
    // Returning is likewise silent — no confirm on either leg of the round trip.
    expect(confirmDialog()).toBeNull();
    // Same DOM node → never unmounted/remounted…
    expect(editorForm()).toBe(editorBefore);
    // …and the in-progress edit is intact with nothing to re-hydrate it from.
    expect(nameInput().value).toBe("Cinematic Portrait EDITED");
    expect(document.body.querySelector(".preset-status-pill").textContent).toContain("Unsaved changes");
  });
});
