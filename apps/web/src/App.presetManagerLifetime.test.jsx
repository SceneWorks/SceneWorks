import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./main.jsx";
import { FakeEventSource, field, response, settle } from "./main.testSupport.jsx";

describe("Preset Manager page lifetime", () => {
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

  async function clickButton(label) {
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent.trim() === label)?.click();
    });
    await settle();
  }

  const editorForm = () => document.body.querySelector(".preset-editor-form");
  const nameInput = () => field(editorForm(), "Name");

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

  it("keeps a dirty preset on cancelled navigation, then releases it after confirmed discard", async () => {
    await renderApp();
    await openPresetForEditing();

    const editorBefore = editorForm();
    expect(editorBefore).not.toBeNull();

    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
      setter.call(nameInput(), "Cinematic Portrait EDITED");
      nameInput().dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    await settle();
    expect(nameInput().value).toBe("Cinematic Portrait EDITED");
    expect(document.body.querySelector(".preset-status-pill").textContent).toContain("Unsaved changes");

    await clickButton("Models");
    expect(confirmDialog()).not.toBeNull();
    expect(editorForm()).toBe(editorBefore);
    await clickButton("Keep editing");
    expect(confirmDialog()).toBeNull();
    expect(nameInput().value).toBe("Cinematic Portrait EDITED");

    await clickButton("Models");
    await clickButton("Discard changes");
    expect(confirmDialog()).toBeNull();
    expect(editorForm()).toBeNull();
    expect(document.body.querySelector(".models-surface")).not.toBeNull();

    await openPresetForEditing();
    expect(editorForm()).not.toBe(editorBefore);
    expect(nameInput().value).toBe("Cinematic Portrait");
  });
});
