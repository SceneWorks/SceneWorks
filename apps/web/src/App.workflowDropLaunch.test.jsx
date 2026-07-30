// "Use this workflow" has to LAND somewhere, from either shell (sc-15951, epic 15945).
//
// The drop guard is installed at App level and therefore runs in BOTH shells, which is why the
// offer panel renders in both. But `SimpleShell` renders its own studios and consumes no
// `studioLaunch`, so pointing the workspace at "Image" from in there is invisible: the panel
// dismissed, nothing happened, and the launch sat queued to fire unprompted the next time the user
// opened the Advanced studio. This suite pins the two halves of the fix — the shell flip, and the
// refusal on a viewport that has Simple locked on, where there is no Advanced workspace to reach.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { resetStartupTimingForTests } from "./startupTiming.js";

const PNG_HEAD = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

const MODEL = {
  id: "krea_2_turbo",
  name: "Krea 2 Turbo",
  type: "image",
  family: "krea2",
  capabilities: ["text_to_image"],
  installState: "installed",
  defaults: { resolution: "1024x1024" },
  limits: { resolutions: ["1024x1024"] },
  loraCompatibility: {},
  ui: {},
};

const INSPECT_BODY = {
  status: "workflow",
  workflow: {
    sceneworksWorkflow: "image",
    schemaVersion: 1,
    producer: { name: "SceneWorks", url: "https://example.invalid", version: "0.8.1" },
    mode: "text_to_image",
    model: "krea_2_turbo",
    prompt: "a lighthouse in heavy fog, cinematic",
    negativePrompt: "",
    seed: 880412,
    width: 1024,
    height: 1024,
    count: 1,
    advanced: { steps: 28 },
  },
  resolution: {
    model: {
      slug: "krea_2_turbo",
      state: "resolved",
      catalogId: "krea_2_turbo",
      name: "Krea 2 Turbo",
      detail: "Krea 2 Turbo is installed on this machine.",
    },
    loras: [],
    styles: [],
    inputs: [],
    omitted: [],
    runnable: true,
    inputImagesRequired: 0,
  },
};

function pngFile() {
  return new File([new Uint8Array([...PNG_HEAD, 1, 2, 3, 4])], "shared.png", { type: "image/png" });
}

describe("App — a dropped workflow launches from either shell", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    // Open in the Simple shell, the way a user who ticked "Use Simple UI by default" does.
    window.localStorage.setItem("sceneworks-simple-ui-default", "true");
    window.innerWidth = 1440;
    global.fetch = vi.fn((url) => {
      const path = new URL(url, "http://localhost").pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
      }
      if (path.endsWith("/ui-preferences")) {
        return Promise.resolve(response({}));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([MODEL]));
      }
      if (path.endsWith("/workflows/inspect")) {
        return Promise.resolve(response(INSPECT_BODY));
      }
      return Promise.resolve(response([]));
    });
  });

  afterEach(async () => {
    await act(async () => root?.unmount());
    container.remove();
    resetStartupTimingForTests();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  async function renderApp() {
    root = createRoot(container);
    await act(async () => root.render(<App />));
    await settle();
  }

  // A drop nothing in the tree claims — the same window-level branch the issue-#1308 guard has
  // always used.
  async function dropOnNeutralChrome() {
    const event = new Event("drop", { bubbles: true, cancelable: true });
    event.dataTransfer = { dropEffect: "copy", files: [pngFile()], types: ["Files"] };
    await act(async () => {
      document.body.dispatchEvent(event);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    await settle();
  }

  const buttonByLabel = (label) =>
    [...document.body.querySelectorAll("button")].find(
      (node) => node.textContent.trim() === label,
    );
  const click = async (node) => {
    await act(async () => node.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    await settle();
  };

  it("flips the Simple shell to the Advanced workspace and applies the recipe there", async () => {
    await renderApp();
    expect(container.querySelector(".su-root")).toBeTruthy();

    await dropOnNeutralChrome();
    const use = buttonByLabel("Use this workflow");
    expect(use).toBeTruthy();

    await click(use);

    // The Simple shell is gone and the full workspace is on screen — without the flip this stayed
    // on Simple and the launch sat queued for whenever the user next opened Advanced.
    expect(container.querySelector(".su-root")).toBeNull();
    expect(container.querySelector("main.app")).toBeTruthy();
    // …and the recipe actually landed: the studio's prompt box holds the shared prompt.
    const prompt = [...container.querySelectorAll("textarea")].find((node) =>
      node.value.includes("a lighthouse in heavy fog"),
    );
    expect(prompt).toBeTruthy();
    expect(document.body.querySelector(".workflow-drop-modal")).toBeNull();
  });

  it("refuses, and says why, on a viewport that has Simple locked on", async () => {
    // A phone has no Advanced workspace to land in (the full workspace is unusable at that width),
    // so the honest answer is a refusal with a reason — not a dismissed panel over a button that
    // did nothing, and not a launch queued to fire unprompted somewhere else.
    window.innerWidth = 375;
    await renderApp();
    await dropOnNeutralChrome();

    await click(buttonByLabel("Use this workflow"));

    expect(document.body.querySelector(".workflow-drop-modal")).toBeTruthy();
    expect(document.body.textContent).toContain("switch on a larger screen");
    expect(container.querySelector(".su-root")).toBeTruthy();
  });
});
