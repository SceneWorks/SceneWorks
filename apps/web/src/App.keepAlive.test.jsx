import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { FakeEventSource, response, settle, changeField } from "./main.testSupport.jsx";

// Selective lazy keep-alive shell (sc-11959). These tests drive the FULL <App /> so the
// keep-alive shell — the visited-set tracking, the KeepAlivePane wrapper, and the
// OUT-screen conditional unmount — is exercised end to end. jsdom does not apply
// styles.css, so the panes are not visually hidden here; we assert the MECHANISM via DOM
// node presence/identity (a kept-alive screen is the SAME node across a nav round trip;
// an OUT screen's node disappears) rather than via computed visibility.
describe("selective lazy keep-alive shell (sc-11959)", () => {
  let container;
  let root;

  function mockFetch(overrides = {}) {
    const {
      projects = [{ id: "project-1", name: "Project One" }],
      models = [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
      assets = [],
      loras = [],
      presets = [],
    } = overrides;
    global.fetch = vi.fn((url) => {
      const path = new URL(url).pathname;
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
        return Promise.resolve(response(projects));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response(models));
      }
      if (path.endsWith("/assets")) {
        return Promise.resolve(response(assets));
      }
      if (path.endsWith("/loras")) {
        return Promise.resolve(response(loras));
      }
      if (path.endsWith("recipe-presets")) {
        return Promise.resolve(response(presets));
      }
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

  // Click a top-level nav item (or any button) by its exact visible text. The sidebar
  // renders before the workspace content, so a nav button always wins over any
  // identically-labelled button inside a kept-alive pane.
  async function clickButton(label) {
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === label)?.click();
    });
    await settle();
  }

  const imageStudio = () => document.body.querySelector(".image-studio");
  const videoStudio = () => document.body.querySelector(".video-studio");
  const modelsSurface = () => document.body.querySelector(".models-surface");
  const promptField = () => document.body.querySelector("textarea[aria-label='Prompt']");
  const documentStudio = () => document.body.querySelector(".document-studio");
  // DocumentStudio's Prompt textarea is the first textarea in its compose form (it wraps
  // the control in a <label> rather than carrying an aria-label).
  const documentPromptField = () => document.body.querySelector(".document-studio .studio-form textarea");

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

  it("does not mount a keep-alive screen until it is first visited", async () => {
    await renderApp();
    // Initial view is Library/Assets — no creative screen has been visited yet.
    expect(imageStudio()).toBeNull();
    expect(videoStudio()).toBeNull();

    await clickButton("Image");
    expect(imageStudio()).not.toBeNull();
    // Video was still never visited, so it is absent from the DOM.
    expect(videoStudio()).toBeNull();
  });

  it("keeps a visited studio mounted (same node) across a nav round trip, preserving edits", async () => {
    await renderApp();
    await clickButton("Image");

    const studioBefore = imageStudio();
    expect(studioBefore).not.toBeNull();

    await changeField(promptField(), "a keep-alive draft prompt");
    expect(promptField().value).toBe("a keep-alive draft prompt");

    // Clear localStorage so a hypothetical remount could NOT re-hydrate the prompt from
    // the persisted snapshot — the only way the value can survive is if the studio was
    // never unmounted.
    window.localStorage.clear();

    await clickButton("Assets");
    // The studio stays mounted (hidden) while Library/Assets is active.
    expect(imageStudio()).not.toBeNull();

    await clickButton("Image");
    // Same DOM node → the studio was never unmounted/remounted.
    expect(imageStudio()).toBe(studioBefore);
    // And its React state survived with no re-hydrate from (now-empty) localStorage.
    expect(promptField().value).toBe("a keep-alive draft prompt");
  });

  it("unmounts OUT screens on navigation but keeps IN screens mounted (hidden)", async () => {
    await renderApp();

    // OUT screen: Models mounts when active, unmounts on navigation away.
    await clickButton("Models");
    expect(modelsSurface()).not.toBeNull();
    await clickButton("Assets");
    expect(modelsSurface()).toBeNull();

    // IN screen: Image mounts when active and STAYS mounted after navigating away.
    await clickButton("Image");
    expect(imageStudio()).not.toBeNull();
    await clickButton("Models");
    expect(modelsSurface()).not.toBeNull(); // OUT screen back
    expect(imageStudio()).not.toBeNull(); // IN screen still resident (hidden)
  });

  it("resets a kept-alive studio when the project changes (key change remounts it)", async () => {
    mockFetch({
      projects: [
        { id: "project-1", name: "Project One" },
        { id: "project-2", name: "Project Two" },
      ],
    });
    await renderApp();
    await clickButton("Image");

    const studioBefore = imageStudio();
    const defaultPrompt = promptField().value;
    await changeField(promptField(), "project-1 in-progress prompt");
    expect(promptField().value).toBe("project-1 in-progress prompt");

    // Switch workspace via the project switcher.
    await act(async () => {
      document.body.querySelector(".project-pill")?.click();
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll(".project-menu-item")]
        .find((item) => item.textContent.includes("Project Two"))
        ?.click();
    });
    await settle();

    // key={activeProject.id} changed → the studio remounts (a fresh DOM node) and drops
    // the in-progress prompt back to the new workspace's defaults.
    const studioAfter = imageStudio();
    expect(studioAfter).not.toBeNull();
    expect(studioAfter).not.toBe(studioBefore);
    expect(promptField().value).toBe(defaultPrompt);
    expect(promptField().value).not.toBe("project-1 in-progress prompt");
  });

  it("DocumentStudio survives a same-project nav round trip but RESETS on a project switch (sc-11959)", async () => {
    // DocumentStudio holds project-scoped local state (prompt + sourceAssetIds) with no
    // activeProject.id reset effect, so like its sibling studios it must be keyed on the
    // project id — otherwise a project switch keeps the previous project's draft + source-
    // asset selection and submit() would stamp those stale ids under the NEW project.
    mockFetch({
      projects: [
        { id: "project-1", name: "Project One" },
        { id: "project-2", name: "Project Two" },
      ],
      // Compose form only renders behind the model-availability gate, so provide an
      // interleave-capable model.
      models: [
        {
          id: "sensenova_u1",
          name: "SenseNova-U1",
          type: "image",
          family: "sensenova",
          capabilities: ["interleave"],
        },
      ],
    });
    await renderApp();
    await clickButton("Document");

    const studioBefore = documentStudio();
    expect(studioBefore).not.toBeNull();
    expect(documentPromptField().value).toBe("");

    await changeField(documentPromptField(), "project-1 illustrated guide");
    expect(documentPromptField().value).toBe("project-1 illustrated guide");

    // Same-project nav round trip: the studio stays mounted (kept alive), so its in-
    // progress draft survives with no remount.
    await clickButton("Assets");
    expect(documentStudio()).not.toBeNull();
    await clickButton("Document");
    expect(documentStudio()).toBe(studioBefore);
    expect(documentPromptField().value).toBe("project-1 illustrated guide");

    // Switch workspace via the project switcher.
    await act(async () => {
      document.body.querySelector(".project-pill")?.click();
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll(".project-menu-item")]
        .find((item) => item.textContent.includes("Project Two"))
        ?.click();
    });
    await settle();

    // key={activeProject.id} changed → DocumentStudio remounts (a fresh DOM node) and its
    // project-scoped prompt drops back to the new workspace's defaults, so no stale draft
    // or sourceAssetIds can leak across the project boundary.
    const studioAfter = documentStudio();
    expect(studioAfter).not.toBeNull();
    expect(studioAfter).not.toBe(studioBefore);
    expect(documentPromptField().value).toBe("");
    expect(documentPromptField().value).not.toBe("project-1 illustrated guide");
  });

  it("Pose Library survives a same-project nav round trip but RESETS on a project switch (sc-11971)", async () => {
    // The Pose Library Create tab holds PROJECT-SCOPED local state (staged sources / phase),
    // so — like the studios — it must be keyed on the project id: keep-alive preserves an
    // in-progress review across a plain nav, but a project switch remounts (resets) it so
    // project-1 source picks can't submit a pose_detect job under project-2.
    mockFetch({
      projects: [
        { id: "project-1", name: "Project One" },
        { id: "project-2", name: "Project Two" },
      ],
    });
    await renderApp();
    await clickButton("Pose Library");

    const surfaceBefore = document.body.querySelector(".pose-library-surface");
    expect(surfaceBefore).not.toBeNull();

    // Same-project nav round trip: kept alive → the SAME DOM node (in-progress Create state
    // rides along with it, no remount).
    await clickButton("Assets");
    expect(document.body.querySelector(".pose-library-surface")).not.toBeNull();
    await clickButton("Pose Library");
    expect(document.body.querySelector(".pose-library-surface")).toBe(surfaceBefore);

    // Switch workspace via the project switcher.
    await act(async () => {
      document.body.querySelector(".project-pill")?.click();
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll(".project-menu-item")]
        .find((item) => item.textContent.includes("Project Two"))
        ?.click();
    });
    await settle();

    // key={activeProject.id} changed → the screen remounts (a fresh DOM node), clearing the
    // Create tab's project-scoped sources/phase.
    const surfaceAfter = document.body.querySelector(".pose-library-surface");
    expect(surfaceAfter).not.toBeNull();
    expect(surfaceAfter).not.toBe(surfaceBefore);
  });

  it("Key Point Library (GLOBAL) survives BOTH a nav round trip and a project switch (sc-11971)", async () => {
    // The Key Point Library addresses the GLOBAL keypoints project, so its capture review /
    // in-progress collection must NOT be dropped by a project switch — it is deliberately
    // NOT keyed on the project id, so keep-alive preserves it across nav AND project change.
    mockFetch({
      projects: [
        { id: "project-1", name: "Project One" },
        { id: "project-2", name: "Project Two" },
      ],
    });
    await renderApp();
    await clickButton("Key Point Library");

    const surfaceBefore = document.body.querySelector(".keypoint-library-surface");
    expect(surfaceBefore).not.toBeNull();

    await clickButton("Assets");
    await clickButton("Key Point Library");
    expect(document.body.querySelector(".keypoint-library-surface")).toBe(surfaceBefore);

    // Switch workspace.
    await act(async () => {
      document.body.querySelector(".project-pill")?.click();
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll(".project-menu-item")]
        .find((item) => item.textContent.includes("Project Two"))
        ?.click();
    });
    await settle();

    // Not keyed on the project → the GLOBAL screen is the SAME node; its work survives.
    expect(document.body.querySelector(".keypoint-library-surface")).toBe(surfaceBefore);
  });

  it("applies a 'Use this recipe' injection on an ALREADY-mounted studio (token-id re-fire)", async () => {
    const generatedAsset = {
      id: "asset-recipe",
      projectId: "project-1",
      generationSetId: "genset-recipe",
      type: "image",
      displayName: "Atrium still",
      file: { path: "assets/images/atrium.png", mimeType: "image/png" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
      generationSet: {
        recipe: {
          mode: "text_to_image",
          model: "z_image_turbo",
          prompt: "mist over a glass atrium",
          negativePrompt: "flat lighting",
          seed: 1234,
          normalizedSettings: { width: 1024, height: 1024, count: 2 },
          rawAdapterSettings: { steps: 14, guidanceScale: 2.5 },
        },
      },
    };
    mockFetch({ assets: [generatedAsset] });
    await renderApp();

    // Mount the studio FIRST and leave a local edit on it, so the recipe must apply to an
    // already-mounted (kept-alive) studio rather than seeding a fresh mount.
    await clickButton("Image");
    const studioBefore = imageStudio();
    await changeField(promptField(), "my own untouched draft");
    expect(promptField().value).toBe("my own untouched draft");

    // Open the asset in the fullscreen preview and inject its recipe into the studio.
    await clickButton("Assets");
    await act(async () => {
      document.body.querySelector(".asset-tile")?.dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    });
    await settle();
    await clickButton("Use this recipe");

    // The studio was NOT remounted (same node), yet the recipe applied because its apply
    // effect is keyed on the studioLaunch token id, which changed.
    expect(imageStudio()).toBe(studioBefore);
    expect(promptField().value).toBe("mist over a glass atrium");
  });

  it("toggles a general preset into the stack on an already-mounted studio (epic 11949 launch)", async () => {
    mockFetch({
      presets: [
        {
          id: "gp-filmic",
          name: "Filmic Look",
          kind: "general",
          scope: "global",
          defaults: { aspect: "16:9", prompt: "cinematic, filmic grade" },
        },
      ],
    });
    await renderApp();

    // Mount the studio; the general chip is available but NOT active yet.
    await clickButton("Image");
    const studioBefore = imageStudio();
    const generalChip = () =>
      [...document.body.querySelectorAll(".general-preset-chips .preset-chip")].find((chip) =>
        chip.textContent.includes("Filmic Look"),
      );
    expect(generalChip()).toBeTruthy();
    expect(generalChip().classList.contains("active")).toBe(false);

    // Launch the general preset from the Preset Manager ("Use in Studio"): epic 11949's
    // sendPresetToStudio stamps a fresh studioLaunch token that toggles it into the stack.
    // (The button pairs an icon with the label, so match on trimmed text.)
    await clickButton("Presets");
    await act(async () => {
      [...document.body.querySelectorAll("button")]
        .find((button) => button.textContent.trim() === "Use in Studio")
        ?.click();
    });
    await settle();

    // Back on the already-mounted studio (same node) — the stack-toggle launch re-fired on
    // the token-id change and the general preset is now active in the stack.
    expect(imageStudio()).toBe(studioBefore);
    expect(generalChip().classList.contains("active")).toBe(true);
  });
});
