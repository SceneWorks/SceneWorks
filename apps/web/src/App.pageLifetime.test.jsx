import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { FakeEventSource, response, settle, changeField, field } from "./main.testSupport.jsx";

describe("foreground page lifetime", () => {
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

  const documentPromptField = () => document.body.querySelector(".document-studio .studio-form textarea");

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    window.sessionStorage.clear();
    mockFetch();
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it("mounts a creative page only when selected", async () => {
    await renderApp();

    expect(imageStudio()).toBeNull();
    expect(videoStudio()).toBeNull();

    await clickButton("Image");
    expect(imageStudio()).not.toBeNull();

    expect(videoStudio()).toBeNull();
  });

  it("unmounts Image Studio and restores its saved prompt on return", async () => {
    window.localStorage.setItem("sceneworks-studio-image-project-1", JSON.stringify({
      model: "z_image_turbo", resolution: "1024x1024", seed: "8675309", count: 3,
    }));
    await renderApp();
    await clickButton("Image");

    const studioBefore = imageStudio();
    expect(studioBefore).not.toBeNull();

    await changeField(promptField(), "a keep-alive draft prompt");
    expect(promptField().value).toBe("a keep-alive draft prompt");
    expect(field(imageStudio(), "Variations").value).toBe("3");

    await clickButton("Assets");

    expect(imageStudio()).toBeNull();

    await clickButton("Image");

    expect(imageStudio()).not.toBe(studioBefore);

    expect(promptField().value).toBe("a keep-alive draft prompt");
    expect(field(imageStudio(), "Variations").value).toBe("3");
    expect(JSON.parse(window.localStorage.getItem("sceneworks-studio-image-project-1")))
      .toMatchObject({ model: "z_image_turbo", resolution: "1024x1024", seed: "8675309", count: 3 });
  });

  it("restores the batch prompt sheet and variables without retaining the page", async () => {
    window.localStorage.setItem("sceneworks-studio-image-project-1", JSON.stringify({
      model: "z_image_turbo", batchMode: true, batchPromptsText: "A {{coat}} coat", batchVariableValues: { coat: ["red", "blue"] },
    }));
    await renderApp();
    await clickButton("Image");
    await changeField(document.querySelector('[aria-label="Batch prompts"]'), "A {{coat}} coat\nA {{coat}} scarf");
    await clickButton("Queue");
    expect(imageStudio()).toBeNull();
    await clickButton("Image");
    expect(document.querySelector('[aria-label="Batch prompts"]').value).toBe("A {{coat}} coat\nA {{coat}} scarf");
    expect(JSON.parse(window.localStorage.getItem("sceneworks-studio-image-project-1"))).toMatchObject({ batchVariableValues: { coat: ["red", "blue"] } });
  });

  it("unmounts both catalog pages and creative pages on navigation", async () => {
    await renderApp();

    await clickButton("Models");
    expect(modelsSurface()).not.toBeNull();
    await clickButton("Assets");
    expect(modelsSurface()).toBeNull();

    await clickButton("Image");
    expect(imageStudio()).not.toBeNull();
    await clickButton("Models");
    expect(modelsSurface()).not.toBeNull(); // OUT screen back
    expect(imageStudio()).toBeNull();
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

    const studioAfter = imageStudio();
    expect(studioAfter).not.toBeNull();
    expect(studioAfter).not.toBe(studioBefore);
    expect(promptField().value).toBe(defaultPrompt);
    expect(promptField().value).not.toBe("project-1 in-progress prompt");
  });

  it("DocumentStudio survives a same-project nav round trip but RESETS on a project switch (sc-11959)", async () => {

    mockFetch({
      projects: [
        { id: "project-1", name: "Project One" },
        { id: "project-2", name: "Project Two" },
      ],

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

    await clickButton("Assets");
    expect(documentStudio()).toBeNull();
    await clickButton("Document");
    expect(documentStudio()).not.toBe(studioBefore);
    expect(documentPromptField().value).toBe("project-1 illustrated guide");

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

    const studioAfter = documentStudio();
    expect(studioAfter).not.toBeNull();
    expect(studioAfter).not.toBe(studioBefore);
    expect(documentPromptField().value).toBe("");
    expect(documentPromptField().value).not.toBe("project-1 illustrated guide");
  });

  it("Pose Library survives a same-project nav round trip but RESETS on a project switch (sc-11971)", async () => {

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

    await clickButton("Assets");
    expect(document.body.querySelector(".pose-library-surface")).toBeNull();
    await clickButton("Pose Library");
    expect(document.body.querySelector(".pose-library-surface")).not.toBe(surfaceBefore);

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

    const surfaceAfter = document.body.querySelector(".pose-library-surface");
    expect(surfaceAfter).not.toBeNull();
    expect(surfaceAfter).not.toBe(surfaceBefore);
  });

  it("Key Point Library (GLOBAL) survives BOTH a nav round trip and a project switch (sc-11971)", async () => {

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
    expect(document.body.querySelector(".keypoint-library-surface")).not.toBe(surfaceBefore);

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

    expect(document.body.querySelector(".keypoint-library-surface")).not.toBe(surfaceBefore);
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

    await clickButton("Image");
    const studioBefore = imageStudio();
    await changeField(promptField(), "my own untouched draft");
    expect(promptField().value).toBe("my own untouched draft");

    await clickButton("Assets");
    await act(async () => {
      document.body.querySelector(".asset-tile")?.dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    });
    await settle();
    await clickButton("Use this recipe");

    expect(imageStudio()).not.toBe(studioBefore);
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

    await clickButton("Image");
    const studioBefore = imageStudio();
    const generalChip = () =>
      [...document.body.querySelectorAll(".general-preset-chips .preset-chip")].find((chip) =>
        chip.textContent.includes("Filmic Look"),
      );
    expect(generalChip()).toBeTruthy();
    expect(generalChip().classList.contains("active")).toBe(false);

    await clickButton("Presets");
    await act(async () => {
      [...document.body.querySelectorAll("button")]
        .find((button) => button.textContent.trim() === "Use in Studio")
        ?.click();
    });
    await settle();

    expect(imageStudio()).not.toBe(studioBefore);
    expect(generalChip().classList.contains("active")).toBe(true);
  });

  it("does NOT re-apply a stale launch token on a nav round trip (an edit over an injected recipe survives)", async () => {

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

    await clickButton("Image");
    const studioBefore = imageStudio();
    await clickButton("Assets");
    await act(async () => {
      document.body.querySelector(".asset-tile")?.dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    });
    await settle();
    await clickButton("Use this recipe");
    expect(imageStudio()).not.toBe(studioBefore);
    expect(promptField().value).toBe("mist over a glass atrium");

    await changeField(promptField(), "my edit after the recipe applied");
    expect(promptField().value).toBe("my edit after the recipe applied");

    await clickButton("Assets");
    await clickButton("Image");

    expect(imageStudio()).not.toBe(studioBefore);
    expect(promptField().value).toBe("my edit after the recipe applied");
  });
});
