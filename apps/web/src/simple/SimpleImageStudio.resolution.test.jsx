import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";
import { SimpleImageStudio } from "./SimpleImageStudio.jsx";
import { SimpleUiContext } from "./SimpleUiContext.js";
import { createStudioStateStore } from "./useStudioState.js";

vi.mock("../api.js", async (importOriginal) => ({
  ...(await importOriginal()),
  apiFetch: vi.fn(async () => ({})),
}));

const model = (id, resolutions) => ({
  id, name: id, type: "image", family: "z-image",
  capabilities: ["text_to_image"],
  defaults: { resolution: "1024x1024" },
  limits: { resolutions },
});

describe("SimpleImageStudio resolution continuity", () => {
  let container;
  let root;
  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });
  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  it.each([
    ["2048x1408", ["1024x1024", "2048x1408"], "2048x1408"],
    ["2048x1408", ["1024x1024", "1536x1024", "2048x1440"], "2048x1440"],
    ["1408x2048", ["1024x1024", "1024x1536", "1536x1024"], "1024x1536"],
    ["", ["1536x1024", "1024x1024"], "1024x1024"],
  ])("preserves or approximates %s when switching models", async (previous, options, expected) => {
    const studioState = createStudioStateStore();
    studioState.values.set("image:model", "source");
    studioState.values.set("image:resolution", previous);
    const openSheet = vi.fn();
    const models = [model("source", ["1024x1024", previous].filter(Boolean)), model("destination", options)];
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ imageModels: models, models, activeProject: { id: "project-1" } }}>
          <SimpleUiContext.Provider value={{ studioState, breakpoint: "desktop", openSheet, closeSheet: vi.fn(), toast: vi.fn() }}>
            <SimpleImageStudio />
          </SimpleUiContext.Provider>
        </AppContext.Provider>,
      );
    });
    await click([...container.querySelectorAll(".su-field")].find((entry) => entry.querySelector("label")?.textContent === "Model").querySelector("button"));
    await act(async () => openSheet.mock.calls.at(-1)[0].onSelect("destination"));
    expect(studioState.values.get("image:resolution")).toBe(expected);
  });
});
