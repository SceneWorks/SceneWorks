import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { loadStudioSettings, useStudioSettingsWriter } from "./useStudioSettings.js";

// sc-13130: the Image Studio persists its Style Catalog selection (`styleId`) through the SAME
// saved-state mechanism the prompt uses, so it survives a reload. This exercises that path end to
// end: write a settings snapshot carrying styleId, then load it back for the same workspace.
function Harness({ settings }) {
  useStudioSettingsWriter("image", "ws-style-test", settings, true);
  return null;
}

describe("studio saved-state carries styleId (sc-13130)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    window.localStorage.clear();
  });

  it("persists a selected styleId and rehydrates it on load", async () => {
    await act(async () => {
      root.render(<Harness settings={{ prompt: "a fox", styleId: "ghibli-style" }} />);
    });
    const restored = loadStudioSettings("image", "ws-style-test");
    expect(restored.styleId).toBe("ghibli-style");
    // The picker reads `saved.styleId ?? null`, so a persisted null is fine too.
    expect(restored.prompt).toBe("a fox");
  });

  it("rehydrates null (None) as pass-through", async () => {
    await act(async () => {
      root.render(<Harness settings={{ prompt: "a fox", styleId: null }} />);
    });
    const restored = loadStudioSettings("image", "ws-style-test");
    expect(restored.styleId).toBeNull();
    expect(restored.styleId ?? null).toBeNull();
  });

  it("a snapshot with no styleId key restores as null (?? null default)", () => {
    window.localStorage.setItem(
      "sceneworks-studio-image-ws-style-test",
      JSON.stringify({ prompt: "legacy prompt, pre-style-picker" }),
    );
    const restored = loadStudioSettings("image", "ws-style-test");
    expect(restored.styleId ?? null).toBeNull();
  });
});
