import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The durable half of the advanced studio settings (sc-15425). localStorage alone does not
// survive a desktop relaunch — the shell's `127.0.0.1:<port>` origin changes each launch and
// takes origin-keyed storage with it — so the settings are mirrored into `ui-preferences.json`
// and re-seeded into the cache on launch. These tests cover the seam, not the studios.
const persistNavigationPreferences = vi.fn(() => Promise.resolve());
vi.mock("../uiPreferences.js", () => ({
  persistNavigationPreferences: (...args) => persistNavigationPreferences(...args),
  putUiPreferences: vi.fn(() => Promise.resolve()),
}));

const { loadStudioSettings, seedStudioSettingsFromServer, useStudioSettingsWriter } =
  await import("./useStudioSettings.js");

function Harness({ settings, ready = true, studio = "image", workspace = "ws-1" }) {
  useStudioSettingsWriter(studio, workspace, settings, ready);
  return null;
}

/** The `advancedStudio` map from the most recent PUT, or null if none was sent. */
function lastSentMap() {
  const call = persistNavigationPreferences.mock.calls.at(-1);
  return call ? call[0].advancedStudio : null;
}

describe("advanced studio settings are durable (sc-15425)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    vi.useFakeTimers();
    window.localStorage.clear();
    persistNavigationPreferences.mockClear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    window.localStorage.clear();
    vi.useRealTimers();
  });

  async function render(element) {
    await act(async () => {
      root.render(element);
    });
    // The write-through is debounced; advance past it.
    await act(async () => {
      vi.advanceTimersByTime(500);
    });
  }

  it("mirrors a studio snapshot into the durable map, keyed by workspace", async () => {
    await render(<Harness settings={{ prompt: "a fox", model: "z_image" }} />);
    expect(lastSentMap()["ws-1"].image).toMatchObject({ prompt: "a fox", model: "z_image" });
  });

  it("does not enqueue a durable write when mounting an unchanged cached snapshot", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-image-ws-1",
      JSON.stringify({ prompt: "a fox", model: "z_image" }),
    );
    await render(<Harness settings={{ prompt: "a fox", model: "z_image" }} />);
    expect(persistNavigationPreferences).not.toHaveBeenCalled();
  });

  it("persists an absent empty snapshot but skips a cached empty snapshot", async () => {
    await render(<Harness settings={{}} />);
    expect(lastSentMap()["ws-1"].image).toEqual({});

    await act(async () => root.unmount());
    persistNavigationPreferences.mockClear();
    window.localStorage.setItem("sceneworks-studio-image-ws-1", JSON.stringify({}));
    root = createRoot(container);
    await render(<Harness settings={{}} />);
    expect(persistNavigationPreferences).not.toHaveBeenCalled();
  });

  it("still writes when a cached snapshot changes or readiness resumes", async () => {
    window.localStorage.setItem("sceneworks-studio-image-ws-1", JSON.stringify({ prompt: "old" }));
    await render(<Harness settings={{ prompt: "new" }} />);
    expect(lastSentMap()["ws-1"].image.prompt).toBe("new");

    persistNavigationPreferences.mockClear();
    await render(<Harness ready={false} settings={{ prompt: "new" }} />);
    await render(<Harness ready settings={{ prompt: "new" }} />);
    expect(lastSentMap()["ws-1"].image.prompt).toBe("new");
  });

  it("restores the cache from the durable copy after a relaunch", async () => {
    await render(<Harness settings={{ prompt: "a fox", model: "z_image" }} />);
    const durable = lastSentMap();

    // Relaunch: new origin, so the cache is gone. Only the server copy survives.
    window.localStorage.clear();
    expect(loadStudioSettings("image", "ws-1")).toEqual({});

    expect(seedStudioSettingsFromServer(durable)).toBeGreaterThan(0);
    expect(loadStudioSettings("image", "ws-1")).toMatchObject({
      prompt: "a fox",
      model: "z_image",
    });
  });

  // The sizing constraint that drove the design: ImageStudio persists ~47 fields, two of them
  // unbounded free text. Including them would push a workspace past the server's per-entry
  // budget, at which point the entry is dropped WHOLE and nothing restores.
  it("keeps the unbounded batch fields OUT of the durable copy but IN the cache", async () => {
    await render(
      <Harness
        settings={{
          prompt: "a fox",
          batchPromptsText: "line one\nline two\nline three",
          batchVariableValues: { subject: ["a", "b"] },
        }}
      />,
    );
    const sent = lastSentMap()["ws-1"].image;
    expect(sent.prompt).toBe("a fox");
    expect(sent).not.toHaveProperty("batchPromptsText");
    expect(sent).not.toHaveProperty("batchVariableValues");

    // Still in the cache, which is what serves within-session navigation.
    const cached = loadStudioSettings("image", "ws-1");
    expect(cached.batchPromptsText).toBe("line one\nline two\nline three");
  });

  // sc-11962's window, one step further out: before the ui-preferences GET lands the cache may
  // be empty, so a studio persisting during that window would overwrite the durable copy with
  // catalog defaults before anything had read it.
  it("writes nothing at all while `ready` is false", async () => {
    await render(<Harness ready={false} settings={{ prompt: "typed before hydration" }} />);
    expect(persistNavigationPreferences).not.toHaveBeenCalled();
    expect(window.localStorage.getItem("sceneworks-studio-image-ws-1")).toBeNull();
  });

  it("keeps workspaces separate in the durable map", async () => {
    await render(<Harness settings={{ prompt: "one" }} workspace="ws-1" />);
    await render(<Harness settings={{ prompt: "two" }} workspace="ws-2" />);
    const map = lastSentMap();
    expect(map["ws-1"].image.prompt).toBe("one");
    expect(map["ws-2"].image.prompt).toBe("two");
  });

  it("carries every studio scope in one map", async () => {
    await render(<Harness settings={{ prompt: "img" }} studio="image" />);
    await render(<Harness settings={{ prompt: "vid" }} studio="video" />);
    await render(<Harness settings={{ activeTab: "looks" }} studio="character" />);
    const entry = lastSentMap()["ws-1"];
    expect(entry.image.prompt).toBe("img");
    expect(entry.video.prompt).toBe("vid");
    expect(entry.character.activeTab).toBe("looks");
  });

  // The cache holds fields the durable copy deliberately doesn't, so seeding MERGES rather than
  // replaces — otherwise every launch would drop an in-progress batch sheet.
  it("merges the seed into the cache instead of replacing it", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-image-ws-1",
      JSON.stringify({ batchPromptsText: "kept", prompt: "stale" }),
    );
    seedStudioSettingsFromServer({ "ws-1": { image: { prompt: "restored", model: "z_image" } } });
    const cached = loadStudioSettings("image", "ws-1");
    expect(cached.prompt).toBe("restored");
    expect(cached.model).toBe("z_image");
    expect(cached.batchPromptsText).toBe("kept");
  });

  it("ignores a malformed durable map rather than corrupting the cache", async () => {
    expect(seedStudioSettingsFromServer(null)).toBe(0);
    expect(seedStudioSettingsFromServer({ "ws-1": "not-an-object" })).toBe(0);
    expect(seedStudioSettingsFromServer({ "ws-1": { image: 42 } })).toBe(0);
    expect(loadStudioSettings("image", "ws-1")).toEqual({});
  });
});
