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

  it("flushes the latest debounced snapshot when the studio unmounts", async () => {
    await act(async () => root.render(<Harness settings={{ prompt: "first" }} />));
    await act(async () => root.render(<Harness settings={{ prompt: "latest" }} />));
    expect(persistNavigationPreferences).not.toHaveBeenCalled();
    await act(async () => root.unmount());
    expect(lastSentMap()["ws-1"].image.prompt).toBe("latest");
    root = createRoot(container);
  });

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

  it("lets server canonical options supersede cached aliases on successive hydrations", () => {
    const key = "sceneworks-studio-image-ws-1";
    window.localStorage.setItem(key, JSON.stringify({ timestepShift: 3, batchPromptsText: "kept" }));
    const hydrate = (image) => seedStudioSettingsFromServer({ "ws-1": { image } });

    expect(hydrate({ schedulerShift: 4 })).toBe(1);
    expect(loadStudioSettings("image", "ws-1")).toEqual({ schedulerShift: 4, batchPromptsText: "kept" });
    // A later historical server snapshot can reintroduce an alias. Its next canonical
    // replacement must work too, including an explicit null rather than an omitted option.
    expect(hydrate({ timestepShift: 5 })).toBe(1);
    expect(loadStudioSettings("image", "ws-1")).toMatchObject({ timestepShift: 5, schedulerShift: 5 });
    expect(hydrate({ schedulerShift: null })).toBe(1);
    expect(loadStudioSettings("image", "ws-1")).toEqual({ schedulerShift: null, batchPromptsText: "kept" });
    expect(hydrate({ prompt: "restored" })).toBe(1);
    expect(loadStudioSettings("image", "ws-1")).toEqual({ schedulerShift: null, batchPromptsText: "kept", prompt: "restored" });
  });

  it("refuses conflicts within either settings source without changing the cached original", () => {
    const key = "sceneworks-studio-image-ws-1";
    for (const [cached, restored] of [
      [{ timestepShift: 3 }, { timestepShift: 4, schedulerShift: 5 }],
      [{ timestepShift: 3, schedulerShift: 4 }, { schedulerShift: 5 }],
    ]) {
      const original = JSON.stringify(cached);
      window.localStorage.setItem(key, original);
      expect(() => seedStudioSettingsFromServer({ "ws-1": { image: restored } }))
        .toThrow(expect.objectContaining({ code: "ConflictingAliases", field: "schedulerShift" }));
      expect(window.localStorage.getItem(key)).toBe(original);
    }
  });

  it("blocks every cache and durable write after a refused server snapshot", async () => {
    const key = "sceneworks-studio-image-ws-1";
    const valid = { "ws-1": { image: { prompt: "server value" } } };
    try {
      expect(() => seedStudioSettingsFromServer({
        "ws-1": { image: { prompt: "valid" }, video: { optionsSchemaVersion: 9, prompt: "future" } },
      })).toThrow(expect.objectContaining({ code: "UnsupportedSchemaVersion" }));
      expect(window.localStorage.getItem(key)).toBeNull();
      await render(<Harness settings={{ prompt: "form fallback" }} />);
      expect(window.localStorage.getItem(key)).toBeNull();
      expect(persistNavigationPreferences).not.toHaveBeenCalled();

      // A later compatible server snapshot can restore writes.
      expect(seedStudioSettingsFromServer(valid)).toBe(1);
      await render(<Harness settings={{ prompt: "edited" }} />);
      expect(lastSentMap()["ws-1"].image.prompt).toBe("edited");
    } finally {
      seedStudioSettingsFromServer({});
    }
  });

  it("does not replace the durable map when localStorage cannot be written", async () => {
    const setItem = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("storage unavailable");
    });
    try {
      await render(<Harness settings={{ prompt: "server only" }} />);
      expect(persistNavigationPreferences).not.toHaveBeenCalled();
    } finally {
      setItem.mockRestore();
    }
  });

  // sc-23000: the YuE2 Song Lab's own snapshot (`yue2lab`) rides the same durable map and is seeded
  // back on launch, and a full-size ABC score stays out of the durable copy (it would push the whole
  // workspace entry past the server's per-entry budget, dropping every studio in it).
  it("mirrors and restores the YuE2 lab snapshot, keeping a full-size score session-only", async () => {
    const bigScore = `X:1\n${"C4|".repeat(8000)}`;
    await render(
      <Harness
        settings={{ optIn: true, lyrics: "[verse]\nhello", suppliedScore: bigScore, coverScore: "X:1\nC4|" }}
        studio="yue2lab"
      />,
    );
    const sent = lastSentMap()["ws-1"].yue2lab;
    expect(sent).toMatchObject({ optIn: true, lyrics: "[verse]\nhello", coverScore: "X:1\nC4|" });
    expect(sent).not.toHaveProperty("suppliedScore");
    // The session cache keeps it, and no other studio key is confused with the lab's.
    expect(loadStudioSettings("yue2lab", "ws-1").suppliedScore).toBe(bigScore);
    expect(lastSentMap()["ws-1"]).not.toHaveProperty("audio");

    window.localStorage.clear();
    expect(seedStudioSettingsFromServer({ "ws-1": { yue2lab: sent } })).toBe(1);
    expect(loadStudioSettings("yue2lab", "ws-1")).toMatchObject({ optIn: true, lyrics: "[verse]\nhello" });
  });

  // sc-23000 fix pass: the server budgets UTF-8 BYTES (128 KiB for the whole workspace entry) and
  // drops an over-budget entry WHOLE — which would lose every other studio's settings too. Twelve
  // legacy presets each carrying a 16 000-character CJK lyric sheet (~48 KiB apiece in UTF-8) must
  // not take the workspace's standard Audio settings down with them.
  it("bounds the YuE2 lab snapshot in UTF-8 bytes so the workspace entry survives", async () => {
    window.localStorage.setItem(
      "sceneworks-studio-audio-ws-1",
      JSON.stringify({ mode: "speech", model: "kokoro_82m", prompt: "keep me" }),
    );
    const cjk = "歌".repeat(16000);
    const presets = Array.from({ length: 12 }, (_, index) => ({
      id: `p${index}`,
      name: `Preset ${index}`,
      settings: { lyrics: cjk, tier: "q8" },
    }));
    await render(
      <Harness settings={{ optIn: true, tier: "q8", lyrics: cjk, compareNotes: cjk, presets }} studio="yue2lab" />,
    );
    const entry = lastSentMap()["ws-1"];
    const bytes = new TextEncoder().encode(JSON.stringify(entry)).length;
    expect(bytes).toBeLessThanOrEqual(128 * 1024);
    expect(entry.audio).toMatchObject({ prompt: "keep me" });
    expect(entry.yue2lab).toMatchObject({ optIn: true, tier: "q8" });
    expect(new TextEncoder().encode(JSON.stringify(entry.yue2lab)).length).toBeLessThanOrEqual(32 * 1024);
    // 16 000 CJK characters are under a 16 KiB CHARACTER count but ~47 KiB in bytes: dropped.
    expect(entry.yue2lab).not.toHaveProperty("lyrics");
  });

  // The per-field cap is UTF-8 bytes too: 8 000 CJK characters are well under 16 KiB of UTF-16
  // units but ~23 KiB on the wire, and the snapshot as a whole is under its 32 KiB budget — so only
  // a byte-measured field cap drops it.
  it("caps a lyric sheet by its UTF-8 size, not its character count", async () => {
    const lyrics = "歌".repeat(8000);
    await render(<Harness settings={{ optIn: true, lyrics, style: "lofi" }} studio="yue2lab" />);
    const sent = lastSentMap()["ws-1"].yue2lab;
    expect(sent).not.toHaveProperty("lyrics");
    expect(sent).toMatchObject({ optIn: true, style: "lofi" });
  });

  it("ignores a malformed durable map rather than corrupting the cache", async () => {
    expect(seedStudioSettingsFromServer(null)).toBe(0);
    expect(seedStudioSettingsFromServer({ "ws-1": "not-an-object" })).toBe(0);
    expect(seedStudioSettingsFromServer({ "ws-1": { image: 42 } })).toBe(0);
    expect(loadStudioSettings("image", "ws-1")).toEqual({});
  });
});
