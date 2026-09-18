// sc-11967 (S8): re-selecting a timeline from the editor dropdown reloads the server copy
// and replaces the in-memory working copy. When the active timeline has unsaved edits that
// silently dropped them. These tests assert the dropdown now guards the switch with the
// app's discard-edits confirm when dirty, and behaves exactly as before when clean.
import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppContext } from "../context/AppContext.js";
import { ScreenActiveContext } from "../context/ScreenActiveContext.js";

// sc-12018 (S8 follow-up): the re-select guard now routes through the desktop-safe appConfirm
// dialog (not the raw window.confirm, which no-ops in the Tauri WebView). Mock it so a test
// controls the user's choice and can assert the guard fired, without mounting a ConfirmHost.
const { appConfirmMock } = vi.hoisted(() => ({ appConfirmMock: vi.fn(async () => true) }));
vi.mock("../appConfirm.jsx", () => ({ appConfirm: appConfirmMock }));

import { EditorScreen } from "./EditorScreen.jsx";

function installAudioContext(events = []) {
  const contexts = [];
  const node = () => ({ connect: vi.fn(), disconnect: vi.fn() });
  class Context {
    constructor() {
      this.destination = {};
      this.currentTime = 0;
      this.gains = [];
      this.createGain = vi.fn(() => {
        const gain = { ...node(), gain: { value: 1, cancelScheduledValues: vi.fn(), setValueAtTime: vi.fn() } };
        this.gains.push(gain);
        return gain;
      });
      this.createMediaElementSource = vi.fn(() => node());
      this.createDynamicsCompressor = vi.fn(() => ({ ...node(), ...Object.fromEntries(["threshold", "knee", "ratio", "attack", "release"].map((key) => [key, { value: 0 }])) }));
      this.resume = vi.fn(async () => { events.push("resume"); });
      this.close = vi.fn(async () => {});
      contexts.push(this);
    }
  }
  vi.stubGlobal("AudioContext", Context);
  return contexts;
}

function makeTimeline(id, name) {
  return { id, name, aspectRatio: "16:9", fps: 30, width: 1280, height: 720, tracks: [{ id: "track_main", name: "Main", items: [] }] };
}

let container;
let root;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  appConfirmMock.mockClear();
  appConfirmMock.mockResolvedValue(true);
  container = document.createElement("div");
  document.body.appendChild(container);
});

// The re-select guard awaits appConfirm, so its setSelectedTimelineId runs a microtask later.
async function flush() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function render(overrides = {}) {
  const setSelectedTimelineId = vi.fn();
  const value = {
    activeProject: { id: "proj_1", name: "Proj" },
    activeTimeline: makeTimeline("tl_1", "Main"),
    mediaAssets: [],
    timelines: [makeTimeline("tl_1", "Main"), makeTimeline("tl_2", "Second")],
    selectedTimelineId: "tl_1",
    setSelectedTimelineId,
    setActiveTimeline: vi.fn(),
    setPreviewAsset: vi.fn(),
    sendAssetToImage: vi.fn(),
    sendAssetToVideo: vi.fn(),
    createTimeline: vi.fn(),
    extractTimelineFrame: vi.fn(),
    exportTimeline: vi.fn(),
    queueTimelineVideoJob: vi.fn(),
    saveTimeline: vi.fn(),
    isActiveTimelineDirty: () => false,
    ...overrides,
  };
  root = createRoot(container);
  act(() => {
    root.render(
      <AppContext.Provider value={value}>
        <EditorScreen />
      </AppContext.Provider>,
    );
  });
  return { setSelectedTimelineId };
}

// The timeline dropdown lives in the redesigned toolbar (design 2a). Firing a native change
// event with a new value routes through the guarded onChange.
function selectTimeline(nextId) {
  const select = container.querySelector(".ve-timeline-select");
  act(() => {
    select.value = nextId;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  return select;
}

describe("EditorScreen timeline re-select guard (sc-11967, sc-12018)", () => {
  it("switches without prompting when the timeline is clean (no regression)", async () => {
    const { setSelectedTimelineId } = render({ isActiveTimelineDirty: () => false });

    selectTimeline("tl_2");
    await flush();

    expect(appConfirmMock).not.toHaveBeenCalled();
    expect(setSelectedTimelineId).toHaveBeenCalledWith("tl_2");
  });

  it("does NOT switch (keeps the current timeline) when dirty and the user cancels the confirm", async () => {
    appConfirmMock.mockResolvedValue(false);
    const { setSelectedTimelineId } = render({ isActiveTimelineDirty: () => true });

    const select = selectTimeline("tl_2");
    await flush();

    // sc-12018: the guard used the desktop-safe appConfirm dialog (not the raw window.confirm).
    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    expect(appConfirmMock.mock.calls[0][0]).toMatchObject({ tone: "danger" });
    expect(setSelectedTimelineId).not.toHaveBeenCalled();
    // Controlled <select> snaps back to the still-active timeline.
    expect(select.value).toBe("tl_1");
  });

  it("switches when dirty and the user accepts the confirm", async () => {
    appConfirmMock.mockResolvedValue(true);
    const { setSelectedTimelineId } = render({ isActiveTimelineDirty: () => true });

    selectTimeline("tl_2");
    await flush();

    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    expect(setSelectedTimelineId).toHaveBeenCalledWith("tl_2");
  });
});

// sc-11961 (S2): under keep-alive the editor stays mounted while another view is
// foregrounded. Its one continuous piece of work is preview playback (video.play()).
// These tests assert the play effect only drives the <video> when the screen is the
// active view, and pauses (does no playback work) while hidden.
describe("EditorScreen preview playback keep-alive gating (sc-11961)", () => {
  function makeVideoAsset() {
    return {
      id: "asset_v1",
      type: "video",
      displayName: "Clip A",
      url: "/api/v1/files/v1.mp4",
      file: { mimeType: "video/mp4", duration: 4 },
    };
  }

  function makeTimelineWithVideoItem() {
    return {
      id: "tl_1",
      name: "Main",
      aspectRatio: "16:9",
      fps: 30,
      width: 1280,
      height: 720,
      tracks: [
        {
          id: "track_main",
          name: "Main",
          items: [
            {
              id: "item_1",
              trackId: "track_main",
              assetId: "asset_v1",
              type: "video",
              displayName: "Clip A",
              sourceIn: 0,
              sourceOut: 4,
              timelineStart: 0,
              timelineEnd: 4,
              speed: 1,
              fit: "fit",
              volume: 1,
              versionAssetIds: ["asset_v1"],
              currentVersionAssetId: "asset_v1",
              versionHistory: [{ assetId: "asset_v1", createdAt: null, source: "original", jobId: null, note: null }],
              transitionIn: { id: "t_in", type: "cut", duration: 0 },
              transitionOut: { id: "t_out", type: "cut", duration: 0 },
            },
          ],
        },
      ],
    };
  }

  function renderEditor(screenActive) {
    const timeline = makeTimelineWithVideoItem();
    const value = {
      activeProject: { id: "proj_1", name: "Proj" },
      activeTimeline: timeline,
      mediaAssets: [makeVideoAsset()],
      timelines: [timeline],
      selectedTimelineId: "tl_1",
      setSelectedTimelineId: vi.fn(),
      setActiveTimeline: vi.fn(),
      setPreviewAsset: vi.fn(),
      sendAssetToImage: vi.fn(),
      sendAssetToVideo: vi.fn(),
      createTimeline: vi.fn(),
      extractTimelineFrame: vi.fn(),
      exportTimeline: vi.fn(),
      queueTimelineVideoJob: vi.fn(),
      saveTimeline: vi.fn(),
      isActiveTimelineDirty: () => false,
    };
    root = createRoot(container);
    act(() => {
      root.render(
        <AppContext.Provider value={value}>
          <ScreenActiveContext.Provider value={screenActive}>
            <EditorScreen />
          </ScreenActiveContext.Provider>
        </AppContext.Provider>,
      );
    });
  }

  // Select the timeline clip (renders the preview <video>) then click Play.
  function selectClipAndPressPlay() {
    const clip = container.querySelector(".ve-clip");
    act(() => clip.dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    const playButton = container.querySelector(".ve-play");
    act(() => playButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
  }

  it("plays the preview when it is the ACTIVE view", () => {
    const play = vi.spyOn(window.HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
    // Stub pause too (jsdom's is unimplemented); the effect pauses once on selection.
    vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(() => {});

    renderEditor(true);
    selectClipAndPressPlay();

    expect(play).toHaveBeenCalled();
  });

  it("does NOT play (only pauses) while the editor is HIDDEN under keep-alive", () => {
    const play = vi.spyOn(window.HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
    const pause = vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(() => {});

    renderEditor(false);
    selectClipAndPressPlay();

    // Pressing Play flips isPlaying, but the hidden screen's effect refuses to drive the
    // <video>: play is never invoked and the element is kept paused — no background decode.
    expect(play).not.toHaveBeenCalled();
    expect(pause).toHaveBeenCalled();
  });
});

// sc-13589 (F-008): under keep-alive the editor stays mounted and its window keydown
// listener stays subscribed while another view is foregrounded. These tests assert the
// timeline shortcuts fire only when the screen is the ACTIVE view — a backgrounded editor
// must never mutate the hidden timeline in response to keys meant for the visible screen.
// Mutation guard: with the screenActive gate removed, the "HIDDEN" case would run
// removeSelectedItem -> commit -> setActiveTimeline and the assertion below goes red.
describe("EditorScreen keep-alive keydown gating (sc-13589)", () => {
  function makeVideoAsset() {
    return {
      id: "asset_v1",
      type: "video",
      displayName: "Clip A",
      url: "/api/v1/files/v1.mp4",
      file: { mimeType: "video/mp4", duration: 4 },
    };
  }

  function makeTimelineWithVideoItem() {
    return {
      id: "tl_1",
      name: "Main",
      aspectRatio: "16:9",
      fps: 30,
      width: 1280,
      height: 720,
      tracks: [
        {
          id: "track_main",
          name: "Main",
          items: [
            {
              id: "item_1",
              trackId: "track_main",
              assetId: "asset_v1",
              type: "video",
              displayName: "Clip A",
              sourceIn: 0,
              sourceOut: 4,
              timelineStart: 0,
              timelineEnd: 4,
              speed: 1,
              fit: "fit",
              volume: 1,
              versionAssetIds: ["asset_v1"],
              currentVersionAssetId: "asset_v1",
              versionHistory: [{ assetId: "asset_v1", createdAt: null, source: "original", jobId: null, note: null }],
              transitionIn: { id: "t_in", type: "cut", duration: 0 },
              transitionOut: { id: "t_out", type: "cut", duration: 0 },
            },
          ],
        },
      ],
    };
  }

  function renderEditor(screenActive) {
    const setActiveTimeline = vi.fn();
    const timeline = makeTimelineWithVideoItem();
    const value = {
      activeProject: { id: "proj_1", name: "Proj" },
      activeTimeline: timeline,
      mediaAssets: [makeVideoAsset()],
      timelines: [timeline],
      selectedTimelineId: "tl_1",
      setSelectedTimelineId: vi.fn(),
      setActiveTimeline,
      setPreviewAsset: vi.fn(),
      sendAssetToImage: vi.fn(),
      sendAssetToVideo: vi.fn(),
      createTimeline: vi.fn(),
      extractTimelineFrame: vi.fn(),
      exportTimeline: vi.fn(),
      queueTimelineVideoJob: vi.fn(),
      saveTimeline: vi.fn(),
      isActiveTimelineDirty: () => false,
    };
    root = createRoot(container);
    act(() => {
      root.render(
        <AppContext.Provider value={value}>
          <ScreenActiveContext.Provider value={screenActive}>
            <EditorScreen />
          </ScreenActiveContext.Provider>
        </AppContext.Provider>,
      );
    });
    return { setActiveTimeline };
  }

  // Clicking the clip sets selectedItemId; Delete then routes through removeSelectedItem.
  function selectClipAndPressDelete() {
    const clip = container.querySelector(".ve-clip");
    act(() => clip.dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    act(() => window.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Delete", bubbles: true })));
  }

  it("deletes the selected clip on Delete while it is the ACTIVE view", () => {
    const { setActiveTimeline } = renderEditor(true);
    selectClipAndPressDelete();

    // removeSelectedItem -> commit -> setActiveTimeline with the selected clip filtered out.
    expect(setActiveTimeline).toHaveBeenCalledTimes(1);
    expect(setActiveTimeline.mock.calls[0][0].tracks[0].items).toHaveLength(0);
  });

  it("undo restores the pre-commit timeline snapshot", () => {
    const { setActiveTimeline } = renderEditor(true);
    selectClipAndPressDelete();
    act(() => {
      window.dispatchEvent(
        new window.KeyboardEvent("keydown", {
          key: "z",
          code: "KeyZ",
          metaKey: true,
          bubbles: true,
        }),
      );
    });

    expect(setActiveTimeline).toHaveBeenCalledTimes(2);
    expect(setActiveTimeline.mock.calls[1][0].tracks[0].items).toHaveLength(1);
    expect(setActiveTimeline.mock.calls[1][0].tracks[0].items[0].id).toBe("item_1");
  });

  it("ignores Delete while the editor is HIDDEN under keep-alive", () => {
    const { setActiveTimeline } = renderEditor(false);
    selectClipAndPressDelete();

    // The backgrounded editor's keydown handler early-returns, so the hidden timeline is
    // never mutated by a Delete meant for the visible screen.
    expect(setActiveTimeline).not.toHaveBeenCalled();
  });
});

describe("incremental film history (sc-23737)", () => {
  it("undo and redo a deletion after another shot arrives without erasing the new shot", async () => {
    let install, latest;
    const clip = (id, start) => ({ id, trackId: "track_main", assetId: id, displayName: id, type: "video", sourceIn: 0, sourceOut: 4, timelineStart: start, timelineEnd: start + 4, speed: 1, fit: "fit", volume: 1 });
    const original = makeTimeline("tl_1", "Film"); original.tracks[0].items = [clip("A",0)];
    function Harness() {
      const [timeline, setTimeline] = React.useState(original);
      install = setTimeline; latest = timeline;
      return <AppContext.Provider value={{ activeProject: { id: "proj_1" }, activeTimeline: timeline, mediaAssets: [], timelines: [timeline], selectedTimelineId: timeline.id,
        setActiveTimeline: setTimeline, setSelectedTimelineId: vi.fn(), setPreviewAsset: vi.fn(), createTimeline: vi.fn(), extractTimelineFrame: vi.fn(), exportTimeline: vi.fn(), queueTimelineVideoJob: vi.fn(), saveTimeline: vi.fn(), isActiveTimelineDirty: () => false }}>
        <EditorScreen />
      </AppContext.Provider>;
    }
    root = createRoot(container); act(() => root.render(<Harness />));
    act(() => container.querySelector(".ve-clip").dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    act(() => window.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Delete", bubbles: true })));
    expect(latest.tracks[0].items).toEqual([]);
    act(() => install((cut) => ({ ...cut, revision: 2, tracks: [{ ...cut.tracks[0], items: [clip("B",4)] }] })));
    act(() => window.dispatchEvent(new window.KeyboardEvent("keydown", { key: "z", code: "KeyZ", metaKey: true, bubbles: true })));
    await flush();
    expect(latest.tracks[0].items.map((i) => i.id).sort()).toEqual(["A", "B"]);
    act(() => window.dispatchEvent(new window.KeyboardEvent("keydown", { key: "z", code: "KeyZ", metaKey: true, shiftKey: true, bubbles: true })));
    await flush();
    expect(latest.tracks[0].items.map((i) => i.id)).toEqual(["B"]);
    expect(appConfirmMock).not.toHaveBeenCalled();
  });
});

describe("timeline audio editing (sc-23739)", () => {
  it("places library audio, auditions it, and persists trim, placement, gain, fades, and mute", () => {
    installAudioContext();
    const audio = { id: "asset_audio", type: "audio", displayName: "Recorded line", url: "/line.wav", file: { mimeType: "audio/wav", duration: 3 } };
    let latest;
    function Harness() {
      const [timeline, setTimeline] = React.useState(makeTimeline("tl_1", "Film"));
      latest = timeline;
      return <AppContext.Provider value={{ activeProject: { id: "proj_1" }, activeTimeline: timeline, mediaAssets: [audio], assets: [audio], timelines: [timeline], selectedTimelineId: timeline.id,
        setActiveTimeline: setTimeline, setSelectedTimelineId: vi.fn(), setPreviewAsset: vi.fn(), createTimeline: vi.fn(), extractTimelineFrame: vi.fn(), exportTimeline: vi.fn(), queueTimelineVideoJob: vi.fn(), saveTimeline: vi.fn(), isActiveTimelineDirty: () => false }}>
        <EditorScreen />
      </AppContext.Provider>;
    }
    const play = vi.spyOn(window.HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
    vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
    root = createRoot(container);
    act(() => root.render(<Harness />));
    act(() => container.querySelector(".ve-bin-item").click());
    expect(latest.tracks.find((track) => track.kind === "audio").items[0]).toMatchObject({ assetId: "asset_audio", type: "audio", timelineStart: 0 });
    act(() => container.querySelector(".ve-audio-clip").click());
    expect(container.querySelector(".ve-program audio")).not.toBeNull();
    act(() => container.querySelector(".ve-play").click());
    expect(play).toHaveBeenCalled();
    act(() => container.querySelector(".ve-play").click());
    const form = container.querySelector('form[aria-label="Edit selected audio"]');
    for (const [name, value] of [["timelineStart", "1.125"], ["sourceIn", "0.125"], ["sourceOut", "2.375"], ["volume", "0.7"], ["fadeInSeconds", "0.125"], ["fadeOutSeconds", "0.225"], ["trackGain", "0.8"]]) {
      const input = form.elements.namedItem(name);
      input.value = value;
    }
    form.elements.namedItem("muted").checked = true;
    for (const name of ["timelineStart", "sourceIn", "sourceOut", "fadeInSeconds", "fadeOutSeconds"]) {
      const input = form.elements.namedItem(name);
      expect(input.step).toBe("any");
      expect(input.validity.stepMismatch).toBe(false);
    }
    expect(form.checkValidity()).toBe(true);
    act(() => form.querySelector('button[type="submit"]').click());
    const track = latest.tracks.find((item) => item.kind === "audio");
    expect(track).toMatchObject({ gain: 0.8, muted: true });
    expect(track.items[0]).toMatchObject({ sourceIn: 0.125, sourceOut: 2.375, timelineStart: 1.125, timelineEnd: 3.375, volume: 0.7, fadeInSeconds: 0.125, fadeOutSeconds: 0.225 });
    const preview = container.querySelector(".ve-program audio");
    expect(preview.currentTime).toBeCloseTo(0.125);
    expect(container.querySelectorAll(".ve-timeline-audio audio")).toHaveLength(0);
    expect(preview.muted).toBe(true);
  });

  it("plays the entire mix on the first gesture and retains every layer across picture/audio reselection", async () => {
    const events = [];
    const contexts = installAudioContext(events);
    vi.spyOn(window.HTMLMediaElement.prototype, "play").mockImplementation(() => {
      events.push("play");
      return Promise.resolve();
    });
    vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(() => {});

    const audio = { id: "boosted", type: "audio", displayName: "Boosted", url: "/boosted.wav", file: { mimeType: "audio/wav", duration: 3 } };
    const video = { id: "video", type: "video", displayName: "Picture", url: "/picture.mp4", file: { mimeType: "video/mp4", duration: 2 } };
    const timeline = makeTimeline("tl_1", "Film");
    timeline.tracks[0].items.push({ id: "video_1", trackId: "track_main", assetId: "video", type: "video", displayName: "Picture", sourceIn: 0, sourceOut: 2, timelineStart: 0, timelineEnd: 2, speed: 1, volume: 1, generatedAudio: "include" });
    timeline.tracks.push({
      id: "track_audio",
      kind: "audio",
      gain: 4,
      muted: false,
      items: [
        { id: "audio_1", trackId: "track_audio", assetId: "boosted", type: "audio", displayName: "Boosted A", sourceIn: 0.25, sourceOut: 2.25, timelineStart: 0, timelineEnd: 2, speed: 1, volume: 2, fadeInSeconds: 0, fadeOutSeconds: 0 },
        { id: "audio_2", trackId: "track_audio", assetId: "boosted", type: "audio", displayName: "Boosted B", sourceIn: 0.5, sourceOut: 2.5, timelineStart: 0, timelineEnd: 2, speed: 1, volume: 2, fadeInSeconds: 0, fadeOutSeconds: 0 },
      ],
    });
    root = createRoot(container);
    act(() => root.render(<React.StrictMode><AppContext.Provider value={{ activeProject: { id: "proj_1" }, activeTimeline: timeline, mediaAssets: [audio, video], timelines: [timeline], selectedTimelineId: timeline.id, setActiveTimeline: vi.fn(), setSelectedTimelineId: vi.fn(), setPreviewAsset: vi.fn(), createTimeline: vi.fn(), extractTimelineFrame: vi.fn(), exportTimeline: vi.fn(), queueTimelineVideoJob: vi.fn(), saveTimeline: vi.fn(), isActiveTimelineDirty: () => false }}><EditorScreen /></AppContext.Provider></React.StrictMode>));
    act(() => container.querySelector(".ve-audio-clip").click());

    const layers = [...container.querySelectorAll(".ve-timeline-audio audio")];
    expect(layers).toHaveLength(3);
    expect(layers.map((layer) => layer.currentTime)).toEqual([0.25, 0.5, 0]);
    expect(contexts).toHaveLength(0);

    act(() => container.querySelector(".ve-play").click());
    expect(contexts).toHaveLength(1);
    expect(contexts[0].createMediaElementSource).toHaveBeenCalledTimes(3);
    expect(contexts[0].gains.map((gain) => gain.gain.value)).toEqual([8, 8, 1]);
    expect(events.slice(0, 4)).toEqual(["resume", "play", "play", "play"]);
    expect(layers.every((layer) => layer.volume === 1 && !layer.muted)).toBe(true);
    expect(container.querySelector(".ve-program video").muted).toBe(true);

    await flush();
    act(() => container.querySelector(".ve-play").click());
    act(() => container.querySelectorAll(".ve-audio-clip")[1].click());
    act(() => container.querySelector(".ve-play").click());
    expect(contexts[0].createMediaElementSource).toHaveBeenCalledTimes(3);
    expect([...container.querySelectorAll(".ve-timeline-audio audio")]).toEqual(layers);

    await flush();
    act(() => container.querySelector(".ve-clip").click());
    expect(contexts[0].close).not.toHaveBeenCalled();
    act(() => container.querySelector(".ve-play").click());
    expect(contexts[0].gains.map((gain) => gain.gain.value)).toEqual([8, 8, 1]);
    act(() => root.unmount());
    root = null;
    await flush();
    expect(contexts[0].close).toHaveBeenCalledTimes(1);
  });

  it("reports unavailable and measured-silent layers while retaining playable dialogue", () => {
    vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
    const timeline = makeTimeline("tl_1", "Film");
    timeline.tracks[0].items = [{ id: "picture", assetId: "picture", timelineStart: 0, timelineEnd: 4, generatedAudio: "include" }];
    timeline.tracks.push({ id: "dialogue", kind: "audio", items: ["voice", "missing"].map((id) => ({ id, assetId: id, timelineStart: 0, timelineEnd: 4 })) });
    render({ activeTimeline: timeline, mediaAssets: [
      { id: "picture", type: "video", file: { hasAudio: false }, url: "/silent.mp4" },
      { id: "voice", type: "audio", url: "/voice.wav" },
    ] });
    expect(container.querySelectorAll(".ve-timeline-audio audio")).toHaveLength(1);
    expect(container.textContent).toContain("Audio preview skips picture: no audio stream.");
    expect(container.textContent).toContain("Audio preview skips missing: asset unavailable.");
  });

  it("pins a completed audio preview to source out and replays it from source in", async () => {
    installAudioContext();
    const animationFrames = [];
    vi.spyOn(performance, "now").mockReturnValue(0);
    vi.stubGlobal("requestAnimationFrame", vi.fn((callback) => {
      animationFrames.push(callback);
      return animationFrames.length;
    }));
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
    const play = vi.spyOn(window.HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
    vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(() => {});

    const audio = { id: "quiet-tone", type: "audio", displayName: "Quiet tone", url: "/quiet-tone.wav", file: { mimeType: "audio/wav", duration: 2 } };
    const timeline = makeTimeline("tl_1", "Film");
    timeline.tracks.push({
      id: "track_audio",
      kind: "audio",
      gain: 1,
      muted: false,
      items: [{
        id: "audio_1",
        trackId: "track_audio",
        assetId: "quiet-tone",
        type: "audio",
        displayName: "Quiet tone",
        sourceIn: 0,
        sourceOut: 2,
        timelineStart: 0,
        timelineEnd: 2,
        speed: 1,
        volume: 1,
        fadeInSeconds: 0,
        fadeOutSeconds: 0,
      }],
    });
    root = createRoot(container);
    act(() => root.render(<AppContext.Provider value={{ activeProject: { id: "proj_1" }, activeTimeline: timeline, mediaAssets: [audio], timelines: [timeline], selectedTimelineId: timeline.id, setActiveTimeline: vi.fn(), setSelectedTimelineId: vi.fn(), setPreviewAsset: vi.fn(), createTimeline: vi.fn(), extractTimelineFrame: vi.fn(), exportTimeline: vi.fn(), queueTimelineVideoJob: vi.fn(), saveTimeline: vi.fn(), isActiveTimelineDirty: () => false }}><EditorScreen /></AppContext.Provider>));
    act(() => container.querySelector(".ve-audio-clip").click());

    const preview = container.querySelector(".ve-program audio");
    expect(preview.currentTime).toBeCloseTo(0);
    act(() => container.querySelector(".ve-play").click());
    expect(animationFrames).toHaveLength(1);

    await flush();
    // A delayed frame used to wrap the playhead to zero. The later ended event
    // then stopped playback and the audio sync effect rewound currentTime.
    act(() => animationFrames.shift()(2100));
    preview.currentTime = 2;
    act(() => preview.dispatchEvent(new Event("ended")));

    expect(container.querySelector(".ve-tc-now").textContent).toBe("00:02:00");
    expect(container.querySelector(".ve-play").title).toBe("Play");
    expect(preview.currentTime).toBeCloseTo(2);

    act(() => container.querySelector(".ve-play").click());
    expect(container.querySelector(".ve-tc-now").textContent).toBe("00:00:00");
    expect(container.querySelector(".ve-play").title).toBe("Pause");
    expect(preview.currentTime).toBeCloseTo(0);
    // One source starts per gesture; the selected monitor never duplicates it.
    expect(play).toHaveBeenCalledTimes(2);
  });

  it("accepts frame-derived fractional video trim endpoints through native form validity", () => {
    const video = { id: "v1", type: "video", displayName: "Shot", url: "/shot.mp4", file: { mimeType: "video/mp4", duration: 15 } };
    const timeline = makeTimeline("tl_1", "Film");
    timeline.tracks[0].items = [{ id: "shot", trackId: "track_main", assetId: "v1", type: "video", displayName: "Shot", sourceIn: 0, sourceOut: 14.375, timelineStart: 0, timelineEnd: 14.375, speed: 1, volume: 1 }];
    const setActiveTimeline = vi.fn();
    root = createRoot(container);
    act(() => root.render(<AppContext.Provider value={{ activeProject: { id: "proj_1" }, activeTimeline: timeline, mediaAssets: [video], timelines: [timeline], selectedTimelineId: timeline.id, setActiveTimeline, setSelectedTimelineId: vi.fn(), setPreviewAsset: vi.fn(), createTimeline: vi.fn(), extractTimelineFrame: vi.fn(), exportTimeline: vi.fn(), queueTimelineVideoJob: vi.fn(), saveTimeline: vi.fn(), isActiveTimelineDirty: () => false }}><EditorScreen /></AppContext.Provider>));
    act(() => container.querySelector(".ve-clip").click());
    const form = container.querySelector('form[aria-label="Edit selected clip"]');
    const sourceIn = form.elements.namedItem("sourceIn");
    const sourceOut = form.elements.namedItem("sourceOut");
    sourceOut.value = "0.05";
    expect(sourceOut.validity.rangeUnderflow).toBe(true);
    expect(form.checkValidity()).toBe(false);
    sourceIn.value = "0.125";
    sourceOut.value = "14.375";
    expect(sourceIn.step).toBe("any");
    expect(sourceOut.step).toBe("any");
    expect(sourceIn.validity.stepMismatch).toBe(false);
    expect(sourceOut.validity.stepMismatch).toBe(false);
    expect(form.checkValidity()).toBe(true);
    act(() => form.querySelector('button[type="submit"]').click());
    expect(setActiveTimeline).toHaveBeenCalledTimes(1);
    expect(setActiveTimeline.mock.calls[0][0].tracks[0].items[0]).toMatchObject({
      sourceIn: 0.125,
      sourceOut: 14.375,
      timelineEnd: 14.25,
    });
  });

  it("keeps linked dialogue placed and explicitly requests adjustment after a shorter picture trim", () => {
    const video = { id: "v1", type: "video", displayName: "Shot", url: "/shot.mp4", file: { mimeType: "video/mp4", duration: 4 } };
    const audio = { id: "a1", type: "audio", displayName: "Line", url: "/line.wav", file: { mimeType: "audio/wav", duration: 3 } };
    const timeline = makeTimeline("tl_1", "Film");
    timeline.tracks[0].items = [{ id: "shot", trackId: "track_main", assetId: "v1", type: "video", displayName: "Shot", sourceIn: 0, sourceOut: 4, timelineStart: 0, timelineEnd: 4, speed: 1, volume: 1, filmHarness: { runId: "run_1", shotId: "SH010" } }];
    timeline.tracks.push({ id: "track_dialogue", kind: "audio", role: "dialogue", gain: 1, muted: false, items: [{ id: "line", trackId: "track_dialogue", assetId: "a1", type: "audio", displayName: "Line", sourceIn: 0, sourceOut: 3, timelineStart: 1, timelineEnd: 4, speed: 1, volume: 1, filmHarness: { runId: "run_1", shotId: "SH010" } }] });
    root = createRoot(container);
    act(() => root.render(<AppContext.Provider value={{ activeProject: { id: "proj_1" }, activeTimeline: timeline, mediaAssets: [video, audio], timelines: [timeline], selectedTimelineId: timeline.id, setActiveTimeline: vi.fn(), setSelectedTimelineId: vi.fn(), setPreviewAsset: vi.fn(), createTimeline: vi.fn(), extractTimelineFrame: vi.fn(), exportTimeline: vi.fn(), queueTimelineVideoJob: vi.fn(), saveTimeline: vi.fn(), isActiveTimelineDirty: () => false }}><EditorScreen /></AppContext.Provider>));
    act(() => container.querySelector(".ve-clip").click());
    const form = container.querySelector('form[aria-label="Edit selected clip"]');
    form.elements.namedItem("sourceOut").value = "2";
    act(() => form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
    expect(container.textContent).toContain("Adjust the linked audio ending past the new picture cut");
  });

  it("badges a film clip with its shot, hands its review to Film mode, and returns to the timeline", () => {
    const video = { id: "v1", type: "video", displayName: "Shot", url: "/shot.mp4", file: { mimeType: "video/mp4", duration: 4 } };
    const timeline = makeTimeline("tl_1", "Film");
    timeline.tracks[0].items = [{ id: "shot", trackId: "track_main", assetId: "v1", type: "video", displayName: "Shot", sourceIn: 0, sourceOut: 4, timelineStart: 0, timelineEnd: 4, speed: 1, volume: 1, filmHarness: { runId: "run_1", shotId: "SH010" } }];
    root = createRoot(container);
    act(() => root.render(<AppContext.Provider value={{ activeProject: { id: "proj_1" }, activeTimeline: timeline, mediaAssets: [video], timelines: [timeline], selectedTimelineId: timeline.id, setActiveTimeline: vi.fn(), setSelectedTimelineId: vi.fn(), setPreviewAsset: vi.fn(), createTimeline: vi.fn(), extractTimelineFrame: vi.fn(), exportTimeline: vi.fn(), queueTimelineVideoJob: vi.fn(), saveTimeline: vi.fn(), isActiveTimelineDirty: () => false }}><EditorScreen /></AppContext.Provider>));

    expect(container.querySelector(".ve-clip .ve-clip-shot").textContent).toBe("SH010");
    act(() => container.querySelector(".ve-clip").click());
    expect(container.querySelector(".ve-rail-eyebrow").textContent).toContain("FILM SHOT · SH010");
    const film = container.querySelector('section[aria-label="Film workspace"]');
    expect(film.hidden).toBe(true);

    act(() => [...container.querySelectorAll(".ve-ctx-btn")].find((button) => button.textContent === "Review this shot").click());
    expect(film.hidden).toBe(false);
    expect(container.querySelector(".ve-timeline")).toBeNull();

    act(() => [...container.querySelectorAll(".ve-mode-switch button")].find((button) => button.textContent === "Timeline").click());
    expect(film.hidden).toBe(true);
    expect(container.querySelector(".ve-clip")).not.toBeNull();
  });
});
