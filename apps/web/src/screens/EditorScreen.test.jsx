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
