// sc-11967 (S8): re-selecting a timeline from the editor dropdown reloads the server copy
// and replaces the in-memory working copy. When the active timeline has unsaved edits that
// silently dropped them. These tests assert the dropdown now guards the switch with the
// app's discard-edits confirm when dirty, and behaves exactly as before when clean.
import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppContext } from "../context/AppContext.js";
import { EditorScreen } from "./EditorScreen.jsx";

function makeTimeline(id, name) {
  return { id, name, aspectRatio: "16:9", fps: 30, width: 1280, height: 720, tracks: [{ id: "track_main", name: "Main", items: [] }] };
}

let container;
let root;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
});

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

// The timeline dropdown is the first <select> in the editor header. Firing a native change
// event with a new value routes through the guarded onChange.
function selectTimeline(nextId) {
  const select = container.querySelector(".editor-header select");
  act(() => {
    select.value = nextId;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  return select;
}

describe("EditorScreen timeline re-select guard (sc-11967)", () => {
  it("switches without prompting when the timeline is clean (no regression)", () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const { setSelectedTimelineId } = render({ isActiveTimelineDirty: () => false });

    selectTimeline("tl_2");

    expect(confirmSpy).not.toHaveBeenCalled();
    expect(setSelectedTimelineId).toHaveBeenCalledWith("tl_2");
  });

  it("does NOT switch (keeps the current timeline) when dirty and the user cancels the confirm", () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    const { setSelectedTimelineId } = render({ isActiveTimelineDirty: () => true });

    const select = selectTimeline("tl_2");

    expect(confirmSpy).toHaveBeenCalledTimes(1);
    expect(setSelectedTimelineId).not.toHaveBeenCalled();
    // Controlled <select> snaps back to the still-active timeline.
    expect(select.value).toBe("tl_1");
  });

  it("switches when dirty and the user accepts the confirm", () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const { setSelectedTimelineId } = render({ isActiveTimelineDirty: () => true });

    selectTimeline("tl_2");

    expect(confirmSpy).toHaveBeenCalledTimes(1);
    expect(setSelectedTimelineId).toHaveBeenCalledWith("tl_2");
  });
});
