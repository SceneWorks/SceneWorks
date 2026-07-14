// sc-11967 (S8): the active video timeline is lifted to App-level useTimelines and
// survives soft navigation in memory. Two paths used to silently clobber the user's
// unsaved in-memory structural edits:
//   1. applyCompletedTimelineGeneration (SSE "generation ready") fetched the backend
//      copy and setActiveTimeline(serverCopy), discarding unsaved edits.
//   2. loadTimeline (dropdown re-select) overwrote activeTimeline with no warning.
// These tests assert the SSE-apply path is now non-destructive (merges the generation
// onto the working copy instead of clobbering) and that the dirty signal used by the
// re-select guard resets correctly on save/load (the post-save-baseline pitfall).
import React, { useState } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useTimelines } from "./useTimelines.js";

// Programmable apiFetch: a per-test router keyed by method + path shape. Each call is
// also recorded so a test can assert the server copy was persisted (generation never
// dropped) without the test caring about response identity.
let apiRouter;
const apiCalls = [];
vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async (path, _token, options = {}) => {
      const method = options.method ?? "GET";
      const body = options.body ? JSON.parse(options.body) : undefined;
      apiCalls.push({ path, method, body });
      return apiRouter({ path, method, body });
    }),
  };
});

async function settle() {
  await act(async () => {
    for (let i = 0; i < 6; i += 1) {
      await Promise.resolve();
    }
  });
}

function makeTimeline(overrides = {}) {
  return {
    id: "tl_1",
    name: "Main",
    aspectRatio: "16:9",
    fps: 30,
    width: 1280,
    height: 720,
    tracks: [{ id: "track_main", name: "Main", items: [] }],
    ...overrides,
  };
}

// A completed extend-generation job for tl_1's main track. applyTimelineGenerationResult
// appends a fresh video item to context.trackId.
function makeExtendJob() {
  return {
    id: "job_gen",
    projectId: "proj_1",
    result: { assetIds: ["asset_gen"], assets: [{ displayName: "Gen clip", createdAt: "2026-07-14T00:00:00Z" }] },
    payload: {
      duration: 4,
      advanced: {
        timelineAction: "extend",
        timelineContext: { timelineId: "tl_1", trackId: "track_main", timelineStart: 4 },
      },
    },
  };
}

let container;
let root;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  apiCalls.length = 0;
  container = document.createElement("div");
  document.body.appendChild(container);
});

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
});

function mount() {
  const activeProject = { id: "proj_1", name: "Proj" };
  const activeProjectRef = { current: activeProject };
  let latest = null;
  function Harness() {
    const [, setN] = useState(0);
    latest = {
      api: useTimelines({
        token: "tok",
        activeProject,
        activeProjectRef,
        setError: () => {},
        requestedGpu: "auto",
        setActiveView: () => {},
        createVideoJob: async () => null,
      }),
      rerender: () => setN((n) => n + 1),
    };
    return null;
  }
  root = createRoot(container);
  act(() => root.render(<Harness />));
  return () => latest;
}

// Drives the hook into the "tl_1 loaded + selected" state, mirroring App's load-on-selection
// effect. The GET during load establishes the clean baseline.
async function loadSelected(get, timeline) {
  apiRouter = ({ method }) => {
    if (method === "GET") return timeline;
    return timeline;
  };
  act(() => {
    get().api.setTimelines([{ id: timeline.id, name: timeline.name }]);
    get().api.setTimelinesProjectId("proj_1");
    get().api.setSelectedTimelineId(timeline.id);
  });
  await settle();
}

describe("useTimelines dirty tracking (sc-11967)", () => {
  it("reports NOT dirty right after a clean load", async () => {
    const get = mount();
    const tl = makeTimeline();
    await loadSelected(get, tl);

    expect(get().api.activeTimeline?.id).toBe("tl_1");
    expect(get().api.isActiveTimelineDirty()).toBe(false);
  });

  it("reports dirty after an in-memory structural edit", async () => {
    const get = mount();
    await loadSelected(get, makeTimeline());

    act(() => {
      const current = get().api.activeTimeline;
      get().api.setActiveTimeline({
        ...current,
        tracks: current.tracks.map((t) => ({
          ...t,
          items: [...t.items, { id: "item_user_edit", assetId: "a", type: "video", timelineStart: 0, timelineEnd: 4 }],
        })),
      });
    });
    await settle();

    expect(get().api.isActiveTimelineDirty()).toBe(true);
  });

  it("resets to NOT dirty after a successful save (post-save baseline pitfall)", async () => {
    const get = mount();
    await loadSelected(get, makeTimeline());

    // Dirty the working copy.
    let edited;
    act(() => {
      const current = get().api.activeTimeline;
      edited = {
        ...current,
        tracks: current.tracks.map((t) => ({ ...t, items: [...t.items, { id: "item_user_edit" }] })),
      };
      get().api.setActiveTimeline(edited);
    });
    await settle();
    expect(get().api.isActiveTimelineDirty()).toBe(true);

    // Save echoes back the persisted timeline (server may reorder/strip fields).
    apiRouter = ({ method, path, body }) => {
      if (method === "PUT") return { ...body.timeline };
      if (path.endsWith("/timelines")) return [{ id: "tl_1", name: "Main" }];
      return body?.timeline ?? edited;
    };
    await act(async () => {
      await get().api.saveTimeline(get().api.activeTimeline);
    });
    await settle();

    expect(get().api.isActiveTimelineDirty()).toBe(false);
  });
});

describe("useTimelines SSE apply is non-destructive when dirty (sc-11967)", () => {
  it("MERGES the generation onto the working copy without clobbering unsaved edits", async () => {
    const get = mount();
    await loadSelected(get, makeTimeline());

    // User makes an unsaved structural edit (adds their own clip).
    act(() => {
      const current = get().api.activeTimeline;
      get().api.setActiveTimeline({
        ...current,
        tracks: current.tracks.map((t) => ({
          ...t,
          items: [...t.items, { id: "item_user_edit", assetId: "user_asset", type: "video", trackId: "track_main", timelineStart: 0, timelineEnd: 4 }],
        })),
      });
    });
    await settle();
    expect(get().api.isActiveTimelineDirty()).toBe(true);

    // A timeline generation job completes. The backend copy (GET single) has NO user edit.
    const serverCopy = makeTimeline();
    apiRouter = ({ method, path, body }) => {
      if (method === "PUT") return { ...body.timeline };
      if (path.endsWith("/timelines")) return [{ id: "tl_1", name: "Main" }]; // LIST
      if (method === "GET") return serverCopy; // single timeline
      return serverCopy;
    };

    await act(async () => {
      get().api.enqueueTimelineGenerationApply(makeExtendJob());
    });
    await settle();

    const items = get().api.activeTimeline.tracks.flatMap((t) => t.items);
    // The user's unsaved edit MUST survive (this is the clobber the story fixes).
    expect(items.some((it) => it.id === "item_user_edit")).toBe(true);
    // AND the generation result MUST be present (never silently dropped).
    expect(items.some((it) => it.assetId === "asset_gen")).toBe(true);
    // The generation was also durably persisted to the server copy.
    expect(apiCalls.some((c) => c.method === "PUT")).toBe(true);
  });
});

describe("useTimelines SSE apply is unchanged when clean (sc-11967 no-regression)", () => {
  it("adopts the saved server copy and stays clean (no spurious dirty)", async () => {
    const get = mount();
    await loadSelected(get, makeTimeline());
    expect(get().api.isActiveTimelineDirty()).toBe(false);

    const serverCopy = makeTimeline();
    apiRouter = ({ method, path, body }) => {
      if (method === "PUT") return { ...body.timeline };
      if (path.endsWith("/timelines")) return [{ id: "tl_1", name: "Main" }]; // LIST
      if (method === "GET") return serverCopy; // single timeline
      return serverCopy;
    };

    await act(async () => {
      get().api.enqueueTimelineGenerationApply(makeExtendJob());
    });
    await settle();

    const items = get().api.activeTimeline.tracks.flatMap((t) => t.items);
    expect(items.some((it) => it.assetId === "asset_gen")).toBe(true);
    // Clean timeline that received a generation is still clean afterwards — the server
    // copy it adopted is the persisted one, so a later re-select must not prompt.
    expect(get().api.isActiveTimelineDirty()).toBe(false);
  });
});
