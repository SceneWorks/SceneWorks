// sc-8858 (F-056): SSE-triggered, project-scoped refreshers must drop a stale
// response — one whose fetch was fired for project A but resolves AFTER the user
// switched the active project to B. Committing it would clobber B's state with A's
// data. refreshTimelines already guards exactly this
// (`activeProjectRef.current.id !== projectId → return`, useTimelines.js); these
// tests assert the same guard now holds for the other project-scoped refreshers:
// characters, loras (useModelsAndLoras), and person-tracks. The assets refresh
// (App.jsx) shares the identical guard mechanism — it reads the same
// `activeProjectRef` — and is exercised by the same shape here at the hook layer.
import React, { useState } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useCharacters } from "./useCharacters.js";
import { useModelsAndLoras } from "./useModelsAndLoras.js";
import { usePersonTracks } from "./usePersonTracks.js";
import { useTimelines } from "./useTimelines.js";

// Controllable fetch: each apiFetch call parks its resolver so a test can resolve
// project A's response AFTER the active project has switched to B.
const pendingRequests = [];
vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(
      () =>
        new Promise((resolve, reject) => {
          pendingRequests.push({ resolve, reject });
        }),
    ),
  };
});

async function settle() {
  await act(async () => {
    for (let i = 0; i < 4; i += 1) {
      await Promise.resolve();
    }
  });
}

// Each hook exposes a project-scoped `refresh*` plus a `set*`/state pair. activeProjectRef
// mirrors the *currently active* project (mutated by the test to simulate a switch),
// independent of the projectId the refresh was fired for.
const CASES = [
  { name: "useCharacters.refreshCharacters", hook: useCharacters, refreshKey: "refreshCharacters", stateKey: "characters" },
  { name: "useModelsAndLoras.refreshLoras", hook: useModelsAndLoras, refreshKey: "refreshLoras", stateKey: "loras" },
  { name: "usePersonTracks.refreshPersonTracks", hook: usePersonTracks, refreshKey: "refreshPersonTracks", stateKey: "personTracks" },
  { name: "useTimelines.refreshTimelines", hook: useTimelines, refreshKey: "refreshTimelines", stateKey: "timelines" },
];

describe.each(CASES)("$name stale-project guard (sc-8858)", ({ hook, refreshKey, stateKey }) => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    pendingRequests.length = 0;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
  });

  // Renders the hook, exposing its live api and a way to force a re-render so committed
  // state re-reads. activeProjectRef is a plain mutable ref the test flips mid-flight.
  function mount(activeProjectRef, setError = () => {}) {
    let latest = null;
    function Harness() {
      const [, setN] = useState(0);
      latest = {
        api: hook({
          token: "tok",
          activeProject: activeProjectRef.current,
          activeProjectRef,
          setError,
          setJobs: () => {},
          requestedGpu: "auto",
          setActiveView: () => {},
          refreshData: async () => {},
          refreshDataWithLoraOverlay: async () => {},
        }),
        rerender: () => setN((n) => n + 1),
      };
      return null;
    }
    root = createRoot(container);
    act(() => root.render(<Harness />));
    return () => latest;
  }

  it("does NOT commit a response for project A after the active project switched to B", async () => {
    const activeProjectRef = { current: { id: "A", name: "A" } };
    const get = mount(activeProjectRef);

    // Fire the refresh for project A (as an SSE job.updated for A would).
    let pending;
    act(() => {
      pending = get().api[refreshKey]("A");
    });

    // User switches to project B while A's fetch is still in flight.
    activeProjectRef.current = { id: "B", name: "B" };

    // A's response resolves — it must be dropped, not committed.
    await act(async () => {
      pendingRequests[0].resolve([{ id: "from-A" }]);
      await pending;
    });
    await settle();

    expect(get().api[stateKey]).toEqual([]);
  });

  it("does NOT commit or surface an error after project A is closed", async () => {
    const activeProjectRef = { current: { id: "A", name: "A" } };
    const setError = vi.fn();
    const get = mount(activeProjectRef, setError);

    let pending;
    act(() => {
      pending = get().api[refreshKey]("A");
    });
    activeProjectRef.current = null;

    await act(async () => {
      pendingRequests[0].reject(new Error("late A failure"));
      await pending;
    });
    await settle();

    expect(get().api[stateKey]).toEqual([]);
    expect(setError).not.toHaveBeenCalledWith("late A failure");
  });

  it("rejects an already-stale explicit refresh before making a request", async () => {
    const activeProjectRef = { current: { id: "B", name: "B" } };
    const get = mount(activeProjectRef);

    await act(async () => {
      const result = await get().api[refreshKey]("A");
      expect(result).toMatchObject({ ok: false, reason: "stale" });
    });

    expect(pendingRequests).toHaveLength(0);
  });

  it("DOES commit a response for the still-active project", async () => {
    const activeProjectRef = { current: { id: "A", name: "A" } };
    const get = mount(activeProjectRef);

    let pending;
    act(() => {
      pending = get().api[refreshKey]("A");
    });

    // Active project unchanged — the response is current and must land.
    await act(async () => {
      pendingRequests[0].resolve([{ id: "from-A" }]);
      await pending;
    });
    await settle();

    expect(get().api[stateKey]).toEqual([{ id: "from-A" }]);
  });
});

// refreshLoras also serves a *global* (projectId undefined) fetch — the LoRA catalog
// with no project scope. That path is not project-scoped, so it must still commit even
// when activeProjectRef points at some project.
describe("useModelsAndLoras.refreshLoras global (no project) still commits (sc-8858)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    pendingRequests.length = 0;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
  });

  it("commits a global (no-project) refresh even when a project later becomes active", async () => {
    // No active project → refreshLoras() resolves projectId to undefined, i.e. the
    // unscoped global catalog fetch. The guard must NOT drop this even if a project
    // becomes active mid-flight, because the fetch was never project-scoped.
    const activeProjectRef = { current: null };
    let latest = null;
    function Harness() {
      const [, setN] = useState(0);
      latest = {
        api: useModelsAndLoras({
          token: "tok",
          activeProject: activeProjectRef.current,
          activeProjectRef,
          setError: () => {},
          setJobs: () => {},
          setActiveView: () => {},
          refreshData: async () => {},
          refreshDataWithLoraOverlay: async () => {},
        }),
        rerender: () => setN((n) => n + 1),
      };
      return null;
    }
    root = createRoot(container);
    act(() => root.render(<Harness />));

    let pending;
    act(() => {
      // No project active → global loras fetch (projectId undefined).
      pending = latest.api.refreshLoras();
    });

    // A project becomes active while the global fetch is in flight — must still commit.
    activeProjectRef.current = { id: "B", name: "B" };

    await act(async () => {
      pendingRequests[0].resolve([{ id: "global-lora" }]);
      await pending;
    });
    await settle();

    expect(latest.api.loras).toEqual([{ id: "global-lora" }]);
  });
});
