// sc-9750 (F-052 follow-up): focused unit coverage for the two hooks extracted from
// App.jsx — useAccessGate (the remote-access gate + media-ticket mint) and useJobEvents
// (the live job/worker/queue SSE stream). The App.*.test.jsx suite already exercises
// both end-to-end through <App />; these tests pin the hook-level contracts directly so
// a regression in the extracted logic is caught at the unit boundary too.
import React, { useState } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useAccessGate } from "./useAccessGate.js";
import { useJobEvents } from "./useJobEvents.js";
import { useTimelines } from "./useTimelines.js";
import { apiFetch } from "../api.js";
import { MAX_TERMINAL_JOBS } from "../sorters.js";

// Controllable apiFetch: each call resolves with whatever the per-path map returns (or
// rejects when the value is an Error), so a test can drive the /access probe, the
// media-ticket mint, and /auth/verify independently. setMediaTicket is a spy.
const apiResponders = new Map();
const setMediaTicketSpy = vi.fn();
vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn((path) => {
      const responder = apiResponders.get(path);
      const value = typeof responder === "function" ? responder() : responder;
      if (value instanceof Error) {
        return Promise.reject(value);
      }
      return Promise.resolve(value ?? {});
    }),
    eventUrl: (path, ticket) => {
      const url = new URL(path, "http://localhost");
      if (ticket) {
        url.searchParams.set("ticket", ticket);
      }
      return `${url.pathname}${url.search}`;
    },
    setMediaTicket: (...args) => setMediaTicketSpy(...args),
  };
});

// Desktop-shell detection defaults to false (remote-browser mode) so the gate exercises
// the auth path; individual tests can leave it false.
vi.mock("../runtime.js", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, isDesktop: false };
});

// A minimal FakeEventSource capturing listeners so tests can dispatch SSE events, plus
// close() bookkeeping so the effect-cleanup assertion has something to check.
class FakeEventSource {
  static instances = [];
  constructor(url) {
    this.url = url;
    this.listeners = {};
    this.closed = false;
    FakeEventSource.instances.push(this);
  }
  addEventListener(event, handler) {
    this.listeners[event] = handler;
  }
  close() {
    this.closed = true;
  }
}

async function settle() {
  await act(async () => {
    for (let i = 0; i < 6; i += 1) {
      await Promise.resolve();
    }
  });
}

describe("useAccessGate (sc-9750)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    apiResponders.clear();
    setMediaTicketSpy.mockClear();
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
  });

  function mount() {
    let latest = null;
    const notices = [];
    function Harness() {
      const [, setN] = useState(0);
      latest = {
        api: useAccessGate({
          setError: () => {},
          pushNotice: (kind, message) => notices.push({ kind, message }),
          dismissNoticeKind: (kind) => {
            for (let i = notices.length - 1; i >= 0; i -= 1) {
              if (notices[i].kind === kind) notices.splice(i, 1);
            }
          },
        }),
        rerender: () => setN((n) => n + 1),
      };
      return null;
    }
    root = createRoot(container);
    act(() => root.render(<Harness />));
    return { get: () => latest, notices };
  }

  it("resolves authenticated+ready with no auth required and mints no media ticket", async () => {
    apiResponders.set("/api/v1/access", { authRequired: false });
    const { get } = mount();
    await settle();

    expect(get().api.access).toEqual({ authRequired: false });
    expect(get().api.authenticated).toBe(true);
    // Auth off → media is immediately ready and the stored ticket is cleared (sc-8810).
    expect(get().api.ready).toBe(true);
    expect(setMediaTicketSpy).toHaveBeenCalledWith("");
  });

  it("holds ready until the media-ticket mint settles when auth is required", async () => {
    window.localStorage.setItem("sceneworks-token", "remote-token");
    apiResponders.set("/api/v1/access", { authRequired: true });
    apiResponders.set("/api/v1/files/ticket", { ticket: "media-1", expiresInSeconds: 300 });
    const { get } = mount();
    await settle();

    expect(get().api.authenticated).toBe(true);
    // Mint succeeded → ready flips true and the ticket is stored.
    expect(get().api.ready).toBe(true);
    expect(setMediaTicketSpy).toHaveBeenCalledWith("media-1");
  });

  it("still reports ready (degraded) and pushes a notice when the mint fails", async () => {
    window.localStorage.setItem("sceneworks-token", "remote-token");
    apiResponders.set("/api/v1/access", { authRequired: true });
    apiResponders.set("/api/v1/files/ticket", () => new Error("mint exploded"));
    const { get, notices } = mount();
    await settle();

    // sc-9063: a failed mint settles the gate (ready:true) so data still loads, and a
    // media-ticket notice explains the degraded media.
    expect(get().api.ready).toBe(true);
    expect(notices.some((n) => n.kind === "media-ticket")).toBe(true);
  });

  it("saveToken verifies the draft before promoting it to the live token", async () => {
    apiResponders.set("/api/v1/access", { authRequired: true });
    apiResponders.set("/api/v1/auth/verify", { ok: true });
    apiResponders.set("/api/v1/files/ticket", { ticket: "media-1", expiresInSeconds: 300 });
    const { get } = mount();
    await settle();

    // Not authenticated yet (no token), gate is up.
    expect(get().api.token).toBe("");
    expect(get().api.authenticated).toBe(false);

    act(() => get().api.setPasswordDraft("  secret  "));
    await settle();
    await act(async () => {
      await get().api.saveToken({ preventDefault: () => {} });
    });
    await settle();

    // Verified draft is trimmed, promoted to the live token, and persisted.
    expect(get().api.token).toBe("secret");
    expect(get().api.authenticated).toBe(true);
    expect(window.localStorage.getItem("sceneworks-token")).toBe("secret");
  });

  it("saveToken keeps the gate up with an inline error on a wrong password", async () => {
    apiResponders.set("/api/v1/access", { authRequired: true });
    apiResponders.set("/api/v1/auth/verify", { ok: false });
    const { get } = mount();
    await settle();

    act(() => get().api.setPasswordDraft("wrong"));
    await settle();
    await act(async () => {
      await get().api.saveToken({ preventDefault: () => {} });
    });
    await settle();

    expect(get().api.token).toBe("");
    expect(get().api.authError).toBe("Incorrect password. Try again.");
    expect(window.localStorage.getItem("sceneworks-token")).toBeNull();
  });

  it("lockRemote clears the stored token and re-shows the gate", async () => {
    window.localStorage.setItem("sceneworks-token", "remote-token");
    apiResponders.set("/api/v1/access", { authRequired: true });
    apiResponders.set("/api/v1/files/ticket", { ticket: "media-1", expiresInSeconds: 300 });
    const { get } = mount();
    await settle();
    expect(get().api.token).toBe("remote-token");

    act(() => get().api.lockRemote());
    await settle();

    expect(get().api.token).toBe("");
    expect(get().api.authenticated).toBe(false);
    expect(window.localStorage.getItem("sceneworks-token")).toBeNull();
  });
});

describe("useJobEvents (sc-9750)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    apiResponders.clear();
    apiFetch.mockClear();
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
  });

  // A superset of the hook's props with stable no-op stand-ins; a test overrides only
  // what it asserts on. Refs mirror how App feeds live handles into the handlers.
  function baseProps(overrides = {}) {
    return {
      access: { authRequired: false },
      ready: true,
      token: "",
      jobsRef: { current: [] },
      setJobs: () => {},
      setWorkers: () => {},
      setQueueSummary: () => {},
      setLatestGenerationSetId: () => {},
      setError: () => {},
      pushNotice: () => {},
      dismissNoticeKind: () => {},
      generatedAssetRefreshesRef: { current: new Map() },
      refreshAssetsRef: { current: () => {} },
      refreshDataRef: { current: () => {} },
      refreshDataWithLoraOverlayRef: { current: () => {} },
      refreshPersonTracksRef: { current: () => {} },
      activeProjectRef: { current: null },
      enqueueTimelineGenerationApply: () => {},
      hasVisibleLocalFailure: () => false,
      ...overrides,
    };
  }

  function mount(initialProps) {
    let setProps = () => {};
    function Harness() {
      const [props, update] = useState(initialProps);
      setProps = update;
      useJobEvents(props);
      return null;
    }
    root = createRoot(container);
    act(() => root.render(<Harness />));
    return { setProps: (next) => act(() => setProps(next)) };
  }

  it("does not open an EventSource until ready is true", async () => {
    const { setProps } = mount(baseProps({ ready: false }));
    await settle();
    expect(FakeEventSource.instances.length).toBe(0);

    setProps(baseProps({ ready: true }));
    await settle();
    expect(FakeEventSource.instances.length).toBe(1);
  });

  it("routes job.updated through setJobs and refreshes generated assets", async () => {
    const jobs = [];
    const refreshedProjects = [];
    const props = baseProps({
      setJobs: (updater) => {
        jobs.length = 0;
        jobs.push(...updater([]));
      },
      refreshAssetsRef: { current: (projectId) => refreshedProjects.push(projectId) },
    });
    mount(props);
    await settle();

    const source = FakeEventSource.instances[0];
    expect(typeof source.listeners["job.updated"]).toBe("function");

    act(() => {
      source.listeners["job.updated"]({
        data: JSON.stringify({
          id: "job-1",
          projectId: "proj-1",
          status: "running",
          result: { generationSetId: "gs-1", assetIds: ["a1"] },
        }),
      });
    });

    expect(jobs.map((job) => job.id)).toContain("job-1");
    // A generation-set result with a new asset count triggers a project asset refresh.
    expect(refreshedProjects).toContain("proj-1");
  });

  it("reconciles stale active job state from jobs.snapshot on reconnect", async () => {
    let jobs = [{
      id: "job-1",
      status: "running",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:01:00Z",
    }];
    const jobsRef = { current: jobs };
    const props = baseProps({
      jobsRef,
      setJobs: (updater) => {
        jobs = updater(jobs);
        jobsRef.current = jobs;
      },
    });
    mount(props);
    await settle();

    const source = FakeEventSource.instances[0];
    expect(typeof source.listeners["jobs.snapshot"]).toBe("function");
    act(() => {
      source.listeners["jobs.snapshot"]({
        data: JSON.stringify([{
          id: "job-1",
          status: "completed",
          createdAt: "2026-01-01T00:00:00Z",
          updatedAt: "2026-01-01T00:02:00Z",
        }]),
      });
    });

    expect(jobs).toHaveLength(1);
    expect(jobs[0].status).toBe("completed");
  });

  it("rejects a globally newer delayed job event with an older per-job revision when updatedAt ties", async () => {
    let jobs = [{
      id: "job-1",
      status: "running",
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:01:00Z",
    }];
    const jobsRef = { current: jobs };
    mount(baseProps({
      jobsRef,
      setJobs: (updater) => {
        jobs = updater(jobs);
        jobsRef.current = jobs;
      },
    }));
    await settle();
    const source = FakeEventSource.instances[0];

    act(() => {
      source.listeners["jobs.snapshot"]({
        lastEventId: "10",
        data: JSON.stringify({
          revision: 10,
          clearedJobIds: [],
          jobs: [{
            ...jobs[0],
            status: "completed",
            revision: 2,
            updatedAt: "2026-01-01T00:01:00Z",
          }],
        }),
      });
      source.listeners["job.updated"]({
        lastEventId: "11",
        data: JSON.stringify({
          ...jobs[0],
          status: "running",
          revision: 1,
          updatedAt: "2026-01-01T00:01:00Z",
        }),
      });
    });

    expect(jobs[0].status).toBe("completed");
  });

  it("applies live and reconnect clear tombstones without dropping capped history", async () => {
    let jobs = [
      { id: "cleared-live", status: "completed", createdAt: "2026-01-03T00:00:00Z" },
      { id: "cleared-offline", status: "failed", createdAt: "2026-01-02T00:00:00Z" },
      { id: "retained-history", status: "completed", createdAt: "2026-01-01T00:00:00Z" },
    ];
    const jobsRef = { current: jobs };
    mount(baseProps({
      jobsRef,
      setJobs: (updater) => {
        jobs = updater(jobs);
        jobsRef.current = jobs;
      },
    }));
    await settle();
    const source = FakeEventSource.instances[0];

    act(() => {
      source.listeners["jobs.cleared"]({
        lastEventId: "11",
        data: JSON.stringify({ ids: ["cleared-live"] }),
      });
      source.listeners["jobs.snapshot"]({
        lastEventId: "12",
        data: JSON.stringify({
          revision: 12,
          jobs: [],
          clearedJobIds: ["cleared-live", "cleared-offline"],
        }),
      });
    });

    expect(jobs.map((job) => job.id)).toEqual(["retained-history"]);
  });

  it("bounds live clear tombstones and safely replaces them at the snapshot barrier", async () => {
    const activeJobs = [
      {
        id: "active-1",
        status: "running",
        createdAt: "2026-01-02T00:00:00Z",
        updatedAt: "2026-01-02T00:01:00Z",
      },
      {
        id: "active-2",
        status: "running",
        createdAt: "2026-01-01T00:00:00Z",
        updatedAt: "2026-01-01T00:01:00Z",
      },
    ];
    let jobs = activeJobs;
    const jobsRef = { current: jobs };
    mount(baseProps({
      jobsRef,
      setJobs: (updater) => {
        jobs = updater(jobs);
        jobsRef.current = jobs;
      },
    }));
    await settle();
    const source = FakeEventSource.instances[0];
    const clearedJobs = Array.from(
      { length: activeJobs.length + MAX_TERMINAL_JOBS + 1 },
      (_, index) => ({
        id: `cleared-${index}`,
        type: "image_generate",
        status: "failed",
        revision: 1,
        createdAt: new Date(Date.UTC(2026, 0, 1, 0, 0, index)).toISOString(),
        updatedAt: new Date(Date.UTC(2026, 0, 1, 1, 0, index)).toISOString(),
        error: `failure-${index}`,
      }),
    );

    act(() => {
      clearedJobs.forEach((job, index) => {
        source.listeners["job.updated"]({
          lastEventId: String(index * 2 + 1),
          data: JSON.stringify(job),
        });
        source.listeners["jobs.cleared"]({
          lastEventId: String(index * 2 + 2),
          data: JSON.stringify({ ids: [job.id] }),
        });
      });
    });
    expect(jobs).toEqual(activeJobs);

    // The newest relevant clear remains protected after more than 200 clears.
    act(() => {
      source.listeners["job.updated"]({
        lastEventId: "500",
        data: JSON.stringify(clearedJobs.at(-1)),
      });
    });
    expect(jobs).toEqual(activeJobs);

    // The oldest no-longer-known tombstone was evicted at the terminal-history
    // bound (complete active set + 200 terminal rows), so a genuinely newer
    // publication for that ID can become visible.
    act(() => {
      source.listeners["job.updated"]({
        lastEventId: "501",
        data: JSON.stringify(clearedJobs[0]),
      });
    });
    expect(jobs.map((job) => job.id)).toContain(clearedJobs[0].id);
    expect(jobs).toHaveLength(activeJobs.length + 1);

    act(() => {
      source.listeners["jobs.cleared"]({
        lastEventId: "502",
        data: JSON.stringify({ ids: [clearedJobs[0].id] }),
      });
      source.listeners["jobs.snapshot"]({
        lastEventId: "600",
        data: JSON.stringify({
          revision: 600,
          jobs: activeJobs,
          clearedJobIds: [],
        }),
      });
      // The authoritative snapshot safely prunes connection-old tombstones:
      // pre-barrier publications remain rejected by revision even without them.
      source.listeners["job.updated"]({
        lastEventId: "599",
        data: JSON.stringify(clearedJobs[0]),
      });
    });
    expect(jobs).toEqual(activeJobs);

    act(() => {
      source.listeners["job.updated"]({
        lastEventId: "601",
        data: JSON.stringify(clearedJobs[0]),
      });
    });
    expect(jobs.map((job) => job.id)).toContain(clearedJobs[0].id);
    expect(jobs).toHaveLength(activeJobs.length + 1);
  });

  it("replays each missed terminal revision exactly once across its snapshot and buffered event", async () => {
    let jobs = [
      {
        id: "model-job",
        type: "model_download",
        status: "running",
        createdAt: "2026-01-02T00:00:00Z",
        updatedAt: "2026-01-02T00:01:00Z",
      },
      {
        id: "image-job",
        type: "image_generate",
        projectId: "project-1",
        status: "running",
        createdAt: "2026-01-01T00:00:00Z",
        updatedAt: "2026-01-01T00:01:00Z",
        result: {},
      },
      {
        id: "failed-job",
        type: "video_generate",
        status: "running",
        createdAt: "2025-12-31T00:00:00Z",
        updatedAt: "2025-12-31T00:01:00Z",
      },
    ];
    const jobsRef = { current: jobs };
    const refreshData = vi.fn();
    const refreshAssets = vi.fn();
    const applyTimeline = vi.fn();
    const pushNotice = vi.fn();
    apiResponders.set("/api/v1/jobs/events/ticket", { ticket: "bounded-ticket" });
    mount(baseProps({
      jobsRef,
      setJobs: (updater) => {
        jobs = updater(jobs);
        jobsRef.current = jobs;
      },
      refreshDataRef: { current: refreshData },
      refreshAssetsRef: { current: refreshAssets },
      enqueueTimelineGenerationApply: applyTimeline,
      pushNotice,
    }));
    await settle();
    const source = FakeEventSource.instances[0];
    expect(source.url).toBe("/api/v1/jobs/events?ticket=bounded-ticket");
    const mintCall = apiFetch.mock.calls.find(([path]) => path === "/api/v1/jobs/events/ticket");
    expect(JSON.parse(mintCall[2].body)).toEqual({
      activeJobIds: [
      "model-job",
      "image-job",
      "failed-job",
      ],
      knownTerminalJobIds: [],
    });

    const completedModel = {
      ...jobs[0],
      status: "completed",
      revision: 7,
      updatedAt: "2026-01-02T00:02:00Z",
    };
    const completedImage = {
      ...jobs[1],
      status: "completed",
      revision: 8,
      updatedAt: "2026-01-01T00:02:00Z",
      result: { generationSetId: "set-1", assetIds: ["asset-1"] },
    };
    const failedVideo = {
      ...jobs[2],
      status: "failed",
      revision: 9,
      updatedAt: "2025-12-31T00:02:00Z",
      error: "renderer failed",
    };
    act(() => {
      source.listeners["jobs.snapshot"]({
        lastEventId: "20",
        data: JSON.stringify({
          revision: 20,
          jobs: [completedModel, completedImage, failedVideo],
          clearedJobIds: [],
        }),
      });
      // These rows committed after the snapshot event barrier, so their SSE ids
      // are newer even though the snapshot already captured the same durable
      // per-job revisions. State may be upserted again, but terminal effects
      // (especially timeline mutation) must not run twice.
      source.listeners["job.updated"]({
        lastEventId: "21",
        data: JSON.stringify(completedModel),
      });
      source.listeners["job.updated"]({
        lastEventId: "22",
        data: JSON.stringify(completedImage),
      });
      source.listeners["job.updated"]({
        lastEventId: "23",
        data: JSON.stringify(failedVideo),
      });
    });

    expect(refreshData).toHaveBeenCalledTimes(1);
    expect(refreshAssets).toHaveBeenCalledTimes(1);
    expect(refreshAssets).toHaveBeenCalledWith("project-1");
    expect(applyTimeline).toHaveBeenCalledTimes(1);
    expect(applyTimeline).toHaveBeenCalledWith(completedImage);
    expect(pushNotice).toHaveBeenCalledTimes(1);
    expect(pushNotice.mock.calls[0][1]).toContain("renderer failed");
  });

  it("keeps hundreds of active ids in the ticket POST while the EventSource URL stays bounded", async () => {
    const jobs = Array.from({ length: 200 }, (_, index) => ({
      id: `00000000-0000-4000-8000-${String(index).padStart(12, "0")}`,
      status: "running",
      createdAt: "2026-01-01T00:00:00Z",
    }));
    apiResponders.set("/api/v1/jobs/events/ticket", { ticket: "short-ticket" });
    mount(baseProps({ jobsRef: { current: jobs } }));
    await settle();

    const mintCall = apiFetch.mock.calls.find(([path]) => path === "/api/v1/jobs/events/ticket");
    expect(JSON.parse(mintCall[2].body)).toEqual({
      activeJobIds: jobs.map((job) => job.id),
      knownTerminalJobIds: [],
    });
    expect(FakeEventSource.instances[0].url).toBe("/api/v1/jobs/events?ticket=short-ticket");
    expect(FakeEventSource.instances[0].url).not.toContain("activeJobIds");
  });

  it("prunes terminal revision dedupe state with the visible terminal cap", async () => {
    let jobs = [];
    const jobsRef = { current: jobs };
    const pushNotice = vi.fn();
    mount(baseProps({
      jobsRef,
      setJobs: (updater) => {
        jobs = updater(jobs);
        jobsRef.current = jobs;
      },
      pushNotice,
    }));
    await settle();
    const source = FakeEventSource.instances[0];
    const terminalJobs = Array.from({ length: MAX_TERMINAL_JOBS + 1 }, (_, index) => ({
      id: `failed-${index}`,
      type: "image_generate",
      status: "failed",
      revision: 1,
      createdAt: new Date(Date.UTC(2026, 0, 1, 0, 0, index)).toISOString(),
      updatedAt: new Date(Date.UTC(2026, 0, 1, 1, 0, index)).toISOString(),
      error: `failure-${index}`,
    }));

    act(() => {
      terminalJobs.forEach((job, index) => {
        source.listeners["job.updated"]({
          lastEventId: String(index + 1),
          data: JSON.stringify(job),
        });
      });
    });

    expect(jobs).toHaveLength(MAX_TERMINAL_JOBS);
    expect(jobs.some((job) => job.id === terminalJobs[0].id)).toBe(false);
    expect(pushNotice).toHaveBeenCalledTimes(MAX_TERMINAL_JOBS + 1);

    // The evicted row's revision bookkeeping must be gone too. If the private
    // Map leaked one entry per terminal job, this same durable revision would be
    // mistaken for an already-visible duplicate and suppress its effect.
    act(() => {
      source.listeners["job.updated"]({
        lastEventId: String(MAX_TERMINAL_JOBS + 2),
        data: JSON.stringify(terminalJobs[0]),
      });
    });

    expect(jobs).toHaveLength(MAX_TERMINAL_JOBS);
    expect(jobs.some((job) => job.id === terminalJobs[0].id)).toBe(false);
    expect(pushNotice).toHaveBeenCalledTimes(MAX_TERMINAL_JOBS + 2);
  });

  it("seeds terminal snapshot revisions without replaying already-observed terminal effects", async () => {
    const completed = {
      id: "already-terminal",
      type: "image_generate",
      projectId: "project-1",
      status: "completed",
      revision: 7,
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:02:00Z",
      result: { generationSetId: "set-1", assetIds: ["asset-1"] },
    };
    let jobs = [completed];
    const jobsRef = { current: jobs };
    const refreshAssets = vi.fn();
    const applyTimeline = vi.fn();
    mount(baseProps({
      jobsRef,
      setJobs: (updater) => {
        jobs = updater(jobs);
        jobsRef.current = jobs;
      },
      refreshAssetsRef: { current: refreshAssets },
      enqueueTimelineGenerationApply: applyTimeline,
    }));
    await settle();
    const source = FakeEventSource.instances[0];

    act(() => {
      source.listeners["jobs.snapshot"]({
        lastEventId: "20",
        data: JSON.stringify({
          revision: 20,
          jobs: [completed],
          clearedJobIds: [],
        }),
      });
      source.listeners["job.updated"]({
        lastEventId: "21",
        data: JSON.stringify({
          ...completed,
          revision: 6,
        }),
      });
      source.listeners["job.updated"]({
        lastEventId: "22",
        data: JSON.stringify(completed),
      });
    });

    expect(jobs).toEqual([completed]);
    expect(refreshAssets).not.toHaveBeenCalled();
    expect(applyTimeline).not.toHaveBeenCalled();
  });

  it("does not run effects for a live terminal row rejected by durable state freshness", async () => {
    const current = {
      id: "newer-terminal",
      type: "image_generate",
      projectId: "project-1",
      status: "completed",
      revision: 8,
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:02:00Z",
      result: { generationSetId: "set-1", assetIds: ["asset-1"] },
    };
    let jobs = [current];
    const jobsRef = { current: jobs };
    const refreshAssets = vi.fn();
    const applyTimeline = vi.fn();
    mount(baseProps({
      jobsRef,
      setJobs: (updater) => {
        jobs = updater(jobs);
        jobsRef.current = jobs;
      },
      refreshAssetsRef: { current: refreshAssets },
      enqueueTimelineGenerationApply: applyTimeline,
    }));
    await settle();

    act(() => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        lastEventId: "40",
        data: JSON.stringify({
          ...current,
          revision: 7,
        }),
      });
    });

    expect(jobs).toEqual([current]);
    expect(refreshAssets).not.toHaveBeenCalled();
    expect(applyTimeline).not.toHaveBeenCalled();
  });

  it("keeps a clear tombstone authoritative over a delayed terminal publication", async () => {
    const completed = {
      id: "cleared-terminal",
      type: "image_generate",
      projectId: "project-1",
      status: "completed",
      revision: 4,
      createdAt: "2026-01-01T00:00:00Z",
      updatedAt: "2026-01-01T00:02:00Z",
      result: { generationSetId: "set-1", assetIds: ["asset-1"] },
    };
    let jobs = [completed];
    const jobsRef = { current: jobs };
    const refreshAssets = vi.fn();
    const applyTimeline = vi.fn();
    mount(baseProps({
      jobsRef,
      setJobs: (updater) => {
        jobs = updater(jobs);
        jobsRef.current = jobs;
      },
      refreshAssetsRef: { current: refreshAssets },
      enqueueTimelineGenerationApply: applyTimeline,
    }));
    await settle();
    const source = FakeEventSource.instances[0];

    act(() => {
      source.listeners["jobs.cleared"]({
        lastEventId: "30",
        data: JSON.stringify({ ids: [completed.id] }),
      });
      source.listeners["job.updated"]({
        lastEventId: "31",
        data: JSON.stringify(completed),
      });
    });

    expect(jobs).toEqual([]);
    expect(refreshAssets).not.toHaveBeenCalled();
    expect(applyTimeline).not.toHaveBeenCalled();
  });

  it("closes the EventSource on unmount", async () => {
    mount(baseProps({ ready: true }));
    await settle();
    const source = FakeEventSource.instances[0];
    expect(source.closed).toBe(false);

    act(() => root.unmount());
    root = null;
    expect(source.closed).toBe(true);
  });
});

// sc-11231 (F-037): useJobEvents' SSE effect captures enqueueTimelineGenerationApply and
// hasVisibleLocalFailure at SUBSCRIBE time (deps are only [access.authRequired, ready,
// token]). If those inputs are plain per-render function declarations, the stream keeps
// invoking the closure from the subscribe render — a latent stale-closure. These tests
// pin the useTimelines side of the fix: the exposed enqueue callback is identity-stable
// AND, because it delegates through a per-commit ref, the callback captured at subscribe
// time still runs against live state (the fresh token) — exactly what the SSE handler needs.
describe("useTimelines.enqueueTimelineGenerationApply stability + liveness (sc-11231)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    apiResponders.clear();
    apiFetch.mockClear();
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
  });

  function mount() {
    const activeProject = { id: "p1", name: "P1" };
    // Stable across renders — mirrors App, where setError is a useState setter (stable).
    const stable = {
      activeProject,
      activeProjectRef: { current: activeProject },
      setError: () => {},
      setActiveView: () => {},
      createVideoJob: async () => null,
    };
    let latest = null;
    function Harness() {
      const [token, setToken] = useState("t1");
      const api = useTimelines({
        token,
        requestedGpu: "auto",
        ...stable,
      });
      latest = { enqueue: api.enqueueTimelineGenerationApply, setToken };
      return null;
    }
    root = createRoot(container);
    act(() => root.render(<Harness />));
    return { get: () => latest };
  }

  it("keeps a stable enqueue identity that still applies with the live token after a re-render", async () => {
    // A completed generation whose result carries assets — enough to reach the timeline
    // GET inside applyCompletedTimelineGeneration, where the live token is threaded.
    const job = {
      id: "job-1",
      projectId: "p1",
      status: "completed",
      payload: { advanced: { timelineContext: { timelineId: "tl1" } } },
      result: { assetIds: ["a1"] },
    };
    apiResponders.set("/api/v1/projects/p1/timelines/tl1", { id: "tl1", tracks: [] });

    const { get } = mount();
    await settle();
    // Capture the callback at "subscribe time" (as useJobEvents would).
    const enqueueAtSubscribe = get().enqueue;

    // Token advances on an unrelated re-render (fresh login / rotation).
    await act(async () => get().setToken("t2"));
    await settle();

    // Stability: the captured callback is the SAME identity the hook still exposes — so a
    // stream that subscribed earlier is not holding a dead closure.
    expect(typeof enqueueAtSubscribe).toBe("function");
    expect(enqueueAtSubscribe).toBe(get().enqueue);

    // Liveness: invoking the callback captured BEFORE the token changed must still run
    // against the live token ("t2"), not the stale "t1". Pre-fix (plain fn delegating to a
    // per-render closure) the subscribe-time callback would have used "t1".
    await act(async () => {
      enqueueAtSubscribe(job);
      await settle();
    });
    const timelineGet = apiFetch.mock.calls.find(
      (call) => call[0] === "/api/v1/projects/p1/timelines/tl1" && (call[2]?.method ?? "GET") === "GET",
    );
    expect(timelineGet).toBeTruthy();
    expect(timelineGet[1]).toBe("t2");
  });
});
