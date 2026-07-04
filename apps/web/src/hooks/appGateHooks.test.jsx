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
    eventUrl: (path, ticket) => `${path}${ticket ? `?ticket=${ticket}` : ""}`,
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
