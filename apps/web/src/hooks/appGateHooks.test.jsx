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
//
// sc-15105: the mock reproduces the one behavior of the real `apiFetch` that the hook
// depends on — a 401 notifies the handler registered via `setUnauthorizedHandler` before
// the rejection propagates. (That api.js contract is pinned directly in api.test.js;
// here it is the seam under test.) `unauthorizedHandler` is captured rather than left in
// the real module so a test can also assert whether one was registered at all.
const apiResponders = new Map();
const setMediaTicketSpy = vi.fn();
let unauthorizedHandler = null;
// Build the rejection the API produces for a stale token, using the real ApiError.
async function unauthorizedError() {
  const { ApiError } = await vi.importActual("../api.js");
  return new ApiError("SceneWorks access token required", {
    status: 401,
    detail: "SceneWorks access token required",
    authRequired: true,
  });
}
vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn((path, token) => {
      const responder = apiResponders.get(path);
      const value = typeof responder === "function" ? responder() : responder;
      if (value instanceof Error) {
        // Mirrors api.js: only a 401 on a request that actually presented the token
        // reaches the handler, and the handler's boolean lands on the error. A test can
        // opt a specific error out of the notify to isolate a caller-side seam.
        if (value.status === 401 && token && value.notifyGate !== false) {
          value.reauthenticating = unauthorizedHandler?.(value) === true;
        }
        return Promise.reject(value);
      }
      return Promise.resolve(value ?? {});
    }),
    setUnauthorizedHandler: (handler) => {
      const next = typeof handler === "function" ? handler : null;
      unauthorizedHandler = next;
      return () => {
        if (unauthorizedHandler === next) {
          unauthorizedHandler = null;
        }
      };
    },
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
    unauthorizedHandler = null;
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
    // Every push, including ones a later dismiss removes from `notices`. Asserting on the
    // surviving array alone cannot see a notice that is raised and then swept away by the
    // gate coming up one render later — which is exactly the misleading flash sc-15105's
    // mint seam exists to prevent.
    const pushed = [];
    // Every gateStatus this hook has rendered. A mid-session demotion is a transient the
    // final state cannot show: the shell blocks, then re-opens if the re-verify passes.
    const statuses = [];
    // App feeds these three in identity-stable (useCallback with []), and the hook's
    // header documents that contract — its effect dependency arrays are built on it.
    // Declared outside the component so the harness honors it too: fresh closures per
    // render would re-run the mint / re-verify effects on every unrelated state change,
    // which is a property of the harness, not of the hook (sc-15105).
    const deps = {
      setError: () => {},
      pushNotice: (kind, message) => {
        pushed.push({ kind, message });
        notices.push({ kind, message });
      },
      dismissNoticeKind: (kind) => {
        for (let i = notices.length - 1; i >= 0; i -= 1) {
          if (notices[i].kind === kind) notices.splice(i, 1);
        }
      },
    };
    function Harness() {
      const [, setN] = useState(0);
      const api = useAccessGate(deps);
      statuses.push(api.gateStatus);
      latest = { api, rerender: () => setN((n) => n + 1) };
      return null;
    }
    root = createRoot(container);
    act(() => root.render(<Harness />));
    return { get: () => latest, notices, pushed, statuses };
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
    // sc-15102: a token restored from storage is re-verified before it authenticates.
    apiResponders.set("/api/v1/auth/verify", { ok: true });
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
    apiResponders.set("/api/v1/auth/verify", { ok: true });
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
    apiResponders.set("/api/v1/auth/verify", { ok: true });
    apiResponders.set("/api/v1/files/ticket", { ticket: "media-1", expiresInSeconds: 300 });
    const { get } = mount();
    await settle();
    expect(get().api.token).toBe("remote-token");

    act(() => get().api.lockRemote());
    await settle();

    expect(get().api.token).toBe("");
    expect(get().api.authenticated).toBe(false);
    expect(get().api.gateStatus).toBe("locked");
    expect(window.localStorage.getItem("sceneworks-token")).toBeNull();
  });

  // sc-15102: the stored token is a guess, not a credential. The host password can be
  // changed or cleared between visits, and no client path re-prompts on a 401 — so a
  // token that fails verification must be dropped back to the password prompt instead
  // of authenticating and 401ing every request behind an app that looks merely broken.
  describe("stored-token re-verification (sc-15102)", () => {
    it("drops a stored token the host rejects and returns to the locked gate", async () => {
      window.localStorage.setItem("sceneworks-token", "stale-token");
      apiResponders.set("/api/v1/access", { authRequired: true });
      apiResponders.set("/api/v1/auth/verify", { ok: false });
      const { get } = mount();
      await settle();

      expect(get().api.token).toBe("");
      expect(get().api.authenticated).toBe(false);
      expect(get().api.gateStatus).toBe("locked");
      expect(get().api.authError).toContain("no longer works");
      expect(window.localStorage.getItem("sceneworks-token")).toBeNull();
    });

    it("never authenticates on an unverified stored token: no mint, no data-load release", async () => {
      window.localStorage.setItem("sceneworks-token", "stale-token");
      apiResponders.set("/api/v1/access", { authRequired: true });
      apiResponders.set("/api/v1/auth/verify", { ok: false });
      apiResponders.set("/api/v1/files/ticket", { ticket: "media-1", expiresInSeconds: 300 });
      const { get } = mount();
      await settle();

      // The mint (and therefore every protected load gated on `ready`) is downstream of
      // `authenticated`, so a rejected token must leave both false.
      expect(get().api.ready).toBe(false);
      expect(setMediaTicketSpy).not.toHaveBeenCalledWith("media-1");
    });

    it("keeps the token and shows 'unlocking' while the host is unreachable", async () => {
      window.localStorage.setItem("sceneworks-token", "remote-token");
      apiResponders.set("/api/v1/access", { authRequired: true });
      apiResponders.set("/api/v1/auth/verify", () => new Error("host down"));
      const { get, notices } = mount();
      await settle();

      // Unreachable ≠ rejected: hold the token, block the shell, and retry with backoff.
      expect(get().api.token).toBe("remote-token");
      expect(get().api.gateStatus).toBe("unlocking");
      expect(get().api.authenticated).toBe(false);
      expect(notices.some((n) => n.kind === "access-verify")).toBe(true);
      expect(window.localStorage.getItem("sceneworks-token")).toBe("remote-token");
    });

    it("saveToken does not re-verify the password it just proved", async () => {
      apiResponders.set("/api/v1/access", { authRequired: true });
      apiResponders.set("/api/v1/auth/verify", { ok: true });
      apiResponders.set("/api/v1/files/ticket", { ticket: "media-1", expiresInSeconds: 300 });
      const { get } = mount();
      await settle();

      // apiFetch is a file-scoped mock that isn't reset between these tests, so count the
      // verifies this unlock is responsible for rather than the mock's lifetime total.
      const verifies = () => apiFetch.mock.calls.filter((call) => call[0] === "/api/v1/auth/verify").length;
      const before = verifies();

      act(() => get().api.setPasswordDraft("secret"));
      await act(async () => {
        await get().api.saveToken({ preventDefault: () => {} });
      });
      await settle();

      expect(get().api.gateStatus).toBe("open");
      // Exactly one: saveToken's own check. The re-verify effect must recognize the token
      // as already proved and not POST it again (which would re-block the shell).
      expect(verifies() - before).toBe(1);
    });
  });

  // sc-15105: the same token can go stale WHILE the tab is open — the host rotates its
  // password under Settings → Remote access and restarts the API. sc-15102 re-verified
  // only at startup, so an already-open remote tab kept `authenticated` true, 401'd every
  // request into the notice bands, and never re-prompted.
  describe("mid-session 401 (sc-15105)", () => {
    // Take a stored token all the way to an unlocked session so the 401 below arrives
    // against tokenStatus "accepted" — the state the bug lived in.
    async function mountUnlocked({ verify, ticket } = {}) {
      window.localStorage.setItem("sceneworks-token", "old-password");
      apiResponders.set("/api/v1/access", { authRequired: true });
      apiResponders.set("/api/v1/auth/verify", verify ?? { ok: true });
      apiResponders.set(
        "/api/v1/files/ticket",
        ticket ?? { ticket: "media-1", expiresInSeconds: 300 },
      );
      const mounted = mount();
      await settle();
      return mounted;
    }

    // Fire `count` 401s the way a real gated route does — concurrently, as a page full of
    // screens would when the host restarts under them. The token argument matters: api.js
    // only routes a 401 to the gate when the request actually presented a credential.
    async function fire401(count = 1, token = "old-password") {
      const stale = await unauthorizedError();
      apiResponders.set("/api/v1/jobs", () => stale);
      await act(async () => {
        await Promise.all(
          Array.from({ length: count }, () =>
            apiFetch("/api/v1/jobs", token).catch(() => {}),
          ),
        );
      });
      await settle();
    }

    const verifyCount = () =>
      apiFetch.mock.calls.filter((call) => call[0] === "/api/v1/auth/verify").length;

    it("re-prompts an already-open tab when the re-verify rejects the token", async () => {
      let verify = { ok: true };
      const { get } = await mountUnlocked({ verify: () => verify });
      expect(get().api.gateStatus).toBe("open");

      // The host rotated its password and restarted: the live token is now wrong.
      verify = { ok: false };
      await fire401();

      // No reload needed — the full-page blocker takes over and asks for the new one.
      expect(get().api.gateStatus).toBe("locked");
      expect(get().api.token).toBe("");
      expect(get().api.authError).toContain("no longer works");
      expect(window.localStorage.getItem("sceneworks-token")).toBeNull();
    });

    it("unlocks with the new password after a mid-session rotation", async () => {
      let verify = { ok: true };
      const { get } = await mountUnlocked({ verify: () => verify });
      verify = { ok: false };
      await fire401();
      expect(get().api.gateStatus).toBe("locked");

      verify = { ok: true };
      act(() => get().api.setPasswordDraft("new-password"));
      await act(async () => {
        await get().api.saveToken({ preventDefault: () => {} });
      });
      await settle();

      expect(get().api.gateStatus).toBe("open");
      expect(get().api.token).toBe("new-password");
      expect(window.localStorage.getItem("sceneworks-token")).toBe("new-password");
    });

    it("does not log the user out when the re-verify still passes", async () => {
      const { get, statuses } = await mountUnlocked({ verify: { ok: true } });
      const before = verifyCount();
      statuses.length = 0;

      await fire401();

      // The 401 really was adjudicated — asserting only the end state would pass just as
      // well with no handler registered at all, since "nothing happened" looks identical.
      expect(verifyCount() - before).toBe(1);
      expect(statuses).toContain("unlocking");
      // ...and one unlucky endpoint did not evict a session the host still accepts.
      expect(statuses[statuses.length - 1]).toBe("open");
      expect(get().api.token).toBe("old-password");
      expect(window.localStorage.getItem("sceneworks-token")).toBe("old-password");
      // The session is fully live again, not merely unblocked.
      expect(get().api.ready).toBe(true);
    });

    it("ignores a 401 from a request that presented no token", async () => {
      // Several callers hit gated routes with an empty token on purpose (the
      // `PUT /api/v1/ui-preferences` writes) and swallow the rejection. Those 401s say
      // nothing about the session, and must not drag the blocker over a live app.
      const { get, statuses } = await mountUnlocked({ verify: { ok: true } });
      const before = verifyCount();
      statuses.length = 0;

      await fire401(1, "");

      expect(verifyCount()).toBe(before);
      expect(statuses.every((status) => status === "open")).toBe(true);
      expect(get().api.token).toBe("old-password");
    });

    it("leaves an already-locked gate alone instead of stranding it on 'unlocking'", async () => {
      // tokenStatus "none" means the gate is up with no token to check. Demoting it to
      // "pending" would render "unlocking" forever: the re-verify effect bails on an empty
      // token, so nothing would ever move it off that state.
      apiResponders.set("/api/v1/access", { authRequired: true });
      const { get } = mount();
      await settle();
      expect(get().api.gateStatus).toBe("locked");

      await fire401(1, "whatever-stale-token");

      expect(get().api.gateStatus).toBe("locked");
    });

    it("issues exactly one /auth/verify for a storm of concurrent 401s", async () => {
      const { get } = await mountUnlocked({ verify: { ok: true } });
      const before = verifyCount();

      await fire401(8);

      // The guard is a ref, not state: all eight land before React flushes.
      expect(verifyCount() - before).toBe(1);
      expect(get().api.gateStatus).toBe("open");

      // And the storm is not a one-shot fuse — a later rotation is still detected.
      apiResponders.set("/api/v1/auth/verify", { ok: false });
      await fire401(3);
      expect(get().api.gateStatus).toBe("locked");
    });

    it("reports a throttled host as a lockout, not as an unreachable one", async () => {
      // A rotation storm can trip the API's per-IP attempt throttle, which answers 429 —
      // including on the public verify route. "Couldn't reach the host" sends the user
      // hunting a network fault instead of waiting out a lockout that clears itself.
      const { ApiError } = await vi.importActual("../api.js");
      apiResponders.set("/api/v1/access", { authRequired: true });
      apiResponders.set(
        "/api/v1/auth/verify",
        () => new ApiError("Too many attempts", { status: 429 }),
      );
      const { get } = mount();
      await settle();

      act(() => get().api.setPasswordDraft("new-password"));
      await act(async () => {
        await get().api.saveToken({ preventDefault: () => {} });
      });
      await settle();

      expect(get().api.authError).toContain("Too many password attempts");
    });

    it("surrenders the media-ticket mint to the gate instead of retrying forever", async () => {
      const stale = await unauthorizedError();
      // The tab unlocked on the old password (first verify), then the host rotated it —
      // so the mint's 401 is the first thing that notices, and the re-verify it triggers
      // finds the token rejected.
      let verifyCalls = 0;
      const { get, pushed } = await mountUnlocked({
        verify: () => (verifyCalls++ === 0 ? { ok: true } : { ok: false }),
        // The mint is itself a gated route, so it 401s along with everything else.
        ticket: () => stale,
      });

      expect(get().api.gateStatus).toBe("locked");
      // The "retrying in the background" notice is never raised: nothing is retrying, and
      // the gate is what the user needs to see. NOTE: this assertion holds with and
      // without the mint's own `isReauthenticating` early return, because `act` flushes
      // the demotion (and so the effect teardown that sets `closed`) before the rejection
      // continuation runs, and the pre-existing `closed` check then short-circuits the
      // catch. A real browser resolves that race the other way — the catch runs first —
      // which is what the early return is actually for. The outcome below is the
      // regression guard; the ordering itself is not reproducible under `act`.
      expect(pushed.some((n) => n.kind === "media-ticket")).toBe(false);
      expect(get().api.ready).toBe(false);
    });

    it("raises no media-ticket notice for a 401 the gate has claimed", async () => {
      // The mint's own seam, isolated from the demotion. `notifyGate:false` keeps the gate
      // from demoting (so the effect is never torn down and its `closed` short-circuit
      // cannot mask the result), while `reauthenticating` marks the error as claimed —
      // exactly the state a real browser is in when the rejection lands before React
      // commits the teardown.
      const stale = await unauthorizedError();
      stale.notifyGate = false;
      stale.reauthenticating = true;
      const { get, pushed } = await mountUnlocked({
        verify: { ok: true },
        ticket: () => stale,
      });

      // No "retrying in the background" lie, and no backoff mint armed behind the gate.
      expect(pushed.some((n) => n.kind === "media-ticket")).toBe(false);
      expect(get().api.gateStatus).toBe("open");
    });

    it("still degrades media normally for a 401 the gate did not claim", async () => {
      const stale = await unauthorizedError();
      stale.notifyGate = false;
      const { get, pushed } = await mountUnlocked({
        verify: { ok: true },
        ticket: () => stale,
      });

      // Unclaimed ⇒ the pre-existing sc-9063 behavior: degraded media, a notice, and the
      // app still usable. Surrendering here would silently break thumbnails instead.
      expect(pushed.some((n) => n.kind === "media-ticket")).toBe(true);
      expect(get().api.ready).toBe(true);
    });

    it("declines a 401 on a deployment that requires no password", async () => {
      // A host that answers /api/v1/access with authRequired:false cannot produce a 401
      // a password prompt would fix — it disagrees with itself. Claiming it would silence
      // the caller's own error handling behind a gate that will never appear.
      apiResponders.set("/api/v1/access", { authRequired: false });
      const { get } = mount();
      await settle();
      expect(get().api.gateStatus).toBe("open");

      const stale = await unauthorizedError();
      apiResponders.set("/api/v1/jobs", () => stale);
      const err = await act(async () =>
        apiFetch("/api/v1/jobs", "some-token").catch((caught) => caught),
      );

      expect(err.reauthenticating).toBe(false);
      expect(get().api.gateStatus).toBe("open");
    });

    it("registers a 401 handler in remote-browser mode", async () => {
      await mountUnlocked({ verify: { ok: true } });
      // The desktop counterpart (no handler at all) is pinned in
      // useAccessGate.desktop.test.jsx, which mocks isDesktop true.
      expect(typeof unauthorizedHandler).toBe("function");
    });
  });

  // sc-15102: `gateStatus` is the single blocking decision App branches on.
  describe("gateStatus", () => {
    it("stays open while the first access probe is in flight, then blocks once it fails", async () => {
      apiResponders.set("/api/v1/access", () => new Error("probe exploded"));
      const { get } = mount();

      // Pre-settle: nothing has failed yet, so a healthy host never flashes a blocker.
      expect(get().api.gateStatus).toBe("open");

      await settle();
      // A failed probe means the auth requirement is unknown — block rather than hand
      // over a navigable, permanently empty shell.
      expect(get().api.gateStatus).toBe("awaiting-host");
    });

    it("is open when the host requires no password", async () => {
      apiResponders.set("/api/v1/access", { authRequired: false });
      const { get } = mount();
      await settle();

      expect(get().api.gateStatus).toBe("open");
    });
  });
});

describe("useJobEvents (sc-9750)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    apiResponders.clear();
    apiFetch.mockClear();
    unauthorizedHandler = null;
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
      refreshModelsRef: { current: () => {} },
      refreshModelAndLorasRef: { current: () => {} },
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

  // sc-15105: the SSE ticket POST is a gated route, so a mid-session password rotation
  // 401s it too. It used to push the raw "SceneWorks access token required" into the error
  // band and arm an exponential-backoff reconnect that could never succeed.
  it("surrenders the SSE ticket to the access gate on a claimed 401", async () => {
    const stale = await unauthorizedError();
    // No gate is mounted in this describe, so mark the error claimed directly and opt it
    // out of the mock's notify (which would otherwise overwrite the flag with false).
    stale.notifyGate = false;
    stale.reauthenticating = true;
    const errors = [];
    apiResponders.set("/api/v1/jobs/events/ticket", () => stale);
    mount(
      baseProps({
        access: { authRequired: true },
        token: "old-password",
        setError: (message) => errors.push(message),
      }),
    );
    await settle();

    // No error-band copy and no EventSource: the gate is taking over, and `ready` will
    // drop out from under this effect.
    expect(errors).toEqual([]);
    expect(FakeEventSource.instances.length).toBe(0);
  });

  it("still reports a 401 the gate did not claim as an ordinary ticket failure", async () => {
    const stale = await unauthorizedError();
    const errors = [];
    apiResponders.set("/api/v1/jobs/events/ticket", () => stale);
    mount(
      baseProps({
        access: { authRequired: true },
        token: "old-password",
        setError: (message) => errors.push(message),
      }),
    );
    await settle();

    // `reauthenticating` false (no gate registered, or the gate declined) ⇒ the pre-existing
    // notice + backoff behavior, not silence.
    expect(errors).toEqual(["SceneWorks access token required"]);
  });

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
      refreshModelsRef: { current: refreshData },
      refreshModelAndLorasRef: { current: () => {} },
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
