import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  apiFetch,
  isReauthenticating,
  isThrottled,
  isUnauthorized,
  setUnauthorizedHandler,
} from "./api.js";

describe("apiFetch response handling", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("stringifies structured API details instead of coercing them to [object Object]", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ detail: { field: "prompt", reason: "required" } }), {
        status: 422,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await expect(apiFetch("/structured-error", "")).rejects.toThrow(
      '{"field":"prompt","reason":"required"}',
    );
  });

  it.each([
    ["204 response", new Response(null, { status: 204 })],
    ["empty successful response", new Response("", { status: 200 })],
  ])("returns null for an empty %s", async (_label, response) => {
    global.fetch = vi.fn(async () => response);

    await expect(apiFetch("/empty", "")).resolves.toBeNull();
  });
});

// sc-15105: the thrown failure now carries the HTTP status, which is the seam the
// mid-session re-prompt hangs on. The message must not move — ~85 call sites do
// `setError(err.message)` and their notice copy is user-visible.
describe("ApiError (sc-15105)", () => {
  afterEach(() => {
    setUnauthorizedHandler(null);
    vi.restoreAllMocks();
  });

  function respondWith(body, status) {
    global.fetch = vi.fn(async () =>
      new Response(body === null ? "" : JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    );
  }

  it("carries status, detail and authRequired off the API's 401 body", async () => {
    respondWith({ detail: "SceneWorks access token required", authRequired: true }, 401);

    const err = await apiFetch("/api/v1/jobs", "stale").catch((caught) => caught);

    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(401);
    expect(err.detail).toBe("SceneWorks access token required");
    expect(err.authRequired).toBe(true);
    // The byte-identical message the bare Error used to carry.
    expect(err.message).toBe("SceneWorks access token required");
  });

  it("carries the status on non-401 failures without claiming authRequired", async () => {
    respondWith({ detail: "nope" }, 500);

    const err = await apiFetch("/api/v1/jobs", "").catch((caught) => caught);

    expect(err.status).toBe(500);
    expect(err.authRequired).toBe(false);
    expect(isUnauthorized(err)).toBe(false);
  });

  it("keeps the synthesized message for a failure body with no detail", async () => {
    respondWith(null, 503);

    const err = await apiFetch("/api/v1/jobs", "").catch((caught) => caught);

    expect(err.message).toBe("Request failed with 503");
    expect(err.status).toBe(503);
  });

  it("isUnauthorized ignores errors that are not an API 401", () => {
    expect(isUnauthorized(new Error("Failed to fetch"))).toBe(false);
    expect(isUnauthorized(undefined)).toBe(false);
    expect(isUnauthorized(new ApiError("x", { status: 403 }))).toBe(false);
    expect(isUnauthorized(new ApiError("x", { status: 401 }))).toBe(true);
  });

  it("flags the API's per-IP attempt throttle separately from an unreachable host", async () => {
    respondWith({ detail: "Too many attempts" }, 429);

    const err = await apiFetch("/api/v1/auth/verify", "guess").catch((caught) => caught);

    expect(isThrottled(err)).toBe(true);
    expect(isThrottled(new Error("Failed to fetch"))).toBe(false);
    expect(isThrottled(new ApiError("x", { status: 401 }))).toBe(false);
  });

  it("notifies the registered handler on a 401, and only on a 401", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);

    respondWith({ detail: "SceneWorks access token required", authRequired: true }, 401);
    await apiFetch("/api/v1/jobs", "stale").catch(() => {});
    expect(handler).toHaveBeenCalledTimes(1);
    expect(handler.mock.calls[0][0].status).toBe(401);

    respondWith({ detail: "forbidden" }, 403);
    await apiFetch("/api/v1/jobs", "stale").catch(() => {});
    respondWith({ ok: true }, 200);
    await apiFetch("/api/v1/jobs", "stale");
    expect(handler).toHaveBeenCalledTimes(1);
  });

  // A 401 on a request that deliberately presented no credential says nothing about the
  // session token, so it must not raise the gate. The motivating case was the
  // `PUT /api/v1/ui-preferences` writes (theme, accent, default quality, last tier) hitting
  // that gated route with "" and a swallowed rejection — routing those to the gate would
  // slam a remote browser into the full-page blocker on a theme toggle. Those writes now
  // send the real token (sc-15136); the rule itself is about the credential, not the route.
  it("ignores a 401 from a request that presented no token", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    respondWith({ detail: "SceneWorks access token required", authRequired: true }, 401);

    const err = await apiFetch("/api/v1/ui-preferences", "").catch((caught) => caught);

    expect(handler).not.toHaveBeenCalled();
    // Still a fully-formed failure for the caller — just not the gate's business.
    expect(err.status).toBe(401);
    expect(err.reauthenticating).toBe(false);
  });

  it("records on the error whether the handler claimed the session", async () => {
    respondWith({ detail: "SceneWorks access token required", authRequired: true }, 401);

    setUnauthorizedHandler(() => true);
    let err = await apiFetch("/api/v1/jobs", "stale").catch((caught) => caught);
    expect(err.reauthenticating).toBe(true);
    expect(isReauthenticating(err)).toBe(true);

    setUnauthorizedHandler(() => false);
    err = await apiFetch("/api/v1/jobs", "stale").catch((caught) => caught);
    expect(err.reauthenticating).toBe(false);
    expect(isReauthenticating(err)).toBe(false);

    // A handler returning a truthy non-boolean is not a claim either.
    setUnauthorizedHandler(() => "yes");
    err = await apiFetch("/api/v1/jobs", "stale").catch((caught) => caught);
    expect(err.reauthenticating).toBe(false);

    setUnauthorizedHandler(null);
    err = await apiFetch("/api/v1/jobs", "stale").catch((caught) => caught);
    expect(err.reauthenticating).toBe(false);
  });

  it("stops notifying once the handler is unregistered", async () => {
    const handler = vi.fn();
    const unregister = setUnauthorizedHandler(handler);
    unregister();
    respondWith({ detail: "SceneWorks access token required" }, 401);

    await apiFetch("/api/v1/jobs", "stale").catch(() => {});

    expect(handler).not.toHaveBeenCalled();
  });

  it("does not let a stale unregister evict a newer handler", async () => {
    const first = vi.fn();
    const second = vi.fn();
    const unregisterFirst = setUnauthorizedHandler(first);
    setUnauthorizedHandler(second);
    // A late cleanup from the previous registration must be a no-op, not a hole in which
    // 401s go unhandled.
    unregisterFirst();
    respondWith({ detail: "SceneWorks access token required" }, 401);

    await apiFetch("/api/v1/jobs", "stale").catch(() => {});

    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });

  it("still throws the ApiError when the handler itself throws", async () => {
    setUnauthorizedHandler(() => {
      throw new Error("handler exploded");
    });
    respondWith({ detail: "SceneWorks access token required" }, 401);

    const err = await apiFetch("/api/v1/jobs", "stale").catch((caught) => caught);

    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(401);
    expect(err.message).toBe("SceneWorks access token required");
    // A throwing handler cannot be treated as having claimed the session.
    expect(err.reauthenticating).toBe(false);
  });
});
