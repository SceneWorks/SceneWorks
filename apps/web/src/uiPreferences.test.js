import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The seam under test only decides WHAT goes on the wire, so the transport is mocked.
vi.mock("./api.js", () => ({ apiFetch: vi.fn(() => Promise.resolve({})) }));

import { apiFetch } from "./api.js";
import { ACCESS_TOKEN_KEY } from "./accessToken.js";
import {
  persistNavigationPreferences,
  putUiPreferences,
  resetNavigationPreferenceQueueForTests,
} from "./uiPreferences.js";

beforeEach(() => {
  window.localStorage.clear();
  apiFetch.mockClear();
  apiFetch.mockImplementation(() => Promise.resolve({}));
});

afterEach(async () => {
  await resetNavigationPreferenceQueueForTests();
  window.localStorage.clear();
});

function deferred() {
  let resolve;
  const promise = new Promise((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

// sc-15136. `/api/v1/ui-preferences` is method-asymmetric on the server: the GET is public (the
// pre-auth theme read) but the PUT writes to disk and is GATED (sc-8869, F-067). Six call sites
// each sent the PUT with "" — so on remote-auth deployments theme, accent, the Simple-UI default,
// the default quality, and the tier sticky never reached `ui-preferences.json`, invisibly, because
// every one of them also writes a localStorage cache.
//
// How that survived: four of the six had NO test on the token at all, and the two that did
// (`lastTierStore.test.js`, `SettingsScreen.test.jsx`) asserted the `""` default — a green that
// says nothing, since it passes with the fix reverted. Every token assertion here pins a
// NON-EMPTY value for that reason.
describe("putUiPreferences", () => {
  it("sends the stored access token so the GATED PUT is authenticated (remote auth)", () => {
    window.localStorage.setItem(ACCESS_TOKEN_KEY, "lan-password-1");

    putUiPreferences({ theme: "dark" });

    expect(apiFetch).toHaveBeenCalledWith(
      "/api/v1/ui-preferences",
      "lan-password-1",
      expect.objectContaining({ method: "PUT", body: JSON.stringify({ theme: "dark" }) }),
    );
  });

  it("reads the token at CALL time, not at import time", () => {
    // The token is promoted mid-session when the user answers the access gate, and the module is
    // imported long before that. Caching it at import would silently un-authenticate every write
    // made in the session that mattered.
    putUiPreferences({ theme: "dark" });
    expect(apiFetch.mock.calls[0][1]).toBe("");

    window.localStorage.setItem(ACCESS_TOKEN_KEY, "promoted-later");
    putUiPreferences({ theme: "light" });
    expect(apiFetch.mock.calls[1][1]).toBe("promoted-later");
  });

  it("sends an empty token on desktop/loopback, where no token is stored", () => {
    // `SCENEWORKS_TRUST_LOOPBACK` bypasses the token check for loopback peers before any
    // comparison, so the desktop shell must keep working with no credential to send.
    putUiPreferences({ accent: "violet" });
    expect(apiFetch.mock.calls[0][1]).toBe("");
  });

  it("sends only the caller's field, so a partial write cannot clobber the others", () => {
    // The endpoint MERGES; every call site relies on that to own one field.
    putUiPreferences({ defaultGenerationQuality: "q4" });
    expect(JSON.parse(apiFetch.mock.calls[0][2].body)).toEqual({ defaultGenerationQuality: "q4" });
  });

  it("returns the request promise so a caller can await it and observe failure", async () => {
    // SettingsScreen awaits this to roll back its optimistic cache and surface an error.
    apiFetch.mockImplementation(() => Promise.reject(new Error("401 Unauthorized")));
    await expect(putUiPreferences({ theme: "dark" })).rejects.toThrow("401 Unauthorized");
  });
});

describe("persistNavigationPreferences", () => {
  it("waits with bounded exponential backoff before retrying transient navigation writes and recovers", async () => {
    vi.useFakeTimers();
    apiFetch
      .mockRejectedValueOnce(new Error("network unavailable"))
      .mockRejectedValueOnce(Object.assign(new Error("throttled"), { status: 429 }))
      .mockResolvedValueOnce({});

    try {
      const persisted = persistNavigationPreferences({ activeView: "Library" });
      await vi.advanceTimersByTimeAsync(0);
      expect(apiFetch).toHaveBeenCalledTimes(1);

      await vi.advanceTimersByTimeAsync(99);
      expect(apiFetch).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(1);
      expect(apiFetch).toHaveBeenCalledTimes(2);

      await vi.advanceTimersByTimeAsync(199);
      expect(apiFetch).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(1);
      await persisted;

      expect(apiFetch).toHaveBeenCalledTimes(3);
    } finally {
      await vi.runAllTimersAsync();
      vi.useRealTimers();
    }
  });

  it("does not retry a permanent 4xx navigation write", async () => {
    vi.useFakeTimers();
    try {
      apiFetch.mockRejectedValueOnce(Object.assign(new Error("HTTP 400"), { status: 400 }));

      const persisted = persistNavigationPreferences({ activeView: "no-retry-400" });
      await vi.advanceTimersByTimeAsync(1000);
      await persisted;
      expect(apiFetch).toHaveBeenCalledTimes(1);
    } finally {
      await vi.runAllTimersAsync();
      vi.useRealTimers();
    }
  });

  it("does not retry an aborted navigation write", async () => {
    apiFetch.mockRejectedValueOnce(Object.assign(new Error("aborted"), { name: "AbortError" }));

    await persistNavigationPreferences({ activeView: "Library" });

    expect(apiFetch).toHaveBeenCalledTimes(1);
  });

  it("serializes writes, coalesces the latest intent, and reads the token at request time", async () => {
    const first = deferred();
    const second = deferred();
    apiFetch
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);
    window.localStorage.setItem(ACCESS_TOKEN_KEY, "lan-password-1");

    const all = persistNavigationPreferences({ activeView: "Library" });
    await vi.waitFor(() => expect(apiFetch).toHaveBeenCalledTimes(1));
    expect(apiFetch.mock.calls[0][1]).toBe("lan-password-1");
    expect(JSON.parse(apiFetch.mock.calls[0][2].body)).toEqual({ activeView: "Library" });

    persistNavigationPreferences({ activeView: "DatasetCatalogs" });
    persistNavigationPreferences({ selectedCatalogId: "older" });
    persistNavigationPreferences({ selectedCatalogId: "newest" });
    expect(apiFetch).toHaveBeenCalledTimes(1);

    window.localStorage.setItem(ACCESS_TOKEN_KEY, "rotated-password");
    first.resolve({});
    await vi.waitFor(() => expect(apiFetch).toHaveBeenCalledTimes(2));
    expect(apiFetch.mock.calls[1][1]).toBe("rotated-password");
    expect(JSON.parse(apiFetch.mock.calls[1][2].body)).toEqual({
      activeView: "DatasetCatalogs",
      selectedCatalogId: "newest",
    });

    second.resolve({});
    await all;
  });
});
