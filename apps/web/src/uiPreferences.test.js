import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The seam under test only decides WHAT goes on the wire, so the transport is mocked.
vi.mock("./api.js", () => ({ apiFetch: vi.fn(() => Promise.resolve({})) }));

import { apiFetch } from "./api.js";
import { ACCESS_TOKEN_KEY } from "./accessToken.js";
import { putUiPreferences } from "./uiPreferences.js";

beforeEach(() => {
  window.localStorage.clear();
  apiFetch.mockClear();
  apiFetch.mockImplementation(() => Promise.resolve({}));
});

afterEach(() => {
  window.localStorage.clear();
});

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
