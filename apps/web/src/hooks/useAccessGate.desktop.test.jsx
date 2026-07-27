// sc-15105: the desktop half of the 401 re-prompt. The desktop shell reaches its own API
// over trusted loopback and is authenticated unconditionally (epic 4484), so it must never
// be prompted for a password — it doesn't even register a 401 reaction. That requires
// `isDesktop: true` at module scope, which the shared appGateHooks.test.jsx pins to false
// for every test in the file, so this case lives in its own file rather than fighting a
// mid-suite remock.
import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useAccessGate } from "./useAccessGate.js";

const apiResponders = new Map();
let unauthorizedHandler = null;
vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn((path) => Promise.resolve(apiResponders.get(path) ?? {})),
    setMediaTicket: () => {},
    setUnauthorizedHandler: (handler) => {
      unauthorizedHandler = typeof handler === "function" ? handler : null;
    },
  };
});

vi.mock("../runtime.js", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, isDesktop: true };
});

async function settle() {
  await act(async () => {
    for (let i = 0; i < 6; i += 1) {
      await Promise.resolve();
    }
  });
}

describe("useAccessGate on the desktop shell (sc-15105)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    apiResponders.clear();
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
    function Harness() {
      latest = useAccessGate({
        setError: () => {},
        pushNotice: () => {},
        dismissNoticeKind: () => {},
      });
      return null;
    }
    root = createRoot(container);
    act(() => root.render(<Harness />));
    return { get: () => latest };
  }

  it("registers no 401 handler, even against a host that requires a password", async () => {
    // The harshest shape: a stored token AND an auth-requiring host. The remote build
    // registers a handler here (pinned in appGateHooks.test.jsx); the desktop build must
    // not, so nothing on this shell can ever be demoted to the password prompt.
    window.localStorage.setItem("sceneworks-token", "some-token");
    apiResponders.set("/api/v1/access", { authRequired: true });
    const { get } = mount();
    await settle();

    expect(unauthorizedHandler).toBeNull();
    expect(get().authenticated).toBe(true);
    expect(get().gateStatus).toBe("open");
  });

  // sc-15165: the cross-tab subscription is the one effect in the hook with NO
  // isDesktopShell guard, on the grounds that `authenticated` and `gateStatus` both
  // short-circuit on it before they read `tokenStatus`. That reasoning is only as good as a
  // test — a desktop shell that locked itself because a browser tab on the same origin hit
  // "forget" would be an unrecoverable regression (there is no password to type back in).
  it("never gates the desktop shell when another tab clears the token", async () => {
    window.localStorage.setItem("sceneworks-token", "some-token");
    apiResponders.set("/api/v1/access", { authRequired: true });
    const { get } = mount();
    await settle();

    act(() => {
      window.localStorage.removeItem("sceneworks-token");
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: "sceneworks-token",
          newValue: null,
          storageArea: window.localStorage,
        }),
      );
    });
    await settle();

    // The token state does follow storage (one live source), but loopback trust is what
    // authenticates here, so the shell stays open and usable.
    expect(get().token).toBe("");
    expect(get().authenticated).toBe(true);
    expect(get().gateStatus).toBe("open");
  });
});
