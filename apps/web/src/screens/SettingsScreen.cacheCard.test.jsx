import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click } from "../testUtils/dom.js";

// Settings "Local model copies" card (sc-19711, epic 19703).
//
// The design fact these tests exist to hold true: the resolved-cache policy is DESKTOP-SHELL-OWNED
// AND RESTART-BOUND. `set_resolved_cache_policy` persists it in the shell; the three sidecar
// processes captured their own copy from the environment at spawn, so `GET /api/v1/model-cache`
// reports the policy the API is ACTUALLY RUNNING UNDER. The card therefore compares
// persisted-vs-effective and only then says "restart to apply".
//
// So these tests drive that real two-source state — a shell settings object and an independent API
// snapshot that can legitimately disagree — rather than mocking a `needsRestart` flag. A test that
// stubbed the comparison would pass with the whole mechanism deleted.
const { appConfirmMock } = vi.hoisted(() => ({ appConfirmMock: vi.fn(async () => true) }));
vi.mock("../appConfirm.jsx", () => ({
  appConfirm: appConfirmMock,
  useConfirm: () => appConfirmMock,
  ConfirmHost: () => null,
}));

const GIB = 1024 * 1024 * 1024;
const DAY = 24 * 60 * 60;

const APP_CONTEXT = {
  theme: "light",
  changeTheme: vi.fn(),
  visibleWorkers: [],
  macCapabilities: {},
};

function policy(overrides = {}) {
  return { enabled: true, maxBytes: 64 * GIB, inactivitySeconds: 14 * DAY, ...overrides };
}

function cacheStatus(overrides = {}) {
  return {
    policy: policy(),
    initialized: true,
    error: null,
    usedBytes: 20 * GIB,
    pinnedBytes: 4 * GIB,
    reclaimableBytes: 12 * GIB,
    entryCount: 3,
    sourceVolumeRelation: "different",
    sourceLibraryPath: "/Volumes/Models/huggingface/hub",
    entries: [],
    ...overrides,
  };
}

describe("SettingsScreen local model copies card (sc-19711)", () => {
  let container;
  let root;
  let invoke;
  let apiFetch;
  let cacheReads;
  let status;
  let shellSettings;
  let savedPolicies;
  let SettingsScreen;
  let AppContext;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    appConfirmMock.mockReset();
    appConfirmMock.mockImplementation(async () => true);
    cacheReads = 0;
    savedPolicies = [];
    status = cacheStatus();
    shellSettings = { dataDir: "/data", resolvedCache: policy() };
    invoke = vi.fn(async (command, args) => {
      switch (command) {
        case "get_app_settings":
          return shellSettings;
        case "get_gpu_info":
          return { platform: "macos", devices: [] };
        case "list_credentials":
          return [];
        case "get_remote_access":
          return null;
        case "set_resolved_cache_policy":
          savedPolicies.push(args.policy);
          // The shell echoes back the WHOLE settings object with the new persisted policy. The
          // running API policy in `status` is deliberately NOT touched — that divergence is the
          // real restart-bound state.
          shellSettings = { ...shellSettings, resolvedCache: args.policy };
          return shellSettings;
        default:
          return null;
      }
    });
    apiFetch = vi.fn(async (path) => {
      if (path === "/api/v1/model-cache") {
        cacheReads += 1;
        if (status instanceof Error) throw status;
        return status;
      }
      return [];
    });
    window.__TAURI__ = { core: { invoke } };
    vi.resetModules();
    vi.doMock("../api.js", () => ({
      apiFetch,
      isAbortError: () => false,
      API_BASE_URL: "",
      eventUrl: () => "",
    }));
    ({ SettingsScreen } = await import("./SettingsScreen.jsx"));
    ({ AppContext } = await import("../context/AppContext.js"));
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    delete window.__TAURI__;
    vi.doUnmock("../api.js");
    vi.restoreAllMocks();
    // Faked per-test by the convergence cases only; restored here either way.
    vi.useRealTimers();
  });

  async function render() {
    await act(async () => {
      root.render(
        <AppContext.Provider value={APP_CONTEXT}>
          <SettingsScreen />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
    await act(async () => {});
    const tab = [...container.querySelectorAll(".mode-tab")].find(
      (button) => button.textContent.trim() === "Settings",
    );
    expect(tab, 'tab "Settings"').toBeTruthy();
    await click(tab);
  }

  async function settle() {
    await act(async () => {
      for (let i = 0; i < 8; i += 1) await Promise.resolve();
    });
  }

  const toggle = () =>
    container.querySelector(
      '[aria-label="Keep a working copy of models from external libraries"]',
    );
  const limitInput = () => container.querySelector("#model-cache-limit");
  const daysInput = () => container.querySelector("#model-cache-days");
  const statusText = () => container.querySelector(".settings-status")?.textContent ?? "";

  // Commit paths fire on blur, so set the value the way React sees it and then blur.
  async function commitField(input, value) {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(input.constructor.prototype, "value")?.set;
      setter?.call(input, value);
      input.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    // React delegates `onBlur` off the BUBBLING `focusout` event, not the non-bubbling native
    // `blur`, so dispatching `blur` here would silently never reach the handler.
    await act(async () => {
      input.dispatchEvent(new window.FocusEvent("focusout", { bubbles: true }));
    });
    await settle();
  }

  // ---- storage readout -------------------------------------------------------------

  // The usage totals come from the resolver's own ledger, and all four have to be visible — the
  // acceptance criterion is that they match the ledger, which starts with showing them.
  it("renders the ledger's usage, count, reclaimable and kept bytes", async () => {
    await render();
    expect(apiFetch).toHaveBeenCalledWith("/api/v1/model-cache", expect.anything());
    const text = container.textContent;
    expect(text).toContain("Local model copies");
    expect(text).toContain("20.0 GiB of 64.0 GiB");
    expect(text).toContain("12.0 GiB"); // reclaimable
    expect(text).toContain("4.0 GiB"); // kept / pinned
    const inset = container.querySelector(".settings-inset");
    expect(inset.textContent).toContain("Copies");
    expect(inset.textContent).toContain("3");
  });

  // The status read is deliberately NOT polled: a journal listing is proportional to the number of
  // cached bundles, and a timer over it is exactly the per-row read cost sc-19708 removed. It is
  // read on mount, and again only after an action that could have changed it.
  it("reads status once on mount and again only after a commit", async () => {
    await render();
    expect(cacheReads).toBe(1);
    await settle();
    expect(cacheReads).toBe(1);
    await commitField(limitInput(), "128");
    expect(cacheReads).toBe(2);
  });

  // A refused input changes nothing, so it must not cost a re-read either.
  it("does not re-read status when a commit is refused", async () => {
    await render();
    await commitField(limitInput(), "0");
    expect(cacheReads).toBe(1);
  });

  it("seeds the editable fields from the SHELL-persisted policy, not the running one", async () => {
    // Persisted 32 GiB / 7 days; the API is still running the old 64 GiB / 14 days.
    shellSettings = { dataDir: "/data", resolvedCache: policy({ maxBytes: 32 * GIB, inactivitySeconds: 7 * DAY }) };
    await render();
    expect(limitInput().value).toBe("32");
    expect(daysInput().value).toBe("7");
  });

  // A failed read must be reported, never rendered as an empty, reassuring cache.
  it("reports a failed status read instead of showing an empty cache", async () => {
    status = new Error("model-cache endpoint unavailable");
    await render();
    expect(container.textContent).toContain("Couldn’t read local copy storage");
    expect(container.textContent).toContain("model-cache endpoint unavailable");
    // Scoped to the cache readout: ".settings-inset" alone also matches unrelated Settings insets.
    expect(container.textContent).not.toContain("Can be reclaimed");
  });

  // The store exists but could not be listed: the API answers 200 with an `error` field. Same
  // rule — say "couldn't read", don't render a confident zero.
  it("reports a snapshot that says the store could not be listed", async () => {
    status = cacheStatus({ error: "resolved-cache journal is unreadable", usedBytes: 0, entryCount: 0 });
    await render();
    expect(container.textContent).toContain("Couldn’t read local copy storage");
    expect(container.textContent).toContain("resolved-cache journal is unreadable");
    // Scoped to the cache readout: ".settings-inset" alone also matches unrelated Settings insets.
    expect(container.textContent).not.toContain("Can be reclaimed");
  });

  // The configured external library location. Every other number in this card is relative to it,
  // and it is what the same-volume warning below tells the user to move — so it has to be on
  // screen. The card must show the API's own `sourceLibraryPath`, because that is the exact path
  // the status read compared volumes against; anything re-derived here could name a location the
  // comparison never used.
  it("shows the configured external library location", async () => {
    await render();
    const inset = container.querySelector(".settings-inset");
    expect(inset.textContent).toContain("Library");
    expect(inset.textContent).toContain("/Volumes/Models/huggingface/hub");
  });

  it("shows whatever library location the API reports, not a fixed one", async () => {
    status = cacheStatus({ sourceLibraryPath: "/mnt/nas/hf/hub" });
    await render();
    expect(container.querySelector(".settings-inset").textContent).toContain("/mnt/nas/hf/hub");
  });

  // Same-volume is not an error, but it silently removes the entire point of the feature.
  it("states plainly that a same-volume configuration buys nothing", async () => {
    status = cacheStatus({ sourceVolumeRelation: "same" });
    await render();
    const notice = container.querySelector(".settings-notice");
    expect(notice.textContent).toContain("same disk as the model library");
    expect(notice.textContent).toContain("neither a disconnect nor a slow drive");
    // "Point the model library at a different volume" is not actionable to someone who cannot see
    // where it currently points, so the warning names the configured location.
    expect(notice.textContent).toContain("/Volumes/Models/huggingface/hub");
  });

  it("shows no same-volume warning when the volumes differ", async () => {
    await render();
    expect(container.textContent).not.toContain("same disk as the model library");
  });

  // ---- restart-bound policy --------------------------------------------------------

  // Persisted and running agree: there is nothing to restart for, and claiming otherwise would be
  // a standing false alarm.
  it("claims no restart while the persisted policy matches the running one", async () => {
    await render();
    expect(container.textContent).not.toContain("restart to apply");
  });

  // The real divergence: the shell already persisted 32 GiB, the API is still running 64 GiB.
  it("says restart-to-apply when the persisted policy differs from the running one", async () => {
    shellSettings = { dataDir: "/data", resolvedCache: policy({ maxBytes: 32 * GIB }) };
    await render();
    expect(container.textContent).toContain("still running under the previous setting");
    expect(container.textContent).toContain("restart to apply");
  });

  // THE round-trip, end to end and with no stubbed comparison: commit a change → the shell
  // persists it → the API snapshot still reports the OLD running policy → the notice appears
  // because the two genuinely disagree now. Deleting the persisted-vs-effective comparison makes
  // this red.
  it("commits a change and then honestly reports it is not yet in effect", async () => {
    await render();
    expect(container.textContent).not.toContain("restart to apply");

    await commitField(limitInput(), "32");

    expect(savedPolicies).toEqual([
      { enabled: true, maxBytes: 32 * GIB, inactivitySeconds: 14 * DAY },
    ]);
    expect(statusText()).toContain("32.0 GiB");
    expect(statusText()).toContain("after a restart");
    // The API still reports the 64 GiB policy it spawned with, so the card must now say so.
    expect(container.textContent).toContain("still running under the previous setting");
    // And the readout keeps showing the RUNNING limit, not the aspirational one.
    expect(container.textContent).toContain("20.0 GiB of 64.0 GiB");
  });

  // Every commit sends a COMPLETE policy, so a partial write can never leave the store running
  // under a half-applied rule.
  it("sends a complete policy on every commit path", async () => {
    await render();
    await commitField(daysInput(), "30");
    expect(savedPolicies).toEqual([
      { enabled: true, maxBytes: 64 * GIB, inactivitySeconds: 30 * DAY },
    ]);
    expect(statusText()).toContain("30 unused days");
    expect(statusText()).toContain("after a restart");
  });

  // ---- enable / disable -------------------------------------------------------------

  // Turning the cache off is not destructive and does not sweep. Say what actually happens to the
  // copies already on disk BEFORE committing.
  it("explains that disabling keeps existing copies, then persists enabled:false", async () => {
    await render();
    await click(toggle());
    await settle();
    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    const dialog = appConfirmMock.mock.calls[0][0];
    expect(dialog.message).toContain("stops making new local copies");
    expect(dialog.message).toContain("stay on disk");
    expect(dialog.message).toContain("3 existing copies");
    expect(dialog.message).not.toMatch(/delet/i);
    expect(savedPolicies).toEqual([
      { enabled: false, maxBytes: 64 * GIB, inactivitySeconds: 14 * DAY },
    ]);
    expect(statusText()).toContain("Existing copies stay on disk");
  });

  it("persists nothing when the disable confirm is declined", async () => {
    appConfirmMock.mockImplementation(async () => false);
    await render();
    await click(toggle());
    await settle();
    expect(savedPolicies).toEqual([]);
    expect(toggle().getAttribute("aria-checked")).toBe("true");
  });

  // Enabling is not destructive and needs no consequence dialog.
  it("enables without a confirm", async () => {
    shellSettings = { dataDir: "/data", resolvedCache: policy({ enabled: false }) };
    status = cacheStatus({ policy: policy({ enabled: false }) });
    await render();
    expect(toggle().getAttribute("aria-checked")).toBe("false");
    await click(toggle());
    await settle();
    expect(appConfirmMock).not.toHaveBeenCalled();
    expect(savedPolicies).toEqual([
      { enabled: true, maxBytes: 64 * GIB, inactivitySeconds: 14 * DAY },
    ]);
  });

  // ---- limit changes ------------------------------------------------------------------

  // Lowering below current usage schedules FUTURE cleanup; nothing is swept at the moment of the
  // change, and the confirm has to be bounded by what is actually reclaimable.
  it("spells out the reclaim before lowering the limit below current usage", async () => {
    await render();
    await commitField(limitInput(), "12");
    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    const { message } = appConfirmMock.mock.calls[0][0];
    expect(message).toContain("next automatic cleanup");
    expect(message).toContain("up to 8.0 GiB");
    expect(savedPolicies).toEqual([
      { enabled: true, maxBytes: 12 * GIB, inactivitySeconds: 14 * DAY },
    ]);
  });

  it("reverts the field and persists nothing when the lower-limit confirm is declined", async () => {
    appConfirmMock.mockImplementation(async () => false);
    await render();
    await commitField(limitInput(), "12");
    expect(savedPolicies).toEqual([]);
    expect(limitInput().value).toBe("64");
  });

  // Raising the limit has no consequence to warn about.
  it("raises the limit without a confirm", async () => {
    await render();
    await commitField(limitInput(), "128");
    expect(appConfirmMock).not.toHaveBeenCalled();
    expect(savedPolicies).toEqual([
      { enabled: true, maxBytes: 128 * GIB, inactivitySeconds: 14 * DAY },
    ]);
  });

  it("commits nothing when the value is unchanged", async () => {
    await render();
    await commitField(limitInput(), "64");
    expect(savedPolicies).toEqual([]);
    expect(appConfirmMock).not.toHaveBeenCalled();
  });

  // A user-typed zero / blank / garbage must be refused and the field restored, never written to
  // the wire as 0 or NaN.
  it.each([["0"], [""], ["-5"], ["abc"]])(
    "refuses the unusable limit %j and restores the persisted value",
    async (value) => {
      await render();
      await commitField(limitInput(), value);
      expect(savedPolicies).toEqual([]);
      expect(statusText()).toContain("at least 1 GiB");
      expect(limitInput().value).toBe("64");
    },
  );

  it.each([["0"], [""], ["abc"]])(
    "refuses the unusable interval %j and restores the persisted value",
    async (value) => {
      await render();
      await commitField(daysInput(), value);
      expect(savedPolicies).toEqual([]);
      expect(statusText()).toContain("at least one day");
      expect(daysInput().value).toBe("14");
    },
  );

  // ---- accessibility -------------------------------------------------------------------

  it("exposes the toggle as a labelled switch reflecting the persisted state", async () => {
    await render();
    expect(toggle().getAttribute("role")).toBe("switch");
    expect(toggle().getAttribute("aria-checked")).toBe("true");
    expect(toggle().getAttribute("aria-label")).toBe(
      "Keep a working copy of models from external libraries",
    );
  });

  it("announces the commit status in a polite live region", async () => {
    await render();
    await commitField(limitInput(), "128");
    const region = container.querySelector(".settings-status");
    expect(region.getAttribute("role")).toBe("status");
    expect(region.getAttribute("aria-live")).toBe("polite");
  });

  // ---- convergence while the store is working --------------------------------------
  //
  // Storage totals move as a promotion runs. Freezing them at whatever the screen opened on would
  // show a user a number that is already wrong, so the card re-reads while (and only while) the
  // store still has an entry in flight.

  async function tick(times = 1) {
    for (let i = 0; i < times; i += 1) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(3000);
      });
      await settle();
    }
  }

  it("re-reads the storage totals while a copy is materializing, then stops", async () => {
    vi.useFakeTimers();
    status = cacheStatus({
      entries: [{ cacheKey: "sha256:aaa", state: "materializing", bytes: 0, pinned: false }],
      usedBytes: 0,
      entryCount: 1,
    });
    await render();
    expect(cacheReads).toBe(1);
    expect(container.querySelector(".settings-inset").textContent).toContain("0 B of 64.0 GiB");

    status = cacheStatus({
      entries: [{ cacheKey: "sha256:aaa", state: "complete", bytes: 12 * GIB, pinned: false }],
      usedBytes: 12 * GIB,
      entryCount: 1,
    });
    await tick();
    expect(cacheReads).toBe(2);
    expect(container.querySelector(".settings-inset").textContent).toContain("12.0 GiB of");

    // Terminal: the timer must be gone, not merely quiet.
    await tick(4);
    expect(cacheReads).toBe(2);
  });

  it("never arms a refresh for a settled cache", async () => {
    vi.useFakeTimers();
    await render();
    await tick(5);
    expect(cacheReads).toBe(1);
  });
});

// The controls are shell-backed and therefore desktop-only, but the usage readout comes from the
// API and must still render for a remote admin — they need to see what the host is holding even
// though they cannot change it from there.
describe("SettingsScreen local model copies card in server (REST) mode", () => {
  let container;
  let root;
  let apiFetch;
  let SettingsScreen;
  let AppContext;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    delete window.__TAURI__;
    appConfirmMock.mockReset();
    appConfirmMock.mockImplementation(async () => true);
    apiFetch = vi.fn(async (path) => {
      if (path === "/api/v1/model-cache") return cacheStatus();
      return [];
    });
    vi.resetModules();
    vi.doMock("../api.js", () => ({
      apiFetch,
      isAbortError: () => false,
      API_BASE_URL: "",
      eventUrl: () => "",
    }));
    ({ SettingsScreen } = await import("./SettingsScreen.jsx"));
    ({ AppContext } = await import("../context/AppContext.js"));
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    vi.doUnmock("../api.js");
    vi.restoreAllMocks();
  });

  async function render() {
    await act(async () => {
      root.render(
        <AppContext.Provider value={APP_CONTEXT}>
          <SettingsScreen />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
    await act(async () => {});
    const tab = [...container.querySelectorAll(".mode-tab")].find(
      (button) => button.textContent.trim() === "Settings",
    );
    await click(tab);
  }

  it("shows the storage readout but no editable controls", async () => {
    await render();
    expect(container.textContent).toContain("Local model copies");
    expect(container.textContent).toContain("20.0 GiB of 64.0 GiB");
    expect(container.querySelector("#model-cache-limit")).toBeNull();
    expect(container.querySelector("#model-cache-days")).toBeNull();
    expect(container.textContent).toContain(
      "These settings are configured on the machine running SceneWorks",
    );
  });

  it("disables the toggle so a remote browser cannot call a missing Tauri bridge", async () => {
    await render();
    const toggle = container.querySelector(
      '[aria-label="Keep a working copy of models from external libraries"]',
    );
    expect(toggle.disabled).toBe(true);
  });

  // With no shell to ask, the running policy IS the persisted one, so there is nothing pending and
  // no restart to claim.
  it("never claims a pending restart when there is no shell to persist a different policy", async () => {
    await render();
    expect(container.textContent).not.toContain("restart to apply");
  });
});
