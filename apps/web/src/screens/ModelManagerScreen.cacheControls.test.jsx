import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click } from "../testUtils/dom.js";

// Per-model resolved-cache controls on a Model Manager card (sc-19711, epic 19703).
//
// What these tests are actually defending:
//
// * The availability badge is the ONE shared resolver's typed judgement, RENDERED — never a second
//   opinion the screen derived from install paths or error text (the architecture bar for this
//   epic). So the assertions drive `modelAvailability` straight off the wire and check the badge
//   changes with it, including for a state this build doesn't know.
// * "Remove local copy" is preview-THEN-confirm against the real backend preview. A blocked
//   preview must never reach a confirm dialog, because the store would then refuse the removal the
//   dialog just promised.
// * Removing a local copy is not an uninstall. The card must say so, and must not be conflated
//   with the download/install job surface above it.
//
// Destructive actions route through the shared desktop-safe appConfirm (sc-12068) rather than
// window.confirm, which silently no-ops inside the Tauri WebView. Mocked so tests control the
// user's answer and can assert the guard fired at all.
const { appConfirmMock } = vi.hoisted(() => ({ appConfirmMock: vi.fn(async () => true) }));
vi.mock("../appConfirm.jsx", () => ({
  appConfirm: appConfirmMock,
  useConfirm: () => appConfirmMock,
  ConfirmHost: () => null,
}));

const GIB = 1024 * 1024 * 1024;

// A model installed on the external library, in the /models shape sc-19708 produces: the typed
// `modelAvailability` rides on the catalog row itself.
function externalModel(overrides = {}) {
  return {
    id: "flux_dev",
    name: "FLUX.1 [dev]",
    type: "image",
    family: "flux",
    installState: "installed",
    downloadable: true,
    modelAvailability: "external_ready",
    ui: { description: "External-library model." },
    ...overrides,
  };
}

// One resolved-cache entry as GET /api/v1/model-cache reports it, joined to its catalog model id
// by the BACKEND (the identity mapping has exactly one implementation, and it is not here).
function cacheEntry(overrides = {}) {
  return {
    cacheKey: "sha256:aaa",
    state: "complete",
    repository: "black-forest-labs/FLUX.1-dev",
    revision: "abc123",
    variant: "q8",
    tier: "q8",
    bytes: 12 * GIB,
    pinned: false,
    artifactPinned: false,
    modelPinOwners: [],
    createdAt: 1,
    lastUsedAt: 2,
    modelIds: ["flux_dev"],
    ...overrides,
  };
}

function cacheStatus(entries, overrides = {}) {
  const used = entries.reduce((sum, entry) => sum + entry.bytes, 0);
  return {
    policy: { enabled: true, maxBytes: 64 * GIB, inactivitySeconds: 14 * 24 * 60 * 60 },
    initialized: true,
    error: null,
    usedBytes: used,
    pinnedBytes: entries.filter((entry) => entry.pinned).reduce((sum, e) => sum + e.bytes, 0),
    reclaimableBytes: entries
      .filter((entry) => entry.state === "complete" && !entry.pinned)
      .reduce((sum, e) => sum + e.bytes, 0),
    entryCount: entries.length,
    sourceVolumeRelation: "different",
    sourceLibraryPath: "/Volumes/Models/huggingface/hub",
    entries,
    ...overrides,
  };
}

describe("ModelManagerScreen local model copies (sc-19711)", () => {
  let container;
  let root;
  let apiFetch;
  let cacheReads;
  let status;
  let preview;
  let removeOutcome;
  let pinCalls;
  let removeCalls;
  let ModelManagerScreen;
  let AppContext;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    delete window.__TAURI__;
    appConfirmMock.mockReset();
    appConfirmMock.mockImplementation(async () => true);
    cacheReads = 0;
    pinCalls = [];
    removeCalls = [];
    status = cacheStatus([cacheEntry()]);
    preview = {
      cacheKey: "sha256:aaa",
      state: "complete",
      reclaimableBytes: 12 * GIB,
      pins: { kind: "known", artifactPinned: false, owners: [] },
      sourceUnavailableWarning: null,
      blocked: null,
    };
    removeOutcome = {
      cacheKey: "sha256:aaa",
      reclaimedBytes: 12 * GIB,
      sourceUnavailableWarning: null,
    };
    apiFetch = vi.fn(async (path, _token, options) => {
      if (path === "/api/v1/host-capabilities") {
        return { platform: "macos", memoryKind: "unified", memoryGb: 64 };
      }
      if (path === "/api/v1/model-cache") {
        cacheReads += 1;
        if (status instanceof Error) throw status;
        return status;
      }
      if (path === "/api/v1/model-cache/removal-preview") {
        return preview;
      }
      if (path === "/api/v1/model-cache/remove") {
        removeCalls.push(JSON.parse(options.body));
        return removeOutcome;
      }
      if (path === "/api/v1/model-cache/pin") {
        pinCalls.push(JSON.parse(options.body));
        return cacheEntry({ pinned: true, artifactPinned: true });
      }
      return [];
    });
    vi.resetModules();
    vi.doMock("../api.js", () => ({
      apiFetch,
      isAbortError: () => false,
      API_BASE_URL: "",
      eventUrl: () => "",
    }));
    ({ AppContext } = await import("../context/AppContext.js"));
    ({ ModelManagerScreen } = await import("./ModelManagerScreen.jsx"));
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

  // Retained so a test can re-render the same screen under a changed backend without rebuilding
  // the whole app context by hand.
  let contextValue;

  async function render(models) {
    const value = {
      activeProject: null,
      jobs: [],
      loras: [],
      models,
      macCapabilities: { macGatingActive: false },
      presets: [],
      jobAction: () => {},
      setActiveView: () => {},
      deleteLora: () => {},
      deleteModel: () => {},
      deleteModelVariant: vi.fn(),
      createModelDownloadJob: vi.fn(),
      createModelConvertJob: () => {},
      createLoraImportJob: () => {},
      createModelImportJob: () => {},
    };
    contextValue = value;
    await act(async () => {
      root.render(
        <AppContext.Provider value={value}>
          <ModelManagerScreen />
        </AppContext.Provider>,
      );
    });
    // Flush the host-capabilities and model-cache effect promises.
    await act(async () => {});
    await act(async () => {});
  }

  // The keep/remove handlers await a promise chain (preview → confirm → mutate → re-read) before
  // the DOM settles, so flush several microtask turns before asserting.
  async function settle() {
    await act(async () => {
      for (let i = 0; i < 8; i += 1) await Promise.resolve();
    });
  }

  const section = () => container.querySelector(".model-local-copy");
  const badgeTexts = () =>
    [...container.querySelectorAll(".model-card-status .status-badge")].map((badge) =>
      badge.textContent.trim(),
    );
  const copyButton = (label) =>
    [...container.querySelectorAll(".model-local-copy-actions button")].find(
      (button) => button.textContent.trim() === label,
    );
  const liveText = () => container.querySelector('[role="status"]')?.textContent ?? "";

  // ---- typed availability badge -------------------------------------------------

  // Each typed state renders its own badge, driven entirely off the wire value. Enumerated so a
  // state silently losing its badge reds here.
  it.each([
    ["local_ready", "local copy"],
    ["external_ready", "on external library"],
    ["installed_external_unavailable", "library disconnected"],
    ["incomplete", "incomplete"],
    ["missing", "not installed"],
  ])("renders the %s availability judgement as its own badge", async (availability, label) => {
    await render([externalModel({ modelAvailability: availability })]);
    expect(badgeTexts()).toContain(label);
  });

  // A state this build doesn't recognize is LABELLED, not silently dropped and not coerced into a
  // reassuring one — the unsupported-artifact-class acceptance criterion in miniature.
  it("labels an unrecognized availability state rather than hiding it", async () => {
    await render([externalModel({ modelAvailability: "evicting" })]);
    expect(badgeTexts()).toContain("unknown state");
  });

  it("renders no availability badge for a row the resolver never judged", async () => {
    await render([externalModel({ modelAvailability: undefined })]);
    expect(badgeTexts()).not.toContain("on external library");
    expect(badgeTexts()).not.toContain("unknown state");
  });

  // ---- the local-copy block ------------------------------------------------------

  it("lists the joined cache entry with its tier, size and removable state", async () => {
    await render([externalModel()]);
    expect(apiFetch).toHaveBeenCalledWith("/api/v1/model-cache", expect.anything());
    const text = section().textContent;
    expect(text).toContain("Q8");
    expect(text).toContain("12.0 GiB");
    expect(text).toContain("Can be removed automatically when space is needed");
    // Removing a local copy is NOT an uninstall, and the card has to say so.
    expect(text).toContain("never uninstalls the model");
  });

  it("describes a pinned copy as kept and offers the inverse action", async () => {
    status = cacheStatus([cacheEntry({ pinned: true, artifactPinned: true })]);
    await render([externalModel()]);
    expect(section().textContent).toContain("Kept — never removed automatically");
    expect(copyButton("Allow automatic removal")).toBeTruthy();
    expect(copyButton("Keep locally")).toBeFalsy();
  });

  // An entry that is not `complete` is not usable, and pinning an unusable bundle is meaningless —
  // so the keep control is disabled while the state resolves. Removal stays available: clearing
  // residue is exactly what a user should be able to do here.
  it("disables Keep locally for an entry that is not ready, but still allows removal", async () => {
    status = cacheStatus([cacheEntry({ state: "materializing" })]);
    await render([externalModel()]);
    expect(section().textContent).toContain("Not ready to use (materializing)");
    expect(copyButton("Keep locally").disabled).toBe(true);
    expect(copyButton("Remove local copy").disabled).toBe(false);
  });

  it("offers the block with no entry yet for a model that could hold a local copy", async () => {
    status = cacheStatus([]);
    await render([externalModel()]);
    expect(section().textContent).toContain("No local copy yet");
    expect(copyButton("Remove local copy")).toBeFalsy();
  });

  // A model that can hold no local copy at all gets no block — an affordance that could never do
  // anything is worse than no affordance.
  it("renders no local-copy block for a model that cannot hold one", async () => {
    status = cacheStatus([]);
    await render([externalModel({ modelAvailability: "missing", installState: "missing" })]);
    expect(section()).toBeNull();
  });

  // A FAILED status read must render nothing rather than an empty-but-confident "no local copies":
  // the screen genuinely does not know, and saying "none" would be a fabricated answer.
  it("renders no local-copy block when the status read failed", async () => {
    status = new Error("cache listing failed");
    await render([externalModel()]);
    expect(cacheReads).toBe(1);
    expect(section()).toBeNull();
    expect(container.textContent).not.toContain("No local copy yet");
  });

  // Hiding the controls silently would leave the user unable to tell "no local copies" from "no
  // answer". The reason is stated ONCE for the whole screen, because one read failed for the whole
  // screen — not once per model card.
  it("explains once why the local-copy controls are hidden", async () => {
    status = new Error("cache listing failed");
    await render([
      externalModel(),
      externalModel({ id: "z", name: "Z-Image-Turbo", family: "z-image" }),
    ]);
    const notices = [...container.querySelectorAll(".inline-warning")].filter((node) =>
      node.textContent.includes("Local model copies can’t be shown"),
    );
    expect(notices).toHaveLength(1);
    expect(notices[0].textContent).toContain("cache listing failed");
    // It must not imply anything about what is cached — that is what could not be determined.
    expect(notices[0].textContent).not.toMatch(/no local copies|none|0 copies/i);
  });

  // The subtler half of the same defect: the request SUCCEEDS but the store could not be listed,
  // so the snapshot carries `error` and an empty `entries`. That empty list is "no answer", not
  // "no copies", and must not be rendered as the latter.
  it("renders no local-copy block when the snapshot reports the store could not be listed", async () => {
    status = cacheStatus([], { error: "resolved-cache journal is unreadable" });
    await render([externalModel()]);
    expect(section()).toBeNull();
    expect(container.textContent).not.toContain("No local copy yet");
    // Same explanation as an outright transport failure — from the user's side it is the same
    // situation: the answer is unavailable.
    expect(container.textContent).toContain("Local model copies can’t be shown");
    expect(container.textContent).toContain("resolved-cache journal is unreadable");
  });

  // A read that recovers must clear the standing explanation rather than leave a stale alarm.
  it("clears the explanation once a later read succeeds", async () => {
    status = new Error("cache listing failed");
    await render([externalModel()]);
    expect(container.textContent).toContain("Local model copies can’t be shown");
    status = cacheStatus([cacheEntry()]);
    // Any action re-reads status; use the one that needs no local-copy button to be present.
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ ...contextValue, models: [externalModel()] }}>
          <ModelManagerScreen key="remount" />
        </AppContext.Provider>,
      );
    });
    await settle();
    expect(container.textContent).not.toContain("Local model copies can’t be shown");
    expect(section()).toBeTruthy();
  });

  // ---- keep locally / allow automatic removal -------------------------------------

  it("pins through the API and re-reads authoritative status rather than flipping the row", async () => {
    await render([externalModel()]);
    const before = cacheReads;
    // The re-read returns the entry as PINNED, so an implementation that optimistically flipped
    // local state and skipped the round trip would still look right — hence asserting the read.
    status = cacheStatus([cacheEntry({ pinned: true, artifactPinned: true })]);
    await click(copyButton("Keep locally"));
    await settle();
    expect(pinCalls).toEqual([{ cacheKey: "sha256:aaa", pinned: true }]);
    expect(cacheReads).toBe(before + 1);
    expect(section().textContent).toContain("Kept — never removed automatically");
    expect(liveText()).toContain("kept until you allow automatic removal");
  });

  it("unpins with pinned:false and reports the copy as reclaimable again", async () => {
    status = cacheStatus([cacheEntry({ pinned: true, artifactPinned: true })]);
    await render([externalModel()]);
    status = cacheStatus([cacheEntry()]);
    await click(copyButton("Allow automatic removal"));
    await settle();
    expect(pinCalls).toEqual([{ cacheKey: "sha256:aaa", pinned: false }]);
    expect(liveText()).toContain("can now be removed automatically");
  });

  it("surfaces a pin failure instead of claiming the change landed", async () => {
    await render([externalModel()]);
    apiFetch.mockImplementation(async (path) => {
      if (path === "/api/v1/model-cache/pin") throw new Error("journal is locked");
      if (path === "/api/v1/model-cache") return status;
      if (path === "/api/v1/host-capabilities") return { platform: "macos", memoryGb: 64 };
      return [];
    });
    await click(copyButton("Keep locally"));
    await settle();
    expect(liveText()).toContain("journal is locked");
    expect(section().textContent).toContain("Can be removed automatically");
  });

  // ---- remove local copy -----------------------------------------------------------

  // The load-bearing ordering: preview FIRST, and the confirm message is built from that preview's
  // own measured bytes — not from the row's cached `bytes`, which can be stale.
  it("previews before confirming and builds the dialog from the preview's own numbers", async () => {
    preview = { ...preview, reclaimableBytes: 9 * GIB };
    await render([externalModel()]);
    await click(copyButton("Remove local copy"));
    await settle();
    expect(apiFetch).toHaveBeenCalledWith(
      "/api/v1/model-cache/removal-preview",
      expect.anything(),
      expect.objectContaining({ method: "POST" }),
    );
    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    const dialog = appConfirmMock.mock.calls[0][0];
    expect(dialog.title).toContain("FLUX.1 [dev]");
    expect(dialog.tone).toBe("danger");
    // 9.0 GiB is the PREVIEW's measurement; 12.0 GiB is the listing's. The dialog must use the
    // former, so this fails if the confirm copy were built from the row.
    expect(dialog.message).toContain("9.0 GiB");
    expect(dialog.message).not.toContain("12.0 GiB");
    expect(removeCalls).toEqual([{ cacheKey: "sha256:aaa" }]);
  });

  it("does not remove anything when the user declines the confirm", async () => {
    appConfirmMock.mockImplementation(async () => false);
    await render([externalModel()]);
    await click(copyButton("Remove local copy"));
    await settle();
    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    expect(removeCalls).toEqual([]);
  });

  // A blocked preview must NOT reach a confirm at all: offering a confirm the store would then
  // refuse is exactly the "UI claims before backend state is committed" failure this story bans.
  it("never offers a confirm for a removal the store would refuse", async () => {
    preview = { ...preview, blocked: "a runtime lease is active" };
    await render([externalModel()]);
    await click(copyButton("Remove local copy"));
    await settle();
    expect(appConfirmMock).not.toHaveBeenCalled();
    expect(removeCalls).toEqual([]);
    expect(liveText()).toContain("can't be removed right now");
    expect(liveText()).toContain("a runtime lease is active");
  });

  // "Can't determine" must survive all the way into the dialog the user actually reads.
  it("carries an unknown pin answer into the confirm as 'couldn't determine'", async () => {
    preview = { ...preview, pins: { kind: "unknown" } };
    await render([externalModel()]);
    await click(copyButton("Remove local copy"));
    await settle();
    const { message } = appConfirmMock.mock.calls[0][0];
    expect(message).toContain("couldn't determine whether this copy is being kept");
    expect(message).not.toMatch(/not pinned|isn't pinned/i);
  });

  // The acceptance criterion: removal must state when it leaves the model unusable until the
  // external library is reconnected — in the confirm, and again in the outcome.
  it("warns before and after removal when the external source is unreachable", async () => {
    preview = { ...preview, sourceUnavailableWarning: "/Volumes/Models is not mounted" };
    removeOutcome = { ...removeOutcome, sourceUnavailableWarning: "/Volumes/Models is not mounted" };
    await render([externalModel({ modelAvailability: "installed_external_unavailable" })]);
    await click(copyButton("Remove local copy"));
    await settle();
    expect(appConfirmMock.mock.calls[0][0].message).toContain(
      "becomes unusable until its library is reconnected",
    );
    expect(liveText()).toContain("stays unusable until its model library is reconnected");
  });

  it("re-reads status after a successful removal and reports the bytes freed", async () => {
    await render([externalModel()]);
    const before = cacheReads;
    status = cacheStatus([]);
    await click(copyButton("Remove local copy"));
    await settle();
    expect(cacheReads).toBe(before + 1);
    expect(liveText()).toContain("freed 12.0 GiB");
    expect(section().textContent).toContain("No local copy yet");
  });

  it("surfaces a removal failure instead of claiming the copy is gone", async () => {
    await render([externalModel()]);
    apiFetch.mockImplementation(async (path) => {
      if (path === "/api/v1/model-cache/removal-preview") return preview;
      if (path === "/api/v1/model-cache/remove") throw new Error("entry is leased");
      if (path === "/api/v1/model-cache") return status;
      if (path === "/api/v1/host-capabilities") return { platform: "macos", memoryGb: 64 };
      return [];
    });
    await click(copyButton("Remove local copy"));
    await settle();
    expect(liveText()).toContain("entry is leased");
    // The row is still there — nothing claimed it was removed.
    expect(section().textContent).toContain("12.0 GiB");
  });

  // Accessibility: the buttons sit far down a long card and their result lands in a status line at
  // the top of the screen, so that line has to be announced rather than only painted.
  it("announces outcomes in a polite live region", async () => {
    await render([externalModel()]);
    await click(copyButton("Keep locally"));
    await settle();
    const region = container.querySelector('[role="status"]');
    expect(region).toBeTruthy();
    expect(region.getAttribute("aria-live")).toBe("polite");
  });

  // The status read is one listing for the whole screen, on mount and after mutations only. A
  // per-row or timer-driven read would put back exactly the cost sc-19708 removed.
  it("reads cache status once for the whole screen regardless of model count", async () => {
    status = cacheStatus([cacheEntry(), cacheEntry({ cacheKey: "sha256:bbb", modelIds: ["z"] })]);
    await render([
      externalModel(),
      externalModel({ id: "z", name: "Z-Image-Turbo", family: "z-image" }),
      externalModel({ id: "y", name: "Yet Another", family: "flux" }),
    ]);
    expect(cacheReads).toBe(1);
  });
});
