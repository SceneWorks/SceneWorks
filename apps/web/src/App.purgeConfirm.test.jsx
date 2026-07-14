// sc-12068: App.purgeAsset's trash-unavailable → permanent-delete fallback confirms through
// the shared desktop-safe appConfirm dialog rather than the raw window.confirm, which silently
// no-ops in the Tauri WebView. This drives the REAL App callback end-to-end (Library Trashcan →
// select a discarded asset → Purge) against a purge endpoint that reports `trash_unavailable`,
// and asserts the guard fired via appConfirm and gated the permanent delete.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";

const { appConfirmMock } = vi.hoisted(() => ({ appConfirmMock: vi.fn(async () => true) }));
vi.mock("./appConfirm.jsx", () => ({
  appConfirm: appConfirmMock,
  useConfirm: () => appConfirmMock,
  ConfirmHost: () => null,
}));

const TRASHED_ASSET = {
  id: "asset-trash",
  projectId: "project-default",
  type: "image",
  displayName: "Discarded Frame",
  status: { favorite: false, rating: 0, rejected: false, trashed: true },
  recipe: { prompt: "discarded" },
};

describe("App purgeAsset trash-unavailable confirm (sc-12068)", () => {
  let container;
  let root;
  let purgeRequests;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    appConfirmMock.mockClear();
    appConfirmMock.mockResolvedValue(true);
    purgeRequests = [];
    global.fetch = vi.fn((url, options = {}) => {
      const parsed = new URL(url);
      const path = parsed.pathname;
      const method = options.method ?? "GET";
      if (path.endsWith("/health")) return Promise.resolve(response({ status: "ok", authRequired: false }));
      if (path.endsWith("/access")) return Promise.resolve(response({ authRequired: false }));
      if (path.endsWith("/jobs/events/ticket")) return Promise.resolve(response({ ticket: "stream-ticket" }));
      if (path.endsWith("/projects") && method === "GET") {
        return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
      }
      if (path.endsWith("/assets") && method === "GET") {
        return Promise.resolve(response([TRASHED_ASSET]));
      }
      if (path.endsWith("/purge") && method === "DELETE") {
        const permanent = parsed.searchParams.get("permanent") === "true";
        purgeRequests.push({ permanent });
        return Promise.resolve(response({ status: permanent ? "purged" : "trash_unavailable" }));
      }
      return Promise.resolve(response([]));
    });
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  async function openTrashcanPurge() {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    // The default view is the Library/Assets tab; switch to the Trashcan collection.
    const trashcanButton = [...document.body.querySelectorAll("button")].find((b) => b.textContent === "Trashcan");
    expect(trashcanButton).toBeTruthy();
    await act(async () => {
      trashcanButton.click();
    });
    await settle();

    // Select the discarded asset so its detail panel (with the Purge control) renders.
    const tile = [...document.body.querySelectorAll(".asset-tile")].find((t) => t.textContent.includes("Discarded Frame"));
    expect(tile).toBeTruthy();
    await act(async () => {
      tile.click();
    });
    await settle();

    const purgeButton = [...document.body.querySelectorAll("button")].find((b) => b.textContent === "Purge");
    expect(purgeButton).toBeTruthy();
    await act(async () => {
      purgeButton.click();
    });
    await settle();
  }

  it("confirms via appConfirm (danger) then issues the permanent purge when accepted", async () => {
    await openTrashcanPurge();

    expect(appConfirmMock).toHaveBeenCalledWith(expect.objectContaining({ tone: "danger" }));
    // First the trash attempt (trash_unavailable), then — after the confirm — the permanent delete.
    expect(purgeRequests).toContainEqual({ permanent: true });
  });

  it("does not permanently purge when the confirm is declined", async () => {
    appConfirmMock.mockResolvedValue(false);
    await openTrashcanPurge();

    expect(appConfirmMock).toHaveBeenCalledWith(expect.objectContaining({ tone: "danger" }));
    // Only the initial trash attempt was made — the permanent retry was gated off.
    expect(purgeRequests).toEqual([{ permanent: false }]);
  });
});
