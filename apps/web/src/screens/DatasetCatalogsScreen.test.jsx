import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConfirmHost } from "../appConfirm.jsx";
import { AppContext } from "../context/AppContext.js";
import { resetNavigationPreferenceQueueForTests } from "../uiPreferences.js";

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }));
vi.mock("../runtime.js", () => ({
  isDesktop: true,
  tauriInvoke: invoke,
}));

import { DatasetCatalogsScreen } from "./DatasetCatalogsScreen.jsx";

function response(payload, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(payload),
  });
}

function deferred() {
  let resolve;
  const promise = new Promise((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

function catalog(overrides = {}) {
  return {
    id: "cat-1",
    name: "Photos",
    path: "C:\\data\\photos.catalog",
    availability: "available",
    sourceConfig: {
      kind: "filesystem",
      paths: ["C:\\data\\photos"],
      options: { recursive: true },
    },
    analyzerVersions: { image_tagger: "v2@abc" },
    checkpoints: {},
    counts: {
      recordCount: 1200,
      candidateCount: 1300,
      processedCount: 700,
      acceptedCount: 650,
      rejectedCount: 50,
      errorCount: 2,
    },
    storage: {
      databaseBytes: 2048,
      manifestBytes: 512,
      artifactBytes: 4096,
      totalBytes: 6656,
    },
    processing: {
      state: "running",
      candidateCount: 1300,
      processedCount: 700,
      acceptedCount: 650,
      rejectedCount: 50,
      errorCount: 2,
      message: "Analyzing shard 7",
      updatedAt: "2026-07-26T12:00:00Z",
    },
    processingControl: {
      desiredState: "running",
      revision: 0,
      updatedAt: "2026-07-26T12:00:00Z",
    },
    ...overrides,
  };
}

async function flush() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe("DatasetCatalogsScreen", () => {
  let container;
  let root;
  let catalogs;
  let requests;

  beforeEach(async () => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    catalogs = [catalog()];
    requests = [];
    invoke.mockReset();
    invoke.mockResolvedValue(null);
    vi.stubGlobal("fetch", vi.fn((url, options = {}) => {
      const path = new URL(url).pathname;
      requests.push({ path, options, body: options.body ? JSON.parse(options.body) : null });
      if (path === "/api/v1/ui-preferences" && (!options.method || options.method === "GET")) {
        return response({ selectedCatalogId: "cat-1" });
      }
      if (path === "/api/v1/ui-preferences") return response({});
      if (path === "/api/v1/catalogs" && (!options.method || options.method === "GET")) {
        return response(catalogs);
      }
      if (path === "/api/v1/catalogs" && options.method === "POST") {
        const created = catalog({ id: "cat-created", name: "Created", path: JSON.parse(options.body).path });
        catalogs = [...catalogs, created];
        return response(created, 201);
      }
      if (path === "/api/v1/catalogs/attach") {
        const attached = catalog({ id: "cat-attached", name: "Attached" });
        catalogs = [...catalogs, attached];
        return response(attached);
      }
      if (path.endsWith("/status")) return response(catalogs[0]);
      if (path.endsWith("/pause")) {
        catalogs[0] = catalog({
          processingControl: {
            desiredState: "paused",
            revision: 1,
            updatedAt: "2026-07-26T12:01:00Z",
          },
        });
        return response(catalogs[0]);
      }
      if (path.endsWith("/resume")) {
        catalogs[0] = catalog({
          processing: { ...catalog().processing, state: "paused", message: "Paused by user" },
          processingControl: {
            desiredState: "running",
            revision: 2,
            updatedAt: "2026-07-26T12:02:00Z",
          },
        });
        return response(catalogs[0]);
      }
      if (options.method === "DELETE") {
        catalogs = [];
        return response({ detached: true, deletedOnDisk: path.endsWith("/on-disk") });
      }
      throw new Error(`Unexpected request: ${options.method ?? "GET"} ${path}`);
    }));
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ token: "token" }}>
          <DatasetCatalogsScreen />
          <ConfirmHost />
        </AppContext.Provider>,
      );
    });
    await flush();
  });

  afterEach(async () => {
    await act(async () => root?.unmount());
    await resetNavigationPreferenceQueueForTests();
    container.remove();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  async function remountWithFakeTimers() {
    await act(async () => root.unmount());
    vi.useFakeTimers();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ token: "token" }}>
          <DatasetCatalogsScreen />
          <ConfirmHost />
        </AppContext.Provider>,
      );
    });
    await flush();
  }

  it("loads persisted selection and presents real list, source, analyzer, progress, errors, and storage data", () => {
    expect(container.textContent).toContain("Photos");
    expect(container.textContent).toContain("1,200 rows");
    expect(container.textContent).toContain("700 analyzed");
    expect(container.textContent).toContain("image_tagger");
    expect(container.textContent).toContain("v2@abc");
    expect(container.textContent).toContain("Analyzing shard 7");
    expect(container.textContent).toContain("6.5 KiB");
    expect(container.querySelector("[role='progressbar']").getAttribute("aria-valuenow")).toBe("54");
  });

  it("offers an API-backed restart for failed schedulable Parquet processing", async () => {
    catalogs = [catalog({
      sourceConfig: {
        kind: "parquet",
        paths: ["C:\\data\\source.parquet"],
        options: {},
      },
      processing: {
        ...catalog().processing,
        state: "failed",
        message: "Catalog processing was interrupted; restart it to continue",
      },
    })];
    await act(async () => root.unmount());
    root = createRoot(container);
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ token: "token" }}>
          <DatasetCatalogsScreen />
          <ConfirmHost />
        </AppContext.Provider>,
      );
    });
    await flush();

    const restart = [...container.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Restart"));
    expect(restart).toBeTruthy();
    expect(restart.disabled).toBe(false);
    await act(async () => restart.click());
    await flush();
    expect(requests.find((item) => item.path.endsWith("/resume"))?.body).toEqual({
      expectedRevision: 0,
    });
  });

  it("creates and attaches catalogs from typed absolute browser-safe paths", async () => {
    const set = (input, value) => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(input, value);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    };
    const createForm = container.querySelector("form[aria-label='Create catalog']");
    await act(async () => {
      set(createForm.querySelector("input[placeholder='Product photography']"), "Created");
      set(createForm.querySelector("input[aria-label='Catalog folder']"), "C:\\catalogs\\created");
      set(createForm.querySelector("input[aria-label='Source folder']"), "C:\\sources\\photos");
    });
    await act(async () => createForm.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
    await flush();
    expect(requests.find((item) => item.path === "/api/v1/catalogs" && item.options.method === "POST")?.body).toEqual({
      name: "Created",
      path: "C:\\catalogs\\created",
      sourceConfig: { kind: "parquet", paths: ["C:\\sources\\photos"], options: {} },
    });
    expect(createForm.textContent).toContain("currently supports Parquet sources");
    expect(createForm.textContent).not.toContain("Image folder");

    const attachForm = container.querySelector("form[aria-label='Attach catalog']");
    await act(async () => set(attachForm.querySelector("input"), "C:\\catalogs\\existing"));
    await act(async () => attachForm.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
    await flush();
    expect(requests.find((item) => item.path === "/api/v1/catalogs/attach")?.body).toEqual({
      path: "C:\\catalogs\\existing",
    });
  });

  it("uses the native folder picker, and cancellation leaves typed input untouched", async () => {
    const input = container.querySelector("input[aria-label='Catalog folder']");
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(input, "C:\\typed");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => container.querySelector("button[aria-label='Choose catalog folder']").click());
    expect(invoke).toHaveBeenCalledWith("choose_folder");
    expect(input.value).toBe("C:\\typed");

    invoke.mockResolvedValueOnce("C:\\picked");
    await act(async () => container.querySelector("button[aria-label='Choose catalog folder']").click());
    await flush();
    expect(input.value).toBe("C:\\picked");
  });

  it("persists pause and resume through typed catalog-id endpoints", async () => {
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Pause")).click());
    await flush();
    expect(requests.some((item) => item.path === "/api/v1/catalogs/cat-1/pause")).toBe(true);
    expect(requests.find((item) => item.path === "/api/v1/catalogs/cat-1/pause").body).toEqual({
      expectedRevision: 0,
    });
    expect(container.textContent).toContain("Pause requested");
    catalogs[0] = catalog({
      processing: { ...catalog().processing, state: "paused", message: "Paused by user" },
      processingControl: {
        desiredState: "paused",
        revision: 1,
        updatedAt: "2026-07-26T12:02:00Z",
      },
    });
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Refresh")).click());
    await flush();
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Resume")).click());
    await flush();
    expect(requests.some((item) => item.path === "/api/v1/catalogs/cat-1/resume")).toBe(true);
    expect(requests.find((item) => item.path === "/api/v1/catalogs/cat-1/resume").body).toEqual({
      expectedRevision: 1,
    });
    expect(container.textContent).toContain("Resume requested");
  });

  it("keeps detach and permanent on-disk deletion as distinct desktop-safe confirmations", async () => {
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Detach").click());
    expect(document.body.textContent).toContain("every catalog file will remain on disk");
    await act(async () => [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Cancel").click());
    expect(requests.some((item) => item.options.method === "DELETE")).toBe(false);

    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Delete on disk").click());
    expect(document.body.textContent).toContain("This cannot be undone");
    await act(async () => [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Delete files permanently").click());
    await flush();
    expect(requests.some((item) => item.path === "/api/v1/catalogs/cat-1/on-disk" && item.options.method === "DELETE")).toBe(true);
  });

  it("shows actionable sanitized errors without echoing a private server path", async () => {
    fetch.mockImplementationOnce(() => response({ detail: "corrupt C:\\private\\secret\\catalog.json" }, 422));
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Refresh")).click());
    await flush();
    expect(container.querySelector("[role='alert']").textContent).toContain("invalid or damaged");
    expect(container.textContent).not.toContain("C:\\private\\secret");
  });

  it("aborts and ignores a delayed pre-pause status response", async () => {
    await remountWithFakeTimers();
    const stale = deferred();
    const original = fetch.getMockImplementation();
    let pollSignal;
    fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/status")) {
        pollSignal = options.signal;
        return stale.promise;
      }
      return original(url, options);
    });
    await act(async () => vi.advanceTimersByTime(3000));
    expect(pollSignal?.aborted).toBe(false);

    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Pause")).click());
    await flush();
    expect(pollSignal.aborted).toBe(true);
    expect(container.textContent).toContain("Pause requested");
    stale.resolve(response(catalog()));
    await flush();
    expect(container.textContent).toContain("Pause requested");
  });

  it("generation-guards a delayed status response after selection changes", async () => {
    catalogs = [
      catalog(),
      catalog({ id: "cat-2", name: "Illustrations", path: "C:\\data\\illustrations.catalog" }),
    ];
    await remountWithFakeTimers();
    const stale = deferred();
    const original = fetch.getMockImplementation();
    fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path === "/api/v1/catalogs/cat-1/status") return stale.promise;
      return original(url, options);
    });
    await act(async () => vi.advanceTimersByTime(3000));
    await act(async () => [...container.querySelectorAll(".catalog-list-item")].find((button) => button.textContent.includes("Illustrations")).click());
    stale.resolve(response(catalog({ name: "Stale Photos" })));
    await flush();
    expect(container.querySelector(".catalog-detail h2").textContent).toBe("Illustrations");
    expect(container.textContent).not.toContain("Stale Photos");
  });

  it("aborts delayed polling on unmount without committing its response", async () => {
    await remountWithFakeTimers();
    const delayed = deferred();
    const original = fetch.getMockImplementation();
    let signal;
    fetch.mockImplementation((url, options = {}) => {
      if (new URL(url).pathname.endsWith("/status")) {
        signal = options.signal;
        return delayed.promise;
      }
      return original(url, options);
    });
    await act(async () => vi.advanceTimersByTime(3000));
    await act(async () => root.unmount());
    root = null;
    expect(signal.aborted).toBe(true);
    delayed.resolve(response(catalog({ name: "Too late" })));
    await flush();
    expect(container.textContent).not.toContain("Too late");
  });
});
