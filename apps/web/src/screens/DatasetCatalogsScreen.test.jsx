import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConfirmHost } from "../appConfirm.jsx";
import { AppContext } from "../context/AppContext.js";
import { ScreenActiveContext } from "../context/ScreenActiveContext.js";
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
    analyzerConfig: {
      revision: 0,
      updatedAt: "2026-07-26T12:00:00Z",
      settings: {
        structuredAnalysisEnabled: true,
        visionAnalysisEnabled: false,
        semanticEmbeddingsEnabled: false,
        thresholds: {
          personMinConfidence: 0.25,
          faceMinConfidence: 0.65,
          poseMinKeypointConfidence: 0.3,
          prominentFrameFraction: 0.2,
          frameEdgeMargin: 0.01,
          minPoseCoverage: 0.72,
        },
      },
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
      if (path === "/api/v1/projects") {
        return response([{ id: "project-1", name: "Portrait project" }]);
      }
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
      if (path.endsWith("/analyzer-config") && options.method === "PUT") {
        catalogs[0] = catalog({
          analyzerConfig: {
            revision: 1,
            updatedAt: "2026-07-26T12:03:00Z",
            settings: JSON.parse(options.body).settings,
          },
        });
        return response(catalogs[0]);
      }
      if (path.endsWith("/analyze") && options.method === "POST") {
        return response({
          id: "job-catalog-analysis",
          type: "catalog_analysis",
          status: "queued",
          payload: JSON.parse(options.body),
        }, 201);
      }
      if (path.endsWith("/saved-views") && (!options.method || options.method === "GET")) {
        return response([]);
      }
      if (path.endsWith("/saved-views") && options.method === "POST") {
        return response({
          id: "view-1",
          name: JSON.parse(options.body).name,
          query: JSON.parse(options.body).query,
          revision: 0,
          createdAt: "2026-07-26T12:00:00Z",
          updatedAt: "2026-07-26T12:00:00Z",
        }, 201);
      }
      if (path.endsWith("/curation/query")) {
        return response({
          items: [
            {
              id: "record-1",
              thumbnailPath: "thumbnails/record-1.jpg",
              metadata: {
                caption: "A full-body photograph outdoors",
                analysis: {
                  medium: "photograph",
                  fullBody: true,
                  qualifiedSingleFullBody: true,
                  personCount: 1,
                },
              },
            },
            {
              id: "record-2",
              thumbnailPath: "thumbnails/record-2.jpg",
              metadata: {
                caption: "A second full-body photograph",
                analysis: {
                  medium: "photograph",
                  fullBody: true,
                  qualifiedSingleFullBody: true,
                  personCount: 1,
                },
              },
            },
          ],
          reviews: [{
            recordId: "record-1",
            decision: "exclude",
            rejectionReason: "Persisted crop problem",
            updatedAt: "2026-07-26T12:00:00Z",
          }],
          nextCursor: "opaque-next",
          totalCount: 28,
        });
      }
      if (path.endsWith("/curation/facets")) {
        return response({
          facets: [{
            field: "medium",
            values: [{ value: "photograph", count: 28 }],
          }],
        });
      }
      if (path.endsWith("/materialize") && options.method === "POST") {
        const body = JSON.parse(options.body);
        return response({
          dataset: { id: "ds-catalog", name: body.name },
          requestedCount: body.requestedCount,
          materializedCount: body.requestedCount,
          selectedRecordIds: ["record-1", "record-2"],
          skipped: [{ recordId: "unavailable", reason: "unavailable" }],
          reusedExisting: false,
        }, 201);
      }
      if (path.endsWith("/review") && options.method === "PUT") {
        const body = JSON.parse(options.body);
        return response(body.decision === "default" ? null : {
          recordId: "record-1",
          decision: body.decision,
          rejectionReason: body.decision === "exclude" ? body.rejectionReason : null,
          updatedAt: "2026-07-26T12:00:00Z",
        });
      }
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

  it("preserves exact storage across null status polls and refreshes it from an exact list response", async () => {
    await remountWithFakeTimers();
    const original = fetch.getMockImplementation();
    let statusPolls = 0;
    fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/status")) {
        statusPolls += 1;
        return response(catalog({
          storage: null,
          processing: {
            ...catalog().processing,
            message: `Status poll ${statusPolls}`,
          },
        }));
      }
      return original(url, options);
    });

    await act(async () => vi.advanceTimersByTime(3000));
    await flush();
    await act(async () => vi.advanceTimersByTime(3000));
    await flush();
    expect(statusPolls).toBe(2);
    expect(container.textContent).toContain("Status poll 2");
    expect(container.textContent).toContain("6.5 KiB");

    catalogs[0] = catalog({
      storage: {
        databaseBytes: 2048,
        manifestBytes: 512,
        artifactBytes: 5632,
        totalBytes: 8192,
      },
    });
    await act(async () => [...container.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Refresh")).click());
    await flush();
    expect(container.textContent).toContain("8.0 KiB");
  });

  it("does not poll a paused or inactive selected catalog", async () => {
    catalogs = [catalog({ processing: { ...catalog().processing, state: "paused" } })];
    await remountWithFakeTimers();
    await act(async () => vi.advanceTimersByTime(9000));
    expect(requests.filter((item) => item.path.endsWith("/status"))).toHaveLength(0);

    catalogs = [catalog()];
    await act(async () => root.unmount());
    root = createRoot(container);
    await act(async () => {
      root.render(
        <ScreenActiveContext.Provider value={false}>
          <AppContext.Provider value={{ token: "token" }}>
            <DatasetCatalogsScreen />
            <ConfirmHost />
          </AppContext.Provider>
        </ScreenActiveContext.Provider>,
      );
    });
    await flush();
    await act(async () => vi.advanceTimersByTime(9000));
    expect(requests.filter((item) => item.path.endsWith("/status"))).toHaveLength(0);
  });

  it("stops after a terminal status response", async () => {
    await remountWithFakeTimers();
    const original = fetch.getMockImplementation();
    let polls = 0;
    fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/status")) {
        polls += 1;
        return response(catalog({ processing: { ...catalog().processing, state: "completed" } }));
      }
      return original(url, options);
    });
    await act(async () => vi.advanceTimersByTime(3000));
    await flush();
    await act(async () => vi.advanceTimersByTime(9000));
    expect(polls).toBe(1);
  });

  it("curates a reproducible primary sample with server paging, facets, saved views, and review overrides", async () => {
    const browse = [...container.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Browse catalog"));
    await act(async () => browse.click());
    await flush();

    expect(container.textContent).toContain("28 matches");
    expect(container.textContent).toContain("A full-body photograph outdoors");
    expect(container.textContent).toContain("photograph (28)");
    const queryRequest = requests.find((item) => item.path.endsWith("/curation/query"));
    expect(queryRequest.body.filters).toEqual(expect.arrayContaining([
      { field: "medium", values: ["photograph"] },
      { field: "personCount", values: ["1"] },
      { field: "qualifiedSingleFullBody", values: ["true"] },
    ]));
    expect(queryRequest.body.sampleSeed).toBe(14959);
    expect(queryRequest.body.deduplicate).toBe(true);

    const thumbnails = container.querySelectorAll(".catalog-thumbnail-card");
    await act(async () => thumbnails[0].click());
    const rejectionReason = container.querySelector("input[aria-label='Rejection reason']");
    expect(rejectionReason.value).toBe("Persisted crop problem");
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")
        .set.call(rejectionReason, "Unsaved reason for first");
      rejectionReason.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => thumbnails[1].click());
    expect(rejectionReason.value).toBe("");
    await act(async () => thumbnails[0].click());
    expect(rejectionReason.value).toBe("Persisted crop problem");
    await act(async () => [...container.querySelectorAll("button")]
      .find((button) => button.textContent === "Include").click());
    await flush();
    expect(requests.find((item) => item.path.endsWith("/review"))?.body).toEqual({
      decision: "include",
      rejectionReason: null,
    });

    const name = container.querySelector("input[placeholder='Full-body photos']");
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")
        .set.call(name, "My seeded sample");
      name.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => [...container.querySelectorAll("button")]
      .find((button) => button.textContent === "Save view").click());
    await flush();
    expect(requests.find((item) => item.path.endsWith("/saved-views")
      && item.options.method === "POST")?.body.name).toBe("My seeded sample");
    expect(container.textContent).toContain("Saved views (1)");

    await act(async () => [...container.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Load next page")).click());
    await flush();
    const pages = requests.filter((item) => item.path.endsWith("/curation/query"));
    expect(pages.at(-1).body.cursor).toBe("opaque-next");
    expect(pages.at(-1).body.includeTotal).toBe(false);
  });

  it("creates a project-owned training dataset from the exact server query with progress and completion", async () => {
    const completion = deferred();
    const original = fetch.getMockImplementation();
    fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/materialize")) {
        requests.push({ path, options, body: JSON.parse(options.body) });
        return completion.promise;
      }
      return original(url, options);
    });
    await act(async () => [...container.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Browse catalog")).click());
    await flush();

    const form = container.querySelector(".catalog-materialize-form");
    const name = form.querySelector("input[aria-label='Training dataset name']");
    const resultCount = form.querySelector("input[aria-label='Training dataset result count']");
    const seed = form.querySelector("input[aria-label='Training dataset sampling seed']");
    const policy = form.querySelector("select[aria-label='Training dataset deduplication policy']");
    const tags = [...form.querySelectorAll("label")].find((label) =>
      label.textContent.includes("Append generated tags")).querySelector("input");
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")
        .set.call(name, "Detached training set");
      name.dispatchEvent(new Event("input", { bubbles: true }));
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")
        .set.call(resultCount, "2");
      resultCount.dispatchEvent(new Event("input", { bubbles: true }));
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")
        .set.call(seed, "9007199254740993");
      seed.dispatchEvent(new Event("input", { bubbles: true }));
      Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, "value")
        .set.call(policy, "none");
      policy.dispatchEvent(new Event("change", { bubbles: true }));
      tags.click();
    });
    await act(async () => [...form.querySelectorAll("button")]
      .find((button) => button.textContent === "Create training dataset").click());
    expect(container.textContent).toContain("Sampling seed must be a whole number");
    expect(requests.some((item) => item.path.endsWith("/materialize"))).toBe(false);
    expect(seed.max).toBe("9007199254740991");
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")
        .set.call(seed, "123");
      seed.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => [...form.querySelectorAll("button")]
      .find((button) => button.textContent === "Create training dataset").click());
    expect(form.querySelector("[role='status']").textContent).toContain("deterministic replacements");

    await act(async () => completion.resolve(await response({
      dataset: { id: "ds-catalog", name: "Detached training set" },
      requestedCount: 2,
      materializedCount: 2,
      selectedRecordIds: ["record-1", "record-2"],
      skipped: [{ recordId: "gone", reason: "unavailable" }],
      reusedExisting: false,
    }, 201)));
    await flush();

    const materialize = requests.find((item) => item.path.endsWith("/materialize"));
    expect(materialize.body).toEqual(expect.objectContaining({
      projectId: "project-1",
      name: "Detached training set",
      requestedCount: 2,
      seed: 123,
      deduplicationPolicy: "none",
      includeGeneratedTags: true,
    }));
    expect(materialize.body.idempotencyKey).toEqual(expect.any(String));
    expect(materialize.body.query.filters).toEqual(expect.arrayContaining([
      { field: "medium", values: ["photograph"] },
      { field: "personCount", values: ["1"] },
    ]));
    expect(form.textContent).toContain("Created Detached training set with 2 images");
    expect(form.textContent).toContain("1 unavailable or duplicate candidates were replaced");
  });

  it("aborts and ignores stale materialization completion after switching catalogs", async () => {
    catalogs = [
      catalog(),
      catalog({ id: "cat-2", name: "Other catalog", path: "C:\\data\\other.catalog" }),
    ];
    await act(async () => [...container.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Refresh")).click());
    await flush();
    const completion = deferred();
    const original = fetch.getMockImplementation();
    let materializeSignal;
    let firstMaterializeBody;
    fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/materialize") && !firstMaterializeBody) {
        materializeSignal = options.signal;
        firstMaterializeBody = JSON.parse(options.body);
        return completion.promise;
      }
      return original(url, options);
    });
    await act(async () => [...container.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Browse catalog")).click());
    await flush();
    const form = container.querySelector(".catalog-materialize-form");
    const name = form.querySelector("input[aria-label='Training dataset name']");
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")
        .set.call(name, "Stale result");
      name.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => [...form.querySelectorAll("button")]
      .find((button) => button.textContent === "Create training dataset").click());
    await act(async () => [...container.querySelectorAll(".catalog-list-item")]
      .find((button) => button.textContent.includes("Other catalog")).click());
    expect(materializeSignal.aborted).toBe(true);
    await act(async () => completion.resolve(await response({
      dataset: { id: "stale", name: "Stale result" },
      requestedCount: 20,
      materializedCount: 20,
      selectedRecordIds: [],
      skipped: [],
      reusedExisting: false,
    }, 201)));
    await flush();
    expect(container.textContent).not.toContain("Created Stale result");
    expect(container.textContent).toContain("Other catalog");
    await act(async () => [...container.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Browse catalog")).click());
    await flush();
    const nextForm = container.querySelector(".catalog-materialize-form");
    await act(async () => [...nextForm.querySelectorAll("button")]
      .find((button) => button.textContent === "Create training dataset").click());
    await flush();
    const nextMaterialize = requests.filter((item) => item.path.endsWith("/materialize")).at(-1);
    expect(nextMaterialize.body.idempotencyKey).not.toBe(firstMaterializeBody.idempotencyKey);
    expect(nextMaterialize.path).toContain("/cat-2/materialize");
  });

  it("renders running progress as indeterminate when the scanner has no total", async () => {
    catalogs = [catalog({
      processing: {
        ...catalog().processing,
        candidateCount: 25_000,
        processedCount: 25_000,
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

    const progress = container.querySelector("[role='progressbar']");
    expect(progress.getAttribute("aria-valuenow")).toBeNull();
    expect(progress.getAttribute("aria-label")).toBe("25,000 processed; total unknown");
    expect(progress.classList.contains("catalog-progress--indeterminate")).toBe(true);
  });

  it("persists typed analyzer settings with the current revision without starting processing", async () => {
    const vision = [...container.querySelectorAll("label")]
      .find((label) => label.textContent.includes("Vision classification"))
      .querySelector("input");
    const personThreshold = container.querySelector("input[aria-label='Person confidence']");
    await act(async () => {
      vision.click();
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")
        .set.call(personThreshold, "0.4");
      personThreshold.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const save = [...container.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Save analyzer settings"));
    await act(async () => save.click());
    await flush();

    const update = requests.find((entry) => entry.path.endsWith("/analyzer-config"));
    expect(update.options.method).toBe("PUT");
    expect(update.body.expectedRevision).toBe(0);
    expect(update.body.settings.visionAnalysisEnabled).toBe(true);
    expect(update.body.settings.personMinConfidence).toBeUndefined();
    expect(update.body.settings.thresholds.personMinConfidence).toBe(0.4);
    expect(container.textContent).toContain("Changing settings does not start processing.");
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

  it("starts the typed survivor-only analysis job with revision and GPU selection", async () => {
    catalogs[0] = catalog({
      processing: { ...catalog().processing, state: "completed" },
      analyzerConfig: {
        ...catalog().analyzerConfig,
        revision: 7,
        settings: {
          ...catalog().analyzerConfig.settings,
          visionAnalysisEnabled: true,
          semanticEmbeddingsEnabled: true,
        },
      },
    });
    await remountWithFakeTimers();
    const gpu = container.querySelector("input[aria-label='Analysis GPU']");
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")
        .set.call(gpu, "gpu-1");
      gpu.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => [...container.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Run analysis")).click());
    await flush();
    expect(requests.find((item) => item.path === "/api/v1/catalogs/cat-1/analyze")?.body).toEqual({
      expectedAnalyzerConfigRevision: 7,
      requestedGpu: "gpu-1",
      batchSize: 16,
    });
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

  // Structured analysis resolves the person detector and DWPose before it can run, and both
  // resolvers went cache-only in epic 17625 — so on a machine without them the run reaches the
  // worker and dies with an install error this screen otherwise gives no way to act on.
  async function remountWithModels(models) {
    await act(async () => root.unmount());
    root = createRoot(container);
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ token: "token", models, createModelDownloadJob: vi.fn() }}>
          <DatasetCatalogsScreen />
          <ConfirmHost />
        </AppContext.Provider>,
      );
    });
    await flush();
  }

  it("offers the structured-analysis preprocessors when they are not installed", async () => {
    await remountWithModels([
      { id: "person_detector", name: "YOLO11m Person Detector", installState: "missing", downloadSizeLabel: "80 MB" },
      { id: "dwpose_pose_detector", name: "DWPose Pose Detector", installState: "missing", downloadSizeLabel: "330 MB" },
    ]);
    const notice = container.querySelector(".required-models-notice");
    expect(notice).toBeTruthy();
    expect(notice.textContent).toContain("YOLO11m Person Detector");
    expect(notice.textContent).toContain("DWPose Pose Detector");
    expect(notice.textContent).toContain("Structured person, face, and pose analysis");
  });

  it("lists only the preprocessor that is actually missing", async () => {
    await remountWithModels([
      { id: "person_detector", name: "YOLO11m Person Detector", installState: "installed" },
      { id: "dwpose_pose_detector", name: "DWPose Pose Detector", installState: "missing" },
    ]);
    const notice = container.querySelector(".required-models-notice");
    expect(notice).toBeTruthy();
    expect(notice.textContent).toContain("DWPose Pose Detector");
    expect(notice.textContent).not.toContain("YOLO11m");
  });

  it("shows nothing once both are installed", async () => {
    await remountWithModels([
      { id: "person_detector", name: "YOLO11m Person Detector", installState: "installed" },
      { id: "dwpose_pose_detector", name: "DWPose Pose Detector", installState: "installed" },
    ]);
    expect(container.querySelector(".required-models-notice")).toBeNull();
  });
});
