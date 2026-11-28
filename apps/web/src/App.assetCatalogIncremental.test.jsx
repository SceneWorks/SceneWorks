import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { useAppContext } from "./context/AppContext.js";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { writeWorkflowEmbedNoticeSeen } from "./workflowEmbed.js";

// The Library catalog is a SUMMARY listing refreshed incrementally: one full read, then only
// what changed since the cursor; the open asset is upgraded to its full envelope, and replay
// paths never run from a summary's partial recipe.

const captures = vi.hoisted(() => ({ current: null }));
function Probe() {
  captures.current = useAppContext();
  return <div data-page="Assets">Assets</div>;
}
vi.mock("./screens/LibraryScreen.jsx", () => ({ LibraryScreen: () => <Probe /> }));
vi.mock("./screens/CharacterStudio.jsx", () => ({ CharacterStudio: () => <Probe /> }));

let root;
let container;
beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
  writeWorkflowEmbedNoticeSeen(true);
  window.EventSource = FakeEventSource;
  FakeEventSource.instances = [];
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.restoreAllMocks();
});

const fullRecipe = {
  model: "z_image_turbo", mode: "text_to_image", prompt: "lighthouse", seed: 42,
  normalizedSettings: { count: 1, width: 1024, height: 1024 },
  loras: [{ id: "lora-a", weight: 0.8 }],
  rawAdapterSettings: { steps: 14, faceLikeness: { score: 0.9 } },
};
const fullAsset = (id, createdAt) => ({
  id, projectId: "project-1", type: "image", displayName: id, status: {}, createdAt,
  file: { path: `assets/${id}.png`, mimeType: "image/png" }, lineage: {},
  generationSetId: "set", generationSet: { id: "set", jobId: null, recipe: fullRecipe, prompt: "lighthouse" },
  recipe: fullRecipe,
});
// What `view=summary` returns for the same asset (see summarize_listed_asset).
const summaryOf = (asset) => ({
  ...asset, listView: "summary",
  generationSet: { id: asset.generationSet.id, jobId: asset.generationSet.jobId },
  recipe: { model: fullRecipe.model, mode: fullRecipe.mode, prompt: fullRecipe.prompt, seed: fullRecipe.seed,
    normalizedSettings: fullRecipe.normalizedSettings, rawAdapterSettings: { faceLikeness: fullRecipe.rawAdapterSettings.faceLikeness } },
});

function stubApi({ catalog, details, onImageJob = () => {} }) {
  const jobs = [];
  global.fetch = vi.fn(async (url, options = {}) => {
    const parsed = new URL(url);
    const path = parsed.pathname;
    if (path.endsWith("/health")) return response({ status: "ok", authRequired: false });
    if (path.endsWith("/access")) return response({ authRequired: false });
    if (path.endsWith("/jobs/events/ticket")) return response({ ticket: "test" });
    if (path.endsWith("/ui-preferences")) return response({ activeView: "Library", simpleUi: false });
    if (path.endsWith("/projects")) return response([{ id: "project-1", name: "Noir" }]);
    if (path.endsWith("/projects/project-1/assets")) return response(catalog(parsed.searchParams));
    const detail = /\/projects\/project-1\/assets\/([^/]+)$/.exec(path);
    if (detail && details[detail[1]]) return response(details[detail[1]]);
    if (detail) return { ok: false, status: 404, json: async () => ({ detail: "Asset not found" }) };
    if (path.endsWith("/models")) return response([{ id: "z_image_turbo", name: "Z", type: "image", family: "z-image" }]);
    if (options.method === "POST" && path.endsWith("/image/jobs")) {
      const body = JSON.parse(options.body);
      onImageJob(body);
      const job = { id: "new-image", type: "image_generate", projectId: "project-1", status: "queued", payload: body,
        createdAt: "2026-10-24T00:00:00Z", updatedAt: "2026-10-24T00:00:00Z" };
      jobs.push(job);
      return response(job);
    }
    if (path.endsWith("/jobs")) return response(jobs);
    return response([]);
  });
}

const catalogRequests = () => fetch.mock.calls
  .map(([url]) => new URL(url))
  .filter((url) => url.pathname.endsWith("/projects/project-1/assets"));

it("reads the summary catalog once, then merges only changes since the cursor", async () => {
  const older = fullAsset("older", "2026-10-24T00:00:01Z");
  const newer = fullAsset("newer", "2026-10-24T00:00:02Z");
  const added = fullAsset("added", "2026-10-24T00:00:03Z");
  stubApi({
    details: { older, newer, added },
    catalog: (params) => params.get("since") === "epoch.2"
      ? { cursor: "epoch.4", reset: false, assets: [summaryOf(added)], removedIds: ["older"] }
      : { cursor: "epoch.2", reset: true, assets: [summaryOf(newer), summaryOf(older)], removedIds: [] },
  });
  await act(async () => root.render(<App />));
  await settle();

  const [first] = catalogRequests();
  expect(first.searchParams.get("view")).toBe("summary");
  expect(first.searchParams.get("since")).toBe("");
  expect(first.searchParams.get("includeTrashed")).toBe("true");
  expect(captures.current.assets.map((asset) => asset.id)).toEqual(["newer", "older"]);
  // The selected (default: first) asset is upgraded in place to its full envelope.
  await vi.waitFor(() => expect(captures.current.assets[0].listView).toBeUndefined());
  expect(captures.current.assets[0].generationSet.recipe.loras).toEqual(fullRecipe.loras);
  expect(captures.current.assets[1].listView).toBe("summary");
  const upgraded = captures.current.assets[0];

  // A finished job refreshes the catalog: only the delta is requested and merged.
  const job = { id: "job-1", type: "image_generate", projectId: "project-1", status: "completed", revision: 2,
    createdAt: "2026-10-24T00:00:03Z", updatedAt: "2026-10-24T00:00:03Z", payload: {},
    result: { assetIds: ["added"], assets: [summaryOf(added)] } };
  await act(async () => FakeEventSource.instances[0].listeners["job.updated"]({ data: JSON.stringify(job) }));
  await settle();
  await vi.waitFor(() => expect(captures.current.assets.map((asset) => asset.id)).toEqual(["added", "newer"]));
  const requests = catalogRequests();
  expect(requests).toHaveLength(2);
  expect(requests[1].searchParams.get("since")).toBe("epoch.2");
  // The untouched asset keeps its upgraded envelope; the new one arrives as a summary.
  expect(captures.current.assets[1]).toBe(upgraded);
  expect(captures.current.assets[0].listView).toBe("summary");
});

it("loads the Library slice on the Library route and widens it in place for a full-catalog route", async () => {
  const library = fullAsset("library", "2026-10-24T00:00:02Z");
  const portrait = { ...fullAsset("portrait", "2026-10-24T00:00:01Z"), origin: "character_studio" };
  const added = { ...fullAsset("added", "2026-10-24T00:00:03Z"), origin: "character_studio" };
  stubApi({
    details: { library, portrait, added },
    catalog: (params) => {
      if (params.get("scope") === "library") {
        return { cursor: "epoch.2", reset: true, assets: [summaryOf(library)], removedIds: [] };
      }
      return params.get("since") === "epoch.3"
        ? { cursor: "epoch.4", reset: false, assets: [summaryOf(added)], removedIds: [] }
        : { cursor: "epoch.3", reset: true, assets: [summaryOf(library), summaryOf(portrait)], removedIds: [] };
    },
  });
  await act(async () => root.render(<App />));
  await settle();

  // The Library route reads only the server's Library slice.
  const [first] = catalogRequests();
  expect(first.searchParams.get("scope")).toBe("library");
  expect(first.searchParams.get("since")).toBe("");
  expect(captures.current.assets.map((asset) => asset.id)).toEqual(["library"]);

  // Character Studio needs Character Studio output: a reset read in the full scope. A Library
  // cursor cannot continue in another scope, so `since` starts over.
  await act(async () => captures.current.setActiveView("Characters"));
  await settle();
  await vi.waitFor(() => expect(captures.current.assets.map((asset) => asset.id)).toEqual(["library", "portrait"]));
  const widened = catalogRequests()[1];
  expect(widened.searchParams.get("scope")).toBe("all");
  expect(widened.searchParams.get("since")).toBe("");

  // Back on the Library the full catalog is kept (never narrowed), and a job refresh continues
  // the full-scope cursor instead of re-reading anything.
  await act(async () => captures.current.setActiveView("Library"));
  await settle();
  const job = { id: "job-1", type: "image_generate", projectId: "project-1", status: "completed", revision: 2,
    createdAt: "2026-10-24T00:00:03Z", updatedAt: "2026-10-24T00:00:03Z", payload: {},
    result: { assetIds: ["added"], assets: [summaryOf(added)] } };
  await act(async () => FakeEventSource.instances[0].listeners["job.updated"]({ data: JSON.stringify(job) }));
  await settle();
  await vi.waitFor(() => expect(captures.current.assets.map((asset) => asset.id)).toEqual(["added", "library", "portrait"]));
  const requests = catalogRequests();
  expect(requests).toHaveLength(3);
  expect(requests[2].searchParams.get("scope")).toBe("all");
  expect(requests[2].searchParams.get("since")).toBe("epoch.3");
});

it("starts a full-catalog route with the full catalog", async () => {
  const portrait = { ...fullAsset("portrait", "2026-10-24T00:00:01Z"), origin: "character_studio" };
  stubApi({
    details: { portrait },
    catalog: () => ({ cursor: "epoch.1", reset: true, assets: [summaryOf(portrait)], removedIds: [] }),
  });
  const originalFetch = global.fetch;
  global.fetch = vi.fn(async (url, options) => (new URL(url).pathname.endsWith("/ui-preferences")
    ? response({ activeView: "Characters", simpleUi: false })
    : originalFetch(url, options)));
  await act(async () => root.render(<App />));
  await settle();

  await vi.waitFor(() => expect(catalogRequests()).toHaveLength(1));
  expect(catalogRequests()[0].searchParams.get("scope")).toBe("all");
  expect(captures.current.assets.map((asset) => asset.id)).toEqual(["portrait"]);
});

it("reports the catalog incomplete until a widening read lands", async () => {
  const library = fullAsset("library", "2026-10-24T00:00:02Z");
  const portrait = { ...fullAsset("portrait", "2026-10-24T00:00:01Z"), origin: "character_studio" };
  let releaseFull;
  const fullCatalog = new Promise((resolve) => { releaseFull = resolve; });
  stubApi({
    details: { library, portrait },
    catalog: (params) => (params.get("scope") === "library"
      ? { cursor: "epoch.2", reset: true, assets: [summaryOf(library)], removedIds: [] }
      : fullCatalog),
  });
  await act(async () => root.render(<App />));
  await settle();
  // The Library route needs only the Library slice, so what it holds is complete for it.
  expect(captures.current.assetCatalogComplete).toBe(true);

  // On a full-catalog route the held slice is non-empty but NOT the whole catalog: restored-id
  // validations read this flag instead of treating "non-empty" as "loaded".
  await act(async () => captures.current.setActiveView("Characters"));
  await settle();
  expect(captures.current.assets.map((asset) => asset.id)).toEqual(["library"]);
  expect(captures.current.assetCatalogComplete).toBe(false);

  await act(async () => releaseFull({ cursor: "epoch.3", reset: true, assets: [summaryOf(library), summaryOf(portrait)], removedIds: [] }));
  await settle();
  await vi.waitFor(() => expect(captures.current.assetCatalogComplete).toBe(true));
  expect(captures.current.assets.map((asset) => asset.id)).toEqual(["library", "portrait"]);
});
