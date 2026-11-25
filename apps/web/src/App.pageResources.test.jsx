import React, { act, useEffect } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { App } from "./App.jsx";
import { useAppContext, useAppStatic } from "./context/AppContext.js";
import * as portableValidation from "./portableValidation.js";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { mountRoot, unmountRoot } from "./testUtils/dom.js";

const captures = vi.hoisted(() => ({ current: null, unmounted: [] }));
function Probe({ page }) {
  captures.static = useAppStatic();
  captures.current = { ...useAppContext(), page };
  useEffect(() => () => { captures.unmounted.push(page); }, [page]);
  return <div data-page={page}>{page}</div>;
}
vi.mock("./screens/LibraryScreen.jsx", () => ({ LibraryScreen: () => <Probe page="Assets" /> }));
vi.mock("./screens/QueueScreen.jsx", () => ({ QueueScreen: () => <Probe page="Queue" /> }));
vi.mock("./screens/SettingsScreen.jsx", () => ({ SettingsScreen: () => <Probe page="Settings" /> }));

let root, container, serverJobs;
const source = { id: "source", projectId: "project", type: "image", displayName: "Source", createdAt: "2026-11-25T12:00:00Z", status: {}, recipe: { prompt: "source recipe" } };
const output = { ...source, id: "output", displayName: "Output" };
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  ({ root, container } = mountRoot());
  captures.unmounted = [];
  FakeEventSource.instances = [];
  window.EventSource = FakeEventSource;
  localStorage.clear();
  serverJobs = [{ id: "job", type: "image_generate", projectId: "project", status: "running", stage: "denoising",
    progress: 0.35, revision: 1, createdAt: "2026-11-25T12:00:00Z", updatedAt: "2026-11-25T12:01:00Z",
    payload: { prompt: "queued prompt", metadata: "x".repeat(100_000) }, result: { assetIds: ["output"], assets: [output] } }];
  vi.stubGlobal("fetch", vi.fn(async (url) => {
    const path = new URL(url).pathname;
    if (path.endsWith("/health")) return response({ status: "ok" });
    if (path.endsWith("/access")) return response({ authRequired: false });
    if (path.endsWith("/capabilities/mac")) return response({ macGatingActive: false });
    if (path.endsWith("/ui-preferences")) return response({ activeView: "Library", simpleUi: false });
    if (path.endsWith("/projects")) return response([{ id: "project", name: "Project" }]);
    if (path.endsWith("/jobs/events/ticket")) return response({ ticket: "ticket" });
    if (path.endsWith("/jobs")) return response(serverJobs);
    if (path.endsWith("/assets/output")) return response(output);
    if (path.endsWith("/assets")) return response([source, output]);
    return response([]);
  }));
});
afterEach(() => { unmountRoot(root, container); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
async function navigate(label) {
  await act(async () => [...container.querySelectorAll(".nav-item")].find((button) => button.textContent === label).click());
  await settle();
}
const assetListRequests = () => fetch.mock.calls.filter(([url]) => new URL(url).pathname.endsWith("/assets")).length;

it("releases Assets on Queue entry, then releases Queue details and ignores off-page catalog refreshes", async () => {
  await act(async () => root.render(<App />));
  await settle();
  expect(captures.current.assets.map((asset) => asset.id)).toEqual(["source", "output"]);
  const initialLists = assetListRequests();

  await navigate("Queue");
  expect(captures.unmounted).toContain("Assets");
  expect(captures.current.assets.map((asset) => asset.id)).toEqual(["output"]);
  expect(captures.current.jobs[0].payload.prompt).toBe("queued prompt");
  expect(assetListRequests()).toBe(initialLists);

  await navigate("Settings");
  expect(captures.unmounted).toContain("Queue");
  expect(captures.current.assets).toEqual([]);
  expect(captures.current.jobs[0]).toMatchObject({ payload: {}, result: {}, progress: 0.35 });
  serverJobs = [{ ...serverJobs[0], revision: 2, progress: 0.75, updatedAt: "2026-11-25T12:02:00Z",
    result: { ...serverJobs[0].result, previewFrame: { dataUrl: "data:image/png;base64,new-frame" } } }];
  await act(async () => FakeEventSource.instances[0].listeners["job.updated"]({ data: JSON.stringify(serverJobs[0]) }));
  await settle();
  expect(captures.current.jobs[0]).toMatchObject({ stage: "denoising", progress: 0.75, payload: {}, result: {} });
  expect(captures.current.assets).toEqual([]);
  expect(assetListRequests()).toBe(initialLists);

  await navigate("Queue");
  expect(captures.current.jobs[0]).toMatchObject({ progress: 0.75, payload: { prompt: "queued prompt" },
    result: { previewFrame: { dataUrl: "data:image/png;base64,new-frame" } } });
  expect(FakeEventSource.instances).toHaveLength(1);
  await navigate("Assets");
  expect(captures.current.assets.map((asset) => asset.id)).toEqual(["source", "output"]);
  expect(captures.current.jobs[0].result).toEqual({});
  expect(assetListRequests()).toBe(initialLists + 1);
});

it("keeps static catalog consumers stable while a background batch updates its aggregate progress", async () => {
  vi.spyOn(portableValidation, "validatePortableRequest").mockResolvedValue({ state: "valid" });
  await act(async () => root.render(<App />));
  await settle();
  const before = captures.static;
  await act(async () => captures.current.backgroundImageBatch.runBatch({
    BATCH_ACTIVE_JOB_LIMIT: 1, batchPrompts: ["one"], batchVariables: {}, resolutionOptions: [],
    structuredPromptModel: false, batchJobCount: 1, selectedModel: null,
    buildBatchJobRequest: (prompt) => ({ prompt, count: 1 }),
    createImageJob: async () => ({ id: "batch", status: "queued" }),
  }));
  const update = (status, revision) => ({ id: "batch", type: "image_generate", projectId: "project", status,
    revision, createdAt: "2026-11-25T12:00:00Z", updatedAt: `2026-11-25T12:0${revision}:00Z`, payload: {}, result: {} });
  await act(async () => FakeEventSource.instances[0].listeners["job.updated"]({ data: JSON.stringify(update("running", 1)) }));
  expect(captures.current.backgroundImageBatch.batchRunProgress).toMatchObject({ running: 1, queued: 0 });
  expect(captures.static).toBe(before);
  await act(async () => FakeEventSource.instances[0].listeners["job.updated"]({ data: JSON.stringify(update("completed", 2)) }));
  expect(captures.current.backgroundImageBatch.batchRunProgress.completed).toBe(1);
});

it("acquires fresh output status on Queue entry instead of reusing embedded job snapshots", async () => {
  const originalFetch = fetch;
  vi.stubGlobal("fetch", vi.fn((url, options) => new URL(url).pathname.endsWith("/assets/output")
    ? Promise.resolve(response({ ...output, status: { trashed: true } })) : originalFetch(url, options)));
  await act(async () => root.render(<App />));
  await settle();
  await navigate("Queue");
  expect(captures.current.assets).toEqual([{ ...output, status: { trashed: true } }]);
  expect(captures.current.jobs[0].result.assets[0].status).toEqual({});
});
