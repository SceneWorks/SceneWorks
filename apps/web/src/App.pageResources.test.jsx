import React, { act, useEffect } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { App } from "./App.jsx";
import { useAppContext } from "./context/AppContext.js";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { mountRoot, unmountRoot } from "./testUtils/dom.js";

const captures = vi.hoisted(() => ({ current: null, unmounted: [] }));
function Probe({ page }) {
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
afterEach(() => { unmountRoot(root, container); vi.unstubAllGlobals(); });
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
