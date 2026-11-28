import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { writeWorkflowEmbedNoticeSeen } from "./workflowEmbed.js";

// The Library catalog is a SUMMARY listing refreshed incrementally: one full read, then only
// what changed since the cursor; the open asset is upgraded to its full envelope, and replay
// paths never run from a summary's partial recipe.

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

it("regenerates a summary catalog asset from its FULL recorded recipe", async () => {
  const asset = fullAsset("image-1", "2026-10-24T00:00:00Z");
  const requests = [];
  stubApi({
    details: { "image-1": asset },
    catalog: () => ({ cursor: "epoch.1", reset: true, assets: [summaryOf(asset)], removedIds: [] }),
    onImageJob: (body) => requests.push(body),
  });
  await act(async () => root.render(<App />));
  await settle();
  await act(async () => [...document.body.querySelectorAll(".nav-item")].find((b) => b.textContent === "Assets").click());
  await settle();
  await act(async () => {
    document.body.querySelector(".asset-tile").dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
  });
  await settle();
  const viewer = document.body.querySelector('[role="dialog"]');
  await act(async () => [...viewer.querySelectorAll("button")].find((b) => b.textContent === "Re-Generate").click());
  await vi.waitFor(async () => {
    await settle();
    expect(requests).toHaveLength(1);
  });
  // LoRAs and adapter settings exist only on the full envelope.
  expect(requests[0]).toMatchObject({ model: "z_image_turbo", prompt: "lighthouse", loras: fullRecipe.loras });
  expect(requests[0].advanced?.steps).toBe(14);
  expect(fetch.mock.calls.some(([url]) => new URL(url).pathname.endsWith("/assets/image-1"))).toBe(true);
});
