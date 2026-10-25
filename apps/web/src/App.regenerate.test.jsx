import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { writeWorkflowEmbedNoticeSeen } from "./workflowEmbed.js";

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

it.each([true, false])("queues without changing the page, viewer, or studio draft (original job available: %s)", async (originalAvailable) => {
  const recipe = { model: "z_image_turbo", mode: "text_to_image", prompt: "lighthouse", seed: 42,
    normalizedSettings: { count: 1, width: 1024, height: 1024 }, rawAdapterSettings: { steps: 14 } };
  const asset = { id: "image-1", projectId: "project-1", type: "image", displayName: "Lighthouse",
    status: {}, file: { path: "assets/lighthouse.png", mimeType: "image/png" }, recipe,
    lineage: originalAvailable ? { jobId: "original" } : {} };
  const original = { id: "original", projectId: "project-1", type: "image_generate" };
  const requests = [];
  const jobs = [];
  const draft = JSON.stringify({ model: "z_image_turbo", prompt: "An unfinished studio draft", seed: "999", count: 3 });
  window.localStorage.setItem("sceneworks-studio-image-project-1", draft);
  global.fetch = vi.fn(async (url, options = {}) => {
    const path = new URL(url).pathname;
    if (path.endsWith("/health")) return response({ status: "ok", authRequired: false });
    if (path.endsWith("/access")) return response({ authRequired: false });
    if (path.endsWith("/jobs/events/ticket")) return response({ ticket: "test" });
    if (path.endsWith("/projects")) return response([{ id: "project-1", name: "Noir" }]);
    if (path.endsWith("/assets")) return response([asset]);
    if (path.endsWith("/models")) return response([{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]);
    if (path.endsWith("/jobs/original")) return response(original);
    if (options.method === "POST" && (path.endsWith("/duplicate") || path.endsWith("/image/jobs"))) {
      const body = JSON.parse(options.body);
      requests.push({ path, body });
      const job = { ...original, id: "new-image", status: "queued", payload: body };
      jobs.push(job);
      return response(job);
    }
    if (path.endsWith("/jobs")) return response(jobs);
    return response([]);
  });
  await act(async () => root.render(<App />));
  await settle();
  await act(async () => [...document.body.querySelectorAll("button")].find((b) => b.textContent === "Assets").click());
  await settle();
  await act(async () => document.body.querySelector(".asset-tile").dispatchEvent(new MouseEvent("dblclick", { bubbles: true })));
  await settle();
  const viewer = document.body.querySelector('[role="dialog"]');
  const image = viewer.querySelector("img");
  const url = window.location.href;
  const page = container.querySelector(".library-screen") ?? container.querySelector(".page-frame");
  const savedDraft = window.localStorage.getItem("sceneworks-studio-image-project-1");
  await act(async () => [...viewer.querySelectorAll("button")].find((b) => b.textContent === "Re-Generate").click());
  await settle();

  expect(requests).toHaveLength(1);
  expect(document.body.querySelector('[role="dialog"]')).toBe(viewer);
  expect(viewer.querySelector("img")).toBe(image);
  expect(window.location.href).toBe(url);
  expect(container.contains(page)).toBe(true);
  expect(window.localStorage.getItem("sceneworks-studio-image-project-1")).toBe(savedDraft);
  expect(document.body.querySelector(".image-studio")).toBeNull();
  if (originalAvailable) {
    expect(requests[0].path).toContain("/jobs/original/duplicate");
    expect(requests[0].body.payloadChanges).toMatchObject({ seed: expect.any(Number), seeds: [] });
  } else {
    expect(requests[0].body).toMatchObject({ model: "z_image_turbo", prompt: "lighthouse", seed: null });
  }
});
