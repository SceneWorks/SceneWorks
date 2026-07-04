import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";

describe("SceneWorks app shell", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    global.fetch = vi.fn((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
      }
      return Promise.resolve(response([]));
    });
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it("shows the real Replace Person panel for a replacement-capable model", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Replace person").click();
    });
    await settle();

    // LTX-2.3 is now the primary replacement-capable model, and the placeholder
    // copy is gone in favor of the real-tracking guidance (sc-1487).
    expect(container.textContent).toContain("Real person tracking");
    expect(container.textContent).not.toContain("V1 placeholder tracking");
  });

  it("updates Replace Person readiness when a capable worker registers over SSE", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Replace person").click();
    });
    await settle();

    // No live detector/tracker worker yet -> detection is gated with a reason.
    expect(container.textContent).toContain("Detection unavailable");

    // A GPU worker registers with the real person capabilities (sc-1484 finding:
    // readiness must track live worker updates, not just the initial load).
    await act(async () => {
      FakeEventSource.instances[0].listeners["worker.updated"]({
        data: JSON.stringify({
          id: "python-gpu-0",
          gpuId: "0",
          gpuName: "GPU 0",
          status: "idle",
          capabilities: ["gpu", "person_detect", "person_track", "person_segment", "person_replace"],
        }),
      });
    });
    await settle();

    // Readiness is recomputed from the live worker list, so the gate clears.
    expect(container.textContent).not.toContain("Detection unavailable");
    expect(container.textContent).not.toContain("Replacement unavailable");
  });

  it("surfaces replacement-unavailable when no live worker can run person replacement", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Replace person").click();
    });
    await settle();

    // Detector/tracker are live, but no worker advertises person_replace.
    await act(async () => {
      FakeEventSource.instances[0].listeners["worker.updated"]({
        data: JSON.stringify({
          id: "python-gpu-0",
          gpuId: "0",
          gpuName: "GPU 0",
          status: "idle",
          capabilities: ["gpu", "person_detect", "person_track"],
        }),
      });
    });
    await settle();

    expect(container.textContent).not.toContain("Detection unavailable");
    // The same replace-readiness flag that gates the submit button (sc-1484 finding).
    expect(container.textContent).toContain("Replacement unavailable");
  });

  it("keeps completed Replace Person detections visible in Video Studio", async () => {
    global.fetch.mockImplementation((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Project One" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(
          response([
            {
              id: "wan_replace",
              name: "Wan Replace",
              type: "video",
              capabilities: ["replace_person", "image_to_video", "text_to_video"],
              defaults: { duration: 4, fps: 24, resolution: "1280x720", quality: "balanced" },
              limits: { durations: [4], fps: [24], resolutions: ["1280x720"] },
            },
          ]),
        );
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(
          response([
            {
              id: "detect-job-1",
              type: "person_detect",
              status: "completed",
              projectId: "project-1",
              payload: { sourceAssetId: "clip-1" },
              result: {
                frameAssetId: "frame-1",
                detections: [{ id: "person-1", label: "person", confidence: 0.82, box: { x: 0.1, y: 0.2, width: 0.3, height: 0.4 } }],
              },
              createdAt: "2026-05-18T22:00:00Z",
            },
          ]),
        );
      }
      if (path.endsWith("/assets")) {
        return Promise.resolve(
          response([
            { id: "clip-1", type: "video", displayName: "Source Clip", file: { mimeType: "video/mp4" }, status: {} },
            { id: "frame-1", type: "image", displayName: "Detection Frame", file: { mimeType: "image/png" }, status: {} },
          ]),
        );
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Replace person").click();
    });
    await settle();

    expect(container.textContent).toContain("1 candidates");
    expect(container.textContent).not.toContain("No analysis yet");
  });

});
