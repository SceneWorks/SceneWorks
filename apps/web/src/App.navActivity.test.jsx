import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";

// The Video Editor nav dot must mean "editor work is running", not "a timeline
// exists" — the latter lit permanently after the first saved timeline was created.
describe("sidebar activity dots", () => {
  let container;
  let root;

  const TIMELINE = {
    id: "timeline-1",
    name: "Cut 1",
    aspectRatio: "16:9",
    fps: 30,
    width: 1280,
    height: 720,
    tracks: [{ id: "track_main", name: "Main", items: [] }],
  };

  async function mount({ jobs = [], timelines = [TIMELINE] } = {}) {
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
        return Promise.resolve(response([{ id: "project-a", name: "Project A" }]));
      }
      if (path.endsWith("/api/v1/jobs")) {
        return Promise.resolve(response(jobs));
      }
      if (path.endsWith("/timelines")) {
        return Promise.resolve(response(timelines));
      }
      if (path.match(/\/timelines\/[^/]+$/)) {
        return Promise.resolve(response(timelines[0]));
      }
      return Promise.resolve(response([]));
    });
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    // Timelines hydrate on first navigation to the Editor route (sc-14783), so the
    // "a timeline exists" state only becomes reachable after visiting it.
    await navigate("Video Editor");
  }

  async function navigate(label) {
    const button = [...container.querySelectorAll(".nav-item")].find(
      (candidate) => candidate.textContent.trim() === label,
    );
    expect(button, `navigation button ${label}`).toBeTruthy();
    await act(async () => {
      button.click();
    });
    await settle();
  }

  function editorNavDot() {
    const button = [...container.querySelectorAll("button.nav-item")].find(
      (candidate) => candidate.textContent.trim() === "Video Editor",
    );
    return button?.querySelector(".nav-pulse") ?? null;
  }

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it("leaves the Video Editor dot off when a loaded timeline is idle", async () => {
    await mount({ jobs: [{ id: "job-1", type: "image_generate", status: "running" }] });

    expect(editorNavDot()).toBeNull();
  });

  it("lights the Video Editor dot while an export is in flight", async () => {
    await mount({ jobs: [{ id: "job-1", type: "timeline_export", status: "running" }] });

    expect(editorNavDot()).not.toBeNull();
  });

  it("lights the Video Editor dot for timeline-initiated generation", async () => {
    await mount({
      jobs: [
        {
          id: "job-1",
          type: "video_generate",
          status: "queued",
          payload: { advanced: { timelineAction: "bridge" } },
        },
      ],
    });

    expect(editorNavDot()).not.toBeNull();
  });

  it("drops the Video Editor dot once the editor job reaches a terminal status", async () => {
    await mount({ jobs: [{ id: "job-1", type: "timeline_export", status: "completed" }] });

    expect(editorNavDot()).toBeNull();
  });
});
