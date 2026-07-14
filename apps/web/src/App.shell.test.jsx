import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App, ErrorBoundary } from "./main.jsx";
import { assetUrl } from "./components/assetMedia.jsx";
import { SetupWizard } from "./screens/SetupWizard.jsx";
import { AppStaticContext, AppLiveContext } from "./context/AppContext.js";
import { FakeEventSource, response, settle, navLabels, changeField } from "./main.testSupport.jsx";

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

  it("encodes fallback asset file path segments", () => {
    const url = assetUrl({
      projectId: "project-1",
      file: { path: "assets\\images/final image #1?.png" },
    });

    expect(url).toBe(
      "http://localhost:8000/api/v1/projects/project-1/files/assets/images/final%20image%20%231%3F.png",
    );
  });

  it("shows a fallback instead of a blank screen when rendering fails", async () => {
    function BrokenScreen() {
      throw new Error("Render smoke signal");
    }

    const preventExpectedError = (event) => event.preventDefault();
    window.addEventListener("error", preventExpectedError);
    vi.spyOn(console, "error").mockImplementation(() => {});
    root = createRoot(container);
    try {
      await act(async () => {
        root.render(
          <ErrorBoundary>
            <BrokenScreen />
          </ErrorBoundary>,
        );
      });
    } finally {
      window.removeEventListener("error", preventExpectedError);
    }

    expect(container.textContent).toContain("Something went wrong");
    expect(container.textContent).toContain("Render smoke signal");
  });

  const wizardModels = [
    {
      id: "z_image_turbo",
      name: "Z-Image-Turbo",
      type: "image",
      recommended: true,
      downloadable: true,
      installState: "missing",
      downloadSizeLabel: "30.6 GB",
      downloadSizeBytes: 32899667397,
      downloadSizeEstimated: true,
      downloads: [{ repo: "Tongyi-MAI/Z-Image-Turbo" }],
    },
    {
      id: "qwen_image",
      name: "Qwen Image",
      type: "image",
      downloadable: true,
      installState: "installed",
      downloadSizeLabel: "53.7 GB",
      downloadSizeBytes: 57704594653,
      downloads: [{ repo: "Qwen/Qwen-Image" }],
    },
    {
      id: "wan_2_2",
      name: "Wan2.2",
      type: "video",
      downloadable: true,
      installState: "missing",
      downloadSizeLabel: "31.8 GB",
      downloadSizeBytes: 34203021834,
      downloadSizeEstimated: true,
      downloads: [{ repo: "Wan-AI/Wan2.2-TI2V-5B" }],
    },
    {
      id: "ltx_2_3",
      name: "LTX-2.3",
      type: "video",
      recommended: true,
      autoDownload: false,
      downloadable: true,
      installState: "missing",
      downloadSizeLabel: "146 GB",
      downloadSizeBytes: 157004895813,
      downloads: [{ repo: "Lightricks/LTX-2.3" }],
    },
  ];

  function renderWizard(overrides = {}) {
    const props = {
      models: wizardModels,
      jobs: [],
      onDownloadModel: vi.fn(),
      onCreateProject: vi.fn(async (name) => ({ id: "project-new", name })),
      onComplete: vi.fn(async () => {}),
      onOpenQueue: vi.fn(),
      ...overrides,
    };
    root = createRoot(container);
    return props;
  }

  it("groups downloadable models, pre-checks recommended ones, and flags installed", async () => {
    const props = renderWizard();
    await act(async () => {
      root.render(<SetupWizard {...props} />);
    });
    await settle();

    expect(container.textContent).toContain("Image models");
    expect(container.textContent).toContain("Video models");
    expect(container.textContent).toContain("Recommended");
    expect(container.textContent).toContain("Already installed");
    expect(container.textContent).toContain("~30.6 GB");

    const checkboxes = [...document.body.querySelectorAll("input[type=checkbox]")];
    // DOM order: image group (z_image_turbo, qwen_image[installed]), then video (wan_2_2, ltx_2_3).
    expect(checkboxes[0].checked).toBe(true); // z_image_turbo — recommended, autoDownload not disabled
    expect(checkboxes[1].disabled).toBe(true); // qwen_image — installed, not selectable
    expect(checkboxes[2].checked).toBe(false); // wan_2_2 — not recommended
    expect(checkboxes[3].checked).toBe(false); // ltx_2_3 — recommended but autoDownload:false (~146 GB)
    // LTX-2.3 is still surfaced as recommended (badge) and shows its size so the choice is informed.
    expect(container.textContent).toContain("146 GB");
    expect(document.body.querySelectorAll(".setup-wizard-tag").length).toBe(2); // z_image_turbo + ltx_2_3
  });

  it("auto-selects recommended models, leaving autoDownload:false ones opt-in", async () => {
    const props = renderWizard();
    await act(async () => {
      root.render(<SetupWizard {...props} />);
    });
    await settle();

    const downloadButton = [...document.body.querySelectorAll("button")].find((button) =>
      button.textContent.startsWith("Download"),
    );
    expect(downloadButton.textContent).toContain("1");
    await act(async () => {
      downloadButton.click();
    });
    await settle();

    // Only the pre-checked image model downloads; LTX-2.3 stays opt-in.
    expect(props.onDownloadModel).toHaveBeenCalledTimes(1);
    expect(props.onDownloadModel.mock.calls[0][0].id).toBe("z_image_turbo");
    // Re-firing is guarded: the button disables once nothing is pending.
    expect(downloadButton.disabled).toBe(true);
  });

  it("advances to the project step and creates a project then marks setup complete", async () => {
    const props = renderWizard();
    await act(async () => {
      root.render(<SetupWizard {...props} />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Continue").click();
    });
    await settle();
    expect(container.textContent).toContain("Create your first project");
    // Skipping downloads is allowed — Continue advanced without firing any.
    expect(props.onDownloadModel).not.toHaveBeenCalled();

    const input = document.body.querySelector("input[type=text]") ?? document.body.querySelector("input");
    await changeField(input, "My First Project");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Finish setup").click();
    });
    await settle();

    expect(props.onCreateProject).toHaveBeenCalledWith("My First Project");
    expect(props.onComplete).toHaveBeenCalledTimes(1);
  });

  it("does not mark setup complete when project creation fails", async () => {
    const props = renderWizard({ onCreateProject: vi.fn(async () => null) });
    await act(async () => {
      root.render(<SetupWizard {...props} />);
    });
    await settle();
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Continue").click();
    });
    await settle();
    const input = document.body.querySelector("input");
    await changeField(input, "Doomed Project");
    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Finish setup").click();
    });
    await settle();

    expect(props.onCreateProject).toHaveBeenCalledWith("Doomed Project");
    expect(props.onComplete).not.toHaveBeenCalled();
    // issue #1435 / sc-11855: the wizard overlays the whole app (incl. Settings),
    // so a failed create must surface in-wizard recovery guidance rather than
    // silently leaving the user stuck on the model screen.
    expect(container.textContent).toContain("different workspace folder");
  });

  it("renders the app navigation against mocked API calls", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    expect(container.textContent).toContain("Assets");
    expect(navLabels(container, "Workspace")).not.toContain("Library");
    expect(navLabels(container, "Library")).toContain("Assets");
    expect(container.textContent).toContain("Train");
    expect(container.textContent).toContain("Queue");
  });

  // sc-10728: the global default generation quality is durable across desktop relaunches only
  // because App re-seeds the localStorage instant-paint cache from the server copy on mount — the
  // shell's per-launch 127.0.0.1:<port> origin wipes origin-keyed localStorage every launch. This
  // asserts the launch-time GET /ui-preferences primes readDefaultGenerationQuality()'s cache.
  it("re-seeds the default generation quality cache from the server on mount (sc-10728)", async () => {
    window.localStorage.clear(); // simulate the wiped cache after a relaunch under a new origin
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
      if (path.endsWith("/ui-preferences")) {
        // The durable server copy the desktop shell persisted in a previous session.
        return Promise.resolve(response({ defaultGenerationQuality: "q4" }));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    expect(window.localStorage.getItem("sceneworks-default-generation-quality")).toBe("q4");
  });

  // sc-4193 regression (F-WEB-3): the queueCounts memo reads queueSummary, so a
  // queue.updated SSE event (which changes queueSummary but NOT jobs) must refresh
  // the topbar "Queue N" chip. Before the fix the memo's deps were [jobs] only, so
  // queue-only updates served stale counts until the next jobs mutation.
  it("updates the Queue chip on a queue.updated event with no jobs change", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    const queueChip = () => [...document.body.querySelectorAll(".queue-chip")][0];
    expect(queueChip()?.textContent).toContain("Queue 0");

    // Pure queue.updated: two active jobs, and no job.updated to mutate `jobs`.
    await act(async () => {
      FakeEventSource.instances[0].listeners["queue.updated"]({
        data: JSON.stringify({
          counts: { active: 2, queued: 2 },
          activeJobs: [{ id: "job-a" }, { id: "job-b" }],
        }),
      });
    });
    await settle();

    expect(queueChip()?.textContent).toContain("Queue 2");
  });

  // sc-8811 regression (F-009): the STATIC context value must keep its identity across an
  // App re-render that touches none of its entries. Before the fix, refreshData /
  // refreshDataWithLoraOverlay were plain per-render declarations passed into
  // useModelsAndLoras, so deleteModel/deleteLora — and therefore the whole memoized
  // context value — got a fresh identity on EVERY App render (each SSE tick, each
  // keystroke), silently defeating the sc-4194 memoization and re-rendering every
  // useAppContext consumer. A queue.updated event only changes queueSummary, which is
  // not part of the context value, so the provider value identity must survive it.
  //
  // sc-8855 (F-053): App now renders TWO providers (AppStaticContext + AppLiveContext).
  // This asserts the STATIC value's identity, which must survive an unrelated re-render.
  it("keeps the static context value identity stable across an unrelated re-render (sc-8811/sc-8855)", async () => {
    const providedValues = [];
    const OriginalProvider = AppStaticContext.Provider;
    // Swap in a recording pass-through provider so the test can observe the exact
    // static-value identities App feeds the context on each committed render.
    AppStaticContext.Provider = function RecordingProvider({ value, children }) {
      providedValues.push(value);
      return <OriginalProvider value={value}>{children}</OriginalProvider>;
    };
    try {
      root = createRoot(container);
      await act(async () => {
        root.render(<App />);
      });
      await settle();

      const before = providedValues[providedValues.length - 1];
      const countBefore = providedValues.length;
      expect(before).toBeTruthy();

      // Pure queue.updated (no workers array, no job.updated): re-renders App via
      // setQueueSummary without changing anything the static value depends on.
      await act(async () => {
        FakeEventSource.instances[0].listeners["queue.updated"]({
          data: JSON.stringify({ counts: { active: 2, queued: 2 }, activeJobs: [{ id: "job-a" }] }),
        });
      });
      await settle();

      const after = providedValues[providedValues.length - 1];
      expect(providedValues.length).toBeGreaterThan(countBefore); // the event really re-rendered App
      expect(after).toBe(before); // identity preserved → consumer tree memoization holds
    } finally {
      AppStaticContext.Provider = OriginalProvider;
    }
  });

  // sc-8855 (F-053): a job/worker SSE tick must NOT change the STATIC context value's
  // identity — that is the whole point of the split. The LIVE value's identity DOES
  // change (it carries the churning jobs/workers fields), but static-only consumers
  // (Settings, Presets, Logs, pose/keypoint pickers) subscribe to the static value and
  // therefore skip re-render on every tick. We record both provider values and assert the
  // static identity is preserved across a job.updated + workers.snapshot tick while the
  // live identity changes.
  it("static context value identity survives a job/worker SSE tick; live value changes (sc-8855)", async () => {
    const staticValues = [];
    const liveValues = [];
    const OriginalStatic = AppStaticContext.Provider;
    const OriginalLive = AppLiveContext.Provider;
    AppStaticContext.Provider = function RecordingStatic({ value, children }) {
      staticValues.push(value);
      return <OriginalStatic value={value}>{children}</OriginalStatic>;
    };
    AppLiveContext.Provider = function RecordingLive({ value, children }) {
      liveValues.push(value);
      return <OriginalLive value={value}>{children}</OriginalLive>;
    };
    try {
      root = createRoot(container);
      await act(async () => {
        root.render(<App />);
      });
      await settle();

      const staticBefore = staticValues[staticValues.length - 1];
      const liveBefore = liveValues[liveValues.length - 1];
      expect(staticBefore).toBeTruthy();
      expect(liveBefore).toBeTruthy();

      // A real job tick: job.updated changes the jobs list (LIVE), which must not touch
      // the static value's identity.
      await act(async () => {
        FakeEventSource.instances[0].listeners["job.updated"]({
          data: JSON.stringify({ id: "job-churn", status: "running", type: "image", projectId: "p1" }),
        });
      });
      await settle();

      const staticAfter = staticValues[staticValues.length - 1];
      const liveAfter = liveValues[liveValues.length - 1];
      // Static identity preserved: cold-only consumers do NOT re-render on the tick.
      expect(staticAfter).toBe(staticBefore);
      // Live identity changed: the jobs field really churned.
      expect(liveAfter).not.toBe(liveBefore);
      // And the static value never carried the churning live keys.
      for (const key of ["jobs", "workersById", "visibleWorkers", "filteredJobs", "gpuOptions"]) {
        expect(staticAfter).not.toHaveProperty(key);
      }
    } finally {
      AppStaticContext.Provider = OriginalStatic;
      AppLiveContext.Provider = OriginalLive;
    }
  });

});
