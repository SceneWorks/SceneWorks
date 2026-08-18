// sc-19709: the relocate branch, driven through the REAL desktop app shell.
//
// Relocating goes through the app's existing library-path configuration (`set_model_library`
// writes the same `hf_home` the first-run storage step writes), which the sidecars receive as
// spawn environment — so it applies on the next launch. That is disclosed, not hidden, and the
// restart is offered: "Restart now" relaunches through the app's graceful teardown, "Later" leaves
// the relocation durable with nothing queued.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }));
vi.mock("./runtime.js", () => ({ isDesktop: true, tauriInvoke: invoke }));

import { App } from "./main.jsx";
import {
  changeField,
  field,
  FakeEventSource,
  response,
  settle,
} from "./main.testSupport.jsx";

const MODEL = {
  id: "z_image_turbo",
  name: "Z-Image Turbo",
  type: "image",
  family: "z-image",
  installState: "installed",
  modelAvailability: "external_ready",
};

const UNAVAILABLE_BODY = {
  detail: "Model 'z_image_turbo' is installed on an external model library…",
  code: "external_model_library_unavailable",
  context: {
    schemaVersion: 1,
    availability: "installed_external_unavailable",
    modelId: "z_image_turbo",
    modelName: "Z-Image Turbo",
    configuredLibraryPath: "/Volumes/Models/hf/hub",
    expectedLibraryPath: "/Volumes/Models/hf/hub",
    expectedVolumeId: "macos-volume:abc",
  },
};

const RELOCATED = {
  libraryRoot: "/Volumes/Models 1/hf/hub",
  hfHome: "/Volumes/Models 1/hf",
  status: { schemaVersion: 1, probeStatus: "available", available: true },
};

function errorResponse(status, body) {
  return { ok: false, status, json: async () => body, text: async () => JSON.stringify(body) };
}

describe("App model library relocation + restart disclosure (sc-19709)", () => {
  let container;
  let root;
  let imageJobPosts;
  let relocateRequests;
  let runningJobs;

  beforeEach(() => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    imageJobPosts = [];
    relocateRequests = [];
    runningJobs = [];
    invoke.mockReset();
    // The real command always answers with `hfHomeDefault` even before a library is configured,
    // so the fixture has to as well: the persist step refuses outright without a previous location.
    invoke.mockImplementation((command) => {
      if (command === "get_storage_setup") {
        return Promise.resolve({
          setupCompleted: true,
          hfHome: "/Volumes/Models/hf",
          hfHomeDefault: "/Users/me/.cache/huggingface",
        });
      }
      if (command === "choose_folder") return Promise.resolve("/Volumes/Models 1/hf");
      return Promise.resolve(null);
    });
    vi.stubGlobal(
      "fetch",
      vi.fn((url, options = {}) => {
        const path = new URL(url).pathname;
        const method = options.method ?? "GET";
        if (path.endsWith("/health")) {
          return Promise.resolve(response({ status: "ok", authRequired: false }));
        }
        if (path.endsWith("/access")) return Promise.resolve(response({ authRequired: false }));
        if (path.endsWith("/jobs/events/ticket")) {
          return Promise.resolve(response({ ticket: "stream-ticket" }));
        }
        if (path.endsWith("/projects")) {
          return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
        }
        if (path.endsWith("/ui-preferences")) {
          return Promise.resolve(
            response({ embedWorkflowInImages: true, workflowEmbedNoticeSeen: true }),
          );
        }
        if (path.endsWith("/models")) return Promise.resolve(response([MODEL]));
        if (path.endsWith("/jobs") && method === "GET") {
          return Promise.resolve(response(runningJobs));
        }
        if (path.endsWith("/model-library")) {
          return Promise.resolve(
            response({
              schemaVersion: 1,
              configuredLibraryPath: "/Volumes/Models/hf/hub",
              expectedLibrary: null,
              probeStatus: "unavailable",
              available: false,
            }),
          );
        }
        if (path.endsWith("/model-library/relocate")) {
          relocateRequests.push(JSON.parse(options.body));
          return Promise.resolve(response(RELOCATED));
        }
        if (path.endsWith("/image/jobs") && method === "POST") {
          imageJobPosts.push(JSON.parse(options.body));
          return Promise.resolve(errorResponse(503, UNAVAILABLE_BODY));
        }
        return Promise.resolve(response([]));
      }),
    );
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  function dialogs() {
    return [...document.body.querySelectorAll(".model-library-modal")];
  }

  function button(label) {
    return dialogs()
      .flatMap((dialog) => [...dialog.querySelectorAll("button")])
      .find((item) => item.textContent === label);
  }

  function status() {
    return dialogs()
      .map((dialog) => dialog.querySelector('[role="status"]')?.textContent ?? "")
      .join(" ");
  }

  // Submit a generation (refused by the seam), then answer the prompt by picking a new library.
  async function submitAndChooseLocation(prompt = "a lighthouse at dusk") {
    const form = document.body.querySelector(".image-studio form");
    const promptField = field(form, "Prompt") ?? form.querySelector("textarea");
    await changeField(promptField, prompt);
    await act(async () => form.requestSubmit());
    await settle();

    expect(button("Choose a different library location")).toBeTruthy();
    await act(async () => button("Choose a different library location").click());
    await settle();
  }

  async function relocate() {
    root = createRoot(container);
    await act(async () => root.render(<App />));
    await settle();

    const imageNav = [...document.body.querySelectorAll(".nav-label")].find(
      (item) => item.textContent === "Image",
    );
    await act(async () => imageNav.closest("button").click());
    await settle();

    await submitAndChooseLocation();
  }

  it("persists through the shell, then discloses the restart and performs it exactly once", async () => {
    await relocate();

    // Validated with nothing written FIRST, then adopted — so an ordinary refusal could never
    // leave the shell's persisted location and the server's binding disagreeing.
    expect(relocateRequests).toEqual([
      { path: "/Volumes/Models 1/hf", dryRun: true },
      { path: "/Volumes/Models 1/hf" },
    ]);
    // Persistence goes through the app's existing library-path configuration command.
    expect(invoke).toHaveBeenCalledWith("set_model_library", {
      path: "/Volumes/Models 1/hf",
    });

    // The blocked prompt is gone; the disclosure has taken its place.
    expect(button("Connect drive and retry")).toBeUndefined();
    const disclosure = dialogs()[0];
    expect(disclosure.textContent).toContain("after it restarts");
    expect(disclosure.textContent).toContain("/Volumes/Models 1/hf");

    const restart = button("Restart now");
    await act(async () => {
      restart.click();
      restart.click();
    });
    await settle();
    const restarts = invoke.mock.calls.filter(([command]) => command === "restart_app");
    expect(restarts).toHaveLength(1);
  });

  it("Later dismisses the disclosure and leaves no queued generation", async () => {
    await relocate();
    const posts = imageJobPosts.length;

    await act(async () => button("Later").click());
    await settle();

    expect(dialogs()).toHaveLength(0);
    expect(invoke.mock.calls.filter(([command]) => command === "restart_app")).toHaveLength(0);
    // The relocation stays durable, and the abandoned submission never fires.
    expect(imageJobPosts).toHaveLength(posts);
    expect(document.body.textContent).toContain("restart SceneWorks to apply it");
    // The notice outlives the dialog, so the abandoned generation is still accounted for.
    expect(document.body.textContent).toContain("was not queued");
  });

  // Reading the previous location is what makes the relocation UNDOABLE. If that read fails there
  // is no undo, so the shell must not be written at all — otherwise a later re-bind failure would
  // leave the shell's copy changed while the prompt claimed the previous location was still in use.
  it("refuses the relocation before any durable write when the previous location cannot be read", async () => {
    invoke.mockImplementation((command) => {
      if (command === "get_storage_setup") {
        return Promise.reject(new Error("settings file is unreadable"));
      }
      if (command === "choose_folder") return Promise.resolve("/Volumes/Models 1/hf");
      return Promise.resolve(null);
    });
    await relocate();

    // The shell was never written, and the server was never re-bound: only the dry-run probe ran.
    expect(invoke.mock.calls.filter(([command]) => command === "set_model_library")).toEqual([]);
    expect(relocateRequests).toEqual([{ path: "/Volumes/Models 1/hf", dryRun: true }]);

    // No restart disclosure — nothing was relocated.
    expect(button("Restart now")).toBeUndefined();
    // …and the message matches the state the user is actually in.
    const prompt = dialogs()[0];
    expect(prompt.textContent).toContain("previous location is still in use");
    expect(prompt.textContent).not.toContain("could not restore the previous location");
    expect(prompt.textContent).toContain("could not be read");
    // The blocked generation is still there to resume, and was never queued behind the failure.
    expect(button("Connect drive and retry")).toBeTruthy();
    expect(imageJobPosts).toHaveLength(1);
  });

  // A restart that never launches must leave the disclosure usable. Before this, the once-only
  // latch and the restarting state were never released, so the dialog stayed on "Restarting
  // SceneWorks…" with both buttons dead — for this relocation AND every later one.
  it("recovers when the restart fails, and a later relocation can restart again", async () => {
    let restartAttempts = 0;
    invoke.mockImplementation((command) => {
      if (command === "get_storage_setup") {
        return Promise.resolve({ setupCompleted: true, hfHomeDefault: "/Volumes/Models/hf" });
      }
      if (command === "choose_folder") return Promise.resolve("/Volumes/Models 1/hf");
      if (command === "restart_app") {
        restartAttempts += 1;
        return restartAttempts <= 2
          ? Promise.reject(new Error("relaunch refused"))
          : Promise.resolve(null);
      }
      return Promise.resolve(null);
    });
    await relocate();

    await act(async () => button("Restart now").click());
    await settle();
    expect(restartAttempts).toBe(1);
    // The failure is surfaced, and the dialog is actionable again rather than stuck.
    expect(document.body.textContent).toContain("SceneWorks could not restart");
    expect(status()).not.toContain("Restarting SceneWorks");
    expect(button("Restart now").disabled).toBe(false);
    expect(button("Later").disabled).toBe(false);

    // Retrying from the very same disclosure works.
    await act(async () => button("Restart now").click());
    await settle();
    expect(restartAttempts).toBe(2);

    // Deferring is still available after the failures, so the disclosure can be dismissed at all.
    expect(button("Later").disabled).toBe(false);
    await act(async () => button("Later").click());
    await settle();
    expect(dialogs()).toHaveLength(0);

    // And a whole new relocation opens a LIVE dialog, not the dead restarting one.
    await submitAndChooseLocation("a second lighthouse");
    expect(button("Restart now")).toBeTruthy();
    expect(button("Restart now").disabled).toBe(false);
    expect(status()).not.toContain("Restarting SceneWorks");
    await act(async () => button("Restart now").click());
    await settle();
    expect(restartAttempts).toBe(3);
    // This one launched, so the dialog correctly stays in its restarting state.
    expect(status()).toContain("Restarting SceneWorks");
  });

  it("withholds Restart now while a job is running", async () => {
    runningJobs = [
      {
        id: "job-running",
        projectId: "project-default",
        type: "image_generate",
        status: "running",
        createdAt: "2026-08-17T12:00:00Z",
      },
    ];
    await relocate();

    expect(button("Restart now").disabled).toBe(true);
    await act(async () => button("Restart now").click());
    expect(invoke.mock.calls.filter(([command]) => command === "restart_app")).toHaveLength(0);
    expect(dialogs()[0].textContent).toContain("still running");
  });
});
