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
    invoke.mockImplementation((command) => {
      if (command === "get_storage_setup") return Promise.resolve({ setupCompleted: true });
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

  async function relocate() {
    root = createRoot(container);
    await act(async () => root.render(<App />));
    await settle();

    const imageNav = [...document.body.querySelectorAll(".nav-label")].find(
      (item) => item.textContent === "Image",
    );
    await act(async () => imageNav.closest("button").click());
    await settle();

    const form = document.body.querySelector(".image-studio form");
    const prompt = field(form, "Prompt") ?? form.querySelector("textarea");
    await changeField(prompt, "a lighthouse at dusk");
    await act(async () => form.requestSubmit());
    await settle();

    expect(button("Choose a different library location")).toBeTruthy();
    await act(async () => button("Choose a different library location").click());
    await settle();
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
