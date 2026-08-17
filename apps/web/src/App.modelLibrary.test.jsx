// sc-19709: the unavailable-model-library recovery, driven through the REAL app shell.
//
// The stale-UI case the story names: the catalog said the library was there, the drive went away
// before the user pressed Generate, and the submission comes back with the typed 503. The app must
// answer with an actionable prompt naming the model and the expected location — not a raw error —
// and a reconnect must resume that submission EXACTLY ONCE, no matter how many retry events land.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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
  downloadable: true,
  // The catalog was read while the drive was still attached.
  modelAvailability: "external_ready",
};

const UNAVAILABLE_BODY = {
  detail:
    "Model 'z_image_turbo' is installed on an external model library that is currently unavailable.",
  code: "external_model_library_unavailable",
  context: {
    schemaVersion: 1,
    availability: "installed_external_unavailable",
    modelId: "z_image_turbo",
    modelName: "Z-Image Turbo",
    configuredLibraryPath: "/Volumes/SceneWorks Models/hf/hub",
    expectedLibraryPath: "/Volumes/SceneWorks Models/hf/hub",
    expectedVolumeId: "macos-volume:abc",
  },
};

function errorResponse(status, body) {
  return {
    ok: false,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

describe("App unavailable model library prompt (sc-19709)", () => {
  let container;
  let root;
  let imageJobPosts;
  let libraryAvailable;
  let probeCalls;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    imageJobPosts = [];
    probeCalls = 0;
    libraryAvailable = false;
    global.fetch = vi.fn((url, options = {}) => {
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
        // The workflow-embed disclosure gates the very first generation; mark it answered so the
        // submission actually reaches the API and this test stays about the model library.
        return Promise.resolve(
          response({ embedWorkflowInImages: true, workflowEmbedNoticeSeen: true }),
        );
      }
      if (path.endsWith("/models")) return Promise.resolve(response([MODEL]));
      if (path.endsWith("/model-library")) {
        probeCalls += 1;
        return Promise.resolve(
          response({
            schemaVersion: 1,
            configuredLibraryPath: "/Volumes/SceneWorks Models/hf/hub",
            expectedLibrary: null,
            probeStatus: libraryAvailable ? "available" : "unavailable",
            available: libraryAvailable,
          }),
        );
      }
      if (path.endsWith("/image/jobs") && method === "POST") {
        imageJobPosts.push(JSON.parse(options.body));
        if (!libraryAvailable) {
          return Promise.resolve(errorResponse(503, UNAVAILABLE_BODY));
        }
        return Promise.resolve(
          response({
            id: `image-job-${imageJobPosts.length}`,
            status: "queued",
            createdAt: "2026-08-17T12:00:00Z",
          }),
        );
      }
      return Promise.resolve(response([]));
    });
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  function dialog() {
    return document.body.querySelector(".model-library-modal");
  }

  function button(label) {
    return [...(dialog()?.querySelectorAll("button") ?? [])].find(
      (item) => item.textContent === label,
    );
  }

  async function generate() {
    root = createRoot(container);
    await act(async () => root.render(<App />));
    await settle();

    const imageNav = [...document.body.querySelectorAll(".nav-label")].find(
      (item) => item.textContent === "Image",
    );
    expect(imageNav).toBeTruthy();
    await act(async () => imageNav.closest("button").click());
    await settle();

    const form = document.body.querySelector(".image-studio form");
    expect(form).toBeTruthy();
    const prompt = field(form, "Prompt") ?? form.querySelector("textarea");
    expect(prompt).toBeTruthy();
    await changeField(prompt, "a lighthouse at dusk");
    await act(async () => form.requestSubmit());
    await settle();
  }

  it("prompts with the typed context, then resumes the blocked submission exactly once", async () => {
    await generate();

    expect(imageJobPosts).toHaveLength(1);
    expect(dialog()).toBeTruthy();
    expect(dialog().textContent).toContain("Z-Image Turbo");
    expect(dialog().textContent).toContain("/Volumes/SceneWorks Models/hf/hub");
    // The raw refusal sentence must not have leaked into the notice band.
    expect(document.body.textContent).not.toContain(UNAVAILABLE_BODY.detail);

    // Retrying while the drive is still missing must not submit anything.
    await act(async () => button("Connect drive and retry").click());
    await settle();
    expect(imageJobPosts).toHaveLength(1);
    expect(dialog()).toBeTruthy();
    expect(probeCalls).toBeGreaterThan(0);

    // The drive comes back. Two retry clicks land before the first probe settles.
    libraryAvailable = true;
    await act(async () => {
      button("Connect drive and retry").click();
      button("Connect drive and retry").click();
    });
    await settle();

    expect(imageJobPosts).toHaveLength(2);
    expect(imageJobPosts[1]).toEqual(imageJobPosts[0]);
    expect(dialog()).toBeNull();
  });

  it("cancel closes the prompt and leaves nothing queued", async () => {
    await generate();
    expect(dialog()).toBeTruthy();

    await act(async () => button("Cancel").click());
    await settle();
    expect(dialog()).toBeNull();

    // Even once the library is back, the abandoned submission must never appear.
    libraryAvailable = true;
    await settle();
    expect(imageJobPosts).toHaveLength(1);
  });

  it("offers server guidance instead of a folder picker outside the desktop shell", async () => {
    await generate();
    expect(button("Choose a different library location")).toBeUndefined();
    expect(dialog().textContent).toContain("desktop setting");
  });
});
