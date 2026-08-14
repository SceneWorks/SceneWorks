// "Drop a shared image anywhere → Workflow found → prefill Image Studio" (sc-15951, epic 15945).
//
// The behaviour under test is mostly about what does NOT happen: every foreign PNG in the world
// answers `no_workflow`, and that path has to stay byte-for-byte the swallow it was before this
// hook existed — no panel, no spinner, no error band.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The transport is stubbed at `inspectWorkflowFile` (not at `apiFetch`): the hook's contract is
// the endpoint's `{ status, workflow, resolution }` body and its typed `ApiError`s, and pinning
// it to a FormData round trip would test axum's multipart rather than this hook.
const inspectWorkflowFile = vi.fn();
vi.mock("../api.js", async () => {
  const actual = await vi.importActual("../api.js");
  return { ...actual, inspectWorkflowFile: (...args) => inspectWorkflowFile(...args) };
});

import { ApiError } from "../api.js";
import { inspectFailureMessage, useWorkflowDrop } from "./useWorkflowDrop.js";

const PNG_HEAD = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function pngFile(name = "shared.png") {
  return new File([new Uint8Array([...PNG_HEAD, 1, 2, 3, 4])], name, { type: "image/png" });
}

function workflowBody(overrides = {}) {
  return {
    status: "workflow",
    workflow: {
      sceneworksWorkflow: "image",
      schemaVersion: 1,
      mode: "text_to_image",
      model: "krea_2_turbo",
      prompt: "a lighthouse in heavy fog",
      negativePrompt: "",
      seed: 4242,
      width: 1024,
      height: 1024,
      count: 4,
      advanced: { steps: 28 },
    },
    resolution: {
      model: {
        slug: "krea_2_turbo",
        state: "resolved",
        catalogId: "krea_2_turbo",
        name: "Krea 2 Turbo",
        detail: "Krea 2 Turbo is installed on this machine.",
      },
      loras: [],
      styles: [],
      inputs: [],
      omitted: [],
      runnable: true,
      inputImagesRequired: 0,
    },
    ...overrides,
  };
}

describe("useWorkflowDrop", () => {
  let container;
  let root;
  let hook;

  function Harness(props) {
    hook = useWorkflowDrop(props);
    return null;
  }

  const render = async (props) => {
    await act(async () => root.render(<Harness {...props} />));
  };

  // Start a drop and let the prescreen (an async range read off the file) finish, WITHOUT waiting
  // for the inspect it then starts — which is the state the abort tests need to observe.
  const startDrop = async (file) => {
    await act(async () => {
      hook.handleDroppedFile(file);
      // Wait until the inspect has actually STARTED, rather than for a fixed tick. The prescreen
      // in front of it is an async range read off the file, and jsdom's `Blob` has no
      // `arrayBuffer`, so it goes through `FileReader` — a macrotask whose completion one
      // `setTimeout(0)` does not reliably outlast. That made both abort tests below intermittently
      // read `signal` as null. Bounded, so a drop that never reaches the endpoint still fails its
      // assertion instead of hanging the suite.
      for (let tick = 0; tick < 50 && inspectWorkflowFile.mock.calls.length === 0; tick += 1) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    });
  };

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    inspectWorkflowFile.mockReset();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  it("opens the offer when the dropped image carries a workflow", async () => {
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    await render({ projectId: "project_1", token: "t" });
    await act(async () => {
      await hook.handleDroppedFile(pngFile());
    });
    expect(hook.offer?.share?.prompt).toBe("a lighthouse in heavy fog");
    expect(hook.offer?.report?.model?.state).toBe("resolved");
    const [sent, options] = inspectWorkflowFile.mock.calls[0];
    expect(sent.name).toBe("shared.png");
    expect(options.projectId).toBe("project_1");
    expect(options.token).toBe("t");
    // A cancellable upload: the endpoint stages up to 512 MiB before it reads a byte, so an
    // abandoned inspect has to STOP rather than merely have its answer ignored.
    expect(options.signal).toBeInstanceOf(AbortSignal);
    expect(options.signal.aborted).toBe(false);
  });

  it("aborts the in-flight inspect when a second drop supersedes it", async () => {
    let firstSignal = null;
    inspectWorkflowFile.mockImplementation(
      (file, options) =>
        new Promise((resolve) => {
          if (!firstSignal) {
            firstSignal = options.signal;
            return; // never settles — the first upload is still running
          }
          resolve(workflowBody());
        }),
    );
    await render({ projectId: "project_1", token: "t" });
    await startDrop(pngFile("first.png"));
    expect(firstSignal?.aborted).toBe(false);
    await act(async () => {
      await hook.handleDroppedFile(pngFile("second.png"));
    });
    expect(firstSignal.aborted).toBe(true);
  });

  it("aborts the in-flight inspect on unmount", async () => {
    let signal = null;
    inspectWorkflowFile.mockImplementation(
      (file, options) =>
        new Promise(() => {
          signal = options.signal;
        }),
    );
    await render({ projectId: "project_1", token: "t" });
    await startDrop(pngFile());
    expect(signal?.aborted).toBe(false);
    // Unmount the harness (rather than the root, which afterEach still owns) — the cleanup
    // effect is what has to abandon the request.
    await act(async () => root.render(null));
    expect(signal.aborted).toBe(true);
  });

  it("aborts the in-flight inspect when the panel is dismissed", async () => {
    let signal = null;
    inspectWorkflowFile.mockImplementation(
      (file, options) =>
        new Promise(() => {
          signal = options.signal;
        }),
    );
    await render({ projectId: "project_1", token: "t" });
    await startDrop(pngFile());
    await act(async () => {
      hook.dismiss();
    });
    expect(signal.aborted).toBe(true);
  });

  it("does nothing at all for an image with no workflow — the common case", async () => {
    inspectWorkflowFile.mockResolvedValue({
      status: "no_workflow",
      workflow: null,
      resolution: null,
      detail: "This image carries no SceneWorks workflow…",
    });
    await render({});
    await act(async () => {
      await hook.handleDroppedFile(pngFile("holiday.png"));
    });
    expect(hook.offer).toBeNull();
  });

  it("never asks the server about a file that is not a PNG", async () => {
    await render({});
    await act(async () => {
      await hook.handleDroppedFile(new File(["not an image"], "notes.txt", { type: "text/plain" }));
    });
    expect(inspectWorkflowFile).not.toHaveBeenCalled();
    expect(hook.offer).toBeNull();
  });

  it("does nothing while disabled", async () => {
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    await render({ enabled: false });
    await act(async () => {
      await hook.handleDroppedFile(pngFile());
    });
    expect(inspectWorkflowFile).not.toHaveBeenCalled();
    expect(hook.offer).toBeNull();
  });

  it("shows the sentence for a file that claims a workflow it cannot read", async () => {
    inspectWorkflowFile.mockRejectedValue(
      new ApiError("This PNG could not be read far enough to look for a workflow.", {
        status: 422,
        code: "workflow_inspect_unreadable",
      }),
    );
    await render({});
    await act(async () => {
      await hook.handleDroppedFile(pngFile());
    });
    expect(hook.offer?.error).toContain("could not be read far enough");
    expect(hook.offer?.share).toBeNull();
  });

  it("stays silent for a failure that says nothing about the file", async () => {
    // A network drop, an axum plain-text multipart rejection (no `code` at all), or a `not_png`
    // that slipped past the prescreen. None of them is worth a modal the user did not ask for.
    for (const error of [
      new ApiError("Failed to fetch", {}),
      new ApiError("Request failed with 400", { status: 400 }),
      new ApiError("Not a PNG", { status: 400, code: "workflow_inspect_not_png" }),
      new ApiError("bad multipart", { status: 400, code: "workflow_inspect_bad_multipart" }),
    ]) {
      inspectWorkflowFile.mockReset();
      inspectWorkflowFile.mockRejectedValue(error);
      await render({});
      await act(async () => {
        await hook.handleDroppedFile(pngFile());
      });
      expect(hook.offer).toBeNull();
    }
  });

  it("lets a second drop supersede a slow first one", async () => {
    let releaseFirst;
    let firstReached;
    const firstInFlight = new Promise((resolve) => {
      firstReached = resolve;
    });
    inspectWorkflowFile
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            releaseFirst = () =>
              resolve(
                workflowBody({ workflow: { ...workflowBody().workflow, prompt: "stale" } }),
              );
            firstReached();
          }),
      )
      .mockResolvedValueOnce(workflowBody());
    await render({});
    let firstDrop;
    await act(async () => {
      firstDrop = hook.handleDroppedFile(pngFile("first.png"));
      // Wait until the first inspect is genuinely in flight, so the race under test is
      // "a response arrives late", not "a drop was superseded before it left".
      await firstInFlight;
    });
    await act(async () => {
      await hook.handleDroppedFile(pngFile("second.png"));
    });
    await act(async () => {
      releaseFirst();
      await firstDrop;
    });
    expect(hook.offer?.share?.prompt).toBe("a lighthouse in heavy fog");
  });

  it("hands the recipe to the shared launch and closes", async () => {
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    const launchRecipe = vi.fn(async () => true);
    await render({ launchRecipe });
    await act(async () => {
      await hook.handleDroppedFile(pngFile());
    });
    await act(async () => {
      await hook.useWorkflow();
    });
    expect(launchRecipe).toHaveBeenCalledTimes(1);
    const { recipe, replaySeed } = launchRecipe.mock.calls[0][0];
    expect(recipe.model).toBe("krea_2_turbo");
    expect(recipe.prompt).toBe("a lighthouse in heavy fog");
    // The shared image's own seed, and ONE image — not the batch of four it came from.
    expect(replaySeed).toBe("4242");
    expect(recipe.normalizedSettings.count).toBe(1);
    expect(hook.offer).toBeNull();
  });

  it("imports through the ordinary upload path and keeps the offer open", async () => {
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    const importAsset = vi.fn(async () => ({ id: "asset_9" }));
    await render({ importAsset });
    const file = pngFile();
    await act(async () => {
      await hook.handleDroppedFile(file);
    });
    await act(async () => {
      await hook.importImage();
    });
    expect(importAsset).toHaveBeenCalledWith(file, { throwOnError: true });
    expect(hook.importState.done).toBe(true);
    // "too", not "instead": the workflow is still there to take into the studio.
    expect(hook.offer?.share).toBeTruthy();
  });

  it("reports an import failure without losing the offer", async () => {
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    await render({ importAsset: vi.fn(async () => { throw new Error("Create or open a project first."); }) });
    await act(async () => {
      await hook.handleDroppedFile(pngFile());
    });
    await act(async () => {
      await hook.importImage();
    });
    expect(hook.importState.error).toBe("Create or open a project first.");
    expect(hook.offer?.share).toBeTruthy();
  });

  it("dismisses the offer, and a late inspect cannot reopen it", async () => {
    let release;
    let reached;
    const inFlight = new Promise((resolve) => {
      reached = resolve;
    });
    inspectWorkflowFile.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = () => resolve(workflowBody());
          reached();
        }),
    );
    await render({});
    let pending;
    await act(async () => {
      pending = hook.handleDroppedFile(pngFile());
      await inFlight;
    });
    await act(async () => hook.dismiss());
    await act(async () => {
      release();
      await pending;
    });
    expect(hook.offer).toBeNull();
  });

  it("drops the whole offer when the active project changes under it", async () => {
    // The panel's answers are project-scoped and its report is not. `inputPicks` holds asset ids
    // out of the project that was open when they were picked, and App's own picker list re-filters
    // on `activeProject.id` — so a pick made before the switch survives into a panel now offering
    // a different library, and `useWorkflow` would hand `launchImageRecipe` a `sourceAssetId` the
    // new project does not contain. The report is stale too: the asset route resolves against the
    // project's LoRA catalog.
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    await render({ projectId: "project_1", token: "t" });
    await act(async () => {
      await hook.handleDroppedFile(pngFile());
    });
    await act(async () => hook.missing.onPickInput("source-0-0", "asset_from_project_1"));
    expect(hook.offer?.share).toBeTruthy();
    expect(hook.missing.inputPicks).toEqual({ "source-0-0": "asset_from_project_1" });

    await render({ projectId: "project_2", token: "t" });
    expect(hook.offer).toBeNull();
    expect(hook.missing.inputPicks).toEqual({});
  });

  it("keeps the offer across a re-render that did NOT change the project", async () => {
    // The other half: the guard fires on a real CHANGE, not on every render. A panel that vanished
    // whenever App re-rendered would be indistinguishable from one that never opened.
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    await render({ projectId: "project_1", token: "t" });
    await act(async () => {
      await hook.handleDroppedFile(pngFile());
    });
    await render({ projectId: "project_1", token: "t", catalogRevision: "" });
    expect(hook.offer?.share?.prompt).toBe("a lighthouse in heavy fog");
  });

  describe("a queued install that fails is not a permanently dead button", () => {
    const installableBody = () =>
      workflowBody({
        resolution: {
          ...workflowBody().resolution,
          model: {
            slug: "fixture_model",
            state: "installable",
            catalogId: "fixture_model",
            name: "Fixture Model",
            detail: "Fixture Model is in the model catalog but is not downloaded.",
            install: { method: "POST", path: "/api/v1/models/fixture_model/download" },
          },
          runnable: false,
        },
      });

    const queueInstall = async (props) => {
      inspectWorkflowFile.mockResolvedValue(installableBody());
      await render(props);
      await act(async () => {
        await hook.handleDroppedFile(pngFile());
      });
      await act(async () => {
        await hook.missing.onInstall({ kind: "model", id: "fixture_model" });
      });
    };

    it("latches Queued while the job is merely running", async () => {
      await queueInstall({
        projectId: "project_1",
        token: "t",
        installModel: vi.fn(async () => ({ id: "job_1", status: "running" })),
      });
      expect(hook.missing.installed["model:fixture_model"]).toBe(true);
      expect(hook.missing.installFailed["model:fixture_model"]).toBeUndefined();
    });

    it("un-latches it once that job has failed", async () => {
      // A failed download changes NO catalog, so `catalogRevision` is byte-identical to what it
      // was before the button was pressed and the re-resolution effect never fires. Without this
      // the row read "Queued" with its button disabled until the panel was closed and reopened —
      // the only retry there was, and one the user has to guess at.
      const installModel = vi.fn(async () => ({ id: "job_1", status: "queued" }));
      await queueInstall({ projectId: "project_1", token: "t", installModel });
      expect(hook.missing.installed["model:fixture_model"]).toBe(true);

      await render({
        projectId: "project_1",
        token: "t",
        installModel,
        failedInstallJobIds: ["job_1"],
      });
      expect(hook.missing.installed["model:fixture_model"]).toBe(false);
      expect(hook.missing.installFailed["model:fixture_model"]).toBe(true);

      // And the button works again: the guard that refused a second press reads the same map.
      await act(async () => {
        await hook.missing.onInstall({ kind: "model", id: "fixture_model" });
      });
      expect(installModel).toHaveBeenCalledTimes(2);
    });

    it("leaves a job it has no id for latched, because unknown is not failed", async () => {
      // A stubbed installer, or an API that answered without an id. Re-enabling here would offer a
      // retry for a download that may well be running.
      await queueInstall({
        projectId: "project_1",
        token: "t",
        installModel: vi.fn(async () => ({})),
        failedInstallJobIds: ["job_1"],
      });
      expect(hook.missing.installed["model:fixture_model"]).toBe(true);
      expect(hook.missing.installFailed["model:fixture_model"]).toBeUndefined();
    });

    // sc-17227 review HIGH 1. The install used to call `start({ id: row.id })` — a STUB. The
    // licence gate lives inside `createModelDownloadJob` and reads the entry's own `gated` /
    // `requiresLicenseAcknowledgment`, so a stub made the predicate false: the gate could not fire
    // and `licenseAcknowledged` was never put on the request body. For a
    // `requiresLicenseAcknowledgment` model the server still refused (a raw error rather than the
    // guided message); for a `gated` one BOTH halves missed — the client by the stub, the server
    // by its deliberate `gated` exclusion — so with a saved HF credential the weights landed with
    // no acknowledgment at all.
    it("hands the installer the real catalog entry, not an { id } stub (sc-17227)", async () => {
      const installModel = vi.fn(async () => ({ id: "job_1", status: "queued" }));
      const entry = {
        id: "fixture_model",
        name: "Fixture Model",
        // A VIDEO model, deliberately: `models` is the IMAGE picker's list, so resolving against
        // it would still hand a stub for exactly the kind of model MiniMax-H3 is.
        type: "video",
        requiresLicenseAcknowledgment: true,
        licenseUrl: "https://huggingface.co/MiniMaxAI/MiniMax-H3",
      };
      await queueInstall({
        projectId: "project_1",
        token: "t",
        installModel,
        models: [],
        catalogModels: [{ id: "some_other_model", type: "image" }, entry],
      });
      expect(installModel).toHaveBeenCalledTimes(1);
      // The exact entry object, so every flag the gate reads is present — not a lookalike.
      expect(installModel.mock.calls[0][0]).toBe(entry);
    });

    it("keeps the old { id } call when the catalog has no entry for it (sc-17227)", async () => {
      // An id the catalog does not carry cannot download anything (`POST
      // /api/v1/models/:id/download` 404s on an unknown id), so there is nothing to gate and
      // nothing to resolve. That path stays byte-identical to the behavior before this fix.
      const installModel = vi.fn(async () => ({ id: "job_1", status: "queued" }));
      await queueInstall({
        projectId: "project_1",
        token: "t",
        installModel,
        catalogModels: [{ id: "unrelated", type: "image" }],
      });
      expect(installModel).toHaveBeenCalledWith({ id: "fixture_model" });
    });
  });
});

describe("inspectFailureMessage", () => {
  it("passes through the sentences the API wrote for a person", () => {
    expect(
      inspectFailureMessage(
        new ApiError("Two workflow chunks.", { status: 422, code: "workflow_inspect_unreadable" }),
      ),
    ).toBe("Two workflow chunks.");
  });

  it("is null for everything else, including a body with no code at all", () => {
    expect(inspectFailureMessage(null)).toBeNull();
    expect(inspectFailureMessage(new ApiError("boom", { status: 400 }))).toBeNull();
    expect(
      inspectFailureMessage(new ApiError("", { status: 500, code: "workflow_inspect_read_failed" })),
    ).toBeNull();
  });
});
