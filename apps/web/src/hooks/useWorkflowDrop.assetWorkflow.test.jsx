// The imported-asset half of the workflow offer (sc-15952, epic 15945).
//
// Two things here are the acceptance criteria in the story that are easiest to fake. The first is
// "once installed, re-runs resolution without a reload" — a test that clicked Download and asserted
// a POST would pass with a panel that never changed, so `re-resolves the open offer` drives the
// catalog forward the way a completed job does and reads the REPORT back out of the offer. The
// second is that a substitute model reaches the launch as the user's own choice and never as a
// default.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const fetchAssetWorkflow = vi.fn();
const inspectWorkflowFile = vi.fn();
vi.mock("../api.js", async () => {
  const actual = await vi.importActual("../api.js");
  return {
    ...actual,
    fetchAssetWorkflow: (...args) => fetchAssetWorkflow(...args),
    inspectWorkflowFile: (...args) => inspectWorkflowFile(...args),
  };
});

import { ApiError } from "../api.js";
import { useWorkflowDrop } from "./useWorkflowDrop.js";

const IMPORTED_ASSET = { id: "asset_1", extra: { importedWorkflow: { sceneworksWorkflow: "image" } } };

function workflowBody(resolutionOverrides = {}) {
  return {
    status: "workflow",
    workflow: {
      sceneworksWorkflow: "image",
      schemaVersion: 1,
      mode: "text_to_image",
      model: "krea_2_turbo",
      prompt: "a lighthouse in heavy fog",
      seed: 4242,
      width: 1024,
      height: 1024,
      count: 1,
      advanced: { steps: 28 },
    },
    resolution: {
      model: {
        slug: "krea_2_turbo",
        state: "installable",
        catalogId: "krea_2_turbo",
        name: "Krea 2 Turbo",
        detail: "Krea 2 Turbo is in the model catalog but is not downloaded on this machine.",
        install: { method: "POST", path: "/api/v1/models/krea_2_turbo/download" },
      },
      loras: [],
      styles: [],
      inputs: [],
      omitted: [],
      runnable: false,
      inputImagesRequired: 0,
      ...resolutionOverrides,
    },
  };
}

const RESOLVED_MODEL = {
  slug: "krea_2_turbo",
  state: "resolved",
  catalogId: "krea_2_turbo",
  name: "Krea 2 Turbo",
  detail: "Krea 2 Turbo is installed on this machine.",
};

describe("useWorkflowDrop — the imported-asset offer", () => {
  let container;
  let root;
  let hook;

  function Harness(props) {
    hook = useWorkflowDrop(props);
    return null;
  }

  const render = async (props) => {
    await act(async () =>
      root.render(<Harness enabled projectId="project_1" token="t" {...props} />),
    );
  };

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    fetchAssetWorkflow.mockReset();
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

  it("opens the same offer from an asset, keyed by asset rather than by file", async () => {
    fetchAssetWorkflow.mockResolvedValue(workflowBody());
    await render({});
    await act(async () => hook.offerAssetWorkflow(IMPORTED_ASSET));
    expect(fetchAssetWorkflow).toHaveBeenCalledWith(
      "project_1",
      "asset_1",
      expect.objectContaining({ token: "t" }),
    );
    expect(hook.offer.assetId).toBe("asset_1");
    expect(hook.offer.file).toBeNull();
    expect(hook.offer.share.prompt).toBe("a lighthouse in heavy fog");
    expect(hook.offer.report.model.state).toBe("installable");
    // No upload happened: the envelope was already on the asset and the report is a GET.
    expect(inspectWorkflowFile).not.toHaveBeenCalled();
  });

  it("names a failure instead of staying silent, unlike the drop path", async () => {
    // A drop that fails invisibly is the correct fallback — nothing was asked for. This button was
    // rendered BECAUSE the record carries an envelope, so silence would be a control that did
    // nothing.
    fetchAssetWorkflow.mockRejectedValue(
      new ApiError("SceneWorks could not read that recipe back.", 500, {}),
    );
    await render({});
    await act(async () => hook.offerAssetWorkflow(IMPORTED_ASSET));
    expect(hook.offer.error).toContain("could not read that recipe back");
    expect(hook.offer.share).toBeNull();
  });

  it("says so when the record and the button disagree", async () => {
    fetchAssetWorkflow.mockResolvedValue({
      status: "no_workflow",
      workflow: null,
      resolution: null,
      detail: "This image carries no SceneWorks workflow.",
    });
    await render({});
    await act(async () => hook.offerAssetWorkflow(IMPORTED_ASSET));
    expect(hook.offer.error).toContain("no SceneWorks workflow");
  });

  // ---- The AC that is easiest to fake ----------------------------------------------------

  it("re-resolves the open offer when a download completes, with no reload and no polling", async () => {
    fetchAssetWorkflow.mockResolvedValueOnce(workflowBody());
    await render({ catalogRevision: "m:krea_2_turbo:missing" });
    await act(async () => hook.offerAssetWorkflow(IMPORTED_ASSET));
    expect(hook.offer.report.model.state).toBe("installable");
    expect(fetchAssetWorkflow).toHaveBeenCalledTimes(1);

    // A re-render with the SAME catalog must not refetch — otherwise every unrelated App render
    // sends the panel back to the API.
    await render({ catalogRevision: "m:krea_2_turbo:missing" });
    expect(fetchAssetWorkflow).toHaveBeenCalledTimes(1);

    // Now the download completes: `useJobEvents` refetches the catalog, App's `catalogRevision`
    // changes, and the panel re-resolves itself.
    fetchAssetWorkflow.mockResolvedValueOnce(workflowBody({ model: RESOLVED_MODEL, runnable: true }));
    await render({ catalogRevision: "m:krea_2_turbo:installed" });
    expect(fetchAssetWorkflow).toHaveBeenCalledTimes(2);
    expect(hook.offer.report.model.state).toBe("resolved");
    expect(hook.offer.report.runnable).toBe(true);
    // The envelope is a fact about the file and cannot have changed under it.
    expect(hook.offer.share.prompt).toBe("a lighthouse in heavy fog");
  });

  it("re-resolves a DROPPED file's offer the same way, through inspect", async () => {
    inspectWorkflowFile.mockResolvedValue(workflowBody());
    await render({ catalogRevision: "before" });
    const file = new File(
      [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2])],
      "shared.png",
      { type: "image/png" },
    );
    await act(async () => {
      await hook.handleDroppedFile(file);
    });
    expect(hook.offer.report.model.state).toBe("installable");
    inspectWorkflowFile.mockResolvedValue(workflowBody({ model: RESOLVED_MODEL, runnable: true }));
    await render({ catalogRevision: "after" });
    expect(hook.offer.report.model.state).toBe("resolved");
    expect(fetchAssetWorkflow).not.toHaveBeenCalled();
  });

  it("leaves the last good report standing when a re-resolution fails", async () => {
    fetchAssetWorkflow.mockResolvedValueOnce(workflowBody());
    await render({ catalogRevision: "before" });
    await act(async () => hook.offerAssetWorkflow(IMPORTED_ASSET));
    fetchAssetWorkflow.mockRejectedValueOnce(new ApiError("network", 500, {}));
    await render({ catalogRevision: "after" });
    // Blanking a panel the user is reading, because a refresh failed, tells them nothing true.
    expect(hook.offer.report.model.state).toBe("installable");
    expect(hook.offer.error).toBe("");
  });

  // ---- Install ---------------------------------------------------------------------------

  it("starts an install through the Model Manager's own flow and marks it queued", async () => {
    const installModel = vi.fn().mockResolvedValue({ id: "job_1" });
    const installLora = vi.fn().mockResolvedValue({ id: "job_2" });
    fetchAssetWorkflow.mockResolvedValue(workflowBody());
    await render({ installModel, installLora });
    await act(async () => hook.offerAssetWorkflow(IMPORTED_ASSET));
    await act(async () => hook.missing.onInstall({ kind: "model", id: "krea_2_turbo" }));
    // Through `createModelDownloadJob`, not a bare POST to `install.path` — that is what puts the
    // job in the queue AND makes its completion refetch the catalog that re-resolves this panel.
    expect(installModel).toHaveBeenCalledWith({ id: "krea_2_turbo" });
    expect(hook.missing.installed["model:krea_2_turbo"]).toBe(true);
    expect(hook.missing.installing["model:krea_2_turbo"]).toBe(false);
    // A second click is a second download job for a model already being fetched.
    await act(async () => hook.missing.onInstall({ kind: "model", id: "krea_2_turbo" }));
    expect(installModel).toHaveBeenCalledTimes(1);
  });

  // ---- What reaches the launch ------------------------------------------------------------

  it("carries the user's chosen model, and nothing when they chose none", async () => {
    const launchRecipe = vi.fn().mockResolvedValue({ ok: true });
    fetchAssetWorkflow.mockResolvedValue(
      workflowBody({
        model: {
          slug: "someones_private_model",
          state: "missing",
          detail: "No model matching `someones_private_model` is in this install's catalog.",
        },
      }),
    );
    const models = [{ id: "z_image_turbo", name: "Z-Image Turbo" }];
    await render({ launchRecipe, models });
    await act(async () => hook.offerAssetWorkflow(IMPORTED_ASSET));

    // Nothing chosen → the recipe names NO model, so the studio keeps whatever it is on and the
    // panel's own gate is what stops the launch. Never a fallback.
    await act(async () => hook.useWorkflow());
    expect(launchRecipe.mock.calls[0][0].recipe.model).toBeUndefined();

    await render({ launchRecipe, models });
    await act(async () => hook.offerAssetWorkflow(IMPORTED_ASSET));
    await act(async () => hook.missing.onSubstituteModel("z_image_turbo"));
    await act(async () => hook.useWorkflow());
    expect(launchRecipe.mock.calls[1][0].recipe.model).toBe("z_image_turbo");
  });

  it("drops a chosen model the catalog stopped offering before the click", async () => {
    const launchRecipe = vi.fn().mockResolvedValue({ ok: true });
    fetchAssetWorkflow.mockResolvedValue(
      workflowBody({
        model: { slug: "someones_private_model", state: "missing", detail: "…" },
      }),
    );
    await render({ launchRecipe, models: [{ id: "z_image_turbo" }] });
    await act(async () => hook.offerAssetWorkflow(IMPORTED_ASSET));
    await act(async () => hook.missing.onSubstituteModel("z_image_turbo"));
    // The model is deleted from the Models page while the panel sits open.
    await render({ launchRecipe, models: [] });
    await act(async () => hook.useWorkflow());
    expect(launchRecipe.mock.calls[0][0].recipe.model).toBeUndefined();
  });

  it("carries the picked images to the launch, source and references apart", async () => {
    const launchRecipe = vi.fn().mockResolvedValue({ ok: true });
    fetchAssetWorkflow.mockResolvedValue(
      workflowBody({
        model: RESOLVED_MODEL,
        runnable: true,
        inputImagesRequired: 3,
        inputs: [
          { kind: "source", count: 1, state: "userSupplied", detail: "Needs a source image." },
          { kind: "reference", count: 2, state: "userSupplied", detail: "Needs 2 reference images." },
        ],
      }),
    );
    await render({ launchRecipe });
    await act(async () => hook.offerAssetWorkflow(IMPORTED_ASSET));
    await act(async () => {
      hook.missing.onPickInput("source-0-0", "the_source");
      hook.missing.onPickInput("reference-1-0", "ref_one");
      hook.missing.onPickInput("reference-1-1", "ref_two");
    });
    await act(async () => hook.useWorkflow());
    const launch = launchRecipe.mock.calls[0][0];
    // `sourceAssetId` is a LAUNCH field (the studio's edit source); the references ride on the
    // recipe. And `assetId` stays null — the imported image is the recipe's OUTPUT, so passing it
    // would silently start the edit from the finished image the sender rendered.
    expect(launch.sourceAssetId).toBe("the_source");
    expect(launch.assetId ?? null).toBeNull();
    expect(launch.recipe.rawAdapterSettings.referenceAssetIds).toEqual(["ref_one", "ref_two"]);
    expect(launch.recipe.rawAdapterSettings.referenceAssetId).toBe("ref_one");
  });

  it("forgets the previous offer's answers when a new one opens", async () => {
    // A substitute model left over from the last image would be the silent substitution this epic
    // exists to prevent, wearing a fresh panel's clothes.
    fetchAssetWorkflow.mockResolvedValue(
      workflowBody({ model: { slug: "x", state: "missing", detail: "…" } }),
    );
    await render({ models: [{ id: "z_image_turbo" }] });
    await act(async () => hook.offerAssetWorkflow(IMPORTED_ASSET));
    await act(async () => hook.missing.onSubstituteModel("z_image_turbo"));
    await act(async () => hook.missing.onPickInput("source-0-0", "the_source"));
    expect(hook.missing.substituteModelId).toBe("z_image_turbo");

    await act(async () => hook.offerAssetWorkflow({ id: "asset_2" }));
    expect(hook.missing.substituteModelId).toBe("");
    expect(hook.missing.inputPicks).toEqual({});
  });

  it("does nothing without a project to read the asset from", async () => {
    await render({ projectId: null });
    await act(async () => hook.offerAssetWorkflow(IMPORTED_ASSET));
    expect(fetchAssetWorkflow).not.toHaveBeenCalled();
    expect(hook.offer).toBeNull();
  });
});
