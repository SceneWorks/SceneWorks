import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppContext } from "../context/AppContext.js";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";
import { RequiredModelsNotice } from "./RequiredModelsNotice.jsx";

function model(overrides = {}) {
  return {
    id: "dwpose_pose_detector",
    name: "DWPose Pose Detector",
    installState: "missing",
    downloadSizeLabel: "330 MB",
    ...overrides,
  };
}

describe("RequiredModelsNotice", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  // The live-job branch renders WorkerProgressCard, which reads the app context, so every render
  // goes through a provider (same shape as ModelAvailabilityGate.test.jsx).
  async function render(props = {}) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ workersById: new Map(), visibleWorkers: [] }}>
          <RequiredModelsNotice feature="Pose ControlNet training" {...props} />
        </AppContext.Provider>,
      );
    });
  }

  const btn = (text) =>
    [...document.body.querySelectorAll("button")].find((el) => el.textContent.trim() === text);

  // The whole point of the component: callers mount it unconditionally and let
  // missingRequiredModels decide, so an empty list must render nothing at all.
  it("renders nothing when no model is missing", async () => {
    await render({ models: [] });
    expect(container.querySelector(".required-models-notice")).toBeNull();
    expect(container.textContent).toBe("");
  });

  it("names the feature, the model and its size", async () => {
    await render({ models: [model()] });
    expect(container.textContent).toContain("Pose ControlNet training");
    expect(container.textContent).toContain("DWPose Pose Detector");
    expect(container.textContent).toContain("330 MB");
  });

  it("pluralizes for several missing models", async () => {
    await render({
      models: [model(), model({ id: "person_detector", name: "YOLO11m Person Detector" })],
    });
    expect(container.textContent).toContain("needs 2 models");
    expect(container.textContent).toContain("aren’t installed");
  });

  it("enqueues the download for the clicked model", async () => {
    const onDownload = vi.fn(async () => ({ id: "job_dl_1" }));
    await render({ models: [model()], onDownload });
    await click(btn("Download"));
    expect(onDownload).toHaveBeenCalledTimes(1);
    expect(onDownload.mock.calls[0][0]).toMatchObject({ id: "dwpose_pose_detector" });
  });

  // The no-downloadJobs path (DatasetCatalogsScreen, which stays off the live job context): without
  // the local state the button would look unclicked until a catalog refetch landed, inviting a
  // duplicate download.
  it("latches to Queued after a click when no job list is supplied", async () => {
    await render({ models: [model()], onDownload: vi.fn(async () => null) });
    await click(btn("Download"));
    expect(btn("Queued")).toBeTruthy();
    expect(btn("Queued").disabled).toBe(true);
  });

  it("shows live job status and progress when a job list is supplied", async () => {
    await render({
      models: [model()],
      downloadJobs: [
        {
          id: "job_dl_1",
          type: "model_download",
          status: "downloading",
          progress: 0.4,
          payload: { modelId: "dwpose_pose_detector" },
        },
      ],
      onDownload: vi.fn(),
    });
    expect(btn("Download")).toBeFalsy();
    expect(btn("downloading")).toBeTruthy();
  });

  // A completed job leaves the jobs list but `models` is what says whether it is still needed, so
  // a terminal job must not keep the row pinned.
  it("ignores a terminal download job", async () => {
    await render({
      models: [model()],
      downloadJobs: [
        {
          id: "job_dl_1",
          type: "model_download",
          status: "completed",
          payload: { modelId: "dwpose_pose_detector" },
        },
      ],
      onDownload: vi.fn(),
    });
    expect(btn("Download")).toBeTruthy();
  });

  it("surfaces a failed enqueue instead of silently doing nothing", async () => {
    const onDownload = vi.fn(async () => {
      throw new Error("worker offline");
    });
    await render({ models: [model()], onDownload });
    await click(btn("Download"));
    expect(container.textContent).toContain("worker offline");
  });

  it("disables Download when no handler is wired", async () => {
    await render({ models: [model()] });
    expect(btn("Download").disabled).toBe(true);
  });
});
