import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../api.js", async (importOriginal) => ({
  ...(await importOriginal()),
  apiFetch: apiFetchMock,
}));

import { useTraining } from "./useTraining.js";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

describe("useTraining deleteTrainingDataset", () => {
  let container;
  let root;
  let latest;
  let projectRef;
  let setError;
  let setJobs;

  function Harness({ project }) {
    latest = useTraining({
      token: "token",
      activeProject: project,
      activeProjectRef: projectRef,
      setError,
      setJobs,
    });
    return null;
  }

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    apiFetchMock.mockReset();
    setError = vi.fn();
    setJobs = vi.fn();
    projectRef = { current: { id: "project_a" } };
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    await act(async () => {
      root.render(<Harness project={{ id: "project_a" }} />);
    });
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  it("issues a DELETE for the dataset, then refreshes the list", async () => {
    // Seed a two-dataset list so a successful refresh visibly replaces state.
    await act(async () => {
      latest.setTrainingDatasets([{ id: "ds_1" }, { id: "ds_2" }]);
    });

    const deleted = { id: "ds_1", status: "deleted" };
    apiFetchMock.mockResolvedValueOnce(deleted); // DELETE
    apiFetchMock.mockResolvedValueOnce([{ id: "ds_2" }]); // refresh GET

    let result;
    await act(async () => {
      result = await latest.deleteTrainingDataset("ds_1");
    });

    expect(result).toEqual(deleted);
    expect(apiFetchMock).toHaveBeenCalledTimes(2);
    expect(apiFetchMock.mock.calls[0]).toEqual([
      "/api/v1/projects/project_a/training/datasets/ds_1",
      "token",
      { method: "DELETE" },
    ]);
    // The follow-up refresh hits the list endpoint and replaces the cached datasets.
    expect(apiFetchMock.mock.calls[1][0]).toBe("/api/v1/projects/project_a/training/datasets");
    expect(latest.trainingDatasets).toEqual([{ id: "ds_2" }]);
  });

  it("encodes the dataset id in the request path", async () => {
    apiFetchMock.mockResolvedValueOnce({ id: "a/b", status: "deleted" });
    apiFetchMock.mockResolvedValueOnce([]);

    await act(async () => {
      await latest.deleteTrainingDataset("a/b");
    });

    expect(apiFetchMock.mock.calls[0][0]).toBe("/api/v1/projects/project_a/training/datasets/a%2Fb");
  });

  it("throws without calling the API when no dataset id is given", async () => {
    await expect(latest.deleteTrainingDataset()).rejects.toThrow("Select a training dataset first.");
    expect(apiFetchMock).not.toHaveBeenCalled();
  });

  it("skips the refresh when the originating project is no longer active", async () => {
    const request = deferred();
    apiFetchMock.mockReturnValueOnce(request.promise); // DELETE stays pending

    const pending = latest.deleteTrainingDataset("ds_1");

    // Switch the active project before the DELETE resolves — the stale refresh must not fire.
    projectRef.current = { id: "project_b" };
    await act(async () => {
      root.render(<Harness project={{ id: "project_b" }} />);
    });

    let result;
    await act(async () => {
      request.resolve({ id: "ds_1", status: "deleted" });
      result = await pending;
    });

    expect(result).toEqual({ id: "ds_1", status: "deleted" });
    // Only the DELETE ran; no follow-up list refresh for the abandoned project.
    expect(apiFetchMock).toHaveBeenCalledTimes(1);
  });
});
