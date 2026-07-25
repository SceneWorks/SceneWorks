import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../api.js", async (importOriginal) => ({
  ...(await importOriginal()),
  apiFetch: apiFetchMock,
}));

import { useSavedVoices } from "./useSavedVoices.js";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

describe("useSavedVoices project scoping", () => {
  let container;
  let root;
  let latest;
  let projectRef;
  let setError;

  function Harness({ project }) {
    latest = useSavedVoices({
      token: "token",
      activeProject: project,
      activeProjectRef: projectRef,
      setError,
    });
    return null;
  }

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    apiFetchMock.mockReset();
    setError = vi.fn();
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

  it("splices a created voice when the project remains active", async () => {
    const created = { id: "voice_a", name: "Narrator" };
    apiFetchMock.mockResolvedValue(created);

    let result;
    await act(async () => {
      result = await latest.createSavedVoice({
        name: "Narrator",
        referenceAudioAssetId: "asset_a",
      });
    });

    expect(result).toBe(created);
    expect(latest.savedVoices).toEqual([created]);
    expect(setError).toHaveBeenLastCalledWith("");
  });

  it("drops a successful create response after switching projects", async () => {
    const request = deferred();
    apiFetchMock.mockReturnValue(request.promise);
    const pending = latest.createSavedVoice({
      name: "Old project voice",
      referenceAudioAssetId: "asset_a",
    });

    projectRef.current = { id: "project_b" };
    await act(async () => {
      root.render(<Harness project={{ id: "project_b" }} />);
      latest.setSavedVoices([{ id: "voice_b", name: "Current project voice" }]);
    });

    let result;
    await act(async () => {
      request.resolve({ id: "voice_a", name: "Old project voice" });
      result = await pending;
    });

    expect(result).toBeNull();
    expect(latest.savedVoices).toEqual([{ id: "voice_b", name: "Current project voice" }]);
    expect(setError).not.toHaveBeenCalled();
  });

  it("drops a create error after switching projects", async () => {
    const request = deferred();
    apiFetchMock.mockReturnValue(request.promise);
    const pending = latest.createSavedVoice({
      name: "Old project voice",
      referenceAudioAssetId: "asset_a",
    });

    projectRef.current = { id: "project_b" };
    await act(async () => {
      root.render(<Harness project={{ id: "project_b" }} />);
    });
    let result;
    await act(async () => {
      request.reject(new Error("old project failed"));
      result = await pending;
    });

    expect(result).toBeNull();
    expect(setError).not.toHaveBeenCalled();
  });

  it("does not delete a same-id voice from the project selected after the request began", async () => {
    const request = deferred();
    apiFetchMock.mockReturnValue(request.promise);
    const pending = latest.deleteSavedVoice("voice_shared");

    projectRef.current = { id: "project_b" };
    await act(async () => {
      root.render(<Harness project={{ id: "project_b" }} />);
      latest.setSavedVoices([{ id: "voice_shared", name: "Project B voice" }]);
    });
    await act(async () => {
      request.resolve({});
      await pending;
    });

    expect(latest.savedVoices).toEqual([{ id: "voice_shared", name: "Project B voice" }]);
    expect(setError).not.toHaveBeenCalled();
  });

  it("reports a delete failure while the originating project remains active", async () => {
    apiFetchMock.mockRejectedValue(new Error("delete failed"));

    let result;
    await act(async () => {
      result = await latest.deleteSavedVoice("voice_a");
    });

    expect(result).toBeNull();
    expect(setError).toHaveBeenLastCalledWith("delete failed");
  });
});
