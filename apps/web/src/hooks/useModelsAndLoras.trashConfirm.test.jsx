// sc-12068: when a model/LoRA delete can't move artifacts to the OS trash, the hook
// falls back to a permanent delete only after confirming. That confirm now routes
// through the shared desktop-safe appConfirm helper (a real in-app dialog) instead of
// the raw window.confirm, which silently no-ops inside the Tauri WebView. These tests
// drive the hook's delete actions with an apiFetch that reports `trashUnavailable` and
// assert the guard fired via appConfirm (danger tone), gating the permanent retry.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { appConfirmMock } = vi.hoisted(() => ({ appConfirmMock: vi.fn(async () => true) }));
vi.mock("../appConfirm.jsx", () => ({
  appConfirm: appConfirmMock,
  useConfirm: () => appConfirmMock,
  ConfirmHost: () => null,
}));

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../api.js", () => ({
  apiFetch: (...args) => apiFetchMock(...args),
  isAbortError: () => false,
}));

import { useModelsAndLoras } from "./useModelsAndLoras.js";

let container;
let root;
let hookApi;

function Harness() {
  hookApi = useModelsAndLoras({
    token: "tok",
    activeProject: { id: "proj-1" },
    activeProjectRef: { current: { id: "proj-1" } },
    setError: () => {},
    setJobs: () => {},
    setActiveView: () => {},
    refreshData: async () => {},
    refreshDataWithLoraOverlay: async () => {},
  });
  return null;
}

beforeEach(async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  appConfirmMock.mockClear();
  appConfirmMock.mockResolvedValue(true);
  apiFetchMock.mockReset();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root.render(<Harness />);
  });
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

describe("useModelsAndLoras trash-unavailable permanent-delete confirm (sc-12068)", () => {
  it("model delete: prompts via appConfirm (danger) and retries permanently when accepted", async () => {
    apiFetchMock
      .mockResolvedValueOnce({ trashUnavailable: true })
      .mockResolvedValueOnce({ removedManifestEntry: true });

    let result;
    await act(async () => {
      result = await hookApi.deleteModel({ id: "m1" });
    });

    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    expect(appConfirmMock.mock.calls[0][0]).toMatchObject({ tone: "danger" });
    expect(apiFetchMock).toHaveBeenCalledTimes(2);
    expect(apiFetchMock.mock.calls[1][0]).toContain("permanent=true");
    expect(result).toMatchObject({ removedManifestEntry: true });
  });

  it("model delete: cancels (no permanent retry) when the confirm is declined", async () => {
    appConfirmMock.mockResolvedValue(false);
    apiFetchMock.mockResolvedValueOnce({ trashUnavailable: true });

    let result;
    await act(async () => {
      result = await hookApi.deleteModel({ id: "m1" });
    });

    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    expect(apiFetchMock).toHaveBeenCalledTimes(1); // no permanent retry issued
    expect(result).toEqual({ cancelled: true });
  });

  it("lora delete: prompts via appConfirm (danger) and retries permanently when accepted", async () => {
    apiFetchMock
      .mockResolvedValueOnce({ trashUnavailable: true })
      .mockResolvedValueOnce({ removedManifestEntry: true });

    let result;
    await act(async () => {
      result = await hookApi.deleteLora({ id: "l1", scope: "global" });
    });

    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    expect(appConfirmMock.mock.calls[0][0]).toMatchObject({ tone: "danger" });
    expect(apiFetchMock).toHaveBeenCalledTimes(2);
    expect(apiFetchMock.mock.calls[1][0]).toContain("permanent=true");
    expect(result).toMatchObject({ removedManifestEntry: true });
  });

  it("does not prompt when the trash delete succeeds outright", async () => {
    apiFetchMock.mockResolvedValueOnce({ removedManifestEntry: true });

    await act(async () => {
      await hookApi.deleteModel({ id: "m1" });
    });

    expect(appConfirmMock).not.toHaveBeenCalled();
    expect(apiFetchMock).toHaveBeenCalledTimes(1);
  });
});
