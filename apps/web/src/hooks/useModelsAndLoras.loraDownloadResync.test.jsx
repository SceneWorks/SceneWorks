// A LoRA row's install badge is a snapshot from the last catalog fetch, but the API probes
// `installState` live from the HF cache on every request. Anything that fills the cache
// out-of-band — the on-demand pull at first generation, a second client, a manual fetch —
// flips the server to "installed" while an open catalog view still renders "Not Installed".
// Clicking Download then 400s, and the old code surfaced that detail verbatim, so the user
// saw "LoRA is already installed" sitting next to a "Not Installed" badge.
//
// The fix keys off the API's `lora_already_installed` code (not the prose detail) and
// resyncs the catalog so the badge corrects itself. These tests pin both halves: the
// benign-disagreement path refetches and stays silent, and an ordinary failure still
// surfaces its error and does NOT refetch.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../api.js", () => ({
  apiFetch: (...args) => apiFetchMock(...args),
  isAbortError: () => false,
}));

import { useModelsAndLoras } from "./useModelsAndLoras.js";

let container;
let root;
let hookApi;
let loraErrors;

function Harness() {
  hookApi = useModelsAndLoras({
    token: "tok",
    activeProject: { id: "proj-1" },
    activeProjectRef: { current: { id: "proj-1" } },
    setError: () => {},
    setLoraError: (value) => loraErrors.push(value),
    setJobs: () => {},
    setActiveView: () => {},
    refreshData: async () => {},
    refreshDataWithLoraOverlay: async () => {},
  });
  return null;
}

/** The download POST paths this hook issues, so a test can count catalog refetches. */
const downloadCalls = (calls) => calls.filter(([path]) => path.endsWith("/download"));
const catalogCalls = (calls) => calls.filter(([path]) => path.startsWith("/api/v1/loras?") || path === "/api/v1/loras");

beforeEach(async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  apiFetchMock.mockReset();
  loraErrors = [];
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

describe("createLoraDownloadJob stale-badge resync", () => {
  it("refetches the catalog and stays silent when the server says it is already installed", async () => {
    const alreadyInstalled = Object.assign(new Error("LoRA is already installed"), {
      status: 400,
      detail: "LoRA is already installed",
      code: "lora_already_installed",
    });
    apiFetchMock.mockImplementation(async (path) => {
      if (path.endsWith("/download")) throw alreadyInstalled;
      // The resync refetch: the row now carries the server's real state.
      return [{ id: "scail2_lightning", installState: "installed" }];
    });

    let result;
    await act(async () => {
      result = await hookApi.createLoraDownloadJob({ id: "scail2_lightning" });
    });

    expect(result).toBe(null);
    expect(downloadCalls(apiFetchMock.mock.calls)).toHaveLength(1);
    // The badge only corrects if the catalog is actually refetched.
    expect(catalogCalls(apiFetchMock.mock.calls)).toHaveLength(1);
    // The contradiction must never reach the user as an error.
    expect(loraErrors).not.toContain("LoRA is already installed");
  });

  it("still surfaces an ordinary download failure without refetching", async () => {
    const boom = Object.assign(new Error("disk full"), { status: 500, detail: "disk full" });
    apiFetchMock.mockImplementation(async (path) => {
      if (path.endsWith("/download")) throw boom;
      return [];
    });

    let result;
    await act(async () => {
      result = await hookApi.createLoraDownloadJob({ id: "scail2_lightning" });
    });

    expect(result).toBe(null);
    expect(loraErrors).toContain("disk full");
    expect(catalogCalls(apiFetchMock.mock.calls)).toHaveLength(0);
  });
});
