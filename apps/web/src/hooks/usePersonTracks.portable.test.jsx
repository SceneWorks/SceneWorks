import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const detect = vi.hoisted(() => vi.fn());
const jobs = vi.hoisted(() => vi.fn());
vi.mock("../api/personTracksResources.js", () => ({
  personTracksApi: { list: vi.fn(), detect, jobs, corrections: vi.fn() },
}));

import { usePersonTracks } from "./usePersonTracks.js";

let actions;
function Harness({ setError }) {
  actions = usePersonTracks({
    token: "token", activeProject: { id: "project-1" },
    activeProjectRef: { current: { id: "project-1" } },
    setError, requestedGpu: "auto", setActiveView: vi.fn(),
  });
  return null;
}

let host;
let root;
beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  detect.mockReset();
  jobs.mockReset();
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

it("surfaces authoritative person job refusals from the final POST", async () => {
  const refusal = { code: "invalid_generation_option", field: "advanced.confidence", detail: "Invalid confidence" };
  detect.mockRejectedValue(Object.assign(new Error(refusal.detail), refusal));
  jobs.mockRejectedValue(Object.assign(new Error(refusal.detail), refusal));
  const setError = vi.fn();
  act(() => root.render(<Harness setError={setError} />));
  await act(async () => {
    expect(await actions.createPersonDetectionJob({ sourceAssetId: "clip-1" })).toBeNull();
    expect(await actions.createPersonTrackJob({ sourceAssetId: "clip-1", detection: { id: "person-1" } })).toBeNull();
  });
  expect(detect).toHaveBeenCalledWith("project-1", "token", expect.objectContaining({
    body: JSON.stringify({ sourceAssetId: "clip-1", requestedGpu: "auto" }),
  }));
  expect(jobs).toHaveBeenCalledWith("project-1", "token", expect.objectContaining({
    body: JSON.stringify({ sourceAssetId: "clip-1", detection: { id: "person-1" }, requestedGpu: "auto" }),
  }));
  expect(setError).toHaveBeenLastCalledWith("Invalid confidence");
});
