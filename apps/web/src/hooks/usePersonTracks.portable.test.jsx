import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const validate = vi.hoisted(() => vi.fn());
const detect = vi.hoisted(() => vi.fn());
const jobs = vi.hoisted(() => vi.fn());
vi.mock("../portableValidation.js", () => ({ validatePortableRequestNow: validate }));
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
  validate.mockReset();
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

it("refuses both person job bodies before POST with the shared typed result", async () => {
  const refusal = { code: "invalid_generation_option", field: "advanced.confidence", detail: "Invalid confidence" };
  validate.mockReturnValue({ state: "invalid", refusal });
  const setError = vi.fn();
  act(() => root.render(<Harness setError={setError} />));
  await act(async () => {
    expect(await actions.createPersonDetectionJob({ sourceAssetId: "clip-1" })).toBeNull();
    expect(await actions.createPersonTrackJob({ sourceAssetId: "clip-1", detection: { id: "person-1" } })).toBeNull();
  });
  expect(validate.mock.calls).toEqual([
    [{ kind: "personDetect", payload: { sourceAssetId: "clip-1", requestedGpu: "auto" } }],
    [{ kind: "personTrack", payload: { sourceAssetId: "clip-1", detection: { id: "person-1" }, requestedGpu: "auto" } }],
  ]);
  expect(detect).not.toHaveBeenCalled();
  expect(jobs).not.toHaveBeenCalled();
  expect(setError).toHaveBeenLastCalledWith("Invalid confidence");
});
