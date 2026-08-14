import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { poseRecordToPayload, usePoseLibrary, workflowPoseRecords } from "./poseLibrary.js";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("workflowPoseRecords", () => {
  it("creates unique session-only picker records without losing duplicate coordinates", () => {
    const pose = {
      keypoints: [[0.1, 0.2, 0.9]],
      hands: [[[0.3, 0.4]]],
      face: [[0.5, 0.6]],
    };
    const records = workflowPoseRecords([pose, pose], "launch-7");

    expect(records).toHaveLength(2);
    expect(records.map((record) => record.id)).toEqual([
      "workflow:launch-7:0",
      "workflow:launch-7:1",
    ]);
    expect(records.map(({ keypoints, hands, face }) => ({ keypoints, hands, face }))).toEqual([
      pose,
      pose,
    ]);
    expect(records.every((record) => record.source === "workflow")).toBe(true);
    expect(records.map(poseRecordToPayload)).toEqual([
      { id: "workflow:launch-7:0", ...pose },
      { id: "workflow:launch-7:1", ...pose },
    ]);
  });

  it("does not truncate a legacy selection above the current 64-pose ceiling", () => {
    const poses = Array.from({ length: 65 }, (_, index) => ({ keypoints: [[index, index + 1]] }));
    const records = workflowPoseRecords(poses, "legacy");

    expect(records).toHaveLength(65);
    expect(records.at(-1)).toMatchObject({
      id: "workflow:legacy:64",
      keypoints: [[64, 65]],
    });
  });

  it("returns no records for an omitted pose setting", () => {
    expect(workflowPoseRecords(undefined, "launch-8")).toEqual([]);
  });
});

describe("usePoseLibrary", () => {
  it("does not reload on a parent rerender when no extra poses were supplied", async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, json: async () => ({ poses: [] }) }));
    vi.stubGlobal("fetch", fetchMock);
    const container = document.createElement("div");
    const root = createRoot(container);

    function Probe({ tick }) {
      const state = usePoseLibrary();
      return React.createElement("output", null, `${tick}:${state.loading}`);
    }

    await act(async () => root.render(React.createElement(Probe, { tick: 1 })));
    await act(async () => {});
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await act(async () => root.render(React.createElement(Probe, { tick: 2 })));
    await act(async () => {});
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(async () => root.unmount());
  });
});
