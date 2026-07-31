import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PoseLibraryPicker } from "./PoseLibraryPicker.jsx";

const POSES = Array.from({ length: 65 }, (_, index) => ({
  id: `pose-${index}`,
  label: `Pose ${index}`,
  category: "test",
  keypoints: [],
  preview: `pose-${index}.png`,
}));

vi.mock("../poseLibrary.js", () => ({
  usePoseLibrary: () => ({
    poses: POSES,
    categories: ["test"],
    loading: false,
    error: null,
  }),
}));

describe("PoseLibraryPicker pose-count ceiling", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  it("disables only new selections at the ceiling and explains how to continue", async () => {
    const onToggle = vi.fn();
    await act(async () => {
      root.render(
        <PoseLibraryPicker
          onClear={vi.fn()}
          onToggle={onToggle}
          selectedIds={POSES.slice(0, 64).map((pose) => pose.id)}
        />,
      );
    });
    const sixtyFifth = container.querySelector('[aria-label="Select pose Pose 64"]');
    expect(sixtyFifth).not.toBeNull();
    expect(sixtyFifth.disabled).toBe(true);
    expect(sixtyFifth.title).toContain("Maximum of 64 poses per job");
    expect(container.querySelector('[role="status"]').textContent).toContain(
      "Maximum of 64 poses per job. Deselect one to choose another.",
    );
    await act(async () => {
      sixtyFifth.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(onToggle).not.toHaveBeenCalled();

    const selected = container.querySelector('[aria-label="Deselect pose Pose 0"]');
    expect(selected.disabled).toBe(false);
    await act(async () => {
      selected.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(onToggle).toHaveBeenCalledWith("pose-0");
  });
});
