import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PoseLibraryPicker } from "./PoseLibraryPicker.jsx";

const libraryMock = vi.hoisted(() => ({
  state: { poses: [], categories: [], loading: false, error: null },
}));

const POSES = Array.from({ length: 65 }, (_, index) => ({
  id: `pose-${index}`,
  label: `Pose ${index}`,
  category: "test",
  keypoints: [],
  preview: `pose-${index}.png`,
}));

vi.mock("../poseLibrary.js", () => ({
  usePoseLibrary: ({ extraPoses = [] } = {}) => {
    const poses = [...libraryMock.state.poses, ...extraPoses];
    return {
      ...libraryMock.state,
      poses,
      categories: [...new Set(poses.map((pose) => pose.category))],
    };
  },
}));

describe("PoseLibraryPicker pose-count ceiling", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    libraryMock.state = { poses: POSES, categories: ["test"], loading: false, error: null };
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

  it.each([
    ["loading", { loading: true, error: null }],
    ["error", { loading: false, error: "offline" }],
    ["empty", { loading: false, error: null }],
  ])("keeps honest shared-workflow placeholders visible while the saved library is %s", async (_name, state) => {
    libraryMock.state = { poses: [], categories: [], ...state };
    const shared = {
      id: "workflow:launch:0",
      label: "Shared workflow pose 1",
      category: "Shared workflow",
      keypoints: [[1, 2]],
      source: "workflow",
    };
    await act(async () => {
      root.render(
        <PoseLibraryPicker
          extraPoses={[shared]}
          onToggle={vi.fn()}
          preferredCategory="Shared workflow"
          selectedIds={[shared.id]}
        />,
      );
    });

    expect(container.querySelector('[aria-label="Shared workflow pose 1 preview unavailable"]')).not.toBeNull();
    expect(container.textContent).toContain("Shared workflow pose");
  });

  it("opens the Shared workflow category and keeps all 65 selected legacy records deselectable", async () => {
    libraryMock.state = { poses: [], categories: [], loading: false, error: null };
    const shared = Array.from({ length: 65 }, (_, index) => ({
      id: `workflow:legacy:${index}`,
      label: `Shared workflow pose ${index + 1}`,
      category: "Shared workflow",
      keypoints: [[index, index]],
      source: "workflow",
    }));
    const onToggle = vi.fn();
    await act(async () => {
      root.render(
        <PoseLibraryPicker
          categoryFilter
          extraPoses={shared}
          onToggle={onToggle}
          preferredCategory="Shared workflow"
          selectedIds={shared.map((pose) => pose.id)}
        />,
      );
    });

    expect(container.querySelector(".pose-category-name").textContent).toBe("Shared workflow");
    expect(container.querySelectorAll(".pose-thumb")).toHaveLength(65);
    const last = container.querySelector('[aria-label="Deselect pose Shared workflow pose 65"]');
    expect(last.disabled).toBe(false);
    await act(async () => last.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onToggle).toHaveBeenCalledWith("workflow:legacy:64");
  });
});
