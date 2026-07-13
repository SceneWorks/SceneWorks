import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ControlPanel } from "./ControlPanel.jsx";

// The pose picker pulls the library over the API; stub it so the panel test stays focused on the mode
// gating + the canny/depth upload toggle. The picker itself is covered by PoseLibraryPicker tests.
vi.mock("../poseLibrary.js", () => ({
  usePoseLibrary: () => ({
    poses: [
      {
        id: "p1",
        label: "Standing",
        category: "Basics",
        keypoints: [],
        preview: "p1.png",
      },
    ],
    categories: ["Basics"],
    loading: false,
    error: null,
  }),
  useUserPoseLoader: () => () => Promise.resolve([]),
}));
// The source picker opens a modal on click; for the panel test we only need its presence/label and the
// onChange seam, so stub it to a lightweight control surfacing the same props.
vi.mock("./AssetPicker.jsx", () => ({
  ImageEditSourcePickerField: ({ label }) => (
    <div data-testid="control-image-picker">{label}</div>
  ),
}));

describe("ControlPanel (sc-8245 gating + toggle)", () => {
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

  async function render(ui) {
    await act(async () => root.render(ui));
  }

  // The panel is collapsed by default (large, optional section); clicking the toggle expands it so
  // the gated inner content (tabs, pose/upload, slider) mounts. Most tests below assert on that
  // content, so they render then expand.
  async function toggle() {
    const head = container.querySelector(".control-panel-toggle");
    await act(async () => {
      head.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
  }

  async function renderExpanded(ui) {
    await render(ui);
    await toggle();
  }

  const tabLabels = () =>
    [...container.querySelectorAll(".control-mode-tab")].map((b) =>
      b.textContent.trim(),
    );

  const tabByLabel = (label) =>
    [...container.querySelectorAll(".control-mode-tab")].find(
      (b) => b.textContent.trim() === label,
    );

  function baseProps(overrides = {}) {
    return {
      supportedModes: ["pose", "canny", "depth"],
      controlMode: "pose",
      onControlModeChange: vi.fn(),
      selectedPoseIds: [],
      onTogglePose: vi.fn(),
      onClearPoses: vi.fn(),
      loadUserPoses: vi.fn(),
      poseBlockText: null,
      controlImageAssetId: "",
      onControlImageChange: vi.fn(),
      controlImagePassthrough: false,
      onControlImagePassthroughChange: vi.fn(),
      controlImageAssets: [],
      importAsset: vi.fn(),
      projectId: "proj_1",
      characters: [],
      controlScaleConfig: {
        label: "Control strength",
        default: 0.9,
        min: 0,
        max: 2,
        step: 0.05,
      },
      controlScale: 0.9,
      onControlScaleChange: vi.fn(),
      ...overrides,
    };
  }

  it("is collapsed by default and expands on clicking the toggle", async () => {
    await render(<ControlPanel {...baseProps()} />);
    // Collapsed: the disclosure toggle is present but the gated inner content is not mounted.
    const head = container.querySelector(".control-panel-toggle");
    expect(head).not.toBeNull();
    expect(head.getAttribute("aria-expanded")).toBe("false");
    expect(tabLabels()).toEqual([]);
    expect(container.querySelector(".control-mode-tabs")).toBeNull();

    await toggle();
    expect(head.getAttribute("aria-expanded")).toBe("true");
    expect(tabLabels()).toEqual(["Pose", "Canny", "Depth"]);

    // Clicking again collapses it back.
    await toggle();
    expect(head.getAttribute("aria-expanded")).toBe("false");
    expect(tabLabels()).toEqual([]);
  });

  it("shows all three tabs when the backbone supports pose+canny+depth", async () => {
    await renderExpanded(<ControlPanel {...baseProps()} />);
    expect(tabLabels()).toEqual(["Pose", "Canny", "Depth"]);
  });

  it("shows only the pose tab for a pose-only backbone", async () => {
    await renderExpanded(
      <ControlPanel
        {...baseProps({ supportedModes: ["pose"], controlMode: "pose" })}
      />,
    );
    expect(tabLabels()).toEqual(["Pose"]);
  });

  it("renders nothing when the backbone supports no control modes", async () => {
    await render(<ControlPanel {...baseProps({ supportedModes: [] })} />);
    expect(container.querySelector(".control-panel")).toBeNull();
  });

  it("pose mode shows the pose library, not the control-image upload", async () => {
    await renderExpanded(
      <ControlPanel {...baseProps({ controlMode: "pose" })} />,
    );
    expect(container.querySelector(".pose-library")).not.toBeNull();
    expect(
      container.querySelector('[data-testid="control-image-picker"]'),
    ).toBeNull();
  });

  it("canny/depth mode shows the control-image upload + the preprocess toggle", async () => {
    await renderExpanded(
      <ControlPanel {...baseProps({ controlMode: "canny" })} />,
    );
    expect(
      container.querySelector('[data-testid="control-image-picker"]'),
    ).not.toBeNull();
    const toggle = container.querySelector('input[type="checkbox"]');
    expect(toggle).not.toBeNull();
    expect(toggle.checked).toBe(false); // preprocess (derive) by default
  });

  it("the preprocess/use-as-is toggle reports its checked state", async () => {
    const onControlImagePassthroughChange = vi.fn();
    await renderExpanded(
      <ControlPanel
        {...baseProps({
          controlMode: "depth",
          onControlImagePassthroughChange,
        })}
      />,
    );
    const toggle = container.querySelector('input[type="checkbox"]');
    await act(async () => {
      // A click toggles the (unchecked) box → React fires onChange with checked=true.
      toggle.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(onControlImagePassthroughChange).toHaveBeenCalledWith(true);
  });

  it("falls back to the first supported mode when the active pick is unsupported", async () => {
    // controlMode is a stranded "canny" but the backbone only supports pose → pose renders.
    await renderExpanded(
      <ControlPanel
        {...baseProps({ supportedModes: ["pose"], controlMode: "canny" })}
      />,
    );
    expect(tabLabels()).toEqual(["Pose"]);
    expect(container.querySelector(".pose-library")).not.toBeNull();
  });

  it("clicking a mode tab calls onControlModeChange", async () => {
    const onControlModeChange = vi.fn();
    await renderExpanded(
      <ControlPanel {...baseProps({ onControlModeChange })} />,
    );
    await act(async () => {
      tabByLabel("Depth").dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });
    expect(onControlModeChange).toHaveBeenCalledWith("depth");
  });

  it("binds the control-scale slider to controlScale + its config range", async () => {
    await renderExpanded(
      <ControlPanel {...baseProps({ controlScale: 1.25 })} />,
    );
    const slider = container.querySelector('input[type="range"]');
    expect(slider).not.toBeNull();
    expect(slider.value).toBe("1.25");
    expect(slider.min).toBe("0");
    expect(slider.max).toBe("2");
    expect(slider.step).toBe("0.05");
  });
});
