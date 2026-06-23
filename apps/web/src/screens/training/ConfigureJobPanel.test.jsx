import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { ConfigureJobPanel } from "./ConfigureJobPanel.jsx";

// The highest-stakes glue in sc-6534: the Train button must disable when the readiness gate is
// Blocked, and stay enabled otherwise. A wrong binding either blocks a trainable set or trains an
// untrainable one — neither is caught by the pure-helper or store tests. ConfigureJobPanel is
// presentational, so a minimal fixture (advanced/network/adapter sections off) mounts it cheaply.

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
});

function mount(node) {
  act(() => root.render(node));
}

const noop = () => {};

function baseProps(overrides = {}) {
  return {
    active: { id: "configure", title: "Configure training job" },
    setActiveView: noop,
    configReady: true,
    trainingTargetsError: "",
    trainingPresetsError: "",
    configError: "",
    configMessage: "",
    selectedTarget: { id: "t1", name: "Target One", baseModel: "base" },
    setSelectedTargetId: noop,
    trainingTargets: [{ id: "t1", name: "Target One" }],
    macTargetBlocked: () => false,
    updateSelectedPreset: noop,
    selectedPreset: null,
    targetPresets: [],
    openDataset: noop,
    activeDataset: { id: "ds1", name: "Set" },
    datasets: [{ id: "ds1", name: "Set" }],
    updateConfigDraft: noop,
    configDraft: { outputName: "out", optimizer: "adamw", resolution: 1024 },
    outputScopes: [],
    visibleQualityPresets: [],
    gpuOptions: ["auto"],
    customizedConfigLabels: [],
    showAdvancedConfig: false,
    setShowAdvancedConfig: noop,
    showNetworkType: false,
    networkTypeOptions: [],
    macLokrOnWanBlocked: false,
    isLokrNetwork: false,
    visibleOptimizerOptions: [],
    visibleLrSchedulerOptions: [],
    showTrainingAdapter: false,
    visibleTrainingAdapterVersions: [],
    visibleResolutionOptions: [],
    configWarnings: [],
    trainingRunMode: "real",
    submittingJob: false,
    setTrainingRunMode: noop,
    resetConfigDefaults: noop,
    submitTrainingJob: noop,
    configSnapshot: null,
    readiness: null,
    readinessLoading: false,
    readinessBlocksTraining: false,
    ...overrides,
  };
}

function submitButton() {
  return container.querySelector(".training-config-actions button.primary-action");
}

describe("ConfigureJobPanel readiness gate", () => {
  it("enables Train when the config is ready and readiness does not block", () => {
    mount(<ConfigureJobPanel {...baseProps()} />);
    const button = submitButton();
    expect(button.textContent).toContain("training");
    expect(button.disabled).toBe(false);
  });

  it("disables Train and shows an advisory when readiness is Blocked", () => {
    mount(
      <ConfigureJobPanel
        {...baseProps({
          readiness: { gate: "blocked", subScores: { technical: 0 }, counts: { fatal: 1 }, itemCount: 2, items: [], datasetFlags: [] },
          readinessBlocksTraining: true,
        })}
      />,
    );
    expect(submitButton().disabled).toBe(true);
    expect(container.textContent).toContain("isn’t ready to train");
  });

  it("keeps Train disabled when the config itself is not ready", () => {
    mount(<ConfigureJobPanel {...baseProps({ configReady: false })} />);
    expect(submitButton().disabled).toBe(true);
  });
});
