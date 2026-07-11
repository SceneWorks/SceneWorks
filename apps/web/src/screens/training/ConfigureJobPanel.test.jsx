import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { configValidation } from "../../training/trainingConfig.js";
import { summarize } from "../../validation/issues.js";
import { ConfigureJobPanel } from "./ConfigureJobPanel.jsx";

// The highest-stakes glue in sc-6534: the Train button must disable when the readiness gate is
// Blocked, and stay enabled otherwise. A wrong binding either blocks a trainable set or trains an
// untrainable one — neither is caught by the pure-helper or store tests. ConfigureJobPanel is
// presentational, so a minimal fixture (advanced/network/adapter sections off) mounts it cheaply.
//
// sc-10647 moved the panel onto the app-wide validation core: one `configValidity` summary now
// gates the button, tones the pill, fills the chip row, and outlines the broken inputs. The tests
// below drive it through the real `configValidation` rules rather than hand-built summaries, so a
// rule whose kind or field is wrong fails here rather than passing on a fixture that agrees with it.

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

// A draft that satisfies every rule, so a test can break exactly one thing.
const VALID_DRAFT = {
  outputName: "out",
  triggerWord: "trg",
  optimizer: "adamw",
  rank: 16,
  alpha: 16,
  learningRate: 0.0001,
  steps: 1000,
  resolution: 1024,
  batchSize: 1,
  gradientAccumulation: 1,
  saveEvery: 250,
};

const TARGET = { id: "t1", name: "Target One", baseModel: "base" };
const DATASET = { id: "ds1", name: "Set" };

// Run the real rules. `configValidity` is never hand-assembled: a fixture that agreed with a
// broken rule set would make these tests vacuous.
function validityFor(draft = VALID_DRAFT, ctx = { activeDataset: DATASET, selectedTarget: TARGET }) {
  return summarize(configValidation(draft, ctx));
}

function baseProps(overrides = {}) {
  return {
    active: { id: "configure", title: "Configure training job" },
    setActiveView: noop,
    configValidity: validityFor(),
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
    configDraft: VALID_DRAFT,
    outputScopes: [],
    qualityTiers: [],
    updateQualityTier: noop,
    gpuOptions: ["auto"],
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
    submittingJob: false,
    resetConfigDefaults: noop,
    submitTrainingJob: noop,
    configSnapshot: null,
    // sc-8942 (F-140): the Dataset Doctor readout props are now one grouped `datasetDoctor`
    // bundle (report/loading + the six fix-action handlers) shared with DatasetEditorPanel.
    datasetDoctor: { report: null, loading: false },
    ...overrides,
  };
}

function submitButton() {
  return container.querySelector(".training-config-actions button.primary-action");
}

function chips() {
  return [...container.querySelectorAll(".validation-chip")].map((chip) => chip.textContent);
}

describe("ConfigureJobPanel readiness gate", () => {
  it("enables Train when the config is ready and readiness does not block", () => {
    mount(<ConfigureJobPanel {...baseProps()} />);
    const button = submitButton();
    expect(button.textContent).toContain("training");
    expect(button.disabled).toBe(false);
  });

  // Readiness is no longer a separate prop (sc-10648): a Blocked gate is one of
  // configValidity's errors, so it disables Train and shows up in the chip row like any
  // other. Drive it through the real rules via ctx.datasetNotReady.
  it("disables Train and names the reason when the dataset readiness gate is Blocked", () => {
    const configValidity = validityFor(VALID_DRAFT, {
      activeDataset: DATASET,
      selectedTarget: TARGET,
      datasetNotReady: true,
    });
    mount(
      <ConfigureJobPanel
        {...baseProps({
          configValidity,
          datasetDoctor: {
            report: { gate: "blocked", subScores: { technical: 0 }, counts: { fatal: 1 }, itemCount: 2, items: [], datasetFlags: [] },
            loading: false,
          },
        })}
      />,
    );
    expect(submitButton().disabled).toBe(true);
    expect(chips()).toContain("This dataset isn’t ready to train yet — open Data Sets to add or fix images.");
  });

  it("keeps Train disabled when the config itself is not ready", () => {
    const configValidity = validityFor(VALID_DRAFT, { activeDataset: null, selectedTarget: TARGET });
    mount(<ConfigureJobPanel {...baseProps({ configValidity })} />);
    expect(submitButton().disabled).toBe(true);
  });
});

// The bidirectional pair the epic's testing contract demands (epic 10644). A test that only
// asserts the happy path passes against a broken implementation — that is exactly how sc-10492
// shipped green. Each of these must fail if the requirement/error split is collapsed either way.
describe("ConfigureJobPanel surfaces broken values and stays quiet about unfilled ones", () => {
  it("chips a cleared number and outlines the input it names", () => {
    const configValidity = validityFor({ ...VALID_DRAFT, rank: "" });
    mount(<ConfigureJobPanel {...baseProps({ configValidity, configDraft: { ...VALID_DRAFT, rank: "" }, showAdvancedConfig: true })} />);

    expect(chips()).toContain("Rank must be greater than zero");
    expect(submitButton().disabled).toBe(true);

    // R5: the chip names Rank, so the Rank box must show it. Twenty-five inputs sit in this form.
    const rank = [...container.querySelectorAll("label")].find((label) => label.textContent.startsWith("Rank"));
    expect(rank.querySelector("input").getAttribute("aria-invalid")).toBe("true");
  });

  // Direction 1: flip `error` → `requirement` and the chip must vanish. Driving the real rules
  // means this fails the moment a numeric rule is mis-kinded.
  it("says nothing about a field the user simply has not filled in", () => {
    const configValidity = validityFor(
      { ...VALID_DRAFT, outputName: "", triggerWord: "" },
      { activeDataset: null, selectedTarget: TARGET },
    );
    mount(<ConfigureJobPanel {...baseProps({ configValidity, configDraft: { ...VALID_DRAFT, outputName: "", triggerWord: "" } })} />);

    // Three requirements are live — dataset, LoRA name, trigger phrase — and none of them speaks.
    expect(container.querySelector(".validation-chips")).toBeNull();
    expect(chips()).toEqual([]);
    // ...yet Start is dead, and the pill is the only thing that says so.
    expect(submitButton().disabled).toBe(true);
    expect(container.querySelector(".ready-pill").textContent).toBe("Needs input");
  });

  // Direction 2: widen the filter to every issue and the requirement hints leak back in. Pinning
  // the exact chip set is what catches that — a `toContain` assertion would not.
  it("shows the broken value without the unfilled-field hints beside it", () => {
    const draft = { ...VALID_DRAFT, outputName: "", rank: "" };
    const configValidity = validityFor(draft, { activeDataset: null, selectedTarget: TARGET });
    mount(<ConfigureJobPanel {...baseProps({ configValidity, configDraft: draft, showAdvancedConfig: true })} />);

    // Pinned exactly. "Select a saved dataset" also exists as the Dataset select's own
    // placeholder <option>, so an assertion over the whole page's text would be answering a
    // different question — scope it to the chip row.
    expect(chips()).toEqual(["Rank must be greater than zero"]);
    expect(container.querySelector(".validation-chips").textContent).not.toContain("Select a saved dataset");
    expect(container.querySelector(".validation-chips").textContent).not.toContain("Name the");
  });

  it("leaves an unfilled field unoutlined, so a fresh form is not red", () => {
    const draft = { ...VALID_DRAFT, outputName: "" };
    const configValidity = validityFor(draft, { activeDataset: DATASET, selectedTarget: TARGET });
    mount(<ConfigureJobPanel {...baseProps({ configValidity, configDraft: draft })} />);

    const name = [...container.querySelectorAll("label")].find((label) => label.textContent.startsWith("LoRA name"));
    expect(name.querySelector("input").getAttribute("aria-invalid")).toBeNull();
    expect(container.querySelector("[aria-invalid]")).toBeNull();
  });

  it("tones the pill Ready when the draft is whole", () => {
    mount(<ConfigureJobPanel {...baseProps()} />);
    const pill = container.querySelector(".ready-pill");
    expect(pill.textContent).toBe("Ready");
    expect(pill.className).toContain("is-ready");
  });
});

// sc-10689: configValidation raises a `> 0` error for eight numeric fields; every one must
// name an input the user can reach, or the chip points at a control that isn't on the screen
// and Start dies unfixably (the epic's own defect class, one step worse). This drove the bug:
// batchSize and gradientAccumulation were validated with no input, so clearing either chipped
// with nothing to outline. The map is the field label configValidation uses, which is also the
// panel's label text — so it double-checks the two agree.
describe("every validated numeric field maps to a reachable, outline-able input", () => {
  const FIELD_LABELS = {
    rank: "Rank",
    alpha: "Alpha",
    learningRate: "Learning rate",
    steps: "Steps",
    resolution: "Resolution",
    batchSize: "Batch size",
    gradientAccumulation: "Gradient accumulation",
    saveEvery: "Checkpoint cadence",
  };

  for (const [field, label] of Object.entries(FIELD_LABELS)) {
    it(`chips ${field} and outlines the ${label} control the chip names`, () => {
      const draft = { ...VALID_DRAFT, [field]: "" };
      const configValidity = validityFor(draft);
      mount(
        <ConfigureJobPanel
          {...baseProps({ configValidity, configDraft: draft, showAdvancedConfig: true, visibleResolutionOptions: [512, 768, 1024] })}
        />,
      );

      expect(chips()).toContain(`${label} must be greater than zero`);

      const control = [...container.querySelectorAll("label")]
        .find((node) => node.textContent.trim().startsWith(label))
        ?.querySelector("input, select");
      expect(control).toBeTruthy();
      expect(control.getAttribute("aria-invalid")).toBe("true");
    });
  }
});
