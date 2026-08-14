import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { configDraftFromTarget, configValidation } from "../../training/trainingConfig.js";
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

it("renders a target-seeded logit-normal timestep schedule", () => {
  const selectedTarget = {
    ...TARGET,
    kernel: "sd3_lora",
    defaults: {
      ...VALID_DRAFT,
      advanced: { timestepType: "logit_normal" },
    },
  };
  const configDraft = configDraftFromTarget(selectedTarget, DATASET, ["auto"]);

  mount(
    <ConfigureJobPanel
      {...baseProps({ selectedTarget, configDraft, showAdvancedConfig: true })}
    />,
  );

  const timestepSelect = [...container.querySelectorAll("label")]
    .find((label) => label.textContent.includes("Timestep type"))
    ?.querySelector("select");
  expect(timestepSelect?.value).toBe("logit_normal");
  expect([...timestepSelect.options].map((option) => option.textContent)).toContain("Logit Normal");
});

it.each([
  ["Anima", "anima_lora"],
  ["Mage", "mage_flow_lora"],
])("does not offer SD3-only logit-normal scheduling for %s", (_name, kernel) => {
  const selectedTarget = {
    ...TARGET,
    kernel,
    defaults: {
      ...VALID_DRAFT,
      advanced: { timestepType: "sigmoid" },
    },
  };
  const configDraft = configDraftFromTarget(selectedTarget, DATASET, ["auto"]);

  mount(
    <ConfigureJobPanel
      {...baseProps({ selectedTarget, configDraft, showAdvancedConfig: true })}
    />,
  );

  const timestepSelect = [...container.querySelectorAll("label")]
    .find((label) => label.textContent.includes("Timestep type"))
    ?.querySelector("select");
  const values = [...timestepSelect.options].map((option) => option.value);
  expect(values).toEqual(["sigmoid", "linear", "uniform", "weighted"]);
  expect(values).not.toContain("logit_normal");
});

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

  // sc-15036: a full base fine-tune produces a MODEL, not an adapter. Two things must follow it
  // through the panel, or the Studio describes a run it is not making: the name field must say
  // "base checkpoint", and the Output scope PICKER must be replaced by an explanation (the model
  // catalog has one global user manifest — a "project" scope it cannot honour would be a control
  // that silently does nothing, the sc-14056 gradient-checkpointing precedent).
  //
  // Discriminating: the SAME props, one field flipped.
  it("names a full base fine-tune's output field and replaces the scope picker", () => {
    const labelStarting = (text) =>
      [...container.querySelectorAll("label")].find((label) => label.textContent.startsWith(text));

    const adapterProps = baseProps({
      outputScopes: ["project", "global"],
      configDraft: { ...VALID_DRAFT, networkType: "lora" },
      isFullFinetune: false,
    });
    mount(<ConfigureJobPanel {...adapterProps} />);
    expect(labelStarting("LoRA name")).toBeTruthy();
    expect(labelStarting("Base checkpoint name")).toBeFalsy();
    expect(labelStarting("Output scope").querySelector("select")).toBeTruthy();

    mount(
      <ConfigureJobPanel
        {...baseProps({
          outputScopes: ["project", "global"],
          configDraft: { ...VALID_DRAFT, networkType: "full" },
          isFullFinetune: true,
        })}
      />,
    );
    expect(labelStarting("Base checkpoint name")).toBeTruthy();
    expect(labelStarting("LoRA name")).toBeFalsy();
    const scope = labelStarting("Output scope");
    expect(scope.querySelector("select")).toBeNull();
    expect(scope.textContent).toContain("global model library");
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

// sc-14056 — the full base fine-tune. Gradient (activation) checkpointing has no full-tune
// implementation yet (sc-14989) and the engine hard-errors on `full_finetune + gradient_checkpointing`
// rather than ignoring the flag, so the worker clears it for a full run. If the panel kept offering a
// checked box, the user would tick a control that is then silently dropped — the exact silence the
// engine refuses. Assert the swap in BOTH directions so this cannot pass by hiding the box always.
describe("ConfigureJobPanel full base fine-tune", () => {
  function checkpointingCheckbox() {
    return [...container.querySelectorAll(".training-advanced-toggles label")].find((node) =>
      node.textContent.includes("Gradient checkpointing"),
    );
  }

  it("offers the gradient-checkpointing toggle on an adapter run", () => {
    mount(<ConfigureJobPanel {...baseProps({ showAdvancedConfig: true, isFullFinetune: false })} />);
    expect(checkpointingCheckbox()).toBeTruthy();
    expect(container.textContent).not.toContain("not available for a full base fine-tune");
  });

  it("replaces it with an explanation on a full base fine-tune", () => {
    mount(
      <ConfigureJobPanel
        {...baseProps({
          showAdvancedConfig: true,
          isFullFinetune: true,
          selectedTarget: {
            ...TARGET,
            defaults: {
              advanced: {
                fullFinetuneConfig: {
                  mixedPrecision: "f32",
                  gradientCheckpointing: false,
                },
              },
            },
          },
          configDraft: { ...VALID_DRAFT, precision: "bf16" },
        })}
      />,
    );
    expect(checkpointingCheckbox()).toBeFalsy();
    expect(container.textContent).toContain(
      "Gradient checkpointing is not available for a full base fine-tune yet",
    );
    const precision = [...container.querySelectorAll("label")]
      .find((node) => node.textContent.trim().startsWith("Precision"))
      ?.querySelector("input");
    expect(precision).toBeTruthy();
    expect(precision.disabled).toBe(true);
    expect(precision.value).toBe("f32");
    expect(container.textContent).toContain("requires F32 for full base fine-tuning");
  });

  it("preserves MLX full-finetune precision and checkpointing controls", () => {
    mount(
      <ConfigureJobPanel
        {...baseProps({
          showAdvancedConfig: true,
          isFullFinetune: true,
          configDraft: { ...VALID_DRAFT, precision: "bf16", gradientCheckpointing: true },
        })}
      />,
    );
    const precision = [...container.querySelectorAll("label")]
      .find((node) => node.textContent.trim().startsWith("Precision"))
      ?.querySelector("input");
    expect(precision.disabled).toBe(false);
    expect(precision.value).toBe("bf16");
    expect(checkpointingCheckbox()).toBeTruthy();
    expect(checkpointingCheckbox().querySelector("input").checked).toBe(true);
  });

  // Mage advertises all three of its trainer paths. Each must render a HUMAN label, not the raw
  // token — an option falling through `networkTypeLabel`'s `?? value` shows "lokr"/"full" verbatim,
  // which is how you can tell a label was never added.
  it("offers all three Mage network types with human labels", () => {
    mount(
      <ConfigureJobPanel
        {...baseProps({
          showAdvancedConfig: true,
          showNetworkType: true,
          networkTypeOptions: ["lora", "lokr", "full"],
          configDraft: { ...VALID_DRAFT, networkType: "full" },
          isFullFinetune: true,
        })}
      />,
    );
    const select = [...container.querySelectorAll("label")]
      .find((node) => node.textContent.trim().startsWith("Network type"))
      ?.querySelector("select");
    expect(select).toBeTruthy();
    const options = [...select.querySelectorAll("option")].map((option) => option.textContent.trim());
    expect(options).toEqual(["LoRA", "LoKr (LyCORIS Kronecker)", "Full base fine-tune"]);
    expect(options).not.toContain("lokr");
    expect(options).not.toContain("full");
  });

  // LoKr is an ADAPTER path, so it must keep the gradient-checkpointing control that the full path
  // replaces — the swap keys off the full-tune flag, not merely off "not lora".
  it("keeps the gradient-checkpointing toggle on a LoKr run", () => {
    mount(
      <ConfigureJobPanel
        {...baseProps({
          showAdvancedConfig: true,
          showNetworkType: true,
          networkTypeOptions: ["lora", "lokr", "full"],
          configDraft: { ...VALID_DRAFT, networkType: "lokr" },
          isLokrNetwork: true,
          isFullFinetune: false,
        })}
      />,
    );
    expect(checkpointingCheckbox()).toBeTruthy();
    // …and the LoKr factor field appears, which is the control LoKr actually needs.
    const factor = [...container.querySelectorAll("label")].find((node) =>
      node.textContent.trim().startsWith("LoKr factor"),
    );
    expect(factor).toBeTruthy();
  });
});

// ControlNet preprocessor provisioning. The panel must both SHOW the offer and have the run
// blocked; those are two different wires (the notice prop vs the validation context), and getting
// only one right is the failure this covers — an offer beside a live Start button, or a dead Start
// button with nothing on screen explaining it.
describe("ConfigureJobPanel — missing control preprocessor", () => {
  const CONTROL_TARGET = {
    id: "krea_pose_control",
    name: "Krea Pose Control",
    outputKind: "control_branch",
    defaults: { advanced: { controlType: "pose" } },
  };
  const DWPOSE = {
    id: "dwpose_pose_detector",
    name: "DWPose Pose Detector",
    installState: "missing",
    downloadSizeLabel: "330 MB",
  };

  it("offers the download and blocks Start training", () => {
    const missingControlModels = [DWPOSE];
    mount(
      <ConfigureJobPanel
        {...baseProps({
          selectedTarget: CONTROL_TARGET,
          missingControlModels,
          // The real rules, with the same list the notice renders — so the button and the offer
          // cannot disagree.
          configValidity: validityFor(VALID_DRAFT, {
            activeDataset: DATASET,
            selectedTarget: CONTROL_TARGET,
            missingControlModels,
          }),
          onDownloadModel: noop,
        })}
      />,
    );
    expect(container.querySelector(".required-models-notice")).toBeTruthy();
    expect(container.textContent).toContain("DWPose Pose Detector");
    expect(container.textContent).toContain("Pose ControlNet training");
    const start = [...container.querySelectorAll("button")].find((el) =>
      el.textContent.includes("Start training"),
    );
    expect(start).toBeTruthy();
    expect(start.disabled).toBe(true);
  });

  it("renders no notice for a provisioned ControlNet run", () => {
    mount(
      <ConfigureJobPanel
        {...baseProps({
          selectedTarget: CONTROL_TARGET,
          missingControlModels: [],
          configValidity: validityFor(VALID_DRAFT, {
            activeDataset: DATASET,
            selectedTarget: CONTROL_TARGET,
            missingControlModels: [],
          }),
        })}
      />,
    );
    expect(container.querySelector(".required-models-notice")).toBeNull();
  });

  // A LoRA target renders no condition at all, so it must never see this.
  it("renders no notice for a LoRA target", () => {
    mount(<ConfigureJobPanel {...baseProps({ missingControlModels: [] })} />);
    expect(container.querySelector(".required-models-notice")).toBeNull();
  });
});
