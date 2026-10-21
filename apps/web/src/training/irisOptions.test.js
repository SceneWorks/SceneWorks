import { describe, expect, it } from "vitest";

import { summarize } from "../validation/issues.js";
import {
  choiceOptionsForTarget,
  configDraftFromTarget,
  configValidation,
  timestepTypeOptionsForTarget,
  trainingConfigSnapshot,
} from "./trainingConfig.js";
import {
  IRIS_OPTION_KEYS,
  irisCaptionFieldOptions,
  irisFloorSummary,
  irisMemoryFloors,
  irisPreviousRuns,
  irisRunOutcome,
  isIrisTarget,
  trainingLaneFromCapabilities,
} from "./irisOptions.js";
import { IRIS_CLIENT_OPTION_KEYS, irisJob, irisTarget } from "./iris.testSupport.js";

const DATASET = { id: "ds1", name: "Set", version: 3, items: [{ id: "i1" }] };

function irisDraft(overrides = {}, irisOverrides = {}) {
  const target = irisTarget();
  const draft = configDraftFromTarget(target, DATASET, ["auto"], "trg");
  return { ...draft, ...overrides, iris: { ...draft.iris, ...irisOverrides } };
}

function snapshotFor(draft, target = irisTarget()) {
  return trainingConfigSnapshot({ activeDataset: DATASET, configDraft: draft, selectedPreset: null, selectedTarget: target });
}

describe("Iris option schema", () => {
  it("covers exactly the client-settable engine option keys (no path keys)", () => {
    expect([...IRIS_OPTION_KEYS].sort()).toEqual([...IRIS_CLIENT_OPTION_KEYS].sort());
    expect(IRIS_OPTION_KEYS).not.toContain("resume_from");
    expect(IRIS_OPTION_KEYS).not.toContain("load_from");
  });

  it("recognizes the Iris kernel only", () => {
    expect(isIrisTarget(irisTarget())).toBe(true);
    expect(isIrisTarget({ kernel: "mage_flow_lora" })).toBe(false);
    expect(isIrisTarget(null)).toBe(false);
  });
});

describe("configDraftFromTarget — Iris", () => {
  it("seeds the iris draft from defaults.advanced.iris", () => {
    const draft = configDraftFromTarget(irisTarget(), DATASET, ["auto"]);
    expect(draft.iris).toMatchObject({
      init: "weights",
      text_conditioning: "cached",
      text_dropout: "0.1",
      gradient_clip: "0.5",
      ema_enabled: true,
      ema_decay: "0.9999",
      on_caption_overflow: "warn",
      caption_field: "caption",
      caption_fields: [],
      keep_last_checkpoints: "2",
      milestone_steps: "",
      nan_loss_tolerance: "20",
      betas: ["0.9", "0.95"],
      auto_lr: "none",
      base_batch_size: "256",
      preview_negative_prompt: "",
      upstream_validation_prompts: false,
      export_dtype: "fp32",
      // Unset in the defaults → empty ("backbone/trainer default").
      prediction: "",
      flow_shift: "",
      muon_nesterov: "",
      preview_weights: "",
    });
    expect(draft.resumeFromJobId).toBe("");
    expect(draft.loadFromJobId).toBe("");
    expect(draft.gradientCheckpointing).toBe(false);
    expect(draft.timestepType).toBe("logit_normal");
    expect(draft.precision).toBe("bf16");
  });

  it("adds no iris draft to other targets", () => {
    const draft = configDraftFromTarget({ id: "x", kernel: "sdxl_lora", defaults: { advanced: {} } }, DATASET, ["auto"]);
    expect(draft).not.toHaveProperty("iris");
    expect(draft).not.toHaveProperty("resumeFromJobId");
  });
});

describe("trainingConfigSnapshot — Iris", () => {
  it("sends the compacted defaults: unset flow knobs and adapter-irrelevant keys are absent", () => {
    const iris = snapshotFor(irisDraft()).config.advanced.iris;
    expect(iris).toEqual({
      text_conditioning: "cached",
      text_dropout: 0.1,
      gradient_clip: 0.5,
      ema_enabled: true,
      ema_decay: 0.9999,
      on_caption_overflow: "warn",
      caption_field: "caption",
      keep_last_checkpoints: 2,
      nan_loss_tolerance: 20,
      betas: [0.9, 0.95],
      auto_lr: "none",
      upstream_validation_prompts: false,
      export_dtype: "fp32",
    });
  });

  it("delivers every offered option to advanced.iris when each is in play", () => {
    const draft = irisDraft(
      { networkType: "full", optimizer: "muon", timestepType: "logit_normal", resumeFromJobId: "job_prev" },
      {
        init: "random",
        prediction: "x",
        flow_shift: "3",
        shift_law: "flux",
        shift_base_tokens: "256",
        logit_mean: "-0.5",
        logit_std: "1.2",
        x_pred_sigma_min: "0.05",
        text_dropout: "0.2",
        gradient_clip: "1",
        betas: ["0.8", "0.99"],
        muon_momentum: "0.9",
        muon_nesterov: true,
        muon_adjust_lr: "spectral_norm",
        auto_lr: "sqrt",
        base_batch_size: "64",
        ema_enabled: true,
        ema_decay: "0.999",
        num_epochs: "3",
        nan_loss_tolerance: "5",
        caption_field: "short",
        caption_fields: ["short", "tags"],
        on_caption_overflow: "error",
        text_conditioning: "on_the_fly",
        keep_last_checkpoints: "0",
        milestone_steps: "100, 200",
        resume_data_policy: "new_phase",
        override_lr_on_resume: true,
        preview_weights: "ema",
        preview_negative_prompt: "blurry",
        export_weights: "raw",
        export_dtype: "bf16",
        upstream_validation_prompts: true,
      },
    );
    const advanced = snapshotFor(draft).config.advanced;
    expect(Object.keys(advanced.iris).sort()).toEqual([...IRIS_CLIENT_OPTION_KEYS].sort());
    expect(advanced.iris).toMatchObject({
      init: "random",
      prediction: "x",
      flow_shift: 3,
      shift_law: "flux",
      shift_base_tokens: 256,
      logit_mean: -0.5,
      logit_std: 1.2,
      x_pred_sigma_min: 0.05,
      text_dropout: 0.2,
      gradient_clip: 1,
      betas: [0.8, 0.99],
      muon_momentum: 0.9,
      muon_nesterov: true,
      muon_adjust_lr: "spectral_norm",
      auto_lr: "sqrt",
      base_batch_size: 64,
      ema_decay: 0.999,
      num_epochs: 3,
      nan_loss_tolerance: 5,
      caption_field: "short",
      caption_fields: ["short", "tags"],
      on_caption_overflow: "error",
      text_conditioning: "on_the_fly",
      keep_last_checkpoints: 0,
      milestone_steps: [100, 200],
      resume_data_policy: "new_phase",
      override_lr_on_resume: true,
      preview_weights: "ema",
      preview_negative_prompt: "blurry",
      export_weights: "raw",
      export_dtype: "bf16",
      upstream_validation_prompts: true,
    });
    expect(advanced.resumeFromJobId).toBe("job_prev");
    expect(advanced).not.toHaveProperty("loadFromJobId");
    // Upstream's seven built-in prompts replace the pool, so the request carries none.
    expect(advanced.samplePrompts).toEqual([]);
    expect(advanced.iris).not.toHaveProperty("resume_from");
    expect(advanced.iris).not.toHaveProperty("load_from");
  });

  it("strips the muon extras unless the optimizer is muon", () => {
    const draft = irisDraft(
      { optimizer: "adamw" },
      { muon_momentum: "0.9", muon_nesterov: false, muon_adjust_lr: "rms_norm" },
    );
    const iris = snapshotFor(draft).config.advanced.iris;
    expect(iris).not.toHaveProperty("muon_momentum");
    expect(iris).not.toHaveProperty("muon_nesterov");
    expect(iris).not.toHaveProperty("muon_adjust_lr");
  });

  it("sends the resume knobs only while resuming", () => {
    const knobs = { resume_data_policy: "exact", override_lr_on_resume: true };
    const fresh = snapshotFor(irisDraft({}, knobs)).config.advanced;
    expect(fresh.iris).not.toHaveProperty("resume_data_policy");
    expect(fresh.iris).not.toHaveProperty("override_lr_on_resume");
    expect(fresh).not.toHaveProperty("resumeFromJobId");

    const resumed = snapshotFor(irisDraft({ resumeFromJobId: "job_a" }, knobs)).config.advanced;
    expect(resumed.iris.resume_data_policy).toBe("exact");
    expect(resumed.iris.override_lr_on_resume).toBe(true);
    expect(resumed.resumeFromJobId).toBe("job_a");
  });

  it("sends loadFromJobId for a full run only", () => {
    expect(snapshotFor(irisDraft({ networkType: "full", loadFromJobId: "job_full" })).config.advanced.loadFromJobId).toBe("job_full");
    expect(snapshotFor(irisDraft({ networkType: "lora", loadFromJobId: "job_full" })).config.advanced).not.toHaveProperty("loadFromJobId");
  });

  it("drops init for an adapter run and the logit knobs off the logit-normal sampler", () => {
    const iris = snapshotFor(
      irisDraft({ networkType: "lokr", timestepType: "uniform" }, { init: "random", logit_mean: "0", logit_std: "1" }),
    ).config.advanced.iris;
    expect(iris).not.toHaveProperty("init");
    expect(iris).not.toHaveProperty("logit_mean");
    expect(iris).not.toHaveProperty("logit_std");
  });

  it("drops the EMA decay with EMA off and the base batch without auto LR", () => {
    const iris = snapshotFor(irisDraft({}, { ema_enabled: false, auto_lr: "none" })).config.advanced.iris;
    expect(iris.ema_enabled).toBe(false);
    expect(iris).not.toHaveProperty("ema_decay");
    expect(iris).not.toHaveProperty("base_batch_size");
  });

  it("keeps the user's own prompts while upstream prompts are off", () => {
    const advanced = snapshotFor(irisDraft({ samplePrompts: "a cat\na dog" })).config.advanced;
    expect(advanced.samplePrompts).toEqual(["a cat", "a dog"]);
  });

  it("never sends gradient checkpointing on a target that refuses it", () => {
    expect(snapshotFor(irisDraft({ gradientCheckpointing: true })).config.advanced.gradientCheckpointing).toBe(false);
  });

  it("leaves a legacy target's payload untouched", () => {
    const legacy = {
      id: "sdxl_lora",
      kernel: "sdxl_lora",
      outputKind: "lora",
      defaults: { rank: 8, alpha: 8, learningRate: 0.0001, steps: 10, resolution: 1024, saveEvery: 0, optimizer: "adamw", advanced: {} },
      limits: {},
    };
    const draft = { ...configDraftFromTarget(legacy, DATASET, ["auto"], "trg"), gradientCheckpointing: true, samplePrompts: "" };
    const advanced = snapshotFor(draft, legacy).config.advanced;
    expect(advanced).not.toHaveProperty("iris");
    expect(advanced).not.toHaveProperty("resumeFromJobId");
    expect(advanced.gradientCheckpointing).toBe(true);
    // Empty pool still falls back to the trigger-derived prompts for other targets.
    expect(advanced.samplePrompts).toHaveLength(4);
  });
});

describe("configValidation — Iris", () => {
  const ctx = { activeDataset: DATASET, selectedTarget: irisTarget() };
  const errorsFor = (draft) => summarize(configValidation(draft, ctx));

  it("passes the seeded draft", () => {
    expect(errorsFor(irisDraft()).ready).toBe(true);
  });

  it("names each micro-step field that is not a multiple of gradient accumulation", () => {
    const summary = errorsFor(
      irisDraft({ gradientAccumulation: "4", steps: "1000", saveEvery: "250", sampleEvery: "250", lrWarmupSteps: "10" }),
    );
    expect(summary.ready).toBe(false);
    expect([...summary.invalidFields].sort()).toEqual(["lrWarmupSteps", "sampleEvery", "saveEvery"]);
    expect(summary.surfaced.map((item) => item.message)).toContain("Checkpoint cadence must be a multiple of the gradient accumulation (4)");
  });

  it("passes when every micro-step field is a multiple", () => {
    const summary = errorsFor(
      irisDraft({ gradientAccumulation: "4", steps: "1000", saveEvery: "200", sampleEvery: "100", lrWarmupSteps: "0" }),
    );
    expect(summary.ready).toBe(true);
  });

  it("does not impose the micro-step rule on a target that does not advertise it", () => {
    const legacy = { ...irisTarget(), kernel: "sdxl_lora", limits: {} };
    const summary = summarize(
      configValidation(irisDraft({ gradientAccumulation: "4", saveEvery: "250" }), { activeDataset: DATASET, selectedTarget: legacy }),
    );
    expect(summary.invalidFields.has("saveEvery")).toBe(false);
  });

  it("caps the sample count at the advertised upper bound", () => {
    expect(errorsFor(irisDraft({ sampleCount: "5" })).invalidFields.has("sampleCount")).toBe(true);
    expect(errorsFor(irisDraft({ sampleCount: "4" })).invalidFields.has("sampleCount")).toBe(false);
  });

  it("outlines an out-of-domain Iris option by its own field", () => {
    const summary = errorsFor(irisDraft({}, { text_dropout: "1.5", gradient_clip: "0", betas: ["0.9", "1"] }));
    expect(summary.invalidFields.has("iris.text_dropout")).toBe(true);
    expect(summary.invalidFields.has("iris.gradient_clip")).toBe(true);
    expect(summary.invalidFields.has("iris.betas")).toBe(true);
  });

  it("ignores a broken value in a control that is not in play", () => {
    // muon momentum is hidden (and not sent) under AdamW, so it cannot block the run.
    expect(errorsFor(irisDraft({ optimizer: "adamw" }, { muon_momentum: "7" })).ready).toBe(true);
    expect(errorsFor(irisDraft({ optimizer: "muon" }, { muon_momentum: "7" })).invalidFields.has("iris.muon_momentum")).toBe(true);
  });

  it("refuses EMA weights with the EMA off", () => {
    const summary = errorsFor(irisDraft({}, { ema_enabled: false, preview_weights: "ema" }));
    expect(summary.invalidFields.has("iris.preview_weights")).toBe(true);
  });

  it("refuses resume together with start-from-weights, and start-from-weights with random init", () => {
    const both = errorsFor(irisDraft({ networkType: "full", resumeFromJobId: "a", loadFromJobId: "b" }));
    expect(both.invalidFields.has("loadFromJobId")).toBe(true);
    const random = errorsFor(irisDraft({ networkType: "full", loadFromJobId: "b" }, { init: "random" }));
    expect(random.invalidFields.has("loadFromJobId")).toBe(true);
  });

  it("refuses malformed milestone steps", () => {
    expect(errorsFor(irisDraft({}, { milestone_steps: "100, x" })).invalidFields.has("iris.milestone_steps")).toBe(true);
    expect(errorsFor(irisDraft({}, { milestone_steps: "0" })).invalidFields.has("iris.milestone_steps")).toBe(true);
  });
});

describe("limit-driven choice lists", () => {
  it("offers exactly the target's advertised lists", () => {
    const target = irisTarget();
    expect(timestepTypeOptionsForTarget(target)).toEqual(["logit_normal", "uniform"]);
    expect(choiceOptionsForTarget(target, "timestepBiases", ["balanced", "high_noise"])).toEqual(["balanced"]);
    expect(choiceOptionsForTarget(target, "lossTypes", ["mse", "mae"])).toEqual(["mse"]);
    expect(choiceOptionsForTarget(target, "mixedPrecisions", null)).toEqual(["bf16", "f32"]);
  });

  it("keeps the legacy fallbacks for a target that advertises nothing", () => {
    expect(timestepTypeOptionsForTarget({ kernel: "sdxl_lora" })).toEqual(["sigmoid", "linear", "uniform", "weighted"]);
    expect(choiceOptionsForTarget({ limits: {} }, "lossTypes", ["mse", "mae"])).toEqual(["mse", "mae"]);
    expect(choiceOptionsForTarget({}, "mixedPrecisions", null)).toBeNull();
  });
});

describe("Iris helpers", () => {
  it("reads the declared floor for the lane, artifact and resolution (next row up)", () => {
    const target = irisTarget();
    expect(irisMemoryFloors(target, "lora", 256, "mlx")).toEqual([{ lane: "mlx", gib: 16, provenance: "declared" }]);
    expect(irisMemoryFloors(target, "full", 384, "candle")).toEqual([{ lane: "candle", gib: 88, provenance: "measured" }]);
    expect(irisMemoryFloors(target, "lokr", 512, null)).toEqual([
      { lane: "mlx", gib: 24, provenance: "declared" },
      { lane: "candle", gib: 32, provenance: "declared" },
    ]);
    expect(irisMemoryFloors(target, "lora", 2048, "mlx")).toEqual([]);
    expect(irisMemoryFloors({}, "lora", 256, "mlx")).toEqual([]);
  });

  it("reads each row's own provenance, and the full_ac row under full activation checkpointing", () => {
    const target = irisTarget();
    // The same table mixes measured, extrapolated and declared rows.
    expect(irisMemoryFloors(target, "full", 512, "mlx")).toEqual([{ lane: "mlx", gib: 88, provenance: "measured" }]);
    expect(irisMemoryFloors(target, "full", 1024, "candle")).toEqual([{ lane: "candle", gib: 104, provenance: "extrapolated" }]);
    // A full run that recomputes every region reads full_ac; selective_layer and adapters do not.
    for (const policy of ["full", "selective_op"]) {
      expect(irisMemoryFloors(target, "full", 1024, "mlx", policy)).toEqual([{ lane: "mlx", gib: 96, provenance: "measured" }]);
    }
    expect(irisMemoryFloors(target, "full", 1024, "mlx", "selective_layer")).toEqual([{ lane: "mlx", gib: 104, provenance: "measured" }]);
    expect(irisMemoryFloors(target, "lora", 1024, "mlx", "full")).toEqual([{ lane: "mlx", gib: 40, provenance: "declared" }]);
    // A lane whose table has no full_ac rows keeps reading full.
    expect(irisMemoryFloors(target, "full", 1024, "candle", "full")).toEqual([{ lane: "candle", gib: 104, provenance: "extrapolated" }]);
    // A table that publishes no provenance is declared, never assumed measured.
    const bare = { memoryFloors: { lanes: target.memoryFloors.lanes } };
    expect(irisMemoryFloors(bare, "full", 512, "mlx")).toEqual([{ lane: "mlx", gib: 88, provenance: "declared" }]);
  });

  it("names every floor with its unit and its own evidence", () => {
    const summary = irisFloorSummary(
      [
        { lane: "mlx", gib: 88, provenance: "measured" },
        { lane: "candle", gib: 104, provenance: "extrapolated" },
        { lane: "x", gib: 128, provenance: "lower_bound" },
      ],
      { mlx: "MLX", candle: "Candle" },
    );
    expect(summary).toBe(
      "MLX 88 GiB (measured) · Candle 104 GiB (extrapolated from measured runs) · x 128 GiB (declared above a measured lower bound)",
    );
    expect(irisFloorSummary([{ lane: "mlx", gib: 16, provenance: "declared" }])).toBe("mlx 16 GiB (declared, not yet measured)");
  });

  it("derives the lane from the capability switches", () => {
    expect(trainingLaneFromCapabilities({ macGatingActive: true })).toBe("mlx");
    expect(trainingLaneFromCapabilities({ candleGatingActive: true })).toBe("candle");
    expect(trainingLaneFromCapabilities({})).toBeNull();
  });

  it("offers the dataset's named caption fields plus the ordinary caption", () => {
    const dataset = { items: [{ captions: { short: "a", tags: "b" } }, { captions: { short: "c" } }, {}] };
    expect(irisCaptionFieldOptions(dataset)).toEqual(["caption", "short", "tags"]);
    expect(irisCaptionFieldOptions(null)).toEqual(["caption"]);
  });

  it("lists terminal Iris runs of this project, newest first, and full runs for start-from-weights", () => {
    const jobs = [
      irisJob({ id: "a", createdAt: "2026-10-01T00:00:00Z" }),
      irisJob({ id: "b", networkType: "full", status: "canceled", createdAt: "2026-10-03T00:00:00Z" }),
      irisJob({ id: "c", status: "running" }),
      irisJob({ id: "d", targetId: "sdxl_lora" }),
      irisJob({ id: "e", projectId: "p2" }),
      irisJob({ id: "f", type: "dataset_caption" }),
    ];
    expect(irisPreviousRuns(jobs, { projectId: "p1" }).map((job) => job.id)).toEqual(["b", "a"]);
    expect(irisPreviousRuns(jobs, { projectId: "p1", fullOnly: true }).map((job) => job.id)).toEqual(["b"]);
  });

  it("describes a cancel that kept a checkpoint and a partial export", () => {
    const job = irisJob({
      id: "x",
      status: "canceled",
      result: { trainedArtifactExported: true, stepsCompleted: 120, checkpoints: [{ step: 100 }] },
    });
    expect(irisRunOutcome(job)).toBe("Canceled at step 120 — checkpoint and partial export saved");
    expect(irisRunOutcome(irisJob({ id: "y", status: "canceled" }))).toBe("Canceled");
    expect(irisRunOutcome(irisJob({ id: "z" }))).toBe("Completed");
  });
});
