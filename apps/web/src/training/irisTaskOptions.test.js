import { describe, expect, it } from "vitest";
import workerReport from "./irisDatasetReport.fixture.json";

import { irisDepthTarget, irisJob, irisRestoreTarget, irisTarget } from "./iris.testSupport.js";
import { irisFloorSummary, irisMemoryFloors, irisObjective, irisPreviousRuns } from "./irisOptions.js";
import {
  irisCheckSummary,
  irisFloorSize,
  irisTrainingResourceIds,
  itemSidecarState,
  matchSidecarFiles,
  sidecarStem,
} from "./irisTaskOptions.js";
import {
  configDraftFromTarget,
  configValidation,
  gradientCheckpointingLanes,
  gradientCheckpointingUnsupported,
  trainingConfigSnapshot,
} from "./trainingConfig.js";
import { datasetPayload } from "./datasetHelpers.js";

const DATASET = { id: "ds1", name: "Set", version: 3, items: [] };
const UNREAD = ["samplePrompts", "sampleSteps", "sampleGuidanceScale", "sampleCount", "timestepType", "timestepBias", "loraTargetModules"];

function draftFor(target, patch = {}) {
  const base = configDraftFromTarget(target, DATASET, ["auto"]);
  return { ...base, outputName: "out", ...patch, irisTask: { ...base.irisTask, ...(patch.irisTask ?? {}) } };
}

function snapshot(target, patch) {
  return trainingConfigSnapshot({ activeDataset: DATASET, configDraft: draftFor(target, patch), selectedTarget: target });
}

function errors(target, patch) {
  return configValidation(draftFor(target, patch), { activeDataset: DATASET, selectedTarget: target }).filter(
    (entry) => entry.kind === "error",
  );
}

describe("Iris objective helpers (sc-25687 / sc-25688)", () => {
  it("tells generation, depth and restoration apart on the shared kernel", () => {
    expect(irisObjective(irisTarget())).toBe("generation");
    expect(irisObjective(irisDepthTarget())).toBe("depth");
    expect(irisObjective(irisRestoreTarget())).toBe("restoration");
    // ui.objective / baseModel fallbacks, and non-Iris targets.
    expect(irisObjective({ kernel: "iris_train", ui: { objective: "depth" } })).toBe("depth");
    expect(irisObjective({ kernel: "iris_train", baseModel: "iris_3b_restore" })).toBe("restoration");
    expect(irisObjective({ kernel: "flux_lora", irisObjective: "depth" })).toBeNull();
  });

  it("seeds a task draft without a trigger word or the generation bag", () => {
    const draft = configDraftFromTarget(irisDepthTarget(), DATASET, ["auto"], "tok");
    expect(draft.triggerWord).toBe("");
    expect(draft.iris).toBeUndefined();
    expect(draft.depthInit).toBe("export");
    expect(draft.irisTask.depth_size).toEqual(["480", "640"]);
    expect(configDraftFromTarget(irisTarget(), DATASET, ["auto"]).irisTask).toBeUndefined();
  });
});

describe("depth snapshot", () => {
  it("sends the compacted iris bag, the start and a full fine-tune — never the generation controls", () => {
    const sent = snapshot(irisDepthTarget());
    const { advanced } = sent.config;
    for (const key of UNREAD) expect(advanced).not.toHaveProperty(key);
    expect(advanced).not.toHaveProperty("iris_restore");
    expect(advanced.networkType).toBe("full");
    expect(advanced.depthInit).toBe("export");
    // Every default except the empty milestone list (empty = trainer default, not sent).
    const { milestone_steps: _empty, ...bag } = irisDepthTarget().defaults.advanced.iris;
    expect(advanced.iris).toEqual(bag);
    expect(sent.config.triggerWord).toBe("");
    expect(sent.config.resolution).toBe(640);
  });

  it("mirrors resolution from the larger depth_size side", () => {
    expect(snapshot(irisDepthTarget(), { irisTask: { depth_size: ["768", "512"] } }).config.resolution).toBe(768);
    expect(errors(irisDepthTarget(), { irisTask: { depth_size: ["500", "640"] } }).map((e) => e.field)).toContain(
      "irisTask.depth_size",
    );
  });

  it("sends reducer_init only with the generation-backbone start", () => {
    const patch = { irisTask: { reducer_init: "channel_mean" } };
    expect(snapshot(irisDepthTarget(), patch).config.advanced.iris).not.toHaveProperty("reducer_init");
    const parent = snapshot(irisDepthTarget(), { ...patch, depthInit: "parent" }).config.advanced;
    expect(parent.iris.reducer_init).toBe("channel_mean");
    expect(parent.depthInit).toBe("parent");
  });

  it("keeps loadFromJobId to the export start, without a resume", () => {
    expect(snapshot(irisDepthTarget(), { loadFromJobId: "job_1" }).config.advanced.loadFromJobId).toBe("job_1");
    expect(snapshot(irisDepthTarget(), { loadFromJobId: "job_1", depthInit: "parent" }).config.advanced).not.toHaveProperty(
      "loadFromJobId",
    );
    expect(errors(irisDepthTarget(), { loadFromJobId: "job_1", resumeFromJobId: "job_0" }).map((e) => e.field)).toContain(
      "loadFromJobId",
    );
  });

  it("refuses EMA weights and decay without the EMA, and an inverted percentile window", () => {
    const off = errors(irisDepthTarget(), { irisTask: { export_weights: "ema" } }).map((e) => e.field);
    expect(off).toContain("irisTask.export_weights");
    expect(errors(irisDepthTarget(), { irisTask: { export_weights: "ema", ema_enabled: true } })).toEqual([]);
    expect(snapshot(irisDepthTarget(), { irisTask: { ema_decay: "0.99" } }).config.advanced.iris).not.toHaveProperty("ema_decay");
    expect(errors(irisDepthTarget(), { irisTask: { low_percentile: "98", high_percentile: "2" } }).map((e) => e.field)).toContain(
      "irisTask.low_percentile",
    );
  });

  it("needs no trigger word and validates micro-steps against the accumulation", () => {
    const all = configValidation(draftFor(irisDepthTarget()), { activeDataset: DATASET, selectedTarget: irisDepthTarget() });
    expect(all.map((entry) => entry.field)).not.toContain("triggerWord");
    expect(errors(irisDepthTarget(), { steps: "320016" }).map((e) => e.field)).toContain("steps");
  });
});

describe("restoration snapshot", () => {
  it("sends the iris_restore bag — never an iris bag, loadFromJobId or the generation controls", () => {
    const { advanced } = snapshot(irisRestoreTarget(), { loadFromJobId: "job_1" }).config;
    for (const key of UNREAD) expect(advanced).not.toHaveProperty(key);
    expect(advanced).not.toHaveProperty("iris");
    expect(advanced).not.toHaveProperty("loadFromJobId");
    expect(advanced).not.toHaveProperty("depthInit");
    const { milestone_steps: _empty, ...bag } = irisRestoreTarget().defaults.advanced.iris_restore;
    expect(advanced.iris_restore).toEqual(bag);
  });

  it("sends lora_init only for an adapter, resume_data_policy only with a resume", () => {
    const patch = { irisTask: { lora_init: "kaiming", resume_data_policy: "new_phase" } };
    const full = snapshot(irisRestoreTarget(), patch).config.advanced.iris_restore;
    expect(full).not.toHaveProperty("lora_init");
    expect(full).not.toHaveProperty("resume_data_policy");
    const lora = snapshot(irisRestoreTarget(), { ...patch, networkType: "lora", resumeFromJobId: "job_0" }).config.advanced;
    expect(lora.iris_restore.lora_init).toBe("kaiming");
    expect(lora.iris_restore.resume_data_policy).toBe("new_phase");
    expect(lora.resumeFromJobId).toBe("job_0");
  });

  it("drops ema_decay and refuses EMA weights once the EMA is off", () => {
    const patch = { irisTask: { ema_enabled: false, preview_weights: "ema" } };
    expect(snapshot(irisRestoreTarget(), patch).config.advanced.iris_restore).not.toHaveProperty("ema_decay");
    expect(errors(irisRestoreTarget(), patch).map((e) => e.field)).toContain("irisTask.preview_weights");
  });

  it("requires step fields to be multiples of 2 × gradient accumulation", () => {
    expect(errors(irisRestoreTarget())).toEqual([]);
    const odd = errors(irisRestoreTarget(), { steps: "10001" });
    expect(odd.map((e) => e.field)).toEqual(["steps"]);
    expect(odd[0].message).toMatch(/2 × the gradient accumulation \(2\)/);
    // accumulation 2 → multiples of 4: 10002 fails, 10000 passes.
    expect(errors(irisRestoreTarget(), { gradientAccumulation: "2", steps: "10002" }).map((e) => e.field)).toEqual(["steps"]);
    expect(errors(irisRestoreTarget(), { gradientAccumulation: "2" })).toEqual([]);
  });

  it("parses the degradation overrides as a JSON object", () => {
    const sent = snapshot(irisRestoreTarget(), { irisTask: { degradation: '{"use_hflip": true}' } });
    expect(sent.config.advanced.iris_restore.degradation).toEqual({ use_hflip: true });
    expect(errors(irisRestoreTarget(), { irisTask: { degradation: "[1]" } }).map((e) => e.field)).toEqual([
      "irisTask.degradation",
    ]);
  });

  it("names the training-only resources", () => {
    expect(irisTrainingResourceIds(irisRestoreTarget())).toEqual([
      "iris_restore_train_vgg16",
      "iris_restore_train_lpips_lin",
      "iris_restore_train_critic_trunk",
    ]);
  });
});

describe("task memory floors and previous runs", () => {
  it("keys depth floors by micro-batch pixels with no artifact level", () => {
    const target = irisDepthTarget();
    const draft = draftFor(target);
    expect(irisFloorSize(target, draft)).toBe(480 * 640);
    expect(irisMemoryFloors(target, "full", irisFloorSize(target, draft), "mlx")).toEqual([{ lane: "mlx", gib: 80, provenance: "declared" }]);
    // Batch 2 doubles the pixels: the next row up on Candle; beyond the table → no floor (refused).
    const small = { ...draft, irisTask: { ...draft.irisTask, depth_size: ["384", "384"] }, batchSize: "2" };
    expect(irisMemoryFloors(target, "full", irisFloorSize(target, small), "candle")).toEqual([{ lane: "candle", gib: 84, provenance: "extrapolated" }]);
    expect(irisMemoryFloors(target, "full", irisFloorSize(target, { ...draft, batchSize: "2" }), "mlx")).toEqual([]);
  });

  it("reads a checkpointed task run's full_ac rows only where the lane publishes them", () => {
    const target = irisDepthTarget();
    const size = irisFloorSize(target, draftFor(target));
    // Candle publishes full_ac rows beside its plain ones; the nested key is never read as a row.
    expect(irisMemoryFloors(target, "full", size, "candle", "full")).toEqual([{ lane: "candle", gib: 68, provenance: "extrapolated" }]);
    expect(irisMemoryFloors(target, "full", size, "candle")).toEqual([{ lane: "candle", gib: 84, provenance: "extrapolated" }]);
    // 1024²: a measured need that spilled off the card without checkpointing; measured with it.
    expect(irisMemoryFloors(target, "full", 1024 * 1024, "candle")).toEqual([{ lane: "candle", gib: 111, provenance: "measured_spill" }]);
    expect(irisMemoryFloors(target, "full", 1024 * 1024, "candle", "full")).toEqual([{ lane: "candle", gib: 68, provenance: "measured" }]);
    // MLX publishes none: the plain row, whatever the toggle says.
    expect(irisMemoryFloors(target, "full", size, "mlx", "full")).toEqual([{ lane: "mlx", gib: 80, provenance: "declared" }]);
    expect(irisFloorSummary([{ lane: "candle", gib: 111, provenance: "measured_spill" }])).toContain("spilling to system memory");
  });

  it("limits gradient checkpointing to the lanes the task target publishes", () => {
    for (const target of [irisDepthTarget(), irisRestoreTarget()]) {
      expect(gradientCheckpointingLanes(target)).toEqual(["candle"]);
      expect(gradientCheckpointingUnsupported(target, "mlx")).toBe(true);
      expect(gradientCheckpointingUnsupported(target, "candle")).toBe(false);
      // An unknown lane keeps the control: the worker refuses it typed where it is missing.
      expect(gradientCheckpointingUnsupported(target, null)).toBe(false);
      expect(gradientCheckpointingUnsupported(target)).toBe(false);
    }
    // A lane joining the published list is supported with no code change here.
    const both = { limits: { gradientCheckpointingLanes: ["candle", "mlx"] } };
    expect(gradientCheckpointingUnsupported(both, "mlx")).toBe(false);
    expect(gradientCheckpointingUnsupported({ limits: { gradientCheckpointingLanes: [] } }, "candle")).toBe(true);
    // Pinned off (Iris generation) on every lane; a target that says nothing is unrestricted.
    expect(gradientCheckpointingLanes(irisTarget())).toBeNull();
    for (const lane of ["mlx", "candle", null]) {
      expect(gradientCheckpointingUnsupported(irisTarget(), lane)).toBe(true);
      expect(gradientCheckpointingUnsupported({ limits: {} }, lane)).toBe(false);
    }
  });

  it("submits the toggle only on a lane that implements it", () => {
    const target = irisDepthTarget();
    const configDraft = { ...draftFor(target), outputName: "out", gradientCheckpointing: true };
    const submitted = (trainingLane) =>
      trainingConfigSnapshot({ activeDataset: DATASET, configDraft, selectedPreset: null, selectedTarget: target, trainingLane }).config
        .advanced.gradientCheckpointing;
    expect(submitted("candle")).toBe(true);
    expect(submitted("mlx")).toBe(false);
    expect(submitted(null)).toBe(true);
    expect(trainingConfigSnapshot({ activeDataset: DATASET, configDraft: { ...configDraft, gradientCheckpointing: false }, selectedPreset: null, selectedTarget: target, trainingLane: "candle" }).config.advanced.gradientCheckpointing).toBe(false);
  });

  it("keys restoration floors by batch × crop² per artifact", () => {
    const target = irisRestoreTarget();
    const draft = draftFor(target);
    expect(irisFloorSize(target, draft)).toBe(1024 * 1024);
    expect(irisMemoryFloors(target, "lora", irisFloorSize(target, draft), "mlx")).toEqual([{ lane: "mlx", gib: 56, provenance: "declared" }]);
    expect(irisMemoryFloors(target, "full", irisFloorSize(target, draft), "mlx")).toEqual([{ lane: "mlx", gib: 128, provenance: "lower_bound" }]);
  });

  it("lists a task's previous runs by the plan's base model", () => {
    const depthRun = irisJob({ id: "d1", targetId: "iris_3b_depth" });
    depthRun.payload.plan.target.baseModel = "iris_3b_depth";
    const jobs = [depthRun, irisJob({ id: "g1" })];
    expect(irisPreviousRuns(jobs, { baseModel: "iris_3b_depth" }).map((job) => job.id)).toEqual(["d1"]);
    expect(irisPreviousRuns(jobs).map((job) => job.id)).toEqual(["g1"]);
  });
});

describe("dataset check summary", () => {
  // The worker's real report shape (a worker test compares this fixture with a real report).
  it("counts the ok items as trainable, states readiness, explains the policy and lists the problems", () => {
    const summary = irisCheckSummary(workerReport);
    expect(summary.counts).toBe("1 trainable · 0 invalid · 1 unreadable");
    expect(summary.ready).toBe(false);
    expect(summary.state).toMatch(/Blocked/);
    expect(summary.policy).toMatch(/Aligned pairs/);
    expect(summary.problems.map((item) => item.itemId)).toEqual(["b"]);
    const ready = irisCheckSummary({ ...workerReport, errors: 0, trainable: true, items: workerReport.items.slice(0, 1) });
    expect(ready.ready).toBe(true);
    expect(ready.state).toBe("Ready to train");
  });
});

describe("sidecar files", () => {
  const items = [
    { id: "i1", displayName: "Photo One.png", path: "items/abc.png" },
    { id: "i2", displayName: "two.jpg", path: "items/two.jpg", irisDepth: { path: "s/two.npy", maskPath: "s/two_mask.png" } },
    { displayName: "unsaved.png" },
  ];

  it("strips the kind suffix only for depth and mask", () => {
    expect(sidecarStem("two_depth.png", "depth")).toBe("two");
    expect(sidecarStem("two.depth.npy", "depth")).toBe("two");
    expect(sidecarStem("two.mask.png", "mask")).toBe("two");
    expect(sidecarStem("two_depth.png", "degraded")).toBe("two_depth");
  });

  it("matches files to saved items by display or stored name and reports the rest", () => {
    const files = ["photo one.npy", "abc_depth.png", "two.depth.npy", "unsaved.npy", "nobody.npy"].map((name) => ({ name }));
    const { matches, unmatched } = matchSidecarFiles(items, files, "depth");
    expect(matches.map(({ item, file }) => `${item.id}:${file.name}`)).toEqual([
      "i1:photo one.npy",
      "i1:abc_depth.png",
      "i2:two.depth.npy",
    ]);
    expect(unmatched).toEqual(["unsaved.npy", "nobody.npy"]);
  });

  it("reads per-item status and survives a dataset save", () => {
    expect(itemSidecarState(items[1])).toEqual({ depth: true, mask: true, degraded: false });
    const asset = { id: "a2", displayName: "two.jpg" };
    const payload = datasetPayload({
      activeDataset: { items: [{ ...items[1], assetId: "a2", irisDegradedImage: "sidecars/two.png" }] },
      assetsById: new Map([["a2", asset]]),
      name: "Set",
      selectedAssetIds: ["a2"],
    });
    expect(payload.items[0].irisDepth).toEqual(items[1].irisDepth);
    expect(payload.items[0].irisDegradedImage).toBe("sidecars/two.png");
  });
});
