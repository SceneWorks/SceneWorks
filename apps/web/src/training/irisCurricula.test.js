import { describe, expect, it } from "vitest";

import { datasetPayload } from "./datasetHelpers.js";
import { curriculaInPlay, irisGlobalBatch, itemSplit, nestDotted, validationRows } from "./irisCurricula.js";
import { irisDraftFromDefaults, irisOptionsSnapshot, irisValidation } from "./irisOptions.js";

const draft = (iris, extra = {}) => ({ iris: { ...irisDraftFromDefaults({}), ...iris }, ...extra });

describe("Iris curricula options (sc-25689)", () => {
  it("nests the dotted keys into the engine's objects and sends only filled, in-play values", () => {
    const snapshot = irisOptionsSnapshot(
      draft(
        {
          stage: "stage2_512",
          parent_stage: "stage1_256",
          shape_policy: "bucket",
          aspect_ratio_bucket: "shared21-512",
          shape_align: "64",
          "repa.weight": "0.5",
          "repa.variant": "irepa",
          "repa.spatial_norm_gamma": "0.7",
          "dist.strategy": "fsdp2",
          "dist.sharding": "hybrid",
          "dist.mesh.dp_shard": "2",
          "dist.world.world_size": "4",
          "dist.world.share_device": true,
          "perf.lazy_sync": true,
          val_samples: "8",
        },
        { resumeFromJobId: "job_a" },
      ),
    );
    expect(snapshot.stage).toBe("stage2_512");
    expect(snapshot.repa).toEqual({ weight: 0.5, variant: "irepa", spatial_norm_gamma: 0.7 });
    expect(snapshot.dist).toEqual({
      strategy: "fsdp2",
      sharding: "hybrid",
      mesh: { dp_shard: 2 },
      world: { world_size: 4, share_device: true },
    });
    expect(snapshot.perf).toEqual({ lazy_sync: true });
    expect(snapshot.aspect_ratio_bucket).toBe("shared21-512");
    // Inert under these choices: area-only alignment, val_samples without validation.
    expect(snapshot.shape_align).toBeUndefined();
    expect(snapshot.val_samples).toBeUndefined();
  });

  it("hides what the run's own choices make inert", () => {
    expect(curriculaInPlay("repa.variant", draft({ "repa.weight": "0" }))).toBe(false);
    expect(curriculaInPlay("repa.variant", draft({ "repa.weight": "0.5" }))).toBe(true);
    expect(curriculaInPlay("dist.sharding", draft({ "dist.strategy": "ddp" }))).toBe(false);
    expect(curriculaInPlay("dist.mesh.dp_shard", draft({ "dist.strategy": "fsdp2", "dist.sharding": "full" }))).toBe(false);
    expect(curriculaInPlay("resume_optimizer", draft({}))).toBe(false);
    expect(curriculaInPlay("preview_order", draft({}, { sampleEvery: "0", samplePrompts: "a" }))).toBe(false);
    expect(curriculaInPlay("preview_order", draft({}, { sampleEvery: "10", samplePrompts: "a" }))).toBe(true);
    expect(curriculaInPlay("ac_selective_every", draft({ activation_checkpointing: "selective_layer" }))).toBe(true);
  });

  it("flags a global-batch mismatch and dcp under ddp before launch", () => {
    const config = draft(
      { expected_global_batch: "8", "dist.world.world_size": "2", "dist.checkpoint_io": "dcp" },
      { batchSize: "2", gradientAccumulation: "1" },
    );
    expect(irisGlobalBatch(config)).toBe(4);
    const fields = irisValidation(config).map((issue) => issue.field);
    expect(fields).toContain("iris.expected_global_batch");
    expect(fields).toContain("iris.dist.checkpoint_io");
  });

  // sc-25690: the engine's rendezvous bound and cancel grace are offered, sent nested under
  // `dist.world`, and — like the rank timeout — never as 0, which the engine refuses.
  it("offers the rank wait bounds and flags a zero before launch", () => {
    const bounds = { "dist.world.rendezvous_timeout_secs": "300", "dist.world.cancel_grace_secs": "60" };
    expect(irisOptionsSnapshot(draft(bounds)).dist).toEqual({
      world: { rendezvous_timeout_secs: 300, cancel_grace_secs: 60 },
    });
    for (const key of ["dist.world.timeout_secs", "dist.world.rendezvous_timeout_secs", "dist.world.cancel_grace_secs"]) {
      const fields = irisValidation(draft({ [key]: "0" })).map((issue) => issue.field);
      expect(fields).toContain(`iris.${key}`);
      expect(irisValidation(draft({ [key]: "1" })).map((issue) => issue.field)).not.toContain(`iris.${key}`);
    }
  });

  it("keeps a preset's tracking tags through the draft and back into the request", () => {
    const seeded = irisDraftFromDefaults({ tracking: { project: "iris", tags: ["stage1", "repa"] } });
    expect(seeded["tracking.tags"]).toEqual(["stage1", "repa"]);
    const snapshot = irisOptionsSnapshot({ iris: seeded });
    expect(snapshot.tracking).toEqual({ project: "iris", tags: ["stage1", "repa"] });
    // Typed as a comma list: trimmed, empty entries dropped.
    expect(irisOptionsSnapshot(draft({ "tracking.tags": [" a", "", "b "] })).tracking).toEqual({ tags: ["a", "b"] });
  });

  it("seeds a preset's nested bag into dotted drafts", () => {
    const seeded = irisDraftFromDefaults({ repa: { weight: 0.5 }, dist: { strategy: "fsdp2", world: { world_size: 2 } } });
    expect(seeded["repa.weight"]).toBe("0.5");
    expect(seeded["dist.strategy"]).toBe("fsdp2");
    expect(seeded["dist.world.world_size"]).toBe("2");
    expect(nestDotted({ "a.b": 1, c: 2 })).toEqual({ a: { b: 1 }, c: 2 });
  });

  it("carries the holdout flag through the dataset save", () => {
    expect(itemSplit({ a: true }, "a", null)).toBe("holdout");
    expect(itemSplit({ a: false }, "a", { split: "holdout" })).toBe("train");
    expect(itemSplit({}, "a", { split: "holdout" })).toBe("holdout");
    const asset = { id: "a", displayName: "a.png" };
    const payload = datasetPayload({
      activeDataset: null,
      assetsById: new Map([["a", asset]]),
      holdoutById: { a: true },
      name: "ds",
      selectedAssetIds: ["a"],
    });
    expect(payload.items[0].split).toBe("holdout");
  });

  it("tabulates holdout validation", () => {
    const rows = validationRows({ validation: [{ step: 50, "val/loss": 0.3, "val/loss_t0": 0.2 }] });
    expect(rows).toEqual([{ step: 50, loss: 0.3, bands: [0.2, undefined, undefined, undefined] }]);
  });
});
