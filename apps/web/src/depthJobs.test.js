import { describe, expect, it } from "vitest";
import {
  buildDepthJobBody,
  defaultDepthModel,
  depthAssetDetails,
  depthAvailability,
  depthModelFromAsset,
  depthProcessingSize,
  depthResolutionError,
  depthSettingsFromAsset,
} from "./depthJobs.js";
import { buildBatchJob } from "./batchOps.js";

const caps = (supported) => ({ features: { imageDepth: { supported } } });
const installed = [{ id: "iris_3b_depth", installState: "installed" }];

describe("Iris depth contract (sc-25682)", () => {
  it("is available only with a native lane AND a complete install, and says why otherwise", () => {
    expect(depthAvailability(installed, caps(true))).toEqual({ available: true, reason: null });
    expect(depthAvailability(installed, caps(false)).available).toBe(false);
    expect(depthAvailability(installed, {}).available).toBe(false);
    expect(depthAvailability([], caps(true)).reason).toMatch(/not in the model catalog/);
    expect(depthAvailability([{ id: "iris_3b_depth", installState: "missing" }], caps(true)).reason).toMatch(/Install Iris 3B Depth/);
    expect(depthAvailability([{ id: "iris_3b_depth", installState: "incomplete" }], caps(true)).reason).toMatch(/incomplete/);
    // The generation install is not a depth install.
    expect(depthAvailability([{ id: "iris_3b", installState: "installed" }], caps(true)).available).toBe(false);
  });

  // S3: a trained depth model (`depthInit: parent` training needs no release export) is runnable on
  // its own; only the release selection needs the release install.
  it("lets an installed trained depth model run without the release install", () => {
    const trained = { id: "my_depth", installState: "installed", trainedCheckpoint: { task: "depth" } };
    const releaseMissing = [{ id: "iris_3b_depth", installState: "missing" }, trained];
    expect(depthAvailability(releaseMissing, caps(true), "my_depth")).toEqual({ available: true, reason: null });
    expect(depthAvailability(releaseMissing, caps(true), "")).toMatchObject({ available: false });
    expect(depthAvailability(releaseMissing, caps(true), "").reason).toMatch(/trained depth models/);
    // The entry point (no selection yet) is offered when ANY depth model can run.
    expect(depthAvailability(releaseMissing, caps(true)).available).toBe(true);
    expect(depthAvailability([{ id: "iris_3b_depth", installState: "missing" }], caps(true)).available).toBe(false);
    // A trained model that is not installed, or of another task, never satisfies the check.
    expect(depthAvailability(releaseMissing, caps(true), "gone").reason).toMatch(/gone is not installed/);
    const restore = { id: "my_restore", installState: "installed", trainedCheckpoint: { task: "restoration" } };
    expect(depthAvailability([restore], caps(true), "my_restore").available).toBe(false);
    // Still needs the native lane.
    expect(depthAvailability(releaseMissing, caps(false), "my_depth").available).toBe(false);
    // The picker starts on the release when installed, else the first trained model.
    expect(defaultDepthModel(releaseMissing)).toBe("my_depth");
    expect(defaultDepthModel([...installed, trained])).toBe("");
  });

  it("bounds the max side exactly like the API and always allows native", () => {
    expect(depthResolutionError({ mode: "capped", maxSide: 1024 })).toBeNull();
    expect(depthResolutionError({ mode: "native" })).toBeNull();
    expect(depthResolutionError({ mode: "capped", maxSide: 63 })).toMatch(/between 64 and 8192/);
    expect(depthResolutionError({ mode: "capped", maxSide: 8193 })).toMatch(/between 64 and 8192/);
    expect(depthResolutionError({ mode: "capped", maxSide: Number.NaN })).not.toBeNull();
  });

  it("shows the model input the worker will run (upstream size law, half-to-even)", () => {
    expect(depthProcessingSize(4000, 3000, { mode: "capped", maxSide: 1024 })).toEqual({ width: 1024, height: 768 });
    expect(depthProcessingSize(100, 50, { mode: "capped", maxSide: 1024 })).toEqual({ width: 96, height: 48 });
    expect(depthProcessingSize(1000, 750, { mode: "native" })).toEqual({ width: 992, height: 752 });
    expect(depthProcessingSize(24, 40, { mode: "native" })).toEqual({ width: 32, height: 32 });
    expect(depthProcessingSize(0, 10, { mode: "native" })).toBeNull();
  });

  it("builds the typed body and reopens a depth asset's recorded settings", () => {
    const body = buildDepthJobBody({ project: { id: "p", name: "P" }, requestedGpu: "auto", sourceAssetId: "a", resolution: { mode: "native" } });
    expect(body).toEqual({ projectId: "p", projectName: "P", sourceAssetId: "a", resolution: { mode: "native" }, requestedGpu: "auto" });
    expect(depthSettingsFromAsset({ recipe: { rawAdapterSettings: { depth: { resolution: { mode: "capped", maxSide: 512 } } } } })).toEqual({ mode: "capped", maxSide: 512 });
    expect(depthSettingsFromAsset({})).toEqual({ mode: "capped", maxSide: 1024 });
  });

  it("describes a reloaded depth asset from its sidecar", () => {
    const rows = Object.fromEntries(
      depthAssetDetails({
        type: "depth",
        lineage: { sourceAssetId: "asset_src" },
        depth: {
          valueConvention: "relative_log_depth",
          dtype: "float32",
          shape: [6, 4],
          npyDescr: "<f4",
          layout: "row-major",
          source: { assetId: "asset_src", width: 4, height: 6, storedWidth: 6, storedHeight: 4, exifOrientation: 6 },
          preprocessing: { resolution: { mode: "capped", maxSide: 1024 }, modelWidth: 16, modelHeight: 16, inputResample: "none", outputResample: "none" },
          model: { id: "iris_3b_depth", backend: "mlx", computeDtype: "bfloat16", weightsRepo: "speridlabs/iris-3b", weightsRevision: "7445443349bc9abe", weightsSubdir: "depth", configSha256: "ab".repeat(32) },
        },
      }),
    );
    expect(rows.Values).toMatch(/relative_log_depth.*not metric/);
    expect(rows["Raw map"]).toMatch(/float32 6×4 \(<f4, row-major\)/);
    expect(rows.Source).toBe("asset_src · 4×6");
    expect(rows["EXIF orientation"]).toMatch(/6 applied \(stored 6×4\)/);
    expect(rows.Resolution).toBe("Capped at 1024px max side");
    expect(rows.Model).toBe("iris_3b_depth (mlx, bfloat16)");
    expect(rows["Config sha256"]).toBe("ab".repeat(32));
    expect(rows.Weights).toBe("speridlabs/iris-3b@7445443349bc depth");
    expect(depthAssetDetails({ type: "image" })).toEqual([]);
    // S5: a trained depth model's lineage names its Training Studio export, not the release weights.
    const trained = Object.fromEntries(
      depthAssetDetails({
        type: "depth",
        depth: {
          model: {
            id: "iris_3b_depth",
            weightsSource: "training",
            weightsRepo: null,
            trainedModelId: "my_depth",
            exportPath: "models/finetunes/my_depth/depth",
          },
        },
      }),
    );
    expect(trained.Weights).toBe("trained my_depth (models/finetunes/my_depth/depth)");
  });
});

describe("trained depth models (sc-25687)", () => {
  const project = { id: "p1", name: "P" };
  it("sends irisModel only for a trained model and reopens it from the recipe", () => {
    const body = (irisModel) => buildDepthJobBody({ project, sourceAssetId: "a1", irisModel });
    expect(body("mine_depth").irisModel).toBe("mine_depth");
    expect(body("")).not.toHaveProperty("irisModel");
    expect(body("iris_3b_depth")).not.toHaveProperty("irisModel");
    expect(depthModelFromAsset({ recipe: { rawAdapterSettings: { depth: { irisModel: "mine_depth" } } } })).toBe("mine_depth");
    expect(depthModelFromAsset({ type: "depth" })).toBe("");
    const job = buildBatchJob({ op: "depth", asset: { id: "a1" }, params: { irisModel: "mine_depth" }, project, requestedGpu: "auto" });
    expect(job.body.irisModel).toBe("mine_depth");
  });
});
