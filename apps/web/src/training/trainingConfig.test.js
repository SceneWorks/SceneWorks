import { describe, expect, it } from "vitest";

import { summarize } from "../validation/issues.js";
import {
  configDraftFromTarget,
  configReseedDecision,
  configValidation,
  ltx25WorkflowPlan,
  mergeCustomizedConfigDraft,
  subjectMaskBackgroundWeightDefault,
  subjectMaskCoverage,
  subjectMaskSubjectWeightDefault,
  subjectMaskWeightMax,
  targetSupportsSubjectMaskLoss,
  timestepTypeOptionsForTarget,
  gradientNoiseEtaMax,
  gradientNoiseEtaSuggested,
  gradientNoiseGammaDefault,
  gradientNoiseGammaMax,
  targetSupportsGradientNoise,
  trainingConfigSnapshot,
  targetSupportsDepthAnchoring,
  targetSupportsVaeAnchorLoss,
  targetSupportsLatentLpipsLoss,
  latentPerceptualLosses,
  latentLossCombinationRefusal,
  depthAnchoringCombinationRefusal,
  auxModelsInstallNote,
  trainingAdapterModelId,
  depthAnchoringNoVideoLtxWorkflows,
  depthAnchoringEveryMax,
  depthAnchoringModelOptions,
  depthAnchoringWeightMax,
  depthAnchoringWeightSuggested,
  faceLossEveryMax,
  faceLossWeightMax,
  faceLossWeightSuggested,
  identityLossReferenceOptions,
  targetSupportsFaceLandmarkLoss,
  targetSupportsIdentityLoss,
  faceLossCombinationRefusal,
  faceLossIssues,
  resolutionBucketRepeatsMax,
  resolutionBucketsMax,
  resolutionBucketStride,
  seedResolutionBuckets,
  targetSupportsResolutionBuckets,
  targetSupportsWeightNoise,
  weightNoiseSigmaMax,
  weightNoiseSigmaSuggested,
  bodyLosses,
  bodyLossCombinationRefusal,
  bodyLossEveryMax,
  bodyLossWeightMax,
  bodyLossWeightSuggested,
  targetSupportsBodyLoss,
} from "./trainingConfig.js";

const ltxWorkflows = [
  "i2v_lora", "t2v_lora", "v2a_lora", "a2v_lora", "t2a_lora",
  "video_extend_lora", "video_inpainting_lora", "video_outpainting_lora",
  "video_suffix_lora", "audio_extend_lora", "audio_inpainting_lora",
  "audio_suffix_lora", "av2av_ic_lora", "v2v_ic_lora", "a2a_ic_lora",
];

// sc-4199: configDraftFromTarget (target/preset → form draft) and
// trainingConfigSnapshot (form draft → worker payload) were pure builders buried
// in the 2.1k-line TrainingStudio screen. Extracted to ../training/trainingConfig.js,
// they are directly testable.

const target = {
  id: "sdxl_lora",
  outputKind: "lora",
  defaults: {
    rank: 8,
    alpha: 8,
    learningRate: 0.0001,
    steps: 1000,
    batchSize: 1,
    gradientAccumulation: 1,
    resolution: 1024,
    saveEvery: 0,
    seed: 42,
    optimizer: "adamw",
    advanced: { networkType: "lora" },
  },
  limits: { networkTypes: ["lora", "lokr"] },
};

it("scopes the SD3 timestep extensions without dropping the shared uniform mode", () => {
  expect(timestepTypeOptionsForTarget({ kernel: "anima_lora" })).toEqual([
    "sigmoid",
    "linear",
    "uniform",
    "weighted",
  ]);
  expect(timestepTypeOptionsForTarget({ kernel: "mage_flow_lora" })).not.toContain(
    "logit_normal",
  );
  expect(timestepTypeOptionsForTarget({ kernel: "sd3_lora" })).toEqual([
    "sigmoid",
    "linear",
    "uniform",
    "weighted",
    "default",
    "logit_normal",
  ]);
});

const dataset = { id: "ds-1", version: 3, name: "Kelsie" };

describe("configDraftFromTarget", () => {
  it("seeds the output name from the dataset name + output kind label", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    expect(draft.outputName).toBe("Kelsie LoRA");
  });

  it("stringifies numeric defaults into form drafts", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    expect(draft.rank).toBe("8");
    expect(draft.learningRate).toBe("0.0001");
    expect(draft.steps).toBe("1000");
    expect(draft.seed).toBe("42");
  });

  it("falls back to the first GPU when the advanced requestedGpu is not offered", () => {
    const gpuTarget = { ...target, defaults: { ...target.defaults, advanced: { ...target.defaults.advanced, requestedGpu: "7" } } };
    expect(configDraftFromTarget(gpuTarget, dataset, ["auto", "0"]).requestedGpu).toBe("auto");
    expect(configDraftFromTarget(gpuTarget, dataset, ["auto", "7"]).requestedGpu).toBe("7");
  });

  it("prefers the explicit trigger phrase over the target default", () => {
    const triggerTarget = { ...target, defaults: { ...target.defaults, triggerWord: "fallback" } };
    expect(configDraftFromTarget(triggerTarget, dataset, ["auto"], "ohwx woman").triggerWord).toBe("ohwx woman");
    expect(configDraftFromTarget(triggerTarget, dataset, ["auto"], "").triggerWord).toBe("fallback");
  });

  it("carries a preset's outputName through previousDraft instead of reseeding", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"], "", null, { outputName: "Custom name" });
    expect(draft.outputName).toBe("Custom name");
  });

  // sc-10689: batchSize and gradientAccumulation are validated with a `> 0` rule and now
  // have inputs. A target/preset whose defaults omit either would seed "" and fail that
  // rule with no fixable box. The draft floors both so the box is never empty.
  it("floors batch size and gradient accumulation so an omitting target can't seed a dead CTA", () => {
    const { batchSize: _b, gradientAccumulation: _g, ...leanDefaults } = target.defaults;
    const leanTarget = { ...target, defaults: leanDefaults };
    const draft = configDraftFromTarget(leanTarget, dataset, ["auto"]);
    expect(draft.batchSize).toBe("1");
    expect(draft.gradientAccumulation).toBe("1");
    const summary = summarize(configValidation(draft, { activeDataset: dataset, selectedTarget: leanTarget }));
    expect(summary.invalidFields.has("batchSize")).toBe(false);
    expect(summary.invalidFields.has("gradientAccumulation")).toBe(false);
  });

  it("carries an explicit batch size / gradient accumulation from the target defaults", () => {
    const richTarget = { ...target, defaults: { ...target.defaults, batchSize: 4, gradientAccumulation: 2 } };
    const draft = configDraftFromTarget(richTarget, dataset, ["auto"]);
    expect(draft.batchSize).toBe("4");
    expect(draft.gradientAccumulation).toBe("2");
  });

  it("defaults the LoKr factor to an auto -1 string and normalizes the adapter version", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    expect(draft.decomposeFactor).toBe("-1");
    const versioned = {
      ...target,
      defaults: { ...target.defaults, advanced: { ...target.defaults.advanced, trainingAdapterVersion: "v2-default" } },
    };
    expect(configDraftFromTarget(versioned, dataset, ["auto"]).trainingAdapterVersion).toBe("v2");
  });
});

describe("trainingConfigSnapshot", () => {
  function snapshot(configDraft, extra = {}) {
    return trainingConfigSnapshot({
      activeDataset: dataset,
      configDraft: { ...configDraft, outputName: "Kelsie LoRA" },
      selectedTarget: target,
      ...extra,
    });
  }

  it("coerces form drafts back into numbers for the worker config", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    const snap = snapshot(draft);
    expect(snap.config.rank).toBe(8);
    expect(snap.config.learningRate).toBe(0.0001);
    expect(snap.config.steps).toBe(1000);
    expect(snap.config.optimizer).toBe("adamw");
  });

  it("threads dataset + output identity and defaults to a dry run", () => {
    const snap = snapshot(configDraftFromTarget(target, dataset, ["auto"]));
    expect(snap.targetId).toBe("sdxl_lora");
    expect(snap.datasetId).toBe("ds-1");
    expect(snap.datasetVersion).toBe(3);
    expect(snap.outputName).toBe("Kelsie LoRA");
    expect(snap.dryRun).toBe(true);
  });

  it("honors an explicit dryRun=false for a real run", () => {
    const snap = snapshot(configDraftFromTarget(target, dataset, ["auto"]), { dryRun: false });
    expect(snap.dryRun).toBe(false);
  });

  it("omits the LoKr factor for a lora network but keeps it for lokr", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    expect(snapshot(draft).config.advanced).not.toHaveProperty("decomposeFactor");
    const lokr = snapshot({ ...draft, networkType: "lokr", decomposeFactor: "16" });
    expect(lokr.config.advanced.networkType).toBe("lokr");
    expect(lokr.config.advanced.decomposeFactor).toBe(16);
  });

  it("drops a blank LoKr factor so the worker applies its own -1 default", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    const snap = snapshot({ ...draft, networkType: "lokr", decomposeFactor: "" });
    expect(snap.config.advanced).not.toHaveProperty("decomposeFactor");
  });

  it("uses the platform-effective full-finetune contract without changing MLX defaults", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    const mlx = snapshot({
      ...draft,
      networkType: "full",
      precision: "bf16",
      gradientCheckpointing: true,
    });
    expect(mlx.config.advanced.networkType).toBe("full");
    expect(mlx.config.advanced.mixedPrecision).toBe("bf16");
    expect(mlx.config.advanced.gradientCheckpointing).toBe(true);

    const candleTarget = {
      ...target,
      defaults: {
        ...target.defaults,
        advanced: {
          ...target.defaults.advanced,
          fullFinetuneConfig: {
            mixedPrecision: "f32",
            gradientCheckpointing: false,
          },
        },
      },
    };
    const candle = trainingConfigSnapshot({
      activeDataset: dataset,
      configDraft: { ...draft, networkType: "full", precision: "bf16", gradientCheckpointing: true },
      selectedPreset: null,
      selectedTarget: candleTarget,
    });
    expect(candle.config.advanced.mixedPrecision).toBe("f32");
    expect(candle.config.advanced.gradientCheckpointing).toBe(false);
    expect(candle.config.advanced).not.toHaveProperty("fullFinetuneConfig");
  });

  it("derives sample prompts from the trigger word", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"], "ohwx woman");
    const snap = snapshot(draft);
    expect(snap.config.advanced.samplePrompts[0]).toContain("ohwx woman");
    expect(snap.config.advanced.samplePrompts).toHaveLength(4);
  });

  it("prefills the sample-prompts draft and defaults the sample count to 4 (sc-8671)", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"], "ohwx woman");
    expect(draft.sampleCount).toBe("4");
    expect(draft.samplePrompts.split("\n")).toHaveLength(4);
    expect(draft.samplePrompts).toContain("ohwx woman");
  });

  it("sends the user's edited prompt pool verbatim and a custom count (sc-8671)", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"], "ohwx woman");
    const snap = snapshot({ ...draft, samplePrompts: "a cat\n  a dog  \n\na bird", sampleCount: "6" });
    expect(snap.config.advanced.sampleCount).toBe(6);
    // Blank lines dropped, surviving lines trimmed; web sends the raw pool (backends cap at count).
    expect(snap.config.advanced.samplePrompts).toEqual(["a cat", "a dog", "a bird"]);
  });

  it("falls back to trigger-derived prompts when the pool is cleared (sc-8671)", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"], "ohwx woman");
    const snap = snapshot({ ...draft, samplePrompts: "   \n  " });
    expect(snap.config.advanced.samplePrompts).toHaveLength(4);
    expect(snap.config.advanced.samplePrompts[0]).toContain("ohwx woman");
  });

  it("drops a blank sample count so the worker applies its own default (sc-8671)", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"], "ohwx woman");
    const snap = snapshot({ ...draft, sampleCount: "" });
    expect(snap.config.advanced).not.toHaveProperty("sampleCount");
  });

  it("carries the preset id/version when a preset is selected", () => {
    const snap = snapshot(configDraftFromTarget(target, dataset, ["auto"]), {
      selectedPreset: { id: "preset-1", version: 5 },
    });
    expect(snap.presetId).toBe("preset-1");
    expect(snap.presetVersion).toBe(5);
  });
});

describe("LTX-2.5 advanced workflow contract", () => {
  const ltxTarget = {
    ...target,
    id: "ltx_2_5_video_lora",
    baseModel: "ltx_2_5",
    defaults: {
      ...target.defaults,
      triggerWord: "subject",
      advanced: {
        ...target.defaults.advanced,
        ltxWorkflow: "t2v_lora",
        ltxVideo: { isGenerated: true, conditions: [] },
        ltxAudio: { isGenerated: true, conditions: [] },
      },
    },
    limits: { ...target.limits, ltxWorkflows },
  };
  const preparedDataset = {
    ...dataset,
    items: [{ id: "item_1", ltxPreparedBundlePath: "prepared/item_1.safetensors" }],
  };

  it.each(ltxWorkflows)("submits the canonical %s video/audio plan", (workflow) => {
    const draft = configDraftFromTarget(ltxTarget, preparedDataset, ["auto"]);
    const plan = ltx25WorkflowPlan(workflow);
    const snapshot = trainingConfigSnapshot({
      activeDataset: preparedDataset,
      selectedTarget: ltxTarget,
      configDraft: { ...draft, ltxWorkflow: workflow, ltxVideo: plan.video, ltxAudio: plan.audio },
    });
    expect(snapshot.config.advanced.ltxWorkflow).toBe(workflow);
    expect(snapshot.config.advanced.ltxVideo).toEqual(plan.video ?? undefined);
    expect(snapshot.config.advanced.ltxAudio).toEqual(plan.audio ?? undefined);
    expect(snapshot.config.advanced.ltxValidation).toMatchObject({
      width: 960,
      height: 544,
      frames: 89,
      fps: 24,
      steps: 30,
      stgBlocks: [28],
      generateAudio: true,
    });
  });

  it("fails closed until every saved item has a prepared bundle", () => {
    const draft = configDraftFromTarget(ltxTarget, preparedDataset, ["auto"]);
    const issues = configValidation(draft, {
      selectedTarget: ltxTarget,
      activeDataset: { ...preparedDataset, items: [{ id: "item_1" }] },
    });
    expect(issues.some((entry) => entry.message.includes("prepared bundle"))).toBe(true);
  });

  it("round-trips and submits bounded non-default validation controls", () => {
    const overrides = {
      width: 1280,
      height: 704,
      frames: 97,
      fps: 30,
      steps: 45,
      videoCfgScale: 0,
      audioCfgScale: 9.5,
      videoStgScale: 2,
      audioStgScale: 0,
      stgBlocks: [0],
      guidanceRescale: 0,
      videoModalityGuidanceScale: 4.25,
      audioModalityGuidanceScale: 0,
      generateAudio: false,
    };
    const targetWithOverrides = {
      ...ltxTarget,
      defaults: {
        ...ltxTarget.defaults,
        advanced: { ...ltxTarget.defaults.advanced, ltxValidation: overrides },
      },
    };
    const draft = configDraftFromTarget(targetWithOverrides, preparedDataset, ["auto"]);
    expect(draft.ltxValidation).toEqual(overrides);
    expect(configValidation(draft, {
      selectedTarget: targetWithOverrides,
      activeDataset: preparedDataset,
    }).filter((entry) => entry.field === "ltxValidation")).toEqual([]);
    const submitted = trainingConfigSnapshot({
      activeDataset: preparedDataset,
      selectedTarget: targetWithOverrides,
      configDraft: draft,
    });
    expect(submitted.config.advanced.ltxValidation).toEqual(overrides);
  });

  it.each([
    ["width", 1000],
    ["height", 16],
    ["frames", 90],
    ["frames", 265],
    ["fps", 121],
    ["steps", 0],
    ["videoCfgScale", -0.1],
    ["audioCfgScale", 21],
    ["guidanceRescale", 1.1],
    ["videoModalityGuidanceScale", Number.POSITIVE_INFINITY],
    ["stgBlocks", []],
    ["stgBlocks", [1, 2]],
    ["stgBlocks", [48]],
  ])("rejects invalid validation override %s=%j", (field, value) => {
    const draft = configDraftFromTarget(ltxTarget, preparedDataset, ["auto"]);
    const issues = configValidation({
      ...draft,
      ltxValidation: { ...draft.ltxValidation, [field]: value },
    }, {
      selectedTarget: ltxTarget,
      activeDataset: preparedDataset,
    });
    expect(issues.some((entry) => entry.field === "ltxValidation")).toBe(true);
  });
});

// The training config's rule set, now expressed in the app-wide vocabulary (epic 10644, sc-10647).
// The kinds are the contract: a `requirement` blocks in silence, an `error` blocks and speaks. Get
// one wrong and either the form nags about an empty box or Start dies with no stated reason.
describe("configValidation", () => {
  const wholeDraft = {
    outputName: "Kelsie LoRA",
    triggerWord: "kelsie",
    rank: 8,
    alpha: 8,
    learningRate: 0.0001,
    steps: 1000,
    resolution: 1024,
    batchSize: 1,
    gradientAccumulation: 1,
    saveEvery: 250,
  };
  const ctx = { activeDataset: dataset, selectedTarget: target };
  const kindsOf = (issues, field) => issues.filter((entry) => entry.field === field).map((entry) => entry.kind);

  it("finds nothing wrong with a whole draft", () => {
    expect(configValidation(wholeDraft, ctx)).toEqual([]);
  });

  it("marks the unfilled fields as requirements, which block without speaking", () => {
    const issues = configValidation({ ...wholeDraft, outputName: "", triggerWord: "  " }, { activeDataset: null, selectedTarget: null });
    expect(kindsOf(issues, "target")).toEqual(["requirement"]);
    expect(kindsOf(issues, "dataset")).toEqual(["requirement"]);
    expect(kindsOf(issues, "outputName")).toEqual(["requirement"]);
    expect(kindsOf(issues, "triggerWord")).toEqual(["requirement"]);
    expect(summarize(issues).surfaced).toEqual([]);
    expect(summarize(issues).ready).toBe(false);
  });

  // Every numeric rule, not a sample: one mis-kinded field is exactly what a sampled table misses.
  for (const field of ["rank", "alpha", "learningRate", "steps", "resolution", "batchSize", "gradientAccumulation", "saveEvery"]) {
    it(`raises an error when ${field} is cleared, and names ${field} as its field`, () => {
      const issues = configValidation({ ...wholeDraft, [field]: "" }, ctx);
      expect(kindsOf(issues, field)).toEqual(["error"]);
      expect(summarize(issues).surfaced).toHaveLength(1);
      expect(summarize(issues).invalidFields.has(field)).toBe(true);
    });

    it(`raises an error when ${field} is zero or negative`, () => {
      expect(kindsOf(configValidation({ ...wholeDraft, [field]: 0 }, ctx), field)).toEqual(["error"]);
      expect(kindsOf(configValidation({ ...wholeDraft, [field]: -1 }, ctx), field)).toEqual(["error"]);
    });
  }

  it("names the output after the target's output kind", () => {
    const issues = configValidation({ ...wholeDraft, outputName: "" }, ctx);
    expect(issues.find((entry) => entry.field === "outputName").message).toBe("Name the LoRA output");
  });

  // sc-15036: the output kind is a per-RUN property. The SAME target produces an adapter or a full
  // base checkpoint depending on `networkType`, so the requirement must name what THIS run will
  // produce. Discriminating: one draft, one field flipped, two different messages.
  it("names a full base fine-tune's output a base checkpoint, not a LoRA", () => {
    const messageFor = (networkType) =>
      configValidation({ ...wholeDraft, outputName: "", networkType }, ctx).find(
        (entry) => entry.field === "outputName",
      ).message;
    expect(messageFor("lora")).toBe("Name the LoRA output");
    expect(messageFor("lokr")).toBe("Name the LoRA output");
    expect(messageFor("full")).toBe("Name the base checkpoint output");
    expect(messageFor("  FULL ")).toBe("Name the base checkpoint output");
  });

  // A broken value and an unfilled field at once: the chip row shows one and stays silent on the
  // other. This is the pairing sc-10492 collapsed and sc-10501 restored.
  it("surfaces the broken value alone when both kinds are live", () => {
    const summary = summarize(configValidation({ ...wholeDraft, outputName: "", rank: "" }, ctx));
    expect(summary.surfaced.map((entry) => entry.message)).toEqual(["Rank must be greater than zero"]);
    expect(summary.invalidFields.has("outputName")).toBe(false);
    expect(summary.ready).toBe(false);
  });

  it("tolerates a missing context", () => {
    expect(() => configValidation(wholeDraft)).not.toThrow();
  });

  // Readiness rides in the same summary as the draft rules (sc-10648), so the Train button
  // has one reason-set instead of a separate `disabled` term. A Blocked dataset is a
  // form-scoped error — the fix is in Data Sets, not an input here.
  describe("dataset readiness gate", () => {
    it("adds a form-scoped error when the dataset is not ready to train", () => {
      const issues = configValidation(wholeDraft, { ...ctx, datasetNotReady: true });
      const readiness = issues.find((entry) => entry.message.includes("isn’t ready to train"));
      expect(readiness).toBeTruthy();
      expect(readiness.kind).toBe("error");
      expect(readiness.field).toBeNull();
      expect(summarize(issues).ready).toBe(false);
    });

    it("says nothing when the dataset is trainable", () => {
      const issues = configValidation(wholeDraft, { ...ctx, datasetNotReady: false });
      expect(issues.some((entry) => entry.message.includes("ready to train"))).toBe(false);
      expect(summarize(issues).ready).toBe(true);
    });

    it("defaults to trainable when readiness is unknown", () => {
      expect(summarize(configValidation(wholeDraft, ctx)).ready).toBe(true);
    });
  });
});

// sc-11970: the config-draft basis effect must NOT wipe a user's config edits when the
// trainingPresets catalog resolves ASYNC. These two pure helpers encode that decision so
// the (hook-driven) screen effect stays thin and the async-vs-user distinction is testable.
describe("configReseedDecision (sc-11970)", () => {
  const basis = (over = {}) => ({ targetId: "sdxl_lora", datasetId: "ds-1", presetId: "", ...over });

  it("seeds on the first pass (empty previous basis)", () => {
    expect(configReseedDecision({ targetId: "", datasetId: "", presetId: "" }, basis(), 0)).toBe("seed");
  });

  it("seeds when the user switches target", () => {
    const prev = basis({ presetId: "balanced" });
    const next = basis({ targetId: "z_image_lora", presetId: "balanced" });
    // Even with customized fields, a genuine target switch fully re-seeds.
    expect(configReseedDecision(prev, next, 3)).toBe("seed");
  });

  it("seeds when the user switches dataset", () => {
    const prev = basis({ presetId: "balanced" });
    const next = basis({ datasetId: "ds-2", presetId: "balanced" });
    expect(configReseedDecision(prev, next, 3)).toBe("seed");
  });

  it("MERGES (does not wipe) when only the default preset resolves async AND the user has edits", () => {
    // target + dataset stable, preset id appears (catalog just loaded), user tweaked fields.
    const prev = basis({ presetId: "" });
    const next = basis({ presetId: "balanced" });
    expect(configReseedDecision(prev, next, 2)).toBe("merge");
  });

  it("seeds on an async preset resolve when the user has NOT customized anything", () => {
    const prev = basis({ presetId: "" });
    const next = basis({ presetId: "balanced" });
    expect(configReseedDecision(prev, next, 0)).toBe("seed");
  });

  it("no-ops when nothing changed", () => {
    const prev = basis({ presetId: "balanced" });
    expect(configReseedDecision(prev, { ...prev }, 4)).toBe("noop");
  });
});

describe("mergeCustomizedConfigDraft (sc-11970)", () => {
  it("overlays only the customized fields onto the freshly-seeded draft", () => {
    const seeded = { learningRate: "0.0001", steps: "1000", rank: "8" };
    const current = { learningRate: "0.0005", steps: "2500", rank: "16" };
    const merged = mergeCustomizedConfigDraft(seeded, current, new Set(["learningRate", "steps"]));
    // Customized fields keep the user's value; untouched fields take the preset seed.
    expect(merged.learningRate).toBe("0.0005");
    expect(merged.steps).toBe("2500");
    expect(merged.rank).toBe("8");
  });

  it("ignores customized field names absent from the current draft", () => {
    const seeded = { learningRate: "0.0001" };
    const merged = mergeCustomizedConfigDraft(seeded, {}, new Set(["learningRate", "ghost"]));
    expect(merged).toEqual({ learningRate: "0.0001" });
  });

  it("returns a fresh object (no mutation of the seed)", () => {
    const seeded = { steps: "1000" };
    const merged = mergeCustomizedConfigDraft(seeded, { steps: "2000" }, new Set(["steps"]));
    expect(merged).not.toBe(seeded);
    expect(seeded.steps).toBe("1000");
  });
});

// ControlNet preprocessor provisioning. A `control_branch` run renders its per-image condition with
// a preprocessor whose resolver is cache-only since epic 17625, so a missing one is a job-time
// failure — it has to gate Start training, not merely warn beside it.
describe("configValidation — missing control preprocessor", () => {
  const dataset = { id: "ds_1", items: [{ id: "i1" }] };
  const controlTarget = { ...target, id: "krea_pose_control", outputKind: "control_branch" };
  const draft = () => configDraftFromTarget(controlTarget, dataset, ["auto"]);
  // Assert on THIS rule's issue rather than the whole summary's `ready`: readiness also depends on
  // unrelated free-text fields, so a `ready === false` assertion would pass whether or not this rule
  // fires. Every issue blocks Start training, so the issue's presence IS the block.
  const preprocessorIssues = (missingControlModels) =>
    configValidation(draft(), {
      activeDataset: dataset,
      selectedTarget: controlTarget,
      missingControlModels,
    }).filter((entry) => /to render this run's control condition/.test(entry.message));

  it("blocks Start training and names the model", () => {
    const issues = preprocessorIssues([{ id: "dwpose_pose_detector", name: "DWPose Pose Detector" }]);
    expect(issues).toHaveLength(1);
    expect(issues[0].message).toContain("DWPose Pose Detector");
    // An `error`, not a silent `requirement`: nothing on the form explains the dead button, so it
    // earns a chip (the sc-10501 distinction). `summarize` surfaces errors and hides requirements.
    expect(summarize(issues).surfaced).toHaveLength(1);
  });

  it("lists every missing model in one issue", () => {
    const issues = preprocessorIssues([
      { id: "person_detector", name: "YOLO11m Person Detector" },
      { id: "dwpose_pose_detector", name: "DWPose Pose Detector" },
    ]);
    expect(issues).toHaveLength(1);
    expect(issues[0].message).toContain("YOLO11m Person Detector and DWPose Pose Detector");
  });

  // The regression that matters: every LoRA run, and every ControlNet run on a provisioned box,
  // passes an empty list (or nothing at all) and must be completely unaffected.
  it("adds no issue when nothing is missing", () => {
    expect(preprocessorIssues([])).toEqual([]);
    expect(preprocessorIssues(undefined)).toEqual([]);
    expect(
      configValidation(draft(), { activeDataset: dataset, selectedTarget: controlTarget }),
    ).toEqual(configValidation(draft(), { activeDataset: dataset, selectedTarget: controlTarget, missingControlModels: [] }));
  });
});

// sc-25213: the de-distill training adapter — the draft's version maps to the catalog model the
// worker resolves (the presets' "v2-default" → v2), a missing one gates Start, and the selection
// round-trips into the job's training snapshot. Mutation: map "v2-default" to v1 ⇒ red.
describe("de-distill training adapter (sc-25213)", () => {
  const repo = "ostris/zimage_turbo_training_adapter";
  it("maps each version to the catalog model the worker loads", () => {
    const id = (trainingAdapterVersion, trainingAdapterRepo = repo) =>
      trainingAdapterModelId({ trainingAdapterRepo, trainingAdapterVersion });
    expect(id("v1")).toBe("zimage_turbo_training_adapter_v1");
    expect(id("v2")).toBe("zimage_turbo_training_adapter_v2");
    expect(id("v2-default")).toBe("zimage_turbo_training_adapter_v2");
    expect(id("")).toBe("zimage_turbo_training_adapter_v2");
    expect(id("v3")).toBeNull();
    expect(id("v1", "")).toBeNull();
    expect(id("v1", "someone/else")).toBeNull();
  });

  it("blocks Start training while the adapter is not installed", () => {
    const dataset = { id: "ds_1", items: [{ id: "i1" }] };
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    const adapterIssues = (missingTrainingAdapterModels) =>
      configValidation(draft, { activeDataset: dataset, selectedTarget: target, missingTrainingAdapterModels }).filter(
        (entry) => /this run trains with it/.test(entry.message),
      );
    const issues = adapterIssues([{ id: "zimage_turbo_training_adapter_v2", name: "Z-Image Turbo Training Adapter v2" }]);
    expect(issues).toHaveLength(1);
    expect(issues[0].message).toContain("Z-Image Turbo Training Adapter v2");
    expect(issues[0].field).toBe("trainingAdapterVersion");
    expect(summarize(issues).surfaced).toHaveLength(1);
    expect(adapterIssues([])).toEqual([]);
    expect(adapterIssues(undefined)).toEqual([]);
  });

  it("round-trips the selection into the job snapshot", () => {
    const dataset = { id: "ds_1", items: [{ id: "i1" }] };
    const draft = {
      ...configDraftFromTarget(target, dataset, ["auto"]),
      trainingAdapterRepo: repo,
      trainingAdapterVersion: "v1",
    };
    const advanced = trainingConfigSnapshot({
      activeDataset: dataset,
      configDraft: { ...draft, outputName: "Kelsie LoRA" },
      selectedTarget: target,
    }).config.advanced;
    expect(advanced.trainingAdapterRepo).toBe(repo);
    expect(advanced.trainingAdapterVersion).toBe("v1");
  });
});

// sc-24826 (epic 2123): the weight-noise knob is off by default, round-trips from the form draft
// into the job's training snapshot, and is bounded by the same max the API enforces.
describe("weight noise (sc-24826)", () => {
  const snap = (draft) =>
    trainingConfigSnapshot({
      activeDataset: dataset,
      configDraft: { ...draft, outputName: "Kelsie LoRA" },
      selectedTarget: target,
    });

  it("seeds off and leaves a default snapshot without the key", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    expect(draft.weightNoiseSigma).toBe("");
    expect(snap(draft).config.advanced).not.toHaveProperty("weightNoiseSigma");
  });

  it("round-trips an enabled sigma into the training snapshot as a number", () => {
    const draft = { ...configDraftFromTarget(target, dataset, ["auto"]), weightNoiseSigma: String(weightNoiseSigmaSuggested) };
    expect(snap(draft).config.advanced.weightNoiseSigma).toBe(0.0125);
    // ...and a preset/target that carries it seeds the draft back.
    const seeded = configDraftFromTarget(
      { ...target, defaults: { ...target.defaults, advanced: { networkType: "lora", weightNoiseSigma: 0.02 } } },
      dataset,
      ["auto"],
    );
    expect(seeded.weightNoiseSigma).toBe("0.02");
    expect(snap(seeded).config.advanced.weightNoiseSigma).toBe(0.02);
  });

  it("uses the API's bound and the upstream suggestion", () => {
    expect(weightNoiseSigmaMax).toBe(0.1);
    expect(weightNoiseSigmaSuggested).toBe(0.0125);
  });

  it("flags a negative, above-limit, or full-fine-tune sigma on the weightNoiseSigma field", () => {
    const whole = {
      outputName: "Kelsie LoRA",
      triggerWord: "kelsie",
      rank: 8,
      alpha: 8,
      learningRate: 0.0001,
      steps: 1000,
      resolution: 1024,
      batchSize: 1,
      gradientAccumulation: 1,
      saveEvery: 250,
    };
    const ctx = { activeDataset: dataset, selectedTarget: target };
    const fieldIssues = (draft) => configValidation(draft, ctx).filter((entry) => entry.field === "weightNoiseSigma");
    for (const ok of ["", "0", "0.0125", String(weightNoiseSigmaMax)]) {
      expect(fieldIssues({ ...whole, weightNoiseSigma: ok })).toEqual([]);
    }
    for (const bad of ["-0.01", String(weightNoiseSigmaMax + 0.001), "abc"]) {
      expect(fieldIssues({ ...whole, weightNoiseSigma: bad }).map((entry) => entry.kind)).toEqual(["error"]);
    }
    expect(fieldIssues({ ...whole, networkType: "full", weightNoiseSigma: "0.0125" })).toHaveLength(1);
  });
});

// sc-2125 (epic 2123): depth anchoring is off by default, round-trips from the form draft into the
// job's training snapshot only while on, and every knob is bounded by the API's limits (E6).
describe("depth anchoring (sc-2125)", () => {
  // Depth anchoring is offered only where the target advertises it (MLX Z-Image today).
  const depthTarget = { ...target, limits: { ...target.limits, supportsDepthAnchoring: true } };
  const snap = (draft) =>
    trainingConfigSnapshot({
      activeDataset: dataset,
      configDraft: { ...draft, outputName: "Kelsie LoRA" },
      selectedTarget: depthTarget,
    });
  const whole = {
    outputName: "Kelsie LoRA",
    triggerWord: "kelsie",
    rank: 8,
    alpha: 8,
    learningRate: 0.0001,
    steps: 1000,
    resolution: 1024,
    batchSize: 1,
    gradientAccumulation: 1,
    saveEvery: 250,
    depthAnchoringModel: "small",
  };
  const ctx = { activeDataset: dataset, selectedTarget: depthTarget };
  const issuesOn = (draft, field) =>
    configValidation({ ...whole, ...draft }, ctx).filter((entry) => entry.field === field);

  it("seeds off and leaves a default snapshot without any depth key", () => {
    const draft = configDraftFromTarget(depthTarget, dataset, ["auto"]);
    expect(draft.depthAnchoringWeight).toBe("");
    expect(draft.depthAnchoringModel).toBe("small");
    const advanced = snap(draft).config.advanced;
    for (const key of Object.keys(advanced)) {
      expect(key.startsWith("depthAnchoring")).toBe(false);
    }
  });

  it("round-trips an enabled configuration into the snapshot as typed values", () => {
    const draft = {
      ...configDraftFromTarget(depthTarget, dataset, ["auto"]),
      depthAnchoringWeight: String(depthAnchoringWeightSuggested),
      depthAnchoringModel: "large",
      depthAnchoringMinT: "0.2",
      depthAnchoringMaxT: "0.8",
      depthAnchoringEvery: "3",
    };
    const advanced = snap(draft).config.advanced;
    expect(advanced.depthAnchoringWeight).toBe(0.1);
    expect(advanced.depthAnchoringModel).toBe("large");
    expect(advanced.depthAnchoringMinT).toBe(0.2);
    expect(advanced.depthAnchoringMaxT).toBe(0.8);
    expect(advanced.depthAnchoringEvery).toBe(3);
    // A preset/target carrying the keys seeds the draft back.
    const seeded = configDraftFromTarget(
      {
        ...target,
        defaults: { ...target.defaults, advanced: { networkType: "lora", depthAnchoringWeight: 0.05, depthAnchoringModel: "base" } },
      },
      dataset,
      ["auto"],
    );
    expect(seeded.depthAnchoringWeight).toBe("0.05");
    expect(seeded.depthAnchoringModel).toBe("base");
  });

  it("uses the API's bounds", () => {
    expect(depthAnchoringWeightMax).toBe(1);
    expect(depthAnchoringEveryMax).toBe(16);
    expect(depthAnchoringModelOptions).toEqual(["small", "base", "large"]);
  });

  it("flags out-of-range knobs on their own fields, only while enabled", () => {
    // Off: stray knob values are not judged.
    expect(issuesOn({ depthAnchoringEvery: "0" }, "depthAnchoringEvery")).toEqual([]);
    const on = { depthAnchoringWeight: "0.1" };
    expect(issuesOn(on, "depthAnchoringWeight")).toEqual([]);
    for (const bad of ["-0.1", String(depthAnchoringWeightMax + 0.01), "abc"]) {
      expect(issuesOn({ depthAnchoringWeight: bad }, "depthAnchoringWeight")).toHaveLength(1);
    }
    expect(issuesOn({ ...on, depthAnchoringModel: "giant" }, "depthAnchoringModel")).toHaveLength(1);
    expect(issuesOn({ ...on, depthAnchoringMinT: "-0.1" }, "depthAnchoringMinT")).toHaveLength(1);
    expect(issuesOn({ ...on, depthAnchoringMaxT: "1.5" }, "depthAnchoringMaxT")).toHaveLength(1);
    expect(issuesOn({ ...on, depthAnchoringMinT: "0.7", depthAnchoringMaxT: "0.3" }, "depthAnchoringMaxT")).toHaveLength(1);
    for (const bad of ["0", "2.5", String(depthAnchoringEveryMax + 1)]) {
      expect(issuesOn({ ...on, depthAnchoringEvery: bad }, "depthAnchoringEvery")).toHaveLength(1);
    }
    for (const ok of ["1", "2", String(depthAnchoringEveryMax)]) {
      expect(issuesOn({ ...on, depthAnchoringEvery: ok }, "depthAnchoringEvery")).toEqual([]);
    }
  });
});

// sc-24827 (epic 2123): annealed gradient noise is off by default, round-trips eta (+ gamma, which
// only travels with an eta) from the draft into the job snapshot, and is bounded by the API's max.
describe("gradient noise (sc-24827)", () => {
  const snap = (draft) =>
    trainingConfigSnapshot({
      activeDataset: dataset,
      configDraft: { ...draft, outputName: "Kelsie LoRA" },
      selectedTarget: target,
    });
  const whole = {
    outputName: "Kelsie LoRA",
    triggerWord: "kelsie",
    rank: 8,
    alpha: 8,
    learningRate: 0.0001,
    steps: 1000,
    resolution: 1024,
    batchSize: 1,
    gradientAccumulation: 1,
    saveEvery: 250,
  };

  it("seeds off and leaves a default snapshot without either key", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    expect(draft.gradientNoiseEta).toBe("");
    expect(draft.gradientNoiseGamma).toBe("");
    expect(snap(draft).config.advanced).not.toHaveProperty("gradientNoiseEta");
    expect(snap(draft).config.advanced).not.toHaveProperty("gradientNoiseGamma");
  });

  it("round-trips eta and gamma into the snapshot as numbers; gamma never travels alone", () => {
    const base = configDraftFromTarget(target, dataset, ["auto"]);
    const on = { ...base, gradientNoiseEta: String(gradientNoiseEtaSuggested), gradientNoiseGamma: "0.7" };
    expect(snap(on).config.advanced.gradientNoiseEta).toBe(0.01);
    expect(snap(on).config.advanced.gradientNoiseGamma).toBe(0.7);
    const gammaOnly = { ...base, gradientNoiseGamma: "0.7" };
    expect(snap(gammaOnly).config.advanced).not.toHaveProperty("gradientNoiseGamma");
    const seeded = configDraftFromTarget(
      { ...target, defaults: { ...target.defaults, advanced: { networkType: "lora", gradientNoiseEta: 0.02, gradientNoiseGamma: 0.6 } } },
      dataset,
      ["auto"],
    );
    expect(seeded.gradientNoiseEta).toBe("0.02");
    expect(seeded.gradientNoiseGamma).toBe("0.6");
  });

  it("uses the API's bounds and the upstream defaults", () => {
    expect(gradientNoiseEtaMax).toBe(0.1);
    expect(gradientNoiseEtaSuggested).toBe(0.01);
    expect(gradientNoiseGammaMax).toBe(1);
    expect(gradientNoiseGammaDefault).toBe(0.55);
  });

  it("flags out-of-range eta/gamma and eta with a full fine-tune on their own fields", () => {
    const ctx = { activeDataset: dataset, selectedTarget: target };
    const fieldIssues = (draft, field) => configValidation(draft, ctx).filter((entry) => entry.field === field);
    for (const ok of ["", "0", "0.01", String(gradientNoiseEtaMax)]) {
      expect(fieldIssues({ ...whole, gradientNoiseEta: ok }, "gradientNoiseEta")).toEqual([]);
    }
    for (const bad of ["-0.01", String(gradientNoiseEtaMax + 0.001), "abc"]) {
      expect(fieldIssues({ ...whole, gradientNoiseEta: bad }, "gradientNoiseEta").map((e) => e.kind)).toEqual(["error"]);
    }
    expect(fieldIssues({ ...whole, networkType: "full", gradientNoiseEta: "0.01" }, "gradientNoiseEta")).toHaveLength(1);
    for (const ok of ["", "0", "0.55", String(gradientNoiseGammaMax)]) {
      expect(fieldIssues({ ...whole, gradientNoiseGamma: ok }, "gradientNoiseGamma")).toEqual([]);
    }
    for (const bad of ["-0.1", String(gradientNoiseGammaMax + 0.001), "x"]) {
      expect(fieldIssues({ ...whole, gradientNoiseGamma: bad }, "gradientNoiseGamma").map((e) => e.kind)).toEqual(["error"]);
    }
  });

  it("reads only explicit true flags as support and blocks a carried-over eta without it", () => {
    expect(targetSupportsGradientNoise({ limits: { supportsGradientNoise: true } })).toBe(true);
    expect(targetSupportsGradientNoise({ limits: { supportsGradientNoise: "true" } })).toBe(false);
    expect(targetSupportsGradientNoise({ limits: { supportsWeightNoise: true } })).toBe(false);
    expect(targetSupportsGradientNoise(null)).toBe(false);
    const message = "This target does not support gradient noise — clear it or pick a supporting target";
    const unsupported = { ...target, limits: { ...target.limits, supportsGradientNoise: undefined } };
    const issues = configValidation({ ...whole, gradientNoiseEta: "0.01" }, { activeDataset: dataset, selectedTarget: unsupported });
    expect(issues.map((entry) => entry.message)).toContain(message);
    const supported = { ...target, limits: { ...target.limits, supportsGradientNoise: true } };
    expect(configValidation({ ...whole, gradientNoiseEta: "0.01" }, { activeDataset: dataset, selectedTarget: supported })).toEqual([]);
    expect(configValidation({ ...whole, gradientNoiseEta: "0" }, { activeDataset: dataset, selectedTarget: unsupported })).toEqual([]);
  });
});

// sc-2127 (epic 2123): multi-resolution buckets are off by default, round-trip from the form draft
// into the job's training snapshot as typed rows, and are held to the same limits as the API.
describe("resolution buckets (sc-2127)", () => {
  const bucketTarget = {
    ...target,
    limits: { ...target.limits, resolutions: [512, 768, 1024], supportsResolutionBuckets: true },
  };
  const snap = (draft) =>
    trainingConfigSnapshot({
      activeDataset: dataset,
      configDraft: { ...draft, outputName: "Kelsie LoRA" },
      selectedTarget: bucketTarget,
    });
  const whole = {
    outputName: "Kelsie LoRA",
    triggerWord: "kelsie",
    rank: 8,
    alpha: 8,
    learningRate: 0.0001,
    steps: 1000,
    resolution: 1024,
    batchSize: 1,
    gradientAccumulation: 1,
    saveEvery: 250,
  };
  const rows = (...pairs) => pairs.map(([resolution, repeats]) => ({ resolution: String(resolution), repeats: String(repeats) }));
  const fieldIssues = (resolutionBuckets) =>
    configValidation({ ...whole, resolutionBuckets }, { activeDataset: dataset, selectedTarget: bucketTarget }).filter(
      (entry) => entry.field === "resolutionBuckets",
    );

  it("seeds off and leaves a default snapshot without the key", () => {
    const draft = configDraftFromTarget(bucketTarget, dataset, ["auto"]);
    expect(draft.resolutionBuckets).toBeNull();
    expect(snap(draft).config.advanced).not.toHaveProperty("resolutionBuckets");
  });

  it("round-trips a 16:4:1 bucket list into the training snapshot as numbers", () => {
    const draft = { ...configDraftFromTarget(bucketTarget, dataset, ["auto"]), resolutionBuckets: rows([512, 16], [768, 4], [1024, 1]) };
    expect(snap(draft).config.advanced.resolutionBuckets).toEqual([
      { resolution: 512, repeats: 16 },
      { resolution: 768, repeats: 4 },
      { resolution: 1024, repeats: 1 },
    ]);
    // ...and a preset/target that carries buckets seeds the draft back.
    const seeded = configDraftFromTarget(
      { ...bucketTarget, defaults: { ...bucketTarget.defaults, advanced: { resolutionBuckets: [{ resolution: 768, repeats: 2 }] } } },
      dataset,
      ["auto"],
    );
    expect(seeded.resolutionBuckets).toEqual(rows([768, 2]));
  });

  it("seeds the toggle with the target's resolutions up to the current one", () => {
    expect(seedResolutionBuckets(bucketTarget, "768")).toEqual(rows([512, 1], [768, 1]));
    expect(seedResolutionBuckets({ limits: {} }, "640")).toEqual(rows([640, 1]));
  });

  it("reads only an explicit true flag as support, and blocks a list on an unsupported target", () => {
    expect(targetSupportsResolutionBuckets(bucketTarget)).toBe(true);
    expect(targetSupportsResolutionBuckets({ limits: { supportsResolutionBuckets: "true" } })).toBe(false);
    expect(targetSupportsResolutionBuckets({ limits: {} })).toBe(false);
    expect(targetSupportsResolutionBuckets(null)).toBe(false);
    // LTX-2.5 withholds the flag: a carried-over list is an error there, off is fine.
    const ltx25 = { ...bucketTarget, id: "ltx_2_5_video_lora", limits: { resolutions: [512, 768, 1024] } };
    const issuesOn = (resolutionBuckets) =>
      configValidation({ ...whole, resolutionBuckets }, { activeDataset: dataset, selectedTarget: ltx25 }).filter(
        (entry) => entry.field === "resolutionBuckets",
      );
    expect(issuesOn(rows([512, 2])).map((entry) => entry.message)).toEqual([
      "This target does not support multi-resolution buckets — turn them off or pick a supporting target",
    ]);
    expect(issuesOn(null)).toEqual([]);
  });

  it("uses the API's limits", () => {
    expect(resolutionBucketsMax).toBe(8);
    expect(resolutionBucketRepeatsMax).toBe(100);
    expect(resolutionBucketStride).toBe(32);
  });

  it("accepts a well-formed list and flags every malformed one on the resolutionBuckets field", () => {
    expect(fieldIssues(null)).toEqual([]);
    expect(fieldIssues(rows([512, 16], [768, 4], [1024, resolutionBucketRepeatsMax]))).toEqual([]);
    const tooMany = Array.from({ length: resolutionBucketsMax + 1 }, (_, i) => [512, i + 1]);
    for (const bad of [
      [],
      rows([512, 0]),
      rows([512, -1]),
      rows([512, 1.5]),
      rows([512, resolutionBucketRepeatsMax + 1]),
      rows([512, ""]),
      rows([500, 1]),
      rows([1536, 1]),
      rows([512, 1], [512, 2]),
      rows(...tooMany),
    ]) {
      const issues = fieldIssues(bad);
      expect(issues.length, JSON.stringify(bad)).toBeGreaterThan(0);
      expect(issues.every((entry) => entry.kind === "error")).toBe(true);
    }
  });
});

describe("weight noise target support (sc-24826)", () => {
  const whole = {
    outputName: "Kelsie LoRA",
    triggerWord: "kelsie",
    rank: 8,
    alpha: 8,
    learningRate: 0.0001,
    steps: 1000,
    resolution: 1024,
    batchSize: 1,
    gradientAccumulation: 1,
    saveEvery: 250,
  };

  it("reads only an explicit true flag as support", () => {
    expect(targetSupportsWeightNoise({ limits: { supportsWeightNoise: true } })).toBe(true);
    expect(targetSupportsWeightNoise({ limits: { supportsWeightNoise: "true" } })).toBe(false);
    expect(targetSupportsWeightNoise({ limits: {} })).toBe(false);
    expect(targetSupportsWeightNoise(null)).toBe(false);
  });

  it("blocks a carried-over positive sigma on a target without support", () => {
    const issues = configValidation({ ...whole, weightNoiseSigma: "0.0125" }, { activeDataset: dataset, selectedTarget: target });
    expect(issues.map((entry) => entry.message)).toContain(
      "This target does not support weight noise — clear it or pick a supporting target",
    );
    const supported = { ...target, limits: { ...target.limits, supportsWeightNoise: true } };
    expect(configValidation({ ...whole, weightNoiseSigma: "0.0125" }, { activeDataset: dataset, selectedTarget: supported })).toEqual([]);
    expect(configValidation({ ...whole, weightNoiseSigma: "0" }, { activeDataset: dataset, selectedTarget: target })).toEqual([]);
  });
});

// sc-2125: depth anchoring follows the S1 target-support mechanism (`limits.supportsDepthAnchoring`).
describe("depth anchoring target support (sc-2125)", () => {
  const whole = {
    outputName: "Kelsie LoRA",
    triggerWord: "kelsie",
    rank: 8,
    alpha: 8,
    learningRate: 0.0001,
    steps: 1000,
    resolution: 1024,
    batchSize: 1,
    gradientAccumulation: 1,
    saveEvery: 250,
    depthAnchoringModel: "small",
  };

  it("reads only an explicit true flag as support", () => {
    expect(targetSupportsDepthAnchoring({ limits: { supportsDepthAnchoring: true } })).toBe(true);
    expect(targetSupportsDepthAnchoring({ limits: { supportsDepthAnchoring: "true" } })).toBe(false);
    expect(targetSupportsDepthAnchoring({ limits: {} })).toBe(false);
    expect(targetSupportsDepthAnchoring(null)).toBe(false);
  });

  it("blocks a carried-over depth weight on a target without support", () => {
    const issues = configValidation({ ...whole, depthAnchoringWeight: "0.1" }, { activeDataset: dataset, selectedTarget: target });
    expect(issues.map((entry) => entry.message)).toContain(
      "This target does not support depth anchoring — clear it or pick a supporting target",
    );
    const supported = { ...target, limits: { ...target.limits, supportsDepthAnchoring: true } };
    expect(configValidation({ ...whole, depthAnchoringWeight: "0.1" }, { activeDataset: dataset, selectedTarget: supported })).toEqual([]);
    expect(configValidation({ ...whole, depthAnchoringWeight: "0" }, { activeDataset: dataset, selectedTarget: target })).toEqual([]);
  });

  // sc-24830 review: an advertising target still refuses the combinations the engine refuses — a
  // full base fine-tune and an LTX-2.5 workflow with no generated video — as a weight issue (the
  // toggle stays visible so the user can untick it). Mutation: return null from
  // depthAnchoringCombinationRefusal ⇒ red.
  it("refuses a full fine-tune and the no-video LTX-2.5 workflows", () => {
    const supported = { ...target, limits: { ...target.limits, supportsDepthAnchoring: true } };
    const ltx = { ...supported, baseModel: "ltx_2_5" };
    const messages = (draft, selectedTarget) =>
      configValidation({ ...whole, depthAnchoringWeight: "0.1", ...draft }, { activeDataset: dataset, selectedTarget })
        .filter((entry) => entry.field === "depthAnchoringWeight")
        .map((entry) => entry.message);
    expect(messages({ networkType: "full" }, supported)).toEqual([
      "Depth anchoring trains a LoRA/LoKr adapter only, not a full fine-tune",
    ]);
    expect(messages({ networkType: "lora" }, supported)).toEqual([]);
    expect(depthAnchoringNoVideoLtxWorkflows).toHaveLength(6);
    for (const workflow of depthAnchoringNoVideoLtxWorkflows) {
      expect(messages({ ltxWorkflow: workflow }, ltx)).toHaveLength(1);
      // The same name on a non-LTX-2.5 target is no refusal.
      expect(depthAnchoringCombinationRefusal(supported, { ltxWorkflow: workflow })).toBeNull();
    }
    expect(depthAnchoringCombinationRefusal(ltx, { ltxWorkflow: "t2v_lora" })).toBeNull();
  });
});

// sc-24828 (epic 2123): subject-masked loss is off by default, round-trips into the snapshot only
// when on, is bounded by the API's bounds, and incomplete mask coverage blocks the run.
describe("subject-masked loss (sc-24828)", () => {
  const snap = (draft) =>
    trainingConfigSnapshot({
      activeDataset: dataset,
      configDraft: { ...draft, outputName: "Kelsie LoRA" },
      selectedTarget: target,
    });
  const whole = {
    outputName: "Kelsie LoRA",
    triggerWord: "kelsie",
    rank: 8,
    alpha: 8,
    learningRate: 0.0001,
    steps: 1000,
    resolution: 1024,
    batchSize: 1,
    gradientAccumulation: 1,
    saveEvery: 250,
    subjectMaskLoss: true,
    subjectMaskBackgroundWeight: "0.1",
    subjectMaskSubjectWeight: "1",
  };
  const report = (masks) => ({ items: masks.map(([hasMask, empty]) => ({ hasMask, empty })) });
  const maskTarget = { ...target, limits: { ...target.limits, supportsSubjectMaskLoss: true } };
  const issuesFor = (draft, subjectMaskReport = null) =>
    configValidation(draft, { activeDataset: dataset, selectedTarget: maskTarget, subjectMaskReport }).filter((entry) =>
      String(entry.field ?? "").startsWith("subjectMask"),
    );

  it("is offered only where the target advertises it; a carried-over true elsewhere blocks the run", () => {
    expect(targetSupportsSubjectMaskLoss(maskTarget)).toBe(true);
    expect(targetSupportsSubjectMaskLoss(target)).toBe(false);
    expect(targetSupportsSubjectMaskLoss({ limits: { supportsSubjectMaskLoss: "yes" } })).toBe(false);
    const unsupported = configValidation(whole, { activeDataset: dataset, selectedTarget: target }).filter((entry) =>
      String(entry.message).includes("subject-masked loss"),
    );
    expect(unsupported.map((entry) => [entry.field, entry.kind])).toEqual([[null, "error"]]);
    expect(
      configValidation({ ...whole, subjectMaskLoss: false }, { activeDataset: dataset, selectedTarget: target }).filter(
        (entry) => String(entry.message).includes("subject-masked loss"),
      ),
    ).toEqual([]);
  });

  it("seeds off with default weights and leaves a default snapshot without the keys", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    expect(draft.subjectMaskLoss).toBe(false);
    expect(draft.subjectMaskBackgroundWeight).toBe(String(subjectMaskBackgroundWeightDefault));
    expect(draft.subjectMaskSubjectWeight).toBe(String(subjectMaskSubjectWeightDefault));
    const advanced = snap(draft).config.advanced;
    for (const key of ["subjectMaskLoss", "subjectMaskBackgroundWeight", "subjectMaskSubjectWeight"]) {
      expect(advanced).not.toHaveProperty(key);
    }
  });

  it("round-trips an enabled run's weights into the snapshot as numbers", () => {
    const draft = {
      ...configDraftFromTarget(target, dataset, ["auto"]),
      subjectMaskLoss: true,
      subjectMaskBackgroundWeight: "0",
      subjectMaskSubjectWeight: "0.8",
    };
    const advanced = snap(draft).config.advanced;
    expect(advanced.subjectMaskLoss).toBe(true);
    expect(advanced.subjectMaskBackgroundWeight).toBe(0);
    expect(advanced.subjectMaskSubjectWeight).toBe(0.8);
  });

  it("uses the API's bounds and defaults", () => {
    expect(subjectMaskWeightMax).toBe(1);
    expect(subjectMaskBackgroundWeightDefault).toBe(0.1);
    expect(subjectMaskSubjectWeightDefault).toBe(1);
  });

  it("flags out-of-range weights on their fields", () => {
    expect(issuesFor(whole)).toEqual([]);
    expect(issuesFor({ ...whole, subjectMaskBackgroundWeight: "0" })).toEqual([]);
    for (const bad of ["-0.1", "1.01", "abc", ""]) {
      expect(issuesFor({ ...whole, subjectMaskBackgroundWeight: bad }).map((entry) => entry.field)).toEqual([
        "subjectMaskBackgroundWeight",
      ]);
    }
    for (const bad of ["0", "1.5", "x"]) {
      expect(issuesFor({ ...whole, subjectMaskSubjectWeight: bad }).map((entry) => entry.field)).toEqual([
        "subjectMaskSubjectWeight",
      ]);
    }
    // Off ⇒ the weights are not this run's concern.
    expect(issuesFor({ ...whole, subjectMaskLoss: false, subjectMaskSubjectWeight: "0" })).toEqual([]);
  });

  it("blocks on incomplete mask coverage (an empty mask counts as missing), not on unknown coverage", () => {
    expect(issuesFor(whole, report([[true, false], [true, false]]))).toEqual([]);
    expect(issuesFor(whole, null)).toEqual([]);
    const partial = issuesFor(whole, report([[true, false], [true, true], [false, false]]));
    expect(partial.map((entry) => [entry.field, entry.kind])).toEqual([["subjectMaskLoss", "error"]]);
    expect(partial[0].message).toContain("missing for 2 of 3 images");
    expect(subjectMaskCoverage(report([[true, false], [true, true], [false, false]]))).toEqual({
      total: 3,
      usable: 1,
      empty: 1,
      missing: 2,
      complete: false,
    });
  });
});

// sc-24832 (epic 2123): the three body losses are off by default, each round-trips from the form
// draft into the job snapshot only while on, every knob is bounded by the API's limits (E6), and a
// loss the target does not advertise blocks Start.
describe("body losses (sc-24832)", () => {
  const allLimits = Object.fromEntries(bodyLosses.map((loss) => [loss.limit, true]));
  const bodyTarget = { ...target, limits: { ...target.limits, ...allLimits } };
  const snap = (draft, selectedTarget = bodyTarget) =>
    trainingConfigSnapshot({
      activeDataset: dataset,
      configDraft: { ...draft, outputName: "Kelsie LoRA" },
      selectedTarget,
    });
  const whole = {
    outputName: "Kelsie LoRA",
    triggerWord: "kelsie",
    rank: 8,
    alpha: 8,
    learningRate: 0.0001,
    steps: 1000,
    resolution: 1024,
    batchSize: 1,
    gradientAccumulation: 1,
    saveEvery: 250,
  };
  const issuesOn = (draft, field, selectedTarget = bodyTarget) =>
    configValidation({ ...whole, ...draft }, { activeDataset: dataset, selectedTarget }).filter(
      (entry) => entry.field === field,
    );

  it("seeds every loss off and leaves a default snapshot without any body key", () => {
    const draft = configDraftFromTarget(bodyTarget, dataset, ["auto"]);
    for (const { prefix } of bodyLosses) expect(draft[`${prefix}Weight`]).toBe("");
    const advanced = snap(draft).config.advanced;
    for (const key of Object.keys(advanced)) {
      expect(/^(bodyProportion|bodyShape|normal)/.test(key)).toBe(false);
    }
  });

  // Mutation: drop the per-loss `bodyLossEnabled` guard in bodyLossSnapshot ⇒ the off losses'
  // knobs leak into the snapshot ⇒ red.
  it("round-trips each enabled loss (and only it) as typed values", () => {
    const draft = {
      ...configDraftFromTarget(bodyTarget, dataset, ["auto"]),
      bodyProportionWeight: String(bodyLossWeightSuggested),
      bodyProportionIncludeHead: true,
      bodyProportionEvery: "1",
      bodyShapeMinT: "0.3",
      normalWeight: "0.2",
      normalMinT: "0.1",
      normalMaxT: "0.9",
      normalRestrictToSubject: true,
    };
    const advanced = snap(draft).config.advanced;
    expect(advanced.bodyProportionWeight).toBe(0.1);
    expect(advanced.bodyProportionIncludeHead).toBe(true);
    expect(advanced.bodyProportionEvery).toBe(1);
    expect(advanced.normalWeight).toBe(0.2);
    expect(advanced.normalMinT).toBe(0.1);
    expect(advanced.normalMaxT).toBe(0.9);
    expect(advanced.normalRestrictToSubject).toBe(true);
    // The shape loss is off: its stray knob never reaches the job.
    expect(advanced.bodyShapeMinT).toBeUndefined();
    expect(advanced.bodyShapeWeight).toBeUndefined();
    const seeded = configDraftFromTarget(
      { ...target, defaults: { ...target.defaults, advanced: { networkType: "lora", bodyShapeWeight: 0.05, bodyShapeMinCos: 0.4 } } },
      dataset,
      ["auto"],
    );
    expect(seeded.bodyShapeWeight).toBe("0.05");
    expect(seeded.bodyShapeMinCos).toBe("0.4");
  });

  it("flags out-of-range knobs on their own fields, only while the loss is on", () => {
    expect(issuesOn({ bodyShapeEvery: "0" }, "bodyShapeEvery")).toEqual([]);
    for (const bad of ["-0.1", String(bodyLossWeightMax + 0.01), "abc"]) {
      expect(issuesOn({ normalWeight: bad }, "normalWeight")).toHaveLength(1);
    }
    const on = { bodyShapeWeight: "0.1" };
    expect(issuesOn(on, "bodyShapeWeight")).toEqual([]);
    expect(issuesOn({ ...on, bodyShapeMinT: "-0.1" }, "bodyShapeMinT")).toHaveLength(1);
    expect(issuesOn({ ...on, bodyShapeMaxT: "1.5" }, "bodyShapeMaxT")).toHaveLength(1);
    // The shape window defaults to [0.4, 0.8]: a lone max below the default min is inverted.
    expect(issuesOn({ ...on, bodyShapeMaxT: "0.3" }, "bodyShapeMaxT")).toHaveLength(1);
    expect(issuesOn({ ...on, bodyShapeMinCos: "1.5" }, "bodyShapeMinCos")).toHaveLength(1);
    for (const bad of ["0", "2.5", String(bodyLossEveryMax + 1)]) {
      expect(issuesOn({ ...on, bodyShapeEvery: bad }, "bodyShapeEvery")).toHaveLength(1);
    }
    expect(issuesOn({ ...on, bodyShapeEvery: String(bodyLossEveryMax) }, "bodyShapeEvery")).toEqual([]);
  });

  // Mutation: drop the `targetSupportsBodyLoss` check in bodyLossIssues ⇒ the carried-over weight
  // passes ⇒ red.
  it("blocks a loss the target does not advertise and reads the flag strictly", () => {
    const proportionOnly = { ...target, limits: { ...target.limits, supportsBodyProportionLoss: true } };
    const blocking = configValidation({ ...whole, normalWeight: "0.1" }, { activeDataset: dataset, selectedTarget: proportionOnly });
    expect(blocking.some((entry) => entry.field === null && /normals loss/.test(entry.message))).toBe(true);
    const fine = configValidation({ ...whole, bodyProportionWeight: "0.1" }, { activeDataset: dataset, selectedTarget: proportionOnly });
    expect(fine.filter((entry) => /does not support the/.test(entry.message))).toEqual([]);
    expect(targetSupportsBodyLoss({ limits: { supportsBodyShapeLoss: "true" } }, bodyLosses[1])).toBe(false);
    expect(targetSupportsBodyLoss({ limits: { supportsBodyShapeLoss: true } }, bodyLosses[1])).toBe(true);
  });

  // Mirrors the API: LTX-2.5 cannot restrict the normal loss to the subject — a field-less issue,
  // since the toggle is hidden there. Mutation: drop the LTX-2.5 check in bodyLossIssues ⇒ red.
  it("flags subject-restricted normals on LTX-2.5 only", () => {
    const ltx25 = { ...bodyTarget, baseModel: "ltx_2_5" };
    const draft = { normalWeight: "0.1", normalRestrictToSubject: true, ltxWorkflow: "t2v_lora" };
    const ltx = configValidation({ ...whole, ...draft }, { activeDataset: dataset, selectedTarget: ltx25 });
    expect(ltx.filter((entry) => entry.field === null && /LTX-2.5 cannot restrict/.test(entry.message))).toHaveLength(1);
    expect(issuesOn(draft, "normalRestrictToSubject", ltx25)).toEqual([]);
    expect(issuesOn(draft, "normalRestrictToSubject")).toEqual([]);
  });

  // Like subject-masked loss, restricted normals need a non-empty subject mask on every image:
  // incomplete coverage is an error on the toggle; full or unknown coverage is not. Mutation: drop
  // the coverage check in bodyLossIssues ⇒ red.
  it("blocks subject-restricted normals on incomplete subject-mask coverage", () => {
    const draft = { ...whole, normalWeight: "0.1", normalRestrictToSubject: true };
    const on = (subjectMaskReport) =>
      configValidation(draft, { activeDataset: dataset, selectedTarget: bodyTarget, subjectMaskReport }).filter(
        (entry) => entry.field === "normalRestrictToSubject",
      );
    const partial = { items: [{ hasMask: true, empty: false }, { hasMask: true, empty: true }, { hasMask: false }] };
    expect(on(partial).map((entry) => entry.message)).toEqual([
      "Subject masks are missing for 2 of 3 images — generate subject masks first",
    ]);
    expect(on({ items: [{ hasMask: true, empty: false }] })).toEqual([]);
    expect(on(null)).toEqual([]);
    // Unrestricted normals never read the masks.
    expect(
      configValidation(
        { ...draft, normalRestrictToSubject: false },
        { activeDataset: dataset, selectedTarget: bodyTarget, subjectMaskReport: partial },
      ).filter((entry) => /Subject masks are missing/.test(entry.message)),
    ).toEqual([]);
  });

  // Mirrors the API's combination refusals, attached to the loss's weight key like the API's (the
  // toggle stays visible, so the user can untick it). Mutation: drop the LTX-2.5 workflow check, or
  // name no field ⇒ red.
  it("refuses a full fine-tune and a video-less LTX-2.5 workflow on the weight key", () => {
    expect(bodyLossCombinationRefusal(bodyTarget, { networkType: "full" })).toMatch(/full fine-tune/);
    const ltx25 = { ...bodyTarget, baseModel: "ltx_2_5" };
    expect(bodyLossCombinationRefusal(ltx25, { ltxWorkflow: "v2a_lora" })).toMatch(/v2a_lora/);
    expect(bodyLossCombinationRefusal(ltx25, { ltxWorkflow: "t2v_lora" })).toBe(null);
    const blocked = configValidation(
      { ...whole, networkType: "full", bodyShapeWeight: "0.1" },
      { activeDataset: dataset, selectedTarget: bodyTarget },
    );
    expect(blocked.filter((entry) => /full fine-tune/.test(entry.message)).map((entry) => entry.field)).toEqual([
      "bodyShapeWeight",
    ]);
  });

  it("uses the API's bounds", () => {
    expect(bodyLossWeightMax).toBe(1);
    expect(bodyLossEveryMax).toBe(16);
    expect(bodyLosses.map((loss) => loss.limit)).toEqual([
      "supportsBodyProportionLoss",
      "supportsBodyShapeLoss",
      "supportsNormalLoss",
    ]);
  });
});

// sc-24831 (epic 2123): the face losses are off by default, round-trip from the draft into the job
// snapshot only while on, are bounded by the API's limits (E6), and follow the target-support
// mechanism (`limits.supportsIdentityLoss` / `limits.supportsFaceLandmarkLoss`).
describe("face losses (sc-24831)", () => {
  const faceTarget = {
    ...target,
    limits: { ...target.limits, supportsIdentityLoss: true, supportsFaceLandmarkLoss: true },
  };
  const whole = {
    outputName: "Kelsie LoRA",
    triggerWord: "kelsie",
    rank: 8,
    alpha: 8,
    learningRate: 0.0001,
    steps: 1000,
    resolution: 1024,
    batchSize: 1,
    gradientAccumulation: 1,
    saveEvery: 250,
    identityLossReference: "dataset_average",
  };
  const snap = (draft, selectedTarget = faceTarget) =>
    trainingConfigSnapshot({
      activeDataset: dataset,
      configDraft: { ...draft, outputName: "Kelsie LoRA" },
      selectedTarget,
    });
  const issuesOn = (draft, field, selectedTarget = faceTarget) =>
    configValidation({ ...whole, ...draft }, { activeDataset: dataset, selectedTarget }).filter(
      (entry) => entry.field === field,
    );

  it("seeds off and leaves a default snapshot without any face-loss key", () => {
    const draft = configDraftFromTarget(faceTarget, dataset, ["auto"]);
    expect(draft.identityLossWeight).toBe("");
    expect(draft.faceLandmarkLossWeight).toBe("");
    expect(draft.identityLossReference).toBe("dataset_average");
    for (const key of Object.keys(snap(draft).config.advanced)) {
      expect(key.startsWith("identityLoss") || key.startsWith("faceLandmarkLoss")).toBe(false);
    }
  });

  it("round-trips enabled configurations into the snapshot as typed values", () => {
    const draft = {
      ...configDraftFromTarget(faceTarget, dataset, ["auto"]),
      identityLossWeight: String(faceLossWeightSuggested),
      identityLossMinT: "0.1",
      identityLossMaxT: "0.9",
      identityLossEvery: "1",
      identityLossMinCos: "0.3",
      identityLossReference: "per_image",
      faceLandmarkLossWeight: "0.05",
      faceLandmarkLossEvery: "4",
    };
    const advanced = snap(draft).config.advanced;
    expect(advanced.identityLossWeight).toBe(0.1);
    expect(advanced.identityLossMinT).toBe(0.1);
    expect(advanced.identityLossMaxT).toBe(0.9);
    expect(advanced.identityLossEvery).toBe(1);
    expect(advanced.identityLossMinCos).toBe(0.3);
    expect(advanced.identityLossReference).toBe("per_image");
    expect(advanced.faceLandmarkLossWeight).toBe(0.05);
    expect(advanced.faceLandmarkLossEvery).toBe(4);
    const seeded = configDraftFromTarget(
      {
        ...target,
        defaults: {
          ...target.defaults,
          advanced: { networkType: "lora", identityLossWeight: 0.2, identityLossReference: "per_image" },
        },
      },
      dataset,
      ["auto"],
    );
    expect(seeded.identityLossWeight).toBe("0.2");
    expect(seeded.identityLossReference).toBe("per_image");
  });

  it("uses the API's bounds", () => {
    expect(faceLossWeightMax).toBe(1);
    expect(faceLossEveryMax).toBe(16);
    expect(identityLossReferenceOptions).toEqual(["dataset_average", "per_image"]);
  });

  // Mutation: drop the identityLossMinCos range check ⇒ the 1.5 case passes ⇒ red.
  it("flags out-of-range knobs on their own fields, only while enabled", () => {
    expect(issuesOn({ identityLossEvery: "0" }, "identityLossEvery")).toEqual([]);
    const on = { identityLossWeight: "0.1" };
    expect(issuesOn(on, "identityLossWeight")).toEqual([]);
    for (const bad of ["-0.1", String(faceLossWeightMax + 0.01), "abc"]) {
      expect(issuesOn({ identityLossWeight: bad }, "identityLossWeight")).toHaveLength(1);
    }
    expect(issuesOn({ ...on, identityLossMinCos: "1.5" }, "identityLossMinCos")).toHaveLength(1);
    expect(issuesOn({ ...on, identityLossMinCos: "-0.5" }, "identityLossMinCos")).toEqual([]);
    expect(issuesOn({ ...on, identityLossReference: "random" }, "identityLossReference")).toHaveLength(1);
    expect(issuesOn({ ...on, identityLossMinT: "0.7", identityLossMaxT: "0.3" }, "identityLossMaxT")).toHaveLength(1);
    for (const bad of ["0", "2.5", String(faceLossEveryMax + 1)]) {
      expect(issuesOn({ ...on, identityLossEvery: bad }, "identityLossEvery")).toHaveLength(1);
    }
    const lm = { faceLandmarkLossWeight: "0.1" };
    expect(issuesOn(lm, "faceLandmarkLossWeight")).toEqual([]);
    expect(issuesOn({ ...lm, faceLandmarkLossMaxT: "3" }, "faceLandmarkLossMaxT")).toHaveLength(1);
  });

  // Mutation: read a truthy (not strictly true) flag as support ⇒ the "true"-string case passes ⇒ red.
  it("reads only an explicit true flag as support and refuses a carried-over weight elsewhere", () => {
    expect(targetSupportsIdentityLoss({ limits: { supportsIdentityLoss: true } })).toBe(true);
    expect(targetSupportsIdentityLoss({ limits: { supportsIdentityLoss: "true" } })).toBe(false);
    expect(targetSupportsFaceLandmarkLoss({ limits: {} })).toBe(false);
    const unsupported = configValidation(
      { ...whole, identityLossWeight: "0.1", faceLandmarkLossWeight: "0.1" },
      { activeDataset: dataset, selectedTarget: target },
    ).filter((entry) => entry.field === null);
    expect(unsupported.map((entry) => entry.message)).toEqual([
      "This target does not support the identity loss — clear it or pick a supporting target",
      "This target does not support the face landmark loss — clear it or pick a supporting target",
    ]);
  });
});

// sc-24831: the face losses are refused for the same combinations as depth anchoring (sc-24830) —
// a full base fine-tune and an LTX-2.5 workflow with no generated video — on the weight field, and
// their controls are not offered there. Mutation: drop `faceLossCombinationRefusal` from
// `faceLossScheduleIssues` ⇒ no issue ⇒ red.
describe("face loss combination refusals (sc-24831)", () => {
  const ltx = { id: "ltx_2_5_video_lora", baseModel: "ltx_2_5", limits: { supportsIdentityLoss: true, supportsFaceLandmarkLoss: true } };
  const full = { ...target, limits: { ...target.limits, supportsIdentityLoss: true, supportsFaceLandmarkLoss: true } };
  it("refuses a full fine-tune and a no-video LTX-2.5 workflow on the weight key", () => {
    for (const [key, label] of [["identityLossWeight", "Identity loss"], ["faceLandmarkLossWeight", "Face landmark loss"]]) {
      const ft = faceLossIssues({ [key]: "0.1", identityLossReference: "dataset_average", networkType: "full" }, full);
      expect(ft).toEqual([[key, `${label} trains a LoRA/LoKr adapter only, not a full fine-tune`]]);
      const audio = faceLossIssues({ [key]: "0.1", identityLossReference: "dataset_average", ltxWorkflow: "t2a_lora" }, ltx);
      expect(audio).toEqual([[key, `${label} needs a generated video stream; the LTX-2.5 workflow t2a_lora generates none`]]);
      expect(faceLossIssues({ [key]: "0.1", identityLossReference: "dataset_average", ltxWorkflow: "t2v_lora" }, ltx)).toEqual([]);
    }
    expect(faceLossCombinationRefusal({ baseModel: "ltx_2_3" }, { ltxWorkflow: "t2a_lora" }, "Identity loss")).toBeNull();
  });
});

// sc-24833: the latent-space perceptual losses follow the S1 target-support mechanism
// (`limits.supportsVaeAnchorLoss` / `limits.supportsLatentLpipsLoss`), are off by default, reach
// the snapshot only when on, and are bounded by the API's bounds (E6).
describe("latent-space perceptual losses (sc-24833)", () => {
  const whole = {
    outputName: "Kelsie LoRA",
    triggerWord: "kelsie",
    rank: 8,
    alpha: 8,
    learningRate: 0.0001,
    steps: 1000,
    resolution: 1024,
    batchSize: 1,
    gradientAccumulation: 1,
    saveEvery: 250,
    depthAnchoringModel: "small",
  };
  const supported = {
    ...target,
    limits: { ...target.limits, supportsVaeAnchorLoss: true, supportsLatentLpipsLoss: true },
  };
  const validate = (draft, selectedTarget = supported) =>
    configValidation({ ...whole, ...draft }, { activeDataset: dataset, selectedTarget });
  const snap = (draft) =>
    trainingConfigSnapshot({ activeDataset: dataset, configDraft: { ...whole, ...draft }, selectedTarget: supported });

  it("reads only an explicit true flag as support, per loss", () => {
    expect(targetSupportsVaeAnchorLoss({ limits: { supportsVaeAnchorLoss: true } })).toBe(true);
    expect(targetSupportsVaeAnchorLoss({ limits: { supportsVaeAnchorLoss: "true" } })).toBe(false);
    expect(targetSupportsVaeAnchorLoss({ limits: { supportsLatentLpipsLoss: true } })).toBe(false);
    expect(targetSupportsLatentLpipsLoss({ limits: { supportsLatentLpipsLoss: true } })).toBe(true);
    expect(targetSupportsLatentLpipsLoss({ limits: {} })).toBe(false);
    expect(targetSupportsLatentLpipsLoss(null)).toBe(false);
  });

  it("is off by default: the draft carries empty weights and the snapshot no keys", () => {
    const draft = configDraftFromTarget(supported, dataset, [], "kelsie");
    for (const loss of latentPerceptualLosses) {
      expect(draft[`${loss.prefix}Weight`]).toBe("");
    }
    const advanced = snap({}).config.advanced;
    expect(advanced).toBeTruthy();
    expect(Object.keys(advanced).filter((k) => /^(vaeAnchor|latentLpips)/.test(k))).toEqual([]);
  });

  // Mutation: drop latentPerceptualSnapshot from trainingConfigSnapshot ⇒ red.
  it("carries the weight and every set knob while on", () => {
    const config = snap({ vaeAnchorWeight: "0.5", vaeAnchorEvery: "2", latentLpipsWeight: "1", latentLpipsMaxT: "0.4" });
    const advanced = config.config.advanced;
    expect(advanced.vaeAnchorWeight).toBe(0.5);
    expect(advanced.vaeAnchorEvery).toBe(2);
    expect(advanced.vaeAnchorMinT).toBeUndefined();
    expect(advanced.latentLpipsWeight).toBe(1);
    expect(advanced.latentLpipsMaxT).toBe(0.4);
  });

  it("blocks a carried-over weight on a target without support", () => {
    for (const loss of latentPerceptualLosses) {
      const issues = validate({ [`${loss.prefix}Weight`]: "1" }, target);
      expect(issues.map((entry) => entry.message)).toContain(
        `This target does not support the ${loss.label} loss — clear it or pick a supporting target`,
      );
      expect(validate({ [`${loss.prefix}Weight`]: "1" })).toEqual([]);
      expect(validate({ [`${loss.prefix}Weight`]: "0" }, target)).toEqual([]);
    }
  });

  // Mirrors the API's latent_loss_combination_refusal. Mutation: drop the refusal check from
  // latentPerceptualIssues ⇒ red.
  it("refuses a full fine-tune and the VAE anchor on a no-video LTX-2.5 workflow", () => {
    for (const loss of latentPerceptualLosses) {
      const issues = validate({ [`${loss.prefix}Weight`]: "1", networkType: "full" });
      expect(issues.map((entry) => entry.field)).toContain(`${loss.prefix}Weight`);
      expect(latentLossCombinationRefusal(supported, { networkType: "lora" }, loss)).toBeNull();
    }
    const ltx = { ...supported, baseModel: "ltx_2_5" };
    const [vaeAnchor, lpips] = latentPerceptualLosses;
    expect(latentLossCombinationRefusal(ltx, { ltxWorkflow: "t2a_lora" }, vaeAnchor)).toContain("t2a_lora");
    expect(latentLossCombinationRefusal(ltx, { ltxWorkflow: "t2v_lora" }, vaeAnchor)).toBeNull();
    expect(latentLossCombinationRefusal(ltx, { ltxWorkflow: "t2a_lora" }, lpips)).toBeNull();
  });

  // Mutation: drop latentPerceptualIssues from configValidation ⇒ red.
  it("enforces the API bounds as field issues", () => {
    for (const loss of latentPerceptualLosses) {
      const fields = (draft) => validate({ [`${loss.prefix}Weight`]: "1", ...draft }).map((entry) => entry.field);
      expect(fields({ [`${loss.prefix}Weight`]: String(loss.weightMax + 1) })).toContain(`${loss.prefix}Weight`);
      expect(fields({ [`${loss.prefix}MinT`]: "1.5" })).toContain(`${loss.prefix}MinT`);
      expect(fields({ [`${loss.prefix}MinT`]: "0.6" })).toContain(`${loss.prefix}MaxT`);
      expect(fields({ [`${loss.prefix}MinT`]: "0.6", [`${loss.prefix}MaxT`]: "0.9" })).toEqual([]);
      expect(fields({ [`${loss.prefix}Every`]: String(loss.everyMax + 1) })).toContain(`${loss.prefix}Every`);
      expect(fields({ [`${loss.prefix}Every`]: "1.5" })).toContain(`${loss.prefix}Every`);
    }
  });
});

// Epic 2123 review: the decoded-x0 help text names the target's own x0 decoder from the API's
// `limits.x0Decoder` (projected from the worker's trainer-keyed mapping) — never TAEF1 for every
// family — and says nothing extra is installed when the trainer decodes through its own VAE.
// Mutation: hard-code "TAEF1" in auxModelsInstallNote, or drop the decoder ⇒ red.
describe("auxModelsInstallNote", () => {
  it("names the target's decoder, its own models, and the base-VAE case", () => {
    const sdxl = { limits: { x0Decoder: { label: "TAESDXL tiny decoder", install: true } } };
    expect(auxModelsInstallNote(sdxl, ["a Depth Anything V2 model"])).toBe(
      "Needs the TAESDXL tiny decoder and a Depth Anything V2 model installed.",
    );
    expect(auxModelsInstallNote(sdxl, ["the InstantID face analysis stack", "MediaPipe FaceMesh v2"])).toBe(
      "Needs the TAESDXL tiny decoder, the InstantID face analysis stack and MediaPipe FaceMesh v2 installed.",
    );
    const mage = { limits: { x0Decoder: { label: "the base model's own VAE", install: false } } };
    expect(auxModelsInstallNote(mage, ["a Depth Anything V2 model"])).toBe(
      "Needs a Depth Anything V2 model installed. The prediction is decoded through the base model's own VAE.",
    );
    expect(auxModelsInstallNote({ limits: {} }, ["ViTPose+ Base"])).toBe("Needs ViTPose+ Base installed.");
  });
});
