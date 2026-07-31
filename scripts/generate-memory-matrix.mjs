#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import {
  evidenceSemantics,
  validateBundle as validateCalibrationBundle,
} from "./memory-calibration-harness.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT_JSON = "docs/generated/memory-matrix.json";
const OUTPUT_MD = "docs/generated/memory-matrix.md";
const EXPECTED_IMAGE_COUNT = 53;
const EXPECTED_MLX_STAGED_COUNT = 39;
// Provider calibration ABI versions are deliberate invalidation switches. A provider-specific
// execution/layout/quantization change that makes measurements unsafe must add or bump its key;
// ecosystem-wide contract changes bump `default`. Exact source revisions remain provenance only.
const CALIBRATION_ABI_VERSIONS = Object.freeze({
  default: 1,
});
const RUNGS = [
  "resident",
  "staged_residency",
  "bounded_decode",
  "bounded_attention",
  "bounded_transformer_residency",
];

export function canonicalSourceText(body) {
  return body.replace(/\r\n?/g, "\n");
}

const GENERATION_CAPABILITIES = new Set([
  "text_to_image",
  "edit_image",
  "image_to_image",
  "image_inpaint",
  "image_detail",
  "character_image",
  "style_variations",
]);

// This is ownership metadata, not conformance data. Drift is checked against the
// source-owned EXPECTED_IMAGE_IDS list and the shipped manifest below.
//
// SC-15812: ownership is keyed BY BACKEND, not by model. An MLX story cannot be closed from CUDA
// hardware and a Candle story cannot be closed from a Mac, so a dual-backend entry has two owners
// and a cell must name the one that actually covers it. An mlx-only entry simply has no `candle`
// key: a missing mapping then fails loudly at generation time instead of silently attributing
// Candle cells to a story scoped to Metal — which is precisely the false green this epic exists to
// prevent, and which shipped in the generated matrix until this change.
export const MODEL_STORIES = {
  mage_flow_edit_base: { mlx: 15450, candle: 15841 },
  mage_flow_edit: { mlx: 15451, candle: 15844 },
  mage_flow_edit_turbo: { mlx: 15452, candle: 15847 },
  mage_flow_base: { mlx: 15453, candle: 15850 },
  mage_flow: { mlx: 15454, candle: 15853 },
  mage_flow_turbo: { mlx: 15455, candle: 15856 },
  z_image_turbo: { mlx: 15456, candle: 15859 },
  z_image: { mlx: 15457, candle: 16170 },
  z_image_edit: { mlx: 15458, candle: 15862 },
  qwen_image: { mlx: 15459, candle: 15865 },
  qwen_image_edit_2511: { mlx: 15460, candle: 15868 },
  qwen_image_edit_2511_lightning: { mlx: 15461, candle: 15871 },
  lens: { mlx: 15462 },
  lens_turbo: { mlx: 15463, candle: 15874 },
  sensenova_u1_8b: { mlx: 15464, candle: 15877 },
  sensenova_u1_8b_infographic_v2: { mlx: 15465, candle: 15880 },
  sensenova_u1_8b_infographic_v3: { mlx: 15466, candle: 15883 },
  sensenova_u1_8b_fast: { mlx: 15467, candle: 15886 },
  sensenova_u1_8b_infographic_v2_fast: { mlx: 15468, candle: 15889 },
  sensenova_u1_8b_infographic_v3_fast: { mlx: 15469, candle: 15892 },
  flux_schnell: { mlx: 15470, candle: 15895 },
  flux_dev: { mlx: 15471, candle: 15898 },
  ideogram_4: { mlx: 15472, candle: 15901 },
  ideogram_4_turbo: { mlx: 15473, candle: 15904 },
  boogu_image: { mlx: 15474, candle: 15907 },
  boogu_image_turbo: { mlx: 15475, candle: 15910 },
  boogu_image_edit: { mlx: 15476 },
  krea_2_turbo: { mlx: 15477, candle: 15913 },
  krea_2_raw: { mlx: 15478, candle: 15916 },
  flux2_klein_9b: { mlx: 15479, candle: 15919 },
  flux2_klein_9b_kv: { mlx: 15480 },
  flux2_klein_9b_true_v2: { mlx: 15481 },
  flux2_dev: { mlx: 15482, candle: 15922 },
  chroma1_hd: { mlx: 15483 },
  chroma1_base: { mlx: 15484 },
  chroma1_flash: { mlx: 15485 },
  kolors: { mlx: 15486, candle: 16171 },
  sd3_5_large: { mlx: 15487, candle: 15925 },
  sd3_5_large_turbo: { mlx: 15488, candle: 15928 },
  sd3_5_medium: { mlx: 15489, candle: 15931 },
  sana_1600m: { mlx: 15490 },
  sana_sprint_1600m: { mlx: 15491 },
  anima_base: { mlx: 15492 },
  anima_aesthetic: { mlx: 15493 },
  anima_turbo: { mlx: 15494 },
  sdxl: { mlx: 15495 },
  realvisxl: { mlx: 15496 },
  realvisxl_lightning: { mlx: 15497 },
  illustrious_xl_v1: { mlx: 15498 },
  illustrious_xl_v2: { mlx: 15499 },
  instantid_realvisxl: { mlx: 15500, candle: 15934 },
  pulid_flux_dev: { mlx: 15501, candle: 15937 },
  bernini_image: { mlx: 15502 },
};

// Keyed by the family's MLX story id, which doubles as the stable family group key. A family with no
// `candle` key owns no dual-backend model; adding one to the catalog fails generation until its
// Candle family twin is filed.
export const FAMILY_STORIES = {
  15509: { mlx: 15509, candle: 15813 },
  15510: { mlx: 15510, candle: 15815 },
  15511: { mlx: 15511, candle: 15817 },
  15512: { mlx: 15512, candle: 15819 },
  15513: { mlx: 15513, candle: 15821 },
  15514: { mlx: 15514, candle: 15823 },
  15515: { mlx: 15515, candle: 15825 },
  15516: { mlx: 15516, candle: 15827 },
  15517: { mlx: 15517, candle: 15829 },
  15518: { mlx: 15518, candle: 15831 },
  15519: { mlx: 15519, candle: 15833 },
  15520: { mlx: 15520 },
  15521: { mlx: 15521, candle: 16169 },
  15522: { mlx: 15522, candle: 15835 },
  15523: { mlx: 15523 },
  15524: { mlx: 15524 },
  15525: { mlx: 15525 },
  15526: { mlx: 15526, candle: 15837 },
  15527: { mlx: 15527, candle: 15839 },
  15528: { mlx: 15528 },
};

/** The stable family group key for a catalog id, which is the family's MLX story id. */
export function familyGroup(modelId) {
  if (modelId.startsWith("mage_flow")) return 15509;
  if (modelId.startsWith("z_image")) return 15510;
  if (modelId.startsWith("qwen_image")) return 15511;
  if (modelId.startsWith("lens")) return 15512;
  if (modelId.startsWith("sensenova")) return 15513;
  if (modelId === "flux_schnell" || modelId === "flux_dev") return 15514;
  if (modelId.startsWith("ideogram")) return 15515;
  if (modelId.startsWith("boogu")) return 15516;
  if (modelId.startsWith("krea_2")) return 15517;
  if (modelId.startsWith("flux2_klein")) return 15518;
  if (modelId === "flux2_dev") return 15519;
  if (modelId.startsWith("chroma1")) return 15520;
  if (modelId === "kolors") return 15521;
  if (modelId.startsWith("sd3_5")) return 15522;
  if (modelId.startsWith("sana")) return 15523;
  if (modelId.startsWith("anima")) return 15524;
  if (["sdxl", "realvisxl", "realvisxl_lightning", "illustrious_xl_v1", "illustrious_xl_v2"].includes(modelId)) return 15525;
  if (modelId === "instantid_realvisxl") return 15526;
  if (modelId === "pulid_flux_dev") return 15527;
  if (modelId === "bernini_image") return 15528;
  throw new Error(`no family story for ${modelId}`);
}

export function familyStory(modelId, backend) {
  const group = familyGroup(modelId);
  const stories = FAMILY_STORIES[group];
  if (!stories) throw new Error(`${modelId}: family SC-${group} has no ownership entry`);
  const story = stories[backend];
  if (!story) {
    throw new Error(
      `${modelId}: family SC-${group} owns no ${backend} story, so a ${backend} cell cannot be attributed — file the ${backend} family twin`,
    );
  }
  return story;
}

export function modelStory(modelId, backend) {
  const stories = MODEL_STORIES[modelId];
  if (!stories) throw new Error(`${modelId}: no owning model story`);
  const story = stories[backend];
  if (!story) {
    throw new Error(
      `${modelId}: no ${backend} owning model story, so a ${backend} cell cannot be attributed — file the ${backend} twin`,
    );
  }
  return story;
}

/**
 * Index every ownership story to the single owner and backend it is scoped to.
 *
 * The per-cell assertion below resolves a story id through this map and compares BOTH halves of what
 * it finds — the backend and the owner — so the map has to answer unambiguously. It is a plain `Map`,
 * so a repeated id is last-write-wins: the guard would then check every cell naming that id against
 * whichever claimant happened to be indexed last, quietly passing that one's cells and rejecting the
 * other's. Not vacuous, but no longer proof of anything. So one id means one owner and one backend,
 * enforced here rather than assumed downstream.
 */
export function buildStoryBackendScope(modelStories = MODEL_STORIES, familyStories = FAMILY_STORIES) {
  const scope = new Map();
  const claim = (storyId, backend, role, owner) => {
    if (!Number.isInteger(storyId)) {
      throw new Error(`${role} for ${owner}: story id ${JSON.stringify(storyId)} is not an integer`);
    }
    // Any repeat is a defect, not just a cross-backend one: two models sharing a story would
    // under-count the split, and a story used as both a model and a family owner is a typo.
    const existing = scope.get(storyId);
    if (existing) {
      throw new Error(
        `SC-${storyId} is claimed as the ${existing.backend} ${existing.role} of ${existing.owner} and as the ${backend} ${role} of ${owner}: an ownership story belongs to exactly one owner and one backend`,
      );
    }
    scope.set(storyId, { backend, role, owner });
  };
  for (const [modelId, byBackend] of Object.entries(modelStories)) {
    for (const [backend, storyId] of Object.entries(byBackend)) claim(storyId, backend, "model story", modelId);
  }
  for (const [group, byBackend] of Object.entries(familyStories)) {
    for (const [backend, storyId] of Object.entries(byBackend)) {
      claim(storyId, backend, "family story", `family SC-${group}`);
    }
  }
  return scope;
}

/**
 * What each cell ownership field must resolve to, expressed in `buildStoryBackendScope`'s own
 * `role`/`owner` vocabulary so the two cannot drift apart.
 */
const CELL_OWNERSHIP_FIELDS = {
  owningModelStory: { role: "model story", owner: (cell) => cell.modelId },
  owningFamilyStory: { role: "family story", owner: (cell) => `family SC-${familyGroup(cell.modelId)}` },
};

/**
 * SC-15812's un-regressable guard: every cell must name the ownership stories that actually cover it —
 * its own entry's (or family's) story, scoped to its own backend.
 *
 * Before this existed the generator copied one model-level story pair onto every cell, so all 2,260
 * candle cells in the shipped matrix named MLX-scoped stories and the epic's authoritative inventory
 * read as covering Candle work that no story would ever cover. Without an assertion the next drift is
 * silent again, so this throws rather than warns.
 *
 * The backend check alone is not enough, and shipping it alone was a real hole: it accepted ANY story
 * scoped to the right backend, so pointing a model's cells at a SIBLING model's same-backend twin
 * (e.g. `boogu_image_turbo`'s candle cells at `boogu_image`'s SC-15907) generated 30 mis-attributed
 * cells and exited 0. That is the same failure class as the original defect — a cell credited to a
 * story that cannot close it — reached by assignment rather than by backend. So the owner identity is
 * asserted too.
 */
export function assertCellOwnershipIsBackendScoped(cells, scope = buildStoryBackendScope()) {
  for (const cell of cells) {
    for (const [field, expected] of Object.entries(CELL_OWNERSHIP_FIELDS)) {
      const storyId = cell[field];
      if (!Number.isInteger(storyId)) {
        throw new Error(`${cell.id}: ${field} is ${JSON.stringify(storyId)}, not an ownership story id`);
      }
      const owner = scope.get(storyId);
      if (!owner) {
        throw new Error(`${cell.id}: ${field} SC-${storyId} is not a known backend-scoped ownership story`);
      }
      if (owner.backend !== cell.backend) {
        throw new Error(
          `${cell.id}: ${field} SC-${storyId} is scoped to ${owner.backend}, but the cell is ${cell.backend} — a ${cell.backend} cell cannot be covered by a ${owner.backend} story`,
        );
      }
      const expectedOwner = expected.owner(cell);
      if (owner.role !== expected.role || owner.owner !== expectedOwner) {
        throw new Error(
          `${cell.id}: ${field} SC-${storyId} is the ${owner.backend} ${owner.role} of ${owner.owner}, but this cell needs the ${cell.backend} ${expected.role} of ${expectedOwner} — a cell credited to another entry's story names a story that cannot close it`,
        );
      }
    }
  }
}

/**
 * The reconcile the story originally pinned as an absolute epic story count ("100 -> ~147"), restated
 * relatively so it stops going stale every time a story is filed. Twin coverage is derived from what
 * the catalog advertises: every dual-backend entry needs a Candle twin, and an mlx-only entry must
 * NOT have one, because an empty Candle story can never be closed.
 */
export function assertTwinCoverage(models, modelStories = MODEL_STORIES, familyStories = FAMILY_STORIES) {
  const dual = models.filter((model) => model.backends.includes("candle"));
  const dualGroups = new Set(dual.map((model) => familyGroup(model.id)));
  for (const model of models) {
    const stories = modelStories[model.id];
    if (!stories) throw new Error(`${model.id}: no owning model story`);
    const isDual = model.backends.includes("candle");
    if (isDual && !stories.candle) throw new Error(`${model.id}: advertises candle but has no Candle twin`);
    if (!isDual && stories.candle) {
      throw new Error(`${model.id}: advertises ${model.backends.join("/")} only but carries Candle twin SC-${stories.candle}`);
    }
  }
  for (const [group, stories] of Object.entries(familyStories)) {
    const owns = dualGroups.has(Number(group));
    if (owns && !stories.candle) throw new Error(`family SC-${group}: owns dual models but has no Candle twin`);
    if (!owns && stories.candle) {
      throw new Error(`family SC-${group}: owns no dual model but carries Candle twin SC-${stories.candle}`);
    }
  }
  const candleModelTwins = new Set(dual.map((model) => modelStories[model.id].candle));
  if (candleModelTwins.size !== dual.length) {
    throw new Error(`${dual.length} dual models map onto only ${candleModelTwins.size} distinct Candle model twins`);
  }
  const candleFamilyTwins = new Set([...dualGroups].map((group) => familyStories[group]?.candle));
  if (candleFamilyTwins.size !== dualGroups.size) {
    throw new Error(`${dualGroups.size} dual families map onto only ${candleFamilyTwins.size} distinct Candle family twins`);
  }
  return { dualModels: dual.length, dualFamilies: dualGroups.size };
}

function sha256(body) {
  return createHash("sha256").update(body).digest("hex");
}

function sortedUnique(values) {
  return [...new Set(values)].sort();
}

function calibrationAbiVersion(provider) {
  return CALIBRATION_ABI_VERSIONS[provider] ?? CALIBRATION_ABI_VERSIONS.default;
}

export function derivedCalibrationFingerprint({
  inferencePin,
  model,
  provider,
  backend,
  tier,
  mode,
  overlay,
  rung,
  parameters,
  manifestCalibration,
}) {
  return sha256(
    JSON.stringify({
      calibrationAbi: {
        provider,
        version: calibrationAbiVersion(provider),
      },
      inferencePin,
      model,
      backend,
      tier,
      mode,
      overlay,
      rung,
      parameters,
      manifestCalibration,
    }),
  );
}

function manifestCalibrationInputs(model, backend, tier) {
  const scope = model[backend] ?? {};
  return {
    minMemoryGb: scope.minMemoryGb ?? null,
    quantize: scope.quantize ?? null,
    sequentialPeakGb: scope.sequentialPeakGb?.[tier] ?? null,
    standardTierLayout: scope.standardTierLayout ?? null,
    vramGbByTier: scope.vramGbByTier?.[tier] ?? null,
  };
}

function runtimeStrategyParameters(parameters) {
  return Object.fromEntries(
    Object.entries(parameters).filter(([key]) =>
      [
        "decodeTileEdge",
        "decodeOverlap",
        "attentionChunkSize",
        "transformerWindowSize",
        "transformerWindowComponent",
      ].includes(key),
    ),
  );
}

function canonicalParameters(parameters) {
  return Object.fromEntries(Object.entries(parameters).sort(([left], [right]) => left.localeCompare(right)));
}

export function calibrationBinding(record, cell) {
  const reasons = [];
  if (record.status !== "complete") reasons.push("record-not-complete");
  if (record.quality.result !== "passed") reasons.push("quality-not-passed");
  if (record.sweep.rangeVerified !== true) reasons.push("range-not-verified");
  if (record.calibrationFingerprint !== cell.calibrationFingerprint) reasons.push("fingerprint-mismatch");
  if (!Array.isArray(cell.engagedRungs)) {
    reasons.push("composition-unavailable");
  } else if (JSON.stringify(record.strategy.engagedRungs) !== JSON.stringify(cell.engagedRungs)) {
    reasons.push("composition-mismatch");
  }
  if (
    JSON.stringify(canonicalParameters(record.strategy.parameters)) !==
    JSON.stringify(canonicalParameters(runtimeStrategyParameters(cell.strategyParameters)))
  ) reasons.push("strategy-parameters-mismatch");
  const resolution = `${record.target.geometry.width}x${record.target.geometry.height}`;
  if (!cell.geometryEnvelope.resolutions?.includes(resolution)) reasons.push("geometry-out-of-envelope");
  if (record.target.geometry.batch !== 1) reasons.push("batch-out-of-envelope");
  if (record.target.geometry.frames !== 1) reasons.push("frames-out-of-envelope");
  if (
    !cell.evidence.loadability.some(
      (artifact) =>
        artifact.repository === record.artifact.repository &&
        artifact.revision === record.artifact.resolvedRevision &&
        artifact.variant === record.artifact.variant,
    )
  ) reasons.push("artifact-loadability-mismatch");
  if (
    record.loadability.result !== "passed" ||
    !record.loadability.resolvedPathFingerprint
  ) reasons.push("loadability-not-passed");
  return { eligible: reasons.length === 0, reasons };
}

function expectedEngagedRungs({
  model,
  provider,
  backend,
  tier,
  mode,
  overlay,
  rung,
  status,
  calibrationPlan,
}) {
  if (Array.isArray(status.engagedRungs)) return status.engagedRungs;
  if (rung === "resident") return ["resident"];
  const matches = calibrationPlan.providers
    .filter(
      (candidate) =>
        candidate.target.modelId === model.id &&
        candidate.target.provider === provider &&
        candidate.backend === backend &&
        candidate.target.tier === tier &&
        candidate.target.mode === mode &&
        candidate.target.overlay === overlay &&
        candidate.rung === rung,
    )
    .map((candidate) => candidate.engagedRungs);
  if (matches.length === 0) return null;
  if (matches.some((candidate) => JSON.stringify(candidate) !== JSON.stringify(matches[0]))) {
    throw new Error(`${model.id}:${provider}:${backend}:${tier}:${mode}:${overlay}:${rung}: conflicting planned compositions`);
  }
  return matches[0];
}

// Derive the exact additive host requirement used by the Rust MLX admission envelope.
// This is a generated-data bridge only. It does not suggest a tier or add model-specific policy.
export function mlxRequiredHostBytes(record) {
  if (record?.backend !== "mlx") return null;
  const memoryBytes = record.hardware?.memoryBytes;
  const mlxLimit = record.hardware?.mlxMemoryLimitBytes;
  const wiredLimit = record.hardware?.wiredLimitBytes;
  const predicted = record.predictedPeakBytes?.overall;
  const wired = record.observedMemory?.overall?.wiredBytes;
  const reclaimable = record.observedMemory?.overall?.reclaimableBytes;
  const inputs = [memoryBytes, mlxLimit, wiredLimit, predicted, wired, reclaimable];
  if (!inputs.every((value) => Number.isSafeInteger(value) && value >= 0)) return null;

  const processCeiling = Math.min(memoryBytes, mlxLimit, wiredLimit);
  const foreignReserve = memoryBytes - processCeiling;
  const nonReclaimableWired = Math.max(0, wired - reclaimable);
  const required = Math.max(predicted, nonReclaimableWired) + foreignReserve;
  return Number.isSafeInteger(required) ? required : null;
}

function parseExpectedImageIds(source) {
  const match = source.match(/const EXPECTED_IMAGE_IDS:\s*&\[&str\]\s*=\s*&\[([\s\S]*?)\n\s*\];/);
  if (!match) throw new Error("could not locate EXPECTED_IMAGE_IDS in engines.rs");
  return [...match[1].matchAll(/"([^"]+)"/g)].map((item) => item[1]);
}

function parseEngineRoutes(source) {
  const table = source.match(/pub\(crate\) const MODEL_TABLE:[\s\S]*?=\s*&\[([\s\S]*?)\n\];/);
  if (!table) throw new Error("could not locate MODEL_TABLE in engines.rs");
  const routes = new Map();
  for (const row of table[1].matchAll(/ModelRow\s*\{([\s\S]*?)\n\s*\},/g)) {
    const model = row[1].match(/sceneworks_id:\s*"([^"]+)"/)?.[1];
    const engine = row[1].match(/engine_id:\s*"([^"]+)"/)?.[1];
    const repo = row[1].match(/default_repo:\s*"([^"]+)"/)?.[1] ?? null;
    if (model && engine) routes.set(model, { engine, repo, kind: "registry" });
  }
  routes.set("instantid_realvisxl", {
    engine: "instantid",
    repo: null,
    kind: "bespoke",
  });
  routes.set("pulid_flux_dev", {
    engine: "pulid_flux",
    repo: null,
    kind: "bespoke",
  });
  return routes;
}

function parseMlxSequentialEngines(source) {
  const test = source.match(
    /fn engine_supports_sequential_is_derived_from_the_registered_capability\(\)\s*\{([\s\S]*?)\n\s*\}\n\n\s*\/\/\/ An id with no registered generator/,
  );
  if (!test) {
    throw new Error("could not locate the MLX sequential-capability registry sweep");
  }
  const beforeNegativeControl = test[1].split("assert!(!engine_supports_sequential")[0];
  return new Set([...beforeNegativeControl.matchAll(/"([^"]+)"/g)].map((item) => item[1]));
}

function inferencePin(cargo) {
  const match = cargo.match(
    /candle-kernels\s*=\s*\{[^}]*?github\.com\/SceneWorks\/inference[^}]*?rev\s*=\s*"([0-9a-f]+)"/,
  );
  if (!match) throw new Error("could not resolve the pinned SceneWorks/inference revision");
  return match[1];
}

export function backendScopes(model, manifestById) {
  const inherited = model.id === "z_image_edit" ? manifestById.get("z_image_turbo") : model;
  const scopes = [];
  if (inherited?.mlx || ["instantid_realvisxl", "pulid_flux_dev"].includes(model.id)) scopes.push("mlx");
  if (inherited?.candle || ["instantid_realvisxl", "pulid_flux_dev"].includes(model.id)) scopes.push("candle");
  return scopes;
}

function tiersFor(model, backend, backendTierOverrides) {
  const override = backendTierOverrides.get(`${model.id}:${backend}`);
  if (override) return override;
  const backendTiers = Object.keys(model[backend]?.vramGbByTier ?? {});
  const downloadTiers = (model.downloads ?? [])
    .map((download) => download.variant)
    .filter((variant) => typeof variant === "string" && /^(bf16|fp16|q\d+|nvfp4|int\d+)/.test(variant));
  const inferred = model[backend]?.quantize === 4 ? ["q4"] : model[backend]?.quantize === 8 ? ["q8"] : [];
  const advertised =
    backend === "candle" && backendTiers.length
      ? backendTiers
      : [...backendTiers, ...downloadTiers, ...inferred];
  return sortedUnique(advertised).filter(
    (tier) => tier !== "int8-convrot",
  ).length
    ? sortedUnique(advertised).filter((tier) => tier !== "int8-convrot")
    : ["default"];
}

function parseBackendTierOverrides(instantIdSource) {
  const candleDense = instantIdSource.match(
    /#\[cfg\(not\(target_os = "macos"\)\)\]\s*let preferred = \{[\s\S]*?"([^"]+)"\s*\};/,
  )?.[1];
  if (!candleDense) {
    throw new Error("could not derive InstantID's dense Candle tier from instantid.rs");
  }
  return new Map([["instantid_realvisxl:candle", [candleDense]]]);
}

function modesFor(model) {
  const modes = (model.capabilities ?? []).filter((capability) => GENERATION_CAPABILITIES.has(capability));
  return modes.length ? sortedUnique(modes) : ["catalog_default"];
}

// SC-16069: the model ids that SHIP a strict-pose control lane. This is a DECLARATION of what exists,
// deliberately separate from whether that lane has been MEASURED.
//
// ## The blind spot this replaces
//
// `overlaysFor` used to emit the `control` overlay only when `model[backend].control` existed — but that
// key is a MEASUREMENT block (`bf16BranchPeakGbByTier`, `decodeTileSaveGb`, …), and `"control"` appears
// exactly once in `config/manifests/builtin.models.jsonc`: inside `krea_2_turbo`'s **candle** block. So a
// shipping lane with no measurements had **zero cells** — it was invisible to the very matrix that exists
// to show what is unmeasured. The MLX Krea control lane is the concrete case: `mlx-gen-krea`'s
// `model_control.rs` registers `krea_2_turbo_control` and the worker routes it
// (`image_jobs/krea_control.rs`), yet the matrix showed no `krea_2_turbo`/`mlx` control cell at all. Absent
// evidence read as absent feature, which is the exact false green this epic exists to prevent.
//
// ## Why both backends, for every id here
//
// Both worker routers wire the SAME seven families, and `crates/sceneworks-worker/src/image_jobs/base.rs`
// says so in the source: `WIRED_MLX_POSE_FAMILIES` is documented as "the MLX twin of
// `WIRED_CANDLE_POSE_FAMILIES` … and the SAME id set — every candle wired family has a matching MLX control
// lane". A model listed here therefore has a control lane on whichever backends it supports at all; the
// per-backend loop below already restricts to those. `candle_declaration_matches_the_memory_matrix_generator`
// in `crates/sceneworks-worker/src/tests/gpu_and_manifest.rs` is the cross-language guard: adding a candle
// control lane without adding it here turns that test red.
export const CONTROL_LANE_MODELS = [
  "flux2_dev",
  "flux_dev",
  "kolors",
  "krea_2_turbo",
  "qwen_image",
  "z_image",
  "z_image_turbo",
];

// Most control lanes reuse their catalog route id. Krea's MLX lane is a distinct production
// provider registered by mlx-gen-krea, so its evidence must bind to that provider instead of being
// orphaned behind the base `krea_2_turbo` route.
const CONTROL_PROVIDER_OVERRIDES = new Map([
  ["krea_2_turbo:mlx", "krea_2_turbo_control"],
]);

function providerFor(model, backend, overlay, route) {
  if (overlay !== "control") return route.engine;
  return CONTROL_PROVIDER_OVERRIDES.get(`${model.id}:${backend}`) ?? route.engine;
}

function matrixOverlayFor(recordOverlay) {
  return /^control:\d+$/.test(recordOverlay) ? "control" : recordOverlay;
}

function overlaysFor(model, backend) {
  const overlays = ["none"];
  if (model.loraCompatibility) overlays.push("lora");
  // A DECLARED lane, not a measured one — see CONTROL_LANE_MODELS.
  if (CONTROL_LANE_MODELS.includes(model.id)) overlays.push("control");
  if ((model.capabilities ?? []).includes("character_image")) overlays.push("identity");
  return sortedUnique(overlays);
}

// Drift in the other direction: a `[backend].control` MEASUREMENT block for a lane nobody declared. That
// would mean measurements exist for a lane the matrix does not believe ships, so the cells they belong to
// were never generated and the evidence is orphaned. Fail loudly at generation time rather than emit a
// matrix that quietly drops them (the same posture as MODEL_STORIES' missing-mapping failure).
function assertDeclaredControlLanes(models) {
  const undeclared = [];
  for (const model of models) {
    for (const backend of ["mlx", "candle"]) {
      if (model[backend]?.control && !CONTROL_LANE_MODELS.includes(model.id)) {
        undeclared.push(`${model.id}:${backend}`);
      }
    }
  }
  if (undeclared.length) {
    throw new Error(
      `memory-matrix: ${undeclared.join(", ")} carry a [backend].control measurement block but are not in ` +
        "CONTROL_LANE_MODELS, so no control cells were generated for them and the measurements are " +
        "orphaned. Add them to CONTROL_LANE_MODELS (sc-16069).",
    );
  }
}

function geometryFor(model, backend) {
  const limits = { ...(model.limits ?? {}), ...(model[backend]?.limits ?? {}) };
  const defaults = { ...(model.defaults ?? {}), ...(model[backend]?.defaults ?? {}) };
  const envelope = {
    defaultResolution: defaults.resolution ?? null,
    resolutions: Array.isArray(limits.resolutions) ? limits.resolutions : [],
    minWidth: limits.minWidth ?? limits.minSize ?? null,
    maxWidth: limits.maxWidth ?? limits.maxSize ?? null,
    minHeight: limits.minHeight ?? limits.minSize ?? null,
    maxHeight: limits.maxHeight ?? limits.maxSize ?? null,
  };
  return Object.fromEntries(
    Object.entries(envelope).filter(
      ([, value]) => value !== null && (!Array.isArray(value) || value.length > 0),
    ),
  );
}

function geometryWithinPixels(model, backend, maxPixels) {
  const envelope = geometryFor(model, backend);
  return {
    ...envelope,
    resolutions: (envelope.resolutions ?? []).filter((resolution) => {
      const [width, height] = resolution.split("x").map(Number);
      return Number.isSafeInteger(width) && Number.isSafeInteger(height) && width * height <= maxPixels;
    }),
  };
}

function artifactEvidence(model, route, tier) {
  const downloads = model.downloads ?? [];
  const tierMatches = downloads.filter((download) => download.variant === tier);
  const relevant = tierMatches.length
    ? [...tierMatches, ...downloads.filter((download) => download.variant == null)]
    : downloads;
  const artifacts = relevant.map((download) => ({
    repository: download.repo ?? null,
    revision: download.revision ?? null,
    variant: download.variant ?? null,
  }));
  if (!artifacts.length && route.repo) {
    artifacts.push({ repository: route.repo, revision: null, variant: null });
  }
  return [
    ...new Map(
      artifacts.map((artifact) => [
        `${artifact.repository}:${artifact.revision}:${artifact.variant}`,
        artifact,
      ]),
    ).values(),
  ];
}

function declaredEvidence(model, backend, tier) {
  const scope = model[backend] ?? {};
  const keys = [
    "minMemoryGb",
    "vramGbByTier",
    "sequentialPeakGb",
    "turboFit",
    "measured",
    "quantize",
    "standardTierLayout",
  ].filter((key) => scope[key] !== undefined);
  return keys.map((key) => ({
    source: `config/manifests/builtin.models.jsonc#models/${model.id}/${backend}/${key}`,
    tier,
  }));
}

function strategyStatus({ backend, rung, route, provider, sequentialEngines, model, tier, mode, overlay }) {
  const declaredCalibrations = (model[backend]?.calibrations ?? []).filter(
    (binding) =>
      binding.provider === provider &&
      binding.tier === tier &&
      binding.mode === mode &&
      matrixOverlayFor(binding.overlay) === overlay &&
      binding.rung === rung,
  );
  if (declaredCalibrations.length) {
    const fingerprints = sortedUnique(declaredCalibrations.map((binding) => binding.fingerprint));
    const parameters = sortedUnique(
      declaredCalibrations.map((binding) => JSON.stringify(binding.parameters ?? {})),
    );
    if (fingerprints.length !== 1 || parameters.length !== 1) {
      throw new Error(
        `${model.id}:${backend}:${tier}:${mode}:${overlay}:${rung} has inconsistent exact calibration bindings`,
      );
    }
    return {
      state: "Implemented/unverified",
      source: "crates/sceneworks-worker/src/mlx_fit_gate.rs#evidence_admission_route",
      parameters: JSON.parse(parameters[0]),
      calibrationFingerprint: fingerprints[0],
    };
  }
  if (
    rung === "resident" &&
    !(model.id === "krea_2_turbo" && backend === "candle" && mode === "text_to_image")
  ) {
    return {
      state: "Implemented/unverified",
      source: `crates/sceneworks-worker/src/engines.rs#${route.kind === "registry" ? "MODEL_TABLE" : "bespoke_advertised"}`,
      parameters: {},
    };
  }
  if (
    rung === "staged_residency" &&
    !(model.id === "krea_2_turbo" && backend === "candle") &&
    ((backend === "mlx" && sequentialEngines.has(route.engine)) ||
      (backend === "candle" &&
        (model.candle?.sequentialPeakGb !== undefined || model.candle?.turboFit !== undefined)))
  ) {
    return {
      state: "Implemented/unverified",
      source:
        backend === "mlx"
          ? "crates/sceneworks-worker/src/mlx_fit_gate.rs#engine_supports_sequential"
          : `config/manifests/builtin.models.jsonc#models/${model.id}/candle`,
      parameters: { phaseOrder: ["conditioning", "denoise", "decode"] },
    };
  }
  if (
    model.id === "krea_2_turbo" &&
    backend === "candle" &&
    mode === "text_to_image" &&
    model.candle?.turboFit?.phaseCurvesByTier?.[tier]
  ) {
    const rungKeys = {
      resident: "resident",
      staged_residency: "threeStage",
      bounded_decode: "tiledVae",
      bounded_attention: "chunkedAttention",
      bounded_transformer_residency: "streamedBlocks",
    };
    const manifestRung = rungKeys[rung];
    if (manifestRung && overlay === "none") {
      const verification = model.candle.turboFit.verification;
      const evidenceRecords = (model.candle.turboFit.evidenceRecords ?? []).filter(
        (record) => record.tier === tier,
      );
      const strategyParameters = model.candle.turboFit.strategyParameters?.[manifestRung];
      return {
        // This catalog cell spans the manifest's full resolution envelope. Exact measured records
        // are narrower, so the aggregate cell must remain unverified; runtime may promote only an
        // exact tier+geometry record after provider fingerprint/loadability checks.
        state: "Implemented/unverified",
        source: "crates/sceneworks-worker/src/vram_gate.rs#krea_turbo_fit",
        parameters: {
          manifestRung,
          formula:
            manifestRung === "resident"
              ? "vramGbByTier+cudaHeadroom"
              : "max(text,denoise,decode)+cudaHeadroom",
          ...strategyParameters,
        },
        engagedRungs: model.candle.turboFit.engagedCompositions?.[manifestRung],
        calibrationFingerprint: model.candle.turboFit.calibrationFingerprint,
        maxPixels: model.candle.turboFit.maxMeasuredPixels,
        historicalVerification: evidenceRecords.map((record) => ({
          source: `Shortcut ${record.sourceStory} activity ${record.sourceActivity}`,
          hardware: verification?.hardware,
          tier: record.tier,
          geometry: `${record.width}x${record.height}`,
          capturedAt: record.capturedAt,
          harnessVersion: record.harnessVersion,
          engagedRungs: record.measuredCompositions?.[manifestRung],
          observedPeakGb: record.observedPeaksGb?.[manifestRung],
          parity: record.parity,
        })),
        currentEnvironmentVerification: [],
        strategyParameterVerification: evidenceRecords
          .filter((record) => Number.isFinite(record.predictedPeaksGb?.[manifestRung]))
          .map((record) => ({
            source: `config/manifests/builtin.models.jsonc#models/${model.id}/candle/turboFit/evidenceRecords`,
            tier: record.tier,
            geometry: `${record.width}x${record.height}`,
            predictedPeakGb: record.predictedPeaksGb[manifestRung],
            engagedRungs: record.measuredCompositions?.[manifestRung],
            exactParameters: strategyParameters,
          })),
      };
    }
    if (rung === "bounded_transformer_residency" && overlay !== "none") {
      return {
        state: "Structurally N/A",
        source: "crates/sceneworks-worker/src/vram_gate.rs#krea_turbo_fit",
        parameters: {},
        structural: [
          {
            source: "crates/sceneworks-worker/src/image_jobs/base.rs#allow_streamed_blocks",
            reason: "load-time adapters are incompatible with streamed transformer blocks",
          },
        ],
      };
    }
  }
  return { state: "Missing", source: null, parameters: {} };
}

function validateMatrix(matrix, expectedIds, backendTierOverrides) {
  const ids = matrix.models.map((model) => model.id);
  if (ids.length !== EXPECTED_IMAGE_COUNT) {
    throw new Error(`expected exactly ${EXPECTED_IMAGE_COUNT} image entries, found ${ids.length}`);
  }
  if (
    new Set(expectedIds).size !== ids.length ||
    ids.some((id) => !expectedIds.includes(id)) ||
    expectedIds.some((id) => !ids.includes(id))
  ) {
    const manifestOnly = ids.filter((id) => !expectedIds.includes(id));
    const sourceOnly = expectedIds.filter((id) => !ids.includes(id));
    throw new Error(
      `manifest image ids, EXPECTED_IMAGE_IDS, and generated ownership rows disagree (manifest-only=${manifestOnly.join(",")}; source-only=${sourceOnly.join(",")})`,
    );
  }
  if (matrix.summary.mlxStagedStaticCoverage !== EXPECTED_MLX_STAGED_COUNT) {
    throw new Error(
      `expected MLX staged static coverage ${EXPECTED_MLX_STAGED_COUNT}/${EXPECTED_IMAGE_COUNT}, found ${matrix.summary.mlxStagedStaticCoverage}`,
    );
  }
  for (const [key, expectedTiers] of backendTierOverrides) {
    const [modelId, backend] = key.split(":");
    const actualTiers = sortedUnique(
      matrix.cells
        .filter((cell) => cell.modelId === modelId && cell.backend === backend)
        .map((cell) => cell.tier),
    );
    if (JSON.stringify(actualTiers) !== JSON.stringify([...expectedTiers].sort())) {
      throw new Error(
        `${key}: backend tier contradiction (expected ${expectedTiers.join(",")}, found ${actualTiers.join(",")})`,
      );
    }
  }
  assertTwinCoverage(matrix.models);
  assertCellOwnershipIsBackendScoped(matrix.cells);
  for (const model of matrix.models) {
    for (const map of ["owningFamilyStories", "owningModelStories"]) {
      const owned = Object.keys(model[map]).sort();
      if (JSON.stringify(owned) !== JSON.stringify([...model.backends].sort())) {
        throw new Error(
          `${model.id}: ${map} covers ${owned.join(",") || "nothing"} but the entry advertises ${model.backends.join(",")}`,
        );
      }
    }
  }
  for (const cell of matrix.cells) {
    if (cell.state !== "Missing" && cell.evidence.staticImplementation.length === 0) {
      throw new Error(`${cell.id}: non-Missing classification has no static evidence`);
    }
    if (cell.state === "Structurally N/A" && cell.evidence.structural.length === 0) {
      throw new Error(`${cell.id}: Structurally N/A classification has no structural evidence`);
    }
    if (cell.state === "Verified") {
      const dynamic = cell.evidence.currentEnvironmentVerification;
      if (!dynamic.length || !cell.calibrationFingerprint || !Object.keys(cell.strategyParameters).length) {
        throw new Error(`${cell.id}: unsupported Full/Verified claim`);
      }
    }
  }
}

export async function buildMatrix({ sourceOverrides = {} } = {}) {
  const sourcePaths = {
    manifest: "config/manifests/builtin.models.jsonc",
    engines: "crates/sceneworks-worker/src/engines.rs",
    mlxFitGate: "crates/sceneworks-worker/src/mlx_fit_gate.rs",
    memoryStrategy: "crates/sceneworks-worker/src/memory_strategy.rs",
    vramGate: "crates/sceneworks-worker/src/vram_gate.rs",
    instantId: "crates/sceneworks-worker/src/image_jobs/instantid.rs",
    calibrationEvidence: "docs/generated/memory-calibration-evidence.json",
    calibrationPlan: "config/memory-calibration-plan.json",
    cargo: "Cargo.toml",
  };
  const sourceEntries = Object.entries(sourcePaths);
  const sourceBodies = await Promise.all(
    sourceEntries.map(async ([name, relative]) =>
      canonicalSourceText(
        Object.hasOwn(sourceOverrides, name)
          ? sourceOverrides[name]
          : await readFile(path.join(ROOT, relative), "utf8"),
      ),
    ),
  );
  const bodies = Object.fromEntries(
    sourceEntries.map(([name], index) => [name, sourceBodies[index]]),
  );
  const manifestBody = bodies.manifest;
  const enginesBody = bodies.engines;
  const mlxFitBody = bodies.mlxFitGate;
  const cargoBody = bodies.cargo;
  const calibrationBundle = validateCalibrationBundle(JSON.parse(bodies.calibrationEvidence));
  const calibrationPlan = JSON.parse(bodies.calibrationPlan);
  const manifest = JSON.parse(stripJsoncComments(manifestBody));
  // JSONC comments and formatting are not part of the manifest contract. Hash the parsed value so
  // provenance embedded in generated artifacts is stable across semantically inert source edits.
  const revisionBodies = {
    ...bodies,
    manifest: JSON.stringify(manifest),
  };
  const images = manifest.models.filter((model) => model.type === "image");
  const manifestById = new Map(images.map((model) => [model.id, model]));
  const expectedIds = parseExpectedImageIds(enginesBody);
  const routes = parseEngineRoutes(enginesBody);
  const sequentialEngines = parseMlxSequentialEngines(mlxFitBody);
  const backendTierOverrides = parseBackendTierOverrides(bodies.instantId);
  const pin = inferencePin(cargoBody);
  const sceneWorksRevision = `source-tree:${sha256(
    sourceEntries
      .filter(([name]) => name !== "calibrationEvidence")
      .map(([name]) => revisionBodies[name])
      .join(""),
  )}`;

  // sc-16069: no orphaned control measurements — a `[backend].control` block for an undeclared lane means
  // its cells were never generated.
  assertDeclaredControlLanes(images);

  const models = images
    .map((model) => {
      const route = routes.get(model.id);
      if (!route) throw new Error(`${model.id}: no resolved route/provider`);
      const backends = backendScopes(model, manifestById);
      return {
        id: model.id,
        name: model.name,
        family: model.family ?? null,
        resolvedRoute: route.engine,
        routeKind: route.kind,
        backends,
        // Per-backend maps, not scalars: the entry has one owner per backend it advertises, and
        // `familyStory`/`modelStory` throw if the catalog advertises a backend nobody owns.
        owningFamilyStories: Object.fromEntries(
          backends.map((backend) => [backend, familyStory(model.id, backend)]),
        ),
        owningModelStories: Object.fromEntries(
          backends.map((backend) => [backend, modelStory(model.id, backend)]),
        ),
      };
    })
    .sort((left, right) => left.id.localeCompare(right.id));

  const cells = [];
  for (const modelSummary of models) {
    const model = manifestById.get(modelSummary.id);
    const route = routes.get(model.id);
    for (const backend of modelSummary.backends) {
      // SC-15812: resolved HERE, inside the per-backend loop, so a cell names the story that owns
      // its (model, backend) pair rather than whichever backend happened to be listed first.
      const owningFamilyStory = modelSummary.owningFamilyStories[backend];
      const owningModelStory = modelSummary.owningModelStories[backend];
      for (const tier of tiersFor(model, backend, backendTierOverrides)) {
        for (const mode of modesFor(model)) {
          for (const overlay of overlaysFor(model, backend)) {
            const provider = providerFor(model, backend, overlay, route);
            for (const rung of RUNGS) {
              const status = strategyStatus({
                backend,
                rung,
                route,
                provider,
                sequentialEngines,
                model,
                tier,
                mode,
                overlay,
              });
              const fingerprint =
                status.state === "Missing"
                  ? null
                  : status.calibrationFingerprint ??
                    derivedCalibrationFingerprint({
                      inferencePin: pin,
                      model: model.id,
                      provider,
                      backend,
                      tier,
                      mode,
                      overlay,
                      rung,
                      parameters: status.parameters,
                      manifestCalibration: manifestCalibrationInputs(model, backend, tier),
                    });
              const calibrationRuns = calibrationBundle.records.filter(
                (record) =>
                  record.target.modelId === model.id &&
                  record.target.provider === provider &&
                  record.backend === backend &&
                  record.target.tier === tier &&
                  record.target.mode === mode &&
                  matrixOverlayFor(record.target.overlay) === overlay &&
                  record.strategy.rung === rung,
              );
              const engagedRungs = expectedEngagedRungs({
                model,
                provider: route.engine,
                backend,
                tier,
                mode,
                overlay,
                rung,
                status,
                calibrationPlan,
              });
              const runSummary = (record) => {
                const overall = record.observedMemory?.overall?.deviceBytes;
                const requiredHostBytes = mlxRequiredHostBytes(record);
                return {
                  source: `docs/generated/memory-calibration-evidence.json#${record.id}`,
                  hardware: record.backend === "candle" ? record.hardware.name : record.hardware.chip,
                  tier: record.target.tier,
                  geometry: `${record.target.geometry.width}x${record.target.geometry.height}`,
                  capturedAt: record.capturedAt,
                  harnessVersion: record.harnessVersion,
                  engagedRungs: record.strategy.engagedRungs,
                  ...(Number.isFinite(overall) ? { observedPeakGb: overall / 1024 ** 3 } : {}),
                  ...(requiredHostBytes !== null ? { requiredHostBytes } : {}),
                  parity: {
                    contract: record.quality.contract,
                    result: record.quality.result === "not_run" ? "not_run" : record.quality.result,
                    metric: "maximum_absolute_error",
                    maximumError: record.quality.maximumError,
                    fixture: record.fixture,
                  },
                };
              };
              const eligibleRuns = calibrationRuns.filter(
                (record) => calibrationBinding(record, {
                  ...{
                    calibrationFingerprint: fingerprint,
                    engagedRungs,
                    strategyParameters: status.parameters,
                    geometryEnvelope: status.maxPixels
                      ? geometryWithinPixels(model, backend, status.maxPixels)
                      : geometryFor(model, backend),
                    evidence: { loadability: artifactEvidence(model, route, tier) },
                  },
                }).eligible,
              );
              const historicalRuns = eligibleRuns.filter(
                (record) =>
                  evidenceSemantics(record, {
                    sceneWorks: sceneWorksRevision,
                    inference: pin,
                  }) === "historical",
              );
              const currentRuns = eligibleRuns.filter(
                (record) =>
                  evidenceSemantics(record, {
                    sceneWorks: sceneWorksRevision,
                    inference: pin,
                  }) === "current" && record.status === "complete",
              );
              cells.push({
                id: [model.id, provider, backend, tier, mode, overlay, rung].join(":"),
                modelId: model.id,
                resolvedRoute: provider,
                provider,
                backend,
                tier,
                mode,
                overlay,
                rung,
                geometryEnvelope: status.maxPixels
                  ? geometryWithinPixels(model, backend, status.maxPixels)
                  : geometryFor(model, backend),
                strategyParameters: status.parameters,
                engagedRungs,
                state: status.state,
                evidenceRevision: {
                  sceneWorks: sceneWorksRevision,
                  inference: pin,
                },
                calibrationFingerprint: fingerprint,
                owningFamilyStory,
                owningModelStory,
                evidence: {
                  staticImplementation: status.source ? [{ source: status.source }] : [],
                  declaredCalibration: declaredEvidence(model, backend, tier),
                  historicalVerification: [
                    ...(status.historicalVerification ?? []),
                    ...historicalRuns.map(runSummary),
                  ],
                  currentEnvironmentVerification: [
                    ...(status.currentEnvironmentVerification ?? []),
                    ...currentRuns.map(runSummary),
                  ],
                  loadability: artifactEvidence(model, route, tier),
                  strategyParameterVerification: status.strategyParameterVerification ?? [],
                  structural: status.structural ?? [],
                },
              });
            }
          }
        }
      }
    }
  }
  cells.sort((left, right) => left.id.localeCompare(right.id));
  const calibrationRuns = calibrationBundle.records.map((record) => {
    const cell = cells.find(
      (candidate) =>
        candidate.modelId === record.target.modelId &&
        candidate.resolvedRoute === record.target.provider &&
        candidate.backend === record.backend &&
        candidate.tier === record.target.tier &&
        candidate.mode === record.target.mode &&
        candidate.overlay === matrixOverlayFor(record.target.overlay) &&
        candidate.rung === record.strategy.rung,
    );
    if (!cell) throw new Error(`${record.id}: calibration record does not map to a matrix cell`);
    return {
      cellId: cell.id,
      binding: calibrationBinding(record, cell),
      semantics: evidenceSemantics(record, {
        sceneWorks: sceneWorksRevision,
        inference: pin,
      }),
      record,
    };
  });

  const modelSlices = Object.fromEntries(
    models.map((model) => [
      model.id,
      cells.filter((cell) => cell.modelId === model.id).map((cell) => cell.id),
    ]),
  );
  const mlxStagedModels = new Set(
    cells
      .filter(
        (cell) =>
          cell.backend === "mlx" &&
          cell.rung === "staged_residency" &&
          cell.state === "Implemented/unverified",
      )
      .map((cell) => cell.modelId),
  );
  const matrix = {
    // 2 (SC-15812): `models[].owningFamilyStory`/`owningModelStory` were both RENAMED (now plural)
    // and RETYPED (integer -> backend->id object). A reader written against 1 gets `undefined` for
    // both, so the two shapes cannot share a version number — that is the whole job of this field.
    schemaVersion: 2,
    generatedFrom: {
      sceneWorksRevision,
      inferenceRevision: pin,
      sources: Object.fromEntries(
        sourceEntries.map(([name, source]) => [
          name,
          { path: source, sha256: sha256(revisionBodies[name]) },
        ]),
      ),
    },
    conformanceStates: [
      "Verified",
      "Implemented/unverified",
      "Structurally N/A",
      "Missing",
      "Route unavailable/broken",
    ],
    evidenceDimensions: [
      "staticImplementation",
      "declaredCalibration",
      "historicalVerification",
      "currentEnvironmentVerification",
      "loadability",
      "strategyParameterVerification",
    ],
    summary: {
      imageModels: models.length,
      cells: cells.length,
      mlxStagedStaticCoverage: mlxStagedModels.size,
      mlxStagedStaticCoverageDenominator: EXPECTED_IMAGE_COUNT,
      fullModels: 0,
      calibrationRuns: calibrationBundle.records.length,
      currentCalibrationRuns: cells.reduce(
        (count, cell) => count + cell.evidence.currentEnvironmentVerification.length,
        0,
      ),
    },
    models,
    cells,
    calibrationRuns,
    modelSlices,
  };
  validateMatrix(matrix, expectedIds, backendTierOverrides);
  return matrix;
}

function renderMarkdown(matrix) {
  const lines = [
    "# Generated memory-ladder matrix",
    "",
    "> Generated by `scripts/generate-memory-matrix.mjs`. Do not edit by hand.",
    "",
    `- SceneWorks revision: \`${matrix.generatedFrom.sceneWorksRevision}\``,
    `- Inference revision: \`${matrix.generatedFrom.inferenceRevision}\``,
    `- Catalog entries: ${matrix.summary.imageModels}`,
    `- Cells: ${matrix.summary.cells}`,
    `- MLX staged-residency static coverage: ${matrix.summary.mlxStagedStaticCoverage}/${matrix.summary.mlxStagedStaticCoverageDenominator}`,
    `- Full models: ${matrix.summary.fullModels}`,
    "",
    "Static capability is never promoted to dynamic verification. Generated cells contain separate declared, historical, current-environment, loadability, and strategy-parameter evidence arrays.",
    "",
    "One row per (catalog entry, backend): ownership is backend-scoped, so a single row per entry could only name one backend's stories (SC-15812).",
    "",
    "| Catalog entry | Backend | Route | Family story | Model story | Staged residency |",
    "| --- | --- | --- | --- | ---: | --- |",
  ];
  for (const model of matrix.models) {
    for (const backend of model.backends) {
      const staged = matrix.cells.some(
        (cell) =>
          cell.modelId === model.id &&
          cell.backend === backend &&
          cell.rung === "staged_residency" &&
          cell.state === "Implemented/unverified",
      );
      lines.push(
        `| \`${model.id}\` | ${backend} | \`${model.resolvedRoute}\` (${model.routeKind}) | SC-${model.owningFamilyStories[backend]} | SC-${model.owningModelStories[backend]} | ${staged ? "Implemented/unverified" : "Missing"} |`,
      );
    }
  }
  lines.push(
    "",
    "Per-model consumers must use `modelSlices` in the JSON artifact. A cell is Full only when every applicable rung is Verified or Structurally N/A; this static baseline intentionally reports zero Full models.",
    "",
  );
  return lines.join("\n");
}

async function main() {
  const matrix = await buildMatrix();
  const json = `${JSON.stringify(matrix, null, 2)}\n`;
  const markdown = renderMarkdown(matrix);
  const check = process.argv.includes("--check");
  if (check) {
    const [existingJsonBody, existingMarkdownBody] = await Promise.all([
      readFile(path.join(ROOT, OUTPUT_JSON), "utf8"),
      readFile(path.join(ROOT, OUTPUT_MD), "utf8"),
    ]);
    const existingJson = canonicalSourceText(existingJsonBody);
    const existingMarkdown = canonicalSourceText(existingMarkdownBody);
    if (existingJson !== json || existingMarkdown !== markdown) {
      throw new Error("generated memory matrix is stale; run npm run generate:memory-matrix");
    }
    return;
  }
  await mkdir(path.join(ROOT, "docs/generated"), { recursive: true });
  await Promise.all([
    writeFile(path.join(ROOT, OUTPUT_JSON), json),
    writeFile(path.join(ROOT, OUTPUT_MD), markdown),
  ]);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
