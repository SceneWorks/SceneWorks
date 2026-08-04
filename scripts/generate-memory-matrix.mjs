#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import { canonicalSourceText, semanticSourceBody } from "./lib/source-revision.mjs";
import { routedLanes } from "./check-tier-integrity.mjs";
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

// sc-17497: the audit is layered. `auditedObjects` pins the CAPTURED side of FLUX.2's Candle/CUDA
// compile closure — the code the measurements were taken against, immutable forever. A live pin
// whose objects are all byte-identical is authorized by that alone, for free (the v1 shape).
//
// When a path moves, object identity can no longer decide: the 42-line `//!` doc comment in
// inference `35251a88` moved `candle-gen`'s tree while provably changing nothing that compiles.
// `artifactProof` is then the answer — the SHA-256 of the LINKED `candle-gen-flux2` lib test binary
// (the exact target that produced the measurements), built at both revisions under one toolchain by
// `scripts/inference-artifact-audit.mjs`. Identical bytes there mean identical compiled code, which
// is the claim the calibration actually depends on.
//
// `null` means "the live pin needs no artifact proof". Setting it demands a real CUDA build: the
// digest is frozen HERE, in source, so the checked-in record cannot authorize itself.
//
// `adjudicates` is the other half, and it is not optional. A digest speaks only for the paths the
// audited binary LINKS: `runtime-cuda` depends on `candle-gen-flux2`, not the reverse, so a commit
// into the CUDA bundle leaves the measurement binary byte-identical. Pinning the adjudicable set
// here keeps an unchanged digest from being read as proof over a path it never compiled.
const FLUX2_COMPATIBILITY_AUDIT = Object.freeze({
  story: "SC-15833",
  modelId: "flux2_dev",
  provider: "flux2_dev",
  capturedInferenceRevision: "5ffd7612e7de4e76b6db00a7148ed3d9c15b4c0d",
  compatibleInferenceRevision: "277f423822bf1899340ed3d867c3d6a773473d7b",
  artifactProof: null,
  auditedObjects: Object.freeze({
    "Cargo.toml": "8f5af6b9d53bbfe3be5d9d79b8949364138a087c",
    "crates/contracts/gen-core": "9a7e86f5893e584a8d0d656147abc4ae93af6922",
    "crates/bundles/runtime-cuda": "aba807f775872760e72fd98f28a5a0d2853cf00f",
    "crates/media/candle-gen/candle-gen": "e8b8b3f0787fac49539a2ef1085c48c9fdc9ec57",
    "crates/media/candle-gen/candle-gen-pid": "f3c8db10f1a872fc8fdb2c7243e607591886a5fa",
    "crates/media/candle-gen/candle-gen-flux2": "f91cd1a302f0d27f82bbc9c60bd4e578390e44b1",
    "crates/media/candle-gen/vendor/candle-kernels": "3b8327cf01d346c8068a5e9d096dcdddca440e99",
  }),
});

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
  lens: { mlx: 15462, candle: 17489 },
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
  boogu_image_edit: { mlx: 15476, candle: 17481 },
  krea_2_turbo: { mlx: 15477, candle: 15913 },
  krea_2_raw: { mlx: 15478, candle: 15916 },
  flux2_klein_9b: { mlx: 15479, candle: 15919 },
  flux2_klein_9b_kv: { mlx: 15480, candle: 17485 },
  flux2_klein_9b_true_v2: { mlx: 15481, candle: 17486 },
  flux2_dev: { mlx: 15482, candle: 15922 },
  chroma1_hd: { mlx: 15483, candle: 17484 },
  chroma1_base: { mlx: 15484, candle: 17482 },
  chroma1_flash: { mlx: 15485, candle: 17483 },
  kolors: { mlx: 15486, candle: 16171 },
  sd3_5_large: { mlx: 15487, candle: 15925 },
  sd3_5_large_turbo: { mlx: 15488, candle: 15928 },
  sd3_5_medium: { mlx: 15489, candle: 15931 },
  sana_1600m: { mlx: 15490, candle: 17492 },
  sana_sprint_1600m: { mlx: 15491, candle: 17493 },
  anima_base: { mlx: 15492, candle: 17477 },
  anima_aesthetic: { mlx: 15493, candle: 17478 },
  anima_turbo: { mlx: 15494, candle: 17479 },
  sdxl: { mlx: 15495, candle: 17494 },
  realvisxl: { mlx: 15496, candle: 17490 },
  realvisxl_lightning: { mlx: 15497, candle: 17491 },
  illustrious_xl_v1: { mlx: 15498, candle: 17487 },
  illustrious_xl_v2: { mlx: 15499, candle: 17488 },
  instantid_realvisxl: { mlx: 15500, candle: 15934 },
  pulid_flux_dev: { mlx: 15501, candle: 15937 },
  bernini_image: { mlx: 15502, candle: 17480 },
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
  15520: { mlx: 15520, candle: 17410 },
  15521: { mlx: 15521, candle: 16169 },
  15522: { mlx: 15522, candle: 15835 },
  15523: { mlx: 15523, candle: 17411 },
  15524: { mlx: 15524, candle: 17412 },
  15525: { mlx: 15525, candle: 17413 },
  15526: { mlx: 15526, candle: 15837 },
  15527: { mlx: 15527, candle: 15839 },
  15528: { mlx: 15528, candle: 17414 },
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
 * Prove that the emitted cells still equal the catalog cross-product that was resolved for this run.
 *
 * This deliberately does NOT pin today's total. Tiers, modes, overlays, backends, and even catalog
 * entries legitimately drift; their resolved axis sizes are the expectation. The schema's low
 * `minItems` can only be a shape sanity check because JSON Schema cannot derive this cross-product.
 * Keeping the exact check here also makes a dropped generation branch fail before either artifact is
 * written.
 */
export function assertCellInventoryMatchesCatalog(cells, expectedByScope) {
  const actualByScope = new Map();
  const seenIds = new Set();

  for (const cell of cells) {
    if (seenIds.has(cell.id)) {
      throw new Error(`${cell.id}: duplicate memory-matrix cell id`);
    }
    seenIds.add(cell.id);

    const scope = `${cell.modelId}:${cell.backend}`;
    if (!expectedByScope.has(scope)) {
      throw new Error(`${cell.id}: emitted a cell for unexpected catalog scope ${scope}`);
    }
    actualByScope.set(scope, (actualByScope.get(scope) ?? 0) + 1);
  }

  for (const [scope, expected] of expectedByScope) {
    if (!Number.isInteger(expected.cells) || expected.cells < 1) {
      throw new Error(`${scope}: catalog axes resolved to no memory-matrix cells`);
    }
    const actual = actualByScope.get(scope) ?? 0;
    if (actual !== expected.cells) {
      throw new Error(
        `${scope}: catalog cross-product expects ${expected.cells} cells ` +
          `(${expected.tiers} tiers x ${expected.modes} modes x ${expected.overlays} overlays x ${expected.rungs} rungs), emitted ${actual}`,
      );
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

// SC-16060. "Implemented" is a claim about the CODE, and `Verified` is `Implemented/unverified` plus
// evidence — never a replacement for it. Before the promotion producer existed no cell could hold
// `Verified`, so every coverage count could spell this as one exact state and stay correct by
// accident. The first promoted cell would have silently dropped out of those counts. Coverage
// surfaces must go through here rather than re-spelling the comparison.
export function isImplemented(state) {
  return ["Implemented/unverified", "Runtime verified", "Verified"].includes(state);
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
  if (!["complete", "runtime_complete"].includes(record.status)) reasons.push("record-not-complete");
  if (record.quality.result !== "passed") reasons.push("quality-not-passed");
  if (record.sweep.rangeVerified !== true) reasons.push("range-not-verified");
  if (record.calibrationFingerprint !== cell.calibrationFingerprint) reasons.push("fingerprint-mismatch");
  if (!Array.isArray(cell.engagedRungs)) {
    reasons.push("composition-unavailable");
  } else if (JSON.stringify(record.strategy.engagedRungs) !== JSON.stringify(cell.engagedRungs)) {
    reasons.push("composition-mismatch");
  }
  if (
    JSON.stringify(canonicalParameters(runtimeStrategyParameters(record.strategy.parameters))) !==
    JSON.stringify(canonicalParameters((() => {
      const parameters = runtimeStrategyParameters(cell.strategyParameters);
      // Older records predate this exact string axis. Preserve their historical binding behavior,
      // while new records that carry the component must match it exactly.
      if (!Object.hasOwn(record.strategy.parameters, "transformerWindowComponent")) {
        delete parameters.transformerWindowComponent;
      }
      return parameters;
    })()))
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

// SC-16060. A cell spans a geometry ENVELOPE; measured evidence covers POINTS inside it. Two
// separate claims live on a cell, and only one of them is geometry-sensitive:
//
//   `state`                  — the rung WORKS. Implemented, parity-passed, engaging the claimed
//                              composition. One measured geometry establishes it, and envelope
//                              membership is the right binding, because whether a rung executes
//                              does not depend on the resolution it executed at.
//   `memoryCharacterization` — the rung's PEAKS are known across the envelope. Geometry-sensitive
//                              by construction: `fixedGb + perMpxGb * megapixels`
//                              (`vram_gate.rs#krea_phase_curve`) has two coefficients, so one
//                              point cannot determine a slope.
//
// Collapsing both into `state` is what let a single 768x768 capture read as certifying a cell whose
// envelope reaches 2048x2048. `point` is the honest middle state the five-value vocabulary could not
// express — and it is exactly where Krea's shipped q8/bf16 curves sit, each carrying one geometry
// point while the gate's own doc comment calls their slopes "fitted from real renders at multiple
// resolutions".
//
// `fitted` asserts the evidence is SUFFICIENT to determine the affine curve, not that a fit has been
// performed. `coveredPixelBound` is the largest measured area, so a consumer can tell how far the
// determinable curve reaches without re-deriving it from `measuredGeometries`; it is null below two
// points because there is no curve to bound.
export function memoryCharacterization(geometries) {
  const measured = sortedUnique(
    geometries.filter((geometry) => /^[1-9][0-9]*x[1-9][0-9]*$/.test(geometry ?? "")),
  );
  const areas = measured.map((geometry) => {
    const [width, height] = geometry.split("x").map(Number);
    return width * height;
  });
  return {
    status: measured.length === 0 ? "unmeasured" : measured.length === 1 ? "point" : "fitted",
    measuredGeometries: measured,
    coveredPixelBound: areas.length > 1 ? Math.max(...areas) : null,
  };
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
        matrixOverlayFor(candidate.target.overlay) === overlay &&
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

export function observedPeakBytes(record) {
  const overall = record?.observedMemory?.overall;
  const value = overall?.deviceBytes ?? overall?.activeBytes;
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
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

// sc-16268: anchored on CODE, not on the `/// An id with no registered generator` doc comment that
// used to terminate the region. Provenance now hashes these sources with their inert comments
// stripped, so a parse that reads comment text would let a semantic change slip past the staleness
// tripwire. The negative control (`assert!(!engine_supports_sequential(...)`) is the real end of the
// positive sweep and was already the split point, so the parsed set is unchanged.
function parseMlxSequentialEngines(source) {
  const test = source.match(
    /fn engine_supports_sequential_is_derived_from_the_registered_capability\(\)\s*\{([\s\S]*?)assert!\(!engine_supports_sequential/,
  );
  if (!test) {
    throw new Error("could not locate the MLX sequential-capability registry sweep");
  }
  return new Set([...test[1].matchAll(/"([^"]+)"/g)].map((item) => item[1]));
}

function inferencePin(cargo) {
  const match = cargo.match(
    /candle-kernels\s*=\s*\{[^}]*?github\.com\/SceneWorks\/inference[^}]*?rev\s*=\s*"([0-9a-f]+)"/,
  );
  if (!match) throw new Error("could not resolve the pinned SceneWorks/inference revision");
  return match[1];
}

export function backendScopes(model, routedBackends) {
  const served = routedBackends.get(model.id) ?? new Set();
  return ["mlx", "candle"].filter((backend) => served.has(backend));
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
// key is a MEASUREMENT block (`peakGbByTier`, `decodeTileSaveGb`, …), and `"control"` appears
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

// Most control lanes reuse their catalog route id. Krea and Z-Image's MLX lanes are distinct
// production providers, so their evidence must bind to those providers instead of being orphaned
// behind the base catalog routes.
const CONTROL_PROVIDER_OVERRIDES = new Map([
  ["krea_2_turbo:mlx", "krea_2_turbo_control"],
  ["z_image:mlx", "z_image_control"],
  ["z_image_turbo:mlx", "z_image_turbo_control"],
]);

// The pinned Z-Image crate exports one provider-id-specific contract for each of its four registry
// variants from the same `memory_strategy_contract(provider_id, spec)` implementation. The manifest
// stores the declaration once on each catalog base entry; allow only these source-proven aliases to
// consume it. This is intentionally narrower than route equivalence: Qwen's separate control
// provider, for example, remains unbounded and must not inherit the base declaration.
const STATIC_CONTRACT_PROVIDER_ALIASES = new Map([
  ["z_image_control", "z_image"],
  ["z_image_turbo_control", "z_image_turbo"],
]);

function staticContractCoversProvider(contract, provider) {
  if (!contract) return false;
  const aliasedProvider = STATIC_CONTRACT_PROVIDER_ALIASES.get(provider);
  return contract.provider === provider ||
    (aliasedProvider !== undefined && aliasedProvider === contract.provider);
}

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

function rustStringSlice(source, name) {
  const match = source.match(
    new RegExp(`const\\s+${name}:\\s*&\\[&str\\]\\s*=\\s*&\\[([\\s\\S]*?)\\];`),
  );
  if (!match) throw new Error(`memory-matrix: could not derive ${name} from image_jobs/base.rs`);
  return [...match[1].matchAll(/"([^"]+)"/g)].map((entry) => entry[1]);
}

// Fail closed in both directions. A worker route missing from CONTROL_LANE_MODELS would otherwise emit
// zero cells, while a measurement block missing from the declaration would be orphaned. The MLX and
// Candle router lists are checked independently: their documented twin relationship is not evidence and
// must not be trusted if either source list drifts.
function assertDeclaredControlLanes(models, imageRoutingSource) {
  const declared = new Set(CONTROL_LANE_MODELS);
  for (const [backend, sourceName] of [
    ["mlx", "WIRED_MLX_POSE_FAMILIES"],
    ["candle", "WIRED_CANDLE_POSE_FAMILIES"],
  ]) {
    const wired = new Set(rustStringSlice(imageRoutingSource, sourceName));
    const undeclaredRoutes = [...wired].filter((id) => !declared.has(id)).sort();
    const unwiredDeclarations = [...declared].filter((id) => !wired.has(id)).sort();
    if (undeclaredRoutes.length || unwiredDeclarations.length) {
      throw new Error(
        `memory-matrix: ${backend} control routes and CONTROL_LANE_MODELS disagree ` +
          `(advertised but undeclared=${undeclaredRoutes.join(",") || "none"}; ` +
          `declared but not advertised=${unwiredDeclarations.join(",") || "none"}). ` +
          "A shipping control route without a declaration would generate zero control cells (sc-16073).",
      );
    }
  }

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

export const RUNG4_APPLICABILITIES = Object.freeze([
  "full",
  "partial",
  "none",
  "requires-different-primitive",
]);
export const RUNG4_IMPLEMENTATIONS = Object.freeze(["shared-primitive", "provider-local", "none"]);
export const RUNG4_REQUEST_PEAKS = Object.freeze(["moves", "does-not-move", "unmeasured"]);

/**
 * Parse and validate the SC-15969 rung-4 applicability survey.
 *
 * The survey is hand-curated evidence, so every way it could be WRONG has to fail here rather than
 * generate a plausible matrix. The checks that matter:
 *
 * - **Total coverage.** Every (family, backend) pair the catalog advertises needs a verdict. A
 *   family added to `familyGroup` without a survey entry fails generation instead of quietly
 *   emitting `Missing` rung-4 cells that read as surveyed.
 * - **`Structurally N/A` is never assumed.** An `applicability: "none"` verdict without structural
 *   evidence is rejected — the epic allows a static verdict *because* the evidence is present, and
 *   an empty array would turn that allowance into a bare assertion.
 * - **Implementation claims are per entry and mutually consistent.** `implementation: "none"` with
 *   a non-empty legacy `implementedEntries` or exact `implementationScopes` claim (or the reverse)
 *   is a contradiction, and every named entry has to belong to the family that claims it.
 */
export function parseRung4Survey(body, { familyGroups } = {}) {
  const parsed = JSON.parse(body);
  const families = parsed.families;
  if (!families || typeof families !== "object") {
    throw new Error("rung-4 survey: missing `families`");
  }
  const survey = new Map();
  for (const [group, family] of Object.entries(families)) {
    for (const [backend, verdict] of Object.entries(family.backends ?? {})) {
      const at = `rung-4 survey ${family.name ?? group} (${backend})`;
      if (!RUNG4_APPLICABILITIES.includes(verdict.structuralApplicability)) {
        throw new Error(`${at}: unknown structuralApplicability ${JSON.stringify(verdict.structuralApplicability)}`);
      }
      if (!RUNG4_IMPLEMENTATIONS.includes(verdict.implementation)) {
        throw new Error(`${at}: unknown implementation ${JSON.stringify(verdict.implementation)}`);
      }
      if (!RUNG4_REQUEST_PEAKS.includes(verdict.requestPeak?.finding)) {
        throw new Error(`${at}: unknown requestPeak finding ${JSON.stringify(verdict.requestPeak?.finding)}`);
      }
      for (const [tier, finding] of Object.entries(verdict.requestPeak?.byTier ?? {})) {
        if (!["bf16", "q4", "q8"].includes(tier)) {
          throw new Error(`${at}: requestPeak.byTier contains a tier outside the matrix vocabulary`);
        }
        if (!RUNG4_REQUEST_PEAKS.includes(finding)) {
          throw new Error(`${at}: requestPeak.byTier.${tier} has unknown finding ${JSON.stringify(finding)}`);
        }
      }
      for (const [index, scope] of (verdict.requestPeak?.scopes ?? []).entries()) {
        const scopeAt = `${at}: requestPeak.scopes[${index}]`;
        if (!RUNG4_REQUEST_PEAKS.includes(scope.finding)) {
          throw new Error(`${scopeAt}.finding has unknown finding ${JSON.stringify(scope.finding)}`);
        }
        for (const [field, values, vocabulary] of [
          ["tiers", scope.tiers, ["bf16", "q4", "q8"]],
          ["overlays", scope.overlays, ["none", "lora", "control", "identity"]],
        ]) {
          if (values?.length === 0) {
            throw new Error(`${scopeAt}.${field} is empty — omit it to mean every cell value`);
          }
          if (values?.some((value) => !vocabulary.includes(value))) {
            throw new Error(`${scopeAt}.${field} contains a value outside the matrix vocabulary`);
          }
        }
        if (scope.entries?.length === 0 || scope.modes?.length === 0) {
          throw new Error(`${scopeAt} has an empty selector — omit it to mean every cell value`);
        }
      }
      if (!verdict.evidence?.length) {
        throw new Error(`${at}: a verdict derived from provider code must cite at least one source`);
      }
      if (verdict.structuralApplicability === "none" && !verdict.structural?.length) {
        throw new Error(
          `${at}: applicability "none" becomes a Structurally N/A cell, which the epic accepts only with static provider evidence — none is cited`,
        );
      }
      const implemented = verdict.implementedEntries ?? [];
      const implementationScopes = verdict.implementationScopes ?? [];
      const carriesLegacyImplementationFields = [
        "implementedEntries",
        "implementedModes",
        "implementedTiers",
        "implementedOverlays",
        "strategyParameters",
      ].some((field) => Object.hasOwn(verdict, field));
      if (carriesLegacyImplementationFields && implementationScopes.length) {
        throw new Error(
          `${at}: use either legacy implementedEntries fields or exact implementationScopes, not both`,
        );
      }
      const hasImplementationClaim = implemented.length > 0 || implementationScopes.length > 0;
      if ((verdict.implementation === "none") !== !hasImplementationClaim) {
        throw new Error(
          `${at}: implementation is ${verdict.implementation} but carries ${hasImplementationClaim ? "an" : "no"} implementation claim — the two must agree`,
        );
      }
      if (implemented.length && !Object.keys(verdict.strategyParameters ?? {}).length) {
        throw new Error(`${at}: an implemented family must publish the rung's own strategy parameters`);
      }
      if (verdict.implementedModes && !implemented.length) {
        throw new Error(`${at}: implementedModes narrows implementedEntries, which is empty`);
      }
      if (verdict.implementedModes?.length === 0) {
        throw new Error(`${at}: implementedModes is empty — omit it to mean every mode`);
      }
      for (const [field, values] of [
        ["implementedTiers", verdict.implementedTiers],
        ["implementedOverlays", verdict.implementedOverlays],
      ]) {
        if (values && !implemented.length) {
          throw new Error(`${at}: ${field} narrows implementedEntries, which is empty`);
        }
        if (values?.length === 0) {
          throw new Error(`${at}: ${field} is empty — omit it to mean every cell value`);
        }
      }
      if (verdict.implementedTiers?.some((value) => !["bf16", "q4", "q8"].includes(value))) {
        throw new Error(`${at}: implementedTiers contains a tier outside the matrix vocabulary`);
      }
      if (
        verdict.implementedOverlays?.some(
          (value) => !["none", "lora", "control", "identity"].includes(value),
        )
      ) {
        throw new Error(`${at}: implementedOverlays contains an overlay outside the matrix vocabulary`);
      }
      const selectorOverlaps = (left, right) =>
        left === undefined || right === undefined || left.some((value) => right.includes(value));
      for (const [index, scope] of implementationScopes.entries()) {
        const scopeAt = `${at}: implementationScopes[${index}]`;
        if (!scope.entries?.length) {
          throw new Error(`${scopeAt}.entries must name at least one catalog entry`);
        }
        if (!Object.keys(scope.strategyParameters ?? {}).length) {
          throw new Error(`${scopeAt} must publish the rung's own strategy parameters`);
        }
        for (const [field, values, vocabulary] of [
          ["tiers", scope.tiers, ["bf16", "q4", "q8"]],
          ["overlays", scope.overlays, ["none", "lora", "control", "identity"]],
        ]) {
          if (values?.length === 0) {
            throw new Error(`${scopeAt}.${field} is empty — omit it to mean every cell value`);
          }
          if (values?.some((value) => !vocabulary.includes(value))) {
            throw new Error(`${scopeAt}.${field} contains a value outside the matrix vocabulary`);
          }
        }
        if (scope.modes?.length === 0) {
          throw new Error(`${scopeAt}.modes is empty — omit it to mean every mode`);
        }
        for (let previous = 0; previous < index; previous += 1) {
          const other = implementationScopes[previous];
          if (
            selectorOverlaps(scope.entries, other.entries) &&
            selectorOverlaps(scope.tiers, other.tiers) &&
            selectorOverlaps(scope.modes, other.modes) &&
            selectorOverlaps(scope.overlays, other.overlays)
          ) {
            throw new Error(
              `${scopeAt} overlaps implementationScopes[${previous}], making strategy parameters ambiguous`,
            );
          }
        }
      }
      if (familyGroups) {
        // Both fields name catalog entries, and both are published onto cells, so both are checked.
        // Only `implementedEntries` was at first, and a typo'd or foreign id in `blockStacks[].entries`
        // then rode onto every rung-4 cell of the family as though it were a real per-entry fact.
        const named = [
          ...implemented.map((id) => [id, "implementedEntries"]),
          ...implementationScopes.flatMap((scope, index) =>
            scope.entries.map((id) => [id, `implementationScopes[${index}].entries`]),
          ),
          ...(verdict.requestPeak?.scopes ?? []).flatMap((scope, index) =>
            (scope.entries ?? []).map((id) => [id, `requestPeak.scopes[${index}].entries`]),
          ),
          ...(verdict.blockStacks ?? []).flatMap((stack) =>
            (stack.entries ?? []).map((id) => [id, `blockStacks[${JSON.stringify(stack.name)}].entries`]),
          ),
        ];
        for (const [id, field] of named) {
          // `familyGroups` throws on an id the catalog does not know at all; both that and a
          // real-but-foreign id are the same defect from this field's point of view.
          let owner = null;
          try {
            owner = familyGroups(id);
          } catch {
            owner = null;
          }
          if (owner !== Number(group)) {
            throw new Error(`${at}: ${field} names ${id}, which belongs to another family`);
          }
        }
      }
      if (verdict.overlayIncompatible && !verdict.overlayIncompatible.structural?.length) {
        throw new Error(`${at}: an overlay incompatibility is a Structurally N/A verdict and needs structural evidence`);
      }
      // AC5 as a machine check rather than a convention. A family the primitive cannot express in its
      // current SHAPE is a finding, not an exemption — so it may not be recorded as implemented, and
      // it may not be recorded silently either: without the finding the value degrades to a bare
      // `Missing` cell indistinguishable from "nobody has written it yet".
      if (verdict.structuralApplicability === "requires-different-primitive") {
        if (hasImplementationClaim) {
          throw new Error(
            `${at}: names implemented entries while declaring the primitive's shape insufficient — one of the two is wrong`,
          );
        }
        if (!verdict.findings?.length) {
          throw new Error(
            `${at}: "requires-different-primitive" must state the shape gap as a finding, which is what distinguishes it from an N/A`,
          );
        }
      }
      survey.set(`${group}:${backend}`, verdict);
    }
  }
  return survey;
}

/**
 * The survey verdict as it appears ON a rung-4 cell.
 *
 * Built here rather than inside `strategyStatus` so that EVERY rung-4 cell carries it, including
 * Krea's turboFit cell, whose state is decided by measured phase curves several arms earlier. A
 * field present on 6,000 rung-4 cells and absent on one is the kind of hole a consumer only finds
 * at runtime.
 *
 * The two findings the story asks for stay SEPARATE and both travel with the cell: structural
 * applicability (can this architecture be windowed) and the request-peak finding (does doing so move
 * the number that matters). A cell can be `partial`/`unmeasured`, which is neither "implemented" nor
 * "not applicable" — the state the five-value conformance vocabulary alone cannot express.
 */
function rung4SurveyCell(survey, modelId, backend, tier, mode, overlay, overlayIncompatible) {
  const verdict = survey.get(`${familyGroup(modelId)}:${backend}`);
  if (!verdict) throw new Error(`${modelId}:${backend}: no rung-4 survey verdict (SC-15969)`);
  const scopedRequestPeak = (verdict.requestPeak.scopes ?? []).find(
    (scope) =>
      (scope.entries ?? [modelId]).includes(modelId) &&
      (scope.tiers ?? [tier]).includes(tier) &&
      (scope.modes ?? [mode]).includes(mode) &&
      (scope.overlays ?? [overlay]).includes(overlay),
  );
  return {
    story: 15969,
    // Always the family's OWN verdict. Overlay incompatibility is a property of the provider's
    // adapter mechanism, not of the architecture — Krea's 28-block trunk is windowable whatever its
    // adapters do — so it travels in its own field. Folding it into `structuralApplicability` would
    // publish `none` for a family whose stack is perfectly windowable, and a consumer filtering that
    // field for architecturally-inapplicable families would read those cells as false positives.
    structuralApplicability: verdict.structuralApplicability,
    requestPeak:
      scopedRequestPeak?.finding ?? verdict.requestPeak.byTier?.[tier] ?? verdict.requestPeak.finding,
    implementation: verdict.implementation,
    overlayIncompatible,
    summary: verdict.summary,
    blockStacks: verdict.blockStacks ?? [],
    findings: verdict.findings ?? [],
  };
}

/**
 * Resolve the exact rung-4 implementation claim for one matrix cell.
 *
 * Most providers publish one parameter shape over a Cartesian entry/tier/mode/overlay selector and
 * continue using the legacy fields. Providers whose catalog entries expose different measured
 * parameter shapes use `implementationScopes`; keeping the parameters on each scope prevents a
 * family-wide claim from inventing unsupported cross-products.
 */
function rung4Implementation(verdict, modelId, tier, mode, overlay) {
  const exact = (verdict?.implementationScopes ?? []).find(
    (scope) =>
      scope.entries.includes(modelId) &&
      (scope.tiers ?? [tier]).includes(tier) &&
      (scope.modes ?? [mode]).includes(mode) &&
      (scope.overlays ?? [overlay]).includes(overlay),
  );
  if (exact) return exact.strategyParameters;
  const legacyImplemented =
    (verdict?.implementedEntries ?? []).includes(modelId) &&
    (verdict?.implementedModes ?? [mode]).includes(mode) &&
    (verdict?.implementedTiers ?? [tier]).includes(tier) &&
    (verdict?.implementedOverlays ?? [overlay]).includes(overlay);
  return legacyImplemented ? verdict.strategyParameters : null;
}

/**
 * Coverage runs BOTH ways.
 *
 * Catalog -> survey is the one that matters at generation time: an unsurveyed family would emit
 * `Missing` rung-4 cells that read as having been surveyed and found wanting.
 *
 * Survey -> catalog matters for the survey's own upkeep. `rung4SurveyRows` is derived from the
 * generated cells, so a verdict for a family or backend the catalog no longer advertises simply
 * never appears anywhere — it would sit in the file being maintained, reviewed and trusted while
 * having no effect at all.
 */
export function assertRung4SurveyCoversEveryFamily(survey, models) {
  const advertised = new Set(
    models.flatMap((model) => model.backends.map((backend) => `${familyGroup(model.id)}:${backend}`)),
  );
  for (const key of advertised) {
    if (!survey.has(key)) {
      const [group, backend] = key.split(":");
      throw new Error(
        `family SC-${group} has no ${backend} rung-4 survey verdict, so its bounded_transformer_residency cells would report Missing without ever having been surveyed (SC-15969)`,
      );
    }
  }
  for (const key of survey.keys()) {
    if (!advertised.has(key)) {
      const [group, backend] = key.split(":");
      throw new Error(
        `rung-4 survey: family SC-${group} carries a ${backend} verdict, but the catalog advertises no ${backend} entry in that family — the verdict reaches no cell (SC-15969)`,
      );
    }
  }
}

/**
 * Whether this entry advertises rung 1 on this backend. Rung 4 requires rung 1 engaged in the same
 * request (`gen_core::memory_strategy`'s `BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES`), so the rung-4 arm
 * below reads the prerequisite from the SAME predicate the rung-1 arm uses. Restating it would let
 * the two drift, and the drift is silent: a family that gained rung-1 capability would keep
 * reporting rung 4 as unreachable.
 */
function stagedResidencyIsAvailable({ backend, model, route, sequentialEngines, manifestById }) {
  const declaredModel =
    model.id === "z_image_edit" && route.engine === "z_image_turbo"
      ? manifestById.get("z_image_turbo")
      : model;
  return backend === "mlx"
    ? sequentialEngines.has(route.engine)
    : declaredModel.candle?.supportsSequentialOffload === true ||
        declaredModel.candle?.sequentialPeakGb !== undefined ||
        declaredModel.candle?.turboFit !== undefined;
}

function staticCandleOverlayIsAvailable({ model, route, overlay, manifestById }) {
  // Legacy Candle capability maps were not exhaustive for the base overlay. PuLID is different:
  // its bespoke entry was added with a deliberately closed identity-only contract, so its `none`
  // coordinate must also consult the declaration instead of inheriting the generic base fallback.
  if (model.id !== "pulid_flux_dev" && overlay === "none") return true;
  const declaredModel =
    model.id === "z_image_edit" && route.engine === "z_image_turbo"
      ? manifestById.get("z_image_turbo")
      : model;
  const capabilities = Object.values(declaredModel.candle?.memoryStrategyCapabilities ?? {});
  if (!capabilities.length) return true;
  return capabilities.some((capability) => (capability.overlays ?? ["none"]).includes(overlay));
}

function declaredEvidence(model, backend, tier) {
  const scope = model[backend] ?? {};
  const keys = [
    "minMemoryGb",
    "vramGbByTier",
    "supportsSequentialOffload",
    "memoryStrategyCapabilities",
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

const V1_AUDIT_METHOD = "git object identity across the complete Candle FLUX.2 runtime dependency closure";
const V2_AUDIT_METHOD =
  "compiled artifact identity for changed paths, git object identity for unchanged paths, across " +
  "the complete Candle FLUX.2 runtime dependency closure";

/**
 * sc-17497: accept either audit shape, and require the artifact layer exactly when object identity
 * has stopped being sufficient.
 *
 * v1 records carry no artifact proof, so they remain valid only while every closure object is
 * byte-identical — which is all v1 ever asserted. v2 adds `auditedArtifact`; it becomes MANDATORY
 * the moment a path moves, and it must agree with the digest frozen in `FLUX2_COMPATIBILITY_AUDIT`.
 * Freezing it in source is what stops the checked-in record from authorizing itself.
 *
 * `expected` is injectable so the tests can exercise the ACCEPTING side of the artifact layer while
 * the shipped constant still carries `artifactProof: null`. Without it the only reachable v2 cases
 * would be rejections, and a validator only ever tested against its own default is a false green.
 */
export function validatedInferenceCompatibility(body, expected = FLUX2_COMPATIBILITY_AUDIT) {
  const audit = JSON.parse(body);
  const schemaVersion = audit.schemaVersion;
  if (
    (schemaVersion !== 1 && schemaVersion !== 2) ||
    audit.story !== expected.story ||
    audit.capturedInferenceRevision !== expected.capturedInferenceRevision ||
    audit.compatibleInferenceRevision !== expected.compatibleInferenceRevision ||
    audit.method !== (schemaVersion === 1 ? V1_AUDIT_METHOD : V2_AUDIT_METHOD) ||
    (schemaVersion === 1 && audit.command !== "git rev-parse <revision>:<path>") ||
    typeof audit.command !== "string" ||
    !Array.isArray(audit.auditedObjects)
  ) {
    throw new Error("SC-15833 inference compatibility audit identity is invalid");
  }
  const objects = new Map();
  const changed = [];
  for (const entry of audit.auditedObjects) {
    if (
      !entry ||
      typeof entry.path !== "string" ||
      !/^[0-9a-f]{40}$/.test(entry.capturedObject ?? "") ||
      !/^[0-9a-f]{40}$/.test(entry.compatibleObject ?? "") ||
      objects.has(entry.path)
    ) {
      throw new Error("SC-15833 inference compatibility audit object pair is invalid");
    }
    // The CAPTURED side is what the measurements were taken against, so it is the half that is
    // frozen. A moved compatible object is not a failure here — it is what the artifact layer below
    // exists to adjudicate.
    objects.set(entry.path, entry.capturedObject);
    if (entry.capturedObject !== entry.compatibleObject) changed.push(entry.path);
  }
  if (
    objects.size !== Object.keys(expected.auditedObjects).length ||
    Object.entries(expected.auditedObjects).some(([objectPath, objectId]) => objects.get(objectPath) !== objectId)
  ) {
    throw new Error("SC-15833 inference compatibility audit closure is incomplete or unrecognized");
  }
  if (schemaVersion === 1 && changed.length > 0) {
    throw new Error("SC-15833 inference compatibility audit object pair is invalid");
  }
  if (schemaVersion === 2) {
    validatedCompatibilityArtifact(audit, changed, expected);
  } else if (expected.artifactProof !== null) {
    throw new Error("SC-15833 inference compatibility audit is missing its expected artifact proof");
  }
  return expected;
}

/**
 * The compiled-artifact half of a v2 record.
 *
 * Only a `cuda` build can authorize a FLUX.2 calibration: the measurements come from an RTX capture,
 * and `scripts/inference-artifact-audit.mjs` can also build the same closure on Metal, which must
 * never be mistaken for proof of the CUDA artifact.
 */
function validatedCompatibilityArtifact(audit, changed, expected) {
  const declared = audit.changedClosurePaths;
  const declaredPaths = Array.isArray(declared) ? declared.slice().sort() : null;
  const changedPaths = changed.slice().sort();
  if (
    !declaredPaths ||
    declaredPaths.length !== changedPaths.length ||
    declaredPaths.some((objectPath, index) => objectPath !== changedPaths[index])
  ) {
    throw new Error("SC-15833 inference compatibility audit misdeclares which closure paths changed");
  }
  if (changed.length === 0) {
    if (expected.artifactProof !== null) {
      throw new Error("SC-15833 inference compatibility audit is missing its expected artifact proof");
    }
    return;
  }
  const artifact = audit.auditedArtifact;
  if (
    !artifact ||
    artifact.lane !== "cuda" ||
    artifact.package !== "candle-gen-flux2" ||
    artifact.test !== "tests::flux2_dev_probed_generate_for_offload_ab" ||
    artifact.profile !== "release" ||
    !/^sha256:[0-9a-f]{64}$/.test(artifact.capturedDigest ?? "") ||
    artifact.capturedDigest !== artifact.compatibleDigest ||
    expected.artifactProof === null ||
    artifact.capturedDigest !== expected.artifactProof.digest
  ) {
    throw new Error("SC-15833 inference compatibility audit compiled-artifact proof is invalid");
  }
  // The digest only speaks for what the binary linked. Anything else that moved is unproven.
  const unadjudicable = changed.filter((objectPath) => !expected.artifactProof.adjudicates.includes(objectPath));
  if (unadjudicable.length > 0) {
    throw new Error(
      `SC-15833 inference compatibility audit cannot adjudicate ${unadjudicable.join(", ")}: the audited ` +
        "artifact does not link it",
    );
  }
}

function compatibilityAuthorizes(binding, { modelId, provider, inferenceRevision, audit }) {
  return modelId === audit.modelId &&
    provider === audit.provider &&
    binding.inferenceRevision === audit.capturedInferenceRevision &&
    binding.compatibleInferenceRevision === audit.compatibleInferenceRevision &&
    inferenceRevision === audit.compatibleInferenceRevision;
}

function strategyStatus({
  backend,
  rung,
  route,
  provider,
  sequentialEngines,
  model,
  tier,
  mode,
  overlay,
  rung4Survey,
  manifestById,
  inferenceRevision,
  inferenceCompatibilityAudit,
}) {
  // `z_image_edit` is a catalog alias, not an inference provider. Its MLX jobs resolve to the
  // `z_image_turbo` descriptor and therefore must consume that provider's static contract just as
  // they already inherit its backend scope. Keeping the declaration on the real provider prevents
  // the catalog alias from becoming a second, independently drifting implementation claim.
  const declaredModel =
    model.id === "z_image_edit" && route.engine === "z_image_turbo"
      ? manifestById.get("z_image_turbo")
      : model;
  // A routing-table lane without a per-backend manifest block is real but wholly untriaged: the
  // optional block is evidence/tuning metadata, not lane-existence metadata. Emit the full slice as
  // Missing so epic 15448 can see and own that work instead of either hiding the lane or inferring
  // implementation claims from another backend.
  if (!declaredModel[backend]) {
    return { state: "Missing", source: null, parameters: {} };
  }
  const staticMemoryContract = declaredModel[backend]?.memoryStrategyContract;
  const staticRung4Verdict = rung === "bounded_transformer_residency"
    ? rung4Survey.get(`${familyGroup(model.id)}:${backend}`)
    : null;
  const staticRung4Implementation = rung === "bounded_transformer_residency"
    ? rung4Implementation(staticRung4Verdict, model.id, tier, mode, overlay)
    : null;
  const staticRung4Allowed =
    rung !== "bounded_transformer_residency" ||
    (["full", "partial"].includes(staticRung4Verdict?.structuralApplicability) &&
      staticRung4Implementation !== null &&
      stagedResidencyIsAvailable({ backend, model, route, sequentialEngines, manifestById }));
  const staticImplementation = staticContractCoversProvider(staticMemoryContract, provider)
    ? staticMemoryContract.implementations.find(
        (implementation) =>
          staticRung4Allowed &&
          implementation.rung === rung &&
          implementation.tiers.includes(tier) &&
          implementation.modes.includes(mode) &&
          implementation.overlays.includes(overlay),
      )
    : undefined;
  const allDeclaredCalibrations = (model[backend]?.calibrations ?? []).filter(
    (binding) =>
      binding.provider === provider &&
      binding.tier === tier &&
      binding.mode === mode &&
      matrixOverlayFor(binding.overlay) === overlay &&
      binding.rung === rung &&
      staticRung4Allowed,
  );
  const currentDeclaredCalibrations = allDeclaredCalibrations.filter(
    (binding) => binding.inferenceRevision === inferenceRevision ||
      compatibilityAuthorizes(binding, {
        modelId: model.id,
        provider,
        inferenceRevision,
        audit: inferenceCompatibilityAudit,
      }),
  );
  const calibrationStatus = (bindings, source, evidenceAdmissionCurrent) => {
    const fingerprints = sortedUnique(bindings.map((binding) => binding.fingerprint));
    const parameters = sortedUnique(
      bindings.map((binding) => JSON.stringify(binding.parameters ?? {})),
    );
    if (fingerprints.length !== 1 || parameters.length !== 1) {
      throw new Error(
        `${model.id}:${backend}:${tier}:${mode}:${overlay}:${rung} has inconsistent exact calibration bindings`,
      );
    }
    return {
      state: "Implemented/unverified",
      source,
      parameters: {
        ...(staticImplementation?.parameters ?? {}),
        ...JSON.parse(parameters[0]),
        ...(staticImplementation
          ? { publishedRanges: staticImplementation.parameterRanges }
          : {}),
      },
      calibrationFingerprint: fingerprints[0],
      engagedRungs: staticImplementation?.engagedRungs,
      requiresCurrentCalibrationBinding: true,
      evidenceAdmissionCurrent,
      compatibleCapturedInferenceRevisions: sortedUnique(
        bindings
          .filter((binding) => compatibilityAuthorizes(binding, {
            modelId: model.id,
            provider,
            inferenceRevision,
            audit: inferenceCompatibilityAudit,
          }))
          .map((binding) => binding.inferenceRevision),
      ),
    };
  };
  if (currentDeclaredCalibrations.length) {
    return calibrationStatus(
      currentDeclaredCalibrations,
      "crates/sceneworks-worker/src/mlx_fit_gate.rs#evidence_admission_route",
      true,
    );
  }
  if (staticImplementation) {
    return {
      // This declaration inventories production capability only. Exact runtime evidence must still
      // pass calibrationBinding before the cell can be promoted to Verified.
      state: "Implemented/unverified",
      source: staticImplementation.source,
      parameters: {
        ...staticImplementation.parameters,
        publishedRanges: staticImplementation.parameterRanges,
      },
      calibrationFingerprint: staticImplementation.fingerprint,
      engagedRungs: staticImplementation.engagedRungs,
      requiresCurrentCalibrationBinding: allDeclaredCalibrations.length > 0,
      evidenceAdmissionCurrent: false,
    };
  }
  if (allDeclaredCalibrations.length) {
    return calibrationStatus(
      allDeclaredCalibrations,
      `config/manifests/builtin.models.jsonc#models/${model.id}/${backend}/calibrations`,
      false,
    );
  }
  const staticCapability = declaredModel[backend]?.memoryStrategyCapabilities?.[rung];
  if (staticCapability?.overlays?.includes(overlay)) {
    return {
      state: "Implemented/unverified",
      source: `config/manifests/builtin.models.jsonc#models/${declaredModel.id}/${backend}/memoryStrategyCapabilities/${rung}`,
      parameters: staticCapability.parameters,
    };
  }
  if (
    rung === "resident" &&
    !(model.id === "krea_2_turbo" && backend === "candle" && mode === "text_to_image") &&
    (backend !== "candle" || model.id !== "pulid_flux_dev" ||
      staticCandleOverlayIsAvailable({ model, route, overlay, manifestById }))
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
    (backend !== "candle" ||
      staticCandleOverlayIsAvailable({ model, route, overlay, manifestById })) &&
    stagedResidencyIsAvailable({ backend, model, route, sequentialEngines, manifestById })
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
          evidenceScope: record.evidenceScope,
          runtimeAdmission: record.evidenceScope === "exact_request",
          tier: record.tier,
          geometry: `${record.width}x${record.height}`,
          capturedAt: record.capturedAt,
          harnessVersion: record.harnessVersion,
          engagedRungs: record.measuredCompositions?.[manifestRung],
          observedPeakGb: record.observedPeaksGb?.[manifestRung],
          ...(record.parity ? { parity: record.parity } : {}),
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
  }
  // SC-15969. Everything above answers a cell from measured or manifest-declared evidence; this arm
  // answers the rung-4 cells none of it reaches, from the per-family applicability survey. It is
  // LAST on purpose — Krea's turboFit rung-4 cell keeps its phase-curve parameters and measured
  // evidence rather than being flattened to the survey's structural verdict.
  //
  // The overlay branch used to be a hardcoded Krea special case here. It now comes from the survey,
  // because the fact it encodes is a property of a provider's ADAPTER MECHANISM (fold-at-load vs
  // forward-time residual), not of rung 4: MLX Z-Image streams overlays fine, and generalising
  // Krea's rule to every family would have been wrong in the other direction.
  if (rung === "bounded_transformer_residency") {
    const verdict = rung4Survey.get(`${familyGroup(model.id)}:${backend}`);
    if (!verdict) {
      throw new Error(`${model.id}:${backend}: no rung-4 survey verdict (SC-15969)`);
    }
    // Implementation is per ENTRY and per MODE — inference may route a catalog entry's modes to
    // different descriptors than the one carrying the contract — and the rung is unreachable without
    // its declared rung-1 prerequisite however good the architecture is.
    const implementationParameters = rung4Implementation(
      verdict,
      model.id,
      tier,
      mode,
      overlay,
    );
    const implementedHere =
      implementationParameters !== null &&
      stagedResidencyIsAvailable({ backend, model, route, sequentialEngines, manifestById });
    if (verdict.structuralApplicability === "none") {
      return {
        state: "Structurally N/A",
        source: verdict.structural[0].source,
        parameters: {},
        structural: verdict.structural,
      };
    }
    // Overlay incompatibility exempts only where the streaming path actually EXISTS. On an entry or
    // mode that has no such path, rung 4 is Missing for the ordinary reason, and marking it
    // structurally exempt would presuppose a path that does not exist — and quietly remove the cell
    // from the calibration plan's workload as "no run needed".
    if (implementedHere && overlay !== "none" && verdict.overlayIncompatible) {
      return {
        state: "Structurally N/A",
        source: verdict.overlayIncompatible.structural[0].source,
        parameters: {},
        structural: verdict.overlayIncompatible.structural,
        overlayIncompatible: true,
      };
    }
    if (implementedHere) {
      return {
        state: "Implemented/unverified",
        source: verdict.evidence[0].source,
        parameters: implementationParameters,
      };
    }
    return { state: "Missing", source: null, parameters: {} };
  }
  return { state: "Missing", source: null, parameters: {} };
}

function validateMatrix(
  matrix,
  expectedIds,
  backendTierOverrides,
  rung4Survey,
  cellInventoryExpectations,
) {
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
  assertCellInventoryMatchesCatalog(matrix.cells, cellInventoryExpectations);
  assertTwinCoverage(matrix.models);
  assertCellOwnershipIsBackendScoped(matrix.cells);
  assertRung4SurveyCoversEveryFamily(rung4Survey, matrix.models);
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
    // SC-15969: the survey verdict rides the rung-4 cells and only those. A rung-4 cell without one
    // has escaped the survey; any other rung carrying one means the field has drifted off its rung.
    if ((cell.rung === "bounded_transformer_residency") !== Boolean(cell.rung4Survey)) {
      throw new Error(
        `${cell.id}: rung4Survey must be present on exactly the bounded_transformer_residency cells`,
      );
    }
    if (["Verified", "Runtime verified"].includes(cell.state)) {
      const dynamic = cell.evidence.currentEnvironmentVerification;
      if (!dynamic.length || !cell.calibrationFingerprint) {
        throw new Error(`${cell.id}: unsupported dynamic verification claim`);
      }
      const requiredStatus = cell.state === "Verified" ? "complete" : "runtime_complete";
      if (!dynamic.some((evidence) => evidence.recordStatus === requiredStatus)) {
        throw new Error(`${cell.id}: ${cell.state} lacks a ${requiredStatus} record`);
      }
    }
    // SC-16060. The two claims are independent, and the invariants that keep them from silently
    // merging back into one field belong here rather than in a consumer.
    const characterization = cell.memoryCharacterization;
    const measured = characterization.measuredGeometries.length;
    const expected = measured === 0 ? "unmeasured" : measured === 1 ? "point" : "fitted";
    if (characterization.status !== expected) {
      throw new Error(
        `${cell.id}: memoryCharacterization is ${characterization.status} on ${measured} measured geometr${measured === 1 ? "y" : "ies"}`,
      );
    }
    // A bound without a determinable curve is the exact overclaim this story exists to stop: it
    // would read as "covered up to here" on the strength of a single point.
    if ((characterization.coveredPixelBound !== null) !== (characterization.status === "fitted")) {
      throw new Error(
        `${cell.id}: coveredPixelBound is only meaningful on a fitted curve (status ${characterization.status})`,
      );
    }
    // `Verified` is the implementation claim and must never imply geometry coverage. A cell may be
    // Verified while `unmeasured`/`point` — that is the honest combination. The reverse cannot hold:
    // measured geometry that bound this cell came from a record, so a cell with no implementation
    // cannot have one.
    if (characterization.status !== "unmeasured" && !isImplemented(cell.state)) {
      throw new Error(
        `${cell.id}: ${cell.state} cell carries measured geometry (${characterization.measuredGeometries.join(",")})`,
      );
    }
  }
}

// Every source this document is DERIVED from. Exported (sc-16268) so the tests that prove the
// staleness tripwire covers all of them derive the list from here instead of mirroring it: a
// hand-copied mirror lets a source be dropped from the fingerprint with every test still green,
// which is the quiet-and-stale outcome the tripwire exists to prevent. `generatedFrom.sources` in
// the artifact is generated from this same map, so the published key set is the assertable copy.
export const SOURCE_PATHS = Object.freeze({
  manifest: "config/manifests/builtin.models.jsonc",
  routingCatalog: "crates/sceneworks-core/src/jobs_store/routing/catalog.rs",
  routingCandle: "crates/sceneworks-core/src/jobs_store/routing/candle.rs",
  routingMlx: "crates/sceneworks-core/src/jobs_store/routing/mlx.rs",
  engines: "crates/sceneworks-worker/src/engines.rs",
  imageRouting: "crates/sceneworks-worker/src/image_jobs/base.rs",
  mlxFitGate: "crates/sceneworks-worker/src/mlx_fit_gate.rs",
  memoryStrategy: "crates/sceneworks-worker/src/memory_strategy.rs",
  vramGate: "crates/sceneworks-worker/src/vram_gate.rs",
  instantId: "crates/sceneworks-worker/src/image_jobs/instantid.rs",
  calibrationEvidence: "docs/generated/memory-calibration-evidence.json",
  calibrationPlan: "config/memory-calibration-plan.json",
  inferenceCompatibility: "docs/calibration/sc-15833/inference-compatibility-277f.json",
  rung4Survey: "config/rung4-applicability-survey.json",
  cargo: "Cargo.toml",
});

// `matrixSourceRevision` is generated provenance written back into the manifest's calibration
// bindings. Including that value in the source-tree hash creates an impossible fixed point:
// regenerating the matrix rotates the value, certification writes the new value into the
// manifest, and the next regeneration rotates it again. Keep every binding field that affects
// eligibility in the semantic hash, but replace only this self-stamped provenance value.
function manifestRevisionBody(body) {
  const parsed = JSON.parse(stripJsoncComments(body));
  const visit = (value) => {
    if (Array.isArray(value)) return value.map(visit);
    if (value === null || typeof value !== "object") return value;
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [
        key,
        key === "matrixSourceRevision" ? "source-tree:<generated>" : visit(child),
      ]),
    );
  };
  return JSON.stringify(visit(parsed));
}

export async function buildMatrix({ sourceOverrides = {}, cellFilter = null } = {}) {
  const sourcePaths = SOURCE_PATHS;
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
  const inferenceCompatibilityAudit = validatedInferenceCompatibility(
    bodies.inferenceCompatibility,
  );
  const rung4Survey = parseRung4Survey(bodies.rung4Survey, { familyGroups: familyGroup });
  const manifest = JSON.parse(stripJsoncComments(manifestBody));
  // Comments and formatting are not part of any of these sources' contracts. Hash each source's
  // SEMANTIC body — parsed value for JSON/JSONC, inert whole-line comments removed for Rust and
  // TOML — so provenance is stable across semantically inert edits (sc-16129 did the manifest;
  // sc-16268 the rest). Parsing below still reads the raw `bodies`; only provenance reads these.
  const revisionBodies = Object.fromEntries(
    sourceEntries.map(([name, relative]) => [
      name,
      name === "manifest"
        ? manifestRevisionBody(bodies[name])
        : semanticSourceBody(relative, bodies[name]),
    ]),
  );
  const images = manifest.models.filter((model) => model.type === "image");
  const manifestById = new Map(images.map((model) => [model.id, model]));
  const expectedIds = parseExpectedImageIds(enginesBody);
  const routes = parseEngineRoutes(enginesBody);
  const routedBackends = routedLanes({
    routingCatalog: bodies.routingCatalog,
    routingCandle: bodies.routingCandle,
    routingMlx: bodies.routingMlx,
  });
  const sequentialEngines = parseMlxSequentialEngines(mlxFitBody);
  const backendTierOverrides = parseBackendTierOverrides(bodies.instantId);
  const pin = inferencePin(cargoBody);
  // NUL-separated (sc-16268): normalisation strips each body's trailing newline, so concatenating
  // bare would let content shift across a source boundary without moving the hash. A NUL cannot
  // occur in any of these text sources, so it is an unambiguous delimiter.
  const sceneWorksRevision = `source-tree:${sha256(
    sourceEntries
      .filter(([name]) => name !== "calibrationEvidence")
      .map(([name]) => revisionBodies[name])
      .join("\0"),
  )}`;

  // sc-16073: no advertised route without cells, and no orphaned control measurements. The worker's MLX
  // and Candle declarations are checked independently rather than trusting their documented twin set.
  assertDeclaredControlLanes(images, bodies.imageRouting);

  const models = images
    .map((model) => {
      const route = routes.get(model.id);
      if (!route) throw new Error(`${model.id}: no resolved route/provider`);
      const backends = backendScopes(model, routedBackends);
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

  // Resolve the complete catalog expectation in its own pass. Keeping this outside the emission loop
  // is load-bearing: if a later regression skips an entire model/backend branch while emitting cells,
  // the expected scope must survive so validation can report the loss.
  const cellInventoryExpectations = new Map();
  for (const modelSummary of models) {
    const model = manifestById.get(modelSummary.id);
    for (const backend of modelSummary.backends) {
      const tiers = tiersFor(model, backend, backendTierOverrides);
      const modes = modesFor(model);
      const overlays = overlaysFor(model, backend);
      cellInventoryExpectations.set(`${model.id}:${backend}`, {
        tiers: tiers.length,
        modes: modes.length,
        overlays: overlays.length,
        rungs: RUNGS.length,
        cells: tiers.length * modes.length * overlays.length * RUNGS.length,
      });
    }
  }

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
                rung4Survey,
                manifestById,
                inferenceRevision: pin,
                inferenceCompatibilityAudit,
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
                provider,
                backend,
                tier,
                mode,
                overlay,
                rung,
                status,
                calibrationPlan,
              });
              const runSummary = (record) => {
                const overall = observedPeakBytes(record);
                const requiredHostBytes = mlxRequiredHostBytes(record);
                return {
                  source: `docs/generated/memory-calibration-evidence.json#${record.id}`,
                  hardware: record.backend === "candle" ? record.hardware.name : record.hardware.chip,
                  tier: record.target.tier,
                  geometry: `${record.target.geometry.width}x${record.target.geometry.height}`,
                  capturedAt: record.capturedAt,
                  harnessVersion: record.harnessVersion,
                  recordStatus: record.status,
                  engagedRungs: record.strategy.engagedRungs,
                  ...(Number.isFinite(overall) ? { observedPeakGb: overall / 1024 ** 3 } : {}),
                  ...(requiredHostBytes !== null ? { requiredHostBytes } : {}),
                  parity: {
                    contract: ["exact", "tolerance", "golden"].includes(record.quality.contract)
                      ? record.quality.contract
                      : record.quality.maximumErrorThreshold === 0 &&
                          record.quality.meanErrorThreshold === 0
                        ? "exact"
                        : "tolerance",
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
                    compatibleCapturedInferenceRevisions:
                      status.compatibleCapturedInferenceRevisions,
                  }) === "historical",
              );
              const currentRuns = eligibleRuns.filter(
                (record) =>
                  evidenceSemantics(record, {
                    sceneWorks: sceneWorksRevision,
                    inference: pin,
                    compatibleCapturedInferenceRevisions:
                      status.compatibleCapturedInferenceRevisions,
                  }) === "current" && ["complete", "runtime_complete"].includes(record.status),
              );
              const currentFullRuns = currentRuns.filter((record) => record.status === "complete");
              const currentRuntimeRuns = currentRuns.filter(
                (record) => record.status === "runtime_complete",
              );
              // SC-16060. Characterization counts every geometry bound to this cell — the
              // manifest-declared captures AND the eligible bundle records — because
              // `calibrationBinding` has already gated on the calibration fingerprint, which
              // `memory-calibration-harness.mjs#evidenceSemantics` names as the invalidation switch
              // that owns SceneWorks drift. The historical/current split is a revision distinction,
              // and staling a measured slope on an unrelated source edit is the failure that comment
              // exists to prevent. Promotion to `Verified` is stricter and does require `current`.
              const characterization = memoryCharacterization([
                ...(status.historicalVerification ?? []).map((row) => row.geometry),
                ...eligibleRuns.map(
                  (record) => `${record.target.geometry.width}x${record.target.geometry.height}`,
                ),
              ]);
              // SC-16060. The producer the vocabulary never had: `Verified` was a listed state with
              // nothing able to emit it, so the guard in `validateMatrix` was unreachable and a test
              // asserting zero of them was green for the trivial reason. Promotion is from
              // `Implemented/unverified` ONLY — `Missing` has no implementation to verify and
              // `Structurally N/A` has nothing to measure, so neither may be lifted by evidence.
              const state =
                status.state === "Implemented/unverified" &&
                  currentFullRuns.length > 0 &&
                  (!status.requiresCurrentCalibrationBinding || status.evidenceAdmissionCurrent)
                  ? "Verified"
                  : status.state === "Implemented/unverified" &&
                      currentRuntimeRuns.length > 0 &&
                      (!status.requiresCurrentCalibrationBinding || status.evidenceAdmissionCurrent)
                    ? "Runtime verified"
                  : status.state;
              const cell = {
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
                state,
                memoryCharacterization: characterization,
                calibrationFingerprint: fingerprint,
                owningFamilyStory,
                owningModelStory,
                ...(rung === "bounded_transformer_residency"
                  ? {
                      rung4Survey: rung4SurveyCell(
                        rung4Survey,
                        model.id,
                        backend,
                        tier,
                        mode,
                        overlay,
                        status.overlayIncompatible === true,
                      ),
                    }
                  : {}),
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
              };
              // Mutation seam used by the inventory regression test. The CLI never supplies a filter;
              // every production build emits the full catalog and validates it below before writing.
              if (!cellFilter || cellFilter(cell)) cells.push(cell);
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
      semantics: cell.evidence.currentEnvironmentVerification.some(
        (evidence) => evidence.source === `docs/generated/memory-calibration-evidence.json#${record.id}`,
      )
        ? "current"
        : evidenceSemantics(record, {
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
  // SC-15969. One row per surveyed (family, backend), derived from the cells rather than re-read from
  // the survey file — so a verdict that never reached a cell cannot appear in the summary as if it had.
  const requestPeakPriority = new Map([
    ["unmeasured", 0],
    ["does-not-move", 1],
    ["moves", 2],
  ]);
  const rung4SurveyRowsByFamily = new Map();
  for (const cell of cells.filter(
    (candidate) =>
      candidate.rung === "bounded_transformer_residency" && candidate.overlay === "none",
  )) {
    const key = `${familyGroup(cell.modelId)}:${cell.backend}`;
    const existing = rung4SurveyRowsByFamily.get(key);
    const requestPeak =
      !existing ||
      requestPeakPriority.get(cell.rung4Survey.requestPeak) >
        requestPeakPriority.get(existing.requestPeak)
        ? cell.rung4Survey.requestPeak
        : existing.requestPeak;
    rung4SurveyRowsByFamily.set(key, {
      familyStory: familyGroup(cell.modelId),
      backend: cell.backend,
      structuralApplicability: cell.rung4Survey.structuralApplicability,
      requestPeak,
      implementation: cell.rung4Survey.implementation,
    });
  }
  const rung4SurveyRows = [...rung4SurveyRowsByFamily.values()].sort((left, right) =>
    `${left.familyStory}:${left.backend}`.localeCompare(`${right.familyStory}:${right.backend}`),
  );
  const tally = (rows, key) =>
    Object.fromEntries(
      sortedUnique(rows.map((row) => row[key])).map((value) => [
        value,
        rows.filter((row) => row[key] === value).length,
      ]),
    );
  const mlxStagedModels = new Set(
    cells
      .filter(
        (cell) =>
          cell.backend === "mlx" &&
          cell.rung === "staged_residency" &&
          isImplemented(cell.state),
      )
      .map((cell) => cell.modelId),
  );
  const matrix = {
    // 2 (SC-15812): `models[].owningFamilyStory`/`owningModelStory` were both RENAMED (now plural)
    // and RETYPED (integer -> backend->id object). A reader written against 1 gets `undefined` for
    // both, so the two shapes cannot share a version number — that is the whole job of this field.
    //
    // 5 (SC-16060): `claims`, `memoryCharacterizationStates`, and `cells[].memoryCharacterization`
    // were ADDED and are REQUIRED, and `conformanceStates` changed SHAPE — from bare strings to
    // `{state, definition}` objects. That last one is not additive: a version-4 reader indexing
    // `conformanceStates` for a string gets an object. It is also the point of the change — the
    // states carried no definitions, which is how the pipeline came to hold two contradictory
    // answers to whether one measured geometry certifies a whole envelope.
    //
    // 4 (SC-15969): `rung4SurveyRows` and `cells[].rung4Survey` were ADDED, and both are REQUIRED —
    // the first at the document root, the second on every rung-4 cell. A version-3 document has
    // neither, so it no longer validates against the schema that describes this one. That is the
    // test: not "does an old reader break" (it does not — the fields are additive), but "is a
    // document of the old shape still a document of this shape".
    //
    // 3 (sc-16268): `cells[].evidenceRevision` was REMOVED. It stamped the same two constants into
    // every one of the ~7,360 rows — one distinct value, never conditional on the row's evidence
    // despite the name — so a fingerprint rotation rewrote ~14,700 lines and made any two
    // concurrent PRs touching a fingerprinted source conflict in a file that cannot be
    // hand-merged (only regenerated). The values survive verbatim in `generatedFrom`
    // (`sceneWorksRevision` / `inferenceRevision`), which is the copy the only real consumer
    // (`scripts/memory-calibration-harness.mjs`) has always read. A reader written against 2 that
    // dereferences `cell.evidenceRevision.sceneWorks` now throws, so this takes a new version.
    schemaVersion: 6,
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
    // SC-16060. These carried no definitions, so every consumer supplied its own — which is how the
    // pipeline came to hold two contradictory answers to "does one measured geometry certify the
    // envelope?". The claim each state belongs to is now named here, in the artifact, rather than
    // being inferred from whichever binding rule a reader happened to find first.
    claims: {
      state: {
        asserts: "the rung WORKS: implemented, parity-passed, engaging the claimed composition",
        geometrySensitive: false,
        binding:
          "scripts/generate-memory-matrix.mjs#calibrationBinding — envelope membership, because " +
          "whether a rung executes does not depend on the resolution it executed at",
      },
      memoryCharacterization: {
        asserts: "the rung's PEAKS are known across the geometry envelope",
        geometrySensitive: true,
        binding:
          "scripts/generate-memory-matrix.mjs#memoryCharacterization — distinct measured " +
          "geometries, because `fixedGb + perMpxGb * megapixels` has two coefficients and one " +
          "point cannot determine a slope",
      },
    },
    conformanceStates: [
      {
        state: "Verified",
        definition:
          "Implemented AND carrying at least one eligible current-environment calibration record. " +
          "Says nothing about geometry coverage — read `memoryCharacterization` for that.",
      },
      {
        state: "Runtime verified",
        definition:
          "Implemented and production-admissible for an exact base-only coordinate through a " +
          "current runtime_complete record. This is intentionally below Full Verified: lifecycle " +
          "recovery and measured negative-mutation coverage remain owned by the catalog story.",
      },
      {
        state: "Implemented/unverified",
        definition: "The code path exists and is statically evidenced; no current measurement binds it.",
      },
      {
        state: "Structurally N/A",
        definition: "The rung cannot apply to this architecture; there is nothing to measure.",
      },
      { state: "Missing", definition: "No implementation of this rung on this route." },
      { state: "Route unavailable/broken", definition: "The route itself does not resolve." },
    ],
    memoryCharacterizationStates: [
      {
        status: "unmeasured",
        definition: "No geometry has been measured for this cell.",
      },
      {
        status: "point",
        definition:
          "Exactly one measured geometry. The peak is known AT that geometry and the slope is " +
          "undeterminable, so nothing is known about the rest of the envelope.",
      },
      {
        status: "fitted",
        definition:
          "Two or more distinct measured geometries — sufficient to determine the affine curve, " +
          "which is not a claim that a fit has been performed. `coveredPixelBound` is the largest " +
          "measured area.",
      },
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
      calibrationRunsByStatus: {
        complete: calibrationBundle.records.filter((record) => record.status === "complete").length,
        runtimeComplete: calibrationBundle.records.filter(
          (record) => record.status === "runtime_complete",
        ).length,
      },
      currentCalibrationRuns: cells.reduce(
        (count, cell) => count + cell.evidence.currentEnvironmentVerification.length,
        0,
      ),
      rung4Survey: {
        story: 15969,
        surveyedFamilyBackends: rung4SurveyRows.length,
        structuralApplicability: tally(rung4SurveyRows, "structuralApplicability"),
        requestPeak: tally(rung4SurveyRows, "requestPeak"),
        implementation: tally(rung4SurveyRows, "implementation"),
      },
    },
    models,
    rung4SurveyRows,
    cells,
    calibrationRuns,
    modelSlices,
  };
  validateMatrix(
    matrix,
    expectedIds,
    backendTierOverrides,
    rung4Survey,
    cellInventoryExpectations,
  );
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
    `- Full complete calibration records: ${matrix.summary.calibrationRunsByStatus.complete}`,
    `- Base-only runtime-complete calibration records: ${matrix.summary.calibrationRunsByStatus.runtimeComplete}`,
    "",
    "Static capability is never promoted to dynamic verification. Generated cells contain separate declared, historical, current-environment, loadability, and strategy-parameter evidence arrays.",
    "`Runtime verified` means the exact base-only coordinate is production-admissible from current runtime evidence; it is deliberately not Full `Verified`, which additionally requires the catalog story's lifecycle and negative-mutation signoff.",
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
          isImplemented(cell.state),
      );
      const stagedCells = matrix.cells.filter(
        (cell) =>
          cell.modelId === model.id &&
          cell.backend === backend &&
          cell.rung === "staged_residency",
      );
      const stagedState = stagedCells.some((cell) => cell.state === "Verified")
        ? "Verified"
        : stagedCells.some((cell) => cell.state === "Runtime verified")
          ? "Runtime verified"
          : "Implemented/unverified";
      lines.push(
        `| \`${model.id}\` | ${backend} | \`${model.resolvedRoute}\` (${model.routeKind}) | SC-${model.owningFamilyStories[backend]} | SC-${model.owningModelStories[backend]} | ${staged ? stagedState : "Missing"} |`,
      );
    }
  }
  lines.push(
    "",
    "Per-model consumers must use `modelSlices` in the JSON artifact. A cell is Full only when every applicable rung is Verified or Structurally N/A; this static baseline intentionally reports zero Full models.",
    "",
    "## Rung 4 — per-family applicability survey (SC-15969)",
    "",
    "Source: `config/rung4-applicability-survey.json`, derived from the pinned inference revision's provider code. The two findings are deliberately separate: **can** the architecture be windowed, and **does** doing so move the request peak. A family can be structurally capable and still correctly default to not using the rung.",
    "",
    "`partial` means windowable over a sub-stack but not the whole trunk — neither Implemented nor Structurally N/A, and recorded rather than rounded to either.",
    "",
    "| Family story | Backend | Structural applicability | Implementation | Request peak |",
    "| --- | --- | --- | --- | --- |",
  );
  for (const row of matrix.rung4SurveyRows) {
    lines.push(
      `| SC-${row.familyStory} | ${row.backend} | ${row.structuralApplicability} | ${row.implementation} | ${row.requestPeak} |`,
    );
  }
  lines.push(
    "",
    `Surveyed family/backend pairs: ${matrix.summary.rung4Survey.surveyedFamilyBackends}. Per-cell verdicts, block-stack inventories and findings are in \`cells[].rung4Survey\` in the JSON artifact.`,
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
