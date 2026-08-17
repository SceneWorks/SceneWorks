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
// SC-18218 removed FLUX.2-dev from the MLX staged-residency census: the pinned MLX provider is
// eager/resident-only, so counting its generic route as staged coverage would contradict the captured
// contract. Bernini is the mirror case — inference sc-18609 made its DECLARED MLX rung-4 ladder
// actually reachable on both variants, so it belongs in the census.
//
// Neither fact is a total. This census used to be pinned to an exact population, which meant hand-
// renewing 37 -> 38 for a reachability change that had nothing to do with the contract being guarded,
// and the number never said which entry moved. `assertMlxStagedCoverageIsStructurallyConsistent`
// replaces it: the same defect (a lane claiming staged coverage it has not implemented) is caught by
// the two named per-model contracts, closure of the verdict under the resolved route, and a strictly
// partial census — all of which hold at any catalog size and name the entry that drifted.
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

const GENERATION_CAPABILITIES = new Set([
  "text_to_image",
  "edit_image",
  "image_to_image",
  "image_inpaint",
  "image_detail",
  "character_image",
]);

export function activeCalibrationPlan(calibrationPlan) {
  const retiredModes = new Set(Object.keys(calibrationPlan.retiredModes ?? {}));
  return {
    ...calibrationPlan,
    providers: calibrationPlan.providers.filter((entry) => !retiredModes.has(entry.target.mode)),
  };
}

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

// The MLX staged-residency census, checked as structure instead of an exact population (see the note
// on EXPECTED_IMAGE_COUNT above). Runs inside `validateMatrix`, so `cells` is still the full resolved
// cross-product and `coverage` is not populated yet; the census-versus-published cross-check belongs to
// tests/test_memory_matrix.py, which reads the artifact after the publication slim.
export function assertMlxStagedCoverageIsStructurallyConsistent(matrix) {
  const staged = new Set(
    matrix.cells
      .filter(
        (cell) =>
          cell.backend === "mlx" && cell.rung === "staged_residency" && isImplemented(cell.state),
      )
      .map((cell) => cell.modelId),
  );
  if (staged.has("flux2_dev")) {
    throw new Error(
      "flux2_dev claims MLX staged coverage, but SC-18218 measured the pinned provider as Resident-only",
    );
  }
  if (!staged.has("bernini_image")) {
    throw new Error(
      "bernini_image lost its MLX staged coverage; inference sc-18609 made its declared rung-4 ladder reachable",
    );
  }
  if (staged.size === 0 || staged.size >= matrix.models.length) {
    throw new Error(
      `MLX staged coverage is partial by construction, found ${staged.size}/${matrix.models.length}`,
    );
  }
  // The verdict is a property of the RESOLVED ROUTE, so entries sharing a route must agree. An entry
  // drifting away from its own siblings is exactly what a pinned total could not see.
  const byRoute = new Map();
  for (const model of matrix.models) {
    const verdicts = byRoute.get(model.resolvedRoute) ?? new Set();
    verdicts.add(staged.has(model.id));
    byRoute.set(model.resolvedRoute, verdicts);
  }
  const split = [...byRoute.entries()].filter(([, verdicts]) => verdicts.size > 1).map(([route]) => route);
  if (split.length) {
    throw new Error(`MLX staged coverage disagrees within resolved route(s) ${split.sort().join(",")}`);
  }
  // A bespoke route carries its own pipeline and never advertises the generic staged ladder.
  const bespoke = matrix.models
    .filter((model) => model.routeKind === "bespoke" && staged.has(model.id))
    .map((model) => model.id);
  if (bespoke.length) {
    throw new Error(`bespoke route(s) ${bespoke.sort().join(",")} claim generic MLX staged coverage`);
  }
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

/**
 * Does one `config/memory-calibration-plan.json` entry target exactly this matrix coordinate?
 *
 * ONE matcher, consumed by both the engaged-composition lookup below and sc-18099's publication
 * predicate. The predicate publishes a cell BECAUSE it is planned, so "planned" has to mean the set
 * the plan actually addresses. A second filter that merely resembled this one would publish — or
 * silently elide — a different population with every test still green, which is the derived-constant
 * failure class this subsystem has already been bitten by.
 *
 * `coordinate` is deliberately the shape a generated cell already has (`modelId`, `provider`,
 * `backend`, `tier`, `mode`, `overlay`, `rung`), so a cell can be passed straight in.
 */
export function planEntryTargetsCoordinate(entry, coordinate) {
  return (
    entry.target.modelId === coordinate.modelId &&
    entry.target.provider === coordinate.provider &&
    entry.backend === coordinate.backend &&
    entry.target.tier === coordinate.tier &&
    entry.target.mode === coordinate.mode &&
    matrixOverlayFor(entry.target.overlay) === coordinate.overlay &&
    entry.rung === coordinate.rung
  );
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
    .filter((candidate) =>
      planEntryTargetsCoordinate(candidate, {
        modelId: model.id,
        provider,
        backend,
        tier,
        mode,
        overlay,
        rung,
      }),
    )
    .map((candidate) => candidate.engagedRungs);
  if (matches.length === 0) return null;
  if (matches.some((candidate) => JSON.stringify(candidate) !== JSON.stringify(matches[0]))) {
    throw new Error(`${model.id}:${provider}:${backend}:${tier}:${mode}:${overlay}:${rung}: conflicting planned compositions`);
  }
  return matches[0];
}

// Derive the smallest host that satisfies the same piecewise reserve policy as the Rust MLX
// admission envelope. Below the capture host, solve
// `peak + ceil(reserve * host / capture) <= host`; at and above it preserve the captured absolute
// reserve. BigInt keeps the peak*host intermediate exact even though every published byte value
// must remain a JSON-safe integer.
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

  const captureHost = BigInt(memoryBytes);
  if (captureHost === 0n) return null;
  const processCeiling = BigInt(Math.min(memoryBytes, mlxLimit, wiredLimit));
  const foreignReserve = captureHost - processCeiling;
  const nonReclaimableWired = BigInt(Math.max(0, wired - reclaimable));
  const peak = BigInt(predicted) > nonReclaimableWired ? BigInt(predicted) : nonReclaimableWired;
  const absoluteRequirement = peak + foreignReserve;

  let required = absoluteRequirement;
  if (foreignReserve < captureHost) {
    const denominator = captureHost - foreignReserve;
    const proportionalRequirement = (peak * captureHost + denominator - 1n) / denominator;
    if (proportionalRequirement <= captureHost) required = proportionalRequirement;
  }

  return required <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(required) : null;
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
    // sc-18099: `summary`, `blockStacks` and `findings` moved to `rung4SurveyRows`. They are
    // constants of the (family, backend) pair, so restating them per cell was 2.72 MB of pure
    // duplication AND left them unreachable for any family the slim publishes no cell from. What
    // stays here is what genuinely varies per coordinate: the request-peak scope resolution and the
    // overlay-incompatibility verdict.
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
    "memoryStrategyStructuralExemptions",
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


/**
 * The live per-provider compile-closure digests, gated against the Cargo pin (sc-17774).
 *
 * This REPLACES `compatibilityAuthorizes`, which was the only escape from pin-identity invalidation
 * and was hardcoded to a single frozen `flux2_dev` audit object carrying one hand-verified
 * `(captured -> compatible)` revision pair. It authorized exactly one target revision for exactly
 * one provider, so it was spent the moment the pin moved one commit further, and it generalised to
 * nothing. Every provider now gets the same relief from a derived digest, with no hand audit.
 *
 * The config is derived offline so a reviewer sees a digest change in the diff rather than having it
 * conjured at check time. That makes a stale config the obvious failure mode, so it is a hard error
 * rather than a fallback: a config keyed to an older pin would report currency for closures nobody
 * re-derived. Whether the digests are REAL is a separate question, graded in CI — `check.yml`
 * re-derives them against a shallow fetch of the pinned inference revision, which is possible
 * because SceneWorks/inference is public.
 */
export function validatedInferenceClosures(body, pin) {
  const closures = JSON.parse(body);
  if (closures.inferenceRevision !== pin) {
    throw new Error(
      `${"config/inference-provider-closures.json"} is keyed to ` +
        `${closures.inferenceRevision?.slice(0, 8) ?? "(unset)"} but Cargo pins ${pin.slice(0, 8)}. ` +
        "Re-run: node scripts/inference-closure-digest.mjs --repo <inference> --write",
    );
  }
  const digests = new Map();
  for (const [provider, entry] of Object.entries(closures.providers ?? {})) {
    if (!/^[0-9a-f]{64}$/.test(entry.digest ?? "")) {
      throw new Error(`inference closure entry for ${provider} has no usable digest`);
    }
    digests.set(provider, entry.digest);
  }
  if (!digests.size) throw new Error("config/inference-provider-closures.json declares no providers");
  return digests;
}

/**
 * A calibration binding is current when THIS PROVIDER'S closure digest is unchanged — never when the
 * inference pin happens to match. `binding.inferenceRevision` stays as capture provenance.
 */
function closureIsCurrent(binding, { backend, provider, inferenceClosureDigests }) {
  const live = inferenceClosureDigests.get(`${backend}:${provider}`);
  if (!live) {
    throw new Error(
      `provider "${provider}" has a bound calibration but no entry in ` +
        "config/inference-provider-closures.json. Declare its inference crate and regenerate.",
    );
  }
  if (!binding.inferenceClosureDigest) {
    throw new Error(
      `a ${provider} calibration binding carries no inferenceClosureDigest. Run ` +
        "node scripts/backfill-closure-digests.mjs --repo <inference> --write",
    );
  }
  return binding.inferenceClosureDigest === live;
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
  inferenceClosureDigests,
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
  const staticContractIsExhaustive =
    staticMemoryContract?.exhaustive === true &&
    staticContractCoversProvider(staticMemoryContract, provider);
  const allDeclaredCalibrations = (model[backend]?.calibrations ?? []).filter(
    (binding) =>
      binding.provider === provider &&
      binding.tier === tier &&
      binding.mode === mode &&
      matrixOverlayFor(binding.overlay) === overlay &&
      binding.rung === rung &&
      staticRung4Allowed,
  );
  if (staticContractIsExhaustive && !staticImplementation) {
    if (allDeclaredCalibrations.length) {
      throw new Error(
        `${model.id}:${backend}:${tier}:${mode}:${overlay}:${rung} declares calibration ` +
          `outside exhaustive provider contract ${staticMemoryContract.provider}`,
      );
    }
    return { state: "Missing", source: null, parameters: {} };
  }
  const currentDeclaredCalibrations = allDeclaredCalibrations.filter((binding) =>
    closureIsCurrent(binding, { backend, provider, inferenceClosureDigests }),
  );
  // Semantic quality receipts authorize exact geometry choices in the runtime planner. They are
  // not numeric tuning ranges and must never be projected into the published calibration matrix,
  // where their presence could be mistaken for measured memory evidence.
  const publishableParameterRanges = (implementation) => {
    const { decodeGeometryPolicies: _semanticReceipt, ...publishedRanges } =
      implementation?.parameterRanges ?? {};
    return publishedRanges;
  };
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
          ? { publishedRanges: publishableParameterRanges(staticImplementation) }
          : {}),
      },
      calibrationFingerprint: fingerprints[0],
      engagedRungs: staticImplementation?.engagedRungs,
      requiresCurrentCalibrationBinding: true,
      evidenceAdmissionCurrent,
      // sc-17774: the record-side currency term. It replaces
      // `compatibleCapturedInferenceRevisions`, which listed the revisions one frozen `flux2_dev`
      // audit had hand-authorized; currency is now decided per provider by this digest.
      inferenceClosureDigest: inferenceClosureDigests.get(`${backend}:${provider}`),
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
        publishedRanges: publishableParameterRanges(staticImplementation),
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
  const staticExemption =
    declaredModel[backend]?.memoryStrategyStructuralExemptions?.[rung];
  if (staticExemption?.overlays?.includes(overlay)) {
    return {
      state: "Structurally N/A",
      source: staticExemption.evidence[0].source,
      parameters: {},
      structural: staticExemption.evidence,
    };
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
  calibrationPlan,
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
  assertMlxStagedCoverageIsStructurallyConsistent(matrix);
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
  assertCalibrationPlanTargetsResolvedCoordinates(calibrationPlan, matrix.cells);
  assertTwinCoverage(matrix.models);
  assertCellOwnershipIsBackendScoped(matrix.cells);
  assertRung4SurveyCoversEveryFamily(rung4Survey, matrix.models);
  for (const model of matrix.models) {
    for (const map of ["owningFamilyStories", "owningModelStories", "axes"]) {
      const owned = Object.keys(model[map]).sort();
      if (JSON.stringify(owned) !== JSON.stringify([...model.backends].sort())) {
        throw new Error(
          `${model.id}: ${map} covers ${owned.join(",") || "nothing"} but the entry advertises ${model.backends.join(",")}`,
        );
      }
    }
    // sc-18099: the published axes must BE the cross-product the inventory guard resolved, not a
    // parallel description of it. Elided coordinates are reconstructable only from these lists, so a
    // list that disagreed with the resolved scope would publish a lane inventory nothing generated.
    for (const [backend, axes] of Object.entries(model.axes)) {
      const expected = cellInventoryExpectations.get(`${model.id}:${backend}`);
      const resolved = axes.tiers.length * axes.modes.length * axes.overlays.length * axes.rungs.length;
      if (!expected || resolved !== expected.cells) {
        throw new Error(
          `${model.id}:${backend}: published axes resolve to ${resolved} coordinates but the catalog cross-product is ${expected?.cells ?? "unresolved"}`,
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

// ── sc-18099: publication ──────────────────────────────────────────────────────────────────────
//
// The generator still RESOLVES the whole catalog cross-product and still validates it — every guard
// above keeps its full reach, `assertCellInventoryMatchesCatalog` included. What changed is what gets
// WRITTEN. 9,140 coordinates at ~2.5 KB apiece produced a 22 MB committed artifact that no two PRs
// could merge without regenerating, and ~98% of it said nothing: `Missing`, `unmeasured`, unplanned,
// no evidence of any kind. The runtime never reads this file; it is a report, and a report that
// restates the same absence 8,967 times is not more honest than one that counts it.
//
// So the elision is COUNTED, not hidden. `summary.elidedCells` and the per-(entry, backend, rung)
// `coverage` census below are derived from the FULL resolved set, so every coverage claim the old
// artifact could support is still answerable — including `mlxStagedStaticCoverage`, which is still
// computed over all 9,140 coordinates with `isImplemented()`. What a coordinate's mere EXISTENCE
// claimed is preserved separately, in `models[].axes`.

/**
 * Why a cell earns a published row. Stated here, published in `summary.publicationPredicate`, and
 * implemented by `isPublishableCell` — one wording, so a reader of the artifact can tell exactly what
 * an absent coordinate means.
 */
export const PUBLICATION_PREDICATE =
  "A coordinate is published when it is PLANNED (an entry in config/memory-calibration-plan.json " +
  "targets it), MEASURED (memoryCharacterization is not `unmeasured`), BOUND to a calibration record " +
  "in docs/generated/memory-calibration-evidence.json, or CITES evidence of its own (historical, " +
  "current-environment, strategy-parameter, or structural). Every elided coordinate is therefore " +
  "unplanned, unmeasured, unbound and uncited; its `state` and its per-rung population are counted " +
  "in `summary.elidedByState` and `coverage`, never dropped.";

/**
 * The publication predicate.
 *
 * Each arm is an EVIDENCE arm — something a human wrote down or a machine measured about this exact
 * coordinate. Two manifest-derived evidence dimensions are deliberately NOT arms:
 * `evidence.declaredCalibration` and `evidence.loadability` are functions of (entry, backend, tier)
 * alone, present on essentially every coordinate, so admitting them would publish the cross-product
 * again under a different name.
 *
 * `evidence.structural` rather than `state === "Structurally N/A"`: `validateMatrix` already proves
 * the two agree, and the cited evidence is the reason the row is worth keeping. Eliding a
 * Structurally N/A verdict would be the one genuinely lossy elision, because an absent coordinate
 * reads as "nothing has been done here" and that verdict says the opposite.
 *
 * A bare `isImplemented()` state is NOT an arm on its own. `state` is counted for every coordinate in
 * `coverage[].implemented`, which is what the coverage claim actually needs; a per-coordinate row
 * adds nothing when the claim is "this route exists", replicated across every tier x mode x overlay.
 */
export function isPublishableCell(cell, { plannedCellIds, calibrationRunCellIds }) {
  if (plannedCellIds.has(cell.id)) return true;
  if (calibrationRunCellIds.has(cell.id)) return true;
  if (cell.memoryCharacterization.status !== "unmeasured") return true;
  return [
    cell.evidence.historicalVerification,
    cell.evidence.currentEnvironmentVerification,
    cell.evidence.strategyParameterVerification,
    cell.evidence.structural,
  ].some((dimension) => dimension.length > 0);
}

/** Every resolved coordinate the shipped calibration plan targets, matched by the plan's own matcher. */
export function plannedCellIds(calibrationPlan, cells) {
  const planned = new Set();
  for (const cell of cells) {
    if (calibrationPlan.providers.some((entry) => planEntryTargetsCoordinate(entry, cell))) {
      planned.add(cell.id);
    }
  }
  return planned;
}

/**
 * Every shipped plan entry must address a coordinate that actually exists. FAIL CLOSED.
 *
 * ## The defect this exists because of
 *
 * Nine `config/memory-calibration-plan.json` entries (the sc-15817 candle-qwen-edit set) carried
 * `mode: "edit"` while the catalog's mode axis — and therefore every matrix coordinate and every
 * record that could ever bind to one — spells that capability `edit_image`. They matched ZERO
 * coordinates. Nothing noticed: `expectedEngagedRungs` simply returned `null` for compositions it
 * could not find, `memory-calibration.schema.json` types `mode` as a free string, and while the
 * matrix published the whole cross-product the entries' targets were on the page regardless. sc-18099
 * made the consequence visible — `qwen_image_edit_2511` and its lightning twin published no cells at
 * all, so the artifact hid the exact lanes the plan was aiming at — but the mismatch was already
 * costing something worse than visibility: a capture run against those entries would have produced
 * records that bind to nothing.
 *
 * So this is not a slim guard. It is the check that should always have existed: a plan entry naming a
 * coordinate the catalog cannot express is a typo, a stale target, or a vocabulary drift, and all
 * three are defects. It throws rather than warns because the failure mode it replaces was silence.
 */
export function assertCalibrationPlanTargetsResolvedCoordinates(calibrationPlan, cells) {
  const unmatched = calibrationPlan.providers.filter(
    (entry) => !cells.some((cell) => planEntryTargetsCoordinate(entry, cell)),
  );
  if (!unmatched.length) return;
  const detail = unmatched
    .map(
      (entry) =>
        `${entry.name} -> ${entry.target.modelId}:${entry.target.provider}:${entry.backend}:` +
        `${entry.target.tier}:${entry.target.mode}:${matrixOverlayFor(entry.target.overlay)}:${entry.rung}`,
    )
    .sort();
  throw new Error(
    `config/memory-calibration-plan.json has ${unmatched.length} entr${unmatched.length === 1 ? "y" : "ies"} ` +
      `that match no resolved matrix coordinate:\n  ${detail.join("\n  ")}\n` +
      "Each names a (model, provider, backend, tier, mode, overlay, rung) the catalog does not resolve. " +
      "Check the axis vocabularies first — modes are catalog CAPABILITY ids (`edit_image`, not `edit`) " +
      "and overlays normalise through matrixOverlayFor. A plan entry that addresses nothing cannot be " +
      "captured against: the record it would produce binds to no cell (sc-18099).",
  );
}

/**
 * The per-(entry, backend, rung) census over the FULL resolved cross-product.
 *
 * This is the "no silent cap" half of the slim: it names how many coordinates each lane resolved to,
 * how many were published, how many were elided, and the state distribution of ALL of them. A
 * consumer that used to count `cells` by state reads this instead and gets the same numbers.
 *
 * ## Why a row can carry `implementedBy`
 *
 * A row spans tier x mode x overlay, so a bare `implemented` count is unambiguous ONLY when it is 0
 * or `coordinates`. In between it hides WHICH coordinates — and that is not academic: 74 of 530 rows
 * are mixed, and five of those are CONTROL lanes that publish no cell at all.
 * `krea_2_turbo:mlx:bounded_transformer_residency` reads `implemented 12/18` while its control
 * overlay is really 0/6, and the pre-slim artifact answered that with a `Missing` control cell. A
 * consumer could not tell "control implemented but unmeasured" from "control not implemented" — the
 * sc-16069 confusion, for the exact lane family sc-16069 was about.
 *
 * So a mixed row publishes per-axis MARGINALS of its implemented count. Marginals, not a joint
 * distribution: `{tier: {bf16: 6}, overlay: {none: 6}}` says the implemented coordinates are all bf16
 * and all overlay-none, which pins the joint only when the marginals are that tight. That is enough
 * for the question the slim must keep answerable — "is this axis value implemented at all" — and it
 * is stated as marginals so nobody reads more out of them.
 *
 * Conditional rather than always present, because an all-or-nothing row already answers every such
 * question from `implemented` alone, and keying every row by overlay instead would cost ~320 KB in an
 * artifact this story exists to shrink.
 */
export function coverageCensus(cells, publishedIds) {
  const rows = new Map();
  for (const cell of cells) {
    const key = `${cell.modelId}:${cell.backend}:${cell.rung}`;
    let row = rows.get(key);
    if (!row) {
      row = {
        modelId: cell.modelId,
        backend: cell.backend,
        rung: cell.rung,
        coordinates: 0,
        published: 0,
        elided: 0,
        implemented: 0,
        states: {},
        // Accumulated for every row, published only on mixed ones.
        axisTotals: { tier: {}, mode: {}, overlay: {} },
        axisImplemented: { tier: {}, mode: {}, overlay: {} },
      };
      rows.set(key, row);
    }
    row.coordinates += 1;
    if (publishedIds.has(cell.id)) row.published += 1;
    else row.elided += 1;
    row.states[cell.state] = (row.states[cell.state] ?? 0) + 1;
    const implemented = isImplemented(cell.state);
    if (implemented) row.implemented += 1;
    for (const axis of ["tier", "mode", "overlay"]) {
      row.axisTotals[axis][cell[axis]] = (row.axisTotals[axis][cell[axis]] ?? 0) + 1;
      row.axisImplemented[axis][cell[axis]] =
        (row.axisImplemented[axis][cell[axis]] ?? 0) + (implemented ? 1 : 0);
    }
  }
  const sortedCounts = (counts) =>
    Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)));
  const finished = [];
  for (const row of rows.values()) {
    const mixed = row.implemented > 0 && row.implemented < row.coordinates;
    const { axisTotals, axisImplemented, ...published } = row;
    published.states = sortedCounts(row.states);
    if (mixed) {
      // Every value of the axis appears, including the zeroes — a value silently absent would be the
      // same blind spot in miniature.
      published.implementedBy = Object.fromEntries(
        ["tier", "mode", "overlay"].map((axis) => [
          axis,
          sortedCounts(
            Object.fromEntries(
              Object.keys(axisTotals[axis]).map((value) => [value, axisImplemented[axis][value] ?? 0]),
            ),
          ),
        ]),
      );
    }
    finished.push(published);
  }
  return finished.sort((left, right) =>
    `${left.modelId}:${left.backend}:${left.rung}`.localeCompare(`${right.modelId}:${right.backend}:${right.rung}`),
  );
}

/**
 * Prove the published document is internally closed before it is written.
 *
 * The slim introduces exactly one new way to ship a broken artifact: a reference that survives while
 * the row it names does not. Both directions are checked, because a dangling `cellId` and a census
 * that disagrees with the published set are the same defect seen from two sides.
 */
export function assertPublishedDocumentIsClosed(matrix, resolvedCoordinateCount) {
  const publishedIds = new Set(matrix.cells.map((cell) => cell.id));
  if (publishedIds.size !== matrix.cells.length) {
    throw new Error("published memory-matrix cells contain a duplicate id");
  }
  for (const run of matrix.calibrationRuns) {
    if (!publishedIds.has(run.cellId)) {
      throw new Error(
        `${run.record.id}: calibration run names cell ${run.cellId}, which the slim did not publish — ` +
          "a bound record must always keep its cell (sc-18099)",
      );
    }
  }
  for (const [modelId, slice] of Object.entries(matrix.modelSlices)) {
    for (const id of slice) {
      if (!publishedIds.has(id)) throw new Error(`modelSlices.${modelId} names unpublished cell ${id}`);
    }
  }
  // Both directions on the hoisted manifest evidence: no cell may point at a scope that is not
  // published, and no scope may be published that no cell points at.
  const referencedScopes = new Set(matrix.cells.map((cell) => cell.evidence.manifestScope));
  for (const cell of matrix.cells) {
    if (!Object.hasOwn(matrix.manifestScopes, cell.evidence.manifestScope)) {
      throw new Error(`${cell.id}: evidence.manifestScope ${cell.evidence.manifestScope} is not published`);
    }
    if (cell.evidence.manifestScope !== manifestScopeKey(cell)) {
      throw new Error(
        `${cell.id}: evidence.manifestScope is ${cell.evidence.manifestScope}, not this cell's scope ${manifestScopeKey(cell)}`,
      );
    }
  }
  for (const key of Object.keys(matrix.manifestScopes)) {
    if (!referencedScopes.has(key)) throw new Error(`manifestScopes.${key} is referenced by no published cell`);
  }
  for (const row of matrix.coverage) {
    if (row.published + row.elided !== row.coordinates) {
      throw new Error(`${row.modelId}:${row.backend}:${row.rung}: coverage published + elided != coordinates`);
    }
    // `implementedBy` is present on exactly the rows a bare `implemented` cannot answer, and each
    // axis must account for the whole count. A marginal that summed to something else would be a
    // more convincing wrong answer than no breakdown at all.
    const mixed = row.implemented > 0 && row.implemented < row.coordinates;
    if (mixed !== Object.hasOwn(row, "implementedBy")) {
      throw new Error(
        `${row.modelId}:${row.backend}:${row.rung}: implementedBy must be present on exactly the rows ` +
          `whose implemented count is partial (implemented ${row.implemented} of ${row.coordinates})`,
      );
    }
    for (const [axis, counts] of Object.entries(row.implementedBy ?? {})) {
      const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
      if (total !== row.implemented) {
        throw new Error(
          `${row.modelId}:${row.backend}:${row.rung}: implementedBy.${axis} sums to ${total}, not ${row.implemented}`,
        );
      }
    }
  }
  const censusCoordinates = matrix.coverage.reduce((total, row) => total + row.coordinates, 0);
  const censusPublished = matrix.coverage.reduce((total, row) => total + row.published, 0);
  if (censusCoordinates !== resolvedCoordinateCount) {
    throw new Error(
      `coverage census covers ${censusCoordinates} coordinates but the catalog resolved ${resolvedCoordinateCount}`,
    );
  }
  if (censusPublished !== matrix.cells.length) {
    throw new Error(
      `coverage census reports ${censusPublished} published cells but ${matrix.cells.length} were written`,
    );
  }
  if (matrix.summary.publishedCells + matrix.summary.elidedCells !== matrix.summary.cells) {
    throw new Error("summary published + elided must equal the resolved coordinate count");
  }
}

/**
 * The manifest-derived evidence dimensions, published once per scope instead of once per coordinate.
 *
 * `declaredCalibration` and `loadability` are functions of (entry, backend, tier) ALONE — the same
 * two arrays are recomputed identically for every mode x overlay x rung under that scope. They are
 * also the two dimensions deliberately excluded from the publication predicate for exactly that
 * reason. Among the published cells there are 35 distinct scopes carrying 182 copies, and the copies
 * were 154 KB of a ~1 MB artifact whose whole purpose is to stop repeating itself.
 *
 * Cells keep an explicit `evidence.manifestScope` key rather than leaving the reader to rebuild it
 * from the coordinate, so the join is stated and `assertPublishedDocumentIsClosed` can check it.
 * `evidenceDimensions` still names all six dimensions: what changed is where two of them are
 * written, not that the model has them.
 */
export function manifestScopeKey(cell) {
  return `${cell.modelId}:${cell.backend}:${cell.tier}`;
}

export function hoistManifestScopes(cells) {
  const scopes = {};
  const hoisted = cells.map((cell) => {
    const key = manifestScopeKey(cell);
    const scope = {
      declaredCalibration: cell.evidence.declaredCalibration,
      loadability: cell.evidence.loadability,
    };
    const existing = scopes[key];
    if (existing) {
      // The hoist is only sound because the two dimensions really are scope-invariant. If a future
      // change makes either depend on mode, overlay or rung, this catches it at generation time
      // rather than silently publishing whichever coordinate happened to be visited first.
      if (JSON.stringify(existing) !== JSON.stringify(scope)) {
        throw new Error(
          `${cell.id}: manifest-derived evidence differs between coordinates of scope ${key}, so it ` +
            "cannot be published per scope — it is no longer a function of (entry, backend, tier)",
        );
      }
    } else {
      scopes[key] = scope;
    }
    const { declaredCalibration, loadability, ...rest } = cell.evidence;
    return { ...cell, evidence: { ...rest, manifestScope: key } };
  });
  return {
    cells: hoisted,
    manifestScopes: Object.fromEntries(
      Object.entries(scopes).sort(([left], [right]) => left.localeCompare(right)),
    ),
  };
}

/**
 * The calibration record as the matrix publishes it (sc-18099).
 *
 * The matrix used to embed each record VERBATIM — 65 copies of rows that already ship, fully
 * schema-validated, in `docs/generated/memory-calibration-evidence.json`. That duplication was 436 KB
 * of the artifact. What the matrix actually needs to say about a record is which coordinate it
 * targets, whether it bound, and how it dates; `id` is the join key back to the full row, and every
 * consumer of the dropped fields already loads that bundle.
 */
function publishedCalibrationRecord(record) {
  return {
    id: record.id,
    status: record.status,
    backend: record.backend,
    target: record.target,
    strategy: { rung: record.strategy.rung, engagedRungs: record.strategy.engagedRungs },
    calibrationFingerprint: record.calibrationFingerprint,
    capturedAt: record.capturedAt,
    harnessVersion: record.harnessVersion,
    repositories: record.repositories,
    source: `docs/generated/memory-calibration-evidence.json#${record.id}`,
  };
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
  inferenceClosures: "config/inference-provider-closures.json",
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

/**
 * @param {object} [options]
 * @param {boolean} [options.publish] When false, return the document BEFORE sc-18099's publication
 *   step: `cells` is the full resolved cross-product, `modelSlices` covers all of it, and
 *   `calibrationRuns[].record` is the unprojected evidence row. `coverage` and the
 *   `summary.publishedCells`/`elidedCells`/`elidedByState` counts are still computed, because they
 *   describe what publication WOULD do and are derived from the full set either way.
 *
 *   This exists for the generator's own tests, which assert which STATE the generator assigns to a
 *   coordinate. That is a claim about generation, not about publication, and most of those
 *   coordinates are elided — asserting them against the published subset would silently reduce
 *   thirteen behavioural tests to vacuous ones. The CLI never passes it: `main()` writes the
 *   published document, and the publication path has its own tests.
 */
export async function buildMatrix({ sourceOverrides = {}, cellFilter = null, publish = true } = {}) {
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
  const calibrationPlan = activeCalibrationPlan(JSON.parse(bodies.calibrationPlan));
  // sc-17774: per-provider compile-closure digests, gated against the Cargo pin. `closureIsCurrent`
  // wants the Map; `evidenceSemantics` takes a plain object so the harness needs no Map plumbing.
  const inferenceClosureDigests = validatedInferenceClosures(
    bodies.inferenceClosures,
    inferencePin(cargoBody),
  );
  const closureDigestsByProvider = Object.fromEntries(inferenceClosureDigests);
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
    // sc-18099: the resolved AXES are published on the entry, not just counted here.
    //
    // Elision is safe for a coordinate's evidence, which is absent by definition. It is NOT safe for
    // a lane's EXISTENCE: sc-16069 exists because keying the control overlay off a measurement block
    // gave the shipping MLX Krea control lane zero cells, and absent evidence read as absent feature.
    // Publishing only planned-or-evidenced cells would recreate that blind spot for every unmeasured
    // axis value — no `control` cell, no `bf16` cell, no way to tell an unmeasured lane from one that
    // does not exist. These four lists ARE the cross-product (tiers x modes x overlays x rungs), so a
    // reader can see every coordinate the catalog resolves whether or not one was published.
    modelSummary.axes = {};
    for (const backend of modelSummary.backends) {
      const tiers = tiersFor(model, backend, backendTierOverrides);
      const modes = modesFor(model);
      const overlays = overlaysFor(model, backend);
      modelSummary.axes[backend] = { tiers, modes, overlays, rungs: [...RUNGS] };
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
                inferenceClosureDigests,
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
              const semantics = (record) =>
                evidenceSemantics(record, {
                  sceneWorks: sceneWorksRevision,
                  inference: pin,
                  inferenceClosureDigests: closureDigestsByProvider,
                });
              const historicalRuns = eligibleRuns.filter(
                (record) => semantics(record) === "historical",
              );
              const currentRuns = eligibleRuns.filter(
                (record) =>
                  semantics(record) === "current" &&
                  ["complete", "runtime_complete"].includes(record.status),
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
            inferenceClosureDigests: closureDigestsByProvider,
          }),
      record,
    };
  });

  // Replaced after validation with the PUBLISHED cells only (sc-18099). Built over the full set here
  // so the document shape below is unchanged and `validateMatrix` still sees the full inventory.
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
      // sc-18099: the family-level half of the verdict lives HERE now instead of on the cells.
      // `summary`, `blockStacks` and `findings` are constants of the (family, backend) pair that
      // used to be restated on all ~1,828 rung-4 cells — 2.72 MB of the old artifact. Most rung-4
      // coordinates are elided, so leaving them on cells would drop the verdict's evidence entirely
      // for every family that publishes none, which is what the survey exists to prevent.
      //
      // Read from the survey map, but only for a family the loop above proved REACHED a cell: the
      // reach proof is this loop, not the field's provenance, so a verdict that generated nothing
      // still cannot appear here as though it had.
      ...(({ summary, blockStacks = [], findings = [] }) => ({ summary, blockStacks, findings }))(
        rung4Survey.get(key),
      ),
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
    // 7 (sc-18099): the document no longer publishes the catalog cross-product. `cells` is now the
    // planned-or-evidenced SUBSET (`PUBLICATION_PREDICATE`), `modelSlices` lists only published ids
    // and may be EMPTY for an entry, `coverage` and `models[].axes` were ADDED and are REQUIRED,
    // `summary` gained `publishedCells`/`elidedCells`/`elidedByState`/`publicationPredicate`,
    // `rung4SurveyRows` absorbed the family-level `summary`/`blockStacks`/`findings` that
    // `cells[].rung4Survey` no longer carries, and `calibrationRuns[].record` is a PROJECTION of the
    // evidence-bundle row rather than the row itself. `summary.cells` keeps its meaning — the number
    // of coordinates the catalog resolved to, which is no longer `cells.length`. A version-6 reader
    // that counts `cells` by state now reads a sample and calls it a census, which is precisely why
    // this is a new version rather than an additive one.
    schemaVersion: 7,
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
      // The number of coordinates the CATALOG resolved to, unchanged in meaning by sc-18099 and no
      // longer equal to `cells.length`. `publishedCells` and `elidedCells` partition it.
      cells: cells.length,
      publishedCells: cells.length,
      elidedCells: 0,
      elidedByState: {},
      publicationPredicate: PUBLICATION_PREDICATE,
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
    coverage: [],
    // Populated by the publication step's `hoistManifestScopes`; empty in the pre-publication view,
    // where the two dimensions are still on the cells.
    manifestScopes: {},
    cells,
    calibrationRuns,
    modelSlices,
  };
  // Validation runs against the FULL resolved cross-product, before anything is elided. Every guard
  // above — inventory, ownership scope, twin coverage, rung-4 survey reach, the two-claims
  // invariants — therefore keeps exactly the reach it had when the artifact was the cross-product.
  // The slim is a publication step, not a generation step (sc-18099).
  validateMatrix(
    matrix,
    expectedIds,
    backendTierOverrides,
    rung4Survey,
    cellInventoryExpectations,
    calibrationPlan,
  );

  const planned = plannedCellIds(calibrationPlan, cells);
  const calibrationRunCellIds = new Set(calibrationRuns.map((run) => run.cellId));
  const published = cells.filter((cell) =>
    isPublishableCell(cell, { plannedCellIds: planned, calibrationRunCellIds }),
  );
  const publishedIds = new Set(published.map((cell) => cell.id));
  matrix.coverage = coverageCensus(cells, publishedIds);
  matrix.summary.publishedCells = published.length;
  matrix.summary.elidedCells = cells.length - published.length;
  matrix.summary.elidedByState = Object.fromEntries(
    sortedUnique(cells.filter((cell) => !publishedIds.has(cell.id)).map((cell) => cell.state)).map(
      (state) => [
        state,
        cells.filter((cell) => !publishedIds.has(cell.id) && cell.state === state).length,
      ],
    ),
  );
  if (!publish) return matrix;
  const hoisted = hoistManifestScopes(published);
  matrix.cells = hoisted.cells;
  matrix.manifestScopes = hoisted.manifestScopes;
  matrix.modelSlices = Object.fromEntries(
    models.map((model) => [
      model.id,
      published.filter((cell) => cell.modelId === model.id).map((cell) => cell.id),
    ]),
  );
  matrix.calibrationRuns = calibrationRuns.map((run) => ({
    ...run,
    record: publishedCalibrationRecord(run.record),
  }));
  assertPublishedDocumentIsClosed(matrix, cells.length);
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
    `- Resolved coordinates: ${matrix.summary.cells}`,
    `- Published cells: ${matrix.summary.publishedCells}`,
    `- Elided coordinates: ${matrix.summary.elidedCells} (${
      Object.entries(matrix.summary.elidedByState)
        .map(([state, count]) => `${state} ${count}`)
        .join(", ") || "none"
    })`,
    `- MLX staged-residency static coverage: ${matrix.summary.mlxStagedStaticCoverage}/${matrix.summary.mlxStagedStaticCoverageDenominator}`,
    `- Full models: ${matrix.summary.fullModels}`,
    `- Full complete calibration records: ${matrix.summary.calibrationRunsByStatus.complete}`,
    `- Base-only runtime-complete calibration records: ${matrix.summary.calibrationRunsByStatus.runtimeComplete}`,
    "",
    `sc-18099: \`cells\` is a SUBSET. ${matrix.summary.publicationPredicate} The counts on this page, \`summary\`, and the per-(entry, backend, rung) \`coverage\` census in the JSON artifact are all derived from every resolved coordinate, published or not, and \`models[].axes\` publishes the axes those coordinates span so an unmeasured lane stays distinguishable from an absent one.`,
    "",
    "Static capability is never promoted to dynamic verification. The six evidence dimensions stay separate: `staticImplementation`, `historicalVerification`, `currentEnvironmentVerification`, `strategyParameterVerification` and `structural` are per-coordinate and ride the cell; `declaredCalibration` and `loadability` are functions of (entry, backend, tier) alone and are published once per scope in `manifestScopes`, which the cell names through `evidence.manifestScope` (sc-18099).",
    "`Runtime verified` means the exact base-only coordinate is production-admissible from current runtime evidence; it is deliberately not Full `Verified`, which additionally requires the catalog story's lifecycle and negative-mutation signoff.",
    "",
    "One row per (catalog entry, backend): ownership is backend-scoped, so a single row per entry could only name one backend's stories (SC-15812).",
    "",
    "| Catalog entry | Backend | Route | Family story | Model story | Staged residency |",
    "| --- | --- | --- | --- | ---: | --- |",
  ];
  for (const model of matrix.models) {
    for (const backend of model.backends) {
      // sc-18099: read the census, not `cells`. This column is a claim about the whole lane, and
      // `cells` is now a subset — scanning it would silently under-report every lane whose staged
      // coordinates were elided, which is the coverage regression the slim must not cause.
      const row = matrix.coverage.find(
        (candidate) =>
          candidate.modelId === model.id &&
          candidate.backend === backend &&
          candidate.rung === "staged_residency",
      );
      if (!row) throw new Error(`${model.id}:${backend}: no staged_residency coverage row`);
      const staged = row.implemented > 0;
      const stagedState = row.states.Verified
        ? "Verified"
        : row.states["Runtime verified"]
          ? "Runtime verified"
          : "Implemented/unverified";
      lines.push(
        `| \`${model.id}\` | ${backend} | \`${model.resolvedRoute}\` (${model.routeKind}) | SC-${model.owningFamilyStories[backend]} | SC-${model.owningModelStories[backend]} | ${staged ? stagedState : "Missing"} |`,
      );
    }
  }
  lines.push(
    "",
    `Per-model consumers read \`modelSlices\` in the JSON artifact for an entry's PUBLISHED cells — but since sc-18099 that is a subset, and ${Object.values(matrix.modelSlices).filter((slice) => slice.length === 0).length} of ${matrix.models.length} entries publish none at all. An empty slice means nothing was planned, measured, bound or cited there; it does NOT mean the entry has no lanes. For "which lanes exist" read \`models[].axes\`, and for "how much of a lane is implemented" read \`coverage\`. A cell is Full only when every applicable rung is Verified or Structurally N/A; this static baseline intentionally reports zero Full models.`,
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
    `Surveyed family/backend pairs: ${matrix.summary.rung4Survey.surveyedFamilyBackends}. sc-18099 split the verdict by what it is a property OF: the family-level summary, block-stack inventory and findings are on \`rung4SurveyRows\` in the JSON artifact — carried once per (family, backend), so they survive a family whose rung-4 cells were all elided — while \`cells[].rung4Survey\` keeps the genuinely per-coordinate half, the resolved request-peak finding and the overlay-incompatibility verdict.`,
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
