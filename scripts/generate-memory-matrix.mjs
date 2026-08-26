#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
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
import {
  reconcileMemoryContracts,
} from "./lib/memory-contract-reconciliation.mjs";
import { CONVERTER_TIER_OVERRIDES, contractIsLoraOnly } from "./lib/manifest-memory-declarations.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT_JSON = "docs/generated/memory-matrix.json";
const OUTPUT_MD = "docs/generated/memory-matrix.md";
const EXPECTED_IMAGE_COUNT = 53;
// sc-18815: the modalities the matrix carries. `utility` and `audio` entries have no memory ladder —
// no rungs, no fit gate, no strategy selector — so they are outside the universe by design rather
// than by omission, and admitting one would need the same three things video needed here.
const MATRIX_MODALITIES = new Set(["image", "video"]);
const EXPECTED_VIDEO_COUNT = 10;
// SC-18218 removed FLUX.2-dev from the MLX staged-residency census when the then-pinned provider was
// eager/resident-only; sc-20799 REVERSED that at pin ebcdc7da7, where all three Dev MLX providers
// declare selectable Sequential staged residency, so flux2_dev is now census-required (the assertion
// direction flipped with the pin). Bernini is the older mirror case — inference sc-18609 made its
// DECLARED MLX rung-4 ladder actually reachable on both variants, so it belongs in the census.
//
// Neither fact is a total. This census used to be pinned to an exact population, which meant hand-
// renewing 37 -> 38 for a reachability change that had nothing to do with the contract being guarded,
// and the number never said which entry moved. `assertMlxStagedCoverageIsStructurallyConsistent`
// replaces it, and is DELIBERATELY WEAKER — the honest scope, so nobody reads more into it:
//
//   guarded — the two named lanes above by id; bespoke routes never claiming the generic ladder;
//             per-route drift, where entries sharing a resolved route disagree with each other; and
//             the census being neither empty nor the whole catalog.
//   NOT guarded — uniform drift on a route nothing else shares. 35 of the 41 resolved routes are
//             singletons (only sdxl, flux2_klein_9b, qwen_image_edit, sensenova_u1_8b,
//             sensenova_u1_8b_fast and z_image_turbo group more than one entry), so a singleton lane
//             silently dropping out of the census — or silently claiming staged coverage it has not
//             implemented — passes every assertion here where the old count reddened. A whole shared
//             family drifting uniformly passes too, for the same reason.
//   ALSO NOT guarded, as of 2026-08-17 — the per-route comparison SKIPS entries whose MLX tier axis is
//             a single synthetic `default`, i.e. entries advertising no tier ladder at all (no
//             `vramGbByTier`, no tier-tagged download variant, and a `quantize` naming no packed tier).
//             `flux2_klein_9b_true_v2` is the only such entry today, and it shares route
//             `flux2_klein_9b` with two tiered siblings. Its verdict is structurally fixed at "not
//             staged" — the tiers its contract declares do not exist for it — so including it reported
//             a disagreement no declaration could resolve. A future single-dense-tier entry therefore
//             joins that blind spot silently. Accepted for the same reason as the rest of this note:
//             the alternative was declaring a packed tier the artifact does not ship.
//
// That is the accepted shape-over-population tradeoff, not an oversight: the exact count caught those
// cases and cost a hand-edit on every unrelated catalog or reachability change, and runtime catching is
// the chosen tradeoff for what shape assertions cannot express. A staged claim a lane cannot honour
// surfaces when the ladder is actually engaged.
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

// The catalog `capabilities` that name a GENERATION MODE, which is the matrix's mode axis. A
// capability outside this set (a UI affordance, a conditioning kind) is not a mode and must not
// multiply the cross-product.
//
// sc-18815 adds the video modes. Every one is a distinct request shape the worker dispatches on
// (`video_jobs::resolve_video_route` branches on `request.mode`), and several select a different
// engine path entirely — `replace_person` and `animate_character` reach the VACE/SCAIL-2 arms rather
// than the family's base generator. Collapsing them to one `catalog_default` mode, which is what the
// image-only vocabulary did to every video entry, would publish one cell for coordinates whose peaks
// are not the same measurement.
const GENERATION_CAPABILITIES = new Set([
  "text_to_image",
  "edit_image",
  "image_to_image",
  "image_inpaint",
  "image_detail",
  "character_image",
  "style_variations",
  "text_to_video",
  "image_to_video",
  "video_to_video",
  "first_last_frame",
  "extend_clip",
  "video_bridge",
  "replace_person",
  "animate_character",
  "reference_to_video",
  "reference_video_to_video",
  "multi_video_to_video",
  "ads2v",
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

/**
 * Ownership for the VIDEO families epic 18803 admits (sc-18815).
 *
 * Kept as its own registry rather than as rows in `FAMILY_STORIES` because the two lanes' ownership
 * rules genuinely differ, and collapsing them would either weaken the image lane's guard or force
 * this lane to name stories that do not exist:
 *
 * - **One story may cover BOTH backends.** Epic 15448 split every image family per backend because
 *   its evidence is produced on backend-specific hardware — an MLX story cannot be closed from CUDA.
 *   A video family here is owned by its rung-4 SURVEY story, whose evidence is provider source read
 *   at the pinned inference revision. sc-18813 delivered the MLX *and* candle LTX verdicts in one
 *   story from one machine; inventing a per-backend pair would name a candle twin nobody filed.
 * - **One story may cover several families.** sc-18828 surveys `scail2`, `krea-realtime` and `svd`
 *   together. They are three unrelated architectures — merging them into one family group to satisfy
 *   a one-story-one-owner rule would publish a false family — so the rule gives way, not the family.
 *
 * `bernini` is deliberately absent: its video and image entries are the SAME provider (`bernini`) and
 * the SAME architecture, so it stays in image family 15528, whose survey verdict is already written
 * about the Bernini renderer rather than about a still-image path. sc-18827 reconciles the video-side
 * entry coverage within that family.
 */
export const VIDEO_FAMILY_STORIES = Object.freeze({
  "ltx-video": 18813,
  "wan-video": 18826,
  scail2: 18828,
  "krea-realtime": 18828,
  svd: 18828,
});

/**
 * The two ownership registries must be DISJOINT (sc-18815 review).
 *
 * Contamination in ONE direction already fails loudly: an image family group added to
 * `VIDEO_FAMILY_STORIES` makes `familyStory` answer with a single video survey story for a lane that
 * owes two per-backend owners, and the conformance suite reds hard.
 *
 * The other direction is silent, and nothing else catches it. `familyStory` consults
 * `VIDEO_FAMILY_STORIES` FIRST, so a VIDEO family name added to `FAMILY_STORIES` is never read — dead
 * ownership metadata that looks filed, answers nothing, and keeps looking filed while the real answer
 * comes from the other registry.
 */
export function assertOwnershipRegistriesAreDisjoint(
  imageFamilies = FAMILY_STORIES,
  videoFamilies = VIDEO_FAMILY_STORIES,
) {
  for (const group of Object.keys(videoFamilies)) {
    if (Object.hasOwn(imageFamilies, group)) {
      throw new Error(
        `memory-matrix: family group ${group} is declared in BOTH FAMILY_STORIES and VIDEO_FAMILY_STORIES — familyStory resolves it from VIDEO_FAMILY_STORIES, so the FAMILY_STORIES row is dead code; delete one`,
      );
    }
  }
}

// At module scope, so neither registry can be loaded in a drifted state by any consumer.
assertOwnershipRegistriesAreDisjoint();

/**
 * Families that are IN the model universe but whose rung-4 verdict has not been written yet, with the
 * story that owes it (sc-18815).
 *
 * This is the only honest way to admit a modality one story at a time. The alternative orderings both
 * publish something false: admitting the family with no marker makes its rung-4 cells read as
 * surveyed and found wanting, and holding the family out of the universe until its verdict lands is
 * exactly the silent absence — `type === "image"` — this story exists to remove.
 *
 * It is a debt, not an exemption. `assertRung4SurveyCoversEveryFamily` rejects a row for a family the
 * catalog does not advertise AND a row whose verdict has since landed, so each entry can only be
 * deleted, never quietly kept; and the cells say `surveyed: false` with this story id on them rather
 * than looking like any other `Missing`.
 */
// SC-18826/SC-18828 supplied the remaining video verdicts. Keep the checked debt mechanism rather
// than deleting it: the next admitted family must still declare its owing story until its survey
// lands, and `assertRung4SurveyCoversEveryFamily` still rejects a stale declaration.
export const PENDING_RUNG4_SURVEYS = new Map();

/**
 * Catalog entries that are IN the universe but that the routing catalog routes on NO backend, with
 * the reason and the story that owns the verdict (sc-18815).
 *
 * SC-18826 closed the sole declared defect by adding the missing MLX-only
 * `wan_2_2_vace_fun_14b` row to `VIDEO_MODEL_CAPS`. The empty map is intentional: the guard below
 * remains fail-closed for the next wholly-unrouted catalog entry and rejects any declaration that
 * outlives its defect.
 *
 * `mochi_1` is the inverse defect and needs no row: it has a `VIDEO_MODEL_CAPS` row and a worker
 * route but NO manifest entry, and the universe is built from the manifest, so it is excluded by
 * construction. That is the correct outcome on its own terms — Mochi is frozen with no weights lane
 * and epic 18803 lists it out of scope — and `mochi_is_routed_but_not_in_the_universe` pins that the
 * exclusion is the manifest's doing rather than a coincidence of some other filter.
 */
export const UNROUTED_CATALOG_ENTRIES = new Map();

/**
 * Every entry that resolved to no backend must be a declared one, and every declared one must still
 * resolve to none. Fails closed both ways so the list can only shrink by fixing the defect.
 */
export function assertUnroutedEntriesAreDeclared(models, declared = UNROUTED_CATALOG_ENTRIES) {
  for (const model of models) {
    if (!model.backends.length && !declared.has(model.id)) {
      throw new Error(
        `${model.id}: the routing catalog routes it on no backend, so it generates zero cells and is indistinguishable from an entry that is not in the catalog at all — declare it in UNROUTED_CATALOG_ENTRIES with its reason and owner, or route it (sc-18815)`,
      );
    }
  }
  const byId = new Map(models.map((model) => [model.id, model]));
  for (const [id, entry] of declared) {
    const model = byId.get(id);
    if (!model) {
      throw new Error(`UNROUTED_CATALOG_ENTRIES names ${id}, which is not in the model universe`);
    }
    if (model.backends.length) {
      throw new Error(
        `${id} is declared unrouted (sc-${entry.owningStory}) but the catalog now routes ${model.backends.join(",")} — delete the row so its cells stop being explained away`,
      );
    }
  }
}

/** `SC-15509` for an image family group, the family's own name for a video one. */
export function familyLabel(group) {
  return typeof group === "number" ? `SC-${group}` : group;
}

/**
 * The stable family group key for a catalog id.
 *
 * Image families (epic SC-15448) are keyed by the family's MLX story id, which doubles as the group
 * key. The video families epic 18803 admits are keyed by the family's own NAME, because the story-id
 * convention does not survive this lane: sc-18828 owns three unrelated families at once, so no
 * assignment of story ids to those families is both distinct and true. A name is stable, carries no
 * false ownership claim, and is the vocabulary the epic itself uses. `VIDEO_FAMILY_STORIES` answers
 * the ownership question the key used to answer implicitly.
 *
 * `bernini` (video) and `bernini_image` share group 15528 on purpose — one engine, one block stack,
 * two catalog entries — so the family is not counted or surveyed twice.
 */
export function familyGroup(modelId) {
  if (modelId.startsWith("ltx_2_3")) return "ltx-video";
  if (modelId.startsWith("wan_2_2")) return "wan-video";
  if (modelId === "scail2_14b") return "scail2";
  if (modelId === "krea_realtime_14b") return "krea-realtime";
  if (modelId === "svd") return "svd";
  if (modelId === "bernini") return 15528;
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
  // A video family's survey story covers both backends (see `VIDEO_FAMILY_STORIES`), so it answers
  // for either without a twin. `bernini` is not here: it is image family 15528 and takes the
  // per-backend path below like every other member of that family.
  if (Object.hasOwn(VIDEO_FAMILY_STORIES, group)) return VIDEO_FAMILY_STORIES[group];
  const stories = FAMILY_STORIES[group];
  if (!stories) throw new Error(`${modelId}: family ${familyLabel(group)} has no ownership entry`);
  const story = stories[backend];
  if (!story) {
    throw new Error(
      `${modelId}: family ${familyLabel(group)} owns no ${backend} story, so a ${backend} cell cannot be attributed — file the ${backend} family twin`,
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
export function assertCellOwnershipIsBackendScoped(
  cells,
  scope = buildStoryBackendScope(),
  modalityByModelId = new Map(),
) {
  for (const cell of cells) {
    // sc-18815: the video lane's ownership is checked by `assertVideoOwnership` against its own rule
    // (no per-entry story, family story from `VIDEO_FAMILY_STORIES`), because `scope` is the image
    // lane's registry and a video cell resolves in it to nothing. Skipping is safe only BECAUSE the
    // other guard runs — the two are a partition of the cells, not a check and an exemption.
    if (modalityByModelId.get(cell.modelId) === "video") continue;
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
export function assertTwinCoverage(
  allModels,
  modelStories = MODEL_STORIES,
  familyStories = FAMILY_STORIES,
) {
  // sc-18815: the twin rule is an IMAGE-lane invariant and stays at full strength there. It encodes
  // epic 15448's per-backend split, which exists because an image family's evidence is produced on
  // backend-specific hardware. The video families this epic admits are owned by a single survey story
  // whose evidence is provider source read at the pinned revision (`VIDEO_FAMILY_STORIES`), so
  // demanding a distinct Candle twin would demand a story nobody filed and could only be satisfied by
  // inventing one. `assertVideoOwnership` checks the video lane against ITS rule instead — the lane
  // is not exempt, it is checked differently because it is owned differently.
  const models = allModels.filter((model) => model.modality !== "video");
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
  if (!staged.has("flux2_dev")) {
    throw new Error(
      "flux2_dev lost its MLX staged coverage; at pin ebcdc7da7 all three Dev providers declare " +
        "selectable Sequential staged residency (sc-20799 retired the SC-18218 resident-only pin)",
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
  //
  // EXEMPT: an entry whose MLX tier axis is a single synthetic `default`. That is not drift, it is an
  // entry with no advertised tier ladder AT ALL — `mlxTiers` falls through to `["default"]` when the
  // catalog declares no `vramGbByTier`, no tier-tagged download variant, and a `quantize` that names no
  // packed tier (`<= 0`, i.e. dense; see `resolve_quant`). Such an entry can never reach an Implemented
  // staged cell, because the tiers its contract declares do not exist for it, so comparing its verdict
  // against tiered siblings on the same route reports a disagreement that no declaration change can
  // resolve.
  //
  // The instance is `flux2_klein_9b_true_v2`: a convert-at-install entry whose transformer is a FIXED
  // DENSE BF16 artifact, sharing route `flux2_klein_9b` with the tiered base and KV entries. Coordinator
  // decision 2026-08-17, building on Michael's report-only ruling the same day: `quantize: 0` stays
  // because it is the only truthful encoding of that artifact — declaring a packed tier it does not ship
  // would hand-declare a q8 that does not exist and mislead tier selection, which is the drift class
  // this epic exists to kill. So the invariant yields on the axis instead of the manifest lying.
  //
  // Scoped as narrowly as that reasoning allows: the exemption removes these entries from the CROSS-
  // ENTRY comparison only. Every other assertion in this function still applies to them, and the
  // comparison keeps full force among tiered entries sharing a route — see
  // `a drifting tiered route-mate still reds after the single-dense-tier exemption` in
  // `generate-memory-matrix.test.mjs`, which exists so the exemption cannot be widened into a hole.
  const hasSingleDenseTierAxis = (model) => {
    const tiers = model.axes?.mlx?.tiers ?? [];
    return tiers.length === 1 && tiers[0] === "default";
  };
  const byRoute = new Map();
  for (const model of matrix.models) {
    if (hasSingleDenseTierAxis(model)) continue;
    const verdicts = byRoute.get(model.resolvedRoute) ?? new Set();
    verdicts.add(staged.has(model.id));
    byRoute.set(model.resolvedRoute, verdicts);
  }
  const split = [...byRoute.entries()].filter(([, verdicts]) => verdicts.size > 1).map(([route]) => route);
  if (split.length) {
    throw new Error(`MLX staged coverage disagrees within resolved route(s) ${split.sort().join(",")}`);
  }
  // A bespoke route carries its own pipeline and never advertises the GENERIC staged ladder.
  //
  // "Generic" is load-bearing and, as of 2026-08-17, actually enforced as written. This used to reject a
  // bespoke route for ANY implemented staged cell, which conflated two different claims: the generic
  // base ladder (the `none` overlay, plus the `lora` overlay that rides it) and a route-local closed
  // overlay the entry declares for itself.
  //
  // `routeKind: "bespoke"` means only "no row in `engines.rs`'s MODEL_TABLE" — a SceneWorks worker
  // DISPATCH fact, and backend-agnostic. It does not mean the engine registry publishes no contract.
  // PuLID is exactly that split: bespoke in worker dispatch, and on candle genuinely unregistered (the
  // candle dump carries its typed `bespokeMemoryRouteWaivers` entry), yet the MLX registry publishes a
  // real `pulid_flux` memory contract with route witnesses at pin 931366f62. SC-18460 wired its MLX
  // declaration route on that basis, and its census cells bear it out: `character_image` +
  // `identity` is Implemented on every tier while `none` and `lora` stay Missing on every tier — the
  // "deliberately closed identity-only contract" this file already describes in
  // `staticCandleOverlayIsAvailable`.
  //
  // So the check now asks the question it always claimed to: does a bespoke route claim staged coverage
  // on a GENERIC coordinate? Its own closed overlay is the one declaration it IS entitled to make.
  // Teeth preserved — if PuLID's `none` or `lora` staged cell ever turns Implemented this still reds;
  // `a bespoke route claiming the generic staged ladder still reds` pins that.
  const GENERIC_STAGED_OVERLAYS = new Set(["none", "lora"]);
  const bespoke = matrix.models
    .filter(
      (model) =>
        model.routeKind === "bespoke" &&
        matrix.cells.some(
          (cell) =>
            cell.backend === "mlx" &&
            cell.rung === "staged_residency" &&
            cell.modelId === model.id &&
            GENERIC_STAGED_OVERLAYS.has(cell.overlay) &&
            isImplemented(cell.state),
        ),
    )
    .map((model) => model.id);
  if (bespoke.length) {
    throw new Error(`bespoke route(s) ${bespoke.sort().join(",")} claim generic MLX staged coverage`);
  }
}

/**
 * The video lane's ownership rule (sc-18815), which is not the image lane's.
 *
 * Two claims, both fail-closed, and both about what must NOT be there as much as what must:
 *
 * 1. **No per-entry ownership story is invented.** Epic 18803 does not slice video per (entry,
 *    backend) — measurement is a runbook (epic 18093) — so `owningModelStories` is `null` on every
 *    video entry. A number appearing there is a fabricated owner, which is the same false-inventory
 *    defect SC-15812's guard was written for, reached from the other direction.
 * 2. **Every advertised backend resolves to the family's real owning story.** `bernini` takes the
 *    per-backend image path (family 15528) because it IS an image-family member; the rest take
 *    `VIDEO_FAMILY_STORIES`. Either way the value has to be the one `familyStory` derives, so a hand
 *    edit here cannot point a cell at a story that does not own it.
 */
export function assertVideoOwnership(allModels, videoFamilyStories = VIDEO_FAMILY_STORIES) {
  for (const model of allModels.filter((entry) => entry.modality === "video")) {
    const group = familyGroup(model.id);
    for (const backend of model.backends) {
      const modelStoryId = model.owningModelStories[backend];
      if (modelStoryId !== null) {
        throw new Error(
          `${model.id}:${backend}: video entries carry no per-entry ownership story, but this one names SC-${modelStoryId} — epic 18803 filed no per-(entry, backend) video stories, so the id cannot be one that owns this cell`,
        );
      }
      const expected = Object.hasOwn(videoFamilyStories, group)
        ? videoFamilyStories[group]
        : FAMILY_STORIES[group]?.[backend];
      if (!expected) {
        throw new Error(
          `${model.id}:${backend}: family ${familyLabel(group)} has no owning story on this backend`,
        );
      }
      if (model.owningFamilyStories[backend] !== expected) {
        throw new Error(
          `${model.id}:${backend}: owningFamilyStory SC-${model.owningFamilyStories[backend]} is not family ${familyLabel(group)}'s ${backend} owner SC-${expected}`,
        );
      }
    }
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

function isTemporalVideoCell(cell, modality) {
  if (modality !== "video") return false;
  const envelope = cell.geometryEnvelope ?? {};
  return (
    (Array.isArray(envelope.durations) && envelope.durations.length > 0) ||
    Number.isFinite(envelope.defaultDuration) ||
    Number.isFinite(envelope.hardMaxDuration) ||
    (Array.isArray(envelope.fps) && envelope.fps.length > 0) ||
    Number.isFinite(envelope.defaultFps)
  );
}

export function calibrationBinding(
  record,
  cell,
  { exactPlanEntries = [], modality = null } = {},
) {
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
  if (record.target.geometry.frames !== 1) {
    if (!isTemporalVideoCell(cell, modality)) reasons.push("frames-out-of-envelope");
    if (
      !exactPlanEntries.some((entry) => planEntryMatchesEvidenceRecord(entry, record))
    ) reasons.push("capture-geometry-unplanned");
  }
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
//                              by construction: the curve `vram_gate.rs#krea_phase_curve` evaluates
//                              is `fixedGb + perMpxGb*mpx + perMpxFrameGb*mpx*frames`, so it takes
//                              as many independent measured geometries as the lane has
//                              coefficients — two on the image lane, three once the temporal term
//                              is carried.
//
// Collapsing both into `state` is what let a single 768x768 capture read as certifying a cell whose
// envelope reaches 2048x2048. `point` is the honest middle state the five-value vocabulary could not
// express — and it is exactly where Krea's shipped q8/bf16 curves sit, each carrying one geometry
// point while the gate's own doc comment calls their slopes "fitted from real renders at multiple
// resolutions".
//
// `fitted` asserts the evidence is SUFFICIENT to determine the affine curve, not that a fit has been
// performed. `coveredPixelBound` is the largest measured area, so a consumer can tell how far the
// determinable curve reaches without re-deriving it from `measuredGeometries`; it is null unless the
// curve is determinable, because otherwise there is no curve to bound.
//
// ## sc-18812: the temporal axis
//
// A geometry is `WxH` (one output frame) or `WxHxfF` for F output frames. COUNTING measured
// geometries is no longer enough, for two separate reasons, and both bite:
//
//   1. Collapsing the temporal axis. Two records at `768x512xf121` and `768x512xf241` are two
//      measurements of a video cell but ONE area, and the pre-sc-18812 key would have deduped them
//      to a single `768x512` and reported `point` — hiding real temporal coverage.
//   2. Counting without rank. Three temporal geometries at ONE area cannot determine
//      `{fixed, perMpx, perMpxFrame}`: with one area the area and cross columns are proportional
//      and the design is singular. sc-18810 found exactly this — crossing TWO areas is what makes
//      the candidate forms identifiable at all — so `fitted` is decided by the RANK of the design
//      matrix, not by a count. This also fixes a latent image-lane defect: `768x512` and `512x768`
//      are two geometries carrying one area, and counting called that `fitted`.
//
// Which form to grade against comes from the cell's DECLARED curve first and the measurements only
// second (`declaresTemporalCurve`). Reading it from the measurements alone fails in the flattering
// direction: a cell carrying `perMpxFrameGb` whose eligible records all happen to be single-frame
// would be graded against two coefficients and could report `fitted` at two areas while its third
// coefficient is undetermined. Measurements can only ADD the temporal axis, never remove it.
//
// `coveredFrameBound` is emitted on cells whose form is temporal — declared or measured — mirroring
// how `geometryEnvelope` gains `durations`/`fps` only for video. No committed curve declares
// `perMpxFrameGb`, so every image cell keeps the exact shape it published before this change.
function parseMeasuredGeometry(geometry) {
  const match = /^([1-9][0-9]*)x([1-9][0-9]*)(?:xf([1-9][0-9]*))?$/.exec(geometry ?? "");
  if (!match) return null;
  return { pixels: Number(match[1]) * Number(match[2]), frames: Number(match[3] ?? 1) };
}

// Exact rank of the design matrix for the applicable curve form. Integer arithmetic throughout —
// `pixels` and `pixels * frames` are exact integers, and a float elimination could call a singular
// design full-rank on rounding alone, which is the one error this must not make.
function designRank(points, temporal) {
  const areas = new Set(points.map((point) => point.pixels));
  if (!temporal) return areas.size >= 2 ? 2 : Math.min(points.length, 1);
  const voxels = new Set(points.map((point) => point.pixels * point.frames));
  // Columns `[1, pixels, pixels*frames]`. Rank 3 iff some three rows have a non-zero determinant.
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      for (let k = j + 1; k < points.length; k += 1) {
        const row = (point) => [1n, BigInt(point.pixels), BigInt(point.pixels * point.frames)];
        const [a, b, c] = [row(points[i]), row(points[j]), row(points[k])];
        const determinant =
          a[0] * (b[1] * c[2] - b[2] * c[1]) -
          a[1] * (b[0] * c[2] - b[2] * c[0]) +
          a[2] * (b[0] * c[1] - b[1] * c[0]);
        if (determinant !== 0n) return 3;
      }
    }
  }
  return areas.size >= 2 || voxels.size >= 2 ? 2 : Math.min(points.length, 1);
}

// The `measuredGeometries` key for one calibration record's target geometry. `frames <= 1` emits
// the historical `WxH` form unchanged, so admitting the temporal axis moves no image cell.
export function measuredGeometryKey({ width, height, frames }) {
  return frames > 1 ? `${width}x${height}xf${frames}` : `${width}x${height}`;
}

// `declaresTemporalCurve` is the CELL's own curve form, read from the manifest rather than inferred
// from what happened to be measured. It matters because the two inputs disagree in exactly the
// direction that over-claims: a cell whose curve carries `perMpxFrameGb` has three coefficients
// whether or not its eligible records are all single-frame, and grading it against two would let it
// report `fitted` on two areas with the temporal coefficient undetermined — the same class of
// over-claim the rank rule was written to close, arrived at from the other side. The measurements
// can only ADD the temporal axis (a multi-frame geometry always implies three coefficients); they
// can never remove it.
export function memoryCharacterization(geometries, { declaresTemporalCurve = false } = {}) {
  const measured = sortedUnique(
    geometries.filter((geometry) => parseMeasuredGeometry(geometry) !== null),
  );
  const points = measured.map(parseMeasuredGeometry);
  const temporal = declaresTemporalCurve || points.some((point) => point.frames > 1);
  const coefficients = temporal ? 3 : 2;
  const determinable = designRank(points, temporal) >= coefficients;
  const status = measured.length === 0 ? "unmeasured" : determinable ? "fitted" : "point";
  return {
    status,
    measuredGeometries: measured,
    coveredPixelBound: status === "fitted" ? Math.max(...points.map((point) => point.pixels)) : null,
    ...(temporal
      ? {
          coveredFrameBound: status === "fitted"
            ? Math.max(...points.map((point) => point.frames))
            : null,
        }
      : {}),
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

/**
 * Does one calibration-plan entry describe this exact physical evidence receipt?
 *
 * This is intentionally separate from `planEntryTargetsCoordinate`. A matrix coordinate is
 * geometry-independent: planning any capture for a rung publishes that rung's cell. Binding a
 * multi-frame receipt is the opposite claim and must match the complete capture identity, including
 * the explicit frame count. Keeping the predicates separate prevents a planned `f121` capture from
 * blessing an unplanned `f241` receipt while preserving the historical one-frame publication rule.
 *
 * Duration and FPS are deliberately absent. The harness records output frames directly, and no
 * consumer may reverse-engineer that observed axis from request duration or FPS.
 */
export function planEntryMatchesEvidenceRecord(entry, record) {
  return (
    entry.backend === record.backend &&
    entry.target.modelId === record.target.modelId &&
    entry.target.provider === record.target.provider &&
    entry.target.tier === record.target.tier &&
    entry.target.mode === record.target.mode &&
    entry.target.overlay === record.target.overlay &&
    entry.rung === record.strategy.rung &&
    entry.target.geometry.width === record.target.geometry.width &&
    entry.target.geometry.height === record.target.geometry.height &&
    entry.target.geometry.batch === record.target.geometry.batch &&
    entry.target.geometry.frames === record.target.geometry.frames &&
    entry.loadShape === record.loadShape &&
    entry.calibrationFingerprint === record.calibrationFingerprint &&
    JSON.stringify(entry.engagedRungs) === JSON.stringify(record.strategy.engagedRungs)
  );
}

function exactPlanEntriesForRecord(calibrationPlan, record) {
  return calibrationPlan.providers.filter((entry) =>
    planEntryMatchesEvidenceRecord(entry, record),
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
  // sc-18864: this read `wiredBytes - reclaimableBytes`. Schema v5 has no `wiredBytes` — it was a
  // copy of `allocatorBytes`, and that subtraction was recovering `activeBytes` the long way round.
  // The mirrored Rust law (`EvidenceRecord::mlx_admission_envelope`) reads the same field, and the
  // arithmetic is unchanged on every committed record.
  const resident = record.observedMemory?.overall?.activeBytes;
  const inputs = [memoryBytes, mlxLimit, wiredLimit, predicted, resident];
  if (!inputs.every((value) => Number.isSafeInteger(value) && value >= 0)) return null;

  const captureHost = BigInt(memoryBytes);
  if (captureHost === 0n) return null;
  const processCeiling = BigInt(Math.min(memoryBytes, mlxLimit, wiredLimit));
  const foreignReserve = captureHost - processCeiling;
  const nonReclaimableWired = BigInt(resident);
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

// The published peak is the ALLOCATOR bound (active + reclaimable), which is what `deviceBytes`
// carried before sc-18864 removed it — so every published cell keeps its exact value. It is a
// footprint figure for the matrix, not a feasibility figure: `mlxRequiredHostBytes` above is the
// one that sizes a host, and it reads the non-reclaimable residency instead.
export function observedPeakBytes(record) {
  const overall = record?.observedMemory?.overall;
  const value = overall?.allocatorBytes ?? overall?.activeBytes;
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

/**
 * The MLX video-route resolver that owns each video family, keyed by family group.
 *
 * One declaration, two consumers: `parseVideoRoutes` iterates it to read the arms, and
 * `videoRouteArmMissing` uses it to name the function that owes a missing one. Restating the list in
 * two places would let the diagnostic point at the wrong file the moment a resolver is renamed.
 *
 * `bernini` (video) sits in IMAGE family 15528 — one engine, one block stack, two catalog entries —
 * so its key is that group's number. It is the only numeric key. Bernini and SCAIL-2 are the two
 * families whose Candle dispatch is bespoke and reuses the same provider id as the MLX resolver,
 * rather than appearing in generic `candle_video_engine_id`.
 */
const VIDEO_ROUTE_RESOLVERS = new Map([
  ["wan-video", { body: "videoRouteWan", fn: "wan_engine_id" }],
  ["ltx-video", { body: "videoRouteLtx", fn: "ltx_engine_id" }],
  ["svd", { body: "videoRouteSvd", fn: "svd_engine_id" }],
  [15528, { body: "videoRouteBernini", fn: "bernini_engine_id" }],
  ["scail2", { body: "videoRouteScail2", fn: "scail2_engine_id" }],
  ["krea-realtime", { body: "videoRouteKreaRealtime", fn: "krea_realtime_engine_id" }],
]);

/** The `*_engine_id` function that should have supplied `modelId`'s provider on `backend`. */
function videoRouteResolverName(modelId, backend) {
  // Bernini has no arm in `candle_video_engine_id` BY DESIGN: `resolve_candle_video_route` matches it
  // off the model id first, so its candle provider is mirrored from the MLX one. A missing candle
  // provider there is therefore a missing `bernini_engine_id` arm, and pointing at the candle file
  // would send a reader to the one place that is correct to be silent.
  const mirroredFromMlx = modelId === "bernini" || modelId === "scail2_14b";
  if (backend !== "mlx" && !mirroredFromMlx) return "candle_video_engine_id";
  const resolver = VIDEO_ROUTE_RESOLVERS.get(familyGroup(modelId));
  // A video family with no row here is itself the defect — say which one, rather than name nothing.
  return (
    resolver?.fn ??
    `the MLX *_engine_id resolver for family ${familyLabel(familyGroup(modelId))}, which VIDEO_ROUTE_RESOLVERS does not declare`
  );
}

function videoRouteArmMissing(modelId, backend, resolved) {
  const served = Object.entries(resolved).map(([lane, engine]) => `${lane}=${engine}`);
  return (
    `${modelId}: the routing catalog routes ${backend}, but no ${backend} provider resolved — ` +
    `expected an arm in ${videoRouteResolverName(modelId, backend)}` +
    (served.length ? ` (resolved only ${served.join(", ")})` : "") +
    ". A cell must not be stamped with the other backend's provider, which would bind its " +
    "calibration evidence, plan row and closure digest to a provider that never ran it, so " +
    "generation stops here (sc-18815)."
  );
}

/**
 * Resolve one catalog entry's route, whichever modality it belongs to.
 *
 * The two lanes are shaped differently and the difference is real, not incidental: the image lane has
 * ONE provider per entry (`MODEL_TABLE` is consulted on both backends), the video lane has one PER
 * BACKEND (LTX is `ltx_2_3` on MLX and `ltx_2_3_distilled` on candle). `engineFor` is what the cell
 * loop consumes, so a cell can never be stamped with the other backend's provider — which would bind
 * its calibration evidence, plan row and closure digest to a provider that never ran it.
 *
 * Throws on an entry with no route at all. An unrouted entry is not a `Missing` cell, it is an entry
 * the generator cannot describe, and silently dropping one is exactly the failure this story exists
 * to remove.
 *
 * Throws, too, on an entry the routing catalog routes on a backend that NO `*_engine_id` arm serves
 * (sc-18815 review). This used to be the quiet path and it was the wrong one to be quiet: deleting
 * LTX's MLX arm from `ltx.rs` left `engineFor("mlx")` returning `null`, the scalar fell back to
 * `video.mlx ?? video.candle`, and every MLX cell was stamped `ltx_2_3_distilled` — CANDLE's provider
 * on an MLX cell, which is the exact substitution this resolver exists to make impossible. Generation
 * still succeeded; the only thing that caught it was JSON-schema validation two lanes downstream,
 * reporting `None is not of type 'string'` and naming neither the resolver nor the cause. So: fail
 * here, name the resolver that owes the arm, and drop the cross-backend scalar fallback entirely so
 * the wrong provider cannot be synthesised even if some future caller skips the check.
 */
function resolveRoute(model, imageRoutes, videoRoutes, backends) {
  const image = imageRoutes.get(model.id);
  if (image) {
    return { ...image, engineFor: () => image.engine };
  }
  const video = videoRoutes.get(model.id);
  if (video) {
    const engineOn = (backend) => {
      const engine = video[backend];
      if (engine) return engine;
      throw new Error(videoRouteArmMissing(model.id, backend, video));
    };
    // Eager, over the backends the CATALOG routes: the failure has to land at build time with the
    // entry named, not lazily at whichever cell happened to be emitted first.
    for (const backend of backends) engineOn(backend);
    const engines = sortedUnique(Object.values(video));
    return {
      // The scalar is the provider on the entry's FIRST ROUTED backend. It deliberately does NOT
      // fall back across backends; the only non-routed case is an entry the catalog routes nowhere
      // (`UNROUTED_CATALOG_ENTRIES`), which emits no cells and whose backends agree by definition.
      engine: video[backends[0]] ?? (engines.length === 1 ? engines[0] : null),
      repo: null,
      kind: "video",
      engineFor: engineOn,
    };
  }
  throw new Error(`${model.id}: no resolved route/provider`);
}

/**
 * The `(sceneworks id -> engine id)` arms one worker video-route resolver declares.
 *
 * The three syntactic forms below are the three the worker actually uses, and each is anchored on the
 * function's own signature so an unrelated `match model` elsewhere in a 90 KB file cannot contribute
 * arms. `&'static str` consts are resolved from the same file (`krea_realtime_engine_id` is spelled
 * entirely in consts), because a parser that silently skipped a const-spelled arm would drop a whole
 * family from the universe and read as "the worker does not route it".
 *
 * Fails closed: an absent function, or a function whose body yields no arm, throws. A video family
 * whose resolver was renamed must be noticed here, not inferred from an empty map.
 */
export function parseVideoEngineIds(source, fnName) {
  const consts = new Map(
    [...source.matchAll(/const\s+([A-Z0-9_]+):\s*&'?\s*(?:static\s+)?str\s*=\s*"([^"]+)"\s*;/g)].map(
      (match) => [match[1], match[2]],
    ),
  );
  const literal = (token) => {
    const quoted = token.match(/^"([^"]+)"$/);
    if (quoted) return quoted[1];
    const resolved = consts.get(token);
    if (!resolved) {
      throw new Error(`memory-matrix: ${fnName} names ${token}, which resolves to no &str const`);
    }
    return resolved;
  };
  const declaration = source.match(
    new RegExp(`fn ${fnName}\\(model: &str\\) -> Option<&'static str> \\{([\\s\\S]*?)\\n\\}`),
  );
  if (!declaration) {
    throw new Error(`memory-matrix: could not locate ${fnName} — the video route resolvers moved`);
  }
  const body = declaration[1];
  const arms = new Map();
  // `match model { "a" => Some("x"), "b" | "c" => Some("y"), … }`. Or-pattern support remains
  // load-bearing for any provider family that maps several catalog ids to one engine; parsing only
  // the final token would silently drop the other ids and report them unrouted.
  for (const arm of body.matchAll(
    /((?:(?:"[^"]+"|[A-Z0-9_]+)\s*\|\s*)*(?:"[^"]+"|[A-Z0-9_]+))\s*=>\s*Some\(("[^"]+"|[A-Z0-9_]+)\)/g,
  )) {
    const engine = literal(arm[2]);
    for (const id of arm[1].split("|")) arms.set(literal(id.trim()), engine);
  }
  // `(model == "a").then_some("x")`
  for (const arm of body.matchAll(
    /model\s*==\s*("[^"]+"|[A-Z0-9_]+)\s*\)\s*\.then_some\(("[^"]+"|[A-Z0-9_]+)\)/g,
  )) {
    arms.set(literal(arm[1]), literal(arm[2]));
  }
  // `matches!(model, "a" | "b").then_some("x")`
  for (const arm of body.matchAll(
    /matches!\(\s*model,([^)]*)\)\s*\.then_some\(("[^"]+"|[A-Z0-9_]+)\)/g,
  )) {
    const engine = literal(arm[2]);
    for (const id of arm[1].matchAll(/"([^"]+)"|([A-Z0-9_]+)/g)) {
      arms.set(literal(id[1] ? `"${id[1]}"` : id[2]), engine);
    }
  }
  if (!arms.size) {
    throw new Error(`memory-matrix: ${fnName} declared no model -> engine arm`);
  }
  return arms;
}

/**
 * The union of engine ids `engines.rs#video_engine_ids` maps each video catalog id to, across
 * backends. Used two ways: as the only source for `wan_2_2_vace_fun_14b` (whose MLX dispatch is a
 * native VACE arm carrying no id string of its own), and as an independent cross-check that every
 * per-backend provider derived from `video_jobs/*` is one the worker's own union agrees is that
 * model's engine. Two sources have to be wrong the same way for a bad provider to reach a cell.
 */
function parseVideoEngineIdUnion(enginesBody) {
  const declaration = enginesBody.match(
    /fn video_engine_ids\(sceneworks_id: &str\) -> &'static \[&'static str\] \{([\s\S]*?)\n\s{4}\}/,
  );
  if (!declaration) {
    throw new Error("memory-matrix: could not locate video_engine_ids in engines.rs");
  }
  const union = new Map();
  for (const arm of declaration[1].matchAll(/((?:"[^"]+"\s*\|\s*)*"[^"]+")\s*=>\s*&\[([^\]]*)\]/g)) {
    const engines = [...arm[2].matchAll(/"([^"]+)"/g)].map((entry) => entry[1]);
    for (const id of arm[1].matchAll(/"([^"]+)"/g)) union.set(id[1], engines);
  }
  if (!union.size) throw new Error("memory-matrix: video_engine_ids declared no arm");
  return union;
}

/**
 * Ids whose MLX provider may come from `engines.rs#video_engine_ids` ALONE (sc-18815 review).
 *
 * `wan_2_2_vace_fun_14b` is the real case: its MLX dispatch is the native VACE arm, which carries no
 * id string, so there is no `*_engine_id` arm to read and the union is the only source.
 *
 * The allowlist exists because the fallback was written for exactly that entry but keyed on
 * `declared.length === 1`, which is a property thousands of ids could satisfy. `mochi_1` already took
 * it — acquiring an MLX provider derived from `mochi.rs`, a file this generator neither reads nor
 * fingerprints, so the provider was outside the staleness tripwire. Inert today (mochi has no
 * manifest entry and so is outside the universe), but it silently downgraded the "two independent
 * declarations must agree" invariant to single-source for whichever ids happened to hit it. Now a new
 * one must be named here, with the reason, before it can.
 */
export const UNION_ONLY_MLX_ROUTES = new Set(["wan_2_2_vace_fun_14b"]);

/**
 * Per-backend video providers, derived from the worker's own route resolvers.
 *
 * The image lane resolves ONE provider per catalog id, because `MODEL_TABLE` is one table consulted
 * on both backends. The video lane genuinely does not: LTX is backend-split (`ltx_2_3` on MLX,
 * `ltx_2_3_distilled` on candle), so a single scalar route would have to be wrong on one of them —
 * and a wrong provider is not cosmetic, it is the key calibration evidence, the plan, and the
 * per-provider closure digests all bind on.
 *
 * Backend membership is NOT decided here. This answers "which provider serves this id on this
 * backend if that backend serves it at all"; the routing catalog (`routedLanes`) decides whether it
 * does. SCAIL-2's Candle `ReplacePersonScail2` dispatch is bespoke and is mirrored explicitly below;
 * Krea-Realtime remains genuinely MLX-only. The catalog wins, per this epic's stated backend
 * authority.
 */
export function parseVideoRoutes(bodies, unionOnlyMlxRoutes = UNION_ONLY_MLX_ROUTES) {
  const union = parseVideoEngineIdUnion(bodies.engines);
  const mlx = new Map();
  for (const { body, fn } of VIDEO_ROUTE_RESOLVERS.values()) {
    for (const [id, engine] of parseVideoEngineIds(bodies[body], fn)) mlx.set(id, engine);
  }
  const candle = parseVideoEngineIds(bodies.videoRouteCandle, "candle_video_engine_id");
  // These lanes route off the model id in `resolve_candle_video_route`, BEFORE the generic
  // `is_candle_video_engine` arm, so they are absent from `candle_video_engine_id` by design and
  // share the MLX resolver's engine id. Reading only the generic function would leave a provider
  // hole on a shipping lane.
  for (const id of ["bernini", "scail2_14b"]) {
    if (mlx.has(id)) candle.set(id, mlx.get(id));
  }

  const routes = new Map();
  for (const id of sortedUnique([...mlx.keys(), ...candle.keys(), ...union.keys()])) {
    const byBackend = {};
    if (mlx.has(id)) byBackend.mlx = mlx.get(id);
    if (candle.has(id)) byBackend.candle = candle.get(id);
    const declared = union.get(id);
    if (declared) {
      // `wan_2_2_vace_fun_14b`'s MLX arm is the native VACE dispatch, which carries no id string, so
      // the union is its only source. Everything else must AGREE with the union.
      if (!byBackend.mlx && declared.length === 1 && unionOnlyMlxRoutes.has(id)) {
        byBackend.mlx = declared[0];
      }
      for (const [backend, engine] of Object.entries(byBackend)) {
        if (!declared.includes(engine)) {
          throw new Error(
            `memory-matrix: video route ${id}:${backend} resolves to provider ${engine}, but ` +
              `engines.rs#video_engine_ids lists ${declared.join(",")} for it — the worker's two ` +
              "video route declarations disagree",
          );
        }
      }
    }
    if (Object.keys(byBackend).length) routes.set(id, byBackend);
  }
  return routes;
}

// sc-16268: anchored on CODE, not on a doc comment. Provenance hashes these sources with inert
// comments stripped, so a parse that reads comment text would let a semantic change slip past the
// staleness tripwire. SC-18816 deliberately parses the broader staged-residency sweep rather than
// the selectable-Sequential sweep: unconditional staging is real rung-1 availability even though
// it does not authorize setting OffloadPolicy::Sequential.
function parseMlxStagedResidencyEngines(source) {
  const test = source.match(
    /fn engine_engages_staged_residency_is_derived_from_the_registered_capability\(\)\s*\{([\s\S]*?)assert!\(!engine_engages_staged_residency/,
  );
  if (!test) {
    throw new Error("could not locate the MLX staged-residency registry sweep");
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
  // The converter-packed tier sets are shared with the projector's `catalogAxes` so the matrix's
  // cell universe and the projection's declaration universe cannot disagree about them (sc-21510).
  return new Map([["instantid_realvisxl:candle", [candleDense]], ...CONVERTER_TIER_OVERRIDES]);
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

function staticContractCoversProvider(contract, provider) {
  if (!contract) return false;
  return contract.implementations.some(
    (implementation) => (implementation.runtimeProvider ?? contract.provider) === provider,
  );
}

export function declarationModelForCoordinate({ backend, rung, route, provider, model, tier, mode, overlay, manifestById }) {
  const routeLocalContract = model[backend]?.memoryStrategyContract;
  const routeLocalImplementation = routeLocalContract?.implementations?.some(
    (implementation) =>
      (implementation.runtimeProvider ?? routeLocalContract.provider) === provider &&
      implementation.rung === rung &&
      implementation.tiers.includes(tier) &&
      implementation.modes.includes(mode) &&
      implementation.overlays.includes(overlay),
  );
  if (routeLocalImplementation) return model;
  return model.id === "z_image_edit" && route.engine === "z_image_turbo"
    ? manifestById.get("z_image_turbo")
    : model;
}

export function providerFor(model, backend, overlay, route, mode) {
  // Per-backend (sc-18815): the image lane's `engineFor` returns the one table route on either
  // backend, so this is unchanged for it, while a video entry gets the provider that backend
  // actually loads instead of whichever one happened to be listed first. No `?? route.engine`
  // fallback: `engineFor` throws on a backend it cannot serve, and falling through to the scalar is
  // how a candle provider reached an MLX cell in the first place.
  //
  // A contract row that names a DISTINCT `runtimeProvider` for this (mode, overlay) owns the lane
  // (sc-21510). This used to hold only for the `control` overlay, so Krea's and FLUX.2 Klein's
  // route-local EDIT providers (`krea_2_edit`, `flux2_klein_9b_edit`, …) could never label a cell:
  // every edit/character cell carried the base engine id, the edit rows bound to nothing, and their
  // real, witnessed capability read as `declared_cell_absent` drift. The mode filter keeps the base
  // rows in charge of the modes they declare; a coordinate no row covers falls through to the base
  // engine exactly as before.
  const engine = route.engineFor(backend);
  const contract = model[backend]?.memoryStrategyContract;
  const declared = [...new Set((contract?.implementations ?? [])
    .filter((implementation) =>
      implementation.overlays?.includes(overlay) &&
      (mode === undefined || implementation.modes?.includes(mode)))
    .map((implementation) => implementation.runtimeProvider ?? contract.provider))];
  if (declared.length === 1) return declared[0];
  if (declared.length > 1) {
    throw new Error(
      `${model.id}:${backend}:${mode ?? "(any mode)"}:${overlay} declares ambiguous runtime providers: ${declared.join(", ")}`,
    );
  }
  return overlay === "control"
    ? (CONTROL_PROVIDER_OVERRIDES.get(`${model.id}:${backend}`) ?? engine)
    : engine;
}

function matrixOverlayFor(recordOverlay) {
  return /^control:\d+$/.test(recordOverlay) ? "control" : recordOverlay;
}

function overlaysFor(model, backend, route) {
  // Some public variants always provision a built-in LoRA. When the backend's exact contract
  // declares only the LoRA load profile for the model's base provider, publishing `none` creates
  // plain cells that cannot reach the provider and lets their staged/decode/attention/resident rows
  // look applicable by accident. One shared STRUCTURAL predicate (sc-20799) — `catalogAxes` mirrors
  // this same call.
  const overlays = contractIsLoraOnly(model, backend, route.engineFor(backend))
    ? ["lora"]
    : ["none"];
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
    // sc-18815: the TEMPORAL half of a video entry's declared envelope. Publishing only the spatial
    // half would say the envelope is fully described when the axis a video peak actually scales on
    // is missing — the same silent omission, one field down. These are the catalog's DECLARED
    // limits, nothing more: how the phase curve represents the temporal axis (raw frames, a
    // latent-depth regressor, a cross term) is measured and decided by sc-18810/sc-18812, and this
    // field neither anticipates nor constrains that. Image entries declare none of these keys, so
    // the filter below drops them and every image envelope is byte-identical to before.
    defaultDuration: defaults.duration ?? null,
    durations: Array.isArray(limits.durations) ? limits.durations : [],
    hardMaxDuration: limits.hardMaxDuration ?? null,
    defaultFps: defaults.fps ?? null,
    fps: Array.isArray(limits.fps) ? limits.fps : [],
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
export const RUNG4_CONTRACT_SUPPORT = Object.freeze([
  "implemented",
  "missing",
  "structurally-not-applicable",
]);

/** A git sha, abbreviated no shorter than 9 hex characters. See `contractRevision` below. */
export const RUNG4_CONTRACT_REVISION_PATTERN = /^[0-9a-f]{9,40}$/;

/**
 * The published matrix cell's `state` vocabulary, restated once for the out-of-matrix records
 * (sc-17153). An out-of-matrix family cannot hold a cell, but it CAN and MUST carry the cell's two
 * distinct claims — `state` (the rung WORKS: a code claim) and `memoryCharacterization` (the rung's
 * PEAKS are known: an evidence claim) — because collapsing them is exactly the SC-16060 failure the
 * cell vocabulary exists to prevent. Consumers classify `state` through `isImplemented()`, never by
 * string comparison; `parseOutOfMatrixRung4Families` enforces both claims below.
 */
export const OUT_OF_MATRIX_CELL_STATES = Object.freeze([
  "Implemented/unverified",
  "Runtime verified",
  "Verified",
  "Missing",
  "Structurally N/A",
]);

/**
 * The prerequisite claim this survey may not carry again (sc-18664).
 *
 * Rung 4's only shared prerequisite is `LoadShape::DeferredMaterialization`
 * (`BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES`, inference
 * `crates/contracts/gen-core/src/memory_strategy.rs:310-313` at the pinned f17c82544). SC-15998
 * removed the rung-1 edge the
 * survey's notes asserted, because it had encoded one provider's coupled loader shape as universal
 * arithmetic.
 *
 * Scoped to `notes` deliberately, and the scope is the whole point. A PROVIDER may still append its
 * own `BoundedTransformerResidency -> StagedResidency` edge through `additional_prerequisites`, and
 * mlx-gen-anima and mlx-gen-chroma do — so the same sentence is TRUE inside those families' entries
 * and false as a blanket note. A document-wide ban would have forced a rewrite of a correct
 * provider-specific verdict, which is exactly what sc-18664 was told not to do.
 */
export const STALE_RUNG1_PREREQUISITE_PATTERNS = Object.freeze([
  /rung 4 requires rung 1/i,
  /requires rung 1 engaged in the same request/i,
]);

/**
 * The excision that lets the ban scan the file the ban is written in.
 *
 * The pattern literals above necessarily contain the banned text, so a self-scan would always fire.
 * This names the declaration and removes it first. It fails CLOSED on purpose: rename or reshape
 * that constant and this stops matching, the literals stay in the scanned text, and the guard
 * throws rather than going quietly green.
 */
const STALE_RUNG1_PATTERN_DECLARATION =
  /export const STALE_RUNG1_PREREQUISITE_PATTERNS = Object\.freeze\(\[[\s\S]*?\]\);/;

/**
 * The generator's OWN prose, held to the same ban as the survey's notes (sc-18664).
 *
 * One rule, two halves. `assertRung4NotesRecordTheContractPrerequisite` bans the removed rung-1
 * prerequisite claim from the survey's `notes`; the identical sentence also sat in this file's
 * `stagedResidencyIsAvailable` docstring — the more authoritative of the two sites, because it is
 * where a reader of the gate itself lands. Banning a sentence in the data file while leaving it in
 * the code that reads the data file is the "one half of a pair moved" defect this epic keeps
 * finding, so both halves run off the SAME `STALE_RUNG1_PREREQUISITE_PATTERNS` constant and a
 * pattern added for either covers the other for free.
 *
 * Doc comments WRAP, and that is not a detail: the removed claim was invisible to a line-oriented
 * grep of this very file from the day it was written (cb1593ac, sc-15969) until a human read it,
 * because `engaged in the same` and `request` sat on different lines. The source is therefore
 * flattened across comment continuations before matching, which is the only reason this guard sees
 * what a grep could not.
 *
 * Scoped to this file and deliberately NOT to `generate-memory-matrix.test.mjs`, which feeds the
 * banned sentence in as mutation INPUT — that is what proves the notes guard bites, and banning it
 * there would delete the evidence.
 */
export function assertGeneratorSourceDoesNotRestateTheRemovedEdge(source) {
  const scanned = source
    .replace(STALE_RUNG1_PATTERN_DECLARATION, "")
    .replaceAll(/\n[ \t]*\*[ \t]?/g, " ");
  for (const pattern of STALE_RUNG1_PREREQUISITE_PATTERNS) {
    if (pattern.test(scanned)) {
      throw new Error(
        `scripts/generate-memory-matrix.mjs restates the rung-1 prerequisite SC-15998 removed (${pattern}). BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES is LoadShape::DeferredMaterialization and nothing else; the generator's rung-1 gate is a proxy for the edge individual providers append through additional_prerequisites, not the shared contract rule (sc-18664)`,
      );
    }
  }
}

const GENERATOR_SOURCE_PATH = fileURLToPath(import.meta.url);
let generatorSourceCache = null;
const generatorSourceText = () =>
  (generatorSourceCache ??= readFileSync(GENERATOR_SOURCE_PATH, "utf8"));

/**
 * The notes' half of the correction: the stale claim is gone AND the real one is stated.
 *
 * Both directions, because an absence check alone passes on notes that say nothing at all — the
 * quiet outcome that would let the next reader re-derive the removed edge from the silence.
 */
export function assertRung4NotesRecordTheContractPrerequisite(notes) {
  const body = (notes ?? []).join("\n");
  for (const pattern of STALE_RUNG1_PREREQUISITE_PATTERNS) {
    if (pattern.test(body)) {
      throw new Error(
        `rung-4 survey notes restate the rung-1 prerequisite SC-15998 removed (${pattern}). Rung 4 requires LoadShape::DeferredMaterialization and nothing else; a provider's own additional_prerequisites edge belongs in that family's entry, not in the shared notes (sc-18664)`,
      );
    }
  }
  for (const required of ["LoadShape::DeferredMaterialization", "SC-15998"]) {
    if (!body.includes(required)) {
      throw new Error(
        `rung-4 survey notes must name ${required}: the correction is a positive statement of rung 4's sole shared prerequisite, not just the absence of the old one (sc-18664)`,
      );
    }
  }
}

/**
 * The family verdict as a FUNCTION of the per-stack verdicts, so it cannot be asserted independently.
 *
 * This is the survey's own `structuralApplicability` vocabulary restated as arithmetic: `full` is one
 * uniform stack that a single BlockPlan bounds with only embedders and heads outside it; `partial` is
 * a trunk that decomposes into two or more separately-indexed stacks needing a plan each, or that
 * carries a remainder which cannot be windowed at all; `none` is no windowable stack anywhere.
 *
 * The restatement is exact for 38 of the 40 verdicts `families` already carries, and NOT for the
 * other two — say so rather than claim it is simply the vocabulary. Qwen-Image (`families.15511`)
 * records `full` on both backends while holding two `blockStacks`, because the second is
 * control-route-only (`entries: ["qwen_image_control"]`) and is not resident on the routes the
 * verdict covers. This function has no notion of route scoping and derives `partial` there.
 *
 * Route scoping is deliberately NOT added, and the reason is that no record needs it to reach the
 * right answer. MiniMax-H3 has its own route-conditional stack — `text_encoder.vision_tower`, the
 * `fl2va` keyframe path only — and is `partial` under either rule, because its two conv stacks and
 * its several separately-indexed denoise stacks each force `partial` on their own. Adding a route
 * axis to serve zero consumers would be speculation; the exception is recorded here and pinned by
 * the test that computes Qwen-Image's derivation and asserts it disagrees with the record.
 */
export function deriveOutOfMatrixApplicability(stacks) {
  const windowable = stacks.filter((stack) => stack.structuralApplicability !== "none");
  if (!windowable.length) return "none";
  const remainder = stacks.length - windowable.length;
  return windowable.length === 1 && remainder === 0 && windowable[0].structuralApplicability === "full"
    ? "full"
    : "partial";
}

/**
 * Validate the surveyed families the MATRIX cannot carry a verdict for (sc-18664).
 *
 * `families` is fenced to the image catalog in both directions — the generator builds from
 * `manifest.models.filter(type === "image")`, and `assertRung4SurveyCoversEveryFamily` rejects a
 * survey key the catalog does not advertise — so a video family placed there fails generation rather
 * than producing a row. MiniMax-H3 is the first family to hit that: it is `"type": "video"`, and
 * `familyGroup` throws on `minimax_h3`, so it has no group key either.
 *
 * These records are therefore validated here and never published. The guard that keeps them from
 * becoming a museum is the catalog check: the day `familyGroup` learns one of the named entries, this
 * throws and the record has to move into `families`, where the coverage fence can see it.
 */
export function parseOutOfMatrixRung4Families(parsed, { familyGroups } = {}) {
  const records = new Map();
  const families = parsed.families ?? {};
  for (const [group, family] of Object.entries(parsed.outOfMatrixFamilies ?? {})) {
    const where = `rung-4 survey out-of-matrix ${family.name ?? group}`;
    if (Object.hasOwn(families, group)) {
      throw new Error(
        `${where}: SC-${group} is also a \`families\` key — one family gets one verdict, in one place`,
      );
    }
    if (!family.catalogEntries?.length) {
      throw new Error(`${where}: must name the catalog entries it is a survey OF`);
    }
    if (familyGroups) {
      for (const id of family.catalogEntries) {
        let owner = null;
        try {
          owner = familyGroups(id);
        } catch {
          owner = null;
        }
        if (owner !== null) {
          throw new Error(
            `${where}: familyGroup now resolves ${id} to family SC-${owner}, so the matrix carries this lane — move the record into \`families\` where the coverage fence can see it (sc-18664)`,
          );
        }
      }
    }
    for (const [backend, verdict] of Object.entries(family.backends ?? {})) {
      const at = `${where} (${backend})`;
      if (!RUNG4_APPLICABILITIES.includes(verdict.structuralApplicability)) {
        throw new Error(
          `${at}: unknown structuralApplicability ${JSON.stringify(verdict.structuralApplicability)}`,
        );
      }
      if (!RUNG4_IMPLEMENTATIONS.includes(verdict.implementation)) {
        throw new Error(`${at}: unknown implementation ${JSON.stringify(verdict.implementation)}`);
      }
      if (!RUNG4_REQUEST_PEAKS.includes(verdict.requestPeak?.finding)) {
        throw new Error(
          `${at}: unknown requestPeak finding ${JSON.stringify(verdict.requestPeak?.finding)}`,
        );
      }
      if (!verdict.evidence?.length) {
        throw new Error(`${at}: a verdict derived from provider code must cite at least one source`);
      }
      // ...and must say which tree those citations resolve in. This is the one thing an
      // out-of-matrix record needs that a `families` verdict does not: `families` is surveyed
      // against the crates the Cargo pin already carries, so `generatedFrom.inferenceRevision`
      // dates every path in it. These records are surveyed at a revision of their own —
      // MiniMax-H3's MLX paths resolve at e09f46aaf and its Candle ones at 79f02e6d0, two DIFFERENT
      // trees, and neither is the pin. What the field asserts has moved since it was introduced: at
      // the pinned f17c82544 both H3 crates exist and both anchors are ANCESTORS of it, so the
      // field now reads "last surveyed here, not re-surveyed since" rather than "the pinned tree
      // has no such crate yet". Without it the record silently mixes revisions and reads as if the
      // matrix's own pin dated it (sc-18664; meaning restated sc-18650 pre-merge review).
      if (!RUNG4_CONTRACT_REVISION_PATTERN.test(verdict.contractRevision ?? "")) {
        throw new Error(
          `${at}: contractRevision must name the inference revision this record's evidence paths resolve at, as a git sha of at least 9 hex characters — got ${JSON.stringify(verdict.contractRevision)}. An out-of-matrix record is surveyed from crates the matrix's own pinned revision need not contain, so without it the record mixes two trees (sc-18664)`,
        );
      }
      const stacks = verdict.stacks ?? [];
      if (!stacks.length) {
        throw new Error(
          `${at}: the family verdict is derived from per-stack verdicts, so the stacks have to be there`,
        );
      }
      const seen = new Set();
      for (const stack of stacks) {
        const stackAt = `${at}: stacks[${JSON.stringify(stack.id)}]`;
        if (!stack.id || seen.has(stack.id)) {
          throw new Error(`${stackAt}: every stack needs its own id, and ids must not repeat`);
        }
        seen.add(stack.id);
        if (!RUNG4_APPLICABILITIES.includes(stack.structuralApplicability)) {
          throw new Error(
            `${stackAt}: unknown structuralApplicability ${JSON.stringify(stack.structuralApplicability)}`,
          );
        }
        if (!stack.reason) {
          throw new Error(`${stackAt}: a per-stack verdict without a stated reason is an assertion`);
        }
        if (stack.windowable !== (stack.structuralApplicability !== "none")) {
          throw new Error(
            `${stackAt}: windowable ${JSON.stringify(stack.windowable)} contradicts structuralApplicability ${JSON.stringify(stack.structuralApplicability)}`,
          );
        }
        if (stack.structuralApplicability === "none" && !stack.structural?.length) {
          throw new Error(
            `${stackAt}: a stack the rung cannot bound is a Structurally N/A claim, which the epic accepts only with static provider evidence — none is cited`,
          );
        }
      }
      const derived = deriveOutOfMatrixApplicability(stacks);
      if (derived !== verdict.structuralApplicability) {
        throw new Error(
          `${at}: records ${verdict.structuralApplicability} but its stacks derive ${derived} — the family verdict follows from the stacks, it is not a separate claim`,
        );
      }
      // AC2's second half: a `partial` family has to say WHICH stacks are the no, or the verdict
      // reads as a shrug. Exact set equality in both directions, so a stale name cannot survive a
      // stack being reclassified.
      const unwindowable = stacks
        .filter((stack) => stack.structuralApplicability === "none")
        .map((stack) => stack.id)
        .sort();
      const declared = [...(verdict.nonWindowableStacks ?? [])].sort();
      if (JSON.stringify(declared) !== JSON.stringify(unwindowable)) {
        throw new Error(
          `${at}: nonWindowableStacks is ${JSON.stringify(declared)} but the stacks the rung cannot bound are ${JSON.stringify(unwindowable)}`,
        );
      }
      if (!RUNG4_CONTRACT_SUPPORT.includes(verdict.contractSupport)) {
        throw new Error(
          `${at}: unknown contractSupport ${JSON.stringify(verdict.contractSupport)} — it mirrors gen_core::MemoryStrategySupport for this provider`,
        );
      }
      // The survey and the provider contract have to agree, in BOTH directions. A survey that says
      // the rung applies while the contract declares the architecture lacks what it optimizes is a
      // contradiction; a survey that says it applies while the contract merely has not built it is
      // the ordinary case, and needs the reason on the record so the gap is not read as a defect.
      if (verdict.contractSupport === "structurally-not-applicable" && derived !== "none") {
        throw new Error(
          `${at}: the contract declares rung 4 StructurallyNotApplicable while this survey names a windowable stack — one of the two is wrong`,
        );
      }
      if (verdict.contractSupport === "implemented" && derived === "none") {
        throw new Error(
          `${at}: the contract declares rung 4 Implemented while this survey finds no windowable stack — one of the two is wrong`,
        );
      }
      if (verdict.contractSupport !== "implemented") {
        if (verdict.implementation !== "none") {
          throw new Error(
            `${at}: claims implementation ${JSON.stringify(verdict.implementation)} while the contract does not declare rung 4 Implemented`,
          );
        }
        if (derived !== "none" && !(verdict.contractReason && verdict.contractSource)) {
          throw new Error(
            `${at}: rung 4 applies to a stack but the contract does not implement it — record contractSource and contractReason, or the gap reads as an unexplained hole`,
          );
        }
      }
      // sc-17153 — the record carries the SAME two distinct claims a published matrix cell does,
      // in the cell vocabulary, so the day this family gains a lane the move into `families` is a
      // relocation rather than a re-derivation. Both claims validated on every generation:
      //
      //   `state`                  — the CODE claim. Classified through the shared
      //                              `isImplemented()` predicate, never by string comparison, and
      //                              required to AGREE with `contractSupport` through that
      //                              predicate: an `Implemented/unverified` record whose contract
      //                              does not implement the rung (or the reverse) is a
      //                              contradiction, not a nuance.
      //   `memoryCharacterization` — the PEAKS claim. `status`, `coveredPixelBound` and (on a
      //                              temporal record) `coveredFrameBound` are DERIVED from
      //                              `measuredGeometries` by the shared
      //                              `memoryCharacterization()` helper, exactly as the family
      //                              verdict is derived from its stacks above — asserting them
      //                              independently is how a record drifts. A characterization on a
      //                              non-implemented state is refused, mirroring the published
      //                              cell rule (`memoryCharacterization is … on … measured
      //                              geometries` in `validateMatrix`).
      if (!OUT_OF_MATRIX_CELL_STATES.includes(verdict.state)) {
        throw new Error(
          `${at}: unknown state ${JSON.stringify(verdict.state)} — an out-of-matrix record carries the matrix cell's own state vocabulary (sc-17153)`,
        );
      }
      if (isImplemented(verdict.state) !== (verdict.contractSupport === "implemented")) {
        throw new Error(
          `${at}: state ${JSON.stringify(verdict.state)} disagrees with contractSupport ${JSON.stringify(verdict.contractSupport)} under isImplemented() — the two claims describe one contract`,
        );
      }
      const characterization = verdict.memoryCharacterization;
      if (
        !characterization ||
        typeof characterization.reason !== "string" ||
        characterization.reason.length === 0
      ) {
        throw new Error(
          `${at}: memoryCharacterization with a non-empty reason is required — the record must say what is measured and what the terminal campaign still owes (sc-17153)`,
        );
      }
      // sc-18663. The comparison is on JSON, and the derived side is `sortedUnique`, so ORDER and
      // duplicates are part of the claim. Say that here rather than let the derivation message below
      // report two arrays that read as identical — that diagnostic sent a reader looking for a
      // status bug when the record merely listed its geometries out of order.
      const measuredGeometries = characterization.measuredGeometries ?? [];
      if (
        !Array.isArray(measuredGeometries) ||
        JSON.stringify(measuredGeometries) !== JSON.stringify(sortedUnique(measuredGeometries))
      ) {
        throw new Error(
          `${at}: measuredGeometries ${JSON.stringify(measuredGeometries)} must be an array that is sorted and duplicate-free — the derivation compares it against sortedUnique(...), so ordering and repeats are part of the claim (sc-18663)`,
        );
      }
      // sc-18663. `memoryCharacterization()` emits a FOURTH key, `coveredFrameBound`, as soon as the
      // form is temporal — any `WxHxfF` geometry. Projecting only the three image-shaped keys made
      // the comparison unsatisfiable for every multi-frame record: the derived object always carried
      // a key the declared one could not, so a video family could not record temporal coverage at
      // all, which is exactly what epic 17137's terminal campaign has to record.
      //
      // The temporal form is INFERRED from the geometries rather than read from a record field, and
      // that is the whole point of the check: a `declaresTemporalCurve`-style field on the record
      // would be a SECOND independent claim — settable to `false` to relax the grading — in the one
      // place whose job is to make the characterization a function of `measuredGeometries` alone. A
      // published cell can read the flag from the manifest because the manifest's curve is not the
      // cell's own assertion; an out-of-matrix record has no such external declaration to read.
      // (Consequence, stated rather than left implicit: a record measured ONLY at single-frame
      // geometries is graded against two coefficients even for a video family. Measurements can only
      // ADD the temporal axis, so a real video capture — which always carries frames — is graded
      // against three.)
      const derivedCharacterization = memoryCharacterization(measuredGeometries);
      const declaredCharacterization = {
        status: characterization.status,
        measuredGeometries: characterization.measuredGeometries,
        coveredPixelBound: characterization.coveredPixelBound,
        // Present on both sides or neither: an omitted bound leaves the key out and mismatches the
        // derived `null`, and a bound smuggled onto a record whose geometries carry no frame axis
        // adds a key the derived object does not have. Both are the same defect — a bound that does
        // not follow from the geometries — and both fail below.
        ...(Object.hasOwn(derivedCharacterization, "coveredFrameBound") ||
        Object.hasOwn(characterization, "coveredFrameBound")
          ? { coveredFrameBound: characterization.coveredFrameBound }
          : {}),
      };
      if (JSON.stringify(declaredCharacterization) !== JSON.stringify(derivedCharacterization)) {
        throw new Error(
          `${at}: memoryCharacterization ${JSON.stringify(declaredCharacterization)} does not derive from its measuredGeometries (${JSON.stringify(derivedCharacterization)}) — status, coveredPixelBound and coveredFrameBound follow from the geometries, they are not separate claims`,
        );
      }
      if (characterization.status !== "unmeasured" && !isImplemented(verdict.state)) {
        throw new Error(
          `${at}: memoryCharacterization is ${characterization.status} while the state is not implemented — a rung that does not work cannot have characterized peaks`,
        );
      }
      records.set(`${group}:${backend}`, verdict);
    }
  }
  return records;
}

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
export function parseRung4Survey(body, { familyGroups, generatorSource } = {}) {
  const parsed = JSON.parse(body);
  const families = parsed.families;
  if (!families || typeof families !== "object") {
    throw new Error("rung-4 survey: missing `families`");
  }
  // sc-18664. All three run on every generation, which is what gives them reach: `--check` in CI and
  // the pre-push hook go through here, so a stale prerequisite note, the same claim restated in the
  // generator's own prose, or an unvalidated out-of-matrix record fails the same way a bad family
  // verdict does. The source scan sits HERE, next to the notes scan it shares a pattern set with,
  // so the two halves of the one ban cannot be wired into different code paths.
  assertRung4NotesRecordTheContractPrerequisite(parsed.notes);
  assertGeneratorSourceDoesNotRestateTheRemovedEdge(generatorSource ?? generatorSourceText());
  parseOutOfMatrixRung4Families(parsed, { familyGroups });
  const survey = new Map();
  const unroutedBackends = new Map();
  const modalityRelationships = new Map();
  for (const [group, family] of Object.entries(families)) {
    for (const [backend, declaration] of Object.entries(family.unroutedBackends ?? {})) {
      const at = `rung-4 survey ${family.name ?? group} unrouted ${backend}`;
      if (!["mlx", "candle"].includes(backend)) {
        throw new Error(`${at}: backend is outside the survey vocabulary`);
      }
      if (family.backends?.[backend]) {
        throw new Error(`${at}: the same backend has both a routed verdict and an unrouted declaration`);
      }
      if (typeof declaration.reason !== "string" || declaration.reason.length < 20) {
        throw new Error(`${at}: reason must explain why the backend is not routed`);
      }
      if (!Number.isInteger(declaration.owningStory)) {
        throw new Error(`${at}: owningStory must be a Shortcut story id`);
      }
      if (
        Object.hasOwn(VIDEO_FAMILY_STORIES, group) &&
        declaration.owningStory !== VIDEO_FAMILY_STORIES[group]
      ) {
        throw new Error(
          `${at}: owningStory sc-${declaration.owningStory} is not family ${group}'s owner sc-${VIDEO_FAMILY_STORIES[group]}`,
        );
      }
      if (
        !declaration.evidence?.length ||
        declaration.evidence.some(
          (item) =>
            typeof item.source !== "string" ||
            !/:\d/.test(item.source) ||
            typeof item.reason !== "string" ||
            item.reason.length < 20,
        )
      ) {
        throw new Error(`${at}: an unrouted declaration must cite routing/provider evidence`);
      }
      unroutedBackends.set(`${group}:${backend}`, declaration);
    }
    if (family.modalityRelationship) {
      const relationship = family.modalityRelationship;
      const at = `rung-4 survey ${family.name ?? group} modalityRelationship`;
      if (typeof relationship.reason !== "string" || relationship.reason.length < 20) {
        throw new Error(`${at}: reason must explain the cross-modality relationship`);
      }
      if (
        !relationship.evidence?.length ||
        relationship.evidence.some(
          (item) =>
            typeof item.source !== "string" ||
            !/:\d/.test(item.source) ||
            typeof item.reason !== "string" ||
            item.reason.length < 20,
        )
      ) {
        throw new Error(`${at}: the relationship must cite catalog/provider evidence`);
      }
      if (typeof relationship.sharedProviderContract !== "boolean") {
        throw new Error(`${at}: sharedProviderContract must state whether provider truth is shared`);
      }
      const entries = Object.entries(relationship.entries ?? {});
      if (
        !relationship.entries?.image?.length ||
        !relationship.entries?.video?.length ||
        entries.some(([modality, ids]) => !["image", "video"].includes(modality) || !ids?.length)
      ) {
        throw new Error(`${at}: entries must name at least one image and one video catalog id`);
      }
      for (const [modality, ids] of entries) {
        for (const id of ids) {
          let owner = null;
          try {
            owner = familyGroups?.(id);
          } catch {
            owner = null;
          }
          if (String(owner) !== group) {
            throw new Error(`${at}: ${modality} entry ${id} belongs to another family`);
          }
        }
      }
      modalityRelationships.set(group, relationship);
    }
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
          // Compared as strings: image family groups are numbers and the video groups sc-18815 adds
          // are names, while a JSON object key is always a string.
          if (String(owner) !== group) {
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
  // Maps may carry metadata without changing their key/value iteration contract. Keeping these
  // declarations attached to the parsed survey means coverage validation cannot accidentally read
  // verdicts from one parse and topology declarations from another.
  survey.unroutedBackends = unroutedBackends;
  survey.modalityRelationships = modalityRelationships;
  // sc-18663: ONE computation of the out-of-matrix catalog entries, attached to the parse that
  // validated the records they come from. Two consumers read it — `buildMatrix` subtracts these
  // entries from the coordinate universe, and `assertCalibrationPlanTargetsResolvedCoordinates`
  // exempts plan rows that target them — and a second, independently spelled set is exactly how an
  // exemption would outlive the subtraction it is the complement of.
  survey.outOfMatrixCatalogEntries = new Set(
    Object.values(parsed.outOfMatrixFamilies ?? {}).flatMap((family) => family.catalogEntries ?? []),
  );
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
  const group = familyGroup(modelId);
  const verdict = survey.get(`${group}:${backend}`);
  if (!verdict) {
    // sc-18815: the modality is admitted before every family's verdict is written, so the honest
    // report is "not surveyed yet, and here is who owes it" — NOT a throw (the entry would vanish
    // from the matrix again) and NOT a bare `Missing` (which is the shape of a surveyed family found
    // to have no implementation). `surveyed: false` is the discriminator, and it is present on every
    // rung-4 cell so a consumer never has to infer it from a null.
    const owner = PENDING_RUNG4_SURVEYS.get(group);
    if (!owner) throw new Error(`${modelId}:${backend}: no rung-4 survey verdict (SC-15969)`);
    return {
      story: 15969,
      surveyed: false,
      pendingSurveyStory: owner,
      structuralApplicability: null,
      requestPeak: "unsurveyed",
      implementation: null,
      overlayIncompatible,
    };
  }
  const scopedRequestPeak = (verdict.requestPeak.scopes ?? []).find(
    (scope) =>
      (scope.entries ?? [modelId]).includes(modelId) &&
      (scope.tiers ?? [tier]).includes(tier) &&
      (scope.modes ?? [mode]).includes(mode) &&
      (scope.overlays ?? [overlay]).includes(overlay),
  );
  return {
    story: 15969,
    surveyed: true,
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
 *
 * `catalogFamilyBackends` (sc-18813) is the third state that direction needs. The universe this
 * generator builds is still `type === "image"` only until sc-18815 admits video, while admitting a
 * modality requires its survey verdicts to already exist — `rung4SurveyCell` destructures
 * `survey.get(key)` for every rung-4 cell, and the catalog -> survey direction above throws first.
 * A verdict written ahead of that admission therefore hits the arm above. That is an artifact of the
 * SLICING, not a structural necessity: one change carrying the admission and the verdict together
 * satisfies both arms and needs no third state, and once the family is in `advertised` the strict
 * arm covers it with no exemption at all. What the third state buys is landing the survey on its own,
 * so it can be reviewed on its own. So a verdict may run AHEAD of admission. It may not run ahead of
 * the ROUTING CATALOG: the tolerated set is the family/backend pairs the routing catalog actually
 * routes (`routedLanes` over `MODEL_CAPS`/`VIDEO_MODEL_CAPS`), not an unchecked exemption. A typo'd
 * group, a backend the catalog does not route, or a verdict for a family nothing in the manifest
 * maps to still throws.
 *
 * It also self-clears: once a modality is admitted, its pairs appear in `advertised` and the strict
 * arm covers them again. Callers that pass nothing keep the original two-state behaviour.
 */
export function assertRung4SurveyCoversEveryFamily(
  survey,
  models,
  { catalogFamilyBackends, pendingSurveys = PENDING_RUNG4_SURVEYS } = {},
) {
  const advertised = new Set(
    models.flatMap((model) => model.backends.map((backend) => `${familyGroup(model.id)}:${backend}`)),
  );
  const groupsInUniverse = new Set(models.map((model) => String(familyGroup(model.id))));
  for (const key of advertised) {
    if (!survey.has(key)) {
      const [group, backend] = key.split(":");
      if (pendingSurveys.has(group)) continue;
      throw new Error(
        `family ${familyLabel(group)} has no ${backend} rung-4 survey verdict, so its bounded_transformer_residency cells would report Missing without ever having been surveyed (SC-15969)`,
      );
    }
  }
  // A pending declaration is a debt, not a licence. It expires the moment the verdict lands, and a
  // row for a family the catalog does not advertise at all is a leftover that would sit in the source
  // being maintained and trusted while naming nothing.
  for (const [group, owner] of pendingSurveys) {
    const covered = ["mlx", "candle"].filter((backend) => advertised.has(`${group}:${backend}`));
    if (!covered.length) {
      throw new Error(
        `rung-4 survey: family ${familyLabel(group)} is declared pending on sc-${owner}, but the catalog advertises no entry in that family — remove the pending row (sc-18815)`,
      );
    }
    if (covered.every((backend) => survey.has(`${group}:${backend}`))) {
      throw new Error(
        `rung-4 survey: family ${familyLabel(group)} now carries a verdict for every advertised backend, so its pending row (sc-${owner}) is stale — delete it so the cells stop reporting unsurveyed (sc-18815)`,
      );
    }
  }
  for (const key of survey.keys()) {
    if (!advertised.has(key) && !catalogFamilyBackends?.has(key)) {
      const [group, backend] = key.split(":");
      throw new Error(
        `rung-4 survey: family ${familyLabel(group)} carries a ${backend} verdict, but the catalog advertises no ${backend} entry in that family — the verdict reaches no cell (SC-15969)`,
      );
    }
  }
  if (survey.unroutedBackends) {
    // Video survey ownership is family-scoped. For every owned video family present in the universe,
    // each absent backend must be declared explicitly — absence is a route fact, never five
    // implementation verdicts. The declaration self-expires if the route appears later.
    for (const group of Object.keys(VIDEO_FAMILY_STORIES).filter((key) => groupsInUniverse.has(key))) {
      for (const backend of ["mlx", "candle"]) {
        const key = `${group}:${backend}`;
        if (!advertised.has(key) && !survey.unroutedBackends.has(key)) {
          throw new Error(
            `rung-4 survey: family ${group} has no ${backend} route — declare it in unroutedBackends so absence is not misread as five Missing rungs`,
          );
        }
      }
    }
    for (const [key, declaration] of survey.unroutedBackends) {
      const [group, backend] = key.split(":");
      if (!groupsInUniverse.has(group)) {
        throw new Error(`rung-4 survey: unrouted ${key} names no family in the model universe`);
      }
      if (advertised.has(key) || catalogFamilyBackends?.has(key)) {
        throw new Error(
          `rung-4 survey: ${key} is declared unrouted (sc-${declaration.owningStory}) but the routing catalog now advertises it — delete the declaration`,
        );
      }
      if (!["mlx", "candle"].includes(backend)) {
        throw new Error(`rung-4 survey: unrouted ${key} names an unknown backend`);
      }
    }
  }
  for (const [group, relationship] of survey.modalityRelationships ?? []) {
    for (const [modality, ids] of Object.entries(relationship.entries)) {
      for (const id of ids) {
        const model = models.find((candidate) => candidate.id === id);
        if (!model || model.modality !== modality || String(familyGroup(id)) !== group) {
          throw new Error(
            `rung-4 survey: ${group} modalityRelationship says ${id} is ${modality}, but the admitted catalog does not`,
          );
        }
      }
    }
  }
}

/**
 * Every `familyGroup:backend` pair the ROUTING CATALOG routes, across every modality.
 *
 * This is deliberately wider than the matrix's model universe and deliberately narrower than "any
 * string": it is what `assertRung4SurveyCoversEveryFamily` accepts from a survey verdict that has
 * been written before its modality is admitted. `familyGroup` throws on an id it does not know, and
 * that throw is the point — a family with no group mapping is not in the catalog's vocabulary and
 * its verdict stays rejected.
 */
export function catalogFamilyBackends(manifestModels, routedBackends) {
  const pairs = new Set();
  for (const model of manifestModels) {
    let group;
    try {
      group = familyGroup(model.id);
    } catch {
      continue;
    }
    for (const backend of routedBackends.get(model.id) ?? []) pairs.add(`${group}:${backend}`);
  }
  return pairs;
}

/**
 * Whether this entry advertises rung 1 on this backend.
 *
 * This is the RUNG-1 arm's own predicate, and after sc-19542 that is the only place it is applied
 * unconditionally. The rung-4 arm used to gate every family on it as well; `rung4ContractAdmits`
 * replaced that, and this function now reaches rung 4 only through the `rung` evaluator in
 * `RUNG4_PREREQUISITE_EVALUATORS`, for the families whose own provider record declares that edge.
 *
 * Availability, not engagement: it answers whether the provider implements rung 1 on this lane. That
 * is the right question for the edge it now serves, because gen-core's `validate_selection` accepts
 * a `Rung { .. EngagedInSameRequest }` prerequisite when `MemoryProviderContract::engages` holds,
 * and for an edge the realization itself appended, `engages` reduces to that rung being declared
 * `Implemented` (inference `crates/contracts/gen-core/src/memory_strategy.rs:1346-1362` at the
 * pinned rev `f17c82544`: `required_by_realization` is true by construction, so the conjunction is
 * the `matches!(self.support(rung), Some(Implemented))` term).
 *
 * This is the reduction of ONE of `validate_selection`'s two accepting arms, not of the whole
 * prerequisite check. The second arm is `StructurallyNotApplicable`, and it lives in the `rung`
 * evaluator below rather than here, because it is a fact about the provider's declaration and not
 * about what this lane makes available.
 */
/**
 * SC-20790 delivered receipt-backed, request-authoritative Candle admission for these bespoke
 * providers with NO manifest declaration row, on purpose: a `staged_residency` row carrying
 * `requestContexts` on the host model would make `declared_candle_request_strategy_contract`
 * treat the base lane's manifest as "relevant" and terminally refuse crossed coordinates, changing
 * runtime behavior a visibility fix must not touch. The worker census const
 * (`memory_route_registry.rs#CANDLE_BESPOKE_REQUEST_PROVIDERS`) is the source of truth for WHICH
 * providers carry this authority; this map binds each to the exact catalog coordinates its lane
 * serves (from the provider's RULES entry + `declared_candle_bespoke_request` shape arm), and
 * `parseCandleBespokeStagedLanes` fails generation when the two drift — a delivered lane without a
 * coordinate map must red the build, never publish as `Missing` (sc-20799).
 */
const CANDLE_BESPOKE_REQUEST_LANE_COORDINATES = new Map([
  [
    "candle_kolors_ipadapter",
    {
      modelId: "kolors",
      overlays: ["identity"],
      modes: ["character_image"],
      tiers: ["bf16", "q4", "q8"],
    },
  ],
  [
    "candle_kolors_control",
    {
      modelId: "kolors",
      overlays: ["control"],
      modes: ["text_to_image", "style_variations", "character_image"],
      tiers: ["bf16", "q4", "q8"],
    },
  ],
]);

/** `model.id:overlay:mode:tier` coordinates with receipt-backed bespoke Candle staged coverage. */
export function parseCandleBespokeStagedLanes(registrySource) {
  const match = registrySource.match(
    /const\s+CANDLE_BESPOKE_REQUEST_PROVIDERS:\s*&\[&str\]\s*=\s*&\[([\s\S]*?)\];/,
  );
  if (!match) {
    throw new Error(
      "memory-matrix: could not derive CANDLE_BESPOKE_REQUEST_PROVIDERS from memory_route_registry.rs",
    );
  }
  const providers = [...match[1].matchAll(/"([^"]+)"/g)].map((entry) => entry[1]);
  if (providers.length === 0) {
    throw new Error("memory-matrix: CANDLE_BESPOKE_REQUEST_PROVIDERS parsed to zero providers");
  }
  const mapped = new Set(CANDLE_BESPOKE_REQUEST_LANE_COORDINATES.keys());
  const unmapped = providers.filter((provider) => !mapped.has(provider)).sort();
  const stale = [...mapped].filter((provider) => !providers.includes(provider)).sort();
  if (unmapped.length || stale.length) {
    throw new Error(
      "memory-matrix: candle bespoke request lanes disagree with the worker census " +
        `(censused but unmapped=${unmapped.join(",") || "none"}; ` +
        `mapped but no longer censused=${stale.join(",") || "none"}). ` +
        "A delivered receipt-backed lane without a coordinate map would publish as Missing (sc-20799).",
    );
  }
  const lanes = new Set();
  for (const provider of providers) {
    const lane = CANDLE_BESPOKE_REQUEST_LANE_COORDINATES.get(provider);
    for (const overlay of lane.overlays) {
      for (const mode of lane.modes) {
        for (const tier of lane.tiers) {
          lanes.add(`${lane.modelId}:${overlay}:${mode}:${tier}`);
        }
      }
    }
  }
  return lanes;
}

export function stagedResidencyIsAvailable({
  backend,
  model,
  route,
  provider,
  tier,
  mode,
  overlay,
  stagedResidencyEngines,
  manifestById,
  candleBespokeStagedLanes,
}) {
  const contract = model[backend]?.memoryStrategyContract;
  const declaredStagedResidency = contract?.implementations?.some(
    (implementation) =>
      (implementation.runtimeProvider ?? contract.provider) === provider &&
      implementation.tiers.includes(tier) &&
      implementation.modes.includes(mode) &&
      implementation.overlays.includes(overlay) &&
      implementation.engagedRungs?.includes("staged_residency"),
  );
  if (declaredStagedResidency) return true;
  const declaredModel =
    !model[backend]?.memoryStrategyContract &&
    model.id === "z_image_edit" && route.engine === "z_image_turbo"
      ? manifestById.get("z_image_turbo")
      : model;
  return backend === "mlx"
    // The MLX provider id specifically: `engine_engages_staged_residency` is a claim about the MLX
    // registry, and a video entry's candle provider is a different id (sc-18815). The predicate is
    // deliberately wider than selectable Sequential: SC-18816 made unconditional staging visible.
    ? stagedResidencyEngines.has(route.engineFor("mlx"))
    : declaredModel.candle?.supportsSequentialOffload === true ||
        declaredModel.candle?.sequentialPeakGb !== undefined ||
        declaredModel.candle?.turboFit !== undefined ||
        // sc-20799: receipt-backed bespoke request admission (SC-20790) — coverage the worker
        // census declares and no manifest row may carry (see the lane map's docstring).
        (candleBespokeStagedLanes?.has(`${model.id}:${overlay}:${mode}:${tier}`) ?? false);
}

/**
 * gen-core's `BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES`, mirrored (sc-19542).
 *
 * `&[MemoryStrategyPrerequisite::LoadShape(LoadShape::DeferredMaterialization)]` at the pinned rev
 * `f17c82544` — inference `crates/contracts/gen-core/src/memory_strategy.rs:310-313`, the identifier
 * on line 312. Exactly one edge, shared by every provider, and no rung edge at all. Everything else
 * a provider demands of rung 4 it appends itself, which is what
 * `config/rung4-contract-prerequisites.json` records per (family, backend).
 *
 * This constant is the gate's own predicate, and the guards derive from it rather than restating it:
 * `assertRung4CalibrationsDeclareTheRequiredLoadShape` reads the required shape out of this array
 * instead of carrying its own copy of the string, so teaching rung 4 a second shared edge moves the
 * guard with the gate.
 */
export const SHARED_RUNG4_PREREQUISITES = Object.freeze([
  Object.freeze({ kind: "load-shape", shape: "deferred_materialization" }),
]);

/**
 * One evaluator per prerequisite kind, mirroring gen-core's `validate_selection` arms.
 *
 * Dispatch is by `kind` with no default arm — an unrecognised kind throws, because a prerequisite
 * the matrix does not know how to evaluate must not be silently satisfied. That is the failure
 * direction sc-19542 exists to remove.
 */
const RUNG4_PREREQUISITE_EVALUATORS = Object.freeze({
  /**
   * gen-core: `if self.load_shape == required { continue }`.
   *
   * `load_shape` is a LOAD-time property of the generator instance (`spec.load_shape`), not a
   * property of a catalog coordinate, so no cell can be demoted on it here: a rung-4 matrix cell is
   * a claim about the loads that satisfy this edge, and the catalog carries no per-cell load shape
   * to test. It is NOT thereby unchecked. Where the catalog does carry a load shape — on a declared
   * rung-4 calibration binding — `assertRung4CalibrationsDeclareTheRequiredLoadShape` grades it
   * against this same edge and refuses generation on a mismatch.
   */
  "load-shape": () => true,
  /**
   * gen-core's `EngagedInSameRequest` arm has TWO ways to be satisfied, and this mirrors both —
   * inference `crates/contracts/gen-core/src/memory_strategy.rs:1862-1873` at the pinned
   * `f17c82544`.
   *
   * 1. `if self.engages(selection.strategy, rung) { continue }`. For an edge the realization itself
   *    appended, `engages`'s `required_by_realization` term is true by construction, so the
   *    conjunction reduces to that rung being declared `Implemented` on this lane — which is what
   *    `stagedResidencyIsAvailable` answers.
   * 2. `if matches!(self.support(rung), Some(StructurallyNotApplicable { .. })) { continue }` — the
   *    edge is satisfied VACUOUSLY, because the provider asserts its architecture has no such
   *    component to shed. gen-core's own comment says so.
   *
   * Arm 2 was missing, and this gate exists to follow the contract's actual rule rather than a
   * reduction of half of it. It is fail-closed and currently unreachable — `mlx-gen-sensenova` is
   * the only provider declaring `StagedResidency` structurally N/A at the pinned revision, and it
   * appends no rung edge for the arm to satisfy — but the first provider that does both would be
   * under-admitted without it.
   *
   * `record.stagedResidencySupport` is `null` wherever the extractor could not read the declaration
   * unambiguously, and `null` falls through to arm 1. That asymmetry is deliberate: arm 2 ADMITS, so
   * only a positive reading of the provider's own declaration may fire it.
   */
  rung: (prerequisite, context, record) => {
    if (
      prerequisite.rung !== "staged_residency" ||
      prerequisite.scope !== "engaged_in_same_request"
    ) {
      throw new Error(
        `rung-4 prerequisite ${JSON.stringify(prerequisite)} names a rung/scope this gate has no ` +
          "evaluator for — teach it one rather than admitting the cell (sc-19542)",
      );
    }
    if (record.stagedResidencySupport === "structurally_not_applicable") return true;
    return stagedResidencyIsAvailable(context);
  },
});

/** The prerequisite kinds a record may carry, derived from the evaluators that can answer them. */
export const RUNG4_PREREQUISITE_KINDS = Object.freeze(Object.keys(RUNG4_PREREQUISITE_EVALUATORS));

/**
 * What a record may say a provider declares for the rung its edges name.
 *
 * gen-core's `MemoryStrategySupport` has three variants; `null` is this gate's fourth answer, for a
 * declaration the extractor could not read unambiguously. Only `structurally_not_applicable` changes
 * an admission, and only in the ADMITTING direction, which is why the unknown case is spelled at all
 * rather than defaulted.
 */
export const RUNG4_DECLARED_SUPPORTS = Object.freeze([
  "implemented",
  "missing",
  "structurally_not_applicable",
  null,
]);

/**
 * Whether rung 4's DECLARED prerequisite graph admits this cell (sc-19542).
 *
 * This is the fix. The arm used to ask `stagedResidencyIsAvailable` of every family, which is a
 * rung-1 availability proxy applied where the contract never asked for one — measurably inert on
 * the catalog as it stood, and wrong the moment a provider's prerequisites change. It now walks the
 * same graph `MemoryProviderContract::validate_selection` walks: the shared edges above, then the
 * edges THIS provider appended through `additional_prerequisites`, each answered by its own
 * evaluator.
 *
 * `record.additionalPrerequisites` is derived from the pinned revision by
 * `scripts/rung4-contract-prerequisites.mjs`, and 21 of the catalog's 40 (family, backend) pairs
 * carry the rung-1 edge while 19 do not — so which families this consults the rung-1 predicate for
 * is now a property of the providers rather than a blanket rule.
 *
 * WHAT THAT 21/19 IS AND IS NOT (measured, sc-19542 review f6)
 *
 * 21/19 is a fact about the RECORD SET, not about this catalog. Instrumenting this arm over a full
 * generation run measures the difference it actually makes:
 *
 *   * 426 evaluations reach `rung4ContractAdmits`, spanning 17 of the 40 (family, backend) pairs.
 *     The other 23 pairs are never evaluated at all: the `&&` chains at both call sites short-
 *     circuit on `structuralApplicability` and on `rung4Implementation` first, so a pair whose
 *     rung-4 verdict is `none`, or which has no implementation parameters, never asks the contract.
 *   * Of those 17 reached pairs, exactly 2 carry NO rung-1 edge and are therefore the only lanes
 *     whose answer this fix can change: `15510:mlx` (Z-Image) and `15527:candle` (PuLID). The other
 *     17 no-edge pairs are in the 23 that never arrive.
 *   * `stagedResidencyIsAvailable` returned `true` at every one of the 426 evaluations, so the
 *     edge-carrying pairs are admitted on the same term the old blanket proxy admitted them on.
 *   * Old and new therefore agree on 40/40 probed keys, which is why the generated artifact is
 *     BYTE-IDENTICAL across this change.
 *
 * So "19 lanes no longer consult the rung-1 proxy" is true of the record set and reaches 2 lanes in
 * practice. The fix is still the right one — it replaces a proxy that happens to agree today with
 * the graph the contract actually walks, and the divergence it prevents is the first time a
 * provider's prerequisites change — but it is a correctness change with a byte-identical artifact,
 * not a change in what this catalog admits. Stated here because a reader who takes 21/19 as the
 * blast radius will over-read every one of those numbers.
 *
 * An edge flagged `conditional` — appended by a `then_some` on a runtime condition, or inside a
 * `#[cfg]`-gated item that is not in every production build — is evaluated as PRESENT. "This
 * provider may demand rung 1" is the fail-closed reading for an admission gate: a conditional
 * prerequisite dropped is a cell admitted that some builds refuse.
 */
export function rung4ContractAdmits(record, context) {
  if (!record) {
    throw new Error(
      "rung-4 admission asked for a (family, backend) with no contract-prerequisite record — the " +
        "coverage fence in parseRung4ContractPrerequisites should have made this unreachable (sc-19542)",
    );
  }
  return [...SHARED_RUNG4_PREREQUISITES, ...record.additionalPrerequisites].every((prerequisite) => {
    const evaluate = RUNG4_PREREQUISITE_EVALUATORS[prerequisite.kind];
    if (!evaluate) {
      throw new Error(
        `rung-4 prerequisite kind ${JSON.stringify(prerequisite.kind)} has no evaluator (sc-19542)`,
      );
    }
    return evaluate(prerequisite, context, record);
  });
}

/**
 * Parse and validate the per-(family, backend) rung-4 prerequisite records (sc-19542).
 *
 * Two things are checked here, and each is a way the record could be WRONG rather than merely absent:
 *
 * - **Keyed to the live pin.** The edges are a fact about one inference revision. A record keyed to
 *   an older pin would report a graph nobody re-derived, so it is a hard error and never a fallback.
 * - **Every edge is one this gate can evaluate.** An edge whose kind has no evaluator, or that cites
 *   no provider file, throws rather than being skipped over by `every()`.
 *
 * Coverage is the third, and it lives in `assertRung4PrerequisiteRecordsCoverEveryFamily` so it runs
 * beside — and after — the survey's own coverage fence. Both answer to the same catalog, and a
 * family that is missing from BOTH should be reported as an unsurveyed family rather than as an
 * unrecorded one.
 */
export function parseRung4ContractPrerequisites(body, { pin }) {
  const parsed = JSON.parse(body);
  if (parsed.inferenceRevision !== pin) {
    throw new Error(
      `config/rung4-contract-prerequisites.json is keyed to ` +
        `${parsed.inferenceRevision?.slice(0, 9) ?? "(unset)"} but Cargo pins ${pin.slice(0, 9)}. ` +
        "Re-run: node scripts/rung4-contract-prerequisites.mjs --repo <inference> --write",
    );
  }
  const records = new Map();
  for (const [group, family] of Object.entries(parsed.families ?? {})) {
    for (const [backend, record] of Object.entries(family.backends ?? {})) {
      const at = `rung-4 contract prerequisites ${family.name ?? group} (${backend})`;
      if (!record.crate) {
        throw new Error(`${at}: must name the inference crate the record was derived from`);
      }
      const edges = record.additionalPrerequisites;
      if (!Array.isArray(edges)) {
        throw new Error(
          `${at}: additionalPrerequisites must be an array — an absent field would read as "no ` +
            'edges", which is the fail-open answer',
        );
      }
      for (const edge of edges) {
        if (!RUNG4_PREREQUISITE_KINDS.includes(edge.kind)) {
          throw new Error(
            `${at}: prerequisite kind ${JSON.stringify(edge.kind)} has no evaluator in this gate`,
          );
        }
        if (!edge.source) {
          throw new Error(`${at}: every edge must cite the provider file it was derived from`);
        }
        // An absent `conditional` would read as "unconditional", which is a claim about the
        // provider rather than the absence of one.
        if (typeof edge.conditional !== "boolean") {
          throw new Error(
            `${at}: every edge must say whether the construction that appends it is conditional`,
          );
        }
      }
      if (!RUNG4_DECLARED_SUPPORTS.includes(record.stagedResidencySupport ?? null)) {
        throw new Error(
          `${at}: stagedResidencySupport ${JSON.stringify(record.stagedResidencySupport)} is not a ` +
            `gen-core MemoryStrategySupport this gate knows (${RUNG4_DECLARED_SUPPORTS.join(", ")})`,
        );
      }
      records.set(`${group}:${backend}`, {
        crate: record.crate,
        additionalPrerequisites: edges,
        stagedResidencySupport: record.stagedResidencySupport ?? null,
      });
    }
  }
  return records;
}

/**
 * Coverage of the prerequisite records, both ways — the same shape as the survey's fence.
 *
 * Catalog -> records is what keeps the gate honest: a family with no record reaches
 * `rung4ContractAdmits` with nothing to admit from. Records -> catalog keeps the FILE honest: a
 * record for a lane the catalog no longer advertises would sit there being maintained and reviewed
 * while deciding nothing.
 */
export function assertRung4PrerequisiteRecordsCoverEveryFamily(records, models) {
  // Scoped to the IMAGE half of the universe, which is the half the derivation script
  // (scripts/rung4-contract-prerequisites.mjs) walks: sc-19542 derived the per-provider edge
  // graphs for the 20 image families, and sc-18815 widened the matrix universe to video entries
  // afterwards. A video lane therefore has no record BY CONSTRUCTION and falls back to the
  // pre-sc-19542 direct predicate at the admission arms (see the record-or-fallback call sites),
  // which is byte-identical to what main shipped for those lanes. Deriving the video providers'
  // graphs widens this fence with them.
  const advertised = new Set(
    models
      .filter((model) => model.modality === "image")
      .flatMap((model) => model.backends.map((backend) => `${familyGroup(model.id)}:${backend}`)),
  );
  for (const key of advertised) {
    if (!records.has(key)) {
      const [group, backend] = key.split(":");
      throw new Error(
        `family SC-${group} has no ${backend} rung-4 contract-prerequisite record, so the rung-4 arm ` +
          "would have no declared graph to admit its cells from (sc-19542)",
      );
    }
  }
  for (const key of records.keys()) {
    if (!advertised.has(key)) {
      const [group, backend] = key.split(":");
      throw new Error(
        `rung-4 contract prerequisites: family SC-${group} carries a ${backend} record, but the ` +
          "catalog advertises no such entry — the record reaches no cell (sc-19542)",
      );
    }
  }
}

/**
 * The shared `LoadShape` edge, graded where the catalog can actually see a load shape (sc-19542).
 *
 * `rung4ContractAdmits` cannot demote a cell on that edge, because a coordinate has no load shape.
 * A declared CALIBRATION BINDING does: `loadShape` is a field on every binding, and today's rung-4
 * bindings carry `deferred_materialization` while their rungs 0-3 siblings carry
 * `eager_materialization` — the contract prerequisite showing up in the evidence. A rung-4 binding
 * claiming a shape the edge forbids describes a measurement gen-core would have refused to run, so
 * it fails generation.
 *
 * The required shape is READ OUT OF `SHARED_RUNG4_PREREQUISITES` rather than restated, so this
 * guard cannot drift from the gate it grades: change the shared edge and this demands the new shape
 * on the next run, with nothing to update by hand.
 */
export function assertRung4CalibrationsDeclareTheRequiredLoadShape(models) {
  const required = SHARED_RUNG4_PREREQUISITES.filter(
    (prerequisite) => prerequisite.kind === "load-shape",
  ).map((prerequisite) => prerequisite.shape);
  if (!required.length) {
    throw new Error(
      "rung 4 declares no shared load-shape prerequisite, so this guard grades nothing — remove it " +
        "with the edge rather than leaving it green (sc-19542)",
    );
  }
  for (const model of models) {
    for (const backend of ["mlx", "candle"]) {
      for (const binding of model[backend]?.calibrations ?? []) {
        if (binding.rung !== "bounded_transformer_residency") continue;
        if (!required.includes(binding.loadShape)) {
          throw new Error(
            `${model.id}:${backend}: a bounded_transformer_residency calibration binding declares ` +
              `loadShape ${JSON.stringify(binding.loadShape)}, but rung 4's shared prerequisite ` +
              `requires ${required.join(" or ")} (sc-19542)`,
          );
        }
      }
    }
  }
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

export function strategyStatus({
  backend,
  rung,
  route,
  provider,
  stagedResidencyEngines,
  model,
  tier,
  mode,
  overlay,
  rung4Survey,
  rung4ContractPrerequisites,
  manifestById,
  inferenceClosureDigests,
  candleBespokeStagedLanes,
}) {
  // A route-local row is authoritative for its exact provider coordinate. Missing sibling rungs may
  // still inherit the resolved provider's declaration (the established MLX alias behavior), but a
  // native route can never mask an explicitly authored alias row such as Candle Z-Image Edit.
  const declaredModel = declarationModelForCoordinate({
    backend,
    rung,
    route,
    provider,
    model,
    tier,
    mode,
    overlay,
    manifestById,
  });
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
  // `staticRung4Verdict` is `undefined` for a family declared pending (sc-18815). Every term below
  // already fails closed on that, and deliberately so: no rung-4 claim may be made for a family
  // nobody has surveyed, whichever direction the missing evidence would have pointed.
  const staticRung4Allowed =
    rung !== "bounded_transformer_residency" ||
    (["full", "partial"].includes(staticRung4Verdict?.structuralApplicability) &&
      staticRung4Implementation !== null &&
      // The image families' declared prerequisite graph (sc-19542) where a record exists; the
      // pre-sc-19542 direct predicate for the video lanes the derivation script does not yet walk
      // (sc-18815 sync reconciliation — the coverage fence documents the split).
      ((key, context) =>
        rung4ContractPrerequisites.has(key)
          ? rung4ContractAdmits(rung4ContractPrerequisites.get(key), context)
          : stagedResidencyIsAvailable(context))(`${familyGroup(model.id)}:${backend}`, {
        backend,
        model,
        route,
        provider,
        tier,
        mode,
        overlay,
        stagedResidencyEngines,
        manifestById,
        candleBespokeStagedLanes,
      }));
  const staticImplementation = staticContractCoversProvider(staticMemoryContract, provider)
    ? staticMemoryContract.implementations.find(
        (implementation) =>
          staticRung4Allowed &&
          (implementation.runtimeProvider ?? staticMemoryContract.provider) === provider &&
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
        // sc-20246: keyed on the declaration actually HAVING ranges, not on a declaration existing.
        // The engine-derived projection publishes no `parameterRanges` (the dumps carry none), and an
        // empty `publishedRanges: {}` on a cell whose parameters came from a calibration reads as
        // "the declaration published an empty range set" rather than "it published none". All 153
        // hand-authored rows declare `parameterRanges`, so this is inert for them.
        ...(staticImplementation?.parameterRanges
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
  // sc-20246: a COVERAGE-ONLY declaration must not displace richer evidence.
  //
  // The engine-derived projection (`scripts/generate-manifest-memory-declarations.mjs`) writes rows
  // that state only rung x tier x mode x overlay coverage — the engine dumps publish no
  // `parameters`, so the projection omits them rather than guessing. Such a row was outranking every
  // source below it: 37 published cells lost their measured decode/attention/window values to a
  // stale-but-real calibration binding being skipped, and the rung-4 survey's window parameters were
  // replaced by `{}` on the Lens cells. A declaration that names no parameters has nothing to say
  // about parameters, so it now yields to whatever does and is used only where nothing else answers
  // (see `coverageStatus` at the tail). Provably inert for hand-authored rows: all 153 of them carry
  // both `parameters` and `parameterRanges`.
  const coverageDeclaration =
    staticImplementation && !staticImplementation.parameters ? staticImplementation : null;
  const coverageStatus = () =>
    coverageDeclaration
      ? {
          state: "Implemented/unverified",
          source: coverageDeclaration.source,
          parameters: {},
        }
      : { state: "Missing", source: null, parameters: {} };
  if (staticImplementation && !coverageDeclaration) {
    return {
      // This declaration inventories production capability only. Exact runtime evidence must still
      // pass calibrationBinding before the cell can be promoted to Verified.
      state: "Implemented/unverified",
      source: staticImplementation.source,
      parameters: {
        ...staticImplementation.parameters,
        ...(staticImplementation.parameterRanges
          ? { publishedRanges: publishableParameterRanges(staticImplementation) }
          : {}),
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
    stagedResidencyIsAvailable({
      backend,
      model,
      route,
      provider,
      tier,
      mode,
      overlay,
      stagedResidencyEngines,
      manifestById,
      candleBespokeStagedLanes,
    })
  ) {
    return {
      state: "Implemented/unverified",
      source:
        backend === "mlx"
          ? "crates/sceneworks-worker/src/mlx_fit_gate.rs#engine_engages_staged_residency"
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
        // sc-18812: a declared curve carrying `perMpxFrameGb` needs three independent geometries,
        // not two. Taking the coefficient count from the DECLARED form rather than from whichever
        // records happen to exist is what stops a temporal cell whose evidence is all single-frame
        // from reporting `fitted` on two areas with its third coefficient undetermined.
        declaresTemporalCurve: Object.values(
          model.candle.turboFit.phaseCurvesByTier?.[tier]?.[manifestRung] ?? {},
        ).some((curve) => curve?.perMpxFrameGb !== undefined),
        historicalVerification: evidenceRecords.map((record) => ({
          source: `Shortcut ${record.sourceStory} activity ${record.sourceActivity}`,
          hardware: verification?.hardware,
          evidenceScope: record.evidenceScope,
          runtimeAdmission: record.evidenceScope === "exact_request",
          tier: record.tier,
          // sc-18812: ONE geometry key producer. Hand-building `WxH` here is what would let a
          // record supporting a temporal curve be characterized as a one-frame design point it
          // never measured — a FABRICATED rank contribution, worse than the collapse this story
          // fixed. `frames` is optional in the manifest and absent reads as 1, so no image record
          // moves; the audit forbids absence on any tier whose curves carry `perMpxFrameGb`.
          geometry: measuredGeometryKey(record),
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
            geometry: measuredGeometryKey(record),
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
    const group = familyGroup(model.id);
    const verdict = rung4Survey.get(`${group}:${backend}`);
    if (!verdict) {
      // sc-18815: a family declared pending has no verdict to answer from, so the cell falls through
      // to `Missing` — and carries `surveyed: false` from `rung4SurveyCell` so the reason is on the
      // cell rather than inferred. Any other family with no verdict is still a generation failure.
      if (!PENDING_RUNG4_SURVEYS.has(group)) {
        throw new Error(`${model.id}:${backend}: no rung-4 survey verdict (SC-15969)`);
      }
      return { state: "Missing", source: null, parameters: {} };
    }
    // Implementation is per ENTRY and per MODE — inference may route a catalog entry's modes to
    // different descriptors than the one carrying the contract — and the rung is unreachable unless
    // this provider's own declared prerequisite graph admits the cell, however good the architecture
    // is.
    const implementationParameters = rung4Implementation(
      verdict,
      model.id,
      tier,
      mode,
      overlay,
    );
    const implementedHere =
      implementationParameters !== null &&
      // The image families' declared prerequisite graph (sc-19542) where a record exists; the
      // pre-sc-19542 direct predicate for the video lanes the derivation script does not yet walk
      // (sc-18815 sync reconciliation — the coverage fence documents the split).
      ((key, context) =>
        rung4ContractPrerequisites.has(key)
          ? rung4ContractAdmits(rung4ContractPrerequisites.get(key), context)
          : stagedResidencyIsAvailable(context))(`${familyGroup(model.id)}:${backend}`, {
        backend,
        model,
        route,
        provider,
        tier,
        mode,
        overlay,
        stagedResidencyEngines,
        manifestById,
        candleBespokeStagedLanes,
      });
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
    return coverageStatus();
  }
  return coverageStatus();
}

function validateMatrix(
  matrix,
  expectedIds,
  backendTierOverrides,
  rung4Survey,
  rung4ContractPrerequisites,
  cellInventoryExpectations,
  calibrationPlan,
  catalogFamilyBackendPairs,
) {
  // sc-18815: censused PER MODALITY. A single total would let a video entry appearing cover for an
  // image entry disappearing, which is exactly the accounting the image-only filter made impossible
  // to notice in the other direction.
  const ids = matrix.models.filter((model) => model.modality === "image").map((model) => model.id);
  const videoIds = matrix.models.filter((model) => model.modality === "video").map((model) => model.id);
  const unknownModality = matrix.models.filter((model) => !MATRIX_MODALITIES.has(model.modality));
  if (unknownModality.length) {
    throw new Error(
      `matrix carries entries of unadmitted modalities: ${unknownModality.map((model) => `${model.id}(${model.modality})`).join(",")}`,
    );
  }
  if (ids.length !== EXPECTED_IMAGE_COUNT) {
    throw new Error(`expected exactly ${EXPECTED_IMAGE_COUNT} image entries, found ${ids.length}`);
  }
  if (videoIds.length !== EXPECTED_VIDEO_COUNT) {
    throw new Error(`expected exactly ${EXPECTED_VIDEO_COUNT} video entries, found ${videoIds.length}`);
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
  assertCalibrationPlanTargetsResolvedCoordinates(calibrationPlan, matrix.cells, {
    outOfMatrixEntries: rung4Survey.outOfMatrixCatalogEntries,
  });
  assertTwinCoverage(matrix.models);
  assertVideoOwnership(matrix.models);
  assertUnroutedEntriesAreDeclared(matrix.models);
  assertCellOwnershipIsBackendScoped(
    matrix.cells,
    buildStoryBackendScope(),
    new Map(matrix.models.map((model) => [model.id, model.modality])),
  );
  assertRung4SurveyCoversEveryFamily(rung4Survey, matrix.models, {
    catalogFamilyBackends: catalogFamilyBackendPairs,
  });
  assertRung4PrerequisiteRecordsCoverEveryFamily(rung4ContractPrerequisites, matrix.models);
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
    assertCharacterizationIsConsistent(cell);
  }
}

// SC-16060. `state` and `memoryCharacterization` are independent claims, and the invariants that
// keep them from silently merging back into one field belong here rather than in a consumer.
//
// sc-18812 rewrote the first of them. Two or more geometries no longer IMPLY `fitted` — rank does,
// and a rank-deficient multi-geometry cell (two resolutions of one area, several frame counts at
// one area) is now legitimately `point`. Re-deriving the status by counting, as this did, would
// have thrown on exactly the cells the rank rule exists to describe. Recomputing through
// `memoryCharacterization` instead would be tautological and could not catch a bug in it. What is
// asserted is therefore the part that stays independent of the rank computation: the two ends of
// the vocabulary are still decided by count, and `fitted` still needs at least as many measured
// geometries as the cell's own form has coefficients.
//
// Exported because it is the only guard here that a unit test can reach without reconstructing the
// generator's whole source universe, and an unreachable guard is how SC-16060 got its `Verified`
// producer wrong in the first place.
export function assertCharacterizationIsConsistent(cell) {
  const characterization = cell.memoryCharacterization;
  const measured = characterization.measuredGeometries.length;
  const plural = measured === 1 ? "y" : "ies";
  const expected = measured === 0 ? "unmeasured" : measured === 1 ? "point" : null;
  if (expected !== null && characterization.status !== expected) {
    throw new Error(
      `${cell.id}: memoryCharacterization is ${characterization.status} on ${measured} measured geometr${plural}`,
    );
  }
  // `coveredFrameBound` is emitted iff the cell's curve form is temporal, so its presence is the
  // published record of how many coefficients this cell had to determine.
  const coefficients = "coveredFrameBound" in characterization ? 3 : 2;
  if (characterization.status === "fitted" && measured < coefficients) {
    throw new Error(
      `${cell.id}: fitted on ${measured} measured geometr${plural}, but its curve has ${coefficients} coefficients`,
    );
  }
  // A bound without a determinable curve is the exact overclaim this story exists to stop: it
  // would read as "covered up to here" on the strength of a single point.
  if ((characterization.coveredPixelBound !== null) !== (characterization.status === "fitted")) {
    throw new Error(
      `${cell.id}: coveredPixelBound is only meaningful on a fitted curve (status ${characterization.status})`,
    );
  }
  // sc-18812: the temporal bound is the same claim on the other axis and gets the same rule.
  if (
    "coveredFrameBound" in characterization &&
    (characterization.coveredFrameBound !== null) !== (characterization.status === "fitted")
  ) {
    throw new Error(
      `${cell.id}: coveredFrameBound is only meaningful on a fitted curve (status ${characterization.status})`,
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
 *
 * ## The one exemption, and why it is not a hole (sc-18663)
 *
 * `buildMatrix` SUBTRACTS the survey's declared out-of-matrix catalog entries from the coordinate
 * universe, so those entries resolve to no cell BY CONSTRUCTION. Requiring a plan row for one of
 * them to match a coordinate is therefore a category error, not a missing-coordinate report: such a
 * family is validated by its `outOfMatrixFamilies` record — every stack, both claims, on every
 * generation — instead of by a published cell, and epic 17137's terminal campaign has to be able to
 * plan captures against MiniMax-H3 before the matrix can carry a verdict for it.
 *
 * The exemption reads the SAME set the universe subtraction reads (`survey.outOfMatrixCatalogEntries`,
 * computed once in `parseRung4Survey`), so the two cannot drift into exempting rows the matrix does
 * carry. It is keyed on `target.modelId`, the same field the subtraction is keyed on, so it is the
 * exact complement of that subtraction and nothing wider. An entry naming a family that is in
 * NEITHER the matrix nor the survey's out-of-matrix set still fails closed, and the day `familyGroup`
 * learns one of the named entries `parseOutOfMatrixRung4Families` throws and forces the family into
 * the matrix — at which point its plan rows are held to the coordinate requirement again.
 */
export function assertCalibrationPlanTargetsResolvedCoordinates(
  calibrationPlan,
  cells,
  { outOfMatrixEntries = new Set() } = {},
) {
  const unmatched = calibrationPlan.providers.filter(
    (entry) =>
      !outOfMatrixEntries.has(entry.target.modelId) &&
      !cells.some((cell) => planEntryTargetsCoordinate(entry, cell)),
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
  // sc-18815: the video lane's route resolvers. The image lane resolves a provider from ONE table
  // (`engines.rs#MODEL_TABLE`); the video lane has no such table — `resolve_video_route` /
  // `resolve_candle_video_route` (`video_jobs/mod.rs`) consult one `*_engine_id` function per family,
  // and those functions are where the model-id -> provider-id mapping actually lives. Deriving the
  // universe's providers from anywhere else would be a restatement that can drift; deriving them from
  // here means a worker route change rotates this artifact's provenance, which is the point.
  videoRouteWan: "crates/sceneworks-worker/src/video_jobs/wan.rs",
  videoRouteLtx: "crates/sceneworks-worker/src/video_jobs/ltx.rs",
  videoRouteSvd: "crates/sceneworks-worker/src/video_jobs/svd.rs",
  videoRouteBernini: "crates/sceneworks-worker/src/video_jobs/bernini.rs",
  videoRouteScail2: "crates/sceneworks-worker/src/video_jobs/scail2.rs",
  videoRouteKreaRealtime: "crates/sceneworks-worker/src/video_jobs/krea_realtime.rs",
  videoRouteCandle: "crates/sceneworks-worker/src/video_jobs/candle.rs",
  mlxFitGate: "crates/sceneworks-worker/src/mlx_fit_gate.rs",
  // sc-20799: in SOURCE_PATHS because it DECIDES cell state — `CANDLE_BESPOKE_REQUEST_PROVIDERS`
  // is where receipt-backed bespoke Candle staged coverage is declared, and a source that changes a
  // cell but sits outside the fingerprint is a provenance hole.
  memoryRouteRegistry: "crates/sceneworks-worker/src/memory_route_registry.rs",
  memoryStrategy: "crates/sceneworks-worker/src/memory_strategy.rs",
  vramGate: "crates/sceneworks-worker/src/vram_gate.rs",
  instantId: "crates/sceneworks-worker/src/image_jobs/instantid.rs",
  calibrationEvidence: "docs/generated/memory-calibration-evidence.json",
  calibrationPlan: "config/memory-calibration-plan.json",
  inferenceClosures: "config/inference-provider-closures.json",
  rung4Survey: "config/rung4-applicability-survey.json",
  // sc-19542. In SOURCE_PATHS because it DECIDES cell state: the rung-4 arm admits from these
  // records. A source that changes a cell and is outside the fingerprint is a provenance hole, and
  // the source-tree revision is the thing that would have to notice a record being edited.
  rung4ContractPrerequisites: "config/rung4-contract-prerequisites.json",
  engineCapabilitiesMlx: "config/engine-capabilities/capabilities.mlx.json",
  engineCapabilitiesCandle: "config/engine-capabilities/capabilities.candle.json",
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
export async function buildMatrix({ sourceOverrides = {}, cellFilter = null, publish = true, onReconciliation = null } = {}) {
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
  // sc-19542: the rung-4 arm's declared prerequisite graph, per (family, backend). Coverage is
  // fenced against the survey's own key set, so the two cannot drift apart.
  const rung4ContractPrerequisites = parseRung4ContractPrerequisites(
    bodies.rung4ContractPrerequisites,
    { pin: inferencePin(cargoBody) },
  );
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
  // sc-18815: the model universe is MODALITY-AWARE, not `type === "image"`. Every entry of an
  // admitted modality is in, whether or not anything has been measured on it — an entry the matrix
  // does not carry cannot even report `Missing`, which is how the video lane read as complete while
  // covering one modality. Adding a modality here is deliberate and gated: it needs route resolution
  // (`resolveRoutes`), a family group, and a rung-4 survey verdict or a declared pending owner.
  //
  // MINUS the survey's declared OUT-OF-MATRIX entries (sc-18664 × sc-18815, reconciled by the
  // sc-17137 main sync). Those records name catalog entries the matrix cannot yet carry a verdict
  // for — MiniMax-H3 is the live case: `familyGroup` has no arm for it, no video-route resolver row
  // exists, and its providers are absent at the pinned inference revision, so admitting it here
  // fails generation at `resolveRoute` rather than producing a row. The records are NOT unvalidated
  // escape hatches: `parseOutOfMatrixRung4Families` (run inside `parseRung4Survey` above) validates
  // every one on every generation and throws the day `familyGroup` learns one of the named entries,
  // which is what forces the family INTO this universe when the epic promotes it.
  // sc-18663: read from the parse above rather than re-derived from a second `JSON.parse` of the
  // same body. The calibration-plan exemption reads the same set, and two spellings of it would let
  // an exemption survive a subtraction being narrowed (or the reverse).
  const outOfMatrixEntries = rung4Survey.outOfMatrixCatalogEntries;
  const entries = manifest.models.filter(
    (model) => MATRIX_MODALITIES.has(model.type) && !outOfMatrixEntries.has(model.id),
  );
  const images = entries.filter((model) => model.type === "image");
  const manifestById = new Map(entries.map((model) => [model.id, model]));
  const expectedIds = parseExpectedImageIds(enginesBody);
  const routes = parseEngineRoutes(enginesBody);
  const videoRoutes = parseVideoRoutes(bodies);
  const routedBackends = routedLanes({
    routingCatalog: bodies.routingCatalog,
    routingCandle: bodies.routingCandle,
    routingMlx: bodies.routingMlx,
  });
  const stagedResidencyEngines = parseMlxStagedResidencyEngines(mlxFitBody);
  const candleBespokeStagedLanes = parseCandleBespokeStagedLanes(bodies.memoryRouteRegistry);
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
  // Over the WHOLE universe, not just the image half (sc-18815). The first arm compares two image
  // source lists and is unaffected; the second looks for a `[backend].control` measurement block on
  // an entry that declares no control lane, and scoping that to `images` would let a video entry
  // acquire one and orphan its measurements silently — the sc-16069 failure in a new modality.
  assertDeclaredControlLanes(entries, bodies.imageRouting);
  // sc-19542: rung 4's shared LoadShape edge, graded where the catalog can see a load shape.
  assertRung4CalibrationsDeclareTheRequiredLoadShape(images);

  const models = entries
    .map((model) => {
      const backends = backendScopes(model, routedBackends);
      const route = resolveRoute(model, routes, videoRoutes, backends);
      return {
        id: model.id,
        name: model.name,
        modality: model.type,
        // sc-18815: the family GROUP key, published so a consumer can join an entry to its
        // `rung4SurveyRows` row or its `summary.rung4Survey.pendingFamilyBackends` row. It used to be
        // derivable from `owningFamilyStories.mlx`, because an image family's group key IS its MLX
        // story id — that coincidence does not survive the video lane, where the key is a family name
        // and the owner is a survey story. Publishing the key makes the join explicit instead of
        // resting on an identity that is now only true for one modality.
        familyGroup: familyGroup(model.id),
        family: model.family ?? null,
        // Video providers are per-backend (LTX is `ltx_2_3` on MLX and `ltx_2_3_distilled` on
        // candle), so the single-valued route the image lane publishes cannot describe them. Publish
        // the resolved provider per backend and keep the scalar as the first-routed-backend one.
        // `resolvedRoutes` can no longer contain `null`: `engineFor` throws on a routed backend no
        // `*_engine_id` arm serves, so the failure names the resolver here instead of surfacing two
        // lanes later as a schema violation (sc-18815 review).
        resolvedRoute: route.engine,
        resolvedRoutes: Object.fromEntries(
          backends.map((backend) => [backend, route.engineFor(backend)]),
        ),
        routeKind: route.kind,
        backends,
        // Per-backend maps, not scalars: the entry has one owner per backend it advertises, and
        // `familyStory`/`modelStory` throw if the catalog advertises a backend nobody owns.
        owningFamilyStories: Object.fromEntries(
          backends.map((backend) => [backend, familyStory(model.id, backend)]),
        ),
        // sc-18815: the video lane has no per-(entry, backend) ownership story and must not invent
        // one. Epic 15448 filed 53 x 2 image stories; epic 18803 deliberately did not slice video
        // that way (measurement is a runbook, per epic 18093), so the truthful value is `null` and
        // `assertCellOwnershipIsBackendScoped` enforces the split in both directions.
        owningModelStories: Object.fromEntries(
          backends.map((backend) => [
            backend,
            model.type === "image" ? modelStory(model.id, backend) : null,
          ]),
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
    const axesRoute = resolveRoute(model, routes, videoRoutes, modelSummary.backends);
    for (const backend of modelSummary.backends) {
      const tiers = tiersFor(model, backend, backendTierOverrides);
      const modes = modesFor(model);
      const overlays = overlaysFor(model, backend, axesRoute);
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
  const reconciliationCells = [];
  for (const modelSummary of models) {
    const model = manifestById.get(modelSummary.id);
    const route = resolveRoute(model, routes, videoRoutes, modelSummary.backends);
    for (const backend of modelSummary.backends) {
      // SC-15812: resolved HERE, inside the per-backend loop, so a cell names the story that owns
      // its (model, backend) pair rather than whichever backend happened to be listed first.
      const owningFamilyStory = modelSummary.owningFamilyStories[backend];
      const owningModelStory = modelSummary.owningModelStories[backend];
      for (const tier of tiersFor(model, backend, backendTierOverrides)) {
        for (const mode of modesFor(model)) {
          for (const overlay of overlaysFor(model, backend, route)) {
            const provider = providerFor(model, backend, overlay, route, mode);
            for (const rung of RUNGS) {
              const status = strategyStatus({
                backend,
                rung,
                route,
                provider,
                stagedResidencyEngines,
                model,
                tier,
                mode,
                overlay,
                rung4Survey,
                rung4ContractPrerequisites,
                manifestById,
                inferenceClosureDigests,
                candleBespokeStagedLanes,
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
                  geometry: measuredGeometryKey(record.target.geometry),
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
                (record) => calibrationBinding(
                  record,
                  {
                    calibrationFingerprint: fingerprint,
                    engagedRungs,
                    strategyParameters: status.parameters,
                    geometryEnvelope: status.maxPixels
                      ? geometryWithinPixels(model, backend, status.maxPixels)
                      : geometryFor(model, backend),
                    evidence: { loadability: artifactEvidence(model, route, tier) },
                  },
                  {
                    exactPlanEntries: exactPlanEntriesForRecord(calibrationPlan, record),
                    modality: model.type,
                  },
                ).eligible,
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
                // sc-18812: the temporal axis is carried into the key, so two records that differ
                // only in frame count are two measured geometries rather than one. The `xfN`
                // suffix is emitted ONLY above one frame, which is what keeps every image cell's
                // published `measuredGeometries` byte-identical to what it was before.
                ...eligibleRuns.map((record) => measuredGeometryKey(record.target.geometry)),
              ], { declaresTemporalCurve: status.declaresTemporalCurve === true });
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
              reconciliationCells.push(cell);
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
  reconciliationCells.sort((left, right) => left.id.localeCompare(right.id));
  // REPORT-ONLY SEAM (Michael's decision, 2026-08-17). The memory-contract reconciliation may never
  // fail a build — not on a finding, and not on its own internal strictness either. The waiver ledger
  // and its bijection check are deleted; see `reconcileMemoryContracts` for why. This try/catch is the
  // structural guarantee that nothing downstream of here can turn a disagreement into a red build: a
  // future capability re-dump that introduces, say, a load profile this lib has not met yet becomes a
  // line in the report instead of a broken `npm run check`. That exact failure cost two rounds before
  // the gate came out. Do not remove the catch, and do not rethrow from it.
  let memoryContractReconciliation;
  try {
    memoryContractReconciliation = reconcileMemoryContracts({
      pin,
      engineFacts: [
        JSON.parse(bodies.engineCapabilitiesMlx),
        JSON.parse(bodies.engineCapabilitiesCandle),
      ],
      manifest,
      cells: reconciliationCells,
      calibrationPlan,
      closures: JSON.parse(bodies.inferenceClosures),
      survey: JSON.parse(bodies.rung4Survey),
    });
  } catch (error) {
    memoryContractReconciliation = {
      providers: 0,
      bespokeWaivers: 0,
      mismatches: 0,
      byLeg: {},
      findings: [],
      unavailable: error instanceof Error ? error.message : String(error),
    };
  }
  // Report-only side channel: the enumeration is deliberately absent from the checked-in document, so
  // `report-memory-contract-reconciliation.mjs` collects it here instead of reassembling the inputs.
  onReconciliation?.(memoryContractReconciliation);
  // sc-18663: the SAME category error as the calibration-plan guard, one step later. An out-of-matrix
  // family's catalog entries are subtracted from the universe above, so its receipts can never find a
  // cell — and demanding one turned the campaign's first MiniMax-H3 record into a generation failure.
  // Those numbers are carried by the family's `outOfMatrixFamilies.memoryCharacterization`, which is
  // DERIVED from its `measuredGeometries` and validated on every generation, so nothing goes
  // unrecorded by being unbound here.
  //
  // Keyed on the SAME set the subtraction uses (`survey.outOfMatrixCatalogEntries`) and on the
  // record's own `target.modelId`, so the skip is the exact complement of the subtraction and cannot
  // outlive it: the day `familyGroup` learns one of those entries the survey record is refused, the
  // family joins the universe, and its records bind here like any other. A record naming a family in
  // NEITHER set still throws below — this narrows the requirement, it does not remove it.
  //
  // One consequence, stated rather than left to be discovered: `summary.calibrationRuns` counts the
  // BUNDLE, so once an out-of-matrix receipt exists it exceeds `matrix.calibrationRuns.length` by
  // exactly the skipped rows. That is the honest pair — how many receipts exist, and how many bound
  // to a published coordinate — not a discrepancy.
  const calibrationRuns = calibrationBundle.records
    .filter((record) => !outOfMatrixEntries.has(record.target.modelId))
    .map((record) => {
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
      const modality = manifestById.get(record.target.modelId)?.type ?? null;
      return {
        cellId: cell.id,
        binding: calibrationBinding(record, cell, {
          exactPlanEntries: exactPlanEntriesForRecord(calibrationPlan, record),
          modality,
        }),
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
      candidate.rung === "bounded_transformer_residency" &&
      candidate.overlay === "none" &&
      // sc-18815: a pending family has no verdict, so it has no row. It is NOT silently absent —
      // `summary.rung4Survey.pendingFamilyBackends` names it and every one of its rung-4 cells
      // carries `surveyed: false` with the owing story. Synthesising a row here would put a
      // verdict-shaped object full of nulls next to twenty real ones.
      candidate.rung4Survey.surveyed,
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
  // sc-18815: censused per modality. `mlxStagedStaticCoverage` is a claim about the 53 IMAGE entries
  // — its denominator says so — so admitting video must not inflate it. The video `bernini` entry is
  // the concrete case: it shares the `bernini` engine with `bernini_image`, which is already counted,
  // so a modality-blind set would have read 39/53 for a lane that gained nothing.
  const modalityById = new Map(models.map((model) => [model.id, model.modality]));
  const stagedByModality = (modality) =>
    new Set(
      cells
        .filter(
          (cell) =>
            cell.backend === "mlx" &&
            cell.rung === "staged_residency" &&
            isImplemented(cell.state) &&
            modalityById.get(cell.modelId) === modality,
        )
        .map((cell) => cell.modelId),
    );
  const mlxStagedModels = stagedByModality("image");
  const mlxStagedVideoModels = stagedByModality("video");
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
    //
    // 8 (sc-18815): the model universe is MODALITY-AWARE. `summary.imageModels` was REMOVED — not
    // renamed-with-an-alias — and replaced by `catalogEntries` + `catalogEntriesByModality`. Leaving
    // `imageModels` in place reading 63 would be worse than the image-only universe it replaced: a
    // reader would get a plausible number under a false name, where now it gets `undefined` and has
    // to look. `models[]` gained required `modality` and `resolvedRoutes` (video providers are
    // per-backend) and grew 53 -> 63; `models[].backends` and `owningModelStories` may be EMPTY for
    // an entry the routing catalog routes nowhere; `cells[].owningModelStory` may be `null`;
    // `cells[].rung4Survey` gained a required `surveyed` discriminator and nullable verdict fields;
    // `rung4SurveyRows[].familyStory` may be a family NAME; and `summary` gained `unroutedEntries`,
    // `videoMlxStagedStaticCoverage`(+Denominator) and `rung4Survey.pendingFamilyBackends`. A
    // version-7 reader that took `owningModelStory` as an integer, or `imageModels` as the entry
    // count, is wrong on both — hence a version, not an additive bump.
    schemaVersion: 8,
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
          "scripts/generate-memory-matrix.mjs#memoryCharacterization — the RANK of the measured " +
          "design matrix, because `fixedGb + perMpxGb*mpx + perMpxFrameGb*mpx*frames` takes as " +
          "many independent geometries as the lane has coefficients: two on the image lane, and " +
          "three once a temporal term is carried, where three frame counts at one area are still " +
          "singular. Which form applies is read from the cell's DECLARED curve first — a curve " +
          "carrying `perMpxFrameGb` is graded against three coefficients even if every eligible " +
          "record happens to be single-frame",
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
          "Measured, but not enough to determine the curve — one geometry, or several that are " +
          "linearly dependent (two resolutions of one area; several frame counts at one area). " +
          "The peak is known AT those geometries and the slopes are undeterminable, so nothing " +
          "is known about the rest of the envelope.",
      },
      {
        status: "fitted",
        definition:
          "Measured geometries of full design rank — sufficient to determine the affine curve, " +
          "which is not a claim that a fit has been performed. `coveredPixelBound` is the largest " +
          "measured area; `coveredFrameBound`, present only on cells carrying a temporal geometry, " +
          "is the largest measured frame count.",
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
      // sc-18815: `imageModels` became a lie the moment video was admitted — it counted the whole
      // universe under a name that claimed one modality. `catalogEntries` is the total the field
      // always actually held, and `catalogEntriesByModality` is the breakdown the old name pretended
      // to be. Renamed rather than kept-and-supplemented: a field named `imageModels` reading 63 is
      // worse than the state this story replaced, and sc-18830 carries the doc follow-through.
      catalogEntries: models.length,
      catalogEntriesByModality: Object.fromEntries(
        [...MATRIX_MODALITIES].map((modality) => [
          modality,
          models.filter((model) => model.modality === modality).length,
        ]),
      ),
      // The number of coordinates the CATALOG resolved to, unchanged in meaning by sc-18099 and no
      // longer equal to `cells.length`. `publishedCells` and `elidedCells` partition it.
      cells: cells.length,
      publishedCells: cells.length,
      elidedCells: 0,
      elidedByState: {},
      publicationPredicate: PUBLICATION_PREDICATE,
      // Still exactly the image-lane claim it has always been (sc-18815); the video lane is counted
      // beside it rather than folded in, because they are different populations with different
      // evidence and one number covering both could not say which lane moved.
      mlxStagedStaticCoverage: mlxStagedModels.size,
      mlxStagedStaticCoverageDenominator: EXPECTED_IMAGE_COUNT,
      videoMlxStagedStaticCoverage: mlxStagedVideoModels.size,
      videoMlxStagedStaticCoverageDenominator: EXPECTED_VIDEO_COUNT,
      // sc-18815: entries in the universe that the routing catalog routes nowhere, and therefore
      // resolve to zero coordinates. Published so a reader can tell them from entries that are
      // simply not in the catalog — see `UNROUTED_CATALOG_ENTRIES`.
      unroutedEntries: [...UNROUTED_CATALOG_ENTRIES]
        .filter(([id]) => models.some((model) => model.id === id))
        .map(([id, entry]) => ({ id, ...entry }))
        .sort((left, right) => left.id.localeCompare(right.id)),
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
      // Counts only. The per-coordinate enumeration lives in
      // `npm run report:memory-contract-reconciliation`, deliberately NOT checked in: a report that
      // rotates with every capability dump would recreate exactly the regeneration churn that
      // retiring the waiver ledger removed.
      memoryContractReconciliation: (({ findings, ...summary }) => summary)(
        memoryContractReconciliation,
      ),
      rung4Survey: {
        story: 15969,
        surveyedFamilyBackends: rung4SurveyRows.length,
        // sc-18815: the families in the universe that no verdict covers yet, and who owes each one.
        // Named here so "surveyed" and "in the matrix" can never be read as the same set again.
        pendingFamilyBackends: [...PENDING_RUNG4_SURVEYS]
          .flatMap(([group, owner]) =>
            ["mlx", "candle"]
              .filter((backend) =>
                models.some(
                  (model) => familyGroup(model.id) === group && model.backends.includes(backend),
                ),
              )
              .map((backend) => ({ family: group, backend, pendingSurveyStory: owner })),
          )
          .sort((left, right) =>
            `${left.family}:${left.backend}`.localeCompare(`${right.family}:${right.backend}`),
          ),
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
    rung4ContractPrerequisites,
    cellInventoryExpectations,
    calibrationPlan,
    catalogFamilyBackends(manifest.models, routedBackends),
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

export function renderMarkdown(matrix) {
  const lines = [
    "# Generated memory-ladder matrix",
    "",
    "> Generated by `scripts/generate-memory-matrix.mjs`. Do not edit by hand.",
    "",
    `- SceneWorks revision: \`${matrix.generatedFrom.sceneWorksRevision}\``,
    `- Inference revision: \`${matrix.generatedFrom.inferenceRevision}\``,
    `- Catalog entries: ${matrix.summary.catalogEntries} (${
      Object.entries(matrix.summary.catalogEntriesByModality)
        .map(([modality, count]) => `${modality} ${count}`)
        .join(", ")
    })`,
    ...(matrix.summary.unroutedEntries.length
      ? [
          `- Unrouted entries (zero resolved coordinates): ${matrix.summary.unroutedEntries
            .map((entry) => `\`${entry.id}\` — ${entry.reason} (sc-${entry.owningStory})`)
            .join("; ")}`,
        ]
      : []),
    `- Resolved coordinates: ${matrix.summary.cells}`,
    `- Published cells: ${matrix.summary.publishedCells}`,
    `- Elided coordinates: ${matrix.summary.elidedCells} (${
      Object.entries(matrix.summary.elidedByState)
        .map(([state, count]) => `${state} ${count}`)
        .join(", ") || "none"
    })`,
    `- MLX staged-residency static coverage: image ${matrix.summary.mlxStagedStaticCoverage}/${matrix.summary.mlxStagedStaticCoverageDenominator}, video ${matrix.summary.videoMlxStagedStaticCoverage}/${matrix.summary.videoMlxStagedStaticCoverageDenominator}`,
    `- Full models: ${matrix.summary.fullModels}`,
    `- Full complete calibration records: ${matrix.summary.calibrationRunsByStatus.complete}`,
    `- Base-only runtime-complete calibration records: ${matrix.summary.calibrationRunsByStatus.runtimeComplete}`,
    "",
    `sc-18099: \`cells\` is a SUBSET. ${matrix.summary.publicationPredicate} The counts on this page, \`summary\`, and the per-(entry, backend, rung) \`coverage\` census in the JSON artifact are all derived from every resolved coordinate, published or not, and \`models[].axes\` publishes the axes those coordinates span so an unmeasured lane stays distinguishable from an absent one.`,
    "",
    "Static capability is never promoted to dynamic verification. The six evidence dimensions stay separate: `staticImplementation`, `historicalVerification`, `currentEnvironmentVerification`, `strategyParameterVerification` and `structural` are per-coordinate and ride the cell; `declaredCalibration` and `loadability` are functions of (entry, backend, tier) alone and are published once per scope in `manifestScopes`, which the cell names through `evidence.manifestScope` (sc-18099).",
    "`Runtime verified` means the exact base-only coordinate is production-admissible from current runtime evidence; it is deliberately not Full `Verified`, which additionally requires the catalog story's lifecycle and negative-mutation signoff.",
    "",
    "sc-18864: `observedPeakGb` is the ALLOCATOR BOUND — a phase's peak-over-window `activeBytes` summed with its instantaneous end-of-phase `reclaimableBytes`, which MLX releases under pressure. It is an upper bound on co-existence, so it is a FOOTPRINT figure, not a feasibility figure, and it may legitimately exceed the capture host. `mlxRequiredHostBytes` is the figure that sizes a host: it reads the non-reclaimable residency instead.",
    "One row per (catalog entry, backend): ownership is backend-scoped, so a single row per entry could only name one backend's stories (SC-15812).",
    "",
    "sc-18815: the `Modality` column exists because the universe is no longer one modality. Video entries carry no per-entry ownership story — epic 18803 does not slice video that way, so `Model story` is `—` rather than a story id that could not close the cell — and their family story is the family's rung-4 survey story, which covers both backends.",
    "",
    "| Catalog entry | Modality | Backend | Route | Family story | Model story | Staged residency |",
    "| --- | --- | --- | --- | --- | ---: | --- |",
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
      // sc-20799: a lane whose EVERY staged coordinate is structurally exempt is not `Missing` —
      // "nobody has written it yet" and "the architecture has no separable component to stage" are
      // different claims, and conflating them is how a resident-only-by-design lane reads as
      // undelivered work. A mixed lane (some exempt, some genuinely absent) still reads `Missing`.
      const structuralOnly =
        !staged &&
        row.coordinates > 0 &&
        (row.states["Structurally N/A"] ?? 0) === row.coordinates;
      lines.push(
        `| \`${model.id}\` | ${model.modality} | ${backend} | \`${model.resolvedRoutes[backend]}\` (${model.routeKind}) | SC-${model.owningFamilyStories[backend]} | ${model.owningModelStories[backend] === null ? "—" : `SC-${model.owningModelStories[backend]}`} | ${staged ? stagedState : structuralOnly ? "Structurally N/A" : "Missing"} |`,
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
    "| Family | Backend | Structural applicability | Implementation | Request peak |",
    "| --- | --- | --- | --- | --- |",
  );
  for (const row of matrix.rung4SurveyRows) {
    lines.push(
      `| ${familyLabel(row.familyStory)} | ${row.backend} | ${row.structuralApplicability} | ${row.implementation} | ${row.requestPeak} |`,
    );
  }
  for (const row of matrix.summary.rung4Survey.pendingFamilyBackends) {
    lines.push(
      `| ${familyLabel(row.family)} | ${row.backend} | _not surveyed (sc-${row.pendingSurveyStory})_ | — | unsurveyed |`,
    );
  }
  lines.push(
    "",
    `Surveyed family/backend pairs: ${matrix.summary.rung4Survey.surveyedFamilyBackends}; awaiting a verdict: ${matrix.summary.rung4Survey.pendingFamilyBackends.length}. sc-18815: a pending pair's rung-4 cells report \`Missing\` and carry \`rung4Survey.surveyed: false\` with the owing story, so "not surveyed yet" never reads as "surveyed and found wanting". sc-18099 split the verdict by what it is a property OF: the family-level summary, block-stack inventory and findings are on \`rung4SurveyRows\` in the JSON artifact — carried once per (family, backend), so they survive a family whose rung-4 cells were all elided — while \`cells[].rung4Survey\` keeps the genuinely per-coordinate half, the resolved request-peak finding and the overlay-incompatibility verdict.`,
    "",
  );
  return lines.join("\n");
}

async function main() {
  // Report-only path (Michael, 2026-08-17): emit the reconciliation enumeration and nothing else, so
  // the report script and the generator share one code path. Never fails on findings.
  if (process.argv.includes("--emit-reconciliation")) {
    let reconciliation = null;
    try {
      await buildMatrix({ onReconciliation: (value) => (reconciliation = value) });
    } catch (error) {
      // The reconciliation is computed BEFORE `validateMatrix`, so an unrelated matrix invariant must
      // not deny the report its enumeration. If the callback already fired we have the findings; emit
      // them and note what stopped the rest of the build.
      if (!reconciliation) {
        reconciliation = { providers: 0, bespokeWaivers: 0, mismatches: 0, byLeg: {}, findings: [] };
      }
      reconciliation.buildIncomplete = error instanceof Error ? error.message : String(error);
    }
    process.stdout.write(`${JSON.stringify(reconciliation)}\n`);
    return;
  }
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
