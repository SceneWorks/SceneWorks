#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import { canonicalSourceText, semanticSourceBody } from "./lib/source-revision.mjs";
import { routedLanes } from "./check-tier-integrity.mjs";
import { CONVERTER_TIER_OVERRIDES, contractIsLoraOnly } from "./lib/manifest-memory-declarations.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT_JSON = "docs/generated/memory-matrix.json";
const OUTPUT_MD = "docs/generated/memory-matrix.md";
// sc-18815: the modalities the matrix carries. `utility` and `audio` entries have no memory ladder —
// no rungs, no fit gate, no strategy selector — so they are outside the universe by design rather
// than by omission, and admitting one would need the same three things video needed here.
const MATRIX_MODALITIES = new Set(["image", "video"]);
// sc-22512 deleted `EXPECTED_IMAGE_COUNT = 53` and `EXPECTED_VIDEO_COUNT = 11`. They were frozen
// catalog populations: adding one model to the shipped catalog reddened generation before anything
// had a chance to say whether the new entry was well-formed. The population is now DERIVED from the
// catalog's own enumeration, and the cross-table agreement between the manifest, the source-owned
// `EXPECTED_IMAGE_IDS` roster and the generated ownership rows carries the whole structural claim —
// that check reds on disagreement, which is present-and-contradictory data, at any population size.
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
//   guarded — bespoke routes never claiming the generic ladder; per-route drift, where entries
//             sharing a resolved route disagree with each other; and the census being contained in
//             the image entries.
//   NOT guarded, as of sc-22512 — which entries are IN the census, and how many. The two named-lane
//             requirements (flux2_dev, bernini_image) and the "neither empty nor the whole catalog"
//             band were removed: all three reddened on the ABSENCE of a static-residency
//             declaration, which an inference pin that stops advertising one, or a catalog whose
//             image lane declares none, produces without anything being wrong with this document.
//             Measurement improves the estimate; its absence is the conservative reading, not a
//             defect. The two ids above are kept as PROSE because the provenance is still worth
//             reading, not as assertions.
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
  if (modelId.startsWith("ltx_2_3") || modelId === "ltx_2_5") return "ltx-video";
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
// beside `MATRIX_MODALITIES` above). Runs inside `validateMatrix`, so `cells` is still the full resolved
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
  // sc-22512 removed the two named-entry census requirements (`flux2_dev` and `bernini_image` were
  // each asserted INTO the census) and the `0 < size < models` partial-coverage band. All three
  // reddened on the ABSENCE of a declaration: an inference pin that stops advertising selectable
  // Sequential residency for one provider, or a catalog whose whole image lane declares none, is a
  // lane nobody has measured — not a defect in this document. What survives is the containment
  // relation, which holds at any coverage level including zero. The mirror of this removal lives in
  // tests/test_memory_matrix.py.
  //
  // The containment replacement that first stood here — "every staged id is an id the matrix
  // universe knows" — was deleted rather than kept, because it could not fail: `staged` is derived
  // by filtering `matrix.cells`, and every cell is generated FROM `matrix.models`, so the id set is
  // a subset of the universe by construction. A check whose throw branch is unreachable reads as
  // coverage while asserting nothing, which is worse than the absent gate it replaced. The live
  // claim below — a claim two independent declarations can genuinely disagree on — is what carries
  // this function.
  //
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

// "Implemented" is a claim about the CODE, and it is carried by every state the collapsed
// vocabulary spells for an implemented rung (sc-22513). Coverage surfaces must go through here
// rather than re-spelling the comparison, so a state added to `IMPLEMENTED_STATES` cannot silently
// drop out of a count.
export function isImplemented(state) {
  return IMPLEMENTED_STATES.includes(state);
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

/**
 * Direct-only Candle dispatches that deliberately do not make a user-facing Candle lane.
 *
 * `parseVideoRoutes` is the matrix's public-route parser: its output is joined with the routing
 * catalog, so adding a direct/replay executor there would make an unmeasured lane look like a
 * schedulable capability. MiniMax-H3 is the current case. Its resolver is intentionally kept in
 * `video_jobs/mod.rs`, outside `candle_video_engine_id`; parse it independently so the contract
 * still proves the internal arm names the real provider while the matrix keeps it excluded until
 * `candle_video_routed` flips.
 */
export function parseInternalCandleVideoRoutes(candleDispatchSource, minimaxSource) {
  const directRouteArm =
    "} else if let Some(engine_id) = minimax_h3_engine_id(&request.model) {\n" +
    "        CandleVideoRoute::MiniMaxH3(engine_id)\n" +
    "    } else if is_candle_video_engine(&request.model) {";
  if (!candleDispatchSource.includes(directRouteArm)) {
    throw new Error(
      "memory-matrix: MiniMax-H3 direct Candle dispatch no longer selects CandleVideoRoute::MiniMaxH3",
    );
  }
  const engine = minimaxSource.match(
    /const\s+MINIMAX_H3_ENGINE_ID:\s*&str\s*=\s*"([^"]+)"\s*;/,
  )?.[1];
  if (
    !engine ||
    !minimaxSource.includes(
      "sceneworks_core::video_request::is_minimax_h3_model(model).then_some(MINIMAX_H3_ENGINE_ID)",
    )
  ) {
    throw new Error(
      "memory-matrix: could not resolve MiniMax-H3's shared internal Candle engine id",
    );
  }
  return new Map([
    ["minimax_h3", engine],
    ["minimax_h3_ref", engine],
  ]);
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

/**
 * The per-lane tier overrides that come from CODE rather than from a manifest declaration: the
 * InstantID Candle dense tier, read out of the worker's own `instantid.rs`, plus the converter
 * families' packed tier sets. These are ROUTING facts — what the lane can load at all.
 *
 * Exported (sc-22729) so the measurability gap set can narrow a model's tier axis by exactly these
 * and nothing else. `tiersFor` also consults `model[backend].vramGbByTier`, which is a MEASUREMENT
 * declaration: a missing key there says a peak has not been recorded, never that the lane refuses
 * the tier, so a gap set that intersected against it would delete the very cells it exists to
 * count.
 */
export function parseBackendTierOverrides(instantIdSource) {
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
 * `Implemented` (inference `crates/contracts/gen-core/src/memory_strategy.rs:1544-1558` at the
 * pinned rev `28f0563baa03640ade1635356d2d54fe8a477f1a`: `required_by_realization` is true by construction, so the conjunction is
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

/**
 * The IMPLEMENTATION axis of a cell (sc-22513, epic 22505 E5).
 *
 * This is a claim about CODE and CATALOG only: does this route, on this backend, at this
 * (tier, mode, overlay), implement this rung — and it is answered from the manifest's declared
 * memory contract, the worker's routing and staged-residency sources, and nothing else. It carries
 * no evidence bookkeeping: no calibration fingerprint, no engaged-rung join, no record currency, no
 * per-geometry characterization. Those were the joins E5 deletes; the MEMORY axis is now exactly
 * `(anchor present?, derivation defined?)` and is combined with this verdict by `cellState`.
 *
 * Three verdicts, and they are exhaustive:
 *   `implemented`      — the rung exists on this coordinate, with a source that says why.
 *   `structurally-na`  — the rung cannot apply to this architecture (a declared exemption).
 *   `missing`          — no implementation of this rung on this route.
 */
export function implementationVerdict({
  backend,
  rung,
  route,
  provider,
  stagedResidencyEngines,
  model,
  tier,
  mode,
  overlay,
  manifestById,
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
  const missing = { implementation: "missing", source: null, parameters: {} };
  // A routing-table lane without a per-backend manifest block is real but wholly untriaged: the
  // optional block is evidence/tuning metadata, not lane-existence metadata. Emit the full slice as
  // missing so the work can be seen and owned instead of either hiding the lane or inferring
  // implementation claims from another backend.
  if (!declaredModel[backend]) return missing;
  const staticMemoryContract = declaredModel[backend]?.memoryStrategyContract;
  const staticImplementation = staticContractCoversProvider(staticMemoryContract, provider)
    ? staticMemoryContract.implementations.find(
        (implementation) =>
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
  // The manifest's per-tier calibration rows are read as a DECLARATION that the rung exists on this
  // coordinate — that is the only thing they still say here. Their fingerprints, engaged rungs and
  // closure currency were the record-join E5 deletes.
  const declaredCalibrationRows = (model[backend]?.calibrations ?? []).filter(
    (binding) =>
      binding.provider === provider &&
      binding.tier === tier &&
      binding.mode === mode &&
      matrixOverlayFor(binding.overlay) === overlay &&
      binding.rung === rung,
  );
  if (staticContractIsExhaustive && !staticImplementation) {
    // A catalog self-consistency guard, not an evidence guard: an EXHAUSTIVE provider contract that
    // does not cover this coordinate while the same entry declares a calibration row for it is a
    // manifest contradiction, and it survives the collapse because nothing about it is measured.
    if (declaredCalibrationRows.length) {
      throw new Error(
        `${model.id}:${backend}:${tier}:${mode}:${overlay}:${rung} declares calibration ` +
          `outside exhaustive provider contract ${staticMemoryContract.provider}`,
      );
    }
    return missing;
  }
  // Semantic quality receipts authorize exact geometry choices in the runtime planner. They are
  // not numeric tuning ranges and must never be projected into the published matrix, where their
  // presence could be mistaken for measured memory evidence.
  const publishableParameterRanges = (implementation) => {
    const { decodeGeometryPolicies: _semanticReceipt, ...publishedRanges } =
      implementation?.parameterRanges ?? {};
    return publishedRanges;
  };
  // A COVERAGE-ONLY declaration (sc-20246) states rung x tier x mode x overlay coverage and no
  // parameters, so it must not displace a richer arm below that has parameters to publish. It is
  // used only where nothing else answers.
  const coverageDeclaration =
    staticImplementation && !staticImplementation.parameters ? staticImplementation : null;
  const coverageVerdict = () =>
    coverageDeclaration
      ? { implementation: "implemented", source: coverageDeclaration.source, parameters: {} }
      : missing;
  if (staticImplementation && !coverageDeclaration) {
    return {
      implementation: "implemented",
      source: staticImplementation.source,
      parameters: {
        ...staticImplementation.parameters,
        ...(staticImplementation.parameterRanges
          ? { publishedRanges: publishableParameterRanges(staticImplementation) }
          : {}),
      },
    };
  }
  if (declaredCalibrationRows.length) {
    const parameters = sortedUnique(
      declaredCalibrationRows.map((binding) => JSON.stringify(binding.parameters ?? {})),
    );
    if (parameters.length !== 1) {
      throw new Error(
        `${model.id}:${backend}:${tier}:${mode}:${overlay}:${rung} has inconsistent declared calibration parameters`,
      );
    }
    return {
      implementation: "implemented",
      source: `config/manifests/builtin.models.jsonc#models/${model.id}/${backend}/calibrations`,
      parameters: JSON.parse(parameters[0]),
    };
  }
  // `modes` and `tiers` are OPTIONAL narrowings, absent meaning "every mode"/"every tier" — the
  // same shape `memoryStrategyCapabilities` uses below. A structural exemption that cannot be
  // narrowed by mode would have to claim the rung is inapplicable on modes whose streaming path was
  // never built at all, which reads as "surveyed and exempted" rather than "not implemented"
  // (sc-22513).
  const staticExemption = declaredModel[backend]?.memoryStrategyStructuralExemptions?.[rung];
  if (
    staticExemption?.overlays?.includes(overlay) &&
    (!staticExemption.modes || staticExemption.modes.includes(mode)) &&
    (!staticExemption.tiers || staticExemption.tiers.includes(tier))
  ) {
    return {
      implementation: "structurally-na",
      source: staticExemption.evidence[0].source,
      parameters: {},
      structural: staticExemption.evidence,
    };
  }
  const staticCapability = declaredModel[backend]?.memoryStrategyCapabilities?.[rung];
  if (
    staticCapability?.overlays?.includes(overlay) &&
    (!staticCapability.tiers || staticCapability.tiers.includes(tier))
  ) {
    return {
      implementation: "implemented",
      source: `config/manifests/builtin.models.jsonc#models/${declaredModel.id}/${backend}/memoryStrategyCapabilities/${rung}`,
      parameters: staticCapability.parameters,
    };
  }
  if (
    rung === "resident" &&
    !(model.id === "krea_2_turbo" && backend === "candle" && mode === "text_to_image") &&
    (backend !== "candle" ||
      model.id !== "pulid_flux_dev" ||
      staticCandleOverlayIsAvailable({ model, route, overlay, manifestById }))
  ) {
    return {
      implementation: "implemented",
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
      implementation: "implemented",
      source:
        backend === "mlx"
          ? "crates/sceneworks-worker/src/mlx_fit_gate.rs#engine_engages_staged_residency"
          : `config/manifests/builtin.models.jsonc#models/${model.id}/candle`,
      parameters: { phaseOrder: ["conditioning", "denoise", "decode"] },
    };
  }
  // Krea 2 Turbo's Candle ladder is declared per rung in the catalog's own `turboFit` block. Only
  // the DECLARATION survives here — which rungs it implements and their strategy parameters. Its
  // evidence records, verification rows and measured pixel envelope were per-geometry bookkeeping.
  if (
    model.id === "krea_2_turbo" &&
    backend === "candle" &&
    mode === "text_to_image" &&
    overlay === "none" &&
    model.candle?.turboFit?.phaseCurvesByTier?.[tier]
  ) {
    const manifestRung = {
      resident: "resident",
      staged_residency: "threeStage",
      bounded_decode: "tiledVae",
      bounded_attention: "chunkedAttention",
      bounded_transformer_residency: "streamedBlocks",
    }[rung];
    // `resident` is the ONE rung this block prices without a phase curve — its formula is
    // `vramGbByTier + cudaHeadroom`, and `phaseCurvesByTier` carries no `resident` key on any tier —
    // so it is declared by `strategyParameters.resident` alone. Every other rung must carry its own
    // curve for this tier, or the block is not declaring it (sc-22513: requiring a curve for
    // `resident` too silently unimplemented Krea Turbo's Candle resident rung on all three tiers).
    const declaresRung =
      manifestRung === "resident"
        ? model.candle.turboFit.strategyParameters?.resident !== undefined
        : Boolean(model.candle.turboFit.phaseCurvesByTier[tier][manifestRung]);
    if (manifestRung && declaresRung) {
      return {
        implementation: "implemented",
        // The DATA read here is the manifest's own `turboFit` block, not the worker's gate: cite
        // where the declaration lives (sc-22513).
        source: `config/manifests/builtin.models.jsonc#models/${model.id}/candle/turboFit`,
        parameters: {
          manifestRung,
          formula:
            manifestRung === "resident"
              ? "vramGbByTier+cudaHeadroom"
              : "max(text,denoise,decode)+cudaHeadroom",
          ...model.candle.turboFit.strategyParameters?.[manifestRung],
        },
      };
    }
  }
  return coverageVerdict();
}

/**
 * The COLLAPSED cell state (sc-22513, epic 22505 E5).
 *
 * A PURE function of four code-and-store facts, and of nothing else:
 *
 *   `implementation`     — `implementationVerdict`'s verdict: a claim about code and catalog. A rung
 *                          that is not implemented is still not implemented; that axis was never
 *                          measurement bookkeeping and it stays.
 *   `anchorPresent`      — the anchor store (`config/memory-anchors.json`) holds a measured anchor
 *                          for this cell's (model, tier, backend lane).
 *   `derivationDefined`  — the analytic derivation is defined AND wired for this cell's lane, read
 *                          off the Rust (`memory_anchor.rs` declares it, the lane's real admission
 *                          source calls it).
 *   `anchorDerivable`    — the ANCHOR itself is one the lane's law accepts (epic 22505 feature-end
 *                          fix round, E5): an anchor the store marks `underivedReason` — an
 *                          axis-free video anchor the pipeline-keyed law refuses, a
 *                          single-geometry image model no coefficient can be fitted for —
 *                          validates its measured point and prices nothing, so a wired lane still
 *                          publishes it as `Anchored/underived` rather than claiming a derived
 *                          peak. The store field and the Rust laws are bound to each other (every
 *                          law refuses an anchor carrying the field), so this is a store fact, not
 *                          a parallel opinion.
 *
 * Nothing else may enter: not a record, not a plan row, not a geometry, not a campaign, not a
 * currency digest. Anchor CURRENCY is reported on the cell beside the state (sc-22511 makes it a
 * report, never a gate) and deliberately does not move it — a staled loader closure means the
 * anchor needs re-extraction, not that the rung stopped existing.
 *
 * The state vocabulary that replaces Missing / Implemented-unverified / Runtime-verified / Verified:
 *
 *   `Missing`           — no implementation of this rung on this route.
 *   `Structurally N/A`  — the rung cannot apply to this architecture.
 *   `Implemented`       — implemented; its peak is priced by the analytic floor, not by an anchor.
 *   `Anchored`          — implemented, a measured anchor covers it, and the lane's derivation can
 *                         price an arbitrary request from that anchor.
 *   `Anchored/underived`— implemented and anchored, but the anchor bounds nothing beyond its own
 *                         measured point: the lane has no wired derivation, or the lane's law
 *                         refuses this anchor (`anchorDerivable` false).
 */
export function cellState({ implementation, anchorPresent, derivationDefined, anchorDerivable }) {
  if (implementation === "missing") return "Missing";
  if (implementation === "structurally-na") return "Structurally N/A";
  if (implementation !== "implemented") {
    throw new Error(`unknown implementation verdict ${JSON.stringify(implementation)}`);
  }
  if (!anchorPresent) return "Implemented";
  return derivationDefined && anchorDerivable ? "Anchored" : "Anchored/underived";
}

/** The states that assert the rung's CODE exists. */
export const IMPLEMENTED_STATES = Object.freeze([
  "Implemented",
  "Anchored",
  "Anchored/underived",
]);

function validateMatrix(matrix, expectedIds, backendTierOverrides, cellInventoryExpectations) {
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
  // sc-22512: the two pinned modality populations (53 image / 11 video) were deleted. They reddened
  // on a catalog that GREW — an extra shipped model refused generation outright — which is the
  // measurement-absence failure surface this story removes.
  //
  // Their first replacement, `ids.length + videoIds.length === matrix.models.length`, was deleted
  // again rather than kept: with `MATRIX_MODALITIES` exhausted by the unadmitted-modality throw
  // directly above, the two filters partition `matrix.models` by arithmetic, so the branch could
  // never be taken. It read as coverage while asserting nothing. The duplicate-id check below is
  // the half of that pair that a real catalog CAN violate — two entries sharing an id — and it
  // holds at any population size, zero included.
  if (new Set([...ids, ...videoIds]).size !== ids.length + videoIds.length) {
    throw new Error("matrix carries duplicate model ids across the modality partition");
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
  assertTwinCoverage(matrix.models);
  assertVideoOwnership(matrix.models);
  assertUnroutedEntriesAreDeclared(matrix.models);
  assertCellOwnershipIsBackendScoped(
    matrix.cells,
    buildStoryBackendScope(),
    new Map(matrix.models.map((model) => [model.id, model.modality])),
  );
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
  // sc-22513: the state is a PURE FUNCTION of the cell's own three published facts, so the guard is
  // that recomputing it from those facts reproduces it — and that two cells agreeing on the triple
  // can never disagree on the state. The second half is what catches a state that has quietly
  // acquired a fourth input: the outside term would have to split one triple into two states.
  const statesByFacts = new Map();
  for (const cell of matrix.cells) {
    const facts = {
      implementation: cell.implementation,
      anchorPresent: cell.anchor !== null,
      derivationDefined: cell.derivationDefined,
      anchorDerivable: cell.anchor !== null && cell.anchor.derivable,
    };
    const recomputed = cellState(facts);
    if (recomputed !== cell.state) {
      throw new Error(
        `${cell.id}: state ${cell.state} is not cellState(${JSON.stringify(facts)}) = ${recomputed}`,
      );
    }
    const key = JSON.stringify(facts);
    const seen = statesByFacts.get(key);
    if (seen && seen.state !== cell.state) {
      throw new Error(
        `${cell.id} and ${seen.id} share the state facts ${key} but hold ${cell.state} and ${seen.state}: ` +
          "the cell state depends on something outside (anchor store, derivation, catalog)",
      );
    }
    if (!seen) statesByFacts.set(key, { id: cell.id, state: cell.state });
    if (cell.state !== "Missing" && cell.evidence.staticImplementation.length === 0) {
      throw new Error(`${cell.id}: non-Missing classification has no static evidence`);
    }
    if (cell.state === "Structurally N/A" && cell.evidence.structural.length === 0) {
      throw new Error(`${cell.id}: Structurally N/A classification has no structural evidence`);
    }
    // An anchor may only be cited by a cell whose own coordinate it was measured on.
    if (cell.anchor && cell.anchor.tier !== cell.tier) {
      throw new Error(`${cell.id}: cites anchor ${cell.anchor.id}, measured at tier ${cell.anchor.tier}`);
    }
  }
}

/**
 * Why a cell earns a published row (sc-18099, rewritten by sc-22513). Stated here, published in
 * `summary.publicationPredicate`, and implemented by `isPublishableCell` — one wording, so a reader
 * of the artifact can tell exactly what an absent coordinate means.
 */
export const PUBLICATION_PREDICATE =
  "A coordinate is published when it carries a claim a per-lane census cannot: a measured memory " +
  "anchor covers its (model, tier, backend lane), or it is structurally exempt. A bare `Implemented` " +
  "or `Missing` state is a property of the whole lane, replicated across every tier x mode x overlay, " +
  "and is counted in `summary.elidedByState` and the per-(entry, backend, rung) `coverage` census " +
  "instead of restated thousands of times.";

/**
 * The publication predicate.
 *
 * The old arms were evidence-join arms — planned by a calibration plan, bound to a record, carrying
 * measured geometry — and every one of them was the per-record bookkeeping E5 deletes. Two arms are
 * left, and both say something the census cannot: an ANCHOR is a per-(model, tier, lane) fact, and a
 * STRUCTURAL exemption cites evidence that an absent row would misreport as "nothing done here".
 *
 * A bare `Implemented` state is deliberately NOT an arm, for the reason sc-18099 gave: it is a claim
 * about the ROUTE, replicated across every tier x mode x overlay, and `coverage[].implemented` (with
 * `implementedBy` marginals on mixed lanes) answers it for every resolved coordinate. Admitting it
 * would publish the 9,305-coordinate cross-product again under a different name — a 6.8 MB artifact
 * whose whole purpose is to stop repeating itself.
 */
export function isPublishableCell(cell) {
  // An anchor is a fact about (model, tier, lane) and therefore rides every rung of that lane,
  // including rungs the route does not implement. It publishes a row only where there is a rung for
  // it to say something about.
  if (cell.state === "Missing") return false;
  return cell.anchor !== null || cell.evidence.structural.length > 0;
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
  for (const [modelId, slice] of Object.entries(matrix.modelSlices)) {
    for (const id of slice) {
      if (!publishedIds.has(id)) throw new Error(`modelSlices.${modelId} names unpublished cell ${id}`);
    }
  }
  // sc-22513: every anchor a published cell cites must be a row of the published anchor inventory,
  // and every inventoried anchor that covers a resolved coordinate must be cited by one. A cited
  // anchor that no longer exists, or an anchor silently reaching no cell, is the same defect seen
  // from two sides — and the second direction is what stops the store drifting away from the matrix.
  const inventory = new Set(matrix.anchors.map((anchor) => anchor.id));
  for (const cell of matrix.cells) {
    if (cell.anchor && !inventory.has(cell.anchor.id)) {
      throw new Error(`${cell.id}: cites anchor ${cell.anchor.id}, which the document does not publish`);
    }
  }
  const cited = new Set(matrix.cells.filter((cell) => cell.anchor).map((cell) => cell.anchor.id));
  for (const anchor of matrix.anchors) {
    if (anchor.cells > 0 && !cited.has(anchor.id)) {
      throw new Error(`anchors.${anchor.id} reports ${anchor.cells} cells but no published cell cites it`);
    }
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

// ── sc-22513: the memory axis ──────────────────────────────────────────────────────────────────
//
// Two facts, and the matrix reads nothing else about memory. Both are derived from checked-in
// sources the fingerprint covers, so a cell's state is reproducible from the tree alone.

/**
 * Index the anchor store by the anchor's identity cell — `(model, backend lane, tier)`, which is the
 * coordinate an anchor is measured AT and the coordinate a cell asks about.
 *
 * The store's second half (`analyticOnly`) is deliberately not indexed as coverage: an analytic-only
 * row is the store's explicit statement that no retained render anchors that cell, which is the
 * `anchorPresent: false` case rather than a third one. It is published as an inventory count so the
 * absence stays visible.
 */
export function indexAnchorStore(body) {
  const store = JSON.parse(body);
  const anchors = new Map();
  for (const anchor of store.anchors ?? []) {
    const key = `${anchor.modelId}:${anchor.backend}:${anchor.tier}`;
    const existing = anchors.get(key);
    // The store may hold several anchors for one identity cell (different pipeline axes). The cell
    // asks a yes/no question, so the first by id is cited and the rest are counted — never merged,
    // which would invent a coordinate no render measured.
    if (!existing || anchor.id.localeCompare(existing.id) < 0) anchors.set(key, anchor);
  }
  return {
    anchors,
    total: (store.anchors ?? []).length,
    analyticOnly: (store.analyticOnly ?? []).length,
  };
}

/** The loader-closure digest table (sc-22511). Currency is REPORTED on a cell, never gated. */
export function indexLoaderClosures(body) {
  const parsed = JSON.parse(body);
  return new Map(
    Object.entries(parsed.models ?? {}).map(([key, entry]) => [key, entry.digest ?? null]),
  );
}

/**
 * The store's own record of HOW an anchor's currency key was derived (sc-22667): `null` when at
 * the record's measurement revision, else the reviewed attestation — measured and attested
 * revisions, class, why and witness — that `anchor-loader-closure.mjs --stamp-anchors` copied in
 * from `config/anchor-currency-attestations.json`. Published verbatim so a current anchor never
 * hides whether it is current by measurement or by attestation.
 */
export function currencyAttestationOf(anchor) {
  const attestation = anchor?.source?.currencyAttestation;
  return attestation && typeof attestation === "object" ? { ...attestation } : null;
}

/**
 * The lanes the analytic derivation is DEFINED and WIRED for, as `<modality>:<backend>` keys.
 *
 * Read off the Rust rather than declared here, in both halves, because both halves are code facts:
 * `memory_anchor.rs` declares `derive_<law>_phase_peaks`, and each lane's REAL admission source is
 * where the lane is translated to an `AnchorBackend` and priced from an anchor. A derivation that
 * exists but is called from nowhere is not defined for any lane, and unwiring a lane collapses its
 * cells to `Anchored/underived` rather than leaving them claiming a derived peak.
 *
 * Epic 22505 feature-end fix round (E5): each modality maps to the sources that actually ADMIT it
 * — video through `video_admission.rs`, image through the candle lane's `vram_gate.rs` /
 * `candle_memory_strategy.rs` and the MLX lane's `mlx_fit_gate.rs`. The previous single-source
 * read crossed `video_admission.rs`'s backends with every declared law, so `image:candle` read as
 * wired off a file that never priced an image.
 *
 * `admissionSourcesByLane` maps `<modality>:<backend>` to `{ law, sources, entryPoints? }`: the
 * lane is wired exactly when the law is declared in the derivation source AND some admission
 * source both calls one of the lane's DECLARED entry points and names the lane's
 * `AnchorBackend`. The entry points default to `derive_<law>_phase_peaks` (directly or through the
 * store's `_for_cell` fall-through, which the call-name check also matches by prefix). The image
 * lanes name the ONE image law as well (sc-22667, epic 22657 E3): `derive_phase_peaks` is the law
 * both `derive_image_phase_peaks` and `derive_mlx_image_phase_peaks` translate onto, and since
 * sc-22664/sc-22665/sc-22667 the candle ladder, the Krea lane and the MLX floor call it (or its
 * activation half, `derive_phase_activation_residues`) directly with the rung's regime rather than
 * through the shallow shims — so a lane priced from the law itself is wired, and a lane that calls
 * nothing declared is not.
 */
export function parseAnchorDerivationLanes(derivationSource, admissionSourcesByLane) {
  const declared = new Set(
    [...derivationSource.matchAll(/pub fn derive_([a-z0-9_]+)_phase_peaks/g)].map(
      (match) => match[1],
    ),
  );
  // Every anchor-derivation entry point the source declares, by full name: the per-law shims,
  // the law itself, and its activation half.
  const declaredEntryPoints = new Set(
    [
      ...derivationSource.matchAll(
        /pub fn (derive_[a-z0-9_]*?(?:phase_peaks|phase_activation_residues))\b/g,
      ),
    ].map((match) => match[1]),
  );
  const backendTokens = { mlx: "AnchorBackend::Mlx", candle: "AnchorBackend::Candle" };
  const lanes = new Set();
  for (const [lane, { law, sources, entryPoints }] of Object.entries(
    admissionSourcesByLane,
  )) {
    if (!declared.has(law)) continue;
    const backend = lane.split(":")[1];
    const token = backendTokens[backend];
    if (!token) throw new Error(`unknown backend in derivation lane ${lane}`);
    const calls = (entryPoints ?? [`derive_${law}_phase_peaks`]).filter((name) =>
      declaredEntryPoints.has(name),
    );
    const wired = sources.some(
      (source) => calls.some((name) => source.includes(name)) && source.includes(token),
    );
    if (wired) lanes.add(lane);
  }
  return lanes;
}

/**
 * The image lanes' entry points onto the law (see `parseAnchorDerivationLanes`): the lane shim,
 * the law itself, and — for the MLX floor — the law's activation half.
 */
export const IMAGE_CANDLE_DERIVATION_ENTRY_POINTS = Object.freeze([
  "derive_image_phase_peaks",
  "derive_phase_peaks",
]);
export const IMAGE_MLX_DERIVATION_ENTRY_POINTS = Object.freeze([
  "derive_mlx_image_phase_peaks",
  "derive_phase_activation_residues",
  "derive_phase_peaks",
]);

/**
 * Catalog entries deliberately held OUT of the matrix universe (sc-18663, re-homed by sc-22513).
 *
 * This used to be read from the rung-4 survey, which has left the fingerprint with the rest of the
 * measurement-absence machinery. The fact itself survives the collapse and is not a memory fact at
 * all: `familyGroup` has no arm for MiniMax-H3 and no video-route resolver row exists, so admitting
 * these entries fails generation at `resolveRoute` rather than producing a row. Declared here, in
 * the generator, exactly like `UNROUTED_CATALOG_ENTRIES` — and it fails LOUDLY the day the family is
 * routed, because `assertOutOfMatrixEntriesAreStillUnroutable` refuses an entry the generator can
 * now resolve.
 */
export const OUT_OF_MATRIX_CATALOG_ENTRIES = new Map([
  ["minimax_h3", { epic: 17137, reason: "no familyGroup arm and no video-route resolver row" }],
  ["minimax_h3_ref", { epic: 17137, reason: "no familyGroup arm and no video-route resolver row" }],
]);

/**
 * The subtraction may never outlive its reason. An out-of-matrix entry the catalog no longer carries
 * is a stale declaration, and one the generator CAN now resolve must join the universe rather than
 * stay silently subtracted.
 */
export function assertOutOfMatrixEntriesAreStillUnroutable(manifestModels, resolver) {
  for (const [id, entry] of OUT_OF_MATRIX_CATALOG_ENTRIES) {
    const model = manifestModels.find((candidate) => candidate.id === id);
    if (!model) {
      throw new Error(
        `OUT_OF_MATRIX_CATALOG_ENTRIES names ${id}, which the catalog no longer carries (epic ${entry.epic})`,
      );
    }
    let resolvable = false;
    try {
      resolver(model);
      resolvable = true;
    } catch {
      resolvable = false;
    }
    if (resolvable) {
      throw new Error(
        `${id} is declared out of the matrix, but the generator now resolves its route: admit it to ` +
          `the universe instead of subtracting it (epic ${entry.epic})`,
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
  // ── The POPULATION: which coordinates exist, and which rungs their code implements ───────────
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
  // sc-20799: in SOURCE_PATHS because it DECIDES the implementation verdict —
  // `CANDLE_BESPOKE_REQUEST_PROVIDERS` is where bespoke Candle staged coverage is declared, and a
  // source that changes a cell but sits outside the fingerprint is a provenance hole.
  memoryRouteRegistry: "crates/sceneworks-worker/src/memory_route_registry.rs",
  instantId: "crates/sceneworks-worker/src/image_jobs/instantid.rs",
  // ── The MEMORY axis: the anchor store, its currency declarations, and the derivation ─────────
  //
  // sc-22513 (epic 22505, E5). These four are the whole memory input. What LEFT this map is the
  // per-record evidence join the matrix no longer performs — the calibration plan, the calibration
  // evidence bundle, the inference-provider closure ledger, both rung-4 survey artifacts, the engine
  // capability dumps and the Cargo pin. None of them can move a cell any more, and a source that
  // cannot move a cell must not rotate the artifact's revision: that is what turned a pin bump or a
  // campaign edit into a full regeneration diff.
  anchorStore: "config/memory-anchors.json",
  anchorLoaderClosures: "config/anchor-loader-closures.json",
  // The derivation itself, and each lane's REAL admission source (epic 22505 feature-end fix
  // round, E5). `derivationDefined` is read off these: `memory_anchor.rs` declares
  // `derive_<law>_phase_peaks`, and per modality the file that maps the lane onto an
  // `AnchorBackend` and prices from it is: video -> `video_admission.rs`; image ->
  // `vram_gate.rs` + `candle_memory_strategy.rs` (candle) and `mlx_fit_gate.rs` (MLX, already a
  // population source above).
  anchorDerivation: "crates/sceneworks-core/src/memory_anchor.rs",
  anchorAdmission: "crates/sceneworks-worker/src/video_admission.rs",
  anchorAdmissionImageVram: "crates/sceneworks-worker/src/vram_gate.rs",
  anchorAdmissionImageCandle: "crates/sceneworks-worker/src/candle_memory_strategy.rs",
  // The extractor. The store is a pure function of the retained evidence and this script (sc-22510),
  // so an extractor change is a change to how every anchor was derived.
  anchorExtractor: "scripts/extract-memory-anchors.mjs",
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
 * @param {boolean} [options.publish] When false, return the document BEFORE the publication step:
 *   `cells` is the full resolved cross-product and `modelSlices` covers all of it. `coverage` and
 *   the `summary.publishedCells`/`elidedCells`/`elidedByState` counts are still computed, because
 *   they describe what publication WOULD do and are derived from the full set either way.
 *
 *   This exists for the generator's own tests, which assert which STATE the generator assigns to a
 *   coordinate. That is a claim about generation, not about publication, and `Missing` coordinates
 *   are all elided — asserting them against the published subset would silently make those tests
 *   vacuous. The CLI never passes it.
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
  const manifest = JSON.parse(stripJsoncComments(manifestBody));
  // Comments and formatting are not part of any of these sources' contracts. Hash each source's
  // SEMANTIC body — parsed value for JSON/JSONC, inert whole-line comments removed for Rust and JS —
  // so provenance is stable across semantically inert edits (sc-16129 did the manifest; sc-16268 the
  // rest). Parsing below still reads the raw `bodies`; only provenance reads these.
  const revisionBodies = Object.fromEntries(
    sourceEntries.map(([name, relative]) => [
      name,
      name === "manifest"
        ? manifestRevisionBody(bodies[name])
        : semanticSourceBody(relative, bodies[name]),
    ]),
  );
  // sc-22513: the MEMORY axis, both halves, read once.
  const anchorStore = indexAnchorStore(bodies.anchorStore);
  const loaderClosures = indexLoaderClosures(bodies.anchorLoaderClosures);
  const derivationLanes = parseAnchorDerivationLanes(bodies.anchorDerivation, {
    "video:mlx": { law: "video", sources: [bodies.anchorAdmission] },
    "video:candle": { law: "video", sources: [bodies.anchorAdmission] },
    "image:candle": {
      law: "image",
      sources: [bodies.anchorAdmissionImageVram, bodies.anchorAdmissionImageCandle],
      entryPoints: IMAGE_CANDLE_DERIVATION_ENTRY_POINTS,
    },
    "image:mlx": {
      law: "mlx_image",
      sources: [bodies.mlxFitGate],
      entryPoints: IMAGE_MLX_DERIVATION_ENTRY_POINTS,
    },
  });
  // sc-18815: the model universe is MODALITY-AWARE, not `type === "image"`. Every entry of an
  // admitted modality is in, whether or not anything has been measured on it — an entry the matrix
  // does not carry cannot even report `Missing`, which is how the video lane read as complete while
  // covering one modality. Adding a modality here is deliberate and gated: it needs route resolution
  // (`resolveRoutes`) and a family group.
  //
  // MINUS `OUT_OF_MATRIX_CATALOG_ENTRIES` (sc-18663, re-homed by sc-22513), whose entries the
  // generator cannot resolve a route for at all. The subtraction is validated in both directions
  // below rather than trusted.
  const entries = manifest.models.filter(
    (model) => MATRIX_MODALITIES.has(model.type) && !OUT_OF_MATRIX_CATALOG_ENTRIES.has(model.id),
  );
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
  assertOutOfMatrixEntriesAreStillUnroutable(manifest.models, (model) =>
    resolveRoute(model, routes, videoRoutes, backendScopes(model, routedBackends)),
  );
  // NUL-separated (sc-16268): normalisation strips each body's trailing newline, so concatenating
  // bare would let content shift across a source boundary without moving the hash. A NUL cannot
  // occur in any of these text sources, so it is an unambiguous delimiter.
  const sceneWorksRevision = `source-tree:${sha256(
    sourceEntries.map(([name]) => revisionBodies[name]).join("\0"),
  )}`;

  // sc-16073: no advertised route without cells, and no orphaned control measurements. The worker's
  // MLX and Candle declarations are checked independently rather than trusting their documented twin
  // set, over the WHOLE universe rather than just the image half (sc-18815).
  assertDeclaredControlLanes(entries, bodies.imageRouting);

  const models = entries
    .map((model) => {
      const backends = backendScopes(model, routedBackends);
      const route = resolveRoute(model, routes, videoRoutes, backends);
      return {
        id: model.id,
        name: model.name,
        modality: model.type,
        // sc-18815: the family GROUP key, published so a consumer can join an entry to its family.
        familyGroup: familyGroup(model.id),
        family: model.family ?? null,
        // Video providers are per-backend (LTX is `ltx_2_3` on MLX and `ltx_2_3_distilled` on
        // candle), so the single-valued route the image lane publishes cannot describe them. Publish
        // the resolved provider per backend and keep the scalar as the first-routed-backend one.
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
        // one, so the truthful value is `null` and `assertCellOwnershipIsBackendScoped` enforces the
        // split in both directions.
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
    // sc-18099: the resolved AXES are published on the entry, not just counted here. Publishing only
    // the non-Missing subset would otherwise recreate the sc-16069 blind spot for every unimplemented
    // axis value — no way to tell an unimplemented lane from one that does not exist.
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
  const anchorCellCounts = new Map();
  for (const modelSummary of models) {
    const model = manifestById.get(modelSummary.id);
    const route = resolveRoute(model, routes, videoRoutes, modelSummary.backends);
    for (const backend of modelSummary.backends) {
      // SC-15812: resolved HERE, inside the per-backend loop, so a cell names the story that owns
      // its (model, backend) pair rather than whichever backend happened to be listed first.
      const owningFamilyStory = modelSummary.owningFamilyStories[backend];
      const owningModelStory = modelSummary.owningModelStories[backend];
      // The memory axis is a property of (model, tier, backend lane) and of the lane's derivation —
      // never of the rung, the mode or the overlay — so it is resolved once per tier here and the
      // rung loop below cannot make it depend on anything narrower.
      const derivationDefined = derivationLanes.has(`${modelSummary.modality}:${backend}`);
      for (const tier of tiersFor(model, backend, backendTierOverrides)) {
        const anchor = anchorStore.anchors.get(`${model.id}:${backend}:${tier}`) ?? null;
        const anchorRow = anchor
          ? {
              id: anchor.id,
              tier: anchor.tier,
              source: `config/memory-anchors.json#${anchor.id}`,
              // sc-22511: REPORTED, never gated. A staled loader closure means the anchor needs
              // re-extraction; it does not mean the rung stopped existing, so it may not — and by
              // `cellState`'s signature cannot — move the state.
              current:
                loaderClosures.get(`${anchor.modelId}:${anchor.backend}`) ===
                anchor.source.loaderClosureDigest,
              // sc-22667: HOW the key was derived. `null` means at the record's own measurement
              // revision; otherwise the reviewed currency attestation (config/
              // anchor-currency-attestations.json) that keyed it at a later revision because the
              // closure diff since the measurement is accounting-only or witnessed unchanged.
              currencyAttestation: currencyAttestationOf(anchor),
              // Anchor-level derivability (epic 22505 feature-end fix round, E5): whether the
              // lane's law accepts THIS anchor, read off the store's own `underivedReason` field
              // — which the Rust laws honor byte-for-byte — with the stated reason published so
              // the matrix says WHY a cell is Anchored/underived rather than merely that it is.
              derivable: !anchor.underivedReason,
              ...(anchor.underivedReason ? { underivedReason: anchor.underivedReason } : {}),
            }
          : null;
        for (const mode of modesFor(model)) {
          for (const overlay of overlaysFor(model, backend, route)) {
            const provider = providerFor(model, backend, overlay, route, mode);
            for (const rung of RUNGS) {
              const verdict = implementationVerdict({
                backend,
                rung,
                route,
                provider,
                stagedResidencyEngines,
                model,
                tier,
                mode,
                overlay,
                manifestById,
                candleBespokeStagedLanes,
              });
              const state = cellState({
                implementation: verdict.implementation,
                anchorPresent: anchorRow !== null,
                derivationDefined,
                anchorDerivable: anchorRow !== null && anchorRow.derivable,
              });
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
                geometryEnvelope: geometryFor(model, backend),
                strategyParameters: verdict.parameters,
                // The three inputs the state is a function of, published so a consumer — and
                // `validateMatrix` — can recompute it rather than trust it (sc-22513).
                implementation: verdict.implementation,
                anchor: anchorRow,
                derivationDefined,
                state,
                owningFamilyStory,
                owningModelStory,
                evidence: {
                  staticImplementation: verdict.source ? [{ source: verdict.source }] : [],
                  structural: verdict.structural ?? [],
                  anchor: anchorRow ? [{ source: anchorRow.source }] : [],
                },
              };
              if (anchorRow && verdict.implementation !== "missing") {
                anchorCellCounts.set(anchorRow.id, (anchorCellCounts.get(anchorRow.id) ?? 0) + 1);
              }
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

  // The anchor inventory: one row per measured anchor, with how many RESOLVED coordinates it covers
  // (every non-`Missing` cell of its lane, published or elided — the count is a property of the
  // catalog, not of the publication step). This is what replaces `calibrationRuns`: the store IS the
  // evidence join now, and a row says which coordinate the anchor was measured at rather than
  // restating a record.
  const anchorInventory = [...anchorStore.anchors.values()]
    .map((anchor) => ({
      id: anchor.id,
      modelId: anchor.modelId,
      backend: anchor.backend,
      tier: anchor.tier,
      provider: anchor.provider,
      mode: anchor.mode,
      geometry: anchor.geometry,
      source: anchor.source.path,
      current:
        loaderClosures.get(`${anchor.modelId}:${anchor.backend}`) ===
        anchor.source.loaderClosureDigest,
      currencyAttestation: currencyAttestationOf(anchor),
      cells: anchorCellCounts.get(anchor.id) ?? 0,
    }))
    .sort((left, right) => left.id.localeCompare(right.id));

  const modelSlices = Object.fromEntries(
    models.map((model) => [
      model.id,
      cells.filter((cell) => cell.modelId === model.id).map((cell) => cell.id),
    ]),
  );
  // sc-18815: censused per modality. `mlxStagedStaticCoverage` is a claim about the image entries —
  // its denominator says so — so admitting video must not inflate it.
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
    // 11 (sc-22513, epic 22505 E5): the matrix COLLAPSED onto the anchor store. `cells[].state` is a
    // pure function of `(implementation, anchor present, derivation defined)` and its vocabulary is
    // new — `Verified`, `Runtime verified` and `Implemented/unverified` are GONE, replaced by
    // `Implemented`, `Anchored` and `Anchored/underived`. `cells[].memoryCharacterization`,
    // `calibrationFingerprint`, `engagedRungs`, `plannedPipelineIdentities`,
    // `pipelineCharacterizations`, `rung4Survey` and the four record-derived evidence dimensions were
    // REMOVED, along with the root `calibrationRuns`, `rung4SurveyRows`, `manifestScopes`,
    // `memoryCharacterizationStates` and `generatedFrom.inferenceRevision`. `cells[]` gained
    // `implementation`, `anchor` and `derivationDefined`; the document gained `anchors`. A
    // version-10 reader that counted `Verified` cells now reads zero of a state nothing emits, which
    // is exactly why this is a new version rather than an additive one.
    //
    // 10 (SC-18783): LTX-2.5 plan/evidence bindings named transformer and decoder identities.
    // 9 (sc-21715): `summary.calibrationRunsByStatus` partitioned the bundle.
    // 8 (sc-18815): the model universe became modality-aware.
    // 7 (sc-18099): the document stopped publishing the whole catalog cross-product.
    // 5 (SC-16060) / 4 (SC-15969) / 3 (sc-16268) / 2 (SC-15812): see the git history.
    schemaVersion: 11,
    generatedFrom: {
      sceneWorksRevision,
      sources: Object.fromEntries(
        sourceEntries.map(([name, source]) => [
          name,
          { path: source, sha256: sha256(revisionBodies[name]) },
        ]),
      ),
    },
    // The one claim a cell makes, named in the artifact rather than left to a consumer to infer.
    claims: {
      state: {
        asserts:
          "whether the rung's CODE exists on this coordinate, and whether a measured anchor plus a " +
          "wired derivation can price its peak",
        geometrySensitive: false,
        binding:
          "scripts/generate-memory-matrix.mjs#cellState — a pure function of " +
          "(implementation, anchor present, derivation defined). No record, plan row, geometry, " +
          "campaign or currency digest may enter it (sc-22513, epic 22505 E5)",
      },
    },
    conformanceStates: [
      {
        state: "Anchored",
        definition:
          "Implemented, a measured anchor covers this (model, tier, backend lane), and the lane's " +
          "analytic derivation is wired — so an unmeasured request geometry is priced from the anchor.",
      },
      {
        state: "Anchored/underived",
        definition:
          "Implemented and anchored, but no derivation is wired for this lane, so the anchor bounds " +
          "nothing beyond its own measured point.",
      },
      {
        state: "Implemented",
        definition:
          "The code path exists and is statically evidenced; no measured anchor covers it, so its " +
          "peak is priced by the lane's analytic floor.",
      },
      {
        state: "Structurally N/A",
        definition: "The rung cannot apply to this architecture; there is nothing to measure.",
      },
      { state: "Missing", definition: "No implementation of this rung on this route." },
    ],
    evidenceDimensions: ["staticImplementation", "structural", "anchor"],
    summary: {
      catalogEntries: models.length,
      catalogEntriesByModality: Object.fromEntries(
        [...MATRIX_MODALITIES].map((modality) => [
          modality,
          models.filter((model) => model.modality === modality).length,
        ]),
      ),
      // The number of coordinates the CATALOG resolved to; `publishedCells` and `elidedCells`
      // partition it.
      cells: cells.length,
      publishedCells: cells.length,
      elidedCells: 0,
      elidedByState: {},
      publicationPredicate: PUBLICATION_PREDICATE,
      mlxStagedStaticCoverage: mlxStagedModels.size,
      // sc-22512: derived from the catalog's own enumeration rather than from a pinned population,
      // so the denominator tracks whatever the catalog ships.
      mlxStagedStaticCoverageDenominator: models.filter((model) => model.modality === "image").length,
      videoMlxStagedStaticCoverage: mlxStagedVideoModels.size,
      videoMlxStagedStaticCoverageDenominator: models.filter((model) => model.modality === "video").length,
      // sc-18815: entries in the universe that the routing catalog routes nowhere, and therefore
      // resolve to zero coordinates.
      unroutedEntries: [...UNROUTED_CATALOG_ENTRIES]
        .filter(([id]) => models.some((model) => model.id === id))
        .map(([id, entry]) => ({ id, ...entry }))
        .sort((left, right) => left.id.localeCompare(right.id)),
      // sc-18663 / sc-22513: entries deliberately subtracted from the universe, named so an absent
      // entry is never confused with an unrouted one.
      outOfMatrixEntries: [...OUT_OF_MATRIX_CATALOG_ENTRIES]
        .map(([id, entry]) => ({ id, ...entry }))
        .sort((left, right) => left.id.localeCompare(right.id)),
      // sc-22513: the anchor census. `anchors` counts the store's measured rows, `analyticOnlyCells`
      // the rows it explicitly classifies as underivable from retained evidence, and
      // `anchoredCells`/`staleAnchors` say how far the store reaches and how much of it the loader
      // closures no longer vouch for. `staleAnchors` is a REPORT: it moves no state.
      anchors: anchorInventory.length,
      analyticOnlyCells: anchorStore.analyticOnly,
      anchoredCells: cells.filter((cell) => cell.anchor !== null && cell.state !== "Missing").length,
      staleAnchors: anchorInventory.filter((anchor) => !anchor.current).length,
      // sc-22667: how many of the CURRENT anchors are current by attestation rather than by
      // measurement at the pin's own closure. A report beside `staleAnchors`, moving nothing.
      attestedAnchors: anchorInventory.filter(
        (anchor) => anchor.current && anchor.currencyAttestation !== null,
      ).length,
      fullModels: 0,
    },
    models,
    anchors: anchorInventory,
    coverage: [],
    cells,
    modelSlices,
  };
  // Validation runs against the FULL resolved cross-product, before anything is elided, so every
  // guard keeps exactly the reach it had when the artifact was the cross-product.
  validateMatrix(matrix, expectedIds, backendTierOverrides, cellInventoryExpectations);

  const published = cells.filter((cell) => isPublishableCell(cell));
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
  matrix.cells = published;
  matrix.modelSlices = Object.fromEntries(
    models.map((model) => [
      model.id,
      published.filter((cell) => cell.modelId === model.id).map((cell) => cell.id),
    ]),
  );
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
    ...(matrix.summary.outOfMatrixEntries.length
      ? [
          `- Out-of-matrix entries (subtracted from the universe): ${matrix.summary.outOfMatrixEntries
            .map((entry) => `\`${entry.id}\` — ${entry.reason} (sc-${entry.epic})`)
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
    `- Measured anchors: ${matrix.summary.anchors} (covering ${matrix.summary.anchoredCells} coordinates; ${matrix.summary.staleAnchors} stale; ${matrix.summary.attestedAnchors} current by attestation)`,
    `- Coordinates the store classifies analytic-only: ${matrix.summary.analyticOnlyCells}`,
    "",
    `sc-22513 (epic 22505, E5): a cell's \`state\` is a PURE FUNCTION of three facts published on the cell itself — \`implementation\` (does the code implement this rung on this route), \`anchor\` (does the store hold a measured anchor for this model x tier x backend lane) and \`derivationDefined\` (is the analytic derivation wired for this lane). Nothing else may enter it: no calibration record, no plan row, no measured geometry, no campaign, no currency digest. The per-geometry \`memoryCharacterization\` claim, the \`Verified\`/\`Runtime verified\` promotion and the per-record calibration join are GONE; the historical corpora they read are retained as validation data for the derivation, never as gates.`,
    "",
    `An anchor's CURRENCY (\`anchor.current\`, from \`config/anchor-loader-closures.json\`) is reported beside the state and deliberately does not move it — a staled loader closure means the anchor needs re-extraction, not that the rung stopped existing (sc-22511).`,
    "",
    "sc-22667: a current anchor also states HOW it is current. `anchor.currencyAttestation` is `null` when its key was derived at the record's own measurement revision; otherwise it is the reviewed attestation from `config/anchor-currency-attestations.json` — the closure diff from the measurement revision to the attested one was read file by file and is accounting-only, or a re-measure on the same hardware witnessed the behaviour unchanged (`class`, `why`, `witness`). An attestation is bounded to the one revision it names: the next pin bump that moves the loader closure past it stales the anchor again.",
    "",
    `sc-18099: \`cells\` is a SUBSET. ${matrix.summary.publicationPredicate} The counts on this page, \`summary\`, and the per-(entry, backend, rung) \`coverage\` census in the JSON artifact are all derived from every resolved coordinate, published or not, and \`models[].axes\` publishes the axes those coordinates span so an unimplemented lane stays distinguishable from an absent one.`,
    "",
    "One row per (catalog entry, backend): ownership is backend-scoped, so a single row per entry could only name one backend's stories (SC-15812).",
    "",
    "sc-18815: the `Modality` column exists because the universe is no longer one modality. Video entries carry no per-entry ownership story — epic 18803 does not slice video that way, so `Model story` is `—` rather than a story id that could not close the cell.",
    "",
    "sc-22513: an `Anchored` / `Anchored/underived` rollup carries `(stale)` when EVERY anchor backing that (entry, backend) is non-current. It is a currency REPORT, not a state — the lane still serves its measured numbers behind the widened margin — but without it a lane whose evidence has all staled reads identically to one measured at the live loader closure. A lane with even one current anchor is unmarked.",
    "",
    "| Catalog entry | Modality | Backend | Route | Family story | Model story | Staged residency |",
    "| --- | --- | --- | --- | --- | ---: | --- |",
  ];
  for (const model of matrix.models) {
    for (const backend of model.backends) {
      // sc-18099: read the census, not `cells`. This column is a claim about the whole lane, and
      // `cells` is a subset — scanning it would silently under-report every lane whose staged
      // coordinates were elided, which is the coverage regression the slim must not cause.
      const row = matrix.coverage.find(
        (candidate) =>
          candidate.modelId === model.id &&
          candidate.backend === backend &&
          candidate.rung === "staged_residency",
      );
      if (!row) throw new Error(`${model.id}:${backend}: no staged_residency coverage row`);
      const staged = row.implemented > 0;
      const stagedState = row.states.Anchored
        ? "Anchored"
        : row.states["Anchored/underived"]
          ? "Anchored/underived"
          : "Implemented";
      // sc-20799: a lane whose EVERY staged coordinate is structurally exempt is not `Missing` —
      // "nobody has written it yet" and "the architecture has no separable component to stage" are
      // different claims, and conflating them is how a resident-only-by-design lane reads as
      // undelivered work. A mixed lane (some exempt, some genuinely absent) still reads `Missing`.
      const structuralOnly =
        !staged &&
        row.coordinates > 0 &&
        (row.states["Structurally N/A"] ?? 0) === row.coordinates;
      // sc-22513: `Anchored` in this rollup is a claim that a MEASURED anchor prices the lane, and
      // 7 of the 10 shipped anchors are non-current. Printing the state with no currency signal let
      // a lane whose every backing anchor has staled read exactly like one measured at the live
      // closure. The marker is a REPORT, like `anchor.current` on the cell — it does not change the
      // state, and a lane with even one current anchor is not marked.
      const laneAnchors = matrix.anchors.filter(
        (anchor) => anchor.modelId === model.id && anchor.backend === backend,
      );
      const allAnchorsStale = laneAnchors.length > 0 && laneAnchors.every((anchor) => !anchor.current);
      const stagedColumn = staged
        ? `${stagedState}${stagedState.startsWith("Anchored") && allAnchorsStale ? " (stale)" : ""}`
        : structuralOnly
          ? "Structurally N/A"
          : "Missing";
      lines.push(
        `| \`${model.id}\` | ${model.modality} | ${backend} | \`${model.resolvedRoutes[backend]}\` (${model.routeKind}) | SC-${model.owningFamilyStories[backend]} | ${model.owningModelStories[backend] === null ? "—" : `SC-${model.owningModelStories[backend]}`} | ${stagedColumn} |`,
      );
    }
  }
  lines.push(
    "",
    `Per-model consumers read \`modelSlices\` in the JSON artifact for an entry's PUBLISHED cells — a subset, and ${Object.values(matrix.modelSlices).filter((slice) => slice.length === 0).length} of ${matrix.models.length} entries publish none at all. An empty slice means this entry implements no rung the matrix can see; it does NOT mean the entry has no lanes. For "which lanes exist" read \`models[].axes\`, and for "how much of a lane is implemented" read \`coverage\`.`,
    "",
    "## Measured anchors (sc-22507, epic 22505)",
    "",
    "Source: `config/memory-anchors.json`, extracted from the retained calibration corpora by `scripts/extract-memory-anchors.mjs`. One anchor per `(model, tier, backend lane)` carries the measured per-phase decomposition of a single retained render; every other geometry is derived analytically from it. A `(model, tier, lane)` the corpus cannot anchor is classified `analyticOnly` in the store rather than left silent.",
    "",
    "| Anchor | Model | Backend | Tier | Measured geometry | Current | Coordinates |",
    "| --- | --- | --- | --- | --- | --- | ---: |",
  );
  for (const anchor of matrix.anchors) {
    const geometry = `${anchor.geometry.width}x${anchor.geometry.height}${
      anchor.geometry.frames > 1 ? `x${anchor.geometry.frames}f` : ""
    }`;
    const attested = anchor.currencyAttestation;
    const current = !anchor.current
      ? "no — re-extract"
      : attested
        ? `yes — attested ${attested.class} ${attested.measuredRevision.slice(0, 8)}→${attested.attestedRevision.slice(0, 8)} (${attested.story})`
        : "yes";
    lines.push(
      `| \`${anchor.id}\` | \`${anchor.modelId}\` | ${anchor.backend} | ${anchor.tier} | ${geometry} | ${current} | ${anchor.cells} |`,
    );
  }
  lines.push("");
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
