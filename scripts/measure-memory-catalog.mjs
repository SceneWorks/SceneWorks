#!/usr/bin/env node
// Measure every memory anchor the plan declares for ONE backend on THIS host, committing each
// measurement as it lands so a crash or a cancel loses at most the anchor in flight.
//
// The harness (memory-calibration-harness.mjs, sc-22514) captures exactly one
// `<model>:<tier>:<backend>` anchor per invocation and refuses to run from a dirty checkout. That is
// the invariant this orchestrator keeps: each anchor is captured on a clean HEAD, then the evidence
// is ingested, the anchor store re-derived, its currency stamped, the matrix regenerated, and ALL of
// that committed before the next capture starts. Nothing here can express a second measurement of a
// cell — it only walks the keys the plan already declares.
//
// Per anchor:  capture → check → ingest → PACKAGED_MEMORY_ANCHOR_SOURCES → extract → stamp →
//              matrix → commit.   A failed capture is logged and the loop moves on; a failed
//              post-step is rolled back so the tree is clean again for the next anchor.
//
//   node scripts/measure-memory-catalog.mjs --backend mlx \
//     --adapter target/release/memory-mlx-adapter --inference-repo ../inference \
//     --work-dir /abs/OUTSIDE/the/repo/calib --campaign sc-NNNN [--model sdxl ...] [--anchors a,b]
//     [--skip-current]
//     [--dry-run] [--no-commit] [--hf-cache DIR ...]   (--hf-cache is repeatable)
//
// `--no-commit` captures and checks each anchor and stops there (status `captured`, raw bundle in
// <work-dir>/captures): the harness refuses complete evidence from a dirty checkout, so the first
// anchor's ingest would leave every later anchor in the same run uncapturable (sc-22724). Ingest a
// retained bundle by hand with the harness, or run without the flag to land it.
import process from "node:process";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { readFile, writeFile, mkdir, cp, rm, realpath, stat, readdir } from "node:fs/promises";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import { hashArtifactInventory } from "./hash-artifact-inventory.mjs";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const PLAN_PATH = "config/memory-calibration-plan.json";
export const MANIFEST_PATH = "config/manifests/builtin.models.jsonc";
export const MATRIX_PATH = "docs/generated/memory-matrix.json";
export const ADAPTER_LIB_PATH = "crates/sceneworks-memory-adapter/src/lib.rs";
export const PACKAGED_SOURCES_PATH = "crates/sceneworks-core/src/memory_anchor.rs";
export const ANCHOR_STORE_PATH = "config/memory-anchors.json";
export const ANCHOR_LOADER_CONFIG_PATH = "config/anchor-loader-closures.json";
export const PROVIDER_CLOSURE_CONFIG_PATH = "config/inference-provider-closures.json";
export const MATRIX_MD_PATH = "docs/generated/memory-matrix.md";
/** A well-formed but meaningless key the extractor accepts for a NEW anchor; `--stamp-anchors`
 *  re-derives every key at its record's own revision right after, before anything is committed. */
export const SEED_DIGEST = "0".repeat(64);
export const HARNESS = "scripts/memory-calibration-harness.mjs";

// LTX-2.5 is bound by the harness itself (`--ltx25-snapshot-root`), at the revision it hard-codes.
export const LTX25_REPOSITORY = "SceneWorks/ltx-2.5-mlx";

/**
 * The three caller-staged SDXL components, as `{ env, repo }` pairs. Declared once: every SDXL-family
 * model — and InstantID, which composes the same SDXL base — stages exactly these, and the Rust side
 * declares the same three ids in `candle.rs` `SDXL_COMPONENTS`. `sdxl_component_env_matches_the_catalog`
 * proves the two lists agree.
 */
export const SDXL_COMPONENTS = Object.freeze([
  { env: "SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_L", repo: "openai/clip-vit-large-patch14" },
  { env: "SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_BIGG", repo: "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" },
  { env: "SCENEWORKS_SDXL_COMPONENT_VAE_FP16_FIX", repo: "madebyollin/sdxl-vae-fp16-fix" },
]);

/**
 * sc-22729. `candle-gen-sdxl`'s `SDXL_ROUTES` pins each route's repository AND revision, and its
 * `path_has_snapshot` matches a staged root against that literal before `SdxlArtifactSeal::capture`
 * will seal a contract. When the pinned revision is not the one this repository ships, no root the
 * manifest can resolve can ever seal — `candle_gen_sdxl::load` errors before reading a weight. That
 * is an INFERENCE-side divergence, not adapter work.
 *
 * NOTHING here is a literal: both halves of the comparison are READ (the engine revision out of the
 * pinned inference checkout, the shipped revision out of the manifest), so the refusal disappears by
 * itself the moment the engine agrees — no edit to this file, and no stale exclusion outliving the
 * fix. The inference-side repair lives on `story/sc-22729-sdxl-route-revisions`.
 */
export const SDXL_ROUTES_PATH = "crates/media/candle-gen/candle-gen-sdxl/src/memory_strategy.rs";

/** `SDXL_ROUTES` as `id → { repository, revision }`. Throws if the table can no longer be read. */
export function parseSdxlRoutes(source) {
  const table = /pub const SDXL_ROUTES: &\[SdxlRoute\] = &\[([\s\S]*?)\n\];/.exec(source);
  if (!table) fail(`${SDXL_ROUTES_PATH} no longer declares a parsable SDXL_ROUTES table`);
  const routes = new Map();
  for (const entry of table[1].matchAll(/SdxlRoute\s*\{([\s\S]*?)\}/g)) {
    const field = (name) => new RegExp(`\\b${name}:\\s*"([^"]+)"`).exec(entry[1])?.[1];
    const [id, repository, revision] = [field("id"), field("repository"), field("revision")];
    if (!id || !repository || !revision) fail(`an SDXL_ROUTES entry declares no id/repository/revision: ${entry[0]}`);
    routes.set(id, { repository, revision });
  }
  if (routes.size === 0) fail(`${SDXL_ROUTES_PATH} declares an empty SDXL_ROUTES table`);
  return routes;
}

/**
 * The engine's route table at `inferenceRepo`, or `null` when no inference checkout is reachable.
 * `null` is NOT a refusal: with nothing to compare against, a cell classifies as it would if the
 * engine agreed, and says so.
 */
export async function readSdxlCandleRoutes(inferenceRepo = process.env.INFERENCE_REPO) {
  if (!inferenceRepo) return null;
  try {
    return parseSdxlRoutes(await readFile(path.join(inferenceRepo, SDXL_ROUTES_PATH), "utf8"));
  } catch (error) {
    if (error.code === "ENOENT" || error.code === "ENOTDIR") return null;
    throw error;
  }
}

export const SDXL_ROUTES_UNCHECKED =
  `no inference checkout (--inference-repo / $INFERENCE_REPO) supplies ${SDXL_ROUTES_PATH}, so the `
  + "engine's route revision was not compared with the shipped one; classified as if the engine agrees";

/**
 * Why the candle lane cannot route `modelId` today, or `null` when it can.
 *
 * Both revisions are derived: the engine's from `SDXL_ROUTES` at the pinned inference checkout, the
 * shipped one from the manifest download the route's own repository names.
 */
export function sdxlCandleRouteDrift(modelId, tier, routes, models) {
  const route = routes.get(modelId);
  if (!route) {
    return `candle-gen-sdxl's SDXL_ROUTES (${SDXL_ROUTES_PATH}) declares no route ${modelId}, so the `
      + "candle lane has no artifact identity to seal for it";
  }
  const shipped = tierDownload(models, modelId, route.repository, tier).revision;
  if (shipped === route.revision) return null;
  return `candle-gen-sdxl pins route ${modelId} at ${route.revision.slice(0, 8)} `
    + `(${SDXL_ROUTES_PATH} SDXL_ROUTES), but this repository ships ${shipped.slice(0, 8)}; `
    + "path_has_snapshot matches that literal, so no staged root can seal the contract and the load "
    + "fails before any weight is read. The candle lane does not route this model at this pin.";
}

/**
 * The built-in Qwen-Image-Edit-2511 Lightning distill LoRA (sc-22728). It is NOT a manifest
 * download — the worker fetches it lazily into the HF cache on first use — so its repository,
 * revision and file are pinned in the worker's own source on both lanes
 * (`crates/sceneworks-worker/src/image_jobs/qwen.rs` `QWEN_LIGHTNING_LORA_{REPO,REVISION}` and
 * `image_jobs/qwen_edit_candle.rs` `QWEN_EDIT_CANDLE_LIGHTNING_LORA_*`), and the candle engine
 * refuses any other path by exact suffix (`edit.rs` `is_exact_lightning_path`). The values below are
 * bound to those constants by a test rather than trusted, because a drift here would send the
 * capture at a LoRA the engine will reject after the load.
 */
export const QWEN_EDIT_LIGHTNING_LORA = Object.freeze({
  env: "QWEN_EDIT_LIGHTNING_LORA",
  repo: "lightx2v/Qwen-Image-Edit-2511-Lightning",
  revision: "d74eba145674fd7e31b949324e148e21e7118abd",
  file: "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
});

/**
 * The converter-written marker every SenseNova `_fast` rehost tier subdir carries (sc-22734). Both
 * engines read it by this exact name (`DISTILL_MERGED_MARKER` in `mlx-gen-sensenova` and
 * `candle-gen-sensenova`), and both WITHHOLD the production calibration identity when it is absent —
 * so a `_fast` tier root without it is not a capturable cell, it is a cell whose capture would fail
 * on an identity mismatch after the load. Declared as `requiredTierFiles` on the three `_fast`
 * family rows so `classifyAnchor` refuses it by name.
 */
export const SENSENOVA_DISTILL_MERGED_MARKER = "distill_merged.json";

/**
 * The shared Mage-Flow text-encoder + VAE rehost (sc-22733). Declared once and referenced by all six
 * Mage family rows: it is the SAME repository and the SAME revision for every variant and both
 * lanes, and both adapters read it through one `SCENEWORKS_MAGE_FLOW_COMPONENTS_*` family. The
 * revision is NOT pinned here — it is read from the manifest's own `coRequisite` download rows, so
 * a components re-host lands by editing the manifest alone.
 */
export const MAGE_COMPONENTS = Object.freeze({
  env: "MAGE_FLOW_COMPONENTS",
  repo: "SceneWorks/Mage-Flow-Components-mlx",
});

/**
 * The two component ids both Mage engines advertise, in descriptor order. Bound to the adapter's own
 * `MAGE_COMPONENT_TEXT_ENCODER` / `MAGE_COMPONENT_VAE` constants by a test, so the directories this
 * script probes are the ones the adapters actually stage.
 */
export const MAGE_COMPONENT_IDS = Object.freeze(["text_encoder", "vae"]);

/**
 * One row per provider arm an adapter implements, mirroring `match provider` in
 * crates/sceneworks-memory-adapter/src/bin/{mlx,candle}.rs and the env families the runbook lists
 * under "Adapter environment". `physical` marks the one arm that emits a provider `sourceCapture`
 * (the Qwen MLX source capture, mlx.rs `qwen_source_capture`): the harness REQUIRES a sourceCapture
 * whenever `--raw-log-dir` is given, so the raw-log pair and `SCENEWORKS_MEMORY_CAPTURE_DIR` must be
 * passed for that arm and for no other.
 *
 * Rows are keyed by PROVIDER by default; sc-22729 adds MODEL-keyed rows (which must declare their
 * `provider`) for the case where several catalog models ride one engine id. See `familyFor`.
 *
 * passed for that arm and for no other. `physical` is NOT inherited by a sibling family: the harness
 * scopes the receipt requirement to `modelId === "qwen_image"` alone
 * (`requiresPhysicalMlxProvenanceForCurrency`, memory-calibration-harness.mjs), and a test below
 * binds this table to that predicate.
 *
 * `sideArtifact` is a second root a family member needs that the MANIFEST does not ship — today only
 * the Qwen edit Lightning distill LoRA, which the worker fetches lazily at a pinned revision. It is
 * keyed by model id because it belongs to one member of a shared-provider family, not to the family.
 *
 * Rows are keyed by PROVIDER by default; sc-22734 adds MODEL-keyed rows (which must declare their
 * `provider`) for the case where several catalog models ride one engine id but ship their own
 * independently pinned artifacts. See `familyFor`.
 */
export const PROVIDER_FAMILIES = Object.freeze({
  qwen_image: { env: "QWEN_IMAGE", repo: "SceneWorks/qwen-image-mlx", arms: ["mlx", "candle"], physical: true },
  // `z_image_edit` anchors ride this family too (sc-22724): the catalog id is an alias for the
  // Turbo provider driven in `edit_image` mode (worker engines.rs `z_image_edit → z_image_turbo`),
  // and its manifest entry ships the same Turbo tiers, which `tierDownload` resolves by model id.
  // Both Qwen edit catalog ids (`qwen_image_edit_2511` and `..._lightning`) plan this ONE engine
  // provider (worker `qwen.rs` `qwen_edit_engine_id`, `qwen_edit_candle.rs` `QWEN_EDIT_PROVIDER_ID`)
  // and ship the SAME per-tier rehost, which `tierDownload` resolves per model id. The Lightning id
  // additionally loads the pinned distill LoRA, declared as its `sideArtifact` below.
  qwen_image_edit: {
    env: "QWEN_IMAGE_EDIT",
    repo: "SceneWorks/qwen-image-edit-2511-mlx",
    arms: ["mlx", "candle"],
    sideArtifact: { qwen_image_edit_2511_lightning: QWEN_EDIT_LIGHTNING_LORA },
  },
  z_image_turbo: { env: "Z_IMAGE", repo: "SceneWorks/z-image-turbo-mlx", arms: ["mlx", "candle"] },
  // The undistilled base is a distinct engine provider (`z_image`) with its own artifact family
  // (sc-22724). Never the Turbo env: a base plan satisfied by Turbo weights re-labels Turbo's peaks.
  z_image: { env: "Z_IMAGE_BASE", repo: "SceneWorks/z-image-mlx", arms: ["mlx", "candle"] },
  krea_2_turbo: { env: "KREA", repo: "SceneWorks/krea-2-turbo-mlx", arms: ["mlx", "candle"] },
  // sc-22729. The SDXL FAMILY: five catalog models the worker routes onto ONE engine id (`sdxl`)
  // on both lanes, each with its own independently pinned tiered rehost. These rows are keyed by
  // MODEL id rather than provider id — `classifyAnchor` prefers a model-keyed family — because the
  // engine id is not an artifact identity: `candle-gen-sdxl` seals a per-route repository and mints
  // a per-route calibration fingerprint, so a `realvisxl` anchor bound to the base-SDXL env family
  // would re-label base SDXL's peaks as the finetune's.
  //
  // `components` are the three caller-staged SDXL corequisites (`tokenizer_clip_l`,
  // `tokenizer_clip_bigg`, `vae_fp16_fix`). `candle-gen-sdxl`'s `validate_shared_component_revisions`
  // REQUIRES all three at exact upstream revisions, so a candle capture stages the same corequisite
  // snapshots the worker's `attach_required_components` stages. The MLX turnkey is self-contained
  // and ignores them, so they are bound on both lanes and simply unused on one.
  //
  // `sdxlRoute` marks the members `candle-gen-sdxl` seals through `SDXL_ROUTES`. It is carried by
  // ALL FIVE, not only the two that disagree today: the check is over the engine's declaration, so
  // a future revision drift on any member is caught the same way rather than needing a new entry.
  sdxl: { provider: "sdxl", env: "SDXL", repo: "SceneWorks/sdxl-base-mlx", arms: ["mlx", "candle"], components: SDXL_COMPONENTS, sdxlRoute: true },
  realvisxl: { provider: "sdxl", env: "REALVISXL", repo: "SceneWorks/realvisxl-mlx", arms: ["mlx", "candle"], components: SDXL_COMPONENTS, sdxlRoute: true },
  realvisxl_lightning: {
    provider: "sdxl", env: "REALVISXL_LIGHTNING", repo: "SceneWorks/realvisxl-lightning-mlx",
    arms: ["mlx", "candle"], components: SDXL_COMPONENTS, sdxlRoute: true,
  },
  // The candle lane routes all five `sdxl` members (`routing/candle.rs` `is_sdxl_family_candle_model`
  // / `SDXL_CONTROL_MODELS`), so all five are DECLARED on both lanes. Whether a member's candle cell
  // is capturable TODAY is a derived fact, not a table entry: `sdxlRoute` asks `sdxlCandleRouteDrift`
  // to compare the engine's own `SDXL_ROUTES` revision with the one the manifest ships. Only the two
  // Illustrious routes disagree at inference c6d6a4db, and only for as long as they disagree.
  illustrious_xl_v1: {
    provider: "sdxl", env: "ILLUSTRIOUS_XL_V1", repo: "SceneWorks/illustrious-xl-v1-mlx",
    arms: ["mlx", "candle"], components: SDXL_COMPONENTS, sdxlRoute: true,
  },
  illustrious_xl_v2: {
    provider: "sdxl", env: "ILLUSTRIOUS_XL_V2", repo: "SceneWorks/illustrious-xl-v2-mlx",
    arms: ["mlx", "candle"], components: SDXL_COMPONENTS, sdxlRoute: true,
  },
  // The InstantID backbone IS the plain RealVisXL rehost (`image_jobs/instantid.rs`
  // `INSTANTID_SDXL_REPO`), bound through its own env family so an InstantID plan can never be
  // satisfied by a plain `realvisxl` root and vice versa. `stagedEnv` is the identity stack: the
  // worker fetches it on first use from a pinned repo rather than declaring it as a manifest
  // download, so there is nothing for the harness to resolve — the operator stages it and the
  // capture binds the staged copy through the same env seams the worker reads.
  instantid_realvisxl: {
    provider: "instantid", env: "INSTANTID_REALVISXL", repo: "SceneWorks/realvisxl-mlx", arms: ["mlx", "candle"],
    components: SDXL_COMPONENTS,
    stagedEnv: ["SCENEWORKS_INSTANTID_WEIGHTS", "SCENEWORKS_INSTANTID_CONTROLNET"],
  },
  flux2_dev: { env: "FLUX2", repo: "SceneWorks/flux2-dev-mlx", arms: ["mlx", "candle"] },
  // sc-22727. TWO catalog models ride this ONE engine provider id (worker engines.rs:
  // `flux2_klein_9b_kv` declares `engine_id: flux2_klein_9b`), and they load DIFFERENT artifacts.
  // On MLX the engine tells them apart by the snapshot path AND by `LoadSpec::resolved_route`
  // (`KleinArtifactInventory::validate_resolved_route`, mlx-gen-flux2/src/artifact_inventory.rs);
  // on Candle ONLY by the snapshot path — `candle-gen-flux2` never reads `resolved_route`. Either
  // way the artifact is the discriminator, so the family carries a per-modelId override: a KV plan
  // resolved through the base rehost's env would re-label the base checkpoint's peaks as the KV
  // variant's.
  flux2_klein_9b: {
    env: "FLUX2_KLEIN", repo: "SceneWorks/flux2-klein-9b-mlx", arms: ["mlx", "candle"],
    variants: {
      flux2_klein_9b_kv: { env: "FLUX2_KLEIN_KV", repo: "SceneWorks/flux2-klein-9b-kv-mlx" },
    },
  },
  // The FLUX.1 family (sc-22726). `flux_dev`/`flux_schnell` are the two base text-to-image
  // providers; `pulid_flux` is the PuLID-FLUX character route, which loads the SAME
  // `SceneWorks/flux1-dev-mlx` backbone (worker image_jobs/pulid.rs `PULID_FLUX_REPO` and
  // pulid_candle.rs `PULID_CANDLE_FLUX_REPO`) and therefore shares the FLUX1_DEV env family, the
  // way `z_image_edit` shares the Turbo family. Its own manifest entry ships the same three tiers,
  // which `tierDownload` resolves by model id.
  flux1_dev: { env: "FLUX1_DEV", repo: "SceneWorks/flux1-dev-mlx", arms: ["mlx", "candle"] },
  flux1_schnell: { env: "FLUX1_SCHNELL", repo: "SceneWorks/flux1-schnell-mlx", arms: ["mlx", "candle"] },
  pulid_flux: {
    env: "FLUX1_DEV", repo: "SceneWorks/flux1-dev-mlx", arms: ["mlx", "candle"],
    // The identity stack is NOT a manifest download on either lane — the worker fetches it on first
    // use — so the anchor binds the operator's pre-staged bundle instead, through the env var both
    // worker lanes read (`SCENEWORKS_PULID_WEIGHTS`), under its strict Candle reading: a directory
    // already holding all five files (pulid_candle.rs `ensure_pulid_candle_weights`; the MLX lane
    // treats it as the directory to fill and outranks it with the `PULID_*` preset — see
    // `PULID_IDENTITY_BUNDLE_ENV` in the adapter's lib.rs). The list below is the adapter's
    // `PULID_IDENTITY_BUNDLE_FILES`, in the same order: the adapter checkpoint, the EVA tower, and
    // the three face models both engines read out of `face_dir` by name. The test parses lib.rs
    // and asserts the two lists are equal, so neither can drift alone.
    bundle: {
      env: "SCENEWORKS_PULID_WEIGHTS",
      files: [
        "pulid_flux_v0.9.1.safetensors",
        "eva02_clip_l_336.safetensors",
        "scrfd_10g.safetensors",
        "arcface_iresnet100.safetensors",
        "bisenet_parsing.safetensors",
      ],
    },
  },
  // The Mage-Flow family (sc-22733). SIX registered engine providers whose catalog ids are
  // identical to their engine ids (worker `engines.rs` MODEL_TABLE), each with its OWN tiered
  // rehost — so six rows rather than one shared family. What they DO share is the text encoder and
  // the VAE: those are bit-identical across all six variants and are hosted once in
  // `SceneWorks/Mage-Flow-Components-mlx`, which every Mage manifest entry declares as per-tier
  // `coRequisite` downloads. Both engines resolve that split through `LoadSpec::components`
  // (`mlx-gen-mage` `resolve_component_dirs`, `candle-gen-mage` `resolved_component_dirs`), so a
  // Mage anchor binds TWO roots: the variant's tier root and the components SNAPSHOT (the tier is
  // the first path element INSIDE it, which is why `components` resolves the snapshot rather than a
  // tier root).
  ...Object.fromEntries(
    [
      ["mage_flow", "MAGE_FLOW", "SceneWorks/Mage-Flow"],
      ["mage_flow_base", "MAGE_FLOW_BASE", "SceneWorks/Mage-Flow-Base"],
      ["mage_flow_turbo", "MAGE_FLOW_TURBO", "SceneWorks/Mage-Flow-Turbo"],
      ["mage_flow_edit", "MAGE_FLOW_EDIT", "SceneWorks/Mage-Flow-Edit"],
      ["mage_flow_edit_base", "MAGE_FLOW_EDIT_BASE", "SceneWorks/Mage-Flow-Edit-Base"],
      ["mage_flow_edit_turbo", "MAGE_FLOW_EDIT_TURBO", "SceneWorks/Mage-Flow-Edit-Turbo"],
    ].map(([provider, env, repo]) => [
      provider,
      { env, repo, arms: ["mlx", "candle"], components: MAGE_COMPONENTS },
    ]),
  ),
  // The SD3.5 family (sc-22730). Three DISTINCT engine providers, each with its own tiered rehost
  // and therefore its own env family — unlike `z_image_edit`/`pulid_flux`, none of them is an alias
  // for another's backbone, so serving one from another's artifact would re-label that route's
  // peaks. The catalog model id equals the engine provider id on BOTH lanes (worker engines.rs sets
  // `engine_id == sceneworks_id` for all three), so the anchor key's modelId and the plan row's
  // provider are the same token and `tierDownload` resolves the same manifest entry either way.
  // One `SceneWorks/sd3.5-<route>-mlx` repo serves both lanes at all three tiers.
  sd3_5_large: { env: "SD3_5_LARGE", repo: "SceneWorks/sd3.5-large-mlx", arms: ["mlx", "candle"] },
  sd3_5_large_turbo: { env: "SD3_5_LARGE_TURBO", repo: "SceneWorks/sd3.5-large-turbo-mlx", arms: ["mlx", "candle"] },
  sd3_5_medium: { env: "SD3_5_MEDIUM", repo: "SceneWorks/sd3.5-medium-mlx", arms: ["mlx", "candle"] },
  // The SANA family (sc-22731). Two routes, and the ONE family in this table whose two lanes load
  // DIFFERENT repositories: the MLX lane opens the per-tier SceneWorks turnkey, the Candle lane
  // opens the upstream dense diffusers snapshot at its ROOT (worker `image_jobs/base.rs`
  // `SANA_CANDLE_DIFFUSERS_REPO` / `SANA_SPRINT_CANDLE_DIFFUSERS_REPO`, resolved through
  // `huggingface_pinned_snapshot_dir`, which never descends into a tier sub-directory). That is
  // what `lanes` expresses; `tiered: false` is why the root carries no `<tier>` component.
  sana_1600m: {
    env: "SANA", repo: "SceneWorks/Sana_1600M_1024px_mlx", arms: ["mlx", "candle"],
    lanes: { candle: { env: "SANA_DENSE", repo: "Efficient-Large-Model/Sana_1600M_1024px_diffusers", tiered: false } },
  },
  sana_sprint_1600m: {
    env: "SANA_SPRINT", repo: "SceneWorks/Sana_Sprint_1.6B_1024px_mlx", arms: ["mlx", "candle"],
    lanes: { candle: { env: "SANA_SPRINT_DENSE", repo: "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", tiered: false } },
  },
  // The three Chroma1 routes (sc-22731). Separate receipt/evidence domains (SC-20788) over three
  // separate rehosts, so three env families — never one shared `CHROMA1`, which would let a Flash
  // plan be satisfied by HD weights. Both lanes open the SAME per-tier turnkey.
  chroma1_hd: { env: "CHROMA1_HD", repo: "SceneWorks/chroma1-hd-mlx", arms: ["mlx", "candle"] },
  chroma1_base: { env: "CHROMA1_BASE", repo: "SceneWorks/chroma1-base-mlx", arms: ["mlx", "candle"] },
  chroma1_flash: { env: "CHROMA1_FLASH", repo: "SceneWorks/chroma1-flash-mlx", arms: ["mlx", "candle"] },
  minimax_h3: {
    env: "MINIMAX_H3", repo: "SceneWorks/minimax-h3-mlx", arms: ["mlx"],
    upstream: { env: "MINIMAX_H3_UPSTREAM", repo: "MiniMaxAI/MiniMax-H3" },
  },
  // The harness prepares and binds the LTX-2.5 snapshot itself (`--ltx25-snapshot-root`), for
  // whichever lane the plan routes: BOTH engine ids below are served from the same public snapshot,
  // and both are declared here because the plan row's `provider` is what selects the family.
  ltx_2_5: { ltx25: true, repo: LTX25_REPOSITORY, arms: ["mlx"] },
  // The Candle arm loads LTX-2.5 under its own engine id (candle.rs `LTX25_ID`, `candle-gen-ltx`
  // `MODEL_25_ID`), so the candle plan rows name `ltx_2_5_distilled` while the anchor key — and
  // therefore the manifest download the snapshot root resolves through — stays `ltx_2_5`.
  ltx_2_5_distilled: { ltx25: true, repo: LTX25_REPOSITORY, arms: ["candle"] },
  // sc-22734. The SenseNova-U1 FAMILY: six catalog models the worker routes onto TWO engine ids on
  // both lanes — `sensenova_u1_8b` (the 50-step quality path) and `sensenova_u1_8b_fast` (the
  // 8-step distill), each carrying a base id and two infographic finetunes.
  //
  // These rows are keyed by MODEL id rather than provider id — `familyFor` prefers a model-keyed
  // row that declares the plan's provider — because the engine id is NOT the artifact identity
  // here. The six models ship six INDEPENDENTLY PINNED tiered rehosts (six repositories, six
  // revisions), and both engines mint a per-ROUTE calibration fingerprint, so an infographic anchor
  // bound to the base SenseNova env family would load base weights and re-label base SenseNova's
  // peaks as the finetune's. A provider-keyed table cannot express that: it has one repo per engine
  // id, and `tierDownload` would be asked for a base-repo download the finetune's manifest entry
  // does not ship.
  sensenova_u1_8b: {
    provider: "sensenova_u1_8b", env: "SENSENOVA_U1_8B",
    repo: "SceneWorks/sensenova-u1-8b-mlx", arms: ["mlx", "candle"],
  },
  sensenova_u1_8b_infographic_v2: {
    provider: "sensenova_u1_8b", env: "SENSENOVA_U1_8B_INFOGRAPHIC_V2",
    repo: "SceneWorks/sensenova-u1-8b-infographic-v2-mlx", arms: ["mlx", "candle"],
  },
  sensenova_u1_8b_infographic_v3: {
    provider: "sensenova_u1_8b", env: "SENSENOVA_U1_8B_INFOGRAPHIC_V3",
    repo: "SceneWorks/sensenova-u1-8b-infographic-v3-mlx", arms: ["mlx", "candle"],
  },
  sensenova_u1_8b_fast: {
    requiredTierFiles: [SENSENOVA_DISTILL_MERGED_MARKER],
    provider: "sensenova_u1_8b_fast", env: "SENSENOVA_U1_8B_FAST",
    repo: "SceneWorks/sensenova-u1-8b-fast-mlx", arms: ["mlx", "candle"],
  },
  sensenova_u1_8b_infographic_v2_fast: {
    requiredTierFiles: [SENSENOVA_DISTILL_MERGED_MARKER],
    provider: "sensenova_u1_8b_fast", env: "SENSENOVA_U1_8B_INFOGRAPHIC_V2_FAST",
    repo: "SceneWorks/sensenova-u1-8b-infographic-v2-fast-mlx", arms: ["mlx", "candle"],
  },
  sensenova_u1_8b_infographic_v3_fast: {
    requiredTierFiles: [SENSENOVA_DISTILL_MERGED_MARKER],
    provider: "sensenova_u1_8b_fast", env: "SENSENOVA_U1_8B_INFOGRAPHIC_V3_FAST",
    repo: "SceneWorks/sensenova-u1-8b-infographic-v3-fast-mlx", arms: ["mlx", "candle"],
  },
});

/**
 * The family row that serves one anchor.
 *
 * The default key is the PROVIDER, so a catalog alias rides its engine's row (`z_image_edit` on
 * `z_image_turbo`) and a lane-specific engine id keeps selecting the family (LTX-2.5's two ids).
 * sc-22729 and sc-22734 add the inverse case: several catalog models on ONE engine id, each with its own
 * artifact family. A MODEL-keyed row wins for those — but ONLY when it declares the provider it
 * belongs to and that provider is the one the plan named, so a model-keyed row can never capture
 * an anchor that some other engine serves.
 */
export function familyFor(modelId, provider, families = PROVIDER_FAMILIES) {
  // Model-keyed resolution — and the provider-keyed `variants` override (sc-22727's FLUX.2 klein
  // pair) — both live in `providerFamily`, so this is a single delegation.
  return providerFamily(provider, modelId, families);
}

export function fail(message) {
  throw new Error(message);
}

/**
 * The artifact family one anchor binds: the provider's row, with any per-modelId override applied.
 * A provider that serves several catalog models from ONE registry id (sc-22727's two klein models)
 * declares the divergent members under `variants`; everything else is the row itself.
 */
export function providerFamily(provider, modelId, families = PROVIDER_FAMILIES) {
  // sc-22729/sc-22734's MODEL-keyed rows win first, but ONLY when the row declares the provider it belongs
  // to and that provider is the one the plan named — so a model-keyed row can never capture an
  // anchor some other engine serves.
  const scoped = families[modelId];
  if (scoped?.provider !== undefined && scoped.provider === provider) return scoped;
  const family = families[provider];
  if (!family) return undefined;
  const variant = family.variants?.[modelId];
  return variant ? { ...family, ...variant, variants: undefined } : family;
}

export function anchorParts(key) {
  const match = /^([a-z][a-z0-9_]*):(q4|q8|bf16):(mlx|candle)$/.exec(key);
  if (!match) fail(`not an anchor key: ${key}`);
  return { modelId: match[1], tier: match[2], backend: match[3] };
}

/** `--adapter` is a binary path, or a JSON array command (the way the harness fixtures are driven). */
export function providerCommand(adapter) {
  if (adapter.trimStart().startsWith("[")) {
    const command = JSON.parse(adapter);
    if (!Array.isArray(command) || command.length === 0 || command.some((part) => typeof part !== "string")) {
      fail("--adapter JSON form must be a non-empty array of strings");
    }
    return command;
  }
  return [path.resolve(adapter)];
}

export function anchorSlug(key) {
  return key.replaceAll(":", "-").replaceAll("_", "-");
}

export function parseArgs(argv) {
  const args = {
    backend: null, adapter: null, inferenceRepo: null, workDir: null, campaign: null,
    anchors: null, models: [], skipCurrent: false, dryRun: false, commit: true, hfCache: [], list: false,
  };
  const value = (flag, index) => {
    const selected = argv[index + 1];
    if (selected === undefined || selected.startsWith("--")) fail(`${flag} requires one value`);
    return selected;
  };
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    switch (flag) {
      case "--backend": args.backend = value(flag, index); index += 1; break;
      case "--adapter": args.adapter = value(flag, index); index += 1; break;
      case "--inference-repo": args.inferenceRepo = value(flag, index); index += 1; break;
      case "--work-dir": args.workDir = value(flag, index); index += 1; break;
      case "--campaign": args.campaign = value(flag, index); index += 1; break;
      case "--hf-cache": args.hfCache.push(value(flag, index)); index += 1; break;
      case "--anchors": args.anchors = value(flag, index).split(",").filter(Boolean); index += 1; break;
      case "--model": args.models.push(value(flag, index)); index += 1; break;
      case "--skip-current": args.skipCurrent = true; break;
      case "--dry-run": args.dryRun = true; break;
      case "--no-commit": args.commit = false; break;
      case "--list": args.list = true; break;
      default: fail(`unknown argument ${flag}`);
    }
  }
  if (!["mlx", "candle"].includes(args.backend)) fail("--backend must be mlx or candle");
  if (!args.list) {
    if (!args.workDir) fail("--work-dir is required (a directory OUTSIDE the checkout)");
    if (!args.inferenceRepo) fail("--inference-repo is required");
    if (!args.dryRun && !args.adapter) fail("--adapter is required for a real capture");
  }
  if (args.campaign && !/^[A-Za-z0-9._-]+$/.test(args.campaign)) {
    fail("--campaign must be one path segment, e.g. sc-12345");
  }
  return args;
}

export async function readPlan(root = ROOT) {
  const plan = JSON.parse(await readFile(path.join(root, PLAN_PATH), "utf8"));
  if (!plan.anchors || typeof plan.anchors !== "object") fail(`${PLAN_PATH} carries no anchors object`);
  return plan;
}

export async function readManifestModels(root = ROOT) {
  const body = await readFile(path.join(root, MANIFEST_PATH), "utf8");
  return JSON.parse(stripJsoncComments(body)).models;
}

export async function readDeclaredLanes(root = ROOT) {
  const config = JSON.parse(await readFile(path.join(root, ANCHOR_LOADER_CONFIG_PATH), "utf8"));
  return new Set(Object.keys(config.models ?? {}));
}

/** `<backend>:<provider>` keys with a crate-closure declaration (runbook §7c). */
export async function readDeclaredProviders(root = ROOT) {
  const config = JSON.parse(await readFile(path.join(root, PROVIDER_CLOSURE_CONFIG_PATH), "utf8"));
  return new Set(Object.keys(config.providers ?? {}));
}

export async function readMatrixCurrency(root = ROOT) {
  let matrix;
  try {
    matrix = JSON.parse(await readFile(path.join(root, MATRIX_PATH), "utf8"));
  } catch {
    return new Map();
  }
  const current = new Map();
  for (const anchor of matrix.anchors ?? []) {
    current.set(`${anchor.modelId}:${anchor.tier}:${anchor.backend}`, anchor.current === true);
  }
  return current;
}

export async function compiledInferencePin(root = ROOT) {
  const source = await readFile(path.join(root, ADAPTER_LIB_PATH), "utf8");
  const match = /^pub const INFERENCE_PIN: &str = "([0-9a-f]{40})";/m.exec(source);
  if (!match) fail(`${ADAPTER_LIB_PATH} declares no INFERENCE_PIN`);
  return match[1];
}

/** The manifest download that ships `tier` of `repo` for `modelId` — its revision names the snapshot. */
export function tierDownload(models, modelId, repo, tier) {
  const model = models.find((entry) => entry.id === modelId);
  if (!model) fail(`manifest has no model ${modelId}`);
  const downloads = (model.downloads ?? []).filter((download) => download.repo === repo);
  const primary = downloads.find((download) => download.variant === tier && !download.coRequisite);
  if (primary) return primary;
  const any = downloads.find((download) => /^[0-9a-f]{40}$/.test(download.revision ?? ""));
  if (!any) fail(`manifest ${modelId} has no download from ${repo}`);
  return any;
}

export function hubRoots(hfCache = []) {
  // Explicit --hf-cache roots first (repeatable), then the HF env convention, then the app cache.
  const roots = [
    ...hfCache,
    process.env.HF_HUB_CACHE ?? (process.env.HF_HOME ? path.join(process.env.HF_HOME, "hub") : path.join(os.homedir(), ".cache", "huggingface", "hub")),
  ];
  if (process.platform === "darwin") {
    roots.push(path.join(
      os.homedir(), "Library", "Application Support", "SceneWorks", "data", "cache", "huggingface", "hub",
    ));
  }
  return roots;
}

export function snapshotPath(hub, repo, revision, ...rest) {
  return path.join(hub, `models--${repo.replaceAll("/", "--")}`, "snapshots", revision, ...rest);
}

async function firstExistingDirectory(candidates) {
  for (const candidate of candidates) {
    try {
      if ((await stat(candidate)).isDirectory()) return candidate;
    } catch { /* absent */ }
  }
  return null;
}

/**
 * Decide what the run can do with one plan anchor: which adapter arm serves it, which weights
 * root it loads, and why it would be skipped. Pure apart from the directory probes.
 */
export async function classifyAnchor(key, planned, { models, backend, hubs, current, captured, declaredLanes, declaredProviders, sdxlRoutes = null, families = PROVIDER_FAMILIES }) {
  const parts = anchorParts(key);
  const row = { key, ...parts, provider: planned.provider, status: "runnable", reason: null, env: {}, roots: [] };
  if (parts.backend !== backend) return { ...row, status: "other_backend", reason: `${parts.backend} lane` };
  const family = familyFor(parts.modelId, planned.provider, families);
  // No shipped family carries `harnessUnsupported` today (sc-22725 gave LTX-2.5's candle engine id
  // a real row). The status stays for the next provider whose adapter arm exists but whose
  // artifacts the harness cannot bind: it is the one refusal that is neither a missing arm nor a
  // missing declaration.
  //
  // KEPT DELIBERATELY (sc-22725 review): the `families` parameter above is a test seam and nothing
  // else — no caller passes it — and it exists so this otherwise-unreachable branch is driven by a
  // synthetic family rather than left uncovered. The alternative considered and rejected was
  // deleting the branch and the parameter together; that would make the next unbindable provider
  // report as `no_adapter_arm`, which is the wrong diagnosis and sends the reader to adapter work.
  if (family?.harnessUnsupported) return { ...row, status: "harness_unsupported", reason: family.harnessUnsupported };
  // sc-22729: the same refusal, scoped to ONE lane and DERIVED. A model whose engine cannot seal an
  // artifact identity for it on a lane is not a missing arm and not a missing declaration — the arm
  // exists and the plan declares the cell — so it reports the engine-side reason rather than
  // sending the reader to adapter work. With no inference checkout to read, there is no refusal at
  // all: the cell classifies normally and carries the note saying the comparison did not happen.
  if (family?.sdxlRoute && backend === "candle") {
    if (sdxlRoutes === null) {
      row.routeCheck = SDXL_ROUTES_UNCHECKED;
    } else {
      const drift = sdxlCandleRouteDrift(parts.modelId, parts.tier, sdxlRoutes, models);
      if (drift) return { ...row, status: "harness_unsupported", reason: drift };
    }
  }
  if (!family || !family.arms.includes(backend)) {
    return {
      ...row, status: "no_adapter_arm",
      reason: `the ${backend} adapter implements no provider arm for ${planned.provider}`,
    };
  }
  if (captured.has(key)) return { ...row, status: "already_captured", reason: captured.get(key) };
  if (declaredLanes && !declaredLanes.has(`${parts.modelId}:${backend}`)) {
    return {
      ...row, status: "lane_undeclared",
      reason: `${parts.modelId}:${backend} has no loader-closure declaration in ${ANCHOR_LOADER_CONFIG_PATH}; --stamp-anchors would refuse it (runbook §7c)`,
    };
  }
  if (declaredProviders && !declaredProviders.has(`${backend}:${planned.provider}`)) {
    return {
      ...row, status: "provider_undeclared",
      reason: `${backend}:${planned.provider} has no crate-closure declaration in ${PROVIDER_CLOSURE_CONFIG_PATH}; the record would carry no closure digest (runbook §7c)`,
    };
  }
  if (current.get(key) === true) row.current = true;

  if (family.ltx25) {
    const download = tierDownload(models, parts.modelId, family.repo, parts.tier);
    const snapshot = await firstExistingDirectory(hubs.map((hub) => snapshotPath(hub, family.repo, download.revision)));
    row.roots.push({ label: "ltx25 snapshot", path: snapshot ?? snapshotPath(hubs[0], family.repo, download.revision) });
    if (!snapshot) return { ...row, status: "weights_missing", reason: `no ${family.repo}@${download.revision.slice(0, 8)} snapshot on this host` };
    row.ltx25SnapshotRoot = snapshot;
    row.physical = false;
    return row;
  }

  // Which artifact THIS lane opens. Almost every family loads one repository on both lanes, so the
  // family row IS the artifact; `lanes[backend]` overrides it for a family whose lanes load
  // different artifacts (sc-22731: SANA's MLX turnkey vs its upstream dense Candle snapshot).
  // `tiered: false` means the load root is the snapshot ITSELF — no `<tier>` component — which is
  // what the worker resolves for such a repository and what the adapter validates against
  // (`validate_huggingface_revision_root`).
  const artifact = family.lanes?.[backend] ?? family;
  const tiered = artifact.tiered !== false;
  const download = tierDownload(models, parts.modelId, artifact.repo, parts.tier);
  const suffix = tiered ? [parts.tier] : [];
  const tierRoot = await firstExistingDirectory(hubs.map((hub) => snapshotPath(hub, artifact.repo, download.revision, ...suffix)));
  row.roots.push({ label: tiered ? "tier root" : "snapshot root", path: tierRoot ?? snapshotPath(hubs[0], artifact.repo, download.revision, ...suffix) });
  if (!tierRoot) {
    return { ...row, status: "weights_missing", reason: `no ${artifact.repo}@${download.revision.slice(0, 8)}${tiered ? `/${parts.tier}` : ""} on this host` };
  }
  row.env[`SCENEWORKS_${artifact.env}_REPOSITORY`] = artifact.repo;
  row.env[`SCENEWORKS_${artifact.env}_REVISION`] = download.revision;
  row.env[`SCENEWORKS_${artifact.env}_ROOT`] = tierRoot;
  row.tierRoot = tierRoot;
  // A converter-written file the ENGINE requires INSIDE the resolved tier root, beyond the weights
  // the manifest download ships (sc-22734 review). The SenseNova `_fast` rehosts are the live case:
  // `mlx-gen-sensenova`'s `production_calibration_identity` and `candle-gen-sensenova`'s
  // `fast_spec_is_the_premerged_turnkey` both WITHHOLD the production identity when a `_fast` tier
  // root carries no `distill_merged.json`, because without the marker the loader merges the distill
  // LoRA at load and the resident shape is not the one the anchor prices. Nine `_fast` MLX cells
  // would therefore hard-fail at capture with an identity mismatch, hours into a booked session,
  // over a fact this probe can read in a millisecond. Classified by name instead.
  const requiredTierFiles = artifact.requiredTierFiles ?? family.requiredTierFiles ?? [];
  const missingTierFiles = [];
  for (const file of requiredTierFiles) {
    try {
      if (!(await stat(path.join(tierRoot, file))).isFile()) missingTierFiles.push(file);
    } catch { missingTierFiles.push(file); }
  }
  if (missingTierFiles.length > 0) {
    return {
      ...row, status: "weights_missing",
      reason: `tier root ${tierRoot} is missing ${missingTierFiles.join(", ")}, which the engine requires before it will publish this cell's calibration identity`,
    };
  }
  if (family.bundle) {
    // A pre-staged loose-file bundle rather than an HF snapshot, so it is probed through the
    // operator env the worker itself honours. Absent or incomplete is `weights_missing` — the same
    // class as a missing tier root, and NOT "runnable" (a cell whose identity stack cannot be bound
    // is not measurable on this host, and saying otherwise sends an operator to book a capture).
    // Absolute before it is probed or handed on: the adapter canonicalizes the value from the
    // HARNESS's cwd (lib.rs `pulid_identity_bundle`), so a relative export that resolved here
    // against node's cwd would be probed in one directory and opened in another.
    const bundleRoot = process.env[family.bundle.env] ? path.resolve(process.env[family.bundle.env]) : undefined;
    row.roots.push({ label: "pulid bundle", path: bundleRoot ?? `$${family.bundle.env}` });
    if (!bundleRoot) {
      return { ...row, status: "weights_missing", reason: `${family.bundle.env} is unset; the PuLID identity bundle is not staged on this host` };
    }
    const missing = [];
    for (const file of family.bundle.files) {
      try {
        if (!(await stat(path.join(bundleRoot, file))).isFile()) missing.push(file);
      } catch { missing.push(file); }
    }
    if (missing.length > 0) {
      return { ...row, status: "weights_missing", reason: `${family.bundle.env} bundle ${bundleRoot} is missing ${missing.join(", ")}` };
    }
    row.env[family.bundle.env] = bundleRoot;
  }
  if (family.upstream) {
    const upstream = tierDownload(models, parts.modelId, family.upstream.repo, parts.tier);
    const upstreamRoot = await firstExistingDirectory(hubs.map((hub) => snapshotPath(hub, family.upstream.repo, upstream.revision)));
    row.roots.push({ label: "upstream root", path: upstreamRoot ?? snapshotPath(hubs[0], family.upstream.repo, upstream.revision) });
    if (!upstreamRoot) {
      return { ...row, status: "weights_missing", reason: `no ${family.upstream.repo}@${upstream.revision.slice(0, 8)} snapshot on this host` };
    }
    row.env[`SCENEWORKS_${family.upstream.env}_REPOSITORY`] = family.upstream.repo;
    row.env[`SCENEWORKS_${family.upstream.env}_REVISION`] = upstream.revision;
    row.env[`SCENEWORKS_${family.upstream.env}_ROOT`] = upstreamRoot;
  }
  // sc-22729: the caller-staged SDXL components. Their revisions come from the model's own
  // corequisite downloads, which are exactly the revisions `candle-gen-sdxl` validates against.
  for (const component of Array.isArray(family.components) ? family.components : []) {
    const download = tierDownload(models, parts.modelId, component.repo, parts.tier);
    const root = await firstExistingDirectory(hubs.map((hub) => snapshotPath(hub, component.repo, download.revision)));
    row.roots.push({ label: `component ${component.env}`, path: root ?? snapshotPath(hubs[0], component.repo, download.revision) });
    if (!root) {
      return { ...row, status: "weights_missing", reason: `no ${component.repo}@${download.revision.slice(0, 8)} snapshot on this host` };
    }
    row.env[component.env] = root;
  }
  // The shared component snapshot a split-layout family stages its text encoder and VAE from
  // (sc-22733, Mage-Flow). Unlike `upstream`, this root is a co-requisite the MANIFEST ships, and
  // unlike the tier root above it is the SNAPSHOT: the tier is the first path element inside it,
  // and the adapters join `<tier>/text_encoder` and `<tier>/vae` themselves. Absent is
  // `weights_missing` — a Mage load whose components are not staged cannot open at all, because the
  // variant rehost carries no `text_encoder/` or `vae/` sibling for the loader to fall back to.
  if (family.components && !Array.isArray(family.components)) {
    // The components are declared PER TIER (one co-requisite row per component per tier), so the
    // row for THIS tier is what says the tier's text encoder and VAE are shipped at all. No row is
    // `weights_missing` — the cell is planned and the arm exists, the host merely cannot bind the
    // artifact — never a thrown error that aborts the whole `--list` (sc-22733 review).
    const download = (models.find((entry) => entry.id === parts.modelId)?.downloads ?? []).find(
      (row) => row.repo === family.components.repo && row.coRequisite && row.variant === parts.tier,
    );
    if (!download) {
      row.roots.push({ label: "components snapshot", path: `$SCENEWORKS_${family.components.env}_ROOT` });
      return {
        ...row, status: "weights_missing",
        reason: `manifest ${parts.modelId} declares no ${family.components.repo} components row for tier ${parts.tier}`,
      };
    }
    const root = await firstExistingDirectory(
      hubs.map((hub) => snapshotPath(hub, family.components.repo, download.revision)),
    );
    row.roots.push({
      label: "components snapshot",
      path: root ?? snapshotPath(hubs[0], family.components.repo, download.revision),
    });
    if (!root) {
      return {
        ...row, status: "weights_missing",
        reason: `no ${family.components.repo}@${download.revision.slice(0, 8)} snapshot on this host`,
      };
    }
    // The tier's own two component directories, not just the snapshot: a partially-fetched
    // components mirror is exactly as unloadable as an absent one, and reporting the cell
    // `runnable` would send an operator to book a capture that cannot open its text encoder.
    const missing = [];
    for (const component of MAGE_COMPONENT_IDS) {
      if (!(await firstExistingDirectory([path.join(root, parts.tier, component)]))) {
        missing.push(`${parts.tier}/${component}`);
      }
    }
    if (missing.length > 0) {
      return {
        ...row, status: "weights_missing",
        reason: `${family.components.repo} snapshot ${root} is missing ${missing.join(", ")}`,
      };
    }
    row.env[`SCENEWORKS_${family.components.env}_REPOSITORY`] = family.components.repo;
    row.env[`SCENEWORKS_${family.components.env}_REVISION`] = download.revision;
    row.env[`SCENEWORKS_${family.components.env}_ROOT`] = root;
  }
  // An identity stack the worker fetches on first use rather than declaring as a manifest download
  // has nothing for the harness to resolve, so the operator stages it and names it here. An unset
  // or absent path is `weights_missing` — the host simply lacks the artifact — not a gap.
  for (const name of family.stagedEnv ?? []) {
    const staged = process.env[name];
    const root = staged ? await firstExistingDirectory([staged]) : null;
    row.roots.push({ label: `staged ${name}`, path: staged ?? `(${name} unset)` });
    if (!root) {
      return { ...row, status: "weights_missing", reason: `${name} is unset or names no directory on this host` };
    }
    row.env[name] = root;
  }
  // A member-specific artifact the manifest does not ship (the Qwen edit Lightning distill LoRA):
  // its repository and revision are pinned in the family row, so the root is the snapshot itself.
  const side = family.sideArtifact?.[parts.modelId];
  if (side) {
    const sideRoot = await firstExistingDirectory(hubs.map((hub) => snapshotPath(hub, side.repo, side.revision)));
    row.roots.push({ label: "side artifact", path: sideRoot ?? snapshotPath(hubs[0], side.repo, side.revision) });
    if (!sideRoot) {
      return { ...row, status: "weights_missing", reason: `no ${side.repo}@${side.revision.slice(0, 8)} snapshot on this host` };
    }
    row.env[`SCENEWORKS_${side.env}_REPOSITORY`] = side.repo;
    row.env[`SCENEWORKS_${side.env}_REVISION`] = side.revision;
    row.env[`SCENEWORKS_${side.env}_ROOT`] = sideRoot;
  }
  row.physical = backend === "mlx" && family.physical === true;
  return row;
}

/** Append one evidence corpus to the Rust loader's compiled-in list, idempotently. */
export function appendPackagedSource(source, relativePath) {
  if (source.includes(`"${relativePath}"`)) return source;
  const start = source.indexOf("PACKAGED_MEMORY_ANCHOR_SOURCES: &[(&str, &str)] = &[");
  if (start === -1) fail(`${PACKAGED_SOURCES_PATH} no longer declares PACKAGED_MEMORY_ANCHOR_SOURCES`);
  const end = source.indexOf("\n];", start);
  if (end === -1) fail("PACKAGED_MEMORY_ANCHOR_SOURCES is not terminated by `];`");
  const entry = [
    "    (",
    `        "${relativePath}",`,
    `        include_str!("../../../${relativePath}"),`,
    "    ),",
  ].join("\n");
  return `${source.slice(0, end)}\n${entry}${source.slice(end)}`;
}

/** Anchors already ingested under the campaign directory, keyed by anchor key. */
export async function capturedInCampaign(root, campaignDir) {
  const captured = new Map();
  let entries;
  try {
    entries = await readdir(path.join(root, campaignDir));
  } catch {
    return captured;
  }
  for (const name of entries) {
    if (!name.endsWith("-evidence.json")) continue;
    let bundle;
    try {
      bundle = JSON.parse(await readFile(path.join(root, campaignDir, name), "utf8"));
    } catch {
      continue;
    }
    for (const record of bundle.records ?? []) {
      const target = record.target ?? {};
      if (target.modelId && target.tier && record.backend) {
        captured.set(`${target.modelId}:${target.tier}:${record.backend}`, `${campaignDir}/${name}`);
      }
    }
  }
  return captured;
}

// ---------------------------------------------------------------------------------------------
// Process plumbing
// ---------------------------------------------------------------------------------------------

function run(command, args, { cwd = ROOT, env = process.env, log = null, detached = false } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, env, stdio: ["ignore", "pipe", "pipe"], detached });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; log?.write(chunk); });
    child.stderr.on("data", (chunk) => { stderr += chunk; log?.write(chunk); });
    child.on("error", reject);
    child.on("close", (code, signal) => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(Object.assign(new Error(`${[command, ...args].join(" ")} exited ${code ?? signal}\n${stderr.slice(-4000)}`), { stdout, stderr, code }));
    });
  });
}

async function git(args, cwd = ROOT) {
  return (await run("git", args, { cwd })).stdout.trim();
}

async function gitIsClean(cwd) {
  return (await git(["status", "--porcelain"], cwd)) === "";
}

async function assertOutsideRepo(dir, repo) {
  const relation = path.relative(await realpath(repo), await realpath(dir));
  if (!relation || (!relation.startsWith("..") && !path.isAbsolute(relation))) {
    fail(`${dir} is inside the checkout ${repo}; the harness refuses in-tree output`);
  }
}

class Log {
  constructor(file) { this.file = file; this.chunks = []; }
  write(chunk) { this.chunks.push(String(chunk)); }
  async flush() { await writeFile(this.file, this.chunks.join("")); }
}

function stamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

// ---------------------------------------------------------------------------------------------
// One anchor, end to end
// ---------------------------------------------------------------------------------------------

/**
 * The extractor carries each anchor's currency key forward from the store and FAILS for an anchor
 * id it has never seen — by design, so a new anchor can never borrow the pin's digest. A fresh
 * capture is exactly that case (record ids are content-derived, so the anchor id is new). Seed the
 * new id with a placeholder key so the store can be regenerated, then let `--stamp-anchors`, which
 * runs next and rewrites every key at its record's own revision, replace it before the commit.
 */
export const NEW_ANCHOR_PATTERN = /anchor (\S+) has no recorded loader-closure digest/;

export async function extractSeedingNewAnchors(exec, root, log, limit = 8) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      await exec(process.execPath, ["scripts/extract-memory-anchors.mjs"], { log });
      return;
    } catch (error) {
      const match = NEW_ANCHOR_PATTERN.exec(String(error.stderr ?? error.message ?? ""));
      if (!match || attempt >= limit) throw error;
      const storePath = path.join(root, ANCHOR_STORE_PATH);
      const store = JSON.parse(await readFile(storePath, "utf8"));
      if (store.anchors.some((anchor) => anchor.id === match[1])) throw error;
      store.anchors.push({ id: match[1], source: { loaderClosureDigest: SEED_DIGEST } });
      await writeFile(storePath, `${JSON.stringify(store, null, 2)}\n`);
      log?.write(`seeded new anchor ${match[1]} for extraction; --stamp-anchors derives its key next\n`);
    }
  }
}

/** The first line of a child's stderr that names the failure, not the Node banner after it. */
export function failureReason(error) {
  const lines = String(error.stderr ?? error.message ?? "").split("\n").map((line) => line.trim()).filter(Boolean);
  const named = lines.find((line) => /^(Error|[A-Za-z]*Error|fatal|error)\b/.test(line));
  return (named ?? lines.find((line) => !/^(at |Node\.js v)/.test(line)) ?? "unknown failure").slice(0, 300);
}

export async function measureAnchor(row, context) {
  const { args, inferencePin, campaignDir, campaignPrefix, workDir, state, root = ROOT } = context;
  const slug = anchorSlug(row.key);
  const log = new Log(path.join(workDir, "logs", `${slug}.log`));
  const captureOutput = path.join(workDir, "captures", `${slug}.json`);
  const rawLogDir = path.join(workDir, "raw", slug);
  const evidenceRelative = `${campaignDir}/${slug}-evidence.json`;
  const started = Date.now();
  const touched = [ANCHOR_STORE_PATH, MATRIX_PATH, MATRIX_MD_PATH, PACKAGED_SOURCES_PATH];
  const exec = (command, commandArgs, options = {}) => run(command, commandArgs, { cwd: root, ...options });
  const gitAt = (gitArgs) => git(gitArgs, root);
  const created = [];
  const finish = async (status, reason = null) => {
    await log.flush();
    return { key: row.key, status, reason, seconds: Math.round((Date.now() - started) / 1000), log: log.file };
  };

  const env = { ...process.env, ...row.env };
  if (row.tierRoot) {
    const inventory = await hashArtifactInventory(row.tierRoot);
    env.SCENEWORKS_MEMORY_MODEL_BYTES = String(inventory.bytes);
    env.SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256 = inventory.sha256;
  }
  if (row.physical) {
    env.SCENEWORKS_MEMORY_CAPTURE_DIR = rawLogDir;
    env.SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX = campaignPrefix;
  }

  // 1. capture — detached from the terminal's process group so a Ctrl-C here does not reach the
  //    adapter mid-command-buffer; the loop stops after the anchor in flight instead.
  const captureArgs = [
    HARNESS, "capture", "--plan", PLAN_PATH, "--anchor", row.key,
    "--provider-command", JSON.stringify(providerCommand(args.adapter)),
    "--sceneworks-repo", root, "--inference-repo", path.resolve(args.inferenceRepo),
    "--output", captureOutput,
  ];
  if (row.physical) captureArgs.push("--raw-log-dir", rawLogDir, "--source-path-prefix", campaignPrefix);
  if (row.ltx25SnapshotRoot) captureArgs.push("--ltx25-snapshot-root", row.ltx25SnapshotRoot);
  try {
    log.write(`$ node ${captureArgs.join(" ")}\n`);
    await exec(process.execPath, captureArgs, { env, log, detached: true });
  } catch (error) {
    return finish("capture_failed", failureReason(error));
  }

  // 2. check the raw bundle before touching the tree.
  const sourceRoot = row.physical ? ["--source-root", rawLogDir] : [];
  try {
    await exec(process.execPath, [HARNESS, "check", "--input", captureOutput, ...sourceRoot], { log });
  } catch (error) {
    return finish("check_failed", failureReason(error));
  }
  // A no-commit run ends here: ingesting would dirty the tree and make the harness refuse every
  // later anchor in the same run (`complete evidence cannot come from a dirty repository`).
  if (!args.commit) return finish("captured");

  // 3..7. ingest + derive + commit. Any failure rolls the tree back to HEAD so the next anchor
  //       still starts clean; the raw capture stays in the work dir for a by-hand ingest.
  try {
    await mkdir(path.join(root, campaignDir), { recursive: true });
    created.push(evidenceRelative);
    await exec(process.execPath, [
      HARNESS, "ingest", "--input", captureOutput, ...sourceRoot, "--output", evidenceRelative,
    ], { log });
    if (row.physical) {
      const receipts = path.join(rawLogDir, campaignDir);
      try {
        for (const name of await readdir(receipts)) {
          created.push(`${campaignDir}/${name}`);
          await cp(path.join(receipts, name), path.join(root, campaignDir, name), { recursive: true, force: false, errorOnExist: true });
        }
      } catch (error) {
        fail(`copy physical receipts from ${receipts}: ${error.message}`);
      }
    }
    const rust = await readFile(path.join(root, PACKAGED_SOURCES_PATH), "utf8");
    await writeFile(path.join(root, PACKAGED_SOURCES_PATH), appendPackagedSource(rust, evidenceRelative));
    await extractSeedingNewAnchors(exec, root, log);
    await exec(process.execPath, [
      "scripts/anchor-loader-closure.mjs", "--repo", path.resolve(args.inferenceRepo), "--stamp-anchors",
    ], { log });
    const stamped = JSON.parse(await readFile(path.join(root, ANCHOR_STORE_PATH), "utf8"));
    if (stamped.anchors.some((anchor) => anchor.source?.loaderClosureDigest === SEED_DIGEST)) {
      fail("a seeded placeholder currency key survived --stamp-anchors; refusing to commit it");
    }
    await exec(process.execPath, ["scripts/generate-memory-matrix.mjs"], { log });

    if (args.commit) {
      // `-f`: the harness's `<session>.log` receipt matches the blanket `*.log` ignore rule.
      await gitAt(["add", "-f", "--", ...touched, ...created]);
      const stray = await gitAt(["status", "--porcelain"]);
      const unstaged = stray.split("\n").filter((line) => line && line[1] !== " ");
      if (unstaged.length > 0) fail(`post-steps changed paths this run does not own:\n${unstaged.join("\n")}`);
      await gitAt(["commit", "--quiet", "-m",
        `chore(${args.campaign}): measure ${row.key} memory anchor\n\n` +
        `Captured by scripts/measure-memory-catalog.mjs at inference ${inferencePin}. ` +
        `Evidence: ${evidenceRelative}; anchor store, currency stamp and matrix regenerated.`]);
      state.commits.push(await gitAt(["rev-parse", "--short", "HEAD"]));
    }
    return finish("committed");
  } catch (error) {
    log.write(`\nROLLBACK: ${error.message}\n`);
    try {
      // Unstage first: `checkout --` restores from the INDEX, which already holds the staged edits.
      await exec("git", ["reset", "--quiet", "--", ...touched, ...created], { log }).catch(() => {});
      await exec("git", ["checkout", "--quiet", "--", ...touched], { log });
      for (const relative of created) await rm(path.join(root, relative), { recursive: true, force: true });
      if (!(await gitIsClean(root))) fail("tree is still dirty after rollback; stop here");
    } catch (rollbackError) {
      state.halt = `rollback failed for ${row.key}: ${rollbackError.message}`;
    }
    return finish("ingest_failed", failureReason(error));
  }
}

// ---------------------------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------------------------

export async function planRun(args, root = ROOT) {
  const plan = await readPlan(root);
  const models = await readManifestModels(root);
  const current = await readMatrixCurrency(root);
  const campaign = args.campaign ?? `catalog-${new Date().toISOString().slice(0, 10)}`;
  const campaignDir = `docs/calibration/${campaign}`;
  const captured = await capturedInCampaign(root, campaignDir);
  const declaredLanes = await readDeclaredLanes(root);
  const declaredProviders = await readDeclaredProviders(root);
  const hubs = hubRoots(args.hfCache);
  // sc-22729: the engine's own SDXL route table, read from the pinned inference checkout. `null`
  // when there is none to read — see `SDXL_ROUTES_UNCHECKED`.
  const sdxlRoutes = await readSdxlCandleRoutes(args.inferenceRepo ?? process.env.INFERENCE_REPO);
  const keys = Object.keys(plan.anchors).sort();
  if (args.anchors) {
    for (const key of args.anchors) if (!plan.anchors[key]) fail(`--anchors names ${key}, which the plan does not declare`);
  }
  for (const model of args.models ?? []) {
    if (!keys.some((key) => anchorParts(key).modelId === model)) fail(`--model ${model} matches no plan anchor`);
  }
  const rows = [];
  for (const key of keys) {
    if (args.anchors && !args.anchors.includes(key)) continue;
    if ((args.models ?? []).length > 0 && !args.models.includes(anchorParts(key).modelId)) continue;
    const row = await classifyAnchor(key, plan.anchors[key], { models, backend: args.backend, hubs, current, captured, declaredLanes, declaredProviders, sdxlRoutes });
    if (row.status === "other_backend" && !args.anchors) continue;
    // An unperformed route-revision comparison is reported on the row it did not happen for, so a
    // run without an inference checkout cannot silently look like a run that proved the engine agrees.
    if (row.routeCheck && ["runnable", "weights_missing"].includes(row.status)) {
      row.reason = row.reason ? `${row.reason}; ${row.routeCheck}` : row.routeCheck;
    }
    if (row.status === "runnable" && args.skipCurrent && row.current) {
      row.status = "current";
      row.reason = "anchor is current at the pinned inference revision (--skip-current)";
    }
    rows.push(row);
  }
  return { plan, rows, campaign, campaignDir, campaignPrefix: campaignDir, hubs };
}

function table(rows, columns) {
  const widths = columns.map((column) => Math.max(column.length, ...rows.map((row) => String(row[column] ?? "").length)));
  const line = (cells) => cells.map((cell, index) => String(cell ?? "").padEnd(widths[index])).join("  ").trimEnd();
  return [line(columns), line(widths.map((width) => "-".repeat(width))), ...rows.map((row) => line(columns.map((column) => row[column])))].join("\n");
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const { rows, campaign, campaignDir, campaignPrefix, hubs } = await planRun(args);

  if (args.list) {
    process.stdout.write(`${table(rows, ["key", "status", "reason"])}\n`);
    return;
  }

  const inferencePin = await compiledInferencePin();
  const preflight = [];
  const inferenceRepo = path.resolve(args.inferenceRepo);
  const inferenceHead = await git(["rev-parse", "HEAD"], inferenceRepo).catch((error) => { preflight.push(`inference repo: ${error.message}`); return null; });
  if (inferenceHead && inferenceHead !== inferencePin) {
    preflight.push(`inference checkout is at ${inferenceHead.slice(0, 12)} but the adapter compiled INFERENCE_PIN ${inferencePin.slice(0, 12)}; git -C ${inferenceRepo} checkout ${inferencePin}`);
  }
  if (!(await gitIsClean(ROOT))) preflight.push("SceneWorks checkout is dirty; commit or stash first (the harness counts untracked paths)");
  if (inferenceHead && !(await gitIsClean(inferenceRepo))) preflight.push("inference checkout is dirty");
  await mkdir(args.workDir, { recursive: true });
  await assertOutsideRepo(args.workDir, ROOT);
  if (args.adapter) {
    const [binary] = providerCommand(args.adapter);
    try { await stat(binary); } catch { if (!args.adapter.trimStart().startsWith("[")) preflight.push(`adapter binary not found at ${args.adapter}`); }
  }
  if (args.backend === "candle" && !process.env.CUDA_VISIBLE_DEVICES) {
    preflight.push("CUDA_VISIBLE_DEVICES is unset; the CUDA capture host pins one visible device (the runbook keeps GPU 1 visible)");
  }
  const branch = await git(["branch", "--show-current"]);
  if (args.commit && (branch === "main" || branch === "")) preflight.push(`refusing to commit measurements onto ${branch || "a detached HEAD"}; check out a story branch`);

  process.stdout.write(`campaign ${campaign} → ${campaignDir}\nhub roots: ${hubs.join(", ")}\n`);
  process.stdout.write(`${table(rows, ["key", "status", "reason"])}\n\n`);
  const runnable = rows.filter((row) => row.status === "runnable");
  if (args.dryRun) {
    for (const row of runnable) {
      process.stdout.write(`${row.key}\n  physical=${row.physical} ltx25=${Boolean(row.ltx25SnapshotRoot)}\n`);
      for (const root of row.roots) process.stdout.write(`  ${root.label}: ${root.path}\n`);
      for (const [name, value] of Object.entries(row.env)) process.stdout.write(`  ${name}=${value}\n`);
    }
    if (preflight.length > 0) process.stdout.write(`\npreflight would refuse:\n- ${preflight.join("\n- ")}\n`);
    process.stdout.write(`\ndry run: ${runnable.length} anchor(s) would be captured, nothing executed\n`);
    return;
  }
  if (preflight.length > 0) fail(`preflight:\n- ${preflight.join("\n- ")}`);
  if (runnable.length === 0) {
    process.stdout.write("nothing runnable on this host for this backend\n");
    return;
  }

  for (const sub of ["logs", "captures", "raw"]) await mkdir(path.join(args.workDir, sub), { recursive: true });
  const state = { commits: [], halt: null, stopRequested: false };
  const onInterrupt = () => {
    if (state.stopRequested) { process.stderr.write("\nsecond interrupt: exiting now; the adapter in flight is NOT killed\n"); process.exit(130); }
    state.stopRequested = true;
    process.stderr.write("\ninterrupt: finishing the anchor in flight, then stopping (press again to exit immediately)\n");
  };
  process.on("SIGINT", onInterrupt);
  process.on("SIGTERM", onInterrupt);

  const results = [];
  const summaryPath = path.join(args.workDir, `summary-${stamp()}.json`);
  const writeSummary = () => writeFile(summaryPath, JSON.stringify({ campaign, backend: args.backend, inferencePin, results, commits: state.commits, rows }, null, 2));
  for (const row of runnable) {
    if (state.halt) break;
    if (state.stopRequested) { results.push({ key: row.key, status: "not_started", reason: "interrupted" }); continue; }
    process.stdout.write(`\n=== ${row.key} (${results.length + 1}/${runnable.length}) ${new Date().toISOString()}\n`);
    const result = await measureAnchor(row, { args, inferencePin, campaignDir, campaignPrefix, workDir: args.workDir, state });
    results.push(result);
    process.stdout.write(`--- ${row.key}: ${result.status}${result.reason ? ` (${result.reason})` : ""} in ${result.seconds}s\n`);
    await writeSummary();
  }
  await writeSummary();
  process.stdout.write(`\n${table(results, ["key", "status", "seconds", "reason"])}\n`);
  process.stdout.write(`\ncommits: ${state.commits.length ? state.commits.join(" ") : "none"}\nsummary: ${summaryPath}\n`);
  if (state.halt) fail(state.halt);
  const failed = results.filter((result) => !["committed", "captured"].includes(result.status));
  if (failed.length > 0) process.exitCode = 2;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    process.stderr.write(`${error.message}\n`);
    process.exit(1);
  });
}
