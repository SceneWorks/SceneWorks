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
 * will seal a contract. For the two Illustrious routes the pinned revision is not the one this
 * repository ships, so no root the manifest can resolve can ever seal — `candle_gen_sdxl::load`
 * errors before reading a weight. That is an INFERENCE-side divergence, not adapter work.
 */
const ILLUSTRIOUS_CANDLE_ROUTE_DRIFT = (modelId, engineRevision, manifestRevision) =>
  `candle-gen-sdxl pins route ${modelId} at ${engineRevision.slice(0, 8)} `
  + `(crates/media/candle-gen/candle-gen-sdxl/src/memory_strategy.rs SDXL_ROUTES), but this `
  + `repository ships ${manifestRevision.slice(0, 8)}; path_has_snapshot matches that literal, so `
  + `no staged root can seal the contract and the load fails before any weight is read. `
  + `The candle lane does not route this model at inference c6d6a4db.`;

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
 */
export const PROVIDER_FAMILIES = Object.freeze({
  qwen_image: { env: "QWEN_IMAGE", repo: "SceneWorks/qwen-image-mlx", arms: ["mlx", "candle"], physical: true },
  // `z_image_edit` anchors ride this family too (sc-22724): the catalog id is an alias for the
  // Turbo provider driven in `edit_image` mode (worker engines.rs `z_image_edit → z_image_turbo`),
  // and its manifest entry ships the same Turbo tiers, which `tierDownload` resolves by model id.
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
  sdxl: { provider: "sdxl", env: "SDXL", repo: "SceneWorks/sdxl-base-mlx", arms: ["mlx", "candle"], components: SDXL_COMPONENTS },
  realvisxl: { provider: "sdxl", env: "REALVISXL", repo: "SceneWorks/realvisxl-mlx", arms: ["mlx", "candle"], components: SDXL_COMPONENTS },
  realvisxl_lightning: {
    provider: "sdxl", env: "REALVISXL_LIGHTNING", repo: "SceneWorks/realvisxl-lightning-mlx",
    arms: ["mlx", "candle"], components: SDXL_COMPONENTS,
  },
  // MLX ONLY at inference c6d6a4db. `candle-gen-sdxl`'s `SDXL_ROUTES` pins these two routes at
  // revisions this repository no longer ships (`c5a92a90…` / `7c5c8b2b…` against the manifest's
  // `778c3f02…` / `672e9851…`), and `path_has_snapshot` matches that literal — so no root the
  // manifest can resolve will ever seal the contract, and `candle_gen_sdxl::load` errors before any
  // weight is read. The candle lane therefore does not route these two models at this pin; the
  // adapter arm exists and refuses them by naming both revisions.
  illustrious_xl_v1: {
    provider: "sdxl", env: "ILLUSTRIOUS_XL_V1", repo: "SceneWorks/illustrious-xl-v1-mlx",
    arms: ["mlx", "candle"], components: SDXL_COMPONENTS,
    laneUnsupported: { candle: ILLUSTRIOUS_CANDLE_ROUTE_DRIFT("illustrious_xl_v1", "c5a92a902dd4e6ee99c2a57981ecf66209905dd1", "778c3f02b7703b0c2755d0c0447592897193c6b5") },
  },
  illustrious_xl_v2: {
    provider: "sdxl", env: "ILLUSTRIOUS_XL_V2", repo: "SceneWorks/illustrious-xl-v2-mlx",
    arms: ["mlx", "candle"], components: SDXL_COMPONENTS,
    laneUnsupported: { candle: ILLUSTRIOUS_CANDLE_ROUTE_DRIFT("illustrious_xl_v2", "7c5c8b2bb75a8f38a7365e70bdf84d38d6204473", "672e9851ede4dc856fa945649b6691975c9d74a3") },
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
  flux2_dev: { env: "FLUX2", repo: "SceneWorks/flux2-dev-mlx", arms: ["mlx"] },
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
});

/**
 * The family row that serves one anchor.
 *
 * The default key is the PROVIDER, so a catalog alias rides its engine's row (`z_image_edit` on
 * `z_image_turbo`) and a lane-specific engine id keeps selecting the family (LTX-2.5's two ids).
 * sc-22729 adds the inverse case: several catalog models on ONE engine id, each with its own
 * artifact family. A MODEL-keyed row wins for those — but ONLY when it declares the provider it
 * belongs to and that provider is the one the plan named, so a model-keyed row can never capture
 * an anchor that some other engine serves.
 */
export function familyFor(modelId, provider, families = PROVIDER_FAMILIES) {
  const scoped = families[modelId];
  if (scoped?.provider !== undefined && scoped.provider === provider) return scoped;
  return families[provider];
}

export function fail(message) {
  throw new Error(message);
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
export async function classifyAnchor(key, planned, { models, backend, hubs, current, captured, declaredLanes, declaredProviders, families = PROVIDER_FAMILIES }) {
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
  // sc-22729: the same refusal, scoped to ONE lane. A model whose engine cannot serve it on a lane
  // is not a missing arm and not a missing declaration — the arm exists and refuses it by name —
  // so it reports the engine-side reason rather than sending the reader to adapter work.
  if (family?.laneUnsupported?.[backend]) {
    return { ...row, status: "harness_unsupported", reason: family.laneUnsupported[backend] };
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

  const download = tierDownload(models, parts.modelId, family.repo, parts.tier);
  const tierRoot = await firstExistingDirectory(hubs.map((hub) => snapshotPath(hub, family.repo, download.revision, parts.tier)));
  row.roots.push({ label: "tier root", path: tierRoot ?? snapshotPath(hubs[0], family.repo, download.revision, parts.tier) });
  if (!tierRoot) {
    return { ...row, status: "weights_missing", reason: `no ${family.repo}@${download.revision.slice(0, 8)}/${parts.tier} on this host` };
  }
  row.env[`SCENEWORKS_${family.env}_REPOSITORY`] = family.repo;
  row.env[`SCENEWORKS_${family.env}_REVISION`] = download.revision;
  row.env[`SCENEWORKS_${family.env}_ROOT`] = tierRoot;
  row.tierRoot = tierRoot;
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
  for (const component of family.components ?? []) {
    const download = tierDownload(models, parts.modelId, component.repo, parts.tier);
    const root = await firstExistingDirectory(hubs.map((hub) => snapshotPath(hub, component.repo, download.revision)));
    row.roots.push({ label: `component ${component.env}`, path: root ?? snapshotPath(hubs[0], component.repo, download.revision) });
    if (!root) {
      return { ...row, status: "weights_missing", reason: `no ${component.repo}@${download.revision.slice(0, 8)} snapshot on this host` };
    }
    row.env[component.env] = root;
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
    const row = await classifyAnchor(key, plan.anchors[key], { models, backend: args.backend, hubs, current, captured, declaredLanes, declaredProviders });
    if (row.status === "other_backend" && !args.anchors) continue;
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
