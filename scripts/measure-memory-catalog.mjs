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
 * One row per provider arm an adapter implements, mirroring `match provider` in
 * crates/sceneworks-memory-adapter/src/bin/{mlx,candle}.rs and the env families the runbook lists
 * under "Adapter environment". `physical` marks the one arm that emits a provider `sourceCapture`
 * (the Qwen MLX source capture, mlx.rs `qwen_source_capture`): the harness REQUIRES a sourceCapture
 * whenever `--raw-log-dir` is given, so the raw-log pair and `SCENEWORKS_MEMORY_CAPTURE_DIR` must be
 * passed for that arm and for no other.
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
  sdxl: { env: "SDXL", repo: "SceneWorks/sdxl-base-mlx", arms: ["mlx"] },
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

export function fail(message) {
  throw new Error(message);
}

/**
 * The artifact family one anchor binds: the provider's row, with any per-modelId override applied.
 * A provider that serves several catalog models from ONE registry id (sc-22727's two klein models)
 * declares the divergent members under `variants`; everything else is the row itself.
 */
export function providerFamily(provider, modelId, families = PROVIDER_FAMILIES) {
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
export async function classifyAnchor(key, planned, { models, backend, hubs, current, captured, declaredLanes, declaredProviders, families = PROVIDER_FAMILIES }) {
  const parts = anchorParts(key);
  const row = { key, ...parts, provider: planned.provider, status: "runnable", reason: null, env: {}, roots: [] };
  if (parts.backend !== backend) return { ...row, status: "other_backend", reason: `${parts.backend} lane` };
  const family = providerFamily(planned.provider, parts.modelId, families);
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
