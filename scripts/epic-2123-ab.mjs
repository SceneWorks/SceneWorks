#!/usr/bin/env node
// Real-weight A/B harness for epic 2123 (perceptual character LoRA techniques, sc-2124).
//
// A resumable, phase-by-phase driver over the REAL rust-api + native worker: it trains one LoRA
// per technique row from the target's default CHARACTER PRESET (fetched from the API, every key
// kept), on the same dataset, seed, step count and resolution, renders a fixed prompt grid with
// each adapter (plus the bare base model, and optionally every intermediate checkpoint), scores
// the grids with the `lora_eval_harness` instrument, computes every adapter's stable rank and
// writes a report.
//
// Everything the harness writes lives under --root (default ~/.cache/sceneworks-epic-2123-ab):
// the API's SCENEWORKS_DATA_DIR / SCENEWORKS_CONFIG_DIR, logs, samples, eval output, the report
// and state.json. Model weights come from the shared Hugging Face cache (HF_HOME defaults to
// ~/.cache/huggingface, exactly as the desktop app resolves it), so already-cached snapshots are
// used in place (no copy, no re-download); a missing model is fetched through the app's own
// download route only with --allow-downloads. The desktop app's data dir is never touched.
//
// GPU discipline: every phase that runs MLX work (masks, baseline-timing, train, samples, and
// the eval instrument) refuses to start without BOTH an explicit `--phase <name>` and
// `--confirm-gpu`, starts the native worker only for its own duration, and runs one job at a
// time. The API and worker are torn down on exit, on Ctrl-C and on SIGTERM (the worker gets
// SIGTERM only, never SIGKILL, so an MLX command buffer is never cut mid-flight).
//
// Usage (see docs/epic-2123-ab.md for the full walkthrough):
//
//   node scripts/epic-2123-ab.mjs --dry-run                       # CPU: setup + every row validated
//   node scripts/epic-2123-ab.mjs --phase setup [--allow-downloads]
//   node scripts/epic-2123-ab.mjs --phase masks --confirm-gpu
//   node scripts/epic-2123-ab.mjs --phase baseline-timing --confirm-gpu
//   node scripts/epic-2123-ab.mjs --phase train --model zimage --confirm-gpu
//   node scripts/epic-2123-ab.mjs --phase samples --model zimage --confirm-gpu [--checkpoints]
//   node scripts/epic-2123-ab.mjs --phase eval --model zimage --confirm-gpu [--face-r50 <path> | --no-face-r50]
//   node scripts/epic-2123-ab.mjs --phase report
//   node scripts/epic-2123-ab.mjs --phase status

import { spawn, spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// ---------------------------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------------------------

const CAPTION_MODES = ["trigger", "full"];
const PHASES = ["setup", "masks", "baseline-timing", "train", "samples", "eval", "report", "status"];
const GPU_PHASES = new Set(["masks", "baseline-timing", "train", "samples"]);

function parseArgs(argv) {
  const opts = {
    root: path.join(os.homedir(), ".cache", "sceneworks-epic-2123-ab"),
    dataset: path.join(os.homedir(), "Datasets", "Kelsie"),
    trigger: "kelsie",
    port: 8766,
    profile: null,
    bin: null,
    build: false,
    phase: null,
    model: null,
    confirmGpu: false,
    allowDownloads: false,
    dryRun: false,
    // null = the preset's own value (steps, rank, learning rate, batch size, resolution).
    steps: null,
    rank: null,
    lr: null,
    seed: 7,
    batch: null,
    resolution: null,
    // Preset ids that replace a target's default character preset (matched by targetId).
    presets: [],
    // `trigger`: every item's caption is the trigger word alone; `full`: the dataset's .txt text.
    captions: "trigger",
    // null = the step count (only the final adapter is written).
    saveEvery: null,
    checkpoints: false,
    depthModel: "small",
    rows: null,
    force: false,
    loraWeight: 0.8,
    genSize: null,
    seeds: [1001, 2002],
    maxWorkerRssGb: Math.floor((os.totalmem() / 2 ** 30) * 0.85),
    pollSeconds: 5,
    hfHome: null,
    evalRelease: false,
    allowDebugBuild: false,
    // The independent identity network (w600k_r50, sc-2124). null = the default path under the
    // root if that file exists, else the metric is skipped; `--no-face-r50` forces it off.
    faceR50: null,
    noFaceR50: false,
  };
  const need = (i, flag) => {
    if (i + 1 >= argv.length) throw new Error(`${flag} needs a value`);
    return argv[i + 1];
  };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    switch (a) {
      case "--root": opts.root = path.resolve(need(i, a)); i += 1; break;
      case "--dataset": opts.dataset = path.resolve(need(i, a)); i += 1; break;
      case "--trigger": opts.trigger = need(i, a); i += 1; break;
      case "--port": opts.port = Number(need(i, a)); i += 1; break;
      case "--profile": opts.profile = need(i, a); i += 1; break;
      case "--bin": opts.bin = path.resolve(need(i, a)); i += 1; break;
      case "--build": opts.build = true; break;
      case "--phase": opts.phase = need(i, a); i += 1; break;
      case "--model": opts.model = need(i, a); i += 1; break;
      case "--confirm-gpu": opts.confirmGpu = true; break;
      case "--allow-downloads": opts.allowDownloads = true; break;
      case "--dry-run": opts.dryRun = true; break;
      case "--steps": opts.steps = Number(need(i, a)); i += 1; break;
      case "--rank": opts.rank = Number(need(i, a)); i += 1; break;
      case "--lr": opts.lr = Number(need(i, a)); i += 1; break;
      case "--seed": opts.seed = Number(need(i, a)); i += 1; break;
      case "--batch": opts.batch = Number(need(i, a)); i += 1; break;
      case "--resolution": opts.resolution = Number(need(i, a)); i += 1; break;
      case "--preset": opts.presets = need(i, a).split(",").map((s) => s.trim()).filter(Boolean); i += 1; break;
      case "--captions": opts.captions = need(i, a); i += 1; break;
      case "--save-every": opts.saveEvery = Number(need(i, a)); i += 1; break;
      case "--checkpoints": opts.checkpoints = true; break;
      case "--gen-size": opts.genSize = Number(need(i, a)); i += 1; break;
      case "--depth-model": opts.depthModel = need(i, a); i += 1; break;
      case "--rows": opts.rows = need(i, a).split(",").map((s) => s.trim()).filter(Boolean); i += 1; break;
      case "--force": opts.force = true; break;
      case "--lora-weight": opts.loraWeight = Number(need(i, a)); i += 1; break;
      case "--seeds": opts.seeds = need(i, a).split(",").map(Number); i += 1; break;
      case "--max-worker-rss-gb": opts.maxWorkerRssGb = Number(need(i, a)); i += 1; break;
      case "--poll-seconds": opts.pollSeconds = Number(need(i, a)); i += 1; break;
      case "--hf-home": opts.hfHome = path.resolve(need(i, a)); i += 1; break;
      case "--eval-release": opts.evalRelease = true; break;
      case "--face-r50": opts.faceR50 = path.resolve(need(i, a)); i += 1; break;
      case "--no-face-r50": opts.noFaceR50 = true; break;
      case "--allow-debug-build": opts.allowDebugBuild = true; break;
      case "--stable-rank": opts.stableRank = need(i, a); i += 1; break;
      case "-h":
      case "--help":
        opts.help = true;
        break;
      default:
        throw new Error(`unknown argument: ${a}`);
    }
  }
  if (opts.phase && !PHASES.includes(opts.phase)) {
    throw new Error(`--phase must be one of ${PHASES.join(", ")}`);
  }
  if (opts.model && !Object.hasOwn(MODELS, opts.model)) {
    throw new Error(`--model must be one of ${Object.keys(MODELS).join(", ")}`);
  }
  if (!["small", "base", "large"].includes(opts.depthModel)) {
    throw new Error("--depth-model must be small, base or large");
  }
  if (!CAPTION_MODES.includes(opts.captions)) {
    throw new Error(`--captions must be one of ${CAPTION_MODES.join(", ")}`);
  }
  for (const [flag, value] of [["--steps", opts.steps], ["--rank", opts.rank], ["--batch", opts.batch], ["--resolution", opts.resolution], ["--save-every", opts.saveEvery], ["--gen-size", opts.genSize]]) {
    if (value !== null && !(Number.isInteger(value) && value > 0)) throw new Error(`${flag} must be a positive integer`);
  }
  if (opts.lr !== null && !(opts.lr > 0)) throw new Error("--lr must be a positive number");
  if (!(opts.loraWeight > 0)) throw new Error("--lora-weight must be a positive number");
  if (opts.faceR50 && opts.noFaceR50) throw new Error("--face-r50 and --no-face-r50 are mutually exclusive");
  // Resolved against the real root here, before a dry run re-roots opts.root.
  opts.faceR50Default = path.join(opts.root, "models", "w600k_r50", "arcface_w600k_r50.safetensors");
  return opts;
}

// ---------------------------------------------------------------------------------------------
// The A/B design
// ---------------------------------------------------------------------------------------------

// Per base model: the training target, the family-specific auxiliary models, and the sample
// render settings. The training config itself comes from the target's default character preset
// (`GET /api/v1/training/presets`), so resolution, steps, rank, learning rate, optimizer,
// timestep schedule and any training adapter are the app's own, not harness constants.
const MODELS = {
  zimage: {
    label: "Z-Image-Turbo",
    target: "z_image_turbo_lora",
    baseModel: "z_image_turbo",
    trainTier: "bf16",
    x0Decoder: "taef1",
    latentLpips: "elatentlpips_flux",
    gen: { width: 1024, height: 1024, advanced: { steps: 8 } },
  },
  sdxl: {
    label: "SDXL",
    target: "sdxl_lora",
    baseModel: "sdxl",
    trainTier: "bf16",
    x0Decoder: "taesdxl",
    latentLpips: "elatentlpips_sdxl",
    gen: { width: 1024, height: 1024, advanced: { steps: 30, guidanceScale: 7.0 } },
  },
};

// The technique rows. Each row overlays `advanced` keys on the identical baseline config. Weights
// follow the sceneworks-core suggested values (themselves the upstream ai-toolkit-perceptual
// starting points), trimmed toward the modest end where upstream gives a range; every other knob
// (noise window, alternation period, gates) is left to the shared parser's upstream default.
// `limits` are the target `limits.supports*` flags the row needs; a row whose target does not
// advertise all of them is skipped and the skip recorded.
function techniqueRows(modelKey, opts) {
  const m = MODELS[modelKey];
  const depthAux = `depth_anything_v2_${opts.depthModel}`;
  return [
    { id: "baseline", label: "Baseline (all off)", advanced: {}, limits: [], aux: [] },
    ...[0.005, 0.0125, 0.02].map((sigma) => ({
      id: `weight_noise_${String(sigma).replace(".", "p")}`,
      label: `Weight noise σ=${sigma}`,
      advanced: { weightNoiseSigma: sigma },
      limits: ["supportsWeightNoise"],
      aux: [],
    })),
    {
      id: "resolution_buckets",
      label: "Resolution buckets",
      // Filled from the target's allowed edges and the run's base resolution (`bucketLadder`).
      advanced: {},
      bucketLadder: true,
      limits: ["supportsResolutionBuckets"],
      aux: [],
    },
    {
      id: "subject_mask",
      label: "Subject-masked loss (bg 0.1)",
      advanced: { subjectMaskLoss: true, subjectMaskBackgroundWeight: 0.1, subjectMaskSubjectWeight: 1.0 },
      limits: ["supportsSubjectMaskLoss"],
      aux: [],
      needsMasks: true,
    },
    {
      id: "depth_anchoring",
      label: `Depth anchoring (DA2-${opts.depthModel}, w=0.1)`,
      advanced: { depthAnchoringWeight: 0.1, depthAnchoringModel: opts.depthModel },
      limits: ["supportsDepthAnchoring"],
      aux: [m.x0Decoder, depthAux],
    },
    {
      id: "identity",
      label: "Identity (ArcFace w=0.05) + face landmark (w=0.05)",
      advanced: { identityLossWeight: 0.05, faceLandmarkLossWeight: 0.05 },
      limits: ["supportsIdentityLoss", "supportsFaceLandmarkLoss"],
      aux: [m.x0Decoder, "instantid_face_stack", "mp_facemesh_v2"],
    },
    {
      id: "body",
      label: "Body: proportion 0.1 + shape 0.05 + normal 0.05",
      advanced: { bodyProportionWeight: 0.1, bodyShapeWeight: 0.05, normalWeight: 0.05 },
      limits: ["supportsBodyProportionLoss", "supportsBodyShapeLoss", "supportsNormalLoss"],
      aux: [m.x0Decoder, "vitpose_plus_base", "hybrik_resnet34", "sapiens_normal_0_3b"],
    },
    {
      id: "latent_perceptual",
      label: "Latent perceptual: VAE anchor 0.5 + E-LatentLPIPS 0.5",
      advanced: { vaeAnchorWeight: 0.5, latentLpipsWeight: 0.5 },
      // Each half is gated separately: the row keeps whichever half the target supports
      // (`optionalLimits`) and is skipped only when neither is.
      limits: [],
      optionalLimits: { supportsVaeAnchorLoss: ["vaeAnchorWeight"], supportsLatentLpipsLoss: ["latentLpipsWeight"] },
      aux: [m.x0Decoder, "flux2_vae", m.latentLpips],
    },
    {
      // The two batch-1 winners together: do their gains stack before both become defaults?
      id: "mask_identity",
      label: "Subject mask (bg 0.1) + identity (0.05) + face landmark (0.05)",
      advanced: {
        subjectMaskLoss: true,
        subjectMaskBackgroundWeight: 0.1,
        subjectMaskSubjectWeight: 1.0,
        identityLossWeight: 0.05,
        faceLandmarkLossWeight: 0.05,
      },
      limits: ["supportsSubjectMaskLoss", "supportsIdentityLoss", "supportsFaceLandmarkLoss"],
      aux: [m.x0Decoder, "instantid_face_stack", "mp_facemesh_v2"],
      needsMasks: true,
    },
    {
      // The two Z-Image winners on the independent likeness score: do buckets and masking stack?
      id: "buckets_mask",
      label: "Resolution buckets + subject mask (bg 0.1)",
      labelSuffix: "subject mask (bg 0.1)",
      advanced: { subjectMaskLoss: true, subjectMaskBackgroundWeight: 0.1, subjectMaskSubjectWeight: 1.0 },
      bucketLadder: true,
      limits: ["supportsResolutionBuckets", "supportsSubjectMaskLoss"],
      aux: [],
      needsMasks: true,
    },
  ];
}

const PROMPTS = [
  ["portrait", "photo of {t}, close-up portrait, soft window light, neutral background"],
  ["street_night", "photo of {t} standing on a city street at night, wearing a red leather jacket, full body"],
  ["cafe", "photo of {t} sitting at a cafe table, wearing a white knit sweater, looking out the window"],
  ["hiking", "photo of {t} hiking a mountain trail, wearing a green rain jacket and a backpack, side view"],
  ["evening_dress", "photo of {t} in a black evening dress on a marble staircase, three-quarter view"],
  ["beach", "photo of {t} laughing on a beach at sunset, wearing a denim shirt, wind in her hair"],
  ["kitchen", "photo of {t} cooking in a kitchen, wearing an apron over a striped t-shirt, seen from above"],
  // Full body with the face toward the camera (replaced a yoga pose that hid the face, so it
  // could not show likeness).
  ["party", "photo of {t} dancing at a rooftop party, wearing a sequin dress, full body, facing the camera"],
];

const BASE_AUX = ["sam3_person_segment", "instantid_face_stack"];

// The bucket ladder for a base resolution: every edge the target trains at up to and including
// the base resolution, smallest first, with the upstream 2^k repeat skew (4:2:1 for three edges,
// 2:1 for two), so the smallest edge is visited most. At 1024 that is Z-Image 512/768/1024 ×4:2:1
// and SDXL 768/1024 ×2:1. Submit-time validation refuses an edge the target does not allow, so
// nothing off the target's list is ever invented. When fewer than two allowed edges sit at or
// below the base (e.g. a 512 Z-Image run), the ladder climbs from the base instead.
function bucketLadder(target, baseResolution) {
  const allowed = [...new Set((target.limits?.resolutions ?? []).map(Number))].filter(Number.isFinite).sort((a, b) => a - b);
  let edges = allowed.filter((r) => r <= baseResolution);
  if (edges.length < 2) edges = allowed.filter((r) => r >= baseResolution);
  if (edges.length < 2) throw new Error(`target ${target.id} allows fewer than two resolutions (${allowed.join(", ")}); no bucket ladder`);
  return edges.map((resolution, i) => ({ resolution, repeats: 2 ** (edges.length - 1 - i) }));
}

function ladderLabel(ladder) {
  return `Resolution buckets ${ladder.map((b) => b.resolution).join("/")} ×${ladder.map((b) => b.repeats).join(":")}`;
}

// The de-distill training adapter catalog model a preset's `advanced` selects (sc-25213), or null —
// the same mapping the worker resolves (`sceneworks_core::training::training_adapter`): v1 → the v1
// file; v2, the presets' "v2-default" and blank → v2. An unknown repo/version is the API's field
// error at submit, so setup has nothing to install for it.
function trainingAdapterModelId(advanced) {
  const repo = String(advanced?.trainingAdapterRepo ?? "").trim();
  if (repo !== "ostris/zimage_turbo_training_adapter") return null;
  const version = String(advanced?.trainingAdapterVersion ?? "").trim().toLowerCase();
  if (version === "v1") return "zimage_turbo_training_adapter_v1";
  if (["", "v2", "v2-default"].includes(version)) return "zimage_turbo_training_adapter_v2";
  return null;
}

function auxModelsFor(modelKey, opts) {
  const set = new Set(BASE_AUX);
  for (const row of techniqueRows(modelKey, opts)) row.aux.forEach((id) => set.add(id));
  return [...set];
}

// ---------------------------------------------------------------------------------------------
// State + logging
// ---------------------------------------------------------------------------------------------

let OPTS;
let STATE_PATH;
let STATE;

function log(...args) {
  console.log(`[epic-2123-ab ${new Date().toISOString().slice(11, 19)}]`, ...args);
}

function loadState(root) {
  STATE_PATH = path.join(root, "state.json");
  try {
    STATE = JSON.parse(fs.readFileSync(STATE_PATH, "utf8"));
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    STATE = { version: 2, createdAt: new Date().toISOString(), training: {}, samples: {}, eval: {}, rank: {} };
  }
  for (const key of ["training", "samples", "eval", "evalKeys", "rank", "datasets", "checkpointLoras"]) STATE[key] ??= {};
  // v1 state held one dataset (imported with the full .txt captions) and its subject masks at
  // the top level. Datasets are now keyed by caption mode, each carrying its own masks.
  if (STATE.dataset) {
    STATE.datasets.full ??= { ...STATE.dataset, captionMode: "full", masks: STATE.masks ?? null };
    delete STATE.dataset;
    delete STATE.masks;
  }
  STATE.version = 2;
}

// The dataset for the run's caption mode (each mode is its own dataset, side by side).
function currentDataset() {
  return STATE.datasets[OPTS.captions] ?? null;
}

function saveState() {
  const tmp = `${STATE_PATH}.tmp`;
  fs.writeFileSync(tmp, `${JSON.stringify(STATE, null, 2)}\n`);
  fs.renameSync(tmp, STATE_PATH);
}

// ---------------------------------------------------------------------------------------------
// Process lifecycle (API + worker), teardown on exit / Ctrl-C
// ---------------------------------------------------------------------------------------------

const PROCS = { api: null, worker: null };
let ACTIVE_JOB = null;
let SHUTTING_DOWN = false;

function resolveBinary(opts, { gpu }) {
  if (opts.bin) return opts.bin;
  // --build defaults to the release profile; otherwise prefer an existing release binary and fall
  // back to debug (which only CPU phases accept).
  const profile =
    opts.profile ??
    (opts.build || fs.existsSync(path.join(REPO, "target", "release", "sceneworks-rust-api")) ? "release" : "debug");
  if (gpu && profile !== "release" && !opts.allowDebugBuild) {
    throw new Error(
      "GPU phases need the RELEASE build (a debug-build trainer is many times slower). Build it with " +
        "--build, or pass --allow-debug-build to override.",
    );
  }
  if (opts.build) buildBinary(profile);
  const bin = path.join(REPO, "target", profile, "sceneworks-rust-api");
  if (!fs.existsSync(bin)) {
    throw new Error(`${bin} does not exist; rerun with --build (or --profile/--bin).`);
  }
  return bin;
}

function prebuiltMlxEnv(buildType) {
  if (process.platform !== "darwin") return {};
  const out = spawnSync("bash", [path.join(REPO, "scripts", "fetch-prebuilt-mlx.sh"), "--build-type", buildType], {
    encoding: "utf8",
  });
  if (out.status !== 0) {
    throw new Error(`fetch-prebuilt-mlx.sh failed (${out.status}): ${out.stderr}`);
  }
  const env = {};
  for (const line of out.stdout.split("\n")) {
    const match = /^(PMETAL_MLX_PREBUILT_DIR|PMETAL_METALLIB_PATH)=(.+)$/.exec(line.trim());
    if (match) env[match[1]] = match[2];
  }
  return env;
}

function buildBinary(profile) {
  const env = { ...process.env, ...prebuiltMlxEnv(profile === "release" ? "Release" : "Debug") };
  const args = ["build", "-p", "sceneworks-rust-api", "--bin", "sceneworks-rust-api"];
  if (profile === "release") args.push("--release");
  log(`building sceneworks-rust-api (${profile})`);
  const out = spawnSync("cargo", args, { cwd: REPO, env, stdio: "inherit" });
  if (out.status !== 0) throw new Error(`cargo build failed (${out.status})`);
}

function baseEnv(opts) {
  const env = { ...process.env };
  env.SCENEWORKS_DATA_DIR = path.join(opts.root, "data");
  env.SCENEWORKS_CONFIG_DIR = path.join(opts.root, "config");
  env.SCENEWORKS_API_HOST = "127.0.0.1";
  env.SCENEWORKS_API_PORT = String(opts.port);
  env.SCENEWORKS_TRUST_LOOPBACK = "1";
  delete env.SCENEWORKS_ACCESS_TOKEN;
  if (opts.hfHome) env.HF_HOME = opts.hfHome;
  return env;
}

function apiUrl(opts) {
  return `http://127.0.0.1:${opts.port}`;
}

function spawnLogged(bin, env, logName) {
  const logDir = path.join(OPTS.root, "logs");
  fs.mkdirSync(logDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const logPath = path.join(logDir, `${logName}-${stamp}.log`);
  const fd = fs.openSync(logPath, "a");
  // Own process group: a terminal Ctrl-C reaches only this driver, which then tears the children
  // down in order (cancel the job, SIGTERM the worker and let it finish its command buffer, stop
  // the API) instead of every process receiving SIGINT at once.
  const child = spawn(bin, [], { env, stdio: ["ignore", fd, fd], detached: true });
  fs.closeSync(fd);
  child.logPath = logPath;
  child.exited = false;
  child.on("exit", () => {
    child.exited = true;
  });
  return child;
}

async function startApi(opts, bin) {
  // Refuse to attach to (or race) a server that is already listening on the port: the harness
  // must only ever talk to its own isolated API.
  if (await healthy(opts)) {
    throw new Error(`something is already serving ${apiUrl(opts)}; stop it or pick another --port`);
  }
  fs.mkdirSync(path.join(opts.root, "data"), { recursive: true });
  fs.mkdirSync(path.join(opts.root, "config"), { recursive: true });
  const env = baseEnv(opts);
  // The in-process utility worker (cpu) serves model downloads; it never claims a GPU job.
  env.SCENEWORKS_RUN_UTILITY_INPROCESS = "1";
  PROCS.api = spawnLogged(bin, env, "api");
  log(`API starting on ${apiUrl(opts)} (log ${PROCS.api.logPath})`);
  for (let i = 0; i < 180; i += 1) {
    if (PROCS.api.exited) throw new Error(`API exited early; see ${PROCS.api.logPath}`);
    if (await healthy(opts)) {
      const health = await api("GET", "/api/v1/health");
      if (path.resolve(health.directories?.data ?? "") !== path.resolve(env.SCENEWORKS_DATA_DIR)) {
        throw new Error(`API reports data dir ${health.directories?.data}, expected ${env.SCENEWORKS_DATA_DIR}`);
      }
      log("API healthy");
      return;
    }
    await sleep(1000);
  }
  throw new Error(`API did not become healthy; see ${PROCS.api.logPath}`);
}

async function startWorker(opts, bin, capability) {
  const env = baseEnv(opts);
  env.SCENEWORKS_WORKER_ONLY = "1";
  env.SCENEWORKS_API_URL = apiUrl(opts);
  env.SCENEWORKS_WORKER_ID = "epic-2123-ab-gpu";
  env.SCENEWORKS_GPU_ID = process.env.SCENEWORKS_GPU_ID || (process.platform === "darwin" ? "mlx" : "0");
  PROCS.worker = spawnLogged(bin, env, "worker");
  log(`GPU worker starting (SCENEWORKS_GPU_ID=${env.SCENEWORKS_GPU_ID}, log ${PROCS.worker.logPath})`);
  for (let i = 0; i < 300; i += 1) {
    if (PROCS.worker.exited) throw new Error(`worker exited early; see ${PROCS.worker.logPath}`);
    const workers = await api("GET", "/api/v1/workers").catch(() => []);
    const mine = (Array.isArray(workers) ? workers : []).find((w) => w.id === env.SCENEWORKS_WORKER_ID);
    if (mine && (mine.capabilities ?? []).includes(capability)) {
      log(`GPU worker registered with ${capability}`);
      return;
    }
    await sleep(1000);
  }
  throw new Error(`worker never advertised ${capability}; see ${PROCS.worker.logPath}`);
}

function waitExit(child, ms) {
  return new Promise((resolve) => {
    if (!child || child.exited) return resolve(true);
    const timer = setTimeout(() => resolve(false), ms);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve(true);
    });
  });
}

// Both stop functions are idempotent and share one in-flight promise, so the signal handler and
// a phase's own `finally` can both call them and both wait for the same, complete teardown.
let STOPPING_WORKER = null;
let STOPPING_API = null;

function stopWorker() {
  STOPPING_WORKER ??= (async () => {
    const w = PROCS.worker;
    if (!w || w.exited) return;
    log("stopping GPU worker (SIGTERM)");
    w.kill("SIGTERM");
    // Never SIGKILL an MLX worker: it finishes its current command buffer and exits.
    while (!(await waitExit(w, 30000))) log("worker still finishing its command buffer…");
  })().finally(() => {
    PROCS.worker = null;
    STOPPING_WORKER = null;
  });
  return STOPPING_WORKER;
}

function stopApi() {
  STOPPING_API ??= (async () => {
    // The worker reports to the API until it exits, so the API always goes down second.
    await stopWorker();
    const a = PROCS.api;
    if (!a || a.exited) return;
    log("stopping API");
    a.kill("SIGTERM");
    if (!(await waitExit(a, 20000))) a.kill("SIGKILL");
  })().finally(() => {
    PROCS.api = null;
    STOPPING_API = null;
  });
  return STOPPING_API;
}

async function shutdown(code) {
  if (SHUTTING_DOWN) return;
  SHUTTING_DOWN = true;
  if (ACTIVE_JOB && PROCS.api && !PROCS.api.exited) {
    log(`canceling active job ${ACTIVE_JOB}`);
    await api("POST", `/api/v1/jobs/${ACTIVE_JOB}/cancel`, {}).catch(() => {});
    // Give the worker a moment to observe the cancel between steps before it is signalled.
    for (let i = 0; i < 30; i += 1) {
      const job = await api("GET", `/api/v1/jobs/${ACTIVE_JOB}`).catch(() => null);
      if (!job || TERMINAL.has(job.status)) break;
      await sleep(1000);
    }
  }
  await stopWorker();
  await stopApi();
  process.exit(code);
}

process.on("SIGINT", () => {
  log("interrupted; tearing down");
  shutdown(130);
});
process.on("SIGTERM", () => {
  log("terminated; tearing down");
  shutdown(143);
});
// Last-resort synchronous cleanup (uncaught error paths): signal whatever is still alive.
process.on("exit", () => {
  for (const child of [PROCS.worker, PROCS.api]) {
    if (child && !child.exited) child.kill("SIGTERM");
  }
});

// ---------------------------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------------------------

const TERMINAL = new Set(["completed", "failed", "canceled", "cancelled", "interrupted"]);

class ApiError extends Error {
  constructor(status, body, method, route) {
    super(`${method} ${route} -> ${status}: ${typeof body === "string" ? body : JSON.stringify(body)}`);
    this.status = status;
    this.body = body;
  }
}

async function api(method, route, body) {
  const init = { method, headers: {} };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.headers["content-type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(`${apiUrl(OPTS)}${route}`, init);
  const text = await res.text();
  let parsed = text;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    // keep text
  }
  if (!res.ok) throw new ApiError(res.status, parsed, method, route);
  return parsed;
}

async function healthy(opts) {
  try {
    const res = await fetch(`${apiUrl(opts)}/api/v1/health`);
    return res.ok;
  } catch {
    return false;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function workerRssGb() {
  const w = PROCS.worker;
  if (!w || w.exited) return 0;
  const out = spawnSync("ps", ["-o", "rss=", "-p", String(w.pid)], { encoding: "utf8" });
  const kb = Number(out.stdout.trim());
  return Number.isFinite(kb) ? kb / 2 ** 20 : 0;
}

// Poll a job to a terminal state, printing progress transitions, tracking the training step
// clock (for s/step) and guarding the worker's resident memory.
async function waitJob(jobId, { onProgress } = {}) {
  ACTIVE_JOB = jobId;
  let lastMessage = "";
  const steps = [];
  const losses = [];
  try {
    for (;;) {
      const job = await api("GET", `/api/v1/jobs/${jobId}`);
      const message = job.message ?? "";
      if (message !== lastMessage) {
        log(`  ${job.status} ${(Number(job.progress ?? 0) * 100).toFixed(0)}% ${message}`);
        lastMessage = message;
      }
      const match = /Training step (\d+) of (\d+) \(loss ([-\d.eE+naN]+)\)/.exec(message);
      if (match) {
        const step = Number(match[1]);
        if (!steps.length || steps.at(-1).step !== step) {
          steps.push({ step, total: Number(match[2]), t: Date.now() / 1000 });
          losses.push({ step, loss: Number(match[3]) });
        }
      }
      onProgress?.(job);
      if (TERMINAL.has(job.status)) return { job, steps, losses };
      if (OPTS.maxWorkerRssGb > 0 && PROCS.worker) {
        const rss = workerRssGb();
        if (rss > OPTS.maxWorkerRssGb) {
          log(`worker RSS ${rss.toFixed(1)} GB exceeds --max-worker-rss-gb ${OPTS.maxWorkerRssGb}; canceling`);
          await api("POST", `/api/v1/jobs/${jobId}/cancel`, {}).catch(() => {});
          throw new Error(`worker RSS guard tripped at ${rss.toFixed(1)} GB`);
        }
      }
      await sleep(OPTS.pollSeconds * 1000);
    }
  } finally {
    ACTIVE_JOB = null;
  }
}

function secondsPerStep(steps) {
  if (steps.length < 2) return null;
  const first = steps[0];
  const last = steps.at(-1);
  if (last.step === first.step) return null;
  return (last.t - first.t) / (last.step - first.step);
}

function wallSeconds(job) {
  if (!job.startedAt || !job.completedAt) return null;
  return (Date.parse(job.completedAt) - Date.parse(job.startedAt)) / 1000;
}

async function cancelStale(jobId) {
  if (!jobId) return;
  const job = await api("GET", `/api/v1/jobs/${jobId}`).catch(() => null);
  if (job && !TERMINAL.has(job.status)) {
    log(`canceling stale job ${jobId} (${job.status}) from an interrupted run`);
    await api("POST", `/api/v1/jobs/${jobId}/cancel`, {}).catch(() => {});
  }
}

// ---------------------------------------------------------------------------------------------
// Phase: setup
// ---------------------------------------------------------------------------------------------

async function phaseSetup(opts) {
  // Project.
  if (STATE.project) {
    const existing = await api("GET", `/api/v1/projects/${STATE.project.id}`).catch(() => null);
    if (!existing) delete STATE.project;
  }
  if (!STATE.project) {
    const project = await api("POST", "/api/v1/projects", { name: "epic-2123-ab" });
    STATE.project = { id: project.id, path: project.path };
    saveState();
    log(`created project ${project.id}`);
  } else {
    log(`project ${STATE.project.id} exists`);
  }

  await ensureDataset(opts, opts.captions);

  // Models: resolve by catalog id against the shared HF cache.
  await resolveModels(opts);
}

// Import the dataset for one caption mode: every <stem>.png with a <stem>.txt caption. In
// `trigger` mode each item's caption text is the trigger word alone, so identity can only bind
// to the trigger (a descriptive caption such as "a blonde woman with wavy hair" would otherwise
// absorb it); `full` keeps the .txt text. A stored dataset is reused only when its caption mode,
// source directory and trigger all match and it still exists in the API.
async function ensureDataset(opts, mode) {
  const existing = STATE.datasets[mode];
  if (existing) {
    const alive = await api("GET", `/api/v1/projects/${STATE.project.id}/training/datasets/${existing.id}`).catch(() => null);
    const matches = existing.captionMode === mode && existing.source === opts.dataset && existing.trigger === opts.trigger;
    if (alive && matches) {
      log(`dataset ${existing.id} exists (${existing.itemCount} items, ${mode} captions)`);
      return existing;
    }
    log(`dataset ${existing.id} (${mode}) is ${alive ? "for a different source/trigger" : "gone"}; re-importing`);
    delete STATE.datasets[mode];
  }
  const pngs = fs
    .readdirSync(opts.dataset)
    .filter((f) => /\.(png|jpe?g|webp)$/i.test(f))
    .sort();
  const items = [];
  for (const file of pngs) {
    const stem = file.replace(/\.[^.]+$/, "");
    const captionPath = path.join(opts.dataset, `${stem}.txt`);
    if (!fs.existsSync(captionPath)) {
      log(`  skipping ${file}: no ${stem}.txt caption`);
      continue;
    }
    const bytes = fs.readFileSync(path.join(opts.dataset, file));
    const form = new FormData();
    const type = /\.png$/i.test(file) ? "image/png" : /\.webp$/i.test(file) ? "image/webp" : "image/jpeg";
    form.append("file", new Blob([bytes], { type }), file);
    const upload = await api("POST", `/api/v1/projects/${STATE.project.id}/training/uploads`, form);
    const text = mode === "trigger" ? opts.trigger : fs.readFileSync(captionPath, "utf8").trim();
    items.push({
      path: upload.file.path,
      displayName: file,
      width: upload.file.width,
      height: upload.file.height,
      caption: { text, triggerWords: [opts.trigger] },
    });
  }
  if (!items.length) throw new Error(`no captioned images in ${opts.dataset}`);
  const dataset = await api("POST", `/api/v1/projects/${STATE.project.id}/training/datasets`, {
    name: `${path.basename(opts.dataset)} (${opts.trigger}, ${mode} captions)`,
    modality: "image",
    items,
  });
  STATE.datasets[mode] = {
    id: dataset.id,
    captionMode: mode,
    itemCount: items.length,
    source: opts.dataset,
    trigger: opts.trigger,
    masks: null,
  };
  saveState();
  log(`imported ${items.length} images as dataset ${dataset.id} (${mode} captions)`);
  return STATE.datasets[mode];
}

async function resolveModels(opts) {
  const catalog = await api("GET", "/api/v1/models");
  const byId = new Map(catalog.map((m) => [m.id, m]));
  const wanted = [];
  for (const key of Object.keys(MODELS)) {
    const m = MODELS[key];
    // Training reads the dense tier; generation reads the catalog default tier.
    wanted.push({ id: m.baseModel, variants: [m.trainTier, "default"], role: `${m.label} base` });
  }
  for (const id of new Set(Object.keys(MODELS).flatMap((k) => auxModelsFor(k, opts)))) {
    wanted.push({ id, variants: ["default"], role: "aux" });
  }
  // The training adapter each model's preset trains with (Z-Image-Turbo's de-distill adapter, at
  // the preset's own version), installed through the same download path — the API refuses a real run
  // whose adapter is missing.
  for (const key of Object.keys(MODELS)) {
    const id = trainingAdapterModelId((await presetFor(key)).config?.advanced);
    if (id) wanted.push({ id, variants: ["default"], role: `${MODELS[key].label} training adapter` });
  }
  const resolution = {};
  const toDownload = [];
  for (const want of wanted) {
    const entry = byId.get(want.id);
    if (!entry) {
      resolution[want.id] = { state: "not-in-catalog" };
      continue;
    }
    const variants = entry.variants ?? [];
    const pick = (name) =>
      name === "default"
        ? variants.find((v) => v.variant === (entry.downloads?.find((d) => d.default)?.variant ?? "default")) ?? variants[0]
        : variants.find((v) => v.variant === name);
    const needed = [...new Set(want.variants.map(pick).filter(Boolean))];
    const missing = needed.filter((v) => v.installState !== "installed");
    resolution[want.id] = {
      role: want.role,
      state: missing.length ? "needs-download" : "linked-from-cache",
      installedPath: entry.installedPath,
      variants: needed.map((v) => ({ variant: v.variant, installState: v.installState, bytes: v.downloadSizeBytes ?? null })),
      downloadBytes: missing.reduce((sum, v) => sum + (v.downloadSizeBytes ?? 0), 0),
    };
    for (const v of missing) toDownload.push({ id: want.id, variant: v.variant, bytes: v.downloadSizeBytes ?? 0 });
  }
  STATE.models = resolution;
  saveState();
  log("model resolution (shared HF cache, zero-copy):");
  for (const [id, r] of Object.entries(resolution)) {
    const detail = r.state === "needs-download" ? `needs ${gb(r.downloadBytes)}` : r.installedPath ?? "";
    log(`  ${id.padEnd(26)} ${r.state.padEnd(18)} ${detail}`);
  }
  const total = toDownload.reduce((sum, d) => sum + d.bytes, 0);
  log(`downloads needed: ${toDownload.length} (${gb(total)})`);
  if (!toDownload.length) return;
  if (!opts.allowDownloads) {
    log("not downloading (pass --allow-downloads to fetch them through the app's download route)");
    return;
  }
  for (const d of toDownload) {
    log(`downloading ${d.id}${d.variant !== "default" ? ` [${d.variant}]` : ""} (${gb(d.bytes)})`);
    const body = { licenseAcknowledged: true };
    if (d.variant && d.variant !== "default") body.variant = d.variant;
    const job = await api("POST", `/api/v1/models/${d.id}/download`, body);
    const { job: done } = await waitJob(job.id);
    if (done.status !== "completed") throw new Error(`download of ${d.id} ended ${done.status}: ${done.error ?? done.message}`);
  }
  await resolveModels({ ...opts, allowDownloads: false });
}

function gb(bytes) {
  return `${(bytes / 1e9).toFixed(2)} GB`;
}

// ---------------------------------------------------------------------------------------------
// Training config
// ---------------------------------------------------------------------------------------------

async function targetFor(modelKey) {
  const targets = await api("GET", "/api/v1/training/targets");
  const list = Array.isArray(targets) ? targets : targets.targets;
  const target = list.find((t) => t.id === MODELS[modelKey].target);
  if (!target) throw new Error(`training target ${MODELS[modelKey].target} not in the catalog`);
  return target;
}

let PRESETS = null;

// The app's training presets, exactly as the Studio receives them (`list_training_presets`).
async function trainingPresets() {
  if (!PRESETS) {
    const registry = await api("GET", "/api/v1/training/presets");
    PRESETS = Array.isArray(registry) ? registry : registry.presets;
    if (!Array.isArray(PRESETS)) throw new Error("GET /api/v1/training/presets returned no preset list");
    const unknown = OPTS.presets.filter((id) => !PRESETS.some((p) => p.id === id));
    if (unknown.length) throw new Error(`--preset ${unknown.join(", ")} not in the API's presets`);
    const targets = new Set(Object.values(MODELS).map((m) => m.target));
    const foreign = PRESETS.filter((p) => OPTS.presets.includes(p.id) && !targets.has(p.targetId));
    if (foreign.length) throw new Error(`--preset ${foreign.map((p) => `${p.id} (${p.targetId})`).join(", ")} is not for ${[...targets].join(" or ")}`);
  }
  return PRESETS;
}

// The preset a model's rows start from: a `--preset` id whose targetId is this model's target,
// else the target's default character preset (recommendedFor "character", `ui.default`), e.g.
// `z_image_turbo_lora.character.adamw8bit.balanced` ("Character balanced").
async function presetFor(modelKey) {
  const targetId = MODELS[modelKey].target;
  const presets = (await trainingPresets()).filter((p) => p.targetId === targetId);
  const chosen = presets.find((p) => OPTS.presets.includes(p.id));
  if (chosen) return chosen;
  const character = presets.filter((p) => (p.recommendedFor ?? []).includes("character"));
  const preset =
    character.find((p) => p.ui?.default === true) ??
    [...character].sort((a, b) => (a.ui?.order ?? Infinity) - (b.ui?.order ?? Infinity))[0];
  if (!preset) throw new Error(`no character preset for target ${targetId}`);
  return preset;
}

// The identical baseline config every row starts from: the preset's config with EVERY key kept
// (optimizer, timestep type/bias, precision, caching, weight decay, `trainingAdapterRepo` /
// `trainingAdapterVersion`, …). Only these are overridden: the seed, the trigger word,
// in-training sampling off, `requestedGpu: auto`, `saveEvery` (`--save-every`, default = steps),
// and whichever of --steps / --rank / --lr / --batch / --resolution was passed explicitly.
// `--rank` keeps the preset's alpha/rank ratio.
function baseConfig(preset, opts) {
  const p = structuredClone(preset.config);
  const config = { ...p, advanced: { ...(p.advanced ?? {}) } };
  config.seed = opts.seed;
  config.triggerWord = opts.trigger;
  config.advanced.sampleEvery = 0;
  config.advanced.requestedGpu = "auto";
  if (opts.steps !== null) config.steps = opts.steps;
  if (opts.rank !== null) {
    config.alpha = Math.max(1, Math.round((opts.rank * p.alpha) / p.rank));
    config.rank = opts.rank;
  }
  if (opts.lr !== null) config.learningRate = opts.lr;
  if (opts.batch !== null) config.batchSize = opts.batch;
  if (opts.resolution !== null) config.resolution = opts.resolution;
  config.saveEvery = opts.saveEvery ?? config.steps;
  return config;
}

// Everything a model's rows share: the target, the preset and the baseline config built from it.
async function modelContext(modelKey, opts) {
  const target = await targetFor(modelKey);
  const preset = await presetFor(modelKey);
  const base = baseConfig(preset, opts);
  return { modelKey, target, preset, base, ladder: bucketLadder(target, base.resolution) };
}

// Resolve a row against the target's support flags: { advanced } to train, or { skip } with why.
function resolveRow(row, target) {
  const limits = target.limits ?? {};
  const missing = row.limits.filter((flag) => limits[flag] !== true);
  if (missing.length) return { skip: `target ${target.id} does not advertise ${missing.join(", ")}` };
  const advanced = { ...row.advanced };
  const dropped = [];
  for (const [flag, keys] of Object.entries(row.optionalLimits ?? {})) {
    if (limits[flag] !== true) {
      keys.forEach((k) => delete advanced[k]);
      dropped.push(flag);
    }
  }
  if (row.optionalLimits && dropped.length === Object.keys(row.optionalLimits).length) {
    return { skip: `target ${target.id} advertises none of ${dropped.join(", ")}` };
  }
  return { advanced, dropped };
}

function rowAdvanced(row, ctx, resolved) {
  return row.bucketLadder ? { ...resolved.advanced, resolutionBuckets: ctx.ladder } : resolved.advanced;
}

function rowLabel(row, ctx) {
  if (!row.bucketLadder) return row.label;
  return row.labelSuffix ? `${ladderLabel(ctx.ladder)} + ${row.labelSuffix}` : ladderLabel(ctx.ladder);
}

function configFor(ctx, advanced) {
  const config = structuredClone(ctx.base);
  config.advanced = { ...config.advanced, ...advanced };
  return config;
}

// A row's identity for resume: the submitted config plus the dataset (and so the caption mode)
// and preset it came from. Rows trained before any of those changed never match, so they are
// retrained rather than silently reused.
function rowKey(ctx, config) {
  const dataset = currentDataset();
  return configKey({ config, captionMode: OPTS.captions, datasetId: dataset?.id ?? null, presetId: ctx.preset.id, presetVersion: ctx.preset.version });
}

// The row's provenance, recorded on its training state and shown in the report header.
function rowProvenance(ctx, config) {
  return {
    captionMode: OPTS.captions,
    datasetId: currentDataset()?.id ?? null,
    presetId: ctx.preset.id,
    presetVersion: ctx.preset.version,
    resolution: config.resolution,
    resolutionBuckets: config.advanced.resolutionBuckets ?? null,
    steps: config.steps,
    rank: config.rank,
    alpha: config.alpha,
    learningRate: config.learningRate,
    optimizer: config.optimizer,
    saveEvery: config.saveEvery,
    trainingAdapterRepo: config.advanced.trainingAdapterRepo ?? null,
    trainingAdapterVersion: config.advanced.trainingAdapterVersion ?? null,
  };
}

function configKey(config) {
  return JSON.stringify(config, Object.keys(flatKeys(config)).sort());
}

function flatKeys(obj, acc = {}) {
  for (const [k, v] of Object.entries(obj)) {
    acc[k] = true;
    if (v && typeof v === "object" && !Array.isArray(v)) flatKeys(v, acc);
    if (Array.isArray(v)) v.forEach((x) => x && typeof x === "object" && flatKeys(x, acc));
  }
  return acc;
}

async function submitTraining(modelKey, rowId, config, dryRun) {
  return api("POST", `/api/v1/projects/${STATE.project.id}/training/jobs`, {
    targetId: MODELS[modelKey].target,
    datasetId: currentDataset().id,
    outputName: `ab-${modelKey}-${rowId}`,
    dryRun,
    config,
  });
}

// Intermediate checkpoints the trainer wrote next to a row's final adapter. Every native trainer
// saves `<stem>-step<NNNNNN>.safetensors` in the plan's output dir every `saveEvery` steps (never at
// the final step, which is the adapter itself) beside `.resume`/`.optim` snapshot siblings, which
// this pattern excludes. Only the final file is registered as a LoRA, so these are imported
// through the app's LoRA import route before a sample can reference them.
function checkpointFiles(rec) {
  if (!rec?.loraPath) return [];
  const dir = path.dirname(rec.loraPath);
  const stem = path.basename(rec.loraPath).replace(/\.safetensors$/, "");
  const escaped = stem.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`^${escaped}-step(\\d+)\\.safetensors$`);
  let names;
  try {
    names = fs.readdirSync(dir);
  } catch {
    return [];
  }
  return names
    .map((name) => ({ name, match: pattern.exec(name) }))
    .filter(({ match }) => match)
    .map(({ name, match }) => ({ step: Number(match[1]), path: path.join(dir, name) }))
    .sort((a, b) => a.step - b.step);
}

// Train one row to completion (or reuse a finished identical run). Returns the state record.
async function trainRow(ctx, row, opts, { stateKey = ctx.modelKey } = {}) {
  const { modelKey, target } = ctx;
  STATE.training[stateKey] ??= {};
  const record = STATE.training[stateKey][row.id] ?? {};
  const resolved = resolveRow(row, target);
  if (resolved.skip) {
    STATE.training[stateKey][row.id] = { status: "skipped", skipReason: resolved.skip, label: rowLabel(row, ctx) };
    saveState();
    log(`  ${row.id}: SKIPPED (${resolved.skip})`);
    return STATE.training[stateKey][row.id];
  }
  const advanced = rowAdvanced(row, ctx, resolved);
  const config = configFor(ctx, advanced);
  const key = rowKey(ctx, config);
  if (record.status === "completed" && record.configKey === key && !opts.force) {
    log(`  ${row.id}: already completed (${record.loraId})`);
    return record;
  }
  // The baseline row of the Z-Image grid is exactly the baseline-timing run: reuse it.
  const timing = STATE.training["baseline-timing"]?.baseline;
  if (row.id === "baseline" && stateKey === "zimage" && timing?.status === "completed" && timing.configKey === key && !opts.force) {
    STATE.training[stateKey][row.id] = { ...timing, reusedFrom: "baseline-timing" };
    saveState();
    log(`  ${row.id}: reusing the baseline-timing run (${timing.loraId})`);
    return STATE.training[stateKey][row.id];
  }
  if (row.needsMasks && !currentDataset()?.masks?.completed) {
    throw new Error(`row ${row.id} needs subject masks for the ${opts.captions}-caption dataset; run --phase masks --confirm-gpu first`);
  }
  await cancelStale(record.jobId);
  const label = rowLabel(row, ctx);
  log(`  ${row.id}: submitting (${label}; ${ctx.preset.id}, ${config.resolution} px, ${config.steps} steps, ${opts.captions} captions)`);
  const job = await submitTraining(modelKey, row.id, config, false);
  const output = job.payload?.plan?.output ?? {};
  const entry = job.payload?.manifestEntry ?? {};
  STATE.training[stateKey][row.id] = {
    status: "running",
    label,
    jobId: job.id,
    configKey: key,
    ...rowProvenance(ctx, config),
    advanced,
    droppedTechniques: resolved.dropped,
    loraId: output.loraId,
    loraPath: output.outputDir && output.fileName ? path.join(output.outputDir, output.fileName) : null,
    family: entry.family ?? target.family ?? null,
    baseModel: entry.baseModel ?? target.baseModel ?? null,
  };
  saveState();
  const { job: done, steps, losses } = await waitJob(job.id);
  const rec = STATE.training[stateKey][row.id];
  rec.status = done.status;
  rec.error = done.error ?? null;
  rec.wallSeconds = wallSeconds(done);
  rec.secondsPerStep = secondsPerStep(steps);
  // The loss of the last `Training step N of M (loss …)` message the poller saw. On an
  // alternating aux-loss row that step may have been an aux-only step, so this is not a
  // diffusion-loss convergence figure.
  rec.finalLoss = losses.at(-1)?.loss ?? null;
  rec.losses = losses;
  rec.completedAt = done.completedAt ?? null;
  rec.checkpoints = checkpointFiles(rec);
  saveState();
  if (done.status !== "completed") {
    throw new Error(`training ${row.id} ended ${done.status}: ${done.error ?? done.message}`);
  }
  log(
    `  ${row.id}: done in ${rec.wallSeconds?.toFixed(0)} s wall` +
      (rec.secondsPerStep ? `, ${rec.secondsPerStep.toFixed(2)} s/step` : "") +
      ` -> ${rec.loraPath}` +
      (rec.checkpoints.length ? ` (+${rec.checkpoints.length} checkpoints)` : ""),
  );
  return rec;
}

// ---------------------------------------------------------------------------------------------
// GPU phases
// ---------------------------------------------------------------------------------------------

async function phaseMasks() {
  requireSetup();
  const dataset = currentDataset();
  await cancelStale(dataset.masks?.jobId);
  const route = `/api/v1/projects/${STATE.project.id}/training/datasets/${dataset.id}`;
  const job = await api("POST", `${route}/subject-mask-jobs`, {});
  dataset.masks = { jobId: job.id, completed: false };
  saveState();
  const { job: done } = await waitJob(job.id);
  const masks = await api("GET", `${route}/subject-masks`).catch(() => null);
  dataset.masks = { jobId: job.id, completed: done.status === "completed", status: done.status, wallSeconds: wallSeconds(done), masks: summarizeMasks(masks) };
  saveState();
  if (done.status !== "completed") throw new Error(`subject-mask job ended ${done.status}: ${done.error ?? done.message}`);
  log(`subject masks done for the ${OPTS.captions}-caption dataset (${JSON.stringify(dataset.masks.masks)})`);
}

function summarizeMasks(masks) {
  if (!masks) return null;
  const list = Array.isArray(masks) ? masks : masks.items ?? masks.masks ?? null;
  return Array.isArray(list) ? { count: list.length } : { keys: Object.keys(masks) };
}

async function phaseBaselineTiming(opts) {
  requireSetup();
  const ctx = await modelContext("zimage", opts);
  const row = techniqueRows("zimage", opts).find((r) => r.id === "baseline");
  const rec = await trainRow(ctx, row, opts, { stateKey: "baseline-timing" });
  log(`baseline timing: ${rec.wallSeconds?.toFixed(0)} s wall, ${rec.secondsPerStep?.toFixed(3) ?? "?"} s/step over ${rec.steps} steps at ${rec.resolution} px`);
}

async function phaseTrain(opts, modelKey) {
  requireSetup();
  const ctx = await modelContext(modelKey, opts);
  const rows = selectedRows(modelKey, opts);
  log(`training ${rows.length} ${MODELS[modelKey].label} rows sequentially from ${ctx.preset.id}`);
  for (const row of rows) await trainRow(ctx, row, opts);
}

function selectedRows(modelKey, opts) {
  const rows = techniqueRows(modelKey, opts);
  if (!opts.rows) return rows;
  const unknown = opts.rows.filter((id) => !rows.some((r) => r.id === id));
  if (unknown.length) throw new Error(`unknown --rows ${unknown.join(", ")}; known: ${rows.map((r) => r.id).join(", ")}`);
  return rows.filter((r) => opts.rows.includes(r.id));
}

function genSize(modelKey, opts) {
  const m = MODELS[modelKey];
  return opts.genSize ? { width: opts.genSize, height: opts.genSize } : { width: m.gen.width, height: m.gen.height };
}

function sampleRequest(modelKey, opts, promptText, seed, loraId) {
  const m = MODELS[modelKey];
  return {
    projectId: STATE.project.id,
    prompt: promptText,
    model: m.baseModel,
    count: 1,
    seed,
    ...genSize(modelKey, opts),
    loras: loraId ? [{ id: loraId, weight: opts.loraWeight }] : [],
    advanced: { ...m.gen.advanced },
  };
}

// What a row's samples were rendered from. A change (retrained adapter, other LoRA weight or
// render size) invalidates the row's existing samples instead of mixing renders in one grid row.
function renderKey(modelKey, opts, loraId) {
  return JSON.stringify({ loraId: loraId ?? null, weight: loraId ? opts.loraWeight : null, ...genSize(modelKey, opts), advanced: MODELS[modelKey].gen.advanced });
}

function imagePathsIn(value, acc = []) {
  if (typeof value === "string") {
    if (/\.(png|jpe?g|webp)$/i.test(value)) acc.push(value);
  } else if (Array.isArray(value)) {
    value.forEach((v) => imagePathsIn(v, acc));
  } else if (value && typeof value === "object") {
    Object.values(value).forEach((v) => imagePathsIn(v, acc));
  }
  return acc;
}

function checkpointRowId(rowId, step) {
  return `${rowId}@${step}`;
}

// Register one intermediate checkpoint as a project LoRA through the app's own import route
// (`POST /api/v1/loras/import`, a local `sourcePath` under the project's loras dir, run by the
// API's in-process utility worker), so the image API can reference it by id. The id carries the
// training job, so a retrained row never resolves to an older run's checkpoint.
async function ensureCheckpointLora(modelKey, rowId, trained, ckpt) {
  const known = STATE.checkpointLoras[ckpt.path];
  if (known && known.trainingJobId === trained.jobId && !OPTS.force) {
    const loras = await api("GET", `/api/v1/loras?projectId=${encodeURIComponent(STATE.project.id)}`);
    if (loras.some((l) => l.id === known.loraId && l.installState === "installed")) return known.loraId;
  }
  let { family, baseModel } = trained;
  if (!family || !baseModel) {
    const job = await api("GET", `/api/v1/jobs/${trained.jobId}`);
    family ??= job.payload?.manifestEntry?.family ?? null;
    baseModel ??= job.payload?.manifestEntry?.baseModel ?? null;
  }
  const loraId = `ab_${modelKey}_${rowId}_${trained.jobId.replace(/^job_/, "").slice(0, 12)}_step${String(ckpt.step).padStart(6, "0")}`;
  log(`  importing checkpoint ${path.basename(ckpt.path)} as LoRA ${loraId}`);
  const job = await api("POST", "/api/v1/loras/import", {
    loraId,
    name: `ab ${modelKey} ${rowId} @ step ${ckpt.step}`,
    sourcePath: ckpt.path,
    scope: "project",
    projectId: STATE.project.id,
    family,
    baseModel,
    triggerWords: [OPTS.trigger],
    notes: `epic-2123 A/B intermediate checkpoint of training job ${trained.jobId}`,
  });
  const { job: done } = await waitJob(job.id);
  if (done.status !== "completed") throw new Error(`checkpoint import ${loraId} ended ${done.status}: ${done.error ?? done.message}`);
  STATE.checkpointLoras[ckpt.path] = { loraId, trainingJobId: trained.jobId, step: ckpt.step, importJobId: job.id };
  saveState();
  return loraId;
}

// Render the fixed prompt grid for one row (or checkpoint sub-row) with one adapter (or none).
async function renderRow(opts, modelKey, rowId, label, loraId) {
  const dir = path.join(opts.root, "samples", modelKey, rowId);
  fs.mkdirSync(dir, { recursive: true });
  const key = renderKey(modelKey, opts, loraId);
  let record = STATE.samples[modelKey][rowId];
  if (record && record.renderKey !== key) {
    log(`  ${rowId}: render inputs changed (adapter, LoRA weight or size); re-rendering`);
    record = null;
  }
  record = STATE.samples[modelKey][rowId] = record ?? { files: {}, renderKey: key };
  record.label = label;
  record.loraId = loraId ?? null;
  for (const [pid, template] of PROMPTS) {
    for (const seed of opts.seeds) {
      const name = `${pid}_s${seed}`;
      const dest = path.join(dir, `${name}.png`);
      if (record.files[name] && fs.existsSync(dest) && !opts.force) continue;
      const promptText = template.replaceAll("{t}", opts.trigger);
      await cancelStale(record.pendingJobId);
      const job = await api("POST", "/api/v1/image/jobs", sampleRequest(modelKey, opts, promptText, seed, loraId));
      record.pendingJobId = job.id;
      saveState();
      log(`  ${rowId} ${name}`);
      const { job: done } = await waitJob(job.id);
      record.pendingJobId = null;
      if (done.status !== "completed") throw new Error(`sample ${rowId}/${name} ended ${done.status}: ${done.error ?? done.message}`);
      // The worker records each written image as an asset fact whose `mediaPath` is relative to
      // the project directory; fall back to any image-looking path in the result.
      const writes = done.result?.assetWrites ?? [];
      const found = writes.find((w) => typeof w?.mediaPath === "string")?.mediaPath ?? imagePathsIn(done.result)[0];
      if (!found) throw new Error(`sample ${rowId}/${name}: no image path in the job result`);
      const source = path.isAbsolute(found) ? found : path.join(STATE.project.path, found);
      fs.copyFileSync(source, dest);
      record.files[name] = { prompt: promptText, seed, source, jobId: job.id };
      saveState();
    }
  }
  fs.writeFileSync(
    path.join(dir, "prompts.json"),
    `${JSON.stringify(Object.fromEntries(PROMPTS.map(([pid, t]) => [pid, t.replaceAll("{t}", opts.trigger)])), null, 2)}\n`,
  );
}

async function phaseSamples(opts, modelKey) {
  requireSetup();
  const rows = [{ id: "base", label: "Base model (no LoRA)" }, ...selectedRows(modelKey, opts)];
  STATE.samples[modelKey] ??= {};
  for (const row of rows) {
    const trained = row.id === "base" ? { status: "completed" } : STATE.training[modelKey]?.[row.id];
    if (!trained || trained.status === "skipped") {
      log(`  ${row.id}: no adapter (${trained?.skipReason ?? "not trained"}); skipping samples`);
      continue;
    }
    if (trained.status !== "completed") {
      log(`  ${row.id}: training not completed (${trained.status}); skipping samples`);
      continue;
    }
    if (opts.checkpoints && row.id !== "base") {
      const checkpoints = checkpointFiles(trained);
      if (!checkpoints.length) log(`  ${row.id}: no intermediate checkpoints on disk (trained with saveEvery ${trained.saveEvery ?? "?"})`);
      for (const ckpt of checkpoints) {
        const loraId = await ensureCheckpointLora(modelKey, row.id, trained, ckpt);
        await renderRow(opts, modelKey, checkpointRowId(row.id, ckpt.step), `${trained.label ?? row.label} @ step ${ckpt.step}`, loraId);
      }
    }
    const label = row.id === "base" ? row.label : `${trained.label ?? row.label}${trained.steps ? ` @ step ${trained.steps}` : ""}`;
    await renderRow(opts, modelKey, row.id, label, trained.loraId);
  }
}

// ---------------------------------------------------------------------------------------------
// Eval: lora_eval_harness (MLX: CLIP + SCRFD/ArcFace) + LoRA stable rank (CPU)
// ---------------------------------------------------------------------------------------------

async function phaseEval(opts, modelKey) {
  // Stable rank first: pure CPU (safetensors parse + power iteration), always safe.
  STATE.rank[modelKey] ??= {};
  for (const [rowId, rec] of Object.entries(STATE.training[modelKey] ?? {})) {
    if (rec.status !== "completed" || !rec.loraPath) continue;
    const files = [[rowId, rec.loraPath], ...checkpointFiles(rec).map((c) => [checkpointRowId(rowId, c.step), c.path])];
    for (const [id, file] of files) {
      STATE.rank[modelKey][id] = loraStableRank(file);
      log(`  ${id}: stable rank ${STATE.rank[modelKey][id].mean.toFixed(3)} over ${STATE.rank[modelKey][id].modules} modules`);
    }
  }
  saveState();

  // The perceptual instrument loads CLIP ViT-L/14 + SCRFD/ArcFace on MLX, so it is a GPU step.
  if (!opts.confirmGpu) {
    log("skipping lora_eval_harness: it runs MLX models; rerun with --confirm-gpu to score the samples");
    return;
  }
  const faceBundle = hfSnapshot("SceneWorks/instantid-mlx");
  if (!faceBundle) throw new Error("SceneWorks/instantid-mlx is not in the HF cache (install instantid_face_stack)");
  const env = { ...process.env, ...prebuiltMlxEnv(opts.evalRelease ? "Release" : "Debug") };
  // The independent identity metric: set FACE_R50_WEIGHTS only for a resolved file, and clear any
  // inherited value otherwise, so whether the metric is scored is decided here alone.
  const faceR50 = resolveFaceR50(opts);
  delete env.FACE_R50_WEIGHTS;
  if (faceR50) env.FACE_R50_WEIGHTS = faceR50.path;
  log(faceR50 ? `w600k_r50 identity metric: ${faceR50.path} (sha256 ${faceR50.sha256.slice(0, 12)})` : "w600k_r50 identity metric: off");
  STATE.eval[modelKey] ??= {};
  STATE.evalKeys[modelKey] ??= {};
  const evalDir = path.join(opts.root, "eval", modelKey);
  fs.mkdirSync(evalDir, { recursive: true });
  for (const [rowId, rec] of Object.entries(STATE.samples[modelKey] ?? {})) {
    const genDir = path.join(opts.root, "samples", modelKey, rowId);
    if (!Object.keys(rec.files ?? {}).length) continue;
    const out = path.join(evalDir, `${rowId}.json`);
    // Score only the current prompt × seed grid: a row folder can still hold images of a prompt
    // that has since been dropped (rendered by an earlier run), and the instrument scores every
    // image in GEN_DIR. Move those aside rather than delete them.
    const wanted = new Set(PROMPTS.flatMap(([pid]) => opts.seeds.map((seed) => `${pid}_s${seed}.png`)));
    for (const file of fs.readdirSync(genDir)) {
      if (!/\.(png|jpe?g|webp)$/i.test(file) || wanted.has(file)) continue;
      fs.mkdirSync(path.join(genDir, "_stale"), { recursive: true });
      fs.renameSync(path.join(genDir, file), path.join(genDir, "_stale", file));
      log(`  ${rowId}: moved stale ${file} (not in the current prompt grid) to _stale/`);
    }
    // Re-score when the row's samples were re-rendered from different inputs, or the prompt grid
    // changed, since its last score.
    // Re-score too when the independent identity network is switched on, off or swapped.
    const evalKey = `${rec.renderKey}|${[...wanted].sort().join(",")}${faceR50 ? `|r50=${faceR50.sha256}` : ""}`;
    const scoredCurrent = STATE.evalKeys[modelKey][rowId] === evalKey;
    if (STATE.eval[modelKey][rowId] && scoredCurrent && fs.existsSync(out) && !opts.force) {
      log(`  ${rowId}: already scored`);
      continue;
    }
    log(`  ${rowId}: scoring ${Object.keys(rec.files).length} images`);
    const args = ["test", "-p", "sceneworks-worker", "--lib"];
    if (opts.evalRelease) args.push("--release");
    args.push("--", "--ignored", "--nocapture", "--exact", "lora_eval_harness::harness::eval_lora_outputs");
    const res = spawnSync("cargo", args, {
      cwd: REPO,
      stdio: "inherit",
      env: {
        ...env,
        REF_DIR: currentDataset()?.source ?? opts.dataset,
        GEN_DIR: genDir,
        PROMPTS_JSON: path.join(genDir, "prompts.json"),
        EVAL_LABEL: `${modelKey}/${rowId}`,
        EVAL_OUT: out,
        SCENEWORKS_INSTANTID_WEIGHTS: faceBundle,
        RUST_TEST_THREADS: "1",
      },
    });
    if (res.status !== 0) throw new Error(`lora_eval_harness failed for ${rowId} (${res.status})`);
    STATE.eval[modelKey][rowId] = JSON.parse(fs.readFileSync(out, "utf8")).aggregates;
    STATE.evalKeys[modelKey][rowId] = evalKey;
    saveState();
  }
}

// The w600k_r50 weights for the independent identity metric, or null when it is off. An explicit
// `--face-r50` path must exist; the default path is used only when present.
function resolveFaceR50(opts) {
  if (opts.noFaceR50) return null;
  const file = opts.faceR50 ?? opts.faceR50Default;
  if (!fs.existsSync(file)) {
    if (opts.faceR50) throw new Error(`--face-r50 ${file} does not exist`);
    return null;
  }
  const sha256 = crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
  return { path: file, sha256 };
}

function hfHubDir() {
  if (process.env.HF_HUB_CACHE) return process.env.HF_HUB_CACHE;
  if (OPTS.hfHome) return path.join(OPTS.hfHome, "hub");
  if (process.env.HF_HOME) return path.join(process.env.HF_HOME, "hub");
  return path.join(os.homedir(), ".cache", "huggingface", "hub");
}

function hfSnapshot(repo) {
  const dir = path.join(hfHubDir(), `models--${repo.replaceAll("/", "--")}`, "snapshots");
  try {
    const snap = fs.readdirSync(dir).find((s) => fs.statSync(path.join(dir, s)).isDirectory());
    return snap ? path.join(dir, snap) : null;
  } catch {
    return null;
  }
}

// --- safetensors + stable rank -----------------------------------------------------------------

function readSafetensors(file) {
  const buf = fs.readFileSync(file);
  const headerLen = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString("utf8"));
  const base = 8 + headerLen;
  const tensors = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    tensors[name] = {
      shape: info.shape,
      get data() {
        const [start, end] = info.data_offsets;
        return decodeTensor(buf.subarray(base + start, base + end), info.dtype);
      },
    };
  }
  return { tensors, metadata: header.__metadata__ ?? {} };
}

function decodeTensor(bytes, dtype) {
  const aligned = Buffer.from(bytes); // own, aligned copy
  if (dtype === "F32") return new Float32Array(aligned.buffer, aligned.byteOffset, aligned.length / 4);
  if (dtype === "F64") return Float32Array.from(new Float64Array(aligned.buffer, aligned.byteOffset, aligned.length / 8));
  const u16 = new Uint16Array(aligned.buffer, aligned.byteOffset, aligned.length / 2);
  const out = new Float32Array(u16.length);
  if (dtype === "BF16") {
    const u32 = new Uint32Array(out.buffer);
    for (let i = 0; i < u16.length; i += 1) u32[i] = u16[i] << 16;
    return out;
  }
  if (dtype === "F16") {
    for (let i = 0; i < u16.length; i += 1) out[i] = halfToFloat(u16[i]);
    return out;
  }
  throw new Error(`unsupported safetensors dtype ${dtype}`);
}

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

// Pair every LoRA module's down/up factors. PEFT layout (what the native trainers write):
// `<path>.lora_A.weight` [r, in] and `<path>.lora_B.weight` [out, r]; kohya `lora_down`/`lora_up`
// is accepted too.
function loraPairs(tensors) {
  const pairs = [];
  for (const name of Object.keys(tensors)) {
    const m = /^(.*)\.(lora_A|lora_down)(\.weight)?$/.exec(name);
    if (!m) continue;
    const upName = `${m[1]}.${m[2] === "lora_A" ? "lora_B" : "lora_up"}${m[3] ?? ""}`;
    if (tensors[upName]) pairs.push({ module: m[1], a: tensors[name], b: tensors[upName] });
  }
  return pairs;
}

// Stable rank of ΔW = B·A: ‖ΔW‖_F² / ‖ΔW‖_2². ‖ΔW‖_F² = trace((BᵀB)(AAᵀ)) is exact over the
// r×r Gram matrices; σ_max² comes from power iteration on ΔWᵀΔW applied implicitly (v → A v →
// B(Av) → Bᵀ… → Aᵀ…), so ΔW (out×in) is never materialized. Scale-invariant, so the stored
// alpha/rank factor does not matter.
function stableRankOf(A, B, r, inDim, outDim) {
  const gA = new Float64Array(r * r);
  for (let i = 0; i < r; i += 1) {
    for (let j = i; j < r; j += 1) {
      let s = 0;
      for (let k = 0; k < inDim; k += 1) s += A[i * inDim + k] * A[j * inDim + k];
      gA[i * r + j] = s;
      gA[j * r + i] = s;
    }
  }
  const gB = new Float64Array(r * r);
  for (let i = 0; i < r; i += 1) {
    for (let j = i; j < r; j += 1) {
      let s = 0;
      for (let k = 0; k < outDim; k += 1) s += B[k * r + i] * B[k * r + j];
      gB[i * r + j] = s;
      gB[j * r + i] = s;
    }
  }
  let fro2 = 0;
  for (let i = 0; i < r; i += 1) for (let j = 0; j < r; j += 1) fro2 += gB[i * r + j] * gA[j * r + i];
  if (!(fro2 > 0)) return null; // an all-zero delta (e.g. B still at its zero init)
  // Power iteration on ΔWᵀΔW = Aᵀ (BᵀB) A, applied implicitly to a unit v ∈ R^in:
  // w = A v (r), z = (BᵀB) w (r), next = Aᵀ z (in). The Rayleigh quotient vᵀΔWᵀΔWv = wᵀz
  // converges to σ_max². Deterministic start vector so reruns agree.
  let v = new Float64Array(inDim);
  let seed = 12345;
  for (let k = 0; k < inDim; k += 1) {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    v[k] = seed / 2147483648 - 0.5;
  }
  normalize(v);
  let lambda = 0;
  for (let iter = 0; iter < 2000; iter += 1) {
    const w = new Float64Array(r);
    for (let i = 0; i < r; i += 1) {
      let s = 0;
      for (let k = 0; k < inDim; k += 1) s += A[i * inDim + k] * v[k];
      w[i] = s;
    }
    const z = new Float64Array(r);
    for (let i = 0; i < r; i += 1) {
      let s = 0;
      for (let j = 0; j < r; j += 1) s += gB[i * r + j] * w[j];
      z[i] = s;
    }
    // Rayleigh quotient vᵀ ΔWᵀΔW v = wᵀ (BᵀB) w for unit v.
    let rq = 0;
    for (let i = 0; i < r; i += 1) rq += w[i] * z[i];
    const next = new Float64Array(inDim);
    for (let i = 0; i < r; i += 1) {
      const zi = z[i];
      if (zi === 0) continue;
      for (let k = 0; k < inDim; k += 1) next[k] += A[i * inDim + k] * zi;
    }
    const norm = normalize(next);
    v = next;
    if (norm === 0) break;
    if (iter > 5 && Math.abs(rq - lambda) <= 1e-10 * Math.abs(rq)) {
      lambda = rq;
      break;
    }
    lambda = rq;
  }
  return lambda > 0 ? fro2 / lambda : null;
}

function normalize(v) {
  let s = 0;
  for (let k = 0; k < v.length; k += 1) s += v[k] * v[k];
  const n = Math.sqrt(s);
  if (n > 0) for (let k = 0; k < v.length; k += 1) v[k] /= n;
  return n;
}

function loraStableRank(file) {
  const { tensors } = readSafetensors(file);
  const pairs = loraPairs(tensors);
  if (!pairs.length) throw new Error(`${file}: no lora_A/lora_B (or lora_down/lora_up) pairs`);
  const values = [];
  for (const { a, b } of pairs) {
    // Orient to A: [r, in], B: [out, r] (accept the transposed MLX-native layout too).
    let [a0, a1] = a.shape;
    let [b0, b1] = b.shape;
    let A = a.data;
    let B = b.data;
    if (a0 !== b1 && a1 === b0) {
      A = transpose(A, a0, a1);
      B = transpose(B, b0, b1);
      [a0, a1] = [a1, a0];
      [b0, b1] = [b1, b0];
    }
    if (a0 !== b1) throw new Error(`${file}: cannot pair shapes ${a.shape} / ${b.shape}`);
    const sr = stableRankOf(A, B, a0, a1, b0);
    if (sr !== null) values.push(sr);
  }
  values.sort((x, y) => x - y);
  return {
    method: "node power iteration (σ_max) + exact Gram trace (‖ΔW‖_F), mean over modules",
    modules: values.length,
    pairs: pairs.length,
    mean: values.reduce((s, x) => s + x, 0) / values.length,
    median: values[Math.floor(values.length / 2)],
    min: values[0],
    max: values.at(-1),
  };
}

function transpose(data, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i += 1) for (let j = 0; j < cols; j += 1) out[j * rows + i] = data[i * cols + j];
  return out;
}

// Self-test of the stable-rank math on synthetic adapters with known answers: rank-1 ΔW → 1;
// orthonormal A rows × orthonormal B columns with equal scale → r; singular values (3,1) → 10/9.
function stableRankSelfTest(dir) {
  const write = (file, tensors) => {
    const header = {};
    let offset = 0;
    const chunks = [];
    for (const [name, { shape, values }] of Object.entries(tensors)) {
      const bytes = Buffer.from(Float32Array.from(values).buffer);
      header[name] = { dtype: "F32", shape, data_offsets: [offset, offset + bytes.length] };
      offset += bytes.length;
      chunks.push(bytes);
    }
    const h = Buffer.from(JSON.stringify(header));
    const len = Buffer.alloc(8);
    len.writeBigUInt64LE(BigInt(h.length));
    fs.writeFileSync(file, Buffer.concat([len, h, ...chunks]));
  };
  const r = 4;
  const inDim = 6;
  const outDim = 5;
  const eye = (rows, cols, scale = 1) =>
    Array.from({ length: rows * cols }, (_, i) => (Math.floor(i / cols) === i % cols ? scale : 0));
  // A with rows [3,0,…] and [0,1,0,…] (others zero) times B = I gives singular values 3 and 1.
  const svA = new Array(r * inDim).fill(0);
  svA[0] = 3;
  svA[inDim + 1] = 1;
  const cases = [
    // A row 0 = 1..6, B column 0 = 1..5, everything else zero: ΔW is an outer product.
    { name: "rank1", a: [1, 2, 3, 4, 5, 6, ...new Array((r - 1) * inDim).fill(0)], b: [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0], expect: 1 },
    { name: "isotropic", a: eye(r, inDim), b: eye(outDim, r, 2), expect: r },
    { name: "sv_3_1", a: svA, b: eye(outDim, r), expect: 10 / 9 },
  ];
  const results = [];
  for (const c of cases) {
    const file = path.join(dir, `selftest-${c.name}.safetensors`);
    write(file, {
      [`m.${c.name}.lora_A.weight`]: { shape: [r, inDim], values: c.a },
      [`m.${c.name}.lora_B.weight`]: { shape: [outDim, r], values: c.b },
    });
    const got = loraStableRank(file).mean;
    results.push({ case: c.name, expect: c.expect, got, ok: Math.abs(got - c.expect) < 1e-6 });
  }
  return results;
}

// ---------------------------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------------------------

const METRICS = [
  ["identity_cosine_mean", "ArcFace likeness", 3],
  // Independent face-recognition network (w600k_r50) that no training technique optimises.
  ["identity_r50_cosine_mean", "R50 likeness", 3],
  ["prompt_adherence_mean", "CLIP adherence", 3],
  ["same_prompt_spread", "Same-prompt spread", 3],
  ["output_spread", "Output spread", 3],
  ["sharpness_mean", "Sharpness", 0],
  ["face_detect_rate", "Face detect", 2],
];

// Report rows for one model: the base model, then per technique row its intermediate-checkpoint
// sub-rows (`<row>@<step>`, ascending) followed by the row's final adapter. Sub-rows come from
// whatever was sampled or ranked, so they appear once `samples --checkpoints` or `eval` ran.
function reportRows(modelKey, opts) {
  const training = STATE.training[modelKey] ?? {};
  const sampled = Object.keys(STATE.samples[modelKey] ?? {});
  const ranked = Object.keys(STATE.rank[modelKey] ?? {});
  const rows = [{ id: "base", label: "Base model (no LoRA)", parent: null }];
  for (const row of techniqueRows(modelKey, opts)) {
    const t = training[row.id];
    const steps = new Set();
    for (const id of [...sampled, ...ranked]) {
      const m = /^(.+)@(\d+)$/.exec(id);
      if (m && m[1] === row.id) steps.add(Number(m[2]));
    }
    for (const step of [...steps].sort((a, b) => a - b)) {
      rows.push({ id: checkpointRowId(row.id, step), label: `${t?.label ?? row.label} @ step ${step}`, parent: row.id, step });
    }
    const final = steps.size && t?.steps ? ` @ step ${t.steps} (final)` : "";
    rows.push({ id: row.id, label: `${t?.label ?? row.label}${final}`, parent: null, step: t?.steps ?? null });
  }
  return rows;
}

function phaseReport(opts) {
  const reportDir = path.join(opts.root, "report");
  fs.mkdirSync(reportDir, { recursive: true });
  const dataset = currentDataset();
  const results = {
    generatedAt: new Date().toISOString(),
    settings: {
      seed: opts.seed,
      captionMode: opts.captions,
      trigger: dataset?.trigger ?? opts.trigger,
      dataset: dataset ?? null,
      datasets: STATE.datasets,
      sampleSeeds: opts.seeds,
      prompts: Object.fromEntries(PROMPTS),
      loraWeight: opts.loraWeight,
      genSize: Object.fromEntries(Object.keys(MODELS).map((k) => [k, genSize(k, opts)])),
    },
    models: STATE.models ?? null,
    baselineTiming: stripLosses(STATE.training["baseline-timing"]?.baseline ?? null),
    grids: {},
  };
  const md = [`# Epic 2123 real-weight A/B`, "", `Generated ${results.generatedAt}.`, ""];
  md.push(
    `Seed ${opts.seed}; trigger \`${results.settings.trigger}\`; ${PROMPTS.length} prompts × ${opts.seeds.length} seeds per row ` +
      `at LoRA weight ${opts.loraWeight}.`,
    "",
  );
  if (results.baselineTiming) {
    const t = results.baselineTiming;
    md.push(
      `Baseline timing (Z-Image, ${t.resolution ?? "?"} px, ${t.steps ?? "?"} steps): ${fmt(t.wallSeconds, 0)} s wall, ` +
        `${fmt(t.secondsPerStep, 3)} s/step.`,
      "",
    );
  }
  for (const modelKey of Object.keys(MODELS)) {
    const training = STATE.training[modelKey] ?? {};
    const samples = STATE.samples[modelKey] ?? {};
    if (!Object.keys(training).length && !Object.keys(samples).length) continue;
    const rows = reportRows(modelKey, opts);
    const evalFor = (id) => {
      // A score is shown only for the samples it was computed on. The eval key is the render key
      // followed by `|`-separated grid (and identity-network) parts.
      const key = STATE.evalKeys?.[modelKey]?.[id];
      const renderKey = samples[id]?.renderKey;
      const current = key === undefined || (renderKey !== undefined && (key === renderKey || key.startsWith(`${renderKey}|`)));
      return current ? STATE.eval[modelKey]?.[id] ?? null : null;
    };
    const table = rows.map((row) => {
      const t = row.id === "base" ? null : training[row.parent ?? row.id] ?? null;
      return {
        id: row.id,
        label: row.label,
        checkpointOf: row.parent,
        step: row.step ?? null,
        status: row.id === "base" ? "n/a" : t?.status ?? "pending",
        skipReason: t?.skipReason ?? null,
        droppedTechniques: t?.droppedTechniques ?? [],
        captionMode: t?.captionMode ?? null,
        datasetId: t?.datasetId ?? null,
        presetId: t?.presetId ?? null,
        resolution: t?.resolution ?? null,
        resolutionBuckets: t?.resolutionBuckets ?? null,
        steps: t?.steps ?? null,
        rank: t?.rank ?? null,
        alpha: t?.alpha ?? null,
        learningRate: t?.learningRate ?? null,
        trainingAdapterVersion: t?.trainingAdapterVersion ?? null,
        wallSeconds: row.parent ? null : t?.wallSeconds ?? null,
        secondsPerStep: row.parent ? null : t?.secondsPerStep ?? null,
        // The last logged `Training step N of M (loss …)`; on an alternating aux-loss row it may
        // be an aux-only step's loss.
        finalLoss: row.parent ? null : t?.finalLoss ?? null,
        loraPath: row.parent ? checkpointFiles(t).find((c) => c.step === row.step)?.path ?? null : t?.loraPath ?? null,
        loraId: samples[row.id]?.loraId ?? (row.parent ? null : t?.loraId ?? null),
        sampleRender: samples[row.id]?.renderKey ? JSON.parse(samples[row.id].renderKey) : null,
        stableRank: STATE.rank[modelKey]?.[row.id] ?? null,
        eval: evalFor(row.id),
      };
    });
    results.grids[modelKey] = table;
    const baseline = table.find((r) => r.id === "baseline");
    md.push(`## ${MODELS[modelKey].label}`, "");
    // Header: what each trained row actually ran with.
    const cfgHead = ["Row", "Captions", "Preset", "Resolution", "Steps", "Rank/alpha", "LR", "Training adapter"];
    md.push(`| ${cfgHead.join(" | ")} |`, `| ${cfgHead.map(() => "---").join(" | ")} |`);
    for (const r of table) {
      if (r.id === "base" || r.checkpointOf || r.status === "skipped" || r.status === "pending") continue;
      const res = r.resolutionBuckets ? r.resolutionBuckets.map((b) => `${b.resolution}×${b.repeats}`).join("/") : r.resolution ?? "–";
      md.push(
        `| ${[r.label, r.captionMode ?? "–", r.presetId ? `\`${r.presetId}\`` : "–", res, r.steps ?? "–", r.rank ? `${r.rank}/${r.alpha}` : "–", r.learningRate ?? "–", r.presetId ? r.trainingAdapterVersion ?? "none" : "–"].join(" | ")} |`,
      );
    }
    md.push("");
    const head = ["Row", "Status", "s/step", "Stable rank", ...METRICS.map(([, label]) => label)];
    md.push(`| ${head.join(" | ")} |`, `| ${head.map(() => "---").join(" | ")} |`);
    for (const r of table) {
      const cells = [
        r.label,
        r.status === "skipped" ? `skipped: ${r.skipReason}` : r.checkpointOf ? "checkpoint" : r.status,
        withDelta(r.secondsPerStep, baseline?.secondsPerStep, 2, r.id),
        withDelta(r.stableRank?.mean, baseline?.stableRank?.mean, 2, r.id),
        ...METRICS.map(([key, , digits]) => withDelta(r.eval?.[key], baseline?.eval?.[key], digits, r.id)),
      ];
      md.push(`| ${cells.join(" | ")} |`);
    }
    md.push(
      "",
      "Deltas are against the Baseline row's final adapter. Stable rank = ‖BA‖_F² / ‖BA‖_2², mean over adapter modules. " +
        "`row @ step N` rows are intermediate checkpoints of that row's run.",
      "",
    );
    const grid = makeGrid(opts, modelKey, rows, reportDir);
    if (grid) md.push(`![${modelKey} grid](${path.basename(grid)})`, "");
  }
  fs.writeFileSync(path.join(reportDir, "results.json"), `${JSON.stringify(results, null, 2)}\n`);
  fs.writeFileSync(path.join(reportDir, "report.md"), `${md.join("\n")}\n`);
  log(`report written to ${reportDir} (results.json, report.md${Object.keys(results.grids).length ? ", grid PNGs" : ""})`);
  return results;
}

function stripLosses(rec) {
  if (!rec) return rec;
  const { losses, ...rest } = rec;
  return { ...rest, lossPoints: losses?.length ?? 0 };
}

function fmt(v, digits) {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "–";
}

function withDelta(v, base, digits, rowId) {
  if (typeof v !== "number" || !Number.isFinite(v)) return "–";
  if (rowId === "baseline" || rowId === "base" || typeof base !== "number" || !Number.isFinite(base)) return v.toFixed(digits);
  const d = v - base;
  return `${v.toFixed(digits)} (${d >= 0 ? "+" : ""}${d.toFixed(digits)})`;
}

// Stitch the sample grid with Python PIL (rows = technique, columns = prompts at the first seed).
const GRID_PY = String.raw`
import json, sys
from PIL import Image, ImageDraw
spec = json.load(open(sys.argv[1]))
cell, label_w, head_h = spec["cell"], spec["labelWidth"], 28
rows, cols = spec["rows"], spec["columns"]
canvas = Image.new("RGB", (label_w + cell * len(cols), head_h + cell * len(rows)), "white")
draw = ImageDraw.Draw(canvas)
for c, name in enumerate(cols):
    draw.text((label_w + c * cell + 4, 8), name, fill="black")
for r, row in enumerate(rows):
    y = head_h + r * cell
    draw.multiline_text((4, y + 4), row["label"], fill="black")
    for c, file in enumerate(row["files"]):
        if not file:
            continue
        im = Image.open(file).convert("RGB")
        im.thumbnail((cell, cell))
        canvas.paste(im, (label_w + c * cell + (cell - im.width) // 2, y + (cell - im.height) // 2))
canvas.save(spec["out"])
`;

function makeGrid(opts, modelKey, rows, reportDir) {
  const seed = opts.seeds[0];
  const gridRows = [];
  for (const row of rows) {
    const dir = path.join(opts.root, "samples", modelKey, row.id);
    const files = PROMPTS.map(([pid]) => {
      const f = path.join(dir, `${pid}_s${seed}.png`);
      return fs.existsSync(f) ? f : null;
    });
    if (files.some(Boolean)) gridRows.push({ label: wrap(row.label, 22), files });
  }
  if (!gridRows.length) return null;
  const out = path.join(reportDir, `grid-${modelKey}.png`);
  const specPath = path.join(reportDir, `grid-${modelKey}.json`);
  fs.writeFileSync(specPath, JSON.stringify({ cell: 256, labelWidth: 180, columns: PROMPTS.map(([p]) => p), rows: gridRows, out }));
  const res = spawnSync("python3", ["-c", GRID_PY, specPath], { encoding: "utf8" });
  if (res.status !== 0) {
    log(`grid for ${modelKey} failed (python3 + PIL needed): ${res.stderr.trim()}`);
    return null;
  }
  return out;
}

function wrap(text, width) {
  const words = text.split(" ");
  const lines = [""];
  for (const w of words) {
    if ((lines.at(-1) + " " + w).trim().length > width) lines.push(w);
    else lines[lines.length - 1] = `${lines.at(-1)} ${w}`.trim();
  }
  return lines.join("\n");
}

// ---------------------------------------------------------------------------------------------
// Dry run: setup + every phase with no GPU; validates every row's config.advanced at the API
// ---------------------------------------------------------------------------------------------

// A tiny PEFT-layout safetensors file, standing in for a trainer checkpoint in the dry run.
function writeSyntheticAdapter(file) {
  const tensors = {
    "selftest.to_q.lora_A.weight": { shape: [2, 4], values: [1, 0, 0, 0, 0, 1, 0, 0] },
    "selftest.to_q.lora_B.weight": { shape: [4, 2], values: [1, 0, 0, 1, 0, 0, 0, 0] },
  };
  const header = {};
  let offset = 0;
  const chunks = [];
  for (const [name, { shape, values }] of Object.entries(tensors)) {
    const bytes = Buffer.from(Float32Array.from(values).buffer);
    header[name] = { dtype: "F32", shape, data_offsets: [offset, offset + bytes.length] };
    offset += bytes.length;
    chunks.push(bytes);
  }
  const h = Buffer.from(JSON.stringify(header));
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(h.length));
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, Buffer.concat([len, h, ...chunks]));
}

async function dryRun(opts) {
  const findings = { datasets: {}, rows: {}, baselines: {}, masksRoute: null, samplesRoute: null, checkpointRoute: {}, stableRankSelfTest: null };
  await phaseSetup(opts);
  // Both caption datasets exist side by side; each is checked to carry the captions its mode means.
  for (const mode of CAPTION_MODES) {
    const ds = await ensureDataset(opts, mode);
    findings.datasets[mode] = { id: ds.id, itemCount: ds.itemCount };
  }
  const dataset = currentDataset();

  // masks: the route accepts the dataset (the job is canceled; no worker is running).
  try {
    const job = await api("POST", `/api/v1/projects/${STATE.project.id}/training/datasets/${dataset.id}/subject-mask-jobs`, {});
    await api("POST", `/api/v1/jobs/${job.id}/cancel`, {}).catch(() => {});
    findings.masksRoute = { accepted: true, jobType: job.type };
  } catch (error) {
    findings.masksRoute = { accepted: false, error: error.message };
  }

  // baseline-timing + every train row on both models: submitted with dryRun:true so the API runs
  // the full submit-time validation (target limits, every technique parser, plan build).
  const contexts = {};
  for (const modelKey of Object.keys(MODELS)) {
    const ctx = (contexts[modelKey] = await modelContext(modelKey, opts));
    findings.baselines[modelKey] = { presetId: ctx.preset.id, presetVersion: ctx.preset.version, bucketLadder: ctx.ladder, config: ctx.base };
    findings.rows[modelKey] = {};
    for (const row of techniqueRows(modelKey, opts)) {
      const resolved = resolveRow(row, ctx.target);
      if (resolved.skip) {
        findings.rows[modelKey][row.id] = { skipped: resolved.skip };
        continue;
      }
      const advanced = rowAdvanced(row, ctx, resolved);
      const config = configFor(ctx, advanced);
      try {
        const job = await submitTraining(modelKey, row.id, config, true);
        await api("POST", `/api/v1/jobs/${job.id}/cancel`, {}).catch(() => {});
        const plan = job.payload?.plan ?? {};
        const planAdvanced = plan.config?.advanced ?? {};
        // Every preset key and every row key must reach the plan unchanged (the training adapter
        // and timestep settings included), and the plan's captions must be the caption mode's.
        const lost = Object.entries(config.advanced).filter(([k, v]) => configKey({ v: planAdvanced[k] ?? null }) !== configKey({ v })).map(([k]) => k);
        const captions = (plan.dataset?.items ?? []).map((item) => item.caption);
        const captionsOk =
          captions.length === dataset.itemCount &&
          (opts.captions === "trigger" ? captions.every((c) => c === opts.trigger) : captions.every((c) => typeof c === "string" && c !== opts.trigger));
        findings.rows[modelKey][row.id] = {
          accepted: true,
          advanced,
          dropped: resolved.dropped,
          planKeysLost: lost,
          planCore: ["steps", "resolution", "rank", "alpha", "saveEvery", "seed"].every((k) => plan.config?.[k] === config[k]),
          captionsOk,
          captionSample: captions[0] ?? null,
          // The train phase reads the adapter id + path from the submit response's plan.
          planOutput: Boolean(plan.output?.loraId && plan.output?.outputDir),
        };
      } catch (error) {
        findings.rows[modelKey][row.id] = { accepted: false, advanced, error: error.message };
      }
    }
  }

  // samples: the generation route accepts the base-model request shape for each model.
  findings.samplesRoute = {};
  for (const modelKey of Object.keys(MODELS)) {
    try {
      const job = await api("POST", "/api/v1/image/jobs", sampleRequest(modelKey, opts, PROMPTS[0][1].replaceAll("{t}", opts.trigger), opts.seeds[0], null));
      await api("POST", `/api/v1/jobs/${job.id}/cancel`, {}).catch(() => {});
      findings.samplesRoute[modelKey] = { accepted: true };
    } catch (error) {
      findings.samplesRoute[modelKey] = { accepted: false, error: error.message };
    }
  }

  // checkpoints: a synthetic `<stem>-step<N>.safetensors` in a project LoRA dir (where the trainer
  // writes them) goes through discovery, the app's LoRA import route (run by the in-process CPU
  // utility worker) and an image job that references the imported id.
  for (const modelKey of Object.keys(MODELS)) {
    const ctx = contexts[modelKey];
    const outputDir = path.join(STATE.project.path, "loras", `dryrun_${modelKey}_ckpt`);
    const stem = `ab-${modelKey}-dryrun`;
    const fake = { jobId: `job_dryrun${modelKey}`, loraPath: path.join(outputDir, `${stem}.safetensors`), family: ctx.target.family, baseModel: ctx.target.baseModel };
    writeSyntheticAdapter(path.join(outputDir, `${stem}-step000100.safetensors`));
    writeSyntheticAdapter(path.join(outputDir, `${stem}-step000100.resume.safetensors`));
    try {
      const found = checkpointFiles(fake);
      if (found.length !== 1 || found[0].step !== 100) throw new Error(`checkpoint discovery found ${JSON.stringify(found)}`);
      const loraId = await ensureCheckpointLora(modelKey, "dryrun", fake, found[0]);
      const job = await api("POST", "/api/v1/image/jobs", sampleRequest(modelKey, opts, PROMPTS[0][1].replaceAll("{t}", opts.trigger), opts.seeds[0], loraId));
      await api("POST", `/api/v1/jobs/${job.id}/cancel`, {}).catch(() => {});
      findings.checkpointRoute[modelKey] = { accepted: true, loraId, imageJobLoras: job.payload?.loras ?? null };
    } catch (error) {
      findings.checkpointRoute[modelKey] = { accepted: false, error: error.message };
    }
  }

  // eval: the stable-rank math against known answers (CPU).
  const selfDir = path.join(opts.root, "selftest");
  fs.mkdirSync(selfDir, { recursive: true });
  findings.stableRankSelfTest = stableRankSelfTest(selfDir);

  // report: the report path end to end over whatever state exists.
  phaseReport(opts);

  STATE.dryRun = { at: new Date().toISOString(), findings };
  saveState();
  const rowResults = Object.values(findings.rows).flatMap((r) => Object.values(r));
  const rejected = rowResults.filter(
    (r) => r.accepted === false || (r.accepted && (r.planKeysLost.length || !r.planCore || !r.captionsOk || !r.planOutput)),
  );
  log("dry-run findings:");
  for (const [modelKey, b] of Object.entries(findings.baselines)) {
    log(`  ${modelKey} baseline (${b.presetId} v${b.presetVersion}, buckets ${ladderLabel(b.bucketLadder)}):`);
    console.log(JSON.stringify(b.config, null, 2));
  }
  for (const [mode, ds] of Object.entries(findings.datasets)) log(`  dataset ${mode}: ${ds.id} (${ds.itemCount} items)`);
  for (const [modelKey, rows] of Object.entries(findings.rows)) {
    for (const [id, r] of Object.entries(rows)) {
      const problems = r.accepted
        ? [r.planKeysLost.length && `plan lost ${r.planKeysLost.join(", ")}`, !r.planCore && "plan core mismatch", !r.captionsOk && `captions wrong (${JSON.stringify(r.captionSample)})`, !r.planOutput && "no plan output"].filter(Boolean)
        : [];
      log(`  ${modelKey}/${id}: ${r.skipped ? `skipped (${r.skipped})` : r.accepted ? (problems.length ? `PROBLEM ${problems.join("; ")}` : "accepted") : `REJECTED ${r.error}`}`);
    }
  }
  log(`  masks route: ${JSON.stringify(findings.masksRoute)}`);
  log(`  samples route: ${JSON.stringify(findings.samplesRoute)}`);
  log(`  checkpoint import + sample route: ${JSON.stringify(findings.checkpointRoute)}`);
  log(`  stable-rank self-test: ${findings.stableRankSelfTest.map((t) => `${t.case}=${t.got.toFixed(4)}${t.ok ? "" : " FAIL"}`).join(", ")}`);
  const ok =
    !rejected.length &&
    findings.masksRoute?.accepted &&
    Object.values(findings.samplesRoute).every((r) => r.accepted) &&
    Object.values(findings.checkpointRoute).every((r) => r.accepted) &&
    findings.stableRankSelfTest.every((t) => t.ok);
  log(ok ? "DRY RUN PASSED" : "DRY RUN FAILED");
  return ok;
}

// ---------------------------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------------------------

function requireSetup() {
  if (!STATE.project || !currentDataset()) throw new Error(`run --phase setup first (no ${OPTS.captions}-caption dataset)`);
}

function printStatus() {
  console.log(JSON.stringify(
    {
      root: OPTS.root,
      project: STATE.project ?? null,
      captionMode: OPTS.captions,
      datasets: STATE.datasets,
      training: Object.fromEntries(
        Object.entries(STATE.training).map(([k, rows]) => [k, Object.fromEntries(Object.entries(rows).map(([id, r]) => [id, r.status]))]),
      ),
      samples: Object.fromEntries(
        Object.entries(STATE.samples).map(([k, rows]) => [k, Object.fromEntries(Object.entries(rows).map(([id, r]) => [id, Object.keys(r.files ?? {}).length]))]),
      ),
      eval: Object.fromEntries(Object.entries(STATE.eval).map(([k, rows]) => [k, Object.keys(rows)])),
    },
    null,
    2,
  ));
}

function usage() {
  const lines = fs.readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n");
  console.log(lines.slice(1, lines.findIndex((l, i) => i > 0 && !l.startsWith("//"))).map((l) => l.replace(/^\/\/ ?/, "")).join("\n"));
}

async function main() {
  OPTS = parseArgs(process.argv.slice(2));
  if (OPTS.stableRank) {
    // Standalone CPU utility: the stable rank of any PEFT/kohya LoRA file.
    console.log(JSON.stringify(loraStableRank(path.resolve(OPTS.stableRank)), null, 2));
    return;
  }
  if (OPTS.help || (!OPTS.phase && !OPTS.dryRun)) {
    usage();
    process.exit(OPTS.help ? 0 : 2);
  }
  if (OPTS.dryRun) {
    // A dry run gets its own isolated tree so it never mixes with the real A/B state.
    OPTS.root = path.join(OPTS.root, "dry-run");
  }
  fs.mkdirSync(OPTS.root, { recursive: true });
  loadState(OPTS.root);

  const phase = OPTS.dryRun ? "dry-run" : OPTS.phase;
  if (phase === "status") return printStatus();
  if (phase === "report") return void phaseReport(OPTS);
  if (["train", "samples", "eval"].includes(phase) && !OPTS.model) {
    throw new Error(`--phase ${phase} needs --model ${Object.keys(MODELS).join("|")}`);
  }
  const gpu = GPU_PHASES.has(phase);
  if (gpu && !OPTS.confirmGpu) {
    throw new Error(`--phase ${phase} runs on the GPU; pass --confirm-gpu to proceed (one job at a time)`);
  }

  const bin = phase === "eval" ? null : resolveBinary(OPTS, { gpu });
  if (phase === "eval") {
    // Eval reads only state.json and files on disk, so it starts no server.
    await phaseEval(OPTS, OPTS.model);
    return;
  }
  await startApi(OPTS, bin);
  try {
    switch (phase) {
      case "dry-run": {
        const ok = await dryRun(OPTS);
        process.exitCode = ok ? 0 : 1;
        break;
      }
      case "setup":
        await phaseSetup(OPTS);
        break;
      case "masks":
        requireSetup();
        await startWorker(OPTS, bin, "image_segment");
        await phaseMasks(OPTS);
        break;
      case "baseline-timing":
        requireSetup();
        await startWorker(OPTS, bin, "lora_train_execute");
        await phaseBaselineTiming(OPTS);
        break;
      case "train":
        requireSetup();
        await startWorker(OPTS, bin, "lora_train_execute");
        await phaseTrain(OPTS, OPTS.model);
        break;
      case "samples":
        requireSetup();
        await startWorker(OPTS, bin, "image_generate");
        await phaseSamples(OPTS, OPTS.model);
        break;
      default:
        throw new Error(`unhandled phase ${phase}`);
    }
  } finally {
    await stopWorker();
    await stopApi();
  }
}

main().catch(async (error) => {
  // A signal-driven teardown is already in progress and owns the exit code.
  if (SHUTTING_DOWN) return;
  console.error(`[epic-2123-ab] ${process.env.EPIC_AB_DEBUG ? error.stack : error.message}`);
  await stopWorker().catch(() => {});
  await stopApi().catch(() => {});
  process.exit(1);
});
