#!/usr/bin/env node
// SW-7182 (feature SW-F371): dispatch-only end-to-end Perturbed-Attention Guidance (PAG)
// real-weight acceptance across all six flagship image and image-edit families over the
// production API/worker path on MLX (nax-macos-2) and Candle CUDA (cuda-windows GPU1).
//
// Reached ONLY via memory-catalog-campaign.yml (workflow_dispatch, profile `tier2-ab`,
// moss_options_json `{"suite":"pag-acceptance", ...}`). Ordinary CI runs only pure-function
// and static-contract tests in scripts/pag-acceptance.test.mjs.
import assert from "node:assert/strict";
import { execFile as execFileCallback, spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { copyFile, cp, mkdir, open, readFile, readdir, realpath, rename, rm, stat, statfs, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { deflateSync, inflateSync } from "node:zlib";
import { fileSha256, sha256 } from "./lib/file-sha256.mjs";
import { lockedInferenceRevision } from "./select-windows-cargo-cache.mjs";
import { boundApiUrl, ownedApiController, redactSensitiveText } from "./flux3-image-klein-app-lifecycle.mjs";
import { bindMetallib } from "./ci/memory-catalog/bind-metallib.mjs";
import {
  BACKENDS,
  CUDA_PHYSICAL_GPU,
  CUDA_RUNNERS,
  MLX_RUNNER,
  SHA,
  absoluteFor,
  childEnvironments,
  childStdio,
  contentDigest,
  decodePng,
  defaultHubs,
  expandSelector,
  gpuInventory,
  jobClaimed,
  jobDiagnostics,
  jobProgressDecision,
  lastLines,
  numericDifference,
  nvidiaSmiPath,
  pmetalSource,
  readCatalog,
  readPatternEvidence,
  removeScratch,
  routeDownloadRows,
  routePresence,
  scrubSecrets,
  snapshotDir,
} from "./tier2-ab.mjs";

const execFile = promisify(execFileCallback);
const fact = (value, message) => assert.ok(value, message);
const GIB = 1024 ** 3;
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const pathsFor = (platform) => (platform === "win32" ? path.win32 : path.posix);

// Frozen numerical acceptance thresholds (fixed before first render per SKILL.md).
export const ACCEPTANCE_THRESHOLDS = Object.freeze({
  minChannelStdDev: 3.0,
  minDynamicRange: 20,
  minPagMeanAbsDiff: 0.2,
  minPagChangedFraction: 0.02,
  nondeterminismSeparationFactor: 1.5,
  zeroNoopToleranceFactor: 2.5,
  zeroNoopToleranceFloor: 1.0,
});

export const FAMILIES = Object.freeze([
  "moss_image",
  "minimax_h3_image",
  "krea_2",
  "z_image",
  "flux",
  "qwen_image",
]);

const TIER_TO_QUANTIZE = Object.freeze({ q4: 4, q8: 8, bf16: 16 });
const WORKFLOW_KEYWORD = "sceneworks:workflow";

const T2I_PROMPT = "A sculpted brass astrolabe beside a cobalt glass inkwell on a sunlit oak desk, crisp typography card reading 'SCENEWORKS PAG', studio photograph";
const EDIT_SINGLE_PROMPT = "Transform the crimson circle and cobalt square into polished ceramic vessels on a marble shelf while keeping their exact left-to-right colors";
const EDIT_H3_MULTI_PROMPT = "Place the crimson disc from <Picture 1> beside the emerald triangle from <Picture 2> as glazed ceramic sculptures on a studio pedestal";

// Thirteen acceptance routes spanning AT1, AT2, AT3, AT4, AT6, and AT7 across all six families.
export const PAG_ROUTES = Object.freeze([
  {
    id: "moss-t2i",
    family: "moss_image",
    model: "moss_image_sft",
    mode: "text_to_image",
    kind: "t2i_ab",
    tierCandidates: [null],
    claimMinutes: 5,
    timeoutMinutes: 25,
    body: {
      mode: "text_to_image",
      model: "moss_image_sft",
      prompt: T2I_PROMPT,
      negativePrompt: "",
      seed: 26325,
      width: 512,
      height: 512,
      count: 1,
      loras: [],
      advanced: {
        steps: 4,
        guidanceScale: 1.0,
        mossImage: { acceleration: "off", max_sequence_length: 128 },
      },
    },
    pagOn: { scale: 2.5 },
    expectedReport: { layers: [20], activeSteps: 4, totalSteps: 4 },
    at: ["AT1", "AT2", "AT6"],
  },
  {
    id: "moss-edit-kv",
    family: "moss_image",
    model: "moss_image_sft",
    mode: "edit_image",
    kind: "edit_pag",
    tierCandidates: [null],
    conditionalComponentIds: ["moss-image-edit"],
    references: "moss_single",
    claimMinutes: 5,
    timeoutMinutes: 25,
    body: {
      mode: "edit_image",
      model: "moss_image_sft",
      prompt: EDIT_SINGLE_PROMPT,
      negativePrompt: "",
      seed: 26326,
      count: 1,
      loras: [],
      advanced: {
        steps: 4,
        guidanceScale: 1.0,
        mossImage: { acceleration: "off", max_sequence_length: 128 },
        mossImageEdit: { reference_max_edge: 256, use_kv_cache: true },
      },
    },
    pagOn: { scale: 2.5 },
    expectedReport: { layers: [20], activeSteps: 4, totalSteps: 4 },
    at: ["AT3", "AT6"],
  },
  {
    id: "h3-image-t2i",
    family: "minimax_h3_image",
    model: "minimax_h3_image",
    mode: "text_to_image",
    kind: "t2i_ab",
    licenseAck: true,
    tierCandidates: ["q4", "q8", "bf16"],
    claimMinutes: 5,
    timeoutMinutes: 35,
    body: {
      mode: "text_to_image",
      model: "minimax_h3_image",
      prompt: T2I_PROMPT,
      negativePrompt: "",
      seed: 7175,
      width: 1024,
      height: 1024,
      count: 1,
      loras: [],
      advanced: {
        steps: 6,
        h3ImageSampler: { sampler: "euler" },
      },
    },
    pagOn: { scale: 1.2 },
    expectedReport: { layers: [25], activeSteps: 4, totalSteps: 6 },
    at: ["AT1", "AT2", "AT4", "AT6"],
  },
  {
    id: "h3-image-edit-multiref",
    family: "minimax_h3_image",
    model: "minimax_h3_image",
    mode: "edit_image",
    kind: "edit_pag",
    licenseAck: true,
    tierCandidates: ["q4", "q8", "bf16"],
    references: "multi_two",
    claimMinutes: 5,
    timeoutMinutes: 40,
    body: {
      mode: "edit_image",
      model: "minimax_h3_image",
      prompt: EDIT_H3_MULTI_PROMPT,
      negativePrompt: "",
      seed: 7176,
      width: 1024,
      height: 1024,
      count: 1,
      loras: [],
      advanced: {
        steps: 6,
        h3ImageSampler: { sampler: "euler" },
      },
    },
    pagOn: { scale: 1.2 },
    expectedReport: { layers: [25], activeSteps: 4, totalSteps: 6 },
    at: ["AT3", "AT4", "AT6"],
  },
  {
    id: "krea-turbo-t2i",
    family: "krea_2",
    model: "krea_2_turbo",
    mode: "text_to_image",
    kind: "t2i_ab",
    tierCandidates: ["q4", "q8", "bf16"],
    claimMinutes: 5,
    timeoutMinutes: 25,
    body: {
      mode: "text_to_image",
      model: "krea_2_turbo",
      prompt: T2I_PROMPT,
      negativePrompt: "",
      seed: 71760,
      width: 768,
      height: 768,
      count: 1,
      loras: [],
      advanced: {
        steps: 4,
        sampler: "euler",
        guidanceScale: 0,
      },
    },
    pagOn: { scale: 1.5 },
    expectedReport: { layers: [14], activeSteps: 3, totalSteps: 4 },
    at: ["AT1", "AT2", "AT4", "AT6"],
  },
  {
    id: "krea-raw-g1-t2i",
    family: "krea_2",
    model: "krea_2_raw",
    mode: "text_to_image",
    kind: "single_pag",
    tierCandidates: ["q4", "q8", "bf16"],
    claimMinutes: 5,
    timeoutMinutes: 25,
    body: {
      mode: "text_to_image",
      model: "krea_2_raw",
      prompt: T2I_PROMPT,
      negativePrompt: "",
      seed: 71761,
      width: 768,
      height: 768,
      count: 1,
      loras: [],
      advanced: {
        steps: 4,
        sampler: "euler",
        guidanceScale: 1.0,
      },
    },
    pagOn: { scale: 1.8 },
    expectedReport: { layers: [14], activeSteps: 4, totalSteps: 4 },
    at: ["AT4", "AT6"],
  },
  {
    id: "krea-turbo-edit",
    family: "krea_2",
    model: "krea_2_turbo",
    mode: "edit_image",
    kind: "edit_pag",
    tierCandidates: ["q4", "q8", "bf16"],
    references: "source_single",
    claimMinutes: 5,
    timeoutMinutes: 25,
    body: {
      mode: "edit_image",
      model: "krea_2_turbo",
      prompt: EDIT_SINGLE_PROMPT,
      negativePrompt: "",
      seed: 71762,
      width: 768,
      height: 768,
      count: 1,
      loras: [],
      advanced: {
        steps: 4,
        sampler: "euler",
        guidanceScale: 0,
      },
    },
    pagOn: { scale: 1.5 },
    expectedReport: { layers: [14], activeSteps: 3, totalSteps: 4 },
    at: ["AT3", "AT4", "AT6"],
  },
  {
    id: "z-image-turbo-t2i",
    family: "z_image",
    model: "z_image_turbo",
    mode: "text_to_image",
    kind: "t2i_ab",
    tierCandidates: ["q4", "q8", "bf16"],
    claimMinutes: 5,
    timeoutMinutes: 25,
    body: {
      mode: "text_to_image",
      model: "z_image_turbo",
      prompt: T2I_PROMPT,
      negativePrompt: "",
      seed: 71770,
      width: 768,
      height: 768,
      count: 1,
      loras: [],
      advanced: {
        steps: 4,
        sampler: "euler",
        guidanceScale: 0,
      },
    },
    pagOn: { scale: 1.2 },
    expectedReport: { layers: [15], activeSteps: 3, totalSteps: 4 },
    at: ["AT1", "AT2", "AT6"],
  },
  {
    id: "z-image-turbo-lora",
    family: "z_image",
    model: "z_image_turbo",
    mode: "text_to_image",
    kind: "single_pag",
    tierCandidates: ["q4", "q8", "bf16"],
    syntheticLora: true,
    claimMinutes: 5,
    timeoutMinutes: 25,
    body: {
      mode: "text_to_image",
      model: "z_image_turbo",
      prompt: T2I_PROMPT,
      negativePrompt: "",
      seed: 71770,
      width: 768,
      height: 768,
      count: 1,
      loras: [{ id: "sw_f371_z_pag_lora", weight: 0.75 }],
      advanced: {
        steps: 4,
        sampler: "euler",
        guidanceScale: 0,
      },
    },
    pagOn: { scale: 1.2 },
    expectedReport: { layers: [15], activeSteps: 3, totalSteps: 4 },
    at: ["AT7"],
  },
  {
    id: "z-image-edit",
    family: "z_image",
    model: "z_image_edit",
    mode: "edit_image",
    kind: "edit_pag",
    tierCandidates: ["q4", "q8", "bf16"],
    references: "source_single",
    claimMinutes: 5,
    timeoutMinutes: 25,
    body: {
      mode: "edit_image",
      model: "z_image_edit",
      prompt: EDIT_SINGLE_PROMPT,
      negativePrompt: "",
      seed: 71771,
      width: 768,
      height: 768,
      count: 1,
      loras: [],
      advanced: {
        steps: 4,
        sampler: "euler",
        guidanceScale: 0,
      },
    },
    pagOn: { scale: 1.2 },
    expectedReport: { layers: [15], activeSteps: 3, totalSteps: 4 },
    at: ["AT3", "AT6"],
  },
  {
    id: "flux1-dev-t2i",
    family: "flux",
    model: "flux_dev",
    mode: "text_to_image",
    kind: "t2i_ab",
    tierCandidates: ["q4", "q8", "bf16"],
    claimMinutes: 5,
    timeoutMinutes: 25,
    body: {
      mode: "text_to_image",
      model: "flux_dev",
      prompt: T2I_PROMPT,
      negativePrompt: "",
      seed: 71780,
      width: 768,
      height: 768,
      count: 1,
      loras: [],
      advanced: {
        steps: 4,
        sampler: "euler",
        guidanceScale: 3.5,
      },
    },
    pagOn: { scale: 1.5 },
    expectedReport: { layers: [28], activeSteps: 4, totalSteps: 4 },
    at: ["AT1", "AT2", "AT4", "AT6"],
  },
  {
    id: "flux2-klein-kv-edit",
    family: "flux",
    model: "flux2_klein_9b_kv",
    mode: "edit_image",
    kind: "edit_pag",
    tierCandidates: ["q4", "q8", "bf16"],
    references: "source_single",
    claimMinutes: 5,
    timeoutMinutes: 25,
    body: {
      mode: "edit_image",
      model: "flux2_klein_9b_kv",
      prompt: EDIT_SINGLE_PROMPT,
      negativePrompt: "",
      seed: 71781,
      width: 768,
      height: 768,
      count: 1,
      loras: [],
      advanced: {
        steps: 4,
        sampler: "euler",
        guidanceScale: 1.0,
      },
    },
    pagOn: { scale: 1.5 },
    expectedReport: { layers: [20], activeSteps: 4, totalSteps: 4 },
    at: ["AT3", "AT4", "AT6"],
  },
  {
    id: "qwen-2-1-turbo-t2i",
    family: "qwen_image",
    group: "qwen-t2i",
    model: "qwen_image_2_1_turbo",
    mode: "text_to_image",
    kind: "t2i_ab",
    licenseAck: true,
    tierCandidates: ["q4", "q8", "bf16"],
    claimMinutes: 5,
    timeoutMinutes: 25,
    body: {
      mode: "text_to_image",
      model: "qwen_image_2_1_turbo",
      prompt: T2I_PROMPT,
      negativePrompt: "",
      seed: 71790,
      width: 1024,
      height: 1024,
      count: 1,
      loras: [],
      advanced: {
        steps: 4,
        sampler: "euler",
        guidanceScale: 1.0,
      },
    },
    pagOn: { scale: 1.2 },
    expectedReport: { layers: [16], activeSteps: 3, totalSteps: 4 },
    at: ["AT1", "AT2", "AT6"],
  },
  {
    id: "qwen-1-t2i",
    family: "qwen_image",
    group: "qwen-t2i",
    model: "qwen_image",
    mode: "text_to_image",
    kind: "t2i_ab",
    tierCandidates: ["q4", "q8", "bf16"],
    claimMinutes: 5,
    timeoutMinutes: 25,
    body: {
      mode: "text_to_image",
      model: "qwen_image",
      prompt: T2I_PROMPT,
      negativePrompt: "",
      seed: 71791,
      width: 768,
      height: 768,
      count: 1,
      loras: [],
      advanced: {
        steps: 4,
        sampler: "euler",
        guidanceScale: 1.0,
      },
    },
    pagOn: { scale: 1.5 },
    expectedReport: { layers: [30], activeSteps: 4, totalSteps: 4 },
    at: ["AT1", "AT2", "AT6"],
  },
  {
    id: "qwen-edit-2511",
    family: "qwen_image",
    model: "qwen_image_edit_2511",
    mode: "edit_image",
    kind: "edit_pag",
    tierCandidates: ["q4", "q8", "bf16"],
    references: "source_single",
    claimMinutes: 5,
    timeoutMinutes: 30,
    body: {
      mode: "edit_image",
      model: "qwen_image_edit_2511",
      prompt: EDIT_SINGLE_PROMPT,
      negativePrompt: "",
      seed: 71792,
      width: 768,
      height: 768,
      count: 1,
      loras: [],
      advanced: {
        steps: 4,
        sampler: "euler",
        guidanceScale: 1.0,
      },
    },
    pagOn: { scale: 1.5 },
    expectedReport: { layers: [30], activeSteps: 4, totalSteps: 4 },
    at: ["AT3", "AT6"],
  },
]);

export const PAG_ROUTE_MAP = Object.freeze(Object.fromEntries(PAG_ROUTES.map((route) => [route.id, route])));

export const PAG_DISPATCH_OPTION_KEYS = Object.freeze(["suite", "routes", "hubs", "preflight_only", "stage_missing"]);

export function dispatchPagAcceptanceOptions(inputs, dispatchedSha) {
  assert.equal(inputs.profile, "tier2-ab");
  fact(SHA.test(dispatchedSha ?? "") && inputs.ref === dispatchedSha, "ref must equal the exact dispatched 40-hex source SHA");
  fact(["mlx", "candle"].includes(inputs.backend), "backend must be mlx or candle");
  fact(inputs.download_missing !== true && inputs.qwen_persistent_weights !== true, "leave catalog download_missing and qwen_persistent_weights off; use moss_options_json.stage_missing when needed");
  let options;
  try { options = JSON.parse(inputs.moss_options_json || "{}"); } catch { throw new Error("moss_options_json must be a JSON object"); }
  fact(options && typeof options === "object" && !Array.isArray(options), "pag-acceptance options must be a JSON object");
  assert.equal(options.suite, "pag-acceptance");
  const unknown = Object.keys(options).filter((key) => !PAG_DISPATCH_OPTION_KEYS.includes(key));
  fact(unknown.length === 0, `unknown pag-acceptance option(s): ${unknown.join(", ")}`);
  fact(options.preflight_only === undefined || typeof options.preflight_only === "boolean", "preflight_only must be boolean");
  fact(options.stage_missing === undefined || typeof options.stage_missing === "boolean", "stage_missing must be boolean");
  const preflightOnly = options.preflight_only ?? true;
  const stageMissingFlag = options.stage_missing ?? false;
  fact(!(stageMissingFlag && preflightOnly), "stage_missing cannot be combined with preflight_only=true");
  fact(!inputs.inference_sha || SHA.test(inputs.inference_sha), "inference_sha must be empty or exact 40-hex");
  const backend = inputs.backend === "mlx" ? "mlx" : "cuda";
  for (const key of ["routes", "hubs"]) {
    fact(options[key] === undefined || (Array.isArray(options[key]) && options[key].every((item) => typeof item === "string" && item.trim())), `${key} must be an array of strings`);
  }
  const selected = selectPagRoutes(options.routes ? JSON.stringify(options.routes) : "").map((route) => route.id);
  if (options.hubs) fact(options.hubs.length > 0 && options.hubs.every((hub) => absoluteFor(backend === "mlx" ? "darwin" : "win32", hub)), "hubs must be absolute paths for the selected backend's OS");
  return {
    suite: "pag-acceptance",
    source_sha: dispatchedSha,
    backend,
    routes: options.routes ? JSON.stringify(selected) : "",
    hubs: options.hubs ? JSON.stringify(options.hubs) : "",
    preflight_only: preflightOnly,
    stage_missing: stageMissingFlag,
    inference_sha: inputs.inference_sha || "",
  };
}

export function selectPagRoutes(value) {
  const text = (value ?? "").trim();
  const ids = !text || text === "default"
    ? PAG_ROUTES.map((route) => route.id)
    : text.startsWith("[")
      ? JSON.parse(text)
      : text.split(/[,\s]+/).map((item) => item.trim()).filter(Boolean);
  fact(Array.isArray(ids) && ids.length > 0 && new Set(ids).size === ids.length, "route list must be non-empty and unique");
  for (const id of ids) fact(Object.hasOwn(PAG_ROUTE_MAP, id), `unknown PAG route ${id}`);
  return ids.map((id) => PAG_ROUTE_MAP[id]);
}

export function pagDefaultHubs(backend, home = os.homedir()) {
  const base = defaultHubs(backend, home);
  if (backend === "cuda") {
    const extra = ["D:\\.cache\\huggingface\\hub", "D:\\huggingface\\hub", path.win32.join(home, ".cache\\huggingface\\hub")];
    return [...new Set([...base, ...extra])];
  }
  const rw = path.posix.join(home, "sceneworks-rw-weights/hub");
  return base.includes(rw) ? base : [...base, rw];
}

export function resolvePagConfig(env, { platform = process.platform, home = os.homedir() } = {}) {
  const paths = pathsFor(platform);
  const backend = env.PAG_ACCEPTANCE_BACKEND;
  fact(BACKENDS.includes(backend), `PAG_ACCEPTANCE_BACKEND must be one of ${BACKENDS.join("|")}`);
  const sourceSha = env.PAG_ACCEPTANCE_SOURCE_SHA ?? "";
  fact(SHA.test(sourceSha), "PAG_ACCEPTANCE_SOURCE_SHA must be an exact 40-hex lowercase commit SHA");
  const inferenceSha = (env.PAG_ACCEPTANCE_INFERENCE_SHA ?? "").trim();
  fact(inferenceSha === "" || SHA.test(inferenceSha), "PAG_ACCEPTANCE_INFERENCE_SHA must be empty or 40-hex");
  const absolute = (key) => { const value = env[key] ?? ""; fact(absoluteFor(platform, value), `${key} must be an absolute path`); return paths.resolve(value); };
  const sourceDir = absolute("PAG_ACCEPTANCE_SOURCE_DIR"), work = absolute("PAG_ACCEPTANCE_WORK");
  const rel = paths.relative(sourceDir, work);
  fact(rel.startsWith("..") || paths.isAbsolute(rel), "work directory must be outside checkout");
  if (env.GITHUB_ACTIONS === "true") {
    fact(env.RUNNER_TEMP && paths.isAbsolute(env.RUNNER_TEMP), "RUNNER_TEMP required in Actions");
    const tempRel = paths.relative(paths.resolve(env.RUNNER_TEMP), work);
    fact(tempRel && !tempRel.startsWith("..") && !paths.isAbsolute(tempRel), "PAG_ACCEPTANCE_WORK must be under RUNNER_TEMP in Actions");
  }
  const routes = selectPagRoutes(env.PAG_ACCEPTANCE_ROUTES);
  const rawHubs = (env.PAG_ACCEPTANCE_HUBS ?? "").trim();
  const hubs = !rawHubs
    ? pagDefaultHubs(backend, home)
    : (rawHubs.startsWith("[") ? JSON.parse(rawHubs) : rawHubs.split(/[;\r\n]+/).map((s) => s.trim()).filter(Boolean));
  fact(Array.isArray(hubs) && hubs.length > 0 && hubs.every((hub) => absoluteFor(platform, hub)), "every hub root must be an absolute path");
  return { backend, sourceSha, inferenceSha, sourceDir, work, routes, hubs: hubs.map((hub) => paths.resolve(hub)) };
}

// ---- deterministic synthetic reference PNGs & rank-4 Z-Image safetensors LoRA -------------------
function pngCrc(buf) {
  let c = 0xffffffff;
  for (const byte of buf) {
    c ^= byte;
    for (let i = 0; i < 8; i++) c = (c >>> 1) ^ ((c & 1) ? 0xedb88320 : 0);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const out = Buffer.alloc(data.length + 12);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, "ascii");
  data.copy(out, 8);
  out.writeUInt32BE(pngCrc(Buffer.concat([Buffer.from(type, "ascii"), data])), data.length + 8);
  return out;
}

export function syntheticReferencePng(variant = "primary", size = 256) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // 8-bit
  ihdr[9] = 2; // RGB
  const stride = 1 + size * 3;
  const raw = Buffer.alloc(size * stride);
  for (let y = 0; y < size; y++) {
    raw[y * stride] = 0;
    for (let x = 0; x < size; x++) {
      const o = y * stride + 1 + x * 3;
      let r = 244, g = 239, b = 230;
      if (variant === "primary") {
        const dx = x - size * 0.3, dy = y - size * 0.5;
        if (dx * dx + dy * dy <= (size * 0.2) ** 2) { r = 220; g = 40; b = 40; }
        else if (x >= size * 0.58 && x <= size * 0.86 && y >= size * 0.3 && y <= size * 0.7) { r = 40; g = 90; b = 220; }
      } else {
        const cx = size * 0.5, topY = size * 0.22, botY = size * 0.78;
        r = 240; g = 200; b = 60;
        if (y >= topY && y <= botY) {
          const half = ((y - topY) / (botY - topY)) * (size * 0.3);
          if (Math.abs(x - cx) <= half) { r = 32; g = 168; b = 88; }
        }
      }
      raw[o] = r; raw[o + 1] = g; raw[o + 2] = b;
    }
  }
  return Buffer.concat([
    Buffer.from("89504e470d0a1a0a", "hex"),
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", deflateSync(raw)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

// Builds a deterministic rank-4 F32 PEFT LoRA targeting Z-Image block 15 (its default PAG block).
export function buildSyntheticZImageLoraSafetensors(dim = 3840, rank = 4) {
  const specs = [
    ["layers.15.attention.to_q.lora_A.weight", [rank, dim], 0.02],
    ["layers.15.attention.to_q.lora_B.weight", [dim, rank], 0.02],
    ["layers.15.attention.to_v.lora_A.weight", [rank, dim], 0.02],
    ["layers.15.attention.to_v.lora_B.weight", [dim, rank], 0.02],
  ];
  const header = {
    __metadata__: {
      networkType: "lora",
      rank: String(rank),
      alpha: String(rank),
    },
  };
  const payloads = [];
  let offset = 0;
  for (const [name, shape, scale] of specs) {
    const count = shape[0] * shape[1];
    const buf = Buffer.alloc(count * 4);
    for (let i = 0; i < count; i++) {
      const val = Math.sin((i + 1) * 0.0137) * scale;
      buf.writeFloatLE(val, i * 4);
    }
    header[name] = { dtype: "F32", shape, data_offsets: [offset, offset + buf.length] };
    offset += buf.length;
    payloads.push(buf);
  }
  const headerBytes = Buffer.from(JSON.stringify(header), "utf8");
  const pad = (8 - (headerBytes.length % 8)) % 8;
  const paddedHeader = pad ? Buffer.concat([headerBytes, Buffer.alloc(pad, 0x20)]) : headerBytes;
  const lenBuf = Buffer.alloc(8);
  lenBuf.writeBigUInt64LE(BigInt(paddedHeader.length), 0);
  return Buffer.concat([lenBuf, paddedHeader, ...payloads]);
}

// ---- PNG workflow chunk parser & image quality stats --------------------------------------------
export function readWorkflowShareFromPng(buffer) {
  fact(buffer.length > 8 && buffer.subarray(0, 8).toString("hex") === "89504e470d0a1a0a", "not a PNG");
  let offset = 8;
  while (offset + 8 <= buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString("ascii", offset + 4, offset + 8);
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    if (type === "iTXt") {
      const nul = data.indexOf(0);
      if (nul > 0 && data.toString("latin1", 0, nul) === WORKFLOW_KEYWORD) {
        const compFlag = data[nul + 1];
        let p = nul + 3; // skip nul + compFlag + compMethod
        const langEnd = data.indexOf(0, p);
        p = langEnd + 1;
        const transEnd = data.indexOf(0, p);
        p = transEnd + 1;
        const payload = data.subarray(p);
        const text = (compFlag === 1 ? inflateSync(payload) : payload).toString("utf8");
        return JSON.parse(text);
      }
    } else if (type === "IEND") break;
    offset += 12 + length;
  }
  return null;
}

export function imageQualityStats(rgba) {
  fact(rgba && rgba.length >= 4 && rgba.length % 4 === 0, "RGBA8 buffer required");
  const pixels = rgba.length / 4;
  let sum = 0, sumSq = 0, min = 255, max = 0;
  const count = pixels * 3;
  for (let i = 0; i < rgba.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const v = rgba[i + c];
      sum += v;
      sumSq += v * v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const mean = sum / count;
  const variance = Math.max(0, sumSq / count - mean * mean);
  return {
    pixels,
    rgbMean: Number(mean.toFixed(3)),
    channelStdDev: Number(Math.sqrt(variance).toFixed(3)),
    minChannel: min,
    maxChannel: max,
    dynamicRange: max - min,
  };
}

export function assertCoherentStats(stats, label, thresholds = ACCEPTANCE_THRESHOLDS) {
  fact(Number.isFinite(stats.channelStdDev) && stats.channelStdDev >= thresholds.minChannelStdDev, `${label}: degenerate flat image (stdDev ${stats.channelStdDev} < ${thresholds.minChannelStdDev})`);
  fact(stats.dynamicRange >= thresholds.minDynamicRange, `${label}: collapsed dynamic range (${stats.dynamicRange} < ${thresholds.minDynamicRange})`);
}

export function assertPagReportMatches(actual, expected, expectedScale, label) {
  fact(actual && typeof actual === "object", `${label}: missing pagExecutionReport`);
  fact(Math.abs(Number(actual.scale) - expectedScale) < 1e-4, `${label}: pagExecutionReport.scale ${actual.scale} != ${expectedScale}`);
  assert.deepEqual(actual.layers, expected.layers, `${label}: pagExecutionReport.layers mismatch`);
  assert.equal(actual.activeSteps, expected.activeSteps, `${label}: pagExecutionReport.activeSteps mismatch`);
  assert.equal(actual.totalSteps, expected.totalSteps, `${label}: pagExecutionReport.totalSteps mismatch`);
}

export function evaluateT2iAbVerdict({ off1, zero, pagOn, off2 = null, expectedReport, pagScale, thresholds = ACCEPTANCE_THRESHOLDS }) {
  for (const [name, run] of [["off1", off1], ["zero", zero], ["pagOn", pagOn]]) {
    if (!run || run.status !== "completed") return { verdict: "error", needOff2: false, reason: `${name} failed: ${run?.error ?? "not run"}` };
    assertCoherentStats(run.stats, name, thresholds);
  }
  if (off1.pagReport !== null || off1.sharePagReport !== null) return { verdict: "error", needOff2: false, reason: "off1 unexpectedly emitted pagExecutionReport" };
  if (zero.pagReport !== null || zero.sharePagReport !== null) return { verdict: "error", needOff2: false, reason: "scale=0 unexpectedly emitted pagExecutionReport" };
  assertPagReportMatches(pagOn.pagReport, expectedReport, pagScale, "recipe pagExecutionReport");
  assertPagReportMatches(pagOn.sharePagReport, expectedReport, pagScale, "PNG workflow pagExecutionReport");

  const zeroVsOff1 = numericDifference(off1.buffer, zero.buffer);
  const pagVsOff1 = numericDifference(off1.buffer, pagOn.buffer);
  if (zeroVsOff1.maxAbsoluteDifference > 0 && !off2) {
    return { verdict: null, needOff2: true, reason: "scale=0 differed from off1; running off2 baseline repeatability check" };
  }
  if (zeroVsOff1.maxAbsoluteDifference > 0) {
    if (!off2 || off2.status !== "completed") return { verdict: "error", needOff2: false, reason: `off2 failed: ${off2?.error ?? "not run"}` };
    const off2VsOff1 = numericDifference(off1.buffer, off2.buffer);
    if (off2VsOff1.maxAbsoluteDifference === 0) {
      return { verdict: "error", needOff2: false, reason: `AT2 violation: off1 == off2 (deterministic runner), but pag.scale=0 differed (meanAbsDiff ${zeroVsOff1.meanAbsoluteDifference})` };
    }
    const zeroCeiling = off2VsOff1.meanAbsoluteDifference * thresholds.zeroNoopToleranceFactor + thresholds.zeroNoopToleranceFloor;
    if (zeroVsOff1.meanAbsoluteDifference > zeroCeiling) {
      return { verdict: "error", needOff2: false, reason: `AT2 violation: pag.scale=0 drift (${zeroVsOff1.meanAbsoluteDifference}) exceeded runner nondeterminism ceiling (${zeroCeiling})` };
    }
    if (pagVsOff1.meanAbsoluteDifference < Math.max(thresholds.minPagMeanAbsDiff, off2VsOff1.meanAbsoluteDifference * thresholds.nondeterminismSeparationFactor)) {
      return { verdict: "error", needOff2: false, reason: `AT1 violation: pagOn meanAbsDiff (${pagVsOff1.meanAbsoluteDifference}) did not separate from runner noise (${off2VsOff1.meanAbsoluteDifference})` };
    }
    return {
      verdict: "passed_with_runner_noise",
      needOff2: false,
      reason: `AT1/AT2/AT6 verified (runner baseline noise ${off2VsOff1.meanAbsoluteDifference.toFixed(3)}, zero=${zeroVsOff1.meanAbsoluteDifference.toFixed(3)}, pagOn=${pagVsOff1.meanAbsoluteDifference.toFixed(3)})`,
      metrics: { zeroVsOff1, pagVsOff1, off2VsOff1 },
    };
  }

  if (pagVsOff1.meanAbsoluteDifference < thresholds.minPagMeanAbsDiff || pagVsOff1.changedFraction < thresholds.minPagChangedFraction) {
    return {
      verdict: "error",
      needOff2: false,
      reason: `AT1 violation: PAG > 0 produced insufficient structural delta vs off1 (meanAbsDiff=${pagVsOff1.meanAbsoluteDifference}, changedFraction=${pagVsOff1.changedFraction})`,
      metrics: { zeroVsOff1, pagVsOff1 },
    };
  }
  return {
    verdict: "passed",
    needOff2: false,
    reason: `AT1/AT2/AT6 verified: scale=0 bit-identical (maxAbsDiff=0), PAG scale=${pagScale} meanAbsDiff=${pagVsOff1.meanAbsoluteDifference.toFixed(3)} (${(pagVsOff1.changedFraction * 100).toFixed(1)}% channels changed)`,
    metrics: { zeroVsOff1, pagVsOff1 },
  };
}

// ---- planning & preflight -----------------------------------------------------------------------
export async function planPagRoutes({ routes, hubs, catalog, platform, fs, evidence = new Map() }) {
  const planned = [], ranGroups = new Map();
  for (const route of routes) {
    const base = { id: route.id, family: route.family, model: route.model, mode: route.mode, kind: route.kind, at: route.at };
    const model = catalog.find((entry) => entry.id === route.model);
    if (!model) { planned.push({ ...base, status: "skipped", skipReason: `model ${route.model} absent from catalog` }); continue; }
    if (model.gated === true) { planned.push({ ...base, status: "skipped", skipReason: `${route.model} is gated` }); continue; }
    if (route.group && ranGroups.has(route.group)) {
      planned.push({ ...base, status: "skipped", skipReason: `group ${route.group} already satisfied by ${ranGroups.get(route.group)}` });
      continue;
    }
    const conditionalRows = (route.conditionalComponentIds ?? []).map((cid) => {
      const found = (model.conditionalComponents ?? []).find((c) => c.componentId === cid);
      fact(found, `model ${route.model} has no conditionalComponent ${cid}`);
      return { repo: found.repo, revision: found.revision, variant: null, files: found.files ?? [], componentId: cid, coRequisite: false, required: null };
    });
    const rowsByTier = [];
    for (const tier of route.tierCandidates) {
      try {
        const rows = routeDownloadRows(model, { platform, tier });
        rows.required.push(...conditionalRows);
        rowsByTier.push(rows);
      } catch { /* tier unavailable on platform */ }
    }
    if (!rowsByTier.length) {
      planned.push({ ...base, status: "skipped", skipReason: `${route.model} has no compatible tier rows on ${platform}` });
      continue;
    }
    const checks = [];
    let selected = null, selectedRows = null;
    for (const hub of hubs) {
      for (const rows of rowsByTier) {
        const res = { ...(await routePresence(hub, rows, fs, evidence)), tier: rows.tier };
        checks.push(res);
        if (res.complete) { selected = res; selectedRows = rows; break; }
      }
      if (selected) break;
    }
    if (!selected) {
      planned.push({
        ...base,
        status: "skipped",
        presence: checks,
        skipReason: `weights absent in hubs (${checks.map((c) => `${c.hub} [${c.tier ?? "untiered"}]: ${c.missing.slice(0, 2).join(", ")}`).join("; ")})`,
      });
      continue;
    }
    if (route.group) ranGroups.set(route.group, route.id);
    planned.push({
      ...base,
      status: "ready",
      hub: selected.hub,
      tier: selectedRows.tier,
      rows: selectedRows,
    });
  }
  return planned;
}

async function git(cwd, ...args) {
  return (await execFile("git", args, { cwd, timeout: 60_000, maxBuffer: 16 * 1024 * 1024 })).stdout.trim();
}
async function freeBytes(dir) {
  try { const fs = await statfs(dir, { bigint: true }); return Number(fs.bavail * fs.bsize); } catch { return null; }
}

async function preflight(env, cfg, report) {
  const platform = process.platform;
  if (cfg.backend === "mlx") fact(platform === "darwin" && process.arch === "arm64", "mlx needs Apple-silicon macOS");
  else fact(platform === "win32", "cuda needs Windows runner");
  if (env.GITHUB_ACTIONS === "true") {
    if (cfg.backend === "mlx") fact(env.RUNNER_NAME === MLX_RUNNER, `mlx PAG acceptance must run on ${MLX_RUNNER} (rw-mage), not ${env.RUNNER_NAME}`);
    else fact(CUDA_RUNNERS.includes(env.RUNNER_NAME), `cuda PAG acceptance must run on ${CUDA_RUNNERS.join("|")}, not ${env.RUNNER_NAME}`);
  }
  if (cfg.backend === "cuda") {
    for (const key of ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "SCENEWORKS_GPU_ID"]) {
      fact(env[key] === CUDA_PHYSICAL_GPU, `${key}=${CUDA_PHYSICAL_GPU} required`);
    }
    fact(env.CUDA_DEVICE_ORDER === "PCI_BUS_ID", "CUDA_DEVICE_ORDER=PCI_BUS_ID required");
    const smi = nvidiaSmiPath(env);
    const csv = (await execFile(smi, ["--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader"], { timeout: 30_000, windowsHide: true })).stdout;
    const xml = (await execFile(smi, ["-q", "-x"], { timeout: 30_000, windowsHide: true, maxBuffer: 64 * 1024 * 1024 })).stdout;
    report.gpus = gpuInventory(csv, xml);
  }
  const head = await git(cfg.sourceDir, "rev-parse", "HEAD"), dirty = await git(cfg.sourceDir, "status", "--porcelain");
  fact(head === cfg.sourceSha && dirty === "", `checkout ${cfg.sourceDir} must be clean at ${cfg.sourceSha} (saw ${head}, dirty=${dirty.length})`);
  const lock = await readFile(path.join(cfg.sourceDir, "Cargo.lock"), "utf8");
  const inferencePin = lockedInferenceRevision(lock);
  if (cfg.inferenceSha) fact(inferencePin === cfg.inferenceSha, `inference pin ${inferencePin} != ${cfg.inferenceSha}`);
  report.source = { sha: cfg.sourceSha, inferencePin, pmetalSource: pmetalSource(lock) };
  await mkdir(cfg.work, { recursive: true });
  const free = await freeBytes(cfg.work);
  fact(free === null || free >= 20 * GIB, `insufficient scratch space: ${free} bytes`);
  const catalog = await readCatalog(cfg.sourceDir);
  const evidence = await readPatternEvidence(cfg.sourceDir);
  report.routes = await planPagRoutes({ routes: cfg.routes, hubs: cfg.hubs, catalog, platform, evidence });
}

export function resolveMissingSnapshotFiles({ routes, plannedRoutes, catalog, evidence, platform }) {
  const byId = new Map(plannedRoutes.map((r) => [r.id, r]));
  const targets = new Map();
  for (const route of routes) {
    const planned = byId.get(route.id);
    if (!planned || planned.status !== "skipped") continue;
    const model = catalog.find((entry) => entry.id === route.model);
    if (!model || model.gated === true) continue;
    const rows = routeDownloadRows(model, { platform, tier: route.tierCandidates[0] });
    for (const cid of route.conditionalComponentIds ?? []) {
      const found = (model.conditionalComponents ?? []).find((c) => c.componentId === cid);
      fact(found, `model ${route.model} has no conditionalComponent ${cid}`);
      rows.required.push({ repo: found.repo, revision: found.revision, variant: null, files: found.files ?? [] });
    }
    for (const row of rows.required) {
      fact(/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(row.repo) && SHA.test(row.revision ?? ""), `unpinned download row ${row.repo}@${row.revision}`);
      const selectors = row.files.length ? row.files : ["*"];
      const evidenceFiles = evidence.get(`${row.repo}@${row.revision}`) ?? null;
      for (const selector of selectors) {
        const files = selector.includes("*") ? (evidenceFiles ? expandSelector(selector, evidenceFiles) : []) : [selector];
        fact(files.length > 0, `cannot expand ${row.repo}@${row.revision}/${selector} without checked-in download-pattern-evidence`);
        for (const relpath of files) {
          if (relpath.split("/").some((segment) => !segment || segment === ".." || segment.startsWith("."))) continue;
          const key = `${row.repo}@${row.revision}:${relpath}`;
          if (!targets.has(key)) targets.set(key, { repo: row.repo, revision: row.revision, relpath });
        }
      }
    }
  }
  return [...targets.values()];
}

async function fetchPublicSnapshotFile(targetHub, { repo, revision, relpath }, signal) {
  const snap = snapshotDir(targetHub, repo, revision);
  const dest = path.join(snap, ...relpath.split("/"));
  try {
    const info = await stat(dest);
    if (info.isFile() && info.size > 0) return { status: "cached", bytes: info.size };
  } catch { /* absent */ }
  await mkdir(path.dirname(dest), { recursive: true });
  const refsDir = path.join(targetHub, `models--${repo.replace("/", "--")}`, "refs");
  await mkdir(refsDir, { recursive: true });
  await writeFile(path.join(refsDir, "main"), revision, { flag: "wx" }).catch(() => {});
  const url = `https://huggingface.co/${repo}/resolve/${revision}/${relpath.split("/").map(encodeURIComponent).join("/")}`;
  let lastErr = null;
  for (let attempt = 1; attempt <= 3; attempt++) {
    if (signal?.aborted) throw signal.reason ?? new Error("aborted");
    const tmp = `${dest}.part-${process.pid}-${attempt}`;
    try {
      const res = await fetch(url, { redirect: "follow", signal });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${repo}@${revision.slice(0, 12)}/${relpath}`);
      await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp));
      const info = await stat(tmp);
      if (relpath.endsWith(".safetensors") && info.size === 0) throw new Error(`zero-byte weight file ${relpath}`);
      await rename(tmp, dest);
      return { status: "downloaded", bytes: info.size };
    } catch (err) {
      lastErr = err;
      await rm(tmp, { force: true }).catch(() => {});
      if (attempt < 3) await delay(1500 * attempt);
    }
  }
  throw lastErr;
}

async function stageMissing(env, cfg, report, signal) {
  const platform = process.platform;
  const catalog = await readCatalog(cfg.sourceDir);
  const evidence = await readPatternEvidence(cfg.sourceDir);
  const files = resolveMissingSnapshotFiles({ routes: cfg.routes, plannedRoutes: report.routes, catalog, evidence, platform });
  report.staging = { targetHub: cfg.hubs[0], requestedFiles: files.length, downloadedFiles: 0, cachedFiles: 0, downloadedBytes: 0 };
  if (files.length === 0) return;
  await mkdir(cfg.hubs[0], { recursive: true });
  const hubFree = await freeBytes(cfg.hubs[0]);
  report.staging.hubFreeBytes = hubFree;
  fact(hubFree === null || hubFree >= 45 * GIB, `insufficient hub disk headroom on ${cfg.hubs[0]}: ${hubFree} bytes (< 45 GiB)`);
  process.stdout.write(`[pag-acceptance] staging ${files.length} public snapshot files into ${cfg.hubs[0]}\n`);
  for (const item of files) {
    const res = await fetchPublicSnapshotFile(cfg.hubs[0], item, signal);
    if (res.status === "downloaded") {
      report.staging.downloadedFiles++;
      report.staging.downloadedBytes += res.bytes;
      process.stdout.write(`[pag-acceptance] staged ${item.repo}@${item.revision.slice(0, 12)}/${item.relpath} (${(res.bytes / (1024 * 1024)).toFixed(1)} MiB)\n`);
    } else {
      report.staging.cachedFiles++;
    }
  }
  report.routes = await planPagRoutes({ routes: cfg.routes, hubs: cfg.hubs, catalog, platform, evidence });
}

// ---- single-checkout release build --------------------------------------------------------------
async function runLogged(command, args, { cwd, env, log }) {
  const handle = await open(log, "w");
  try {
    const child = spawn(command, args, { cwd, env, stdio: ["ignore", handle.fd, handle.fd], windowsHide: true });
    return await new Promise((resolve) => {
      child.once("error", (error) => resolve({ code: null, error: error.message }));
      child.once("exit", (code, signal) => resolve({ code, signal }));
    });
  } finally { await handle.close(); }
}

async function tailText(file, bytes = 64_000) {
  try { return (await readFile(file, "utf8")).slice(-bytes); } catch { return ""; }
}

async function buildRelease(env, cfg, report) {
  const scratch = path.join(cfg.work, "scratch"), target = path.join(scratch, "target"), evidence = path.join(cfg.work, "evidence");
  await mkdir(target, { recursive: true }); await mkdir(evidence, { recursive: true });
  const exe = process.platform === "win32" ? "sceneworks-rust-api.exe" : "sceneworks-rust-api";
  const args = [
    "build", "--release", "--locked", "-p", "sceneworks-rust-api",
    "--message-format=json-render-diagnostics",
    ...(cfg.backend === "cuda" ? ["--features", "backend-candle", "--bin", "sceneworks-rust-api"] : []),
  ];
  const log = path.join(scratch, "build.jsonl"), started = Date.now();
  const result = await runLogged("cargo", args, { cwd: cfg.sourceDir, env: { ...scrubSecrets(env), CARGO_TARGET_DIR: target }, log });
  if (result.code !== 0) {
    await writeFile(path.join(evidence, "build.tail.log"), await tailText(log, 64_000));
    throw new Error(`release build failed (${JSON.stringify(result)}); see build.tail.log`);
  }
  let metallib = null;
  if (cfg.backend === "mlx") {
    const native = await bindMetallib({
      messages: log,
      lockFile: path.join(cfg.sourceDir, "Cargo.lock"),
      targetRoot: target,
      temporaryRoot: scratch,
      consumer: "app",
      explicitOverride: env.PMETAL_METALLIB_PATH ?? "",
      explicitNativeDirectory: env.PMETAL_MLX_PREBUILT_DIR ?? "",
    });
    metallib = { path: native.path, sha256: native.sha256 };
  }
  const bin = path.join(scratch, "bin"); await mkdir(bin, { recursive: true });
  for (const entry of await readdir(path.join(target, "release"), { withFileTypes: true })) {
    if (entry.isFile() && (entry.name === exe || /\.(dll|dylib|so)$/i.test(entry.name))) {
      await copyFile(path.join(target, "release", entry.name), path.join(bin, entry.name));
    }
  }
  const binary = path.join(bin, exe);
  const built = { binary, binarySha256: await fileSha256(binary), metallib, buildSeconds: (Date.now() - started) / 1000 };
  report.source = { ...report.source, binarySha256: built.binarySha256, buildSeconds: built.buildSeconds, metallibSha256: metallib?.sha256 ?? null };
  await writeFile(path.join(scratch, "build.json"), JSON.stringify({ sourceSha: cfg.sourceSha, backend: cfg.backend, built }, null, 2) + "\n");
  return built;
}

// ---- hub-grouped execution (one API+worker per hub, running all ready routes sequentially) ------
async function teardownChildren(children) {
  const sentinel = children.find((c) => c.label === "sentinel");
  const exited = (c) => c.exitCode !== null || c.signalCode !== null;
  const within = async (c, ms) => { let t; try { return await Promise.race([c.done, new Promise((r) => { t = setTimeout(() => r(null), ms); })]); } finally { clearTimeout(t); } };
  for (const c of children) if (c.label !== "sentinel") c.stdin?.end();
  sentinel?.stdin?.end();
  for (const c of children.filter((item) => item.label !== "sentinel")) {
    if (!c.pid) continue;
    if (!exited(c)) await within(c, 45_000);
    if (!exited(c)) {
      if (process.platform === "win32") await execFile("taskkill", ["/PID", String(c.pid), "/T", "/F"], { timeout: 20_000 }).catch(() => {});
      else { try { process.kill(-c.pid, "SIGTERM"); } catch {} await within(c, 30_000); if (!exited(c)) { try { process.kill(-c.pid, "SIGKILL"); } catch {} } }
      await within(c, 10_000);
    }
  }
  if (sentinel && !exited(sentinel)) sentinel.kill();
}

function buildRequestBody(def, plan, projectId, refs, pagOverride) {
  const body = structuredClone(def.body);
  body.projectId = projectId;
  body.advanced = { ...body.advanced };
  if (plan.tier && TIER_TO_QUANTIZE[plan.tier] && !Object.hasOwn(body.advanced, "mlxQuantize")) {
    body.advanced.mlxQuantize = TIER_TO_QUANTIZE[plan.tier];
    body.advanced.mlxQuantizeExplicit = true;
  }
  if (def.references === "moss_single") {
    body.referenceAssetIds = [refs.primary];
  } else if (def.references === "multi_two") {
    body.referenceAssetIds = [refs.primary, refs.secondary];
  } else if (def.references === "source_single") {
    body.sourceAssetId = refs.primary;
  }
  if (pagOverride !== null && pagOverride !== undefined) {
    body.advanced.pag = structuredClone(pagOverride);
  }
  return body;
}

async function submitAndVerifyOneJob({ endpoint, projectId, projectPath, workerId, def, plan, refs, runLabel, pagOverride, evidenceDir }) {
  const body = buildRequestBody(def, plan, projectId, refs, pagOverride);
  const posted = Date.now();
  const response = await endpoint.exchange("POST", "/api/v1/image/jobs", body);
  if (response.status >= 300 || !response.body?.id) {
    return { label: runLabel, status: "error", error: `POST /api/v1/image/jobs refused (${response.status}): ${JSON.stringify(response.body).slice(0, 1000)}` };
  }
  const jobId = response.body.id;
  let claimedAtMs = null, maxProgress = 0, monotonicProgress = true, progressUpdates = 0;
  let completedJob = null;
  while (true) {
    const current = await endpoint.call("GET", `/api/v1/jobs/${jobId}`);
    if (claimedAtMs === null && jobClaimed(current)) claimedAtMs = Date.now();
    if (typeof current.progress === "number" && Number.isFinite(current.progress)) {
      if (current.progress + 1e-6 < maxProgress) monotonicProgress = false;
      if (current.progress > maxProgress) { maxProgress = current.progress; progressUpdates++; }
    }
    const decision = jobProgressDecision({
      job: current,
      postedMs: posted,
      claimedAtMs,
      nowMs: Date.now(),
      claimMinutes: def.claimMinutes,
      timeoutMinutes: def.timeoutMinutes,
    });
    if (decision.state === "stop") {
      return { label: runLabel, jobId, status: "error", error: decision.reason, diagnostics: jobDiagnostics(current) };
    }
    if (decision.state === "completed") { completedJob = current; break; }
    await delay(500);
  }
  const seconds = Number(((Date.now() - posted) / 1000).toFixed(2));
  // Wait for API asset persistence + library hydration.
  let persisted = null;
  for (let i = 0; i < 120; i++) {
    const current = await endpoint.call("GET", `/api/v1/jobs/${jobId}`);
    if (Array.isArray(current.result?.assets) && current.result.assets.length > 0 && current.result.assetWrites === undefined) {
      persisted = current;
      break;
    }
    await delay(500);
  }
  fact(persisted, `${def.id}/${runLabel}: asset persistence timed out`);
  fact(persisted.workerId === workerId, `${def.id}/${runLabel}: ran on unexpected worker ${persisted.workerId}`);
  const asset = persisted.result.assets[0];
  const relative = asset.file?.path;
  fact(typeof relative === "string" && !relative.split("/").includes(".."), "invalid saved asset path");
  const diskPath = path.resolve(projectPath, ...relative.split("/"));
  const savedBytes = await readFile(diskPath);
  await mkdir(evidenceDir, { recursive: true });
  const outPng = path.join(evidenceDir, `${def.id}-${runLabel}.png`);
  await writeFile(outPng, savedBytes);

  const { width, height, rgba } = decodePng(savedBytes);
  const stats = imageQualityStats(rgba);
  const share = readWorkflowShareFromPng(savedBytes);
  fact(share && typeof share === "object", `${def.id}/${runLabel}: saved PNG missing sceneworks:workflow iTXt chunk`);

  // Also read hydrated library asset to verify GenerationSet recipe round-trip.
  const libraryAssets = await endpoint.call("GET", `/api/v1/projects/${projectId}/assets`);
  const hydrated = Array.isArray(libraryAssets) ? libraryAssets.find((item) => item.id === asset.id) : null;
  const recipe = hydrated?.generationSet?.recipe ?? hydrated?.recipe ?? asset.recipe ?? null;
  const pagReport = recipe?.rawAdapterSettings?.pagExecutionReport ?? null;
  const sharePagReport = share?.advanced?.pagExecutionReport ?? null;
  const requestedRecipePag = recipe?.normalizedSettings?.pag ?? recipe?.rawAdapterSettings?.pag ?? null;
  const sharePag = share?.advanced?.pag ?? null;

  return {
    label: runLabel,
    jobId,
    status: "completed",
    seconds,
    width,
    height,
    fileSha256: sha256(savedBytes),
    contentSha256: contentDigest({ width, height, format: "rgba8" }, rgba),
    pngFile: path.basename(outPng),
    stats,
    pagReport,
    sharePagReport,
    requestedRecipePag,
    sharePag,
    progress: { maxProgress, progressUpdates, monotonicProgress },
    buffer: rgba,
  };
}

async function runAcceptance(env, cfg, report, signal) {
  const scratch = path.join(cfg.work, "scratch"), evidence = path.join(cfg.work, "evidence");
  const recorded = JSON.parse(await readFile(path.join(scratch, "build.json"), "utf8"));
  fact(recorded.sourceSha === cfg.sourceSha && recorded.backend === cfg.backend, "build.json mismatch");
  const built = recorded.built;
  fact((await fileSha256(built.binary)) === built.binarySha256, "binary digest changed since build");
  const secrets = new Set([env.HF_TOKEN].filter((v) => v && v.length >= 8));

  const readyByHub = new Map();
  for (const plan of report.routes.filter((r) => r.status === "ready")) {
    if (!readyByHub.has(plan.hub)) readyByHub.set(plan.hub, []);
    readyByHub.get(plan.hub).push(plan);
  }

  for (const [hub, hubPlans] of readyByHub.entries()) {
    if (signal.aborted) break;
    const sessionDir = path.join(scratch, "session-" + sha256(Buffer.from(hub)).slice(0, 8));
    const state = path.join(sessionDir, "state"), xet = path.join(sessionDir, "xet"), logs = path.join(sessionDir, "logs");
    for (const d of [path.join(state, "data", "projects"), path.join(state, "data", "loras"), path.join(state, "credentials"), xet, logs]) {
      await mkdir(d, { recursive: true });
    }
    await cp(path.join(cfg.sourceDir, "config"), path.join(state, "config"), { recursive: true });

    // Seed synthetic rank-4 Z-Image LoRA into state/data/loras + state/config/user.loras.jsonc for AT7.
    await writeFile(path.join(state, "data", "loras", "sw_f371_z_pag_lora.safetensors"), buildSyntheticZImageLoraSafetensors());
    await writeFile(
      path.join(state, "config", "user.loras.jsonc"),
      JSON.stringify({
        schemaVersion: 1,
        loras: [{
          id: "sw_f371_z_pag_lora",
          name: "SW-F371 Z-Image Block-15 PAG LoRA",
          family: "z-image",
          compatibility: { families: ["z-image"] },
          source: { provider: "local", path: "loras/sw_f371_z_pag_lora.safetensors" },
        }],
      }, null, 2) + "\n",
    );

    const workerId = `pag-accept-${process.pid}-${Date.now().toString(36)}`;
    const envs = childEnvironments(env, { backend: cfg.backend, state, hub, xet, workerId, metallib: built.metallib?.path });
    const children = [], handles = [], controller = new AbortController();
    const posix = process.platform !== "win32";
    const launch = async (name, command, args, childEnv) => {
      const log = await open(path.join(logs, `${name}.log`), "wx");
      handles.push(log);
      const child = spawn(command, args, { cwd: cfg.sourceDir, env: childEnv, detached: posix && name !== "sentinel", stdio: childStdio(name, log.fd), windowsHide: true });
      child.label = name;
      children.push(child);
      child.done = new Promise((r) => {
        child.once("error", (e) => r({ error: e.message }));
        child.once("exit", (code, sig) => r({ code, signal: sig }));
      });
      return child;
    };

    let endpoint;
    try {
      const sentinel = await launch("sentinel", process.execPath, ["-e", 'process.stdin.resume();process.stdin.on("end",()=>process.exit(0));'], env);
      envs.api.SCENEWORKS_PARENT_PID = String(sentinel.pid);
      const api = await launch("api", built.binary, [], envs.api);
      let url = null;
      for (let i = 0; i < 360 && !url; i++) {
        url = boundApiUrl(await readFile(path.join(logs, "api.log"), "utf8"));
        if (!url) await delay(500);
      }
      fact(url, "API bound-port announcement timed out");
      endpoint = ownedApiController({ url, state, api, controller });
      let healthy = false;
      for (let i = 0; i < 240 && !healthy; i++) {
        try { healthy = Boolean(await endpoint.verify()); } catch { healthy = false; }
        if (!healthy) await delay(500);
      }
      fact(healthy, "API readiness timed out");
      const worker = await endpoint.startWorker(() => launch("worker", built.binary, [], { ...envs.worker(url), SCENEWORKS_PARENT_PID: String(sentinel.pid) }));
      let registered = null;
      for (let i = 0; i < 480 && !registered; i++) {
        fact(worker.exitCode === null && worker.signalCode === null, `worker exited early (${worker.exitCode}/${worker.signalCode})`);
        const row = (await endpoint.call("GET", "/api/v1/workers")).find((w) => w.id === workerId);
        if (row?.status === "idle") registered = row;
        else await delay(500);
      }
      fact(registered, "worker idle registration timed out");

      // Acknowledge licenses for models that require them.
      for (const modelId of new Set(hubPlans.map((p) => PAG_ROUTE_MAP[p.id]).filter((d) => d.licenseAck).map((d) => d.model))) {
        const ack = await endpoint.exchange("PUT", `/api/v1/models/${modelId}/license-acknowledgment`, null);
        fact(ack.status < 300, `license acknowledgment failed for ${modelId} (${ack.status})`);
      }

      const project = await endpoint.call("POST", "/api/v1/projects", { name: "SW-F371 PAG Acceptance" });
      const projectPath = await realpath(project.path);

      const uploadPng = async (name, bytes) => {
        const form = new FormData();
        form.append("file", new Blob([bytes], { type: "image/png" }), name);
        const up = await endpoint.exchange("POST", `/api/v1/projects/${project.id}/assets`, form, { multipart: true });
        fact(up.status < 300 && up.body?.id, `reference upload ${name} failed (${up.status})`);
        return up.body.id;
      };
      const refs = {
        primary: await uploadPng("ref-primary.png", syntheticReferencePng("primary")),
        secondary: await uploadPng("ref-secondary.png", syntheticReferencePng("secondary")),
      };

      const pngOutDir = path.join(evidence, "pngs");
      for (const plan of hubPlans) {
        if (signal.aborted) break;
        const def = PAG_ROUTE_MAP[plan.id];
        process.stdout.write(`[pag-acceptance] ${new Date().toISOString()} route=${plan.id} model=${plan.model} tier=${plan.tier ?? "untiered"} phase=start\n`);
        plan.runs = [];
        const stripBuf = ({ buffer: _buf, ...rest }) => rest;
        try {
          if (def.kind === "t2i_ab") {
            const off1 = await submitAndVerifyOneJob({ endpoint, projectId: project.id, projectPath, workerId, def, plan, refs, runLabel: "off1", pagOverride: null, evidenceDir: pngOutDir });
            plan.runs.push(stripBuf(off1));
            const zero = off1.status === "completed"
              ? await submitAndVerifyOneJob({ endpoint, projectId: project.id, projectPath, workerId, def, plan, refs, runLabel: "zero", pagOverride: { scale: 0 }, evidenceDir: pngOutDir })
              : null;
            if (zero) plan.runs.push(stripBuf(zero));
            const pagOn = zero?.status === "completed"
              ? await submitAndVerifyOneJob({ endpoint, projectId: project.id, projectPath, workerId, def, plan, refs, runLabel: "pag_on", pagOverride: def.pagOn, evidenceDir: pngOutDir })
              : null;
            if (pagOn) plan.runs.push(stripBuf(pagOn));
            let verdict = evaluateT2iAbVerdict({ off1, zero, pagOn, expectedReport: def.expectedReport, pagScale: def.pagOn.scale });
            if (verdict.needOff2) {
              const off2 = await submitAndVerifyOneJob({ endpoint, projectId: project.id, projectPath, workerId, def, plan, refs, runLabel: "off2", pagOverride: null, evidenceDir: pngOutDir });
              plan.runs.push(stripBuf(off2));
              verdict = evaluateT2iAbVerdict({ off1, zero, pagOn, off2, expectedReport: def.expectedReport, pagScale: def.pagOn.scale });
            }
            plan.status = "ran";
            plan.verdict = verdict.verdict;
            plan.verdictReason = verdict.reason;
            plan.metrics = verdict.metrics ?? null;
          } else {
            const pagOn = await submitAndVerifyOneJob({ endpoint, projectId: project.id, projectPath, workerId, def, plan, refs, runLabel: "pag_on", pagOverride: def.pagOn, evidenceDir: pngOutDir });
            plan.runs.push(stripBuf(pagOn));
            if (pagOn.status !== "completed") {
              plan.status = "ran";
              plan.verdict = "error";
              plan.verdictReason = pagOn.error;
            } else {
              assertCoherentStats(pagOn.stats, `${plan.id}/pag_on`);
              assertPagReportMatches(pagOn.pagReport, def.expectedReport, def.pagOn.scale, `${plan.id} recipe pagExecutionReport`);
              assertPagReportMatches(pagOn.sharePagReport, def.expectedReport, def.pagOn.scale, `${plan.id} PNG workflow pagExecutionReport`);
              fact(pagOn.progress.monotonicProgress, `${plan.id}: non-monotonic job progress`);
              plan.status = "ran";
              plan.verdict = "passed";
              plan.verdictReason = `${def.at.join("/")} verified: coherent ${pagOn.width}x${pagOn.height} output (stdDev=${pagOn.stats.channelStdDev}), pagExecutionReport layers=[${pagOn.pagReport.layers.join(",")}] active=${pagOn.pagReport.activeSteps}/${pagOn.pagReport.totalSteps}`;
            }
          }
        } catch (err) {
          plan.status = "ran";
          plan.verdict = "error";
          plan.verdictReason = err.message;
        }
        process.stdout.write(`[pag-acceptance] ${new Date().toISOString()} route=${plan.id} verdict=${plan.verdict} reason=${plan.verdictReason}\n`);
        await writePagReport(cfg, report);
      }
    } finally {
      await teardownChildren(children);
      endpoint?.close();
      await Promise.all(handles.map((h) => h.close().catch(() => {})));
      for (const name of ["api", "worker"]) {
        const text = await tailText(path.join(logs, `${name}.log`), 512_000);
        if (text) await writeFile(path.join(evidence, `${name}.tail.log`), redactSensitiveText(text, secrets));
      }
    }
  }
}

export function summarizePagReport(routes, requiredFamilies = FAMILIES) {
  const familiesCovered = new Set();
  const familiesPassed = new Set();
  let ready = 0, ran = 0, passed = 0, skipped = 0, errors = 0;
  for (const r of routes) {
    if (r.status === "ready") { ready++; familiesCovered.add(r.family); }
    else if (r.status === "skipped") skipped++;
    else if (r.status === "ran") {
      ran++;
      familiesCovered.add(r.family);
      if (r.verdict === "passed" || r.verdict === "passed_with_runner_noise") {
        passed++;
        familiesPassed.add(r.family);
      } else {
        errors++;
      }
    }
  }
  const missingFamilies = requiredFamilies.filter((f) => !(ran > 0 ? familiesPassed : familiesCovered).has(f));
  return {
    totalRoutes: routes.length,
    ready,
    ran,
    passed,
    skipped,
    errors,
    requiredFamilies: [...requiredFamilies],
    familiesCovered: [...familiesCovered],
    familiesPassed: [...familiesPassed],
    missingFamilies,
  };
}

export function pagExitCode(report) {
  const requiredFamilies = report.requiredFamilies ?? FAMILIES;
  const s = summarizePagReport(report.routes ?? [], requiredFamilies);
  if (report.fatal) return 1;
  if (report.mode === "preflight" || report.mode === "stage") return s.ready > 0 && s.missingFamilies.length === 0 ? 0 : 1;
  if (report.mode === "build") return 0;
  return s.ran > 0 && s.errors === 0 && s.missingFamilies.length === 0 ? 0 : 1;
}

export function renderPagMarkdown(report) {
  const requiredFamilies = report.requiredFamilies ?? FAMILIES;
  const s = summarizePagReport(report.routes ?? [], requiredFamilies);
  const lines = [
    `# SW-F371 PAG Real-Weight Acceptance (${report.backend}) — ${report.mode}`,
    "",
    `- status: **${report.status}** (exit ${report.exitCode})`,
    `- source: \`${report.source?.sha}\` (inference \`${report.source?.inferencePin ?? "?"}\`, binary \`${report.source?.binarySha256 ?? "not built"}\`)`,
    `- runner: ${report.runner?.name ?? "local"} (${report.runner?.platform}/${report.runner?.arch})`,
    `- summary: passed=${s.passed}/${s.ran} ran (${s.skipped} skipped, ${s.errors} errors); families=${s.familiesPassed.length || s.familiesCovered.length}/${requiredFamilies.length} (missing: ${s.missingFamilies.join(", ") || "none"})`,
    "",
    "| Route | Family | Model (Tier) | ATs | Status | Verdict | Details |",
    "|---|---|---|---|---|---|---|",
  ];
  for (const r of report.routes ?? []) {
    lines.push(`| ${r.id} | ${r.family} | ${r.model}${r.tier ? ` (${r.tier})` : ""} | ${(r.at ?? []).join(",")} | ${r.status} | ${r.verdict ?? "–"} | ${String(r.verdictReason ?? r.skipReason ?? "").replace(/\|/g, "/").slice(0, 320)} |`);
  }
  if (report.fatal) lines.push("", `**Fatal:** ${report.fatal}`);
  return lines.join("\n") + "\n";
}

async function writePagReport(cfg, report) {
  const dir = path.join(cfg.work, "evidence");
  await mkdir(dir, { recursive: true });
  report.summary = summarizePagReport(report.routes ?? [], report.requiredFamilies ?? FAMILIES);
  const baseName = report.mode === "run" ? "report" : report.mode;
  await writeFile(path.join(dir, `${baseName}.json`), JSON.stringify(report, null, 2) + "\n");
  await writeFile(path.join(dir, `${baseName}.md`), renderPagMarkdown(report));
}

async function cleanup(env, cfg) {
  const allowedRoots = [env.RUNNER_TEMP, env.GITHUB_WORKSPACE].filter(Boolean);
  fact(allowedRoots.length > 0, "--cleanup requires RUNNER_TEMP or GITHUB_WORKSPACE");
  const protectedRoots = [...cfg.hubs, ...pagDefaultHubs(cfg.backend)];
  let failed = 0;
  for (const target of [cfg.work, cfg.sourceDir]) {
    try { await removeScratch(target, { allowedRoots, protectedRoots }); }
    catch { failed++; }
  }
  return failed ? 1 : 0;
}

export async function main(argv = process.argv.slice(2), env = process.env) {
  const modes = { "--preflight-only": "preflight", "--stage-missing": "stage", "--build-only": "build", "--run": "run", "--cleanup": "cleanup" };
  fact(argv.length === 1 && modes[argv[0]], "exactly one of --preflight-only, --stage-missing, --build-only, --run, --cleanup is required");
  const mode = modes[argv[0]];
  const cfg = resolvePagConfig(env);
  if (mode === "cleanup") return cleanup(env, cfg);
  const explicitRoutes = Boolean((env.PAG_ACCEPTANCE_ROUTES ?? "").trim() && (env.PAG_ACCEPTANCE_ROUTES ?? "").trim() !== "default");
  const requiredFamilies = explicitRoutes ? [...new Set(cfg.routes.map((r) => r.family))] : [...FAMILIES];
  const report = {
    schemaVersion: 1,
    kind: "sw-f371-pag-acceptance",
    mode,
    status: "running",
    backend: cfg.backend,
    startedAt: new Date().toISOString(),
    thresholds: ACCEPTANCE_THRESHOLDS,
    requiredFamilies,
    runner: {
      name: env.RUNNER_NAME ?? null,
      platform: process.platform,
      arch: process.arch,
      totalMemoryBytes: os.totalmem(),
      runId: env.GITHUB_RUN_ID ?? null,
      runAttempt: env.GITHUB_RUN_ATTEMPT ?? null,
    },
    source: { sha: cfg.sourceSha },
    routes: [],
  };
  const controller = new AbortController();
  const interrupt = () => controller.abort(new Error("interrupted by signal"));
  process.once("SIGTERM", interrupt);
  process.once("SIGINT", interrupt);
  try {
    await preflight(env, cfg, report);
    if (mode === "stage") await stageMissing(env, cfg, report, controller.signal);
    if (mode === "build") await buildRelease(env, cfg, report);
    if (mode === "run") await runAcceptance(env, cfg, report, controller.signal);
    report.status = "finished";
  } catch (err) {
    report.status = "failed";
    report.fatal = err.message;
  } finally {
    process.removeListener("SIGTERM", interrupt);
    process.removeListener("SIGINT", interrupt);
    report.finishedAt = new Date().toISOString();
    report.exitCode = pagExitCode(report);
    await writePagReport(cfg, report).catch((e) => { process.stderr.write(`report write failed: ${e.message}\n`); report.exitCode = 1; });
  }
  process.stdout.write(renderPagMarkdown(report));
  return report.exitCode;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().then((code) => { process.exitCode = code; }).catch((err) => { process.stderr.write(`${err.stack ?? err.message}\n`); process.exitCode = 1; });
}
