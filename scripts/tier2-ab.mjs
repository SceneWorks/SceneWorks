#!/usr/bin/env node
// SW-7033 (feature SW-F348): dispatch-only Tier 2 base/head A/B over the PRODUCTION path.
//
// For each selected route (backend + model + mode + one fixed request and seed) this driver starts
// the release API and a separate worker built from each checkout, POSTs the SAME request JSON,
// lets the worker claim, load and render it, reads the saved asset back through the product file
// route, and compares base#1 / head / base#2 on the same runner against the same model hub. It is
// reached only from the `tier2-ab` profile of memory-catalog-campaign.yml (workflow_dispatch); no
// push, PR or merge trigger can reach it, and ordinary CI runs only its pure-function tests.
//
// Stop rule (REQUIREMENTS DEC-7): head == base#1 -> `equal`, stop. Otherwise base#2 runs once:
// base#1 != base#2 -> `nondeterministic_on_runner` (numeric difference recorded once, no more
// runs); base#1 == base#2 -> `head_differs` (non-zero exit, payload diff printed). Time and memory
// are observations and never fail the job. A route whose weights are absent from every hub root is
// `skipped` with its reason; nothing is ever downloaded and the hub is never written or deleted.
//
// Modes: --preflight-only (inputs, both checkouts, pins, tools, hub presence; no build, no model)
//        --build-only     (release API/worker for base then head into one shared scratch target)
//        --run            (the comparison; requires --build-only output from the same work dir)
// Configuration is environment-only so the same invocation works from bash, cmd and PowerShell:
//   TIER2_AB_BACKEND mlx|cuda, TIER2_AB_BASE_SHA/HEAD_SHA (40 hex), TIER2_AB_BASE_DIR/HEAD_DIR,
//   TIER2_AB_WORK (absolute, under RUNNER_TEMP in Actions), TIER2_AB_ROUTES (empty = defaults,
//   or a JSON array / comma list of route ids), TIER2_AB_HUBS (empty = backend defaults, or a JSON
//   array / ';'-separated list of hub roots), TIER2_AB_INFERENCE_SHA (optional cross-check).
import assert from "node:assert/strict";
import { execFile as execFileCallback, spawn } from "node:child_process";
import { chmod, copyFile, cp, lstat, mkdir, open, readFile, readdir, realpath, rm, stat, statfs, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { inflateSync } from "node:zlib";
import { createHash } from "node:crypto";
import { fileSha256, sha256 } from "./lib/file-sha256.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";
import { lockedInferenceRevision } from "./select-windows-cargo-cache.mjs";
import { buildChildEnv } from "./qwen-image-2-1-app-lifecycle.mjs";
import { bindOwnedEnv, boundApiUrl, ownedApiController, redactSensitiveText } from "./flux3-image-klein-app-lifecycle.mjs";
import { bindMetallib } from "./ci/memory-catalog/bind-metallib.mjs";

const execFile = promisify(execFileCallback);
const fact = (value, message) => assert.ok(value, message);
const GIB = 1024 ** 3;
export const SHA = /^[0-9a-f]{40}$/;
export const BACKENDS = Object.freeze(["mlx", "cuda"]);
export const RUN_LABELS = Object.freeze(["base#1", "head", "base#2"]);
export const VERDICTS = Object.freeze(["equal", "nondeterministic_on_runner", "head_differs", "error", "skipped"]);
// The physical CUDA card every existing real-weight CUDA acceptance lane uses (load-progress,
// catalog walk): logical CUDA 0 masked to physical GPU1. The workflow sets the masks; preflight
// refuses anything else.
export const CUDA_PHYSICAL_GPU = "1";
export const MLX_RUNNER = "nax-macos-2";
export const CUDA_RUNNERS = Object.freeze(["cuda-windows", "cuda-windows-2"]);

const PROMPT_IMAGE = "A red glass apple beside a blue ceramic cup, studio photograph";
// Request bodies are fixed per route; `projectId` (and the uploaded reference id) are the only
// per-run values. Each body is a request a shipped acceptance driver already sends at both SHAs:
// flux1-owner-worker-proof (FLUX.1), flux3 app lifecycle (Klein, smaller size), load-progress
// (MOSS-Image, Chatterbox), sdxl admitted replay (SDXL q4); the LTX-2.5 body uses the shortest
// declared duration/fps of the unchanged catalog entry and the worker's default distilled variant.
const IMAGE_ROUTE = (id, backend, model, tier, group, body) => ({ id, backend, model, kind: "image", mode: "text_to_image", tier, group, endpoint: "/api/v1/image/jobs", claimMinutes: 5, timeoutMinutes: 20, body });
export const ROUTES = Object.freeze(Object.fromEntries([
  IMAGE_ROUTE("mlx-flux1-dev-t2i", "mlx", "flux_dev", "q4", "mlx-flux-t2i", { mode: "text_to_image", model: "flux_dev", prompt: PROMPT_IMAGE, negativePrompt: "", seed: 33111, width: 768, height: 768, count: 1, advanced: { steps: 4, sampler: "euler", guidanceScale: 3.5 }, loras: [] }),
  IMAGE_ROUTE("mlx-flux3-klein-t2i", "mlx", "flux3_klein", "bf16", "mlx-flux-t2i", { mode: "text_to_image", model: "flux3_klein", prompt: "A small teal ceramic teapot on a pale yellow table, soft daylight, studio photograph", negativePrompt: "", seed: 26234, width: 512, height: 512, count: 1, advanced: { steps: 4, sampler: "euler", guidanceScale: 0 }, loras: [] }),
  IMAGE_ROUTE("mlx-moss-image-t2i", "mlx", "moss_image_sft", null, null, { mode: "text_to_image", model: "moss_image_sft", prompt: "A red ceramic cup on a white table", width: 256, height: 256, count: 1, seed: 26325, advanced: { steps: 2, guidanceScale: 1.0, mossImage: { acceleration: "off", max_sequence_length: 64 } } }),
  { id: "mlx-ltx25-distilled-t2v", backend: "mlx", model: "ltx_2_5", kind: "video", mode: "text_to_video", tier: "q4", group: null, endpoint: "/api/v1/video/jobs", claimMinutes: 10, timeoutMinutes: 45, body: { mode: "text_to_video", model: "ltx_2_5", prompt: "A paper boat drifting across a calm pond, gentle morning light", negativePrompt: "", duration: 4, fps: 24, width: 768, height: 512, seed: 25001, loras: [], advanced: { transformerVariant: "distilled" } } },
  { id: "mlx-chatterbox-tts", backend: "mlx", model: "chatterbox_tts", kind: "audio", mode: "text_to_speech", tier: null, group: null, endpoint: "/api/v1/audio/jobs", claimMinutes: 5, timeoutMinutes: 15, referenceAudio: true, body: { model: "chatterbox_tts", prompt: "The red cup is on the table.", seed: 27980 } },
  IMAGE_ROUTE("cuda-sdxl-q4-t2i", "cuda", "sdxl", "q4", null, { model: "sdxl", mode: "text_to_image", prompt: "a photorealistic red apple on a wooden table, studio lighting", width: 1024, height: 1024, count: 1, seed: 16402, loras: [], advanced: { mlxQuantize: 4 } }),
  IMAGE_ROUTE("cuda-moss-image-t2i", "cuda", "moss_image_sft", null, null, { mode: "text_to_image", model: "moss_image_sft", prompt: "A red ceramic cup on a white table", width: 256, height: 256, count: 1, seed: 26325, advanced: { steps: 2, guidanceScale: 1.0, mossImage: { acceleration: "off", max_sequence_length: 64 } } }),
  // CUDA video group (first present wins). LTX-2.3 maps to the Candle `ltx_2_3_distilled` provider,
  // which registers its component footprint, so the worker's LTX admission can pass; Wan2.2 TI2V-5B
  // maps to `wan2_2_ti2v_5b` (no LTX admission). Both have measured Candle memory anchors
  // (config/memory-anchors.json: sc-20772 LTX-2.3 q4, sc-19223 Wan2.2 TI2V-5B q4), so their weights
  // have been on the Windows box. Short clips: LTX-2.3 768x512, 4 s at 24 fps (97 frames), 8 steps;
  // Wan2.2 TI2V-5B 832x480, 4 s at 16 fps (65 frames), 8 steps. GPU memory is the same nvidia-smi
  // physical-GPU1 peak above the pre-request baseline as every CUDA route.
  { id: "cuda-ltx23-distilled-t2v", backend: "cuda", model: "ltx_2_3", kind: "video", mode: "text_to_video", tier: "q4", group: "cuda-video", endpoint: "/api/v1/video/jobs", claimMinutes: 10, timeoutMinutes: 45, body: { mode: "text_to_video", model: "ltx_2_3", prompt: "A paper boat drifting across a calm pond, gentle morning light", negativePrompt: "", duration: 4, fps: 24, width: 768, height: 512, seed: 25001, loras: [], advanced: { steps: 8 } } },
  { id: "cuda-wan22-ti2v-5b-t2v", backend: "cuda", model: "wan_2_2", kind: "video", mode: "text_to_video", tier: "q4", group: "cuda-video", endpoint: "/api/v1/video/jobs", claimMinutes: 10, timeoutMinutes: 45, body: { mode: "text_to_video", model: "wan_2_2", prompt: "A paper boat drifting across a calm pond, gentle morning light", negativePrompt: "", duration: 4, fps: 16, width: 832, height: 480, seed: 25001, loras: [], advanced: { steps: 8 } } },
  // Not a default: at inference pin 4a36a7f8 the Candle `ltx_2_5_distilled` provider registers no
  // component footprint (candle-gen-ltx lib.rs REGISTRATION_25), so the worker refuses every
  // LTX-2.5 request at admission (video_jobs/candle.rs ltx_resident_weight_bytes) on base and head
  // alike. Kept selectable for the run after that engine fix.
  { id: "cuda-ltx25-distilled-t2v", backend: "cuda", model: "ltx_2_5", kind: "video", mode: "text_to_video", tier: "q4", group: null, endpoint: "/api/v1/video/jobs", claimMinutes: 10, timeoutMinutes: 45, body: { mode: "text_to_video", model: "ltx_2_5", prompt: "A paper boat drifting across a calm pond, gentle morning light", negativePrompt: "", duration: 4, fps: 24, width: 768, height: 512, seed: 25001, loras: [], advanced: { transformerVariant: "distilled" } } },
].map((route) => [route.id, Object.freeze(route)])));
export const DEFAULT_ROUTE_IDS = Object.freeze({
  mlx: Object.freeze(["mlx-flux1-dev-t2i", "mlx-flux3-klein-t2i", "mlx-moss-image-t2i", "mlx-ltx25-distilled-t2v", "mlx-chatterbox-tts"]),
  cuda: Object.freeze(["cuda-sdxl-q4-t2i", "cuda-moss-image-t2i", "cuda-ltx23-distilled-t2v", "cuda-wan22-ti2v-5b-t2v"]),
});
// Read-only hub roots tried in order per route. The first root holding every required file wins.
export function defaultHubs(backend, home = os.homedir()) {
  if (backend === "cuda") return ["E:\\huggingface\\hub"];
  return ["/Volumes/Models/huggingface/hub", path.posix.join(home, ".cache/huggingface/hub"), path.posix.join(home, "sceneworks-flux1-owner-proof-weights/hub"), path.posix.join(home, "sceneworks-flux3-app-weights/hub")];
}

const pathsFor = (platform) => (platform === "win32" ? path.win32 : path.posix);
// Drive-qualified on Windows (a root-relative "\x" is not accepted), rooted on POSIX.
export const absoluteFor = (platform, value) => typeof value === "string" && !/[\r\n]/.test(value) && (platform === "win32" ? /^[A-Za-z]:[\\/]/.test(value) : value.startsWith("/"));
function listInput(value, separator) {
  const text = (value ?? "").trim();
  if (!text || text === "default") return null;
  if (text.startsWith("[")) {
    const parsed = JSON.parse(text);
    fact(Array.isArray(parsed) && parsed.every((item) => typeof item === "string" && item.trim()), "list input must be a JSON array of non-empty strings");
    return parsed.map((item) => item.trim());
  }
  return text.split(separator).map((item) => item.trim()).filter(Boolean);
}
export function selectRoutes(backend, value) {
  fact(BACKENDS.includes(backend), `backend must be one of ${BACKENDS.join("|")}`);
  const ids = listInput(value, /[,\s]+/) ?? [...DEFAULT_ROUTE_IDS[backend]];
  fact(ids.length > 0 && new Set(ids).size === ids.length, "route list must be non-empty and unique");
  for (const id of ids) {
    fact(Object.hasOwn(ROUTES, id), `unknown route ${id}; known: ${Object.keys(ROUTES).join(", ")}`);
    fact(ROUTES[id].backend === backend, `route ${id} belongs to the ${ROUTES[id].backend} backend, not ${backend}`);
  }
  return ids.map((id) => ROUTES[id]);
}
export function resolveConfig(env, { platform = process.platform, home = os.homedir() } = {}) {
  const paths = pathsFor(platform);
  const backend = env.TIER2_AB_BACKEND;
  fact(BACKENDS.includes(backend), `TIER2_AB_BACKEND must be one of ${BACKENDS.join("|")}`);
  const baseSha = env.TIER2_AB_BASE_SHA ?? "", headSha = env.TIER2_AB_HEAD_SHA ?? "";
  fact(SHA.test(baseSha) && SHA.test(headSha), "TIER2_AB_BASE_SHA and TIER2_AB_HEAD_SHA must be exact 40-hex lowercase commit SHAs; branch names are refused");
  fact(baseSha !== headSha, "base and head SHAs must differ");
  const inferenceSha = (env.TIER2_AB_INFERENCE_SHA ?? "").trim();
  fact(inferenceSha === "" || SHA.test(inferenceSha), "TIER2_AB_INFERENCE_SHA must be empty or 40-hex");
  const absolute = (key) => { const value = env[key] ?? ""; fact(absoluteFor(platform, value), `${key} must be an absolute path`); return paths.resolve(value); };
  const baseDir = absolute("TIER2_AB_BASE_DIR"), headDir = absolute("TIER2_AB_HEAD_DIR"), work = absolute("TIER2_AB_WORK");
  fact(baseDir !== headDir, "base and head checkouts must be separate directories");
  for (const dir of [baseDir, headDir]) {
    const rel = paths.relative(dir, work);
    fact(rel.startsWith("..") || paths.isAbsolute(rel), "work directory must be outside both checkouts");
  }
  if (env.GITHUB_ACTIONS === "true") {
    fact(env.RUNNER_TEMP && paths.isAbsolute(env.RUNNER_TEMP), "RUNNER_TEMP required in Actions");
    const rel = paths.relative(paths.resolve(env.RUNNER_TEMP), work);
    fact(rel && !rel.startsWith("..") && !paths.isAbsolute(rel), "TIER2_AB_WORK must be under RUNNER_TEMP in Actions");
  }
  const routes = selectRoutes(backend, env.TIER2_AB_ROUTES);
  const hubs = listInput(env.TIER2_AB_HUBS, /[;\r\n]+/) ?? defaultHubs(backend, home);
  fact(hubs.length > 0 && hubs.every((hub) => absoluteFor(platform, hub)), "every hub root must be an absolute path");
  // Optional operator knob: one claim bound (minutes) for every route, e.g. to fail fast.
  const claimText = (env.TIER2_AB_CLAIM_MINUTES ?? "").trim();
  const claimMinutesOverride = claimText === "" ? null : Number(claimText);
  fact(claimMinutesOverride === null || (Number.isFinite(claimMinutesOverride) && claimMinutesOverride > 0 && claimMinutesOverride <= 60), "TIER2_AB_CLAIM_MINUTES must be a number in (0, 60]");
  return { backend, baseSha, headSha, inferenceSha, baseDir, headDir, work, routes, hubs: hubs.map((hub) => paths.resolve(hub)), claimMinutesOverride };
}

// Pure selector for the registered dispatcher (memory-catalog-campaign.yml, profile tier2-ab).
// It cannot build, start or load anything. moss_options_json carries the profile's options
// because the dispatcher is at GitHub's 25-input ceiling (same convention as load-progress).
export const DISPATCH_OPTION_KEYS = Object.freeze(["base_sha", "head_sha", "routes", "hubs", "preflight_only"]);
export function dispatchOptions(inputs, dispatchedSha) {
  assert.equal(inputs.profile, "tier2-ab");
  fact(SHA.test(dispatchedSha ?? "") && inputs.ref === dispatchedSha, "ref must equal the exact dispatched source SHA (the driver's own commit)");
  fact(["mlx", "candle"].includes(inputs.backend), "backend must be mlx or candle");
  fact(inputs.download_missing !== true && inputs.qwen_persistent_weights !== true, "tier2-ab never downloads; leave download_missing and qwen_persistent_weights off");
  let options;
  try { options = JSON.parse(inputs.moss_options_json || "{}"); } catch { throw new Error("moss_options_json must be a JSON object"); }
  fact(options && typeof options === "object" && !Array.isArray(options), "tier2-ab options must be a JSON object");
  const unknown = Object.keys(options).filter((key) => !DISPATCH_OPTION_KEYS.includes(key));
  fact(unknown.length === 0, `unknown tier2-ab option(s): ${unknown.join(", ")}`);
  fact(SHA.test(options.base_sha ?? "") && SHA.test(options.head_sha ?? "") && options.base_sha !== options.head_sha, "base_sha and head_sha must be distinct exact 40-hex SHAs");
  fact(options.preflight_only === undefined || typeof options.preflight_only === "boolean", "preflight_only must be boolean");
  fact(!inputs.inference_sha || SHA.test(inputs.inference_sha), "inference_sha must be empty or exact 40-hex");
  const backend = inputs.backend === "mlx" ? "mlx" : "cuda";
  for (const key of ["routes", "hubs"]) fact(options[key] === undefined || (Array.isArray(options[key]) && options[key].every((item) => typeof item === "string" && item.trim())), `${key} must be an array of strings`);
  const routes = selectRoutes(backend, options.routes ? JSON.stringify(options.routes) : "").map((route) => route.id);
  if (options.hubs) fact(options.hubs.length > 0 && options.hubs.every((hub) => absoluteFor(backend === "mlx" ? "darwin" : "win32", hub)), "hubs must be absolute paths for the selected backend's OS");
  return { base_sha: options.base_sha, head_sha: options.head_sha, backend, routes: options.routes ? JSON.stringify(routes) : "", hubs: options.hubs ? JSON.stringify(options.hubs) : "", preflight_only: options.preflight_only ?? true, inference_sha: inputs.inference_sha || "" };
}

// ---- catalog rows and hub presence (metadata only: stat/readdir, no hashing, no model reads) ----
const PLATFORM_NAMES = { darwin: "macos", win32: "windows", linux: "linux" };
export function routeDownloadRows(model, { platform, tier }) {
  fact(Array.isArray(model?.downloads) && model.downloads.length > 0, `catalog entry ${model?.id} has no downloads`);
  const name = PLATFORM_NAMES[platform];
  const rows = model.downloads.filter((row) => row.provider === "huggingface" && !row.localDerivation && (!row.platforms || row.platforms.includes(name)));
  const variantRows = rows.filter((row) => row.variant);
  const wanted = tier ?? variantRows.find((row) => row.default)?.variant ?? null;
  if (variantRows.length) fact(variantRows.some((row) => row.variant === wanted), `${model.id} has no ${wanted} download on ${name}`);
  const selected = rows.filter((row) => !row.variant || row.variant === wanted);
  const strip = ({ repo, revision, variant = null, files = [], componentId = null, coRequisite = false, required = null }) => ({ repo, revision, variant, files, componentId, coRequisite, required });
  return { tier: wanted, required: selected.filter((row) => row.required !== "soft").map(strip), optional: selected.filter((row) => row.required === "soft").map(strip) };
}
export function snapshotDir(hub, repo, revision, paths = path) {
  fact(/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(repo) && SHA.test(revision), "pinned repo/revision required");
  return paths.join(hub, `models--${repo.replace("/", "--")}`, "snapshots", revision);
}
const globRegex = (pattern) => new RegExp(`^${pattern.split("*").map((part) => part.replace(/[.+?^${}()|[\]\\]/g, "\\$&")).join("[^/]*")}$`);
export async function selectorPresent(snapshot, selector, fs = { stat, readdir }) {
  const nonEmptyDir = async (dir) => { try { return (await fs.readdir(dir)).length > 0; } catch { return false; } };
  if (selector === "*" ) return nonEmptyDir(snapshot);
  if (!selector.includes("*")) { try { const info = await fs.stat(path.join(snapshot, ...selector.split("/"))); return info.isFile() && info.size > 0; } catch { return false; } }
  const parts = selector.split("/"), last = parts.pop();
  fact(!parts.some((part) => part.includes("*")), `unsupported selector ${selector}`);
  const dir = path.join(snapshot, ...parts);
  if (last === "*") return nonEmptyDir(dir);
  try { const re = globRegex(last); return (await fs.readdir(dir)).some((entry) => re.test(entry)); } catch { return false; }
}
export async function routePresence(hub, rows, fs) {
  const check = async (list) => {
    const missing = [];
    for (const row of list) {
      const snapshot = snapshotDir(hub, row.repo, row.revision);
      const selectors = row.files.length ? row.files : ["*"];
      for (const selector of selectors) if (!(await selectorPresent(snapshot, selector, fs))) missing.push(`${row.repo}@${row.revision.slice(0, 12)}/${selector}`);
    }
    return missing;
  };
  const missing = await check(rows.required), optionalMissing = await check(rows.optional);
  return { hub, complete: missing.length === 0, missing, optionalMissing };
}
export async function readCatalog(checkout) {
  return JSON.parse(stripJsoncComments(await readFile(path.join(checkout, "config", "manifests", "builtin.models.jsonc"), "utf8"))).models;
}
// Pick, per route, the first hub holding every required file; route groups are alternatives
// (the first present member runs, the rest are skipped naming it).
export async function planRoutes({ routes, hubs, baseCatalog, headCatalog, platform, fs }) {
  const planned = [], ranGroups = new Map();
  for (const route of routes) {
    const entry = { id: route.id, model: route.model, kind: route.kind, mode: route.mode, group: route.group, request: requestTemplate(route) };
    const baseModel = baseCatalog.find((model) => model.id === route.model), headModel = headCatalog.find((model) => model.id === route.model);
    if (!baseModel || !headModel) { planned.push({ ...entry, status: "skipped", skipReason: `model ${route.model} is not in the ${!baseModel ? "base" : "head"} catalog` }); continue; }
    let rows, baseRows;
    try { rows = routeDownloadRows(headModel, { platform, tier: route.tier }); baseRows = routeDownloadRows(baseModel, { platform, tier: route.tier }); } catch (error) { planned.push({ ...entry, status: "skipped", skipReason: error.message }); continue; }
    if (JSON.stringify(baseRows) !== JSON.stringify(rows)) { planned.push({ ...entry, status: "skipped", skipReason: `pinned download rows for ${route.model} differ between base and head catalogs`, rows: { base: baseRows, head: rows } }); continue; }
    if (route.group && ranGroups.has(route.group)) { planned.push({ ...entry, tier: rows.tier, status: "skipped", skipReason: `alternative ${ranGroups.get(route.group)} of group ${route.group} is selected` }); continue; }
    const checks = [];
    for (const hub of hubs) { const result = await routePresence(hub, rows, fs); checks.push(result); if (result.complete) break; }
    const selected = checks.find((result) => result.complete);
    if (!selected) { planned.push({ ...entry, tier: rows.tier, rows, presence: checks, status: "skipped", skipReason: `weights absent from every hub root (${checks.map((c) => `${c.hub}: ${c.missing.slice(0, 4).join(", ")}${c.missing.length > 4 ? ` +${c.missing.length - 4} more` : ""}`).join("; ")})` }); continue; }
    if (route.group) ranGroups.set(route.group, route.id);
    planned.push({ ...entry, tier: rows.tier, rows, presence: checks, hub: selected.hub, status: "ready" });
  }
  return planned;
}
export function requestTemplate(route) {
  const body = { projectId: "<project>", ...(route.referenceAudio ? { referenceAudioAssetId: "<reference-audio>" } : {}), ...route.body };
  return { endpoint: route.endpoint, body, sha256: sha256(JSON.stringify(body)) };
}
export function requestBody(route, projectId, referenceAudioAssetId) {
  return { projectId, ...(route.referenceAudio ? { referenceAudioAssetId } : {}), ...structuredClone(route.body) };
}

// ---- decoded-content fidelity -------------------------------------------------------------------
// PNG -> RGBA8. Saved PNGs carry a workflow iTXt/tEXt chunk (workflow_png.rs) that can differ per
// job, so the file hash alone cannot compare runs; the decoded pixels can.
export function decodePng(buffer) {
  fact(buffer.length > 8 && buffer.subarray(0, 8).toString("hex") === "89504e470d0a1a0a", "not a PNG");
  let offset = 8, header, palette, transparency; const idat = [];
  while (offset + 8 <= buffer.length) {
    const length = buffer.readUInt32BE(offset), type = buffer.toString("ascii", offset + 4, offset + 8), data = buffer.subarray(offset + 8, offset + 8 + length);
    fact(data.length === length, "truncated PNG chunk");
    if (type === "IHDR") header = { width: data.readUInt32BE(0), height: data.readUInt32BE(4), depth: data[8], color: data[9], interlace: data[12] };
    else if (type === "PLTE") palette = data;
    else if (type === "tRNS") transparency = data;
    else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    offset += 12 + length;
  }
  fact(header && idat.length, "PNG has no IHDR/IDAT");
  const { width, height, depth, color, interlace } = header;
  const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[color];
  fact(channels && depth === 8 && interlace === 0, `unsupported PNG layout (color ${color}, depth ${depth}, interlace ${interlace}); 8-bit non-interlaced only`);
  const raw = inflateSync(Buffer.concat(idat)), stride = width * channels;
  fact(raw.length === height * (stride + 1), "PNG scanline data length mismatch");
  const pixels = Buffer.alloc(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)], row = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1)), out = pixels.subarray(y * stride, (y + 1) * stride), prior = y ? pixels.subarray((y - 1) * stride, y * stride) : null;
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? out[x - channels] : 0, b = prior ? prior[x] : 0, c = prior && x >= channels ? prior[x - channels] : 0;
      let predictor = 0;
      if (filter === 1) predictor = a; else if (filter === 2) predictor = b; else if (filter === 3) predictor = (a + b) >> 1;
      else if (filter === 4) { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); predictor = pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      else fact(filter === 0, `invalid PNG filter ${filter}`);
      out[x] = (row[x] + predictor) & 0xff;
    }
  }
  const rgba = Buffer.alloc(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const s = pixels.subarray(i * channels, (i + 1) * channels), o = i * 4;
    if (color === 0) { rgba[o] = rgba[o + 1] = rgba[o + 2] = s[0]; rgba[o + 3] = 255; }
    else if (color === 4) { rgba[o] = rgba[o + 1] = rgba[o + 2] = s[0]; rgba[o + 3] = s[1]; }
    else if (color === 2) { rgba[o] = s[0]; rgba[o + 1] = s[1]; rgba[o + 2] = s[2]; rgba[o + 3] = 255; }
    else if (color === 6) s.copy(rgba, o);
    else { fact(palette && s[0] * 3 + 2 < palette.length, "PNG palette index out of range"); palette.copy(rgba, o, s[0] * 3, s[0] * 3 + 3); rgba[o + 3] = transparency && s[0] < transparency.length ? transparency[s[0]] : 255; }
  }
  return { width, height, rgba };
}
export function parseWav(buffer) {
  fact(buffer.length >= 12 && buffer.toString("ascii", 0, 4) === "RIFF" && buffer.toString("ascii", 8, 12) === "WAVE", "not a RIFF/WAVE file");
  let offset = 12, format, data;
  while (offset + 8 <= buffer.length) {
    const id = buffer.toString("ascii", offset, offset + 4), size = buffer.readUInt32LE(offset + 4), body = buffer.subarray(offset + 8, offset + 8 + size);
    if (id === "fmt ") format = { audioFormat: body.readUInt16LE(0), channels: body.readUInt16LE(2), sampleRate: body.readUInt32LE(4), bitsPerSample: body.readUInt16LE(14) };
    else if (id === "data") data = body;
    offset += 8 + size + (size % 2);
  }
  fact(format && data, "WAV has no fmt/data chunk");
  return { ...format, data };
}
// Digest of decoded content plus its shape, so equal bytes at different shapes never collide.
export const contentDigest = (shape, ...buffers) => { const hash = createHash("sha256").update(JSON.stringify(shape)); for (const buffer of buffers) hash.update(buffer); return hash.digest("hex"); };
async function ffmpegDecode(ffmpeg, file, kind) {
  const args = kind === "video-frames" ? ["-v", "error", "-nostdin", "-i", file, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"] : ["-v", "error", "-nostdin", "-i", file, "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"];
  return new Promise((resolve) => {
    const child = spawn(ffmpeg, args, { stdio: ["ignore", "pipe", "pipe"], windowsHide: true }), chunks = [], errors = [];
    child.stdout.on("data", (chunk) => chunks.push(chunk)); child.stderr.on("data", (chunk) => errors.push(chunk));
    child.once("error", (error) => resolve({ ok: false, error: error.message }));
    child.once("close", (code) => resolve(code === 0 ? { ok: true, data: Buffer.concat(chunks) } : { ok: false, error: Buffer.concat(errors).toString().slice(0, 600) || `exit ${code}` }));
  });
}
// Returns { method, sha256, shape } plus the decoded buffer (not serialized) for difference metrics.
export async function decodedContent(file, kind, { ffmpeg = "ffmpeg" } = {}) {
  const bytes = await readFile(file);
  if (kind === "image") { const { width, height, rgba } = decodePng(bytes); return { method: "png_rgba8", shape: { width, height }, sha256: contentDigest({ width, height, format: "rgba8" }, rgba), buffer: rgba }; }
  if (kind === "audio" && bytes.toString("ascii", 0, 4) === "RIFF") { const wav = parseWav(bytes); const shape = { audioFormat: wav.audioFormat, channels: wav.channels, sampleRate: wav.sampleRate, bitsPerSample: wav.bitsPerSample, bytes: wav.data.length }; return { method: "wav_pcm_samples", shape, sha256: contentDigest(shape, wav.data), buffer: wav.data }; }
  if (kind === "audio") { const pcm = await ffmpegDecode(ffmpeg, file, "audio"); if (pcm.ok) return { method: "ffmpeg_s16le", shape: { bytes: pcm.data.length }, sha256: contentDigest({ format: "s16le" }, pcm.data), buffer: pcm.data }; return { method: "file_bytes_no_decoder", decoderError: pcm.error, sha256: sha256(bytes), buffer: bytes }; }
  const frames = await ffmpegDecode(ffmpeg, file, "video-frames");
  if (!frames.ok) return { method: "file_bytes_no_decoder", decoderError: frames.error, sha256: sha256(bytes), buffer: bytes };
  const audio = await ffmpegDecode(ffmpeg, file, "audio");
  const shape = { format: "rgba", frameBytes: frames.data.length, audioBytes: audio.ok ? audio.data.length : null };
  return { method: "ffmpeg_rgba_frames+s16le_audio", shape, sha256: contentDigest(shape, frames.data, audio.ok ? audio.data : Buffer.alloc(0)), buffer: frames.data };
}
// One numeric difference metric over decoded bytes (RGBA8 channels, PCM bytes or raw frames).
export function numericDifference(a, b) {
  if (!a || !b) return { comparable: false, reason: "decoded buffer unavailable" };
  if (a.length !== b.length) return { comparable: false, reason: "decoded length differs", lengthA: a.length, lengthB: b.length };
  let max = 0, total = 0, changed = 0;
  for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d) { changed++; total += d; if (d > max) max = d; } }
  return { comparable: true, units: "decoded byte values (0-255)", elements: a.length, changedElements: changed, changedFraction: changed / a.length, maxAbsoluteDifference: max, meanAbsoluteDifference: total / a.length };
}

// ---- verdict (REQUIREMENTS DEC-7 bounded stop rule) ---------------------------------------------
const contentOf = (run) => run?.output?.content?.sha256;
export function verdictFor({ base1, head, base2 }) {
  const failed = (run, label) => !run || run.status !== "completed" || !contentOf(run) ? { verdict: "error", needBase2: false, reason: `${label} did not complete: ${run?.error ?? "not run"}` } : null;
  const early = failed(base1, "base#1") ?? failed(head, "head");
  if (early) return early;
  if (contentOf(head) === contentOf(base1)) return { verdict: "equal", needBase2: false, reason: "head decoded content equals base#1" };
  if (!base2) return { verdict: null, needBase2: true, reason: "head differs from base#1; base#2 required" };
  const late = failed(base2, "base#2");
  if (late) return late;
  if (contentOf(base2) !== contentOf(base1)) return { verdict: "nondeterministic_on_runner", needBase2: false, reason: "base#1 and base#2 differ on this runner; no further runs" };
  return { verdict: "head_differs", needBase2: false, reason: "base#1 equals base#2 and head differs" };
}
export function exitCodeFor(report) {
  if (report.mode === "preflight") return report.summary.ready > 0 && !report.fatal ? 0 : 1;
  const bad = report.routes.some((route) => ["head_differs", "error"].includes(route.verdict) || route.hubMutation?.changed);
  return bad || report.summary.ran === 0 || report.fatal ? 1 : 0;
}

// ---- job polling bounds and diagnostics ---------------------------------------------------------
// A job is claimed once a worker id is attached or it leaves the queued/pending states. Two
// bounds per route: claimMinutes from submission until the claim, timeoutMinutes from the claim
// until completion. Either stops the run early with a specific reason instead of a long ceiling.
const UNCLAIMED = new Set(["queued", "pending_caption", "pending_workflow"]);
export const jobClaimed = (job) => Boolean(job?.workerId) || (job?.status !== undefined && !UNCLAIMED.has(job.status));
export function jobProgressDecision({ job, postedMs, claimedAtMs = null, nowMs, claimMinutes, timeoutMinutes }) {
  const where = `${job?.status ?? "?"}/${job?.stage ?? "?"}${job?.message ? ` (${String(job.message).slice(0, 200)})` : ""}`;
  if (["failed", "canceled", "interrupted"].includes(job?.status)) return { state: "stop", reason: `job ${job.status}: ${job.error ?? job.message ?? "no error served"}` };
  if (job?.status === "completed") return { state: "completed" };
  if (!jobClaimed(job)) {
    if (nowMs - postedMs > claimMinutes * 60_000) return { state: "stop", reason: `unclaimed: job stayed ${where} for more than ${claimMinutes} min after submission` };
    return { state: "waiting_claim" };
  }
  const claimed = claimedAtMs ?? nowMs;
  if (nowMs - claimed > timeoutMinutes * 60_000) return { state: "stop", reason: `execution bound: job still ${where} ${timeoutMinutes} min after the claim` };
  return { state: "running", claimedAtMs: claimed };
}
const JOB_FIELDS = ["id", "status", "stage", "message", "error", "workerId", "assignedGpu", "requestedGpu", "attempts", "cancelRequested", "progress", "createdAt", "updatedAt", "startedAt"];
// What the API serves about a job's state, including any refusal/blocked/admission field.
export function jobDiagnostics(job) {
  if (!job || typeof job !== "object") return null;
  const out = Object.fromEntries(JOB_FIELDS.filter((key) => job[key] !== undefined).map((key) => [key, job[key]]));
  const reasons = {};
  for (const [scope, value] of [["job", job], ["result", job.result], ["payload", job.payload]]) for (const [key, child] of Object.entries(value && typeof value === "object" ? value : {})) if (/block|refus|reason|admission|claim|pending/i.test(key)) reasons[`${scope}.${key}`] = child;
  if (Object.keys(reasons).length) out.reasons = reasons;
  return out;
}
export function lastLines(text, count = 50) { const lines = String(text ?? "").replace(/\r\n/g, "\n").split("\n"); while (lines.length && lines.at(-1) === "") lines.pop(); return lines.slice(-count); }
// One line per phase transition on the job log, so a watcher sees where a run is.
export function progressLine(fields, at = new Date()) {
  return `[tier2-ab] ${at.toISOString()} ${Object.entries(fields).filter(([, value]) => value !== undefined && value !== null && value !== "").map(([key, value]) => `${key}=${String(value).replace(/\s+/g, " ").slice(0, 300)}`).join(" ")}`;
}
let progressSink = (line) => process.stdout.write(`${line}\n`);
const progress = (fields) => progressSink(progressLine(fields));

// ---- scratch cleanup ----------------------------------------------------------------------------
// Head's worker writes immutable (read-only) attempt bundles under state/data/models, so a plain
// recursive delete fails. Restore owner write permission ONLY inside an attempt-owned scratch
// root, never following a link: on POSIX only directories need it (unlinking a read-only file
// needs a writable parent); on Windows read-only files are cleared too, except multiply linked
// files, which may share bytes with a hub and are reported instead of modified.
const inside = (root, candidate, paths) => { const rel = paths.relative(root, candidate); return rel !== "" && !rel.startsWith("..") && !paths.isAbsolute(rel); };
export async function removeScratch(target, { allowedRoots, protectedRoots = [], platform = process.platform, fs = { lstat, readdir, chmod, rm } } = {}) {
  const paths = pathsFor(platform), resolved = paths.resolve(target);
  const norm = (value) => (platform === "win32" ? paths.resolve(value).toLowerCase() : paths.resolve(value));
  fact(allowedRoots.some((root) => root && inside(norm(root), norm(resolved), paths)), `refusing to remove ${resolved}: not strictly inside ${allowedRoots.filter(Boolean).join(" or ") || "an allowed scratch root"}`);
  for (const hub of protectedRoots.filter(Boolean)) fact(norm(hub) !== norm(resolved) && !inside(norm(resolved), norm(hub), paths) && !inside(norm(hub), norm(resolved), paths), `refusing to remove ${resolved}: it overlaps the model hub ${hub}`);
  const result = { target: resolved, removed: false, restoredDirectories: 0, restoredFiles: 0, linkedReadOnly: [] };
  let top; try { top = await fs.lstat(resolved); } catch (error) { if (error.code === "ENOENT") return { ...result, removed: true, absent: true }; throw error; }
  const walk = async (dir) => {
    const info = await fs.lstat(dir);
    if (info.isSymbolicLink()) return;
    if (info.isDirectory()) {
      if ((info.mode & 0o700) !== 0o700) { await fs.chmod(dir, (info.mode & 0o7777) | 0o700); result.restoredDirectories++; }
      for (const name of await fs.readdir(dir)) await walk(paths.join(dir, name));
    } else if (platform === "win32" && (info.mode & 0o200) === 0) {
      if (info.nlink > 1) result.linkedReadOnly.push(dir);
      else { await fs.chmod(dir, (info.mode & 0o7777) | 0o200); result.restoredFiles++; }
    }
  };
  if (!top.isSymbolicLink()) await walk(resolved);
  fact(result.linkedReadOnly.length === 0, `refusing to change ${result.linkedReadOnly.length} multiply linked read-only file(s), e.g. ${result.linkedReadOnly[0]}`);
  await fs.rm(resolved, { recursive: true, force: true });
  result.removed = true;
  return result;
}

// ---- identity and redaction ---------------------------------------------------------------------
const IDENTITY_KEY = /model|artifact|binding|source|identity|revision|repo|tier|variant|quant|adapter|engine|snapshot|claim|path|digest|sha256|lease|ledger/i;
export function redactPaths(value, roots) {
  const ordered = Object.entries(roots).filter(([, root]) => root).map(([label, root]) => [label, root.replace(/\\/g, "/").replace(/\/+$/, "")]).sort((a, b) => b[1].length - a[1].length);
  const redact = (text) => {
    // Windows paths are compared with forward slashes, case-insensitively.
    let out = /(^|[^A-Za-z])[A-Za-z]:\\|^\\\\/.test(text) ? text.replace(/\\/g, "/") : text;
    for (const [label, root] of ordered) out = out.replace(new RegExp(root.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"), `<${label}>`);
    return out;
  };
  const walk = (node) => typeof node === "string" ? redact(node) : Array.isArray(node) ? node.map(walk) : node && typeof node === "object" ? Object.fromEntries(Object.entries(node).map(([key, child]) => [key, walk(child)])) : node;
  return walk(value);
}
export function identitySubset(payload) {
  const walk = (node) => {
    if (!node || typeof node !== "object") return undefined;
    if (Array.isArray(node)) { const items = node.map(walk); return items.some((item) => item !== undefined) ? items.map((item) => item ?? null) : undefined; }
    const out = {};
    for (const [key, child] of Object.entries(node)) {
      if (/prompt/i.test(key)) continue;
      if (IDENTITY_KEY.test(key)) out[key] = child; else { const nested = walk(child); if (nested !== undefined) out[key] = nested; }
    }
    return Object.keys(out).length ? out : undefined;
  };
  return walk(payload) ?? {};
}
export function diffObjects(a, b, limit = 200) {
  const flat = (node, prefix = "", out = new Map()) => {
    if (node && typeof node === "object") { const entries = Object.entries(node); if (!entries.length) out.set(prefix || "/", JSON.stringify(node)); for (const [key, child] of entries) flat(child, `${prefix}/${key}`, out); }
    else out.set(prefix || "/", JSON.stringify(node));
    return out;
  };
  const left = flat(a), right = flat(b), diffs = [];
  for (const key of [...new Set([...left.keys(), ...right.keys()])].sort()) if (left.get(key) !== right.get(key)) diffs.push({ path: key, base: left.get(key) ?? null, head: right.get(key) ?? null });
  return { count: diffs.length, entries: diffs.slice(0, limit), truncated: diffs.length > limit };
}

// ---- memory sampling ----------------------------------------------------------------------------
// macOS `top` MEM column is the process physical footprint (phys_footprint), which on Apple
// silicon includes Metal/IOAccelerator allocations made by the process. Values like 2640K,
// 512M+, 31G-.
export function parseTopMem(text) {
  const match = /^([0-9.]+)([BKMGT]?)[+-]?$/.exec(text.trim());
  fact(match, `unparseable top MEM value ${text}`);
  return Math.round(Number(match[1]) * ({ "": 1, B: 1, K: 1024, M: 1024 ** 2, G: 1024 ** 3, T: 1024 ** 4 })[match[2]]);
}
export function descendants(rows, root) {
  const children = new Map();
  for (const { pid, ppid } of rows) { if (!children.has(ppid)) children.set(ppid, []); children.get(ppid).push(pid); }
  const out = [root], queue = [root];
  while (queue.length) for (const child of children.get(queue.shift()) ?? []) if (!out.includes(child)) { out.push(child); queue.push(child); }
  return out;
}
async function sampleMacTree(pid) {
  const table = (await execFile("ps", ["-A", "-o", "pid=,ppid=,rss="], { timeout: 10_000, maxBuffer: 16 * 1024 * 1024 })).stdout;
  const rows = table.split("\n").map((line) => line.trim().split(/\s+/).map(Number)).filter((row) => row.length === 3 && row.every(Number.isFinite)).map(([p, pp, rss]) => ({ pid: p, ppid: pp, rssKiB: rss }));
  const tree = descendants(rows, pid), rssBytes = rows.filter((row) => tree.includes(row.pid)).reduce((sum, row) => sum + row.rssKiB * 1024, 0);
  // Footprint of the worker process itself (the MLX owner); RSS covers any descendants too.
  const top = (await execFile("top", ["-l", "1", "-stats", "pid,mem", "-pid", String(pid)], { timeout: 15_000, maxBuffer: 4 * 1024 * 1024 })).stdout;
  let footprintBytes = null;
  for (const line of top.split("\n")) { const parts = line.trim().split(/\s+/); if (parts.length === 2 && Number(parts[0]) === pid) footprintBytes = parseTopMem(parts[1]); }
  fact(footprintBytes !== null, "worker footprint sample unavailable");
  return { pids: tree.length, rssBytes, footprintBytes };
}
async function sampleWindowsProcess(pid) {
  const script = `$p = Get-Process -Id ${Number(pid)} -ErrorAction Stop; "$($p.WorkingSet64) $($p.PeakWorkingSet64) $($p.PrivateMemorySize64)"`;
  const out = (await execFile("powershell", ["-NoProfile", "-NonInteractive", "-Command", script], { timeout: 15_000, windowsHide: true })).stdout.trim().split(/\s+/).map(Number);
  fact(out.length === 3 && out.every(Number.isFinite), "worker working-set sample unavailable");
  return { workingSetBytes: out[0], peakWorkingSetBytes: out[1], privateBytes: out[2] };
}
export function nvidiaSmiPath(env = process.env, platform = process.platform) {
  return platform === "win32" ? path.win32.join(env.SystemRoot ?? "C:\\Windows", "System32", "nvidia-smi.exe") : "nvidia-smi";
}
async function sampleGpu(smi) {
  const out = (await execFile(smi, ["-i", CUDA_PHYSICAL_GPU, "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], { timeout: 10_000, windowsHide: true })).stdout.trim().split(",").map((v) => v.trim());
  fact(out.length === 4 && out[0] === CUDA_PHYSICAL_GPU && /^GPU-/.test(out[1]) && /^\d+$/.test(out[2]), "physical GPU1 telemetry unavailable");
  return { uuid: out[1], usedBytes: Number(out[2]) * 1024 ** 2, utilization: Number(out[3]) };
}

// Every GPU's identity from nvidia-smi CSV (index, bus id) joined with the driver XML (display,
// memory, utilization and processes with C/G/C+G type). On Windows WDDM per-process memory is N/A.
export function gpuInventory(csv, xml) {
  const byUuid = new Map(csv.split(/\r?\n/).filter(Boolean).map((line) => { const [index, uuid, bus, ...name] = line.split(",").map((value) => value.trim()); return [uuid, { index: Number(index), pciBusId: bus, name: name.join(",") }]; }));
  const field = (block, tag) => (new RegExp(`<${tag}>([^<]*)</${tag}>`).exec(block)?.[1] ?? "").trim();
  return [...xml.matchAll(/<gpu id="([^"]+)">([\s\S]*?)<\/gpu>/g)].map(([, bus, block]) => {
    const uuid = field(block, "uuid"), memory = /<fb_memory_usage>([\s\S]*?)<\/fb_memory_usage>/.exec(block)?.[1] ?? "", utilization = /<utilization>([\s\S]*?)<\/utilization>/.exec(block)?.[1] ?? "";
    const processes = [...block.matchAll(/<process_info>([\s\S]*?)<\/process_info>/g)].map(([, row]) => ({ pid: field(row, "pid"), type: field(row, "type"), name: field(row, "process_name").split(/[\\/]/).pop(), usedMemory: field(row, "used_memory") }));
    return { index: byUuid.get(uuid)?.index ?? null, uuid, pciBusId: byUuid.get(uuid)?.pciBusId ?? bus, name: field(block, "product_name"), displayAttached: field(block, "display_attached"), displayActive: field(block, "display_active"), displayMode: field(block, "display_mode"), driverModel: field(block, "current_dm"), memoryTotal: field(memory, "total"), memoryUsed: field(memory, "used"), memoryFree: field(memory, "free"), utilization: field(utilization, "gpu_util"), processes };
  });
}

// ---- reporting ----------------------------------------------------------------------------------
export function summarize(routes) {
  const summary = { routes: routes.length, ready: 0, ran: 0, skipped: 0 };
  for (const verdict of VERDICTS) summary[verdict] = 0;
  for (const route of routes) {
    if (route.status === "ready") summary.ready++;
    if (route.status === "ran") summary.ran++;
    if (route.status === "not_run") summary.notRun = (summary.notRun ?? 0) + 1;
    if (route.status === "skipped") summary.skipped++;
    else if (route.verdict && route.verdict !== "skipped" && Object.hasOwn(summary, route.verdict)) summary[route.verdict]++;
  }
  return summary;
}
const gib = (bytes) => (Number.isFinite(bytes) ? `${(bytes / GIB).toFixed(2)} GiB` : "–");
const secs = (value) => (Number.isFinite(value) ? `${value.toFixed(1)} s` : "–");
export function renderMarkdown(report) {
  const lines = [`# Tier 2 A/B (${report.backend}) — ${report.mode}`, "", `- status: **${report.status}** (exit ${report.exitCode})`, `- base: \`${report.base.sha}\` (inference ${report.base.inferencePin ?? "?"}, binary ${report.base.binarySha256 ?? "not built"})`, `- head: \`${report.head.sha}\` (inference ${report.head.inferencePin ?? "?"}, binary ${report.head.binarySha256 ?? "not built"})`, `- runner: ${report.runner.name ?? "local"} (${report.runner.platform}/${report.runner.arch}, ${gib(report.runner.totalMemoryBytes)} RAM)`, `- window: ${report.startedAt} → ${report.finishedAt ?? "running"}`, `- memory method: ${report.memoryMethod}`, `- fidelity method: decoded content (PNG→RGBA8, WAV→PCM samples, video→ffmpeg RGBA frames + s16le audio); file sha256 recorded separately`, ""];
  if (report.gpuBinding?.policy === "shared-display") lines.push("- GPU sharing: physical GPU1 is the host's display GPU. Memory peaks are reported above each run's pre-request baseline (\"shared GPU, baseline N MiB\") and times were taken alongside display work. WDDM reports no per-process GPU memory, so admission and attribution use device memory.used/free and the driver's process types.", "");
  if (Array.isArray(report.gpus)) {
    lines.push("| nvidia-smi index | PCI bus | UUID | display attached | used / total | util | processes (C, G, C+G) |", "|---|---|---|---|---|---|---|");
    for (const gpu of report.gpus) { const count = (type) => gpu.processes.filter((row) => row.type === type).length; lines.push(`| ${gpu.index} | ${gpu.pciBusId} | ${gpu.uuid} | ${gpu.displayAttached} | ${gpu.memoryUsed} / ${gpu.memoryTotal} | ${gpu.utilization} | C ${count("C")} (${gpu.processes.filter((row) => row.type === "C").map((row) => `${row.name}:${row.pid}`).join(" ")}), G ${count("G")}, C+G ${count("C+G")} |`); }
    lines.push("");
  }
  if (report.fatal) lines.push(`**Fatal:** ${report.fatal}`, "");
  lines.push("| route | model | status | verdict | base#1 time | head time | base#2 time | base#1 mem | head mem | notes |", "|---|---|---|---|---|---|---|---|---|---|");
  for (const route of report.routes) {
    const run = (label) => route.runs?.find((item) => item.label === label);
    const time = (label) => secs(run(label)?.timing?.enqueueToCompleteSeconds), mem = (label) => gib(run(label)?.memory?.peakBytes);
    lines.push(`| ${route.id} | ${route.model} | ${route.status} | ${route.verdict ?? (route.status === "not_run" ? "not_run" : "–")} | ${time("base#1")} | ${time("head")} | ${time("base#2")} | ${mem("base#1")} | ${mem("head")} | ${(route.skipReason ?? route.verdictReason ?? "").replace(/\|/g, "/").slice(0, 300)} |`);
  }
  for (const route of report.routes.filter((item) => item.comparison || item.runs?.length)) {
    lines.push("", `## ${route.id}`, "", `- hub: ${route.hub}`, `- request sha256: ${route.request.sha256}`);
    for (const runRecord of route.runs ?? []) lines.push(`- ${runRecord.label}: ${runRecord.status}; content ${runRecord.output?.content?.method ?? "-"} ${runRecord.output?.content?.sha256 ?? "-"}; file ${runRecord.output?.fileSha256 ?? "-"}; load ${secs(runRecord.timing?.loadSeconds)}, denoise ${secs(runRecord.timing?.denoiseSeconds)}; startup ${secs(runRecord.timing?.startupSeconds)}; claim ${secs(runRecord.timing?.claimSeconds)}${runRecord.error ? `; error: ${runRecord.error}` : ""}${runRecord.diagnostics?.processes?.worker ? `; worker ${runRecord.diagnostics.processes.worker.alive ? "alive" : `exited code ${runRecord.diagnostics.processes.worker.exitCode} signal ${runRecord.diagnostics.processes.worker.signal}`} (stdin ${runRecord.diagnostics.processes.worker.stdin})` : ""}${Array.isArray(runRecord.diagnostics?.workersResponse) ? `; /workers: ${runRecord.diagnostics.workersResponse.map((row) => `${row.id}=${row.status}`).join(", ").slice(0, 300)}` : ""}${runRecord.diagnostics?.lastJob ? `; last job ${runRecord.diagnostics.lastJob.status}/${runRecord.diagnostics.lastJob.stage}${runRecord.diagnostics.lastJob.message ? ` "${String(runRecord.diagnostics.lastJob.message).slice(0, 200)}"` : ""}${runRecord.diagnostics.lastJob.reasons ? ` reasons ${JSON.stringify(runRecord.diagnostics.lastJob.reasons).slice(0, 300)}` : ""}` : ""}`);
    const diff = route.comparison?.payloadIdentityDiff;
    lines.push(`- payload identity diff (base#1 vs head): ${diff?.count ?? 0} field(s)`);
    for (const entry of (diff?.entries ?? []).slice(0, 40)) lines.push(`  - \`${entry.path}\`: base ${entry.base} → head ${entry.head}`);
    if (route.comparison?.numericDifference) lines.push(`- numeric difference (${route.comparison.numericDifferenceOf}): ${JSON.stringify(route.comparison.numericDifference)}`);
    if (route.hubMutation?.changed) lines.push(`- **hub mutation detected**: ${JSON.stringify(route.hubMutation.diff.entries.slice(0, 10))}`);
    for (const runRecord of (route.runs ?? []).filter((item) => item.hubMutation?.changed)) lines.push(`- hub mutation during ${runRecord.label} (${runRecord.side}): ${runRecord.hubMutation.diff.count} change(s), e.g. ${JSON.stringify(runRecord.hubMutation.diff.entries.slice(0, 6))}`);
  }
  return lines.join("\n") + "\n";
}

// ---- side identity, preflight -------------------------------------------------------------------
async function git(cwd, ...args) { return (await execFile("git", args, { cwd, timeout: 60_000, maxBuffer: 16 * 1024 * 1024 })).stdout.trim(); }
export function pmetalSource(lock) { return /name = "pmetal-mlx-sys"\nversion = "[^"]+"\nsource = "([^"]+)"/.exec(lock.replace(/\r\n/g, "\n"))?.[1] ?? null; }
async function sideIdentity(label, dir, sha) {
  const head = await git(dir, "rev-parse", "HEAD"), dirty = await git(dir, "status", "--porcelain");
  fact(head === sha, `${label} checkout ${dir} is at ${head}, expected ${sha}`);
  fact(dirty === "", `${label} checkout ${dir} is not clean`);
  const lock = await readFile(path.join(dir, "Cargo.lock"), "utf8");
  return { sha, checkout: dir, inferencePin: lockedInferenceRevision(lock), lockSha256: sha256(lock), pmetalSource: pmetalSource(lock) };
}
async function tool(command, args) { try { const { stdout, stderr } = await execFile(command, args, { timeout: 20_000, windowsHide: true }); return { available: true, version: `${stdout}${stderr}`.split(/\r?\n/)[0].slice(0, 200) }; } catch (error) { return { available: false, error: error.message.slice(0, 200) }; } }
async function freeBytes(dir) { try { const fs = await statfs(dir, { bigint: true }); return Number(fs.bavail * fs.bsize); } catch { return null; } }

async function preflight(env, cfg, report) {
  const platform = process.platform;
  if (cfg.backend === "mlx") fact(platform === "darwin" && process.arch === "arm64", "the mlx backend needs a native Apple-silicon Mac");
  else fact(platform === "win32", "the cuda backend needs the Windows CUDA runner");
  if (env.GITHUB_ACTIONS === "true") {
    if (cfg.backend === "mlx") fact(env.RUNNER_NAME === MLX_RUNNER, `mlx A/B must run on ${MLX_RUNNER} (rw-mage), not ${env.RUNNER_NAME}`);
    else fact(CUDA_RUNNERS.includes(env.RUNNER_NAME), `cuda A/B must run on a real-weights CUDA listener (${CUDA_RUNNERS.join(", ")}), not ${env.RUNNER_NAME}`);
  }
  if (cfg.backend === "cuda") for (const key of ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "SCENEWORKS_GPU_ID"]) fact(env[key] === CUDA_PHYSICAL_GPU, `${key}=${CUDA_PHYSICAL_GPU} required (physical GPU1)`);
  if (cfg.backend === "cuda") fact(env.CUDA_DEVICE_ORDER === "PCI_BUS_ID", "CUDA_DEVICE_ORDER=PCI_BUS_ID required");
  report.base = { ...report.base, ...(await sideIdentity("base", cfg.baseDir, cfg.baseSha)) };
  report.head = { ...report.head, ...(await sideIdentity("head", cfg.headDir, cfg.headSha)) };
  if (cfg.inferenceSha) for (const side of [report.base, report.head]) fact(side.inferencePin === cfg.inferenceSha, `inference pin ${side.inferencePin} differs from the requested ${cfg.inferenceSha}`);
  if (cfg.backend === "mlx") fact(report.base.pmetalSource && report.base.pmetalSource === report.head.pmetalSource, "base and head must pin the same pmetal-mlx-sys revision (one prebuilt MLX serves both builds)");
  const ffmpeg = env.SCENEWORKS_FFMPEG || "ffmpeg";
  report.tools = { ffmpeg: { command: ffmpeg, ...(await tool(ffmpeg, ["-version"])) }, git: await tool("git", ["--version"]) };
  if (cfg.backend === "mlx") report.tools.python312 = await tool("python3.12", ["--version"]);
  if (cfg.backend === "cuda") {
    report.tools.nvidiaSmi = await tool(nvidiaSmiPath(env), ["-i", CUDA_PHYSICAL_GPU, "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"]); fact(report.tools.nvidiaSmi.available, "nvidia-smi cannot query physical GPU1");
    // Every GPU's identity, so one preflight answers which card the lease and the worker use.
    try {
      const smi = nvidiaSmiPath(env);
      const csv = (await execFile(smi, ["--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader"], { timeout: 30_000, windowsHide: true })).stdout;
      const xml = (await execFile(smi, ["-q", "-x"], { timeout: 30_000, windowsHide: true, maxBuffer: 64 * 1024 * 1024 })).stdout;
      report.gpus = gpuInventory(csv, xml);
      report.gpuBinding = { leasedPhysicalIndex: CUDA_PHYSICAL_GPU, cudaVisibleDevices: env.CUDA_VISIBLE_DEVICES, cudaDeviceOrder: env.CUDA_DEVICE_ORDER, note: "nvidia-smi index is PCI bus order; with CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=1 exposes that same card as the worker's logical CUDA 0", policy: env.TIER2_AB_GPU_POLICY || "strict-idle" };
    } catch (error) { report.gpus = { error: error.message.slice(0, 300) }; }
  }
  if (cfg.backend === "mlx") fact(report.tools.python312.available, "python3.12 is required by the shared Metal execution lock");
  await mkdir(cfg.work, { recursive: true });
  report.disk = { workFreeBytes: await freeBytes(cfg.work), note: "statfs available bytes; macOS purgeable space is not included" };
  fact(report.disk.workFreeBytes === null || report.disk.workFreeBytes >= 20 * GIB, `only ${gib(report.disk.workFreeBytes)} free for the shared build target and evidence; at least 20 GiB required`);
  const [baseCatalog, headCatalog] = await Promise.all([readCatalog(cfg.baseDir), readCatalog(cfg.headDir)]);
  report.routes = await planRoutes({ routes: cfg.routes, hubs: cfg.hubs, baseCatalog, headCatalog, platform });
  for (const route of report.routes.filter((item) => item.kind === "video" && item.status === "ready")) if (!report.tools.ffmpeg.available) route.note = "ffmpeg unavailable: fidelity falls back to file bytes and the worker cannot mux video";
}

// ---- build --------------------------------------------------------------------------------------
async function runLogged(command, args, { cwd, env, log }) {
  const handle = await open(log, "w");
  try {
    const child = spawn(command, args, { cwd, env, stdio: ["ignore", handle.fd, handle.fd], windowsHide: true });
    const result = await new Promise((resolve) => { child.once("error", (error) => resolve({ code: null, error: error.message })); child.once("exit", (code, signal) => resolve({ code, signal })); });
    return result;
  } finally { await handle.close(); }
}
async function tail(file, bytes = 8000) { try { const text = await readFile(file, "utf8"); return text.slice(-bytes); } catch { return ""; } }
async function build(env, cfg, report) {
  const scratch = path.join(cfg.work, "scratch"), target = path.join(scratch, "target"), evidence = path.join(cfg.work, "evidence");
  await mkdir(target, { recursive: true }); await mkdir(evidence, { recursive: true });
  const exe = process.platform === "win32" ? "sceneworks-rust-api.exe" : "sceneworks-rust-api";
  const built = {};
  // Base first, then head: both share one target, so the second build reuses every unchanged
  // dependency unit. Each binary (plus loose native libraries) is copied out before the next build.
  for (const [label, dir, sha] of [["base", cfg.baseDir, cfg.baseSha], ["head", cfg.headDir, cfg.headSha]]) {
    const args = ["build", "--release", "--locked", "-p", "sceneworks-rust-api", "--message-format=json-render-diagnostics", ...(cfg.backend === "cuda" ? ["--features", "backend-candle", "--bin", "sceneworks-rust-api"] : [])];
    const log = path.join(scratch, `build-${label}.jsonl`), started = Date.now();
    const result = await runLogged("cargo", args, { cwd: dir, env: { ...env, CARGO_TARGET_DIR: target }, log });
    if (result.code !== 0) { await writeFile(path.join(evidence, `build-${label}.tail.log`), await tail(log, 64_000)); throw new Error(`${label} release build failed (${JSON.stringify(result)}); see build-${label}.tail.log`); }
    fact(await git(dir, "status", "--porcelain") === "" && await git(dir, "rev-parse", "HEAD") === sha, `${label} build changed its checkout`);
    let metallib = null;
    if (cfg.backend === "mlx") {
      const native = await bindMetallib({ messages: log, lockFile: path.join(dir, "Cargo.lock"), targetRoot: target, temporaryRoot: scratch, consumer: "app", explicitOverride: env.PMETAL_METALLIB_PATH ?? "", explicitNativeDirectory: env.PMETAL_MLX_PREBUILT_DIR ?? "" });
      metallib = { path: native.path, sha256: native.sha256 };
    }
    const bin = path.join(scratch, "bin", label); await mkdir(bin, { recursive: true });
    for (const entry of await readdir(path.join(target, "release"), { withFileTypes: true })) if (entry.isFile() && (entry.name === exe || /\.(dll|dylib|so)$/i.test(entry.name))) await copyFile(path.join(target, "release", entry.name), path.join(bin, entry.name));
    const binary = path.join(bin, exe);
    built[label] = { binary, binarySha256: await fileSha256(binary), metallib, buildSeconds: (Date.now() - started) / 1000 };
    report[label] = { ...report[label], binarySha256: built[label].binarySha256, buildSeconds: built[label].buildSeconds, metallibSha256: metallib?.sha256 ?? null };
  }
  await writeFile(path.join(scratch, "build.json"), JSON.stringify({ baseSha: cfg.baseSha, headSha: cfg.headSha, backend: cfg.backend, built }, null, 2) + "\n");
  return built;
}

// ---- one run: fresh state, fresh API + worker, one project, one job ------------------------------
async function hubFingerprint(hub, repos) {
  const list = async (dir) => { try { return (await readdir(dir, { withFileTypes: true })).map((entry) => entry.name).sort(); } catch { return null; } };
  const mtime = async (target) => { try { return (await stat(target)).mtimeMs; } catch { return null; } };
  const out = { root: { entries: await list(hub), mtime: await mtime(hub) }, hfHome: { entries: await list(path.dirname(hub)) }, repos: {} };
  for (const { repo, revision } of repos) {
    const dir = path.join(hub, `models--${repo.replace("/", "--")}`), snap = path.join(dir, "snapshots", revision);
    // Every directory under the snapshot (bounded depth), with its own mtime and listing, and each
    // entry's lstat mtime/ctime (a link is not followed), so a re-created entry is visible too.
    const nested = {};
    const walk = async (dir, rel, depth) => {
      let entries; try { entries = await readdir(dir, { withFileTypes: true }); } catch { nested[rel || "."] = null; return; }
      nested[rel || "."] = { mtime: await mtime(dir), entries: entries.map((entry) => entry.name).sort() };
      for (const entry of entries) {
        const child = path.join(dir, entry.name), childRel = rel ? `${rel}/${entry.name}` : entry.name;
        if (entry.isDirectory() && depth < 4) await walk(child, childRel, depth + 1);
        else { try { const info = await lstat(child); nested[`${childRel}#entry`] = { mtime: info.mtimeMs, ctime: info.ctimeMs, link: info.isSymbolicLink(), size: info.size }; } catch { nested[`${childRel}#entry`] = null; } }
      }
    };
    await walk(snap, "", 0);
    out.repos[`${repo}@${revision}`] = { dir: await list(dir), dirMtime: await mtime(dir), refs: await list(path.join(dir, "refs")), snapshots: await list(path.join(dir, "snapshots")), snapshot: nested, blobsMtime: await mtime(path.join(dir, "blobs")) };
  }
  return out;
}
// The worker treats stdin EOF as a graceful-shutdown request on Windows whenever
// SCENEWORKS_WORKER_CHILD=1 (worker lib.rs wait_for_process_shutdown_source, at base and head).
// A child started with stdin "ignore" reads EOF from NUL at once, latches shutdown, posts an
// Offline heartbeat right after registering and never claims; the in-process utility loops in
// the API do the same. So API and worker get a held stdin pipe, exactly like the proven Windows
// admitted-replay launcher (measure-sdxl-cuda-endpoints.mjs), and teardown closes it to request
// the graceful shutdown. stdout and stderr both go to the process's own log.
export const CHILD_STDIN = Object.freeze({ sentinel: "pipe", api: "pipe", worker: "pipe" });
export const processState = (child) => ({ pid: child.pid ?? null, alive: Boolean(child.pid) && child.exitCode === null && child.signalCode === null, exitCode: child.exitCode ?? null, signal: child.signalCode ?? null, stdin: CHILD_STDIN[child.label] ?? "ignore" });
export function childStdio(name, logFd) { return [CHILD_STDIN[name] ?? "ignore", logFd, logFd]; }
export function childEnvironments(env, { backend, state, hub, xet, workerId, metallib }) {
  const weights = path.dirname(hub);
  let base = buildChildEnv(env, { state, weights, hub, xet, workerId, url: "http://127.0.0.1:0" });
  if (backend === "cuda") {
    base = { ...base, SCENEWORKS_GPU_ID: CUDA_PHYSICAL_GPU, SCENEWORKS_BACKEND_MLX_ENABLED: "false", SCENEWORKS_BACKEND_CANDLE_ENABLED: "true" };
    delete base.SCENEWORKS_MLX_REQUIRED; delete base.SCENEWORKS_MLX_UNSUPPORTED_MODE;
  }
  if (metallib) base.PMETAL_METALLIB_PATH = metallib;
  // Head's admission resolves pinned file metadata once per fresh data dir; a gated repository
  // needs the runner credential. Both sides receive the identical environment.
  if (env.HF_TOKEN) base.HF_TOKEN = env.HF_TOKEN;
  return { api: bindOwnedEnv(base, { state, workerId }), worker: (url) => ({ ...bindOwnedEnv(base, { state, workerId }, url), SCENEWORKS_WORKER_ONLY: "1" }) };
}
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
// Decoded output buffers stay in memory (never serialized) for one numeric difference metric.
const decodedBuffers = new WeakMap();
async function executeRun({ env, cfg, route, plan, label, side, built, evidenceRoot, secrets, signal }) {
  const sideCfg = side === "base" ? { dir: cfg.baseDir, sha: cfg.baseSha } : { dir: cfg.headDir, sha: cfg.headSha };
  const slug = `${route.id}-${label.replace("#", "")}`;
  const runRoot = path.join(cfg.work, "scratch", "runs", slug), evidence = path.join(evidenceRoot, "runs", route.id, label.replace("#", ""));
  await rm(runRoot, { recursive: true, force: true }); await mkdir(runRoot, { recursive: true }); await mkdir(evidence, { recursive: true });
  const state = path.join(runRoot, "state"), xet = path.join(runRoot, "xet"), logs = path.join(runRoot, "logs");
  for (const dir of [path.join(state, "data", "projects"), path.join(state, "credentials"), xet, logs]) await mkdir(dir, { recursive: true });
  await cp(path.join(sideCfg.dir, "config"), path.join(state, "config"), { recursive: true });
  const workerId = `tier2-ab-${side}-${process.pid}-${Date.now().toString(36)}`;
  const claimMinutes = cfg.claimMinutesOverride ?? route.claimMinutes;
  const record = { label, side, sha: sideCfg.sha, binarySha256: built[side].binarySha256, workerId, status: "running", phase: "starting", startedAt: new Date().toISOString(), timing: {}, memory: {}, identity: {}, watchdog: { capBytes: null, claimMinutes, timeoutMinutes: route.timeoutMinutes } };
  // run.json is rewritten at every phase transition, so a cancelled job still leaves evidence.
  const persist = () => writeFile(path.join(evidence, "run.json"), redactSensitiveText(JSON.stringify(record, null, 2) + "\n", secrets)).catch(() => {});
  const phase = async (name, fields = {}) => { record.phase = name; progress({ route: route.id, run: label, phase: name, ...fields }); await persist(); };
  const envs = childEnvironments(env, { backend: cfg.backend, state, hub: plan.hub, xet, workerId, metallib: built[side].metallib?.path });
  const children = [], handles = [], controller = new AbortController();
  const onAbort = () => controller.abort(signal.reason ?? new Error("A/B interrupted"));
  signal.addEventListener("abort", onAbort, { once: true });
  const posix = process.platform !== "win32";
  const launch = async (name, command, args, childEnv) => {
    const log = await open(path.join(logs, `${name}.log`), "wx"); handles.push(log);
    const child = spawn(command, args, { cwd: sideCfg.dir, env: childEnv, detached: posix && name !== "sentinel", stdio: childStdio(name, log.fd), windowsHide: true });
    child.label = name; children.push(child);
    child.done = new Promise((resolve) => { child.once("error", (error) => resolve({ error: error.message })); child.once("exit", (code, sig) => resolve({ code, signal: sig })); });
    return child;
  };
  let endpoint, apiUrl, job, accepted, samplers = [], sampling = false;
  const runStarted = Date.now();
  // Outer backstop only; the claim and execution bounds below stop a run first and say why.
  const runMinutes = claimMinutes + route.timeoutMinutes + 15;
  record.watchdog.runMinutes = runMinutes;
  const deadline = runStarted + runMinutes * 60_000;
  const watchdog = (reason) => { if (!controller.signal.aborted) { record.watchdog.tripped = reason; controller.abort(new Error(`watchdog: ${reason}`)); } };
  const timer = setTimeout(() => watchdog(`per-run backstop ${runMinutes} min`), runMinutes * 60_000);
  let lastJob = null;
  const alive = () => { if (controller.signal.aborted) throw controller.signal.reason; };
  const wait = async (probe, seconds, what) => { const end = Math.min(Date.now() + seconds * 1000, deadline); while (Date.now() < end) { alive(); const value = await probe(); if (value) return value; await delay(500); } throw new Error(`${what} timed out`); };
  try {
    // Desktop parent-death contract: API and worker exit gracefully when this sentinel ends.
    const sentinel = await launch("sentinel", process.execPath, ["-e", 'process.stdin.resume();process.stdin.on("end",()=>process.exit(0));'], env);
    envs.api.SCENEWORKS_PARENT_PID = String(sentinel.pid);
    const api = await launch("api", built[side].binary, [], envs.api);
    const url = await wait(async () => boundApiUrl(await readFile(path.join(logs, "api.log"), "utf8")), 180, "API bound-port announcement");
    apiUrl = url;
    endpoint = ownedApiController({ url, state, api, controller });
    record.identity.health = await wait(async () => { try { return await endpoint.verify(); } catch (error) { if (controller.signal.aborted) throw error; return false; } }, 180, "API readiness");
    const worker = await endpoint.startWorker(() => launch("worker", built[side].binary, [], { ...envs.worker(url), SCENEWORKS_PARENT_PID: String(sentinel.pid) }));
    // Registration fails fast, with a cause, when the worker exits or reports offline.
    let offlineSince = null;
    const registration = await wait(async () => {
      if (worker.exitCode !== null || worker.signalCode !== null) { record.failureClass = "worker_registration"; throw new Error(`owned worker exited before registering idle (code ${worker.exitCode}, signal ${worker.signalCode})`); }
      const row = (await endpoint.call("GET", "/api/v1/workers")).find((item) => item.id === workerId);
      if (row?.status === "offline") { offlineSince ??= Date.now(); if (Date.now() - offlineSince > 15_000) { record.failureClass = "worker_registration"; throw new Error("owned worker registered but reports offline (it requested its own shutdown)"); } } else offlineSince = null;
      return row?.status === "idle" && row;
    }, 300, "owned worker registration").catch((error) => { record.failureClass ??= "worker_registration"; throw error; });
    const expectedGpu = cfg.backend === "mlx" ? "mlx" : CUDA_PHYSICAL_GPU;
    fact(registration.gpuId === expectedGpu, `owned worker registered gpuId ${registration.gpuId}, expected ${expectedGpu}`);
    record.identity.worker = { id: registration.id, gpuId: registration.gpuId, gpuName: registration.gpuName, capabilities: registration.capabilities };
    record.timing.startupSeconds = (Date.now() - runStarted) / 1000;
    await phase("services_up", { side, startup_s: record.timing.startupSeconds.toFixed(1) });
    const project = await endpoint.call("POST", "/api/v1/projects", { name: `SW-7033 tier2 A/B ${route.id} ${label}` });
    const projectPath = await realpath(project.path), stateReal = await realpath(state);
    fact(!path.relative(stateReal, projectPath).startsWith(".."), "project escaped the run's own state");
    let referenceAudioAssetId;
    if (route.referenceAudio) {
      const form = new FormData(); form.append("file", new Blob([referenceWav()], { type: "audio/wav" }), "reference.wav");
      const upload = await endpoint.exchange("POST", `/api/v1/projects/${project.id}/assets`, form, { multipart: true });
      fact(upload.status < 300 && upload.body?.id, `reference audio upload refused (${upload.status}): ${JSON.stringify(upload.body).slice(0, 600)}`);
      referenceAudioAssetId = upload.body.id;
    }
    const body = requestBody(route, project.id, referenceAudioAssetId);
    record.requestSha256 = sha256(JSON.stringify({ ...body, projectId: "<project>", ...(route.referenceAudio ? { referenceAudioAssetId: "<reference-audio>" } : {}) }));
    fact(record.requestSha256 === plan.request.sha256, "request body diverged from the route template");
    // Memory baselines and samplers start before the POST and stop at completion.
    const memory = { samples: 0, errors: 0, peakBytes: null };
    sampling = true;
    if (cfg.backend === "mlx") {
      const cap = Math.min(100 * GIB, Math.floor(os.totalmem() * 0.8)); record.watchdog.capBytes = cap;
      memory.method = "macOS top MEM (phys_footprint, includes Metal allocations) + ps RSS, summed over the worker process tree, sampled about every 2 s";
      samplers.push((async () => { while (sampling) { try { const s = await sampleMacTree(worker.pid); memory.samples++; memory.peakBytes = Math.max(memory.peakBytes ?? 0, s.footprintBytes); memory.peakRssBytes = Math.max(memory.peakRssBytes ?? 0, s.rssBytes); if (Math.max(s.footprintBytes, s.rssBytes) > cap) watchdog(`worker footprint ${gib(s.footprintBytes)} exceeded cap ${gib(cap)}`); } catch { memory.errors++; } await delay(1000); } })());
    } else {
      const cap = Math.floor(os.totalmem() * 0.75); record.watchdog.capBytes = cap;
      const smi = nvidiaSmiPath(env);
      const baseline = []; for (let i = 0; i < 5; i++) { baseline.push((await sampleGpu(smi)).usedBytes); await delay(200); }
      memory.method = "nvidia-smi memory.used on physical GPU1 sampled every ~500 ms, peak minus the max of 5 pre-request samples (device-wide, not a per-process allocator peak); worker host working set via Get-Process every ~3 s";
      memory.gpuBaselineBytes = Math.max(...baseline);
      if (env.TIER2_AB_GPU_POLICY === "shared-display") { memory.sharedGpu = `shared GPU, baseline ${Math.round(memory.gpuBaselineBytes / 1024 ** 2)} MiB`; record.timing.sharedGpu = "times taken on the host's display GPU, alongside display work"; }
      samplers.push((async () => { while (sampling) { try { const s = await sampleGpu(smi); memory.samples++; memory.gpuPeakBytes = Math.max(memory.gpuPeakBytes ?? 0, s.usedBytes); memory.gpuUuid = s.uuid; if (s.utilization > 0) memory.gpuActive = true; } catch { memory.errors++; } await delay(500); } })());
      samplers.push((async () => { while (sampling) { try { const s = await sampleWindowsProcess(worker.pid); memory.hostPeakWorkingSetBytes = Math.max(memory.hostPeakWorkingSetBytes ?? 0, s.peakWorkingSetBytes); if (s.workingSetBytes > cap) watchdog(`worker working set ${gib(s.workingSetBytes)} exceeded cap ${gib(cap)}`); } catch { memory.errors++; } await delay(3000); } })());
    }
    const posted = Date.now();
    const response = await endpoint.exchange("POST", route.endpoint, body);
    record.acceptedStatus = response.status;
    fact(response.status < 300 && response.body?.id, `API refused the request (${response.status}): ${JSON.stringify(response.body).slice(0, 1500)}`);
    accepted = response.body; record.jobId = accepted.id;
    const roots = { hub: plan.hub, "hf-home": path.dirname(plan.hub), state: await realpath(state), checkout: sideCfg.dir, work: cfg.work, home: os.homedir() };
    // Recorded at submission so a run that never completes still carries its payload identity.
    record.identity.payload = redactPaths(identitySubset(accepted.payload ?? {}), roots);
    await phase("submitted", { side, job: accepted.id });
    const stages = []; let last = "", claimedAtMs = null;
    record.timing.stages = stages;
    while (true) {
      alive();
      if (Date.now() > deadline) throw new Error("per-run backstop reached while polling");
      const current = await endpoint.call("GET", `/api/v1/jobs/${accepted.id}`);
      lastJob = current;
      const key = `${current.status}/${current.stage}`;
      if (key !== last) {
        stages.push({ atSeconds: (Date.now() - posted) / 1000, status: current.status, stage: current.stage, message: String(current.message ?? "").slice(0, 200) }); last = key;
        if (claimedAtMs === null && jobClaimed(current)) { claimedAtMs = Date.now(); record.timing.claimSeconds = (claimedAtMs - posted) / 1000; await phase("claimed", { side, job: accepted.id, worker: current.workerId === workerId ? "owned" : current.workerId, after_s: record.timing.claimSeconds.toFixed(1) }); }
        else await phase(`stage_${current.stage}`, { side, job: accepted.id, status: current.status, at_s: ((Date.now() - posted) / 1000).toFixed(1) });
      }
      const decision = jobProgressDecision({ job: current, postedMs: posted, claimedAtMs, nowMs: Date.now(), claimMinutes, timeoutMinutes: route.timeoutMinutes });
      if (decision.state === "stop") { record.stopReason = decision.reason; throw new Error(decision.reason); }
      if (decision.state === "completed") { job = current; break; }
      await delay(500);
    }
    record.timing.enqueueToCompleteSeconds = (Date.now() - posted) / 1000;
    await phase("completed", { side, job: accepted.id, total_s: record.timing.enqueueToCompleteSeconds.toFixed(1) });
    sampling = false; await Promise.all(samplers); samplers = [];
    if (cfg.backend === "cuda") {
      memory.peakBytes = memory.gpuPeakBytes ?? null;
      memory.peakAboveBaselineBytes = Number.isFinite(memory.gpuPeakBytes) ? Math.max(0, memory.gpuPeakBytes - memory.gpuBaselineBytes) : null;
    }
    memory.workerReportedPeakGpuMemoryPct = job.peakGpuMemoryPct ?? null;
    record.memory = memory;
    const at = (stage) => stages.find((item) => item.stage === stage)?.atSeconds;
    record.timing.claimSeconds ??= null;
    const loadStart = at("loading_model") ?? record.timing.claimSeconds, generate = at("generating") ?? at("rendering"), finish = at("saving") ?? at("muxing") ?? record.timing.enqueueToCompleteSeconds;
    record.timing.loadSeconds = Number.isFinite(loadStart) && Number.isFinite(generate) ? generate - loadStart : null;
    record.timing.denoiseSeconds = Number.isFinite(generate) ? finish - generate : null;
    record.timing.splitSource = "first poll (500 ms) observing job stage loading_model / generating / saving";
    // Terminal status can precede API persistence of the worker's asset writes.
    const persisted = await wait(async () => { const current = await endpoint.call("GET", `/api/v1/jobs/${accepted.id}`); return Array.isArray(current.result?.assets) && current.result.assets.length > 0 && current.result.assetWrites === undefined && current; }, 120, "API asset persistence");
    job = persisted;
    const assets = job.result.assets.filter((asset) => asset.type === route.kind || (!asset.type && String(asset.file?.mimeType ?? "").startsWith(`${route.kind}/`)));
    fact(assets.length === 1, `expected exactly one ${route.kind} asset, saw ${assets.length}`);
    const relative = assets[0].file?.path;
    fact(typeof relative === "string" && relative && !relative.split("/").includes(".."), "saved asset path invalid");
    const media = path.resolve(projectPath, ...relative.split("/"));
    fact(!path.relative(projectPath, media).startsWith(".."), "saved asset escaped the project");
    const served = await endpoint.served(`/api/v1/projects/${project.id}/files/${relative.split("/").map(encodeURIComponent).join("/")}`);
    const saved = await readFile(media);
    fact(sha256(served.bytes) === sha256(saved), "API-served bytes differ from the saved project file");
    const output = path.join(evidence, `output${path.extname(relative)}`); await writeFile(output, saved);
    const content = await decodedContent(output, route.kind, { ffmpeg: env.SCENEWORKS_FFMPEG || "ffmpeg" });
    record.output = { file: path.relative(evidenceRoot, output).split(path.sep).join("/"), bytes: saved.length, fileSha256: sha256(saved), content: { method: content.method, sha256: content.sha256, shape: content.shape, decoderError: content.decoderError }, mimeType: assets[0].file?.mimeType ?? null };
    decodedBuffers.set(record, content.buffer);
    record.identity.payload = redactPaths(identitySubset(job.payload), roots);
    record.identity.result = redactPaths({ adapter: job.result.adapter ?? null, model: job.result.model ?? null, backend: job.backend ?? null, assignedGpu: job.assignedGpu ?? null, workerId: job.workerId === workerId ? "<owned>" : job.workerId }, roots);
    fact(job.workerId === workerId, `job ran on ${job.workerId}, not the owned worker`);
    record.status = "completed";
  } catch (error) {
    record.status = "error"; record.error = error.message;
  } finally {
    clearTimeout(timer); signal.removeEventListener("abort", onAbort);
    sampling = false; await Promise.all(samplers);
    if (record.status !== "completed" && apiUrl) {
      // What the API serves right now about this job and the owned worker (direct requests: the
      // owned-endpoint gate refuses everything once the run has been aborted).
      const direct = (route) => fetch(`${apiUrl}${route}`, { signal: AbortSignal.timeout(10_000) }).then((response) => response.json(), () => null);
      if (accepted?.id) lastJob = (await direct(`/api/v1/jobs/${encodeURIComponent(accepted.id)}`)) ?? lastJob;
      const workers = await direct("/api/v1/workers");
      // Process state BEFORE teardown: is each child alive, and how did it exit if not.
      const processes = Object.fromEntries(children.filter((child) => child.label !== "sentinel").map((child) => [child.label, processState(child)]));
      record.diagnostics = { processes, workersResponse: workers === null ? null : JSON.parse(redactSensitiveText(JSON.stringify(workers), secrets)), lastJob: jobDiagnostics(lastJob), ownedWorker: Array.isArray(workers) ? workers.filter((row) => String(row?.id ?? "").startsWith(workerId)).map(({ id, status, gpuId, currentJobId, capabilities }) => ({ id, status, gpuId, currentJobId, capabilityCount: capabilities?.length ?? null })) : null };
    }
    // Direct request: the owned-endpoint gate refuses every call once the watchdog has aborted.
    if (accepted?.id && record.status !== "completed" && apiUrl) {
      record.cancelRequested = await fetch(`${apiUrl}/api/v1/jobs/${encodeURIComponent(accepted.id)}/cancel`, { method: "POST", headers: { "content-type": "application/json" }, body: "{}", redirect: "error", signal: AbortSignal.timeout(15_000) }).then((response) => response.status, (error) => error.message);
      // Give the worker a bounded chance to leave its owner drive before the graceful shutdown.
      for (let i = 0; i < 60; i++) { const status = await fetch(`${apiUrl}/api/v1/jobs/${encodeURIComponent(accepted.id)}`, { signal: AbortSignal.timeout(5_000) }).then((response) => response.json()).then((value) => value.status, () => null); if (!["queued", "running"].includes(status)) break; await delay(1000); }
    }
    record.cleanup = await teardown(children);
    endpoint?.close();
    await Promise.all(handles.map((handle) => handle.close().catch(() => {})));
    for (const name of ["api", "worker"]) {
      const text = await tail(path.join(logs, `${name}.log`), 2 * 1024 * 1024);
      if (text) await writeFile(path.join(evidence, `${name}.log`), redactSensitiveText(text, secrets));
      if (record.status !== "completed") (record.diagnostics ??= {})[`${name}LogTail`] = lastLines(redactSensitiveText(text, secrets), 50);
    }
    record.finishedAt = new Date().toISOString();
    await phase(record.status === "completed" ? "run_done" : "run_error", { side, job: accepted?.id, reason: record.status === "completed" ? undefined : record.error, last_status: record.diagnostics?.lastJob ? `${record.diagnostics.lastJob.status}/${record.diagnostics.lastJob.stage}` : undefined });
  }
  return record;
}
// Graceful first: ending the sentinel's stdin triggers the API/worker parent-death shutdown (the
// worker finishes or abandons its job without a signal mid Metal command buffer). Only a process
// still alive after that grace is signalled, and that is recorded.
async function teardown(children) {
  const results = [], sentinel = children.find((child) => child.label === "sentinel");
  const exited = (child) => child.exitCode !== null || child.signalCode !== null;
  const within = async (child, ms) => { let timer; try { return await Promise.race([child.done, new Promise((resolve) => { timer = setTimeout(() => resolve(null), ms); })]); } finally { clearTimeout(timer); } };
  // Close each service's stdin first (the worker's graceful-shutdown signal on Windows), then the
  // sentinel (the parent-death signal on every platform).
  for (const child of children) if (child.label !== "sentinel") child.stdin?.end();
  sentinel?.stdin?.end();
  for (const child of children.filter((item) => item.label !== "sentinel")) {
    if (!child.pid) continue;
    let forced = null;
    if (!exited(child)) await within(child, 60_000);
    if (!exited(child)) {
      if (process.platform === "win32") { forced = "taskkill /T /F"; await execFile("taskkill", ["/PID", String(child.pid), "/T", "/F"], { timeout: 20_000 }).catch(() => {}); }
      else { forced = "SIGTERM"; try { process.kill(-child.pid, "SIGTERM"); } catch { /* already gone */ } await within(child, 60_000); if (!exited(child)) { forced = "SIGKILL"; try { process.kill(-child.pid, "SIGKILL"); } catch { /* already gone */ } } }
      await within(child, 15_000);
    }
    results.push({ name: child.label, pid: child.pid, exited: exited(child), forced });
  }
  if (sentinel && !exited(sentinel)) sentinel.kill();
  return results;
}
// Two seconds of deterministic voiced harmonics (the load-progress acceptance reference).
export function referenceWav() {
  const samples = 48_000, data = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i++) { const t = i / 24_000; data.writeInt16LE(Math.trunc(Math.sin(2 * Math.PI * 170 * t) * 6000 + Math.sin(2 * Math.PI * 340 * t) * 2000), i * 2); }
  const header = Buffer.alloc(44);
  header.write("RIFF", 0); header.writeUInt32LE(36 + data.length, 4); header.write("WAVEfmt ", 8); header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22); header.writeUInt32LE(24_000, 24); header.writeUInt32LE(48_000, 28); header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34); header.write("data", 36); header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
}

async function runComparison(env, cfg, report, signal) {
  const scratch = path.join(cfg.work, "scratch"), evidence = path.join(cfg.work, "evidence");
  const recorded = JSON.parse(await readFile(path.join(scratch, "build.json"), "utf8"));
  fact(recorded.baseSha === cfg.baseSha && recorded.headSha === cfg.headSha && recorded.backend === cfg.backend, "build.json belongs to a different base/head/backend");
  const built = recorded.built;
  for (const side of ["base", "head"]) { fact(await fileSha256(built[side].binary) === built[side].binarySha256, `${side} binary changed since build`); report[side].binarySha256 = built[side].binarySha256; }
  const secrets = new Set([env.HF_TOKEN].filter((value) => value && value.length >= 8));
  for (const route of report.routes.filter((item) => item.status === "skipped")) progress({ route: route.id, phase: "skipped", reason: route.skipReason });
  let firstRoute = true;
  for (const route of report.routes.filter((item) => item.status === "ready")) {
    if (route.status !== "ready") continue;
    const definition = ROUTES[route.id];
    const repos = [...route.rows.required, ...route.rows.optional];
    const before = await hubFingerprint(route.hub, repos);
    route.runs = []; route.status = "running";
    progress({ route: route.id, phase: "route_start", model: route.model, hub: route.hub, claim_min: cfg.claimMinutesOverride ?? definition.claimMinutes, exec_min: definition.timeoutMinutes });
    await writeReport(cfg, report);
    const run = async (label) => {
      const runBefore = await hubFingerprint(route.hub, repos);
      const record = await executeRun({ env, cfg, route: definition, plan: route, label, side: label.startsWith("base") ? "base" : "head", built, evidenceRoot: evidence, secrets, signal });
      const runAfter = await hubFingerprint(route.hub, repos);
      record.hubMutation = { changed: JSON.stringify(runBefore.repos) !== JSON.stringify(runAfter.repos), diff: diffObjects(runBefore.repos, runAfter.repos, 50) };
      if (record.hubMutation.changed) progress({ route: route.id, run: label, phase: "hub_mutation", entries: record.hubMutation.diff.count });
      route.runs.push(record); await writeReport(cfg, report); return record;
    };
    const base1 = await run("base#1");
    // A worker that cannot register on the very first run will not register on the next five
    // either (same host, same launch); stop the whole comparison with that verdict.
    if (base1.failureClass === "worker_registration" && firstRoute) {
      route.status = "ran"; route.verdict = "error"; route.verdictReason = `base#1 worker registration failed: ${base1.error}`;
      for (const other of report.routes.filter((item) => item.status === "ready")) { other.status = "not_run"; other.skipReason = "stopped: the owned worker could not register on base#1 of the first route"; }
      report.fatal = `owned worker registration failed on base#1 of the first route (${base1.error}); remaining runs not attempted`;
      progress({ route: route.id, phase: "verdict", verdict: "error", reason: report.fatal });
      await writeReport(cfg, report);
      break;
    }
    firstRoute = false;
    const head = signal.aborted ? null : await run("head");
    let decision = verdictFor({ base1, head });
    let base2 = null;
    if (decision.needBase2 && !signal.aborted) { base2 = await run("base#2"); decision = verdictFor({ base1, head, base2 }); }
    if (decision.verdict === null) decision = { verdict: "error", reason: "interrupted before base#2" };
    route.status = "ran"; route.verdict = decision.verdict; route.verdictReason = decision.reason;
    route.comparison = { payloadIdentityDiff: diffObjects(base1.identity?.payload ?? {}, head?.identity?.payload ?? {}), resultIdentityDiff: diffObjects(base1.identity?.result ?? {}, head?.identity?.result ?? {}) };
    if (decision.verdict === "nondeterministic_on_runner") Object.assign(route.comparison, { numericDifferenceOf: "base#1 vs base#2", numericDifference: numericDifference(decodedBuffers.get(base1), decodedBuffers.get(base2)) });
    if (decision.verdict === "head_differs") Object.assign(route.comparison, { numericDifferenceOf: "base#1 vs head", numericDifference: numericDifference(decodedBuffers.get(base1), decodedBuffers.get(head)) });
    const after = await hubFingerprint(route.hub, repos);
    // Only the route's own repositories decide `changed`; the shared root listing can legitimately
    // change under another runner job, so it is recorded but never fails this one.
    route.hubMutation = { changed: JSON.stringify(before.repos) !== JSON.stringify(after.repos), rootChanged: JSON.stringify([before.root, before.hfHome]) !== JSON.stringify([after.root, after.hfHome]), diff: diffObjects(before, after, 50), method: "readdir/stat of the hub root, its parent, and each route repository's dir/refs/snapshots/blobs (metadata only)" };
    progress({ route: route.id, phase: "verdict", verdict: decision.verdict, reason: decision.reason });
    if (decision.verdict === "head_differs") process.stderr.write(`\n[tier2-ab] ${route.id}: head_differs. Payload identity diff (base#1 -> head):\n${JSON.stringify(route.comparison.payloadIdentityDiff, null, 2)}\n`);
    await writeReport(cfg, report);
    if (signal.aborted) break;
  }
}

// preflight.json/build.json evidence for the cheap modes; report.json/report.md for the comparison.
export const reportName = (mode) => (mode === "run" ? "report" : mode);
async function writeReport(cfg, report) {
  const dir = path.join(cfg.work, "evidence"); await mkdir(dir, { recursive: true });
  report.summary = summarize(report.routes ?? []);
  await writeFile(path.join(dir, `${reportName(report.mode)}.json`), JSON.stringify(report, null, 2) + "\n");
  await writeFile(path.join(dir, `${reportName(report.mode)}.md`), renderMarkdown(report));
}

// Workflow cleanup: the attempt-owned work dir and both side checkouts, never a hub root.
async function cleanup(env, cfg) {
  const allowedRoots = [env.RUNNER_TEMP, env.GITHUB_WORKSPACE].filter(Boolean);
  fact(allowedRoots.length, "--cleanup needs RUNNER_TEMP or GITHUB_WORKSPACE to bound what it may remove");
  const protectedRoots = [...cfg.hubs, ...defaultHubs(cfg.backend)];
  let failed = 0;
  for (const target of [cfg.work, cfg.baseDir, cfg.headDir]) {
    try { const result = await removeScratch(target, { allowedRoots, protectedRoots }); progress({ phase: "cleanup", target, removed: result.removed, restored_dirs: result.restoredDirectories, restored_files: result.restoredFiles }); }
    catch (error) { failed++; progress({ phase: "cleanup_failed", target, reason: error.message }); }
  }
  return failed ? 1 : 0;
}

export async function main(argv = process.argv.slice(2), env = process.env) {
  const modes = { "--preflight-only": "preflight", "--build-only": "build", "--run": "run", "--cleanup": "cleanup" };
  fact(argv.length === 1 && modes[argv[0]], "exactly one of --preflight-only, --build-only, --run, --cleanup is required");
  const mode = modes[argv[0]];
  const cfg = resolveConfig(env);
  if (mode === "cleanup") return cleanup(env, cfg);
  progress({ phase: "driver_started", mode, backend: cfg.backend, runner: env.RUNNER_NAME });
  const report = { schemaVersion: 1, kind: "sw-7033-tier2-ab", mode, status: "running", backend: cfg.backend, startedAt: new Date().toISOString(), runner: { name: env.RUNNER_NAME ?? null, platform: process.platform, arch: process.arch, totalMemoryBytes: os.totalmem(), runId: env.GITHUB_RUN_ID ?? null, runAttempt: env.GITHUB_RUN_ATTEMPT ?? null, dispatchSha: env.GITHUB_SHA ?? null }, base: { sha: cfg.baseSha }, head: { sha: cfg.headSha }, inputs: { routes: cfg.routes.map((route) => route.id), hubs: cfg.hubs }, memoryMethod: cfg.backend === "mlx" ? "worker process-tree phys_footprint (top MEM) + RSS, ~2 s cadence; watchdog cap min(100 GiB, 80% RAM)" : "physical GPU1 nvidia-smi memory.used peak above a 5-sample pre-request baseline, ~500 ms cadence; worker working-set watchdog at 75% RAM", routes: [] };
  const controller = new AbortController();
  const interrupt = () => controller.abort(new Error("A/B interrupted by signal"));
  process.once("SIGTERM", interrupt); process.once("SIGINT", interrupt);
  try {
    await preflight(env, cfg, report);
    if (mode === "build") await build(env, cfg, report);
    if (mode === "run") await runComparison(env, cfg, report, controller.signal);
    if (controller.signal.aborted) throw controller.signal.reason;
    report.status = "finished";
  } catch (error) {
    report.status = "failed"; report.fatal = error.message;
  } finally {
    process.removeListener("SIGTERM", interrupt); process.removeListener("SIGINT", interrupt);
    report.finishedAt = new Date().toISOString();
    report.summary = summarize(report.routes);
    report.exitCode = mode === "build" ? (report.fatal ? 1 : 0) : exitCodeFor(report);
    await writeReport(cfg, report).catch((error) => { process.stderr.write(`report write failed: ${error.message}\n`); report.exitCode = 1; });
  }
  process.stdout.write(renderMarkdown(report));
  return report.exitCode;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().then((code) => { process.exitCode = code; }).catch((error) => { process.stderr.write(`${error.stack ?? error.message}\n`); process.exitCode = 1; });
}
