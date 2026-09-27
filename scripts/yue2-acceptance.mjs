#!/usr/bin/env node
// YuE2 terminal acceptance driver (sc-23002, epic 22988): runs the epic's acceptance matrix through
// the REAL app — the source-built API, a worker started from the same binary
// (`SCENEWORKS_WORKER_ONLY=1`) and the registered `yue2` provider on this host's production device
// (Metal on macOS, CUDA under `backend-candle` on Windows). No surrogate loader, no test seam: every
// case submits through the HTTP API and every assertion reads the API's responses and the run
// artifacts the worker published into the app's data directory.
//
//   node scripts/yue2-acceptance.mjs --platform metal|cuda --out <dir>
//        [--data-dir <dir>] [--hf-hub <dir>] [--api-bin <path>] [--skip <case>]... [--dry-run]
//        [--gpu-id N] [--port N] [--allow-metal-worker-kill] [--job-timeout-minutes N]
//
// Runbook (Metal under the watchdog, CUDA by dispatch): docs/epic-22988-yue2-terminal.md.
//
// OWNERSHIP. The driver starts the API and the worker as children in ITS OWN process group — it
// never detaches them — so an external guard wrapping the driver (on macOS
// scripts/memory-calibration-watchdog.py, which samples and terminates exactly one process group)
// covers both, and it stops both on exit, on failure and on SIGINT/SIGTERM. An idle service is
// stopped with SIGTERM; a render in flight is cancelled through the API first (the engine's own
// cancel path) and only then stopped.
//
// EVIDENCE. `<out>/evidence/` holds one record per case (`records/<case>.json`), `summary.json`,
// `summary.md` and the service logs. Outputs are CC BY-NC 4.0: audio is NEVER copied into the
// evidence — it stays in the app data dir (`--data-dir`, default `<out>/app-data`) and the records
// carry its SHA-256, length and RMS. Upload `<out>/evidence` only.
//
// DRY RUN (`--dry-run`) starts the API alone — no worker, so nothing loads weights or renders — and
// runs only the cases that need no worker (catalog, isolation, licence gate, transcription refusal).
// Every other case is recorded `skipped` with that reason; a dry run never produces a passing verdict.
//
// NOT HERE. AT1's "long request exercising chunk/context boundaries" is the q4 `long-context` case of
// scripts/yue2-memory-profile.mjs (config/yue2-memory-profile-plan.json), which runs it through the
// same job path (`yue2_jobs::resolve_load` / `build_request`); the summary names it under
// `coveredElsewhere`.
import { spawn, execFile as execFileCallback } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import { copyFile, mkdir, open, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { createServer } from "node:net";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";

import { fileSha256 } from "./lib/file-sha256.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

const execFile = promisify(execFileCallback);

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const RECORD_SCHEMA = "sceneworks-yue2-acceptance-record-v1";
export const SUMMARY_SCHEMA = "sceneworks-yue2-acceptance-summary-v1";
export const MODEL_ID = "yue2";
export const STATUSES = Object.freeze(["passed", "failed", "blocked", "skipped"]);
export const PLATFORMS = Object.freeze({
  metal: Object.freeze({ os: "darwin", gpuId: "mlx", device: "metal", capEnv: "SCENEWORKS_MLX_MEMORY_CAP_GB" }),
  cuda: Object.freeze({ os: "win32", gpuId: "0", device: "cuda", capEnv: "SCENEWORKS_CUDA_VRAM_CAP_GB" }),
});
/** The engine's semantic token rate (16 000 tokens = 640 s, config/yue2-memory-profile-plan.json). */
export const SEMANTIC_TOKENS_PER_SECOND = 25;
/** Songs stay short (~30–45 s): the semantic budget every rendering case sends. */
export const SONG_SEMANTIC_MAX_TOKENS = 45 * SEMANTIC_TOKENS_PER_SECOND;
/** The capacity (GB) the admission case caps the worker at, and the request priced far above it. */
export const ADMISSION_CAP_GB = 24;
/** Files every published song / cached-decode run records (`candle_audio_yue2::run::verify_run`). */
export const SONG_RUN_FILES = Object.freeze([
  "plan.json", "abc_tokens.npy", "prefix.npy", "plan_manifest.json", "semantic.npy", "latent.npy",
  "latent.json", "audio.wav", "request.json", "config.json",
]);
export const PLAN_RUN_FILES = Object.freeze(["plan.json", "abc_tokens.npy", "prefix.npy", "plan_manifest.json"]);
export const COVERED_ELSEWHERE = Object.freeze([
  Object.freeze({
    acceptance: "AT1: at least one long request exercising chunk/context boundaries",
    by: "scripts/yue2-memory-profile.mjs case yue2:q4:<metal|cuda>:long-context (planning off, 16 000 forced semantic tokens, two acoustic chunks) — run it with `run --backend <metal|cuda>`",
  }),
]);

export function fail(message) {
  throw new Error(`yue2-acceptance: ${message}`);
}

// ---- Cases ---------------------------------------------------------------------------------------
//
// `dry`: the case needs no worker, so it also runs in a dry run. `needs`: cases whose published
// outputs (or installed weights) this case relies on; `dryNeeds` replaces them in a dry run. A case
// whose dependency did not pass is recorded `skipped` with the dependency's own status and reason —
// never silently dropped.

const c = (id, acceptance, title, { dry = false, needs = [], dryNeeds = null, expectBlocked = false } = {}) =>
  Object.freeze({
    id,
    acceptance: Object.freeze(acceptance),
    title,
    dry,
    needs: Object.freeze(needs),
    dryNeeds: dryNeeds && Object.freeze(dryNeeds),
    expectBlocked,
  });

export const CASES = Object.freeze([
  c("catalog-preflight", ["AT4", "AT5"], "Catalog install plan, pinned rows and derivations read through the API", { dry: true }),
  c("isolation-v1-v2", ["AT5"], "V1/V2 isolation: distinct ids, repos, licences and defaults; V2 refused on a commercial-use route and on the generic audio route", { dry: true }),
  c("transcription-blocked", ["AT2"], "Recording transcription refused with component_blocked (recorded as blocked)", { dry: true, expectBlocked: true }),
  c("install-cold", ["AT4"], "Cold install through the app's install flow: bf16 + both decoders, q8/q4 derived locally"),
  c("licence-gate", ["AT5"], "Licence acknowledgment gate: refused without it, allowed (and executed) with it", { dry: true, needs: ["install-cold"], dryNeeds: [] }),
  c("create-cot-full", ["AT1"], "create, cot=full, EN lyrics, bf16 tier", { needs: ["install-cold"] }),
  c("create-cot-melody", ["AT1"], "create, cot=melody, q8 tier", { needs: ["install-cold"] }),
  c("create-cot-off", ["AT1"], "create, cot=off, q4 tier", { needs: ["install-cold"] }),
  c("supplied-abc-full", ["AT1"], "create from a supplied full ABC score (planning full)", { needs: ["install-cold"] }),
  c("supplied-abc-melody", ["AT1"], "create from a supplied melody ABC score (planning melody)", { needs: ["install-cold"] }),
  c("lyrics-en-zh", ["AT1"], "English and Chinese lyrics (ZH rendered here; EN is create-cot-full)", { needs: ["install-cold", "create-cot-full"] }),
  c("plan-restore", ["AT1"], "plan-only, then restore (fromPlan) with identical plan tokens", { needs: ["install-cold"] }),
  c("cover-melody", ["AT2"], "cover of a reviewed score version, melody mode", { needs: ["install-cold"] }),
  c("cover-full", ["AT2"], "cover of a reviewed score version, full mode", { needs: ["install-cold"] }),
  c("score-edits", ["AT2"], "harmony-only invariant-preserving edit + lyric/style/tempo/form revisions, each rendered (renderVersion), versions and renders read back", { needs: ["install-cold"] }),
  c("decode-standard-legacy", ["AT3"], "standard and legacy cached decodes of ONE run's latent, same latent identity", { needs: ["install-cold", "create-cot-full"] }),
  c("artifact-corrupt", ["AT4"], "corrupt source artifact refused, then repaired by restoring the recorded bytes", { needs: ["install-cold", "create-cot-full"] }),
  c("artifact-missing", ["AT4"], "missing source run refused, then recovered when restored", { needs: ["install-cold", "create-cot-full"] }),
  c("cancel-ar", ["AT4"], "cancel during AR (semantic tokens): clean cancel, only .partial removed", { needs: ["install-cold"] }),
  c("cancel-nar", ["AT4"], "cancel during NAR (acoustic synthesis): clean cancel, only .partial removed", { needs: ["install-cold"] }),
  c("cancel-decode", ["AT4"], "cancel during decode: clean cancel, only .partial removed", { needs: ["install-cold", "create-cot-full"] }),
  c("worker-kill-resume", ["AT4"], "worker killed mid-run, restarted, retried job resumes with exact identity", { needs: ["install-cold"] }),
  c("admission", ["AT4"], "admission: a fitting render admitted and an over-budget request refused before load at the same capacity", { needs: ["install-cold"] }),
  c("batch-serial", ["AT4"], "queued 2-take batch renders serially", { needs: ["install-cold"] }),
  c("offline-restart", ["AT4"], "restart offline (HF_HUB_OFFLINE=1, hub unreachable): installed and renders; install needs the network and recovers online", { needs: ["install-cold", "create-cot-full"] }),
]);

export function caseById(id) {
  const found = CASES.find((item) => item.id === id);
  if (!found) fail(`unknown case ${id}; cases: ${CASES.map((item) => item.id).join(", ")}`);
  return found;
}

/**
 * What runs and what is skipped, and why, before anything starts. A skip always carries a reason.
 * Dependency-driven skips are decided at run time (`dependencySkip`), from the dependency's record.
 */
export function planCases({ platform, skip = [], dryRun = false, allowMetalWorkerKill = false } = {}) {
  if (!PLATFORMS[platform]) fail(`--platform must be metal or cuda, not ${platform}`);
  for (const id of skip) caseById(id);
  return CASES.map((item) => {
    if (skip.includes(item.id)) return { id: item.id, action: "skip", reason: "skipped by the operator (--skip)" };
    if (dryRun && !item.dry) {
      return { id: item.id, action: "skip", reason: "dry run: no worker is started, so nothing that loads weights, installs or renders runs" };
    }
    if (item.id === "worker-kill-resume" && platform === "metal" && !allowMetalWorkerKill) {
      return {
        id: item.id,
        action: "skip",
        reason:
          "a signal-kill of a Metal process mid-command-buffer can wedge this Mac's GPU client until a reboot " +
          "(kIOGPUCommandBufferCallbackErrorSubmissionsIgnored); run it only with --allow-metal-worker-kill, on the owner's say-so",
      };
    }
    return { id: item.id, action: "run", reason: null };
  });
}

/** A dependency that did not pass makes this case `skipped`, naming it and its own outcome. */
export function dependencySkip(item, records, { dryRun = false } = {}) {
  for (const need of (dryRun && item.dryNeeds) || item.needs) {
    const record = records.get(need);
    if (!record) return `dependency ${need} has not run`;
    if (record.status !== "passed") return `dependency ${need} ${record.status}: ${record.reason ?? "no reason recorded"}`;
  }
  return null;
}

// ---- Arguments -------------------------------------------------------------------------------------

export function parseArgs(argv) {
  const options = { skip: [], dryRun: false, allowMetalWorkerKill: false, jobTimeoutMinutes: 90 };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const value = () => {
      const next = argv[index + 1];
      if (next === undefined || next.startsWith("--")) fail(`${arg} needs a value`);
      index += 1;
      return next;
    };
    if (arg === "--platform") options.platform = value();
    else if (arg === "--out") options.out = value();
    else if (arg === "--data-dir") options.dataDir = value();
    else if (arg === "--hf-hub") options.hfHub = value();
    else if (arg === "--api-bin") options.apiBin = value();
    else if (arg === "--skip") options.skip.push(value());
    else if (arg === "--gpu-id") options.gpuId = value();
    else if (arg === "--port") options.port = Number(value());
    else if (arg === "--job-timeout-minutes") options.jobTimeoutMinutes = Number(value());
    else if (arg === "--dry-run") options.dryRun = true;
    else if (arg === "--allow-metal-worker-kill") options.allowMetalWorkerKill = true;
    else fail(`unknown argument ${arg}`);
  }
  if (!PLATFORMS[options.platform]) fail("--platform metal|cuda is required");
  if (!options.out) fail("--out <dir> is required");
  if (options.port !== undefined && (!Number.isInteger(options.port) || options.port < 1 || options.port > 65535)) fail("--port must be a TCP port");
  if (!Number.isFinite(options.jobTimeoutMinutes) || options.jobTimeoutMinutes <= 0) fail("--job-timeout-minutes must be positive");
  for (const id of options.skip) caseById(id);
  return options;
}

// ---- Job progress → stages -------------------------------------------------------------------------

const STAGE_MESSAGES = [
  ["eligibility", "Checking YuE2 eligibility"],
  ["load", "Loading YuE2"],
  ["plan", "Planning the score"],
  ["semantic", "Generating semantic tokens"],
  ["acoustic", "Acoustic synthesis"],
  ["decode", "Decoding audio"],
  ["saving", "Saving the YuE2 run"],
];

/** The pipeline stage a job snapshot is in, from its status and the worker's progress message. */
export function stageOf(snapshot) {
  const status = snapshot?.status;
  if (["completed", "failed", "canceled", "interrupted"].includes(status)) return status;
  if (["queued", "pending_caption", "pending_workflow"].includes(status)) return "queued";
  const message = String(snapshot?.message ?? "");
  // The API's own note while an active job waits for its worker to acknowledge a cancel.
  if (message.startsWith("Cancellation requested")) return "cancel-requested";
  for (const [stage, prefix] of STAGE_MESSAGES) if (message.startsWith(prefix)) return stage;
  return status ?? "unknown";
}

/**
 * Per-stage wall time from the observed stage sequence: each stage runs from the first poll that saw
 * it to the first poll that saw the next stage. Poll-granular (the driver polls every ~250 ms); the
 * engine's own per-stage timings (`result.json` `timing`) are recorded beside it.
 */
export function stageTimeline(observations) {
  const firsts = [];
  for (const observation of observations) {
    if (!firsts.length || firsts.at(-1).stage !== observation.stage) firsts.push({ stage: observation.stage, at: observation.at });
  }
  const stages = {};
  for (let index = 0; index < firsts.length; index += 1) {
    const { stage, at } = firsts[index];
    const next = firsts[index + 1];
    const seconds = next ? (next.at - at) / 1000 : 0;
    stages[stage] = (stages[stage] ?? 0) + seconds;
  }
  return { order: firsts.map((entry) => entry.stage), seconds: stages };
}

// ---- WAV -------------------------------------------------------------------------------------------

/** Duration and RMS of a PCM16 / IEEE-float32 WAV (the asset and the run's `audio.wav`). */
export function parseWav(buffer) {
  if (buffer.length < 12 || buffer.toString("ascii", 0, 4) !== "RIFF" || buffer.toString("ascii", 8, 12) !== "WAVE") fail("not a RIFF/WAVE file");
  let format = null;
  let data = null;
  for (let at = 12; at + 8 <= buffer.length;) {
    const id = buffer.toString("ascii", at, at + 4);
    const size = buffer.readUInt32LE(at + 4);
    const body = at + 8;
    if (id === "fmt ") {
      let tag = buffer.readUInt16LE(body);
      if (tag === 0xfffe && size >= 26) tag = buffer.readUInt16LE(body + 24);
      format = { tag, channels: buffer.readUInt16LE(body + 2), sampleRate: buffer.readUInt32LE(body + 4), bits: buffer.readUInt16LE(body + 14) };
    } else if (id === "data") {
      data = { start: body, size: Math.min(size, buffer.length - body) };
    }
    at = body + size + (size % 2);
  }
  if (!format || !data) fail("the WAV has no fmt or data chunk");
  let read;
  if (format.tag === 1 && format.bits === 16) read = (offset) => buffer.readInt16LE(offset) / 32768;
  else if (format.tag === 3 && format.bits === 32) read = (offset) => buffer.readFloatLE(offset);
  else fail(`unsupported WAV encoding (tag ${format.tag}, ${format.bits} bits)`);
  const bytes = format.bits / 8;
  const samples = Math.floor(data.size / bytes);
  let sum = 0;
  for (let index = 0; index < samples; index += 1) {
    const value = read(data.start + index * bytes);
    sum += value * value;
  }
  const frames = samples / format.channels;
  return {
    encoding: format.tag === 1 ? "pcm16" : "float32",
    sampleRate: format.sampleRate,
    channels: format.channels,
    frames,
    durationSeconds: frames / format.sampleRate,
    rms: samples ? Math.sqrt(sum / samples) : 0,
  };
}

// ---- Memory sampling -------------------------------------------------------------------------------

/** Sum of `phys_footprint` over the requested pids from `/usr/bin/footprint -j` output. */
export function footprintBytes(payload, pids) {
  const wanted = new Set(pids);
  let total = 0;
  let seen = 0;
  for (const process of payload?.processes ?? []) {
    const value = process?.auxiliary?.phys_footprint;
    if (!wanted.has(process?.pid)) continue;
    if (!Number.isSafeInteger(value) || value < 0) fail(`footprint gave no phys_footprint for pid ${process?.pid}`);
    total += value;
    seen += 1;
  }
  if (!seen) fail("footprint described none of the service pids");
  return total;
}

/** `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` → bytes. */
export function nvidiaUsedBytes(text) {
  const mib = Number(String(text).trim().split(/\r?\n/)[0]);
  if (!Number.isFinite(mib) || mib < 0) fail(`nvidia-smi reported no memory.used (${JSON.stringify(text)})`);
  return mib * 1024 * 1024;
}

/** Peak of a sample series, with the sampler's gaps kept visible. */
export function peakOf(samples, faults, sampler) {
  const good = samples.filter((sample) => Number.isFinite(sample.bytes));
  return {
    sampler,
    samples: good.length,
    faults: faults.length,
    firstFault: faults[0] ?? null,
    peakBytes: good.length ? Math.max(...good.map((sample) => sample.bytes)) : null,
    baselineBytes: good.length ? good[0].bytes : null,
  };
}

// ---- Evidence --------------------------------------------------------------------------------------

/** Whether the provenance device is the platform's production device (never a CPU fallback). */
export function deviceMatches(platform, device) {
  return PLATFORMS[platform]?.device === device;
}

/**
 * The assertions every completed render's evidence must satisfy: it ran on the production device, the
 * app's run record is the engine's (identity, truncation), the engine's artifact manifest verifies
 * byte for byte, and the audio is 48 kHz stereo and audible.
 */
export function renderAssertions({ platform, job, runResult, runConfig, artifactCheck, runAudio }) {
  const y2 = job?.result?.yue2 ?? {};
  const out = [];
  const check = (name, ok, detail) => out.push({ name, ok: Boolean(ok), detail });
  check("job completed", job?.status === "completed", `status ${job?.status}${job?.error ? `: ${job.error}` : ""}`);
  check("production device", deviceMatches(platform, runConfig?.device), `config.json device ${runConfig?.device}, expected ${PLATFORMS[platform]?.device}`);
  check("run identity", typeof runResult?.identity === "string" && runResult.identity === y2.run?.identity, `result.json ${runResult?.identity} vs job ${y2.run?.identity}`);
  const truncation = runResult?.truncated;
  check(
    "truncation reported",
    truncation && typeof truncation.abc === "boolean" && typeof truncation.semantic === "boolean" &&
      y2.truncated?.abc === truncation.abc && y2.truncated?.semantic === truncation.semantic,
    `engine ${JSON.stringify(truncation)} vs job ${JSON.stringify(y2.truncated)}`,
  );
  check("artifacts verify", artifactCheck?.ok === true, artifactCheck?.detail ?? "not checked");
  check("48 kHz stereo", runAudio?.sampleRate === 48000 && runAudio?.channels === 2, `${runAudio?.sampleRate} Hz x ${runAudio?.channels}`);
  check("audible", Number.isFinite(runAudio?.rms) && runAudio.rms > 1e-3, `rms ${runAudio?.rms}`);
  return out;
}

export function validateRecord(record) {
  const where = record?.caseId ?? "?";
  if (record?.schema !== RECORD_SCHEMA) fail(`${where}: not a ${RECORD_SCHEMA} record`);
  if (!CASES.some((item) => item.id === record.caseId)) fail(`${where}: unknown case`);
  if (!STATUSES.includes(record.status)) fail(`${where}: unknown status ${record.status}`);
  if (!PLATFORMS[record.platform]) fail(`${where}: unknown platform ${record.platform}`);
  if (record.status !== "passed" && !(typeof record.reason === "string" && record.reason.trim())) {
    fail(`${where}: a ${record.status} case must say why`);
  }
  if (!Array.isArray(record.assertions) || !Array.isArray(record.requests) || !Array.isArray(record.jobs)) fail(`${where}: assertions, requests and jobs must be arrays`);
  if (record.status === "blocked" && !caseById(record.caseId).expectBlocked) fail(`${where}: only an expected refusal may be recorded as blocked`);
  if (record.status === "passed") {
    if (!record.assertions.length) fail(`${where}: a passed case asserted nothing`);
    const failed = record.assertions.filter((assertion) => !assertion.ok);
    if (failed.length) fail(`${where}: passed with failing assertions: ${failed.map((a) => a.name).join(", ")}`);
    for (const job of record.jobs.filter((entry) => entry.rendered)) {
      if (!/^[0-9a-f]{64}$/.test(job.output?.runAudioSha256 ?? "")) fail(`${where}: rendered job ${job.jobId} has no output sha256`);
      if (!Number.isFinite(job.output?.durationSeconds) || !Number.isFinite(job.output?.rms)) fail(`${where}: rendered job ${job.jobId} has no duration/rms`);
      if (typeof job.truncated?.abc !== "boolean" || typeof job.truncated?.semantic !== "boolean") fail(`${where}: rendered job ${job.jobId} has no truncation flags`);
      if (!job.device || !job.dtype?.model) fail(`${where}: rendered job ${job.jobId} has no device/dtype provenance`);
      if (!job.runIdentity) fail(`${where}: rendered job ${job.jobId} has no run identity`);
      if (!job.stageTimings?.observed) fail(`${where}: rendered job ${job.jobId} has no stage timings`);
      if (!Number.isFinite(job.durationSeconds)) fail(`${where}: rendered job ${job.jobId} has no duration`);
    }
    if (record.memorySampled && !(record.peakMemory && Number.isFinite(record.peakMemory.peakBytes))) {
      fail(`${where}: a passed case has no peak memory sample (${record.peakMemory?.firstFault ?? "no sample"})`);
    }
  }
  return record;
}

/** The run's verdict: `pass` only when every case passed (or was refused as designed) and none was skipped. */
export function buildSummary(records, meta) {
  const counts = Object.fromEntries(STATUSES.map((status) => [status, 0]));
  for (const record of records) counts[record.status] += 1;
  const missing = CASES.filter((item) => !records.some((record) => record.caseId === item.id)).map((item) => item.id);
  let verdict = "pass";
  // A run that stopped early (`meta.fatal`) fails: what it recorded stands, and it is never "incomplete".
  if (counts.failed || meta.fatal) verdict = "fail";
  else if (meta.dryRun || counts.skipped || missing.length) verdict = "incomplete";
  return {
    schema: SUMMARY_SCHEMA,
    verdict,
    platform: meta.platform,
    dryRun: Boolean(meta.dryRun),
    fatal: meta.fatal ?? null,
    counts,
    missing,
    identity: meta.identity ?? null,
    coveredElsewhere: COVERED_ELSEWHERE,
    startedAt: meta.startedAt ?? null,
    finishedAt: meta.finishedAt ?? null,
    cases: records.map((record) => ({
      caseId: record.caseId,
      acceptance: record.acceptance,
      status: record.status,
      reason: record.reason ?? null,
      durationSeconds: record.durationSeconds ?? null,
      jobs: record.jobs.map((job) => ({
        jobId: job.jobId,
        kind: job.kind,
        status: job.status,
        tier: job.tier ?? null,
        runIdentity: job.runIdentity ?? null,
        outputSha256: job.output?.runAudioSha256 ?? null,
        audioSeconds: job.output?.durationSeconds ?? null,
        rms: job.output?.rms ?? null,
        truncated: job.truncated ?? null,
        device: job.device ?? null,
        dtype: job.dtype ?? null,
      })),
      peakMemoryBytes: record.peakMemory?.peakBytes ?? null,
    })),
  };
}

const gib = (bytes) => (Number.isFinite(bytes) ? `${(bytes / 1024 ** 3).toFixed(2)} GiB` : "—");
const cell = (value) => String(value ?? "—").replaceAll("|", "\\|").replaceAll("\n", " ");

export function renderMarkdown(summary) {
  const lines = [
    `# YuE2 terminal acceptance — ${summary.platform}${summary.dryRun ? " (dry run)" : ""}`,
    "",
    `Verdict: **${summary.verdict}** — ${STATUSES.map((status) => `${summary.counts[status]} ${status}`).join(", ")}${summary.missing.length ? `; not run: ${summary.missing.join(", ")}` : ""}.`,
    "",
  ];
  if (summary.identity) {
    lines.push(`SceneWorks ${summary.identity.sceneworksRevision ?? "?"}${summary.identity.dirty ? " (dirty)" : ""}, inference pin ${summary.identity.inferencePin ?? "?"}, binary sha256 ${summary.identity.apiBinarySha256 ?? "?"}.`, "");
  }
  lines.push(
    "| Case | Acceptance | Status | Jobs (kind · tier · device/dtype) | Output sha256 | Audio s | RMS | Truncated (abc/semantic) | Peak memory | Reason |",
    "|---|---|---|---|---|---|---|---|---|---|",
  );
  for (const item of summary.cases) {
    const rendered = item.jobs.filter((job) => job.outputSha256);
    const jobs = item.jobs.map((job) => `${job.kind ?? "?"}·${job.tier ?? "-"}·${job.device ?? "-"}/${job.dtype?.model ?? "-"} ${job.status}`).join("<br>");
    lines.push(`| ${cell(item.caseId)} | ${cell(item.acceptance.join(", "))} | ${item.status} | ${cell(jobs || "—")} | ${cell(rendered.map((job) => job.outputSha256.slice(0, 16)).join("<br>") || "—")} | ${cell(rendered.map((job) => job.audioSeconds?.toFixed(1)).join("<br>") || "—")} | ${cell(rendered.map((job) => job.rms?.toFixed(4)).join("<br>") || "—")} | ${cell(rendered.map((job) => `${job.truncated?.abc}/${job.truncated?.semantic}`).join("<br>") || "—")} | ${gib(item.peakMemoryBytes)} | ${cell(item.reason)} |`);
  }
  lines.push("", "Covered by another harness:");
  for (const note of summary.coveredElsewhere) lines.push(`- ${note.acceptance}: ${note.by}`);
  lines.push("", "Audio outputs are CC BY-NC 4.0 and stay on the capture host; this bundle carries their hashes only.", "");
  return lines.join("\n");
}

/** The service environment: isolated data/config dirs, the loopback API, the platform's worker device. */
export function serviceEnv({ platform, base = process.env, url, port, dataDir, configDir, hfHub, workerId, gpuId, offline = false, extra = {} }) {
  const env = { ...base };
  for (const inherited of ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_ENDPOINT", "SCENEWORKS_ACCESS_TOKEN", "SCENEWORKS_WORKER_ONLY", "HF_HUB_OFFLINE", "SCENEWORKS_HUGGINGFACE_BASE_URL", "SCENEWORKS_MLX_MEMORY_CAP_GB", "SCENEWORKS_CUDA_VRAM_CAP_GB", "SCENEWORKS_RUN_UTILITY_INPROCESS"]) delete env[inherited];
  Object.assign(env, {
    SCENEWORKS_API_HOST: "127.0.0.1",
    SCENEWORKS_API_PORT: String(port),
    SCENEWORKS_API_URL: url,
    SCENEWORKS_DATA_DIR: dataDir,
    SCENEWORKS_CONFIG_DIR: configDir,
    SCENEWORKS_JOBS_DB_PATH: path.join(dataDir, "cache", "jobs.db"),
    SCENEWORKS_WORKER_ID: workerId,
    SCENEWORKS_GPU_ID: gpuId ?? PLATFORMS[platform].gpuId,
    HF_HUB_CACHE: hfHub,
    // The cancel check rides the heartbeat (clamped to >= 5 s); the shortest keeps cancels prompt.
    SCENEWORKS_HEARTBEAT_SECONDS: "5",
  });
  if (offline) {
    // The app reads no HF_HUB_OFFLINE; the unreachable hub base is what actually takes the network
    // away from the worker's download path (discard port on loopback).
    env.HF_HUB_OFFLINE = "1";
    env.TRANSFORMERS_OFFLINE = "1";
    env.SCENEWORKS_HUGGINGFACE_BASE_URL = "http://127.0.0.1:9";
  }
  Object.assign(env, extra);
  return env;
}

// ---- Runtime ---------------------------------------------------------------------------------------

/** Whether `target` is the repository or inside it (a different Windows drive is outside). */
export function insideRepository(target, root = ROOT) {
  const relative = path.relative(root, path.resolve(target));
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const nowIso = () => new Date().toISOString();

async function freePort() {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen({ host: "127.0.0.1", port: 0 }, () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

async function git(args) {
  return (await execFile("git", ["-C", ROOT, ...args])).stdout.trim();
}

async function runIdentity(apiBin) {
  const cargo = await readFile(path.join(ROOT, "Cargo.toml"), "utf8");
  return {
    sceneworksRevision: await git(["rev-parse", "HEAD"]),
    dirty: (await git(["status", "--porcelain"])).length > 0,
    inferencePin: cargo.match(/SceneWorks\/inference",\s*rev\s*=\s*"([0-9a-f]{40})"/)?.[1] ?? null,
    apiBinary: path.relative(ROOT, apiBin),
    apiBinarySha256: await fileSha256(apiBin),
    node: process.version,
    host: { platform: process.platform, arch: process.arch, release: os.release(), memoryBytes: os.totalmem() },
  };
}

class Service {
  constructor(options, paths) {
    this.options = options;
    this.paths = paths;
    this.api = null;
    this.worker = null;
    this.workerId = `yue2-acceptance-${randomBytes(6).toString("hex")}`;
    this.offline = false;
    this.workerExtra = {};
  }

  env(role) {
    return serviceEnv({
      platform: this.options.platform,
      url: this.url,
      port: this.port,
      dataDir: this.paths.dataDir,
      configDir: this.paths.configDir,
      hfHub: this.paths.hfHub,
      workerId: this.workerId,
      gpuId: this.options.gpuId,
      offline: this.offline,
      extra: role === "worker" ? { SCENEWORKS_WORKER_ONLY: "1", ...this.workerExtra } : {},
    });
  }

  async spawnChild(role) {
    const log = await open(path.join(this.paths.logs, `${role}.log`), "a");
    // NOT detached: the child stays in the driver's process group, which the external guard samples.
    const child = spawn(this.options.apiBin, [], { cwd: ROOT, env: this.env(role), stdio: ["ignore", log.fd, log.fd], windowsHide: true });
    await log.close();
    child.spawnError = null;
    child.once("error", (error) => { child.spawnError = error; });
    return child;
  }

  alive(child) {
    return Boolean(child && child.exitCode === null && child.signalCode === null && !child.spawnError);
  }

  async request(method, route, body, { timeoutMs = 120_000 } = {}) {
    const response = await fetch(new URL(route, this.url), {
      method,
      headers: body === undefined ? {} : { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
    const text = await response.text();
    let json = null;
    try { json = text ? JSON.parse(text) : null; } catch { json = { raw: text }; }
    return { status: response.status, body: json };
  }

  async startApi() {
    this.port ??= this.options.port ?? await freePort();
    this.url = `http://127.0.0.1:${this.port}`;
    this.api = await this.spawnChild("api");
    for (let attempt = 0; attempt < 240; attempt += 1) {
      if (!this.alive(this.api)) fail(`the API exited during startup (see ${path.join(this.paths.logs, "api.log")})`);
      try {
        const health = await this.request("GET", "/api/v1/health", undefined, { timeoutMs: 2000 });
        if (health.status === 200 && health.body?.readiness?.status === "ready") return health.body;
      } catch { /* booting */ }
      await sleep(500);
    }
    fail("the API did not become ready in 120 s");
  }

  async startWorker() {
    this.worker = await this.spawnChild("worker");
    for (let attempt = 0; attempt < 360; attempt += 1) {
      if (!this.alive(this.worker)) fail(`the worker exited during startup (see ${path.join(this.paths.logs, "worker.log")})`);
      const workers = await this.request("GET", "/api/v1/workers");
      const mine = Array.isArray(workers.body) ? workers.body.find((worker) => worker.id === this.workerId) : null;
      if (mine && mine.status === "idle" && mine.currentJobId == null) return mine;
      await sleep(500);
    }
    fail("the worker did not register idle in 180 s");
  }

  pids() {
    return [this.api, this.worker].filter((child) => this.alive(child)).map((child) => child.pid);
  }

  /** Stop one child: SIGTERM (taskkill /T on Windows), escalate after `graceMs`. */
  async stopChild(child, { force = false, graceMs = 15_000 } = {}) {
    if (!this.alive(child)) return;
    const exited = new Promise((resolve) => child.once("exit", resolve));
    if (process.platform === "win32") {
      await execFile("taskkill.exe", ["/PID", String(child.pid), "/T", "/F"], { windowsHide: true })
        .catch((error) => console.error(`yue2-acceptance: taskkill ${child.pid}: ${error.message}`));
    } else {
      child.kill(force ? "SIGKILL" : "SIGTERM");
    }
    const done = await Promise.race([exited.then(() => true), sleep(graceMs).then(() => false)]);
    if (!done && process.platform !== "win32") {
      child.kill("SIGKILL");
      await Promise.race([exited, sleep(graceMs)]);
    }
    if (this.alive(child)) fail(`pid ${child.pid} did not exit`);
  }

  /** Cancel any active job through the API (the engine's own cancel path) and wait for the worker to idle. */
  async quiesce({ timeoutMs = 15 * 60_000 } = {}) {
    if (!this.alive(this.api)) return;
    const deadline = Date.now() + timeoutMs;
    const jobs = await this.request("GET", "/api/v1/jobs").catch(() => null);
    const active = (Array.isArray(jobs?.body) ? jobs.body : jobs?.body?.items ?? []).filter((job) => ["preparing", "downloading", "loading_model", "running", "saving", "queued"].includes(job.status));
    for (const job of active) await this.request("POST", `/api/v1/jobs/${job.id}/cancel`).catch(() => {});
    while (this.alive(this.worker) && Date.now() < deadline) {
      const workers = await this.request("GET", "/api/v1/workers").catch(() => null);
      const mine = Array.isArray(workers?.body) ? workers.body.find((worker) => worker.id === this.workerId) : null;
      if (!mine || (mine.status !== "busy" && mine.currentJobId == null)) return;
      await sleep(1000);
    }
  }

  async stop({ graceful = true } = {}) {
    // A failed quiesce is reported, and the stop still happens (the processes must not outlive the driver).
    if (graceful) await this.quiesce().catch((error) => console.error(`yue2-acceptance: quiesce before stop: ${error.message}`));
    await this.stopChild(this.worker);
    await this.stopChild(this.api);
  }

  async restart({ offline = this.offline, workerExtra = {}, worker = true } = {}) {
    await this.stop();
    this.offline = offline;
    this.workerExtra = workerExtra;
    const health = await this.startApi();
    if (worker) await this.startWorker();
    return health;
  }
}

class Sampler {
  constructor(platform, service, gpuId) {
    this.platform = platform;
    this.service = service;
    this.gpuId = gpuId ?? "0";
    this.samples = [];
    this.faults = [];
    this.timer = null;
  }

  async tick() {
    try {
      if (this.platform === "metal") {
        const pids = this.service.pids();
        if (!pids.length) return;
        const file = path.join(os.tmpdir(), `yue2-acceptance-footprint-${process.pid}.json`);
        await execFile("/usr/bin/footprint", ["--noCategories", "-j", file, ...pids.flatMap((pid) => ["-p", String(pid)])], { timeout: 20_000 });
        const bytes = footprintBytes(JSON.parse(await readFile(file, "utf8")), pids);
        await rm(file, { force: true });
        this.samples.push({ at: Date.now(), bytes });
      } else {
        const { stdout } = await execFile("nvidia-smi", ["--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", String(this.gpuId)], { timeout: 20_000, windowsHide: true });
        this.samples.push({ at: Date.now(), bytes: nvidiaUsedBytes(stdout) });
      }
    } catch (error) {
      this.faults.push(error.message);
    }
  }

  start(intervalMs = 2000) {
    this.samples = [];
    this.faults = [];
    let running = false;
    this.tick();
    this.timer = setInterval(async () => {
      if (running) return;
      running = true;
      try { await this.tick(); } finally { running = false; }
    }, intervalMs);
  }

  async stop() {
    clearInterval(this.timer);
    await this.tick();
    return peakOf(this.samples, this.faults, this.platform === "metal" ? "footprint phys_footprint (API + worker)" : `nvidia-smi memory.used (GPU ${this.gpuId})`);
  }
}

/** One case's evidence, built as the case runs. */
class Recorder {
  constructor(item, platform) {
    this.record = {
      schema: RECORD_SCHEMA,
      caseId: item.id,
      title: item.title,
      acceptance: item.acceptance,
      platform,
      memorySampled: false,
      status: null,
      reason: null,
      startedAt: nowIso(),
      finishedAt: null,
      durationSeconds: null,
      assertions: [],
      requests: [],
      jobs: [],
      notes: [],
      peakMemory: null,
    };
    this.started = Date.now();
  }

  check(name, ok, detail) {
    this.record.assertions.push({ name, ok: Boolean(ok), detail: detail ?? null });
    return Boolean(ok);
  }

  /** An assertion the case cannot continue without. */
  require(name, ok, detail) {
    if (!this.check(name, ok, detail)) throw new CaseFailure(`${name}: ${detail ?? "failed"}`);
  }

  note(text) {
    this.record.notes.push(text);
  }
}

class CaseFailure extends Error {}

/** Everything a case needs: the service, the API, the project and earlier cases' outputs. */
class Context {
  constructor(options, service, paths) {
    this.options = options;
    this.platform = options.platform;
    this.service = service;
    this.paths = paths;
    this.outputs = new Map();
    this.fixtures = null;
    this.project = null;
    this.seed = 2300200;
  }

  nextSeed() {
    this.seed += 1;
    return this.seed;
  }

  async call(rec, method, route, body, expect) {
    const response = await this.service.request(method, route, body);
    const entry = { method, path: route, body: body ?? null, status: response.status, response: summarizeResponse(response.body) };
    rec.record.requests.push(entry);
    if (expect !== undefined) {
      const wanted = Array.isArray(expect) ? expect : [expect];
      rec.require(`${method} ${route} → ${wanted.join("|")}`, wanted.includes(response.status), `HTTP ${response.status}: ${JSON.stringify(response.body).slice(0, 600)}`);
    }
    return response;
  }

  async submit(rec, body, expect = 201) {
    const response = await this.call(rec, "POST", `/api/v1/projects/${this.project.id}/yue2/jobs`, body, expect);
    return response.body;
  }

  async job(jobId) {
    const response = await this.service.request("GET", `/api/v1/jobs/${jobId}`);
    if (response.status !== 200) fail(`GET job ${jobId} → HTTP ${response.status}`);
    return response.body;
  }

  /**
   * Poll a job to a terminal status (or until `until(snapshot, stage)` is true), recording every
   * stage transition. `/api/v1/workers` is read alongside so the server's stale-worker sweep runs.
   */
  async follow(jobId, { until = null, timeoutMs = this.options.jobTimeoutMinutes * 60_000, pollMs = 250 } = {}) {
    const deadline = Date.now() + timeoutMs;
    const observations = [];
    let snapshot;
    let lastSweep = 0;
    while (Date.now() < deadline) {
      snapshot = await this.job(jobId);
      const stage = stageOf(snapshot);
      if (!observations.length || observations.at(-1).stage !== stage || observations.at(-1).message !== snapshot.message) {
        observations.push({ at: Date.now(), stage, status: snapshot.status, message: snapshot.message ?? null, progress: snapshot.progress ?? null });
      }
      if (until && until(snapshot, stage)) return { snapshot, observations, reached: true };
      if (["completed", "failed", "canceled", "interrupted"].includes(snapshot.status)) return { snapshot, observations, reached: !until };
      if (Date.now() - lastSweep > 5000) {
        lastSweep = Date.now();
        // Only a nudge for the server's lazy stale-worker sweep; a dead API fails the next job poll.
        await this.service.request("GET", "/api/v1/workers").catch(() => null);
      }
      await sleep(pollMs);
    }
    fail(`job ${jobId} did not reach the expected state in ${Math.round(timeoutMs / 60000)} min (last: ${snapshot?.status} ${snapshot?.message})`);
  }

  runDir(job) {
    const rel = job?.result?.yue2?.run?.dir ?? (job?.payload?.yue2?.runId ? `yue2/runs/${job.payload.yue2.runId}` : null);
    if (!rel) return null;
    return path.join(this.project.path, ...rel.split("/"));
  }

  /** Follow a job to its end and collect its evidence from the API and the run it published. */
  async evidence(rec, jobId, { expectStatus = "completed", followed = null } = {}) {
    const started = Date.now();
    const { snapshot, observations } = followed ?? await this.follow(jobId);
    const entry = {
      jobId,
      kind: snapshot.payload?.yue2?.kind ?? null,
      runId: snapshot.payload?.yue2?.runId ?? null,
      tier: snapshot.result?.yue2?.tier ?? snapshot.result?.yue2?.effectiveSettings?.tier ?? snapshot.payload?.yue2?.tier ?? null,
      status: snapshot.status,
      error: snapshot.error ?? null,
      message: snapshot.message ?? null,
      request: snapshot.payload?.yue2 ?? null,
      batch: snapshot.payload?.yue2?.batch ?? null,
      createdAt: snapshot.createdAt ?? null,
      startedAt: snapshot.startedAt ?? null,
      completedAt: snapshot.completedAt ?? null,
      durationSeconds: snapshot.startedAt && snapshot.completedAt ? (Date.parse(snapshot.completedAt) - Date.parse(snapshot.startedAt)) / 1000 : (Date.now() - started) / 1000,
      backend: snapshot.backend ?? null,
      stageTimings: { observed: stageTimeline(observations), observations },
      rendered: false,
    };
    rec.record.jobs.push(entry);
    rec.require(`job ${jobId} ${expectStatus}`, snapshot.status === expectStatus, `status ${snapshot.status}: ${snapshot.error ?? snapshot.message ?? ""}`);
    if (snapshot.status !== "completed") return { snapshot, entry };
    const y2 = snapshot.result?.yue2 ?? {};
    const runDir = this.runDir(snapshot);
    const runResult = JSON.parse(await readFile(path.join(runDir, "result.json"), "utf8"));
    entry.runDir = path.relative(this.paths.dataDir, runDir);
    entry.runIdentity = runResult.identity ?? null;
    entry.planIdentity = runResult.plan_identity ?? null;
    entry.truncated = runResult.truncated ?? null;
    entry.stageTimings.engine = runResult.timing ?? null;
    entry.stages = runResult.stages ?? null;
    entry.usagePolicy = y2.usagePolicy ?? null;
    entry.warnings = y2.warnings ?? [];
    entry.effectiveSettings = y2.effectiveSettings ?? null;
    entry.model = y2.model ?? null;
    entry.decoder = y2.decoder ?? null;
    entry.scoreVersionId = y2.scoreVersionId ?? null;
    const configFile = existsSync(path.join(runDir, "config.json")) ? "config.json" : "provenance.json";
    const configBody = JSON.parse(await readFile(path.join(runDir, configFile), "utf8"));
    const runConfig = configFile === "config.json" ? configBody : configBody.config ?? configBody;
    entry.device = runConfig.device ?? null;
    entry.dtype = { model: runConfig.model_dtype ?? null, vae: runConfig.vae_dtype ?? null };
    entry.engineConfig = { cot: runConfig.cot ?? null, weightTier: runConfig.weight_tier ?? null, quantization: runConfig.quantization ?? null, offloadAr: runConfig.offload_ar ?? null, queryTile: runConfig.query_tile ?? null, vaeDecode: runConfig.vae_decode ?? null, vaeCoreFrames: runConfig.vae_core_frames ?? null, decoderRelease: runConfig.decoder_release ?? null };
    const artifactCheck = await verifyRunArtifacts(runDir, runResult);
    entry.artifacts = artifactCheck;
    if (runResult.kind === "plan") {
      rec.check(`${jobId}: plan run on the production device`, deviceMatches(this.platform, entry.device), `device ${entry.device}`);
      rec.check(`${jobId}: plan artifacts verify`, artifactCheck.ok, artifactCheck.detail);
      rec.check(`${jobId}: plan truncation reported`, typeof runResult.truncated?.abc === "boolean", JSON.stringify(runResult.truncated));
      return { snapshot, entry, runDir, runResult, runConfig };
    }
    const audioPath = path.join(runDir, "audio.wav");
    const audioBytes = await readFile(audioPath);
    const runAudio = parseWav(audioBytes);
    const assetPath = snapshot.result?.assetWrites?.[0]?.mediaPath;
    let assetSha = null;
    if (assetPath && existsSync(path.join(this.project.path, assetPath))) assetSha = await fileSha256(path.join(this.project.path, assetPath));
    entry.rendered = true;
    entry.output = {
      runAudioSha256: sha256(audioBytes),
      runAudioBytes: audioBytes.length,
      assetId: snapshot.result?.assetIds?.[0] ?? snapshot.result?.assetWrites?.[0]?.assetId ?? null,
      assetMediaPath: assetPath ?? null,
      assetSha256: assetSha,
      sampleRate: runAudio.sampleRate,
      channels: runAudio.channels,
      durationSeconds: runAudio.durationSeconds,
      rms: runAudio.rms,
      clampedSamples: runResult.clamped_samples ?? null,
    };
    entry.latent = runResult.latent ?? null;
    entry.engineDecoder = runResult.decoder ?? null;
    for (const assertion of renderAssertions({ platform: this.platform, job: snapshot, runResult, runConfig, artifactCheck, runAudio })) {
      rec.check(`${jobId}: ${assertion.name}`, assertion.ok, assertion.detail);
    }
    return { snapshot, entry, runDir, runResult, runConfig, runAudio };
  }

  async jobIds() {
    const response = await this.service.request("GET", "/api/v1/jobs");
    if (response.status !== 200 || !Array.isArray(response.body)) fail(`GET /api/v1/jobs → HTTP ${response.status}`);
    return new Set(response.body.map((job) => job.id));
  }

  /**
   * One install request through the app, and every `model_download` job it queued (the chosen decoder
   * is queued as its own job, whose id the route does not return) followed to its end.
   */
  async install(rec, body) {
    const before = await this.jobIds();
    const response = await this.call(rec, "POST", `/api/v1/models/${MODEL_ID}/download`, body, 201);
    const all = await this.service.request("GET", "/api/v1/jobs");
    const queued = all.body.filter((job) => !before.has(job.id) && job.type === "model_download");
    rec.require(`install ${JSON.stringify(body)} queued its jobs`, queued.some((job) => job.id === response.body.id), queued.map((job) => job.id).join(", "));
    const jobs = [];
    for (const job of queued) {
      const { snapshot, observations } = await this.follow(job.id);
      const entry = {
        jobId: job.id,
        kind: "model_download",
        status: snapshot.status,
        repo: snapshot.payload?.repo ?? null,
        revision: snapshot.payload?.revision ?? null,
        variant: snapshot.payload?.variant ?? null,
        localDerivation: snapshot.payload?.localDerivation ?? null,
        error: snapshot.error ?? null,
        startedAt: snapshot.startedAt ?? null,
        completedAt: snapshot.completedAt ?? null,
        durationSeconds: snapshot.startedAt && snapshot.completedAt ? (Date.parse(snapshot.completedAt) - Date.parse(snapshot.startedAt)) / 1000 : null,
        stageTimings: { observed: stageTimeline(observations) },
        rendered: false,
      };
      rec.record.jobs.push(entry);
      jobs.push(entry);
    }
    return { primary: response.body.id, jobs };
  }

  async models(rec) {
    const response = await this.call(rec, "GET", "/api/v1/models", undefined, 200);
    const entry = response.body.find((model) => model.id === MODEL_ID);
    rec.require("yue2 is in the catalog", Boolean(entry), "GET /api/v1/models lists no yue2");
    return { all: response.body, entry };
  }

  songBody(overrides = {}) {
    return {
      kind: "create",
      style: this.fixtures.song.style,
      lyrics: this.fixtures.song.lyrics,
      seed: this.nextSeed(),
      semanticSampling: { maxTokens: SONG_SEMANTIC_MAX_TOKENS },
      ...overrides,
    };
  }
}

function summarizeResponse(body) {
  if (!body || typeof body !== "object") return body;
  const pick = (value) => value && typeof value === "object" ? { id: value.id, status: value.status, code: value.code } : value;
  if (Array.isArray(body)) return { items: body.length };
  const out = {};
  for (const key of ["id", "status", "code", "detail", "batchId", "modelId", "acknowledged", "termsSha256", "required", "dryRun", "installState", "variant"]) if (body[key] !== undefined) out[key] = body[key];
  if (Array.isArray(body.jobs)) out.jobs = body.jobs.map((job) => ({ id: job.id, status: job.status, runId: job.payload?.yue2?.runId ?? null, seed: job.payload?.yue2?.seed ?? null }));
  if (body.version) out.version = { id: body.version.id, parentVersionId: body.version.parentVersionId, origin: body.version.origin, scoreSha256: body.version.score?.sha256, requestSha256: body.version.requestSha256 };
  if (body.context !== undefined) out.context = body.context;
  if (body.payload?.yue2) out.payload = { yue2: pick(body.payload.yue2) };
  return out;
}

/** Re-hash every artifact the engine's manifest records — independently of the engine's verify_run. */
export async function verifyRunArtifacts(runDir, runResult) {
  const artifacts = runResult?.artifacts;
  if (!artifacts || typeof artifacts !== "object") return { ok: false, detail: "result.json has no artifact manifest", files: 0 };
  const required = runResult.kind === "plan" ? PLAN_RUN_FILES : SONG_RUN_FILES;
  const missing = required.filter((name) => !artifacts[name]);
  if (missing.length) return { ok: false, detail: `the manifest omits ${missing.join(", ")}`, files: Object.keys(artifacts).length };
  for (const [name, digest] of Object.entries(artifacts)) {
    const file = path.join(runDir, ...name.split("/"));
    let info;
    try { info = await stat(file); } catch { return { ok: false, detail: `${name} is missing`, files: Object.keys(artifacts).length }; }
    if (info.size !== digest.bytes) return { ok: false, detail: `${name} is ${info.size} bytes, recorded ${digest.bytes}`, files: Object.keys(artifacts).length };
    const actual = await fileSha256(file);
    if (actual !== digest.sha256) return { ok: false, detail: `${name} hashes to ${actual}, recorded ${digest.sha256}`, files: Object.keys(artifacts).length };
  }
  return { ok: true, detail: `${Object.keys(artifacts).length} artifacts re-hashed`, files: Object.keys(artifacts).length };
}

async function listRuns(project) {
  const dir = path.join(project.path, "yue2", "runs");
  try { return (await readdir(dir)).sort(); } catch (error) { if (error.code === "ENOENT") return []; throw error; }
}

/** Snapshot of every OTHER run (name → result.json sha256), to prove a cancel touched nothing else. */
async function otherRuns(project, runId) {
  const out = {};
  for (const name of await listRuns(project)) {
    if (runId && (name === runId || name.startsWith(`${runId}.`))) continue;
    const result = path.join(project.path, "yue2", "runs", name, "result.json");
    out[name] = existsSync(result) ? await fileSha256(result) : "(no result.json)";
  }
  return out;
}

// ---- Case implementations --------------------------------------------------------------------------

const V1_FAMILY = "yue";

const CASE_RUNNERS = {
  async "catalog-preflight"(ctx, rec) {
    const { entry } = await ctx.models(rec);
    const manifest = ctx.fixtures.manifestEntry;
    const primary = (rows) => rows.filter((row) => !row.coRequisite);
    for (const tier of ["bf16", "q8", "q4"]) {
      const row = manifest.downloads.find((download) => !download.coRequisite && download.variant === tier);
      rec.require(`catalog declares ${tier}`, Boolean(row), `builtin manifest ${tier} row`);
      const variant = entry.variants?.find((item) => item.variant === tier);
      rec.check(`API reports the ${tier} tier`, Boolean(variant), JSON.stringify(variant ?? null).slice(0, 300));
      if (tier !== "bf16") rec.check(`${tier} is a pinned local derivation`, /^[0-9a-f]{64}$/.test(row.localDerivation?.weightsSha256 ?? ""), JSON.stringify(row.localDerivation ?? null));
      rec.check(`${tier} pinned to a 40-hex revision`, /^[0-9a-f]{40}$/.test(row.revision ?? ""), row.revision);
    }
    for (const component of ["vae", "vae_legacy"]) {
      const row = manifest.downloads.find((download) => download.componentId === component);
      rec.check(`decoder ${component} pinned`, Boolean(row) && /^[0-9a-f]{40}$/.test(row.revision ?? ""), JSON.stringify(row ?? null).slice(0, 200));
    }
    rec.check("transcription components are conditional and blocked", (manifest.conditionalComponents ?? []).length > 0 && manifest.conditionalComponents.every((component) => component.blocked?.reason), JSON.stringify((manifest.conditionalComponents ?? []).map((component) => component.id ?? component.componentId)));
    rec.check("no transcription component is a generation download", primary(manifest.downloads).every((row) => !/SheetSage|MERT/i.test(row.repo)), "");
    rec.note(`initial install state: ${entry.installState}; variants ${JSON.stringify((entry.variants ?? []).map((variant) => ({ variant: variant.variant, installState: variant.installState })))}`);
    ctx.outputs.set("initialInstall", { installState: entry.installState, variants: (entry.variants ?? []).map((variant) => ({ variant: variant.variant, installState: variant.installState, installed: variant.installed })) });
  },

  async "isolation-v1-v2"(ctx, rec) {
    const { all, entry } = await ctx.models(rec);
    const v1 = all.filter((model) => model.family === V1_FAMILY);
    const v1Manifest = ctx.fixtures.manifest.models.filter((model) => model.family === V1_FAMILY);
    rec.require("the YuE1 entries are cataloged", v1.length > 0 && v1.length === v1Manifest.length, `API ${v1.length}, manifest ${v1Manifest.length}`);
    rec.check("V1 and V2 ids are distinct", v1.every((model) => model.id !== MODEL_ID) && entry.family !== V1_FAMILY, JSON.stringify(v1.map((model) => model.id)));
    const repos = (model) => new Set((model.downloads ?? []).map((row) => row.repo));
    const v2Repos = repos(entry);
    rec.check("V1 and V2 share no download repo (distinct cache namespaces)", v1.every((model) => [...repos(model)].every((repo) => !v2Repos.has(repo))), JSON.stringify([...v2Repos]));
    rec.check("V2 is experimental and noncommercial", entry.experimental === true && entry.nonCommercial === true && entry.requiresLicenseAcknowledgment === true, JSON.stringify({ experimental: entry.experimental, nonCommercial: entry.nonCommercial }));
    rec.check("V1 carries no experimental/noncommercial/licence-gate flag", v1.every((model) => !model.experimental && !model.nonCommercial && !model.requiresLicenseAcknowledgment), "");
    for (const model of v1Manifest) {
      const live = v1.find((item) => item.id === model.id);
      const same = live && JSON.stringify((live.downloads ?? []).map((row) => [row.repo, row.revision, row.variant ?? null])) === JSON.stringify(model.downloads.map((row) => [row.repo, row.revision, row.variant ?? null]));
      rec.check(`V1 ${model.id} rows unchanged from the checked-in catalog`, same, "");
    }
    const alternatives = entry.commercialUse?.alternatives ?? [];
    rec.check("V2's commercial pointer names the eligible V1 ids", alternatives.length > 0 && alternatives.every((id) => v1.some((model) => model.id === id)), JSON.stringify(alternatives));
    const refused = await ctx.submit(rec, { ...ctx.songBody({ kind: "plan", semanticSampling: undefined }), commercialUse: true, licenseAcknowledged: true }, 403);
    rec.check("a commercial-use V2 job is refused", refused?.code === "commercial_use_refused", JSON.stringify(refused).slice(0, 400));
    rec.check("the refusal points at YuE1", JSON.stringify(refused?.context ?? {}).includes("yue_"), JSON.stringify(refused?.context ?? {}).slice(0, 300));
    const generic = await ctx.call(rec, "POST", "/api/v1/audio/jobs", { projectId: ctx.project.id, model: MODEL_ID, prompt: "pop", lyrics: "[verse]\nla" }, [400, 403, 422]);
    rec.check("the generic audio route refuses V2", generic.body?.code === "yue2_song_route_required", JSON.stringify(generic.body).slice(0, 400));
    for (const model of v1.slice(0, 1)) {
      const ack = await ctx.call(rec, "GET", `/api/v1/models/${model.id}/license-acknowledgment`, undefined, 200);
      rec.check(`V1 ${model.id} needs no V2 licence acknowledgment`, ack.body?.required === false, JSON.stringify(ack.body));
    }
  },

  async "licence-gate"(ctx, rec) {
    await ctx.call(rec, "DELETE", `/api/v1/models/${MODEL_ID}/license-acknowledgment`, undefined, 200);
    const before = await ctx.call(rec, "GET", `/api/v1/models/${MODEL_ID}/license-acknowledgment`, undefined, 200);
    rec.require("the acknowledgment is withdrawn", before.body?.required === true && before.body?.acknowledged === false, JSON.stringify(before.body));
    const planBody = { kind: "plan", style: ctx.fixtures.song.style, lyrics: ctx.fixtures.song.lyrics, seed: ctx.nextSeed() };
    const refused = await ctx.submit(rec, planBody, 403);
    rec.check("refused without the acknowledgment", refused?.code === "license_acknowledgment_required", JSON.stringify(refused).slice(0, 300));
    const download = await ctx.call(rec, "POST", `/api/v1/models/${MODEL_ID}/download`, { variant: "bf16" }, 403);
    rec.check("install refused without the acknowledgment", download.body?.code === "license_acknowledgment_required", JSON.stringify(download.body).slice(0, 300));
    const allowed = await ctx.submit(rec, { ...planBody, licenseAcknowledged: true }, 201);
    const after = await ctx.call(rec, "GET", `/api/v1/models/${MODEL_ID}/license-acknowledgment`, undefined, 200);
    rec.check("the acknowledgment is recorded for the current terms", after.body?.acknowledged === true && after.body?.termsSha256 === before.body?.termsSha256, JSON.stringify(after.body).slice(0, 300));
    const jobId = allowed.jobs?.[0]?.id;
    rec.require("the acknowledged job is queued", Boolean(jobId), JSON.stringify(allowed).slice(0, 300));
    rec.check("the job carries the usage policy it was granted", allowed.jobs[0].payload?.usagePolicy?.licenseAcknowledgment?.termsSha256 === after.body?.termsSha256 && allowed.jobs[0].payload?.usagePolicy?.nonCommercial === true, JSON.stringify(allowed.jobs[0].payload?.usagePolicy ?? null).slice(0, 400));
    if (ctx.options.dryRun) {
      await ctx.call(rec, "POST", `/api/v1/jobs/${jobId}/cancel`, undefined, 200);
      const canceled = await ctx.job(jobId);
      rec.check("dry run: the queued job is cancelled before any worker", canceled.status === "canceled", canceled.status);
      rec.note("dry run: execution-time eligibility was not exercised (no worker)");
      return;
    }
    const { entry } = await ctx.evidence(rec, jobId);
    rec.check("the worker's execution-time eligibility granted it", entry.usagePolicy?.licenseAcknowledgment?.termsSha256 === after.body?.termsSha256, JSON.stringify(entry.usagePolicy ?? null).slice(0, 300));
  },

  async "transcription-blocked"(ctx, rec) {
    const transcribe = await ctx.submit(rec, { kind: "transcribe", sourceAudioAssetId: "asset_acceptance_source", licenseAcknowledged: true }, 403);
    rec.check("transcribe refused with component_blocked", transcribe?.code === "component_blocked", JSON.stringify(transcribe).slice(0, 400));
    rec.check("the refusal names the blocked components and their unblock condition", (transcribe?.context?.blocked ?? []).length > 0 && transcribe.context.blocked.every((item) => item.reason && item.unblock), JSON.stringify(transcribe?.context ?? null).slice(0, 400));
    const cover = await ctx.submit(rec, { kind: "cover", lyrics: ctx.fixtures.song.lyrics, cover: { mode: "full", sourceAudioAssetId: "asset_acceptance_source" }, licenseAcknowledged: true }, 403);
    rec.check("a cover from a source recording refused with component_blocked", cover?.code === "component_blocked", JSON.stringify(cover).slice(0, 400));
    return { status: "blocked", reason: `recording transcription is blocked as designed: ${(transcribe?.context?.blocked ?? []).map((item) => `${item.componentId}: ${item.reason}`).join("; ")}` };
  },

  async "install-cold"(ctx, rec) {
    const before = (await ctx.models(rec)).entry;
    rec.note(`before install: ${before.installState}; ${JSON.stringify((before.variants ?? []).map((variant) => [variant.variant, variant.installState]))}; hub ${ctx.paths.hfHub}`);
    const installed = [];
    for (const [variant, decoder] of [["bf16", "standard"], ["bf16", "legacy"], ["q8", "standard"], ["q4", "standard"]]) {
      const { jobs } = await ctx.install(rec, { variant, licenseAcknowledged: true, choices: { decoder } });
      for (const job of jobs) {
        rec.check(`install ${variant}/${decoder}: ${job.repo}${job.localDerivation ? ` → derive ${job.localDerivation.variant}` : ""} completed`, job.status === "completed", `${job.status}: ${job.error ?? ""}`);
        installed.push(job);
      }
    }
    const repos = new Set(installed.map((job) => job.repo));
    rec.check("the model and both decoders were installed through the app", ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae", "m-a-p/YuE2-Vae-legacy"].every((repo) => repos.has(repo)), JSON.stringify([...repos]));
    rec.check("q8 and q4 were derived by install jobs", ["q8", "q4"].every((tier) => installed.some((job) => job.localDerivation?.variant === tier && job.status === "completed")), "");
    const after = (await ctx.models(rec)).entry;
    rec.check("yue2 reads installed", after.installState === "installed", after.installState);
    for (const tier of ["bf16", "q8", "q4"]) {
      const variant = after.variants?.find((item) => item.variant === tier);
      rec.check(`${tier} installed`, variant?.installed === true && variant?.installState === "installed", JSON.stringify(variant ?? null).slice(0, 300));
    }
    for (const tier of ["q8", "q4"]) {
      const row = ctx.fixtures.manifestEntry.downloads.find((download) => !download.coRequisite && download.variant === tier);
      const derived = path.join(ctx.paths.dataDir, "models", "derived", MODEL_ID, tier, row.localDerivation.conversion, row.localDerivation.weightsFile);
      const digest = existsSync(derived) ? await fileSha256(derived) : null;
      rec.check(`derived ${tier} weights hash to the pinned sha256`, digest === row.localDerivation.weightsSha256, `${digest} vs ${row.localDerivation.weightsSha256}`);
    }
  },

  async "create-cot-full"(ctx, rec) {
    const body = ctx.songBody({ planning: "full" });
    const response = await ctx.submit(rec, body);
    const { entry } = await ctx.evidence(rec, response.jobs[0].id);
    rec.check("effective tier is the default bf16", entry.effectiveSettings?.tier === "bf16", entry.effectiveSettings?.tier);
    rec.check("engine ran cot=full", entry.engineConfig.cot === "full", entry.engineConfig.cot);
    rec.check("EN lyrics rendered", /[A-Za-z]/.test(entry.effectiveSettings?.lyrics ?? ""), "");
    ctx.outputs.set("source", { jobId: response.jobs[0].id, entry });
  },

  async "create-cot-melody"(ctx, rec) {
    const response = await ctx.submit(rec, ctx.songBody({ planning: "melody", tier: "q8" }));
    const { entry } = await ctx.evidence(rec, response.jobs[0].id);
    rec.check("engine ran cot=melody", entry.engineConfig.cot === "melody", entry.engineConfig.cot);
    rec.check("q8 tier", entry.engineConfig.weightTier === "q8" || entry.effectiveSettings?.tier === "q8", JSON.stringify(entry.engineConfig));
  },

  async "create-cot-off"(ctx, rec) {
    const response = await ctx.submit(rec, ctx.songBody({ planning: "off", tier: "q4" }));
    const { entry, runDir } = await ctx.evidence(rec, response.jobs[0].id);
    rec.check("engine ran cot=off", entry.engineConfig.cot === "off", entry.engineConfig.cot);
    rec.check("q4 tier", entry.engineConfig.weightTier === "q4" || entry.effectiveSettings?.tier === "q4", JSON.stringify(entry.engineConfig));
    rec.check("an unplanned run carries no score", !existsSync(path.join(runDir, "score.abc")), "");
  },

  async "supplied-abc-full"(ctx, rec) {
    const response = await ctx.submit(rec, ctx.songBody({ planning: "full", score: ctx.fixtures.scoreAbc }));
    const { runDir, entry } = await ctx.evidence(rec, response.jobs[0].id);
    const planned = await readFile(path.join(runDir, "score.abc"), "utf8");
    rec.check("the run planned from the supplied full ABC", planned.trim() === ctx.fixtures.scoreAbc.trim(), `score.abc sha256 ${sha256(planned)} vs supplied ${sha256(ctx.fixtures.scoreAbc)}`);
    rec.check("engine ran cot=full", entry.engineConfig.cot === "full", entry.engineConfig.cot);
  },

  async "supplied-abc-melody"(ctx, rec) {
    const response = await ctx.submit(rec, ctx.songBody({ planning: "melody", score: ctx.fixtures.melodyAbc }));
    const { runDir, entry } = await ctx.evidence(rec, response.jobs[0].id);
    const planned = await readFile(path.join(runDir, "score.abc"), "utf8");
    rec.check("the run planned from the supplied melody ABC", planned.trim() === ctx.fixtures.melodyAbc.trim(), `score.abc sha256 ${sha256(planned)} vs supplied ${sha256(ctx.fixtures.melodyAbc)}`);
    rec.check("engine ran cot=melody", entry.engineConfig.cot === "melody", entry.engineConfig.cot);
  },

  async "lyrics-en-zh"(ctx, rec) {
    const source = ctx.outputs.get("source");
    rec.check("EN render passed (create-cot-full)", source?.entry?.rendered === true, source?.jobId);
    rec.note(`EN evidence: create-cot-full job ${source?.jobId}, output ${source?.entry?.output?.runAudioSha256}`);
    const response = await ctx.submit(rec, ctx.songBody({ planning: "full", style: ZH_STYLE, lyrics: ZH_LYRICS }));
    const { entry } = await ctx.evidence(rec, response.jobs[0].id);
    rec.check("ZH lyrics rendered", /[\u4e00-\u9fff]/.test(entry.effectiveSettings?.lyrics ?? ""), "");
  },

  async "plan-restore"(ctx, rec) {
    const plan = await ctx.submit(rec, { kind: "plan", style: ctx.fixtures.song.style, lyrics: ctx.fixtures.song.lyrics, seed: ctx.nextSeed() });
    const planned = await ctx.evidence(rec, plan.jobs[0].id);
    rec.require("the plan-only run published a plan", planned.runResult?.kind === "plan", planned.runResult?.kind);
    const restored = await ctx.submit(rec, { kind: "fromPlan", planJobId: plan.jobs[0].id, semanticSampling: { maxTokens: SONG_SEMANTIC_MAX_TOKENS } });
    const song = await ctx.evidence(rec, restored.jobs[0].id);
    rec.check("the restored run keeps the plan identity", song.runResult?.plan_identity === planned.runResult?.plan_identity, `${song.runResult?.plan_identity} vs ${planned.runResult?.plan_identity}`);
    for (const file of ["abc_tokens.npy", "prefix.npy", "plan.json"]) {
      const [a, b] = await Promise.all([fileSha256(path.join(planned.runDir, file)), fileSha256(path.join(song.runDir, file))]);
      rec.check(`restored ${file} is byte-identical`, a === b, `${a} vs ${b}`);
    }
    if (existsSync(path.join(planned.runDir, "score.abc"))) {
      const [a, b] = await Promise.all([fileSha256(path.join(planned.runDir, "score.abc")), fileSha256(path.join(song.runDir, "score.abc"))]);
      rec.check("restored score.abc is byte-identical", a === b, `${a} vs ${b}`);
    }
  },

  async "cover-melody"(ctx, rec) { await coverCase(ctx, rec, "melody"); },
  async "cover-full"(ctx, rec) { await coverCase(ctx, rec, "full"); },

  async "score-edits"(ctx, rec) {
    const root = await createVersion(ctx, rec, ctx.fixtures.scoreAbc, "full");
    const provenance = { actor: "agent", agentName: "yue2-acceptance", channel: "api" };
    const edit = async (name, operation, brief) => {
      const response = await ctx.call(rec, "POST", `/api/v1/projects/${ctx.project.id}/yue2/score-versions/${root.id}/edits`, { operation, brief, provenance, dryRun: false }, 201);
      const version = response.body.version;
      rec.check(`${name}: a new version derived from the source`, version?.parentVersionId === root.id && version?.origin === "edit" && version.id !== root.id, JSON.stringify({ id: version?.id, parent: version?.parentVersionId }));
      rec.check(`${name}: invariant report matches its contract`, version?.edit?.invariants?.match === true && (version.edit.invariants.violations ?? []).length === 0, JSON.stringify(version?.edit?.invariants?.violations ?? null));
      return { name, version };
    };
    const edits = [];
    const harmony = await edit("harmony-only", { op: "replace_score", abc: ctx.fixtures.jazzAbc, allow: { harmony: true } }, "Reharmonize with seventh chords; keep the melody, lyrics, form and tempo.");
    const checks = Object.fromEntries((harmony.version.edit?.invariants?.checks ?? []).map((item) => [item.name, item.status]));
    for (const name of ["notes:Vocal", "notes:Ins", "barGrid:Vocal", "barGrid:Ins", "sections", "tempo", "lyrics", "style"]) {
      rec.check(`harmony-only edit preserves ${name}`, checks[name] === "unchanged", checks[name]);
    }
    rec.check("harmony-only edit changed the harmony as declared", checks.harmony === "changedAsDeclared" && (harmony.version.edit?.invariants?.harmonyChanges ?? []).length > 0, checks.harmony);
    edits.push(harmony);
    edits.push(await edit("lyric", { op: "set_lyrics", lyrics: EDIT_LYRICS }, "Rewrite the lyrics; keep the music."));
    edits.push(await edit("style", { op: "set_style", style: EDIT_STYLE }, "Restyle as a slow soul ballad."));
    edits.push(await edit("tempo", { op: "set_tempo", bpm: 100 }, "Lift the tempo to 100 BPM."));
    edits.push(await edit("form", { op: "arrange_sections", sectionOrder: [0, 1, 1], lyrics: FORM_LYRICS }, "Repeat the chorus at the end."));
    const renders = [];
    for (const target of [{ name: "source", version: root }, ...edits]) {
      const response = await ctx.submit(rec, { kind: "renderVersion", versionId: target.version.id, semanticSampling: { maxTokens: SONG_SEMANTIC_MAX_TOKENS } });
      const { entry } = await ctx.evidence(rec, response.jobs[0].id);
      renders.push({ name: target.name, versionId: target.version.id, jobId: entry.jobId, assetId: entry.output?.assetId });
    }
    const listed = await ctx.call(rec, "GET", `/api/v1/projects/${ctx.project.id}/yue2/score-versions`, undefined, 200);
    const ids = new Set((listed.body?.items ?? []).map((item) => item.id));
    rec.check("every source and derivative version is retained", [root, ...edits.map((item) => item.version)].every((version) => ids.has(version.id)), JSON.stringify([...ids]));
    for (const render of renders) {
      const response = await ctx.call(rec, "GET", `/api/v1/projects/${ctx.project.id}/yue2/score-versions/${render.versionId}/renders`, undefined, 200);
      const record = (Array.isArray(response.body) ? response.body : []).find((item) => item.jobId === render.jobId);
      rec.check(`${render.name}: the render is recorded against its version`, record?.status === "completed" && Boolean(record?.audioAssetId) && record?.scoreSha256 === (render.name === "source" ? root : edits.find((item) => item.name === render.name).version).score?.sha256, JSON.stringify(record ?? null).slice(0, 300));
      render.recordId = record?.id ?? null;
    }
    const comparison = await ctx.call(rec, "POST", `/api/v1/projects/${ctx.project.id}/yue2/comparisons`, { versionA: root.id, versionB: harmony.version.id, renderA: renders[0].recordId, renderB: renders[1].recordId, provenance, notes: "acceptance: source vs harmony-only edit" }, [200, 201]);
    rec.check("a source-vs-derivative listening comparison is recorded", Boolean(comparison.body?.id), JSON.stringify(comparison.body).slice(0, 300));
  },

  async "decode-standard-legacy"(ctx, rec) {
    const source = ctx.outputs.get("source");
    const sourceLatent = source.entry.latent?.sha256;
    rec.require("the source run recorded its latent identity", /^[0-9a-f]{64}$/.test(sourceLatent ?? ""), JSON.stringify(source.entry.latent ?? null));
    const decoded = {};
    for (const decoder of ["standard", "legacy"]) {
      const response = await ctx.submit(rec, { kind: "decode", sourceJobId: source.jobId, decoder });
      const { entry, runDir } = await ctx.evidence(rec, response.jobs[0].id);
      const origin = JSON.parse(await readFile(path.join(runDir, "source_generation.json"), "utf8"));
      rec.check(`${decoder}: decoded the source's latent`, entry.latent?.sha256 === sourceLatent && origin.source_latent_sha256 === await fileSha256(path.join(ctx.runDirOf(source), "latent.npy")), `${entry.latent?.sha256} vs ${sourceLatent}`);
      rec.check(`${decoder}: source run identity recorded`, origin.identity === source.entry.runIdentity, `${origin.identity} vs ${source.entry.runIdentity}`);
      rec.check(`${decoder}: decoder attributed`, entry.engineConfig.decoderRelease && entry.decoder?.id, JSON.stringify({ release: entry.engineConfig.decoderRelease, decoder: entry.decoder }));
      decoded[decoder] = entry;
    }
    rec.check("the two decoders are distinct identities", decoded.standard.decoder?.id !== decoded.legacy.decoder?.id && decoded.standard.engineConfig.decoderRelease !== decoded.legacy.engineConfig.decoderRelease, JSON.stringify([decoded.standard.decoder, decoded.legacy.decoder]));
    rec.check("both decodes share ONE latent identity", decoded.standard.latent?.sha256 === decoded.legacy.latent?.sha256, "");
  },

  async "artifact-corrupt"(ctx, rec) {
    const source = ctx.outputs.get("source");
    const latent = path.join(ctx.runDirOf(source), "latent.npy");
    const backup = `${latent}.acceptance-backup`;
    const original = await fileSha256(latent);
    await copyFile(latent, backup);
    try {
      const handle = await open(latent, "r+");
      const probe = Buffer.alloc(1);
      const at = (await handle.stat()).size - 1;
      await handle.read(probe, 0, 1, at);
      probe[0] ^= 0xff;
      await handle.write(probe, 0, 1, at);
      await handle.close();
      rec.note(`flipped the last byte of ${path.relative(ctx.paths.dataDir, latent)}`);
      const refused = await ctx.submit(rec, { kind: "decode", sourceJobId: source.jobId });
      const { snapshot } = await ctx.evidence(rec, refused.jobs[0].id, { expectStatus: "failed" });
      rec.check("the corrupt source is refused by identity verification", /does not verify|hashes to|recorded/i.test(snapshot.error ?? ""), snapshot.error);
    } finally {
      await copyFile(backup, latent);
      await rm(backup, { force: true });
    }
    rec.require("the recorded bytes are restored", await fileSha256(latent) === original, "");
    const repaired = await ctx.submit(rec, { kind: "decode", sourceJobId: source.jobId });
    await ctx.evidence(rec, repaired.jobs[0].id);
  },

  async "artifact-missing"(ctx, rec) {
    const source = ctx.outputs.get("source");
    const dir = ctx.runDirOf(source);
    const moved = `${dir}.acceptance-moved`;
    await rename(dir, moved);
    try {
      const refused = await ctx.submit(rec, { kind: "decode", sourceJobId: source.jobId });
      const { snapshot } = await ctx.evidence(rec, refused.jobs[0].id, { expectStatus: "failed" });
      rec.check("the missing source run is refused", /missing/i.test(snapshot.error ?? ""), snapshot.error);
    } finally {
      await rename(moved, dir);
    }
    const recovered = await ctx.submit(rec, { kind: "decode", sourceJobId: source.jobId });
    await ctx.evidence(rec, recovered.jobs[0].id);
  },

  async "cancel-ar"(ctx, rec) {
    await cancelCase(ctx, rec, ctx.songBody({ planning: "off", tier: "q4" }), "semantic");
  },
  async "cancel-nar"(ctx, rec) {
    // Many midpoint ODE steps keep the acoustic stage longer than the cancel's heartbeat latency.
    await cancelCase(ctx, rec, ctx.songBody({ planning: "off", tier: "q4", steps: 400 }), "acoustic");
  },
  async "cancel-decode"(ctx, rec) {
    const source = ctx.outputs.get("source");
    // The smallest decode tile makes the decode long enough to cancel inside it.
    await cancelCase(ctx, rec, { kind: "decode", sourceJobId: source.jobId, memory: { tileVaeDecode: true, decodeTileEdge: 1 } }, "decode");
  },

  async "worker-kill-resume"(ctx, rec) {
    const submitted = await ctx.submit(rec, ctx.songBody({ planning: "full", tier: "q4" }));
    const jobId = submitted.jobs[0].id;
    const runId = submitted.jobs[0].payload.yue2.runId;
    const partial = path.join(ctx.project.path, "yue2", "runs", `${runId}.partial`);
    const reached = await ctx.follow(jobId, { until: (_snapshot, stage) => stage === "semantic" || stage === "acoustic" });
    rec.require("the job reached generation past its plan checkpoint", reached.reached, reached.snapshot.status);
    // Let the plan checkpoint land on disk before the kill.
    const planRecord = path.join(partial, "stages", "plan.json");
    for (let attempt = 0; attempt < 120 && !existsSync(planRecord); attempt += 1) await sleep(500);
    rec.require("the plan checkpoint was recorded", existsSync(planRecord), planRecord);
    const checkpoint = JSON.parse(await readFile(planRecord, "utf8"));
    const killedPid = ctx.service.worker.pid;
    await ctx.service.stopChild(ctx.service.worker, { force: true, graceMs: 30_000 });
    rec.note(`SIGKILLed worker pid ${killedPid} during ${reached.observations.at(-1)?.message}`);
    await ctx.service.startWorker();
    const interrupted = await ctx.follow(jobId, { timeoutMs: 5 * 60_000 });
    rec.check("the orphaned job is interrupted, not left running", ["interrupted", "failed"].includes(interrupted.snapshot.status), `${interrupted.snapshot.status}: ${interrupted.snapshot.message}`);
    rec.check("its working directory survived the kill", existsSync(partial), partial);
    const retry = await ctx.call(rec, "POST", `/api/v1/jobs/${jobId}/retry`, {}, 201);
    rec.check("the retry keeps the run id", retry.body?.payload?.yue2?.runId === runId, `${retry.body?.payload?.yue2?.runId} vs ${runId}`);
    const { runResult } = await ctx.evidence(rec, retry.body.id);
    const planStage = runResult.stages?.plan;
    rec.check("the resumed run reused the plan checkpoint", planStage?.reused === true, JSON.stringify(planStage ?? null));
    rec.check("with the exact recorded identity", planStage?.identity === checkpoint.identity, `${planStage?.identity} vs ${checkpoint.identity}`);
    rec.check("the run published under its own id", !existsSync(partial) && existsSync(path.join(ctx.project.path, "yue2", "runs", runId, "result.json")), "");
  },

  async "admission"(ctx, rec) {
    const capEnv = PLATFORMS[ctx.platform].capEnv;
    await ctx.service.restart({ workerExtra: { [capEnv]: String(ADMISSION_CAP_GB) } });
    rec.note(`worker restarted with ${capEnv}=${ADMISSION_CAP_GB}`);
    try {
      const over = await ctx.submit(rec, {
        kind: "create", style: ctx.fixtures.song.style, lyrics: ctx.fixtures.song.lyrics, seed: ctx.nextSeed(),
        tier: "bf16", precision: "fp32", planning: "off", cfgScale: 1.5,
        semanticSampling: { minTokens: 16000, maxTokens: 16000 },
        memory: { stageResidency: false, chunkAttention: true, attentionChunkSize: 4294967295, tileVaeDecode: true, decodeTileEdge: 1024 },
      });
      const refused = await ctx.follow(over.jobs[0].id);
      const { snapshot } = await ctx.evidence(rec, over.jobs[0].id, { expectStatus: "failed", followed: refused });
      rec.check("refused by admission with the shortfall", /GB short/.test(snapshot.error ?? ""), snapshot.error);
      rec.check("refused before load (never reported loading)", !refused.observations.some((observation) => observation.stage === "load" || ["plan", "semantic", "acoustic", "decode"].includes(observation.stage)), JSON.stringify(refused.observations.map((observation) => observation.stage)));
      const fits = await ctx.submit(rec, ctx.songBody({ planning: "off", tier: "q4", semanticSampling: { maxTokens: 20 * SEMANTIC_TOKENS_PER_SECOND } }));
      const { entry } = await ctx.evidence(rec, fits.jobs[0].id);
      rec.note(`admitted controls (engine config): ${JSON.stringify({ offloadAr: entry.engineConfig.offloadAr, queryTile: entry.engineConfig.queryTile, vaeCoreFrames: entry.engineConfig.vaeCoreFrames })}`);
    } finally {
      await ctx.service.restart({ workerExtra: {} });
    }
  },

  async "batch-serial"(ctx, rec) {
    const seed = ctx.nextSeed();
    const response = await ctx.submit(rec, ctx.songBody({ planning: "off", tier: "q4", seed, count: 2, semanticSampling: { maxTokens: 20 * SEMANTIC_TOKENS_PER_SECOND } }));
    rec.require("two takes queued as one batch", response.jobs?.length === 2 && Boolean(response.batchId), JSON.stringify(response).slice(0, 300));
    const takes = [];
    for (const job of response.jobs) takes.push(await ctx.evidence(rec, job.id));
    const [a, b] = takes.map((take) => take.snapshot);
    rec.check("seeds are seed and seed+1", a.payload.yue2.seed === seed && b.payload.yue2.seed === seed + 1, `${a.payload.yue2.seed}, ${b.payload.yue2.seed}`);
    rec.check("each take has its own run", a.payload.yue2.runId !== b.payload.yue2.runId, "");
    rec.check("the takes ran serially on the admitted GPU", Date.parse(b.startedAt) >= Date.parse(a.completedAt), `take 1 ${a.startedAt}→${a.completedAt}, take 2 ${b.startedAt}→${b.completedAt}`);
  },

  async "offline-restart"(ctx, rec) {
    const source = ctx.outputs.get("source");
    try {
      const health = await ctx.service.restart({ offline: true });
      rec.note(`restarted offline: HF_HUB_OFFLINE=1, SCENEWORKS_HUGGINGFACE_BASE_URL unreachable; interrupted on startup ${health.interruptedJobsOnStartup}`);
      const { entry } = await ctx.models(rec);
      rec.check("offline: yue2 reads installed", entry.installState === "installed", entry.installState);
      for (const tier of ["bf16", "q8", "q4"]) rec.check(`offline: ${tier} installed`, entry.variants?.find((variant) => variant.variant === tier)?.installed === true, "");
      const decode = await ctx.submit(rec, { kind: "decode", sourceJobId: source.jobId });
      await ctx.evidence(rec, decode.jobs[0].id);
      const render = await ctx.submit(rec, ctx.songBody({ planning: "off", tier: "q4", semanticSampling: { maxTokens: 20 * SEMANTIC_TOKENS_PER_SECOND } }));
      await ctx.evidence(rec, render.jobs[0].id);
      await ctx.call(rec, "DELETE", `/api/v1/models/${MODEL_ID}/variants/q4`, undefined, 200);
      const gone = (await ctx.models(rec)).entry.variants?.find((variant) => variant.variant === "q4");
      rec.check("q4 removed reads not installed", gone?.installed === false, JSON.stringify(gone ?? null).slice(0, 200));
      const offlineInstall = await ctx.install(rec, { variant: "q4", licenseAcknowledged: true });
      const primary = offlineInstall.jobs.find((job) => job.jobId === offlineInstall.primary);
      rec.check("offline: the q4 reinstall fails visibly (the install flow needs the hub)", primary?.status === "failed", `${primary?.status}: ${primary?.error}`);
    } finally {
      await ctx.service.restart({ offline: false });
    }
    const online = await ctx.install(rec, { variant: "q4", licenseAcknowledged: true });
    for (const job of online.jobs) rec.check(`online: ${job.repo}${job.localDerivation ? " (derive q4)" : ""} recovers`, job.status === "completed", `${job.status}: ${job.error ?? ""}`);
    const back = (await ctx.models(rec)).entry.variants?.find((variant) => variant.variant === "q4");
    rec.check("q4 installed again", back?.installed === true, JSON.stringify(back ?? null).slice(0, 200));
  },
};

/** The ids that have a runner — every case must (the test suite holds it to that). */
export function caseRunnerIds() {
  return Object.keys(CASE_RUNNERS);
}

Context.prototype.runDirOf = function runDirOf(output) {
  return path.join(this.paths.dataDir, output.entry.runDir);
};

async function createVersion(ctx, rec, abc, cot) {
  const { id: _drop, ...song } = ctx.fixtures.song;
  const response = await ctx.call(rec, "POST", `/api/v1/projects/${ctx.project.id}/yue2/score-versions`, {
    abc,
    request: { ...song, cot },
    origin: "import",
    provenance: { actor: "agent", agentName: "yue2-acceptance", channel: "api", source: { kind: "external" } },
  }, 201);
  rec.require("the reviewed score version is stored", Boolean(response.body?.id), JSON.stringify(response.body).slice(0, 300));
  return response.body;
}

async function coverCase(ctx, rec, mode) {
  const version = await createVersion(ctx, rec, ctx.fixtures.scoreAbc, "full");
  const response = await ctx.submit(rec, { kind: "cover", style: ctx.fixtures.song.style, lyrics: ctx.fixtures.song.lyrics, cover: { mode, versionId: version.id }, semanticSampling: { maxTokens: SONG_SEMANTIC_MAX_TOKENS } });
  const { entry, snapshot } = await ctx.evidence(rec, response.jobs[0].id);
  const coverSource = snapshot.result?.yue2?.sources?.coverVersion;
  rec.check("the cover is bound to the reviewed score version", coverSource?.id === version.id && coverSource?.scoreSha256 === version.score?.sha256, JSON.stringify(coverSource ?? null));
  rec.check(`cover mode ${mode}`, entry.effectiveSettings?.cover?.mode === mode, JSON.stringify(entry.effectiveSettings?.cover ?? null));
  rec.note(`cover warnings: ${JSON.stringify(entry.warnings)}`);
}

async function cancelCase(ctx, rec, body, target) {
  const others = await otherRuns(ctx.project, null);
  const submitted = await ctx.submit(rec, body);
  const jobId = submitted.jobs[0].id;
  const runId = submitted.jobs[0].payload.yue2.runId;
  const runsDir = path.join(ctx.project.path, "yue2", "runs");
  const reached = await ctx.follow(jobId, { until: (_snapshot, stage) => stage === target, pollMs: 100 });
  rec.require(`the job reached ${target}`, reached.reached, `${reached.snapshot.status}: ${reached.snapshot.message}`);
  await ctx.call(rec, "POST", `/api/v1/jobs/${jobId}/cancel`, undefined, 200);
  const done = await ctx.follow(jobId);
  const { snapshot } = await ctx.evidence(rec, jobId, { expectStatus: "canceled", followed: { snapshot: done.snapshot, observations: [...reached.observations, ...done.observations] } });
  const lastWorking = [...reached.observations, ...done.observations].filter((observation) => !["canceled", "cancel-requested"].includes(observation.stage)).at(-1);
  rec.check(`the cancel landed during ${target}`, lastWorking?.stage === target, `last working stage ${lastWorking?.stage} (${lastWorking?.message})`);
  rec.check("canceled by the user, cleanly", snapshot.message === "YuE2 job canceled by user.", snapshot.message);
  rec.check("the run's .partial working directory is removed", !existsSync(path.join(runsDir, `${runId}.partial`)), "");
  rec.check("nothing was published for the canceled run", !existsSync(path.join(runsDir, runId)), "");
  const after = await otherRuns(ctx.project, runId);
  rec.check("every other run is untouched", JSON.stringify(after) === JSON.stringify(others), `${Object.keys(others).length} runs compared`);
  const workers = await ctx.service.request("GET", "/api/v1/workers");
  const mine = workers.body?.find?.((worker) => worker.id === ctx.service.workerId);
  rec.check("the worker is idle again", mine?.status === "idle" && mine?.currentJobId == null, JSON.stringify(mine ?? null).slice(0, 200));
}

const ZH_STYLE = "Chinese Mandarin pop ballad, gentle female vocal, piano and strings, 84 BPM";
const ZH_LYRICS = "[Verse]\n夜色慢慢落在窗前\n灯火照亮回家的路\n风轻轻吹过旧街边\n我把思念写进歌里\n\n[Chorus]\n等天亮的时候\n我们一起唱\n让心里的光\n照到远方";
const EDIT_LYRICS = "[Verse]\nSilver rain across the square\nWhispers drifting through the air\nLeave the lamp beside the door\nMorning knows what it is for\n\n[Chorus]\nLet the light come shining through\nEvery song returns to you\nHold a little room for dawn\nWe will sing and carry on";
const EDIT_STYLE = "English, slow soul ballad, warm male voice, electric piano, soft bass and brushed drums, 88 BPM";
const FORM_LYRICS = "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\nFold the night and leave it here\nMorning has a sky to clear\n\n[Chorus]\nLet the day come into view\nEvery road begins with you\nHold a little room for light\nWe will sing beyond the night\n\n[Chorus]\nLet the day come into view\nEvery road begins with you\nHold a little room for light\nWe will sing beyond the night";

// ---- Orchestration ---------------------------------------------------------------------------------

async function loadFixtures() {
  const dir = path.join(ROOT, "crates", "sceneworks-core", "src", "yue2_score", "fixtures");
  const manifest = JSON.parse(stripJsoncComments(await readFile(path.join(ROOT, "config", "manifests", "builtin.models.jsonc"), "utf8")));
  return {
    song: JSON.parse(await readFile(path.join(dir, "song.json"), "utf8")),
    scoreAbc: await readFile(path.join(dir, "score.abc"), "utf8"),
    melodyAbc: await readFile(path.join(dir, "melody.abc"), "utf8"),
    jazzAbc: await readFile(path.join(dir, "score-jazz.abc"), "utf8"),
    manifest,
    manifestEntry: manifest.models.find((model) => model.id === MODEL_ID),
  };
}

async function writeJson(file, value) {
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`);
}

async function runCase(ctx, item, planned, records, sampler) {
  const rec = new Recorder(item, ctx.platform);
  const finish = async (status, reason) => {
    rec.record.status = status;
    rec.record.reason = reason;
    rec.record.finishedAt = nowIso();
    rec.record.durationSeconds = (Date.now() - rec.started) / 1000;
    try {
      validateRecord(rec.record);
    } catch (error) {
      // A record that cannot stand as evidence is a FAILED case, never a silently weaker pass.
      rec.record.status = "failed";
      rec.record.reason = `evidence incomplete: ${error.message}${reason ? `; ${reason}` : ""}`;
    }
    records.set(item.id, rec.record);
    await writeJson(path.join(ctx.paths.records, `${item.id}.json`), rec.record);
    console.log(`${item.id}: ${rec.record.status}${rec.record.reason ? ` — ${rec.record.reason}` : ""}`);
  };
  if (planned.action === "skip") return finish("skipped", planned.reason);
  const blocked = dependencySkip(item, records, { dryRun: ctx.options.dryRun });
  if (blocked) return finish("skipped", blocked);
  const sample = !ctx.options.dryRun;
  if (sample) sampler.start();
  let outcome = null;
  const stopSampler = async () => {
    if (!sample) return;
    rec.record.memorySampled = true;
    rec.record.peakMemory = await sampler.stop();
  };
  try {
    outcome = await CASE_RUNNERS[item.id](ctx, rec);
  } catch (error) {
    await stopSampler();
    return finish("failed", error instanceof CaseFailure ? error.message : `${error.name}: ${error.message}`);
  }
  await stopSampler();
  const failed = rec.record.assertions.filter((assertion) => !assertion.ok);
  if (failed.length) return finish("failed", failed.map((assertion) => `${assertion.name}: ${assertion.detail ?? ""}`).join("; "));
  if (outcome?.status === "blocked") return finish("blocked", outcome.reason);
  return finish("passed", null);
}

export async function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  if (process.platform !== PLATFORMS[options.platform].os) fail(`--platform ${options.platform} runs on ${PLATFORMS[options.platform].os}, not ${process.platform}`);
  const out = path.resolve(options.out);
  const evidence = path.join(out, "evidence");
  const dataDir = path.resolve(options.dataDir ?? path.join(out, "app-data"));
  const paths = {
    out,
    evidence,
    records: path.join(evidence, "records"),
    logs: path.join(evidence, "logs"),
    dataDir,
    configDir: `${dataDir}-config`,
    hfHub: path.resolve(options.hfHub ?? path.join(out, "hf-hub")),
  };
  if (insideRepository(dataDir) || insideRepository(out)) fail("--out and --data-dir must be outside the repository");
  if (existsSync(paths.records) && (await readdir(paths.records)).length) fail(`${paths.records} already holds records; use a fresh --out`);
  if (existsSync(path.join(paths.dataDir, "cache", "jobs.db"))) fail(`${paths.dataDir} already holds an app; the acceptance run starts cold — use a fresh --data-dir`);
  for (const dir of [paths.records, paths.logs, paths.dataDir, paths.configDir, paths.hfHub]) await mkdir(dir, { recursive: true });
  options.apiBin = path.resolve(options.apiBin ?? path.join(ROOT, "target", "release", process.platform === "win32" ? "sceneworks-rust-api.exe" : "sceneworks-rust-api"));
  if (!existsSync(options.apiBin)) fail(`no API binary at ${options.apiBin}; build it first (cargo build --release --locked -p sceneworks-rust-api${options.platform === "cuda" ? " --features backend-candle" : ""})`);

  const startedAt = nowIso();
  const identity = await runIdentity(options.apiBin);
  identity.hfHub = { path: paths.hfHub, mode: options.hfHub ? "preseeded hub (install re-verifies against the Hub)" : "fresh (install downloads)" };
  const plan = planCases(options);
  const service = new Service(options, paths);
  const ctx = new Context(options, service, paths);
  const sampler = new Sampler(options.platform, service, options.gpuId);
  const records = new Map();
  let stopping = false;
  const onSignal = (signal) => {
    if (stopping) return;
    stopping = true;
    console.error(`yue2-acceptance: ${signal}; stopping the service`);
    service.stop({ graceful: true }).finally(() => process.exit(130));
  };
  process.once("SIGINT", onSignal);
  process.once("SIGTERM", onSignal);
  let fatal = null;
  try {
    ctx.fixtures = await loadFixtures();
    identity.health = await service.startApi();
    if (!options.dryRun) identity.worker = await service.startWorker();
    const project = await service.request("POST", "/api/v1/projects", { name: `YuE2 acceptance ${options.platform}` });
    if (project.status !== 201) fail(`could not create the acceptance project: HTTP ${project.status} ${JSON.stringify(project.body)}`);
    ctx.project = project.body;
    for (const item of CASES) await runCase(ctx, item, plan.find((entry) => entry.id === item.id), records, sampler);
  } catch (error) {
    fatal = error;
    console.error(error.stack ?? error.message);
  } finally {
    await service.stop({ graceful: true }).catch((error) => console.error(`service stop: ${error.message}`));
  }
  const ordered = CASES.map((item) => records.get(item.id)).filter(Boolean);
  const summary = buildSummary(ordered, { platform: options.platform, dryRun: options.dryRun, identity, startedAt, finishedAt: nowIso(), fatal: fatal?.message });
  await writeJson(path.join(evidence, "summary.json"), summary);
  await writeFile(path.join(evidence, "summary.md"), renderMarkdown(summary));
  console.log(`verdict: ${summary.verdict} → ${path.join(evidence, "summary.md")}`);
  return summary.verdict === "pass" || (options.dryRun && summary.counts.failed === 0 && !fatal) ? 0 : 1;
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().then((code) => { process.exitCode = code; }, (error) => { console.error(error.message); process.exitCode = 1; });
}
