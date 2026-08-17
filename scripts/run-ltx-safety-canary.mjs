#!/usr/bin/env node

import { execFile, spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { constants as fsConstants, createReadStream } from "node:fs";
import {
  chmod, copyFile, link, lstat, mkdtemp, mkdir, open, readFile, readdir, realpath, rm,
  rename, stat, unlink, writeFile,
} from "node:fs/promises";
import { arch } from "node:os";
import path from "node:path";
import { createInterface } from "node:readline";
import { isDeepStrictEqual, promisify } from "node:util";
import { fileURLToPath } from "node:url";

import { hashArtifactInventory } from "./hash-artifact-inventory.mjs";
import {
  canonicalJson, expandPlan, runProviderPlan, validateBundle,
} from "./memory-calibration-harness.mjs";

const execFileAsync = promisify(execFile);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CONTAINED_CAMPAIGN_ROOT = "/Volumes/Data/sceneworks-safety-canary";
export const MAX_FOOTPRINT_BYTES = 53_347_146_863;
export const MAX_RUNTIME_SECONDS = 1_800;
export const CHILD_ATTESTATION_TIMEOUT_SECONDS = 30;
export const CARGO_METADATA_TIMEOUT_MS = 15 * 60 * 1000;
export const PREPARATION_SCHEMA_VERSION = 3;
export const PREPARATION_LOCK_POLL_MS = 50;
export const PREPARATION_LOCK_ORPHAN_GRACE_MS = 30_000;
// Free swap capacity is not a safety margin: the prelaunch free-memory floor is already more than
// two hard footprint limits, and the watchdog continuously enforces a full-limit runtime margin.
// Swap remains mandatory telemetry, but there is deliberately no arbitrary free-capacity floor.
export const MIN_PREFLIGHT_FREE_BYTES = MAX_FOOTPRINT_BYTES * 2;
export const MIN_RUNTIME_FREE_BYTES = MAX_FOOTPRINT_BYTES;
const PROVIDER = "ltx_2_3";
const FINGERPRINT = "sc-19109-ltx-2-3-mlx-memory-ladder-v1";
export const SAFETY_CANARY_PROFILE = "safety";
export const PRODUCT_ENVELOPE_CANARY_PROFILE = "product-envelope";
export const CAMPAIGN_ENTRY_PROFILE = "campaign-entry";
export const BOUNDED_CARRIER_PROFILE = "bounded-carrier";
export const BOUNDED_CAMPAIGN_ENTRY_PROFILE = "bounded-campaign-entry";
export const BOUNDED_CAMPAIGN_ENTRY_Q8_PROFILE = "bounded-campaign-entry-q8";
export const BOUNDED_CAMPAIGN_ENTRY_BF16_PROFILE = "bounded-campaign-entry-bf16";
export const CAMPAIGN_ENTRY_PROVIDER =
  "mlx-ltx-2-3-q4-768x512-f121-fps30-staged_residency";
export const CAMPAIGN_ENTRY_FIXTURE =
  "ltx-2-3-mlx-q4-768x512-f121-fps30-seed18946";
export const CAMPAIGN_ENTRY_LOGICAL_CASE_ID = "implan-9b107d4d1ca0d61d4faa";
export const CAMPAIGN_ENTRY_IDENTITY = "sc-20191-q4-768x512-f121-fps30-staged-v1";
export const CAMPAIGN_FAILURE_RECEIPT_TYPE = "sceneworks_campaign_entry_failure_v1";
export const BOUNDED_CARRIER_ACTION = "bounded_carrier_proof";
export const BOUNDED_CARRIER_IDENTITY =
  "sc-20254-q4-768x512-f121-fps30-bounded-192-64-v1";
export const BOUNDED_CARRIER_LOGICAL_CASE_ID =
  "diagnostic-sc20254-q4-768x512-f121-fps30-bounded-192-64";
export const BOUNDED_CARRIER_FIXTURE =
  "ltx-2-3-mlx-q4-768x512-f121-fps30-seed18946-bounded-decode-192-64-proof";
export const BOUNDED_CARRIER_SUCCESS_RECEIPT_TYPE =
  "sceneworks_bounded_carrier_success_v1";
export const BOUNDED_CARRIER_FAILURE_RECEIPT_TYPE =
  "sceneworks_bounded_carrier_failure_v1";
export const BOUNDED_CAMPAIGN_ENTRY_ACTION = "bounded_campaign_entry";
export const BOUNDED_CAMPAIGN_ENTRY_PROVIDER =
  "mlx-ltx-2-3-q4-768x512-f121-fps30-bounded_decode-192x64";
export const BOUNDED_CAMPAIGN_ENTRY_FIXTURE =
  "ltx-2-3-mlx-q4-768x512-f121-fps30-bounded-decode-192-64-seed18946";
export const BOUNDED_CAMPAIGN_ENTRY_LOGICAL_CASE_ID = "implan-964db61ed3789af6386b";
export const BOUNDED_CAMPAIGN_ENTRY_IDENTITY =
  "sc-20318-q4-768x512-f121-fps30-bounded-192-64-authoritative-v1";
export const BOUNDED_CAMPAIGN_FAILURE_RECEIPT_TYPE =
  "sceneworks_bounded_campaign_entry_failure_v1";
export const PROVIDER_PHASE_PROTOCOL = "sceneworks-provider-phase-v1";
export const WATCHDOG_EVENT_CHAIN_PROTOCOL = "sceneworks-watchdog-event-chain-v1";
export const PROVIDER_PHASES = Object.freeze([
  "common_load",
  "primary_conditioning",
  "primary_denoise",
  "primary_decode",
  "lifecycle_warm_repeat",
  "lifecycle_cancel",
  "lifecycle_cancel_recovery",
  "lifecycle_error",
  "lifecycle_error_recovery",
  "cleanup",
]);
export const BOUNDED_CARRIER_PHASES = Object.freeze([
  "common_load",
  "primary_conditioning",
  "primary_denoise",
  "primary_decode",
  "cleanup",
]);
const CAMPAIGN_ENTRY_ACTION = "campaign_entry";
const CAMPAIGN_ENTRY_PLAN = "docs/calibration/sc-18946/ltx-mlx-single-pass-sweep.json";
const BOUNDED_CAMPAIGN_ENTRY_PLAN = "docs/calibration/sc-18946/ltx-mlx-rung2-sweep.json";
const CAMPAIGN_ENTRY_SAFETY = Object.freeze({
  disposition: "safety_refused_open",
  tierInventoryBytes: 20_467_690_460,
  incidentCrashFootprintBytes: 96_970_084_480,
  incidentCase: "mlx-ltx-2-3-q4-1280x704-f305-fps30-bounded_decode",
  commonLoad: "complete numeric tier plus shared Gemma stack before geometry-specific work",
  predictedDecodeBytes: 19_476_906_240,
  incidentPredictedDecodeBytes: 18_540_396_800,
  incidentCalibratedProjectionBytes: 97_906_593_920,
  projectionAssumptions: Object.freeze([
    "pinned provider decode cost is the only geometry-varying term used",
    "immutable tier inventory delta is added byte-for-byte",
    "incident binding phase is unknown, so the projection is not a physical-footprint bound and cannot admit execution",
  ]),
  reason: "incident-calibrated projection is diagnostic only; no proved bound or hard containment admits this row",
});
const BOUNDED_CAMPAIGN_ENTRY_SAFETY = Object.freeze({
  disposition: "safety_refused_open",
  tierInventoryBytes: 20_467_690_460,
  incidentCrashFootprintBytes: 96_970_084_480,
  incidentCase: "mlx-ltx-2-3-q4-1280x704-f305-fps30-bounded_decode",
  commonLoad: "complete numeric tier plus shared Gemma stack before geometry-specific work",
  predictedDecodeBytes: 6_264_848_640,
  incidentPredictedDecodeBytes: 18_540_396_800,
  incidentCalibratedProjectionBytes: 84_694_536_320,
  projectionAssumptions: CAMPAIGN_ENTRY_SAFETY.projectionAssumptions,
  reason: "incident-calibrated projection is diagnostic only; ordinary run remains refused and only the exact privately contained SC-20318 action is admitted",
});
export const BOUNDED_CAMPAIGN_ENTRY_SPECS = Object.freeze({
  q4: Object.freeze({
    profile: BOUNDED_CAMPAIGN_ENTRY_PROFILE,
    story: "sc-20318",
    tier: "q4",
    provider: BOUNDED_CAMPAIGN_ENTRY_PROVIDER,
    fixture: BOUNDED_CAMPAIGN_ENTRY_FIXTURE,
    logicalCaseId: BOUNDED_CAMPAIGN_ENTRY_LOGICAL_CASE_ID,
    identity: BOUNDED_CAMPAIGN_ENTRY_IDENTITY,
  }),
  q8: Object.freeze({
    profile: BOUNDED_CAMPAIGN_ENTRY_Q8_PROFILE,
    story: "sc-20430",
    tier: "q8",
    provider: "mlx-ltx-2-3-q8-768x512-f121-fps30-bounded_decode-192x64",
    fixture: "ltx-2-3-mlx-q8-768x512-f121-fps30-bounded-decode-192-64-seed18946",
    logicalCaseId: "implan-d47640caa0c469f2ee13",
    identity: "sc-20430-q8-768x512-f121-fps30-bounded-192-64-authoritative-v1",
  }),
  bf16: Object.freeze({
    profile: BOUNDED_CAMPAIGN_ENTRY_BF16_PROFILE,
    story: "sc-20430",
    tier: "bf16",
    provider: "mlx-ltx-2-3-bf16-768x512-f121-fps30-bounded_decode-192x64",
    fixture: "ltx-2-3-mlx-bf16-768x512-f121-fps30-bounded-decode-192-64-seed18946",
    logicalCaseId: "implan-b3926164bf6bfbee98e1",
    identity: "sc-20430-bf16-768x512-f121-fps30-bounded-192-64-authoritative-v1",
  }),
});

function boundedCampaignEntrySpec(value = "q4") {
  const spec = BOUNDED_CAMPAIGN_ENTRY_SPECS[value]
    ?? Object.values(BOUNDED_CAMPAIGN_ENTRY_SPECS).find((candidate) => candidate.profile === value);
  if (spec === undefined) fail(`unsupported bounded campaign entry ${JSON.stringify(value)}`);
  return spec;
}

function boundedCampaignSafety(spec) {
  if (spec.tier === "q4") return structuredClone(BOUNDED_CAMPAIGN_ENTRY_SAFETY);
  const inventory = NUMERIC_TIER_INVENTORIES[spec.tier];
  return {
    ...structuredClone(BOUNDED_CAMPAIGN_ENTRY_SAFETY),
    tierInventoryBytes: inventory.bytes,
    incidentCalibratedProjectionBytes:
      BOUNDED_CAMPAIGN_ENTRY_SAFETY.incidentCalibratedProjectionBytes
      + inventory.bytes - Q4_INVENTORY_BYTES,
    reason: "incident-calibrated projection is diagnostic only; ordinary run remains refused and only the exact privately contained SC-20430 action is admitted",
  };
}
const CANARY_PROFILES = Object.freeze({
  [SAFETY_CANARY_PROFILE]: Object.freeze({
    action: "canary",
    identity: "sc-19741-safety",
    story: "sc-19741",
    status: "diagnostic_canary_complete",
    fixture: "ltx-2-3-mlx-q4-256x256-f9-fps24-seed1234-safety-canary",
    width: 256,
    height: 256,
    frames: 9,
    fps: 24,
    videoMode: "no_audio",
    audio: false,
    spatialDecodeTiles: 4,
    frameTimelineSeconds: 1 / 3,
  }),
  [PRODUCT_ENVELOPE_CANARY_PROFILE]: Object.freeze({
    action: "product_envelope_canary",
    identity: "sc-20169-product-envelope",
    story: "sc-20169",
    status: "diagnostic_product_envelope_canary_complete",
    fixture: "ltx-2-3-mlx-q4-768x512-f97-fps24-seed1234-product-envelope-canary",
    width: 768,
    height: 512,
    frames: 97,
    fps: 24,
    videoMode: "default_av",
    audio: true,
    spatialDecodeTiles: 24,
    frameTimelineSeconds: 4,
  }),
});
const ARTIFACT_REPOSITORY = "SceneWorks/ltx-2.3-mlx";
const ARTIFACT_REVISION = "01df27d308466533aa09d251e3aebdcc627d07eb";
const Q4_INVENTORY_BYTES = 20_467_690_460;
const Q4_INVENTORY_SHA256 = "4e811932e87bb258f642ada790525e36ef2a55959c520e755f1807caf6fa225a";
const NUMERIC_TIER_INVENTORIES = Object.freeze({
  q4: Object.freeze({ files: 11, bytes: Q4_INVENTORY_BYTES, sha256: Q4_INVENTORY_SHA256 }),
  q8: Object.freeze({
    files: 11,
    bytes: 29_728_720_716,
    sha256: "bb0bb7577157a158ca39494837d64cb36ded0380ca7ee0c930fea7311f22a247",
  }),
  bf16: Object.freeze({
    files: 10,
    bytes: 47_092_811_992,
    sha256: "006caeaa9a8638b337cdf5a8622ce8535380b18ebaf90b36c3e2d5d15354f2a8",
  }),
});
const TEXT_ENCODER_INVENTORY_FILES = 17;
const TEXT_ENCODER_INVENTORY_BYTES = 26_427_894_918;
const TEXT_ENCODER_INVENTORY_SHA256 = "abde2d155aa8991747cc2999d40688d29a50261c080c0d51fac20357653928d7";
const LTX_ONES_CACHE_IDENTITY = "mlx-gen-ltx-transformer-ones-cache-av-bfloat16-v1";
const LTX_ONES_CACHE_VIDEO_DIMENSION = 4_096;
const LTX_ONES_CACHE_AUDIO_DIMENSION = 2_048;
const BFLOAT16_BYTES_PER_ELEMENT = 2;

function ltxOnesCacheBytes() {
  const elements = LTX_ONES_CACHE_VIDEO_DIMENSION + LTX_ONES_CACHE_AUDIO_DIMENSION;
  const bytes = elements * BFLOAT16_BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(bytes)) fail("LTX ONES_CACHE byte arithmetic overflowed");
  return bytes;
}

function fail(message) {
  throw new Error(message);
}

function canaryProfile(name) {
  const profile = CANARY_PROFILES[name];
  if (profile === undefined) fail(`unsupported canary profile ${JSON.stringify(name)}`);
  return profile;
}

export class CanaryInterrupted extends Error {
  constructor(signalName) {
    super(`canary runner interrupted by ${signalName}`);
    this.signalName = signalName;
    this.exitCode = 128 + (signalName === "SIGINT" ? 2 : 15);
  }
}

export function telemetryResolutionBytes(memoryBytes) {
  if (!Number.isSafeInteger(memoryBytes) || memoryBytes <= 0) fail("host memory must be a positive integer");
  return Math.ceil(memoryBytes / 100);
}

export function preflightFreeFloor(memoryBytes) {
  return MIN_PREFLIGHT_FREE_BYTES + telemetryResolutionBytes(memoryBytes);
}

export function runtimeFreeFloor(memoryBytes) {
  return MIN_RUNTIME_FREE_BYTES + telemetryResolutionBytes(memoryBytes);
}

export function privateArtifactRoots(scratch, tier = "q4") {
  if (!Object.hasOwn(NUMERIC_TIER_INVENTORIES, tier)) fail(`unsupported prepared tier ${tier}`);
  const snapshotRoot = path.join(
    scratch,
    "artifacts",
    `models--${ARTIFACT_REPOSITORY.replaceAll("/", "--")}`,
    "snapshots",
    ARTIFACT_REVISION,
  );
  return {
    numericTier: path.join(snapshotRoot, tier),
    textEncoder: path.join(snapshotRoot, "gemma"),
  };
}

export function preparationIdentity(sceneWorksTree, inferenceTree, toolchainChannel, tier = "q4") {
  for (const [label, value] of [
    ["SceneWorks tree", sceneWorksTree], ["inference tree", inferenceTree],
  ]) {
    if (!/^[0-9a-f]{40}$/.test(value)) fail(`${label} must be an exact git tree`);
  }
  if (!/^\d+\.\d+\.\d+$/.test(toolchainChannel)) fail("toolchain channel must be exact");
  const numericTier = NUMERIC_TIER_INVENTORIES[tier];
  if (numericTier === undefined) fail(`unsupported preparation tier ${tier}`);
  return {
    schemaVersion: PREPARATION_SCHEMA_VERSION,
    sceneWorksTree,
    inferenceTree,
    toolchainChannel,
    platform: process.platform,
    architecture: arch(),
    artifact: {
      repository: ARTIFACT_REPOSITORY,
      revision: ARTIFACT_REVISION,
      numericTier: structuredClone(numericTier),
      textEncoder: {
        files: TEXT_ENCODER_INVENTORY_FILES,
        bytes: TEXT_ENCODER_INVENTORY_BYTES,
        sha256: TEXT_ENCODER_INVENTORY_SHA256,
      },
    },
  };
}

export function preparationCacheKey(identity) {
  return createHash("sha256").update(JSON.stringify(identity)).digest("hex");
}

export function preparationCacheRoot(output, key) {
  if (!/^[0-9a-f]{64}$/.test(key)) fail("preparation cache key must be an exact SHA-256");
  return path.join(path.dirname(path.resolve(output)), "prepared", key);
}

export function preparationLockPath(preparationRoot) {
  return path.join(path.dirname(preparationRoot), `.${path.basename(preparationRoot)}.lock`);
}

export async function sealedArtifactIdentity(root) {
  const digest = createHash("sha256");
  let files = 0;
  let bytes = 0;
  async function visit(directory, relativeDirectory) {
    const directoryMetadata = await lstat(directory, { bigint: true });
    if (!directoryMetadata.isDirectory() || directoryMetadata.isSymbolicLink()
        || Number(directoryMetadata.mode & 0o777n) !== 0o500) {
      fail(`sealed artifact directory changed: ${directory}`);
    }
    digest.update(`${relativeDirectory}/\0d\0${directoryMetadata.dev}\0${directoryMetadata.ino}`
      + `\0${directoryMetadata.mtimeNs}\0${directoryMetadata.ctimeNs}\n`);
    const entries = await readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      const absolute = path.join(directory, entry.name);
      const relative = path.posix.join(relativeDirectory, entry.name);
      const metadata = await lstat(absolute, { bigint: true });
      if (entry.isDirectory()) {
        await visit(absolute, relative);
      } else if (entry.isFile() && !metadata.isSymbolicLink()
          && Number(metadata.mode & 0o777n) === 0o400) {
        files += 1;
        bytes += Number(metadata.size);
        digest.update(`${relative}\0f\0${metadata.dev}\0${metadata.ino}\0${metadata.size}`
          + `\0${metadata.mtimeNs}\0${metadata.ctimeNs}\n`);
      } else {
        fail(`sealed artifact entry changed: ${absolute}`);
      }
    }
  }
  await visit(root, ".");
  return { files, bytes, sha256: digest.digest("hex") };
}

export function watchdogFailureSummary(status, eventBytes) {
  let hardStopReason = null;
  let validEvents = 0;
  let malformed = false;
  for (const line of eventBytes.trim() ? eventBytes.trim().split("\n") : []) {
    try {
      const event = JSON.parse(line);
      validEvents += 1;
      if (event?.event === "hard_stop" && typeof event.reason === "string"
          && event.reason !== "") {
        hardStopReason = event.reason;
      }
    } catch {
      malformed = true;
    }
  }
  let reason = "event_stream_unavailable";
  if (hardStopReason !== null) {
    reason = hardStopReason.replaceAll(/[\u0000-\u001f\u007f]/g, " ").slice(0, 512);
    if (malformed) reason += ";event_stream_malformed";
  } else if (malformed) {
    reason = "event_stream_malformed";
  } else if (validEvents > 0) {
    reason = "hard_stop_event_absent";
  }
  return `watchdog failed closed: code=${status.code} signal=${status.signal} reason=${reason}`;
}

export function preservePrimaryFailure(primary, cleanup) {
  if (primary === null) return cleanup;
  if (cleanup === null) return primary;
  const error = primary instanceof Error ? primary : new Error(String(primary));
  const cleanupMessage = cleanup instanceof Error ? cleanup.message : String(cleanup);
  error.message += `; scratch cleanup failed: ${cleanupMessage
    .replaceAll(/[\u0000-\u001f\u007f]/g, " ").slice(0, 512)}`;
  if (error.cause === undefined) error.cause = cleanup;
  return error;
}

export function preserveFailureReceiptSuppression(primary, receiptError) {
  const error = primary instanceof Error ? primary : new Error(String(primary));
  const detail = receiptError instanceof Error ? receiptError.message : String(receiptError);
  error.message += `; SC-20216 failure receipt suppressed: ${detail
    .replaceAll(/[\u0000-\u001f\u007f]/g, " ").slice(0, 512)}`;
  if (error.cause === undefined) error.cause = receiptError;
  return error;
}

function killProcessGroup(child, signalName) {
  if (child.pid === undefined) return;
  try {
    process.kill(-child.pid, signalName);
  } catch (error) {
    if (error.code !== "ESRCH") throw error;
  }
}

export async function runOwnedCommand(executable, args, {
  cwd = ROOT,
  env = process.env,
  signal,
  timeout = 0,
  maxBuffer = 10 * 1024 * 1024,
} = {}) {
  signal?.throwIfAborted();
  return new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    let terminationReason = null;
    let killTimer = null;
    let timeoutTimer = null;
    const child = spawn(executable, args, {
      cwd, env, detached: true, stdio: ["ignore", "pipe", "pipe"],
    });
    const terminate = (reason) => {
      if (terminationReason !== null) return;
      terminationReason = reason;
      killProcessGroup(child, "SIGTERM");
      killTimer = setTimeout(() => killProcessGroup(child, "SIGKILL"), 2_000);
      killTimer.unref();
    };
    const onAbort = () => terminate(signal.reason ?? new Error("command aborted"));
    if (signal) signal.addEventListener("abort", onAbort, { once: true });
    if (timeout > 0) {
      timeoutTimer = setTimeout(
        () => terminate(new Error(`command timed out after ${timeout} ms: ${executable}`)), timeout,
      );
      timeoutTimer.unref();
    }
    const append = (stream, chunk) => {
      const next = stream + chunk;
      if (Buffer.byteLength(next) > maxBuffer) {
        terminate(new Error(`command output exceeded ${maxBuffer} bytes: ${executable}`));
      }
      return next;
    };
    child.stdout.on("data", (chunk) => { stdout = append(stdout, chunk); });
    child.stderr.on("data", (chunk) => { stderr = append(stderr, chunk); });
    child.on("error", (error) => terminate(error));
    child.on("close", (code, childSignal) => {
      if (signal) signal.removeEventListener("abort", onAbort);
      if (timeoutTimer !== null) clearTimeout(timeoutTimer);
      if (killTimer !== null) clearTimeout(killTimer);
      if (terminationReason !== null) {
        killProcessGroup(child, "SIGKILL");
        reject(terminationReason);
        return;
      }
      if (code !== 0 || childSignal !== null) {
        const error = new Error(
          `command failed: ${executable} (code=${code}, signal=${childSignal})\n${stderr}`,
        );
        error.code = code;
        error.signal = childSignal;
        error.stdout = stdout;
        error.stderr = stderr;
        reject(error);
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

async function git(cwd, args, signal) {
  return (await execFileAsync("git", ["-C", cwd, ...args], {
    encoding: "utf8", signal,
  })).stdout.trim();
}

async function cleanHead(cwd, label, signal) {
  const head = await git(cwd, ["rev-parse", "HEAD"], signal);
  if ((await git(cwd, ["status", "--porcelain"], signal)) !== "") fail(`${label} repository is dirty`);
  return head;
}

async function observedSourceState(cwd, expectedRevision, expectedTree) {
  try {
    const [revision, tree, status] = await Promise.all([
      git(cwd, ["rev-parse", "HEAD"]),
      git(cwd, ["rev-parse", "HEAD^{tree}"]),
      git(cwd, ["status", "--porcelain"]),
    ]);
    return {
      observed: true,
      clean: status === "",
      revision,
      tree,
      matchesPrelaunch: status === "" && revision === expectedRevision && tree === expectedTree,
    };
  } catch (error) {
    return {
      observed: false,
      matchesPrelaunch: false,
      error: String(error?.message ?? error)
        .replaceAll(/[\u0000-\u001f\u007f]/g, " ").slice(0, 512),
    };
  }
}

export function cargoSourceStatusIsClean(status) {
  const lines = status.trim() ? status.trim().split("\n") : [];
  return lines.length === 0 || (lines.length === 1 && lines[0] === "?? .cargo-ok");
}

async function cleanCargoSourceHead(cwd, signal) {
  const head = await git(cwd, ["rev-parse", "HEAD"], signal);
  const status = await git(cwd, ["status", "--porcelain=v1", "--untracked-files=all"], signal);
  if (!cargoSourceStatusIsClean(status)) fail("Cargo inference source repository is dirty");
  if (status !== "") {
    const sentinel = await lstat(path.join(cwd, ".cargo-ok"));
    if (!sentinel.isFile() || sentinel.isSymbolicLink() || sentinel.size !== 0) {
      fail("Cargo inference source .cargo-ok sentinel is not Cargo's empty regular file");
    }
  }
  return head;
}

export function canaryRequest(
  memoryBytes,
  textEncoderInventory,
  profileName = SAFETY_CANARY_PROFILE,
) {
  const profile = canaryProfile(profileName);
  assertInventory(textEncoderInventory, {
    files: TEXT_ENCODER_INVENTORY_FILES,
    bytes: TEXT_ENCODER_INVENTORY_BYTES,
    sha256: TEXT_ENCODER_INVENTORY_SHA256,
  }, "immutable text-encoder");
  return {
    action: profile.action,
    hardware: { memoryBytes },
    planned: {
      _diagnosticOnly: true,
      evidenceScope: "fixture",
      target: {
        provider: PROVIDER,
        modelId: PROVIDER,
        tier: "q4",
        mode: "text_to_video",
        overlay: "none",
        geometry: {
          width: profile.width, height: profile.height, batch: 1, frames: profile.frames,
        },
      },
      backend: "mlx",
      loadShape: "eager_materialization",
      strategy: {
        rung: "bounded_decode",
        engagedRungs: ["resident", "staged_residency", "bounded_decode"],
        parameters: { decodeTileEdge: 192, decodeOverlap: 64 },
      },
      calibrationFingerprint: FINGERPRINT,
      fixture: profile.fixture,
      _watchdog: { maxFootprintBytes: MAX_FOOTPRINT_BYTES },
      _canary: {
        identity: profile.identity, videoMode: profile.videoMode, fps: profile.fps, seed: 1234,
      },
      _artifact: {
        repository: ARTIFACT_REPOSITORY,
        revision: ARTIFACT_REVISION,
        numericTierInventory: {
          files: 11, bytes: Q4_INVENTORY_BYTES, sha256: Q4_INVENTORY_SHA256,
        },
        textEncoderInventory,
      },
    },
  };
}

export function boundedCarrierRequest(memoryBytes, textEncoderInventory) {
  assertInventory(textEncoderInventory, {
    files: TEXT_ENCODER_INVENTORY_FILES,
    bytes: TEXT_ENCODER_INVENTORY_BYTES,
    sha256: TEXT_ENCODER_INVENTORY_SHA256,
  }, "immutable text-encoder");
  return {
    action: BOUNDED_CARRIER_ACTION,
    hardware: { memoryBytes },
    planned: {
      _diagnosticOnly: true,
      logicalCaseId: BOUNDED_CARRIER_LOGICAL_CASE_ID,
      evidenceScope: "fixture",
      backend: "mlx",
      loadShape: "eager_materialization",
      target: {
        provider: PROVIDER,
        modelId: PROVIDER,
        tier: "q4",
        mode: "text_to_video",
        overlay: "none",
        geometry: { width: 768, height: 512, batch: 1, frames: 121 },
      },
      strategy: {
        rung: "bounded_decode",
        engagedRungs: ["resident", "staged_residency", "bounded_decode"],
        parameters: { decodeTileEdge: 192, decodeOverlap: 64 },
      },
      calibrationFingerprint: FINGERPRINT,
      fixture: BOUNDED_CARRIER_FIXTURE,
      negative: false,
      expectedResult: "passed",
      modelLoadPolicy: "fresh_per_case",
      modelLoadGroup: null,
      _watchdog: { maxFootprintBytes: MAX_FOOTPRINT_BYTES },
      _boundedCarrier: {
        identity: BOUNDED_CARRIER_IDENTITY,
        fps: 30,
        seed: 18_946,
        videoMode: "default_av",
        artifact: {
          repository: ARTIFACT_REPOSITORY,
          revision: ARTIFACT_REVISION,
          numericTierInventory: {
            files: 11, bytes: Q4_INVENTORY_BYTES, sha256: Q4_INVENTORY_SHA256,
          },
          textEncoderInventory: {
            files: textEncoderInventory.files,
            bytes: textEncoderInventory.bytes,
            sha256: textEncoderInventory.sha256,
          },
        },
      },
    },
  };
}

export function campaignEntryPlan(config) {
  const matches = config?.providers?.filter((provider) =>
    provider.name === CAMPAIGN_ENTRY_PROVIDER) ?? [];
  if (matches.length !== 1) fail("SC-20191 requires exactly one frozen campaign-entry provider");
  const provider = matches[0];
  const exactProvider = {
    name: CAMPAIGN_ENTRY_PROVIDER,
    _role: "fit",
    _campaignPhase: "original_fit",
    _coverageExpectation: "captured_or_attempted_or_arithmetic_unmeasurable",
    _latentTokens: 6_144,
    _writableFrameCap: 682,
    _outputVoxels: 47_579_136,
    _measurementSafety: structuredClone(CAMPAIGN_ENTRY_SAFETY),
    evidenceScope: "authoritative",
    backend: "mlx",
    loadShape: "eager_materialization",
    target: {
      provider: PROVIDER,
      modelId: PROVIDER,
      tier: "q4",
      mode: "text_to_video",
      overlay: "none",
      geometry: { width: 768, height: 512, batch: 1, frames: 121 },
    },
    rung: "staged_residency",
    engagedRungs: ["resident", "staged_residency"],
    calibrationFingerprint: FINGERPRINT,
    fixture: CAMPAIGN_ENTRY_FIXTURE,
    cases: [{ parameters: {}, expectedResult: "passed" }],
  };
  if (!isDeepStrictEqual(provider, exactProvider)) {
    fail("SC-20191 frozen campaign-entry provider changed");
  }
  const planned = expandPlan({ providers: [provider] });
  if (planned.length !== 1
      || planned[0].logicalCaseId !== CAMPAIGN_ENTRY_LOGICAL_CASE_ID
      || planned[0].modelLoadPolicy !== "fresh_per_case"
      || planned[0].modelLoadGroup !== null) {
    fail("SC-20191 campaign-entry logical identity changed");
  }
  return { provider, planned: planned[0] };
}

export function boundedCampaignEntryPlan(config, requested = "q4") {
  const spec = boundedCampaignEntrySpec(requested);
  const matches = config?.providers?.filter((provider) =>
    provider.name === spec.provider) ?? [];
  if (matches.length !== 1) fail(`${spec.story} requires exactly one bounded campaign provider`);
  const provider = matches[0];
  const exactProvider = {
    name: spec.provider,
    _role: "bounded_carrier_entry",
    _campaignPhase: "bounded_carrier_ratification",
    _coverageExpectation: "captured_or_attempted",
    _latentTokens: 6_144,
    _writableFrameCap: 682,
    _outputVoxels: 47_579_136,
    _measurementSafety: boundedCampaignSafety(spec),
    evidenceScope: "authoritative",
    backend: "mlx",
    loadShape: "eager_materialization",
    target: {
      provider: PROVIDER, modelId: PROVIDER, tier: spec.tier, mode: "text_to_video",
      overlay: "none", geometry: { width: 768, height: 512, batch: 1, frames: 121 },
    },
    rung: "bounded_decode",
    engagedRungs: ["resident", "staged_residency", "bounded_decode"],
    calibrationFingerprint: FINGERPRINT,
    fixture: spec.fixture,
    cases: [{ parameters: { decodeTileEdge: 192, decodeOverlap: 64 }, expectedResult: "passed" }],
    _predictedDecodeBytes: 6_264_848_640,
    _predictedDecodeFormula:
      "3.3e9 + 40*(width*height*frames) + 300*(192*192*96) bytes",
  };
  if (!isDeepStrictEqual(provider, exactProvider)) {
    fail(`${spec.story} bounded campaign provider changed`);
  }
  const planned = expandPlan({ providers: [provider] });
  if (planned.length !== 1
      || planned[0].logicalCaseId !== spec.logicalCaseId
      || planned[0].modelLoadPolicy !== "fresh_per_case"
      || planned[0].modelLoadGroup !== null) {
    fail(`${spec.story} bounded campaign logical identity changed`);
  }
  return { provider, planned: planned[0], spec };
}

export function validateCampaignEntryHarnessRequest(request, expectedPlanned, {
  sceneWorksRevision, inferenceRevision, sceneWorksRepo, inferenceRepo,
} = {}) {
  if (request?.action !== "run"
      || !isDeepStrictEqual(request?.planned, expectedPlanned)
      || request.planned.logicalCaseId !== CAMPAIGN_ENTRY_LOGICAL_CASE_ID
      || request.planned.evidenceScope !== "authoritative"
      || request.planned.fixture !== CAMPAIGN_ENTRY_FIXTURE
      || request.planned.negative !== false
      || request.planned.expectedResult !== "passed"
      || request.planned.modelLoadPolicy !== "fresh_per_case"
      || request.planned.modelLoadGroup !== null
      || request.repositories?.sceneWorks?.dirty !== false
      || request.repositories?.inference?.dirty !== false
      || (sceneWorksRevision !== undefined
        && request.repositories.sceneWorks.revision !== sceneWorksRevision)
      || (inferenceRevision !== undefined
        && request.repositories.inference.revision !== inferenceRevision)
      || (sceneWorksRepo !== undefined && request.repositoryPaths?.sceneWorks !== sceneWorksRepo)
      || (inferenceRepo !== undefined && request.repositoryPaths?.inference !== inferenceRepo)
      || !Number.isSafeInteger(request.hardware?.memoryBytes)
      || request.hardware.memoryBytes <= 0) {
    fail("SC-20191 provider wrapper received a non-canonical harness request");
  }
  for (const privateField of ["_watchdog", "_campaignEntry", "_measurementSafety"]) {
    if (Object.hasOwn(request.planned, privateField)) {
      fail(`canonical SC-18946 request unexpectedly contains ${privateField}`);
    }
  }
  return request;
}

export function campaignEntryAdapterRequest(request, expectedPlanned, expectedSource) {
  validateCampaignEntryHarnessRequest(request, expectedPlanned, expectedSource);
  const transformed = structuredClone(request);
  transformed.action = CAMPAIGN_ENTRY_ACTION;
  transformed.planned._watchdog = { maxFootprintBytes: MAX_FOOTPRINT_BYTES };
  transformed.planned._measurementSafety = structuredClone(CAMPAIGN_ENTRY_SAFETY);
  transformed.planned._campaignEntry = {
    identity: CAMPAIGN_ENTRY_IDENTITY,
    artifact: {
      repository: ARTIFACT_REPOSITORY,
      revision: ARTIFACT_REVISION,
      numericTierInventory: {
        files: 11, bytes: Q4_INVENTORY_BYTES, sha256: Q4_INVENTORY_SHA256,
      },
      textEncoderInventory: {
        files: TEXT_ENCODER_INVENTORY_FILES,
        bytes: TEXT_ENCODER_INVENTORY_BYTES,
        sha256: TEXT_ENCODER_INVENTORY_SHA256,
      },
    },
  };
  return transformed;
}

export function boundedCampaignEntryAdapterRequest(
  request, expectedPlanned, expectedSource, requested = "q4",
) {
  const spec = boundedCampaignEntrySpec(requested);
  if (request?.action !== "run"
      || !isDeepStrictEqual(request?.planned, expectedPlanned)
      || request.planned.logicalCaseId !== spec.logicalCaseId
      || request.planned.fixture !== spec.fixture
      || request.planned.evidenceScope !== "authoritative"
      || request.planned.modelLoadPolicy !== "fresh_per_case"
      || request.planned.modelLoadGroup !== null
      || request.repositories?.sceneWorks?.dirty !== false
      || request.repositories?.inference?.dirty !== false
      || (expectedSource?.sceneWorksRevision !== undefined
        && request.repositories.sceneWorks.revision !== expectedSource.sceneWorksRevision)
      || (expectedSource?.inferenceRevision !== undefined
        && request.repositories.inference.revision !== expectedSource.inferenceRevision)
      || (expectedSource?.sceneWorksRepo !== undefined
        && request.repositoryPaths?.sceneWorks !== expectedSource.sceneWorksRepo)
      || (expectedSource?.inferenceRepo !== undefined
        && request.repositoryPaths?.inference !== expectedSource.inferenceRepo)
      || !Number.isSafeInteger(request.hardware?.memoryBytes)
      || request.hardware.memoryBytes <= 0) {
    fail(`${spec.story} provider wrapper received a non-canonical harness request`);
  }
  for (const field of ["_watchdog", "_boundedCampaignEntry", "_measurementSafety"]) {
    if (Object.hasOwn(request.planned, field)) {
      fail(`canonical ${spec.story} request unexpectedly contains ${field}`);
    }
  }
  const transformed = structuredClone(request);
  transformed.action = BOUNDED_CAMPAIGN_ENTRY_ACTION;
  transformed.planned._watchdog = { maxFootprintBytes: MAX_FOOTPRINT_BYTES };
  transformed.planned._measurementSafety = boundedCampaignSafety(spec);
  transformed.planned._boundedCampaignEntry = {
    identity: spec.identity,
    fps: 30,
    seed: 18_946,
    videoMode: "default_av",
    spatialDecodeTiles: 24,
    artifact: {
      repository: ARTIFACT_REPOSITORY,
      revision: ARTIFACT_REVISION,
      numericTierInventory: {
        ...NUMERIC_TIER_INVENTORIES[spec.tier],
      },
      textEncoderInventory: {
        files: TEXT_ENCODER_INVENTORY_FILES,
        bytes: TEXT_ENCODER_INVENTORY_BYTES,
        sha256: TEXT_ENCODER_INVENTORY_SHA256,
      },
    },
  };
  return transformed;
}

function diagnosticMeasurements(response) {
  const measurements = response?.diagnostics?.measurements;
  if (!Array.isArray(measurements)) fail("SC-20191 response omitted typed diagnostics");
  const values = new Map();
  for (const measurement of measurements) {
    if (typeof measurement?.name !== "string" || !Number.isSafeInteger(measurement?.value)
        || measurement.value < 0 || values.has(measurement.name)) {
      fail("SC-20191 response diagnostics are malformed or duplicated");
    }
    values.set(measurement.name, measurement.value);
  }
  return values;
}

export function validateCampaignEntryAdapterResponse(response, {
  inferenceRevision, hostMemoryBytes,
} = {}) {
  const diagnostics = diagnosticMeasurements(response);
  for (const [name, expected] of [
    ["renderedFrames", 121], ["outputFps", 30], ["audioTrackDecoded", 1],
    ["decodeTilingEngaged", 0], ["decodeTileSpatialPx", 0],
    ["decodeTileOverlapPx", 0], ["latentTemporalDepth", 16], ["latentTokens", 6_144],
  ]) {
    if (diagnostics.get(name) !== expected) {
      fail(`SC-20191 response diagnostic ${name} must be ${expected}`);
    }
  }
  const sidecar = response?._campaignEntry;
  if (response?.status !== "runtime_complete"
      || response?.loadShape !== "eager_materialization"
      || response?.artifact?.repository !== ARTIFACT_REPOSITORY
      || response?.artifact?.resolvedRevision !== ARTIFACT_REVISION
      || response?.artifact?.variant !== "q4"
      || !isDeepStrictEqual(response?.strategy, {
        rung: "staged_residency", engagedRungs: ["resident", "staged_residency"], parameters: {},
      })
      || response?.output?.frames !== 121
      || response?.output?.fps !== 30
      || response?.output?.audio?.present !== true
      || !Number.isSafeInteger(response?.output?.audio?.samples)
      || response.output.audio.samples <= 0
      || !Number.isSafeInteger(response?.output?.audio?.sampleRate)
      || response.output.audio.sampleRate <= 0
      || !Number.isSafeInteger(response?.output?.audio?.channels)
      || response.output.audio.channels <= 0
      || response?.output?.firstFrameNondegenerate !== true
      || sidecar?.identity !== CAMPAIGN_ENTRY_IDENTITY
      || (inferenceRevision !== undefined && sidecar?.inferenceRevision !== inferenceRevision)
      || sidecar?.watchdog?.required !== true
      || sidecar?.watchdog?.protocol !== "sceneworks-memory-watchdog-v1"
      || sidecar?.watchdog?.maxFootprintBytes !== MAX_FOOTPRINT_BYTES
      || sidecar?.watchdog?.maxRuntimeSeconds !== MAX_RUNTIME_SECONDS
      || (hostMemoryBytes !== undefined && sidecar?.watchdog?.hostMemoryBytes !== hostMemoryBytes)
      || sidecar?.watchdog?.minInitialMemoryFreeBytes
        !== preflightFreeFloor(sidecar?.watchdog?.hostMemoryBytes)
      || sidecar?.watchdog?.minMemoryFreeBytes
        !== runtimeFreeFloor(sidecar?.watchdog?.hostMemoryBytes)
      || Object.hasOwn(sidecar?.watchdog ?? {}, "minSwapFreeBytes")
      || Object.hasOwn(sidecar?.watchdog ?? {}, "minInitialMemoryFreePercent")
      || sidecar?.mlxLimits?.memoryLimitBytes !== MAX_FOOTPRINT_BYTES
      || !Number.isSafeInteger(sidecar?.mlxLimits?.wiredLimitBytes)
      || sidecar.mlxLimits.wiredLimitBytes <= 0
      || sidecar.mlxLimits.wiredLimitBytes > MAX_FOOTPRINT_BYTES) {
    fail("SC-20191 adapter response changed the exact contained campaign entry");
  }
  const cleanup = sidecar.cleanup;
  for (const field of [
    "preProviderActiveBytes", "preProviderCacheBytes",
    "postCleanupActiveBytes", "postCleanupCacheBytes",
  ]) {
    if (!Number.isSafeInteger(cleanup?.[field]) || cleanup[field] < 0) {
      fail(`SC-20191 cleanup ${field} must be a non-negative safe integer`);
    }
  }
  if (cleanup.preProviderCacheBytes !== 0
      || !isDeepStrictEqual(cleanup.expectedPersistentActive, {
        identity: LTX_ONES_CACHE_IDENTITY,
        videoDimension: LTX_ONES_CACHE_VIDEO_DIMENSION,
        audioDimension: LTX_ONES_CACHE_AUDIO_DIMENSION,
        dtype: "bfloat16",
        bytesPerElement: BFLOAT16_BYTES_PER_ELEMENT,
        bytes: ltxOnesCacheBytes(),
      })) {
    fail("SC-20191 cleanup changed the exact named ONES_CACHE contract");
  }
  const expectedPostActive = cleanup.preProviderActiveBytes + cleanup.expectedPersistentActive.bytes;
  if (!Number.isSafeInteger(expectedPostActive)) fail("SC-20191 cleanup active-byte arithmetic overflowed");
  if (cleanup.postCleanupActiveBytes !== expectedPostActive
      || cleanup.postCleanupCacheBytes !== cleanup.preProviderCacheBytes) {
    fail("SC-20191 cleanup did not return to the exact intentional persistent baseline");
  }
  return response;
}

export function validateBoundedCampaignEntryResponse(response, {
  inferenceRevision, hostMemoryBytes, tier = "q4",
} = {}) {
  const spec = boundedCampaignEntrySpec(tier);
  const diagnostics = diagnosticMeasurements(response);
  for (const [name, expected] of [
    ["renderedFrames", 121], ["outputFps", 30], ["audioTrackDecoded", 1],
    ["decodeTilingEngaged", 1], ["decodeTileSpatialPx", 192],
    ["decodeTileOverlapPx", 64], ["spatialDecodeTiles", 24],
    ["latentTemporalDepth", 16], ["latentTokens", 6_144],
    ["warmOutputAudioChannels", 2], ["providerRequestScopeRenders", 2],
  ]) {
    if (diagnostics.get(name) !== expected) {
      fail(`SC-20318 response diagnostic ${name} must be ${expected}`);
    }
  }
  for (const name of [
    "warmConditioningActivePeak", "warmDenoiseActivePeak", "warmDecodeActivePeak",
    "warmOutputAudioSamples", "warmOutputAudioSampleRate",
  ]) {
    if (!Number.isSafeInteger(diagnostics.get(name)) || diagnostics.get(name) <= 0) {
      fail(`SC-20318 response diagnostic ${name} must be positive`);
    }
  }
  const scenarios = new Map(response?.scenarios?.map((item) => [item.name, item]) ?? []);
  const quality = response?.quality;
  for (const name of [
    "maximumError", "meanError", "rootMeanSquareError",
    "maximumErrorThreshold", "meanErrorThreshold", "rootMeanSquareErrorThreshold",
  ]) {
    if (!Number.isFinite(quality?.[name]) || quality[name] < 0) {
      fail(`SC-20318 quality ${name} is invalid`);
    }
  }
  const sidecar = response?._boundedCampaignEntry;
  if (response?.status !== "runtime_complete"
      || response?.loadShape !== "eager_materialization"
      || response?.artifact?.repository !== ARTIFACT_REPOSITORY
      || response?.artifact?.resolvedRevision !== ARTIFACT_REVISION
      || response?.artifact?.variant !== spec.tier
      || !isDeepStrictEqual(response?.strategy, {
        rung: "bounded_decode",
        engagedRungs: ["resident", "staged_residency", "bounded_decode"],
        parameters: { decodeTileEdge: 192, decodeOverlap: 64 },
      })
      || scenarios.get("warm_repeat")?.result !== "passed"
      || scenarios.get("cancel")?.result !== "not_run"
      || scenarios.get("error")?.result !== "not_run"
      || quality?.result !== "passed" || quality?.identicalInputs !== true
      || quality.maximumError > quality.maximumErrorThreshold
      || quality.meanError > quality.meanErrorThreshold
      || quality.rootMeanSquareError > quality.rootMeanSquareErrorThreshold
      || response?.output?.frames !== 121 || response?.output?.fps !== 30
      || response?.output?.audio?.present !== true
      || !Number.isSafeInteger(response?.output?.audio?.samples)
      || response.output.audio.samples <= 0
      || !Number.isSafeInteger(response?.output?.audio?.sampleRate)
      || response.output.audio.sampleRate <= 0
      || response?.output?.audio?.channels !== 2
      || response?.output?.firstFrameNondegenerate !== true
      || sidecar?.identity !== spec.identity
      || (inferenceRevision !== undefined && sidecar?.inferenceRevision !== inferenceRevision)
      || sidecar?.watchdog?.required !== true
      || sidecar?.watchdog?.protocol !== "sceneworks-memory-watchdog-v1"
      || sidecar?.watchdog?.providerPhaseProtocol !== PROVIDER_PHASE_PROTOCOL
      || sidecar?.watchdog?.providerPhaseProfile !== "bounded-campaign-entry"
      || !isDeepStrictEqual(sidecar?.watchdog?.providerPhases, BOUNDED_CARRIER_PHASES)
      || sidecar?.watchdog?.maxFootprintBytes !== MAX_FOOTPRINT_BYTES
      || sidecar?.watchdog?.maxRuntimeSeconds !== MAX_RUNTIME_SECONDS
      || (hostMemoryBytes !== undefined && sidecar?.watchdog?.hostMemoryBytes !== hostMemoryBytes)
      || sidecar?.watchdog?.minInitialMemoryFreeBytes
        !== preflightFreeFloor(sidecar?.watchdog?.hostMemoryBytes)
      || sidecar?.watchdog?.minMemoryFreeBytes
        !== runtimeFreeFloor(sidecar?.watchdog?.hostMemoryBytes)
      || Object.hasOwn(sidecar?.watchdog ?? {}, "minSwapFreeBytes")
      || sidecar?.mlxLimits?.memoryLimitBytes !== MAX_FOOTPRINT_BYTES
      || !Number.isSafeInteger(sidecar?.mlxLimits?.wiredLimitBytes)
      || sidecar.mlxLimits.wiredLimitBytes <= 0
      || sidecar.mlxLimits.wiredLimitBytes > MAX_FOOTPRINT_BYTES) {
    fail(`${spec.story} adapter response changed the exact bounded campaign entry`);
  }
  const cleanup = sidecar.cleanup;
  for (const field of [
    "preProviderActiveBytes", "preProviderCacheBytes",
    "postCleanupActiveBytes", "postCleanupCacheBytes",
  ]) {
    if (!Number.isSafeInteger(cleanup?.[field]) || cleanup[field] < 0) {
      fail(`SC-20318 cleanup ${field} must be a non-negative safe integer`);
    }
  }
  if (cleanup.preProviderCacheBytes !== 0
      || !isDeepStrictEqual(cleanup.expectedPersistentActive, {
        identity: LTX_ONES_CACHE_IDENTITY,
        videoDimension: LTX_ONES_CACHE_VIDEO_DIMENSION,
        audioDimension: LTX_ONES_CACHE_AUDIO_DIMENSION,
        dtype: "bfloat16", bytesPerElement: BFLOAT16_BYTES_PER_ELEMENT,
        bytes: ltxOnesCacheBytes(),
      })) fail("SC-20318 cleanup changed the named ONES_CACHE contract");
  const expectedPost = cleanup.preProviderActiveBytes + cleanup.expectedPersistentActive.bytes;
  if (!Number.isSafeInteger(expectedPost)
      || cleanup.postCleanupActiveBytes !== expectedPost
      || cleanup.postCleanupCacheBytes !== cleanup.preProviderCacheBytes) {
    fail("SC-20318 cleanup did not return to the exact persistent baseline");
  }
  return response;
}

export function validateBoundedCarrierResponse(response, {
  inferenceRevision, hostMemoryBytes,
} = {}) {
  let canonicallyIngestible = true;
  try {
    validateBundle(response);
  } catch {
    canonicallyIngestible = false;
  }
  if (canonicallyIngestible
      || response?.schemaVersion !== 1
      || response?.recordType !== "sceneworks_bounded_carrier_proof_response_v1"
      || response?.status !== "diagnostic_bounded_carrier_complete"
      || response?.story !== "sc-20254"
      || response?.logicalCaseId !== BOUNDED_CARRIER_LOGICAL_CASE_ID
      || response?.fixture !== BOUNDED_CARRIER_FIXTURE
      || response?.identity !== BOUNDED_CARRIER_IDENTITY
      || response?.diagnosticOnly !== true
      || response?.promotable !== false
      || response?.ingestible !== false
      || (inferenceRevision !== undefined && response?.inferenceRevision !== inferenceRevision)
      || response?.calibrationFingerprint !== FINGERPRINT
      || response?.artifact?.repository !== ARTIFACT_REPOSITORY
      || response?.artifact?.resolvedRevision !== ARTIFACT_REVISION
      || response?.artifact?.variant !== "q4"
      || !isDeepStrictEqual(response?.artifact?.numericTierInventory, {
        files: 11, bytes: Q4_INVENTORY_BYTES, sha256: Q4_INVENTORY_SHA256,
      })
      || !isDeepStrictEqual(response?.artifact?.textEncoderInventory, {
        files: TEXT_ENCODER_INVENTORY_FILES,
        bytes: TEXT_ENCODER_INVENTORY_BYTES,
        sha256: TEXT_ENCODER_INVENTORY_SHA256,
      })
      || !isDeepStrictEqual(response?.target, {
        provider: PROVIDER,
        tier: "q4",
        geometry: { width: 768, height: 512, frames: 121, fps: 30 },
        seed: 18_946,
        videoMode: "default_av",
        audio: true,
      })
      || !isDeepStrictEqual(response?.strategy, {
        rung: "bounded_decode",
        engagedRungs: ["resident", "staged_residency", "bounded_decode"],
        parameters: { decodeTileEdge: 192, decodeOverlap: 64 },
        spatialDecodeTiles: 24,
      })
      || response?.watchdog?.required !== true
      || response?.watchdog?.protocol !== "sceneworks-memory-watchdog-v1"
      || response?.watchdog?.providerPhaseProtocol !== PROVIDER_PHASE_PROTOCOL
      || response?.watchdog?.providerPhaseProfile !== "bounded-carrier"
      || !isDeepStrictEqual(response?.watchdog?.providerPhases, BOUNDED_CARRIER_PHASES)
      || response?.watchdog?.maxFootprintBytes !== MAX_FOOTPRINT_BYTES
      || response?.watchdog?.maxRuntimeSeconds !== MAX_RUNTIME_SECONDS
      || (hostMemoryBytes !== undefined && response?.watchdog?.hostMemoryBytes !== hostMemoryBytes)
      || response?.watchdog?.minInitialMemoryFreeBytes
        !== preflightFreeFloor(response?.watchdog?.hostMemoryBytes)
      || response?.watchdog?.minMemoryFreeBytes
        !== runtimeFreeFloor(response?.watchdog?.hostMemoryBytes)
      || Object.hasOwn(response?.watchdog ?? {}, "minSwapFreeBytes")
      || Object.hasOwn(response?.watchdog ?? {}, "minInitialMemoryFreePercent")
      || response?.mlxLimits?.memoryLimitBytes !== MAX_FOOTPRINT_BYTES
      || !Number.isSafeInteger(response?.mlxLimits?.wiredLimitBytes)
      || response.mlxLimits.wiredLimitBytes <= 0
      || response.mlxLimits.wiredLimitBytes > MAX_FOOTPRINT_BYTES
      || !isDeepStrictEqual(response?.output, {
        frames: 121,
        fps: 30,
        audio: response?.output?.audio,
        frameTimelineSeconds: 4,
        firstFrameNondegenerate: true,
      })
      || response?.output?.audio?.present !== true
      || !Number.isSafeInteger(response?.output?.audio?.samples)
      || response.output.audio.samples <= 0
      || !Number.isSafeInteger(response?.output?.audio?.sampleRate)
      || response.output.audio.sampleRate <= 0
      || !Number.isSafeInteger(response?.output?.audio?.channels)
      || response.output.audio.channels <= 0) {
    fail("SC-20254 response changed the exact non-ingestible bounded carrier");
  }
  const observed = response.observedMemory;
  const phaseActive = [];
  for (const phaseName of ["conditioning", "denoise", "decode"]) {
    const phase = observed?.[phaseName];
    for (const field of ["activeBytes", "allocatorBytes", "reclaimableBytes"]) {
      if (!Number.isSafeInteger(phase?.[field]) || phase[field] < 0) {
        fail(`SC-20254 observedMemory.${phaseName}.${field} is invalid`);
      }
    }
    if (phase.activeBytes <= 0
        || phase.allocatorBytes !== phase.activeBytes + phase.reclaimableBytes) {
      fail(`SC-20254 observedMemory.${phaseName} is not exact`);
    }
    phaseActive.push(phase.activeBytes);
  }
  for (const field of [
    "preProviderActiveBytes", "preProviderCacheBytes",
    "postCleanupActiveBytes", "postCleanupCacheBytes",
  ]) {
    if (!Number.isSafeInteger(observed?.[field]) || observed[field] < 0) {
      fail(`SC-20254 observedMemory.${field} must be a non-negative safe integer`);
    }
  }
  if (observed.peakActiveBytes !== Math.max(...phaseActive)
      || observed.preProviderCacheBytes !== 0
      || !isDeepStrictEqual(observed.expectedPersistentActive, {
        identity: LTX_ONES_CACHE_IDENTITY,
        videoDimension: LTX_ONES_CACHE_VIDEO_DIMENSION,
        audioDimension: LTX_ONES_CACHE_AUDIO_DIMENSION,
        dtype: "bfloat16",
        bytesPerElement: BFLOAT16_BYTES_PER_ELEMENT,
        bytes: ltxOnesCacheBytes(),
      })) {
    fail("SC-20254 allocator phase or named persistent identity changed");
  }
  const expectedPostActive = observed.preProviderActiveBytes
    + observed.expectedPersistentActive.bytes;
  if (!Number.isSafeInteger(expectedPostActive)) {
    fail("SC-20254 cleanup active-byte arithmetic overflowed");
  }
  if (observed.postCleanupActiveBytes !== expectedPostActive
      || observed.postCleanupCacheBytes !== observed.preProviderCacheBytes) {
    fail("SC-20254 cleanup did not return to its exact intentional persistent baseline");
  }
  return response;
}

export function validateBoundedCarrierWatchdogEvents(events, hostMemoryBytes) {
  return validateCampaignEntryWatchdogEvents(
    events, hostMemoryBytes, BOUNDED_CARRIER_PHASES,
  );
}

export function validateCampaignEntryWatchdogEvents(
  events, hostMemoryBytes, expectedPhases = PROVIDER_PHASES,
) {
  if (!Array.isArray(events) || events.length === 0) fail("SC-20191 watchdog stream is empty");
  const startedIndex = events.findIndex((event) => event.event === "started");
  const attestedIndex = events.findIndex((event) => event.event === "child_attested");
  const completedIndex = events.findIndex((event) => event.event === "child_completed");
  const samples = events.filter((event) => event.event === "sample");
  const runtimeFloor = runtimeFreeFloor(hostMemoryBytes);
  if (events.filter((event) => event.event === "started").length !== 1
      || events.filter((event) => event.event === "child_attested").length !== 1
      || events.filter((event) => event.event === "child_completed").length !== 1
      || startedIndex < 0
      || attestedIndex < 0
      || completedIndex < 0
      || startedIndex >= attestedIndex
      || completedIndex <= attestedIndex
      || !events.slice(0, attestedIndex).some((event) =>
        event.event === "sample" && event.phase === "before_child_release")
      || !events.slice(0, attestedIndex).some((event) =>
        event.event === "sample" && event.phase === "child_attested_before_allocation")
      || !events.slice(attestedIndex + 1, completedIndex).some((event) =>
        event.event === "sample")
      || samples.length === 0
      || samples.some((event) =>
        !Number.isSafeInteger(event.physicalFootprintBytes)
        || event.physicalFootprintBytes < 0
        || event.physicalFootprintBytes >= MAX_FOOTPRINT_BYTES
        || !Number.isSafeInteger(event.memoryFreeBytes)
        || event.memoryFreeBytes < runtimeFloor
        || !Number.isSafeInteger(event.swapFreeBytes)
        || event.swapFreeBytes < 0)
      || events.some((event) => event.event === "hard_stop" || event.event === "terminated")) {
    fail("SC-20191 watchdog stream is incomplete or crossed a safety boundary");
  }
  validateProviderPhaseTimeline(events, { complete: true, expectedPhases });
  return Math.max(...samples.map((event) => event.physicalFootprintBytes));
}

function validProcessIdentities(value) {
  return Array.isArray(value) && value.length > 0
    && new Set(value.map((identity) => identity?.pid)).size === value.length
    && value.every((identity) => Number.isSafeInteger(identity?.pid) && identity.pid > 0
      && Number.isSafeInteger(identity?.pgid) && identity.pgid > 0
      && typeof identity?.started === "string" && identity.started.length > 0);
}

function stableCompactJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableCompactJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${stableCompactJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export function validateWatchdogEventChain(events) {
  if (!Array.isArray(events) || events.length === 0) {
    fail("SC-20216 watchdog event chain is empty");
  }
  let previous = "0".repeat(64);
  for (const [index, event] of events.entries()) {
    const { eventHash, ...payload } = event ?? {};
    const expectedHash = createHash("sha256").update(stableCompactJson(payload)).digest("hex");
    if (event?.eventSequence !== index + 1
        || event?.previousEventHash !== previous
        || eventHash !== expectedHash) {
      fail("SC-20216 watchdog event chain is missing, mutated, duplicated, or reordered");
    }
    previous = eventHash;
  }
  return {
    protocol: WATCHDOG_EVENT_CHAIN_PROTOCOL,
    count: events.length,
    head: previous,
  };
}

export function validateProviderPhaseTimeline(events, {
  complete = false, expectedPhases = PROVIDER_PHASES,
} = {}) {
  validateWatchdogEventChain(events);
  let current = null;
  let sequence = 0;
  let previousAt = -Infinity;
  for (const event of events) {
    if (typeof event?.at !== "number" || !Number.isFinite(event.at) || event.at < previousAt) {
      fail("SC-20216 watchdog event timestamps are missing or reordered");
    }
    previousAt = event.at;
    if (event.event === "provider_phase") {
      const expectedSequence = sequence + 1;
      const expectedName = expectedPhases[sequence];
      if (event.authenticated !== true
          || event.providerPhase?.sequence !== expectedSequence
          || event.providerPhase?.name !== expectedName) {
        fail("SC-20216 provider phase telemetry is missing, stale, malformed, or reordered");
      }
      sequence = expectedSequence;
      current = { sequence, name: expectedName };
    } else if ([
      "started", "sample", "child_attested", "child_completed", "hard_stop", "terminated",
    ].includes(event.event) && (!Object.hasOwn(event, "providerPhase")
      || !isDeepStrictEqual(event.providerPhase, current))) {
      fail("SC-20216 watchdog event is not bound to the latest authenticated provider phase");
    }
  }
  if (sequence === 0 || (complete && sequence !== expectedPhases.length)) {
    fail("SC-20216 watchdog stream omitted required authenticated provider phases");
  }
  return current;
}

export function validateCampaignEntryFailureEvents(
  events, hostMemoryBytes, expectedPhases = PROVIDER_PHASES,
) {
  if (!Array.isArray(events) || events.length === 0) fail("SC-20216 failure stream is empty");
  validateProviderPhaseTimeline(events, { expectedPhases });
  const eventChain = validateWatchdogEventChain(events);
  const hardStops = events.filter((event) => event.event === "hard_stop");
  const terminated = events.filter((event) => event.event === "terminated");
  const started = events.filter((event) => event.event === "started");
  const hardStopIndex = events.findIndex((event) => event.event === "hard_stop");
  const terminatedIndex = events.findIndex((event) => event.event === "terminated");
  const attestedIndex = events.findIndex((event) => event.event === "child_attested");
  const allowedEvents = new Set([
    "started", "sample", "child_attested", "provider_phase", "hard_stop", "terminated",
  ]);
  const samples = events.filter((event) => event.event === "sample");
  if (started.length !== 1 || hardStops.length !== 1 || terminated.length !== 1
      || events[0]?.event !== "started"
      || hardStopIndex < 0 || terminatedIndex !== events.length - 1
      || terminatedIndex !== hardStopIndex + 1
      || events.filter((event) => event.event === "child_attested").length !== 1
      || attestedIndex <= 0 || attestedIndex >= hardStopIndex
      || events.some((event) => event.event === "child_completed")
      || events.some((event) => !allowedEvents.has(event.event))
      || !events.slice(0, attestedIndex).some((event) =>
        event.event === "sample" && event.phase === "before_child_release")
      || !events.slice(0, attestedIndex).some((event) =>
        event.event === "sample" && event.phase === "child_attested_before_allocation")
      || !events.slice(attestedIndex + 1, hardStopIndex).some((event) =>
        event.event === "provider_phase")
      || typeof hardStops[0].reason !== "string" || hardStops[0].reason.length === 0
      || terminated[0].reason !== hardStops[0].reason
      || !validProcessIdentities(started[0].processIdentities)
      || !validProcessIdentities(hardStops[0].processIdentities)
      || !validProcessIdentities(terminated[0].processIdentities)
      || !isDeepStrictEqual(hardStops[0].processIdentities, terminated[0].processIdentities)
      || started[0].processIdentities.every((identity) => identity.pid !== started[0].pid)
      || [hardStops[0], terminated[0]].some((event) =>
        started[0].processIdentities.some((startedIdentity) =>
          !event.processIdentities.some((identity) => isDeepStrictEqual(identity, startedIdentity))))
      || [started[0], hardStops[0], terminated[0]].some((event) =>
        event.processIdentities.some((identity) => identity.pgid !== started[0].pgid))) {
    fail("SC-20216 failure stream is incomplete or lacks exact process termination identity");
  }
  const runtimeFloor = runtimeFreeFloor(hostMemoryBytes);
  if (samples.some((event) => !Number.isSafeInteger(event.physicalFootprintBytes)
      || event.physicalFootprintBytes < 0
      || !Number.isSafeInteger(event.memoryFreeBytes)
      || event.memoryFreeBytes < 0
      || !Number.isSafeInteger(event.swapFreeBytes)
      || event.swapFreeBytes < 0)) {
    fail("SC-20216 failure stream contains malformed watchdog telemetry");
  }
  const violating = events.findIndex((event) => event.event === "sample"
    && (event.physicalFootprintBytes >= MAX_FOOTPRINT_BYTES
      || event.memoryFreeBytes < runtimeFloor));
  const thresholdReason = hardStops[0].reason.startsWith("physical_footprint_at_or_above_")
    || hardStops[0].reason.startsWith("host_memory_free_below_");
  if (violating >= 0) {
    const sample = events[violating];
    const expectedReason = sample.physicalFootprintBytes >= MAX_FOOTPRINT_BYTES
      ? `physical_footprint_at_or_above_${MAX_FOOTPRINT_BYTES}:observed_${sample.physicalFootprintBytes}`
      : `host_memory_free_below_${runtimeFloor}:observed_${sample.memoryFreeBytes}`;
    if (violating !== hardStopIndex - 1 || hardStops[0].reason !== expectedReason
        || events.slice(0, violating).some((event) => event.event === "sample"
          && (event.physicalFootprintBytes >= MAX_FOOTPRINT_BYTES
            || event.memoryFreeBytes < runtimeFloor))) {
      fail("SC-20216 failure stream does not bind the first exact violating sample and reason");
    }
  } else if (thresholdReason) {
    fail("SC-20216 threshold hard stop omitted its exact first violating sample");
  }
  return {
    reason: hardStops[0].reason,
    firstViolatingEventIndex: violating >= 0 ? violating : null,
    firstViolatingSample: violating >= 0 ? structuredClone(events[violating]) : null,
    terminalProviderPhase: structuredClone(hardStops[0].providerPhase),
    started: structuredClone(started[0]),
    eventChain,
  };
}

export function campaignEntryCanonicalFragment(response, maxObservedFootprintBytes) {
  if (!Number.isSafeInteger(maxObservedFootprintBytes)
      || maxObservedFootprintBytes < 0
      || maxObservedFootprintBytes >= MAX_FOOTPRINT_BYTES) {
    fail("SC-20191 canonical fragment requires a contained physical-footprint maximum");
  }
  const canonical = structuredClone(response);
  const sidecar = canonical._campaignEntry;
  const output = canonical.output;
  if (!sidecar || !output) fail("SC-20191 canonical fragment omitted private validated evidence");
  delete canonical._campaignEntry;
  delete canonical.output;
  const measurements = canonical.diagnostics?.measurements;
  if (!Array.isArray(measurements)) fail("SC-20191 canonical fragment omitted diagnostics");
  const existing = new Set(measurements.map((measurement) => measurement.name));
  const append = (name, unit, value) => {
    if (existing.has(name) || !Number.isSafeInteger(value) || value < 0) {
      fail(`SC-20191 canonical diagnostic ${name} is duplicated or invalid`);
    }
    existing.add(name);
    measurements.push({ name, unit, value });
  };
  append("campaignWatchdogMaxFootprintBytes", "bytes", sidecar.watchdog.maxFootprintBytes);
  append("campaignWatchdogMaxObservedFootprintBytes", "bytes", maxObservedFootprintBytes);
  append("campaignWatchdogMaxRuntimeSeconds", "seconds", sidecar.watchdog.maxRuntimeSeconds);
  append("campaignWatchdogHostMemoryBytes", "bytes", sidecar.watchdog.hostMemoryBytes);
  append("campaignWatchdogMinInitialMemoryFreeBytes", "bytes",
    sidecar.watchdog.minInitialMemoryFreeBytes);
  append("campaignWatchdogMinMemoryFreeBytes", "bytes", sidecar.watchdog.minMemoryFreeBytes);
  append("campaignMlxMemoryLimitBytes", "bytes", sidecar.mlxLimits.memoryLimitBytes);
  append("campaignMlxWiredLimitBytes", "bytes", sidecar.mlxLimits.wiredLimitBytes);
  append("campaignPreProviderActiveBytes", "bytes", sidecar.cleanup.preProviderActiveBytes);
  append("campaignPreProviderCacheBytes", "bytes", sidecar.cleanup.preProviderCacheBytes);
  append("campaignExpectedPersistentActiveBytes", "bytes",
    sidecar.cleanup.expectedPersistentActive.bytes);
  append("campaignPostCleanupActiveBytes", "bytes", sidecar.cleanup.postCleanupActiveBytes);
  append("campaignPostCleanupCacheBytes", "bytes", sidecar.cleanup.postCleanupCacheBytes);
  append("campaignOutputAudioSamples", "count", output.audio.samples);
  append("campaignOutputAudioSampleRate", "hertz", output.audio.sampleRate);
  append("campaignOutputAudioChannels", "count", output.audio.channels);
  append("campaignFirstFrameNondegenerate", "boolean", output.firstFrameNondegenerate ? 1 : 0);
  append("campaignOwnedProcessGroupResidue", "count", 0);
  append("campaignProviderContainmentComplete", "boolean", 1);
  return canonical;
}

export function boundedCampaignEntryCanonicalFragment(response, maxObservedFootprintBytes) {
  if (!Number.isSafeInteger(maxObservedFootprintBytes)
      || maxObservedFootprintBytes < 0
      || maxObservedFootprintBytes >= MAX_FOOTPRINT_BYTES) {
    fail("SC-20318 canonical fragment requires a contained physical-footprint maximum");
  }
  const canonical = structuredClone(response);
  const sidecar = canonical._boundedCampaignEntry;
  const output = canonical.output;
  if (!sidecar || !output) fail("SC-20318 canonical fragment omitted private evidence");
  delete canonical._boundedCampaignEntry;
  delete canonical.output;
  const measurements = canonical.diagnostics?.measurements;
  if (!Array.isArray(measurements)) fail("SC-20318 canonical fragment omitted diagnostics");
  const existing = new Set(measurements.map((measurement) => measurement.name));
  const append = (name, unit, value) => {
    if (existing.has(name) || !Number.isSafeInteger(value) || value < 0) {
      fail(`SC-20318 canonical diagnostic ${name} is duplicated or invalid`);
    }
    existing.add(name);
    measurements.push({ name, unit, value });
  };
  append("boundedCampaignWatchdogMaxFootprintBytes", "bytes", sidecar.watchdog.maxFootprintBytes);
  append("boundedCampaignWatchdogMaxObservedFootprintBytes", "bytes", maxObservedFootprintBytes);
  append("boundedCampaignWatchdogMaxRuntimeSeconds", "seconds", sidecar.watchdog.maxRuntimeSeconds);
  append("boundedCampaignWatchdogHostMemoryBytes", "bytes", sidecar.watchdog.hostMemoryBytes);
  append("boundedCampaignWatchdogMinInitialMemoryFreeBytes", "bytes",
    sidecar.watchdog.minInitialMemoryFreeBytes);
  append("boundedCampaignWatchdogMinMemoryFreeBytes", "bytes",
    sidecar.watchdog.minMemoryFreeBytes);
  append("boundedCampaignMlxMemoryLimitBytes", "bytes", sidecar.mlxLimits.memoryLimitBytes);
  append("boundedCampaignMlxWiredLimitBytes", "bytes", sidecar.mlxLimits.wiredLimitBytes);
  append("boundedCampaignPreProviderActiveBytes", "bytes", sidecar.cleanup.preProviderActiveBytes);
  append("boundedCampaignPreProviderCacheBytes", "bytes", sidecar.cleanup.preProviderCacheBytes);
  append("boundedCampaignExpectedPersistentActiveBytes", "bytes",
    sidecar.cleanup.expectedPersistentActive.bytes);
  append("boundedCampaignPostCleanupActiveBytes", "bytes", sidecar.cleanup.postCleanupActiveBytes);
  append("boundedCampaignPostCleanupCacheBytes", "bytes", sidecar.cleanup.postCleanupCacheBytes);
  append("boundedCampaignOutputAudioSamples", "count", output.audio.samples);
  append("boundedCampaignOutputAudioSampleRate", "hertz", output.audio.sampleRate);
  append("boundedCampaignOutputAudioChannels", "count", output.audio.channels);
  append("boundedCampaignFirstFrameNondegenerate", "boolean",
    output.firstFrameNondegenerate ? 1 : 0);
  append("boundedCampaignOwnedProcessGroupResidue", "count", 0);
  append("boundedCampaignContainmentComplete", "boolean", 1);
  return canonical;
}

export function validateCampaignEntryBundle(bundle) {
  validateBundle(bundle);
  if (bundle.records.length !== 1 || (bundle.sourceSessions?.length ?? 0) !== 0) {
    fail("SC-20191 must publish exactly one canonical record and no synthetic source session");
  }
  const record = bundle.records[0];
  if (record.logicalCaseId !== CAMPAIGN_ENTRY_LOGICAL_CASE_ID
      || record.status !== "runtime_complete"
      || record.evidenceScope !== "authoritative"
      || record.fixture !== CAMPAIGN_ENTRY_FIXTURE
      || !isDeepStrictEqual(record.target, {
        provider: PROVIDER, modelId: PROVIDER, tier: "q4", mode: "text_to_video",
        overlay: "none", geometry: { width: 768, height: 512, batch: 1, frames: 121 },
      })
      || !isDeepStrictEqual(record.strategy, {
        rung: "staged_residency", engagedRungs: ["resident", "staged_residency"], parameters: {},
      })) {
    fail("SC-20191 canonical bundle changed identity");
  }
  const diagnostics = diagnosticMeasurements(record);
  for (const [name, expected] of [
    ["renderedFrames", 121], ["outputFps", 30], ["audioTrackDecoded", 1],
    ["decodeTilingEngaged", 0], ["decodeTileSpatialPx", 0],
    ["decodeTileOverlapPx", 0], ["latentTemporalDepth", 16], ["latentTokens", 6_144],
    ["campaignWatchdogMaxFootprintBytes", MAX_FOOTPRINT_BYTES],
    ["campaignWatchdogMaxRuntimeSeconds", MAX_RUNTIME_SECONDS],
    ["campaignMlxMemoryLimitBytes", MAX_FOOTPRINT_BYTES],
    ["campaignPreProviderCacheBytes", 0],
    ["campaignExpectedPersistentActiveBytes", ltxOnesCacheBytes()],
    ["campaignPostCleanupCacheBytes", 0],
    ["campaignFirstFrameNondegenerate", 1],
    ["campaignOwnedProcessGroupResidue", 0],
    ["campaignProviderContainmentComplete", 1],
  ]) {
    if (diagnostics.get(name) !== expected) {
      fail(`SC-20191 canonical bundle diagnostic ${name} must be ${expected}`);
    }
  }
  const pre = diagnostics.get("campaignPreProviderActiveBytes");
  const post = diagnostics.get("campaignPostCleanupActiveBytes");
  const observed = diagnostics.get("campaignWatchdogMaxObservedFootprintBytes");
  const hostMemory = diagnostics.get("campaignWatchdogHostMemoryBytes");
  for (const name of [
    "campaignOutputAudioSamples", "campaignOutputAudioSampleRate", "campaignOutputAudioChannels",
    "campaignMlxWiredLimitBytes", "campaignWatchdogHostMemoryBytes",
    "campaignWatchdogMinInitialMemoryFreeBytes", "campaignWatchdogMinMemoryFreeBytes",
  ]) {
    if (!Number.isSafeInteger(diagnostics.get(name)) || diagnostics.get(name) <= 0) {
      fail(`SC-20191 canonical bundle diagnostic ${name} must be positive`);
    }
  }
  if (!Number.isSafeInteger(pre) || !Number.isSafeInteger(post)
      || post !== pre + ltxOnesCacheBytes()
      || hostMemory !== record.hardware.memoryBytes
      || diagnostics.get("campaignWatchdogMinInitialMemoryFreeBytes")
        !== preflightFreeFloor(hostMemory)
      || diagnostics.get("campaignWatchdogMinMemoryFreeBytes") !== runtimeFreeFloor(hostMemory)
      || diagnostics.get("campaignMlxWiredLimitBytes") > MAX_FOOTPRINT_BYTES
      || !Number.isSafeInteger(observed) || observed < 0 || observed >= MAX_FOOTPRINT_BYTES) {
    fail("SC-20191 canonical bundle cleanup or footprint attestation changed");
  }
  return bundle;
}

export function validateBoundedCampaignEntryBundle(bundle, requested = "q4") {
  const spec = boundedCampaignEntrySpec(requested);
  validateBundle(bundle);
  if (bundle.records.length !== 1 || (bundle.sourceSessions?.length ?? 0) !== 0) {
    fail("SC-20318 must publish exactly one canonical record and no source session");
  }
  const record = bundle.records[0];
  const scenarios = new Map(record?.scenarios?.map((item) => [item.name, item]) ?? []);
  if (record.logicalCaseId !== spec.logicalCaseId
      || record.status !== "runtime_complete"
      || record.evidenceScope !== "authoritative"
      || record.fixture !== spec.fixture
      || record.target?.tier !== spec.tier
      || !isDeepStrictEqual(record.strategy, {
        rung: "bounded_decode",
        engagedRungs: ["resident", "staged_residency", "bounded_decode"],
        parameters: { decodeTileEdge: 192, decodeOverlap: 64 },
      })
      || scenarios.get("warm_repeat")?.result !== "passed"
      || scenarios.get("cancel")?.result !== "not_run"
      || scenarios.get("error")?.result !== "not_run") {
    fail("SC-20318 canonical bundle changed identity or parity-only lifecycle");
  }
  const diagnostics = diagnosticMeasurements(record);
  for (const [name, expected] of [
    ["renderedFrames", 121], ["outputFps", 30], ["audioTrackDecoded", 1],
    ["decodeTilingEngaged", 1], ["decodeTileSpatialPx", 192],
    ["decodeTileOverlapPx", 64], ["spatialDecodeTiles", 24],
    ["providerRequestScopeRenders", 2], ["warmOutputAudioChannels", 2],
    ["boundedCampaignWatchdogMaxFootprintBytes", MAX_FOOTPRINT_BYTES],
    ["boundedCampaignWatchdogMaxRuntimeSeconds", MAX_RUNTIME_SECONDS],
    ["boundedCampaignMlxMemoryLimitBytes", MAX_FOOTPRINT_BYTES],
    ["boundedCampaignPreProviderCacheBytes", 0],
    ["boundedCampaignExpectedPersistentActiveBytes", ltxOnesCacheBytes()],
    ["boundedCampaignPostCleanupCacheBytes", 0],
    ["boundedCampaignOutputAudioChannels", 2],
    ["boundedCampaignFirstFrameNondegenerate", 1],
    ["boundedCampaignOwnedProcessGroupResidue", 0],
    ["boundedCampaignContainmentComplete", 1],
  ]) {
    if (diagnostics.get(name) !== expected) {
      fail(`SC-20318 canonical bundle diagnostic ${name} must be ${expected}`);
    }
  }
  const pre = diagnostics.get("boundedCampaignPreProviderActiveBytes");
  const post = diagnostics.get("boundedCampaignPostCleanupActiveBytes");
  const observed = diagnostics.get("boundedCampaignWatchdogMaxObservedFootprintBytes");
  const host = diagnostics.get("boundedCampaignWatchdogHostMemoryBytes");
  const phaseAnchors = [
    record.observedMemory?.conditioning?.activeBytes,
    record.observedMemory?.denoise?.activeBytes,
    record.observedMemory?.decode?.activeBytes,
    diagnostics.get("warmConditioningActivePeak"),
    diagnostics.get("warmDenoiseActivePeak"),
    diagnostics.get("warmDecodeActivePeak"),
  ];
  if (phaseAnchors.some((value) => !Number.isSafeInteger(value) || value <= 0)
      || !Number.isSafeInteger(pre) || post !== pre + ltxOnesCacheBytes()
      || host !== record.hardware.memoryBytes
      || diagnostics.get("boundedCampaignWatchdogMinInitialMemoryFreeBytes")
        !== preflightFreeFloor(host)
      || diagnostics.get("boundedCampaignWatchdogMinMemoryFreeBytes") !== runtimeFreeFloor(host)
      || diagnostics.get("boundedCampaignMlxWiredLimitBytes") > MAX_FOOTPRINT_BYTES
      || !Number.isSafeInteger(observed) || observed < 0 || observed >= MAX_FOOTPRINT_BYTES) {
    fail("SC-20318 canonical bundle cleanup or footprint attestation changed");
  }
  return bundle;
}

export function boundedSelectorAnchorReport(bundlesByTier) {
  const phaseNames = ["conditioning", "denoise", "decode"];
  let matchBasis = null;
  const anchors = Object.keys(BOUNDED_CAMPAIGN_ENTRY_SPECS).map((tier) => {
    const bundle = bundlesByTier?.[tier];
    validateBoundedCampaignEntryBundle(bundle, tier);
    const record = bundle.records[0];
    const diagnostics = diagnosticMeasurements(record);
    const recordMatchBasis = {
      hardware: Object.fromEntries([
        "memoryBytes", "model", "chip", "osVersion", "metalDevice",
        "mlxMemoryLimitBytes", "wiredLimitBytes",
      ].map((name) => [name, record.hardware[name]])),
      inference: {
        revision: record.repositories.inference.revision,
        closureDigest: record.repositories.inference.closureDigest,
      },
      calibrationFingerprint: record.calibrationFingerprint,
      artifact: {
        repository: record.artifact.repository,
        resolvedRevision: record.artifact.resolvedRevision,
      },
      geometry: structuredClone(record.target.geometry),
      strategy: structuredClone(record.strategy),
    };
    if (matchBasis === null) matchBasis = recordMatchBasis;
    else if (!isDeepStrictEqual(recordMatchBasis, matchBasis)) {
      fail(`SC-20430 ${tier} capture does not match the bounded-selector comparison basis`);
    }
    const selected = Object.fromEntries(phaseNames.map((phase) => {
      const value = record.observedMemory?.[phase]?.activeBytes;
      if (!Number.isSafeInteger(value) || value <= 0) {
        fail(`SC-20430 ${tier} selected ${phase} phase anchor must be positive`);
      }
      return [phase, value];
    }));
    const warm = Object.fromEntries(phaseNames.map((phase) => {
      const diagnostic = `warm${phase[0].toUpperCase()}${phase.slice(1)}ActivePeak`;
      const value = diagnostics.get(diagnostic);
      if (!Number.isSafeInteger(value) || value <= 0) {
        fail(`SC-20430 ${tier} warm ${phase} phase anchor must be positive`);
      }
      return [phase, value];
    }));
    const binding = (values) => {
      const maximum = Math.max(...Object.values(values));
      return {
        phases: phaseNames.filter((phase) => values[phase] === maximum),
        activePeakBytes: maximum,
      };
    };
    return {
      tier,
      logicalCaseId: record.logicalCaseId,
      recordId: record.id,
      fixture: record.fixture,
      source: {
        sceneWorksRevision: record.repositories.sceneWorks.revision,
        inferenceRevision: record.repositories.inference.revision,
        inferenceClosureDigest: record.repositories.inference.closureDigest,
      },
      selected: { activePeakBytesByPhase: selected, binding: binding(selected) },
      warm: { activePeakBytesByPhase: warm, binding: binding(warm) },
    };
  });
  const q8Source = anchors.find(({ tier }) => tier === "q8").source;
  const bf16Source = anchors.find(({ tier }) => tier === "bf16").source;
  if (q8Source.sceneWorksRevision !== bf16Source.sceneWorksRevision
      || q8Source.inferenceRevision !== bf16Source.inferenceRevision
      || q8Source.inferenceClosureDigest !== bf16Source.inferenceClosureDigest) {
    fail("SC-20430 q8 and bf16 captures must share the exact ratification source");
  }
  const report = {
    schemaVersion: 1,
    recordType: "sceneworks_bounded_selector_anchor_v1",
    status: "diagnostic_anchor_complete",
    diagnosticOnly: true,
    promotable: false,
    ingestible: false,
    scope: "matched_q4_q8_bf16_bounded_selector",
    coefficientTransferClaim: false,
    pooledWithStagedOrSinglePass: false,
    sourcePolicy: {
      q4MayPrecedeRatificationSource: true,
      q8AndBf16SceneWorksRevision: q8Source.sceneWorksRevision,
    },
    matchBasis,
    anchors,
  };
  try {
    validateBundle(report);
    fail("SC-20430 bounded-selector anchor report became canonically ingestible");
  } catch (error) {
    if (String(error?.message ?? error).includes("became canonically ingestible")) throw error;
  }
  return report;
}

function exactFileIdentity(file, stats) {
  return {
    path: file,
    device: stats.dev.toString(),
    inode: stats.ino.toString(),
    size: Number(stats.size),
    mtimeNs: stats.mtimeNs.toString(),
    ctimeNs: stats.ctimeNs.toString(),
    mode: Number(stats.mode & 0o777n),
  };
}

export async function validateQ8IndependentAudit(auditPath, {
  sceneWorksRevision, inferenceRevision,
} = {}) {
  const absoluteAuditPath = path.resolve(auditPath);
  const handle = await open(
    absoluteAuditPath, fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW,
  ).catch((error) => fail(`SC-20430 q8 audit must be an exact sealed file: ${error.message}`));
  let auditBytes;
  let auditFile;
  try {
    const before = await handle.stat({ bigint: true });
    if (!before.isFile() || Number(before.mode & 0o777n) !== 0o400 || before.size <= 0n
        || before.size > BigInt(Number.MAX_SAFE_INTEGER)) {
      fail("SC-20430 q8 audit must be a non-empty read-only regular file");
    }
    auditBytes = await handle.readFile();
    const after = await handle.stat({ bigint: true });
    if (!isDeepStrictEqual(exactFileIdentity(absoluteAuditPath, before),
      exactFileIdentity(absoluteAuditPath, after))) {
      fail("SC-20430 q8 audit changed while it was read");
    }
    auditFile = exactFileIdentity(absoluteAuditPath, after);
  } finally {
    await handle.close();
  }
  const audit = JSON.parse(auditBytes.toString("utf8"));
  const canonicalOutput = audit?.q8?.canonicalOutput;
  if (audit?.schemaVersion !== 1
      || audit?.recordType !== "sceneworks_sc20430_q8_independent_audit_v1"
      || audit?.result !== "passed"
      || audit?.independent !== true
      || audit?.diagnosticOnly !== true
      || audit?.ingestible !== false
      || !path.isAbsolute(canonicalOutput ?? "")
      || !/^[0-9a-f]{64}$/.test(audit?.q8?.canonicalSha256 ?? "")
      || audit?.q8?.logicalCaseId !== BOUNDED_CAMPAIGN_ENTRY_SPECS.q8.logicalCaseId
      || !/^imc-[0-9a-f]{20}$/.test(audit?.q8?.recordId ?? "")
      || !/^[0-9a-f]{40}$/.test(audit?.q8?.sceneWorksRevision ?? "")
      || !/^[0-9a-f]{40}$/.test(audit?.q8?.inferenceRevision ?? "")) {
    fail("SC-20430 bf16 release requires an exact passed independent q8 audit");
  }
  const bundleBytes = await readFile(canonicalOutput, "utf8");
  const bundle = JSON.parse(bundleBytes);
  validateBoundedCampaignEntryBundle(bundle, "q8");
  const digest = createHash("sha256").update(canonicalJson(bundle)).digest("hex");
  if (digest !== audit.q8.canonicalSha256
      || bundle.records[0].id !== audit.q8.recordId
      || bundle.records[0].repositories?.sceneWorks?.revision !== audit.q8.sceneWorksRevision
      || bundle.records[0].repositories?.inference?.revision !== audit.q8.inferenceRevision
      || (sceneWorksRevision !== undefined && audit.q8.sceneWorksRevision !== sceneWorksRevision)
      || (inferenceRevision !== undefined && audit.q8.inferenceRevision !== inferenceRevision)) {
    fail("SC-20430 q8 independent audit does not bind the exact canonical bundle");
  }
  return {
    schemaVersion: 1,
    recordType: "sceneworks_sc20430_q8_release_authorization_v1",
    auditFile: { ...auditFile, sha256: createHash("sha256").update(auditBytes).digest("hex") },
    q8: structuredClone(audit.q8),
  };
}

async function assertQ8ReleaseAuthorization(expected, auditPath, revisions) {
  if (expected === null) return null;
  const current = await validateQ8IndependentAudit(auditPath, revisions);
  if (!isDeepStrictEqual(current, expected)) {
    fail("SC-20430 q8 independent-audit authorization changed after release approval");
  }
  return current;
}

export async function publishBoundedSelectorAnchorReport({
  q4Bundle, q8Bundle, bf16Bundle, output, signal,
}) {
  const report = boundedSelectorAnchorReport({
    q4: JSON.parse(await readFile(path.resolve(q4Bundle), "utf8")),
    q8: JSON.parse(await readFile(path.resolve(q8Bundle), "utf8")),
    bf16: JSON.parse(await readFile(path.resolve(bf16Bundle), "utf8")),
  });
  await publishExclusiveJson(path.resolve(output), report, signal);
  return report;
}

export function campaignEntryFailurePath(canonicalOutput) {
  return path.join(path.dirname(canonicalOutput), "sc-20216-campaign-entry-failure.json");
}

export function campaignEntryOutcomeReservationPath(canonicalOutput) {
  return path.join(path.dirname(canonicalOutput), ".sc-20216-campaign-entry-outcome");
}

export function boundedCampaignEntryFailurePath(canonicalOutput, requested = "q4") {
  const spec = boundedCampaignEntrySpec(requested);
  const filename = spec.tier === "q4"
    ? "sc-20318-bounded-campaign-entry-failure.json"
    : `sc-20430-${spec.tier}-bounded-campaign-entry-failure.json`;
  return path.join(path.dirname(canonicalOutput), filename);
}

export function boundedCampaignEntryOutcomeReservationPath(canonicalOutput, requested = "q4") {
  const spec = boundedCampaignEntrySpec(requested);
  const filename = spec.tier === "q4"
    ? ".sc-20318-bounded-campaign-entry-outcome"
    : `.sc-20430-${spec.tier}-bounded-campaign-entry-outcome`;
  return path.join(path.dirname(canonicalOutput), filename);
}

export function campaignEntryFailureReceipt({
  sceneWorksRevision, sceneWorksTree, inferenceRevision, inferenceTree,
  identity, preparationKey, preparationRoot, prepared, hostMemoryBytes, events, outcome,
}) {
  const failure = validateCampaignEntryFailureEvents(events, hostMemoryBytes);
  const receipt = {
    schemaVersion: 1,
    recordType: CAMPAIGN_FAILURE_RECEIPT_TYPE,
    status: "diagnostic_failure",
    diagnosticOnly: true,
    promotable: false,
    ingestible: false,
    canonicalBundlePublished: false,
    outcome: structuredClone(outcome),
    story: "sc-20216",
    source: {
      sceneWorks: { revision: sceneWorksRevision, tree: sceneWorksTree },
      inference: { revision: inferenceRevision, tree: inferenceTree },
    },
    campaignCase: {
      plan: CAMPAIGN_ENTRY_PLAN,
      provider: CAMPAIGN_ENTRY_PROVIDER,
      logicalCaseId: CAMPAIGN_ENTRY_LOGICAL_CASE_ID,
      fixture: CAMPAIGN_ENTRY_FIXTURE,
      identity: CAMPAIGN_ENTRY_IDENTITY,
      action: CAMPAIGN_ENTRY_ACTION,
      target: {
        provider: PROVIDER, tier: "q4",
        geometry: { width: 768, height: 512, batch: 1, frames: 121, fps: 30 },
      },
      strategy: {
        rung: "staged_residency", engagedRungs: ["resident", "staged_residency"], parameters: {},
      },
    },
    artifacts: {
      repository: ARTIFACT_REPOSITORY,
      revision: ARTIFACT_REVISION,
      numericTierInventory: structuredClone(identity.artifact.numericTier),
      textEncoderInventory: structuredClone(identity.artifact.textEncoder),
      adapter: structuredClone(prepared.adapter),
      metallib: structuredClone(prepared.metallib),
      preparation: {
        key: preparationKey,
        root: preparationRoot,
        identity: structuredClone(identity),
        manifest: structuredClone(prepared.manifest),
      },
    },
    watchdog: {
      protocol: "sceneworks-memory-watchdog-v1",
      providerPhaseProtocol: PROVIDER_PHASE_PROTOCOL,
      providerPhases: [...PROVIDER_PHASES],
      maxFootprintBytes: MAX_FOOTPRINT_BYTES,
      maxRuntimeSeconds: MAX_RUNTIME_SECONDS,
      hostMemoryBytes,
      minInitialMemoryFreeBytes: preflightFreeFloor(hostMemoryBytes),
      minMemoryFreeBytes: runtimeFreeFloor(hostMemoryBytes),
      failure,
      eventChain: structuredClone(failure.eventChain),
      events: structuredClone(events),
    },
    cleanup: {
      ownedProcessGroupResidueVerified: true,
      runScratchRemoved: true,
      runRootEmpty: true,
      sourceIdentityVerified: true,
      runtimeAssetIdentityVerified: true,
      preparedCacheVerified: true,
      preparationTransientResidueVerified: true,
    },
  };
  return validateCampaignEntryFailureReceipt(receipt);
}

export function boundedCampaignEntryFailureReceipt({
  sceneWorksRevision, sceneWorksTree, inferenceRevision, inferenceTree,
  identity, preparationKey, preparationRoot, prepared, hostMemoryBytes, events, outcome,
  tier = "q4",
}) {
  const spec = boundedCampaignEntrySpec(tier);
  const failure = validateCampaignEntryFailureEvents(
    events, hostMemoryBytes, BOUNDED_CARRIER_PHASES,
  );
  const receipt = {
    schemaVersion: 1,
    recordType: BOUNDED_CAMPAIGN_FAILURE_RECEIPT_TYPE,
    status: "diagnostic_failure",
    diagnosticOnly: true,
    promotable: false,
    ingestible: false,
    canonicalBundlePublished: false,
    outcome: structuredClone(outcome),
    story: spec.story,
    source: {
      sceneWorks: { revision: sceneWorksRevision, tree: sceneWorksTree },
      inference: { revision: inferenceRevision, tree: inferenceTree },
    },
    campaignCase: {
      plan: BOUNDED_CAMPAIGN_ENTRY_PLAN,
      provider: spec.provider,
      logicalCaseId: spec.logicalCaseId,
      fixture: spec.fixture,
      identity: spec.identity,
      action: BOUNDED_CAMPAIGN_ENTRY_ACTION,
      target: {
        provider: PROVIDER, tier: spec.tier,
        geometry: { width: 768, height: 512, batch: 1, frames: 121, fps: 30 },
      },
      strategy: {
        rung: "bounded_decode",
        engagedRungs: ["resident", "staged_residency", "bounded_decode"],
        parameters: { decodeTileEdge: 192, decodeOverlap: 64 },
        spatialDecodeTiles: 24,
      },
    },
    artifacts: {
      repository: ARTIFACT_REPOSITORY,
      revision: ARTIFACT_REVISION,
      numericTierInventory: structuredClone(identity.artifact.numericTier),
      textEncoderInventory: structuredClone(identity.artifact.textEncoder),
      adapter: structuredClone(prepared.adapter),
      metallib: structuredClone(prepared.metallib),
      preparation: {
        key: preparationKey, root: preparationRoot,
        identity: structuredClone(identity), manifest: structuredClone(prepared.manifest),
      },
    },
    watchdog: {
      protocol: "sceneworks-memory-watchdog-v1",
      providerPhaseProtocol: PROVIDER_PHASE_PROTOCOL,
      providerPhases: [...BOUNDED_CARRIER_PHASES],
      maxFootprintBytes: MAX_FOOTPRINT_BYTES,
      maxRuntimeSeconds: MAX_RUNTIME_SECONDS,
      hostMemoryBytes,
      minInitialMemoryFreeBytes: preflightFreeFloor(hostMemoryBytes),
      minMemoryFreeBytes: runtimeFreeFloor(hostMemoryBytes),
      failure,
      eventChain: structuredClone(failure.eventChain),
      events: structuredClone(events),
    },
    cleanup: {
      ownedProcessGroupResidueVerified: true,
      runScratchRemoved: true,
      runRootEmpty: true,
      sourceIdentityVerified: true,
      runtimeAssetIdentityVerified: true,
      preparedCacheVerified: true,
      preparationTransientResidueVerified: true,
    },
  };
  return validateBoundedCampaignEntryFailureReceipt(receipt, spec.tier);
}

function validateContainedReceiptArtifactIdentity(receipt, label, expectedTier = null) {
  for (const source of [receipt.source?.sceneWorks, receipt.source?.inference]) {
    if (!/^[0-9a-f]{40}$/.test(source?.revision ?? "")
        || !/^[0-9a-f]{40}$/.test(source?.tree ?? "")) {
      fail(`${label} receipt source identity is malformed`);
    }
  }
  const preparation = receipt.artifacts?.preparation;
  const identity = preparation?.identity;
  const validFileIdentity = (file, mode) => path.isAbsolute(file?.path ?? "")
    && /^[0-9a-f]{64}$/.test(file?.sha256 ?? "")
    && typeof file?.device === "string" && typeof file?.inode === "string"
    && Number.isSafeInteger(file?.size) && file.size > 0
    && typeof file?.mtimeNs === "string" && typeof file?.ctimeNs === "string"
    && file?.mode === mode;
  const expectedNumericTier = expectedTier === null
    ? NUMERIC_TIER_INVENTORIES.q4 : NUMERIC_TIER_INVENTORIES[expectedTier];
  const expectedIdentity = {
    schemaVersion: PREPARATION_SCHEMA_VERSION,
    sceneWorksTree: receipt.source.sceneWorks.tree,
    inferenceTree: receipt.source.inference.tree,
    toolchainChannel: identity?.toolchainChannel,
    platform: "darwin",
    architecture: "arm64",
    artifact: {
      repository: ARTIFACT_REPOSITORY,
      revision: ARTIFACT_REVISION,
      numericTier: expectedNumericTier,
      textEncoder: {
        files: TEXT_ENCODER_INVENTORY_FILES,
        bytes: TEXT_ENCODER_INVENTORY_BYTES,
        sha256: TEXT_ENCODER_INVENTORY_SHA256,
      },
    },
  };
  if (receipt.artifacts?.repository !== ARTIFACT_REPOSITORY
      || receipt.artifacts?.revision !== ARTIFACT_REVISION
      || !isDeepStrictEqual(receipt.artifacts?.numericTierInventory, expectedNumericTier)
      || !isDeepStrictEqual(receipt.artifacts?.textEncoderInventory, {
        files: TEXT_ENCODER_INVENTORY_FILES,
        bytes: TEXT_ENCODER_INVENTORY_BYTES,
        sha256: TEXT_ENCODER_INVENTORY_SHA256,
      })
      || !/^[0-9a-f]{64}$/.test(preparation?.key ?? "")
      || !path.isAbsolute(preparation?.root ?? "")
      || path.basename(preparation?.root ?? "") !== preparation.key
      || preparation?.manifest?.key !== preparation.key
      || preparation?.key !== preparationCacheKey(expectedIdentity)
      || preparation?.manifest?.schemaVersion !== PREPARATION_SCHEMA_VERSION
      || !isDeepStrictEqual(preparation?.manifest?.identity, identity)
      || preparation?.manifest?.preparedFrom?.sceneWorksRevision
        !== receipt.source.sceneWorks.revision
      || preparation?.manifest?.preparedFrom?.inferenceRevision
        !== receipt.source.inference.revision
      || !isDeepStrictEqual(preparation?.manifest?.artifacts?.numericTier?.content,
        receipt.artifacts.numericTierInventory)
      || !isDeepStrictEqual(preparation?.manifest?.artifacts?.textEncoder?.content,
        receipt.artifacts.textEncoderInventory)
      || identity?.sceneWorksTree !== receipt.source.sceneWorks.tree
      || identity?.inferenceTree !== receipt.source.inference.tree
      || identity?.schemaVersion !== PREPARATION_SCHEMA_VERSION
      || !/^\d+\.\d+\.\d+$/.test(identity?.toolchainChannel ?? "")
      || !isDeepStrictEqual(identity, expectedIdentity)
      || !isDeepStrictEqual(identity?.artifact, {
        repository: ARTIFACT_REPOSITORY,
        revision: ARTIFACT_REVISION,
        numericTier: receipt.artifacts.numericTierInventory,
        textEncoder: receipt.artifacts.textEncoderInventory,
      })
      || !validFileIdentity(receipt.artifacts?.adapter, 0o500)
      || !validFileIdentity(receipt.artifacts?.metallib, 0o400)
      || !isDeepStrictEqual(preparation?.manifest?.adapter,
        preparedFileManifestIdentity(receipt.artifacts.adapter))
      || !isDeepStrictEqual(preparation?.manifest?.metallib,
        preparedFileManifestIdentity(receipt.artifacts.metallib))) {
    fail(`${label} receipt artifact or sealed-preparation identity changed`);
  }
}

export function validateCampaignEntryFailureReceipt(receipt) {
  let canonicallyIngestible = true;
  try {
    validateBundle(receipt);
  } catch {
    canonicallyIngestible = false;
  }
  const expectedCampaignCase = {
    plan: CAMPAIGN_ENTRY_PLAN,
    provider: CAMPAIGN_ENTRY_PROVIDER,
    logicalCaseId: CAMPAIGN_ENTRY_LOGICAL_CASE_ID,
    fixture: CAMPAIGN_ENTRY_FIXTURE,
    identity: CAMPAIGN_ENTRY_IDENTITY,
    action: CAMPAIGN_ENTRY_ACTION,
    target: {
      provider: PROVIDER, tier: "q4",
      geometry: { width: 768, height: 512, batch: 1, frames: 121, fps: 30 },
    },
    strategy: {
      rung: "staged_residency", engagedRungs: ["resident", "staged_residency"], parameters: {},
    },
  };
  const expectedCleanup = {
    ownedProcessGroupResidueVerified: true,
    runScratchRemoved: true,
    runRootEmpty: true,
    sourceIdentityVerified: true,
    runtimeAssetIdentityVerified: true,
    preparedCacheVerified: true,
    preparationTransientResidueVerified: true,
  };
  if (canonicallyIngestible
      || receipt?.recordType !== CAMPAIGN_FAILURE_RECEIPT_TYPE
      || receipt?.status !== "diagnostic_failure"
      || receipt?.diagnosticOnly !== true
      || receipt?.promotable !== false
      || receipt?.ingestible !== false
      || receipt?.canonicalBundlePublished !== false
      || receipt?.outcome?.canonicalBundleAbsentAtPublication !== true
      || receipt?.outcome?.outcomeReservationHeldAtPublication !== true
      || receipt?.outcome?.outcomeChoice !== "failure"
      || !path.isAbsolute(receipt?.outcome?.canonicalOutput ?? "")
      || receipt?.outcome?.failureOutput
        !== campaignEntryFailurePath(receipt?.outcome?.canonicalOutput ?? "")
      || receipt?.outcome?.reservation
        !== campaignEntryOutcomeReservationPath(receipt?.outcome?.canonicalOutput ?? "")
      || receipt?.outcome?.choice !== `${receipt?.outcome?.reservation}.choice`
      || receipt?.story !== "sc-20216"
      || !isDeepStrictEqual(receipt?.campaignCase, expectedCampaignCase)
      || receipt?.watchdog?.protocol !== "sceneworks-memory-watchdog-v1"
      || receipt?.watchdog?.providerPhaseProtocol !== PROVIDER_PHASE_PROTOCOL
      || !isDeepStrictEqual(receipt?.watchdog?.providerPhases, PROVIDER_PHASES)
      || receipt?.watchdog?.eventChain?.protocol !== WATCHDOG_EVENT_CHAIN_PROTOCOL
      || receipt?.watchdog?.maxFootprintBytes !== MAX_FOOTPRINT_BYTES
      || receipt?.watchdog?.maxRuntimeSeconds !== MAX_RUNTIME_SECONDS
      || receipt?.watchdog?.minInitialMemoryFreeBytes
        !== preflightFreeFloor(receipt?.watchdog?.hostMemoryBytes)
      || receipt?.watchdog?.minMemoryFreeBytes
        !== runtimeFreeFloor(receipt?.watchdog?.hostMemoryBytes)
      || !isDeepStrictEqual(receipt?.cleanup, expectedCleanup)) {
    fail("SC-20216 failure receipt is ingestible, incomplete, or identity-drifted");
  }
  validateContainedReceiptArtifactIdentity(receipt, "SC-20216 failure");
  const validated = validateCampaignEntryFailureEvents(
    receipt.watchdog.events, receipt.watchdog.hostMemoryBytes,
  );
  if (!isDeepStrictEqual(receipt.watchdog.failure, validated)
      || !isDeepStrictEqual(receipt.watchdog.eventChain, validated.eventChain)) {
    fail("SC-20216 failure receipt summary does not match its complete event stream");
  }
  return receipt;
}

export function validateBoundedCampaignEntryFailureReceipt(receipt, requested = "q4") {
  const spec = boundedCampaignEntrySpec(requested);
  let ingestible = true;
  try { validateBundle(receipt); } catch { ingestible = false; }
  const expectedCase = {
    plan: BOUNDED_CAMPAIGN_ENTRY_PLAN,
    provider: spec.provider,
    logicalCaseId: spec.logicalCaseId,
    fixture: spec.fixture,
    identity: spec.identity,
    action: BOUNDED_CAMPAIGN_ENTRY_ACTION,
    target: {
      provider: PROVIDER, tier: spec.tier,
      geometry: { width: 768, height: 512, batch: 1, frames: 121, fps: 30 },
    },
    strategy: {
      rung: "bounded_decode",
      engagedRungs: ["resident", "staged_residency", "bounded_decode"],
      parameters: { decodeTileEdge: 192, decodeOverlap: 64 },
      spatialDecodeTiles: 24,
    },
  };
  const expectedCleanup = {
    ownedProcessGroupResidueVerified: true,
    runScratchRemoved: true,
    runRootEmpty: true,
    sourceIdentityVerified: true,
    runtimeAssetIdentityVerified: true,
    preparedCacheVerified: true,
    preparationTransientResidueVerified: true,
  };
  const authorization = receipt?.outcome?.q8ReleaseAuthorization;
  const exactBf16Authorization = spec.tier === "bf16"
    && authorization?.schemaVersion === 1
    && authorization?.recordType === "sceneworks_sc20430_q8_release_authorization_v1"
    && path.isAbsolute(authorization?.auditFile?.path ?? "")
    && authorization?.auditFile?.mode === 0o400
    && Number.isSafeInteger(authorization?.auditFile?.size)
    && authorization.auditFile.size > 0
    && typeof authorization?.auditFile?.device === "string"
    && typeof authorization?.auditFile?.inode === "string"
    && typeof authorization?.auditFile?.mtimeNs === "string"
    && typeof authorization?.auditFile?.ctimeNs === "string"
    && /^[0-9a-f]{64}$/.test(authorization?.auditFile?.sha256 ?? "")
    && path.isAbsolute(authorization?.q8?.canonicalOutput ?? "")
    && /^[0-9a-f]{64}$/.test(authorization?.q8?.canonicalSha256 ?? "")
    && authorization?.q8?.logicalCaseId === BOUNDED_CAMPAIGN_ENTRY_SPECS.q8.logicalCaseId
    && /^imc-[0-9a-f]{20}$/.test(authorization?.q8?.recordId ?? "")
    && authorization?.q8?.sceneWorksRevision === receipt?.source?.sceneWorks?.revision
    && authorization?.q8?.inferenceRevision === receipt?.source?.inference?.revision;
  if (ingestible
      || receipt?.recordType !== BOUNDED_CAMPAIGN_FAILURE_RECEIPT_TYPE
      || receipt?.status !== "diagnostic_failure"
      || receipt?.diagnosticOnly !== true || receipt?.promotable !== false
      || receipt?.ingestible !== false || receipt?.canonicalBundlePublished !== false
      || receipt?.outcome?.canonicalBundleAbsentAtPublication !== true
      || receipt?.outcome?.outcomeReservationHeldAtPublication !== true
      || receipt?.outcome?.outcomeChoice !== "failure"
      || receipt?.outcome?.failureOutput
        !== boundedCampaignEntryFailurePath(receipt?.outcome?.canonicalOutput ?? "", spec.tier)
      || receipt?.outcome?.reservation
        !== boundedCampaignEntryOutcomeReservationPath(
          receipt?.outcome?.canonicalOutput ?? "", spec.tier,
        )
      || receipt?.outcome?.choice !== `${receipt?.outcome?.reservation}.choice`
      || receipt?.story !== spec.story
      || (spec.tier === "bf16" ? !exactBf16Authorization : authorization != null)
      || !isDeepStrictEqual(receipt?.campaignCase, expectedCase)
      || receipt?.watchdog?.protocol !== "sceneworks-memory-watchdog-v1"
      || receipt?.watchdog?.providerPhaseProtocol !== PROVIDER_PHASE_PROTOCOL
      || !isDeepStrictEqual(receipt?.watchdog?.providerPhases, BOUNDED_CARRIER_PHASES)
      || receipt?.watchdog?.maxFootprintBytes !== MAX_FOOTPRINT_BYTES
      || receipt?.watchdog?.maxRuntimeSeconds !== MAX_RUNTIME_SECONDS
      || !isDeepStrictEqual(receipt?.cleanup, expectedCleanup)) {
    fail("SC-20318 failure receipt is ingestible, incomplete, or identity-drifted");
  }
  validateContainedReceiptArtifactIdentity(receipt, "SC-20318 failure", spec.tier);
  const validated = validateCampaignEntryFailureEvents(
    receipt.watchdog.events, receipt.watchdog.hostMemoryBytes, BOUNDED_CARRIER_PHASES,
  );
  if (!isDeepStrictEqual(receipt.watchdog.failure, validated)
      || !isDeepStrictEqual(receipt.watchdog.eventChain, validated.eventChain)) {
    fail("SC-20318 failure receipt summary changed from its authenticated stream");
  }
  return receipt;
}

export function boundedCarrierSuccessPath(canonicalOutput) {
  return path.join(path.dirname(canonicalOutput), "sc-20254-bounded-carrier-success.json");
}

export function boundedCarrierFailurePath(canonicalOutput) {
  return path.join(path.dirname(canonicalOutput), "sc-20254-bounded-carrier-failure.json");
}

export function boundedCarrierOutcomeReservationPath(canonicalOutput) {
  return path.join(path.dirname(canonicalOutput), ".sc-20254-bounded-carrier-outcome");
}

function boundedCarrierCaseIdentity() {
  return {
    action: BOUNDED_CARRIER_ACTION,
    logicalCaseId: BOUNDED_CARRIER_LOGICAL_CASE_ID,
    fixture: BOUNDED_CARRIER_FIXTURE,
    identity: BOUNDED_CARRIER_IDENTITY,
    target: {
      provider: PROVIDER, tier: "q4",
      geometry: { width: 768, height: 512, batch: 1, frames: 121, fps: 30 },
      seed: 18_946, videoMode: "default_av", audio: true,
    },
    strategy: {
      rung: "bounded_decode",
      engagedRungs: ["resident", "staged_residency", "bounded_decode"],
      parameters: { decodeTileEdge: 192, decodeOverlap: 64 },
      spatialDecodeTiles: 24,
    },
  };
}

function boundedCarrierReceiptBase({
  recordType, status, sceneWorksRevision, sceneWorksTree, inferenceRevision, inferenceTree,
  identity, preparationKey, preparationRoot, prepared, hostMemoryBytes, events, outcome,
}) {
  return {
    schemaVersion: 1,
    recordType,
    status,
    diagnosticOnly: true,
    promotable: false,
    ingestible: false,
    canonicalBundlePublished: false,
    outcome: structuredClone(outcome),
    story: "sc-20254",
    source: {
      sceneWorks: { revision: sceneWorksRevision, tree: sceneWorksTree },
      inference: { revision: inferenceRevision, tree: inferenceTree },
    },
    boundedCarrier: boundedCarrierCaseIdentity(),
    artifacts: {
      repository: ARTIFACT_REPOSITORY,
      revision: ARTIFACT_REVISION,
      numericTierInventory: structuredClone(identity.artifact.numericTier),
      textEncoderInventory: structuredClone(identity.artifact.textEncoder),
      adapter: structuredClone(prepared.adapter),
      metallib: structuredClone(prepared.metallib),
      preparation: {
        key: preparationKey,
        root: preparationRoot,
        identity: structuredClone(identity),
        manifest: structuredClone(prepared.manifest),
      },
    },
    watchdog: {
      protocol: "sceneworks-memory-watchdog-v1",
      providerPhaseProtocol: PROVIDER_PHASE_PROTOCOL,
      providerPhaseProfile: "bounded-carrier",
      providerPhases: [...BOUNDED_CARRIER_PHASES],
      maxFootprintBytes: MAX_FOOTPRINT_BYTES,
      maxRuntimeSeconds: MAX_RUNTIME_SECONDS,
      hostMemoryBytes,
      minInitialMemoryFreeBytes: preflightFreeFloor(hostMemoryBytes),
      minMemoryFreeBytes: runtimeFreeFloor(hostMemoryBytes),
      eventChain: validateWatchdogEventChain(events),
      events: structuredClone(events),
    },
    cleanup: {
      ownedProcessGroupResidueVerified: true,
      runScratchRemoved: true,
      runRootEmpty: true,
      sourceIdentityVerified: true,
      runtimeAssetIdentityVerified: true,
      preparedCacheVerified: true,
      preparationTransientResidueVerified: true,
    },
  };
}

export function boundedCarrierSuccessReceipt(options) {
  const maxObservedFootprintBytes = validateBoundedCarrierWatchdogEvents(
    options.events, options.hostMemoryBytes,
  );
  const receipt = boundedCarrierReceiptBase({
    ...options,
    recordType: BOUNDED_CARRIER_SUCCESS_RECEIPT_TYPE,
    status: "diagnostic_success",
  });
  receipt.response = structuredClone(validateBoundedCarrierResponse(options.response, {
    inferenceRevision: options.inferenceRevision,
    hostMemoryBytes: options.hostMemoryBytes,
  }));
  receipt.watchdog.maxObservedFootprintBytes = maxObservedFootprintBytes;
  return validateBoundedCarrierReceipt(receipt);
}

export function boundedCarrierFailureReceipt(options) {
  const failure = validateCampaignEntryFailureEvents(
    options.events, options.hostMemoryBytes, BOUNDED_CARRIER_PHASES,
  );
  const receipt = boundedCarrierReceiptBase({
    ...options,
    recordType: BOUNDED_CARRIER_FAILURE_RECEIPT_TYPE,
    status: "diagnostic_failure",
  });
  receipt.watchdog.failure = failure;
  return validateBoundedCarrierReceipt(receipt);
}

export function validateBoundedCarrierReceipt(receipt) {
  let canonicallyIngestible = true;
  try {
    validateBundle(receipt);
  } catch {
    canonicallyIngestible = false;
  }
  const success = receipt?.recordType === BOUNDED_CARRIER_SUCCESS_RECEIPT_TYPE
    && receipt?.status === "diagnostic_success";
  const failure = receipt?.recordType === BOUNDED_CARRIER_FAILURE_RECEIPT_TYPE
    && receipt?.status === "diagnostic_failure";
  const expectedCleanup = {
    ownedProcessGroupResidueVerified: true,
    runScratchRemoved: true,
    runRootEmpty: true,
    sourceIdentityVerified: true,
    runtimeAssetIdentityVerified: true,
    preparedCacheVerified: true,
    preparationTransientResidueVerified: true,
  };
  const canonicalOutput = receipt?.outcome?.canonicalOutput ?? "";
  if (canonicallyIngestible
      || (!success && !failure)
      || receipt?.diagnosticOnly !== true
      || receipt?.promotable !== false
      || receipt?.ingestible !== false
      || receipt?.canonicalBundlePublished !== false
      || receipt?.story !== "sc-20254"
      || !isDeepStrictEqual(receipt?.boundedCarrier, boundedCarrierCaseIdentity())
      || !path.isAbsolute(canonicalOutput)
      || receipt?.outcome?.successOutput !== boundedCarrierSuccessPath(canonicalOutput)
      || receipt?.outcome?.failureOutput !== boundedCarrierFailurePath(canonicalOutput)
      || receipt?.outcome?.reservation !== boundedCarrierOutcomeReservationPath(canonicalOutput)
      || receipt?.outcome?.choice !== `${receipt?.outcome?.reservation}.choice`
      || receipt?.outcome?.canonicalPublicationClaim
        !== canonicalPublicationClaimPath(canonicalOutput)
      || receipt?.outcome?.canonicalPublicationClaimHeldAtPublication !== true
      || receipt?.outcome?.canonicalBundleAbsentAtPublication !== true
      || receipt?.outcome?.outcomeReservationHeldAtPublication !== true
      || receipt?.outcome?.outcomeChoice !== (success ? "success" : "failure")
      || receipt?.watchdog?.protocol !== "sceneworks-memory-watchdog-v1"
      || receipt?.watchdog?.providerPhaseProtocol !== PROVIDER_PHASE_PROTOCOL
      || receipt?.watchdog?.providerPhaseProfile !== "bounded-carrier"
      || !isDeepStrictEqual(receipt?.watchdog?.providerPhases, BOUNDED_CARRIER_PHASES)
      || receipt?.watchdog?.eventChain?.protocol !== WATCHDOG_EVENT_CHAIN_PROTOCOL
      || receipt?.watchdog?.maxFootprintBytes !== MAX_FOOTPRINT_BYTES
      || receipt?.watchdog?.maxRuntimeSeconds !== MAX_RUNTIME_SECONDS
      || receipt?.watchdog?.minInitialMemoryFreeBytes
        !== preflightFreeFloor(receipt?.watchdog?.hostMemoryBytes)
      || receipt?.watchdog?.minMemoryFreeBytes
        !== runtimeFreeFloor(receipt?.watchdog?.hostMemoryBytes)
      || !isDeepStrictEqual(receipt?.cleanup, expectedCleanup)) {
    fail("SC-20254 receipt is ingestible, incomplete, or identity-drifted");
  }
  validateContainedReceiptArtifactIdentity(receipt, "SC-20254");
  const expectedEventChain = validateWatchdogEventChain(receipt.watchdog.events);
  if (!isDeepStrictEqual(receipt.watchdog.eventChain, expectedEventChain)) {
    fail("SC-20254 receipt event chain does not match its complete stream");
  }
  if (success) {
    const maximum = validateBoundedCarrierWatchdogEvents(
      receipt.watchdog.events, receipt.watchdog.hostMemoryBytes,
    );
    validateBoundedCarrierResponse(receipt.response, {
      inferenceRevision: receipt.source.inference.revision,
      hostMemoryBytes: receipt.watchdog.hostMemoryBytes,
    });
    if (receipt.watchdog.maxObservedFootprintBytes !== maximum
        || Object.hasOwn(receipt.watchdog, "failure")) {
      fail("SC-20254 success receipt watchdog summary changed");
    }
  } else {
    const validated = validateCampaignEntryFailureEvents(
      receipt.watchdog.events, receipt.watchdog.hostMemoryBytes, BOUNDED_CARRIER_PHASES,
    );
    if (!isDeepStrictEqual(receipt.watchdog.failure, validated)
        || Object.hasOwn(receipt, "response")) {
      fail("SC-20254 failure receipt summary changed");
    }
  }
  return receipt;
}

function sameInventory(left, right) {
  return left?.root === right?.root
    && left?.files === right?.files
    && left?.bytes === right?.bytes
    && left?.sha256 === right?.sha256;
}

function assertInventory(actual, expected, label) {
  if (!actual
      || (expected.root !== undefined && actual.root !== expected.root)
      || actual.files !== expected.files
      || actual.bytes !== expected.bytes
      || actual.sha256 !== expected.sha256) {
    fail(`${label} inventory does not match its immutable identity`);
  }
  return actual;
}

export function inventoryAtRoot(inventory, root) {
  return {
    root,
    files: inventory.files,
    bytes: inventory.bytes,
    sha256: inventory.sha256,
  };
}

export function validateCanaryResponse(
  response,
  expectedInferenceRevision,
  expectedTextEncoderInventory,
  expectedHostMemoryBytes,
  profileName = SAFETY_CANARY_PROFILE,
) {
  const profile = canaryProfile(profileName);
  if (response?.status !== profile.status
      || response?.diagnosticOnly !== true
      || response?.promotable !== false
      || response?.ingestible !== false) {
    fail("adapter response is not a non-promotable diagnostic canary");
  }
  if (response?.calibrationFingerprint !== FINGERPRINT
      || (expectedInferenceRevision !== undefined
        && response?.inferenceRevision !== expectedInferenceRevision)
      || response?.canaryIdentity !== profile.identity
      || response?.target?.provider !== PROVIDER
      || response?.target?.tier !== "q4"
      || response?.target?.geometry?.width !== profile.width
      || response?.target?.geometry?.height !== profile.height
      || response?.target?.geometry?.frames !== profile.frames
      || response?.target?.geometry?.fps !== profile.fps
      || response?.target?.videoMode !== profile.videoMode
      || response?.target?.audio !== profile.audio
      || response?.output?.frames !== profile.frames
      || response?.output?.fps !== profile.fps
      || response?.output?.frameTimelineSeconds !== profile.frameTimelineSeconds) {
    fail("adapter response changed the exact canary identity");
  }
  const outputAudio = response?.output?.audio;
  if (outputAudio?.present !== profile.audio
      || (profile.audio && (!Number.isSafeInteger(outputAudio?.samples)
        || outputAudio.samples <= 0
        || !Number.isSafeInteger(outputAudio?.sampleRate)
        || outputAudio.sampleRate <= 0
        || !Number.isSafeInteger(outputAudio?.channels)
        || outputAudio.channels <= 0))
      || (!profile.audio && (outputAudio?.samples !== 0
        || outputAudio?.sampleRate !== 0
        || outputAudio?.channels !== 0))) {
    fail("adapter response did not attest the exact canary audio output");
  }
  if (response?.artifact?.repository !== ARTIFACT_REPOSITORY
      || response?.artifact?.resolvedRevision !== ARTIFACT_REVISION
      || response?.artifact?.numericTierInventory?.files !== 11
      || response?.artifact?.numericTierInventory?.bytes !== Q4_INVENTORY_BYTES
      || response?.artifact?.numericTierInventory?.sha256 !== Q4_INVENTORY_SHA256
      || !Number.isInteger(response?.artifact?.textEncoderInventory?.files)
      || response.artifact.textEncoderInventory.files !== TEXT_ENCODER_INVENTORY_FILES
      || !Number.isInteger(response?.artifact?.textEncoderInventory?.bytes)
      || response.artifact.textEncoderInventory.bytes !== TEXT_ENCODER_INVENTORY_BYTES
      || response?.artifact?.textEncoderInventory?.sha256 !== TEXT_ENCODER_INVENTORY_SHA256
      || (expectedTextEncoderInventory !== undefined
        && !sameInventory(response?.artifact?.textEncoderInventory, expectedTextEncoderInventory))) {
    fail("adapter response changed the immutable canary artifact identity");
  }
  if (response?.strategy?.rung !== "bounded_decode"
      || JSON.stringify(response?.strategy?.engagedRungs)
        !== JSON.stringify(["resident", "staged_residency", "bounded_decode"])
      || response?.strategy?.parameters?.decodeTileEdge !== 192
      || response?.strategy?.parameters?.decodeOverlap !== 64
      || response?.strategy?.spatialDecodeTiles !== profile.spatialDecodeTiles
      || response.strategy.spatialDecodeTiles <= 1
      || response?.watchdog?.required !== true
      || response?.watchdog?.protocol !== "sceneworks-memory-watchdog-v1"
      || response?.watchdog?.maxFootprintBytes !== MAX_FOOTPRINT_BYTES
      || response?.watchdog?.maxRuntimeSeconds !== MAX_RUNTIME_SECONDS
      || !Number.isSafeInteger(response?.watchdog?.hostMemoryBytes)
      || (expectedHostMemoryBytes !== undefined
        && response?.watchdog?.hostMemoryBytes !== expectedHostMemoryBytes)
      || response?.watchdog?.minInitialMemoryFreeBytes
        !== preflightFreeFloor(response?.watchdog?.hostMemoryBytes)
      || Object.hasOwn(response?.watchdog ?? {}, "minInitialMemoryFreePercent")
      || response?.watchdog?.minMemoryFreeBytes
        !== runtimeFreeFloor(response?.watchdog?.hostMemoryBytes)
      || Object.hasOwn(response?.watchdog ?? {}, "minSwapFreeBytes")
      || response?.mlxLimits?.memoryLimitBytes !== MAX_FOOTPRINT_BYTES
      || !Number.isInteger(response?.mlxLimits?.wiredLimitBytes)
      || response.mlxLimits.wiredLimitBytes <= 0
      || response.mlxLimits.wiredLimitBytes > MAX_FOOTPRINT_BYTES
      || response?.output?.firstFrameNondegenerate !== true) {
    fail("adapter response did not attest the exact bounded canary execution");
  }
  const phaseActive = [];
  for (const phaseName of ["conditioning", "denoise", "decode"]) {
    const phase = response?.observedMemory?.[phaseName];
    for (const field of ["activeBytes", "allocatorBytes", "reclaimableBytes"]) {
      if (!Number.isSafeInteger(phase?.[field]) || phase[field] < 0) {
        fail(`adapter response observedMemory.${phaseName}.${field} must be a non-negative safe integer`);
      }
    }
    if (phase.activeBytes <= 0
        || phase.allocatorBytes !== phase.activeBytes + phase.reclaimableBytes
        || !Number.isSafeInteger(phase.allocatorBytes)) {
      fail(`adapter response observedMemory.${phaseName} is not an exact phase-memory attestation`);
    }
    phaseActive.push(phase.activeBytes);
  }
  if (response?.observedMemory?.peakActiveBytes !== Math.max(...phaseActive)) {
    fail("adapter response observedMemory.peakActiveBytes did not equal the maximum phase active bytes");
  }
  for (const field of [
    "preProviderActiveBytes",
    "preProviderCacheBytes",
    "postCleanupActiveBytes",
    "postCleanupCacheBytes",
  ]) {
    const value = response?.observedMemory?.[field];
    if (!Number.isSafeInteger(value) || value < 0) {
      fail(`adapter response observedMemory.${field} must be a non-negative safe integer`);
    }
  }
  if (response.observedMemory.preProviderCacheBytes !== 0) {
    fail(`adapter response observedMemory.preProviderCacheBytes ${response.observedMemory.preProviderCacheBytes} did not attest the cleared pre-provider cache 0`);
  }
  const persistent = response?.observedMemory?.expectedPersistentActive;
  for (const [field, expected] of [
    ["identity", LTX_ONES_CACHE_IDENTITY],
    ["videoDimension", LTX_ONES_CACHE_VIDEO_DIMENSION],
    ["audioDimension", LTX_ONES_CACHE_AUDIO_DIMENSION],
    ["dtype", "bfloat16"],
    ["bytesPerElement", BFLOAT16_BYTES_PER_ELEMENT],
    ["bytes", ltxOnesCacheBytes()],
  ]) {
    if (persistent?.[field] !== expected) {
      fail(`adapter response observedMemory.expectedPersistentActive.${field} ${JSON.stringify(persistent?.[field])} did not equal ${JSON.stringify(expected)}`);
    }
  }
  const expectedPostActive = response.observedMemory.preProviderActiveBytes
    + persistent.bytes;
  if (!Number.isSafeInteger(expectedPostActive)) {
    fail("adapter response pre-provider plus persistent active-byte arithmetic overflowed");
  }
  if (response.observedMemory.postCleanupActiveBytes !== expectedPostActive) {
    fail(`adapter response observedMemory.postCleanupActiveBytes ${response.observedMemory.postCleanupActiveBytes} did not equal pre-provider active ${response.observedMemory.preProviderActiveBytes} plus intentional persistent active ${persistent.bytes} = ${expectedPostActive}`);
  }
  if (response.observedMemory.postCleanupCacheBytes
      !== response.observedMemory.preProviderCacheBytes) {
    fail(`adapter response observedMemory.postCleanupCacheBytes ${response.observedMemory.postCleanupCacheBytes} did not return to observedMemory.preProviderCacheBytes ${response.observedMemory.preProviderCacheBytes}`);
  }
  return response;
}

export function parseMemoryFreePercent(output) {
  const value = Number(output.match(/System-wide memory free percentage:\s*(\d+)%/)?.[1]);
  if (!Number.isInteger(value) || value < 0 || value > 100) {
    fail("memory_pressure did not report a valid system-wide free percentage");
  }
  return value;
}

export function parseSwapFreeBytes(output) {
  const match = output.match(/\bfree\s*=\s*([0-9]+(?:\.[0-9]+)?)([MG])\b/i);
  if (!match) fail("vm.swapusage did not report free swap");
  const multiplier = match[2].toUpperCase() === "G" ? 1024 ** 3 : 1024 ** 2;
  return Number(match[1]) * multiplier;
}

export function foreignHeavyProcesses(output) {
  const heavy = [];
  for (const line of output.split("\n")) {
    const match = line.trim().match(/^(\d+)\s+(\d+)\s+(.+)$/);
    if (!match) continue;
    const command = match[3];
    const executable = path.basename(command.trim().split(/\s+/, 1)[0]);
    const isPython = /^python(?:3(?:\.\d+)?)?$/.test(executable);
    if (["sceneworks-worker", "memory-mlx-adapter", "Runner.Worker"].includes(executable)
        || executable.includes("real_weights")
        || executable === "real_weight_tiling"
        || (isPython && /(?:MiniMax|minimax|real_weights)/.test(command))) {
      heavy.push(line.trim());
    }
  }
  return heavy;
}

async function assertHostPreflight(memoryBytes, signal) {
  const [host, processes] = await Promise.all([
    sampleHostPressure(memoryBytes, signal),
    execFileAsync("/bin/ps", ["-ww", "-axo", "pid=,ppid=,command="], {
      encoding: "utf8", timeout: 2_000, signal,
    }),
  ]);
  const foreign = foreignHeavyProcesses(processes.stdout);
  if (foreign.length) fail(`foreign heavy workload is active:\n${foreign.join("\n")}`);
  return host;
}

async function sampleHostPressure(memoryBytes, signal) {
  const [pressure, swap] = await Promise.all([
    execFileAsync("/usr/bin/memory_pressure", [], { encoding: "utf8", timeout: 2_000, signal }),
    execFileAsync("/usr/sbin/sysctl", ["vm.swapusage"], { encoding: "utf8", timeout: 2_000, signal }),
  ]);
  const memoryFreePercent = parseMemoryFreePercent(pressure.stdout);
  const swapFreeBytes = parseSwapFreeBytes(swap.stdout);
  const memoryFreeBytes = Math.floor(memoryBytes * memoryFreePercent / 100);
  const freeFloor = preflightFreeFloor(memoryBytes);
  if (memoryFreeBytes < freeFloor) {
    fail(`host memory free ${memoryFreeBytes} is below ${freeFloor}`);
  }
  return { memoryFreePercent, memoryFreeBytes, swapFreeBytes };
}

async function sha256(file, signal) {
  const digest = createHash("sha256");
  signal?.throwIfAborted();
  for await (const chunk of createReadStream(file, { signal })) {
    digest.update(chunk);
    signal?.throwIfAborted();
  }
  return digest.digest("hex");
}

export async function repositoryToolchain() {
  const source = await readFile(path.join(ROOT, "rust-toolchain.toml"), "utf8");
  const channels = [...source.matchAll(/^channel\s*=\s*"([^"]+)"\s*$/gm)].map((match) => match[1]);
  if (channels.length !== 1 || !/^\d+\.\d+\.\d+$/.test(channels[0])) {
    fail("rust-toolchain.toml must contain one concrete semantic version channel");
  }
  return channels[0];
}

export async function exactToolchain(scratch, signal) {
  const home = process.env.HOME ?? fail("HOME is required to resolve the pinned rustup toolchain");
  const resolvedRustupHome = path.resolve(process.env.RUSTUP_HOME ?? path.join(home, ".rustup"));
  const resolutionEnv = Object.fromEntries([
    "HOME", "SSL_CERT_FILE", "SSL_CERT_DIR",
  ].flatMap((name) => process.env[name] === undefined ? [] : [[name, process.env[name]]]));
  resolutionEnv.RUSTUP_HOME = resolvedRustupHome;
  const systemPath = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"].join(":");
  resolutionEnv.PATH = [path.join(home, ".cargo/bin"), systemPath].join(":");
  const rustup = (await runOwnedCommand("/usr/bin/which", ["rustup"], {
    cwd: ROOT, env: resolutionEnv, timeout: 2_000, signal,
  })).stdout.trim();
  if (!path.isAbsolute(rustup)) fail("the sanitized PATH did not resolve absolute rustup");
  const rustupReal = await realpath(rustup);
  if (!(await stat(rustupReal)).isFile()) fail("resolved rustup is not a regular file");
  const channel = await repositoryToolchain();
  const cargo = (await runOwnedCommand(rustupReal, [
    "which", "--toolchain", channel, "cargo",
  ], { cwd: ROOT, env: resolutionEnv, timeout: 2_000, signal })).stdout.trim();
  const rustc = (await runOwnedCommand(rustupReal, [
    "which", "--toolchain", channel, "rustc",
  ], { cwd: ROOT, env: resolutionEnv, timeout: 2_000, signal })).stdout.trim();
  if (!path.isAbsolute(cargo) || !path.isAbsolute(rustc)) {
    fail("rustup did not resolve absolute pinned toolchain executables");
  }
  const version = (await runOwnedCommand(rustc, ["-Vv"], {
    cwd: ROOT, env: resolutionEnv, timeout: 2_000, signal,
  })).stdout;
  if (!new RegExp(`^release: ${channel.replaceAll(".", "\\.")}$`, "m").test(version)) {
    fail(`resolved rustc does not match pinned channel ${channel}`);
  }
  const privateHome = path.join(scratch, "home");
  const privateTemp = path.join(scratch, "tmp");
  const privateCargoHome = path.join(scratch, "cargo-home");
  const privateTarget = path.join(scratch, "target");
  await Promise.all([
    mkdir(privateHome, { recursive: true, mode: 0o700 }),
    mkdir(privateTemp, { recursive: true, mode: 0o700 }),
    mkdir(privateCargoHome, { recursive: true, mode: 0o700 }),
    mkdir(privateTarget, { recursive: true, mode: 0o700 }),
  ]);
  await exactMetadata(privateHome, "directory", 0o700);
  await exactEntries(privateHome, []);
  return {
    cargo,
    channel,
    env: {
      ...resolutionEnv,
      HOME: privateHome,
      PATH: systemPath,
      CARGO_BUILD_JOBS: "2",
      CARGO_HOME: privateCargoHome,
      CARGO_TARGET_DIR: privateTarget,
      CARGO_TERM_COLOR: "never",
      RUSTC: rustc,
      RUSTUP_TOOLCHAIN: channel,
      TMPDIR: privateTemp,
    },
  };
}

export async function cloneArtifactTree(source, destination, expectedDevice, signal) {
  signal?.throwIfAborted();
  const sourceMetadata = await lstat(source, { bigint: true });
  if (!sourceMetadata.isDirectory() || sourceMetadata.isSymbolicLink()
      || sourceMetadata.dev !== expectedDevice) {
    fail("artifact root must be a same-volume regular directory for private APFS cloning");
  }
  await mkdir(destination, { mode: 0o700 });
  for (const entry of await readdir(source, { withFileTypes: true })) {
    const input = path.join(source, entry.name);
    const output = path.join(destination, entry.name);
    const metadata = await lstat(input, { bigint: true });
    if (metadata.dev !== expectedDevice) {
      fail(`artifact entry ${input} crosses the private-clone volume`);
    }
    if (metadata.isDirectory()) {
      await cloneArtifactTree(input, output, expectedDevice, signal);
    } else if (metadata.isFile() || metadata.isSymbolicLink()) {
      const cloneSource = metadata.isSymbolicLink() ? await realpath(input) : input;
      const cloneMetadata = await stat(cloneSource, { bigint: true });
      if (!cloneMetadata.isFile() || cloneMetadata.dev !== expectedDevice) {
        fail(`artifact entry ${input} does not resolve to a same-volume regular file`);
      }
      await lstat(output).then(
        () => fail(`private artifact clone destination already exists: ${output}`),
        (error) => { if (error.code !== "ENOENT") throw error; },
      );
      if (process.platform === "darwin") {
        await runOwnedCommand("/bin/cp", ["-c", cloneSource, output], {
          timeout: 30_000, signal,
        });
      } else {
        // The production controller is Darwin-only because it requires APFS clonefile and
        // phys_footprint. Keep the source-only contract test portable without pretending that
        // a Linux copy is an admissible production artifact clone.
        await copyFile(cloneSource, output, fsConstants.COPYFILE_EXCL);
      }
      signal?.throwIfAborted();
      await chmod(output, 0o400);
    } else {
      fail(`artifact entry ${input} is not a regular file or directory`);
    }
  }
  await chmod(destination, 0o500);
}

export async function cleanupCanaryScratch(scratch) {
  async function makeRemovable(entry) {
    const metadata = await lstat(entry).catch((error) => {
      if (error.code === "ENOENT") return null;
      throw error;
    });
    if (metadata === null || metadata.isSymbolicLink()) return;
    if (metadata.isDirectory()) {
      await chmod(entry, 0o700);
      for (const child of await readdir(entry)) await makeRemovable(path.join(entry, child));
    } else {
      await chmod(entry, 0o600);
    }
  }
  await makeRemovable(scratch);
  await rm(scratch, { recursive: true, force: true });
}

async function pathExists(entry) {
  return lstat(entry).then(() => true, (error) => {
    if (error.code === "ENOENT") return false;
    throw error;
  });
}

export async function prepareArtifactClone(
  source, destination, expectedInventory, expectedDevice, signal,
) {
  await mkdir(path.dirname(destination), { recursive: true, mode: 0o700 });
  let reused = await pathExists(destination);
  if (!reused) {
    const staging = await mkdtemp(path.join(path.dirname(destination), ".artifact-stage-"));
    const staged = path.join(staging, "tree");
    try {
      await cloneArtifactTree(source, staged, expectedDevice, signal);
      assertInventory(
        await hashArtifactInventory(staged, { signal }),
        inventoryAtRoot(expectedInventory, staged),
        "staged private artifact clone",
      );
      await rename(staged, destination);
    } finally {
      await cleanupCanaryScratch(staging);
    }
  }
  const inventory = assertInventory(
    await hashArtifactInventory(destination, { signal }),
    inventoryAtRoot(expectedInventory, destination),
    reused ? "reused private artifact clone" : "published private artifact clone",
  );
  const seal = await sealedArtifactIdentity(destination);
  if (seal.files !== inventory.files || seal.bytes !== inventory.bytes) {
    fail("sealed artifact metadata does not match its content inventory");
  }
  return { inventory, seal, reused };
}

function sameJson(left, right) {
  return isDeepStrictEqual(left, right);
}

async function exactMetadata(entry, kind, mode) {
  const metadata = await lstat(entry, { bigint: true });
  const validKind = kind === "directory" ? metadata.isDirectory() : metadata.isFile();
  if (!validKind || metadata.isSymbolicLink() || Number(metadata.mode & 0o777n) !== mode) {
    fail(`prepared canary ${kind} mode changed: ${entry}`);
  }
  return metadata;
}

async function exactEntries(directory, expected) {
  const actual = (await readdir(directory)).sort();
  if (!sameJson(actual, [...expected].sort())) {
    fail(`prepared canary structure changed: ${directory}`);
  }
}

function preparedFileManifestIdentity(file) {
  return {
    sha256: file.sha256,
    size: file.size,
    seal: {
      device: file.device,
      inode: file.inode,
      mtimeNs: file.mtimeNs,
      ctimeNs: file.ctimeNs,
      mode: file.mode,
    },
  };
}

async function validatePreparedStructure(preparationRoot, tier = "q4") {
  await exactMetadata(preparationRoot, "directory", 0o500);
  await exactEntries(preparationRoot, ["adapter", "artifacts", "prepared.json", "prepared.sha256"]);
  const roots = privateArtifactRoots(preparationRoot, tier);
  const revisionDirectory = path.dirname(roots.numericTier);
  const snapshotsDirectory = path.dirname(revisionDirectory);
  const modelDirectory = path.dirname(snapshotsDirectory);
  const artifactsDirectory = path.join(preparationRoot, "artifacts");
  const adapterDirectory = path.join(preparationRoot, "adapter");
  for (const directory of [
    artifactsDirectory, modelDirectory, snapshotsDirectory, revisionDirectory, adapterDirectory,
  ]) {
    await exactMetadata(directory, "directory", 0o500);
  }
  await exactEntries(artifactsDirectory, [path.basename(modelDirectory)]);
  await exactEntries(modelDirectory, ["snapshots"]);
  await exactEntries(snapshotsDirectory, [ARTIFACT_REVISION]);
  await exactEntries(revisionDirectory, ["gemma", tier]);
  await exactEntries(adapterDirectory, ["memory-mlx-adapter", "mlx.metallib"]);
  return roots;
}

export async function validatePreparedCache(preparationRoot, key, identity, signal, tier = "q4") {
  const rootMetadata = await lstat(preparationRoot, { bigint: true }).catch((error) => {
    if (error.code === "ENOENT") return null;
    throw error;
  });
  if (rootMetadata === null) return null;
  const manifestPath = path.join(preparationRoot, "prepared.json");
  const completionPath = path.join(preparationRoot, "prepared.sha256");
  if (!(await pathExists(manifestPath)) || !(await pathExists(completionPath))) return null;
  const roots = await validatePreparedStructure(preparationRoot, tier);
  await exactMetadata(manifestPath, "file", 0o400);
  await exactMetadata(completionPath, "file", 0o400);
  const manifestBytes = await readFile(manifestPath, "utf8");
  const completion = await readFile(completionPath, "utf8");
  const manifestDigest = createHash("sha256").update(manifestBytes).digest("hex");
  if (completion !== `${manifestDigest}\n`) fail("prepared canary completion seal changed");
  const manifest = JSON.parse(manifestBytes);
  if (manifest?.schemaVersion !== PREPARATION_SCHEMA_VERSION
      || manifest?.key !== key
      || path.basename(preparationRoot) !== key
      || preparationCacheKey(identity) !== key
      || !sameJson(manifest?.identity, identity)) {
    fail("prepared canary cache identity does not match the exact source trees");
  }
  for (const [name, root] of Object.entries(roots)) {
    if (!sameJson(manifest?.artifacts?.[name]?.content, identity?.artifact?.[name])) {
      fail(`prepared canary ${name} manifest is not bound to the canonical artifact identity`);
    }
    const seal = await sealedArtifactIdentity(root);
    if (!sameJson(seal, manifest?.artifacts?.[name]?.seal)) {
      fail(`prepared canary ${name} seal changed`);
    }
    if (seal.files !== identity.artifact[name].files
        || seal.bytes !== identity.artifact[name].bytes) {
      fail(`prepared canary ${name} seal does not match the canonical artifact shape`);
    }
  }
  const adapterPath = path.join(preparationRoot, "adapter", "memory-mlx-adapter");
  const adapter = await adapterIdentity(adapterPath, signal);
  if (!sameJson(preparedFileManifestIdentity(adapter), manifest?.adapter)) {
    fail("prepared canary adapter changed");
  }
  const metallibPath = path.join(preparationRoot, "adapter", "mlx.metallib");
  const metallib = await metallibIdentity(metallibPath, signal);
  if (!sameJson(preparedFileManifestIdentity(metallib), manifest?.metallib)) {
    fail("prepared canary metallib changed");
  }
  return { manifest, roots, adapter, metallib, reused: true };
}

async function sealPreparationDirectories(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (entry.isDirectory()) await sealPreparationDirectories(path.join(directory, entry.name));
  }
  if (((await lstat(directory)).mode & 0o777) !== 0o500) await chmod(directory, 0o500);
}

async function writePreparedManifest(
  stage, key, identity, preparedFrom, builtArtifacts, signal, tier = "q4",
) {
  if (!/^[0-9a-f]{40}$/.test(preparedFrom.sceneWorksRevision)
      || !/^[0-9a-f]{40}$/.test(preparedFrom.inferenceRevision)) {
    fail("prepared canary source revisions must be exact commits");
  }
  const roots = privateArtifactRoots(stage, tier);
  const artifacts = {};
  for (const [name, root] of Object.entries(roots)) {
    const content = assertInventory(
      builtArtifacts?.[name], identity.artifact[name], `new prepared ${name}`,
    );
    const seal = await sealedArtifactIdentity(root);
    artifacts[name] = {
      content: { files: content.files, bytes: content.bytes, sha256: content.sha256 },
      seal,
    };
  }
  const adapter = await adapterIdentity(path.join(stage, "adapter", "memory-mlx-adapter"), signal);
  const metallib = await metallibIdentity(path.join(stage, "adapter", "mlx.metallib"), signal);
  const manifest = {
    schemaVersion: PREPARATION_SCHEMA_VERSION,
    key,
    identity,
    preparedFrom,
    artifacts,
    adapter: preparedFileManifestIdentity(adapter),
    metallib: preparedFileManifestIdentity(metallib),
  };
  const bytes = `${JSON.stringify(manifest, null, 2)}\n`;
  const manifestPath = path.join(stage, "prepared.json");
  const completionPath = path.join(stage, "prepared.sha256");
  await writeFile(manifestPath, bytes, { flag: "wx", mode: 0o400 });
  await writeFile(
    completionPath,
    `${createHash("sha256").update(bytes).digest("hex")}\n`,
    { flag: "wx", mode: 0o400 },
  );
  await chmod(manifestPath, 0o400);
  await chmod(completionPath, 0o400);
  await sealPreparationDirectories(stage);
  return manifest;
}

function validProcessIdentity(identity) {
  return identity !== null && typeof identity === "object"
    && typeof identity.startIdentity === "string" && identity.startIdentity !== ""
    && typeof identity.executable === "string" && identity.executable !== "";
}

export async function osProcessIdentity(pid, signal) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return null;
  let stdout;
  try {
    ({ stdout } = await execFileAsync(
      "/bin/ps", ["-p", String(pid), "-o", "pid=,lstart=,comm="],
      { encoding: "utf8", timeout: 2_000, signal },
    ));
  } catch (error) {
    if (error.code === 1) return null;
    throw error;
  }
  const lines = stdout.trim() ? stdout.trim().split("\n") : [];
  if (lines.length === 0) return null;
  if (lines.length !== 1) fail(`ps returned multiple identities for process ${pid}`);
  const fields = lines[0].trim().split(/\s+/);
  if (fields.length < 7 || Number(fields[0]) !== pid) {
    fail(`ps did not return a complete start identity for process ${pid}`);
  }
  const identity = {
    // BSD and procps both spell lstart as five whitespace-separated fields. It is an opaque OS
    // process-birth token here; owner.json's wall-clock startedAt is diagnostic only.
    startIdentity: fields.slice(1, 6).join(" "),
    executable: fields.slice(6).join(" "),
  };
  if (!validProcessIdentity(identity)) fail(`ps returned an invalid identity for process ${pid}`);
  return identity;
}

async function abortableDelay(milliseconds, signal) {
  signal?.throwIfAborted();
  await new Promise((resolve, reject) => {
    const finish = () => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    };
    const timer = setTimeout(finish, milliseconds);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal.reason ?? new Error("preparation lock wait aborted"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export async function acquirePreparationLock(
  preparationRoot, signal, processIdentityProbe = osProcessIdentity,
) {
  const lock = preparationLockPath(preparationRoot);
  const token = randomUUID();
  while (true) {
    signal?.throwIfAborted();
    let acquired = false;
    try {
      await mkdir(lock, { mode: 0o700 });
      acquired = true;
    } catch (error) {
      if (error.code !== "EEXIST") throw error;
    }
    if (acquired) {
      try {
        const processIdentity = await processIdentityProbe(process.pid, signal);
        if (!validProcessIdentity(processIdentity)) {
          fail("cannot bind prepared canary lock to this process start identity");
        }
        await writeFile(
          path.join(lock, "owner.json"),
          `${JSON.stringify({
            schemaVersion: 2,
            pid: process.pid,
            processIdentity,
            token,
            startedAt: Date.now(),
          })}\n`,
          { flag: "wx", mode: 0o400 },
        );
      } catch (error) {
        await cleanupCanaryScratch(lock);
        throw error;
      }
      return async () => {
        const owner = JSON.parse(await readFile(path.join(lock, "owner.json"), "utf8"));
        const actualIdentity = await processIdentityProbe(process.pid);
        if (owner.token !== token || owner.pid !== process.pid
            || !sameJson(owner.processIdentity, actualIdentity)) {
          fail("prepared canary lock ownership changed");
        }
        await cleanupCanaryScratch(lock);
      };
    }
    const ownerPath = path.join(lock, "owner.json");
    let owner = null;
    try {
      owner = JSON.parse(await readFile(ownerPath, "utf8"));
    } catch {
      // A process can be killed between creating its lock directory and completing owner.json.
      // Recover only after the orphan grace below, never while a live owner can still publish.
    }
    const lockMetadata = await lstat(lock, { bigint: true }).catch((error) => {
      if (error.code === "ENOENT") return null;
      throw error;
    });
    if (lockMetadata === null) continue;
    const oldEnough = Date.now() - Number(lockMetadata.mtimeMs) >= PREPARATION_LOCK_ORPHAN_GRACE_MS;
    const verifiableOwner = owner?.schemaVersion === 2
      && Number.isSafeInteger(owner.pid) && owner.pid > 0
      && validProcessIdentity(owner.processIdentity);
    const observedIdentity = verifiableOwner
      ? await processIdentityProbe(owner.pid, signal) : null;
    const staleOwner = verifiableOwner
      ? !sameJson(observedIdentity, owner.processIdentity) : oldEnough;
    if (staleOwner) {
      const stale = `${lock}.stale-${randomUUID()}`;
      try {
        await rename(lock, stale);
        await cleanupCanaryScratch(stale);
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
      continue;
    }
    await abortableDelay(PREPARATION_LOCK_POLL_MS, signal);
  }
}

async function cleanupStalePreparationStages(preparationRoot) {
  const parent = path.dirname(preparationRoot);
  const prefix = `.${path.basename(preparationRoot)}.stage-`;
  for (const entry of await readdir(parent)) {
    if (entry.startsWith(prefix)) await cleanupCanaryScratch(path.join(parent, entry));
  }
}

export async function prepareCanaryCache({
  preparationRoot,
  key,
  identity,
  preparedFrom,
  build,
  signal,
  hooks = {},
  processIdentityProbe = osProcessIdentity,
  tier = "q4",
}) {
  const parent = path.dirname(preparationRoot);
  await mkdir(parent, { recursive: true, mode: 0o700 });
  await exactMetadata(parent, "directory", 0o700);
  const alreadyPrepared = await validatePreparedCache(preparationRoot, key, identity, signal, tier);
  if (alreadyPrepared !== null) return alreadyPrepared;
  const release = await acquirePreparationLock(preparationRoot, signal, processIdentityProbe);
  let stage = null;
  let operationError = null;
  let cleanupError = null;
  try {
    const wonRace = await validatePreparedCache(preparationRoot, key, identity, signal, tier);
    if (wonRace !== null) return wonRace;
    if (await pathExists(preparationRoot)) await cleanupCanaryScratch(preparationRoot);
    await cleanupStalePreparationStages(preparationRoot);
    stage = await mkdtemp(path.join(parent, `.${key}.stage-`));
    await hooks.stageCreated?.(stage, signal);
    const built = await build(stage, signal, hooks);
    signal?.throwIfAborted();
    await hooks.buildComplete?.(stage, signal);
    await writePreparedManifest(stage, key, identity, preparedFrom, built?.artifacts, signal, tier);
    signal?.throwIfAborted();
    await hooks.beforePublish?.(stage, signal);
    signal?.throwIfAborted();
    await rename(stage, preparationRoot);
    stage = null;
    const prepared = await validatePreparedCache(preparationRoot, key, identity, signal, tier);
    if (prepared === null) fail("atomic prepared canary publication is incomplete");
    return { ...prepared, reused: false };
  } catch (error) {
    operationError = error;
  } finally {
    try {
      if (stage !== null) await cleanupCanaryScratch(stage);
      await release();
    } catch (error) {
      cleanupError = error;
    }
  }
  throw preservePrimaryFailure(operationError, cleanupError);
}

export async function inferenceCargoSource(
  cargo, cargoEnv, inferenceRepo, inferenceRevision, signal,
) {
  const metadata = JSON.parse((await runOwnedCommand(cargo, [
    "metadata", "--locked", "--format-version", "1",
  ], {
    cwd: ROOT, env: cargoEnv, timeout: CARGO_METADATA_TIMEOUT_MS,
    maxBuffer: 50 * 1024 * 1024, signal,
  })).stdout);
  const packages = metadata.packages.filter((pkg) =>
    pkg.source?.startsWith("git+https://github.com/SceneWorks/inference"));
  if (!packages.length || packages.some((pkg) => !pkg.source.endsWith(`#${inferenceRevision}`))) {
    fail("Cargo metadata does not resolve every inference package at the exact pin");
  }
  const roots = new Set();
  for (const pkg of packages) {
    roots.add(await git(
      path.dirname(pkg.manifest_path), ["rev-parse", "--show-toplevel"], signal,
    ));
  }
  if (roots.size !== 1) fail("Cargo resolved inference packages from multiple source checkouts");
  const cargoSource = [...roots][0];
  if (await cleanCargoSourceHead(cargoSource, signal) !== inferenceRevision) {
    fail("Cargo inference source checkout does not match the exact pin");
  }
  const [cargoTree, verifiedTree] = await Promise.all([
    git(cargoSource, ["rev-parse", "HEAD^{tree}"], signal),
    git(inferenceRepo, ["rev-parse", "HEAD^{tree}"], signal),
  ]);
  if (cargoTree !== verifiedTree) fail("Cargo inference source tree differs from the verified checkout");
  return cargoSource;
}

async function preparedFileIdentity(filePath, expectedMode, label, signal) {
  const metadata = await lstat(filePath, { bigint: true });
  if (!metadata.isFile() || metadata.isSymbolicLink()
      || Number(metadata.mode & 0o777n) !== expectedMode) {
    fail(`prepared canary ${label} must be a read-only regular file with mode ${expectedMode.toString(8)}`);
  }
  return {
    path: filePath,
    sha256: await sha256(filePath, signal),
    device: metadata.dev.toString(),
    inode: metadata.ino.toString(),
    size: Number(metadata.size),
    mtimeNs: metadata.mtimeNs.toString(),
    ctimeNs: metadata.ctimeNs.toString(),
    mode: Number(metadata.mode & 0o777n),
  };
}

async function adapterIdentity(adapterPath, signal) {
  return preparedFileIdentity(adapterPath, 0o500, "adapter executable", signal);
}

async function metallibIdentity(metallibPath, signal) {
  return preparedFileIdentity(metallibPath, 0o400, "metallib", signal);
}

async function assertAdapterIdentity(expected, signal) {
  const actual = await adapterIdentity(expected.path, signal);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    fail("canary adapter identity changed after its exact build");
  }
}

async function assertMetallibIdentity(expected, signal) {
  const actual = await metallibIdentity(expected.path, signal);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    fail("canary metallib identity changed after its exact build");
  }
}

async function assertRuntimeAssetIdentities(adapter, metallib, signal) {
  await Promise.all([
    assertAdapterIdentity(adapter, signal),
    assertMetallibIdentity(metallib, signal),
  ]);
}

export async function locateBuiltMetallib(target) {
  const buildDirectory = path.join(target, "release", "build");
  const metallibs = [];
  for (const entry of await readdir(buildDirectory, { withFileTypes: true })) {
    if (!entry.name.startsWith("pmetal-mlx-sys-") || !entry.isDirectory()) continue;
    const candidate = path.join(buildDirectory, entry.name, "out", "build", "lib", "mlx.metallib");
    const metadata = await lstat(candidate).catch((error) => {
      if (error.code === "ENOENT") return null;
      throw error;
    });
    if (metadata === null) continue;
    if (!metadata.isFile() || metadata.isSymbolicLink()) {
      fail(`built pmetal metallib is not a regular file: ${candidate}`);
    }
    metallibs.push(candidate);
  }
  if (metallibs.length !== 1) {
    fail(`exactly one built pmetal metallib is required, found ${metallibs.length}`);
  }
  return metallibs[0];
}

async function buildExactAdapter(
  sceneWorksRevision, inferenceRepo, inferenceRevision, buildRoot, preparationRoot, signal,
) {
  await mkdir(buildRoot, { mode: 0o700 });
  const toolchain = await exactToolchain(buildRoot, signal);
  const { cargo } = toolchain;
  const cargoEnv = toolchain.env;
  const target = cargoEnv.CARGO_TARGET_DIR;
  const cargoSource = await inferenceCargoSource(
    cargo, cargoEnv, inferenceRepo, inferenceRevision, signal,
  );
  await runOwnedCommand(cargo, [
    "build", "--locked", "--release", "-p", "sceneworks-memory-adapter",
    "--bin", "memory-mlx-adapter", "--features", "mlx",
  ], {
    cwd: ROOT,
    env: cargoEnv,
    timeout: 30 * 60 * 1000,
    maxBuffer: 10 * 1024 * 1024,
    signal,
  });
  if (await cleanHead(ROOT, "SceneWorks", signal) !== sceneWorksRevision) {
    fail("SceneWorks HEAD changed while building the canary adapter");
  }
  if (await cleanHead(inferenceRepo, "inference", signal) !== inferenceRevision) {
    fail("verified inference checkout changed while building the canary adapter");
  }
  if (await cleanCargoSourceHead(cargoSource, signal) !== inferenceRevision) {
    fail("Cargo inference source changed while building the canary adapter");
  }
  await inferenceCargoSource(cargo, cargoEnv, inferenceRepo, inferenceRevision, signal);
  const built = path.join(target, "release/memory-mlx-adapter");
  const builtMetallib = await locateBuiltMetallib(target);
  const adapterDirectory = path.join(preparationRoot, "adapter");
  const adapterPath = path.join(adapterDirectory, "memory-mlx-adapter");
  const metallibPath = path.join(adapterDirectory, "mlx.metallib");
  await mkdir(adapterDirectory, { mode: 0o700 });
  await copyFile(built, adapterPath, fsConstants.COPYFILE_EXCL);
  await copyFile(builtMetallib, metallibPath, fsConstants.COPYFILE_EXCL);
  await chmod(adapterPath, 0o500);
  await chmod(metallibPath, 0o400);
  await chmod(adapterDirectory, 0o500);
  return Promise.all([
    adapterIdentity(adapterPath, signal),
    metallibIdentity(metallibPath, signal),
  ]);
}

export function canaryWatchdogEnvironment(baseEnvironment, roots, metallibPath, privateHome) {
  if (!path.isAbsolute(metallibPath)) fail("prepared canary metallib path must be absolute");
  if (!path.isAbsolute(privateHome)) fail("private canary runtime HOME must be absolute");
  const environment = { ...baseEnvironment };
  for (const name of ["HOME", "PMETAL_METALLIB_PATH", "PMETAL_CACHE_DIR", "XDG_CACHE_HOME"]) {
    delete environment[name];
  }
  return {
    ...environment,
    HOME: privateHome,
    // The pinned pmetal resolver reads this before its mutable user cache. Keep the override after
    // the inherited environment so a concurrent build can never redirect this prepared adapter.
    PMETAL_METALLIB_PATH: metallibPath,
    SCENEWORKS_LTX_ROOT: roots.numericTier,
    SCENEWORKS_LTX_TEXT_ENCODER_ROOT: roots.textEncoder,
  };
}

async function readStandardInput() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

async function runProtocolCommand(executable, input, { env, signal, cwd = ROOT } = {}) {
  signal?.throwIfAborted();
  return new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    let spawnError = null;
    let stdinError = null;
    const child = spawn(executable, [], {
      cwd, env, detached: true, stdio: ["pipe", "pipe", "pipe"],
    });
    const terminate = () => killProcessGroup(child, "SIGKILL");
    const onAbort = () => terminate();
    if (signal) signal.addEventListener("abort", onAbort, { once: true });
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => { spawnError = error; terminate(); });
    child.stdin.on("error", (error) => { stdinError = error; });
    child.on("close", (code, childSignal) => {
      if (signal) signal.removeEventListener("abort", onAbort);
      if (spawnError) return reject(spawnError);
      if (stdinError) return reject(stdinError);
      if (signal?.aborted) return reject(signal.reason ?? new Error("protocol command aborted"));
      return code === 0 && childSignal === null
        ? resolve(stdout)
        : reject(new Error(
          `campaign adapter failed: code=${code} signal=${childSignal}: ${stderr.trim() || "no stderr"}`,
        ));
    });
    child.stdin.end(input);
  });
}

export function ownedProcessGroupResidue(processTable, pgid) {
  if (!Number.isSafeInteger(pgid) || pgid <= 0) fail("watchdog PGID must be a positive integer");
  return processTable.split("\n").map((line) => line.trim()).filter(Boolean).filter((line) => {
    const match = line.match(/^(\d+)\s+(\d+)\s+(.+)$/);
    return match !== null && Number(match[2]) === pgid;
  });
}

async function assertRunRootEmpty(runRoot) {
  const entries = await readdir(runRoot).catch((error) => {
    if (error.code === "ENOENT") return [];
    throw error;
  });
  if (entries.length !== 0) fail(`SC-20191 run scratch retained residue: ${entries.join(", ")}`);
}

async function assertCampaignSourceState(options, signal, operation = "SC-20191 campaign entry") {
  const pairs = [
    [ROOT, "SceneWorks", options["scene-revision"], options["scene-tree"]],
    [options["inference-repo"], "inference", options["inference-revision"], options["inference-tree"]],
  ];
  for (const [repo, label, revision, tree] of pairs) {
    if (await cleanHead(repo, label, signal) !== revision
        || await git(repo, ["rev-parse", "HEAD^{tree}"], signal) !== tree) {
      fail(`${label} source changed during ${operation}`);
    }
  }
}

function campaignProviderOptions(argv) {
  const options = parseArgs(argv);
  for (const name of [
    "inference-repo", "preparation-root", "preparation-key", "run-root",
    "scene-revision", "scene-tree", "inference-revision", "inference-tree",
    "canonical-output", "failure-output", "outcome-reservation", "outcome-token",
  ]) {
    if (!options[name]) fail(`campaign provider requires --${name}`);
  }
  const unexpected = Object.keys(options).filter((name) => ![
    "inference-repo", "preparation-root", "preparation-key", "run-root",
    "scene-revision", "scene-tree", "inference-revision", "inference-tree",
    "canonical-output", "failure-output", "outcome-reservation", "outcome-token",
  ].includes(name));
  if (unexpected.length) fail(`unsupported campaign provider option(s): ${unexpected.join(", ")}`);
  for (const name of [
    "inference-repo", "preparation-root", "run-root", "canonical-output", "failure-output",
    "outcome-reservation",
  ]) {
    options[name] = path.resolve(options[name]);
  }
  return options;
}

async function campaignProviderInvocation(options, input, {
  signal, setActiveWatchdog = () => {}, canonicalClaim, executionClaim,
  bounded = false, boundedTier = null, q8ReleaseAuthorization = null,
} = {}) {
  signal?.throwIfAborted();
  const boundedEntry = bounded || boundedTier !== null;
  const boundedSpec = boundedEntry ? boundedCampaignEntrySpec(boundedTier ?? "q4") : null;
  const spec = boundedEntry ? {
    story: boundedSpec.story.toUpperCase(),
    plan: BOUNDED_CAMPAIGN_ENTRY_PLAN,
    planEntry: (config) => boundedCampaignEntryPlan(config, boundedSpec.tier),
    failurePath: (output) => boundedCampaignEntryFailurePath(output, boundedSpec.tier),
    outcomePath: (output) => boundedCampaignEntryOutcomeReservationPath(output, boundedSpec.tier),
    adapterRequest: (request, planned, source) =>
      boundedCampaignEntryAdapterRequest(request, planned, source, boundedSpec.tier),
    validateResponse: validateBoundedCampaignEntryResponse,
    canonicalFragment: boundedCampaignEntryCanonicalFragment,
    phases: BOUNDED_CARRIER_PHASES,
    phaseProfile: "bounded-campaign-entry",
    childName: `${boundedSpec.story}-${boundedSpec.tier}-bounded-campaign-entry`,
  } : {
    story: "SC-20191",
    plan: CAMPAIGN_ENTRY_PLAN,
    planEntry: campaignEntryPlan,
    failurePath: campaignEntryFailurePath,
    outcomePath: campaignEntryOutcomeReservationPath,
    adapterRequest: campaignEntryAdapterRequest,
    validateResponse: validateCampaignEntryAdapterResponse,
    canonicalFragment: campaignEntryCanonicalFragment,
    phases: PROVIDER_PHASES,
    phaseProfile: "campaign-entry",
    childName: "sc-20191-campaign-entry",
  };
  const request = JSON.parse(input);
  const campaignOutcome = {
    canonicalOutput: options["canonical-output"],
    failureOutput: options["failure-output"],
    reservation: options["outcome-reservation"],
    choice: `${options["outcome-reservation"]}.choice`,
    token: options["outcome-token"],
    canonicalClaim,
    executionClaim,
    q8ReleaseAuthorization,
  };
  if (campaignOutcome.failureOutput !== spec.failurePath(campaignOutcome.canonicalOutput)
      || campaignOutcome.reservation !== spec.outcomePath(campaignOutcome.canonicalOutput)) {
    fail(`${spec.story} campaign provider outcome paths changed`);
  }
  const config = JSON.parse(await readFile(path.join(ROOT, spec.plan), "utf8"));
  const { planned } = spec.planEntry(config);
  const toolchainChannel = await repositoryToolchain();
  const preparationTier = boundedSpec?.tier ?? "q4";
  const identity = preparationIdentity(
    options["scene-tree"], options["inference-tree"], toolchainChannel,
    preparationTier,
  );
  if (preparationCacheKey(identity) !== options["preparation-key"]) {
    fail("SC-20191 campaign provider preparation key changed");
  }
  await assertCampaignSourceState(options, signal);
  const prepared = await validatePreparedCache(
    options["preparation-root"], options["preparation-key"], identity, signal, preparationTier,
  );
  if (prepared === null) fail("SC-20191 prepared cache disappeared before provider invocation");
  await assertRuntimeAssetIdentities(prepared.adapter, prepared.metallib, signal);
  await mkdir(options["run-root"], { recursive: true, mode: 0o700 });
  await assertRunRootEmpty(options["run-root"]);
  const runScratch = await mkdtemp(path.join(options["run-root"], "sc-20191-run-"));
  let operationError = null;
  let cleanupError = null;
  let failureEvents = null;
  let failureHostMemoryBytes = null;
  let output = null;
  try {
    const runtimeHome = path.join(runScratch, "home");
    await mkdir(runtimeHome, { mode: 0o700 });
    await exactMetadata(runtimeHome, "directory", 0o700);
    await exactEntries(runtimeHome, []);
    const environment = canaryWatchdogEnvironment(
      process.env, prepared.roots, prepared.metallib.path, runtimeHome,
    );
    if (request.action === "probe") {
      const probeOutput = await runProtocolCommand(prepared.adapter.path, input, {
        env: environment, signal,
      });
      JSON.parse(probeOutput);
      await assertCampaignSourceState(options, signal);
      await assertRuntimeAssetIdentities(prepared.adapter, prepared.metallib, signal);
      if (await validatePreparedCache(
        options["preparation-root"], options["preparation-key"], identity, signal, preparationTier,
      ) === null) fail("SC-20191 prepared cache disappeared after provider probe");
      output = probeOutput;
    } else {
      const adapterRequest = spec.adapterRequest(request, planned, {
        sceneWorksRevision: options["scene-revision"],
        inferenceRevision: options["inference-revision"],
        sceneWorksRepo: ROOT,
        inferenceRepo: options["inference-repo"],
      });
      const requestPath = path.join(runScratch, "request.json");
      const responsePath = path.join(runScratch, "response.json");
      const eventsPath = path.join(runScratch, "watchdog.jsonl");
      await writeFile(requestPath, canonicalJson(adapterRequest), { flag: "wx" });
      const hostMemoryBytes = request.hardware.memoryBytes;
      const runtimeMemoryFreeFloor = runtimeFreeFloor(hostMemoryBytes);
      const watchdogArgs = [
        path.join(ROOT, "scripts/memory-calibration-watchdog.py"),
        "--max-footprint-bytes", String(MAX_FOOTPRINT_BYTES),
        "--max-runtime-seconds", String(MAX_RUNTIME_SECONDS),
        "--host-memory-bytes", String(hostMemoryBytes),
        "--min-memory-free-bytes", String(runtimeMemoryFreeFloor),
        "--sample-interval", "0.25",
        "--telemetry-timeout", "1",
        "--child-attestation-timeout", String(CHILD_ATTESTATION_TIMEOUT_SECONDS),
        "--term-grace", "1",
        "--event-file", eventsPath,
        "--require-child-attestation",
        "--require-provider-phases",
        "--provider-phase-profile", spec.phaseProfile,
        "--",
        "/bin/sh", "-c", 'set -C; exec "$1" <"$2" >"$3"',
        spec.childName, prepared.adapter.path, requestPath, responsePath,
      ];
      await assertCampaignSourceState(options, signal);
      await assertRuntimeAssetIdentities(prepared.adapter, prepared.metallib, signal);
      if (await validatePreparedCache(
        options["preparation-root"], options["preparation-key"], identity, signal, preparationTier,
      ) === null) fail("SC-20191 prepared cache disappeared before model release");
      if (boundedSpec?.tier === "bf16") {
        await assertQ8ReleaseAuthorization(
          q8ReleaseAuthorization, options["q8-audit"], {
            sceneWorksRevision: options["scene-revision"],
            inferenceRevision: options["inference-revision"],
          },
        );
      }
      // This immediate actual-GPU-owner and two-boundary pressure check is deliberately the final
      // operation before the watchdog takes ownership of the model process.
      await assertHostPreflight(hostMemoryBytes, signal);
      const status = await new Promise((resolve, reject) => {
        const child = spawn("/usr/bin/python3", watchdogArgs, {
          stdio: "inherit", env: environment,
        });
        setActiveWatchdog(child);
        const onAbort = () => child.kill(signal.reason?.signalName ?? "SIGTERM");
        if (signal) signal.addEventListener("abort", onAbort, { once: true });
        child.on("error", reject);
        child.on("close", (code, childSignal) => {
          if (signal) signal.removeEventListener("abort", onAbort);
          setActiveWatchdog(null);
          resolve({ code, signal: childSignal });
        });
      });
      const eventBytes = await readFile(eventsPath, "utf8").catch(() => "");
      if (status.code !== 0 || status.signal) {
        const watchdogError = new Error(watchdogFailureSummary(status, eventBytes));
        try {
          const events = eventBytes.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
          validateCampaignEntryFailureEvents(events, hostMemoryBytes, spec.phases);
          const started = events.find((event) => event.event === "started");
          const processTable = (await execFileAsync(
            "/bin/ps", ["-ww", "-axo", "pid=,pgid=,command="],
            { encoding: "utf8", timeout: 2_000 },
          )).stdout;
          const residue = ownedProcessGroupResidue(processTable, started?.pgid);
          if (residue.length) fail(`SC-20216 watchdog process group retained residue:\n${residue.join("\n")}`);
          await assertCampaignSourceState(options);
          await assertRuntimeAssetIdentities(prepared.adapter, prepared.metallib);
          if (await validatePreparedCache(
            options["preparation-root"], options["preparation-key"], identity, undefined,
            preparationTier,
          ) === null) fail("SC-20216 prepared cache disappeared after watchdog failure");
          failureEvents = events;
          failureHostMemoryBytes = hostMemoryBytes;
        } catch (error) {
          throw preserveFailureReceiptSuppression(watchdogError, error);
        }
        throw watchdogError;
      }
      signal?.throwIfAborted();
      const response = JSON.parse(await readFile(responsePath, "utf8"));
      const events = eventBytes.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
      spec.validateResponse(response, {
        inferenceRevision: options["inference-revision"], hostMemoryBytes,
        tier: boundedSpec?.tier ?? "q4",
      });
      const maxObservedFootprintBytes = validateCampaignEntryWatchdogEvents(
        events, hostMemoryBytes, spec.phases,
      );
      const started = events.find((event) => event.event === "started");
      const processTable = (await execFileAsync(
        "/bin/ps", ["-ww", "-axo", "pid=,pgid=,command="],
        { encoding: "utf8", timeout: 2_000 },
      )).stdout;
      const residue = ownedProcessGroupResidue(processTable, started?.pgid);
      if (residue.length) fail(`SC-20191 watchdog process group retained residue:\n${residue.join("\n")}`);
      await assertCampaignSourceState(options, signal);
      await assertRuntimeAssetIdentities(prepared.adapter, prepared.metallib, signal);
      if (await validatePreparedCache(
        options["preparation-root"], options["preparation-key"], identity, signal, preparationTier,
      ) === null) fail("SC-20191 prepared cache disappeared after campaign entry");
      output = canonicalJson(spec.canonicalFragment(response, maxObservedFootprintBytes));
    }
  } catch (error) {
    operationError = error;
  } finally {
    try {
      await cleanupCanaryScratch(runScratch);
      await assertRunRootEmpty(options["run-root"]);
    } catch (error) {
      cleanupError = error;
    }
  }
  const finalError = preservePrimaryFailure(operationError, cleanupError);
  if (finalError !== null) {
    let publicationError = null;
    if (failureEvents !== null && cleanupError === null) {
      try {
        const publishFailure = boundedEntry
          ? publishBoundedCampaignEntryFailure : publishCampaignEntryFailureReceipt;
        await publishFailure(campaignOutcome, {
          verify: async () => {
            await assertRunRootEmpty(options["run-root"]);
            await assertCampaignSourceState(options);
            await assertRuntimeAssetIdentities(prepared.adapter, prepared.metallib);
            if (await validatePreparedCache(
              options["preparation-root"], options["preparation-key"], identity, undefined,
              preparationTier,
            ) === null) fail("SC-20216 prepared cache disappeared before failure publication");
            await assertPreparationHasNoTransientResidue(options["preparation-root"]);
            if (boundedSpec?.tier === "bf16") {
              await assertQ8ReleaseAuthorization(
                q8ReleaseAuthorization, options["q8-audit"], {
                  sceneWorksRevision: options["scene-revision"],
                  inferenceRevision: options["inference-revision"],
                },
              );
            }
          },
          build: (outcome) => (boundedEntry
            ? boundedCampaignEntryFailureReceipt({
              sceneWorksRevision: options["scene-revision"],
              sceneWorksTree: options["scene-tree"],
              inferenceRevision: options["inference-revision"],
              inferenceTree: options["inference-tree"],
              identity,
              preparationKey: options["preparation-key"],
              preparationRoot: options["preparation-root"],
              prepared,
              hostMemoryBytes: failureHostMemoryBytes,
              events: failureEvents,
              outcome,
              tier: boundedSpec.tier,
            })
            : campaignEntryFailureReceipt({
            sceneWorksRevision: options["scene-revision"],
            sceneWorksTree: options["scene-tree"],
            inferenceRevision: options["inference-revision"],
            inferenceTree: options["inference-tree"],
            identity,
            preparationKey: options["preparation-key"],
            preparationRoot: options["preparation-root"],
            prepared,
            hostMemoryBytes: failureHostMemoryBytes,
            events: failureEvents,
            outcome,
          })),
        });
      } catch (error) {
        publicationError = error;
      }
    }
    if (publicationError !== null) {
      throw preserveFailureReceiptSuppression(finalError, publicationError);
    }
    throw finalError;
  }
  return output;
}

async function campaignProvider(argv) {
  const output = await campaignProviderInvocation(
    campaignProviderOptions(argv), await readStandardInput(),
  );
  process.stdout.write(output);
}

async function boundedSelectorReportController(argv) {
  const options = parseArgs(argv);
  const expected = ["q4-bundle", "q8-bundle", "bf16-bundle", "output"];
  if (!isDeepStrictEqual(Object.keys(options).sort(), [...expected].sort())) {
    fail(`bounded-selector report requires exactly ${expected.map((name) => `--${name}`).join(", ")}`);
  }
  await publishBoundedSelectorAnchorReport({
    q4Bundle: options["q4-bundle"],
    q8Bundle: options["q8-bundle"],
    bf16Bundle: options["bf16-bundle"],
    output: options.output,
  });
  process.stdout.write(`${path.resolve(options.output)}\n`);
}

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    if (!name?.startsWith("--") || value === undefined) fail(`invalid argument ${name ?? ""}`);
    options[name.slice(2)] = value;
  }
  return options;
}

async function publishExclusiveJson(output, value, signal) {
  await mkdir(path.dirname(output), { recursive: true });
  signal?.throwIfAborted();
  const temporary = `${output}.tmp-${process.pid}-${randomUUID()}`;
  try {
    await writeFile(temporary, canonicalJson(value), { flag: "wx" });
    signal?.throwIfAborted();
    await link(temporary, output);
  } finally {
    await unlink(temporary).catch((error) => { if (error.code !== "ENOENT") throw error; });
  }
}

export async function acquireCampaignEntryOutcome(canonicalOutput) {
  const outcome = {
    canonicalOutput,
    failureOutput: campaignEntryFailurePath(canonicalOutput),
    reservation: campaignEntryOutcomeReservationPath(canonicalOutput),
    choice: `${campaignEntryOutcomeReservationPath(canonicalOutput)}.choice`,
    token: randomUUID(),
  };
  await mkdir(path.dirname(canonicalOutput), { recursive: true });
  await writeFile(outcome.reservation, `${outcome.token}\n`, { flag: "wx", mode: 0o400 });
  if (await pathExists(outcome.canonicalOutput) || await pathExists(outcome.failureOutput)) {
    await unlink(outcome.reservation);
    fail("SC-20216 campaign entry already has an immutable canonical or failure outcome");
  }
  return outcome;
}

async function assertCampaignEntryOutcomeOwner(outcome) {
  if (await readFile(outcome?.reservation ?? "", "utf8").catch(() => null)
      !== `${outcome?.token}\n`) {
    fail("SC-20216 campaign entry outcome reservation ownership changed");
  }
}

export async function releaseUnpublishedCampaignEntryOutcome(outcome) {
  await assertCampaignEntryOutcomeOwner(outcome);
  if (await pathExists(outcome.choice)
      || await pathExists(outcome.canonicalOutput) || await pathExists(outcome.failureOutput)) return;
  await unlink(outcome.reservation);
}

async function publishCampaignEntryOutcome(outcome, kind, build, signal) {
  await assertCampaignEntryOutcomeOwner(outcome);
  if (typeof outcome?.canonicalClaim?.assertOwner !== "function") {
    fail("SC-20216 publication requires the shared canonical claim");
  }
  await outcome.canonicalClaim.assertOwner();
  const target = kind === "canonical" ? outcome.canonicalOutput : outcome.failureOutput;
  const other = kind === "canonical" ? outcome.failureOutput : outcome.canonicalOutput;
  await writeFile(outcome.choice, `${kind}\n`, { flag: "wx", mode: 0o400 });
  if (await pathExists(target) || await pathExists(other)) {
    fail("SC-20216 campaign entry outcome is already fixed");
  }
  const value = await build();
  await publishExclusiveJson(target, value, signal);
  return value;
}

export async function publishCampaignEntryCanonicalOutcome(outcome, bundle, signal) {
  await publishCampaignEntryOutcome(outcome, "canonical", async () => bundle, signal);
}

export async function publishCampaignEntryFailureReceipt(outcome, { verify, build }) {
  if (typeof verify !== "function" || typeof build !== "function") {
    fail("SC-20216 failure publication requires validation and receipt builders");
  }
  await verify();
  return publishCampaignEntryOutcome(outcome, "failure", async () => {
    if (await pathExists(outcome.canonicalOutput)) {
      fail("SC-20216 canonical outcome already exists; failure receipt suppressed");
    }
    return validateCampaignEntryFailureReceipt(await build({
      canonicalOutput: outcome.canonicalOutput,
      failureOutput: outcome.failureOutput,
      reservation: outcome.reservation,
      choice: outcome.choice,
      outcomeChoice: "failure",
      canonicalBundleAbsentAtPublication: true,
      outcomeReservationHeldAtPublication: true,
    }));
  });
}

export async function acquireBoundedCampaignEntryOutcome(
  canonicalOutput, requested = "q4", q8ReleaseAuthorization = null,
) {
  const spec = boundedCampaignEntrySpec(requested);
  if ((spec.tier === "bf16") !== (q8ReleaseAuthorization !== null)) {
    fail("SC-20430 bf16 outcome must bind exactly one q8 release authorization");
  }
  const outcome = {
    canonicalOutput,
    failureOutput: boundedCampaignEntryFailurePath(canonicalOutput, spec.tier),
    reservation: boundedCampaignEntryOutcomeReservationPath(canonicalOutput, spec.tier),
    choice: `${boundedCampaignEntryOutcomeReservationPath(canonicalOutput, spec.tier)}.choice`,
    token: randomUUID(),
    tier: spec.tier,
    q8ReleaseAuthorization: structuredClone(q8ReleaseAuthorization),
  };
  outcome.ownerBytes = spec.tier === "bf16"
    ? canonicalJson({ token: outcome.token, tier: spec.tier, q8ReleaseAuthorization })
    : `${outcome.token}\n`;
  await mkdir(path.dirname(canonicalOutput), { recursive: true });
  await writeFile(outcome.reservation, outcome.ownerBytes, { flag: "wx", mode: 0o400 });
  if (await pathExists(outcome.canonicalOutput) || await pathExists(outcome.failureOutput)) {
    await unlink(outcome.reservation);
    fail("SC-20318 already has an immutable canonical or failure outcome");
  }
  return outcome;
}

async function assertBoundedCampaignEntryOutcomeOwner(outcome) {
  if (await readFile(outcome?.reservation ?? "", "utf8").catch(() => null)
      !== outcome?.ownerBytes) fail("SC-20318 outcome reservation ownership changed");
}

export async function releaseUnpublishedBoundedCampaignEntryOutcome(outcome) {
  await assertBoundedCampaignEntryOutcomeOwner(outcome);
  if (await pathExists(outcome.choice)
      || await pathExists(outcome.canonicalOutput) || await pathExists(outcome.failureOutput)) return;
  await unlink(outcome.reservation);
}

async function publishBoundedCampaignEntryOutcome(outcome, kind, build, signal) {
  await assertBoundedCampaignEntryOutcomeOwner(outcome);
  await outcome.canonicalClaim.assertOwner();
  await outcome.executionClaim.assertOwner();
  const target = kind === "canonical" ? outcome.canonicalOutput : outcome.failureOutput;
  const other = kind === "canonical" ? outcome.failureOutput : outcome.canonicalOutput;
  await writeFile(outcome.choice, `${kind}\n`, { flag: "wx", mode: 0o400 });
  if (await pathExists(target) || await pathExists(other)) fail("SC-20318 outcome is already fixed");
  const value = await build();
  await publishExclusiveJson(target, value, signal);
  return value;
}

async function publishBoundedCampaignEntryFailure(outcome, { verify, build }) {
  await verify();
  return publishBoundedCampaignEntryOutcome(outcome, "failure", async () => {
    if (await pathExists(outcome.canonicalOutput)) fail("SC-20318 canonical exists; failure suppressed");
    return validateBoundedCampaignEntryFailureReceipt(await build({
      canonicalOutput: outcome.canonicalOutput,
      failureOutput: outcome.failureOutput,
      reservation: outcome.reservation,
      choice: outcome.choice,
      outcomeChoice: "failure",
      canonicalBundleAbsentAtPublication: true,
      outcomeReservationHeldAtPublication: true,
      q8ReleaseAuthorization: structuredClone(outcome.q8ReleaseAuthorization),
    }), outcome.tier);
  });
}

export function canonicalPublicationClaimPath(canonicalOutput) {
  return path.join(
    path.dirname(canonicalOutput),
    `.${path.basename(canonicalOutput)}.ltx-canonical-publication-claim`,
  );
}

export function containedCampaignExecutionClaimTarget(
  _output, containmentRoot = CONTAINED_CAMPAIGN_ROOT,
) {
  return path.join(path.resolve(containmentRoot), ".sc-18946-contained-execution");
}

const CANONICAL_RECLAMATION_LOCK_HELPER = String.raw`
import base64
import fcntl
import os
import stat
import sys

lock_path = sys.argv[1]
lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
initial_stat = os.fstat(lock_fd)
if (
    not stat.S_ISREG(initial_stat.st_mode)
    or initial_stat.st_nlink != 1
    or stat.S_IMODE(initial_stat.st_mode) != 0o600
):
    raise RuntimeError("reclamation lock is not an exact private regular file")
fcntl.flock(lock_fd, fcntl.LOCK_EX)
lock_stat = os.fstat(lock_fd)
if lock_stat.st_size > 16384:
    raise RuntimeError("reclamation lock owner is oversized")
os.lseek(lock_fd, 0, os.SEEK_SET)
previous_owner = b""
while len(previous_owner) < lock_stat.st_size:
    previous_owner += os.read(lock_fd, lock_stat.st_size - len(previous_owner))
encoded_previous_owner = base64.b64encode(previous_owner).decode("ascii") or "-"
print(
    f"LOCKED {lock_stat.st_dev} {lock_stat.st_ino} {encoded_previous_owner}",
    flush=True,
)
owner_command = sys.stdin.readline().strip().split(" ")
if owner_command == ["RELEASE"]:
    sys.exit(0)
if owner_command[0] != "OWNER" or len(owner_command) != 2:
    raise RuntimeError("missing reclamation lock owner")
owner_bytes = base64.b64decode(owner_command[1])
os.fchmod(lock_fd, 0o600)
os.ftruncate(lock_fd, 0)
os.lseek(lock_fd, 0, os.SEEK_SET)
remaining = memoryview(owner_bytes)
while remaining:
    remaining = remaining[os.write(lock_fd, remaining):]
os.fsync(lock_fd)
owned_stat = os.fstat(lock_fd)
print(
    f"OWNED {owned_stat.st_dev} {owned_stat.st_ino} "
    f"{stat.S_IMODE(owned_stat.st_mode)} {owned_stat.st_size} {owned_stat.st_nlink}",
    flush=True,
)
command = sys.stdin.readline().strip().split(" ")
if command[0] == "RENAME" and len(command) == 3:
    source = base64.b64decode(command[1]).decode("utf-8")
    target = base64.b64decode(command[2]).decode("utf-8")
    os.rename(source, target)
    print("RENAMED", flush=True)
elif command != ["RELEASE"]:
    raise RuntimeError("invalid reclamation lock command")
`;

function encodedLockPath(value) {
  return Buffer.from(value, "utf8").toString("base64");
}

function encodedLockOwner(value) {
  return Buffer.from(`${JSON.stringify(value)}\n`, "utf8").toString("base64");
}

async function acquireCanonicalClaimReclamationMutex(
  claimPath, signal, processIdentityProbe,
) {
  // Keep one stable inode: unlinking it could split an existing waiter from a new opener. The
  // kernel lock, not the retained diagnostic owner bytes, is authoritative and is released when
  // this helper exits after an explicit command or loss of its parent pipe.
  const mutexPath = `${claimPath}.reclaiming`;
  const deadline = Date.now() + 2_000;
  signal?.throwIfAborted();
  const child = spawn("/usr/bin/python3", ["-c", CANONICAL_RECLAMATION_LOCK_HELPER, mutexPath], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  child.stderr.setEncoding("utf8");
  let stderr = "";
  child.stderr.on("data", (chunk) => { stderr += chunk; });
  const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
  const iterator = lines[Symbol.asyncIterator]();
  const childClosed = new Promise((resolve) => {
    child.once("close", (code, childSignal) => resolve([code, childSignal]));
  });
  let termination = null;
  const terminate = (error) => {
    if (termination !== null) return;
    termination = error;
    child.kill("SIGKILL");
  };
  const timeout = setTimeout(() => terminate(
    new Error("canonical claim reclamation serialization remained unavailable"),
  ), Math.max(0, deadline - Date.now()));
  const onAbort = () => terminate(signal.reason ?? new Error("canonical claim reclamation aborted"));
  signal?.addEventListener("abort", onAbort, { once: true });
  let locked;
  try {
    locked = await iterator.next();
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", onAbort);
  }
  if (termination !== null) {
    await childClosed;
    throw termination;
  }
  if (locked.done) fail(`canonical reclamation lock helper exited before ownership: ${stderr}`);
  const match = /^LOCKED ([0-9]+) ([0-9]+) ([A-Za-z0-9+/=]+|-)$/.exec(locked.value);
  if (match === null) fail("canonical reclamation lock helper returned invalid ownership");
  const filesystemIdentity = { device: match[1], inode: match[2] };
  const previousOwnerBytes = match[3] === "-"
    ? "" : Buffer.from(match[3], "base64").toString("utf8");
  if (previousOwnerBytes !== "") {
    let previousOwner;
    try {
      previousOwner = JSON.parse(previousOwnerBytes);
    } catch {
      previousOwner = null;
    }
    if (previousOwner?.schemaVersion !== 1
        || previousOwner?.kind !== "canonical-claim-reclamation-mutex"
        || !Number.isSafeInteger(previousOwner?.pid) || previousOwner.pid <= 0
        || !validProcessIdentity(previousOwner?.processIdentity)
        || typeof previousOwner?.token !== "string" || previousOwner.token === ""
        || !sameJson(previousOwner?.filesystemIdentity, filesystemIdentity)) {
      child.stdin.end("RELEASE\n");
      await childClosed;
      fail("preexisting canonical reclamation lock owner is invalid");
    }
  }
  const token = randomUUID();
  let processIdentity;
  try {
    processIdentity = await processIdentityProbe(process.pid, signal);
    if (!validProcessIdentity(processIdentity)) {
      fail("cannot bind canonical reclamation lock to this process start identity");
    }
  } catch (error) {
    child.stdin.end("RELEASE\n");
    await childClosed;
    throw error;
  }
  const owner = {
    schemaVersion: 1,
    kind: "canonical-claim-reclamation-mutex",
    pid: process.pid,
    processIdentity,
    token,
    filesystemIdentity,
  };
  try {
    const ownerBytes = Buffer.byteLength(`${JSON.stringify(owner)}\n`);
    child.stdin.write(`OWNER ${encodedLockOwner(owner)}\n`);
    const owned = await iterator.next();
    const ownedMatch = owned.done ? null
      : /^OWNED ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+)$/
      .exec(owned.value);
    if (ownedMatch === null
        || ownedMatch[1] !== filesystemIdentity.device
        || ownedMatch[2] !== filesystemIdentity.inode
        || Number(ownedMatch[3]) !== 0o600
        || Number(ownedMatch[4]) !== ownerBytes
        || Number(ownedMatch[5]) !== 1) {
      fail("canonical reclamation lock helper changed its locked inode during owner publication");
    }
  } catch (error) {
    child.stdin.end("RELEASE\n");
    await childClosed;
    throw error;
  }
  const assertOwner = async () => {
    if (child.exitCode !== null) fail("canonical reclamation lock helper exited unexpectedly");
    const [actualOwner, metadata, actualIdentity] = await Promise.all([
      readFile(mutexPath, "utf8").then(JSON.parse),
      lstat(mutexPath, { bigint: true }),
      processIdentityProbe(process.pid),
    ]);
    const actualFilesystemIdentity = {
      device: String(metadata.dev), inode: String(metadata.ino),
    };
    if (!metadata.isFile() || metadata.isSymbolicLink()
        || Number(metadata.mode & 0o777n) !== 0o600
        || !sameJson(actualOwner, owner)
        || !sameJson(actualFilesystemIdentity, filesystemIdentity)
        || !sameJson(actualIdentity, processIdentity)) {
      fail("canonical reclamation lock ownership changed");
    }
  };
  let closed = false;
  const finish = async (command, expectedLine) => {
    if (closed) fail("canonical reclamation lock was already released");
    try {
      await assertOwner();
      child.stdin.end(`${command}\n`);
      const response = await iterator.next();
      if (expectedLine === null ? !response.done : response.done || response.value !== expectedLine) {
        fail(`canonical reclamation lock helper rejected its command: ${stderr}`);
      }
      const [code, childSignal] = await childClosed;
      if (code !== 0 || childSignal !== null) {
        fail(`canonical reclamation lock helper failed: code=${code} signal=${childSignal} ${stderr}`);
      }
      closed = true;
    } catch (error) {
      child.kill("SIGKILL");
      await childClosed;
      closed = true;
      throw error;
    }
  };
  return {
    rename: async (source, target) => finish(
      `RENAME ${encodedLockPath(source)} ${encodedLockPath(target)}`, "RENAMED",
    ),
    release: async () => { if (!closed) await finish("RELEASE", null); },
  };
}

export async function acquireCanonicalPublicationClaim(
  canonicalOutput, signal, processIdentityProbe = osProcessIdentity, reclamationHooks = {},
) {
  const claimPath = canonicalPublicationClaimPath(canonicalOutput);
  const token = randomUUID();
  for (;;) {
    signal?.throwIfAborted();
    let acquired = false;
    try {
      await mkdir(claimPath, { mode: 0o700 });
      acquired = true;
      const processIdentity = await processIdentityProbe(process.pid, signal);
      if (!validProcessIdentity(processIdentity)) {
        fail("cannot bind canonical publication claim to this process start identity");
      }
      await writeFile(path.join(claimPath, "owner.json"), `${JSON.stringify({
        schemaVersion: 1,
        pid: process.pid,
        processIdentity,
        token,
      })}\n`, { flag: "wx", mode: 0o400 });
      const assertOwner = async () => {
        const owner = JSON.parse(await readFile(path.join(claimPath, "owner.json"), "utf8"));
        const actualIdentity = await processIdentityProbe(process.pid);
        if (owner.token !== token || owner.pid !== process.pid
            || !sameJson(owner.processIdentity, actualIdentity)) {
          fail("canonical publication claim ownership changed");
        }
      };
      return {
        path: claimPath,
        assertOwner,
        release: async () => {
          await assertOwner();
          await cleanupCanaryScratch(claimPath);
        },
      };
    } catch (error) {
      if (error.code !== "EEXIST") {
        if (acquired) await cleanupCanaryScratch(claimPath);
        throw error;
      }
    }
    await reclamationHooks.beforeSerialization?.(claimPath);
    const reclamation = await acquireCanonicalClaimReclamationMutex(
      claimPath, signal, processIdentityProbe,
    );
    try {
      signal?.throwIfAborted();
      // A contender may have replaced the stale claim while this process waited for the
      // reclamation mutex. Re-read every byte and re-probe the current owner while serialized;
      // only this exact observation may authorize the subsequent rename.
      const ownerPath = path.join(claimPath, "owner.json");
      const owner = await readFile(ownerPath, "utf8").then(JSON.parse).catch(() => null);
      const metadata = await lstat(claimPath, { bigint: true }).catch((error) => {
        if (error.code === "ENOENT") return null;
        throw error;
      });
      if (metadata === null) continue;
      const oldEnough = Date.now() - Number(metadata.mtimeMs)
        >= PREPARATION_LOCK_ORPHAN_GRACE_MS;
      const verifiableOwner = owner?.schemaVersion === 1
        && Number.isSafeInteger(owner.pid) && owner.pid > 0
        && validProcessIdentity(owner.processIdentity);
      const observedIdentity = verifiableOwner
        ? await processIdentityProbe(owner.pid, signal) : null;
      const staleOwner = verifiableOwner
        ? !sameJson(observedIdentity, owner.processIdentity) : oldEnough;
      if (!staleOwner) {
        fail("another live contained LTX run owns the canonical publication claim");
      }
      await reclamationHooks.afterStaleRevalidation?.(claimPath);
      signal?.throwIfAborted();
      const stale = `${claimPath}.stale-${randomUUID()}`;
      try {
        await reclamation.rename(claimPath, stale);
        await cleanupCanaryScratch(stale);
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
      continue;
    } finally {
      if (reclamation.release !== null) await reclamation.release();
    }
  }
}

export async function acquireBoundedCarrierOutcome(canonicalOutput) {
  const outcome = {
    canonicalOutput,
    successOutput: boundedCarrierSuccessPath(canonicalOutput),
    failureOutput: boundedCarrierFailurePath(canonicalOutput),
    reservation: boundedCarrierOutcomeReservationPath(canonicalOutput),
    choice: `${boundedCarrierOutcomeReservationPath(canonicalOutput)}.choice`,
    token: randomUUID(),
  };
  await mkdir(path.dirname(canonicalOutput), { recursive: true });
  await writeFile(outcome.reservation, `${outcome.token}\n`, { flag: "wx", mode: 0o400 });
  if (await pathExists(outcome.canonicalOutput)
      || await pathExists(outcome.successOutput)
      || await pathExists(outcome.failureOutput)) {
    await unlink(outcome.reservation);
    fail("SC-20254 already has an immutable canonical, success, or failure outcome");
  }
  return outcome;
}

async function assertBoundedCarrierOutcomeOwner(outcome) {
  if (await readFile(outcome?.reservation ?? "", "utf8").catch(() => null)
      !== `${outcome?.token}\n`) {
    fail("SC-20254 outcome reservation ownership changed");
  }
}

export async function releaseUnpublishedBoundedCarrierOutcome(outcome) {
  await assertBoundedCarrierOutcomeOwner(outcome);
  if (await pathExists(outcome.choice)
      || await pathExists(outcome.canonicalOutput)
      || await pathExists(outcome.successOutput)
      || await pathExists(outcome.failureOutput)) return;
  await unlink(outcome.reservation);
}

async function publishBoundedCarrierOutcome(outcome, kind, build, signal) {
  await assertBoundedCarrierOutcomeOwner(outcome);
  if (typeof outcome?.canonicalClaim?.assertOwner !== "function") {
    fail("SC-20254 publication requires the shared canonical claim");
  }
  await outcome.canonicalClaim.assertOwner();
  const target = kind === "success" ? outcome.successOutput : outcome.failureOutput;
  const other = kind === "success" ? outcome.failureOutput : outcome.successOutput;
  await writeFile(outcome.choice, `${kind}\n`, { flag: "wx", mode: 0o400 });
  if (await pathExists(outcome.canonicalOutput)
      || await pathExists(target) || await pathExists(other)) {
    fail("SC-20254 bounded-carrier outcome is already fixed");
  }
  const value = await build({
    canonicalOutput: outcome.canonicalOutput,
    successOutput: outcome.successOutput,
    failureOutput: outcome.failureOutput,
    reservation: outcome.reservation,
    choice: outcome.choice,
    canonicalPublicationClaim: outcome.canonicalClaim.path,
    canonicalPublicationClaimHeldAtPublication: true,
    outcomeChoice: kind,
    canonicalBundleAbsentAtPublication: true,
    outcomeReservationHeldAtPublication: true,
  });
  await publishExclusiveJson(target, value, signal);
  return value;
}

export async function publishBoundedCarrierSuccessReceipt(outcome, { verify, build, signal }) {
  if (typeof verify !== "function" || typeof build !== "function") {
    fail("SC-20254 success publication requires validation and receipt builders");
  }
  await verify();
  return publishBoundedCarrierOutcome(outcome, "success", async (publication) => {
    if (await pathExists(outcome.canonicalOutput)) {
      fail("SC-20254 canonical bundle exists; success receipt suppressed");
    }
    return validateBoundedCarrierReceipt(await build(publication));
  }, signal);
}

export async function publishBoundedCarrierFailureReceipt(outcome, { verify, build, signal }) {
  if (typeof verify !== "function" || typeof build !== "function") {
    fail("SC-20254 failure publication requires validation and receipt builders");
  }
  await verify();
  return publishBoundedCarrierOutcome(outcome, "failure", async (publication) => {
    if (await pathExists(outcome.canonicalOutput)) {
      fail("SC-20254 canonical bundle exists; failure receipt suppressed");
    }
    return validateBoundedCarrierReceipt(await build(publication));
  }, signal);
}

async function assertPreparationHasNoTransientResidue(preparationRoot) {
  const parent = path.dirname(preparationRoot);
  const key = path.basename(preparationRoot);
  const entries = await readdir(parent).catch((error) => {
    if (error.code === "ENOENT") return [];
    throw error;
  });
  const residue = entries.filter((name) =>
    name === `.${key}.lock`
    || name.startsWith(`.${key}.stage-`)
    || name.startsWith(`.${key}.build-`)
    || name.startsWith(`.${key}.tmp-`));
  if (residue.length) fail(`SC-20191 preparation retained transient residue: ${residue.join(", ")}`);
}

async function runCampaignEntryController({
  output, inferenceRepo, sceneWorksRevision, inferenceRevision,
  sceneWorksTree, inferenceTree, identity, preparationKey, preparationRoot,
  prepared, signal, setActiveWatchdog, bounded = false, boundedTier = null,
  q8AuditPath = null, q8ReleaseAuthorization = null,
}) {
  const boundedEntry = bounded || boundedTier !== null;
  const boundedSpec = boundedEntry ? boundedCampaignEntrySpec(boundedTier ?? "q4") : null;
  const spec = boundedEntry ? {
    plan: BOUNDED_CAMPAIGN_ENTRY_PLAN,
    planEntry: (config) => boundedCampaignEntryPlan(config, boundedSpec.tier),
    acquireOutcome: (output) => acquireBoundedCampaignEntryOutcome(
      output, boundedSpec.tier, q8ReleaseAuthorization,
    ),
    releaseOutcome: releaseUnpublishedBoundedCampaignEntryOutcome,
    provider: boundedSpec.provider,
    logicalCaseId: boundedSpec.logicalCaseId,
    validateBundle: (bundle) => validateBoundedCampaignEntryBundle(bundle, boundedSpec.tier),
  } : {
    plan: CAMPAIGN_ENTRY_PLAN,
    planEntry: campaignEntryPlan,
    acquireOutcome: acquireCampaignEntryOutcome,
    releaseOutcome: releaseUnpublishedCampaignEntryOutcome,
    provider: CAMPAIGN_ENTRY_PROVIDER,
    logicalCaseId: CAMPAIGN_ENTRY_LOGICAL_CASE_ID,
    validateBundle: validateCampaignEntryBundle,
  };
  const config = JSON.parse(await readFile(path.join(ROOT, spec.plan), "utf8"));
  spec.planEntry(config);
  if (boundedSpec?.tier === "bf16") {
    await assertQ8ReleaseAuthorization(q8ReleaseAuthorization, q8AuditPath, {
      sceneWorksRevision, inferenceRevision,
    });
  }
  const executionClaim = await acquireCanonicalPublicationClaim(
    containedCampaignExecutionClaimTarget(output), signal,
  );
  let canonicalClaim;
  try {
    canonicalClaim = await acquireCanonicalPublicationClaim(output, signal);
  } catch (error) {
    await executionClaim.release();
    throw error;
  }
  let campaignOutcome;
  try {
    campaignOutcome = await spec.acquireOutcome(output);
    campaignOutcome.canonicalClaim = canonicalClaim;
    campaignOutcome.executionClaim = executionClaim;
  } catch (error) {
    let releaseError = null;
    try { await canonicalClaim.release(); } catch (cleanup) { releaseError = cleanup; }
    try { await executionClaim.release(); } catch (cleanup) {
      releaseError = preservePrimaryFailure(releaseError, cleanup);
    }
    throw preservePrimaryFailure(error, releaseError);
  }
  const runRoot = path.join(path.dirname(output), "runs");
  const providerOptions = {
    "inference-repo": inferenceRepo,
    "preparation-root": preparationRoot,
    "preparation-key": preparationKey,
    "run-root": runRoot,
    "scene-revision": sceneWorksRevision,
    "scene-tree": sceneWorksTree,
    "inference-revision": inferenceRevision,
    "inference-tree": inferenceTree,
    "canonical-output": campaignOutcome.canonicalOutput,
    "failure-output": campaignOutcome.failureOutput,
    "outcome-reservation": campaignOutcome.reservation,
    "outcome-token": campaignOutcome.token,
    ...(boundedSpec?.tier === "bf16" ? { "q8-audit": q8AuditPath } : {}),
  };
  const providerCommand = [fileURLToPath(import.meta.url), "--campaign-provider"];
  let operationError = null;
  let cleanupError = null;
  let bundle = null;
  try {
    await mkdir(runRoot, { recursive: true, mode: 0o700 });
    await assertRunRootEmpty(runRoot);
    bundle = await runProviderPlan({
      config,
      providerCommand,
      sceneWorksRepo: ROOT,
      inferenceRepo,
      backend: "mlx",
      providerName: spec.provider,
      onProviderInvocation: ({ action, cases }) => {
        if (action !== "run" || cases.length !== 1
            || cases[0].logicalCaseId !== spec.logicalCaseId) {
          fail("contained LTX campaign attempted a batch or a foreign logical case");
        }
      },
      // No checkpoint callback: no partial or pre-validation bundle may reach the output path.
      executeProvider: async (command, args, input) => {
        if (command !== providerCommand[0]
            || !isDeepStrictEqual(args, providerCommand.slice(1))) {
          fail("SC-20191 harness attempted a foreign provider command");
        }
        return campaignProviderInvocation(providerOptions, input, {
          signal, setActiveWatchdog, canonicalClaim, executionClaim,
          bounded: boundedEntry, boundedTier: boundedSpec?.tier ?? null,
          q8ReleaseAuthorization,
        });
      },
    });
    signal.throwIfAborted();
    await cleanupCanaryScratch(runRoot);
    await assertCampaignSourceState(providerOptions, signal);
    await assertRuntimeAssetIdentities(prepared.adapter, prepared.metallib, signal);
    if (await validatePreparedCache(
      preparationRoot, preparationKey, identity, signal, boundedSpec?.tier ?? "q4",
    ) === null) fail("SC-20191 prepared cache disappeared before publication");
    await assertPreparationHasNoTransientResidue(preparationRoot);
    if (boundedSpec?.tier === "bf16") {
      await assertQ8ReleaseAuthorization(q8ReleaseAuthorization, q8AuditPath, {
        sceneWorksRevision, inferenceRevision,
      });
    }
    spec.validateBundle(bundle);
    if (boundedEntry) {
      await publishBoundedCampaignEntryOutcome(
        campaignOutcome, "canonical", async () => bundle, signal,
      );
    } else {
      await publishCampaignEntryCanonicalOutcome(campaignOutcome, bundle, signal);
    }
  } catch (error) {
    operationError = error;
  } finally {
    for (const cleanup of [
      async () => {
        if (await pathExists(runRoot)) await cleanupCanaryScratch(runRoot);
      },
      async () => spec.releaseOutcome(campaignOutcome),
      async () => canonicalClaim.release(),
      async () => executionClaim.release(),
    ]) {
      try {
        await cleanup();
      } catch (error) {
        cleanupError = preservePrimaryFailure(cleanupError, error);
      }
    }
  }
  const finalError = preservePrimaryFailure(operationError, cleanupError);
  if (finalError !== null) throw finalError;
  process.stdout.write(`${output}\n`);
}

async function runBoundedCarrierController({
  output, inferenceRepo, sceneWorksRevision, inferenceRevision,
  sceneWorksTree, inferenceTree, identity, preparationKey, preparationRoot,
  prepared, hostMemoryBytes, signal, setActiveWatchdog,
}) {
  const executionClaim = await acquireCanonicalPublicationClaim(
    containedCampaignExecutionClaimTarget(output), signal,
  );
  let canonicalClaim;
  try {
    canonicalClaim = await acquireCanonicalPublicationClaim(output, signal);
  } catch (error) {
    await executionClaim.release();
    throw error;
  }
  let outcome;
  try {
    outcome = await acquireBoundedCarrierOutcome(output);
    outcome.canonicalClaim = canonicalClaim;
  } catch (error) {
    let releaseError = null;
    try { await canonicalClaim.release(); } catch (cleanup) { releaseError = cleanup; }
    try { await executionClaim.release(); } catch (cleanup) {
      releaseError = preservePrimaryFailure(releaseError, cleanup);
    }
    throw preservePrimaryFailure(error, releaseError);
  }
  const runRoot = path.join(path.dirname(output), "sc-20254-runs");
  const sourceOptions = {
    "inference-repo": inferenceRepo,
    "scene-revision": sceneWorksRevision,
    "scene-tree": sceneWorksTree,
    "inference-revision": inferenceRevision,
    "inference-tree": inferenceTree,
  };
  let runScratch = null;
  let operationError = null;
  let cleanupError = null;
  let failureEvents = null;
  let response = null;
  let successEvents = null;
  try {
    await mkdir(runRoot, { recursive: true, mode: 0o700 });
    await assertRunRootEmpty(runRoot);
    runScratch = await mkdtemp(path.join(runRoot, "sc-20254-run-"));
    const runtimeHome = path.join(runScratch, "home");
    await mkdir(runtimeHome, { mode: 0o700 });
    await exactMetadata(runtimeHome, "directory", 0o700);
    await exactEntries(runtimeHome, []);
    const requestPath = path.join(runScratch, "request.json");
    const responsePath = path.join(runScratch, "response.json");
    const eventsPath = path.join(runScratch, "watchdog.jsonl");
    const request = boundedCarrierRequest(
      hostMemoryBytes,
      inventoryAtRoot(identity.artifact.textEncoder, prepared.roots.textEncoder),
    );
    await writeFile(requestPath, canonicalJson(request), { flag: "wx" });
    const launchPrepared = await validatePreparedCache(
      preparationRoot, preparationKey, identity, signal,
    );
    if (launchPrepared === null) fail("SC-20254 prepared cache disappeared before launch");
    await assertRuntimeAssetIdentities(
      launchPrepared.adapter, launchPrepared.metallib, signal,
    );
    const runtimeMemoryFreeFloor = runtimeFreeFloor(hostMemoryBytes);
    const watchdogArgs = [
      path.join(ROOT, "scripts/memory-calibration-watchdog.py"),
      "--max-footprint-bytes", String(MAX_FOOTPRINT_BYTES),
      "--max-runtime-seconds", String(MAX_RUNTIME_SECONDS),
      "--host-memory-bytes", String(hostMemoryBytes),
      "--min-memory-free-bytes", String(runtimeMemoryFreeFloor),
      "--sample-interval", "0.25",
      "--telemetry-timeout", "1",
      "--child-attestation-timeout", String(CHILD_ATTESTATION_TIMEOUT_SECONDS),
      "--term-grace", "1",
      "--event-file", eventsPath,
      "--require-child-attestation",
      "--require-provider-phases",
      "--provider-phase-profile", "bounded-carrier",
      "--",
      "/bin/sh", "-c", 'set -C; exec "$1" <"$2" >"$3"',
      "sc-20254-bounded-carrier", prepared.adapter.path, requestPath, responsePath,
    ];
    const environment = canaryWatchdogEnvironment(
      process.env, prepared.roots, prepared.metallib.path, runtimeHome,
    );
    await assertCampaignSourceState(
      sourceOptions, signal, "SC-20254 bounded carrier",
    );
    // The sole fresh actual-GPU-owner and two-boundary host check remains the final operation before
    // the watchdog owns and releases the one provider render.
    await assertHostPreflight(hostMemoryBytes, signal);
    const status = await new Promise((resolve, reject) => {
      const child = spawn("/usr/bin/python3", watchdogArgs, {
        stdio: "inherit", env: environment,
      });
      setActiveWatchdog(child);
      const onAbort = () => child.kill(signal.reason?.signalName ?? "SIGTERM");
      signal.addEventListener("abort", onAbort, { once: true });
      child.on("error", reject);
      child.on("close", (code, childSignal) => {
        signal.removeEventListener("abort", onAbort);
        setActiveWatchdog(null);
        resolve({ code, signal: childSignal });
      });
    });
    const eventBytes = await readFile(eventsPath, "utf8").catch(() => "");
    if (status.code !== 0 || status.signal) {
      const watchdogError = new Error(watchdogFailureSummary(status, eventBytes));
      try {
        const events = eventBytes.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
        validateCampaignEntryFailureEvents(
          events, hostMemoryBytes, BOUNDED_CARRIER_PHASES,
        );
        const started = events.find((event) => event.event === "started");
        const processTable = (await execFileAsync(
          "/bin/ps", ["-ww", "-axo", "pid=,pgid=,command="],
          { encoding: "utf8", timeout: 2_000 },
        )).stdout;
        const residue = ownedProcessGroupResidue(processTable, started?.pgid);
        if (residue.length) fail(`SC-20254 watchdog process group retained residue:\n${residue.join("\n")}`);
        await assertCampaignSourceState(
          sourceOptions, undefined, "SC-20254 bounded carrier",
        );
        await assertRuntimeAssetIdentities(prepared.adapter, prepared.metallib);
        if (await validatePreparedCache(
          preparationRoot, preparationKey, identity,
        ) === null) fail("SC-20254 prepared cache disappeared after watchdog failure");
        failureEvents = events;
      } catch (error) {
        throw preserveFailureReceiptSuppression(watchdogError, error);
      }
      throw watchdogError;
    }
    signal.throwIfAborted();
    response = validateBoundedCarrierResponse(
      JSON.parse(await readFile(responsePath, "utf8")),
      { inferenceRevision, hostMemoryBytes },
    );
    successEvents = eventBytes.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
    validateBoundedCarrierWatchdogEvents(successEvents, hostMemoryBytes);
    const started = successEvents.find((event) => event.event === "started");
    const processTable = (await execFileAsync(
      "/bin/ps", ["-ww", "-axo", "pid=,pgid=,command="],
      { encoding: "utf8", timeout: 2_000 },
    )).stdout;
    const residue = ownedProcessGroupResidue(processTable, started?.pgid);
    if (residue.length) fail(`SC-20254 watchdog process group retained residue:\n${residue.join("\n")}`);
    await assertCampaignSourceState(sourceOptions, signal, "SC-20254 bounded carrier");
    await assertRuntimeAssetIdentities(prepared.adapter, prepared.metallib, signal);
    if (await validatePreparedCache(
      preparationRoot, preparationKey, identity, signal,
    ) === null) fail("SC-20254 prepared cache disappeared after bounded carrier");
  } catch (error) {
    operationError = error;
  } finally {
    try {
      if (runScratch !== null) await cleanupCanaryScratch(runScratch);
      await assertRunRootEmpty(runRoot);
    } catch (error) {
      cleanupError = error;
    }
  }
  const finalError = preservePrimaryFailure(operationError, cleanupError);
  let terminalError = null;
  try {
    if (finalError !== null) {
      let publicationError = null;
      if (failureEvents !== null && cleanupError === null) {
        try {
          await publishBoundedCarrierFailureReceipt(outcome, {
            verify: async () => {
              await assertRunRootEmpty(runRoot);
              await assertCampaignSourceState(
                sourceOptions, undefined, "SC-20254 bounded carrier",
              );
              await assertRuntimeAssetIdentities(prepared.adapter, prepared.metallib);
              if (await validatePreparedCache(
                preparationRoot, preparationKey, identity,
              ) === null) fail("SC-20254 prepared cache disappeared before failure publication");
              await assertPreparationHasNoTransientResidue(preparationRoot);
            },
            build: (publication) => boundedCarrierFailureReceipt({
              sceneWorksRevision, sceneWorksTree, inferenceRevision, inferenceTree,
              identity, preparationKey, preparationRoot, prepared,
              hostMemoryBytes, events: failureEvents, outcome: publication,
            }),
          });
        } catch (error) {
          publicationError = error;
        }
      }
      if (publicationError !== null) {
        throw preserveFailureReceiptSuppression(finalError, publicationError);
      }
      throw finalError;
    }
    await publishBoundedCarrierSuccessReceipt(outcome, {
      signal,
      verify: async () => {
        await assertRunRootEmpty(runRoot);
        await assertCampaignSourceState(sourceOptions, signal, "SC-20254 bounded carrier");
        await assertRuntimeAssetIdentities(prepared.adapter, prepared.metallib, signal);
        if (await validatePreparedCache(
          preparationRoot, preparationKey, identity, signal,
        ) === null) fail("SC-20254 prepared cache disappeared before success publication");
        await assertPreparationHasNoTransientResidue(preparationRoot);
      },
      build: (publication) => boundedCarrierSuccessReceipt({
        sceneWorksRevision, sceneWorksTree, inferenceRevision, inferenceTree,
        identity, preparationKey, preparationRoot, prepared,
        hostMemoryBytes, events: successEvents, response, outcome: publication,
      }),
    });
    process.stdout.write(`${outcome.successOutput}\n`);
  } catch (error) {
    terminalError = error;
  }
  let releaseError = null;
  for (const release of [
    async () => releaseUnpublishedBoundedCarrierOutcome(outcome),
    async () => canonicalClaim.release(),
    async () => executionClaim.release(),
  ]) {
    try {
      await release();
    } catch (error) {
      releaseError = preservePrimaryFailure(releaseError, error);
    }
  }
  const completedError = preservePrimaryFailure(terminalError, releaseError);
  if (completedError !== null) throw completedError;
}

async function controller(argv) {
  if (process.platform !== "darwin") fail("LTX safety canary requires Darwin phys_footprint telemetry");
  const options = parseArgs(argv);
  for (const name of ["inference-repo", "output"]) {
    if (!options[name]) fail(`--${name} is required`);
  }
  const unexpected = Object.keys(options).filter(
    (name) => !["inference-repo", "output", "profile", "q8-audit"].includes(name),
  );
  if (unexpected.length) fail(`unsupported option(s): ${unexpected.join(", ")}`);
  const profileName = options.profile ?? SAFETY_CANARY_PROFILE;
  const campaignEntry = profileName === CAMPAIGN_ENTRY_PROFILE;
  const boundedCarrier = profileName === BOUNDED_CARRIER_PROFILE;
  const boundedCampaignSpec = Object.values(BOUNDED_CAMPAIGN_ENTRY_SPECS)
    .find((candidate) => candidate.profile === profileName) ?? null;
  const boundedCampaignEntry = boundedCampaignSpec !== null;
  if (boundedCampaignSpec?.tier === "bf16") {
    if (!options["q8-audit"]) fail("--q8-audit is required for the SC-20430 bf16 profile");
  } else if (options["q8-audit"] !== undefined) {
    fail("--q8-audit is only valid for the SC-20430 bf16 profile");
  }
  const profile = campaignEntry
    ? { story: "sc-20191" }
    : boundedCarrier ? { story: "sc-20254" }
      : boundedCampaignEntry ? { story: boundedCampaignSpec.story } : canaryProfile(profileName);
  const inferenceRepo = path.resolve(options["inference-repo"]);
  const output = path.resolve(options.output);
  const relativeOutput = path.relative(ROOT, output);
  if (relativeOutput === "" || (!relativeOutput.startsWith("..") && !path.isAbsolute(relativeOutput))) {
    fail("--output must stay outside the SceneWorks repository");
  }
  const sceneWorksRevision = await cleanHead(ROOT, "SceneWorks");
  const inferenceRevision = await cleanHead(inferenceRepo, "inference");
  let q8ReleaseAuthorization = null;
  if (boundedCampaignSpec?.tier === "bf16") {
    q8ReleaseAuthorization = await validateQ8IndependentAudit(options["q8-audit"], {
      sceneWorksRevision, inferenceRevision,
    });
  }
  const [sceneWorksTree, inferenceTree, toolchainChannel] = await Promise.all([
    git(ROOT, ["rev-parse", "HEAD^{tree}"]),
    git(inferenceRepo, ["rev-parse", "HEAD^{tree}"]),
    repositoryToolchain(),
  ]);
  const adapterSource = await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/lib.rs"), "utf8");
  const pin = adapterSource.match(/pub const INFERENCE_PIN: &str = "([0-9a-f]{40})";/)?.[1];
  if (!pin || inferenceRevision !== pin) fail(`inference checkout ${inferenceRevision} does not match adapter pin ${pin}`);
  if (process.env.SCENEWORKS_LTX_REPOSITORY !== ARTIFACT_REPOSITORY
      || process.env.SCENEWORKS_LTX_REVISION !== ARTIFACT_REVISION) {
    fail("SCENEWORKS_LTX_REPOSITORY/REVISION do not name the immutable canary artifact");
  }
  const memoryBytes = Number((await execFileAsync(
    "/usr/sbin/sysctl", ["-n", "hw.memsize"], { timeout: 2_000 },
  )).stdout.trim());
  if (!Number.isSafeInteger(memoryBytes) || memoryBytes < MIN_PREFLIGHT_FREE_BYTES) {
    fail(`host memory ${memoryBytes} cannot preserve two canary stop boundaries`);
  }
  const preparationTier = boundedCampaignSpec?.tier ?? "q4";
  const identity = preparationIdentity(
    sceneWorksTree, inferenceTree, toolchainChannel, preparationTier,
  );
  const preparationKey = preparationCacheKey(identity);
  const preparationRoot = preparationCacheRoot(output, preparationKey);
  let activeWatchdog = null;
  let runScratch = null;
  let interruptedSignal = null;
  const cancellation = new AbortController();
  const signalHandlers = Object.fromEntries(["SIGINT", "SIGTERM"].map((signalName) => [
    signalName,
    () => {
      if (interruptedSignal === null) {
        interruptedSignal = signalName;
        cancellation.abort(new CanaryInterrupted(signalName));
      }
      if (activeWatchdog !== null) activeWatchdog.kill(signalName);
    },
  ]));
  for (const [signalName, handler] of Object.entries(signalHandlers)) {
    process.on(signalName, handler);
  }
  let operationError = null;
  let cleanupError = null;
  try {
    const { signal } = cancellation;
    const prepared = await prepareCanaryCache({
      preparationRoot,
      key: preparationKey,
      identity,
      preparedFrom: { sceneWorksRevision, inferenceRevision },
      signal,
      tier: preparationTier,
      build: async (stage, buildSignal, hooks) => {
        const numericTierRoot = await realpath(path.resolve(
          process.env.SCENEWORKS_LTX_ROOT ?? fail("SCENEWORKS_LTX_ROOT is required to prepare q4"),
        ));
        const textEncoderRoot = await realpath(path.resolve(
          process.env.SCENEWORKS_LTX_TEXT_ENCODER_ROOT
            ?? fail("SCENEWORKS_LTX_TEXT_ENCODER_ROOT is required to prepare the text encoder"),
        ));
        const numericTierInventory = assertInventory(
          await hashArtifactInventory(numericTierRoot, { signal: buildSignal }),
          identity.artifact.numericTier,
          "local q4",
        );
        const textEncoderInventory = assertInventory(
          await hashArtifactInventory(textEncoderRoot, { signal: buildSignal }),
          identity.artifact.textEncoder,
          "local text-encoder",
        );
        const preparationDevice = (await stat(stage, { bigint: true })).dev;
        const roots = privateArtifactRoots(stage, preparationTier);
        await mkdir(path.dirname(roots.numericTier), { recursive: true, mode: 0o700 });
        await cloneArtifactTree(numericTierRoot, roots.numericTier, preparationDevice, buildSignal);
        const preparedNumericTier = assertInventory(
          await hashArtifactInventory(roots.numericTier, { signal: buildSignal }),
          inventoryAtRoot(numericTierInventory, roots.numericTier),
          "new prepared numericTier",
        );
        await hooks.afterNumericClone?.(stage, buildSignal);
        await cloneArtifactTree(
          textEncoderRoot, roots.textEncoder, preparationDevice, buildSignal,
        );
        const preparedTextEncoder = assertInventory(
          await hashArtifactInventory(roots.textEncoder, { signal: buildSignal }),
          inventoryAtRoot(textEncoderInventory, roots.textEncoder),
          "new prepared textEncoder",
        );
        await hooks.afterArtifactClone?.(stage, buildSignal);
        const buildRoot = path.join(stage, `.build-${randomUUID()}`);
        let buildError = null;
        let buildCleanupError = null;
        try {
          await buildExactAdapter(
            sceneWorksRevision,
            inferenceRepo,
            inferenceRevision,
            buildRoot,
            stage,
            buildSignal,
          );
        } catch (error) {
          buildError = error;
        } finally {
          try {
            await cleanupCanaryScratch(buildRoot);
          } catch (error) {
            buildCleanupError = error;
          }
        }
        const finalBuildError = preservePrimaryFailure(buildError, buildCleanupError);
        if (finalBuildError !== null) throw finalBuildError;
        return {
          artifacts: {
            numericTier: preparedNumericTier,
            textEncoder: preparedTextEncoder,
          },
        };
      },
    });
    const preparationReused = prepared.reused;
    const {
      numericTier: privateNumericTierRoot,
      textEncoder: privateTextEncoderRoot,
    } = prepared.roots;
    const adapter = prepared.adapter;
    const metallib = prepared.metallib;
    const textEncoderInventory = inventoryAtRoot(
      identity.artifact.textEncoder, privateTextEncoderRoot,
    );
    if (boundedCarrier) {
      await runBoundedCarrierController({
        output,
        inferenceRepo,
        sceneWorksRevision,
        inferenceRevision,
        sceneWorksTree,
        inferenceTree,
        identity,
        preparationKey,
        preparationRoot,
        prepared,
        hostMemoryBytes: memoryBytes,
        signal,
        setActiveWatchdog: (child) => { activeWatchdog = child; },
      });
      return;
    }
    if (boundedCampaignEntry) {
      await runCampaignEntryController({
        output,
        inferenceRepo,
        sceneWorksRevision,
        inferenceRevision,
        sceneWorksTree,
        inferenceTree,
        identity,
        preparationKey,
        preparationRoot,
        prepared,
        signal,
        setActiveWatchdog: (child) => { activeWatchdog = child; },
        bounded: true,
        boundedTier: boundedCampaignSpec.tier,
        q8AuditPath: options["q8-audit"] ?? null,
        q8ReleaseAuthorization,
      });
      return;
    }
    if (campaignEntry) {
      await runCampaignEntryController({
        output,
        inferenceRepo,
        sceneWorksRevision,
        inferenceRevision,
        sceneWorksTree,
        inferenceTree,
        identity,
        preparationKey,
        preparationRoot,
        prepared,
        signal,
        setActiveWatchdog: (child) => { activeWatchdog = child; },
      });
      return;
    }
    const runRoot = path.join(path.dirname(output), "runs");
    await mkdir(runRoot, { recursive: true, mode: 0o700 });
    runScratch = await mkdtemp(path.join(runRoot, `${profile.story}-run-`));
    const runtimeHome = path.join(runScratch, "home");
    await mkdir(runtimeHome, { mode: 0o700 });
    await exactMetadata(runtimeHome, "directory", 0o700);
    await exactEntries(runtimeHome, []);
    const requestPath = path.join(runScratch, "request.json");
    const responsePath = path.join(runScratch, "response.json");
    const eventsPath = path.join(runScratch, "watchdog.jsonl");
    const request = canaryRequest(memoryBytes, textEncoderInventory, profileName);
    await writeFile(requestPath, `${JSON.stringify(request)}\n`, { flag: "wx" });
    const launchPrepared = await validatePreparedCache(
      preparationRoot, preparationKey, identity, signal,
    );
    if (launchPrepared === null) fail("prepared canary cache disappeared before launch");
    await assertRuntimeAssetIdentities(
      launchPrepared.adapter, launchPrepared.metallib, signal,
    );
    const watchdog = path.join(ROOT, "scripts/memory-calibration-watchdog.py");
    const runtimeMemoryFreeFloor = runtimeFreeFloor(memoryBytes);
    const watchdogArgs = [
      watchdog,
      "--max-footprint-bytes", String(MAX_FOOTPRINT_BYTES),
      "--max-runtime-seconds", String(MAX_RUNTIME_SECONDS),
      "--host-memory-bytes", String(memoryBytes),
      "--min-memory-free-bytes", String(runtimeMemoryFreeFloor),
      "--sample-interval", "0.25",
      "--telemetry-timeout", "1",
      "--child-attestation-timeout", String(CHILD_ATTESTATION_TIMEOUT_SECONDS),
      "--term-grace", "1",
      "--event-file", eventsPath,
      "--require-child-attestation",
      "--",
      "/bin/sh", "-c", 'set -C; exec "$1" <"$2" >"$3"',
      `${profile.story}-canary`, adapter.path, requestPath, responsePath,
    ];
    const watchdogEnv = canaryWatchdogEnvironment(
      process.env,
      { numericTier: privateNumericTierRoot, textEncoder: privateTextEncoderRoot },
      metallib.path,
      runtimeHome,
    );
    if (await cleanHead(ROOT, "SceneWorks", signal) !== sceneWorksRevision
        || await cleanHead(inferenceRepo, "inference", signal) !== inferenceRevision
        || await git(ROOT, ["rev-parse", "HEAD^{tree}"], signal) !== sceneWorksTree
        || await git(inferenceRepo, ["rev-parse", "HEAD^{tree}"], signal) !== inferenceTree) {
      fail("verified source checkout changed during canary preparation");
    }
    signal.throwIfAborted();
    // There is no sustained-quiet gate. Unrelated compilation may overlap preparation; this one
    // fresh check exclusively protects the instant at which the watchdog releases model loading.
    const preLaunchHost = await assertHostPreflight(memoryBytes, signal);
    const status = await new Promise((resolve, reject) => {
      const childProcess = spawn("/usr/bin/python3", watchdogArgs, {
        stdio: "inherit",
        env: watchdogEnv,
      });
      activeWatchdog = childProcess;
      if (interruptedSignal !== null) childProcess.kill(interruptedSignal);
      childProcess.on("error", reject);
      childProcess.on("close", (code, signal) => resolve({ code, signal }));
    });
    activeWatchdog = null;
    if (interruptedSignal !== null) throw new CanaryInterrupted(interruptedSignal);
    if (status.code !== 0 || status.signal) {
      const eventBytes = await readFile(eventsPath, "utf8").catch(() => "");
      fail(watchdogFailureSummary(status, eventBytes));
    }
    await assertRuntimeAssetIdentities(adapter, metallib, signal);
    if (await validatePreparedCache(preparationRoot, preparationKey, identity, signal) === null) {
      fail("prepared canary cache disappeared after the run");
    }
    const [sceneWorksAfterRun, inferenceAfterRun] = await Promise.all([
      observedSourceState(ROOT, sceneWorksRevision, sceneWorksTree),
      observedSourceState(inferenceRepo, inferenceRevision, inferenceTree),
    ]);
    const response = validateCanaryResponse(
      JSON.parse(await readFile(responsePath, "utf8")),
      inferenceRevision,
      textEncoderInventory,
      memoryBytes,
      profileName,
    );
    const events = (await readFile(eventsPath, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
    const attestedIndex = events.findIndex((event) => event.event === "child_attested");
    if (!events.some((event) => event.event === "started")
      || attestedIndex < 0
      || !events.slice(0, attestedIndex).some((event) =>
        event.event === "sample" && event.phase === "before_child_release")
      || !events.slice(0, attestedIndex).some((event) =>
        event.event === "sample" && event.phase === "child_attested_before_allocation")
      || events.filter((event) => event.event === "sample").some((event) =>
        !Number.isInteger(event.memoryFreeBytes)
        || event.memoryFreeBytes < runtimeMemoryFreeFloor
        || !Number.isInteger(event.swapFreeBytes)
        || event.swapFreeBytes < 0)
      || events.some((event) => event.event === "hard_stop" || event.event === "terminated")) {
      fail("watchdog event stream is incomplete or contains a hard stop");
    }
    const maxObservedFootprintBytes = Math.max(...events
      .filter((event) => event.event === "sample")
      .map((event) => event.physicalFootprintBytes));
    if (maxObservedFootprintBytes >= MAX_FOOTPRINT_BYTES) fail("successful watchdog stream crossed its stop boundary");
    const receipt = {
    schemaVersion: 1,
    story: profile.story,
    canaryIdentity: profile.identity,
    diagnosticOnly: true,
    promotable: false,
    ingestible: false,
    sceneWorksRevision,
    inferenceRevision,
    adapter,
    metallib,
    preparation: {
      key: preparationKey,
      root: preparationRoot,
      identity,
      preparedFrom: prepared.manifest.preparedFrom,
      reused: preparationReused,
      verification: "initial-content-hash-atomic-publish-read-only-metadata-seal",
    },
    hostPreflight: { preLaunch: preLaunchHost },
    sourceAfterRun: { sceneWorks: sceneWorksAfterRun, inference: inferenceAfterRun },
    request,
    response,
    watchdog: {
      maxObservedFootprintBytes,
      minimumRuntimeFreeBytes: runtimeMemoryFreeFloor,
      swapTelemetryRequired: true,
      ownedGroupResidueVerified: true,
      events,
    },
    };
    await mkdir(path.dirname(output), { recursive: true });
    signal.throwIfAborted();
    const temporary = `${output}.tmp-${process.pid}`;
    try {
      await writeFile(temporary, `${JSON.stringify(receipt, null, 2)}\n`, { flag: "wx" });
      signal.throwIfAborted();
      await link(temporary, output);
    } finally {
      await unlink(temporary).catch((error) => { if (error.code !== "ENOENT") throw error; });
    }
    process.stdout.write(`${output}\n`);
  } catch (error) {
    operationError = error;
  } finally {
    try {
      if (runScratch !== null) await cleanupCanaryScratch(runScratch);
    } catch (error) {
      cleanupError = error;
    }
    for (const [signalName, handler] of Object.entries(signalHandlers)) {
      process.off(signalName, handler);
    }
  }
  const primaryError = interruptedSignal !== null
    ? new CanaryInterrupted(interruptedSignal) : operationError;
  const finalError = preservePrimaryFailure(primaryError, cleanupError);
  if (finalError !== null) throw finalError;
}

try {
  if (process.argv[1] === fileURLToPath(import.meta.url)) {
    if (process.argv[2] === "--campaign-provider") {
      await campaignProvider(process.argv.slice(3));
    } else if (process.argv[2] === "--bounded-selector-report") {
      await boundedSelectorReportController(process.argv.slice(3));
    } else {
      await controller(process.argv.slice(2));
    }
  }
} catch (error) {
  process.stderr.write(`${error.stack ?? error}\n`);
  process.exitCode = error instanceof CanaryInterrupted ? error.exitCode : 1;
}
