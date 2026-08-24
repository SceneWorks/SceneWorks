#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream, readFileSync } from "node:fs";
import {
  appendFile, cp, lstat, mkdir, readdir, readFile, realpath, rename, rm, stat, statfs, writeFile,
} from "node:fs/promises";
import { freemem, totalmem } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";

import { hashArtifactInventory, listCachedArtifactFiles } from "./hash-artifact-inventory.mjs";
import { claimKey, patternToRegExp } from "./check-download-patterns.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

export const PROFILE_NAME = "epic-20738-candle-cuda-terminal-v1";
export const PROFILE_PATH = "config/terminal-evidence/epic-20738-cuda.json";
export const PROFILE_SCHEMA_PATH = "config/terminal-evidence/epic-20738-profile.schema.json";
export const RECEIPT_SCHEMA_PATH = "config/terminal-evidence/epic-20738-receipt.schema.json";
export const CACHE_PREFLIGHT_SCHEMA_PATH = "config/terminal-evidence/epic-20738-cache-preflight.schema.json";
const DOWNLOAD_EVIDENCE_PATH = "config/download-pattern-evidence.json";
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SCHEMA_VALIDATORS = new Map();
let LEGACY_IMPORTED_RECEIPT_VALIDATOR = null;
const TRUSTED_LEGACY_IMPORT_CONTEXT = Symbol("trusted legacy prefix import");
const SHA40 = /^[0-9a-f]{40}$/;
const SAFE_ID = /^[a-z0-9][a-z0-9-]+$/;
const BLOCKED = ["anima", "sana", "vace", "flux2", "true_v2", "true-v2", "eros"];
const EXPECTED_CELLS = [
  "chroma1-base-q4", "chroma1-base-q8", "chroma1-flash-q4", "chroma1-flash-q8",
  "chroma1-hd-q4", "chroma1-hd-q8", "flux1-dev-q4", "flux1-dev-q8",
  "flux1-schnell-q4", "flux1-schnell-q8", "scail2-q4", "scail2-multi-reference-q4",
  "scail2-q8", "ltx-2-3-q8", "sdxl-openpose", "realvisxl-openpose",
  "realvisxl-lightning-openpose", "illustrious-v1-openpose", "illustrious-v2-openpose",
];
const SDXL_POSE_MODELS = [
  "sdxl", "realvisxl", "realvisxl_lightning", "illustrious_xl_v1", "illustrious_xl_v2",
];
const PROMOTED_SDXL_POSE_MODELS = [
  "sdxl",
  "realvisxl",
  "realvisxl_lightning",
  "illustrious_xl_v1",
  "illustrious_xl_v2",
];
const EXPECTED_CELL_SEMANTICS_SHA256 = "2fcd20e4909f0bd0ba6c78c6a85247267c354735f77f4ed4912d47941a8512c1";
const LEGACY_CELL_SEMANTICS_SHA256 = "dc0e529b40e898727eb9401562a928345b958b4c94677d0206ccc70471f6f879";
const EXPECTED_ARTIFACT_SEMANTICS_SHA256 = "1e98392f71b1ad3d10d4bf18a6f23a497f5ffe588127ac59c54e53d392e6e255";
const LEGACY_RECOVERY_ARTIFACT_SEMANTICS_SHA256 = "5b9ef60c18ab15caeca7ff0411b199618f0aa22cc051a70607aa7a0f7c6cd932";
const LEGACY_ARTIFACT_SEMANTICS_SHA256 = "f2bb7a77b83ce11cc32c3a1f9639534a67a149bc464a9730fb5c0988b4a03f9e";
const LEGACY_SCENEWORKS_HEAD = "8886a9e69f26beec05688c81b414859bd102f6d0";
const HISTORICAL_INFERENCE_PIN = "b646a6f89ba9f6b07efe53dd583d8a42e21e9871";
const LTX_CURRENT_REVISION = "01df27d308466533aa09d251e3aebdcc627d07eb";
const LTX_APPROVED_PARENT_REVISION = "254989c3ca7ee691187647f350b112c0c448789d";
const ILLUSTRIOUS_V1_LEGACY_REVISION = "c5a92a902dd4e6ee99c2a57981ecf66209905dd1";
const ILLUSTRIOUS_V2_LEGACY_REVISION = "7c5c8b2bb75a8f38a7365e70bdf84d38d6204473";
const ILLUSTRIOUS_CURRENT_AUTHORITIES = new Map([
  ["illustrious-v1-q4", "778c3f02b7703b0c2755d0c0447592897193c6b5"],
  ["illustrious-v2-q4", "672e9851ede4dc856fa945649b6691975c9d74a3"],
]);
const REVIEWED_FLUX_MISSING = {
  artifactId: "flux1-schnell-q8",
  repository: "SceneWorks/flux1-schnell-mlx",
  revision: "bba3ae01dfd94089f173c05edd4e1a4c551f2599",
  file: "q8/transformer/model.safetensors",
};
const ILLUSTRIOUS_Q4_MARKER = {
  bits: 4, groupSize: 64, components: ["text_encoder", "text_encoder_2", "unet"],
};
const ILLUSTRIOUS_Q4_FILE_LIST_SHA256 = "13ff6afbd67d66ae25a7e6eaa88f8f3313d9d1694ea482845c2995b0ffe44a59";
const IMPORTED_PREFIX_CELLS = 7;
const RECOVERY_IMPORTED_PREFIX_CELLS = 9;
const PREFIX_ARTIFACT = `sc-20945-epic-20738-${LEGACY_SCENEWORKS_HEAD}-`;
const RECOVERY_ARTIFACT_ID = "9488587517";
const RECOVERY_ARTIFACT_NAME = "sc-20945-epic-20738-62be42127e2b4ff07321e2c369de92fc6edef526-32616545132-1";
const RECOVERY_ARTIFACT_SIZE = 4_322_200;
const RECOVERY_ARTIFACT_DIGEST = "sha256:765c8f4ed419e7a7d0fbc20dcd65f5be6f0be7ce9ed9f151915208bf541692bf";
const RECOVERY_RUN_ID = "32616545132";
const RECOVERY_RUN_ATTEMPT = "1";
const RECOVERY_SCENEWORKS_HEAD = "62be42127e2b4ff07321e2c369de92fc6edef526";
const RECOVERY_REMAINING_AUTHORITIES = 14;
const RECOVERY_REMAINING_FILES = 160;
const RECOVERY_REMAINING_SOURCE_BYTES = 151_407_076_080;
const LEGACY_RECOVERY_REMAINING_SOURCE_BYTES = 151_407_075_690;
const RECOVERY_REMAINING_ARTIFACT_IDS = [
  "flux1-schnell-q8", "scail2-q4", "scail2-q8", "ltx23-q8", "ltx23-gemma",
  "sdxl-base-q4", "sdxl-openpose", "sdxl-tokenizer-l", "sdxl-tokenizer-bigg",
  "sdxl-vae-fix", "realvisxl-q4", "realvisxl-lightning-q4", "illustrious-v1-q4",
  "illustrious-v2-q4",
];
const SPARSE_RECOVERY_ARTIFACT_ID = "9492288293";
const SPARSE_RECOVERY_ARTIFACT_NAME = "sc-20945-epic-20738-43c718b7e9a852bd5029448d18841fed0f508c3a-32628540694-1";
const SPARSE_RECOVERY_ARTIFACT_SIZE = 15_452_320;
const SPARSE_RECOVERY_ARTIFACT_DIGEST = "sha256:dbae4c7d67d824bb8568909231614c6bcc268868087eb19974ce013bfc557724";
const SPARSE_RECOVERY_RUN_ID = "32628540694";
const SPARSE_RECOVERY_RUN_ATTEMPT = "1";
const SPARSE_RECOVERY_SCENEWORKS_HEAD = "43c718b7e9a852bd5029448d18841fed0f508c3a";
const SPARSE_EXECUTION_ORDINALS = [14, 18, 19];
const SPARSE_IMPORTED_ORDINALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17];
const SPARSE_REMAINING_AUTHORITIES = 8;
const SPARSE_REMAINING_FILES = 70;
const SPARSE_REMAINING_SOURCE_BYTES = 66_821_159_668;
const LEGACY_SPARSE_REMAINING_SOURCE_BYTES = 66_821_159_278;
const SPARSE_REMAINING_ARTIFACT_IDS = [
  "ltx23-q8", "ltx23-gemma", "illustrious-v1-q4", "sdxl-openpose",
  "sdxl-tokenizer-l", "sdxl-tokenizer-bigg", "sdxl-vae-fix", "illustrious-v2-q4",
];
const PASS18_RECOVERY_ARTIFACT_ID = "9498929065";
const PASS18_RECOVERY_ARTIFACT_NAME = "sc-20945-epic-20738-655414ef3e4dec1fe9142901caea538e73ac1490-32655428377-1";
const PASS18_RECOVERY_ARTIFACT_SIZE = 16_071_005;
const PASS18_RECOVERY_ARTIFACT_DIGEST = "sha256:fae791001dd4e2015ce0567290b9b0a1d67de9e503712d2b9a60a0f9af07ec9c";
const PASS18_RECOVERY_RUN_ID = "32655428377";
const PASS18_RECOVERY_RUN_ATTEMPT = "1";
const PASS18_RECOVERY_SCENEWORKS_HEAD = "655414ef3e4dec1fe9142901caea538e73ac1490";
const PASS18_EXECUTION_ORDINALS = [14];
const PASS18_IMPORTED_ORDINALS = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19,
];
const PASS18_REMAINING_ARTIFACT_IDS = ["ltx23-q8", "ltx23-gemma"];
const PASS18_REMAINING_AUTHORITIES = 2;
const PASS18_REMAINING_FILES = 28;
const PASS18_REMAINING_SOURCE_BYTES = 56_156_615_634;
// The final 19/19 artifact is immutable source evidence.  The five OpenPose receipts lack the
// later causal-control witness, so the next continuation imports only 1..14 and replaces exactly
// 15..19 at the corrected pin.
const OPENPOSE_RECOVERY_ARTIFACT_ID = "9500244306";
const OPENPOSE_RECOVERY_ARTIFACT_NAME = "sc-20945-epic-20738-7d7a3efa3088204311e3be01330f35702774feb6-32664618999-1";
const OPENPOSE_RECOVERY_ARTIFACT_SIZE = 17_134_718;
const OPENPOSE_RECOVERY_ARTIFACT_DIGEST = "sha256:9af191547befc7833f82f4d9057fff8abfe09b21eada2be2b4f252d73ae30818";
const OPENPOSE_RECOVERY_RUN_ID = "32664618999";
const OPENPOSE_RECOVERY_RUN_ATTEMPT = "1";
const OPENPOSE_RECOVERY_SCENEWORKS_HEAD = "7d7a3efa3088204311e3be01330f35702774feb6";
const OPENPOSE_RECOVERY_IMPORTED_ORDINALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
const OPENPOSE_RECOVERY_EXECUTION_ORDINALS = [15, 16, 17, 18, 19];
const OPENPOSE_RECOVERY_REMAINING_ARTIFACT_IDS = [
  "sdxl-base-q4", "sdxl-openpose", "sdxl-tokenizer-l", "sdxl-tokenizer-bigg", "sdxl-vae-fix",
  "realvisxl-q4", "realvisxl-lightning-q4", "illustrious-v1-q4", "illustrious-v2-q4",
];
const OPENPOSE_RECOVERY_REMAINING_AUTHORITIES = 9;
const OPENPOSE_RECOVERY_REMAINING_FILES = 101;
// The failed 32679720253 candidate is diagnostic only. Its frozen census establishes this exact
// five-cell recovery plan; it must never be treated as accepted terminal evidence.
const OPENPOSE_RECOVERY_SOURCE_BYTES = 21_191_060_168;
const OPENPOSE_RECOVERY_PEAK_ORDINAL = 17;
const OPENPOSE_RECOVERY_PEAK_ARTIFACT_IDS = [
  "sdxl-openpose", "sdxl-tokenizer-l", "sdxl-tokenizer-bigg", "sdxl-vae-fix",
  "realvisxl-lightning-q4",
];
const OPENPOSE_RECOVERY_PEAK_STAGED_BYTES = 14_576_195_187;
const OPENPOSE_RECOVERY_FREE_FLOOR_BYTES = 106_929_602_242;
const OPENPOSE_CONTROL_DELTA_FLOOR = 0.01;
const RECOVERY_ORIGINAL_ARTIFACT_ID = "9477529627";
const RECOVERY_ORIGINAL_ARTIFACT_NAME = "sc-20945-epic-20738-8886a9e69f26beec05688c81b414859bd102f6d0-32570707303-1";
const RECOVERY_ORIGINAL_ARTIFACT_DIGEST = "sha256:f3164a32a485fdedd671f4e11f30038213d30a7eb2b541bda90bef30e63188f3";
const RECOVERY_BOUNDARY_LOG = {
  path: "controller.log", bytes: 47,
  sha256: "6043cbeeeed54deec723c1ab5fcec6b36a8de4f7b31928c6b57222fd7dfec770",
};
const LEGACY_NAIVE_LINK_LENGTH_SOURCE_BYTES = 173_667_044_229;
const REVIEWED_ALL_AT_ONCE_SOURCE_BYTES = 179_028_698_654;
const LEGACY_REVIEWED_ALL_AT_ONCE_SOURCE_BYTES = 179_028_698_264;
const PRE_HYDRATION_JIT_SOURCE_PEAK_BYTES = 56_156_615_634;
const REVIEWED_JIT_SOURCE_PEAK_BYTES = 63_979_929_282;
const NON_MODEL_DISK_RESERVE_BYTES = 40 * 1024 ** 3;
const PRE_HYDRATION_FREE_FLOOR_BYTES = 99_106_288_594;
const REVIEWED_FREE_FLOOR_BYTES = 106_929_602_242;
const FLUX_Q4_SIDECAR_BYTES = 7_396_392_960 + 494 * 16_384;
const FLUX_Q8_SIDECAR_BYTES = 12_573_868_032 + 494 * 16_384;
const DERIVED_DISPOSITION_NOT_APPLICABLE = "not-applicable";
const DERIVED_DISPOSITION_RESIDENT = "resident-empty";
const DERIVED_DISPOSITION_BOUNDED = "bounded-transformer-sidecars";
const DERIVED_DISPOSITION_PROVIDER_FAILED = "provider-failed-empty";
const REQUEST_MEMORY_STRATEGY_KEYS = [
  "requestMemoryPresent", "stageResidency", "strategy", "streamTransformerBlocks",
];
const CURRENT_DOWNLOAD_EVIDENCE_SHA256 = "1fa06ef39a0e2c321a4fa15fa1128c0157ba8cf22fd868ac54c6cefaec13a5ee";
const LEGACY_DOWNLOAD_EVIDENCE_SHA256 = "9eda09eeacb9386167ca4a080b4805b9c7dd3cd5134ca037ce342ad434b17e0b";
const SCAIL2_REFERENCE_DELTA_FLOOR = 1e-6;

export function reviewedMissingDownloadPlan({
  frozenMissing, profile, artifactExpectedFiles, downloadEvidenceSha256,
}) {
  const groups = new Map();
  for (const row of frozenMissing) {
    const group = groups.get(row.artifactId) ?? [];
    group.push(row);
    groups.set(row.artifactId, group);
  }
  const plan = [];
  for (const [artifactId, rows] of groups) {
    const artifact = profile.artifacts[artifactId];
    if (!artifact) fail(`reviewed missing-file plan referenced unknown authority ${artifactId}`);
    if (artifactId === REVIEWED_FLUX_MISSING.artifactId) {
      if (rows.length !== 1 || JSON.stringify(rows[0]) !== JSON.stringify(REVIEWED_FLUX_MISSING)) {
        fail("Flux hydration drifted from the sole reviewed missing file");
      }
    } else {
      const revision = ILLUSTRIOUS_CURRENT_AUTHORITIES.get(artifactId);
      const expected = artifactExpectedFiles[artifactId];
      const prefix = "q4/";
      const rowFiles = rows.map((row) => row.file);
      const missingSet = new Set(rowFiles.map((file) => (
        file.startsWith(prefix) ? file.slice(prefix.length) : file
      )));
      const canonicalMissing = expected?.filter((file) => missingSet.has(file)).map(
        (file) => `${prefix}${file}`,
      );
      if (downloadEvidenceSha256 !== CURRENT_DOWNLOAD_EVIDENCE_SHA256
        || artifact.revision !== revision || artifact.subdirectory !== "q4"
        || JSON.stringify(artifact.allowPatterns) !== JSON.stringify(["q4/*"])
        || !Array.isArray(expected)
        || canonicalSha256(expected) !== ILLUSTRIOUS_Q4_FILE_LIST_SHA256
        || missingSet.size !== rows.length
        || JSON.stringify(rowFiles) !== JSON.stringify(canonicalMissing)
        || rows.some((row) => row.repository !== artifact.repository
          || row.revision !== revision || !row.file.startsWith(prefix)
          || !expected.includes(row.file.slice(prefix.length)))) {
        fail(`Illustrious hydration drifted from exact current q4 authority ${artifactId}`);
      }
    }
    plan.push({ id: artifactId, missingFiles: rows.map((row) => row.file) });
  }
  return plan;
}

function fail(message) {
  throw new Error(message);
}

function object(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${label} must be an object`);
  return value;
}

function exactKeys(value, expected, label) {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    fail(`${label} fields must be exactly ${wanted.join(", ")}; got ${actual.join(", ")}`);
  }
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonicalize(value[key])]));
  }
  return value;
}

function canonicalSha256(value) {
  return createHash("sha256").update(JSON.stringify(canonicalize(value))).digest("hex");
}

const DOWNLOADED_FILE_FIELDS = ["path", "bytes", "sha256", "lfsSha256", "commitSha"];

function canonicalDownloadedFiles(files, label) {
  if (!Array.isArray(files)) fail(`${label} must be an array`);
  const paths = new Set();
  return files.map((file, index) => {
    object(file, `${label}[${index}]`);
    exactKeys(file, DOWNLOADED_FILE_FIELDS, `${label}[${index}]`);
    if (typeof file.path !== "string" || !file.path || file.path.includes("\\")
      || file.path.startsWith("/") || file.path.includes("..")
      || !Number.isSafeInteger(file.bytes) || file.bytes < 1
      || !/^[0-9a-f]{64}$/.test(file.sha256) || !/^[0-9a-f]{64}$/.test(file.lfsSha256)
      || !SHA40.test(file.commitSha) || paths.has(file.path)) {
      fail(`${label}[${index}] is malformed or ambiguous`);
    }
    paths.add(file.path);
    return Object.fromEntries(DOWNLOADED_FILE_FIELDS.map((field) => [field, file[field]]));
  });
}

export function assertExactDownloadedFilePartition(actual, expected, label) {
  const left = canonicalDownloadedFiles(actual, `${label} actual`);
  const right = canonicalDownloadedFiles(expected, `${label} expected`);
  if (left.length !== right.length || left.some((file, index) => (
    JSON.stringify(file) !== JSON.stringify(right[index])
  ))) fail(`${label} drifted from the frozen downloaded-file partition`);
}

function exactExecutionOrdinals(profile, ordinals, label = "execution ordinals") {
  if (!Array.isArray(ordinals) || ordinals.length === 0
    || ordinals.some((ordinal) => !Number.isInteger(ordinal)
      || ordinal < 1 || ordinal > profile.cells.length)
    || new Set(ordinals).size !== ordinals.length
    || ordinals.some((ordinal, index) => index > 0 && ordinal <= ordinals[index - 1])) {
    fail(`${label} must be a non-empty strictly increasing in-range integer list`);
  }
  return [...ordinals];
}

export function authorityLifetimePlan(profile, executionOrdinals, sourceAudits) {
  const ordinals = exactExecutionOrdinals(profile, executionOrdinals);
  const cells = ordinals.map((ordinal) => profile.cells[ordinal - 1]);
  const ids = [...new Set(cells.flatMap((cell) => cell.artifactIds))];
  return ids.map((artifactId) => {
    const uses = [];
    for (const ordinal of ordinals) {
      if (profile.cells[ordinal - 1].artifactIds.includes(artifactId)) uses.push(ordinal);
    }
    const audit = sourceAudits instanceof Map ? sourceAudits.get(artifactId) : sourceAudits[artifactId];
    if (!audit) fail(`source audit is missing lifetime authority ${artifactId}`);
    const files = [
      ...audit.reusedFiles.map((file) => ({ ...file, persistent: false })),
      ...(audit.downloadedFiles ?? []).map((file) => ({ ...file, persistent: true })),
    ];
    const sourceBytes = files.reduce((sum, file) => sum + file.bytes, 0);
    if (!Number.isSafeInteger(sourceBytes) || sourceBytes < 1) {
      fail(`source byte census is invalid for lifetime authority ${artifactId}`);
    }
    return {
      artifactId,
      role: profile.artifacts[artifactId].role,
      firstOrdinal: uses[0],
      lastOrdinal: uses.at(-1),
      sourceBytes,
      expectedFiles: audit.expectedFiles?.length
        ?? audit.reusedFiles.length + (audit.downloadedFiles ?? []).length,
      physicalFiles: files.map((file) => ({
        key: `${profile.artifacts[artifactId].repository}@${profile.artifacts[artifactId].revision}/${
          profile.artifacts[artifactId].subdirectory === "."
            ? file.path : `${profile.artifacts[artifactId].subdirectory}/${file.path}`
        }`,
        bytes: file.bytes,
        persistent: file.persistent,
      })),
    };
  });
}

export function estimateJitDiskPlan(
  lifetimes,
  freeBytes,
  persistentMissingBytes = 0,
  nonModelPaths = [],
  reviewedAllAtOnceSourceBytes = REVIEWED_ALL_AT_ONCE_SOURCE_BYTES,
  reviewedJitSourcePeakBytes = REVIEWED_JIT_SOURCE_PEAK_BYTES,
  reviewedFreeFloorBytes = REVIEWED_FREE_FLOOR_BYTES,
) {
  if (!Number.isSafeInteger(freeBytes) || freeBytes < 0
    || !Number.isSafeInteger(persistentMissingBytes) || persistentMissingBytes < 0
    || !Number.isSafeInteger(reviewedAllAtOnceSourceBytes) || reviewedAllAtOnceSourceBytes < 0) {
    fail("disk estimator requires exact non-negative byte counts");
  }
  const ordinals = [...new Set(lifetimes.flatMap((row) => [row.firstOrdinal, row.lastOrdinal]))]
    .sort((left, right) => left - right);
  const cells = ordinals.map((ordinal) => {
    const active = lifetimes.filter((row) => (
      row.firstOrdinal <= ordinal && row.lastOrdinal >= ordinal
    ));
    const physical = new Map();
    // Reviewed downloads persist across preflight and earlier cells, then become their authority's
    // stage root and are deleted at that authority's exact release boundary.
    for (const row of lifetimes) {
      if (ordinal > row.lastOrdinal) continue;
      for (const file of row.physicalFiles.filter((entry) => entry.persistent)) {
        physical.set(file.key, file.bytes);
      }
    }
    for (const file of active.flatMap((row) => row.physicalFiles)) {
      const existing = physical.get(file.key);
      if (existing !== undefined && existing !== file.bytes) {
        fail(`physical source identity has conflicting byte sizes: ${file.key}`);
      }
      physical.set(file.key, file.bytes);
    }
    const stagedBytes = [...physical.values()].reduce((sum, bytes) => sum + bytes, 0);
    const sidecarReserveBytes = active.reduce((sum, row) => {
      if (!row.artifactId.startsWith("flux1-")) return sum;
      return sum + (row.artifactId.endsWith("-q4")
        ? FLUX_Q4_SIDECAR_BYTES : FLUX_Q8_SIDECAR_BYTES);
    }, 0);
    const modelAndSidecarBytes = stagedBytes + sidecarReserveBytes;
    return {
      ordinal,
      artifactIds: active.map((row) => row.artifactId),
      stagedBytes,
      sidecarReserveBytes,
      modelAndSidecarBytes,
      requiredAdditionalBytes: Math.max(
        modelAndSidecarBytes + NON_MODEL_DISK_RESERVE_BYTES,
        reviewedFreeFloorBytes,
      ),
    };
  });
  const peak = cells.reduce((highest, row) => (
    row.modelAndSidecarBytes > highest.modelAndSidecarBytes ? row : highest
  ), { ordinal: 0, artifactIds: [], stagedBytes: 0, sidecarReserveBytes: 0,
    modelAndSidecarBytes: 0, requiredAdditionalBytes: reviewedFreeFloorBytes });
  const allPhysical = new Map();
  for (const file of lifetimes.flatMap((row) => row.physicalFiles)) {
    const existing = allPhysical.get(file.key);
    if (existing !== undefined && existing !== file.bytes) {
      fail(`physical source identity has conflicting byte sizes: ${file.key}`);
    }
    allPhysical.set(file.key, file.bytes);
  }
  const logicalSourceBytes = lifetimes.reduce((sum, row) => sum + row.sourceBytes, 0);
  const allAtOnceSourceBytes = [...allPhysical.values()].reduce((sum, bytes) => sum + bytes, 0);
  const persistentBytesFromPlan = [...new Map(lifetimes.flatMap((row) => (
    row.physicalFiles.filter((file) => file.persistent).map((file) => [file.key, file.bytes])
  ))).values()].reduce((sum, bytes) => sum + bytes, 0);
  if (persistentBytesFromPlan !== persistentMissingBytes) {
    fail(`persistent missing-file byte census drifted: plan=${persistentBytesFromPlan}, input=${persistentMissingBytes}`);
  }
  const currentIllustriousHydrationBytes = lifetimes.filter((row) => (
    ILLUSTRIOUS_CURRENT_AUTHORITIES.has(row.artifactId)
  )).flatMap((row) => row.physicalFiles).filter((file) => file.persistent)
    .reduce((sum, file) => sum + file.bytes, 0);
  const modeledJitPeakBytes = PRE_HYDRATION_JIT_SOURCE_PEAK_BYTES
    + currentIllustriousHydrationBytes;
  const allAtOnceSidecarReserveBytes = Math.max(FLUX_Q4_SIDECAR_BYTES, FLUX_Q8_SIDECAR_BYTES);
  if (reviewedAllAtOnceSourceBytes === REVIEWED_ALL_AT_ONCE_SOURCE_BYTES
    && logicalSourceBytes === reviewedAllAtOnceSourceBytes
    && (allAtOnceSourceBytes !== reviewedAllAtOnceSourceBytes
      || modeledJitPeakBytes > reviewedJitSourcePeakBytes
      || peak.modelAndSidecarBytes !== modeledJitPeakBytes)) {
    fail(`reviewed disk census drifted: all=${allAtOnceSourceBytes}, peak=${peak.modelAndSidecarBytes}`);
  }
  return {
    freeBytes,
    persistentMissingBytes,
    nonModelReserveBytes: NON_MODEL_DISK_RESERVE_BYTES,
    nonModelPaths,
    legacyNaiveLinkLengthSourceBytes: LEGACY_NAIVE_LINK_LENGTH_SOURCE_BYTES,
    reviewedAllAtOnceSourceBytes,
    preHydrationJitSourcePeakBytes: PRE_HYDRATION_JIT_SOURCE_PEAK_BYTES,
    reviewedJitSourcePeakBytes,
    logicalSourceBytes,
    allAtOnceSourceBytes,
    allAtOnceRequiredBytes: allAtOnceSourceBytes + allAtOnceSidecarReserveBytes
      + NON_MODEL_DISK_RESERVE_BYTES,
    peakOrdinal: peak.ordinal,
    peakArtifactIds: peak.artifactIds,
    peakStagedBytes: peak.stagedBytes,
    peakSidecarReserveBytes: peak.sidecarReserveBytes,
    peakModelAndSidecarBytes: peak.modelAndSidecarBytes,
    peakRequiredAdditionalBytes: Math.max(
      peak.modelAndSidecarBytes + NON_MODEL_DISK_RESERVE_BYTES,
      reviewedFreeFloorBytes,
    ),
    admitted: freeBytes >= Math.max(
      peak.modelAndSidecarBytes + NON_MODEL_DISK_RESERVE_BYTES,
      reviewedFreeFloorBytes,
    ),
    cells,
  };
}

function isExactOrdinals(actual, expected) {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

function reviewedRecoveryDiskPlan(executionOrdinals) {
  if (isExactOrdinals(executionOrdinals, OPENPOSE_RECOVERY_EXECUTION_ORDINALS)) {
    return {
      sourceBytes: OPENPOSE_RECOVERY_SOURCE_BYTES,
      jitPeakBytes: OPENPOSE_RECOVERY_PEAK_STAGED_BYTES,
      freeFloorBytes: OPENPOSE_RECOVERY_FREE_FLOOR_BYTES,
    };
  }
  return {
    sourceBytes: REVIEWED_ALL_AT_ONCE_SOURCE_BYTES,
    jitPeakBytes: REVIEWED_JIT_SOURCE_PEAK_BYTES,
    freeFloorBytes: REVIEWED_FREE_FLOOR_BYTES,
  };
}

// The OpenPose continuation is deliberately a closed selector, not a permissive small-campaign
// mode. Every byte and authority below is bound to the diagnostic census from run 32679720253.
export function assertExactOpenPoseRecoveryPreflight({
  executionOrdinals, remainingArtifactIds, artifactExpectedFiles, lifetimePlan, diskPlan,
}) {
  if (!isExactOrdinals(executionOrdinals, OPENPOSE_RECOVERY_EXECUTION_ORDINALS)) {
    fail("OpenPose recovery execution vector is not the exact reviewed [15,16,17,18,19] selector");
  }
  if (JSON.stringify(remainingArtifactIds) !== JSON.stringify(OPENPOSE_RECOVERY_REMAINING_ARTIFACT_IDS)
    || remainingArtifactIds.length !== OPENPOSE_RECOVERY_REMAINING_AUTHORITIES
    || remainingArtifactIds.reduce((sum, id) => sum + artifactExpectedFiles[id].length, 0)
      !== OPENPOSE_RECOVERY_REMAINING_FILES
    || JSON.stringify(lifetimePlan.map((row) => row.artifactId))
      !== JSON.stringify(OPENPOSE_RECOVERY_REMAINING_ARTIFACT_IDS)) {
    fail("OpenPose recovery authority set drifted from the exact five-cell plan");
  }
  if (diskPlan.reviewedAllAtOnceSourceBytes !== OPENPOSE_RECOVERY_SOURCE_BYTES
    || diskPlan.logicalSourceBytes !== OPENPOSE_RECOVERY_SOURCE_BYTES
    || diskPlan.allAtOnceSourceBytes !== OPENPOSE_RECOVERY_SOURCE_BYTES
    || diskPlan.peakOrdinal !== OPENPOSE_RECOVERY_PEAK_ORDINAL
    || JSON.stringify(diskPlan.peakArtifactIds) !== JSON.stringify(OPENPOSE_RECOVERY_PEAK_ARTIFACT_IDS)
    || diskPlan.peakStagedBytes !== OPENPOSE_RECOVERY_PEAK_STAGED_BYTES
    || diskPlan.peakSidecarReserveBytes !== 0
    || diskPlan.peakModelAndSidecarBytes !== OPENPOSE_RECOVERY_PEAK_STAGED_BYTES
    || diskPlan.reviewedJitSourcePeakBytes !== OPENPOSE_RECOVERY_PEAK_STAGED_BYTES
    || diskPlan.peakRequiredAdditionalBytes !== OPENPOSE_RECOVERY_FREE_FLOOR_BYTES) {
    fail("OpenPose recovery disk/JIT plan drifted from the exact five-cell census");
  }
}

export function cellSemanticsSha256(cells) {
  const tuples = cells.map((cell) => ({
    id: cell.id,
    modelId: cell.modelId,
    engineId: cell.engineId,
    kind: cell.kind,
    requestedTier: cell.requestedTier,
    capability: cell.capability ?? null,
    artifactIds: cell.artifactIds,
    request: cell.request,
  }));
  return canonicalSha256(tuples);
}

// Match the worker's family policy for immutable packed roots. The selected tier remains q4/q8
// evidence, while this value records whether the final LoadSpec asks the provider to quantize at
// load. SDXL keeps its production advisory Q4 selector, and Candle SCAIL carries the exact requested
// Q4/Q8 tier hint used by its production cold-load plan.
export function expectedLoadSpecQuantBits(cell) {
  const family = `${cell?.kind ?? ""}/${cell?.engineId ?? ""}`;
  if (cell?.kind === "sdxlOpenPose" && cell.engineId === "sdxl") return 4;
  if (cell?.kind === "scail2" && cell.engineId === "scail2_14b") {
    if (cell.requestedTier === "q4") return 4;
    if (cell.requestedTier === "q8") return 8;
    fail(`unreviewed terminal SCAIL tier ${cell.requestedTier}`);
  }
  if ((cell?.kind === "image" && new Set([
    "chroma1_base", "chroma1_flash", "chroma1_hd", "flux1_dev", "flux1_schnell",
  ]).has(cell.engineId))
    || (cell?.kind === "ltx" && cell.engineId === "ltx_2_3_distilled")) return null;
  fail(`unreviewed terminal load-quant family ${family}`);
}

export function validateRuntimeResult(runtimeResult, cell) {
  object(runtimeResult, `${cell.id} runtime result`);
  const expectedQuant = expectedLoadSpecQuantBits(cell);
  if (runtimeResult.requestedTier !== cell.requestedTier
    || runtimeResult.resolvedTier !== cell.requestedTier || runtimeResult.denseFallback !== false) {
    fail(`${cell.id} runtime result did not prove exact-tier/no-fallback execution`);
  }
  if (!Object.hasOwn(runtimeResult, "loadSpecQuantBits")
    || runtimeResult.loadSpecQuantBits !== expectedQuant) {
    fail(`${cell.id} runtime result did not prove family loadSpecQuantBits=${expectedQuant}`);
  }
  validateRequestMemoryStrategy(runtimeResult.requestMemoryStrategy, cell);
  validateScail2ReferenceCounterfactuals(runtimeResult.metrics, cell, "runtime result");
  validateOpenPoseControlCounterfactual(runtimeResult.metrics, cell, "runtime result");
  return runtimeResult;
}

// The ordered six-reference cell needs causal evidence, not merely one successful render with six
// inputs. Each same-seed leave-one-out output must have a nontrivial pixel delta and position-bound
// witnesses. Keeping this pure makes controller/schema mutation tests exercise the evidence contract
// without a CUDA run.
export function validateScail2ReferenceCounterfactuals(metrics, cell, label = "metrics") {
  const isMultiReference = cell?.kind === "scail2" && cell?.capability === "multiReference";
  if (!isMultiReference) {
    if (metrics?.referenceCounterfactuals != null) {
      fail(`${cell.id} ${label} carries unreviewed SCAIL2 reference counterfactual evidence`);
    }
    return null;
  }
  object(metrics, `${cell.id} ${label}`);
  if (metrics.kind !== "scail2" || metrics.referencePairs !== 6) {
    fail(`${cell.id} ${label} does not bind the ordered six-reference SCAIL2 render`);
  }
  const counterfactuals = metrics.referenceCounterfactuals;
  if (!Array.isArray(counterfactuals) || counterfactuals.length !== 6) {
    fail(`${cell.id} ${label} must contain exactly six leave-one-reference-out deltas`);
  }
  for (const [index, evidence] of counterfactuals.entries()) {
    const reference = index + 1;
    object(evidence, `${cell.id} ${label} reference ${reference}`);
    exactKeys(
      evidence,
      ["reference", "meanAbsDelta", "firstFrameMeanAbsDelta", "lastFrameMeanAbsDelta", "witnesses"],
      `${cell.id} ${label} reference ${reference}`,
    );
    if (evidence.reference !== reference) {
      fail(`${cell.id} ${label} references must be exactly ordered 1 through 6`);
    }
    if (!Number.isFinite(evidence.meanAbsDelta) || evidence.meanAbsDelta <= SCAIL2_REFERENCE_DELTA_FLOOR) {
      fail(`${cell.id} ${label} reference ${reference} has zero, nonfinite, or trivial causal delta`);
    }
    if (!Number.isFinite(evidence.firstFrameMeanAbsDelta)
      || !Number.isFinite(evidence.lastFrameMeanAbsDelta)
      || evidence.firstFrameMeanAbsDelta < 0 || evidence.lastFrameMeanAbsDelta < 0) {
      fail(`${cell.id} ${label} reference ${reference} has malformed frame witnesses`);
    }
    object(evidence.witnesses, `${cell.id} ${label} reference ${reference} witnesses`);
    exactKeys(evidence.witnesses, ["omittedReference", "firstFrame", "lastFrame"],
      `${cell.id} ${label} reference ${reference} witnesses`);
    if (evidence.witnesses.omittedReference !== `input-reference-${reference}.png`
      || evidence.witnesses.firstFrame !== `counterfactual-reference-${reference}-first.png`
      || evidence.witnesses.lastFrame !== `counterfactual-reference-${reference}-last.png`) {
      fail(`${cell.id} ${label} reference ${reference} witnesses are not position-bound`);
    }
  }
  return counterfactuals;
}

// The five OpenPose cells are causal tests, not merely nondegenerate-image checks. The candidate
// keeps the seed and every non-control input fixed, swaps only to a mirrored whole-body pose, and
// records both input and output witnesses. An implementation that drops Conditioning::Control
// produces the same seeded bytes and fails the 0.01 mean-absolute-delta floor.
export function validateOpenPoseControlCounterfactual(metrics, cell, label = "metrics") {
  const isOpenPose = cell?.kind === "sdxlOpenPose";
  if (!isOpenPose) {
    if (metrics?.controlCounterfactual != null) {
      fail(`${cell.id} ${label} carries unreviewed OpenPose control counterfactual evidence`);
    }
    return null;
  }
  object(metrics, `${cell.id} ${label}`);
  const evidence = metrics.controlCounterfactual;
  object(evidence, `${cell.id} ${label} OpenPose control counterfactual`);
  exactKeys(
    evidence,
    ["kind", "sameSeed", "meanAbsDelta", "deltaFloor", "witnesses"],
    `${cell.id} ${label} OpenPose control counterfactual`,
  );
  if (evidence.kind !== "mirroredPose" || evidence.sameSeed !== true
    || evidence.deltaFloor !== OPENPOSE_CONTROL_DELTA_FLOOR
    || !Number.isFinite(evidence.meanAbsDelta)
    || evidence.meanAbsDelta <= OPENPOSE_CONTROL_DELTA_FLOOR) {
    fail(`${cell.id} ${label} does not prove material same-seed OpenPose control influence`);
  }
  object(evidence.witnesses, `${cell.id} ${label} OpenPose witnesses`);
  exactKeys(
    evidence.witnesses,
    ["baselineControl", "counterfactualControl", "baselineOutput", "counterfactualOutput"],
    `${cell.id} ${label} OpenPose witnesses`,
  );
  if (evidence.witnesses.baselineControl !== "input-openpose.png"
    || evidence.witnesses.counterfactualControl !== "input-openpose-counterfactual.png"
    || evidence.witnesses.baselineOutput !== "output.png"
    || evidence.witnesses.counterfactualOutput !== "counterfactual-output.png") {
    fail(`${cell.id} ${label} OpenPose witnesses are not immutable position-bound files`);
  }
  return evidence;
}

function validateRequestMemoryStrategy(strategy, cell) {
  object(strategy, `${cell.id} request memory strategy`);
  if (JSON.stringify(Object.keys(strategy).sort()) !== JSON.stringify(REQUEST_MEMORY_STRATEGY_KEYS)) {
    fail(`${cell.id} request memory strategy is not a closed exact record`);
  }
  const isFlux = cell.kind === "image"
    && new Set(["flux1_dev", "flux1_schnell"]).has(cell.engineId);
  const booleans = typeof strategy.requestMemoryPresent === "boolean"
    && typeof strategy.stageResidency === "boolean"
    && typeof strategy.streamTransformerBlocks === "boolean";
  if (!booleans) fail(`${cell.id} request memory strategy booleans are missing or malformed`);
  if (!isFlux) {
    if (strategy.strategy !== "not-applicable" || strategy.requestMemoryPresent
      || strategy.stageResidency || strategy.streamTransformerBlocks) {
      fail(`${cell.id} non-FLUX request memory strategy is not exact not-applicable`);
    }
    return strategy;
  }
  const exact = (name, present, stage, stream) => strategy.strategy === name
    && strategy.requestMemoryPresent === present
    && strategy.stageResidency === stage
    && strategy.streamTransformerBlocks === stream;
  if (exact("default-resident", false, false, false)
    || exact("explicit-resident", true, false, false)
    || exact("staged-resident", true, true, false)
    || exact("bounded-transformer", true, true, true)) {
    return strategy;
  }
  fail(`${cell.id} FLUX request memory strategy is not an exact supported policy`);
}

function dispositionForMemoryStrategy(strategy) {
  if (strategy.strategy === "not-applicable") return DERIVED_DISPOSITION_NOT_APPLICABLE;
  if (strategy.strategy === "bounded-transformer") return DERIVED_DISPOSITION_BOUNDED;
  if (new Set(["default-resident", "explicit-resident", "staged-resident"])
    .has(strategy.strategy)) return DERIVED_DISPOSITION_RESIDENT;
  fail(`unreviewed request memory strategy ${strategy.strategy}`);
}

export function loadProfile(profilePath = PROFILE_PATH) {
  return JSON.parse(readFileSync(profilePath, "utf8"));
}

export function validateProfile(profile) {
  object(profile, "profile");
  if (profile.schemaVersion !== 1 || profile.epic !== 20738 || profile.story !== 20945) {
    fail("profile identity must be schema v1 for epic 20738 / story 20945");
  }
  if (profile.profile !== PROFILE_NAME) fail(`profile name must be ${PROFILE_NAME}`);
  object(profile.artifacts, "profile.artifacts");
  if (Object.keys(profile.artifacts).length !== 23) {
    fail(`profile must contain exactly 23 reviewed artifact authorities`);
  }
  if (!Array.isArray(profile.cells) || profile.cells.length !== 19) {
    fail(`profile must contain exactly 19 cells, got ${profile.cells?.length ?? "non-array"}`);
  }
  const ids = profile.cells.map((cell) => cell.id);
  if (JSON.stringify(ids) !== JSON.stringify(EXPECTED_CELLS)) {
    fail(`profile cells or serial order drifted:\nexpected ${EXPECTED_CELLS.join(",")}\nactual ${ids.join(",")}`);
  }
  if (new Set(ids).size !== ids.length) fail("profile cell ids must be unique");
  const semanticDigest = cellSemanticsSha256(profile.cells);
  if (semanticDigest !== EXPECTED_CELL_SEMANTICS_SHA256) {
    fail(`profile's exact ordered semantic tuples drifted: ${semanticDigest}`);
  }

  for (const [id, artifact] of Object.entries(profile.artifacts)) {
    if (!SAFE_ID.test(id)) fail(`artifact id is not safe: ${id}`);
    const illustriousRevision = ILLUSTRIOUS_CURRENT_AUTHORITIES.get(id);
    exactKeys(
      artifact,
      illustriousRevision
        ? ["authority", "role", "repository", "revision", "subdirectory", "allowPatterns", "quantizationMarker"]
        : ["authority", "role", "repository", "revision", "subdirectory", "allowPatterns"],
      `artifact ${id}`,
    );
    object(artifact.authority, `artifact ${id}.authority`);
    if (artifact.authority.kind === "manifest") {
      const selectors = ["variant", "componentId", "component"].filter((key) => artifact.authority[key]);
      if (selectors.length !== 1) fail(`artifact ${id} must have exactly one manifest selector`);
      exactKeys(artifact.authority, ["kind", "model", selectors[0]], `artifact ${id}.authority`);
    } else if (artifact.authority.kind === "explicitPublicArtifact") {
      exactKeys(artifact.authority, ["kind", "file"], `artifact ${id}.authority`);
    } else {
      fail(`artifact ${id} has unknown authority kind ${artifact.authority.kind}`);
    }
    if (!SHA40.test(artifact.revision)) fail(`artifact ${id} revision must be exact lowercase 40-hex`);
    if (illustriousRevision && (artifact.revision !== illustriousRevision
      || canonicalSha256(artifact.quantizationMarker) !== canonicalSha256(ILLUSTRIOUS_Q4_MARKER))) {
      fail(`artifact ${id} must bind its exact current revision and q4/group-64 component markers`);
    }
    if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(artifact.repository)) {
      fail(`artifact ${id} repository must be exact owner/name`);
    }
    if (!Array.isArray(artifact.allowPatterns) || artifact.allowPatterns.length === 0
      || artifact.allowPatterns.some((pattern) => typeof pattern !== "string" || !pattern
        || path.isAbsolute(pattern) || pattern.split(/[\\/]/).includes(".."))) {
      fail(`artifact ${id} allowPatterns must be non-empty confined relative patterns`);
    }
    if (path.isAbsolute(artifact.subdirectory) || artifact.subdirectory.split(/[\\/]/).includes("..")) {
      fail(`artifact ${id} subdirectory must stay inside the snapshot`);
    }
    const searchable = JSON.stringify({ id, artifact }).toLowerCase();
    const blocked = BLOCKED.find((word) => searchable.includes(word));
    if (blocked) fail(`artifact ${id} touches blocked surface ${blocked}`);
  }

  let multireference = 0;
  const poseModels = [];
  for (const [index, cell] of profile.cells.entries()) {
    object(cell, `cell ${index + 1}`);
    exactKeys(
      cell,
      cell.capability
        ? ["id", "kind", "modelId", "engineId", "requestedTier", "capability", "artifactIds", "request"]
        : ["id", "kind", "modelId", "engineId", "requestedTier", "artifactIds", "request"],
      `cell ${index + 1}`,
    );
    object(cell.request, `cell ${cell.id}.request`);
    if (!SAFE_ID.test(cell.id)) fail(`unsafe cell id ${cell.id}`);
    if (!new Set(["image", "scail2", "ltx", "sdxlOpenPose"]).has(cell.kind)) {
      fail(`cell ${cell.id} has unsupported kind ${cell.kind}`);
    }
    if (!new Set(["q4", "q8"]).has(cell.requestedTier)) fail(`cell ${cell.id} must request q4 or q8`);
    if (!Array.isArray(cell.artifactIds) || cell.artifactIds.length === 0
      || cell.artifactIds.some((id) => !profile.artifacts[id])) {
      fail(`cell ${cell.id} must reference only declared artifacts`);
    }
    const primary = cell.artifactIds.map((id) => profile.artifacts[id]).filter((artifact) => artifact.role === "primary");
    if (primary.length !== 1) fail(`cell ${cell.id} must have exactly one primary artifact`);
    if (primary[0].subdirectory !== cell.requestedTier) {
      fail(`cell ${cell.id} requested tier does not match its immutable primary subdirectory`);
    }
    const searchable = JSON.stringify(cell).toLowerCase();
    const blocked = BLOCKED.find((word) => searchable.includes(word));
    if (blocked) fail(`cell ${cell.id} touches blocked surface ${blocked}`);
    if (cell.capability === "multiReference") {
      multireference += 1;
      if (cell.modelId !== "scail2_14b" || cell.kind !== "scail2" || cell.request.referencePairs !== 6) {
        fail("the sole multiReference cell must be the ordered six-pair SCAIL2 boundary");
      }
    } else if (cell.request.referencePairs && cell.request.referencePairs !== 1) {
      fail(`cell ${cell.id} has an unreviewed reference-pair count`);
    }
    if (cell.kind === "sdxlOpenPose") poseModels.push(cell.modelId);
  }
  if (multireference !== 1) fail("profile must contain exactly one SCAIL2 multiReference cell");
  if (JSON.stringify(poseModels) !== JSON.stringify(SDXL_POSE_MODELS)) {
    fail(`OpenPose cells must cover the exact five approved SDXL backbones: ${SDXL_POSE_MODELS.join(", ")}`);
  }
  const artifactDigest = canonicalSha256(profile.artifacts);
  if (artifactDigest !== EXPECTED_ARTIFACT_SEMANTICS_SHA256) {
    fail(`profile's exact artifact definitions drifted: ${artifactDigest}`);
  }
  return profile;
}

export function expectedArtifactFilesFromEvidence(profile, evidence) {
  object(evidence, "download-pattern evidence");
  exactKeys(evidence, ["repos"], "download-pattern evidence");
  if (!Array.isArray(evidence.repos)) fail("download-pattern evidence repos must be an array");
  const rows = new Map();
  for (const row of evidence.repos) {
    object(row, "download-pattern evidence row");
    const key = claimKey(row.repo, row.revision);
    if (row.key !== key || rows.has(key) || row.resolvedSha !== row.revision
      || row.servedRepo !== row.repo || row.gated !== false || !SHA40.test(row.revision)
      || !Array.isArray(row.files) || row.files.length === 0
      || row.files.some((file) => typeof file !== "string" || !file
        || path.isAbsolute(file) || file.split(/[\\/]/).includes(".."))
      || new Set(row.files).size !== row.files.length
      || JSON.stringify([...row.files].sort()) !== JSON.stringify(row.files)) {
      fail(`download-pattern evidence row is not an exact immutable file census: ${key}`);
    }
    rows.set(key, row);
  }
  const result = {};
  for (const [id, artifact] of Object.entries(profile.artifacts)) {
    const key = claimKey(artifact.repository, artifact.revision);
    const row = rows.get(key);
    if (!row) fail(`download-pattern evidence is missing exact authority ${key}`);
    const matches = new Set();
    for (const pattern of artifact.allowPatterns) {
      const regexp = patternToRegExp(pattern);
      const patternMatches = row.files.filter((file) => regexp.test(file));
      if (patternMatches.length === 0) {
        fail(`download-pattern evidence pattern has no exact files for ${id}: ${pattern}`);
      }
      patternMatches.forEach((file) => matches.add(file));
    }
    const prefix = artifact.subdirectory === "." ? "" : `${artifact.subdirectory}/`;
    const expectedFiles = [...matches].sort().map((file) => {
      if (prefix && !file.startsWith(prefix)) {
        fail(`download-pattern evidence file escaped selected subdirectory for ${id}: ${file}`);
      }
      const relative = prefix ? file.slice(prefix.length) : file;
      if (!relative || relative.split("/").includes("..")) {
        fail(`download-pattern evidence produced an invalid selected filename for ${id}: ${file}`);
      }
      return relative;
    });
    if (new Set(expectedFiles).size !== expectedFiles.length) {
      fail(`download-pattern evidence produced duplicate selected filenames for ${id}`);
    }
    result[id] = expectedFiles;
  }
  return result;
}

export function expectedCurrentArtifactFilesFromEvidenceBytes(profile, evidenceBytes) {
  const digest = createHash("sha256").update(evidenceBytes).digest("hex");
  if (digest !== CURRENT_DOWNLOAD_EVIDENCE_SHA256) {
    fail(`current download-pattern evidence digest drifted: ${digest}`);
  }
  let evidence;
  try {
    evidence = JSON.parse(evidenceBytes);
  } catch (error) {
    fail(`current download-pattern evidence is not JSON: ${error.message}`);
  }
  return {
    artifactExpectedFiles: expectedArtifactFilesFromEvidence(profile, evidence),
    downloadEvidenceSha256: digest,
  };
}

function expectedLegacyArtifactFilesFromEvidence(profile, evidence) {
  const v1 = profile.artifacts?.["illustrious-v1-q4"];
  const v2 = profile.artifacts?.["illustrious-v2-q4"];
  if (v1?.revision !== ILLUSTRIOUS_V1_LEGACY_REVISION
    || v2?.revision !== ILLUSTRIOUS_V2_LEGACY_REVISION
    || v1.quantizationMarker !== undefined || v2.quantizationMarker !== undefined) {
    fail("legacy download-pattern selector is not bound to the frozen Illustrious authorities");
  }
  return expectedArtifactFilesFromEvidence(profile, evidence);
}

export function validateManifestAuthorities(profile, manifest) {
  validateProfile(profile);
  const models = new Map(manifest.models.map((model) => [model.id, model]));
  for (const [id, artifact] of Object.entries(profile.artifacts)) {
    const authority = artifact.authority;
    if (authority.kind === "manifest") {
      const model = models.get(authority.model);
      if (!model) fail(`artifact ${id} authority model is absent: ${authority.model}`);
      const row = (model.downloads ?? []).find((download) => {
        if (authority.variant) return download.variant === authority.variant;
        if (authority.componentId) return download.componentId === authority.componentId;
        if (authority.component === "gemma") return (download.files ?? []).some((file) => file.startsWith("gemma/"));
        return false;
      });
      if (!row) fail(`artifact ${id} has no matching manifest authority row`);
      const approvedLtxParent = new Set(["ltx23-q8", "ltx23-gemma"]).has(id)
        && artifact.repository === "SceneWorks/ltx-2.3-mlx"
        && artifact.revision === LTX_APPROVED_PARENT_REVISION
        && row.revision === LTX_CURRENT_REVISION;
      if (row.repo !== artifact.repository
        || (row.revision !== artifact.revision && !approvedLtxParent)) {
        fail(`artifact ${id} identity disagrees with manifest authority`);
      }
      if (JSON.stringify(artifact.allowPatterns) !== JSON.stringify(row.files ?? [])) {
        fail(`artifact ${id} allowPatterns are not the manifest's exact download surface`);
      }
      if (artifact.quantizationMarker) {
        const policies = (model.mlx?.memoryStrategyContract?.implementations ?? []).flatMap(
          (implementation) => implementation.parameterRanges?.decodeGeometryPolicies ?? [],
        );
        if (policies.length === 0) {
          fail(`artifact ${id} has no active MLX decode policy to bind to its current authority`);
        }
        const expectedFingerprint = `${artifact.repository}@${artifact.revision}:q4`;
        for (const policy of policies) {
          const active = policy.artifact;
          if (active?.repository !== artifact.repository || active.revision !== artifact.revision
            || active.variant !== "q4" || active.fingerprint !== expectedFingerprint) {
            fail(`artifact ${id} active MLX memory strategy is not bound to its current authority`);
          }
        }
      }
    } else if (authority.kind === "explicitPublicArtifact") {
      // The frozen profile keeps one shared authority for all five measured cells. Only the
      // independently promotable cells bind that authority to production sc-20747 soft
      // co-requisite rows. The final sparse recovery promotes all five measured cells.
      const expected = {
        repository: "xinsir/controlnet-openpose-sdxl-1.0",
        revision: "23f966cd5cfdd3f7729c903e243d87152162d2b7",
        file: "diffusion_pytorch_model.safetensors",
      };
      if (artifact.repository !== expected.repository || artifact.revision !== expected.revision
        || authority.file !== expected.file || artifact.subdirectory !== "."
        || artifact.allowPatterns.length !== 1 || artifact.allowPatterns[0] !== expected.file) {
        fail(`artifact ${id} drifted from the frozen public OpenPose ControlNet tuple`);
      }
      const consumers = manifest.models.flatMap((model) => (model.downloads ?? [])
        .filter((download) => download.componentId === "controlnet_openpose")
        .map((download) => ({ model: model.id, download })));
      const actualModels = consumers.map(({ model }) => model).sort();
      const expectedModels = [...PROMOTED_SDXL_POSE_MODELS].sort();
      if (JSON.stringify(actualModels) !== JSON.stringify(expectedModels)) {
        fail(`OpenPose ControlNet manifest authority must have the exact five promoted consumers`);
      }
      for (const modelId of PROMOTED_SDXL_POSE_MODELS) {
        const rows = consumers.filter(({ model }) => model === modelId);
        if (rows.length !== 1) {
          fail(`OpenPose ControlNet manifest authority must have exactly one row for ${modelId}`);
        }
        const row = rows[0].download;
        if (row.provider !== "huggingface" || row.repo !== expected.repository
          || row.revision !== expected.revision || row.coRequisite !== true
          || row.required !== "soft" || JSON.stringify(row.files) !== JSON.stringify([expected.file])
          || JSON.stringify(row.platforms) !== JSON.stringify(["macos", "windows", "linux"])) {
          fail(`OpenPose ControlNet manifest authority drifted for ${modelId}`);
        }
      }
    } else {
      fail(`artifact ${id} has unknown authority kind ${authority.kind}`);
    }
  }
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { encoding: "utf8", windowsHide: true, ...options });
  if (result.status !== 0) {
    fail(`${command} ${args.join(" ")} failed (${result.status}): ${(result.stderr || result.stdout || "").trim()}`);
  }
  return result.stdout.trim();
}

function schemaValidator(schemaPath) {
  const absolute = path.resolve(ROOT, schemaPath);
  if (!SCHEMA_VALIDATORS.has(absolute)) {
    const ajv = new Ajv2020({ allErrors: true, allowUnionTypes: true, strict: true });
    addFormats(ajv);
    SCHEMA_VALIDATORS.set(absolute, ajv.compile(JSON.parse(readFileSync(absolute, "utf8"))));
  }
  return SCHEMA_VALIDATORS.get(absolute);
}

export function validateDocumentWithSchema(schemaPath, document) {
  const value = typeof document === "string"
    ? JSON.parse(readFileSync(path.resolve(document), "utf8"))
    : document;
  const validate = schemaValidator(schemaPath);
  if (!validate(value)) {
    const details = validate.errors.map((error) => `${error.instancePath || "/"}: ${error.message}`).join("\n");
    fail(`Draft 2020-12 validation failed for ${schemaPath}:\n${details}`);
  }
  return value;
}

function validateTrustedLegacyImportedReceiptDocument(receipt) {
  if (!LEGACY_IMPORTED_RECEIPT_VALIDATOR) {
    const absolute = path.resolve(ROOT, RECEIPT_SCHEMA_PATH);
    const schema = JSON.parse(readFileSync(absolute, "utf8"));
    schema.$id = schema.$id.replace(/\.json$/, "-trusted-legacy-import.json");
    schema.required = schema.required.filter((field) => field !== "authorityLifecycle");
    const ajv = new Ajv2020({ allErrors: true, allowUnionTypes: true, strict: true });
    addFormats(ajv);
    LEGACY_IMPORTED_RECEIPT_VALIDATOR = ajv.compile(schema);
  }
  if (!LEGACY_IMPORTED_RECEIPT_VALIDATOR(receipt)) {
    const details = LEGACY_IMPORTED_RECEIPT_VALIDATOR.errors
      .map((error) => `${error.instancePath || "/"}: ${error.message}`).join("\n");
    fail(`trusted legacy prefix receipt failed Draft 2020-12 validation:\n${details}`);
  }
  return receipt;
}

async function writeJsonAtomically(file, value, { schemaPath } = {}) {
  await mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.tmp-${process.pid}-${randomUUID()}`;
  try {
    await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    if (schemaPath) validateDocumentWithSchema(schemaPath, value);
    await rename(temporary, file);
  } finally {
    await rm(temporary, { force: true });
  }
}

function git(repo, args) {
  return run("git", ["-C", repo, ...args]);
}

export function repositoryIdentity(repo, label) {
  const root = path.resolve(repo);
  const top = path.resolve(git(root, ["rev-parse", "--show-toplevel"]));
  if (top.toLowerCase() !== root.toLowerCase()) fail(`${label} repository root mismatch: ${top} != ${root}`);
  const sha = git(root, ["rev-parse", "HEAD"]);
  if (!SHA40.test(sha)) fail(`${label} HEAD is not exact lowercase 40-hex`);
  const status = git(root, ["status", "--porcelain=v1", "--untracked-files=all"]);
  if (status) fail(`${label} repository must be clean before terminal measurement:\n${status}`);
  return { sha, clean: true, root };
}

export function inferencePins(cargoToml) {
  const pins = [...cargoToml.matchAll(/git\s*=\s*"https:\/\/github\.com\/SceneWorks\/inference(?:\.git)?"[^}\n]*?rev\s*=\s*"([0-9a-f]{40})"/g)]
    .map((match) => match[1]);
  return [...new Set(pins)];
}

// Every SceneWorks manifest that carries a `SceneWorks/inference` git pin. `scripts/bump-inference.mjs`
// rewrites exactly these three, so they are the repository's single inference-revision source.
export const INFERENCE_PIN_MANIFESTS = [
  "Cargo.toml",
  "crates/sceneworks-worker/Cargo.toml",
  "crates/sceneworks-memory-adapter/Cargo.toml",
];

// Derive the live inference revision from the checked-out manifests instead of restating it as a
// harness constant. A second copy of the pin is a copy no `npm run bump:inference` touches, so it
// goes stale silently and reds this controller for a reason that has nothing to do with the epic.
export async function liveInferencePin(root = "") {
  const perManifest = new Map();
  for (const manifest of INFERENCE_PIN_MANIFESTS) {
    const found = inferencePins(await readFile(path.join(root, manifest), "utf8"));
    if (found.length !== 1) {
      fail(`${manifest} must declare exactly one SceneWorks/inference revision; got ${found.join(",") || "none"}`);
    }
    perManifest.set(manifest, found[0]);
  }
  const distinct = new Set(perManifest.values());
  if (distinct.size !== 1) {
    fail(`SceneWorks manifests disagree on the inference revision: ${
      [...perManifest].map(([manifest, pin]) => `${manifest}=${pin}`).join(", ")}`);
  }
  return [...distinct][0];
}

function assertOutsideRepository(candidate, repositories, label) {
  const target = path.resolve(candidate);
  for (const repository of repositories) {
    const relative = path.relative(repository, target);
    if (!relative || (!relative.startsWith("..") && !path.isAbsolute(relative))) {
      fail(`${label} must be outside repository ${repository}`);
    }
  }
  return target;
}

function isWithin(parent, candidate, { allowEqual = false } = {}) {
  const relative = path.relative(parent, candidate);
  return (allowEqual && !relative) || Boolean(relative && !relative.startsWith("..") && !path.isAbsolute(relative));
}

function comparable(candidate) {
  const normalized = path.resolve(candidate).replace(/[\\/]+$/, "");
  return process.platform === "win32" ? normalized.toLowerCase() : normalized;
}

async function nearestExistingParent(candidate) {
  let current = path.dirname(path.resolve(candidate));
  for (;;) {
    try {
      return { lexical: current, resolved: await realpath(current) };
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      const parent = path.dirname(current);
      if (parent === current) throw error;
      current = parent;
    }
  }
}

async function assertNewConfinedDirectory(candidate, runnerTemp, repositories, label) {
  const target = assertOutsideRepository(candidate, repositories, label);
  if (!isWithin(runnerTemp, target)) fail(`${label} must be a descendant of the resolved RUNNER_TEMP`);
  try {
    await lstat(target);
    fail(`${label} must not already exist`);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  const parent = await nearestExistingParent(target);
  if (!isWithin(runnerTemp, parent.resolved, { allowEqual: true })) {
    fail(`${label} traverses a symlink or reparse point outside RUNNER_TEMP`);
  }
  const throughResolvedParent = path.resolve(parent.resolved, path.relative(parent.lexical, target));
  if (!isWithin(runnerTemp, throughResolvedParent)) {
    fail(`${label} resolves outside RUNNER_TEMP`);
  }
  return target;
}

export async function validateTrustedCacheRoot(candidate, repositories, runnerTemp) {
  if (!candidate || !path.isAbsolute(candidate)) {
    fail("trusted cache root must be an explicit absolute path");
  }
  const target = assertOutsideRepository(candidate, repositories, "trusted cache root");
  const metadata = await lstat(target);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    fail("trusted cache root must be an existing ordinary directory, not a symlink/reparse point");
  }
  const resolved = await realpath(target);
  if (comparable(resolved) !== comparable(target)) {
    fail("trusted cache root traverses a symlink or reparse point");
  }
  if (isWithin(resolved, runnerTemp, { allowEqual: true })
    || isWithin(runnerTemp, resolved, { allowEqual: true })) {
    fail("trusted cache root and RUNNER_TEMP must be separate trees");
  }
  return resolved;
}

export async function validateCampaignPaths({
  runnerTemp, output, scratch, cacheRoot, repositories,
}) {
  if (!runnerTemp) fail("RUNNER_TEMP is required for terminal evidence confinement");
  const runnerMetadata = await stat(runnerTemp);
  if (!runnerMetadata.isDirectory()) fail("RUNNER_TEMP must be an existing directory");
  const resolvedRunnerTemp = await realpath(runnerTemp);
  const resolvedRepositories = await Promise.all(repositories.map((repository) => realpath(repository)));
  const resolvedCacheRoot = await validateTrustedCacheRoot(
    cacheRoot, resolvedRepositories, resolvedRunnerTemp,
  );
  const resolvedOutput = await assertNewConfinedDirectory(
    output, resolvedRunnerTemp, resolvedRepositories, "output",
  );
  const resolvedScratch = await assertNewConfinedDirectory(
    scratch, resolvedRunnerTemp, resolvedRepositories, "scratch",
  );
  if (comparable(resolvedOutput) === comparable(resolvedScratch)
    || isWithin(resolvedOutput, resolvedScratch) || isWithin(resolvedScratch, resolvedOutput)) {
    fail("output and scratch must be distinct, non-nested RUNNER_TEMP descendants");
  }
  return {
    runnerTemp: resolvedRunnerTemp,
    output: resolvedOutput,
    scratch: resolvedScratch,
    cacheRoot: resolvedCacheRoot,
    repositories: resolvedRepositories,
  };
}

async function assertExistingConfinedTree(candidate, guard, label) {
  const target = assertOutsideRepository(candidate, guard.repositories, label);
  if (!isWithin(guard.runnerTemp, target)) fail(`${label} escaped RUNNER_TEMP before cleanup`);
  const metadata = await lstat(target);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    fail(`${label} is not a controller-owned ordinary directory before cleanup`);
  }
  const resolved = await realpath(target);
  if (comparable(resolved) !== comparable(target)) {
    fail(`${label} was replaced by a symlink or reparse point before cleanup`);
  }
  const parent = await realpath(path.dirname(target));
  if (!isWithin(guard.runnerTemp, parent, { allowEqual: true })) {
    fail(`${label} parent escaped RUNNER_TEMP before cleanup`);
  }
  return target;
}

export async function safeRemoveTree(candidate, guard, label = "cleanup target") {
  const target = await assertExistingConfinedTree(candidate, guard, label);
  // This confinement/reparse check intentionally sits immediately beside the only recursive remove.
  await rm(target, { recursive: true, force: false });
  await assertPathAbsent(target, label);
}

// A downloaded authority can release its own persistent store at its lifetime boundary. The final
// store sweep therefore treats only an already-absent store as success; every other cleanup error
// remains a campaign failure and final lifecycle inventory still proves the absence.
export async function cleanupMissingFileStore(operations, missingStore, guard, label) {
  try {
    await operations.cleanup(missingStore, guard, label);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  await assertPathAbsent(missingStore, label);
}

async function assertPathAbsent(target, label) {
  try {
    await lstat(target);
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  fail(`${label} still exists after recursive cleanup`);
}

async function sha256File(file) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

export async function hashedFiles(root, { exclude = new Set() } = {}) {
  const absolute = path.resolve(root);
  const rootMetadata = await lstat(absolute);
  if (!rootMetadata.isDirectory() || rootMetadata.isSymbolicLink()
    || comparable(await realpath(absolute)) !== comparable(absolute)) {
    fail(`evidence root is not an ordinary non-reparse directory: ${absolute}`);
  }
  const files = [];
  async function visit(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
      const file = path.join(directory, entry.name);
      const relative = path.relative(absolute, file).split(path.sep).join("/");
      const metadata = await lstat(file);
      if (entry.isSymbolicLink() || metadata.isSymbolicLink()) {
        fail(`evidence tree contains a symlink or reparse point: ${file}`);
      }
      if (exclude.has(relative)) {
        if (!entry.isFile() || !metadata.isFile()) {
          fail(`excluded evidence entry is not an ordinary regular file: ${file}`);
        }
        continue;
      }
      if (entry.isDirectory() && metadata.isDirectory()) await visit(file);
      else if (entry.isFile() && metadata.isFile()) {
        files.push({ path: relative, bytes: metadata.size, sha256: await sha256File(file) });
      } else fail(`evidence tree contains a non-regular entry: ${file}`);
    }
  }
  await visit(absolute);
  return files;
}

export async function directoryInventory(root) {
  const files = await hashedFiles(root);
  const hash = createHash("sha256");
  for (const file of files) {
    hash.update(file.path);
    hash.update("\0");
    hash.update(String(file.bytes));
    hash.update("\0");
    hash.update(file.sha256);
    hash.update("\n");
  }
  return {
    root: path.resolve(root),
    files: files.length,
    bytes: files.reduce((total, file) => total + file.bytes, 0),
    sha256: hash.digest("hex"),
  };
}

export function parseNvidiaSmi(raw) {
  return raw.trim().split(/\r?\n/).filter(Boolean).map((line) => {
    const [timestamp, index, name, uuid, pciBusId, computeCapability, driverVersion,
      memoryTotalMiB, memoryUsedMiB, memoryFreeMiB] = line
      .split(",").map((value) => value.trim());
    return {
      timestamp, index: Number(index), name, uuid, pciBusId, computeCapability, driverVersion,
      memoryTotalMiB: Number(memoryTotalMiB),
      memoryUsedMiB: Number(memoryUsedMiB),
      memoryFreeMiB: Number(memoryFreeMiB),
      raw: line,
    };
  });
}

function sampleGpu() {
  const raw = run("nvidia-smi", [
    "--query-gpu=timestamp,index,name,uuid,pci.bus_id,compute_cap,driver_version,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
  ]);
  return parseNvidiaSmi(raw);
}

function workflowExecution(expectedHead) {
  const required = ["GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_WORKFLOW", "RUNNER_NAME"];
  for (const name of required) if (!process.env[name]) fail(`terminal dispatch requires ${name}`);
  if (process.env.GITHUB_SHA !== expectedHead) fail("GITHUB_SHA does not match the clean SceneWorks HEAD");
  return {
    runId: process.env.GITHUB_RUN_ID,
    runAttempt: process.env.GITHUB_RUN_ATTEMPT,
    headSha: process.env.GITHUB_SHA,
    headRef: process.env.GITHUB_REF ?? "",
    workflow: process.env.GITHUB_WORKFLOW,
    runnerName: process.env.RUNNER_NAME,
    runnerOs: process.env.RUNNER_OS ?? "Windows",
    runnerArch: process.env.RUNNER_ARCH ?? "X64",
  };
}

export function receiptSkeleton({
  cell, ordinal, repositories, artifacts, execution, gpuIdentity, systemMemory, startedAt,
}) {
  return {
    schemaVersion: 1,
    profile: PROFILE_NAME,
    cell: {
      id: cell.id, ordinal, modelId: cell.modelId, engineId: cell.engineId, kind: cell.kind,
      requestedTier: cell.requestedTier, resolvedTier: cell.requestedTier, denseFallback: false,
      loadSpecQuantBits: expectedLoadSpecQuantBits(cell),
    },
    status: "failed",
    error: "cell has not completed",
    repositories,
    artifacts,
    execution,
    hardware: {
      gpuIdentity,
      systemMemory,
      cudaComputeCap: process.env.CUDA_COMPUTE_CAP ?? "",
      cudaVisibleDevices: process.env.CUDA_VISIBLE_DEVICES ?? "",
      rawVramSamples: [],
    },
    cleanup: { attempted: false, completed: false, error: null },
    inputs: [], outputs: [], logs: [],
    startedAt,
    completedAt: startedAt,
  };
}

function validateReceiptInternal(
  receipt, expectedCell, profile, lifecycleContext = null, trustedContext = null,
) {
  if (receipt.schemaVersion !== 1 || receipt.profile !== PROFILE_NAME) fail("receipt identity mismatch");
  if (!new Set(["passed", "failed"]).has(receipt.status)) fail("receipt status is invalid");
  if (receipt.cell.ordinal < 1 || receipt.cell.ordinal > EXPECTED_CELLS.length
    || receipt.cell.id !== EXPECTED_CELLS[receipt.cell.ordinal - 1]) {
    fail("receipt cell identity or ordinal drifted from the serialized profile");
  }
  if (!expectedCell || !profile || receipt.cell.id !== expectedCell.id
    || receipt.cell.modelId !== expectedCell.modelId || receipt.cell.engineId !== expectedCell.engineId
    || receipt.cell.kind !== expectedCell.kind || receipt.cell.requestedTier !== expectedCell.requestedTier) {
    fail("receipt cell semantics drifted from the frozen profile");
  }
  if (receipt.cell.requestedTier !== receipt.cell.resolvedTier || receipt.cell.denseFallback !== false) {
    fail(`${receipt.cell.id} receipt permits a dense or cross-tier fallback`);
  }
  const expectedQuant = expectedLoadSpecQuantBits(expectedCell);
  if (!Object.hasOwn(receipt.cell, "loadSpecQuantBits")
    || receipt.cell.loadSpecQuantBits !== expectedQuant) {
    fail(`${receipt.cell.id} receipt does not bind family loadSpecQuantBits=${expectedQuant}`);
  }
  const isMultiReference = expectedCell.kind === "scail2"
    && expectedCell.capability === "multiReference";
  if (receipt.status === "passed" && isMultiReference) {
    validateScail2ReferenceCounterfactuals({
      kind: "scail2",
      referencePairs: 6,
      referenceCounterfactuals: receipt.cell.referenceCounterfactuals,
    }, expectedCell, "receipt");
  } else if (Object.hasOwn(receipt.cell, "referenceCounterfactuals")) {
    fail(`${receipt.cell.id} receipt carries counterfactual evidence outside the passed multi-reference cell`);
  }
  if (receipt.status === "passed" && expectedCell.kind === "sdxlOpenPose") {
    validateOpenPoseControlCounterfactual({
      kind: "sdxlOpenPose",
      controlCounterfactual: receipt.cell.controlCounterfactual,
    }, expectedCell, "receipt");
  } else if (Object.hasOwn(receipt.cell, "controlCounterfactual")) {
    fail(`${receipt.cell.id} receipt carries OpenPose control evidence outside a passed OpenPose cell`);
  }
  if (!receipt.repositories?.sceneworks?.clean || !receipt.repositories?.inference?.clean) {
    fail("receipt does not bind clean paired repositories");
  }
  if (!SHA40.test(receipt.repositories.sceneworks.sha) || !SHA40.test(receipt.repositories.inference.sha)
    || receipt.execution?.headSha !== receipt.repositories.sceneworks.sha
    || !receipt.execution.runId || !receipt.execution.runAttempt || !receipt.execution.runnerName) {
    fail("receipt source or workflow execution identity is incomplete");
  }
  if (!Array.isArray(receipt.artifacts)
    || receipt.artifacts.length !== expectedCell.artifactIds.length
    || receipt.artifacts.some((artifact, index) => {
      const expectedId = expectedCell.artifactIds[index];
      const expected = profile.artifacts[expectedId];
      return artifact.id !== expectedId || artifact.role !== expected.role
        || artifact.repository !== expected.repository || artifact.revision !== expected.revision
        || artifact.subdirectory !== expected.subdirectory
        || JSON.stringify(artifact.allowPatterns) !== JSON.stringify(expected.allowPatterns)
        || !artifact.inventory || typeof artifact.inventory.root !== "string";
    })) {
    fail("receipt artifact authority binding is incomplete");
  }
  if (receipt.status === "passed" && receipt.artifacts.some((artifact) => (
    artifact.inventory.complete !== true || !/^[0-9a-f]{64}$/.test(artifact.inventory.sha256)
      || artifact.inventory.files < 1 || artifact.inventory.bytes < 1
      || typeof artifact.selectedRoot !== "string" || !artifact.selectedRoot
      || comparable(artifact.inventory.root) !== comparable(artifact.selectedRoot)
      || artifact.inventory.error !== null
  ))) {
    fail("passed receipt artifact inventory is incomplete");
  }
  if (receipt.artifacts.some((artifact) => artifact.inventory.complete === false
    && typeof artifact.inventory.error !== "string")) {
    fail("failed artifact inventory must explain why it is incomplete");
  }
  if (!Array.isArray(receipt.hardware?.gpuIdentity)
    || (receipt.status === "passed" && receipt.hardware.gpuIdentity.length === 0)
    || receipt.hardware.gpuIdentity.some((gpu) => !gpu.name || !gpu.uuid || !gpu.pciBusId
      || !gpu.computeCapability || !gpu.driverVersion || !Number.isInteger(gpu.index)
      || !Number.isFinite(gpu.memoryTotalMiB) || !gpu.raw)
    || !Number.isSafeInteger(receipt.hardware?.systemMemory?.totalBytes)
    || !Number.isSafeInteger(receipt.hardware?.systemMemory?.availableBytesAtStart)
    || !Array.isArray(receipt.hardware?.rawVramSamples)
    || (receipt.status === "passed" && receipt.hardware.rawVramSamples.length === 0)
    || receipt.hardware.rawVramSamples.some((sample) => typeof sample.raw !== "string")) {
    fail("receipt GPU identity or raw VRAM samples are incomplete");
  }
  const hashedFile = (file) => typeof file?.path === "string" && Number.isInteger(file.bytes)
    && /^[0-9a-f]{64}$/.test(file.sha256);
  if (!Array.isArray(receipt.inputs) || receipt.inputs.some((file) => !hashedFile(file))
    || (receipt.status === "passed" && receipt.inputs.length === 0)
    || !Array.isArray(receipt.outputs) || receipt.outputs.some((file) => !hashedFile(file))
    || (receipt.status === "passed" && receipt.outputs.length === 0)
    || !Array.isArray(receipt.logs) || !receipt.logs.length
    || receipt.logs.some((file) => !hashedFile(file))) {
    fail("receipt input, output, or log hashes are incomplete");
  }
  if (!receipt.cleanup || typeof receipt.cleanup.attempted !== "boolean"
    || typeof receipt.cleanup.completed !== "boolean"
    || (receipt.cleanup.error !== null && typeof receipt.cleanup.error !== "string")) {
    fail("receipt cleanup evidence is incomplete");
  }
  if (!receipt.authorityLifecycle && trustedContext !== TRUSTED_LEGACY_IMPORT_CONTEXT) {
    fail("continuation receipt is missing required authority lifecycle evidence");
  }
  if (receipt.authorityLifecycle) {
    const lifecycle = receipt.authorityLifecycle;
    if (lifecycle.ordinal !== receipt.cell.ordinal || lifecycle.cellId !== receipt.cell.id
      || JSON.stringify(lifecycle.activeArtifactIds) !== JSON.stringify(expectedCell.artifactIds)
      || !new Set(["not-attempted", "failed", "completed"]).has(lifecycle.providerExecution)
      || lifecycle.staged.some((entry) => !expectedCell.artifactIds.includes(entry.artifactId))
      || lifecycle.released.some((entry) => !expectedCell.artifactIds.includes(entry.artifactId))) {
      fail("receipt authority lifecycle identity drifted");
    }
    if ((lifecycle.providerExecution === "not-attempted" && lifecycle.derivedAfter.length)
      || (lifecycle.providerExecution === "failed" && receipt.status !== "failed")) {
      fail("receipt provider execution state drifted from its outcome evidence");
    }
    let expectedSuccessDisposition = null;
    if (lifecycle.providerExecution === "completed") {
      validateRequestMemoryStrategy(lifecycle.requestMemoryStrategy, expectedCell);
      expectedSuccessDisposition = dispositionForMemoryStrategy(lifecycle.requestMemoryStrategy);
    } else if (lifecycle.requestMemoryStrategy !== null) {
      fail("receipt without completed provider execution cannot claim a request memory strategy");
    }
    for (const derived of lifecycle.derivedAfter) {
      const isFlux = derived.artifactId.startsWith("flux1-");
      const payloadFloor = derived.artifactId.endsWith("-q4")
        ? 7_396_392_960 : 12_573_868_032;
      const payloadCeiling = derived.artifactId.endsWith("-q4")
        ? FLUX_Q4_SIDECAR_BYTES : FLUX_Q8_SIDECAR_BYTES;
      if (isFlux) {
        const staged = lifecycle.staged.find((entry) => entry.artifactId === derived.artifactId);
        const oneBoundNamespace = derived.inventories.length === 1
          && staged?.derivedNamespaces.length === 1
          && comparable(derived.inventories[0].root) === comparable(staged.derivedNamespaces[0])
          && /^[0-9a-f]{64}$/.test(path.basename(derived.inventories[0].root))
          && path.basename(path.dirname(derived.inventories[0].root)) === "candle-device-format-v1";
        const exactEmptyFailure = lifecycle.providerExecution === "failed"
          && derived.files === 0 && derived.bytes === 0 && oneBoundNamespace
          && derived.inventories[0].files === 0 && derived.inventories[0].bytes === 0
          && derived.inventories[0].sha256 === createHash("sha256").digest("hex");
        const exactResident = lifecycle.providerExecution === "completed"
          && expectedSuccessDisposition === DERIVED_DISPOSITION_RESIDENT
          && derived.files === 0 && derived.bytes === 0 && derived.inventories.length === 0
          && staged?.derivedNamespaces.length === 0;
        const exactComplete = lifecycle.providerExecution === "completed"
          && expectedSuccessDisposition === DERIVED_DISPOSITION_BOUNDED
          && derived.files === 494 && derived.bytes >= payloadFloor
          && derived.bytes <= payloadCeiling && oneBoundNamespace;
        const expectedDisposition = exactResident ? DERIVED_DISPOSITION_RESIDENT
          : exactEmptyFailure ? DERIVED_DISPOSITION_PROVIDER_FAILED
            : exactComplete ? DERIVED_DISPOSITION_BOUNDED : null;
        if (!expectedDisposition || derived.derivedDisposition !== expectedDisposition) {
          fail("receipt FLUX derived lifecycle is neither exact resident-empty nor one exact bounded b646 namespace");
        }
      } else if (derived.derivedDisposition !== DERIVED_DISPOSITION_NOT_APPLICABLE
        || derived.files !== 0 || derived.bytes !== 0 || derived.inventories.length !== 0) {
        fail("receipt non-FLUX derived lifecycle is not exactly empty");
      }
    }
    if (lifecycleContext) {
      const { lifetimeById, scratchRoot, requiredFreeBytes, completeLifecycle = false } = lifecycleContext;
      const startingIds = expectedCell.artifactIds.filter(
        (id) => lifetimeById.get(id)?.firstOrdinal === receipt.cell.ordinal,
      );
      const endingIds = expectedCell.artifactIds.filter(
        (id) => lifetimeById.get(id)?.lastOrdinal === receipt.cell.ordinal,
      );
      const expectedPhases = [
        ...startingIds.map((id) => `before-stage:${id}`),
        "before-execution",
      ];
      const actualPhases = lifecycle.diskProbes.map((probe) => probe.phase);
      if (JSON.stringify(actualPhases)
        !== JSON.stringify(expectedPhases.slice(0, actualPhases.length))
        || lifecycle.diskProbes.some((probe) => (
          probe.ordinal !== receipt.cell.ordinal
          || comparable(probe.root) !== comparable(scratchRoot)
          || probe.requiredFreeBytes !== requiredFreeBytes
          || !Number.isSafeInteger(probe.freeBytes)
          || probe.freeBytes < 0
        ))) {
        fail("receipt disk probes drifted from the exact authority transition schedule");
      }
      const stagedIds = lifecycle.staged.map((entry) => entry.artifactId);
      const releasedIds = lifecycle.released.map((entry) => entry.artifactId);
      if (JSON.stringify(stagedIds) !== JSON.stringify(startingIds.slice(0, stagedIds.length))
        || releasedIds.some((id) => !new Set([...startingIds, ...endingIds]).has(id))) {
        fail("receipt staged/released authorities drifted from the lifetime plan");
      }
      const expectedDerivedIds = lifecycle.providerExecution === "not-attempted"
        ? [] : expectedCell.artifactIds;
      if (completeLifecycle && (JSON.stringify(actualPhases) !== JSON.stringify(expectedPhases)
        || lifecycle.diskProbes.some((probe) => probe.freeBytes < requiredFreeBytes)
        || JSON.stringify(stagedIds) !== JSON.stringify(startingIds)
        || JSON.stringify(releasedIds) !== JSON.stringify(endingIds)
        || JSON.stringify(lifecycle.derivedAfter.map((entry) => entry.artifactId))
          !== JSON.stringify(expectedDerivedIds))) {
        fail("completed receipt lifecycle omitted an exact transition, probe, or derivation");
      }
    }
    if (receipt.status === "passed" && (!lifecycle.verifiedBefore || !lifecycle.verifiedAfter
      || lifecycle.providerExecution !== "completed"
      || lifecycle.diskProbes.length < lifecycle.staged.length + 1
      || lifecycle.released.some((entry) => !entry.stageRemoved || !entry.derivedRemoved))) {
      fail("passed receipt authority lifecycle is incomplete");
    }
  }
  if (receipt.status === "passed" && (receipt.error !== null || !receipt.cleanup.completed
    || receipt.cleanup.error !== null)) {
    fail("passed receipt contains a failure or incomplete cleanup");
  }
  if (receipt.status === "failed" && (typeof receipt.error !== "string" || !receipt.error)) {
    fail("failed receipt must retain an error");
  }
  return receipt;
}

export function validateReceipt(receipt, expectedCell, profile, lifecycleContext = null) {
  return validateReceiptInternal(receipt, expectedCell, profile, lifecycleContext);
}

function cachedArtifactRoots(cacheRoot, artifact) {
  const repository = `models--${artifact.repository.replace("/", "--")}`;
  const snapshotRoot = path.join(cacheRoot, repository, "snapshots", artifact.revision);
  return {
    snapshotRoot,
    selectedRoot: path.resolve(snapshotRoot, artifact.subdirectory),
  };
}

function pendingArtifact(id, artifact, cacheRoot) {
  const inventoryRoot = cachedArtifactRoots(cacheRoot, artifact).selectedRoot;
  return {
    id,
    role: artifact.role,
    repository: artifact.repository,
    revision: artifact.revision,
    subdirectory: artifact.subdirectory,
    selectedRoot: null,
    allowPatterns: artifact.allowPatterns,
    inventory: {
      root: inventoryRoot, complete: false, sha256: null, files: 0, bytes: 0,
      error: "provisioning has not completed",
    },
  };
}

async function evidenceFiles(cellDir) {
  const files = await hashedFiles(cellDir, { exclude: new Set(["receipt.json"]) });
  return {
    inputs: files.filter((file) => file.path === "cell.json" || file.path === "generated-inputs.json"
      || file.path.startsWith("input-")),
    outputs: files.filter((file) => !file.path.endsWith(".log") && file.path !== "cell.json"
      && file.path !== "generated-inputs.json" && !file.path.startsWith("input-")),
    logs: files.filter((file) => file.path.endsWith(".log")),
  };
}

function legacyPrefixProfile(currentProfile) {
  const legacy = structuredClone(currentProfile);
  legacy.cells[SPARSE_EXECUTION_ORDINALS[0] - 1].request.steps = 4;
  legacy.artifacts["ltx23-q8"].revision = LTX_CURRENT_REVISION;
  legacy.artifacts["ltx23-gemma"].revision = LTX_CURRENT_REVISION;
  legacy.artifacts["illustrious-v1-q4"].revision = ILLUSTRIOUS_V1_LEGACY_REVISION;
  legacy.artifacts["illustrious-v2-q4"].revision = ILLUSTRIOUS_V2_LEGACY_REVISION;
  delete legacy.artifacts["illustrious-v1-q4"].quantizationMarker;
  delete legacy.artifacts["illustrious-v2-q4"].quantizationMarker;
  if (cellSemanticsSha256(legacy.cells) !== LEGACY_CELL_SEMANTICS_SHA256
    || canonicalSha256(legacy.artifacts) !== LEGACY_ARTIFACT_SEMANTICS_SHA256) {
    fail("legacy prefix compatibility profile drifted");
  }
  return legacy;
}

function legacyRecoveryProfile(currentProfile) {
  const legacy = structuredClone(currentProfile);
  legacy.cells[SPARSE_EXECUTION_ORDINALS[0] - 1].request.steps = 4;
  legacy.artifacts["illustrious-v1-q4"].revision = ILLUSTRIOUS_V1_LEGACY_REVISION;
  legacy.artifacts["illustrious-v2-q4"].revision = ILLUSTRIOUS_V2_LEGACY_REVISION;
  delete legacy.artifacts["illustrious-v1-q4"].quantizationMarker;
  delete legacy.artifacts["illustrious-v2-q4"].quantizationMarker;
  if (cellSemanticsSha256(legacy.cells) !== LEGACY_CELL_SEMANTICS_SHA256
    || canonicalSha256(legacy.artifacts) !== LEGACY_RECOVERY_ARTIFACT_SEMANTICS_SHA256) {
    fail("legacy recovery compatibility profile drifted");
  }
  return legacy;
}

async function ordinaryTreeFiles(root) {
  const absolute = path.resolve(root);
  const metadata = await lstat(absolute);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()
    || comparable(await realpath(absolute)) !== comparable(absolute)) {
    fail(`imported prefix root must be an ordinary confined directory: ${absolute}`);
  }
  const files = [];
  async function visit(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const candidate = path.join(directory, entry.name);
      const entryMetadata = await lstat(candidate);
      if (entry.isSymbolicLink() || entryMetadata.isSymbolicLink()) {
        fail(`imported prefix contains a symlink/reparse point: ${candidate}`);
      }
      const resolved = await realpath(candidate);
      if (!isWithin(absolute, resolved)) {
        fail(`imported prefix entry escaped its candidate root: ${candidate}`);
      }
      if (entry.isDirectory()) await visit(resolved);
      else if (entry.isFile()) files.push(resolved);
      else fail(`imported prefix contains a non-regular entry: ${candidate}`);
    }
  }
  await visit(absolute);
  return files;
}

async function validatePrefixCandidate(candidate, currentProfile) {
  const metadataPath = path.join(candidate, "artifact-metadata.json");
  const evidenceRoot = path.join(candidate, "evidence");
  const metadata = JSON.parse(await readFile(metadataPath, "utf8"));
  exactKeys(metadata, [
    "artifactId", "artifactName", "artifactDigest", "runId", "runAttempt", "headSha", "inferenceSha",
    "profile", "cellSemanticsSha256", "artifactSemanticsSha256",
  ], "imported prefix artifact metadata");
  const runIdentity = /^[1-9][0-9]*$/;
  if (!runIdentity.test(metadata.artifactId) || !runIdentity.test(metadata.runId)
    || !runIdentity.test(metadata.runAttempt)
    || metadata.artifactName !== `${PREFIX_ARTIFACT}${metadata.runId}-${metadata.runAttempt}`
    || !/^sha256:[0-9a-f]{64}$/.test(metadata.artifactDigest)
    || metadata.headSha !== LEGACY_SCENEWORKS_HEAD
    || metadata.inferenceSha !== HISTORICAL_INFERENCE_PIN
    || metadata.profile !== PROFILE_NAME
    || metadata.cellSemanticsSha256 !== LEGACY_CELL_SEMANTICS_SHA256
    || metadata.artifactSemanticsSha256 !== LEGACY_ARTIFACT_SEMANTICS_SHA256) {
    fail("imported prefix artifact metadata does not bind the reviewed old run/profile");
  }
  const files = await ordinaryTreeFiles(evidenceRoot);
  const receiptFiles = files.filter((file) => path.basename(file) === "receipt.json");
  if (receiptFiles.length !== IMPORTED_PREFIX_CELLS + 1) {
    fail(`imported artifact must contain exactly ${IMPORTED_PREFIX_CELLS} PASS receipts plus one boundary skeleton`);
  }
  const legacyProfile = legacyPrefixProfile(currentProfile);
  const receipts = [];
  for (let index = 0; index < IMPORTED_PREFIX_CELLS; index += 1) {
    const cell = legacyProfile.cells[index];
    const ordinalName = `${String(index + 1).padStart(2, "0")}-${cell.id}`;
    const cellDir = path.join(evidenceRoot, ordinalName);
    const receiptPath = path.join(cellDir, "receipt.json");
    if (!receiptFiles.some((file) => comparable(file) === comparable(receiptPath))) {
      fail(`imported prefix is missing the exact primary receipt for ${ordinalName}`);
    }
    const receipt = JSON.parse(await readFile(receiptPath, "utf8"));
    if (receipt.status !== "passed"
      || receipt.cell?.ordinal !== index + 1
      || receipt.cell?.id !== cell.id
      || receipt.repositories?.sceneworks?.sha !== LEGACY_SCENEWORKS_HEAD
      || receipt.repositories?.inference?.sha !== HISTORICAL_INFERENCE_PIN
      || receipt.execution?.headSha !== LEGACY_SCENEWORKS_HEAD
      || receipt.execution?.runId !== metadata.runId
      || receipt.execution?.runAttempt !== metadata.runAttempt) {
      fail(`imported prefix receipt ${ordinalName} is not an exact PASS from the bound old run`);
    }
    validateTrustedLegacyImportedReceiptDocument(receipt);
    validateReceiptInternal(receipt, cell, legacyProfile, null, TRUSTED_LEGACY_IMPORT_CONTEXT);
    const rehashed = await evidenceFiles(cellDir);
    for (const field of ["inputs", "outputs", "logs"]) {
      if (JSON.stringify(rehashed[field]) !== JSON.stringify(receipt[field])) {
        fail(`imported prefix receipt ${ordinalName} failed ${field} rehash`);
      }
    }
    receipts.push({ cell, ordinalName, receipt });
  }

  // The timeout raced with cell 8 setup: preserve the two-file initial receipt as quarantined
  // boundary residue, but never promote it into either PASS or failure evidence. Any model input,
  // runtime result, output, later cell, cleanup attempt, or nonzero model inventory means work had
  // progressed past the reviewed continuation boundary and invalidates the whole candidate.
  const boundaryIndex = IMPORTED_PREFIX_CELLS;
  const boundaryCell = legacyProfile.cells[boundaryIndex];
  const boundaryOrdinalName = `${String(boundaryIndex + 1).padStart(2, "0")}-${boundaryCell.id}`;
  const boundaryDir = path.join(evidenceRoot, boundaryOrdinalName);
  const boundaryFiles = files.filter((file) => isWithin(boundaryDir, file, { allowEqual: true }));
  const expectedBoundaryFiles = [
    path.join(boundaryDir, "controller.log"),
    path.join(boundaryDir, "receipt.json"),
  ].map(comparable).sort();
  if (JSON.stringify(boundaryFiles.map(comparable).sort()) !== JSON.stringify(expectedBoundaryFiles)) {
    fail("imported boundary residue must contain only the initial controller log and receipt");
  }
  const allowedRoots = [
    ...receipts.map(({ ordinalName }) => path.join(evidenceRoot, ordinalName)),
    boundaryDir,
  ];
  if (files.some((file) => !allowedRoots.some((root) => isWithin(root, file, { allowEqual: true })))) {
    fail("imported artifact contains evidence outside the seven PASS cells and cell-8 boundary residue");
  }
  const boundaryReceipt = JSON.parse(await readFile(path.join(boundaryDir, "receipt.json"), "utf8"));
  if (boundaryReceipt.status !== "failed"
    || boundaryReceipt.cell?.ordinal !== boundaryIndex + 1
    || boundaryReceipt.cell?.id !== boundaryCell.id
    || boundaryReceipt.repositories?.sceneworks?.sha !== LEGACY_SCENEWORKS_HEAD
    || boundaryReceipt.repositories?.inference?.sha !== HISTORICAL_INFERENCE_PIN
    || boundaryReceipt.execution?.headSha !== LEGACY_SCENEWORKS_HEAD
    || boundaryReceipt.execution?.runId !== metadata.runId
    || boundaryReceipt.execution?.runAttempt !== metadata.runAttempt) {
    fail("imported cell-8 residue is not bound to the exact legacy run boundary");
  }
  validateTrustedLegacyImportedReceiptDocument(boundaryReceipt);
  validateReceiptInternal(
    boundaryReceipt, boundaryCell, legacyProfile, null, TRUSTED_LEGACY_IMPORT_CONTEXT,
  );
  const boundaryLog = (await hashedFiles(boundaryDir, { exclude: new Set(["receipt.json"]) }));
  if (boundaryReceipt.error !== "cell has not completed"
    || boundaryReceipt.startedAt !== boundaryReceipt.completedAt
    || boundaryReceipt.cleanup.attempted !== false
    || boundaryReceipt.cleanup.completed !== false
    || boundaryReceipt.cleanup.error !== null
    || boundaryReceipt.inputs.length !== 0 || boundaryReceipt.outputs.length !== 0
    || JSON.stringify(boundaryReceipt.logs) !== JSON.stringify(boundaryLog)
    || boundaryLog.length !== 1 || boundaryLog[0].path !== "controller.log"
    || boundaryReceipt.artifacts.some((artifact) => artifact.selectedRoot !== null
      || artifact.inventory.complete !== false || artifact.inventory.sha256 !== null
      || artifact.inventory.files !== 0 || artifact.inventory.bytes !== 0
      || artifact.inventory.error !== "provisioning has not completed")) {
    fail("imported cell-8 residue is not the exact non-executed pre-provision boundary skeleton");
  }
  return {
    candidate, evidenceRoot, metadata, receipts,
    boundaryResidue: {
      cell: boundaryCell,
      ordinalName: boundaryOrdinalName,
      files: boundaryLog,
      receipt: boundaryReceipt,
    },
  };
}

export async function selectImportedPrefix(prefixCandidates, currentProfile) {
  const root = path.resolve(prefixCandidates);
  const entries = await readdir(root, { withFileTypes: true });
  const valid = [];
  const rejected = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    if (!entry.isDirectory() || entry.isSymbolicLink()) {
      rejected.push(`${entry.name}: candidate is not an ordinary directory`);
      continue;
    }
    try {
      valid.push(await validatePrefixCandidate(path.join(root, entry.name), currentProfile));
    } catch (error) {
      rejected.push(`${entry.name}: ${errorText(error)}`);
    }
  }
  if (valid.length !== 1) {
    fail(`expected exactly one valid uploaded contiguous PASS prefix; found ${valid.length}; ${rejected.join(" | ")}`);
  }
  return valid[0];
}

export async function importPrefixEvidence(prefix, output) {
  const destination = path.join(output, "_imported-prefix");
  await mkdir(destination);
  for (const { ordinalName } of prefix.receipts) {
    await cp(path.join(prefix.evidenceRoot, ordinalName), path.join(destination, ordinalName), {
      recursive: true, errorOnExist: true, force: false, dereference: false,
    });
  }
  await ordinaryTreeFiles(destination);
  for (const { ordinalName, receipt } of prefix.receipts) {
    const rehashed = await evidenceFiles(path.join(destination, ordinalName));
    for (const field of ["inputs", "outputs", "logs"]) {
      if (JSON.stringify(rehashed[field]) !== JSON.stringify(receipt[field])) {
        fail(`copied imported prefix ${ordinalName} failed ${field} rehash`);
      }
    }
  }
  const boundaryDestination = path.join(output, "_imported-boundary-residue");
  await mkdir(boundaryDestination);
  await cp(
    path.join(prefix.evidenceRoot, prefix.boundaryResidue.ordinalName),
    path.join(boundaryDestination, prefix.boundaryResidue.ordinalName),
    { recursive: true, errorOnExist: true, force: false, dereference: false },
  );
  await ordinaryTreeFiles(boundaryDestination);
  const copiedBoundaryFiles = await hashedFiles(
    path.join(boundaryDestination, prefix.boundaryResidue.ordinalName),
    { exclude: new Set(["receipt.json"]) },
  );
  if (JSON.stringify(copiedBoundaryFiles) !== JSON.stringify(prefix.boundaryResidue.files)) {
    fail("copied imported boundary residue failed exact log rehash");
  }
  const lineage = {
    kind: "contiguous-pass-prefix",
    sourceArtifactId: prefix.metadata.artifactId,
    sourceArtifactName: prefix.metadata.artifactName,
    sourceArtifactDigest: prefix.metadata.artifactDigest,
    sourceRunId: prefix.metadata.runId,
    sourceRunAttempt: prefix.metadata.runAttempt,
    sourceHeadSha: prefix.metadata.headSha,
    sourceInferenceSha: prefix.metadata.inferenceSha,
    sourceProfile: prefix.metadata.profile,
    sourceCellSemanticsSha256: prefix.metadata.cellSemanticsSha256,
    sourceArtifactSemanticsSha256: prefix.metadata.artifactSemanticsSha256,
    importedOrdinals: Array.from({ length: IMPORTED_PREFIX_CELLS }, (_, index) => index + 1),
    quarantinedBoundaryResidue: {
      ordinal: IMPORTED_PREFIX_CELLS + 1,
      cellId: prefix.boundaryResidue.cell.id,
      path: `_imported-boundary-residue/${prefix.boundaryResidue.ordinalName}`,
      disposition: "non-executed-pre-provision-skeleton-excluded-from-prefix",
      files: prefix.boundaryResidue.files,
    },
  };
  await writeFile(
    path.join(destination, "lineage.json"),
    `${JSON.stringify(lineage, null, 2)}\n`,
    { encoding: "utf8", flag: "wx" },
  );
  return {
    lineage,
    outcomes: prefix.receipts.map(({ cell, ordinalName }) => ({
      id: cell.id,
      status: "passed",
      receipt: `_imported-prefix/${ordinalName}/receipt.json`,
      error: null,
      emergencyReceiptError: null,
      source: "imported",
    })),
  };
}

function validateRecoveryReceiptProvenance(
  receipt, cell, { headSha, runId, runAttempt, inferenceSha = HISTORICAL_INFERENCE_PIN }, label,
) {
  if (receipt.cell?.ordinal !== cell.ordinal || receipt.cell?.id !== cell.id
    || receipt.repositories?.sceneworks?.sha !== headSha || receipt.repositories?.sceneworks?.clean !== true
    || receipt.repositories?.inference?.sha !== inferenceSha || receipt.repositories?.inference?.clean !== true
    || receipt.execution?.headSha !== headSha || receipt.execution?.runId !== runId
    || receipt.execution?.runAttempt !== runAttempt) {
    fail(`${label} is not bound to its exact audited provenance`);
  }
}

async function validateArchivedReceipt({ root, ordinalName, cell, profile, metadata }) {
  const cellDir = path.join(root, ordinalName);
  const receipt = JSON.parse(await readFile(path.join(cellDir, "receipt.json"), "utf8"));
  if (receipt.status !== "passed") {
    fail(`recovery receipt ${ordinalName} is not an exact PASS from the bound recovery run`);
  }
  validateRecoveryReceiptProvenance(receipt, cell, metadata, `recovery receipt ${ordinalName}`);
  validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt);
  validateReceipt(receipt, cell, profile);
  const rehashed = await evidenceFiles(cellDir);
  for (const field of ["inputs", "outputs", "logs"]) {
    if (JSON.stringify(rehashed[field]) !== JSON.stringify(receipt[field])) {
      fail(`recovery receipt ${ordinalName} failed ${field} rehash`);
    }
  }
  return { cell, ordinalName, receipt, root: cellDir };
}

function expectedRecoveryOriginalLineage() {
  return {
    kind: "contiguous-pass-prefix",
    sourceArtifactId: RECOVERY_ORIGINAL_ARTIFACT_ID,
    sourceArtifactName: RECOVERY_ORIGINAL_ARTIFACT_NAME,
    sourceArtifactDigest: RECOVERY_ORIGINAL_ARTIFACT_DIGEST,
    sourceRunId: "32570707303",
    sourceRunAttempt: "1",
    sourceHeadSha: LEGACY_SCENEWORKS_HEAD,
    sourceInferenceSha: HISTORICAL_INFERENCE_PIN,
    sourceProfile: PROFILE_NAME,
    sourceCellSemanticsSha256: LEGACY_CELL_SEMANTICS_SHA256,
    sourceArtifactSemanticsSha256: LEGACY_ARTIFACT_SEMANTICS_SHA256,
    importedOrdinals: [1, 2, 3, 4, 5, 6, 7],
    quarantinedBoundaryResidue: {
      ordinal: 8,
      cellId: "flux1-dev-q8",
      path: "_imported-boundary-residue/08-flux1-dev-q8",
      disposition: "non-executed-pre-provision-skeleton-excluded-from-prefix",
      files: [RECOVERY_BOUNDARY_LOG],
    },
  };
}

async function validateRecoveryBoundaryResidue(evidenceRoot, originalLineage, currentProfile) {
  const legacy = legacyPrefixProfile(currentProfile);
  const boundaryCell = { ...legacy.cells[IMPORTED_PREFIX_CELLS], ordinal: IMPORTED_PREFIX_CELLS + 1 };
  const ordinalName = `08-${boundaryCell.id}`;
  const boundaryDir = path.join(evidenceRoot, "_imported-boundary-residue", ordinalName);
  const files = await ordinaryTreeFiles(boundaryDir);
  const expected = [path.join(boundaryDir, "controller.log"), path.join(boundaryDir, "receipt.json")]
    .map(comparable).sort();
  if (JSON.stringify(files.map(comparable).sort()) !== JSON.stringify(expected)) {
    fail("recovery imported boundary residue must contain only controller.log and receipt.json");
  }
  const receipt = JSON.parse(await readFile(path.join(boundaryDir, "receipt.json"), "utf8"));
  validateRecoveryReceiptProvenance(receipt, boundaryCell, {
    headSha: LEGACY_SCENEWORKS_HEAD, runId: "32570707303", runAttempt: "1",
  }, "recovery imported boundary residue");
  validateTrustedLegacyImportedReceiptDocument(receipt);
  validateReceiptInternal(receipt, boundaryCell, legacy, null, TRUSTED_LEGACY_IMPORT_CONTEXT);
  const logs = await hashedFiles(boundaryDir, { exclude: new Set(["receipt.json"]) });
  if (JSON.stringify(logs) !== JSON.stringify([RECOVERY_BOUNDARY_LOG])
    || JSON.stringify(receipt.logs) !== JSON.stringify(logs)
    || JSON.stringify(originalLineage.quarantinedBoundaryResidue.files) !== JSON.stringify(logs)
    || receipt.status !== "failed" || receipt.error !== "cell has not completed"
    || receipt.startedAt !== receipt.completedAt || receipt.cleanup.attempted !== false
    || receipt.cleanup.completed !== false || receipt.cleanup.error !== null
    || receipt.inputs.length !== 0 || receipt.outputs.length !== 0
    || receipt.artifacts.some((artifact) => artifact.selectedRoot !== null
      || artifact.inventory.complete !== false || artifact.inventory.sha256 !== null
      || artifact.inventory.files !== 0 || artifact.inventory.bytes !== 0
      || artifact.inventory.error !== "provisioning has not completed")) {
    fail("recovery imported boundary residue is not the audited non-executed pre-provision skeleton");
  }
}

function validateRecoveryMetadata(metadata) {
  exactKeys(metadata, [
    "artifactId", "artifactName", "artifactSize", "artifactDigest", "runId", "runAttempt", "headSha",
    "inferenceSha", "profile", "cellSemanticsSha256", "artifactSemanticsSha256",
  ], "recovery artifact metadata");
  if (metadata.artifactId !== RECOVERY_ARTIFACT_ID || metadata.artifactName !== RECOVERY_ARTIFACT_NAME
    || metadata.artifactSize !== RECOVERY_ARTIFACT_SIZE || metadata.artifactDigest !== RECOVERY_ARTIFACT_DIGEST
    || metadata.runId !== RECOVERY_RUN_ID || metadata.runAttempt !== RECOVERY_RUN_ATTEMPT
    || metadata.headSha !== RECOVERY_SCENEWORKS_HEAD || metadata.inferenceSha !== HISTORICAL_INFERENCE_PIN
    || metadata.profile !== PROFILE_NAME || metadata.cellSemanticsSha256 !== LEGACY_CELL_SEMANTICS_SHA256
    || metadata.artifactSemanticsSha256 !== LEGACY_RECOVERY_ARTIFACT_SEMANTICS_SHA256) {
    fail("recovery artifact metadata does not bind the audited partial run");
  }
}

async function validateRecoveryCacheEvidence(evidenceRoot, currentProfile) {
  const expectedArtifactIds = [...new Set(currentProfile.cells.slice(IMPORTED_PREFIX_CELLS).flatMap(
    (cell) => cell.artifactIds,
  ))];
  const evidence = JSON.parse(await readFile(DOWNLOAD_EVIDENCE_PATH, "utf8"));
  const artifactExpectedFiles = expectedLegacyArtifactFilesFromEvidence(currentProfile, evidence);
  if (expectedArtifactIds.length !== 16 || expectedArtifactIds.reduce(
    (sum, id) => sum + artifactExpectedFiles[id].length, 0,
  ) !== 199) fail("audited partial-run cache scope drifted");
  const documents = new Map();
  for (const filename of ["cache-preflight-initial.json", "cache-preflight.json"]) {
    const document = JSON.parse(await readFile(path.join(evidenceRoot, filename), "utf8"));
    exactKeys(document, [
      "schemaVersion", "profile", "evidencePhase", "status", "error", "downloadEvidenceSha256",
      "expectedArtifactIds", "sourceCacheRoot", "campaignStagingRoot", "derivedSidecarRoot", "missingFileStore",
      "frozenMissingFiles", "sidecarObstructions", "reusedFiles", "downloadedFiles", "networkDownloadCount",
      "phases", "lifetimePlan", "diskPlan", "derivedSidecarLifecycle", "offlineBeforeCells",
    ], `recovery ${filename}`);
    if (document.schemaVersion !== 1 || document.profile !== PROFILE_NAME
      || JSON.stringify(document.expectedArtifactIds) !== JSON.stringify(expectedArtifactIds)
      || document.downloadEvidenceSha256 !== LEGACY_DOWNLOAD_EVIDENCE_SHA256
      || document.offlineBeforeCells !== true || !Number.isInteger(document.networkDownloadCount)
      || !document.phases || !Array.isArray(document.phases.sourceCensus)
      || !Array.isArray(document.phases.staging) || !Array.isArray(document.phases.finalOffline)) {
      fail(`recovery ${filename} does not bind the audited cache contract`);
    }
    validateCachePreflightEvidence(document, {
      remainingArtifactIds: expectedArtifactIds,
      artifactExpectedFiles,
      downloadEvidenceSha256: LEGACY_DOWNLOAD_EVIDENCE_SHA256,
      guard: { cacheRoot: document.sourceCacheRoot },
      stagingRoot: document.campaignStagingRoot,
      derivedSidecarRoot: document.derivedSidecarRoot,
      missingStore: document.missingFileStore,
      expectedNonModelPaths: document.diskPlan?.nonModelPaths ?? [],
      profile: currentProfile,
    });
    documents.set(filename, document);
  }
  const initial = documents.get("cache-preflight-initial.json");
  const final = documents.get("cache-preflight.json");
  if (initial.evidencePhase !== "initial" || initial.status !== "passed" || initial.error !== null
    || initial.phases.staging.length !== 0 || initial.phases.finalOffline.length !== 0
    || initial.sidecarObstructions.length !== 0 || initial.networkDownloadCount !== 1
    || final.evidencePhase !== "final" || final.status !== "failed" || typeof final.error !== "string"
    || final.error.length === 0 || final.phases.staging.length !== 2 || final.phases.finalOffline.length !== 2
    || final.sidecarObstructions.length !== 16 || final.networkDownloadCount !== 1
    || canonicalSha256(initial) === canonicalSha256(final)) {
    fail("recovery cache evidence phases do not match the audited initial/final roles");
  }
  return { initial, final };
}

async function validateRecoveryFailureLineage(evidenceRoot, summary, currentProfile, metadata) {
  const staleName = `10-${currentProfile.cells[9].id}`;
  const staleDir = path.join(evidenceRoot, staleName);
  const stale = JSON.parse(await readFile(path.join(staleDir, "receipt.json"), "utf8"));
  validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, stale);
  validateReceipt(stale, currentProfile.cells[9], currentProfile);
  const staleFiles = await evidenceFiles(staleDir);
  validateRecoveryReceiptProvenance(stale, { ...currentProfile.cells[9], ordinal: 10 }, metadata, "recovery stale cell-10 sentinel");
  if (stale.status !== "failed" || stale.error !== "cell has not completed"
    || stale.authorityLifecycle?.providerExecution !== "not-attempted"
    || JSON.stringify(staleFiles.logs) === JSON.stringify(stale.logs)) {
    fail("recovery artifact stale cell-10 sentinel is not the audited non-prefix receipt");
  }
  const failures = [];
  for (let index = RECOVERY_IMPORTED_PREFIX_CELLS; index < currentProfile.cells.length; index += 1) {
    const cell = currentProfile.cells[index];
    const ordinalName = `${String(index + 1).padStart(2, "0")}-${cell.id}`;
    const receiptDir = path.join(evidenceRoot, "_emergency", ordinalName);
    const receipt = JSON.parse(await readFile(path.join(receiptDir, "receipt.json"), "utf8"));
    validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt);
    validateReceipt(receipt, cell, currentProfile);
    const rehashed = await evidenceFiles(receiptDir);
    for (const field of ["inputs", "outputs", "logs"]) {
      if (JSON.stringify(rehashed[field]) !== JSON.stringify(receipt[field])) {
        fail(`recovery emergency receipt ${ordinalName} failed ${field} rehash`);
      }
    }
    validateRecoveryReceiptProvenance(receipt, { ...cell, ordinal: index + 1 }, metadata, `recovery emergency receipt ${ordinalName}`);
    if (receipt.status !== "failed") {
      fail(`recovery emergency receipt ${ordinalName} is not retained failure lineage`);
    }
    failures.push({
      ordinal: index + 1, cellId: cell.id, status: "failed", path: `_emergency/${ordinalName}/receipt.json`,
    });
  }
  if (JSON.stringify(summary.receipts.slice(RECOVERY_IMPORTED_PREFIX_CELLS).map((receipt) => ({
    ordinal: currentProfile.cells.findIndex((cell) => cell.id === receipt.id) + 1,
    cellId: receipt.id, status: receipt.status, path: receipt.receipt,
  }))) !== JSON.stringify(failures)) {
    fail("recovery campaign summary does not retain the audited emergency failure lineage");
  }
  return { staleSentinel: { ordinal: 10, cellId: currentProfile.cells[9].id, path: `${staleName}/receipt.json` }, failures };
}

function validateRecoverySummary(summary, currentProfile, metadata) {
  exactKeys(summary, [
    "schemaVersion", "profile", "repositories", "execution", "receipts", "lineage", "cachePreflight",
    "authorityLifecycle", "diskFreeProbes", "finalAuthorityLifecycle", "passed", "failed", "campaignErrors",
  ], "recovery campaign summary");
  exactKeys(summary.repositories, ["sceneworks", "inference"], "recovery campaign summary repositories");
  for (const repository of ["sceneworks", "inference"]) {
    exactKeys(summary.repositories[repository], ["sha", "clean"], `recovery campaign summary ${repository} repository`);
  }
  exactKeys(summary.execution, ["runId", "runAttempt", "headSha", "headRef", "workflow", "runnerName", "runnerOs", "runnerArch"], "recovery campaign summary execution");
  if (summary.schemaVersion !== 1 || summary.profile !== PROFILE_NAME
    || summary.repositories.sceneworks.sha !== metadata.headSha || summary.repositories.sceneworks.clean !== true
    || summary.repositories.inference.sha !== HISTORICAL_INFERENCE_PIN || summary.repositories.inference.clean !== true
    || summary.execution.runId !== metadata.runId || summary.execution.runAttempt !== metadata.runAttempt
    || summary.execution.headSha !== metadata.headSha || summary.execution.headRef !== "refs/heads/feature/sc-20738-candle-cuda-parity"
    || summary.execution.workflow !== "Windows Candle worker" || summary.execution.runnerName !== "cuda-windows-2"
    || summary.execution.runnerOs !== "Windows" || summary.execution.runnerArch !== "X64"
    || !Array.isArray(summary.receipts) || summary.receipts.length !== currentProfile.cells.length
    || summary.passed !== RECOVERY_IMPORTED_PREFIX_CELLS || summary.failed !== currentProfile.cells.length - RECOVERY_IMPORTED_PREFIX_CELLS
    || !Array.isArray(summary.authorityLifecycle) || summary.authorityLifecycle.length !== 12
    || !Array.isArray(summary.diskFreeProbes) || summary.diskFreeProbes.length !== 5
    || !summary.finalAuthorityLifecycle || !Array.isArray(summary.campaignErrors) || summary.campaignErrors.length !== 1
    || !summary.campaignErrors[0].startsWith("JIT authority stage/copy/hash failed before cell flux1-schnell-q8: Error: offline JIT stage changed the frozen download partition for flux1-schnell-q8")) {
    fail("recovery campaign summary does not bind the audited partial run");
  }
  for (let index = 0; index < currentProfile.cells.length; index += 1) {
    const cell = currentProfile.cells[index];
    const ordinal = `${String(index + 1).padStart(2, "0")}-${cell.id}`;
    const row = summary.receipts[index];
    const imported = index < IMPORTED_PREFIX_CELLS;
    const passed = index < RECOVERY_IMPORTED_PREFIX_CELLS;
    exactKeys(row, (index === 7 || index === 8)
      ? ["id", "status", "receipt", "error", "source"]
      : ["id", "status", "receipt", "error", "emergencyReceiptError", "source"], `recovery summary receipt ${ordinal}`);
    const expectedPath = imported ? `_imported-prefix/${ordinal}/receipt.json`
      : passed ? `${ordinal}/receipt.json` : `_emergency/${ordinal}/receipt.json`;
    if (row.id !== cell.id || row.status !== (passed ? "passed" : "failed") || row.receipt !== expectedPath
      || row.source !== (imported ? "imported" : "continuation") || row.error !== (passed ? null : row.error)
      || (passed ? row.error !== null : typeof row.error !== "string" || !row.error.includes(cell.id))
      || (index !== 7 && index !== 8 && row.emergencyReceiptError !== null)) {
      fail(`recovery campaign summary receipt ${ordinal} does not bind its audited path/source/status`);
    }
  }
  for (const [offset, lifecycle] of summary.authorityLifecycle.entries()) {
    const index = offset + 7;
    const cell = currentProfile.cells[index];
    exactKeys(lifecycle, ["ordinal", "cellId", "staged", "activeArtifactIds", "providerExecution", "requestMemoryStrategy", "verifiedBefore", "verifiedAfter", "derivedAfter", "released", "diskProbes"], `recovery lifecycle ${index + 1}`);
    const completed = index < 9;
    const boundaryFailure = index === 9;
    if (lifecycle.ordinal !== index + 1 || lifecycle.cellId !== cell.id
      || JSON.stringify(lifecycle.activeArtifactIds) !== JSON.stringify(cell.artifactIds)
      || lifecycle.providerExecution !== (completed ? "completed" : "not-attempted")
      || (completed && (lifecycle.staged.length !== 1 || lifecycle.requestMemoryStrategy?.strategy !== "default-resident"
        || lifecycle.requestMemoryStrategy?.requestMemoryPresent !== false || lifecycle.requestMemoryStrategy?.stageResidency !== false
        || lifecycle.requestMemoryStrategy?.streamTransformerBlocks !== false || lifecycle.verifiedBefore !== true
        || lifecycle.verifiedAfter !== true || lifecycle.derivedAfter.length !== 1 || lifecycle.released.length !== 1
        || lifecycle.diskProbes.length !== 2))
      || (!completed && (lifecycle.staged.length !== 0 || lifecycle.requestMemoryStrategy !== null
        || lifecycle.verifiedBefore !== false || lifecycle.verifiedAfter !== false || lifecycle.derivedAfter.length !== 0
        || lifecycle.released.length !== (boundaryFailure ? 1 : 0) || lifecycle.diskProbes.length !== (boundaryFailure ? 1 : 0)))) {
      fail(`recovery lifecycle ${index + 1} does not bind the audited execution state`);
    }
  }
  const expectedProbePhases = [
    ["before-stage:flux1-dev-q8", 8], ["before-execution", 8], ["before-stage:flux1-schnell-q4", 9],
    ["before-execution", 9], ["before-stage:flux1-schnell-q8", 10],
  ];
  for (const [index, probe] of summary.diskFreeProbes.entries()) {
    exactKeys(probe, ["phase", "ordinal", "root", "freeBytes", "requiredFreeBytes"], `recovery disk probe ${index + 1}`);
    if (probe.phase !== expectedProbePhases[index][0] || probe.ordinal !== expectedProbePhases[index][1]
      || !Number.isSafeInteger(probe.freeBytes) || !Number.isSafeInteger(probe.requiredFreeBytes)
      || probe.requiredFreeBytes !== PRE_HYDRATION_FREE_FLOOR_BYTES) {
      fail(`recovery disk probe ${index + 1} does not bind the audited partial-run lifecycle`);
    }
  }
  exactKeys(summary.finalAuthorityLifecycle, ["stage", "derived", "derivedNamespaces", "missingStoreAbsent"], "recovery final authority lifecycle");
  for (const field of ["stage", "derived"]) {
    exactKeys(summary.finalAuthorityLifecycle[field], ["root", "files", "bytes", "sha256"], `recovery final ${field} inventory`);
    if (summary.finalAuthorityLifecycle[field].files !== 0 || summary.finalAuthorityLifecycle[field].bytes !== 0
      || summary.finalAuthorityLifecycle[field].sha256 !== createHash("sha256").digest("hex")) {
      fail(`recovery final ${field} cleanup inventory is not empty`);
    }
  }
  if (summary.finalAuthorityLifecycle.derivedNamespaces.length !== 0 || summary.finalAuthorityLifecycle.missingStoreAbsent !== true) {
    fail("recovery final lifecycle does not retain the audited cleanup state");
  }
}

export async function selectRecoveryContinuation(prefixCandidates, currentProfile) {
  const root = path.resolve(prefixCandidates);
  const entries = await readdir(root, { withFileTypes: true });
  if (entries.length !== 1 || !entries[0].isDirectory() || entries[0].isSymbolicLink()) {
    fail("expected exactly one audited recovery-continuation artifact candidate");
  }
  const candidate = path.join(root, entries[0].name);
  const metadata = JSON.parse(await readFile(path.join(candidate, "artifact-metadata.json"), "utf8"));
  validateRecoveryMetadata(metadata);
  const recoveryProfile = legacyRecoveryProfile(currentProfile);
  const evidenceRoot = path.join(candidate, "evidence");
  const files = await ordinaryTreeFiles(evidenceRoot);
  const allowedTop = new Set([
    "_imported-prefix", "_imported-boundary-residue", "08-flux1-dev-q8", "09-flux1-schnell-q4",
    "10-flux1-schnell-q8", "_emergency", "cache-preflight-initial.json", "cache-preflight.json",
    "campaign-summary.json",
  ]);
  if (files.some((file) => !allowedTop.has(path.relative(evidenceRoot, file).split(path.sep)[0]))) {
    fail("recovery artifact contains unreviewed evidence outside the audited partial run");
  }
  const summary = JSON.parse(await readFile(path.join(evidenceRoot, "campaign-summary.json"), "utf8"));
  validateRecoverySummary(summary, recoveryProfile, metadata);
  const cacheEvidence = await validateRecoveryCacheEvidence(evidenceRoot, recoveryProfile);
  if (cacheEvidence.final.error !== summary.campaignErrors[0]) {
    fail("recovery final cache evidence does not bind the campaign failure");
  }
  const cacheRecord = summary.cachePreflight;
  const cachePath = path.join(evidenceRoot, cacheRecord?.path ?? "");
  const cacheMetadata = await stat(cachePath);
  if (cacheRecord?.path !== "cache-preflight.json" || cacheRecord.bytes !== cacheMetadata.size
    || cacheRecord.sha256 !== await sha256File(cachePath)) fail("recovery summary cache-preflight hash drifted");
  const originalLineage = JSON.parse(await readFile(path.join(evidenceRoot, "_imported-prefix", "lineage.json"), "utf8"));
  exactKeys(originalLineage, [
    "kind", "sourceArtifactId", "sourceArtifactName", "sourceArtifactDigest", "sourceRunId", "sourceRunAttempt",
    "sourceHeadSha", "sourceInferenceSha", "sourceProfile", "sourceCellSemanticsSha256",
    "sourceArtifactSemanticsSha256", "importedOrdinals", "quarantinedBoundaryResidue",
  ], "recovery original lineage");
  if (canonicalSha256(originalLineage) !== canonicalSha256(expectedRecoveryOriginalLineage())) {
    fail("recovery artifact original lineage is not the exact audited 1-7 import");
  }
  await validateRecoveryBoundaryResidue(evidenceRoot, originalLineage, currentProfile);
  if (canonicalSha256(summary.lineage?.imported) !== canonicalSha256(originalLineage)
    || summary.lineage?.continuation?.runId !== metadata.runId
    || summary.lineage?.continuation?.runAttempt !== metadata.runAttempt
    || summary.lineage?.continuation?.headSha !== metadata.headSha
    || summary.lineage?.continuation?.inferenceSha !== HISTORICAL_INFERENCE_PIN
    || summary.lineage?.continuation?.profileCellSemanticsSha256 !== LEGACY_CELL_SEMANTICS_SHA256
    || summary.lineage?.continuation?.profileArtifactSemanticsSha256 !== LEGACY_RECOVERY_ARTIFACT_SEMANTICS_SHA256
    || summary.lineage?.continuation?.startOrdinal !== IMPORTED_PREFIX_CELLS + 1) {
    fail("recovery campaign summary lineage does not bind the original and partial-run segments");
  }
  const legacy = legacyPrefixProfile(currentProfile);
  const receipts = [];
  for (let index = 0; index < IMPORTED_PREFIX_CELLS; index += 1) {
    const cell = { ...legacy.cells[index], ordinal: index + 1 };
    const ordinalName = `${String(index + 1).padStart(2, "0")}-${cell.id}`;
    const receipt = JSON.parse(await readFile(path.join(evidenceRoot, "_imported-prefix", ordinalName, "receipt.json"), "utf8"));
    validateRecoveryReceiptProvenance(receipt, cell, {
      headSha: LEGACY_SCENEWORKS_HEAD, runId: "32570707303", runAttempt: "1",
    }, `recovery imported receipt ${ordinalName}`);
    validateTrustedLegacyImportedReceiptDocument(receipt);
    validateReceiptInternal(receipt, cell, legacy, null, TRUSTED_LEGACY_IMPORT_CONTEXT);
    const rehashed = await evidenceFiles(path.join(evidenceRoot, "_imported-prefix", ordinalName));
    for (const field of ["inputs", "outputs", "logs"]) if (JSON.stringify(rehashed[field]) !== JSON.stringify(receipt[field])) fail(`recovery imported receipt ${ordinalName} failed ${field} rehash`);
    receipts.push({ cell: currentProfile.cells[index], ordinalName, receipt, root: path.join(evidenceRoot, "_imported-prefix", ordinalName) });
  }
  for (let index = IMPORTED_PREFIX_CELLS; index < RECOVERY_IMPORTED_PREFIX_CELLS; index += 1) {
    const cell = { ...recoveryProfile.cells[index], ordinal: index + 1 };
    receipts.push(await validateArchivedReceipt({
      root: evidenceRoot, ordinalName: `${String(index + 1).padStart(2, "0")}-${cell.id}`,
      cell, profile: recoveryProfile, metadata,
    }));
  }
  if (summary.receipts.slice(0, RECOVERY_IMPORTED_PREFIX_CELLS).some((receipt, index) => (
    receipt.id !== currentProfile.cells[index].id || receipt.status !== "passed"
  ))) fail("recovery campaign summary PASS prefix is not exactly cells 1-9");
  const failureLineage = await validateRecoveryFailureLineage(
    evidenceRoot, summary, recoveryProfile, metadata,
  );
  return { candidate, evidenceRoot, metadata, receipts, originalLineage, failureLineage };
}

export async function importRecoveryContinuation(prefix, output) {
  const destination = path.join(output, "_imported-prefix");
  await mkdir(destination);
  for (const receipt of prefix.receipts) {
    await cp(receipt.root, path.join(destination, receipt.ordinalName), {
      recursive: true, errorOnExist: true, force: false, dereference: false,
    });
  }
  await ordinaryTreeFiles(destination);
  for (const { ordinalName, receipt } of prefix.receipts) {
    const rehashed = await evidenceFiles(path.join(destination, ordinalName));
    for (const field of ["inputs", "outputs", "logs"]) {
      if (JSON.stringify(rehashed[field]) !== JSON.stringify(receipt[field])) {
        fail(`copied recovery prefix ${ordinalName} failed ${field} rehash`);
      }
    }
  }
  const lineage = {
    kind: "recovery-continuation-prefix",
    original: prefix.originalLineage,
    recovery: {
      sourceArtifactId: prefix.metadata.artifactId, sourceArtifactName: prefix.metadata.artifactName,
      sourceArtifactSize: prefix.metadata.artifactSize, sourceArtifactDigest: prefix.metadata.artifactDigest,
      sourceRunId: prefix.metadata.runId, sourceRunAttempt: prefix.metadata.runAttempt,
      sourceHeadSha: prefix.metadata.headSha, sourceInferenceSha: prefix.metadata.inferenceSha,
      sourceProfile: prefix.metadata.profile, sourceCellSemanticsSha256: prefix.metadata.cellSemanticsSha256,
      sourceArtifactSemanticsSha256: prefix.metadata.artifactSemanticsSha256, startOrdinal: 8,
    },
    importedOrdinals: Array.from({ length: RECOVERY_IMPORTED_PREFIX_CELLS }, (_, index) => index + 1),
    quarantined: prefix.failureLineage,
  };
  await writeFile(path.join(destination, "lineage.json"), `${JSON.stringify(lineage, null, 2)}\n`, {
    encoding: "utf8", flag: "wx",
  });
  return {
    lineage,
    outcomes: prefix.receipts.map(({ cell, ordinalName }) => ({
      id: cell.id, status: "passed", receipt: `_imported-prefix/${ordinalName}/receipt.json`,
      error: null, emergencyReceiptError: null, source: "imported-recovery",
    })),
  };
}

function expectedSparseSourceLineage() {
  return {
    kind: "recovery-continuation-prefix",
    original: expectedRecoveryOriginalLineage(),
    recovery: {
      sourceArtifactId: RECOVERY_ARTIFACT_ID,
      sourceArtifactName: RECOVERY_ARTIFACT_NAME,
      sourceArtifactSize: RECOVERY_ARTIFACT_SIZE,
      sourceArtifactDigest: RECOVERY_ARTIFACT_DIGEST,
      sourceRunId: RECOVERY_RUN_ID,
      sourceRunAttempt: RECOVERY_RUN_ATTEMPT,
      sourceHeadSha: RECOVERY_SCENEWORKS_HEAD,
      sourceInferenceSha: HISTORICAL_INFERENCE_PIN,
      sourceProfile: PROFILE_NAME,
      sourceCellSemanticsSha256: LEGACY_CELL_SEMANTICS_SHA256,
      sourceArtifactSemanticsSha256: LEGACY_RECOVERY_ARTIFACT_SEMANTICS_SHA256,
      startOrdinal: 8,
    },
    importedOrdinals: Array.from({ length: RECOVERY_IMPORTED_PREFIX_CELLS }, (_, index) => index + 1),
    quarantined: {
      staleSentinel: { ordinal: 10, cellId: "flux1-schnell-q8", path: "10-flux1-schnell-q8/receipt.json" },
      failures: Array.from({ length: 10 }, (_, offset) => {
        const ordinal = offset + 10;
        return {
          ordinal,
          cellId: EXPECTED_CELLS[ordinal - 1],
          status: "failed",
          path: `_emergency/${String(ordinal).padStart(2, "0")}-${EXPECTED_CELLS[ordinal - 1]}/receipt.json`,
        };
      }),
    },
  };
}

function validateSparseRecoveryMetadata(metadata) {
  exactKeys(metadata, [
    "artifactId", "artifactName", "artifactSize", "artifactDigest", "runId", "runAttempt", "headSha",
    "inferenceSha", "profile", "cellSemanticsSha256", "artifactSemanticsSha256",
  ], "sparse recovery artifact metadata");
  if (metadata.artifactId !== SPARSE_RECOVERY_ARTIFACT_ID
    || metadata.artifactName !== SPARSE_RECOVERY_ARTIFACT_NAME
    || metadata.artifactSize !== SPARSE_RECOVERY_ARTIFACT_SIZE
    || metadata.artifactDigest !== SPARSE_RECOVERY_ARTIFACT_DIGEST
    || metadata.runId !== SPARSE_RECOVERY_RUN_ID
    || metadata.runAttempt !== SPARSE_RECOVERY_RUN_ATTEMPT
    || metadata.headSha !== SPARSE_RECOVERY_SCENEWORKS_HEAD
    || metadata.inferenceSha !== HISTORICAL_INFERENCE_PIN
    || metadata.profile !== PROFILE_NAME
    || metadata.cellSemanticsSha256 !== LEGACY_CELL_SEMANTICS_SHA256
    || metadata.artifactSemanticsSha256 !== LEGACY_RECOVERY_ARTIFACT_SEMANTICS_SHA256) {
    fail("sparse recovery artifact metadata does not bind exact artifact 9492288293");
  }
}

async function validateSparseRecoveryCacheEvidence(evidenceRoot, legacyProfile, summary) {
  const executionOrdinals = Array.from({ length: 10 }, (_, index) => index + 10);
  const expectedArtifactIds = [...new Set(executionOrdinals.flatMap(
    (ordinal) => legacyProfile.cells[ordinal - 1].artifactIds,
  ))];
  const evidence = JSON.parse(await readFile(DOWNLOAD_EVIDENCE_PATH, "utf8"));
  const artifactExpectedFiles = expectedLegacyArtifactFilesFromEvidence(legacyProfile, evidence);
  const documents = new Map();
  for (const filename of ["cache-preflight-initial.json", "cache-preflight.json"]) {
    const file = path.join(evidenceRoot, filename);
    const document = JSON.parse(await readFile(file, "utf8"));
    validateDocumentWithSchema(CACHE_PREFLIGHT_SCHEMA_PATH, document);
    if (!document.diskPlan || Object.hasOwn(document.diskPlan, "preHydrationJitSourcePeakBytes")) {
      fail(`sparse recovery ${filename} does not have the exact archived legacy disk-plan shape`);
    }
    const projectedDiskPlan = estimateJitDiskPlan(
      document.lifetimePlan,
      document.diskPlan.freeBytes,
      document.downloadedFiles.reduce((sum, file) => sum + file.bytes, 0),
      document.diskPlan.nonModelPaths,
      LEGACY_REVIEWED_ALL_AT_ONCE_SOURCE_BYTES,
    );
    const expectedLegacyDiskPlan = structuredClone(projectedDiskPlan);
    delete expectedLegacyDiskPlan.preHydrationJitSourcePeakBytes;
    expectedLegacyDiskPlan.reviewedJitSourcePeakBytes = PRE_HYDRATION_JIT_SOURCE_PEAK_BYTES;
    for (const cell of expectedLegacyDiskPlan.cells) {
      cell.requiredAdditionalBytes = Math.max(
        cell.modelAndSidecarBytes + NON_MODEL_DISK_RESERVE_BYTES,
        PRE_HYDRATION_FREE_FLOOR_BYTES,
      );
    }
    expectedLegacyDiskPlan.peakRequiredAdditionalBytes = Math.max(
      expectedLegacyDiskPlan.peakModelAndSidecarBytes + NON_MODEL_DISK_RESERVE_BYTES,
      PRE_HYDRATION_FREE_FLOOR_BYTES,
    );
    expectedLegacyDiskPlan.admitted = expectedLegacyDiskPlan.freeBytes
      >= expectedLegacyDiskPlan.peakRequiredAdditionalBytes;
    if (JSON.stringify(document.diskPlan) !== JSON.stringify(expectedLegacyDiskPlan)) {
      fail(`sparse recovery ${filename} archived legacy disk plan drifted`);
    }
    validateCachePreflightEvidence({ ...document, diskPlan: projectedDiskPlan }, {
      remainingArtifactIds: expectedArtifactIds,
      artifactExpectedFiles,
      downloadEvidenceSha256: LEGACY_DOWNLOAD_EVIDENCE_SHA256,
      guard: { cacheRoot: document.sourceCacheRoot },
      stagingRoot: document.campaignStagingRoot,
      derivedSidecarRoot: document.derivedSidecarRoot,
      missingStore: document.missingFileStore,
      expectedNonModelPaths: document.diskPlan?.nonModelPaths ?? [],
      profile: legacyProfile,
      executionOrdinals,
    });
    if (document.status !== "passed" || document.error !== null
      || document.offlineBeforeCells !== true || document.networkDownloadCount !== 1
      || document.evidencePhase !== (filename.includes("initial") ? "initial" : "final")) {
      fail(`sparse recovery ${filename} is not the audited passing cache phase`);
    }
    documents.set(filename, { file, document });
  }
  const final = documents.get("cache-preflight.json").file;
  const bytes = await stat(final);
  if (summary.cachePreflight?.path !== "cache-preflight.json"
    || summary.cachePreflight.bytes !== bytes.size
    || summary.cachePreflight.sha256 !== await sha256File(final)) {
    fail("sparse recovery summary cache-preflight hash drifted");
  }
}

function validateSparseRecoverySummary(summary, legacyProfile, metadata) {
  exactKeys(summary, [
    "schemaVersion", "profile", "repositories", "execution", "receipts", "lineage", "cachePreflight",
    "authorityLifecycle", "diskFreeProbes", "finalAuthorityLifecycle", "passed", "failed", "campaignErrors",
  ], "sparse recovery campaign summary");
  if (summary.schemaVersion !== 1 || summary.profile !== PROFILE_NAME
    || summary.repositories?.sceneworks?.sha !== metadata.headSha
    || summary.repositories.sceneworks.clean !== true
    || summary.repositories?.inference?.sha !== HISTORICAL_INFERENCE_PIN
    || summary.repositories.inference.clean !== true
    || summary.execution?.runId !== metadata.runId
    || summary.execution?.runAttempt !== metadata.runAttempt
    || summary.execution?.headSha !== metadata.headSha
    || summary.execution?.headRef !== "refs/heads/feature/sc-20738-candle-cuda-parity"
    || summary.execution?.workflow !== "Windows Candle worker"
    || summary.execution?.runnerName !== "cuda-windows-2"
    || summary.execution?.runnerOs !== "Windows" || summary.execution?.runnerArch !== "X64"
    || !Array.isArray(summary.receipts) || summary.receipts.length !== legacyProfile.cells.length
    || summary.passed !== SPARSE_IMPORTED_ORDINALS.length
    || summary.failed !== SPARSE_EXECUTION_ORDINALS.length
    || !Array.isArray(summary.campaignErrors) || summary.campaignErrors.length !== 0
    || !Array.isArray(summary.authorityLifecycle) || summary.authorityLifecycle.length !== 10
    || !Array.isArray(summary.diskFreeProbes) || summary.diskFreeProbes.length !== 24) {
    fail("sparse recovery campaign summary does not bind the audited 16-PASS/3-failure run");
  }
  const failed = new Set(SPARSE_EXECUTION_ORDINALS);
  for (let index = 0; index < legacyProfile.cells.length; index += 1) {
    const ordinal = index + 1;
    const cell = legacyProfile.cells[index];
    const row = summary.receipts[index];
    const imported = ordinal <= RECOVERY_IMPORTED_PREFIX_CELLS;
    const expectedStatus = failed.has(ordinal) ? "failed" : "passed";
    const ordinalName = `${String(ordinal).padStart(2, "0")}-${cell.id}`;
    const expectedPath = imported ? `_imported-prefix/${ordinalName}/receipt.json`
      : `${ordinalName}/receipt.json`;
    const expectedSource = imported ? "imported-recovery" : "continuation";
    if (row?.id !== cell.id || row.status !== expectedStatus || row.receipt !== expectedPath
      || row.source !== expectedSource
      || (expectedStatus === "passed" ? row.error !== null
        : typeof row.error !== "string" || row.error.length === 0)) {
      fail(`sparse recovery summary receipt ${ordinalName} drifted from exact path/source/status`);
    }
  }
  const sourceLineage = expectedSparseSourceLineage();
  if (canonicalSha256(summary.lineage?.imported) !== canonicalSha256(sourceLineage)
    || summary.lineage?.continuation?.runId !== metadata.runId
    || summary.lineage?.continuation?.runAttempt !== metadata.runAttempt
    || summary.lineage?.continuation?.headSha !== metadata.headSha
    || summary.lineage?.continuation?.inferenceSha !== HISTORICAL_INFERENCE_PIN
    || summary.lineage?.continuation?.profileCellSemanticsSha256 !== LEGACY_CELL_SEMANTICS_SHA256
    || summary.lineage?.continuation?.profileArtifactSemanticsSha256 !== LEGACY_RECOVERY_ARTIFACT_SEMANTICS_SHA256
    || summary.lineage?.continuation?.startOrdinal !== 10) {
    fail("sparse recovery campaign lineage does not bind its exact imported and continuation segments");
  }
  const empty = createHash("sha256").digest("hex");
  if (summary.finalAuthorityLifecycle?.stage?.files !== 0
    || summary.finalAuthorityLifecycle?.stage?.bytes !== 0
    || summary.finalAuthorityLifecycle?.stage?.sha256 !== empty
    || summary.finalAuthorityLifecycle?.derived?.files !== 0
    || summary.finalAuthorityLifecycle?.derived?.bytes !== 0
    || summary.finalAuthorityLifecycle?.derived?.sha256 !== empty
    || summary.finalAuthorityLifecycle?.derivedNamespaces?.length !== 0
    || summary.finalAuthorityLifecycle?.missingStoreAbsent !== true) {
    fail("sparse recovery final authority cleanup is not exact and empty");
  }
}

export async function selectSparseRecovery(prefixCandidates, currentProfile) {
  validateProfile(currentProfile);
  const root = path.resolve(prefixCandidates);
  const entries = await readdir(root, { withFileTypes: true });
  if (entries.length !== 1 || !entries[0].isDirectory() || entries[0].isSymbolicLink()) {
    fail("expected exactly one exact sparse-recovery artifact candidate");
  }
  const candidate = path.join(root, entries[0].name);
  const metadata = JSON.parse(await readFile(path.join(candidate, "artifact-metadata.json"), "utf8"));
  validateSparseRecoveryMetadata(metadata);
  const evidenceRoot = path.join(candidate, "evidence");
  const files = await ordinaryTreeFiles(evidenceRoot);
  const allowedTop = new Set([
    "_imported-prefix", ...Array.from({ length: 10 }, (_, index) => {
      const ordinal = index + 10;
      return `${String(ordinal).padStart(2, "0")}-${EXPECTED_CELLS[ordinal - 1]}`;
    }),
    "cache-preflight-initial.json", "cache-preflight.json", "campaign-summary.json",
  ]);
  if (files.some((file) => !allowedTop.has(path.relative(evidenceRoot, file).split(path.sep)[0]))) {
    fail("sparse recovery artifact contains evidence outside the exact audited campaign");
  }
  const legacyProfile = legacyRecoveryProfile(currentProfile);
  const archivedLineage = JSON.parse(await readFile(
    path.join(evidenceRoot, "_imported-prefix", "lineage.json"), "utf8",
  ));
  if (canonicalSha256(archivedLineage) !== canonicalSha256(expectedSparseSourceLineage())) {
    fail("sparse recovery archived lineage file drifted from the exact audited 1-9 source");
  }
  const summary = JSON.parse(await readFile(path.join(evidenceRoot, "campaign-summary.json"), "utf8"));
  validateSparseRecoverySummary(summary, legacyProfile, metadata);
  await validateSparseRecoveryCacheEvidence(evidenceRoot, legacyProfile, summary);

  const receipts = [];
  const failures = [];
  const compatibility = [];
  const archivedLifecycles = [];
  for (let index = 0; index < legacyProfile.cells.length; index += 1) {
    const ordinal = index + 1;
    const cell = { ...legacyProfile.cells[index], ordinal };
    const ordinalName = `${String(ordinal).padStart(2, "0")}-${cell.id}`;
    const imported = ordinal <= RECOVERY_IMPORTED_PREFIX_CELLS;
    const cellDir = path.join(evidenceRoot, ...(imported ? ["_imported-prefix", ordinalName] : [ordinalName]));
    const receipt = JSON.parse(await readFile(path.join(cellDir, "receipt.json"), "utf8"));
    const provenance = ordinal <= IMPORTED_PREFIX_CELLS
      ? { headSha: LEGACY_SCENEWORKS_HEAD, runId: "32570707303", runAttempt: "1" }
      : ordinal <= RECOVERY_IMPORTED_PREFIX_CELLS
        ? { headSha: RECOVERY_SCENEWORKS_HEAD, runId: RECOVERY_RUN_ID, runAttempt: RECOVERY_RUN_ATTEMPT }
        : metadata;
    validateRecoveryReceiptProvenance(receipt, cell, provenance, `sparse recovery receipt ${ordinalName}`);
    if (ordinal <= IMPORTED_PREFIX_CELLS) {
      validateTrustedLegacyImportedReceiptDocument(receipt);
      const originalProfile = legacyPrefixProfile(currentProfile);
      validateReceiptInternal(receipt, { ...originalProfile.cells[index], ordinal }, originalProfile, null, TRUSTED_LEGACY_IMPORT_CONTEXT);
    } else {
      validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt);
      validateReceipt(receipt, cell, legacyProfile);
    }
    const rehashed = await evidenceFiles(cellDir);
    for (const field of ["inputs", "outputs", "logs"]) {
      if (JSON.stringify(rehashed[field]) !== JSON.stringify(receipt[field])) {
        fail(`sparse recovery receipt ${ordinalName} failed ${field} rehash`);
      }
    }
    const expectedStatus = SPARSE_EXECUTION_ORDINALS.includes(ordinal) ? "failed" : "passed";
    if (receipt.status !== expectedStatus) {
      fail(`sparse recovery receipt ${ordinalName} is not the exact audited ${expectedStatus.toUpperCase()}`);
    }
    if (expectedStatus === "failed") {
      failures.push({ ordinal, cellId: cell.id, path: `${ordinalName}/receipt.json`, status: "failed" });
      if (!imported) archivedLifecycles.push(receipt.authorityLifecycle);
      continue;
    }
    const currentCellSha256 = canonicalSha256(currentProfile.cells[index]);
    const legacyCellSha256 = canonicalSha256(legacyProfile.cells[index]);
    if (legacyCellSha256 !== currentCellSha256) {
      fail(`sparse recovery PASS ${ordinalName} is incompatible with the corrected profile`);
    }
    compatibility.push({ ordinal, cellId: cell.id, cellSemanticsSha256: currentCellSha256 });
    receipts.push({ cell: currentProfile.cells[index], ordinalName, receipt, root: cellDir });
    if (!imported) archivedLifecycles.push(receipt.authorityLifecycle);
  }
  if (JSON.stringify(receipts.map(({ cell }) => currentProfile.cells.indexOf(cell) + 1))
    !== JSON.stringify(SPARSE_IMPORTED_ORDINALS)
    || JSON.stringify(failures.map(({ ordinal }) => ordinal)) !== JSON.stringify(SPARSE_EXECUTION_ORDINALS)) {
    fail("sparse recovery PASS/failure partition drifted from [1-13,15-17]/[14,18,19]");
  }
  if (canonicalSha256(summary.authorityLifecycle) !== canonicalSha256(archivedLifecycles)
    || canonicalSha256(summary.diskFreeProbes)
      !== canonicalSha256(archivedLifecycles.flatMap((lifecycle) => lifecycle.diskProbes))) {
    fail("sparse recovery summary lifecycle/probes drifted from the rehashed receipts");
  }
  return {
    candidate, evidenceRoot, metadata, receipts, failures, compatibility,
    sourceLineage: summary.lineage,
  };
}

export async function importSparseRecovery(prefix, output) {
  const destination = path.join(output, "_imported-prefix");
  await mkdir(destination);
  for (const receipt of prefix.receipts) {
    await cp(receipt.root, path.join(destination, receipt.ordinalName), {
      recursive: true, errorOnExist: true, force: false, dereference: false,
    });
  }
  await ordinaryTreeFiles(destination);
  for (const { ordinalName, receipt } of prefix.receipts) {
    const rehashed = await evidenceFiles(path.join(destination, ordinalName));
    for (const field of ["inputs", "outputs", "logs"]) {
      if (JSON.stringify(rehashed[field]) !== JSON.stringify(receipt[field])) {
        fail(`copied sparse recovery PASS ${ordinalName} failed ${field} rehash`);
      }
    }
  }
  const lineage = {
    kind: "sparse-pass-recovery",
    sourceArtifactId: prefix.metadata.artifactId,
    sourceArtifactName: prefix.metadata.artifactName,
    sourceArtifactSize: prefix.metadata.artifactSize,
    sourceArtifactDigest: prefix.metadata.artifactDigest,
    sourceRunId: prefix.metadata.runId,
    sourceRunAttempt: prefix.metadata.runAttempt,
    sourceHeadSha: prefix.metadata.headSha,
    sourceInferenceSha: prefix.metadata.inferenceSha,
    sourceProfile: prefix.metadata.profile,
    sourceCellSemanticsSha256: prefix.metadata.cellSemanticsSha256,
    targetCellSemanticsSha256: EXPECTED_CELL_SEMANTICS_SHA256,
    sourceArtifactSemanticsSha256: prefix.metadata.artifactSemanticsSha256,
    importedOrdinals: SPARSE_IMPORTED_ORDINALS,
    executionOrdinals: SPARSE_EXECUTION_ORDINALS,
    compatibility: prefix.compatibility,
    quarantinedFailures: prefix.failures,
    priorLineage: prefix.sourceLineage,
  };
  await writeFile(path.join(destination, "lineage.json"), `${JSON.stringify(lineage, null, 2)}\n`, {
    encoding: "utf8", flag: "wx",
  });
  return {
    lineage,
    outcomes: prefix.receipts.map(({ cell, ordinalName }) => ({
      id: cell.id, status: "passed", receipt: `_imported-prefix/${ordinalName}/receipt.json`,
      error: null, emergencyReceiptError: null, source: "imported-sparse-recovery",
    })),
  };
}

function validatePass18RecoveryMetadata(metadata) {
  exactKeys(metadata, [
    "artifactId", "artifactName", "artifactSize", "artifactDigest", "runId", "runAttempt", "headSha",
    "inferenceSha", "profile", "cellSemanticsSha256", "artifactSemanticsSha256",
  ], "18-PASS recovery artifact metadata");
  if (metadata.artifactId !== PASS18_RECOVERY_ARTIFACT_ID
    || metadata.artifactName !== PASS18_RECOVERY_ARTIFACT_NAME
    || metadata.artifactSize !== PASS18_RECOVERY_ARTIFACT_SIZE
    || metadata.artifactDigest !== PASS18_RECOVERY_ARTIFACT_DIGEST
    || metadata.runId !== PASS18_RECOVERY_RUN_ID
    || metadata.runAttempt !== PASS18_RECOVERY_RUN_ATTEMPT
    || metadata.headSha !== PASS18_RECOVERY_SCENEWORKS_HEAD
    || metadata.inferenceSha !== HISTORICAL_INFERENCE_PIN
    || metadata.profile !== PROFILE_NAME
    || metadata.cellSemanticsSha256 !== EXPECTED_CELL_SEMANTICS_SHA256
    || metadata.artifactSemanticsSha256 !== EXPECTED_ARTIFACT_SEMANTICS_SHA256) {
    fail("18-PASS recovery metadata does not bind exact artifact 9498929065");
  }
}

function validatePass18ImportedLineage(lineage, currentProfile) {
  const compatibility = SPARSE_IMPORTED_ORDINALS.map((ordinal) => ({
    ordinal,
    cellId: currentProfile.cells[ordinal - 1].id,
    cellSemanticsSha256: canonicalSha256(currentProfile.cells[ordinal - 1]),
  }));
  if (lineage?.kind !== "sparse-pass-recovery"
    || lineage.sourceArtifactId !== SPARSE_RECOVERY_ARTIFACT_ID
    || lineage.sourceArtifactName !== SPARSE_RECOVERY_ARTIFACT_NAME
    || lineage.sourceArtifactSize !== SPARSE_RECOVERY_ARTIFACT_SIZE
    || lineage.sourceArtifactDigest !== SPARSE_RECOVERY_ARTIFACT_DIGEST
    || lineage.sourceRunId !== SPARSE_RECOVERY_RUN_ID
    || lineage.sourceRunAttempt !== SPARSE_RECOVERY_RUN_ATTEMPT
    || lineage.sourceHeadSha !== SPARSE_RECOVERY_SCENEWORKS_HEAD
    || lineage.sourceInferenceSha !== HISTORICAL_INFERENCE_PIN
    || lineage.sourceProfile !== PROFILE_NAME
    || lineage.sourceCellSemanticsSha256 !== LEGACY_CELL_SEMANTICS_SHA256
    || lineage.targetCellSemanticsSha256 !== EXPECTED_CELL_SEMANTICS_SHA256
    || lineage.sourceArtifactSemanticsSha256 !== LEGACY_RECOVERY_ARTIFACT_SEMANTICS_SHA256
    || JSON.stringify(lineage.importedOrdinals) !== JSON.stringify(SPARSE_IMPORTED_ORDINALS)
    || JSON.stringify(lineage.executionOrdinals) !== JSON.stringify(SPARSE_EXECUTION_ORDINALS)
    || canonicalSha256(lineage.compatibility) !== canonicalSha256(compatibility)
    || canonicalSha256(lineage.quarantinedFailures) !== canonicalSha256(
      SPARSE_EXECUTION_ORDINALS.map((ordinal) => ({
        ordinal, cellId: currentProfile.cells[ordinal - 1].id,
        path: `${String(ordinal).padStart(2, "0")}-${currentProfile.cells[ordinal - 1].id}/receipt.json`,
        status: "failed",
      })),
    )
    || canonicalSha256(lineage.priorLineage?.imported) !== canonicalSha256(expectedSparseSourceLineage())
    || lineage.priorLineage?.continuation?.runId !== SPARSE_RECOVERY_RUN_ID
    || lineage.priorLineage?.continuation?.runAttempt !== SPARSE_RECOVERY_RUN_ATTEMPT
    || lineage.priorLineage?.continuation?.headSha !== SPARSE_RECOVERY_SCENEWORKS_HEAD
    || lineage.priorLineage?.continuation?.inferenceSha !== HISTORICAL_INFERENCE_PIN
    || lineage.priorLineage?.continuation?.profileCellSemanticsSha256 !== LEGACY_CELL_SEMANTICS_SHA256
    || lineage.priorLineage?.continuation?.profileArtifactSemanticsSha256
      !== LEGACY_RECOVERY_ARTIFACT_SEMANTICS_SHA256
    || lineage.priorLineage?.continuation?.startOrdinal !== 10) {
    fail("18-PASS recovery prior sparse lineage drifted from exact artifact 9492288293");
  }
}

export async function validatePass18RecoveryCacheEvidence(evidenceRoot, currentProfile, summary) {
  const executionOrdinals = SPARSE_EXECUTION_ORDINALS;
  const expectedArtifactIds = [...new Set(executionOrdinals.flatMap(
    (ordinal) => currentProfile.cells[ordinal - 1].artifactIds,
  ))];
  const evidenceBytes = await readFile(DOWNLOAD_EVIDENCE_PATH, "utf8");
  const { artifactExpectedFiles, downloadEvidenceSha256 } =
    expectedCurrentArtifactFilesFromEvidenceBytes(currentProfile, evidenceBytes);
  for (const filename of ["cache-preflight-initial.json", "cache-preflight.json"]) {
    const file = path.join(evidenceRoot, filename);
    const document = JSON.parse(await readFile(file, "utf8"));
    validateDocumentWithSchema(CACHE_PREFLIGHT_SCHEMA_PATH, document);
    validateCachePreflightEvidence(document, {
      remainingArtifactIds: expectedArtifactIds,
      artifactExpectedFiles,
      downloadEvidenceSha256,
      guard: { cacheRoot: document.sourceCacheRoot },
      stagingRoot: document.campaignStagingRoot,
      derivedSidecarRoot: document.derivedSidecarRoot,
      missingStore: document.missingFileStore,
      expectedNonModelPaths: document.diskPlan?.nonModelPaths ?? [],
      profile: currentProfile,
      executionOrdinals,
    });
    if (document.status !== "passed" || document.error !== null
      || document.offlineBeforeCells !== true || document.networkDownloadCount !== 32
      || document.downloadedFiles.reduce((sum, row) => sum + row.bytes, 0) !== 7_823_307_202
      || document.diskPlan?.preHydrationJitSourcePeakBytes !== PRE_HYDRATION_JIT_SOURCE_PEAK_BYTES
      || document.diskPlan?.reviewedJitSourcePeakBytes !== REVIEWED_JIT_SOURCE_PEAK_BYTES
      || document.diskPlan?.peakRequiredAdditionalBytes !== REVIEWED_FREE_FLOOR_BYTES
      || document.evidencePhase !== (filename.includes("initial") ? "initial" : "final")) {
      fail(`18-PASS recovery ${filename} is not the exact current passing cache phase`);
    }
  }
  const final = path.join(evidenceRoot, "cache-preflight.json");
  const bytes = await stat(final);
  if (summary.cachePreflight?.path !== "cache-preflight.json"
    || summary.cachePreflight.bytes !== bytes.size
    || summary.cachePreflight.sha256 !== await sha256File(final)) {
    fail("18-PASS recovery summary cache-preflight hash drifted");
  }
}

export function validatePass18RecoverySummary(summary, currentProfile, metadata) {
  exactKeys(summary, [
    "schemaVersion", "profile", "repositories", "execution", "receipts", "lineage", "cachePreflight",
    "authorityLifecycle", "diskFreeProbes", "finalAuthorityLifecycle", "passed", "failed", "campaignErrors",
  ], "18-PASS recovery campaign summary");
  if (summary.schemaVersion !== 1 || summary.profile !== PROFILE_NAME
    || summary.repositories?.sceneworks?.sha !== metadata.headSha
    || summary.repositories.sceneworks.clean !== true
    || summary.repositories?.inference?.sha !== HISTORICAL_INFERENCE_PIN
    || summary.repositories.inference.clean !== true
    || summary.execution?.runId !== metadata.runId
    || summary.execution?.runAttempt !== metadata.runAttempt
    || summary.execution?.headSha !== metadata.headSha
    || summary.execution?.headRef !== "refs/heads/feature/sc-20738-candle-cuda-parity"
    || summary.execution?.workflow !== "Windows Candle worker"
    || summary.execution?.runnerName !== "cuda-windows"
    || summary.execution?.runnerOs !== "Windows" || summary.execution?.runnerArch !== "X64"
    || !Array.isArray(summary.receipts) || summary.receipts.length !== currentProfile.cells.length
    || summary.passed !== PASS18_IMPORTED_ORDINALS.length || summary.failed !== 1
    || !Array.isArray(summary.campaignErrors) || summary.campaignErrors.length !== 0
    || !Array.isArray(summary.authorityLifecycle) || summary.authorityLifecycle.length !== 3
    || !Array.isArray(summary.diskFreeProbes) || summary.diskFreeProbes.length !== 11) {
    fail("18-PASS recovery summary does not bind the audited 18-PASS/1-failure run");
  }
  const imported = new Set(SPARSE_IMPORTED_ORDINALS);
  for (let index = 0; index < currentProfile.cells.length; index += 1) {
    const ordinal = index + 1;
    const cell = currentProfile.cells[index];
    const ordinalName = `${String(ordinal).padStart(2, "0")}-${cell.id}`;
    const row = summary.receipts[index];
    const expectedStatus = ordinal === PASS18_EXECUTION_ORDINALS[0] ? "failed" : "passed";
    const expectedPath = imported.has(ordinal)
      ? `_imported-prefix/${ordinalName}/receipt.json` : `${ordinalName}/receipt.json`;
    const expectedSource = ordinal <= 17 && ordinal !== 14
      ? "imported-sparse-recovery" : "continuation";
    if (row?.id !== cell.id || row.status !== expectedStatus || row.receipt !== expectedPath
      || row.source !== expectedSource
      || (expectedStatus === "passed" ? row.error !== null
        : typeof row.error !== "string" || row.error.length === 0)) {
      fail(`18-PASS recovery summary receipt ${ordinalName} drifted from exact path/source/status`);
    }
  }
  validatePass18ImportedLineage(summary.lineage?.imported, currentProfile);
  if (summary.lineage?.continuation?.runId !== metadata.runId
    || summary.lineage?.continuation?.runAttempt !== metadata.runAttempt
    || summary.lineage?.continuation?.headSha !== metadata.headSha
    || summary.lineage?.continuation?.inferenceSha !== HISTORICAL_INFERENCE_PIN
    || summary.lineage?.continuation?.profileCellSemanticsSha256 !== EXPECTED_CELL_SEMANTICS_SHA256
    || summary.lineage?.continuation?.profileArtifactSemanticsSha256 !== EXPECTED_ARTIFACT_SEMANTICS_SHA256
    || JSON.stringify(summary.lineage?.continuation?.executionOrdinals)
      !== JSON.stringify(SPARSE_EXECUTION_ORDINALS)) {
    fail("18-PASS recovery continuation lineage drifted from the exact audited run");
  }
  const empty = createHash("sha256").digest("hex");
  if (summary.finalAuthorityLifecycle?.stage?.files !== 0
    || summary.finalAuthorityLifecycle?.stage?.bytes !== 0
    || summary.finalAuthorityLifecycle?.stage?.sha256 !== empty
    || summary.finalAuthorityLifecycle?.derived?.files !== 0
    || summary.finalAuthorityLifecycle?.derived?.bytes !== 0
    || summary.finalAuthorityLifecycle?.derived?.sha256 !== empty
    || summary.finalAuthorityLifecycle?.derivedNamespaces?.length !== 0
    || summary.finalAuthorityLifecycle?.missingStoreAbsent !== true) {
    fail("18-PASS recovery final authority cleanup is not exact and empty");
  }
}

export async function selectPass18Recovery(
  prefixCandidates, currentProfile, { validateCacheEvidence = validatePass18RecoveryCacheEvidence } = {},
) {
  validateProfile(currentProfile);
  const root = path.resolve(prefixCandidates);
  const entries = await readdir(root, { withFileTypes: true });
  if (entries.length !== 1 || !entries[0].isDirectory() || entries[0].isSymbolicLink()) {
    fail("expected exactly one exact 18-PASS recovery artifact candidate");
  }
  const candidate = path.join(root, entries[0].name);
  const metadata = JSON.parse(await readFile(path.join(candidate, "artifact-metadata.json"), "utf8"));
  validatePass18RecoveryMetadata(metadata);
  const evidenceRoot = path.join(candidate, "evidence");
  const files = await ordinaryTreeFiles(evidenceRoot);
  const rootOrdinals = SPARSE_EXECUTION_ORDINALS.map((ordinal) => (
    `${String(ordinal).padStart(2, "0")}-${EXPECTED_CELLS[ordinal - 1]}`
  ));
  const allowedTop = new Set([
    "_imported-prefix", ...rootOrdinals,
    "cache-preflight-initial.json", "cache-preflight.json", "campaign-summary.json",
  ]);
  if (files.some((file) => !allowedTop.has(path.relative(evidenceRoot, file).split(path.sep)[0]))) {
    fail("18-PASS recovery artifact contains evidence outside the exact audited campaign");
  }
  const summary = JSON.parse(await readFile(path.join(evidenceRoot, "campaign-summary.json"), "utf8"));
  validatePass18RecoverySummary(summary, currentProfile, metadata);
  await validateCacheEvidence(evidenceRoot, currentProfile, summary);

  const imported = new Set(SPARSE_IMPORTED_ORDINALS);
  const receipts = [];
  const failures = [];
  const compatibility = [];
  const archivedLifecycles = [];
  for (let index = 0; index < currentProfile.cells.length; index += 1) {
    const ordinal = index + 1;
    const cell = { ...currentProfile.cells[index], ordinal };
    const ordinalName = `${String(ordinal).padStart(2, "0")}-${cell.id}`;
    const cellDir = path.join(evidenceRoot, ...(imported.has(ordinal)
      ? ["_imported-prefix", ordinalName] : [ordinalName]));
    const receipt = JSON.parse(await readFile(path.join(cellDir, "receipt.json"), "utf8"));
    const provenance = ordinal <= IMPORTED_PREFIX_CELLS
      ? { headSha: LEGACY_SCENEWORKS_HEAD, runId: "32570707303", runAttempt: "1" }
      : ordinal <= RECOVERY_IMPORTED_PREFIX_CELLS
        ? { headSha: RECOVERY_SCENEWORKS_HEAD, runId: RECOVERY_RUN_ID, runAttempt: RECOVERY_RUN_ATTEMPT }
        : ordinal <= 17 && ordinal !== PASS18_EXECUTION_ORDINALS[0]
          ? { headSha: SPARSE_RECOVERY_SCENEWORKS_HEAD, runId: SPARSE_RECOVERY_RUN_ID, runAttempt: SPARSE_RECOVERY_RUN_ATTEMPT }
          : metadata;
    validateRecoveryReceiptProvenance(receipt, cell, provenance, `18-PASS recovery receipt ${ordinalName}`);
    if (ordinal <= IMPORTED_PREFIX_CELLS) {
      validateTrustedLegacyImportedReceiptDocument(receipt);
      const originalProfile = legacyPrefixProfile(currentProfile);
      validateReceiptInternal(receipt, { ...originalProfile.cells[index], ordinal }, originalProfile, null, TRUSTED_LEGACY_IMPORT_CONTEXT);
    } else {
      validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt);
      validateReceipt(receipt, cell, currentProfile);
    }
    const rehashed = await evidenceFiles(cellDir);
    for (const field of ["inputs", "outputs", "logs"]) {
      if (JSON.stringify(rehashed[field]) !== JSON.stringify(receipt[field])) {
        fail(`18-PASS recovery receipt ${ordinalName} failed ${field} rehash`);
      }
    }
    const expectedStatus = ordinal === PASS18_EXECUTION_ORDINALS[0] ? "failed" : "passed";
    if (receipt.status !== expectedStatus) {
      fail(`18-PASS recovery receipt ${ordinalName} is not the exact audited ${expectedStatus.toUpperCase()}`);
    }
    if (!imported.has(ordinal)) archivedLifecycles.push(receipt.authorityLifecycle);
    if (expectedStatus === "failed") {
      failures.push({ ordinal, cellId: cell.id, path: `${ordinalName}/receipt.json`, status: "failed" });
      continue;
    }
    compatibility.push({
      ordinal, cellId: cell.id, cellSemanticsSha256: canonicalSha256(currentProfile.cells[index]),
    });
    receipts.push({ cell: currentProfile.cells[index], ordinalName, receipt, root: cellDir });
  }
  if (JSON.stringify(receipts.map(({ cell }) => currentProfile.cells.indexOf(cell) + 1))
      !== JSON.stringify(PASS18_IMPORTED_ORDINALS)
    || JSON.stringify(failures.map(({ ordinal }) => ordinal)) !== JSON.stringify(PASS18_EXECUTION_ORDINALS)) {
    fail("18-PASS recovery partition drifted from [1-13,15-19]/[14]");
  }
  if (canonicalSha256(summary.authorityLifecycle) !== canonicalSha256(archivedLifecycles)
    || canonicalSha256(summary.diskFreeProbes)
      !== canonicalSha256(archivedLifecycles.flatMap((lifecycle) => lifecycle.diskProbes))) {
    fail("18-PASS recovery summary lifecycle/probes drifted from rehashed receipts");
  }
  return {
    candidate, evidenceRoot, metadata, receipts, failures, compatibility,
    sourceLineage: summary.lineage,
  };
}

export async function importPass18Recovery(prefix, output) {
  const destination = path.join(output, "_imported-prefix");
  await mkdir(destination);
  for (const receipt of prefix.receipts) {
    await cp(receipt.root, path.join(destination, receipt.ordinalName), {
      recursive: true, errorOnExist: true, force: false, dereference: false,
    });
  }
  await ordinaryTreeFiles(destination);
  for (const { ordinalName, receipt } of prefix.receipts) {
    const rehashed = await evidenceFiles(path.join(destination, ordinalName));
    for (const field of ["inputs", "outputs", "logs"]) {
      if (JSON.stringify(rehashed[field]) !== JSON.stringify(receipt[field])) {
        fail(`copied 18-PASS recovery ${ordinalName} failed ${field} rehash`);
      }
    }
  }
  const lineage = {
    kind: "terminal-single-cell-recovery",
    sourceArtifactId: prefix.metadata.artifactId,
    sourceArtifactName: prefix.metadata.artifactName,
    sourceArtifactSize: prefix.metadata.artifactSize,
    sourceArtifactDigest: prefix.metadata.artifactDigest,
    sourceRunId: prefix.metadata.runId,
    sourceRunAttempt: prefix.metadata.runAttempt,
    sourceHeadSha: prefix.metadata.headSha,
    sourceInferenceSha: prefix.metadata.inferenceSha,
    sourceProfile: prefix.metadata.profile,
    sourceCellSemanticsSha256: prefix.metadata.cellSemanticsSha256,
    targetCellSemanticsSha256: EXPECTED_CELL_SEMANTICS_SHA256,
    sourceArtifactSemanticsSha256: prefix.metadata.artifactSemanticsSha256,
    importedOrdinals: PASS18_IMPORTED_ORDINALS,
    executionOrdinals: PASS18_EXECUTION_ORDINALS,
    compatibility: prefix.compatibility,
    quarantinedFailures: prefix.failures,
    priorLineage: prefix.sourceLineage,
  };
  await writeFile(path.join(destination, "lineage.json"), `${JSON.stringify(lineage, null, 2)}\n`, {
    encoding: "utf8", flag: "wx",
  });
  return {
    lineage,
    outcomes: prefix.receipts.map(({ cell, ordinalName }) => ({
      id: cell.id, status: "passed", receipt: `_imported-prefix/${ordinalName}/receipt.json`,
      error: null, emergencyReceiptError: null, source: "imported-pass18-recovery",
    })),
  };
}

function validateOpenPoseRecoveryMetadata(metadata) {
  exactKeys(metadata, [
    "artifactId", "artifactName", "artifactSize", "artifactDigest", "runId", "runAttempt", "headSha",
    "inferenceSha", "profile", "cellSemanticsSha256", "artifactSemanticsSha256",
  ], "OpenPose recovery source artifact metadata");
  if (metadata.artifactId !== OPENPOSE_RECOVERY_ARTIFACT_ID
    || metadata.artifactName !== OPENPOSE_RECOVERY_ARTIFACT_NAME
    || metadata.artifactSize !== OPENPOSE_RECOVERY_ARTIFACT_SIZE
    || metadata.artifactDigest !== OPENPOSE_RECOVERY_ARTIFACT_DIGEST
    || metadata.runId !== OPENPOSE_RECOVERY_RUN_ID
    || metadata.runAttempt !== OPENPOSE_RECOVERY_RUN_ATTEMPT
    || metadata.headSha !== OPENPOSE_RECOVERY_SCENEWORKS_HEAD
    // The capture revision belongs to the archived artifact, not to this repository: the artifact's
    // own `sha256:` digest is already asserted above, so it pins the recorded revision far more
    // tightly than a harness constant could. Require only the exact 40-hex shape here and bind the
    // value by agreement with the artifact's own campaign summary and per-cell receipts below.
    || !SHA40.test(metadata.inferenceSha)
    || metadata.profile !== PROFILE_NAME
    || metadata.cellSemanticsSha256 !== EXPECTED_CELL_SEMANTICS_SHA256
    || metadata.artifactSemanticsSha256 !== EXPECTED_ARTIFACT_SEMANTICS_SHA256) {
    fail("OpenPose recovery metadata does not bind exact artifact 9500244306");
  }
}

function validateOpenPoseRecoverySummary(summary, currentProfile, metadata) {
  if (summary?.schemaVersion !== 1 || summary.profile !== PROFILE_NAME
    || summary.repositories?.sceneworks?.sha !== metadata.headSha
    || summary.repositories?.sceneworks?.clean !== true
    || summary.repositories?.inference?.sha !== metadata.inferenceSha
    || summary.repositories?.inference?.clean !== true
    || summary.execution?.runId !== metadata.runId
    || summary.execution?.runAttempt !== metadata.runAttempt
    || summary.execution?.headSha !== metadata.headSha
    || summary.execution?.headRef !== "refs/heads/feature/sc-20738-candle-cuda-parity"
    || summary.execution?.workflow !== "Windows Candle worker"
    || summary.passed !== currentProfile.cells.length || summary.failed !== 0
    || !Array.isArray(summary.campaignErrors) || summary.campaignErrors.length !== 0
    || !Array.isArray(summary.receipts) || summary.receipts.length !== currentProfile.cells.length
    || JSON.stringify(summary.lineage?.continuation?.executionOrdinals) !== JSON.stringify(PASS18_EXECUTION_ORDINALS)) {
    fail("OpenPose recovery source summary does not bind the audited 19-PASS final run");
  }
  for (const [index, cell] of currentProfile.cells.entries()) {
    const ordinal = index + 1;
    const expectedPath = ordinal === 14
      ? `14-${cell.id}/receipt.json`
      : `_imported-prefix/${String(ordinal).padStart(2, "0")}-${cell.id}/receipt.json`;
    const source = ordinal === 14 ? "continuation" : "imported-pass18-recovery";
    const row = summary.receipts[index];
    if (row?.id !== cell.id || row.status !== "passed" || row.receipt !== expectedPath
      || row.source !== source || row.error !== null) {
      fail(`OpenPose recovery source summary receipt ${ordinal} is not exact 19-PASS evidence`);
    }
  }
}

// Import the 14 non-OpenPose PASS receipts from the immutable 19/19 final artifact, then run
// exactly the five cells whose old receipts lack causal ControlNet evidence.  The source OpenPose
// receipts are intentionally quarantined rather than rewritten: the replacement campaign owns new
// input/output hashes and a new receipt at its own immutable head.
export async function selectOpenPoseRecovery(prefixCandidates, currentProfile) {
  validateProfile(currentProfile);
  const root = path.resolve(prefixCandidates);
  const entries = await readdir(root, { withFileTypes: true });
  if (entries.length !== 1 || !entries[0].isDirectory() || entries[0].isSymbolicLink()) {
    fail("expected exactly one exact 19-PASS OpenPose recovery artifact candidate");
  }
  const candidate = path.join(root, entries[0].name);
  const metadata = JSON.parse(await readFile(path.join(candidate, "artifact-metadata.json"), "utf8"));
  validateOpenPoseRecoveryMetadata(metadata);
  const evidenceRoot = path.join(candidate, "evidence");
  const files = await ordinaryTreeFiles(evidenceRoot);
  const allowedTop = new Set([
    "_imported-prefix", `14-${EXPECTED_CELLS[13]}`,
    "cache-preflight-initial.json", "cache-preflight.json", "campaign-summary.json",
  ]);
  if (files.some((file) => !allowedTop.has(path.relative(evidenceRoot, file).split(path.sep)[0]))) {
    fail("OpenPose recovery source artifact contains evidence outside the exact audited campaign");
  }
  const summary = JSON.parse(await readFile(path.join(evidenceRoot, "campaign-summary.json"), "utf8"));
  validateOpenPoseRecoverySummary(summary, currentProfile, metadata);

  const receipts = [];
  const compatibility = [];
  for (const ordinal of OPENPOSE_RECOVERY_IMPORTED_ORDINALS) {
    const index = ordinal - 1;
    const cell = { ...currentProfile.cells[index], ordinal };
    const ordinalName = `${String(ordinal).padStart(2, "0")}-${cell.id}`;
    const cellDir = path.join(evidenceRoot, ...(ordinal === 14 ? [ordinalName] : ["_imported-prefix", ordinalName]));
    const receipt = JSON.parse(await readFile(path.join(cellDir, "receipt.json"), "utf8"));
    const provenance = ordinal <= IMPORTED_PREFIX_CELLS
      ? { headSha: LEGACY_SCENEWORKS_HEAD, runId: "32570707303", runAttempt: "1" }
      : ordinal <= RECOVERY_IMPORTED_PREFIX_CELLS
        ? { headSha: RECOVERY_SCENEWORKS_HEAD, runId: RECOVERY_RUN_ID, runAttempt: RECOVERY_RUN_ATTEMPT }
        : ordinal <= 13
          ? { headSha: SPARSE_RECOVERY_SCENEWORKS_HEAD, runId: SPARSE_RECOVERY_RUN_ID, runAttempt: SPARSE_RECOVERY_RUN_ATTEMPT }
          : metadata;
    validateRecoveryReceiptProvenance(receipt, cell, provenance, `OpenPose recovery imported receipt ${ordinalName}`);
    if (ordinal <= IMPORTED_PREFIX_CELLS) {
      validateTrustedLegacyImportedReceiptDocument(receipt);
      const legacyProfile = legacyPrefixProfile(currentProfile);
      validateReceiptInternal(receipt, { ...legacyProfile.cells[index], ordinal }, legacyProfile, null, TRUSTED_LEGACY_IMPORT_CONTEXT);
    } else {
      validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt);
      validateReceipt(receipt, cell, currentProfile);
    }
    const rehashed = await evidenceFiles(cellDir);
    for (const field of ["inputs", "outputs", "logs"]) {
      if (JSON.stringify(rehashed[field]) !== JSON.stringify(receipt[field])) {
        fail(`OpenPose recovery imported receipt ${ordinalName} failed ${field} rehash`);
      }
    }
    if (receipt.status !== "passed") fail(`OpenPose recovery imported receipt ${ordinalName} is not PASS`);
    compatibility.push({
      ordinal, cellId: cell.id, cellSemanticsSha256: canonicalSha256(currentProfile.cells[index]),
    });
    receipts.push({ cell: currentProfile.cells[index], ordinalName, receipt, root: cellDir });
  }
  return { candidate, evidenceRoot, metadata, receipts, compatibility, sourceLineage: summary.lineage };
}

export async function importOpenPoseRecovery(prefix, output) {
  const destination = path.join(output, "_imported-prefix");
  await mkdir(destination);
  for (const receipt of prefix.receipts) {
    await cp(receipt.root, path.join(destination, receipt.ordinalName), {
      recursive: true, errorOnExist: true, force: false, dereference: false,
    });
  }
  await ordinaryTreeFiles(destination);
  for (const { ordinalName, receipt } of prefix.receipts) {
    const rehashed = await evidenceFiles(path.join(destination, ordinalName));
    for (const field of ["inputs", "outputs", "logs"]) {
      if (JSON.stringify(rehashed[field]) !== JSON.stringify(receipt[field])) {
        fail(`copied OpenPose recovery ${ordinalName} failed ${field} rehash`);
      }
    }
  }
  const lineage = {
    kind: "terminal-openpose-counterfactual-recovery",
    sourceArtifactId: prefix.metadata.artifactId,
    sourceArtifactName: prefix.metadata.artifactName,
    sourceArtifactSize: prefix.metadata.artifactSize,
    sourceArtifactDigest: prefix.metadata.artifactDigest,
    sourceRunId: prefix.metadata.runId,
    sourceRunAttempt: prefix.metadata.runAttempt,
    sourceHeadSha: prefix.metadata.headSha,
    sourceInferenceSha: prefix.metadata.inferenceSha,
    sourceProfile: prefix.metadata.profile,
    sourceCellSemanticsSha256: prefix.metadata.cellSemanticsSha256,
    targetCellSemanticsSha256: EXPECTED_CELL_SEMANTICS_SHA256,
    sourceArtifactSemanticsSha256: prefix.metadata.artifactSemanticsSha256,
    importedOrdinals: OPENPOSE_RECOVERY_IMPORTED_ORDINALS,
    executionOrdinals: OPENPOSE_RECOVERY_EXECUTION_ORDINALS,
    compatibility: prefix.compatibility,
    priorLineage: prefix.sourceLineage,
  };
  await writeFile(path.join(destination, "lineage.json"), `${JSON.stringify(lineage, null, 2)}\n`, {
    encoding: "utf8", flag: "wx",
  });
  return {
    lineage,
    outcomes: prefix.receipts.map(({ cell, ordinalName }) => ({
      id: cell.id, status: "passed", receipt: `_imported-prefix/${ordinalName}/receipt.json`,
      error: null, emergencyReceiptError: null, source: "imported-openpose-recovery",
    })),
  };
}

async function writeArtifactRequest({ id, artifact, expectedFiles, scratch, phase }) {
  const requestDir = path.join(scratch, `${phase}-requests`);
  await mkdir(requestDir, { recursive: true });
  const requestPath = path.join(requestDir, `${id}.json`);
  await writeFile(requestPath, `${JSON.stringify({
    id, repository: artifact.repository, revision: artifact.revision,
    subdirectory: artifact.subdirectory, allowPatterns: artifact.allowPatterns,
    expectedFiles,
  }, null, 2)}\n`, "utf8");
  return requestPath;
}

function parseProvisionerOutput(raw, id, phase) {
  const lines = raw.trim().split(/\r?\n/);
  if (!lines.at(-1)) fail(`${phase} produced no structured result for ${id}`);
  try {
    return JSON.parse(lines.at(-1));
  } catch (error) {
    fail(`${phase} produced invalid structured result for ${id}: ${errorText(error)}`);
  }
}

async function validateProvisionResult({
  result, id, artifact, cacheRoot, phase, allowIncomplete = false, staged = false,
}) {
  exactKeys(
    result,
    [
      "id", "cacheRoot", "snapshotRoot", "selectedRoot", "complete", "missingFiles",
      "matchedFiles", "selectedFiles", "reusedFiles", "downloadedFiles",
      ...(staged ? ["sourceCacheRoot"] : []),
    ],
    `${phase} result for ${id}`,
  );
  const expected = cachedArtifactRoots(cacheRoot, artifact);
  const exactRoot = async (candidate, wanted) => {
    try {
      return comparable(await realpath(candidate)) === comparable(wanted);
    } catch (error) {
      if (error?.code !== "ENOENT" || !allowIncomplete) throw error;
      return comparable(path.resolve(candidate)) === comparable(wanted);
    }
  };
  if (result.id !== id || comparable(await realpath(result.cacheRoot)) !== comparable(cacheRoot)
    || !await exactRoot(result.snapshotRoot, expected.snapshotRoot)
    || !await exactRoot(result.selectedRoot, expected.selectedRoot)
    || (staged && typeof result.sourceCacheRoot !== "string")) {
    fail(`${phase} result drifted from exact authority ${id}`);
  }
  if (typeof result.complete !== "boolean" || !Array.isArray(result.missingFiles)
    || result.missingFiles.some((file) => typeof file !== "string" || !file)
    || result.complete !== (result.missingFiles.length === 0)
    || (!allowIncomplete && !result.complete)
    || !Array.isArray(result.matchedFiles) || (!allowIncomplete && result.matchedFiles.length === 0)
    || result.matchedFiles.some((file) => typeof file !== "string" || !file)
    || !Array.isArray(result.selectedFiles)
    || result.selectedFiles.some((file) => typeof file !== "string" || !file)
    || !Array.isArray(result.reusedFiles) || !Array.isArray(result.downloadedFiles)) {
    fail(`${phase} result has incomplete exact-file evidence for ${id}`);
  }
  const validateFiles = (files, keys, label) => {
    for (const file of files) {
      exactKeys(file, keys, `${phase} ${label} for ${id}`);
      if (typeof file.path !== "string" || !file.path
        || path.isAbsolute(file.path) || file.path.split(/[\\/]/).includes("..")
        || !Number.isSafeInteger(file.bytes) || file.bytes < 1
        || !/^[0-9a-f]{64}$/.test(file.sha256)) {
        fail(`${phase} ${label} has an invalid confined byte identity for ${id}`);
      }
    }
  };
  validateFiles(result.reusedFiles, ["path", "bytes", "sha256"], "reused file");
  validateFiles(
    result.downloadedFiles,
    ["path", "bytes", "sha256", "lfsSha256", "commitSha"],
    "downloaded file",
  );
  if (result.downloadedFiles.some((file) => !/^[0-9a-f]{64}$/.test(file.lfsSha256)
    || !SHA40.test(file.commitSha) || file.sha256 !== file.lfsSha256)
    || new Set(result.matchedFiles).size !== result.matchedFiles.length
    || new Set(result.selectedFiles).size !== result.selectedFiles.length
    || new Set(result.missingFiles).size !== result.missingFiles.length
    || JSON.stringify([...result.matchedFiles].sort()) !== JSON.stringify([
      ...result.reusedFiles.map((file) => file.path),
      ...result.downloadedFiles.map((file) => file.path),
    ].sort())) {
    fail(`${phase} file partition drifted for ${id}`);
  }
  return result;
}

async function auditArtifact({ id, artifact, expectedFiles, scratch, python, cacheRoot }) {
  const requestPath = await writeArtifactRequest({
    id, artifact, expectedFiles, scratch, phase: "audit",
  });
  const raw = run(python, [
    "scripts/provision-epic-20738-terminal-artifact.py",
    "--request", requestPath,
    "--cache-root", cacheRoot,
    "--audit",
  ], {
    env: {
      ...process.env,
      HF_HUB_OFFLINE: "1",
      TRANSFORMERS_OFFLINE: "1",
      HF_HUB_DISABLE_IMPLICIT_TOKEN: "1",
      HF_HUB_DISABLE_PROGRESS_BARS: "1",
      HF_HUB_DISABLE_TELEMETRY: "1",
      HF_HUB_VERBOSITY: "error",
    },
    maxBuffer: 8 * 1024 * 1024,
  });
  const result = await validateProvisionResult({
    result: parseProvisionerOutput(raw, id, "source cache census"),
    id, artifact, cacheRoot, phase: "source cache census", allowIncomplete: true,
  });
  return { ...artifact, ...result };
}

async function stageArtifact({
  id, artifact, expectedFiles, scratch, python, cacheRoot, stagingRoot, missingStore,
}) {
  const requestPath = await writeArtifactRequest({
    id, artifact, expectedFiles, scratch, phase: "stage",
  });
  const raw = run(python, [
    "scripts/provision-epic-20738-terminal-artifact.py",
    "--request", requestPath,
    "--cache-root", cacheRoot,
    "--stage-root", stagingRoot,
    ...(missingStore ? ["--missing-store", missingStore] : []),
  ], {
    env: {
      ...process.env,
      HF_HUB_OFFLINE: "1",
      TRANSFORMERS_OFFLINE: "1",
      HF_HUB_DISABLE_IMPLICIT_TOKEN: "1",
      HF_HUB_DISABLE_PROGRESS_BARS: "1",
      HF_HUB_DISABLE_TELEMETRY: "1",
      HF_HUB_VERBOSITY: "error",
    },
    maxBuffer: 8 * 1024 * 1024,
  });
  const result = await validateProvisionResult({
    result: parseProvisionerOutput(raw, id, "campaign staging"),
    id, artifact, cacheRoot: stagingRoot, phase: "campaign staging", staged: true,
  });
  if (comparable(await realpath(result.sourceCacheRoot)) !== comparable(cacheRoot)) {
    fail(`campaign staging result drifted from trusted source cache for ${id}`);
  }
  return result;
}

async function downloadReviewedMissing({
  id, artifact, expectedFiles, missingFiles, scratch, python, cacheRoot, missingStore,
}) {
  const requestPath = await writeArtifactRequest({
    id, artifact, expectedFiles, scratch, phase: "download-missing",
  });
  const raw = run(python, [
    "scripts/provision-epic-20738-terminal-artifact.py",
    "--request", requestPath,
    "--cache-root", cacheRoot,
    "--missing-store", missingStore,
    "--download-reviewed-missing",
  ], {
    env: {
      ...process.env,
      HF_HUB_OFFLINE: "0",
      TRANSFORMERS_OFFLINE: "0",
      HF_HUB_DISABLE_IMPLICIT_TOKEN: "1",
      HF_HUB_DISABLE_PROGRESS_BARS: "1",
      HF_HUB_DISABLE_TELEMETRY: "1",
      HF_HUB_VERBOSITY: "error",
    },
    maxBuffer: 8 * 1024 * 1024,
  });
  const result = parseProvisionerOutput(raw, id, "reviewed missing-file fill");
  exactKeys(result, ["id", "storeRoot", "downloadedFiles"], `missing-file fill for ${id}`);
  if (result.id !== id || comparable(await realpath(result.storeRoot)) !== comparable(missingStore)
    || result.downloadedFiles.length !== missingFiles.length) {
    fail(`reviewed missing-file fill identity drifted for ${id}`);
  }
  const prefix = artifact.subdirectory === "." ? "" : `${artifact.subdirectory}/`;
  for (const [index, file] of result.downloadedFiles.entries()) {
    exactKeys(
      file,
      ["path", "bytes", "sha256", "lfsSha256", "commitSha"],
      `reviewed missing-file fill for ${id}`,
    );
    const expectedPath = prefix && missingFiles[index].startsWith(prefix)
      ? missingFiles[index].slice(prefix.length) : missingFiles[index];
    if (file.path !== expectedPath || file.commitSha !== artifact.revision
      || !Number.isSafeInteger(file.bytes) || file.bytes < 1
      || !/^[0-9a-f]{64}$/.test(file.sha256) || file.sha256 !== file.lfsSha256) {
      fail("reviewed missing-file fill did not prove exact commit/path/size/LFS identity");
    }
  }
  return result;
}

async function provisionArtifact({ id, artifact, expectedFiles, scratch, python, cacheRoot }) {
  const requestPath = await writeArtifactRequest({
    id, artifact, expectedFiles, scratch, phase: "offline",
  });
  const raw = run(python, [
    "scripts/provision-epic-20738-terminal-artifact.py",
    "--request", requestPath,
    "--cache-root", cacheRoot,
  ], {
    env: {
      ...process.env,
      HF_HUB_OFFLINE: "1",
      TRANSFORMERS_OFFLINE: "1",
      HF_HUB_DISABLE_IMPLICIT_TOKEN: "1",
      HF_HUB_DISABLE_PROGRESS_BARS: "1",
      HF_HUB_DISABLE_TELEMETRY: "1",
      HF_HUB_VERBOSITY: "error",
    },
    maxBuffer: 8 * 1024 * 1024,
  });
  const result = await validateProvisionResult({
    result: parseProvisionerOutput(raw, id, "final offline staging validation"),
    id, artifact, cacheRoot, phase: "final offline staging validation",
  });
  const selectedRoot = await realpath(result.selectedRoot);
  const selectedFiles = await listCachedArtifactFiles(selectedRoot, cacheRoot);
  if (JSON.stringify(selectedFiles) !== JSON.stringify(result.selectedFiles)) {
    fail(`cache-only provision result selected-file inventory drifted for ${id}`);
  }
  const inventory = await hashArtifactInventory(
    selectedRoot,
    { includeFiles: result.matchedFiles, trustedRoot: cacheRoot },
  );
  return {
    id,
    ...artifact,
    snapshotRoot: result.snapshotRoot,
    selectedRoot,
    matchedFiles: result.matchedFiles,
    selectedFiles,
    inventory,
    cacheEvidence: {
      reusedFiles: result.reusedFiles,
      downloadedFiles: result.downloadedFiles,
    },
  };
}

export async function installSidecarObstructions(sharedArtifacts) {
  const roots = new Map();
  for (const artifact of sharedArtifacts.values()) {
    const componentRoots = new Set([artifact.selectedRoot]);
    for (const file of artifact.matchedFiles) {
      const [component] = file.split("/");
      if (file.includes("/")) componentRoots.add(path.join(artifact.selectedRoot, component));
    }
    artifact.sidecarObstructions = [];
    for (const root of componentRoots) {
      const key = comparable(root);
      const entry = roots.get(key) ?? { root, artifactIds: [] };
      entry.artifactIds.push(artifact.id);
      roots.set(key, entry);
    }
  }
  const records = [];
  for (const entry of roots.values()) {
    const obstruction = path.join(entry.root, ".candle-device-format-v1");
    await writeFile(
      obstruction,
      "SceneWorks terminal harness: model-adjacent Candle sidecars are forbidden.\n",
      { encoding: "utf8", flag: "wx" },
    );
    const metadata = await lstat(obstruction);
    if (!metadata.isFile() || metadata.isSymbolicLink()) {
      fail(`Candle sidecar obstruction is not an ordinary regular file: ${obstruction}`);
    }
    const record = {
      artifactIds: entry.artifactIds.sort(),
      root: entry.root,
      path: ".candle-device-format-v1",
      bytes: metadata.size,
      sha256: await sha256File(obstruction),
    };
    records.push(record);
    for (const artifactId of record.artifactIds) {
      sharedArtifacts.get(artifactId).sidecarObstructions.push(record);
    }
  }
  return records.sort((left, right) => left.root.localeCompare(right.root));
}

function rustCanonicalPath(resolved, platform = process.platform) {
  if (platform !== "win32" || resolved.startsWith("\\\\?\\")) return resolved;
  if (resolved.startsWith("\\\\")) return `\\\\?\\UNC\\${resolved.slice(2)}`;
  return `\\\\?\\${resolved}`;
}

export async function expectedB646DerivedNamespace(artifact, derivedSidecarRoot) {
  if (!artifact.matchedFiles.includes("transformer/model.safetensors")) {
    fail(`FLUX artifact lacks the exact packed transformer component: ${artifact.id}`);
  }
  const component = await realpath(path.join(artifact.selectedRoot, "transformer"));
  const digest = createHash("sha256")
    .update("sceneworks-candle-device-format-component-v1\0")
    .update(rustCanonicalPath(component))
    .digest("hex");
  return path.join(derivedSidecarRoot, "candle-device-format-v1", digest);
}

export async function inspectDerivedSidecarRoot(
  derivedSidecarRoot, expectedNamespace = null,
  { materializeExactEmpty = false, allowExactEmpty = false } = {},
) {
  let rootEntries = await readdir(derivedSidecarRoot, { withFileTypes: true });
  if (!expectedNamespace) {
    if (rootEntries.length) fail("non-FLUX derived sidecar root must be exactly empty");
    return { rootInventory: await directoryInventory(derivedSidecarRoot), namespaces: [] };
  }
  const expectedVersionRoot = path.join(derivedSidecarRoot, "candle-device-format-v1");
  if (allowExactEmpty && rootEntries.length === 0) {
    return { rootInventory: await directoryInventory(derivedSidecarRoot), namespaces: [] };
  }
  if (materializeExactEmpty && rootEntries.length === 0) {
    if (comparable(path.dirname(expectedNamespace)) !== comparable(expectedVersionRoot)
      || !/^[0-9a-f]{64}$/.test(path.basename(expectedNamespace))) {
      fail("empty FLUX sidecar namespace target is not the exact b646 cache path");
    }
    await mkdir(expectedNamespace, { recursive: true });
    rootEntries = await readdir(derivedSidecarRoot, { withFileTypes: true });
  }
  if (rootEntries.length !== 1 || rootEntries[0].name !== "candle-device-format-v1"
    || !rootEntries[0].isDirectory()) {
    fail("FLUX derived sidecar root has stray files, directories, or versions");
  }
  const versionRoot = expectedVersionRoot;
  const versionMetadata = await lstat(versionRoot);
  if (versionMetadata.isSymbolicLink()) fail("derived sidecar version root is a reparse point");
  const namespaceEntries = await readdir(versionRoot, { withFileTypes: true });
  const expectedName = path.basename(expectedNamespace);
  if (namespaceEntries.length !== 1 || namespaceEntries[0].name !== expectedName
    || !namespaceEntries[0].isDirectory()) {
    fail("FLUX derived sidecar root does not contain one exact b646 namespace");
  }
  const namespace = path.join(versionRoot, expectedName);
  if (comparable(namespace) !== comparable(expectedNamespace)
    || (await lstat(namespace)).isSymbolicLink()) {
    fail("FLUX derived sidecar namespace identity is not canonical");
  }
  for (const entry of await readdir(namespace, { withFileTypes: true })) {
    const candidate = path.join(namespace, entry.name);
    const metadata = await lstat(candidate);
    if (!entry.isFile() || metadata.isSymbolicLink()) {
      fail(`FLUX derived sidecar namespace contains a non-regular entry: ${candidate}`);
    }
  }
  const inventory = await directoryInventory(namespace);
  const rootInventory = await directoryInventory(derivedSidecarRoot);
  if (rootInventory.files !== inventory.files || rootInventory.bytes !== inventory.bytes) {
    fail("derived sidecar global inventory does not equal its one canonical namespace");
  }
  return { rootInventory, namespaces: [{ path: namespace, inventory }] };
}

async function diskFreeBytes(directory) {
  const filesystem = await statfs(directory);
  const bytes = Number(filesystem.bavail) * Number(filesystem.bsize);
  if (!Number.isSafeInteger(bytes) || bytes < 0) fail("filesystem free-byte count is invalid");
  return bytes;
}

export async function verifyArtifactUnchanged(artifact, cacheRoot) {
  const selectedFiles = await listCachedArtifactFiles(artifact.selectedRoot, cacheRoot);
  const obstructionFiles = artifact.sidecarObstructions.map((entry) => path.relative(
    artifact.selectedRoot,
    path.join(entry.root, entry.path),
  ).split(path.sep).join("/"));
  const expectedSelectedFiles = [...artifact.selectedFiles, ...obstructionFiles].sort();
  if (JSON.stringify(selectedFiles) !== JSON.stringify(expectedSelectedFiles)) {
    fail(`staged authority selected-file set mutated for ${artifact.id}`);
  }
  for (const entry of artifact.sidecarObstructions) {
    const obstruction = path.join(entry.root, entry.path);
    const obstructionMetadata = await lstat(obstruction);
    if (!obstructionMetadata.isFile() || obstructionMetadata.isSymbolicLink()
      || obstructionMetadata.size !== entry.bytes
      || await sha256File(obstruction) !== entry.sha256) {
      fail(`Candle sidecar obstruction mutated for ${artifact.id}`);
    }
  }
  const inventory = await hashArtifactInventory(artifact.selectedRoot, {
    includeFiles: artifact.matchedFiles,
    trustedRoot: cacheRoot,
  });
  if (inventory.files !== artifact.inventory.files || inventory.bytes !== artifact.inventory.bytes
    || inventory.sha256 !== artifact.inventory.sha256) {
    fail(`staged authority bytes mutated for ${artifact.id}`);
  }
}

function cargoCellArgs() {
  return [
    "test", "--locked", "--release", "-p", "sceneworks-worker", "--features", "backend-candle",
    "epic_20738_terminal_cuda_cell", "--", "--ignored", "--nocapture", "--test-threads=1",
  ];
}

async function executeCell({
  sceneworks, cellFile, cellDir, logFile, samples, runtimeScratch, derivedSidecarRoot,
}) {
  const writable = {
    temp: path.join(runtimeScratch, "writable", "temp"),
    hf: path.join(runtimeScratch, "writable", "huggingface"),
    transformers: path.join(runtimeScratch, "writable", "transformers"),
    xdg: path.join(runtimeScratch, "writable", "xdg"),
    torch: path.join(runtimeScratch, "writable", "torch"),
  };
  await Promise.all(Object.values(writable).map((directory) => mkdir(directory, { recursive: true })));
  await new Promise((resolve, reject) => {
    const log = createWriteStream(logFile, { flags: "wx" });
    const child = spawn("cargo", cargoCellArgs(), {
      cwd: sceneworks,
      windowsHide: true,
      env: {
        ...process.env,
        SCENEWORKS_EPIC_20738_CELL_FILE: cellFile,
        SCENEWORKS_EPIC_20738_OUTPUT_DIR: cellDir,
        SCENEWORKS_ENABLE_EPIC_20738_TERMINAL_CUDA: "1",
        TEMP: writable.temp,
        TMP: writable.temp,
        TMPDIR: writable.temp,
        HF_HOME: writable.hf,
        HUGGINGFACE_HUB_CACHE: writable.hf,
        HF_HUB_CACHE: writable.hf,
        TRANSFORMERS_CACHE: writable.transformers,
        XDG_CACHE_HOME: writable.xdg,
        TORCH_HOME: writable.torch,
        SCENEWORKS_CANDLE_DEVICE_CACHE_DIR: derivedSidecarRoot,
        HF_HUB_OFFLINE: "1",
        TRANSFORMERS_OFFLINE: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    child.stdout.pipe(log, { end: false });
    child.stderr.pipe(log, { end: false });
    let sampling = false;
    const timer = setInterval(() => {
      if (sampling) return;
      sampling = true;
      try { samples.push(...sampleGpu()); } catch (error) {
        samples.push({ timestamp: new Date().toISOString(), raw: `nvidia-smi sampling failed: ${error.message}` });
      } finally { sampling = false; }
    }, 1000);
    child.on("error", (error) => {
      clearInterval(timer);
      log.end(() => reject(error));
    });
    child.on("close", (code, signal) => {
      clearInterval(timer);
      log.end(() => {
        if (code === 0) resolve();
        else reject(new Error(`cell cargo process failed with code ${code}, signal ${signal ?? "none"}`));
      });
    });
  });
}

function errorText(error) {
  return error?.stack ?? error?.message ?? String(error);
}

function recordError(errors, label, error) {
  const message = `${label}: ${errorText(error)}`;
  errors.push(message);
  return message;
}

function sampleGpuBestEffort(samples, errors, label) {
  try {
    samples.push(...sampleGpu());
  } catch (error) {
    const message = recordError(errors, label, error);
    samples.push({ timestamp: new Date().toISOString(), raw: message });
  }
}

async function invokeFault(fault, stage, index) {
  if (fault) await fault(stage, index);
}

async function refreshEvidence(receipt, cellDir, errors, { fault, index } = {}) {
  let primaryFailure = null;
  try {
    await invokeFault(fault, "evidenceHash", index);
    const evidence = await evidenceFiles(cellDir);
    receipt.inputs = evidence.inputs;
    receipt.outputs = evidence.outputs;
    receipt.logs = evidence.logs;
    if (!receipt.logs.length) fail("no controller or runtime log was available to hash");
  } catch (error) {
    const message = recordError(errors, "evidence hashing failed", error);
    primaryFailure = message;
    receipt.inputs = [];
    receipt.outputs = [];
    const fallbackLog = path.join(cellDir, "evidence-fallback.log");
    await writeFile(fallbackLog, `${new Date().toISOString()} ${message}\n`, "utf8");
    await invokeFault(fault, "evidenceRehash", index);
    const fallback = await hashedFiles(cellDir, { exclude: new Set(["receipt.json"]) });
    receipt.logs = fallback.filter((file) => file.path.endsWith(".log"));
  }
  return primaryFailure;
}

async function writePrimaryReceipt({
  file, receipt, cell, profile, fault, index, lifecycleContext,
}) {
  await invokeFault(fault, "semanticValidation", index);
  validateReceipt(receipt, cell, profile, lifecycleContext);
  await invokeFault(fault, "schemaValidation", index);
  validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt);
  await invokeFault(fault, "receiptWrite", index);
  await writeJsonAtomically(file, receipt);
  await invokeFault(fault, "receiptStat", index);
  const metadata = await stat(file);
  if (!metadata.isFile() || metadata.size < 1) fail("primary receipt was not durably materialized");
  await invokeFault(fault, "receiptHash", index);
  if (!/^[0-9a-f]{64}$/.test(await sha256File(file))) {
    fail("primary receipt hash finalization failed");
  }
}

function emptyAuthorityLifecycle(cell, index) {
  return {
    ordinal: index + 1,
    cellId: cell.id,
    staged: [],
    activeArtifactIds: [...cell.artifactIds],
    providerExecution: "not-attempted",
    requestMemoryStrategy: null,
    verifiedBefore: false,
    verifiedAfter: false,
    derivedAfter: [],
    released: [],
    diskProbes: [],
  };
}

async function writeEmergencyReceipt({
  profile, cell, index, ordinalName, output, cacheRoot, repositories, execution, gpuIdentity,
  systemMemory, error, cleanup, authorityLifecycle, lifecycleContext,
}) {
  // This path deliberately bypasses all injected/primary finalization operations. If the primary
  // log, evidence hash, semantic/schema check, or atomic writer fails, a fresh controller-owned
  // directory under the already-confined output root gets a separately constructed receipt.
  const cellDir = path.join(output, "_emergency", ordinalName);
  await mkdir(cellDir, { recursive: true });
  const startedAt = new Date().toISOString();
  const message = `unhandled cell setup/finalization failure: ${errorText(error)}`;
  await writeFile(path.join(cellDir, "controller.log"), `${startedAt} ${message}\n`, "utf8");
  const artifacts = cell.artifactIds.map((id) => pendingArtifact(
    id, profile.artifacts[id], cacheRoot,
  ));
  const receipt = receiptSkeleton({
    cell, ordinal: index + 1, repositories, artifacts, execution, gpuIdentity, systemMemory,
    startedAt,
  });
  receipt.error = message;
  receipt.authorityLifecycle = authorityLifecycle ?? emptyAuthorityLifecycle(cell, index);
  receipt.cleanup = cleanup ?? { attempted: false, completed: true, error: null };
  receipt.hardware.rawVramSamples = gpuIdentity.length
    ? [...gpuIdentity]
    : [{ timestamp: startedAt, raw: message }];
  const evidence = await evidenceFiles(cellDir);
  receipt.logs = evidence.logs;
  receipt.completedAt = new Date().toISOString();
  validateReceipt(receipt, cell, profile, lifecycleContext);
  validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt);
  await writeJsonAtomically(path.join(cellDir, "receipt.json"), receipt);
  return `_emergency/${ordinalName}/receipt.json`;
}

async function writeLastResortEmergencyReceipt({
  profile, cell, index, ordinalName, output, cacheRoot, repositories, execution, gpuIdentity,
  systemMemory, errors, cleanup, authorityLifecycle, lifecycleContext,
}) {
  // A second implementation intentionally avoids evidenceFiles(), writeJsonAtomically(), and every
  // primary/injected callback. It hashes its one log directly and publishes under a different name.
  const cellDir = path.join(output, "_emergency", ordinalName);
  await mkdir(cellDir, { recursive: true });
  const startedAt = new Date().toISOString();
  const logFile = path.join(cellDir, "controller-last-resort.log");
  await writeFile(logFile, `${startedAt} ${errors.join("\n")}\n`, "utf8");
  const metadata = await stat(logFile);
  const receipt = receiptSkeleton({
    cell,
    ordinal: index + 1,
    repositories,
    artifacts: cell.artifactIds.map((id) => pendingArtifact(
      id, profile.artifacts[id], cacheRoot,
    )),
    execution,
    gpuIdentity,
    systemMemory,
    startedAt,
  });
  receipt.error = errors.join("\n");
  receipt.authorityLifecycle = authorityLifecycle ?? emptyAuthorityLifecycle(cell, index);
  receipt.cleanup = cleanup;
  receipt.hardware.rawVramSamples = gpuIdentity.length
    ? [...gpuIdentity]
    : [{ timestamp: startedAt, raw: receipt.error }];
  receipt.logs = [{
    path: "controller-last-resort.log", bytes: metadata.size, sha256: await sha256File(logFile),
  }];
  receipt.completedAt = new Date().toISOString();
  validateReceipt(receipt, cell, profile, lifecycleContext);
  validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt);
  const finalPath = path.join(cellDir, "receipt-last-resort.json");
  const temporary = `${finalPath}.tmp-${process.pid}-${randomUUID()}`;
  try {
    await writeFile(temporary, `${JSON.stringify(receipt, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await rename(temporary, finalPath);
  } finally {
    await rm(temporary, { force: true });
  }
  return `_emergency/${ordinalName}/receipt-last-resort.json`;
}

async function writeSummaryDurably({ output, summary, fault }) {
  const primary = path.join(output, "campaign-summary.json");
  try {
    await invokeFault(fault, "summaryWrite", null);
    await writeJsonAtomically(primary, summary);
    return "campaign-summary.json";
  } catch (error) {
    summary.campaignErrors.push(`primary campaign summary write failed: ${errorText(error)}`);
    const fallbackDir = path.join(output, "_emergency");
    await mkdir(fallbackDir, { recursive: true });
    const fallback = path.join(fallbackDir, "campaign-summary-fallback.json");
    const temporary = `${fallback}.tmp-${process.pid}-${randomUUID()}`;
    try {
      // Independent path and primitives: neither the primary summary hook nor writer is reused.
      await writeFile(temporary, `${JSON.stringify(summary, null, 2)}\n`, {
        encoding: "utf8", flag: "wx",
      });
      await rename(temporary, fallback);
      return "_emergency/campaign-summary-fallback.json";
    } catch (fallbackError) {
      const failureLog = path.join(fallbackDir, "campaign-summary-fallback-error.log");
      await writeFile(
        failureLog,
        `${new Date().toISOString()} ${errorText(fallbackError)}\n${JSON.stringify(summary)}\n`,
        "utf8",
      );
      throw new Error(`primary and fallback campaign summary writes failed; evidence: ${failureLog}`, {
        cause: fallbackError,
      });
    } finally {
      await rm(temporary, { force: true });
    }
  }
}

function cachePreflightDocument({
  evidencePhase, status, error, downloadEvidenceSha256, remainingArtifactIds, guard, stagingRoot,
  derivedSidecarRoot, frozenMissing, sidecarObstructions, cacheProvisioning,
  derivedSidecarLifecycle, missingStore, lifetimePlan, diskPlan, missingDownloads,
  networkOfflineEstablished,
}) {
  return {
    schemaVersion: 1,
    profile: PROFILE_NAME,
    evidencePhase,
    status,
    error,
    downloadEvidenceSha256,
    expectedArtifactIds: remainingArtifactIds,
    sourceCacheRoot: guard.cacheRoot,
    campaignStagingRoot: stagingRoot,
    derivedSidecarRoot,
    missingFileStore: missingStore,
    frozenMissingFiles: frozenMissing,
    sidecarObstructions,
    reusedFiles: cacheProvisioning.sourceCensus.flatMap((artifact) => artifact.reusedFiles.map(
      (file) => ({ artifactId: artifact.id, ...file }),
    )),
    downloadedFiles: missingDownloads.flatMap((download) => download.downloadedFiles.map((file) => ({
      artifactId: download.id, ...file,
    }))),
    networkDownloadCount: missingDownloads.reduce(
      (count, download) => count + download.downloadedFiles.length, 0,
    ),
    phases: cacheProvisioning,
    lifetimePlan,
    diskPlan,
    derivedSidecarLifecycle,
    offlineBeforeCells: networkOfflineEstablished,
  };
}

export function validateCachePreflightEvidence(document, {
  remainingArtifactIds, artifactExpectedFiles, downloadEvidenceSha256, guard, stagingRoot,
  derivedSidecarRoot, missingStore, expectedNonModelPaths, profile, executionOrdinals = null,
}) {
  validateDocumentWithSchema(CACHE_PREFLIGHT_SCHEMA_PATH, document);
  const plannedOrdinals = executionOrdinals ?? document.diskPlan?.cells.map((cell) => cell.ordinal);
  const exactOpenPoseRecovery = isExactOrdinals(
    plannedOrdinals, OPENPOSE_RECOVERY_EXECUTION_ORDINALS,
  );
  if (document.diskPlan
    && !exactOpenPoseRecovery
    && document.diskPlan.preHydrationJitSourcePeakBytes
      !== PRE_HYDRATION_JIT_SOURCE_PEAK_BYTES) {
    fail("current cache preflight disk plan omitted or drifted from the pre-hydration JIT peak");
  }
  if (document.profile !== profile.profile
    || document.downloadEvidenceSha256 !== downloadEvidenceSha256
    || document.sourceCacheRoot !== guard.cacheRoot
    || document.campaignStagingRoot !== stagingRoot
    || document.derivedSidecarRoot !== derivedSidecarRoot
    || document.missingFileStore !== missingStore
    || JSON.stringify(document.expectedArtifactIds) !== JSON.stringify(remainingArtifactIds)) {
    fail("cache preflight identity/path binding drifted");
  }
  if (document.diskPlan
    && JSON.stringify(document.diskPlan.nonModelPaths) !== JSON.stringify(expectedNonModelPaths)) {
    fail("cache preflight non-model paths drifted from runtime-owned paths");
  }
  const ids = new Set(remainingArtifactIds);
  for (const phase of ["sourceCensus", "staging", "finalOffline"]) {
    const rows = document.phases[phase];
    if (new Set(rows.map((row) => row.id)).size !== rows.length
      || rows.some((row) => !ids.has(row.id)
        || JSON.stringify(row.expectedFiles) !== JSON.stringify(artifactExpectedFiles[row.id])
        || row.expectedFiles.some((file) => file.includes(".candle-device-format-v1")
          || file.endsWith(".incomplete")))) {
      fail(`cache preflight ${phase} authority census drifted`);
    }
  }
  for (const row of document.phases.sourceCensus) {
    const prefix = row.subdirectory === "." ? "" : `${row.subdirectory}/`;
    const present = row.reusedFiles.map((file) => file.path);
    const missing = row.missingFiles.map((file) => (
      prefix && file.startsWith(prefix) ? file.slice(prefix.length) : file
    ));
    if (row.complete !== (row.missingFiles.length === 0)
      || JSON.stringify([...present, ...missing].sort())
      !== JSON.stringify([...artifactExpectedFiles[row.id]].sort())) {
      fail(`cache preflight source census did not cover every expected file for ${row.id}`);
    }
  }
  const frozen = document.phases.sourceCensus.flatMap((row) => row.missingFiles.map((file) => ({
    artifactId: row.id, repository: row.repository, revision: row.revision, file,
  })));
  if (JSON.stringify(document.frozenMissingFiles) !== JSON.stringify(frozen)) {
    fail("cache preflight frozen missing-file census drifted");
  }
  if (new Set([CURRENT_DOWNLOAD_EVIDENCE_SHA256, LEGACY_DOWNLOAD_EVIDENCE_SHA256])
    .has(downloadEvidenceSha256) && profile.cells.length === 19) {
    const exactScope = new Map([
      [16, 199],
      [RECOVERY_REMAINING_AUTHORITIES, RECOVERY_REMAINING_FILES],
      [SPARSE_REMAINING_AUTHORITIES, SPARSE_REMAINING_FILES],
    ]);
    const expectedFiles = exactScope.get(remainingArtifactIds.length);
    if (expectedFiles !== undefined && remainingArtifactIds.reduce(
      (sum, id) => sum + artifactExpectedFiles[id].length, 0,
    ) !== expectedFiles) {
      fail(`continuation must bind the exact logical ${remainingArtifactIds.length}-authority/${expectedFiles}-file census`);
    }
  }
  for (const row of document.phases.staging) {
    if (JSON.stringify([
      ...row.reusedFiles.map((file) => file.path),
      ...row.downloadedFiles.map((file) => file.path),
    ].sort()) !== JSON.stringify([...artifactExpectedFiles[row.id]].sort())) {
      fail(`cache preflight staging did not copy the exact expected census for ${row.id}`);
    }
  }
  for (const row of document.phases.finalOffline) {
    if (row.inventory.files !== artifactExpectedFiles[row.id].length) {
      fail(`cache preflight final inventory count drifted for ${row.id}`);
    }
  }
  const sourceFiles = document.phases.sourceCensus.flatMap((artifact) => artifact.reusedFiles.map(
    (file) => ({ artifactId: artifact.id, ...file }),
  ));
  if (JSON.stringify(document.reusedFiles) !== JSON.stringify(sourceFiles)) {
    fail("cache preflight top-level hit/download partition drifted");
  }
  const reviewedPlan = reviewedMissingDownloadPlan({
    frozenMissing: frozen, profile, artifactExpectedFiles, downloadEvidenceSha256,
  });
  const expectedDownloads = reviewedPlan.flatMap(({ id, missingFiles }) => {
    const artifact = profile.artifacts[id];
    const prefix = artifact.subdirectory === "." ? "" : `${artifact.subdirectory}/`;
    return missingFiles.map((file) => ({
      artifactId: id,
      path: prefix && file.startsWith(prefix) ? file.slice(prefix.length) : file,
      commitSha: artifact.revision,
    }));
  });
  if (document.downloadedFiles.length !== expectedDownloads.length
    || document.downloadedFiles.some((file, index) => (
      file.artifactId !== expectedDownloads[index].artifactId
      || file.path !== expectedDownloads[index].path
      || file.commitSha !== expectedDownloads[index].commitSha
      || file.sha256 !== file.lfsSha256
    ))) fail("cache preflight contains an unreviewed model download");
  if (frozen.length !== document.downloadedFiles.length) {
    fail("cache preflight missing/download partition is incomplete");
  }
  if (document.networkDownloadCount !== document.downloadedFiles.length
    || document.networkDownloadCount !== frozen.length) {
    fail("cache preflight network count drifted from the frozen missing set");
  }
  if (!document.offlineBeforeCells && (document.phases.staging.length
    || document.phases.finalOffline.length || document.sidecarObstructions.length
    || document.derivedSidecarLifecycle.afterCells.length)) {
    fail("cache preflight records cell lifecycle evidence before network-offline establishment");
  }
  if (document.status === "passed") {
    const plannedExecutionOrdinals = exactExecutionOrdinals(
      profile,
      executionOrdinals ?? document.diskPlan?.cells.map((cell) => cell.ordinal),
      "cache preflight execution ordinals",
    );
    if (executionOrdinals && JSON.stringify(document.diskPlan.cells.map((cell) => cell.ordinal))
      !== JSON.stringify(plannedExecutionOrdinals)) {
      fail("cache preflight disk plan omitted explicit execution ordinals");
    }
    const plannedAudits = new Map(document.phases.sourceCensus.map((row) => [row.id, {
      ...row,
      downloadedFiles: document.downloadedFiles.filter((file) => file.artifactId === row.id),
    }]));
    const expectedLifetimePlan = authorityLifetimePlan(
      profile,
      plannedExecutionOrdinals,
      plannedAudits,
    );
    if (JSON.stringify(document.lifetimePlan) !== JSON.stringify(expectedLifetimePlan)) {
      fail("cache preflight authority lifetime plan drifted from the frozen census");
    }
    const reviewedPlan = reviewedRecoveryDiskPlan(plannedExecutionOrdinals);
    const expectedDiskPlan = estimateJitDiskPlan(
      expectedLifetimePlan,
      document.diskPlan.freeBytes,
      document.downloadedFiles.reduce((sum, file) => sum + file.bytes, 0),
      document.diskPlan.nonModelPaths,
      downloadEvidenceSha256 === LEGACY_DOWNLOAD_EVIDENCE_SHA256
        ? LEGACY_REVIEWED_ALL_AT_ONCE_SOURCE_BYTES : reviewedPlan.sourceBytes,
      reviewedPlan.jitPeakBytes,
      reviewedPlan.freeFloorBytes,
    );
    if (JSON.stringify(document.diskPlan) !== JSON.stringify(expectedDiskPlan)) {
      fail("cache preflight disk plan drifted from exact target-byte census");
    }
    if (JSON.stringify(document.phases.sourceCensus.map((row) => row.id))
      !== JSON.stringify(remainingArtifactIds)) {
      fail("passed cache preflight omitted sourceCensus authorities");
    }
    if (document.lifetimePlan.length !== remainingArtifactIds.length
      || JSON.stringify(document.lifetimePlan.map((row) => row.artifactId))
        !== JSON.stringify(remainingArtifactIds)) {
      fail("passed cache preflight lifetime plan omitted reviewed authorities");
    }
    const initial = document.derivedSidecarLifecycle.initial;
    if (!initial || initial.root !== derivedSidecarRoot || initial.files !== 0
      || initial.bytes !== 0 || initial.sha256 !== createHash("sha256").digest("hex")) {
      fail("derived Candle cache was not proven empty before cells");
    }
    if (!document.diskPlan.admitted
      || document.diskPlan.freeBytes < document.diskPlan.peakRequiredAdditionalBytes) {
      fail("passed cache preflight did not prove sufficient JIT disk capacity");
    }
    if (exactOpenPoseRecovery) {
      assertExactOpenPoseRecoveryPreflight({
        executionOrdinals: plannedExecutionOrdinals,
        remainingArtifactIds,
        artifactExpectedFiles,
        lifetimePlan: document.lifetimePlan,
        diskPlan: document.diskPlan,
      });
    }
    if (new Set([CURRENT_DOWNLOAD_EVIDENCE_SHA256, LEGACY_DOWNLOAD_EVIDENCE_SHA256])
      .has(downloadEvidenceSha256) && !exactOpenPoseRecovery) {
      const expectedRemainingIds = [...new Set(plannedExecutionOrdinals.flatMap(
        (ordinal) => profile.cells[ordinal - 1].artifactIds,
      ))];
      const expectedFileCount = expectedRemainingIds.reduce(
        (sum, id) => sum + artifactExpectedFiles[id].length, 0,
      );
      const startTen = JSON.stringify(plannedExecutionOrdinals)
        === JSON.stringify(Array.from({ length: 10 }, (_, index) => index + 10));
      const sparse = JSON.stringify(plannedExecutionOrdinals)
        === JSON.stringify(SPARSE_EXECUTION_ORDINALS);
      const pass18 = JSON.stringify(plannedExecutionOrdinals)
        === JSON.stringify(PASS18_EXECUTION_ORDINALS);
      if (JSON.stringify(remainingArtifactIds) !== JSON.stringify(expectedRemainingIds)
        || (startTen && (JSON.stringify(expectedRemainingIds)
          !== JSON.stringify(RECOVERY_REMAINING_ARTIFACT_IDS)
          || expectedFileCount !== RECOVERY_REMAINING_FILES))
        || (sparse && (JSON.stringify(expectedRemainingIds)
          !== JSON.stringify(SPARSE_REMAINING_ARTIFACT_IDS)
          || expectedFileCount !== SPARSE_REMAINING_FILES))
        || (pass18 && (JSON.stringify(expectedRemainingIds)
          !== JSON.stringify(PASS18_REMAINING_ARTIFACT_IDS)
          || expectedFileCount !== PASS18_REMAINING_FILES))) {
        fail("passed cache preflight execution/census drifted from the reviewed profile");
      }
      const legacyEvidence = downloadEvidenceSha256 === LEGACY_DOWNLOAD_EVIDENCE_SHA256;
      const expectedSourceBytes = sparse
        ? legacyEvidence ? LEGACY_SPARSE_REMAINING_SOURCE_BYTES : SPARSE_REMAINING_SOURCE_BYTES
        : pass18 ? PASS18_REMAINING_SOURCE_BYTES
        : startTen
          ? legacyEvidence ? LEGACY_RECOVERY_REMAINING_SOURCE_BYTES : RECOVERY_REMAINING_SOURCE_BYTES
          : legacyEvidence ? LEGACY_REVIEWED_ALL_AT_ONCE_SOURCE_BYTES : REVIEWED_ALL_AT_ONCE_SOURCE_BYTES;
      const expectedAllAtOnceSourceBytes = legacyEvidence
        ? LEGACY_REVIEWED_ALL_AT_ONCE_SOURCE_BYTES : REVIEWED_ALL_AT_ONCE_SOURCE_BYTES;
      const currentIllustriousHydrationBytes = document.downloadedFiles.filter((file) => (
        ILLUSTRIOUS_CURRENT_AUTHORITIES.has(file.artifactId)
      )).reduce((sum, file) => sum + file.bytes, 0);
      const expectedJitPeakBytes = PRE_HYDRATION_JIT_SOURCE_PEAK_BYTES
        + currentIllustriousHydrationBytes;
      if (document.diskPlan.reviewedAllAtOnceSourceBytes !== expectedAllAtOnceSourceBytes
        || document.diskPlan.preHydrationJitSourcePeakBytes
          !== PRE_HYDRATION_JIT_SOURCE_PEAK_BYTES
        || document.diskPlan.reviewedJitSourcePeakBytes !== REVIEWED_JIT_SOURCE_PEAK_BYTES
        || document.diskPlan.logicalSourceBytes !== expectedSourceBytes
        || document.diskPlan.allAtOnceSourceBytes !== expectedSourceBytes
        || expectedJitPeakBytes > REVIEWED_JIT_SOURCE_PEAK_BYTES
        || document.diskPlan.peakModelAndSidecarBytes !== expectedJitPeakBytes
        || document.diskPlan.peakRequiredAdditionalBytes !== REVIEWED_FREE_FLOOR_BYTES) {
        fail("passed cache preflight drifted from reviewed target-byte totals and JIT floor");
      }
    }
    if (document.evidencePhase === "initial") {
      if (document.phases.staging.length || document.phases.finalOffline.length
        || document.derivedSidecarLifecycle.afterCells.length
        || document.sidecarObstructions.length) {
        fail("initial cache preflight contains post-cell lifecycle evidence");
      }
    } else {
      const plannedIds = document.lifetimePlan.map((row) => row.artifactId);
      if (JSON.stringify(document.phases.staging.map((row) => row.id))
          !== JSON.stringify(plannedIds)
        || JSON.stringify(document.phases.finalOffline.map((row) => row.id))
          !== JSON.stringify(plannedIds)) {
        fail("final cache preflight omitted an exact staged/offline authority transition");
      }
      const expectedCells = plannedExecutionOrdinals.map((ordinal) => ({
        ordinal,
        cellId: profile.cells[ordinal - 1].id,
      }));
      if (JSON.stringify(document.derivedSidecarLifecycle.afterCells.map(
        ({ ordinal, cellId }) => ({ ordinal, cellId }),
      )) !== JSON.stringify(expectedCells)) {
        fail("final cache preflight omitted exact per-cell derived lifecycle coverage");
      }
      const obstructed = new Set(document.sidecarObstructions.flatMap((row) => row.artifactIds));
      if (plannedIds.some((id) => !obstructed.has(id))) {
        fail("final cache preflight omitted a staged authority sidecar obstruction");
      }
    }
  }
  let previous = 0;
  for (const snapshot of document.derivedSidecarLifecycle.afterCells) {
    const expected = profile.cells[snapshot.ordinal - 1];
    if (!expected || snapshot.cellId !== expected.id || snapshot.ordinal <= previous
      || snapshot.inventory.root !== derivedSidecarRoot) {
      fail("derived Candle sidecar lifecycle ordering drifted");
    }
    const flux = expected.artifactIds.find((id) => id.startsWith("flux1-"));
    const payloadFloor = flux?.endsWith("-q4") ? 7_396_392_960 : 12_573_868_032;
    const payloadCeiling = flux?.endsWith("-q4")
      ? FLUX_Q4_SIDECAR_BYTES : FLUX_Q8_SIDECAR_BYTES;
    const emptyProviderFailure = snapshot.providerExecution === "failed"
      && snapshot.inventory.files === 0 && snapshot.inventory.bytes === 0
      && snapshot.inventory.sha256 === createHash("sha256").digest("hex");
    let expectedSuccessDisposition = null;
    if (snapshot.providerExecution === "completed") {
      validateRequestMemoryStrategy(snapshot.requestMemoryStrategy, expected);
      expectedSuccessDisposition = dispositionForMemoryStrategy(snapshot.requestMemoryStrategy);
    } else if (snapshot.requestMemoryStrategy !== null) {
      fail("failed provider cache lifecycle cannot claim a request memory strategy");
    }
    const emptyResident = snapshot.providerExecution === "completed"
      && expectedSuccessDisposition === (flux
        ? DERIVED_DISPOSITION_RESIDENT : DERIVED_DISPOSITION_NOT_APPLICABLE)
      && snapshot.inventory.files === 0 && snapshot.inventory.bytes === 0
      && snapshot.inventory.sha256 === createHash("sha256").digest("hex");
    const exactBounded = flux && snapshot.providerExecution === "completed"
      && expectedSuccessDisposition === DERIVED_DISPOSITION_BOUNDED
      && snapshot.inventory.files === 494
      && snapshot.inventory.bytes >= payloadFloor
      && snapshot.inventory.bytes <= payloadCeiling;
    const expectedDisposition = !flux ? DERIVED_DISPOSITION_NOT_APPLICABLE
      : emptyResident ? DERIVED_DISPOSITION_RESIDENT
        : emptyProviderFailure ? DERIVED_DISPOSITION_PROVIDER_FAILED
          : exactBounded ? DERIVED_DISPOSITION_BOUNDED : null;
    if (!expectedDisposition || snapshot.derivedDisposition !== expectedDisposition
      || (!flux && (snapshot.inventory.files !== 0 || snapshot.inventory.bytes !== 0
        || snapshot.inventory.sha256 !== createHash("sha256").digest("hex")))) {
      fail("derived Candle sidecar lifecycle inventory drifted from the cell family contract");
    }
    previous = snapshot.ordinal;
  }
  return document;
}

async function writeCachePreflightDurably({
  output, filename, document, validation, fault, injectFaults = true,
}) {
  const validate = () => validateCachePreflightEvidence(document, validation);
  try {
    if (injectFaults) await invokeFault(fault, "preflightSchema", null);
    validate();
    if (injectFaults) await invokeFault(fault, "preflightWrite", null);
    const primary = path.join(output, filename);
    await writeJsonAtomically(primary, document, { schemaPath: CACHE_PREFLIGHT_SCHEMA_PATH });
    if (injectFaults) await invokeFault(fault, "preflightStat", null);
    const metadata = await stat(primary);
    if (injectFaults) await invokeFault(fault, "preflightHash", null);
    return {
      evidence: { path: filename, bytes: metadata.size, sha256: await sha256File(primary) },
      error: null,
    };
  } catch (error) {
    return { evidence: null, error };
  }
}

async function writeCachePreflightFallback({ output, filename, document, validation }) {
  validateCachePreflightEvidence(document, validation);
  const fallbackDir = path.join(output, "_emergency");
  await mkdir(fallbackDir, { recursive: true });
  const basename = `${path.basename(filename, ".json")}-fallback.json`;
  const fallback = path.join(fallbackDir, basename);
  await writeJsonAtomically(fallback, document, { schemaPath: CACHE_PREFLIGHT_SCHEMA_PATH });
  const metadata = await stat(fallback);
  return {
    path: `_emergency/${basename}`,
    bytes: metadata.size,
    sha256: await sha256File(fallback),
  };
}

export async function runCampaign(args, options = {}) {
  let prepared = options.prepared;
  if (!prepared) {
    if (process.env.SCENEWORKS_ENABLE_EPIC_20738_TERMINAL_CUDA !== "1") {
      fail("terminal hardware execution is opt-in; set SCENEWORKS_ENABLE_EPIC_20738_TERMINAL_CUDA=1 only for the frozen final dispatch");
    }
    validateDocumentWithSchema(PROFILE_SCHEMA_PATH, args.profile);
    const profile = validateProfile(loadProfile(args.profile));
    const sceneworks = repositoryIdentity(args.sceneworks, "SceneWorks");
    const inference = repositoryIdentity(args.inference, "inference");
    if (sceneworks.sha !== args.sceneworksRevision) fail("SceneWorks HEAD does not match --sceneworks-revision");
    if (inference.sha !== args.inferenceRevision) fail("inference HEAD does not match --inference-revision");
    const pin = await liveInferencePin(sceneworks.root);
    if (pin !== inference.sha) {
      fail(`SceneWorks pins inference ${pin} but the checked-out inference HEAD is ${inference.sha}`);
    }
    const manifest = JSON.parse(stripJsoncComments(await readFile(
      path.join(sceneworks.root, "config/manifests/builtin.models.jsonc"), "utf8",
    )));
    validateManifestAuthorities(profile, manifest);
    const downloadEvidencePath = path.join(sceneworks.root, DOWNLOAD_EVIDENCE_PATH);
    const downloadEvidenceBytes = await readFile(downloadEvidencePath, "utf8");
    const { artifactExpectedFiles, downloadEvidenceSha256 } =
      expectedCurrentArtifactFilesFromEvidenceBytes(profile, downloadEvidenceBytes);

    const repositories = {
      sceneworks: { sha: sceneworks.sha, clean: true },
      inference: { sha: inference.sha, clean: true },
    };
    const execution = workflowExecution(sceneworks.sha);
    const guard = await validateCampaignPaths({
      runnerTemp: process.env.RUNNER_TEMP,
      output: args.output,
      scratch: args.scratch,
      cacheRoot: args.cacheRoot,
      repositories: [sceneworks.root, inference.root],
    });
    const prefixCandidates = await realpath(args.prefixCandidates);
    const prefixMetadata = await lstat(prefixCandidates);
    if (!prefixMetadata.isDirectory() || prefixMetadata.isSymbolicLink()
      || !isWithin(guard.runnerTemp, prefixCandidates)
      || guard.repositories.some((repository) => isWithin(repository, prefixCandidates, { allowEqual: true }))) {
      fail("prefix candidates must be an ordinary RUNNER_TEMP directory outside both repositories");
    }
    const { output, scratch } = guard;
    await mkdir(output, { recursive: false });
    await mkdir(scratch, { recursive: false });
    const startupErrors = [];
    let gpuIdentity = [];
    try {
      gpuIdentity = sampleGpu();
    } catch (error) {
      recordError(startupErrors, "initial GPU identity sample failed", error);
    }
    prepared = {
      profile, repositories, execution, guard, output, scratch, startupErrors, gpuIdentity,
      systemMemory: { totalBytes: totalmem(), availableBytesAtStart: freemem() },
      sceneworksRoot: sceneworks.root,
      prefixCandidates,
      python: args.python,
      artifactExpectedFiles,
      downloadEvidenceSha256,
    };
  }
  const {
    profile, repositories, execution, guard, output, scratch, startupErrors, gpuIdentity,
    systemMemory, sceneworksRoot, prefixCandidates, python,
    artifactExpectedFiles, downloadEvidenceSha256,
  } = prepared;
  if (!artifactExpectedFiles || !/^[0-9a-f]{64}$/.test(downloadEvidenceSha256)) {
    fail("terminal preparation did not bind authoritative download-pattern evidence");
  }
  const fault = options.fault;
  const operations = {
    auditArtifact: options.operations?.auditArtifact ?? auditArtifact,
    downloadReviewedMissing:
      options.operations?.downloadReviewedMissing ?? downloadReviewedMissing,
    stageArtifact: options.operations?.stageArtifact ?? stageArtifact,
    provisionArtifact: options.operations?.provisionArtifact ?? provisionArtifact,
    installSidecarObstructions:
      options.operations?.installSidecarObstructions ?? installSidecarObstructions,
    directoryInventory: options.operations?.directoryInventory ?? directoryInventory,
    expectedB646DerivedNamespace:
      options.operations?.expectedB646DerivedNamespace ?? expectedB646DerivedNamespace,
    inspectDerivedSidecarRoot:
      options.operations?.inspectDerivedSidecarRoot ?? inspectDerivedSidecarRoot,
    diskFreeBytes: options.operations?.diskFreeBytes ?? diskFreeBytes,
    verifyArtifactUnchanged:
      options.operations?.verifyArtifactUnchanged ?? verifyArtifactUnchanged,
    executeCell: options.operations?.executeCell ?? executeCell,
    cleanup: options.operations?.cleanup ?? safeRemoveTree,
    sample: options.operations?.sample ?? sampleGpuBestEffort,
  };
  const allOrdinals = profile.cells.map((_, index) => index + 1);
  const legacyContinuationOrdinals = allOrdinals.slice(IMPORTED_PREFIX_CELLS);
  const requestedOrdinals = options.executionOrdinals ?? (options.importedPrefix
    ? legacyContinuationOrdinals : OPENPOSE_RECOVERY_EXECUTION_ORDINALS);
  const executionOrdinals = exactExecutionOrdinals(profile, requestedOrdinals);
  let imported = { lineage: null, outcomes: [] };
  if (JSON.stringify(executionOrdinals) === JSON.stringify(OPENPOSE_RECOVERY_EXECUTION_ORDINALS)) {
    const selected = options.prefixSelection
      ?? await selectOpenPoseRecovery(prefixCandidates, profile);
    imported = options.importedPrefix ?? await importOpenPoseRecovery(selected, output);
  } else if (JSON.stringify(executionOrdinals) === JSON.stringify(PASS18_EXECUTION_ORDINALS)) {
    const selected = options.prefixSelection
      ?? await selectPass18Recovery(prefixCandidates, profile);
    imported = options.importedPrefix ?? await importPass18Recovery(selected, output);
  } else if (JSON.stringify(executionOrdinals) === JSON.stringify(SPARSE_EXECUTION_ORDINALS)) {
    const selected = options.prefixSelection
      ?? await selectSparseRecovery(prefixCandidates, profile);
    imported = options.importedPrefix ?? await importSparseRecovery(selected, output);
  } else if (JSON.stringify(executionOrdinals) === JSON.stringify(legacyContinuationOrdinals)) {
    const selected = options.prefixSelection
      ?? await selectImportedPrefix(prefixCandidates, profile);
    imported = options.importedPrefix ?? await importPrefixEvidence(selected, output);
  } else if (JSON.stringify(executionOrdinals) !== JSON.stringify(allOrdinals)) {
    fail(`unreviewed terminal execution ordinals ${JSON.stringify(executionOrdinals)}`);
  }
  const receipts = [...imported.outcomes];
  const campaignErrors = [];
  let quarantineReason = null;
  const sharedArtifacts = new Map();
  const preflightScratch = path.join(scratch, "cache-preflight");
  const stagingRoot = path.join(scratch, "authority-stage");
  const derivedSidecarRoot = path.join(scratch, "derived-candle-device-cache");
  const missingStore = path.join(scratch, "persistent-missing-file");
  const expectedNonModelPaths = [
    { kind: "cargoTarget", path: path.resolve(process.env.CARGO_TARGET_DIR ?? path.join(sceneworksRoot, "target")) },
    { kind: "cargoHome", path: path.resolve(process.env.CARGO_HOME ?? path.join(process.env.USERPROFILE ?? scratch, ".cargo")) },
    { kind: "campaignOutput", path: output },
    { kind: "pythonVenv", path: python ? path.dirname(path.dirname(path.resolve(python))) : "fixture-unavailable" },
  ];
  const remainingArtifactIds = [...new Set(
    executionOrdinals.flatMap((ordinal) => profile.cells[ordinal - 1].artifactIds),
  )];
  const remainingFileCount = remainingArtifactIds.reduce(
    (sum, id) => sum + artifactExpectedFiles[id].length, 0,
  );
  if (downloadEvidenceSha256 === CURRENT_DOWNLOAD_EVIDENCE_SHA256
    && JSON.stringify(executionOrdinals) === JSON.stringify(OPENPOSE_RECOVERY_EXECUTION_ORDINALS)
    && JSON.stringify(remainingArtifactIds) !== JSON.stringify(OPENPOSE_RECOVERY_REMAINING_ARTIFACT_IDS)) {
    fail("OpenPose recovery scope drifted from its exact five-cell authority set");
  }
  const exactOpenPoseRecovery = isExactOrdinals(
    executionOrdinals, OPENPOSE_RECOVERY_EXECUTION_ORDINALS,
  );
  if (downloadEvidenceSha256 === CURRENT_DOWNLOAD_EVIDENCE_SHA256
    && JSON.stringify(executionOrdinals) === JSON.stringify(SPARSE_EXECUTION_ORDINALS)
    && (JSON.stringify(remainingArtifactIds) !== JSON.stringify(SPARSE_REMAINING_ARTIFACT_IDS)
      || remainingArtifactIds.length !== SPARSE_REMAINING_AUTHORITIES
      || remainingFileCount !== SPARSE_REMAINING_FILES)) {
    fail(`sparse recovery scope drifted: expected ${SPARSE_REMAINING_AUTHORITIES} authorities / ${SPARSE_REMAINING_FILES} files`);
  }
  if (downloadEvidenceSha256 === CURRENT_DOWNLOAD_EVIDENCE_SHA256
    && JSON.stringify(executionOrdinals) === JSON.stringify(PASS18_EXECUTION_ORDINALS)
    && (JSON.stringify(remainingArtifactIds) !== JSON.stringify(PASS18_REMAINING_ARTIFACT_IDS)
      || remainingArtifactIds.length !== PASS18_REMAINING_AUTHORITIES
      || remainingFileCount !== PASS18_REMAINING_FILES)) {
    fail(`18-PASS recovery scope drifted: expected ${PASS18_REMAINING_AUTHORITIES} authorities / ${PASS18_REMAINING_FILES} files`);
  }
  const sourceAudits = new Map();
  const cacheProvisioning = { sourceCensus: [], staging: [], finalOffline: [] };
  const censusErrors = [];
  try {
    await invokeFault(fault, "preflightMkdir", null);
    await mkdir(preflightScratch);
    await mkdir(stagingRoot);
    await mkdir(derivedSidecarRoot);
    await mkdir(missingStore);
  } catch (error) {
    quarantineReason = `cache preflight directory preparation failed: ${errorText(error)}`;
  }
  for (const artifactId of quarantineReason ? [] : remainingArtifactIds) {
    try {
      const artifact = await operations.auditArtifact({
        id: artifactId,
        artifact: profile.artifacts[artifactId],
        expectedFiles: artifactExpectedFiles[artifactId],
        scratch: preflightScratch,
        python,
        cacheRoot: guard.cacheRoot,
      });
      sourceAudits.set(artifactId, artifact);
      cacheProvisioning.sourceCensus.push({
        id: artifactId,
        repository: artifact.repository,
        revision: artifact.revision,
        subdirectory: artifact.subdirectory,
        allowPatterns: artifact.allowPatterns,
        expectedFiles: artifactExpectedFiles[artifactId],
        complete: artifact.complete,
        missingFiles: artifact.missingFiles,
        reusedFiles: artifact.reusedFiles,
      });
    } catch (error) {
      censusErrors.push(`${artifactId}: ${errorText(error)}`);
    }
  }

  // The source census is frozen before transfer. Only exact reviewed missing-file plans may use
  // the network; present bytes never fall back to another snapshot, glob, or mutable ref.
  const frozenMissing = cacheProvisioning.sourceCensus.flatMap((artifact) => (
    artifact.missingFiles.map((file) => ({
      artifactId: artifact.id,
      repository: artifact.repository,
      revision: artifact.revision,
      file,
    }))
  ));
  let reviewedDownloadPlan = [];
  if (quarantineReason) {
    // Preserve the first fail-closed setup error.
  } else if (censusErrors.length) {
    quarantineReason = `source cache census failed before transfer: ${censusErrors.join(" | ")}`;
  } else {
    try {
      reviewedDownloadPlan = reviewedMissingDownloadPlan({
        frozenMissing, profile, artifactExpectedFiles, downloadEvidenceSha256,
      });
    } catch (error) {
      quarantineReason = `source cache census found an unapproved missing-file set: ${error instanceof Error ? error.message : String(error)}`;
    }
  }

  const missingDownloads = [];
  let networkOfflineEstablished = false;
  let sidecarObstructions = [];
  const derivedSidecarLifecycle = { initial: null, afterCells: [] };
  let lifetimePlan = [];
  let diskPlan = null;
  if (!quarantineReason) {
    try {
      for (const plan of reviewedDownloadPlan) {
        const authorityMissingStore = path.join(missingStore, plan.id);
        await mkdir(authorityMissingStore);
        missingDownloads.push(await operations.downloadReviewedMissing({
          id: plan.id,
          artifact: profile.artifacts[plan.id],
          expectedFiles: artifactExpectedFiles[plan.id],
          missingFiles: plan.missingFiles,
          scratch: preflightScratch,
          python,
          cacheRoot: guard.cacheRoot,
          missingStore: authorityMissingStore,
        }));
      }
      for (const [artifactId, audit] of sourceAudits) {
        audit.expectedFiles = artifactExpectedFiles[artifactId];
        audit.downloadedFiles = missingDownloads.find((download) => download.id === artifactId)
          ?.downloadedFiles ?? [];
      }
      lifetimePlan = authorityLifetimePlan(profile, executionOrdinals, sourceAudits);
      const freeBytes = await operations.diskFreeBytes(scratch);
      const reviewedPlan = reviewedRecoveryDiskPlan(executionOrdinals);
      diskPlan = estimateJitDiskPlan(
        lifetimePlan,
        freeBytes,
        missingDownloads.reduce((sum, download) => sum + download.downloadedFiles.reduce(
          (subtotal, file) => subtotal + file.bytes, 0,
        ), 0),
        expectedNonModelPaths,
        downloadEvidenceSha256 === LEGACY_DOWNLOAD_EVIDENCE_SHA256
          ? LEGACY_REVIEWED_ALL_AT_ONCE_SOURCE_BYTES : reviewedPlan.sourceBytes,
        reviewedPlan.jitPeakBytes,
        reviewedPlan.freeFloorBytes,
      );
      if (exactOpenPoseRecovery) {
        assertExactOpenPoseRecoveryPreflight({
          executionOrdinals, remainingArtifactIds, artifactExpectedFiles, lifetimePlan, diskPlan,
        });
      }
      if (!diskPlan.admitted) {
        fail(`JIT staging disk admission refused: requires ${diskPlan.peakRequiredAdditionalBytes} bytes at cell ${diskPlan.peakOrdinal}, only ${diskPlan.freeBytes} free`);
      }
      if (missingDownloads.length === 0) {
        await operations.cleanup(missingStore, guard, "unused missing-file store");
        await assertPathAbsent(missingStore, "unused missing-file store");
      }
      derivedSidecarLifecycle.initial = await operations.directoryInventory(derivedSidecarRoot);
    } catch (error) {
      quarantineReason = `cache-only transfer/disk preflight failed: ${errorText(error)}`;
    }
  }
  if (!quarantineReason) networkOfflineEstablished = true;
  if (quarantineReason) {
    quarantineReason = `${quarantineReason}; no continuation GPU cell started`;
    campaignErrors.push(quarantineReason);
    try {
      await lstat(missingStore);
      await operations.cleanup(missingStore, guard, "failed preflight missing-file store");
      await assertPathAbsent(missingStore, "failed preflight missing-file store");
    } catch (error) {
      if (error?.code !== "ENOENT") {
        campaignErrors.push(`failed preflight store cleanup failed: ${errorText(error)}`);
      }
    }
  }
  const cacheValidation = {
    remainingArtifactIds, artifactExpectedFiles, downloadEvidenceSha256, guard, stagingRoot,
    derivedSidecarRoot, missingStore, expectedNonModelPaths, profile, executionOrdinals,
  };
  let initialDocument = cachePreflightDocument({
    evidencePhase: "initial",
    status: quarantineReason ? "failed" : "passed",
    error: quarantineReason,
    downloadEvidenceSha256,
    remainingArtifactIds,
    guard,
    stagingRoot,
    derivedSidecarRoot,
    frozenMissing,
    sidecarObstructions,
    cacheProvisioning,
    derivedSidecarLifecycle,
    missingStore,
    lifetimePlan,
    diskPlan,
    missingDownloads,
    networkOfflineEstablished,
  });
  let initialWrite = await writeCachePreflightDurably({
    output, filename: "cache-preflight-initial.json", document: initialDocument,
    validation: cacheValidation, fault,
  });
  if (initialWrite.error) {
    const evidenceError = `cache preflight schema/write/stat/hash failed: ${errorText(initialWrite.error)}`;
    if (!quarantineReason) {
      quarantineReason = evidenceError;
      campaignErrors.push(`${quarantineReason}; no continuation GPU cell started`);
    } else {
      campaignErrors.push(evidenceError);
    }
    initialDocument = cachePreflightDocument({
      evidencePhase: "initial",
      status: "failed",
      error: `${quarantineReason}; no continuation GPU cell started`,
      downloadEvidenceSha256,
      remainingArtifactIds,
      guard,
      stagingRoot,
      derivedSidecarRoot,
      frozenMissing: [],
      sidecarObstructions: [],
      cacheProvisioning: { sourceCensus: [], staging: [], finalOffline: [] },
      derivedSidecarLifecycle: { initial: null, afterCells: [] },
      missingStore,
      lifetimePlan: [],
      diskPlan: null,
      missingDownloads: [],
      networkOfflineEstablished,
    });
    initialWrite = {
      evidence: await writeCachePreflightFallback({
        output, filename: "cache-preflight-initial.json", document: initialDocument,
        validation: cacheValidation,
      }),
      error: null,
    };
    try {
      await lstat(missingStore);
      await operations.cleanup(missingStore, guard, "failed evidence missing-file store");
      await assertPathAbsent(missingStore, "failed evidence missing-file store");
    } catch (error) {
      if (error?.code !== "ENOENT") {
        campaignErrors.push(`failed evidence store cleanup failed: ${errorText(error)}`);
      }
    }
  }
  const lifetimeById = new Map(lifetimePlan.map((row) => [row.artifactId, row]));
  const lifecycleContext = (completeLifecycle = false) => ({
    lifetimeById,
    scratchRoot: scratch,
    requiredFreeBytes: diskPlan?.peakRequiredAdditionalBytes ?? REVIEWED_FREE_FLOOR_BYTES,
    completeLifecycle,
  });
  const authorityLifecycle = [];
  const diskFreeProbes = [];
  const probeDisk = async (lifecycle, phase, ordinal) => {
    const freeBytes = await operations.diskFreeBytes(scratch);
    const record = {
      phase,
      ordinal,
      root: scratch,
      freeBytes,
      requiredFreeBytes: diskPlan?.peakRequiredAdditionalBytes ?? REVIEWED_FREE_FLOOR_BYTES,
    };
    diskFreeProbes.push(record);
    lifecycle.diskProbes.push(record);
    if (!diskPlan || freeBytes < diskPlan.peakRequiredAdditionalBytes) {
      fail(`disk capacity fell below the ${diskPlan?.peakRequiredAdditionalBytes ?? "unavailable"}-byte JIT floor during ${phase}: ${freeBytes}`);
    }
    return record;
  };
  for (const ordinal of executionOrdinals) {
    const index = ordinal - 1;
    const cell = profile.cells[index];
    const ordinalName = `${String(index + 1).padStart(2, "0")}-${cell.id}`;
    let emergencyCleanup = { attempted: false, completed: true, error: null };
    const errors = [...startupErrors];
    const emergencyFailureMessages = errors;
    try {
    // Artifacts from a scratch tree whose removal was not proven may not coexist with a later cell.
    // Continue the reviewed sequence for durable outcomes, but do not enter any later lifecycle.
    if (quarantineReason) {
      fail(`cell ${cell.id} blocked before setup, provisioning, and execution: ${quarantineReason}`);
    }
    await invokeFault(fault, "setup", index);
    let cellDir = path.join(output, ordinalName);
    const cellScratch = path.join(scratch, ordinalName);
    const startedAt = new Date().toISOString();
    const artifacts = cell.artifactIds.map((id) => pendingArtifact(
      id, profile.artifacts[id], stagingRoot,
    ));
    const samples = gpuIdentity.length
      ? [...gpuIdentity]
      : [{ timestamp: startedAt, raw: startupErrors.join("\n") || "GPU identity unavailable" }];
    const receipt = receiptSkeleton({
      cell, ordinal: index + 1, repositories, artifacts, execution, gpuIdentity, systemMemory,
      startedAt,
    });
    receipt.hardware.rawVramSamples = samples;
    let cellDirCreated = false;
    let scratchCreated = false;
    let executionPassed = false;
    let controllerLog = path.join(cellDir, "controller.log");
    let receiptPath = path.join(cellDir, "receipt.json");
    let activeArtifactIndex = null;
    let authorityTransitioning = false;
    let executionAttempted = false;
    let transitionArtifactIds = [];
    const transitionRoots = new Map();
    const lifecycle = {
      ordinal: index + 1,
      cellId: cell.id,
      staged: [],
      activeArtifactIds: [...cell.artifactIds],
      providerExecution: "not-attempted",
      requestMemoryStrategy: null,
      verifiedBefore: false,
      verifiedAfter: false,
      derivedAfter: [],
      released: [],
      diskProbes: [],
    };
    receipt.authorityLifecycle = lifecycle;

    try {
      await mkdir(cellDir);
      cellDirCreated = true;
      const logFile = path.join(cellDir, "runtime.log");
      await writeFile(controllerLog, `${startedAt} starting ${cell.id}\n`, { encoding: "utf8", flag: "wx" });
      receipt.logs = (await evidenceFiles(cellDir)).logs;
      await writePrimaryReceipt({
        file: receiptPath, receipt, cell, profile, fault, index,
        lifecycleContext: lifecycleContext(false),
      });

      await mkdir(cellScratch);
      scratchCreated = true;
      const startingIds = cell.artifactIds.filter(
        (artifactId) => lifetimeById.get(artifactId)?.firstOrdinal === index + 1,
      );
      const newlyStaged = new Map();
      transitionArtifactIds = startingIds;
      authorityTransitioning = startingIds.length > 0;
      for (const artifactId of startingIds) {
        await invokeFault(fault, "stage", index);
        await probeDisk(lifecycle, `before-stage:${artifactId}`, index + 1);
        const audit = sourceAudits.get(artifactId);
        const artifactDownload = missingDownloads.find((download) => download.id === artifactId);
        const artifactStageRoot = artifactDownload?.storeRoot ?? path.join(stagingRoot, artifactId);
        transitionRoots.set(artifactId, artifactStageRoot);
        if (!artifactDownload) await mkdir(artifactStageRoot);
        const staged = await operations.stageArtifact({
          id: artifactId,
          artifact: profile.artifacts[artifactId],
          expectedFiles: artifactExpectedFiles[artifactId],
          scratch: preflightScratch,
          python,
          cacheRoot: guard.cacheRoot,
          stagingRoot: artifactStageRoot,
          missingStore: artifactDownload?.storeRoot ?? null,
        });
        if (JSON.stringify(staged.reusedFiles) !== JSON.stringify(audit.reusedFiles)) {
          fail(`trusted source cache changed after frozen census for ${artifactId}`);
        }
        const expectedDownloaded = artifactDownload?.downloadedFiles ?? [];
        assertExactDownloadedFilePartition(
          staged.downloadedFiles ?? [], expectedDownloaded,
          `offline JIT stage downloaded-file partition for ${artifactId}`,
        );
        cacheProvisioning.staging.push({
          id: artifactId,
          repository: profile.artifacts[artifactId].repository,
          revision: profile.artifacts[artifactId].revision,
          subdirectory: profile.artifacts[artifactId].subdirectory,
          allowPatterns: profile.artifacts[artifactId].allowPatterns,
          expectedFiles: artifactExpectedFiles[artifactId],
          reusedFiles: staged.reusedFiles,
          downloadedFiles: staged.downloadedFiles ?? [],
        });
        const artifact = await operations.provisionArtifact({
          id: artifactId,
          artifact: profile.artifacts[artifactId],
          expectedFiles: artifactExpectedFiles[artifactId],
          scratch: preflightScratch,
          python,
          cacheRoot: artifactStageRoot,
        });
        artifact.stageRoot = artifactStageRoot;
        newlyStaged.set(artifactId, artifact);
        sharedArtifacts.set(artifactId, artifact);
        cacheProvisioning.finalOffline.push({
          id: artifactId,
          repository: artifact.repository,
          revision: artifact.revision,
          subdirectory: artifact.subdirectory,
          allowPatterns: artifact.allowPatterns,
          expectedFiles: artifactExpectedFiles[artifactId],
          inventory: artifact.inventory,
        });
      }
      if (newlyStaged.size) {
        const obstructions = await operations.installSidecarObstructions(newlyStaged);
        sidecarObstructions.push(...obstructions);
        for (const [artifactId, artifact] of newlyStaged) {
          if (!Array.isArray(artifact.sidecarObstructions)
            || artifact.sidecarObstructions.length === 0) {
            fail(`JIT staging did not obstruct model-adjacent sidecars for ${artifactId}`);
          }
          artifact.derivedNamespaces = [];
          lifecycle.staged.push({
            artifactId,
            stageRoot: artifact.stageRoot,
            inventory: artifact.inventory,
            obstructionCount: artifact.sidecarObstructions.length,
            derivedNamespaces: artifact.derivedNamespaces,
          });
        }
      }
      authorityTransitioning = false;
      for (const [artifactIndex, artifactId] of cell.artifactIds.entries()) {
        activeArtifactIndex = artifactIndex;
        await invokeFault(fault, "provision", index);
        const artifact = sharedArtifacts.get(artifactId);
        if (!artifact) fail(`artifact ${artifactId} was not admitted by cache-only preflight`);
        artifacts[artifactIndex] = {
          id: artifact.id, role: artifact.role, repository: artifact.repository,
          revision: artifact.revision, subdirectory: artifact.subdirectory,
          selectedRoot: artifact.selectedRoot, allowPatterns: artifact.allowPatterns,
          inventory: { ...artifact.inventory, complete: true, error: null },
        };
      }
      activeArtifactIndex = null;
      receipt.artifacts = artifacts;
      const runtimeCell = {
        ...cell,
        artifacts: artifacts.map(({ id, role, repository, revision, subdirectory, selectedRoot }) => ({
          id, role, repository, revision, subdirectory, root: selectedRoot,
        })),
      };
      const cellFile = path.join(cellDir, "cell.json");
      await writeFile(cellFile, `${JSON.stringify(runtimeCell, null, 2)}\n`, "utf8");
      const verifyCellCache = async (phase) => {
        for (const artifactId of cell.artifactIds) {
          await operations.verifyArtifactUnchanged(
            sharedArtifacts.get(artifactId),
            sharedArtifacts.get(artifactId).stageRoot,
          );
        }
        await appendFile(
          controllerLog,
          `${new Date().toISOString()} shared cache ${phase} inventory verified for ${cell.id}\n`,
          "utf8",
        );
      };
      try {
        await verifyCellCache("before");
        lifecycle.verifiedBefore = true;
      } catch (error) {
        quarantineReason = `shared cache changed before cell ${cell.id}: ${errorText(error)}`;
        campaignErrors.push(quarantineReason);
        throw error;
      }
      await operations.inspectDerivedSidecarRoot(derivedSidecarRoot, null);
      operations.sample(samples, errors, "pre-execution VRAM sample failed");
      await probeDisk(lifecycle, "before-execution", index + 1);
      await invokeFault(fault, "execute", index);
      let runtimeFailure = null;
      executionAttempted = true;
      try {
        await operations.executeCell({
          sceneworks: sceneworksRoot,
          cellFile,
          cellDir,
          logFile,
          samples,
          runtimeScratch: cellScratch,
          derivedSidecarRoot,
        });
        lifecycle.providerExecution = "completed";
      } catch (error) {
        lifecycle.providerExecution = "failed";
        runtimeFailure = error;
      }
      try {
        await verifyCellCache("after");
        lifecycle.verifiedAfter = true;
      } catch (error) {
        quarantineReason = `shared cache mutated during cell ${cell.id}: ${errorText(error)}`;
        campaignErrors.push(quarantineReason);
        throw error;
      }
      if (runtimeFailure === null) {
        try {
          const runtimeResultPath = path.join(cellDir, "runtime-result.json");
          await invokeFault(fault, "runtimeResultRead", index);
          const linkMetadata = await lstat(runtimeResultPath);
          if (!linkMetadata.isFile() || linkMetadata.isSymbolicLink()) {
            fail(`runtime-result is not an ordinary non-reparse file for ${cell.id}`);
          }
          const before = await stat(runtimeResultPath);
          const runtimeResultBytes = await readFile(runtimeResultPath);
          await invokeFault(fault, "runtimeResultHash", index);
          const rawSha256 = createHash("sha256").update(runtimeResultBytes).digest("hex");
          const fileSha256 = await sha256File(runtimeResultPath);
          const after = await stat(runtimeResultPath);
          if (!before.isFile() || before.size !== runtimeResultBytes.length
            || after.size !== before.size || rawSha256 !== fileSha256) {
            fail(`runtime-result file/hash changed while validating ${cell.id}`);
          }
          await invokeFault(fault, "runtimeResultParse", index);
          const runtimeResult = JSON.parse(runtimeResultBytes.toString("utf8"));
          await invokeFault(fault, "runtimeResultSemantic", index);
          validateRuntimeResult(runtimeResult, cell);
          receipt.cell.loadSpecQuantBits = runtimeResult.loadSpecQuantBits;
          if (cell.kind === "scail2" && cell.capability === "multiReference") {
            receipt.cell.referenceCounterfactuals = structuredClone(runtimeResult.metrics.referenceCounterfactuals);
          }
          if (cell.kind === "sdxlOpenPose") {
            receipt.cell.controlCounterfactual = structuredClone(runtimeResult.metrics.controlCounterfactual);
          }
          lifecycle.requestMemoryStrategy = structuredClone(runtimeResult.requestMemoryStrategy);
        } catch (error) {
          quarantineReason = `runtime-result evidence failed after cell ${cell.id}: ${errorText(error)}`;
          campaignErrors.push(quarantineReason);
          throw error;
        }
      }
      try {
        const fluxArtifacts = cell.artifactIds.filter((artifactId) => artifactId.startsWith("flux1-"));
        if (fluxArtifacts.length > 1) fail(`cell ${cell.id} has multiple FLUX sidecar authorities`);
        const expectedNamespace = fluxArtifacts.length === 1
          ? await operations.expectedB646DerivedNamespace(
            sharedArtifacts.get(fluxArtifacts[0]), derivedSidecarRoot,
          ) : null;
        const derivedInspection = await operations.inspectDerivedSidecarRoot(
          derivedSidecarRoot, expectedNamespace,
          {
            materializeExactEmpty: runtimeFailure !== null && expectedNamespace !== null,
            allowExactEmpty: runtimeFailure === null && expectedNamespace !== null,
          },
        );
        if (fluxArtifacts.length === 1) {
          sharedArtifacts.get(fluxArtifacts[0]).derivedNamespaces.splice(
            0, Infinity, ...derivedInspection.namespaces.map((entry) => entry.path),
          );
        }
        for (const artifactId of cell.artifactIds) {
          const artifact = sharedArtifacts.get(artifactId);
          const inventories = [];
          for (const namespace of artifact.derivedNamespaces) {
            const bound = derivedInspection.namespaces.find(
              (entry) => comparable(entry.path) === comparable(namespace),
            );
            if (!bound) fail(`derived namespace was not bound by global inspection: ${namespace}`);
            inventories.push(bound.inventory);
          }
          const files = inventories.reduce((sum, inventory) => sum + inventory.files, 0);
          const bytes = inventories.reduce((sum, inventory) => sum + inventory.bytes, 0);
          const exactEmptyProviderFailure = runtimeFailure !== null
            && files === 0 && bytes === 0 && inventories.length === 1;
          const exactResident = runtimeFailure === null
            && dispositionForMemoryStrategy(lifecycle.requestMemoryStrategy)
              === DERIVED_DISPOSITION_RESIDENT
            && files === 0 && bytes === 0 && inventories.length === 0;
          const exactBounded = runtimeFailure === null
            && dispositionForMemoryStrategy(lifecycle.requestMemoryStrategy)
              === DERIVED_DISPOSITION_BOUNDED
            && files === 494
            && bytes >= (artifactId.endsWith("-q4") ? 7_396_392_960 : 12_573_868_032)
            && bytes <= (artifactId.endsWith("-q4")
              ? FLUX_Q4_SIDECAR_BYTES : FLUX_Q8_SIDECAR_BYTES)
            && inventories.length === 1;
          const derivedDisposition = artifactId.startsWith("flux1-")
            ? exactResident ? DERIVED_DISPOSITION_RESIDENT
              : exactEmptyProviderFailure ? DERIVED_DISPOSITION_PROVIDER_FAILED
                : exactBounded ? DERIVED_DISPOSITION_BOUNDED : null
            : DERIVED_DISPOSITION_NOT_APPLICABLE;
          if (artifactId.startsWith("flux1-") && !derivedDisposition) {
            fail(`FLUX derived-sidecar contract drifted for ${artifactId}: ${files} files/${bytes} bytes`);
          }
          if (!artifactId.startsWith("flux1-") && (files !== 0 || bytes !== 0
              || inventories.length !== 0)) {
            fail(`non-FLUX authority unexpectedly created derived sidecars: ${artifactId}`);
          }
          lifecycle.derivedAfter.push({
            artifactId, derivedDisposition, files, bytes, inventories,
          });
        }
        const derivedDisposition = fluxArtifacts.length === 0
          ? DERIVED_DISPOSITION_NOT_APPLICABLE
          : lifecycle.derivedAfter.find((entry) => entry.artifactId === fluxArtifacts[0])
            ?.derivedDisposition;
        derivedSidecarLifecycle.afterCells.push({
          ordinal: index + 1,
          cellId: cell.id,
          providerExecution: lifecycle.providerExecution,
          requestMemoryStrategy: lifecycle.requestMemoryStrategy,
          derivedDisposition,
          inventory: derivedInspection.rootInventory,
        });
      } catch (error) {
        quarantineReason = `derived Candle sidecar evidence failed after cell ${cell.id}: ${errorText(error)}`;
        campaignErrors.push(quarantineReason);
        throw error;
      }
      if (runtimeFailure) throw runtimeFailure;
      operations.sample(samples, errors, "post-execution VRAM sample failed");
      executionPassed = true;
    } catch (error) {
      const message = recordError(errors, "cell lifecycle failed", error);
      if (!executionAttempted && !quarantineReason) {
        quarantineReason = authorityTransitioning
          ? `JIT authority stage/copy/hash failed before cell ${cell.id}: ${errorText(error)}`
          : `cell pre-execution admission/evidence failed before ${cell.id}: ${errorText(error)}`;
        campaignErrors.push(quarantineReason);
      }
      if (activeArtifactIndex !== null && artifacts[activeArtifactIndex].inventory.complete === false) {
        artifacts[activeArtifactIndex].inventory.error = message;
      }
      if (cellDirCreated) {
        try { await appendFile(controllerLog, `${new Date().toISOString()} ${message}\n`, "utf8"); } catch {
          // The fallback log below is independently created and hashed if this path is unavailable.
        }
      }
      operations.sample(samples, [], "failure VRAM sample failed");
    } finally {
      const releaseIds = [...new Set([
        ...cell.artifactIds.filter((id) => lifetimeById.get(id)?.lastOrdinal === index + 1),
        ...(authorityTransitioning ? transitionArtifactIds : []),
      ])];
      for (const artifactId of releaseIds) {
        const artifact = sharedArtifacts.get(artifactId);
        if (!artifact) {
          const partialRoot = transitionRoots.get(artifactId);
          if (partialRoot) {
            const partialRelease = {
              artifactId,
              stageRoot: partialRoot,
              stagedInventory: null,
              derivedBeforeCleanup: [],
              stageRemoved: false,
              derivedRemoved: true,
            };
            try {
              partialRelease.stagedInventory = await operations.directoryInventory(partialRoot);
              await operations.cleanup(partialRoot, guard, `partial staged authority ${artifactId}`);
              await assertPathAbsent(partialRoot, `partial staged authority ${artifactId}`);
              partialRelease.stageRemoved = true;
            } catch (error) {
              partialRelease.error = errorText(error);
              if (!quarantineReason) {
                quarantineReason = `partial authority ${artifactId} cleanup failed: ${errorText(error)}`;
                campaignErrors.push(quarantineReason);
              }
            }
            lifecycle.released.push(partialRelease);
          }
          continue;
        }
        const release = {
          artifactId,
          stageRoot: artifact.stageRoot,
          stagedInventory: artifact.inventory,
          derivedBeforeCleanup: [],
          stageRemoved: false,
          derivedRemoved: false,
        };
        try {
          await invokeFault(fault, "authorityCleanup", index);
          await operations.verifyArtifactUnchanged(artifact, artifact.stageRoot);
          for (const namespace of artifact.derivedNamespaces) {
            release.derivedBeforeCleanup.push(await operations.directoryInventory(namespace));
            await operations.cleanup(namespace, guard, `derived sidecars for ${artifactId}`);
            await assertPathAbsent(namespace, `derived sidecars for ${artifactId}`);
            const versionRoot = path.dirname(namespace);
            await operations.cleanup(versionRoot, guard, `derived sidecar version for ${artifactId}`);
            await assertPathAbsent(versionRoot, `derived sidecar version for ${artifactId}`);
          }
          release.derivedRemoved = true;
          await operations.cleanup(artifact.stageRoot, guard, `staged authority ${artifactId}`);
          await assertPathAbsent(artifact.stageRoot, `staged authority ${artifactId}`);
          release.stageRemoved = true;
          sharedArtifacts.delete(artifactId);
        } catch (error) {
          const message = recordError(errors, `authority lifecycle cleanup failed for ${artifactId}`, error);
          release.error = message;
          if (!quarantineReason) {
            quarantineReason = `authority ${artifactId} cleanup isolation failed: ${errorText(error)}`;
            campaignErrors.push(quarantineReason);
          }
        }
        lifecycle.released.push(release);
      }
      receipt.cleanup.attempted = scratchCreated;
      if (scratchCreated) {
        try {
          await invokeFault(fault, "cleanup", index);
          await operations.cleanup(cellScratch, guard, `scratch for ${cell.id}`);
          await assertPathAbsent(cellScratch, `scratch for ${cell.id}`);
          receipt.cleanup.completed = true;
        } catch (error) {
          const message = recordError(errors, "cell scratch cleanup failed", error);
          receipt.cleanup.error = message;
          if (!quarantineReason) {
            quarantineReason = `prior cell ${cell.id} scratch cleanup isolation failed: ${errorText(error)}`;
            campaignErrors.push(quarantineReason);
          }
        }
      } else {
        receipt.cleanup.completed = true;
      }
      emergencyCleanup = structuredClone(receipt.cleanup);
      authorityLifecycle.push(structuredClone(lifecycle));

      try {
        if (!cellDirCreated) {
          cellDir = path.join(output, `${ordinalName}-failure`);
          await mkdir(cellDir);
          cellDirCreated = true;
          controllerLog = path.join(cellDir, "controller.log");
          receiptPath = path.join(cellDir, "receipt.json");
        }
        try {
          await invokeFault(fault, "finalLog", index);
          await appendFile(
            controllerLog,
            `${new Date().toISOString()} finalizing ${cell.id}; errors=${errors.length}\n`,
            "utf8",
          );
        } catch (error) {
          recordError(errors, "controller log finalization failed", error);
          if (!quarantineReason) {
            quarantineReason = `primary receipt finalization failed for ${cell.id}: ${errorText(error)}`;
            campaignErrors.push(quarantineReason);
          }
          controllerLog = path.join(cellDir, "controller-fallback.log");
          await invokeFault(fault, "fallbackLog", index);
          await writeFile(controllerLog, `${new Date().toISOString()} ${errors.join("\n")}\n`, "utf8");
        }
        receipt.artifacts = artifacts;
        receipt.hardware.rawVramSamples = samples;
        const evidenceFailure = await refreshEvidence(receipt, cellDir, errors, { fault, index });
        if (evidenceFailure && !quarantineReason) {
          quarantineReason = `primary receipt evidence finalization failed for ${cell.id}: ${evidenceFailure}`;
          campaignErrors.push(quarantineReason);
        }
        receipt.status = executionPassed && errors.length === 0 && receipt.cleanup.completed ? "passed" : "failed";
        receipt.error = receipt.status === "passed" ? null : (errors.join("\n") || "cell did not complete");
        receipt.completedAt = new Date().toISOString();
        await writePrimaryReceipt({
          file: receiptPath, receipt, cell, profile, fault, index,
          lifecycleContext: lifecycleContext(true),
        });
        receipts.push({
          id: cell.id, status: receipt.status,
          receipt: `${path.basename(cellDir)}/receipt.json`, error: receipt.error,
          source: "continuation",
        });
      } catch (error) {
        throw new Error(recordError(errors, "best-effort receipt finalization failed", error), {
          cause: error,
        });
      }
    }
    } catch (error) {
      let receiptPath;
      let emergencyError = [...emergencyFailureMessages, errorText(error)].filter(Boolean).join("\n");
      if (!quarantineReason) {
        quarantineReason = `primary receipt semantic/schema/write/stat/hash finalization failed for ${cell.id}: ${errorText(error)}`;
        campaignErrors.push(quarantineReason);
      }
      let emergencyAuthorityLifecycle = authorityLifecycle.find(
        (entry) => entry.ordinal === index + 1,
      );
      if (!emergencyAuthorityLifecycle) {
        emergencyAuthorityLifecycle = emptyAuthorityLifecycle(cell, index);
        authorityLifecycle.push(emergencyAuthorityLifecycle);
      }
      try {
        await invokeFault(fault, "emergencyReceipt", index);
        receiptPath = await writeEmergencyReceipt({
          profile, cell, index, ordinalName, output, cacheRoot: stagingRoot,
          repositories, execution, gpuIdentity,
          systemMemory, error: new Error(emergencyError), cleanup: emergencyCleanup,
          authorityLifecycle: emergencyAuthorityLifecycle,
          lifecycleContext: lifecycleContext(false),
        });
      } catch (receiptError) {
        emergencyError += `\nemergency receipt failed: ${errorText(receiptError)}`;
        try {
          receiptPath = await writeLastResortEmergencyReceipt({
            profile, cell, index, ordinalName, output, cacheRoot: stagingRoot,
            repositories, execution, gpuIdentity,
            systemMemory, errors: [emergencyError], cleanup: emergencyCleanup,
            authorityLifecycle: emergencyAuthorityLifecycle,
            lifecycleContext: lifecycleContext(false),
          });
        } catch (lastResortError) {
          emergencyError += `\nlast-resort emergency receipt failed: ${errorText(lastResortError)}`;
        }
      }
      receipts.push({
        id: cell.id, status: "failed", receipt: receiptPath, error: emergencyError,
        emergencyReceiptError: receiptPath ? null : emergencyError, source: "continuation",
      });
    }
  }

  if (missingDownloads.length) {
    try {
      await cleanupMissingFileStore(operations, missingStore, guard, "reviewed missing-file store");
    } catch (error) {
      const message = `reviewed missing-file store cleanup failed: ${errorText(error)}`;
      campaignErrors.push(message);
      if (!quarantineReason) quarantineReason = message;
    }
  }

  let finalLifecycle = null;
  try {
    const stage = await operations.directoryInventory(stagingRoot);
    const derivedInspection = await operations.inspectDerivedSidecarRoot(derivedSidecarRoot, null);
    const derived = derivedInspection.rootInventory;
    let missingStoreAbsent = false;
    try {
      await lstat(missingStore);
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
      missingStoreAbsent = true;
    }
    finalLifecycle = { stage, derived, derivedNamespaces: [], missingStoreAbsent };
    if (stage.files !== 0 || stage.bytes !== 0 || derived.files !== 0 || derived.bytes !== 0
      || !missingStoreAbsent || sharedArtifacts.size !== 0) {
      fail("final JIT stage/derived/missing-file lifecycle is not empty");
    }
  } catch (error) {
    const message = `final authority lifecycle verification failed: ${errorText(error)}`;
    campaignErrors.push(message);
    if (!quarantineReason) quarantineReason = message;
  }

  let finalDocument = cachePreflightDocument({
    evidencePhase: "final",
    status: quarantineReason ? "failed" : "passed",
    error: quarantineReason,
    downloadEvidenceSha256,
    remainingArtifactIds,
    guard,
    stagingRoot,
    derivedSidecarRoot,
    frozenMissing,
    sidecarObstructions,
    cacheProvisioning,
    derivedSidecarLifecycle,
    missingStore,
    lifetimePlan,
    diskPlan,
    missingDownloads,
    networkOfflineEstablished,
  });
  let finalWrite = await writeCachePreflightDurably({
    output, filename: "cache-preflight.json", document: finalDocument,
    validation: cacheValidation, fault, injectFaults: false,
  });
  if (finalWrite.error) {
    const evidenceError = `final cache preflight evidence failed: ${errorText(finalWrite.error)}`;
    campaignErrors.push(evidenceError);
    finalDocument = cachePreflightDocument({
      evidencePhase: "final",
      status: "failed",
      error: quarantineReason ? `${quarantineReason}; ${evidenceError}` : evidenceError,
      downloadEvidenceSha256,
      remainingArtifactIds,
      guard,
      stagingRoot,
      derivedSidecarRoot,
      frozenMissing: [],
      sidecarObstructions: [],
      cacheProvisioning: { sourceCensus: [], staging: [], finalOffline: [] },
      derivedSidecarLifecycle: { initial: null, afterCells: [] },
      missingStore,
      lifetimePlan: [],
      diskPlan: null,
      missingDownloads: [],
      networkOfflineEstablished,
    });
    finalWrite = {
      evidence: await writeCachePreflightFallback({
        output, filename: "cache-preflight.json", document: finalDocument,
        validation: cacheValidation,
      }),
      error: null,
    };
  }
  const cachePreflightEvidence = finalWrite.evidence;

  try {
    await invokeFault(fault, "campaignCleanup", null);
    await operations.cleanup(scratch, guard, "campaign scratch");
  } catch (error) {
    campaignErrors.push(recordError([], "campaign scratch cleanup failed", error));
  }

  receipts.sort((left, right) => (
    profile.cells.findIndex((cell) => cell.id === left.id)
      - profile.cells.findIndex((cell) => cell.id === right.id)
  ));
  if (receipts.length !== profile.cells.length
    || receipts.some((receipt, index) => receipt.id !== profile.cells[index].id)) {
    fail("terminal receipt assembly did not cover each exact profile cell once in serial order");
  }
  const summary = {
    schemaVersion: 1, profile: PROFILE_NAME, repositories, execution, receipts,
    lineage: {
      imported: imported.lineage,
      continuation: {
        runId: execution.runId,
        runAttempt: execution.runAttempt,
        headSha: execution.headSha,
        inferenceSha: repositories.inference.sha,
        profileCellSemanticsSha256: EXPECTED_CELL_SEMANTICS_SHA256,
        profileArtifactSemanticsSha256: EXPECTED_ARTIFACT_SEMANTICS_SHA256,
        executionOrdinals,
      },
    },
    cachePreflight: cachePreflightEvidence,
    authorityLifecycle,
    diskFreeProbes,
    finalAuthorityLifecycle: finalLifecycle,
    passed: receipts.filter((receipt) => receipt.status === "passed").length,
    failed: receipts.filter((receipt) => receipt.status === "failed").length,
    campaignErrors,
  };
  const summaryPath = await writeSummaryDurably({ output, summary, fault });
  summary.passed = receipts.filter((receipt) => receipt.status === "passed").length;
  summary.failed = receipts.filter((receipt) => receipt.status === "failed").length;
  if (!options.suppressVerdict && (summary.failed || campaignErrors.length)) {
    fail(`terminal campaign completed all 19 cells with ${summary.failed} cell failure(s) and ${campaignErrors.length} campaign cleanup failure(s)`);
  }
  return { summary, summaryPath };
}

function value(args, flag) {
  const index = args.indexOf(flag);
  if (index < 0 || !args[index + 1]) fail(`missing ${flag}`);
  return args[index + 1];
}

async function main() {
  const [command, ...argv] = process.argv.slice(2);
  const profilePath = argv.includes("--profile") ? value(argv, "--profile") : PROFILE_PATH;
  if (command === "check") {
    validateDocumentWithSchema(PROFILE_SCHEMA_PATH, profilePath);
    validateDocumentWithSchema(CACHE_PREFLIGHT_SCHEMA_PATH, {
      schemaVersion: 1,
      profile: PROFILE_NAME,
      evidencePhase: "initial",
      status: "failed",
      error: "schema self-check fixture",
      downloadEvidenceSha256: "0".repeat(64),
      expectedArtifactIds: ["schema-self-check"],
      sourceCacheRoot: "fixture-cache",
      campaignStagingRoot: "fixture-staging",
      derivedSidecarRoot: "fixture-derived",
      missingFileStore: "fixture-missing",
      frozenMissingFiles: [],
      sidecarObstructions: [],
      reusedFiles: [],
      downloadedFiles: [],
      networkDownloadCount: 0,
      phases: { sourceCensus: [], staging: [], finalOffline: [] },
      lifetimePlan: [],
      diskPlan: null,
      derivedSidecarLifecycle: { initial: null, afterCells: [] },
      offlineBeforeCells: false,
    });
    const profile = validateProfile(loadProfile(profilePath));
    const pin = await liveInferencePin();
    const manifest = JSON.parse(stripJsoncComments(await readFile("config/manifests/builtin.models.jsonc", "utf8")));
    validateManifestAuthorities(profile, manifest);
    const downloadEvidenceBytes = await readFile(DOWNLOAD_EVIDENCE_PATH, "utf8");
    const { artifactExpectedFiles: expected } =
      expectedCurrentArtifactFilesFromEvidenceBytes(profile, downloadEvidenceBytes);
    if (Object.keys(expected).length !== 23) fail("terminal exact filename census is incomplete");
    process.stdout.write(`${PROFILE_NAME}: exactly 19 serialized cells and immutable authorities OK (inference pin ${pin})\n`);
    return;
  }
  if (command === "run") {
    await runCampaign({
      profile: profilePath,
      sceneworks: value(argv, "--sceneworks-repo"),
      inference: value(argv, "--inference-repo"),
      sceneworksRevision: value(argv, "--sceneworks-revision"),
      inferenceRevision: value(argv, "--inference-revision"),
      output: value(argv, "--output"), scratch: value(argv, "--scratch"), python: value(argv, "--python"),
      cacheRoot: value(argv, "--cache-root"),
      prefixCandidates: value(argv, "--prefix-candidates"),
    });
    return;
  }
  fail("usage: epic-20738-terminal-cuda-harness.mjs check [--profile path] | run --profile path --sceneworks-repo path --inference-repo path --sceneworks-revision sha40 --inference-revision sha40 --output path --scratch path --python path --cache-root path --prefix-candidates path");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
