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
const EXPECTED_CELL_SEMANTICS_SHA256 = "dc0e529b40e898727eb9401562a928345b958b4c94677d0206ccc70471f6f879";
const EXPECTED_ARTIFACT_SEMANTICS_SHA256 = "5b9ef60c18ab15caeca7ff0411b199618f0aa22cc051a70607aa7a0f7c6cd932";
const LEGACY_ARTIFACT_SEMANTICS_SHA256 = "f2bb7a77b83ce11cc32c3a1f9639534a67a149bc464a9730fb5c0988b4a03f9e";
const LEGACY_SCENEWORKS_HEAD = "8886a9e69f26beec05688c81b414859bd102f6d0";
const FROZEN_INFERENCE_PIN = "b646a6f89ba9f6b07efe53dd583d8a42e21e9871";
const LTX_CURRENT_REVISION = "01df27d308466533aa09d251e3aebdcc627d07eb";
const LTX_APPROVED_PARENT_REVISION = "254989c3ca7ee691187647f350b112c0c448789d";
const IMPORTED_PREFIX_CELLS = 7;
const PREFIX_ARTIFACT = `sc-20945-epic-20738-${LEGACY_SCENEWORKS_HEAD}-`;
const LEGACY_NAIVE_LINK_LENGTH_SOURCE_BYTES = 173_667_044_229;
const REVIEWED_ALL_AT_ONCE_SOURCE_BYTES = 179_028_698_264;
const REVIEWED_JIT_SOURCE_PEAK_BYTES = 56_156_615_634;
const NON_MODEL_DISK_RESERVE_BYTES = 40 * 1024 ** 3;
const REVIEWED_FREE_FLOOR_BYTES = 99_106_288_594;
const FLUX_Q4_SIDECAR_BYTES = 7_396_392_960 + 494 * 16_384;
const FLUX_Q8_SIDECAR_BYTES = 12_573_868_032 + 494 * 16_384;
const REVIEWED_DOWNLOAD_EVIDENCE_SHA256 = "9eda09eeacb9386167ca4a080b4805b9c7dd3cd5134ca037ce342ad434b17e0b";

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

export function authorityLifetimePlan(profile, startIndex, sourceAudits) {
  const cells = profile.cells.slice(startIndex);
  const ids = [...new Set(cells.flatMap((cell) => cell.artifactIds))];
  return ids.map((artifactId) => {
    const uses = [];
    for (let index = startIndex; index < profile.cells.length; index += 1) {
      if (profile.cells[index].artifactIds.includes(artifactId)) uses.push(index + 1);
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
) {
  if (!Number.isSafeInteger(freeBytes) || freeBytes < 0
    || !Number.isSafeInteger(persistentMissingBytes) || persistentMissingBytes < 0) {
    fail("disk estimator requires exact non-negative byte counts");
  }
  const ordinals = [...new Set(lifetimes.flatMap((row) => [row.firstOrdinal, row.lastOrdinal]))]
    .sort((left, right) => left - right);
  const cells = ordinals.map((ordinal) => {
    const active = lifetimes.filter((row) => (
      row.firstOrdinal <= ordinal && row.lastOrdinal >= ordinal
    ));
    const physical = new Map();
    // The reviewed download is persistent across the preflight and earlier cells, but belongs to
    // its authority's lifetime: it is deleted with Schnell q8 after cell 10, never charged forever.
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
        REVIEWED_FREE_FLOOR_BYTES,
      ),
    };
  });
  const peak = cells.reduce((highest, row) => (
    row.modelAndSidecarBytes > highest.modelAndSidecarBytes ? row : highest
  ), { ordinal: 0, artifactIds: [], stagedBytes: 0, sidecarReserveBytes: 0,
    modelAndSidecarBytes: 0, requiredAdditionalBytes: REVIEWED_FREE_FLOOR_BYTES });
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
  const allAtOnceSidecarReserveBytes = Math.max(FLUX_Q4_SIDECAR_BYTES, FLUX_Q8_SIDECAR_BYTES);
  if (logicalSourceBytes === REVIEWED_ALL_AT_ONCE_SOURCE_BYTES
    && (allAtOnceSourceBytes !== REVIEWED_ALL_AT_ONCE_SOURCE_BYTES
      || (persistentMissingBytes > 0
        ? peak.modelAndSidecarBytes !== REVIEWED_JIT_SOURCE_PEAK_BYTES
        : peak.modelAndSidecarBytes > REVIEWED_JIT_SOURCE_PEAK_BYTES))) {
    fail(`reviewed disk census drifted: all=${allAtOnceSourceBytes}, peak=${peak.modelAndSidecarBytes}`);
  }
  return {
    freeBytes,
    persistentMissingBytes,
    nonModelReserveBytes: NON_MODEL_DISK_RESERVE_BYTES,
    nonModelPaths,
    legacyNaiveLinkLengthSourceBytes: LEGACY_NAIVE_LINK_LENGTH_SOURCE_BYTES,
    reviewedAllAtOnceSourceBytes: REVIEWED_ALL_AT_ONCE_SOURCE_BYTES,
    reviewedJitSourcePeakBytes: REVIEWED_JIT_SOURCE_PEAK_BYTES,
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
      REVIEWED_FREE_FLOOR_BYTES,
    ),
    admitted: freeBytes >= Math.max(
      peak.modelAndSidecarBytes + NON_MODEL_DISK_RESERVE_BYTES,
      REVIEWED_FREE_FLOOR_BYTES,
    ),
    cells,
  };
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
  return runtimeResult;
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
    exactKeys(artifact, ["authority", "role", "repository", "revision", "subdirectory", "allowPatterns"], `artifact ${id}`);
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
    } else if (authority.kind === "explicitPublicArtifact") {
      // The profile keeps one shared authority for all five cells. Bind it to the production
      // sc-20747 soft co-requisite rows without copying those rows into a fake utility model or
      // allowing another consumer/duplicate row to drift into the terminal campaign.
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
      const expectedModels = [...SDXL_POSE_MODELS].sort();
      if (JSON.stringify(actualModels) !== JSON.stringify(expectedModels)) {
        fail(`OpenPose ControlNet manifest authority must have the exact five approved consumers`);
      }
      for (const modelId of SDXL_POSE_MODELS) {
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
        const exactComplete = derived.files === 494 && derived.bytes >= payloadFloor
          && derived.bytes <= payloadCeiling && oneBoundNamespace;
        if (!exactEmptyFailure && !exactComplete) {
          fail("receipt FLUX derived lifecycle is not one exact bounded b646 namespace");
        }
      } else if (derived.files !== 0 || derived.bytes !== 0 || derived.inventories.length !== 0) {
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
  legacy.artifacts["ltx23-q8"].revision = LTX_CURRENT_REVISION;
  legacy.artifacts["ltx23-gemma"].revision = LTX_CURRENT_REVISION;
  if (cellSemanticsSha256(legacy.cells) !== EXPECTED_CELL_SEMANTICS_SHA256
    || canonicalSha256(legacy.artifacts) !== LEGACY_ARTIFACT_SEMANTICS_SHA256) {
    fail("legacy prefix compatibility profile drifted");
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
    || metadata.inferenceSha !== FROZEN_INFERENCE_PIN
    || metadata.profile !== PROFILE_NAME
    || metadata.cellSemanticsSha256 !== EXPECTED_CELL_SEMANTICS_SHA256
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
      || receipt.repositories?.inference?.sha !== FROZEN_INFERENCE_PIN
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
    || boundaryReceipt.repositories?.inference?.sha !== FROZEN_INFERENCE_PIN
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
  if (result.id !== id || comparable(await realpath(result.cacheRoot)) !== comparable(cacheRoot)
    || comparable(await realpath(result.snapshotRoot)) !== comparable(expected.snapshotRoot)
    || comparable(await realpath(result.selectedRoot)) !== comparable(expected.selectedRoot)
    || (staged && typeof result.sourceCacheRoot !== "string")) {
    fail(`${phase} result drifted from exact authority ${id}`);
  }
  if (typeof result.complete !== "boolean" || !Array.isArray(result.missingFiles)
    || result.missingFiles.some((file) => typeof file !== "string" || !file)
    || result.complete !== (result.missingFiles.length === 0)
    || (!allowIncomplete && !result.complete)
    || !Array.isArray(result.matchedFiles) || result.matchedFiles.length === 0
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
  id, artifact, expectedFiles, scratch, python, cacheRoot, missingStore,
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
    || result.downloadedFiles.length !== 1) {
    fail(`reviewed missing-file fill identity drifted for ${id}`);
  }
  const [file] = result.downloadedFiles;
  exactKeys(
    file,
    ["path", "bytes", "sha256", "lfsSha256", "commitSha"],
    `reviewed missing-file fill for ${id}`,
  );
  if (file.path !== "transformer/model.safetensors"
    || file.commitSha !== "bba3ae01dfd94089f173c05edd4e1a4c551f2599"
    || !Number.isSafeInteger(file.bytes) || file.bytes < 1
    || !/^[0-9a-f]{64}$/.test(file.sha256) || file.sha256 !== file.lfsSha256) {
    fail("reviewed missing-file fill did not prove exact commit/size/LFS identity");
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
  derivedSidecarRoot, expectedNamespace = null, { materializeExactEmpty = false } = {},
) {
  let rootEntries = await readdir(derivedSidecarRoot, { withFileTypes: true });
  if (!expectedNamespace) {
    if (rootEntries.length) fail("non-FLUX derived sidecar root must be exactly empty");
    return { rootInventory: await directoryInventory(derivedSidecarRoot), namespaces: [] };
  }
  const expectedVersionRoot = path.join(derivedSidecarRoot, "candle-device-format-v1");
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
  derivedSidecarLifecycle, missingStore, lifetimePlan, diskPlan, missingDownload,
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
    downloadedFiles: missingDownload?.downloadedFiles.map((file) => ({
      artifactId: missingDownload.id, ...file,
    })) ?? [],
    networkDownloadCount: missingDownload ? 1 : 0,
    phases: cacheProvisioning,
    lifetimePlan,
    diskPlan,
    derivedSidecarLifecycle,
    offlineBeforeCells: networkOfflineEstablished,
  };
}

export function validateCachePreflightEvidence(document, {
  remainingArtifactIds, artifactExpectedFiles, downloadEvidenceSha256, guard, stagingRoot,
  derivedSidecarRoot, missingStore, expectedNonModelPaths, profile,
}) {
  validateDocumentWithSchema(CACHE_PREFLIGHT_SCHEMA_PATH, document);
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
  if (downloadEvidenceSha256 === REVIEWED_DOWNLOAD_EVIDENCE_SHA256
    && profile.cells.length === 19 && remainingArtifactIds.length === 16
    && remainingArtifactIds.reduce(
      (sum, id) => sum + artifactExpectedFiles[id].length,
      0,
    ) !== 199) {
    fail("continuation must bind the exact logical 16-authority/199-file census");
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
  if (document.downloadedFiles.length > 1 || document.downloadedFiles.some((file) => (
    file.artifactId !== "flux1-schnell-q8"
      || file.path !== "transformer/model.safetensors"
      || file.commitSha !== "bba3ae01dfd94089f173c05edd4e1a4c551f2599"
      || file.sha256 !== file.lfsSha256
  ))) fail("cache preflight contains an unreviewed model download");
  if ((frozen.length === 1) !== (document.downloadedFiles.length === 1)) {
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
    const plannedAudits = new Map(document.phases.sourceCensus.map((row) => [row.id, {
      ...row,
      downloadedFiles: document.downloadedFiles.filter((file) => file.artifactId === row.id),
    }]));
    const expectedLifetimePlan = authorityLifetimePlan(
      profile,
      profile.cells.length - document.diskPlan.cells.length,
      plannedAudits,
    );
    if (JSON.stringify(document.lifetimePlan) !== JSON.stringify(expectedLifetimePlan)) {
      fail("cache preflight authority lifetime plan drifted from the frozen census");
    }
    const expectedDiskPlan = estimateJitDiskPlan(
      expectedLifetimePlan,
      document.diskPlan.freeBytes,
      document.downloadedFiles.reduce((sum, file) => sum + file.bytes, 0),
      document.diskPlan.nonModelPaths,
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
    if (downloadEvidenceSha256 === REVIEWED_DOWNLOAD_EVIDENCE_SHA256
      && (document.diskPlan.logicalSourceBytes !== REVIEWED_ALL_AT_ONCE_SOURCE_BYTES
        || document.diskPlan.allAtOnceSourceBytes !== REVIEWED_ALL_AT_ONCE_SOURCE_BYTES
        || document.diskPlan.peakRequiredAdditionalBytes !== REVIEWED_FREE_FLOOR_BYTES
        || (frozen.length === 1
          ? document.diskPlan.peakModelAndSidecarBytes !== REVIEWED_JIT_SOURCE_PEAK_BYTES
          : document.diskPlan.peakModelAndSidecarBytes > REVIEWED_JIT_SOURCE_PEAK_BYTES))) {
      fail("passed cache preflight drifted from reviewed target-byte totals and JIT floor");
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
      const startIndex = profile.cells.length - document.diskPlan.cells.length;
      const expectedCells = profile.cells.slice(startIndex).map((cell, offset) => ({
        ordinal: startIndex + offset + 1,
        cellId: cell.id,
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
    if (flux ? (!emptyProviderFailure && (snapshot.inventory.files !== 494
        || snapshot.inventory.bytes < payloadFloor
        || snapshot.inventory.bytes > payloadCeiling))
      : (snapshot.inventory.files !== 0 || snapshot.inventory.bytes !== 0
        || snapshot.inventory.sha256 !== createHash("sha256").digest("hex"))) {
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
    const pins = inferencePins(await readFile(path.join(sceneworks.root, "Cargo.toml"), "utf8"));
    if (pins.length !== 1 || pins[0] !== inference.sha) {
      fail(`SceneWorks must have exactly one inference revision and it must equal checked-out inference HEAD; got ${pins.join(",")}`);
    }
    const manifest = JSON.parse(stripJsoncComments(await readFile(
      path.join(sceneworks.root, "config/manifests/builtin.models.jsonc"), "utf8",
    )));
    validateManifestAuthorities(profile, manifest);
    const downloadEvidencePath = path.join(sceneworks.root, DOWNLOAD_EVIDENCE_PATH);
    const downloadEvidenceBytes = await readFile(downloadEvidencePath, "utf8");
    const artifactExpectedFiles = expectedArtifactFilesFromEvidence(
      profile, JSON.parse(downloadEvidenceBytes),
    );

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
      downloadEvidenceSha256: createHash("sha256").update(downloadEvidenceBytes).digest("hex"),
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
  const startIndex = options.startIndex ?? IMPORTED_PREFIX_CELLS;
  let imported = { lineage: null, outcomes: [] };
  if (startIndex === IMPORTED_PREFIX_CELLS) {
    const selected = options.prefixSelection
      ?? await selectImportedPrefix(prefixCandidates, profile);
    imported = options.importedPrefix ?? await importPrefixEvidence(selected, output);
  } else if (startIndex !== 0) {
    fail(`unreviewed terminal continuation start index ${startIndex}`);
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
    profile.cells.slice(startIndex).flatMap((cell) => cell.artifactIds),
  )];
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

  // The source census is frozen before transfer. A complete cache uses no network; the only
  // incomplete census allowed is this one exact file, never a snapshot/glob/ref-main fallback.
  const frozenMissing = cacheProvisioning.sourceCensus.flatMap((artifact) => (
    artifact.missingFiles.map((file) => ({
      artifactId: artifact.id,
      repository: artifact.repository,
      revision: artifact.revision,
      file,
    }))
  ));
  const reviewedMissing = [{
    artifactId: "flux1-schnell-q8",
    repository: "SceneWorks/flux1-schnell-mlx",
    revision: "bba3ae01dfd94089f173c05edd4e1a4c551f2599",
    file: "q8/transformer/model.safetensors",
  }];
  if (quarantineReason) {
    // Preserve the first fail-closed setup error.
  } else if (censusErrors.length) {
    quarantineReason = `source cache census failed before transfer: ${censusErrors.join(" | ")}`;
  } else if (frozenMissing.length > 1
    || (frozenMissing.length === 1
      && JSON.stringify(frozenMissing) !== JSON.stringify(reviewedMissing))) {
    quarantineReason = `source cache census found an unapproved missing-file set: ${JSON.stringify(frozenMissing)}`;
  }

  let missingDownload = null;
  let networkOfflineEstablished = false;
  let sidecarObstructions = [];
  const derivedSidecarLifecycle = { initial: null, afterCells: [] };
  let lifetimePlan = [];
  let diskPlan = null;
  if (!quarantineReason) {
    try {
      if (frozenMissing.length === 1) {
        missingDownload = await operations.downloadReviewedMissing({
          id: "flux1-schnell-q8",
          artifact: profile.artifacts["flux1-schnell-q8"],
          expectedFiles: artifactExpectedFiles["flux1-schnell-q8"],
          scratch: preflightScratch,
          python,
          cacheRoot: guard.cacheRoot,
          missingStore,
        });
      }
      for (const [artifactId, audit] of sourceAudits) {
        audit.expectedFiles = artifactExpectedFiles[artifactId];
        audit.downloadedFiles = missingDownload?.id === artifactId
          ? missingDownload.downloadedFiles : [];
      }
      lifetimePlan = authorityLifetimePlan(profile, startIndex, sourceAudits);
      const freeBytes = await operations.diskFreeBytes(scratch);
      diskPlan = estimateJitDiskPlan(
        lifetimePlan,
        freeBytes,
        missingDownload?.downloadedFiles.reduce((sum, file) => sum + file.bytes, 0) ?? 0,
        expectedNonModelPaths,
      );
      if (!diskPlan.admitted) {
        fail(`JIT staging disk admission refused: requires ${diskPlan.peakRequiredAdditionalBytes} bytes at cell ${diskPlan.peakOrdinal}, only ${diskPlan.freeBytes} free`);
      }
      if (!missingDownload) {
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
    derivedSidecarRoot, missingStore, expectedNonModelPaths, profile,
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
    missingDownload,
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
      missingDownload: null,
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
  for (let index = startIndex; index < profile.cells.length; index += 1) {
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
        const artifactStageRoot = artifactId === "flux1-schnell-q8" && missingDownload
          ? missingStore : path.join(stagingRoot, artifactId);
        transitionRoots.set(artifactId, artifactStageRoot);
        if (artifactStageRoot !== missingStore) await mkdir(artifactStageRoot);
        const staged = await operations.stageArtifact({
          id: artifactId,
          artifact: profile.artifacts[artifactId],
          expectedFiles: artifactExpectedFiles[artifactId],
          scratch: preflightScratch,
          python,
          cacheRoot: guard.cacheRoot,
          stagingRoot: artifactStageRoot,
          missingStore: artifactId === "flux1-schnell-q8" && missingDownload ? missingStore : null,
        });
        if (JSON.stringify(staged.reusedFiles) !== JSON.stringify(audit.reusedFiles)) {
          fail(`trusted source cache changed after frozen census for ${artifactId}`);
        }
        const expectedDownloaded = missingDownload?.id === artifactId
          ? missingDownload.downloadedFiles : [];
        if (JSON.stringify(staged.downloadedFiles ?? []) !== JSON.stringify(expectedDownloaded)) {
          fail(`offline JIT stage changed the frozen download partition for ${artifactId}`);
        }
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
      try {
        const fluxArtifacts = cell.artifactIds.filter((artifactId) => artifactId.startsWith("flux1-"));
        if (fluxArtifacts.length > 1) fail(`cell ${cell.id} has multiple FLUX sidecar authorities`);
        const expectedNamespace = fluxArtifacts.length === 1
          ? await operations.expectedB646DerivedNamespace(
            sharedArtifacts.get(fluxArtifacts[0]), derivedSidecarRoot,
          ) : null;
        const derivedInspection = await operations.inspectDerivedSidecarRoot(
          derivedSidecarRoot, expectedNamespace,
          { materializeExactEmpty: runtimeFailure !== null && expectedNamespace !== null },
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
          if (artifactId.startsWith("flux1-") && !exactEmptyProviderFailure && (files !== 494
              || bytes < (artifactId.endsWith("-q4") ? 7_396_392_960 : 12_573_868_032)
              || bytes > (artifactId.endsWith("-q4")
                ? FLUX_Q4_SIDECAR_BYTES : FLUX_Q8_SIDECAR_BYTES))) {
            fail(`FLUX derived-sidecar contract drifted for ${artifactId}: ${files} files/${bytes} bytes`);
          }
          if (!artifactId.startsWith("flux1-") && (files !== 0 || bytes !== 0)) {
            fail(`non-FLUX authority unexpectedly created derived sidecars: ${artifactId}`);
          }
          lifecycle.derivedAfter.push({ artifactId, files, bytes, inventories });
        }
        derivedSidecarLifecycle.afterCells.push({
          ordinal: index + 1,
          cellId: cell.id,
          providerExecution: lifecycle.providerExecution,
          inventory: derivedInspection.rootInventory,
        });
      } catch (error) {
        quarantineReason = `derived Candle sidecar evidence failed after cell ${cell.id}: ${errorText(error)}`;
        campaignErrors.push(quarantineReason);
        throw error;
      }
      if (runtimeFailure) throw runtimeFailure;
      operations.sample(samples, errors, "post-execution VRAM sample failed");
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
        executionPassed = true;
      } catch (error) {
        quarantineReason = `runtime-result evidence failed after cell ${cell.id}: ${errorText(error)}`;
        campaignErrors.push(quarantineReason);
        throw error;
      }
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
    missingDownload,
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
      missingDownload: null,
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
        startOrdinal: startIndex + 1,
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
    const manifest = JSON.parse(stripJsoncComments(await readFile("config/manifests/builtin.models.jsonc", "utf8")));
    validateManifestAuthorities(profile, manifest);
    const downloadEvidence = JSON.parse(await readFile(DOWNLOAD_EVIDENCE_PATH, "utf8"));
    const expected = expectedArtifactFilesFromEvidence(profile, downloadEvidence);
    if (Object.keys(expected).length !== 23) fail("terminal exact filename census is incomplete");
    process.stdout.write(`${PROFILE_NAME}: exactly 19 serialized cells and immutable authorities OK\n`);
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
