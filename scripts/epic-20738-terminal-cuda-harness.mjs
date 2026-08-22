#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream, readFileSync } from "node:fs";
import {
  appendFile, cp, lstat, mkdir, readdir, readFile, realpath, rename, rm, stat, writeFile,
} from "node:fs/promises";
import { freemem, totalmem } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";

import { hashArtifactInventory, listCachedArtifactFiles } from "./hash-artifact-inventory.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

export const PROFILE_NAME = "epic-20738-candle-cuda-terminal-v1";
export const PROFILE_PATH = "config/terminal-evidence/epic-20738-cuda.json";
export const PROFILE_SCHEMA_PATH = "config/terminal-evidence/epic-20738-profile.schema.json";
export const RECEIPT_SCHEMA_PATH = "config/terminal-evidence/epic-20738-receipt.schema.json";
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SCHEMA_VALIDATORS = new Map();
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
  const files = [];
  async function visit(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
      const file = path.join(directory, entry.name);
      const relative = path.relative(absolute, file).split(path.sep).join("/");
      if (exclude.has(relative)) continue;
      if (entry.isDirectory()) await visit(file);
      else if (entry.isFile()) {
        const metadata = await stat(file);
        files.push({ path: relative, bytes: metadata.size, sha256: await sha256File(file) });
      }
    }
  }
  await visit(absolute);
  return files;
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

export function validateReceipt(receipt, expectedCell, profile) {
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
  if (receipt.status === "passed" && (receipt.error !== null || !receipt.cleanup.completed
    || receipt.cleanup.error !== null)) {
    fail("passed receipt contains a failure or incomplete cleanup");
  }
  if (receipt.status === "failed" && (typeof receipt.error !== "string" || !receipt.error)) {
    fail("failed receipt must retain an error");
  }
  return receipt;
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
    validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt);
    validateReceipt(receipt, cell, legacyProfile);
    if (receipt.status !== "passed"
      || receipt.repositories.sceneworks.sha !== LEGACY_SCENEWORKS_HEAD
      || receipt.repositories.inference.sha !== FROZEN_INFERENCE_PIN
      || receipt.execution.headSha !== LEGACY_SCENEWORKS_HEAD
      || receipt.execution.runId !== metadata.runId
      || receipt.execution.runAttempt !== metadata.runAttempt) {
      fail(`imported prefix receipt ${ordinalName} is not an exact PASS from the bound old run`);
    }
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
  validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, boundaryReceipt);
  validateReceipt(boundaryReceipt, boundaryCell, legacyProfile);
  const boundaryLog = (await hashedFiles(boundaryDir, { exclude: new Set(["receipt.json"]) }));
  if (boundaryReceipt.status !== "failed"
    || boundaryReceipt.error !== "cell has not completed"
    || boundaryReceipt.repositories.sceneworks.sha !== LEGACY_SCENEWORKS_HEAD
    || boundaryReceipt.repositories.inference.sha !== FROZEN_INFERENCE_PIN
    || boundaryReceipt.execution.headSha !== LEGACY_SCENEWORKS_HEAD
    || boundaryReceipt.execution.runId !== metadata.runId
    || boundaryReceipt.execution.runAttempt !== metadata.runAttempt
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

async function writeArtifactRequest({ id, artifact, scratch, phase }) {
  const requestDir = path.join(scratch, `${phase}-requests`);
  await mkdir(requestDir, { recursive: true });
  const requestPath = path.join(requestDir, `${id}.json`);
  await writeFile(requestPath, `${JSON.stringify({
    id, repository: artifact.repository, revision: artifact.revision,
    subdirectory: artifact.subdirectory, allowPatterns: artifact.allowPatterns,
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

async function auditArtifact({ id, artifact, scratch, python, cacheRoot }) {
  const requestPath = await writeArtifactRequest({ id, artifact, scratch, phase: "audit" });
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
  id, artifact, scratch, python, cacheRoot, stagingRoot, allowReviewedDownload,
}) {
  const requestPath = await writeArtifactRequest({ id, artifact, scratch, phase: "stage" });
  const raw = run(python, [
    "scripts/provision-epic-20738-terminal-artifact.py",
    "--request", requestPath,
    "--cache-root", cacheRoot,
    "--stage-root", stagingRoot,
    ...(allowReviewedDownload ? ["--allow-reviewed-download"] : []),
  ], {
    env: {
      ...process.env,
      HF_HUB_OFFLINE: allowReviewedDownload ? "0" : "1",
      TRANSFORMERS_OFFLINE: allowReviewedDownload ? "0" : "1",
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

async function provisionArtifact({ id, artifact, scratch, python, cacheRoot }) {
  const requestPath = await writeArtifactRequest({ id, artifact, scratch, phase: "offline" });
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

async function verifyArtifactUnchanged(artifact, cacheRoot) {
  const selectedFiles = await listCachedArtifactFiles(artifact.selectedRoot, cacheRoot);
  if (JSON.stringify(selectedFiles) !== JSON.stringify(artifact.selectedFiles)) {
    fail(`shared cache selected-file set mutated for ${artifact.id}`);
  }
  const inventory = await hashArtifactInventory(artifact.selectedRoot, {
    includeFiles: artifact.matchedFiles,
    trustedRoot: cacheRoot,
  });
  if (inventory.files !== artifact.inventory.files || inventory.bytes !== artifact.inventory.bytes
    || inventory.sha256 !== artifact.inventory.sha256) {
    fail(`shared cache selected-file bytes mutated for ${artifact.id}`);
  }
}

function cargoCellArgs() {
  return [
    "test", "--locked", "--release", "-p", "sceneworks-worker", "--features", "backend-candle",
    "epic_20738_terminal_cuda_cell", "--", "--ignored", "--nocapture", "--test-threads=1",
  ];
}

async function executeCell({
  sceneworks, cellFile, cellDir, logFile, samples, runtimeScratch,
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
  try {
    await invokeFault(fault, "evidenceHash", index);
    const evidence = await evidenceFiles(cellDir);
    receipt.inputs = evidence.inputs;
    receipt.outputs = evidence.outputs;
    receipt.logs = evidence.logs;
    if (!receipt.logs.length) fail("no controller or runtime log was available to hash");
  } catch (error) {
    const message = recordError(errors, "evidence hashing failed", error);
    receipt.inputs = [];
    receipt.outputs = [];
    const fallbackLog = path.join(cellDir, "evidence-fallback.log");
    await writeFile(fallbackLog, `${new Date().toISOString()} ${message}\n`, "utf8");
    await invokeFault(fault, "evidenceRehash", index);
    const fallback = await hashedFiles(cellDir, { exclude: new Set(["receipt.json"]) });
    receipt.logs = fallback.filter((file) => file.path.endsWith(".log"));
  }
}

async function writePrimaryReceipt({ file, receipt, cell, profile, fault, index }) {
  await invokeFault(fault, "semanticValidation", index);
  validateReceipt(receipt, cell, profile);
  await invokeFault(fault, "schemaValidation", index);
  validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt);
  await invokeFault(fault, "receiptWrite", index);
  await writeJsonAtomically(file, receipt);
}

async function writeEmergencyReceipt({
  profile, cell, index, ordinalName, output, cacheRoot, repositories, execution, gpuIdentity,
  systemMemory, error, cleanup,
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
  receipt.cleanup = cleanup ?? { attempted: false, completed: true, error: null };
  receipt.hardware.rawVramSamples = gpuIdentity.length
    ? [...gpuIdentity]
    : [{ timestamp: startedAt, raw: message }];
  const evidence = await evidenceFiles(cellDir);
  receipt.logs = evidence.logs;
  receipt.completedAt = new Date().toISOString();
  validateReceipt(receipt, cell, profile);
  validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt);
  await writeJsonAtomically(path.join(cellDir, "receipt.json"), receipt);
  return `_emergency/${ordinalName}/receipt.json`;
}

async function writeLastResortEmergencyReceipt({
  profile, cell, index, ordinalName, output, cacheRoot, repositories, execution, gpuIdentity,
  systemMemory, errors, cleanup,
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
  receipt.cleanup = cleanup;
  receipt.hardware.rawVramSamples = gpuIdentity.length
    ? [...gpuIdentity]
    : [{ timestamp: startedAt, raw: receipt.error }];
  receipt.logs = [{
    path: "controller-last-resort.log", bytes: metadata.size, sha256: await sha256File(logFile),
  }];
  receipt.completedAt = new Date().toISOString();
  validateReceipt(receipt, cell, profile);
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
    };
  }
  const {
    profile, repositories, execution, guard, output, scratch, startupErrors, gpuIdentity,
    systemMemory, sceneworksRoot, prefixCandidates, python,
  } = prepared;
  const fault = options.fault;
  const operations = {
    auditArtifact: options.operations?.auditArtifact ?? auditArtifact,
    stageArtifact: options.operations?.stageArtifact ?? stageArtifact,
    provisionArtifact: options.operations?.provisionArtifact ?? provisionArtifact,
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
  await mkdir(preflightScratch);
  await mkdir(stagingRoot);
  const remainingArtifactIds = [...new Set(
    profile.cells.slice(startIndex).flatMap((cell) => cell.artifactIds),
  )];
  const sourceAudits = new Map();
  const cacheProvisioning = { sourceCensus: [], staging: [], finalOffline: [] };
  const censusErrors = [];
  for (const artifactId of remainingArtifactIds) {
    try {
      const artifact = await operations.auditArtifact({
        id: artifactId,
        artifact: profile.artifacts[artifactId],
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
  if (censusErrors.length) {
    quarantineReason = `source cache census failed before transfer: ${censusErrors.join(" | ")}`;
  } else if (frozenMissing.length > 1
    || (frozenMissing.length === 1
      && JSON.stringify(frozenMissing) !== JSON.stringify(reviewedMissing))) {
    quarantineReason = `source cache census found an unapproved missing-file set: ${JSON.stringify(frozenMissing)}`;
  }

  const stagingErrors = [];
  if (!quarantineReason) {
    for (const artifactId of remainingArtifactIds) {
      try {
        const audit = sourceAudits.get(artifactId);
        const allowReviewedDownload = audit.missingFiles.length === 1;
        const staged = await operations.stageArtifact({
          id: artifactId,
          artifact: profile.artifacts[artifactId],
          scratch: preflightScratch,
          python,
          cacheRoot: guard.cacheRoot,
          stagingRoot,
          allowReviewedDownload,
        });
        const downloaded = staged.downloadedFiles ?? [];
        if (JSON.stringify(staged.reusedFiles) !== JSON.stringify(audit.reusedFiles)) {
          fail(`trusted source cache changed after frozen census for ${artifactId}`);
        }
        if ((!allowReviewedDownload && downloaded.length)
          || (allowReviewedDownload && (downloaded.length !== 1
            || downloaded[0].path !== "transformer/model.safetensors"
            || !/^[0-9a-f]{64}$/.test(downloaded[0].sha256)
            || downloaded[0].sha256 !== downloaded[0].lfsSha256
            || downloaded[0].commitSha !== reviewedMissing[0].revision
            || !Number.isSafeInteger(downloaded[0].bytes) || downloaded[0].bytes < 1))) {
          fail(`campaign staging download partition drifted for ${artifactId}`);
        }
        cacheProvisioning.staging.push({
          id: artifactId,
          repository: profile.artifacts[artifactId].repository,
          revision: profile.artifacts[artifactId].revision,
          subdirectory: profile.artifacts[artifactId].subdirectory,
          allowPatterns: profile.artifacts[artifactId].allowPatterns,
          reusedFiles: staged.reusedFiles,
          downloadedFiles: downloaded,
        });
      } catch (error) {
        stagingErrors.push(`${artifactId}: ${errorText(error)}`);
        break;
      }
    }
    if (stagingErrors.length) {
      quarantineReason = `copy-once campaign staging failed: ${stagingErrors.join(" | ")}`;
    }
  }

  const finalErrors = [];
  if (!quarantineReason) {
    for (const artifactId of remainingArtifactIds) {
      try {
        const artifact = await operations.provisionArtifact({
          id: artifactId,
          artifact: profile.artifacts[artifactId],
          scratch: preflightScratch,
          python,
          cacheRoot: stagingRoot,
        });
        sharedArtifacts.set(artifactId, artifact);
        cacheProvisioning.finalOffline.push({
          id: artifactId,
          repository: artifact.repository,
          revision: artifact.revision,
          subdirectory: artifact.subdirectory,
          allowPatterns: artifact.allowPatterns,
          inventory: artifact.inventory,
        });
      } catch (error) {
        finalErrors.push(`${artifactId}: ${errorText(error)}`);
      }
    }
    if (finalErrors.length) {
      quarantineReason = `full offline validation of campaign staging failed: ${finalErrors.join(" | ")}`;
    }
  }
  if (quarantineReason) {
    quarantineReason = `${quarantineReason}; no continuation GPU cell started`;
    campaignErrors.push(quarantineReason);
  }
  const cachePreflightPath = path.join(output, "cache-preflight.json");
  await writeFile(
    cachePreflightPath,
    `${JSON.stringify({
      sourceCacheRoot: guard.cacheRoot,
      campaignStagingRoot: stagingRoot,
      frozenMissingFiles: frozenMissing,
      reusedFiles: cacheProvisioning.staging.flatMap((artifact) => artifact.reusedFiles.map(
        (file) => ({ artifactId: artifact.id, ...file }),
      )),
      downloadedFiles: cacheProvisioning.staging.flatMap((artifact) => artifact.downloadedFiles.map(
        (file) => ({ artifactId: artifact.id, ...file }),
      )),
      phases: cacheProvisioning,
      offlineBeforeCells: !quarantineReason,
    }, null, 2)}\n`,
    { encoding: "utf8", flag: "wx" },
  );
  const cachePreflightMetadata = await stat(cachePreflightPath);
  const cachePreflightEvidence = {
    path: "cache-preflight.json",
    bytes: cachePreflightMetadata.size,
    sha256: await sha256File(cachePreflightPath),
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

    try {
      await mkdir(cellDir);
      cellDirCreated = true;
      const logFile = path.join(cellDir, "runtime.log");
      await writeFile(controllerLog, `${startedAt} starting ${cell.id}\n`, { encoding: "utf8", flag: "wx" });
      receipt.logs = (await evidenceFiles(cellDir)).logs;
      await writePrimaryReceipt({ file: receiptPath, receipt, cell, profile, fault, index });

      await mkdir(cellScratch);
      scratchCreated = true;
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
            stagingRoot,
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
      } catch (error) {
        quarantineReason = `shared cache changed before cell ${cell.id}: ${errorText(error)}`;
        campaignErrors.push(quarantineReason);
        throw error;
      }
      operations.sample(samples, errors, "pre-execution VRAM sample failed");
      await invokeFault(fault, "execute", index);
      let runtimeFailure = null;
      try {
        await operations.executeCell({
          sceneworks: sceneworksRoot,
          cellFile,
          cellDir,
          logFile,
          samples,
          runtimeScratch: cellScratch,
        });
      } catch (error) {
        runtimeFailure = error;
      }
      try {
        await verifyCellCache("after");
      } catch (error) {
        quarantineReason = `shared cache mutated during cell ${cell.id}: ${errorText(error)}`;
        campaignErrors.push(quarantineReason);
        throw error;
      }
      if (runtimeFailure) throw runtimeFailure;
      operations.sample(samples, errors, "post-execution VRAM sample failed");
      const runtimeResult = JSON.parse(await readFile(path.join(cellDir, "runtime-result.json"), "utf8"));
      validateRuntimeResult(runtimeResult, cell);
      receipt.cell.loadSpecQuantBits = runtimeResult.loadSpecQuantBits;
      executionPassed = true;
    } catch (error) {
      const message = recordError(errors, "cell lifecycle failed", error);
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
          controllerLog = path.join(cellDir, "controller-fallback.log");
          await invokeFault(fault, "fallbackLog", index);
          await writeFile(controllerLog, `${new Date().toISOString()} ${errors.join("\n")}\n`, "utf8");
        }
        receipt.artifacts = artifacts;
        receipt.hardware.rawVramSamples = samples;
        await refreshEvidence(receipt, cellDir, errors, { fault, index });
        receipt.status = executionPassed && errors.length === 0 && receipt.cleanup.completed ? "passed" : "failed";
        receipt.error = receipt.status === "passed" ? null : (errors.join("\n") || "cell did not complete");
        receipt.completedAt = new Date().toISOString();
        await writePrimaryReceipt({ file: receiptPath, receipt, cell, profile, fault, index });
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
      try {
        await invokeFault(fault, "emergencyReceipt", index);
        receiptPath = await writeEmergencyReceipt({
          profile, cell, index, ordinalName, output, cacheRoot: stagingRoot,
          repositories, execution, gpuIdentity,
          systemMemory, error: new Error(emergencyError), cleanup: emergencyCleanup,
        });
      } catch (receiptError) {
        emergencyError += `\nemergency receipt failed: ${errorText(receiptError)}`;
        try {
          receiptPath = await writeLastResortEmergencyReceipt({
            profile, cell, index, ordinalName, output, cacheRoot: stagingRoot,
            repositories, execution, gpuIdentity,
            systemMemory, errors: [emergencyError], cleanup: emergencyCleanup,
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
    const profile = validateProfile(loadProfile(profilePath));
    const manifest = JSON.parse(stripJsoncComments(await readFile("config/manifests/builtin.models.jsonc", "utf8")));
    validateManifestAuthorities(profile, manifest);
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
