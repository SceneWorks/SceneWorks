#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream, readFileSync } from "node:fs";
import {
  appendFile, lstat, mkdir, readdir, readFile, realpath, rename, rm, stat, writeFile,
} from "node:fs/promises";
import { freemem, totalmem } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { hashArtifactInventory } from "./hash-artifact-inventory.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

export const PROFILE_NAME = "epic-20738-candle-cuda-terminal-v1";
export const PROFILE_PATH = "config/terminal-evidence/epic-20738-cuda.json";
export const PROFILE_SCHEMA_PATH = "config/terminal-evidence/epic-20738-profile.schema.json";
export const RECEIPT_SCHEMA_PATH = "config/terminal-evidence/epic-20738-receipt.schema.json";
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SCHEMA_VALIDATOR_PATH = path.join(ROOT, "scripts/validate-epic-20738-terminal-schema.py");
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
const EXPECTED_ARTIFACT_SEMANTICS_SHA256 = "f2bb7a77b83ce11cc32c3a1f9639534a67a149bc464a9730fb5c0988b4a03f9e";

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
      if (row.repo !== artifact.repository || row.revision !== artifact.revision) {
        fail(`artifact ${id} identity disagrees with manifest authority`);
      }
      if (JSON.stringify(artifact.allowPatterns) !== JSON.stringify(row.files ?? [])) {
        fail(`artifact ${id} allowPatterns are not the manifest's exact download surface`);
      }
    } else if (authority.kind === "explicitPublicArtifact") {
      // The five-backbone route lands in sc-20747. This terminal story freezes only its immutable
      // public weight tuple; it deliberately does not duplicate or depend on that production route.
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

export function validateDocumentWithSchema(python, schemaPath, documentPath) {
  return run(python, [
    SCHEMA_VALIDATOR_PATH,
    "--schema", path.resolve(ROOT, schemaPath),
    "--document", path.resolve(documentPath),
  ]);
}

async function writeJsonAtomically(file, value, { python, schemaPath } = {}) {
  await mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.tmp-${process.pid}-${randomUUID()}`;
  try {
    await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    if (schemaPath) validateDocumentWithSchema(python, schemaPath, temporary);
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

export async function validateCampaignPaths({ runnerTemp, output, scratch, repositories }) {
  if (!runnerTemp) fail("RUNNER_TEMP is required for terminal evidence confinement");
  const runnerMetadata = await stat(runnerTemp);
  if (!runnerMetadata.isDirectory()) fail("RUNNER_TEMP must be an existing directory");
  const resolvedRunnerTemp = await realpath(runnerTemp);
  const resolvedRepositories = await Promise.all(repositories.map((repository) => realpath(repository)));
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

function pendingArtifact(id, artifact, scratch) {
  const inventoryRoot = path.resolve(scratch, "artifacts", id, artifact.subdirectory);
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

async function provisionArtifact({ id, artifact, scratch, python }) {
  const destination = path.join(scratch, "artifacts", id);
  const requestDir = path.join(scratch, "provision-requests");
  await mkdir(requestDir, { recursive: true });
  const requestPath = path.join(requestDir, `${id}.json`);
  await writeFile(requestPath, `${JSON.stringify({
    id, repository: artifact.repository, revision: artifact.revision,
    subdirectory: artifact.subdirectory, allowPatterns: artifact.allowPatterns, destination,
  }, null, 2)}\n`, "utf8");
  const raw = run(python, ["scripts/provision-epic-20738-terminal-artifact.py", "--request", requestPath], {
    env: {
      ...process.env,
      HF_HUB_DISABLE_IMPLICIT_TOKEN: "1",
      HF_HUB_DISABLE_PROGRESS_BARS: "1",
      HF_HUB_DISABLE_TELEMETRY: "1",
      HF_HUB_VERBOSITY: "error",
      HF_HOME: path.join(scratch, "hf-home"),
    },
    maxBuffer: 8 * 1024 * 1024,
  });
  const result = JSON.parse(raw.split(/\r?\n/).at(-1));
  const selectedRoot = await realpath(result.selectedRoot);
  // huggingface_hub's local-dir transport metadata is not part of the reviewed artifact surface.
  // Keep it available for the downloader, but exclude the root-local .cache directory from the
  // byte inventory when subdirectory "." selects the snapshot root.
  const inventory = await hashArtifactInventory(selectedRoot, { excludeDirectories: [".cache"] });
  return { id, ...artifact, snapshotRoot: result.snapshotRoot, selectedRoot, inventory };
}

function cargoCellArgs() {
  return [
    "test", "--locked", "--release", "-p", "sceneworks-worker", "--features", "backend-candle",
    "epic_20738_terminal_cuda_cell", "--", "--ignored", "--nocapture", "--test-threads=1",
  ];
}

async function executeCell({ sceneworks, cellFile, cellDir, logFile, samples }) {
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

async function refreshEvidence(receipt, cellDir, errors) {
  try {
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
    const fallback = await hashedFiles(cellDir, { exclude: new Set(["receipt.json"]) });
    receipt.logs = fallback.filter((file) => file.path.endsWith(".log"));
  }
}

async function writeEmergencyReceipt({
  args, profile, cell, index, ordinalName, output, scratch, repositories, execution, gpuIdentity,
  systemMemory, error,
}) {
  const cellDir = path.join(output, `${ordinalName}-emergency`);
  await mkdir(cellDir);
  const startedAt = new Date().toISOString();
  const message = `unhandled cell setup/finalization failure: ${errorText(error)}`;
  await writeFile(path.join(cellDir, "controller.log"), `${startedAt} ${message}\n`, "utf8");
  const artifacts = cell.artifactIds.map((id) => pendingArtifact(
    id, profile.artifacts[id], path.join(scratch, ordinalName),
  ));
  const receipt = receiptSkeleton({
    cell, ordinal: index + 1, repositories, artifacts, execution, gpuIdentity, systemMemory,
    startedAt,
  });
  receipt.error = message;
  receipt.cleanup = { attempted: false, completed: true, error: null };
  receipt.hardware.rawVramSamples = gpuIdentity.length
    ? [...gpuIdentity]
    : [{ timestamp: startedAt, raw: message }];
  const evidence = await evidenceFiles(cellDir);
  receipt.logs = evidence.logs;
  receipt.completedAt = new Date().toISOString();
  validateReceipt(receipt, cell, profile);
  await writeJsonAtomically(path.join(cellDir, "receipt.json"), receipt, {
    python: args.python, schemaPath: RECEIPT_SCHEMA_PATH,
  });
  return `${path.basename(cellDir)}/receipt.json`;
}

async function runCampaign(args) {
  if (process.env.SCENEWORKS_ENABLE_EPIC_20738_TERMINAL_CUDA !== "1") {
    fail("terminal hardware execution is opt-in; set SCENEWORKS_ENABLE_EPIC_20738_TERMINAL_CUDA=1 only for the frozen final dispatch");
  }
  validateDocumentWithSchema(args.python, PROFILE_SCHEMA_PATH, args.profile);
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
    repositories: [sceneworks.root, inference.root],
  });
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
  const systemMemory = { totalBytes: totalmem(), availableBytesAtStart: freemem() };
  const receipts = [];
  const campaignErrors = [];
  for (const [index, cell] of profile.cells.entries()) {
    const ordinalName = `${String(index + 1).padStart(2, "0")}-${cell.id}`;
    try {
    let cellDir = path.join(output, ordinalName);
    const cellScratch = path.join(scratch, ordinalName);
    const startedAt = new Date().toISOString();
    const errors = [...startupErrors];
    const artifacts = cell.artifactIds.map((id) => pendingArtifact(id, profile.artifacts[id], cellScratch));
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
      validateReceipt(receipt, cell, profile);
      await writeJsonAtomically(receiptPath, receipt, {
        python: args.python, schemaPath: RECEIPT_SCHEMA_PATH,
      });

      await mkdir(cellScratch);
      scratchCreated = true;
      for (const [artifactIndex, artifactId] of cell.artifactIds.entries()) {
        activeArtifactIndex = artifactIndex;
        const artifact = await provisionArtifact({
          id: artifactId, artifact: profile.artifacts[artifactId], scratch: cellScratch, python: args.python,
        });
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
      sampleGpuBestEffort(samples, errors, "pre-execution VRAM sample failed");
      await executeCell({ sceneworks: sceneworks.root, cellFile, cellDir, logFile, samples });
      sampleGpuBestEffort(samples, errors, "post-execution VRAM sample failed");
      const runtimeResult = JSON.parse(await readFile(path.join(cellDir, "runtime-result.json"), "utf8"));
      if (runtimeResult.requestedTier !== cell.requestedTier
        || runtimeResult.resolvedTier !== cell.requestedTier || runtimeResult.denseFallback !== false) {
        fail(`${cell.id} runtime result did not prove exact-tier/no-fallback execution`);
      }
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
      sampleGpuBestEffort(samples, [], "failure VRAM sample failed");
    } finally {
      receipt.cleanup.attempted = scratchCreated;
      if (scratchCreated) {
        try {
          await safeRemoveTree(cellScratch, guard, `scratch for ${cell.id}`);
          receipt.cleanup.completed = true;
        } catch (error) {
          const message = recordError(errors, "cell scratch cleanup failed", error);
          receipt.cleanup.error = message;
        }
      } else {
        receipt.cleanup.completed = true;
      }

      try {
        if (!cellDirCreated) {
          cellDir = path.join(output, `${ordinalName}-failure`);
          await mkdir(cellDir);
          cellDirCreated = true;
          controllerLog = path.join(cellDir, "controller.log");
          receiptPath = path.join(cellDir, "receipt.json");
        }
        try {
          await appendFile(
            controllerLog,
            `${new Date().toISOString()} finalizing ${cell.id}; errors=${errors.length}\n`,
            "utf8",
          );
        } catch (error) {
          recordError(errors, "controller log finalization failed", error);
          controllerLog = path.join(cellDir, "controller-fallback.log");
          await writeFile(controllerLog, `${new Date().toISOString()} ${errors.join("\n")}\n`, "utf8");
        }
        receipt.artifacts = artifacts;
        receipt.hardware.rawVramSamples = samples;
        await refreshEvidence(receipt, cellDir, errors);
        receipt.status = executionPassed && errors.length === 0 && receipt.cleanup.completed ? "passed" : "failed";
        receipt.error = receipt.status === "passed" ? null : (errors.join("\n") || "cell did not complete");
        receipt.completedAt = new Date().toISOString();
        validateReceipt(receipt, cell, profile);
        try {
          await writeJsonAtomically(receiptPath, receipt, {
            python: args.python, schemaPath: RECEIPT_SCHEMA_PATH,
          });
        } catch (error) {
          recordError(errors, "atomic receipt write failed", error);
          receipt.status = "failed";
          receipt.error = errors.join("\n");
          await appendFile(controllerLog, `${new Date().toISOString()} ${receipt.error}\n`, "utf8");
          await refreshEvidence(receipt, cellDir, errors);
          receipt.error = errors.join("\n");
          receipt.completedAt = new Date().toISOString();
          validateReceipt(receipt, cell, profile);
          await writeJsonAtomically(receiptPath, receipt, {
            python: args.python, schemaPath: RECEIPT_SCHEMA_PATH,
          });
        }
        receipts.push({
          id: cell.id, status: receipt.status,
          receipt: `${path.basename(cellDir)}/receipt.json`, error: receipt.error,
        });
      } catch (error) {
        const message = recordError(errors, "best-effort receipt finalization failed", error);
        receipts.push({ id: cell.id, status: "failed", receipt: null, error: message });
      }
    }
    } catch (error) {
      let receiptPath = null;
      let emergencyError = errorText(error);
      try {
        receiptPath = await writeEmergencyReceipt({
          args, profile, cell, index, ordinalName, output, scratch, repositories, execution, gpuIdentity,
          systemMemory, error,
        });
      } catch (receiptError) {
        emergencyError += `\nemergency receipt failed: ${errorText(receiptError)}`;
      }
      receipts.push({ id: cell.id, status: "failed", receipt: receiptPath, error: emergencyError });
    }
  }

  try {
    await safeRemoveTree(scratch, guard, "campaign scratch");
  } catch (error) {
    campaignErrors.push(recordError([], "campaign scratch cleanup failed", error));
  }

  const summary = {
    schemaVersion: 1, profile: PROFILE_NAME, repositories, execution, receipts,
    passed: receipts.filter((receipt) => receipt.status === "passed").length,
    failed: receipts.filter((receipt) => receipt.status === "failed").length,
    campaignErrors,
  };
  await writeJsonAtomically(path.join(output, "campaign-summary.json"), summary);
  if (summary.failed || campaignErrors.length) {
    fail(`terminal campaign completed all 19 cells with ${summary.failed} cell failure(s) and ${campaignErrors.length} campaign cleanup failure(s)`);
  }
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
    const python = value(argv, "--python");
    validateDocumentWithSchema(python, PROFILE_SCHEMA_PATH, profilePath);
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
    });
    return;
  }
  fail("usage: epic-20738-terminal-cuda-harness.mjs check --python path [--profile path] | run --profile path --sceneworks-repo path --inference-repo path --sceneworks-revision sha40 --inference-revision sha40 --output path --scratch path --python path");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
