#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, readFileSync } from "node:fs";
import {
  appendFile, mkdir, readdir, readFile, realpath, rm, stat, writeFile,
} from "node:fs/promises";
import { freemem, totalmem } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { hashArtifactInventory } from "./hash-artifact-inventory.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

export const PROFILE_NAME = "epic-20738-candle-cuda-terminal-v1";
export const PROFILE_PATH = "config/terminal-evidence/epic-20738-cuda.json";
export const RECEIPT_SCHEMA_PATH = "config/terminal-evidence/epic-20738-receipt.schema.json";
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
    inputs: [], outputs: [], logs: [],
    startedAt,
    completedAt: startedAt,
  };
}

export function validateReceipt(receipt) {
  if (receipt.schemaVersion !== 1 || receipt.profile !== PROFILE_NAME) fail("receipt identity mismatch");
  if (!new Set(["passed", "failed"]).has(receipt.status)) fail("receipt status is invalid");
  if (receipt.cell.ordinal < 1 || receipt.cell.ordinal > EXPECTED_CELLS.length
    || receipt.cell.id !== EXPECTED_CELLS[receipt.cell.ordinal - 1]) {
    fail("receipt cell identity or ordinal drifted from the serialized profile");
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
  if (!Array.isArray(receipt.artifacts) || !receipt.artifacts.length
    || receipt.artifacts.some((artifact) => !artifact.inventory || !SHA40.test(artifact.revision))) {
    fail("receipt artifact authority binding is incomplete");
  }
  if (receipt.status === "passed" && receipt.artifacts.some((artifact) => (
    artifact.inventory.complete !== true || !/^[0-9a-f]{64}$/.test(artifact.inventory.sha256)
  ))) {
    fail("passed receipt artifact inventory is incomplete");
  }
  if (receipt.artifacts.some((artifact) => artifact.inventory.complete === false
    && typeof artifact.inventory.error !== "string")) {
    fail("failed artifact inventory must explain why it is incomplete");
  }
  if (!Array.isArray(receipt.hardware?.gpuIdentity) || !receipt.hardware.gpuIdentity.length
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
    || !Array.isArray(receipt.outputs) || receipt.outputs.some((file) => !hashedFile(file))
    || !Array.isArray(receipt.logs) || !receipt.logs.length
    || receipt.logs.some((file) => !hashedFile(file))) {
    fail("receipt input, output, or log hashes are incomplete");
  }
  return receipt;
}

function pendingArtifact(id, artifact) {
  return {
    id,
    role: artifact.role,
    repository: artifact.repository,
    revision: artifact.revision,
    subdirectory: artifact.subdirectory,
    selectedRoot: null,
    allowPatterns: artifact.allowPatterns,
    inventory: { complete: false, sha256: null, files: 0, bytes: 0, error: "provisioning has not completed" },
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

async function runCampaign(args) {
  if (process.env.SCENEWORKS_ENABLE_EPIC_20738_TERMINAL_CUDA !== "1") {
    fail("terminal hardware execution is opt-in; set SCENEWORKS_ENABLE_EPIC_20738_TERMINAL_CUDA=1 only for the frozen final dispatch");
  }
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
  const output = assertOutsideRepository(args.output, [sceneworks.root, inference.root], "output");
  const scratch = assertOutsideRepository(args.scratch, [sceneworks.root, inference.root], "scratch");
  if (path.resolve(output) === path.resolve(scratch)) fail("output and scratch must be separate");
  await mkdir(output, { recursive: false });
  await mkdir(scratch, { recursive: false });
  const gpuIdentity = sampleGpu();
  const systemMemory = { totalBytes: totalmem(), availableBytesAtStart: freemem() };
  const provisioned = new Map();
  const receipts = [];

  try {
    for (const [index, cell] of profile.cells.entries()) {
      const startedAt = new Date().toISOString();
      const cellDir = path.join(output, `${String(index + 1).padStart(2, "0")}-${cell.id}`);
      await mkdir(cellDir);
      const logFile = path.join(cellDir, "runtime.log");
      const controllerLog = path.join(cellDir, "controller.log");
      await writeFile(controllerLog, `${startedAt} starting ${cell.id}\n`, "utf8");
      const artifacts = cell.artifactIds.map((id) => pendingArtifact(id, profile.artifacts[id]));
      // Seed every provisional receipt with the campaign-start raw sample so an interruption during
      // provisioning still leaves schema-valid hardware evidence for this cell.
      const samples = [...gpuIdentity];
      const receiptPath = path.join(cellDir, "receipt.json");
      const receipt = receiptSkeleton({
        cell, ordinal: index + 1, repositories, artifacts, execution, gpuIdentity, systemMemory,
        startedAt,
      });
      receipt.hardware.rawVramSamples = samples;
      let activeArtifactIndex = null;
      receipt.logs = (await evidenceFiles(cellDir)).logs;
      await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, "utf8");
      try {
        for (const [artifactIndex, artifactId] of cell.artifactIds.entries()) {
          activeArtifactIndex = artifactIndex;
          if (!provisioned.has(artifactId)) {
            provisioned.set(artifactId, await provisionArtifact({
              id: artifactId, artifact: profile.artifacts[artifactId], scratch, python: args.python,
            }));
          }
          const artifact = provisioned.get(artifactId);
          artifacts[artifactIndex] = {
            id: artifact.id, role: artifact.role, repository: artifact.repository,
            revision: artifact.revision, subdirectory: artifact.subdirectory,
            selectedRoot: artifact.selectedRoot, allowPatterns: artifact.allowPatterns,
            inventory: { ...artifact.inventory, complete: true, error: null },
          };
        }
        activeArtifactIndex = null;
        receipt.artifacts = artifacts;
        await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, "utf8");
        const runtimeCell = {
          ...cell,
          artifacts: artifacts.map(({ id, role, repository, revision, subdirectory, selectedRoot }) => ({
            id, role, repository, revision, subdirectory, root: selectedRoot,
          })),
        };
        const cellFile = path.join(cellDir, "cell.json");
        await writeFile(cellFile, `${JSON.stringify(runtimeCell, null, 2)}\n`, "utf8");
        samples.push(...sampleGpu());
        await executeCell({ sceneworks: sceneworks.root, cellFile, cellDir, logFile, samples });
        samples.push(...sampleGpu());
        const runtimeResult = JSON.parse(await readFile(path.join(cellDir, "runtime-result.json"), "utf8"));
        if (runtimeResult.requestedTier !== cell.requestedTier
          || runtimeResult.resolvedTier !== cell.requestedTier || runtimeResult.denseFallback !== false) {
          fail(`${cell.id} runtime result did not prove exact-tier/no-fallback execution`);
        }
        receipt.status = "passed";
        receipt.error = null;
      } catch (error) {
        receipt.status = "failed";
        receipt.error = error.stack ?? error.message;
        if (activeArtifactIndex !== null && artifacts[activeArtifactIndex].inventory.complete === false) {
          artifacts[activeArtifactIndex].inventory.error = receipt.error;
        }
        await appendFile(controllerLog, `${new Date().toISOString()} ${receipt.error}\n`, "utf8");
        try { samples.push(...sampleGpu()); } catch { /* the controller error log retains the failure */ }
      }
      receipt.completedAt = new Date().toISOString();
      receipt.hardware.rawVramSamples = samples;
      const evidence = await evidenceFiles(cellDir);
      receipt.inputs = evidence.inputs;
      receipt.outputs = evidence.outputs;
      receipt.logs = evidence.logs;
      validateReceipt(receipt);
      await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, "utf8");
      receipts.push({ id: cell.id, status: receipt.status, receipt: path.basename(cellDir) + "/receipt.json" });

      const nextIds = new Set(profile.cells[index + 1]?.artifactIds ?? []);
      for (const artifactId of [...provisioned.keys()]) {
        if (!nextIds.has(artifactId)) {
          const destination = path.join(scratch, "artifacts", artifactId);
          const relative = path.relative(path.join(scratch, "artifacts"), destination);
          if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) fail("artifact cleanup escaped campaign scratch");
          await rm(destination, { recursive: true, force: true });
          provisioned.delete(artifactId);
        }
      }
    }
  } finally {
    await rm(scratch, { recursive: true, force: true });
  }

  const summary = {
    schemaVersion: 1, profile: PROFILE_NAME, repositories, execution, receipts,
    passed: receipts.filter((receipt) => receipt.status === "passed").length,
    failed: receipts.filter((receipt) => receipt.status === "failed").length,
  };
  await writeFile(path.join(output, "campaign-summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  if (summary.failed) fail(`terminal campaign completed all 19 cells with ${summary.failed} failure(s)`);
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
  fail("usage: epic-20738-terminal-cuda-harness.mjs check [--profile path] | run --profile path --sceneworks-repo path --inference-repo path --sceneworks-revision sha40 --inference-revision sha40 --output path --scratch path --python path");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
