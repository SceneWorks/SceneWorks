#!/usr/bin/env node

import { execFile, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { constants as fsConstants, createReadStream } from "node:fs";
import {
  chmod, copyFile, link, lstat, mkdtemp, mkdir, readFile, readdir, realpath, rm,
  stat, unlink, writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

import { hashArtifactInventory } from "./hash-artifact-inventory.mjs";

const execFileAsync = promisify(execFile);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const MAX_FOOTPRINT_BYTES = 53_347_146_863;
export const MAX_RUNTIME_SECONDS = 1_800;
export const CHILD_ATTESTATION_TIMEOUT_SECONDS = 30;
export const MIN_MEMORY_FREE_PERCENT = 70;
export const MIN_SWAP_FREE_BYTES = 1024 ** 3;
export const MIN_PREFLIGHT_FREE_BYTES = MAX_FOOTPRINT_BYTES * 2;
export const MIN_RUNTIME_FREE_BYTES = MAX_FOOTPRINT_BYTES;
const PROVIDER = "ltx_2_3";
const FINGERPRINT = "sc-19109-ltx-2-3-mlx-memory-ladder-v1";
const FIXTURE = "ltx-2-3-mlx-q4-256x256-f9-fps24-seed1234-safety-canary";
const ARTIFACT_REPOSITORY = "SceneWorks/ltx-2.3-mlx";
const ARTIFACT_REVISION = "01df27d308466533aa09d251e3aebdcc627d07eb";
const Q4_INVENTORY_BYTES = 20_467_690_460;
const Q4_INVENTORY_SHA256 = "4e811932e87bb258f642ada790525e36ef2a55959c520e755f1807caf6fa225a";
const TEXT_ENCODER_INVENTORY_FILES = 17;
const TEXT_ENCODER_INVENTORY_BYTES = 26_427_894_918;
const TEXT_ENCODER_INVENTORY_SHA256 = "abde2d155aa8991747cc2999d40688d29a50261c080c0d51fac20357653928d7";

function fail(message) {
  throw new Error(message);
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

export function privateArtifactRoots(scratch) {
  const snapshotRoot = path.join(
    scratch,
    "artifacts",
    `models--${ARTIFACT_REPOSITORY.replaceAll("/", "--")}`,
    "snapshots",
    ARTIFACT_REVISION,
  );
  return {
    numericTier: path.join(snapshotRoot, "q4"),
    textEncoder: path.join(snapshotRoot, "gemma"),
  };
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

export function canaryRequest(memoryBytes, textEncoderInventory) {
  assertInventory(textEncoderInventory, {
    files: TEXT_ENCODER_INVENTORY_FILES,
    bytes: TEXT_ENCODER_INVENTORY_BYTES,
    sha256: TEXT_ENCODER_INVENTORY_SHA256,
  }, "immutable text-encoder");
  return {
    action: "canary",
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
        geometry: { width: 256, height: 256, batch: 1, frames: 9 },
      },
      backend: "mlx",
      loadShape: "eager_materialization",
      strategy: {
        rung: "bounded_decode",
        engagedRungs: ["resident", "staged_residency", "bounded_decode"],
        parameters: { decodeTileEdge: 192, decodeOverlap: 64 },
      },
      calibrationFingerprint: FINGERPRINT,
      fixture: FIXTURE,
      _watchdog: { maxFootprintBytes: MAX_FOOTPRINT_BYTES },
      _canary: { videoMode: "no_audio", fps: 24, seed: 1234 },
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
) {
  if (response?.status !== "diagnostic_canary_complete"
      || response?.diagnosticOnly !== true
      || response?.promotable !== false
      || response?.ingestible !== false) {
    fail("adapter response is not a non-promotable diagnostic canary");
  }
  if (response?.calibrationFingerprint !== FINGERPRINT
      || (expectedInferenceRevision !== undefined
        && response?.inferenceRevision !== expectedInferenceRevision)
      || response?.target?.provider !== PROVIDER
      || response?.target?.tier !== "q4"
      || response?.target?.geometry?.width !== 256
      || response?.target?.geometry?.height !== 256
      || response?.target?.geometry?.frames !== 9
      || response?.target?.geometry?.fps !== 24
      || response?.target?.audio !== false) {
    fail("adapter response changed the exact canary identity");
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
      || response?.watchdog?.required !== true
      || response?.watchdog?.protocol !== "sceneworks-memory-watchdog-v1"
      || response?.watchdog?.maxFootprintBytes !== MAX_FOOTPRINT_BYTES
      || response?.watchdog?.maxRuntimeSeconds !== MAX_RUNTIME_SECONDS
      || !Number.isSafeInteger(response?.watchdog?.hostMemoryBytes)
      || (expectedHostMemoryBytes !== undefined
        && response?.watchdog?.hostMemoryBytes !== expectedHostMemoryBytes)
      || response?.watchdog?.minInitialMemoryFreeBytes
        !== preflightFreeFloor(response?.watchdog?.hostMemoryBytes)
      || response?.watchdog?.minInitialMemoryFreePercent !== MIN_MEMORY_FREE_PERCENT
      || response?.watchdog?.minMemoryFreeBytes
        !== runtimeFreeFloor(response?.watchdog?.hostMemoryBytes)
      || response?.watchdog?.minSwapFreeBytes !== MIN_SWAP_FREE_BYTES
      || response?.mlxLimits?.memoryLimitBytes !== MAX_FOOTPRINT_BYTES
      || !Number.isInteger(response?.mlxLimits?.wiredLimitBytes)
      || response.mlxLimits.wiredLimitBytes <= 0
      || response.mlxLimits.wiredLimitBytes > MAX_FOOTPRINT_BYTES
      || response?.observedMemory?.postCleanupActiveBytes !== 0
      || response?.observedMemory?.postCleanupCacheBytes !== 0
      || response?.output?.firstFrameNondegenerate !== true) {
    fail("adapter response did not attest the exact bounded canary execution");
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
    if ([
      "cargo", "rustc", "clang", "clang++", "cc", "c++", "cmake", "ninja", "metal", "air-lld",
      "sceneworks-worker", "memory-mlx-adapter", "Runner.Worker",
    ].includes(executable)
        || executable.includes("real_weights")
        || executable === "real_weight_tiling"
        || (isPython && /(?:MiniMax|minimax)/.test(command))) {
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
  if (memoryFreePercent < MIN_MEMORY_FREE_PERCENT) {
    fail(`host memory free ${memoryFreePercent}% is below ${MIN_MEMORY_FREE_PERCENT}%`);
  }
  if (swapFreeBytes < MIN_SWAP_FREE_BYTES) {
    fail(`host swap free ${swapFreeBytes} is below ${MIN_SWAP_FREE_BYTES}`);
  }
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
  const resolutionEnv = Object.fromEntries([
    "HOME", "RUSTUP_HOME", "SSL_CERT_FILE", "SSL_CERT_DIR",
  ].flatMap((name) => process.env[name] === undefined ? [] : [[name, process.env[name]]]));
  resolutionEnv.PATH = [
    path.join(home, ".cargo/bin"), "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin",
  ].join(":");
  const rustup = (await execFileAsync("/usr/bin/which", ["rustup"], {
    cwd: ROOT, encoding: "utf8", env: resolutionEnv, timeout: 2_000, signal,
  })).stdout.trim();
  if (!path.isAbsolute(rustup)) fail("the sanitized PATH did not resolve absolute rustup");
  const rustupReal = await realpath(rustup);
  if (!(await stat(rustupReal)).isFile()) fail("resolved rustup is not a regular file");
  const channel = await repositoryToolchain();
  const cargo = (await execFileAsync(rustupReal, [
    "which", "--toolchain", channel, "cargo",
  ], { cwd: ROOT, encoding: "utf8", env: resolutionEnv, timeout: 2_000, signal })).stdout.trim();
  const rustc = (await execFileAsync(rustupReal, [
    "which", "--toolchain", channel, "rustc",
  ], { cwd: ROOT, encoding: "utf8", env: resolutionEnv, timeout: 2_000, signal })).stdout.trim();
  if (!path.isAbsolute(cargo) || !path.isAbsolute(rustc)) {
    fail("rustup did not resolve absolute pinned toolchain executables");
  }
  const version = (await execFileAsync(rustc, ["-Vv"], {
    cwd: ROOT, encoding: "utf8", env: resolutionEnv, timeout: 2_000, signal,
  })).stdout;
  if (!new RegExp(`^release: ${channel.replaceAll(".", "\\.")}$`, "m").test(version)) {
    fail(`resolved rustc does not match pinned channel ${channel}`);
  }
  const privateTemp = path.join(scratch, "tmp");
  await mkdir(privateTemp, { mode: 0o700 });
  return {
    cargo,
    channel,
    env: {
      ...resolutionEnv,
      CARGO_BUILD_JOBS: "2",
      CARGO_HOME: path.join(scratch, "cargo-home"),
      CARGO_TARGET_DIR: path.join(scratch, "target"),
      CARGO_TERM_COLOR: "never",
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
        await execFileAsync("/bin/cp", ["-c", cloneSource, output], {
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

export async function inferenceCargoSource(
  cargo, cargoEnv, inferenceRepo, inferenceRevision, signal,
) {
  const metadata = JSON.parse((await execFileAsync(cargo, [
    "metadata", "--locked", "--format-version", "1",
  ], {
    cwd: ROOT, encoding: "utf8", env: cargoEnv, timeout: 5 * 60 * 1000,
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

async function adapterIdentity(adapterPath, signal) {
  const metadata = await stat(adapterPath, { bigint: true });
  return {
    path: adapterPath,
    sha256: await sha256(adapterPath, signal),
    device: metadata.dev.toString(),
    inode: metadata.ino.toString(),
    size: Number(metadata.size),
  };
}

async function assertAdapterIdentity(expected, signal) {
  const actual = await adapterIdentity(expected.path, signal);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    fail("canary adapter identity changed after its exact build");
  }
}

async function buildExactAdapter(
  sceneWorksRevision, inferenceRepo, inferenceRevision, scratch, signal,
) {
  const toolchain = await exactToolchain(scratch, signal);
  const { cargo } = toolchain;
  const cargoEnv = toolchain.env;
  const target = cargoEnv.CARGO_TARGET_DIR;
  const cargoSource = await inferenceCargoSource(
    cargo, cargoEnv, inferenceRepo, inferenceRevision, signal,
  );
  await execFileAsync(cargo, [
    "build", "--locked", "--release", "-p", "sceneworks-memory-adapter",
    "--bin", "memory-mlx-adapter", "--features", "mlx",
  ], {
    cwd: ROOT,
    encoding: "utf8",
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
  const adapterDirectory = path.join(scratch, "executable");
  await mkdir(adapterDirectory, { mode: 0o700 });
  const adapterPath = path.join(adapterDirectory, "memory-mlx-adapter");
  await copyFile(built, adapterPath, fsConstants.COPYFILE_EXCL);
  await chmod(adapterPath, 0o500);
  await chmod(adapterDirectory, 0o500);
  return adapterIdentity(adapterPath, signal);
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

async function controller(argv) {
  if (process.platform !== "darwin") fail("LTX safety canary requires Darwin phys_footprint telemetry");
  const options = parseArgs(argv);
  for (const name of ["inference-repo", "output"]) {
    if (!options[name]) fail(`--${name} is required`);
  }
  const unexpected = Object.keys(options).filter((name) => !["inference-repo", "output"].includes(name));
  if (unexpected.length) fail(`unsupported option(s): ${unexpected.join(", ")}`);
  const inferenceRepo = path.resolve(options["inference-repo"]);
  const output = path.resolve(options.output);
  const relativeOutput = path.relative(ROOT, output);
  if (relativeOutput === "" || (!relativeOutput.startsWith("..") && !path.isAbsolute(relativeOutput))) {
    fail("--output must stay outside the SceneWorks repository");
  }
  const sceneWorksRevision = await cleanHead(ROOT, "SceneWorks");
  const inferenceRevision = await cleanHead(inferenceRepo, "inference");
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
  const preBuildHost = await assertHostPreflight(memoryBytes);
  const numericTierRoot = await realpath(path.resolve(
    process.env.SCENEWORKS_LTX_ROOT ?? fail("SCENEWORKS_LTX_ROOT is required"),
  ));
  const textEncoderRoot = await realpath(path.resolve(
    process.env.SCENEWORKS_LTX_TEXT_ENCODER_ROOT
      ?? fail("SCENEWORKS_LTX_TEXT_ENCODER_ROOT is required"),
  ));
  const numericTierInventory = assertInventory(
    await hashArtifactInventory(numericTierRoot),
    { files: 11, bytes: Q4_INVENTORY_BYTES, sha256: Q4_INVENTORY_SHA256 },
    "local q4",
  );
  const textEncoderInventory = assertInventory(
    await hashArtifactInventory(textEncoderRoot),
    {
      files: TEXT_ENCODER_INVENTORY_FILES,
      bytes: TEXT_ENCODER_INVENTORY_BYTES,
      sha256: TEXT_ENCODER_INVENTORY_SHA256,
    },
    "local text-encoder",
  );
  const scratch = await mkdtemp(path.join(tmpdir(), "sc19741-ltx-canary-"));
  let activeWatchdog = null;
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
    const scratchDevice = (await stat(scratch, { bigint: true })).dev;
    const {
      numericTier: privateNumericTierRoot,
      textEncoder: privateTextEncoderRoot,
    } = privateArtifactRoots(scratch);
    await mkdir(path.dirname(privateNumericTierRoot), { recursive: true, mode: 0o700 });
    await cloneArtifactTree(numericTierRoot, privateNumericTierRoot, scratchDevice, signal);
    await cloneArtifactTree(textEncoderRoot, privateTextEncoderRoot, scratchDevice, signal);
    assertInventory(
      await hashArtifactInventory(privateNumericTierRoot, { signal }),
      inventoryAtRoot(numericTierInventory, privateNumericTierRoot), "private q4 clone",
    );
    assertInventory(
      await hashArtifactInventory(privateTextEncoderRoot, { signal }),
      inventoryAtRoot(textEncoderInventory, privateTextEncoderRoot),
      "private text-encoder clone",
    );
    const adapter = await buildExactAdapter(
      sceneWorksRevision, inferenceRepo, inferenceRevision, scratch, signal,
    );
    const preLaunchHost = await assertHostPreflight(memoryBytes, signal);
    assertInventory(
      await hashArtifactInventory(numericTierRoot, { signal }), numericTierInventory,
      "pre-launch q4",
    );
    assertInventory(
      await hashArtifactInventory(textEncoderRoot, { signal }), textEncoderInventory,
      "pre-launch text-encoder",
    );
    assertInventory(
      await hashArtifactInventory(privateNumericTierRoot, { signal }),
      inventoryAtRoot(numericTierInventory, privateNumericTierRoot),
      "pre-launch private q4",
    );
    assertInventory(
      await hashArtifactInventory(privateTextEncoderRoot, { signal }),
      inventoryAtRoot(textEncoderInventory, privateTextEncoderRoot),
      "pre-launch private text-encoder",
    );
    await assertAdapterIdentity(adapter, signal);
    const requestPath = path.join(scratch, "request.json");
    const responsePath = path.join(scratch, "response.json");
    const eventsPath = path.join(scratch, "watchdog.jsonl");
    const request = canaryRequest(memoryBytes, textEncoderInventory);
    await writeFile(requestPath, `${JSON.stringify(request)}\n`, { flag: "wx" });
    const watchdog = path.join(ROOT, "scripts/memory-calibration-watchdog.py");
    const runtimeMemoryFreeFloor = runtimeFreeFloor(memoryBytes);
    signal.throwIfAborted();
    const status = await new Promise((resolve, reject) => {
      const childProcess = spawn("/usr/bin/python3", [
      watchdog,
      "--max-footprint-bytes", String(MAX_FOOTPRINT_BYTES),
      "--max-runtime-seconds", String(MAX_RUNTIME_SECONDS),
      "--host-memory-bytes", String(memoryBytes),
      "--min-memory-free-bytes", String(runtimeMemoryFreeFloor),
      "--min-swap-free-bytes", String(MIN_SWAP_FREE_BYTES),
      "--sample-interval", "0.25",
      "--telemetry-timeout", "1",
      "--child-attestation-timeout", String(CHILD_ATTESTATION_TIMEOUT_SECONDS),
      "--term-grace", "1",
      "--event-file", eventsPath,
      "--require-child-attestation",
      "--",
      "/bin/sh", "-c", 'set -C; exec "$1" <"$2" >"$3"',
      "sc19741-canary", adapter.path, requestPath, responsePath,
      ], {
        stdio: "inherit",
        env: {
          ...process.env,
          SCENEWORKS_LTX_ROOT: privateNumericTierRoot,
          SCENEWORKS_LTX_TEXT_ENCODER_ROOT: privateTextEncoderRoot,
        },
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
    await assertAdapterIdentity(adapter, signal);
    assertInventory(
      await hashArtifactInventory(numericTierRoot, { signal }), numericTierInventory, "post-run q4",
    );
    assertInventory(
      await hashArtifactInventory(textEncoderRoot, { signal }), textEncoderInventory,
      "post-run text-encoder",
    );
    assertInventory(
      await hashArtifactInventory(privateNumericTierRoot, { signal }),
      inventoryAtRoot(numericTierInventory, privateNumericTierRoot),
      "post-run private q4",
    );
    assertInventory(
      await hashArtifactInventory(privateTextEncoderRoot, { signal }),
      inventoryAtRoot(textEncoderInventory, privateTextEncoderRoot),
      "post-run private text-encoder",
    );
    if (await cleanHead(ROOT, "SceneWorks", signal) !== sceneWorksRevision
        || await cleanHead(inferenceRepo, "inference", signal) !== inferenceRevision) {
      fail("verified source checkout changed during the canary");
    }
    const response = validateCanaryResponse(
      JSON.parse(await readFile(responsePath, "utf8")),
      inferenceRevision,
      textEncoderInventory,
      memoryBytes,
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
        || event.swapFreeBytes < MIN_SWAP_FREE_BYTES)
      || events.some((event) => event.event === "hard_stop" || event.event === "terminated")) {
      fail("watchdog event stream is incomplete or contains a hard stop");
    }
    const maxObservedFootprintBytes = Math.max(...events
      .filter((event) => event.event === "sample")
      .map((event) => event.physicalFootprintBytes));
    if (maxObservedFootprintBytes >= MAX_FOOTPRINT_BYTES) fail("successful watchdog stream crossed its stop boundary");
    const receipt = {
    schemaVersion: 1,
    story: "sc-19741",
    diagnosticOnly: true,
    promotable: false,
    ingestible: false,
    sceneWorksRevision,
    inferenceRevision,
    adapter,
    hostPreflight: { preBuild: preBuildHost, preLaunch: preLaunchHost },
    request,
    response,
    watchdog: {
      maxObservedFootprintBytes,
      minimumRuntimeFreeBytes: runtimeMemoryFreeFloor,
      minimumSwapFreeBytes: MIN_SWAP_FREE_BYTES,
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
      await cleanupCanaryScratch(scratch);
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
    await controller(process.argv.slice(2));
  }
} catch (error) {
  process.stderr.write(`${error.stack ?? error}\n`);
  process.exitCode = error instanceof CanaryInterrupted ? error.exitCode : 1;
}
