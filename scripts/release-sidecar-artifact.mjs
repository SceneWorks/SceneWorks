#!/usr/bin/env node
// Transfer the exact CUDA sidecar from a secret-free candidate build to release packaging.
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createReadStream } from "node:fs";
import { appendFile, copyFile, mkdir, open, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const targets = {
  windows: "x86_64-pc-windows-msvc",
  linux: "x86_64-unknown-linux-gnu",
};
const command = (name, args) => execFileSync(name, args, { cwd: repoRoot, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 }).trim();
async function sha256(file) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}
export function artifactBinary(dir, platform) {
  if (!targets[platform]) throw new Error(`unsupported release sidecar platform: ${platform}`);
  return path.join(dir, platform === "windows" ? "sceneworks-api.exe" : "sceneworks-api");
}
export function cudaReleaseLine(output) {
  const line = output.split(/\r?\n/).find((candidate) => /^Cuda compilation tools, release 12\.9,/.test(candidate.trim()));
  if (!line) throw new Error("release sidecar requires CUDA 12.9 nvcc");
  return line.trim();
}
export function selectCandidateRun(runs, sourceSha, platform) {
  if (!targets[platform] || !/^[0-9a-f]{40}$/.test(sourceSha)) throw new Error("invalid sidecar lookup input");
  return runs.find((run) => run.head_sha === sourceSha && run.head_branch === "release/next" &&
    run.path === ".github/workflows/release-sidecars.yml" && run.event === "push" &&
    run.status === "completed" && run.conclusion === "success") ?? null;
}
export function verifyReceiptFields(receipt, expected, binaryBytes, binarySha256) {
  for (const [key, value] of Object.entries(expected)) {
    if (receipt[key] !== value) throw new Error(`release sidecar ${key} mismatch: ${receipt[key]} != ${value}`);
  }
  if (typeof receipt.nvcc !== "string" || !/release 12\.9,/.test(receipt.nvcc)) {
    throw new Error("release sidecar lacks CUDA 12.9 compiler provenance");
  }
  if (typeof receipt.host_compiler !== "string" || !receipt.host_compiler.trim()) {
    throw new Error("release sidecar lacks host compiler provenance");
  }
  if (!Number.isSafeInteger(receipt.binary_bytes) || receipt.binary_bytes < 1 ||
      receipt.binary_bytes !== binaryBytes || receipt.binary_sha256 !== binarySha256) {
    throw new Error("release sidecar binary size or SHA-256 mismatch");
  }
}
async function verifyExecutableFormat(binary, platform) {
  const file = await open(binary, "r");
  try {
    const header = Buffer.alloc(4);
    const { bytesRead } = await file.read(header, 0, 4, 0);
    if (bytesRead !== 4 || (platform === "windows" ? header.toString("ascii", 0, 2) !== "MZ" :
      !header.equals(Buffer.from([0x7f, 0x45, 0x4c, 0x46])))) {
      throw new Error(`release sidecar is not a ${platform} executable`);
    }
  } finally { await file.close(); }
}
function checkoutSha() {
  const sourceSha = command("git", ["rev-parse", "HEAD"]);
  if (process.env.GITHUB_SHA && process.env.GITHUB_SHA !== sourceSha) {
    throw new Error(`checkout ${sourceSha} differs from workflow source ${process.env.GITHUB_SHA}`);
  }
  return sourceSha;
}
function sourceIdentity() {
  return { source_sha: checkoutSha(), rustc: command("rustc", ["--version"]) };
}
async function expectedIdentity(platform) {
  if (!targets[platform]) throw new Error(`unsupported release sidecar platform: ${platform}`);
  if (process.env.CUDA_COMPUTE_CAP !== "80") throw new Error("release sidecar requires CUDA_COMPUTE_CAP=80");
  return {
    schema: 1,
    platform,
    target: targets[platform],
    features: "embed-web,backend-candle",
    compute_cap: "80",
    ...sourceIdentity(),
    cargo_lock_sha256: await sha256(path.join(repoRoot, "Cargo.lock")),
  };
}
export async function sealReleaseSidecar(platform, builtBinary, artifactDir) {
  const identity = await expectedIdentity(platform);
  if (command("rustc", ["-vV"]).match(/^host:\s*(\S+)/m)?.[1] !== identity.target) {
    throw new Error("release sidecar builder has the wrong Rust host target");
  }
  const nvcc = cudaReleaseLine(command("nvcc", ["--version"]));
  const hostCompiler = platform === "windows" ? process.env.VCToolsVersion : command("g++", ["--version"]).split("\n")[0];
  if (!hostCompiler?.trim()) throw new Error("release sidecar requires host compiler provenance");
  await mkdir(artifactDir, { recursive: true });
  const binary = artifactBinary(artifactDir, platform);
  await copyFile(builtBinary, binary);
  const info = await stat(binary);
  if (!info.isFile() || info.size === 0) throw new Error("empty release sidecar binary");
  await verifyExecutableFormat(binary, platform);
  const receipt = {
    ...identity,
    nvcc,
    host_compiler: hostCompiler.trim(),
    binary_bytes: info.size,
    binary_sha256: await sha256(binary),
    producer_run_id: process.env.GITHUB_RUN_ID ?? "local",
  };
  await writeFile(path.join(artifactDir, "receipt.json"), JSON.stringify(receipt, null, 2) + "\n", { flag: "wx" });
  return receipt;
}
export async function verifyReleaseSidecar(platform, artifactDir) {
  const expected = await expectedIdentity(platform);
  const receipt = JSON.parse(await readFile(path.join(artifactDir, "receipt.json"), "utf8"));
  if (process.env.SCENEWORKS_RELEASE_SIDECAR_RUN_ID &&
      receipt.producer_run_id !== process.env.SCENEWORKS_RELEASE_SIDECAR_RUN_ID) {
    throw new Error("release sidecar producer run mismatch");
  }
  const binary = artifactBinary(artifactDir, platform);
  const info = await stat(binary);
  if (!info.isFile()) throw new Error("release sidecar artifact is not a file");
  await verifyExecutableFormat(binary, platform);
  verifyReceiptFields(receipt, expected, info.size, await sha256(binary));
  return binary;
}
async function findRun(platform) {
  const sourceSha = checkoutSha();
  const repository = process.env.GITHUB_REPOSITORY;
  if (repository !== "SceneWorks/SceneWorks" || !process.env.GH_TOKEN) {
    throw new Error("release sidecar lookup requires the SceneWorks repository and GH_TOKEN");
  }
  const uri = `repos/${repository}/actions/runs?branch=release%2Fnext&event=push&head_sha=${sourceSha}&per_page=100`;
  const runs = JSON.parse(command("gh", ["api", uri])).workflow_runs ?? [];
  const run = selectCandidateRun(runs, sourceSha, platform);
  if (!run) throw new Error(`no successful release-sidecars run for ${sourceSha}; wait for its release/next push run before tagging`);
  const artifacts = JSON.parse(command("gh", ["api", `repos/${repository}/actions/runs/${run.id}/artifacts?per_page=100`])).artifacts ?? [];
  if (!artifacts.some((item) => item.name === `release-sidecar-${platform}` && !item.expired)) {
    throw new Error(`release-sidecars run ${run.id} has no live ${platform} sidecar artifact`);
  }
  if (!process.env.GITHUB_OUTPUT) throw new Error("GITHUB_OUTPUT is required");
  await appendFile(process.env.GITHUB_OUTPUT, `run-id=${run.id}\n`);
  console.log(`Using release-sidecars run ${run.id} for ${sourceSha} (${platform})`);
}
async function main() {
  const [mode, platform, first, second] = process.argv.slice(2);
  if (mode === "seal" && first && second) {
    const receipt = await sealReleaseSidecar(platform, first, second);
    console.log(`Sealed ${platform} sidecar ${receipt.binary_sha256}`);
  } else if (mode === "verify" && first && !second) {
    console.log(`Verified ${await verifyReleaseSidecar(platform, first)}`);
  } else if (mode === "find" && !first) {
    await findRun(platform);
  } else {
    throw new Error("usage: release-sidecar-artifact.mjs seal PLATFORM BINARY ARTIFACT_DIR | verify PLATFORM ARTIFACT_DIR | find PLATFORM");
  }
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
