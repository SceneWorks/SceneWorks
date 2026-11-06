// Build-output selection only; no compilation or kernel execution.
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile, realpath, mkdir, appendFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { lockedInferenceRevision } from "./select-windows-cargo-cache.mjs";

const hash = (value) => createHash("sha256").update(value).digest("hex");
const required = (value) => {
  if (typeof value !== "string" || !value.trim() || value.includes("\0")) throw new Error("missing FlashAttention compiler input");
  return value;
};
export function linuxFlashCacheIdentity({ lock, nvcc, hostCompiler, arch, target, nvccPrependFlags = "", nvccAppendFlags = "" }) {
  const pin = lockedInferenceRevision(required(lock));
  if (!/^[0-9]{2,3}$/.test(arch ?? "") || !/^[a-z0-9_]+-unknown-linux-gnu$/.test(target ?? "")) {
    throw new Error("missing or unsupported FlashAttention arch/target");
  }
  if ([nvccPrependFlags, nvccAppendFlags].some((flags) => typeof flags !== "string" || flags.includes("\0"))) {
    throw new Error("invalid NVCC environment flags");
  }
  // The pin covers every kernel/header, build.rs flags and CUTLASS commit.
  // The complete lock also isolates the cudaforge version. No run ID enters.
  return hash(JSON.stringify({ pin, lock: hash(lock), nvcc: required(nvcc), hostCompiler: required(hostCompiler), arch, target, nvccPrependFlags, nvccAppendFlags }));
}
export async function prepareLinuxFlashCache(root, inputs) {
  if (!root || !path.isAbsolute(root)) throw new Error("absolute FlashAttention cache root required");
  const directory = path.join(root, "candle-flash-attn", linuxFlashCacheIdentity(inputs));
  await mkdir(directory, { recursive: true });
  return realpath(directory);
}
// Match cudaforge 0.1.5 toolkit.rs::find_nvcc, including its Linux CUDA_PATH
// nvcc.exe fallback. The pinned Flash build uses detect(), not cuda_root().
export function selectCudaforgeNvcc({ env, pathNvcc, exists }) {
  if (env.NVCC !== undefined && exists(env.NVCC)) return env.NVCC;
  if (pathNvcc) return pathNvcc;
  const candidates = [
    ...(env.CUDA_HOME !== undefined ? [path.posix.join(env.CUDA_HOME, "bin", "nvcc")] : []),
    ...(env.CUDA_PATH !== undefined ? [path.posix.join(env.CUDA_PATH, "bin", "nvcc.exe")] : []),
    ...["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda", "/usr", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA", "C:/CUDA"]
      .map((root) => path.posix.join(root, "bin", "nvcc")),
  ];
  const selected = candidates.find((candidate) => exists(candidate));
  if (!selected) throw new Error("No nvcc found in cudaforge search paths");
  return selected;
}
export async function configureLinuxFlashCache(githubEnv, root, inputs) {
  if (!githubEnv || !path.isAbsolute(githubEnv)) throw new Error("absolute GitHub environment file required");
  const directory = await prepareLinuxFlashCache(root, inputs);
  await appendFile(githubEnv, `CANDLE_FLASH_ATTN_BUILD_DIR=${directory}\n`);
  return directory;
}
const execute = (command, args) => execFileSync(command, args, { encoding: "utf8", timeout: 10000 }).trim();
async function compilerIdentity(command) {
  const binary = await realpath(command.includes("/") ? command : execute("which", [command]));
  return JSON.stringify({ command, binary, version: required(execute(binary, ["--version"])), sha256: hash(await readFile(binary)) });
}
async function main() {
  if (process.platform !== "linux" || !process.env.GITHUB_ENV) throw new Error("Linux Actions environment required");
  const target = process.env.CARGO_BUILD_TARGET || /^host: (.+)$/m.exec(execute("rustc", ["-vV"]))?.[1];
  // NVCC_CCBIN is cudaforge's explicit override; otherwise nvcc uses the
  // standard Linux gcc/g++ drivers from PATH. Record both default identities.
  const hostCompiler = process.env.NVCC_CCBIN
    ? await compilerIdentity(process.env.NVCC_CCBIN)
    : JSON.stringify({ gcc: await compilerIdentity("gcc"), gpp: await compilerIdentity("g++") });
  let pathNvcc;
  try { pathNvcc = execute("which", ["nvcc"]); } catch { /* cudaforge also falls through a failed PATH lookup. */ }
  const nvcc = selectCudaforgeNvcc({ env: process.env, pathNvcc, exists: existsSync });
  const directory = await configureLinuxFlashCache(process.env.GITHUB_ENV, required(process.env.RUNNER_TEMP), {
    lock: await readFile("Cargo.lock", "utf8"), nvcc: await compilerIdentity(nvcc),
    hostCompiler, arch: process.env.CUDA_COMPUTE_CAP, target,
    nvccPrependFlags: process.env.NVCC_PREPEND_FLAGS, nvccAppendFlags: process.env.NVCC_APPEND_FLAGS,
  });
  console.log(`Persistent FlashAttention build directory: ${directory}`);
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
