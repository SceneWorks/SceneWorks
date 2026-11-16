// Fetch the exact inference pin's Windows/Linux FlashAttention archive before a Candle build.
// No CUDA or model execution. Missing releases fall back to the vendored source build.
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { appendFile, lstat, mkdir, mkdtemp, readFile, readdir, rename, rm } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { execFileSync, spawnSync } from "node:child_process";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { lockedInferenceRevision } from "./select-windows-cargo-cache.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const names = new Set(["libflashattention.a", "flash-attn-prebuilt.txt", "LICENSE-MIT", "LICENSE-APACHE", "LICENSE-FLASH-ATTENTION", "LICENSE-CUTLASS.txt", "AUTHORS-FLASH-ATTENTION"]);
const execute = (command, args) => execFileSync(command, args, { encoding: "utf8", timeout: 15000 }).trim();

export async function candleFlashSourceSha256(workspace = root) {
  // Cargo resolves the exact git-pinned crate and supplies its physical checkout path.
  // This may fetch dependencies on a clean local machine, just as cargo build would.
  const metadata = JSON.parse(execFileSync("cargo", ["metadata", "--locked", "--format-version", "1", "--features", "sceneworks-worker/backend-candle"], {
    cwd: workspace, encoding: "utf8", maxBuffer: 32 * 1024 * 1024, timeout: 300000,
    env: { ...process.env, CARGO_NET_OFFLINE: "false" },
  }));
  const packages = metadata.packages.filter(pkg => pkg.name === "candle-flash-attn" && pkg.source?.includes("github.com/SceneWorks/inference"));
  if (packages.length !== 1) throw new Error("expected one pinned candle-flash-attn source");
  const crate = path.dirname(packages[0].manifest_path);
  const files = ["build.rs", "Cargo.toml", ...[...names].filter(name => name.startsWith("LICENSE-") || name.startsWith("AUTHORS-")),
    ...(await readdir(path.join(crate, "kernels"))).map(name => `kernels/${name}`)].sort();
  const digest = createHash("sha256");
  for (const name of files) {
    const fullPath = path.join(crate, name);
    const info = await lstat(fullPath);
    if (!info.isFile() || info.isSymbolicLink()) continue;
    digest.update(name);
    digest.update(Buffer.from([0]));
    digest.update(await readFile(fullPath));
    digest.update(Buffer.from([0]));
  }
  return digest.digest("hex");
}

export function parseManifest(text) {
  const fields = Object.create(null);
  for (const line of text.trimEnd().split(/\r?\n/)) {
    const separator = line.indexOf("=");
    if (separator < 1) throw new Error("invalid FlashAttention manifest");
    const key = line.slice(0, separator);
    if (Object.hasOwn(fields, key)) throw new Error("duplicate FlashAttention manifest key");
    fields[key] = line.slice(separator + 1);
  }
  return fields;
}

export function validateManifest(fields, { sourceSha256, target, archiveSha256 }) {
  if (fields.format !== "1" || fields.source_sha256 !== sourceSha256 || fields.target !== target ||
      !/^[a-f0-9]{40}$/.test(fields.inference_revision ?? "") ||
      fields.cuda_compute_cap !== "80" || !/^[a-f0-9]{64}$/.test(fields.source_sha256 ?? "") ||
      !/^[a-f0-9]{64}$/.test(fields.archive_sha256 ?? "") || !/^\d+\.\d+$/.test(fields.cuda_version ?? "") ||
      !/^\d+(?:\.\d+)+$/.test(fields.host_compiler_version ?? "")) {
    throw new Error("FlashAttention manifest does not match the pinned revision and target");
  }
  if (archiveSha256 !== fields.archive_sha256) throw new Error("FlashAttention archive checksum mismatch");
}

async function fileSha256(file) {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(file)) digest.update(chunk);
  return digest.digest("hex");
}

async function validateDirectory(directory, sourceSha256, target) {
  const entries = await readdir(directory);
  if (entries.length !== names.size || entries.some(name => !names.has(name))) throw new Error("FlashAttention package has unexpected files");
  for (const name of entries) {
    const file = await lstat(path.join(directory, name));
    if (!file.isFile() || file.isSymbolicLink()) throw new Error("FlashAttention package contains a non-file");
  }
  const fields = parseManifest(await readFile(path.join(directory, "flash-attn-prebuilt.txt"), "utf8"));
  validateManifest(fields, { sourceSha256, target, archiveSha256: await fileSha256(path.join(directory, "libflashattention.a")) });
}

async function download(url, destination) {
  const response = await fetch(url, { signal: AbortSignal.timeout(300000) });
  if (response.status === 404) return false;
  if (!response.ok || !response.body) throw new Error(`FlashAttention download failed: HTTP ${response.status}`);
  await pipeline(Readable.fromWeb(response.body), createWriteStream(destination, { flags: "wx" }));
  return true;
}

export async function fetchPrebuilt({ sourceSha256, target, cacheRoot, downloadAsset = download }) {
  if (!/^[0-9a-f]{64}$/.test(sourceSha256)) throw new Error("invalid FlashAttention source digest");
  if (!["x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu"].includes(target)) return null;
  const destination = path.join(cacheRoot, sourceSha256, target);
  const present = await lstat(destination).then(() => true, error => {
    if (error.code === "ENOENT") return false;
    throw error;
  });
  if (present) {
    await validateDirectory(destination, sourceSha256, target);
    return destination;
  }
  const parent = path.dirname(destination);
  await mkdir(parent, { recursive: true });
  const temporary = await mkdtemp(path.join(parent, ".flash-attn-"));
  try {
    const asset = `flash-attn-${target}.tar.gz`;
    const url = `https://github.com/SceneWorks/inference/releases/download/flash-attn-${sourceSha256}/${asset}`;
    if (!await downloadAsset(url, path.join(temporary, asset))) return null;
    const listed = execute("tar", ["-tzf", path.join(temporary, asset)]).split(/\r?\n/);
    if (listed.length !== names.size || listed.some(name => !names.has(name)) || new Set(listed).size !== names.size) {
      throw new Error("FlashAttention package has unexpected archive entries");
    }
    const unpacked = path.join(temporary, "unpacked");
    await mkdir(unpacked);
    execFileSync("tar", ["-xzf", path.join(temporary, asset), "-C", unpacked], { timeout: 120000 });
    await validateDirectory(unpacked, sourceSha256, target);
    try {
      await rename(unpacked, destination);
    } catch (error) {
      if (error.code !== "EEXIST" && error.code !== "ENOTEMPTY") throw error;
      await validateDirectory(destination, sourceSha256, target);
    }
    return destination;
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

async function main() {
  const separator = process.argv.indexOf("--");
  const options = separator < 0 ? process.argv.slice(2) : process.argv.slice(2, separator);
  const command = separator < 0 ? [] : process.argv.slice(separator + 1);
  if (options.some(option => option !== "--github-env") || (options.includes("--github-env") && command.length)) {
    throw new Error("usage: node scripts/fetch-flash-prebuilt.mjs [--github-env] [-- cargo ...]");
  }
  const revision = lockedInferenceRevision(await readFile(path.join(root, "Cargo.lock"), "utf8"));
  const sourceSha256 = await candleFlashSourceSha256();
  const target = process.env.CARGO_BUILD_TARGET || /^host: (.+)$/m.exec(execute("rustc", ["-vV"]))?.[1];
  const cacheRoot = process.env.CANDLE_FLASH_ATTN_PREBUILT_CACHE || path.join(homedir(), ".cache", "sceneworks", "flash-attn");
  const directory = await fetchPrebuilt({ sourceSha256, target, cacheRoot });
  if (directory) {
    console.error(`FlashAttention prebuilt: ${directory}`);
    if (options.includes("--github-env")) {
      if (!process.env.GITHUB_ENV) throw new Error("GITHUB_ENV is required");
      await appendFile(process.env.GITHUB_ENV, `CANDLE_FLASH_ATTN_PREBUILT_DIR=${directory}\n`);
    }
  } else {
    console.error(`FlashAttention prebuilt unavailable for inference ${revision}, source ${sourceSha256}, ${target}; compiling from source`);
  }
  if (command.length) {
    const result = spawnSync(command[0], command.slice(1), { stdio: "inherit", env: { ...process.env, ...(directory ? { CANDLE_FLASH_ATTN_PREBUILT_DIR: directory } : {}) } });
    process.exitCode = result.status ?? 1;
  } else if (!directory) {
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch(error => { console.error(`FlashAttention prebuilt: ${error.message}`); process.exitCode = 2; });
}
