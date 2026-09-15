import path from "node:path";
import { appendFile, lstat, mkdir, readFile, readdir, realpath } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { pathToFileURL } from "node:url";

const execute = promisify(execFile);
const fail = () => { throw new Error("Windows Cargo cache requires a safe listener identity and physical persistent directory"); };
const absoluteWindows = value => typeof value === "string" && /^[A-Za-z]:\\/.test(value) && !/[\x00-\x1f<>"|?*]/.test(value) && !value.slice(2).includes(":") && !value.split(/[\\/]/).some(part => part === ".." || /[. ]$/.test(part));

export function selectWindowsCargoCache({ runnerName, cargoHome, runnerToolCache }) {
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(runnerName ?? "")) fail();
  // These services have distinct dependency homes; never reuse USERPROFILE/.cargo
  // or a different listener's cache. Unknown installations use the isolated fallback.
  const listener = /^cuda-windows(?:-([2-4]))?$/.exec(runnerName);
  const serviceCache = listener ? `D:\\cargo-home-${listener[1] ?? "1"}` : null;
  if (serviceCache && absoluteWindows(cargoHome) && path.win32.normalize(cargoHome).toLowerCase() === serviceCache.toLowerCase()) {
    return { runner: runnerName, mode: "service-cache", cache: serviceCache };
  }
  if (!absoluteWindows(runnerToolCache)) fail();
  return { runner: runnerName, mode: "listener-cache", cache: path.win32.join(runnerToolCache, "SceneWorks", "cargo", runnerName.toLowerCase()) };
}

export async function ensurePhysicalCargoCache(directory) {
  if (!path.isAbsolute(directory)) fail();
  const segments = [];
  for (let current = path.resolve(directory); ; current = path.dirname(current)) {
    segments.unshift(current);
    if (current === path.dirname(current)) break;
  }
  for (const current of segments) {
    let info;
    try { info = await lstat(current); } catch (error) {
      if (error.code !== "ENOENT") throw error;
      try { await mkdir(current); } catch (createError) { if (createError.code !== "EEXIST") throw createError; }
      info = await lstat(current);
    }
    if (!info.isDirectory() || info.isSymbolicLink()) fail();
  }
  const physical = await realpath(directory);
  const normalized = value => process.platform === "win32" ? path.resolve(value).toLowerCase() : path.resolve(value);
  if (normalized(physical) !== normalized(directory)) fail();
}

export function lockedInferenceRevision(lock) {
  const lines = lock.split(/\r?\n/).filter(line => line.startsWith('source = "git+https://github.com/SceneWorks/inference'));
  const sources = lines.map(line => /^source = "git\+https:\/\/github\.com\/SceneWorks\/inference\?rev=([a-f0-9]{40})#([a-f0-9]{40})"$/.exec(line));
  if (sources.some(match => !match)) throw new Error("Cargo lock must name one exact inference revision");
  const revisions = new Set(sources.map(match => match[1]));
  if (revisions.size !== 1 || sources.some(match => match[1] !== match[2])) throw new Error("Cargo lock must name one exact inference revision");
  return [...revisions][0];
}

export async function cachedInferenceRevision(cache, revision) {
  const databaseRoot = path.join(cache, "git", "db");
  const entries = await readdir(databaseRoot, { withFileTypes: true }).catch(error => {
    if (error.code === "ENOENT") return [];
    throw error;
  });
  for (const entry of entries) {
    if (!/^inference-[a-f0-9]+$/.test(entry.name) || !entry.isDirectory() || entry.isSymbolicLink()) continue;
    try {
      await execute("git", ["--git-dir", path.join(databaseRoot, entry.name), "cat-file", "-e", `${revision}^{commit}`], { timeout: 5000, maxBuffer: 16384 });
      return true;
    } catch { /* A missing object is diagnostic only; cargo fetch remains authoritative. */ }
  }
  return false;
}

async function main() {
  if (process.platform !== "win32") fail();
  const selected = selectWindowsCargoCache({ runnerName: process.env.RUNNER_NAME, cargoHome: process.env.CARGO_HOME, runnerToolCache: process.env.RUNNER_TOOL_CACHE });
  await ensurePhysicalCargoCache(selected.cache);
  const revision = lockedInferenceRevision(await readFile("Cargo.lock", "utf8"));
  const present = await cachedInferenceRevision(selected.cache, revision);
  if (!process.env.GITHUB_ENV) throw new Error("GitHub environment output is required");
  await appendFile(process.env.GITHUB_ENV, `CARGO_HOME=${selected.cache}\n`);
  console.log(JSON.stringify({ kind: "windows-cargo-cache", ...selected, inference_revision: revision, inference_commit_present: present }));
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch(() => { console.error("Windows Cargo cache selection failed; no shared-cache fallback was applied"); process.exitCode = 1; });
}
