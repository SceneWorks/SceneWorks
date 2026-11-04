#!/usr/bin/env node
// Historical evidence retrieval only. Never executes a spike or regenerates a measurement.
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { sha256 as sha256Bytes } from "./lib/file-sha256.mjs";
import { lstat, mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

const exec = promisify(execFile);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const MANIFEST = "docs/archive/historical-artifacts.json";
const hex = (value, length) => typeof value === "string" && new RegExp(`^[0-9a-f]{${length}}$`).test(value);
const confined = (value) => typeof value === "string" && value.length > 0
  && !/[\\:\x00-\x1f\x7f]/.test(value)
  && value.split("/").every((part) => part && part !== "." && part !== "..");

export function validateManifest(manifest) {
  if (manifest?.version !== 1 || manifest.source?.repository !== "SceneWorks/SceneWorks"
      || !hex(manifest.source.commit, 40) || !hex(manifest.source.tree, 40)
      || !Array.isArray(manifest.entries) || manifest.entries.length === 0) {
    throw new Error("invalid historical artifact manifest/source");
  }
  const paths = new Set();
  for (const entry of manifest.entries) {
    if (!confined(entry.path) || paths.has(entry.path) || !confined(entry.group)
        || !Number.isSafeInteger(entry.bytes) || entry.bytes <= 0 || !hex(entry.sha256, 64)
        || !hex(entry.gitBlob, 40) || !["100644", "100755"].includes(entry.mode)) {
      throw new Error(`invalid or duplicate historical artifact entry: ${entry.path}`);
    }
    paths.add(entry.path);
  }
  return manifest;
}

export function verifyBytes(entry, bytes) {
  const sha256 = sha256Bytes(bytes);
  const blob = createHash("sha1").update(`blob ${bytes.length}\0`).update(bytes).digest("hex");
  if (bytes.length !== entry.bytes || sha256 !== entry.sha256 || blob !== entry.gitBlob) {
    throw new Error(`historical artifact digest/size mismatch: ${entry.path}`);
  }
}

async function github(endpoint, raw = false) {
  const { stdout } = await exec("gh", ["api", "--method", "GET", ...(raw
    ? ["-H", "Accept: application/vnd.github.raw+json"] : []), endpoint], {
    encoding: "buffer", maxBuffer: 32 * 1024 * 1024, timeout: 60_000,
  });
  return raw ? stdout : JSON.parse(stdout.toString("utf8"));
}

// Check the path-to-object binding at the archived commit, not just an arbitrary blob's digest.
export function verifyTree(manifest, tree) {
  if (tree.truncated || tree.sha !== manifest.source.tree || !Array.isArray(tree.tree)) {
    throw new Error("archive tree is truncated or does not match the pinned source tree");
  }
  const entries = new Map(tree.tree.map((entry) => [entry.path, entry]));
  for (const entry of manifest.entries) {
    const archived = entries.get(entry.path);
    if (archived?.type !== "blob" || archived.sha !== entry.gitBlob
        || archived.mode !== entry.mode || archived.size !== entry.bytes) {
      throw new Error(`archive path/blob/size/mode mismatch: ${entry.path}`);
    }
  }
}

async function destination(root, relative) {
  const target = path.join(root, relative);
  // Reject symlinks, including an existing leaf, before opening anything for writing.
  let current = root;
  for (const part of relative.split("/")) {
    current = path.join(current, part);
    try {
      if ((await lstat(current)).isSymbolicLink()) throw new Error(`refusing symlink: ${current}`);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
  await mkdir(path.dirname(target), { recursive: true });
  const parent = await realpath(path.dirname(target));
  const relation = path.relative(root, parent);
  if (relation.startsWith(`..${path.sep}`) || relation === ".." || path.isAbsolute(relation)) {
    throw new Error(`archive destination escaped output root: ${relative}`);
  }
  return target;
}

export async function restore(manifest, outputRoot, { group, api = github } = {}) {
  validateManifest(manifest);
  const prefix = `repos/${manifest.source.repository}/git`;
  const commit = await api(`${prefix}/commits/${manifest.source.commit}`);
  if (commit.sha !== manifest.source.commit || commit.tree?.sha !== manifest.source.tree) {
    throw new Error("archive commit/tree mismatch");
  }
  verifyTree(manifest, await api(`${prefix}/trees/${manifest.source.tree}?recursive=1`));
  const entries = group ? manifest.entries.filter((entry) => entry.group === group) : manifest.entries;
  if (!entries.length) throw new Error(`no historical artifacts in group: ${group}`);
  await mkdir(outputRoot, { recursive: true });
  const root = await realpath(outputRoot);
  const blobs = new Map();
  for (const entry of entries) {
    if (!blobs.has(entry.gitBlob)) blobs.set(entry.gitBlob, await api(`${prefix}/blobs/${entry.gitBlob}`, true));
    const bytes = blobs.get(entry.gitBlob);
    verifyBytes(entry, bytes);
    const target = await destination(root, entry.path);
    try {
      await writeFile(target, bytes, { flag: "wx", mode: Number.parseInt(entry.mode.slice(-3), 8) });
    } catch (error) {
      if (error.code !== "EEXIST") throw error;
      verifyBytes(entry, await readFile(target));
    }
  }
  return { files: entries.length, bytes: entries.reduce((sum, entry) => sum + entry.bytes, 0), source: manifest.source.commit };
}

export async function verify(manifest, root, group) {
  validateManifest(manifest);
  const entries = group ? manifest.entries.filter((entry) => entry.group === group) : manifest.entries;
  if (!entries.length) throw new Error(`no historical artifacts in group: ${group}`);
  for (const entry of entries) verifyBytes(entry, await readFile(path.join(root, entry.path)));
  return { files: entries.length, bytes: entries.reduce((sum, entry) => sum + entry.bytes, 0), source: manifest.source.commit };
}

async function main() {
  const [command, ...args] = process.argv.slice(2);
  const options = {};
  while (args.length) {
    const flag = args.shift();
    if (!["--root", "--output", "--group"].includes(flag) || !args.length || args[0].startsWith("--")) {
      throw new Error(`unknown or incomplete option: ${flag}`);
    }
    if (options[flag]) throw new Error(`duplicate option: ${flag}`);
    options[flag] = args.shift();
  }
  const manifest = validateManifest(JSON.parse(await readFile(path.join(ROOT, MANIFEST), "utf8")));
  if (command === "restore" && options["--output"] && !options["--root"]) {
    console.log(JSON.stringify(await restore(manifest, path.resolve(options["--output"]), { group: options["--group"] })));
  } else if (command === "verify" && options["--root"] && !options["--output"]) {
    console.log(JSON.stringify(await verify(manifest, path.resolve(options["--root"]), options["--group"])));
  } else {
    throw new Error("usage: historical-artifacts.mjs restore --output DIR [--group GROUP] | verify --root DIR [--group GROUP]");
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
