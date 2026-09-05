#!/usr/bin/env node
// Transport small authorized component configs through an Actions secret.
// Content never enters source control, command arguments, logs, or artifacts.
import { createHash } from "node:crypto";
import { lstat, mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";
const COMPONENTS = ["starcoder1", "starcoder2", "siglip"];
const MAX_BUNDLE_BYTES = 32 * 1024;
const MAX_CONFIG_BYTES = 8 * 1024;
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const fail = message => { throw new Error(`terminal component transport: ${message}`); };
const exactKeys = (value, keys) => value && typeof value === "object" && !Array.isArray(value) && JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...keys].sort());

async function regularPath(file, directory = false) {
  if (!path.isAbsolute(file)) fail("absolute component paths required");
  const parsed = path.parse(file);
  let current = parsed.root;
  for (const segment of file.slice(parsed.root.length).split(path.sep).filter(Boolean)) {
    current = path.join(current, segment);
    if ((await lstat(current)).isSymbolicLink()) fail("component path contains a symlink");
  }
  const info = await lstat(file);
  if (directory ? !info.isDirectory() : !info.isFile()) fail("component path has the wrong type");
  return info;
}

export function validateComponentBundle(payload, lock) {
  if (typeof payload !== "string" || Buffer.byteLength(payload) > MAX_BUNDLE_BYTES) fail("component bundle missing or exceeds 32 KiB");
  let bundle;
  try { bundle = JSON.parse(payload); } catch { fail("component bundle is not JSON"); }
  if (!exactKeys(bundle, ["schema_version", "components"]) || bundle.schema_version !== 1 || !exactKeys(bundle.components, COMPONENTS)) fail("bundle must contain exactly three components");
  const files = new Map(), manifest = {};
  for (const key of COMPONENTS) {
    const entry = bundle.components[key], expected = lock.components?.[key];
    if (!expected || !/^[a-f0-9]{40}$/.test(expected.revision ?? "") || !/^[a-f0-9]{64}$/.test(expected.config_sha256 ?? "")) fail(`committed ${key} revision/hash is not ready`);
    if (!exactKeys(entry, ["repository", "revision", "config_path", "config_sha256", "content_base64"]) || entry.repository !== expected.repository || entry.revision !== expected.revision || entry.config_sha256 !== expected.config_sha256 || entry.config_path !== `${key}.json`) fail(`${key} identity differs from the committed lock`);
    if (typeof entry.content_base64 !== "string") fail(`${key} content missing`);
    const bytes = Buffer.from(entry.content_base64, "base64");
    if (!bytes.length || bytes.length > MAX_CONFIG_BYTES || bytes.toString("base64") !== entry.content_base64 || hash(bytes) !== expected.config_sha256) fail(`${key} config bytes differ from the committed hash`);
    let config;
    try { config = JSON.parse(bytes.toString("utf8")); } catch { fail(`${key} config is not JSON`); }
    if (!config || typeof config !== "object" || Array.isArray(config)) fail(`${key} config must be an object`);
    files.set(entry.config_path, bytes);
    manifest[key] = { repository: entry.repository, revision: entry.revision, config_path: entry.config_path, config_sha256: entry.config_sha256 };
  }
  files.set("components.json", Buffer.from(JSON.stringify(manifest, null, 2) + "\n"));
  return { files, manifest, bundle_sha256: hash(payload) };
}

export async function packComponentBundle(lock, sourceRoot, output) {
  await regularPath(sourceRoot, true);
  const manifestFile = path.join(sourceRoot, "components.json"); await regularPath(manifestFile);
  if ((await lstat(manifestFile)).size > MAX_CONFIG_BYTES) fail("component manifest too large");
  const manifest = JSON.parse(await readFile(manifestFile, "utf8"));
  if (!exactKeys(manifest, COMPONENTS)) fail("source manifest must contain exactly three components");
  const components = {};
  for (const key of COMPONENTS) {
    const entry = manifest[key];
    if (entry.config_path !== `${key}.json`) fail("source config filenames must match component names");
    const file = path.join(sourceRoot, entry.config_path), info = await regularPath(file);
    if (info.size > MAX_CONFIG_BYTES) fail(`${key} config too large`);
    components[key] = { ...entry, content_base64: (await readFile(file)).toString("base64") };
  }
  const payload = JSON.stringify({ schema_version: 1, components });
  const checked = validateComponentBundle(payload, lock);
  await writeFile(output, payload, { flag: "wx", mode: 0o600 });
  return { component_count: COMPONENTS.length, bundle_sha256: checked.bundle_sha256 };
}

export async function installComponentBundle(payload, lock, destination) {
  const checked = validateComponentBundle(payload, lock);
  if (!path.isAbsolute(destination)) fail("absolute destination required");
  await regularPath(path.dirname(destination), true);
  const existing = await lstat(destination).catch(error => error.code === "ENOENT" ? null : Promise.reject(error));
  if (existing) {
    await regularPath(destination, true);
    if (JSON.stringify((await readdir(destination)).sort()) !== JSON.stringify([...checked.files.keys()].sort())) fail("existing component directory has a different inventory");
    for (const [name, bytes] of checked.files) {
      const file = path.join(destination, name); const info = await regularPath(file);
      if (info.size !== bytes.length || hash(await readFile(file)) !== hash(bytes)) fail("existing component bytes differ; preserve the original directory");
    }
  } else {
    const staging = `${destination}.staging-${process.pid}`;
    await mkdir(staging);
    try {
      for (const [name, bytes] of checked.files) await writeFile(path.join(staging, name), bytes, { flag: "wx", mode: 0o600 });
      await rename(staging, destination);
    } finally { await rm(staging, { recursive: true, force: true }); }
  }
  // Re-read durable bytes; return metadata only for logs/readiness records.
  for (const [name, bytes] of checked.files) if (hash(await readFile(path.join(destination, name))) !== hash(bytes)) fail("component readback mismatch");
  return { component_count: COMPONENTS.length, bundle_sha256: checked.bundle_sha256, components: checked.manifest };
}

if (isExecutedModule(import.meta.url)) {
  const [mode, lockPath, first, second] = process.argv.slice(2);
  Promise.resolve().then(async () => {
    const lock = JSON.parse(await readFile(lockPath, "utf8"));
    const result = mode === "pack" ? await packComponentBundle(lock, first, second) : mode === "install" ? await installComponentBundle(process.env.STARVECTOR_UPSTREAM_COMPONENTS_JSON, lock, first) : fail("usage: pack <lock> <source-root> <output> | install <lock> <destination>");
    console.log(JSON.stringify(result));
  }).catch(error => { console.error(error.message); process.exitCode = 1; });
}
