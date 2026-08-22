#!/usr/bin/env node

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { appendFile, lstat, readdir, realpath, stat } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

async function fileSha256(file, signal) {
  // Read every byte even when a Hugging Face snapshot symlink points at a 64-hex blob name. The
  // name is useful cache metadata, but trusting it would let a corrupted or replaced same-size blob
  // retain the old inventory receipt.
  const hash = createHash("sha256");
  signal?.throwIfAborted();
  for await (const chunk of createReadStream(file, { signal })) {
    hash.update(chunk);
    signal?.throwIfAborted();
  }
  return hash.digest("hex");
}

async function inventoryFiles(root, relative = "", signal, excludeDirectories = new Set()) {
  signal?.throwIfAborted();
  const directory = path.join(root, relative);
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const child = path.join(relative, entry.name);
    if (entry.isDirectory()) {
      const normalized = child.split(path.sep).join("/");
      if (!excludeDirectories.has(normalized)) {
        files.push(...await inventoryFiles(root, child, signal, excludeDirectories));
      }
    } else if (entry.isFile() || entry.isSymbolicLink()) {
      const absolute = path.join(root, child);
      const resolved = await stat(absolute);
      files.push({
        path: child.split(path.sep).join("/"),
        bytes: resolved.size,
        sha256: await fileSha256(absolute, signal),
      });
    }
  }
  return files;
}

export async function hashArtifactInventory(
  root,
  { signal, excludeDirectories = [], includeFiles, trustedRoot } = {},
) {
  const absolute = path.resolve(root);
  const excluded = new Set(excludeDirectories.map((directory) => {
    const normalized = directory.split(/[\\/]/).filter(Boolean).join("/");
    if (!normalized || path.isAbsolute(directory) || normalized.split("/").includes("..")) {
      throw new Error(`artifact inventory exclusion must be a confined relative directory: ${directory}`);
    }
    return normalized;
  }));
  if (includeFiles !== undefined && excludeDirectories.length) {
    throw new Error("artifact inventory cannot combine includeFiles with excluded directories");
  }
  let files;
  if (includeFiles !== undefined) {
    if (!Array.isArray(includeFiles) || includeFiles.length === 0) {
      throw new Error("artifact inventory includeFiles must be a non-empty relative file array");
    }
    const normalized = includeFiles.map((file) => {
      const relative = file.split(/[\\/]/).filter(Boolean).join("/");
      if (!relative || path.isAbsolute(file) || relative.split("/").includes("..")) {
        throw new Error(`artifact inventory inclusion must be a confined relative file: ${file}`);
      }
      return relative;
    });
    if (new Set(normalized).size !== normalized.length) {
      throw new Error("artifact inventory includeFiles must not contain duplicates");
    }
    const resolvedRoot = await realpath(absolute);
    const resolvedTrustedRoot = trustedRoot ? await realpath(path.resolve(trustedRoot)) : resolvedRoot;
    files = [];
    for (const relative of normalized.sort()) {
      signal?.throwIfAborted();
      const file = path.join(absolute, ...relative.split("/"));
      const metadata = await lstat(file);
      if (!metadata.isFile() && !metadata.isSymbolicLink()) {
        throw new Error(`artifact inventory inclusion must resolve from a file: ${relative}`);
      }
      const resolved = await realpath(file);
      const relation = path.relative(resolvedTrustedRoot, resolved);
      if (!relation || relation.startsWith("..") || path.isAbsolute(relation)) {
        throw new Error(`artifact inventory inclusion escaped its trusted root: ${relative}`);
      }
      const resolvedMetadata = await stat(resolved);
      if (!resolvedMetadata.isFile() || resolvedMetadata.size < 1) {
        throw new Error(`artifact inventory inclusion is missing or empty: ${relative}`);
      }
      files.push({
        path: relative,
        bytes: resolvedMetadata.size,
        sha256: await fileSha256(resolved, signal),
      });
    }
  } else {
    files = await inventoryFiles(absolute, "", signal, excluded);
  }
  if (files.length === 0) throw new Error(`artifact inventory is empty: ${absolute}`);
  const bytes = files.reduce((total, file) => total + file.bytes, 0);
  const hash = createHash("sha256");
  for (const file of files) {
    hash.update(file.path);
    hash.update("\0");
    hash.update(String(file.bytes));
    hash.update("\0");
    hash.update(file.sha256);
    hash.update("\n");
  }
  return { root: absolute, files: files.length, bytes, sha256: hash.digest("hex") };
}

export async function listCachedArtifactFiles(root, trustedRoot) {
  const absolute = path.resolve(root);
  const boundary = await realpath(path.resolve(trustedRoot));
  const resolvedRoot = await realpath(absolute);
  if (!path.relative(boundary, resolvedRoot)
    || path.relative(boundary, resolvedRoot).startsWith("..")
    || path.isAbsolute(path.relative(boundary, resolvedRoot))) {
    throw new Error("cached artifact root escaped its trusted cache root");
  }
  const files = [];
  async function visit(directory, relative = "") {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
      const childRelative = path.join(relative, entry.name);
      const candidate = path.join(absolute, childRelative);
      const metadata = await lstat(candidate);
      if (entry.isDirectory() && !entry.isSymbolicLink()) {
        const resolved = await realpath(candidate);
        const relation = path.relative(resolvedRoot, resolved);
        if (!relation || relation.startsWith("..") || path.isAbsolute(relation)) {
          throw new Error(`cached artifact directory escaped its selected root: ${childRelative}`);
        }
        await visit(resolved, childRelative);
      } else if (entry.isFile() || entry.isSymbolicLink()) {
        const resolved = await realpath(candidate);
        const relation = path.relative(boundary, resolved);
        const resolvedMetadata = await stat(resolved);
        if (!relation || relation.startsWith("..") || path.isAbsolute(relation)
          || !resolvedMetadata.isFile() || resolvedMetadata.size < 1) {
          throw new Error(`cached artifact file is broken, empty, or escaped: ${childRelative}`);
        }
        files.push(childRelative.split(path.sep).join("/"));
      } else {
        throw new Error(`cached artifact contains a non-regular entry: ${childRelative}`);
      }
    }
  }
  await visit(resolvedRoot);
  return files.sort();
}

function value(args, flag) {
  const index = args.indexOf(flag);
  return index === -1 ? undefined : args[index + 1];
}

async function main() {
  const args = process.argv.slice(2);
  const root = value(args, "--root");
  if (!root) throw new Error("usage: hash-artifact-inventory.mjs --root <directory> [--github-env <file>]");
  const inventory = await hashArtifactInventory(root);
  const githubEnv = value(args, "--github-env");
  if (githubEnv) {
    await appendFile(
      githubEnv,
      `SCENEWORKS_MEMORY_MODEL_BYTES=${inventory.bytes}\n` +
        `SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256=${inventory.sha256}\n`,
      "utf8",
    );
  }
  process.stdout.write(`${JSON.stringify(inventory)}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
