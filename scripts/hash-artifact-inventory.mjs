#!/usr/bin/env node

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { appendFile, readdir, stat } from "node:fs/promises";
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

async function inventoryFiles(root, relative = "", signal) {
  signal?.throwIfAborted();
  const directory = path.join(root, relative);
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const child = path.join(relative, entry.name);
    if (entry.isDirectory()) {
      files.push(...await inventoryFiles(root, child, signal));
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

export async function hashArtifactInventory(root, { signal } = {}) {
  const absolute = path.resolve(root);
  const files = await inventoryFiles(absolute, "", signal);
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
