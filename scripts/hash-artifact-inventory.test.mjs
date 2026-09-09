import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { hashArtifactInventory, listCachedArtifactFiles } from "./hash-artifact-inventory.mjs";

test("artifact inventory binds sorted relative paths, sizes, and exact content hashes", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "artifact-inventory-"));
  await mkdir(path.join(root, "nested"));
  await writeFile(path.join(root, "z.bin"), "zeta");
  await writeFile(path.join(root, "nested/a.bin"), "alpha");
  const result = await hashArtifactInventory(root);
  const alpha = createHash("sha256").update("alpha").digest("hex");
  const zeta = createHash("sha256").update("zeta").digest("hex");
  const expected = createHash("sha256")
    .update(["nested/a.bin", "5", alpha].join("\0") + "\n")
    .update(["z.bin", "4", zeta].join("\0") + "\n")
    .digest("hex");
  assert.deepEqual(result, { root, files: 2, bytes: 9, sha256: expected });
});

test("artifact inventory verifies symlink bytes instead of trusting a 64-hex blob name", async () => {
  const base = await mkdtemp(path.join(tmpdir(), "artifact-inventory-symlink-"));
  const root = path.join(base, "snapshot");
  const blobs = path.join(base, "blobs");
  await mkdir(root);
  await mkdir(blobs);
  const claimed = "a".repeat(64);
  const blob = path.join(blobs, claimed);
  await writeFile(blob, "first");
  await symlink(blob, path.join(root, "weights.safetensors"));
  const first = await hashArtifactInventory(root);
  assert.notEqual(first.sha256, claimed);
  await writeFile(blob, "other");
  const mutated = await hashArtifactInventory(root);
  assert.notEqual(mutated.sha256, first.sha256);
});

test("artifact inventory honors controller cancellation before reading model bytes", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "artifact-inventory-abort-"));
  await writeFile(path.join(root, "weights.safetensors"), "weights");
  const controller = new AbortController();
  controller.abort(new Error("operator interrupt"));
  await assert.rejects(
    () => hashArtifactInventory(root, { signal: controller.signal }),
    /operator interrupt/,
  );
});

test("artifact inventory can exclude downloader metadata without excluding artifact bytes", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "artifact-inventory-metadata-"));
  await mkdir(path.join(root, ".cache"));
  await writeFile(path.join(root, ".cache/transport.json"), "ephemeral");
  await writeFile(path.join(root, "weights.safetensors"), "weights");
  const result = await hashArtifactInventory(root, { excludeDirectories: [".cache"] });
  assert.equal(result.files, 1);
  assert.equal(result.bytes, 7);
  await assert.rejects(
    () => hashArtifactInventory(root, { excludeDirectories: ["../outside"] }),
    /confined relative directory/,
  );
});

test("artifact inventory hashes only the cache provisioner's exact allow-listed files", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "artifact-inventory-allow-list-"));
  await writeFile(path.join(root, "weights.safetensors"), "weights");
  await writeFile(path.join(root, "unreviewed.bin"), "not authority");
  const result = await hashArtifactInventory(root, { includeFiles: ["weights.safetensors"] });
  assert.equal(result.files, 1);
  assert.equal(result.bytes, 7);
  await assert.rejects(
    () => hashArtifactInventory(root, { includeFiles: ["../escape.bin"] }),
    /confined relative file/,
  );
  await assert.rejects(
    () => hashArtifactInventory(root, { includeFiles: ["weights.safetensors", "weights.safetensors"] }),
    /must not contain duplicates/,
  );
});

test("cached artifact listing permits confined HF blob links and rejects escaping links", async (t) => {
  const trusted = await mkdtemp(path.join(tmpdir(), "artifact-cache-links-"));
  const selected = path.join(trusted, "models--owner--model", "snapshots", "a".repeat(40), "q8");
  const blobs = path.join(trusted, "models--owner--model", "blobs");
  await mkdir(selected, { recursive: true });
  await mkdir(blobs, { recursive: true });
  const blob = path.join(blobs, "b".repeat(64));
  await writeFile(blob, "trusted weights");
  const link = path.join(selected, "weights.safetensors");
  try {
    await symlink(blob, link);
  } catch (error) {
    if (error.code === "EPERM") {
      t.diagnostic("symlink/reparse fixture is unavailable on this host");
      return;
    }
    throw error;
  }
  assert.deepEqual(await listCachedArtifactFiles(selected, trusted), ["weights.safetensors"]);

  const outside = await mkdtemp(path.join(tmpdir(), "artifact-cache-outside-"));
  const escaped = path.join(outside, "escaped.safetensors");
  await writeFile(escaped, "outside weights");
  await symlink(escaped, path.join(selected, "escaped.safetensors"));
  await assert.rejects(
    () => listCachedArtifactFiles(selected, trusted),
    /broken, empty, or escaped/,
  );
});

// sc-22738 — a snapshot's own `.gitattributes` is not artifact content, and hashing it made the
// receipt depend on HOW the operator obtained the snapshot rather than on the weights: a whole-repo
// hub fetch carries it, a glob-scoped one does not, and `candle-gen-sdxl`'s `collect_files` REFUSES
// to seal any source containing a dot-prefixed file at all.
test("artifact inventory ignores dotfiles and dot-directories, so the receipt is the weights", async () => {
  const clean = await mkdtemp(path.join(tmpdir(), "artifact-inventory-clean-"));
  const trusted = await mkdtemp(path.join(tmpdir(), "artifact-inventory-dot-"));
  const polluted = path.join(trusted, "snapshot");
  await mkdir(polluted);
  for (const root of [clean, polluted]) {
    await mkdir(path.join(root, "unet"));
    await writeFile(path.join(root, "unet/model.safetensors"), "weights");
    await writeFile(path.join(root, "model_index.json"), "{}");
  }
  await writeFile(path.join(polluted, ".gitattributes"), "*.safetensors filter=lfs\n");
  await mkdir(path.join(polluted, ".cache", "huggingface", "download"), { recursive: true });
  await writeFile(path.join(polluted, ".cache/huggingface/download/model.metadata"), "meta");
  await writeFile(path.join(polluted, "unet/.gitattributes"), "nested\n");

  const before = await hashArtifactInventory(clean);
  const after = await hashArtifactInventory(polluted);
  assert.equal(after.files, 2, "only the two artifact files are inventoried");
  assert.equal(after.bytes, before.bytes, "a dotfile adds no bytes to the receipt");
  assert.equal(after.sha256, before.sha256, "the same weights mint the same receipt on either host");

  // `listCachedArtifactFiles` deliberately does NOT filter: it is the tamper check over a staged
  // authority, and the CUDA harness's Candle sidecar obstructions are dot-named files it installs
  // inside the artifact on purpose and re-verifies through this listing.
  assert.deepEqual(
    await listCachedArtifactFiles(polluted, trusted),
    [
      ".cache/huggingface/download/model.metadata", ".gitattributes",
      "model_index.json", "unet/.gitattributes", "unet/model.safetensors",
    ],
  );
});
