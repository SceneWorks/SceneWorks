#!/usr/bin/env node

import { createHash } from "node:crypto";
import {
  lstat,
  mkdir,
  open,
  readdir,
  readlink,
  realpath,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const SC19057_WAN_REPOSITORY = "SceneWorks/wan2.2-ti2v-5b-candle";
export const SC19057_WAN_REVISION = "9b173dc8660334a87a11e67de58939afe68f8cb2";
export const SC19057_WAN_VARIANT = "q4";
export const SC19057_WAN_TOTAL_BYTES = 17_338_835_457;

// Exact immutable revision inventory from the Hugging Face repository. A 64-character
// object is an LFS SHA-256; a 40-character object is a Git blob object ID. Keeping the
// content address beside each logical path prevents a different same-sized snapshot
// from satisfying the terminal preflight.
export const SC19057_WAN_FILES = Object.freeze({
  ".gitattributes": [1_860, "75b8d68701da756cee0d6c9fe3bc7b5ed74955a6"],
  "README.md": [17_633, "f940f55ea91f0d5713b7fa7b42be2590ea17a9e7"],
  "assets/comp_effic.png": [202_156, "75ee012dcfb08365bec67a3ec7afc126fc2817f79b9f80e38711792d4770e32b"],
  "assets/logo.png": [56_322, "0c55854cbd9692975f217714ffd83fd4b37f5dca"],
  "assets/moe_2.png": [527_914, "4ea471ccb64349bd08bc9a78f336ae000e9ca3b40da9a652b8028b214a8c6093"],
  "assets/moe_arch.png": [74_900, "7822af1e65215ee2a9449c9b7616afd713f67a01"],
  "assets/performance.png": [306_535, "97ef99c13c8ae717a8a11c8d8ec927b69077c647cc6689755d08fc38e7fbb830"],
  "assets/vae.png": [165_486, "4aaea5e187f1c5908e15ade5bef24c9fb59882986bc3d2ad75f7fe820f3d772f"],
  "examples/i2v_input.JPG": [250_628, "077e3d965090c9028c69c00931675f42e1acc815c6eb450ab291b3b72d211a8e"],
  "model_index.json": [499, "fe52bfbdc8e5bbc8a6a607dd301f6e1ab0889cd1"],
  "scheduler/scheduler_config.json": [820, "950d26faea717c8902ee197982026cb9c1b6463e"],
  "text_encoder/config.json": [855, "ab4a73bce055c6e32e66133032dcb3adfb26ee8d"],
  "text_encoder/model-00001-of-00003.safetensors": [4_935_812_536, "a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58"],
  "text_encoder/model-00002-of-00003.safetensors": [4_983_103_192, "d57d948ece4837d850b7a859a4415121d57cacf8b9ee1d4db200c67f592902d7"],
  "text_encoder/model-00003-of-00003.safetensors": [1_442_935_480, "0da9ee284e21d1406df708788db1d502d95d75f69faa25cd26151bf8829b7c5f"],
  "text_encoder/model.safetensors.index.json": [22_476, "f3d3d4da90eb33e14c92d88ea346370fa3c0b5b2"],
  "tokenizer/special_tokens_map.json": [7_079, "2ed25bf989a28d20b5d4b5822fbc24666d12a6f7"],
  "tokenizer/spiece.model": [4_548_313, "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"],
  "tokenizer/tokenizer.json": [16_837_459, "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef"],
  "tokenizer/tokenizer_config.json": [61_758, "09d434f9457238f697f4c208aab47f58caa15bfe"],
  "transformer/config.json": [495, "8180887e8ed86e4fd842824fb15fcabf43d19512"],
  "transformer/model.safetensors": [3_135_121_496, "50e683b0f8c76fdb1222677316a464b786b872274178a9323826dc583efc09b1"],
  "transformer/quantize_config.json": [56, "3e239833da45bcb4c2ff32bb2c169512bf7a79aa"],
  "vae/config.json": [1_701, "29f65bc63e9daadb95e4c1a8344d162be4b7d533"],
  "vae/diffusion_pytorch_model.safetensors": [2_818_777_808, "62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b"],
});

function pathKey(value) {
  return path.resolve(value).replaceAll("\\", "/").toLowerCase();
}

function samePath(left, right) {
  return pathKey(left) === pathKey(right);
}

function isInside(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative !== "" && relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

async function listLogicalFiles(root) {
  const files = [];
  async function visit(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      const entryPath = path.join(directory, entry.name);
      const metadata = await lstat(entryPath);
      if (metadata.isDirectory() && !metadata.isSymbolicLink()) {
        await visit(entryPath);
      } else if (metadata.isFile() || metadata.isSymbolicLink()) {
        files.push({ logicalPath: entryPath, logicalMetadata: metadata });
      } else {
        throw new Error(`unsupported artifact entry type at ${entryPath}`);
      }
    }
  }
  await visit(root);
  return files;
}

function rawLinkTargetIsAbsolute(target) {
  return path.posix.isAbsolute(target) || path.win32.isAbsolute(target) || /^[A-Za-z]:/.test(target);
}

function normalizedLinkTarget(link, rawTarget) {
  const platformRelativeTarget = rawTarget.replaceAll(/[\\/]/g, path.sep);
  return path.resolve(path.dirname(link), path.normalize(platformRelativeTarget));
}

export function validateRawLinkTarget({ logicalPath, rawTarget, repositoryRoot, expectedObject }) {
  if (rawLinkTargetIsAbsolute(rawTarget)) {
    throw new Error("artifact link target must be relative, not absolute or drive-qualified");
  }
  const expectedLexicalTarget = path.join(repositoryRoot, "blobs", expectedObject);
  const actualLexicalTarget = normalizedLinkTarget(logicalPath, rawTarget);
  if (!samePath(actualLexicalTarget, expectedLexicalTarget)) {
    throw new Error("artifact link does not name its exact direct repository blob");
  }
  return actualLexicalTarget;
}

async function inspectOpenFile(file) {
  const handle = await open(file, "r");
  try {
    const metadata = await handle.stat();
    if (!metadata.isFile()) {
      throw new Error(`resolved artifact payload is not a regular file: ${file}`);
    }
    const sha256 = createHash("sha256");
    const gitBlob = createHash("sha1");
    gitBlob.update(`blob ${metadata.size}\0`);
    const buffer = Buffer.allocUnsafe(1024 * 1024);
    let streamedBytes = 0;
    while (true) {
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, streamedBytes);
      if (bytesRead === 0) break;
      const chunk = buffer.subarray(0, bytesRead);
      sha256.update(chunk);
      gitBlob.update(chunk);
      streamedBytes += bytesRead;
    }
    return {
      metadata,
      streamedBytes,
      sha256: sha256.digest("hex"),
      gitBlob: gitBlob.digest("hex"),
    };
  } finally {
    await handle.close();
  }
}

function normalizeExpectedFiles(expectedFiles) {
  const entries = Object.entries(expectedFiles);
  const logicalKeys = new Set();
  const objectKeys = new Set();
  for (const [logicalPath, [bytes, object]] of entries) {
    const normalized = logicalPath.replaceAll("\\", "/");
    if (normalized.startsWith("/") || normalized.split("/").includes("..")) {
      throw new Error(`expected inventory path escapes the artifact root: ${logicalPath}`);
    }
    if (logicalKeys.has(normalized.toLowerCase())) {
      throw new Error(`duplicate expected logical artifact path: ${logicalPath}`);
    }
    logicalKeys.add(normalized.toLowerCase());
    if (!Number.isSafeInteger(bytes) || bytes < 0) {
      throw new Error(`invalid expected byte count for ${logicalPath}`);
    }
    if (!/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/.test(object)) {
      throw new Error(`invalid content-addressed cache object for ${logicalPath}`);
    }
    if (objectKeys.has(object)) {
      throw new Error(`duplicate expected content object ${object}`);
    }
    objectKeys.add(object);
  }
  return new Map(entries.map(([logicalPath, value]) => [logicalPath.replaceAll("\\", "/").toLowerCase(), value]));
}

async function writeJson(file, value) {
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

export async function inventorySc19057WanArtifact({
  root,
  evidence,
  expectedFiles = SC19057_WAN_FILES,
  expectedTotalBytes = SC19057_WAN_TOTAL_BYTES,
  identity = {},
}) {
  if (!evidence) throw new Error("SC-19057 evidence directory is required");
  await mkdir(evidence, { recursive: true });
  const receiptPath = path.join(evidence, "wan-q4-inventory-preflight.json");
  const receipt = {
    schemaVersion: 1,
    story: "SC-19057",
    repository: SC19057_WAN_REPOSITORY,
    revision: SC19057_WAN_REVISION,
    variant: SC19057_WAN_VARIANT,
    status: "STARTED",
    ...identity,
  };
  await writeJson(receiptPath, receipt);

  try {
    if (!root) throw new Error("SCENEWORKS_PROVISIONED_ROOT is required");
    const expected = normalizeExpectedFiles(expectedFiles);
    if (expected.size !== Object.keys(expectedFiles).length) {
      throw new Error("expected artifact inventory contains duplicate logical paths");
    }
    const expectedSum = [...expected.values()].reduce((sum, [bytes]) => sum + bytes, 0);
    if (expectedSum !== expectedTotalBytes) {
      throw new Error(`expected artifact inventory totals ${expectedSum}, not ${expectedTotalBytes}`);
    }

    const absoluteRoot = path.resolve(root);
    const rootMetadata = await lstat(absoluteRoot);
    if (!rootMetadata.isDirectory() || rootMetadata.isSymbolicLink()) {
      throw new Error("the exact SC-19057 q4 root must be a normal directory");
    }
    const canonicalRoot = await realpath(absoluteRoot);

    const revisionRoot = path.dirname(absoluteRoot);
    const snapshotsRoot = path.dirname(revisionRoot);
    const repositoryRoot = path.dirname(snapshotsRoot);
    const expectedRepositoryDirectory = `models--${SC19057_WAN_REPOSITORY.replace("/", "--")}`;
    if (
      path.basename(absoluteRoot).toLowerCase() !== SC19057_WAN_VARIANT ||
      path.basename(revisionRoot).toLowerCase() !== SC19057_WAN_REVISION ||
      path.basename(snapshotsRoot).toLowerCase() !== "snapshots" ||
      path.basename(repositoryRoot).toLowerCase() !== expectedRepositoryDirectory.toLowerCase()
    ) {
      throw new Error("the SC-19057 artifact root is not the exact repository, revision, and q4 variant");
    }

    const blobsRoot = path.join(repositoryRoot, "blobs");
    const blobsMetadata = await lstat(blobsRoot);
    if (!blobsMetadata.isDirectory() || blobsMetadata.isSymbolicLink()) {
      throw new Error("the exact repository blobs root must be a normal directory");
    }
    const canonicalRepositoryRoot = await realpath(repositoryRoot);
    const canonicalBlobsRoot = await realpath(blobsRoot);
    if (
      !samePath(canonicalRoot, path.join(canonicalRepositoryRoot, "snapshots", SC19057_WAN_REVISION, SC19057_WAN_VARIANT)) ||
      !samePath(canonicalBlobsRoot, path.join(canonicalRepositoryRoot, "blobs"))
    ) {
      throw new Error("the exact snapshot or blobs root escapes its resolved repository cache");
    }

    const logicalFiles = await listLogicalFiles(absoluteRoot);
    const seenLogicalPaths = new Set();
    const seenPhysicalPaths = new Set();
    const inventory = [];
    let totalBytes = 0;

    for (const { logicalPath, logicalMetadata } of logicalFiles) {
      if (!isInside(absoluteRoot, logicalPath)) {
        throw new Error(`logical artifact path escapes the exact q4 root: ${logicalPath}`);
      }
      const relativePath = path.relative(absoluteRoot, logicalPath).replaceAll("\\", "/");
      const logicalKey = relativePath.toLowerCase();
      if (seenLogicalPaths.has(logicalKey)) {
        throw new Error(`duplicate logical artifact path: ${relativePath}`);
      }
      seenLogicalPaths.add(logicalKey);
      const expectedFile = expected.get(logicalKey);
      if (!expectedFile) {
        throw new Error(`unexpected file in the exact q4 artifact: ${relativePath}`);
      }
      const [expectedBytes, expectedObject] = expectedFile;

      if (logicalMetadata.isSymbolicLink()) {
        const rawTarget = await readlink(logicalPath);
        try {
          const lexicalTarget = validateRawLinkTarget({ logicalPath, rawTarget, repositoryRoot, expectedObject });
          const targetMetadata = await lstat(lexicalTarget);
          if (!targetMetadata.isFile() || targetMetadata.isSymbolicLink()) {
            throw new Error("artifact link target must be a normal content-addressed file");
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(`${message}: ${relativePath}`);
        }
      }
      const physicalPath = await realpath(logicalPath);
      if (logicalMetadata.isSymbolicLink()) {
        if (!isInside(canonicalBlobsRoot, physicalPath) || !samePath(path.dirname(physicalPath), canonicalBlobsRoot)) {
          throw new Error(`artifact link escapes the exact repository blobs root: ${relativePath}`);
        }
        if (path.basename(physicalPath).toLowerCase() !== expectedObject) {
          throw new Error(`artifact link resolves to the wrong content object: ${relativePath}`);
        }
      } else if (!isInside(canonicalRoot, physicalPath)) {
        throw new Error(`artifact file escapes the exact immutable snapshot: ${relativePath}`);
      }
      if (seenPhysicalPaths.has(pathKey(physicalPath))) {
        throw new Error(`duplicate resolved artifact file: ${relativePath}`);
      }
      seenPhysicalPaths.add(pathKey(physicalPath));
      const inspected = await inspectOpenFile(physicalPath);
      if (inspected.streamedBytes !== inspected.metadata.size) {
        throw new Error(`artifact changed size while being read: ${relativePath}`);
      }
      if (inspected.streamedBytes !== expectedBytes) {
        throw new Error(`artifact byte count mismatch for ${relativePath}: expected ${expectedBytes}, found ${inspected.streamedBytes}`);
      }

      const authoritativeHash = expectedObject.length === 64 ? inspected.sha256 : inspected.gitBlob;
      if (authoritativeHash !== expectedObject) {
        throw new Error(`artifact content hash mismatch for ${relativePath}`);
      }
      totalBytes += inspected.streamedBytes;
      inventory.push({
        path: relativePath,
        bytes: inspected.streamedBytes,
        sha256: inspected.sha256,
        cacheObject: expectedObject,
        resolvedFromLink: logicalMetadata.isSymbolicLink(),
      });
    }

    if (inventory.length !== expected.size) {
      throw new Error(`SC-19057's immutable q4 artifact must contain exactly ${expected.size} files, found ${inventory.length}`);
    }
    if (totalBytes !== expectedTotalBytes) {
      throw new Error(`SC-19057's immutable q4 artifact must contain exactly ${expectedTotalBytes.toLocaleString("en-US")} bytes, found ${totalBytes}`);
    }
    inventory.sort((left, right) => left.path.localeCompare(right.path));
    const result = {
      repository: SC19057_WAN_REPOSITORY,
      revision: SC19057_WAN_REVISION,
      variant: SC19057_WAN_VARIANT,
      fileCount: inventory.length,
      totalBytes,
      files: inventory,
    };
    await writeJson(path.join(evidence, "wan-q4-artifact-inventory.json"), result);
    await writeJson(receiptPath, { ...receipt, status: "PASS", fileCount: inventory.length, totalBytes });
    return result;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    await writeJson(receiptPath, { ...receipt, status: "FAIL", error: message });
    throw error;
  }
}

function argument(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

async function main() {
  await inventorySc19057WanArtifact({
    root: argument("--root"),
    evidence: argument("--evidence"),
    identity: {
      runnerName: process.env.RUNNER_NAME ?? null,
      runId: process.env.GITHUB_RUN_ID ?? null,
      runAttempt: process.env.GITHUB_RUN_ATTEMPT ?? null,
      head: process.env.GITHUB_SHA ?? null,
    },
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
