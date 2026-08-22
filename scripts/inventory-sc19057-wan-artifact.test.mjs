import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  SC19057_WAN_FILES,
  SC19057_WAN_REPOSITORY,
  SC19057_WAN_REVISION,
  SC19057_WAN_TOTAL_BYTES,
  inventorySc19057WanArtifact,
} from "./inventory-sc19057-wan-artifact.mjs";

function gitBlobId(content) {
  return createHash("sha1").update(`blob ${content.length}\0`).update(content).digest("hex");
}

async function fixture(t, content = Buffer.from("resolved immutable artifact bytes")) {
  const scratch = await mkdtemp(path.join(os.tmpdir(), "sc19057-inventory-"));
  t.after(() => rm(scratch, { recursive: true, force: true }));
  const repository = path.join(scratch, `models--${SC19057_WAN_REPOSITORY.replace("/", "--")}`);
  const root = path.join(repository, "snapshots", SC19057_WAN_REVISION, "q4");
  const blobs = path.join(repository, "blobs");
  const evidence = path.join(scratch, "evidence");
  await mkdir(root, { recursive: true });
  await mkdir(blobs, { recursive: true });
  return { scratch, repository, root, blobs, evidence, content };
}

async function linkArtifact({ root, blobs, logical = "model.bin", content, object = gitBlobId(content) }) {
  const blob = path.join(blobs, object);
  await writeFile(blob, content);
  const logicalPath = path.join(root, ...logical.split("/"));
  await mkdir(path.dirname(logicalPath), { recursive: true });
  await symlink(path.relative(path.dirname(logicalPath), blob), logicalPath, "file");
  return { blob, logicalPath, object };
}

test("SC-19057 pins one closed 25-file immutable artifact manifest", () => {
  const files = Object.values(SC19057_WAN_FILES);
  assert.equal(files.length, 25);
  assert.equal(files.reduce((sum, [bytes]) => sum + bytes, 0), SC19057_WAN_TOTAL_BYTES);
  assert.equal(new Set(files.map(([, object]) => object)).size, 25);
  assert.equal(SC19057_WAN_TOTAL_BYTES, 17_338_835_457);
});

test("SC-19057 inventories resolved link bytes and hashes rather than zero-sized Windows link metadata", async (t) => {
  const setup = await fixture(t);
  const { logicalPath, object } = await linkArtifact(setup);
  const linkMetadataBytes = (await lstat(logicalPath)).size;
  assert.notEqual(linkMetadataBytes, setup.content.length, "fixture must distinguish link metadata from target bytes");

  const result = await inventorySc19057WanArtifact({
    root: setup.root,
    evidence: setup.evidence,
    expectedFiles: { "model.bin": [setup.content.length, object] },
    expectedTotalBytes: setup.content.length,
  });

  assert.equal(result.fileCount, 1);
  assert.equal(result.totalBytes, setup.content.length);
  assert.equal(result.files[0].bytes, setup.content.length);
  assert.equal(result.files[0].resolvedFromLink, true);
  assert.equal(result.files[0].cacheObject, object);
  assert.equal(result.files[0].sha256, createHash("sha256").update(setup.content).digest("hex"));
  assert.equal(JSON.parse(await readFile(path.join(setup.evidence, "wan-q4-inventory-preflight.json"), "utf8")).status, "PASS");
});

test("SC-19057 records an auditable failure before rejecting a broken cache link", async (t) => {
  const setup = await fixture(t);
  const object = "a".repeat(64);
  const logicalPath = path.join(setup.root, "broken.bin");
  await symlink(path.relative(setup.root, path.join(setup.blobs, object)), logicalPath, "file");

  await assert.rejects(
    inventorySc19057WanArtifact({
      root: setup.root,
      evidence: setup.evidence,
      expectedFiles: { "broken.bin": [1, object] },
      expectedTotalBytes: 1,
      identity: { runId: "broken-link-test" },
    }),
    /ENOENT|no such file|cannot find/i,
  );
  const receipt = JSON.parse(await readFile(path.join(setup.evidence, "wan-q4-inventory-preflight.json"), "utf8"));
  assert.equal(receipt.status, "FAIL");
  assert.equal(receipt.runId, "broken-link-test");
  assert.match(receipt.error, /ENOENT|no such file|cannot find/i);
});

test("SC-19057 rejects a link that resolves outside the exact repository blobs root", async (t) => {
  const setup = await fixture(t);
  const object = gitBlobId(setup.content);
  const outside = path.join(setup.scratch, "outside", object);
  await mkdir(path.dirname(outside), { recursive: true });
  await writeFile(outside, setup.content);
  await symlink(path.relative(setup.root, outside), path.join(setup.root, "escape.bin"), "file");

  await assert.rejects(
    inventorySc19057WanArtifact({
      root: setup.root,
      evidence: setup.evidence,
      expectedFiles: { "escape.bin": [setup.content.length, object] },
      expectedTotalBytes: setup.content.length,
    }),
    /escapes the exact repository blobs root/,
  );
});

test("SC-19057 rejects wrong size, content hash, count, and duplicate content-addresses", async (t) => {
  await t.test("wrong size", async (t) => {
    const setup = await fixture(t);
    const { object } = await linkArtifact(setup);
    await assert.rejects(
      inventorySc19057WanArtifact({
        root: setup.root,
        evidence: setup.evidence,
        expectedFiles: { "model.bin": [setup.content.length + 1, object] },
        expectedTotalBytes: setup.content.length + 1,
      }),
      /byte count mismatch/,
    );
  });

  await t.test("wrong hash with unchanged size", async (t) => {
    const setup = await fixture(t);
    const original = Buffer.from(setup.content);
    const object = gitBlobId(original);
    const { blob } = await linkArtifact({ ...setup, content: original, object });
    await writeFile(blob, Buffer.alloc(original.length, 0x78));
    await assert.rejects(
      inventorySc19057WanArtifact({
        root: setup.root,
        evidence: setup.evidence,
        expectedFiles: { "model.bin": [original.length, object] },
        expectedTotalBytes: original.length,
      }),
      /content hash mismatch/,
    );
  });

  await t.test("wrong count", async (t) => {
    const setup = await fixture(t);
    const { object } = await linkArtifact(setup);
    await assert.rejects(
      inventorySc19057WanArtifact({
        root: setup.root,
        evidence: setup.evidence,
        expectedFiles: {
          "model.bin": [setup.content.length, object],
          "missing.bin": [1, "b".repeat(64)],
        },
        expectedTotalBytes: setup.content.length + 1,
      }),
      /must contain exactly 2 files, found 1/,
    );
  });

  await t.test("duplicate content object", async (t) => {
    const setup = await fixture(t);
    const { object } = await linkArtifact(setup);
    await assert.rejects(
      inventorySc19057WanArtifact({
        root: setup.root,
        evidence: setup.evidence,
        expectedFiles: {
          "model.bin": [setup.content.length, object],
          "duplicate.bin": [setup.content.length, object],
        },
        expectedTotalBytes: setup.content.length * 2,
      }),
      /duplicate expected content object/,
    );
  });
});
