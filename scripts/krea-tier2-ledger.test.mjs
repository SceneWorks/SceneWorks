import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { captureKreaCompanionLedger, replayKreaCompanionLedger } from "./krea-tier2-ledger.mjs";
import { COMPANION, kreaSnapshotFile } from "./krea-tier2-proof.mjs";
import { sha256 } from "./lib/file-sha256.mjs";

const exactLedgerLeaf = "6a7de3e1ed5b03ddc81cb03d6f0be548657a5f7bc4f662fefd7a71947ee94b1c.json";
const ledgerDir = (state) => path.join(state, "data", "cache", "trusted-pinned-files");

async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), "krea-companion-ledger-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const state = path.join(root, "baseline"), hub = path.join(root, "hub");
  const selectors = [
    "bf16/transformer/config.json", "bf16/text_encoder/*", "bf16/vae/*",
    "bf16/tokenizer/*", "bf16/scheduler/*", "bf16/model_index.json",
  ];
  const paths = [
    "bf16/model_index.json", "bf16/scheduler/scheduler_config.json", "bf16/text_encoder/config.json",
    "bf16/text_encoder/model.safetensors", "bf16/tokenizer/chat_template.jinja", "bf16/tokenizer/tokenizer.json",
    "bf16/tokenizer/tokenizer_config.json", "bf16/transformer/config.json", "bf16/vae/config.json",
    "bf16/vae/diffusion_pytorch_model.safetensors",
  ];
  const lfsPaths = new Set([paths[3], paths[5], paths[9]]);
  const files = [], members = [];
  for (const file of paths) {
    const bytes = Buffer.from(`fixture ${file}`);
    const digest = sha256(bytes);
    const stageDigest = lfsPaths.has(file) ? { sha256: digest } : {
      gitBlobSha1: createHash("sha1").update(`blob ${bytes.length}\0`).update(bytes).digest("hex"),
    };
    const staged = { ...COMPANION, file, bytes: bytes.length, ...stageDigest };
    files.push(staged);
    members.push({ relativePath: file, sizeBytes: bytes.length, sha256: `sha256:${digest}` });
    const location = kreaSnapshotFile(hub, staged);
    await mkdir(path.dirname(location), { recursive: true });
    await writeFile(location, bytes);
  }
  const ledger = { schemaVersion: 1, baseUrl: "https://huggingface.co", ...COMPANION, selectors, files: members };
  const location = path.join(ledgerDir(state), exactLedgerLeaf);
  const body = `${JSON.stringify(ledger, null, 2)}\n`;
  await mkdir(path.dirname(location), { recursive: true });
  await writeFile(location, body);
  return { root, state, hub, ledger, body, location, staging: { files, stagedFileCount: files.length }, sourceJobId: "base-job" };
}

test("replays only the original API ledger bytes into a fresh Hires state", async (t) => {
  const input = await fixture(t);
  const seed = await captureKreaCompanionLedger(input);
  const state = path.join(input.root, "next");
  await mkdir(state);
  assert.deepEqual(await replayKreaCompanionLedger(seed, state), { ...seed.receipt, mode: "replayed" });
  assert.equal(seed.receipt.sourceJobId, input.sourceJobId);
  assert.equal(seed.receipt.fileCount, 10);
  assert.equal(seed.receipt.sha256, sha256(input.body));
  assert.equal(await readFile(path.join(ledgerDir(state), exactLedgerLeaf), "utf8"), input.body);
  assert.deepEqual(await readdir(path.join(state, "data", "cache")), ["trusted-pinned-files"]);
  assert.deepEqual(await readdir(ledgerDir(state)), [exactLedgerLeaf]);
  await assert.rejects(replayKreaCompanionLedger(seed, state), { code: "EEXIST" });
  assert.deepEqual(await readdir(ledgerDir(state)), [exactLedgerLeaf]);
  assert.equal(await readFile(path.join(ledgerDir(state), exactLedgerLeaf), "utf8"), input.body);
});

test("refuses absent or widened ledger metadata, substituted bytes and incorrect Git framing", async (t) => {
  const input = await fixture(t);
  const invalidateLedger = async (change, expected) => {
    const bad = structuredClone(input.ledger);
    change(bad);
    await writeFile(input.location, JSON.stringify(bad));
    await assert.rejects(captureKreaCompanionLedger(input), expected);
  };
  await invalidateLedger((value) => { value.baseUrl = "https://wrong.example"; }, /different origin/);
  await invalidateLedger((value) => { value.revision = "f".repeat(40); }, /different origin/);
  await invalidateLedger((value) => { value.selectors.reverse(); }, /different origin/);
  await invalidateLedger((value) => { value.unknown = true; }, /different origin/);
  await invalidateLedger((value) => { value.files.pop(); }, /full companion closure/);
  await invalidateLedger((value) => { value.files[9] = value.files[0]; }, /exact, unique, and sorted/);
  await invalidateLedger((value) => { value.files.reverse(); }, /exact, unique, and sorted/);
  await invalidateLedger((value) => { value.files[0].sizeBytes++; }, /differs from staging/);
  await invalidateLedger((value) => { value.files[0].sha256 = `sha256:${"f".repeat(64)}`; }, /differs from staging/);
  await writeFile(input.location, input.body);
  const lfs = input.staging.files.find(({ sha256 }) => sha256);
  const blob = input.staging.files.find(({ gitBlobSha1 }) => gitBlobSha1);
  for (const [item, reason] of [[lfs, /staged Krea LFS file differs/], [blob, /staged Krea Git blob differs/]]) {
    const file = kreaSnapshotFile(input.hub, item);
    const original = await readFile(file);
    await writeFile(file, Buffer.alloc(item.bytes, 1));
    await assert.rejects(captureKreaCompanionLedger(input), reason);
    await writeFile(file, original);
  }
  await rm(input.location);
  await assert.rejects(captureKreaCompanionLedger(input), { code: "ENOENT" });
  await assert.rejects(replayKreaCompanionLedger(null, input.state), /verified baseline companion ledger/);
});

test("replay refuses redirected cache parents and a corrupted in-memory seed", async (t) => {
  const input = await fixture(t);
  const seed = await captureKreaCompanionLedger(input);
  const state = path.join(input.root, "next"), outside = path.join(input.root, "outside");
  await mkdir(path.join(state, "data"), { recursive: true });
  await mkdir(outside);
  await symlink(outside, path.join(state, "data", "cache"), process.platform === "win32" ? "junction" : "dir");
  await assert.rejects(replayKreaCompanionLedger(seed, state), /rejects symlink, junction/);
  assert.deepEqual(await readdir(outside), []);
  seed.body[0] = 0;
  await assert.rejects(replayKreaCompanionLedger(seed, state), /verified baseline companion ledger/);
});
