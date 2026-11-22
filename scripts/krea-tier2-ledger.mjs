import assert from "node:assert/strict";
import { createHash, randomUUID } from "node:crypto";
import { link, lstat, open, readFile, rm } from "node:fs/promises";
import path from "node:path";

import { COMPANION, kreaSnapshotFile } from "./krea-tier2-proof.mjs";
import { fileSha256, sha256 } from "./lib/file-sha256.mjs";
import { assertTerminalPhysicalContainment, ensureTerminalPhysicalDirectory } from "./lib/starvector-terminal-pin-paths.mjs";

// The API keys its host-local trusted ledger by the exact ordered resident-companion selectors.
// Same request as artifact_selection::selected_resident_base_closure_for_row and
// portable_admission::trusted_pinned_ledger_key. Only the API's original bytes are replayed.
const selectors = Object.freeze([
  "bf16/transformer/config.json",
  "bf16/text_encoder/*",
  "bf16/vae/*",
  "bf16/tokenizer/*",
  "bf16/scheduler/*",
  "bf16/model_index.json",
]);
const expectedLedger = Object.freeze({ schemaVersion: 1, baseUrl: "https://huggingface.co", ...COMPANION, selectors });
const { baseUrl, repo, revision } = expectedLedger;
const ledgerLeaf = `${sha256([baseUrl, repo, revision, ...selectors].join("\0"))}.json`;
const expectedCompanions = 10;
const maxMetadataBytes = 64 * 1024;

function ledgerFile(state) {
  return path.join(state, "data", "cache", "trusted-pinned-files", ledgerLeaf);
}

function isCompanion(file) {
  return /^bf16\/[A-Za-z0-9_./-]+$/.test(file) && file.split("/").every((part) => part && !part.startsWith("."))
    && selectors.some((selector) => selector.endsWith("/*") ? file.startsWith(selector.slice(0, -1)) : file === selector);
}

export async function captureKreaCompanionLedger({ state, hub, staging, sourceJobId, signal }) {
  assert.ok(sourceJobId, "Krea companion ledger requires a successful baseline job");
  assert.ok(Array.isArray(staging?.files) && staging.stagedFileCount === staging.files.length, "Krea companion ledger requires verified staging");
  const companions = staging.files.filter((item) => item.repo === repo && item.revision === revision);
  const staged = new Map(companions.map((item) => [item.file, item]));
  assert.ok(companions.length === expectedCompanions && staged.size === expectedCompanions && companions.every(({ file }) => isCompanion(file)),
    "Krea companion ledger requires exactly the verified BF16 companion files");

  const location = ledgerFile(state);
  await assertTerminalPhysicalContainment(state, path.dirname(location));
  const info = await lstat(location);
  assert.ok(info.isFile() && info.nlink === 1 && info.size > 0 && info.size <= maxMetadataBytes, "Krea baseline trusted ledger must be one ordinary bounded file");
  const body = await readFile(location, { signal });
  assert.equal(body.length, info.size, "Krea baseline trusted ledger changed while being read");
  const { files, ...definition } = JSON.parse(body.toString("utf8"));
  assert.deepEqual(definition, expectedLedger, "Krea baseline trusted ledger has a different origin, schema or selectors");
  assert.ok(Array.isArray(files) && files.length === expectedCompanions, "Krea baseline trusted ledger does not cover the full companion closure");
  let previous = "";
  for (const entry of files) {
    signal?.throwIfAborted();
    const item = staged.get(entry?.relativePath);
    assert.ok(item && previous < item.file, "Krea baseline ledger companion paths must be exact, unique, and sorted");
    previous = item.file;
    const source = kreaSnapshotFile(hub, item);
    await assertTerminalPhysicalContainment(hub, path.dirname(source));
    const stat = await lstat(source);
    assert.ok(stat.isFile() && stat.nlink === 1 && stat.size > 0 && stat.size === item.bytes, `staged Krea companion changed: ${item.file}`);

    let digest;
    if (item.sha256) {
      // Recheck staged LFS bytes once after priming and outside the measured interval.
      digest = await fileSha256(source, { signal });
      assert.equal(digest, item.sha256, `staged Krea LFS file differs from the verified source: ${item.file}`);
    } else {
      // The stage authenticates Git files by blob identity. The app ledger stores raw SHA-256.
      assert.ok(stat.size <= maxMetadataBytes, `staged Krea Git blob is too large: ${item.file}`);
      const bytes = await readFile(source, { signal });
      const gitOid = createHash("sha1").update(`blob ${bytes.length}\0`).update(bytes).digest("hex");
      assert.equal(gitOid, item.gitBlobSha1, `staged Krea Git blob differs from the verified source: ${item.file}`);
      digest = sha256(bytes);
    }
    assert.deepEqual(entry, { relativePath: item.file, sizeBytes: item.bytes, sha256: `sha256:${digest}` },
      `Krea baseline trusted ledger differs from staging: ${item.file}`);
  }
  return { body, receipt: Object.freeze({ sourceJobId, ...expectedLedger, fileCount: files.length, sha256: sha256(body) }) };
}

export async function replayKreaCompanionLedger(seed, state) {
  assert.ok(seed && sha256(seed.body) === seed.receipt.sha256, "Krea Hires.fix needs the verified baseline companion ledger");
  const location = ledgerFile(state);
  const parent = await ensureTerminalPhysicalDirectory(state, path.dirname(location));
  const partial = path.join(parent, `.krea-ledger-${randomUUID()}.part`);
  const file = await open(partial, "wx");
  try {
    try { await file.writeFile(seed.body); }
    finally { await file.close(); }
    await assertTerminalPhysicalContainment(state, parent);
    await link(partial, location);
  } finally {
    await rm(partial, { force: true });
  }
  return { ...seed.receipt, mode: "replayed" };
}
