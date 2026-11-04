import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { MANIFEST, restore, validateManifest, verify, verifyBytes, verifyTree } from "./historical-artifacts.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
function fixture() {
  const bytes = Buffer.from([0, 255, 128, 10, 13, 0]);
  const entry = { path: "docs/calibration/example/render.rgb", group: "calibration", bytes: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    gitBlob: createHash("sha1").update(`blob ${bytes.length}\0`).update(bytes).digest("hex"), mode: "100644" };
  const source = { repository: "SceneWorks/SceneWorks", commit: "1".repeat(40), tree: "2".repeat(40) };
  const manifest = { version: 1, source, entries: [entry] };
  const tree = { sha: source.tree, truncated: false, tree: [{ path: entry.path, type: "blob",
    mode: entry.mode, sha: entry.gitBlob, size: bytes.length }] };
  const api = async (endpoint) => endpoint.includes("/commits/")
    ? { sha: source.commit, tree: { sha: source.tree } } : endpoint.includes("/trees/") ? tree : bytes;
  return { manifest, entry, bytes, tree, api };
}

test("cold retrieval restores exact binary bytes and independently verifies them", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "historical-artifacts-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const f = fixture();
  assert.equal((await restore(f.manifest, dir, { api: f.api })).files, 1);
  assert.deepEqual(await readFile(path.join(dir, f.entry.path)), f.bytes);
  assert.equal((await verify(f.manifest, dir)).bytes, f.bytes.length);
  // Existing correct evidence is idempotent; never overwrite differing evidence.
  await restore(f.manifest, dir, { api: f.api });
  await writeFile(path.join(dir, f.entry.path), "corrupt");
  await assert.rejects(restore(f.manifest, dir, { api: f.api }), /digest\/size mismatch/);
  assert.equal(await readFile(path.join(dir, f.entry.path), "utf8"), "corrupt");
});

test("retrieval refuses an unbound source, wrong path/blob/mode/size, truncated tree and corrupt bytes", async (t) => {
  const f = fixture();
  const dir = await mkdtemp(path.join(os.tmpdir(), "historical-artifacts-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  await assert.rejects(restore(f.manifest, dir, { api: async () => ({ sha: "3".repeat(40) }) }), /commit\/tree mismatch/);
  for (const change of [{ path: "wrong.rgb" }, { sha: "0".repeat(40) }, { mode: "100755" }, { size: 1 }]) {
    assert.throws(() => verifyTree(f.manifest, { ...f.tree, tree: [{ ...f.tree.tree[0], ...change }] }), /path\/blob\/size\/mode mismatch/);
  }
  assert.throws(() => verifyTree(f.manifest, { ...f.tree, truncated: true }), /truncated/);
  const corrupt = Buffer.from(f.bytes); corrupt[0] = 1;
  await assert.rejects(restore(f.manifest, dir, { api: (endpoint) => endpoint.includes("/blobs/") ? corrupt : f.api(endpoint) }), /digest\/size mismatch/);
  assert.throws(() => verifyBytes({ ...f.entry, sha256: "0".repeat(64) }, f.bytes), /digest\/size mismatch/);
});

test("manifest and destination reject traversal, duplicate paths, symlink escapes and unknown groups", async (t) => {
  const f = fixture();
  for (const bad of ["../outside.rgb", "/absolute.rgb", "a/../b.rgb", "a\\b.rgb", "a:b.rgb", "a//b.rgb"]) {
    assert.throws(() => validateManifest({ ...f.manifest, entries: [{ ...f.entry, path: bad }] }), /invalid/);
  }
  assert.throws(() => validateManifest({ ...f.manifest, entries: [f.entry, f.entry] }), /duplicate/);
  const dir = await mkdtemp(path.join(os.tmpdir(), "historical-artifacts-"));
  const outside = await mkdtemp(path.join(os.tmpdir(), "historical-artifacts-outside-"));
  t.after(() => Promise.all([rm(dir, { recursive: true, force: true }), rm(outside, { recursive: true, force: true })]));
  await symlink(outside, path.join(dir, "docs"), process.platform === "win32" ? "junction" : "dir");
  await assert.rejects(restore(f.manifest, dir, { api: f.api }), /symlink/);
  await assert.rejects(restore(f.manifest, dir, { api: f.api, group: "typo" }), /no historical artifacts/);
});

test("canonical pose previews retain the exact archived bytes for every removed duplicate", async () => {
  const manifest = validateManifest(JSON.parse(await readFile(path.join(ROOT, MANIFEST), "utf8")));
  const poses = manifest.entries.filter((entry) => entry.group === "poses");
  assert.ok(poses.length > 0);
  for (const entry of poses) {
    assert.equal(entry.replacement, `apps/web/public/${entry.path}`);
    verifyBytes(entry, await readFile(path.join(ROOT, entry.replacement)));
  }
});

test("every archived calibration frame preserves its historical source-session digest receipt", async () => {
  const manifest = validateManifest(JSON.parse(await readFile(path.join(ROOT, MANIFEST), "utf8")));
  const evidence = JSON.parse(await readFile(path.join(ROOT, "docs/generated/memory-calibration-evidence.json"), "utf8"));
  const outputs = new Map(evidence.sourceSessions.flatMap((session) => session.outputs ?? [])
    .map((output) => [output.path, output]));
  for (const entry of manifest.entries.filter((entry) => entry.group === "calibration")) {
    const receipt = outputs.get(entry.path);
    assert.ok(receipt, `archived frame lost its source-session receipt: ${entry.path}`);
    assert.equal(entry.sha256, receipt.sha256);
    // Early CUDA sessions retained a digest without a size; the archive additionally pins size.
    if (receipt.bytes !== undefined) assert.equal(entry.bytes, receipt.bytes);
  }
});


test("Krea smoke defaults use canonical repository poses and retired MOSS scripts stay absent", async () => {
  for (const file of ["crates/sceneworks-worker/src/krea_control_mlx_smoke.rs", "crates/sceneworks-worker/src/image_jobs/tests.rs"]) {
    const source = await readFile(path.join(ROOT, file), "utf8");
    assert.match(source, /"KREA_CTRL_POSE"/);
    assert.match(source, /"\/\.\.\/\.\.\/apps\/web\/public\/poses\/dance_01\.png"/);
    assert.doesNotMatch(source, /"\/Users\/[^"\n]*poses\/dance_01\.png"/);
  }
  for (const file of ["scripts/lib/moss-edit-terminal-actions.mjs", "scripts/ci/moss-upstream-reference-common.mjs", "scripts/ci/moss-upstream-quality.mjs", "scripts/lib/moss-edit-quality-amendment.test.mjs", "scripts/moss-edit-terminal-actions.test.mjs"]) {
    await assert.rejects(readFile(path.join(ROOT, file)), { code: "ENOENT" });
  }
});
