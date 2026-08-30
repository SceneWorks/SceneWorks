import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  buildProductionClosure,
  checkManifestProductionClosure,
  checkProductionClosure,
  closureSha256,
  stableJson,
  validateProductionClosureShape,
} from "./starvector-production-closure.mjs";

async function fixture() {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-closure-"));
  await mkdir(path.join(root, "nested"));
  await writeFile(path.join(root, "z.txt"), "z\n");
  await writeFile(path.join(root, "nested", "a.txt"), "alpha\n");
  return root;
}

test("production closure is deterministic, normalized, and hashes recursively sorted compact JSON", async () => {
  const root = await fixture();
  const first = await buildProductionClosure({ root, paths: ["z.txt", "nested/a.txt"] });
  const second = await buildProductionClosure({ root, paths: ["nested/a.txt", "z.txt"] });
  assert.deepEqual(first, second);
  assert.deepEqual(first.entries.map(({ path: entryPath }) => entryPath), ["nested/a.txt", "z.txt"]);
  assert.equal(first.sha256, closureSha256(first.entries));
  assert.equal(stableJson({ z: 1, a: { y: 2, x: 3 } }), '{"a":{"x":3,"y":2},"z":1}');
  await checkProductionClosure(first, { root, paths: ["z.txt", "nested/a.txt"] });
});

test("production closure rejects symlinks, duplicates, traversal, backslashes, and catalog self-reference", async () => {
  const root = await fixture();
  await symlink(path.join(root, "z.txt"), path.join(root, "linked.txt"));
  await assert.rejects(() => buildProductionClosure({ root, paths: ["linked.txt"] }), /symlinks are forbidden/);
  await assert.rejects(() => buildProductionClosure({ root, paths: ["z.txt", "z.txt"] }), /duplicate entry path/);
  for (const bad of ["../z.txt", "nested//a.txt", "nested\\a.txt", "/z.txt", "config/manifests/builtin.models.jsonc"]) {
    await assert.rejects(() => buildProductionClosure({ root, paths: [bad] }), /not normalized|POSIX|relative|not an allowed/);
  }
});

test("shape and tree checks detect every closure mutation", async () => {
  const root = await fixture();
  const paths = ["nested/a.txt", "z.txt"];
  const closure = await buildProductionClosure({ root, paths });
  for (const mutate of [
    (value) => { value.entries[0].byteSize += 1; },
    (value) => { value.entries[0].sha256 = "0".repeat(64); },
    (value) => { value.entries.reverse(); value.sha256 = closureSha256(value.entries); },
    (value) => { value.entries.push({ ...value.entries[0] }); value.sha256 = closureSha256(value.entries); },
    (value) => { value.extra = true; },
  ]) {
    const changed = structuredClone(closure);
    mutate(changed);
    assert.throws(() => validateProductionClosureShape(changed));
  }
  await writeFile(path.join(root, "z.txt"), "changed\n");
  await assert.rejects(() => checkProductionClosure(closure, { root, paths }), /differs from the current source tree/);
  assert.equal(await readFile(path.join(root, "z.txt"), "utf8"), "changed\n");
});

test("live manifest seals the exact frozen-tree production closure", async () => {
  const closure = await checkManifestProductionClosure();
  assert.equal(closure.sha256, "6a8197ed2285342175833e9c20c4296ce47a3324ada6db5702b35bad6cdcd08a");
  assert.equal(closure.entries.length, 27);
});
