import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { rustFacadeSources, rustFacadeSource, rustFacadeFunction } from "./lib/rust-facade-source.mjs";

test("facade readers follow authored glob owners while keeping scoped/test namespaces separate", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "rust-facade-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const entry = path.join(root, "main.rs");
  await writeFile(entry, '#[path = "family.rs"]\nmod family;\nuse family::*;\n#[path = "scoped.rs"]\nmod scoped;\n#[cfg(test)]\n#[path = "tests.rs"]\nmod tests;\nfn main() {}\n');
  await writeFile(path.join(root, "family.rs"), 'pub(super) fn run() {\n    let value = "scoped braces { }";\n}\n');
  assert.deepEqual((await rustFacadeSources(entry)).map((x) => path.basename(x.path)), ["main.rs", "family.rs"]);
  assert.match(await rustFacadeSource(entry), /pub\(super\) fn run/);
  assert.deepEqual(await rustFacadeFunction(entry, "run"), { path: path.join(root, "family.rs"), source: 'pub(super) fn run() {\n    let value = "scoped braces { }";\n}' });
  await writeFile(path.join(root, "family.rs"), "pub(super) fn open<'a>(value: &'a str) -> &'a str {\n    value\n}\n");
  assert.match((await rustFacadeFunction(entry, "open")).source, /fn open<'a>/);
  await assert.rejects(rustFacadeFunction(entry, "missing"), /expected one owner/);
  await assert.rejects(rustFacadeFunction(entry, "run.*"), /Invalid Rust function name/);
});
