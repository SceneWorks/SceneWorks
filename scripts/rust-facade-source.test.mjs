import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { rustFacadeSources, rustFacadeSource, rustFacadeFunction, rustProductionSources, rustProductionSource, rustSourceWithTests, rustFunctionSlices, rustFunctionSource } from "./lib/rust-facade-source.mjs";

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


test("facade readers retain visibility on authored owner imports", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "rust-facade-visibility-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const visibility of ["pub", "pub(crate)", "pub(super)"]) {
    const entry = path.join(root, "main.rs");
    await writeFile(entry, `#[path = "owner.rs"]\nmod owner;\n${visibility} use owner::*;\n`);
    await writeFile(path.join(root, "owner.rs"), "fn actual_owner() {}\n");
    assert.deepEqual((await rustFacadeSources(entry)).map((x) => path.basename(x.path)), ["main.rs", "owner.rs"]);
  }
});


test("production source views include scoped concerns and exclude test or quoted declarations", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "rust-production-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const entry = path.join(root, "main.rs");
  await writeFile(entry, `#[path = "scoped.rs"] mod scoped;
#[cfg(test)] #[path = "absent-test.rs"] mod tests;
const QUOTED: &str = r#"#[path = "absent.rs"] mod phantom;"#;
// #[path = "absent-comment.rs"] mod comment;
`);
  await writeFile(path.join(root, "scoped.rs"), "fn admitted() { anchor.is_current(); }\n");
  assert.deepEqual((await rustProductionSources(entry)).map((x) => path.basename(x.path)), ["main.rs", "scoped.rs"]);
  assert.match(await rustProductionSource(entry), /anchor\.is_current/);
  assert.doesNotMatch(await rustFacadeSource(entry), /anchor\.is_current/);
});


test("test-authored source projections follow sibling declarations only when explicitly requested", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "rust-test-source-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const entry = path.join(root, "main.rs");
  await writeFile(entry, '#[cfg(test)] #[path = "fixture.rs"] mod tests;\n');
  await writeFile(path.join(root, "fixture.rs"), 'fn source_contract() {}\n');
  assert.doesNotMatch(await rustProductionSource(entry), /fn source_contract/);
  assert.match(await rustSourceWithTests(entry), /fn source_contract/);
});


test("function source boundaries retain visible owners and ignore quoted syntax", () => {
  const source = `const QUOTED: &str = r#"fn phantom() {}"#;
pub(super) fn first() { let _ = r#" } "#; if true { nested(); } }
pub(crate) fn second<T>() { /* } */ next(); }
`;
  assert.deepEqual(rustFunctionSlices(source).map((item) => item.name), ["first", "second"]);
  assert.match(rustFunctionSource(source, "first"), /nested\(\)/);
  assert.doesNotMatch(rustFunctionSource(source, "first"), /next\(\)/);
  assert.throws(() => rustFunctionSource(source, "phantom"), /expected one owner/);
});


test("a Cargo bin entry resolves ordinary sibling modules from its crate root", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "rust-bin-source-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const bin = path.join(root, "bin");
  await mkdir(bin);
  const entry = path.join(bin, "native.rs");
  await writeFile(entry, "#[cfg(test)] mod fixtures;\n");
  await writeFile(path.join(bin, "fixtures.rs"), "fn actual_fixture() {}\n");
  assert.match(await rustSourceWithTests(entry), /fn actual_fixture/);
});


test("file URL entries preserve escaped paths through facade and production traversal", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "rust facade #% snow☃-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const entry = path.join(root, "main.rs");
  const owner = path.join(root, "owner.rs");
  await writeFile(entry, '#[path = "owner.rs"] mod owner; pub use owner::*;\n');
  await writeFile(owner, "pub(crate) fn actual_owner() {}\n");
  const url = pathToFileURL(entry);
  assert.match(url.href, /%20/);
  for (const read of [rustFacadeSources, rustProductionSources]) {
    assert.deepEqual((await read(url)).map(({ path }) => path), [entry, owner]);
  }
  for (const read of [rustFacadeSource, rustProductionSource, rustSourceWithTests]) {
    assert.match(await read(url), /fn actual_owner/);
  }
  assert.equal((await rustFacadeFunction(url, "actual_owner")).path, owner);
});
