import assert from "node:assert/strict";
import { readFileSync, readdirSync, mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { discoverRootTests, rootTestConcurrency } from "./run-root-tests.mjs";

const portableInventory = (files) => files.map((file) => file.replaceAll("\\", "/")).sort();

test("root discovery includes nested tests and leaves desktop to its suite", () => {
  const files = discoverRootTests();
  assert.ok(files.includes("scripts/test-list-coverage.test.mjs"));
  assert.ok(files.includes("scripts/lib/jsonc.test.mjs"));
  assert.ok(!files.some((file) => file.startsWith("apps/desktop/")));
  const pkg = JSON.parse(readFileSync("package.json", "utf8"));
  assert.equal(pkg.scripts.check, "node scripts/run-root-tests.mjs");
});

test("root test concurrency respects available CPUs and caps at four", () => {
  assert.equal(rootTestConcurrency(1), 1);
  assert.equal(rootTestConcurrency(2), 2);
  assert.equal(rootTestConcurrency(18), 4);
});

test("desktop's separately owned suite lists every desktop Node test", () => {
  const desktop = path.join("apps", "desktop");
  const script = JSON.parse(readFileSync(path.join(desktop, "package.json"), "utf8")).scripts.test;
  const listed = script.split(/\s+/).filter((value) => value.endsWith(".test.mjs")).sort();
  const walk = (dir) => readdirSync(path.join(desktop, dir), { withFileTypes: true }).flatMap((entry) => {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory() && !["node_modules", "target", "dist"].includes(entry.name)) return walk(file);
    return entry.isFile() && entry.name.endsWith(".test.mjs") ? [file] : [];
  });
  assert.deepEqual(portableInventory(listed), portableInventory(walk("")));
});


test("root discovery excludes named Cargo build caches while retaining nested source tests", () => {
  const root = mkdtempSync(path.join(tmpdir(), "root-test-discovery-"));
  try {
    for (const directory of ["target", "target-auth-security", "nested/source"]) {
      mkdirSync(path.join(root, directory), { recursive: true });
      writeFileSync(path.join(root, directory, "fixture.test.mjs"), "");
    }
    const files = discoverRootTests(root);
    assert.equal(files.length, 1);
    assert.ok(files[0].endsWith(path.join("nested", "source", "fixture.test.mjs").split(path.sep).join("/")));
  } finally { rmSync(root, { recursive: true, force: true }); }
});


test("desktop inventories compare portable paths without hiding missing or extra tests", () => {
  const listed = ["scripts/build-sidecar-platform.test.mjs", "scripts/nested/runtime.test.mjs"];
  const discovered = listed.map((file) => path.win32.join(...file.split("/")));
  assert.notDeepEqual(discovered, listed, "fixture must exercise Windows-shaped discovery");
  assert.deepEqual(portableInventory(listed), portableInventory(discovered));
  assert.deepEqual(portableInventory(discovered), portableInventory(listed));
  assert.notDeepEqual(portableInventory(listed), portableInventory(discovered.slice(1)));
  assert.notDeepEqual(portableInventory([...listed, "scripts/extra.test.mjs"]), portableInventory(discovered));
  assert.notDeepEqual(portableInventory([...listed, listed[0]]), portableInventory(discovered));
});
