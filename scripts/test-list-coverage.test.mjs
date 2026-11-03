import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { discoverRootTests } from "./run-root-tests.mjs";

test("root discovery includes nested tests and leaves desktop to its suite", () => {
  const files = discoverRootTests();
  assert.ok(files.includes("scripts/test-list-coverage.test.mjs"));
  assert.ok(files.includes("scripts/lib/jsonc.test.mjs"));
  assert.ok(!files.some((file) => file.startsWith("apps/desktop/")));
  const pkg = JSON.parse(readFileSync("package.json", "utf8"));
  assert.equal(pkg.scripts.check, "node scripts/run-root-tests.mjs");
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
  assert.deepEqual(listed, walk("").sort());
});
