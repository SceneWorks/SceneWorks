import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
const script = new URL("./capture-catalog-baseline.mjs", import.meta.url);
test("a candidate checkout cannot replace a frozen old-source fixture", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "catalog-baseline-"));
  const output = path.join(dir, "old.json");
  writeFileSync(output, "preserve the before side\n");
  try {
    assert.throws(() => execFileSync(process.execPath, [script.pathname, "--source", "0".repeat(40), "--output", output], { stdio: "pipe" }), /Command failed/);
    assert.equal(readFileSync(output, "utf8"), "preserve the before side\n");
    assert.equal(existsSync(path.join(dir, "candidate.json")), false);
  } finally { rmSync(dir, { recursive: true }); }
});
