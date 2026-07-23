import assert from "node:assert/strict";
import {
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const scriptsDir = dirname(fileURLToPath(import.meta.url));
const script = join(scriptsDir, "build-update-manifest.mjs");

function runManifest(args) {
  return spawnSync(process.execPath, [script, ...args], {
    encoding: "utf8",
  });
}

test("merges a signed linux-x86_64 updater without losing existing platforms", (t) => {
  const tempDir = mkdtempSync(join(os.tmpdir(), "sceneworks-updater-test-"));
  t.after(() => rmSync(tempDir, { recursive: true }));
  const input = join(tempDir, "latest.json");
  const output = join(tempDir, "merged.json");
  const signature = join(tempDir, "SceneWorks.AppImage.tar.gz.sig");
  writeFileSync(
    input,
    JSON.stringify({
      version: "0.8.1",
      notes: "SceneWorks v0.8.1",
      pub_date: "2026-07-23T00:00:00.000Z",
      platforms: {
        "darwin-aarch64": { signature: "mac-signature", url: "https://mac" },
        "windows-x86_64": {
          signature: "windows-signature",
          url: "https://windows",
        },
      },
    }),
  );
  writeFileSync(signature, "linux-signature\n");

  const result = runManifest([
    "--target",
    "linux-x86_64",
    "--version",
    "0.8.1",
    "--url",
    "https://github.com/SceneWorks/SceneWorks/releases/download/v0.8.1/SceneWorks.AppImage.tar.gz",
    "--sig",
    signature,
    "--in",
    input,
    "--out",
    output,
  ]);

  assert.equal(result.status, 0, result.stderr);
  const manifest = JSON.parse(readFileSync(output, "utf8"));
  assert.equal(manifest.version, "0.8.1");
  assert.equal(manifest.notes, "SceneWorks v0.8.1");
  assert.equal(Object.keys(manifest.platforms).length, 3);
  assert.deepEqual(manifest.platforms["linux-x86_64"], {
    signature: "linux-signature",
    url: "https://github.com/SceneWorks/SceneWorks/releases/download/v0.8.1/SceneWorks.AppImage.tar.gz",
  });
  assert.equal(
    manifest.platforms["darwin-aarch64"].signature,
    "mac-signature",
  );
  assert.equal(
    manifest.platforms["windows-x86_64"].signature,
    "windows-signature",
  );
});

test("refuses to merge a platform into a manifest from another version", (t) => {
  const tempDir = mkdtempSync(join(os.tmpdir(), "sceneworks-updater-test-"));
  t.after(() => rmSync(tempDir, { recursive: true }));
  const input = join(tempDir, "latest.json");
  const output = join(tempDir, "merged.json");
  const signature = join(tempDir, "payload.sig");
  writeFileSync(
    input,
    JSON.stringify({ version: "0.8.0", platforms: {} }),
  );
  writeFileSync(signature, "linux-signature\n");

  const result = runManifest([
    "--target",
    "linux-x86_64",
    "--version",
    "0.8.1",
    "--url",
    "https://example.invalid/update",
    "--sig",
    signature,
    "--in",
    input,
    "--out",
    output,
  ]);

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /input version 0\.8\.0.*requested version 0\.8\.1/);
});
