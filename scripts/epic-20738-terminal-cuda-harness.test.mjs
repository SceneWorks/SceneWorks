import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  PROFILE_NAME,
  inferencePins,
  loadProfile,
  parseNvidiaSmi,
  receiptSkeleton,
  validateManifestAuthorities,
  validateProfile,
  validateReceipt,
} from "./epic-20738-terminal-cuda-harness.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

const profile = () => structuredClone(loadProfile());

test("checked-in terminal profile is the exact serialized 19-cell campaign", async () => {
  const checked = validateProfile(profile());
  assert.equal(checked.cells.length, 19);
  assert.deepEqual(
    checked.cells.map((cell) => cell.id),
    [
      "chroma1-base-q4", "chroma1-base-q8", "chroma1-flash-q4", "chroma1-flash-q8",
      "chroma1-hd-q4", "chroma1-hd-q8", "flux1-dev-q4", "flux1-dev-q8",
      "flux1-schnell-q4", "flux1-schnell-q8", "scail2-q4",
      "scail2-multi-reference-q4", "scail2-q8", "ltx-2-3-q8", "sdxl-openpose",
      "realvisxl-openpose", "realvisxl-lightning-openpose", "illustrious-v1-openpose",
      "illustrious-v2-openpose",
    ],
  );
  assert.equal(checked.cells.filter((cell) => cell.capability === "multiReference").length, 1);
  assert.equal(checked.cells.find((cell) => cell.capability === "multiReference").request.referencePairs, 6);
  assert.doesNotMatch(JSON.stringify(checked).toLowerCase(), /anima|sana|vace|flux2|true[_-]?v2|eros/);

  const manifest = JSON.parse(stripJsoncComments(await readFile(
    "config/manifests/builtin.models.jsonc", "utf8",
  )));
  assert.doesNotThrow(() => validateManifestAuthorities(checked, manifest));
});

test("profile validator rejects count, order, tier, boundary, blocked, and authority drift", async () => {
  const manifest = JSON.parse(stripJsoncComments(await readFile(
    "config/manifests/builtin.models.jsonc", "utf8",
  )));
  const missing = profile();
  missing.cells.pop();
  assert.throws(() => validateProfile(missing), /exactly 19/);

  const reordered = profile();
  [reordered.cells[0], reordered.cells[1]] = [reordered.cells[1], reordered.cells[0]];
  assert.throws(() => validateProfile(reordered), /serial order drifted/);

  const crossTier = profile();
  crossTier.cells[0].requestedTier = "q8";
  assert.throws(() => validateProfile(crossTier), /does not match/);

  const boundary = profile();
  boundary.cells.find((cell) => cell.capability === "multiReference").request.referencePairs = 5;
  assert.throws(() => validateProfile(boundary), /six-pair/);

  const blocked = profile();
  blocked.artifacts["scail2-q4"].repository = "SceneWorks/eros-model";
  assert.throws(() => validateProfile(blocked), /blocked surface eros/);

  const manifestDrift = profile();
  manifestDrift.artifacts["chroma1-base-q4"].revision = "a".repeat(40);
  assert.throws(() => validateManifestAuthorities(manifestDrift, manifest), /disagrees with manifest/);

  const controlDrift = profile();
  controlDrift.artifacts["sdxl-openpose"].revision = "b".repeat(40);
  assert.throws(() => validateManifestAuthorities(controlDrift, manifest), /frozen public OpenPose/);
});

function fixtureReceipt(status = "passed") {
  const cell = profile().cells[0];
  const receipt = receiptSkeleton({
    cell,
    ordinal: 1,
    repositories: {
      sceneworks: { sha: "1".repeat(40), clean: true },
      inference: { sha: "2".repeat(40), clean: true },
    },
    artifacts: [{
      id: "chroma1-base-q4",
      repository: "SceneWorks/chroma1-base-mlx",
      revision: "3".repeat(40),
      subdirectory: "q4",
      inventory: { complete: true, sha256: "4".repeat(64), files: 3, bytes: 42, error: null },
    }],
    execution: {
      runId: "123", runAttempt: "2", headSha: "1".repeat(40), workflow: "Windows Candle worker",
      runnerName: "cuda-windows", runnerOs: "Windows", runnerArch: "X64",
    },
    gpuIdentity: [{
      index: 0, name: "fixture GPU", uuid: "GPU-fixture", pciBusId: "00000000:01:00.0",
      computeCapability: "12.0", driverVersion: "999.1", memoryTotalMiB: 49140,
      memoryUsedMiB: 10, memoryFreeMiB: 49130, raw: "fixture raw nvidia-smi line",
    }],
    systemMemory: { totalBytes: 128 * 1024 ** 3, availableBytesAtStart: 100 * 1024 ** 3 },
    startedAt: "2026-08-21T12:00:00.000Z",
  });
  receipt.status = status;
  receipt.error = status === "passed" ? null : "fixture failure";
  receipt.hardware.rawVramSamples = [{ raw: "fixture raw VRAM sample" }];
  receipt.logs = [{ path: "controller.log", bytes: 7, sha256: "5".repeat(64) }];
  return receipt;
}

test("receipt fixture binds paired clean sources, exact tier, artifacts, runner, VRAM, and log hash", () => {
  const valid = fixtureReceipt();
  assert.equal(validateReceipt(valid), valid);
  const dense = fixtureReceipt();
  dense.cell.denseFallback = true;
  assert.throws(() => validateReceipt(dense), /dense or cross-tier/);
  const dirty = fixtureReceipt();
  dirty.repositories.inference.clean = false;
  assert.throws(() => validateReceipt(dirty), /clean paired/);
  const missingInventory = fixtureReceipt();
  missingInventory.artifacts[0].inventory.sha256 = null;
  assert.throws(() => validateReceipt(missingInventory), /inventory is incomplete/);

  const failedProvision = fixtureReceipt("failed");
  failedProvision.artifacts[0].inventory = {
    complete: false, sha256: null, files: 0, bytes: 0, error: "download failed before inventory",
  };
  assert.doesNotThrow(() => validateReceipt(failedProvision));
});

test("raw nvidia-smi and one exact inference pin fixtures parse deterministically", () => {
  assert.deepEqual(
    parseNvidiaSmi("2026/08/21 12:00:00, 0, NVIDIA RTX PRO 6000, GPU-abc, 00000000:01:00.0, 12.0, 590.1, 97887, 1234, 96653\n"),
    [{
      timestamp: "2026/08/21 12:00:00", index: 0, name: "NVIDIA RTX PRO 6000",
      uuid: "GPU-abc", pciBusId: "00000000:01:00.0", computeCapability: "12.0",
      driverVersion: "590.1", memoryTotalMiB: 97887, memoryUsedMiB: 1234,
      memoryFreeMiB: 96653,
      raw: "2026/08/21 12:00:00, 0, NVIDIA RTX PRO 6000, GPU-abc, 00000000:01:00.0, 12.0, 590.1, 97887, 1234, 96653",
    }],
  );
  assert.deepEqual(inferencePins(`dep = { git = "https://github.com/SceneWorks/inference", rev = "${"a".repeat(40)}" }`), ["a".repeat(40)]);
});

test("schemas and workflow preserve the opt-in single-job terminal contract", async () => {
  const [profileSchema, receiptSchema, workflow] = await Promise.all([
    readFile("config/terminal-evidence/epic-20738-profile.schema.json", "utf8").then(JSON.parse),
    readFile("config/terminal-evidence/epic-20738-receipt.schema.json", "utf8").then(JSON.parse),
    readFile(".github/workflows/windows-candle.yml", "utf8"),
  ]);
  assert.equal(profileSchema.properties.cells.minItems, 19);
  assert.equal(profileSchema.properties.cells.maxItems, 19);
  assert.deepEqual(receiptSchema.properties.cell.properties.denseFallback, { const: false });
  assert.equal(receiptSchema.properties.profile.const, PROFILE_NAME);
  assert.match(workflow, /run_epic_20738_terminal_cuda:[\s\S]*?default: false/);
  assert.match(workflow, /windows-candle-\$\{\{[^\n]*epic-20738-terminal/);
  assert.equal((workflow.match(/^jobs:\s*$/gm) ?? []).length, 1);
  assert.equal((workflow.match(/^  candle-worker:\s*$/gm) ?? []).length, 1);
  assert.doesNotMatch(workflow, /^  [a-zA-Z0-9_-]+:\s*\n\s+strategy:\s*\n\s+matrix:/m);
  assert.equal((workflow.match(/epic-20738-terminal-cuda-harness\.mjs run/g) ?? []).length, 1);
  assert.match(workflow, /id: terminal_campaign[\s\S]*?continue-on-error: true/);
  assert.match(workflow, /name: Upload every epic-20738 terminal receipt[\s\S]*?if: \$\{\{ always\(\)/);
});
