import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  PROFILE_NAME,
  PROFILE_PATH,
  PROFILE_SCHEMA_PATH,
  RECEIPT_SCHEMA_PATH,
  cellSemanticsSha256,
  inferencePins,
  loadProfile,
  parseNvidiaSmi,
  receiptSkeleton,
  runCampaign,
  safeRemoveTree,
  validateCampaignPaths,
  validateDocumentWithSchema,
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
  assert.equal(cellSemanticsSha256(checked.cells), "dc0e529b40e898727eb9401562a928345b958b4c94677d0206ccc70471f6f879");
  assert.doesNotMatch(JSON.stringify(checked).toLowerCase(), /anima|sana|vace|flux2|true[_-]?v2|eros/);

  const manifest = JSON.parse(stripJsoncComments(await readFile(
    "config/manifests/builtin.models.jsonc", "utf8",
  )));
  assert.doesNotThrow(() => validateManifestAuthorities(checked, manifest));
});

test("profile validator rejects count, order, every semantic tuple mutation, blocked, and authority drift", async () => {
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
  assert.throws(() => validateProfile(crossTier), /semantic tuples drifted/);

  const boundary = profile();
  boundary.cells.find((cell) => cell.capability === "multiReference").request.referencePairs = 5;
  assert.throws(() => validateProfile(boundary), /semantic tuples drifted/);

  const unchangedIdModelSwap = profile();
  unchangedIdModelSwap.cells[0].modelId = unchangedIdModelSwap.cells[2].modelId;
  assert.throws(() => validateProfile(unchangedIdModelSwap), /semantic tuples drifted/);

  const artifactSwap = profile();
  artifactSwap.cells[0].artifactIds[0] = artifactSwap.cells[2].artifactIds[0];
  assert.throws(() => validateProfile(artifactSwap), /semantic tuples drifted/);

  const steps = profile();
  steps.cells[0].request.steps += 1;
  assert.throws(() => validateProfile(steps), /semantic tuples drifted/);

  const unreviewedKnob = profile();
  unreviewedKnob.cells[0].request.unreviewedKnob = true;
  assert.throws(() => validateProfile(unreviewedKnob), /semantic tuples drifted/);

  const blocked = profile();
  blocked.artifacts["scail2-q4"].repository = "SceneWorks/eros-model";
  assert.throws(() => validateProfile(blocked), /blocked surface eros/);

  const manifestDrift = profile();
  manifestDrift.artifacts["chroma1-base-q4"].revision = "a".repeat(40);
  assert.throws(() => validateManifestAuthorities(manifestDrift, manifest), /artifact definitions drifted/);

  const controlDrift = profile();
  controlDrift.artifacts["sdxl-openpose"].revision = "b".repeat(40);
  assert.throws(() => validateManifestAuthorities(controlDrift, manifest), /artifact definitions drifted/);

  const controlManifestDrift = structuredClone(manifest);
  controlManifestDrift.models.find((model) => model.id === "realvisxl").downloads
    .find((download) => download.componentId === "controlnet_openpose").required = "hard";
  assert.throws(
    () => validateManifestAuthorities(profile(), controlManifestDrift),
    /manifest authority drifted for realvisxl/,
  );

  const duplicateControlAuthority = structuredClone(manifest);
  const sdxl = duplicateControlAuthority.models.find((model) => model.id === "sdxl");
  sdxl.downloads.push(structuredClone(
    sdxl.downloads.find((download) => download.componentId === "controlnet_openpose"),
  ));
  assert.throws(
    () => validateManifestAuthorities(profile(), duplicateControlAuthority),
    /exact five approved consumers/,
  );

  const extraControlConsumer = structuredClone(manifest);
  const extra = structuredClone(extraControlConsumer.models.find((model) => model.id === "sdxl")
    .downloads.find((download) => download.componentId === "controlnet_openpose"));
  extraControlConsumer.models.find((model) => model.id === "flux_dev").downloads.push(extra);
  assert.throws(
    () => validateManifestAuthorities(profile(), extraControlConsumer),
    /exact five approved consumers/,
  );

  const artifactDefinitionDrift = profile();
  artifactDefinitionDrift.artifacts["chroma1-base-q4"].allowPatterns.push("duplicate-unreviewed.bin");
  assert.throws(() => validateProfile(artifactDefinitionDrift), /artifact definitions drifted/);
});

function fixtureReceipt(status = "passed") {
  const checked = profile();
  const cell = checked.cells[0];
  const artifact = checked.artifacts[cell.artifactIds[0]];
  const selectedRoot = path.resolve("fixture-runner", "scratch", "01-chroma1-base-q4", "artifacts", cell.artifactIds[0], artifact.subdirectory);
  const receipt = receiptSkeleton({
    cell,
    ordinal: 1,
    repositories: {
      sceneworks: { sha: "1".repeat(40), clean: true },
      inference: { sha: "2".repeat(40), clean: true },
    },
    artifacts: [{
      id: cell.artifactIds[0], role: artifact.role, repository: artifact.repository,
      revision: artifact.revision, subdirectory: artifact.subdirectory,
      selectedRoot, allowPatterns: artifact.allowPatterns,
      inventory: { root: selectedRoot, complete: true, sha256: "4".repeat(64), files: 3, bytes: 42, error: null },
    }],
    execution: {
      runId: "123", runAttempt: "2", headSha: "1".repeat(40), workflow: "Windows Candle worker",
      headRef: "refs/heads/story/sc-20945-epic-20738-candle-cuda-parity",
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
  receipt.cleanup = { attempted: true, completed: true, error: null };
  receipt.inputs = [{ path: "cell.json", bytes: 7, sha256: "6".repeat(64) }];
  receipt.outputs = [{ path: "runtime-result.json", bytes: 7, sha256: "7".repeat(64) }];
  receipt.logs = [{ path: "controller.log", bytes: 7, sha256: "5".repeat(64) }];
  return receipt;
}

function validateFixture(receipt) {
  const checked = profile();
  return validateReceipt(receipt, checked.cells[0], checked);
}

test("receipt fixture binds paired clean sources, exact tier, artifacts, runner, VRAM, and log hash", () => {
  const valid = fixtureReceipt();
  assert.equal(validateFixture(valid), valid);
  const dense = fixtureReceipt();
  dense.cell.denseFallback = true;
  assert.throws(() => validateFixture(dense), /dense or cross-tier/);
  const dirty = fixtureReceipt();
  dirty.repositories.inference.clean = false;
  assert.throws(() => validateFixture(dirty), /clean paired/);
  const missingInventory = fixtureReceipt();
  missingInventory.artifacts[0].inventory.sha256 = null;
  assert.throws(() => validateFixture(missingInventory), /inventory is incomplete/);

  const failedProvision = fixtureReceipt("failed");
  failedProvision.artifacts[0].inventory = {
    root: failedProvision.artifacts[0].selectedRoot,
    complete: false, sha256: null, files: 0, bytes: 0, error: "download failed before inventory",
  };
  failedProvision.artifacts[0].selectedRoot = null;
  assert.doesNotThrow(() => validateFixture(failedProvision));

  const cellDrift = fixtureReceipt();
  cellDrift.cell.modelId = "chroma1_flash";
  assert.throws(() => validateFixture(cellDrift), /cell semantics drifted/);

  const reviewedArtifactDrift = fixtureReceipt();
  const checked = profile();
  const replacement = checked.artifacts[checked.cells[2].artifactIds[0]];
  Object.assign(reviewedArtifactDrift.artifacts[0], {
    id: checked.cells[2].artifactIds[0], role: replacement.role, repository: replacement.repository,
    revision: replacement.revision, subdirectory: replacement.subdirectory,
    allowPatterns: replacement.allowPatterns,
  });
  assert.throws(() => validateFixture(reviewedArtifactDrift), /authority binding is incomplete/);
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

test("Draft 2020-12 schemas validate the profile and close adversarial receipt drift", async () => {
  assert.doesNotThrow(() => validateDocumentWithSchema(PROFILE_SCHEMA_PATH, PROFILE_PATH));
  const temporary = await mkdtemp(path.join(tmpdir(), "sc-20945-schema-"));
  try {
    const writeReceipt = async (name, receipt) => {
      const file = path.join(temporary, name);
      await writeFile(file, `${JSON.stringify(receipt, null, 2)}\n`, "utf8");
      return file;
    };
    const valid = await writeReceipt("valid.json", fixtureReceipt());
    assert.doesNotThrow(() => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, valid));

    const openInventory = fixtureReceipt();
    openInventory.artifacts[0].inventory.injected = true;
    const openInventoryFile = await writeReceipt("open-inventory.json", openInventory);
    assert.throws(
      () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, openInventoryFile),
      /must NOT have additional properties/,
    );

    const openGpuSample = fixtureReceipt();
    openGpuSample.hardware.rawVramSamples[0].injected = true;
    const openGpuFile = await writeReceipt("open-gpu.json", openGpuSample);
    assert.throws(
      () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, openGpuFile),
      /must NOT have additional properties/,
    );

    const passedWithoutOutput = fixtureReceipt();
    passedWithoutOutput.outputs = [];
    const noOutputFile = await writeReceipt("passed-without-output.json", passedWithoutOutput);
    assert.throws(
      () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, noOutputFile),
      /must NOT have fewer than 1 items/,
    );

    const failedWithoutError = fixtureReceipt("failed");
    failedWithoutError.error = null;
    const noErrorFile = await writeReceipt("failed-without-error.json", failedWithoutError);
    assert.throws(
      () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, noErrorFile),
      /must be string/,
    );
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test("campaign paths are fresh non-nested RUNNER_TEMP descendants and cleanup rejects reparses", async (t) => {
  const temporary = await mkdtemp(path.join(tmpdir(), "sc-20945-paths-"));
  const runnerTemp = path.join(temporary, "runner-temp");
  const sceneworks = path.join(temporary, "sceneworks");
  const inference = path.join(temporary, "inference");
  const outside = path.join(temporary, "outside");
  await Promise.all([runnerTemp, sceneworks, inference, outside].map((directory) => mkdir(directory)));
  const repositories = [sceneworks, inference];
  try {
    const output = path.join(runnerTemp, "output");
    const scratch = path.join(runnerTemp, "scratch");
    const guard = await validateCampaignPaths({ runnerTemp, output, scratch, repositories });
    assert.equal(guard.output, output);
    await mkdir(scratch);
    await safeRemoveTree(scratch, guard, "fixture scratch");

    await assert.rejects(
      validateCampaignPaths({
        runnerTemp,
        output: path.join(temporary, "outside-output"),
        scratch: path.join(runnerTemp, "scratch-outside-case"),
        repositories,
      }),
      /descendant of the resolved RUNNER_TEMP/,
    );
    await assert.rejects(
      validateCampaignPaths({
        runnerTemp,
        output: path.join(runnerTemp, "nested"),
        scratch: path.join(runnerTemp, "nested", "scratch"),
        repositories,
      }),
      /distinct, non-nested/,
    );

    const existing = path.join(runnerTemp, "existing");
    await mkdir(existing);
    await assert.rejects(
      validateCampaignPaths({
        runnerTemp, output: existing, scratch: path.join(runnerTemp, "fresh"), repositories,
      }),
      /must not already exist/,
    );

    const link = path.join(runnerTemp, "outside-link");
    try {
      await symlink(outside, link, process.platform === "win32" ? "junction" : "dir");
    } catch (error) {
      if (error.code === "EPERM") {
        t.diagnostic("symlink/reparse fixture is unavailable on this host");
        return;
      }
      throw error;
    }
    await assert.rejects(
      validateCampaignPaths({
        runnerTemp,
        output: path.join(link, "escaped-output"),
        scratch: path.join(runnerTemp, "fresh-scratch"),
        repositories,
      }),
      /symlink or reparse point outside RUNNER_TEMP/,
    );

    const guardedOutput = path.join(runnerTemp, "guarded-output");
    const guardedScratch = path.join(runnerTemp, "guarded-scratch");
    const reparseGuard = await validateCampaignPaths({
      runnerTemp, output: guardedOutput, scratch: guardedScratch, repositories,
    });
    await symlink(outside, guardedScratch, process.platform === "win32" ? "junction" : "dir");
    await writeFile(path.join(outside, "keep.txt"), "keep", "utf8");
    await assert.rejects(
      safeRemoveTree(guardedScratch, reparseGuard, "replaced scratch"),
      /ordinary directory|symlink or reparse point/,
    );
    assert.equal(await readFile(path.join(outside, "keep.txt"), "utf8"), "keep");
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test("injected lifecycle faults preserve all 19 outcomes and quarantine after cleanup isolation failure", async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), "sc-20945-faults-"));
  const runnerTemp = path.join(temporary, "runner-temp");
  const sceneworks = path.join(temporary, "sceneworks");
  const inference = path.join(temporary, "inference");
  await Promise.all([runnerTemp, sceneworks, inference].map((directory) => mkdir(directory)));
  const output = path.join(runnerTemp, "output");
  const scratch = path.join(runnerTemp, "scratch");
  const guard = await validateCampaignPaths({
    runnerTemp, output, scratch, repositories: [sceneworks, inference],
  });
  await mkdir(output);
  await mkdir(scratch);
  const checked = profile();
  const repositories = {
    sceneworks: { sha: "1".repeat(40), clean: true },
    inference: { sha: "2".repeat(40), clean: true },
  };
  const execution = {
    runId: "fault-run", runAttempt: "1", headSha: "1".repeat(40), headRef: "refs/heads/fault",
    workflow: "fixture", runnerName: "fixture-runner", runnerOs: "Windows", runnerArch: "X64",
  };
  const gpuIdentity = [{
    timestamp: "2026/08/21 12:00:00", index: 0, name: "fixture GPU", uuid: "GPU-fixture",
    pciBusId: "00000000:01:00.0", computeCapability: "12.0", driverVersion: "999.1",
    memoryTotalMiB: 49140, memoryUsedMiB: 10, memoryFreeMiB: 49130, raw: "fixture raw",
  }];
  const attemptedCells = [];
  const provisionedCells = [];
  const executedCells = [];
  const injectedStages = [];
  const fault = async (stage, index) => {
    if (stage === "setup") attemptedCells.push(index);
    const shouldFail = (index === 0 && new Set(["setup", "emergencyReceipt"]).has(stage))
      || (index === 1 && new Set(["finalLog", "fallbackLog"]).has(stage))
      || (index === 2 && new Set(["evidenceHash", "evidenceRehash"]).has(stage))
      || (index === 3 && stage === "semanticValidation")
      || (index === 4 && stage === "schemaValidation")
      || (index === 5 && stage === "receiptWrite")
      || (index === 6 && stage === "cleanup")
      || (index === null && stage === "summaryWrite");
    if (shouldFail) {
      injectedStages.push(`${index}:${stage}`);
      throw new Error(`injected ${stage} failure for ${index}`);
    }
  };
  const operations = {
    provisionArtifact: async ({ id, artifact, scratch: cellScratch }) => {
      provisionedCells.push(path.basename(cellScratch));
      const selectedRoot = path.resolve(cellScratch, "artifacts", id, artifact.subdirectory);
      return {
        id, ...artifact, selectedRoot,
        inventory: { root: selectedRoot, files: 1, bytes: 7, sha256: "a".repeat(64) },
      };
    },
    executeCell: async ({ cellDir, logFile }) => {
      await writeFile(logFile, "fixture runtime\n", "utf8");
      const runtimeCell = JSON.parse(await readFile(path.join(cellDir, "cell.json"), "utf8"));
      executedCells.push(runtimeCell.id);
      await writeFile(path.join(cellDir, "runtime-result.json"), `${JSON.stringify({
        requestedTier: runtimeCell.requestedTier,
        resolvedTier: runtimeCell.requestedTier,
        denseFallback: false,
      })}\n`, "utf8");
      await writeFile(path.join(cellDir, "output.bin"), "fixture", "utf8");
    },
    sample: () => {},
  };
  try {
    const result = await runCampaign({}, {
      prepared: {
        profile: checked, repositories, execution, guard, output, scratch,
        startupErrors: [], gpuIdentity,
        systemMemory: { totalBytes: 128 * 1024 ** 3, availableBytesAtStart: 100 * 1024 ** 3 },
        sceneworksRoot: sceneworks,
      },
      fault, operations, suppressVerdict: true,
    });
    assert.deepEqual(attemptedCells, Array.from({ length: 7 }, (_, index) => index));
    assert.ok(provisionedCells.length > 0);
    assert.ok(provisionedCells.every((name) => /^0[1-7]-/.test(name)));
    assert.match(provisionedCells.at(-1), /^07-/);
    assert.ok(executedCells.length > 0);
    assert.ok(executedCells.every((id) => checked.cells.slice(0, 7).some((cell) => cell.id === id)));
    assert.equal(executedCells.at(-1), checked.cells[6].id);
    assert.equal(result.summary.receipts.length, 19);
    assert.equal(new Set(result.summary.receipts.map((receipt) => receipt.id)).size, 19);
    assert.equal(result.summaryPath, "_emergency/campaign-summary-fallback.json");
    assert.match(result.summary.campaignErrors.join("\n"), /primary campaign summary write failed/);
    for (const [index, outcome] of result.summary.receipts.entries()) {
      assert.ok(outcome.receipt, `${outcome.id} must retain a receipt path`);
      const receiptFile = path.join(output, ...outcome.receipt.split("/"));
      const receipt = JSON.parse(await readFile(receiptFile, "utf8"));
      assert.doesNotThrow(() => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt));
      assert.doesNotThrow(() => validateReceipt(receipt, checked.cells[index], checked));
    }
    for (const index of [0, 1, 2, 3, 4, 5]) {
      assert.match(result.summary.receipts[index].receipt, /^_emergency\//);
      assert.equal(result.summary.receipts[index].emergencyReceiptError, null);
    }
    assert.match(result.summary.receipts[0].receipt, /receipt-last-resort\.json$/);
    assert.match(result.summary.receipts[0].error, /emergency receipt failed/);
    assert.doesNotMatch(result.summary.receipts[6].receipt, /^_emergency\//);
    assert.match(result.summary.campaignErrors.join("\n"), /prior cell .* scratch cleanup isolation failed/);
    for (const outcome of result.summary.receipts.slice(7)) {
      assert.match(outcome.receipt, /^_emergency\//);
      assert.match(outcome.error, new RegExp(`prior cell ${checked.cells[6].id} scratch cleanup isolation failed`));
    }
    const durableSummary = JSON.parse(await readFile(path.join(output, ...result.summaryPath.split("/")), "utf8"));
    assert.equal(durableSummary.receipts.length, 19);
    assert.ok(injectedStages.includes("null:summaryWrite"));
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test("schemas, clean Node dependencies, and workflow preserve the opt-in single-job terminal contract", async () => {
  const [profileSchema, receiptSchema, workflow, checks, packageJson, packageLock] = await Promise.all([
    readFile("config/terminal-evidence/epic-20738-profile.schema.json", "utf8").then(JSON.parse),
    readFile("config/terminal-evidence/epic-20738-receipt.schema.json", "utf8").then(JSON.parse),
    readFile(".github/workflows/windows-candle.yml", "utf8"),
    readFile(".github/workflows/check.yml", "utf8"),
    readFile("package.json", "utf8").then(JSON.parse),
    readFile("package-lock.json", "utf8").then(JSON.parse),
  ]);
  const normalizedWorkflow = workflow.replace(/\r\n/g, "\n");
  const terminalStep = (document, id) => {
    const lines = document.split("\n");
    const idIndex = lines.findIndex((line) => line.trim() === `id: ${id}`);
    assert.notEqual(idIndex, -1, `missing terminal step ${id}`);
    let start = idIndex;
    while (start >= 0 && !lines[start].startsWith("      - ")) start -= 1;
    let end = idIndex + 1;
    while (end < lines.length && !lines[end].startsWith("      - ")) end += 1;
    return lines.slice(start, end).join("\n");
  };
  const requiresOutcomes = (block, ids, label) => {
    for (const id of ids) {
      assert.match(
        block,
        new RegExp(`steps\\.${id}\\.outcome == 'success'`),
        `${label} must require ${id} success`,
      );
    }
  };
  const assertTerminalWorkflowGates = (document) => {
    const node = terminalStep(document, "terminal_node");
    const npm = terminalStep(document, "terminal_npm");
    const tests = terminalStep(document, "terminal_tests");
    const validation = terminalStep(document, "terminal_validation");
    const python = terminalStep(document, "terminal_python");
    const inference = terminalStep(document, "terminal_inference");
    const campaign = terminalStep(document, "terminal_campaign");
    const verdict = terminalStep(document, "terminal_verdict");
    assert.match(node, /continue-on-error: true/);
    assert.match(npm, /continue-on-error: true/);
    assert.match(tests, /continue-on-error: true/);
    requiresOutcomes(npm, ["terminal_node"], "npm install");
    requiresOutcomes(tests, ["terminal_node", "terminal_npm"], "targeted tests");
    requiresOutcomes(validation, ["terminal_node", "terminal_npm", "terminal_tests"], "profile check");
    requiresOutcomes(
      python,
      ["terminal_node", "terminal_npm", "terminal_tests", "terminal_validation"],
      "Python setup",
    );
    requiresOutcomes(
      inference,
      ["terminal_node", "terminal_npm", "terminal_tests", "terminal_validation"],
      "inference checkout",
    );
    requiresOutcomes(
      campaign,
      ["terminal_node", "terminal_npm", "terminal_tests", "terminal_validation", "terminal_python", "terminal_inference"],
      "terminal campaign",
    );
    for (const [name, id] of [
      ["NODE_OUTCOME", "terminal_node"],
      ["NPM_OUTCOME", "terminal_npm"],
      ["TESTS_OUTCOME", "terminal_tests"],
      ["VALIDATION_OUTCOME", "terminal_validation"],
      ["PYTHON_OUTCOME", "terminal_python"],
      ["INFERENCE_OUTCOME", "terminal_inference"],
      ["CAMPAIGN_OUTCOME", "terminal_campaign"],
    ]) {
      assert.match(verdict, new RegExp(`${name}: \\$\\{\\{ steps\\.${id}\\.outcome \\}\\}`));
    }
    assert.match(verdict, /Where-Object \{ \$_ -ne 'success' \}/);
  };
  assertTerminalWorkflowGates(normalizedWorkflow);
  for (const id of ["terminal_node", "terminal_npm", "terminal_tests"]) {
    const relaxed = normalizedWorkflow.replaceAll(
      `steps.${id}.outcome == 'success'`,
      "true",
    );
    assert.throws(() => assertTerminalWorkflowGates(relaxed), new RegExp(id));
  }
  for (const [name, id] of [
    ["NODE_OUTCOME", "terminal_node"],
    ["NPM_OUTCOME", "terminal_npm"],
    ["TESTS_OUTCOME", "terminal_tests"],
  ]) {
    const unboundVerdict = normalizedWorkflow.replace(
      `${name}: \${{ steps.${id}.outcome }}`,
      `${name}: success`,
    );
    assert.throws(() => assertTerminalWorkflowGates(unboundVerdict), new RegExp(name));
  }
  assert.equal(profileSchema.properties.cells.minItems, 19);
  assert.equal(profileSchema.properties.cells.maxItems, 19);
  assert.deepEqual(receiptSchema.properties.cell.properties.denseFallback, { const: false });
  assert.equal(receiptSchema.$defs.inventory.additionalProperties, false);
  assert.equal(receiptSchema.$defs.gpuSample.additionalProperties, false);
  assert.equal(receiptSchema.properties.profile.const, PROFILE_NAME);
  assert.match(workflow, /run_epic_20738_terminal_cuda:[\s\S]*?default: false/);
  assert.match(workflow, /windows-candle-\$\{\{[^\n]*epic-20738-terminal/);
  assert.equal((workflow.match(/^jobs:\s*$/gm) ?? []).length, 1);
  assert.equal((workflow.match(/^  candle-worker:\s*$/gm) ?? []).length, 1);
  assert.doesNotMatch(workflow, /^  [a-zA-Z0-9_-]+:\s*\n\s+strategy:\s*\n\s+matrix:/m);
  assert.equal((workflow.match(/epic-20738-terminal-cuda-harness\.mjs run/g) ?? []).length, 1);
  assert.match(workflow, /id: terminal_campaign[\s\S]*?continue-on-error: true/);
  assert.doesNotMatch(workflow, /jsonschema|harness\.mjs check --python/);
  assert.match(
    workflow,
    /npm ci --ignore-scripts[\s\S]*?node --test scripts\/epic-20738-terminal-cuda-harness\.test\.mjs scripts\/hash-artifact-inventory\.test\.mjs[\s\S]*?harness\.mjs check/,
  );
  assert.equal(packageJson.devDependencies.ajv, "8.20.0");
  assert.equal(packageJson.devDependencies["ajv-formats"], "3.0.1");
  assert.equal(packageLock.packages[""].devDependencies.ajv, "8.20.0");
  assert.match(checks, /parity-scaffold:[\s\S]*?npm ci --ignore-scripts[\s\S]*?npm run check/);
  assert.match(workflow, /name: Upload every epic-20738 terminal receipt[\s\S]*?if: \$\{\{ always\(\)/);
});
