import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { cp, lstat, mkdtemp, mkdir, readdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  PROFILE_NAME,
  PROFILE_PATH,
  PROFILE_SCHEMA_PATH,
  RECEIPT_SCHEMA_PATH,
  CACHE_PREFLIGHT_SCHEMA_PATH,
  cellSemanticsSha256,
  directoryInventory,
  expectedArtifactFilesFromEvidence,
  expectedLoadSpecQuantBits,
  inferencePins,
  hashedFiles,
  installSidecarObstructions,
  importPrefixEvidence,
  loadProfile,
  parseNvidiaSmi,
  receiptSkeleton,
  runCampaign,
  safeRemoveTree,
  selectImportedPrefix,
  validateCampaignPaths,
  validateCachePreflightEvidence,
  validateDocumentWithSchema,
  validateManifestAuthorities,
  validateProfile,
  validateReceipt,
  validateRuntimeResult,
  verifyArtifactUnchanged,
} from "./epic-20738-terminal-cuda-harness.mjs";
import { hashArtifactInventory } from "./hash-artifact-inventory.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

const profile = () => structuredClone(loadProfile());

function fixtureSidecarObstructions(sharedArtifacts) {
  return [...sharedArtifacts.values()].flatMap((artifact) => {
    const roots = new Set([artifact.selectedRoot]);
    for (const file of artifact.matchedFiles ?? []) {
      const [component] = file.split("/");
      if (file.includes("/")) roots.add(path.join(artifact.selectedRoot, component));
    }
    const records = [...roots].map((root) => ({
      artifactIds: [artifact.id], root, path: ".candle-device-format-v1",
      bytes: 7, sha256: "e".repeat(64),
    }));
    artifact.sidecarObstructions = records;
    return records;
  });
}

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
  assert.deepEqual(
    ["ltx23-q8", "ltx23-gemma"].map((id) => checked.artifacts[id].revision),
    [
      "254989c3ca7ee691187647f350b112c0c448789d",
      "254989c3ca7ee691187647f350b112c0c448789d",
    ],
    "terminal LTX must truthfully use the exact cached production-approved parent",
  );

  const manifest = JSON.parse(stripJsoncComments(await readFile(
    "config/manifests/builtin.models.jsonc", "utf8",
  )));
  assert.doesNotThrow(() => validateManifestAuthorities(checked, manifest));
});

test("download evidence freezes the exact 23-authority filename census", async () => {
  const checked = profile();
  const evidence = JSON.parse(await readFile("config/download-pattern-evidence.json", "utf8"));
  const expected = expectedArtifactFilesFromEvidence(checked, evidence);
  assert.equal(Object.keys(expected).length, 23);
  assert.equal(Object.values(expected).flat().length, 315);
  assert.equal(expected["flux1-schnell-q4"].length, 19);
  assert.equal(expected["flux1-schnell-q8"].length, 19);
  assert.equal(expected["ltx23-q8"].length, 11);
  assert.ok(Object.values(expected).flat().every((file) => (
    !file.includes(".candle-device-format-v1") && !file.endsWith(".incomplete")
  )));

  const missingRow = structuredClone(evidence);
  missingRow.repos = missingRow.repos.filter((row) => row.key !== (
    "SceneWorks/flux1-schnell-mlx@bba3ae01dfd94089f173c05edd4e1a4c551f2599"
  ));
  assert.throws(
    () => expectedArtifactFilesFromEvidence(checked, missingRow),
    /missing exact authority/,
  );
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

  const ltxCurrentInsteadOfApprovedCache = profile();
  ltxCurrentInsteadOfApprovedCache.artifacts["ltx23-q8"].revision =
    "01df27d308466533aa09d251e3aebdcc627d07eb";
  assert.throws(
    () => validateProfile(ltxCurrentInsteadOfApprovedCache),
    /artifact definitions drifted/,
  );
});

test("all 19 cells require the exact family loadSpec quant policy", () => {
  const checked = profile();
  const sdxlCells = new Set([
    "sdxl-openpose", "realvisxl-openpose", "realvisxl-lightning-openpose",
    "illustrious-v1-openpose", "illustrious-v2-openpose",
  ]);
  assert.equal(checked.cells.length, 19);
  for (const [index, cell] of checked.cells.entries()) {
    const expected = sdxlCells.has(cell.id)
      ? 4
      : cell.kind === "scail2" ? Number(cell.requestedTier.slice(1)) : null;
    assert.equal(expectedLoadSpecQuantBits(cell), expected, cell.id);
    const valid = {
      requestedTier: cell.requestedTier,
      resolvedTier: cell.requestedTier,
      denseFallback: false,
      loadSpecQuantBits: expected,
    };
    assert.equal(validateRuntimeResult(valid, cell), valid, cell.id);
    const missing = { ...valid };
    delete missing.loadSpecQuantBits;
    assert.throws(() => validateRuntimeResult(missing, cell), /loadSpecQuantBits/, cell.id);
    assert.throws(
      () => validateRuntimeResult({ ...valid, loadSpecQuantBits: expected === null ? 4 : null }, cell),
      /loadSpecQuantBits/,
      cell.id,
    );
    const receipt = fixtureReceipt("passed", index);
    assert.equal(validateReceipt(receipt, cell, checked), receipt, cell.id);
    delete receipt.cell.loadSpecQuantBits;
    assert.throws(() => validateReceipt(receipt, cell, checked), /loadSpecQuantBits/, cell.id);
    receipt.cell.loadSpecQuantBits = expected === null ? 4 : null;
    assert.throws(() => validateReceipt(receipt, cell, checked), /loadSpecQuantBits/, cell.id);
  }
});

function fixtureReceipt(status = "passed", cellIndex = 0) {
  const checked = profile();
  const cell = checked.cells[cellIndex];
  const artifacts = cell.artifactIds.map((artifactId) => {
    const artifact = checked.artifacts[artifactId];
    const selectedRoot = path.resolve(
      "fixture-runner", "scratch", `${String(cellIndex + 1).padStart(2, "0")}-${cell.id}`,
      "artifacts", artifactId, artifact.subdirectory,
    );
    return {
      id: artifactId, role: artifact.role, repository: artifact.repository,
      revision: artifact.revision, subdirectory: artifact.subdirectory,
      selectedRoot, allowPatterns: artifact.allowPatterns,
      inventory: {
        root: selectedRoot, complete: true, sha256: "4".repeat(64), files: 3, bytes: 42, error: null,
      },
    };
  });
  const receipt = receiptSkeleton({
    cell,
    ordinal: cellIndex + 1,
    repositories: {
      sceneworks: { sha: "1".repeat(40), clean: true },
      inference: { sha: "2".repeat(40), clean: true },
    },
    artifacts,
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
  const missingQuant = fixtureReceipt();
  delete missingQuant.cell.loadSpecQuantBits;
  assert.throws(() => validateFixture(missingQuant), /loadSpecQuantBits/);
  const wrongQuant = fixtureReceipt();
  wrongQuant.cell.loadSpecQuantBits = 4;
  assert.throws(() => validateFixture(wrongQuant), /loadSpecQuantBits/);
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

    const missingQuant = fixtureReceipt();
    delete missingQuant.cell.loadSpecQuantBits;
    const missingQuantFile = await writeReceipt("missing-load-quant.json", missingQuant);
    assert.throws(
      () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, missingQuantFile),
      /must have required property 'loadSpecQuantBits'/,
    );

    const wrongQuant = fixtureReceipt();
    wrongQuant.cell.loadSpecQuantBits = 4;
    const wrongQuantFile = await writeReceipt("wrong-load-quant.json", wrongQuant);
    assert.throws(
      () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, wrongQuantFile),
      /must be equal to constant/,
    );

    const scailQ8 = fixtureReceipt("passed", 12);
    const scailQ8File = await writeReceipt("scail-q8.json", scailQ8);
    assert.doesNotThrow(() => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, scailQ8File));
    scailQ8.cell.loadSpecQuantBits = 4;
    const wrongScailQ8File = await writeReceipt("wrong-scail-q8.json", scailQ8);
    assert.throws(
      () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, wrongScailQ8File),
      /must be equal to constant/,
    );

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

async function writePrefixCandidate(root, candidateName = "12345") {
  const candidate = path.join(root, candidateName);
  const evidence = path.join(candidate, "evidence");
  await mkdir(evidence, { recursive: true });
  const runId = "32570707303";
  const runAttempt = "1";
  for (let index = 0; index < 7; index += 1) {
    const cell = profile().cells[index];
    const ordinal = `${String(index + 1).padStart(2, "0")}-${cell.id}`;
    const cellDir = path.join(evidence, ordinal);
    await mkdir(cellDir);
    await writeFile(path.join(cellDir, "cell.json"), `{"id":"${cell.id}"}\n`, "utf8");
    await writeFile(path.join(cellDir, "runtime-result.json"), '{"passed":true}\n', "utf8");
    await writeFile(path.join(cellDir, "controller.log"), "complete\n", "utf8");
    const receipt = fixtureReceipt("passed", index);
    receipt.repositories.sceneworks.sha = "8886a9e69f26beec05688c81b414859bd102f6d0";
    receipt.repositories.inference.sha = "b646a6f89ba9f6b07efe53dd583d8a42e21e9871";
    receipt.execution.headSha = receipt.repositories.sceneworks.sha;
    receipt.execution.runId = runId;
    receipt.execution.runAttempt = runAttempt;
    const files = await hashedFiles(cellDir);
    receipt.inputs = files.filter((file) => file.path === "cell.json");
    receipt.outputs = files.filter((file) => file.path === "runtime-result.json");
    receipt.logs = files.filter((file) => file.path === "controller.log");
    await writeFile(
      path.join(cellDir, "receipt.json"),
      `${JSON.stringify(receipt, null, 2)}\n`,
      "utf8",
    );
  }
  const boundaryIndex = 7;
  const boundaryCell = profile().cells[boundaryIndex];
  const boundaryOrdinal = `${String(boundaryIndex + 1).padStart(2, "0")}-${boundaryCell.id}`;
  const boundaryDir = path.join(evidence, boundaryOrdinal);
  await mkdir(boundaryDir);
  await writeFile(path.join(boundaryDir, "controller.log"), `starting ${boundaryCell.id}\n`, "utf8");
  const boundary = receiptSkeleton({
    cell: boundaryCell,
    ordinal: boundaryIndex + 1,
    repositories: {
      sceneworks: { sha: "8886a9e69f26beec05688c81b414859bd102f6d0", clean: true },
      inference: { sha: "b646a6f89ba9f6b07efe53dd583d8a42e21e9871", clean: true },
    },
    artifacts: boundaryCell.artifactIds.map((id) => ({
      id,
      role: profile().artifacts[id].role,
      repository: profile().artifacts[id].repository,
      revision: profile().artifacts[id].revision,
      subdirectory: profile().artifacts[id].subdirectory,
      selectedRoot: null,
      allowPatterns: profile().artifacts[id].allowPatterns,
      inventory: {
        root: `E:/trusted/${id}`,
        complete: false,
        sha256: null,
        files: 0,
        bytes: 0,
        error: "provisioning has not completed",
      },
    })),
    execution: {
      runId,
      runAttempt,
      headSha: "8886a9e69f26beec05688c81b414859bd102f6d0",
      workflow: "Windows Candle worker",
      headRef: "refs/heads/feature/sc-20738-candle-cuda-parity",
      runnerName: "cuda-windows",
      runnerOs: "Windows",
      runnerArch: "X64",
    },
    gpuIdentity: [{
      index: 0, name: "fixture GPU", uuid: "GPU-fixture", pciBusId: "00000000:01:00.0",
      computeCapability: "12.0", driverVersion: "999.1", memoryTotalMiB: 49140,
      memoryUsedMiB: 10, memoryFreeMiB: 49130, raw: "fixture raw nvidia-smi line",
    }],
    systemMemory: { totalBytes: 128 * 1024 ** 3, availableBytesAtStart: 100 * 1024 ** 3 },
    startedAt: "2026-08-22T00:39:59.000Z",
  });
  boundary.hardware.rawVramSamples = [{ raw: "fixture raw nvidia-smi line" }];
  boundary.logs = await hashedFiles(boundaryDir);
  await writeFile(
    path.join(boundaryDir, "receipt.json"),
    `${JSON.stringify(boundary, null, 2)}\n`,
    "utf8",
  );
  const metadata = {
    artifactId: candidateName,
    artifactName: `sc-20945-epic-20738-8886a9e69f26beec05688c81b414859bd102f6d0-${runId}-${runAttempt}`,
    artifactDigest: `sha256:${"f".repeat(64)}`,
    runId,
    runAttempt,
    headSha: "8886a9e69f26beec05688c81b414859bd102f6d0",
    inferenceSha: "b646a6f89ba9f6b07efe53dd583d8a42e21e9871",
    profile: PROFILE_NAME,
    cellSemanticsSha256: "dc0e529b40e898727eb9401562a928345b958b4c94677d0206ccc70471f6f879",
    artifactSemanticsSha256: "f2bb7a77b83ce11cc32c3a1f9639534a67a149bc464a9730fb5c0988b4a03f9e",
  };
  await writeFile(
    path.join(candidate, "artifact-metadata.json"),
    `${JSON.stringify(metadata, null, 2)}\n`,
    "utf8",
  );
  return { candidate, evidence, metadata };
}

test("prefix discovery accepts exactly one rehashed contiguous old-profile PASS prefix", async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), "sc-20945-prefix-"));
  try {
    const validRoot = path.join(temporary, "valid");
    await mkdir(validRoot);
    const fixture = await writePrefixCandidate(validRoot);
    const selected = await selectImportedPrefix(validRoot, profile());
    assert.equal(selected.metadata.runId, fixture.metadata.runId);
    assert.deepEqual(selected.receipts.map(({ cell }) => cell.id), profile().cells.slice(0, 7).map(({ id }) => id));
    assert.equal(selected.boundaryResidue.cell.id, "flux1-dev-q8");
    assert.deepEqual(selected.boundaryResidue.files.map(({ path: file }) => file), ["controller.log"]);
    const importedOutput = path.join(temporary, "imported-output");
    await mkdir(importedOutput);
    const imported = await importPrefixEvidence(selected, importedOutput);
    assert.equal(imported.outcomes.length, 7);
    assert.deepEqual(
      (await readdir(path.join(importedOutput, "_imported-prefix"))).sort(),
      [
        "01-chroma1-base-q4", "02-chroma1-base-q8", "03-chroma1-flash-q4",
        "04-chroma1-flash-q8", "05-chroma1-hd-q4", "06-chroma1-hd-q8",
        "07-flux1-dev-q4", "lineage.json",
      ],
    );
    assert.deepEqual(
      await readdir(path.join(importedOutput, "_imported-boundary-residue")),
      ["08-flux1-dev-q8"],
    );
    assert.equal(
      imported.lineage.quarantinedBoundaryResidue.disposition,
      "non-executed-pre-provision-skeleton-excluded-from-prefix",
    );

    const tamperedRoot = path.join(temporary, "tampered");
    await mkdir(tamperedRoot);
    const tampered = await writePrefixCandidate(tamperedRoot);
    await writeFile(path.join(tampered.evidence, "01-chroma1-base-q4", "controller.log"), "changed\n");
    await assert.rejects(
      selectImportedPrefix(tamperedRoot, profile()),
      /expected exactly one valid.*failed logs rehash/,
    );

    const executedBoundaryRoot = path.join(temporary, "executed-boundary");
    await mkdir(executedBoundaryRoot);
    const executedBoundary = await writePrefixCandidate(executedBoundaryRoot);
    await writeFile(
      path.join(executedBoundary.evidence, "08-flux1-dev-q8", "cell.json"),
      '{"id":"flux1-dev-q8"}\n',
      "utf8",
    );
    await assert.rejects(
      selectImportedPrefix(executedBoundaryRoot, profile()),
      /expected exactly one valid.*boundary residue must contain only/,
    );

    const duplicateRoot = path.join(temporary, "duplicate");
    await mkdir(duplicateRoot);
    const first = await writePrefixCandidate(duplicateRoot, "12345");
    const duplicate = path.join(duplicateRoot, "67890");
    await cp(first.candidate, duplicate, { recursive: true });
    const duplicateMetadata = JSON.parse(await readFile(path.join(duplicate, "artifact-metadata.json"), "utf8"));
    duplicateMetadata.artifactId = "67890";
    await writeFile(
      path.join(duplicate, "artifact-metadata.json"),
      `${JSON.stringify(duplicateMetadata, null, 2)}\n`,
      "utf8",
    );
    await assert.rejects(
      selectImportedPrefix(duplicateRoot, profile()),
      /expected exactly one valid uploaded contiguous PASS prefix; found 2/,
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
  const cacheRoot = outside;
  try {
    const output = path.join(runnerTemp, "output");
    const scratch = path.join(runnerTemp, "scratch");
    const guard = await validateCampaignPaths({
      runnerTemp, output, scratch, cacheRoot, repositories,
    });
    assert.equal(guard.output, output);
    await mkdir(scratch);
    await safeRemoveTree(scratch, guard, "fixture scratch");

    await assert.rejects(
      validateCampaignPaths({
        runnerTemp,
        output: path.join(temporary, "outside-output"),
        scratch: path.join(runnerTemp, "scratch-outside-case"),
        cacheRoot,
        repositories,
      }),
      /descendant of the resolved RUNNER_TEMP/,
    );
    await assert.rejects(
      validateCampaignPaths({
        runnerTemp,
        output: path.join(runnerTemp, "nested"),
        scratch: path.join(runnerTemp, "nested", "scratch"),
        cacheRoot,
        repositories,
      }),
      /distinct, non-nested/,
    );

    const existing = path.join(runnerTemp, "existing");
    await mkdir(existing);
    await assert.rejects(
      validateCampaignPaths({
        runnerTemp, output: existing, scratch: path.join(runnerTemp, "fresh"), cacheRoot, repositories,
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
        cacheRoot,
        repositories,
      }),
      /symlink or reparse point outside RUNNER_TEMP/,
    );
    await assert.rejects(
      validateCampaignPaths({
        runnerTemp,
        output: path.join(runnerTemp, "cache-link-output"),
        scratch: path.join(runnerTemp, "cache-link-scratch"),
        cacheRoot: link,
        repositories,
      }),
      /trusted cache root.*symlink|trusted cache root.*reparse/,
    );

    const guardedOutput = path.join(runnerTemp, "guarded-output");
    const guardedScratch = path.join(runnerTemp, "guarded-scratch");
    const reparseGuard = await validateCampaignPaths({
      runnerTemp, output: guardedOutput, scratch: guardedScratch, cacheRoot, repositories,
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

test("staged authorities obstruct adjacent Candle sidecars and use a separate derived cache", async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), "sc-20974-sidecar-"));
  const staging = path.join(temporary, "stage");
  const selected = path.join(staging, "models--fixture--model", "snapshots", "a".repeat(40), "q4");
  const unprotected = path.join(temporary, "unprotected");
  const derived = path.join(temporary, "derived");
  try {
    await Promise.all([
      mkdir(selected, { recursive: true }), mkdir(unprotected), mkdir(derived),
    ]);
    await mkdir(path.join(selected, "transformer"));
    await writeFile(path.join(selected, "transformer", "weights.bin"), "weights", "utf8");
    const inventory = await hashArtifactInventory(selected, {
      includeFiles: ["transformer/weights.bin"], trustedRoot: staging,
    });
    const artifact = {
      id: "fixture-q4",
      selectedRoot: selected,
      selectedFiles: ["transformer/weights.bin"],
      matchedFiles: ["transformer/weights.bin"],
      inventory,
    };
    const artifacts = new Map([[artifact.id, artifact]]);
    const records = await installSidecarObstructions(artifacts);
    assert.equal(records.length, 2);
    for (const component of [selected, path.join(selected, "transformer")]) {
      const obstruction = path.join(component, ".candle-device-format-v1");
      const metadata = await lstat(obstruction);
      assert.equal(metadata.isFile(), true);
      await assert.rejects(mkdir(obstruction), /EEXIST/);
    }
    await verifyArtifactUnchanged(artifact, staging);

    const adjacent = path.join(unprotected, ".candle-device-format-v1");
    await mkdir(adjacent);
    await writeFile(path.join(adjacent, "derived.bin"), "would mutate adjacency", "utf8");
    assert.equal((await directoryInventory(unprotected)).files, 1);
    const initialDerived = await directoryInventory(derived);
    assert.equal(initialDerived.files, 0);
    assert.equal(initialDerived.bytes, 0);

    const controller = await readFile("scripts/epic-20738-terminal-cuda-harness.mjs", "utf8");
    assert.match(controller, /SCENEWORKS_CANDLE_DEVICE_CACHE_DIR: derivedSidecarRoot/);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test("injected lifecycle faults preserve all 19 outcomes and quarantine after cleanup isolation failure", async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), "sc-20945-faults-"));
  const runnerTemp = path.join(temporary, "runner-temp");
  const sceneworks = path.join(temporary, "sceneworks");
  const inference = path.join(temporary, "inference");
  const cacheRoot = path.join(temporary, "cache");
  await Promise.all([runnerTemp, sceneworks, inference, cacheRoot].map((directory) => mkdir(directory)));
  const output = path.join(runnerTemp, "output");
  const scratch = path.join(runnerTemp, "scratch");
  const guard = await validateCampaignPaths({
    runnerTemp, output, scratch, cacheRoot, repositories: [sceneworks, inference],
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
  const preflightCalls = [];
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
    auditArtifact: async ({ id, artifact, scratch: cellScratch }) => {
      preflightCalls.push(`audit:${path.basename(cellScratch)}:${id}`);
      return {
        id, ...artifact, complete: true, missingFiles: [],
        reusedFiles: [{ path: "weights.bin", bytes: 7, sha256: "a".repeat(64) }],
      };
    },
    stageArtifact: async ({ id, artifact, scratch: cellScratch }) => {
      preflightCalls.push(`stage:${path.basename(cellScratch)}:${id}`);
      return {
        id, ...artifact,
        reusedFiles: [{ path: "weights.bin", bytes: 7, sha256: "a".repeat(64) }],
        downloadedFiles: [],
      };
    },
    provisionArtifact: async ({ id, artifact, scratch: cellScratch, cacheRoot: stagedRoot }) => {
      preflightCalls.push(`offline:${path.basename(cellScratch)}:${id}`);
      const selectedRoot = path.resolve(stagedRoot, "artifacts", id, artifact.subdirectory);
      return {
        id, ...artifact, selectedRoot, matchedFiles: ["weights.bin"], selectedFiles: ["weights.bin"],
        inventory: { root: selectedRoot, files: 1, bytes: 7, sha256: "a".repeat(64) },
      };
    },
    installSidecarObstructions: async (artifacts) => fixtureSidecarObstructions(artifacts),
    directoryInventory: async (root) => ({
      root, files: 0, bytes: 0, sha256: createHash("sha256").digest("hex"),
    }),
    executeCell: async ({ cellDir, logFile }) => {
      await writeFile(logFile, "fixture runtime\n", "utf8");
      const runtimeCell = JSON.parse(await readFile(path.join(cellDir, "cell.json"), "utf8"));
      executedCells.push(runtimeCell.id);
      await writeFile(path.join(cellDir, "runtime-result.json"), `${JSON.stringify({
        requestedTier: runtimeCell.requestedTier,
        resolvedTier: runtimeCell.requestedTier,
        denseFallback: false,
        loadSpecQuantBits: expectedLoadSpecQuantBits(runtimeCell),
      })}\n`, "utf8");
      await writeFile(path.join(cellDir, "output.bin"), "fixture", "utf8");
    },
    verifyArtifactUnchanged: async () => {},
    sample: () => {},
  };
  try {
    const result = await runCampaign({}, {
      prepared: {
        profile: checked, repositories, execution, guard, output, scratch,
        startupErrors: [], gpuIdentity,
        systemMemory: { totalBytes: 128 * 1024 ** 3, availableBytesAtStart: 100 * 1024 ** 3 },
        sceneworksRoot: sceneworks,
        artifactExpectedFiles: Object.fromEntries(Object.keys(checked.artifacts).map(
          (id) => [id, ["weights.bin"]],
        )),
        downloadEvidenceSha256: "d".repeat(64),
      },
      fault, operations, suppressVerdict: true, startIndex: 0,
    });
    assert.deepEqual(
      attemptedCells,
      Array.from({ length: 7 }, (_, index) => index),
      result.summary.campaignErrors.join("\n"),
    );
    assert.equal(preflightCalls.length, 23 * 3);
    assert.ok(preflightCalls.every((call) => call.includes(":cache-preflight:")));
    assert.deepEqual(
      preflightCalls.map((call) => call.split(":")[0]),
      [
        ...Array(23).fill("audit"),
        ...Array(23).fill("stage"),
        ...Array(23).fill("offline"),
      ],
    );
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

test("continuation freezes census, stages once, downloads only the reviewed miss, and reuses authorities", async () => {
  async function scenario({
    missingId = null, downloadSucceeds = false, mutateCache = false, mutateSourceAtStage = false,
    preflightFault = null, omitObstructions = false,
  } = {}) {
    const temporary = await mkdtemp(path.join(tmpdir(), "sc-20974-preflight-"));
    const runnerTemp = path.join(temporary, "runner-temp");
    const sceneworks = path.join(temporary, "sceneworks");
    const inference = path.join(temporary, "inference");
    const cacheRoot = path.join(temporary, "cache");
    await Promise.all([runnerTemp, sceneworks, inference, cacheRoot].map((directory) => mkdir(directory)));
    const output = path.join(runnerTemp, "output");
    const scratch = path.join(runnerTemp, "scratch");
    const guard = await validateCampaignPaths({
      runnerTemp, output, scratch, cacheRoot, repositories: [sceneworks, inference],
    });
    await mkdir(output);
    await mkdir(scratch);
    const checked = profile();
    const artifactExpectedFiles = Object.fromEntries(Object.keys(checked.artifacts).map(
      (id) => [id, id === "flux1-schnell-q8"
        ? ["model_index.json", "transformer/model.safetensors"]
        : ["model_index.json"]],
    ));
    if (missingId && missingId !== "flux1-schnell-q8") {
      artifactExpectedFiles[missingId] = ["unreviewed-missing.safetensors"];
    }
    const events = [];
    const runtimeCells = [];
    let verificationCalls = 0;
    const operations = {
      auditArtifact: async ({ id, artifact }) => {
        events.push(`audit:${id}`);
        const missingFiles = id === missingId
          ? [id === "flux1-schnell-q8"
            ? "q8/transformer/model.safetensors"
            : `${artifact.subdirectory}/unreviewed-missing.safetensors`]
          : [];
        const missingSelected = new Set(missingFiles.map((file) => (
          file.startsWith(`${artifact.subdirectory}/`)
            ? file.slice(artifact.subdirectory.length + 1) : file
        )));
        return {
          id, ...artifact, complete: missingFiles.length === 0, missingFiles,
          reusedFiles: artifactExpectedFiles[id].filter((file) => !missingSelected.has(file)).map(
            (file) => ({ path: file, bytes: 100, sha256: "a".repeat(64) }),
          ),
        };
      },
      stageArtifact: async ({ id, artifact, stagingRoot, allowReviewedDownload }) => {
        events.push(`stage:${id}`);
        if (allowReviewedDownload && !downloadSucceeds) {
          throw new Error("fixture exact missing-file transfer failed");
        }
        return {
          id, ...artifact,
          reusedFiles: artifactExpectedFiles[id].filter((file) => !(
            allowReviewedDownload && file === "transformer/model.safetensors"
          )).map((file, index) => ({
            path: file,
            bytes: 100,
            sha256: mutateSourceAtStage
              && events.filter((event) => event.startsWith("stage:")).length === 1
              && index === 0 ? "c".repeat(64) : "a".repeat(64),
          })),
          downloadedFiles: allowReviewedDownload ? [{
            path: "transformer/model.safetensors",
            bytes: 200,
            sha256: "b".repeat(64),
            lfsSha256: "b".repeat(64),
            commitSha: "bba3ae01dfd94089f173c05edd4e1a4c551f2599",
          }] : [],
          selectedRoot: path.join(stagingRoot, id, artifact.subdirectory),
        };
      },
      provisionArtifact: async ({ id, artifact, cacheRoot: stagedRoot }) => {
        events.push(`offline:${id}`);
        const selectedRoot = path.join(stagedRoot, id, artifact.subdirectory);
        return {
          id, ...artifact, selectedRoot,
          matchedFiles: artifactExpectedFiles[id],
          selectedFiles: artifactExpectedFiles[id],
          inventory: {
            root: selectedRoot,
            files: artifactExpectedFiles[id].length,
            bytes: 7,
            sha256: "a".repeat(64),
          },
        };
      },
      installSidecarObstructions: async (artifacts) => (
        omitObstructions ? [] : fixtureSidecarObstructions(artifacts)
      ),
      directoryInventory: async (root) => ({
        root, files: 0, bytes: 0, sha256: createHash("sha256").digest("hex"),
      }),
      executeCell: async ({ cellDir, logFile }) => {
        await writeFile(logFile, "fixture runtime\n", "utf8");
        const runtimeCell = JSON.parse(await readFile(path.join(cellDir, "cell.json"), "utf8"));
        runtimeCells.push(runtimeCell);
        await writeFile(path.join(cellDir, "runtime-result.json"), `${JSON.stringify({
          requestedTier: runtimeCell.requestedTier,
          resolvedTier: runtimeCell.requestedTier,
          denseFallback: false,
          loadSpecQuantBits: expectedLoadSpecQuantBits(runtimeCell),
        })}\n`, "utf8");
        await writeFile(path.join(cellDir, "output.bin"), "fixture", "utf8");
      },
      verifyArtifactUnchanged: async () => {
        verificationCalls += 1;
        if (mutateCache && verificationCalls === 2) {
          throw new Error("fixture cache bytes changed");
        }
      },
      sample: () => {},
    };
    const execution = {
      runId: "continuation", runAttempt: "1", headSha: "3".repeat(40), headRef: "refs/heads/continuation",
      workflow: "fixture", runnerName: "fixture-runner", runnerOs: "Windows", runnerArch: "X64",
    };
    const importedPrefix = {
      lineage: { kind: "fixture-prefix" },
      outcomes: checked.cells.slice(0, 7).map((cell, index) => ({
        id: cell.id, status: "passed", receipt: `_imported-prefix/${index + 1}/receipt.json`,
        error: null, emergencyReceiptError: null, source: "imported",
      })),
    };
    try {
      const result = await runCampaign({}, {
        prepared: {
          profile: checked,
          repositories: {
            sceneworks: { sha: execution.headSha, clean: true },
            inference: { sha: "4".repeat(40), clean: true },
          },
          execution,
          guard,
          output,
          scratch,
          startupErrors: [],
          gpuIdentity: [{
            timestamp: "2026/08/22 12:00:00", index: 0, name: "fixture GPU", uuid: "GPU-fixture",
            pciBusId: "00000000:01:00.0", computeCapability: "12.0", driverVersion: "999.1",
            memoryTotalMiB: 49140, memoryUsedMiB: 10, memoryFreeMiB: 49130, raw: "fixture raw",
          }],
          systemMemory: { totalBytes: 128 * 1024 ** 3, availableBytesAtStart: 100 * 1024 ** 3 },
          sceneworksRoot: sceneworks,
          python: "python",
          artifactExpectedFiles,
          downloadEvidenceSha256: "d".repeat(64),
        },
        prefixSelection: {},
        importedPrefix,
        operations,
        fault: preflightFault ? async (stage) => {
          if (stage === preflightFault) throw new Error(`injected ${stage}`);
        } : undefined,
        suppressVerdict: true,
      });
      const cacheEvidence = JSON.parse(await readFile(path.join(output, "cache-preflight.json"), "utf8"));
      return {
        result, events, runtimeCells, checked, cacheEvidence,
        cacheValidation: {
          remainingArtifactIds: [...new Set(checked.cells.slice(7).flatMap(
            (cell) => cell.artifactIds,
          ))],
          artifactExpectedFiles,
          downloadEvidenceSha256: "d".repeat(64),
          guard,
          stagingRoot: path.join(scratch, "authority-stage"),
          derivedSidecarRoot: path.join(scratch, "derived-candle-device-cache"),
          profile: checked,
        },
      };
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  }

  const failed = await scenario({ missingId: "flux1-schnell-q8" });
  const remainingIds = [...new Set(failed.checked.cells.slice(7).flatMap((cell) => cell.artifactIds))];
  assert.deepEqual(failed.events.filter((event) => event.startsWith("audit:")), remainingIds.map((id) => `audit:${id}`));
  assert.equal(failed.events.some((event) => event.startsWith("offline:")), false);
  assert.equal(failed.runtimeCells.length, 0);
  assert.equal(failed.result.summary.passed, 7);
  assert.equal(failed.result.summary.failed, 12);
  assert.match(
    failed.result.summary.campaignErrors.join("\n"),
    /copy-once campaign staging failed[\s\S]*exact missing-file transfer failed[\s\S]*no continuation GPU cell started/,
  );

  const unapproved = await scenario({ missingId: "ltx23-q8" });
  assert.equal(unapproved.events.some((event) => event.startsWith("stage:")), false);
  assert.equal(unapproved.runtimeCells.length, 0);
  assert.match(
    unapproved.result.summary.campaignErrors.join("\n"),
    /source cache census found an unapproved missing-file set.*no continuation GPU cell started/,
  );

  const sourceChanged = await scenario({ mutateSourceAtStage: true });
  assert.equal(sourceChanged.runtimeCells.length, 0);
  assert.equal(sourceChanged.events.some((event) => event.startsWith("offline:")), false);
  assert.match(
    sourceChanged.result.summary.campaignErrors.join("\n"),
    /trusted source cache changed after frozen census[\s\S]*no continuation GPU cell started/,
  );

  const passed = await scenario();
  assert.deepEqual(passed.events, [
    ...remainingIds.map((id) => `audit:${id}`),
    ...remainingIds.map((id) => `stage:${id}`),
    ...remainingIds.map((id) => `offline:${id}`),
  ]);
  assert.equal(passed.runtimeCells.length, 12, passed.result.summary.campaignErrors.join("\n"));
  const scailRoots = passed.runtimeCells
    .filter((cell) => cell.artifacts.some((artifact) => artifact.id === "scail2-q4"))
    .map((cell) => cell.artifacts.find((artifact) => artifact.id === "scail2-q4").root);
  assert.equal(new Set(scailRoots).size, 1, "repeated SCAIL q4 authority must be reused");
  const controlRoots = passed.runtimeCells
    .filter((cell) => cell.kind === "sdxlOpenPose")
    .map((cell) => cell.artifacts.find((artifact) => artifact.id === "sdxl-openpose").root);
  assert.equal(controlRoots.length, 5);
  assert.equal(new Set(controlRoots).size, 1, "five SDXL cells must reuse one cached ControlNet");

  const filled = await scenario({ missingId: "flux1-schnell-q8", downloadSucceeds: true });
  assert.deepEqual(filled.events, [
    ...remainingIds.map((id) => `audit:${id}`),
    ...remainingIds.map((id) => `stage:${id}`),
    ...remainingIds.map((id) => `offline:${id}`),
  ]);
  assert.equal(filled.runtimeCells.length, 12);
  assert.deepEqual(filled.cacheEvidence.downloadedFiles, [{
    artifactId: "flux1-schnell-q8",
    path: "transformer/model.safetensors",
    bytes: 200,
    sha256: "b".repeat(64),
    lfsSha256: "b".repeat(64),
    commitSha: "bba3ae01dfd94089f173c05edd4e1a4c551f2599",
  }]);
  assert.deepEqual(filled.cacheEvidence.frozenMissingFiles, [{
    artifactId: "flux1-schnell-q8",
    repository: "SceneWorks/flux1-schnell-mlx",
    revision: "bba3ae01dfd94089f173c05edd4e1a4c551f2599",
    file: "q8/transformer/model.safetensors",
  }]);
  assert.equal(filled.cacheEvidence.offlineBeforeCells, true);
  assert.equal(filled.cacheEvidence.phases.finalOffline.length, remainingIds.length);
  assert.equal(filled.cacheEvidence.phases.sourceCensus.length, remainingIds.length);
  assert.equal(filled.cacheEvidence.phases.staging.length, remainingIds.length);
  assert.doesNotThrow(() => validateCachePreflightEvidence(
    filled.cacheEvidence,
    filled.cacheValidation,
  ));
  const extraField = structuredClone(filled.cacheEvidence);
  extraField.unreviewed = true;
  assert.throws(
    () => validateCachePreflightEvidence(extraField, filled.cacheValidation),
    /additional properties/,
  );
  const censusDrift = structuredClone(filled.cacheEvidence);
  censusDrift.phases.finalOffline[0].expectedFiles = ["other.bin"];
  assert.throws(
    () => validateCachePreflightEvidence(censusDrift, filled.cacheValidation),
    /authority census drifted/,
  );
  const obstructionMissing = structuredClone(filled.cacheEvidence);
  obstructionMissing.sidecarObstructions.pop();
  assert.throws(
    () => validateCachePreflightEvidence(obstructionMissing, filled.cacheValidation),
    /did not obstruct every staged component root/,
  );
  const downloadDrift = structuredClone(filled.cacheEvidence);
  downloadDrift.downloadedFiles[0].commitSha = "f".repeat(40);
  assert.throws(
    () => validateCachePreflightEvidence(downloadDrift, filled.cacheValidation),
    /partition drifted|unreviewed model download/,
  );

  const mutated = await scenario({ mutateCache: true });
  assert.equal(mutated.runtimeCells.length, 1, "the first cell runs before its after-inventory detects mutation");
  assert.match(
    mutated.result.summary.campaignErrors.join("\n"),
    /shared cache mutated during cell flux1-dev-q8/,
  );
  assert.equal(mutated.result.summary.failed, 12);

  for (const stage of [
    "preflightMkdir", "preflightSchema", "preflightWrite", "preflightStat", "preflightHash",
  ]) {
    const faulted = await scenario({ preflightFault: stage });
    assert.equal(faulted.runtimeCells.length, 0, stage);
    assert.equal(faulted.result.summary.passed, 7, stage);
    assert.equal(faulted.result.summary.failed, 12, stage);
    assert.match(faulted.result.summary.campaignErrors.join("\n"), new RegExp(stage));
    assert.ok(faulted.result.summary.receipts.slice(7).every((outcome) => outcome.receipt));
    assert.doesNotThrow(() => validateDocumentWithSchema(
      CACHE_PREFLIGHT_SCHEMA_PATH,
      faulted.cacheEvidence,
    ));
  }
  const semanticFault = await scenario({ omitObstructions: true });
  assert.equal(semanticFault.runtimeCells.length, 0);
  assert.equal(semanticFault.result.summary.passed, 7);
  assert.equal(semanticFault.result.summary.failed, 12);
  assert.match(
    semanticFault.result.summary.campaignErrors.join("\n"),
    /did not obstruct every staged component root/,
  );
  assert.doesNotThrow(() => validateDocumentWithSchema(
    CACHE_PREFLIGHT_SCHEMA_PATH,
    semanticFault.cacheEvidence,
  ));
});

test("schemas, clean Node dependencies, and workflow preserve the opt-in single-job terminal contract", async () => {
  const [profileSchema, receiptSchema, cacheSchema, workflow, checks, packageJson, packageLock] = await Promise.all([
    readFile("config/terminal-evidence/epic-20738-profile.schema.json", "utf8").then(JSON.parse),
    readFile("config/terminal-evidence/epic-20738-receipt.schema.json", "utf8").then(JSON.parse),
    readFile(CACHE_PREFLIGHT_SCHEMA_PATH, "utf8").then(JSON.parse),
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
    const prefix = terminalStep(document, "terminal_prefix");
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
      prefix,
      ["terminal_node", "terminal_npm", "terminal_tests", "terminal_validation"],
      "prefix discovery",
    );
    requiresOutcomes(
      python,
      ["terminal_node", "terminal_npm", "terminal_tests", "terminal_validation", "terminal_prefix"],
      "Python setup",
    );
    requiresOutcomes(
      inference,
      ["terminal_node", "terminal_npm", "terminal_tests", "terminal_validation", "terminal_prefix"],
      "inference checkout",
    );
    requiresOutcomes(
      campaign,
      ["terminal_node", "terminal_npm", "terminal_tests", "terminal_validation", "terminal_prefix", "terminal_python", "terminal_inference"],
      "terminal campaign",
    );
    for (const [name, id] of [
      ["NODE_OUTCOME", "terminal_node"],
      ["NPM_OUTCOME", "terminal_npm"],
      ["TESTS_OUTCOME", "terminal_tests"],
      ["VALIDATION_OUTCOME", "terminal_validation"],
      ["PREFIX_OUTCOME", "terminal_prefix"],
      ["PYTHON_OUTCOME", "terminal_python"],
      ["INFERENCE_OUTCOME", "terminal_inference"],
      ["CAMPAIGN_OUTCOME", "terminal_campaign"],
    ]) {
      assert.match(verdict, new RegExp(`${name}: \\$\\{\\{ steps\\.${id}\\.outcome \\}\\}`));
    }
    assert.match(verdict, /Where-Object \{ \$_ -ne 'success' \}/);
  };
  assertTerminalWorkflowGates(normalizedWorkflow);
  for (const id of ["terminal_node", "terminal_npm", "terminal_tests", "terminal_prefix"]) {
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
    ["PREFIX_OUTCOME", "terminal_prefix"],
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
  assert.deepEqual(receiptSchema.properties.cell.properties.loadSpecQuantBits, { enum: [null, 4, 8] });
  assert.equal(receiptSchema.$defs.inventory.additionalProperties, false);
  assert.equal(receiptSchema.$defs.gpuSample.additionalProperties, false);
  assert.equal(receiptSchema.properties.profile.const, PROFILE_NAME);
  assert.equal(cacheSchema.additionalProperties, false);
  assert.equal(cacheSchema.properties.profile.const, PROFILE_NAME);
  assert.match(workflow, /run_epic_20738_terminal_cuda:[\s\S]*?default: false/);
  assert.doesNotMatch(workflow, /^      sceneworks_revision:/m);
  assert.equal((workflow.match(/SCENEWORKS_REVISION: \$\{\{ github\.sha \}\}/g) ?? []).length, 2);
  assert.doesNotMatch(workflow, /inputs\.provision_krea_snapshot/);
  assert.match(workflow, /PROVISION_SNAPSHOT: \$\{\{ inputs\.provision_snapshot \}\}/);
  assert.match(workflow, /windows-candle-\$\{\{[^\n]*epic-20738-terminal/);
  assert.equal((workflow.match(/^jobs:\s*$/gm) ?? []).length, 1);
  assert.equal((workflow.match(/^  candle-worker:\s*$/gm) ?? []).length, 1);
  assert.doesNotMatch(workflow, /^  [a-zA-Z0-9_-]+:\s*\n\s+strategy:\s*\n\s+matrix:/m);
  assert.equal((workflow.match(/epic-20738-terminal-cuda-harness\.mjs run/g) ?? []).length, 1);
  assert.match(workflow, /id: terminal_campaign[\s\S]*?continue-on-error: true/);
  assert.doesNotMatch(workflow, /jsonschema|harness\.mjs check --python/);
  assert.match(
    workflow,
    /npm ci --ignore-scripts[\s\S]*?node --test scripts\/epic-20738-terminal-cuda-harness\.test\.mjs scripts\/hash-artifact-inventory\.test\.mjs[\s\S]*?python -m unittest scripts\/provision_epic_20738_terminal_artifact_test\.py[\s\S]*?harness\.mjs check/,
  );
  const terminalBlock = workflow.slice(workflow.indexOf("# SC-20945 terminal epic evidence."));
  assert.equal((terminalBlock.match(/SCENEWORKS_TERMINAL_CACHE_ROOT: 'E:\\huggingface\\hub'/g) ?? []).length, 2);
  assert.match(terminalBlock, /--cache-root "%SCENEWORKS_TERMINAL_CACHE_ROOT%"/);
  assert.match(terminalBlock, /--prefix-candidates "%SCENEWORKS_TERMINAL_PREFIX_CANDIDATES%"/);
  assert.match(terminalBlock, /HF_HUB_OFFLINE: "1"/);
  assert.match(terminalBlock, /TRANSFORMERS_OFFLINE: "1"/);
  assert.match(terminalBlock, /gh api "repos\/\$env:GITHUB_REPOSITORY\/actions\/artifacts\?per_page=100&page=\$page"/);
  assert.match(terminalBlock, /gh run download \$runId --name \$artifact\.name --dir \$evidence/);
  assert.match(terminalBlock, /artifactDigest = \[string\]\$artifact\.digest/);
  assert.match(terminalBlock, /8886a9e69f26beec05688c81b414859bd102f6d0/);
  assert.doesNotMatch(terminalBlock, /snapshot_download/);
  assert.match(terminalBlock, /python -m venv \$venv/);
  assert.match(terminalBlock, /pip install[^\n]*'huggingface_hub==0\.36\.0'/);
  assert.match(terminalBlock, /Scripts\\python\.exe/);
  assert.match(terminalBlock, /SCENEWORKS_TERMINAL_PYTHON=\$python/);
  assert.match(terminalBlock, /from huggingface_hub import __version__/);
  assert.match(terminalBlock, /__version__ == '0\.36\.0'/);
  assert.match(terminalBlock, /q8\/transformer\/model\.safetensors/);
  const controller = await readFile("scripts/epic-20738-terminal-cuda-harness.mjs", "utf8");
  assert.match(controller, /HF_HUB_OFFLINE: allowReviewedDownload \? "0" : "1"/);
  assert.match(controller, /HF_HUB_OFFLINE: "1"[\s\S]*TRANSFORMERS_OFFLINE: "1"/);
  for (const variable of [
    "TEMP", "TMP", "TMPDIR", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TORCH_HOME",
  ]) {
    assert.match(controller, new RegExp(`${variable}: writable\\.`));
  }
  assert.doesNotMatch(
    workflow,
    /^      (?:resume|start_cell|terminal_cache_root|terminal_prefix)[^:]*:/m,
    "continuation/cache authority must not be a caller-controlled dispatch input",
  );
  assert.equal(packageJson.devDependencies.ajv, "8.20.0");
  assert.equal(packageJson.devDependencies["ajv-formats"], "3.0.1");
  assert.equal(packageLock.packages[""].devDependencies.ajv, "8.20.0");
  assert.match(checks, /parity-scaffold:[\s\S]*?npm ci --ignore-scripts[\s\S]*?npm run check/);
  assert.match(workflow, /name: Upload every epic-20738 terminal receipt[\s\S]*?if: \$\{\{ always\(\)/);
});
