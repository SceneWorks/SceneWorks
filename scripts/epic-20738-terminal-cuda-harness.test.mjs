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
  assertExactDownloadedFilePartition,
  authorityLifetimePlan,
  cellSemanticsSha256,
  directoryInventory,
  expectedB646DerivedNamespace,
  expectedArtifactFilesFromEvidence,
  expectedCurrentArtifactFilesFromEvidenceBytes,
  expectedLoadSpecQuantBits,
  estimateJitDiskPlan,
  inferencePins,
  hashedFiles,
  installSidecarObstructions,
  inspectDerivedSidecarRoot,
  importPrefixEvidence,
  loadProfile,
  parseNvidiaSmi,
  receiptSkeleton,
  runCampaign,
  safeRemoveTree,
  importSparseRecovery,
  importRecoveryContinuation,
  selectImportedPrefix,
  selectRecoveryContinuation,
  selectSparseRecovery,
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
const recoveryArtifactProfile = () => {
  const legacy = profile();
  legacy.cells[13].request.steps = 4;
  for (const [id, revision] of [
    ["illustrious-v1-q4", "c5a92a902dd4e6ee99c2a57981ecf66209905dd1"],
    ["illustrious-v2-q4", "7c5c8b2bb75a8f38a7365e70bdf84d38d6204473"],
  ]) {
    legacy.artifacts[id].revision = revision;
    delete legacy.artifacts[id].quantizationMarker;
  }
  return legacy;
};

function scail2ReferenceCounterfactuals() {
  return Array.from({ length: 6 }, (_, index) => {
    const reference = index + 1;
    return {
      reference,
      meanAbsDelta: reference / 100,
      firstFrameMeanAbsDelta: reference / 200,
      lastFrameMeanAbsDelta: reference / 300,
      witnesses: {
        omittedReference: `input-reference-${reference}.png`,
        firstFrame: `counterfactual-reference-${reference}-first.png`,
        lastFrame: `counterfactual-reference-${reference}-last.png`,
      },
    };
  });
}

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
  assert.equal(cellSemanticsSha256(checked.cells), "2fcd20e4909f0bd0ba6c78c6a85247267c354735f77f4ed4912d47941a8512c1");
  assert.equal(checked.cells.find((cell) => cell.id === "ltx-2-3-q8").request.steps, 8);
  assert.doesNotMatch(JSON.stringify(checked).toLowerCase(), /anima|sana|vace|flux2|true[_-]?v2|eros/);
  assert.deepEqual(
    ["ltx23-q8", "ltx23-gemma"].map((id) => checked.artifacts[id].revision),
    [
      "254989c3ca7ee691187647f350b112c0c448789d",
      "254989c3ca7ee691187647f350b112c0c448789d",
    ],
    "terminal LTX must truthfully use the exact cached production-approved parent",
  );
  assert.deepEqual(
    ["illustrious-v1-q4", "illustrious-v2-q4"].map((id) => ({
      revision: checked.artifacts[id].revision,
      marker: checked.artifacts[id].quantizationMarker,
    })),
    [
      {
        revision: "778c3f02b7703b0c2755d0c0447592897193c6b5",
        marker: { bits: 4, groupSize: 64, components: ["text_encoder", "text_encoder_2", "unet"] },
      },
      {
        revision: "672e9851ede4dc856fa945649b6691975c9d74a3",
        marker: { bits: 4, groupSize: 64, components: ["text_encoder", "text_encoder_2", "unet"] },
      },
    ],
  );

  const manifest = JSON.parse(stripJsoncComments(await readFile(
    "config/manifests/builtin.models.jsonc", "utf8",
  )));
  assert.doesNotThrow(() => validateManifestAuthorities(checked, manifest));
  for (const [modelId, currentRevision, legacyRevision, exactSizes] of [
    [
      "illustrious_xl_v1",
      "778c3f02b7703b0c2755d0c0447592897193c6b5",
      "c5a92a902dd4e6ee99c2a57981ecf66209905dd1",
      [3_911_462_020, 5_385_646_612, 7_108_498_305],
    ],
    [
      "illustrious_xl_v2",
      "672e9851ede4dc856fa945649b6691975c9d74a3",
      "7c5c8b2bb75a8f38a7365e70bdf84d38d6204473",
      [3_911_461_696, 5_385_646_332, 7_108_498_177],
    ],
  ]) {
    const model = manifest.models.find(({ id }) => id === modelId);
    const currentDownloads = model.downloads.slice(0, 3);
    assert.deepEqual(currentDownloads.map(({ revision }) => revision),
      [currentRevision, currentRevision, currentRevision]);
    assert.deepEqual(currentDownloads.map(({ estimatedSizeBytes }) => estimatedSizeBytes), exactSizes);
    assert.deepEqual(currentDownloads.map(({ footprint }) => footprint.diskSizeBytes), exactSizes);
    assert.equal(JSON.stringify(model).split(`"${currentRevision}"`).length - 1, 3,
      `${modelId} new authority may appear only in the three current download rows`);
    assert.match(JSON.stringify(model), new RegExp(legacyRevision),
      `${modelId} historical evidence must retain its producing revision`);
  }
});

test("current and legacy Illustrious download selectors fail closed independently", async () => {
  const checked = profile();
  const evidenceBytes = await readFile("config/download-pattern-evidence.json", "utf8");
  const current = expectedCurrentArtifactFilesFromEvidenceBytes(checked, evidenceBytes);
  assert.equal(current.downloadEvidenceSha256,
    "1fa06ef39a0e2c321a4fa15fa1128c0157ba8cf22fd868ac54c6cefaec13a5ee");
  assert.equal(current.artifactExpectedFiles["illustrious-v1-q4"].length, 19);
  assert.equal(current.artifactExpectedFiles["illustrious-v2-q4"].length, 19);

  const digestMutation = evidenceBytes.replace('"gated": false', '"gated": true');
  assert.throws(
    () => expectedCurrentArtifactFilesFromEvidenceBytes(checked, digestMutation),
    /current download-pattern evidence digest drifted/,
  );

  const evidence = JSON.parse(evidenceBytes);
  const legacy = structuredClone(checked);
  for (const [id, revision] of [
    ["illustrious-v1-q4", "c5a92a902dd4e6ee99c2a57981ecf66209905dd1"],
    ["illustrious-v2-q4", "7c5c8b2bb75a8f38a7365e70bdf84d38d6204473"],
  ]) {
    legacy.artifacts[id].revision = revision;
    delete legacy.artifacts[id].quantizationMarker;
  }
  const legacyExpected = expectedArtifactFilesFromEvidence(legacy, evidence);
  assert.deepEqual(legacyExpected["illustrious-v1-q4"], current.artifactExpectedFiles["illustrious-v1-q4"]);
  assert.deepEqual(legacyExpected["illustrious-v2-q4"], current.artifactExpectedFiles["illustrious-v2-q4"]);

  const missingLegacy = structuredClone(evidence);
  missingLegacy.repos = missingLegacy.repos.filter(({ revision }) => (
    revision !== "c5a92a902dd4e6ee99c2a57981ecf66209905dd1"
  ));
  assert.throws(
    () => expectedArtifactFilesFromEvidence(legacy, missingLegacy),
    /missing exact authority SceneWorks\/illustrious-xl-v1-mlx@c5a92a9/,
  );
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

test("downloaded-file partition compares exact semantics but ignores object insertion order", () => {
  const expected = [{
    path: "transformer/model.safetensors", bytes: 200, sha256: "a".repeat(64),
    lfsSha256: "b".repeat(64), commitSha: "bba3ae01dfd94089f173c05edd4e1a4c551f2599",
  }];
  const reordered = [{
    commitSha: expected[0].commitSha, lfsSha256: expected[0].lfsSha256,
    sha256: expected[0].sha256, bytes: expected[0].bytes, path: expected[0].path,
  }];
  assert.doesNotThrow(() => assertExactDownloadedFilePartition(reordered, expected, "fixture"));
  for (const field of ["path", "bytes", "sha256", "lfsSha256", "commitSha"]) {
    const drifted = structuredClone(expected);
    drifted[0][field] = field === "path" ? "other.safetensors"
      : field === "bytes" ? 201 : "f".repeat(field === "commitSha" ? 40 : 64);
    assert.throws(() => assertExactDownloadedFilePartition(drifted, expected, "fixture"), /partition/);
  }
  for (const malformed of [null, [], [{ ...expected[0], extra: true }], [{ ...expected[0], path: "../escape" }]]) {
    assert.throws(() => assertExactDownloadedFilePartition(malformed, expected, "fixture"), /array|fields|malformed|partition/);
  }
  assert.throws(
    () => assertExactDownloadedFilePartition([...expected, structuredClone(expected[0])], expected, "fixture"),
    /ambiguous/,
  );
});

test("JIT disk estimator uses followed target bytes, exact lifetimes, and the 99GB floor", () => {
  const file = (key, bytes, persistent = false) => ({ key, bytes, persistent });
  const row = (artifactId, firstOrdinal, lastOrdinal, files) => ({
    artifactId,
    role: artifactId.includes("support") ? "support" : "primary",
    firstOrdinal,
    lastOrdinal,
    sourceBytes: files.reduce((sum, entry) => sum + entry.bytes, 0),
    expectedFiles: files.length,
    physicalFiles: files,
  });
  const persistent = file("schnell-q8/missing", 12_637_521_668, true);
  const lifetimes = [
    row("flux1-dev-q8", 8, 8, [file("dev-q8", 18_010_071_290)]),
    row("flux1-schnell-q4", 9, 9, [file("schnell-q4", 9_611_551_284)]),
    row("flux1-schnell-q8", 10, 10, [file("schnell-q8/cache", 5_361_654_035), persistent]),
    row("scail2-q4", 11, 12, [file("scail-q4", 23_993_093_306)]),
    row("scail2-q8", 13, 13, [file("scail-q8", 32_067_131_269)]),
    row("ltx23-q8", 14, 14, [file("ltx-q8", 29_728_720_716)]),
    row("ltx23-gemma", 14, 14, [file("ltx-gemma", 26_427_894_918)]),
    row("sdxl-base-q4", 15, 15, [file("sdxl-base-q4", 2_703_202_068)]),
    row("realvisxl-q4", 16, 16, [file("realvisxl-q4", 3_911_656_467)]),
    row("realvisxl-lightning-q4", 17, 17, [file("realvisxl-lightning-q4", 3_911_657_599)]),
    row("illustrious-v1-q4", 18, 18, [file("illustrious-v1-q4", 3_911_656_986)]),
    row("illustrious-v2-q4", 19, 19, [file("illustrious-v2-q4", 3_911_656_662)]),
    row("sdxl-openpose", 15, 19, [file("sdxl-openpose", 2_502_139_104)]),
    row("sdxl-tokenizer-l", 15, 19, [file("sdxl-tokenizer-l", 2_224_003)]),
    row("sdxl-tokenizer-bigg", 15, 19, [file("sdxl-tokenizer-bigg", 2_224_041)]),
    row("sdxl-vae-fix", 15, 19, [file("sdxl-vae-fix", 334_643_238)]),
  ];
  const floor = 99_106_288_594;
  const admitted = estimateJitDiskPlan(lifetimes, floor, persistent.bytes);
  assert.equal(admitted.logicalSourceBytes, 179_028_698_654);
  assert.equal(admitted.allAtOnceSourceBytes, 179_028_698_654);
  assert.equal(
    admitted.allAtOnceSourceBytes - admitted.legacyNaiveLinkLengthSourceBytes,
    5_361_654_425,
    "link-length accounting must never undercount followed Schnell q8 blob bytes",
  );
  assert.equal(admitted.cells.find((entry) => entry.ordinal === 8).modelAndSidecarBytes, 43_229_554_686);
  assert.equal(admitted.cells.find((entry) => entry.ordinal === 9).modelAndSidecarBytes, 29_653_559_608);
  assert.equal(admitted.cells.find((entry) => entry.ordinal === 10).modelAndSidecarBytes, 30_581_137_431);
  assert.equal(admitted.cells.find((entry) => entry.ordinal === 13).modelAndSidecarBytes, 32_067_131_269);
  assert.equal(admitted.cells.find((entry) => entry.ordinal === 14).stagedBytes, 56_156_615_634);
  assert.equal(admitted.peakModelAndSidecarBytes, 56_156_615_634);
  assert.equal(admitted.peakRequiredAdditionalBytes, floor);
  assert.equal(admitted.admitted, true);
  assert.equal(estimateJitDiskPlan(lifetimes, floor - 1, persistent.bytes).admitted, false);
  assert.ok(admitted.allAtOnceRequiredBytes > floor, "all-at-once staging stays rejected");

  const completeCache = structuredClone(lifetimes);
  const completeQ8 = completeCache.find((entry) => entry.artifactId === "flux1-schnell-q8");
  completeQ8.physicalFiles = completeQ8.physicalFiles.map((entry) => ({
    ...entry,
    persistent: false,
  }));
  const completePlan = estimateJitDiskPlan(completeCache, floor, 0);
  assert.equal(completePlan.peakModelAndSidecarBytes, admitted.peakModelAndSidecarBytes);
  assert.equal(
    completePlan.cells.find((entry) => entry.ordinal === 8).modelAndSidecarBytes,
    admitted.cells.find((entry) => entry.ordinal === 8).modelAndSidecarBytes - persistent.bytes,
  );
  assert.equal(completePlan.peakRequiredAdditionalBytes, floor);
  assert.equal(completePlan.admitted, true);
  assert.equal(estimateJitDiskPlan(completeCache, floor - 1, 0).admitted, false);
});

test("sparse execution lifetimes cover only 14, 18, and 19 while retaining shared SDXL helpers", () => {
  const checked = profile();
  const sourceBytes = new Map([
    ["ltx23-q8", 29_728_720_716],
    ["ltx23-gemma", 26_427_894_918],
    ["illustrious-v1-q4", 3_911_656_986],
    ["illustrious-v2-q4", 3_911_656_662],
    ["sdxl-openpose", 2_502_139_104],
    ["sdxl-tokenizer-l", 2_224_003],
    ["sdxl-tokenizer-bigg", 2_224_041],
    ["sdxl-vae-fix", 334_643_238],
  ]);
  const audits = new Map([...sourceBytes].map(([id, bytes]) => [id, {
    reusedFiles: [{ path: "fixture.bin", bytes, sha256: "a".repeat(64) }],
    downloadedFiles: [],
    expectedFiles: ["fixture.bin"],
  }]));
  const lifetimes = authorityLifetimePlan(checked, [14, 18, 19], audits);
  assert.deepEqual(lifetimes.map(({ artifactId }) => artifactId), [
    "ltx23-q8", "ltx23-gemma", "illustrious-v1-q4", "sdxl-openpose",
    "sdxl-tokenizer-l", "sdxl-tokenizer-bigg", "sdxl-vae-fix", "illustrious-v2-q4",
  ]);
  for (const id of ["sdxl-openpose", "sdxl-tokenizer-l", "sdxl-tokenizer-bigg", "sdxl-vae-fix"]) {
    const lifetime = lifetimes.find((row) => row.artifactId === id);
    assert.equal(lifetime.firstOrdinal, 18);
    assert.equal(lifetime.lastOrdinal, 19);
  }
  const plan = estimateJitDiskPlan(lifetimes, 99_106_288_594, 0);
  assert.deepEqual(plan.cells.map(({ ordinal }) => ordinal), [14, 18, 19]);
  assert.equal(plan.logicalSourceBytes, 66_821_159_668);
  assert.equal(plan.peakModelAndSidecarBytes, 56_156_615_634);
  assert.equal(plan.peakRequiredAdditionalBytes, 99_106_288_594);
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

  for (const count of [4, 7, 9]) {
    const ltxSteps = profile();
    ltxSteps.cells.find((cell) => cell.id === "ltx-2-3-q8").request.steps = count;
    assert.throws(
      () => validateProfile(ltxSteps),
      /semantic tuples drifted/,
      `LTX fixed schedule mutation ${count} must fail closed`,
    );
  }

  const unreviewedKnob = profile();
  unreviewedKnob.cells[0].request.unreviewedKnob = true;
  assert.throws(() => validateProfile(unreviewedKnob), /semantic tuples drifted/);

  const blocked = profile();
  blocked.artifacts["scail2-q4"].repository = "SceneWorks/eros-model";
  assert.throws(() => validateProfile(blocked), /blocked surface eros/);

  const illustriousRevision = profile();
  illustriousRevision.artifacts["illustrious-v1-q4"].revision =
    "c5a92a902dd4e6ee99c2a57981ecf66209905dd1";
  assert.throws(() => validateProfile(illustriousRevision), /exact current revision and q4\/group-64/);

  for (const mutation of [
    (marker) => { marker.bits = 8; },
    (marker) => { marker.groupSize = 32; },
    (marker) => { marker.components.reverse(); },
  ]) {
    const illustriousMarker = profile();
    mutation(illustriousMarker.artifacts["illustrious-v2-q4"].quantizationMarker);
    assert.throws(
      () => validateProfile(illustriousMarker),
      /exact current revision and q4\/group-64/,
    );
  }

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
    /exact three promoted consumers/,
  );

  const extraControlConsumer = structuredClone(manifest);
  const extra = structuredClone(extraControlConsumer.models.find((model) => model.id === "sdxl")
    .downloads.find((download) => download.componentId === "controlnet_openpose"));
  extraControlConsumer.models.find((model) => model.id === "flux_dev").downloads.push(extra);
  assert.throws(
    () => validateManifestAuthorities(profile(), extraControlConsumer),
    /exact three promoted consumers/,
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
      requestMemoryStrategy: {
        strategy: cell.engineId.startsWith("flux1_") ? "default-resident" : "not-applicable",
        requestMemoryPresent: false,
        stageResidency: false,
        streamTransformerBlocks: false,
      },
      metrics: cell.capability === "multiReference" ? {
        kind: "scail2",
        referencePairs: 6,
        referenceCounterfactuals: scail2ReferenceCounterfactuals(),
      } : { referenceCounterfactuals: null },
    };
    assert.equal(validateRuntimeResult(valid, cell), valid, cell.id);
    if (cell.engineId.startsWith("flux1_")) {
      const streamedWithoutStaging = structuredClone(valid);
      streamedWithoutStaging.requestMemoryStrategy = {
        strategy: "bounded-transformer",
        requestMemoryPresent: true,
        stageResidency: false,
        streamTransformerBlocks: true,
      };
      assert.throws(
        () => validateRuntimeResult(streamedWithoutStaging, cell),
        /request memory strategy/,
        `${cell.id} must reject streamed memory without staged residency`,
      );
    }
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

test("SCAIL2 six-reference causal metrics reject missing, duplicate, nonfinite, zero, and copied evidence", () => {
  const checked = profile();
  const cell = checked.cells.find((candidate) => candidate.capability === "multiReference");
  const valid = {
    requestedTier: cell.requestedTier,
    resolvedTier: cell.requestedTier,
    denseFallback: false,
    loadSpecQuantBits: 4,
    requestMemoryStrategy: {
      strategy: "not-applicable", requestMemoryPresent: false,
      stageResidency: false, streamTransformerBlocks: false,
    },
    metrics: {
      kind: "scail2", referencePairs: 6,
      referenceCounterfactuals: scail2ReferenceCounterfactuals(),
    },
  };
  assert.doesNotThrow(() => validateRuntimeResult(valid, cell));
  const missing = structuredClone(valid);
  missing.metrics.referenceCounterfactuals.pop();
  assert.throws(() => validateRuntimeResult(missing, cell), /exactly six/);
  const duplicate = structuredClone(valid);
  duplicate.metrics.referenceCounterfactuals[5].reference = 5;
  assert.throws(() => validateRuntimeResult(duplicate, cell), /exactly ordered/);
  const nonfinite = structuredClone(valid);
  nonfinite.metrics.referenceCounterfactuals[0].meanAbsDelta = Infinity;
  assert.throws(() => validateRuntimeResult(nonfinite, cell), /nonfinite/);
  const zero = structuredClone(valid);
  zero.metrics.referenceCounterfactuals[2].meanAbsDelta = 0;
  assert.throws(() => validateRuntimeResult(zero, cell), /zero, nonfinite, or trivial/);
  const copied = structuredClone(valid);
  copied.metrics.referenceCounterfactuals[4].witnesses = structuredClone(
    copied.metrics.referenceCounterfactuals[0].witnesses,
  );
  assert.throws(() => validateRuntimeResult(copied, cell), /position-bound/);
});

function fixtureReceipt(status = "passed", cellIndex = 0, checked = profile()) {
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
  if (status === "passed" && cell.capability === "multiReference") {
    receipt.cell.referenceCounterfactuals = scail2ReferenceCounterfactuals();
  }
  const fluxArtifact = cell.artifactIds.find((artifactId) => artifactId.startsWith("flux1-"));
  receipt.authorityLifecycle = {
    ordinal: cellIndex + 1,
    cellId: cell.id,
    staged: fluxArtifact ? [{
      artifactId: fluxArtifact,
      stageRoot: artifacts.find((artifact) => artifact.id === fluxArtifact).selectedRoot,
      inventory: {
        root: artifacts.find((artifact) => artifact.id === fluxArtifact).selectedRoot,
        files: 3, bytes: 42, sha256: "4".repeat(64),
      },
      obstructionCount: 1,
      derivedNamespaces: [],
    }] : [],
    activeArtifactIds: [...cell.artifactIds],
    providerExecution: "completed",
    requestMemoryStrategy: fluxArtifact ? {
      strategy: "default-resident",
      requestMemoryPresent: false,
      stageResidency: false,
      streamTransformerBlocks: false,
    } : {
      strategy: "not-applicable",
      requestMemoryPresent: false,
      stageResidency: false,
      streamTransformerBlocks: false,
    },
    verifiedBefore: true,
    verifiedAfter: true,
    derivedAfter: cell.artifactIds.map((artifactId) => ({
      artifactId,
      derivedDisposition: artifactId === fluxArtifact
        ? "resident-empty" : "not-applicable",
      files: 0,
      bytes: 0,
      inventories: [],
    })),
    released: [],
    diskProbes: [...(fluxArtifact ? [{
      phase: `before-stage:${fluxArtifact}`,
      ordinal: cellIndex + 1,
      root: path.resolve("fixture-runner", "scratch"),
      freeBytes: 128 * 1024 ** 3,
      requiredFreeBytes: 99_106_288_594,
    }] : []), {
      phase: "before-execution",
      ordinal: cellIndex + 1,
      root: path.resolve("fixture-runner", "scratch"),
      freeBytes: 128 * 1024 ** 3,
      requiredFreeBytes: 99_106_288_594,
    }],
  };
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

    const scailMultiReference = fixtureReceipt("passed", 11);
    const scailMultiReferenceFile = await writeReceipt("scail-multi-reference.json", scailMultiReference);
    assert.doesNotThrow(() => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, scailMultiReferenceFile));
    delete scailMultiReference.cell.referenceCounterfactuals;
    const missingScailEvidenceFile = await writeReceipt("scail-multi-reference-missing.json", scailMultiReference);
    assert.throws(
      () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, missingScailEvidenceFile),
      /must have required property 'referenceCounterfactuals'/,
    );
    scailMultiReference.cell.referenceCounterfactuals = scail2ReferenceCounterfactuals();
    scailMultiReference.cell.referenceCounterfactuals[4].reference = 4;
    const duplicateScailEvidenceFile = await writeReceipt("scail-multi-reference-duplicate.json", scailMultiReference);
    assert.throws(
      () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, duplicateScailEvidenceFile),
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

    const missingProviderState = fixtureReceipt();
    delete missingProviderState.authorityLifecycle.providerExecution;
    const missingProviderStateFile = await writeReceipt(
      "missing-provider-state.json", missingProviderState,
    );
    assert.throws(
      () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, missingProviderStateFile),
      /must have required property 'providerExecution'/,
    );

    const missingRequestMemory = fixtureReceipt();
    delete missingRequestMemory.authorityLifecycle.requestMemoryStrategy;
    const missingRequestMemoryFile = await writeReceipt(
      "missing-request-memory.json", missingRequestMemory,
    );
    assert.throws(
      () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, missingRequestMemoryFile),
      /must have required property 'requestMemoryStrategy'/,
    );
    const missingRequestMemoryField = fixtureReceipt();
    delete missingRequestMemoryField.authorityLifecycle.requestMemoryStrategy.stageResidency;
    const missingRequestMemoryFieldFile = await writeReceipt(
      "missing-request-memory-field.json", missingRequestMemoryField,
    );
    assert.throws(
      () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, missingRequestMemoryFieldFile),
      /must have required property 'stageResidency'/,
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
  await writeFile(path.join(boundaryDir, "controller.log"), "2026-08-22T14:52:25.935Z starting flux1-dev-q8\n", "utf8");
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

async function writeRecoveryCandidate(root) {
  const recoveryProfile = recoveryArtifactProfile();
  const legacyRoot = path.join(path.dirname(root), `${path.basename(root)}-legacy-source`);
  await mkdir(legacyRoot);
  const legacy = await writePrefixCandidate(legacyRoot);
  const candidate = path.join(root, "9488587517");
  const evidence = path.join(candidate, "evidence");
  await mkdir(evidence, { recursive: true });
  const originalLineage = {
    kind: "contiguous-pass-prefix",
    sourceArtifactId: "9477529627",
    sourceArtifactName: "sc-20945-epic-20738-8886a9e69f26beec05688c81b414859bd102f6d0-32570707303-1",
    sourceArtifactDigest: "sha256:f3164a32a485fdedd671f4e11f30038213d30a7eb2b541bda90bef30e63188f3",
    sourceRunId: "32570707303",
    sourceRunAttempt: "1",
    sourceHeadSha: "8886a9e69f26beec05688c81b414859bd102f6d0",
    sourceInferenceSha: "b646a6f89ba9f6b07efe53dd583d8a42e21e9871",
    sourceProfile: PROFILE_NAME,
    sourceCellSemanticsSha256: "dc0e529b40e898727eb9401562a928345b958b4c94677d0206ccc70471f6f879",
    sourceArtifactSemanticsSha256: "f2bb7a77b83ce11cc32c3a1f9639534a67a149bc464a9730fb5c0988b4a03f9e",
    importedOrdinals: [1, 2, 3, 4, 5, 6, 7],
    quarantinedBoundaryResidue: {
      ordinal: 8, cellId: "flux1-dev-q8", path: "_imported-boundary-residue/08-flux1-dev-q8",
      disposition: "non-executed-pre-provision-skeleton-excluded-from-prefix",
      files: [{ path: "controller.log", bytes: 47, sha256: "6043cbeeeed54deec723c1ab5fcec6b36a8de4f7b31928c6b57222fd7dfec770" }],
    },
  };
  const imported = path.join(evidence, "_imported-prefix");
  await mkdir(imported);
  for (let index = 0; index < 7; index += 1) {
    const cell = recoveryProfile.cells[index];
    const ordinal = `${String(index + 1).padStart(2, "0")}-${cell.id}`;
    await cp(path.join(legacy.evidence, ordinal), path.join(imported, ordinal), { recursive: true });
  }
  await writeFile(path.join(imported, "lineage.json"), `${JSON.stringify(originalLineage, null, 2)}\n`);
  await mkdir(path.join(evidence, "_imported-boundary-residue"));
  await cp(path.join(legacy.evidence, "08-flux1-dev-q8"), path.join(evidence, "_imported-boundary-residue", "08-flux1-dev-q8"), { recursive: true });
  const recoveryMetadata = {
    artifactId: "9488587517",
    artifactName: "sc-20945-epic-20738-62be42127e2b4ff07321e2c369de92fc6edef526-32616545132-1",
    artifactSize: 4_322_200,
    artifactDigest: "sha256:765c8f4ed419e7a7d0fbc20dcd65f5be6f0be7ce9ed9f151915208bf541692bf",
    runId: "32616545132", runAttempt: "1", headSha: "62be42127e2b4ff07321e2c369de92fc6edef526",
    inferenceSha: "b646a6f89ba9f6b07efe53dd583d8a42e21e9871", profile: PROFILE_NAME,
    cellSemanticsSha256: "dc0e529b40e898727eb9401562a928345b958b4c94677d0206ccc70471f6f879",
    artifactSemanticsSha256: "5b9ef60c18ab15caeca7ff0411b199618f0aa22cc051a70607aa7a0f7c6cd932",
  };
  await writeFile(path.join(candidate, "artifact-metadata.json"), `${JSON.stringify(recoveryMetadata, null, 2)}\n`);
  async function writeCurrentReceipt(parent, index, status) {
    const cell = recoveryProfile.cells[index];
    const ordinal = `${String(index + 1).padStart(2, "0")}-${cell.id}`;
    const cellDir = path.join(parent, ordinal);
    await mkdir(cellDir, { recursive: true });
    await writeFile(path.join(cellDir, "cell.json"), `{"id":"${cell.id}"}\n`);
    await writeFile(path.join(cellDir, "runtime-result.json"), "{}\n");
    await writeFile(path.join(cellDir, "controller.log"), `${status} ${cell.id}\n`);
    const receipt = fixtureReceipt(status, index, recoveryProfile);
    receipt.repositories.sceneworks.sha = recoveryMetadata.headSha;
    receipt.repositories.inference.sha = recoveryMetadata.inferenceSha;
    receipt.execution.headSha = recoveryMetadata.headSha;
    receipt.execution.runId = recoveryMetadata.runId;
    receipt.execution.runAttempt = recoveryMetadata.runAttempt;
    const files = await hashedFiles(cellDir);
    receipt.inputs = files.filter((file) => file.path === "cell.json");
    receipt.outputs = files.filter((file) => file.path === "runtime-result.json");
    receipt.logs = files.filter((file) => file.path === "controller.log");
    await writeFile(path.join(cellDir, "receipt.json"), `${JSON.stringify(receipt, null, 2)}\n`);
    return { cell, ordinal, receipt };
  }
  const recovered = [];
  for (const index of [7, 8]) recovered.push(await writeCurrentReceipt(evidence, index, "passed"));
  const staleCell = recoveryProfile.cells[9];
  const staleDir = path.join(evidence, `10-${staleCell.id}`);
  await mkdir(staleDir);
  await writeFile(path.join(staleDir, "controller.log"), "stale sentinel changed after receipt\n");
  const stale = receiptSkeleton({
    cell: staleCell, ordinal: 10,
    repositories: { sceneworks: { sha: recoveryMetadata.headSha, clean: true }, inference: { sha: recoveryMetadata.inferenceSha, clean: true } },
    artifacts: fixtureReceipt("failed", 9, recoveryProfile).artifacts,
    execution: { ...fixtureReceipt("failed", 9, recoveryProfile).execution, runId: recoveryMetadata.runId, runAttempt: recoveryMetadata.runAttempt, headSha: recoveryMetadata.headSha },
    gpuIdentity: fixtureReceipt("failed", 9, recoveryProfile).hardware.gpuIdentity,
    systemMemory: fixtureReceipt("failed", 9, recoveryProfile).hardware.systemMemory, startedAt: "2026-08-23T00:00:00.000Z",
  });
  stale.hardware.rawVramSamples = [{ raw: "fixture raw" }];
  stale.logs = [{ path: "controller.log", bytes: 1, sha256: "0".repeat(64) }];
  stale.authorityLifecycle = {
    ordinal: 10, cellId: staleCell.id, staged: [], activeArtifactIds: [...staleCell.artifactIds],
    providerExecution: "not-attempted", requestMemoryStrategy: null, verifiedBefore: false,
    verifiedAfter: false, derivedAfter: [], released: [], diskProbes: [],
  };
  await writeFile(path.join(staleDir, "receipt.json"), `${JSON.stringify(stale, null, 2)}\n`);
  const emergency = path.join(evidence, "_emergency");
  const failures = [];
  for (let index = 9; index < 19; index += 1) {
    const row = await writeCurrentReceipt(emergency, index, "failed");
    failures.push({ id: row.cell.id, status: "failed", receipt: `_emergency/${row.ordinal}/receipt.json` });
  }
  const expected = expectedArtifactFilesFromEvidence(recoveryProfile, JSON.parse(await readFile(
    "config/download-pattern-evidence.json", "utf8",
  )));
  const ids = [...new Set(recoveryProfile.cells.slice(7).flatMap((cell) => cell.artifactIds))];
  const sourceBytes = new Map([
    ["flux1-dev-q8", 18_010_071_290], ["flux1-schnell-q4", 9_611_551_284], ["flux1-schnell-q8", 17_999_175_703],
    ["scail2-q4", 23_993_093_306], ["scail2-q8", 32_067_131_269], ["ltx23-q8", 29_728_720_716],
    ["ltx23-gemma", 26_427_894_918], ["sdxl-base-q4", 2_703_202_068], ["sdxl-openpose", 2_502_139_104],
    ["sdxl-tokenizer-l", 2_224_003], ["sdxl-tokenizer-bigg", 2_224_041], ["sdxl-vae-fix", 334_643_238],
    ["realvisxl-q4", 3_911_656_467], ["realvisxl-lightning-q4", 3_911_657_599], ["illustrious-v1-q4", 3_911_656_791], ["illustrious-v2-q4", 3_911_656_467],
  ]);
  const downloaded = { artifactId: "flux1-schnell-q8", path: "transformer/model.safetensors", bytes: 12_637_521_668,
    sha256: "c62fff59c0a5205204def102f5183b703ae8a8bb4b6b952c640f27e20d3e03f7", lfsSha256: "c62fff59c0a5205204def102f5183b703ae8a8bb4b6b952c640f27e20d3e03f7", commitSha: "bba3ae01dfd94089f173c05edd4e1a4c551f2599" };
  const sourceCensus = ids.map((id) => {
    const artifact = recoveryProfile.artifacts[id];
    const missingFiles = id === "flux1-schnell-q8" ? ["q8/transformer/model.safetensors"] : [];
    const presentFiles = expected[id].filter((file) => id !== "flux1-schnell-q8" || file !== "transformer/model.safetensors");
    const reusedTotal = sourceBytes.get(id) - (id === downloaded.artifactId ? downloaded.bytes : 0);
    const reusedFiles = presentFiles.map((file, index) => ({ path: file, bytes: index === 0 ? reusedTotal - presentFiles.length + 1 : 1, sha256: "a".repeat(64) }));
    return { id, repository: artifact.repository, revision: artifact.revision, subdirectory: artifact.subdirectory,
      allowPatterns: artifact.allowPatterns, expectedFiles: expected[id], complete: missingFiles.length === 0, missingFiles, reusedFiles };
  });
  const stagedDownload = (({ path: file, bytes, sha256, lfsSha256, commitSha }) => ({ path: file, bytes, sha256, lfsSha256, commitSha }))(downloaded);
  const plannedAudits = new Map(sourceCensus.map((row) => [row.id, {
    ...row, downloadedFiles: row.id === downloaded.artifactId ? [stagedDownload] : [],
  }]));
  const lifetimePlan = authorityLifetimePlan(
    recoveryProfile, Array.from({ length: 12 }, (_, index) => index + 8), plannedAudits,
  );
  const diskPlan = estimateJitDiskPlan(
    lifetimePlan, 248_674_082_816, downloaded.bytes, [], 179_028_698_264,
  );
  const emptyInventory = { root: "fixture-derived", files: 0, bytes: 0, sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" };
  const staging = ["flux1-dev-q8", "flux1-schnell-q4"].map((id) => {
    const row = sourceCensus.find((candidate) => candidate.id === id);
    return { id: row.id, repository: row.repository, revision: row.revision, subdirectory: row.subdirectory,
      allowPatterns: row.allowPatterns, expectedFiles: row.expectedFiles, reusedFiles: expected[id].map((file) => ({ path: file, bytes: 1, sha256: "a".repeat(64) })), downloadedFiles: [] };
  });
  const finalOffline = staging.map((row) => ({ id: row.id, repository: row.repository, revision: row.revision,
    subdirectory: row.subdirectory, allowPatterns: row.allowPatterns, expectedFiles: row.expectedFiles,
    inventory: { ...emptyInventory, root: `fixture-stage/${row.id}`, files: row.expectedFiles.length, bytes: row.expectedFiles.length, sha256: "a".repeat(64) } }));
  const cacheBase = {
    schemaVersion: 1, profile: PROFILE_NAME,
    downloadEvidenceSha256: "9eda09eeacb9386167ca4a080b4805b9c7dd3cd5134ca037ce342ad434b17e0b",
    expectedArtifactIds: ids, sourceCacheRoot: "fixture-cache", campaignStagingRoot: "fixture-stage",
    derivedSidecarRoot: "fixture-derived", missingFileStore: "fixture-missing", frozenMissingFiles: [{ artifactId: "flux1-schnell-q8", repository: recoveryProfile.artifacts["flux1-schnell-q8"].repository, revision: recoveryProfile.artifacts["flux1-schnell-q8"].revision, file: "q8/transformer/model.safetensors" }],
    sidecarObstructions: [], reusedFiles: sourceCensus.flatMap((row) => row.reusedFiles.map((file) => ({ artifactId: row.id, ...file }))),
    downloadedFiles: [downloaded], networkDownloadCount: 1, phases: { sourceCensus, staging: [], finalOffline: [] },
    lifetimePlan, diskPlan, derivedSidecarLifecycle: { initial: emptyInventory, afterCells: [] }, offlineBeforeCells: true,
  };
  const campaignError = "JIT authority stage/copy/hash failed before cell flux1-schnell-q8: Error: offline JIT stage changed the frozen download partition for flux1-schnell-q8";
  await writeFile(path.join(evidence, "cache-preflight-initial.json"), `${JSON.stringify({ ...cacheBase, evidencePhase: "initial", status: "passed", error: null }, null, 2)}\n`);
  await writeFile(path.join(evidence, "cache-preflight.json"), `${JSON.stringify({ ...cacheBase, evidencePhase: "final", status: "failed", error: campaignError,
    sidecarObstructions: Array.from({ length: 16 }, (_, index) => ({ artifactIds: [ids[index]], root: `fixture-stage/${ids[index]}`, path: ".candle-device-format-v1", bytes: 75, sha256: "a".repeat(64) })),
    phases: { sourceCensus, staging, finalOffline },
  }, null, 2)}\n`);
  const cacheBytes = await readFile(path.join(evidence, "cache-preflight.json"));
  const summary = {
    schemaVersion: 1, profile: PROFILE_NAME,
    repositories: { sceneworks: { sha: recoveryMetadata.headSha, clean: true }, inference: { sha: recoveryMetadata.inferenceSha, clean: true } },
    execution: { runId: recoveryMetadata.runId, runAttempt: recoveryMetadata.runAttempt, headSha: recoveryMetadata.headSha,
      headRef: "refs/heads/feature/sc-20738-candle-cuda-parity", workflow: "Windows Candle worker", runnerName: "cuda-windows-2", runnerOs: "Windows", runnerArch: "X64" },
    receipts: [
      ...recoveryProfile.cells.slice(0, 7).map((cell, index) => ({ id: cell.id, status: "passed", receipt: `_imported-prefix/${String(index + 1).padStart(2, "0")}-${cell.id}/receipt.json`, error: null, emergencyReceiptError: null, source: "imported" })),
      ...recovered.map((row) => ({ id: row.cell.id, status: "passed", receipt: `${row.ordinal}/receipt.json`, error: null, source: "continuation" })),
      ...failures.map((row, index) => ({ ...row, error: index === 0 ? `cell lifecycle failed: Error: ${campaignError}` : `Error: cell ${row.id} blocked before setup, provisioning, and execution: ${campaignError}`, emergencyReceiptError: null, source: "continuation" })),
    ],
    lineage: {
      imported: originalLineage,
      continuation: {
        runId: recoveryMetadata.runId, runAttempt: recoveryMetadata.runAttempt, headSha: recoveryMetadata.headSha,
        inferenceSha: recoveryMetadata.inferenceSha,
        profileCellSemanticsSha256: recoveryMetadata.cellSemanticsSha256,
        profileArtifactSemanticsSha256: recoveryMetadata.artifactSemanticsSha256, startOrdinal: 8,
      },
    },
    cachePreflight: { path: "cache-preflight.json", bytes: cacheBytes.length, sha256: createHash("sha256").update(cacheBytes).digest("hex") },
    authorityLifecycle: recoveryProfile.cells.slice(7).map((cell, offset) => ({ ordinal: offset + 8, cellId: cell.id,
      staged: offset < 2 ? [{ artifactId: cell.artifactIds[0] }] : [], activeArtifactIds: [...cell.artifactIds], providerExecution: offset < 2 ? "completed" : "not-attempted",
      requestMemoryStrategy: offset < 2 ? { strategy: "default-resident", requestMemoryPresent: false, stageResidency: false, streamTransformerBlocks: false } : null,
      verifiedBefore: offset < 2, verifiedAfter: offset < 2, derivedAfter: offset < 2 ? [{}] : [], released: offset < 3 ? [{}] : [], diskProbes: offset < 2 ? [{}, {}] : offset === 2 ? [{}] : [],
    })),
    diskFreeProbes: [["before-stage:flux1-dev-q8", 8], ["before-execution", 8], ["before-stage:flux1-schnell-q4", 9], ["before-execution", 9], ["before-stage:flux1-schnell-q8", 10]].map(([phase, ordinal]) => ({ phase, ordinal, root: "fixture", freeBytes: 100_000_000_000, requiredFreeBytes: 99_106_288_594 })),
    finalAuthorityLifecycle: { stage: { root: "fixture", files: 0, bytes: 0, sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" }, derived: { root: "fixture", files: 0, bytes: 0, sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" }, derivedNamespaces: [], missingStoreAbsent: true },
    passed: 9, failed: 10, campaignErrors: [campaignError],
  };
  await writeFile(path.join(evidence, "campaign-summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  return { candidate, evidence, summary };
}

async function writeSparseRecoveryCandidate(root) {
  const checked = recoveryArtifactProfile();
  const recoveryRoot = path.join(path.dirname(root), `${path.basename(root)}-recovery-source`);
  await mkdir(recoveryRoot);
  await writeRecoveryCandidate(recoveryRoot);
  const selectedRecovery = await selectRecoveryContinuation(recoveryRoot, profile());
  const candidate = path.join(root, "9492288293");
  const evidence = path.join(candidate, "evidence");
  await mkdir(evidence, { recursive: true });
  const imported = await importRecoveryContinuation(selectedRecovery, evidence);
  const metadata = {
    artifactId: "9492288293",
    artifactName: "sc-20945-epic-20738-43c718b7e9a852bd5029448d18841fed0f508c3a-32628540694-1",
    artifactSize: 15_452_320,
    artifactDigest: "sha256:dbae4c7d67d824bb8568909231614c6bcc268868087eb19974ce013bfc557724",
    runId: "32628540694",
    runAttempt: "1",
    headSha: "43c718b7e9a852bd5029448d18841fed0f508c3a",
    inferenceSha: "b646a6f89ba9f6b07efe53dd583d8a42e21e9871",
    profile: PROFILE_NAME,
    cellSemanticsSha256: "dc0e529b40e898727eb9401562a928345b958b4c94677d0206ccc70471f6f879",
    artifactSemanticsSha256: "5b9ef60c18ab15caeca7ff0411b199618f0aa22cc051a70607aa7a0f7c6cd932",
  };
  await writeFile(path.join(candidate, "artifact-metadata.json"), `${JSON.stringify(metadata, null, 2)}\n`);

  const rootReceipts = [];
  for (let index = 9; index < 19; index += 1) {
    const cell = checked.cells[index];
    const ordinal = `${String(index + 1).padStart(2, "0")}-${cell.id}`;
    const cellDir = path.join(evidence, ordinal);
    await mkdir(cellDir);
    await writeFile(path.join(cellDir, "cell.json"), `{"id":"${cell.id}"}\n`);
    await writeFile(path.join(cellDir, "runtime-result.json"), "{}\n");
    await writeFile(path.join(cellDir, "controller.log"), `audited ${cell.id}\n`);
    const status = [13, 17, 18].includes(index) ? "failed" : "passed";
    const receipt = fixtureReceipt(status, index, checked);
    receipt.repositories.sceneworks.sha = metadata.headSha;
    receipt.repositories.inference.sha = metadata.inferenceSha;
    receipt.execution.headSha = metadata.headSha;
    receipt.execution.runId = metadata.runId;
    receipt.execution.runAttempt = metadata.runAttempt;
    const files = await hashedFiles(cellDir);
    receipt.inputs = files.filter((file) => file.path === "cell.json");
    receipt.outputs = files.filter((file) => file.path === "runtime-result.json");
    receipt.logs = files.filter((file) => file.path === "controller.log");
    await writeFile(path.join(cellDir, "receipt.json"), `${JSON.stringify(receipt, null, 2)}\n`);
    rootReceipts.push({ cell, ordinal, receipt });
  }

  const sourceInitial = JSON.parse(await readFile(
    path.join(selectedRecovery.evidenceRoot, "cache-preflight-initial.json"), "utf8",
  ));
  const executionOrdinals = Array.from({ length: 10 }, (_, index) => index + 10);
  const ids = [...new Set(executionOrdinals.flatMap(
    (ordinal) => checked.cells[ordinal - 1].artifactIds,
  ))];
  const sourceCensus = sourceInitial.phases.sourceCensus.filter((row) => ids.includes(row.id));
  const downloadedFiles = sourceInitial.downloadedFiles.filter((file) => ids.includes(file.artifactId));
  const audits = new Map(sourceCensus.map((row) => [row.id, {
    ...row,
    downloadedFiles: downloadedFiles.filter((file) => file.artifactId === row.id).map(
      ({ artifactId, ...file }) => file,
    ),
  }]));
  const lifetimePlan = authorityLifetimePlan(checked, executionOrdinals, audits);
  const diskPlan = estimateJitDiskPlan(
    lifetimePlan,
    sourceInitial.diskPlan.freeBytes,
    downloadedFiles.reduce((sum, file) => sum + file.bytes, 0),
    sourceInitial.diskPlan.nonModelPaths,
    179_028_698_264,
  );
  const emptyInventory = {
    root: sourceInitial.derivedSidecarRoot,
    files: 0,
    bytes: 0,
    sha256: createHash("sha256").digest("hex"),
  };
  const cacheBase = {
    ...sourceInitial,
    expectedArtifactIds: ids,
    frozenMissingFiles: sourceInitial.frozenMissingFiles.filter((row) => ids.includes(row.artifactId)),
    sidecarObstructions: [],
    reusedFiles: sourceCensus.flatMap((row) => row.reusedFiles.map(
      (file) => ({ artifactId: row.id, ...file }),
    )),
    downloadedFiles,
    networkDownloadCount: downloadedFiles.length,
    phases: { sourceCensus, staging: [], finalOffline: [] },
    lifetimePlan,
    diskPlan,
    derivedSidecarLifecycle: { initial: emptyInventory, afterCells: [] },
    offlineBeforeCells: true,
  };
  await writeFile(path.join(evidence, "cache-preflight-initial.json"), `${JSON.stringify({
    ...cacheBase, evidencePhase: "initial", status: "passed", error: null,
  }, null, 2)}\n`);
  const staging = sourceCensus.map((row) => ({
    id: row.id,
    repository: row.repository,
    revision: row.revision,
    subdirectory: row.subdirectory,
    allowPatterns: row.allowPatterns,
    expectedFiles: row.expectedFiles,
    reusedFiles: row.reusedFiles,
    downloadedFiles: audits.get(row.id).downloadedFiles,
  }));
  const finalOffline = staging.map((row) => ({
    id: row.id,
    repository: row.repository,
    revision: row.revision,
    subdirectory: row.subdirectory,
    allowPatterns: row.allowPatterns,
    expectedFiles: row.expectedFiles,
    inventory: {
      root: `fixture-stage/${row.id}`,
      files: row.expectedFiles.length,
      bytes: [...row.reusedFiles, ...row.downloadedFiles].reduce((sum, file) => sum + file.bytes, 0),
      sha256: "a".repeat(64),
    },
  }));
  const sidecarObstructions = ids.map((id) => ({
    artifactIds: [id], root: `fixture-stage/${id}`, path: ".candle-device-format-v1",
    bytes: 7, sha256: "a".repeat(64),
  }));
  const afterCells = executionOrdinals.map((ordinal) => {
    const cell = checked.cells[ordinal - 1];
    const failed = [14, 18, 19].includes(ordinal);
    const flux = cell.artifactIds.some((id) => id.startsWith("flux1-"));
    return {
      ordinal,
      cellId: cell.id,
      providerExecution: failed ? "failed" : "completed",
      requestMemoryStrategy: failed ? null : {
        strategy: flux ? "default-resident" : "not-applicable",
        requestMemoryPresent: false,
        stageResidency: false,
        streamTransformerBlocks: false,
      },
      derivedDisposition: flux ? "resident-empty" : "not-applicable",
      inventory: emptyInventory,
    };
  });
  const finalCache = {
    ...cacheBase,
    evidencePhase: "final",
    status: "passed",
    error: null,
    sidecarObstructions,
    phases: { sourceCensus, staging, finalOffline },
    derivedSidecarLifecycle: { initial: emptyInventory, afterCells },
  };
  await writeFile(path.join(evidence, "cache-preflight.json"), `${JSON.stringify(finalCache, null, 2)}\n`);
  const cacheBytes = await readFile(path.join(evidence, "cache-preflight.json"));
  const diskFreeProbes = executionOrdinals.flatMap((ordinal) => {
    const starting = lifetimePlan.filter((row) => row.firstOrdinal === ordinal).map((row) => ({
      phase: `before-stage:${row.artifactId}`,
      ordinal,
      root: "fixture",
      freeBytes: 128 * 1024 ** 3,
      requiredFreeBytes: 99_106_288_594,
    }));
    return [...starting, {
      phase: "before-execution",
      ordinal,
      root: "fixture",
      freeBytes: 128 * 1024 ** 3,
      requiredFreeBytes: 99_106_288_594,
    }];
  });
  for (const { ordinal, receipt } of rootReceipts) {
    receipt.authorityLifecycle.diskProbes = diskFreeProbes.filter(
      (probe) => probe.ordinal === receipt.cell.ordinal,
    );
    await writeFile(
      path.join(evidence, ordinal, "receipt.json"),
      `${JSON.stringify(receipt, null, 2)}\n`,
    );
  }
  const summary = {
    schemaVersion: 1,
    profile: PROFILE_NAME,
    repositories: {
      sceneworks: { sha: metadata.headSha, clean: true },
      inference: { sha: metadata.inferenceSha, clean: true },
    },
    execution: {
      runId: metadata.runId,
      runAttempt: metadata.runAttempt,
      headSha: metadata.headSha,
      headRef: "refs/heads/feature/sc-20738-candle-cuda-parity",
      workflow: "Windows Candle worker",
      runnerName: "cuda-windows-2",
      runnerOs: "Windows",
      runnerArch: "X64",
    },
    receipts: [
      ...imported.outcomes,
      ...rootReceipts.map(({ cell, ordinal, receipt }) => ({
        id: cell.id,
        status: receipt.status,
        receipt: `${ordinal}/receipt.json`,
        error: receipt.error,
        source: "continuation",
      })),
    ],
    lineage: {
      imported: imported.lineage,
      continuation: {
        runId: metadata.runId,
        runAttempt: metadata.runAttempt,
        headSha: metadata.headSha,
        inferenceSha: metadata.inferenceSha,
        profileCellSemanticsSha256: metadata.cellSemanticsSha256,
        profileArtifactSemanticsSha256: metadata.artifactSemanticsSha256,
        startOrdinal: 10,
      },
    },
    cachePreflight: {
      path: "cache-preflight.json",
      bytes: cacheBytes.length,
      sha256: createHash("sha256").update(cacheBytes).digest("hex"),
    },
    authorityLifecycle: rootReceipts.map(({ receipt }) => receipt.authorityLifecycle),
    diskFreeProbes,
    finalAuthorityLifecycle: {
      stage: { ...emptyInventory, root: "fixture-stage" },
      derived: { ...emptyInventory, root: "fixture-derived" },
      derivedNamespaces: [],
      missingStoreAbsent: true,
    },
    passed: 16,
    failed: 3,
    campaignErrors: [],
  };
  await writeFile(path.join(evidence, "campaign-summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  return { candidate, evidence, metadata, summary };
}

test("sparse recovery imports only per-cell-compatible PASS receipts and quarantines 14, 18, and 19", async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), "sc-21306-sparse-recovery-"));
  try {
    const validRoot = path.join(temporary, "valid");
    await mkdir(validRoot);
    const fixture = await writeSparseRecoveryCandidate(validRoot);
    const selected = await selectSparseRecovery(validRoot, profile());
    assert.deepEqual(
      selected.receipts.map(({ cell }) => profile().cells.findIndex((row) => row.id === cell.id) + 1),
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17],
    );
    assert.deepEqual(selected.failures.map(({ ordinal }) => ordinal), [14, 18, 19]);
    assert.equal(selected.compatibility.length, 16);
    const output = path.join(temporary, "output");
    await mkdir(output);
    const imported = await importSparseRecovery(selected, output);
    assert.equal(imported.outcomes.length, 16);
    assert.deepEqual(imported.lineage.executionOrdinals, [14, 18, 19]);
    assert.equal((await readdir(path.join(output, "_imported-prefix"))).includes("14-ltx-2-3-q8"), false);

    for (const [label, mutate, pattern] of [
      ["metadata", async (candidate) => {
        const file = path.join(candidate, "artifact-metadata.json");
        const value = JSON.parse(await readFile(file));
        value.artifactDigest = `sha256:${"0".repeat(64)}`;
        await writeFile(file, JSON.stringify(value));
      }, /artifact 9492288293/],
      ["pass-rehash", async (candidate) => {
        await writeFile(path.join(candidate, "evidence", "15-sdxl-openpose", "controller.log"), "tampered\n");
      }, /rehash/],
      ["failed-promoted", async (candidate) => {
        const file = path.join(candidate, "evidence", "14-ltx-2-3-q8", "receipt.json");
        const value = JSON.parse(await readFile(file));
        value.status = "passed";
        value.error = null;
        await writeFile(file, JSON.stringify(value));
      }, /exact audited FAILED|summary receipt/],
      ["summary-partition", async (candidate) => {
        const file = path.join(candidate, "evidence", "campaign-summary.json");
        const value = JSON.parse(await readFile(file));
        value.receipts[17].status = "passed";
        await writeFile(file, JSON.stringify(value));
      }, /path.source.status/],
      ["archived-lineage", async (candidate) => {
        const file = path.join(candidate, "evidence", "_imported-prefix", "lineage.json");
        const value = JSON.parse(await readFile(file));
        value.importedOrdinals.pop();
        await writeFile(file, JSON.stringify(value));
      }, /archived lineage/],
    ]) {
      const root = path.join(temporary, label);
      await mkdir(root);
      await cp(fixture.candidate, path.join(root, "9492288293"), { recursive: true });
      await mutate(path.join(root, "9492288293"));
      await assert.rejects(selectSparseRecovery(root, profile()), pattern, label);
    }
    const incompatible = profile();
    incompatible.cells[14].request.steps += 1;
    await assert.rejects(selectSparseRecovery(validRoot, incompatible), /semantic tuples drifted/);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test("recovery continuation imports only audited PASS cells and rejects ambiguity or tampering", async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), "sc-21252-recovery-"));
  try {
    const validRoot = path.join(temporary, "valid");
    await mkdir(validRoot);
    const fixture = await writeRecoveryCandidate(validRoot);
    const selected = await selectRecoveryContinuation(validRoot, profile());
    assert.deepEqual(selected.receipts.map(({ ordinalName }) => ordinalName), [
      "01-chroma1-base-q4", "02-chroma1-base-q8", "03-chroma1-flash-q4", "04-chroma1-flash-q8",
      "05-chroma1-hd-q4", "06-chroma1-hd-q8", "07-flux1-dev-q4", "08-flux1-dev-q8", "09-flux1-schnell-q4",
    ]);
    const output = path.join(temporary, "output");
    await mkdir(output);
    const imported = await importRecoveryContinuation(selected, output);
    assert.equal(imported.outcomes.length, 9);
    assert.equal(imported.outcomes.some((outcome) => outcome.id === "flux1-schnell-q8"), false);
    assert.equal((await readdir(path.join(output, "_imported-prefix"))).includes("10-flux1-schnell-q8"), false);
    for (const [label, mutate, pattern] of [
      ["metadata", async (candidate) => { const file = path.join(candidate, "artifact-metadata.json"); const value = JSON.parse(await readFile(file)); value.artifactDigest = "sha256:" + "0".repeat(64); await writeFile(file, JSON.stringify(value)); }, /metadata/],
      ["summary", async (candidate) => { const file = path.join(candidate, "evidence", "campaign-summary.json"); const value = JSON.parse(await readFile(file)); value.passed = 8; await writeFile(file, JSON.stringify(value)); }, /summary/],
      ["receipt", async (candidate) => { await writeFile(path.join(candidate, "evidence", "08-flux1-dev-q8", "controller.log"), "tampered\n"); }, /rehash/],
      ["legacy-provenance", async (candidate) => { const file = path.join(candidate, "evidence", "_imported-prefix", "01-chroma1-base-q4", "receipt.json"); const value = JSON.parse(await readFile(file)); value.repositories.sceneworks.sha = "0".repeat(40); await writeFile(file, JSON.stringify(value)); }, /provenance/],
      ["legacy-clean", async (candidate) => { const file = path.join(candidate, "evidence", "_imported-prefix", "02-chroma1-base-q8", "receipt.json"); const value = JSON.parse(await readFile(file)); value.repositories.inference.clean = false; await writeFile(file, JSON.stringify(value)); }, /provenance/],
      ["recovery-provenance", async (candidate) => { const file = path.join(candidate, "evidence", "08-flux1-dev-q8", "receipt.json"); const value = JSON.parse(await readFile(file)); value.execution.headSha = "0".repeat(40); await writeFile(file, JSON.stringify(value)); }, /provenance/],
      ["stale-provenance", async (candidate) => { const file = path.join(candidate, "evidence", "10-flux1-schnell-q8", "receipt.json"); const value = JSON.parse(await readFile(file)); value.execution.runAttempt = "2"; await writeFile(file, JSON.stringify(value)); }, /provenance/],
      ["emergency-provenance", async (candidate) => { const file = path.join(candidate, "evidence", "_emergency", "11-scail2-q4", "receipt.json"); const value = JSON.parse(await readFile(file)); value.repositories.sceneworks.clean = false; await writeFile(file, JSON.stringify(value)); }, /clean/],
      ["lineage", async (candidate) => { const file = path.join(candidate, "evidence", "_imported-prefix", "lineage.json"); const value = JSON.parse(await readFile(file)); value.quarantinedBoundaryResidue.files[0].sha256 = "0".repeat(64); await writeFile(file, JSON.stringify(value)); }, /original lineage/],
      ["boundary", async (candidate) => { await writeFile(path.join(candidate, "evidence", "_imported-boundary-residue", "08-flux1-dev-q8", "controller.log"), "tampered\n"); }, /boundary residue/],
      ["summary-path", async (candidate) => { const file = path.join(candidate, "evidence", "campaign-summary.json"); const value = JSON.parse(await readFile(file)); value.receipts[8].receipt = "09-flux1-schnell-q4/wrong.json"; await writeFile(file, JSON.stringify(value)); }, /path.source.status/],
      ["summary-source", async (candidate) => { const file = path.join(candidate, "evidence", "campaign-summary.json"); const value = JSON.parse(await readFile(file)); value.receipts[7].source = "imported"; await writeFile(file, JSON.stringify(value)); }, /path.source.status/],
      ["cache-role", async (candidate) => { const evidence = path.join(candidate, "evidence"); const initial = await readFile(path.join(evidence, "cache-preflight-initial.json")); await writeFile(path.join(evidence, "cache-preflight.json"), initial); const summaryFile = path.join(evidence, "campaign-summary.json"); const summary = JSON.parse(await readFile(summaryFile)); summary.cachePreflight.bytes = initial.length; summary.cachePreflight.sha256 = createHash("sha256").update(initial).digest("hex"); await writeFile(summaryFile, JSON.stringify(summary)); }, /cache evidence phases/],
    ]) {
      const root = path.join(temporary, label);
      await mkdir(root);
      await cp(fixture.candidate, path.join(root, "9488587517"), { recursive: true });
      await mutate(path.join(root, "9488587517"));
      await assert.rejects(selectRecoveryContinuation(root, profile()), pattern, label);
    }
    const ambiguous = path.join(temporary, "ambiguous");
    await mkdir(ambiguous);
    await cp(fixture.candidate, path.join(ambiguous, "9488587517"), { recursive: true });
    await cp(fixture.candidate, path.join(ambiguous, "duplicate"), { recursive: true });
    await assert.rejects(selectRecoveryContinuation(ambiguous, profile()), /exactly one/);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test("start-10 cache preflight binds the reviewed remaining census without weakening global references", async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), "sc-21280-start10-census-"));
  try {
    const fixture = await writeRecoveryCandidate(temporary);
    const initial = JSON.parse(await readFile(path.join(fixture.evidence, "cache-preflight-initial.json"), "utf8"));
    const checked = recoveryArtifactProfile();
    const ids = [...new Set(checked.cells.slice(9).flatMap((cell) => cell.artifactIds))];
    const sourceCensus = initial.phases.sourceCensus.filter((row) => ids.includes(row.id));
    const plannedAudits = new Map(sourceCensus.map((row) => [row.id, {
      ...row,
      downloadedFiles: initial.downloadedFiles.filter((file) => file.artifactId === row.id).map(
        ({ artifactId, ...file }) => file,
      ),
    }]));
    const lifetimePlan = authorityLifetimePlan(
      checked, Array.from({ length: 10 }, (_, index) => index + 10), plannedAudits,
    );
    const diskPlan = estimateJitDiskPlan(
      lifetimePlan, initial.diskPlan.freeBytes,
      initial.downloadedFiles.reduce((sum, file) => sum + file.bytes, 0),
      initial.diskPlan.nonModelPaths,
      179_028_698_264,
    );
    const startTen = {
      ...initial,
      expectedArtifactIds: ids,
      phases: { ...initial.phases, sourceCensus },
      reusedFiles: sourceCensus.flatMap((row) => row.reusedFiles.map(
        (file) => ({ artifactId: row.id, ...file }),
      )),
      frozenMissingFiles: sourceCensus.flatMap((row) => row.missingFiles.map((file) => ({
        artifactId: row.id, repository: row.repository, revision: row.revision, file,
      }))),
      lifetimePlan,
      diskPlan,
    };
    const validation = {
      remainingArtifactIds: ids,
      artifactExpectedFiles: Object.fromEntries(sourceCensus.map((row) => [row.id, row.expectedFiles])),
      downloadEvidenceSha256: startTen.downloadEvidenceSha256,
      guard: { cacheRoot: startTen.sourceCacheRoot },
      stagingRoot: startTen.campaignStagingRoot,
      derivedSidecarRoot: startTen.derivedSidecarRoot,
      missingStore: startTen.missingFileStore,
      expectedNonModelPaths: startTen.diskPlan.nonModelPaths,
      profile: checked,
    };
    assert.equal(startTen.diskPlan.cells.length, 10);
    assert.equal(startTen.diskPlan.logicalSourceBytes, 151_407_075_690);
    assert.equal(startTen.diskPlan.allAtOnceSourceBytes, 151_407_075_690);
    assert.equal(startTen.diskPlan.reviewedAllAtOnceSourceBytes, 179_028_698_264);
    assert.equal(startTen.diskPlan.peakModelAndSidecarBytes, 56_156_615_634);
    assert.equal(startTen.diskPlan.peakRequiredAdditionalBytes, 99_106_288_594);
    assert.doesNotThrow(() => validateCachePreflightEvidence(startTen, validation));

    const shiftedIds = [...new Set(checked.cells.slice(10).flatMap((cell) => cell.artifactIds))];
    const shiftedCensus = sourceCensus.filter((row) => shiftedIds.includes(row.id));
    const shiftedAudits = new Map(shiftedCensus.map((row) => [row.id, {
      ...row, downloadedFiles: [],
    }]));
    const shifted = structuredClone(startTen);
    shifted.expectedArtifactIds = shiftedIds;
    shifted.phases.sourceCensus = shiftedCensus;
    shifted.reusedFiles = shiftedCensus.flatMap((row) => row.reusedFiles.map(
      (file) => ({ artifactId: row.id, ...file }),
    ));
    shifted.frozenMissingFiles = [];
    shifted.downloadedFiles = [];
    shifted.networkDownloadCount = 0;
    shifted.lifetimePlan = authorityLifetimePlan(
      checked, Array.from({ length: 9 }, (_, index) => index + 11), shiftedAudits,
    );
    shifted.diskPlan = estimateJitDiskPlan(
      shifted.lifetimePlan, initial.diskPlan.freeBytes, 0, initial.diskPlan.nonModelPaths,
      179_028_698_264,
    );
    const shiftedValidation = {
      ...validation,
      remainingArtifactIds: shiftedIds,
      artifactExpectedFiles: Object.fromEntries(shiftedCensus.map((row) => [row.id, row.expectedFiles])),
    };
    assert.throws(
      () => validateCachePreflightEvidence(shifted, shiftedValidation),
      /target-byte totals/,
      "a forged later start cannot repurpose the reviewed start-10 total",
    );

    for (const [label, mutate, pattern] of [
      ["census", (value) => { value.phases.sourceCensus.reverse(); }, /top-level hit\/download partition|omitted sourceCensus|start\/census/],
      ["total", (value) => { value.diskPlan.logicalSourceBytes += 1; }, /disk plan|target-byte totals/],
      ["peak", (value) => { value.diskPlan.peakModelAndSidecarBytes -= 1; }, /disk plan|target-byte totals/],
      ["floor", (value) => { value.diskPlan.peakRequiredAdditionalBytes -= 1; }, /disk plan|target-byte totals/],
    ]) {
      const drifted = structuredClone(startTen);
      mutate(drifted);
      assert.throws(() => validateCachePreflightEvidence(drifted, validation), pattern, label);
    }
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

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

test("b646 derived sidecars occupy one exact canonical namespace with no global residue", async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), "sc-20974-derived-"));
  const selectedRoot = path.join(temporary, "stage", "q8");
  const transformer = path.join(selectedRoot, "transformer");
  const derived = path.join(temporary, "derived");
  try {
    await Promise.all([mkdir(transformer, { recursive: true }), mkdir(derived)]);
    const artifact = {
      id: "flux1-fixture-q8",
      selectedRoot,
      matchedFiles: ["transformer/model.safetensors"],
    };
    const expected = await expectedB646DerivedNamespace(artifact, derived);
    assert.match(path.basename(expected), /^[0-9a-f]{64}$/);
    const resident = await inspectDerivedSidecarRoot(
      derived, expected, { allowExactEmpty: true },
    );
    assert.equal(resident.namespaces.length, 0);
    assert.equal(resident.rootInventory.files, 0);
    const empty = await inspectDerivedSidecarRoot(
      derived, expected, { materializeExactEmpty: true },
    );
    assert.equal(empty.namespaces.length, 1);
    assert.equal(empty.namespaces[0].inventory.files, 0);
    assert.equal(empty.rootInventory.files, 0);
    await mkdir(expected, { recursive: true });
    await Promise.all(Array.from({ length: 494 }, (_, index) => (
      writeFile(path.join(expected, `${String(index).padStart(3, "0")}.safetensors`), "x")
    )));
    const exact = await inspectDerivedSidecarRoot(derived, expected);
    assert.equal(exact.namespaces.length, 1);
    assert.equal(exact.rootInventory.files, 494);
    assert.equal(exact.rootInventory.bytes, 494);

    const extra = path.join(path.dirname(expected), "f".repeat(64));
    await mkdir(extra);
    await assert.rejects(
      inspectDerivedSidecarRoot(derived, expected),
      /one exact b646 namespace/,
    );
    await rm(extra, { recursive: true });
    await writeFile(path.join(derived, "stray.bin"), "stray");
    await assert.rejects(
      inspectDerivedSidecarRoot(derived, expected),
      /stray files, directories, or versions/,
    );
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test("cleanup isolation failure preserves all 19 durable outcomes and quarantines later cells", async () => {
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
    const shouldFail = (index === 6 && stage === "cleanup")
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
    expectedB646DerivedNamespace: async (artifact, root) => (
      path.join(root, "candle-device-format-v1", "a".repeat(64))
    ),
    inspectDerivedSidecarRoot: async (root, expectedNamespace, options = {}) => {
      if (!expectedNamespace || options.allowExactEmpty) return {
        rootInventory: {
          root, files: 0, bytes: 0, sha256: createHash("sha256").digest("hex"),
        },
        namespaces: [],
      };
      const id = executedCells.at(-1);
      assert.ok(id?.startsWith("flux1-"));
      await mkdir(expectedNamespace, { recursive: true });
      const bytes = id.endsWith("q4") ? 7_396_392_960 : 12_573_868_032;
      const inventory = {
        root: expectedNamespace, files: 494, bytes, sha256: createHash("sha256").digest("hex"),
      };
      return {
        rootInventory: { ...inventory, root },
        namespaces: [{ path: expectedNamespace, inventory }],
      };
    },
    directoryInventory: async (root) => ({
      root,
      files: root.includes("flux1-") ? 494 : 0,
      bytes: root.includes("flux1-")
        ? (root.endsWith("q4") ? 7_396_392_960 : 12_573_868_032) : 0,
      sha256: createHash("sha256").digest("hex"),
    }),
    diskFreeBytes: async () => 256 * 1024 ** 3,
    executeCell: async ({ cellDir, logFile }) => {
      await writeFile(logFile, "fixture runtime\n", "utf8");
      const runtimeCell = JSON.parse(await readFile(path.join(cellDir, "cell.json"), "utf8"));
      executedCells.push(runtimeCell.id);
      const flux = runtimeCell.kind === "image"
        && new Set(["flux1_dev", "flux1_schnell"]).has(runtimeCell.engineId);
      await writeFile(path.join(cellDir, "runtime-result.json"), `${JSON.stringify({
        requestedTier: runtimeCell.requestedTier,
        resolvedTier: runtimeCell.requestedTier,
        denseFallback: false,
        loadSpecQuantBits: expectedLoadSpecQuantBits(runtimeCell),
        requestMemoryStrategy: {
          strategy: flux ? "default-resident" : "not-applicable",
          requestMemoryPresent: false,
          stageResidency: false,
          streamTransformerBlocks: false,
        },
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
      fault, operations, suppressVerdict: true,
      executionOrdinals: Array.from({ length: 19 }, (_, index) => index + 1),
    });
    assert.deepEqual(
      attemptedCells,
      Array.from({ length: 7 }, (_, index) => index),
      result.summary.campaignErrors.join("\n"),
    );
    assert.equal(preflightCalls.filter((call) => call.startsWith("audit:")).length, 23);
    assert.equal(preflightCalls.filter((call) => call.startsWith("stage:")).length, 7);
    assert.equal(preflightCalls.filter((call) => call.startsWith("offline:")).length, 7);
    assert.ok(executedCells.length > 0);
    assert.ok(executedCells.every((id) => checked.cells.slice(0, 7).some((cell) => cell.id === id)));
    assert.equal(executedCells.at(-1), checked.cells[6].id);
    assert.equal(result.summary.receipts.length, 19);
    assert.equal(new Set(result.summary.receipts.map((receipt) => receipt.id)).size, 19);
    assert.equal(result.summaryPath, "_emergency/campaign-summary-fallback.json");
    assert.match(result.summary.campaignErrors.join("\n"), /primary campaign summary write failed/);
    for (const [index, outcome] of result.summary.receipts.entries()) {
      assert.ok(outcome.receipt, `${outcome.id} must retain a receipt path: ${outcome.error}`);
      const receiptFile = path.join(output, ...outcome.receipt.split("/"));
      const receipt = JSON.parse(await readFile(receiptFile, "utf8"));
      assert.doesNotThrow(() => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, receipt));
      assert.doesNotThrow(() => validateReceipt(receipt, checked.cells[index], checked));
    }
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

test("continuation freezes census, downloads once, and JIT stages exact authority lifetimes", async () => {
  async function scenario({
    missingId = null, downloadSucceeds = false, mutateCache = false, mutateSourceAtStage = false,
    preflightFault = null, cellFault = null, omitObstructions = false,
    derivedContractDrift = null, diskFreeValues = [256 * 1024 ** 3],
    runtimeResultMode = "valid", providerRuntimeFailureAt = null,
    providerFailureDerivedMode = "empty", successfulFluxDerivedMode = "resident",
    runtimeMemoryMode = "current", sparse = false,
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
    let diskFreeCall = 0;
    let cellFaultCalls = 0;
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
      downloadReviewedMissing: async ({ id, missingStore }) => {
        events.push(`download:${id}`);
        if (!downloadSucceeds) throw new Error("fixture exact missing-file transfer failed");
        return {
          id,
          storeRoot: missingStore,
          downloadedFiles: [{
            path: "transformer/model.safetensors",
            bytes: 200,
            sha256: "b".repeat(64),
            lfsSha256: "b".repeat(64),
            commitSha: "bba3ae01dfd94089f173c05edd4e1a4c551f2599",
          }],
        };
      },
      stageArtifact: async ({ id, artifact, stagingRoot, missingStore }) => {
        events.push(`stage:${id}`);
        const usesMissing = id === "flux1-schnell-q8" && missingId === id;
        return {
          id, ...artifact,
          reusedFiles: artifactExpectedFiles[id].filter((file) => !(
            usesMissing && file === "transformer/model.safetensors"
          )).map((file, index) => ({
            path: file,
            bytes: 100,
            sha256: mutateSourceAtStage
              && events.filter((event) => event.startsWith("stage:")).length === 1
              && index === 0 ? "c".repeat(64) : "a".repeat(64),
          })),
          downloadedFiles: usesMissing && missingStore ? [{
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
      expectedB646DerivedNamespace: async (artifact, root) => (
        path.join(root, "candle-device-format-v1", "a".repeat(64))
      ),
      inspectDerivedSidecarRoot: async (root, expectedNamespace, options = {}) => {
        if (!expectedNamespace) return {
          rootInventory: {
            root, files: 0, bytes: 0, sha256: createHash("sha256").digest("hex"),
          },
          namespaces: [],
        };
        const id = runtimeCells.at(-1)?.id;
        assert.ok(id?.startsWith("flux1-"));
        const providerFailed = id === providerRuntimeFailureAt;
        if (providerFailed) assert.equal(options.materializeExactEmpty, true);
        if (!providerFailed && successfulFluxDerivedMode === "resident") {
          assert.equal(options.allowExactEmpty, true);
          return {
            rootInventory: {
              root, files: 0, bytes: 0, sha256: createHash("sha256").digest("hex"),
            },
            namespaces: [],
          };
        }
        if (!providerFailed && successfulFluxDerivedMode === "stray") {
          throw new Error("fixture FLUX derived sidecar root has stray files");
        }
        await mkdir(expectedNamespace, { recursive: true });
        const bytes = id.endsWith("q4") ? 7_396_392_960 : 12_573_868_032;
        const inventory = {
          root: expectedNamespace,
          files: providerFailed
            ? (providerFailureDerivedMode === "partial" ? 1 : 0)
            : (derivedContractDrift?.files ?? 494),
          bytes: providerFailed
            ? (providerFailureDerivedMode === "partial" ? 1 : 0)
            : (derivedContractDrift?.bytes ?? bytes),
          sha256: createHash("sha256").digest("hex"),
        };
        return {
          rootInventory: { ...inventory, root },
          namespaces: [{ path: expectedNamespace, inventory }],
        };
      },
      directoryInventory: async (root) => ({
        root,
        files: root.includes("flux1-") ? 494 : 0,
        bytes: root.includes("flux1-")
          ? (root.endsWith("q4") ? 7_396_392_960 : 12_573_868_032) : 0,
        sha256: createHash("sha256").digest("hex"),
      }),
      diskFreeBytes: async () => diskFreeValues[Math.min(
        diskFreeCall++, diskFreeValues.length - 1,
      )],
      executeCell: async ({ cellDir, logFile }) => {
        await writeFile(logFile, "fixture runtime\n", "utf8");
        const runtimeCell = JSON.parse(await readFile(path.join(cellDir, "cell.json"), "utf8"));
        runtimeCells.push(runtimeCell);
        events.push(`execute:${runtimeCell.id}`);
        if (runtimeCell.id === providerRuntimeFailureAt) {
          throw new Error("fixture provider runtime failed");
        }
        if (runtimeResultMode !== "missing" || runtimeCells.length !== 1) {
          const isFlux = runtimeCell.kind === "image"
            && new Set(["flux1_dev", "flux1_schnell"]).has(runtimeCell.engineId);
          const requestMemoryStrategy = !isFlux ? {
            strategy: "not-applicable",
            requestMemoryPresent: false,
            stageResidency: false,
            streamTransformerBlocks: false,
          } : runtimeMemoryMode === "bounded" ? {
            strategy: "bounded-transformer",
            requestMemoryPresent: true,
            stageResidency: true,
            streamTransformerBlocks: true,
          } : runtimeMemoryMode === "staged" ? {
            strategy: "staged-resident",
            requestMemoryPresent: true,
            stageResidency: true,
            streamTransformerBlocks: false,
          } : runtimeMemoryMode === "bounded-no-stage" ? {
            strategy: "bounded-transformer",
            requestMemoryPresent: true,
            stageResidency: false,
            streamTransformerBlocks: true,
          } : runtimeMemoryMode === "wrong" ? {
            strategy: "bounded-transformer",
            requestMemoryPresent: true,
            stageResidency: true,
            streamTransformerBlocks: false,
          } : {
            strategy: "default-resident",
            requestMemoryPresent: false,
            stageResidency: false,
            streamTransformerBlocks: false,
          };
          const runtimeResultObject = {
            requestedTier: runtimeCell.requestedTier,
            resolvedTier: runtimeResultMode === "wrong-semantic" && runtimeCells.length === 1
              ? (runtimeCell.requestedTier === "q4" ? "q8" : "q4")
              : runtimeCell.requestedTier,
            denseFallback: false,
            loadSpecQuantBits: expectedLoadSpecQuantBits(runtimeCell),
            requestMemoryStrategy,
            ...(runtimeCell.capability === "multiReference" ? {
              metrics: {
                kind: "scail2",
                referencePairs: 6,
                referenceCounterfactuals: scail2ReferenceCounterfactuals(),
              },
            } : {}),
          };
          if (runtimeMemoryMode === "missing") delete runtimeResultObject.requestMemoryStrategy;
          if (runtimeMemoryMode === "missing-field") {
            delete runtimeResultObject.requestMemoryStrategy.streamTransformerBlocks;
          }
          const runtimeResult = runtimeResultMode === "malformed" && runtimeCells.length === 1
            ? "{"
            : `${JSON.stringify(runtimeResultObject)}\n`;
          if (runtimeResultMode === "symlink" && runtimeCells.length === 1) {
            await writeFile(path.join(cellDir, "runtime-result-target.json"), runtimeResult, "utf8");
            await symlink(
              "runtime-result-target.json", path.join(cellDir, "runtime-result.json"), "file",
            );
          } else {
            await writeFile(path.join(cellDir, "runtime-result.json"), runtimeResult, "utf8");
          }
        }
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
    const selectedExecutionOrdinals = sparse
      ? [14, 18, 19] : Array.from({ length: 12 }, (_, index) => index + 8);
    const selectedExecutionSet = new Set(selectedExecutionOrdinals);
    const importedPrefix = {
      lineage: { kind: "fixture-prefix" },
      outcomes: checked.cells.flatMap((cell, index) => selectedExecutionSet.has(index + 1) ? [] : [{
        id: cell.id, status: "passed", receipt: `_imported-prefix/${index + 1}/receipt.json`,
        error: null, emergencyReceiptError: null, source: "imported",
      }]),
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
        fault: preflightFault || cellFault ? async (stage, index) => {
          if (stage === preflightFault) throw new Error(`injected ${stage}`);
          if (cellFault && stage === cellFault.stage && index === (cellFault.index ?? 7)) {
            cellFaultCalls += 1;
            if (cellFaultCalls === (cellFault.occurrence ?? 1)) {
              throw new Error(`injected ${stage} at cell ${index}`);
            }
          }
        } : undefined,
        suppressVerdict: true,
        executionOrdinals: selectedExecutionOrdinals,
      });
      const cacheEvidence = JSON.parse(await readFile(
        path.join(output, ...result.summary.cachePreflight.path.split("/")),
        "utf8",
      ));
      const continuationReceipts = [];
      for (const outcome of result.summary.receipts.filter((row) => row.source === "continuation")) {
        if (!outcome.receipt) continue;
        continuationReceipts.push(JSON.parse(await readFile(
          path.join(output, ...outcome.receipt.split("/")), "utf8",
        )));
      }
      return {
        result, events, runtimeCells, checked, cacheEvidence,
        continuationReceipts,
        cacheValidation: {
          remainingArtifactIds: [...new Set(selectedExecutionOrdinals.flatMap(
            (ordinal) => checked.cells[ordinal - 1].artifactIds,
          ))],
          artifactExpectedFiles,
          downloadEvidenceSha256: "d".repeat(64),
          guard,
          stagingRoot: path.join(scratch, "authority-stage"),
          derivedSidecarRoot: path.join(scratch, "derived-candle-device-cache"),
          missingStore: path.join(scratch, "persistent-missing-file"),
          expectedNonModelPaths: [
            { kind: "cargoTarget", path: path.resolve(sceneworks, "target") },
            { kind: "cargoHome", path: path.resolve(process.env.CARGO_HOME ?? path.join(process.env.USERPROFILE ?? scratch, ".cargo")) },
            { kind: "campaignOutput", path: output },
            { kind: "pythonVenv", path: path.dirname(path.dirname(path.resolve("python"))) },
          ],
          profile: checked,
        },
        lifecycleContext: {
          lifetimeById: new Map(cacheEvidence.lifetimePlan.map((row) => [row.artifactId, row])),
          scratchRoot: scratch,
          requiredFreeBytes: cacheEvidence.diskPlan?.peakRequiredAdditionalBytes
            ?? 99_106_288_594,
          completeLifecycle: true,
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
  assert.equal(failed.cacheEvidence.offlineBeforeCells, false);
  assert.match(
    failed.result.summary.campaignErrors.join("\n"),
    /cache-only transfer\/disk preflight failed[\s\S]*exact missing-file transfer failed[\s\S]*no continuation GPU cell started/,
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
    /JIT authority stage\/copy\/hash failed[\s\S]*trusted source cache changed after frozen census/,
  );

  const diskNoGo = await scenario({ diskFreeValues: [1] });
  assert.equal(diskNoGo.runtimeCells.length, 0);
  assert.match(
    diskNoGo.result.summary.campaignErrors.join("\n"),
    /JIT staging disk admission refused[\s\S]*no continuation GPU cell started/,
  );

  const diskDropsBeforeStage = await scenario({
    diskFreeValues: [256 * 1024 ** 3, 1],
  });
  assert.equal(diskDropsBeforeStage.runtimeCells.length, 0);
  assert.match(
    diskDropsBeforeStage.result.summary.campaignErrors.join("\n"),
    /disk capacity fell below.*before-stage:flux1-dev-q8/,
  );

  const diskDropsBeforeExecution = await scenario({
    diskFreeValues: [256 * 1024 ** 3, 256 * 1024 ** 3, 1],
  });
  assert.equal(diskDropsBeforeExecution.runtimeCells.length, 0);
  assert.match(
    diskDropsBeforeExecution.result.summary.campaignErrors.join("\n"),
    /disk capacity fell below.*before-execution/,
  );
  assert.ok(diskDropsBeforeExecution.result.summary.receipts.slice(7).every(
    (outcome) => outcome.status === "failed" && outcome.receipt,
  ));

  for (const runtimeResultMode of ["missing", "malformed", "wrong-semantic"]) {
    const faulted = await scenario({ runtimeResultMode });
    assert.equal(faulted.runtimeCells.length, 1, runtimeResultMode);
    assert.ok(faulted.result.summary.receipts.slice(8).every(
      (outcome) => outcome.status === "failed" && outcome.receipt,
    ), runtimeResultMode);
    assert.match(
      faulted.result.summary.campaignErrors.join("\n"),
      /runtime-result evidence failed after cell flux1-dev-q8/,
      runtimeResultMode,
    );
    assert.equal(faulted.cacheEvidence.offlineBeforeCells, true, runtimeResultMode);
  }
  for (const runtimeMemoryMode of ["missing", "missing-field", "wrong", "bounded-no-stage"]) {
    const faulted = await scenario({ runtimeMemoryMode });
    assert.equal(faulted.runtimeCells.length, 1, runtimeMemoryMode);
    assert.ok(faulted.result.summary.receipts.slice(8).every(
      (outcome) => outcome.status === "failed" && outcome.receipt,
    ), runtimeMemoryMode);
    assert.match(
      faulted.result.summary.campaignErrors.join("\n"),
      /runtime-result evidence failed after cell flux1-dev-q8.*request memory strategy/,
      runtimeMemoryMode,
    );
  }
  const runtimeHashFault = await scenario({ cellFault: { stage: "runtimeResultHash" } });
  assert.equal(runtimeHashFault.runtimeCells.length, 1);
  assert.ok(runtimeHashFault.result.summary.receipts.slice(8).every(
    (outcome) => outcome.status === "failed" && outcome.receipt,
  ));
  assert.match(
    runtimeHashFault.result.summary.campaignErrors.join("\n"),
    /runtime-result evidence failed after cell flux1-dev-q8.*runtimeResultHash/,
  );
  assert.equal(runtimeHashFault.cacheEvidence.offlineBeforeCells, true);
  const falseAfterOffline = structuredClone(runtimeHashFault.cacheEvidence);
  falseAfterOffline.offlineBeforeCells = false;
  assert.doesNotThrow(() => validateDocumentWithSchema(
    CACHE_PREFLIGHT_SCHEMA_PATH,
    falseAfterOffline,
  ));
  assert.throws(
    () => validateCachePreflightEvidence(falseAfterOffline, runtimeHashFault.cacheValidation),
    /cell lifecycle evidence before network-offline establishment/,
  );

  const linkedRuntimeResult = await scenario({ runtimeResultMode: "symlink" });
  assert.equal(linkedRuntimeResult.runtimeCells.length, 1);
  assert.ok(linkedRuntimeResult.result.summary.receipts.slice(8).every(
    (outcome) => outcome.status === "failed" && outcome.receipt,
  ));
  assert.match(
    linkedRuntimeResult.result.summary.campaignErrors.join("\n"),
    /runtime-result evidence failed after cell flux1-dev-q8.*ordinary non-reparse file/,
  );
  assert.match(
    linkedRuntimeResult.result.summary.receipts[7].error,
    /evidence tree contains a symlink or reparse point/,
  );

  const providerFailed = await scenario({ providerRuntimeFailureAt: "flux1-dev-q8" });
  assert.equal(providerFailed.runtimeCells.length, 12);
  assert.equal(providerFailed.result.summary.failed, 1);
  assert.equal(providerFailed.result.summary.passed, 18);
  assert.equal(
    providerFailed.result.summary.campaignErrors.some((error) => error.includes("runtime-result evidence")),
    false,
  );
  const providerFailureReceipt = providerFailed.continuationReceipts[0];
  assert.equal(providerFailureReceipt.authorityLifecycle.providerExecution, "failed");
  assert.equal(providerFailureReceipt.authorityLifecycle.requestMemoryStrategy, null);
  assert.equal(
    providerFailureReceipt.authorityLifecycle.derivedAfter[0].derivedDisposition,
    "provider-failed-empty",
  );
  assert.equal(providerFailureReceipt.authorityLifecycle.derivedAfter[0].files, 0);
  assert.equal(providerFailureReceipt.authorityLifecycle.derivedAfter[0].inventories.length, 1);
  assert.equal(providerFailureReceipt.authorityLifecycle.staged[0].derivedNamespaces.length, 1);
  assert.equal(providerFailureReceipt.authorityLifecycle.released[0].derivedRemoved, true);
  assert.equal(providerFailed.cacheEvidence.derivedSidecarLifecycle.afterCells[0].providerExecution, "failed");
  assert.equal(
    providerFailed.cacheEvidence.derivedSidecarLifecycle.afterCells[0].requestMemoryStrategy,
    null,
  );
  const forgedCompletedProvider = structuredClone(providerFailureReceipt);
  forgedCompletedProvider.authorityLifecycle.providerExecution = "completed";
  forgedCompletedProvider.authorityLifecycle.requestMemoryStrategy = {
    strategy: "default-resident",
    requestMemoryPresent: false,
    stageResidency: false,
    streamTransformerBlocks: false,
  };
  assert.throws(
    () => validateReceipt(
      forgedCompletedProvider,
      providerFailed.checked.cells[7],
      providerFailed.checked,
      providerFailed.lifecycleContext,
    ),
    /FLUX derived lifecycle/,
  );
  const missingCacheProviderState = structuredClone(providerFailed.cacheEvidence);
  delete missingCacheProviderState.derivedSidecarLifecycle.afterCells[0].providerExecution;
  assert.throws(
    () => validateDocumentWithSchema(CACHE_PREFLIGHT_SCHEMA_PATH, missingCacheProviderState),
    /required property 'providerExecution'/,
  );
  const missingCacheDisposition = structuredClone(providerFailed.cacheEvidence);
  delete missingCacheDisposition.derivedSidecarLifecycle.afterCells[0].derivedDisposition;
  assert.throws(
    () => validateDocumentWithSchema(CACHE_PREFLIGHT_SCHEMA_PATH, missingCacheDisposition),
    /required property 'derivedDisposition'/,
  );
  const missingDisposition = structuredClone(providerFailureReceipt);
  delete missingDisposition.authorityLifecycle.derivedAfter[0].derivedDisposition;
  assert.throws(
    () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, missingDisposition),
    /required property 'derivedDisposition'/,
  );
  const wrongDisposition = structuredClone(providerFailureReceipt);
  wrongDisposition.authorityLifecycle.derivedAfter[0].derivedDisposition = "resident-empty";
  assert.throws(
    () => validateReceipt(
      wrongDisposition,
      providerFailed.checked.cells[7],
      providerFailed.checked,
      providerFailed.lifecycleContext,
    ),
    /FLUX derived lifecycle/,
  );

  const providerPartial = await scenario({
    providerRuntimeFailureAt: "flux1-dev-q8",
    providerFailureDerivedMode: "partial",
  });
  assert.equal(providerPartial.runtimeCells.length, 1);
  assert.match(
    providerPartial.result.summary.campaignErrors.join("\n"),
    /derived Candle sidecar evidence failed after cell flux1-dev-q8.*derived-sidecar contract drifted/,
  );
  assert.ok(providerPartial.result.summary.receipts.slice(8).every(
    (outcome) => outcome.status === "failed" && outcome.receipt,
  ));

  for (const cellFault of [
    { stage: "finalLog" },
    { stage: "evidenceHash" },
    { stage: "semanticValidation", occurrence: 2 },
    { stage: "schemaValidation", occurrence: 2 },
    { stage: "receiptWrite", occurrence: 2 },
    { stage: "receiptStat", occurrence: 2 },
    { stage: "receiptHash", occurrence: 2 },
  ]) {
    const faulted = await scenario({ cellFault });
    assert.equal(faulted.runtimeCells.length, 1, cellFault.stage);
    assert.equal(faulted.result.summary.receipts.length, 19, cellFault.stage);
    assert.ok(faulted.result.summary.receipts.slice(8).every(
      (outcome) => outcome.status === "failed" && outcome.receipt,
    ), cellFault.stage);
    assert.match(
      faulted.result.summary.campaignErrors.join("\n"),
      /primary receipt.*finalization failed/,
      cellFault.stage,
    );
  }

  const passed = await scenario();
  assert.deepEqual(
    passed.events.filter((event) => event.startsWith("audit:")),
    remainingIds.map((id) => `audit:${id}`),
  );
  assert.deepEqual(
    passed.events.filter((event) => event.startsWith("stage:")),
    remainingIds.map((id) => `stage:${id}`),
    passed.result.summary.campaignErrors.join("\n"),
  );
  assert.equal(passed.events.filter((event) => event.startsWith("offline:")).length, 16);
  assert.equal(passed.runtimeCells.length, 12, passed.result.summary.campaignErrors.join("\n"));
  assert.equal(
    passed.continuationReceipts[0].authorityLifecycle.derivedAfter[0].derivedDisposition,
    "resident-empty",
  );
  assert.equal(
    passed.cacheEvidence.derivedSidecarLifecycle.afterCells[0].derivedDisposition,
    "resident-empty",
  );
  const missingCacheRequestMemory = structuredClone(passed.cacheEvidence);
  delete missingCacheRequestMemory.derivedSidecarLifecycle.afterCells[0].requestMemoryStrategy;
  assert.throws(
    () => validateDocumentWithSchema(CACHE_PREFLIGHT_SCHEMA_PATH, missingCacheRequestMemory),
    /required property 'requestMemoryStrategy'/,
  );
  assert.ok(
    passed.events.indexOf("execute:flux1-dev-q8")
      < passed.events.indexOf("stage:flux1-schnell-q4"),
    "later authorities must not be staged all at once before the first GPU cell",
  );
  const scailRoots = passed.runtimeCells
    .filter((cell) => cell.artifacts.some((artifact) => artifact.id === "scail2-q4"))
    .map((cell) => cell.artifacts.find((artifact) => artifact.id === "scail2-q4").root);
  assert.equal(new Set(scailRoots).size, 1, "repeated SCAIL q4 authority must be reused");
  const controlRoots = passed.runtimeCells
    .filter((cell) => cell.kind === "sdxlOpenPose")
    .map((cell) => cell.artifacts.find((artifact) => artifact.id === "sdxl-openpose").root);
  assert.equal(controlRoots.length, 5);
  assert.equal(new Set(controlRoots).size, 1, "five SDXL cells must reuse one cached ControlNet");
  const lifecycle = passed.result.summary.authorityLifecycle;
  assert.deepEqual(lifecycle.find((row) => row.ordinal === 11).staged.map((row) => row.artifactId), [
    "scail2-q4",
  ]);
  assert.deepEqual(lifecycle.find((row) => row.ordinal === 12).released.map((row) => row.artifactId), [
    "scail2-q4",
  ]);
  assert.deepEqual(lifecycle.find((row) => row.ordinal === 14).staged.map((row) => row.artifactId), [
    "ltx23-q8", "ltx23-gemma",
  ]);
  assert.deepEqual(lifecycle.find((row) => row.ordinal === 15).staged.map((row) => row.artifactId), [
    "sdxl-base-q4", "sdxl-openpose", "sdxl-tokenizer-l", "sdxl-tokenizer-bigg", "sdxl-vae-fix",
  ]);
  assert.deepEqual(lifecycle.find((row) => row.ordinal === 19).released.map((row) => row.artifactId), [
    "illustrious-v2-q4", "sdxl-openpose", "sdxl-tokenizer-l", "sdxl-tokenizer-bigg", "sdxl-vae-fix",
  ]);
  assert.equal(passed.result.summary.finalAuthorityLifecycle.stage.files, 0);
  assert.equal(passed.result.summary.finalAuthorityLifecycle.derived.files, 0);
  assert.equal(passed.result.summary.finalAuthorityLifecycle.missingStoreAbsent, true);

  const sparse = await scenario({ sparse: true });
  assert.deepEqual(sparse.runtimeCells.map(({ id }) => id), [
    "ltx-2-3-q8", "illustrious-v1-openpose", "illustrious-v2-openpose",
  ]);
  assert.deepEqual(sparse.cacheEvidence.diskPlan.cells.map(({ ordinal }) => ordinal), [14, 18, 19]);
  assert.deepEqual(sparse.result.summary.lineage.continuation.executionOrdinals, [14, 18, 19]);
  assert.equal(sparse.result.summary.receipts.length, 19);
  assert.deepEqual(
    sparse.result.summary.receipts.map(({ id }) => id),
    profile().cells.map(({ id }) => id),
  );
  assert.deepEqual(
    sparse.result.summary.authorityLifecycle.find((row) => row.ordinal === 18)
      .staged.map(({ artifactId }) => artifactId),
    [
      "illustrious-v1-q4", "sdxl-openpose", "sdxl-tokenizer-l",
      "sdxl-tokenizer-bigg", "sdxl-vae-fix",
    ],
  );
  assert.deepEqual(
    sparse.result.summary.authorityLifecycle.find((row) => row.ordinal === 19)
      .released.map(({ artifactId }) => artifactId),
    [
      "illustrious-v2-q4", "sdxl-openpose", "sdxl-tokenizer-l",
      "sdxl-tokenizer-bigg", "sdxl-vae-fix",
    ],
  );

  const residentReceipt = passed.continuationReceipts[0];
  assert.equal(
    residentReceipt.authorityLifecycle.derivedAfter[0].derivedDisposition,
    "resident-empty",
  );
  assert.equal(residentReceipt.authorityLifecycle.derivedAfter[0].files, 0);
  assert.equal(residentReceipt.authorityLifecycle.derivedAfter[0].inventories.length, 0);
  assert.equal(residentReceipt.authorityLifecycle.staged[0].derivedNamespaces.length, 0);
  assert.equal(residentReceipt.authorityLifecycle.released[0].derivedRemoved, true);

  const bounded = await scenario({
    successfulFluxDerivedMode: "bounded", runtimeMemoryMode: "bounded",
  });
  assert.equal(bounded.runtimeCells.length, 12, bounded.result.summary.campaignErrors.join("\n"));
  assert.equal(bounded.result.summary.failed, 0);
  assert.equal(bounded.result.summary.passed, 19);
  assert.equal(bounded.runtimeCells[1].id, "flux1-schnell-q4");
  assert.equal(
    bounded.continuationReceipts[0].authorityLifecycle.derivedAfter[0].derivedDisposition,
    "bounded-transformer-sidecars",
  );
  assert.equal(
    bounded.cacheEvidence.derivedSidecarLifecycle.afterCells[0].derivedDisposition,
    "bounded-transformer-sidecars",
  );

  const boundedReceiptWithoutStaging = structuredClone(bounded.continuationReceipts[0]);
  boundedReceiptWithoutStaging.authorityLifecycle.requestMemoryStrategy.stageResidency = false;
  assert.throws(
    () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, boundedReceiptWithoutStaging),
    /must be equal to constant/,
  );
  assert.throws(
    () => validateReceipt(
      boundedReceiptWithoutStaging,
      bounded.checked.cells[7],
      bounded.checked,
      bounded.lifecycleContext,
    ),
    /request memory strategy/,
  );

  const boundedCacheWithoutStaging = structuredClone(bounded.cacheEvidence);
  boundedCacheWithoutStaging.derivedSidecarLifecycle.afterCells[0]
    .requestMemoryStrategy.stageResidency = false;
  assert.throws(
    () => validateDocumentWithSchema(CACHE_PREFLIGHT_SCHEMA_PATH, boundedCacheWithoutStaging),
    /must be equal to constant/,
  );
  assert.throws(
    () => validateCachePreflightEvidence(
      boundedCacheWithoutStaging, bounded.cacheValidation,
    ),
    /request memory strategy|must be equal to constant/,
  );

  const residentReceiptWithBoundedInventory = structuredClone(residentReceipt);
  residentReceiptWithBoundedInventory.authorityLifecycle.staged[0].derivedNamespaces =
    structuredClone(bounded.continuationReceipts[0].authorityLifecycle.staged[0].derivedNamespaces);
  residentReceiptWithBoundedInventory.authorityLifecycle.derivedAfter =
    structuredClone(bounded.continuationReceipts[0].authorityLifecycle.derivedAfter);
  assert.throws(
    () => validateReceipt(
      residentReceiptWithBoundedInventory,
      passed.checked.cells[7],
      passed.checked,
      passed.lifecycleContext,
    ),
    /neither exact resident-empty nor one exact bounded/,
  );
  const boundedReceiptWithEmptyInventory = structuredClone(bounded.continuationReceipts[0]);
  boundedReceiptWithEmptyInventory.authorityLifecycle.staged[0].derivedNamespaces = [];
  boundedReceiptWithEmptyInventory.authorityLifecycle.derivedAfter =
    structuredClone(residentReceipt.authorityLifecycle.derivedAfter);
  assert.throws(
    () => validateReceipt(
      boundedReceiptWithEmptyInventory,
      bounded.checked.cells[7],
      bounded.checked,
      bounded.lifecycleContext,
    ),
    /neither exact resident-empty nor one exact bounded/,
  );

  const residentCacheWithBoundedInventory = structuredClone(passed.cacheEvidence);
  Object.assign(residentCacheWithBoundedInventory.derivedSidecarLifecycle.afterCells[0], {
    derivedDisposition: "bounded-transformer-sidecars",
    inventory: {
      ...residentCacheWithBoundedInventory.derivedSidecarLifecycle.afterCells[0].inventory,
      files: 494,
      bytes: 12_573_868_032,
      sha256: "8".repeat(64),
    },
  });
  assert.throws(
    () => validateCachePreflightEvidence(
      residentCacheWithBoundedInventory, passed.cacheValidation,
    ),
    /derived Candle sidecar lifecycle inventory drifted/,
  );
  const boundedCacheWithEmptyInventory = structuredClone(bounded.cacheEvidence);
  Object.assign(boundedCacheWithEmptyInventory.derivedSidecarLifecycle.afterCells[0], {
    derivedDisposition: "resident-empty",
    inventory: {
      ...boundedCacheWithEmptyInventory.derivedSidecarLifecycle.afterCells[0].inventory,
      files: 0,
      bytes: 0,
      sha256: createHash("sha256").digest("hex"),
    },
  });
  assert.throws(
    () => validateCachePreflightEvidence(
      boundedCacheWithEmptyInventory, bounded.cacheValidation,
    ),
    /derived Candle sidecar lifecycle inventory drifted/,
  );

  const stagedResident = await scenario({ runtimeMemoryMode: "staged" });
  assert.equal(stagedResident.runtimeCells.length, 12);
  assert.equal(
    stagedResident.continuationReceipts[0].authorityLifecycle.requestMemoryStrategy.strategy,
    "staged-resident",
  );
  assert.equal(
    stagedResident.continuationReceipts[0].authorityLifecycle.derivedAfter[0].derivedDisposition,
    "resident-empty",
  );

  const filled = await scenario({ missingId: "flux1-schnell-q8", downloadSucceeds: true });
  assert.deepEqual(
    filled.events.filter((event) => event.startsWith("audit:")),
    remainingIds.map((id) => `audit:${id}`),
  );
  assert.deepEqual(filled.events.filter((event) => event.startsWith("download:")), [
    "download:flux1-schnell-q8",
  ]);
  assert.equal(filled.events.filter((event) => event.startsWith("stage:")).length, 16);
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
  for (const [label, mutate, pattern] of [
    ["staging", (value) => value.phases.staging.pop(), /staged\/offline authority transition/],
    ["offline", (value) => value.phases.finalOffline.pop(), /staged\/offline authority transition/],
    ["afterCells", (value) => value.derivedSidecarLifecycle.afterCells.pop(), /per-cell derived lifecycle/],
    ["nonModelPaths", (value) => { value.diskPlan.nonModelPaths[0].path += "-drift"; }, /non-model paths/],
  ]) {
    const drifted = structuredClone(filled.cacheEvidence);
    mutate(drifted);
    assert.throws(
      () => validateCachePreflightEvidence(drifted, filled.cacheValidation),
      pattern,
      label,
    );
  }
  assert.equal(filled.cacheEvidence.lifetimePlan.length, 16);
  assert.equal(filled.cacheEvidence.diskPlan.admitted, true);
  const downloadDrift = structuredClone(filled.cacheEvidence);
  downloadDrift.downloadedFiles[0].commitSha = "f".repeat(40);
  assert.throws(
    () => validateCachePreflightEvidence(downloadDrift, filled.cacheValidation),
    /partition drifted|unreviewed model download/,
  );

  const firstReceipt = filled.continuationReceipts[0];
  const firstCell = filled.checked.cells[7];
  assert.doesNotThrow(() => validateReceipt(
    firstReceipt, firstCell, filled.checked, filled.lifecycleContext,
  ));
  const missingLifecycle = structuredClone(firstReceipt);
  delete missingLifecycle.authorityLifecycle;
  assert.throws(
    () => validateReceipt(missingLifecycle, firstCell, filled.checked, filled.lifecycleContext),
    /missing required authority lifecycle/,
  );
  assert.throws(
    () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, missingLifecycle),
    /required property 'authorityLifecycle'/,
  );
  const forgedLegacyLifecycleOmission = structuredClone(firstReceipt);
  forgedLegacyLifecycleOmission.repositories.sceneworks.sha = "8886a9e69f26beec05688c81b414859bd102f6d0";
  forgedLegacyLifecycleOmission.execution.headSha = "8886a9e69f26beec05688c81b414859bd102f6d0";
  delete forgedLegacyLifecycleOmission.authorityLifecycle;
  assert.throws(
    () => validateReceipt(
      forgedLegacyLifecycleOmission, firstCell, filled.checked, filled.lifecycleContext,
    ),
    /missing required authority lifecycle/,
  );
  assert.throws(
    () => validateReceipt(
      forgedLegacyLifecycleOmission, firstCell, filled.checked, filled.lifecycleContext,
      { trustedLegacyImport: true },
    ),
    /missing required authority lifecycle/,
  );
  assert.throws(
    () => validateDocumentWithSchema(RECEIPT_SCHEMA_PATH, forgedLegacyLifecycleOmission),
    /required property 'authorityLifecycle'/,
  );
  for (const [label, mutate] of [
    ["phase", (value) => { value.authorityLifecycle.diskProbes[0].phase = "before-stage:wrong"; }],
    ["ordinal", (value) => { value.authorityLifecycle.diskProbes[0].ordinal += 1; }],
    ["root", (value) => { value.authorityLifecycle.diskProbes[0].root += "-wrong"; }],
    ["floor", (value) => { value.authorityLifecycle.diskProbes[0].requiredFreeBytes -= 1; }],
    ["free", (value) => { value.authorityLifecycle.diskProbes[0].freeBytes = 1; }],
    ["transition", (value) => { value.authorityLifecycle.staged.pop(); }],
    ["derived", (value) => { value.authorityLifecycle.derivedAfter.pop(); }],
  ]) {
    const drifted = structuredClone(firstReceipt);
    mutate(drifted);
    assert.throws(
      () => validateReceipt(drifted, firstCell, filled.checked, filled.lifecycleContext),
      /disk probes drifted|completed receipt lifecycle omitted|FLUX derived lifecycle/,
      label,
    );
  }

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
    assert.equal(faulted.cacheEvidence.offlineBeforeCells, stage === "preflightMkdir" ? false : true);
  }
  const semanticFault = await scenario({ omitObstructions: true });
  assert.equal(semanticFault.runtimeCells.length, 0);
  assert.equal(semanticFault.result.summary.passed, 7);
  assert.equal(semanticFault.result.summary.failed, 12);
  assert.match(
    semanticFault.result.summary.campaignErrors.join("\n"),
    /did not obstruct model-adjacent sidecars/,
  );
  assert.doesNotThrow(() => validateDocumentWithSchema(
    CACHE_PREFLIGHT_SCHEMA_PATH,
    semanticFault.cacheEvidence,
  ));

  for (const drift of [
    { files: 493, bytes: 12_573_868_032 },
    { files: 494, bytes: 12_573_868_032 + 494 * 16_384 + 1 },
  ]) {
    const derivedFault = await scenario({
      derivedContractDrift: drift,
      successfulFluxDerivedMode: "bounded",
      runtimeMemoryMode: "bounded",
    });
    assert.equal(derivedFault.runtimeCells.length, 1);
    assert.match(
      derivedFault.result.summary.campaignErrors.join("\n"),
      /derived Candle sidecar evidence failed.*FLUX derived-sidecar contract drifted/,
    );
    assert.ok(derivedFault.result.summary.receipts.slice(8).every(
      (outcome) => outcome.status === "failed" && outcome.receipt,
    ));
  }
  const strayDerived = await scenario({ successfulFluxDerivedMode: "stray" });
  assert.equal(strayDerived.runtimeCells.length, 1);
  assert.match(
    strayDerived.result.summary.campaignErrors.join("\n"),
    /derived Candle sidecar evidence failed.*stray files/,
  );
  assert.ok(strayDerived.result.summary.receipts.slice(8).every(
    (outcome) => outcome.status === "failed" && outcome.receipt,
  ));
  for (const [label, options] of [
    ["resident-with-sidecars", { successfulFluxDerivedMode: "bounded" }],
    ["bounded-with-empty", { runtimeMemoryMode: "bounded" }],
  ]) {
    const mismatch = await scenario(options);
    assert.equal(mismatch.runtimeCells.length, 1, label);
    assert.match(
      mismatch.result.summary.campaignErrors.join("\n"),
      /derived Candle sidecar evidence failed.*FLUX derived-sidecar contract drifted/,
      label,
    );
    assert.ok(mismatch.result.summary.receipts.slice(8).every(
      (outcome) => outcome.status === "failed" && outcome.receipt,
    ), label);
  }
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
  assert.match(terminalBlock, /actions\/artifacts\/\$artifactId/);
  assert.match(terminalBlock, /9492288293[\s\S]*?15452320[\s\S]*?dbae4c7d67d824bb8568909231614c6bcc268868087eb19974ce013bfc557724/);
  assert.match(terminalBlock, /\$artifact\.expired[\s\S]*?\$artifact\.size_in_bytes[\s\S]*?\$artifact\.digest/);
  assert.match(terminalBlock, /gh run download \$runId --name \$artifactName --dir \$evidence/);
  assert.match(terminalBlock, /43c718b7e9a852bd5029448d18841fed0f508c3a/);
  assert.doesNotMatch(terminalBlock, /snapshot_download/);
  assert.match(terminalBlock, /python -m venv \$venv/);
  assert.match(terminalBlock, /pip install[^\n]*'huggingface_hub==0\.36\.0'/);
  assert.match(terminalBlock, /Scripts\\python\.exe/);
  assert.match(terminalBlock, /SCENEWORKS_TERMINAL_PYTHON=\$python/);
  assert.match(terminalBlock, /from huggingface_hub import __version__/);
  assert.match(terminalBlock, /__version__ == '0\.36\.0'/);
  assert.doesNotMatch(terminalBlock, /q8\/transformer\/model\.safetensors/);
  const controller = await readFile("scripts/epic-20738-terminal-cuda-harness.mjs", "utf8");
  assert.match(controller, /SPARSE_EXECUTION_ORDINALS = \[14, 18, 19\][\s\S]*?SPARSE_REMAINING_AUTHORITIES = 8[\s\S]*?SPARSE_REMAINING_FILES = 70/);
  assert.match(controller, /stale cell-10 sentinel/);
  assert.match(controller, /selectRecoveryContinuation[\s\S]*?importRecoveryContinuation/);
  assert.match(controller, /selectSparseRecovery[\s\S]*?importSparseRecovery/);
  assert.doesNotMatch(controller, /JSON\.stringify\(staged\.downloadedFiles/);
  assert.match(controller, /async function downloadReviewedMissing[\s\S]*?HF_HUB_OFFLINE: "0"/);
  assert.match(controller, /async function stageArtifact[\s\S]*?HF_HUB_OFFLINE: "1"/);
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
