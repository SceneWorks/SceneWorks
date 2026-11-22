import assert from "node:assert/strict";
import { lstat, readFile, realpath, writeFile } from "node:fs/promises";
import path from "node:path";

import { sha256 } from "./lib/file-sha256.mjs";

export const KREA_LINKED_SOURCE_REPAIR_REVISION = "5b2c605c9371f7ecdf11dcbd6e76ac5e3d52d957";
export const KREA_LINKED_SOURCE_REPAIR_CHECKOUT = "tier2-ab-krea-repair";
const WINDOWS_RAW_ROUTING_FILE = "crates/sceneworks-core/src/jobs_store/routing/candle.rs";
const WINDOWS_RAW_ROUTING_BASE_SHA256 = "dabba99e040d63659cec71cb84067431cc2bd2f1c1f89a018eea750ad7e038c7";
const WINDOWS_RAW_ROUTING_REPAIR_SHA256 = "596a7690cf9f166fbcecef2e1b34bb223838069081140c232683dfbcc8225a4d";
const SOURCES = [
  ["apps/rust-api/src/portable_admission.rs",
    "41409a5904443d20b710e1d17c4f5f1ef8851b67777d9f9a032820db8cd4f9a5",
    "1f96bafdc500cb9c938d176345389a4938b859efd9e7b4d8309d4593c1e06b52"],
  ["apps/rust-api/src/portable_routes.rs",
    "6500291202277bda57baad01b7b84cada9df97c59ff02a52a81b308e6f17e5f0",
    "62405de585a204660319e50130baaf3aa57904131d15c949271c7aeb8e9307d8"],
  ["crates/sceneworks-core/src/model_artifacts/artifact_selection.rs",
    "5009f47916fedb508365da42b47fa9e702ea3f89e9ec1e0228fcaf020465ab0e",
    "77e579b5d86ec1764130bcddb8de2b39b0d7f9b384cca9a0924bd3fa28784908"],
  ["crates/sceneworks-worker/src/external_library_runtime.rs",
    "2e1a729ab50ed7961b0eb330fc0cdeeefd5ed4b4f6e9e089fee870ef19ffd7d7",
    "9a84528d02e654b94cc39ca8c962a8d157398069282aa40d9e78d7364d754bb0"],
];
const REPAIR_SHA256 = SOURCES.map(([, , repair]) => repair);
const REVIEWED_CANDIDATES = [
  [KREA_LINKED_SOURCE_REPAIR_REVISION, REPAIR_SHA256],
  ["e98dea13f401500998947e4359ebd56fae113bf9", [
    "9d9191f521d43c3f0850e4d80f305c187b97a3119f3e0681044670a7e61a1676",
    REPAIR_SHA256[1], REPAIR_SHA256[2],
    "be3b26400a7babc34ce1bd0f6fabf433cf11d3af62d41766b6439e3a72852d5d",
  ]],
];

async function readSource(root, file) {
  const location = path.join(root, file);
  const stat = await lstat(location);
  assert.ok(stat.isFile() && stat.nlink === 1 && await realpath(location) === location,
    `Krea linked-source build requires an ordinary checkout file: ${file}`);
  return readFile(location);
}

function gitBytesSha256(bytes, label) {
  const canonical = bytes.toString("utf8").replaceAll("\r\n", "\n");
  assert.ok(bytes.equals(Buffer.from(canonical)) || bytes.equals(Buffer.from(canonical.replaceAll("\n", "\r\n"))),
    `invalid checkout line endings for ${label}`);
  return sha256(canonical);
}

export function reviewedKreaLinkedSourceCandidate(hashes) {
  const candidate = REVIEWED_CANDIDATES.find(([, approved]) =>
    approved.length === hashes.length && approved.every((hash, index) => hash === hashes[index]));
  assert.ok(candidate, "candidate Krea linked-source fingerprints do not match one reviewed source set");
  return candidate[0];
}

export async function bindKreaLinkedSourceBuild(baseDir, headDir, repairDir, { windowsRawRoutingRevision } = {}) {
  const [base, head, repair] = await Promise.all([realpath(baseDir), realpath(headDir), realpath(repairDir)]);
  assert.equal(new Set([base, head, repair]).size, 3, "Krea linked-source builds require distinct checkouts");
  const replacements = [];
  const files = [];
  for (const [file, baseGitSha256, repairGitSha256] of SOURCES) {
    const [original, candidate, repaired] = await Promise.all([readSource(base, file), readSource(head, file), readSource(repair, file)]);
    assert.equal(gitBytesSha256(original, `base/${file}`), baseGitSha256, `unexpected Krea linked-source bytes for base/${file}`);
    assert.equal(gitBytesSha256(repaired, `repair/${file}`), repairGitSha256, `unexpected Krea linked-source bytes for repair/${file}`);
    replacements.push({ file, original, repaired });
    files.push({ file, baseGitSha256, repairGitSha256, candidateGitSha256: gitBytesSha256(candidate, `head/${file}`),
      baseCheckoutSha256: sha256(original), baseBuildSha256: sha256(repaired), candidateCheckoutSha256: sha256(candidate) });
  }
  const candidateReviewRevision = reviewedKreaLinkedSourceCandidate(files.map(({ candidateGitSha256 }) => candidateGitSha256));
  const receipt = { repairRevision: KREA_LINKED_SOURCE_REPAIR_REVISION,
    candidateReviewRevision, appliesTo: "base build only", files };
  if (windowsRawRoutingRevision !== undefined) {
    assert.match(windowsRawRoutingRevision, /^[a-f0-9]{40}$/);
    const file = WINDOWS_RAW_ROUTING_FILE;
    const [original, repaired] = await Promise.all([readSource(base, file), readSource(head, file)]);
    const baseGitSha256 = gitBytesSha256(original, "base Windows Raw routing");
    const sharedGitSha256 = gitBytesSha256(repaired, "candidate Windows Raw routing");
    assert.equal(baseGitSha256, WINDOWS_RAW_ROUTING_BASE_SHA256, "unexpected frozen Windows Raw routing source");
    assert.equal(sharedGitSha256, WINDOWS_RAW_ROUTING_REPAIR_SHA256, "candidate Windows Raw routing source is unreviewed");
    replacements.push({ file, original, repaired });
    receipt.windowsRawRouting = { file, repairRevision: windowsRawRoutingRevision, baseGitSha256, sharedGitSha256,
      baseCheckoutSha256: sha256(original), sharedBuildSha256: sha256(repaired) };
  }
  return { base, repair, replacements, receipt };
}

// Only the frozen proof checkout changes. Restore every attempted write before publishing a binary receipt.
export async function withKreaLinkedSourceBuild({ base, replacements }, build) {
  const touched = [];
  try {
    for (const { file, original, repaired } of replacements) {
      assert.ok((await readSource(base, file)).equals(original), `Krea baseline source changed before build: ${file}`);
      touched.push({ file, original });
      await writeFile(path.join(base, file), repaired);
      assert.ok((await readSource(base, file)).equals(repaired), `Krea baseline source repair did not match: ${file}`);
    }
    return await build();
  } finally {
    const restored = await Promise.allSettled(touched.map(async ({ file, original }) => {
      await readSource(base, file);
      await writeFile(path.join(base, file), original);
      assert.ok((await readSource(base, file)).equals(original), `Krea baseline source was not restored: ${file}`);
    }));
    const errors = restored.filter((result) => result.status === "rejected").map((result) => result.reason);
    if (errors.length) throw new AggregateError(errors, "could not restore the frozen Krea baseline checkout");
  }
}
