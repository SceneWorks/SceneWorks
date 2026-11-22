import assert from "node:assert/strict";
import { lstat, readFile, realpath, writeFile } from "node:fs/promises";
import path from "node:path";

import { sha256 } from "./lib/file-sha256.mjs";

const REPAIR_REVISION = "5b2c605c9371f7ecdf11dcbd6e76ac5e3d52d957";
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

async function readSource(root, file) {
  const location = path.join(root, file);
  const stat = await lstat(location);
  assert.ok(stat.isFile() && stat.nlink === 1 && await realpath(location) === location,
    `Krea linked-source build requires an ordinary checkout file: ${file}`);
  return readFile(location);
}

function assertGitBytes(bytes, expected, label) {
  const canonical = bytes.toString("utf8").replaceAll("\r\n", "\n");
  assert.ok(bytes.equals(Buffer.from(canonical)) || bytes.equals(Buffer.from(canonical.replaceAll("\n", "\r\n"))),
    `invalid checkout line endings for ${label}`);
  assert.equal(sha256(canonical), expected, `unexpected Krea linked-source bytes for ${label}`);
}

export async function bindKreaLinkedSourceBuild(baseDir, headDir) {
  const [base, head] = await Promise.all([realpath(baseDir), realpath(headDir)]);
  assert.notEqual(base, head, "Krea linked-source builds require distinct checkouts");
  const replacements = [];
  const files = [];
  for (const [file, baseGitSha256, repairGitSha256] of SOURCES) {
    const [original, repaired] = await Promise.all([readSource(base, file), readSource(head, file)]);
    assertGitBytes(original, baseGitSha256, `base/${file}`);
    assertGitBytes(repaired, repairGitSha256, `head/${file}`);
    replacements.push({ file, original, repaired });
    files.push({ file, baseGitSha256, repairGitSha256, baseCheckoutSha256: sha256(original), pairedBuildSha256: sha256(repaired) });
  }
  return { base, replacements, receipt: { repairRevision: REPAIR_REVISION, appliesTo: "base build only", files } };
}

// The candidate already contains the reviewed bytes. Only the frozen proof checkout changes,
// and every attempted write is restored before the build can publish a binary receipt.
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
