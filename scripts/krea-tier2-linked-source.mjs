import assert from "node:assert/strict";
import { lstat, readFile, realpath, writeFile } from "node:fs/promises";
import { userInfo } from "node:os";
import path from "node:path";

import { sha256 } from "./lib/file-sha256.mjs";
import { assertTerminalPhysicalContainment } from "./lib/starvector-terminal-pin-paths.mjs";

export const KREA_LINKED_SOURCE_REPAIR_REVISION = "5b2c605c9371f7ecdf11dcbd6e76ac5e3d52d957";
export const KREA_LINKED_SOURCE_REPAIR_CHECKOUT = "tier2-ab-krea-repair";
const WINDOWS_RAW_ROUTING_FILE = "crates/sceneworks-core/src/jobs_store/routing/candle.rs";
const WINDOWS_RAW_ROUTING_BASE_SHA256 = "dabba99e040d63659cec71cb84067431cc2bd2f1c1f89a018eea750ad7e038c7";
const WINDOWS_RAW_ROUTING_REPAIR_SHA256 = "596a7690cf9f166fbcecef2e1b34bb223838069081140c232683dfbcc8225a4d";
const KREA_APPLE_RAW_PACKAGE = "mlx-gen-krea";
const KREA_APPLE_RAW_MANIFEST = "crates/media/mlx-gen/mlx-gen-krea/Cargo.toml";
export const KREA_APPLE_RAW_PROJECTOR = Object.freeze({
  inferenceRevision: "e86b8d5fc493c16b2d9d852e0c9ecbdefee58294",
  file: "crates/media/mlx-gen/mlx-gen-krea/src/transformer/mod.rs",
  originalSha256: "e6dcb5b38cb0011e5550a049edbc4703a94a4765af414d6f09ce623661af7344",
  buildSha256: "ef16f9e2d7462a95d7c2e796c552aaa8c91a12980a29dedb3d70456b935c30d5",
  mismatchRunId: "45051714564",
  appliesTo: "Apple baseline Raw phase adapter clear",
});
const PROJECTOR_ANCHOR = "        self.txt_in_l2.set_adapters(Vec::new());\n        self.final_linear.set_adapters(Vec::new());\n";
const PROJECTOR_CLEAR = `        self.adaptable_mut(&["text_fusion", "projector"])
            .expect("Krea text-fusion projector is an adapter target")
            .clear_adapters();
`;
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
  ["468000983b90f3ffe0004777d3adfeda3679ce01", [
    "9d9191f521d43c3f0850e4d80f305c187b97a3119f3e0681044670a7e61a1676",
    REPAIR_SHA256[1], REPAIR_SHA256[2],
    "bd92ec4aae54f89581a06dad9ef3072f7ef278821038164097731fa0c766d2c9",
  ]],
  // Shared cache loading changes only the runtime guard in this source set. Publication still
  // verifies and freezes snapshots; this records source review, not a new physical proof.
  ["14025270f2d790b7980ddbb6360a6e168dc51ccf", [
    "9d9191f521d43c3f0850e4d80f305c187b97a3119f3e0681044670a7e61a1676",
    REPAIR_SHA256[1], REPAIR_SHA256[2],
    "2b79936713a3157a71abc4e79cf02f7151bcda26ecb70a3c49bf9213347bd26f",
  ]],
  // Portable admission's proof waits read only the claim-owning worker ids, not every live
  // worker's parsed claim JSON (`portable_source_claim_worker_ids_for_job_route`). The same
  // workers are asked and answer, so this records source review, not a new physical proof.
  ["b5c8ff577aa4ad465e097063b41e1923f8ac3e63", [
    "d7ae4ed9ec1b5add926c7149991b93b60af806f728486c96a8bdc4d8b4d147b2",
    REPAIR_SHA256[1], REPAIR_SHA256[2],
    "2b79936713a3157a71abc4e79cf02f7151bcda26ecb70a3c49bf9213347bd26f",
  ]],
  // Portable admission shares each worker's parsed claim set across one admission instead of
  // re-parsing it per proof read (cache keyed by the trigger-maintained claim version). The
  // same claims decide the same route, so this records source review, not a new physical proof.
  ["a6f74c142832e388bee92bb50496cd088c7b7809", [
    "0422af9cd9e0f23fe2ab41228f44b5085436da7598ce46dd6ca467f56bb72395",
    REPAIR_SHA256[1], REPAIR_SHA256[2],
    "2b79936713a3157a71abc4e79cf02f7151bcda26ecb70a3c49bf9213347bd26f",
  ]],
  // SW-7300 removes the Mage test case from portable routes and the Mage resident-base row.
  // The Krea companion assertions and resident family, model, tier, and shape are the same.
  ["310b9b6d2a43d9389d8e3267495dd97bb47d294d", [
    "0422af9cd9e0f23fe2ab41228f44b5085436da7598ce46dd6ca467f56bb72395",
    "beda363a2e2f974bd07dd2853b9365254187508306c48d86d65b450be0fe5619",
    "97b1c43b7486c202e8b9c713380e765ae8e07148dfddcdf14f9f277747b11072",
    "2b79936713a3157a71abc4e79cf02f7151bcda26ecb70a3c49bf9213347bd26f",
  ]],
  // The runtime guard serves verified local sources (imported checkpoints, trained adapters,
  // claimed components, plan layers) from their originals instead of a per-route copy, keys any
  // remaining bundle by its copy layout, and binds saved quantizations to the admitted local
  // bytes. The same frozen ledger is verified through the same hash-once store before any loader
  // sees a path, so this records source review, not a new physical proof.
  ["6ffb4ef8404571ca511f4d8681c239aa56a3456e", [
    "0422af9cd9e0f23fe2ab41228f44b5085436da7598ce46dd6ca467f56bb72395",
    REPAIR_SHA256[1], REPAIR_SHA256[2],
    "a5ce0fc14c82e819d6f7394e7567b2f764e835854cbc9e7af69c1835af7df047",
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

export function appleRawProjectorReceipt(routes) {
  return routes.some(({ backend, proof }) => backend === "mlx" && proof === "raw") ? KREA_APPLE_RAW_PROJECTOR : undefined;
}

export async function appleRawBaselineCargoEnv(env, accountHome = userInfo().homedir) {
  // Git's HTTP transport can read .netrc independently of its credential helper. Keep HOME for
  // rustup and native builds, but do not run the fresh dependency fetch with that credential file.
  for (const home of new Set([env.HOME, accountHome].filter(Boolean))) {
    assert.ok(path.isAbsolute(home), "Apple Krea baseline requires an absolute home directory");
    const netrc = await lstat(path.join(home, ".netrc")).catch((error) => {
      if (error.code !== "ENOENT") throw error;
      return null;
    });
    assert.ok(netrc === null, "Apple Krea baseline Cargo fetch requires no ambient .netrc");
  }
  const clean = Object.fromEntries(Object.entries(env).filter(([key]) => !/^(?:GIT_|SSH_|GCM_|NETRC$|NETRC_FILE$)/i.test(key)));
  return {
    ...clean,
    CARGO_NET_GIT_FETCH_WITH_CLI: "true",
    GIT_CONFIG_NOSYSTEM: "1",
    GIT_CONFIG_GLOBAL: "/dev/null",
    // An empty helper clears the list even if a lower-priority config registered one.
    GIT_CONFIG_COUNT: "2",
    GIT_CONFIG_KEY_0: "credential.helper",
    GIT_CONFIG_VALUE_0: "",
    GIT_CONFIG_KEY_1: "credential.interactive",
    GIT_CONFIG_VALUE_1: "false",
    GIT_TERMINAL_PROMPT: "0",
    GIT_ASKPASS: "/usr/bin/false",
    SSH_ASKPASS: "/usr/bin/false",
  };
}

export async function bindKreaAppleRawProjector(cargoHome, metadata, git) {
  const receipt = KREA_APPLE_RAW_PROJECTOR;
  const packages = metadata.packages.filter(({ name }) => name === KREA_APPLE_RAW_PACKAGE);
  const [provider] = packages;
  assert.ok(packages.length === 1 && provider.source ===
    `git+https://github.com/SceneWorks/inference?rev=${receipt.inferenceRevision}#${receipt.inferenceRevision}`,
  "Cargo did not resolve the exact frozen Apple Krea provider");
  const manifest = provider.manifest_path;
  const base = path.resolve(path.dirname(manifest), "../../../..");
  const checkouts = path.join(cargoHome, "git", "checkouts");
  const relative = path.relative(checkouts, base);
  assert.match(relative.split(path.sep).join("/"), /^inference-[a-f0-9]+\/[a-f0-9]+$/, "Krea projector repair must use the private Cargo checkout");
  assert.equal(manifest, path.join(base, KREA_APPLE_RAW_MANIFEST), "unexpected frozen Krea manifest path");
  await assertTerminalPhysicalContainment(cargoHome, path.dirname(manifest));
  assert.equal(await git(base, "rev-parse", "--show-toplevel"), base);
  assert.equal(await git(base, "rev-parse", "HEAD"), receipt.inferenceRevision);
  assert.equal(await git(base, "status", "--porcelain", "--untracked-files=no"), "", "frozen Cargo checkout changed before the Apple Krea repair");
  const original = await readSource(base, receipt.file);
  assert.equal(sha256(original), receipt.originalSha256, "unreviewed frozen Krea projector source");
  const pieces = original.toString("utf8").split(PROJECTOR_ANCHOR);
  assert.equal(pieces.length, 2, "frozen Krea adapter clear must have one exact anchor");
  const repaired = Buffer.from(pieces.join(PROJECTOR_ANCHOR + PROJECTOR_CLEAR));
  assert.equal(sha256(repaired), receipt.buildSha256, "unreviewed Apple Krea projector repair bytes");
  return { base, manifest, replacements: [{ file: receipt.file, original, repaired }], receipt };
}

export async function withKreaAppleRawProjector(binding, build, messages) {
  return withKreaLinkedSourceBuild(binding, async () => {
    const result = await build();
    if (result.code === 0) {
      const compiled = (await readFile(messages, "utf8")).split(/\r?\n/)
        .filter((line) => line.startsWith("{"))
        .map((line) => JSON.parse(line))
        .filter((message) => message.reason === "compiler-artifact" && message.target?.name === "mlx_gen_krea");
      assert.ok(compiled.length === 1 && compiled[0].manifest_path === binding.manifest && compiled[0].fresh === false,
        "Apple Krea baseline build did not compile its private repaired provider");
      assert.equal(sha256(await readSource(binding.base, binding.receipt.file)), binding.receipt.buildSha256,
        "frozen Apple Krea projector repair changed during the build");
    }
    return result;
  });
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
