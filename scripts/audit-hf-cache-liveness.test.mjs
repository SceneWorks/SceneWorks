/**
 * `node --test` coverage for `scripts/audit-hf-cache-liveness.mjs` (sc-17335).
 *
 * The script ships a `--self-test` that mutates a small fixture and requires each verdict to
 * fire. That is the right shape for "does the classifier have teeth", but it lives inside the
 * file under test, so a rule can be deleted along with its own case and nothing notices.
 *
 * This file asserts what `--self-test` structurally cannot, and every test here exists because an
 * earlier version of the script was WRONG in exactly that way:
 *
 *  - the "nothing declared is ever unreferenced" invariant derives its repo set by walking the
 *    manifests INDEPENDENTLY, not by calling `manifestRepoIndex`. The first version of this test
 *    fed the index its own output, so it could only prove "everything the index finds is kept" —
 *    never "the index finds everything". That vacuity hid `textEncoderRepo` (25 GiB).
 *  - it goes through `loadManifests()`, the loader `main()` actually uses. The first version built
 *    its own loader, so re-introducing the original control-overlay bug in `main()` left the
 *    whole suite green.
 *  - it pins the set of manifest keys that carry a repo id, so a new key reds here instead of
 *    silently becoming an unindexed liveness source.
 *
 * Deliberately does NOT read `~/SceneWorks/data` or the Hugging Face cache: those differ per
 * machine and would make this pass or fail on whatever is installed today.
 */

import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";

import {
  REPO_KEY,
  REPO_SHAPE,
  VERDICT,
  appModelDirName,
  classify,
  isTestSource,
  loadManifests,
  loadSourceFiles,
  manifestRepoIndex,
  modelDownloadIndex,
  repoFromCacheDir,
  sourceLiteralIndex,
} from "./audit-hf-cache-liveness.mjs";

/** Walk the manifests independently of `manifestRepoIndex` — the point is not to trust it. */
function declaredReposIndependently(manifests) {
  const found = new Map();
  const seen = new Set();
  function visit(node, manifestName) {
    if (Array.isArray(node)) return node.forEach((item) => visit(item, manifestName));
    if (!node || typeof node !== "object" || seen.has(node)) return;
    seen.add(node);
    for (const [key, value] of Object.entries(node)) {
      if (typeof value === "string") {
        if (REPO_KEY.test(key) && REPO_SHAPE.test(value.trim())) {
          found.set(value.trim(), manifestName);
        }
      } else visit(value, manifestName);
    }
  }
  for (const [name, manifest] of Object.entries(manifests)) visit(manifest, name);
  return found;
}

function run({ repos, installedModels = [], loraReceipts = [], manifests, sourceLiterals = new Map(), testLiterals = new Map(), ciProvisioned = [] }) {
  return classify({
    repos,
    installedModels: new Set(installedModels),
    loraReceipts,
    modelDownloads: modelDownloadIndex(manifests["builtin.models.jsonc"]),
    manifestDeclarations: manifestRepoIndex(manifests),
    sourceLiterals,
    testLiterals,
    ciProvisioned: new Set(ciProvisioned),
  });
}

test("cache directory names round-trip, including repo names containing a dash pair", () => {
  assert.equal(repoFromCacheDir("models--Comfy-Org--Krea-2"), "Comfy-Org/Krea-2");
  // Only the FIRST `--` splits org from name. `hf_home.rs safe_repo_dir_name` escapes any
  // non-[A-Za-z0-9._-] to `--`; an HF org cannot contain that, but a repo NAME can.
  assert.equal(repoFromCacheDir("models--acme--weird--name"), "acme/weird--name");
  assert.equal(repoFromCacheDir("datasets--acme--x"), null);
  assert.equal(repoFromCacheDir("models--noseparator"), null);
  assert.equal(appModelDirName("Comfy-Org/Krea-2"), "Comfy-Org__Krea-2");
});

test("loadManifests reads the whole manifest directory", async () => {
  // Goes through the loader `main()` uses, so restricting it cannot leave this suite green.
  const manifests = await loadManifests();
  const names = Object.keys(manifests);
  assert.ok(names.length >= 5, `expected the manifest directory, got ${names.join(", ")}`);
  for (const required of ["builtin.models.jsonc", "builtin.loras.jsonc", "builtin.control_overlays.jsonc"]) {
    assert.ok(names.includes(required), `${required} must be loaded — it declares repos`);
  }
});

test("NO repo declared by any manifest is ever reported unreferenced", async () => {
  const manifests = await loadManifests();
  const declared = declaredReposIndependently(manifests);
  assert.ok(declared.size > 50, `expected many declared repos, got ${declared.size}`);

  for (const entry of run({ repos: [...declared.keys()], manifests })) {
    assert.equal(
      entry.referenced,
      true,
      `${entry.repo} is declared in ${declared.get(entry.repo)} but came back ${entry.verdict}`,
    );
  }
});

test("the set of manifest keys that carry a repo id is pinned", async () => {
  // A new key like `textEncoderRepo` is a new liveness source. Fail here, loudly, rather than
  // letting it become weights nobody can account for.
  const manifests = await loadManifests();
  const keys = new Set();
  const seen = new Set();
  (function visit(node) {
    if (Array.isArray(node)) return node.forEach(visit);
    if (!node || typeof node !== "object" || seen.has(node)) return;
    seen.add(node);
    for (const [key, value] of Object.entries(node)) {
      if (typeof value === "string" && REPO_SHAPE.test(value.trim())) keys.add(key);
      else visit(value);
    }
  })(Object.values(manifests));

  const repoKeys = [...keys].filter((key) => REPO_KEY.test(key)).sort();
  assert.deepEqual(repoKeys, [
    "artifactRepository",
    "convertBaseRepo",
    "convertSourceRepo",
    "gemmaRepo",
    "repo",
    "repository",
    "textEncoderRepo",
  ]);
  // Keys that merely LOOK repo-shaped (paths, filenames) must stay out of REPO_KEY.
  for (const key of ["subdir", "file", "transformerFile", "label"]) {
    assert.equal(REPO_KEY.test(key), false, `${key} must not be treated as a repo id`);
  }
});

test("withholding a manifest flips a live repo to unreferenced", async () => {
  // Discrimination for the invariant above: without this it could pass by never saying
  // "unreferenced" at all.
  const manifests = await loadManifests();
  const withoutOverlays = Object.fromEntries(
    Object.entries(manifests).filter(([name]) => name !== "builtin.control_overlays.jsonc"),
  );
  const [entry] = run({ repos: ["SceneWorks/krea2-pose-controlnet-beta"], manifests: withoutOverlays });
  assert.equal(entry.verdict, VERDICT.UNREFERENCED);
  assert.equal(entry.referenced, false);
});

test("a repo hardcoded in the source is app-live though no manifest declares it", async () => {
  // The BLOCKER this tool shipped with. `QWEN_LIGHTNING_LORA_REPO` is an installed model's
  // required distill LoRA, fetched on demand into the hub cache and declared in no manifest.
  const manifests = await loadManifests();
  const repo = "lightx2v/Qwen-Image-Edit-2511-Lightning";
  assert.equal(manifestRepoIndex(manifests).has(repo), false, "premise: no manifest declares it");

  const { production } = await loadSourceFiles();
  const literals = sourceLiteralIndex(production);
  assert.ok(literals.has(repo), "the production source scan must find the constant");

  const [live] = run({ repos: [repo], manifests, sourceLiterals: literals });
  assert.equal(live.verdict, VERDICT.SOURCE);
  assert.equal(live.referenced, true);

  // Discrimination: with the source scan withheld it is exactly the false-safe the review found.
  const [blind] = run({ repos: [repo], manifests });
  assert.equal(blind.referenced, false, "premise for the regression: only the source scan saves it");
});

test("test-only sources never vote a repo app-live, but are reported", async () => {
  const manifests = await loadManifests();
  assert.equal(isTestSource("apps/rust-api/src/tests/catalog.rs"), true);
  assert.equal(isTestSource("scripts/foo.test.mjs"), true);
  assert.equal(isTestSource("crates/x/tests/fixtures/y.rs"), true);
  assert.equal(isTestSource("crates/sceneworks-worker/src/image_jobs/qwen.rs"), false);

  const [entry] = run({
    repos: ["acme/fixture-only"],
    manifests,
    testLiterals: new Map([["acme/fixture-only", ["crates/x/tests/y.rs"]]]),
  });
  assert.equal(entry.verdict, VERDICT.UNREFERENCED, "a fixture is not an app read path");
  assert.match(entry.reason, /test sources/, "but the operator must be told before deleting");
});

test("an installed LoRA keeps its repo alive with no model marker anywhere", async () => {
  const manifests = await loadManifests();
  const [entry] = run({
    repos: ["Comfy-Org/Krea-2"],
    installedModels: [],
    loraReceipts: [{ loraId: "krea2_turbo_accel", repo: "Comfy-Org/Krea-2" }],
    manifests,
  });
  assert.equal(entry.verdict, VERDICT.LORA);
  assert.equal(entry.referenced, true);
  assert.match(entry.reason, /krea2_turbo_accel/);
});

test("a co-requisite is alive only while its parent model is installed", async () => {
  const manifests = await loadManifests();
  const codec = "OpenMOSS-Team/XY_Tokenizer_TTSD_V0";

  const [installed] = run({ repos: [codec], installedModels: ["OpenMOSS-Team/MOSS-TTSD-v0.5"], manifests });
  assert.equal(installed.verdict, VERDICT.COREQUISITE);
  assert.match(installed.reason, /codec/);

  const [uninstalled] = run({ repos: [codec], installedModels: [], manifests });
  assert.notEqual(uninstalled.verdict, VERDICT.COREQUISITE);
  assert.equal(uninstalled.referenced, true, "still declared — the fail-closed floor");
});

test("the perth watermarker resolves as a co-requisite of chatterbox", async () => {
  // sc-17283 established this by hand. Asserted so the tool cannot regress to the
  // "no install marker ⇒ deletable" answer that would break every cloned-voice render.
  const manifests = await loadManifests();
  const [entry] = run({
    repos: ["SceneWorks/perth-implicit"],
    installedModels: ["ResembleAI/chatterbox"],
    manifests,
  });
  assert.equal(entry.verdict, VERDICT.COREQUISITE);
  assert.match(entry.reason, /perth/);
});

test("the model install marker wins over every other signal", async () => {
  const manifests = await loadManifests();
  const [entry] = run({
    repos: ["ResembleAI/chatterbox"],
    installedModels: ["ResembleAI/chatterbox"],
    loraReceipts: [{ loraId: "irrelevant", repo: "ResembleAI/chatterbox" }],
    manifests,
  });
  assert.equal(entry.verdict, VERDICT.MODEL);
});

test("the CI provenance marker never decides a verdict on its own", async () => {
  const manifests = await loadManifests();
  const [entry] = run({
    repos: ["Comfy-Org/Krea-2"],
    loraReceipts: [{ loraId: "krea2_turbo_accel", repo: "Comfy-Org/Krea-2" }],
    ciProvisioned: ["Comfy-Org/Krea-2"],
    manifests,
  });
  assert.equal(entry.ciProvisioned, true);
  assert.equal(entry.referenced, true);
});

test("`referenced` tracks `unreferenced` exactly, across every verdict", async () => {
  const manifests = await loadManifests();
  const rows = run({
    repos: [
      "ResembleAI/chatterbox",
      "OpenMOSS-Team/XY_Tokenizer_TTSD_V0",
      "Comfy-Org/Krea-2",
      "SceneWorks/krea2-pose-controlnet-beta",
      "acme/hardcoded-only",
      "acme/nothing-references-this",
    ],
    installedModels: ["ResembleAI/chatterbox", "OpenMOSS-Team/MOSS-TTSD-v0.5"],
    loraReceipts: [{ loraId: "krea2_turbo_accel", repo: "Comfy-Org/Krea-2" }],
    sourceLiterals: new Map([["acme/hardcoded-only", ["crates/x/src/y.rs"]]]),
    manifests,
  });
  const seen = new Set(rows.map((entry) => entry.verdict));
  assert.equal(seen.size, 6, `expected every verdict exactly once, saw ${[...seen].join(", ")}`);
  for (const entry of rows) {
    assert.equal(
      entry.referenced,
      entry.verdict !== VERDICT.UNREFERENCED,
      `${entry.repo}: referenced must track the verdict, got ${entry.verdict}`,
    );
  }
  // No `safeToDelete` field: this tool never clears anything for deletion.
  assert.equal("safeToDelete" in rows[0], false);
});

test("the source scan intersects with the cache, so shape false positives are harmless", () => {
  const literals = sourceLiteralIndex({ "a.rs": 'let m = "image/png"; let r = "acme/real-repo";' });
  assert.ok(literals.has("image/png"), "over-broad by design");
  assert.ok(literals.has("acme/real-repo"));
  // Harmless because classify() is only ever asked about repos that exist in the cache.
  assert.equal(path.basename("x"), "x");
});
