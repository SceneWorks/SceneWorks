// The memory anchor currency key, asked of the REAL pinned inference source (sc-22511, epic 22505).
//
// The claims this file exists for are claims about a repository, not about a fixture: "a pin bump
// leaves the key unchanged", "a sibling model's edit leaves it unchanged", "an edit inside the
// model's own loader changes it". A synthetic two-crate workspace can be made to agree with a walk
// that is wrong about the real tree — inference's own `mlx-gen-ltx` hosts TWO models, and
// `mlx-gen/src/request_scope.rs` carries a `#[cfg(test)]` block that `include_str!`s eight sibling
// models' memory-strategy source — so every claim below is driven against the pinned revision in a
// real inference clone, with exactly one REAL file overlaid.
//
// Those cases need a clone containing the pin. ON CI THEY ARE NOT OPTIONAL: `check.yml`'s
// parity-scaffold job fetches the pinned revision (and the revision the anchors were MEASURED at)
// into `$RUNNER_TEMP/inference` and points `INFERENCE_REPO` at it, so a missing clone under
// `process.env.CI` is a hard FAILURE rather than a skip. Most of the cases here — every claim the
// walk exists to make — used to skip on CI, which made green mean "nothing ran".
// Off CI (a laptop with no inference checkout) they still skip, and the hermetic pure-function
// cases beside them always run.
import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  ANCHOR_CURRENCY_ATTESTATIONS_PATH,
  ANCHOR_LOADER_CLOSURE_VERSION,
  anchorCurrencyRevision,
  anchorMeasurementRevision,
  assertModelIsNamedByEntryPoints,
  buildAnchorLoaderConfig,
  indexCurrencyAttestations,
  expandUsePath,
  firstPartyCrates,
  gitTree,
  loaderClosureDigest,
  loaderClosureFiles,
  loaderClosureText,
  moduleDirOf,
  overlayTree,
  referencesIn,
  stampAnchorStore,
  stripCfgTest,
} from "./anchor-loader-closure.mjs";
import { inferencePinFromCargo } from "./inference-closure-digest.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const config = JSON.parse(readFileSync(path.join(root, "config/anchor-loader-closures.json"), "utf8"));
const MODEL = "ltx_2_5:mlx";
const PIN = inferencePinFromCargo(readFileSync(path.join(root, "Cargo.toml"), "utf8"));

/** The revision the packaged anchors were MEASURED at, from the retained evidence they cite. */
const MEASURED_AT = JSON.parse(
  readFileSync(path.join(root, "docs/calibration/sc-18791/ltx25-mlx-evidence.seed.json"), "utf8"),
).records[0].repositories.inference.revision;

function inferenceRepo() {
  const candidate = process.env.INFERENCE_REPO ?? path.join(os.homedir(), "Repos/inference");
  try {
    execFileSync("git", ["-C", candidate, "cat-file", "-e", `${PIN}^{commit}`], {
      stdio: "ignore",
    });
    return candidate;
  } catch {
    return null;
  }
}

const repo = inferenceRepo();
if (!repo && process.env.CI) {
  throw new Error(
    `no inference clone containing ${PIN.slice(0, 8)}. On CI this is a FAILURE, not a skip: ` +
      "check.yml's parity-scaffold job fetches the pinned revision into $RUNNER_TEMP/inference and " +
      "sets INFERENCE_REPO. If that step was removed or its fetch failed, every real claim in this " +
      "file would silently skip and CI green would mean nothing ran.",
  );
}
const skip = repo ? false : `no inference clone containing ${PIN.slice(0, 8)}`;

/**
 * The measurement revision is a SECOND fetch in the CI step, so on CI its absence is a failure for
 * the same reason. Off CI a shallow or partial clone may honestly not carry it.
 */
function hasRevision(revision) {
  try {
    execFileSync("git", ["-C", repo, "cat-file", "-e", `${revision}^{commit}`], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

/**
 * `revisions` must all be present. On CI a missing one is a FAILURE — the parity-scaffold step
 * fetches every revision `--anchor-revisions` names, so an absence means that step regressed and
 * the case would otherwise pass without asserting anything. Off CI it is a named skip.
 */
function requireRevisions(t, revisions) {
  const missing = [...new Set(revisions)].filter((revision) => !hasRevision(revision));
  if (missing.length === 0) return true;
  const list = missing.map((revision) => revision.slice(0, 8)).join(", ");
  if (process.env.CI) {
    throw new Error(
      `the inference clone at ${repo} does not carry ${list}. On CI this is a FAILURE: the ` +
        "parity-scaffold step fetches the pin and every revision `--anchor-revisions` names, so " +
        "these claims are actually asked. Passing quietly here is how a test asserts nothing.",
    );
  }
  t.skip(`local clone does not carry ${list}`);
  return false;
}

function requireMeasurementRevision(t) {
  return requireRevisions(t, [MEASURED_AT]);
}

/** The key at one revision, over an optionally overlaid tree. */
function keyAt(revision, overrides = {}) {
  const base = gitTree(repo, revision);
  const tree = Object.keys(overrides).length ? overlayTree(base, overrides) : base;
  const crates = firstPartyCrates(tree, tree.paths());
  return loaderClosureDigest({
    model: MODEL,
    entryPoints: config.models[MODEL].entryPoints,
    tree,
    crates,
  });
}

/** One real file's body at the pin, with a real code line appended. */
function edited(tree, file, line) {
  const body = tree.read([file]).get(file);
  assert.ok(body && body.length > 0, `${file} must be a real file with content`);
  return `${body}\n${line}\n`;
}

// ---------------------------------------------------------------------------------------------
// Hermetic: the pieces the walk is built from.
// ---------------------------------------------------------------------------------------------

test("a `use` item's brace groups expand to one edge each, aliases CARRIED", () => {
  assert.deepEqual(expandUsePath("a::b::{c, d::{e, f}, g as h}"), [
    { segments: ["a", "b", "c"], alias: null },
    { segments: ["a", "b", "d", "e"], alias: null },
    { segments: ["a", "b", "d", "f"], alias: null },
    // The alias is the name a consumer asks for; the segments still address the source item.
    { segments: ["a", "b", "g"], alias: "h" },
  ]);
  // The shape that matters: a naive `::` split would read `d::e` as a TOP-LEVEL path and resolve
  // it against the wrong module.
  assert.deepEqual(expandUsePath("mlx_gen::{Result, array::contiguous}"), [
    { segments: ["mlx_gen", "Result"], alias: null },
    { segments: ["mlx_gen", "array", "contiguous"], alias: null },
  ]);
  // The real crate-root shape this exists for.
  assert.deepEqual(expandUsePath("vocoder::{Generator as VocoderGenerator, LtxVocoder}"), [
    { segments: ["vocoder", "Generator"], alias: "VocoderGenerator" },
    { segments: ["vocoder", "LtxVocoder"], alias: null },
  ]);
});

test("`#[cfg(test_only_api)]` is NOT a test cfg and its code is shipped", () => {
  // Without a boundary on the attribute match, `#[cfg(test` prefix-matched this and silently
  // dropped shipped loader code from the hash — a false green in the direction that matters.
  const body = "#[cfg(test_only_api)]\npub fn shipped() { let _ = 1; }\npub fn after() {}";
  assert.ok(stripCfgTest(body).includes("pub fn shipped"), stripCfgTest(body));
  assert.ok(stripCfgTest(body).includes("pub fn after"));
  // The real one is still stripped, both spellings.
  assert.ok(!stripCfgTest("#[cfg(test)]\nmod t { fn a() {} }\npub fn k() {}").includes("mod t"));
  assert.ok(
    !stripCfgTest("#[cfg(all( test ))]\nmod t { fn a() {} }\npub fn k() {}").includes("mod t"),
  );
});

test("`mod` declarations and `#[path]` overrides are reported, `include!` is data", () => {
  const { mods, includes } = referencesIn(
    [
      "pub mod plain;",
      "mod private_impl;",
      "pub(crate) mod scoped;",
      '#[path = "../elsewhere/thing.rs"]',
      "mod relocated;",
      "mod inline { pub fn f() {} }",
      "mod commented; // a trailing comment is not part of the declaration",
      'const A: &str = include!("data/table.rs");',
      'const B: &str = include_str!("data/prompt.txt");',
    ].join("\n"),
  );
  assert.deepEqual(mods, [
    { name: "plain", path: null },
    { name: "private_impl", path: null },
    { name: "scoped", path: null },
    { name: "relocated", path: "../elsewhere/thing.rs" },
    { name: "commented", path: null },
  ]);
  assert.deepEqual(includes, ["data/table.rs", "data/prompt.txt"]);
});

test("`#[cfg(test)]` items are not shipped source", () => {
  const body = [
    "pub fn load() {}",
    "#[cfg(test)]",
    "mod tests {",
    "    use mlx_gen_z_image::model;",
    "    fn nested() { let _ = 1; }",
    "}",
    "pub fn after() {}",
  ].join("\n");
  const shipped = stripCfgTest(body);
  assert.ok(shipped.includes("pub fn load"));
  assert.ok(shipped.includes("pub fn after"));
  assert.ok(!shipped.includes("mlx_gen_z_image"), shipped);

  // The file-declaration spelling has no braces to balance.
  assert.ok(!stripCfgTest("#[cfg(test)]\nmod tests;\npub fn kept() {}").includes("mod tests"));
  // A non-test cfg is shipped source and stays.
  assert.ok(stripCfgTest('#[cfg(feature = "x")]\npub fn kept() {}').includes("pub fn kept"));
});

test("a module file owns the directory Cargo says it does", () => {
  assert.equal(moduleDirOf("a/src/lib.rs"), "a/src");
  assert.equal(moduleDirOf("a/src/diff_vae.rs"), "a/src/diff_vae");
  assert.equal(moduleDirOf("a/src/diff_vae/mod.rs"), "a/src/diff_vae");
});

// ---------------------------------------------------------------------------------------------
// The real thing.
// ---------------------------------------------------------------------------------------------

test("the checked-in key matches the derivation over the pinned inference source", { skip }, () => {
  const derived = keyAt(PIN);
  assert.equal(derived.digest, config.models[MODEL].digest);
  assert.equal(derived.files.length, config.models[MODEL].closureFileCount);
  assert.deepEqual(derived.files, config.models[MODEL].closureFiles);
  assert.equal(config.digestVersion, ANCHOR_LOADER_CLOSURE_VERSION);
});

test("the closure is the loader's own crates, not the repository", { skip }, () => {
  const { files } = keyAt(PIN);
  const crateOf = (file) => file.slice(0, file.indexOf("/src/"));
  const crates = new Set(files.map(crateOf));
  assert.ok(crates.has("crates/media/mlx-gen/mlx-gen-ltx"), [...crates].join(" "));
  // Not one other model crate — the whole point of the unit.
  const otherModels = [...crates].filter(
    (crate) => /mlx-gen-|candle-gen-/.test(crate) && crate !== "crates/media/mlx-gen/mlx-gen-ltx",
  );
  assert.deepEqual(otherModels, [], `sibling model crates leaked into the closure: ${otherModels}`);
});

/**
 * THE PIN-BUMP CASE, and it is not hypothetical: the anchors were measured at `MEASURED_AT`, the
 * repository has moved to `PIN` since, and the loader's source did not. Under the old crate-level
 * unit that revision change alone rotated the currency term; under this one the key is identical,
 * which is E9's claim — an anchor predating an unrelated change stays authoritative.
 */
test("a pin bump with unchanged loader source leaves the key unchanged", { skip }, (t) => {
  // Off CI a local clone that does not carry the measurement revision cannot ask this, and says so
  // by name. On CI `requireMeasurementRevision` throws instead of skipping.
  if (!requireMeasurementRevision(t)) return;
  assert.notEqual(MEASURED_AT, PIN, "the two revisions must actually differ");
  // The claim is about the KEY, not about whether this particular pin happened to leave the
  // loader alone — sc-22414's coherence guard (670dc1f4) moved it, by design, and E9 says that
  // stales the anchor. So the pin bump under test is the tree at `PIN` with the loader's own
  // closure held at its MEASURED_AT content: every other file in the repository — and the
  // revision itself — has moved, and the key must not notice.
  const measured = keyAt(MEASURED_AT);
  const measuredTree = gitTree(repo, MEASURED_AT);
  const pinTree = gitTree(repo, PIN);
  // Overlay only the closure files whose content moved: an overlaid file carries a synthetic
  // content id, so holding an UNCHANGED file would itself perturb a non-Rust file's hash.
  const bodies = measuredTree.read(measured.files);
  const held = Object.fromEntries(
    measured.files
      .filter((file) => pinTree.contentId(file) !== measuredTree.contentId(file))
      .map((file) => [file, bodies.get(file)]),
  );
  assert.ok(Object.keys(held).length > 0, "this pin moved at least one loader file");
  const bumped = keyAt(PIN, held);
  assert.deepEqual(bumped.files, measured.files, "the held closure walks to the same files");
  assert.equal(bumped.digest, measured.digest);
  // Teeth: the same bump WITHOUT holding the loader still is a real loader move at this pin, and
  // the key says so — the other half of E9, asserted against the real tree rather than assumed.
  if (keyAt(PIN).digest !== measured.digest) {
    assert.notDeepEqual(keyAt(PIN).files, measured.files, "a moved key names its moved files");
  }
});

test("a sibling model's edit leaves the key unchanged", { skip }, () => {
  const base = gitTree(repo, PIN);
  const current = keyAt(PIN).digest;
  for (const sibling of [
    "crates/media/mlx-gen/mlx-gen-z-image/src/model.rs",
    "crates/media/mlx-gen/mlx-gen-wan/src/model.rs",
    "crates/media/mlx-gen/mlx-gen-qwen-image/src/memory_strategy.rs",
  ]) {
    assert.ok(base.has(sibling), `${sibling} must be a real file at the pin`);
    const overridden = keyAt(PIN, {
      [sibling]: edited(base, sibling, "pub const SC_22511_SIBLING_EDIT: u64 = 1;"),
    });
    assert.equal(overridden.digest, current, `${sibling} moved this model's key`);
    assert.ok(!overridden.files.includes(sibling));
  }
});

test("an edit to a shared crate the loader does not reach leaves the key unchanged", { skip }, () => {
  const base = gitTree(repo, PIN);
  const { digest, files } = keyAt(PIN);
  for (const shared of [
    // A file inside a shared crate the loader DOES link and DOES partly reach — but this file is
    // not on any path from the loader's entry points.
    "crates/contracts/gen-core/src/sdxl_ldm.rs",
    "crates/contracts/gen-core/src/wan_i2v_memory.rs",
    // A shared crate on the other lane entirely.
    "crates/llm/candle-llm/src/lib.rs",
  ]) {
    assert.ok(base.has(shared), `${shared} must be a real file at the pin`);
    assert.ok(!files.includes(shared), `${shared} is IN the closure — pick an unreached file`);
    const overridden = keyAt(PIN, {
      [shared]: edited(base, shared, "pub const SC_22511_UNREACHED_EDIT: u64 = 1;"),
    });
    assert.equal(overridden.digest, digest, `${shared} moved this model's key`);
  }
});

/**
 * The falsifier. A unit that absolved everything would pass every test above, so the "must move"
 * direction carries the same weight: an edit anywhere on the loader's own path — its entry point,
 * a module it reaches inside its crate, or a shared crate file it genuinely executes — stales it.
 */
test("an edit inside the model's own loader path stales the key", { skip }, () => {
  const base = gitTree(repo, PIN);
  const { digest, files } = keyAt(PIN);
  for (const reached of [
    "crates/media/mlx-gen/mlx-gen-ltx/src/model.rs",
    "crates/media/mlx-gen/mlx-gen-ltx/src/memory_strategy_2_5.rs",
    "crates/media/mlx-gen/mlx-gen-ltx/src/pipeline.rs",
    "crates/media/mlx-gen/mlx-gen-ltx/src/transformer.rs",
    "crates/media/mlx-gen/src/memory_strategy_shared.rs",
    "crates/contracts/gen-core/src/memory_strategy.rs",
  ].filter((file) => files.includes(file))) {
    const overridden = keyAt(PIN, {
      [reached]: edited(base, reached, "pub const SC_22511_LOADER_EDIT: u64 = 1;"),
    });
    assert.notEqual(overridden.digest, digest, `${reached} is in the closure but did not move it`);
  }
  // The list above is filtered against the real closure, so assert it was not filtered to nothing.
  assert.ok(files.includes("crates/media/mlx-gen/mlx-gen-ltx/src/pipeline.rs"));
  assert.ok(files.includes("crates/contracts/gen-core/src/memory_strategy.rs"));
});

test("a comment-only edit inside the loader does NOT stale the key", { skip }, () => {
  const base = gitTree(repo, PIN);
  const file = "crates/media/mlx-gen/mlx-gen-ltx/src/pipeline.rs";
  const overridden = keyAt(PIN, {
    [file]: edited(base, file, "// sc-22511: a documentation line changes nothing that compiles"),
  });
  assert.equal(overridden.digest, keyAt(PIN).digest);
});

test("a `#[cfg(test)]` edit inside the loader does NOT stale the key", { skip }, () => {
  const base = gitTree(repo, PIN);
  const file = "crates/media/mlx-gen/mlx-gen-ltx/src/pipeline.rs";
  const overridden = keyAt(PIN, {
    [file]: edited(
      base,
      file,
      "#[cfg(test)]\nmod sc_22511_tests {\n    #[test]\n    fn t() { assert!(true); }\n}",
    ),
  });
  assert.equal(overridden.digest, keyAt(PIN).digest);
});

// ---------------------------------------------------------------------------------------------
// The walk's four resolution shapes, each asked over the REAL pinned tree. Every one of them
// dropped an edge before sc-22511's review: a dropped edge takes real loader files out of the unit,
// so edits there leave anchors reading "current" — the expensive direction to be wrong in.
// ---------------------------------------------------------------------------------------------

const LTX_SRC = "crates/media/mlx-gen/mlx-gen-ltx/src";

/** The closure over the pinned tree with `overrides` applied and `entryPoints` declared. */
function filesFor(overrides, entryPoints) {
  const tree = overlayTree(gitTree(repo, PIN), overrides);
  return new Set(
    loaderClosureFiles({ tree, entryPoints, crates: firstPartyCrates(tree, tree.paths()) }),
  );
}

test("an ALIASED crate-root re-export is followed under the name it publishes", { skip }, () => {
  const base = gitTree(repo, PIN);
  // The real line, read out of the real crate root rather than retyped, so this cannot pass
  // against a shape inference does not have.
  const line = base
    .read([`${LTX_SRC}/lib.rs`])
    .get(`${LTX_SRC}/lib.rs`)
    .split("\n")
    .find((l) => l.startsWith("pub use vocoder::"));
  assert.equal(line, "pub use vocoder::{Generator as VocoderGenerator, LtxVocoder, VocoderWithBwe};");

  // Isolated deliberately: at the pin, `vocoder.rs` is also reachable by several unaliased routes,
  // so the whole closure would answer "in" no matter what the alias gate did. Cutting the crate
  // root down to the ONE re-export and asking for the item under its ALIAS makes the alias the
  // only thing that can admit it.
  const closure = filesFor(
    {
      [`${LTX_SRC}/lib.rs`]: `${line}\n`,
      [`${LTX_SRC}/model.rs`]:
        'pub const MODEL_ID: &str = "ltx_2_5";\npub type Probe = crate::VocoderGenerator;\n',
    },
    [`${LTX_SRC}/model.rs`],
  );
  assert.ok(
    closure.has(`${LTX_SRC}/vocoder.rs`),
    "the module is reachable ONLY under the alias `VocoderGenerator`; gating the re-export on the " +
      "source segment `Generator` drops it and every edit inside it stops staling the anchor",
  );
});

test("a leading `super` RUN is consumed whole, not just its first hop", { skip }, () => {
  const base = gitTree(repo, PIN);
  // `gen-core/src/sampling/unified.rs` is two module levels down and is really in this loader's
  // closure; `super::super::` from there addresses the crate root's own modules. The pinned tree
  // spells exactly this at `unified.rs:255` (inside `#[cfg(test)]`, so it is stripped before the
  // hash — the shape is live, its current site is not shipped).
  const unified = "crates/contracts/gen-core/src/sampling/unified.rs";
  const target = "crates/contracts/gen-core/src/sdxl_ldm.rs";
  const current = keyAt(PIN);
  assert.ok(current.files.includes(unified), `${unified} must be in the closure`);
  assert.ok(!current.files.includes(target), `${target} must start OUTSIDE the closure`);

  const closure = filesFor(
    { [unified]: edited(base, unified, "pub type Probe = super::super::sdxl_ldm::SdxlLdm;") },
    config.models[MODEL].entryPoints,
  );
  assert.ok(
    closure.has(target),
    "handling only the first `super` resolves the rest against the WRONG module and drops the edge",
  );
});

test("a `mod` declaration ALONE admits the module file", { skip }, () => {
  // Impl-only modules are reached by method syntax and named by no path anywhere, so a purely
  // path-driven walk never sees them. Cutting `diff_vae.rs` down to its own `mod budget;` line
  // removes every path that could name the child: only the declaration is left to admit it.
  const closure = filesFor(
    { [`${LTX_SRC}/diff_vae.rs`]: "mod budget;\n" },
    config.models[MODEL].entryPoints,
  );
  assert.ok(
    closure.has(`${LTX_SRC}/diff_vae/budget.rs`),
    "a declared module of an admitted file must be in the unit even when no path names it",
  );
});

test("a `#[path]` override resolves the module to the file it names", { skip }, () => {
  const declared = config.models[MODEL].entryPoints;
  const target = `${LTX_SRC}/convert.rs`;
  assert.ok(!keyAt(PIN).files.includes(target), `${target} must start OUTSIDE the closure`);
  const closure = filesFor(
    { [`${LTX_SRC}/diff_vae.rs`]: '#[path = "convert.rs"]\nmod probe;\n' },
    declared,
  );
  assert.ok(
    closure.has(target),
    "`#[path]` is relative to the declaring file's own directory; ignoring it resolves the module " +
      "to a file that does not exist and silently drops it",
  );
});

// ---------------------------------------------------------------------------------------------
// The anchors' recorded half: a key frozen at the revision the measurement was taken at.
// ---------------------------------------------------------------------------------------------

/** The packaged store, every corpus its anchors cite, and the declaration in derivable form. */
function packagedStore() {
  const store = JSON.parse(readFileSync(path.join(root, "config/memory-anchors.json"), "utf8"));
  const corpora = new Map();
  for (const anchor of store.anchors) {
    const cited = anchor.source.path;
    if (!corpora.has(cited)) {
      corpora.set(cited, JSON.parse(readFileSync(path.join(root, cited), "utf8")));
    }
  }
  const declared = Object.fromEntries(
    Object.entries(config.models).map(([model, entry]) => [
      model,
      { entryPoints: entry.entryPoints },
    ]),
  );
  const attestations = indexCurrencyAttestations(
    JSON.parse(readFileSync(path.join(root, ANCHOR_CURRENCY_ATTESTATIONS_PATH), "utf8")),
  );
  // Both the measurement revisions and the attested ones: the stamp derives at the latter for an
  // attested anchor, and the attestation check re-reads the former.
  const revisions = store.anchors.flatMap((anchor) => [
    anchorMeasurementRevision(anchor, corpora.get(anchor.source.path)),
    anchorCurrencyRevision(anchor, corpora.get(anchor.source.path), attestations).revision,
  ]);
  return { store, corpora, declared, revisions, attestations };
}

test("every packaged anchor's key is the derivation at ITS OWN measurement revision", { skip }, (t) => {
  const { store, corpora, declared, revisions, attestations } = packagedStore();
  if (!requireRevisions(t, revisions)) return;
  const { store: stamped } = stampAnchorStore({ repo, store, declared, corpora, attestations });
  // The whole `source` block, not just the digest: an attested anchor's key is one claim with the
  // attestation that justifies deriving it later than the measurement (sc-22667).
  assert.deepEqual(
    stamped.anchors.map((anchor) => anchor.source),
    store.anchors.map((anchor) => anchor.source),
    "re-run: node scripts/anchor-loader-closure.mjs --repo <clone> --stamp-anchors",
  );

  // AND IT IS NOT THE PIN'S DIGEST. If the recorded half were derived at the pin, currency would
  // compare a value with itself and report "current" through every loader change there is. The
  // packaged store carries anchors whose loaders HAVE moved since they were measured, and they must
  // read as stale.
  const stale = store.anchors.filter(
    (anchor) =>
      anchor.source.loaderClosureDigest !==
      config.models[`${anchor.modelId}:${anchor.backend}`].digest,
  );
  assert.ok(
    stale.length > 0,
    "no packaged anchor is stale — the recorded key is being derived at the pin, not at its " +
      "measurement, and currency has become a tautology",
  );
  // And no UNATTESTED anchor reads current unless its measurement revision's closure really equals
  // the pin's: currency by attestation is the only other way to be current, and it is declared.
  for (const anchor of store.anchors) {
    if (anchor.source.currencyAttestation) continue;
    const declaredAtPin = config.models[`${anchor.modelId}:${anchor.backend}`].digest;
    if (anchor.source.loaderClosureDigest === declaredAtPin) {
      const measured = anchorMeasurementRevision(anchor, corpora.get(anchor.source.path));
      assert.notEqual(
        measured,
        PIN,
        `${anchor.id}: measured at the pin itself, so currency is a value against itself here`,
      );
    }
  }
});

test("an ATTESTED anchor's key is the derivation at the attested revision, carried with its justification (sc-22667)", { skip }, (t) => {
  const { store, corpora, declared, revisions, attestations } = packagedStore();
  if (!requireRevisions(t, revisions)) return;
  const attested = store.anchors.filter((anchor) => attestations.has(anchor.id));
  assert.ok(attested.length > 0, "the packaged store must carry an attested anchor for this case");
  const { store: withAttestations, report } = stampAnchorStore({ repo, store, declared, corpora, attestations });
  const { store: withoutAttestations } = stampAnchorStore({ repo, store, declared, corpora });
  for (const anchor of attested) {
    const attestation = attestations.get(anchor.id);
    const now = withAttestations.anchors.find((entry) => entry.id === anchor.id);
    const raw = withoutAttestations.anchors.find((entry) => entry.id === anchor.id);
    const row = report.find((entry) => entry.id === anchor.id);
    // Derived at the attested revision, and the report says so.
    assert.equal(row.revision, attestation.attestedRevision);
    assert.equal(row.attested, true);
    // The attestation DID something: at the measurement revision the closure differs — otherwise
    // the entry is dead weight and should be deleted, not carried.
    assert.notEqual(
      now.source.loaderClosureDigest,
      raw.source.loaderClosureDigest,
      `${anchor.id}: the closure did not move between measurement and attested revision`,
    );
    // The justification travels with the key: the store's copy is the config's entry minus the
    // file list, field for field.
    assert.deepEqual(now.source.currencyAttestation, {
      measuredRevision: attestation.measuredRevision,
      attestedRevision: attestation.attestedRevision,
      attestedAt: attestation.attestedAt,
      story: attestation.story,
      class: attestation.class,
      why: attestation.why,
      witness: attestation.witness,
    });
    // And an unattested stamp carries NO attestation — a stale entry cannot linger in the store.
    assert.equal(raw.source.currencyAttestation, undefined);
  }
});

test("an attestation whose measurement revision no longer matches the record is refused (sc-22667)", () => {
  const { store, corpora, attestations } = packagedStore();
  const anchor = store.anchors.find((entry) => attestations.has(entry.id));
  assert.ok(anchor, "needs an attested packaged anchor");
  const entry = attestations.get(anchor.id);
  const moved = new Map([[anchor.id, { ...entry, measuredRevision: "0".repeat(40) }]]);
  assert.throws(
    () => anchorCurrencyRevision(anchor, corpora.get(anchor.source.path), moved),
    /the measurement moved under the attestation/,
  );
  // The unmoved entry resolves to the attested revision; an unattested anchor to its own.
  assert.equal(
    anchorCurrencyRevision(anchor, corpora.get(anchor.source.path), attestations).revision,
    entry.attestedRevision,
  );
  assert.equal(
    anchorCurrencyRevision(anchor, corpora.get(anchor.source.path), new Map()).revision,
    entry.measuredRevision,
  );
});

test("an attestation without its justification, or of its own measurement revision, is refused (sc-22667)", () => {
  const config = JSON.parse(readFileSync(path.join(root, ANCHOR_CURRENCY_ATTESTATIONS_PATH), "utf8"));
  assert.ok(config.attestations.length > 0);
  const [entry] = config.attestations;
  const doctored = (patch) => ({ attestations: [{ ...entry, ...patch }] });
  for (const field of ["why", "witness", "class", "attestedAt", "story"]) {
    assert.throws(
      () => indexCurrencyAttestations(doctored({ [field]: "  " })),
      new RegExp(`states no ${field}`),
      field,
    );
  }
  assert.throws(
    () => indexCurrencyAttestations(doctored({ attestedRevision: entry.measuredRevision })),
    /attests its own measurement revision/,
  );
  assert.throws(
    () => indexCurrencyAttestations(doctored({ attestedRevision: "abc" })),
    /attestedRevision is not a 40-hex revision/,
  );
  assert.throws(
    () => indexCurrencyAttestations(doctored({ filesChangedSinceMeasurement: [{ path: "x" }] })),
    /must list every closure file/,
  );
  assert.throws(
    () => indexCurrencyAttestations({ attestations: [entry, entry] }),
    /is attested twice/,
  );
  // The checked-in file itself is well-formed, and every entry names a packaged anchor.
  const indexed = indexCurrencyAttestations(config);
  const store = JSON.parse(readFileSync(path.join(root, "config/memory-anchors.json"), "utf8"));
  for (const id of indexed.keys()) {
    assert.ok(store.anchors.some((anchor) => anchor.id === id), `${id} is not a packaged anchor`);
  }
});

test("an entry point absent at a historical revision narrows the unit, it does not throw", { skip }, (t) => {
  // A measurement can predate a file today's declaration names — `mlx-gen-ltx`'s per-model
  // `memory_strategy.rs` postdates the LTX-2.3 capture. The entry-point list is part of the hashed
  // text, so a closure derived over a shorter list cannot equal the pin's, and the anchor reads
  // NOT CURRENT. That is the truth about it, and it must not be an error.
  const { store, corpora, declared, revisions } = packagedStore();
  if (!requireRevisions(t, revisions)) return;
  const ltx23 = store.anchors.filter((anchor) => anchor.modelId === "ltx_2_3");
  assert.ok(ltx23.length > 0, "the packaged store must carry an ltx_2_3 anchor");
  const { report } = stampAnchorStore({ repo, store, declared, corpora });
  const row = report.find((entry) => entry.id === ltx23[0].id);
  const historical = gitTree(repo, row.revision);
  const absent = config.models["ltx_2_3:mlx"].entryPoints.filter((file) => !historical.has(file));
  assert.ok(absent.length > 0, "this case needs an entry point that really is absent back then");
  assert.notEqual(row.digest, config.models["ltx_2_3:mlx"].digest);
});

test("an entry point that never names the model is refused", { skip }, () => {
  assert.throws(
    () =>
      buildAnchorLoaderConfig({
        repo,
        revision: PIN,
        declared: {
          [MODEL]: { entryPoints: ["crates/media/mlx-gen/mlx-gen-wan/src/model.rs"] },
        },
      }),
    /never carry the literal "ltx_2_5"/,
  );
});

// sc-22724: `z_image_edit` is a SceneWorks catalog id for the `z_image_turbo` provider driven in
// `edit_image` mode (worker engines.rs); the inference tree never spells "z_image_edit". The alias
// is declared EXPLICITLY as `engineId`, the literal rule is asked of the engine id, and the engine
// id is hashed into the closure text — so an alias is never a way around the check.
test("a catalog alias declares the engine id it resolves to, and that id must be named by the entry points", () => {
  const tree = {
    read: (files) => new Map(files.map((file) => [
      file,
      file.endsWith("model.rs") ? 'pub const MODEL_ID: &str = "z_image_turbo";' : "// nothing named here",
    ])),
  };
  const entryPoints = ["crates/x/src/model.rs", "crates/x/src/memory_strategy.rs"];
  assert.throws(
    () => assertModelIsNamedByEntryPoints({ model: "z_image_edit:mlx", entryPoints, tree }),
    /never carry the literal "z_image_edit"/,
  );
  assertModelIsNamedByEntryPoints({ model: "z_image_edit:mlx", engineId: "z_image_turbo", entryPoints, tree });
  // The alias is not a wildcard: an engine id the entry points never name is refused the same way.
  assert.throws(
    () => assertModelIsNamedByEntryPoints({ model: "z_image_edit:mlx", engineId: "z_image", entryPoints, tree }),
    /never carry the literal "z_image"/,
  );
  // And it is for aliases only.
  assert.throws(
    () => assertModelIsNamedByEntryPoints({ model: "z_image_turbo:mlx", engineId: "z_image_turbo", entryPoints, tree }),
    /its own model id/,
  );
  assert.throws(
    () => assertModelIsNamedByEntryPoints({ model: "z_image_edit:mlx", engineId: "Z-Image Turbo", entryPoints, tree }),
    /malformed engineId/,
  );
  // The engine id is part of the hashed text; a declaration without one hashes exactly as before.
  const files = [["crates/x/src/model.rs", "s:abc"]];
  const plain = loaderClosureText({ model: "z_image_turbo:mlx", entryPoints, files });
  assert.ok(!plain.includes("# engine:"), "no alias line for a model that is its own engine");
  const aliased = loaderClosureText({ model: "z_image_edit:mlx", engineId: "z_image_turbo", entryPoints, files });
  assert.ok(aliased.includes("\n# engine: z_image_turbo\n"));
  assert.notEqual(aliased.replace("z_image_edit", "z_image_turbo"), plain, "the alias line keeps the two closures distinct even over identical files");
});

test("a declared entry point that is not shipped source is refused", { skip }, () => {
  const tree = gitTree(repo, PIN);
  const crates = firstPartyCrates(tree, tree.paths());
  assert.throws(
    () =>
      loaderClosureFiles({
        tree,
        crates,
        entryPoints: ["crates/media/mlx-gen/mlx-gen-ltx/tests/e2e_parity.rs"],
      }),
    /not shipped source/,
  );
});
