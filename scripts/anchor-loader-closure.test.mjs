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
// Those cases need a clone containing the pin; they SKIP where there is none (a CI lane that has
// not fetched inference). The pure-function cases beside them are hermetic and always run.
import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  ANCHOR_LOADER_CLOSURE_VERSION,
  buildAnchorLoaderConfig,
  expandUsePath,
  firstPartyCrates,
  gitTree,
  loaderClosureDigest,
  loaderClosureFiles,
  moduleDirOf,
  overlayTree,
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
const skip = repo ? false : `no inference clone containing ${PIN.slice(0, 8)}`;

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

test("a `use` item's brace groups expand to one edge each, aliases dropped", () => {
  assert.deepEqual(expandUsePath("a::b::{c, d::{e, f}, g as h}"), [
    ["a", "b", "c"],
    ["a", "b", "d", "e"],
    ["a", "b", "d", "f"],
    ["a", "b", "g"],
  ]);
  // The shape that matters: a naive `::` split would read `d::e` as a TOP-LEVEL path and resolve
  // it against the wrong module.
  assert.deepEqual(expandUsePath("mlx_gen::{Result, array::contiguous}"), [
    ["mlx_gen", "Result"],
    ["mlx_gen", "array", "contiguous"],
  ]);
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
test("a pin bump with unchanged loader source leaves the key unchanged", { skip }, () => {
  let measured;
  try {
    measured = keyAt(MEASURED_AT);
  } catch {
    return; // The clone is shallow and does not carry the measurement revision.
  }
  assert.notEqual(MEASURED_AT, PIN, "the two revisions must actually differ");
  assert.equal(measured.digest, keyAt(PIN).digest);
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
