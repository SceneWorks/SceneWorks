import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { CHECK_JOBS, assertCheckResults, checkDecision, embeddedRustInputs, selectCheckJobs } from "./check-job-relevance.mjs";

const root = process.cwd();
const embeddedRust = embeddedRustInputs(root);
const decision = (...files) => checkDecision(files, { embeddedRust });
const full = (value) => Object.keys(value).filter((key) => key !== "reason").every((key) => value[key] === true);

test("ordinary web-only changes skip native and historical jobs; embedded web source does not", () => {
  const ui = decision("apps/web/src/App.jsx");
  assert.equal(ui.web, true);
  assert.equal(ui.scaffold, true);
  assert.equal(ui.rust, false);
  assert.equal(ui.history, false);
  assert.ok(embeddedRust.size > 10, "read the real Rust include_str! inputs");
  for (const file of embeddedRust) assert.equal(decision(file).rust, true, file);
});

test("unconsumed docs-only changes skip build/test jobs", () => {
  const docs = decision("README.md", "docs/new-contributor-note.md");
  for (const key of ["web", "rust", "scaffold", "history", "docker"]) assert.equal(docs[key], false, key);
});

test("native, pin, anchor and tooling changes select their real dependent suites", () => {
  for (const file of ["crates/sceneworks-core/src/memory_anchor.rs", "Cargo.toml", "Cargo.lock", "config/memory-anchors.json", "docs/calibration/sc-18791/ltx25-mlx-evidence.seed.json", "scripts/anchor-loader-closure.mjs", "scripts/lib/source-revision.mjs", "release/starvector-terminal-campaign-v1.json"]) {
    const selected = decision(file);
    assert.equal(selected.rust, true, file);
    assert.equal(selected.scaffold, true, file);
    assert.equal(selected.history, true, file);
  }
});

test("cross-boundary inputs read by real web tests/builds remain reachable", () => {
  for (const file of ["config/manifests/builtin.models.jsonc", "config/engine-capabilities/capabilities.mlx.json", "packages/schemas/model-manifest.schema.json", "apps/desktop/Entitlements.plist", "apps/desktop/licenses/README.md", "crates/sceneworks-core/src/yue2_score/fixtures/score.abc", "tests/fixtures/moss-batch-recipe.json", "docs/workflow-share-envelope.md", "docs/film-script-writing.md", "docs/film-editor.md"]) {
    assert.equal(decision(file).web, true, file);
  }
  assert.equal(decision("docs/calibration-runbook.md").history, true);
  for (const file of ["docs/workflow-share-envelope.md", "docs/sc-17139-minimax-h3-hosting-layout.md"]) {
    assert.equal(decision(file).rust, true, file);
  }
  assert.equal(decision("docker/rust.Dockerfile").history, true);
  assert.equal(decision("docs/epic-20738-terminal-cuda.md").scaffold, true);
});

test("unknown, empty, malformed and failed diffs run all jobs", () => {
  for (const files of [[], ["new-build-system/input"], ["../escape"], ["docs/bad\npath.md"]]) assert.ok(full(checkDecision(files)));
  const event = { pull_request: { base: { sha: "a".repeat(40) }, head: { sha: "b".repeat(40) } } };
  for (const git of [() => { throw new Error("missing merge base"); }, () => "README.md", () => ""]) {
    assert.ok(full(selectCheckJobs({ root, eventName: "pull_request", event, git })));
  }
  assert.ok(full(selectCheckJobs({ root, eventName: "pull_request", event: {} })));
});

test("dispatch, main push and merge groups explicitly run full acceptance", () => {
  for (const eventName of ["workflow_dispatch", "push", "merge_group"]) {
    assert.ok(full(selectCheckJobs({ root, eventName, event: {}, git: () => assert.fail("must not diff") })));
  }
});

test("PR diffs use NUL-delimited no-renames three-dot input and include deleted inputs", () => {
  const event = { pull_request: { base: { sha: "a".repeat(40) }, head: { sha: "b".repeat(40) } } };
  const selected = selectCheckJobs({ root, eventName: "pull_request", event, git: (command, args) => {
    assert.equal(command, "git");
    assert.deepEqual(args, ["diff", "--name-only", "--no-renames", "-z", `${"a".repeat(40)}...${"b".repeat(40)}`]);
    return "docs/rename.md\0config/memory-anchors.json\0";
  } });
  assert.equal(selected.history, true);
});

function results(selected) {
  return { changes: { result: "success", outputs: Object.fromEntries(Object.keys(selected).filter((key) => key !== "reason").map((key) => [key, String(selected[key])])) }, "supply-chain": { result: "success" }, ...Object.fromEntries(Object.entries(CHECK_JOBS).map(([job, key]) => [job, { result: selected[key] ? "success" : "skipped" }])) };
}
test("parity permits only successful jobs or proven irrelevant skips", () => {
  const expected = results(decision("README.md"));
  assert.doesNotThrow(() => assertCheckResults(expected));
  assert.doesNotThrow(() => assertCheckResults(results(decision("Cargo.toml"))));
  for (const result of ["failure", "cancelled", "skipped"]) {
    const failed = results(decision("Cargo.toml")); failed["parity-history"].result = result;
    assert.throws(() => assertCheckResults(failed));
  }
  for (const mutation of [
    (value) => { delete value.changes; },
    (value) => { value.changes.result = "failure"; },
    (value) => { value.changes.result = "skipped"; },
    (value) => { delete value.changes.outputs.history; },
    (value) => { value.changes.outputs.rust = ""; },
    (value) => { value["supply-chain"].result = "skipped"; },
    (value) => { delete value["parity-history"]; },
    (value) => { value["parity-docker"].result = "failure"; },
    (value) => { value.newJob = { result: "success" }; },
  ]) {
    const value = structuredClone(expected); mutation(value);
    assert.throws(() => assertCheckResults(value));
  }
});

test("workflow retains required names, exact aggregator dependencies and fail-safe selectors", () => {
  const source = readFileSync(".github/workflows/check.yml", "utf8");
  assert.match(source, /^  workflow_dispatch:/m);
  assert.doesNotMatch(source.slice(0, source.indexOf("jobs:")), /- "codex\/\*\*"|paths:/);
  const needs = source.match(/^    needs: \[([^\]]+)\]/m)[1].split(", ").sort();
  assert.deepEqual(needs, ["changes", "supply-chain", ...Object.keys(CHECK_JOBS)].sort());
  assert.match(source, /run: node scripts\/check-job-relevance.mjs --assert-results/);
  for (const [job, output] of Object.entries(CHECK_JOBS)) {
    const block = source.split(`  ${job}:\n`)[1].split(/^  [\w-]+:\n/m)[0];
    assert.match(block, /needs: changes/);
    assert.ok(block.includes(`needs.changes.result != 'success' || needs.changes.outputs.${output} != 'false'`), job);
  }
});
