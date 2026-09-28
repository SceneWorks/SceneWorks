import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { produceUpstreamReferences, promoteUpstreamManifests, UPSTREAM_TIER_TIMEOUT_MS, validateUpstreamInputs } from "./starvector-terminal-upstream.mjs";
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const parityIndices = [0, 30, 60, 90].flatMap(start => Array.from({ length: 5 }, (_, offset) => start + offset));
const referenceRows = () => Array.from({ length: 120 }, (_, index) => ({ png_sha256: `input-${index}` }));

async function writeRejectedTier(output, tier, rows, rejectedIndex = 2) {
  const tierRoot = path.join(output, `upstream-${tier}`);
  await mkdir(tierRoot);
  const events = [];
  const rejections = [];
  for (let index = 0; index < 20; index += 1) {
    const source = parityIndices[index];
    const identity = { case_index: index, source_case_index: source, seed: index, input_png_sha256: rows[source].png_sha256 };
    const caseRoot = path.join(tierRoot, `case-${String(index).padStart(2, "0")}`);
    await mkdir(caseRoot);
    const raw = `<svg id="${tier}-${index}"/>`;
    await writeFile(path.join(caseRoot, "raw.svg"), raw);
    const sanitizer = index === rejectedIndex
      ? { outcome: "rejected", error_code: "provider SVG is invalid", canonical_svg_sha256: null, preview_png_sha256: null, published_paths: [], staging_residue: [], result_contains_inline_svg: false }
      : { outcome: "sanitized_inert" };
    await writeFile(path.join(caseRoot, "sanitizer.stdout.log"), JSON.stringify(sanitizer));
    await writeFile(path.join(caseRoot, "sanitizer.stderr.log"), "");
    events.push({ event: "case_started", ...identity });
    if (index === rejectedIndex) {
      const rejection = { ...identity, error_code: sanitizer.error_code, raw_svg_sha256: sha(raw), sanitizer_stdout: `upstream-${tier}/case-${String(index).padStart(2, "0")}/sanitizer.stdout.log`, sanitizer_stderr: `upstream-${tier}/case-${String(index).padStart(2, "0")}/sanitizer.stderr.log` };
      rejections.push(rejection);
      events.push({ event: "case_rejected", ...rejection });
    } else events.push({ event: "case_completed", ...identity });
  }
  events.push({ event: "failed", failure_kind: "svg_case_rejections", completed_cases: 19, collected_cases: 20, rejected_cases: rejections });
  await writeFile(path.join(tierRoot, "transcript.jsonl"), events.map(event => JSON.stringify(event)).join("\n") + "\n");
}
test("both upstream models validate entirely offline before caller claims an attempt", async () => {
  const calls = [];
  const reports = await validateUpstreamInputs({ sceneWorksRoot: "/repo", python: "/oracle/python", upstreamRoot: "/source", weightsRoot: "/weights", assetsRoot: "/assets", componentsRoot: "/components", sanitizer: "/sanitize" }, "/output", async (command, args, options) => {
    calls.push({ command, args, options }); return { stdout: JSON.stringify({ tier: args.at(-1), status: "validated" }) };
  });
  assert.deepEqual(reports.map(x => x.tier), ["1b", "8b"]);
  for (const {command,args,options} of calls) { assert.equal(command, "/oracle/python"); assert.equal(args[1], "validate"); assert.equal(options.env.HF_HUB_OFFLINE, "1"); assert.equal(options.env.TRANSFORMERS_OFFLINE, "1"); assert.ok(args.includes("--components-root")); }
});

test("the upstream tier supervisor permits every case to use the shipping generation ceiling", () => {
  const limitCases = JSON.parse(readFileSync("scripts/lib/starvector-terminal-limit-cases.json", "utf8"));
  const perCaseMs = limitCases.shipping_detail_budgets["8b"].maxWallTimeMs;
  assert.equal(perCaseMs, 300000);
  assert.ok(UPSTREAM_TIER_TIMEOUT_MS > 20 * perCaseMs);
});

test("one upstream job precedes all four native tuples and every tuple consumes its artifact", async () => {
  const { readFile } = await import("node:fs/promises");
  const workflow = await readFile(".github/workflows/starvector-terminal.yml", "utf8");
  assert.equal((workflow.match(/starvector-terminal-upstream\.mjs run/g) ?? []).length, 1);
  assert.match(workflow, /mlx-1b:\n    needs: upstream-reference/);
  assert.equal((workflow.match(/Download the shared upstream reference from this workflow attempt/g) ?? []).length, 4);
});

test("case-level rejection collects all twenty rows in both tiers and publishes no manifest", async t => {
  const output = await mkdtemp(path.join(tmpdir(), "upstream-collection-"));
  t.after(() => rm(output, { recursive: true, force: true }));
  const calls = [];
  const rows = referenceRows();
  const execute = async (command, args) => {
    const tier = args[args.indexOf("--tier") + 1];
    calls.push({ command, tier, deferred: args.includes("--defer-manifest") });
    await writeRejectedTier(output, tier, rows);
    const error = new Error(`${tier} collected rejection`); error.code = 3; throw error;
  };
  const options = { sceneWorksRoot: "/repo", python: "/oracle/python", upstreamRoot: "/source", weightsRoot: "/weights", assetsRoot: "/assets", componentsRoot: "/components", sanitizer: "/sanitize" };
  await assert.rejects(produceUpstreamReferences(options, output, { backend: "candle", device: "cuda:0" }, rows, execute), error => error.code === 3 && /1b, 8b/.test(error.message));
  assert.deepEqual(calls, [
    { command: "/oracle/python", tier: "1b", deferred: true },
    { command: "/oracle/python", tier: "8b", deferred: true },
  ]);
  for (const tier of ["1b", "8b"]) {
    assert.match(await readFile(path.join(output, `upstream-${tier}`, "case-19", "raw.svg"), "utf8"), new RegExp(`${tier}-19`));
    await assert.rejects(access(path.join(output, `upstream-reference-${tier}.json`)));
  }
  await assert.rejects(access(path.join(output, "upstream-controller.json")));
});

test("non-case oracle failure remains fail-fast and does not start the next tier", async () => {
  const calls = [];
  const failure = new Error("CUDA identity failed"); failure.code = 1;
  const execute = async (unusedCommand, args) => { calls.push(args[args.indexOf("--tier") + 1]); throw failure; };
  const options = { sceneWorksRoot: "/repo", python: "/oracle/python", upstreamRoot: "/source", weightsRoot: "/weights", assetsRoot: "/assets", componentsRoot: "/components", sanitizer: "/sanitize" };
  await assert.rejects(produceUpstreamReferences(options, "/output", { backend: "candle", device: "cuda:0" }, [], execute), /CUDA identity failed/);
  assert.deepEqual(calls, ["1b"]);
});

test("exit code three without a complete rejection transcript remains fail-fast", async t => {
  const output = await mkdtemp(path.join(tmpdir(), "upstream-unauthenticated-"));
  t.after(() => rm(output, { recursive: true, force: true }));
  const calls = [];
  const execute = async (unusedCommand, args) => { calls.push(args[args.indexOf("--tier") + 1]); const error = new Error("runtime exited 3"); error.code = 3; throw error; };
  const options = { sceneWorksRoot: "/repo", python: "/oracle/python", upstreamRoot: "/source", weightsRoot: "/weights", assetsRoot: "/assets", componentsRoot: "/components", sanitizer: "/sanitize" };
  await assert.rejects(produceUpstreamReferences(options, output, { backend: "candle", device: "cuda:0" }, referenceRows(), execute));
  assert.deepEqual(calls, ["1b"]);
});

test("a successful tier stays pending when the other tier reports case rejections", async t => {
  const output = await mkdtemp(path.join(tmpdir(), "upstream-mixed-"));
  t.after(() => rm(output, { recursive: true, force: true }));
  const rows = referenceRows();
  const execute = async (unusedCommand, args) => {
    const tier = args[args.indexOf("--tier") + 1];
    if (tier === "1b") { await writeRejectedTier(output, tier, rows); const error = new Error("collected rejection"); error.code = 3; throw error; }
    await writeFile(path.join(output, "upstream-reference-8b.pending.json"), "validated pending evidence\n");
  };
  const options = { sceneWorksRoot: "/repo", python: "/oracle/python", upstreamRoot: "/source", weightsRoot: "/weights", assetsRoot: "/assets", componentsRoot: "/components", sanitizer: "/sanitize" };
  await assert.rejects(produceUpstreamReferences(options, output, { backend: "candle", device: "cuda:0" }, rows, execute), error => error.code === 3);
  assert.equal(await readFile(path.join(output, "upstream-reference-8b.pending.json"), "utf8"), "validated pending evidence\n");
  await assert.rejects(access(path.join(output, "upstream-reference-8b.json")));
});

test("canonical manifests are promoted only after both pending manifests validate", async t => {
  const output = await mkdtemp(path.join(tmpdir(), "upstream-promotion-"));
  t.after(() => rm(output, { recursive: true, force: true }));
  const loaded = [];
  const execute = async (unusedCommand, args) => {
    const tier = args[args.indexOf("--tier") + 1];
    await writeFile(path.join(output, `upstream-reference-${tier}.pending.json`), `${tier}\n`);
  };
  const loadReference = async (root, tier, rows, flags) => { loaded.push({ root, tier, rows, flags }); };
  const options = { sceneWorksRoot: "/repo", python: "/oracle/python", upstreamRoot: "/source", weightsRoot: "/weights", assetsRoot: "/assets", componentsRoot: "/components", sanitizer: "/sanitize" };
  await produceUpstreamReferences(options, output, { backend: "candle", device: "cuda:0" }, ["row"], execute, loadReference);
  assert.deepEqual(loaded.map(item => [item.tier, item.flags]), [["1b", { pending: true }], ["8b", { pending: true }]]);
  assert.equal(await readFile(path.join(output, "upstream-reference-1b.json"), "utf8"), "1b\n");
  assert.equal(await readFile(path.join(output, "upstream-reference-8b.json"), "utf8"), "8b\n");
});

test("partial manifest promotion restores pending evidence even when rename rollback fails", async t => {
  const output = await mkdtemp(path.join(tmpdir(), "upstream-rollback-"));
  t.after(() => rm(output, { recursive: true, force: true }));
  for (const tier of ["1b", "8b"]) await writeFile(path.join(output, `upstream-reference-${tier}.pending.json`), `${tier}\n`);
  let rollbackFailed = false;
  const move = async (from, to) => {
    if (from.endsWith("8b.pending.json")) throw Object.assign(new Error("second promotion failed"), { code: "EIO" });
    if (from.endsWith("1b.json") && !rollbackFailed) { rollbackFailed = true; throw Object.assign(new Error("rename rollback failed"), { code: "EIO" }); }
    const { rename } = await import("node:fs/promises");
    await rename(from, to);
  };
  await assert.rejects(promoteUpstreamManifests(output, { rename: move }), /second promotion failed/);
  for (const tier of ["1b", "8b"]) {
    assert.equal(await readFile(path.join(output, `upstream-reference-${tier}.pending.json`), "utf8"), `${tier}\n`);
    await assert.rejects(access(path.join(output, `upstream-reference-${tier}.json`)));
  }
});
