// Contract tests for the download-pattern gate (sc-18854).
//
// The gate's own `--self-test` is the per-guard mutation harness; test 1 below runs it under
// `node --test` so a regression shows up in the same place as everything else. The tests after
// that are the ones the harness CANNOT give you, because they bind the gate to the REAL catalog
// and the REAL committed evidence rather than to synthetic inputs.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  EVIDENCE_FILE,
  KNOWN_ZERO_MATCHES,
  claimKey,
  collectClaims,
  gradeRecordedEvidence,
  patternToRegExp,
  runSelfTest,
} from "./check-download-patterns.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";
import {
  LTX_2_3_MLX_PRE_BF16_FILES,
  LTX_2_3_MLX_PRE_BF16_REVISION,
  LTX_2_3_MLX_REPO,
} from "./fixtures/ltx-2.3-mlx-pre-bf16-listing.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const readJsonc = async (rel) =>
  JSON.parse(stripJsoncComments(await readFile(path.join(root, rel), "utf8")));

async function realInputs() {
  const models = await readJsonc("config/manifests/builtin.models.jsonc");
  const loras = await readJsonc("config/manifests/builtin.loras.jsonc");
  const evidence = JSON.parse(await readFile(path.join(root, EVIDENCE_FILE), "utf8"));
  return { claims: collectClaims({ models, loras }), evidence };
}

test("the per-guard mutation harness passes", () => {
  const failures = runSelfTest({ log: () => {} });
  assert.deepEqual(failures, []);
});

// THE sc-18853 REGRESSION TEST, on real data, in both directions.
//
// A gate that passes both ways is worthless, so this asserts the same real manifest pattern
// against two real upstream listings of the same repo and requires opposite verdicts.
test("sc-18853: the real bf16/* row reds at the pin that predates the upload, and passes at the pin that has it", async () => {
  const { claims, evidence } = await realInputs();

  const bf16 = claims.find((claim) => claim.label === "ltx_2_3/bf16");
  // Fail loudly rather than vacuously if the row is ever renamed or dropped — a `find` that
  // returns undefined would otherwise turn this whole test into a no-op that still passes.
  assert.ok(bf16, "ltx_2_3/bf16 download row must exist for this regression test to mean anything");
  assert.equal(bf16.repo, LTX_2_3_MLX_REPO);
  assert.ok(bf16.declared.includes("bf16/*"), "the row must still declare bf16/*");

  // Guard the fixtures themselves, so a verdict flip cannot be caused by the listings being
  // the wrong way round.
  assert.equal(
    LTX_2_3_MLX_PRE_BF16_FILES.filter((file) => file.startsWith("bf16/")).length,
    0,
    "the pre-upload listing must genuinely hold no bf16/ path",
  );
  const currentEntry = evidence.repos.find(
    (entry) => entry.key === claimKey(bf16.repo, bf16.revision),
  );
  assert.ok(currentEntry, "the current ltx_2_3 pin must have a recorded listing");
  assert.ok(
    currentEntry.files.some((file) => file.startsWith("bf16/")),
    "the current pin's recorded listing must genuinely hold bf16/ paths",
  );

  // Direction 1 — the defect: the row pinned to 254989c3…, whose snapshot has no bf16/.
  const atBadPin = gradeRecordedEvidence({
    claims: [{ ...bf16, revision: LTX_2_3_MLX_PRE_BF16_REVISION, declared: ["bf16/*"] }],
    evidence: {
      repos: [
        {
          key: claimKey(LTX_2_3_MLX_REPO, LTX_2_3_MLX_PRE_BF16_REVISION),
          repo: LTX_2_3_MLX_REPO,
          revision: LTX_2_3_MLX_PRE_BF16_REVISION,
          resolvedSha: LTX_2_3_MLX_PRE_BF16_REVISION,
          files: LTX_2_3_MLX_PRE_BF16_FILES,
        },
      ],
    },
    waivers: [],
  });
  assert.ok(
    atBadPin.problems.some(
      (problem) => problem.kind === "zero-match" && problem.message.includes("bf16/*"),
    ),
    `expected a zero-match for bf16/* at the pre-upload pin, got ${JSON.stringify(atBadPin.problems)}`,
  );

  // Direction 2 — the fix: the same pattern at the pin that actually holds bf16/.
  const atGoodPin = gradeRecordedEvidence({
    claims: [{ ...bf16, declared: ["bf16/*"] }],
    evidence: { repos: [currentEntry] },
    waivers: [],
  });
  assert.deepEqual(atGoodPin.problems, []);
});

test("the committed evidence grades the real catalog clean", async () => {
  const { claims, evidence } = await realInputs();

  // Anti-collapse: `gradeRecordedEvidence` returns no problems for an EMPTY claim set, so a
  // regression that made `collectClaims` yield nothing (a JSONC parse change, a renamed
  // manifest key) would make the assertion below vacuously true. Pin that both manifest
  // sources are actually contributing, with floors rather than a census so an ordinary
  // catalog addition does not have to edit this test.
  assert.ok(
    claims.some((claim) => claim.label.startsWith("lora:")),
    "LoRA source.file/source.files claims must be collected",
  );
  assert.ok(
    claims.some((claim) => !claim.label.startsWith("lora:")),
    "model downloads[].files claims must be collected",
  );
  assert.ok(claims.length >= 250, `expected >=250 download entries, got ${claims.length}`);
  const patterns = claims.reduce((total, claim) => total + claim.declared.length, 0);
  assert.ok(patterns >= 350, `expected >=350 declared patterns, got ${patterns}`);
  assert.ok(
    (evidence.repos ?? []).length >= 90,
    `expected >=90 recorded repo@revision keys, got ${(evidence.repos ?? []).length}`,
  );

  const { problems, waived } = gradeRecordedEvidence({
    claims,
    evidence,
    waivers: KNOWN_ZERO_MATCHES,
  });
  assert.deepEqual(
    problems,
    [],
    `committed evidence must grade clean; re-record with \`node scripts/check-download-patterns.mjs --write\``,
  );
  // Every waiver must be live: `waiver-stale-unclaimed` above already reds an orphan, so an
  // equal count proves each declared waiver is genuinely absorbing a current zero-match rather
  // than sitting inert.
  assert.equal(waived.length, KNOWN_ZERO_MATCHES.length);
});

// The offline gate reuses `patternToRegExp`, so its semantics are now load-bearing for CI and
// not just for a manual pre-flight. These mirror the Rust `glob` crate's default MatchOptions
// used by the worker's `pattern_matches` (imports.rs): `*` and `?` DO cross `/`. A regex that
// stopped crossing `/` would under-report matches and invent failures the worker never raises;
// one that over-matched would hide a real zero-match.
test("glob translation matches the worker's Rust glob semantics", () => {
  assert.equal(patternToRegExp("q8/*").test("q8/vae/config.json"), true);
  assert.equal(patternToRegExp("q8/*").test("q8/transformer.safetensors"), true);
  assert.equal(patternToRegExp("q8/*").test("q4/transformer.safetensors"), false);
  assert.equal(patternToRegExp("bf16/*").test("bf16/vae_decoder.safetensors"), true);
  assert.equal(patternToRegExp("bf16/*").test("q8/vae_decoder.safetensors"), false);
  // Anchored at both ends: a declared file name must not match a longer sibling.
  assert.equal(patternToRegExp("demo.safetensors").test("demo.safetensors"), true);
  assert.equal(patternToRegExp("demo.safetensors").test("other-demo.safetensors"), false);
  // `.` is a literal, not "any character" — otherwise a typo'd glob would match by accident.
  assert.equal(patternToRegExp("demo.safetensors").test("demoXsafetensors"), false);
  assert.equal(patternToRegExp("a?c").test("abc"), true);
  assert.equal(patternToRegExp("a?c").test("ac"), false);
});

test("an entry with no revision keys on the default branch, and a pinned one keys on the pin", () => {
  assert.equal(claimKey("Org/demo", null), "Org/demo@main");
  assert.equal(claimKey("Org/demo", "a".repeat(40)), `Org/demo@${"a".repeat(40)}`);
});
