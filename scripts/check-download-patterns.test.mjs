// Contract tests for the download-pattern gate (sc-18854).
//
// The gate's own `--self-test` is the per-guard mutation harness; test 1 below runs it under
// `node --test` so a regression shows up in the same place as everything else. The tests after
// that are the ones the harness CANNOT give you, because they bind the gate to the REAL catalog
// and the REAL committed evidence rather than to synthetic inputs.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  EVIDENCE_FILE,
  KNOWN_REPO_CONDITIONS,
  KNOWN_ZERO_MATCHES,
  argumentError,
  claimKey,
  collectClaims,
  gradeRecordedEvidence,
  patternToRegExp,
  patternTranslationError,
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
    repoConditions: KNOWN_REPO_CONDITIONS,
  });
  assert.deepEqual(
    problems,
    [],
    `committed evidence must grade clean; re-record with \`node scripts/check-download-patterns.mjs --write\``,
  );
  // Every waiver must be live: the `*-stale-unclaimed` / `*-stale-cleared` guards above already
  // red an orphan, so an equal count proves each declared waiver is genuinely absorbing a current
  // defect rather than sitting inert.
  assert.equal(waived.length, KNOWN_ZERO_MATCHES.length + KNOWN_REPO_CONDITIONS.length);
});

// THE sc-18924 REGRESSION TEST. The recorded fixture is timeless only if every manifest claim
// names an immutable tree. Before this story, 11 real keys used a moving default branch and could
// stay falsely green after upstream changed because nothing forced a re-record.
//
// 95 -> 96 on the sc-19703 sync: sc-19708 declared `instantid_face_stack`
// (`SceneWorks/instantid-mlx`, the SCRFD + ArcFace pair), which landed on the epic branch while
// this gate landed on main, so neither PR could see the other. The census is a shape assertion —
// it moves whenever the catalog gains or loses a repo@revision key, together with a re-record.
test("all 96 real download keys are pinned to immutable lowercase commit SHAs", async () => {
  const { claims, evidence } = await realInputs();
  const immutableRevision = /^[0-9a-f]{40}$/u;
  const keys = new Set(claims.map((claim) => claimKey(claim.repo, claim.revision)));

  assert.equal(keys.size, 96, "update the 96/96 disclosure when the real key census changes");
  assert.equal(
    evidence.repos.length,
    96,
    "the evidence census must stay aligned with the 96/96 disclosure",
  );
  for (const claim of claims) {
    assert.match(
      claim.revision ?? "",
      immutableRevision,
      `${claim.label} must pin an immutable lowercase 40-hex revision`,
    );
  }
  for (const entry of evidence.repos) {
    assert.match(
      entry.revision ?? "",
      immutableRevision,
      `${entry.key} must not record a moving default branch`,
    );
    assert.equal(entry.key, claimKey(entry.repo, entry.revision));
  }
});

// ANTI-VACUITY for the two new repo-level guards.
//
// `evidence-gated` and `evidence-repo-id-mismatch` read fields the RECORDER writes. If the recorder
// ever stopped writing them, `entry.gated` would be `undefined` — falsy — and every gated repo
// would grade green again with no test failing anywhere. That is exactly the regression this whole
// review was about, so pin the fields' presence on the real artifact rather than trusting them.
test("the recorded evidence carries servedRepo and gated on every key", async () => {
  const { evidence } = await realInputs();
  const readable = evidence.repos.filter((entry) => !entry.error);
  assert.ok(readable.length >= 90, `expected >=90 readable keys, got ${readable.length}`);
  for (const entry of readable) {
    assert.ok(
      Object.hasOwn(entry, "servedRepo"),
      `${entry.key} has no servedRepo — re-record; without it evidence-repo-id-mismatch is inert`,
    );
    assert.ok(
      Object.hasOwn(entry, "gated"),
      `${entry.key} has no gated — re-record; without it evidence-gated is inert`,
    );
  }
});

// THE sc-18923 REGRESSION TEST, on the real immutable public rehost.
//
// The original defect was two gated repos grading green behind waivers. Both manifest claims must
// now resolve to the same exact public revision, each declared file must exist, and there must be
// no waiver capable of absorbing a future gate. Mutating that real listing back to gated must red.
test("the HDR and LipDub claims share one ungated immutable rehost with no gating waiver", async () => {
  const { claims, evidence } = await realInputs();

  const hdr = claims.find((claim) => claim.label === "lora:ltx_2_3_ic_hdr");
  const lipdub = claims.find((claim) => claim.label === "lora:ltx_2_3_ic_lipdub");
  assert.ok(hdr, "lora:ltx_2_3_ic_hdr must exist for this regression test to mean anything");
  assert.ok(lipdub, "lora:ltx_2_3_ic_lipdub must exist for this regression test to mean anything");
  assert.equal(hdr.repo, "SceneWorks/ltx-2.3-ic-loras");
  assert.equal(lipdub.repo, hdr.repo);
  assert.match(hdr.revision, /^[0-9a-f]{40}$/u);
  assert.equal(lipdub.revision, hdr.revision);
  const entry = evidence.repos.find((row) => row.key === claimKey(hdr.repo, hdr.revision));
  assert.ok(entry, "the shared IC-LoRA rehost must have a recorded listing");
  assert.equal(entry.servedRepo, hdr.repo, "the public rehost must not resolve through a rename");
  assert.equal(entry.resolvedSha, hdr.revision, "the recorded listing must bind the manifest SHA");
  assert.equal(entry.gated, false, "the replacement must remain publicly downloadable");
  for (const claim of [hdr, lipdub]) {
    assert.ok(entry.files.includes(claim.declared[0]), `${claim.label} file must exist at the pin`);
  }
  for (const file of [
    "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors",
    "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors",
    "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors",
    "LICENSE.md",
    "README.md",
  ]) {
    assert.ok(entry.files.includes(file), `the immutable public rehost must retain ${file}`);
  }
  assert.equal(
    KNOWN_REPO_CONDITIONS.filter(
      (waiver) => waiver.kind === "evidence-gated" && waiver.repo === hdr.repo,
    ).length,
    0,
    "the public replacement must never carry forward a gating waiver",
  );

  const clean = gradeRecordedEvidence({
    claims: [hdr, lipdub],
    evidence: { repos: [entry] },
    waivers: [],
    repoConditions: [],
  });
  assert.deepEqual(clean.problems, []);

  const regated = gradeRecordedEvidence({
    claims: [hdr, lipdub],
    evidence: { repos: [{ ...entry, gated: "auto" }] },
    waivers: [],
    repoConditions: [],
  });
  assert.deepEqual(
    regated.problems.map((problem) => problem.kind),
    ["evidence-gated"],
    `expected exactly evidence-gated, got ${JSON.stringify(regated.problems)}`,
  );
});

// The repo-id guard's reason for existing: it catches what the sha check structurally cannot.
//
// sc-18917 and sc-18926 fixed both live redirects at their sources. Keep the guard load-bearing by
// mutating the now-honest DFN5B listing back into its historical dead-name shape; this also proves
// the check reaches an UNREVISIONED redirected key, where `claim.revision &&` makes the sha guard
// inert outright rather than merely satisfied.
test("evidence-repo-id-mismatch catches synthetic pinned and unrevisioned redirects", async () => {
  const { claims, evidence } = await realInputs();

  const canonicalRepo = "apple/DFN5B-CLIP-ViT-H-14-378";
  const repo = "apple/DFN5B-CLIP-ViT-H-14-384";
  const servedBy = "apple/DFN5B-CLIP-ViT-H-14-378";
  const canonicalClaim = claims.find((row) => row.repo === canonicalRepo);
  assert.ok(canonicalClaim, `a manifest entry must declare ${canonicalRepo}`);
  const canonicalEntry = evidence.repos.find(
    (row) => row.key === claimKey(canonicalClaim.repo, canonicalClaim.revision),
  );
  assert.ok(canonicalEntry, `${canonicalRepo} must have a recorded listing`);
  assert.equal(canonicalEntry.servedRepo, canonicalRepo, "production evidence must be canonical");
  const claim = { ...canonicalClaim, repo };
  assert.ok(claim.revision, `${repo} must still be the PINNED shape for the first half to bind`);
  const entry = {
    ...canonicalEntry,
    key: claimKey(repo, claim.revision),
    repo,
  };
  assert.equal(entry.servedRepo, servedBy, `${repo} must genuinely still be served by ${servedBy}`);

  // Shape 1 — PINNED. The sha check passes, because an HF rename redirect preserves the sha. So
  // the redirect is invisible without the repo-id guard even when the entry IS pinned.
  const pinned = gradeRecordedEvidence({
    claims: [claim],
    evidence: { repos: [entry] },
    waivers: [],
    repoConditions: [],
  }).problems;
  assert.ok(
    pinned.some((problem) => problem.kind === "evidence-repo-id-mismatch"),
    `expected evidence-repo-id-mismatch for ${repo}, got ${JSON.stringify(pinned)}`,
  );
  assert.ok(
    !pinned.some((problem) => problem.kind === "evidence-revision-mismatch"),
    `${repo}: the sha check must NOT fire — that is why the repo-id guard is needed`,
  );

  // Shape 2 — MALFORMED AND UNREVISIONED. sc-18924 rejects this shape independently, but the
  // repo-id guard must still report the redirect rather than hiding it behind the pin failure.
  // Same real listing, re-keyed to `@main`; the sha guard remains unreachable because
  // `claim.revision &&` short-circuits before it can look.
  const unrevisionedKey = claimKey(repo, null);
  assert.notEqual(unrevisionedKey, entry.key, "the re-key must actually change the key");
  const unrevisioned = gradeRecordedEvidence({
    claims: [{ ...claim, revision: null }],
    evidence: { repos: [{ ...entry, key: unrevisionedKey, revision: null }] },
    waivers: [],
    repoConditions: [],
  }).problems;
  assert.ok(
    unrevisioned.some((problem) => problem.kind === "evidence-repo-id-mismatch"),
    `expected evidence-repo-id-mismatch unrevisioned, got ${JSON.stringify(unrevisioned)}`,
  );
  assert.ok(
    unrevisioned.some((problem) => problem.kind === "manifest-revision-required"),
    `expected the sc-18924 immutable-pin guard too, got ${JSON.stringify(unrevisioned)}`,
  );
  assert.ok(
    !unrevisioned.some((problem) => problem.kind === "evidence-revision-mismatch"),
    "unrevisioned: the sha-equality guard is inert; the revision and repo-id guards own this shape",
  );

  // The other direction, on both shapes: a listing served by the repo it names must NOT fire.
  // Without this the two assertions above would still pass if the guard fired unconditionally.
  for (const [name, graded] of [
    ["pinned", { claim, entry }],
    ["unrevisioned", { claim: { ...claim, revision: null }, entry: { ...entry, key: unrevisionedKey, revision: null } }],
  ]) {
    const honest = gradeRecordedEvidence({
      claims: [graded.claim],
      evidence: { repos: [{ ...graded.entry, servedRepo: graded.claim.repo }] },
      waivers: [],
      repoConditions: [],
    }).problems;
    assert.ok(
      !honest.some((problem) => problem.kind === "evidence-repo-id-mismatch"),
      `${name}: a repo serving its own listing must not fire, got ${JSON.stringify(honest)}`,
    );
    if (name === "unrevisioned") {
      assert.ok(
        honest.some((problem) => problem.kind === "manifest-revision-required"),
        "an honest servedRepo does not make a moving default branch immutable",
      );
    }
  }
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

// `**` is where the JS translation and the Rust glob DISAGREE, and it disagrees in the false-green
// direction, so the translator refuses instead of guessing.
//
// Rust: `glob::Pattern::new("a**b")` is an Err, and `pattern_matches` (imports.rs:262-269) does
// `.is_ok_and(...)`, so the pattern matches NOTHING and the worker hard-fails the download.
// The old JS translation produced `[\s\S]*[\s\S]*`, which matches EVERYTHING.
test("untranslatable globs are refused rather than mistranslated", () => {
  for (const pattern of ["q8/**", "***", "a**b", "**/model.safetensors"]) {
    assert.ok(
      patternTranslationError(pattern),
      `${pattern} must be refused — Rust matches nothing, a regex translation matches everything`,
    );
    assert.throws(() => patternToRegExp(pattern), /untranslatable glob/);
  }
  // An unterminated `[` is the same class: glob rejects it, the old fallback made it a literal.
  assert.ok(patternTranslationError("q8/[abc"));
  assert.throws(() => patternToRegExp("q8/[abc"), /untranslatable glob/);

  // The other direction — everything the catalog actually uses must still translate. A refusal
  // that was too broad would red the lane on valid globs, so this is not a one-way assertion.
  for (const pattern of ["q8/*", "bf16/*", "demo.safetensors", "a?c", "q8/[abc]/*", "*"]) {
    assert.equal(patternTranslationError(pattern), null, `${pattern} must still translate`);
    assert.doesNotThrow(() => patternToRegExp(pattern));
  }
});

// Anti-vacuity for the guard above: it is unreachable on today's catalog, so pin WHY it is
// unreachable rather than letting "no pattern uses `**`" silently become "some do, and they pass".
test("no declared pattern in the catalog uses a construct the translator refuses", async () => {
  const { claims } = await realInputs();
  const refused = [];
  for (const claim of claims) {
    for (const pattern of claim.declared) {
      if (patternTranslationError(pattern)) refused.push(`${claim.label}  ${pattern}`);
    }
  }
  assert.deepEqual(refused, [], `these declared patterns are untranslatable: ${refused.join(", ")}`);
});

test("an entry with no revision keys on the default branch, and a pinned one keys on the pin", () => {
  assert.equal(claimKey("Org/demo", null), "Org/demo@main");
  assert.equal(claimKey("Org/demo", "a".repeat(40)), `Org/demo@${"a".repeat(40)}`);
});

// sc-18854 review, MINOR 2. `--self-test` and `--check` used to `return` BEFORE the
// `--dry-run` rejection, so `--check --dry-run` exited 0 having graded offline with the flag
// ignored — the reviewer would believe they had verified the artifact against a live re-record
// when nothing had touched the network. Validation now runs before the mode dispatch.
//
// Each case is one guard, both directions: the accepted rows are what stops a rejection from
// being written too broadly.
test("argumentError rejects every flag combination a mode would otherwise swallow", () => {
  // Rejected — the two the reviewer named, plus the --write twins of the same swallow.
  for (const argv of [
    ["--check", "--dry-run"],
    ["--self-test", "--dry-run"],
    ["--check", "--write"],
    ["--self-test", "--write"],
    ["--check", "--write", "--dry-run"],
    ["--self-test", "--write", "--dry-run"],
  ]) {
    const error = argumentError(argv);
    assert.ok(error, `${argv.join(" ")} must be rejected, not silently ignored`);
    assert.match(error, /ignores/);
    assert.match(error, /--write --dry-run/, "the rejection must name the mode that does work");
  }

  // Rejected for the older reason: --dry-run belongs to the recorder.
  assert.match(argumentError(["--dry-run"]), /only applies to the recorder/);
  assert.match(argumentError(["--model", "ltx_2_3", "--dry-run"]), /only applies to the recorder/);
  assert.match(argumentError(["--check", "--model", "ltx_2_3"]), /grades the whole catalog/);

  // ACCEPTED — the other direction. A guard that rejected these would break `npm run check`,
  // the recorder, and the reviewer tool itself.
  for (const argv of [
    [],
    ["--check"],
    ["--self-test"],
    ["--write"],
    ["--write", "--dry-run"],
    ["--model", "ltx_2_3"],
  ]) {
    assert.equal(argumentError(argv), null, `${argv.join(" ") || "(no flags)"} must be accepted`);
  }
});

// The test above cannot see the DEFECT, and that is the whole point of this one. The bug was
// PLACEMENT, not the predicate: `argumentError`'s body was already correct where it sat, and
// moving the call back below the `--self-test`/`--check` dispatch leaves every assertion above
// green while `--check --dry-run` goes back to exiting 0 with the flag ignored. So drive the real
// CLI and assert the rejected combinations did no work at all.
const WORK_DONE = /graded \d+ pattern|checked \d+ pattern|self-test case\(s\) passed/;

test("the CLI rejects those combinations before any mode runs, not after", () => {
  for (const argv of [
    ["--check", "--dry-run"],
    ["--self-test", "--dry-run"],
    ["--check", "--write"],
    ["--self-test", "--write", "--dry-run"],
    ["--dry-run"],
  ]) {
    // No replay preload: if validation is doing its job, nothing here reaches the network.
    const run = spawnSync(process.execPath, ["scripts/check-download-patterns.mjs", ...argv], {
      cwd: root,
      encoding: "utf8",
    });
    const output = `${run.stdout}${run.stderr}`;
    assert.equal(run.status, 1, `${argv.join(" ")} must exit 1, got ${run.status}: ${output}`);
    assert.doesNotMatch(
      output,
      WORK_DONE,
      `${argv.join(" ")} must be rejected BEFORE the mode runs — this output shows it ran anyway`,
    );
  }

  // Both accepted offline modes must still do their work and exit 0, so the rejection above
  // cannot be passing by rejecting everything.
  for (const [argv, expected] of [
    [["--check"], /graded \d+ pattern/],
    [["--self-test"], /self-test case\(s\) passed/],
  ]) {
    const run = spawnSync(process.execPath, ["scripts/check-download-patterns.mjs", ...argv], {
      cwd: root,
      encoding: "utf8",
    });
    const output = `${run.stdout}${run.stderr}`;
    assert.equal(run.status, 0, `${argv.join(" ")} must still exit 0: ${output}`);
    assert.match(output, expected);
    assert.match(output, WORK_DONE);
  }
});

// sc-18854 review, MINOR 1. END-TO-END, because the defect was an exit-code COLLISION that no
// in-process test of a pure function can see: `runLive` wrote the live zero-match verdict and the
// artifact-comparison verdict to the same `process.exitCode`, so `--write --dry-run` exited 1 on a
// clean tree while printing "is UP TO DATE". The runbook documents that exit code as the one
// signal separating an honest re-record from a hand edit, so it was wrong on every clean run.
//
// The network is replayed from the committed artifact (see the fixture header), so this is offline
// and deterministic. Both directions are asserted, and the clean direction deliberately runs on a
// tree that DOES have a live zero-match (the tracked LipDub row) — without that the test would be
// vacuous, since it is the collision that has to not happen.
const PERTURBED_KEY = "SceneWorks/ltx-2.3-mlx@01df27d308466533aa09d251e3aebdcc627d07eb";
const runReplayed = (args, env = {}, cwd = root) => {
  const result = spawnSync(
    process.execPath,
    [
      "--import",
      "./scripts/fixtures/replay-download-pattern-network.mjs",
      "scripts/check-download-patterns.mjs",
      ...args,
    ],
    { cwd, encoding: "utf8", env: { ...process.env, ...env } },
  );
  return { status: result.status, output: `${result.stdout}${result.stderr}` };
};
const runDryRun = (env = {}) => runReplayed(["--write", "--dry-run"], env);

// A COMPLETE temporary catalog — the recorder, its one local import, the replay fixture, both
// manifests and the committed evidence — with ONE deliberate defect injected into a manifest entry.
//
// WHY THIS EXISTS (sc-18917). The collision test below needs a run in which a LIVE verdict fires
// while the recorded artifact is still byte-identical to a re-record. Until sc-18917 that
// combination was supplied for free by a real catalog defect — the LipDub row declared a file
// upstream had renamed away — and the test leaned on it, saying so in its own comment. Repointing
// that entry removes it, and it cannot simply be re-borrowed elsewhere: a live zero-match can only
// come from a listing that lacks a declared pattern, and that is the very listing the recorder
// transcribes, so ANY perturbation of the replayed network moves the artifact as well.
//
// The one input that moves a live verdict WITHOUT moving the artifact is the MANIFEST — a declared
// pattern matching nothing leaves every listing untouched — and the manifest path is fixed relative
// to the script, hence a temporary root. The result is a collision proof that holds permanently
// rather than one that expires the moment the catalog is healthy.
const INJECT_TARGET_ID = "ltx_2_3_ic_lipdub";
const INJECTED_PATTERN = "sw-injected-zero-match-sc-18917.safetensors";

async function withInjectedZeroMatch(run) {
  const dir = await mkdtemp(path.join(os.tmpdir(), "sw-download-patterns-"));
  try {
    await mkdir(path.join(dir, "scripts", "lib"), { recursive: true });
    await mkdir(path.join(dir, "scripts", "fixtures"), { recursive: true });
    await mkdir(path.join(dir, "config", "manifests"), { recursive: true });
    // `copyFile` for the evidence specifically: the whole point is that it stays byte-identical, so
    // a re-record inside the temp root reproduces it and the UP-TO-DATE branch is reachable.
    for (const rel of [
      "scripts/check-download-patterns.mjs",
      "scripts/lib/jsonc.mjs",
      "scripts/fixtures/replay-download-pattern-network.mjs",
      "config/download-pattern-evidence.json",
    ]) {
      await copyFile(path.join(root, rel), path.join(dir, rel));
    }

    const models = await readJsonc("config/manifests/builtin.models.jsonc");
    const loras = await readJsonc("config/manifests/builtin.loras.jsonc");
    const target = loras.loras.find((lora) => lora.id === INJECT_TARGET_ID);
    assert.ok(target, `${INJECT_TARGET_ID} must exist in the LoRA manifest`);
    const declared = target.source?.file;
    assert.ok(declared, `${INJECT_TARGET_ID} must declare a single source.file to widen`);
    // `source.files` is the manifest's own multi-pattern shape, so the injected catalog is still a
    // LEGAL one: the real declared file, plus one pattern that cannot match. Nothing about the
    // repo@revision key changes, so the replayed listing — and therefore the artifact — does not.
    delete target.source.file;
    target.source.files = [declared, INJECTED_PATTERN];
    // Written as plain JSON: `readManifests` strips JSONC comments before parsing, and comment-free
    // JSON is a fixed point of that. Comments carry no meaning to the script.
    await writeFile(
      path.join(dir, "config/manifests/builtin.models.jsonc"),
      JSON.stringify(models),
      "utf8",
    );
    await writeFile(
      path.join(dir, "config/manifests/builtin.loras.jsonc"),
      JSON.stringify(loras),
      "utf8",
    );
    return await run({ dir, declaredFile: declared });
  } finally {
    // In a `finally` because a failed assertion inside `run` is the likeliest exit path, and that
    // is exactly the path a leaked temp dir hides on.
    await rm(dir, { recursive: true, force: true });
  }
}

// The injection is only meaningful if the pattern genuinely matches nothing while the entry's REAL
// declared file genuinely matches. Assert both against the committed listing, so a stale fixture
// cannot turn the collision test into a run with two zero-matches or none.
test("the injected zero-match pattern is absent and the real declared file is present", async () => {
  const { claims, evidence } = await realInputs();
  const claim = claims.find((row) => row.label === `lora:${INJECT_TARGET_ID}`);
  assert.ok(claim, `lora:${INJECT_TARGET_ID} must exist for the injection to bind`);
  const entry = evidence.repos.find((row) => row.key === claimKey(claim.repo, claim.revision));
  assert.ok(entry, `${INJECT_TARGET_ID} must have a recorded listing`);
  assert.ok(!entry.files.includes(INJECTED_PATTERN), "the injected pattern must match nothing");
  assert.deepEqual(
    claim.declared.filter((pattern) => !entry.files.includes(pattern)),
    [],
    `every declared pattern of ${INJECT_TARGET_ID} must be present at its pinned revision`,
  );
});

test("--write --dry-run: the exit code carries the artifact verdict and only that", async () => {
  // Direction 1 — THE COLLISION ITSELF. A live zero-match fires while the artifact is unchanged,
  // so the exit code must be 0. This is the assertion the whole temp-root construction exists for.
  await withInjectedZeroMatch(async ({ dir }) => {
    const collided = runReplayed(["--write", "--dry-run"], {}, dir);
    assert.match(
      collided.output,
      /ZERO-MATCH PATTERNS \(1\)/,
      `the injected live zero-match must fire, or this proves nothing: ${collided.output}`,
    );
    assert.ok(
      collided.output.includes(INJECTED_PATTERN),
      "the zero-match must be the injected pattern, not some unrelated catalog drift",
    );
    assert.match(collided.output, /ARTIFACT VERDICT: UP TO DATE \(\d+ key\(s\)\) — exit 0/);
    assert.doesNotMatch(collided.output, /WOULD CHANGE/);
    assert.equal(
      collided.status,
      0,
      `a live zero-match must not move this mode's exit code: ${collided.output}`,
    );
  });

  // Direction 2 — the real tree, which sc-18917 left with no live zero-match at all.
  const clean = runDryRun();
  assert.match(clean.output, /ARTIFACT VERDICT: UP TO DATE \(\d+ key\(s\)\) — exit 0/);
  assert.doesNotMatch(clean.output, /WOULD CHANGE/);
  assert.equal(clean.status, 0, `a clean artifact must exit 0: ${clean.output}`);

  // Direction 3: one appended file in one replayed listing — the smallest possible divergence
  // between the committed artifact and a re-record — must red.
  const tampered = runDryRun({ REPLAY_EXTRA_FILE: PERTURBED_KEY });
  assert.match(tampered.output, /ARTIFACT VERDICT: WOULD CHANGE — exit 1/);
  assert.doesNotMatch(tampered.output, /UP TO DATE/);
  assert.equal(tampered.status, 1, "a divergent artifact must exit 1");
});

// The counterpart of the test above: routing the live verdicts through a flag must not have
// DROPPED them. Same replayed network, plain live mode (no --write, so no dry-run verdict owns the
// exit code), each verdict isolated to a single catalog entry and asserted in both directions.
test("plain live mode still reds on a zero-match and on an unreachable repo, and greens otherwise", async () => {
  const clean = runReplayed(["--model", "ltx_2_3"]);
  assert.match(clean.output, /Every declared download pattern matches at least one file\./);
  assert.equal(clean.status, 0, `a fully matching entry must exit 0: ${clean.output}`);

  await withInjectedZeroMatch(async ({ dir }) => {
    const zeroMatch = runReplayed(["--model", INJECT_TARGET_ID], {}, dir);
    // Exactly ONE: the injected pattern. Two would mean the entry's real declared file stopped
    // matching, which is the sc-18917 defect coming back.
    assert.match(zeroMatch.output, /ZERO-MATCH PATTERNS \(1\)/);
    assert.ok(zeroMatch.output.includes(INJECTED_PATTERN));
    assert.equal(zeroMatch.status, 1, "a zero-match must still red the live mode");

    // Same temp root, a different entry: still green. Without this the red above could be caused
    // by the temporary catalog being broken rather than by the injected pattern.
    const greenInTemp = runReplayed(["--model", "ltx_2_3"], {}, dir);
    assert.match(greenInTemp.output, /Every declared download pattern matches at least one file\./);
    assert.equal(greenInTemp.status, 0, `the temp root must otherwise be healthy: ${greenInTemp.output}`);
  });

  // Same entry as the green run, with only its listing made unreadable.
  const outage = runReplayed(["--model", "ltx_2_3"], { REPLAY_ERROR_KEY: PERTURBED_KEY });
  assert.match(outage.output, /UNREACHABLE \(could not verify/);
  assert.match(outage.output, /HTTP 503/);
  assert.equal(outage.status, 1, "an unreachable repo must still red the live mode");
});

// The fixture is only meaningful if the replay is faithful, and it is only faithful if the key it
// perturbs above genuinely exists. A typo'd key would make the perturbation a no-op and the
// direction-2 assertion above would then be testing nothing.
test("the replay perturbation targets a key that is actually recorded", async () => {
  const { evidence } = await realInputs();
  assert.ok(
    evidence.repos.some((entry) => entry.key === PERTURBED_KEY),
    `the perturbed key ${PERTURBED_KEY} must exist in the committed evidence`,
  );
});
