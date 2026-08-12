// SceneWorks catalog guard: every declared download pattern must match a real file
// (sc-12283, epic 8506 "Catalog-wide quant matrix"; wired into CI by sc-18854).
//
// WHY THIS EXISTS
// ---------------
// A manifest `downloads[]` entry scopes what to fetch with a `files` glob list, e.g.
//
//   "files": ["q8/transformer/*", "q8/text_encoder/*", "q8/vae/*", "q8/tokenizer/*", ...]
//
// The worker's filter ORs across that list (`allow_pattern_matches`), so a pattern
// matching NOTHING used to be invisible: the tier downloaded, the job completed, and
// an install marker was written for a tier missing a whole component — "installed" by
// every marker we keep, and unloadable in practice. That is the shape behind
// SceneWorks#850's "tokenizer: No such file or directory (os error 2)".
//
// As of sc-12283 the worker HARD-FAILS a download when any single declared pattern
// matches zero files. That is the right behavior — a partial install is worse than a
// clear error — but it moves the cost of a bad entry onto the USER, who sees a failed
// download. This script moves it back to authoring time.
//
// Covers builtin.models.jsonc `downloads[].files` AND builtin.loras.jsonc
// `source.file` / `source.files` — the LoRA download path gained the same hard-fail in
// sc-12288, so it needs the same pre-flight.
//
// HOW IT RUNS IN CI (sc-18854)
// ----------------------------
// The matching needs a file listing, and the only authority for that is the Hugging Face
// API — so the check cannot be BOTH live and hermetic. sc-18854 split it in two along the
// same seam `scan-inference-provenance.mjs` → `config/inference-provenance-candidates.tsv`
// → `check-license-coverage.mjs` already uses in this repo: record the network answer into
// a committed artifact, and have CI grade that artifact offline.
//
//   --write   RECORDER. Talks to HF. Fetches one listing per `repo@revision` key and
//             writes `config/download-pattern-evidence.json`. Run by a human/agent when a
//             `downloads[]` or LoRA `source` entry is added, re-pinned, or re-hosted.
//   --check   GATE. Zero network. Re-derives the claim set from the manifests and grades
//             it against the committed listings. Runs on every PR inside `npm run check`
//             (→ check.yml `parity-scaffold` → the required `parity` aggregator).
//
// The recorder is deliberately DUMB and the gate is deliberately SMART: `--write` only
// transcribes listings, and `--check` owns every verdict. There is no way to record a
// passing snapshot for a failing pattern, because the recorder never evaluates a pattern.
//
// WHY THE FIXTURE CANNOT SILENTLY GO STALE
// ----------------------------------------
// The usual objection to a recorded fixture is decay. It does not apply here, for two
// structural reasons:
//
//  1. **83 of 96 keys are pinned to an immutable 40-hex revision.** A git SHA's file listing
//     is timeless — re-reading it in a year returns the same bytes. There is nothing to
//     decay. The 13 unrevisioned keys read a moving default branch and DO carry residual
//     risk; each records the `resolvedSha` it actually read so a re-record shows the drift
//     in the diff. (Pinning those is tracked per-entry, e.g. sc-18917.)
//  2. **Coverage is graded as a set, in both directions.** Adding an entry, or re-pinning
//     one, changes its `repo@revision` key, and a key with no recorded listing is a hard
//     failure naming the re-record command. Conversely a recorded key that nothing claims
//     any more is also a failure. So the fixture cannot drift out of alignment with the
//     catalog without the gate saying so — no digest needed, because the set comparison is
//     strictly more precise and names the offending key.
//
// That is why this is not a "detection lags by a cron interval" design: the gate fires on
// the PR that introduces the bad entry, which is strictly earlier than any scheduled lane
// could manage, and it does it without putting huggingface.co on a required context.
//
// This is exactly the sc-18853 shape: `ltx_2_3`'s bf16 row declared `bf16/*` at revision
// `254989c3…`, three weeks before `bf16/` was uploaded. Under this gate that row is a new
// claim key, the re-record resolves `bf16/*` at `254989c3…` to zero files, and `--check`
// reds the PR — instead of `hf download --include 'bf16/*'` exiting 0 having fetched
// nothing and the engine failing later with `missing transformer.safetensors`.
//
// USAGE
//   node scripts/check-download-patterns.mjs             # LIVE pre-flight, all HF entries
//   node scripts/check-download-patterns.mjs --model krea_2_raw
//   node scripts/check-download-patterns.mjs --write     # LIVE + record the evidence file
//   node scripts/check-download-patterns.mjs --check     # OFFLINE gate (this is what CI runs)
//   node scripts/check-download-patterns.mjs --self-test # mutation harness for the gate
//
// Exits non-zero if any declared pattern matches zero files. Public repos need no
// auth; a token is picked up from $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN if set, so a
// repo that is later gated still resolves. Note the metadata API answers 200 for a
// GATED repo, so a green run does not prove a repo is fetchable without a token.

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { stripJsoncComments } from "./lib/jsonc.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_MANIFEST = "config/manifests/builtin.models.jsonc";
const LORA_MANIFEST = "config/manifests/builtin.loras.jsonc";
export const EVIDENCE_FILE = "config/download-pattern-evidence.json";
const RECORD_COMMAND = "node scripts/check-download-patterns.mjs --write";

const STORY_REF_PATTERN = /^sc-\d+$/;

// Known, TRACKED zero-matches. Not a licence — a debt with an owner.
//
// A zero-match here is a real catalog defect that we have decided not to fix in the change
// that discovered it. Carrying it as data keeps the lane honest (green means "no NEW zero
// match") without pretending the defect is gone. Each entry is keyed by the full
// label + repo + revision + pattern tuple, so it cannot silently widen into a blanket
// exemption for an entry that is later re-pinned or re-globbed.
//
// It is SELF-EXPIRING IN BOTH DIRECTIONS, following `PENDING_RUNG4_SURVEYS` in
// `generate-memory-matrix.mjs`: the gate fails if a waived pattern starts matching (the
// debt was paid — delete the waiver) and fails if the waived tuple is no longer claimed by
// any manifest entry (the entry moved — re-key or delete the waiver). Deliberately NOT a
// date-based expiry: a calendar expiry turns a green lane red on a day when nothing
// changed, which is the same "required context reds for a reason unrelated to this PR"
// failure mode this design exists to avoid. Structural expiry fires exactly when the
// underlying fact changes, which is the only moment the entry is actually wrong.
export const KNOWN_ZERO_MATCHES = [
  {
    label: "lora:ltx_2_3_ic_lipdub",
    repo: "Lightricks/LTX-2.3-22b-IC-LoRA-LipDub",
    revision: null,
    pattern: "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors",
    story: "sc-18917",
    reason:
      "Upstream renamed the repo LipDub -> DubIt and the weight file with it; the old path " +
      "still resolves through HF's rename redirect, so the listing returns 200 but holds " +
      "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors instead. The DubIt repo is also access-gated, " +
      "so correcting the filename alone may not make it fetchable. Fixing the manifest entry " +
      "is sc-18917; it must delete this waiver in the same change.",
  },
];

// Glob semantics must mirror the worker's `pattern_matches` (imports.rs), which uses the
// Rust `glob` crate with default MatchOptions — `*` and `?` DO cross `/` there
// (require_literal_separator is false), so a `q8/*` pattern legitimately matches
// `q8/vae/config.json`. Translating to a regex without that behavior would under-report
// matches and produce false failures here that the worker would never raise.
//
// The offline gate reuses this exact function rather than reimplementing the semantics —
// a second implementation would drift, and it would drift silently in the passing
// direction (a looser regex matches more, so a real zero-match reads as a hit).
export function patternToRegExp(pattern) {
  let out = "";
  for (let index = 0; index < pattern.length; index += 1) {
    const char = pattern[index];
    if (char === "*") {
      out += "[\\s\\S]*";
    } else if (char === "?") {
      out += "[\\s\\S]";
    } else if (char === "[") {
      const close = pattern.indexOf("]", index + 1);
      if (close === -1) {
        out += "\\[";
      } else {
        let body = pattern.slice(index + 1, close);
        if (body.startsWith("!")) body = `^${body.slice(1)}`;
        out += `[${body}]`;
        index = close;
      }
    } else {
      out += char.replace(/[.+^${}()|\\]/g, "\\$&");
    }
  }
  return new RegExp(`^${out}$`);
}

// The unit of evidence. An entry with no `revision` reads the default branch, which is what
// the worker fetches for it, so "main" is the honest key rather than a missing one.
export function claimKey(repo, revision) {
  return `${repo}@${revision ?? "main"}`;
}

function waiverKey({ label, repo, revision, pattern }) {
  // Joined on an ESCAPED NUL so the key is unambiguous: repos, labels and globs all
  // legitimately contain "/", "@", ":" and spaces, so any printable separator could be
  // forged by a crafted pattern into colliding with a different tuple. Written as the
  // escape sequence and never as a literal NUL byte -- a literal one makes grep classify
  // the file as binary and silently report "no match" for every later search of it.
  return [label, claimKey(repo, revision), pattern].join("\u0000");
}

// Flatten both manifests to a common shape: { id, label, repo, revision, declared[] }. A model
// declares `downloads[].files`; a LoRA declares `source.file` (one) or `source.files` (a list).
//
// An empty declaration is a deliberate whole-repo fetch, not an omission — there is no
// per-pattern claim to verify, so it produces no claim at all. (The worker's aggregate
// zero-file check still covers an empty repo at download time.)
export function collectClaims({ models, loras }) {
  const claims = [];
  for (const model of models.models ?? []) {
    for (const download of model.downloads ?? []) {
      if (download.provider !== "huggingface") continue;
      const declared = download.files ?? [];
      if (declared.length === 0) continue;
      claims.push({
        id: model.id,
        label: `${model.id}/${download.variant ?? "-"}`,
        repo: download.repo,
        revision: download.revision ?? null,
        declared,
      });
    }
  }
  for (const lora of loras.loras ?? []) {
    const source = lora.source ?? {};
    const provider = source.provider ?? lora.provider;
    const repo = source.repo ?? lora.repo;
    if (provider !== "huggingface" || !repo) continue;
    const single = source.file ?? lora.file;
    const declared = single ? [single] : (source.files ?? lora.files ?? []);
    if (declared.length === 0) continue;
    claims.push({
      id: lora.id,
      label: `lora:${lora.id}`,
      repo,
      revision: source.revision ?? lora.revision ?? null,
      declared,
    });
  }
  return claims;
}

// THE GATE. Pure: no network, no filesystem, no clock. Every verdict the CI lane reaches is
// reached here, which is what lets `--self-test` mutate one input at a time and prove each
// guard fires on its own rather than proving the set is collectively load-bearing.
//
// Returns { problems, waived }. `problems` non-empty means exit 1.
export function gradeRecordedEvidence({ claims, evidence, waivers }) {
  const problems = [];
  const waived = [];
  const fail = (kind, message) => problems.push({ kind, message });

  // Waiver shape first: an allowlist without a reason is a catch-all wearing a decision's
  // clothes, and one without a story reference has no owner.
  const waiverIndex = new Map();
  for (const waiver of waivers ?? []) {
    const label = waiver?.label ?? "<unlabelled>";
    const pattern = waiver?.pattern ?? "<no pattern>";
    if (typeof waiver?.reason !== "string" || waiver.reason.trim() === "") {
      fail("waiver-malformed", `${label}  ${pattern}  waiver has no reason`);
      continue;
    }
    if (!STORY_REF_PATTERN.test(waiver?.story ?? "")) {
      fail(
        "waiver-malformed",
        `${label}  ${pattern}  waiver story must look like sc-NNNN, got ${JSON.stringify(waiver?.story ?? null)}`,
      );
      continue;
    }
    waiverIndex.set(waiverKey(waiver), waiver);
  }

  const recorded = new Map((evidence?.repos ?? []).map((entry) => [entry.key, entry]));
  const claimedKeys = new Set();
  const claimedWaiverKeys = new Set();

  for (const claim of claims) {
    const key = claimKey(claim.repo, claim.revision);
    claimedKeys.add(key);
    const at = claim.revision ? `@${claim.revision.slice(0, 12)}` : "";
    const entry = recorded.get(key);

    if (!entry) {
      fail(
        "evidence-missing-key",
        `${claim.label}  ${key}  no recorded listing — re-record with \`${RECORD_COMMAND}\``,
      );
      continue;
    }
    if (entry.error) {
      // An unreachable repo is not a pass: we made no claim about it either way.
      fail(
        "evidence-unreachable",
        `${claim.label}  ${key}  recorded as unreachable (${entry.error}) — set $HF_TOKEN if gated and re-record`,
      );
      continue;
    }
    // The recorder must have read the snapshot it claims to have read. Guards against an
    // HF redirect or a recorder bug silently substituting a different tree — which is the
    // exact failure mode (default branch standing in for a pin) that sc-18809 fixed.
    if (claim.revision && entry.resolvedSha && entry.resolvedSha !== claim.revision) {
      fail(
        "evidence-revision-mismatch",
        `${claim.label}  ${key}  recorded listing resolved to ${entry.resolvedSha.slice(0, 12)} — re-record with \`${RECORD_COMMAND}\``,
      );
      continue;
    }

    const files = entry.files ?? [];
    for (const pattern of claim.declared) {
      const key2 = waiverKey({
        label: claim.label,
        repo: claim.repo,
        revision: claim.revision,
        pattern,
      });
      claimedWaiverKeys.add(key2);
      const regexp = patternToRegExp(pattern);
      const matched = files.some((file) => regexp.test(file));
      const waiver = waiverIndex.get(key2);

      if (matched) {
        if (waiver) {
          fail(
            "waiver-stale-now-matches",
            `${claim.label}  ${claim.repo}${at}  ${pattern}  now matches — delete the ${waiver.story} waiver`,
          );
        }
        continue;
      }
      if (waiver) {
        waived.push(`${claim.label}  ${claim.repo}${at}  ${pattern}  (${waiver.story})`);
        continue;
      }
      fail("zero-match", `${claim.label}  ${claim.repo}${at}  ${pattern}`);
    }
  }

  for (const key of recorded.keys()) {
    if (claimedKeys.has(key)) continue;
    fail(
      "evidence-orphan-key",
      `${key}  recorded but no manifest entry claims it — re-record with \`${RECORD_COMMAND}\``,
    );
  }
  for (const [key, waiver] of waiverIndex) {
    if (claimedWaiverKeys.has(key)) continue;
    fail(
      "waiver-stale-unclaimed",
      `${waiver.label}  ${waiver.pattern}  no manifest entry declares this any more — delete the ${waiver.story} waiver`,
    );
  }

  return { problems, waived };
}

async function readManifests() {
  const models = JSON.parse(
    stripJsoncComments(await readFile(path.join(root, MODEL_MANIFEST), "utf8")),
  );
  const loras = JSON.parse(
    stripJsoncComments(await readFile(path.join(root, LORA_MANIFEST), "utf8")),
  );
  return { models, loras };
}

const fileCache = new Map();

// The file list of `repo` AT `revision` — the entry's own pin, not the default branch (sc-18809).
//
// This used to fetch `/api/models/<repo>` unqualified, which lists whatever `main` holds today. A
// download entry pinned to a revision that PREDATES its own files therefore passed: the glob matched
// on `main`, while the pinned snapshot had no such path, so the fetch resolved zero files and the
// worker's hard-fail fired at the user instead. That is not hypothetical — `ltx_2_3`'s bf16 row
// declared `bf16/*` at `254989c3…`, three weeks before `bf16/` was uploaded, and this gate passed it.
// The pin is the thing the worker actually downloads, so it is the thing to verify.
async function repoFiles(repo, revision) {
  const key = claimKey(repo, revision);
  if (fileCache.has(key)) return fileCache.get(key);
  const token = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN;
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  // `/api/models/<repo>/revision/<rev>` is the revision-qualified twin of `/api/models/<repo>`; an
  // entry with no `revision` keeps the unqualified (default-branch) reading, which is what it fetches.
  const url = revision
    ? `https://huggingface.co/api/models/${repo}/revision/${encodeURIComponent(revision)}?blobs=false`
    : `https://huggingface.co/api/models/${repo}?blobs=false`;
  const response = await fetch(url, { headers });
  if (!response.ok) {
    const result = { error: `HTTP ${response.status}` };
    fileCache.set(key, result);
    return result;
  }
  const body = await response.json();
  const result = {
    resolvedSha: body.sha ?? null,
    files: (body.siblings ?? []).map((sibling) => sibling.rfilename).sort(),
  };
  fileCache.set(key, result);
  return result;
}

// LIVE mode: resolve every claim against the HF API. `--write` additionally transcribes the
// listings to the evidence file. Note the recorder writes what it read even when a pattern
// matches nothing — the verdict belongs to `--check`, not to the transcription step, so a
// failing catalog still produces an accurate (and reviewable) snapshot.
async function runLive({ only, write }) {
  const { models, loras } = await readManifests();
  const claims = collectClaims({ models, loras }).filter(
    (claim) => !only || claim.id === only,
  );

  const failures = [];
  const unreachable = [];
  const recorded = new Map();
  let entries = 0;
  let patterns = 0;

  for (const claim of claims) {
    const key = claimKey(claim.repo, claim.revision);
    const at = claim.revision ? `@${claim.revision.slice(0, 12)}` : "";
    const { files, error, resolvedSha } = await repoFiles(claim.repo, claim.revision);
    if (!recorded.has(key)) {
      recorded.set(
        key,
        error
          ? { key, repo: claim.repo, revision: claim.revision, error }
          : { key, repo: claim.repo, revision: claim.revision, resolvedSha, files },
      );
    }
    if (error) {
      unreachable.push(`${claim.label}  ${claim.repo}${at}  (${error})`);
      continue;
    }
    entries += 1;
    for (const pattern of claim.declared) {
      patterns += 1;
      const regexp = patternToRegExp(pattern);
      if (!files.some((file) => regexp.test(file))) {
        failures.push(`${claim.label}  ${claim.repo}${at}  ${pattern}`);
      }
    }
  }

  console.log(`checked ${patterns} pattern(s) across ${entries} download entr(ies)`);

  if (write) {
    if (only) {
      console.error(
        "\n--write records the WHOLE catalog; combining it with --model would write a snapshot" +
          " that the offline gate then rejects as missing every other key.",
      );
      process.exitCode = 1;
      return;
    }
    const repos = [...recorded.values()].sort((a, b) => (a.key < b.key ? -1 : 1));
    // No timestamp: the content is a pure function of the catalog and upstream state, so a
    // re-record with nothing changed is a no-op diff. `git log` on the file answers "when",
    // and `resolvedSha` answers the more useful "which tree" for unrevisioned entries.
    await writeFile(
      path.join(root, EVIDENCE_FILE),
      `${JSON.stringify({ repos }, null, 2)}\n`,
      "utf8",
    );
    console.log(`wrote ${EVIDENCE_FILE} (${repos.length} repo@revision key(s))`);
  }

  if (unreachable.length > 0) {
    console.log(`\nUNREACHABLE (could not verify — set $HF_TOKEN if these are gated):`);
    for (const line of unreachable) console.log(`  ${line}`);
  }
  if (failures.length > 0) {
    console.error(
      `\nZERO-MATCH PATTERNS (${failures.length}) — the worker will hard-fail these downloads:`,
    );
    for (const line of failures) console.error(`  ${line}`);
    console.error(
      `\nEither the glob is wrong, the tier is not published yet, or the entry's pinned revision` +
        ` predates the files it declares (sc-18809 — the trailing @<rev> above is what was checked).`,
    );
    process.exitCode = 1;
    return;
  }
  if (unreachable.length > 0) {
    process.exitCode = 1;
    return;
  }
  console.log("\nEvery declared download pattern matches at least one file.");
}

// OFFLINE gate. This is the CI path; it must not perform any network I/O.
async function runCheck() {
  const { models, loras } = await readManifests();
  const claims = collectClaims({ models, loras });
  let evidence;
  try {
    evidence = JSON.parse(await readFile(path.join(root, EVIDENCE_FILE), "utf8"));
  } catch (error) {
    console.error(
      `Could not read ${EVIDENCE_FILE}: ${error.message}\nRecord it with \`${RECORD_COMMAND}\`.`,
    );
    process.exitCode = 1;
    return;
  }

  const { problems, waived } = gradeRecordedEvidence({
    claims,
    evidence,
    waivers: KNOWN_ZERO_MATCHES,
  });
  const patterns = claims.reduce((total, claim) => total + claim.declared.length, 0);
  console.log(
    `graded ${patterns} pattern(s) across ${claims.length} download entr(ies)` +
      ` against ${(evidence.repos ?? []).length} recorded repo@revision key(s) — no network`,
  );
  if (waived.length > 0) {
    console.log(`\nWAIVED ZERO-MATCHES (${waived.length}) — tracked, not fixed:`);
    for (const line of waived) console.log(`  ${line}`);
  }
  if (problems.length > 0) {
    console.error(`\nPROBLEMS (${problems.length}):`);
    for (const problem of problems) console.error(`  [${problem.kind}] ${problem.message}`);
    process.exitCode = 1;
    return;
  }
  console.log("\nEvery declared download pattern matches a recorded file at its pinned revision.");
}

// ---------------------------------------------------------------------------------------
// Mutation harness. Each case perturbs ONE input and asserts the ONE guard it targets
// fires — proving each guard is individually load-bearing, not that the set collectively
// is. The baseline case proves the same inputs pass unmutated, so a guard that fired
// unconditionally would be caught too.
//
// The inputs are synthetic rather than the real catalog on purpose: the live catalog has
// no `evidence-revision-mismatch` and no `waiver-stale-now-matches` instance, so an
// assertion written against real data would be vacuously true — passing because the case
// cannot occur, not because the guard works.
// ---------------------------------------------------------------------------------------

const SELF_TEST_CLAIMS = [
  {
    id: "demo",
    label: "demo/q8",
    repo: "Org/demo",
    revision: "a".repeat(40),
    declared: ["q8/transformer/*", "q8/vae/*"],
  },
  {
    id: "demo_lora",
    label: "lora:demo_lora",
    repo: "Org/demo-lora",
    revision: null,
    declared: ["demo.safetensors"],
  },
];

const SELF_TEST_EVIDENCE = {
  repos: [
    {
      key: `Org/demo@${"a".repeat(40)}`,
      repo: "Org/demo",
      revision: "a".repeat(40),
      resolvedSha: "a".repeat(40),
      files: ["q8/transformer/model.safetensors", "q8/vae/config.json"],
    },
    {
      key: "Org/demo-lora@main",
      repo: "Org/demo-lora",
      revision: null,
      resolvedSha: "b".repeat(40),
      files: ["demo.safetensors"],
    },
  ],
};

const SELF_TEST_WAIVERS = [];

const clone = (value) => JSON.parse(JSON.stringify(value));

// Each case: mutate a deep copy of the baseline, expect exactly `kind` to appear.
const SELF_TEST_CASES = [
  {
    name: "zero-match: a declared pattern matching no recorded file fails (the sc-18853 guard)",
    kind: "zero-match",
    mutate: (input) => {
      // Exactly sc-18853: the tier is declared, the pin predates the upload, so the
      // recorded listing at that pin has no bf16/ path.
      input.claims[0].declared.push("bf16/*");
    },
  },
  {
    name: "evidence-missing-key: re-pinning an entry without re-recording fails",
    kind: "evidence-missing-key",
    mutate: (input) => {
      input.claims[0].revision = "c".repeat(40);
    },
  },
  {
    name: "evidence-orphan-key: a recorded key nothing claims fails",
    kind: "evidence-orphan-key",
    mutate: (input) => {
      input.evidence.repos.push({
        key: "Org/gone@main",
        repo: "Org/gone",
        revision: null,
        resolvedSha: "d".repeat(40),
        files: ["x"],
      });
    },
  },
  {
    name: "evidence-unreachable: a repo the recorder could not read is not a pass",
    kind: "evidence-unreachable",
    mutate: (input) => {
      input.evidence.repos[0] = {
        key: input.evidence.repos[0].key,
        repo: "Org/demo",
        revision: "a".repeat(40),
        error: "HTTP 401",
      };
    },
  },
  {
    name: "evidence-revision-mismatch: a listing recorded from a different tree fails",
    kind: "evidence-revision-mismatch",
    mutate: (input) => {
      input.evidence.repos[0].resolvedSha = "e".repeat(40);
    },
  },
  {
    name: "waiver-stale-now-matches: a waiver whose pattern now matches fails",
    kind: "waiver-stale-now-matches",
    mutate: (input) => {
      input.waivers.push({
        label: "lora:demo_lora",
        repo: "Org/demo-lora",
        revision: null,
        pattern: "demo.safetensors",
        story: "sc-1",
        reason: "already fixed upstream",
      });
    },
  },
  {
    name: "waiver-stale-unclaimed: a waiver no manifest entry declares fails",
    kind: "waiver-stale-unclaimed",
    mutate: (input) => {
      input.waivers.push({
        label: "lora:removed",
        repo: "Org/removed",
        revision: null,
        pattern: "removed.safetensors",
        story: "sc-2",
        reason: "entry was deleted from the catalog",
      });
    },
  },
  {
    name: "waiver-malformed: a waiver with no story reference fails",
    kind: "waiver-malformed",
    mutate: (input) => {
      input.claims[1].declared = ["absent.safetensors"];
      input.waivers.push({
        label: "lora:demo_lora",
        repo: "Org/demo-lora",
        revision: null,
        pattern: "absent.safetensors",
        story: "TODO",
        reason: "upstream has not published it",
      });
    },
  },
  {
    name: "waiver-malformed: a waiver with an empty reason fails",
    kind: "waiver-malformed",
    mutate: (input) => {
      input.claims[1].declared = ["absent.safetensors"];
      input.waivers.push({
        label: "lora:demo_lora",
        repo: "Org/demo-lora",
        revision: null,
        pattern: "absent.safetensors",
        story: "sc-3",
        reason: "   ",
      });
    },
  },
];

export function selfTestBaseline() {
  return {
    claims: clone(SELF_TEST_CLAIMS),
    evidence: clone(SELF_TEST_EVIDENCE),
    waivers: clone(SELF_TEST_WAIVERS),
  };
}

export function runSelfTest({ log = console.log } = {}) {
  const failures = [];

  const baseline = gradeRecordedEvidence(selfTestBaseline());
  if (baseline.problems.length !== 0) {
    failures.push(
      `baseline must pass unmutated, got: ${baseline.problems.map((p) => p.kind).join(", ")}`,
    );
  } else {
    log("  ok  baseline passes unmutated");
  }

  // A waived zero-match must be tolerated — otherwise the waiver mechanism is inert and the
  // "green means no NEW zero-match" claim is untested.
  const waivedInput = selfTestBaseline();
  waivedInput.claims[1].declared = ["absent.safetensors"];
  waivedInput.waivers.push({
    label: "lora:demo_lora",
    repo: "Org/demo-lora",
    revision: null,
    pattern: "absent.safetensors",
    story: "sc-4",
    reason: "tracked upstream gap",
  });
  const waivedResult = gradeRecordedEvidence(waivedInput);
  if (waivedResult.problems.length !== 0 || waivedResult.waived.length !== 1) {
    failures.push(
      `a well-formed waiver must absorb its zero-match, got problems=[${waivedResult.problems
        .map((p) => p.kind)
        .join(", ")}] waived=${waivedResult.waived.length}`,
    );
  } else {
    log("  ok  a well-formed waiver absorbs its zero-match");
  }

  for (const testCase of SELF_TEST_CASES) {
    const input = selfTestBaseline();
    testCase.mutate(input);
    const { problems } = gradeRecordedEvidence(input);
    const kinds = problems.map((problem) => problem.kind);
    if (!kinds.includes(testCase.kind)) {
      failures.push(`${testCase.name}: expected [${testCase.kind}], got [${kinds.join(", ")}]`);
    } else {
      log(`  ok  ${testCase.name}`);
    }
  }

  return failures;
}

async function main() {
  const argv = process.argv.slice(2);

  if (argv.includes("--self-test")) {
    console.log("check-download-patterns --self-test");
    const failures = runSelfTest();
    if (failures.length > 0) {
      console.error(`\nSELF-TEST FAILURES (${failures.length}):`);
      for (const failure of failures) console.error(`  ${failure}`);
      process.exitCode = 1;
      return;
    }
    console.log(`\n${SELF_TEST_CASES.length + 2} self-test case(s) passed.`);
    return;
  }

  if (argv.includes("--check")) {
    if (argv.includes("--model")) {
      console.error(
        "--check grades the whole catalog (coverage is a set property); --model applies to the live modes only.",
      );
      process.exitCode = 1;
      return;
    }
    await runCheck();
    return;
  }

  const only = argv.includes("--model") ? argv[argv.indexOf("--model") + 1] : null;
  await runLive({ only, write: argv.includes("--write") });
}

// Only run when invoked as a script — the test file imports the pure functions above.
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
