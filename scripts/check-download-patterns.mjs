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
//  1. **95 of 95 keys are pinned to immutable lowercase 40-hex revisions.** A git SHA's file
//     listing is timeless — re-reading it in a year returns the same bytes. There is nothing to
//     decay. sc-18924 closed the last 11-key moving-default-branch window, and the offline gate now
//     rejects every missing, branch, or tag revision before it can reintroduce that class.
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
//   node scripts/check-download-patterns.mjs --write --dry-run  # would the record change? (no write)
//   node scripts/check-download-patterns.mjs --check     # OFFLINE gate (this is what CI runs)
//   node scripts/check-download-patterns.mjs --self-test # mutation harness for the gate
//
// Exits non-zero if any declared pattern matches zero files. Public repos need no
// auth; a token is picked up from $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN if set, so a
// repo that is later gated still resolves.
//
// GATING IS DETECTABLE — CORRECTION (sc-18854 review)
// --------------------------------------------------
// This header, and sc-18917's description, previously asserted: "the metadata API answers 200 for
// a GATED repo, so this gate can never detect gating — only a real download can." The first half
// is true and the conclusion does not follow. **The 200 body carries a `gated` field**, and
// `repoFiles` was already parsing that body — it simply kept `sha` and `siblings` and threw the
// rest away. The impossibility was an artefact of what we chose to record, not of the API.
//
// The cost of that false claim was two live catalog entries grading green while being unfetchable:
// the Lightricks HDR and DubIt IC-LoRA repos answered `gated:"auto"` and returned 401 to an
// unauthenticated weight fetch. sc-18923 removed both instances by publishing the exact unmodified
// files, complete frozen license, attribution and provenance at the immutable public
// `SceneWorks/ltx-2.3-ic-loras` rehost, then repointing both manifest entries. The gate now proves
// those replacements are ungated instead of carrying either historical waiver.
//
// So `gated` is now RECORDED per key and `evidence-gated` is a HARD FAILURE by default, waivable
// through `KNOWN_REPO_CONDITIONS` on the same terms as a zero-match.
//
// The same body also carries `id` — the repo HF ACTUALLY served. `fetch` follows the 307 a renamed
// repo issues, so an entry naming a dead repo resolves silently; `evidence-repo-id-mismatch`
// catches that. ONE live instance remains, waived and tracked (sc-18926); the other — LipDub ->
// DubIt — was fixed at the source by sc-18917 rather than waived onward.
//
// What a green run still does NOT prove is that the bytes download. Public replacements such as
// sc-18923's rehost need a real ANONYMOUS fetch at the exact revision; a deliberately gated source
// would instead need an authenticated fetch by an account that accepted its terms. The ceiling is
// real — it is just one rung higher than this header used to claim.
//
// The LIVE modes deliberately ignore `KNOWN_ZERO_MATCHES` and report the raw truth, so `--write`
// exits 1 on any tracked zero-match gap. There is none today — sc-18917 fixed the last one — so
// `--write` currently exits 0, but do NOT read that as the modes agreeing: the remaining redirect
// condition still prints above and is waived only by the offline gate. Policy lives in the gate;
// the pre-flight is for a human who wants the unfiltered picture. Do not "fix" that exit code by
// teaching the recorder about waivers — a recorder that knows which failures are acceptable is one
// step from a recorder that records them as passing.

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
const CHECK_COMMAND = "npm run check:download-patterns:offline";

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
// EMPTY, and that is a verdict rather than an oversight: the catalog currently declares no
// pattern that matches zero files. The list's only entry was the LipDub/DubIt rename, deleted by
// sc-18917 when it repointed the manifest entry — the gate forced it, reporting
// `waiver-stale-unclaimed` the moment the waived tuple stopped being claimed. Keep the list (and
// this comment) rather than deleting the mechanism: the next real zero-match needs somewhere with
// an owner to go, and an empty list is the honest statement that today there is none.
export const KNOWN_ZERO_MATCHES = [];

// Known, TRACKED repo-level defects: the repo is access-gated, or HF served a DIFFERENT repo than
// the manifest names. Same discipline as `KNOWN_ZERO_MATCHES` — a reason, an `sc-NNNN` owner, and
// self-expiry in both directions — but keyed by `kind + repo@revision` rather than by pattern,
// because these are facts about the REPO and not about any one glob.
//
// WHY WAIVED RATHER THAN A HARD RED (the sc-18854 review asked for this decision explicitly)
// ------------------------------------------------------------------------------------------
// `evidence-gated` and `evidence-repo-id-mismatch` are HARD FAILURES BY DEFAULT. That is the whole
// point: a gated or redirected repo entering the catalog must red the PR that adds it, because
// that is the last moment the cost is cheap. What is waived is only the instances that ALREADY
// SHIP.
//
// The alternative — make known instances unconditionally fatal — was considered and rejected. It
// would red the required `parity` lane on every unrelated PR while a defect was being repaired,
// which pushes people toward deleting the guard rather than fixing its source.
//
// Waiving costs nothing in detection: the semantics land on "green means no NEW gated repo and no
// NEW redirect", which is exactly the guarantee zero-match already provides. And unlike a silent
// pass, each instance now has an owner, a reason, and a story that CANNOT be closed without
// deleting its waiver — the gate reds if the condition clears and reds if the entry moves.
//
// sc-18923 paid both gated-LoRA debts, so no `evidence-gated` waiver remains. New gated catalog
// sources hard-fail unless a new live owner and explicit rationale are added here.
export const KNOWN_REPO_CONDITIONS = [
  {
    kind: "evidence-repo-id-mismatch",
    repo: "apple/DFN5B-CLIP-ViT-H-14-384",
    revision: "01b771ed0d1395ca5ffdd279897d665ebe00dfd2",
    story: "sc-18926",
    reason:
      "mmaudio's `clip` co-requisite names apple/DFN5B-CLIP-ViT-H-14-384, which does not exist — " +
      "apple publishes only …-ViT-H-14 and …-ViT-H-14-378, and the declared name 307-redirects to " +
      "the 378 repo. Measured benign: the served repo reports image_size 378 in both " +
      "open_clip_config.json and preprocessor_config.json, which is the CLIP upstream MMAudio " +
      "conditions on, and the pinned sha is already that repo's. So the BYTES are right and only " +
      "the name is stale — but it depends on HF serving a rename redirect forever. sc-18926 " +
      "repoints both entries and deletes this waiver.",
  },
];

const REPO_CONDITION_KINDS = new Set(["evidence-gated", "evidence-repo-id-mismatch"]);

// Glob semantics must mirror the worker's `pattern_matches` (imports.rs), which uses the
// Rust `glob` crate with default MatchOptions — `*` and `?` DO cross `/` there
// (require_literal_separator is false), so a `q8/*` pattern legitimately matches
// `q8/vae/config.json`. Translating to a regex without that behavior would under-report
// matches and produce false failures here that the worker would never raise.
//
// The offline gate reuses this exact function rather than reimplementing the semantics —
// a second implementation would drift, and it would drift silently in the passing
// direction (a looser regex matches more, so a real zero-match reads as a hit).
// Globs this translator REFUSES to translate, because translating them would be wrong in the
// FALSE-GREEN direction (sc-18854 review).
//
// The worker does `glob::Pattern::new(&pattern).is_ok_and(|p| p.matches(&value))`
// (imports.rs:262-269). `Pattern::new` ERRORS on `**` that is not a whole path component — `***`,
// `a**b`, `**a` — and `is_ok_and` turns that error into `false`, so such a pattern matches
// NOTHING and the worker's per-pattern hard-fail fires. A regex translation of `**` matches
// EVERYTHING. Silently translating it therefore reads a guaranteed download failure as a hit,
// which is the one direction this gate must never be wrong in.
//
// We reject ALL `**`, not only the forms Rust rejects, because `**` is redundant here anyway: the
// worker runs with default MatchOptions (`require_literal_separator: false`), so a single `*`
// already crosses `/`. There is nothing an author can express with `**` that `*` does not cover,
// so refusing it costs no expressiveness and removes a whole class of divergence.
//
// An unterminated `[` is the same story: `Pattern::new` rejects it (matches nothing) while this
// translator used to fall back to a literal `\[` (matches a literal bracket).
//
// Unreachable on today's catalog — 0 of the declared patterns contain `**`, `[`, `?` or `\` — but
// the gate is load-bearing for CI now, so an unreachable divergence is still a latent false green.
export function patternTranslationError(pattern) {
  if (pattern.includes("**")) {
    return (
      "`**` is not translatable: Rust's glob::Pattern::new rejects it outside a whole path" +
      " component and the worker reads that as MATCHING NOTHING, while a regex translation" +
      " matches everything. `*` already crosses `/` here — use a single `*`."
    );
  }
  for (let index = pattern.indexOf("["); index !== -1; ) {
    const close = pattern.indexOf("]", index + 1);
    if (close === -1) {
      return (
        "unterminated `[` character class: Rust's glob::Pattern::new rejects it (matching" +
        " nothing), while this translator would fall back to a literal `[`."
      );
    }
    index = pattern.indexOf("[", close + 1);
  }
  return null;
}

export function patternToRegExp(pattern) {
  // Throw rather than translate. Callers are expected to check `patternTranslationError` first and
  // report `pattern-untranslatable`; this is the backstop that stops a future second caller from
  // reintroducing the false green by forgetting to.
  const untranslatable = patternTranslationError(pattern);
  if (untranslatable) {
    throw new Error(`untranslatable glob ${JSON.stringify(pattern)}: ${untranslatable}`);
  }
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

// The unit of evidence. `@main` remains a useful diagnostic key for a malformed claim, but
// sc-18924 makes it a hard authoring failure: every production claim must carry a lowercase
// 40-hex commit SHA.
export function claimKey(repo, revision) {
  return `${repo}@${revision ?? "main"}`;
}

const IMMUTABLE_REVISION_PATTERN = /^[0-9a-f]{40}$/u;

function waiverKey({ label, repo, revision, pattern }) {
  // Joined on a NUL so the key is unambiguous: repos, labels and globs all legitimately
  // contain "/", "@", ":" and spaces, so any printable separator could be forged by a crafted
  // pattern into colliding with a different tuple. Shares TUPLE_SEPARATOR with repoConditionKey
  // so the two key builders cannot drift apart, and never appears as a literal NUL byte in this
  // source -- a literal one makes grep classify the file as binary and silently report "no
  // match" for every later search of it.
  return [label, claimKey(repo, revision), pattern].join(TUPLE_SEPARATOR);
}

// The same NUL separator `waiverKey` uses, as a value rather than a typed escape so both key
// builders provably share one definition and cannot drift apart. `String.fromCharCode(0)` keeps
// the source itself free of a literal NUL byte, which is what would make grep treat this file as
// binary and silently report "no match" for every later search of it.
const TUPLE_SEPARATOR = String.fromCharCode(0);

// Repo-level waivers key on the CONDITION plus the repo@revision, not on a pattern — these are
// facts about the REPO, true or false regardless of which glob is being checked.
function repoConditionKey({ kind, repo, revision }) {
  return [kind, claimKey(repo, revision)].join(TUPLE_SEPARATOR);
}

// Shared shape validation + indexing for BOTH waiver lists.
//
// The duplicate check is the sc-18854 review's finding: this used to be a bare `index.set(...)`, so
// a second waiver on the same tuple SILENTLY REPLACED the first. Two waivers naming different
// stories would leave one unreported and un-expiring — and the "every waiver is live" invariant the
// test suite asserts would still read as satisfied while one owner had quietly vanished. A
// duplicate is always an authoring mistake, so it fails rather than resolving to either candidate.
function indexWaivers({ waivers, keyOf, describe, malformedKind, duplicateKind, validate, fail }) {
  const index = new Map();
  for (const waiver of waivers ?? []) {
    const where = describe(waiver);
    if (typeof waiver?.reason !== "string" || waiver.reason.trim() === "") {
      fail(malformedKind, `${where}  waiver has no reason`);
      continue;
    }
    if (!STORY_REF_PATTERN.test(waiver?.story ?? "")) {
      fail(
        malformedKind,
        `${where}  waiver story must look like sc-NNNN, got ${JSON.stringify(waiver?.story ?? null)}`,
      );
      continue;
    }
    const invalid = validate?.(waiver);
    if (invalid) {
      fail(malformedKind, `${where}  ${invalid}`);
      continue;
    }
    const key = keyOf(waiver);
    const existing = index.get(key);
    if (existing) {
      fail(
        duplicateKind,
        `${where}  duplicate waiver for this exact tuple — ${existing.story} and ${waiver.story}` +
          ` both claim it, so one would be silently dropped. Keep one.`,
      );
      continue;
    }
    index.set(key, waiver);
  }
  return index;
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
export function gradeRecordedEvidence({ claims, evidence, waivers, repoConditions }) {
  const problems = [];
  const waived = [];
  const fail = (kind, message) => problems.push({ kind, message });

  // Waiver shape first: an allowlist without a reason is a catch-all wearing a decision's
  // clothes, and one without a story reference has no owner.
  const waiverIndex = indexWaivers({
    waivers,
    keyOf: waiverKey,
    describe: (waiver) => `${waiver?.label ?? "<unlabelled>"}  ${waiver?.pattern ?? "<no pattern>"}`,
    malformedKind: "waiver-malformed",
    duplicateKind: "waiver-duplicate",
    fail,
  });
  const repoWaiverIndex = indexWaivers({
    waivers: repoConditions,
    keyOf: repoConditionKey,
    describe: (waiver) =>
      `${waiver?.kind ?? "<no kind>"}  ${claimKey(waiver?.repo ?? "<no repo>", waiver?.revision)}`,
    malformedKind: "repo-waiver-malformed",
    duplicateKind: "repo-waiver-duplicate",
    // A repo waiver that names a condition the gate does not raise would sit inert forever, so it
    // is malformed rather than harmless.
    validate: (waiver) =>
      REPO_CONDITION_KINDS.has(waiver?.kind)
        ? null
        : `waiver kind must be one of ${[...REPO_CONDITION_KINDS].join(", ")}, got ${JSON.stringify(waiver?.kind ?? null)}`,
    fail,
  });

  const recorded = new Map((evidence?.repos ?? []).map((entry) => [entry.key, entry]));
  const claimedKeys = new Set();
  const claimedWaiverKeys = new Set();
  // Keys whose recorded entry was actually readable — a repo-condition waiver on an unreachable
  // key must not be graded stale, because the condition could not be evaluated either way.
  const evaluatedKeys = new Set();
  // Repo conditions that genuinely HOLD right now, so an obsolete waiver can be told apart from a
  // live one. Also the dedupe set: several claims share one repo@revision and the condition is a
  // property of the key, so it is reported once rather than once per claim.
  const activeRepoConditions = new Set();

  // A repo-level defect: recorded once per key, waivable, and NOT a reason to skip pattern grading
  // — gating and a rename redirect are orthogonal to whether the globs match, and suppressing the
  // pattern verdicts would just mean a second round of failures once the waiver is lifted.
  const noteRepoCondition = ({ kind, claim, key, detail }) => {
    const conditionKey = repoConditionKey({ kind, repo: claim.repo, revision: claim.revision });
    if (activeRepoConditions.has(conditionKey)) return;
    activeRepoConditions.add(conditionKey);
    const waiver = repoWaiverIndex.get(conditionKey);
    if (waiver) {
      waived.push(`[${kind}] ${claim.label}  ${key}  ${detail}  (${waiver.story})`);
      return;
    }
    fail(kind, `${claim.label}  ${key}  ${detail}`);
  };

  for (const claim of claims) {
    const key = claimKey(claim.repo, claim.revision);
    claimedKeys.add(key);
    const at = claim.revision ? `@${claim.revision.slice(0, 12)}` : "";
    const entry = recorded.get(key);
    const immutableRevision =
      typeof claim.revision === "string" && IMMUTABLE_REVISION_PATTERN.test(claim.revision);

    // sc-18924: an evidence fixture is timeless only when its manifest claim is immutable. A
    // recorded `resolvedSha` on an unrevisioned key merely says which moving default branch the
    // recorder happened to observe; nothing forces another read after upstream moves it. Reject
    // missing revisions, branches and tags at authoring time so that stale-green class cannot
    // regrow. Keep grading the entry after reporting this problem so independent pattern, gating,
    // redirect and waiver defects are not hidden behind it.
    if (!immutableRevision) {
      fail(
        "manifest-revision-required",
        `${claim.label}  ${key}  manifest Hugging Face downloads must pin ` +
          `\`revision\` to an immutable lowercase 40-hex commit SHA (sc-18924)`,
      );
    }

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
    // The recorder must have read the snapshot it claims to have read. NOTE what this does and
    // does not cover, because its old comment over-claimed (sc-18854 review): an HF rename redirect
    // preserves the sha, so a redirected pinned entry passes it. It catches a recorder bug or a
    // moved immutable pin — not a substituted repo. `evidence-repo-id-mismatch` below covers
    // substitution. A branch or tag normally resolves to a different 40-hex SHA; its authoring
    // failure was already reported above, so do not mislabel that expected difference or stop
    // grading independent redirect, gating and pattern defects behind it.
    if (immutableRevision && entry.resolvedSha && entry.resolvedSha !== claim.revision) {
      fail(
        "evidence-revision-mismatch",
        `${claim.label}  ${key}  recorded listing resolved to ${entry.resolvedSha.slice(0, 12)} — re-record with \`${RECORD_COMMAND}\``,
      );
      continue;
    }
    // Past every reason the entry could not be graded, so the repo conditions below are genuinely
    // evaluated for this key and a waiver on it can be judged live or stale.
    evaluatedKeys.add(key);
    // The repo HF ACTUALLY served. `fetch` follows redirects, so a manifest entry naming a repo
    // that upstream renamed away resolves silently and every downstream check passes against a
    // tree nobody declared. Unlike the sha check above this also remains meaningful for a malformed
    // unrevisioned input, even though sc-18924 now rejects that input independently.
    if (entry.servedRepo && entry.servedRepo !== claim.repo) {
      noteRepoCondition({
        kind: "evidence-repo-id-mismatch",
        claim,
        key,
        detail: `listing was served by ${entry.servedRepo} — the declared repo no longer exists except as a redirect`,
      });
    }
    // Gating. The metadata API answers 200 for a gated repo AND says so in `gated`; recording that
    // field is the whole correction described in this file's header. `gated:"auto"` is not open
    // access — terms are auto-approved ON ACCEPTANCE, so it still needs a token whose account has
    // accepted the licence, and the worker's token is optional.
    if (entry.gated) {
      noteRepoCondition({
        kind: "evidence-gated",
        claim,
        key,
        detail: `repo is access-gated (gated: ${JSON.stringify(entry.gated)}) — an unauthenticated download returns 401`,
      });
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
      // Refuse untranslatable globs instead of mistranslating them — see
      // `patternTranslationError`. Registered above first so a waiver on this tuple is not ALSO
      // reported as unclaimed; one defect, one message.
      const untranslatable = patternTranslationError(pattern);
      if (untranslatable) {
        fail(
          "pattern-untranslatable",
          `${claim.label}  ${claim.repo}${at}  ${pattern}  ${untranslatable}`,
        );
        continue;
      }
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

  // Repo-condition waivers self-expire in both directions, exactly like the pattern ones: the
  // condition clearing means the debt was paid, and the key going unclaimed means the entry moved.
  for (const [key, waiver] of repoWaiverIndex) {
    if (activeRepoConditions.has(key)) continue;
    const target = claimKey(waiver.repo, waiver.revision);
    if (!claimedKeys.has(target)) {
      fail(
        "repo-waiver-stale-unclaimed",
        `${waiver.kind}  ${target}  no manifest entry claims this key any more — re-key or delete the ${waiver.story} waiver`,
      );
      continue;
    }
    // Claimed but never graded: an `evidence-missing-key` / `evidence-unreachable` /
    // `evidence-revision-mismatch` problem already fired for it, and the condition could not be
    // evaluated. Reporting the waiver as stale on top of that would be a second message for one
    // defect, and a misleading one — nothing here says the condition cleared.
    if (!evaluatedKeys.has(target)) continue;
    fail(
      "repo-waiver-stale-cleared",
      `${waiver.kind}  ${target}  no longer holds — delete the ${waiver.story} waiver`,
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
    // `id` is the repo HF ACTUALLY served. `fetch` follows the 307 a renamed repo issues, so
    // without this the substitution is invisible — the listing simply looks healthy.
    servedRepo: body.id ?? null,
    // The field this script's header used to claim did not exist. It does, on the same 200 body
    // that was already being parsed for `sha` and `siblings`. `false` when open; a string
    // ("auto" | "manual") when gated. Discarding it is what let a 401-on-download entry grade
    // green — see the GATING IS DETECTABLE block at the top of this file.
    gated: body.gated ?? false,
    files: (body.siblings ?? []).map((sibling) => sibling.rfilename).sort(),
  };
  fileCache.set(key, result);
  return result;
}

// LIVE mode: resolve every claim against the HF API. `--write` additionally transcribes the
// listings to the evidence file. Note the recorder writes what it read even when a pattern
// matches nothing — the verdict belongs to `--check`, not to the transcription step, so a
// failing catalog still produces an accurate (and reviewable) snapshot.
async function runLive({ only, write, dryRun }) {
  const { models, loras } = await readManifests();
  const claims = collectClaims({ models, loras }).filter(
    (claim) => !only || claim.id === only,
  );

  const failures = [];
  const unreachable = [];
  const untranslatable = [];
  const gatedRepos = [];
  const redirected = [];
  const recorded = new Map();
  let entries = 0;
  let patterns = 0;

  for (const claim of claims) {
    const key = claimKey(claim.repo, claim.revision);
    const at = claim.revision ? `@${claim.revision.slice(0, 12)}` : "";
    const { files, error, resolvedSha, servedRepo, gated } = await repoFiles(
      claim.repo,
      claim.revision,
    );
    if (!recorded.has(key)) {
      recorded.set(
        key,
        error
          ? { key, repo: claim.repo, revision: claim.revision, error }
          : { key, repo: claim.repo, revision: claim.revision, resolvedSha, servedRepo, gated, files },
      );
      // Report repo-level facts once per key rather than once per claim. The LIVE modes show the
      // unfiltered picture — no waivers applied — for the same reason they ignore
      // KNOWN_ZERO_MATCHES: policy lives in the gate.
      if (!error && servedRepo && servedRepo !== claim.repo) {
        redirected.push(`${claim.label}  ${claim.repo}${at}  served by ${servedRepo}`);
      }
      if (!error && gated) {
        gatedRepos.push(`${claim.label}  ${claim.repo}${at}  gated: ${JSON.stringify(gated)}`);
      }
    }
    if (error) {
      unreachable.push(`${claim.label}  ${claim.repo}${at}  (${error})`);
      continue;
    }
    entries += 1;
    for (const pattern of claim.declared) {
      patterns += 1;
      const translationError = patternTranslationError(pattern);
      if (translationError) {
        untranslatable.push(`${claim.label}  ${claim.repo}${at}  ${pattern}  ${translationError}`);
        continue;
      }
      const regexp = patternToRegExp(pattern);
      if (!files.some((file) => regexp.test(file))) {
        failures.push(`${claim.label}  ${claim.repo}${at}  ${pattern}`);
      }
    }
  }

  console.log(`checked ${patterns} pattern(s) across ${entries} download entr(ies)`);

  // The live verdicts below accumulate into a flag instead of writing straight to
  // `process.exitCode`, because `--write --dry-run` publishes an ANTI-TAMPER verdict and has to
  // own its exit code outright (sc-18854 review). Folding the live verdicts in historically made
  // the mode exit 1 whenever a tracked live defect was present even if the re-record was exact, so a
  // reviewer would see a non-zero exit, find no artifact diff to explain it, and conclude the
  // artifact had been hand-edited. A tamper detector whose result is masked by a different verdict
  // detects nothing.
  let liveFailure = false;
  let dryRunVerdict = null;

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
    // and `resolvedSha` proves which immutable tree the API returned for each pin.
    const next = `${JSON.stringify({ repos }, null, 2)}\n`;
    const target = path.join(root, EVIDENCE_FILE);
    if (dryRun) {
      // REVIEWER TOOL (sc-18854 review). The gate's whole trust chain terminates in a generated
      // file that a reviewer cannot practically diff by eye, so this answers "is the committed
      // artifact what a re-record would produce?" without touching the working tree. Deliberately
      // wired into NOTHING automatic: it needs the network, and the point of the split is that no
      // required lane depends on huggingface.co.
      //
      // The verdict is computed here but REPORTED LAST, so the line a reviewer is looking for is
      // not buried above several screens of live diagnostics.
      let current = null;
      try {
        current = await readFile(target, "utf8");
      } catch {
        current = null;
      }
      dryRunVerdict = { upToDate: current === next, missing: current === null, keys: repos.length };
    } else {
      await writeFile(target, next, "utf8");
      console.log(`wrote ${EVIDENCE_FILE} (${repos.length} repo@revision key(s))`);
    }
  }

  if (redirected.length > 0) {
    console.log(`\nSERVED BY A DIFFERENT REPO (${redirected.length}) — declared name is a redirect:`);
    for (const line of redirected) console.log(`  ${line}`);
  }
  if (gatedRepos.length > 0) {
    console.log(`\nACCESS-GATED (${gatedRepos.length}) — an unauthenticated download returns 401:`);
    for (const line of gatedRepos) console.log(`  ${line}`);
  }
  if (untranslatable.length > 0) {
    console.error(`\nUNTRANSLATABLE PATTERNS (${untranslatable.length}):`);
    for (const line of untranslatable) console.error(`  ${line}`);
    liveFailure = true;
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
    liveFailure = true;
  } else if (unreachable.length > 0) {
    liveFailure = true;
  } else {
    console.log("\nEvery declared download pattern matches at least one file.");
  }

  if (dryRunVerdict) {
    // The ONLY thing that sets the exit code in this mode. The runbook documents it as answering
    // exactly one question, so it must answer exactly that one.
    if (dryRunVerdict.upToDate) {
      console.log(
        `\n=== ARTIFACT VERDICT: UP TO DATE (${dryRunVerdict.keys} key(s)) — exit 0 ===\n` +
          `${EVIDENCE_FILE} is byte-identical to what a re-record produces: an honest re-record,` +
          ` not a hand edit. Nothing was written.`,
      );
    } else {
      console.error(
        `\n=== ARTIFACT VERDICT: WOULD CHANGE — exit 1 ===\n${EVIDENCE_FILE} is not what a` +
          ` re-record produces. Re-record with \`${RECORD_COMMAND}\` and commit the result.` +
          (dryRunVerdict.missing ? " (no committed artifact found)" : ""),
      );
      process.exitCode = 1;
    }
    console.log(
      `This mode's exit code carries the artifact verdict above and NOTHING else — the live` +
        ` verdicts printed before it (zero-match, gated, redirect, untranslatable, unreachable)` +
        ` do not change it. Grade those with \`${CHECK_COMMAND}\`.`,
    );
    return;
  }

  if (liveFailure) process.exitCode = 1;
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
    repoConditions: KNOWN_REPO_CONDITIONS,
  });
  const patterns = claims.reduce((total, claim) => total + claim.declared.length, 0);
  console.log(
    `graded ${patterns} pattern(s) across ${claims.length} download entr(ies)` +
      ` against ${(evidence.repos ?? []).length} recorded repo@revision key(s) — no network`,
  );
  if (waived.length > 0) {
    console.log(`\nWAIVED (${waived.length}) — tracked, not fixed:`);
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

const SELF_TEST_LORA_REVISION = "b".repeat(40);

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
    revision: SELF_TEST_LORA_REVISION,
    declared: ["demo.safetensors"],
  },
];

// `servedRepo` equals `repo` and `gated` is false throughout, so the baseline exercises the
// repo-condition guards in their PASSING direction too — a guard that fired unconditionally would
// show up as a baseline failure rather than hiding behind the mutation cases.
const SELF_TEST_EVIDENCE = {
  repos: [
    {
      key: `Org/demo@${"a".repeat(40)}`,
      repo: "Org/demo",
      revision: "a".repeat(40),
      resolvedSha: "a".repeat(40),
      servedRepo: "Org/demo",
      gated: false,
      files: ["q8/transformer/model.safetensors", "q8/vae/config.json"],
    },
    {
      key: `Org/demo-lora@${SELF_TEST_LORA_REVISION}`,
      repo: "Org/demo-lora",
      revision: SELF_TEST_LORA_REVISION,
      resolvedSha: SELF_TEST_LORA_REVISION,
      servedRepo: "Org/demo-lora",
      gated: false,
      files: ["demo.safetensors"],
    },
  ],
};

const SELF_TEST_WAIVERS = [];
const SELF_TEST_REPO_CONDITIONS = [];

const clone = (value) => JSON.parse(JSON.stringify(value));

// Each case: mutate a deep copy of the baseline and expect `kind` plus any `alsoKinds` to appear.
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
    name: "manifest-revision-required: a missing revision fails even when matching @main evidence exists",
    kind: "manifest-revision-required",
    mutate: (input) => {
      input.claims[1].revision = null;
      input.evidence.repos[1].key = "Org/demo-lora@main";
      input.evidence.repos[1].revision = null;
    },
  },
  {
    name: "manifest-revision-required: a mutable tag fails without hiding independent defects",
    kind: "manifest-revision-required",
    alsoKinds: ["evidence-repo-id-mismatch", "evidence-gated", "zero-match"],
    mutate: (input) => {
      input.claims[1].revision = "latest";
      input.evidence.repos[1].key = "Org/demo-lora@latest";
      input.evidence.repos[1].revision = "latest";
      // Recorder-realistic tag evidence: HF reports the mutable ref in `revision` and the
      // immutable tree it currently resolves to in `resolvedSha`.
      input.evidence.repos[1].resolvedSha = "c".repeat(40);
      input.evidence.repos[1].servedRepo = "Org/demo-lora-renamed";
      input.evidence.repos[1].gated = "auto";
      input.evidence.repos[1].files = [];
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
        revision: SELF_TEST_LORA_REVISION,
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
        revision: SELF_TEST_LORA_REVISION,
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
        revision: SELF_TEST_LORA_REVISION,
        pattern: "absent.safetensors",
        story: "sc-3",
        reason: "   ",
      });
    },
  },
  {
    name: "waiver-duplicate: two pattern waivers on one tuple fail instead of one silently winning",
    kind: "waiver-duplicate",
    mutate: (input) => {
      input.claims[1].declared = ["absent.safetensors"];
      const base = {
        label: "lora:demo_lora",
        repo: "Org/demo-lora",
        revision: SELF_TEST_LORA_REVISION,
        pattern: "absent.safetensors",
        reason: "tracked upstream gap",
      };
      input.waivers.push({ ...base, story: "sc-5" }, { ...base, story: "sc-6" });
    },
  },
  {
    name: "pattern-untranslatable: a `**` glob is refused, not translated (Rust matches nothing, a regex matches everything)",
    kind: "pattern-untranslatable",
    mutate: (input) => {
      input.claims[0].declared.push("q8/**/model.safetensors");
    },
  },
  {
    name: "pattern-untranslatable: an unterminated `[` class is refused",
    kind: "pattern-untranslatable",
    mutate: (input) => {
      input.claims[0].declared.push("q8/[abc.safetensors");
    },
  },
  {
    name: "evidence-gated: a recorded gated repo fails (the sc-18923 guard)",
    kind: "evidence-gated",
    mutate: (input) => {
      // Exactly the HDR shape: listing is healthy and the declared file is present, so every
      // pattern matches — and the download still 401s.
      input.evidence.repos[1].gated = "auto";
    },
  },
  {
    name: "evidence-repo-id-mismatch: a pinned listing served by a different repo fails",
    kind: "evidence-repo-id-mismatch",
    mutate: (input) => {
      input.evidence.repos[1].servedRepo = "Org/demo-lora-renamed";
    },
  },
  {
    name: "repo-waiver-stale-cleared: a repo waiver whose condition no longer holds fails",
    kind: "repo-waiver-stale-cleared",
    mutate: (input) => {
      input.repoConditions.push({
        kind: "evidence-gated",
        repo: "Org/demo-lora",
        revision: SELF_TEST_LORA_REVISION,
        story: "sc-7",
        reason: "upstream un-gated it",
      });
    },
  },
  {
    name: "repo-waiver-stale-unclaimed: a repo waiver no manifest entry claims fails",
    kind: "repo-waiver-stale-unclaimed",
    mutate: (input) => {
      input.repoConditions.push({
        kind: "evidence-gated",
        repo: "Org/removed",
        revision: null,
        story: "sc-8",
        reason: "entry was deleted from the catalog",
      });
    },
  },
  {
    name: "repo-waiver-malformed: a repo waiver naming a condition the gate never raises fails",
    kind: "repo-waiver-malformed",
    mutate: (input) => {
      input.evidence.repos[1].gated = "auto";
      input.repoConditions.push({
        kind: "evidence-probably-fine",
        repo: "Org/demo-lora",
        revision: SELF_TEST_LORA_REVISION,
        story: "sc-9",
        reason: "a kind nothing emits would sit inert forever",
      });
    },
  },
  {
    name: "repo-waiver-malformed: a repo waiver with no story reference fails",
    kind: "repo-waiver-malformed",
    mutate: (input) => {
      input.evidence.repos[1].gated = "auto";
      input.repoConditions.push({
        kind: "evidence-gated",
        repo: "Org/demo-lora",
        revision: SELF_TEST_LORA_REVISION,
        story: "TODO",
        reason: "upstream gated it",
      });
    },
  },
  {
    name: "repo-waiver-duplicate: two repo waivers on one tuple fail instead of one silently winning",
    kind: "repo-waiver-duplicate",
    mutate: (input) => {
      input.evidence.repos[1].gated = "auto";
      const base = {
        kind: "evidence-gated",
        repo: "Org/demo-lora",
        revision: SELF_TEST_LORA_REVISION,
        reason: "upstream gated it",
      };
      input.repoConditions.push({ ...base, story: "sc-10" }, { ...base, story: "sc-11" });
    },
  },
];

export function selfTestBaseline() {
  return {
    claims: clone(SELF_TEST_CLAIMS),
    evidence: clone(SELF_TEST_EVIDENCE),
    waivers: clone(SELF_TEST_WAIVERS),
    repoConditions: clone(SELF_TEST_REPO_CONDITIONS),
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
    revision: SELF_TEST_LORA_REVISION,
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

  // Same proof for the repo-condition list: without this, every `evidence-gated` case below could
  // pass while the waiver mechanism for it was inert, and the "tracked, not silent" claim would be
  // untested. Both directions of the same guard, one mutation apart.
  const gatedInput = selfTestBaseline();
  gatedInput.evidence.repos[1].gated = "auto";
  gatedInput.repoConditions.push({
    kind: "evidence-gated",
    repo: "Org/demo-lora",
    revision: SELF_TEST_LORA_REVISION,
    story: "sc-12",
    reason: "tracked upstream gating",
  });
  const gatedResult = gradeRecordedEvidence(gatedInput);
  if (gatedResult.problems.length !== 0 || gatedResult.waived.length !== 1) {
    failures.push(
      `a well-formed repo waiver must absorb its condition, got problems=[${gatedResult.problems
        .map((p) => p.kind)
        .join(", ")}] waived=${gatedResult.waived.length}`,
    );
  } else {
    log("  ok  a well-formed repo waiver absorbs its gated condition");
  }

  for (const testCase of SELF_TEST_CASES) {
    const input = selfTestBaseline();
    testCase.mutate(input);
    const { problems } = gradeRecordedEvidence(input);
    const kinds = problems.map((problem) => problem.kind);
    const expectedKinds = [testCase.kind, ...(testCase.alsoKinds ?? [])];
    const missingKinds = expectedKinds.filter((kind) => !kinds.includes(kind));
    if (missingKinds.length > 0) {
      failures.push(
        `${testCase.name}: expected [${expectedKinds.join(", ")}], got [${kinds.join(", ")}]`,
      );
    } else {
      log(`  ok  ${testCase.name}`);
    }
  }

  return failures;
}

// Argument validation, split out so it runs BEFORE the mode dispatch and can be unit-tested.
// Returns an error string, or null if the combination is legal.
//
// Placement is the whole point (sc-18854 review): `--self-test` and `--check` return early, so a
// rejection written after them silently ACCEPTS the flag it exists to reject. `--check --dry-run`
// exited 0 having graded offline — no network, no artifact comparison — which is precisely the
// "reviewer believes they verified the artifact when they did not" failure the bare `--dry-run`
// rejection was written to prevent, and `--check` is the documented CI command so it is the
// likelier typo of the two.
export function argumentError(argv) {
  const selfTest = argv.includes("--self-test");
  const check = argv.includes("--check");
  const write = argv.includes("--write");
  const dryRun = argv.includes("--dry-run");

  if ((selfTest || check) && (write || dryRun)) {
    const mode = selfTest ? "--self-test" : "--check";
    const offending = [write ? "--write" : null, dryRun ? "--dry-run" : null].filter(Boolean);
    return (
      `${mode} ignores ${offending.join(" and ")}: it never reads the network and never touches` +
      ` ${EVIDENCE_FILE}. To verify the committed artifact against a live re-record, run` +
      ` \`${RECORD_COMMAND} --dry-run\`.`
    );
  }
  if (dryRun && !write) {
    // Fail rather than ignore it. `--dry-run` alone reads as "do the safe thing", and silently
    // treating it as a plain live run would let a reviewer believe they had verified the artifact.
    return (
      "--dry-run only applies to the recorder; use `--write --dry-run` to check whether the" +
      " committed evidence file is what a re-record would produce."
    );
  }
  if (check && argv.includes("--model")) {
    return "--check grades the whole catalog (coverage is a set property); --model applies to the live modes only.";
  }
  return null;
}

async function main() {
  const argv = process.argv.slice(2);

  const error = argumentError(argv);
  if (error) {
    console.error(error);
    process.exitCode = 1;
    return;
  }

  if (argv.includes("--self-test")) {
    console.log("check-download-patterns --self-test");
    const failures = runSelfTest();
    if (failures.length > 0) {
      console.error(`\nSELF-TEST FAILURES (${failures.length}):`);
      for (const failure of failures) console.error(`  ${failure}`);
      process.exitCode = 1;
      return;
    }
    console.log(`\n${SELF_TEST_CASES.length + 3} self-test case(s) passed.`);
    return;
  }

  if (argv.includes("--check")) {
    await runCheck();
    return;
  }

  const only = argv.includes("--model") ? argv[argv.indexOf("--model") + 1] : null;
  await runLive({ only, write: argv.includes("--write"), dryRun: argv.includes("--dry-run") });
}

// Only run when invoked as a script — the test file imports the pure functions above.
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
