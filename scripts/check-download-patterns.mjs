// SceneWorks catalog guard: every declared download pattern must match a real file
// (sc-12283, epic 8506 "Catalog-wide quant matrix").
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
// download. This script moves it back to authoring time: run it after editing a
// `downloads[]` entry, or after publishing/re-hosting a tier, and a typo'd glob or an
// unpublished tier surfaces here instead of in someone's download queue.
//
// WHY IT IS NOT IN CI
// -------------------
// It talks to the Hugging Face API. As of sc-18809 the unit of work is one request per
// `repo@revision` rather than per repo, because each entry is now resolved at its OWN
// pin (see `repoFiles`). Measured against the current catalog that is **96 requests**
// covering 392 patterns across 282 download entries — 83 revision-qualified keys plus
// 13 unrevisioned repos read at their default branch.
//
// Note the request count did NOT go up when the revision qualification landed: no repo
// in the catalog is currently declared at two different revisions, so `repo@revision`
// is still injective over `repo` (96 → 96). Re-measure rather than assume if that ever
// changes — two tiers of one repo pinned apart would add exactly one request each.
// (The stale "~53 repos / 217 patterns" this comment used to quote was the sc-12283
// sweep's snapshot, and the catalog has since roughly doubled on both axes.)
//
// It is still too slow and too flaky a dependency for the parity lane, which must stay
// hermetic and fast. It is a deliberate manual pre-flight, in the spirit of that
// sc-12283 sweep: at the time it shipped, every declared pattern matched, which is what
// made hard-failing safe.
//
// Covers builtin.models.jsonc `downloads[].files` AND builtin.loras.jsonc
// `source.file` / `source.files` — the LoRA download path gained the same hard-fail in
// sc-12288, so it needs the same pre-flight.
//
// USAGE
//   node scripts/check-download-patterns.mjs            # all HF model + LoRA entries
//   node scripts/check-download-patterns.mjs --model krea_2_raw
//
// Exits non-zero if any declared pattern matches zero files. Public repos need no
// auth; a token is picked up from $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN if set, so a
// repo that is later gated still resolves.

import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { stripJsoncComments } from "./lib/jsonc.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_MANIFEST = "config/manifests/builtin.models.jsonc";
const LORA_MANIFEST = "config/manifests/builtin.loras.jsonc";

// Glob semantics must mirror the worker's `pattern_matches` (imports.rs), which uses the
// Rust `glob` crate with default MatchOptions — `*` and `?` DO cross `/` there
// (require_literal_separator is false), so a `q8/*` pattern legitimately matches
// `q8/vae/config.json`. Translating to a regex without that behavior would under-report
// matches and produce false failures here that the worker would never raise.
function patternToRegExp(pattern) {
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
  const key = `${repo}@${revision ?? "main"}`;
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
  const result = { files: (body.siblings ?? []).map((sibling) => sibling.rfilename) };
  fileCache.set(key, result);
  return result;
}

async function main() {
  const only = process.argv.includes("--model")
    ? process.argv[process.argv.indexOf("--model") + 1]
    : null;

  const failures = [];
  const unreachable = [];
  let repos = 0;
  let patterns = 0;

  // Flatten both manifests to a common shape: { label, repo, declared[] }. A model declares
  // `downloads[].files`; a LoRA declares `source.file` (one) or `source.files` (a list).
  const claims = [];
  const models = JSON.parse(
    stripJsoncComments(await readFile(path.join(root, MODEL_MANIFEST), "utf8")),
  );
  for (const model of models.models ?? []) {
    for (const download of model.downloads ?? []) {
      if (download.provider !== "huggingface") continue;
      claims.push({
        id: model.id,
        label: `${model.id}/${download.variant ?? "-"}`,
        repo: download.repo,
        revision: download.revision ?? null,
        declared: download.files ?? [],
      });
    }
  }
  const loras = JSON.parse(
    stripJsoncComments(await readFile(path.join(root, LORA_MANIFEST), "utf8")),
  );
  for (const lora of loras.loras ?? []) {
    const source = lora.source ?? {};
    const provider = source.provider ?? lora.provider;
    const repo = source.repo ?? lora.repo;
    if (provider !== "huggingface" || !repo) continue;
    const single = source.file ?? lora.file;
    claims.push({
      id: lora.id,
      label: `lora:${lora.id}`,
      repo,
      revision: source.revision ?? lora.revision ?? null,
      declared: single ? [single] : (source.files ?? lora.files ?? []),
    });
  }

  for (const claim of claims) {
    if (only && claim.id !== only) continue;
    // An empty declaration is a deliberate whole-repo fetch, not an omission — there is no
    // per-pattern claim to verify. (The worker's aggregate zero-file check still covers an
    // empty repo at download time.)
    if (claim.declared.length === 0) continue;
    const at = claim.revision ? `@${claim.revision.slice(0, 12)}` : "";
    const { files, error } = await repoFiles(claim.repo, claim.revision);
    if (error) {
      unreachable.push(`${claim.label}  ${claim.repo}${at}  (${error})`);
      continue;
    }
    repos += 1;
    for (const pattern of claim.declared) {
      patterns += 1;
      const regexp = patternToRegExp(pattern);
      if (!files.some((file) => regexp.test(file))) {
        failures.push(`${claim.label}  ${claim.repo}${at}  ${pattern}`);
      }
    }
  }

  console.log(`checked ${patterns} pattern(s) across ${repos} download entr(ies)`);
  if (unreachable.length > 0) {
    console.log(`\nUNREACHABLE (could not verify — set $HF_TOKEN if these are gated):`);
    for (const line of unreachable) console.log(`  ${line}`);
  }
  if (failures.length > 0) {
    console.error(`\nZERO-MATCH PATTERNS (${failures.length}) — the worker will hard-fail these downloads:`);
    for (const line of failures) console.error(`  ${line}`);
    console.error(
      `\nEither the glob is wrong, the tier is not published yet, or the entry's pinned revision` +
        ` predates the files it declares (sc-18809 — the trailing @<rev> above is what was checked).`,
    );
    process.exitCode = 1;
    return;
  }
  // An unreachable repo is not a pass: we made no claim about it either way.
  if (unreachable.length > 0) {
    process.exitCode = 1;
    return;
  }
  console.log("\nEvery declared download pattern matches at least one file.");
}

await main();
