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
// It talks to the Hugging Face API — one request per repo (~53 today). That is too
// slow and too flaky a dependency for the parity lane, which must stay hermetic and
// fast. It is a deliberate manual pre-flight, in the spirit of the sc-12283 sweep that
// gated the worker change: at the time of writing, all 217 patterns across all 53
// repos matched, which is what made hard-failing safe to ship.
//
// USAGE
//   node scripts/check-download-patterns.mjs            # all HF download entries
//   node scripts/check-download-patterns.mjs --model krea_2_raw
//
// Exits non-zero if any declared pattern matches zero files. Public repos need no
// auth; a token is picked up from $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN if set, so a
// repo that is later gated still resolves.

import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MANIFEST = "config/manifests/builtin.models.jsonc";

// -- JSONC parsing (manifests carry // comments). Mirrors scripts/check-no-nc-weights.mjs. --
function stripJsoncComments(body) {
  let result = "";
  let inString = false;
  let escaped = false;
  for (let index = 0; index < body.length; index += 1) {
    const char = body[index];
    const next = body[index + 1];
    if (inString) {
      result += char;
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }
    if (char === '"') {
      inString = true;
      result += char;
      continue;
    }
    if (char === "/" && next === "/") {
      while (index < body.length && body[index] !== "\n") {
        index += 1;
      }
      result += "\n";
      continue;
    }
    if (char === "/" && next === "*") {
      index += 2;
      while (index < body.length && !(body[index] === "*" && body[index + 1] === "/")) {
        index += 1;
      }
      index += 1;
      continue;
    }
    result += char;
  }
  return result;
}

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

async function repoFiles(repo) {
  if (fileCache.has(repo)) return fileCache.get(repo);
  const token = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN;
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await fetch(`https://huggingface.co/api/models/${repo}?blobs=false`, { headers });
  if (!response.ok) {
    const result = { error: `HTTP ${response.status}` };
    fileCache.set(repo, result);
    return result;
  }
  const body = await response.json();
  const result = { files: (body.siblings ?? []).map((sibling) => sibling.rfilename) };
  fileCache.set(repo, result);
  return result;
}

async function main() {
  const only = process.argv.includes("--model")
    ? process.argv[process.argv.indexOf("--model") + 1]
    : null;

  const manifest = JSON.parse(stripJsoncComments(await readFile(path.join(root, MANIFEST), "utf8")));
  const failures = [];
  const unreachable = [];
  let repos = 0;
  let patterns = 0;

  for (const model of manifest.models ?? []) {
    if (only && model.id !== only) continue;
    for (const download of model.downloads ?? []) {
      if (download.provider !== "huggingface") continue;
      const declared = download.files ?? [];
      // An empty `files` list is a deliberate whole-repo download, not an omission —
      // there is no per-pattern claim to verify.
      if (declared.length === 0) continue;
      const { files, error } = await repoFiles(download.repo);
      if (error) {
        unreachable.push(`${model.id}/${download.variant ?? "-"}  ${download.repo}  (${error})`);
        continue;
      }
      repos += 1;
      for (const pattern of declared) {
        patterns += 1;
        const regexp = patternToRegExp(pattern);
        if (!files.some((file) => regexp.test(file))) {
          failures.push(`${model.id}/${download.variant ?? "-"}  ${download.repo}  ${pattern}`);
        }
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
    console.error(`\nEither the glob is wrong, or the tier is not published yet.`);
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
