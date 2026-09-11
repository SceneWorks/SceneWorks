// Stamp each calibration record with the provider closure digest it was CAPTURED under (sc-17774).
//
// Currency is `record.repositories.inference.closureDigest === <the live digest for that provider>`.
// The live half lives in `config/inference-provider-closures.json`; this script produces the
// captured half, by re-deriving each record's provider closure at the revision the record was
// actually measured at.
//
// It is idempotent and safe to re-run: a record whose digest already matches the derivation is left
// alone, and a record whose digest DISAGREES is reported as a conflict rather than overwritten,
// because silently rewriting a captured digest is how a stale measurement would be laundered into a
// current one.
//
// Needs an inference clone containing every CAPTURED revision, which is why this runs at capture /
// pin-bump time rather than in CI: `check.yml` shallow-fetches only the pinned revision, and a
// record's captured digest is derived at whatever revision it was measured at. The result is checked
// in and the LIVE half is re-derived in CI (scripts/inference-closure-digest.mjs --check).

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { digestsAtRevision, resolveRevision } from "./inference-closure-digest.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const BUNDLE = "docs/generated/memory-calibration-evidence.json";
const CLOSURES = "config/inference-provider-closures.json";
const MANIFEST = "config/manifests/builtin.models.jsonc";

export function recordsNeedingDigest(bundle) {
  // Only records that can ever be `current` need one; fixtures, candidates and gated records never
  // reach the currency comparison.
  return bundle.records.filter((record) =>
    ["complete", "runtime_complete"].includes(record.status) &&
    record.evidenceScope === "authoritative",
  );
}

/** Group the records by the (provider, revision) pair whose digest they need. */
export function digestWorkload(records) {
  const byRevision = new Map();
  for (const record of records) {
    const revision = record.repositories.inference.revision;
    if (!byRevision.has(revision)) byRevision.set(revision, new Set());
    byRevision.get(revision).add(`${record.backend}:${record.target.provider}`);
  }
  return byRevision;
}

const REVISION_KEY = /"inferenceRevision":\s*"([0-9a-f]{40})"(\s*,)?/;
const DIGEST_KEY = /"inferenceClosureDigest":\s*"([0-9a-f]{64})"/;
const PROVIDER_KEY = /"provider":\s*"([^"]+)"/;

/**
 * Column at which this line's `//` comment tail begins, or `Infinity`.
 *
 * Quote-aware, because the manifest is full of `"https://…"` values and `"q4/*"` globs — a naive
 * `indexOf("//")` would declare half of those lines commented.
 */
function commentColumn(line) {
  let inString = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (inString) {
      if (char === "\\") index += 1;
      else if (char === '"') inString = false;
      continue;
    }
    if (char === '"') inString = true;
    else if (char === "/" && line[index + 1] === "/") return index;
  }
  return Infinity;
}

/**
 * Same-length copy of `line` with string contents and the comment tail blanked out.
 *
 * Brace depth is counted off this, never off the raw line. Both kinds of decoration in this file
 * carry braces that are not structure: prose comments quote shapes while explaining them (see the
 * `branchTierByBaseTier` note above `candle.control`), and string values hold globs like `"q4/*"`
 * and URLs. Counting either would move an object boundary and silently mis-scope a binding.
 */
function maskLine(line) {
  const masked = line.split("");
  let inString = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (inString) {
      masked[index] = " ";
      if (char === "\\") {
        if (index + 1 < line.length) masked[index + 1] = " ";
        index += 1;
      } else if (char === '"') inString = false;
      continue;
    }
    if (char === '"') {
      masked[index] = " ";
      inString = true;
    } else if (char === "/" && line[index + 1] === "/") {
      for (let tail = index; tail < line.length; tail += 1) masked[tail] = " ";
      return masked.join("");
    }
  }
  return masked.join("");
}

/**
 * Every `"inferenceClosureDigest": "<64hex>"` occurrence outside a `//` comment tail, as
 * `{ line, column }` (both zero-based), scanned per occurrence rather than per line.
 *
 * This is THE definition of "a digest the manifest carries": the orphan scan in `stampManifest`
 * consumes it, and it is exported (sc-18208) so `stale-lane-report.test.mjs` can derive its
 * population-coverage count from the gate's own predicate. The test used to re-count with a
 * resembling regex over the raw body, which also matched digest shapes QUOTED IN COMMENTS — clean
 * today, but one doc comment quoting the shape away from false-redding a required check.
 */
export function digestOccurrences(lines) {
  const occurrences = [];
  lines.forEach((line, index) => {
    const limit = commentColumn(line);
    for (const match of line.matchAll(new RegExp(DIGEST_KEY, "g"))) {
      if (match.index >= limit) break;
      occurrences.push({ line: index, column: match.index });
    }
  });
  return occurrences;
}

/**
 * Refuse to touch a manifest containing `/* … *\/` block comments.
 *
 * Everything here is line-scoped, so a comment spanning lines would be read as code: its braces
 * would move object boundaries and a 64-hex value quoted inside it would be rewritten as if it
 * were a digest. JSONC permits them and `scripts/lib/jsonc.mjs` reads them, so "the manifest has
 * none today" is a fact about today. Failing loudly is the honest response to syntax this cannot
 * scope correctly; corrupting shipped runtime data is not.
 */
function assertNoBlockComments(lines) {
  const line = lines.findIndex((text) => maskLine(text).includes("/*"));
  if (line !== -1) {
    throw new Error(
      `line ${line + 1}: the manifest contains a /* block comment, which this line-based stamper ` +
        "cannot scope. Rewrite it as // comments, or teach commentColumn/maskLine to carry block " +
        "state across lines.",
    );
  }
}

/**
 * Every `[line, from, to]` slice that lies DIRECTLY inside the object enclosing `lines[index]`,
 * walking outward in both directions from column `anchor` and stopping at the braces that open and
 * close that object.
 *
 * "Directly" is the whole point: slices nested one level deeper are excluded, so a sub-object's
 * `inferenceClosureDigest` is never mistaken for the binding's own, and the walk stops dead at the
 * enclosing close, so a sibling's is never claimed either. sc-17989: the previous version stopped
 * at the first line merely BEGINNING with `}` or `]`, which is a nested close as often as the real
 * one — it broke early on any block with a nested sub-object above its digest (inserting a second
 * `inferenceClosureDigest`) and ran on past the real close when a sibling object opened mid-line.
 */
function enclosingSpans(lines, index, anchorStart, anchorEnd) {
  const spans = [];
  const clipped = (line, from, to) => [line, from, Math.min(to, commentColumn(lines[line]))];

  let depth = 0;
  for (let scan = index; scan < lines.length; scan += 1) {
    const masked = maskLine(lines[scan]);
    let start = scan === index ? anchorEnd : 0;
    let spanStart = start;
    let closed = false;
    for (let column = start; column < masked.length; column += 1) {
      const char = masked[column];
      if (char === "{" || char === "[") {
        if (depth === 0) spans.push(clipped(scan, spanStart, column));
        depth += 1;
      } else if (char === "}" || char === "]") {
        depth -= 1;
        if (depth < 0) {
          spans.push(clipped(scan, spanStart, column));
          closed = true;
          break;
        }
        if (depth === 0) spanStart = column + 1;
      }
    }
    if (closed) break;
    if (depth === 0) spans.push(clipped(scan, spanStart, lines[scan].length));
  }

  depth = 0;
  for (let scan = index; scan >= 0; scan -= 1) {
    const masked = maskLine(lines[scan]);
    const end = scan === index ? anchorStart : lines[scan].length;
    let spanEnd = end;
    let opened = false;
    for (let column = end - 1; column >= 0; column -= 1) {
      const char = masked[column];
      if (char === "}" || char === "]") {
        if (depth === 0) spans.push(clipped(scan, column + 1, spanEnd));
        depth += 1;
      } else if (char === "{" || char === "[") {
        depth -= 1;
        if (depth < 0) {
          spans.push(clipped(scan, column + 1, spanEnd));
          opened = true;
          break;
        }
        if (depth === 0) spanEnd = column;
      }
    }
    if (opened) break;
    if (depth === 0) spans.push(clipped(scan, 0, spanEnd));
  }

  return spans;
}

/**
 * First match of `pattern` inside `spans`, as `{ line, column, value }` with `column` absolute on
 * that line. Absolute because a digest is rewritten by position, not by `String.replace` — the same
 * 64-hex value can legitimately appear twice on one line and only one of them is this binding's.
 */
function findInSpans(lines, spans, pattern) {
  for (const [line, from, to] of spans) {
    if (to <= from) continue;
    const match = lines[line].slice(from, to).match(pattern);
    if (match) return { line, column: from + match.index, value: match[1] };
  }
  return null;
}

/**
 * The `"mlx"` / `"candle"` block this binding sits in, or `null`.
 *
 * Walks OUTWARD by brace depth rather than scanning up line by line. A line scan finds the nearest
 * `"mlx": {` header above the binding whether or not that block is still open — the module header
 * below records this exact failure from the first draft, where a `candle` header was paired with an
 * `mlx` binding further down and invented a `candle:qwen_image` lane that does not exist.
 */
function enclosingBackend(lines, index, anchor) {
  let depth = 0;
  for (let scan = index; scan >= 0; scan -= 1) {
    const masked = maskLine(lines[scan]);
    const end = scan === index ? anchor : lines[scan].length;
    for (let column = Math.min(end, masked.length) - 1; column >= 0; column -= 1) {
      const char = masked[column];
      if (char === "}" || char === "]") depth += 1;
      else if (char === "{" || char === "[") {
        if (depth > 0) {
          depth -= 1;
          continue;
        }
        // Crossed the brace that opens the enclosing object: if it belongs to a backend key this is
        // the answer, otherwise carry on outward at the parent's level.
        const backend = lines[scan].slice(0, column).match(/"(mlx|candle)":\s*$/)?.[1];
        if (backend) return backend;
      }
    }
  }
  return null;
}

/**
 * Stamp `inferenceClosureDigest` onto every calibration binding in the JSONC manifest.
 *
 * Line-based on purpose: the manifest is JSONC and its comments carry the reasoning for individual
 * bindings, so a parse/serialise round trip would silently delete them. Each binding is located by
 * its `inferenceRevision` key and paired with the `provider` key in the same object — see
 * `enclosingSpans` for what "the same object" means and why the naive version of it corrupts data.
 *
 * Comment tails are excluded from every match, not just from brace counting. The manifest narrates
 * digest provenance in prose ("the closure of `candle-gen-krea` these measurements were taken
 * against"), so a scan that reads comments will happily rewrite a 64-hex value quoted inside one
 * and leave the block with no real digest key at all.
 *
 * `turboFit` and `candle.control` (the two whole-block fits, formerly separate per-model
 * invalidation mechanisms — survey sc-17775 §9.4) are stamped here too, since sc-17989 gave them
 * the `provider`/`inferenceRevision` keys the locator pairs on.
 *
 * Returns `stamped` (bindings whose digest was written or rewritten), `skipped` (a would-be
 * closure binding carrying a provider or digest that cannot be paired with the other binding
 * fields), `orphans` (an
 * `inferenceClosureDigest` no located binding reached — a digest that has fallen out of the gate,
 * which is what losing an `inferenceRevision` or even just its trailing comma looks like), and
 * `located`.
 *
 * `located` is the binding POPULATION this locator found — `{ key, revision, digest, line }` per
 * binding, `digest` being the value already present (`null` when one had to be inserted). It is
 * reported so a reader of the manifest's admission surface can take THIS population rather than
 * re-deriving one: sc-18098's stale-lane report did re-derive it, walked only
 * `<model>.<backend>.calibrations[]`, and thereby missed `turboFit` and `candle.control` — the
 * exact two whole-block fits sc-17989 brought under the gate, reported as "never captured" while
 * they were stale production bindings. Coverage belongs to one locator, and this is it.
 */
export function stampManifest(body, digestFor) {
  const lines = body.split("\n");
  assertNoBlockComments(lines);
  const stamped = [];
  const skipped = [];
  const located = [];
  // Keyed by `line:column` of the `"inferenceClosureDigest"` key, not by line. Two bindings can
  // share a line — the five flux1 bindings already pack revision and digest together, so packing
  // two records onto one line is the same editorial style one step further — and a line-granular
  // set would mark BOTH digests reached the moment one binding claimed either.
  const claimed = new Set();

  for (let index = 0; index < lines.length; index += 1) {
    // Two shapes occur: `inferenceRevision` alone on its line, and packed mid-line alongside
    // `sceneWorksRevision`/`matrixSourceRevision` (the five flux1 bindings). Inserting immediately
    // after the key handles both without reflowing anyone's formatting. The cursor loop is what
    // makes a SECOND binding on the same line visible; matching once per line silently left it
    // unstamped and — because its digest looked claimed — unreported.
    for (let cursor = 0; cursor < lines[index].length; ) {
      const limit = commentColumn(lines[index]);
      const match = lines[index].slice(cursor).match(REVISION_KEY);
      if (!match) break;
      const start = cursor + match.index;
      if (start >= limit) break;
      const end = start + match[0].length;
      cursor = end;

      const revision = match[1];
      const spans = enclosingSpans(lines, index, start, end);
      const found = findInSpans(lines, spans, DIGEST_KEY);
      const provider = findInSpans(lines, spans, PROVIDER_KEY)?.value ?? null;
      if (!provider) {
        // `inferenceRevision` is also a legitimate provenance field outside memory-fit closure
        // bindings (for example StarVector's terminal-candidate identity). With neither a provider
        // nor an inferenceClosureDigest in the same object, it is outside this stamper's population.
        // A digest without a provider remains a hard error below and in the orphan scan.
        if (!found) continue;
        skipped.push(`line ${index + 1}: ${revision.slice(0, 8)} has no provider in its object`);
        continue;
      }
      // A provider id is not unique across backends, so find the enclosing mlx/candle block.
      const backend = enclosingBackend(lines, index, start);
      if (!backend) {
        skipped.push(`line ${index + 1}: ${revision.slice(0, 8)} is in no mlx/candle block`);
        continue;
      }

      const key = `${backend}:${provider}`;
      if (found) {
        const seen = `${found.line}:${found.column}`;
        if (claimed.has(seen)) {
          skipped.push(
            `line ${index + 1}: ${revision.slice(0, 8)} shares its digest with another ` +
              `inferenceRevision in the same object (line ${found.line + 1})`,
          );
          continue;
        }
        claimed.add(seen);
        located.push({ key, revision, digest: found.value, line: index + 1 });
        const wanted = digestFor(key, revision);
        if (found.value !== wanted) {
          // Re-derive in place on a digest-version bump rather than appending a second key. By
          // position, not `String.replace` — the same hex can appear twice on one line.
          const at = lines[found.line].indexOf(found.value, found.column);
          lines[found.line] =
            lines[found.line].slice(0, at) + wanted + lines[found.line].slice(at + found.value.length);
          stamped.push(`${key}@${revision.slice(0, 8)}`);
        }
        continue;
      }
      // No digest yet. Insert one right after the revision key, supplying the separating comma when
      // the revision is the object's last key and therefore has none of its own.
      const lead = match[2] ? ' "' : ', "';
      located.push({ key, revision, digest: null, line: index + 1 });
      const digest = match[2]
        ? ` "inferenceClosureDigest": "${digestFor(key, revision)}",`
        : `, "inferenceClosureDigest": "${digestFor(key, revision)}"`;
      lines[index] = lines[index].slice(0, end) + digest + lines[index].slice(end);
      claimed.add(`${index}:${end + lead.length - 1}`);
      stamped.push(`${key}@${revision.slice(0, 8)}`);
      cursor = end + digest.length;
    }
  }

  // A digest no binding reached is outside the gate even though it looks graded. That is what a
  // dropped `inferenceRevision` leaves behind, and reporting it is the difference between "the gate
  // covers everything" being enforced and being asserted. Scanned per occurrence rather than per
  // line, for the same reason `claimed` is keyed by column.
  const orphans = [];
  for (const { line, column } of digestOccurrences(lines)) {
    if (claimed.has(`${line}:${column}`)) continue;
    orphans.push(
      `line ${line + 1}: an inferenceClosureDigest no inferenceRevision/provider pair reaches`,
    );
  }

  return { body: lines.join("\n"), stamped, skipped, orphans, located };
}

/**
 * The `--verify` verdict for one run: an exit code and the lines explaining it.
 *
 * Split out of `main` so the gate's decision — the whole point of this script — is reachable
 * without a real inference clone, and therefore testable. `main` only prints what this returns.
 *
 * Coverage is checked BEFORE drift on purpose: "this digest is not graded at all" outranks "this
 * graded digest moved", and a run reporting only the second while the first is true is how
 * `turboFit` and `candle.control` sat outside the gate through the whole of sc-17774.
 */
export function verifyOutcome({ recordDrift, manifest }) {
  const ungated = [...manifest.skipped, ...manifest.orphans];
  if (ungated.length) {
    return {
      code: 1,
      errors: [
        "manifest digests are outside this gate — an `inferenceRevision` the stamper cannot pair " +
          "with a `provider`, or an `inferenceClosureDigest` no such pair reaches:",
        ...ungated.map((note) => `  ${note}`),
      ],
    };
  }
  const drift = [
    recordDrift ? `${recordDrift} record digest(s)` : null,
    manifest.stamped.length ? `${manifest.stamped.length} manifest binding(s)` : null,
  ].filter(Boolean);
  if (drift.length) {
    return {
      code: 1,
      errors: [
        `captured closure digests do not match their own revisions: ${drift.join(" and ")} would ` +
          "change. A checked-in digest that disagrees with the source it was derived from grades " +
          "nothing. Re-derive with scripts/backfill-closure-digests.mjs --repo <inference> --write.",
      ],
    };
  }
  return { code: 0, errors: [] };
}

export async function main(argv = process.argv.slice(2)) {
  const value = (flag) => {
    const index = argv.indexOf(flag);
    return index === -1 ? undefined : argv[index + 1];
  };
  const repo = value("--repo");
  // `--revisions` needs no clone: it answers "which inference revisions must be present for this
  // check to run at all", which is a pure function of the bundle and the manifest. CI uses it to
  // shallow-fetch exactly those before running the verification pass below (sc-17935).
  const listRevisions = argv.includes("--revisions");
  if (!repo && !listRevisions) {
    console.error(
      "usage: node scripts/backfill-closure-digests.mjs --repo <inference-checkout> " +
        "[--write|--verify]\n" +
        "       node scripts/backfill-closure-digests.mjs --revisions",
    );
    return 2;
  }

  const bundlePath = path.join(ROOT, BUNDLE);
  const bundle = JSON.parse(await readFile(bundlePath, "utf8"));
  const closures = JSON.parse(await readFile(path.join(ROOT, CLOSURES), "utf8"));
  const declared = Object.fromEntries(
    Object.entries(closures.providers).map(([provider, entry]) => [provider, entry.crate]),
  );

  const manifestPath = path.join(ROOT, MANIFEST);
  const manifestBody = await readFile(manifestPath, "utf8");

  const records = recordsNeedingDigest(bundle);
  const workload = digestWorkload(records);

  // The shipped manifest's bindings are what the RUNTIME reads, so they need the same stamp and can
  // reference revisions the evidence bundle does not. Collect what they need by running the stamper
  // in a dry pass — reusing its enclosing-block scan rather than a second, subtly different regex.
  // The first draft did use a separate forward regex and it paired a `candle` block header with an
  // `mlx` binding further down, inventing a `candle:qwen_image` lane that does not exist.
  stampManifest(manifestBody, (key, revision) => {
    if (!workload.has(revision)) workload.set(revision, new Set());
    workload.get(revision).add(key);
    return "0".repeat(64);
  });

  if (listRevisions) {
    for (const revision of [...workload.keys()].sort()) console.log(revision);
    return 0;
  }

  const derived = new Map();
  for (const [revision, providers] of workload) {
    const resolved = resolveRevision(path.resolve(repo), revision);
    const missing = [...providers].filter((provider) => !declared[provider]);
    if (missing.length) {
      throw new Error(
        `providers ${missing.join(", ")} have records but no declaration in ${CLOSURES}. ` +
          "Add their inference crate and regenerate.",
      );
    }
    const wanted = Object.fromEntries([...providers].map((provider) => [provider, declared[provider]]));
    const digests = digestsAtRevision({ repo: path.resolve(repo), revision: resolved, providers: wanted });
    for (const [provider, entry] of digests) derived.set(`${provider}@${revision}`, entry.digest);
  }

  let stamped = 0;
  let unchanged = 0;
  // A CLOSURE_DIGEST_VERSION bump legitimately re-derives the same underlying fact, so `--restamp`
  // exists for that. It is NOT a way past a genuine conflict: rewriting a captured digest that
  // disagrees with its own revision would launder a stale measurement into a current one.
  let restamped = 0;
  const conflicts = [];
  for (const record of records) {
    const key = `${record.backend}:${record.target.provider}@${record.repositories.inference.revision}`;
    const digest = derived.get(key);
    const existing = record.repositories.inference.closureDigest;
    if (existing === digest) {
      unchanged += 1;
      continue;
    }
    if (existing && existing !== digest && !argv.includes("--restamp")) {
      conflicts.push(`${record.id}: recorded ${existing.slice(0, 12)}, derives ${digest.slice(0, 12)}`);
      continue;
    }
    if (existing && existing !== digest) restamped += 1;
    record.repositories.inference.closureDigest = digest;
    stamped += 1;
  }

  if (conflicts.length) {
    console.error(
      "refusing to overwrite captured closure digests — a captured digest that disagrees with its " +
        "revision means the record or the declaration is wrong, and overwriting would launder it:",
    );
    for (const conflict of conflicts) console.error(`  ${conflict}`);
    return 1;
  }

  const manifest = stampManifest(manifestBody, (provider, revision) => {
    const digest = derived.get(`${provider}@${revision}`);
    if (!digest) throw new Error(`no derived digest for ${provider}@${revision.slice(0, 8)}`);
    return digest;
  });

  const summary = [...workload].map(([revision, providers]) =>
    `${revision.slice(0, 8)} (${[...providers].sort().join(", ")})`,
  );
  console.log(`revisions derived: ${summary.join("; ")}`);
  console.log(`records: ${stamped} stamped, ${unchanged} already correct, ${records.length} total`);
  console.log(
    `manifest bindings: ${manifest.stamped.length} stamped` +
      (restamped ? `; ${restamped} record digests RESTAMPED (--restamp)` : ""),
  );
  // Printed on BOTH paths and before any failure return, so one run reports every problem it can
  // see. Reporting these only after the drift check would hide a fallen-out digest behind an
  // unrelated drift, and only surface it on the re-run after that drift was fixed.
  for (const note of manifest.skipped) console.log(`  not a calibration binding — ${note}`);
  for (const note of manifest.orphans) console.log(`  ungraded digest — ${note}`);

  // sc-17935: `--verify` is the CI spelling. A plain dry run PRINTS what it would change and still
  // exits 0, which is right for a pin-bump preview and useless as a gate — a hand-edited manifest
  // binding was reported and passed. Verification has to fail on any drift, including the drift a
  // restamp would silently absorb, so it deliberately does not honour `--restamp`.
  if (argv.includes("--verify")) {
    const outcome = verifyOutcome({ recordDrift: stamped, manifest });
    for (const line of outcome.errors) console.error(line);
    if (outcome.code === 0) {
      console.log(
        `captured closure digests match their revisions (${records.length} records + every one of ` +
          "the manifest's digests, none unreachable)",
      );
    }
    return outcome.code;
  }

  if (!argv.includes("--write")) {
    console.log("(dry run — pass --write to update the bundle and manifest)");
    return 0;
  }
  await writeFile(bundlePath, `${JSON.stringify(bundle, null, 2)}\n`);
  await writeFile(manifestPath, manifest.body);
  console.log(`wrote ${BUNDLE} and ${MANIFEST}`);
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = await main();
}
