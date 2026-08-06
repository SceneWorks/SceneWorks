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
// Needs an inference clone containing every captured revision — CI has none, which is exactly why
// the result is checked in.

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

/**
 * Stamp `inferenceClosureDigest` onto every calibration binding in the JSONC manifest.
 *
 * Line-based on purpose: the manifest is JSONC and its comments carry the reasoning for individual
 * bindings, so a parse/serialise round trip would silently delete them. Each binding is located by
 * its `inferenceRevision` line and paired with the `provider` line inside the same object.
 *
 * `kreaTurboFit` also carries an `inferenceRevision` and is deliberately NOT stamped: it is the
 * separate third invalidation mechanism (`crates/sceneworks-worker/src/krea_control_fit.rs`, survey
 * sc-17775 §9.4), which this change does not reach. Any `inferenceRevision` line with no `provider`
 * in its object is reported rather than skipped quietly.
 */
export function stampManifest(body, digestFor) {
  const lines = body.split("\n");
  const out = [];
  const stamped = [];
  const skipped = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    // Two shapes occur: `inferenceRevision` alone on its line, and packed mid-line alongside
    // `sceneWorksRevision`/`matrixSourceRevision` (the five flux1 bindings). Inserting immediately
    // after the key handles both without reflowing anyone's formatting.
    const match = line.match(/"inferenceRevision":\s*"([0-9a-f]{40})",/);
    if (!match) {
      out.push(line);
      continue;
    }
    const revision = match[1];
    // Find this object's provider: it may sit later on the same line, or on a following line before
    // the object closes.
    let provider = line.slice(match.index).match(/"provider":\s*"([^"]+)"/)?.[1] ?? null;
    for (let scan = index + 1; !provider && scan < lines.length; scan += 1) {
      if (/^\s*[}\]]/.test(lines[scan])) break;
      provider = lines[scan].match(/"provider":\s*"([^"]+)"/)?.[1] ?? null;
    }
    if (!provider) {
      skipped.push(`line ${index + 1}: ${revision.slice(0, 8)} has no provider in its object`);
      out.push(line);
      continue;
    }
    // A provider id is not unique across backends, so find the enclosing `"mlx"` / `"candle"` block.
    let backend = null;
    for (let scan = index; !backend && scan >= 0; scan -= 1) {
      backend = lines[scan].match(/^\s*"(mlx|candle)":\s*\{/)?.[1] ?? null;
    }
    if (!backend) {
      skipped.push(`line ${index + 1}: ${revision.slice(0, 8)} is in no mlx/candle block`);
      out.push(line);
      continue;
    }
    const key = `${backend}:${provider}`;
    const already = line.match(/"inferenceClosureDigest":\s*"([0-9a-f]{64})",/)?.[1]
      ?? lines[index + 1]?.match(/"inferenceClosureDigest":\s*"([0-9a-f]{64})",/)?.[1];
    if (already) {
      const wanted = digestFor(key, revision);
      if (already === wanted) {
        out.push(line);
        continue;
      }
      // Re-derive in place on a digest-version bump rather than appending a second key.
      out.push(line.replace(already, wanted));
      if (!lines[index + 1]?.includes(already)) stamped.push(`${key}@${revision.slice(0, 8)}`);
      continue;
    }
    const insertAt = match.index + match[0].length;
    const digest = ` "inferenceClosureDigest": "${digestFor(key, revision)}",`;
    out.push(line.slice(0, insertAt) + digest + line.slice(insertAt));
    stamped.push(`${key}@${revision.slice(0, 8)}`);
  }
  return { body: out.join("\n"), stamped, skipped };
}

export async function main(argv = process.argv.slice(2)) {
  const value = (flag) => {
    const index = argv.indexOf(flag);
    return index === -1 ? undefined : argv[index + 1];
  };
  const repo = value("--repo");
  if (!repo) {
    console.error(
      "usage: node scripts/backfill-closure-digests.mjs --repo <inference-checkout> [--write]",
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
  for (const note of manifest.skipped) console.log(`  not a calibration binding — ${note}`);

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
