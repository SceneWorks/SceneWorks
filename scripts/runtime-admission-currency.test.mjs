import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// sc-22738 — Michael's standing rule, verbatim (2026-09-07): "The App Runtime should ALWAYS
// continue behaving as if the measurement were valid."
//
// Measurement currency — the per-provider inference compile-closure digest
// (`config/inference-provider-closures.json`), the per-model anchor loader-closure digest
// (`config/anchor-loader-closures.json`), `is_current`, "stale" anchors/bounds/records, the
// calibration bundle's version stamps — exists ONLY so the probe tooling knows what to re-capture.
// It must never change what a live request gets: a critical fix in shared engine code touches
// nearly every closure, and shipping it must neither wait on hours of re-measurement nor silently
// regress admission. Sibling rules already in force: no new gates; a disabled/unwired check script
// is never a blocker.
//
// This is the STRUCTURAL half of that rule, beside the other platform-review contracts. The
// behavioural half lives in `cargo test` (a stale exceeded bound still refuses; a stale anchor,
// curve and calibration record bind, match and grade exactly as current ones). Both are needed:
// a behavioural test proves the seam it drives, while this one proves nobody has re-introduced a
// currency comparison on a seam no test happens to drive. The allowlist is the tooling — nothing
// else may mention these identifiers in production code.

async function source(path) {
  return readFile(new URL(`../${path}`, import.meta.url), "utf8");
}

/** Every Rust module that decides admission or assembles the evidence admission reads. */
export const RUNTIME_ADMISSION_MODULES = Object.freeze([
  // The evidence types and lookups the worker reads.
  "crates/sceneworks-core/src/memory_anchor.rs",
  "crates/sceneworks-core/src/memory_calibration.rs",
  "crates/sceneworks-core/src/video_memory_curves.rs",
  // The gates and the shared selector.
  "crates/sceneworks-worker/src/memory_strategy.rs",
  "crates/sceneworks-worker/src/ladder_margin_policy.rs",
  "crates/sceneworks-worker/src/mlx_fit_gate.rs",
  "crates/sceneworks-worker/src/candle_memory_strategy.rs",
  "crates/sceneworks-worker/src/vram_gate.rs",
  "crates/sceneworks-worker/src/krea_control_fit.rs",
  "crates/sceneworks-worker/src/video_admission.rs",
  "crates/sceneworks-worker/src/video_jobs/wan.rs",
  // The capture-side mirrors that reproduce production admission before a probe render.
  "crates/sceneworks-memory-adapter/src/bin/mlx.rs",
  "crates/sceneworks-memory-adapter/src/bin/candle.rs",
]);

/**
 * The tooling that OWNS currency, and must keep it: the runner's `exceeded_current` /
 * `already_captured` grading, the matrix's per-anchor `current` and `summary.staleAnchors`, the
 * stale-lane report, and the closure derivation/stamper. Each entry names one identifier the file
 * must still carry, so the guard cannot pass by someone deleting currency from the tooling too.
 */
export const TOOLING_CURRENCY_OWNERS = Object.freeze({
  "scripts/measure-memory-catalog.mjs": ["exceeded_current", "--skip-current"],
  "scripts/generate-memory-matrix.mjs": ["staleAnchors"],
  "scripts/stale-lane-report.mjs": ["stale"],
  "scripts/anchor-loader-closure.mjs": ["--stamp-anchors"],
  "scripts/memory-calibration-harness.mjs": ["closureDigest"],
});

/**
 * Identifiers and expressions that grade measurement currency. A match in the PRODUCTION slice
 * of a runtime admission module is a re-introduced demotion. Definitions of the two tooling/test
 * accessors that legitimately remain in core (`pub fn packaged_closure_digest`,
 * `pub fn digest_for`) are exempted by `isDefinition`, because a definition compares nothing.
 */
export const BANNED_CURRENCY_PATTERNS = Object.freeze([
  /\bis_current\b/,
  /anchor_currency_matches/,
  /packaged_anchor_loader_closures/,
  /packaged_closure_digest\s*\(/,
  /\.digest_for\s*\(/,
  /expected_closure_digest/,
  /live_closure_digest/,
  /closure_digests\b/,
  /StaleEvidenceReason/,
  /StaleBundle/,
  /BundleLoad\b/,
  // gen-core's `MemoryEvidenceVerdict::Stale` is an upstream DIMENSION verdict label (a record
  // exists but did not verify against a provider-declared identity) and is not matched here.
  /\bEvidenceVerdict::Stale\b/,
  /LegacyAdmissionReason::Stale\w*/,
  /CandidateCurrency/,
  /StaleClosure/,
  /closure_is_stale/,
  /stale_admitted_peak_bytes/,
  /stale_minimum_host_bytes/,
  /stale_fallback_reason/,
  /UNCALIBRATED_CLOSURE/,
  /ClosureDigestLookup/,
  // Any comparison of a closure digest, whichever side it sits on.
  /closure_digest\b[^;{]*?(==|!=)/,
  /(==|!=)[^;{]*?\bclosure_digest\b/,
  /closureDigest"?\)?[^;{]*?(==|!=)/,
  /(==|!=)[^;{]*?closureDigest/,
]);

/**
 * Functions that compare a packaged artifact's captured digest against ITS OWN source records at
 * load — provenance self-consistency (the fitted curve cites the records it was fitted from),
 * never a comparison against the live ledger and never per request. Graded out, by name, so the
 * exemption is visible here rather than implied by a looser pattern.
 */
export const PROVENANCE_SELF_CONSISTENCY = Object.freeze({
  "crates/sceneworks-core/src/video_memory_curves.rs": ["source_record_matches_curve"],
});

/** Blank out the bodies of the named functions so their lines are not graded. */
export function withoutFunctions(text, names) {
  let result = text;
  for (const name of names) {
    const at = result.indexOf(`fn ${name}(`);
    assert.ok(at >= 0, `${name} must still exist to be exempted`);
    const end = result.indexOf("\n}\n", at);
    assert.ok(end > at, `${name} must close`);
    const body = result.slice(at, end);
    result = result.slice(0, at) + body.replace(/[^\n]/g, " ") + result.slice(end);
  }
  return result;
}

/** The production slice: everything before the `#[cfg(test)] mod tests` module (test modules may
 *  still construct stale fixtures, that is their job). Earlier `#[cfg(test)]` items are the small
 *  injection seams production functions consult and are graded with the production code. A file
 *  with no test module is graded whole. */
export function productionSlice(text) {
  const boundary = /#\[cfg\(test\)\]\s*\n\s*mod tests\b/.exec(text);
  return boundary ? text.slice(0, boundary.index) : text;
}

function isComment(line) {
  return /^\s*(\/\/|\*|\/\*)/.test(line);
}

function isDefinition(line) {
  return /^\s*pub(\(crate\))? (const )?fn (packaged_closure_digest|digest_for)\b/.test(line);
}

/** Every banned match in the production slice of one module, as `line: text` strings. */
export function currencyViolations(text, exemptFunctions = []) {
  const violations = [];
  withoutFunctions(productionSlice(text), exemptFunctions)
    .split("\n")
    .forEach((line, index) => {
      if (isComment(line) || isDefinition(line)) {
        return;
      }
      for (const pattern of BANNED_CURRENCY_PATTERNS) {
        if (pattern.test(line)) {
          violations.push(`${index + 1}: ${line.trim()}  [${pattern}]`);
          break;
        }
      }
    });
  return violations;
}

/** Struct bodies that must carry no currency field — the "cannot express stale" half. */
export const CURRENCY_FREE_STRUCTS = Object.freeze({
  "crates/sceneworks-worker/src/memory_strategy.rs": ["RequestScope", "Candidate"],
  "crates/sceneworks-worker/src/ladder_margin_policy.rs": ["AdmissionSubject"],
  "crates/sceneworks-core/src/video_memory_curves.rs": ["VideoCurveQuery", "VideoCurveEvaluation"],
  "crates/sceneworks-worker/src/video_admission.rs": ["VideoRequestIdentity", "VideoAdmissionInputs"],
});

export function structBody(text, name) {
  const match = new RegExp(`pub(?:\\(crate\\))? struct ${name}(?:<[^>]*>)?\\s*\\{`).exec(text);
  assert.ok(match, `${name} must still be declared`);
  const start = match.index + match[0].length;
  const end = text.indexOf("\n}", start);
  return text.slice(start, end);
}

test("no runtime admission module grades measurement currency", async () => {
  const offenders = [];
  for (const path of RUNTIME_ADMISSION_MODULES) {
    const violations = currencyViolations(
      await source(path),
      PROVENANCE_SELF_CONSISTENCY[path] ?? [],
    );
    if (violations.length > 0) {
      offenders.push(`${path}\n  ${violations.join("\n  ")}`);
    }
  }
  assert.deepEqual(
    offenders,
    [],
    "measurement currency is a re-capture signal for the probe tooling only; the runtime must " +
      "behave as if every measurement were valid (sc-22738)",
  );
});

test("the evidence-carrying request and candidate types cannot express staleness", async () => {
  for (const [path, structs] of Object.entries(CURRENCY_FREE_STRUCTS)) {
    const text = productionSlice(await source(path));
    for (const name of structs) {
      const body = structBody(text, name);
      assert.doesNotMatch(
        body.replace(/^\s*\/\/.*$/gm, ""),
        /closure|stale|current/i,
        `${path}#${name} carries a currency field`,
      );
    }
  }
});

test("the exceeded-bound lookup takes no loader closures", async () => {
  const text = productionSlice(await source("crates/sceneworks-core/src/memory_anchor.rs"));
  const signature = /pub fn binding_exceeded_bound\(([^)]*)\)/.exec(text);
  assert.ok(signature, "binding_exceeded_bound must exist");
  assert.doesNotMatch(signature[1], /closure/i, "a stale bound refuses exactly what it refused when measured");
});

test("the probe tooling still owns currency", async () => {
  for (const [path, markers] of Object.entries(TOOLING_CURRENCY_OWNERS)) {
    const text = await source(path);
    for (const marker of markers) {
      assert.ok(text.includes(marker), `${path} must keep ${marker}: the tooling decides what to re-capture`);
    }
  }
});

test("the provenance exemption is exactly one load-time self-consistency check", async () => {
  // The exemption must not widen silently: the same comparison anywhere else in the file is
  // still a violation, and the exempted function must still be the one comparing a curve to its
  // own source records (not to the live ledger).
  const path = "crates/sceneworks-core/src/video_memory_curves.rs";
  const text = await source(path);
  const exempt = PROVENANCE_SELF_CONSISTENCY[path];
  assert.ok(currencyViolations(text).length > 0, "the exempted comparison exists");
  assert.deepEqual(currencyViolations(text, exempt), []);
  const body = withoutFunctions(text, []).slice(text.indexOf("fn source_record_matches_curve("));
  assert.match(body.slice(0, body.indexOf("\n}\n")), /value_at_str\(record, &\["repositories", "inference", "closureDigest"\]\)/);
  assert.doesNotMatch(body.slice(0, body.indexOf("\n}\n")), /packaged_closure_digest|inference-provider-closures/);
});

test("the guard itself catches a re-introduced currency comparison", async () => {
  // Self-check, in the style of the other platform contracts: each mutation below is a way a
  // future change could put currency back, and each must be caught — in production code, and
  // NOT in a test module or a comment.
  const selector = await source("crates/sceneworks-worker/src/memory_strategy.rs");
  assert.deepEqual(currencyViolations(selector), []);
  const mutations = [
    "    if candidate.closure_digest != request.expected_closure_digest {\n        return Ok(CandidateGrade::Estimate);\n    }\n",
    "    let live = sceneworks_core::memory_calibration::packaged_closure_digest(\"mlx\", route);\n",
    "    if !anchor.is_current(closures) { return None; }\n",
    "    if record.repositories.inference.closure_digest.as_deref() == Some(expected) {\n",
    "    if value_at_str(record, &[\"closureDigest\"]) != Some(curve.closure_digest.as_str()) {\n",
    "    let stale = crate::memory_strategy::stale_admitted_peak_bytes(backend, peak);\n",
  ];
  const anchor = "fn candidate_exclusion(";
  const at = selector.indexOf(anchor);
  assert.ok(at > 0);
  for (const mutation of mutations) {
    const mutated = selector.slice(0, at) + mutation + selector.slice(at);
    assert.ok(
      currencyViolations(mutated).length > 0,
      `the guard must flag: ${mutation.trim()}`,
    );
    // The same text in a comment, or after the test-module boundary, is not a violation.
    const commented = selector.slice(0, at) + "    // " + mutation.trimStart() + selector.slice(at);
    assert.deepEqual(currencyViolations(commented), [], "a comment is not a comparison");
    const testBoundary = selector.indexOf("#[cfg(test)]\nmod tests");
    assert.ok(testBoundary > 0, "the selector has a test module");
    assert.deepEqual(
      currencyViolations(`${selector.slice(0, testBoundary)}#[cfg(test)]\nmod tests {\n${mutation}}\n`),
      [],
      "a test fixture may construct a stale digest",
    );
  }
});
