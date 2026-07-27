// Unit tests for the inference-provenance scanner's pure surface (sc-15191).
//
// Two things are worth testing here without an inference clone:
//   * the MARKER vocabulary — it exists to catch honest port phrasings, and it regressed into
//     invisibility once already (sc-15138: `mirroring` / `from the reference` matched nothing, so a
//     whole new crate never entered the audit). The negative cases matter just as much: an
//     over-broad marker drags first-party crates into an inventory that then ASSERTS they are ports.
//   * the crate-prefix inventory's serialize/parse/hash round trip, which the fail-closed coverage
//     guard in check-license-coverage.mjs pins its population against.
import assert from "node:assert/strict";
import test from "node:test";

import {
  MARKER,
  cratePopulationSha256,
  parseCrates,
  serializeCrates,
} from "./scan-inference-provenance.mjs";

const matches = (text) => {
  MARKER.lastIndex = 0;
  return MARKER.test(text);
};

test("MARKER keeps the long-standing port vocabulary", () => {
  for (const phrase of [
    "//! Adapted from the reference implementation.",
    "// ported from diffusers",
    "//! A faithful Rust port of the sampler",
    "// vendored from candle-transformers",
    "// transcribed from Cephes",
    "// copied essentially verbatim from upstream",
  ]) {
    assert.ok(matches(phrase), `expected a marker match: ${phrase}`);
  }
});

test("MARKER catches the sc-15138 phrasings that made a whole crate invisible", () => {
  for (const phrase of [
    "//! Mirrors the reference `causal_model.py` block schedule.",
    "//! mirroring the reference KV-cache eviction",
    "/// Mirrors upstream `rope_params` exactly.",
    "// derived from the original torch implementation",
    "//! reimplemented from the diffusers pipeline",
    "// translated from the python prototype",
    "//! The 3-axis factorization from the reference is preserved.",
    "// based on the upstream scheduler",
    "// follows the reference ordering",
  ]) {
    assert.ok(matches(phrase), `expected a marker match: ${phrase}`);
  }
});

test("MARKER does NOT fire on ordinary technical English", () => {
  // Every one of these appears in first-party inference crates (catalogs, bundle composition roots,
  // the gen-core testkit). Matching them would force a ported-source area whose disposition claims
  // an upstream port that does not exist — a false statement in the audit, not a harmless extra row.
  for (const phrase of [
    "// mirrors the `gen_core` lib-name convention",
    "/// Asserted against the catalog's own compile-time answer, mirroring the gate above.",
    "/// a value derived from the seed and the clip's pixels",
    "/// The negatives are derived from the descriptor's advertised surface.",
    "// mirror padding on both edges",
    "// the request shape is derived from the descriptor's kind",
  ]) {
    assert.ok(!matches(phrase), `expected NO marker match: ${phrase}`);
  }
});

test("crate inventory round-trips through serialize/parse", () => {
  const crates = ["crates/a", "crates/b/c", "crates/d"];
  const parsed = parseCrates(serializeCrates(crates));
  assert.deepEqual(parsed, crates);
});

test("crate population hash is order- and content-sensitive", () => {
  const base = ["crates/a", "crates/b"];
  assert.equal(cratePopulationSha256(base), cratePopulationSha256(["crates/a", "crates/b"]));
  assert.notEqual(cratePopulationSha256(base), cratePopulationSha256(["crates/b", "crates/a"]));
  assert.notEqual(cratePopulationSha256(base), cratePopulationSha256([...base, "crates/c"]));
});

test("parseCrates rejects a malformed row rather than silently dropping a crate", () => {
  assert.throws(() => parseCrates("crates/a\ncrates/b c\n"), /malformed crate prefix row/);
});
