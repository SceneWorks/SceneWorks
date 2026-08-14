/**
 * Semantic source bodies for generated-artifact provenance (sc-16268; extends sc-16129).
 *
 * `docs/generated/memory-matrix.*` stamps a
 * `source-tree:<sha256>` fingerprint over the sources it is derived from. That fingerprint is a
 * STALENESS TRIPWIRE — evidence captured under a different SceneWorks revision is not "current" — so
 * it deliberately covers the WHOLE source, not just the handful of values the generators parse out
 * of it. A change to `HEADROOM_GB`, which no generator parses but every memory number depends on,
 * must still rotate it.
 *
 * Hashing the RAW text overshoots: a doc-comment-only edit rotates the fingerprint and forces a full
 * regeneration whose diff contains zero value changes. sc-16129 fixed that for the JSONC manifest by
 * hashing the parsed value. This module generalises the same principle to the rest:
 *
 *   - JSON / JSONC   -> hash `JSON.stringify(parsed)`: comments and formatting are not the contract.
 *   - Rust / JS / TOML -> hash the text with its INERT LINES REMOVED.
 *
 * ## What "inert" means, and why the quote guard is the whole safety argument
 *
 * A line is inert when it is blank, or when its first non-whitespace characters are the language's
 * line-comment marker AND it contains no `"`. The quote half is not decoration. Every parser in these generators
 * extracts DOUBLE-QUOTED STRING LITERALS (`EXPECTED_IMAGE_IDS`, `MODEL_TABLE` rows, the MLX
 * sequential-capability sweep, InstantID's dense Candle tier, the `candle-kernels` `rev = "…"` pin),
 * so a comment line with no `"` cannot contribute a parsed value — dropping it from the hash cannot
 * hide a semantic change. A commented-out line that DOES carry a literal keeps its quote, stays
 * hashed, and still trips the tripwire. On the current tree that guard costs ~3% of comment lines
 * (77 of 2,839) and leaves ~18% of the hashed Rust/TOML text inert.
 *
 * Line-based, never inline: an inline stripper would have to know that the `//` in a URL inside a
 * string literal is not a comment. Block comments (`/* … *\/`) are deliberately NOT stripped for the
 * same reason — doing it safely needs a real tokenizer, and they are a small minority here. They
 * churn the fingerprint; that is noisy-but-safe, which is the correct side to err on.
 *
 * Two known, accepted blind spots, both requiring a MULTI-LINE STRING LITERAL: a blank line or a
 * quote-free `//` line inside a Rust raw string is dropped from the hash. In these sources those
 * literals are JSON test fixtures, where neither carries meaning. Anything a generator actually
 * parses is a single-line double-quoted literal, which the quote guard already protects.
 *
 * The invariant that keeps this honest is enforced by test, not by inspection:
 * `scripts/generate-memory-matrix.test.mjs` regenerates the entire matrix from fully
 * comment-stripped sources and asserts it is IDENTICAL to the baseline. That fails the day a parser
 * starts depending on comment text (one did — see `parseMlxSequentialEngines`, re-anchored on code
 * by sc-16268) or a stripped line starts carrying meaning.
 */

import path from "node:path";

import { stripJsoncComments } from "./jsonc.mjs";

/** Line-comment marker per source language. An unlisted extension is a deliberate failure. */
const LINE_COMMENT_MARKERS = Object.freeze({
  ".rs": "//",
  ".mjs": "//",
  ".js": "//",
  ".toml": "#",
});

const JSON_EXTENSIONS = Object.freeze([".json", ".jsonc"]);

/** Normalise line endings so a fingerprint does not depend on the checkout's platform. */
export function canonicalSourceText(body) {
  return body.replace(/\r\n?/g, "\n");
}

/**
 * True when `line` cannot carry a value any generator parses: a blank line, or a whole-line comment
 * with no `"`. Blank lines are in for the same reason comments are — a comment block is almost never
 * added without one, so leaving them hashed would keep exactly the churn this removes. Exported for
 * the tests that pin the quote guard.
 */
export function isInertLine(line, marker) {
  const trimmed = line.trimStart();
  if (!trimmed) return true;
  return trimmed.startsWith(marker) && !trimmed.includes('"');
}

/** Drop inert lines. Idempotent, so re-stripping a stripped body is a no-op. */
export function stripInertLines(body, marker) {
  return canonicalSourceText(body)
    .split("\n")
    .filter((line) => !isInertLine(line, marker))
    .join("\n");
}

/**
 * The body to HASH for `relativePath` — its semantic content, not its bytes. The caller keeps using
 * the raw body for parsing; only provenance reads this.
 *
 * Throws for an extension with no declared normalisation: a newly fingerprinted source must state
 * what is and is not part of its contract rather than silently defaulting to raw-text hashing.
 */
export function semanticSourceBody(relativePath, body) {
  const extension = path.extname(relativePath);
  const canonical = canonicalSourceText(body);
  if (JSON_EXTENSIONS.includes(extension)) {
    const parsed = JSON.parse(extension === ".jsonc" ? stripJsoncComments(canonical) : canonical);
    return JSON.stringify(parsed);
  }
  const marker = LINE_COMMENT_MARKERS[extension];
  if (!marker) {
    throw new Error(
      `${relativePath}: no declared provenance normalisation for "${extension}" sources. Add one to ` +
        "scripts/lib/source-revision.mjs — hashing raw text would churn the fingerprint on inert edits.",
    );
  }
  return stripInertLines(canonical, marker);
}
