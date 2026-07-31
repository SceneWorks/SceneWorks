import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalSourceText,
  isInertLine,
  semanticSourceBody,
  stripInertLines,
} from "./source-revision.mjs";

test("line endings are normalised before anything else looks at the body", () => {
  const canonical = "alpha\nbeta\ngamma\n";
  assert.equal(canonicalSourceText(canonical), canonical);
  assert.equal(canonicalSourceText("alpha\r\nbeta\r\ngamma\r\n"), canonical);
  assert.equal(canonicalSourceText("alpha\rbeta\rgamma\r"), canonical);
});

test("lines are inert only when they cannot carry a parsed literal", () => {
  assert.equal(isInertLine("// a plain prose comment", "//"), true);
  assert.equal(isInertLine("        /// an indented doc comment", "//"), true);
  assert.equal(isInertLine("//! a module doc comment", "//"), true);
  assert.equal(isInertLine("", "//"), true);
  assert.equal(isInertLine("   \t ", "//"), true);
  // The quote guard: a commented-out line carrying a string literal is exactly what a parser
  // could otherwise pick up, so it stays part of the fingerprint.
  assert.equal(isInertLine('    // "z_image_turbo",', "//"), false);
  assert.equal(isInertLine('// `"control"` appears once in the manifest', "//"), false);
  // Never inline: a `//` inside a string literal is not a comment.
  assert.equal(isInertLine('let repo = "https://example.invalid/x"; // pin', "//"), false);
  assert.equal(isInertLine("const HEADROOM_GB: f64 = 1.5; // headroom", "//"), false);
  // TOML uses a different marker, and a trailing `#` is not a whole-line comment.
  assert.equal(isInertLine("# a cargo comment", "#"), true);
  assert.equal(isInertLine('rev = "abc123" # the pin', "#"), false);
});

test("stripping removes only inert lines and is idempotent", () => {
  const body = [
    "// leading prose",
    "",
    "const EXPECTED: &[&str] = &[",
    '    // "commented_out_id",',
    '    "real_id",',
    "];",
    "    /// indented doc",
    "let url = \"https://example.invalid\"; // trailing",
  ].join("\n");
  const stripped = stripInertLines(body, "//");
  assert.equal(
    stripped,
    [
      "const EXPECTED: &[&str] = &[",
      '    // "commented_out_id",',
      '    "real_id",',
      "];",
      "let url = \"https://example.invalid\"; // trailing",
    ].join("\n"),
  );
  assert.equal(stripInertLines(stripped, "//"), stripped);
});

test("block comments are deliberately left in the fingerprint", () => {
  // Stripping them safely needs a tokenizer (a `/*` can live inside a string literal), and they are
  // a small minority of the hashed text. Noisy-but-safe is the correct side to err on.
  const body = ["/**", " * a jsdoc block", " */", "const x = 1;"].join("\n");
  assert.equal(stripInertLines(body, "//"), body);
});

test("semanticSourceBody normalises per language and fails closed on an unknown one", () => {
  // The trailing newline goes with the other blank lines — inert is inert.
  assert.equal(semanticSourceBody("crates/x/src/lib.rs", "// doc\nlet a = 1;\n"), "let a = 1;");
  assert.equal(semanticSourceBody("Cargo.toml", '# doc\nrev = "a"\n'), 'rev = "a"');
  assert.equal(
    semanticSourceBody("scripts/x.mjs", "// doc\n\nexport const a = 1;\n"),
    "export const a = 1;",
  );
  // JSON and JSONC are hashed as their parsed value, so formatting and comments are inert.
  assert.equal(
    semanticSourceBody("config/plan.json", '{\n  "b": 1,\n  "a": [2]\n}\n'),
    semanticSourceBody("config/plan.json", '{"b":1,"a":[2]}'),
  );
  assert.equal(
    semanticSourceBody("config/m.jsonc", '{\n  // a comment\n  "a": 1\n}\n'),
    '{"a":1}',
  );
  assert.throws(
    () => semanticSourceBody("docs/notes.md", "# heading\n"),
    /no declared provenance normalisation/,
    "a newly fingerprinted source must declare its contract instead of defaulting to raw text",
  );
});

test("a semantic change is still visible after normalisation", () => {
  // The mutation check for the stripper itself: it must not be a value-blind no-op.
  const before = semanticSourceBody("crates/x/src/lib.rs", "// doc\nconst HEADROOM_GB: f64 = 1.5;\n");
  const after = semanticSourceBody("crates/x/src/lib.rs", "const HEADROOM_GB: f64 = 2.5;\n");
  assert.notEqual(before, after);
});
