// The manifest stamper, over synthetic JSONC bodies (sc-17989).
//
// The stamper is line-based on purpose — the manifest is JSONC and the comments beside individual
// bindings carry their reasoning, so a parse/serialise round trip would delete them. That makes its
// "is a digest already here" lookup a text scan rather than a property read, and the exact reach of
// that scan is the whole correctness question: too narrow and it inserts a SECOND
// `inferenceClosureDigest` beside the one already there, corrupting shipped runtime data; too wide
// and it claims a sibling object's digest.

import assert from "node:assert/strict";
import test from "node:test";

import { stampManifest, verifyOutcome } from "./backfill-closure-digests.mjs";

const REVISION = "a4f409ae8ce73eda2ee8117b89b5f479666606b8";
const OTHER_REVISION = "1899a7228deb99b65535745d09d4a5f5524565c4";
const WANTED = "b".repeat(64);
const STALE = "c".repeat(64);

/** Every derivation returns `WANTED`, so any surviving `STALE` is a digest the stamper did not fix. */
const stamp = (body) => stampManifest(body, () => WANTED);

const digestKeyCount = (body) => (body.match(/"inferenceClosureDigest"/g) ?? []).length;

test("a digest separated from its revision by comment lines is UPDATED, never duplicated", () => {
  // `turboFit`'s real shape: the digest sits three lines below `inferenceRevision`, behind two
  // comment lines. A two-line lookup window reports "no digest here" and appends a second key.
  const body = [
    '      "candle": {',
    '        "turboFit": {',
    `          "inferenceRevision": "${REVISION}",`,
    "          // sc-17774: the currency term. `inferenceRevision` above is capture provenance and",
    "          // is no longer compared — the digest of the compile closure is.",
    `          "inferenceClosureDigest": "${STALE}",`,
    '          "provider": "krea_2_turbo",',
    '          "measured": true',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(digestKeyCount(result.body), 1, "a second inferenceClosureDigest was inserted");
  assert.match(result.body, new RegExp(`"inferenceClosureDigest": "${WANTED}"`));
  assert.doesNotMatch(result.body, new RegExp(STALE));
  assert.deepEqual(result.stamped, [`candle:krea_2_turbo@${REVISION.slice(0, 8)}`]);
  assert.deepEqual(result.skipped, []);
  // The comments the line-based approach exists to protect are still there.
  assert.match(result.body, /sc-17774: the currency term/);
});

test("a digest already correct behind comment lines is left completely alone", () => {
  const body = [
    '      "candle": {',
    '        "turboFit": {',
    `          "inferenceRevision": "${REVISION}",`,
    "          // provenance note",
    `          "inferenceClosureDigest": "${WANTED}",`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(result.body, body);
  assert.deepEqual(result.stamped, []);
});

test("a drifted digest on the line RIGHT AFTER the revision is rewritten and counted", () => {
  // The pre-sc-17989 lookup DID see this line, but wrote the replacement onto the revision line
  // (where the stale digest does not appear, so nothing changed) and then deliberately did not
  // count it — so `--verify` saw zero drift and passed a digest that disagreed with its source.
  const body = [
    '      "candle": {',
    '        "block": {',
    `          "inferenceRevision": "${REVISION}",`,
    `          "inferenceClosureDigest": "${STALE}",`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(digestKeyCount(result.body), 1);
  assert.doesNotMatch(result.body, new RegExp(STALE));
  assert.deepEqual(result.stamped, [`candle:krea_2_turbo@${REVISION.slice(0, 8)}`]);
});

test("a drifted digest packed onto the revision line itself is rewritten in place", () => {
  const body = [
    '      "candle": {',
    '        "block": {',
    `          "inferenceRevision": "${REVISION}", "inferenceClosureDigest": "${STALE}",`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(digestKeyCount(result.body), 1);
  assert.doesNotMatch(result.body, new RegExp(STALE));
  assert.deepEqual(result.stamped, [`candle:krea_2_turbo@${REVISION.slice(0, 8)}`]);
});

test("a 64-hex value quoted inside a COMMENT is never mistaken for the digest key", () => {
  // The manifest narrates digest provenance in prose. A scan that reads comment tails rewrites the
  // prose, leaves the block with no real `inferenceClosureDigest`, and reports success.
  const body = [
    '      "candle": {',
    '        "block": {',
    `          "inferenceRevision": "${REVISION}",`,
    `          // superseded: this used to read "inferenceClosureDigest": "${STALE}" before sc-17935`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.match(result.body, new RegExp(STALE), "the prose comment was rewritten");
  const lines = result.body.split("\n");
  assert.match(lines[2], new RegExp(`"inferenceClosureDigest": "${WANTED}"`), "no real key was added");
  assert.deepEqual(result.orphans, [], "the comment must not read as an ungraded digest either");
});

test("a `//` inside a string value does not truncate the line at that column", () => {
  // `"https://…"` and `"q4/*"` are everywhere in this manifest; treating either as a comment start
  // would hide the keys that follow it on the same line.
  const body = [
    '      "candle": {',
    '        "block": {',
    '          "licenseUrl": "https://www.krea.ai/krea-2-licensing",',
    `          "inferenceRevision": "${REVISION}", "inferenceClosureDigest": "${STALE}",`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.doesNotMatch(result.body, new RegExp(STALE));
  assert.deepEqual(result.stamped, [`candle:krea_2_turbo@${REVISION.slice(0, 8)}`]);
  assert.deepEqual(result.orphans, []);
});

test("a NESTED sub-object between the revision and the digest does not end the object", () => {
  // The real `turboFit` shape one tidy-up away: its `verification` block sits above its digest.
  // Breaking at the first line starting with `}` stops at the NESTED close and inserts a second key.
  const body = [
    '      "candle": {',
    '        "turboFit": {',
    `          "inferenceRevision": "${REVISION}",`,
    '          "verification": {',
    '            "hardware": "RTX PRO 6000 Blackwell (sm_120)"',
    "          },",
    `          "inferenceClosureDigest": "${STALE}",`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(digestKeyCount(result.body), 1, "a second inferenceClosureDigest was inserted");
  assert.doesNotMatch(result.body, new RegExp(STALE));
  assert.deepEqual(result.stamped, [`candle:krea_2_turbo@${REVISION.slice(0, 8)}`]);
});

test("a digest belonging to a NESTED sub-object is not claimed as the binding's own", () => {
  const body = [
    '      "candle": {',
    '        "block": {',
    `          "inferenceRevision": "${REVISION}",`,
    '          "provider": "krea_2_turbo",',
    '          "nested": {',
    `            "inferenceClosureDigest": "${STALE}"`,
    "          }",
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  const lines = result.body.split("\n");
  assert.match(lines[5], new RegExp(STALE), "the nested object's digest was rewritten");
  assert.match(lines[2], new RegExp(`"inferenceClosureDigest": "${WANTED}"`), "the binding was not stamped");
  assert.deepEqual(result.orphans, [
    "line 6: an inferenceClosureDigest no inferenceRevision/provider pair reaches",
  ]);
});

test("a digest ABOVE its revision in the same object is updated, not duplicated", () => {
  const body = [
    '      "candle": {',
    '        "block": {',
    `          "inferenceClosureDigest": "${STALE}",`,
    "          // the revision this was taken at",
    `          "inferenceRevision": "${REVISION}",`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(digestKeyCount(result.body), 1);
  assert.doesNotMatch(result.body, new RegExp(STALE));
  assert.deepEqual(result.stamped, [`candle:krea_2_turbo@${REVISION.slice(0, 8)}`]);
  assert.deepEqual(result.orphans, []);
});

test("a provider ABOVE its revision in the same object is still found", () => {
  const body = [
    '      "candle": {',
    '        "block": {',
    '          "provider": "krea_2_turbo",',
    `          "inferenceRevision": "${REVISION}"`,
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.deepEqual(result.skipped, []);
  assert.deepEqual(result.stamped, [`candle:krea_2_turbo@${REVISION.slice(0, 8)}`]);
});

test("a revision with NO trailing comma gets a comma-prefixed digest, keeping the JSON valid", () => {
  // The locator used to require the comma, so a revision written last in its object was invisible
  // and its digest silently left the gate.
  const body = [
    '      "candle": {',
    '        "block": {',
    '          "provider": "krea_2_turbo",',
    `          "inferenceRevision": "${REVISION}"`,
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.match(
    result.body,
    new RegExp(`"inferenceRevision": "${REVISION}", "inferenceClosureDigest": "${WANTED}"$`, "m"),
  );
  // The fixture is an object BODY, so wrapping it is enough to prove the insertion kept it parseable
  // — a missing separator here would ship a manifest the runtime cannot read at all.
  const parsed = JSON.parse(`{${result.body}}`);
  assert.equal(parsed.candle.block.inferenceClosureDigest, WANTED);
  assert.equal(parsed.candle.block.inferenceRevision, REVISION);
});

test("sibling objects written one per line do not bleed into each other", () => {
  const body = [
    '      "candle": {',
    '        "records": [',
    `          { "inferenceRevision": "${REVISION}", "provider": "krea_2_turbo" },`,
    `          { "inferenceClosureDigest": "${STALE}", "provider": "krea_2_turbo_control" }`,
    "        ]",
    "      }",
  ].join("\n");

  const result = stamp(body);

  const lines = result.body.split("\n");
  assert.match(lines[2], new RegExp(WANTED), "the first object was not stamped");
  assert.match(lines[3], new RegExp(STALE), "the sibling's digest was rewritten");
});

test("an inferenceClosureDigest no binding reaches is reported as an ORPHAN", () => {
  // Losing the `inferenceRevision` — or merely its trailing comma, before this — leaves a digest
  // that still looks graded and is checked by nothing.
  const body = [
    '      "candle": {',
    '        "block": {',
    `          "inferenceClosureDigest": "${STALE}",`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(result.body, body, "an unreachable digest must not be rewritten");
  assert.deepEqual(result.stamped, []);
  assert.deepEqual(result.orphans, [
    "line 3: an inferenceClosureDigest no inferenceRevision/provider pair reaches",
  ]);
});

test("the lookup stops at the object close and never claims a SIBLING's digest", () => {
  // The first object has no digest of its own; the second does. Reaching across the `},` would
  // rewrite the sibling's digest and leave the first object unstamped.
  const body = [
    '      "candle": {',
    '        "first": {',
    `          "inferenceRevision": "${REVISION}",`,
    '          "provider": "krea_2_turbo"',
    "        },",
    '        "second": {',
    `          "inferenceClosureDigest": "${STALE}",`,
    '          "provider": "krea_2_turbo_control"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  const lines = result.body.split("\n");
  assert.match(lines[2], new RegExp(`"inferenceClosureDigest": "${WANTED}"`), "first was not stamped");
  assert.match(lines[6], new RegExp(STALE), "the sibling's digest was rewritten");
  assert.equal(digestKeyCount(result.body), 2);
  assert.deepEqual(result.stamped, [`candle:krea_2_turbo@${REVISION.slice(0, 8)}`]);
});

test("a revision with no provider in its object is reported, not silently skipped", () => {
  const body = [
    '      "candle": {',
    '        "block": {',
    `          "inferenceRevision": "${REVISION}",`,
    `          "inferenceClosureDigest": "${STALE}",`,
    '          "measured": true',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(result.body, body, "an unlocatable block must not be rewritten");
  assert.deepEqual(result.stamped, []);
  assert.equal(result.skipped.length, 1);
  assert.match(result.skipped[0], /a4f409ae has no provider/);
});

test("a provenance-only inferenceRevision is outside the closure-binding population", () => {
  const body = [
    '      "vector": {',
    '        "terminalCandidate": {',
    `          "inferenceRevision": "${REVISION}",`,
    '          "corpusSha256": "f"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(result.body, body);
  assert.deepEqual(result.stamped, []);
  assert.deepEqual(result.skipped, []);
  assert.deepEqual(result.orphans, []);
  assert.deepEqual(result.located, []);
});

test("a revision outside any mlx/candle block is reported rather than stamped under a guess", () => {
  const body = [
    '      "somethingElse": {',
    `        "inferenceRevision": "${REVISION}",`,
    '        "provider": "krea_2_turbo"',
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(result.body, body);
  assert.equal(result.skipped.length, 1);
  assert.match(result.skipped[0], /is in no mlx\/candle block/);
});

test("the backend and provider are paired per block, so two lanes get their own digests", () => {
  const body = [
    '      "candle": {',
    '        "turboFit": {',
    `          "inferenceRevision": "${REVISION}",`,
    "          // comment between the two keys",
    `          "inferenceClosureDigest": "${STALE}",`,
    '          "provider": "krea_2_turbo"',
    "        },",
    '        "control": {',
    `          "inferenceRevision": "${OTHER_REVISION}",`,
    `          "inferenceClosureDigest": "${STALE}",`,
    '          "provider": "krea_2_turbo_control"',
    "        }",
    "      }",
  ].join("\n");

  const seen = [];
  const result = stampManifest(body, (key, revision) => {
    seen.push(`${key}@${revision.slice(0, 8)}`);
    return WANTED;
  });

  assert.deepEqual(seen, [
    `candle:krea_2_turbo@${REVISION.slice(0, 8)}`,
    `candle:krea_2_turbo_control@${OTHER_REVISION.slice(0, 8)}`,
  ]);
  assert.equal(digestKeyCount(result.body), 2);
  assert.doesNotMatch(result.body, new RegExp(STALE));
});

test("stamping is idempotent — a second pass over its own output changes nothing", () => {
  const body = [
    '      "candle": {',
    '        "turboFit": {',
    `          "inferenceRevision": "${REVISION}",`,
    "          // comment",
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const first = stamp(body);
  assert.equal(digestKeyCount(first.body), 1);
  assert.deepEqual(first.stamped, [`candle:krea_2_turbo@${REVISION.slice(0, 8)}`]);

  const second = stamp(first.body);
  assert.equal(second.body, first.body);
  assert.deepEqual(second.stamped, []);
});

// --- the `--verify` verdict ------------------------------------------------------------------
//
// `main` needs a real inference clone, so the gate's DECISION is split out and exercised here. Every
// case below is a way the gate can be wrong, and each one used to be reachable: sc-17935 shipped a
// dry run that reported drift and exited 0, and before sc-17989 an unlocatable digest was a printed
// note on the same passing path.

const clean = { stamped: [], skipped: [], orphans: [] };

test("--verify passes only when nothing drifted and nothing is unreachable", () => {
  assert.deepEqual(verifyOutcome({ recordDrift: 0, manifest: clean }), { code: 0, errors: [] });
});

test("--verify fails on a drifted record digest", () => {
  const outcome = verifyOutcome({ recordDrift: 3, manifest: clean });
  assert.equal(outcome.code, 1);
  assert.match(outcome.errors.join("\n"), /3 record digest\(s\)/);
});

test("--verify fails on a drifted manifest binding", () => {
  const outcome = verifyOutcome({
    recordDrift: 0,
    manifest: { ...clean, stamped: ["candle:krea_2_turbo@a4f409ae"] },
  });
  assert.equal(outcome.code, 1);
  assert.match(outcome.errors.join("\n"), /1 manifest binding\(s\)/);
});

test("--verify fails on a block the stamper cannot pair, even with zero drift", () => {
  const outcome = verifyOutcome({
    recordDrift: 0,
    manifest: { ...clean, skipped: ["line 5342: a4f409ae has no provider in its object"] },
  });
  assert.equal(outcome.code, 1, "an unlocatable digest passed the gate");
  assert.match(outcome.errors.join("\n"), /outside this gate/);
  assert.match(outcome.errors.join("\n"), /line 5342/);
});

test("--verify fails on an orphaned digest, even with zero drift", () => {
  const outcome = verifyOutcome({
    recordDrift: 0,
    manifest: { ...clean, orphans: ["line 99: an inferenceClosureDigest no pair reaches"] },
  });
  assert.equal(outcome.code, 1, "an ungraded digest passed the gate");
  assert.match(outcome.errors.join("\n"), /line 99/);
});

test("a coverage gap is reported ahead of drift, not hidden behind it", () => {
  // Both are true at once. Reporting only the drift sends you to fix the digest, get a green, and
  // never learn that another digest is not being graded at all.
  const outcome = verifyOutcome({
    recordDrift: 2,
    manifest: { ...clean, stamped: ["candle:krea_2_turbo@a4f409ae"], orphans: ["line 99: orphan"] },
  });
  assert.equal(outcome.code, 1);
  assert.match(outcome.errors.join("\n"), /outside this gate/);
  assert.doesNotMatch(outcome.errors.join("\n"), /record digest\(s\)/);
});

// --- shapes the manifest does not use TODAY -----------------------------------------------------
//
// The five flux1 bindings already pack a revision and its digest onto one line, so compaction is an
// established style in this file rather than a hypothetical. Each case below was silently wrong
// before: the wrong answer was a corrupted manifest or a green gate over an ungraded digest, never
// an error, which is the whole failure mode sc-17989 exists to remove.

test("a SECOND binding packed onto the same line is stamped, not silently skipped", () => {
  const body = [
    '      "candle": {',
    '        "records": [',
    `          { "inferenceRevision": "${REVISION}", "inferenceClosureDigest": "${STALE}", "provider": "flux1_dev" }, ` +
      `{ "inferenceRevision": "${OTHER_REVISION}", "inferenceClosureDigest": "${STALE}", "provider": "flux1_schnell" }`,
    "        ]",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.deepEqual(result.stamped, [
    `candle:flux1_dev@${REVISION.slice(0, 8)}`,
    `candle:flux1_schnell@${OTHER_REVISION.slice(0, 8)}`,
  ]);
  assert.doesNotMatch(result.body, new RegExp(STALE), "the second binding's digest was left stale");
  assert.deepEqual(result.orphans, []);
});

test("an unclaimed digest packed beside a claimed one on the same line is an orphan", () => {
  // Line-granular claim tracking marks BOTH digests reached the moment one binding claims either,
  // so the ungraded one passes the gate looking graded.
  const body = [
    '      "candle": {',
    '        "records": [',
    `          { "inferenceRevision": "${REVISION}", "inferenceClosureDigest": "${STALE}", "provider": "flux1_dev" }, ` +
      `{ "inferenceClosureDigest": "${STALE}", "provider": "flux1_schnell" }`,
    "        ]",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.deepEqual(result.stamped, [`candle:flux1_dev@${REVISION.slice(0, 8)}`]);
  assert.deepEqual(result.orphans, [
    "line 3: an inferenceClosureDigest no inferenceRevision/provider pair reaches",
  ]);
});

test("an unbalanced brace inside a COMMENT does not end the object", () => {
  // The only thing `maskLine` exists for. With raw-line brace counting the `}` in this prose closes
  // the object early, the digest below it is missed, and a second key is inserted.
  const body = [
    '      "candle": {',
    '        "block": {',
    `          "inferenceRevision": "${REVISION}",`,
    '          // shape: { "q4": "q8" } — the branch follows the tier, except }',
    `          "inferenceClosureDigest": "${STALE}",`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(digestKeyCount(result.body), 1, "a second inferenceClosureDigest was inserted");
  assert.doesNotMatch(result.body, new RegExp(STALE));
  assert.deepEqual(result.stamped, [`candle:krea_2_turbo@${REVISION.slice(0, 8)}`]);
});

test("an unbalanced brace inside a STRING VALUE does not open an object", () => {
  const body = [
    '      "candle": {',
    '        "block": {',
    '          "numericPolicy": "selector boundary exact { per rung",',
    `          "inferenceRevision": "${REVISION}",`,
    `          "inferenceClosureDigest": "${STALE}",`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(digestKeyCount(result.body), 1);
  assert.doesNotMatch(result.body, new RegExp(STALE));
  assert.deepEqual(result.skipped, [], "the provider was scoped out by a brace inside a string");
});

test("a backend block that has already CLOSED is not claimed by a later binding", () => {
  // A line scan upward finds the nearest `"mlx": {` header whether or not that block is still open.
  // The module header records this exact bug inventing a `candle:qwen_image` lane that never existed.
  const body = [
    '      "backends": {',
    '        "mlx": {',
    '          "someLane": { "measured": true }',
    "        },",
    '        "sharedCalibration": {',
    `          "inferenceRevision": "${REVISION}",`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.deepEqual(result.stamped, [], "a binding was stamped under a closed backend block");
  assert.equal(result.skipped.length, 1);
  assert.match(result.skipped[0], /is in no mlx\/candle block/);
});

test("a binding nested several levels inside its backend block still resolves the backend", () => {
  const body = [
    '      "candle": {',
    '        "turboFit": {',
    '          "evidenceRecords": [',
    "            {",
    `              "inferenceRevision": "${REVISION}",`,
    '              "provider": "krea_2_turbo"',
    "            }",
    "          ]",
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.deepEqual(result.stamped, [`candle:krea_2_turbo@${REVISION.slice(0, 8)}`]);
  assert.deepEqual(result.skipped, []);
});

test("a /* block comment is refused outright rather than scoped wrongly", () => {
  // Line-scoped scanning cannot see a comment that spans lines: its braces would move object
  // boundaries and a hex value inside it would be rewritten as a digest. JSONC allows them and
  // scripts/lib/jsonc.mjs reads them, so "the manifest has none" is a fact about today only.
  const body = [
    '      "candle": {',
    "        /* superseded block",
    `           "inferenceClosureDigest": "${STALE}" */`,
    `        "inferenceRevision": "${REVISION}",`,
    '        "provider": "krea_2_turbo"',
    "      }",
  ].join("\n");

  assert.throws(() => stamp(body), /block comment/);
});

test("a `/*` inside a string value is not mistaken for a block comment", () => {
  const body = [
    '      "candle": {',
    '        "block": {',
    '          "files": ["q4/*", "q8/*"],',
    `          "inferenceRevision": "${REVISION}",`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  assert.doesNotThrow(() => stamp(body));
  assert.deepEqual(stamp(body).stamped, [`candle:krea_2_turbo@${REVISION.slice(0, 8)}`]);
});

test("two inferenceRevisions sharing one object's digest are reported, not silently thrashed", () => {
  const body = [
    '      "candle": {',
    '        "block": {',
    `          "inferenceRevision": "${REVISION}",`,
    `          "inferenceRevision": "${OTHER_REVISION}",`,
    `          "inferenceClosureDigest": "${STALE}",`,
    '          "provider": "krea_2_turbo"',
    "        }",
    "      }",
  ].join("\n");

  const result = stamp(body);

  assert.equal(result.skipped.length, 1);
  assert.match(result.skipped[0], /shares its digest with another inferenceRevision/);
});
