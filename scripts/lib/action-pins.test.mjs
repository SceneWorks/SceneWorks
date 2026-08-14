import assert from "node:assert/strict";
import test from "node:test";

import { findActionPinViolations } from "./action-pins.mjs";

const SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1";

test("a SHA-pinned action carrying its version comment is accepted", () => {
  const workflow = [
    "jobs:",
    "  build:",
    "    steps:",
    `      - uses: actions/checkout@${SHA} # v7.0.1`,
    `        uses: docker/metadata-action@${SHA} # v6.2.0`,
    `      - uses: dtolnay/rust-toolchain@${SHA} # stable branch`,
    `      - uses: Swatinem/rust-cache@${SHA} # v2 (v2.9.1+1)`,
    "      - uses: ./.github/actions/local-thing",
  ].join("\n");

  assert.deepEqual(findActionPinViolations(workflow), []);
});

test("a tag-pinned action is rejected", () => {
  const violations = findActionPinViolations("      - uses: actions/checkout@v7", "wf.yml");
  assert.equal(violations.length, 1);
  assert.equal(violations[0].line, 1);
  assert.match(violations[0].reason, /mutable ref 'v7'/);
});

test("a branch-pinned action is rejected", () => {
  const violations = findActionPinViolations("      - uses: some/action@main");
  assert.equal(violations.length, 1);
  assert.match(violations[0].reason, /mutable ref 'main'/);
});

test("an abbreviated SHA is rejected", () => {
  const violations = findActionPinViolations(`      - uses: actions/checkout@${SHA.slice(0, 12)}`);
  assert.equal(violations.length, 1);
  assert.match(violations[0].reason, /40-character commit SHA/);
});

test("a full SHA without the version comment is rejected", () => {
  const violations = findActionPinViolations(`      - uses: actions/checkout@${SHA}`);
  assert.equal(violations.length, 1);
  assert.match(violations[0].reason, /missing the trailing/);
});

test("an empty version comment is rejected", () => {
  const violations = findActionPinViolations(`      - uses: actions/checkout@${SHA} #`);
  assert.equal(violations.length, 1);
  assert.match(violations[0].reason, /missing the trailing/);
});

test("an unversioned reference is rejected", () => {
  const violations = findActionPinViolations("      - uses: docker://alpine");
  assert.equal(violations.length, 1);
  assert.match(violations[0].reason, /not pinned to a commit SHA/);
});

test("violations report the file and line they came from", () => {
  const workflow = [
    "jobs:",
    `      - uses: actions/checkout@${SHA} # v7.0.1`,
    "      - uses: evil/action@v1",
  ].join("\n");

  const violations = findActionPinViolations(workflow, ".github/workflows/x.yml");
  assert.equal(violations.length, 1);
  assert.equal(violations[0].path, ".github/workflows/x.yml");
  assert.equal(violations[0].line, 3);
  assert.equal(violations[0].reference, "evil/action@v1");
});

test("a commented-out uses line is not treated as a step", () => {
  assert.deepEqual(findActionPinViolations("      # uses: actions/checkout@v7"), []);
});
