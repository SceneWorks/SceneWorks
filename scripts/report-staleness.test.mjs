// sc-19751: the staleness report must stay a report.
//
// The single property that must never regress is the exit code: this exists because gates on this
// information fired on unrelated changes, so a future edit that makes it fail on a finding would
// reintroduce exactly what sc-19728 and sc-19751 removed. Every case below feeds it findings and
// asserts 0.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SCRIPT = path.join(ROOT, "scripts/report-staleness.mjs");

const STAMP_A = "a".repeat(64);
const STAMP_B = "b".repeat(64);

function catalog(models) {
  const file = path.join(mkdtempSync(path.join(os.tmpdir(), "staleness-")), "models.jsonc");
  writeFileSync(file, `// a jsonc catalog\n${JSON.stringify({ models }, null, 2)}\n`);
  return file;
}

function row(stamp, disposition = { kind: "admitted" }) {
  return { implementationFingerprint: stamp, disposition };
}

function model(id, { rows = [], tiles = false } = {}) {
  return {
    id,
    mlx: {
      memoryStrategyContract: {
        implementations: [
          {
            strategy: tiles ? "BoundedDecode" : "Resident",
            parameterRanges: rows.length > 0 ? { decodeGeometryPolicies: rows } : {},
          },
        ],
      },
    },
  };
}

function report(args) {
  const result = spawnSync("node", [SCRIPT, ...args], { encoding: "utf8" });
  if (result.error) throw result.error;
  return { status: result.status, output: `${result.stdout ?? ""}${result.stderr ?? ""}` };
}

test("findings never change the exit code", () => {
  const file = catalog([
    model("refused_everywhere", { rows: [row(STAMP_B, { kind: "refused", reason: "over tolerance" })] }),
    model("unmeasured", { tiles: true }),
  ]);
  const { status, output } = report(["--catalog", file, "--fingerprint", STAMP_A]);
  assert.equal(status, 0, `a report must never gate:\n${output}`);
  assert.match(output, /refused on quality/);
  assert.match(output, /tile the VAE decode but carry no measurement/);
});

test("a stamp from another source closure is labelled, not condemned", () => {
  const file = catalog([model("m", { rows: [row(STAMP_B)] })]);
  const { status, output } = report(["--catalog", file, "--fingerprint", STAMP_A]);
  assert.equal(status, 0);
  assert.match(output, /measured by different source/);
  assert.match(output, /nothing refuses it/, "the report must say a differing stamp is not a defect");

  const matching = report(["--catalog", catalog([model("m", { rows: [row(STAMP_A)] })]), "--fingerprint", STAMP_A]);
  assert.match(matching.output, /\[CURRENT\]/);
});

test("without a fingerprint it declines to guess", () => {
  const file = catalog([model("m", { rows: [row(STAMP_B)] })]);
  const { status, output } = report(["--catalog", file]);
  assert.equal(status, 0);
  assert.match(output, /not compared/);
  assert.doesNotMatch(output, /CURRENT/);
});

test("a refused row is reported as a measurement, not staleness", () => {
  const file = catalog([
    model("m", {
      rows: [row(STAMP_A, { kind: "refused", reason: "max_abs_rgb_u8 exceeded 48: seed 7=76" })],
    }),
  ]);
  const { output } = report(["--catalog", file, "--fingerprint", STAMP_A]);
  assert.match(output, /admitted  0/);
  assert.match(output, /no coordinate admitted/);
  assert.match(output, /Remeasuring will not change it/);
  assert.match(output, /seed 7=76/, "the measured reason must survive into the report");
});

test("unmeasured routes collapse per model rather than per tier", () => {
  const tiered = {
    id: "three_tiers",
    mlx: {
      memoryStrategyContract: {
        implementations: [1, 2, 3].map(() => ({
          strategy: "BoundedDecode",
          parameterRanges: {},
        })),
      },
    },
  };
  const { output } = report(["--catalog", catalog([tiered])]);
  assert.match(output, /1 route tile/);
  assert.match(output, /three_tiers \(mlx\)\s+\(3 implementations\)/);
});

test("a malformed --fingerprint is an invocation error, not a finding", () => {
  const { status } = report(["--catalog", catalog([]), "--fingerprint", "nope"]);
  assert.equal(status, 2, "exit 2 distinguishes 'you typed it wrong' from 'here are findings'");
});
