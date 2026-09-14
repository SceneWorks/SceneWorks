import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import test from "node:test";

import { buildMatrix, SOURCE_PATHS } from "./generate-memory-matrix.mjs";

/**
 * sc-22513 (epic 22505, E5) — the removed-source probe, in its own file ON PURPOSE.
 *
 * It edits shared repo artifacts on disk and regenerates the matrix between the edit and the
 * restore. Sibling suites in `npm run check` hash those same bytes — `extract-memory-anchors`
 * embeds a sha256 of the evidence bundle, `compare-engine-capability-facts` reads the MLX dump —
 * and `node --test` runs test FILES in parallel, so probing from inside the main generator suite
 * would make those siblings flake at random. `npm run check` therefore runs this file in a second,
 * SERIAL `node --test` segment, after the parallel pool has drained.
 */

// Every path the collapse removed from `SOURCE_PATHS`. Six of the seven were load-bearing before
// this change: the plan decided which coordinates were published, the evidence bundle and the
// closure ledger decided which records promoted them, the rung-4 survey and its prerequisite graph
// decided the whole `bounded_transformer_residency` implementation axis, and the MLX capability dump
// fed the contract projection. Probing only one would leave the other six free to keep moving cells
// while the test read green.
const REMOVED_SOURCES = [
  "config/memory-calibration-plan.json",
  "docs/generated/memory-calibration-evidence.json",
  "config/inference-provider-closures.json",
  "config/rung4-applicability-survey.json",
  "config/rung4-contract-prerequisites.json",
  "config/engine-capabilities/capabilities.mlx.json",
  "docs/generated/video-memory-curves.json",
];

test("no removed source is in the fingerprint (the list this file probes is the right one)", () => {
  const kept = new Set(Object.values(SOURCE_PATHS));
  for (const removed of REMOVED_SOURCES) {
    assert.ok(!kept.has(removed), `${removed} is back in SOURCE_PATHS — probe it as a source instead`);
  }
});

test("a change to a REMOVED source produces no matrix diff at all (sc-22513)", async () => {
  // Mutated ON DISK rather than through `sourceOverrides`, because an override keyed by a name the
  // map no longer has would be silently ignored and the test would pass without touching anything.
  const baseline = JSON.stringify(await buildMatrix());
  let probed = 0;
  for (const relative of REMOVED_SOURCES) {
    const url = new URL(`../${relative}`, import.meta.url);
    let original;
    try {
      original = await readFile(url, "utf8");
    } catch (error) {
      // A retired artifact that no longer exists on disk cannot move a cell either; skip it rather
      // than failing, so this list can outlive the files.
      if (error.code === "ENOENT") continue;
      throw error;
    }
    const parsed = JSON.parse(original);
    // A structural mutation, not just an added key: reverse the first top-level array the file
    // carries, so a generator that still parsed this file would see reordered rows and not merely an
    // unknown property it could ignore.
    const arrayKey = Object.keys(parsed).find((key) => Array.isArray(parsed[key]));
    const mutated = JSON.stringify(
      {
        ...parsed,
        ...(arrayKey ? { [arrayKey]: [...parsed[arrayKey]].reverse() } : {}),
        sc22513Probe: "this file is no longer a matrix input",
      },
      null,
      2,
    );
    try {
      await writeFile(url, `${mutated}\n`);
      assert.notEqual(
        await readFile(url, "utf8"),
        original,
        `${relative}: the probe must really have edited the file`,
      );
      assert.equal(JSON.stringify(await buildMatrix()), baseline, relative);
    } finally {
      // Restore before anything else can observe the edit, on the failure path too.
      await writeFile(url, original);
    }
    assert.equal(await readFile(url, "utf8"), original, `${relative} was restored`);
    probed += 1;
  }
  assert.equal(probed, REMOVED_SOURCES.length, "every removed source that exists on disk was probed");
});
