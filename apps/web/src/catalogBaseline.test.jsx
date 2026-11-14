import { renderBaseline } from "../../../scripts/rearchitecture/baseline-json.mjs";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { currentCandidate } from "./testUtils/catalogBaseline/candidate.js";
import { beforeAll, describe, expect, it, vi } from "vitest";
vi.mock("./api.js", async (original) => ({ ...await original(), apiFetch: vi.fn(async () => ({})) }));
import old from "./testUtils/catalogBaseline/old-catalog.json";
import { population, unionPopulation, requireOldPopulation, keyFor, observe } from "./testUtils/catalogBaseline/harness.jsx";
const observationsPath = "src/testUtils/catalogBaseline/old-observations.json";
const candidate = currentCandidate();
const candidateRows = population(candidate);
const baseline = existsSync(observationsPath) ? JSON.parse(readFileSync(observationsPath, "utf8")) : {};
const capture = process.env.CAPTURE_OLD_CATALOG === old.source.revision;
const oldRows = capture ? population(old.catalog) : baseline.population;
const captured = {};
function unpack(row) { return { ...row, controls: row.controls.map((id) => baseline.controls[id]),
  submission: { ...row.submission, calls: baseline.calls[row.submission.calls] },
  changes: row.changes.map((change) => ({ ...change, calls: baseline.calls[change.calls], persisted: change.persisted == null ? null : baseline.states[change.persisted] })),
  persisted: baseline.states[row.persisted],
}; }
function pack(rows) {
  const controls = [], calls = [], states = [], indices = [new Map(), new Map(), new Map()];
  const intern = (pool, index, value) => { const key = JSON.stringify(value); if (!index.has(key)) { index.set(key, pool.length); pool.push(value); } return index.get(key); };
  const packed = Object.fromEntries(Object.entries(rows).map(([key, row]) => [key, { ...row,
    controls: row.controls.map((control) => intern(controls, indices[0], control)),
    submission: { ...row.submission, calls: intern(calls, indices[1], row.submission.calls) },
    changes: row.changes.map((change) => ({ ...change, calls: intern(calls, indices[1], change.calls), persisted: change.persisted == null ? null : intern(states, indices[2], change.persisted) })),
    persisted: intern(states, indices[2], row.persisted),
  }]));
  return { rows: packed, controls, calls, states };
}
beforeAll(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  vi.spyOn(Date, "now").mockReturnValue(1794000000000);
  vi.spyOn(Math, "random").mockReturnValue(0.25);
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Baseline fixture must not fetch or execute a model"); }));
  if (capture) {
    expect(execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim()).toBe(old.source.revision);
    expect(execFileSync("git", ["diff", old.source.revision, "--", ...Object.keys(old.source.trees)], { encoding: "utf8", maxBuffer: 4 * 1024 * 1024 })).toBe("");
  } else requireOldPopulation(old.catalog, candidate, oldRows);
});
describe("pinned public catalog (SW-7028)", () => {
  it("retains every old catalog model even when no generation studio uses it", () => {
    expect(old.catalog.models).toHaveLength(143);
    expect(old.source.revision).toBe("d3f7b05bf22826e44fd43da19cff0985fb17d68b");
    expect(old.source.inferenceSource).toContain("4a36a7f8c9645110df4df656ca9d492e55ab6c7c");
    expect(() => requireOldPopulation(old.catalog, { ...candidate, models: candidate.models.filter((m) => m.id !== "starvector_1b") })).toThrow(/starvector_1b/);
    const reduced = candidateRows.filter((row) => row.model !== "starvector_1b");
    expect(unionPopulation(oldRows, reduced).filter((row) => row.model === "starvector_1b")).toHaveLength(6);
  });
  for (const row of unionPopulation(oldRows, candidateRows)) {
    it(keyFor(row), async () => {
      const actual = await observe(capture ? old.catalog : candidate, row);
      if (capture) captured[keyFor(row)] = actual;
      else expect(actual).toEqual(unpack(baseline.rows[keyFor(row)]));
    });
  }
  it("publishes only explicit old-tree captures", () => {
    if (capture) { expect(Object.keys(captured)).toEqual(oldRows.map(keyFor)); writeFileSync(observationsPath, `${renderBaseline({ source: old.source, population: oldRows, ...pack(captured) })}\n`); }
    else expect(Object.keys(baseline.rows)).toEqual(oldRows.map(keyFor));
  });
});
