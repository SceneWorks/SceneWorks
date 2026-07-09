import { describe, expect, it } from "vitest";

import {
  computeKpis,
  deriveFilterOptions,
  filterRows,
  groupPhaseTimings,
  isGenerationRow,
  median,
  scatterByQuant,
  sortRows,
} from "./statsData.js";

function row(jobId, type, status, createdAt, metrics) {
  return { jobId, type, status, createdAt, metrics };
}

const ROWS = [
  row("a", "image_generate", "done", "2026-07-01T10:00:00Z", {
    model: "qwen_image",
    quantLabel: "q8",
    totalMs: 9400,
    peakMemoryPct: 71,
    steps: 20,
    guidanceScale: 4.0,
  }),
  row("b", "image_generate", "failed", "2026-07-02T10:00:00Z", {
    model: "flux2_dev",
    quantLabel: "bf16",
    totalMs: 20500,
    peakMemoryPct: 94,
    steps: 28,
    guidanceScale: 3.5,
  }),
  row("c", "video_generate", "done", "2026-07-03T10:00:00Z", {
    model: "qwen_image",
    quantLabel: "q4",
    totalMs: 7500,
    peakMemoryPct: 54,
    steps: 20,
  }),
];

describe("filterRows", () => {
  it("filters by type, model, quant, status", () => {
    expect(filterRows(ROWS, { type: "image_generate" }).map((r) => r.jobId)).toEqual(["a", "b"]);
    expect(filterRows(ROWS, { model: "qwen_image" }).map((r) => r.jobId)).toEqual(["a", "c"]);
    expect(filterRows(ROWS, { quant: "bf16" }).map((r) => r.jobId)).toEqual(["b"]);
    expect(filterRows(ROWS, { status: "failed" }).map((r) => r.jobId)).toEqual(["b"]);
  });
  it("returns all rows with no filters", () => {
    expect(filterRows(ROWS, {}).length).toBe(3);
  });
});

describe("sortRows", () => {
  it("sorts numeric columns ascending and descending", () => {
    expect(sortRows(ROWS, { key: "total", dir: "asc" }).map((r) => r.jobId)).toEqual(["c", "a", "b"]);
    expect(sortRows(ROWS, { key: "total", dir: "desc" }).map((r) => r.jobId)).toEqual(["b", "a", "c"]);
  });
  it("sinks missing values to the bottom regardless of direction", () => {
    const withGap = [...ROWS, row("d", "image_generate", "done", "2026-07-04T10:00:00Z", { model: "sdxl" })];
    // d has no totalMs → last in both directions
    expect(sortRows(withGap, { key: "total", dir: "asc" }).at(-1).jobId).toBe("d");
    expect(sortRows(withGap, { key: "total", dir: "desc" }).at(-1).jobId).toBe("d");
  });
  it("does not mutate the input", () => {
    const copy = [...ROWS];
    sortRows(ROWS, { key: "total", dir: "asc" });
    expect(ROWS).toEqual(copy);
  });
});

describe("deriveFilterOptions", () => {
  it("collects sorted, de-duplicated option lists", () => {
    const opts = deriveFilterOptions(ROWS);
    expect(opts.types).toEqual(["image_generate", "video_generate"]);
    expect(opts.models).toEqual(["flux2_dev", "qwen_image"]);
    expect(opts.quants).toEqual(["bf16", "q4", "q8"]);
    expect(opts.statuses).toEqual(["done", "failed"]);
  });
});

describe("median", () => {
  it("handles odd and even counts and empties", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([1, 2, 3, 4])).toBe(2.5);
    expect(median([])).toBe(null);
  });
});

const GEN_ROWS = [
  row("g1", "image_generate", "completed", "2026-07-01T10:00:00Z", {
    model: "qwen_image", quantLabel: "q8", steps: 20, guidanceScale: 4.0,
    loadMs: 2000, sampleMs: 6000, decodeMs: 1000, totalMs: 9000,
  }),
  row("g2", "image_generate", "completed", "2026-07-02T10:00:00Z", {
    model: "qwen_image", quantLabel: "q8", steps: 20, guidanceScale: 4.0,
    loadMs: 2200, sampleMs: 6400, decodeMs: 1000, totalMs: 9600,
  }),
  row("g3", "image_generate", "completed", "2026-07-03T10:00:00Z", {
    model: "sdxl", quantLabel: "q4", steps: 30, guidanceScale: 7.0,
    loadMs: 1400, sampleMs: 3300, decodeMs: 600, totalMs: 5300,
  }),
  // Non-generation → excluded from the charts.
  row("c1", "caption", "completed", "2026-07-04T10:00:00Z", { model: "joycaption", totalMs: 1200 }),
];

describe("isGenerationRow", () => {
  it("accepts image/video gen types and rejects others", () => {
    expect(isGenerationRow({ type: "image_generate" })).toBe(true);
    expect(isGenerationRow({ type: "video_generate" })).toBe(true);
    expect(isGenerationRow({ type: "caption" })).toBe(false);
    expect(isGenerationRow(null)).toBe(false);
  });
});

describe("groupPhaseTimings", () => {
  it("medians load/sample/decode seconds per group, generation rows only", () => {
    const groups = groupPhaseTimings(GEN_ROWS, "quant");
    expect(groups.map((g) => g.group)).toEqual(["q4", "q8"]); // caption excluded
    const q8 = groups.find((g) => g.group === "q8");
    expect(q8.count).toBe(2);
    expect(q8.load).toBe(2.1); // median(2000,2200)=2100ms → 2.1s
    expect(q8.sample).toBe(6.2);
    expect(q8.decode).toBe(1);
  });
  it("re-buckets when the group dimension changes", () => {
    expect(groupPhaseTimings(GEN_ROWS, "model").map((g) => g.group)).toEqual(["qwen_image", "sdxl"]);
  });
});

describe("scatterByQuant", () => {
  it("builds a steps→seconds series per quant, generation rows only", () => {
    const series = scatterByQuant(GEN_ROWS);
    expect(series.map((s) => s.quant)).toEqual(["q4", "q8"]);
    expect(series.find((s) => s.quant === "q8").points).toEqual([
      { steps: 20, total: 9 },
      { steps: 20, total: 9.6 },
    ]);
  });
});

describe("computeKpis", () => {
  it("counts runs, medians, and the fastest quant tier", () => {
    const kpis = computeKpis(ROWS);
    expect(kpis.runs).toBe(3);
    expect(kpis.medianTotalMs).toBe(9400);
    expect(kpis.medianPeakMemPct).toBe(71);
    // q4 (7500) is the fastest single-run tier here
    expect(kpis.fastestQuant).toBe("q4");
  });
});
