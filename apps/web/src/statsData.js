// Pure data helpers for the Generation Stats screen (epic 10402, sc-10408).
// Kept out of StatsScreen.jsx so the filter / sort / aggregation logic is
// unit-testable without rendering. Operates on GenerationMetricsRow objects
// ({ jobId, type, status, createdAt, metrics: { model, quantLabel, totalMs, … } }).

function num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

// Column key → sort value. Numeric columns return a number|null; text columns a
// string. Missing values sort last regardless of direction (see sortRows).
export const SORT_ACCESSORS = {
  createdAt: (r) => r.createdAt ?? "",
  model: (r) => r.metrics?.model ?? "",
  quant: (r) => r.metrics?.quantLabel ?? "",
  sampler: (r) => r.metrics?.sampler ?? "",
  scheduler: (r) => r.metrics?.scheduler ?? "",
  cfg: (r) => num(r.metrics?.guidanceScale),
  steps: (r) => num(r.metrics?.steps),
  images: (r) => num(r.metrics?.imageCount),
  load: (r) => num(r.metrics?.loadMs),
  sample: (r) => num(r.metrics?.sampleMs),
  decode: (r) => num(r.metrics?.decodeMs),
  total: (r) => num(r.metrics?.totalMs),
  peakMem: (r) => num(r.metrics?.peakMemoryPct),
  gpu: (r) => num(r.metrics?.peakGpuLoadPct),
  status: (r) => r.status ?? "",
};

function isEmpty(value) {
  return value === null || value === undefined || value === "";
}

export function filterRows(rows, filters = {}) {
  return rows.filter((r) => {
    if (filters.type && r.type !== filters.type) return false;
    if (filters.model && r.metrics?.model !== filters.model) return false;
    if (filters.quant && r.metrics?.quantLabel !== filters.quant) return false;
    if (filters.status && r.status !== filters.status) return false;
    return true;
  });
}

export function sortRows(rows, { key = "createdAt", dir = "desc" } = {}) {
  const accessor = SORT_ACCESSORS[key] ?? SORT_ACCESSORS.createdAt;
  const sign = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const av = accessor(a);
    const bv = accessor(b);
    // Missing values always sink to the bottom, whichever way we're sorting.
    if (isEmpty(av)) return isEmpty(bv) ? 0 : 1;
    if (isEmpty(bv)) return -1;
    if (av < bv) return -sign;
    if (av > bv) return sign;
    return 0;
  });
}

export function deriveFilterOptions(rows) {
  const types = new Set();
  const models = new Set();
  const quants = new Set();
  const statuses = new Set();
  for (const r of rows) {
    if (r.type) types.add(r.type);
    if (r.metrics?.model) models.add(r.metrics.model);
    if (r.metrics?.quantLabel) quants.add(r.metrics.quantLabel);
    if (r.status) statuses.add(r.status);
  }
  const sorted = (set) => [...set].sort();
  return {
    types: sorted(types),
    models: sorted(models),
    quants: sorted(quants),
    statuses: sorted(statuses),
  };
}

export function median(values) {
  const nums = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!nums.length) return null;
  const mid = Math.floor(nums.length / 2);
  return nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
}

// The job types the comparison charts (sc-10409) aggregate — generation jobs,
// where quant/sampler/scheduler/cfg/phases are meaningful. The list/detail view
// still shows every type; only the charts restrict to these.
const GENERATION_TYPES = new Set([
  "image_generate",
  "image_edit",
  "image_detail",
  "video_generate",
]);

export function isGenerationRow(row) {
  return GENERATION_TYPES.has(row?.type);
}

// Group-by dimensions for the charts. Continuous axes (cfg) are bucketed to a
// label so they group cleanly. Returns null to drop a row from a group.
export const GROUP_BY = {
  quant: (r) => r.metrics?.quantLabel ?? null,
  model: (r) => r.metrics?.model ?? null,
  scheduler: (r) => r.metrics?.scheduler ?? null,
  sampler: (r) => r.metrics?.sampler ?? null,
  guidanceMethod: (r) => r.metrics?.guidanceMethod ?? null,
  pid: (r) => (r.metrics?.usePid ? "PiD" : "native"),
  cfg: (r) => {
    const c = Number(r.metrics?.guidanceScale);
    return Number.isFinite(c) ? `cfg ${(Math.round(c * 2) / 2).toFixed(1)}` : null;
  },
};

export const GROUP_BY_LABELS = {
  quant: "quant",
  model: "model",
  scheduler: "scheduler",
  sampler: "sampler",
  guidanceMethod: "guidance method",
  pid: "PiD on/off",
  cfg: "cfg",
};

function msToSeconds(ms) {
  return ms === null ? 0 : Math.round(ms / 100) / 10;
}

// Median load/sample/decode SECONDS per group (generation rows only), for the
// stacked phase-timing bar chart. Sorted by group label; carries a run count.
export function groupPhaseTimings(rows, groupKey) {
  const accessor = GROUP_BY[groupKey] ?? GROUP_BY.quant;
  const groups = new Map();
  const push = (arr, value) => {
    const n = Number(value);
    if (Number.isFinite(n)) arr.push(n);
  };
  for (const r of rows) {
    if (!isGenerationRow(r)) continue;
    const group = accessor(r);
    if (group === null || group === undefined || group === "") continue;
    if (!groups.has(group)) {
      groups.set(group, { load: [], sample: [], decode: [], count: 0 });
    }
    const bucket = groups.get(group);
    // Amortize per image so batch sizes compare fairly (sc-10426): the timings
    // are batch totals, so a 4-image job's phases divide by its image count.
    const per = Math.max(1, Number(r.metrics?.imageCount) || 1);
    push(bucket.load, Number(r.metrics?.loadMs) / per);
    push(bucket.sample, Number(r.metrics?.sampleMs) / per);
    push(bucket.decode, Number(r.metrics?.decodeMs) / per);
    bucket.count += 1;
  }
  return [...groups.entries()]
    .map(([group, b]) => ({
      group: String(group),
      load: msToSeconds(median(b.load)),
      sample: msToSeconds(median(b.sample)),
      decode: msToSeconds(median(b.decode)),
      count: b.count,
    }))
    .sort((a, b) => (a.group < b.group ? -1 : a.group > b.group ? 1 : 0));
}

// Scatter points (steps → total seconds) grouped into a series per quant tier,
// for the steps-vs-time chart. Generation rows only.
export function scatterByQuant(rows) {
  const byQuant = new Map();
  for (const r of rows) {
    if (!isGenerationRow(r)) continue;
    const quant = r.metrics?.quantLabel ?? "unknown";
    const steps = Number(r.metrics?.steps);
    const per = Math.max(1, Number(r.metrics?.imageCount) || 1);
    const total = Number(r.metrics?.totalMs) / per; // per-image (sc-10426)
    if (!Number.isFinite(steps) || !Number.isFinite(total)) continue;
    if (!byQuant.has(quant)) byQuant.set(quant, []);
    byQuant.get(quant).push({ steps, total: msToSeconds(total) });
  }
  return [...byQuant.entries()]
    .map(([quant, points]) => ({ quant, points }))
    .sort((a, b) => (a.quant < b.quant ? -1 : a.quant > b.quant ? 1 : 0));
}

export function computeKpis(rows) {
  const totals = rows.map((r) => num(r.metrics?.totalMs)).filter((v) => v !== null);
  const mems = rows.map((r) => num(r.metrics?.peakMemoryPct)).filter((v) => v !== null);
  // Fastest quant = the quant tier with the lowest median total time.
  const byQuant = new Map();
  for (const r of rows) {
    const quant = r.metrics?.quantLabel;
    const total = num(r.metrics?.totalMs);
    if (!quant || total === null) continue;
    if (!byQuant.has(quant)) byQuant.set(quant, []);
    byQuant.get(quant).push(total);
  }
  let fastestQuant = null;
  let fastestMedian = Infinity;
  for (const [quant, arr] of byQuant) {
    const m = median(arr);
    if (m !== null && m < fastestMedian) {
      fastestMedian = m;
      fastestQuant = quant;
    }
  }
  return {
    runs: rows.length,
    medianTotalMs: median(totals),
    medianPeakMemPct: median(mems),
    fastestQuant,
  };
}
