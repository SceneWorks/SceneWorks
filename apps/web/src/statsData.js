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
