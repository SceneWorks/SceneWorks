import React, { useMemo, useState } from "react";

import { useAppStatic } from "../context/AppContext.js";
import { useGenerationMetrics } from "../hooks/useGenerationMetrics.js";
import { formatBytes, formatMs, formatPercent, quantLabel } from "../formatting.js";
import { computeKpis, deriveFilterOptions, filterRows, sortRows } from "../statsData.js";

// Generation Stats screen (epic 10402, sc-10408): a filterable, sortable list of
// every run with its captured metrics, plus a per-run detail panel. Reads the
// aggregate feed via useGenerationMetrics (GET /api/v1/metrics); comparison
// charts are added on top by sc-10409.

const TYPE_LABELS = {
  image_generate: "Image",
  image_edit: "Image edit",
  image_detail: "Detail",
  image_vqa: "VQA",
  image_interleave: "Interleave",
  video_generate: "Video",
  training: "Training",
  caption: "Caption",
  prompt_refine: "Prompt",
};

function typeLabel(type) {
  return TYPE_LABELS[type] ?? type ?? "—";
}

function num1(value) {
  const n = Number(value);
  return Number.isFinite(n) ? String(Math.round(n * 10) / 10) : "—";
}

function pidLabel(metrics) {
  if (!metrics?.usePid) return "—";
  return metrics.pidTarget ? String(metrics.pidTarget).toUpperCase() : "on";
}

function formatDate(iso) {
  if (!iso) return "—";
  const ms = Date.parse(iso);
  if (!Number.isFinite(ms)) return "—";
  const d = new Date(ms);
  return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

function statusTone(status) {
  if (status === "completed" || status === "done") return "success";
  if (status === "failed" || status === "canceled" || status === "interrupted") return "danger";
  return "neutral";
}

const COLUMNS = [
  { key: "model", label: "model", get: (r) => r.metrics?.model ?? "—" },
  { key: "quant", label: "quant", get: (r) => quantLabel(r.metrics?.quantLabel) },
  { key: "sampler", label: "sampler", get: (r) => r.metrics?.sampler ?? "—" },
  { key: "scheduler", label: "sched", get: (r) => r.metrics?.scheduler ?? "—" },
  { key: "cfg", label: "cfg", numeric: true, get: (r) => num1(r.metrics?.guidanceScale) },
  { key: "steps", label: "steps", numeric: true, get: (r) => r.metrics?.steps ?? "—" },
  { key: null, label: "PiD", get: (r) => pidLabel(r.metrics) },
  { key: "load", label: "load", numeric: true, get: (r) => formatMs(r.metrics?.loadMs) },
  { key: "sample", label: "sample", numeric: true, get: (r) => formatMs(r.metrics?.sampleMs) },
  { key: "decode", label: "decode", numeric: true, get: (r) => formatMs(r.metrics?.decodeMs) },
  { key: "total", label: "total", numeric: true, get: (r) => formatMs(r.metrics?.totalMs) },
  { key: "peakMem", label: "peak mem", numeric: true, get: (r) => formatPercent(r.metrics?.peakMemoryPct) },
  { key: "gpu", label: "gpu", numeric: true, get: (r) => formatPercent(r.metrics?.peakGpuLoadPct) },
  { key: "status", label: "status" },
  { key: "createdAt", label: "created", get: (r) => formatDate(r.createdAt) },
];

function FilterSelect({ label, value, onChange, options, render = (v) => v, allLabel }) {
  return (
    <label className="stats-filter">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">{allLabel}</option>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {render(opt)}
          </option>
        ))}
      </select>
    </label>
  );
}

function RunDetail({ row, onClose }) {
  const m = row.metrics ?? {};
  const items = [
    ["Job", `${typeLabel(row.type)} · ${row.jobId}`],
    ["Model", m.model ?? "—"],
    ["Quant", `${quantLabel(m.quantLabel)}${m.quantBits ? ` (${m.quantBits}-bit)` : ""}`],
    ["Backend", m.backend ?? "—"],
    [
      "Sampler / scheduler",
      `${m.sampler ?? "—"} / ${m.scheduler ?? "—"}${
        m.schedulerShift != null ? ` (shift ${num1(m.schedulerShift)})` : ""
      }`,
    ],
    [
      "Guidance",
      `${num1(m.guidanceScale)}${
        m.guidanceMethod && m.guidanceMethod !== "cfg" ? ` · ${m.guidanceMethod}` : ""
      }${m.trueCfgScale != null ? ` · trueCfg ${num1(m.trueCfgScale)}` : ""}`,
    ],
    ["Steps", m.steps ?? "—"],
    ["PiD", pidLabel(m)],
    ["Size", m.width && m.height ? `${m.width}×${m.height}` : "—"],
    ["Seed", m.seed ?? "—"],
    ["LoRAs", m.loras?.length ? m.loras.join(", ") : "—"],
    [
      "Load / sample / decode",
      `${formatMs(m.loadMs)} / ${formatMs(m.sampleMs)} / ${formatMs(m.decodeMs)}`,
    ],
    ["Total time", formatMs(m.totalMs)],
    ["Peak memory", `${formatPercent(m.peakMemoryPct)} (${formatBytes(m.peakMemoryBytes)})`],
    ["Peak GPU load", formatPercent(m.peakGpuLoadPct)],
    ["Created", formatDate(row.createdAt)],
  ];
  return (
    <div className="stats-detail">
      <div className="stats-detail-head">
        <h3>Run detail</h3>
        <button type="button" className="stats-detail-close" onClick={onClose} aria-label="Close detail">
          ×
        </button>
      </div>
      <div className="stats-detail-grid">
        {items.map(([label, value]) => (
          <div className="stats-kv" key={label}>
            <span className="stats-kv-label">{label}</span>
            <span className="stats-kv-value">{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function StatsScreen() {
  const { token } = useAppStatic();
  const { rows, loading, error, refresh } = useGenerationMetrics({ token });
  const [filters, setFilters] = useState({ type: "", model: "", quant: "", status: "" });
  const [sort, setSort] = useState({ key: "createdAt", dir: "desc" });
  const [selectedId, setSelectedId] = useState(null);

  const options = useMemo(() => deriveFilterOptions(rows), [rows]);
  const filtered = useMemo(() => filterRows(rows, filters), [rows, filters]);
  const sorted = useMemo(() => sortRows(filtered, sort), [filtered, sort]);
  const kpis = useMemo(() => computeKpis(filtered), [filtered]);
  const selected = useMemo(
    () => sorted.find((r) => r.jobId === selectedId) ?? null,
    [sorted, selectedId],
  );

  const setFilter = (key, value) => setFilters((prev) => ({ ...prev, [key]: value }));
  const toggleSort = (key) => {
    if (!key) return;
    setSort((prev) =>
      prev.key === key
        ? { key, dir: prev.dir === "asc" ? "desc" : "asc" }
        : { key, dir: key === "createdAt" ? "desc" : "asc" },
    );
  };

  return (
    <section className="main-surface stats-surface">
      <div className="surface-header hero">
        <div className="section-heading">
          <p className="eyebrow">System</p>
          <h2>Generation stats</h2>
          <p className="hero-blurb">
            Compare runs by model, quant, and settings — with per-phase timing, peak memory, and GPU load
            for every job.
          </p>
        </div>
      </div>

      <div className="hero-stats">
        <div className="hero-stat">
          <span className="hero-stat-label">Runs</span>
          <span className="hero-stat-value">{kpis.runs}</span>
        </div>
        <div className="hero-stat">
          <span className="hero-stat-label">Median total</span>
          <span className="hero-stat-value">{formatMs(kpis.medianTotalMs)}</span>
        </div>
        <div className="hero-stat">
          <span className="hero-stat-label">Median peak mem</span>
          <span className="hero-stat-value">{formatPercent(kpis.medianPeakMemPct)}</span>
        </div>
        <div className="hero-stat">
          <span className="hero-stat-label">Fastest quant</span>
          <span className="hero-stat-value">{kpis.fastestQuant ?? "—"}</span>
        </div>
      </div>

      <div className="stats-filters">
        <FilterSelect
          label="Job type"
          value={filters.type}
          onChange={(v) => setFilter("type", v)}
          options={options.types}
          render={typeLabel}
          allLabel="All jobs"
        />
        <FilterSelect
          label="Model"
          value={filters.model}
          onChange={(v) => setFilter("model", v)}
          options={options.models}
          allLabel="All models"
        />
        <FilterSelect
          label="Quant"
          value={filters.quant}
          onChange={(v) => setFilter("quant", v)}
          options={options.quants}
          allLabel="All quants"
        />
        <FilterSelect
          label="Status"
          value={filters.status}
          onChange={(v) => setFilter("status", v)}
          options={options.statuses}
          allLabel="All statuses"
        />
        <button type="button" className="stats-refresh" onClick={refresh} disabled={loading}>
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      {error ? <p className="stats-error">{error}</p> : null}

      <div className="stats-table-wrap">
        <table className="stats-table">
          <thead>
            <tr>
              {COLUMNS.map((col) => (
                <th
                  key={col.label}
                  className={col.numeric ? "num" : ""}
                  onClick={() => toggleSort(col.key)}
                  style={{ cursor: col.key ? "pointer" : "default" }}
                >
                  {col.label}
                  {sort.key === col.key ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr
                key={r.jobId}
                className={r.jobId === selectedId ? "selected" : ""}
                onClick={() => setSelectedId(r.jobId)}
              >
                {COLUMNS.map((col) => (
                  <td key={col.label} className={col.numeric ? "num mono" : ""}>
                    {col.key === "status" ? (
                      <span className={`stats-pill stats-pill-${statusTone(r.status)}`}>
                        {r.status ?? "—"}
                      </span>
                    ) : (
                      col.get(r)
                    )}
                  </td>
                ))}
              </tr>
            ))}
            {!sorted.length && !loading ? (
              <tr>
                <td colSpan={COLUMNS.length} className="stats-empty">
                  No runs yet. Generate something and it&apos;ll show up here.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      {selected ? <RunDetail row={selected} onClose={() => setSelectedId(null)} /> : null}
    </section>
  );
}
