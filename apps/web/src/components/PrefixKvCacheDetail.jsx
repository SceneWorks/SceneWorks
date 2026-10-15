import React from "react";
import { formatBytes } from "../formatting.js";

// Which prompt path a Qwen-Image 2.1 render ran (sc-25908, epic 25899 E6).
//
// A provider that declares the prefix KV cache (`qwen_image_2_1`, `qwen_image_2_1_turbo`) is
// priced per request: the admission runs its memory ladder cached-first and sheds the cache
// before it would refuse, and the engine itself sheds a cached selection whose real prefix
// outgrows the priced bound. The job result carries one record per written image
// (`result.prefixKvCache`, the worker's `execution_planner::prefix_kv_cache_record`):
// `{selected, pricedBytes, path, cacheBytes, overflow: {actualBytes, pricedBytes} | null}` —
// `selected` is what the admission chose, `path` what the render REPORTS it ran.
//
// Absent for every job whose provider prices no cache, so every other card renders unchanged.
// The path is in the summary, which stays visible while the disclosure is COLLAPSED (the default
// on the busy Queue); the bytes and the shed explanation are the detail behind it.

const PATH_LABELS = { cached: "cached", prefill: "prefill" };

function pathLabel(path) {
  return PATH_LABELS[path] ?? "not reported";
}

function records(job) {
  const list = job?.result?.prefixKvCache;
  return Array.isArray(list) ? list.filter((record) => record && typeof record === "object") : [];
}

function summaryText(list) {
  const counts = new Map();
  for (const record of list) {
    const label = pathLabel(record.path);
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  const paths =
    counts.size === 1
      ? [...counts.keys()][0]
      : [...counts.entries()].map(([label, count]) => `${count} ${label}`).join(", ");
  const shed = list.filter((record) => record.overflow).length;
  return `Prompt cache: ${paths}${shed ? ` · ${shed === 1 ? "cache shed" : `cache shed on ${shed}`}` : ""}`;
}

export function PrefixKvCacheDetail({ job }) {
  const list = records(job);
  if (list.length === 0) return null;
  return (
    <details className="worker-progress-card__prefix-cache">
      <summary>{summaryText(list)}</summary>
      <ul className="worker-progress-card__prefix-cache-list">
        {list.map((record, index) => (
          <li key={index}>
            <span>
              {list.length > 1 ? `Image ${index + 1}: ` : ""}
              ran <strong>{pathLabel(record.path)}</strong>
              {record.path === "cached" && record.cacheBytes != null
                ? ` · ${formatBytes(record.cacheBytes)} cache`
                : ""}
              {record.pricedBytes != null ? ` · ${formatBytes(record.pricedBytes)} priced` : ""}
              {record.selected && record.selected !== record.path
                ? ` · admission selected ${pathLabel(record.selected)}`
                : ""}
            </span>
            {record.overflow ? (
              <p className="worker-progress-card__prefix-cache-overflow" role="note">
                The engine shed the prompt cache: the prompt and reference images needed{" "}
                {formatBytes(record.overflow.actualBytes)}, more than the{" "}
                {formatBytes(record.overflow.pricedBytes)} the memory admission priced, so this
                image ran the prefill path, without a cached prompt prefix.
              </p>
            ) : null}
          </li>
        ))}
      </ul>
    </details>
  );
}
