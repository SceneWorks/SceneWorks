import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiFetch } from "../api.js";
import { useAppContext } from "../context/AppContext.js";
import { isDesktop, tauriInvoke } from "../runtime.js";

// In-app Logs viewer (sc-3452). Shows the current session's activity — most
// importantly the GPU routing decisions (`gpu_route_decision`) and claim
// contention (`claim_lock_contention`) — so "which backend ran this job?" is
// answerable from inside the app instead of by tailing ~/Library/Logs/SceneWorks.
//
// Data source: on the desktop the rich multi-source buffer (api + worker +
// mlx-worker) is read via the `get_session_logs` Tauri command (sc-3451); on
// web/Docker (and a remote LAN browser) the API-side buffer is read over HTTP
// (`GET /api/v1/logs`, sc-3453). `isDesktop`/`tauriInvoke` come from the unified
// runtime helper (epic 4484 story 6).

const SOURCES = ["api", "worker", "mlx-worker"];
const LEVELS = ["info", "warn", "error"];
const POLL_MS = 2000;
const MAX_ROWS = 2000;
// The full snapshot is already in memory, so text search filters client-side
// over the held `entries` instead of refetching. We still debounce the term
// before it drives the (cheap, in-memory) filter to keep typing snappy on large
// buffers, and — critically — searching no longer touches the fetch/poll deps,
// which stops the per-keystroke `limit:1000` refetch + 2s-poll re-arm (sc-8849).
const SEARCH_DEBOUNCE_MS = 250;

// Events that answer the routing question get visual emphasis.
const HIGHLIGHT_EVENTS = new Set(["gpu_route_decision", "claim_lock_contention"]);

// Note: text `search` is intentionally NOT a fetch parameter. Source/level are
// cheap, coarse toggles that legitimately change what the server returns, but
// the full snapshot is already held in memory, so free-text search filters
// client-side (see `visibleEntries`) rather than issuing a fresh limit:1000
// fetch per keystroke (sc-8849).
async function fetchLogs(token, { afterSeq, limit, source, level }) {
  if (isDesktop) {
    return (
      (await tauriInvoke("get_session_logs", {
        afterSeq,
        limit,
        source: source || undefined,
        level: level || undefined,
      })) ?? []
    );
  }
  const params = new URLSearchParams();
  if (afterSeq != null) params.set("afterSeq", String(afterSeq));
  if (limit != null) params.set("limit", String(limit));
  if (source) params.set("source", source);
  if (level) params.set("level", level);
  const query = params.toString();
  return (await apiFetch(`/api/v1/logs${query ? `?${query}` : ""}`, token)) ?? [];
}

export function LogsScreen() {
  const { token } = useAppContext();
  const [entries, setEntries] = useState([]);
  const [source, setSource] = useState("");
  const [level, setLevel] = useState("");
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [paused, setPaused] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(null);

  const lastSeqRef = useRef(undefined);
  const bottomRef = useRef(null);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  // Full (re)load: source/level filters changed, or first mount. Text search is
  // deliberately absent from the deps so typing doesn't refetch (sc-8849).
  const loadSnapshot = useCallback(async () => {
    try {
      const rows = await fetchLogs(token, { limit: 1000, source, level });
      lastSeqRef.current = rows.length ? rows[rows.length - 1].seq : undefined;
      setEntries(rows);
      setError("");
    } catch (err) {
      setError(String(err?.message ?? err));
    }
  }, [token, source, level]);

  // Incremental tail: append only entries newer than the last seq we hold.
  const poll = useCallback(async () => {
    if (pausedRef.current) return;
    try {
      const rows = await fetchLogs(token, {
        afterSeq: lastSeqRef.current,
        limit: 1000,
        source,
        level,
      });
      if (!rows.length) return;
      lastSeqRef.current = rows[rows.length - 1].seq;
      setEntries((prev) => {
        const merged = prev.concat(rows);
        return merged.length > MAX_ROWS ? merged.slice(merged.length - MAX_ROWS) : merged;
      });
      setError("");
    } catch (err) {
      setError(String(err?.message ?? err));
    }
  }, [token, source, level]);

  useEffect(() => {
    loadSnapshot();
  }, [loadSnapshot]);

  // Debounce the raw search term (~250ms) before it drives the client-side
  // filter, so a fast typist doesn't recompute the filtered list on every
  // keystroke. This never touches the fetch/poll deps (sc-8849).
  useEffect(() => {
    const id = setTimeout(() => setDebouncedSearch(search), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(id);
  }, [search]);

  // Client-side text filter over the already-held snapshot: no network, and no
  // stale-prefix interleave because there are no in-flight per-keystroke fetches.
  const visibleEntries = useMemo(() => {
    const needle = debouncedSearch.trim().toLowerCase();
    if (!needle) return entries;
    return entries.filter((entry) => {
      const haystack = `${entry.message ?? ""} ${entry.raw ?? ""}`.toLowerCase();
      return haystack.includes(needle);
    });
  }, [entries, debouncedSearch]);

  useEffect(() => {
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, [poll]);

  // Auto-scroll to newest unless the user paused (or scrolled up).
  useEffect(() => {
    if (!paused && bottomRef.current) {
      bottomRef.current.scrollIntoView?.({ block: "end" });
    }
  }, [entries, paused]);

  return (
    <section className="main-surface logs-screen">
      <div className="logs-toolbar" role="toolbar" aria-label="Log filters">
        <div className="segmented-control" role="group" aria-label="Source">
          <button
            type="button"
            className={source === "" ? "active" : ""}
            onClick={() => setSource("")}
          >
            All sources
          </button>
          {SOURCES.map((value) => (
            <button
              key={value}
              type="button"
              className={source === value ? "active" : ""}
              onClick={() => setSource(value)}
            >
              {value}
            </button>
          ))}
        </div>
        <div className="segmented-control" role="group" aria-label="Level">
          <button
            type="button"
            className={level === "" ? "active" : ""}
            onClick={() => setLevel("")}
          >
            All levels
          </button>
          {LEVELS.map((value) => (
            <button
              key={value}
              type="button"
              className={level === value ? "active" : ""}
              onClick={() => setLevel(value)}
            >
              {value}
            </button>
          ))}
        </div>
        <input
          type="search"
          className="logs-search"
          placeholder="Search log text…"
          aria-label="Search logs"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <button
          type="button"
          className={paused ? "logs-live paused" : "logs-live"}
          aria-pressed={paused}
          onClick={() => setPaused((value) => !value)}
        >
          {paused ? "Paused" : "● Live"}
        </button>
      </div>

      {error ? (
        <p className="logs-error" role="alert">
          Couldn’t load logs: {error}
        </p>
      ) : null}

      <div className="logs-list" aria-live="polite">
        {visibleEntries.length === 0 && !error ? (
          <p className="logs-empty">No log entries yet for this session.</p>
        ) : null}
        {visibleEntries.map((entry) => {
          const eventName = entry.event?.event;
          const highlighted = eventName && HIGHLIGHT_EVENTS.has(eventName);
          const isOpen = expanded === entry.seq;
          return (
            <div
              key={entry.seq}
              className={`logs-row level-${entry.level}${highlighted ? " highlighted" : ""}`}
              onClick={() => setExpanded(isOpen ? null : entry.seq)}
            >
              <span className="logs-time">{shortTime(entry.timestamp)}</span>
              <span className={`logs-chip source-${entry.source}`}>{entry.source}</span>
              <span className={`logs-chip level-${entry.level}`}>{entry.level}</span>
              <span className="logs-message">{entry.message}</span>
              {isOpen && entry.event ? (
                <pre className="logs-detail">{JSON.stringify(entry.event, null, 2)}</pre>
              ) : null}
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </section>
  );
}

function shortTime(timestamp) {
  if (!timestamp) return "";
  // ISO 8601 → HH:MM:SS (drop date + zone for a compact column).
  const match = /T(\d{2}:\d{2}:\d{2})/.exec(timestamp);
  return match ? match[1] : timestamp;
}

export default LogsScreen;
