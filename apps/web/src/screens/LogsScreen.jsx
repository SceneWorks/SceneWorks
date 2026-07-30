import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiFetch } from "../api.js";
import { useAppStatic } from "../context/AppContext.js";
import { isDesktop, tauriInvoke } from "../runtime.js";
import { WorkPanel } from "../components/WorkPanel.jsx";
import { writeClipboardText } from "../clipboard.js";

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
// The server session-log buffer holds up to DEFAULT_CAPACITY entries
// (crates/sceneworks-core/src/session_log.rs) and its `query()` applies the
// text-search filter BEFORE truncating to `limit`. Because free-text search is
// now performed client-side over the held snapshot (sc-8849), the snapshot must
// mirror the *entire* server buffer or matches in the older rows become
// silently unfindable. So the initial fetch and the in-memory row cap both
// track the server capacity rather than an arbitrary smaller number. There is
// no shared constant across the HTTP/FFI boundary, so this is pinned by comment.
const SESSION_LOG_CAPACITY = 5000; // == sceneworks-core session_log DEFAULT_CAPACITY
const MAX_ROWS = SESSION_LOG_CAPACITY;
// Render-window cap (sc-11224 / F-033). The in-memory snapshot mirrors the whole
// 5,000-entry server buffer so search stays complete, but mounting all 5,000 as DOM
// rows — and re-rendering them every 2s poll — is the actual cost. The viewer is a
// live tail: newest rows matter, so we render only the last LOG_RENDER_WINDOW rows of
// the (already search-filtered) list. Search still runs over the ENTIRE snapshot
// (visibleEntries below); windowing only bounds how many *matches* we paint, always
// the newest — which is what auto-scroll-to-bottom shows anyway.
const LOG_RENDER_WINDOW = 400;
// Poll backoff (sc-11224 / F-033). A healthy poll keeps the 2s cadence; consecutive
// failures back off exponentially up to a cap so a dead API (e.g. remote-auth mode
// with an unreachable host) isn't hammered every 2s forever. Mirrors the media-ticket
// / access-probe backoff loop in hooks/useAccessGate.js. Reset to POLL_MS on success.
const POLL_BACKOFF_MAX_MS = 30000;
// The full snapshot is already in memory, so text search filters client-side
// over the held `entries` instead of refetching. We still debounce the term
// before it drives the (cheap, in-memory) filter to keep typing snappy on large
// buffers, and — critically — searching no longer touches the fetch/poll deps,
// which stops the per-keystroke refetch + 2s-poll re-arm (sc-8849).
const SEARCH_DEBOUNCE_MS = 250;

// Events that answer the routing question get visual emphasis.
const HIGHLIGHT_EVENTS = new Set(["gpu_route_decision", "claim_lock_contention"]);

// How long the copy button reads "Copied" before reverting (#1966).
const COPY_FEEDBACK_MS = 1500;
// Sticky-offset fallback for the filter panel when the topbar can't be measured
// (jsdom, or a shell that renders the Logs screen without the app chrome).
// Roughly the topbar's natural height: 12px padding × 2 + a two-line title.
const TOPBAR_FALLBACK_PX = 64;

// What the per-entry copy button puts on the clipboard (#1966). The structured
// event is what people actually paste into a bug report, so prefer its
// pretty-printed form and fall back to the raw log line for entries the parser
// didn't turn into an event.
export function logEntryClipboardText(entry) {
  if (!entry) return "";
  if (entry.event) return JSON.stringify(entry.event, null, 2);
  return entry.raw ?? entry.message ?? "";
}

// The filter panel pins below the global topbar rather than at the top of the
// scroll box: `.workspace` is the scrolling ancestor and `.topbar` is already
// sticky at its top edge, so a `top: 0` panel would slide underneath it. The
// topbar's height is content-driven (its status pills wrap on narrow windows),
// so measure it instead of hardcoding an offset.
function useTopbarHeight() {
  const [height, setHeight] = useState(TOPBAR_FALLBACK_PX);
  useEffect(() => {
    const topbar = globalThis.document?.querySelector(".topbar");
    if (!topbar) return undefined;
    const measure = () => {
      setHeight(topbar.getBoundingClientRect().height || TOPBAR_FALLBACK_PX);
    };
    measure();
    if (typeof ResizeObserver !== "function") return undefined;
    const observer = new ResizeObserver(measure);
    observer.observe(topbar);
    return () => observer.disconnect();
  }, []);
  return height;
}

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
  const { token } = useAppStatic();
  const [entries, setEntries] = useState([]);
  const [source, setSource] = useState("");
  const [level, setLevel] = useState("");
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [paused, setPaused] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(null);
  // seq of the entry whose copy button just fired, plus whether it succeeded.
  const [copied, setCopied] = useState(null);

  const lastSeqRef = useRef(undefined);
  const bottomRef = useRef(null);
  const copyTimerRef = useRef(null);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  const topbarHeight = useTopbarHeight();

  // Clear the transient "Copied" flag on unmount so a late timer can't set
  // state on a torn-down tree.
  useEffect(() => () => clearTimeout(copyTimerRef.current), []);

  const copyEntry = useCallback(async (entry) => {
    const ok = await writeClipboardText(logEntryClipboardText(entry));
    setCopied({ seq: entry.seq, ok });
    clearTimeout(copyTimerRef.current);
    copyTimerRef.current = setTimeout(() => setCopied(null), COPY_FEEDBACK_MS);
  }, []);

  // Full (re)load: source/level filters changed, or first mount. Text search is
  // deliberately absent from the deps so typing doesn't refetch (sc-8849).
  const loadSnapshot = useCallback(async () => {
    try {
      // Fetch the full server buffer (not a 1000-row tail) so the client-side
      // search covers all held history — matches in the oldest ~4000 rows would
      // otherwise be unfindable (sc-8849). Only this initial snapshot is large;
      // the 2s poll below stays incremental (afterSeq) and returns small deltas.
      const rows = await fetchLogs(token, { limit: SESSION_LOG_CAPACITY, source, level });
      lastSeqRef.current = rows.length ? rows[rows.length - 1].seq : undefined;
      setEntries(rows);
      setError("");
    } catch (err) {
      setError(String(err?.message ?? err));
    }
  }, [token, source, level]);

  // Incremental tail: append only entries newer than the last seq we hold. Returns
  // `true` when the tick is healthy (a successful fetch, or a paused no-op) and
  // `false` when the fetch failed, so the scheduler below can back off (sc-11224).
  const poll = useCallback(async () => {
    if (pausedRef.current) return true;
    try {
      // Incremental: afterSeq means the server only returns rows newer than the
      // ones we hold, so this is a small delta each tick — raising the cap to the
      // buffer capacity just guards against a >1000-row burst between polls; it
      // does NOT make each poll refetch the whole buffer.
      const rows = await fetchLogs(token, {
        afterSeq: lastSeqRef.current,
        limit: SESSION_LOG_CAPACITY,
        source,
        level,
      });
      if (!rows.length) {
        setError("");
        return true;
      }
      lastSeqRef.current = rows[rows.length - 1].seq;
      setEntries((prev) => {
        const merged = prev.concat(rows);
        return merged.length > MAX_ROWS ? merged.slice(merged.length - MAX_ROWS) : merged;
      });
      setError("");
      return true;
    } catch (err) {
      setError(String(err?.message ?? err));
      return false;
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
      // Search `message` + `raw`. The server searched `raw` only; including the
      // (raw-derived) `message` here is a deliberate harmless superset — it can
      // only surface *more* matches, never hide one, so full-history parity with
      // the old server-side search is preserved (sc-8849).
      const haystack = `${entry.message ?? ""} ${entry.raw ?? ""}`.toLowerCase();
      return haystack.includes(needle);
    });
  }, [entries, debouncedSearch]);

  // Windowed render tail (sc-11224 / F-033): `visibleEntries` already ran the search
  // over the ENTIRE snapshot, so every match is accounted for; we only paint the last
  // LOG_RENDER_WINDOW of them. The live tail auto-scrolls to the newest row, so the
  // window is exactly what the user sees. `hiddenCount` surfaces how many older
  // (matching) rows are scrolled out of the rendered window.
  const renderedEntries =
    visibleEntries.length > LOG_RENDER_WINDOW ? visibleEntries.slice(-LOG_RENDER_WINDOW) : visibleEntries;
  const hiddenCount = visibleEntries.length - renderedEntries.length;

  // Self-scheduling poll with failure backoff (sc-11224 / F-033). Replaces a fixed
  // setInterval that fired every 2s regardless of outcome — a dead API in remote-auth
  // mode was hammered forever. A healthy tick re-arms at POLL_MS; consecutive failures
  // grow the delay exponentially up to POLL_BACKOFF_MAX_MS and reset on the next
  // success. Mirrors the media-ticket / access-probe backoff in useAccessGate.js.
  useEffect(() => {
    let closed = false;
    let timer = null;
    let attempt = 0;
    const schedule = (ms) => {
      timer = setTimeout(run, ms);
    };
    async function run() {
      const ok = await poll();
      if (closed) return;
      if (ok) {
        attempt = 0;
        schedule(POLL_MS);
      } else {
        const delay = Math.min(POLL_BACKOFF_MAX_MS, POLL_MS * 2 ** attempt);
        attempt += 1;
        schedule(delay);
      }
    }
    schedule(POLL_MS);
    return () => {
      closed = true;
      if (timer) clearTimeout(timer);
    };
  }, [poll]);

  // Auto-scroll to newest unless the user paused (or scrolled up).
  useEffect(() => {
    if (!paused && bottomRef.current) {
      bottomRef.current.scrollIntoView?.({ block: "end" });
    }
  }, [entries, paused]);

  return (
    <section className="page-frame logs-screen">
      {/* Pinned so the filters/pause control stay reachable while the tail
          scrolls (#1966); `--logs-sticky-top` is the measured topbar height. */}
      <WorkPanel
        eyebrow="Filter the stream"
        className="logs-panel"
        style={{ "--logs-sticky-top": `${topbarHeight}px` }}
      >
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
      </WorkPanel>

      {error ? (
        <p className="logs-error" role="alert">
          Couldn’t load logs: {error}
        </p>
      ) : null}

      <div className="logs-list" aria-live="polite">
        {visibleEntries.length === 0 && !error ? (
          <p className="logs-empty">No log entries yet for this session.</p>
        ) : null}
        {hiddenCount > 0 ? (
          <p className="logs-windowed" role="status">
            Showing the latest {renderedEntries.length.toLocaleString()} of {visibleEntries.length.toLocaleString()} entries.
          </p>
        ) : null}
        {renderedEntries.map((entry) => {
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
              {isOpen ? (
                <div className="logs-detail-panel">
                  {/* Copying the expanded entry used to mean hand-selecting the
                      JSON out of a live-tailing list (#1966). `stopPropagation`
                      keeps the click off the row's expand/collapse toggle. */}
                  <button
                    type="button"
                    className="logs-copy"
                    aria-label="Copy log entry to clipboard"
                    onClick={(clickEvent) => {
                      clickEvent.stopPropagation();
                      copyEntry(entry);
                    }}
                  >
                    {copied?.seq === entry.seq
                      ? copied.ok
                        ? "Copied"
                        : "Couldn’t copy"
                      : "Copy"}
                  </button>
                  {entry.event ? (
                    <pre className="logs-detail">{JSON.stringify(entry.event, null, 2)}</pre>
                  ) : null}
                </div>
              ) : null}
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </section>
  );
}

const TIME_FORMAT = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});

function shortTime(timestamp) {
  if (!timestamp) return "";
  // ISO 8601 (UTC, trailing Z) → HH:MM:SS in the system's local timezone.
  const date = new Date(timestamp);
  if (!Number.isNaN(date.getTime())) return TIME_FORMAT.format(date);
  // Fall back to the raw HH:MM:SS substring if the value can't be parsed.
  const match = /T(\d{2}:\d{2}:\d{2})/.exec(timestamp);
  return match ? match[1] : timestamp;
}

export default LogsScreen;
