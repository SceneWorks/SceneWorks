import React, { useCallback, useEffect, useRef, useState } from "react";

import { apiFetch } from "../api.js";
import { useAppContext } from "../context/AppContext.js";

// In-app Logs viewer (sc-3452). Shows the current session's activity — most
// importantly the MLX↔torch routing decisions (`mlx_route_decision`) and claim
// contention (`claim_lock_contention`) — so "why did this run on MPS not MLX?" is
// answerable from inside the app instead of by tailing ~/Library/Logs/SceneWorks.
//
// Data source: on the desktop the rich multi-source buffer (api + worker +
// mlx-worker) is read via the `get_session_logs` Tauri command (sc-3451); on
// web/Docker the API-side buffer is read over HTTP (`GET /api/v1/logs`, sc-3453).
const isDesktop = typeof window !== "undefined" && !!window.__TAURI__;
const tauriInvoke = (command, args) => window.__TAURI__.core.invoke(command, args);

const SOURCES = ["api", "worker", "mlx-worker"];
const LEVELS = ["info", "warn", "error"];
const POLL_MS = 2000;
const MAX_ROWS = 2000;

// Events that answer the routing question get visual emphasis.
const HIGHLIGHT_EVENTS = new Set(["mlx_route_decision", "claim_lock_contention"]);

async function fetchLogs(token, { afterSeq, limit, source, level, search }) {
  if (isDesktop) {
    return (
      (await tauriInvoke("get_session_logs", {
        afterSeq,
        limit,
        source: source || undefined,
        level: level || undefined,
        search: search || undefined,
      })) ?? []
    );
  }
  const params = new URLSearchParams();
  if (afterSeq != null) params.set("afterSeq", String(afterSeq));
  if (limit != null) params.set("limit", String(limit));
  if (source) params.set("source", source);
  if (level) params.set("level", level);
  if (search) params.set("search", search);
  const query = params.toString();
  return (await apiFetch(`/api/v1/logs${query ? `?${query}` : ""}`, token)) ?? [];
}

export function LogsScreen() {
  const { token } = useAppContext();
  const [entries, setEntries] = useState([]);
  const [source, setSource] = useState("");
  const [level, setLevel] = useState("");
  const [search, setSearch] = useState("");
  const [paused, setPaused] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(null);

  const lastSeqRef = useRef(undefined);
  const bottomRef = useRef(null);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  // Full (re)load: filters changed, or first mount.
  const loadSnapshot = useCallback(async () => {
    try {
      const rows = await fetchLogs(token, { limit: 1000, source, level, search });
      lastSeqRef.current = rows.length ? rows[rows.length - 1].seq : undefined;
      setEntries(rows);
      setError("");
    } catch (err) {
      setError(String(err?.message ?? err));
    }
  }, [token, source, level, search]);

  // Incremental tail: append only entries newer than the last seq we hold.
  const poll = useCallback(async () => {
    if (pausedRef.current) return;
    try {
      const rows = await fetchLogs(token, {
        afterSeq: lastSeqRef.current,
        limit: 1000,
        source,
        level,
        search,
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
  }, [token, source, level, search]);

  useEffect(() => {
    loadSnapshot();
  }, [loadSnapshot]);

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
        {entries.length === 0 && !error ? (
          <p className="logs-empty">No log entries yet for this session.</p>
        ) : null}
        {entries.map((entry) => {
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
