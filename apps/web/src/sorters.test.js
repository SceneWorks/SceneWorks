import { describe, expect, it, vi } from "vitest";
import {
  MAX_TERMINAL_JOBS,
  capTerminalJobs,
  insertNewest,
  sortNewest,
  upsertJobNewest,
} from "./sorters.js";

// sc-8860 (F-058): the `jobs` array used to be fully copy-sorted on every SSE
// `job.updated` tick (`[job, ...items.filter(...)].sort(sortNewest)`) and nothing
// pruned terminal jobs, so a long session grew the array unbounded and paid an
// O(n log n) sort per progress tick. These tests pin the replacement: an O(n)
// createdAt-ordered insertion that yields output identical to the old sort, plus a
// terminal-jobs cap that never drops an active job.

const job = (id, createdAt, status = "queued") => ({ id, createdAt, status });

// The pre-fix expression, kept here as the oracle every new path must match.
const legacyUpsert = (items, incoming) =>
  [incoming, ...items.filter((item) => item.id !== incoming.id)].sort(sortNewest);

describe("insertNewest (sc-8860)", () => {
  it("keeps the list newest-first, matching the old copy-sort output", () => {
    const items = [
      job("c", "2026-07-01T03:00:00Z"),
      job("a", "2026-07-01T01:00:00Z"),
    ];
    // Insert one that belongs in the middle by createdAt.
    const incoming = job("b", "2026-07-01T02:00:00Z");
    const next = insertNewest(items, incoming);
    expect(next.map((j) => j.id)).toEqual(["c", "b", "a"]);
    // Byte-for-byte order parity with the replaced expression.
    expect(next.map((j) => j.id)).toEqual(legacyUpsert(items, incoming).map((j) => j.id));
  });

  it("replaces an existing job in place (no duplicate) and re-positions it", () => {
    const items = [
      job("c", "2026-07-01T03:00:00Z"),
      job("b", "2026-07-01T02:00:00Z"),
      job("a", "2026-07-01T01:00:00Z"),
    ];
    // `b` gets a progress update: same id, same createdAt (createdAt never changes).
    const updated = job("b", "2026-07-01T02:00:00Z", "running");
    const next = insertNewest(items, updated);
    expect(next.map((j) => j.id)).toEqual(["c", "b", "a"]);
    expect(next.filter((j) => j.id === "b")).toHaveLength(1);
    expect(next.find((j) => j.id === "b").status).toBe("running");
    expect(next.map((j) => j.id)).toEqual(legacyUpsert(items, updated).map((j) => j.id));
  });

  it("sorts an entry with no createdAt to the front (parity with descending compare)", () => {
    const items = [job("a", "2026-07-01T01:00:00Z")];
    const optimistic = { id: "x", status: "queued" }; // no createdAt
    const next = insertNewest(items, optimistic);
    expect(next[0].id).toBe("x");
    // The legacy expression relies on `createdAt.localeCompare`; guard against a
    // throw by only comparing ids for the well-formed entries.
    expect(next.map((j) => j.id)).toEqual(["x", "a"]);
  });

  it("produces the same order as a full sort over a shuffled batch of inserts", () => {
    const created = Array.from({ length: 50 }, (_, i) =>
      job(`j${i}`, new Date(Date.UTC(2026, 6, 1, 0, i)).toISOString()),
    );
    const shuffled = [...created].sort(() => Math.random() - 0.5);
    let acc = [];
    for (const j of shuffled) {
      acc = insertNewest(acc, j);
    }
    const expected = [...created].sort(sortNewest).map((j) => j.id);
    expect(acc.map((j) => j.id)).toEqual(expected);
  });
});

describe("capTerminalJobs (sc-8860)", () => {
  it("caps retained terminal jobs at MAX_TERMINAL_JOBS, newest kept", () => {
    // 250 terminal jobs, newest-first.
    const terminal = Array.from({ length: 250 }, (_, i) =>
      job(`t${i}`, new Date(Date.UTC(2026, 6, 1, 0, 0, 250 - i)).toISOString(), "completed"),
    ).sort(sortNewest);
    const capped = capTerminalJobs(terminal);
    expect(capped).toHaveLength(MAX_TERMINAL_JOBS);
    // The newest MAX_TERMINAL_JOBS survive (the array was newest-first).
    expect(capped).toEqual(terminal.slice(0, MAX_TERMINAL_JOBS));
  });

  it("never drops an active job even when the terminal tail is over the cap", () => {
    const active = Array.from({ length: 10 }, (_, i) =>
      job(`a${i}`, new Date(Date.UTC(2026, 6, 2, 0, i)).toISOString(), "running"),
    );
    const terminal = Array.from({ length: 300 }, (_, i) =>
      job(`t${i}`, new Date(Date.UTC(2026, 6, 1, 0, 0, i)).toISOString(), "failed"),
    );
    const capped = capTerminalJobs([...active, ...terminal].sort(sortNewest));
    const activeIds = new Set(active.map((j) => j.id));
    // Every active job is retained.
    for (const j of active) {
      expect(capped.some((c) => c.id === j.id)).toBe(true);
    }
    // Terminal jobs are capped.
    const keptTerminal = capped.filter((j) => !activeIds.has(j.id));
    expect(keptTerminal).toHaveLength(MAX_TERMINAL_JOBS);
    // Total = all active + capped terminal.
    expect(capped).toHaveLength(active.length + MAX_TERMINAL_JOBS);
  });

  it("returns the same reference (no copy) when under the cap", () => {
    const items = [job("t0", "2026-07-01T00:00:00Z", "completed")];
    expect(capTerminalJobs(items)).toBe(items);
  });
});

describe("upsertJobNewest (sc-8860)", () => {
  it("matches the legacy copy-sort order for the common in-session churn", () => {
    let items = [];
    const feed = [
      job("a", "2026-07-01T01:00:00Z", "queued"),
      job("b", "2026-07-01T02:00:00Z", "queued"),
      job("a", "2026-07-01T01:00:00Z", "running"), // progress on a
      job("c", "2026-07-01T03:00:00Z", "queued"),
      job("b", "2026-07-01T02:00:00Z", "completed"), // b finishes
    ];
    let legacy = [];
    for (const j of feed) {
      items = upsertJobNewest(items, j);
      legacy = legacyUpsert(legacy, j);
    }
    expect(items.map((j) => j.id)).toEqual(legacy.map((j) => j.id));
    expect(items.map((j) => j.id)).toEqual(["c", "b", "a"]);
  });

  it("does NOT re-sort the whole array on a progress tick", () => {
    // Build a large already-ordered list, then push one progress update. If the
    // implementation re-sorted the whole array it would call the comparator O(n log n)
    // times; the ordered-insertion path must not invoke Array.prototype.sort at all.
    const base = Array.from({ length: 500 }, (_, i) =>
      job(`j${i}`, new Date(Date.UTC(2026, 6, 1, 0, 500 - i)).toISOString(), "running"),
    ).sort(sortNewest);
    const sortSpy = vi.spyOn(Array.prototype, "sort");
    const tick = { ...base[250], status: "completed" }; // update a middle job
    const next = upsertJobNewest(base, tick);
    expect(sortSpy).not.toHaveBeenCalled();
    sortSpy.mockRestore();
    // Order still correct.
    expect(next.map((j) => j.id)).toEqual(base.map((j) => j.id));
    expect(next.find((j) => j.id === tick.id).status).toBe("completed");
  });

  it("caps terminal jobs while inserting so the array can't grow unbounded", () => {
    let items = [];
    // Insert MAX_TERMINAL_JOBS + 50 terminal jobs one at a time (oldest first so each
    // insertion lands at the front — the worst case for the old copy-sort).
    for (let i = 0; i < MAX_TERMINAL_JOBS + 50; i += 1) {
      items = upsertJobNewest(
        items,
        job(`t${i}`, new Date(Date.UTC(2026, 6, 1, 0, 0, i)).toISOString(), "completed"),
      );
    }
    expect(items.filter((j) => j.status === "completed")).toHaveLength(MAX_TERMINAL_JOBS);
    expect(items.length).toBeLessThanOrEqual(MAX_TERMINAL_JOBS);
  });
});
