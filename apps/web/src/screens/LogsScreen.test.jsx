import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Web path: LogsScreen reads GET /api/v1/logs via apiFetch (no window.__TAURI__ in
// jsdom). The mock honours the `source` query param so we can assert the screen
// requests server-side filtering; the snapshot vs afterSeq distinction keeps the
// 2s poll from duplicating rows.
let logRows = [];
vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async (path) => {
      if (path.includes("afterSeq=")) return [];
      if (path.includes("source=mlx-worker")) {
        return logRows.filter((row) => row.source === "mlx-worker");
      }
      return logRows;
    }),
  };
});

import { apiFetch } from "../api.js";
import { LogsScreen } from "./LogsScreen.jsx";
import { AppContext } from "../context/AppContext.js";

function row(seq, source, level, message, event) {
  return { seq, source, level, timestamp: "2026-06-07T01:02:0" + seq + "Z", message, event, raw: message };
}

describe("LogsScreen", () => {
  let container;
  let root;

  beforeEach(() => {
    logRows = [
      row(
        0,
        "api",
        "info",
        "gpu_route_decision decision=claimed_by_candle reason=candle_worker model=ltx_2_3",
        { event: "gpu_route_decision", decision: "claimed_by_candle", reason: "candle_worker" },
      ),
      row(1, "worker", "error", "image_inference_failed error=boom", { event: "image_inference_failed", error: "boom" }),
      row(2, "mlx-worker", "info", "claimed jobId=j1", { event: "claimed", jobId: "j1" }),
    ];
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  // Controlled <input>: set the value through the native setter React tracks,
  // then dispatch the input event so React's synthetic onChange fires.
  async function typeSearch(input, value) {
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      "value",
    ).set;
    await act(async () => {
      setter.call(input, value);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
  }

  async function render() {
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ token: "test-token" }}>
          <LogsScreen />
        </AppContext.Provider>,
      );
    });
    await act(async () => {}); // flush the load effect
  }

  it("renders captured log entries with their messages", async () => {
    await render();
    expect(container.querySelectorAll(".logs-row").length).toBe(3);
    expect(container.textContent).toContain("claimed_by_candle");
    expect(container.textContent).toContain("image_inference_failed");
  });

  it("highlights routing-decision events", async () => {
    await render();
    const highlighted = container.querySelector(".logs-row.highlighted");
    expect(highlighted).toBeTruthy();
    expect(highlighted.textContent).toContain("claimed_by_candle");
  });

  it("requests server-side filtering when a source is selected", async () => {
    await render();
    const btn = [...container.querySelectorAll(".segmented-control button")].find(
      (b) => b.textContent === "mlx-worker",
    );
    await act(async () => {
      btn.click();
    });
    await act(async () => {});
    const paths = apiFetch.mock.calls.map((call) => call[0]);
    expect(paths.some((path) => path.includes("source=mlx-worker"))).toBe(true);
    // After the filtered refetch only the mlx-worker row remains.
    expect(container.querySelectorAll(".logs-row").length).toBe(1);
    expect(container.textContent).toContain("claimed");
  });

  it("shows an empty state when there are no entries", async () => {
    logRows = [];
    await render();
    expect(container.textContent).toContain("No log entries yet");
  });

  it("filters client-side without issuing a fetch per keystroke (sc-8849)", async () => {
    vi.useFakeTimers();
    try {
      await render();
      const callsAfterLoad = apiFetch.mock.calls.length;
      const input = container.querySelector("input.logs-search");

      // Type the term one character at a time.
      const term = "boom";
      for (let i = 1; i <= term.length; i += 1) {
        await typeSearch(input, term.slice(0, i));
      }

      // No keystroke should have triggered a fetch (search is client-side only).
      expect(apiFetch.mock.calls.length).toBe(callsAfterLoad);

      // After the debounce window elapses the filter applies once.
      await act(async () => {
        vi.advanceTimersByTime(300);
      });
      expect(apiFetch.mock.calls.length).toBe(callsAfterLoad);
      // Only the row whose message contains "boom" survives.
      const rows = container.querySelectorAll(".logs-row");
      expect(rows.length).toBe(1);
      expect(rows[0].textContent).toContain("image_inference_failed");
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not re-arm the 2s poll on every keystroke (sc-8849)", async () => {
    vi.useFakeTimers();
    const setInterval = vi.spyOn(globalThis, "setInterval");
    try {
      await render();
      const intervalsAfterLoad = setInterval.mock.calls.length;
      const input = container.querySelector("input.logs-search");

      for (const value of ["c", "ca", "can", "cand"]) {
        await typeSearch(input, value);
      }
      await act(async () => {
        vi.advanceTimersByTime(300);
      });

      // The poll interval must not be recreated as the search term changes.
      expect(setInterval.mock.calls.length).toBe(intervalsAfterLoad);
    } finally {
      setInterval.mockRestore();
      vi.useRealTimers();
    }
  });

  it("clears the search filter and restores all rows (sc-8849)", async () => {
    vi.useFakeTimers();
    try {
      await render();
      const input = container.querySelector("input.logs-search");

      await typeSearch(input, "boom");
      await act(async () => {
        vi.advanceTimersByTime(300);
      });
      expect(container.querySelectorAll(".logs-row").length).toBe(1);

      await typeSearch(input, "");
      await act(async () => {
        vi.advanceTimersByTime(300);
      });
      expect(container.querySelectorAll(".logs-row").length).toBe(3);
    } finally {
      vi.useRealTimers();
    }
  });
});
