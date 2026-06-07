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
        "mlx_route_decision decision=fell_back_to_torch reason=no_idle_mlx_worker model=qwen_image_edit_2511_lightning",
        { event: "mlx_route_decision", decision: "fell_back_to_torch", reason: "no_idle_mlx_worker" },
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
    expect(container.textContent).toContain("fell_back_to_torch");
    expect(container.textContent).toContain("image_inference_failed");
  });

  it("highlights routing-decision events", async () => {
    await render();
    const highlighted = container.querySelector(".logs-row.highlighted");
    expect(highlighted).toBeTruthy();
    expect(highlighted.textContent).toContain("fell_back_to_torch");
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
});
