import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AppContext } from "../context/AppContext.js";

// Deterministic metrics feed so the render needs no network.
vi.mock("../hooks/useGenerationMetrics.js", () => ({
  useGenerationMetrics: () => ({
    rows: [
      {
        jobId: "job-a",
        type: "image_generate",
        status: "completed",
        createdAt: "2026-07-01T10:00:00Z",
        metrics: {
          model: "qwen_image",
          quantLabel: "q8",
          sampler: "default",
          scheduler: "default",
          guidanceScale: 4.0,
          steps: 20,
          totalMs: 9400,
          loadMs: 2100,
          sampleMs: 6400,
          decodeMs: 900,
          peakMemoryPct: 71,
          peakMemoryBytes: 12_884_901_888,
          peakGpuLoadPct: 88,
          backend: "mlx",
          width: 1024,
          height: 1024,
          seed: 42,
        },
      },
    ],
    loading: false,
    error: "",
    refresh: () => {},
  }),
}));

// Imported after the mock so StatsScreen picks up the mocked hook.
const { StatsScreen } = await import("./StatsScreen.jsx");

let container;
let root;

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
});

function render() {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => {
    root.render(
      <AppContext.Provider value={{ token: "t" }}>
        <StatsScreen />
      </AppContext.Provider>,
    );
  });
}

describe("StatsScreen", () => {
  it("renders a row per metrics record with model, quant, and formatted timing", () => {
    render();
    const text = container.textContent;
    expect(text).toContain("qwen_image");
    expect(text).toContain("q8");
    expect(text).toContain("9.4s"); // totalMs formatted
    expect(text).toContain("Runs"); // KPI label
    expect(container.querySelectorAll(".stats-table tbody tr").length).toBe(1);
  });

  it("opens the run-detail panel on row click", () => {
    render();
    const row = container.querySelector(".stats-table tbody tr");
    act(() => {
      row.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(container.querySelector(".stats-detail")).not.toBeNull();
    expect(container.textContent).toContain("Run detail");
    expect(container.textContent).toContain("mlx"); // backend surfaced in detail
  });
});
