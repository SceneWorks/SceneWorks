import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppContext } from "../context/AppContext.js";

// Mutable feed so individual tests can swap in their own rows before rendering.
// vi.hoisted runs before the (hoisted) vi.mock factory, so the mock can close
// over it.
const feed = vi.hoisted(() => ({ rows: [] }));

function makeRow(i) {
  return {
    jobId: `job-${i}`,
    type: "image_generate",
    status: "completed",
    createdAt: `2026-07-01T10:00:00Z`,
    metrics: {
      model: "qwen_image",
      quantLabel: "q8",
      sampler: "default",
      scheduler: "default",
      guidanceScale: 4.0,
      steps: 20,
      imageCount: 2,
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
  };
}

const SINGLE_ROW = makeRow("a");

// Deterministic metrics feed so the render needs no network.
vi.mock("../hooks/useGenerationMetrics.js", () => ({
  useGenerationMetrics: () => ({
    rows: feed.rows,
    loading: false,
    error: "",
    refresh: () => {},
  }),
}));

// recharts' ResponsiveContainer observes size via ResizeObserver, which jsdom
// doesn't implement — stub it so the charts mount in the test env.
globalThis.ResizeObserver ||= class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Imported after the mock so StatsScreen picks up the mocked hook.
const { StatsScreen } = await import("./StatsScreen.jsx");

let container;
let root;

beforeEach(() => {
  feed.rows = [SINGLE_ROW];
});

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

function clickRow(index = 0) {
  const row = container.querySelectorAll(".stats-table tbody tr:not(.stats-detail-row)")[index];
  act(() => {
    row.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  return row;
}

describe("StatsScreen", () => {
  it("renders a row per metrics record with model, quant, and formatted timing", () => {
    render();
    const text = container.textContent;
    expect(text).toContain("qwen_image");
    expect(text).toContain("q8");
    expect(text).toContain("9.4s"); // totalMs formatted
    expect(text).toContain("Runs"); // KPI label
    expect(
      container.querySelectorAll(".stats-table tbody tr:not(.stats-detail-row)").length,
    ).toBe(1);
  });

  it("opens the run-detail panel on row click", () => {
    render();
    clickRow();
    expect(container.querySelector(".stats-detail")).not.toBeNull();
    expect(container.textContent).toContain("Run detail");
    expect(container.textContent).toContain("mlx"); // backend surfaced in detail
    expect(container.textContent).toContain("Per image"); // per-image breakdown (sc-10426)
  });

  it("expands the detail inline as the row directly after the selected row", () => {
    render();
    const row = clickRow();
    const next = row.nextElementSibling;
    expect(next).not.toBeNull();
    expect(next.classList.contains("stats-detail-row")).toBe(true);
    expect(next.querySelector(".stats-detail")).not.toBeNull();
    // The detail cell spans the full column count.
    const headerCols = container.querySelectorAll(".stats-table thead th").length;
    expect(Number(next.querySelector("td").getAttribute("colspan"))).toBe(headerCols);
  });

  it("collapses the detail when the selected row is clicked again", () => {
    render();
    clickRow();
    expect(container.querySelector(".stats-detail-row")).not.toBeNull();
    clickRow(); // toggle off
    expect(container.querySelector(".stats-detail-row")).toBeNull();
  });

  it("pages the list by 100 rows with working next/previous controls", () => {
    feed.rows = Array.from({ length: 150 }, (_, i) => makeRow(i));
    render();

    const dataRows = () =>
      container.querySelectorAll(".stats-table tbody tr:not(.stats-detail-row)").length;

    // Page 1: first 100 rows, range summary, Next enabled / Previous disabled.
    expect(dataRows()).toBe(100);
    expect(container.textContent).toContain("1–100 of 150");
    const [prev, next] = container.querySelectorAll(".stats-page-btn");
    expect(prev.disabled).toBe(true);
    expect(next.disabled).toBe(false);

    // Advance to page 2: remaining 50 rows, Next now disabled.
    act(() => {
      next.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(dataRows()).toBe(50);
    expect(container.textContent).toContain("101–150 of 150");
    const [prev2, next2] = container.querySelectorAll(".stats-page-btn");
    expect(prev2.disabled).toBe(false);
    expect(next2.disabled).toBe(true);
  });

  it("hides pagination controls when a single page suffices", () => {
    render();
    expect(container.querySelector(".stats-pagination")).toBeNull();
  });
});
