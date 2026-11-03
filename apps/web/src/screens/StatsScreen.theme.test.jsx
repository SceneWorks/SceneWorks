import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";

vi.mock("../hooks/useGenerationMetrics.js", () => ({
  useGenerationMetrics: () => ({ rows: [{ jobId: "a", type: "image_generate", status: "completed", createdAt: "2026-07-01T10:00:00Z", metrics: { model: "qwen", quantLabel: "q8", imageCount: 1, totalMs: 9400, loadMs: 2100, sampleMs: 6400, decodeMs: 900 } }], loading: false, error: "", refresh: () => {} }),
}));
vi.mock("recharts", () => {
  const wrap = ({ children }) => <div>{children}</div>;
  const empty = () => null;
  return { Bar: empty, BarChart: wrap, CartesianGrid: ({ stroke }) => <i data-grid={stroke} />, Legend: empty,
    ResponsiveContainer: wrap, Scatter: empty, ScatterChart: wrap, Tooltip: empty, XAxis: empty, YAxis: empty };
});
const { StatsScreen } = await import("./StatsScreen.jsx");
let container, root;
afterEach(() => { act(() => root?.unmount()); container?.remove(); vi.unstubAllGlobals(); });
it("uses the app theme even when the operating system prefers the opposite", () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const matchMedia = vi.fn(() => ({ matches: true }));
  vi.stubGlobal("matchMedia", matchMedia);
  container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container);
  const render = (theme) => act(() => root.render(<AppContext.Provider value={{ token: "", theme }}><StatsScreen /></AppContext.Provider>));
  render("light");
  expect(container.querySelector('[data-grid]')?.dataset.grid).toBe("#e1e0d9");
  render("dark");
  expect(container.querySelector('[data-grid]')?.dataset.grid).toBe("#2c2c2a");
  expect(matchMedia).not.toHaveBeenCalled();
});
