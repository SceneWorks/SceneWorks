import { afterEach, describe, expect, it, vi } from "vitest";
import {
  STARTUP_MARK_ORDER,
  beginAssetRequest,
  recordStartupMark,
  resetStartupTimingForTests,
  settleAssetRequest,
} from "./startupTiming.js";

function fakePerformance() {
  let now = 10;
  return {
    mark: vi.fn(),
    measure: vi.fn(),
    clearMarks: vi.fn(),
    clearMeasures: vi.fn(),
    now: vi.fn(() => ++now),
  };
}

const originalPerformance = globalThis.performance;

afterEach(() => {
  resetStartupTimingForTests();
  Object.defineProperty(globalThis, "performance", {
    configurable: true,
    value: originalPerformance,
  });
});

describe("startup timing", () => {
  it("records the expected readiness order without secret-bearing detail", () => {
    const performance = fakePerformance();
    Object.defineProperty(globalThis, "performance", { configurable: true, value: performance });

    for (const mark of STARTUP_MARK_ORDER.slice(0, 5)) {
      expect(recordStartupMark(mark)).toBe(true);
    }
    const requestStarted = beginAssetRequest();
    settleAssetRequest(requestStarted);
    expect(recordStartupMark("assets-ready-render")).toBe(true);

    expect(performance.mark.mock.calls.map(([name]) => name)).toEqual(
      STARTUP_MARK_ORDER.map((name) => `sceneworks.${name}`),
    );
    expect(performance.mark.mock.calls.flat().join(" ")).not.toMatch(
      /token|ticket|project-\d|prompt|query/i,
    );
  });

  it("keeps one mark and one measure per name when asset refreshes repeat", () => {
    const performance = fakePerformance();
    Object.defineProperty(globalThis, "performance", { configurable: true, value: performance });

    for (let index = 0; index < 20; index += 1) {
      const started = beginAssetRequest();
      settleAssetRequest(started);
    }

    expect(performance.clearMarks).toHaveBeenCalledTimes(40);
    expect(performance.clearMeasures).toHaveBeenCalledTimes(20);
    expect(performance.measure).toHaveBeenCalledTimes(20);
    expect(new Set(performance.measure.mock.calls.map(([name]) => name))).toEqual(
      new Set(["sceneworks.asset-request"]),
    );
  });

  it("rejects lifecycle marks that would make readiness order move backwards", () => {
    const performance = fakePerformance();
    Object.defineProperty(globalThis, "performance", { configurable: true, value: performance });

    expect(recordStartupMark("access-resolved")).toBe(true);
    expect(recordStartupMark("projects-committed")).toBe(true);
    expect(recordStartupMark("media-authorization-settled")).toBe(false);
    expect(performance.mark.mock.calls.map(([name]) => name)).toEqual([
      "sceneworks.access-resolved",
      "sceneworks.projects-committed",
    ]);
  });

  it("is inert when the Performance Timeline API is unavailable", () => {
    Object.defineProperty(globalThis, "performance", { configurable: true, value: undefined });

    expect(() => {
      recordStartupMark("bootstrap-start");
      settleAssetRequest(beginAssetRequest());
    }).not.toThrow();
  });
});
