import { afterEach, describe, expect, it, vi } from "vitest";

import { renderThemeInit } from "./accentIds.js";
import accentsSource from "./accents.js?raw";
import indexSource from "../index.html?raw";
import mainSource from "./main.jsx?raw";
import startupTimingSource from "./startupTiming.js?raw";
import templateSource from "./theme-init.template.js?raw";

afterEach(() => {
  delete globalThis.__SCENEWORKS_BOOTSTRAP_TIMING_STARTED__;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("startup timing entry", () => {
  it("marks bootstrap in the blocking pre-paint entry before the ESM graph", () => {
    const prePaintEntry = indexSource.indexOf('<script src="/theme-init.js"></script>');
    const moduleEntry = indexSource.indexOf(
      '<script type="module" src="/src/main.jsx"></script>',
    );
    const bootstrapMark = templateSource.indexOf(
      'timing.mark("sceneworks.bootstrap-start")',
    );
    const firstAppRead = templateSource.indexOf("window.localStorage");

    expect(prePaintEntry).toBeGreaterThan(-1);
    expect(moduleEntry).toBeGreaterThan(prePaintEntry);
    expect(bootstrapMark).toBeGreaterThan(-1);
    expect(firstAppRead).toBeGreaterThan(bootstrapMark);
    expect(mainSource).not.toContain('recordStartupMark("bootstrap-start")');
    expect(startupTimingSource).toContain(
      "globalThis.__SCENEWORKS_BOOTSTRAP_TIMING_STARTED__ === true",
    );
  });

  it("executes the generated bootstrap mark before application storage reads", () => {
    const events = [];
    vi.stubGlobal("performance", {
      clearMarks: vi.fn(),
      mark: vi.fn((name) => events.push(`mark:${name}`)),
    });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation((key) => {
      events.push(`storage:${key}`);
      return null;
    });

    const generated = renderThemeInit(templateSource, accentsSource);
    Function(generated)();

    expect(events[0]).toBe("mark:sceneworks.bootstrap-start");
    expect(events.some((event) => event.startsWith("storage:"))).toBe(true);
    expect(globalThis.__SCENEWORKS_BOOTSTRAP_TIMING_STARTED__).toBe(true);
  });
});
