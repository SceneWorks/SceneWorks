import { describe, expect, it } from "vitest";

import { buildBundleReport, routeFamilies } from "./vite-plugin-bundle-report.js";

const roomyBudgets = {
  initial: { rawBytes: 10000, gzipBytes: 10000, brotliBytes: 10000 },
  lazyChunk: { rawBytes: 10000, gzipBytes: 10000, brotliBytes: 10000 },
  largestRoute: { rawBytes: 10000, gzipBytes: 10000, brotliBytes: 10000 },
};

function chunk(
  fileName,
  { code = "export default 1;", imports = [], isEntry = false, isDynamicEntry = false, modules = {} } = {},
) {
  return { type: "chunk", fileName, name: fileName, code, imports, isEntry, isDynamicEntry, modules };
}

function asset(fileName, source = "window.__THEME__ = true;") {
  return { type: "asset", fileName, source };
}

function completeRouteChunks() {
  return Object.fromEntries(
    Object.entries(routeFamilies).map(([route, [moduleId]], index) => {
      const fileName = `assets/route-${index}.js`;
      return [
        fileName,
        chunk(fileName, {
          code: `export const route = ${JSON.stringify(route)};`,
          imports: ["assets/shared.js"],
          isDynamicEntry: true,
          modules: { [`C:/repo/apps/web${moduleId}`]: {} },
        }),
      ];
    }),
  );
}

describe("bundle report", () => {
  it("separates the entry graph from route static dependencies and records compression", () => {
    const report = buildBundleReport(
      {
        "assets/index.js": chunk("assets/index.js", {
          isEntry: true,
          imports: ["assets/shared.js"],
          modules: { "C:/repo/apps/web/src/main.jsx": {} },
        }),
        "assets/shared.js": chunk("assets/shared.js", {
          modules: { "C:/repo/apps/web/src/shared.js": {} },
        }),
        "theme-init.js": asset("theme-init.js"),
        ...completeRouteChunks(),
      },
      roomyBudgets,
    );

    expect(report.initial.chunks).toEqual([
      "assets/index.js",
      "assets/shared.js",
      "theme-init.js",
    ]);
    expect(report.chunks.find((item) => item.fileName === "theme-init.js")).toMatchObject({
      type: "asset",
      initial: true,
    });
    expect(report.routes.Stats.chunks).toHaveLength(1);
    expect(report.routes.Stats.chunks[0]).toMatch(/^assets\/route-/);
    expect(report.initial.gzipBytes).toBeGreaterThan(0);
    expect(report.initial.brotliBytes).toBeGreaterThan(0);
    expect(report.violations).toEqual([]);
  });

  it("fails when a lazy route leaks into the initial graph or a budget regresses", () => {
    const routeChunks = completeRouteChunks();
    const statsFile = Object.keys(routeChunks).find((fileName) =>
      Object.keys(routeChunks[fileName].modules)[0].endsWith("/src/screens/StatsScreen.jsx"),
    );
    const report = buildBundleReport(
      {
        "assets/index.js": chunk("assets/index.js", {
          isEntry: true,
          imports: [statsFile],
          modules: { "C:/repo/apps/web/src/main.jsx": {} },
          code: "x".repeat(2000),
        }),
        "assets/shared.js": chunk("assets/shared.js"),
        ...routeChunks,
      },
      {
        ...roomyBudgets,
        initial: { rawBytes: 1, gzipBytes: 1, brotliBytes: 1 },
      },
    );

    expect(
      report.violations.some((message) =>
        message.startsWith("Stats/Recharts leaked into the initial Assets graph"),
      ),
    ).toBe(true);
    expect(report.violations.some((message) => message.includes("Initial JavaScript"))).toBe(true);
  });

  // sc-24217 policy: initial and each lazy chunk are gates; aggregate total is telemetry.
  function entryWith(extra = {}) {
    return {
      "assets/index.js": chunk("assets/index.js", {
        isEntry: true,
        modules: { "C:/repo/apps/web/src/main.jsx": {} },
      }),
      "assets/shared.js": chunk("assets/shared.js"),
      ...completeRouteChunks(),
      ...extra,
    };
  }

  it("reports total JavaScript as telemetry but never fails the build on it", () => {
    const report = buildBundleReport(entryWith(), {
      ...roomyBudgets,
      total: { rawBytes: 1, gzipBytes: 1, brotliBytes: 1 },
    });

    expect(report.total.gzipBytes).toBeGreaterThan(1);
    expect(report.total.chunks).toContain("assets/shared.js");
    expect(report.violations).toEqual([]);
  });

  it("fails when one lazy chunk plus its non-initial static closure exceeds lazyChunk", () => {
    const report = buildBundleReport(
      entryWith({
        "assets/heavy.js": chunk("assets/heavy.js", {
          isDynamicEntry: true,
          imports: ["assets/heavy-dep.js", "assets/index.js"],
          code: "a".repeat(300),
        }),
        "assets/heavy-dep.js": chunk("assets/heavy-dep.js", { code: "b".repeat(300) }),
      }),
      { ...roomyBudgets, lazyChunk: { rawBytes: 500, gzipBytes: 10000, brotliBytes: 10000 } },
    );

    expect(report.lazyChunks["assets/heavy.js"]).toMatchObject({
      chunks: ["assets/heavy-dep.js", "assets/heavy.js"],
      rawBytes: 600,
    });
    expect(report.violations).toEqual([
      "Lazy chunk (assets/heavy.js) rawBytes is 600 bytes (budget 500 bytes)",
    ]);
  });

  it("does not count an initial chunk as a lazy chunk", () => {
    const report = buildBundleReport(
      entryWith({
        "assets/index.js": chunk("assets/index.js", {
          isEntry: true,
          imports: ["assets/eager.js"],
          modules: { "C:/repo/apps/web/src/main.jsx": {} },
        }),
        "assets/eager.js": chunk("assets/eager.js", { isDynamicEntry: true, code: "c".repeat(600) }),
      }),
      { ...roomyBudgets, lazyChunk: { rawBytes: 500, gzipBytes: 10000, brotliBytes: 10000 } },
    );

    expect(report.lazyChunks["assets/eager.js"]).toBeUndefined();
    expect(report.violations).toEqual([]);
  });

  it("fails when the initial graph exceeds its budget", () => {
    const report = buildBundleReport(entryWith(), {
      ...roomyBudgets,
      initial: { rawBytes: 10000, gzipBytes: 10000, brotliBytes: 1 },
    });

    expect(report.violations).toEqual([
      expect.stringMatching(/^Initial JavaScript brotliBytes is \d+ bytes \(budget 1 bytes\)$/),
    ]);
  });

  it("fails closed when the lazyChunk budget is missing", () => {
    const { lazyChunk: _omitted, ...withoutLazy } = roomyBudgets;
    const report = buildBundleReport(entryWith(), withoutLazy);

    expect(report.violations).toEqual(["bundle-budgets.json is missing the lazyChunk budget"]);
  });

  it.each(["initial", "largestRoute"])("fails closed when the %s budget is missing", (name) => {
    const { [name]: _omitted, ...without } = roomyBudgets;
    const report = buildBundleReport(entryWith(), without);

    expect(report.violations).toEqual([`bundle-budgets.json is missing the ${name} budget`]);
  });

  it.each(["/src/modelCatalog.js", "/src/data/previewSupport.json"])(
    "fails when the offline model catalog (%s) lands in the initial graph",
    (moduleSuffix) => {
      const report = buildBundleReport(
        entryWith({
          "assets/index.js": chunk("assets/index.js", {
            isEntry: true,
            modules: {
              "C:/repo/apps/web/src/main.jsx": {},
              [`C:/repo/apps/web${moduleSuffix}`]: {},
            },
          }),
        }),
        roomyBudgets,
      );

      expect(report.violations).toEqual([
        `Offline model catalog leaked into the initial Assets graph via C:/repo/apps/web${moduleSuffix}`,
      ]);
    },
  );
});
