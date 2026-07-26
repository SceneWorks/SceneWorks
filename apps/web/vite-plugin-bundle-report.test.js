import { describe, expect, it } from "vitest";

import { buildBundleReport, routeFamilies } from "./vite-plugin-bundle-report.js";

const roomyBudgets = {
  initial: { rawBytes: 10000, gzipBytes: 10000, brotliBytes: 10000 },
  total: { rawBytes: 10000, gzipBytes: 10000, brotliBytes: 10000 },
  largestRoute: { rawBytes: 10000, gzipBytes: 10000, brotliBytes: 10000 },
};

function chunk(fileName, { code = "export default 1;", imports = [], isEntry = false, modules = {} } = {}) {
  return { type: "chunk", fileName, name: fileName, code, imports, isEntry, modules };
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
});
