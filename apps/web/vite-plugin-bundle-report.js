import { brotliCompressSync, constants, gzipSync } from "node:zlib";

export const routeFamilies = Object.freeze({
  "Image Studio": ["/src/screens/ImageStudio.jsx"],
  "Video Studio": ["/src/screens/VideoStudio.jsx"],
  "Audio Studio": ["/src/screens/AudioStudio.jsx"],
  Characters: ["/src/screens/CharacterStudio.jsx"],
  "Document Studio": ["/src/screens/DocumentStudio.jsx"],
  Training: ["/src/screens/TrainingStudio.jsx"],
  Presets: ["/src/screens/PresetManagerScreen.jsx"],
  "Pose Library": ["/src/screens/PoseLibraryScreen.jsx"],
  "Key Point Library": ["/src/screens/KeyPointLibraryScreen.jsx"],
  "Video Editor": ["/src/screens/EditorScreen.jsx"],
  "Image Editor": ["/src/screens/ImageEditor.jsx"],
  Queue: ["/src/screens/QueueScreen.jsx"],
  Models: ["/src/screens/ModelManagerScreen.jsx"],
  Settings: ["/src/screens/SettingsScreen.jsx"],
  Stats: ["/src/screens/StatsScreen.jsx"],
  Logs: ["/src/screens/LogsScreen.jsx"],
  Licenses: ["/src/screens/LicensesScreen.jsx"],
  "Setup Wizard": ["/src/screens/SetupWizard.jsx"],
  "Simple UI": ["/src/simple/SimpleShell.jsx"],
});

export const initialExclusions = Object.freeze({
  "Stats/Recharts": [
    "/src/screens/StatsScreen.jsx",
    "/node_modules/recharts/",
    "/node_modules/recharts-scale/",
    "/node_modules/victory-vendor/",
    "/node_modules/d3-",
  ],
  "Image Editor/Konva": [
    "/src/screens/ImageEditor.jsx",
    "/node_modules/konva/",
    "/node_modules/react-konva/",
    "/node_modules/react-reconciler/",
  ],
  "Licenses/license corpus": [
    "/src/screens/LicensesScreen.jsx",
    "/src/data/bundledLicenses.js",
    "/apps/desktop/licenses/",
  ],
  Models: ["/src/screens/ModelManagerScreen.jsx"],
  "Inactive studio families": [
    "/src/screens/ImageStudio.jsx",
    "/src/screens/VideoStudio.jsx",
    "/src/screens/AudioStudio.jsx",
    "/src/screens/CharacterStudio.jsx",
    "/src/screens/DocumentStudio.jsx",
    "/src/screens/TrainingStudio.jsx",
    "/src/screens/PresetManagerScreen.jsx",
    "/src/screens/PoseLibraryScreen.jsx",
    "/src/screens/KeyPointLibraryScreen.jsx",
    "/src/screens/EditorScreen.jsx",
  ],
  "Simple UI": ["/src/simple/SimpleShell.jsx", "/src/simple/studioParts.jsx"],
  "Setup Wizard": ["/src/screens/SetupWizard.jsx"],
});

function normalize(value) {
  return String(value).replaceAll("\\", "/");
}

function compressedSizes(code) {
  const source = Buffer.from(code);
  return {
    rawBytes: source.byteLength,
    gzipBytes: gzipSync(source, { level: 9 }).byteLength,
    brotliBytes: brotliCompressSync(source, {
      params: { [constants.BROTLI_PARAM_QUALITY]: 11 },
    }).byteLength,
  };
}

function assetSource(asset) {
  return typeof asset.source === "string" ? asset.source : Buffer.from(asset.source);
}

function collectStaticChunks(startNames, chunks) {
  const collected = new Set();
  const pending = [...startNames];
  while (pending.length) {
    const fileName = pending.pop();
    if (collected.has(fileName) || !chunks.has(fileName)) {
      continue;
    }
    collected.add(fileName);
    pending.push(...chunks.get(fileName).imports);
  }
  return collected;
}

function summarize(fileNames, chunkSizes) {
  const result = { rawBytes: 0, gzipBytes: 0, brotliBytes: 0 };
  for (const fileName of fileNames) {
    const size = chunkSizes.get(fileName);
    if (!size) continue;
    result.rawBytes += size.rawBytes;
    result.gzipBytes += size.gzipBytes;
    result.brotliBytes += size.brotliBytes;
  }
  return result;
}

function budgetViolations(label, actual, budget) {
  return Object.entries(budget)
    .filter(([metric, limit]) => actual[metric] > limit)
    .map(
      ([metric, limit]) =>
        `${label} ${metric} is ${actual[metric]} bytes (budget ${limit} bytes)`,
    );
}

export function buildBundleReport(bundle, budgets) {
  const chunks = new Map(
    Object.values(bundle)
      .filter((item) => item.type === "chunk")
      .map((chunk) => [chunk.fileName, chunk]),
  );
  // Vite's pre-paint theme bootstrap is emitted as a JavaScript asset rather
  // than a Rollup chunk. It is nevertheless fetched by index.html on first
  // paint, so include emitted *.js assets in the initial and total JS budgets.
  const scriptAssets = new Map(
    Object.values(bundle)
      .filter((item) => item.type === "asset" && item.fileName.endsWith(".js"))
      .map((asset) => [asset.fileName, asset]),
  );
  const chunkSizes = new Map(
    [
      ...[...chunks].map(([fileName, chunk]) => [
        fileName,
        compressedSizes(chunk.code),
      ]),
      ...[...scriptAssets].map(([fileName, asset]) => [
        fileName,
        compressedSizes(assetSource(asset)),
      ]),
    ],
  );
  const initialNames = new Set([
    ...collectStaticChunks(
      [...chunks.values()].filter((chunk) => chunk.isEntry).map((chunk) => chunk.fileName),
      chunks,
    ),
    ...scriptAssets.keys(),
  ]);

  const reportNames = new Set([
    ...chunks.keys(),
    ...scriptAssets.keys(),
  ]);
  const routeMembership = new Map(
    [...reportNames].map((fileName) => [fileName, []]),
  );

  const routeChunks = {};
  const violations = [];
  const initialModuleIds = [...initialNames].flatMap((fileName) =>
    Object.keys(chunks.get(fileName)?.modules ?? {}).map(normalize),
  );

  for (const [domain, excludedFragments] of Object.entries(initialExclusions)) {
    const leaked = initialModuleIds.find((moduleId) =>
      excludedFragments.some((fragment) => moduleId.includes(fragment)),
    );
    if (leaked) {
      violations.push(`${domain} leaked into the initial Assets graph via ${leaked}`);
    }
  }

  for (const [route, moduleSuffixes] of Object.entries(routeFamilies)) {
    const starts = [...chunks.values()]
      .filter((chunk) =>
        Object.keys(chunk.modules).some((moduleId) =>
          moduleSuffixes.some((suffix) => normalize(moduleId).endsWith(suffix)),
        ),
      )
      .map((chunk) => chunk.fileName);
    if (!starts.length) {
      violations.push(`${route} did not produce a route chunk`);
      continue;
    }
    const loaded = collectStaticChunks(starts, chunks);
    for (const initialName of initialNames) loaded.delete(initialName);
    for (const fileName of loaded) routeMembership.get(fileName)?.push(route);
    routeChunks[route] = {
      entryChunks: starts.sort(),
      chunks: [...loaded].sort(),
      ...summarize(loaded, chunkSizes),
    };
  }

  const allNames = reportNames;
  const initial = {
    chunks: [...initialNames].sort(),
    ...summarize(initialNames, chunkSizes),
  };
  const total = {
    chunks: [...allNames].sort(),
    ...summarize(allNames, chunkSizes),
  };
  const largestRoute = Object.entries(routeChunks).reduce(
    (largest, [route, values]) =>
      values.gzipBytes > largest.gzipBytes ? { route, ...values } : largest,
    { route: null, rawBytes: 0, gzipBytes: 0, brotliBytes: 0, chunks: [] },
  );

  violations.push(...budgetViolations("Initial JavaScript", initial, budgets.initial));
  violations.push(...budgetViolations("Total JavaScript", total, budgets.total));
  violations.push(
    ...budgetViolations(
      `Largest route (${largestRoute.route ?? "none"})`,
      largestRoute,
      budgets.largestRoute,
    ),
  );

  return {
    schemaVersion: 1,
    initial,
    routes: routeChunks,
    total,
    largestRoute,
    chunks: [...allNames]
      .map((fileName) => {
        const chunk = chunks.get(fileName);
        return {
          fileName,
          name: chunk?.name ?? fileName,
          type: chunk ? "chunk" : "asset",
          entry: Boolean(chunk?.isEntry),
          initial: initialNames.has(fileName),
          routes: routeMembership.get(fileName).sort(),
          ...chunkSizes.get(fileName),
        };
      })
      .sort((left, right) => left.fileName.localeCompare(right.fileName)),
    budgets,
    violations,
  };
}

export default function bundleReportPlugin({ budgets }) {
  let violations = [];

  return {
    name: "sceneworks-bundle-report",
    generateBundle(_options, bundle) {
      const report = buildBundleReport(bundle, budgets);
      violations = report.violations;
      this.emitFile({
        type: "asset",
        fileName: "bundle-report.json",
        source: `${JSON.stringify(report, null, 2)}\n`,
      });
    },
    writeBundle() {
      if (violations.length) {
        this.error(`Bundle budget failed:\n- ${violations.join("\n- ")}`);
      }
    },
  };
}
