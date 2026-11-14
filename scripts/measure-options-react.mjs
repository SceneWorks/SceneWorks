// Measure the actual Image Studio React input path with the controlled API fixture.
// Build with `npm run build -- --config vite.verify.config.mjs`, then start
// `npm exec -- vite preview --config vite.verify.config.mjs --port 5317` in apps/web.
// The production bundle and controlled API are both served from that origin.
// node scripts/measure-options-react.mjs --origin http://127.0.0.1:5317 --samples 30
// For a deployed proxy, keep the fixture preview running locally and add
// `--fixture-origin http://127.0.0.1:5317`. Assets still load from --origin.
import { chromium } from "playwright";

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, all) => {
  if (value.startsWith("--") && all[index + 1] && !all[index + 1].startsWith("--")) {
    pairs.push([value.slice(2), all[index + 1]]);
  }
  return pairs;
}, []));
const origin = args.origin && new URL(args.origin).origin;
const fixtureOrigin = args["fixture-origin"] && new URL(args["fixture-origin"]).origin;
const samples = args.samples === undefined ? 30 : Number(args.samples);
if (!origin || !Number.isInteger(samples) || samples < 5 || samples > 50) {
  throw new Error("Pass --origin <controlled fixture origin> and --samples 5..50");
}
const reportResponse = await fetch(`${origin}/bundle-report.json`, { signal: AbortSignal.timeout(10000) });
if (!reportResponse.ok) throw new Error(`Bundle report unavailable (${reportResponse.status})`);
const bundleReport = await reportResponse.json();
const reportedRoute = bundleReport.routes?.["Image Studio"];
const reportedInteractiveRoute = bundleReport.interactiveRoutes?.["Image Studio"];
if (!reportedRoute || !reportedInteractiveRoute) throw new Error("Image Studio route closures are missing from bundle report");
const bundleChunks = new Map(bundleReport.chunks.map((chunk) => [chunk.fileName, chunk]));
const fetchedChunks = new Set();

function p95(values) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(ordered.length * 0.95) - 1];
}

const browser = await chromium.launch({
  headless: true,
  ...(process.env.PLAYWRIGHT_CHROME ? { executablePath: process.env.PLAYWRIGHT_CHROME } : {}),
});
try {
  const context = await browser.newContext({ serviceWorkers: "block" });
  try {
    const page = await context.newPage();
    page.on("response", (response) => {
      const fileName = new URL(response.url()).pathname.slice(1);
      if (bundleChunks.has(fileName)) fetchedChunks.add(fileName);
    });
    if (fixtureOrigin) {
      await page.route("**/api/**", async (route) => {
        const request = route.request();
        const url = new URL(request.url());
        if (request.method() === "OPTIONS") {
          await route.fulfill({ status: 204, headers: {
            "access-control-allow-origin": origin,
            "access-control-allow-credentials": "true",
            "access-control-allow-methods": "GET, POST, PUT, DELETE, OPTIONS",
            "access-control-allow-headers": request.headers()["access-control-request-headers"] ?? "authorization, content-type",
          } });
          return;
        }
        if (url.pathname === "/api/v1/jobs/events") {
          await route.fulfill({ status: 200, contentType: "text/event-stream", body: "" });
          return;
        }
        const response = await fetch(`${fixtureOrigin}${url.pathname}${url.search}`, {
          method: request.method(),
          headers: request.headers()["content-type"]
            ? { "content-type": request.headers()["content-type"] } : {},
          ...(request.method() === "GET" || request.method() === "HEAD"
            ? {} : { body: request.postDataBuffer() }),
        });
        await route.fulfill({
          status: response.status,
          headers: {
            "content-type": response.headers.get("content-type") ?? "application/json",
            "access-control-allow-origin": origin,
            "access-control-allow-credentials": "true",
          },
          body: Buffer.from(await response.arrayBuffer()),
        });
      });
    }
    await page.goto(origin, { waitUntil: "domcontentloaded", timeout: 10000 });
    await page.getByRole("button", { name: "Image", exact: true }).first().click();
    await page.locator(".advanced-section-toggle").first().click();
    const steps = page.locator(".advanced-panel label").filter({ hasText: /^Steps/ }).locator("input");
    await steps.evaluate((field) => { field.dataset.validatorMeasure = "steps"; });
    await steps.fill("1");

    async function measureEdit(selector, value, feedbackSelector, match) {
      await page.evaluate(({ selector: fieldSelector, feedbackSelector: targetSelector, match: text }) => {
        const field = document.querySelector(fieldSelector);
        if (!field) throw new Error(`Missing input ${fieldSelector}`);
        window.__validatorEditTiming = new Promise((resolve, reject) => {
          let start = null;
          const timeout = setTimeout(() => { observer.disconnect(); reject(new Error(`No feedback matching ${text}`)); }, 5000);
          const observer = new MutationObserver(() => {
            if (start === null || ![...document.querySelectorAll(targetSelector)]
              .some((node) => node.textContent.includes(text))) return;
            observer.disconnect();
            clearTimeout(timeout);
            requestAnimationFrame(() => resolve(performance.now() - start));
          });
          observer.observe(document.body, { childList: true, subtree: true, characterData: true });
          field.addEventListener("input", () => { start = performance.now(); }, { capture: true, once: true });
        });
      }, { selector, feedbackSelector, match });
      await page.locator(selector).fill(value);
      return page.evaluate(() => window.__validatorEditTiming);
    }

    const single = [];
    for (let index = 0; index < samples; index += 1) {
      await steps.fill("1");
      await page.locator('[aria-label="Generate errors"]').filter({ hasText: "steps" })
        .waitFor({ state: "hidden", timeout: 5000 });
      single.push(await measureEdit('[data-validator-measure="steps"]', "0", '[aria-label="Generate errors"]', "steps"));
    }

    await steps.fill("1");
    await page.locator('[aria-label="Generate errors"]').filter({ hasText: "steps" })
      .waitFor({ state: "hidden", timeout: 5000 });
    await page.getByRole("button", { name: /Batch/ }).first().click();
    const batch = page.getByRole("textbox", { name: "Batch prompts" });
    await batch.waitFor({ state: "visible" });
    const batchTimes = [];
    for (let index = 0; index < samples; index += 1) {
      const prompts = Array.from({ length: 32 }, (_, item) => `prompt ${item + 1} edit ${index}`).join("\n");
      batchTimes.push(await measureEdit('[aria-label="Batch prompts"]', prompts,
        '.batch-run .batch-hint[role="status"]', "Options checked for 32 requests"));
    }
    const observedInteractiveAssets = [...fetchedChunks]
      .filter((name) => !bundleReport.initial.chunks.includes(name)).sort();
    const observedInteractiveBytes = observedInteractiveAssets.reduce((size, name) => {
      const chunk = bundleChunks.get(name);
      for (const metric of ["rawBytes", "gzipBytes", "brotliBytes"]) size[metric] += chunk[metric];
      return size;
    }, { rawBytes: 0, gzipBytes: 0, brotliBytes: 0 });
    const bundleBudget = bundleReport.budgets.largestRoute;
    const withinBundleBudget = [reportedInteractiveRoute, observedInteractiveBytes].every((size) =>
      size.rawBytes <= bundleBudget.rawBytes && size.gzipBytes <= bundleBudget.gzipBytes
      && size.brotliBytes <= bundleBudget.brotliBytes);
    const result = {
      origin, fixtureOrigin: fixtureOrigin ?? origin, browser: `Chromium ${browser.version()}`, samples,
      surface: "React Image Studio with controlled API fixture",
      routeStaticClosure: { chunks: reportedRoute.chunks.length,
        rawBytes: reportedRoute.rawBytes, gzipBytes: reportedRoute.gzipBytes,
        brotliBytes: reportedRoute.brotliBytes },
      reportedInteractiveClosure: { chunks: reportedInteractiveRoute.chunks.length,
        rawBytes: reportedInteractiveRoute.rawBytes, gzipBytes: reportedInteractiveRoute.gzipBytes,
        brotliBytes: reportedInteractiveRoute.brotliBytes },
      observedInteractiveAssets,
      observedInteractiveBytes,
      observedValidatorWasm: [...fetchedChunks].some((name) => bundleChunks.get(name)?.type === "wasm"),
      singleInputToPaintP95Ms: p95(single), batch32InputToPaintP95Ms: p95(batchTimes),
      inputVisibleBudgetMs: 100,
      interactionBundleBudget: bundleBudget,
      withinBudget: p95(single) <= 100 && p95(batchTimes) <= 100
        && [...fetchedChunks].some((name) => bundleChunks.get(name)?.type === "wasm")
        && withinBundleBudget,
    };
    console.log(JSON.stringify(result, null, 2));
    if (!result.withinBudget) process.exitCode = 1;
  } finally {
    await context.close();
  }
} finally {
  await browser.close();
}
