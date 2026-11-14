// Pure browser measurements only. Each cold sample gets a fresh Chromium process, loads
// the deployed JS/WASM through the requested origin, and calls the real Rust export once.
// Usage: node scripts/measure-options-validator.mjs --origin https://HOST --profile runpod
import { readFileSync } from "node:fs";
import { deepStrictEqual } from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, all) => {
  if (value.startsWith("--") && all[index + 1] && !all[index + 1].startsWith("--")) {
    pairs.push([value.slice(2), all[index + 1]]);
  }
  return pairs;
}, []));
const targetUrl = args.origin && new URL(args.origin);
const origin = targetUrl?.origin;
if (!origin || !["http:", "https:"].includes(targetUrl.protocol) || !["desktop", "runpod"].includes(args.profile)) {
  throw new Error("Pass --origin <http(s) origin> and --profile desktop|runpod");
}
const samples = args.samples === undefined ? 30 : Number(args.samples);
if (!Number.isInteger(samples) || samples < 5 || samples > 50) {
  throw new Error("--samples must be an integer from 5 to 50");
}
const corpus = JSON.parse(readFileSync(fileURLToPath(new URL(
  "../crates/sceneworks-options/tests/fixtures/validation_corpus.json", import.meta.url,
)), "utf8"));
const sample = corpus.cases.find((entry) => entry.name === "image model defaults");
if (!sample) throw new Error("Shared image corpus sample is missing");
const fixture = {
  kind: sample.kind,
  payload: JSON.stringify(sample.payload),
  modelEntry: JSON.stringify(corpus.modelFacts[sample.model]),
  expected: sample.expected,
};

let moduleUrl = args.module ? new URL(args.module, origin).href : null;
if (!moduleUrl) {
  const response = await fetch(`${origin}/bundle-report.json`, { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error(`Bundle report unavailable (${response.status}); pass --module explicitly`);
  const report = await response.json();
  const candidates = report.chunks.filter((chunk) =>
    chunk.type === "chunk" && /options_validator.*\.js$/.test(chunk.fileName));
  if (candidates.length !== 1) throw new Error(`Expected one validator chunk; found ${candidates.length}`);
  moduleUrl = new URL(`/${candidates[0].fileName}`, origin).href;
}

let browserVersion;
const cold = [];
let pure = [];
for (let index = 0; index < samples; index += 1) {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROME ? { executablePath: process.env.PLAYWRIGHT_CHROME } : {}),
  });
  browserVersion ??= browser.version();
  try {
    const context = await browser.newContext({ serviceWorkers: "block" });
    try {
      const page = await context.newPage();
      await page.route(`${origin}/__options_validator_benchmark__`, (route) => route.fulfill({
        status: 200,
        contentType: "text/html",
        headers: { "Content-Security-Policy": "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'" },
        body: "<!doctype html><title>Options validator benchmark</title>",
      }));
      await page.goto(`${origin}/__options_validator_benchmark__`, { waitUntil: "domcontentloaded", timeout: 10000 });
      const result = await page.evaluate(async ({ moduleUrl: url, fixture: input, measurePure }) => {
        const start = performance.now();
        const validator = await import(url);
        await validator.default();
        const first = JSON.parse(validator.validate_request(input.kind, input.payload, input.modelEntry));
        const coldMs = performance.now() - start;
        const pureMs = [];
        if (measurePure) {
          for (let count = 0; count < 500; count += 1) {
            const t = performance.now();
            validator.validate_request(input.kind, input.payload, input.modelEntry);
            pureMs.push(performance.now() - t);
          }
        }
        return { coldMs, first, pureMs };
      }, { moduleUrl, fixture, measurePure: index === 0 });
      if (result.first.validatorSchemaVersion !== corpus.schemaVersion) {
        throw new Error(`Browser validator version disagreed with corpus on sample ${index + 1}`);
      }
      deepStrictEqual(result.first.result, fixture.expected);
      cold.push(result.coldMs);
      if (index === 0) pure = result.pureMs;
    } finally {
      await context.close();
    }
  } finally {
    await browser.close();
  }
}

function p95(values) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(ordered.length * 0.95) - 1];
}
const coldP95 = p95(cold);
const pureP95 = p95(pure);
const coldBudgetMs = args.profile === "desktop" ? 250 : 1000;
const result = { origin, moduleUrl, profile: args.profile, browser: `Chromium ${browserVersion}`,
  platform: process.platform, cache: "fresh browser process per cold sample", coldSamples: samples,
  coldFetchCompileFirstP95Ms: coldP95, coldMinMs: Math.min(...cold), coldMaxMs: Math.max(...cold), coldBudgetMs,
  pureSamples: pure.length, pureValidationP95Ms: pureP95, pureBudgetMs: 16,
  coldWithinBudget: coldP95 <= coldBudgetMs, pureWithinBudget: pureP95 <= 16 };
console.log(JSON.stringify(result, null, 2));
if (!result.coldWithinBudget || !result.pureWithinBudget) process.exitCode = 1;
