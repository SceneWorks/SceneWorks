// Pure browser measurements only. Each cold sample gets a fresh Chromium process, loads
// the deployed bridge, JS and WASM through the requested origin, and validates once.
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
  payload: sample.payload,
  modelEntry: corpus.modelFacts[sample.model],
  expected: sample.expected,
};

let moduleUrl = args.module ? new URL(args.module, origin).href : null;
if (!moduleUrl) {
  const response = await fetch(`${origin}/bundle-report.json`, { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error(`Bundle report unavailable (${response.status}); pass --module explicitly`);
  const report = await response.json();
  const candidates = report.chunks.filter((chunk) =>
    chunk.type === "chunk" && /portableValidationEngine.*\.js$/.test(chunk.fileName));
  if (candidates.length !== 1) throw new Error(`Expected one validator bridge chunk; found ${candidates.length}`);
  moduleUrl = new URL(`/${candidates[0].fileName}`, origin).href;
}

let browserVersion;
const cold = [];
let pure = [];
let singleInputVisible = [];
let batchInputVisible = [];
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
      const result = await page.evaluate(async ({ moduleUrl: url, fixture: input, measurePure, version }) => {
        const start = performance.now();
        const { load } = await import(url);
        const validate = await load();
        const request = { kind: input.kind, payload: input.payload, modelEntry: input.modelEntry };
        const serverRequired = (reason) => ({ state: "server-required", reason });
        const first = validate(request, version, serverRequired);
        const coldMs = performance.now() - start;
        const pureMs = [];
        if (measurePure) {
          for (let count = 0; count < 500; count += 1) {
            const t = performance.now();
            validate(request, version, serverRequired);
            pureMs.push(performance.now() - t);
          }
        }
        const inputVisible = async (batchSize) => {
          const field = document.createElement("input");
          const feedback = document.createElement("span");
          document.body.append(field, feedback);
          const timings = [];
          field.addEventListener("input", async () => {
            for (let item = 0; item < batchSize; item += 1) {
              validate(request, version, serverRequired);
              if ((item + 1) % 8 === 0 && item + 1 < batchSize) {
                await new Promise((resolve) => setTimeout(resolve, 0));
              }
            }
            feedback.textContent = field.value;
          });
          for (let edit = 0; edit < 100; edit += 1) {
            const value = `edit-${edit}`;
            const start = performance.now();
            field.value = value;
            field.dispatchEvent(new Event("input"));
            while (feedback.textContent !== value) {
              await new Promise((resolve) => setTimeout(resolve, 0));
            }
            await new Promise((resolve) => requestAnimationFrame(resolve));
            timings.push(performance.now() - start);
          }
          field.remove();
          feedback.remove();
          return timings;
        };
        return { coldMs, first, pureMs,
          singleInputVisibleMs: measurePure ? await inputVisible(1) : [],
          batchInputVisibleMs: measurePure ? await inputVisible(32) : [] };
      }, { moduleUrl, fixture, measurePure: index === 0, version: corpus.schemaVersion });
      deepStrictEqual(result.first.state === "valid"
        ? { ok: true, resolved: result.first.resolved }
        : { ok: false, refusal: result.first.refusal }, fixture.expected);
      cold.push(result.coldMs);
      if (index === 0) {
        pure = result.pureMs;
        singleInputVisible = result.singleInputVisibleMs;
        batchInputVisible = result.batchInputVisibleMs;
      }
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
const singleInputVisibleP95 = p95(singleInputVisible);
const batchInputVisibleP95 = p95(batchInputVisible);
const coldBudgetMs = args.profile === "desktop" ? 250 : 1000;
const result = { origin, moduleUrl, profile: args.profile, browser: `Chromium ${browserVersion}`,
  platform: process.platform, cache: "fresh browser process per cold sample", coldSamples: samples,
  coldFetchCompileFirstP95Ms: coldP95, coldMinMs: Math.min(...cold), coldMaxMs: Math.max(...cold), coldBudgetMs,
  pureSamples: pure.length, pureValidationP95Ms: pureP95, pureBudgetMs: 16,
  inputVisibleHarness: "DOM input event to next animation frame after validation; 32 resolved requests for batch",
  inputVisibleSamples: singleInputVisible.length, singleInputVisibleP95Ms: singleInputVisibleP95,
  batchInputVisibleP95Ms: batchInputVisibleP95, inputVisibleBudgetMs: 100,
  coldWithinBudget: coldP95 <= coldBudgetMs, pureWithinBudget: pureP95 <= 16,
  inputVisibleWithinBudget: singleInputVisibleP95 <= 100 && batchInputVisibleP95 <= 100 };
console.log(JSON.stringify(result, null, 2));
if (!result.coldWithinBudget || !result.pureWithinBudget || !result.inputVisibleWithinBudget) process.exitCode = 1;
