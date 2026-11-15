// Measure the first authenticated hybrid validation request from a fresh browser process.
// The server must expose the real /api/v1/options/validate endpoint at --origin.
// SCENEWORKS_API_TOKEN may supply bearer authentication without placing a token in argv.
// Supply --request-file with { kind, payload } when the target has a different catalog.
import { readFileSync } from "node:fs";
import { chromium } from "playwright";

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, all) => {
  if (value.startsWith("--") && all[index + 1] && !all[index + 1].startsWith("--")) {
    pairs.push([value.slice(2), all[index + 1]]);
  }
  return pairs;
}, []));
const target = args.origin && new URL(args.origin);
const samples = args.samples === undefined ? 30 : Number(args.samples);
if (!target || !["http:", "https:"].includes(target.protocol)
    || !["desktop", "runpod"].includes(args.profile)
    || !Number.isInteger(samples) || samples < 5 || samples > 50) {
  throw new Error("Pass --origin <http(s) API origin> --profile desktop|runpod [--samples 5..50]");
}
const request = args["request-file"]
  ? JSON.parse(readFileSync(args["request-file"], "utf8"))
  : { kind: "image", payload: { projectId: "p", model: "flux_dev", prompt: "a tree" } };
if (typeof request.kind !== "string" || !request.payload || Array.isArray(request.payload)
    || typeof request.payload !== "object") {
  throw new Error("--request-file must contain { kind, payload: object }");
}
const body = { validatorSchemaVersion: 1, kind: request.kind, requests: [request.payload] };
const token = process.env.SCENEWORKS_API_TOKEN;
const timings = [];
let browserVersion;
for (let index = 0; index < samples; index += 1) {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.PLAYWRIGHT_CHROME ? { executablePath: process.env.PLAYWRIGHT_CHROME } : {}) });
  browserVersion ??= browser.version();
  try {
    const context = await browser.newContext({ serviceWorkers: "block" });
    try {
      const page = await context.newPage();
      await page.goto(target.origin, { waitUntil: "domcontentloaded", timeout: 15000 });
      const result = await page.evaluate(async ({ body: input, token: auth }) => {
        const start = performance.now();
        const response = await fetch("/api/v1/options/validate", {
          method: "POST", credentials: "include",
          headers: { "content-type": "application/json",
            ...(auth ? { authorization: `Bearer ${auth}` } : {}) },
          body: JSON.stringify(input),
        });
        const data = await response.json();
        return { milliseconds: performance.now() - start, status: response.status, data };
      }, { body, token });
      if (result.status !== 200 || result.data?.validatorSchemaVersion !== 1
          || result.data.results?.length !== 1 || result.data.results[0]?.ok !== true) {
        throw new Error(`Validation sample ${index + 1} failed: HTTP ${result.status}, ${JSON.stringify(result.data)}`);
      }
      timings.push(result.milliseconds);
    } finally { await context.close(); }
  } finally { await browser.close(); }
}
const ordered = [...timings].sort((a, b) => a - b);
const p95 = ordered[Math.ceil(ordered.length * 0.95) - 1];
const budgetMs = args.profile === "desktop" ? 250 : 1000;
console.log(JSON.stringify({ origin: target.origin, profile: args.profile,
  browser: `Chromium ${browserVersion}`, cache: "fresh browser process per request",
  requestKind: request.kind, samples, firstValidationP95Ms: p95,
  minimumMs: ordered[0], maximumMs: ordered.at(-1), budgetMs,
  withinBudget: p95 <= budgetMs }, null, 2));
if (p95 > budgetMs) process.exitCode = 1;
