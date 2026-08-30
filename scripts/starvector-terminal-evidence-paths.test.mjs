import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const text = async (name) => readFile(path.join(root, "scripts", name), "utf8");

test("terminal metrics bind actual product attachments rather than corpus preview fixtures", async () => {
  const metrics = await text("starvector-terminal-metrics.py");
  assert.match(metrics, /actual product quality preview/);
  assert.match(metrics, /previewPngPath/);
  assert.match(metrics, /sourceRasterPath/);
  assert.match(metrics, /consumed a raster other than the sealed submitted input/);
  assert.doesNotMatch(metrics, /case\["preview_png"\]/);
  assert.match(metrics, /importlib\.metadata\.version/);
  assert.match(metrics, /STARVECTOR_TERMINAL_LPIPS_LINEAR/);
  assert.match(metrics, /HF_HUB_OFFLINE/);
  assert.match(metrics, /metric-runtime-transcript\.json/);
  assert.doesNotMatch(metrics, /STARVECTOR_TERMINAL_EXECUTION_IDENTITY/);
});

test("route seals worker-observed raster, lifecycle, and hardware artifacts", async () => {
  const route = await text("starvector-terminal-route.mjs");
  assert.match(route, /sourceRasterSha256 !== entry\.image_quality/);
  assert.match(route, /item\.sourceRasterPath/);
  assert.match(route, /runs\/\$\{tuple\}\/hardware\/raw-probe/);
  assert.match(route, /runs\/\$\{tuple\}\/lifecycle-memory/);
  assert.match(route, /worker-owned peak accelerator memory observations/);
  assert.match(route, /producer\/transcript/);
  assert.match(route, /STARVECTOR_TERMINAL_CONTROLLER_CONTEXT/);
});

test("case bundle cannot predeclare observed provider, hardware, or receipt facts", async () => {
  const bundle = await text("starvector-terminal-case-bundle.mjs");
  assert.doesNotMatch(bundle, /run_identity/);
  assert.doesNotMatch(bundle, /terminal_suite_identity/);
  assert.doesNotMatch(bundle, /preview_png: row\.preview/);
});

test("source-built service materializes a receipt-backed offline closure", async () => {
  const service = await text("starvector-terminal-product-service.mjs");
  assert.match(service, /terminal_service_closure/);
  assert.match(service, /HF_HUB_OFFLINE: "1"/);
  assert.match(service, /TRANSFORMERS_OFFLINE: "1"/);
  assert.match(service, /copyRegularTree/);
  assert.match(service, /isSymbolicLink/);
});
