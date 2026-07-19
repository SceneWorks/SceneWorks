// sc-13135 — Batch-generate one preview thumbnail per style for the Style picker.
//
// There is no UI to do this except 286 manual Image Studio runs (8 group-level "overall"
// styles + 278 sub-styles). This script drives the local rust-api to generate one image per
// style id from a SINGLE shared reference prompt + fixed seed (so the STYLE is the only
// variable), downscales each result to a small square thumb via macOS `sips`, and writes
// `<out>/<id>.png`. It is resumable (skips ids whose thumb already exists) and never aborts
// the whole batch on a single failure.
//
// The outgoing `prompt` is composed CLIENT-SIDE (composeStyledPrompt + styleTextForId), NOT
// by sending a `styleId` to the API — so the script works on any API build, current or older,
// regardless of whether that build knows about the Style axis.
//
// Usage (from apps/web):
//   npm run gen:style-thumbnails -- --dry-run --limit 3      # inspect payloads, no I/O
//   npm run gen:style-thumbnails -- --project <projectId>    # generate into an existing project
//   npm run gen:style-thumbnails                             # auto-create a scratch project
//
// Notes on --model: the default `z_image_turbo` is a fast, always-available model. Passing
// `--model krea_2_turbo` gives the tuned "reference look" the picker was designed around, but
// that model needs a ~20GB manual download first (it is not provisioned by default).

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { composeStyledPrompt } from "../src/styleComposer.js";

// styleCatalog.js is the canonical accessor (STYLE_GROUPS + styleTextForId), but it loads
// styles.json via a bare `import ... from "./styles.json"` with no `with { type: "json" }`
// attribute — Vite/Vitest resolve that, but plain Node (which runs this CLI) rejects it. So the
// script loads the SAME styles.json through createRequire (require() reads JSON natively) and
// reconstructs the two accessors with byte-identical semantics: a sub-style id → its `prompt`, a
// group id → that group's `description` (sc-13171). The unit test imports the REAL styleTextForId
// from styleCatalog.js and asserts this reconstruction matches, so any drift fails CI.
const require = createRequire(import.meta.url);
const CATALOG = require("../src/data/styles.json");
export const STYLE_GROUPS = CATALOG.groups;

const STYLE_TEXT_BY_ID = new Map();
for (const group of STYLE_GROUPS) {
  STYLE_TEXT_BY_ID.set(group.id, group.description);
  for (const style of group.styles) {
    STYLE_TEXT_BY_ID.set(style.id, style.prompt);
  }
}

/** Free-text style prompt for a style/group id, or null when unknown (mirrors styleCatalog.js). */
function styleTextForId(id) {
  return id ? STYLE_TEXT_BY_ID.get(id) ?? null : null;
}

// Repo root (this file lives at apps/web/scripts/); the default --out is resolved from here so
// the script produces the same paths no matter the caller's CWD. Under plain Node import.meta.url
// is a file: URL; under Vitest it is a virtual scheme fileURLToPath rejects, so fall back to the
// CWD (apps/web) → repo root. The default out dir is never used for I/O in the tests, only here.
const REPO_ROOT = (() => {
  try {
    return fileURLToPath(new URL("../../../", import.meta.url));
  } catch {
    return resolve(process.cwd(), "..", "..");
  }
})();

const SIPS_BIN = "/usr/bin/sips";

const DEFAULTS = Object.freeze({
  prompt: "a red fox in a snowy forest",
  model: "z_image_turbo",
  seed: 20240719,
  size: 1024,
  thumb: 128,
  base: "http://127.0.0.1:8000",
  out: resolve(REPO_ROOT, "apps/web/public/style-thumbs"),
  project: null,
  keepFull: null,
  limit: null,
  concurrency: 1,
  dryRun: false,
  force: false,
});

// ---------------------------------------------------------------------------
// Pure logic (exported for the unit test — no I/O in here).
// ---------------------------------------------------------------------------

/**
 * Every style id to render, in stable order: for each of the 8 groups, the GROUP id (its
 * "overall" style, sc-13171) first, then each of that group's sub-style ids. Ids are globally
 * unique across the catalog, so the flat list has no duplicates (286 total for the shipped data).
 */
export function enumerateStyleIds() {
  const ids = [];
  for (const group of STYLE_GROUPS) {
    ids.push(group.id);
    for (const style of group.styles) {
      ids.push(style.id);
    }
  }
  return ids;
}

/**
 * Compose the exact outgoing `prompt` for one style id, wrapping the shared reference prompt in
 * the catalog style text. Resolves BOTH sub-style ids and group ids via styleTextForId (a group
 * id resolves to that group's description). Mirrors what the live studio preview sends.
 */
export function composeThumbnailPrompt(id, referencePrompt) {
  return composeStyledPrompt({ styleText: styleTextForId(id), userPrompt: referencePrompt });
}

/**
 * Build the JSON body for POST /api/v1/image/jobs. camelCase; `count: 1` is REQUIRED (the API's
 * blanket default is 4). width == height == size. A fixed seed keeps the noise identical across
 * every style so the style text is the only variable.
 */
export function buildJobPayload({ id, prompt, model, seed, size, projectId, count = 1 }) {
  return {
    projectId,
    prompt,
    model,
    count,
    width: size,
    height: size,
    seed,
    // Retained for provenance / debugging; the API ignores unknown fields and we compose the
    // prompt ourselves, so this never changes what is generated.
    styleId: id,
  };
}

/** Final thumbnail path for a style id: `<outDir>/<id>.png`. */
export function thumbnailPath(outDir, id) {
  return `${outDir}/${id}.png`;
}

/** Resumability rule: skip when the thumb already exists and --force was not passed. */
export function shouldSkip(path, force) {
  return !force && existsSync(path);
}

/**
 * Parse `--flag value` / `--bool` argv into a resolved options object (DEFAULTS overlaid with
 * overrides). Unknown flags throw so a typo fails loud instead of silently no-op'ing.
 */
export function parseArgs(argv) {
  const opts = { ...DEFAULTS };
  const toInt = (flag, raw) => {
    const n = Number.parseInt(raw, 10);
    if (!Number.isFinite(n)) {
      throw new Error(`${flag} expects an integer, got "${raw}"`);
    }
    return n;
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      const v = argv[i + 1];
      if (v === undefined) {
        throw new Error(`${arg} expects a value`);
      }
      i += 1;
      return v;
    };
    switch (arg) {
      case "--prompt":
        opts.prompt = next();
        break;
      case "--model":
        opts.model = next();
        break;
      case "--seed":
        opts.seed = toInt(arg, next());
        break;
      case "--size":
        opts.size = toInt(arg, next());
        break;
      case "--thumb":
        opts.thumb = toInt(arg, next());
        break;
      case "--project":
        opts.project = next();
        break;
      case "--base":
        opts.base = next().replace(/\/+$/, "");
        break;
      case "--out":
        opts.out = next();
        break;
      case "--keep-full":
        opts.keepFull = next();
        break;
      case "--limit":
        opts.limit = toInt(arg, next());
        break;
      case "--concurrency":
        opts.concurrency = toInt(arg, next());
        break;
      case "--dry-run":
        opts.dryRun = true;
        break;
      case "--force":
        opts.force = true;
        break;
      case "--help":
      case "-h":
        opts.help = true;
        break;
      default:
        throw new Error(`Unknown flag: ${arg}`);
    }
  }
  return opts;
}

// ---------------------------------------------------------------------------
// I/O helpers (only reached by main()).
// ---------------------------------------------------------------------------

const TERMINAL = new Set(["completed", "failed", "canceled", "interrupted"]);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function requireSips() {
  if (!existsSync(SIPS_BIN)) {
    throw new Error(
      `${SIPS_BIN} not found. This script relies on macOS \`sips\` to downscale thumbnails ` +
        `(sharp/ffmpeg are not available here). Run on macOS, or add a resize step.`,
    );
  }
}

async function apiFetch(base, path, init) {
  const res = await fetch(`${base}${path}`, init);
  return res;
}

async function assertApiReachable(base) {
  try {
    const res = await apiFetch(base, "/api/v1/projects", { method: "GET" });
    if (!res.ok) {
      throw new Error(`GET /api/v1/projects → HTTP ${res.status}`);
    }
  } catch (err) {
    throw new Error(
      `rust-api not reachable at ${base} (${err.message}). Start it (SCENEWORKS_API_PORT ` +
        `defaults to 8000) or pass --base, or use --dry-run to inspect payloads offline.`,
    );
  }
}

async function createScratchProject(base) {
  const name = `Style Thumbnails ${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}`;
  const res = await apiFetch(base, "/api/v1/projects", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (res.status !== 201) {
    throw new Error(`POST /api/v1/projects → HTTP ${res.status}: ${await res.text()}`);
  }
  const summary = await res.json();
  return summary.id;
}

async function submitJob(base, payload) {
  const res = await apiFetch(base, "/api/v1/image/jobs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (res.status !== 201) {
    throw new Error(`submit → HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  }
  const snap = await res.json();
  if (!snap?.id) {
    throw new Error("submit response missing job id");
  }
  return snap.id;
}

async function pollJob(base, jobId, { intervalMs = 1500, timeoutMs = 10 * 60 * 1000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const res = await apiFetch(base, `/api/v1/jobs/${jobId}`, { method: "GET" });
    if (!res.ok) {
      throw new Error(`poll → HTTP ${res.status}`);
    }
    const snap = await res.json();
    if (TERMINAL.has(snap.status)) {
      return snap;
    }
    if (Date.now() > deadline) {
      throw new Error(`timed out after ${Math.round(timeoutMs / 1000)}s (last status: ${snap.status})`);
    }
    await sleep(intervalMs);
  }
}

async function downloadImage(base, projectId, relativePath) {
  const res = await apiFetch(base, `/api/v1/projects/${projectId}/files/${relativePath}`, {
    method: "GET",
  });
  if (!res.ok) {
    throw new Error(`download → HTTP ${res.status}`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.length === 0) {
    throw new Error("downloaded image was empty");
  }
  return buf;
}

function resizeToThumb(fullPath, thumbOut, thumbPx) {
  const result = spawnSync(SIPS_BIN, ["-z", String(thumbPx), String(thumbPx), fullPath, "--out", thumbOut], {
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(`sips resize failed: ${result.stderr || result.stdout || `exit ${result.status}`}`);
  }
}

/**
 * Generate one thumbnail end-to-end: submit → poll → download → sips-resize → write thumb.
 * The full-res PNG is written to a temp path, resized, then discarded (unless --keep-full).
 */
async function generateOne(opts, id) {
  const prompt = composeThumbnailPrompt(id, opts.prompt);
  const payload = buildJobPayload({
    id,
    prompt,
    model: opts.model,
    seed: opts.seed,
    size: opts.size,
    projectId: opts.project,
  });

  const jobId = await submitJob(opts.base, payload);
  const snap = await pollJob(opts.base, jobId);
  if (snap.status !== "completed") {
    const reason = snap.error || snap.result?.error || snap.status;
    throw new Error(`job ${snap.status}: ${typeof reason === "string" ? reason : JSON.stringify(reason)}`);
  }

  const rel = snap.result?.assets?.[0]?.file?.path;
  if (!rel) {
    throw new Error("completed job had no result asset file path");
  }
  const buf = await downloadImage(opts.base, opts.project, rel);

  const fullPath = opts.keepFull
    ? `${opts.keepFull}/${id}.png`
    : `${opts.out}/.${id}.full.png`;
  writeFileSync(fullPath, buf);

  const thumbOut = thumbnailPath(opts.out, id);
  try {
    resizeToThumb(fullPath, thumbOut, opts.thumb);
  } finally {
    if (!opts.keepFull) {
      try {
        rmSync(fullPath, { force: true });
      } catch {
        // best-effort cleanup of the temp full-res file
      }
    }
  }
}

function printHelp() {
  console.log(
    [
      "generate-style-thumbnails — one preview thumbnail per style (sc-13135)",
      "",
      "Flags:",
      "  --prompt <text>     reference prompt (default: \"a red fox in a snowy forest\")",
      "  --model <id>        gen model (default: z_image_turbo; krea_2_turbo = tuned look, ~20GB dl)",
      "  --seed <int>        fixed seed shared across styles (default: 20240719)",
      "  --size <px>         native gen size, sent as width+height (default: 1024)",
      "  --thumb <px>        final downscaled square size (default: 128)",
      "  --project <id>      target project (omit → auto-create a scratch project)",
      "  --base <url>        rust-api base (default: http://127.0.0.1:8000)",
      "  --out <dir>         thumbnail output dir (default: apps/web/public/style-thumbs)",
      "  --keep-full <dir>   also keep native-res PNGs here (default: discard after resize)",
      "  --limit <n>         only process the first N ids (testing)",
      "  --concurrency <n>   workers (default: 1, serial — one GPU)",
      "  --dry-run           compose + print payloads/paths; no network, no files",
      "  --force             regenerate even if <out>/<id>.png exists (default: skip existing)",
    ].join("\n"),
  );
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    printHelp();
    return;
  }

  let ids = enumerateStyleIds();
  if (opts.limit != null) {
    ids = ids.slice(0, opts.limit);
  }

  if (opts.dryRun) {
    console.log(`DRY RUN — ${ids.length} id(s), no network calls, no files written.\n`);
    ids.forEach((id, i) => {
      const prompt = composeThumbnailPrompt(id, opts.prompt);
      const payload = buildJobPayload({
        id,
        prompt,
        model: opts.model,
        seed: opts.seed,
        size: opts.size,
        projectId: opts.project ?? "<auto-created-at-runtime>",
      });
      console.log(`[${i + 1}/${ids.length}] ${id} → ${thumbnailPath(opts.out, id)}`);
      console.log(`  payload: ${JSON.stringify(payload)}`);
      console.log(`  prompt:  ${JSON.stringify(prompt)}\n`);
    });
    return;
  }

  requireSips();
  await assertApiReachable(opts.base);

  if (!opts.project) {
    opts.project = await createScratchProject(opts.base);
    console.log(`Auto-created scratch project: ${opts.project}`);
  }

  mkdirSync(opts.out, { recursive: true });
  if (opts.keepFull) {
    mkdirSync(opts.keepFull, { recursive: true });
  }

  if (opts.concurrency !== 1) {
    console.warn(`(--concurrency ${opts.concurrency} requested; running serially — one GPU worker.)`);
  }

  const total = ids.length;
  let generated = 0;
  let skipped = 0;
  const failed = [];

  for (let i = 0; i < total; i += 1) {
    const id = ids[i];
    const label = `[${i + 1}/${total}] ${id}`;
    const thumb = thumbnailPath(opts.out, id);

    if (shouldSkip(thumb, opts.force)) {
      skipped += 1;
      console.log(`${label} SKIP (exists)`);
      continue;
    }

    let lastErr = null;
    let ok = false;
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      try {
        await generateOne(opts, id);
        ok = true;
        break;
      } catch (err) {
        lastErr = err;
        if (attempt < 3) {
          console.warn(`${label} retry ${attempt}/2 after: ${err.message}`);
          await sleep(1000 * attempt);
        }
      }
    }

    if (ok) {
      generated += 1;
      console.log(`${label} ✓`);
    } else {
      failed.push({ id, reason: lastErr?.message ?? "unknown" });
      console.error(`${label} FAIL(${lastErr?.message ?? "unknown"})`);
    }
  }

  console.log("\n─── Summary ───");
  console.log(`generated: ${generated}`);
  console.log(`skipped:   ${skipped}`);
  console.log(`failed:    ${failed.length}`);
  if (failed.length > 0) {
    console.log("failed ids:");
    for (const f of failed) {
      console.log(`  - ${f.id}: ${f.reason}`);
    }
  }
  console.log(`thumbnails: ${opts.out}`);

  if (failed.length > 0) {
    process.exitCode = 1;
  }
}

// Only run main() when invoked directly (not when imported by the test).
if (process.argv[1] && basename(process.argv[1]) === "generate-style-thumbnails.mjs") {
  main().catch((err) => {
    console.error(`fatal: ${err.message}`);
    process.exitCode = 1;
  });
}
