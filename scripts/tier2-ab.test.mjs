// SW-7033 Tier 2 A/B: pure-function and static-workflow contracts only. No build, API, worker,
// model, GPU or hub is touched; synthetic PNG/WAV bytes and temp directories only.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, readFile, rm } from "node:fs/promises";
import { deflateSync } from "node:zlib";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { sha256 } from "./lib/file-sha256.mjs";
import {
  DEFAULT_ROUTE_IDS, ROUTES, contentDigest, decodePng, decodedContent, descendants, diffObjects, dispatchOptions,
  exitCodeFor, identitySubset, numericDifference, parseTopMem, parseWav, planRoutes, readCatalog, redactPaths, referenceWav,
  renderMarkdown, reportName, requestBody, requestTemplate, resolveConfig, routeDownloadRows, routePresence, selectRoutes,
  summarize, verdictFor,
} from "./tier2-ab.mjs";

const BASE = "d3f7b05bf22826e44fd43da19cff0985fb17d68b", HEAD = "d2b0c76c9b9b9e00b5174aceec61c11a186dac68";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const env = (changes = {}) => ({ TIER2_AB_BACKEND: "mlx", TIER2_AB_BASE_SHA: BASE, TIER2_AB_HEAD_SHA: HEAD, TIER2_AB_BASE_DIR: "/w/base", TIER2_AB_HEAD_DIR: "/w/head", TIER2_AB_WORK: "/t/work", ...changes });

test("inputs: exact 40-hex SHAs, distinct sides, absolute paths, known backend-matched routes", () => {
  const cfg = resolveConfig(env(), { platform: "darwin", home: "/Users/runner" });
  assert.deepEqual(cfg.routes.map((route) => route.id), DEFAULT_ROUTE_IDS.mlx);
  assert.equal(cfg.hubs[0], "/Volumes/Models/huggingface/hub");
  for (const bad of [{ TIER2_AB_BASE_SHA: "main" }, { TIER2_AB_HEAD_SHA: HEAD.slice(0, 39) }, { TIER2_AB_HEAD_SHA: HEAD.toUpperCase() }, { TIER2_AB_HEAD_SHA: BASE }, { TIER2_AB_BACKEND: "candle" }, { TIER2_AB_BASE_DIR: "relative" }, { TIER2_AB_HEAD_DIR: "/w/base" }, { TIER2_AB_WORK: "/w/base/work" }, { TIER2_AB_INFERENCE_SHA: "abc" }, { TIER2_AB_ROUTES: "cuda-sdxl-q4-t2i" }, { TIER2_AB_ROUTES: "nope" }, { TIER2_AB_ROUTES: "mlx-moss-image-t2i,mlx-moss-image-t2i" }, { TIER2_AB_HUBS: "relative/hub" }]) {
    assert.throws(() => resolveConfig(env(bad), { platform: "darwin" }), undefined, JSON.stringify(bad));
  }
  assert.throws(() => resolveConfig(env({ GITHUB_ACTIONS: "true", RUNNER_TEMP: "/r" }), { platform: "darwin" }), /under RUNNER_TEMP/);
  assert.equal(resolveConfig(env({ GITHUB_ACTIONS: "true", RUNNER_TEMP: "/t" }), { platform: "darwin" }).work, "/t/work");
  const cuda = resolveConfig(env({ TIER2_AB_BACKEND: "cuda", TIER2_AB_BASE_DIR: "D:\\w\\base", TIER2_AB_HEAD_DIR: "D:\\w\\head", TIER2_AB_WORK: "D:\\t\\work", TIER2_AB_ROUTES: '["cuda-moss-image-t2i"]', TIER2_AB_HUBS: "E:\\hub;F:\\hub2" }), { platform: "win32" });
  assert.deepEqual(cuda.routes.map((route) => route.id), ["cuda-moss-image-t2i"]);
  assert.deepEqual(cuda.hubs, ["E:\\hub", "F:\\hub2"]);
  assert.deepEqual(selectRoutes("cuda", "").map((route) => route.id), DEFAULT_ROUTE_IDS.cuda);
});

test("default routes send one fixed request per route and the same JSON to both sides", () => {
  for (const ids of Object.values(DEFAULT_ROUTE_IDS)) for (const id of ids) {
    const route = ROUTES[id], template = requestTemplate(route);
    assert.equal(typeof route.body.seed, "number", `${id} needs a fixed seed`);
    assert.equal(route.body.model, route.model);
    assert.match(template.sha256, /^[0-9a-f]{64}$/);
    const a = requestBody(route, "p-1", "ref-1"), b = requestBody(route, "p-2", "ref-2");
    const strip = (body) => JSON.stringify({ ...body, projectId: "<project>", ...(route.referenceAudio ? { referenceAudioAssetId: "<reference-audio>" } : {}) });
    assert.equal(strip(a), strip(b)); assert.equal(strip(a), JSON.stringify(template.body));
  }
});

test("dispatch selector maps catalog inputs, defaults to preflight, refuses mismatches", () => {
  const inputs = { profile: "tier2-ab", ref: "a".repeat(40), backend: "candle", inference_sha: "", moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD }) };
  assert.deepEqual(dispatchOptions(inputs, inputs.ref), { base_sha: BASE, head_sha: HEAD, backend: "cuda", routes: "", hubs: "", preflight_only: true, inference_sha: "" });
  const custom = dispatchOptions({ ...inputs, backend: "mlx", moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, routes: ["mlx-moss-image-t2i"], hubs: ["/Volumes/X/hub"], preflight_only: false }) }, inputs.ref);
  assert.equal(custom.preflight_only, false); assert.equal(custom.routes, '["mlx-moss-image-t2i"]'); assert.equal(custom.hubs, '["/Volumes/X/hub"]');
  for (const bad of [{ ref: "b".repeat(40) }, { backend: "cpu" }, { download_missing: true }, { moss_options_json: "[]" }, { moss_options_json: "{" }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, case: "x" }) }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: BASE }) }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, routes: ["mlx-moss-image-t2i"] }) }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, hubs: ["/posix/hub"] }) }, { moss_options_json: JSON.stringify({ base_sha: BASE, head_sha: HEAD, preflight_only: "no" }) }, { inference_sha: "x" }]) {
    assert.throws(() => dispatchOptions({ ...inputs, ...bad }, inputs.ref), undefined, JSON.stringify(bad));
  }
});

const run = (sha, status = "completed") => ({ status, output: { content: { sha256: sha } }, error: status === "completed" ? undefined : "boom" });
test("verdict: all four DEC-7 outcomes plus error, with the bounded stop rule", () => {
  assert.deepEqual(verdictFor({ base1: run("a"), head: run("a") }), { verdict: "equal", needBase2: false, reason: "head decoded content equals base#1" });
  const pending = verdictFor({ base1: run("a"), head: run("b") });
  assert.equal(pending.verdict, null); assert.equal(pending.needBase2, true);
  assert.equal(verdictFor({ base1: run("a"), head: run("b"), base2: run("c") }).verdict, "nondeterministic_on_runner");
  assert.equal(verdictFor({ base1: run("a"), head: run("b"), base2: run("a") }).verdict, "head_differs");
  assert.equal(verdictFor({ base1: run("a"), head: run("b", "error") }).verdict, "error");
  assert.equal(verdictFor({ base1: run("a", "error"), head: run("a") }).verdict, "error");
  assert.equal(verdictFor({ base1: run("a"), head: run("b"), base2: run("a", "error") }).verdict, "error");
  // equal never asks for base#2 even when base#2 would differ
  assert.equal(verdictFor({ base1: run("a"), head: run("a"), base2: run("z") }).verdict, "equal");
});

test("exit policy: head_differs, error, hub mutation and zero ran fail; nondeterminism and skips do not", () => {
  const report = (routes, mode = "run") => ({ mode, routes, summary: summarize(routes) });
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "equal" }, { status: "skipped", skipReason: "absent" }])), 0);
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "nondeterministic_on_runner" }])), 0);
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "head_differs" }])), 1);
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "error" }])), 1);
  assert.equal(exitCodeFor(report([{ status: "ran", verdict: "equal", hubMutation: { changed: true } }])), 1);
  assert.equal(exitCodeFor(report([{ status: "skipped" }, { status: "skipped" }])), 1, "zero routes ran");
  assert.equal(exitCodeFor(report([{ status: "ready" }], "preflight")), 0);
  assert.equal(exitCodeFor(report([{ status: "skipped" }], "preflight")), 1);
  assert.equal(exitCodeFor({ ...report([{ status: "ran", verdict: "equal" }]), fatal: "x" }), 1);
});

// --- synthetic PNG encoder with every filter type and arbitrary extra chunks ---
const crc32 = (bytes) => { let c = 0xffffffff; for (const v of bytes) { c ^= v; for (let i = 0; i < 8; i++) c = (c >>> 1) ^ ((c & 1) ? 0xedb88320 : 0); } return (c ^ 0xffffffff) >>> 0; };
const chunk = (kind, data) => { const type = Buffer.from(kind), out = Buffer.alloc(data.length + 12); out.writeUInt32BE(data.length); type.copy(out, 4); data.copy(out, 8); out.writeUInt32BE(crc32(Buffer.concat([type, data])), out.length - 4); return out; };
function png(pixels, width, height, channels, { filters = [0], extra = [] } = {}) {
  const header = Buffer.alloc(13); header.writeUInt32BE(width); header.writeUInt32BE(height, 4); header[8] = 8; header[9] = channels === 4 ? 6 : 2;
  const stride = width * channels, rows = [];
  for (let y = 0; y < height; y++) {
    const filter = filters[y % filters.length], row = pixels.subarray(y * stride, (y + 1) * stride), prior = y ? pixels.subarray((y - 1) * stride, y * stride) : Buffer.alloc(stride), out = Buffer.alloc(stride + 1);
    out[0] = filter;
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? row[x - channels] : 0, b = prior[x], c = x >= channels ? prior[x - channels] : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const predictor = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][filter];
      out[x + 1] = (row[x] - predictor) & 0xff;
    }
    rows.push(out);
  }
  return Buffer.concat([Buffer.from("89504e470d0a1a0a", "hex"), chunk("IHDR", header), ...extra, chunk("IDAT", deflateSync(Buffer.concat(rows))), chunk("IEND", Buffer.alloc(0))]);
}
const itxt = (text) => chunk("iTXt", Buffer.concat([Buffer.from("sceneworks:workflow\0\0\0\0\0"), Buffer.from(text)]));
const pixels = (w, h, channels, seed = 7) => { const out = Buffer.alloc(w * h * channels); let s = seed; for (let i = 0; i < out.length; i++) { s = (s * 1103515245 + 12345) >>> 0; out[i] = s >>> 24; } return out; };

test("decoded PNG content ignores metadata chunks and filter choice; file hash does not", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-png-")); t.after(() => rm(dir, { recursive: true, force: true }));
  const rgb = pixels(9, 5, 3);
  const a = png(rgb, 9, 5, 3, { extra: [itxt('{"jobId":"job-1","createdAt":"2026-11-12T00:00:00Z"}')] });
  const b = png(rgb, 9, 5, 3, { filters: [1, 2, 3, 4, 0], extra: [itxt('{"jobId":"job-2","createdAt":"2026-11-12T09:09:09Z"}'), chunk("tEXt", Buffer.from("Comment\0other"))] });
  const changed = Buffer.from(rgb); changed[10] ^= 1;
  const c = png(changed, 9, 5, 3);
  for (const [name, bytes] of [["a.png", a], ["b.png", b], ["c.png", c]]) await writeFile(path.join(dir, name), bytes);
  const [da, db, dc] = await Promise.all(["a.png", "b.png", "c.png"].map((name) => decodedContent(path.join(dir, name), "image")));
  assert.notEqual(a.toString("hex"), b.toString("hex"));
  assert.equal(da.method, "png_rgba8"); assert.deepEqual(da.shape, { width: 9, height: 5 });
  assert.equal(da.sha256, db.sha256, "same pixels must give the same decoded hash");
  assert.notEqual(da.sha256, dc.sha256, "one changed channel must change the decoded hash");
  const decoded = decodePng(a);
  assert.equal(decoded.rgba.length, 9 * 5 * 4); assert.equal(decoded.rgba[3], 255); assert.deepEqual([...decoded.rgba.subarray(0, 3)], [...rgb.subarray(0, 3)]);
  const rgba = pixels(4, 3, 4, 3);
  assert.deepEqual(decodePng(png(rgba, 4, 3, 4, { filters: [4, 3, 1] })).rgba, rgba);
  assert.deepEqual(numericDifference(da.buffer, dc.buffer).changedElements, 1);
  assert.equal(numericDifference(da.buffer, db.buffer).maxAbsoluteDifference, 0);
  assert.equal(numericDifference(da.buffer, Buffer.alloc(3)).comparable, false);
  assert.throws(() => decodePng(Buffer.from("not a png")), /not a PNG/);
});

test("WAV fidelity hashes PCM samples, not header or trailing chunks", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-wav-")); t.after(() => rm(dir, { recursive: true, force: true }));
  const wav = referenceWav(), parsed = parseWav(wav);
  assert.equal(parsed.sampleRate, 24_000); assert.equal(parsed.channels, 1); assert.equal(parsed.data.length, 96_000);
  const list = Buffer.concat([Buffer.from("LIST"), Buffer.from([4, 0, 0, 0]), Buffer.from("INFO")]);
  const tagged = Buffer.concat([wav, list]); tagged.writeUInt32LE(tagged.length - 8, 4);
  await writeFile(path.join(dir, "a.wav"), wav); await writeFile(path.join(dir, "b.wav"), tagged);
  const [a, b] = await Promise.all([decodedContent(path.join(dir, "a.wav"), "audio"), decodedContent(path.join(dir, "b.wav"), "audio")]);
  assert.equal(a.method, "wav_pcm_samples"); assert.equal(a.sha256, b.sha256);
  assert.notEqual(contentDigest({ x: 1 }, Buffer.from([1])), contentDigest({ x: 2 }, Buffer.from([1])));
});

test("catalog rows: default or requested tier, platform filter, soft rows optional, derived tiers ignored", () => {
  const model = { id: "m", downloads: [
    { provider: "huggingface", repo: "o/m", revision: "1".repeat(40), variant: "q4", default: true, files: ["q4/*"] },
    { provider: "huggingface", repo: "o/m", revision: "1".repeat(40), variant: "q8", files: ["q8/*"] },
    { provider: "huggingface", repo: "o/m", revision: "1".repeat(40), variant: "q8", files: [], localDerivation: { fromVariant: "bf16" } },
    { provider: "huggingface", repo: "o/tok", revision: "2".repeat(40), coRequisite: true, componentId: "tok", files: ["tokenizer.json"], platforms: ["windows"] },
    { provider: "huggingface", repo: "o/ctl", revision: "3".repeat(40), coRequisite: true, required: "soft", files: ["c.safetensors"] },
  ] };
  const mac = routeDownloadRows(model, { platform: "darwin", tier: null });
  assert.equal(mac.tier, "q4"); assert.deepEqual(mac.required.map((row) => row.repo), ["o/m"]); assert.deepEqual(mac.optional.map((row) => row.repo), ["o/ctl"]);
  const win = routeDownloadRows(model, { platform: "win32", tier: "q8" });
  assert.deepEqual(win.required.map((row) => `${row.repo}:${row.files.join(",")}`), ["o/m:q8/*", "o/tok:tokenizer.json"]);
  assert.throws(() => routeDownloadRows(model, { platform: "darwin", tier: "bf16" }), /no bf16/);
});

test("hub presence: metadata only, first complete hub wins, groups and catalog drift skip with reasons", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-hub-")); t.after(() => rm(dir, { recursive: true, force: true }));
  const rev = "a".repeat(40), empty = path.join(dir, "empty"), full = path.join(dir, "full");
  const snap = path.join(full, "models--O--img", "snapshots", rev);
  await mkdir(path.join(snap, "q4"), { recursive: true }); await mkdir(empty);
  await writeFile(path.join(snap, "q4", "model.safetensors"), "x"); await writeFile(path.join(snap, "tokenizer.json"), "y");
  const model = { id: "img_model", downloads: [{ provider: "huggingface", repo: "O/img", revision: rev, variant: "q4", default: true, files: ["q4/*", "tokenizer.json"] }, { provider: "huggingface", repo: "O/soft", revision: rev, required: "soft", coRequisite: true, files: ["s.bin"] }] };
  const rows = routeDownloadRows(model, { platform: "darwin", tier: "q4" });
  assert.equal((await routePresence(empty, rows)).complete, false);
  const present = await routePresence(full, rows);
  assert.equal(present.complete, true); assert.equal(present.optionalMissing.length, 1);
  const route = (id, group, modelId = "img_model") => ({ id, backend: "mlx", model: modelId, kind: "image", mode: "text_to_image", tier: "q4", group, endpoint: "/api/v1/image/jobs", body: { model: modelId, seed: 1 } });
  const planned = await planRoutes({ routes: [route("r1", "g"), route("r2", "g"), route("r3", null, "absent_model")], hubs: [empty, full], baseCatalog: [model], headCatalog: [model], platform: "darwin" });
  assert.deepEqual(planned.map((item) => item.status), ["ready", "skipped", "skipped"]);
  assert.equal(planned[0].hub, full); assert.equal(planned[0].presence.length, 2);
  assert.match(planned[1].skipReason, /alternative r1/); assert.match(planned[2].skipReason, /not in the base catalog/);
  const drifted = { ...model, downloads: [{ ...model.downloads[0], revision: "b".repeat(40) }] };
  assert.match((await planRoutes({ routes: [route("r1", null)], hubs: [full], baseCatalog: [drifted], headCatalog: [model], platform: "darwin" }))[0].skipReason, /differ between base and head/);
  const missing = await planRoutes({ routes: [route("r1", null)], hubs: [empty], baseCatalog: [model], headCatalog: [model], platform: "darwin" });
  assert.match(missing[0].skipReason, /weights absent from every hub root/);
});

test("shipped catalogs at this checkout satisfy every default route's model and tier", async () => {
  const models = await readCatalog(root);
  for (const [backend, ids] of Object.entries(DEFAULT_ROUTE_IDS)) for (const id of ids) {
    const route = ROUTES[id], model = models.find((entry) => entry.id === route.model);
    assert.ok(model, `${id}: ${route.model} missing from catalog`);
    const rows = routeDownloadRows(model, { platform: backend === "mlx" ? "darwin" : "win32", tier: route.tier });
    assert.ok(rows.required.length > 0, `${id} needs pinned rows`);
  }
});

test("identity subset, path redaction and diff are stable and hub-relative", () => {
  const payload = { prompt: "secret prompt", seed: 3, model: "flux_dev", modelBinding: { artifact: { path: "E:\\huggingface\\hub\\models--O--m\\snapshots\\x\\q4", sha256: "abc" } }, advanced: { steps: 4, tier: "q4" }, output: { dir: "/tmp/state/data/projects/p" } };
  const subset = identitySubset(payload);
  assert.deepEqual(Object.keys(subset).sort(), ["advanced", "model", "modelBinding"]);
  assert.deepEqual(subset.advanced, { tier: "q4" });
  const redacted = redactPaths(subset, { hub: "E:\\huggingface\\hub", "hf-home": "E:\\huggingface" });
  assert.equal(redacted.modelBinding.artifact.path, "<hub>/models--O--m/snapshots/x/q4");
  assert.equal(redactPaths("/Users/r/work/state/x", { work: "/Users/r/work", state: "/Users/r/work/state" }), "<state>/x");
  const diff = diffObjects({ a: 1, b: { c: [1, 2] } }, { a: 1, b: { c: [1, 3] }, d: true });
  assert.deepEqual(diff.entries.map((entry) => entry.path), ["/b/c/1", "/d"]); assert.equal(diff.count, 2);
});

test("memory sampling helpers parse top MEM units and walk the worker tree", () => {
  assert.equal(parseTopMem("2640K"), 2640 * 1024); assert.equal(parseTopMem("512M+"), 512 * 1024 ** 2); assert.equal(parseTopMem("31G-"), 31 * 1024 ** 3); assert.equal(parseTopMem("12"), 12);
  assert.throws(() => parseTopMem("n/a"));
  assert.deepEqual(descendants([{ pid: 2, ppid: 1 }, { pid: 3, ppid: 2 }, { pid: 4, ppid: 3 }, { pid: 5, ppid: 9 }], 2), [2, 3, 4]);
});

test("report shape: summary counts, markdown table and per-mode file names", () => {
  const routes = [{ id: "a", model: "m", status: "ran", verdict: "equal", request: { sha256: "r" }, hub: "/h", runs: [{ label: "base#1", status: "completed", timing: { enqueueToCompleteSeconds: 12.5 }, memory: { peakBytes: 2 * 1024 ** 3 }, output: { content: { method: "png_rgba8", sha256: "x" }, fileSha256: "f" } }, { label: "head", status: "completed", timing: { enqueueToCompleteSeconds: 13 }, memory: { peakBytes: 2 * 1024 ** 3 }, output: { content: { sha256: "x" } } }], comparison: { payloadIdentityDiff: { count: 1, entries: [{ path: "/modelBinding", base: null, head: "{}" }] } } }, { id: "b", model: "n", status: "skipped", skipReason: "weights absent" }];
  const summary = summarize(routes);
  assert.deepEqual({ ran: summary.ran, skipped: summary.skipped, equal: summary.equal }, { ran: 1, skipped: 1, equal: 1 });
  const markdown = renderMarkdown({ backend: "mlx", mode: "run", status: "finished", exitCode: 0, base: { sha: BASE }, head: { sha: HEAD }, runner: { name: "nax-macos-2", platform: "darwin", arch: "arm64", totalMemoryBytes: 128 * 1024 ** 3 }, startedAt: "s", finishedAt: "f", memoryMethod: "m", routes, summary });
  assert.match(markdown, /\| a \| m \| ran \| equal \| 12\.5 s \| 13\.0 s \| – \| 2\.00 GiB/);
  assert.match(markdown, /\| b \| n \| skipped \| – .*weights absent/);
  assert.match(markdown, /`\/modelBinding`: base null → head \{\}/);
  assert.equal(reportName("run"), "report"); assert.equal(reportName("preflight"), "preflight");
});

test("CLI refuses missing mode and invalid inputs before touching any checkout", () => {
  const cli = path.join(root, "scripts", "tier2-ab.mjs");
  const none = spawnSync(process.execPath, [cli], { encoding: "utf8", env: { ...process.env, ...env() } });
  assert.notEqual(none.status, 0); assert.match(none.stderr, /exactly one of --preflight-only/);
  const bad = spawnSync(process.execPath, [cli, "--preflight-only"], { encoding: "utf8", env: { ...process.env, ...env({ TIER2_AB_BASE_SHA: "main" }) } });
  assert.notEqual(bad.status, 0); assert.match(bad.stderr, /40-hex/);
});

// ---- --run wiring against a fake API/worker binary (plain Node; no build, model or GPU) ----
// The fake speaks only the API surface the driver uses: port-0 announcement, owned health,
// worker registration, project, job lifecycle with stages, saved PNG and the product file route.
// Each job's PNG carries a different iTXt chunk; pixels come from the seed plus mode.json's offset.
const FAKE_APP = String.raw`#!/usr/bin/env node
const http = require("http"), fs = require("fs"), path = require("path"), zlib = require("zlib");
const env = process.env, mode = JSON.parse(fs.readFileSync(path.join(__dirname, "mode.json"), "utf8"));
const parent = Number(env.SCENEWORKS_PARENT_PID);
setInterval(() => { try { process.kill(parent, 0); } catch { process.exit(0); } }, 100);
if (env.SCENEWORKS_WORKER_ONLY === "1") {
  const register = () => { const req = http.request(env.SCENEWORKS_API_URL + "/fake/register", { method: "POST" }, (res) => res.resume()); req.on("error", () => setTimeout(register, 100)); req.end(env.SCENEWORKS_WORKER_ID); };
  register(); return;
}
const data = env.SCENEWORKS_DATA_DIR, projects = path.join(data, "projects");
fs.mkdirSync(path.dirname(env.SCENEWORKS_JOBS_DB_PATH), { recursive: true }); fs.writeFileSync(env.SCENEWORKS_JOBS_DB_PATH, "");
const workers = [], jobs = new Map(), owned = new Map();
const crc = (b) => { let c = 0xffffffff; for (const v of b) { c ^= v; for (let i = 0; i < 8; i++) c = (c >>> 1) ^ ((c & 1) ? 0xedb88320 : 0); } return (c ^ 0xffffffff) >>> 0; };
const chunk = (t, d) => { const o = Buffer.alloc(d.length + 12); o.writeUInt32BE(d.length); o.write(t, 4); d.copy(o, 8); o.writeUInt32BE(crc(Buffer.concat([Buffer.from(t), d])), o.length - 4); return o; };
const png = (value, note) => { const h = Buffer.alloc(13); h.writeUInt32BE(4); h.writeUInt32BE(4, 4); h[8] = 8; h[9] = 2; const rows = Buffer.alloc(4 * 13, value & 255); for (let y = 0; y < 4; y++) rows[y * 13] = 0; return Buffer.concat([Buffer.from("89504e470d0a1a0a", "hex"), chunk("IHDR", h), chunk("iTXt", Buffer.from("workflow\0\0\0\0\0" + note)), chunk("IDAT", zlib.deflateSync(rows)), chunk("IEND", Buffer.alloc(0))]); };
const send = (res, status, value) => { res.writeHead(status, { "content-type": "application/json" }); res.end(JSON.stringify(value)); };
const server = http.createServer((req, res) => {
  let body = ""; req.on("data", (c) => (body += c)); req.on("end", () => {
    const url = new URL(req.url, "http://x"), p = url.pathname;
    if (p === "/fake/register") { workers.push({ id: body, status: "idle", gpuId: "mlx", gpuName: "Apple Silicon (MLX)", capabilities: ["gpu", "image_generate"] }); return send(res, 200, {}); }
    if (p === "/api/v1/health") return send(res, 200, { status: "ok", service: "sceneworks-api", runtime: "rust", readiness: { status: "ready" }, authRequired: false, directories: { data, config: env.SCENEWORKS_CONFIG_DIR, projects, jobsDb: env.SCENEWORKS_JOBS_DB_PATH } });
    if (p === "/api/v1/workers") return send(res, 200, workers);
    if (p === "/api/v1/projects" && req.method === "POST") { const id = "p" + owned.size; const dir = path.join(projects, id); fs.mkdirSync(dir, { recursive: true }); owned.set(id, fs.realpathSync(dir)); return send(res, 201, { id, path: fs.realpathSync(dir) }); }
    if (p === "/api/v1/image/jobs") { const request = JSON.parse(body); const id = "job-" + Date.now() + "-" + jobs.size; jobs.set(id, { id, request, created: Date.now() }); return send(res, 201, { id, type: "image_generate", payload: request }); }
    const job = /^\/api\/v1\/jobs\/([^/]+)$/.exec(p);
    if (job) {
      const j = jobs.get(job[1]), age = Date.now() - j.created, done = age > 900;
      const stage = age < 150 ? "queued" : age < 400 ? "loading_model" : age < 900 ? "generating" : "completed";
      const view = { id: j.id, status: done ? "completed" : age < 150 ? "queued" : "running", stage, message: "", workerId: workers[0].id, backend: "mlx", assignedGpu: "mlx", payload: { ...j.request, modelBinding: { revision: mode.binding, path: path.join(env.HF_HUB_CACHE, "models--x") } }, result: {} };
      if (done) { const rel = "assets/images/" + j.id + ".png"; const file = path.join(owned.get(j.request.projectId), ...rel.split("/")); if (!fs.existsSync(file)) { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, png(j.request.seed + mode.offset, j.id)); } view.result = { adapter: "fake", model: j.request.model, assets: [{ id: "a-" + j.id, type: "image", file: { path: rel, mimeType: "image/png" } }], assetIds: ["a-" + j.id] }; }
      return send(res, 200, view);
    }
    const file = /^\/api\/v1\/projects\/([^/]+)\/files\/(.+)$/.exec(p);
    if (file) { res.writeHead(200); return res.end(fs.readFileSync(path.join(owned.get(file[1]), ...decodeURIComponent(file[2]).split("/")))); }
    if (/\/cancel$/.test(p)) return send(res, 200, {});
    send(res, 404, { error: p });
  });
});
server.listen(0, "127.0.0.1", () => console.log(JSON.stringify({ event: "api_listening", address: "127.0.0.1:" + server.address().port })));
`;
const macWithPython = process.platform === "darwin" && process.arch === "arm64" && spawnSync("python3.12", ["--version"]).status === 0;
test("--run drives fake base/head services through the stop rule and writes evidence", { skip: !macWithPython && "needs Apple-silicon macOS with python3.12 (the mlx preflight's own requirements)", timeout: 120_000 }, async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tier2-ab-run-")); t.after(() => rm(dir, { recursive: true, force: true }));
  const pin = "4".repeat(40), revision = "f".repeat(40);
  const lock = `[[package]]\nname = "pmetal-mlx-sys"\nversion = "0.1.0"\nsource = "git+https://example.invalid/mlx-rs?rev=${"9".repeat(40)}#${"9".repeat(40)}"\n\n[[package]]\nname = "gen-core"\nversion = "0.1.0"\nsource = "git+https://github.com/SceneWorks/inference?rev=${pin}#${pin}"\n`;
  const catalog = JSON.stringify({ models: [{ id: "moss_image_sft", downloads: [{ provider: "huggingface", repo: "OpenMOSS-Team/MOSS-Image-SFT", revision, files: ["a.bin"] }] }] });
  const git = (cwd, ...args) => { const result = spawnSync("git", ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", ...args], { cwd, encoding: "utf8" }); assert.equal(result.status, 0, result.stderr); return result.stdout.trim(); };
  const sides = {};
  for (const [label, note] of [["base", "b"], ["head", "h"]]) {
    const repo = path.join(dir, label); await mkdir(path.join(repo, "config", "manifests"), { recursive: true });
    await writeFile(path.join(repo, "Cargo.lock"), lock); await writeFile(path.join(repo, "config", "manifests", "builtin.models.jsonc"), catalog); await writeFile(path.join(repo, "note"), note);
    git(repo, "init", "-q"); git(repo, "add", "."); git(repo, "commit", "-qm", label); sides[label] = { repo, sha: git(repo, "rev-parse", "HEAD") };
  }
  const hub = path.join(dir, "hub"), snapshot = path.join(hub, "models--OpenMOSS-Team--MOSS-Image-SFT", "snapshots", revision);
  await mkdir(snapshot, { recursive: true }); await writeFile(path.join(snapshot, "a.bin"), "w");
  const scenario = async (name, headOffset) => {
    const work = path.join(dir, `work-${name}`), built = {};
    for (const label of ["base", "head"]) {
      const bin = path.join(work, "scratch", "bin", label), binary = path.join(bin, "sceneworks-rust-api");
      await mkdir(bin, { recursive: true }); await writeFile(binary, FAKE_APP, { mode: 0o755 });
      await writeFile(path.join(bin, "mode.json"), JSON.stringify({ offset: label === "head" ? headOffset : 0, binding: label }));
      built[label] = { binary, binarySha256: sha256(await readFile(binary)), metallib: null };
    }
    await writeFile(path.join(work, "scratch", "build.json"), JSON.stringify({ baseSha: sides.base.sha, headSha: sides.head.sha, backend: "mlx", built }));
    const childEnv = { PATH: process.env.PATH, HOME: process.env.HOME, TIER2_AB_BACKEND: "mlx", TIER2_AB_BASE_SHA: sides.base.sha, TIER2_AB_HEAD_SHA: sides.head.sha, TIER2_AB_BASE_DIR: sides.base.repo, TIER2_AB_HEAD_DIR: sides.head.repo, TIER2_AB_WORK: work, TIER2_AB_ROUTES: "mlx-moss-image-t2i", TIER2_AB_HUBS: hub };
    const result = spawnSync(process.execPath, [path.join(root, "scripts", "tier2-ab.mjs"), "--run"], { env: childEnv, encoding: "utf8", timeout: 100_000 });
    return { status: result.status, stderr: result.stderr, report: JSON.parse(await readFile(path.join(work, "evidence", "report.json"), "utf8")), work };
  };
  const equal = await scenario("equal", 0);
  assert.equal(equal.status, 0, equal.stderr + JSON.stringify(equal.report.fatal));
  const route = equal.report.routes[0];
  assert.equal(route.verdict, "equal"); assert.deepEqual(route.runs.map((run) => run.label), ["base#1", "head"], "stop after head == base#1");
  assert.notEqual(route.runs[0].output.fileSha256, route.runs[1].output.fileSha256, "metadata chunk differs per job");
  assert.equal(route.runs[0].output.content.sha256, route.runs[1].output.content.sha256);
  assert.equal(route.runs[0].identity.payload.modelBinding.path, "<hub>/models--x");
  assert.equal(route.comparison.payloadIdentityDiff.count, 1, "binding revision base->head is reported");
  assert.ok(route.runs.every((run) => run.timing.enqueueToCompleteSeconds > 0 && Number.isFinite(run.timing.loadSeconds) && run.memory.peakBytes > 0 && run.cleanup.every((c) => c.exited && !c.forced)));
  assert.equal(route.hubMutation.changed, false);
  const differs = await scenario("differs", 1);
  assert.equal(differs.status, 1); assert.match(differs.stderr, /head_differs/);
  assert.equal(differs.report.routes[0].verdict, "head_differs");
  assert.deepEqual(differs.report.routes[0].runs.map((run) => run.label), ["base#1", "head", "base#2"]);
  assert.equal(differs.report.routes[0].comparison.numericDifference.changedElements, 48);
  assert.match(await readFile(path.join(differs.work, "evidence", "report.md"), "utf8"), /head_differs/);
});

// ---- static workflow contracts (dispatch-only, runner labels, GPU binding, cleanup) ----
const workflowText = (name) => readFile(path.join(root, ".github", "workflows", name), "utf8");
function job(workflow, name) {
  const start = workflow.indexOf(`\n  ${name}:\n`);
  assert.ok(start >= 0, `${name} job exists`);
  const next = workflow.slice(start + 1).search(/\n  [a-z][a-z0-9-]*:\n/);
  return workflow.slice(start, next < 0 ? undefined : start + 1 + next);
}

test("tier2-ab is reachable only as a dispatched campaign profile and keeps the 25-input ceiling", async () => {
  const dispatcher = await workflowText("memory-catalog-campaign.yml");
  assert.match(dispatcher, /^on:\n  workflow_dispatch:\n/m);
  assert.doesNotMatch(dispatcher, /^  (push|pull_request|pull_request_target|schedule|merge_group|workflow_run):/m);
  assert.match(dispatcher, /          - tier2-ab\n/);
  assert.match(dispatcher, /if \(i\.profile === 'tier2-ab'\) \{ emit\('enabled','false'\); process\.exit\(0\); \}/);
  const inputs = dispatcher.split("    inputs:\n")[1].split("\n# `gh pr create`")[0];
  assert.equal([...inputs.matchAll(/^      ([A-Za-z_][A-Za-z0-9_]*):$/gm)].length, 25);
  const selector = job(dispatcher, "tier2-ab-selector");
  assert.match(selector, /if: \$\{\{ needs\.moss-selector-preflight\.outputs\.enabled != 'true' && inputs\.profile == 'tier2-ab' \}\}/);
  assert.match(selector, /dispatchOptions/); assert.match(selector, /runs-on: ubuntu-latest/);
  const caller = job(dispatcher, "tier2-ab");
  assert.match(caller, /if: \$\{\{ needs\.moss-selector-preflight\.outputs\.enabled != 'true' && inputs\.profile == 'tier2-ab' \}\}/);
  assert.match(caller, /uses: \.\/\.github\/workflows\/tier2-ab\.yml/);
  assert.match(caller, /HF_TOKEN: \$\{\{ secrets\.HF_TOKEN \}\}/);
  for (const catalogJob of ["mlx", "candle"]) assert.doesNotMatch(job(dispatcher, catalogJob).split("\n").find((line) => line.startsWith("    if:")), /tier2-ab/);
  const reusable = await workflowText("tier2-ab.yml");
  const triggers = reusable.slice(reusable.indexOf("\non:"), reusable.indexOf("\npermissions:"));
  assert.match(triggers, /workflow_call:/);
  assert.doesNotMatch(triggers, /\n  (workflow_dispatch|push|pull_request|schedule|merge_group):/);
  assert.match(reusable, /permissions:\n  contents: read/);
});

test("tier2-ab jobs pin the isolated runners, physical GPU1, ceilings, Metal lock and cleanup", async () => {
  const reusable = await workflowText("tier2-ab.yml");
  const mlx = job(reusable, "mlx"), cuda = job(reusable, "cuda");
  assert.match(mlx, /runs-on: \[self-hosted, macOS, ARM64, rw-mage\]/);
  assert.doesNotMatch(mlx, /\b(nax|real-weights|rw-starvector|weights)\]/);
  assert.match(cuda, /runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/);
  for (const item of [mlx, cuda]) {
    assert.match(item, /timeout-minutes: [0-9]+/);
    assert.match(item, /cancel-in-progress: false/);
    assert.match(item, /--preflight-only/); assert.match(item, /--build-only/); assert.match(item, /--run/);
    assert.match(item, /if: \$\{\{ !inputs\.preflight_only \}\}/);
    assert.match(item, /actions\/upload-artifact@[0-9a-f]{40}/);
    assert.match(item, /name: Remove every scratch directory[\s\S]*if: \$\{\{ always\(\) \}\}/);
  }
  assert.match(mlx, /metal_execution_entry [0-9]+ node tier2-ab-driver\/scripts\/tier2-ab\.mjs --run/);
  for (const [key, value] of [["CUDA_VISIBLE_DEVICES", '"1"'], ["NVIDIA_VISIBLE_DEVICES", '"1"'], ["SCENEWORKS_GPU_ID", '"1"'], ["CUDA_DEVICE_ORDER", "PCI_BUS_ID"], ["RUSTC_WRAPPER", '""']]) assert.match(cuda, new RegExp(`\\n      ${key}: ${value}\\n`));
  assert.match(cuda, /python -I scripts\/ci\/windows_cuda_execution\.py --actor tier2-ab-cuda --budget-minutes [0-9]+ --cuda-ordinals 0 -- node scripts\/tier2-ab\.mjs --run/);
  assert.match(cuda, /Add-Content -Path \$env:GITHUB_ENV -Value 'RUSTC_WRAPPER='/);
  assert.match(cuda, /vcvars64\.bat/);
  assert.doesNotMatch(reusable, /download_missing|--download-missing|hf download|huggingface-cli/);
});
