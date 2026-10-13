#!/usr/bin/env node
// S9 product driver. A collected render is evidence, never Michael's quality decision.
import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, writeFile, mkdir, rename, open, stat, unlink } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { hostname } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { stripJsoncComments } from "./lib/jsonc.mjs";
import { collectComparison } from "./h3-veda-metrics.mjs";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const FRAMES = [124, 141, 243, 345];
export const CANVASES = ["1536x672", "672x1536", "1344x768", "768x1344", "1024x768", "768x1024", "768x768", "576x320", "320x576"];
export const PROFILES = { t2va: "minimax_h3", fl2va_first: "minimax_h3", fl2va_last: "minimax_h3", fl2va_both: "minimax_h3", r2va_mixed: "minimax_h3_ref" };
export const PROOFS = ["off_identity", "exact_frozen_masks", "kernel_reference_not_dense", "missing_predictor", "corrupt_predictor", "unsupported_kernel", "failed_self_test", "independent_budgets", "full_layer_step_controls", "global_reference_denominator", "persistence_replay", "managed_digest", "bounded_admission_peaks", "published_input_boundaries", "final_ci_refs_pin_tracker"];
const HEX = /^[a-f0-9]{64}$/;
const HEAD = /^[a-f0-9]{40}$/;
const ACTIVE = new Set(["queued", "pending_caption", "pending_workflow", "preparing", "downloading", "loading_model", "running", "saving"]);
const TERMINAL = new Set(["completed", "failed", "canceled", "interrupted"]);
const assert = (condition, message) => { if (!condition) throw new Error(`H3 S9: ${message}`); };
export function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((k) => [k, canonical(value[k])]));
  return value;
}
export const digest = (value) => createHash("sha256").update(typeof value === "string" || Buffer.isBuffer(value) ? value : JSON.stringify(canonical(value))).digest("hex");
export async function hashFile(file) { const hash = createHash("sha256"); for await (const chunk of createReadStream(file)) hash.update(chunk); return hash.digest("hex"); }
export async function boundFile(binding) {
  assert(binding && path.isAbsolute(binding.path) && HEX.test(binding.sha256), "unbound file path/hash");
  const s = await stat(binding.path);
  assert(s.isFile() && Number.isSafeInteger(binding.bytes) && s.size === binding.bytes, `file size/type mismatch: ${binding.path}`);
  assert(await hashFile(binding.path) === binding.sha256, `file digest mismatch: ${binding.path}`);
  return { path: binding.path, bytes: s.size, sha256: binding.sha256, mtimeMs: s.mtimeMs, ino: s.ino };
}
async function atomic(file, value) {
  await mkdir(path.dirname(file), { recursive: true });
  const temp = `${file}.${process.pid}.${randomUUID()}.tmp`;
  await writeFile(temp, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
  await rename(temp, file);
}
export async function catalogs(root = ROOT) {
  const load = async (name) => JSON.parse(stripJsoncComments(await readFile(path.join(root, "config/manifests", name), "utf8")));
  return { models: (await load("builtin.models.jsonc")).models.filter((m) => Object.values(PROFILES).includes(m.id)), loras: (await load("builtin.loras.jsonc")).loras.filter((l) => l.family === "minimax-h3" && l.role === "accelerator") };
}
export function matrix(catalog, styles = []) {
  assert(catalog.models.length === 2 && catalog.loras.length === 4, "shipped H3 catalog changed; reconcile required matrix");
  const regimes = [{ id: "base_50", sampling: { steps: 50, schedulerShift: 12, audioSchedulerShift: 3 } }, ...catalog.loras];
  const rows = [];
  for (const [profile, model] of Object.entries(PROFILES)) for (const recipe of regimes) for (const canvas of CANVASES) for (const backend of ["mlx", "candle_cuda"]) for (const tier of ["q4", "q8", "bf16"]) for (const frames of FRAMES) {
    const refusal = !!recipe.modelIds?.length && !recipe.modelIds.includes(model);
    const duration = catalog.models.find((m) => m.id === model)?.limits?.durations?.find((d) => Math.round(d * 24) === frames);
    assert(Number.isFinite(duration), `published duration missing: ${model}/${frames}`);
    const base = { duration, profile, model, recipe: recipe.id, canvas, backend, tier, frames, sampling: recipe.sampling, kind: refusal ? "expected_refusal" : "media", style: null };
    const add = (r) => rows.push({ ...r, id: [r.backend, r.tier, r.profile, r.recipe, r.canvas, r.frames, r.style ?? "none"].join("/") });
    add(base);
    // Compatible Turbo-plus-style is approved scope. No all14/carrier stress cross product.
    if (!refusal && recipe.id !== "base_50") for (const style of styles) if (style.models.includes(model)) add({ ...base, style: style.id });
  }
  return rows;
}
export function validateShape(m, catalog) {
  assert(m.schemaVersion === 1 && m.scope === "named-required" && /^[a-z0-9-]+$/.test(m.campaignId ?? ""), "unknown/unbound campaign schema or scope");
  assert(m.sourceFreeze?.stage === "permanent-inference-main-pin" && HEAD.test(m.sourceFreeze.inferencePin), "permanent inference main/pin freeze required");
  assert(m.projectId && m.projectId !== "UNBOUND", "project asset scope unbound");
  assert(m.proofRoutes && PROOFS.every((id) => ["product_job", "native_source_contract", "integration_receipt"].includes(m.proofRoutes[id]?.route) && m.proofRoutes[id]?.requirement), "required terminal proof route missing");
  assert(m.styles?.length > 0, "required compatible style corpus missing");
  assert(new Set(m.styles.map((s) => s.id)).size === m.styles.length, "duplicate style identities");
  for (const s of m.styles) {
    assert(s.id && s.source?.repo && HEAD.test(s.source.revision) && s.source.file && s.licenseEvidence && HEX.test(s.compatibilityEvidence?.sha256), "style provenance/compatibility unbound");
    assert(Number.isFinite(s.weight) && s.weight > 0 && s.models.length > 0 && s.models.every((id) => Object.values(PROFILES).includes(id)), "invalid style weight/model eligibility");
  }
  for (const model of new Set(Object.values(PROFILES))) assert(m.styles.some((s) => s.models.includes(model)), `style coverage missing for ${model}`);
  assert(m.artifacts?.length && m.assets && m.prompts && m.bindings, "model/carrier corpus unbound");
  assert(m.styles.every((s) => typeof s.trigger === "string" && !s.trigger.includes("UNBOUND")), "style trigger unbound");
  for (const profile of Object.keys(PROFILES)) {
    assert(m.prompts[profile]?.trim() && !m.prompts[profile].includes("UNBOUND"), `prompt unbound: ${profile}`);
    const b = m.bindings[profile]; assert(b && Number.isSafeInteger(b.seed), `seed/binding missing: ${profile}`);
    const first = !!b.first, last = !!b.last;
    assert(first === ["fl2va_first", "fl2va_both"].includes(profile) && last === ["fl2va_last", "fl2va_both"].includes(profile), `keyframe shape mismatch: ${profile}`);
    if (profile === "r2va_mixed") assert(b.images?.length && b.clips?.length && b.audio?.length && b.clips.some((id) => m.assets[id]?.hasAudio === true), "mixed R2VA image/clip soundtrack/standalone audio required");
    else assert(!b.images?.length && !b.clips?.length && !b.audio?.length, `unexpected reference carriers: ${profile}`);
    for (const id of [b.first, b.last, ...(b.images ?? []), ...(b.clips ?? []), ...(b.audio ?? [])].filter(Boolean)) assert(m.assets[id]?.assetId && m.assets[id].assetId !== "UNBOUND", `asset ID unbound: ${id}`);
  }
  const allowedIds = new Set(catalog.loras.map((l) => l.id));
  assert(m.turboArtifacts && Object.keys(m.turboArtifacts).length === allowedIds.size && Object.keys(m.turboArtifacts).every((id) => allowedIds.has(id)), "missing/unknown published Turbo artifact");
  assert(m.limits && Number.isSafeInteger(m.limits.jobTimeoutMs) && m.limits.jobTimeoutMs > 0 && Number.isSafeInteger(m.limits.maxJobs) && m.limits.maxJobs > 0 && Number.isSafeInteger(m.limits.maxMediaBytes) && m.limits.maxMediaBytes > 0, "bounded execution limits required");
  const rows = matrix(catalog, m.styles); assert(rows.filter((r) => !r.style).length === 5400, "named matrix coverage drift");
  return rows;
}
export async function inspectSources(m) {
  const git = (root, args) => execFileSync("git", ["-C", root, ...args], { encoding: "utf8", timeout: 10000 }).trim();
  for (const key of ["sceneworks", "inference"]) {
    const s = m.sourceFreeze[key]; assert(path.isAbsolute(s?.root ?? "") && HEAD.test(s.head), `source identity unbound: ${key}`);
    assert(git(s.root, ["rev-parse", "HEAD"]) === s.head && git(s.root, ["status", "--porcelain"]) === "", `dirty/stale ${key} source`);
  }
  const f = m.sourceFreeze;
  assert(f.inference.head === f.inferencePin && git(f.inference.root, ["rev-parse", "refs/remotes/origin/main"]) === f.inferencePin, "inference permanent main identity differs");
  const cargo = await readFile(path.join(f.sceneworks.root, "Cargo.toml"), "utf8");
  const pins = [...cargo.matchAll(/git\s*=\s*"https:\/\/github.com\/SceneWorks\/inference"\s*,\s*rev\s*=\s*"([a-f0-9]{40})"/g)].map((x) => x[1]);
  assert(pins.length && pins.every((p) => p === f.inferencePin), "SceneWorks actual inference pins stale");
  for (const [file, expected] of Object.entries(f.files ?? {})) assert(HEX.test(expected) && await hashFile(path.join(f.sceneworks.root, file)) === expected, `source file fingerprint differs: ${file}`);
  for (const file of ["config/manifests/builtin.models.jsonc", "config/manifests/builtin.loras.jsonc", "config/terminal-evidence/epic-25140-requirements.json", "scripts/h3-veda-acceptance.mjs", "scripts/h3-veda-metrics.mjs"]) assert(f.files?.[file], `required source fingerprint omitted: ${file}`);
}
export async function validateManifest(m, { catalog, sourceCheck = inspectSources } = {}) {
  catalog ??= await catalogs(m.sourceFreeze?.sceneworks?.root);
  const rows = validateShape(m, catalog);
  for (const p of m.productProofCases ?? []) {
    assert(PROOFS.includes(p.proof) && ["mlx", "candle_cuda"].includes(p.backend) && Object.keys(PROFILES).includes(p.profile), "unknown terminal product proof");
    const template = rows.find((r) => r.kind === "media" && !r.style && r.backend === p.backend && r.tier === p.tier && r.profile === p.profile && r.recipe === p.recipe && r.canvas === p.canvas && r.frames === p.frames);
    assert(template && ["sparse", "fallback_dense", "controlled_dense"].includes(p.expectation), "terminal product proof outside named configuration or expected outcome missing");
    assert(p.veda?.enabled === true && typeof p.veda.generated === "string" && typeof p.veda.reference === "string" && typeof p.veda.fullAttentionLayers === "string" && typeof p.veda.fullAttentionSteps === "string", "proof control binding incomplete");
    if (p.expectation === "fallback_dense") assert(p.fallbackReason?.trim() && p.setupReceipt?.sha256, "fallback cause/setup receipt unbound");
    rows.push({ ...template, ...p, kind: "product_proof", id: `proof/${p.proof}/${p.id}/${p.backend}` });
  }
  assert(new Set(rows.map((r) => r.id)).size === rows.length, "duplicate terminal case identity");
  await sourceCheck(m);
  const snapshots = [];
  for (const p of m.productProofCases ?? []) if (p.setupReceipt) {
    const binding = await boundFile(p.setupReceipt), setup = JSON.parse(await readFile(binding.path, "utf8"));
    assert(setup.fingerprint === digest(m.sourceFreeze) && setup.backend === p.backend && setup.proof === p.proof, "forced-condition setup provenance differs"); snapshots.push(binding);
  }
  for (const a of m.artifacts) {
    assert(a.id && a.repo && HEAD.test(a.revision) && a.file && a.coverage?.length, "model artifact identity incomplete");
    // A frozen catalog inventory binds every expected file, not a self-selected happy-path subset.
    snapshots.push(await boundFile(a));
  }
  assert(new Set(m.artifacts.map((a) => a.id)).size === m.artifacts.length, "duplicate model artifacts");
  assert(m.sourceFreeze.artifactInventorySha256 === m.artifactInventory.sha256, "complete artifact inventory absent from source freeze");
  const inventoryBinding = await boundFile(m.artifactInventory); snapshots.push(inventoryBinding);
  const inventory = JSON.parse(await readFile(inventoryBinding.path, "utf8"));
  assert(inventory.catalogSha256 === m.sourceFreeze.files["config/manifests/builtin.models.jsonc"] && Array.isArray(inventory.files) && inventory.files.length === m.artifacts.length, "artifact inventory/catalog set differs");
  assert(digest(inventory.files) === digest(m.artifacts.map(({ path: unused, coverage, ...a }) => a)), "frozen artifact inventory membership differs");
  for (const backend of ["mlx", "candle_cuda"]) for (const tier of ["q4", "q8", "bf16"]) for (const model of new Set(Object.values(PROFILES))) for (const component of ["dit", "text_encoder", "video_vae", "audio_vae", "predictor"]) assert(m.artifacts.some((a) => a.coverage.some((c) => c.backend === backend && c.tier === tier && c.model === model && c.component === component)), `missing model component: ${backend}/${tier}/${model}/${component}`);
  for (const a of m.artifacts) {
    const matches = catalog.models.flatMap((model) => model.downloads ?? []).some((d) => d.repo === a.repo && d.revision === a.revision && (d.files ?? []).some((pattern) => new RegExp("^" + pattern.split("*").map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join(".*") + "$").test(a.file)));
    assert(matches, `artifact revision absent from frozen catalog: ${a.id}`);
  }
  for (const [id, a] of Object.entries(m.turboArtifacts)) {
    const l = catalog.loras.find((l) => l.id === id); assert(a.repo === l.source.repo && a.revision === l.source.revision && a.file === l.source.file && digest(a.modelIds) === digest(l.modelIds), `Turbo source changed: ${id}`); snapshots.push(await boundFile(a));
  }
  for (const s of m.styles) { snapshots.push(await boundFile(s)); snapshots.push(await boundFile(s.compatibilityEvidence)); }
  for (const a of Object.values(m.assets)) { snapshots.push(await boundFile(a)); snapshots.push(await boundFile(a.normalized)); }
  for (const backend of ["mlx", "candle_cuda"]) {
    const r = m.runners?.[backend]; assert(r?.physicalGuardAdapter && m.sourceFreeze.resourceContract?.[backend]?.canonicalLocks?.length, `physical guard adapter/contract unavailable: ${backend}`); snapshots.push(await boundFile(r.physicalGuardAdapter)); assert(r && new URL(r.apiUrl).protocol.match(/^https?:$/) && r.requestedGpu && r.workerId && r.reportedBackend && r.kernel && r.physicalGpuId, `runner identity missing: ${backend}`);
    snapshots.push(await boundFile(r.apiBinary)); snapshots.push(await boundFile(r.workerBinary));
    const attestation = JSON.parse(await readFile((await boundFile(r.attestation)).path, "utf8"));
    assert(attestation.inferencePin === m.sourceFreeze.inferencePin && attestation.sceneworksHead === m.sourceFreeze.sceneworks.head && attestation.apiBinarySha256 === r.apiBinary.sha256 && attestation.workerBinarySha256 === r.workerBinary.sha256 && attestation.workerId === r.workerId, `installed product provenance differs: ${backend}`);
  }
  return { rows, catalog, snapshots, fingerprint: digest(m) };
}
async function unchanged(snapshots) { for (const b of snapshots) { const s = await stat(b.path); assert(s.size === b.bytes && s.mtimeMs === b.mtimeMs && s.ino === b.ino, `artifact/carrier changed during campaign: ${b.path}`); } }
export function requestFor(m, row, arm) {
  const b = m.bindings[row.profile], [width, height] = row.canvas.split("x").map(Number);
  const recipe = m.turboArtifacts[row.recipe]; const style = m.styles.find((s) => s.id === row.style);
  const loras = [recipe && { id: row.recipe, family: "minimax-h3", modelIds: recipe.modelIds, path: recipe.path, weight: 1, file: recipe.file }, style && { id: style.id, family: "minimax-h3", path: style.path, weight: style.weight, file: style.source.file }].filter(Boolean);
  const id = (key) => m.assets[key].assetId;
  const request = { projectId: m.projectId, model: row.model, mode: row.profile === "t2va" ? "text_to_video" : row.profile === "r2va_mixed" ? "reference_to_video" : row.profile === "fl2va_both" ? "first_last_frame" : "image_to_video", prompt: m.prompts[row.profile] + (style ? ` ${style.trigger}` : ""), seed: b.seed, duration: row.duration, fps: 24, width, height, requestedGpu: m.runners[row.backend].requestedGpu, loras, fitMode: "crop", advanced: { steps: row.sampling.steps, mlxQuantize: row.tier === "bf16" ? 0 : Number(row.tier.slice(1)), schedulerShift: row.sampling.schedulerShift } };
  if (b.first) request.sourceAssetId = id(b.first); if (b.last) request.lastFrameAssetId = id(b.last);
  for (const [field, list] of [["referenceAssetIds", b.images], ["sourceClipAssetIds", b.clips], ["referenceAudioAssetIds", b.audio]]) if (list?.length) request[field] = list.map(id);
  if (arm === "sparse") request.advanced.veda = row.veda ?? { enabled: true, generated: "90%", reference: "90%", fullAttentionLayers: "", fullAttentionSteps: "" };
  return request;
}
export class ProductApi {
  constructor(url, token, fetchImpl = fetch) { this.base = new URL(url); this.token = token; this.fetch = fetchImpl; }
  async json(route, { method = "GET", body } = {}) {
    const response = await this.fetch(new URL(route, this.base), { method, headers: { "Content-Type": "application/json", ...(this.token ? { "X-SceneWorks-Token": this.token } : {}) }, body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(30000) });
    const value = await response.json(); return { status: response.status, value };
  }
  async verifyCarrier(asset, binding, maxBytes) {
    const url = new URL(asset.url, this.base);
    assert(url.origin === this.base.origin && url.pathname.startsWith(`/api/v1/projects/${encodeURIComponent(binding.projectId)}/files/`), "carrier media URL outside bound project");
    const r = await this.fetch(url, { headers: this.token ? { "X-SceneWorks-Token": this.token } : {}, redirect: "error", signal: AbortSignal.timeout(300000) });
    assert(r.ok && r.body, "carrier media unavailable"); let bytes = 0; const hash = createHash("sha256");
    for await (const chunk of r.body) { bytes += chunk.length; assert(bytes <= maxBytes && bytes <= binding.bytes, "carrier size differs"); hash.update(chunk); }
    assert(bytes === binding.bytes && hash.digest("hex") === binding.sha256, "product carrier bytes differ from corpus");
  }
  async download(asset, target, maxBytes) {
    const url = new URL(asset.url, this.base); assert(url.origin === this.base.origin && url.pathname.startsWith(`/api/v1/projects/`), "output media URL outside product origin");
    const response = await this.fetch(url, { headers: this.token ? { "X-SceneWorks-Token": this.token } : {}, signal: AbortSignal.timeout(300000), redirect: "error" });
    assert(response.ok && response.body, "media fetch failed"); const file = await open(target, "wx"); let count = 0;
    try { for await (const chunk of response.body) { count += chunk.length; assert(count <= maxBytes, "media byte limit exceeded"); await file.write(chunk); } } finally { await file.close(); }
    return { path: target, bytes: count, sha256: await hashFile(target) };
  }
}
export async function reconcile(api, intent) {
  const r = await api.json(`/api/v1/jobs?projectId=${encodeURIComponent(intent.request.projectId)}&limit=500`);
  assert(r.status === 200 && Array.isArray(r.value) && r.value.length < 500, "ambiguous reconciliation window; no blind resubmission");
  const found = r.value.filter((j) => j.payload?.advanced?.s9Harness?.key === intent.key);
  assert(found.length <= 1, "duplicate enqueue marker; manual resolution required");
  if (found.length) { assert(digest(found[0].payload.advanced.s9Harness) === digest(intent.request.advanced.s9Harness), "enqueue identity marker differs"); return found[0]; }
  throw new Error("H3 S9: ambiguous submission not found; do not resubmit (cleared/older jobs may be absent)");
}
export function observed(job, metrics, row, arm, runner) {
  assert(job.status === "completed" && job.workerId === runner.workerId, "job not completed on frozen worker");
  const assets = job.result?.assets; assert(Array.isArray(assets) && assets.length === 1, "actual product output asset missing/count differs");
  const asset = assets[0]; assert(asset.lineage?.jobId === job.id && asset.file?.width === Number(row.canvas.split("x")[0]) && asset.file?.height === Number(row.canvas.split("x")[1]) && asset.file?.frameCount === row.frames && asset.file?.fps === 24 && asset.file?.hasAudio === true, "actual output geometry/audio/lineage differs");
  const report = asset.recipe?.rawAdapterSettings?.vedaExecution;
  assert(report && metrics?.vedaExecution && digest(report) === digest(metrics.vedaExecution), "missing/disagreeing native observed Veda outcome");
  assert(metrics.model === row.model && metrics.quantLabel === row.tier && metrics.width === asset.file.width && metrics.height === asset.file.height && metrics.steps === row.sampling.steps && metrics.backend === runner.reportedBackend, "actual metrics tier/schedule/backend identity differs");
  assert(Number.isFinite(metrics.totalMs) && metrics.totalMs > 0 && Number.isFinite(metrics.peakMemoryBytes) && metrics.peakMemoryBytes > 0, "actual timing/peak metrics absent");
  assert(report.denseTokenPairs > 0 && Number.isFinite(report.totalAttentionFraction) && report.totalAttentionFraction > 0 && report.totalAttentionFraction <= 1, "attention denominator unobserved");
  if (arm === "dense") assert(report.requested === false && report.effective === "dense" && report.sparseCalls === 0 && report.denseCalls > 0 && report.totalAttentionFraction === 1 && report.predictorLoadAttempts === 0 && report.workspaceBytes === 0 && !report.fallbackReason, "off path not observed unchanged dense");
  else if (row.expectation === "controlled_dense") assert(report.requested === true && report.effective === "dense" && report.sparseCalls === 0 && report.denseCalls > 0 && report.totalAttentionFraction === 1 && !report.fallbackReason, "forced full-attention controls not observed dense");
  else if (row.expectation === "fallback_dense") assert(report.requested === true && report.effective === "dense" && report.sparseCalls === 0 && report.denseCalls > 0 && typeof row.fallbackReason === "string" && row.fallbackReason.length > 0 && report.fallbackReason === row.fallbackReason, "expected explicit dense fallback not observed");
  else assert(report.requested === true && ["sparse", "mixed"].includes(report.effective) && report.sparseCalls > 0 && report.predictorScoreCalls > 0 && report.kernel === runner.kernel && report.plan && ["exact", "nearest"].includes(report.planMatch) && !report.fallbackReason, "enabled request did not execute learned sparse (fallback is unresolved)");
  return asset;
}
// This adapter must hold the established physical OS lock; campaign.json/marker is not that lock.
export async function physicalPhase(guard, m, backend, phase, { now } = {}) {
  const r = m.runners[backend], expected = m.sourceFreeze.resourceContract?.[backend];
  assert(expected?.canonicalLocks?.length && expected.physicalGpuId === r.physicalGpuId && digest(guard.lockIdentities) === digest(expected.canonicalLocks), "wrong canonical physical resource lock");
  assert(guard.owned === true && guard.ownerToken && await guard.isOwnerAlive() === true && await guard.ownsCanonicalLocks() === true, "unowned/dead physical guard owner");
  const proof = await guard.beforePhase(phase); now ??= Date.now();
  assert(proof.phase === phase && proof.ownerToken === guard.ownerToken && proof.admitted === true && proof.physicalGpuId === r.physicalGpuId && proof.observedAt <= now && now - proof.observedAt <= 10000 && proof.conflictingJobs === 0 && proof.queueObservedAt <= now && now - proof.queueObservedAt <= 10000, "stale/unadmitted current phase or conflicting queue");
  assert(Array.isArray(proof.driverSamples) && proof.driverSamples.length === 5 && proof.driverSamples.every((x, i, a) => x.admitted === true && x.driverAvailable === true && Number.isFinite(x.observedAt) && x.observedAt <= now && now - x.observedAt <= 15000 && (!i || x.observedAt > a[i - 1].observedAt)), "five timed driver admission samples missing/stale");
  return proof;
}
export async function acquirePhysicalGuard(m, backend, runDir) {
  const binding = m.runners[backend].physicalGuardAdapter;
  assert(binding, "physical guard adapter unavailable; campaign marker cannot authorize GPU use");
  const source = await boundFile(binding), root = m.sourceFreeze.sceneworks.root;
  const relative = path.relative(root, source.path);
  assert(relative && !relative.startsWith("..") && m.sourceFreeze.files[relative] === source.sha256, "physical guard adapter not bound to frozen current source");
  const adapter = await import(pathToFileURL(source.path).href);
  assert(adapter.protocol === "epic-25140-physical-guard-v1" && typeof adapter.acquire === "function", "actual physical guard adapter unavailable");
  const guard = await adapter.acquire({ manifest: m, backend, runDir });
  try { await physicalPhase(guard, m, backend, "campaign-start"); } catch (error) { if (guard?.release) await guard.release(); throw error; }
  assert(typeof guard.release === "function" && typeof guard.retainForOwnedJobs === "function" && typeof guard.submit === "function" && typeof guard.attachJob === "function", "physical guard release/owned-job retention adapter missing"); return guard;
}
export async function claimLease(m, backend, runDir) {
  const r = m.runners[backend]; assert(path.isAbsolute(r.leasePath), "campaign resume marker path required");
  await mkdir(path.dirname(r.leasePath), { recursive: true }); let handle, takeover;
  try { handle = await open(r.leasePath, "wx"); } catch (error) {
    if (error.code !== "EEXIST") throw error;
    const prior = JSON.parse(await readFile(r.leasePath, "utf8"));
    assert(prior.campaignId === m.campaignId && prior.runDir === runDir && prior.physicalGpuId === r.physicalGpuId && prior.host === hostname() && prior.fingerprint === digest(m), "foreign/stale campaign GPU lease");
    let alive = true; try { process.kill(prior.pid, 0); } catch (e) { assert(e.code === "ESRCH", "cannot prove prior lease owner exit"); alive = false; }
    assert(!alive, "campaign GPU lease owner still alive");
    takeover = await open(`${r.leasePath}.takeover`, "wx");
    try { const current = JSON.parse(await readFile(r.leasePath, "utf8")); assert(current.token === prior.token, "lease changed during owned resume"); handle = await open(r.leasePath, "w"); } catch (error) { await takeover.close(); await unlink(`${r.leasePath}.takeover`); throw error; }
  }
  const owner = { campaignId: m.campaignId, backend, physicalGpuId: r.physicalGpuId, host: hostname(), pid: process.pid, runDir, token: randomUUID(), fingerprint: digest(m), deadline: Date.now() + m.limits.jobTimeoutMs };
  await handle.writeFile(JSON.stringify(owner)); await handle.close();
  if (takeover) { await takeover.close(); await unlink(`${r.leasePath}.takeover`); }
  const check = async () => { const actual = JSON.parse(await readFile(r.leasePath, "utf8")); assert(actual.token === owner.token, "lease ownership changed; do not release foreign lease"); };
  const release = async () => { await check(); await unlink(r.leasePath); }; release.check = check; return release;
}
export async function runCampaign(m, { backend, runDir, validate = validateManifest, api = new ProductApi(m.runners?.[backend]?.apiUrl, process.env.SCENEWORKS_TOKEN), lease = claimLease, physical = acquirePhysicalGuard, sleep = delay, compare = collectComparison, sourceCheck = inspectSources } = {}) {
  assert(["mlx", "candle_cuda"].includes(backend) && path.isAbsolute(runDir), "explicit backend and external output directory required");
  assert(!runDir.startsWith(`${m.sourceFreeze.sceneworks.root}/`) && !runDir.startsWith(`${m.sourceFreeze.inference.root}/`), "receipts must stay outside frozen source repos");
  const v = await validate(m); await mkdir(runDir, { recursive: true });
  const campaignFile = path.join(runDir, "campaign.json"); let state;
  try { state = JSON.parse(await readFile(campaignFile, "utf8")); assert(state.fingerprint === v.fingerprint, "resume input/source/artifact identity differs"); } catch (e) { if (e.code !== "ENOENT") throw e; state = { fingerprint: v.fingerprint, campaignId: m.campaignId, namedCells: 5400, optionalStressExecuted: 0, cells: {}, qualityAcceptance: "pending Michael", completedAcceptance: false }; }
  const guard = await physical(m, backend, runDir); let release;
  try { release = await lease(m, backend, runDir); } catch (error) { await guard.release(); throw error; }
  let submitted = 0;
  try {
    for (const row of v.rows.filter((r) => r.backend === backend)) {
      let cell = state.cells[row.id];
      if (["media_collected_review_pending", "media_collected_evidence_incomplete", "product_proof_observed"].includes(cell?.status) || cell?.status === "expected_refusal_observed") {
        for (const arm of Object.values(cell.arms ?? {})) if (arm.media) await boundFile(arm.media);
        if (cell.comparison) for (const file of [cell.comparison.sideBySide, cell.comparison.denseAudio, cell.comparison.sparseAudio]) await boundFile(file);
        continue;
      }
      if (submitted >= m.limits.maxJobs) break;
      await sourceCheck(m); await unchanged(v.snapshots); await physicalPhase(guard, m, backend, "prepare-carriers");
      for (const asset of Object.values(m.assets)) {
        const r = await api.json(`/api/v1/projects/${encodeURIComponent(m.projectId)}/assets/${encodeURIComponent(asset.assetId)}`);
        assert(r.status === 200 && r.value.id === asset.assetId, "product carrier asset missing/stale");
        await api.verifyCarrier(r.value, { ...asset, projectId: m.projectId }, m.limits.maxMediaBytes);
      }
      cell ??= { kind: row.kind, status: "planned", arms: {} }; state.cells[row.id] = cell;
      for (const arm of row.kind === "expected_refusal" ? ["refusal"] : row.kind === "product_proof" ? ["sparse"] : ["dense", "sparse"]) {
        if (submitted >= m.limits.maxJobs) break;
        await sourceCheck(m); await unchanged(v.snapshots); if (release.check) await release.check(); await physicalPhase(guard, m, backend, "submit-or-reconcile");
        let attempt = cell.arms[arm];
        if (attempt?.status === "collected") { await boundFile(attempt.media); continue; }
        if (attempt?.status === "expected_refusal") continue;
        const request = requestFor(m, row, arm); const key = digest({ fingerprint: v.fingerprint, row: row.id, arm, request });
        request.advanced.s9Harness = { campaignId: m.campaignId, caseId: row.id, arm, key, fingerprint: v.fingerprint };
        if (!attempt) { attempt = { key, request, requestSha256: digest(request), status: "intent", deadline: Date.now() + m.limits.jobTimeoutMs }; cell.arms[arm] = attempt; await atomic(campaignFile, state); }
        assert(attempt.key === key && attempt.requestSha256 === digest(request), "persisted arm request identity differs");
        let job;
        if (attempt.jobId) { const r = await api.json(`/api/v1/jobs/${encodeURIComponent(attempt.jobId)}`); assert(r.status === 200, "recorded job not readable; do not resubmit"); job = r.value; }
        else if (attempt.status === "submitting" || attempt.status === "ambiguous") { job = await reconcile(api, attempt); attempt.jobId = job.id; await atomic(campaignFile, state); await guard.attachJob(attempt); }
        else {
          assert(attempt.status === "intent", "failed attempt requires explicit human resolution; no automatic rerender");
          attempt.status = "submitting"; await atomic(campaignFile, state); submitted++;
          let r; try { r = await guard.submit(attempt); } catch (error) { attempt.status = "ambiguous"; attempt.error = String(error); await atomic(campaignFile, state); throw error; }
          if (row.kind === "expected_refusal") {
            const text = JSON.stringify(r.value); assert([400, 422].includes(r.status) && text.includes(row.recipe) && /declared.*not|not.*compatible/i.test(text), "expected modelIds refusal absent or unrelated error"); attempt.status = "expected_refusal"; attempt.response = r; cell.status = "expected_refusal_observed"; await atomic(campaignFile, state); continue;
          }
          assert(r.status === 201 && r.value.id, `product enqueue failed: HTTP ${r.status}`); job = r.value; attempt.jobId = job.id; attempt.status = "enqueued"; await atomic(campaignFile, state); await guard.attachJob(attempt);
        }
        assert(job.payload?.advanced?.s9Harness?.key === key && job.projectId === m.projectId, "product job request readback differs");
        for (const field of ["model", "mode", "prompt", "seed", "width", "height", "duration", "fps", "sourceAssetId", "lastFrameAssetId", "referenceAssetIds", "sourceClipAssetIds", "referenceAudioAssetIds"]) assert(digest(job.payload[field] ?? null) === digest(request[field] ?? null), `actual request readback differs: ${field}`);
        assert(digest(job.payload.advanced?.veda ?? null) === digest(request.advanced.veda ?? null), "actual request Veda controls differ");
        attempt.job = job; await atomic(campaignFile, state);
        while (ACTIVE.has(job.status)) {
          if (Date.now() >= attempt.deadline) { attempt.status = "timeout_owned_job_active"; attempt.lastJob = job; await atomic(campaignFile, state);
            const cancel = await api.json(`/api/v1/jobs/${encodeURIComponent(attempt.jobId)}/cancel`, { method: "POST", body: {} }); attempt.cancelResponse = cancel;
            for (let i = 0; i < 12; i++) { await sleep(5000); const r = await api.json(`/api/v1/jobs/${encodeURIComponent(attempt.jobId)}`); if (r.status === 200) { attempt.job = r.value; if (TERMINAL.has(r.value.status)) { attempt.status = "timeout_cancelled_read_back"; break; } } }
            await atomic(campaignFile, state); throw new Error("H3 S9: owned job timeout; cancellation/readback receipt retained"); }
          await sleep(5000); const r = await api.json(`/api/v1/jobs/${encodeURIComponent(attempt.jobId)}`); assert(r.status === 200, "owned job poll failed; preserve identity"); job = r.value; attempt.job = job; await atomic(campaignFile, state);
        }
        assert(TERMINAL.has(job.status), "unknown product job status"); attempt.job = job;
        if (job.status !== "completed") { attempt.status = job.status; cell.status = job.status; await atomic(campaignFile, state); continue; }
        await physicalPhase(guard, m, backend, "collect-completed-evidence");
        const metricsResult = await api.json(`/api/v1/jobs/${encodeURIComponent(job.id)}/metrics`); assert(metricsResult.status === 200, "metrics readback failed"); attempt.metrics = metricsResult.value;
        try {
          const asset = observed(job, attempt.metrics, row, arm, m.runners[backend]);
          const reread = await api.json(`/api/v1/projects/${encodeURIComponent(m.projectId)}/assets/${encodeURIComponent(asset.id)}`); assert(reread.status === 200 && digest(reread.value.recipe?.rawAdapterSettings?.vedaExecution) === digest(asset.recipe.rawAdapterSettings.vedaExecution), "persisted asset observed outcome differs");
          const dir = path.join(runDir, key); await mkdir(dir, { recursive: true });
          attempt.media = await api.download(reread.value, path.join(dir, `${arm}.mp4`), m.limits.maxMediaBytes); attempt.asset = reread.value; attempt.status = "collected"; if (row.kind === "product_proof") cell.status = "product_proof_observed";
        } catch (error) { attempt.status = "evidence_gap_or_fallback"; attempt.error = String(error); cell.status = "unresolved"; await atomic(campaignFile, state); continue; }
        await atomic(campaignFile, state);
      }
      if (cell.arms.dense?.status === "collected" && cell.arms.sparse?.status === "collected") {
        cell.comparison = await compare(cell.arms.dense.media, cell.arms.sparse.media, row, path.join(runDir, digest(row.id)), m.limits.jobTimeoutMs);
        cell.review = { reviewer: "Michael", disposition: "pending" }; cell.status = "media_collected_evidence_incomplete";
        // Product metrics expose only a combined peak and phase timing. Preserve missing instrumentation.
        cell.nativeMeasurementGaps = ["per-step timing", "separate host/device peaks", "complete sparse bound versus observed peak"];
      }
      await atomic(campaignFile, state);
    }
  } finally {
    // An ambiguous enqueue or active timeout keeps the lease for reconciliation/cancel. Never free a GPU that may still be rendering.
    const uncertain = Object.values(state.cells).some((c) => Object.values(c.arms).some((a) => ["submitting", "ambiguous", "timeout_owned_job_active"].includes(a.status) || (a.jobId && !TERMINAL.has(a.job?.status))));
    if (!uncertain) { await release(); await guard.release(); }
    else { assert(typeof guard.retainForOwnedJobs === "function" && typeof guard.submit === "function" && typeof guard.attachJob === "function", "physical guard cannot retain uncertain owned jobs"); await guard.retainForOwnedJobs(Object.values(state.cells).flatMap((c) => Object.values(c.arms)).filter((a) => ["submitting", "ambiguous", "timeout_owned_job_active"].includes(a.status) || (a.jobId && !TERMINAL.has(a.job?.status))).map((a) => ({ jobId: a.jobId ?? null, key: a.key, requestSha256: a.requestSha256 }))); }
    await atomic(campaignFile, state);
  }
  return state;
}
export function summary(state) {
  const cells = Object.values(state.cells), count = (status) => cells.filter((c) => c.status === status).length;
  return { namedCells: 5400, knownCells: cells.length, expectedRefusalsObserved: count("expected_refusal_observed"), mediaCollectedPendingMichael: count("media_collected_review_pending") + count("media_collected_evidence_incomplete"), instrumentationIncomplete: count("media_collected_evidence_incomplete"), productProofsObserved: count("product_proof_observed"), unresolved: count("unresolved"), qualityAcceptance: "pending Michael", completedAcceptance: false };
}
async function main() {
  const [command, manifestPath, ...args] = process.argv.slice(2);
  const m = JSON.parse(await readFile(manifestPath, "utf8"));
  if (command === "validate") { const v = await validateManifest(m); console.log(JSON.stringify({ fingerprint: v.fingerprint, cells: v.rows.length, named: 5400, expectedRefusals: v.rows.filter((r) => r.kind === "expected_refusal").length, acceptance: false })); return; }
  assert(command === "run" && args.length === 2, "usage: validate MANIFEST | run MANIFEST BACKEND EXTERNAL_RUN_DIR");
  console.log(JSON.stringify(summary(await runCampaign(m, { backend: args[0], runDir: path.resolve(args[1]) }))));
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((error) => { console.error(error.message); process.exitCode = 1; });
