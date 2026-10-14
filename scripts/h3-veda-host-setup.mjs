// CPU-only host discovery and synthetic corpus staging. Never launches the product or guardian.
import { execFileSync } from "node:child_process";
import { readFile, writeFile, mkdir, lstat, stat, realpath } from "node:fs/promises";
import path from "node:path";
import { ROOT, digest, hashFile, matrix, catalogs, PROFILES, PROOFS } from "./h3-veda-acceptance.mjs";

const TEMPLATE = "config/terminal-evidence/epic-25140-host-template.json";
const PLAN = "config/terminal-evidence/epic-25140-corpus.plan.json";
const BUNDLE = "config/terminal-evidence/epic-25140-carriers.json";
const SHA = /^[a-f0-9]{40}$/;
const check = (ok, why) => { if (!ok) throw new Error(`H3 host setup: ${why}`); };
const inside = (root, file) => { const r = path.relative(root, file); return r && !r.startsWith(`..${path.sep}`) && r !== ".." && !path.isAbsolute(r); };
const json = async (file) => JSON.parse(await readFile(file, "utf8"));
const command = (argv) => execFileSync(argv[0], argv.slice(1), { encoding: "utf8", timeout: 15000, maxBuffer: 256 * 1024, stdio: ["ignore", "pipe", "pipe"] });

export function assertSetupSource({ backend, runnerName, expectedRunner, sceneWorksSha, inferenceSha, root = ROOT, inferenceRoot = path.resolve(root, "../inference"), exec = command }) {
  check(["mlx", "candle_cuda"].includes(backend), "explicit backend required");
  check(runnerName === expectedRunner && (backend === "mlx" ? runnerName === "nax-macos-2" : ["cuda-windows", "cuda-windows-2"].includes(runnerName)), "exact named physical runner required");
  check(SHA.test(sceneWorksSha ?? "") && SHA.test(inferenceSha ?? ""), "exact current source heads required");
  for (const [dir, head] of [[root, sceneWorksSha], [inferenceRoot, inferenceSha]]) {
    check(exec(["git", "-C", dir, "rev-parse", "HEAD"]).trim() === head && exec(["git", "-C", dir, "status", "--porcelain"]).trim() === "", "source dirty/stale");
  }
  // Inventory observes current source, without pretending that a feature revision is permanent main.
  return { sceneworks: { root, head: sceneWorksSha }, inference: { root: inferenceRoot, head: inferenceSha } };
}

export function physicalMetadata(backend, { platform = process.platform, exec = command } = {}) {
  if (backend === "mlx") {
    check(platform === "darwin", "Mac inventory must run on the Mac");
    const hardware = JSON.parse(exec(["system_profiler", "SPHardwareDataType", "-json"])).SPHardwareDataType;
    const displays = JSON.parse(exec(["system_profiler", "SPDisplaysDataType", "-json"])).SPDisplaysDataType;
    check(hardware?.length === 1 && /^[a-f0-9-]{36}$/i.test(hardware[0].platform_UUID ?? ""), "actual Mac platform UUID absent");
    const devices = displays?.filter((d) => d.sppci_model && d.spdisplays_mtlgpufamilysupport);
    check(devices?.length === 1, "actual Metal device absent/ambiguous");
    check(hardware[0].platform_UUID !== "D299B954-4C63-5322-86A5-BD5667051634", "Mac2 must differ from observed primary Mac");
    return { physicalGpuId: hardware[0].platform_UUID, metalDeviceName: devices[0].sppci_model, metalSupport: devices[0].spdisplays_mtlgpufamilysupport, admission: false };
  }
  check(backend === "candle_cuda" && platform === "win32", "CUDA inventory must run on Windows");
  const rows = exec(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader,nounits"]).trim().split(/\r?\n/).map((line) => line.split(",").map((v) => v.trim()));
  check(rows.length > 0 && rows.every((r) => r.length === 5 && /^\d+$/.test(r[0]) && /^GPU-[a-f0-9-]+$/i.test(r[1]) && r[2] && r[3] && /^\d+$/.test(r[4])) && new Set(rows.map((r) => r[1])).size === rows.length, "physical CUDA metadata malformed/ambiguous");
  return { devices: rows.map(([index, uuid, name, driver, memoryMiB]) => ({ index, uuid, name, driver, memoryMiB: Number(memoryMiB) })), admission: false };
}

export function declaredCacheRoots(template, backend, env = process.env) {
  return [...new Set(template.cacheRootCandidates[backend].map((value) => value.replace(/\$\{([A-Z_]+)\}/g, (_, name) => env[name] ?? "UNBOUND")).filter((value) => !value.includes("UNBOUND")).map((value) => path.resolve(value)))];
}

export function snapshotPath(cacheRoot, a) {
  check(/^[\w.-]+\/[\w.-]+$/.test(a.repo) && SHA.test(a.revision) && typeof a.file === "string" && !path.isAbsolute(a.file) && !a.file.split(/[\\/]/).includes(".."), "unsafe declared artifact identity");
  return path.join(cacheRoot, "hub", `models--${a.repo.replaceAll("/", "--")}`, "snapshots", a.revision, ...a.file.split("/"));
}

export async function cacheMetadata(template, backend, roots) {
  const artifacts = [...template.artifacts.filter((a) => a.coverage.some((c) => c.backend === backend)), ...Object.values(template.turboArtifacts), ...template.styles.map((s) => ({ ...s, ...s.source }))];
  const results = [];
  for (const root of roots) {
    const info = await lstat(root).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e));
    const result = { root, exists: !!info, safeDirectory: !!info?.isDirectory() && !info.isSymbolicLink(), files: [], contentVerified: false };
    if (result.safeDirectory) for (const a of artifacts) {
      const file = snapshotPath(root, a), entry = await lstat(file).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e));
      let bytes = null, confined = false;
      if (entry) { const resolved = await realpath(file); confined = inside(root, resolved); if (confined) { const s = await stat(file); if (s.isFile()) bytes = s.size; } }
      result.files.push({ id: a.id ?? `${a.repo}@${a.revision}/${a.file}`, repo: a.repo, revision: a.revision, file: a.file, path: file, exists: !!entry, confined, expectedBytes: a.bytes, observedBytes: bytes, sizeMatches: confined && bytes === a.bytes, expectedSha256: a.sha256, expectedGitBlob: a.gitBlob ?? null, contentVerified: false });
    }
    results.push(result);
  }
  return results;
}

export async function inventory({ backend, runnerName, expectedRunner, sceneWorksSha, inferenceSha, root = ROOT, env = process.env, authority, sourceCheck = assertSetupSource, probe = physicalMetadata }) {
  const source = sourceCheck({ backend, runnerName, expectedRunner, sceneWorksSha, inferenceSha, root });
  check(typeof authority === "function", "live Actions read-authority probe required");
  const owner = authority({ backend, runnerName, sceneWorksSha });
  check(owner?.runnerName === runnerName && owner.ownRunReadable === true && owner.ownRunnerIdentityVerified === true && (
    (owner.resourceScope === "same_repository_actions_identity_only" && owner.resourceQueueVerified === false && owner.readable !== true) ||
    (owner.resourceScope === "cross_repository_runner_and_queue_read" && owner.resourceQueueVerified === true && owner.readable === true)
  ), "live Actions ownership unavailable");
  const template = await json(path.join(root, TEMPLATE));
  const physical = probe(backend);
  const cache = await cacheMetadata(template, backend, declaredCacheRoots(template, backend, env));
  const outputRoot = path.join(path.dirname(root), ".h3-veda-epic-25140", sceneWorksSha, runnerName);
  return { schemaVersion: 1, kind: "epic25140_cpu_host_inventory", observedAt: Date.now(), backend, runnerName, source, owner, physical, cache, ownedStageRoot: outputRoot, templateSha256: await hashFile(path.join(root, TEMPLATE)), planSha256: await hashFile(path.join(root, PLAN)), campaignReady: false, physicalAdmission: false, qualityAcceptance: "pending Michael", metadataOnly: true };
}

const SOURCE_FILES = [TEMPLATE, PLAN, BUNDLE, "config/manifests/builtin.models.jsonc", "config/manifests/builtin.loras.jsonc", "config/terminal-evidence/epic-25140-requirements.json", ".github/workflows/h3-veda-terminal.yml", "scripts/h3-veda-host-setup.mjs", "scripts/h3-veda-actions.mjs", "scripts/h3-veda-acceptance.mjs", "scripts/h3-veda-corpus.mjs", "scripts/h3-veda-metrics.mjs", "scripts/h3-veda-guard-adapter.mjs", "scripts/h3-veda-guardian.py", "scripts/h3-veda-guardian-operations.py", "apps/rust-api/src/lib.rs", "crates/sceneworks-worker/src/video_jobs/minimax_h3.rs"];
const INFERENCE_FILES = ["scripts/release/veda_cuda_guard.py", "scripts/media_campaign_windows.py", "crates/media/mlx-gen/mlx-gen-minimax-h3/src/veda/mod.rs", "crates/kernels/candle-quant-kernels/src/veda_int8.rs"];

export async function stage({ observation, runDir, corpusRoot, cacheRoot, root = ROOT, sourceCheck = assertSetupSource }) {
  check(observation?.kind === "epic25140_cpu_host_inventory" && observation.metadataOnly === true && observation.campaignReady === false && Date.now() - observation.observedAt <= 60000 && observation.observedAt <= Date.now(), "fresh actual host inventory required");
  check(observation.owner?.readable === true && observation.owner.ownRunReadable === true && observation.owner.ownRunnerIdentityVerified === true && observation.owner.resourceQueueVerified === true && observation.owner.resourceScope === "cross_repository_runner_and_queue_read" && observation.owner.runnerName === observation.runnerName, "stage requires full live Actions and resource-queue read authority");
  check(observation.source.sceneworks.root === root && SHA.test(observation.source.sceneworks.head), "inventory source differs");
  sourceCheck({ backend: observation.backend, runnerName: observation.runnerName, expectedRunner: observation.runnerName, sceneWorksSha: observation.source.sceneworks.head, inferenceSha: observation.source.inference.head, root, inferenceRoot: observation.source.inference.root });
  const template = await json(path.join(root, TEMPLATE)), plan = await json(path.join(root, PLAN));
  check(await hashFile(path.join(root, TEMPLATE)) === observation.templateSha256 && await hashFile(path.join(root, PLAN)) === observation.planSha256 && template.planSha256 === observation.planSha256, "template/plan changed after inventory");
  check(await hashFile(path.join(root, "config/manifests/builtin.models.jsonc")) === template.catalogSha256 && plan.catalogSha256 === template.catalogSha256, "catalog differs from source corpus/template");
  const bundlePath = path.join(root, BUNDLE); check(await hashFile(bundlePath) === template.carrierBundleSha256, "carrier bundle changed");
  const bundle = await json(bundlePath);
  check(bundle.planSha256 === observation.planSha256 && Object.keys(bundle.assets).length === 33 && Object.keys(plan.assets).length === 33 && Object.keys(plan.assets).every((k) => Object.hasOwn(bundle.assets, k)), "complete33 source bundle required");
  runDir ||= observation.ownedStageRoot; corpusRoot ||= path.join(runDir, "corpus");
  for (const dir of [runDir, corpusRoot]) check(path.isAbsolute(dir) && dir !== root && !inside(root, dir) && dir !== observation.source.inference.root && !inside(observation.source.inference.root, dir), "stage paths must stay outside both frozen source trees");
  check(inside(runDir, corpusRoot), "corpus must be under owned stage root");
  // Validate all bytes and paths before the first write. No archive extraction or encoder regeneration.
  const carriers = Object.entries(plan.assets).map(([key, a]) => {
    check(!path.isAbsolute(a.file) && !a.file.split(/[\\/]/).includes(".."), "unsafe carrier path");
    const bytes = Buffer.from(bundle.assets[key], "base64"); check(bytes.length === a.bytes && digest(bytes) === a.sha256, `source carrier differs: ${key}`);
    return { key, a, bytes, output: path.join(corpusRoot, ...a.file.split("/")) };
  });
  check(new Set(carriers.map((c) => c.output)).size === 33 && carriers.reduce((n, c) => n + c.bytes.length, 0) === 901110, "finite approved corpus counts/bytes differ");
  check(!await lstat(runDir).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e)), "stage root exists; refuse implicit overwrite/reuse");
  for (let parent = path.dirname(runDir); parent !== path.dirname(parent); parent = path.dirname(parent)) {
    const s = await lstat(parent).catch((e) => e.code === "ENOENT" ? null : Promise.reject(e));
    check(!s || (s.isDirectory() && !s.isSymbolicLink()), "stage ancestor is unsafe/symlink");
  }
  if (cacheRoot) check(observation.cache.some((c) => c.root === path.resolve(cacheRoot) && c.safeDirectory), "cache root not observed in declared allowlist");
  const m = { schemaVersion: 1, scope: "named-required", campaignId: `epic25140-s9-${observation.source.sceneworks.head.slice(0, 12)}`, projectId: "UNBOUND", prompts: template.prompts, bindings: plan.bindings, boundaryBindings: plan.boundaryBindings, assets: Object.fromEntries(carriers.map(({ key, a, output }) => [key, { ...a, path: output, assetId: "UNBOUND", normalized: null }])), artifacts: template.artifacts.map((a) => ({ ...a, path: cacheRoot ? snapshotPath(cacheRoot, a) : null })), turboArtifacts: Object.fromEntries(Object.entries(template.turboArtifacts).map(([id, a]) => [id, { ...a, path: cacheRoot ? snapshotPath(cacheRoot, a) : null }])), styles: template.styles.map((s) => ({ ...s, path: cacheRoot ? snapshotPath(cacheRoot, { ...s, ...s.source }) : null })), proofRoutes: template.proofRoutes, productProofCases: template.productProofCases, limits: template.limits, runners: {}, sourceFreeze: { stage: "preparation-unbound", ...observation.source, inferencePin: observation.source.inference.head, files: {}, inferenceFiles: {}, resourceContract: {} }, preparation: { campaignReady: false, observation, requiredAxes: template.requiredAxes, requiredProductProofSetup: template.requiredProductProofSetup, unresolved: ["permanent inference-main/pin freeze must be verified", "both real source-built API/worker binary attestations required", "counterpart physical identity/cache inventory unavailable", "normal owned corpus imports/readbacks required", "all forced-condition product proof setup receipts required"] } };
  if (!cacheRoot) m.preparation.unresolved.push("explicit observed cache root not selected");
  for (const a of m.artifacts) if (!a.sha256) m.preparation.unresolved.push(`source SHA256 missing; exact Git blob retained: ${a.id}`);
  const styles = m.styles;
  for (const file of [...SOURCE_FILES, ...styles.map((s) => s.compatibilityEvidenceSource)]) m.sourceFreeze.files[file] = await hashFile(path.join(root, file));
  for (const file of INFERENCE_FILES) m.sourceFreeze.inferenceFiles[file] = await hashFile(path.join(observation.source.inference.root, file));
  const mlxText = await readFile(path.join(observation.source.inference.root, INFERENCE_FILES[2]), "utf8"), cudaText = await readFile(path.join(observation.source.inference.root, INFERENCE_FILES[3]), "utf8");
  const kernels = { mlx: mlxText.match(/pub const KERNEL: &str = "([^"]+)"/)?.[1], candle_cuda: cudaText.match(/name: "([^"]+)"/)?.[1] }; check(kernels.mlx && kernels.candle_cuda, "native source kernel identity absent");
  for (const backend of ["mlx", "candle_cuda"]) {
    const active = backend === observation.backend, device = backend === "mlx" ? observation.physical : observation.physical.devices?.length === 1 ? observation.physical.devices[0] : null;
    const physicalGpuId = active ? backend === "mlx" ? device.physicalGpuId : device?.uuid ?? null : null;
    m.runners[backend] = { workerId: `h3-s9-${backend}`, requestedGpu: backend === "mlx" ? "mlx" : active ? device?.index ?? null : null, reportedBackend: backend === "mlx" ? "mlx" : "cuda", kernel: kernels[backend], physicalGpuId, apiUrl: active ? "http://127.0.0.1:58141" : null, guardian: active ? { port: 58141, maxSeconds: 3300, cleanupSeconds: 60, dataDir: path.join(runDir, "data"), configDir: path.join(runDir, "config") } : null, leasePath: active ? path.join(runDir, "campaign.lease.json") : null, physicalGuardAdapter: active ? { path: path.join(root, "scripts/h3-veda-guard-adapter.mjs"), bytes: (await stat(path.join(root, "scripts/h3-veda-guard-adapter.mjs"))).size, sha256: m.sourceFreeze.files["scripts/h3-veda-guard-adapter.mjs"] } : null, apiBinary: null, workerBinary: null, attestation: null };
    if (active && physicalGpuId) m.sourceFreeze.resourceContract[backend] = backend === "mlx" ? { physicalGpuId, metalDeviceName: device.metalDeviceName, hostRunnerNames: ["nax-macos-2"], canonicalLocks: ["/Users/michael/Documents/Codex/2026-10-10/epic-25140/s0-gpu-driver.lock", "/Users/michael/Documents/Codex/2026-10-10/epic-25140/s2s4-aggregate-target.lock"] } : { physicalGpuId, runnerName: observation.runnerName, canonicalLocks: [path.join(process.env.PUBLIC ?? "UNBOUND", "SceneWorks", "gpu-locks", `${physicalGpuId}.lock`)] };
    else if (active) m.preparation.unresolved.push("multiple CUDA devices: explicit physical UUID/index selection required");
  }
  const rows = matrix(await catalogs(root), styles); check(rows.filter((r) => !r.style).length === 5400, "required matrix drift");
  check(PROOFS.every((id) => m.proofRoutes[id]) && Object.keys(PROFILES).every((id) => m.prompts[id]), "full terminal proof/profile scope absent");
  for (const backend of ["mlx", "candle_cuda"]) m.sourceFreeze.resourceContract[backend] ??= null;
  m.preparation.unresolved.push("actual host model payload hashes and runtime snapshot resolution unverified");
  m.preparation.missingBindings = { permanentSourceFreeze: true, binaries: ["mlx", "candle_cuda"], counterpartInventory: observation.backend === "mlx" ? "candle_cuda" : "mlx", activeDeviceSelection: !m.runners[observation.backend].physicalGpuId, modelPayloadVerification: true, cacheSelection: !cacheRoot, productImports: true, productProofSetup: template.requiredProductProofSetup, unknownArtifactHashes: m.artifacts.filter((a) => !a.sha256).map((a) => a.id) };
  await mkdir(path.dirname(runDir), { recursive: true });
  await mkdir(runDir, { recursive: false });
  await writeFile(path.join(runDir, "s9-stage-owner.json"), JSON.stringify({ schemaVersion: 1, source: observation.source, runnerName: observation.runnerName, planSha256: observation.planSha256, templateSha256: observation.templateSha256 }) + "\n", { flag: "wx" });
  for (const c of carriers) { await mkdir(path.dirname(c.output), { recursive: true }); await writeFile(c.output, c.bytes, { flag: "wx" }); check(await hashFile(c.output) === c.a.sha256, "staged byte readback differs"); }
  for (const s of styles) s.compatibilityEvidence = { path: path.join(root, s.compatibilityEvidenceSource), bytes: (await stat(path.join(root, s.compatibilityEvidenceSource))).size, sha256: m.sourceFreeze.files[s.compatibilityEvidenceSource] };
  const inventoryPath = path.join(runDir, "artifact-inventory.preparation.json"), inventoryData = { catalogSha256: template.catalogSha256, files: m.artifacts.map(({ path: unused, coverage: unusedCoverage, ...a }) => a), status: "preparation_source_metadata_not_host_payload_verification" };
  await writeFile(inventoryPath, JSON.stringify(inventoryData, null, 2) + "\n", { flag: "wx" }); m.artifactInventory = { path: inventoryPath, bytes: (await stat(inventoryPath)).size, sha256: await hashFile(inventoryPath) }; m.sourceFreeze.artifactInventorySha256 = m.artifactInventory.sha256;
  const manifestPath = path.join(runDir, "manifest.preparation.json"); await writeFile(manifestPath, JSON.stringify(m, null, 2) + "\n", { flag: "wx" });
  const result = { schemaVersion: 1, kind: "epic25140_cpu_staging", backend: observation.backend, runnerName: observation.runnerName, source: observation.source, runDir, corpusRoot, manifestPath, manifestSha256: await hashFile(manifestPath), planSha256: observation.planSha256, stagedCarriers: 33, stagedBytes: 901110, namedCells: 5400, styleInclusiveCells: rows.length, optionalStressAdded: 0, artifactDescriptors: m.artifacts.length, styles: styles.length, unresolved: m.preparation.unresolved, campaignReady: false, productImported: false, physicalAdmission: false, qualityAcceptance: "pending Michael" };
  await writeFile(path.join(runDir, "stage-receipt.json"), JSON.stringify(result, null, 2) + "\n", { flag: "wx" }); return result;
}
