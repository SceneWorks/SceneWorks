#!/usr/bin/env node
// Materialize the terminal route bundle from pre-provisioned, immutable corpus
// rows.  This process reads only local assets; it never fetches datasets/models.
import { createHash } from "node:crypto";
import { lstat, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { INFERENCE_REVISION, terminalSourceRowsSha256 } from "./starvector-terminal-campaign.mjs";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";
import { loadUpstreamReference, verifyUpstreamExecution } from "./lib/starvector-terminal-upstream-reference.mjs";

import { validateLimitCases } from "./lib/starvector-terminal-limit-cases.mjs";

const sha = (value) => createHash("sha256").update(value).digest("hex");
const die = (message) => { throw new Error(`starvector terminal bundle: ${message}`); };
const json = async (file) => JSON.parse(await readFile(file, "utf8"));
const shippingMaxNewTokens = Object.freeze({ "1b": 7933, "8b": 15422 });

async function bindLocalFile(root, relative, expected, label) {
  if (!relative || path.isAbsolute(relative) || relative.split(/[\\/]/).includes("..")) die(`${label} path is unsafe`);
  const file = path.join(root, ...relative.split(/[\\/]/)); const info = await lstat(file);
  if (!info.isFile() || info.isSymbolicLink()) die(`${label} must be a regular non-symlink asset`);
  const bytes = await readFile(file); if (sha(bytes) !== expected) die(`${label} digest mismatches immutable row identity`);
  return { path: file, byte_size: info.size, sha256: expected };
}

export async function materializeBundle({ corpusPath, assetsRoot, output, permanentPin, bindingPath, hostileGenerator = null, upstreamRoot = process.env.STARVECTOR_TERMINAL_UPSTREAM_ROOT ?? (process.env.RUNNER_TEMP ? path.join(process.env.RUNNER_TEMP, "starvector-upstream") : undefined) }) {
  if (permanentPin !== INFERENCE_REVISION) die("permanent pin must equal the exact inference corpus revision");
  const corpus = await json(corpusPath), index = await json(path.join(assetsRoot, "starvector-terminal-row-index-v1.json")), binding = await json(bindingPath);
  if (index.schema_version !== 2 || index.inference_revision !== INFERENCE_REVISION || index.row_identity_sha256 !== corpus.upstream_image_quality_cases.row_identity_sha256 || !Array.isArray(index.rows) || index.rows.length !== 120) die("pre-provisioned corpus row index identity/count mismatch");
  const rows = [];
  for (const [position, row] of index.rows.entries()) {
    if (row.case_index !== position || !row.filename || !/^[a-f0-9]{64}$/.test(row.svg_sha256) || !/^[a-f0-9]{64}$/.test(row.png_sha256) || !/^[a-f0-9]{64}$/.test(row.reference_png_sha256)) die("corpus row record is not immutable");
    row.svg = await bindLocalFile(assetsRoot, row.svg_path, row.svg_sha256, "source SVG"); row.input_png = await bindLocalFile(assetsRoot, row.input_png_path, row.png_sha256, "input PNG"); row.reference = await bindLocalFile(assetsRoot, row.reference_png, row.reference_png_sha256, "reference PNG"); rows.push(row);
  }
  const rowHash = terminalSourceRowsSha256(rows);
  if (rowHash !== corpus.upstream_image_quality_cases.row_identity_sha256) die("corpus source row identities drifted");
  if (!binding?.project_id || !Array.isArray(binding.assets) || binding.assets.length !== 120) die("tuple-local API project/asset binding is missing");
  const imported = new Map(binding.assets.map((item) => [item.case_index, item]));
  const route = (row, suffix, tier = "1b") => { const asset = imported.get(row.case_index), detailBudget = row.detail_budgets?.[tier]; if (!asset?.asset_id || asset.input_png_sha256 !== row.input_png.sha256) die("imported project asset identity mismatches source row"); if (detailBudget?.maxNewTokens !== shippingMaxNewTokens[tier] || detailBudget.maxSvgBytes !== 262144 || detailBudget.maxWallTimeMs !== 120000) die(`corpus row lacks the ${tier} shipping Detail budget`); return { case_id: `quality-v1-${row.case_index}${suffix}`, projectId: binding.project_id, sourceAssetId: asset.asset_id, model: tier === "8b" ? "starvector_8b" : "starvector_1b", source_svg: row.svg.path, source_svg_sha256: row.svg.sha256, input_png: row.input_png.path, input_png_sha256: row.input_png.sha256, reference_png: row.reference.path, reference_png_sha256: row.reference.sha256, sampling: row.sampling, detailBudget }; };
  const routeScenarios = (records, label, tier) => {
    if (!Array.isArray(records)) die(`pre-provisioned ${label} cases are missing`);
    if (label === "limit") validateLimitCases(records, tier);
    return records.map((record, index) => {
      const scenario = {};
      for (const key of ["case_id", "case_index", "operation", "finish_reason", "sampling", "detailBudget", "cancel_after_create", "cancel_after_progress", "worker_unloaded", "source_case_index", "scenario"]) if (record[key] !== undefined) scenario[key] = record[key];
      const sourceIndex = record.source_case_index ?? record.case_index ?? index;
      if (!Number.isInteger(sourceIndex) || sourceIndex < 0 || sourceIndex >= rows.length) die(`${label} source case index is invalid`);
      return { ...route(rows[sourceIndex], `-${label}`, tier), ...scenario };
    });
  };
  const parityRows = corpus.upstream_image_quality_cases.sources.flatMap((_, sourceIndex) => rows.slice(sourceIndex * 30, sourceIndex * 30 + 5));
  if (parityRows.length !== 20) die("pinned corpus must select five deterministic parity rows from each of four sources");
  const tuples = Object.fromEntries(["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"].map((tuple) => { const tier = tuple.split(":")[1]; return [tuple, { image_quality: rows.map((row) => route(row, "", tier)), deterministic_parity: parityRows.map((row, seed) => ({ ...route(row, "-parity", tier), seed })), lifecycle: routeScenarios(index.lifecycle_cases?.[tuple], "lifecycle", tier), limits: routeScenarios(index.limit_cases?.[tuple], "limit", tier) }]; }));
  if (!hostileGenerator) {
    await verifyUpstreamExecution(upstreamRoot);
    // The same two immutable model-specific oracle bundles serve both backends.
    for (const tier of ["1b", "8b"]) {
      const upstream = await loadUpstreamReference(upstreamRoot, tier, index.rows);
      for (const backend of ["mlx", "candle-cuda"]) {
        const entry = tuples[`${backend}:${tier}`];
        entry.upstream_reference = upstream.upstream_reference;
        entry.upstream_paths = { config: upstream.config_path, processor: upstream.processor_path, transcript: upstream.transcript_path };
        entry.deterministic_parity = entry.deterministic_parity.map((item, i) => {
          const reference = upstream.cases[i];
          if (reference.outcome === "accepted") return { ...item, upstream_outcome: "accepted", upstream_svg: reference.upstream_svg, upstream_svg_sha256: reference.upstream_svg_sha256, upstream_preview_png: reference.upstream_preview_png, upstream_preview_png_sha256: reference.upstream_preview_png_sha256 };
          return { ...item, upstream_outcome: "rejected", upstream_rejection_stage: reference.rejection_stage, upstream_rejection_code: reference.rejection_code, upstream_rejection_reason: reference.rejection_reason, upstream_raw_svg: reference.upstream_raw_svg, upstream_raw_svg_sha256: reference.upstream_raw_svg_sha256, ...(reference.rejection_stage === "sanitizer" ? { upstream_sanitizer_stdout: reference.sanitizer_stdout, upstream_sanitizer_stdout_sha256: reference.sanitizer_stdout_sha256, upstream_sanitizer_stderr: reference.sanitizer_stderr, upstream_sanitizer_stderr_sha256: reference.sanitizer_stderr_sha256 } : {}) };
        });
      }
    }
  }
  const inferenceRoot = path.resolve(path.dirname(corpusPath), "..");
  const validator = hostileGenerator ? null : await import(pathToFileURL(path.join(inferenceRoot, "scripts", "release", "starvector_terminal_evidence.mjs")).href);
  const payloadFor = hostileGenerator ?? validator.hostilePayload;
  const hostileDir = path.join(path.dirname(output), "hostile-inputs");
  await mkdir(hostileDir, { recursive: true });
  const hostile_sanitizer = [];
  for (let case_index = 0; case_index < 200; case_index += 1) {
    const payload = Buffer.from(payloadFor(case_index));
    const input_sha256 = sha(payload), case_id = `hostile-v1-${case_index}`;
    const input_path = path.join(hostileDir, `${case_index}.svg`);
    await writeFile(input_path, payload);
    hostile_sanitizer.push({ case_index, case_id, input_path, input_sha256 });
  }
  const hostileIdentity = sha(hostile_sanitizer.map((item) => item.input_sha256).join("\n"));
  if (hostileIdentity !== corpus.sceneworks_owned_suites?.hostile_sanitizer?.content_identity_sha256) die("generated hostile payload identities drifted from the immutable corpus");
  // The receipt validator seals its stable semantic corpus identity, not raw
  // JSON bytes (whose whitespace is not contract data).  Test-only injected
  // hostile generators cannot load the inference module and retain a local
  // byte digest solely for their isolated fixture.
  const corpus_sha256 = validator ? validator.validatePlan(corpus) : sha(await readFile(corpusPath));
  const bundle = { schema_version: 1, inference_revision: INFERENCE_REVISION, corpus_sha256, row_identity_sha256: rowHash, tuples, hostile_sanitizer, prompt_composition: index.prompt_composition.map((record) => ({ ...record, projectId: binding.project_id })) };
  await mkdir(path.dirname(output), { recursive: true }); const bytes = JSON.stringify(bundle, null, 2) + "\n"; await writeFile(output, bytes); await writeFile(`${output}.sha256`, sha(bytes) + "\n"); return bundle;
}
if (isExecutedModule(import.meta.url)) { const [corpusPath, assetsRoot, output, permanentPin, bindingPath] = process.argv.slice(2); materializeBundle({ corpusPath, assetsRoot, output, permanentPin, bindingPath }).catch((error) => { console.error(error.message); process.exitCode = 1; }); }
