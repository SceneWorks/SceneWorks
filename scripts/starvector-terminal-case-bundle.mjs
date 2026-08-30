#!/usr/bin/env node
// Materialize the terminal route bundle from pre-provisioned, immutable corpus
// rows.  This process reads only local assets; it never fetches datasets/models.
import { createHash } from "node:crypto";
import { lstat, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const sha = (value) => createHash("sha256").update(value).digest("hex");
const die = (message) => { throw new Error(`starvector terminal bundle: ${message}`); };
const REV = "669f749b428ed65acfdfee917bcf601b8cf6db1c";
const json = async (file) => JSON.parse(await readFile(file, "utf8"));

async function bindLocalFile(root, relative, expected, label) {
  if (!relative || path.isAbsolute(relative) || relative.split(/[\\/]/).includes("..")) die(`${label} path is unsafe`);
  const file = path.join(root, ...relative.split(/[\\/]/)); const info = await lstat(file);
  if (!info.isFile() || info.isSymbolicLink()) die(`${label} must be a regular non-symlink asset`);
  const bytes = await readFile(file); if (sha(bytes) !== expected) die(`${label} digest mismatches immutable row identity`);
  return { path: file, byte_size: info.size, sha256: expected };
}

export async function materializeBundle({ corpusPath, assetsRoot, output, permanentPin, bindingPath, hostileGenerator = null }) {
  if (permanentPin !== REV) die("permanent pin must equal the exact inference corpus revision");
  const corpus = await json(corpusPath), index = await json(path.join(assetsRoot, "starvector-terminal-row-index-v1.json")), binding = await json(bindingPath);
  if (index.inference_revision !== REV || index.row_identity_sha256 !== corpus.upstream_image_quality_cases.row_identity_sha256 || !Array.isArray(index.rows) || index.rows.length !== 120) die("pre-provisioned corpus row index identity/count mismatch");
  const rows = [];
  for (const [position, row] of index.rows.entries()) {
    if (row.case_index !== position || !row.filename || !/^[a-f0-9]{64}$/.test(row.svg_sha256) || !/^[a-f0-9]{64}$/.test(row.png_sha256) || !/^[a-f0-9]{64}$/.test(row.reference_png_sha256)) die("corpus row record is not immutable");
    row.svg = await bindLocalFile(assetsRoot, row.svg_path, row.svg_sha256, "source SVG"); row.input_png = await bindLocalFile(assetsRoot, row.input_png_path, row.png_sha256, "input PNG"); row.reference = await bindLocalFile(assetsRoot, row.reference_png, row.reference_png_sha256, "reference PNG"); rows.push(row);
  }
  const rowHash = sha(rows.map((row) => JSON.stringify({ dataset: row.dataset, revision: row.revision, row_index: row.row_index, filename: row.filename, svg_sha256: row.svg_sha256 })).join("\n"));
  if (rowHash !== corpus.upstream_image_quality_cases.row_identity_sha256) die("corpus source row identities drifted");
  if (!binding?.project_id || !Array.isArray(binding.assets) || binding.assets.length !== 120) die("tuple-local API project/asset binding is missing");
  const imported = new Map(binding.assets.map((item) => [item.case_index, item]));
  const route = (row, suffix, tier = "1b") => { const asset = imported.get(row.case_index); if (!asset?.asset_id || asset.input_png_sha256 !== row.input_png.sha256) die("imported project asset identity mismatches source row"); return { case_id: `quality-v1-${row.case_index}${suffix}`, projectId: binding.project_id, sourceAssetId: asset.asset_id, model: tier === "8b" ? "starvector_8b" : "starvector_1b", source_svg: row.svg.path, source_svg_sha256: row.svg.sha256, input_png: row.input_png.path, input_png_sha256: row.input_png.sha256, reference_png: row.reference.path, reference_png_sha256: row.reference.sha256, sampling: row.sampling, detailBudget: row.detail_budget }; };
  const parityRows = corpus.upstream_image_quality_cases.sources.flatMap((_, sourceIndex) => rows.slice(sourceIndex * 30, sourceIndex * 30 + 5));
  if (parityRows.length !== 20) die("pinned corpus must select five deterministic parity rows from each of four sources");
  const tuples = Object.fromEntries(["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"].map((tuple) => { const tier = tuple.split(":")[1]; return [tuple, { image_quality: rows.map((row) => route(row, "", tier)), deterministic_parity: parityRows.map((row, seed) => ({ ...route(row, "-parity", tier), seed })), lifecycle: index.lifecycle_cases?.[tuple], limits: index.limit_cases?.[tuple] }]; }));
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
  const bundle = { schema_version: 1, inference_revision: REV, corpus_sha256, row_identity_sha256: rowHash, tuples, hostile_sanitizer, prompt_composition: index.prompt_composition };
  await mkdir(path.dirname(output), { recursive: true }); const bytes = JSON.stringify(bundle, null, 2) + "\n"; await writeFile(output, bytes); await writeFile(`${output}.sha256`, sha(bytes) + "\n"); return bundle;
}
if (import.meta.url === `file://${process.argv[1]}`) { const [corpusPath, assetsRoot, output, permanentPin, bindingPath] = process.argv.slice(2); materializeBundle({ corpusPath, assetsRoot, output, permanentPin, bindingPath }).catch((error) => { console.error(error.message); process.exitCode = 1; }); }
