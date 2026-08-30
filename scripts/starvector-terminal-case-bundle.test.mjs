import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { materializeBundle } from "./starvector-terminal-case-bundle.mjs";

const pin = "669f749b428ed65acfdfee917bcf601b8cf6db1c", sha = (value) => createHash("sha256").update(value).digest("hex");
test("source bundle refuses row identity drift and seals the resulting bytes", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-bundle-")), assets = path.join(root, "assets"), output = path.join(root, "bundle.json"); await mkdir(assets);
  await writeFile(path.join(assets, "source.svg"), "svg"); await writeFile(path.join(assets, "input.png"), "input"); await writeFile(path.join(assets, "reference.png"), "reference"); await writeFile(path.join(assets, "preview.png"), "preview");
  const rows = Array.from({ length: 120 }, (_, case_index) => ({ case_index, dataset: `d${Math.floor(case_index / 30)}`, revision: "a".repeat(40), row_index: case_index % 30, filename: `${case_index}.svg`, svg_sha256: sha("svg"), png_sha256: sha("input"), reference_png_sha256: sha("reference"), preview_png_sha256: sha("preview"), svg_path: "source.svg", input_png_path: "input.png", reference_png: "reference.png", preview_png: "preview.png", asset_id: `asset-${case_index}` }));
  const rowHash = sha(rows.map((row) => JSON.stringify({ dataset: row.dataset, revision: row.revision, row_index: row.row_index, filename: row.filename, svg_sha256: row.svg_sha256 })).join("\n"));
  const hostileGenerator = (caseIndex) => `hostile-${caseIndex}`;
  const hostileIdentity = sha(Array.from({ length: 200 }, (_, caseIndex) => sha(hostileGenerator(caseIndex))).join("\n"));
  const corpus = { upstream_image_quality_cases: { row_identity_sha256: rowHash }, sceneworks_owned_suites: { hostile_sanitizer: { content_identity_sha256: hostileIdentity } } }; await writeFile(path.join(root, "corpus.json"), JSON.stringify(corpus));
  const index = { inference_revision: pin, row_identity_sha256: rowHash, rows, lifecycle_cases: Object.fromEntries(["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"].map((key) => [key, [{}, {}, {}, {}]])), limit_cases: Object.fromEntries(["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"].map((key) => [key, [{}, {}, {}, {}, {}, {}]])), run_identity: {}, hardware: {}, hostile_sanitizer: [], prompt_composition: [] }; await writeFile(path.join(assets, "starvector-terminal-row-index-v1.json"), JSON.stringify(index)); const bindingPath = path.join(root, "binding.json"); await writeFile(bindingPath, JSON.stringify({ project_id: "project", assets: rows.map((row) => ({ case_index: row.case_index, asset_id: `asset-${row.case_index}`, input_png_sha256: row.png_sha256 })) }));
  const bundle = await materializeBundle({ corpusPath: path.join(root, "corpus.json"), assetsRoot: assets, output, permanentPin: pin, bindingPath, hostileGenerator }); assert.match(await readFile(`${output}.sha256`, "utf8"), /^[a-f0-9]{64}/); assert.equal(bundle.hostile_sanitizer.length, 200); assert.equal(await readFile(bundle.hostile_sanitizer[199].input_path, "utf8"), "hostile-199");
  index.rows[0].filename = "drift.svg"; await writeFile(path.join(assets, "starvector-terminal-row-index-v1.json"), JSON.stringify(index)); await assert.rejects(() => materializeBundle({ corpusPath: path.join(root, "corpus.json"), assetsRoot: assets, output, permanentPin: pin, bindingPath, hostileGenerator }), /source row identities drifted/);
});
