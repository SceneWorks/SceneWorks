import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { serializeTerminalSourceRows, terminalSourceRowsSha256 } from "./starvector-terminal-campaign.mjs";
import { materializeBundle } from "./starvector-terminal-case-bundle.mjs";

const pin = "b2d9e0917499517cf8c1518e0d360cac8693b0c0", sha = (value) => createHash("sha256").update(value).digest("hex");
const pinnedSources = [
  ["starvector/svg-stack-simple", "1d2a96a17cc0c4c1f337b7631adc8c5885bc72ea"],
  ["starvector/svg-icons-simple", "e1918a27ba6649e856e5db0710d8a6c7046762c1"],
  ["starvector/svg-emoji-simple", "fa75b3617872ae57e6f3cb450aee65dbccbd69e0"],
  ["starvector/svg-fonts-simple", "453c739ea13ad2685127f721c333f14d99485299"],
];

test("pinned corpus row identity includes the canonical terminal newline", async () => {
  const fixture = await readFile(new URL("./fixtures/starvector-terminal-pinned-row-identities-v1.tsv", import.meta.url), "utf8");
  const rows = fixture.trimEnd().split(/\r?\n/).map((line, caseIndex) => {
    const [filename, svg_sha256] = line.split("\t"), [dataset, revision] = pinnedSources[Math.floor(caseIndex / 30)];
    return { dataset, revision, row_index: caseIndex % 30, filename, svg_sha256 };
  });
  const canonical = serializeTerminalSourceRows(rows);
  assert.equal(rows.length, 120);
  assert.equal(terminalSourceRowsSha256(rows), "f9529c2e5a86bef6644054c909c4f621991f6384d9b33a029ad46ff2e6cd3b88");
  assert.equal(sha(canonical.slice(0, -1)), "23864c733f1171c6b12f5d156fe3fbff5623a6ff975e690da00bdbdef2826f7d");
  assert.notEqual(sha(canonical.slice(0, -1)), terminalSourceRowsSha256(rows));
});

test("source bundle refuses row identity drift and seals the resulting bytes", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-bundle-")), assets = path.join(root, "assets"), output = path.join(root, "bundle.json"); await mkdir(assets);
  await writeFile(path.join(assets, "source.svg"), "svg"); await writeFile(path.join(assets, "input.png"), "input"); await writeFile(path.join(assets, "reference.png"), "reference"); await writeFile(path.join(assets, "preview.png"), "preview");
  const rows = Array.from({ length: 120 }, (_, case_index) => ({ case_index, dataset: `d${Math.floor(case_index / 30)}`, revision: "a".repeat(40), row_index: case_index % 30, filename: `${case_index}.svg`, svg_sha256: sha("svg"), png_sha256: sha("input"), reference_png_sha256: sha("reference"), preview_png_sha256: sha("preview"), svg_path: "source.svg", input_png_path: "input.png", reference_png: "reference.png", preview_png: "preview.png", asset_id: `asset-${case_index}` }));
  const rowHash = terminalSourceRowsSha256(rows);
  const hostileGenerator = (caseIndex) => `hostile-${caseIndex}`;
  const hostileIdentity = sha(Array.from({ length: 200 }, (_, caseIndex) => sha(hostileGenerator(caseIndex))).join("\n"));
  const corpus = { upstream_image_quality_cases: { row_identity_sha256: rowHash, sources: Array.from({ length: 4 }, (_, sourceIndex) => ({ dataset: `d${sourceIndex}` })) }, sceneworks_owned_suites: { hostile_sanitizer: { content_identity_sha256: hostileIdentity } } }; await writeFile(path.join(root, "corpus.json"), JSON.stringify(corpus));
  const index = { inference_revision: pin, row_identity_sha256: rowHash, rows, lifecycle_cases: Object.fromEntries(["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"].map((key) => [key, [{}, {}, {}, {}]])), limit_cases: Object.fromEntries(["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"].map((key) => [key, [{}, {}, {}, {}, {}, {}]])), run_identity: {}, hardware: {}, hostile_sanitizer: [], prompt_composition: [] }; await writeFile(path.join(assets, "starvector-terminal-row-index-v1.json"), JSON.stringify(index)); const bindingPath = path.join(root, "binding.json"); await writeFile(bindingPath, JSON.stringify({ project_id: "project", assets: rows.map((row) => ({ case_index: row.case_index, asset_id: `asset-${row.case_index}`, input_png_sha256: row.png_sha256 })) }));
  const bundle = await materializeBundle({ corpusPath: path.join(root, "corpus.json"), assetsRoot: assets, output, permanentPin: pin, bindingPath, hostileGenerator }); assert.equal(bundle.row_identity_sha256, rowHash); assert.match(await readFile(`${output}.sha256`, "utf8"), /^[a-f0-9]{64}/); assert.equal(bundle.hostile_sanitizer.length, 200); assert.equal(await readFile(bundle.hostile_sanitizer[199].input_path, "utf8"), "hostile-199");
  const parityCaseIds = bundle.tuples["mlx:1b"].deterministic_parity.map((record) => record.case_id);
  assert.deepEqual(parityCaseIds, [0, 30, 60, 90].flatMap((start) => Array.from({ length: 5 }, (_, offset) => `quality-v1-${start + offset}-parity`)));
  index.rows[0].filename = "drift.svg"; await writeFile(path.join(assets, "starvector-terminal-row-index-v1.json"), JSON.stringify(index)); await assert.rejects(() => materializeBundle({ corpusPath: path.join(root, "corpus.json"), assetsRoot: assets, output, permanentPin: pin, bindingPath, hostileGenerator }), /source row identities drifted/);
});
