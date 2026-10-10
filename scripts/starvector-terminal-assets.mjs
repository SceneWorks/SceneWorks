#!/usr/bin/env node
// Create a tuple-local project and import the exact 120 raster inputs through
// the production API.  Asset ids are never preassigned in the corpus index.
import { createHash } from "node:crypto";
import { lstat, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";

const sha = (value) => createHash("sha256").update(value).digest("hex");
const die = (message) => { throw new Error(`starvector terminal assets: ${message}`); };
const json = async (file) => JSON.parse(await readFile(file, "utf8"));
async function request(url, init) { const response = await fetch(url, init); const body = await response.json(); if (!response.ok) die(`${response.status}: ${JSON.stringify(body)}`); return body; }
async function localPng(root, relative, digest) { if (!relative || path.isAbsolute(relative) || relative.split(/[\\/]/).includes("..")) die("unsafe tuple asset path"); const file = path.join(root, ...relative.split(/[\\/]/)); const info = await lstat(file), bytes = await readFile(file); if (!info.isFile() || info.isSymbolicLink() || sha(bytes) !== digest) die("tuple input PNG identity mismatch"); return { file, bytes, size: info.size }; }
export async function importTupleAssets({ assetsRoot, apiUrl, tuple, output }) {
  const index = await json(path.join(assetsRoot, "starvector-terminal-row-index-v1.json")); if (!Array.isArray(index.rows) || index.rows.length !== 120) die("expected 120 immutable source rows");
  const project = await request(new URL("/api/v1/projects", apiUrl), { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: `StarVector terminal ${tuple}` }) }); if (!project.id) die("product API did not create tuple-local project");
  const assets = [];
  for (const row of index.rows) { const input = await localPng(assetsRoot, row.input_png_path, row.png_sha256), form = new FormData(); form.append("file", new Blob([input.bytes], { type: "image/png" }), `${row.case_index}.png`); const imported = await request(new URL(`/api/v1/projects/${project.id}/assets`, apiUrl), { method: "POST", body: form }); if (!imported.id) die("product API import did not return an asset id"); assets.push({ case_index: row.case_index, asset_id: imported.id, input_png_sha256: row.png_sha256, input_png_bytes: input.size }); }
  const binding = { project_id: project.id, tuple, assets, aggregate_sha256: sha(JSON.stringify(assets)) }; await writeFile(output, JSON.stringify(binding, null, 2) + "\n"); return binding;
}
if (isExecutedModule(import.meta.url)) { const [assetsRoot, apiUrl, tuple, output] = process.argv.slice(2); importTupleAssets({ assetsRoot, apiUrl, tuple, output }).catch((error) => { console.error(error.message); process.exitCode = 1; }); }
