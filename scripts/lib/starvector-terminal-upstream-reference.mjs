import { createHash } from "node:crypto";
import { lstat, readFile } from "node:fs/promises";
import path from "node:path";
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const fail = (message) => { throw new Error(`upstream reference: ${message}`); };
export const UPSTREAM_REVISION = "0e083c1911760aa31bc576ca7f337a7f8ee605ec";
export const PARITY_SOURCE_INDICES = [0, 30, 60, 90].flatMap((start) => Array.from({ length: 5 }, (_, offset) => start + offset));

async function verified(root, relative, digest) {
  if (!/^[a-f0-9]{64}$/.test(digest ?? "") || typeof relative !== "string" || !relative || relative.includes("\\") || path.posix.isAbsolute(relative) || path.win32.isAbsolute(relative) || relative.split("/").some((part) => !part || part === "." || part === "..")) fail("unsafe golden artifact path or hash");
  let file = root;
  for (const part of relative.split("/")) { file = path.join(file, part); if ((await lstat(file)).isSymbolicLink()) fail("golden artifact symlink"); }
  if (!(await lstat(file)).isFile() || hash(await readFile(file)) !== digest) fail(`golden artifact content changed: ${relative}`);
  return file;
}

export async function loadUpstreamReference(root, tier, rows) {
  if (!root || !path.isAbsolute(root) || !["1b", "8b"].includes(tier)) fail("absolute shared oracle root and model tier required");
  const manifestPath = path.join(root, `upstream-reference-${tier}.json`);
  const info = await lstat(manifestPath);
  if (!info.isFile() || info.isSymbolicLink()) fail("oracle manifest is not a regular file");
  const value = JSON.parse(await readFile(manifestPath, "utf8")), reference = value.upstream_reference;
  const revision = tier === "1b" ? "380ab95d25a8e9ab1dc825debe238b4953ae13b9" : "518beea8dcb5f7a37c5911e92d1d62a76beee7f9";
  if (value.schema_version !== 1 || reference?.implementation_repository !== "https://github.com/joanrod/star-vector" || reference.implementation_revision !== UPSTREAM_REVISION || reference.checkpoint_repository !== `starvector/starvector-${tier}-im2svg` || reference.checkpoint_revision !== revision || !/^[a-f0-9]{64}$/.test(reference.checkpoint_inventory_sha256 ?? "") || value.cases?.length !== 20) fail("exact upstream implementation/checkpoint/case identity required");
  const paths = {};
  for (const role of ["config", "processor", "transcript"]) paths[`${role}_path`] = await verified(root, value[`${role}_path`], reference[`${role}_sha256`]);
  const cases = [];
  for (const [index, item] of value.cases.entries()) {
    const sourceIndex = PARITY_SOURCE_INDICES[index], row = rows[sourceIndex];
    if (item.case_index !== index || item.source_case_index !== sourceIndex || item.seed !== index || item.input_png_sha256 !== row?.png_sha256) fail(`upstream case ${index} does not bind the selected source raster/seed`);
    cases.push({ ...item, upstream_svg: await verified(root, item.upstream_svg, item.upstream_svg_sha256), upstream_preview_png: await verified(root, item.upstream_preview_png, item.upstream_preview_png_sha256) });
  }
  return { upstream_reference: reference, ...paths, cases };
}

export async function verifyUpstreamExecution(root, env = process.env) {
  const file = path.join(root, "upstream-controller.json");
  if (!(await lstat(file)).isFile() || (await lstat(file)).isSymbolicLink()) fail("upstream controller is not a regular file");
  const controller = JSON.parse(await readFile(file, "utf8"));
  for (const [key, expected] of Object.entries({ campaign_run_id: env.STARVECTOR_TERMINAL_CAMPAIGN_RUN_ID, inference_revision: env.STARVECTOR_TERMINAL_PERMANENT_PIN, workflow_run_id: env.GITHUB_RUN_ID, workflow_run_attempt: Number(env.GITHUB_RUN_ATTEMPT), sceneworks_revision: env.GITHUB_SHA })) {
    if (expected === undefined || controller[key] !== expected) fail(`upstream artifact ${key} differs from this workflow attempt`);
  }
  const entries = controller.artifacts?.entries;
  if (!Array.isArray(entries) || hash(JSON.stringify(entries)) !== controller.artifacts.aggregate_sha256) fail("upstream artifact inventory is invalid");
  for (const tier of ["1b", "8b"]) {
    const entry = entries.find(item => item.path === `upstream-reference-${tier}.json`);
    if (!entry) fail("upstream manifest absent from current workflow inventory");
    const manifest = await verified(root, entry.path, entry.sha256);
    if ((await lstat(manifest)).size !== entry.byte_size) fail("upstream manifest size changed");
  }
  return controller;
}
