import { createHash } from "node:crypto";
import { lstat, readFile } from "node:fs/promises";
import path from "node:path";
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const fail = (message) => { throw new Error(`upstream reference: ${message}`); };
export const UPSTREAM_REVISION = "0e083c1911760aa31bc576ca7f337a7f8ee605ec";
export const PARITY_SOURCE_INDICES = [0, 30, 60, 90].flatMap((start) => Array.from({ length: 5 }, (_, offset) => start + offset));

export function normalizeStarVectorRejection(reason, stage = "sanitizer") {
  if (stage === "generation_limit") {
    if (!["token_limit", "byte_limit", "wall_time_limit"].includes(reason)) fail("unknown upstream generation-limit rejection");
    return reason;
  }
  if (stage !== "sanitizer" || typeof reason !== "string" || !reason.startsWith("provider SVG ")) fail("rejection is not a typed SVG policy outcome");
  const value = reason.toLowerCase();
  if (value.includes("malformed") || value.includes("not valid utf-8") || value.includes("not utf-8")) return "malformed_svg";
  if (value.includes("<animate") || value.includes("<set>") || value.includes("animation")) return "animation";
  if (value.includes("<text>")) return "text";
  if (value.includes("external") || value.includes("data:") || value.includes("file:") || value.includes("http:") || value.includes("https:") || value.includes("@import")) return "external_io";
  if (value.includes("<use>") || value.includes("href")) return "unsafe_href_use";
  return "svg_policy";
}

async function verified(root, relative, digest) {
  if (!/^[a-f0-9]{64}$/.test(digest ?? "") || typeof relative !== "string" || !relative || relative.includes("\\") || path.posix.isAbsolute(relative) || path.win32.isAbsolute(relative) || relative.split("/").some((part) => !part || part === "." || part === "..")) fail("unsafe golden artifact path or hash");
  let file = root;
  for (const part of relative.split("/")) { file = path.join(file, part); if ((await lstat(file)).isSymbolicLink()) fail("golden artifact symlink"); }
  if (!(await lstat(file)).isFile() || hash(await readFile(file)) !== digest) fail(`golden artifact content changed: ${relative}`);
  return file;
}

export async function loadUpstreamReference(root, tier, rows, { pending = false } = {}) {
  if (!root || !path.isAbsolute(root) || !["1b", "8b"].includes(tier)) fail("absolute shared oracle root and model tier required");
  const manifestPath = path.join(root, `upstream-reference-${tier}${pending ? ".pending" : ""}.json`);
  const info = await lstat(manifestPath);
  if (!info.isFile() || info.isSymbolicLink()) fail("oracle manifest is not a regular file");
  const value = JSON.parse(await readFile(manifestPath, "utf8")), reference = value.upstream_reference;
  const revision = tier === "1b" ? "380ab95d25a8e9ab1dc825debe238b4953ae13b9" : "518beea8dcb5f7a37c5911e92d1d62a76beee7f9";
  if (![1, 2].includes(value.schema_version) || reference?.implementation_repository !== "https://github.com/joanrod/star-vector" || reference.implementation_revision !== UPSTREAM_REVISION || reference.checkpoint_repository !== `starvector/starvector-${tier}-im2svg` || reference.checkpoint_revision !== revision || !/^[a-f0-9]{64}$/.test(reference.checkpoint_inventory_sha256 ?? "") || value.cases?.length !== 20) fail("exact upstream implementation/checkpoint/case identity required");
  const paths = {};
  for (const role of ["config", "processor", "transcript"]) paths[`${role}_path`] = await verified(root, value[`${role}_path`], reference[`${role}_sha256`]);
  const cases = [];
  for (const [index, item] of value.cases.entries()) {
    const sourceIndex = PARITY_SOURCE_INDICES[index], row = rows[sourceIndex];
    if (item.case_index !== index || item.source_case_index !== sourceIndex || item.seed !== index || item.input_png_sha256 !== row?.png_sha256) fail(`upstream case ${index} does not bind the selected source raster/seed`);
    if (value.schema_version === 1) {
      cases.push({ ...item, outcome: "accepted", upstream_svg: await verified(root, item.upstream_svg, item.upstream_svg_sha256), upstream_preview_png: await verified(root, item.upstream_preview_png, item.upstream_preview_png_sha256) });
    } else if (item.outcome === "accepted") {
      const expected = ["case_index", "input_png_sha256", "outcome", "seed", "source_case_index", "upstream_preview_png", "upstream_preview_png_sha256", "upstream_svg", "upstream_svg_sha256"];
      if (Object.keys(item).sort().join("|") !== expected.sort().join("|")) fail(`accepted upstream case ${index} has unexpected evidence`);
      cases.push({ ...item, upstream_svg: await verified(root, item.upstream_svg, item.upstream_svg_sha256), upstream_preview_png: await verified(root, item.upstream_preview_png, item.upstream_preview_png_sha256) });
    } else if (item.outcome === "rejected") {
      const common = ["case_index", "input_png_sha256", "outcome", "rejection_code", "rejection_reason", "rejection_stage", "seed", "source_case_index", "upstream_raw_svg", "upstream_raw_svg_sha256"];
      const sanitizer = ["sanitizer_stderr", "sanitizer_stderr_sha256", "sanitizer_stdout", "sanitizer_stdout_sha256"];
      const expected = item.rejection_stage === "sanitizer" ? [...common.filter(key => key !== "rejection_code"), ...sanitizer] : common;
      if (Object.keys(item).sort().join("|") !== expected.sort().join("|")) fail(`rejected upstream case ${index} has unexpected evidence`);
      const rejectionCode = normalizeStarVectorRejection(item.rejection_stage === "generation_limit" ? item.rejection_code : item.rejection_reason, item.rejection_stage);
      const checked = { ...item, rejection_code: rejectionCode, upstream_raw_svg: await verified(root, item.upstream_raw_svg, item.upstream_raw_svg_sha256) };
      if (item.rejection_stage === "sanitizer") {
        checked.sanitizer_stdout = await verified(root, item.sanitizer_stdout, item.sanitizer_stdout_sha256);
        checked.sanitizer_stderr = await verified(root, item.sanitizer_stderr, item.sanitizer_stderr_sha256);
        const event = JSON.parse(await readFile(checked.sanitizer_stdout, "utf8"));
        if (event?.outcome !== "rejected" || event.error_code !== item.rejection_reason || event.canonical_svg_sha256 !== null || event.preview_png_sha256 !== null || JSON.stringify(event.published_paths) !== "[]" || JSON.stringify(event.staging_residue) !== "[]" || event.result_contains_inline_svg !== false) fail(`rejected upstream case ${index} sanitizer evidence differs`);
      }
      cases.push(checked);
    } else fail(`upstream case ${index} has unknown outcome`);
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
