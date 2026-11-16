#!/usr/bin/env node
// Print why a candle catalog walk failed, into the job log.
//
// The candle lane runs run.sh inside scripts/ci/windows_cuda_execution.py. When that step failed,
// the job log showed only the wrapper's one-line JSON (run 43794813358, ~4 h): the wrapper's
// receipt (which holds its own error) sat outside the uploaded work dir, and nothing printed the
// walk's log or its failing cases. This prints, in order: the wrapper receipt's diagnostics (and
// copies the receipt into the work dir so the artifact keeps it), the tail of the walk log, and
// every failing case from the newest summary JSON. It only reports; it never changes the result.
import { copyFile, mkdir, readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const DEFAULT_TAIL_LINES = 200;
// The statuses measure-memory-catalog.mjs does not count as a failed anchor.
const PASSING = new Set(["committed", "committed_exceeded", "captured", "exceeded", "artifact_unsupported"]);

async function readOptional(file) {
  try {
    return await readFile(file, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

function receiptLines(receipt) {
  const lines = [`resource_completed: ${receipt.resource_completed}`];
  if (receipt.error) lines.push(`error: ${receipt.error}`);
  if (receipt.child) {
    const { returncode, timeout, seconds, owned_tree_empty: empty } = receipt.child;
    lines.push(`child: returncode=${returncode} timeout=${timeout} seconds=${seconds} owned_tree_empty=${empty}`);
  } else {
    lines.push("child: never launched (the wrapper failed before starting the walk)");
  }
  for (const key of ["lock_wait_seconds", "post_child_idle_verified", "admission_policy"]) {
    if (receipt[key] !== undefined) lines.push(`${key}: ${receipt[key]}`);
  }
  return lines;
}

export async function walkReport({ workDir, receipt, tailLines = DEFAULT_TAIL_LINES }) {
  const out = [];
  await mkdir(workDir, { recursive: true });

  out.push("=== CUDA execution wrapper receipt");
  const receiptText = receipt ? await readOptional(receipt) : null;
  if (receiptText === null) {
    out.push(receipt ? `no receipt at ${receipt}` : "no receipt path given");
  } else {
    await copyFile(receipt, path.join(workDir, "cuda-resource-receipt.json"));
    try {
      out.push(...receiptLines(JSON.parse(receiptText)));
    } catch (error) {
      out.push(`receipt is not JSON (${error.message}); raw:`, receiptText.trimEnd());
    }
  }

  out.push("", "=== walk log (campaign.log)");
  const log = await readOptional(path.join(workDir, "campaign.log"));
  if (log === null) {
    out.push(`no campaign.log in ${workDir}: run.sh never started the walk`);
  } else {
    const lines = log.replace(/\n$/, "").split("\n");
    const shown = lines.slice(-tailLines);
    out.push(`last ${shown.length} of ${lines.length} line(s):`, ...shown, "=== end walk log");
  }

  out.push("", "=== failing cases (newest summary-*.json)");
  const summaries = (await readdir(workDir)).filter((name) => /^summary-.*\.json$/.test(name)).sort();
  if (summaries.length === 0) {
    out.push("no summary JSON: the walk did not finish any anchor");
  } else {
    const newest = summaries.at(-1);
    const summary = JSON.parse(await readFile(path.join(workDir, newest), "utf8"));
    const failed = (summary.results ?? []).filter((result) => !PASSING.has(result.status));
    out.push(`${newest}: ${summary.results?.length ?? 0} result(s), ${failed.length} failing`);
    for (const result of failed) out.push(`- ${result.key}: ${result.status}${result.reason ? ` (${result.reason})` : ""}`);
  }
  return `${out.join("\n")}\n`;
}

function parseArgs(argv) {
  const args = { tailLines: DEFAULT_TAIL_LINES };
  for (let index = 0; index < argv.length; index += 2) {
    const [flag, value] = [argv[index], argv[index + 1]];
    if (value === undefined) throw new Error(`${flag} needs a value`);
    if (flag === "--work-dir") args.workDir = value;
    else if (flag === "--receipt") args.receipt = value;
    else if (flag === "--tail") {
      if (!/^[1-9][0-9]*$/.test(value)) throw new Error("--tail must be a positive integer");
      args.tailLines = Number(value);
    } else throw new Error(`unknown flag ${flag}`);
  }
  if (!args.workDir) throw new Error("--work-dir is required");
  return args;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    process.stdout.write(await walkReport(parseArgs(process.argv.slice(2))));
  } catch (error) {
    process.stderr.write(`walk-report: ${error.message}\n`);
    process.exit(1);
  }
}
