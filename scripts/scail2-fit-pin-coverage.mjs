// Compare the physical producer's exact commit with a candidate compile target.
// First run without a checksum to inspect the full Git delta. After independent
// review, pass the approved tree_diff_sha256 to print one exact coverage entry.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { providerClosureDigest } from "./inference-closure-digest.mjs";

const ledger = JSON.parse(readFileSync(new URL("../config/inference-provider-closures.json", import.meta.url)));
const revision = /^[0-9a-f]{40}$/u;

// These files form the reviewed diagnostic/bounded-proof delta and the separate
// image follow-up. Other files in scripts/ci include production artifact producers.
const proofInputs = new Set([
  "scripts/ci/flux1_pair/cuda_probe.py",
  "scripts/ci/scail2_plural_fit.py",
  "scripts/ci/scail_plural/README.md",
  "scripts/ci/scail_plural/contract.py",
  "scripts/ci/scail_plural/cuda_stage.py",
  "scripts/ci/scail_plural/wddm_retry.py",
  "scripts/ci/stage_pinned_hub_snapshots.py",
  "scripts/ci/video_tier1_comparison.py",
  "scripts/ci/video_tier1_cuda_diagnostic.py",
  "scripts/ci/video_tier1_cuda_diagnostic_retained.json",
  "scripts/ci/video_tier1_followup.py",
  "scripts/ci/wan_pair/common.rs.in",
  "scripts/ci/wan_pair/conditioning.rs.in",
  "scripts/ci/wddm_process_memory.py",
  "scripts/media_campaign_windows.py",
  "scripts/tests/test_flux1_cuda_probe.py",
  "scripts/tests/test_scail2_cuda_stage.py",
  "scripts/tests/test_scail2_plural_fit.py",
  "scripts/tests/test_stage_pinned_hub_snapshots.py",
  "scripts/tests/test_video_tier1_cuda_diagnostic.py",
  "scripts/tests/test_video_tier1_followup.py",
  "scripts/tests/test_wan_tier1_comparison.py",
  "scripts/tests/test_wddm_process_memory.py",
  "scripts/ci/model-inference-proofs.json",
]);
const imageInputs = new Set([
  "crates/media/candle-gen/candle-gen-sdxl/src/pipeline.rs",
  "crates/media/mlx-gen/mlx-gen-boogu/src/model.rs",
  "crates/media/mlx-gen/mlx-gen-boogu/src/pipeline.rs",
  "crates/media/mlx-gen/mlx-gen-qwen-image/src/model.rs",
]);

function git(repo, ...args) {
  return execFileSync("git", ["-C", repo, ...args], { maxBuffer: 32 * 1024 * 1024 });
}

export function verifyCoverage(source, target, files, diffDigest, approvedDiff) {
  if (source.provider !== target.provider || source.digest !== target.digest ||
      source.sharedContractDigest !== target.sharedContractDigest) {
    throw new Error("The provider or shared native contract changed");
  }
  const crates = new Set([...source.crates, ...target.crates]);
  for (const file of files) {
    // A nested provider can be a sibling of the root mlx-gen crate. Match the
    // actual Cargo owner of these individually audited sibling module paths.
    const owner = file.split("/src/")[0];
    if (!proofInputs.has(file) && !(imageInputs.has(file) && !crates.has(owner))) {
      throw new Error(`Unclassified physical or build input: ${file}`);
    }
  }
  if (approvedDiff !== diffDigest) throw new Error("The complete Git delta lacks matching independent review");
}

export function audit(repo, capture, target, backend, approvedDiff) {
  if (!revision.test(capture) || !revision.test(target) || capture === target || !["mlx", "cuda"].includes(backend)) {
    throw new Error("Expected distinct complete commit IDs and mlx or cuda");
  }
  git(repo, "merge-base", "--is-ancestor", capture, target);
  const provider = `${backend === "cuda" ? "candle" : "mlx"}:scail2_14b`;
  const crateDir = ledger.providers[provider].crate;
  const source = providerClosureDigest({ repo, revision: capture, provider, crateDir });
  const dest = providerClosureDigest({ repo, revision: target, provider, crateDir });
  const args = ["diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--ignore-submodules=none"];
  const raw = git(repo, ...args, "--raw", "--no-abbrev", "-z", capture, target, "--");
  const diffDigest = createHash("sha256").update(raw).digest("hex");
  const files = git(repo, ...args, "--name-only", "-z", capture, target, "--").toString().split("\0").filter(Boolean);
  if (approvedDiff === undefined) {
    return { capture_revision: capture, target_revision: target, backend, changed_files: files,
      tree_diff_sha256: diffDigest, capture_provider_digest: source.digest, target_provider_digest: dest.digest,
      capture_shared_contract_digest: source.sharedContractDigest, target_shared_contract_digest: dest.sharedContractDigest };
  }
  verifyCoverage(source, dest, files, diffDigest, approvedDiff);
  const packaged = ledger.providers[provider];
  if (ledger.inferenceRevision !== target || packaged.digest !== dest.digest ||
      packaged.sharedContractDigest !== dest.sharedContractDigest) {
    throw new Error("The application provider ledger does not describe this exact target revision and both native digests");
  }
  return { capture_revision: capture, target_revision: target, backend, provider_digest: dest.digest,
    shared_contract_digest: dest.sharedContractDigest, tree_diff_sha256: diffDigest };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  if (args.length !== 4 && args.length !== 5) {
    throw new Error("Usage: node scripts/scail2-fit-pin-coverage.mjs INFERENCE_REPO CAPTURE_SHA TARGET_SHA mlx|cuda [APPROVED_GIT_DIFF_SHA256]");
  }
  process.stdout.write(`${JSON.stringify(audit(...args), null, 2)}\n`);
}
