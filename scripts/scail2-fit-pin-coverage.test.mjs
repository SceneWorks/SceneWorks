import assert from "node:assert/strict";
import test from "node:test";

import { verifyCoverage } from "./scail2-fit-pin-coverage.mjs";

test("scripts and separate image modules require review of the whole Git delta", () => {
  const source = { provider: "mlx:scail2_14b", digest: "1", sharedContractDigest: "2", crates: ["crates/media/mlx-gen/mlx-gen-scail2"] };
  const files = ["scripts/ci/video_tier1_comparison.py", "scripts/ci/wan_pair/production_timing.py",
    "scripts/tests/test_scail2_owned_zero_recovery.py", "scripts/tests/test_wan_production_timing.py", "crates/media/mlx-gen/mlx-gen-qwen-image/src/model.rs"];
  assert.doesNotThrow(() => verifyCoverage(source, source, files, "whole-diff", "whole-diff"));
  assert.throws(() => verifyCoverage(source, source, files, "whole-diff", "other-diff"), /independent review/);
  for (const changed of [{ digest: "3" }, { sharedContractDigest: "4" }, { provider: "candle:scail2_14b" }]) {
    assert.throws(() => verifyCoverage(source, { ...source, ...changed }, files, "d", "d"), /native contract changed/);
  }
  for (const outside of [
    "crates/media/mlx-gen/mlx-gen-catalog/src/lib.rs", "crates/bundles/runtime-macos/src/lib.rs",
    "crates/kernels/candle-quant-kernels/kernels/quant.cu", "Cargo.lock", ".github/workflows/release.yml",
    "crates/media/mlx-gen/mlx-gen-qwen-image/build.rs", "crates/media/mlx-gen/mlx-gen-qwen-image/src/new.rs",
    "scripts/ci/flash_attn_prebuilt.py", "scripts/ci/wan_pair/other.rs.in", "scripts/tests/new_probe.py",
  ]) {
    assert.throws(() => verifyCoverage(source, source, [outside], "d", "d"), /Unclassified physical or build input/);
  }
  const linked = { ...source, crates: [...source.crates, "crates/media/mlx-gen/mlx-gen-qwen-image"] };
  assert.throws(() => verifyCoverage(linked, linked, files, "d", "d"), /Unclassified physical or build input/);
});


test("native cache changes require exact source-equivalence review of both closures and the entire delta", () => {
  const source = { provider: "candle:scail2_14b", digest: "old", sharedContractDigest: "contract-old", crates: [] };
  const target = { ...source, digest: "new", sharedContractDigest: "contract-new" };
  const files = ["crates/media/candle-gen/vendor/candle-core/src/tensor.rs"];
  const review = { provider: source.provider, tree_diff_sha256: "reviewed", changed_files: files,
    capture_provider_digest: source.digest, target_provider_digest: target.digest,
    capture_shared_contract_digest: source.sharedContractDigest, target_shared_contract_digest: target.sharedContractDigest,
    review: ["Exact cache implementation reviewed; native representation and execution policies preserved."] };
  assert.doesNotThrow(() => verifyCoverage(source, target, files, "reviewed", "reviewed", review));
  for (const changed of [{ target_provider_digest: "other" }, { capture_provider_digest: "other" },
    { target_shared_contract_digest: "other" }, { capture_shared_contract_digest: "other" },
    { tree_diff_sha256: "other" }, { changed_files: [] }, { provider: "mlx:scail2_14b" }, { review: [] }]) {
    assert.throws(() => verifyCoverage(source, target, files, "reviewed", "reviewed", { ...review, ...changed }), /does not match/);
  }
  assert.throws(() => verifyCoverage(source, target, [...files, "Cargo.lock"], "reviewed", "reviewed", review), /does not match/);
  assert.throws(() => verifyCoverage(source, target, files, "reviewed", "other", review), /independent review/);
  assert.throws(() => verifyCoverage(source, target, files, "reviewed", "reviewed"), /native contract changed/);
});
