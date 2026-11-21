import assert from "node:assert/strict";
import test from "node:test";

import { verifyCoverage } from "./scail2-fit-pin-coverage.mjs";

test("scripts and separate image modules require review of the whole Git delta", () => {
  const source = { provider: "mlx:scail2_14b", digest: "1", sharedContractDigest: "2", crates: ["crates/media/mlx-gen/mlx-gen-scail2"] };
  const files = ["scripts/ci/video_tier1_comparison.py", "crates/media/mlx-gen/mlx-gen-qwen-image/src/model.rs"];
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
