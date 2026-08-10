import assert from "node:assert/strict";
import test from "node:test";

import { findOnnxruntimePinViolations } from "./onnxruntime-pins.mjs";

// A minimal stand-in for each real pin site: just enough text to exercise every pattern, so a
// pattern that stops matching the REAL files is caught by the check itself (which requires a
// minimum match count) rather than passing vacuously here.
function sources({ win = "1.26.0", linux = "1.26.0", mac = "1.26.0", docker = "1.26.0", soname = "1.26.0", minor = 26 } = {}) {
  return {
    "apps/desktop/src/cuda_provision.rs": [
      `const REDIST_VERSION: &str = "cuda12.9-ort${win}-cudnn9.23-1";`,
      `url: "https://files.pythonhosted.org/x/onnxruntime_gpu-${win}-cp312-cp312-win_amd64.whl",`,
    ].join("\n"),
    "apps/desktop/src/linux_cuda_provision.rs": [
      `const REDIST_VERSION: &str = "cuda12.9-ort${linux}-cudnn9.23-linux-x86_64-1";`,
      `url: "https://files.pythonhosted.org/x/onnxruntime_gpu-${linux}-cp312-cp312-manylinux_2_27_x86_64.whl",`,
      `assert!(dest.join("libonnxruntime.so.${soname}").is_file());`,
    ].join("\n"),
    "apps/desktop/src/setup.rs": `touch(&root.join("onnxruntime/capi/libonnxruntime.so.${soname}"));`,
    "apps/desktop/scripts/stage-onnxruntime.py": [
      `ONNXRUNTIME_VERSION = "${mac}"`,
      `"onnxruntime-${mac}-cp312-cp312-macosx_14_0_arm64.whl"`,
      `"onnxruntime-${mac}-cp313-cp313-macosx_14_0_arm64.whl"`,
    ].join("\n"),
    "docker/rust.Dockerfile": `ARG ONNXRUNTIME_GPU_VERSION=${docker}`,
    "crates/sceneworks-worker/src/pose_jobs.rs": `const PROVISIONED_ONNXRUNTIME_MINOR: u32 = ${minor};`,
  };
}

test("agreeing pins pass", () => {
  assert.deepEqual(findOnnxruntimePinViolations(sources()), []);
});

test("a bump applied to Windows but not Linux is rejected — the half-applied shape", () => {
  const violations = findOnnxruntimePinViolations(sources({ win: "1.28.0" }));
  const disagree = violations.find((v) => v.kind === "disagree");
  assert.ok(disagree, "a version split across provisioners must be reported");
  assert.deepEqual(disagree.versions, ["1.26.0", "1.28.0"]);
});

test("a bump that misses the macOS staging script is rejected", () => {
  const violations = findOnnxruntimePinViolations(
    sources({ win: "1.28.0", linux: "1.28.0", docker: "1.28.0", soname: "1.28.0", minor: 28 }),
  );
  assert.ok(violations.some((v) => v.kind === "disagree"));
});

test("a bump that misses the Dockerfile is rejected", () => {
  const violations = findOnnxruntimePinViolations(
    sources({ win: "1.28.0", linux: "1.28.0", mac: "1.28.0", soname: "1.28.0", minor: 28 }),
  );
  assert.ok(violations.some((v) => v.kind === "disagree"));
});

test("wheels bumped everywhere but the compile-time floor left behind is rejected", () => {
  // The dangerous direction: installs would download 1.28 while the binary still declares 26, so
  // the `const` assert in pose_jobs.rs keeps passing against a number that no longer describes
  // reality.
  const violations = findOnnxruntimePinViolations(
    sources({ win: "1.28.0", linux: "1.28.0", mac: "1.28.0", docker: "1.28.0", soname: "1.28.0" }),
  );
  const mismatch = violations.find((v) => v.kind === "floor-mismatch");
  assert.ok(mismatch, "a stale PROVISIONED_ONNXRUNTIME_MINOR must be reported");
  assert.equal(mismatch.declared, 26);
  assert.equal(mismatch.provisioned, "1.28.0");
});

test("the floor moving without the wheels is rejected", () => {
  const violations = findOnnxruntimePinViolations(sources({ minor: 27 }));
  assert.ok(violations.some((v) => v.kind === "floor-mismatch"));
});

test("a pattern that no longer matches fails instead of passing vacuously", () => {
  // Guard rot: the Dockerfile ARG gets renamed, the regex finds nothing, and a check that only
  // compared what it found would go green while the Dockerfile drifted freely.
  const all = sources();
  all["docker/rust.Dockerfile"] = "ARG ORT_VERSION=1.26.0";
  const violations = findOnnxruntimePinViolations(all);
  const stale = violations.find((v) => v.kind === "stale-pattern");
  assert.ok(stale, "a pattern matching nothing must be its own failure");
  assert.equal(stale.file, "docker/rust.Dockerfile");
});

test("a missing pin site is reported rather than skipped", () => {
  const all = sources();
  delete all["apps/desktop/scripts/stage-onnxruntime.py"];
  const violations = findOnnxruntimePinViolations(all);
  assert.ok(
    violations.some(
      (v) => v.kind === "missing-file" && v.file === "apps/desktop/scripts/stage-onnxruntime.py",
    ),
  );
});

test("a missing floor file is reported", () => {
  const all = sources();
  delete all["crates/sceneworks-worker/src/pose_jobs.rs"];
  const violations = findOnnxruntimePinViolations(all);
  assert.ok(
    violations.some(
      (v) => v.kind === "missing-file" && v.file === "crates/sceneworks-worker/src/pose_jobs.rs",
    ),
  );
});
