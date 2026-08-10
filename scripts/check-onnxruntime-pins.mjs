import { readFile } from "node:fs/promises";

import {
  ONNXRUNTIME_MINOR_SITE,
  ONNXRUNTIME_VERSION_SITES,
  findOnnxruntimePinViolations,
  formatOnnxruntimePinViolations,
} from "./lib/onnxruntime-pins.mjs";

// Every channel that puts an onnxruntime on a user's machine must put the SAME one there, and it
// must be the version `ort` was compiled to accept. See scripts/lib/onnxruntime-pins.mjs for why
// a compile-time assert alone cannot cover this.

const files = [...ONNXRUNTIME_VERSION_SITES.map((site) => site.file), ONNXRUNTIME_MINOR_SITE.file];

const sources = Object.fromEntries(
  await Promise.all(
    files.map(async (file) => {
      try {
        return [file, await readFile(file, "utf8")];
      } catch {
        return [file, null];
      }
    }),
  ),
);

const violations = findOnnxruntimePinViolations(sources);

if (violations.length > 0) {
  console.error(
    `ONNX Runtime pin mismatch.\n\n` +
      `The 'ort' crate dlopens onnxruntime and rejects, at RUNTIME, any dylib older than the\n` +
      `api-N feature it was built with — "Failed to load ONNX Runtime dylib: BadVersion". So the\n` +
      `version every distribution channel installs and the version the binary demands must agree,\n` +
      `and neither a compile nor a test can see a .whl URL.\n\n` +
      `${formatOnnxruntimePinViolations(violations)}\n\n` +
      `Before bumping the runtime to satisfy a newer 'ort': onnxruntime-gpu 1.27+ moved its CUDA\n` +
      `execution provider to CUDA 13, and 1.27's own cu13 deps do not resolve on PyPI. Pinning\n` +
      `ort's api-N feature down (root Cargo.toml) is usually the correct direction.`,
  );
  process.exit(1);
}

const version = [
  ...new Set(
    ONNXRUNTIME_VERSION_SITES.flatMap((site) =>
      site.patterns.flatMap((pattern) =>
        [...sources[site.file].matchAll(new RegExp(pattern.re.source, "g"))].map((m) => m[1]),
      ),
    ),
  ),
];

console.log(
  `ONNX Runtime pin check passed (onnxruntime ${version[0]} across ` +
    `${ONNXRUNTIME_VERSION_SITES.length} distribution sites + the compile-time floor).`,
);
