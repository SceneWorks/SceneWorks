/**
 * Every site that names an ONNX Runtime version must name the SAME one, and it must be the
 * version the `ort` crate is compiled to accept.
 *
 * WHY — the `ort` crate dlopens onnxruntime through its `load-dynamic` feature and refuses, at
 * RUNTIME, any dylib whose minor is lower than the `api-N` feature it was built with
 * (`LoadError::BadVersion`). So two independent things must agree:
 *
 *   1. what the BINARY demands — `ort`'s `api-N` feature, bound to
 *      `PROVISIONED_ONNXRUNTIME_MINOR` by a `const` assert in `pose_jobs.rs`; and
 *   2. what a USER actually ends up with — four separate download/stage pins, one per
 *      distribution channel (Windows first-run provisioner, Linux first-run provisioner, the
 *      macOS app-bundle staging script, the Docker image).
 *
 * The `const` assert covers (1) against a hand-copied constant. Nothing covered (2): bump the
 * wheel URLs without touching the constant — or touch the constant without the wheels — and the
 * assert still passes while installs download a runtime the binary rejects. That is not a
 * theoretical gap. Dependabot's cargo-group bump (#2092) moved `ort` rc.12 -> rc.13, raising its
 * default `api-24` -> `api-27` against a fleet pinned to onnxruntime 1.26.0, and every `ort`
 * surface (DWPose pose_detect, YOLO person_detect, ONNX upscale) broke on every platform. It
 * compiled clean and passed CI, because nothing in a compile can see a `.whl` URL.
 *
 * This check closes that: one version string, enforced across a Rust crate, two Rust provisioners,
 * a Python staging script and a Dockerfile — surfaces no single `cargo test` can read (and that a
 * crate-local `include_str!` cannot reach either; Docker builds from a subset context).
 *
 * A pattern that matches NOTHING is a failure, not a pass. Guards rot by silently ceasing to
 * match after a rename or a refactor, and a guard that finds nothing looks exactly like a guard
 * that found everything in order.
 */

const SEMVER = String.raw`\d+\.\d+\.\d+`;

/**
 * Sites carrying a full `x.y.z` ONNX Runtime version. `min` is how many matches the pattern must
 * find; falling below it means the pattern went stale, which is reported as its own failure.
 */
export const ONNXRUNTIME_VERSION_SITES = [
  {
    file: "apps/desktop/src/cuda_provision.rs",
    what: "Windows first-run provisioner (downloaded into %APPDATA%\\SceneWorks\\gpu-runtime)",
    patterns: [
      {
        label: "onnxruntime-gpu wheel URL",
        re: new RegExp(String.raw`onnxruntime_gpu-(${SEMVER})-cp\d+-cp\d+-win_amd64\.whl`, "g"),
        min: 1,
      },
      {
        label: "REDIST_VERSION marker",
        re: new RegExp(String.raw`REDIST_VERSION[^=]*=\s*"[^"]*?-ort(${SEMVER})-`, "g"),
        min: 1,
      },
    ],
  },
  {
    file: "apps/desktop/src/linux_cuda_provision.rs",
    what: "Linux first-run provisioner",
    patterns: [
      {
        label: "onnxruntime-gpu wheel URL",
        re: new RegExp(String.raw`onnxruntime_gpu-(${SEMVER})-cp\d+-cp\d+-manylinux`, "g"),
        min: 1,
      },
      {
        label: "REDIST_VERSION marker",
        re: new RegExp(String.raw`REDIST_VERSION[^=]*=\s*"[^"]*?-ort(${SEMVER})-`, "g"),
        min: 1,
      },
      {
        // The SONAME the resolver's fixtures build and assert on. A bump that misses these
        // leaves the tests green while asserting against a version nobody ships.
        label: "libonnxruntime SONAME literal",
        re: new RegExp(String.raw`libonnxruntime\.so\.(${SEMVER})`, "g"),
        min: 1,
      },
    ],
  },
  {
    file: "apps/desktop/src/setup.rs",
    what: "runtime resolver fixtures",
    patterns: [
      {
        label: "libonnxruntime SONAME literal",
        re: new RegExp(String.raw`libonnxruntime\.so\.(${SEMVER})`, "g"),
        min: 1,
      },
    ],
  },
  {
    file: "apps/desktop/scripts/stage-onnxruntime.py",
    what: "macOS app-bundle staging (shipped inside the .app, not downloaded)",
    patterns: [
      {
        label: "ONNXRUNTIME_VERSION",
        re: new RegExp(String.raw`ONNXRUNTIME_VERSION\s*=\s*"(${SEMVER})"`, "g"),
        min: 1,
      },
      {
        // One per CPython ABI tag. They are pinned URL+sha256, so a half-applied bump shows up
        // here as disagreeing versions rather than as a sha mismatch at build time.
        label: "pinned macOS wheel URL",
        re: new RegExp(String.raw`onnxruntime-(${SEMVER})-cp\d+-cp\d+-macosx`, "g"),
        min: 1,
      },
    ],
  },
  {
    file: "docker/rust.Dockerfile",
    what: "Docker candle worker image",
    patterns: [
      {
        label: "ONNXRUNTIME_GPU_VERSION",
        re: new RegExp(String.raw`ARG\s+ONNXRUNTIME_GPU_VERSION=(${SEMVER})`, "g"),
        min: 1,
      },
    ],
  },
];

/**
 * The compile-time floor the `ort` crate is held to. A bare minor, so it is compared against the
 * `y` of the consensus `x.y.z` above.
 */
export const ONNXRUNTIME_MINOR_SITE = {
  file: "crates/sceneworks-worker/src/pose_jobs.rs",
  what: "compile-time floor asserted against ort::MINOR_VERSION",
  pattern: {
    label: "PROVISIONED_ONNXRUNTIME_MINOR",
    re: /PROVISIONED_ONNXRUNTIME_MINOR:\s*u32\s*=\s*(\d+)\s*;/g,
    min: 1,
  },
};

function collect(text, pattern) {
  // `matchAll` needs the `g` flag and a fresh lastIndex; the descriptors are module-level and
  // reused across calls, so never iterate a shared regex statefully.
  return [...String(text ?? "").matchAll(new RegExp(pattern.re.source, "g"))].map(
    (match) => match[1],
  );
}

/**
 * `sources` maps each site's `file` to its text. Returns a flat list of violations; empty means
 * every site agrees and the compile-time floor matches.
 */
export function findOnnxruntimePinViolations(sources) {
  const violations = [];
  const found = [];

  for (const site of ONNXRUNTIME_VERSION_SITES) {
    const text = sources[site.file];
    if (typeof text !== "string") {
      violations.push({ kind: "missing-file", file: site.file });
      continue;
    }
    for (const pattern of site.patterns) {
      const values = collect(text, pattern);
      if (values.length < pattern.min) {
        violations.push({
          kind: "stale-pattern",
          file: site.file,
          label: pattern.label,
          expected: pattern.min,
          got: values.length,
        });
        continue;
      }
      for (const version of values) {
        found.push({ file: site.file, label: pattern.label, version });
      }
    }
  }

  const versions = [...new Set(found.map((entry) => entry.version))].sort();
  if (versions.length > 1) {
    violations.push({ kind: "disagree", versions, found });
  }

  const minorText = sources[ONNXRUNTIME_MINOR_SITE.file];
  if (typeof minorText !== "string") {
    violations.push({ kind: "missing-file", file: ONNXRUNTIME_MINOR_SITE.file });
    return violations;
  }
  const minors = collect(minorText, ONNXRUNTIME_MINOR_SITE.pattern);
  if (minors.length < ONNXRUNTIME_MINOR_SITE.pattern.min) {
    violations.push({
      kind: "stale-pattern",
      file: ONNXRUNTIME_MINOR_SITE.file,
      label: ONNXRUNTIME_MINOR_SITE.pattern.label,
      expected: ONNXRUNTIME_MINOR_SITE.pattern.min,
      got: minors.length,
    });
    return violations;
  }

  // Only meaningful once the download pins agree on ONE version — otherwise "which one should the
  // floor match?" has no answer, and the `disagree` violation above is the thing to fix first.
  if (versions.length === 1) {
    const declared = Number(minors[0]);
    const actual = Number(versions[0].split(".")[1]);
    if (declared !== actual) {
      violations.push({
        kind: "floor-mismatch",
        file: ONNXRUNTIME_MINOR_SITE.file,
        declared,
        provisioned: versions[0],
      });
    }
  }

  return violations;
}

export function formatOnnxruntimePinViolations(violations) {
  return violations
    .map((violation) => {
      switch (violation.kind) {
        case "missing-file":
          return `  ${violation.file}: not found — a pin site moved or was deleted`;
        case "stale-pattern":
          return (
            `  ${violation.file}: the "${violation.label}" pattern matched ${violation.got} time(s), ` +
            `expected at least ${violation.expected} — the guard went stale, fix ` +
            `scripts/lib/onnxruntime-pins.mjs rather than deleting the check`
          );
        case "disagree":
          return [
            `  pin sites name ${violation.versions.length} different ONNX Runtime versions ` +
              `(${violation.versions.join(", ")}) — a half-applied bump:`,
            ...violation.found.map(
              (entry) => `      ${entry.version}  ${entry.file} (${entry.label})`,
            ),
          ].join("\n");
        case "floor-mismatch":
          return (
            `  ${violation.file}: PROVISIONED_ONNXRUNTIME_MINOR = ${violation.declared}, but every ` +
            `install downloads onnxruntime ${violation.provisioned}. The constant is what binds ` +
            `ort's api-N feature to reality, so this makes the compile-time assert meaningless`
          );
        default:
          return `  ${JSON.stringify(violation)}`;
      }
    })
    .join("\n");
}
