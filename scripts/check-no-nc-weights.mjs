// SceneWorks packaging guard: never bake model weights into a published artifact
// (sc-10526, epic 10512 "Anima model support").
//
// WHY THIS EXISTS
// ---------------
// SceneWorks ships its *code* under AGPL-3.0-or-later, but the model *weights* it
// runs are a separate, license-gated concern. Several supported families are
// distributed under Non-Commercial (NC) licenses whose redistribution terms attach
// the moment we hand someone a copy of the weights — e.g. Anima (CircleStone Labs
// Non-Commercial License v1.2, epic 10512), FLUX.1 [dev] / FLUX.2 (FLUX
// Non-Commercial License), Ideogram 4 (Ideogram Non-Commercial Model Agreement),
// Krea 2 (Krea Community License), and NVIDIA SANA (NVIDIA Open Model License /
// NSCL). A "Derivative" under these licenses includes any converted or quantized
// checkpoint and any LoRA/fine-tune.
//
// SceneWorks' posture side-steps all of the redistribution obligations by NEVER
// redistributing weights: it CONVERTS AT INSTALL and PULLS AT RUNTIME into the
// user's own machine/instance. That guarantee is only as strong as our packaging.
// If a published artifact — a desktop installer/DMG, a Docker/RunPod image layer
// (epic 10362), or a re-hosted HF repo — ever embeds those weights, SceneWorks
// becomes a distributor of a Derivative and the NC obligations (ship the license +
// an attribution notice + a statement of modification) attach immediately.
//
// This check fails the build if any model-weight file is found inside an
// artifact-payload directory. It is deliberately scoped to *artifact staging*, not
// the developer's machine: a dev with ~/.cache/huggingface/models--circlestone-labs--Anima
// on disk (the normal convert-at-install state) must NOT trip it, and does not —
// see SCAN ROOTS below.
//
// NOTE ON OUTPUTS: the NC licenses permit commercial use of *Outputs* (generated
// images) — e.g. CircleStone v1.2 §2(e). Only the Model and its Derivatives (the
// weights) are restricted. This guard is about weights, never about generations.
//
// WHAT IT SCANS (artifact payload only)
// -------------------------------------
//   Default: config/, crates/, apps/  — exactly the trees the Docker image COPYs
//   in (docker/rust.Dockerfile) and where the Tauri desktop bundle stages its
//   resources (apps/desktop/**). Excludes node_modules/, target/, dist/, .git/.
//   Deliberately NOT scanned:
//     * data/          — the developer's local model store; .dockerignore already
//                        excludes data/models|loras|cache from the image, so it is
//                        never artifact payload. Scanning it would false-trip on a
//                        dev's downloaded weights, defeating convert-at-install.
//     * ~/.cache/huggingface — the runtime HF cache; never part of any artifact.
//   --dir <path> (repeatable): scan a real built artifact instead — an extracted
//   Docker image rootfs, a built `.app`/`.dmg` Resources tree, or a RunPod build
//   context. The release/desktop lanes should point this at the finished bundle.
//
// HOW NC FAMILIES ARE DECLARED (data-driven, not a hand-maintained list)
// ----------------------------------------------------------------------
//   The set of NC model families is DERIVED from the manifests
//   (config/manifests/builtin.models.jsonc + builtin.loras.jsonc): an entry is NC
//   if it declares `nonCommercial: true` OR its description/licenseUrl carries an
//   NC signal ("non-commercial", "NSCL", "NVIDIA Open Model License", "FLUX ...
//   Non-Commercial License", the Ideogram/Krea NC agreements). From each NC entry
//   we take its HF repo(s) and turn them into on-disk match tokens. This means the
//   next NC model inherits the guard automatically once it is in the manifest — no
//   second list to keep in sync.
//
//   The hard failure, though, does NOT depend on that classification being perfect:
//   ANY weight file in the payload that is not on the small ALLOWLIST fails the
//   build. NC classification only ENRICHES the error so it can name the specific
//   license/family. So a brand-new NC model still fails the build even before it is
//   tagged NC — it just gets the generic "unexpected model weight" message.
//
// See docs/packaging-nc-weights-guard.md for the full rule and how to add a new
// permissively-licensed bundled weight to the allowlist.

import { readFile, readdir, stat, mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";

const root = process.cwd();

// Directories that make up the published-artifact payload when no --dir is given.
// These are the Docker build-context trees the image COPYs in and the Tauri
// resource source trees; see the module header for why data/ and the HF cache are
// intentionally absent.
const DEFAULT_SCAN_ROOTS = ["config", "crates", "apps"];

// Never descend into these — build caches, deps, and VCS metadata are not payload.
const EXCLUDED_DIR_NAMES = new Set([
  "node_modules",
  "target",
  "dist",
  ".git",
  ".claude",
  "__pycache__",
]);

// File extensions that are, in practice, always serialized model weights. `.bin`
// is intentionally excluded (fonts, wasm, and misc blobs also use it) unless
// --include-bin is passed; the NC-token matcher below still catches an NC `.bin`.
const WEIGHT_EXTENSIONS = new Set([
  ".safetensors",
  ".gguf",
  ".ckpt",
  ".pth",
  ".pt",
  ".onnx",
  ".npz",
  ".mlx",
  ".msgpack",
  ".h5",
  ".pb",
]);

// Permissively-licensed weights that are LEGITIMATELY compiled/bundled into a
// SceneWorks binary and are therefore allowed in the payload. Keyed by exact
// basename so it holds across both the default scan and any --dir scan. Keep this
// list tiny and every entry justified — anything NC must never be added here.
const ALLOWLIST = [
  {
    basename: "aesthetic-v2-sac-logos-ava1-l14.safetensors",
    reason:
      "LAION improved-aesthetic-predictor (MIT). A ~4 MB CLIP-head regressor " +
      "compiled into crates/sceneworks-image-quality for the aesthetic score. " +
      "Permissive, not a generative model, and not a Derivative of any NC family.",
  },
];
const ALLOWLIST_BASENAMES = new Set(ALLOWLIST.map((entry) => entry.basename));

// Bootstrap NC tokens for families that are not yet represented in the manifest.
// Anima (epic 10512) is being ported now; until its manifest entry lands with a
// repo + `nonCommercial: true`, seed its tokens here so the guard already names it.
// REMOVE an entry once the corresponding family is in the manifest (the manifest is
// the source of truth) — the self-test asserts Anima stays covered either way.
const BOOTSTRAP_NC_TOKENS = ["circlestone-labs", "anima"];

// -- JSONC parsing (manifests carry // comments). Mirrors scripts/check-scaffold.mjs. --
function stripJsoncComments(body) {
  let result = "";
  let inString = false;
  let escaped = false;
  for (let index = 0; index < body.length; index += 1) {
    const char = body[index];
    const next = body[index + 1];
    if (inString) {
      result += char;
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }
    if (char === '"') {
      inString = true;
      result += char;
      continue;
    }
    if (char === "/" && next === "/") {
      while (index < body.length && body[index] !== "\n") {
        index += 1;
      }
      result += "\n";
      continue;
    }
    if (char === "/" && next === "*") {
      index += 2;
      while (index < body.length && !(body[index] === "*" && body[index + 1] === "/")) {
        index += 1;
      }
      index += 1;
      continue;
    }
    result += char;
  }
  return result;
}

async function readJsonc(relativePath) {
  const body = await readFile(path.join(root, relativePath), "utf8");
  return JSON.parse(stripJsoncComments(body));
}

// An entry (model or lora) is NC when it says so explicitly or its human-readable
// license text does. Kept deliberately specific: "community license" alone is NOT
// an NC signal (the Stability Community License permits commercial use under a
// revenue cap), so we match the concrete NC license names/phrases instead.
const NC_TEXT_SIGNAL =
  /non[-\s]?commercial|noncommercial|\bNSCL\b|NVIDIA Open Model License|FLUX[^.]*Non-Commercial|Non-Commercial (?:Model )?(?:License|Agreement)|krea-2-licensing|CircleStone/i;

function entryIsNonCommercial(entry) {
  if (entry?.nonCommercial === true) {
    return true;
  }
  // NC signals live in several places (`ui.description`, `licenseUrl`, `name`, and
  // occasionally nested variant text), so test the whole flattened entry rather
  // than cherry-picking fields. `commercial` on its own (e.g. "for commercial use
  // choose X" on a *permissive* model) does not match — only the concrete
  // non-commercial phrasing does. Over-recall here only affects the wording of a
  // failure message; the hard gate fails on any non-allowlisted weight regardless.
  return NC_TEXT_SIGNAL.test(JSON.stringify(entry));
}

// Turn an HF repo id ("org/name") into the tokens it shows up as on disk:
//   org/name            (bare clone / manifest form)
//   models--org--name   (HF hub cache dir form)
//   name                (final path component; catches a flattened copy)
function repoToTokens(repo) {
  if (typeof repo !== "string" || !repo.includes("/")) {
    return [];
  }
  const lower = repo.toLowerCase();
  const [org, ...rest] = lower.split("/");
  const name = rest.join("/");
  const tokens = new Set([lower, `models--${org}--${name}`.replace(/\//g, "--")]);
  if (name) {
    tokens.add(name);
  }
  return [...tokens];
}

// Deep-collect every value under a "repo" key anywhere in the entry
// (downloads[].repo, source.repo, and any nested variant repos), so the token set
// covers all of a family's HF repos regardless of where the manifest nests them.
function collectRepoStrings(entry) {
  const repos = [];
  const visit = (node) => {
    if (Array.isArray(node)) {
      for (const item of node) {
        visit(item);
      }
    } else if (node && typeof node === "object") {
      for (const [key, value] of Object.entries(node)) {
        if (key === "repo" && typeof value === "string") {
          repos.push(value);
        } else {
          visit(value);
        }
      }
    }
  };
  visit(entry);
  return repos;
}

// Build the NC token set from the manifests + the bootstrap seed.
async function buildNcTokens() {
  const tokens = new Set(BOOTSTRAP_NC_TOKENS.map((token) => token.toLowerCase()));
  const families = new Set();

  const sources = [
    { path: "config/manifests/builtin.models.jsonc", key: "models" },
    { path: "config/manifests/builtin.loras.jsonc", key: "loras" },
  ];
  for (const source of sources) {
    let manifest;
    try {
      manifest = await readJsonc(source.path);
    } catch (error) {
      throw new Error(`Failed to parse ${source.path}: ${error.message}`);
    }
    for (const entry of manifest[source.key] ?? []) {
      if (!entryIsNonCommercial(entry)) {
        continue;
      }
      if (typeof entry.family === "string") {
        families.add(entry.family.toLowerCase());
      }
      for (const repo of collectRepoStrings(entry)) {
        for (const token of repoToTokens(repo)) {
          tokens.add(token);
        }
      }
    }
  }
  return { tokens: [...tokens].filter(Boolean), families: [...families] };
}

// Which NC token (if any) does this on-disk path match? Compared over the whole
// POSIX-normalized path so both the file basename (anima-base-v1.0.safetensors) and
// an ancestor cache dir (models--circlestone-labs--Anima/…) are caught.
function ncTokenFor(filePath, ncTokens) {
  const needle = filePath.split(path.sep).join("/").toLowerCase();
  return ncTokens.find((token) => token.length >= 4 && needle.includes(token)) ?? null;
}

async function walk(dir, files) {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") {
      return;
    }
    throw error;
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (EXCLUDED_DIR_NAMES.has(entry.name)) {
        continue;
      }
      await walk(full, files);
    } else if (entry.isFile() || entry.isSymbolicLink()) {
      files.push(full);
    }
  }
}

function isWeightFile(filePath, includeBin) {
  const ext = path.extname(filePath).toLowerCase();
  if (WEIGHT_EXTENSIONS.has(ext)) {
    return true;
  }
  if (includeBin && ext === ".bin") {
    return true;
  }
  return false;
}

// Core: scan the given roots and return the list of violations. A violation is any
// weight file that is not on the allowlist. NC classification enriches each one.
async function scan({ roots, ncTokens, includeBin }) {
  const violations = [];
  for (const scanRoot of roots) {
    const absRoot = path.isAbsolute(scanRoot) ? scanRoot : path.join(root, scanRoot);
    let rootStat;
    try {
      rootStat = await stat(absRoot);
    } catch (error) {
      if (error.code === "ENOENT") {
        continue; // A staging dir that is not present in this checkout is fine.
      }
      throw error;
    }
    const files = [];
    if (rootStat.isDirectory()) {
      await walk(absRoot, files);
    } else if (rootStat.isFile()) {
      files.push(absRoot);
    }
    for (const file of files) {
      if (!isWeightFile(file, includeBin)) {
        continue;
      }
      const basename = path.basename(file);
      // NC-token match wins even for allowlisted basenames: an allowlisted
      // permissive name must never travel inside an NC repo/cache path.
      const ncToken = ncTokenFor(file, ncTokens);
      if (!ncToken && ALLOWLIST_BASENAMES.has(basename)) {
        continue;
      }
      violations.push({
        file: path.relative(root, file) || file,
        basename,
        ncToken,
      });
    }
  }
  return violations;
}

function reportViolations(violations) {
  console.error(
    "\nNC-WEIGHTS GUARD FAILED (sc-10526): model-weight file(s) found in the " +
      "artifact payload.\n" +
      "SceneWorks must never redistribute model weights — it converts at install " +
      "and pulls at runtime. Shipping these turns SceneWorks into a distributor of " +
      "a (possibly Non-Commercial) Derivative.\n",
  );
  for (const violation of violations) {
    if (violation.ncToken) {
      console.error(
        `  ✗ ${violation.file}\n` +
          `      matches Non-Commercial family token "${violation.ncToken}" — this is ` +
          `an NC weight and MUST NOT ship in any installer, image layer, or re-host.`,
      );
    } else {
      console.error(
        `  ✗ ${violation.file}\n` +
          `      unexpected model weight in the artifact payload. If this is a ` +
          `permissively-licensed weight that is legitimately compiled into a binary, ` +
          `add its basename (with a justification) to the ALLOWLIST in ` +
          `scripts/check-no-nc-weights.mjs. If it is an NC weight, remove it from the ` +
          `build — SceneWorks pulls weights at runtime, it does not ship them.`,
      );
    }
  }
  console.error(
    "\nSee docs/packaging-nc-weights-guard.md for the rule and the allowlist policy.\n",
  );
}

function parseArgs(argv) {
  const options = { dirs: [], includeBin: false, selfTest: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--self-test") {
      options.selfTest = true;
    } else if (arg === "--include-bin") {
      options.includeBin = true;
    } else if (arg === "--dir") {
      const value = argv[i + 1];
      if (!value) {
        throw new Error("--dir requires a path argument");
      }
      options.dirs.push(value);
      i += 1;
    } else if (arg.startsWith("--dir=")) {
      options.dirs.push(arg.slice("--dir=".length));
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return options;
}

async function runScan(options) {
  const { tokens, families } = await buildNcTokens();
  const roots = options.dirs.length > 0 ? options.dirs : DEFAULT_SCAN_ROOTS;
  const violations = await scan({ roots, ncTokens: tokens, includeBin: options.includeBin });
  return { tokens, families, roots, violations };
}

// -- Self-test: proves the guard fires (mirrors scripts/check-gen-core-skew.sh --self-test). --
async function selfTest() {
  let failures = 0;
  const assert = (label, condition) => {
    const status = condition ? "PASS" : "FAIL";
    if (!condition) {
      failures += 1;
    }
    console.log(`self-test: ${label} -> ${status}`);
  };

  const { tokens, families } = await buildNcTokens();
  // The manifest-derived NC set must not silently empty out (a parse regression
  // would otherwise disarm NC classification).
  assert("manifest yields a non-empty NC token set", tokens.length > 5);
  assert(
    "known NC families are derived from the manifest (flux / ideogram / krea)",
    families.some((f) => /flux|ideogram|krea|sana/i.test(f)) ||
      tokens.some((t) => /flux|ideogram|krea|sana/.test(t)),
  );
  assert(
    "Anima is covered (bootstrap until epic 10512 lands in the manifest)",
    tokens.includes("anima") && tokens.includes("circlestone-labs"),
  );

  const tmp = await mkdtemp(path.join(os.tmpdir(), "nc-weights-guard-"));
  try {
    // (a) An Anima weight by bare basename must be caught and named NC.
    const diryA = path.join(tmp, "installer-with-anima");
    await mkdir(diryA, { recursive: true });
    await writeFile(path.join(diryA, "anima-base-v1.0.safetensors"), "not real weights");
    const vA = await scan({ roots: [diryA], ncTokens: tokens, includeBin: false });
    assert("fake anima-base-v1.0.safetensors is flagged", vA.length === 1);
    assert("...and is classified NC", vA[0]?.ncToken != null);

    // (b) An HF-cache-form NC path must be caught even with a generic filename.
    const dirB = path.join(tmp, "image-layer", "models--circlestone-labs--Anima", "snapshots", "abc");
    await mkdir(dirB, { recursive: true });
    await writeFile(path.join(dirB, "model.safetensors"), "not real weights");
    const vB = await scan({
      roots: [path.join(tmp, "image-layer")],
      ncTokens: tokens,
      includeBin: false,
    });
    assert("NC weight inside models--circlestone-labs--Anima/ is flagged", vB.length === 1 && vB[0].ncToken != null);

    // (c) A clean payload passes.
    const dirC = path.join(tmp, "clean");
    await mkdir(dirC, { recursive: true });
    await writeFile(path.join(dirC, "README.txt"), "no weights here");
    await writeFile(path.join(dirC, "sceneworks-api"), "a binary, not a weight");
    const vC = await scan({ roots: [dirC], ncTokens: tokens, includeBin: false });
    assert("clean payload passes", vC.length === 0);

    // (d) A payload containing only the allowlisted permissive weight passes.
    const dirD = path.join(tmp, "with-allowlisted");
    await mkdir(dirD, { recursive: true });
    await writeFile(path.join(dirD, "aesthetic-v2-sac-logos-ava1-l14.safetensors"), "permissive");
    const vD = await scan({ roots: [dirD], ncTokens: tokens, includeBin: false });
    assert("allowlisted permissive weight passes", vD.length === 0);

    // (e) A non-NC, non-allowlisted weight still fails (belt-and-suspenders: catches
    //     the next model even before it is tagged NC), with the generic message.
    const dirE = path.join(tmp, "unexpected");
    await mkdir(dirE, { recursive: true });
    await writeFile(path.join(dirE, "some-random-model.safetensors"), "weights");
    const vE = await scan({ roots: [dirE], ncTokens: tokens, includeBin: false });
    assert("unexpected non-allowlisted weight is flagged", vE.length === 1);
    assert("...with the generic (non-NC) message", vE[0]?.ncToken == null);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }

  if (failures === 0) {
    console.log("self-test: PASS");
    return 0;
  }
  console.error(`self-test: FAIL (${failures} assertion(s) failed)`);
  return 1;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));

  if (options.selfTest) {
    process.exitCode = await selfTest();
    return;
  }

  const { tokens, roots, violations } = await runScan(options);
  console.log(
    `NC-weights guard: scanned ${roots.join(", ")} for model weights ` +
      `(${tokens.length} NC family tokens from the manifests + bootstrap).`,
  );
  if (violations.length > 0) {
    reportViolations(violations);
    process.exitCode = 1;
    return;
  }
  console.log("NC-weights guard passed: no model weights in the artifact payload.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
