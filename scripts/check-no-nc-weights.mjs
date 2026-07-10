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
//   Symlinked directories are followed (loop-safe), so a link to a weights dir is
//   not missed.
//
//   In addition to the file walk, EVERY run inspects the committed Tauri bundle
//   config (apps/desktop/tauri*.conf.json): a declared `bundle.resources` /
//   `externalBin` spec that is an archive, stages a weight, matches an NC token, or
//   is rooted at a weights dir fails the build. That is the cheap defense for the one
//   vector the file walk cannot see into — a weight sealed inside an archive staged
//   as a resource. (Decompressing archives to scan their CONTENTS is a tracked
//   follow-up, not done here; see docs "Known limitation".)
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

import {
  readFile,
  readdir,
  stat,
  realpath,
  mkdtemp,
  mkdir,
  writeFile,
  symlink,
  rm,
} from "node:fs/promises";
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

// File extensions that are, in practice, always serialized model weights. A bare
// `.bin` is intentionally NOT in this set — fonts, wasm, and misc blobs also use
// `.bin`, so treating every `.bin` as a weight by default would false-trip. But a
// `.bin` is NOT a blind spot: three things catch an NC `.bin` with no extra flags —
//   (1) scan() runs the STRONG NC repo-path match (models--org--name / org/name)
//       on every file regardless of extension, so any blob inside a redistributed
//       NC repo/cache directory fails (see scan());
//   (2) a `.bin` whose path matches a bare NC family token (e.g. `anima-base.bin`,
//       or a `.bin` under `models--…--Anima/`) is treated as a weight; and
//   (3) the canonical Hugging Face `.bin` weight filenames below — one of the two
//       dominant HF weight formats (text encoders, VAEs, full checkpoints) — are
//       treated as weights by name even when they carry no NC token.
// `--include-bin` promotes *every* `.bin` to a weight (belt-and-suspenders).
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

// Canonical Hugging Face `.bin` weight filenames. Unlike a generic `foo.bin`, these
// names are, by HF convention, ALWAYS serialized model weights (PyTorch pickle
// checkpoints), so they count as weights by default even though `.bin` as a bare
// extension does not. `pytorch_model.bin` / `diffusion_pytorch_model.bin` /
// `open_clip_pytorch_model.bin` are the second dominant HF weight format after
// safetensors. The optional `-00001-of-00002` group covers sharded checkpoints.
const CANONICAL_BIN_WEIGHT =
  /^(?:pytorch_model|diffusion_pytorch_model|open_clip_pytorch_model)(?:-\d{5}-of-\d{5})?\.bin$/i;

// Archive/container extensions. A weight sealed inside one of these would slip both
// the loose-tree file scan AND the source-tree scan, so we refuse to let one be
// STAGED as a Tauri bundle resource in the first place (see inspectResourceSpecs).
const ARCHIVE_EXTENSIONS = new Set([
  ".zip",
  ".tar",
  ".tgz",
  ".gz",
  ".bz2",
  ".xz",
  ".zst",
  ".7z",
  ".rar",
  ".dmg",
  ".msi",
  ".cab",
  ".pkg",
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
// the source of truth). This is not left to vigilance: the self-test asserts that
// once the manifest itself yields an `anima` / `circlestone-labs` token, this array
// MUST be empty — so the "temporary" seed cannot silently become permanent.
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

// Build the NC token set from the manifests ALONE (no bootstrap seed). Kept
// separate so the self-test can assert that the bootstrap seed is redundant the
// moment the manifest starts producing the same family tokens.
async function buildManifestNcTokens() {
  const tokens = new Set();
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

// Build the NC token set from the manifests + the bootstrap seed (what the guard
// actually runs against).
async function buildNcTokens() {
  const { tokens, families } = await buildManifestNcTokens();
  const merged = new Set(tokens);
  for (const token of BOOTSTRAP_NC_TOKENS) {
    merged.add(token.toLowerCase());
  }
  return { tokens: [...merged].filter(Boolean), families };
}

// A STRONG token is a multi-segment HF repo/cache form (`org/name` or
// `models--org--name`). Its presence in a path means a genuine redistributed NC
// repo directory is on disk, so any file under it fails regardless of extension.
// These never collide with ordinary source paths. A WEAK token is a bare model
// name (`anima`, `ideogram-4`) — it legitimately appears in docs/config that only
// *reference* a model (e.g. apps/web/public/prompt-guides/ideogram-4.md), so a weak
// match alone must not fail a non-weight file.
function isStrongToken(token) {
  return token.includes("/") || token.includes("--");
}

// Boundary-aware containment. The token must sit on non-alphanumeric boundaries
// (path separators, `-`, `_`, `.`, or the string ends) so the 5-char token "anima"
// matches ".../models--circlestone-labs--anima/..." and "anima-base.safetensors"
// but NOT ".../animations/foo.js" (the trailing "t" is alphanumeric → different
// word). Without this, a short token substring-matches unrelated paths.
function tokenMatchesPath(needle, token) {
  let from = 0;
  for (;;) {
    const index = needle.indexOf(token, from);
    if (index === -1) {
      return false;
    }
    const before = index === 0 ? "" : needle[index - 1];
    const after = index + token.length >= needle.length ? "" : needle[index + token.length];
    const boundaryBefore = before === "" || !/[a-z0-9]/.test(before);
    const boundaryAfter = after === "" || !/[a-z0-9]/.test(after);
    if (boundaryBefore && boundaryAfter) {
      return true;
    }
    from = index + 1;
  }
}

// The first STRONG NC token (if any) whose repo/cache path form appears in the file
// path. Compared over the whole POSIX-normalized path so an ancestor cache dir
// (models--circlestone-labs--Anima/…) is caught even for a generically-named blob.
function strongNcTokenFor(needle, ncTokens) {
  return (
    ncTokens.find(
      (token) => token.length >= 4 && isStrongToken(token) && tokenMatchesPath(needle, token),
    ) ?? null
  );
}

// The first WEAK NC token (bare family/model name) whose boundary-aware form appears
// in the file path — used only to classify a file that is ALSO a weight/blob.
function weakNcTokenFor(needle, ncTokens) {
  return (
    ncTokens.find(
      (token) => token.length >= 4 && !isStrongToken(token) && tokenMatchesPath(needle, token),
    ) ?? null
  );
}

async function walk(dir, files, visited = new Set()) {
  // Loop guard: track REAL directory paths so a symlink cycle (A→B→A) or a link
  // pointing back up the tree cannot spin forever. Keyed on realpath, not the
  // logical path, so two routes to the same dir are visited once.
  let realDir;
  try {
    realDir = await realpath(dir);
  } catch (error) {
    if (error.code === "ENOENT") {
      return; // Broken link or vanished dir.
    }
    throw error;
  }
  if (visited.has(realDir)) {
    return;
  }
  visited.add(realDir);

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
      await walk(full, files, visited);
    } else if (entry.isSymbolicLink()) {
      // A symlink's Dirent reflects the LINK, not its target, so a symlink to a
      // directory of weights would otherwise be recorded as a single "file" and
      // only extension-checked — never descended. Resolve the target: descend a
      // linked dir (loop-safe via `visited`), record a linked file, skip a broken
      // link.
      let target;
      try {
        target = await stat(full); // Follows the link.
      } catch (error) {
        if (error.code === "ENOENT") {
          continue; // Dangling symlink.
        }
        throw error;
      }
      if (target.isDirectory()) {
        if (EXCLUDED_DIR_NAMES.has(entry.name)) {
          continue;
        }
        await walk(full, files, visited);
      } else if (target.isFile()) {
        files.push(full);
      }
    } else if (entry.isFile()) {
      files.push(full);
    }
  }
}

// Is this a model weight? A definite weight is any WEIGHT_EXTENSIONS file, a
// canonical HF `.bin` (pytorch_model.bin & friends), a `.bin` that matches a bare
// NC family token (so an NC blob named `anima-base.bin` counts), or — under
// --include-bin — any `.bin` at all.
function isWeightFile(filePath, { includeBin, weakNcMatch }) {
  const ext = path.extname(filePath).toLowerCase();
  if (WEIGHT_EXTENSIONS.has(ext)) {
    return true;
  }
  if (ext === ".bin") {
    if (includeBin || CANONICAL_BIN_WEIGHT.test(path.basename(filePath)) || weakNcMatch) {
      return true;
    }
  }
  return false;
}

// Core: scan the given roots and return the list of violations.
//
// Two independent ways a file becomes a violation:
//   (1) STRONG NC match — its path contains a genuine NC repo/cache directory
//       (models--org--name / org/name). Fails regardless of extension: any blob of
//       any name inside a redistributed NC repo must not ship. This is what closes
//       the `.bin` blind spot for the HF-cache-form vector with no flags.
//   (2) It is a model weight (see isWeightFile) that is not on the allowlist. NC
//       classification (a bare family token) enriches the message; a bare NC token
//       also promotes an otherwise-ambiguous `.bin` to a weight.
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
      const basename = path.basename(file);
      const needle = file.split(path.sep).join("/").toLowerCase();

      // (1) A file inside a genuine NC repo/cache directory — fail regardless of
      // extension or allowlist. An allowlisted permissive name must never travel
      // inside an NC repo path either.
      const strongToken = strongNcTokenFor(needle, ncTokens);
      if (strongToken) {
        violations.push({ file: path.relative(root, file) || file, basename, ncToken: strongToken });
        continue;
      }

      // (2) Otherwise, is it a weight? A bare NC token both classifies it and
      // promotes an ambiguous `.bin` to a weight.
      const weakToken = weakNcTokenFor(needle, ncTokens);
      if (!isWeightFile(file, { includeBin, weakNcMatch: Boolean(weakToken) })) {
        continue;
      }
      // A non-NC, allowlisted permissive weight is legitimately bundled.
      if (!weakToken && ALLOWLIST_BASENAMES.has(basename)) {
        continue;
      }
      violations.push({
        file: path.relative(root, file) || file,
        basename,
        ncToken: weakToken,
      });
    }
  }
  return violations;
}

// Inspect declared Tauri bundle-resource specs (the SOURCE globs the packager copies
// into Contents/Resources) and refuse the realistic evasion the loose-tree scan
// cannot see: a weight sealed in an archive, a weight staged directly, or a resource
// glob rooted at an NC repo / a weights directory. This is a config-level check —
// it does not decompress anything (see docs "Known limitation").
function specHasExtIn(spec, extensions) {
  // Match an extension as a suffix of a path component, tolerant of trailing globs
  // (`weights/*.safetensors`, `blob.tar.gz`, `models/**/*.bin`).
  const lower = spec.toLowerCase();
  for (const ext of extensions) {
    const escaped = ext.replace(/[.]/g, "\\.");
    if (new RegExp(`${escaped}($|[/*?\\s])`).test(lower)) {
      return true;
    }
  }
  return false;
}

function inspectResourceSpecs(specs, ncTokens) {
  const violations = [];
  for (const raw of specs) {
    if (typeof raw !== "string" || raw.length === 0) {
      continue;
    }
    const needle = raw.split(path.sep).join("/").toLowerCase();
    const strongToken = strongNcTokenFor(needle, ncTokens);
    const weakToken = weakNcTokenFor(needle, ncTokens);
    if (strongToken || weakToken) {
      violations.push({
        spec: raw,
        reason: `resource path matches Non-Commercial family token "${strongToken || weakToken}"`,
      });
      continue;
    }
    if (specHasExtIn(needle, ARCHIVE_EXTENSIONS)) {
      violations.push({
        spec: raw,
        reason:
          "resource is an archive/container — a weight sealed inside it would evade " +
          "both the loose-tree and source-tree scans; unpack it at runtime instead",
      });
      continue;
    }
    if (specHasExtIn(needle, WEIGHT_EXTENSIONS) || /\.bin($|[/*?\s])/.test(needle)) {
      violations.push({ spec: raw, reason: "resource stages a model-weight file into the bundle" });
      continue;
    }
    if (/(^|\/)(weights|checkpoints|loras?|safetensors)(\/|$)/.test(needle)) {
      violations.push({ spec: raw, reason: "resource glob is rooted at a weights directory" });
      continue;
    }
  }
  return violations;
}

// Read the declared bundle resources / externalBin from the committed Tauri configs
// and run them through inspectResourceSpecs. Missing config files are fine (a
// non-desktop context). Returns { specs, violations }.
async function inspectTauriConfigs(ncTokens) {
  const configRelPaths = [
    "apps/desktop/tauri.conf.json",
    "apps/desktop/tauri.macos.conf.json",
    "apps/desktop/updater.release.conf.json",
  ];
  const specs = [];
  for (const relPath of configRelPaths) {
    let body;
    try {
      body = await readFile(path.join(root, relPath), "utf8");
    } catch (error) {
      if (error.code === "ENOENT") {
        continue;
      }
      throw error;
    }
    const conf = JSON.parse(body);
    const bundle = conf?.bundle ?? {};
    const resources = bundle.resources;
    if (Array.isArray(resources)) {
      specs.push(...resources);
    } else if (resources && typeof resources === "object") {
      // Object form maps SOURCE path -> destination; the source is the key.
      specs.push(...Object.keys(resources));
    }
    if (Array.isArray(bundle.externalBin)) {
      specs.push(...bundle.externalBin);
    }
  }
  return { specs, violations: inspectResourceSpecs(specs, ncTokens) };
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

function reportResourceViolations(resourceViolations) {
  console.error(
    "\nNC-WEIGHTS GUARD FAILED (sc-10526): a declared Tauri bundle resource could " +
      "carry model weights into the installer.\n" +
      "Bundle resources are copied verbatim into the app payload; an archive or a " +
      "weights directory staged here would ship weights even though the loose-tree " +
      "scan cannot see inside it. Pull weights at runtime instead of bundling them.\n",
  );
  for (const violation of resourceViolations) {
    console.error(`  ✗ bundle resource "${violation.spec}"\n      ${violation.reason}.`);
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
  // The Tauri resource-config check reads the committed configs at the repo root and
  // is independent of the file scan, so it runs on every invocation (default and
  // --dir alike) as a cheap second line of defense against a bundled archive.
  const { specs, violations: resourceViolations } = await inspectTauriConfigs(tokens);
  return { tokens, families, roots, violations, resourceViolations, resourceSpecCount: specs.length };
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

  // Bootstrap-seed redundancy: the moment the MANIFEST itself yields an Anima /
  // CircleStone token, the hardcoded BOOTSTRAP_NC_TOKENS seed is redundant and MUST
  // be emptied — otherwise the "temporary" seed silently becomes permanent. This
  // assertion fails loudly (forcing the seed's removal) once Anima lands in the
  // manifest with NC classification.
  const manifestOnly = await buildManifestNcTokens();
  const manifestHasAnima = manifestOnly.tokens.some((t) => /anima|circlestone/.test(t));
  assert(
    "bootstrap seed is empty once the manifest covers Anima (no permanent hardcode)",
    !manifestHasAnima || BOOTSTRAP_NC_TOKENS.length === 0,
  );

  // Token tightening: a bare 5-char token like "anima" must NOT substring-match an
  // unrelated path such as ".../animations/" (this is what the shipped
  // apps/web/public/prompt-guides/ideogram-4.md relies on: a doc that merely names
  // an NC model is not a weight and must not fail the scan).
  assert(
    'weak token "anima" does not match ".../animations/foo.js"',
    weakNcTokenFor("apps/web/src/animations/foo.js", tokens) == null &&
      strongNcTokenFor("apps/web/src/animations/foo.js", tokens) == null,
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

    // (f) THE `.bin` BLIND SPOT (the major review finding). A `.bin` inside an
    //     HF-cache-form NC directory MUST fail by default — no --include-bin.
    const dirF = path.join(tmp, "layer-bin", "models--circlestone-labs--Anima", "snapshots", "abc");
    await mkdir(dirF, { recursive: true });
    await writeFile(path.join(dirF, "pytorch_model.bin"), "not real weights");
    await writeFile(path.join(dirF, "anima-base.bin"), "not real weights");
    const vF = await scan({
      roots: [path.join(tmp, "layer-bin")],
      ncTokens: tokens,
      includeBin: false,
    });
    assert(
      "NC .bin under models--circlestone-labs--Anima/ fails by default (no --include-bin)",
      vF.length === 2 && vF.every((v) => v.ncToken != null),
    );

    // (g) A bare NC-named `.bin` (not under an NC dir) is still caught by default.
    const dirG = path.join(tmp, "loose-bin");
    await mkdir(dirG, { recursive: true });
    await writeFile(path.join(dirG, "anima-base.bin"), "not real weights");
    const vG = await scan({ roots: [dirG], ncTokens: tokens, includeBin: false });
    assert("bare NC-named anima-base.bin is flagged NC by default", vG.length === 1 && vG[0].ncToken != null);

    // (h) A canonical HF `.bin` weight name is caught even with no NC token.
    const dirH = path.join(tmp, "hf-bin");
    await mkdir(dirH, { recursive: true });
    await writeFile(path.join(dirH, "diffusion_pytorch_model.bin"), "weights");
    const vH = await scan({ roots: [dirH], ncTokens: tokens, includeBin: false });
    assert("canonical HF diffusion_pytorch_model.bin is flagged (generic)", vH.length === 1 && vH[0].ncToken == null);

    // (i) A generic, non-canonical, non-NC `.bin` (font/wasm/etc.) must NOT trip the
    //     guard by default — no false positives.
    const dirI = path.join(tmp, "generic-bin");
    await mkdir(dirI, { recursive: true });
    await writeFile(path.join(dirI, "glyphs.bin"), "a font blob, not a weight");
    const vI = await scan({ roots: [dirI], ncTokens: tokens, includeBin: false });
    assert("generic non-weight glyphs.bin passes by default", vI.length === 0);
    const vIbin = await scan({ roots: [dirI], ncTokens: tokens, includeBin: true });
    assert("...but --include-bin promotes it to a weight", vIbin.length === 1);

    // (j) SYMLINKED DIRECTORY (the walk() finding). A symlink to a directory of
    //     weights must be descended, not treated as one opaque "file".
    const realWeights = path.join(tmp, "real-weights-dir");
    await mkdir(realWeights, { recursive: true });
    await writeFile(path.join(realWeights, "some-random-model.safetensors"), "weights");
    const dirJ = path.join(tmp, "bundle-with-symlink");
    await mkdir(dirJ, { recursive: true });
    await symlink(realWeights, path.join(dirJ, "linked-weights"), "dir");
    const vJ = await scan({ roots: [dirJ], ncTokens: tokens, includeBin: false });
    assert("weight inside a SYMLINKED directory is flagged (walk follows dir links)", vJ.length === 1);

    // (k) A symlink LOOP must terminate (visited-set), not hang or throw.
    const loopDir = path.join(tmp, "loop");
    await mkdir(loopDir, { recursive: true });
    await symlink(loopDir, path.join(loopDir, "self"), "dir");
    const vK = await scan({ roots: [loopDir], ncTokens: tokens, includeBin: false });
    assert("symlink loop terminates cleanly", vK.length === 0);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }

  // (l) Tauri bundle-resource inspection. The committed configs must be clean, and a
  //     crafted spec that stages an archive / a weights dir / an NC repo must fail.
  const { violations: liveResourceViolations } = await inspectTauriConfigs(tokens);
  assert(
    "committed Tauri bundle resources declare no weight-bearing entry",
    liveResourceViolations.length === 0,
  );
  const cleanSpecs = ["onnxruntime/**/*", "mlx/**/*"]; // current, legitimately clean
  const dirtySpecs = [
    "weights.tar.gz", // archive container
    "models/**/*.safetensors", // staged weight glob
    "checkpoints/**/*", // weights directory
    "vendor/pytorch_model.bin", // staged weight
    "models--circlestone-labs--Anima/**/*", // NC repo directory
  ];
  const craftedResources = inspectResourceSpecs([...cleanSpecs, ...dirtySpecs], tokens);
  const flaggedSpecs = new Set(craftedResources.map((v) => v.spec));
  assert(
    "every archive / weight / NC-repo resource spec is flagged",
    dirtySpecs.every((s) => flaggedSpecs.has(s)),
  );
  assert(
    "clean resource specs (onnxruntime, mlx) are not flagged",
    cleanSpecs.every((s) => !flaggedSpecs.has(s)),
  );

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

  const { tokens, roots, violations, resourceViolations, resourceSpecCount } =
    await runScan(options);
  console.log(
    `NC-weights guard: scanned ${roots.join(", ")} for model weights ` +
      `(${tokens.length} NC family tokens from the manifests + bootstrap) and ` +
      `${resourceSpecCount} declared Tauri bundle-resource spec(s).`,
  );
  let failed = false;
  if (violations.length > 0) {
    reportViolations(violations);
    failed = true;
  }
  if (resourceViolations.length > 0) {
    reportResourceViolations(resourceViolations);
    failed = true;
  }
  if (failed) {
    process.exitCode = 1;
    return;
  }
  console.log(
    "NC-weights guard passed: no model weights in the artifact payload and no " +
      "weight-bearing bundle resources declared.",
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
