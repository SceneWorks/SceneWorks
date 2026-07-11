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
//   as a resource.
//
//   --scan-archives (opt-in, sc-10551): defense-in-depth for a weight that reached a
//   built artifact sealed INSIDE an archive some other way (not via a declared Tauri
//   resource). For every archive found under the scan roots it DECOMPRESSES the
//   container and runs the exact same weight/NC-token checks over the entry paths
//   inside — recursively, so a nested archive-in-archive is caught too. Pure Node
//   (no external deps): `.zip`/`.nsis.zip` (central-directory listing, deflate/store),
//   `.tar`, `.tar.gz`/`.tgz`, and bare `.gz`. Real release payloads it decompresses:
//   the macOS `bundle/macos/*.app.tar.gz` updater tarball and the Windows
//   `*.nsis.zip` updater. Opaque installers (`.dmg`/`.msi`/`-setup.exe`) have no
//   pure-JS reader; their CONTENTS duplicate the already-walked `.app`/resource tree,
//   so `--skip-uninspectable` downgrades them from a hard failure to a warning on the
//   release lanes. ARCHIVE-BOMB GUARDS (required): the scan fails CLOSED — never
//   OOMs — if an archive exceeds the on-disk size cap, the cumulative/per-entry
//   uncompressed-size caps (zip is refused on its DECLARED sizes before any inflate),
//   the entry-count cap, or the nesting-depth cap. See DEFAULT_ARCHIVE_LIMITS and
//   scanArchivesUnderRoots().
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
import zlib from "node:zlib";

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
// These are exactly the tokens `repoToTokens("circlestone-labs/anima")` will emit
// once sc-10523 wires the manifest entry — the two STRONG repo/cache forms
// (`circlestone-labs/anima`, `models--circlestone-labs--anima`) plus the WEAK bare
// name (`anima`). Seeding the strong forms is deliberate: it makes the header
// comment's "any blob regardless of extension" guarantee (see scan() case (1)) TRUE
// for Anima today, so an arbitrary-extension blob (e.g. `model.dat`) under a
// redistributed `models--circlestone-labs--Anima/` directory fails right now, not
// only after the manifest lands. Because the seed mirrors the manifest output
// exactly, the guard behaves identically before and after sc-10523.
// REMOVE every entry once the corresponding family is in the manifest (the manifest
// is the source of truth). This is not left to vigilance: the self-test asserts that
// once the manifest itself yields an `anima` / `circlestone` token, this array MUST
// be empty — so the "temporary" seed cannot silently become permanent.
// EMPTIED by sc-10523 (epic 10512): Anima now lands in config/manifests/builtin.models.jsonc as three
// `nonCommercial: true` entries (family `anima`, repo `circlestone-labs/Anima`), so
// `buildManifestNcTokens()` derives the exact tokens this seed used to carry
// (`circlestone-labs/anima`, `models--circlestone-labs--anima`, `anima`) straight from the manifest —
// the manifest is now the single source of truth. The self-test tripwire below asserts this array MUST
// be empty once the manifest yields an Anima/CircleStone token; keeping a seed here would silently
// re-hardcode the family and defeat the manifest-derivation design. Re-seed here ONLY for a NEW NC
// family that is being ported before its manifest entry exists, and remove it the moment that lands.
const BOOTSTRAP_NC_TOKENS = [];

// Archive-bomb guard limits for --scan-archives (sc-10551). A malicious or accidental
// "zip bomb" declares (or inflates to) an astronomically large payload; without caps a
// naive decompress-and-scan would exhaust memory/disk. Every one of these is enforced
// as a fail-CLOSED refusal (the archive is reported as a build failure, not silently
// skipped, and never fully inflated past the cap). All are overridable from the CLI
// (--max-archive-bytes / --max-uncompressed / --max-entry-bytes / --max-entries /
// --max-depth) so a lane whose artifact is legitimately larger can raise them without a
// code change. Chosen generously relative to real SceneWorks updater payloads (a macOS
// `.app.tar.gz` is a few hundred MB compressed) while still bounding a runaway.
const DEFAULT_ARCHIVE_LIMITS = {
  // Refuse to even readFile an archive whose ON-DISK size exceeds this (a huge file is
  // itself the first bomb signal, and bounds the Buffer we load).
  maxArchiveFileBytes: 2 * 1024 * 1024 * 1024, // 2 GiB
  // Cumulative UNCOMPRESSED bytes across an entire archive tree (all nesting levels of
  // one top-level archive). For gzip this is the gunzip `maxOutputLength` (Node throws
  // before inflating past it); for zip it is checked against the DECLARED sizes in the
  // central directory BEFORE any entry is inflated. Kept < Node's Buffer max (~4 GiB).
  maxTotalUncompressedBytes: 3 * 1024 * 1024 * 1024, // 3 GiB
  // Per-entry uncompressed cap (zip: declared size; deflate inflate `maxOutputLength`).
  maxEntryUncompressedBytes: 1 * 1024 * 1024 * 1024, // 1 GiB
  // Max number of entries across the archive tree (defeats a "many tiny files" bomb).
  maxEntries: 200_000,
  // Max archive-in-archive nesting depth (defeats a quine / deeply-nested bomb and
  // bounds recursion). Top-level archive = depth 1.
  maxDepth: 8,
};

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

// THE single NC/weight classification for one path — the shared signature logic used
// by BOTH the loose-tree file walk (scan) and the archive-content scan (sc-10551), so
// the two guards can never drift. `needle` is the POSIX-normalized, lowercased path
// (a real filesystem path for the walk, or `archive!/entry` for an archive entry) and
// `basename` its last segment. Returns a violation descriptor `{ ncToken }` or null:
//   * STRONG NC repo/cache token in the path        -> violation (any extension)
//   * weight file (isWeightFile) not on the allowlist -> violation (ncToken = the weak
//     family token if one matched, else null for the generic "unexpected weight")
// Order and semantics mirror the original inline scan() logic exactly.
function classifyEntry({ needle, basename, includeBin, ncTokens }) {
  const strongToken = strongNcTokenFor(needle, ncTokens);
  if (strongToken) {
    return { ncToken: strongToken };
  }
  const weakToken = weakNcTokenFor(needle, ncTokens);
  if (!isWeightFile(basename, { includeBin, weakNcMatch: Boolean(weakToken) })) {
    return null;
  }
  // A non-NC, allowlisted permissive weight is legitimately bundled.
  if (!weakToken && ALLOWLIST_BASENAMES.has(basename)) {
    return null;
  }
  return { ncToken: weakToken };
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
      const hit = classifyEntry({ needle, basename, includeBin, ncTokens });
      if (hit) {
        violations.push({ file: path.relative(root, file) || file, basename, ncToken: hit.ncToken });
      }
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

// ===========================================================================
// --scan-archives (sc-10551): decompress-and-scan archive CONTENTS
// ===========================================================================
//
// Defense-in-depth over the file-name walk and the Tauri resource-config check: open
// every archive under the scan roots and run the SAME classifyEntry() over the entry
// paths inside — recursively, with hard archive-bomb guards. Pure Node built-ins only
// (zlib for gzip/deflate; hand-rolled tar + zip-central-directory readers), so there is
// no new dependency and the whole thing is exercised hermetically by --self-test.

// A bomb guard fired (or a container is corrupt): the archive is REFUSED — reported as
// a build failure, never silently skipped, and never inflated past the cap.
class ArchiveBombError extends Error {
  constructor(reason) {
    super(reason);
    this.name = "ArchiveBombError";
    this.reason = reason;
  }
}

// No pure-JS reader for this container (e.g. `.dmg`, `.msi`, an unsupported zip
// compression method). By default this fails the build too (fail closed); the release
// lanes pass --skip-uninspectable to downgrade it to a warning, because the opaque
// installer's CONTENTS duplicate the already-walked `.app`/resource tree.
class UninspectableArchiveError extends Error {
  constructor(reason) {
    super(reason);
    this.name = "UninspectableArchiveError";
    this.reason = reason;
  }
}

// -- tar (ustar/GNU/PAX) reader over an already-decompressed buffer -----------------
function readTarString(buf, start, len) {
  let end = start;
  const limit = Math.min(start + len, buf.length);
  while (end < limit && buf[end] !== 0) {
    end += 1;
  }
  return buf.toString("utf8", start, end);
}

function readTarOctal(buf, start, len) {
  const text = readTarString(buf, start, len).trim();
  if (text === "") {
    return 0;
  }
  const value = parseInt(text, 8);
  return Number.isFinite(value) ? value : 0;
}

// Return the regular-file entries of a tar buffer as { name, size, dataStart },
// enforcing the entry-count / per-entry / cumulative-size caps as it walks. `budget`
// is shared across the whole archive tree (all nesting levels of one top-level
// archive) so a bomb spread across nested archives is still bounded.
function parseTarEntries(buf, limits, budget) {
  const entries = [];
  let offset = 0;
  let pendingName = null; // from a GNU 'L' long-name or PAX 'path=' record
  while (offset + 512 <= buf.length) {
    const block = buf.subarray(offset, offset + 512);
    if (block.every((byte) => byte === 0)) {
      break; // end-of-archive zero block
    }
    let name = readTarString(block, 0, 100);
    const prefix = readTarString(block, 345, 155);
    if (prefix) {
      name = `${prefix}/${name}`;
    }
    const size = readTarOctal(block, 124, 12);
    const typeByte = block[156];
    const typeflag = String.fromCharCode(typeByte || 0);
    const dataStart = offset + 512;
    const dataBlocks = Math.ceil(size / 512) * 512;

    // GNU long name: the data payload is the real name of the NEXT entry.
    if (typeflag === "L") {
      pendingName = readTarString(buf, dataStart, size).replace(/\0+$/, "");
      offset = dataStart + dataBlocks;
      continue;
    }
    // GNU long link name: not a file we scan; skip its payload.
    if (typeflag === "K") {
      offset = dataStart + dataBlocks;
      continue;
    }
    // PAX extended header: pull `path=` if present, applies to the next entry.
    if (typeflag === "x" || typeflag === "g") {
      const records = buf.toString("utf8", dataStart, dataStart + size);
      const match = records.match(/(?:^|\n)\d+ path=([^\n]*)\n/);
      if (match) {
        pendingName = match[1];
      }
      offset = dataStart + dataBlocks;
      continue;
    }

    const effectiveName = pendingName ?? name;
    pendingName = null;

    budget.entries += 1;
    if (budget.entries > limits.maxEntries) {
      throw new ArchiveBombError(`entry count exceeds cap ${limits.maxEntries}`);
    }
    if (size > limits.maxEntryUncompressedBytes) {
      throw new ArchiveBombError(
        `tar entry "${effectiveName}" is ${size} B (> per-entry cap ${limits.maxEntryUncompressedBytes} B)`,
      );
    }
    budget.totalBytes += size;
    if (budget.totalBytes > limits.maxTotalUncompressedBytes) {
      throw new ArchiveBombError(
        `cumulative uncompressed size exceeds cap ${limits.maxTotalUncompressedBytes} B`,
      );
    }

    // Regular file (ustar '0', legacy '\0', contiguous '7'); dirs/links are not scanned.
    if (typeByte === 0x30 || typeByte === 0 || typeByte === 0x37) {
      entries.push({ name: effectiveName, size, dataStart });
    }
    offset = dataStart + dataBlocks;
  }
  return entries;
}

// -- zip reader (central-directory listing; per-entry inflate only when recursing) ---
function findZipEocd(buf) {
  const signature = 0x06054b50;
  const minLen = 22;
  const maxBack = Math.min(buf.length, minLen + 0xffff); // + max comment length
  for (let i = buf.length - minLen; i >= buf.length - maxBack && i >= 0; i -= 1) {
    if (buf.readUInt32LE(i) === signature) {
      return i;
    }
  }
  return -1;
}

// Resolve the ZIP64 extra field for any of the three fields that carry the 0xFFFFFFFF
// "see ZIP64" sentinel. The 8-byte values appear in a fixed order (uncompressed,
// compressed, local-header-offset), each present only if its 32-bit field was sentinel.
function readZip64Extra(buf, start, extraLen, current) {
  const out = { ...current };
  let q = start;
  const end = start + extraLen;
  while (q + 4 <= end) {
    const id = buf.readUInt16LE(q);
    const size = buf.readUInt16LE(q + 2);
    const body = q + 4;
    if (id === 0x0001) {
      let r = body;
      if (current.uncompSize === 0xffffffff) {
        out.uncompSize = Number(buf.readBigUInt64LE(r));
        r += 8;
      }
      if (current.compSize === 0xffffffff) {
        out.compSize = Number(buf.readBigUInt64LE(r));
        r += 8;
      }
      if (current.localOffset === 0xffffffff) {
        out.localOffset = Number(buf.readBigUInt64LE(r));
        r += 8;
      }
      return out;
    }
    q = body + size;
  }
  return out;
}

// List zip entries from the central directory (names + declared sizes), enforcing the
// bomb caps on the DECLARED uncompressed sizes BEFORE anything is inflated — the key
// zip-bomb defense (a tiny archive that declares a huge output is refused up front).
function parseZipEntries(buf, limits, budget) {
  const eocd = findZipEocd(buf);
  if (eocd === -1) {
    throw new ArchiveBombError("zip: End-Of-Central-Directory record not found (truncated or not a zip)");
  }
  let totalEntries = buf.readUInt16LE(eocd + 10);
  let cdOffset = buf.readUInt32LE(eocd + 16);
  if (totalEntries === 0xffff || cdOffset === 0xffffffff) {
    const locatorOffset = eocd - 20;
    if (locatorOffset < 0 || buf.readUInt32LE(locatorOffset) !== 0x07064b50) {
      throw new UninspectableArchiveError("zip: ZIP64 end-locator missing; cannot verify a large zip");
    }
    const z64 = Number(buf.readBigUInt64LE(locatorOffset + 8));
    if (z64 < 0 || z64 + 56 > buf.length || buf.readUInt32LE(z64) !== 0x06064b50) {
      throw new UninspectableArchiveError("zip: ZIP64 EOCD record missing/out of range");
    }
    totalEntries = Number(buf.readBigUInt64LE(z64 + 32));
    cdOffset = Number(buf.readBigUInt64LE(z64 + 48));
  }

  const entries = [];
  let p = cdOffset;
  for (let i = 0; i < totalEntries; i += 1) {
    if (p + 46 > buf.length || buf.readUInt32LE(p) !== 0x02014b50) {
      throw new ArchiveBombError("zip: malformed central-directory header");
    }
    const method = buf.readUInt16LE(p + 10);
    let compSize = buf.readUInt32LE(p + 20);
    let uncompSize = buf.readUInt32LE(p + 24);
    const nameLen = buf.readUInt16LE(p + 28);
    const extraLen = buf.readUInt16LE(p + 30);
    const commentLen = buf.readUInt16LE(p + 32);
    let localOffset = buf.readUInt32LE(p + 42);
    const name = buf.toString("utf8", p + 46, p + 46 + nameLen);
    if (compSize === 0xffffffff || uncompSize === 0xffffffff || localOffset === 0xffffffff) {
      ({ compSize, uncompSize, localOffset } = readZip64Extra(buf, p + 46 + nameLen, extraLen, {
        compSize,
        uncompSize,
        localOffset,
      }));
    }

    budget.entries += 1;
    if (budget.entries > limits.maxEntries) {
      throw new ArchiveBombError(`entry count exceeds cap ${limits.maxEntries}`);
    }
    if (uncompSize > limits.maxEntryUncompressedBytes) {
      throw new ArchiveBombError(
        `zip entry "${name}" declares ${uncompSize} B (> per-entry cap ${limits.maxEntryUncompressedBytes} B)`,
      );
    }
    budget.totalBytes += uncompSize;
    if (budget.totalBytes > limits.maxTotalUncompressedBytes) {
      throw new ArchiveBombError(
        `cumulative declared uncompressed size exceeds cap ${limits.maxTotalUncompressedBytes} B`,
      );
    }

    entries.push({ name, method, compSize, uncompSize, localOffset });
    p += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

// Extract one zip entry's bytes — only called when RECURSING into a nested archive
// (plain detection needs the name only). Bounded by the per-entry inflate cap.
function extractZipEntryData(buf, entry, limits) {
  const lo = entry.localOffset;
  if (lo + 30 > buf.length || buf.readUInt32LE(lo) !== 0x04034b50) {
    throw new ArchiveBombError(`zip entry "${entry.name}": bad local file header`);
  }
  const nameLen = buf.readUInt16LE(lo + 26);
  const extraLen = buf.readUInt16LE(lo + 28);
  const dataStart = lo + 30 + nameLen + extraLen;
  const stored = buf.subarray(dataStart, dataStart + entry.compSize);
  if (entry.method === 0) {
    return stored; // stored (no compression)
  }
  if (entry.method === 8) {
    try {
      return zlib.inflateRawSync(stored, { maxOutputLength: limits.maxEntryUncompressedBytes });
    } catch (error) {
      throw new ArchiveBombError(
        `zip entry "${entry.name}" inflate exceeded the per-entry cap or is corrupt: ${error.code || error.message}`,
      );
    }
  }
  throw new UninspectableArchiveError(
    `zip entry "${entry.name}" uses unsupported compression method ${entry.method}`,
  );
}

// Decide how to read a buffer: magic bytes first (robust against a mislabeled file —
// e.g. a weight archive renamed to hide it), then the display path's extension.
function sniffArchiveFormat(buf, displayPath) {
  const lower = displayPath.toLowerCase();
  if (buf.length >= 4) {
    const magic = buf.readUInt32LE(0);
    if (magic === 0x04034b50 || magic === 0x06054b50 || magic === 0x08074b50) {
      return "zip";
    }
  }
  if (buf.length >= 2 && buf[0] === 0x1f && buf[1] === 0x8b) {
    return "gzip";
  }
  if (buf.length >= 262 && buf.toString("ascii", 257, 262) === "ustar") {
    return "tar";
  }
  if (lower.endsWith(".zip") || lower.endsWith(".nsis.zip")) {
    return "zip";
  }
  if (lower.endsWith(".tar.gz") || lower.endsWith(".tgz") || lower.endsWith(".gz")) {
    return "gzip";
  }
  if (lower.endsWith(".tar")) {
    return "tar";
  }
  return "unknown";
}

function looksLikeArchive(name) {
  const lower = name.toLowerCase();
  if (lower.endsWith(".tar.gz") || lower.endsWith(".nsis.zip")) {
    return true;
  }
  return ARCHIVE_EXTENSIONS.has(path.posix.extname(lower));
}

// Classify one archive entry (and recurse if the entry is itself an archive). Shares
// classifyEntry() with the loose-tree walk, so detection can never drift between them.
function checkArchiveEntry(archiveDisplay, entryName, getData, ctx, depth, budget) {
  const posixName = entryName.split("\\").join("/");
  const display = `${archiveDisplay}!/${posixName}`;
  const needle = display.split(path.sep).join("/").toLowerCase();
  const basename = path.posix.basename(posixName);
  const hit = classifyEntry({ needle, basename, includeBin: ctx.includeBin, ncTokens: ctx.ncTokens });
  if (hit) {
    ctx.violations.push({ file: display, basename, ncToken: hit.ncToken });
  }
  if (looksLikeArchive(basename)) {
    const inner = getData();
    try {
      scanArchiveBuffer(inner, display, ctx, depth + 1, budget);
    } catch (error) {
      if (error instanceof UninspectableArchiveError && ctx.skipUninspectable) {
        ctx.warnings.push({ file: display, reason: error.reason });
        return;
      }
      throw error; // ArchiveBombError (fatal) and a non-skipped Uninspectable propagate
    }
  }
}

// Read one archive buffer: dispatch by sniffed format, enforce depth, and classify
// every entry inside. Throws ArchiveBombError / UninspectableArchiveError on refusal.
function scanArchiveBuffer(buf, displayPath, ctx, depth, budget) {
  if (depth > ctx.limits.maxDepth) {
    throw new ArchiveBombError(`archive nesting depth exceeds cap ${ctx.limits.maxDepth} at "${displayPath}"`);
  }
  const format = sniffArchiveFormat(buf, displayPath);

  if (format === "zip") {
    const entries = parseZipEntries(buf, ctx.limits, budget);
    for (const entry of entries) {
      checkArchiveEntry(displayPath, entry.name, () => extractZipEntryData(buf, entry, ctx.limits), ctx, depth, budget);
    }
    return;
  }

  if (format === "gzip") {
    let inflated;
    try {
      inflated = zlib.gunzipSync(buf, { maxOutputLength: ctx.limits.maxTotalUncompressedBytes });
    } catch (error) {
      throw new ArchiveBombError(
        `gunzip of "${displayPath}" exceeded the total cap ${ctx.limits.maxTotalUncompressedBytes} B or is corrupt: ${error.code || error.message}`,
      );
    }
    if (inflated.length >= 262 && inflated.toString("ascii", 257, 262) === "ustar") {
      const entries = parseTarEntries(inflated, ctx.limits, budget);
      for (const entry of entries) {
        checkArchiveEntry(
          displayPath,
          entry.name,
          () => inflated.subarray(entry.dataStart, entry.dataStart + entry.size),
          ctx,
          depth,
          budget,
        );
      }
      return;
    }
    // Bare .gz of a single file: the entry name is the archive name minus .gz/.tgz.
    budget.entries += 1;
    if (budget.entries > ctx.limits.maxEntries) {
      throw new ArchiveBombError(`entry count exceeds cap ${ctx.limits.maxEntries}`);
    }
    budget.totalBytes += inflated.length;
    if (budget.totalBytes > ctx.limits.maxTotalUncompressedBytes) {
      throw new ArchiveBombError(`cumulative uncompressed size exceeds cap ${ctx.limits.maxTotalUncompressedBytes} B`);
    }
    const innerName = path.posix.basename(displayPath.split(path.sep).join("/")).replace(/\.(gz|tgz)$/i, "");
    checkArchiveEntry(displayPath, innerName, () => inflated, ctx, depth, budget);
    return;
  }

  if (format === "tar") {
    const entries = parseTarEntries(buf, ctx.limits, budget);
    for (const entry of entries) {
      checkArchiveEntry(
        displayPath,
        entry.name,
        () => buf.subarray(entry.dataStart, entry.dataStart + entry.size),
        ctx,
        depth,
        budget,
      );
    }
    return;
  }

  throw new UninspectableArchiveError(
    `no pure-JS reader for "${displayPath}" (opaque container such as .dmg/.msi/.7z); its contents are not verifiable here`,
  );
}

// Walk the scan roots, find every archive FILE, and decompress-and-scan its contents.
// Returns { violations, refusals, warnings }. A refusal (bomb guard, or an
// uninspectable format without --skip-uninspectable) fails the build; a warning does
// not. This never inflates past the caps, so a bomb is refused rather than OOMing.
async function scanArchivesUnderRoots({ roots, ncTokens, includeBin, limits, skipUninspectable }) {
  const ctx = { ncTokens, includeBin, limits, skipUninspectable, violations: [], warnings: [] };
  const refusals = [];
  let archiveCount = 0;
  for (const scanRoot of roots) {
    const absRoot = path.isAbsolute(scanRoot) ? scanRoot : path.join(root, scanRoot);
    let rootStat;
    try {
      rootStat = await stat(absRoot);
    } catch (error) {
      if (error.code === "ENOENT") {
        continue;
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
      if (!looksLikeArchive(file)) {
        continue;
      }
      archiveCount += 1;
      const display = path.relative(root, file) || file;
      let fileStat;
      try {
        fileStat = await stat(file);
      } catch (error) {
        if (error.code === "ENOENT") {
          continue;
        }
        throw error;
      }
      if (fileStat.size > limits.maxArchiveFileBytes) {
        refusals.push({
          file: display,
          reason:
            `archive file is ${fileStat.size} B (> on-disk cap ${limits.maxArchiveFileBytes} B) — refusing to load it; ` +
            `raise --max-archive-bytes if this artifact is legitimately this large`,
        });
        continue;
      }
      const buf = await readFile(file);
      const budget = { entries: 0, totalBytes: 0 };
      try {
        scanArchiveBuffer(buf, display, ctx, 1, budget);
      } catch (error) {
        if (error instanceof ArchiveBombError) {
          refusals.push({ file: display, reason: `archive-bomb guard: ${error.reason}` });
        } else if (error instanceof UninspectableArchiveError) {
          if (skipUninspectable) {
            ctx.warnings.push({ file: display, reason: error.reason });
          } else {
            refusals.push({
              file: display,
              reason:
                `${error.reason}. Pass --skip-uninspectable to warn instead of fail when the opaque installer's ` +
                `contents are already covered by the loose-tree walk (the .app/resource tree it wraps)`,
            });
          }
        } else {
          throw error;
        }
      }
    }
  }
  return { violations: ctx.violations, refusals, warnings: ctx.warnings, archiveCount };
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

// A weight was found INSIDE an archive (sc-10551). `violation.file` reads like
// `bundle/App.app.tar.gz!/Contents/.../model.safetensors` so the archive + inner path
// are both visible.
function reportArchiveViolations(violations) {
  console.error(
    "\nNC-WEIGHTS GUARD FAILED (sc-10551): model-weight file(s) found INSIDE an archive " +
      "in the artifact payload.\n" +
      "A weight sealed inside a `.tar.gz` / `.zip` (e.g. the updater payload) still ships " +
      "the weight even though the loose-tree file scan cannot see into the container. " +
      "SceneWorks converts at install and pulls at runtime — it never redistributes " +
      "weights.\n",
  );
  for (const violation of violations) {
    if (violation.ncToken) {
      console.error(
        `  ✗ ${violation.file}\n` +
          `      matches Non-Commercial family token "${violation.ncToken}" — this is an ` +
          `NC weight and MUST NOT ship inside any archive in an installer/image/re-host.`,
      );
    } else {
      console.error(
        `  ✗ ${violation.file}\n` +
          `      unexpected model weight sealed inside an archive. Remove it from the ` +
          `build (weights are pulled at runtime), or if it is a legitimately-bundled ` +
          `permissive weight add its basename to the ALLOWLIST.`,
      );
    }
  }
  console.error(
    "\nSee docs/packaging-nc-weights-guard.md for the rule and the allowlist policy.\n",
  );
}

// An archive was REFUSED (archive-bomb guard tripped, or an opaque format with no
// pure-JS reader and no --skip-uninspectable). Fail closed — never OOM, never a silent
// pass.
function reportArchiveRefusals(refusals) {
  console.error(
    "\nNC-WEIGHTS GUARD FAILED (sc-10551): refused to scan an archive (fail-closed).\n" +
      "An archive-bomb guard tripped or the container has no pure-JS reader. The guard " +
      "refuses rather than exhaust memory/disk or silently skip an un-verified archive.\n",
  );
  for (const refusal of refusals) {
    console.error(`  ✗ ${refusal.file}\n      ${refusal.reason}.`);
  }
  console.error(
    "\nSee docs/packaging-nc-weights-guard.md for the archive-scan limits and flags.\n",
  );
}

// Non-fatal: an opaque archive that --skip-uninspectable downgraded to a warning
// (its contents duplicate the already-walked loose tree).
function printArchiveWarnings(warnings) {
  for (const warning of warnings) {
    console.warn(
      `  ⚠ ${warning.file}\n      not decompressed (opaque format): ${warning.reason}. ` +
        `Its contents are covered by the loose-tree walk of the sibling .app/resource tree.`,
    );
  }
}

function parseIntArg(value, flag) {
  if (value === undefined) {
    throw new Error(`${flag} requires an integer argument`);
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${flag} requires a positive integer (got "${value}")`);
  }
  return parsed;
}

function parseArgs(argv) {
  const options = {
    dirs: [],
    includeBin: false,
    selfTest: false,
    scanArchives: false,
    skipUninspectable: false,
    limits: { ...DEFAULT_ARCHIVE_LIMITS },
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--self-test") {
      options.selfTest = true;
    } else if (arg === "--include-bin") {
      options.includeBin = true;
    } else if (arg === "--scan-archives") {
      options.scanArchives = true;
    } else if (arg === "--skip-uninspectable") {
      options.skipUninspectable = true;
    } else if (arg === "--dir") {
      const value = argv[i + 1];
      if (!value) {
        throw new Error("--dir requires a path argument");
      }
      options.dirs.push(value);
      i += 1;
    } else if (arg.startsWith("--dir=")) {
      options.dirs.push(arg.slice("--dir=".length));
    } else if (arg === "--max-archive-bytes") {
      options.limits.maxArchiveFileBytes = parseIntArg(argv[(i += 1)], arg);
    } else if (arg === "--max-uncompressed") {
      options.limits.maxTotalUncompressedBytes = parseIntArg(argv[(i += 1)], arg);
    } else if (arg === "--max-entry-bytes") {
      options.limits.maxEntryUncompressedBytes = parseIntArg(argv[(i += 1)], arg);
    } else if (arg === "--max-entries") {
      options.limits.maxEntries = parseIntArg(argv[(i += 1)], arg);
    } else if (arg === "--max-depth") {
      options.limits.maxDepth = parseIntArg(argv[(i += 1)], arg);
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
  // --scan-archives (sc-10551): decompress-and-scan the CONTENTS of every archive under
  // the roots. Opt-in so it never slows an ordinary build; wired into the release lane.
  let archive = null;
  if (options.scanArchives) {
    archive = await scanArchivesUnderRoots({
      roots,
      ncTokens: tokens,
      includeBin: options.includeBin,
      limits: options.limits,
      skipUninspectable: options.skipUninspectable,
    });
  }
  return {
    tokens,
    families,
    roots,
    violations,
    resourceViolations,
    resourceSpecCount: specs.length,
    archive,
  };
}

// -- Archive fixture builders (self-test only) --------------------------------------
// Build real, valid `.tar` / `.tar.gz` / `.zip` bytes in pure JS so the archive-scan
// self-test needs no external `tar`/`zip` tool and is fully deterministic. The readers
// above do not verify CRCs, so stored/deflate-with-zero-CRC fixtures are sufficient.
function makeTarHeader(name, size) {
  const header = Buffer.alloc(512);
  header.write(name, 0, "utf8"); // name (fixtures use <100-byte names)
  header.write("0000644", 100, "ascii"); // mode
  header.write("0000000", 108, "ascii"); // uid
  header.write("0000000", 116, "ascii"); // gid
  header.write(`${size.toString(8).padStart(11, "0")} `, 124, "ascii"); // size (octal)
  header.write("00000000000 ", 136, "ascii"); // mtime
  header.write("        ", 148, "ascii"); // checksum field = 8 spaces while summing
  header[156] = 0x30; // typeflag '0' (regular file)
  header.write("ustar\0", 257, "ascii"); // magic
  header.write("00", 263, "ascii"); // version
  let sum = 0;
  for (let i = 0; i < 512; i += 1) {
    sum += header[i];
  }
  header.write(`${sum.toString(8).padStart(6, "0")}\0 `, 148, "ascii"); // 6-oct + NUL + space
  return header;
}

function makeTar(entries) {
  const parts = [];
  for (const entry of entries) {
    const content = Buffer.isBuffer(entry.content) ? entry.content : Buffer.from(entry.content, "utf8");
    parts.push(makeTarHeader(entry.name, content.length), content);
    const pad = (512 - (content.length % 512)) % 512;
    if (pad) {
      parts.push(Buffer.alloc(pad));
    }
  }
  parts.push(Buffer.alloc(1024)); // two zero blocks = end of archive
  return Buffer.concat(parts);
}

function makeTarGz(entries) {
  return zlib.gzipSync(makeTar(entries));
}

function makeZip(entries) {
  const locals = [];
  const central = [];
  let offset = 0;
  for (const entry of entries) {
    const raw = Buffer.isBuffer(entry.content) ? entry.content : Buffer.from(entry.content, "utf8");
    const method = entry.method ?? 0;
    const stored = method === 8 ? zlib.deflateRawSync(raw) : raw;
    const nameBuf = Buffer.from(entry.name, "utf8");

    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4); // version needed
    local.writeUInt16LE(0, 6); // flags
    local.writeUInt16LE(method, 8);
    local.writeUInt16LE(0, 10); // mod time
    local.writeUInt16LE(0, 12); // mod date
    local.writeUInt32LE(0, 14); // crc (readers ignore it)
    local.writeUInt32LE(stored.length, 18); // compressed size
    local.writeUInt32LE(raw.length, 22); // uncompressed size
    local.writeUInt16LE(nameBuf.length, 26);
    local.writeUInt16LE(0, 28); // extra length
    const localOffset = offset;
    locals.push(local, nameBuf, stored);
    offset += 30 + nameBuf.length + stored.length;

    const cd = Buffer.alloc(46);
    cd.writeUInt32LE(0x02014b50, 0);
    cd.writeUInt16LE(20, 4); // version made by
    cd.writeUInt16LE(20, 6); // version needed
    cd.writeUInt16LE(0, 8); // flags
    cd.writeUInt16LE(method, 10);
    cd.writeUInt16LE(0, 12); // mod time
    cd.writeUInt16LE(0, 14); // mod date
    cd.writeUInt32LE(0, 16); // crc
    cd.writeUInt32LE(stored.length, 20); // compressed size
    cd.writeUInt32LE(raw.length, 24); // uncompressed size
    cd.writeUInt16LE(nameBuf.length, 28);
    cd.writeUInt16LE(0, 30); // extra length
    cd.writeUInt16LE(0, 32); // comment length
    cd.writeUInt16LE(0, 34); // disk number start
    cd.writeUInt16LE(0, 36); // internal attrs
    cd.writeUInt32LE(0, 38); // external attrs
    cd.writeUInt32LE(localOffset, 42); // local header offset
    central.push(cd, nameBuf);
  }
  const localPart = Buffer.concat(locals);
  const centralPart = Buffer.concat(central);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(0, 4); // disk number
  eocd.writeUInt16LE(0, 6); // disk with central dir
  eocd.writeUInt16LE(entries.length, 8); // entries on this disk
  eocd.writeUInt16LE(entries.length, 10); // total entries
  eocd.writeUInt32LE(centralPart.length, 12); // central dir size
  eocd.writeUInt32LE(localPart.length, 16); // central dir offset
  eocd.writeUInt16LE(0, 20); // comment length
  return Buffer.concat([localPart, centralPart, eocd]);
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
  // Match on how the token ARRIVES, not an exact bootstrap string. The bootstrap
  // seed and the eventual manifest wiring produce different literal tokens for the
  // same family — the seed used to carry a bare `circlestone-labs`, but the manifest
  // yields `circlestone-labs/anima` + `models--circlestone-labs--anima` + `anima` and
  // NO bare `circlestone-labs`. Asserting on the exact bare token would fire a second,
  // unadvertised failure the moment sc-10523 lands and an engineer (correctly) empties
  // the seed per the tripwire below. A substring match is robust to both worlds.
  assert(
    "Anima is covered (bootstrap until epic 10512 lands in the manifest)",
    tokens.some((t) => /anima/.test(t)) && tokens.some((t) => /circlestone/.test(t)),
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

  // ========================================================================
  // (m)-(x) --scan-archives (sc-10551): decompress-and-scan archive CONTENTS
  // ========================================================================
  // A weight sealed inside an archive must be caught (planted-positive), a clean
  // archive must pass (negative), and an archive-bomb-shaped input must be REFUSED —
  // not OOM (fail-closed). Each assertion is failure-capable: a scanner that did not
  // look inside archives would flag 0 violations and fail the positive assertions.
  const freshCtx = (limits) => ({
    ncTokens: tokens,
    includeBin: false,
    limits: limits ?? { ...DEFAULT_ARCHIVE_LIMITS },
    skipUninspectable: false,
    violations: [],
    warnings: [],
  });
  const newBudget = () => ({ entries: 0, totalBytes: 0 });
  const expectRefused = (label, fn) => {
    let caught = null;
    try {
      fn();
    } catch (error) {
      caught = error;
    }
    assert(label, caught instanceof ArchiveBombError);
  };

  // (m) NC weights (`.safetensors` + a canonical/NC `.bin`) sealed in a `.tar.gz` — the
  //     macOS updater format — are detected and NC-classified. Planting BOTH extensions
  //     mirrors the sc-10526 review lesson that a `.bin` must not be a blind spot.
  {
    const gz = makeTarGz([
      { name: "app/README.txt", content: "no weights here" },
      { name: "models--circlestone-labs--Anima/model.safetensors", content: "w" },
      { name: "models--circlestone-labs--Anima/pytorch_model.bin", content: "w" },
      { name: "anima-base.safetensors", content: "w" },
    ]);
    const ctx = freshCtx();
    scanArchiveBuffer(gz, "bundle/macos/App.app.tar.gz", ctx, 1, newBudget());
    assert("tar.gz: NC weights sealed inside are detected", ctx.violations.length === 3);
    assert(
      "tar.gz: the NC .bin inside is caught by default (no --include-bin)",
      ctx.violations.some((v) => v.basename === "pytorch_model.bin" && v.ncToken != null),
    );
    assert("tar.gz: every detected weight is NC-classified", ctx.violations.every((v) => v.ncToken != null));
  }

  // (n) NC weights (`.safetensors` + a deflate-compressed NC `.bin`) sealed in a `.zip`
  //     — the Windows `.nsis.zip` updater format — are detected and NC-classified.
  {
    const zip = makeZip([
      { name: "app/README.txt", content: "no weights here" },
      { name: "models--circlestone-labs--Anima/model.safetensors", content: "w" },
      { name: "anima-base.bin", content: "w", method: 8 },
    ]);
    const ctx = freshCtx();
    scanArchiveBuffer(zip, "bundle/App.nsis.zip", ctx, 1, newBudget());
    assert("zip: NC weights sealed inside are detected", ctx.violations.length === 2);
    assert(
      "zip: the NC .bin inside is detected and NC-classified",
      ctx.violations.some((v) => v.basename === "anima-base.bin" && v.ncToken != null),
    );
  }

  // (o) Nested archive: a `.tar.gz` sealed inside a DEFLATE-compressed zip entry is
  //     recursed into (exercises deflate extraction + gunzip + tar + the depth path),
  //     and the NC weight inside is caught with a nesting-aware path.
  {
    const inner = makeTarGz([
      { name: "models--circlestone-labs--Anima/model.safetensors", content: "w" },
    ]);
    const outer = makeZip([
      { name: "app/ok.txt", content: "fine" },
      { name: "payload/inner.tar.gz", content: inner, method: 8 },
    ]);
    const ctx = freshCtx();
    scanArchiveBuffer(outer, "bundle/outer.zip", ctx, 1, newBudget());
    assert(
      "nested tar.gz inside a deflate zip entry is recursed and the NC weight detected",
      ctx.violations.length === 1 && ctx.violations[0].ncToken != null,
    );
    assert(
      "nested violation path shows the archive nesting (outer.zip!/…/inner.tar.gz!/…)",
      /outer\.zip!\/payload\/inner\.tar\.gz!\//.test(ctx.violations[0].file),
    );
  }

  // (p) Clean archives pass. A generic `.bin` (font/wasm blob) inside is NOT a weight
  //     by default — no false positives.
  {
    const cleanGz = makeTarGz([
      { name: "app/index.js", content: "x" },
      { name: "app/logo.png", content: "y" },
    ]);
    const cleanZip = makeZip([
      { name: "app/index.js", content: "x" },
      { name: "app/glyphs.bin", content: "a font blob" },
    ]);
    const c1 = freshCtx();
    scanArchiveBuffer(cleanGz, "bundle/clean.tar.gz", c1, 1, newBudget());
    const c2 = freshCtx();
    scanArchiveBuffer(cleanZip, "bundle/clean.zip", c2, 1, newBudget());
    assert("clean tar.gz passes", c1.violations.length === 0);
    assert("clean zip passes (a generic .bin is not a weight by default)", c2.violations.length === 0);
  }

  // (q) The allowlisted permissive weight passes even inside an archive; (r) an
  //     unexpected non-allowlisted weight inside an archive still fails (generic).
  {
    const okGz = makeTarGz([
      { name: "vendor/aesthetic-v2-sac-logos-ava1-l14.safetensors", content: "permissive" },
    ]);
    const badGz = makeTarGz([{ name: "vendor/some-random-model.safetensors", content: "w" }]);
    const okCtx = freshCtx();
    scanArchiveBuffer(okGz, "bundle/ok.tar.gz", okCtx, 1, newBudget());
    const badCtx = freshCtx();
    scanArchiveBuffer(badGz, "bundle/bad.tar.gz", badCtx, 1, newBudget());
    assert("allowlisted permissive weight inside an archive passes", okCtx.violations.length === 0);
    assert(
      "unexpected non-allowlisted weight inside an archive is flagged (generic)",
      badCtx.violations.length === 1 && badCtx.violations[0].ncToken == null,
    );
  }

  // (s) A bare `.gz` of a single weight (name = archive name minus .gz) is detected.
  {
    const gz = zlib.gzipSync(Buffer.from("weights"));
    const ctx = freshCtx();
    scanArchiveBuffer(gz, "bundle/anima-base.safetensors.gz", ctx, 1, newBudget());
    assert(
      "bare .gz of a weight is detected by its derived inner name",
      ctx.violations.length === 1 && ctx.violations[0].ncToken != null,
    );
  }

  // (t) ARCHIVE-BOMB: too many entries → refused (not OOM). Small configured cap
  //     exercises the exact refusal path a real "many tiny files" bomb would hit.
  {
    const many = [];
    for (let i = 0; i < 10; i += 1) {
      many.push({ name: `f${i}.txt`, content: "x" });
    }
    const gz = makeTarGz(many);
    expectRefused("archive-bomb: entry count over cap is refused", () =>
      scanArchiveBuffer(gz, "bundle/many.tar.gz", freshCtx({ ...DEFAULT_ARCHIVE_LIMITS, maxEntries: 3 }), 1, newBudget()),
    );
  }

  // (u) ARCHIVE-BOMB: a zip whose DECLARED per-entry uncompressed size exceeds the cap
  //     is refused BEFORE any inflate — the classic zip-bomb defense.
  {
    const zip = makeZip([{ name: "payload.bin", content: "A".repeat(5000) }]);
    expectRefused("archive-bomb: zip declared-oversized entry is refused before inflating", () =>
      scanArchiveBuffer(
        zip,
        "bundle/bomb.zip",
        freshCtx({ ...DEFAULT_ARCHIVE_LIMITS, maxEntryUncompressedBytes: 1000 }),
        1,
        newBudget(),
      ),
    );
  }

  // (v) ARCHIVE-BOMB: a gzip whose INFLATED size exceeds the total cap is refused
  //     (Node's gunzip maxOutputLength throws before fully inflating).
  {
    const gz = makeTarGz([{ name: "big.txt", content: "A".repeat(50000) }]);
    expectRefused("archive-bomb: gzip inflating past the total cap is refused (not OOM)", () =>
      scanArchiveBuffer(
        gz,
        "bundle/big.tar.gz",
        freshCtx({ ...DEFAULT_ARCHIVE_LIMITS, maxTotalUncompressedBytes: 1000 }),
        1,
        newBudget(),
      ),
    );
  }

  // (w) ARCHIVE-BOMB: deeply-nested archive beyond the depth cap is refused (bounds
  //     recursion; defeats a quine/nesting bomb). Wrap a core tar.gz repeatedly.
  {
    let nested = makeTarGz([{ name: "core.txt", content: "x" }]);
    for (let i = 0; i < 6; i += 1) {
      nested = makeTarGz([{ name: "inner.tar.gz", content: nested }]);
    }
    expectRefused("archive-bomb: nesting depth over cap is refused (no infinite recursion)", () =>
      scanArchiveBuffer(nested, "bundle/nested.tar.gz", freshCtx({ ...DEFAULT_ARCHIVE_LIMITS, maxDepth: 3 }), 1, newBudget()),
    );
  }

  // (x) End-to-end on real files: scanArchivesUnderRoots plants a `.app.tar.gz` and a
  //     `.nsis.zip` on disk and finds the NC weights in both; and an opaque `.dmg`
  //     fails closed by default but is downgraded to a warning by --skip-uninspectable.
  const archTmp = await mkdtemp(path.join(os.tmpdir(), "nc-weights-archives-"));
  try {
    await mkdir(path.join(archTmp, "macos"), { recursive: true });
    await writeFile(
      path.join(archTmp, "macos", "App.app.tar.gz"),
      makeTarGz([
        { name: "Contents/Resources/models--circlestone-labs--Anima/model.safetensors", content: "w" },
      ]),
    );
    await writeFile(
      path.join(archTmp, "App.nsis.zip"),
      makeZip([{ name: "resources/anima-base.safetensors", content: "w" }]),
    );
    const e2e = await scanArchivesUnderRoots({
      roots: [archTmp],
      ncTokens: tokens,
      includeBin: false,
      limits: { ...DEFAULT_ARCHIVE_LIMITS },
      skipUninspectable: true,
    });
    assert(
      "end-to-end: NC weights found inside BOTH a .app.tar.gz and a .nsis.zip on disk",
      e2e.violations.length === 2 && e2e.violations.every((v) => v.ncToken != null),
    );

    await writeFile(path.join(archTmp, "installer.dmg"), Buffer.from("opaque disk image bytes"));
    const failClosed = await scanArchivesUnderRoots({
      roots: [path.join(archTmp)],
      ncTokens: tokens,
      includeBin: false,
      limits: { ...DEFAULT_ARCHIVE_LIMITS },
      skipUninspectable: false,
    });
    assert(
      "uninspectable .dmg fails closed by default (recorded as a refusal)",
      failClosed.refusals.some((r) => /installer\.dmg/.test(r.file)),
    );
    const skipped = await scanArchivesUnderRoots({
      roots: [path.join(archTmp)],
      ncTokens: tokens,
      includeBin: false,
      limits: { ...DEFAULT_ARCHIVE_LIMITS },
      skipUninspectable: true,
    });
    assert(
      "--skip-uninspectable downgrades an opaque .dmg to a warning (no refusal)",
      skipped.warnings.some((w) => /installer\.dmg/.test(w.file)) &&
        !skipped.refusals.some((r) => /installer\.dmg/.test(r.file)),
    );
  } finally {
    await rm(archTmp, { recursive: true, force: true });
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

  const { tokens, roots, violations, resourceViolations, resourceSpecCount, archive } =
    await runScan(options);
  console.log(
    `NC-weights guard: scanned ${roots.join(", ")} for model weights ` +
      `(${tokens.length} NC family tokens from the manifests + bootstrap) and ` +
      `${resourceSpecCount} declared Tauri bundle-resource spec(s)` +
      (archive ? `; decompress-and-scanned ${archive.archiveCount} archive(s)` : "") +
      ".",
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
  if (archive) {
    if (archive.warnings.length > 0) {
      printArchiveWarnings(archive.warnings);
    }
    if (archive.violations.length > 0) {
      reportArchiveViolations(archive.violations);
      failed = true;
    }
    if (archive.refusals.length > 0) {
      reportArchiveRefusals(archive.refusals);
      failed = true;
    }
  }
  if (failed) {
    process.exitCode = 1;
    return;
  }
  console.log(
    "NC-weights guard passed: no model weights in the artifact payload and no " +
      "weight-bearing bundle resources declared" +
      (archive ? " and no weights inside any scanned archive" : "") +
      ".",
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
