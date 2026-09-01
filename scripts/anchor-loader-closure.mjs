// The memory anchor currency key: a digest of ONE model's OWN loader closure (sc-22511, epic 22505).
//
// WHY THIS EXISTS, AND HOW IT DIFFERS FROM THE PROVIDER CLOSURE DIGEST
//
// `scripts/inference-closure-digest.mjs` answers "did the code this PROVIDER compiles change?" — a
// CRATE-level unit (one crate plus every first-party crate it links, plus the locked packages it
// reaches, plus workspace codegen inputs). That unit is right for a calibration campaign, which
// measures a whole compiled lane. It is too wide for a memory ANCHOR, whose contract (E9) is
// stricter: nothing invalidates an anchor except a direct change to the code that LOADS THAT MODEL.
// Under the crate unit, all three of these still rotate a model's currency:
//
//   * a pin bump with identical loader content   — the locked set and workspace inputs move with the
//     repository, not with the model;
//   * a sibling model in the SAME crate          — `mlx:ltx_2_3` and `mlx:ltx_2_5` are one crate
//     (`mlx-gen-ltx`), so a 2.3-only edit rotates 2.5;
//   * a shared crate the loader never reaches    — crate-level linkage is coarser than execution.
//
// So this module derives a narrower, FILE-level unit: the transitive set of source files reachable
// from the model's declared LOADER ENTRY POINTS, through Rust's own module and import graph. The
// digest is the content of those files and NOTHING else — no revision, no `Cargo.lock`, no
// `rust-toolchain.toml`, no root `[profile]`, no crate manifests. A pin bump whose loader files are
// byte-identical therefore leaves the key unchanged, which is the whole point: an anchor predating
// an unrelated change stays authoritative forever.
//
// WHAT IS IN THE UNIT
//
//   * every intra-crate module named by a `crate::` / `super::` / `self::` / in-scope-module path in
//     a reached file, transitively — REFERENCED modules, not every `mod` a crate declares;
//   * every FIRST-PARTY CRATE a reached file names in a path (`mlx_gen::array::contiguous`,
//     `use gen_core::…`), entered at the module the path names and walked the same way — so a shared
//     crate the loader genuinely reaches IS in the unit, and one it does not is not;
//   * a `pub use` re-export in a reached file, followed for the identifiers something actually asked
//     that file for — which is how an item imported through a crate root's routing table is still
//     attributed to the module that defines it, without the table dragging its whole crate along;
//   * `include_str!` / `include_bytes!` targets, whose CONTENT is a compile input (they are hashed,
//     never walked as code — an included file that happens to be Rust is data here).
//
// Rust source is hashed SEMANTICALLY (`stripInertLines`, shared with the provider digest), so a
// doc-only edit inside the loader does not stale an anchor. Anything the stripper does not
// understand is hashed by its exact bytes.
//
// WHAT IS DELIBERATELY NOT IN THE UNIT — read before trusting it
//
//   * External dependency versions. A `Cargo.lock` bump of a crate the loader compiles changes
//     generated code without touching a source file here, and this unit will not see it. That is the
//     price of the E9 contract: the lock moves with the REPOSITORY, so including it re-couples every
//     model to every pin bump — the exact coupling this key exists to remove. Named here rather than
//     hidden: the lock's identity is still recorded, per-provider, by the crate-level digest in
//     `config/inference-provider-closures.json`, which the stale-lane report and the memory matrix
//     continue to read.
//   * Crate manifests and workspace codegen inputs, for the same reason. A manifest edit that
//     actually changes what the loader executes shows up as a `use` in a reached file.
//   * Item granularity. Reaching ANY item of a module pulls the whole FILE. Rust compiles a file as
//     one unit with free cross-item inlining, so finer scoping would be a false green. The
//     over-trigger is real and bounded: sibling models in the same crate are separated (different
//     files), sibling items in the same file are not — `mlx-gen-ltx/src/model.rs` hosts both LTX-2.3
//     and LTX-2.5 loaders, so a 2.3 edit THERE does stale 2.5.
//   * Macro-generated references. A module reached only through a name a macro builds is invisible
//     to a source-level walk, and would be missed. The walk is deliberately generous everywhere else
//     (an unresolvable head is followed if it names any in-scope module or first-party crate) so the
//     residue is small, but it is a real under-inclusion and not a proven-empty one.
//   * Realised memory behaviour. Like every code-identity unit, this answers "did the loading code
//     change?", never "did the measured peak change?".
//
// SCENEWORKS-SIDE CODE IS NOT IN THE UNIT, AND THAT IS A JUDGEMENT
//
// The worker glue that selects the model and shapes its request lives in THIS repository, and it is
// deliberately outside the closure. Two reasons, both about what the anchor measures: the anchor is
// a decomposition of an inference-side render's allocator behaviour, and the SceneWorks side neither
// allocates nor executes any of it; and a SceneWorks-side edit lands in the same commit graph as the
// anchor store itself, so it is reviewable in the diff, whereas the pinned inference source is not.
// The SceneWorks-side strategy PARAMETERS that do bound the render (rung declarations, tile edges)
// are separately bound to the anchor by the store's source handshake in `memory_anchor.rs`, which
// validates every anchor against the retained record's `strategy.engagedRungs` and geometry.

import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  fileContentHash,
  inferencePinFromCargo,
  readBlobs,
  resolveRevision,
  sha256,
  treeBlobs,
} from "./inference-closure-digest.mjs";
import { canonicalSourceText, stripInertLines } from "./lib/source-revision.mjs";

/**
 * Hashed into the canonical text, so a unit change reads as a digest change rather than silently
 * comparing two different questions.
 */
export const ANCHOR_LOADER_CLOSURE_VERSION = "anchor-loader-closure v1";

export const ANCHOR_LOADER_CONFIG_PATH = "config/anchor-loader-closures.json";

/** Directories under a crate that are never compiled into the shipped loader. */
const NON_SHIPPED_DIRS = ["tests/", "benches/", "examples/", "testdata/", "fixtures/"];

/** Rust keywords that appear as the head of a path but name no module or crate. */
const PATH_HEAD_KEYWORDS = new Set(["crate", "self", "super", "Self", "std", "core", "alloc"]);

/**
 * A read-only source tree: existence, a content id, and bodies.
 *
 * Everything below is written against this interface rather than against git, so the tests can
 * derive the SAME key over the SAME real repository paths with one file overlaid — which is how the
 * "a sibling edit does not stale this model" claims are asked, instead of being asserted over a
 * synthetic fixture that could agree with a broken walk.
 */
export function gitTree(repo, revision, { blobs } = {}) {
  const resolved = blobs ?? treeBlobs(repo, revision);
  return {
    paths: () => [...resolved.keys()],
    has: (file) => resolved.has(file),
    contentId: (file) => resolved.get(file),
    read(files) {
      const oids = files.map((file) => resolved.get(file)).filter(Boolean);
      const bodies = readBlobs(repo, oids);
      return new Map(files.map((file) => [file, bodies.get(resolved.get(file)) ?? ""]));
    },
  };
}

/**
 * `base` with some files replaced (or added, or removed when the body is `null`).
 *
 * Exported because it is the only honest way to ask a staleness question of a REAL tree: take the
 * pinned revision, change exactly one real file, and re-derive.
 */
export function overlayTree(base, overrides) {
  const entries = new Map(Object.entries(overrides));
  return {
    paths: () =>
      [...new Set([...base.paths(), ...entries.keys()])].filter(
        (file) => entries.get(file) !== null,
      ),
    has: (file) => (entries.has(file) ? entries.get(file) !== null : base.has(file)),
    contentId: (file) =>
      entries.has(file)
        ? entries.get(file) === null
          ? undefined
          : `overlay:${sha256(entries.get(file))}`
        : base.contentId(file),
    read(files) {
      const passthrough = files.filter((file) => !entries.has(file));
      const bodies = passthrough.length ? base.read(passthrough) : new Map();
      for (const file of files) {
        if (entries.has(file)) bodies.set(file, entries.get(file) ?? "");
      }
      return bodies;
    },
  };
}

/** `crates/media/mlx-gen/mlx-gen-ltx/src/a/b.rs` -> `crates/media/mlx-gen/mlx-gen-ltx`. */
export function crateDirOf(file) {
  const index = file.indexOf("/src/");
  return index === -1 ? null : file.slice(0, index);
}

/**
 * The directory child modules of `file` live in.
 *
 * `src/lib.rs` and `src/a/mod.rs` own their own directory; `src/a.rs` owns `src/a/`. Both spellings
 * are handled because both are used in the inference tree (`diff_vae.rs` + `diff_vae/budget.rs`).
 */
export function moduleDirOf(file) {
  const base = path.posix.basename(file);
  const dir = path.posix.dirname(file);
  if (base === "lib.rs" || base === "main.rs" || base === "mod.rs") return dir;
  return path.posix.join(dir, base.replace(/\.rs$/, ""));
}

/** The file backing module `name` under `dir`, in Cargo's two spellings. */
export function moduleFile(tree, dir, name) {
  const flat = path.posix.join(dir, `${name}.rs`);
  if (tree.has(flat)) return flat;
  const nested = path.posix.join(dir, name, "mod.rs");
  if (tree.has(nested)) return nested;
  return null;
}

function isShippedSource(file) {
  const crate = crateDirOf(file);
  if (!crate) return false;
  const relative = file.slice(crate.length + 1);
  return relative.startsWith("src/") && !NON_SHIPPED_DIRS.some((dir) => relative.startsWith(dir));
}

/**
 * `crate_name -> crates/its/dir` for every first-party crate in the workspace.
 *
 * Derived from the tree itself (every `<dir>/Cargo.toml` with a `src/lib.rs`) rather than from the
 * root manifest's `[workspace.dependencies]`: an import in a reached file names the CRATE, and a
 * crate reachable through a path dependency declared in a leaf manifest is just as real as one
 * declared at the root. Over-listing is harmless — a crate nothing imports is never entered.
 */
export function firstPartyCrates(tree, blobPaths) {
  const crates = new Map();
  for (const file of blobPaths) {
    if (!file.endsWith("/Cargo.toml")) continue;
    const dir = file.slice(0, -"/Cargo.toml".length);
    if (!tree.has(`${dir}/src/lib.rs`)) continue;
    crates.set(path.posix.basename(dir).replace(/-/g, "_"), dir);
  }
  return crates;
}

/**
 * The shipped body of a Rust source file: `#[cfg(test)]` items removed.
 *
 * This is not tidiness, it is the difference between a unit that discriminates and one that does
 * not. `mlx-gen/src/request_scope.rs` carries a `#[cfg(test)]` conformance test that
 * `include_str!`s the memory-strategy source of EVERY sibling model crate; walked as loader code it
 * drags eight other models into this model's closure, and hashed as loader content it lets a
 * sibling's edit stale this model's anchors. Test code compiles into no shipped loader, so it is
 * neither walked nor hashed here.
 *
 * Brace-depth counting from the attribute, over source with string and character literals and line
 * comments removed first — a `{` inside `write!(f, "{{")` would otherwise leave the depth stuck
 * open. A `#[cfg(test)] mod tests;` file declaration (no braces) is dropped as the single line it is.
 *
 * WHEN THE PARSE DOES NOT CLOSE, NOTHING IS DROPPED. An unbalanced run to end-of-file means this
 * approximate scanner lost the item's boundary — a raw string carrying an unmatched brace across
 * lines is the shape that does it — and dropping the remainder would silently take real loader code
 * out of the hash, which is a false green. So the whole run is KEPT instead: hashing test code costs
 * a spurious staleness at worst, and this failure mode is the one caught by
 * `an edit inside the model's own loader path stales the key`, which found it.
 */
export function stripCfgTest(body) {
  const lines = body.split("\n");
  const kept = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (!/^\s*#\[cfg\((?:test|all\(\s*test\b)/.test(line)) {
      kept.push(line);
      continue;
    }
    // Consume the attribute and the item it applies to.
    let depth = 0;
    let opened = false;
    let closed = false;
    const consumed = [];
    let cursor = index;
    for (; cursor < lines.length; cursor += 1) {
      const item = lines[cursor];
      consumed.push(item);
      const code = item
        .replace(/"(?:\\.|[^"\\])*"/g, '""')
        .replace(/'(?:\\.|[^'\\])*'/g, "''")
        .replace(/\/\/.*$/, "");
      for (const char of code) {
        if (char === "{") {
          depth += 1;
          opened = true;
        } else if (char === "}") {
          depth -= 1;
        }
      }
      if (opened && depth <= 0) {
        closed = true;
        break;
      }
      if (!opened && /;\s*$/.test(item.trim())) {
        closed = true;
        break;
      }
    }
    if (closed) {
      index = cursor;
      continue;
    }
    kept.push(...consumed);
    index = cursor;
  }
  return kept.join("\n");
}

/**
 * Expand one `use` path, brace groups and all, into full segment lists.
 *
 * `use a::b::{c, d::{e, f}, g as h};` is four distinct edges, and a walk that read the line with a
 * naive `::` split would see `d::e` as a top-level path and resolve it against the WRONG module. The
 * alias is dropped: `g as h` is still an edge to `a::b::g`.
 */
export function expandUsePath(text) {
  const out = [];
  const walk = (prefix, body) => {
    let depth = 0;
    let current = "";
    const parts = [];
    for (const char of body) {
      if (char === "{") depth += 1;
      if (char === "}") depth -= 1;
      if (char === "," && depth === 0) {
        parts.push(current);
        current = "";
        continue;
      }
      current += char;
    }
    parts.push(current);
    for (const part of parts) {
      const trimmed = part.trim().replace(/\s+as\s+[A-Za-z_][A-Za-z0-9_]*$/, "");
      if (!trimmed) continue;
      const brace = trimmed.indexOf("{");
      if (brace === -1) {
        const segments = [...prefix, ...trimmed.split("::")].map((s) => s.trim()).filter(Boolean);
        if (segments.length) out.push(segments);
        continue;
      }
      const head = trimmed
        .slice(0, brace)
        .split("::")
        .map((s) => s.trim())
        .filter(Boolean);
      walk([...prefix, ...head], trimmed.slice(brace + 1, trimmed.lastIndexOf("}")));
    }
  };
  walk([], text);
  return out;
}

/**
 * The references one Rust source body makes, split by the ONE distinction that decides whether this
 * unit discriminates between models at all.
 *
 * `pub use` lines are a crate root's ROUTING TABLE, not its code. Following them unconditionally is
 * what turns a loader closure into the whole repository: entering `mlx_gen` for one helper would
 * re-export its way into every model crate the workspace hub names, and an edit to an unrelated
 * model would stale this one — the exact coupling E9 forbids. So they are followed only for the
 * identifiers something actually asked that file for. Everything else — code lines and the file's
 * OWN (private or `pub(crate)`) `use` imports — is followed unconditionally, because those are the
 * dependencies the file compiles and executes.
 */
export function referencesIn(rawBody) {
  const body = stripCfgTest(rawBody);
  const code = [];
  const imports = [];
  const reexports = [];
  const includes = [];
  const lines = body.split("\n");
  for (let index = 0; index < lines.length; index += 1) {
    const raw = lines[index];
    const line = raw.trim();
    if (line.startsWith("//")) continue;
    const use = line.match(/^(pub\s+)?use\s+([\s\S]*)$/);
    if (use) {
      // A `use` item may wrap across lines; join until the terminating `;`.
      let text = use[2];
      let cursor = index;
      while (!text.includes(";") && cursor + 1 < lines.length) {
        cursor += 1;
        text += lines[cursor].trim();
      }
      index = cursor;
      const paths = expandUsePath(text.slice(0, text.indexOf(";") === -1 ? undefined : text.indexOf(";")));
      if (use[1]) {
        for (const segments of paths) reexports.push(segments);
      } else {
        for (const segments of paths) imports.push(segments);
      }
      continue;
    }
    for (const match of raw.matchAll(/\b([A-Za-z_][A-Za-z0-9_]*)((?:::[A-Za-z_][A-Za-z0-9_]*)+)/g)) {
      code.push([match[1], ...match[2].split("::").slice(1)]);
    }
    for (const match of raw.matchAll(/\binclude_(?:str|bytes)!\s*\(\s*"([^"]+)"/g)) {
      includes.push(match[1]);
    }
  }
  return { code, imports, reexports, includes };
}

const isModuleSegment = (segment) => /^[a-z_][a-z0-9_]*$/.test(segment);

/**
 * The transitive set of shipped source files reachable from `entryPoints`.
 *
 * The walk is REQUEST-DRIVEN: a file is admitted together with the identifiers that were asked of
 * it, and a `pub use` re-export is followed only when it exports one of them. That is what keeps a
 * shared crate's hub from dragging in every model beside it while still following the re-export a
 * reached file genuinely imports through.
 *
 * It is content-driven, so it sees exactly what the tree it is handed contains — the pinned
 * revision, or the pinned revision with one file overlaid. `"*"` requests everything from a file
 * (the entry points, and any file reached as a whole module).
 */
export function loaderClosureFiles({ tree, entryPoints, crates }) {
  for (const entry of entryPoints) {
    if (!tree.has(entry)) throw new Error(`loader entry point ${entry} does not exist in this tree`);
    if (!isShippedSource(entry)) throw new Error(`loader entry point ${entry} is not shipped source`);
  }
  /** file -> the identifiers asked of it. */
  const requested = new Map();
  let frontier = [];
  const request = (file, ident) => {
    if (!file || !isShippedSource(file)) return;
    const wanted = requested.get(file);
    if (!wanted) {
      requested.set(file, new Set([ident]));
      frontier.push(file);
      return;
    }
    if (wanted.has("*") || wanted.has(ident)) return;
    wanted.add(ident);
    // A new identifier can open a re-export the previous pass had no reason to follow.
    if (file.endsWith(".rs")) frontier.push(file);
  };
  // `include_str!`/`include_bytes!` targets are DATA: their content is a compile input, so it is
  // hashed, but they are never walked as loader code even when the included file happens to be
  // Rust source.
  const requestData = (file) => {
    if (!requested.has(file)) requested.set(file, new Set(["<data>"]));
  };
  for (const entry of entryPoints) request(entry, "*");

  /**
   * Resolve one path from a base directory: every leading segment that names a module is admitted
   * as a whole module, and the first segment that does not is requested as an ITEM of the last
   * module reached.
   */
  const resolveFrom = (dir, segments, fallbackFile) => {
    let current = dir;
    let file = null;
    for (const segment of segments) {
      if (!isModuleSegment(segment)) break;
      const next = moduleFile(tree, current, segment);
      if (!next) break;
      file = next;
      request(file, "*");
      current = moduleDirOf(file);
      segments = segments.slice(1);
    }
    const target = file ?? fallbackFile;
    if (target && segments.length) request(target, segments[0]);
    return Boolean(file);
  };

  const processed = new Set();
  while (frontier.length) {
    const wave = [...new Set(frontier)].filter((file) => file.endsWith(".rs"));
    frontier = [];
    const bodies = wave.length ? tree.read(wave) : new Map();
    for (const file of wave) {
      const crateDir = crateDirOf(file);
      const crateRoot = `${crateDir}/src`;
      const libRs = `${crateRoot}/lib.rs`;
      const dir = moduleDirOf(file);
      const parentDir = path.posix.dirname(dir);
      const wanted = requested.get(file) ?? new Set();
      const { code, imports, reexports, includes } = referencesIn(bodies.get(file) ?? "");

      const follow = (segments) => {
        const [head, ...rest] = segments;
        if (head === "crate") return resolveFrom(crateRoot, rest, libRs);
        if (head === "self") return resolveFrom(dir, rest, file);
        if (head === "super") return resolveFrom(parentDir, rest, null);
        if (PATH_HEAD_KEYWORDS.has(head)) return false;
        // An in-scope module of this file's own module, then of the crate root.
        if (resolveFrom(dir, segments, null)) return true;
        if (resolveFrom(crateRoot, segments, null)) return true;
        const otherCrate = crates.get(head);
        if (otherCrate && otherCrate !== crateDir) {
          return resolveFrom(`${otherCrate}/src`, rest, `${otherCrate}/src/lib.rs`);
        }
        return false;
      };

      if (!processed.has(file)) {
        processed.add(file);
        for (const segments of [...code, ...imports]) follow(segments);
        for (const include of includes) {
          const target = path.posix.normalize(path.posix.join(path.posix.dirname(file), include));
          if (isShippedSource(target) && tree.has(target)) requestData(target);
        }
      }
      for (const segments of reexports) {
        const exported = segments[segments.length - 1];
        if (wanted.has("*") || wanted.has(exported) || exported === "*") follow(segments);
      }
    }
  }
  return [...requested.keys()].sort();
}

/**
 * The canonical, diffable closure text. The digest is `sha256` of this.
 *
 * The REVISION IS ABSENT BY CONSTRUCTION — it is provenance, not content, and hashing it would make
 * every anchor stale on every pin bump, which is precisely the coupling E9 forbids.
 */
export function loaderClosureText({ model, entryPoints, files }) {
  return `${[
    `# ${ANCHOR_LOADER_CLOSURE_VERSION}`,
    `# model: ${model}`,
    "[entry-points]",
    ...[...entryPoints].sort(),
    "[source]",
    ...files.map(([file, hash]) => `${hash} ${file}`),
  ].join("\n")}\n`;
}

/**
 * The content hash of one closure file.
 *
 * Rust is hashed as SHIPPED SEMANTIC source — `#[cfg(test)]` items removed, then the inert lines the
 * shared stripper removes — so neither a doc-comment edit nor a unit-test edit inside the loader
 * stales an anchor, while any line that compiles into it does. Everything else falls through to the
 * sibling module's hash, which is the raw blob identity for a file type the stripper cannot read.
 */
export function loaderFileHash(file, contentId, body) {
  if (!file.endsWith(".rs")) return fileContentHash(file, contentId, body);
  return `s:${sha256(stripInertLines(canonicalSourceText(stripCfgTest(body ?? "")), "//"))}`;
}

/** Compute one `(model, backend lane)`'s loader closure digest over a tree. */
export function loaderClosureDigest({ model, entryPoints, tree, crates }) {
  const files = loaderClosureFiles({ tree, entryPoints, crates });
  const bodies = tree.read(files);
  const hashed = files.map((file) => [
    file,
    loaderFileHash(file, tree.contentId(file), bodies.get(file)),
  ]);
  const text = loaderClosureText({ model, entryPoints, files: hashed });
  return { model, entryPoints, files, digest: sha256(text), text };
}

/**
 * A declaration is only worth its entry pointer, so check it rather than trust it: the model id must
 * appear as a string literal in the declared entry points' own source. An entry point that never
 * names the model digests some other model's code path and reports currency for a loader it never
 * looked at — a false green, and the expensive kind.
 */
export function assertModelIsNamedByEntryPoints({ model, entryPoints, tree }) {
  const id = model.split(":")[0];
  const bodies = tree.read(entryPoints);
  const named = entryPoints.some((file) => (bodies.get(file) ?? "").includes(`"${id}"`));
  if (!named) {
    throw new Error(
      `anchor loader declaration "${model}" names entry points that never carry the literal "${id}". ` +
        `Fix the declaration in ${ANCHOR_LOADER_CONFIG_PATH} — a wrong entry point digests the ` +
        "wrong loader.",
    );
  }
}

/** Every declared model's digest at one revision, sharing the per-revision tree read. */
export function anchorLoaderDigests({ repo, revision, declared, tree }) {
  const resolved = tree ?? gitTree(repo, revision);
  const crates = firstPartyCrates(resolved, resolved.paths());
  const out = new Map();
  for (const [model, entry] of Object.entries(declared)) {
    assertModelIsNamedByEntryPoints({ model, entryPoints: entry.entryPoints, tree: resolved });
    out.set(model, loaderClosureDigest({ model, entryPoints: entry.entryPoints, tree: resolved, crates }));
  }
  return out;
}

/** Build the checked-in config body at one revision. */
export function buildAnchorLoaderConfig({ repo, revision, declared }) {
  const digests = anchorLoaderDigests({ repo, revision, declared });
  const models = {};
  for (const model of Object.keys(declared).sort()) {
    const entry = digests.get(model);
    models[model] = {
      entryPoints: [...entry.entryPoints].sort(),
      digest: entry.digest,
      closureFileCount: entry.files.length,
      closureFiles: entry.files,
    };
  }
  return {
    _comment:
      "Per-model loader-closure digests: the memory anchor currency key (sc-22511). An anchor is " +
      "CURRENT while the digest it recorded still equals the digest here for its (model, backend " +
      "lane). The unit is the source files the model's loader reaches, and nothing else — not the " +
      "pin, not sibling models, not shared crates the loader never reaches. Regenerate when the " +
      "pinned inference revision changes: node scripts/anchor-loader-closure.mjs --repo " +
      "<inference> --write. A regeneration that leaves the digests unchanged is the expected case.",
    digestVersion: ANCHOR_LOADER_CLOSURE_VERSION,
    inferenceRevision: revision,
    models,
  };
}

function usage() {
  return [
    "usage: node scripts/anchor-loader-closure.mjs --repo <inference-checkout> [options]",
    "",
    "  --repo PATH        an inference clone containing the pinned revision (required)",
    "  --revision SHA     revision to digest (default: the pin in SceneWorks' Cargo.toml)",
    "  --write            rewrite config/anchor-loader-closures.json",
    "  --check            fail if the checked-in config does not match the derivation",
    "  --model ID         print one model's canonical closure text and exit",
    "",
    "With neither --write nor --check it prints the derived digests.",
  ].join("\n");
}

export async function main(argv = process.argv.slice(2)) {
  const { readFile, writeFile } = await import("node:fs/promises");
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const value = (flag) => {
    const index = argv.indexOf(flag);
    return index === -1 ? undefined : argv[index + 1];
  };

  const repo = value("--repo");
  if (!repo) {
    console.error(usage());
    return 2;
  }

  const configPath = path.join(root, ANCHOR_LOADER_CONFIG_PATH);
  const existing = JSON.parse(await readFile(configPath, "utf8"));
  const declared = Object.fromEntries(
    Object.entries(existing.models).map(([model, entry]) => [
      model,
      { entryPoints: entry.entryPoints },
    ]),
  );
  const pinned = inferencePinFromCargo(await readFile(path.join(root, "Cargo.toml"), "utf8"));
  const revision = resolveRevision(path.resolve(repo), value("--revision") ?? pinned);

  const one = value("--model");
  if (one) {
    if (!declared[one]) throw new Error(`model "${one}" is not declared in ${ANCHOR_LOADER_CONFIG_PATH}`);
    const digests = anchorLoaderDigests({
      repo: path.resolve(repo),
      revision,
      declared: { [one]: declared[one] },
    });
    process.stdout.write(digests.get(one).text);
    return 0;
  }

  const built = buildAnchorLoaderConfig({ repo: path.resolve(repo), revision, declared });
  const body = `${JSON.stringify(built, null, 2)}\n`;

  if (argv.includes("--write")) {
    await writeFile(configPath, body);
    console.log(`wrote ${ANCHOR_LOADER_CONFIG_PATH} at ${revision.slice(0, 8)}`);
    return 0;
  }
  if (argv.includes("--check")) {
    const current = `${JSON.stringify(existing, null, 2)}\n`;
    if (current !== body) {
      console.error(
        `${ANCHOR_LOADER_CONFIG_PATH} does not match the derivation at ${revision.slice(0, 8)}. ` +
          "Re-run with --write.",
      );
      return 1;
    }
    console.log(`${ANCHOR_LOADER_CONFIG_PATH} matches ${revision.slice(0, 8)}`);
    return 0;
  }

  for (const [model, entry] of Object.entries(built.models)) {
    console.log(`${model.padEnd(20)} ${entry.digest.slice(0, 16)}  files=${entry.closureFileCount}`);
  }
  return 0;
}

// See the tail of `inference-closure-digest.mjs` for why this is `fileURLToPath` and not
// `new URL(...).pathname`.
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = await main();
}
