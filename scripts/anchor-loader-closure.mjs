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
//     a reached file, transitively. A leading `super` RUN is consumed whole, so
//     `super::super::StepCoeffs` resolves two module levels up rather than dropping the edge;
//   * every `mod x;` a reached file DECLARES, when that file was admitted as a whole module. Rust
//     modules are reachable by method syntax with no named path anywhere, so an impl-only module is
//     invisible to a path-driven walk; admitting declarations closes that hole. It over-includes a
//     declared-but-uncalled module, and that over-inclusion stays inside a crate the loader already
//     reached, so sibling-model separation is untouched. `#[path = "…"]` overrides are honoured;
//   * every FIRST-PARTY CRATE a reached file names in a path (`mlx_gen::array::contiguous`,
//     `use gen_core::…`), entered at the module the path names and walked the same way — so a shared
//     crate the loader genuinely reaches IS in the unit, and one it does not is not;
//   * a `pub use` re-export in a reached file, followed for the identifiers something actually asked
//     that file for — under the name the re-export PUBLISHES, so an aliased re-export
//     (`pub use vocoder::{Generator as VocoderGenerator}`) is followed under its alias. This is how
//     an item imported through a crate root's routing table is still attributed to the module that
//     defines it, without the table dragging its whole crate along;
//   * `include!` / `include_str!` / `include_bytes!` targets, whose CONTENT is a compile input (they
//     are hashed, never walked as code — an included file that happens to be Rust is data here,
//     because its text is parsed in the INCLUDING file's module scope, not the target's).
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
//     (an unresolvable head is followed if it names any in-scope module or first-party crate, and a
//     whole-module file contributes every `mod` it declares) so the residue is small, but it is a
//     real under-inclusion and not a proven-empty one.
//   * A non-literal include target. `include!(concat!(env!("CARGO_MANIFEST_DIR"), "/…"))` — the
//     spelling `mlx-gen-sam3/src/geometry.rs` uses — has no string literal to resolve at this level,
//     so its content is not hashed. Every such site in the pinned tree today is inside
//     `#[cfg(test)]`, which is stripped before any of this runs, so the residue is currently empty;
//     a shipped one would be a real under-inclusion.
//   * Inline `mod x { … }` bodies as separate units. They are already part of the file that
//     declares them, so they are hashed with it; only `mod x;` FILE declarations resolve to a path.
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
export const ANCHOR_LOADER_CLOSURE_VERSION = "anchor-loader-closure v2";

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
 * The attribute match is BOUNDED — `test)` or `all(test`, never a bare `test` prefix. Without the
 * boundary a `#[cfg(test_only_api)]` item (or any other cfg whose name merely starts with `test`)
 * would be read as a test item and its shipped code silently dropped from the hash: a false green
 * in the direction that matters, since the dropped code is loader code.
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
    if (!/^\s*#\[cfg\((?:test\s*\)|all\(\s*test\b)/.test(line)) {
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
 * Expand one `use` path, brace groups and all, into `{ segments, alias }` edges.
 *
 * `use a::b::{c, d::{e, f}, g as h};` is four distinct edges, and a walk that read the line with a
 * naive `::` split would see `d::e` as a top-level path and resolve it against the WRONG module.
 *
 * THE ALIAS IS CARRIED, NOT DROPPED, and that is load-bearing for `pub use`. A re-export is followed
 * only for identifiers something actually asked the file for, and what a consumer asks for is the
 * name the re-export PUBLISHES — the alias, when there is one. `mlx-gen-ltx/src/lib.rs` really does
 * write `pub use vocoder::{Generator as VocoderGenerator, …}`; gating on the source segment
 * (`Generator`) never matches a request for `VocoderGenerator`, so a module reachable only under its
 * alias would drop out of the closure and edits there would leave anchors reading "current".
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
      const raw = part.trim();
      if (!raw) continue;
      const aliased = raw.match(/\s+as\s+([A-Za-z_][A-Za-z0-9_]*)$/);
      const trimmed = aliased ? raw.slice(0, aliased.index) : raw;
      const alias = aliased ? aliased[1] : null;
      if (!trimmed) continue;
      const brace = trimmed.indexOf("{");
      if (brace === -1) {
        const segments = [...prefix, ...trimmed.split("::")].map((s) => s.trim()).filter(Boolean);
        if (segments.length) out.push({ segments, alias });
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
 *
 * `mod x;` declarations are reported separately as `mods`, carrying any `#[path = "…"]` override
 * that precedes them. A declaration is not a reference — the walk admits them only for a file it
 * already holds as a WHOLE module (see `loaderClosureFiles`), which is what makes impl-only modules
 * (reached by method syntax and never by a named path) visible at all.
 */
export function referencesIn(rawBody) {
  const body = stripCfgTest(rawBody);
  const code = [];
  const imports = [];
  const reexports = [];
  const includes = [];
  const mods = [];
  const lines = body.split("\n");
  /** The most recent `#[path = "…"]` attribute, consumed by the next `mod` declaration. */
  let pathOverride = null;
  for (let index = 0; index < lines.length; index += 1) {
    const raw = lines[index];
    const line = raw.trim();
    if (line.startsWith("//")) continue;
    // A trailing `// …` is not part of either declaration, and `mod x; // note` is legal Rust.
    const bare = line.replace(/\s*\/\/.*$/, "").trim();
    const pathAttr = bare.match(/^#\[\s*path\s*=\s*"([^"]+)"\s*\]$/);
    if (pathAttr) {
      pathOverride = pathAttr[1];
      continue;
    }
    // `mod x;` — a file declaration. `mod x { … }` is inline and needs no file.
    const modDecl = bare.match(/^(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+([a-z_][A-Za-z0-9_]*)\s*;$/);
    if (modDecl) {
      mods.push({ name: modDecl[1], path: pathOverride });
      pathOverride = null;
      continue;
    }
    if (bare && !bare.startsWith("#[")) pathOverride = null;
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
        for (const edge of paths) reexports.push(edge);
      } else {
        for (const edge of paths) imports.push(edge.segments);
      }
      continue;
    }
    for (const match of raw.matchAll(/\b([A-Za-z_][A-Za-z0-9_]*)((?:::[A-Za-z_][A-Za-z0-9_]*)+)/g)) {
      code.push([match[1], ...match[2].split("::").slice(1)]);
    }
    // `include!` is hashed exactly like `include_str!`/`include_bytes!`: all three splice a file's
    // CONTENT into this compilation unit, so all three are compile inputs. `include!` differs only
    // in that what it splices is Rust — which is why it is still DATA here and never walked: the
    // included text is parsed in this file's module scope, not the target's, so following it as a
    // module would resolve its paths against the wrong directory.
    for (const match of raw.matchAll(/\binclude(?:_str|_bytes)?!\s*\(\s*"([^"]+)"/g)) {
      includes.push(match[1]);
    }
  }
  return { code, imports, reexports, includes, mods };
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
      const { code, imports, reexports, includes, mods } = referencesIn(bodies.get(file) ?? "");

      const follow = (segments) => {
        const [head, ...rest] = segments;
        if (head === "crate") return resolveFrom(crateRoot, rest, libRs);
        if (head === "self") return resolveFrom(dir, rest, file);
        if (head === "super") {
          // Consume the WHOLE leading `super` run, one parent directory each. Handling only the
          // first `super` resolved `super::super::X` against this file's parent module and dropped
          // the edge when nothing there matched — `gen-core/src/sampling/unified.rs` names
          // `super::super::StepCoeffs` for real.
          let up = parentDir;
          let remainder = rest;
          while (remainder[0] === "super") {
            up = path.posix.dirname(up);
            remainder = remainder.slice(1);
          }
          return resolveFrom(up, remainder, null);
        }
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
      // A file held as a WHOLE module owns every module it declares, including the impl-only ones
      // that no path ever names (reached by method syntax on a type defined elsewhere). Those are
      // invisible to a path-driven walk, so they are admitted from the declaration itself. This
      // over-includes — a declared module nothing calls is still hashed — but the over-inclusion
      // stays INSIDE crates the loader already reached, so sibling-model separation is untouched.
      if (wanted.has("*")) {
        for (const decl of mods) {
          const target = decl.path
            ? path.posix.normalize(path.posix.join(path.posix.dirname(file), decl.path))
            : moduleFile(tree, dir, decl.name);
          if (target && tree.has(target)) request(target, "*");
        }
      }
      for (const edge of reexports) {
        // The name a consumer can ask for is the one the re-export PUBLISHES: the alias when there
        // is one, the last source segment otherwise.
        const exported = edge.alias ?? edge.segments[edge.segments.length - 1];
        if (wanted.has("*") || wanted.has(exported) || exported === "*") follow(edge.segments);
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
export function loaderClosureText({ model, engineId, entryPoints, files }) {
  return `${[
    `# ${ANCHOR_LOADER_CLOSURE_VERSION}`,
    `# model: ${model}`,
    // Present only for a catalog alias, so every declaration without one hashes exactly as before.
    ...(engineId ? [`# engine: ${engineId}`] : []),
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
export function loaderClosureDigest({ model, engineId, entryPoints, tree, crates }) {
  const files = loaderClosureFiles({ tree, entryPoints, crates });
  const bodies = tree.read(files);
  const hashed = files.map((file) => [
    file,
    loaderFileHash(file, tree.contentId(file), bodies.get(file)),
  ]);
  const text = loaderClosureText({ model, engineId, entryPoints, files: hashed });
  return { model, engineId, entryPoints, files, digest: sha256(text), text };
}

/**
 * A declaration is only worth its entry pointer, so check it rather than trust it: the model id must
 * appear as a string literal in the declared entry points' own source. An entry point that never
 * names the model digests some other model's code path and reports currency for a loader it never
 * looked at — a false green, and the expensive kind.
 *
 * A CATALOG ALIAS names the engine id it resolves to instead (`engineId`, sc-22724): `z_image_edit`
 * is a SceneWorks-side id for the `z_image_turbo` provider driven in `edit_image` mode
 * (`crates/sceneworks-worker/src/engines.rs`), and the inference tree carries no such literal. The
 * alias is EXPLICIT in the declaration — the literal rule is then asked of the engine id, and the
 * engine id becomes part of the hashed closure text — so an alias can never be a silent way around
 * the check: a declaration that names neither its own id nor a real engine id is still refused.
 */
export function assertModelIsNamedByEntryPoints({ model, engineId, entryPoints, tree }) {
  const modelId = model.split(":")[0];
  if (engineId !== undefined && (typeof engineId !== "string" || !/^[a-z][a-z0-9_]*$/.test(engineId))) {
    throw new Error(`anchor loader declaration "${model}" carries a malformed engineId ${JSON.stringify(engineId)}`);
  }
  if (engineId === modelId) {
    throw new Error(
      `anchor loader declaration "${model}" declares engineId "${engineId}", which is its own model id — ` +
        "engineId is for a catalog alias only; drop it",
    );
  }
  const id = engineId ?? modelId;
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
    const { entryPoints, engineId } = entry;
    assertModelIsNamedByEntryPoints({ model, engineId, entryPoints, tree: resolved });
    out.set(model, loaderClosureDigest({ model, engineId, entryPoints, tree: resolved, crates }));
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
      ...(entry.engineId ? { engineId: entry.engineId } : {}),
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
      "<inference> --write. A regeneration that leaves the digests unchanged is the expected case. " +
      "A catalog alias (a model id the inference tree never names, such as z_image_edit) declares " +
      "the engine id it resolves to as engineId; the alias is part of its hashed closure text.",
    digestVersion: ANCHOR_LOADER_CLOSURE_VERSION,
    inferenceRevision: revision,
    models,
  };
}

export const ANCHOR_STORE_PATH = "config/memory-anchors.json";

/**
 * Every packaged anchor's currency key, derived AT THE REVISION THAT ANCHOR WAS MEASURED AT.
 *
 * This is the half that makes currency mean anything, and it is the half that is easy to get
 * silently wrong. An anchor's `source.loaderClosureDigest` records what the model's loader looked
 * like WHEN THE MEASUREMENT WAS TAKEN. Currency is then the comparison of that recorded value
 * against the digest at the CURRENT pin. Stamping the current pin's digest instead would make every
 * anchor eternally current — the comparison would be a value against itself, and the whole key
 * would report "fine" through any loader change at all.
 *
 * So each anchor is digested at its own record's `repositories.inference.revision`, read out of the
 * retained evidence the anchor already cites. Seven distinct revisions across the packaged store
 * today; a clone must carry all of them.
 */
export function anchorMeasurementRevision(anchor, corpus) {
  const record = (corpus.records ?? []).find((entry) => entry.id === anchor.source?.recordId);
  const revision = record?.repositories?.inference?.revision;
  if (typeof revision !== "string" || !/^[0-9a-f]{40}$/.test(revision)) {
    throw new Error(
      `anchor ${anchor.id} cites record ${anchor.source?.recordId} in ${anchor.source?.path}, ` +
        "which declares no inference revision — its currency key cannot be derived",
    );
  }
  return revision;
}

export const ANCHOR_CURRENCY_ATTESTATIONS_PATH = "config/anchor-currency-attestations.json";

const REVISION_RE = /^[0-9a-f]{40}$/;

/** The attestation fields the store carries — the file list stays in the config and its doc. */
const STORE_ATTESTATION_FIELDS = [
  "measuredRevision",
  "attestedRevision",
  "attestedAt",
  "story",
  "class",
  "why",
  "witness",
];

/**
 * `config/anchor-currency-attestations.json`, indexed by anchor id, every entry shape-checked.
 *
 * WHAT AN ATTESTATION IS, AND WHAT IT IS NOT (sc-22667). The currency key is derived at the
 * measurement revision, so a pin bump that touches a model's loader closure stales its anchors.
 * That is the right default, and it is also only HALF of this repository's invalidation doctrine:
 * a differing closure digest says the loader's SOURCE moved, not that its memory behaviour did.
 * An attestation is the second half written down — a reviewed statement that the diff from the
 * anchor's measurement revision to one named later revision was read file by file and is
 * accounting-only (contract pricing, byte walkers, facts, tests), or that a re-measure on the
 * same hardware witnessed the behaviour unchanged across it. With one on file, the key is derived
 * at `attestedRevision` instead, and the attestation itself is copied into the store so the
 * matrix reports the anchor as current BY ATTESTATION rather than by measurement.
 *
 * It is bounded on both ends. `measuredRevision` must still equal what the cited record says, so
 * a re-capture that moves the record invalidates the entry loudly rather than re-keying a new
 * measurement at an old justification; and `attestedRevision` is one revision, so the next pin
 * bump that moves the closure past it stales the anchor again — an attestation is never "current
 * from now on". What it must never be is a way past a load-or-device-path change with no witness;
 * that is precisely the false green the key exists to prevent, and the reviewed narrative for
 * every entry lives in docs/calibration/sc-22657/anchor-currency-attestation-sc-22667.md.
 */
export function indexCurrencyAttestations(config) {
  const out = new Map();
  for (const entry of config?.attestations ?? []) {
    if (typeof entry?.anchorId !== "string" || entry.anchorId.length === 0) {
      throw new Error(`${ANCHOR_CURRENCY_ATTESTATIONS_PATH}: an attestation names no anchorId`);
    }
    if (out.has(entry.anchorId)) {
      throw new Error(`${ANCHOR_CURRENCY_ATTESTATIONS_PATH}: ${entry.anchorId} is attested twice`);
    }
    for (const field of ["measuredRevision", "attestedRevision"]) {
      if (!REVISION_RE.test(entry[field] ?? "")) {
        throw new Error(
          `${ANCHOR_CURRENCY_ATTESTATIONS_PATH}: ${entry.anchorId} ${field} is not a 40-hex revision`,
        );
      }
    }
    if (entry.measuredRevision === entry.attestedRevision) {
      throw new Error(
        `${ANCHOR_CURRENCY_ATTESTATIONS_PATH}: ${entry.anchorId} attests its own measurement ` +
          "revision — an anchor measured at the revision it is keyed to needs no attestation",
      );
    }
    for (const field of ["attestedAt", "story", "class", "why", "witness"]) {
      if (typeof entry[field] !== "string" || entry[field].trim().length === 0) {
        throw new Error(
          `${ANCHOR_CURRENCY_ATTESTATIONS_PATH}: ${entry.anchorId} states no ${field} — an ` +
            "attestation without its justification is a re-stamp, which is what this file forbids",
        );
      }
    }
    if (
      !Array.isArray(entry.filesChangedSinceMeasurement) ||
      entry.filesChangedSinceMeasurement.some(
        (file) => typeof file?.path !== "string" || typeof file?.class !== "string",
      )
    ) {
      throw new Error(
        `${ANCHOR_CURRENCY_ATTESTATIONS_PATH}: ${entry.anchorId} must list every closure file ` +
          "changed between measuredRevision and attestedRevision with its class",
      );
    }
    out.set(entry.anchorId, entry);
  }
  return out;
}

/**
 * The revision one anchor's currency key is derived at: its attestation's `attestedRevision`
 * when one is on file (and still describes the record — its `measuredRevision` must equal the
 * record's), else the measurement revision itself.
 */
export function anchorCurrencyRevision(anchor, corpus, attestations = new Map()) {
  const measured = anchorMeasurementRevision(anchor, corpus);
  const attestation = attestations.get(anchor.id);
  if (!attestation) return { revision: measured, attestation: null };
  if (attestation.measuredRevision !== measured) {
    throw new Error(
      `anchor ${anchor.id} is attested from ${attestation.measuredRevision.slice(0, 8)} but its ` +
        `record was measured at ${measured.slice(0, 8)} — the measurement moved under the ` +
        `attestation; re-read the diff and rewrite the entry in ${ANCHOR_CURRENCY_ATTESTATIONS_PATH}, ` +
        "or delete it",
    );
  }
  return { revision: attestation.attestedRevision, attestation };
}

/** The attestation as the store carries it: the justification, not the file list. */
function storeAttestation(attestation) {
  return Object.fromEntries(STORE_ATTESTATION_FIELDS.map((field) => [field, attestation[field]]));
}

/**
 * `store` with every anchor's `source.loaderClosureDigest` re-derived at its own measurement
 * revision — or, for an anchor with a currency attestation on file, at the attested revision,
 * with the attestation copied into `source.currencyAttestation` (and that field dropped from any
 * anchor no longer attested). Returns the new store and a per-anchor report.
 */
export function stampAnchorStore({ repo, store, declared, corpora, attestations = new Map() }) {
  const byRevision = new Map();
  for (const anchor of store.anchors) {
    const corpus = corpora.get(anchor.source?.path);
    if (!corpus) {
      throw new Error(`anchor ${anchor.id} cites ${anchor.source?.path}, which was not read`);
    }
    const { revision } = anchorCurrencyRevision(anchor, corpus, attestations);
    const model = `${anchor.modelId}:${anchor.backend}`;
    if (!declared[model]) {
      throw new Error(
        `anchor ${anchor.id} has no loader-closure declaration for ${model} in ` +
          `${ANCHOR_LOADER_CONFIG_PATH}`,
      );
    }
    const bucket = byRevision.get(revision) ?? new Set();
    bucket.add(model);
    byRevision.set(revision, bucket);
  }
  // One tree read per revision, shared by every model measured at it.
  const digests = new Map();
  for (const [revision, models] of byRevision) {
    const tree = gitTree(repo, revision);
    // AN ENTRY POINT THAT DID NOT EXIST YET IS DROPPED, NOT AN ERROR. A historical revision can
    // predate a file today's declaration names — `mlx-gen-ltx/src/memory_strategy.rs` postdates the
    // LTX-2.3 capture — and that IS the difference the key should report: the entry-point list is
    // part of the hashed text, so a closure derived over a smaller list cannot equal the pin's, and
    // the anchor reads not-current. Which is the truth about it.
    const historical = Object.fromEntries(
      [...models].map((model) => [
        model,
        {
          ...(declared[model].engineId ? { engineId: declared[model].engineId } : {}),
          entryPoints: declared[model].entryPoints.filter((file) => tree.has(file)),
        },
      ]),
    );
    for (const [model, entry] of Object.entries(historical)) {
      if (entry.entryPoints.length === 0) {
        throw new Error(
          `no declared entry point of ${model} exists at ${revision.slice(0, 8)} — its anchors ` +
            "cannot be keyed to that measurement at all",
        );
      }
    }
    const perModel = anchorLoaderDigests({ repo, revision, declared: historical, tree });
    for (const [model, entry] of perModel) digests.set(`${revision}|${model}`, entry.digest);
  }
  const report = [];
  const anchors = store.anchors.map((anchor) => {
    const { revision, attestation } = anchorCurrencyRevision(
      anchor,
      corpora.get(anchor.source.path),
      attestations,
    );
    const digest = digests.get(`${revision}|${anchor.modelId}:${anchor.backend}`);
    const { currencyAttestation: _previous, ...source } = anchor.source;
    const stamped = attestation
      ? { ...source, loaderClosureDigest: digest, currencyAttestation: storeAttestation(attestation) }
      : { ...source, loaderClosureDigest: digest };
    report.push({
      id: anchor.id,
      revision,
      digest,
      attested: Boolean(attestation),
      changed: JSON.stringify(stamped) !== JSON.stringify(anchor.source),
    });
    return { ...anchor, source: stamped };
  });
  return { store: { ...store, anchors }, report };
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
    "  --stamp-anchors    re-derive every packaged anchor's currency key AT ITS OWN measurement",
    "                     revision — or at the revision a reviewed currency attestation in",
    "                     config/anchor-currency-attestations.json names for it — and write it",
    "                     into config/memory-anchors.json (add --check to verify instead of",
    "                     write). The clone must carry every cited revision.",
    "  --anchor-revisions print every revision the packaged anchors were measured at, plus every",
    "                     attested revision (needs no --repo); this is the fetch list a shallow",
    "                     CI clone must satisfy",
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

  // Every revision the packaged anchors were measured at, one per line. Needs no clone: it is a
  // read of the store and the corpora it cites, which is what makes it usable as the fetch list in
  // CI. Deriving the list from the same store the stamping walks is what keeps the two from
  // drifting into "CI fetched four of the seven revisions and the fifth silently skipped".
  if (argv.includes("--anchor-revisions")) {
    const store = JSON.parse(await readFile(path.join(root, ANCHOR_STORE_PATH), "utf8"));
    const attestations = indexCurrencyAttestations(
      JSON.parse(await readFile(path.join(root, ANCHOR_CURRENCY_ATTESTATIONS_PATH), "utf8")),
    );
    const corpora = new Map();
    const revisions = new Set();
    for (const anchor of store.anchors) {
      const cited = anchor.source?.path;
      if (cited && !corpora.has(cited)) {
        corpora.set(cited, JSON.parse(await readFile(path.join(root, cited), "utf8")));
      }
      // Both the measurement revision and, where attested, the attested one: the attestation
      // check re-reads the record's revision, and the stamp derives at the attested one.
      revisions.add(anchorMeasurementRevision(anchor, corpora.get(cited)));
      revisions.add(anchorCurrencyRevision(anchor, corpora.get(cited), attestations).revision);
    }
    process.stdout.write(`${[...revisions].sort().join("\n")}\n`);
    return 0;
  }

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
      {
        ...(entry.engineId ? { engineId: entry.engineId } : {}),
        entryPoints: entry.entryPoints,
      },
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

  if (argv.includes("--stamp-anchors")) {
    const storePath = path.join(root, ANCHOR_STORE_PATH);
    const store = JSON.parse(await readFile(storePath, "utf8"));
    const corpora = new Map();
    for (const anchor of store.anchors) {
      const cited = anchor.source?.path;
      if (cited && !corpora.has(cited)) {
        corpora.set(cited, JSON.parse(await readFile(path.join(root, cited), "utf8")));
      }
    }
    const attestations = indexCurrencyAttestations(
      JSON.parse(await readFile(path.join(root, ANCHOR_CURRENCY_ATTESTATIONS_PATH), "utf8")),
    );
    const { store: stamped, report } = stampAnchorStore({
      repo: path.resolve(repo),
      store,
      declared,
      corpora,
      attestations,
    });
    for (const row of report) {
      console.log(
        `${row.changed ? "*" : " "} ${row.revision.slice(0, 8)} ${row.digest.slice(0, 16)} ` +
          `${row.attested ? "attested " : ""}${row.id}`,
      );
    }
    const body = `${JSON.stringify(stamped, null, 2)}\n`;
    if (argv.includes("--check")) {
      if (body !== `${JSON.stringify(store, null, 2)}\n`) {
        console.error(
          `${ANCHOR_STORE_PATH} carries currency keys that are not the derivation at each anchor's ` +
            "own measurement revision. Re-run with --stamp-anchors.",
        );
        return 1;
      }
      console.log(`${ANCHOR_STORE_PATH} currency keys match their measurement revisions`);
      return 0;
    }
    await writeFile(storePath, body);
    console.log(`stamped ${report.length} anchors in ${ANCHOR_STORE_PATH}`);
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
