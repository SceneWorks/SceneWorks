// Per-provider compile-closure digests over the pinned inference source (sc-17774).
//
// WHY THIS EXISTS
//
// Calibration currency used to be keyed on the inference PIN: `record.repositories.inference
// .revision === <the Cargo pin>`. The unit of invalidation was therefore the whole inference
// repository at commit granularity, so a commit to `mlx-gen-z-image` demoted `flux2_dev`'s measured
// ladder, and a documentation-only commit anywhere demoted all of them. Measured over the 90 days
// up to `fbb00d6b`: 2812 non-merge commits, every one of which demoted every calibrated provider.
//
// A calibration measures the memory behaviour of ONE provider's code path. The honest unit is
// therefore that code path, and nothing else. This module derives it:
//
//   provider -> its inference crate -> that crate's transitive first-party path-dependency closure
//            -> the source blobs of those crates
//            + the locked external packages that closure actually reaches
//            + the workspace inputs that change codegen for everything (`[profile]`, `[patch]`,
//              `rust-toolchain.toml`, `.cargo/config.toml`)
//
// Same 90-day window, same providers, under this unit: ~9-10.5% of commits instead of 100%.
//
// NO BUILD, AND DERIVED OFFLINE — BUT VERIFIED IN CI
//
// Everything here reads `git ls-tree` / `git show` / `git cat-file` at a revision, so it runs in
// seconds and needs no toolchain, no GPU and no weights. It does need an inference clone, which a
// SceneWorks checkout does not contain — so digests are derived at pin-bump time and checked into
// `config/inference-provider-closures.json`, where a reviewer sees the change in the diff instead of
// it being conjured at check time. The consumers compare against that file and hard-error when its
// recorded revision is not the live Cargo pin; a stale config is never a silent pass.
//
// Checked in is NOT the same as unverified, and the distinction is load-bearing. An earlier draft of
// this comment said CI "does not have" an inference clone, citing the framing at
// `check-license-coverage.mjs:481`. That source says something narrower — inference's sources are not
// IN a SceneWorks checkout — and the extrapolation was wrong: `SceneWorks/inference` is PUBLIC, and a
// `--depth=1` fetch of the pinned revision succeeds with no token, no credential helper and no HOME.
// (#2164 established this after the opposite premise had gone unchallenged long enough to put a
// same-repo guard on a required CI check, turning it into a vacuous green.) So `check.yml` re-derives
// every lane's digest against real pinned source on each run — without it the currency term would be
// checked-in data that nothing grades, which is the same fail-open shape this epic exists to remove.
//
// WHAT THIS UNIT DOES NOT SEE — read before trusting it
//
//   - Feature unification. Cargo.lock records versions, not resolved features, so a crate OUTSIDE
//     this closure enabling a feature on a dependency INSIDE it moves no byte here. That gap is
//     owned by the resolved-feature witness (sc-17606) and is NOT closed by this module. Narrowing
//     the unit makes that witness strictly more load-bearing, exactly as sc-17776 predicted.
//   - Crate-mates. Providers sharing a crate share its digest: a `qwen_image_edit` edit moves
//     `qwen_image`. That is deliberate. Rust compiles a crate as one unit with cross-module
//     inlining, so file-level scoping inside a crate would be a false green, and sc-17776 measured
//     that only a behaviour witness can absolve this class soundly. The over-trigger is real and
//     narrowed from "the whole repository" to "one crate plus what it links".
//   - Realised VRAM behaviour. This is a code-identity unit. It answers "did the code the
//     measurements ran change?", never "did the measured quantity change?".

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import path from "node:path";

import { canonicalSourceText, stripInertLines } from "./lib/source-revision.mjs";

export const CLOSURE_DIGEST_VERSION = "inference-closure-digest v2";

/**
 * Languages whose whole-line comments are stripped before hashing.
 *
 * This is the one capability the deleted `flux2_dev` artifact audit had that a raw content hash does
 * not: absolving a documentation-only edit. That audit bought it by linking a CUDA binary at two
 * revisions on one Windows box — for one provider. The same result is available to EVERY provider
 * for free by hashing semantic source instead of raw bytes, which is why the audit could be dropped
 * without losing ground. The measured motivation is real: sc-17775 §5.4 found the vendored
 * `candle-kernels` tree moved three times in 60 days, all documentation, and the original flux2
 * window turned on a 42-line `//!` block that provably changed nothing that compiles.
 *
 * `scripts/lib/source-revision.mjs` owns the stripper and the quote-guard safety argument; this
 * reuses it rather than restating it. Anything not listed here — `.metal`, `.cu`, `.json`, embedded
 * fixtures — is hashed as raw bytes, because stripping a language this module does not understand
 * is how a real change becomes invisible.
 */
const LINE_COMMENT_MARKER = Object.freeze({ ".rs": "//", ".toml": "#" });

/** Workspace files that change codegen for every crate, so they belong in every closure. */
const WORKSPACE_INPUTS = ["rust-toolchain.toml", ".cargo/config.toml"];

function git(repo, args, { maxBuffer = 64 * 1024 * 1024 } = {}) {
  return execFileSync("git", ["-C", repo, ...args], { encoding: "utf8", maxBuffer });
}

export function sha256(body) {
  return createHash("sha256").update(body).digest("hex");
}

/**
 * `path -> blob oid` for a whole revision. Git blob oids ARE content hashes, so the tree listing
 * alone is enough to digest file content without reading a single file body.
 */
export function treeBlobs(repo, revision, { runGit = git } = {}) {
  const listing = runGit(repo, ["ls-tree", "-r", revision]);
  const blobs = new Map();
  for (const line of listing.split("\n")) {
    if (!line) continue;
    // `<mode> <type> <oid>\t<path>`
    const match = line.match(/^\d+ (\w+) ([0-9a-f]+)\t(.*)$/);
    if (!match) throw new Error(`unparsable ls-tree line: ${line.slice(0, 120)}`);
    if (match[1] !== "blob") continue;
    blobs.set(match[3], match[2]);
  }
  if (!blobs.size) throw new Error(`revision ${revision} listed no blobs`);
  return blobs;
}

function readFileAt(repo, revision, file, runGit) {
  return runGit(repo, ["show", `${revision}:${file}`]);
}

/**
 * The content hash for one closure file: semantic for languages the stripper understands, raw blob
 * identity for everything else.
 *
 * Returning the blob oid unchanged for non-strippable files is deliberate — it is already a content
 * hash, so an unrecognised file type is hashed by its exact bytes rather than being skipped.
 */
export function fileContentHash(file, oid, body) {
  const marker = LINE_COMMENT_MARKER[path.extname(file)];
  if (!marker) return oid;
  return `s:${sha256(stripInertLines(canonicalSourceText(body), marker))}`;
}

/**
 * Read many blobs in ONE `git cat-file --batch`.
 *
 * A closure is 130-250 files and the config covers six providers across several revisions, so a
 * `git show` per file would be thousands of subprocesses. `--batch` streams them all through one.
 */
export function readBlobs(repo, oids, { runGit = git } = {}) {
  const unique = [...new Set(oids)];
  if (!unique.length) return new Map();
  const raw = execFileSync("git", ["-C", repo, "cat-file", "--batch"], {
    input: `${unique.join("\n")}\n`,
    encoding: "latin1",
    maxBuffer: 512 * 1024 * 1024,
  });
  const bodies = new Map();
  let offset = 0;
  while (offset < raw.length) {
    const newline = raw.indexOf("\n", offset);
    if (newline === -1) break;
    const header = raw.slice(offset, newline);
    const match = header.match(/^([0-9a-f]{40}) (\w+) (\d+)$/);
    if (!match) throw new Error(`unparsable cat-file header: ${header.slice(0, 80)}`);
    const size = Number(match[3]);
    const start = newline + 1;
    bodies.set(match[1], Buffer.from(raw.slice(start, start + size), "latin1").toString("utf8"));
    offset = start + size + 1;
  }
  if (bodies.size !== unique.length) {
    throw new Error(`cat-file returned ${bodies.size} blobs for ${unique.length} requests`);
  }
  return bodies;
}

/**
 * Split a TOML body into `[header]` sections. Good enough for the manifest fields this module
 * reads — dependency tables and `path =` keys — and deliberately not a general TOML parser.
 */
function tomlSections(body) {
  const sections = [];
  let current = { header: "", lines: [] };
  for (const line of body.split("\n")) {
    const header = line.match(/^\s*\[\[?([^\]]+)\]\]?\s*$/);
    if (header) {
      sections.push(current);
      current = { header: header[1].trim(), lines: [line] };
      continue;
    }
    current.lines.push(line);
  }
  sections.push(current);
  return sections;
}

/**
 * `dependency name -> repo-relative crate dir` for the root manifest's `[workspace.dependencies]`.
 *
 * Inference declares its first-party crates once at the root and consumes them as
 * `sceneworks-gen-core = { workspace = true }`. A walk that only reads inline `path = "…"` therefore
 * misses them entirely — and misses `gen-core`, which owns the memory-strategy contract itself. That
 * would be a false green of the worst kind: the shared contract could change under a provider while
 * its closure digest held still. `[package.metadata.legacy-workspace-dependencies]` in the crate
 * manifests is inert metadata that looks exactly like the real thing; it is deliberately not read.
 */
export function workspaceDependencyPaths(rootManifestBody) {
  const paths = new Map();
  for (const section of tomlSections(rootManifestBody)) {
    if (section.header !== "workspace.dependencies") continue;
    for (const line of section.lines) {
      const match = line.match(/^\s*([A-Za-z0-9_-]+)\s*=\s*\{[^}]*\bpath\s*=\s*"([^"]+)"/);
      if (match) paths.set(match[1], path.posix.normalize(match[2]));
    }
  }
  return paths;
}

/**
 * Sibling crate directories this manifest depends on, as repo-relative POSIX paths.
 *
 * `[dev-dependencies]` are excluded: they do not link into the shipped runtime the calibration
 * harness measures through. `[build-dependencies]` ARE included — a build script changes generated
 * code. `[target.'cfg(...)'.dependencies]` are included; the platform arms are cheap to over-include
 * and dropping them would be a false green on the lane that is not being built.
 */
export function manifestPathDependencies(body, crateDir, workspacePaths = new Map()) {
  const dirs = new Set();
  for (const section of tomlSections(body)) {
    const header = section.header;
    if (!/(^|\.)(dependencies|build-dependencies)$/.test(header)) continue;
    if (/(^|\.)dev-dependencies$/.test(header)) continue;
    for (const line of section.lines) {
      // Inline path dependency — relative to the declaring crate.
      const inline = line.match(/\bpath\s*=\s*"([^"]+)"/);
      if (inline) {
        dirs.add(path.posix.normalize(path.posix.join(crateDir, inline[1])));
        continue;
      }
      // `name = { workspace = true }` / `name.workspace = true` — resolved through the root map,
      // whose paths are already repo-relative.
      const inherited =
        line.match(/^\s*([A-Za-z0-9_-]+)\s*=\s*\{[^}]*\bworkspace\s*=\s*true/)?.[1] ??
        line.match(/^\s*([A-Za-z0-9_-]+)\.workspace\s*=\s*true/)?.[1];
      if (inherited && workspacePaths.has(inherited)) dirs.add(workspacePaths.get(inherited));
    }
  }
  return [...dirs].sort();
}

/**
 * Transitive first-party closure of `crateDir`, as sorted repo-relative crate directories.
 *
 * Cycles are impossible in cargo and would loop here; the `seen` set makes that moot either way.
 */
export function firstPartyClosure({ repo, revision, crateDir, blobs, workspacePaths, runGit = git }) {
  const seen = new Set();
  const names = new Map();
  const stack = [crateDir];
  while (stack.length) {
    const dir = stack.pop();
    if (seen.has(dir)) continue;
    const manifest = dir ? `${dir}/Cargo.toml` : "Cargo.toml";
    if (!blobs.has(manifest)) {
      throw new Error(`closure walk reached ${dir || "<root>"}, which has no Cargo.toml at ${revision}`);
    }
    const body = readFileAt(repo, revision, manifest, runGit);
    const name = body.match(/^\s*name\s*=\s*"([^"]+)"/m)?.[1];
    if (!name) throw new Error(`${manifest} at ${revision} declares no package name`);
    seen.add(dir);
    names.set(dir, name);
    for (const dependency of manifestPathDependencies(body, dir, workspacePaths)) stack.push(dependency);
  }
  const crates = [...seen].sort();
  return { crates, packageNames: crates.map((dir) => names.get(dir)) };
}

/**
 * Cargo.lock as `"<name> <version>" -> { name, version, checksum, dependencies }`.
 *
 * A lock dependency entry is `"name"`, `"name version"` or `"name version (source)"`; only the
 * leading name is needed to walk, and the version disambiguates the rare duplicate-name case.
 */
export function parseCargoLock(body) {
  const packages = new Map();
  const byName = new Map();
  for (const section of tomlSections(body)) {
    if (section.header !== "package") continue;
    const text = section.lines.join("\n");
    const name = text.match(/^\s*name\s*=\s*"([^"]+)"/m)?.[1];
    const version = text.match(/^\s*version\s*=\s*"([^"]+)"/m)?.[1];
    if (!name || !version) continue;
    const checksum = text.match(/^\s*checksum\s*=\s*"([^"]+)"/m)?.[1] ?? null;
    const source = text.match(/^\s*source\s*=\s*"([^"]+)"/m)?.[1] ?? null;
    const block = text.match(/^\s*dependencies\s*=\s*\[([\s\S]*?)\]/m)?.[1] ?? "";
    const dependencies = [...block.matchAll(/"([^"]+)"/g)].map((match) => match[1].split(" ")[0]);
    const entry = { name, version, checksum, source, dependencies };
    packages.set(`${name} ${version}`, entry);
    if (!byName.has(name)) byName.set(name, []);
    byName.get(name).push(entry);
  }
  if (!packages.size) throw new Error("Cargo.lock parsed to zero packages");
  return { packages, byName };
}

/**
 * The locked packages the closure actually reaches, walked from its first-party package names.
 *
 * This is what keeps `Cargo.lock` out of the digest as a whole file. The lock moves on essentially
 * every inference commit that touches any dependency anywhere; restricting it to the reached set
 * means an unrelated crate's dependency bump cannot demote this provider, while a bump the provider
 * genuinely compiles still does.
 */
export function lockedClosure(lock, packageNames) {
  const reached = new Map();
  const stack = [...packageNames];
  while (stack.length) {
    const name = stack.pop();
    for (const entry of lock.byName.get(name) ?? []) {
      const key = `${entry.name} ${entry.version}`;
      if (reached.has(key)) continue;
      reached.set(key, entry);
      for (const dependency of entry.dependencies) stack.push(dependency);
    }
  }
  return [...reached.values()].sort((a, b) =>
    a.name.localeCompare(b.name) || a.version.localeCompare(b.version),
  );
}

/**
 * The codegen-affecting slices of the workspace root manifest.
 *
 * `[profile.*]` changes optimisation and therefore realised memory; `[patch.*]` changes which source
 * is compiled at all. `[workspace.dependencies]` is deliberately NOT included — it only names
 * versions, which the locked closure already carries exactly, and including it would re-introduce a
 * whole-repo trigger through the back door.
 */
export function rootManifestSlices(body) {
  return tomlSections(body)
    .filter((section) => /^(profile|patch)(\.|$)/.test(section.header))
    .map((section) => `[${section.header}]\n${section.lines.slice(1).join("\n").trim()}`)
    .sort();
}

/**
 * The canonical closure text for one provider. The digest is `sha256` of this, and the text itself
 * is kept diffable on purpose: when a digest moves, the answer to "what moved?" has to be readable
 * without re-deriving anything.
 *
 * The REVISION IS DELIBERATELY ABSENT. It is provenance, not content, and hashing it would make
 * every digest move on every pin bump — reconstructing the exact pin-identity coupling this module
 * exists to remove. The first draft of this function did hash it, and the sc-17726 window silently
 * reported all six providers moving; the check that caught it is
 * `a closure digest is independent of the revision it was read at` in the test file.
 */
export function closureText({ provider, crateDir, crates, files, locked, rootSlices, workspace }) {
  const lines = [
    `# ${CLOSURE_DIGEST_VERSION}`,
    `# provider: ${provider}`,
    `# crate: ${crateDir}`,
    "[crates]",
    ...crates,
    "[source]",
    ...files.map(([file, oid]) => `${oid} ${file}`),
    "[workspace]",
    ...workspace.map(([file, oid]) => `${oid} ${file}`),
    "[root-manifest]",
    ...rootSlices,
    "[locked]",
    ...locked.map((entry) => `${entry.name} ${entry.version} ${entry.checksum ?? "-"}`),
  ];
  return `${lines.join("\n")}\n`;
}

/**
 * Compute one provider's closure digest at one revision.
 *
 * `crateDir` is the provider's inference crate, declared in
 * `config/inference-provider-closures.json`. Every source blob under each closure crate's `src/`
 * plus its `Cargo.toml` is digested; `tests/`, `benches/`, `examples/` and `testdata/` are outside
 * the shipped closure and excluded, which is the `#[cfg(test)]` class sc-17776 measured the shipped
 * artifact unit refusing on.
 */
export function providerClosureDigest({
  repo, revision, provider, crateDir, blobs, lock, rootManifest, readBodies, runGit = git,
}) {
  const resolvedBlobs = blobs ?? treeBlobs(repo, revision, { runGit });
  const resolvedRootManifest = rootManifest ?? readFileAt(repo, revision, "Cargo.toml", runGit);
  const { crates, packageNames } = firstPartyClosure({
    repo,
    revision,
    crateDir,
    blobs: resolvedBlobs,
    workspacePaths: workspaceDependencyPaths(resolvedRootManifest),
    runGit,
  });

  // A crate directory can be the PARENT of its siblings (`crates/media/mlx-gen` hosts
  // `mlx-gen-qwen-image`), so a bare prefix match would sweep unrelated crates into the closure and
  // resurrect the cross-model coupling this module exists to remove. Match each crate's own `src/`.
  const owned = (file) =>
    crates.some((crate) => file === `${crate}/Cargo.toml` || file.startsWith(`${crate}/src/`));
  const owned_files = [...resolvedBlobs].filter(([file]) => owned(file)).sort(([a], [b]) => a.localeCompare(b));
  if (!owned_files.length) throw new Error(`closure for ${provider} matched no source files at ${revision}`);
  const wanted = [...owned_files, ...WORKSPACE_INPUTS.map((file) => [file, resolvedBlobs.get(file)])]
    .filter(([file, oid]) => oid && LINE_COMMENT_MARKER[path.extname(file)])
    .map(([, oid]) => oid);
  const bodies = (readBodies ?? ((oids) => readBlobs(repo, oids)))(wanted);
  const files = owned_files.map(([file, oid]) => [file, fileContentHash(file, oid, bodies.get(oid))]);

  const workspace = WORKSPACE_INPUTS.filter((file) => resolvedBlobs.has(file))
    .map((file) => [file, fileContentHash(file, resolvedBlobs.get(file), bodies.get(resolvedBlobs.get(file)))]);
  if (workspace.length !== WORKSPACE_INPUTS.length) {
    throw new Error(
      `expected every workspace build input (${WORKSPACE_INPUTS.join(", ")}) at ${revision}; ` +
        `found ${workspace.map(([file]) => file).join(", ") || "none"}`,
    );
  }

  const resolvedLock = lock ?? parseCargoLock(readFileAt(repo, revision, "Cargo.lock", runGit));
  const locked = lockedClosure(resolvedLock, packageNames);
  const rootSlices = rootManifestSlices(resolvedRootManifest);

  const text = closureText({
    provider,
    crateDir,
    crates,
    files,
    locked,
    rootSlices,
    workspace,
  });
  return {
    provider,
    crate: crateDir,
    revision,
    digest: sha256(text),
    crates,
    sourceFileCount: files.length,
    lockedPackageCount: locked.length,
    text,
  };
}

/** Compute every declared provider's digest at one revision, sharing the per-revision reads. */
export function digestsAtRevision({ repo, revision, providers, readBodies, runGit = git }) {
  const blobs = treeBlobs(repo, revision, { runGit });
  const lock = parseCargoLock(readFileAt(repo, revision, "Cargo.lock", runGit));
  const rootManifest = readFileAt(repo, revision, "Cargo.toml", runGit);
  const out = new Map();
  for (const [provider, crateDir] of Object.entries(providers)) {
    out.set(
      provider,
      providerClosureDigest({
        repo, revision, provider, crateDir, blobs, lock, rootManifest, readBodies, runGit,
      }),
    );
  }
  return out;
}

export function resolveRevision(repo, revision, { runGit = git } = {}) {
  const resolved = runGit(repo, ["rev-parse", `${revision}^{commit}`]).trim();
  if (!/^[0-9a-f]{40}$/.test(resolved)) {
    throw new Error(`could not resolve ${revision} in ${repo}`);
  }
  return resolved;
}

/**
 * A provider declaration is only worth as much as its crate pointer, so check it rather than trust
 * it: the provider id must appear as a string literal in the declared crate's own source at the
 * revision. A declaration pointing at the wrong crate would otherwise digest an unrelated code path
 * and report currency for a provider it never looked at — a false green, and the expensive kind.
 */
export function assertProviderIsDeclaredInCrate({ repo, revision, provider, crateDir, runGit = git }) {
  let hits = "";
  try {
    hits = runGit(repo, ["grep", "-l", "-F", `"${provider}"`, revision, "--", `${crateDir}/src/`]);
  } catch {
    // `git grep` exits 1 on "no match", which execFileSync raises. Treat as the empty result and
    // let the check below produce the real message.
    hits = "";
  }
  if (!hits.trim()) {
    throw new Error(
      `provider "${provider}" is declared against ${crateDir}, but that crate's source at ` +
        `${revision.slice(0, 8)} never names it. Fix the declaration in ` +
        "config/inference-provider-closures.json — a wrong crate pointer digests the wrong code path.",
    );
  }
}

export const CLOSURE_CONFIG_PATH = "config/inference-provider-closures.json";

/** Build the checked-in config body for one revision. */
export function buildClosureConfig({ repo, revision, declared, runGit = git }) {
  // Keys are `<backend>:<provider>`. A provider id is NOT unique on its own: `krea_2_turbo_control`
  // exists on mlx (`mlx-gen-krea`) and on candle (`candle-gen-krea`) — two genuinely different code
  // paths. Keying on the bare id would have compared one backend's measurements against the other
  // backend's closure, which is the same class of error this whole change exists to remove.
  for (const [key, crateDir] of Object.entries(declared)) {
    const provider = key.split(":").slice(1).join(":");
    if (!provider) throw new Error(`closure declaration key "${key}" must be "<backend>:<provider>"`);
    assertProviderIsDeclaredInCrate({ repo, revision, provider, crateDir, runGit });
  }
  const digests = digestsAtRevision({ repo, revision, providers: declared, runGit });
  const providers = {};
  for (const provider of Object.keys(declared).sort()) {
    const entry = digests.get(provider);
    providers[provider] = {
      crate: entry.crate,
      digest: entry.digest,
      closureCrates: entry.crates,
      sourceFileCount: entry.sourceFileCount,
      lockedPackageCount: entry.lockedPackageCount,
    };
  }
  return {
    _comment:
      "Per-provider inference compile-closure digests (sc-17774). Calibration currency compares a " +
      "record's captured closureDigest against the entry here for the SAME provider, so a change to " +
      "one model's code path can no longer demote another's. Regenerate on every inference pin bump: " +
      "node scripts/inference-closure-digest.mjs --repo <inference> --write. Derivation, and what " +
      "this unit deliberately does NOT see, are documented in scripts/inference-closure-digest.mjs.",
    digestVersion: CLOSURE_DIGEST_VERSION,
    inferenceRevision: revision,
    providers,
  };
}

function usage() {
  return [
    "usage: node scripts/inference-closure-digest.mjs --repo <inference-checkout> [options]",
    "",
    "  --repo PATH        an inference clone containing the pinned revision (required)",
    "  --revision SHA     revision to digest (default: the pin in SceneWorks' Cargo.toml)",
    "  --write            rewrite config/inference-provider-closures.json",
    "  --check            fail if the checked-in config does not match the derivation",
    "  --provider ID      print one provider's canonical closure text and exit",
    "",
    "With neither --write nor --check it prints the derived digests.",
  ].join("\n");
}

export async function main(argv = process.argv.slice(2)) {
  const { readFile, writeFile } = await import("node:fs/promises");
  const { fileURLToPath } = await import("node:url");
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

  const configPath = path.join(root, CLOSURE_CONFIG_PATH);
  const existing = JSON.parse(await readFile(configPath, "utf8"));
  const declared = Object.fromEntries(
    Object.entries(existing.providers).map(([provider, entry]) => [provider, entry.crate]),
  );

  const cargo = await readFile(path.join(root, "Cargo.toml"), "utf8");
  const pinned = cargo.match(
    /candle-kernels\s*=\s*\{[^}]*?github\.com\/SceneWorks\/inference[^}]*?rev\s*=\s*"([0-9a-f]+)"/,
  )?.[1];
  if (!pinned) throw new Error("could not resolve the pinned SceneWorks/inference revision");

  const revision = resolveRevision(path.resolve(repo), value("--revision") ?? pinned);

  const one = value("--provider");
  if (one) {
    if (!declared[one]) throw new Error(`provider "${one}" is not declared in ${CLOSURE_CONFIG_PATH}`);
    process.stdout.write(
      providerClosureDigest({
        repo: path.resolve(repo),
        revision,
        provider: one,
        crateDir: declared[one],
      }).text,
    );
    return 0;
  }

  const built = buildClosureConfig({ repo: path.resolve(repo), revision, declared });
  const body = `${JSON.stringify(built, null, 2)}\n`;

  if (argv.includes("--write")) {
    await writeFile(configPath, body);
    console.log(`wrote ${CLOSURE_CONFIG_PATH} at ${revision.slice(0, 8)}`);
    return 0;
  }
  if (argv.includes("--check")) {
    const current = `${JSON.stringify(existing, null, 2)}\n`;
    if (current !== body) {
      console.error(
        `${CLOSURE_CONFIG_PATH} does not match the derivation at ${revision.slice(0, 8)}. ` +
          "Re-run with --write.",
      );
      return 1;
    }
    console.log(`${CLOSURE_CONFIG_PATH} matches ${revision.slice(0, 8)}`);
    return 0;
  }

  for (const [provider, entry] of Object.entries(built.providers)) {
    console.log(
      `${provider.padEnd(24)} ${entry.digest.slice(0, 16)}  crates=${entry.closureCrates.length} ` +
        `files=${entry.sourceFileCount} locked=${entry.lockedPackageCount}`,
    );
  }
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  process.exitCode = await main();
}
