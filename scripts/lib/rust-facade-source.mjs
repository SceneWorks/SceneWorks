import { rustCodeMask, withoutRustTestModules } from "./rust-source-slices.mjs";
import { readFile } from "node:fs/promises";
import path from "node:path";

/** Read a Rust root and the #[path] modules it brings into its namespace with a glob.
 * Scoped family modules and external test modules retain their namespaces. This follows the
 * authored facade instead of mirroring its changing file list.
 */
export async function rustFacadeSources(entry, seen = new Set()) {
  const absolute = path.resolve(entry);
  if (seen.has(absolute)) return [];
  seen.add(absolute);
  const source = await readFile(absolute, "utf8");
  const files = [{ path: absolute, source }];
  const declarations = /#\[path\s*=\s*"([^"]+)"\]\s*mod\s+([A-Za-z_][A-Za-z0-9_]*);\s*(?:pub(?:\([^)]*\))?\s+)?use\s+\2::\*;/g;
  for (const match of source.matchAll(declarations)) {
    files.push(...await rustFacadeSources(path.resolve(path.dirname(absolute), match[1]), seen));
  }
  return files;
}

export async function rustFacadeSource(entry) {
  return (await rustFacadeSources(entry)).map((file) => file.source).join("\n");
}

/** Exact top-level function spans, including parent/crate visibility. Literal/comment braces and
 * quoted function declarations are data; adjacent concern files cannot become one function body.
 */
export function rustFunctionSlices(source) {
  const code = rustCodeMask(source);
  const declarations = /^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:<|\()/gm;
  const functions = [];
  for (const match of code.matchAll(declarations)) {
    const start = match.index;
    const opening = code.indexOf("{", start);
    if (opening < 0) throw new Error(`${match[1]} has no function body`);
    let end = opening + 1;
    let depth = 1;
    while (end < code.length && depth > 0) {
      if (code[end] === "{") depth += 1;
      if (code[end] === "}") depth -= 1;
      end += 1;
    }
    if (depth > 0) throw new Error(`${match[1]} has no closing brace`);
    functions.push({ name: match[1], start, end, source: source.slice(start, end) });
  }
  return functions;
}

export function rustFunctionSource(source, name) {
  const matches = rustFunctionSlices(source).filter((item) => item.name === name);
  if (matches.length !== 1) throw new Error(`expected one owner of ${name}, found ${matches.length}`);
  return matches[0].source;
}

/** Find one authored top-level function, retaining its actual file identity for source guards. */
export async function rustFacadeFunction(entry, name) {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new Error(`Invalid Rust function name ${name}`);
  const matches = [];
  for (const file of await rustFacadeSources(entry)) {
    for (const item of rustFunctionSlices(file.source)) {
      if (item.name === name) matches.push({ path: file.path, source: item.source });
    }
  }
  if (matches.length !== 1) throw new Error(`${entry}: expected one owner of ${name}, found ${matches.length}`);
  return matches[0];
}

/** Read every authored external production module, including scoped concerns. Runtime guards
 * need the complete dependency tree rather than only the names flattened into a facade.
 */
export async function rustProductionSources(entry, seen = new Set(), includeTests = false) {
  const absolute = path.resolve(entry);
  const initialEntry = seen.size === 0;
  if (seen.has(absolute)) return [];
  seen.add(absolute);
  const source = await readFile(absolute, "utf8");
  const production = includeTests ? source : withoutRustTestModules(source);
  const code = rustCodeMask(production);
  const files = [{ path: absolute, source }];
  const declarations = /(?:#\[path\s*=\s*"([^"]+)"\]\s*)?(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*);/g;
  for (const match of production.matchAll(declarations)) {
    // A quoted/commented declaration is data. The masked source retains actual mod tokens only.
    if (!new RegExp(`\\bmod\\s+${match[2]};`).test(code.slice(match.index, match.index + match[0].length))) continue;
    const directory = (["lib", "main", "mod"].includes(path.basename(absolute, ".rs"))
      || (initialEntry && path.basename(path.dirname(absolute)) === "bin"))
      ? path.dirname(absolute) : absolute.slice(0, -3);
    let child = match[1]
      ? path.resolve(path.dirname(absolute), match[1])
      : path.join(directory, `${match[2]}.rs`);
    if (!match[1]) {
      // Rust permits both owner.rs and owner/mod.rs. Only the authored module is read.
      try { await readFile(child); }
      catch (error) {
        if (error.code !== "ENOENT") throw error;
        child = path.join(directory, match[2], "mod.rs");
      }
    }
    files.push(...await rustProductionSources(child, seen, includeTests));
  }
  return files;
}

export async function rustProductionSource(entry) {
  return (await rustProductionSources(entry)).map((file) => file.source).join("\n");
}


/** Source projections that historically consume test-authored registry sweeps retain that input
 * after tests move to authored sibling modules. This is never the runtime guard's source view.
 */
export async function rustSourceWithTests(entry) {
  return (await rustProductionSources(entry, new Set(), true)).map((file) => file.source).join("\n");
}
