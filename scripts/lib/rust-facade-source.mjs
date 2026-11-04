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
  const declarations = /#\[path\s*=\s*"([^"]+)"\]\s*mod\s+([A-Za-z_][A-Za-z0-9_]*);\s*use\s+\2::\*;/g;
  for (const match of source.matchAll(declarations)) {
    files.push(...await rustFacadeSources(path.resolve(path.dirname(absolute), match[1]), seen));
  }
  return files;
}

export async function rustFacadeSource(entry) {
  return (await rustFacadeSources(entry)).map((file) => file.source).join("\n");
}

/** Find one authored top-level function, retaining its actual file identity for source guards. */
export async function rustFacadeFunction(entry, name) {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new Error(`Invalid Rust function name ${name}`);
  const matches = [];
  const declaration = new RegExp(`^(?:pub(?:\\([^)]*\\))?\\s+)?(?:const\\s+)?fn\\s+${name}\\s*(?:<[^\\n]*>)?\\s*\\(`, "m");
  for (const file of await rustFacadeSources(entry)) {
    const found = declaration.exec(file.source);
    if (!found) continue;
    const end = file.source.indexOf("\n}", found.index);
    if (end < 0) throw new Error(`${file.path}: ${name} has no top-level closing brace`);
    matches.push({ path: file.path, source: file.source.slice(found.index, end + 2) });
  }
  if (matches.length !== 1) throw new Error(`${entry}: expected one owner of ${name}, found ${matches.length}`);
  return matches[0];
}
