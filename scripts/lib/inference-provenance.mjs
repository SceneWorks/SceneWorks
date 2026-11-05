// The same neutral dependency declaration core/build.rs compiles into new evidence.
// Historical measurement receipts are never rewritten by this reader.
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
export const INFERENCE_PROVENANCE_MANIFEST = "crates/sceneworks-core/Cargo.toml";

export async function inferenceRevision(root = ROOT) {
  const source = await readFile(path.join(root, INFERENCE_PROVENANCE_MANIFEST), "utf8");
  const declaration = /^sceneworks-gen-core = \{[^\r\n]+\}[ \t]*\r?$/m.exec(source)?.[0];
  const revision = /\brev = "([0-9a-f]{40})"/.exec(declaration ?? "")?.[1];
  if (!revision || !declaration.includes('git = "https://github.com/SceneWorks/inference"')) {
    throw new Error(`${INFERENCE_PROVENANCE_MANIFEST} declares no exact inference contract pin`);
  }
  return revision;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.stdout.write(`${await inferenceRevision()}\n`);
}
