#!/usr/bin/env node
// sc-17776 RESEARCH INSTRUMENT — not a validator, not wired into any gate, not a shipped layer.
//
// It exists to answer one question with a measurement instead of an argument: if the calibration
// audit hashed only "the code the measurements ran" rather than the whole linked test binary, which
// of the over-triggers documented in `docs/calibration-invalidation-survey-sc-17775.md` would it
// absolve, and would it still catch a genuine change?
//
// ## Why the walk is over the OBJECTS, not the linked image
//
// The obvious reading of "reachability-scoped digest" is a symbol + relocation walk of the PE. That
// is not possible on the artifact this audit hashes, and the reason is the audit's own determinism
// fix: `reproducibleLinkFlags()` passes `/DEBUG:NONE`, so no PDB is emitted, and an MSVC-linked
// image carries no COFF symbol table either. The linked binary has no symbol boundaries and no
// symbol-level relocations left — only base relocations. Recovering a call graph from it would mean
// disassembling 181 MB of x86-64.
//
// The pre-link COFF objects have exactly what is missing: one COMDAT section per function, a symbol
// table, and a relocation record for every cross-section reference. That is also the graph
// `link.exe /OPT:REF` itself walks, so a BFS over it models link-time DCE rather than approximating
// it from the far side.
//
// ## What is hashed
//
// Per reachable section: its canonical section kind, its raw bytes, and the (offset, type, target
// symbol NAME) of every relocation out of it. The bytes alone are not enough — a call site is a
// zero placeholder plus a relocation, so two functions calling different callees have identical
// bytes and are told apart only by the relocation targets. Container identity (which object file,
// which codegen unit) is deliberately NOT hashed: CGU names move when modules are reshuffled, which
// would reintroduce a source-shaped trigger into a unit whose whole point is to be codegen-shaped.
//
// Several digests are emitted per run so the false-green direction can be measured rather than
// assumed. For each scope in `SCOPES`:
//   scopes.<name>.all   — code + data sections reachable from that scope's roots. The candidate.
//   scopes.<name>.text  — executable sections only. Blind to a changed constant table; reported so
//                         the cost of that blindness is a number in the table rather than a
//                         footnote. Measured: it cannot see a provider bumping its own
//                         CALIBRATION_FINGERPRINT.
// And once per run:
//   linkedAll / linkedText — every section in every input object, no reachability filter. The
//                         control: it is what the current whole-binary unit approximates, so a run
//                         where it and a scope's digest agree is a run where scoping bought nothing.

import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";

const IMAGE_SCN_CNT_CODE = 0x00000020;
const IMAGE_SCN_MEM_EXECUTE = 0x20000000;
const IMAGE_SCN_LNK_REMOVE = 0x00000800;
const IMAGE_SCN_LNK_NRELOC_OVFL = 0x01000000;
const IMAGE_SYM_CLASS_EXTERNAL = 2;

/** `.text$mn`, `.rdata$foo`, `.xdata$x` all collapse to their grouped section. The suffix is a
 *  linker ordering key, and it moves with codegen-unit naming rather than with code. */
export function canonicalSectionName(name) {
  const dollar = name.indexOf("$");
  return dollar === -1 ? name : name.slice(0, dollar);
}

/**
 * Strip the codegen-unit-local suffix LLVM appends when it internalises a symbol.
 *
 * `_ZN…provider_contract_for17h722cbf49E.llvm.15178270987596298186` — the number is derived from the
 * module, so it moves when ANY code in the same crate moves, including code the walk never reaches.
 * Hashing it makes the digest strictly noisier than the linked binary it is meant to improve on:
 * measured directly, `5ffd7612` and `a4f409ae` link to byte-identical binaries (the window sc-17607
 * certified) and their un-canonicalised object digests still differ.
 */
export function canonicalSymbolName(name) {
  return name.replace(/\.llvm\.\d+/g, "");
}

export function isExecutable(characteristics) {
  return (characteristics & (IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE)) !== 0;
}

/**
 * Parse one COFF object into sections, symbols and relocations.
 *
 * Only what the walk needs. Auxiliary symbol records are skipped rather than decoded: COMDAT
 * selection lives there, and the walk deliberately does not model it — see `buildGraph`.
 */
export function parseCoff(buf) {
  if (buf.length < 20) throw new Error("truncated COFF header");
  const numberOfSections = buf.readUInt16LE(2);
  const pointerToSymbolTable = buf.readUInt32LE(8);
  const numberOfSymbols = buf.readUInt32LE(12);
  const sizeOfOptionalHeader = buf.readUInt16LE(16);
  const stringTableAt = pointerToSymbolTable + numberOfSymbols * 18;

  const stringAt = (offset) => {
    if (stringTableAt === 0 || stringTableAt >= buf.length) return "";
    const start = stringTableAt + offset;
    let end = start;
    while (end < buf.length && buf[end] !== 0) end += 1;
    return buf.toString("utf8", start, end);
  };
  const shortName = (slice) => {
    if (slice[0] === 0) return stringAt(slice.readUInt32LE(4));
    let end = 0;
    while (end < 8 && slice[end] !== 0) end += 1;
    return slice.toString("utf8", 0, end);
  };

  const sections = [];
  let at = 20 + sizeOfOptionalHeader;
  for (let i = 0; i < numberOfSections; i += 1, at += 40) {
    const header = buf.subarray(at, at + 40);
    let name = header.subarray(0, 8);
    // A long section name is `/<decimal offset into the string table>`, not the `\0`+offset form
    // symbols use. Rust emits these constantly (`.text$_ZN...`), so getting it wrong would collapse
    // most COMDATs onto one bucket name.
    let sectionName;
    if (name[0] === 0x2f) {
      let end = 1;
      while (end < 8 && name[end] !== 0) end += 1;
      sectionName = stringAt(Number.parseInt(name.toString("utf8", 1, end), 10));
    } else {
      let end = 0;
      while (end < 8 && name[end] !== 0) end += 1;
      sectionName = name.toString("utf8", 0, end);
    }
    const sizeOfRawData = header.readUInt32LE(16);
    const pointerToRawData = header.readUInt32LE(20);
    const pointerToRelocations = header.readUInt32LE(24);
    let numberOfRelocations = header.readUInt16LE(32);
    const characteristics = header.readUInt32LE(36);
    let relocAt = pointerToRelocations;
    if ((characteristics & IMAGE_SCN_LNK_NRELOC_OVFL) !== 0 && numberOfRelocations === 0xffff) {
      // The real count lives in the VirtualAddress of a leading dummy record.
      numberOfRelocations = buf.readUInt32LE(pointerToRelocations) - 1;
      relocAt += 10;
    }
    const relocations = [];
    for (let r = 0; r < numberOfRelocations; r += 1) {
      const base = relocAt + r * 10;
      relocations.push({
        offset: buf.readUInt32LE(base),
        symbol: buf.readUInt32LE(base + 4),
        type: buf.readUInt16LE(base + 8),
      });
    }
    sections.push({
      name: sectionName,
      canonical: canonicalSectionName(sectionName),
      characteristics,
      executable: isExecutable(characteristics),
      discardable: (characteristics & IMAGE_SCN_LNK_REMOVE) !== 0,
      bytes:
        pointerToRawData === 0
          ? Buffer.alloc(0)
          : buf.subarray(pointerToRawData, pointerToRawData + sizeOfRawData),
      relocations,
    });
  }

  const symbols = [];
  for (let i = 0; i < numberOfSymbols; ) {
    const base = pointerToSymbolTable + i * 18;
    const record = buf.subarray(base, base + 18);
    const aux = record[17];
    symbols[i] = {
      name: shortName(record.subarray(0, 8)),
      value: record.readUInt32LE(8),
      section: record.readInt16LE(12),
      storageClass: record[16],
    };
    for (let a = 1; a <= aux; a += 1) symbols[i + a] = null;
    i += 1 + aux;
  }

  return { sections, symbols };
}

/** Every `.o` member of a thin-or-fat `ar` archive (an rlib), plus its member names. */
export function parseArchive(buf) {
  const members = [];
  if (buf.length < 8 || buf.toString("ascii", 0, 8) !== "!<arch>\n") return members;
  let at = 8;
  let longNames = null;
  while (at + 60 <= buf.length) {
    const header = buf.subarray(at, at + 60);
    const rawName = header.toString("ascii", 0, 16).trim();
    const size = Number.parseInt(header.toString("ascii", 48, 58).trim(), 10);
    if (!Number.isFinite(size)) break;
    const body = buf.subarray(at + 60, at + 60 + size);
    if (rawName === "//") {
      longNames = body.toString("ascii");
    } else if (rawName !== "/" && rawName !== "/SYM64/") {
      let name = rawName.endsWith("/") ? rawName.slice(0, -1) : rawName;
      if (name.startsWith("/") && longNames) {
        // Two conventions share this slot. GNU `ar` terminates each long name with "/\n"; the
        // Microsoft COFF variant that rustc emits on `*-pc-windows-msvc` terminates with NUL.
        // Handling only the GNU form silently yields zero members for every rlib, which reads as an
        // empty (and therefore trivially stable) graph rather than as a parse failure.
        const offset = Number.parseInt(name.slice(1), 10);
        let end = offset;
        while (end < longNames.length && longNames[end] !== "\n" && longNames[end] !== "\0") end += 1;
        name = longNames.slice(offset, end).replace(/\/$/, "");
      }
      if (name.endsWith(".o") || name.endsWith(".obj")) members.push({ name, body });
    }
    at += 60 + size + (size % 2);
  }
  return members;
}

/**
 * The reachability graph over a set of COFF objects.
 *
 * A node is one section of one object. An edge is a relocation: the referencing section reaches
 * whichever section defines the referenced symbol, resolved inside the object first and through the
 * global symbol table otherwise.
 *
 * It keeps the FIRST definition of an external symbol, with the input list sorted, so
 * COMDAT duplicates resolve deterministically. `link.exe` may pick a different copy; for identical
 * COMDATs (which is what duplicates are) the contents match, so the digest does not depend on which
 * one wins. `duplicateDefinitions` reports the count so that assumption stays visible.
 */
export function buildGraph(objects) {
  const nodes = new Map();
  const globals = new Map();
  let duplicateDefinitions = 0;

  for (const { key, coff } of objects) {
    coff.sections.forEach((section, index) => {
      if (section.discardable) return;
      nodes.set(`${key}#${index}`, { key, index, section, coff });
    });
    for (const symbol of coff.symbols) {
      if (!symbol || symbol.section <= 0) continue;
      if (symbol.storageClass !== IMAGE_SYM_CLASS_EXTERNAL) continue;
      const id = `${key}#${symbol.section - 1}`;
      if (!nodes.has(id)) continue;
      if (globals.has(symbol.name)) duplicateDefinitions += 1;
      else globals.set(symbol.name, id);
    }
  }
  return { nodes, globals, duplicateDefinitions };
}

export function resolveEdge(graph, node, relocation) {
  const symbol = node.coff.symbols[relocation.symbol];
  if (!symbol) return null;
  if (symbol.section > 0) {
    const local = `${node.key}#${symbol.section - 1}`;
    return graph.nodes.has(local) ? local : null;
  }
  return graph.globals.get(symbol.name) ?? null;
}

/** Breadth-first so `parents` yields a SHORTEST path — the one worth reading when explaining why
 *  something unexpected is in the reachable set. */
export function reachableFrom(graph, rootIds) {
  const seen = new Set(rootIds);
  const parents = new Map();
  let frontier = [...rootIds];
  while (frontier.length > 0) {
    const next = [];
    for (const id of frontier) {
      const node = graph.nodes.get(id);
      if (!node) continue;
      for (const relocation of node.section.relocations) {
        const target = resolveEdge(graph, node, relocation);
        if (!target || seen.has(target)) continue;
        seen.add(target);
        parents.set(target, id);
        next.push(target);
      }
    }
    frontier = next;
  }
  seen.parents = parents;
  return seen;
}

/** The symbol names defined in a section, for rendering a path a human can read. */
export function sectionSymbols(graph, id) {
  const node = graph.nodes.get(id);
  if (!node) return [id];
  const names = [];
  for (const symbol of node.coff.symbols) {
    if (symbol && symbol.section - 1 === node.index) names.push(canonicalSymbolName(symbol.name));
  }
  return names.length > 0 ? names : [`${node.key}#${node.index} (${node.section.name})`];
}

export function explainPath(graph, reachable, id) {
  const path = [];
  for (let at = id; at !== undefined; at = reachable.parents.get(at)) {
    path.push(sectionSymbols(graph, at)[0]);
    if (!reachable.parents.has(at)) break;
  }
  return path.reverse();
}

/**
 * The content fingerprint of one section: what it is, what is in it, and what it points at.
 *
 * Relocation TARGET NAMES are hashed, offsets and types with them. Without that, two functions
 * whose only difference is which callee they call hash the same, because the call displacement in
 * the object is a zero placeholder that the linker fills in.
 */
export function sectionFingerprint(graph, node) {
  const hash = createHash("sha256");
  hash.update(node.section.canonical);
  hash.update("\0");
  hash.update(node.section.bytes);
  const edges = node.section.relocations.map((relocation) => {
    const symbol = node.coff.symbols[relocation.symbol];
    return `${relocation.offset}:${relocation.type}:${symbol ? canonicalSymbolName(symbol.name) : "?"}`;
  });
  for (const edge of edges.sort()) {
    hash.update(edge);
    hash.update("\0");
  }
  return hash.digest("hex");
}

export function digestOf(graph, ids, { executableOnly = false } = {}) {
  const fingerprints = [];
  let bytes = 0;
  for (const id of ids) {
    const node = graph.nodes.get(id);
    if (!node) continue;
    if (executableOnly && !node.section.executable) continue;
    fingerprints.push(sectionFingerprint(graph, node));
    bytes += node.section.bytes.length;
  }
  fingerprints.sort();
  const hash = createHash("sha256");
  for (const fingerprint of fingerprints) hash.update(fingerprint);
  return { digest: `sha256:${hash.digest("hex")}`, sections: fingerprints.length, bytes };
}

function collectFiles(dir, out = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) collectFiles(full, out);
    else out.push(full);
  }
  return out;
}

/** Every COFF object in `files`, sorted, so `buildGraph`'s first-wins resolution is deterministic. */
export function loadObjects(files) {
  const inputs = [...new Set(files)].filter((file) => /\.(rlib|o|obj)$/.test(file)).sort();
  const objects = [];
  const failures = [];
  for (const file of inputs) {
    const base = path.basename(file);
    const buf = readFileSync(file);
    const members = file.endsWith(".rlib")
      ? parseArchive(buf)
      : [{ name: base, body: buf }];
    // An rlib with no object members is a PARSE bug, not an empty crate — and it fails in the
    // direction that looks like success: fewer nodes, a smaller graph, a digest that is stable
    // because it describes almost nothing. The MSVC long-name table cost exactly this once.
    if (members.length === 0) failures.push(`${base}: no object members`);
    for (const member of members) {
      try {
        objects.push({ key: `${base}/${member.name}`, coff: parseCoff(member.body) });
      } catch (error) {
        failures.push(`${base}/${member.name}: ${error.message}`);
      }
    }
  }
  return { objects, failures, archives: inputs.length };
}

export function rootsMatching(graph, pattern) {
  const roots = new Set();
  const names = [];
  for (const node of graph.nodes.values()) {
    for (const symbol of node.coff.symbols) {
      if (!symbol || symbol.section <= 0) continue;
      if (!pattern.test(symbol.name)) continue;
      const id = `${node.key}#${symbol.section - 1}`;
      if (graph.nodes.has(id)) {
        roots.add(id);
        names.push(symbol.name);
      }
    }
  }
  return { roots: [...roots], names: [...new Set(names)].sort() };
}

/**
 * The roots each scope is walked from.
 *
 * `measured` is the literal reading of "the code the measurements ran": the test function named in
 * the approved capture command. `ladder` is a strictly narrower scope — the registered memory
 * contract and its safety check, i.e. the code that decides the rungs a calibration measures. The
 * two are reported side by side because narrowing trades coverage for stability in a direction that
 * has to be measured, not assumed: the ladder scope cannot see a change in the transformer's
 * attention chunking, which moves real VRAM without moving the contract table.
 */
export const SCOPES = Object.freeze({
  measured: /flux2_dev_probed_generate_for_offload_ab/,
  ladder: /candle_gen_flux2.*(provider_contract|registered_safety_check)/,
});

export function probe({ inputs, scopes = SCOPES, explain = null }) {
  const { objects, failures, archives } = loadObjects(inputs);
  const graph = buildGraph(objects);
  const all = [...graph.nodes.keys()];
  const result = {
    inputs: { archives, objects: objects.length, parseFailures: failures },
    duplicateDefinitions: graph.duplicateDefinitions,
    linkedAll: digestOf(graph, all),
    linkedText: digestOf(graph, all, { executableOnly: true }),
    scopes: {},
  };
  for (const [name, pattern] of Object.entries(scopes)) {
    const { roots, names } = rootsMatching(graph, pattern);
    // A scope whose roots do not exist at this revision is recorded as absent rather than thrown:
    // `provider_contract_for` and `registered_safety_check` were introduced with the ladder itself,
    // so any window predating it has no ladder to scope to. Absent is a different fact from
    // identical, and collapsing the two would read as "the ladder did not change".
    if (roots.length === 0) {
      result.scopes[name] = { roots: [], absent: true };
      continue;
    }
    const reachable = reachableFrom(graph, roots);
    result.scopes[name] = {
      roots: names,
      all: digestOf(graph, reachable),
      text: digestOf(graph, reachable, { executableOnly: true }),
    };
    if (explain) {
      const hits = [...reachable].filter((id) =>
        sectionSymbols(graph, id).some((symbol) => explain.test(symbol)),
      );
      result.scopes[name].explain = hits.slice(0, 3).map((id) => explainPath(graph, reachable, id));
      result.scopes[name].explainHits = hits.length;
    }
  }
  return result;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const value = (flag) => {
    const at = args.indexOf(flag);
    return at === -1 ? undefined : args[at + 1];
  };
  const inputs = (value("--inputs") ?? "")
    .split(",")
    .filter(Boolean)
    .flatMap((entry) => (statSync(entry).isDirectory() ? collectFiles(entry) : [entry]));
  const result = probe({ inputs, explain: value("--explain") ? new RegExp(value("--explain")) : null });
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}
