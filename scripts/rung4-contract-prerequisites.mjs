// Per-(family, backend) rung-4 prerequisite records, derived from the pinned inference source
// (sc-19542).
//
// WHY THIS EXISTS
//
// `generate-memory-matrix.mjs`'s rung-4 arm used to gate EVERY family on
// `stagedResidencyIsAvailable` — a blanket rung-1 availability proxy. That is not the contract rule.
// gen-core's rule for `MemoryStrategy::BoundedTransformerResidency` is
// `MemoryProviderContract::validate_selection`, which walks
// `MemoryProviderContract::requires(BoundedTransformerResidency)`: the SHARED constant
// `BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES` (one `LoadShape(DeferredMaterialization)` edge and
// nothing else) followed by whatever THAT provider appended through `additional_prerequisites`.
// Providers genuinely differ: mlx-gen-anima and mlx-gen-chroma push a
// `BoundedTransformerResidency -> StagedResidency (EngagedInSameRequest)` edge when the load is
// streamable, while mlx-gen-bernini deliberately pushes none. Applying one provider's edge to all
// twenty families is a proxy that happens to agree today and diverges silently the first time a
// provider's prerequisites change.
//
// This module derives the real per-provider edge set from the pinned revision so the generator can
// consult it instead. `config/rung4-contract-prerequisites.json` holds the records; this script
// writes them (`--write`) and re-derives them for comparison (`--check`).
//
// NO BUILD, AND DERIVED OFFLINE — BUT VERIFIED IN CI
//
// Everything here reads `git ls-tree` / `git show` at a revision, so it runs in seconds and needs no
// toolchain, no GPU and no weights. It does need an inference clone, which a SceneWorks checkout
// does not contain — so the records are checked in, where a reviewer sees a change in the diff
// instead of it being conjured at check time, and `check.yml`'s `parity-digests` job re-derives them
// against a `--depth=1` fetch of the pinned revision (SceneWorks/inference is public, so that costs
// seconds and needs no token). `npm run check` can only assert the file is KEYED to the live pin;
// without the CI step the records would be checked-in data nothing grades.
//
// THE EXTRACTOR FAILS CLOSED
//
// Reading Rust with regular expressions is only sound if every shape it does not understand is an
// ERROR rather than a zero. `additionalPrerequisiteEdges` therefore recognises a closed set of four
// construction shapes and throws on any other occurrence of the field. A provider that starts
// building its prerequisite vector some new way fails the derivation instead of silently recording
// "no edges" — which is the fail-open direction, and the one that would put this module back where
// the blanket proxy was.

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const RECORDS_PATH = "config/rung4-contract-prerequisites.json";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/**
 * The rung-4 edge every recognised provider pushes, spelled as it appears in gen-core.
 *
 * `MemoryStrategyPrerequisite::Rung { rung: MemoryStrategy::StagedResidency, scope:
 * MemoryPrerequisiteScope::EngagedInSameRequest }` — the only variant any media provider appends at
 * the pinned revision. A second variant appearing is an unrecognised shape, and the extractor says
 * so rather than dropping it.
 */
const STAGED_RESIDENCY_EDGE =
  /^MemoryStrategyPrerequisite::Rung\{rung:MemoryStrategy::StagedResidency,scope:MemoryPrerequisiteScope::EngagedInSameRequest,?\}$/;

/** How this module names that edge in a record. */
export const STAGED_RESIDENCY_ENGAGED_IN_SAME_REQUEST = Object.freeze({
  kind: "rung",
  rung: "staged_residency",
  scope: "engaged_in_same_request",
});

/** The rung whose prerequisite graph this module records. */
const RUNG4 = "BoundedTransformerResidency";

/**
 * Strip comments and whitespace so a construction spanning fifteen rustfmt'd lines is one token
 * sequence.
 *
 * Line comments go first and whole-line block comments after, because a `//` inside a string literal
 * would otherwise eat the rest of the line — no provider has one in these constructions today, and
 * an unrecognised shape is an error rather than a silent zero, so the failure mode of getting this
 * wrong is a red derivation and not a wrong record.
 */
export function normalizeRust(source) {
  return source
    .replace(/\/\/.*$/gm, "")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\s+/g, "");
}

/**
 * Every path qualifier a provider may write in front of these types.
 *
 * Providers reach gen-core three ways — directly, re-exported through `mlx_gen::gen_core`, or
 * through `candle_gen::gen_core` — and the same type therefore appears under three spellings. They
 * are erased before matching so the shape patterns stay single-form.
 */
function eraseCrateQualifiers(text) {
  return text.replace(/(?:crate::)?(?:mlx_gen|candle_gen|gen_core)::/g, "");
}

/** Rust `path::to::Ident` -> the shape patterns' single-form spelling. */
function canonical(text) {
  return eraseCrateQualifiers(normalizeRust(text));
}

/**
 * Blank every string, raw-string and char literal so brace counting cannot be thrown by a `}` inside
 * one. Lengths are preserved, so offsets into the result index the original.
 */
function blankLiterals(source) {
  const out = source.split("");
  let at = 0;
  const blank = (from, to) => {
    for (let i = from; i < to && i < out.length; i += 1) out[i] = " ";
  };
  while (at < source.length) {
    const ch = source[at];
    if (ch === "/" && source[at + 1] === "/") {
      const end = source.indexOf("\n", at);
      const stop = end === -1 ? source.length : end;
      // Blank, not skip. A brace in a doc comment counts as a brace otherwise, and these files are
      // full of them — that is what unbalanced the first run.
      blank(at, stop);
      at = stop;
      continue;
    }
    if (ch === "/" && source[at + 1] === "*") {
      const end = source.indexOf("*/", at + 2);
      const stop = end === -1 ? source.length : end + 2;
      blank(at, stop);
      at = stop;
      continue;
    }
    // Char literals: `'{'`, `'}'`, `'\''`. A lifetime (`'a`) has no closing quote and is left alone.
    if (ch === "'") {
      const literal = /^'(?:\\.|[^'\\])'/.exec(source.slice(at, at + 8));
      if (literal) {
        blank(at, at + literal[0].length);
        at += literal[0].length;
        continue;
      }
    }
    const raw = /^r(#*)"/.exec(source.slice(at, at + 32));
    if (raw) {
      const terminator = `"${raw[1]}`;
      const start = at + raw[0].length;
      const end = source.indexOf(terminator, start);
      if (end === -1) throw new Error("unterminated raw string literal");
      blank(start, end);
      at = end + terminator.length;
      continue;
    }
    if (ch === '"') {
      let i = at + 1;
      while (i < source.length && source[i] !== '"') i += source[i] === "\\" ? 2 : 1;
      blank(at + 1, i);
      at = i + 1;
      continue;
    }
    at += 1;
  }
  return out.join("");
}

/**
 * Remove `#[cfg(test)] mod … { … }` blocks, brace-matched.
 *
 * The first version of this cut the file at the first `#[cfg(test)]` and kept the prefix. That is
 * wrong in the FAIL-OPEN direction and it really did fail: `mlx-gen-krea/src/memory_strategy.rs`
 * carries a `#[cfg(test)] use` on line 13, so the "production" text was its first 820 of 30,249
 * bytes and the crate derived zero edges while its source pushes one. A silent zero is exactly the
 * shape sc-19542 exists to remove, so the extent is now matched properly and an attribute that is
 * not on a `mod` (the `use` above) removes nothing, because a `use` declares no prerequisites.
 */
export function stripTestModules(source) {
  const scan = blankLiterals(source);
  const attribute = /#\[cfg\((?:[^)]*\()?[^)]*\btest\b[^\]]*\]\s*/g;
  const cuts = [];
  for (let found = attribute.exec(scan); found; found = attribute.exec(scan)) {
    const after = scan.slice(found.index + found[0].length);
    const mod = /^(?:pub\s+)?mod\s+\w+\s*\{/.exec(after);
    if (!mod) continue;
    let depth = 0;
    let at = found.index + found[0].length + mod[0].length - 1;
    for (; at < scan.length; at += 1) {
      if (scan[at] === "{") depth += 1;
      else if (scan[at] === "}") {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    if (depth !== 0) {
      throw new Error("unbalanced braces in a #[cfg(test)] module — the extractor cannot bound it");
    }
    cuts.push([found.index, at + 1]);
  }
  let kept = source;
  for (const [from, to] of cuts.reverse()) kept = kept.slice(0, from) + kept.slice(to);
  return kept;
}

/**
 * The strategies receiving a rung-1 edge in one `additional_prerequisites` construction.
 *
 * Four shapes are recognised, which is every shape present at the pinned revision:
 *
 *   1. `additional_prerequisites: Vec::new()`                                   — no edges.
 *   2. `additional_prerequisites: [A, B].into_iter().map(|s| (s, EDGE)).collect()`
 *      and the `contract.additional_prerequisites = [...]` assignment form of the same.
 *   3. `contract.additional_prerequisites.push((STRATEGY, EDGE))`
 *   4. `additional_prerequisites: COND.then_some((STRATEGY, EDGE)).into_iter().collect()`
 *
 * A read (`.is_empty()`, `.iter()`) is not a construction and contributes nothing. Anything else
 * throws.
 */
export function additionalPrerequisiteEdges(source, where) {
  const text = canonical(source);
  const strategies = [];
  const field = "additional_prerequisites";
  for (let at = text.indexOf(field); at !== -1; at = text.indexOf(field, at + 1)) {
    const rest = text.slice(at + field.length);
    // A read, not a construction.
    if (/^\.(?:is_empty|iter|len|contains|first|last)\b/.test(rest)) continue;

    const vecNew = /^:Vec::new\(\)/.exec(rest);
    if (vecNew) continue;

    const arrayMap =
      /^\s*[:=]\[((?:MemoryStrategy::\w+,?)+)\]\.into_iter\(\)\.map\(\|(\w+)\|\{?\((\w+),(MemoryStrategyPrerequisite::Rung\{[^}]*\},?)\)\}?\)\.collect\(\)/.exec(
        rest,
      );
    if (arrayMap) {
      const [, list, binder, applied, edge] = arrayMap;
      if (binder !== applied) {
        throw new Error(`${where}: ${field} maps ${binder} but pairs ${applied}`);
      }
      assertRecognisedEdge(edge, where);
      strategies.push(...list.split(",").filter(Boolean).map(stripStrategy));
      continue;
    }

    const push =
      /^\.push\(\(MemoryStrategy::(\w+),(MemoryStrategyPrerequisite::Rung\{[^}]*\},?),?\)\)/.exec(
        rest,
      );
    if (push) {
      assertRecognisedEdge(push[2], where);
      strategies.push(push[1]);
      continue;
    }

    const thenSome =
      /^:[\w.()]+\.then_some\(\(MemoryStrategy::(\w+),(MemoryStrategyPrerequisite::Rung\{[^}]*\},?),?\)\)\.into_iter\(\)\.collect\(\)/.exec(
        rest,
      );
    if (thenSome) {
      assertRecognisedEdge(thenSome[2], where);
      strategies.push(thenSome[1]);
      continue;
    }

    throw new Error(
      `${where}: unrecognised \`${field}\` construction at ${JSON.stringify(rest.slice(0, 140))}. ` +
        "The extractor recognises a closed set of shapes and fails closed on anything else, because " +
        "reading a new shape as “no edges” is the fail-open direction this record exists to remove (sc-19542)",
    );
  }
  return strategies;
}

function stripStrategy(token) {
  const match = /^MemoryStrategy::(\w+)$/.exec(token);
  if (!match) throw new Error(`not a MemoryStrategy path: ${token}`);
  return match[1];
}

function assertRecognisedEdge(raw, where) {
  // rustfmt's trailing comma after the struct literal rides along in the capture.
  const edge = raw.replace(/,+$/, "");
  if (!STAGED_RESIDENCY_EDGE.test(edge)) {
    throw new Error(
      `${where}: unrecognised prerequisite edge ${JSON.stringify(edge)} — this module records the ` +
        "StagedResidency/EngagedInSameRequest edge and knows no other variant, so a new one must be " +
        "taught here rather than dropped (sc-19542)",
    );
  }
}

function git(repo, args) {
  return execFileSync("git", ["-C", repo, ...args], {
    encoding: "utf8",
    maxBuffer: 512 * 1024 * 1024,
  });
}

/** Non-test Rust sources of one crate at `revision`. */
function crateSources(repo, revision, cratePath) {
  const listed = git(repo, ["ls-tree", "-r", "--name-only", revision, "--", `${cratePath}/src/`]);
  return listed
    .split("\n")
    .filter((line) => line.endsWith(".rs"))
    .sort();
}

/**
 * Derive one (family, backend) record's rung-4 additive edges from the pinned tree.
 *
 * Scoped to the crate's `src/` because that is the production contract; `tests/` asserts ABOUT the
 * contract (`assert!(contract.additional_prerequisites.is_empty())`) and constructing a record from
 * a test fixture would record the fixture rather than the provider.
 */
export function deriveCrateEdges(repo, revision, cratePath) {
  const sources = crateSources(repo, revision, cratePath);
  if (!sources.length) {
    throw new Error(
      `${cratePath}: no Rust sources at ${revision.slice(0, 9)} — the record names a crate the ` +
        "pinned revision does not contain (sc-19542)",
    );
  }
  const edges = [];
  for (const source of sources) {
    const body = git(repo, ["show", `${revision}:${source}`]);
    // `#[cfg(test)]` modules build contracts to assert about them; those are not the provider's
    // declaration.
    const production = stripTestModules(body);
    for (const strategy of additionalPrerequisiteEdges(production, source)) {
      if (strategy === RUNG4) edges.push({ ...STAGED_RESIDENCY_ENGAGED_IN_SAME_REQUEST, source });
    }
  }
  return edges;
}

export function inferencePin(cargoToml) {
  const match =
    /candle-kernels\s*=\s*\{[^}]*?github\.com\/SceneWorks\/inference[^}]*?rev\s*=\s*"([0-9a-f]+)"/.exec(
      cargoToml,
    );
  if (!match) throw new Error("no inference pin in Cargo.toml");
  return match[1];
}

/** Re-derive every record and return the disagreements, as `[where, expected, actual]`. */
export function compareRecords(records, repo) {
  const failures = [];
  for (const [group, family] of Object.entries(records.families)) {
    for (const [backend, record] of Object.entries(family.backends)) {
      const where = `${family.name} (${backend})`;
      const derived = deriveCrateEdges(repo, records.inferenceRevision, record.crate);
      const recorded = record.additionalPrerequisites ?? [];
      const shape = (edges) =>
        JSON.stringify(edges.map(({ kind, rung, scope, source }) => ({ kind, rung, scope, source })));
      if (shape(derived) !== shape(recorded)) {
        failures.push([`${group} ${where}`, shape(derived), shape(recorded)]);
      }
    }
  }
  return failures;
}

function readRecords() {
  return JSON.parse(readFileSync(path.join(repoRoot, RECORDS_PATH), "utf8"));
}

function selfTest() {
  const cases = [
    ["empty vec", "additional_prerequisites: Vec::new(),", []],
    [
      "array map",
      `additional_prerequisites: [
          MemoryStrategy::BoundedDecode,
          MemoryStrategy::BoundedTransformerResidency,
       ]
       .into_iter()
       .map(|strategy| {
           (
               strategy,
               MemoryStrategyPrerequisite::Rung {
                   rung: MemoryStrategy::StagedResidency,
                   scope: MemoryPrerequisiteScope::EngagedInSameRequest,
               },
           )
       })
       .collect(),`,
      ["BoundedDecode", "BoundedTransformerResidency"],
    ],
    [
      "push",
      `contract.additional_prerequisites.push((
           MemoryStrategy::BoundedTransformerResidency,
           mlx_gen::gen_core::MemoryStrategyPrerequisite::Rung {
               rung: MemoryStrategy::StagedResidency,
               scope: mlx_gen::gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
           },
       ));`,
      ["BoundedTransformerResidency"],
    ],
    [
      "then_some",
      `additional_prerequisites: streamable_transformer
           .then_some((
               MemoryStrategy::BoundedTransformerResidency,
               mlx_gen::gen_core::MemoryStrategyPrerequisite::Rung {
                   rung: MemoryStrategy::StagedResidency,
                   scope: mlx_gen::gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
               },
           ))
           .into_iter()
           .collect(),`,
      ["BoundedTransformerResidency"],
    ],
    ["read is not a construction", "assert!(contract.additional_prerequisites.is_empty());", []],
  ];
  for (const [name, source, expected] of cases) {
    const actual = additionalPrerequisiteEdges(source, name);
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
      throw new Error(`self-test ${name}: expected ${expected} got ${actual}`);
    }
  }
  // The regression that produced 13 silently-empty records on the first derivation run: a
  // `#[cfg(test)] use` above the production construction. The attribute is real, it is not on a
  // `mod`, and cutting the file there loses the provider's own declaration.
  const earlyTestUse = `
#[cfg(test)]
use gen_core::MemoryGeometry;

pub fn contract() -> MemoryProviderContract {
    contract.additional_prerequisites.push((
        MemoryStrategy::BoundedTransformerResidency,
        MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));
}

#[cfg(test)]
mod tests {
    #[test]
    fn brace_in_a_string_does_not_end_the_module() {
        assert_eq!(render("}"), "}");
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedDecode,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
    }
}
`;
  const production = additionalPrerequisiteEdges(stripTestModules(earlyTestUse), "early cfg(test)");
  if (JSON.stringify(production) !== JSON.stringify(["BoundedTransformerResidency"])) {
    throw new Error(
      `self-test: production edges survived a #[cfg(test)] use and a test module — got ${production}`,
    );
  }
  // ...and the same text WITHOUT stripping sees the test module's edge, so the case above is graded
  // by the stripping rather than by the fixture happening to contain one construction.
  const unstripped = additionalPrerequisiteEdges(earlyTestUse, "unstripped");
  if (unstripped.length !== 2) {
    throw new Error(`self-test: the fixture must carry a test-module edge to grade — got ${unstripped}`);
  }

  let threw = false;
  try {
    additionalPrerequisiteEdges("additional_prerequisites: build_them(spec),", "novel shape");
  } catch {
    threw = true;
  }
  if (!threw) throw new Error("self-test: an unrecognised construction must fail closed");
  threw = false;
  try {
    additionalPrerequisiteEdges(
      `contract.additional_prerequisites.push((
           MemoryStrategy::BoundedTransformerResidency,
           MemoryStrategyPrerequisite::LoadShape(LoadShape::DeferredMaterialization),
       ));`,
      "novel edge",
    );
  } catch {
    threw = true;
  }
  if (!threw) throw new Error("self-test: an unrecognised edge variant must fail closed");
  process.stdout.write("rung4-contract-prerequisites self-test: ok\n");
}

function main(argv) {
  if (argv.includes("--self-test")) {
    selfTest();
    return;
  }
  const records = readRecords();
  const pin = inferencePin(readFileSync(path.join(repoRoot, "Cargo.toml"), "utf8"));
  if (records.inferenceRevision !== pin) {
    process.stderr.write(
      `${RECORDS_PATH} is keyed to ${records.inferenceRevision?.slice(0, 9) ?? "(unset)"} but ` +
        `Cargo pins ${pin.slice(0, 9)}. Re-run: node scripts/rung4-contract-prerequisites.mjs ` +
        "--repo <inference> --write\n",
    );
    process.exit(1);
  }
  const repoIndex = argv.indexOf("--repo");
  if (repoIndex === -1) {
    process.stderr.write(
      "usage: node scripts/rung4-contract-prerequisites.mjs --repo <inference-checkout> [--check|--write]\n" +
        "       node scripts/rung4-contract-prerequisites.mjs --self-test\n",
    );
    process.exit(2);
  }
  const repo = path.resolve(argv[repoIndex + 1]);
  if (argv.includes("--write")) {
    for (const family of Object.values(records.families)) {
      for (const [backend, record] of Object.entries(family.backends)) {
        void backend;
        record.additionalPrerequisites = deriveCrateEdges(repo, pin, record.crate);
      }
    }
    writeFileSync(path.join(repoRoot, RECORDS_PATH), `${JSON.stringify(records, null, 2)}\n`);
    process.stdout.write(`wrote ${RECORDS_PATH}\n`);
    return;
  }
  const failures = compareRecords(records, repo);
  if (failures.length) {
    for (const [where, derived, recorded] of failures) {
      process.stderr.write(`${where}: pinned source derives ${derived}, record says ${recorded}\n`);
    }
    process.stderr.write(
      `${RECORDS_PATH} disagrees with the pinned inference source. Re-run: node ` +
        "scripts/rung4-contract-prerequisites.mjs --repo <inference> --write\n",
    );
    process.exit(1);
  }
  process.stdout.write(
    `${RECORDS_PATH}: every record agrees with ${pin.slice(0, 9)}\n`,
  );
}

if (process.argv[1] && import.meta.url === `file://${process.argv[1]}`) {
  main(process.argv.slice(2));
}
