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
//
// ...EXCEPT FOR ONE RESIDUAL FAIL-OPEN, STATED RATHER THAN HIDDEN
//
// Everything above is conditioned on the construction being IN THE CRATE'S OWN `src/`, because that
// is all `crateSources` lists. The closed set is closed over occurrences the extractor SEES, and an
// edge appended by a shared builder living outside the crate is not one of them: the crate's sources
// contain no `additional_prerequisites` occurrence at all, so there is no unrecognised shape to
// throw on, and `deriveCrateRecord` returns `additionalPrerequisites: []` at exit 0. That is a
// silent zero of exactly the class this module exists to remove — reached by a route the "fails
// closed" argument above does not cover.
//
// It is not reachable at the pinned revision: every provider constructs its own contract in its own
// `src/`. It becomes reachable the moment a shared `gen_core` contract builder ships (sc-19591 is
// scheduled to add one), because a provider that adopts it stops declaring the field itself and
// silently loses its recorded edges without any derivation going red.
//
// Closing it needs the extractor to attribute a shared builder's appends back to each ADOPTING
// crate — resolving the builder's own edge set once and unioning it into every caller's record —
// which is a different derivation shape from "read this crate's own constructions", not a wider
// glob. Whoever lands that builder must land it here in the same change; a record set that
// silently drops to `[]` looks identical to a provider that genuinely declares no edges.

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
 * Every path qualifier a provider may write in front of these types.
 *
 * Providers reach gen-core three ways — directly, re-exported through `mlx_gen::gen_core`, or
 * through `candle_gen::gen_core` — and the same type therefore appears under three spellings. They
 * are erased before matching so the shape patterns stay single-form.
 */
function eraseCrateQualifiers(text) {
  return text.replace(/(?:crate::)?(?:mlx_gen|candle_gen|gen_core)::/g, "");
}

/**
 * Collapse whitespace and erase crate qualifiers, keeping a map from every surviving character back
 * to its offset in the BLANKED source.
 *
 * The shape patterns need the collapsed single-token form; the reset and conditionality rules need
 * to know WHERE in the file each occurrence sits (which function encloses it, whether a `#[cfg]`
 * gates it). Collapsing whitespace and erasing crate qualifiers both destroy offsets, so both are
 * done as deletions over a `(char, offset)` pair list rather than as string replaces.
 */
function canonicalWithOffsets(scan) {
  const chars = [];
  const offsets = [];
  for (let at = 0; at < scan.length; at += 1) {
    if (/\s/.test(scan[at])) continue;
    chars.push(scan[at]);
    offsets.push(at);
  }
  const dense = chars.join("");
  const keep = new Array(dense.length).fill(true);
  const qualifier = /(?:crate::)?(?:mlx_gen|candle_gen|gen_core)::/g;
  for (let found = qualifier.exec(dense); found; found = qualifier.exec(dense)) {
    for (let at = found.index; at < found.index + found[0].length; at += 1) keep[at] = false;
  }
  const text = [];
  const index = [];
  for (let at = 0; at < dense.length; at += 1) {
    if (!keep[at]) continue;
    text.push(dense[at]);
    index.push(offsets[at]);
  }
  return { text: text.join(""), index };
}

/** Index just past the `}` matching the `{` at `open`. Throws rather than running off the end. */
function matchBrace(scan, open) {
  let depth = 0;
  for (let at = open; at < scan.length; at += 1) {
    if (scan[at] === "{") depth += 1;
    else if (scan[at] === "}") {
      depth -= 1;
      if (depth === 0) return at + 1;
    }
  }
  throw new Error("unbalanced braces — the extractor cannot bound the item");
}

/** Index of the `)` matching the `(` at `open`. */
function matchParen(scan, open) {
  let depth = 0;
  for (let at = open; at < scan.length; at += 1) {
    if (scan[at] === "(") depth += 1;
    else if (scan[at] === ")") {
      depth -= 1;
      if (depth === 0) return at;
    }
  }
  throw new Error("unbalanced parentheses in a `#[cfg(…)]` predicate");
}

/**
 * Parse a `#[cfg(…)]` predicate into `{ name, children? }`.
 *
 * The grammar is tiny — `ident`, `ident = "value"`, `all(..)`, `any(..)`, `not(..)` — and anything
 * outside it throws. That is deliberate: the point of parsing at all is that a predicate the
 * extractor cannot classify must not be classified by accident.
 */
export function parseCfgPredicate(text) {
  let at = 0;
  const skip = () => {
    while (at < text.length && /\s/.test(text[at])) at += 1;
  };
  const fail = () =>
    new Error(`cannot parse cfg predicate at ${JSON.stringify(text.slice(at, at + 60))}`);
  const node = () => {
    skip();
    const name = /^[A-Za-z_]\w*/.exec(text.slice(at));
    if (!name) throw fail();
    at += name[0].length;
    skip();
    if (text[at] === "(") {
      const close = matchParen(text, at);
      const inner = text.slice(at + 1, close);
      at = close + 1;
      const children = inner.trim().length ? splitTopLevel(inner).map(parseCfgPredicate) : [];
      return { name: name[0], children };
    }
    if (text[at] === "=") {
      at += 1;
      skip();
      if (text[at] !== '"') throw fail();
      const end = text.indexOf('"', at + 1);
      if (end === -1) throw fail();
      at = end + 1;
      return { name: name[0], valued: true };
    }
    return { name: name[0] };
  };
  const root = node();
  skip();
  if (at !== text.length) throw fail();
  return root;
}

/** Split `a, all(b, c), d` on top-level commas, dropping a trailing empty element. */
function splitTopLevel(text) {
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let at = 0; at < text.length; at += 1) {
    const ch = text[at];
    if (ch === "(") depth += 1;
    else if (ch === ")") depth -= 1;
    else if (ch === "," && depth === 0) {
      parts.push(text.slice(start, at));
      start = at + 1;
    }
  }
  parts.push(text.slice(start));
  return parts.filter((part) => part.trim().length);
}

/**
 * What a `#[cfg]`-attributed item is worth in a PRODUCTION build, evaluated by substituting
 * `test := false` and reducing.
 *
 * - `false` — the item exists only under `cfg(test)`. It is not the provider's declaration, so it
 *   is stripped.
 * - `true` — the item is in every production build unconditionally. `#[cfg(not(test))]` lands here,
 *   which is the point: the previous substring regex `/#\[cfg\((?:[^)]*\()?[^)]*\btest\b[^\]]*\]/`
 *   MATCHED `#[cfg(not(test))]` and stripped the item as though it were a test module. That is
 *   backwards — a `#[cfg(not(test))] mod production { … push(…) }` genuinely ships, and stripping
 *   it derived `[]` from a file that pushes the rung-1 edge, at exit 0. The idiom is live at
 *   `candle-gen-krea/src/transformer/block.rs:142` and `mlx-gen-sana/src/transformer.rs:115`, and
 *   `candle-gen-catalog/src/lib.rs:227` documents it as one that "genuinely ships".
 * - `"conditional"` — the item ships in SOME production builds (a feature, a target, `any(test,
 *   feature = "testkit")`). Its constructions are recorded, and marked conditional so the record
 *   says so rather than presenting a feature-gated contract variant as the provider's unconditional
 *   declaration.
 *
 * A predicate the parser cannot read throws, so the fail-open direction — silently deciding an
 * unreadable `#[cfg]` is "not a test module, carry on" — is closed.
 */
export function cfgProductionState(predicate) {
  const reduce = (node) => {
    if (node.children) {
      const kids = node.children.map(reduce);
      if (node.name === "all") {
        if (kids.includes(false)) return false;
        return kids.every((kid) => kid === true) ? true : "conditional";
      }
      if (node.name === "any") {
        if (kids.includes(true)) return true;
        return kids.every((kid) => kid === false) ? false : "conditional";
      }
      if (node.name === "not") {
        if (kids.length !== 1) {
          throw new Error(`cfg \`not\` takes exactly one predicate, got ${kids.length}`);
        }
        return kids[0] === "conditional" ? "conditional" : !kids[0];
      }
      throw new Error(`unrecognised cfg combinator \`${node.name}\``);
    }
    // A production build sets no `test` cfg. Everything else — features, targets, `debug_assertions`
    // — depends on how the crate is built, which is exactly what "conditional" records.
    if (node.name === "test" && !node.valued) return false;
    return "conditional";
  };
  return reduce(parseCfgPredicate(predicate));
}

/**
 * The keywords that open a Rust ITEM or statement, as opposed to a struct field, an enum variant, a
 * match arm or a tuple element.
 *
 * `#[cfg]` attaches to both, and they end differently: an item ends at a top-level `;` or after its
 * brace-matched body, while a field/variant/arm ends at the next top-level `,` (or at the `}` / `)`
 * closing whatever contains it). `candle-gen-krea/src/quant.rs:577` is the live field case —
 * `#[cfg(feature = "cuda")] lt,` inside a struct literal — and reading it with the item rule walks
 * off the end of the file.
 */
const ITEM_KEYWORDS = new Set([
  "async",
  "const",
  "default",
  "enum",
  "extern",
  "fn",
  "impl",
  "let",
  "macro_rules",
  "mod",
  "pub",
  "static",
  "struct",
  "trait",
  "type",
  "union",
  "unsafe",
  "use",
]);

/** Where a `#[cfg]`-attributed item, field, variant or arm ends. */
function itemEnd(scan, from) {
  // Skip any further attributes, so `#[cfg(a)] #[inline] fn f()` is classified by `fn`.
  let at = from;
  for (;;) {
    while (at < scan.length && /\s/.test(scan[at])) at += 1;
    if (scan[at] !== "#") break;
    let bracket = scan.indexOf("[", at);
    if (bracket === -1) break;
    let depth = 0;
    for (; bracket < scan.length; bracket += 1) {
      if (scan[bracket] === "[") depth += 1;
      else if (scan[bracket] === "]") {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    at = bracket + 1;
  }
  const keyword = /^[A-Za-z_]\w*/.exec(scan.slice(at));
  const isItem = keyword !== null && ITEM_KEYWORDS.has(keyword[0]);
  let depth = 0;
  for (; at < scan.length; at += 1) {
    const ch = scan[at];
    if (ch === "(" || ch === "[") {
      depth += 1;
      continue;
    }
    if (ch === ")" || ch === "]") {
      if (depth === 0) return at;
      depth -= 1;
      continue;
    }
    if (depth > 0) continue;
    if (ch === ";") return at + 1;
    if (ch === "{") {
      const end = matchBrace(scan, at);
      if (!isItem) return end;
      // `use a::{b, c};` and `static X: S = S { … };` carry their terminator after the braces.
      let next = end;
      while (next < scan.length && /\s/.test(scan[next])) next += 1;
      return scan[next] === ";" ? next + 1 : end;
    }
    if (!isItem && (ch === "," || ch === "}")) return ch === "," ? at + 1 : at;
  }
  throw new Error("a `#[cfg(…)]` attribute is not followed by an item the extractor can bound");
}

/**
 * Every `#[cfg(…)]`-attributed item, with its extent and its production state.
 *
 * `scan` must be `blankLiterals` output: a `#[cfg(` inside a string or a doc comment would
 * otherwise be read as an attribute.
 */
export function cfgAttributedItems(scan) {
  const items = [];
  const attribute = /#\s*\[\s*cfg\s*\(/g;
  for (let found = attribute.exec(scan); found; found = attribute.exec(scan)) {
    const open = found.index + found[0].length - 1;
    const close = matchParen(scan, open);
    let at = close + 1;
    while (at < scan.length && /\s/.test(scan[at])) at += 1;
    if (scan[at] !== "]") {
      throw new Error(
        `a \`#[cfg(…)\` attribute does not close where the extractor expects: ${JSON.stringify(
          scan.slice(found.index, found.index + 80),
        )}`,
      );
    }
    items.push({
      start: found.index,
      end: itemEnd(scan, at + 1),
      state: cfgProductionState(scan.slice(open + 1, close)),
      predicate: scan.slice(open + 1, close).replace(/\s+/g, " ").trim(),
    });
    attribute.lastIndex = at + 1;
  }
  return items;
}

/**
 * Remove every item that exists ONLY under `cfg(test)`, brace-matched.
 *
 * Three regressions live in this one function's history, and each was a silent zero:
 *
 * 1. A first draft cut the file at the first `#[cfg(test)]` and kept the prefix.
 *    `mlx-gen-krea/src/memory_strategy.rs` carries a `#[cfg(test)] use` on line 13, so the
 *    "production" text was its first 820 of 30,249 bytes and the crate derived zero edges while its
 *    source pushes one.
 * 2. The replacement classified the attribute by SUBSTRING, so `#[cfg(not(test))]` matched and a
 *    `#[cfg(not(test))] mod production { … push(…) }` was stripped as a test module — see
 *    `cfgProductionState`.
 * 3. It only ever stripped a `mod`, so a `#[cfg(test)] fn build_contract() { … push(…) }` helper
 *    contributed its fixture's edges to the provider's record. Any item shape is stripped now.
 */
export function stripTestOnlyItems(source) {
  const scan = blankLiterals(source);
  const cuts = [];
  for (const item of cfgAttributedItems(scan)) {
    if (item.state !== false) continue;
    // Nested `#[cfg(test)]` inside an already-cut item would double-cut.
    if (cuts.some(([from, to]) => item.start >= from && item.end <= to)) continue;
    cuts.push([item.start, item.end]);
  }
  cuts.sort((left, right) => left[0] - right[0]);
  let kept = source;
  for (const [from, to] of cuts.reverse()) kept = kept.slice(0, from) + kept.slice(to);
  return kept;
}

/**
 * Retained under its old name so a caller that only wants test modules gone keeps working; the
 * behaviour is `stripTestOnlyItems`, which strips every test-only item and not just `mod`s.
 */
export const stripTestModules = stripTestOnlyItems;

/** Extent of every function body, innermost-last, for scoping the reset rule below. */
function functionExtents(scan) {
  const extents = [];
  const signature = /\bfn\s+\w+/g;
  for (let found = signature.exec(scan); found; found = signature.exec(scan)) {
    let depth = 0;
    let body = -1;
    for (let at = found.index + found[0].length; at < scan.length; at += 1) {
      const ch = scan[at];
      if (ch === "(" || ch === "[") depth += 1;
      else if (ch === ")" || ch === "]") depth -= 1;
      else if (depth === 0 && ch === ";") break;
      else if (depth === 0 && ch === "{") {
        body = at;
        break;
      }
    }
    // A trait or extern declaration has no body and encloses nothing.
    if (body === -1) continue;
    extents.push([found.index, matchBrace(scan, body)]);
  }
  return extents;
}

/**
 * Blank every comment, string, raw-string and char literal so brace counting cannot be thrown by a
 * `}` inside one. Lengths are preserved, so offsets into the result index the original.
 *
 * THIS IS THE ONLY PRIMITIVE THAT MAY SEE RAW SOURCE, and that is the whole correctness argument
 * for the extractor. An earlier version stripped comments with regexes over RAW source and claimed
 * the failure mode of getting it wrong "is a red derivation and not a wrong record". That claim was
 * FALSE, and the counterexample is two ordinary constants. One holds a glob whose slash-star OPENS
 * a block comment from inside a string literal (`"shards/<star>.safetensors"`); a later one holds a
 * ratio whose star-slash CLOSES it (`"peak<star>/base ratio"`). A block-comment regex then DELETES
 * everything between them — including a real
 * `.push((MemoryStrategy::BoundedTransformerResidency, …))`. The extractor sees no construction at
 * all, so it has nothing to fail closed ON: it derives `[]` and exits 0. A silent zero, which is
 * the exact defect class sc-19542 exists to remove. (Both spellings are written here as `<star>`
 * because this is itself a block comment; the executable form is in
 * `scripts/rung4-contract-prerequisites.test.mjs`.)
 *
 * The pass below is left-to-right and single, so a comment opener inside a literal and a literal
 * quote inside a comment are each consumed by whichever construct opened first — which is what a
 * pair of independent regexes cannot do. Every entry point takes this output rather than `source`
 * (`stripTestOnlyItems`, `additionalPrerequisiteEdges`, `stagedResidencySupport`), as do the
 * helpers they hand it to (`cfgAttributedItems`, `functionExtents`, `canonicalWithOffsets`);
 * handing any of them raw source reintroduces the silent zero above.
 */
export function blankLiterals(source) {
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
 * The strategies receiving a rung-1 edge from a crate's `additional_prerequisites` constructions.
 *
 * Returns `[{ strategy, conditional }]`. Five shapes are recognised, which is every shape present at
 * the pinned revision:
 *
 *   1. `additional_prerequisites: Vec::new()`                                   — no edges.
 *   2a. `additional_prerequisites: [A, B].into_iter().map(|s| (s, EDGE)).collect()` — field INIT.
 *   2b. `contract.additional_prerequisites = [A, B]…collect()`                  — ASSIGNMENT.
 *   3. `contract.additional_prerequisites.push((STRATEGY, EDGE))`
 *   4. `additional_prerequisites: COND.then_some((STRATEGY, EDGE)).into_iter().collect()`
 *   5. `= [A, B, X].into_iter().filter(|s| COND || *s != X).map(|s| (s, EDGE)).collect()`
 *
 * A read (`.is_empty()`, `.iter()`) is not a construction and contributes nothing. Anything else
 * throws.
 *
 * ASSIGNMENT IS A RESET, NOT AN APPEND
 *
 * 2b replaces the vector; 2a and 3 add to it. Treating them alike over-counts, and over-counting
 * keeps the rung-1 proxy on lanes the contract exempts — the exact defect sc-19542 exists to remove.
 * The live case is `candle-gen-krea/src/lib.rs`: `:1396` initialises the base contract with
 * `[BoundedDecode, BoundedAttention, BoundedTransformerResidency]` and `:1700` assigns
 * `[BoundedDecode, BoundedAttention]`, dropping rung 4.
 *
 * The reset is scoped to the enclosing FUNCTION, and that scoping is load-bearing rather than
 * decorative: those two sites are in different functions building different contracts
 * (`krea_2_turbo` and `krea_2_turbo_control`), so a file-flat reset would erase the base provider's
 * real edge — a silent zero reached by "fixing" an over-count. Occurrences are unioned across
 * functions, because a crate declaring several contract variants declares all of them and the
 * record's (family, backend) key cannot hold one per variant (see
 * `config/rung4-contract-prerequisites.json#description`).
 *
 * CONDITIONAL CONSTRUCTIONS ARE RECORDED AS CONDITIONAL
 *
 * Shape 4 is conditional by construction (`mlx-gen-anima/src/lib.rs:438` builds it "when
 * streamable"), and any construction inside a `#[cfg]`-gated item that is not in every production
 * build is conditional too (`candle-gen-krea/src/lib.rs:1627` is `#[cfg(any(feature = "cuda",
 * test))]`). Both are marked rather than presented as unconditional declarations. The GATE still
 * treats a conditional edge as present, because for an admission gate "this provider may demand
 * rung 1" is the fail-closed reading of a prerequisite that holds on some builds.
 *
 * The limit this cannot represent, stated rather than hidden: a `.push` inside an `if` BLOCK (as
 * opposed to a `then_some` expression) is recorded unconditionally. That is the over-restrictive
 * direction for admission, so it fails closed, but it is not detected.
 */
export function additionalPrerequisiteEdges(source, where) {
  const scan = blankLiterals(source);
  const conditionalRanges = cfgAttributedItems(scan)
    .filter((item) => item.state === "conditional")
    .map((item) => [item.start, item.end]);
  const testOnly = cfgAttributedItems(scan).find((item) => item.state === false);
  if (testOnly) {
    throw new Error(
      `${where}: a test-only item survived stripping (\`cfg(${testOnly.predicate})\`) — its ` +
        "constructions are fixtures, not the provider's declaration (sc-19542)",
    );
  }
  const scopes = functionExtents(scan);
  const { text, index } = canonicalWithOffsets(scan);

  // Per enclosing function (or the file, for a construction outside every function), in source
  // order. `Map` iteration is insertion-ordered, which is all the union below needs.
  const perScope = new Map();
  const bucketFor = (offset) => {
    let key = "<file>";
    let width = Infinity;
    for (const [from, to] of scopes) {
      if (offset < from || offset >= to) continue;
      if (to - from < width) {
        width = to - from;
        key = `${from}:${to}`;
      }
    }
    if (!perScope.has(key)) perScope.set(key, []);
    return perScope.get(key);
  };

  const field = "additional_prerequisites";
  for (let at = text.indexOf(field); at !== -1; at = text.indexOf(field, at + 1)) {
    const rest = text.slice(at + field.length);
    // A read, not a construction.
    if (/^\.(?:is_empty|iter|len|contains|first|last)\b/.test(rest)) continue;

    const offset = index[at];
    const gated = conditionalRanges.some(([from, to]) => offset >= from && offset < to);
    const bucket = bucketFor(offset);
    const add = (strategies, { conditional, resets }) => {
      if (resets) bucket.length = 0;
      for (const strategy of strategies) bucket.push({ strategy, conditional: conditional || gated });
    };

    const vecNew = /^\s*([:=])Vec::new\(\)/.exec(rest);
    if (vecNew) {
      // `= Vec::new()` empties the vector; `: Vec::new()` initialises it empty. Both leave nothing.
      add([], { conditional: false, resets: vecNew[1] === "=" });
      continue;
    }

    const arrayMap =
      /^\s*([:=])\[((?:MemoryStrategy::\w+,?)+)\]\.into_iter\(\)\.map\(\|(\w+)\|\{?\((\w+),(MemoryStrategyPrerequisite::Rung\{[^}]*\},?)\)\}?\)\.collect\(\)/.exec(
        rest,
      );
    if (arrayMap) {
      const [, operator, list, binder, applied, edge] = arrayMap;
      if (binder !== applied) {
        throw new Error(`${where}: ${field} maps ${binder} but pairs ${applied}`);
      }
      assertRecognisedEdge(edge, where);
      add(list.split(",").filter(Boolean).map(stripStrategy), {
        conditional: false,
        resets: operator === "=",
      });
      continue;
    }

    // SHAPE 5 (sc-20799, inference ebcdc7da). `candle-gen-sana` declares ONE array of rung
    // candidates and removes the members this build cannot execute, instead of branching the whole
    // construction. Exactly one predicate is recognised — `COND || *binder != MemoryStrategy::X`,
    // which keeps every member and drops X unless COND holds — so X is recorded CONDITIONAL and the
    // rest unconditional. Any other predicate throws: a filter the extractor cannot read must not
    // silently become "keeps everything", which is the same fail-open direction as reading an
    // unknown construction as "no edges". Checked before the plain `.into_iter().map` shape only for
    // readability; that regex requires `.map` immediately after `.into_iter()` and cannot match this.
    const arrayFilterMap =
      /^\s*([:=])\[((?:MemoryStrategy::\w+,?)+)\]\.into_iter\(\)\.filter\(\|(\w+)\|(\w+)\|\|\*(\w+)!=MemoryStrategy::(\w+)\)\.map\(\|(\w+)\|\{?\((\w+),(MemoryStrategyPrerequisite::Rung\{[^}]*\},?)\)\}?\)\.collect\(\)/.exec(
        rest,
      );
    if (arrayFilterMap) {
      const [
        ,
        operator,
        list,
        filterBinder,
        ,
        derefBinder,
        excluded,
        mapBinder,
        applied,
        edge,
      ] = arrayFilterMap;
      if (filterBinder !== derefBinder) {
        throw new Error(
          `${where}: ${field} binds ${filterBinder} in its filter but compares ${derefBinder}`,
        );
      }
      if (mapBinder !== applied) {
        throw new Error(`${where}: ${field} maps ${mapBinder} but pairs ${applied}`);
      }
      assertRecognisedEdge(edge, where);
      const members = list.split(",").filter(Boolean).map(stripStrategy);
      if (!members.includes(excluded)) {
        throw new Error(
          `${where}: ${field} filters out MemoryStrategy::${excluded}, which the mapped array does ` +
            "not contain — the extractor cannot say which member the predicate governs",
        );
      }
      if (operator === "=") bucket.length = 0;
      for (const strategy of members) {
        bucket.push({ strategy, conditional: gated || strategy === excluded });
      }
      continue;
    }

    const push =
      /^\.push\(\(MemoryStrategy::(\w+),(MemoryStrategyPrerequisite::Rung\{[^}]*\},?),?\)\)/.exec(
        rest,
      );
    if (push) {
      assertRecognisedEdge(push[2], where);
      add([push[1]], { conditional: false, resets: false });
      continue;
    }

    const thenSome =
      /^\s*([:=])[\w.()]+\.then_some\(\(MemoryStrategy::(\w+),(MemoryStrategyPrerequisite::Rung\{[^}]*\},?),?\)\)\.into_iter\(\)\.collect\(\)/.exec(
        rest,
      );
    if (thenSome) {
      assertRecognisedEdge(thenSome[3], where);
      add([thenSome[2]], { conditional: true, resets: thenSome[1] === "=" });
      continue;
    }

    throw new Error(
      `${where}: unrecognised \`${field}\` construction at ${JSON.stringify(rest.slice(0, 140))}. ` +
        "The extractor recognises a closed set of shapes and fails closed on anything else, because " +
        "reading a new shape as “no edges” is the fail-open direction this record exists to remove (sc-19542)",
    );
  }
  return [...perScope.values()].flat();
}

/**
 * `MemoryProviderContract::support(MemoryStrategy::StagedResidency)` as this crate declares it, or
 * `null` when the extractor cannot read it unambiguously.
 *
 * gen-core's `validate_selection` has TWO accepting arms for a `Rung { .. EngagedInSameRequest }`
 * edge, not one — `crates/contracts/gen-core/src/memory_strategy.rs:1862-1873` at the pinned
 * `f17c82544`. The first is `engages`; the second is that the prerequisite rung is declared
 * `StructurallyNotApplicable`, which satisfies the edge VACUOUSLY ("the architecture has no such
 * component to shed"). The gate needs this to implement the second arm.
 *
 * `null` is the fail-CLOSED answer, and the asymmetry is deliberate: the N/A arm ADMITS, so a wrong
 * positive would admit a cell the contract refuses, while a wrong `null` merely falls back to the
 * `engages` term. So this reports `structurally_not_applicable` only when every unguarded match arm
 * assigning `StagedResidency` a support in the crate agrees on it, and `null` on disagreement, on a
 * guarded arm (`MemoryStrategy::StagedResidency if streamable =>`), or on an arm body the extractor
 * cannot read down to a variant.
 */
export function stagedResidencySupport(source) {
  const arms =
    /MemoryStrategy::StagedResidency((?:\|MemoryStrategy::\w+)*)( if [^=]*?)?=>\s*\{?\s*(?:MemoryStrategySupport::)?(\w+)/g;
  const normalized = eraseCrateQualifiers(blankLiterals(source))
    .replace(/\s+/g, " ")
    .replace(/\s*::\s*/g, "::")
    .replace(/\s*=>\s*/g, "=>")
    .replace(/\s*\|\s*/g, "|");
  const declared = new Set();
  for (let found = arms.exec(normalized); found; found = arms.exec(normalized)) {
    // A guarded arm is one of several, so it does not decide the provider's declaration on its own.
    if (found[2]) return null;
    declared.add(found[3]);
  }
  if (declared.size !== 1) return null;
  const [only] = [...declared];
  return SUPPORT_SPELLINGS[only] ?? null;
}

/** gen-core's `MemoryStrategySupport` variants, in this module's snake-case record spelling. */
const SUPPORT_SPELLINGS = Object.freeze({
  Implemented: "implemented",
  Missing: "missing",
  StructurallyNotApplicable: "structurally_not_applicable",
});

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

/**
 * Non-test Rust sources of one crate at `revision`.
 *
 * `<crate>/src/` ONLY, which bounds what every guard downstream can possibly see: a construction
 * outside this list is not "unrecognised", it is invisible, and invisible reads as `[]` at exit 0.
 * See the residual fail-open in the module header — a shared out-of-crate contract builder is the
 * case that makes this bite, and it must be taught to this module in the same change that ships it.
 */
function crateSources(repo, revision, cratePath) {
  const listed = git(repo, ["ls-tree", "-r", "--name-only", revision, "--", `${cratePath}/src/`]);
  return listed
    .split("\n")
    .filter((line) => line.endsWith(".rs"))
    .sort();
}

/**
 * Derive one (family, backend) record from the pinned tree: its rung-4 additive edges, and the
 * support it declares for the rung those edges name.
 *
 * Scoped to the crate's `src/` because that is the production contract; `tests/` asserts ABOUT the
 * contract (`assert!(contract.additional_prerequisites.is_empty())`) and constructing a record from
 * a test fixture would record the fixture rather than the provider.
 */
export function deriveCrateRecord(repo, revision, cratePath) {
  const sources = crateSources(repo, revision, cratePath);
  if (!sources.length) {
    throw new Error(
      `${cratePath}: no Rust sources at ${revision.slice(0, 9)} — the record names a crate the ` +
        "pinned revision does not contain (sc-19542)",
    );
  }
  const additionalPrerequisites = [];
  const supports = new Set();
  for (const source of sources) {
    const body = git(repo, ["show", `${revision}:${source}`]);
    // Test-only items build contracts to assert about them; those are not the provider's
    // declaration.
    const production = stripTestOnlyItems(body);
    for (const edge of additionalPrerequisiteEdges(production, source)) {
      if (edge.strategy !== RUNG4) continue;
      additionalPrerequisites.push({
        ...STAGED_RESIDENCY_ENGAGED_IN_SAME_REQUEST,
        source,
        conditional: edge.conditional,
      });
    }
    const support = stagedResidencySupport(production);
    if (support) supports.add(support);
  }
  // One crate, one declaration. Two files disagreeing is exactly the ambiguity the N/A arm must not
  // resolve on its own, so it becomes `null` and the gate falls back to the `engages` term.
  return {
    additionalPrerequisites,
    stagedResidencySupport: supports.size === 1 ? [...supports][0] : null,
  };
}

export function inferencePin(cargoToml) {
  const match =
    /candle-kernels\s*=\s*\{[^}]*?github\.com\/SceneWorks\/inference[^}]*?rev\s*=\s*"([0-9a-f]+)"/.exec(
      cargoToml,
    );
  if (!match) throw new Error("no inference pin in Cargo.toml");
  return match[1];
}

/**
 * Re-derive every record and return the disagreements, as `[where, expected, actual]`.
 *
 * Every field a record carries is compared, `conditional` and `stagedResidencySupport` included: a
 * field the comparison skips is a field `--check` cannot grade, and both of these decide admission.
 */
export function compareRecords(records, repo) {
  const failures = [];
  const shape = (record) =>
    JSON.stringify({
      additionalPrerequisites: (record.additionalPrerequisites ?? []).map(
        ({ kind, rung, scope, source, conditional }) => ({
          kind,
          rung,
          scope,
          source,
          conditional: conditional === true,
        }),
      ),
      stagedResidencySupport: record.stagedResidencySupport ?? null,
    });
  for (const [group, family] of Object.entries(records.families)) {
    for (const [backend, record] of Object.entries(family.backends)) {
      const where = `${family.name} (${backend})`;
      const derived = deriveCrateRecord(repo, records.inferenceRevision, record.crate);
      if (shape(derived) !== shape(record)) {
        failures.push([`${group} ${where}`, shape(derived), shape(record)]);
      }
    }
  }
  return failures;
}

function readRecords() {
  return JSON.parse(readFileSync(path.join(repoRoot, RECORDS_PATH), "utf8"));
}

/** `[{strategy, conditional}]` -> the compact form the self-test asserts against. */
const spell = (edges) =>
  edges.map(({ strategy, conditional }) => (conditional ? `${strategy}?` : strategy));

function expect(name, actual, expected) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(
      `self-test ${name}: expected ${JSON.stringify(expected)} got ${JSON.stringify(actual)}`,
    );
  }
}

function expectThrows(name, run) {
  let threw = false;
  try {
    run();
  } catch {
    threw = true;
  }
  if (!threw) throw new Error(`self-test ${name}: must fail closed`);
}

const EDGE = `MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            }`;

function selfTest() {
  expect(
    "empty vec",
    spell(additionalPrerequisiteEdges("additional_prerequisites: Vec::new(),", "empty")),
    [],
  );
  expect(
    "field init",
    spell(
      additionalPrerequisiteEdges(
        `additional_prerequisites: [
             MemoryStrategy::BoundedDecode,
             MemoryStrategy::BoundedTransformerResidency,
         ]
         .into_iter()
         .map(|strategy| { (strategy, ${EDGE}) })
         .collect(),`,
        "field init",
      ),
    ),
    ["BoundedDecode", "BoundedTransformerResidency"],
  );
  expect(
    "push appends",
    spell(
      additionalPrerequisiteEdges(
        `contract.additional_prerequisites.push((MemoryStrategy::BoundedTransformerResidency, ${EDGE}));`,
        "push",
      ),
    ),
    ["BoundedTransformerResidency"],
  );
  expect(
    "then_some is conditional",
    spell(
      additionalPrerequisiteEdges(
        `additional_prerequisites: streamable
             .then_some((MemoryStrategy::BoundedTransformerResidency, ${EDGE}))
             .into_iter()
             .collect(),`,
        "then_some",
      ),
    ),
    ["BoundedTransformerResidency?"],
  );
  expect(
    "a read is not a construction",
    spell(
      additionalPrerequisiteEdges("assert!(contract.additional_prerequisites.is_empty());", "read"),
    ),
    [],
  );

  // `=` REPLACES the vector; `:` and `.push` add to it. Same scope, so the assignment wins.
  expect(
    "assignment resets within one scope",
    spell(
      additionalPrerequisiteEdges(
        `fn contract() -> MemoryProviderContract {
             contract.additional_prerequisites.push((MemoryStrategy::BoundedTransformerResidency, ${EDGE}));
             contract.additional_prerequisites = [MemoryStrategy::BoundedDecode]
                 .into_iter()
                 .map(|strategy| { (strategy, ${EDGE}) })
                 .collect();
         }`,
        "reset",
      ),
    ),
    ["BoundedDecode"],
  );
  // ...but only within one scope: two functions build two contracts, and the crate declares both.
  expect(
    "assignment in another function does not erase the first",
    spell(
      additionalPrerequisiteEdges(
        `fn base() -> MemoryProviderContract {
             contract.additional_prerequisites.push((MemoryStrategy::BoundedTransformerResidency, ${EDGE}));
         }
         fn control() -> MemoryProviderContract {
             contract.additional_prerequisites = [MemoryStrategy::BoundedDecode]
                 .into_iter()
                 .map(|strategy| { (strategy, ${EDGE}) })
                 .collect();
         }`,
        "two scopes",
      ),
    ),
    ["BoundedTransformerResidency", "BoundedDecode"],
  );
  // A `#[cfg]`-gated item ships on some builds only, so its constructions are conditional.
  expect(
    "a cfg-gated construction is conditional",
    spell(
      additionalPrerequisiteEdges(
        `#[cfg(any(feature = "cuda", test))]
         fn control() -> MemoryProviderContract {
             contract.additional_prerequisites.push((MemoryStrategy::BoundedTransformerResidency, ${EDGE}));
         }`,
        "cfg-gated",
      ),
    ),
    ["BoundedTransformerResidency?"],
  );

  // BLOCKER 1's reproduction. `#[cfg(not(test))]` genuinely ships — the previous substring regex
  // matched it, stripped the module, and derived `[]` at exit 0 from a file that pushes the edge.
  const notTest = `
#[cfg(not(test))]
mod production {
    pub fn contract() -> MemoryProviderContract {
        contract.additional_prerequisites.push((MemoryStrategy::BoundedTransformerResidency, ${EDGE}));
    }
}
`;
  expect(
    "cfg(not(test)) ships and its edge survives",
    spell(additionalPrerequisiteEdges(stripTestOnlyItems(notTest), "not(test)")),
    ["BoundedTransformerResidency"],
  );

  // BLOCKER 2's reproduction. Comment stripping over RAW source let a `/*` in one string literal and
  // a `*/` in a later one delete the construction between them, at exit 0.
  const literalTrap = `
const SHARD_GLOB: &str = "shards/*.safetensors";
pub fn contract() -> MemoryProviderContract {
    contract.additional_prerequisites.push((MemoryStrategy::BoundedTransformerResidency, ${EDGE}));
}
const RATIO_DOC: &str = "peak*/base ratio";
`;
  expect(
    "a block-comment opener inside a literal cannot delete a construction",
    spell(additionalPrerequisiteEdges(literalTrap, "literal trap")),
    ["BoundedTransformerResidency"],
  );

  // Test-only items of ANY shape are the fixture, not the declaration.
  const testOnlyHelper = `
#[cfg(test)]
use gen_core::MemoryGeometry;

pub fn contract() -> MemoryProviderContract {
    contract.additional_prerequisites.push((MemoryStrategy::BoundedTransformerResidency, ${EDGE}));
}

#[cfg(test)]
fn fixture() -> MemoryProviderContract {
    contract.additional_prerequisites.push((MemoryStrategy::BoundedDecode, ${EDGE}));
}

#[cfg(test)]
mod tests {
    #[test]
    fn brace_in_a_literal_does_not_end_the_module() {
        assert_eq!(render("}"), "}");
        contract.additional_prerequisites.push((MemoryStrategy::BoundedAttention, ${EDGE}));
    }
}
`;
  expect(
    "test-only items are stripped whatever shape they take",
    spell(additionalPrerequisiteEdges(stripTestOnlyItems(testOnlyHelper), "test-only")),
    ["BoundedTransformerResidency"],
  );
  // Non-vacuity: the fixture really does carry two edges INSIDE test-only items, so the assertion
  // above is graded by the stripping and not by there being one construction to find.
  expect(
    "…and the fixture really carries the edges that stripping removes",
    ["BoundedDecode", "BoundedAttention"].map((strategy) => [
      testOnlyHelper.includes(strategy),
      stripTestOnlyItems(testOnlyHelper).includes(strategy),
    ]),
    [
      [true, false],
      [true, false],
    ],
  );
  // A caller that forgets to strip does not get a polluted record; it gets an error.
  expectThrows("an unstripped test-only item", () =>
    additionalPrerequisiteEdges(testOnlyHelper, "unstripped"),
  );

  expect("cfg(test) is test-only", cfgProductionState("test"), false);
  expect("cfg(not(test)) is unconditional", cfgProductionState("not(test)"), true);
  expect(
    "cfg(any(feature, test)) is conditional",
    cfgProductionState('any(feature = "cuda", test)'),
    "conditional",
  );

  expect(
    "an unguarded N/A arm is read",
    stagedResidencySupport(
      `match capability.strategy {
           MemoryStrategy::StagedResidency => MemoryStrategySupport::StructurallyNotApplicable {
               reason: "one flat fused checkpoint".to_owned(),
           },
       }`,
    ),
    "structurally_not_applicable",
  );
  expect(
    "a guarded arm is not a declaration this gate may act on",
    stagedResidencySupport(
      `match capability.strategy {
           MemoryStrategy::StagedResidency if streamable => MemoryStrategySupport::StructurallyNotApplicable {
               reason: "…".to_owned(),
           },
       }`,
    ),
    null,
  );
  expect(
    "disagreeing arms read as unknown rather than as N/A",
    stagedResidencySupport(
      `MemoryStrategy::StagedResidency => MemoryStrategySupport::StructurallyNotApplicable { reason: "".to_owned() },
       MemoryStrategy::StagedResidency => MemoryStrategySupport::Implemented,`,
    ),
    null,
  );

  expectThrows("an unrecognised construction", () =>
    additionalPrerequisiteEdges("additional_prerequisites: build_them(spec),", "novel shape"),
  );
  expectThrows("an unrecognised edge variant", () =>
    additionalPrerequisiteEdges(
      `contract.additional_prerequisites.push((
           MemoryStrategy::BoundedTransformerResidency,
           MemoryStrategyPrerequisite::LoadShape(LoadShape::DeferredMaterialization),
       ));`,
      "novel edge",
    ),
  );
  expectThrows("an unparseable cfg predicate", () => cfgProductionState("weird | thing"));
  process.stdout.write("rung4-contract-prerequisites self-test: ok\n");
}

function main(argv) {
  if (argv.includes("--self-test")) {
    selfTest();
    return;
  }
  const records = readRecords();
  const pin = inferencePin(readFileSync(path.join(repoRoot, "Cargo.toml"), "utf8"));
  const repoIndex = argv.indexOf("--repo");
  if (repoIndex === -1) {
    process.stderr.write(
      "usage: node scripts/rung4-contract-prerequisites.mjs --repo <inference-checkout> [--check|--write]\n" +
        "       node scripts/rung4-contract-prerequisites.mjs --self-test\n",
    );
    process.exit(2);
  }
  const repo = path.resolve(argv[repoIndex + 1]);
  // `--write` is handled BEFORE the staleness guard below, and re-keys the records to the current
  // pin. Both halves are required for a pin bump to be resolvable at all: the guard used to run
  // first and exit 1 while telling the operator to run `--write`, which the guard itself made
  // unreachable — and `--write` only ever rewrote the derived fields, never `inferenceRevision`, so
  // even reaching it left the records keyed to the old revision and the guard still firing. The
  // first pin bump after these records landed (sc-17137's main sync, 014134e30 → 0e4adc6f0) is what
  // exposed it; re-deriving is the whole point of the flag, so it must not be gated on the records
  // already agreeing with the pin.
  if (argv.includes("--write")) {
    for (const family of Object.values(records.families)) {
      for (const [backend, record] of Object.entries(family.backends)) {
        void backend;
        const derived = deriveCrateRecord(repo, pin, record.crate);
        record.additionalPrerequisites = derived.additionalPrerequisites;
        record.stagedResidencySupport = derived.stagedResidencySupport;
      }
    }
    records.inferenceRevision = pin;
    writeFileSync(path.join(repoRoot, RECORDS_PATH), `${JSON.stringify(records, null, 2)}\n`);
    process.stdout.write(`wrote ${RECORDS_PATH} keyed to ${pin.slice(0, 9)}\n`);
    return;
  }
  if (records.inferenceRevision !== pin) {
    process.stderr.write(
      `${RECORDS_PATH} is keyed to ${records.inferenceRevision?.slice(0, 9) ?? "(unset)"} but ` +
        `Cargo pins ${pin.slice(0, 9)}. Re-run: node scripts/rung4-contract-prerequisites.mjs ` +
        "--repo <inference> --write\n",
    );
    process.exit(1);
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
