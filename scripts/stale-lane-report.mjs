#!/usr/bin/env node

/**
 * sc-18098 — the stale-lane batch report (epic 18093 R3).
 *
 * WHAT THIS IS FOR
 *
 * Measurement currency used to be treated as a correctness gate: a lane whose provider compile
 * closure had moved was DEMOTED, and the only remedy was a re-capture. Epic 18093 retired that.
 * `crates/sceneworks-worker/src/ladder_margin_policy.rs` (sc-18095/18096/18097) keeps stale-closure
 * measured evidence ELIGIBLE, serving its measured numbers behind a widened admission margin, and
 * admits estimate-backed candidates behind a wider one still. Currency is therefore a SIGNAL about
 * how much conservatism the runtime is currently buying, not a per-PR obligation.
 *
 * A signal needs somewhere to be read. That is this script: an on-demand batch view of which lanes
 * are stale, how much of the corpus and of the shipped admission surface each one covers, and the
 * margin widening the runtime is applying to it right now. Nothing here runs in `npm run check`,
 * `npm run rust:check`, the pre-push hook or CI — wiring it into a gate would rebuild exactly the
 * per-PR pressure R3 exists to remove.
 *
 * WHAT "STALE" MEANS HERE — the same predicate the runtime uses, not a lookalike
 *
 * A lane is `<backend>:<provider>`, keyed exactly as `config/inference-provider-closures.json`,
 * `memory-calibration-harness.mjs#evidenceSemantics` and `generate-memory-matrix.mjs#closureIsCurrent`
 * key it. A record or manifest calibration binding is stale when the closure digest it was captured
 * under differs from the live digest for its own lane. The live table is loaded through
 * `validatedInferenceClosures` (which moved here from the matrix generator at sc-22513), so the report and the
 * matrix cannot disagree about which lanes are current.
 *
 * TWO POPULATIONS, KEPT SEPARATE
 *
 *   RECORDS  — `docs/generated/memory-calibration-evidence.json`. The measurement corpus. What a
 *              re-capture would have to reproduce.
 *   BINDINGS — every `inferenceClosureDigest` in `config/manifests/builtin.models.jsonc`. The
 *              SHIPPED admission surface: these are what the worker's fit gates actually consult, so
 *              a stale binding is a production decision running under a widened margin today, while
 *              a stale record is only corpus debt. They are ranked in that order for that reason.
 *
 * THE BINDING POPULATION IS THE LOCATOR'S, NOT THIS FILE'S
 *
 * The first version of this report walked `<model>.<backend>.calibrations[]` and called it the
 * admission surface. That is 31 of the manifest's 33 bindings. The missing two are `turboFit` and
 * `candle.control` — whole-block fits that sit OUTSIDE `calibrations[]`, are read by
 * `vram_gate.rs` and `krea_control_fit.rs`, and are exactly the pair sc-17989 dragged back under the
 * closure gate after they had sat outside it for the whole of sc-17774. Re-deriving the population
 * reopened that hole in a new file: `candle:krea_2_turbo` and `candle:krea_2_turbo_control` printed
 * as "declared but never captured" while both were stale production bindings.
 *
 * So the population comes from `backfill-closure-digests.mjs#stampManifest` — the CI gate's own
 * locator, brace-depth accurate and orphan-checked — and this file refuses to run if that locator
 * reports a skipped or orphaned digest. Model attribution (the MODELS column) is read off the
 * parsed manifest and cross-checked against the locator's per-lane counts, so the cosmetic walk
 * cannot quietly disagree with the authoritative one.
 *
 * MARGIN: DERIVED, NEVER RESTATED
 *
 * The widening column is `recaptureSpread` from `scripts/derive-ladder-margins.mjs`, computed
 * over the same evidence corpus this report reads. That module's constants are pinned against
 * `crates/sceneworks-worker/src/ladder_margin_policy.rs` by `scripts/derive-ladder-margins.test.mjs`,
 * so the number printed here is the number the runtime applies, and a drift on either side reds that
 * test. No margin literal appears in this file.
 *
 * Run: `node scripts/stale-lane-report.mjs [--json]`  (`npm run report:stale-lanes`)
 */

import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { recordsNeedingDigest, stampManifest } from "./backfill-closure-digests.mjs";
import { deriveMargins } from "./derive-ladder-margins.mjs";
import { inferencePinFromCargo } from "./inference-closure-digest.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export const SOURCE_PATHS = Object.freeze({
  closures: "config/inference-provider-closures.json",
  evidence: "docs/generated/memory-calibration-evidence.json",
  manifest: "config/manifests/builtin.models.jsonc",
  cargo: "Cargo.toml",
  plan: "config/memory-calibration-plan.json",
  mlxAdapter: "crates/sceneworks-memory-adapter/src/bin/mlx.rs",
  candleAdapter: "crates/sceneworks-memory-adapter/src/bin/candle.rs",
});

/**
 * The provider closure ledger, validated (sc-17774).
 *
 * Lived in `scripts/generate-memory-matrix.mjs` until sc-22513 collapsed the matrix onto the anchor
 * store — the ledger is no longer a matrix input, and this report is its only remaining consumer, so
 * the predicate moved here rather than being re-implemented or left exported from a module that does
 * not use it. Unchanged in behaviour: it fails closed on a ledger with no revision, an unusable
 * digest, or no providers at all.
 */
export function validatedInferenceClosures(body) {
  const closures = JSON.parse(body);
  if (!/^[0-9a-f]{40}$/.test(closures.inferenceRevision ?? "")) {
    throw new Error(
      "config/inference-provider-closures.json must record the full inference revision its " +
        "digests were derived at",
    );
  }
  const digests = new Map();
  for (const [provider, entry] of Object.entries(closures.providers ?? {})) {
    if (!/^[0-9a-f]{64}$/.test(entry.digest ?? "")) {
      throw new Error(`inference closure entry for ${provider} has no usable digest`);
    }
    digests.set(provider, entry.digest);
  }
  // sc-22512: an EMPTY ledger no longer throws. A repo that declares no provider closures is an
  // unmeasured repo, not a broken one — every binding then reads as not-current, which is the
  // conservative estimate. The two throws above stay: they red on data that IS present and is
  // malformed (no 40-hex derivation revision; an entry with no usable 64-hex digest). Carried across
  // sc-22513, which moved this function here out of the memory-matrix generator.
  return digests;
}

/** Provenance for the margin column, printed so a reader can check it rather than trust it. */
export const MARGIN_SOURCE =
  "scripts/derive-ladder-margins.mjs#recaptureSpread (pinned against " +
  "crates/sceneworks-worker/src/ladder_margin_policy.rs by scripts/derive-ladder-margins.test.mjs)";

/**
 * Provenance for the CAPTURE column (sc-18212), printed like {@link MARGIN_SOURCE} so a reader can
 * check the derivation rather than trust it.
 */
export const CAPTURABILITY_SOURCE =
  "provider-dispatch match arms parsed from crates/sceneworks-memory-adapter/src/bin/{mlx,candle}.rs" +
  ' — every match block able to refuse with "five-rung calibration does not implement provider"; a' +
  " provider is capturable only if EVERY such dispatch admits it";

export function laneOf(backend, provider) {
  return `${backend}:${provider}`;
}

/**
 * The bounded flagship MLX T2I population (SC-18377/SC-18379), derived from the shipped manifest.
 * A declared `memoryStrategyContract.provider` is authoritative. If the whole contract is absent,
 * the catalog id is the only available candidate identity and deliberately remains visible as an
 * uncovered lane. That fallback is what makes the exact pre-SC-18377 Krea omission detectable.
 */
export function recommendedMlxT2iLanes(manifest) {
  return (manifest?.models ?? [])
    .filter(
      (model) =>
        model.recommended === true &&
        model.type === "image" &&
        model.capabilities?.includes("text_to_image") &&
        model.mlx &&
        typeof model.id === "string",
    )
    .map((model) => {
      const declaredProvider = model.mlx?.memoryStrategyContract?.provider;
      const contractDeclared = typeof declaredProvider === "string";
      const provider = contractDeclared ? declaredProvider : model.id;
      return {
        modelId: model.id,
        provider,
        lane: laneOf("mlx", provider),
        contractDeclared,
        providerSource: contractDeclared ? "memoryStrategyContract.provider" : "model.id fallback",
      };
    })
    .sort((left, right) => left.modelId.localeCompare(right.modelId));
}

/*
 * CAPTURABILITY — derived from the adapter's own dispatch, never from a hand list (sc-18212)
 *
 * `candle:z_image` is declared in the closure table and carries 90 plan entries, yet
 * `crates/sceneworks-memory-adapter/src/bin/candle.rs` has no arm for it — no invocation of the
 * adapter can ever capture the lane. The first version of this report printed it under "declared but
 * never captured", which reads as pending measurement work: an operator following §3 of
 * docs/calibration-runbook.md would book a CUDA box for a capture that fails by design. The sc-18104
 * screening found the same declaration/reachability split on four candle lanes and one planned-but-
 * undeclared lane (`candle:qwen_image_edit`).
 *
 * The fix is a "capturable" signal derived from the SAME source of truth that decides whether the
 * adapter's `run()` can serve a provider: the dispatch `match` arms in the adapter binaries. Both
 * adapters refuse an unimplemented provider by name (sc-18104) with the shared phrase
 * "five-rung calibration does not implement provider", so every match block that can emit that
 * refusal IS a dispatch gate, and the union of its non-fallback arms is the provider set it admits.
 * A provider counts as capturable only if every such gate admits it (the candle adapter has two —
 * entry dispatch and generator loading — and a provider missing from either cannot complete a
 * capture). The one exception is a BESPOKE PRE-GATE (sc-22736): an arm the run entry routes to
 * BEFORE the shared gates — `if provider == LTX25_ID { return run_ltx25_capture(request); }`, or
 * `if matches!(provider, A | B) { return module::run(request); }` — never reaches those gates at
 * all, so the providers it names are capturable on that arm alone.
 *
 * sc-22737: the pre-gate scan is now EXHAUSTIVE over the run entry's pre-gate region rather than a
 * best-effort regex sweep. Ignoring a guard shape it did not recognize was silent and wrong in both
 * directions at once — four working candle lanes (`ltx_2_3_distilled`, `minimax_h3`, and the
 * SenseNova pair) read "uncapturable" because their guards spell `provider == A || provider == B`
 * and `if let Some(arm) = sc22737_video_arm(request)?`, neither of which the old sweep matched. An
 * unrecognized guard now THROWS: it is the only way an operator finds out the parser has fallen
 * behind the adapter instead of being told a shipped arm does not exist. `bespokePreGateProviders`
 * lists the four shapes read, including how the `if let` form follows the arm table to its ids.
 * Parsing the adapter source is the same discipline `generate-memory-matrix.mjs` applies
 * to `image_jobs/base.rs`: a hand-maintained provider list here would be a new false green, going
 * stale the day an arm is added or retired. Every anchor below throws rather than degrades — a
 * refactor that moves the dispatch out of reach must red the tests, not silently report nothing
 * uncapturable.
 */

const DISPATCH_REFUSAL = "five-rung calibration does not implement provider";

/** Skip a `"…"`, `r#"…"#` or `'c'` literal starting at `index`; returns the index after it. */
function skipStringLike(text, index) {
  const char = text[index];
  if (char === '"') {
    let i = index + 1;
    while (i < text.length) {
      if (text[i] === "\\") i += 2;
      else if (text[i] === '"') return i + 1;
      else i += 1;
    }
    return i;
  }
  if (char === "r" && !/[A-Za-z0-9_]/.test(text[index - 1] ?? " ")) {
    const raw = /^r(#*)"/.exec(text.slice(index, index + 8));
    if (raw) {
      const close = `"${raw[1]}`;
      const end = text.indexOf(close, index + raw[0].length);
      return end === -1 ? text.length : end + close.length;
    }
  }
  if (char === "'") {
    // A char literal, not a lifetime: `'{'` must not unbalance a brace scan, `'static` must not
    // swallow everything to the next apostrophe.
    const literal = /^'(\\.|[^'\\])'/.exec(text.slice(index, index + 8));
    if (literal) return index + literal[0].length;
  }
  return index;
}

/** Rust source with line comments and (nested) block comments removed; string literals preserved. */
function stripRustComments(source) {
  let out = "";
  let i = 0;
  while (i < source.length) {
    // Strings first: a `//` inside a string literal (a URL, a glob) is not a comment.
    const skipped = skipStringLike(source, i);
    if (skipped !== i) {
      out += source.slice(i, skipped);
      i = skipped;
      continue;
    }
    if (source.startsWith("//", i)) {
      while (i < source.length && source[i] !== "\n") i += 1;
      continue;
    }
    if (source.startsWith("/*", i)) {
      let depth = 1;
      i += 2;
      while (i < source.length && depth > 0) {
        if (source.startsWith("/*", i)) {
          depth += 1;
          i += 2;
        } else if (source.startsWith("*/", i)) {
          depth -= 1;
          i += 2;
        } else {
          i += 1;
        }
      }
      continue;
    }
    out += source[i];
    i += 1;
  }
  return out;
}

/**
 * Comment-free Rust source with every `#[cfg(test)] mod … { … }` excised.
 *
 * The MLX adapter's tests quote the refusal phrase verbatim (they pin the refusal-by-name behaviour
 * sc-18104 introduced), so leaving them in would hand the dispatch scan below a phantom dispatch
 * site whose "arms" are test scaffolding.
 */
function stripTestModules(source) {
  let out = source;
  for (;;) {
    const module = /#\[cfg\(test\)\]\s*mod\s+[A-Za-z0-9_]+\s*\{/.exec(out);
    if (!module) return out;
    let depth = 0;
    let i = module.index + module[0].length - 1;
    while (i < out.length) {
      const skipped = skipStringLike(out, i);
      if (skipped !== i) {
        i = skipped;
        continue;
      }
      if (out[i] === "{") depth += 1;
      else if (out[i] === "}") {
        depth -= 1;
        if (depth === 0) {
          i += 1;
          break;
        }
      }
      i += 1;
    }
    if (depth !== 0) {
      throw new Error("unbalanced braces while excising a #[cfg(test)] module from an adapter source");
    }
    out = out.slice(0, module.index) + out.slice(i);
  }
}

/** The `{ … }` body of the `match` starting at `matchStart`: `{ inner, end }` (end past the `}`). */
function matchBlockBody(text, matchStart) {
  let i = matchStart;
  while (i < text.length && text[i] !== "{") i = Math.max(skipStringLike(text, i), i + 1);
  let depth = 0;
  const open = i;
  while (i < text.length) {
    const skipped = skipStringLike(text, i);
    if (skipped !== i) {
      i = skipped;
      continue;
    }
    if (text[i] === "{") depth += 1;
    else if (text[i] === "}") {
      depth -= 1;
      if (depth === 0) return { inner: text.slice(open + 1, i), end: i + 1 };
    }
    i += 1;
  }
  throw new Error("unbalanced braces while extracting an adapter dispatch match block");
}

/**
 * Top-level arms of a match body: `{ pattern, hasArrow }`. An arm ends at a depth-0 comma, or at
 * the depth-0 `}` that closes a block body — rustfmt drops the trailing comma after a braced arm,
 * so splitting on commas alone made a block-bodied arm swallow the NEXT arm's pattern (sc-22726).
 */
function matchArms(inner) {
  const arms = [];
  let depth = 0;
  let start = 0;
  let arrow = -1;
  let i = 0;
  while (i < inner.length) {
    const skipped = skipStringLike(inner, i);
    if (skipped !== i) {
      i = skipped;
      continue;
    }
    const char = inner[i];
    if ("([{".includes(char)) depth += 1;
    else if (")]}".includes(char)) {
      depth -= 1;
      // A depth-0 `}` after the arrow closes a block body: the arm ends here whether or not a
      // comma follows. Before the arrow it is a struct/brace PATTERN and the arm continues.
      if (depth === 0 && char === "}" && arrow !== -1) {
        arms.push({ pattern: inner.slice(start, arrow).trim(), hasArrow: true });
        start = i + 1;
        arrow = -1;
      }
    } else if (depth === 0 && char === "=" && inner[i + 1] === ">") {
      if (arrow === -1) arrow = i;
      i += 2;
      continue;
    } else if (depth === 0 && char === ",") {
      arms.push({ pattern: inner.slice(start, arrow === -1 ? i : arrow).trim(), hasArrow: arrow !== -1 });
      start = i + 1;
      arrow = -1;
    }
    i += 1;
  }
  const tail = inner.slice(start, arrow === -1 ? inner.length : arrow).trim();
  if (tail) arms.push({ pattern: tail, hasArrow: arrow !== -1 });
  return arms.filter((arm) => arm.hasArrow);
}

/**
 * The provider ids one adapter binary can dispatch, parsed from its source (see the block comment
 * above). `label` names the source in every error so a red says which adapter moved.
 */
export function adapterCapturableProviders(source, label) {
  const cleaned = stripTestModules(stripRustComments(source));
  const consts = new Map(
    [...cleaned.matchAll(/\bconst\s+([A-Z][A-Z0-9_]*)\s*:\s*&str\s*=\s*"((?:\\.|[^"\\])*)"\s*;/g)].map(
      (item) => [item[1], item[2]],
    ),
  );
  const gates = [];
  // Where each shared gate begins, and which function it lives in: `preGateRegion` needs both to
  // decide how much of `fn run` executes BEFORE any gate.
  const gateStarts = [];
  const gateFns = [];
  let from = 0;
  for (;;) {
    const refusal = cleaned.indexOf(DISPATCH_REFUSAL, from);
    if (refusal === -1) break;
    from = refusal + DISPATCH_REFUSAL.length;
    const matchStart = cleaned.lastIndexOf("match ", refusal);
    if (matchStart === -1) {
      throw new Error(`${label}: the dispatch refusal phrase appears outside any match block`);
    }
    gateStarts.push(matchStart);
    const owner = [...cleaned.slice(0, matchStart).matchAll(/^fn\s+([a-z_][A-Za-z0-9_]*)\s*\(/gm)].at(-1);
    if (!owner) throw new Error(`${label}: a dispatch gate lives outside any top-level fn`);
    gateFns.push(owner[1]);
    const block = matchBlockBody(cleaned, matchStart);
    if (block.end <= refusal) {
      throw new Error(`${label}: the dispatch refusal phrase escaped its own match block`);
    }
    const providers = new Set();
    for (const arm of matchArms(block.inner)) {
      // An OR-pattern (`A | B => …`) is one arm serving several providers — the Candle FLUX.2 arm
      // dispatches `FLUX2_DEV_ID | FLUX2_KLEIN_ID` (sc-22727), and the SenseNova arms dispatch
      // `SENSENOVA_ID | SENSENOVA_FAST_ID` (sc-22734). Each alternative is then the same shape the
      // single-pattern arms are, so split first and judge each half by the same rules; a `|` inside
      // a string literal cannot reach here because a literal arm has no `|` outside its quotes, and
      // a leading `|` (rustfmt's multi-line spelling) yields an empty alternative, which is dropped.
      const alternatives = /^"((?:\\.|[^"\\])*)"$/.test(arm.pattern)
        ? [arm.pattern]
        : arm.pattern.split("|").map((part) => part.trim()).filter(Boolean);
      for (const pattern of alternatives) {
        const literal = /^"((?:\\.|[^"\\])*)"$/.exec(pattern);
        if (literal) {
          providers.add(literal[1]);
        } else if (/^[A-Z][A-Z0-9_]*$/.test(pattern)) {
          if (!consts.has(pattern)) {
            throw new Error(`${label}: dispatch arm ${pattern} does not resolve to a &str const`);
          }
          providers.add(consts.get(pattern));
        } else if (!/^[a-z_][A-Za-z0-9_]*$/.test(pattern)) {
          // Lower-case identifiers are the fallback binding of the refusal arm itself; anything
          // else is a dispatch shape this parser has never seen and must not guess about.
          throw new Error(`${label}: unrecognized dispatch arm pattern ${JSON.stringify(pattern)}`);
        }
      }
    }
    if (providers.size === 0) {
      throw new Error(`${label}: a dispatch match block admits no provider at all`);
    }
    gates.push(providers);
  }
  if (gates.length === 0) {
    throw new Error(
      `${label}: no provider dispatch found — the anchor phrase ${JSON.stringify(DISPATCH_REFUSAL)} ` +
        "has moved, so capturability can no longer be derived from this adapter",
    );
  }
  const gated = gates.reduce((acc, gate) => new Set([...acc].filter((id) => gate.has(id))));
  return [...new Set([...gated, ...bespokePreGateProviders(cleaned, consts, label, gateStarts, gateFns)])].sort();
}

/**
 * The function `main` routes the `"run"` action to, or `null` when the source declares no `main` at
 * all (a fragment, which has no binary entry and therefore no pre-gates).
 *
 * Derived rather than named, because WHICH function is the entry is the whole basis for saying a
 * dispatch runs before the gates: `run_qwen_provider` opens with a `return run_qwen_vae_probe(request)`
 * guard that looks exactly like a pre-gate and is not one — it is reached only THROUGH the gate.
 */
export function runEntryName(cleaned, label) {
  const main = /\bfn\s+main\s*\(\s*\)\s*\{/.exec(cleaned);
  if (!main) return null;
  const body = matchBlockBody(cleaned, main.index + main[0].length - 1).inner;
  const arm = /"run"\s*=>\s*([a-z_][A-Za-z0-9_]*)\s*\(\s*&\s*request\s*\)/.exec(body);
  if (!arm) {
    throw new Error(
      `${label}: fn main declares no "run" => <entry>(&request) arm, so the bespoke pre-gate scan has ` +
        "no entry point to be exhaustive over",
    );
  }
  return arm[1];
}

/**
 * The slice of the adapter's run entry that executes BEFORE any shared dispatch gate — the only
 * place a bespoke pre-gate can live, and therefore the only place this parser has to be exhaustive.
 *
 * The boundary is derived, not named: a gate is a `match` block carrying the refusal phrase, so the
 * region ends at whichever comes first inside `run` — such a block written into `run` itself (the
 * MLX adapter's one big `match provider`), or the first call to a function that contains one (the
 * Candle adapter's `plain_execution_path`). Everything after that point has already been refused or
 * admitted by a gate, so a dispatch there adds no capturability and is not scanned.
 */
function preGateRegion(cleaned, label, gateStarts, gateFns) {
  const entryName = runEntryName(cleaned, label);
  if (entryName === null) return "";
  const entry = new RegExp(String.raw`\bfn\s+${entryName}\s*\(`).exec(cleaned);
  if (!entry) throw new Error(`${label}: main dispatches "run" to ${entryName}, which is not defined here`);
  const { inner, end } = matchBlockBody(cleaned, entry.index + entry[0].length);
  const bodyStart = end - 1 - inner.length;
  let cut = inner.length;
  for (const start of gateStarts) {
    if (start >= bodyStart && start < end) cut = Math.min(cut, start - bodyStart);
  }
  for (const name of gateFns) {
    if (name === "run") continue;
    const call = new RegExp(`\\b${name}\\s*\\(`).exec(inner);
    if (call) cut = Math.min(cut, call.index);
  }
  return inner.slice(0, cut);
}

/**
 * The providers a bespoke pre-gate routes before the shared dispatch gates (sc-22736, sc-22737) —
 * see the block comment above.
 *
 * Four guard shapes ship in the two adapter bins, and all four are read here:
 *
 *   - `if provider == <id> { return <call>(request); }` — LTX-2.5, PuLID, Qwen-Edit, InstantID;
 *   - `if provider == <id> || provider == <id> … { return <call>(request); }` — the SenseNova pair;
 *   - `if matches!(provider, <id> | <id> …) { return <module>::run(request); }` — Wan 2.2 + SCAIL-2;
 *   - `if let Some(<b>) = <helper>(request)? { return <call>(request, <b>); }` — sc-22737's video
 *     block, whose ids live in the `[Arm; N]` table the helper iterates. The table is FOLLOWED:
 *     each element const's `engine_id` field is resolved through the same `&str` const map the
 *     match arms use, so the ids stay derived from the adapter rather than transcribed here.
 *
 * FAIL CLOSED. Every top-level `if` in the pre-gate region whose block returns a dispatch call —
 * `return <path>(request …)`, as opposed to a `return Err(…)` refusal — must be one of those four.
 * A fifth shape THROWS: silently ignoring it is exactly the bug this function is fixing, and it
 * reads as "no adapter arm can serve this lane", which sends an operator to book a capture host for
 * a lane that is already implemented (or, worse, retires an arm as unreachable).
 */
export function bespokePreGateProviders(cleaned, consts, label, gateStarts = [], gateFns = []) {
  const providers = new Set();
  const resolve = (pattern) => {
    const literal = /^"((?:\\.|[^"\\])*)"$/.exec(pattern);
    if (literal) return literal[1];
    if (/^[A-Z][A-Z0-9_]*$/.test(pattern)) {
      if (!consts.has(pattern)) {
        throw new Error(`${label}: bespoke dispatch on ${pattern} does not resolve to a &str const`);
      }
      return consts.get(pattern);
    }
    throw new Error(`${label}: unrecognized bespoke pre-gate pattern ${JSON.stringify(pattern)}`);
  };
  const region = preGateRegion(cleaned, label, gateStarts, gateFns);
  const ID = String.raw`"(?:\\.|[^"\\])*"|[A-Z][A-Z0-9_]*`;
  const CALL = String.raw`(?!Err\b|Ok\b)[A-Za-z_][A-Za-z0-9_:]*`;
  // `return <path>(request);` — the whole guarded block, nothing else in it. `return Err(…)` is a
  // refusal, and a call carrying MORE than the request is a validator, not a dispatch.
  const DISPATCH = String.raw`return\s+${CALL}\s*\(\s*request\s*\)\s*;?`;
  const guards = [
    // `provider == A`, optionally `|| provider == B || …`.
    {
      re: new RegExp(String.raw`^if\s+provider\s*==\s*(${ID})((?:\s*\|\|\s*provider\s*==\s*(?:${ID}))*)\s*\{\s*${DISPATCH}\s*\}$`),
      ids: (m) => [m[1], ...[...m[2].matchAll(new RegExp(String.raw`provider\s*==\s*(${ID})`, "g"))].map((x) => x[1])],
    },
    // `matches!(provider, A | B | …)`.
    {
      re: new RegExp(String.raw`^if\s+matches!\(\s*provider\s*,\s*([\s\S]+?)\s*,?\s*\)\s*\{\s*${DISPATCH}\s*\}$`),
      ids: (m) => m[1].split("|").map((part) => part.trim()).filter(Boolean),
    },
    // `if let Some(<binding>) = <helper>(request)? { return <call>(request, <binding>); }` — the one
    // shape whose dispatch carries a second argument, and only ever the guard's own binding.
    {
      re: new RegExp(String.raw`^if\s+let\s+Some\(\s*([a-z_][A-Za-z0-9_]*)\s*\)\s*=\s*([a-z_][A-Za-z0-9_]*)\s*\(\s*request\s*\)\s*\?\s*\{\s*return\s+${CALL}\s*\(\s*request\s*,\s*\1\s*\)\s*;?\s*\}$`),
      ids: (m) => armTableEngineIds(cleaned, m[2], label),
    },
  ];
  // A statement is a candidate pre-gate when it dispatches: either the plain `return <call>(request);`
  // shape, or the `if let Some(…)` shape above. Everything else in the region is ordinary code.
  const candidate = new RegExp(String.raw`\{\s*${DISPATCH}\s*\}$|^if\s+let\s+Some\(`);
  for (const statement of topLevelIfStatements(region)) {
    if (!candidate.test(statement)) continue;
    const guard = guards.map((shape) => ({ shape, match: shape.re.exec(statement) })).find((hit) => hit.match);
    if (!guard) {
      throw new Error(
        `${label}: a bespoke pre-gate in fn run returns a dispatch through a guard shape this parser ` +
          `does not read — ${JSON.stringify(statement.replace(/\s+/g, " ").slice(0, 120))}. Teach ` +
          "bespokePreGateProviders the shape; an unread pre-gate reports its lanes as uncapturable.",
      );
    }
    for (const pattern of guard.shape.ids(guard.match)) providers.add(resolve(pattern));
  }
  return providers;
}

/** Top-level `if …` statements of a function body, brace-balanced, in source order. */
function topLevelIfStatements(body) {
  const statements = [];
  let i = 0;
  while (i < body.length) {
    const skipped = skipStringLike(body, i);
    if (skipped !== i) { i = skipped; continue; }
    if (/\bif$/.test(body.slice(0, i + 2).slice(-2)) && /^if\b/.test(body.slice(i))) {
      // Only an `if` that STARTS a statement: preceded by `{`, `}` or `;` (whitespace aside).
      const before = body.slice(0, i).replace(/\s+$/, "");
      if (before === "" || "{};".includes(before.at(-1))) {
        const open = body.indexOf("{", i);
        if (open !== -1) {
          const block = matchBlockBody(body, i);
          statements.push(body.slice(i, block.end).trim());
          i = block.end;
          continue;
        }
      }
    }
    i += 1;
  }
  return statements;
}

/**
 * The provider ids the `[Arm; N]` table a pre-gate helper iterates names in its `engine_id` fields.
 *
 * `sc22737_video_arm` answers `Some` exactly for the engine ids in `SC22737_VIDEO_ARMS`, so those
 * ids ARE the pre-gate's provider set. Every step is derived off the adapter source and throws when
 * it cannot be: the helper, the table it iterates, the table's element consts, and each element's
 * `engine_id`.
 */
function armTableEngineIds(cleaned, helper, label) {
  const fn = new RegExp(String.raw`\bfn\s+${helper}\s*\(`).exec(cleaned);
  if (!fn) throw new Error(`${label}: bespoke pre-gate calls ${helper}(request), which this file does not define`);
  const body = matchBlockBody(cleaned, fn.index + fn[0].length).inner;
  const tables = [...new Set([...body.matchAll(/\b([A-Z][A-Z0-9_]{2,})\s*\.\s*iter\s*\(/g)].map((m) => m[1]))];
  if (tables.length !== 1) {
    throw new Error(
      `${label}: ${helper} iterates ${tables.length} const arm tables (${tables.join(", ") || "none"}); ` +
        "this parser reads a pre-gate helper only when exactly one table decides which providers it serves",
    );
  }
  const table = new RegExp(String.raw`\bconst\s+${tables[0]}\s*:\s*\[[^\]]*\]\s*=\s*\[([\s\S]*?)\]\s*;`).exec(cleaned);
  if (!table) throw new Error(`${label}: ${helper} iterates ${tables[0]}, which is not a const array literal here`);
  const elements = table[1].split(",").map((part) => part.trim()).filter(Boolean);
  const ids = new Set();
  for (const element of elements) {
    if (!/^[A-Z][A-Z0-9_]*$/.test(element)) {
      throw new Error(`${label}: ${tables[0]} names an element ${JSON.stringify(element)} this parser cannot resolve`);
    }
    const declaration = new RegExp(String.raw`\bconst\s+${element}\s*:[^=]*=\s*[A-Za-z_][A-Za-z0-9_]*\s*\{([\s\S]*?)\n\}\s*;`).exec(cleaned);
    if (!declaration) throw new Error(`${label}: ${tables[0]} element ${element} is not a struct-literal const here`);
    const engine = /\bengine_id\s*:\s*("(?:\\.|[^"\\])*"|[A-Z][A-Z0-9_]*)\s*,/.exec(declaration[1]);
    if (!engine) throw new Error(`${label}: ${tables[0]} element ${element} declares no engine_id`);
    ids.add(engine[1]);
  }
  if (ids.size === 0) throw new Error(`${label}: ${tables[0]} names no engine id at all`);
  return [...ids];
}

/**
 * Per-lane anchor-plan coverage: `Map<lane, { entries, authoritative }>`.
 *
 * sc-22514: the plan is now one anchor per `<modelId>:<tier>:<backend>` key, so an entry count IS
 * a cell count rather than a grid-row count.
 */
export function planLaneCoverage(plan) {
  const byLane = new Map();
  // An old-shape (`providers` array) plan has no `anchors` object, and `?? {}` would report every
  // lane as having zero planned entries instead of failing.
  if (!plan.anchors) throw new Error("calibration plan is not an anchor plan (no `anchors` object)");
  for (const [key, entry] of Object.entries(plan.anchors)) {
    const backend = key.split(":")[2];
    const provider = entry.provider;
    if (typeof backend !== "string" || !backend || typeof provider !== "string") {
      throw new Error(`anchor-plan entry ${JSON.stringify(key)} names no backend/provider lane`);
    }
    const lane = laneOf(backend, provider);
    if (!byLane.has(lane)) byLane.set(lane, { entries: 0, authoritative: 0 });
    byLane.get(lane).entries += 1;
    if (entry.evidenceScope === "authoritative") byLane.get(lane).authoritative += 1;
  }
  return byLane;
}

/**
 * Which catalog models declare a binding on each lane, for the MODELS column only.
 *
 * A binding is ANY object under a model's `mlx`/`candle` block carrying both `provider` and
 * `inferenceRevision` — the same shape `stampManifest`'s locator pairs on, at any depth, so
 * `turboFit` and `candle.control` are included by construction rather than by enumeration. This is
 * attribution, not population: `manifestBindings` cross-checks the per-lane counts against the
 * locator and throws if the two walks disagree.
 */
export function laneModelAttribution(manifest) {
  const byLane = new Map();
  const visit = (value, lane) => {
    if (Array.isArray(value)) {
      for (const item of value) visit(item, lane);
      return;
    }
    if (value === null || typeof value !== "object") return;
    if (typeof value.provider === "string" && typeof value.inferenceRevision === "string") {
      lane(value.provider);
    }
    for (const child of Object.values(value)) visit(child, lane);
  };
  for (const model of manifest.models ?? []) {
    for (const backend of ["mlx", "candle"]) {
      if (!model[backend]) continue;
      visit(model[backend], (provider) => {
        const key = laneOf(backend, provider);
        if (!byLane.has(key)) byLane.set(key, { models: new Set(), count: 0 });
        byLane.get(key).models.add(model.id);
        byLane.get(key).count += 1;
      });
    }
  }
  return byLane;
}

/**
 * Every manifest closure binding, flattened to `{ lane, modelId, digest }`.
 *
 * The population is `stampManifest`'s — the CI gate's own locator (see the module header for why
 * this file must not re-derive it). Its coverage checks are load-bearing here too: a `skipped`
 * revision or an `orphan` digest means part of the admission surface is invisible, and a report that
 * silently omitted it would understate exactly what it exists to surface.
 */
export function manifestBindings({ manifestBody, manifest }) {
  // The locator is a stamper; it is used here purely as a scanner, so the rewritten body is
  // discarded and the replacement digest is an inert constant that never reaches disk.
  const { located, skipped, orphans } = stampManifest(manifestBody, () => "0".repeat(64));
  if (skipped.length || orphans.length) {
    throw new Error(
      "the manifest closure locator cannot see the whole admission surface, so the stale-lane " +
        `report would understate it:\n  ${[...skipped, ...orphans].join("\n  ")}`,
    );
  }
  const attribution = laneModelAttribution(manifest);
  const counted = new Map();
  for (const binding of located) counted.set(binding.key, (counted.get(binding.key) ?? 0) + 1);
  for (const lane of new Set([...counted.keys(), ...attribution.keys()])) {
    const authoritative = counted.get(lane) ?? 0;
    const attributed = attribution.get(lane)?.count ?? 0;
    if (authoritative !== attributed) {
      throw new Error(
        `manifest binding walks disagree on ${lane}: the locator found ${authoritative}, the parsed ` +
          `attribution found ${attributed}. One of the two is missing a binding shape.`,
      );
    }
  }
  return located.map((binding) => ({
    lane: binding.key,
    modelId: [...(attribution.get(binding.key)?.models ?? [])].sort().join("+") || null,
    digest: binding.digest,
  }));
}

/**
 * Every evidence record, flattened to `{ lane, modelId, digest, eligible }`.
 *
 * `eligible` is `recordsNeedingDigest`'s verdict — the digest backfiller's OWN population predicate,
 * not a re-spelled filter (sc-18252). Only complete/runtime_complete authoritative records ever
 * reach the currency comparison; a fixture, candidate or gated record legitimately carries no
 * closure digest, and counting it as "stale" would both inflate the widened evidence surface and
 * flip a never-captured lane out of the pending-capture list.
 */
export function evidenceBindings(records) {
  const eligible = new Set(recordsNeedingDigest({ records }));
  return records.map((record) => ({
    lane: laneOf(record.backend, record.target?.provider),
    modelId: record.target?.modelId ?? null,
    digest: record.repositories?.inference?.closureDigest ?? null,
    eligible: eligible.has(record),
  }));
}

function tally(items, liveDigest) {
  const stale = items.filter((item) => item.digest !== liveDigest);
  return {
    total: items.length,
    stale: stale.length,
    current: items.length - stale.length,
    staleItems: stale,
  };
}

function shortDigests(items) {
  return [...new Set(items.map((item) => item.digest ?? "(absent)"))]
    .sort()
    .map((digest) => (digest === "(absent)" ? digest : digest.slice(0, 12)));
}

/**
 * Rank stale lanes by impact.
 *
 * The ordering is lexicographic over quantities that are all reported, so a consumer that disagrees
 * with the weighting can re-rank from `--json` rather than being stuck with it:
 *
 *   1. `widenedAdmissionSurface` = stale BINDINGS x the margin the runtime widens them by. Bindings
 *      are the shipped admission surface, so this is production impact and it leads.
 *   2. `widenedEvidenceSurface`  = stale RECORDS x the same margin. Corpus debt: what a re-capture
 *      of this lane would actually have to cover.
 *   3. lane name, so the ordering is total and the output is diffable.
 *
 * Multiplying a count by the margin rather than ranking on the count alone is the "margin-widening
 * impact" the epic asks for: two lanes with equal stale counts are not equally costly when one runs
 * on the MLX margin and the other on candle's, which is 2.5x narrower.
 */
export function rankLanes(lanes) {
  return [...lanes].sort(
    (left, right) =>
      right.impact.widenedAdmissionSurface - left.impact.widenedAdmissionSurface ||
      right.impact.widenedEvidenceSurface - left.impact.widenedEvidenceSurface ||
      left.lane.localeCompare(right.lane),
  );
}

/**
 * Build the report.
 *
 * @param liveDigests    `Map<lane, digest>` from `validatedInferenceClosures`.
 * @param declarations   the `providers` block of the closure config (for the crate pointer).
 * @param records        the evidence bundle's records.
 * @param manifest       the parsed builtin model manifest (model attribution only).
 * @param manifestBody   the raw JSONC body — the authoritative binding population is located in it.
 * @param plan           the parsed calibration plan — planned lanes join the universe (sc-18212).
 * @param adapterSources raw Rust source of the two adapter binaries; capturability is parsed from
 *                       them, so a report built without them refuses rather than guesses.
 */
export function buildStaleLaneReport({
  liveDigests,
  declarations,
  records,
  manifest,
  manifestBody,
  plan,
  adapterSources,
  meta = {},
}) {
  if (typeof adapterSources?.mlx !== "string" || typeof adapterSources?.candle !== "string") {
    throw new Error(
      "buildStaleLaneReport needs both adapter sources — capturability cannot be reported without them",
    );
  }
  const arms = {
    mlx: adapterCapturableProviders(adapterSources.mlx, SOURCE_PATHS.mlxAdapter),
    candle: adapterCapturableProviders(adapterSources.candle, SOURCE_PATHS.candleAdapter),
  };
  const planCoverage = planLaneCoverage(plan ?? { providers: [] });
  const flagshipCoverage = recommendedMlxT2iLanes(manifest).map((entry) => {
    const declared = liveDigests.has(entry.lane);
    const planned = planCoverage.has(entry.lane);
    const capturable = arms.mlx.includes(entry.provider);
    return {
      ...entry,
      declared,
      planned,
      capturable,
      covered: entry.contractDeclared && declared && planned && capturable,
    };
  });
  const margins = deriveMargins(records);
  const bindings = manifestBindings({ manifestBody, manifest });
  const evidence = evidenceBindings(records);

  const laneFacts = (lane) => {
    const backend = lane.split(":")[0];
    const provider = lane.split(":").slice(1).join(":");
    return {
      backend,
      provider,
      capturable: (arms[backend] ?? []).includes(provider),
      plan: planCoverage.get(lane) ?? { entries: 0, authoritative: 0 },
      laneBindings: bindings.filter((item) => item.lane === lane),
      laneRecords: evidence.filter((item) => item.lane === lane && item.eligible),
      ineligibleRecords: evidence.filter((item) => item.lane === lane && !item.eligible).length,
    };
  };

  const lanes = [];
  for (const [lane, liveDigest] of [...liveDigests].sort(([left], [right]) => left.localeCompare(right))) {
    const { backend, provider, capturable, plan: laneWork, laneBindings, laneRecords, ineligibleRecords } =
      laneFacts(lane);
    const bindingTally = tally(laneBindings, liveDigest);
    const recordTally = tally(laneRecords, liveDigest);
    const backendMargins = margins[backend]?.margins ?? null;
    // A lane the derivation does not model gets no invented margin: the impact terms fall back to
    // the raw counts so the lane still ranks, and the null is visible in the output.
    const recaptureSpread = backendMargins?.recaptureSpread ?? null;
    const weight = recaptureSpread ?? 1;
    const measured = bindingTally.total + recordTally.total > 0;
    const staleCount = bindingTally.stale + recordTally.stale;
    lanes.push({
      lane,
      backend,
      provider,
      declared: true,
      capturable,
      plan: laneWork,
      crate: declarations?.[lane]?.crate ?? null,
      liveDigest,
      liveDigestShort: liveDigest.slice(0, 12),
      capturedDigests: shortDigests([...laneBindings, ...laneRecords]),
      // Measurement status and capturability are orthogonal facts, EXCEPT for the unmeasured case:
      // "unmeasured" (pending capture) is a promise that a capture is possible, so an armless lane
      // gets "uncapturable" instead — the recategorization sc-18212 exists for. A measured armless
      // lane keeps its staleness status (its evidence and margins are real) and carries
      // `capturable: false` alongside.
      status: !measured
        ? capturable
          ? "unmeasured"
          : "uncapturable"
        : staleCount === 0
          ? "current"
          : bindingTally.current + recordTally.current > 0
            ? "partially-stale"
            : "stale",
      models: [
        ...new Set([...laneBindings, ...laneRecords].map((item) => item.modelId).filter(Boolean)),
      ].sort(),
      bindings: { total: bindingTally.total, stale: bindingTally.stale, current: bindingTally.current },
      records: {
        total: recordTally.total,
        stale: recordTally.stale,
        current: recordTally.current,
        ineligible: ineligibleRecords,
      },
      margin: backendMargins
        ? {
            recaptureSpread: backendMargins.recaptureSpread,
            hardFloor: backendMargins.hardFloor,
            source: MARGIN_SOURCE,
          }
        : null,
      impact: {
        widenedAdmissionSurface: bindingTally.stale * weight,
        widenedEvidenceSurface: recordTally.stale * weight,
      },
    });
  }

  // Lanes the plan targets but the closure table never declared (sc-18104 §2d rows one and four).
  // They have no live digest, so nothing about them can be graded for currency — but hiding them
  // reproduced the exact blindness the runbook's §1 warning documents.
  const undeclaredLanes = [...planCoverage.keys()]
    .filter((lane) => !liveDigests.has(lane))
    .sort()
    .map((lane) => {
      const { backend, provider, capturable, plan: laneWork, laneBindings, laneRecords, ineligibleRecords } =
        laneFacts(lane);
      return {
        lane,
        backend,
        provider,
        declared: false,
        capturable,
        plan: laneWork,
        status: "undeclared",
        bindings: { total: laneBindings.length },
        records: { total: laneRecords.length, ineligible: ineligibleRecords },
      };
    });

  const stale = rankLanes(lanes.filter((lane) => lane.status === "stale" || lane.status === "partially-stale"));
  const uncapturable = [...lanes, ...undeclaredLanes].filter((lane) => !lane.capturable);
  return {
    generatedAgainst: {
      inferenceRevision: meta.inferenceRevision ?? null,
      digestVersion: meta.digestVersion ?? null,
      evidenceRecords: records.length,
    },
    marginSource: MARGIN_SOURCE,
    capturability: {
      source: CAPTURABILITY_SOURCE,
      arms,
      uncapturableLanes: uncapturable.map((lane) => lane.lane),
    },
    flagshipApparatusCoverage: {
      source:
        "recommended:true MLX image manifest entries with text_to_image; declared provider or model.id fallback when the whole contract is absent",
      lanes: flagshipCoverage,
      missingLanes: flagshipCoverage.filter((entry) => !entry.covered).map((entry) => entry.lane),
    },
    totals: {
      declaredLanes: lanes.length,
      staleLanes: stale.length,
      currentLanes: lanes.filter((lane) => lane.status === "current").length,
      unmeasuredLanes: lanes.filter((lane) => lane.status === "unmeasured").length,
      uncapturableLanes: uncapturable.length,
      undeclaredLanes: undeclaredLanes.length,
      staleBindings: lanes.reduce((sum, lane) => sum + lane.bindings.stale, 0),
      staleRecords: lanes.reduce((sum, lane) => sum + lane.records.stale, 0),
    },
    staleLanes: stale.map((lane, index) => ({ rank: index + 1, ...lane })),
    currentLanes: lanes.filter((lane) => lane.status === "current"),
    unmeasuredLanes: lanes.filter((lane) => lane.status === "unmeasured"),
    uncapturableLanes: uncapturable,
    undeclaredLanes,
  };
}

export async function loadSources(root = ROOT) {
  const [closuresBody, evidenceBody, manifestBody, cargoBody, planBody, mlxAdapter, candleAdapter] =
    await Promise.all(
      Object.values(SOURCE_PATHS).map((relative) => readFile(path.join(root, relative), "utf8")),
    );
  const closures = JSON.parse(closuresBody);
  return {
    // The SAME predicate the matrix generator applies, so the report and the matrix cannot disagree
    // about which lanes are current. It no longer requires the table to be keyed to the live pin:
    // `inferenceRevision` is the revision the digests were derived at, and demanding it equal the
    // Cargo pin made every pin bump a forced re-derivation, which demoted every lane at once.
    liveDigests: validatedInferenceClosures(closuresBody),
    declarations: closures.providers,
    records: JSON.parse(evidenceBody).records,
    manifest: JSON.parse(stripJsoncComments(manifestBody)),
    manifestBody,
    plan: JSON.parse(planBody),
    adapterSources: { mlx: mlxAdapter, candle: candleAdapter },
    meta: { inferenceRevision: closures.inferenceRevision, digestVersion: closures.digestVersion },
  };
}

function percent(fraction) {
  return fraction === null || fraction === undefined ? "n/a" : `${(fraction * 100).toFixed(2)}%`;
}

export function formatReport(report) {
  const out = [];
  const meta = report.generatedAgainst;
  out.push(
    `sc-18098 stale-lane report — ${SOURCE_PATHS.closures} @ ` +
      `${meta.inferenceRevision?.slice(0, 8) ?? "(unset)"} (${meta.digestVersion ?? "unversioned"}), ` +
      `${meta.evidenceRecords} evidence records`,
  );
  out.push("");
  out.push(
    "Staleness is a SIGNAL, not a gate (epic 18093 R2/R3). A stale lane keeps serving its measured " +
      "numbers\nbehind the widened margin below; nothing in `npm run check`, `rust:check`, the " +
      "pre-push hook or CI\ndemands a re-capture. This report exists so the debt is visible on " +
      "demand instead of enforced\nper-PR.",
  );
  out.push("");
  const totals = report.totals;
  out.push(
    `${totals.declaredLanes} declared lanes: ${totals.staleLanes} stale, ${totals.currentLanes} current, ` +
      `${totals.unmeasuredLanes} pending capture; ${totals.uncapturableLanes} lanes (declared or ` +
      `planned) have no adapter arm, and ${totals.undeclaredLanes} planned lanes were never declared.`,
  );
  out.push(
    `${totals.staleBindings} shipped calibration bindings are serving under a widened margin; ` +
      `${totals.staleRecords} eligible evidence records are stale corpus debt, not runtime inputs.`,
  );
  out.push("");

  if (report.staleLanes.length === 0) {
    out.push("No stale lanes. Every captured lane's closure digest matches the live derivation.");
  } else {
    out.push("STALE LANES, ranked by widened admission surface (stale bindings x recapture spread), then evidence surface:");
    out.push("");
    const header = ["#", "LANE", "BINDINGS", "RECORDS", "RECAPTURE", "IMPACT", "CAPTURE", "MODELS"];
    const rows = report.staleLanes.map((lane) => [
      String(lane.rank),
      lane.lane,
      `${lane.bindings.stale}/${lane.bindings.total}`,
      `${lane.records.stale}/${lane.records.total}`,
      percent(lane.margin?.recaptureSpread ?? null),
      lane.impact.widenedAdmissionSurface.toFixed(3),
      lane.capturable ? "yes" : "NO ARM",
      lane.models.join(", ") || "(none)",
    ]);
    const widths = header.map((_, column) =>
      Math.max(header[column].length, ...rows.map((row) => row[column].length)),
    );
    const line = (row) => row.map((cell, column) => cell.padEnd(widths[column])).join("  ").trimEnd();
    out.push(line(header));
    out.push(widths.map((width) => "-".repeat(width)).join("  "));
    for (const row of rows) out.push(line(row));
    out.push("");
    for (const lane of report.staleLanes) {
      out.push(
        `  ${lane.lane}  crate=${lane.crate ?? "(undeclared)"}  live=${lane.liveDigestShort}  ` +
          `captured=${lane.capturedDigests.join(",")}  status=${lane.status}`,
      );
    }
  }

  if (report.currentLanes.length) {
    out.push("");
    out.push(
      `CURRENT (no widening applied): ${report.currentLanes.map((lane) => lane.lane).join(", ")}`,
    );
  }
  if (report.unmeasuredLanes.length) {
    out.push("");
    out.push(
      "PENDING CAPTURE (declared, adapter arm exists, no measurement yet): " +
        report.unmeasuredLanes
          .map((lane) => `${lane.lane} (${lane.plan.entries} plan entries, ${lane.plan.authoritative} authoritative)`)
          .join(", "),
    );
  }
  if (report.uncapturableLanes.length) {
    out.push("");
    out.push(
      "DECLARED/PLANNED BUT UNCAPTURABLE — no adapter arm can serve these; a capture host booked for",
    );
    out.push(
      "one is wasted (docs/calibration-runbook.md §2c). A declared lane needs adapter work before",
    );
    out.push(
      "measurement; a planned-but-undeclared lane needs both an adapter arm and a closure declaration:",
    );
    for (const lane of report.uncapturableLanes) {
      out.push(
        `  ${lane.lane}  declared=${lane.declared ? "yes" : "NO"}  plan=${lane.plan.entries} entries ` +
          `(${lane.plan.authoritative} authoritative)  bindings=${lane.bindings.total ?? 0} shipped  ` +
          `records=${lane.records.total ?? 0} eligible  status=${lane.status}`,
      );
    }
  }
  if (report.undeclaredLanes.some((lane) => lane.capturable)) {
    out.push("");
    out.push(
      "PLANNED BUT NEVER DECLARED (adapter arm exists, but no closure entry — a capture cannot derive",
    );
    out.push("a digest, so its evidence could never become current; runbook §2d row four):");
    for (const lane of report.undeclaredLanes.filter((item) => item.capturable)) {
      out.push(
        `  ${lane.lane}  plan=${lane.plan.entries} entries (${lane.plan.authoritative} authoritative)`,
      );
    }
  }
  out.push("");
  out.push("RECOMMENDED MLX T2I APPARATUS COVERAGE (manifest-derived):");
  for (const entry of report.flagshipApparatusCoverage.lanes) {
    out.push(
      `  ${entry.modelId} -> ${entry.lane}  contract=${entry.contractDeclared ? "yes" : "NO"}  ` +
        `closure=${entry.declared ? "yes" : "NO"}  ` +
        `plan=${entry.planned ? "yes" : "NO"}  adapter=${entry.capturable ? "yes" : "NO"}`,
    );
  }
  out.push("");
  out.push(`Margin source: ${report.marginSource}`);
  out.push(`Capturability source: ${report.capturability.source}`);
  out.push(
    `Adapter arms — mlx: ${report.capturability.arms.mlx.join(", ")}; candle: ` +
      `${report.capturability.arms.candle.join(", ")}`,
  );
  return `${out.join("\n")}\n`;
}

async function main(argv = process.argv.slice(2)) {
  const sources = await loadSources();
  const report = buildStaleLaneReport(sources);
  process.stdout.write(argv.includes("--json") ? `${JSON.stringify(report, null, 2)}\n` : formatReport(report));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main().catch((error) => {
    process.stderr.write(`${error?.stack ?? error}\n`);
    process.exitCode = 1;
  });
}
