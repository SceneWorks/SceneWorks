#!/usr/bin/env node
// Measure every memory anchor the plan declares for ONE backend on THIS host, committing each
// measurement as it lands so a crash or a cancel loses at most the anchor in flight.
//
// The harness (memory-calibration-harness.mjs, sc-22514) captures exactly one
// `<model>:<tier>:<backend>` anchor per invocation and refuses to run from a dirty checkout. That is
// the invariant this orchestrator keeps: each anchor is captured on a clean HEAD, then the evidence
// is ingested, the anchor store re-derived, its currency stamped, the matrix regenerated, and ALL of
// that committed before the next capture starts. Nothing here can express a second measurement of a
// cell — it only walks the keys the plan already declares.
//
// Per anchor:  capture → check → ingest → PACKAGED_MEMORY_ANCHOR_SOURCES → extract → stamp →
//              matrix → commit.   A failed capture is logged and the loop moves on; a failed
//              post-step is rolled back so the tree is clean again for the next anchor.
//
//   node scripts/measure-memory-catalog.mjs --backend mlx \
//     --adapter target/release/memory-mlx-adapter --inference-repo ../inference \
//     --work-dir /abs/OUTSIDE/the/repo/calib --campaign sc-NNNN [--model sdxl ...] [--anchors a,b]
//     [--skip-current]
//     [--dry-run] [--no-commit] [--hf-cache DIR ...]   (--hf-cache is repeatable)
//
// A cell the store already carries a CURRENT measured lower bound for classifies `exceeded_current`
// and is never scheduled, with or without `--skip-current` (sc-22738): the host has proved it cannot
// finish there under this loader closure, and a re-run could only re-establish the same inequality.
// A bound that has STALED classifies nothing, so the cell is capturable again the moment its
// evidence stops speaking for production.
//
// `--no-commit` captures and checks each anchor and stops there (status `captured`, raw bundle in
// <work-dir>/captures): the harness refuses complete evidence from a dirty checkout, so the first
// anchor's ingest would leave every later anchor in the same run uncapturable (sc-22724). Ingest a
// retained bundle by hand with the harness, or run without the flag to land it.
import process from "node:process";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { readFile, writeFile, mkdir, cp, rm, realpath, stat, readdir } from "node:fs/promises";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import { hashArtifactInventory } from "./hash-artifact-inventory.mjs";
// ONE spelling of the refusal signature, shared with the arm that records it (sc-22738). The runner
// classifies a failure by it and the harness re-checks the same stderr before writing a bound, so a
// second transcription here would be a way for the two to disagree about what a refusal even is.
import { metalSubmissionsIgnored } from "./memory-calibration-harness.mjs";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const PLAN_PATH = "config/memory-calibration-plan.json";
export const MANIFEST_PATH = "config/manifests/builtin.models.jsonc";
export const MATRIX_PATH = "docs/generated/memory-matrix.json";
export const ADAPTER_LIB_PATH = "crates/sceneworks-memory-adapter/src/lib.rs";
export const PACKAGED_SOURCES_PATH = "crates/sceneworks-core/src/memory_anchor.rs";
export const ANCHOR_STORE_PATH = "config/memory-anchors.json";
export const ANCHOR_LOADER_CONFIG_PATH = "config/anchor-loader-closures.json";
export const PROVIDER_CLOSURE_CONFIG_PATH = "config/inference-provider-closures.json";
export const MATRIX_MD_PATH = "docs/generated/memory-matrix.md";
/** A well-formed but meaningless key the extractor accepts for a NEW anchor; `--stamp-anchors`
 *  re-derives every key at its record's own revision right after, before anything is committed. */
export const SEED_DIGEST = "0".repeat(64);
export const HARNESS = "scripts/memory-calibration-harness.mjs";

// LTX-2.5 is bound by the harness itself (`--ltx25-snapshot-root`), at the revision it hard-codes.
export const LTX25_REPOSITORY = "SceneWorks/ltx-2.5-mlx";
/** The ONE LTX-2.3 rehost, which the manifest ships to all three platforms (sc-22737). */
export const LTX_2_3_REPOSITORY = "SceneWorks/ltx-2.3-mlx";

/**
 * The three caller-staged SDXL components, as `{ env, repo }` pairs. Declared once: every SDXL-family
 * model — and InstantID, which composes the same SDXL base — stages exactly these, and the Rust side
 * declares the same three ids in `candle.rs` `SDXL_COMPONENTS`. `sdxl_component_env_matches_the_catalog`
 * proves the two lists agree.
 */
export const SDXL_COMPONENTS = Object.freeze([
  { env: "SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_L", repo: "openai/clip-vit-large-patch14" },
  { env: "SCENEWORKS_SDXL_COMPONENT_TOKENIZER_CLIP_BIGG", repo: "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" },
  { env: "SCENEWORKS_SDXL_COMPONENT_VAE_FP16_FIX", repo: "madebyollin/sdxl-vae-fp16-fix" },
]);

/**
 * sc-22729. `candle-gen-sdxl`'s `SDXL_ROUTES` pins each route's repository AND revision, and its
 * `path_has_snapshot` matches a staged root against that literal before `SdxlArtifactSeal::capture`
 * will seal a contract. When the pinned revision is not the one this repository ships, no root the
 * manifest can resolve can ever seal — `candle_gen_sdxl::load` errors before reading a weight. That
 * is an INFERENCE-side divergence, not adapter work.
 *
 * NOTHING here is a literal: both halves of the comparison are READ (the engine revision out of the
 * pinned inference checkout, the shipped revision out of the manifest), so the refusal disappears by
 * itself the moment the engine agrees — no edit to this file, and no stale exclusion outliving the
 * fix. The inference-side repair lives on `story/sc-22729-sdxl-route-revisions`.
 */
export const SDXL_ROUTES_PATH = "crates/media/candle-gen/candle-gen-sdxl/src/memory_strategy.rs";

/** `SDXL_ROUTES` as `id → { repository, revision }`. Throws if the table can no longer be read. */
export function parseSdxlRoutes(source) {
  const table = /pub const SDXL_ROUTES: &\[SdxlRoute\] = &\[([\s\S]*?)\n\];/.exec(source);
  if (!table) fail(`${SDXL_ROUTES_PATH} no longer declares a parsable SDXL_ROUTES table`);
  const routes = new Map();
  for (const entry of table[1].matchAll(/SdxlRoute\s*\{([\s\S]*?)\}/g)) {
    const field = (name) => new RegExp(`\\b${name}:\\s*"([^"]+)"`).exec(entry[1])?.[1];
    const [id, repository, revision] = [field("id"), field("repository"), field("revision")];
    if (!id || !repository || !revision) fail(`an SDXL_ROUTES entry declares no id/repository/revision: ${entry[0]}`);
    routes.set(id, { repository, revision });
  }
  if (routes.size === 0) fail(`${SDXL_ROUTES_PATH} declares an empty SDXL_ROUTES table`);
  return routes;
}

/**
 * The engine's route table at `inferenceRepo`, or `null` when no inference checkout is reachable.
 * `null` is NOT a refusal: with nothing to compare against, a cell classifies as it would if the
 * engine agreed, and says so.
 */
export async function readSdxlCandleRoutes(inferenceRepo = process.env.INFERENCE_REPO) {
  if (!inferenceRepo) return null;
  try {
    return parseSdxlRoutes(await readFile(path.join(inferenceRepo, SDXL_ROUTES_PATH), "utf8"));
  } catch (error) {
    if (error.code === "ENOENT" || error.code === "ENOTDIR") return null;
    throw error;
  }
}

export const SDXL_ROUTES_UNCHECKED =
  `no inference checkout (--inference-repo / $INFERENCE_REPO) supplies ${SDXL_ROUTES_PATH}, so the `
  + "engine's route revision was not compared with the shipped one; classified as if the engine agrees";

export const WAN_MLX_LOADER_PATH = "crates/media/mlx-gen/mlx-gen-wan/src/model.rs";

export const WAN_MLX_SEAL_UNCHECKED =
  `no inference checkout (--inference-repo / $INFERENCE_REPO) supplies ${WAN_MLX_LOADER_PATH}, so `
  + "the pinned Wan MLX loaders were not asked which routes seal a memory receipt; classified as if "
  + "every declared route seals one";

/**
 * The provider ids whose `mlx-gen-wan` LOADER seals a memory receipt at this pin (sc-22738).
 *
 * A REGISTERED memory-strategy contract is not a published one. `mlx-gen-wan/src/lib.rs` registers
 * `i2v_memory_strategy::{t2v_14b,i2v_14b}::MEMORY_REGISTRATION` alike, so the source text the
 * anchor loader closure names as the memory-strategy entry point — the file
 * `readDeclaredStrategySupport` reads — is IDENTICAL for the two A14B routes and can never tell
 * them apart. The divergence lives one file over, in the loader: a route whose `load_*` does not
 * call `i2v_memory_strategy::prepare` hands back a generator whose
 * `Generator::memory_strategy_contract()` is `None`, refuses every optimized rung with "has no
 * prepared I2V memory receipt", and opens no request scope. The capture arm then refuses the cell
 * AFTER a multi-minute load ("loaded … exposed no memory-strategy contract"), which is exactly the
 * booked-capture-that-cannot-finish this classifier exists to prevent.
 *
 * Nothing here is a curated list of routes: the sealing call sites name a provider-id const, the
 * const's value is read out of the same file, and a route that starts (or stops) sealing moves the
 * classification on its own at the next pin. It is fail-closed — a `prepare` call whose provider
 * const this file does not define, or a source that seals nothing at all, THROWS rather than
 * reporting "no evidence".
 */
export function parseWanMlxSealedProviders(source, sourcePath = WAN_MLX_LOADER_PATH) {
  const text = stripRustComments(source);
  const sealed = new Set();
  for (const [, constant] of text.matchAll(/\bi2v_memory_strategy::prepare\(\s*spec\s*,\s*(\w+)\s*\)/g)) {
    const declared = new RegExp(`\\bconst\\s+${constant}\\s*:\\s*&str\\s*=\\s*"([^"]+)"`).exec(text);
    if (!declared) {
      return fail(
        `${sourcePath} seals a Wan memory receipt for ${constant}, which this file declares no `
          + "`const … : &str` value for, so the provider id it seals cannot be read. Teach "
          + "parseWanMlxSealedProviders the new shape — an unreadable seal must never be treated as "
          + "'no evidence'.",
      );
    }
    sealed.add(declared[1]);
  }
  if (sealed.size === 0) {
    return fail(
      `${sourcePath} calls i2v_memory_strategy::prepare for no route at all, which no shipped `
        + "revision of mlx-gen-wan does. Either the loaders moved or this parser no longer reads "
        + "them; refusing to report every Wan MLX route as unsealed on a parser failure.",
    );
  }
  return sealed;
}

/**
 * The sealing routes at `inferenceRepo`, or `null` when no inference checkout is reachable. `null`
 * is NOT a refusal — see `WAN_MLX_SEAL_UNCHECKED`.
 */
export async function readWanMlxSealedProviders(inferenceRepo = process.env.INFERENCE_REPO) {
  if (!inferenceRepo) return null;
  try {
    return parseWanMlxSealedProviders(await readFile(path.join(inferenceRepo, WAN_MLX_LOADER_PATH), "utf8"));
  } catch (error) {
    if (error.code === "ENOENT" || error.code === "ENOTDIR") return null;
    throw error;
  }
}

/** Why the MLX lane cannot capture `provider` at this pin, or `null` when it can. */
export function wanMlxSealGap(provider, sealed) {
  if (sealed.has(provider)) return null;
  return `the pinned mlx-gen-wan loader (${WAN_MLX_LOADER_PATH}) seals no memory receipt for `
    + `${provider}, so the LOADED provider publishes no memory-strategy contract however the `
    + "registry declares one: the capture arm refuses the cell after the load ("
    + `"loaded ${provider} exposed no memory-strategy contract"). The MLX lane cannot measure this `
    + "route at this pin.";
}

/**
 * Why the candle lane cannot route `modelId` today, or `null` when it can.
 *
 * Both revisions are derived: the engine's from `SDXL_ROUTES` at the pinned inference checkout, the
 * shipped one from the manifest download the route's own repository names.
 */
export function sdxlCandleRouteDrift(modelId, tier, routes, models) {
  const route = routes.get(modelId);
  if (!route) {
    return `candle-gen-sdxl's SDXL_ROUTES (${SDXL_ROUTES_PATH}) declares no route ${modelId}, so the `
      + "candle lane has no artifact identity to seal for it";
  }
  const shipped = tierDownload(models, modelId, route.repository, tier).revision;
  if (shipped === route.revision) return null;
  return `candle-gen-sdxl pins route ${modelId} at ${route.revision.slice(0, 8)} `
    + `(${SDXL_ROUTES_PATH} SDXL_ROUTES), but this repository ships ${shipped.slice(0, 8)}; `
    + "path_has_snapshot matches that literal, so no staged root can seal the contract and the load "
    + "fails before any weight is read. The candle lane does not route this model at this pin.";
}

/**
 * sc-22736 (pin 8a65db2a). A plan row may override the lane's default anchor rung only where the
 * provider CONTRACT refuses that rung, and the evidence for the refusal must be DERIVED. Two derived
 * sources already exist — the manifest's `memoryStrategyStructuralExemptions` and the checked-in
 * capability dump's `implementedRungs` — and neither can speak for candle SCAIL-2:
 *
 *   - the manifest key means `StructurallyNotApplicable`, and `candle-gen-scail2` classifies every
 *     non-resident rung as `Missing` (not implemented yet), so an exemption there would be a lie;
 *   - `config/engine-capabilities/capabilities.candle.json` carries no `scail2_14b` contract at all,
 *     because the dump enumerates `ProviderRegistry::memory_contract_surfaces()` and the crate
 *     registers a strategy without a weights-free surface resolver. That is an inference-side gap;
 *     regenerating the dump at this pin (or any pin) would not conjure the surface.
 *
 * So the THIRD derived source is the engine's own declaration, read as source text out of the
 * pinned inference checkout at the path the anchor loader closure already names as that
 * (model, lane)'s memory-strategy entry point — a path that is itself derived at the pin and
 * `--check`ed by `scripts/anchor-loader-closure.mjs`. Nothing here is a curated list of model ids:
 * a provider that grows a dump surface stops consulting this, and a provider that stops declaring
 * the rung as unimplemented moves the requirement on its own.
 *
 * It is fail-closed in both directions. A `strategies:` expression this parser does not recognize
 * THROWS rather than returning "no evidence", so a refactor upstream reds the rule instead of
 * quietly re-admitting every override; and a provider whose declaration is absent yields `null`,
 * which the caller refuses.
 */
const STRATEGY_ENTRY_POINT_RE = /memory_strategy[^/]*\.rs$/;

/** snake_case rung (`staged_residency`) → the Rust `MemoryStrategy` variant (`StagedResidency`). */
function strategyVariant(rung) {
  return rung.split("_").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join("");
}

function unreadable(sourcePath, detail) {
  return fail(
    `${sourcePath} declares a strategies: field in a shape scripts/measure-memory-catalog.mjs cannot `
      + "read, so it cannot say which rungs the contract implements. Teach parseDeclaredStrategySupport "
      + `the new shape — an unreadable declaration must never be treated as 'no evidence'. (${detail})`,
  );
}

/**
 * Rust source with line and block comments blanked out (newlines kept so offsets and line breaks
 * survive). Every scan below runs over this, because the declarations carry paragraphs of prose that
 * contain `=>`, `MemoryStrategy::…`, braces and quotes, and a scanner that reads them as code is a
 * scanner that reads the wrong arm.
 */
function stripRustComments(source) {
  let out = "";
  let i = 0;
  while (i < source.length) {
    const two = source.slice(i, i + 2);
    if (two === "//") {
      while (i < source.length && source[i] !== "\n") { out += " "; i += 1; }
      continue;
    }
    if (two === "/*") {
      let depth = 0;
      while (i < source.length) {
        if (source.slice(i, i + 2) === "/*") { depth += 1; out += "  "; i += 2; continue; }
        if (source.slice(i, i + 2) === "*/") {
          depth -= 1; out += "  "; i += 2;
          if (depth === 0) break;
          continue;
        }
        out += source[i] === "\n" ? "\n" : " ";
        i += 1;
      }
      continue;
    }
    if (source[i] === '"') {
      out += source[i]; i += 1;
      while (i < source.length && source[i] !== '"') {
        if (source[i] === "\\") { out += source.slice(i, i + 2); i += 2; continue; }
        out += source[i]; i += 1;
      }
      out += source[i] ?? ""; i += 1;
      continue;
    }
    out += source[i]; i += 1;
  }
  return out;
}

/** The text from `start` up to the first depth-0 occurrence of any `stops` character, or `null`. */
function scanTo(source, start, stops) {
  let depth = 0;
  for (let i = start; i < source.length; i += 1) {
    const ch = source[i];
    if (ch === "(" || ch === "[" || ch === "{") depth += 1;
    else if (ch === ")" || ch === "]" || ch === "}") {
      if (depth === 0) return stops.includes(ch) ? source.slice(start, i) : null;
      depth -= 1;
    } else if (depth === 0 && stops.includes(ch)) return source.slice(start, i);
  }
  return null;
}

/** The `{ … }` block that opens at or after `from`, contents only, or `null`. */
function braceBody(source, from) {
  const open = source.indexOf("{", from);
  if (open === -1) return null;
  let depth = 0;
  for (let i = open; i < source.length; i += 1) {
    if (source[i] === "{") depth += 1;
    else if (source[i] === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(open + 1, i);
    }
  }
  return null;
}

/** `MemoryStrategySupport::Variant` (struct-variant payload allowed) as a bare variant name. */
function supportVariant(body, sourcePath) {
  let text = body.trim();
  if (text.startsWith("{")) text = (braceBody(text, 0) ?? "").trim();
  const named = /^MemoryStrategySupport::(\w+)\s*/.exec(text);
  if (!named) return unreadable(sourcePath, `support arm "${body.trim().slice(0, 60)}" names no MemoryStrategySupport variant`);
  const rest = text.slice(named[0].length).trim();
  // `StructurallyNotApplicable { reason: … }` carries a payload; anything else trailing is a shape
  // this parser has not been taught, and guessing at it would invent evidence.
  if (rest !== "" && !(rest.startsWith("{") && braceBody(rest, 0) !== null && rest.slice(rest.lastIndexOf("}") + 1).trim() === "")) {
    return unreadable(sourcePath, `support arm "${body.trim().slice(0, 60)}" carries an expression this parser cannot read`);
  }
  return named[1];
}

/**
 * Whether an `if` condition holds for `variant`: `true`, `false`, or `null` when the source text
 * does not decide it.
 *
 * The conditions that ship are disjunctions of conjunctions over `strategy == MemoryStrategy::X`
 * and non-strategy terms (`candle-gen-sdxl` gates `StagedResidency` on `surface ==
 * SdxlSurface::Bespoke`). A conjunct naming a DIFFERENT variant makes its term statically false, so
 * most rungs still resolve; only the rung the runtime term actually gates comes back `null`.
 */
function evaluateStrategyCondition(condition, variant) {
  let unknown = false;
  for (const term of condition.split("||")) {
    let text = term.trim();
    if (text.startsWith("(") && text.endsWith(")")) text = text.slice(1, -1).trim();
    let termFalse = false;
    let termUnknown = false;
    for (const conjunct of text.split("&&")) {
      const named = /^\s*\(?\s*strategy\s*==\s*MemoryStrategy::(\w+)\s*\)?\s*$/.exec(conjunct);
      if (named) {
        if (named[1] !== variant) termFalse = true;
      } else {
        termUnknown = true;
      }
    }
    if (termFalse) continue;
    if (termUnknown) { unknown = true; continue; }
    return true;
  }
  return unknown ? null : false;
}

/**
 * `match strategy { … }` arms as `{ variants, guarded, support }`, in source order. `variants` is
 * `null` for the `_` catch-all.
 */
function parseMatchArms(armsText, sourcePath) {
  const arms = [];
  let i = 0;
  while (i < armsText.length) {
    if (/\s/.test(armsText[i])) { i += 1; continue; }
    const pattern = scanTo(armsText, i, "=");
    if (pattern === null || armsText[i + pattern.length + 1] !== ">") {
      return unreadable(sourcePath, `match arm near "${armsText.slice(i, i + 60).trim()}" has no => `);
    }
    i += pattern.length + 2;
    while (i < armsText.length && /\s/.test(armsText[i])) i += 1;
    let body;
    if (armsText[i] === "{") {
      const inner = braceBody(armsText, i);
      if (inner === null) return unreadable(sourcePath, "unbalanced match-arm block");
      body = inner;
      i += 1 + inner.length + 1;
      while (i < armsText.length && /[\s,]/.test(armsText[i])) i += 1;
    } else {
      const value = scanTo(armsText, i, ",");
      body = value ?? armsText.slice(i);
      i += body.length + 1;
    }
    const [patterns, ...guard] = pattern.split(/\bif\b/);
    const variants = [];
    let catchAll = false;
    for (const alternative of patterns.split("|")) {
      const text = alternative.trim();
      if (text === "_") { catchAll = true; continue; }
      const variant = /^MemoryStrategy::(\w+)$/.exec(text);
      if (!variant) return unreadable(sourcePath, `match pattern "${text}" is not a MemoryStrategy variant`);
      variants.push(variant[1]);
    }
    arms.push({
      variants: catchAll && variants.length === 0 ? null : variants,
      guarded: guard.length > 0,
      support: supportVariant(body, sourcePath),
    });
  }
  return arms;
}

/**
 * `rung → MemoryStrategySupport` for a provider whose contract declares support over
 * `MemoryStrategy::ALL`, or `null` when the file declares no `strategies:` field at all.
 *
 * Throws when a `strategies:` field IS declared in a shape this parser cannot read: an unrecognized
 * declaration is NOT evidence that a rung is unimplemented.
 *
 * Three shapes are read, because all three ship at the pin (sc-22737):
 *
 *   - `strategies: MemoryStrategy::ALL … .map(…)` — the declaration inline in `build_contract`
 *     (`candle-gen-scail2`, `candle-gen-sdxl`, `mlx-gen-krea`, `mlx-gen-z-image`, …);
 *   - `strategies: strategies()` / `strategies(spec)` / `strategies(streamable)` — the SAME
 *     expression hoisted into a file-local helper (`candle-gen-bernini`, both LTX modules, both
 *     MiniMax-H3 modules, both Wan modules). The call is followed to that `fn`'s body; a call whose
 *     helper is not in the file is unreadable, not "no evidence";
 *   - inside the closure, `support:` as either the `if strategy == … { … } else { … }` form or a
 *     `match strategy { … }` with `|`-alternatives, `_`, and guarded arms.
 *
 * A rung whose FIRST matching arm carries an `if` guard is condition-dependent — `mlx-gen-minimax-h3`
 * declares `BoundedTransformerResidency` `Implemented` only when `streamable` — so asking for it
 * throws rather than picking one side. Guessing either way would state a rung's support as fact when
 * the source text does not decide it.
 */
export function parseDeclaredStrategySupport(rawSource, sourcePath) {
  const source = stripRustComments(rawSource);
  const field = /\bstrategies:/.exec(source);
  if (!field) return null;
  const valueStart = field.index + field[0].length;
  const value = scanTo(source, valueStart, ",") ?? source.slice(valueStart);
  // A helper call (`strategies(spec)`) is followed to the file-local `fn` it names; the inline form
  // is its own body. Anything else — a const, a method call, a builder — is refused.
  let body;
  const call = /^\s*(\w+)\s*\(/.exec(value);
  if (/^\s*MemoryStrategy::ALL\b/.test(value)) {
    body = value;
  } else if (call) {
    const helper = new RegExp(`\\bfn\\s+${call[1]}\\s*\\(`).exec(source);
    if (!helper) {
      return unreadable(sourcePath, `strategies: calls ${call[1]}(…), which this file does not define`);
    }
    body = braceBody(source, helper.index + helper[0].length);
    if (body === null) return unreadable(sourcePath, `fn ${call[1]} has no readable body`);
  } else {
    return unreadable(sourcePath, `strategies: value "${value.trim().slice(0, 60)}" is neither MemoryStrategy::ALL nor a local helper call`);
  }

  const closure = /\.map\(\s*\|\s*strategy\s*\|\s*MemoryStrategyCapability\s*\{/.exec(body);
  if (!closure) return unreadable(sourcePath, "no .map(|strategy| MemoryStrategyCapability { … }) over MemoryStrategy::ALL");
  const fields = body.slice(closure.index + closure[0].length);
  const supportAt = /\bsupport:\s*/.exec(fields);
  if (!supportAt) return unreadable(sourcePath, "the capability closure declares no support: field");
  const supportStart = supportAt.index + supportAt[0].length;
  const supportExpr = (scanTo(fields, supportStart, ",") ?? fields.slice(supportStart)).trim();

  // The `if <condition over strategy> { A } else { B }` form.
  if (/^if\b/.test(supportExpr)) {
    let depth = 0;
    let condEnd = -1;
    for (let i = 2; i < supportExpr.length; i += 1) {
      const ch = supportExpr[i];
      if (ch === "(" || ch === "[") depth += 1;
      else if (ch === ")" || ch === "]") depth -= 1;
      else if (ch === "{" && depth === 0) { condEnd = i; break; }
    }
    if (condEnd === -1) return unreadable(sourcePath, "support: if with no block");
    const condition = supportExpr.slice(2, condEnd);
    const thenBody = braceBody(supportExpr, condEnd);
    if (thenBody === null) return unreadable(sourcePath, "support: if has an unbalanced then-block");
    const afterThen = supportExpr.slice(condEnd + 1 + thenBody.length + 1);
    if (!/^\s*else\s*\{/.test(afterThen)) {
      return unreadable(sourcePath, "support: if has no plain else block (an else-if chain is not read)");
    }
    const elseBody = braceBody(afterThen, 0);
    if (elseBody === null) return unreadable(sourcePath, "support: else has an unbalanced block");
    if (afterThen.slice(afterThen.indexOf("{") + elseBody.length + 2).trim() !== "") {
      return unreadable(sourcePath, "support: if/else is followed by an expression this parser cannot read");
    }
    const thenSupport = supportVariant(thenBody, sourcePath);
    const elseSupport = supportVariant(elseBody, sourcePath);
    return (rung) => {
      const variant = strategyVariant(rung);
      const holds = evaluateStrategyCondition(condition, variant);
      if (holds === null) {
        return fail(
          `${sourcePath} gates MemoryStrategy::${variant} on a condition that is not about the strategy `
            + "itself, so its support is a runtime value rather than a fact this source text states. Read "
            + "the rung from the engine capability dump instead of guessing which side of the branch holds.",
        );
      }
      return holds ? thenSupport : elseSupport;
    };
  }

  if (/^match\s+strategy\s*\{/.test(supportExpr)) {
    const armsText = braceBody(supportExpr, 0);
    if (armsText === null) return unreadable(sourcePath, "unbalanced match strategy { … }");
    const arms = parseMatchArms(armsText, sourcePath);
    return (rung) => {
      const variant = strategyVariant(rung);
      const arm = arms.find((candidate) => candidate.variants === null || candidate.variants.includes(variant));
      if (!arm) {
        return unreadable(sourcePath, `match strategy { … } has no arm for MemoryStrategy::${variant}`);
      }
      if (arm.guarded) {
        return fail(
          `${sourcePath} declares MemoryStrategy::${variant} behind an \`if\` guard, so its support is a `
            + "runtime condition rather than a fact this source text states. Read the rung from the engine "
            + "capability dump instead of guessing which side of the guard holds.",
        );
      }
      return arm.support;
    };
  }

  return unreadable(sourcePath, `support: expression "${supportExpr.slice(0, 60)}" is neither the if/else nor the match form`);
}

/**
 * The support declaration for one `(modelId, backend)` from the pinned inference checkout, or `null`
 * when no checkout is reachable, the closure names no memory-strategy entry point, or the file
 * declares no `strategies:` field. `null` is a REFUSAL at every call site, never an assumption.
 */
export async function readDeclaredStrategySupport(
  modelId,
  backend,
  closures,
  inferenceRepo = process.env.INFERENCE_REPO,
) {
  if (!inferenceRepo) return null;
  const entryPoints = closures.models?.[`${modelId}:${backend}`]?.entryPoints ?? [];
  const entry = entryPoints.find((candidate) => STRATEGY_ENTRY_POINT_RE.test(candidate));
  if (!entry) return null;
  let source;
  try {
    source = await readFile(path.join(inferenceRepo, entry), "utf8");
  } catch (error) {
    if (error.code === "ENOENT" || error.code === "ENOTDIR") return null;
    throw error;
  }
  return parseDeclaredStrategySupport(source, entry);
}

/**
 * The built-in Qwen-Image-Edit-2511 Lightning distill LoRA (sc-22728). It is NOT a manifest
 * download — the worker fetches it lazily into the HF cache on first use — so its repository,
 * revision and file are pinned in the worker's own source on both lanes
 * (`crates/sceneworks-worker/src/image_jobs/qwen.rs` `QWEN_LIGHTNING_LORA_{REPO,REVISION}` and
 * `image_jobs/qwen_edit_candle.rs` `QWEN_EDIT_CANDLE_LIGHTNING_LORA_*`), and the candle engine
 * refuses any other path by exact suffix (`edit.rs` `is_exact_lightning_path`). The values below are
 * bound to those constants by a test rather than trusted, because a drift here would send the
 * capture at a LoRA the engine will reject after the load.
 */
export const QWEN_EDIT_LIGHTNING_LORA = Object.freeze({
  env: "QWEN_EDIT_LIGHTNING_LORA",
  repo: "lightx2v/Qwen-Image-Edit-2511-Lightning",
  revision: "d74eba145674fd7e31b949324e148e21e7118abd",
  file: "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
});

/**
 * The converter-written marker every SenseNova `_fast` rehost tier subdir carries (sc-22734). Both
 * engines read it by this exact name (`DISTILL_MERGED_MARKER` in `mlx-gen-sensenova` and
 * `candle-gen-sensenova`), and both WITHHOLD the production calibration identity when it is absent —
 * so a `_fast` tier root without it is not a capturable cell, it is a cell whose capture would fail
 * on an identity mismatch after the load. Declared as `requiredTierFiles` on the three `_fast`
 * family rows so `classifyAnchor` refuses it by name.
 */
export const SENSENOVA_DISTILL_MERGED_MARKER = "distill_merged.json";

/**
 * The files the InstantID identity stack must carry, per staged directory (sc-22738).
 *
 * `SCENEWORKS_INSTANTID_WEIGHTS` is a bundle DIRECTORY and `SCENEWORKS_INSTANTID_CONTROLNET` an
 * IdentityNet directory, and the adapter refuses each by name when a file is absent
 * (`lib.rs` `instantid_identity_bundle_at` over `INSTANTID_IDENTITY_BUNDLE_FILES`, and
 * `instantid_controlnet_dir` over `INSTANTID_CONTROLNET_WEIGHT_FILE`). `--list` asked only whether
 * the two directories existed, so a half-staged identity stack classified `runnable` and the
 * capture died on the missing file — the same shape of miss the Qwen Lightning LoRA cost. Bound to
 * those Rust constants by a test rather than restated, so neither list can drift alone.
 */
export const INSTANTID_IDENTITY_BUNDLE_FILES = Object.freeze([
  "ip-adapter.safetensors",
  "scrfd_10g.safetensors",
  "arcface_iresnet100.safetensors",
]);
export const INSTANTID_CONTROLNET_WEIGHT_FILE = "diffusion_pytorch_model.safetensors";

/**
 * The stage-two official refinement LoRA both LTX-2.5 arms attach, relative to the SNAPSHOT root
 * (sc-22738) — `mlx_ltx25.rs` `DEV_ADAPTER` and `candle.rs` `LTX25_DISTILL_LORA_RELATIVE_PATH`,
 * which are the same string and are bound to this one by a test. It sits BESIDE the
 * `<variant>/<tier>` load root rather than inside it, so nothing the tier probe reads can see it.
 */
export const LTX25_DEV_REFINEMENT_LORA = "distilled_lora/ltx-2.5-22b-distilled-lora-450-bf16.safetensors";

/** The stock enhancer co-requisite the MLX LTX-2.5 arm requires beside the load root
 *  (`mlx_ltx25.rs` `load_artifact`); the Candle arm never opens it. */
export const LTX25_ENHANCER_DIR = "enhancer";

/**
 * What `mlx_gen_minimax_h3::model::load` opens in the UPSTREAM snapshot root, at the pinned
 * revision (`crates/media/mlx-gen/mlx-gen-minimax-h3/src/model.rs`), read off the loader rather
 * than guessed from the manifest's download globs.
 *
 * `spec.weights` for BOTH MiniMax entries is the dense `MiniMaxAI/MiniMax-H3` snapshot — only the
 * DiT and (at q4/q8) the text encoder are redirected onto the rehost — and the loader probes these
 * six documents under it before it will build the generator. `FL2VA/audio_vae` carries the audio
 * VAE's constructor arguments, which the repackaged root config does not.
 *
 * sc-22738 declares them because the campaign found the hard way what an undeclared read costs:
 * this host holds a `MiniMaxAI/MiniMax-H3` snapshot with `FL2VA/`, `audio_vae/`, `tokenizer/` and
 * `vae/` and NO `text_encoder/`, `--list` called `minimax_h3:bf16:mlx` runnable, and the booked
 * capture died on `read …/text_encoder: No such file or directory` before it rendered anything.
 */
export const MINIMAX_UPSTREAM_ROOT_FILES = Object.freeze([
  "vae/config.json",
  "audio_vae/config.json",
  "tokenizer/tokenizer.json",
  "FL2VA/audio_vae/config.json",
  "FL2VA/audio_vae/config.yaml",
  "FL2VA/audio_vae/metadata.json",
]);

/**
 * The text encoder's config, wherever the tier puts it.
 *
 * The TE's tier is DERIVED from the DiT's rather than being a free axis (`mlx.rs`
 * `minimax_text_encoder_source`, and the manifest's three `componentId: "text_encoder"`
 * co-requisite rows): `q4`/`q8` stage `<tier>/text_encoder` from the rehost, `bf16` takes the dense
 * `text_encoder/` from the upstream root. So the SAME probe is declared against two different roots
 * depending on the tier — which is exactly what the per-tier form of a required-file declaration is
 * for, and why one flat list could not express it.
 */
export const MINIMAX_TEXT_ENCODER_CONFIG = "text_encoder/config.json";

/**
 * The two DiT partitions the loader probes inside the resolved tier root: the base one, and
 * `transformer_ref/` — which is probed as the base partition's SIBLING at load time for EVERY
 * entry, not only for `minimax_h3_ref`, because `ref2va` is a first-class task of the engine.
 */
export const MINIMAX_TIER_DIT_FILES = Object.freeze([
  "transformer/config.json",
  "transformer_ref/config.json",
]);

/**
 * The files a required-file declaration demands for `tier`.
 *
 * A declaration is either a flat list (every tier needs the same files — the SenseNova `_fast`
 * marker) or a per-tier object `{ all, q4, q8, bf16 }` whose `all` entry applies to every tier and
 * whose tier entry is added on top. Absent is the empty list: a family that declares nothing keeps
 * classifying exactly as it did.
 */
export function requiredFilesFor(declaration, tier) {
  if (!declaration) return [];
  if (Array.isArray(declaration)) return declaration;
  return [...(declaration.all ?? []), ...(declaration[tier] ?? [])];
}

/**
 * The shared Mage-Flow text-encoder + VAE rehost (sc-22733). Declared once and referenced by all six
 * Mage family rows: it is the SAME repository and the SAME revision for every variant and both
 * lanes, and both adapters read it through one `SCENEWORKS_MAGE_FLOW_COMPONENTS_*` family. The
 * revision is NOT pinned here — it is read from the manifest's own `coRequisite` download rows, so
 * a components re-host lands by editing the manifest alone.
 */
export const MAGE_COMPONENTS = Object.freeze({
  env: "MAGE_FLOW_COMPONENTS",
  repo: "SceneWorks/Mage-Flow-Components-mlx",
});

/**
 * The two component ids both Mage engines advertise, in descriptor order. Bound to the adapter's own
 * `MAGE_COMPONENT_TEXT_ENCODER` / `MAGE_COMPONENT_VAE` constants by a test, so the directories this
 * script probes are the ones the adapters actually stage.
 */
export const MAGE_COMPONENT_IDS = Object.freeze(["text_encoder", "vae"]);

/**
 * One row per provider arm an adapter implements, mirroring `match provider` in
 * crates/sceneworks-memory-adapter/src/bin/{mlx,candle}.rs and the env families the runbook lists
 * under "Adapter environment".
 *
 * `sourceCapture` marks an MLX arm that emits a provider `sourceCapture` fragment. The coupling the
 * harness enforces is BIDIRECTIONAL (`capturePlannedCase`): `--raw-log-dir` without a sourceCapture
 * fails, and a sourceCapture without the raw-log pair fails too. So exactly these arms get
 * `SCENEWORKS_MEMORY_CAPTURE_DIR` + `SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX`, the harness's
 * `--raw-log-dir`/`--source-path-prefix`, `ingest --source-root`, and the receipt copy into the
 * campaign directory — and no other arm gets any of them.
 *
 * sc-22738: this used to be spelled `physical`, which conflated the emission with the CURRENCY rule
 * below and so bound the pair to `qwen_image` alone. `crates/.../bin/mlx_ltx25.rs` has emitted a
 * `physical_mlx` sourceCapture since SC-18783 and `required_env`s the capture dir unconditionally
 * (`prepare_source_capture`), so all three `ltx_2_5:*:mlx` captures died on
 * `required environment variable SCENEWORKS_MEMORY_CAPTURE_DIR is not set` — for `bf16` after 883s,
 * because the harness re-hashes the ~90GB LTX-2.5 snapshot before the adapter is ever spawned. The
 * two facts are now two flags, and `every MLX arm that emits a sourceCapture declares it`
 * (measure-memory-catalog.test.mjs) derives the expected set from the adapter sources.
 *
 * `physical` is the narrower CURRENCY rule: the harness demands a validated physical source session
 * before it will call an anchor current, and it scopes that demand to `modelId === "qwen_image"`
 * alone (`requiresPhysicalMlxProvenanceForCurrency`, memory-calibration-harness.mjs), which a test
 * below binds this table to. `physical` therefore implies `sourceCapture`; the reverse does not
 * hold, and LTX-2.5 is the arm that proves it. `physical` is NOT inherited by a sibling family.
 *
 * Rows are keyed by PROVIDER by default; sc-22729 adds MODEL-keyed rows (which must declare their
 * `provider`) for the case where several catalog models ride one engine id. See `familyFor`.
 *
 * `sideArtifact` is a second root a family member needs that the MANIFEST does not ship — today only
 * the Qwen edit Lightning distill LoRA, which the worker fetches lazily at a pinned revision. It is
 * keyed by model id because it belongs to one member of a shared-provider family, not to the family.
 *
 * Rows are keyed by PROVIDER by default; sc-22734 adds MODEL-keyed rows (which must declare their
 * `provider`) for the case where several catalog models ride one engine id but ship their own
 * independently pinned artifacts. See `familyFor`.
 *
 * ## What is NOT in this table, and why that is not a gap (sc-22738)
 *
 * STRICT-CONTROL OVERLAY PROVIDERS — `z_image_control`, `z_image_turbo_control`,
 * `krea_2_turbo_control` and their siblings in `SHIPPED_CONTROL_WEIGHTS`
 * (`crates/sceneworks-core/src/control_weights.rs`) — are engine ids for a base catalog entry
 * loaded WITH a ControlNet attached. They ship no tiered downloads because they are not manifest
 * models at all, so they can never be one of epic 22723 E1's `<modelId>:<tier>:<backend>` cells and
 * `--list` has nothing to ask about them. `config/inference-provider-closures.json` declares four
 * such lanes and those declarations must STAY — production control renders load under the `_control`
 * provider id and their route currency is graded per (backend, provider) — so the closure table and
 * this table are DELIBERATELY not in bijection. `scripts/stale-lane-report.mjs` reports them under
 * their own heading (`shippedControlOverlayProviders`) rather than as "uncapturable".
 *
 * `arms` is bound to the Rust dispatch by `every declared adapter arm is a provider that lane's
 * adapter really dispatches` (scripts/measure-memory-catalog.test.mjs): a declared arm with no
 * `match provider` arm behind it reds, so `no_adapter_arm` is derived from the adapter and not from
 * this table's spelling.
 */
export const PROVIDER_FAMILIES = Object.freeze({
  qwen_image: {
    env: "QWEN_IMAGE", repo: "SceneWorks/qwen-image-mlx", arms: ["mlx", "candle"],
    sourceCapture: true, physical: true,
  },
  // `z_image_edit` anchors ride this family too (sc-22724): the catalog id is an alias for the
  // Turbo provider driven in `edit_image` mode (worker engines.rs `z_image_edit → z_image_turbo`),
  // and its manifest entry ships the same Turbo tiers, which `tierDownload` resolves by model id.
  // Both Qwen edit catalog ids (`qwen_image_edit_2511` and `..._lightning`) plan this ONE engine
  // provider (worker `qwen.rs` `qwen_edit_engine_id`, `qwen_edit_candle.rs` `QWEN_EDIT_PROVIDER_ID`)
  // and ship the SAME per-tier rehost, which `tierDownload` resolves per model id. The Lightning id
  // additionally loads the pinned distill LoRA, declared as its `sideArtifact` below.
  qwen_image_edit: {
    env: "QWEN_IMAGE_EDIT",
    repo: "SceneWorks/qwen-image-edit-2511-mlx",
    arms: ["mlx", "candle"],
    sideArtifact: { qwen_image_edit_2511_lightning: QWEN_EDIT_LIGHTNING_LORA },
  },
  z_image_turbo: { env: "Z_IMAGE", repo: "SceneWorks/z-image-turbo-mlx", arms: ["mlx", "candle"] },
  // The undistilled base is a distinct engine provider (`z_image`) with its own artifact family
  // (sc-22724). Never the Turbo env: a base plan satisfied by Turbo weights re-labels Turbo's peaks.
  z_image: { env: "Z_IMAGE_BASE", repo: "SceneWorks/z-image-mlx", arms: ["mlx", "candle"] },
  krea_2_turbo: { env: "KREA", repo: "SceneWorks/krea-2-turbo-mlx", arms: ["mlx", "candle"] },
  // sc-22735. The UNDISTILLED Krea 2 base is a separate engine provider (`krea_2_raw`) served by
  // the same two crates as Turbo (`mlx-gen-krea` / `candle-gen-krea`) off its OWN tiered rehost, so
  // it gets its own env family: a raw plan satisfied by Turbo weights would re-label Turbo's peaks
  // as the true-CFG base model's, the `z_image` / `z_image_turbo` split for the same reason.
  krea_2_raw: { env: "KREA_RAW", repo: "SceneWorks/krea-2-raw-mlx", arms: ["mlx", "candle"] },
  // sc-22735. The VIDEO member of the family, and MLX-ONLY: `mlx-gen-krea-realtime` is the only
  // engine that registers it, the worker's video route table has no candle arm for it, and every
  // manifest download is `platforms: ["macos"]`. The tier root is the `<tier>/` subdir of the one
  // rehost, the same shape as every other tiered family here.
  krea_realtime_14b: { env: "KREA_REALTIME", repo: "SceneWorks/krea-realtime-14b-mlx", arms: ["mlx"] },
  // sc-22729. The SDXL FAMILY: five catalog models the worker routes onto ONE engine id (`sdxl`)
  // on both lanes, each with its own independently pinned tiered rehost. These rows are keyed by
  // MODEL id rather than provider id — `classifyAnchor` prefers a model-keyed family — because the
  // engine id is not an artifact identity: `candle-gen-sdxl` seals a per-route repository and mints
  // a per-route calibration fingerprint, so a `realvisxl` anchor bound to the base-SDXL env family
  // would re-label base SDXL's peaks as the finetune's.
  //
  // `components` are the three caller-staged SDXL corequisites (`tokenizer_clip_l`,
  // `tokenizer_clip_bigg`, `vae_fp16_fix`). `candle-gen-sdxl`'s `validate_shared_component_revisions`
  // REQUIRES all three at exact upstream revisions, so a candle capture stages the same corequisite
  // snapshots the worker's `attach_required_components` stages. The MLX turnkey is self-contained
  // and ignores them, so they are bound on both lanes and simply unused on one.
  //
  // `sdxlRoute` marks the members `candle-gen-sdxl` seals through `SDXL_ROUTES`. It is carried by
  // ALL FIVE, not only the two that disagree today: the check is over the engine's declaration, so
  // a future revision drift on any member is caught the same way rather than needing a new entry.
  sdxl: { provider: "sdxl", env: "SDXL", repo: "SceneWorks/sdxl-base-mlx", arms: ["mlx", "candle"], components: SDXL_COMPONENTS, sdxlRoute: true },
  realvisxl: { provider: "sdxl", env: "REALVISXL", repo: "SceneWorks/realvisxl-mlx", arms: ["mlx", "candle"], components: SDXL_COMPONENTS, sdxlRoute: true },
  realvisxl_lightning: {
    provider: "sdxl", env: "REALVISXL_LIGHTNING", repo: "SceneWorks/realvisxl-lightning-mlx",
    arms: ["mlx", "candle"], components: SDXL_COMPONENTS, sdxlRoute: true,
  },
  // The candle lane routes all five `sdxl` members (`routing/candle.rs` `is_sdxl_family_candle_model`
  // / `SDXL_CONTROL_MODELS`), so all five are DECLARED on both lanes. Whether a member's candle cell
  // is capturable TODAY is a derived fact, not a table entry: `sdxlRoute` asks `sdxlCandleRouteDrift`
  // to compare the engine's own `SDXL_ROUTES` revision with the one the manifest ships. Only the two
  // Illustrious routes disagree at inference c6d6a4db, and only for as long as they disagree.
  illustrious_xl_v1: {
    provider: "sdxl", env: "ILLUSTRIOUS_XL_V1", repo: "SceneWorks/illustrious-xl-v1-mlx",
    arms: ["mlx", "candle"], components: SDXL_COMPONENTS, sdxlRoute: true,
  },
  illustrious_xl_v2: {
    provider: "sdxl", env: "ILLUSTRIOUS_XL_V2", repo: "SceneWorks/illustrious-xl-v2-mlx",
    arms: ["mlx", "candle"], components: SDXL_COMPONENTS, sdxlRoute: true,
  },
  // The InstantID backbone IS the plain RealVisXL rehost (`image_jobs/instantid.rs`
  // `INSTANTID_SDXL_REPO`), bound through its own env family so an InstantID plan can never be
  // satisfied by a plain `realvisxl` root and vice versa. `stagedEnv` is the identity stack: the
  // worker fetches it on first use from a pinned repo rather than declaring it as a manifest
  // download, so there is nothing for the harness to resolve — the operator stages it and the
  // capture binds the staged copy through the same env seams the worker reads.
  instantid_realvisxl: {
    provider: "instantid", env: "INSTANTID_REALVISXL", repo: "SceneWorks/realvisxl-mlx", arms: ["mlx", "candle"],
    components: SDXL_COMPONENTS,
    // Each staged directory declares the files the adapter opens INSIDE it (sc-22738): a directory
    // that exists but is half-staged is exactly as unloadable as an absent one.
    stagedEnv: [
      { env: "SCENEWORKS_INSTANTID_WEIGHTS", files: INSTANTID_IDENTITY_BUNDLE_FILES },
      { env: "SCENEWORKS_INSTANTID_CONTROLNET", files: [INSTANTID_CONTROLNET_WEIGHT_FILE] },
    ],
  },
  flux2_dev: { env: "FLUX2", repo: "SceneWorks/flux2-dev-mlx", arms: ["mlx", "candle"] },
  // sc-22727. TWO catalog models ride this ONE engine provider id (worker engines.rs:
  // `flux2_klein_9b_kv` declares `engine_id: flux2_klein_9b`), and they load DIFFERENT artifacts.
  // On MLX the engine tells them apart by the snapshot path AND by `LoadSpec::resolved_route`
  // (`KleinArtifactInventory::validate_resolved_route`, mlx-gen-flux2/src/artifact_inventory.rs);
  // on Candle ONLY by the snapshot path — `candle-gen-flux2` never reads `resolved_route`. Either
  // way the artifact is the discriminator, so the family carries a per-modelId override: a KV plan
  // resolved through the base rehost's env would re-label the base checkpoint's peaks as the KV
  // variant's.
  flux2_klein_9b: {
    env: "FLUX2_KLEIN", repo: "SceneWorks/flux2-klein-9b-mlx", arms: ["mlx", "candle"],
    variants: {
      flux2_klein_9b_kv: { env: "FLUX2_KLEIN_KV", repo: "SceneWorks/flux2-klein-9b-kv-mlx" },
    },
  },
  // The FLUX.1 family (sc-22726). `flux_dev`/`flux_schnell` are the two base text-to-image
  // providers; `pulid_flux` is the PuLID-FLUX character route, which loads the SAME
  // `SceneWorks/flux1-dev-mlx` backbone (worker image_jobs/pulid.rs `PULID_FLUX_REPO` and
  // pulid_candle.rs `PULID_CANDLE_FLUX_REPO`) and therefore shares the FLUX1_DEV env family, the
  // way `z_image_edit` shares the Turbo family. Its own manifest entry ships the same three tiers,
  // which `tierDownload` resolves by model id.
  flux1_dev: { env: "FLUX1_DEV", repo: "SceneWorks/flux1-dev-mlx", arms: ["mlx", "candle"] },
  flux1_schnell: { env: "FLUX1_SCHNELL", repo: "SceneWorks/flux1-schnell-mlx", arms: ["mlx", "candle"] },
  pulid_flux: {
    env: "FLUX1_DEV", repo: "SceneWorks/flux1-dev-mlx", arms: ["mlx", "candle"],
    // The identity stack is NOT a manifest download on either lane — the worker fetches it on first
    // use — so the anchor binds the operator's pre-staged bundle instead, through the env var both
    // worker lanes read (`SCENEWORKS_PULID_WEIGHTS`), under its strict Candle reading: a directory
    // already holding all five files (pulid_candle.rs `ensure_pulid_candle_weights`; the MLX lane
    // treats it as the directory to fill and outranks it with the `PULID_*` preset — see
    // `PULID_IDENTITY_BUNDLE_ENV` in the adapter's lib.rs). The list below is the adapter's
    // `PULID_IDENTITY_BUNDLE_FILES`, in the same order: the adapter checkpoint, the EVA tower, and
    // the three face models both engines read out of `face_dir` by name. The test parses lib.rs
    // and asserts the two lists are equal, so neither can drift alone.
    bundle: {
      env: "SCENEWORKS_PULID_WEIGHTS",
      files: [
        "pulid_flux_v0.9.1.safetensors",
        "eva02_clip_l_336.safetensors",
        "scrfd_10g.safetensors",
        "arcface_iresnet100.safetensors",
        "bisenet_parsing.safetensors",
      ],
    },
  },
  // The Mage-Flow family (sc-22733). SIX registered engine providers whose catalog ids are
  // identical to their engine ids (worker `engines.rs` MODEL_TABLE), each with its OWN tiered
  // rehost — so six rows rather than one shared family. What they DO share is the text encoder and
  // the VAE: those are bit-identical across all six variants and are hosted once in
  // `SceneWorks/Mage-Flow-Components-mlx`, which every Mage manifest entry declares as per-tier
  // `coRequisite` downloads. Both engines resolve that split through `LoadSpec::components`
  // (`mlx-gen-mage` `resolve_component_dirs`, `candle-gen-mage` `resolved_component_dirs`), so a
  // Mage anchor binds TWO roots: the variant's tier root and the components SNAPSHOT (the tier is
  // the first path element INSIDE it, which is why `components` resolves the snapshot rather than a
  // tier root).
  ...Object.fromEntries(
    [
      ["mage_flow", "MAGE_FLOW", "SceneWorks/Mage-Flow"],
      ["mage_flow_base", "MAGE_FLOW_BASE", "SceneWorks/Mage-Flow-Base"],
      ["mage_flow_turbo", "MAGE_FLOW_TURBO", "SceneWorks/Mage-Flow-Turbo"],
      ["mage_flow_edit", "MAGE_FLOW_EDIT", "SceneWorks/Mage-Flow-Edit"],
      ["mage_flow_edit_base", "MAGE_FLOW_EDIT_BASE", "SceneWorks/Mage-Flow-Edit-Base"],
      ["mage_flow_edit_turbo", "MAGE_FLOW_EDIT_TURBO", "SceneWorks/Mage-Flow-Edit-Turbo"],
    ].map(([provider, env, repo]) => [
      provider,
      { env, repo, arms: ["mlx", "candle"], components: MAGE_COMPONENTS },
    ]),
  ),
  // The SD3.5 family (sc-22730). Three DISTINCT engine providers, each with its own tiered rehost
  // and therefore its own env family — unlike `z_image_edit`/`pulid_flux`, none of them is an alias
  // for another's backbone, so serving one from another's artifact would re-label that route's
  // peaks. The catalog model id equals the engine provider id on BOTH lanes (worker engines.rs sets
  // `engine_id == sceneworks_id` for all three), so the anchor key's modelId and the plan row's
  // provider are the same token and `tierDownload` resolves the same manifest entry either way.
  // One `SceneWorks/sd3.5-<route>-mlx` repo serves both lanes at all three tiers.
  sd3_5_large: { env: "SD3_5_LARGE", repo: "SceneWorks/sd3.5-large-mlx", arms: ["mlx", "candle"] },
  sd3_5_large_turbo: { env: "SD3_5_LARGE_TURBO", repo: "SceneWorks/sd3.5-large-turbo-mlx", arms: ["mlx", "candle"] },
  sd3_5_medium: { env: "SD3_5_MEDIUM", repo: "SceneWorks/sd3.5-medium-mlx", arms: ["mlx", "candle"] },
  // The SANA family (sc-22731). Two routes, and the ONE family in this table whose two lanes load
  // DIFFERENT repositories: the MLX lane opens the per-tier SceneWorks turnkey, the Candle lane
  // opens the upstream dense diffusers snapshot at its ROOT (worker `image_jobs/base.rs`
  // `SANA_CANDLE_DIFFUSERS_REPO` / `SANA_SPRINT_CANDLE_DIFFUSERS_REPO`, resolved through
  // `huggingface_pinned_snapshot_dir`, which never descends into a tier sub-directory). That is
  // what `lanes` expresses; `tiered: false` is why the root carries no `<tier>` component.
  sana_1600m: {
    env: "SANA", repo: "SceneWorks/Sana_1600M_1024px_mlx", arms: ["mlx", "candle"],
    lanes: { candle: { env: "SANA_DENSE", repo: "Efficient-Large-Model/Sana_1600M_1024px_diffusers", tiered: false } },
  },
  sana_sprint_1600m: {
    env: "SANA_SPRINT", repo: "SceneWorks/Sana_Sprint_1.6B_1024px_mlx", arms: ["mlx", "candle"],
    lanes: { candle: { env: "SANA_SPRINT_DENSE", repo: "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", tiered: false } },
  },
  // The three Chroma1 routes (sc-22731). Separate receipt/evidence domains (SC-20788) over three
  // separate rehosts, so three env families — never one shared `CHROMA1`, which would let a Flash
  // plan be satisfied by HD weights. Both lanes open the SAME per-tier turnkey.
  chroma1_hd: { env: "CHROMA1_HD", repo: "SceneWorks/chroma1-hd-mlx", arms: ["mlx", "candle"] },
  chroma1_base: { env: "CHROMA1_BASE", repo: "SceneWorks/chroma1-base-mlx", arms: ["mlx", "candle"] },
  chroma1_flash: { env: "CHROMA1_FLASH", repo: "SceneWorks/chroma1-flash-mlx", arms: ["mlx", "candle"] },
  // MiniMax-H3 (sc-18663; the Candle lane and the reference entry added by sc-22737). ONE engine
  // provider id serves BOTH catalog entries — `mlx-gen-minimax-h3` and `candle-gen-minimax-h3` each
  // register a single `MODEL_ID`, and the two entries are two DiT partitions of it (`transformer/`
  // and `transformer_ref/`) which the engine selects from the CONDITIONING, not from the spec. So
  // one family row keyed on the provider, with the reference entry as a `variants` override.
  //
  // `upstream` is the dense `MiniMaxAI/MiniMax-H3` snapshot the manifest ships as a co-requisite:
  // it is the only tree carrying `vae/`, `audio_vae/`, `tokenizer/` and the `FL2VA/` documents, and
  // BOTH adapters make it the load ROOT while redirecting `transformer/` and `text_encoder/` onto
  // the packed rehost.
  //
  // The candle `bf16` leg of the BASE entry is the one cell that binds no rehost tier at all: the
  // manifest ships `SceneWorks/minimax-h3-mlx` `bf16` as `platforms: ["macos"]`, and the off-Mac
  // dense leg is `MiniMaxAI/MiniMax-H3` `bf16` shipped for `["windows","linux"]` with the weights
  // at the snapshot ROOT. `candle_load_plan` stages nothing for it (`quant.is_some() || is_reference`
  // is false), so declaring the rehost here would report `weights_missing` for a cell that is in
  // fact stageable from the upstream snapshot alone — wrong in the direction that hides work.
  minimax_h3: {
    env: "MINIMAX_H3", repo: "SceneWorks/minimax-h3-mlx", arms: ["mlx", "candle"],
    upstream: {
      env: "MINIMAX_H3_UPSTREAM", repo: "MiniMaxAI/MiniMax-H3",
      // sc-22738: what the load READS in this root, per tier. `bf16` adds the dense text encoder,
      // which is the file this host does not hold and which the classifier called runnable.
      requiredFiles: { all: MINIMAX_UPSTREAM_ROOT_FILES, bf16: [MINIMAX_TEXT_ENCODER_CONFIG] },
    },
    // The two DiT partitions live in the tier root on every rehost-backed cell, and `q4`/`q8` take
    // the packed text encoder from it as well (`bf16` takes the dense one from upstream above).
    requiredTierFiles: {
      all: MINIMAX_TIER_DIT_FILES,
      q4: [MINIMAX_TEXT_ENCODER_CONFIG],
      q8: [MINIMAX_TEXT_ENCODER_CONFIG],
    },
    artifacts: {
      candle: {
        // The one cell whose tier root IS the upstream snapshot root: everything the load opens is
        // in that flat tree, so the DiT partitions are required THERE. The text encoder and the
        // root documents are already covered by the `upstream` declaration above, which probes the
        // same directory — declaring them twice would only duplicate the reason string.
        bf16: {
          env: "MINIMAX_H3_UPSTREAM", repo: "MiniMaxAI/MiniMax-H3", layout: "flat",
          requiredTierFiles: MINIMAX_TIER_DIT_FILES,
        },
      },
    },
    // The reference entry stages the tier tree at EVERY tier, on both lanes, because
    // `transformer_ref/` is published only in the rehost — which is why the manifest ships its
    // `bf16` rehost download for `["macos","windows","linux"]` while the base entry's is macOS-only.
    // Dropping the base entry's candle-bf16 override is therefore not a tidy-up: it is the whole
    // difference between the two entries' artifact axes.
    variants: { minimax_h3_ref: { artifacts: undefined } },
  },
  // The harness prepares and binds the LTX-2.5 snapshot itself (`--ltx25-snapshot-root`), for
  // whichever lane the plan routes: BOTH engine ids below are served from the same public snapshot,
  // and both are declared here because the plan row's `provider` is what selects the family.
  // `requiredSnapshotEntries` (sc-22738) are what each arm opens BESIDE `<variant>/<tier>`: the
  // official stage-two refinement LoRA the dev variant attaches on both lanes, and — MLX only —
  // the stock enhancer co-requisite `load_artifact` demands of every load. Neither is inside the
  // load root, so the snapshot probe above cannot see them; without these an anchor whose planned
  // cases include the dev variant classified `runnable` and failed after the booked session opened.
  //
  // sc-22738: the MLX arm emits a `physical_mlx` sourceCapture on EVERY run (`mlx_ltx25.rs`
  // `prepare_source_capture` → `source_capture`, which persists the canonical selected/reference AV
  // pair under `$SCENEWORKS_MEMORY_CAPTURE_DIR/$SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX`), so it needs
  // exactly the raw-log provenance the Qwen MLX arm needs — but NOT the qwen-only currency receipt
  // rule, which is why this is `sourceCapture` and not `physical`. The candle row below emits none
  // (`candle.rs` has no sourceCapture site), so it must NOT be given the pair: the harness refuses a
  // `--raw-log-dir` whose provider returned no sourceCapture.
  ltx_2_5: {
    ltx25: true, repo: LTX25_REPOSITORY, arms: ["mlx"], sourceCapture: true,
    requiredSnapshotEntries: [
      { path: LTX25_DEV_REFINEMENT_LORA },
      { path: LTX25_ENHANCER_DIR, dir: true },
    ],
  },
  // The Candle arm loads LTX-2.5 under its own engine id (candle.rs `LTX25_ID`, `candle-gen-ltx`
  // `MODEL_25_ID`), so the candle plan rows name `ltx_2_5_distilled` while the anchor key — and
  // therefore the manifest download the snapshot root resolves through — stays `ltx_2_5`.
  ltx_2_5_distilled: {
    ltx25: true, repo: LTX25_REPOSITORY, arms: ["candle"],
    requiredSnapshotEntries: [{ path: LTX25_DEV_REFINEMENT_LORA }],
  },
  // The turnkey still family (sc-22732). Five catalog models over three engine crates, each a plain
  // reference-free text-to-image route with its text encoder, transformer and decoder packed inside
  // the per-tier snapshot — so one root is the whole load and no `upstream` or `bundle` is needed.
  // The engine id equals the catalog model id for all five, so the family key, the plan row's
  // `provider` and the anchor key's modelId are the same token.
  kolors: { env: "KOLORS", repo: "SceneWorks/kolors-mlx", arms: ["mlx", "candle"] },
  lens: { env: "LENS", repo: "SceneWorks/lens-mlx", arms: ["mlx", "candle"] },
  // Its OWN rehost at its OWN revision, split from base Lens the way `flux1_schnell` is split from
  // `flux1_dev`: a turbo plan satisfied by base weights would re-label the base model's peaks.
  lens_turbo: { env: "LENS_TURBO", repo: "SceneWorks/lens-turbo-mlx", arms: ["mlx", "candle"] },
  // Ideogram is the only shipped family whose tiers do NOT all come from one repository, which is
  // what `tiers` exists for: `q4`/`q8` are the packed `SceneWorks/ideogram-4-mlx` turnkey, and
  // `bf16` is the separate `SceneWorks/ideogram-4` repo at a separate revision (worker
  // `image_jobs/base.rs` `IDEOGRAM_BF16_REPO`, and the manifest's own third `downloads[]` entry).
  // Without the override `tierDownload` would fall back to the packed repo's q4 download — its
  // "any download from this repo" arm — and bind bf16 to the wrong repository AND the wrong
  // revision, which the record's loadability fingerprint is the only place that would ever show.
  // Both Ideogram members share both repositories at both revisions and differ by provider.
  ideogram_4: {
    env: "IDEOGRAM", repo: "SceneWorks/ideogram-4-mlx", arms: ["mlx", "candle"],
    tiers: { bf16: { env: "IDEOGRAM_BF16", repo: "SceneWorks/ideogram-4" } },
  },
  ideogram_4_turbo: {
    env: "IDEOGRAM", repo: "SceneWorks/ideogram-4-mlx", arms: ["mlx", "candle"],
    tiers: { bf16: { env: "IDEOGRAM_BF16", repo: "SceneWorks/ideogram-4" } },
  },
  // The Wan 2.2 family (sc-22736). The FIRST families whose artifact is per (lane, TIER) rather
  // than per lane, which is why `familyArtifact` exists: each route ships a `SceneWorks/…-mlx`
  // rehost on macOS and a separate `SceneWorks/…-candle` rehost on Windows/Linux, and the candle
  // rehosts carry `q4` and `q8` ONLY — the candle dense leg is the upstream `Wan-AI/…-Diffusers`
  // checkpoint, which the manifest ships with no pinned revision and with the weights at the
  // snapshot ROOT rather than under a `bf16/` subtree (`layout: "flat"`).
  //
  // One env family per (route, lane, layout), never one shared `WAN22`: the three routes are three
  // different checkpoints, and a plan for one satisfied by another's weights would re-label its
  // peaks.
  wan2_2_ti2v_5b: {
    // sc-22738: this route's MLX generator publishes a memory-strategy contract only if the
    // pinned `mlx-gen-wan` loader seals its receipt — see `parseWanMlxSealedProviders`.
    wanMlxSeal: true,
    env: "WAN22_TI2V_5B_MLX", repo: "SceneWorks/wan2.2-ti2v-5b-mlx", arms: ["mlx", "candle"],
    artifacts: {
      candle: {
        "*": { env: "WAN22_TI2V_5B_CANDLE", repo: "SceneWorks/wan2.2-ti2v-5b-candle" },
        bf16: { env: "WAN22_TI2V_5B_DENSE", repo: "Wan-AI/Wan2.2-TI2V-5B-Diffusers", layout: "flat" },
      },
    },
  },
  wan2_2_t2v_14b: {
    // sc-22738: this route's MLX generator publishes a memory-strategy contract only if the
    // pinned `mlx-gen-wan` loader seals its receipt — see `parseWanMlxSealedProviders`.
    wanMlxSeal: true,
    env: "WAN22_T2V_A14B_MLX", repo: "SceneWorks/wan2.2-t2v-a14b-mlx", arms: ["mlx", "candle"],
    artifacts: {
      candle: {
        "*": { env: "WAN22_T2V_A14B_CANDLE", repo: "SceneWorks/wan2.2-t2v-a14b-candle" },
        bf16: { env: "WAN22_T2V_A14B_DENSE", repo: "Wan-AI/Wan2.2-T2V-A14B-Diffusers", layout: "flat" },
      },
    },
  },
  wan2_2_i2v_14b: {
    // sc-22738: this route's MLX generator publishes a memory-strategy contract only if the
    // pinned `mlx-gen-wan` loader seals its receipt — see `parseWanMlxSealedProviders`.
    wanMlxSeal: true,
    env: "WAN22_I2V_A14B_MLX", repo: "SceneWorks/wan2.2-i2v-a14b-mlx", arms: ["mlx", "candle"],
    artifacts: {
      candle: {
        "*": { env: "WAN22_I2V_A14B_CANDLE", repo: "SceneWorks/wan2.2-i2v-a14b-candle" },
        bf16: { env: "WAN22_I2V_A14B_DENSE", repo: "Wan-AI/Wan2.2-I2V-A14B-Diffusers", layout: "flat" },
      },
    },
  },
  // SCAIL-2 (sc-22736) is the opposite shape and the reason `artifacts` is an override rather than
  // the rule: the manifest ships ONE `SceneWorks/scail2-mlx` repository, with all three tiers, on
  // `platforms: ["macos", "windows", "linux"]`, and BOTH engine lanes open that same per-tier
  // turnkey — which is exactly why the two lanes' calibration identities carry a backend token.
  scail2_14b: { env: "SCAIL2", repo: "SceneWorks/scail2-mlx", arms: ["mlx", "candle"] },
  // Bernini (sc-22737). ONE engine provider id (`bernini`) serves BOTH shipped catalog entries —
  // the video entry `bernini` and the still entry `bernini_image` — because they are not two
  // providers: `crates/sceneworks-worker/src/engines.rs` maps `bernini_image` onto `engine_id:
  // "bernini"`, and `video_jobs/bernini.rs` calls `inference_runtime::load("bernini")` for the
  // video entry. The plan row's `provider` is what selects the family, so ONE row serves both, and
  // no `variants` override is needed: the two entries load the SAME artifact at the same tier and
  // differ only in the modality of the render.
  //
  // The two lanes load DIFFERENT repositories, which is why this needs the per-lane override: the
  // manifest ships `SceneWorks/bernini-mlx` per-tier on `["macos"]` and `SceneWorks/bernini` — a
  // single untiered download — on `["windows","linux"]`. The off-Mac download names no `variant`
  // because the tier subtrees (`q4/`, `q8/`, `bf16/`) live INSIDE that one snapshot, so the load
  // root is still the tier directory and the layout stays `tiered`.
  bernini: {
    env: "BERNINI", repo: "SceneWorks/bernini-mlx", arms: ["mlx", "candle"],
    artifacts: { candle: { "*": { env: "BERNINI_CANDLE", repo: "SceneWorks/bernini" } } },
  },
  // LTX-2.3 (sc-22737). Like LTX-2.5 above, the Candle arm loads this family under its OWN engine
  // id (`candle-gen-ltx`'s distilled `MODEL_ID`, spelled `ltx_2_3_distilled` in candle.rs), so the
  // Candle plan rows name that provider while the anchor key — and therefore the manifest download
  // the root resolves through — stays `ltx_2_3`. Both rows point at the ONE rehost the manifest
  // ships for all three platforms.
  //
  // BOTH lanes bind the DENSE GEMMA text encoder, which is a sibling directory of the same
  // snapshot rather than a separate repository (`SCENEWORKS_LTX_TEXT_ENCODER_ROOT`, validated by
  // BOTH adapters against `<snapshot>/gemma` — `mlx.rs#ltx_load_spec` and the candle twin). That
  // is what `siblingRoots` declares — see its use in `describeAnchor`.
  //
  // sc-22738: the MLX row was declared WITHOUT it, on the belief that only the Candle arm read the
  // root. `mlx.rs#ltx_load_spec` has required `SCENEWORKS_LTX_TEXT_ENCODER_ROOT` since sc-18808
  // landed the video arm, so all three booked `ltx_2_3:*:mlx` captures died on
  // `required environment variable SCENEWORKS_LTX_TEXT_ENCODER_ROOT is not set` — after the tier
  // root had already been probed and the capture scheduled. `every SCENEWORKS_LTX_* root the
  // adapters require is bound by the LTX-2.3 rows` (measure-memory-catalog.test.mjs) now derives
  // the requirement from the adapter sources, so neither row can drift from its arm again.
  //
  // The resolution is PRODUCTION's own: `video_jobs/ltx.rs#bundled_ltx_gemma_dir` takes the
  // selected tier dir's parent snapshot and joins `gemma`, which is exactly
  // `path.dirname(resolved.root)/gemma` here. Production additionally honors a `$LTX_GEMMA_DIR`
  // override and, failing that, scans SIBLING snapshot revisions (sc-14377) — neither is mirrored,
  // and deliberately: both adapters snapshot-validate the text-encoder root against the SAME
  // repository AND revision as the tier root, so a cross-revision or operator-overridden gemma is
  // refused by the arm. The mirrored branch is the only one a capture can use.
  //
  // There is no `ltx_2_3:bf16:candle` cell, and its absence is a ROUTING fact rather than an
  // omission: the manifest ships LTX-2.3's `bf16` download as `platforms: ["macos"]`, and the
  // worker's own Candle tier resolver
  // (`video_jobs/candle.rs#candle_ltx_bundle_tier_across_revisions`) returns `None` for
  // `CandleLtxTier::Bf16`. `measure-memory-catalog.test.mjs` asserts that exemption against BOTH
  // of those sources, so it cannot outlive either reason.
  ltx_2_3: {
    env: "LTX", repo: LTX_2_3_REPOSITORY, arms: ["mlx"],
    siblingRoots: [{ env: "LTX_TEXT_ENCODER", dir: "gemma" }],
  },
  ltx_2_3_distilled: {
    env: "LTX", repo: LTX_2_3_REPOSITORY, arms: ["candle"],
    siblingRoots: [{ env: "LTX_TEXT_ENCODER", dir: "gemma" }],
  },
  // sc-22734. The SenseNova-U1 FAMILY: six catalog models the worker routes onto TWO engine ids on
  // both lanes — `sensenova_u1_8b` (the 50-step quality path) and `sensenova_u1_8b_fast` (the
  // 8-step distill), each carrying a base id and two infographic finetunes.
  //
  // These rows are keyed by MODEL id rather than provider id — `familyFor` prefers a model-keyed
  // row that declares the plan's provider — because the engine id is NOT the artifact identity
  // here. The six models ship six INDEPENDENTLY PINNED tiered rehosts (six repositories, six
  // revisions), and both engines mint a per-ROUTE calibration fingerprint, so an infographic anchor
  // bound to the base SenseNova env family would load base weights and re-label base SenseNova's
  // peaks as the finetune's. A provider-keyed table cannot express that: it has one repo per engine
  // id, and `tierDownload` would be asked for a base-repo download the finetune's manifest entry
  // does not ship.
  sensenova_u1_8b: {
    provider: "sensenova_u1_8b", env: "SENSENOVA_U1_8B",
    repo: "SceneWorks/sensenova-u1-8b-mlx", arms: ["mlx", "candle"],
  },
  sensenova_u1_8b_infographic_v2: {
    provider: "sensenova_u1_8b", env: "SENSENOVA_U1_8B_INFOGRAPHIC_V2",
    repo: "SceneWorks/sensenova-u1-8b-infographic-v2-mlx", arms: ["mlx", "candle"],
  },
  sensenova_u1_8b_infographic_v3: {
    provider: "sensenova_u1_8b", env: "SENSENOVA_U1_8B_INFOGRAPHIC_V3",
    repo: "SceneWorks/sensenova-u1-8b-infographic-v3-mlx", arms: ["mlx", "candle"],
  },
  sensenova_u1_8b_fast: {
    requiredTierFiles: [SENSENOVA_DISTILL_MERGED_MARKER],
    provider: "sensenova_u1_8b_fast", env: "SENSENOVA_U1_8B_FAST",
    repo: "SceneWorks/sensenova-u1-8b-fast-mlx", arms: ["mlx", "candle"],
  },
  sensenova_u1_8b_infographic_v2_fast: {
    requiredTierFiles: [SENSENOVA_DISTILL_MERGED_MARKER],
    provider: "sensenova_u1_8b_fast", env: "SENSENOVA_U1_8B_INFOGRAPHIC_V2_FAST",
    repo: "SceneWorks/sensenova-u1-8b-infographic-v2-fast-mlx", arms: ["mlx", "candle"],
  },
  sensenova_u1_8b_infographic_v3_fast: {
    requiredTierFiles: [SENSENOVA_DISTILL_MERGED_MARKER],
    provider: "sensenova_u1_8b_fast", env: "SENSENOVA_U1_8B_INFOGRAPHIC_V3_FAST",
    repo: "SceneWorks/sensenova-u1-8b-infographic-v3-fast-mlx", arms: ["mlx", "candle"],
  },
});

/**
 * The family row that serves one anchor.
 *
 * The default key is the PROVIDER, so a catalog alias rides its engine's row (`z_image_edit` on
 * `z_image_turbo`) and a lane-specific engine id keeps selecting the family (LTX-2.5's two ids).
 * sc-22729 and sc-22734 add the inverse case: several catalog models on ONE engine id, each with its own
 * artifact family. A MODEL-keyed row wins for those — but ONLY when it declares the provider it
 * belongs to and that provider is the one the plan named, so a model-keyed row can never capture
 * an anchor that some other engine serves.
 */
export function familyFor(modelId, provider, families = PROVIDER_FAMILIES) {
  // Model-keyed resolution — and the provider-keyed `variants` override (sc-22727's FLUX.2 klein
  // pair) — both live in `providerFamily`, so this is a single delegation.
  return providerFamily(provider, modelId, families);
}

export function fail(message) {
  throw new Error(message);
}

/**
 * The artifact family one anchor binds: the provider's row, with any per-modelId override applied.
 * A provider that serves several catalog models from ONE registry id (sc-22727's two klein models)
 * declares the divergent members under `variants`; everything else is the row itself.
 */
export function providerFamily(provider, modelId, families = PROVIDER_FAMILIES) {
  // sc-22729/sc-22734's MODEL-keyed rows win first, but ONLY when the row declares the provider it belongs
  // to and that provider is the one the plan named — so a model-keyed row can never capture an
  // anchor some other engine serves.
  const scoped = families[modelId];
  if (scoped?.provider !== undefined && scoped.provider === provider) return scoped;
  const family = families[provider];
  if (!family) return undefined;
  const variant = family.variants?.[modelId];
  return variant ? { ...family, ...variant, variants: undefined } : family;
}

export function anchorParts(key) {
  const match = /^([a-z][a-z0-9_]*):(q4|q8|bf16):(mlx|candle)$/.exec(key);
  if (!match) fail(`not an anchor key: ${key}`);
  return { modelId: match[1], tier: match[2], backend: match[3] };
}

/** `--adapter` is a binary path, or a JSON array command (the way the harness fixtures are driven). */
export function providerCommand(adapter) {
  if (adapter.trimStart().startsWith("[")) {
    const command = JSON.parse(adapter);
    if (!Array.isArray(command) || command.length === 0 || command.some((part) => typeof part !== "string")) {
      fail("--adapter JSON form must be a non-empty array of strings");
    }
    return command;
  }
  return [path.resolve(adapter)];
}

export function anchorSlug(key) {
  return key.replaceAll(":", "-").replaceAll("_", "-");
}

export function parseArgs(argv) {
  const args = {
    backend: null, adapter: null, inferenceRepo: null, workDir: null, campaign: null,
    anchors: null, models: [], skipCurrent: false, dryRun: false, commit: true, hfCache: [], list: false,
  };
  const value = (flag, index) => {
    const selected = argv[index + 1];
    if (selected === undefined || selected.startsWith("--")) fail(`${flag} requires one value`);
    return selected;
  };
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    switch (flag) {
      case "--backend": args.backend = value(flag, index); index += 1; break;
      case "--adapter": args.adapter = value(flag, index); index += 1; break;
      case "--inference-repo": args.inferenceRepo = value(flag, index); index += 1; break;
      case "--work-dir": args.workDir = value(flag, index); index += 1; break;
      case "--campaign": args.campaign = value(flag, index); index += 1; break;
      case "--hf-cache": args.hfCache.push(value(flag, index)); index += 1; break;
      case "--anchors": args.anchors = value(flag, index).split(",").filter(Boolean); index += 1; break;
      case "--model": args.models.push(value(flag, index)); index += 1; break;
      case "--skip-current": args.skipCurrent = true; break;
      case "--dry-run": args.dryRun = true; break;
      case "--no-commit": args.commit = false; break;
      case "--list": args.list = true; break;
      default: fail(`unknown argument ${flag}`);
    }
  }
  if (!["mlx", "candle"].includes(args.backend)) fail("--backend must be mlx or candle");
  if (!args.list) {
    if (!args.workDir) fail("--work-dir is required (a directory OUTSIDE the checkout)");
    if (!args.inferenceRepo) fail("--inference-repo is required");
    if (!args.dryRun && !args.adapter) fail("--adapter is required for a real capture");
  }
  if (args.campaign && !/^[A-Za-z0-9._-]+$/.test(args.campaign)) {
    fail("--campaign must be one path segment, e.g. sc-12345");
  }
  return args;
}

export async function readPlan(root = ROOT) {
  const plan = JSON.parse(await readFile(path.join(root, PLAN_PATH), "utf8"));
  if (!plan.anchors || typeof plan.anchors !== "object") fail(`${PLAN_PATH} carries no anchors object`);
  return plan;
}

export async function readManifestModels(root = ROOT) {
  const body = await readFile(path.join(root, MANIFEST_PATH), "utf8");
  return JSON.parse(stripJsoncComments(body)).models;
}

export async function readDeclaredLanes(root = ROOT) {
  const config = JSON.parse(await readFile(path.join(root, ANCHOR_LOADER_CONFIG_PATH), "utf8"));
  return new Set(Object.keys(config.models ?? {}));
}

/** `<backend>:<provider>` keys with a crate-closure declaration (runbook §7c). */
export async function readDeclaredProviders(root = ROOT) {
  const config = JSON.parse(await readFile(path.join(root, PROVIDER_CLOSURE_CONFIG_PATH), "utf8"));
  return new Set(Object.keys(config.providers ?? {}));
}

export async function readMatrixCurrency(root = ROOT) {
  let matrix;
  try {
    matrix = JSON.parse(await readFile(path.join(root, MATRIX_PATH), "utf8"));
  } catch {
    return new Map();
  }
  const current = new Map();
  for (const anchor of matrix.anchors ?? []) {
    current.set(`${anchor.modelId}:${anchor.tier}:${anchor.backend}`, anchor.current === true);
  }
  return current;
}

/**
 * The measured lower bounds the store already holds, keyed like an anchor
 * (`<modelId>:<tier>:<backend>`), each with whether it is CURRENT (sc-22738).
 *
 * A bound is evidence about a cell exactly as a record is — the run established that this cell's
 * peak is at or above the footprint the guard stopped it at — but it is carried in the store's
 * `exceededBounds` array and its bundle's `records` array is empty, so neither `capturedInCampaign`
 * (which indexes `records`) nor the matrix's currency map (which publishes only anchors) could see
 * it. A cell the host has already proved it cannot finish therefore classified `runnable` and the
 * next `--skip-current` re-run would book the same 76-minute render again.
 *
 * Currency is the SAME rule a record's is (`generate-memory-matrix.mjs`, both `current:` sites):
 * the row's `source.loaderClosureDigest` against the digest `config/anchor-loader-closures.json`
 * carries for that `(modelId, backend)` at the pinned inference revision. So a bound stops
 * classifying its cell the moment a pin bump or a closure edit stales it — the same moment it stops
 * refusing anything in production — and the cell becomes capturable again with no edit here.
 */
export async function readExceededBounds(root = ROOT) {
  const bounds = new Map();
  let store;
  let closures;
  try {
    store = JSON.parse(await readFile(path.join(root, ANCHOR_STORE_PATH), "utf8"));
    closures = JSON.parse(await readFile(path.join(root, ANCHOR_LOADER_CONFIG_PATH), "utf8"));
  } catch {
    return bounds;
  }
  for (const bound of store.exceededBounds ?? []) {
    if (!bound.modelId || !bound.tier || !bound.backend) continue;
    const declared = closures.models?.[`${bound.modelId}:${bound.backend}`]?.digest;
    bounds.set(`${bound.modelId}:${bound.tier}:${bound.backend}`, {
      current: declared !== undefined && declared === bound.source?.loaderClosureDigest,
      source: bound.source?.path ?? ANCHOR_STORE_PATH,
      observedFootprintBytes: bound.observedFootprintBytes ?? null,
    });
  }
  return bounds;
}

export async function compiledInferencePin(root = ROOT) {
  const source = await readFile(path.join(root, ADAPTER_LIB_PATH), "utf8");
  const match = /^pub const INFERENCE_PIN: &str = "([0-9a-f]{40})";/m.exec(source);
  if (!match) fail(`${ADAPTER_LIB_PATH} declares no INFERENCE_PIN`);
  return match[1];
}

/** The manifest download that ships `tier` of `repo` for `modelId` — its revision names the snapshot. */
export function tierDownload(models, modelId, repo, tier) {
  const model = models.find((entry) => entry.id === modelId);
  if (!model) fail(`manifest has no model ${modelId}`);
  const downloads = (model.downloads ?? []).filter((download) => download.repo === repo);
  const primary = downloads.find((download) => download.variant === tier && !download.coRequisite);
  if (primary) return primary;
  const any = downloads.find((download) => /^[0-9a-f]{40}$/.test(download.revision ?? ""));
  if (!any) fail(`manifest ${modelId} has no download from ${repo}`);
  return any;
}

/**
 * The manifest download that ships EXACTLY `tier` of `repo` for `modelId`, or `undefined`.
 *
 * [`tierDownload`] deliberately falls back to any pinned download of the repository, because every
 * family before sc-22736 rehosted all three tiers under ONE revision and the fallback merely names
 * that revision. A per-(lane, tier) artifact cannot use it: the question there is whether this
 * repository ships this tier at all, and the fallback answers "yes" for every tier of every
 * repository that ships one.
 */
export function tierVariantDownload(models, modelId, repo, tier) {
  const model = models.find((entry) => entry.id === modelId);
  if (!model) fail(`manifest has no model ${modelId}`);
  return (model.downloads ?? []).find(
    (download) => download.repo === repo && download.variant === tier && !download.coRequisite,
  );
}

/**
 * The ARTIFACT one anchor binds, resolved per (lane, tier).
 *
 * Every family before sc-22736 rehosts all three tiers of both lanes in ONE repository under a
 * `<tier>/` subtree, so the family row's `env`/`repo` is the whole answer. The Wan 2.2 family is
 * not shaped that way, and the manifest is where that shows:
 *
 * * each model ships a `SceneWorks/…-mlx` rehost on `platforms: ["macos"]` and a separate
 *   `SceneWorks/…-candle` rehost on `["windows","linux"]`, so the two lanes load DIFFERENT
 *   repositories at different revisions, and
 * * the candle rehosts ship `q4` and `q8` only — the candle bf16 leg is the UPSTREAM
 *   `Wan-AI/Wan2.2-*-Diffusers` checkpoint, which the manifest ships with NO pinned revision and
 *   with the weights at the snapshot root rather than under a `bf16/` subtree.
 *
 * A row therefore declares `artifacts[backend][tier]`, or `artifacts[backend]["*"]` for a whole
 * lane; anything an override omits falls back to the row itself. `layout: "flat"` marks the second
 * shape above — probing `<snapshot>/bf16` there would report `weights_missing` for a cell that is
 * staged, which is the wrong answer in the direction that hides work.
 */
export function familyArtifact(family, backend, tier) {
  // Two override shapes stack, narrowest last:
  //
  // * `lanes[backend]` (sc-22731) — a whole lane loads a different repository, and `tiered: false`
  //   there means the load root is the snapshot ITSELF, with no `<tier>` component.
  // * `artifacts[backend][tier]`, or `artifacts[backend]["*"]` (sc-22736) — one CELL loads a
  //   different repository, which is what the Wan 2.2 candle bf16 leg needs.
  // * `tiers[tier]` (sc-22732) — one TIER loads a different repository on BOTH lanes, which is what
  //   `ideogram_4`'s bf16 leg needs: it ships from `SceneWorks/ideogram-4` at its own revision while
  //   its q4/q8 siblings come from the packed `SceneWorks/ideogram-4-mlx` turnkey.
  //
  // Anything an override omits falls back to the family row.
  const lane = family.lanes?.[backend];
  const perTier = family.artifacts?.[backend];
  const override = perTier?.[tier] ?? perTier?.["*"];
  const merged = {
    env: family.env,
    repo: family.repo,
    ...(lane ?? {}),
    ...(family.tiers?.[tier] ?? {}),
    ...(override ?? {}),
  };
  // `layout: "flat"` and `tiered: false` say the same thing; the first is this function's word for
  // it and the second is the family table's.
  return { ...merged, layout: merged.layout ?? (merged.tiered === false ? "flat" : "tiered") };
}

/** Snapshot revision directories of `repo` staged under one hub root. */
async function hostSnapshotRevisions(hub, repo) {
  try {
    return await readdir(path.join(hub, `models--${repo.replaceAll("/", "--")}`, "snapshots"));
  } catch {
    return [];
  }
}

/**
 * Where `tier` of `artifact` lives on this host, the revision that names it, and why it is absent.
 *
 * A download the manifest pins resolves the way every family always has. A download the manifest
 * ships WITHOUT a revision (the upstream Wan Diffusers checkpoints) has no revision to probe with,
 * so the revision is read off whatever snapshot of that repository is staged here. That is not a
 * weaker binding than the pinned case in the only place it matters — the adapter arm still
 * validates the repository it was handed and the resolved revision is written into the record —
 * but it IS host-dependent, so the reason string says so rather than reporting a missing pin as a
 * missing snapshot.
 */
export async function resolveArtifactRoot(models, modelId, tier, artifact, hubs) {
  const suffix = artifact.layout === "flat" ? [] : [tier];
  const label = suffix.length > 0 ? `/${tier}` : "";
  const declared = tierVariantDownload(models, modelId, artifact.repo, tier);
  const revision = tierDownloadRevision(models, modelId, artifact.repo, tier, declared);
  if (revision) {
    const root = await firstExistingDirectory(
      hubs.map((hub) => snapshotPath(hub, artifact.repo, revision, ...suffix)),
    );
    return {
      root,
      revision,
      expected: snapshotPath(hubs[0], artifact.repo, revision, ...suffix),
      reason: `no ${artifact.repo}@${revision.slice(0, 8)}${label} on this host`,
    };
  }
  for (const hub of hubs) {
    for (const candidate of await hostSnapshotRevisions(hub, artifact.repo)) {
      const root = snapshotPath(hub, artifact.repo, candidate, ...suffix);
      if (await firstExistingDirectory([root])) return { root, revision: candidate, expected: root, reason: null };
    }
  }
  return {
    root: null,
    revision: null,
    expected: snapshotPath(hubs[0], artifact.repo, "<revision>", ...suffix),
    reason:
      `${artifact.repo} is shipped without a pinned revision and no snapshot of it is staged on ` +
      `this host, so ${modelId}:${tier} has no root to bind`,
  };
}

/**
 * The revision of a pinned rehost that ships this tier under one revision for all of them.
 *
 * Only consulted when the manifest declares no download for exactly `(repo, tier)`: a rehost that
 * ships `q4`, `q8` and `bf16` under one revision is still resolvable through [`tierDownload`], and
 * a repository that ships NO tier of this model at all is a declaration error rather than a host
 * one, so it fails loudly here instead of reporting `weights_missing`.
 */
function tierDownloadRevision(models, modelId, repo, tier, declared) {
  if (declared) return declared.revision ?? null;
  return tierDownload(models, modelId, repo, tier).revision ?? null;
}

export function hubRoots(hfCache = []) {
  // Explicit --hf-cache roots first (repeatable), then the HF env convention, then the app cache.
  const roots = [
    ...hfCache,
    process.env.HF_HUB_CACHE ?? (process.env.HF_HOME ? path.join(process.env.HF_HOME, "hub") : path.join(os.homedir(), ".cache", "huggingface", "hub")),
  ];
  if (process.platform === "darwin") {
    roots.push(path.join(
      os.homedir(), "Library", "Application Support", "SceneWorks", "data", "cache", "huggingface", "hub",
    ));
  }
  return roots;
}

export function snapshotPath(hub, repo, revision, ...rest) {
  return path.join(hub, `models--${repo.replaceAll("/", "--")}`, "snapshots", revision, ...rest);
}

/** Whether `candidate` is a readable regular file (a symlink into the HF blob store counts). */
async function isFile(candidate) {
  try {
    return (await stat(candidate)).isFile();
  } catch {
    return false;
  }
}

async function firstExistingDirectory(candidates) {
  for (const candidate of candidates) {
    try {
      if ((await stat(candidate)).isDirectory()) return candidate;
    } catch { /* absent */ }
  }
  return null;
}

/**
 * Decide what the run can do with one plan anchor: which adapter arm serves it, which weights
 * root it loads, and why it would be skipped. Pure apart from the directory probes.
 */
export async function classifyAnchor(key, planned, { models, backend, hubs, current, captured, bounds = new Map(), declaredLanes, declaredProviders, sdxlRoutes = null, wanMlxSealed = null, families = PROVIDER_FAMILIES }) {
  const parts = anchorParts(key);
  const row = {
    key, ...parts, provider: planned.provider, status: "runnable", reason: null, env: {}, roots: [],
    // sc-22738: an arm that emits a provider `sourceCapture` is the one that needs the raw-log pair
    // and the capture-dir environment. Defaulted here so no early-return row can reach
    // `measureAnchor` with the flag undefined; both terminal paths below set it from the family.
    sourceCapture: false, physical: false,
  };
  if (parts.backend !== backend) return { ...row, status: "other_backend", reason: `${parts.backend} lane` };
  const family = familyFor(parts.modelId, planned.provider, families);
  // No shipped family carries `harnessUnsupported` today (sc-22725 gave LTX-2.5's candle engine id
  // a real row). The status stays for the next provider whose adapter arm exists but whose
  // artifacts the harness cannot bind: it is the one refusal that is neither a missing arm nor a
  // missing declaration.
  //
  // KEPT DELIBERATELY (sc-22725 review): the `families` parameter above is a test seam and nothing
  // else — no caller passes it — and it exists so this otherwise-unreachable branch is driven by a
  // synthetic family rather than left uncovered. The alternative considered and rejected was
  // deleting the branch and the parameter together; that would make the next unbindable provider
  // report as `no_adapter_arm`, which is the wrong diagnosis and sends the reader to adapter work.
  if (family?.harnessUnsupported) return { ...row, status: "harness_unsupported", reason: family.harnessUnsupported };
  // sc-22729: the same refusal, scoped to ONE lane and DERIVED. A model whose engine cannot seal an
  // artifact identity for it on a lane is not a missing arm and not a missing declaration — the arm
  // exists and the plan declares the cell — so it reports the engine-side reason rather than
  // sending the reader to adapter work. With no inference checkout to read, there is no refusal at
  // all: the cell classifies normally and carries the note saying the comparison did not happen.
  if (family?.sdxlRoute && backend === "candle") {
    if (sdxlRoutes === null) {
      row.routeCheck = SDXL_ROUTES_UNCHECKED;
    } else {
      const drift = sdxlCandleRouteDrift(parts.modelId, parts.tier, sdxlRoutes, models);
      if (drift) return { ...row, status: "harness_unsupported", reason: drift };
    }
  }
  // sc-22738: the same shape again, on the MLX Wan lane. Declaration is not reach — the registry's
  // memory-strategy registration is shared by routes whose loaders do not all seal a receipt — so
  // the fact is read from the loader itself rather than from the declaration.
  if (family?.wanMlxSeal && backend === "mlx") {
    if (wanMlxSealed === null) {
      row.routeCheck = WAN_MLX_SEAL_UNCHECKED;
    } else {
      const gap = wanMlxSealGap(planned.provider, wanMlxSealed);
      if (gap) return { ...row, status: "harness_unsupported", reason: gap };
    }
  }
  if (!family || !family.arms.includes(backend)) {
    return {
      ...row, status: "no_adapter_arm",
      reason: `the ${backend} adapter implements no provider arm for ${planned.provider}`,
    };
  }
  if (captured.has(key)) return { ...row, status: "already_captured", reason: captured.get(key) };
  // sc-22738: a CURRENT measured lower bound is a captured cell, not an uncaptured one. The run that
  // produced it established the one fact a re-run could establish — this cell's peak is at or above
  // a footprint this host had to stop — so booking it again spends another guarded render (76
  // minutes for the bernini bf16 stop) to learn the same thing, and the only reason it stayed
  // `runnable` is that a bound's bundle carries no `records` entry for the classifier to see. A
  // STALE bound classifies nothing: the cell goes back to `runnable`, because the loader closure it
  // was measured under is no longer the one production loads.
  const bound = bounds.get(key);
  if (bound?.current) {
    return {
      ...row, status: "exceeded_current", current: true,
      reason:
        `exceeded bound recorded at the pinned inference revision (${bound.source}); the host `
        + "already proved this cell cannot complete under this loader closure",
    };
  }
  if (declaredLanes && !declaredLanes.has(`${parts.modelId}:${backend}`)) {
    return {
      ...row, status: "lane_undeclared",
      reason: `${parts.modelId}:${backend} has no loader-closure declaration in ${ANCHOR_LOADER_CONFIG_PATH}; --stamp-anchors would refuse it (runbook §7c)`,
    };
  }
  if (declaredProviders && !declaredProviders.has(`${backend}:${planned.provider}`)) {
    return {
      ...row, status: "provider_undeclared",
      reason: `${backend}:${planned.provider} has no crate-closure declaration in ${PROVIDER_CLOSURE_CONFIG_PATH}; the record would carry no closure digest (runbook §7c)`,
    };
  }
  if (current.get(key) === true) row.current = true;

  if (family.ltx25) {
    const download = tierDownload(models, parts.modelId, family.repo, parts.tier);
    const snapshot = await firstExistingDirectory(hubs.map((hub) => snapshotPath(hub, family.repo, download.revision)));
    row.roots.push({ label: "ltx25 snapshot", path: snapshot ?? snapshotPath(hubs[0], family.repo, download.revision) });
    if (!snapshot) return { ...row, status: "weights_missing", reason: `no ${family.repo}@${download.revision.slice(0, 8)} snapshot on this host` };
    const missingSnapshot = [];
    for (const entry of family.requiredSnapshotEntries ?? []) {
      const candidate = path.join(snapshot, entry.path);
      const present = entry.dir
        ? (await firstExistingDirectory([candidate])) !== null
        : await isFile(candidate);
      if (!present) missingSnapshot.push(entry.path);
    }
    if (missingSnapshot.length > 0) {
      return {
        ...row, status: "weights_missing",
        reason: `${family.repo} snapshot ${snapshot} is missing ${missingSnapshot.join(", ")}, which the ${backend} arm opens beside the load root`,
      };
    }
    row.ltx25SnapshotRoot = snapshot;
    row.sourceCapture = backend === "mlx" && family.sourceCapture === true;
    row.physical = backend === "mlx" && family.physical === true;
    // sc-22738: the weights identity a footprint hard stop would be bound against. It is the
    // runner's binding of what it set up for the capture, not a provider attestation — a killed
    // run attests nothing — so it names exactly the snapshot the row resolved above. LTX-2.5 rows
    // carry no inventory digest because the harness seals the snapshot itself.
    row.artifact = {
      repository: family.repo,
      resolvedRevision: download.revision,
      variant: parts.tier,
    };
    return row;
  }

  // The artifact is resolved per (lane, tier), not per family: SANA's two lanes load different
  // repositories (sc-22731) and a Wan 2.2 candle bf16 cell loads the upstream Diffusers checkpoint
  // rather than the packed rehost its q4/q8 siblings live in (sc-22736).
  const artifact = familyArtifact(family, backend, parts.tier);
  const resolved = await resolveArtifactRoot(models, parts.modelId, parts.tier, artifact, hubs);
  row.roots.push({
    label: artifact.layout === "flat" ? "snapshot root" : "tier root",
    path: resolved.root ?? resolved.expected,
  });
  if (!resolved.root) {
    return { ...row, status: "weights_missing", reason: resolved.reason };
  }
  const tierRoot = resolved.root;
  row.env[`SCENEWORKS_${artifact.env}_REPOSITORY`] = artifact.repo;
  row.env[`SCENEWORKS_${artifact.env}_REVISION`] = resolved.revision;
  row.env[`SCENEWORKS_${artifact.env}_ROOT`] = tierRoot;
  row.tierRoot = tierRoot;
  // A converter-written file the ENGINE requires INSIDE the resolved tier root, beyond the weights
  // the manifest download ships (sc-22734 review). The SenseNova `_fast` rehosts are the live case:
  // `mlx-gen-sensenova`'s `production_calibration_identity` and `candle-gen-sensenova`'s
  // `fast_spec_is_the_premerged_turnkey` both WITHHOLD the production identity when a `_fast` tier
  // root carries no `distill_merged.json`, because without the marker the loader merges the distill
  // LoRA at load and the resident shape is not the one the anchor prices. Nine `_fast` MLX cells
  // would therefore hard-fail at capture with an identity mismatch, hours into a booked session,
  // over a fact this probe can read in a millisecond. Classified by name instead.
  // sc-22738 generalized the declaration from a flat list to a per-tier one: MiniMax-H3's text
  // encoder is packed inside the tier root at q4/q8 and taken from the dense upstream root at bf16,
  // so which file a root must carry depends on the tier being classified.
  const requiredTierFiles = requiredFilesFor(artifact.requiredTierFiles ?? family.requiredTierFiles, parts.tier);
  const missingTierFiles = [];
  for (const file of requiredTierFiles) {
    try {
      if (!(await stat(path.join(resolved.root, file))).isFile()) missingTierFiles.push(file);
    } catch { missingTierFiles.push(file); }
  }
  if (missingTierFiles.length > 0) {
    return {
      ...row, status: "weights_missing",
      reason: `tier root ${resolved.root} is missing ${missingTierFiles.join(", ")}, which the engine requires before it will publish this cell's calibration identity`,
    };
  }
  if (family.bundle) {
    // A pre-staged loose-file bundle rather than an HF snapshot, so it is probed through the
    // operator env the worker itself honours. Absent or incomplete is `weights_missing` — the same
    // class as a missing tier root, and NOT "runnable" (a cell whose identity stack cannot be bound
    // is not measurable on this host, and saying otherwise sends an operator to book a capture).
    // Absolute before it is probed or handed on: the adapter canonicalizes the value from the
    // HARNESS's cwd (lib.rs `pulid_identity_bundle`), so a relative export that resolved here
    // against node's cwd would be probed in one directory and opened in another.
    const bundleRoot = process.env[family.bundle.env] ? path.resolve(process.env[family.bundle.env]) : undefined;
    row.roots.push({ label: "pulid bundle", path: bundleRoot ?? `$${family.bundle.env}` });
    if (!bundleRoot) {
      return { ...row, status: "weights_missing", reason: `${family.bundle.env} is unset; the PuLID identity bundle is not staged on this host` };
    }
    const missing = [];
    for (const file of family.bundle.files) {
      try {
        if (!(await stat(path.join(bundleRoot, file))).isFile()) missing.push(file);
      } catch { missing.push(file); }
    }
    if (missing.length > 0) {
      return { ...row, status: "weights_missing", reason: `${family.bundle.env} bundle ${bundleRoot} is missing ${missing.join(", ")}` };
    }
    row.env[family.bundle.env] = bundleRoot;
  }
  // A directory of the artifact's OWN snapshot that the adapter binds through its own env var,
  // beside the tier root (sc-22737, LTX-2.3's dense Gemma text encoder). Unlike `upstream` it is
  // not a second repository and unlike `components` it is not a co-requisite download: it is a
  // SIBLING of `<snapshot>/<tier>`, shipped inside the same snapshot at the same revision, which
  // is exactly how candle.rs validates it (`validate_huggingface_snapshot_root(root, repo, rev,
  // "gemma", …)`). Absent is `weights_missing` for the same reason a missing tier root is: the
  // load cannot open, and calling the cell `runnable` would send an operator to book a capture
  // that fails on its text encoder.
  for (const sibling of family.siblingRoots ?? []) {
    const siblingRoot = await firstExistingDirectory([path.join(path.dirname(resolved.root), sibling.dir)]);
    row.roots.push({
      label: `${sibling.dir} root`,
      path: siblingRoot ?? path.join(path.dirname(resolved.root), sibling.dir),
    });
    if (!siblingRoot) {
      return {
        ...row, status: "weights_missing",
        reason: `no ${sibling.dir}/ beside ${artifact.repo}@${resolved.revision.slice(0, 8)}/${parts.tier} on this host`,
      };
    }
    row.env[`SCENEWORKS_${sibling.env}_ROOT`] = siblingRoot;
  }
  if (family.upstream) {
    const upstream = tierDownload(models, parts.modelId, family.upstream.repo, parts.tier);
    const upstreamRoot = await firstExistingDirectory(hubs.map((hub) => snapshotPath(hub, family.upstream.repo, upstream.revision)));
    row.roots.push({ label: "upstream root", path: upstreamRoot ?? snapshotPath(hubs[0], family.upstream.repo, upstream.revision) });
    if (!upstreamRoot) {
      return { ...row, status: "weights_missing", reason: `no ${family.upstream.repo}@${upstream.revision.slice(0, 8)} snapshot on this host` };
    }
    // sc-22738. A PRESENT snapshot is not a complete one. The dense MiniMax-H3 tree staged on the
    // capture host carried `vae/`, `audio_vae/`, `tokenizer/` and `FL2VA/` but no `text_encoder/`,
    // and because this branch only asked whether the snapshot directory existed, `--list` reported
    // `minimax_h3:bf16:mlx` runnable and the booked capture died on the missing directory instead.
    // The declared files are the ones the pinned loader opens under this root, so an incomplete
    // mirror is `weights_missing` — named file by file — exactly like an absent one.
    const missingUpstream = [];
    for (const file of requiredFilesFor(family.upstream.requiredFiles, parts.tier)) {
      try {
        if (!(await stat(path.join(upstreamRoot, file))).isFile()) missingUpstream.push(file);
      } catch { missingUpstream.push(file); }
    }
    if (missingUpstream.length > 0) {
      return {
        ...row, status: "weights_missing",
        reason: `${family.upstream.repo} snapshot ${upstreamRoot} is missing ${missingUpstream.join(", ")}, which the pinned loader opens under the snapshot root`,
      };
    }
    row.env[`SCENEWORKS_${family.upstream.env}_REPOSITORY`] = family.upstream.repo;
    row.env[`SCENEWORKS_${family.upstream.env}_REVISION`] = upstream.revision;
    row.env[`SCENEWORKS_${family.upstream.env}_ROOT`] = upstreamRoot;
  }
  // sc-22729: the caller-staged SDXL components. Their revisions come from the model's own
  // corequisite downloads, which are exactly the revisions `candle-gen-sdxl` validates against.
  for (const component of Array.isArray(family.components) ? family.components : []) {
    const download = tierDownload(models, parts.modelId, component.repo, parts.tier);
    const root = await firstExistingDirectory(hubs.map((hub) => snapshotPath(hub, component.repo, download.revision)));
    row.roots.push({ label: `component ${component.env}`, path: root ?? snapshotPath(hubs[0], component.repo, download.revision) });
    if (!root) {
      return { ...row, status: "weights_missing", reason: `no ${component.repo}@${download.revision.slice(0, 8)} snapshot on this host` };
    }
    row.env[component.env] = root;
  }
  // The shared component snapshot a split-layout family stages its text encoder and VAE from
  // (sc-22733, Mage-Flow). Unlike `upstream`, this root is a co-requisite the MANIFEST ships, and
  // unlike the tier root above it is the SNAPSHOT: the tier is the first path element inside it,
  // and the adapters join `<tier>/text_encoder` and `<tier>/vae` themselves. Absent is
  // `weights_missing` — a Mage load whose components are not staged cannot open at all, because the
  // variant rehost carries no `text_encoder/` or `vae/` sibling for the loader to fall back to.
  if (family.components && !Array.isArray(family.components)) {
    // The components are declared PER TIER (one co-requisite row per component per tier), so the
    // row for THIS tier is what says the tier's text encoder and VAE are shipped at all. No row is
    // `weights_missing` — the cell is planned and the arm exists, the host merely cannot bind the
    // artifact — never a thrown error that aborts the whole `--list` (sc-22733 review).
    const download = (models.find((entry) => entry.id === parts.modelId)?.downloads ?? []).find(
      (row) => row.repo === family.components.repo && row.coRequisite && row.variant === parts.tier,
    );
    if (!download) {
      row.roots.push({ label: "components snapshot", path: `$SCENEWORKS_${family.components.env}_ROOT` });
      return {
        ...row, status: "weights_missing",
        reason: `manifest ${parts.modelId} declares no ${family.components.repo} components row for tier ${parts.tier}`,
      };
    }
    const root = await firstExistingDirectory(
      hubs.map((hub) => snapshotPath(hub, family.components.repo, download.revision)),
    );
    row.roots.push({
      label: "components snapshot",
      path: root ?? snapshotPath(hubs[0], family.components.repo, download.revision),
    });
    if (!root) {
      return {
        ...row, status: "weights_missing",
        reason: `no ${family.components.repo}@${download.revision.slice(0, 8)} snapshot on this host`,
      };
    }
    // The tier's own two component directories, not just the snapshot: a partially-fetched
    // components mirror is exactly as unloadable as an absent one, and reporting the cell
    // `runnable` would send an operator to book a capture that cannot open its text encoder.
    const missing = [];
    for (const component of MAGE_COMPONENT_IDS) {
      if (!(await firstExistingDirectory([path.join(root, parts.tier, component)]))) {
        missing.push(`${parts.tier}/${component}`);
      }
    }
    if (missing.length > 0) {
      return {
        ...row, status: "weights_missing",
        reason: `${family.components.repo} snapshot ${root} is missing ${missing.join(", ")}`,
      };
    }
    row.env[`SCENEWORKS_${family.components.env}_REPOSITORY`] = family.components.repo;
    row.env[`SCENEWORKS_${family.components.env}_REVISION`] = download.revision;
    row.env[`SCENEWORKS_${family.components.env}_ROOT`] = root;
  }
  // An identity stack the worker fetches on first use rather than declaring as a manifest download
  // has nothing for the harness to resolve, so the operator stages it and names it here. An unset
  // or absent path is `weights_missing` — the host simply lacks the artifact — not a gap.
  for (const staging of family.stagedEnv ?? []) {
    const name = staging.env;
    const staged = process.env[name];
    const root = staged ? await firstExistingDirectory([staged]) : null;
    row.roots.push({ label: `staged ${name}`, path: staged ?? `(${name} unset)` });
    if (!root) {
      return { ...row, status: "weights_missing", reason: `${name} is unset or names no directory on this host` };
    }
    const missingStaged = [];
    for (const file of staging.files ?? []) {
      if (!(await isFile(path.join(root, file)))) missingStaged.push(file);
    }
    if (missingStaged.length > 0) {
      return {
        ...row, status: "weights_missing",
        reason: `${name} directory ${root} is missing ${missingStaged.join(", ")}, which the adapter arm opens under it`,
      };
    }
    row.env[name] = root;
  }
  // A member-specific artifact the manifest does not ship (the Qwen edit Lightning distill LoRA):
  // its repository and revision are pinned in the family row, so the root is the snapshot itself.
  const side = family.sideArtifact?.[parts.modelId];
  if (side) {
    const sideRoot = await firstExistingDirectory(hubs.map((hub) => snapshotPath(hub, side.repo, side.revision)));
    row.roots.push({ label: "side artifact", path: sideRoot ?? snapshotPath(hubs[0], side.repo, side.revision) });
    if (!sideRoot) {
      return { ...row, status: "weights_missing", reason: `no ${side.repo}@${side.revision.slice(0, 8)} snapshot on this host` };
    }
    // sc-22738. A PRESENT snapshot is not the pinned FILE. The Lightning snapshot staged on this
    // capture host holds the 8-step distill and NOT the pinned 4-step one the arm joins
    // (mlx.rs / candle.rs `qwen_edit_lightning_adapter`, `protocol::QWEN_EDIT_LIGHTNING_FILE`), so
    // asking only whether the snapshot directory existed reported all three
    // `qwen_image_edit_2511_lightning` cells runnable and every booked capture died on
    // `the Lightning distill LoRA is not at …`. The declared file IS the one the arm opens, so an
    // incomplete mirror is `weights_missing` — named — exactly like an absent one.
    if (side.file && !(await isFile(path.join(sideRoot, side.file)))) {
      return {
        ...row, status: "weights_missing",
        reason: `${side.repo} snapshot ${sideRoot} is missing ${side.file}, which the adapter arm attaches as this member's distill LoRA`,
      };
    }
    row.env[`SCENEWORKS_${side.env}_REPOSITORY`] = side.repo;
    row.env[`SCENEWORKS_${side.env}_REVISION`] = side.revision;
    row.env[`SCENEWORKS_${side.env}_ROOT`] = sideRoot;
  }
  row.sourceCapture = backend === "mlx" && family.sourceCapture === true;
  row.physical = backend === "mlx" && family.physical === true;
  // sc-22738: see the LTX-2.5 branch — the tier root this row resolved, named so a hard stop can
  // state WHICH weights reached the footprint it bounds. `measureAnchor` fills the inventory digest
  // in from the same hash it exports to the adapter.
  row.artifact = {
    repository: artifact.repo,
    resolvedRevision: resolved.revision,
    variant: parts.tier,
  };
  return row;
}

/** Append one evidence corpus to the Rust loader's compiled-in list, idempotently. */
export function appendPackagedSource(source, relativePath) {
  const start = source.indexOf("PACKAGED_MEMORY_ANCHOR_SOURCES: &[(&str, &str)] = &[");
  if (start === -1) fail(`${PACKAGED_SOURCES_PATH} no longer declares PACKAGED_MEMORY_ANCHOR_SOURCES`);
  const end = source.indexOf("\n];", start);
  if (end === -1) fail("PACKAGED_MEMORY_ANCHOR_SOURCES is not terminated by `];`");
  // Idempotence is decided INSIDE the list, not over the whole file (sc-22738). Scanning the file
  // made any other mention of the path — a doc comment, a test fixture that cites the corpus it
  // exercises — read as "already packaged", and the append then silently did nothing: the corpus
  // stayed out of the Rust loader while the run reported success, and the store's own handshake
  // rejected every row derived from it. Caught by this story's own adapter fixture, which names
  // the very corpus it was about to package.
  const block = source.slice(start, end);
  if (block.includes(`"${relativePath}"`)) return source;
  const entry = [
    "    (",
    `        "${relativePath}",`,
    `        include_str!("../../../${relativePath}"),`,
    "    ),",
  ].join("\n");
  // IN SORTED POSITION, not at the end (sc-22738). `memory_anchor.rs` asserts the compiled-in list
  // stays sorted, and appending blindly only happened to hold while every new corpus landed under
  // `docs/generated/`. This campaign's corpora live under `docs/calibration/sc-22738/`, which sorts
  // BEFORE every `docs/generated/` entry — so an append put the list out of order and the next
  // `cargo test -p sceneworks-core` reds on a commit the runner had already made.
  const successor = [...block.matchAll(/^ {8}"([^"]+)",$/gm)].find(
    (match) => match[1] > relativePath,
  );
  if (!successor) return `${source.slice(0, end)}\n${entry}${source.slice(end)}`;
  // Back up from the successor's path line to the `    (` that opens its tuple.
  const opener = block.lastIndexOf("    (\n", successor.index);
  if (opener === -1) fail("PACKAGED_MEMORY_ANCHOR_SOURCES entry is not a `    (` tuple");
  const at = start + opener;
  return `${source.slice(0, at)}${entry}\n${source.slice(at)}`;
}

/**
 * The edition `rustfmt` is invoked with, bound to `rustfmt.toml` by a test.
 *
 * `cargo fmt` passes the workspace edition explicitly; a bare `rustfmt` would fall back to the
 * config file's, so the two are stated in one place and asserted equal rather than left to drift.
 */
export const RUSTFMT_EDITION = "2021";

/**
 * Format one Rust file in place with `rustfmt`, or fail the run.
 *
 * A campaign host always has the toolchain the same run builds with, so an absent or failing
 * `rustfmt` is a broken host, not a condition to route around: formatting silently skipped would
 * put the tree back in the state this exists to prevent.
 */
export async function formatRustSource(file) {
  try {
    // `cwd` is the file's directory so `rustfmt.toml` is discovered from its ancestors, exactly as
    // `cargo fmt` resolves it for the crate.
    await run("rustfmt", ["--edition", RUSTFMT_EDITION, file], { cwd: path.dirname(file) });
  } catch (error) {
    fail(`rustfmt could not format ${file}: ${error.message}`);
  }
}

/**
 * Package `relativePath` into [`PACKAGED_SOURCES_PATH`] and leave the file rustfmt-stable.
 *
 * The append writes the tuple on one line (sc-22738), which fits `max_width = 100` only while the
 * corpus name is short: `docs/calibration/sc-22738/flux2-dev-bf16-mlx-exceeded-evidence.json` makes
 * the `include_str!` line 101 columns and rustfmt wants it wrapped, so the runner's own commit red
 * the `parity-rust` lane's `cargo fmt --check` — on a tree it had already pushed. Earlier commits
 * passed only because their names happened to be a few characters shorter.
 *
 * The width rule is therefore not re-implemented here: rustfmt itself is run over the written file,
 * so whatever the checked-in `rustfmt.toml` says — now or after a config change — is what lands.
 *
 * Returns whether the file changed; a corpus already packaged rewrites (and reformats) nothing.
 */
export async function writePackagedSource(root, relativePath) {
  const file = path.join(root, PACKAGED_SOURCES_PATH);
  const source = await readFile(file, "utf8");
  const appended = appendPackagedSource(source, relativePath);
  if (appended === source) return false;
  await writeFile(file, appended);
  await formatRustSource(file);
  return true;
}

/**
 * The two Rust builder stages' evidence-copy blocks, and the line that opens each of them.
 *
 * Every `include_str!` `memory_anchor.rs` compiles in must ALSO be copied into both Docker builder
 * contexts, or `docker build` breaks while `cargo build` on a checkout stays green — the two see
 * different trees. `scripts/platform-review-contracts.test.mjs` ("Rust Docker builders copy every
 * production generated embed from sceneworks-core") asserts exactly that, counting the `COPY <path>
 * ./<dir>/` line twice. `appendPackagedSource` added the embed and left the Dockerfile alone, so
 * every anchor commit — a completed capture's as much as a hard stop's — landed a red tree that had
 * to be repaired by hand afterwards (PR #2759 and the bernini q4 seed both did).
 */
export const DOCKERFILE_PATH = "docker/rust.Dockerfile";
export const DOCKERFILE_EMBED_ANCHOR = "COPY docs/generated/memory-calibration-evidence.json ./docs/generated/";

/**
 * `dockerfile` with `relativePath` copied into BOTH builder stages, idempotently.
 *
 * Placement follows the lines already there rather than inventing an order: inside the same
 * directory group in sorted position (so `docs/calibration/sc-22738/` stays readable as one block),
 * else after the last line of the same `docs/<kind>/` family, else at the end of the block. The
 * block is the contiguous run of `COPY` lines around each occurrence of [`DOCKERFILE_EMBED_ANCHOR`],
 * which is the one embed both stages have carried since the file was written.
 *
 * Fails when the two blocks cannot be found: a Dockerfile this cannot read must red the run rather
 * than silently commit an embed the image will not carry.
 */
export function insertEvidenceCopy(dockerfile, relativePath) {
  const directory = relativePath.slice(0, relativePath.lastIndexOf("/") + 1);
  const entry = `COPY ${relativePath} ./${directory}`;
  const lines = dockerfile.split("\n");
  const anchors = lines.flatMap((line, index) => (line === DOCKERFILE_EMBED_ANCHOR ? [index] : []));
  if (anchors.length !== 2) {
    fail(
      `${DOCKERFILE_PATH} carries ${anchors.length} "${DOCKERFILE_EMBED_ANCHOR}" lines, not the two `
        + "builder stages this run must copy the new evidence into",
    );
  }
  const family = `COPY ${directory.split("/").slice(0, 2).join("/")}/`;
  // Last block first: an insertion shifts every LATER index, never an earlier one.
  for (const anchor of [...anchors].reverse()) {
    let start = anchor;
    while (start > 0 && lines[start - 1].startsWith("COPY ")) start -= 1;
    let end = anchor;
    while (end + 1 < lines.length && lines[end + 1].startsWith("COPY ")) end += 1;
    if (lines.slice(start, end + 1).includes(entry)) continue;
    let at = null;
    for (let index = start; index <= end; index += 1) {
      if (!lines[index].startsWith(`COPY ${directory}`)) continue;
      if (lines[index] > entry) { at = index; break; }
      at = index + 1;
    }
    if (at === null) {
      for (let index = start; index <= end; index += 1) if (lines[index].startsWith(family)) at = index + 1;
    }
    lines.splice(at ?? end + 1, 0, entry);
  }
  return lines.join("\n");
}

/** Anchors already ingested under the campaign directory, keyed by anchor key. */
export async function capturedInCampaign(root, campaignDir) {
  const captured = new Map();
  let entries;
  try {
    entries = await readdir(path.join(root, campaignDir));
  } catch {
    return captured;
  }
  for (const name of entries) {
    if (!name.endsWith("-evidence.json")) continue;
    let bundle;
    try {
      bundle = JSON.parse(await readFile(path.join(root, campaignDir, name), "utf8"));
    } catch {
      continue;
    }
    for (const record of bundle.records ?? []) {
      const target = record.target ?? {};
      if (target.modelId && target.tier && record.backend) {
        captured.set(`${target.modelId}:${target.tier}:${record.backend}`, `${campaignDir}/${name}`);
      }
    }
  }
  return captured;
}

// ---------------------------------------------------------------------------------------------
// Process plumbing
// ---------------------------------------------------------------------------------------------

function run(command, args, { cwd = ROOT, env = process.env, log = null, detached = false, input = null } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, env, stdio: [input === null ? "ignore" : "pipe", "pipe", "pipe"], detached });
    if (input !== null) child.stdin.end(input);
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; log?.write(chunk); });
    child.stderr.on("data", (chunk) => { stderr += chunk; log?.write(chunk); });
    child.on("error", reject);
    child.on("close", (code, signal) => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(Object.assign(new Error(`${[command, ...args].join(" ")} exited ${code ?? signal}\n${stderr.slice(-4000)}`), { stdout, stderr, code }));
    });
  });
}

async function git(args, cwd = ROOT) {
  return (await run("git", args, { cwd })).stdout.trim();
}

async function gitIsClean(cwd) {
  return (await git(["status", "--porcelain"], cwd)) === "";
}

async function assertOutsideRepo(dir, repo) {
  const relation = path.relative(await realpath(repo), await realpath(dir));
  if (!relation || (!relation.startsWith("..") && !path.isAbsolute(relation))) {
    fail(`${dir} is inside the checkout ${repo}; the harness refuses in-tree output`);
  }
}

class Log {
  constructor(file) { this.file = file; this.chunks = []; }
  write(chunk) { this.chunks.push(String(chunk)); }
  async flush() { await writeFile(this.file, this.chunks.join("")); }
}

function stamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

// ---------------------------------------------------------------------------------------------
// Physical containment (sc-22738)
// ---------------------------------------------------------------------------------------------

export const WATCHDOG = "scripts/memory-calibration-watchdog.py";

/**
 * The SC-18946 incident: the kernel-maintained physical footprint the q4 f305 LTX-2.3 provider
 * reached on a 128 GiB host before the host watchdog panicked. The one place this runner states
 * it; the MLX adapter declares the same figure as `LTX_Q4_F305_CRASH_FOOTPRINT_BYTES`
 * (`crates/sceneworks-memory-adapter/src/bin/mlx.rs`), and a test binds the two.
 */
export const LTX_Q4_F305_CRASH_FOOTPRINT_BYTES = 96_970_084_480;

/**
 * The MLX lane's fixed unified-memory reserve, 2 GiB
 * (`sceneworks_core::memory_anchor::LEGACY_UNIFIED_FALLBACK_RESERVE_GB`, the reserve the worker's
 * `live_request_budget` and the adapter's LTX admission both present as `reserved_headroom_bytes`).
 * The guard uses it twice: as the margin under the incident footprint and as the absolute OS floor
 * on whole-host free memory. Bound to the Rust declaration by a test.
 */
export const UNIFIED_RESERVE_BYTES = 2 * 1024 * 1024 * 1024;

/**
 * The adapter's own host probe — the same `probe` action the harness runs first inside every
 * capture — so the ceilings below come from the figures the record will carry, not from a second
 * reading of the host. `memoryBytes` is required of every adapter; `wiredLimitBytes` is what the
 * MLX adapter resolves from host policy and is required when the anchor is an MLX one — the candle
 * adapter reports none, and on Darwin its capture is guarded from the host figure alone.
 */
export async function probeAdapter(command, { cwd = ROOT, env = process.env, wiredLimitRequired = true } = {}) {
  const { stdout } = await run(command[0], command.slice(1), { cwd, env, input: `${JSON.stringify({ action: "probe" })}\n` });
  const hardware = JSON.parse(stdout).hardware;
  const positive = (field) => Number.isSafeInteger(hardware?.[field]) && hardware[field] > 0;
  if (!positive("memoryBytes")) fail("adapter probe reported no positive hardware.memoryBytes");
  if (hardware.wiredLimitBytes !== undefined || wiredLimitRequired) {
    if (!positive("wiredLimitBytes")) fail("adapter probe reported no positive hardware.wiredLimitBytes");
    if (hardware.wiredLimitBytes > hardware.memoryBytes) fail("adapter probe reported a wired ceiling above host memory");
  }
  return hardware;
}

/**
 * The two ceilings the guard runs with, derived from the probe and the incident:
 *
 * * `maxFootprintBytes` — the guarded group's physical-footprint hard stop — is the SMALLER of the
 *   incident footprint less the unified reserve and host memory less the unified reserve. The
 *   incident term is what keeps the kill line under 96,970,084,480 bytes on any host large enough
 *   to reach it (sc-22738 review).
 * * `minMemoryFreeBytes` — the whole-host free floor — is the unified reserve, an ABSOLUTE OS
 *   reserve independent of the wired limit. The watchdog compares it against `memory_pressure`'s
 *   whole-host free reading, which the guarded group's own footprint reduces: a floor of
 *   `memoryBytes − wiredLimitBytes` would have turned the per-group ceiling into a cap on TOTAL
 *   host use and killed a group at the ceiling less the host baseline (sc-22738 review).
 *
 * `wiredLimitBytes` is NOT a term in either ceiling. It is a Metal-buffer ceiling; the watchdog
 * samples the kernel's `phys_footprint`, which counts every non-Metal page the process owns too,
 * so the two quantities are not comparable. On this 128 GiB host the wired limit (87,044,670,532)
 * as a hard stop killed `flux2_dev:bf16` at 87,140,069,928 bytes and 660 s — an anchor the same
 * host had already rendered to completion unguarded. The probe still reports and records it (it is
 * how the MLX lane's own admission reasons about device memory); it is advisory here, never a kill
 * line (sc-22738, measured 2026-09-06).
 */
export function watchdogCeilings({ memoryBytes }) {
  const bounds = [LTX_Q4_F305_CRASH_FOOTPRINT_BYTES - UNIFIED_RESERVE_BYTES, memoryBytes - UNIFIED_RESERVE_BYTES];
  return { maxFootprintBytes: Math.min(...bounds), minMemoryFreeBytes: UNIFIED_RESERVE_BYTES };
}

/**
 * The footprint hard stop every Darwin capture runs under: `scripts/memory-calibration-watchdog.py`
 * wrapped around the harness, the same guard `scripts/run-ltx-safety-canary.mjs` puts around its
 * contained runs, minus the nonce-authenticated child attestation and provider-phase channel that
 * only the frozen canary profiles speak. Without those the guard still does the one thing the
 * SC-18946 incident needed: it samples the guarded group's kernel `phys_footprint` every quarter
 * second and terminates the whole group the instant it reaches the ceiling, before the host
 * watchdog can panic.
 *
 * Both ceilings are DERIVED (`watchdogCeilings`) from the adapter's probed host memory and the
 * incident, never chosen here; the probe's wired limit is validated but is not a kill line. No wall-time ceiling: a runtime-complete video anchor runs six renders, and time is
 * not the hazard this guard exists for. Darwin-only by construction — the footprint sampler is
 * `/usr/bin/footprint`.
 */
export function watchdogGuard({ hardware, eventFile }) {
  const { memoryBytes, wiredLimitBytes } = hardware;
  if (!Number.isSafeInteger(memoryBytes) || memoryBytes <= 0) fail("watchdog guard needs a positive hardware.memoryBytes");
  if (wiredLimitBytes !== undefined) {
    if (!Number.isSafeInteger(wiredLimitBytes) || wiredLimitBytes <= 0) fail("watchdog guard needs a positive hardware.wiredLimitBytes");
    if (wiredLimitBytes > memoryBytes) fail("watchdog guard: wired ceiling above host memory");
  }
  const { maxFootprintBytes, minMemoryFreeBytes } = watchdogCeilings({ memoryBytes });
  if (maxFootprintBytes <= 0) fail(`watchdog guard: no positive footprint ceiling on a ${memoryBytes}-byte host`);
  return [
    // This runner's own tool, resolved against ITS checkout: the guard is not an artifact of the
    // tree being measured.
    path.join(ROOT, WATCHDOG),
    "--max-footprint-bytes", String(maxFootprintBytes),
    "--host-memory-bytes", String(memoryBytes),
    "--min-memory-free-bytes", String(minMemoryFreeBytes),
    "--sample-interval", "2",
    "--telemetry-timeout", "10",
    "--term-grace", "1",
    "--event-file", eventFile,
  ];
}

/**
 * The guard's `hard_stop` reason from its event log, or `null` when the log carries none (the
 * capture failed for a reason of its own, or never ran under the guard). A missing log is `null`,
 * never a throw: it is read on the failure path, where the ORIGINAL error must still surface.
 */
export async function watchdogHardStop(eventFile) {
  let body;
  try {
    body = await readFile(eventFile, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
  for (const line of body.split("\n")) {
    if (!line.trim()) continue;
    const event = JSON.parse(line);
    if (event.event === "hard_stop") return `watchdog hard stop: ${event.reason}`;
  }
  return null;
}

/**
 * How a process-scoped Metal refusal is named on a row and in a commit subject (sc-22738). The
 * machine-readable spelling lives on the bound itself (`metal_submissions_ignored:observed_…`); this
 * is the human one, and it is a constant so the outcome table in the runbook has something to match.
 */
export const METAL_REFUSAL = "Metal refused this process's submissions at the host's wired limit";

/**
 * Whether a capture of this anchor runs under the footprint guard: EVERY capture on Darwin. A
 * candle capture running on a Mac draws from the same unified pool the sampler measures, so the
 * host-RAM hazard is the same whichever adapter is under the harness (sc-22738 review); the guard
 * exists on no other platform.
 */
export function guardsCapture(key, platform = process.platform) {
  anchorParts(key);
  return platform === "darwin";
}

// ---------------------------------------------------------------------------------------------
// One anchor, end to end
// ---------------------------------------------------------------------------------------------

/**
 * The extractor carries each anchor's currency key forward from the store and FAILS for an anchor
 * id it has never seen — by design, so a new anchor can never borrow the pin's digest. A fresh
 * capture is exactly that case (record ids are content-derived, so the anchor id is new). Seed the
 * new id with a placeholder key so the store can be regenerated, then let `--stamp-anchors`, which
 * runs next and rewrites every key at its record's own revision, replace it before the commit.
 */
export const NEW_ANCHOR_PATTERN = /anchor (\S+) has no recorded loader-closure digest/;

export async function extractSeedingNewAnchors(exec, root, log, limit = 8) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      await exec(process.execPath, ["scripts/extract-memory-anchors.mjs"], { log });
      return;
    } catch (error) {
      const match = NEW_ANCHOR_PATTERN.exec(String(error.stderr ?? error.message ?? ""));
      if (!match || attempt >= limit) throw error;
      const storePath = path.join(root, ANCHOR_STORE_PATH);
      const store = JSON.parse(await readFile(storePath, "utf8"));
      // sc-22738: a measured lower bound is seeded into ITS OWN array. The extractor reads the
      // currency key out of whichever array the row lives in, so seeding a bound among the anchors
      // would leave the extractor still refusing it on the next attempt and burn the retry budget.
      const rows = match[1].startsWith("exceeded:")
        ? (store.exceededBounds ??= [])
        : store.anchors;
      if (rows.some((row) => row.id === match[1])) throw error;
      rows.push({ id: match[1], source: { loaderClosureDigest: SEED_DIGEST } });
      await writeFile(storePath, `${JSON.stringify(store, null, 2)}\n`);
      log?.write(`seeded new anchor ${match[1]} for extraction; --stamp-anchors derives its key next\n`);
    }
  }
}

/** Longest failure reason a summary row carries. The ONLY bound: never a delimiter, see below. */
export const FAILURE_REASON_LIMIT = 300;

/**
 * The LAST line of a child's stderr that names the failure, not the Node banner after it.
 *
 * sc-22738: this used to take the FIRST line matching `/^(Error|…Error|fatal|error)\b/`, falling
 * back to the first non-banner line. Both arms read the wrong end of the message. The harness
 * rejects a failed provider with `${command} exited ${code}: ${stderr.trim()}` — the adapter's WHOLE
 * stderr, many lines, terminal error last — and adapters print informational lines before it. Every
 * Mage anchor in the 09-06 campaign therefore reported `capture_failed` with
 * `GPU-view coherence retries during this render: mlx_gen=0 mlx_llm=0 (sc-22414)`, the sc-22414
 * coherence tally the adapter prints on the way out, while the actual refusal —
 * `a synchronized Mage-Flow lifecycle phase reported a zero active peak` — sat on the next line and
 * reached no summary row. Reading from the END gets the adapter's terminal error under both
 * shapes, because Node's own frames and version banner are the only thing that follows it.
 *
 * The full stderr is unchanged and still in the anchor log; this only picks what the one-line
 * summary quotes. It is bounded by LENGTH alone — never truncated at a `;` or a `)`, because both
 * occur inside real adapter messages (`mlx_gen=0 mlx_llm=0 (sc-22414)`) and cutting there loses the
 * half that names the cell.
 */
export function failureReason(error) {
  const stderr = String(error.stderr ?? "");
  // No child stderr means the runner itself refused (`fail`), and those messages are ONE statement
  // that may wrap — "post-steps changed paths this run does not own:" and then the paths. Keeping
  // only the last line would drop the sentence and quote a bare path, so join instead.
  if (!stderr.trim()) {
    return String(error.message ?? "unknown failure")
      .split("\n").map((line) => line.trim()).filter(Boolean)
      .join(" ").slice(0, FAILURE_REASON_LIMIT) || "unknown failure";
  }
  const lines = stderr
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !/^(at |Node\.js v)/.test(line));
  return (lines.at(-1) ?? "unknown failure").slice(0, FAILURE_REASON_LIMIT);
}

export async function measureAnchor(row, context) {
  const { args, inferencePin, campaignDir, campaignPrefix, workDir, state, root = ROOT } = context;
  const slug = anchorSlug(row.key);
  const log = new Log(path.join(workDir, "logs", `${slug}.log`));
  const captureOutput = path.join(workDir, "captures", `${slug}.json`);
  const rawLogDir = path.join(workDir, "raw", slug);
  const evidenceRelative = `${campaignDir}/${slug}-evidence.json`;
  // sc-22738: a hard stop's bundle is a SEPARATE file from a completed capture's. The two state
  // different things about the same cell and a later completed capture must be able to land beside
  // the bound rather than overwrite it.
  const exceededOutput = path.join(workDir, "captures", `${slug}-exceeded.json`);
  const exceededEvidenceRelative = `${campaignDir}/${slug}-exceeded-evidence.json`;
  const started = Date.now();
  // sc-22738: the previous ATTEMPT's Metal refusal, read once and cleared here, so this anchor's own
  // outcome is the only thing that can leave the flag set for the anchor after it. Read at the top
  // rather than cleared on every exit path: there are eight of those, and a missed one would silently
  // turn "two in a row" into "two in this run".
  const previousMetalRefusal = state.metalRefusedLast ?? null;
  state.metalRefusedLast = null;
  // sc-22738: the Dockerfile moves with the packaged-source list, on BOTH commit paths — an embed
  // the image does not copy is a red `platform-review-contracts` suite on a commit already made.
  const touched = [ANCHOR_STORE_PATH, MATRIX_PATH, MATRIX_MD_PATH, PACKAGED_SOURCES_PATH, DOCKERFILE_PATH];
  const exec = (command, commandArgs, options = {}) => run(command, commandArgs, { cwd: root, ...options });
  const gitAt = (gitArgs) => git(gitArgs, root);
  const created = [];
  const finish = async (status, reason = null) => {
    await log.flush();
    return { key: row.key, status, reason, seconds: Math.round((Date.now() - started) / 1000), log: log.file };
  };

  const env = { ...process.env, ...row.env };
  if (row.tierRoot) {
    const inventory = await hashArtifactInventory(row.tierRoot);
    env.SCENEWORKS_MEMORY_MODEL_BYTES = String(inventory.bytes);
    env.SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256 = inventory.sha256;
    // The same digest a hard stop's artifact binding cites, so a bound names the exact weight files
    // the guarded render had open rather than a revision that could resolve to several trees.
    if (row.artifact) row.artifact = { ...row.artifact, inventorySha256: inventory.sha256 };
  }
  if (row.sourceCapture) {
    env.SCENEWORKS_MEMORY_CAPTURE_DIR = rawLogDir;
    env.SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX = campaignPrefix;
  }

  // 3..7. ingest + derive + commit — the SAME path for a completed capture and for a footprint
  //        hard stop's measured lower bound (sc-22738). Both put a bundle under the campaign
  //        directory, compile it into the Rust loader's source list, re-derive the store, stamp
  //        every currency key and regenerate the matrix; only the bundle's contents and the commit
  //        subject differ. Any failure rolls the tree back to HEAD so the next anchor still starts
  //        clean; the raw capture stays in the work dir for a by-hand ingest.
  const ingestAndCommit = async ({ captureOutput: input, evidenceRelative: target, sourceRoot, status, subject, message }) => {
    try {
      await mkdir(path.join(root, campaignDir), { recursive: true });
      created.push(target);
      await exec(process.execPath, [
        HARNESS, "ingest", "--input", input, ...sourceRoot, "--output", target,
      ], { log });
      if (row.sourceCapture && status === "committed") {
        const receipts = path.join(rawLogDir, campaignDir);
        try {
          for (const name of await readdir(receipts)) {
            created.push(`${campaignDir}/${name}`);
            await cp(path.join(receipts, name), path.join(root, campaignDir, name), { recursive: true, force: false, errorOnExist: true });
          }
        } catch (error) {
          fail(`copy physical receipts from ${receipts}: ${error.message}`);
        }
      }
      await writePackagedSource(root, target);
      // The same corpus, into both Docker builder contexts. Idempotent, so a re-ingest of a bundle
      // already packaged rewrites nothing.
      const dockerfile = await readFile(path.join(root, DOCKERFILE_PATH), "utf8");
      await writeFile(path.join(root, DOCKERFILE_PATH), insertEvidenceCopy(dockerfile, target));
      await extractSeedingNewAnchors(exec, root, log);
      await exec(process.execPath, [
        "scripts/anchor-loader-closure.mjs", "--repo", path.resolve(args.inferenceRepo), "--stamp-anchors",
      ], { log });
      const stamped = JSON.parse(await readFile(path.join(root, ANCHOR_STORE_PATH), "utf8"));
      const seeded = [...stamped.anchors, ...(stamped.exceededBounds ?? [])]
        .some((entry) => entry.source?.loaderClosureDigest === SEED_DIGEST);
      if (seeded) {
        fail("a seeded placeholder currency key survived --stamp-anchors; refusing to commit it");
      }
      await exec(process.execPath, ["scripts/generate-memory-matrix.mjs"], { log });

      // `-f`: the harness's `<session>.log` receipt matches the blanket `*.log` ignore rule.
      await gitAt(["add", "-f", "--", ...touched, ...created]);
      const stray = await gitAt(["status", "--porcelain"]);
      const unstaged = stray.split("\n").filter((line) => line && line[1] !== " ");
      if (unstaged.length > 0) fail(`post-steps changed paths this run does not own:\n${unstaged.join("\n")}`);
      await gitAt(["commit", "--quiet", "-m", message]);
      state.commits.push(await gitAt(["rev-parse", "--short", "HEAD"]));
      return finish(status);
    } catch (error) {
      log.write(`\nROLLBACK (${subject}): ${error.message}\n`);
      try {
        // Unstage first: `checkout --` restores from the INDEX, which already holds the staged edits.
        await exec("git", ["reset", "--quiet", "--", ...touched, ...created], { log }).catch(() => {});
        await exec("git", ["checkout", "--quiet", "--", ...touched], { log });
        for (const relative of created) await rm(path.join(root, relative), { recursive: true, force: true });
        if (!(await gitIsClean(root))) fail("tree is still dirty after rollback; stop here");
      } catch (rollbackError) {
        state.halt = `rollback failed for ${row.key}: ${rollbackError.message}`;
      }
      return finish("ingest_failed", failureReason(error));
    }
  };

  // 1. capture — detached from the terminal's process group so a Ctrl-C here does not reach the
  //    adapter mid-command-buffer; the loop stops after the anchor in flight instead.
  const captureArgs = [
    HARNESS, "capture", "--plan", PLAN_PATH, "--anchor", row.key,
    "--provider-command", JSON.stringify(providerCommand(args.adapter)),
    "--sceneworks-repo", root, "--inference-repo", path.resolve(args.inferenceRepo),
    "--output", captureOutput,
  ];
  if (row.sourceCapture) captureArgs.push("--raw-log-dir", rawLogDir, "--source-path-prefix", campaignPrefix);
  if (row.ltx25SnapshotRoot) captureArgs.push("--ltx25-snapshot-root", row.ltx25SnapshotRoot);
  const watchdogEvents = path.join(workDir, "logs", `${slug}-watchdog.jsonl`);
  const providerStderrFile = path.join(workDir, "logs", `${slug}-provider-stderr.txt`);
  // The host's Metal wired ceiling as THIS anchor's probe read it, kept for the refusal arm below:
  // the bound is keyed on the limit the refused render actually ran under, not on a later reading.
  let probedHardware = null;
  try {
    if (guardsCapture(row.key)) {
      // sc-22738: every Darwin capture runs inside the footprint hard stop, with ceilings derived
      // from the adapter's own probe for this host and the incident — see `watchdogGuard`.
      const hardware = await probeAdapter(providerCommand(args.adapter), {
        cwd: root, env, wiredLimitRequired: anchorParts(row.key).backend === "mlx",
      });
      probedHardware = hardware;
      const guard = watchdogGuard({ hardware, eventFile: watchdogEvents });
      // The guard APPENDS to its event log; this capture's verdict must not read an earlier one's.
      await rm(watchdogEvents, { force: true });
      log.write(`$ /usr/bin/python3 ${guard.join(" ")} -- node ${captureArgs.join(" ")}\n`);
      await exec("/usr/bin/python3", [...guard, "--", process.execPath, ...captureArgs], { env, log, detached: true });
    } else {
      log.write(`$ node ${captureArgs.join(" ")}\n`);
      await exec(process.execPath, captureArgs, { env, log, detached: true });
    }
  } catch (error) {
    // A hard stop is recorded in the guard's event log, not on stderr: name it on the row.
    const hardStop = await watchdogHardStop(watchdogEvents);
    // sc-22738 (measured 2026-09-06): the SECOND way a run ends having measured a bound. Metal
    // refused this process's submissions once its working set reached the host's wired limit —
    // `flux2_dev:bf16:mlx` at 775 s, sampled peak 86,988,010,336 against an 87,044,670,532-byte
    // limit — while the guard, whose kill line is far higher and whose quantity counts non-Metal
    // pages too, never fired. Nothing in the event log names it, so the adapter's own stderr is the
    // witness, and until now it read as an ordinary `capture_failed` while production went on
    // admitting the identical request at the ladder's bf16 rung.
    const refused = !hardStop && guardsCapture(row.key) && metalSubmissionsIgnored(error.stderr ?? error.message);
    // Whether the PREVIOUS anchor of this run refused the same way; cleared for every anchor at the
    // top of its own attempt, so the count is over consecutive attempts rather than over the run.
    const refusedBefore = previousMetalRefusal;
    if (refused) state.metalRefusedLast = row.key;
    if (!hardStop && !refused) return finish("capture_failed", failureReason(error));
    // DISCRIMINATE THE SCOPE. One refusal is process-scoped: Metal ignored the submissions of the
    // process that had exhausted the wired limit, and the next anchor committed normally four
    // minutes later. The SAME string is also what a wedged HOST says — the GPU stays in the
    // error state and refuses every process until the machine is rebooted (`SubmissionsIgnored`
    // has two scopes) — and on a wedged host every remaining anchor would "measure" a bound at
    // whatever footprint it happened to reach, filling the store with inequalities about the
    // driver rather than about the models. Two in a row is the discriminator: it stops the walk
    // and names the reboot instead of recording a second bound.
    if (refused && refusedBefore) {
      state.halt =
        `${refusedBefore} and then ${row.key} both failed with the Metal submissions-ignored `
        + "refusal: two consecutive refusals are a WEDGED HOST, not two process-scoped bounds. The "
        + "GPU stays in its error state until the machine is REBOOTED; reboot, then re-run the walk. "
        + "No bound was recorded for either anchor beyond the first.";
      return finish("capture_failed", `${METAL_REFUSAL} on the anchor after ${refusedBefore}; halting the walk`);
    }
    const cause = hardStop ?? METAL_REFUSAL;
    // sc-22738: the run established one fact — this cell's peak is AT LEAST the footprint the
    // guard saw — and until now that fact died with the process. Record it as evidence through
    // the same check → ingest → extract → stamp → matrix → commit path a completed capture takes,
    // so production stops admitting the request the host had to kill.
    if (!row.artifact) {
      return finish("capture_failed", `${cause}; no artifact binding for this row to bound it against`);
    }
    if (!args.commit) return finish("exceeded", cause);
    // The refusal arm's witness is the adapter's stderr, so it is written out for the harness to
    // re-check and hash rather than passed through a shell argument.
    const refusalArgs = [];
    if (refused) {
      await writeFile(providerStderrFile, String(error.stderr ?? error.message ?? ""));
      refusalArgs.push(
        "--provider-stderr", providerStderrFile,
        "--wired-limit-bytes", String(probedHardware?.wiredLimitBytes ?? 0),
      );
    }
    try {
      await exec(process.execPath, [
        HARNESS, "record-exceeded", "--plan", PLAN_PATH, "--anchor", row.key,
        "--provider-command", JSON.stringify(providerCommand(args.adapter)),
        "--sceneworks-repo", root, "--inference-repo", path.resolve(args.inferenceRepo),
        "--watchdog-events", watchdogEvents,
        "--artifact", JSON.stringify(row.artifact),
        ...refusalArgs,
        "--output", exceededOutput,
      ], { env, log });
    } catch (recordError) {
      return finish("capture_failed", `${cause}; recording it failed: ${failureReason(recordError)}`);
    }
    return ingestAndCommit({
      captureOutput: exceededOutput,
      evidenceRelative: exceededEvidenceRelative,
      sourceRoot: [],
      status: "committed_exceeded",
      subject: `exceeded bound for ${row.key}`,
      message:
        `chore(${args.campaign}): record the ${row.key} ${refused ? "Metal refusal" : "footprint hard stop"} as a measured bound\n\n` +
        `Captured by scripts/measure-memory-catalog.mjs at inference ${inferencePin}. ` +
        `${cause}. Evidence: ${exceededEvidenceRelative}; anchor store, currency stamp and ` +
        "matrix regenerated.",
    });
  }

  // 2. check the raw bundle before touching the tree.
  const sourceRoot = row.sourceCapture ? ["--source-root", rawLogDir] : [];
  try {
    await exec(process.execPath, [HARNESS, "check", "--input", captureOutput, ...sourceRoot], { log });
  } catch (error) {
    return finish("check_failed", failureReason(error));
  }
  // A no-commit run ends here: ingesting would dirty the tree and make the harness refuse every
  // later anchor in the same run (`complete evidence cannot come from a dirty repository`).
  if (!args.commit) return finish("captured");

  return ingestAndCommit({
    captureOutput,
    evidenceRelative,
    sourceRoot,
    status: "committed",
    subject: `anchor ${row.key}`,
    message:
      `chore(${args.campaign}): measure ${row.key} memory anchor\n\n` +
      `Captured by scripts/measure-memory-catalog.mjs at inference ${inferencePin}. ` +
      `Evidence: ${evidenceRelative}; anchor store, currency stamp and matrix regenerated.`,
  });
}

// ---------------------------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------------------------

export async function planRun(args, root = ROOT) {
  const plan = await readPlan(root);
  const models = await readManifestModels(root);
  const current = await readMatrixCurrency(root);
  const campaign = args.campaign ?? `catalog-${new Date().toISOString().slice(0, 10)}`;
  const campaignDir = `docs/calibration/${campaign}`;
  const captured = await capturedInCampaign(root, campaignDir);
  // sc-22738: measured lower bounds are campaign-independent — they live in the committed store, not
  // under one campaign directory — so a bound recorded by an earlier campaign still speaks for the
  // cell as long as its currency key is the pinned one.
  const bounds = await readExceededBounds(root);
  const declaredLanes = await readDeclaredLanes(root);
  const declaredProviders = await readDeclaredProviders(root);
  const hubs = hubRoots(args.hfCache);
  // sc-22729: the engine's own SDXL route table, read from the pinned inference checkout. `null`
  // when there is none to read — see `SDXL_ROUTES_UNCHECKED`.
  const sdxlRoutes = await readSdxlCandleRoutes(args.inferenceRepo ?? process.env.INFERENCE_REPO);
  // sc-22738: which Wan MLX routes the pinned loader actually seals a receipt for. `null` when there
  // is no checkout to read — see `WAN_MLX_SEAL_UNCHECKED`.
  const wanMlxSealed = await readWanMlxSealedProviders(args.inferenceRepo ?? process.env.INFERENCE_REPO);
  const keys = Object.keys(plan.anchors).sort();
  if (args.anchors) {
    for (const key of args.anchors) if (!plan.anchors[key]) fail(`--anchors names ${key}, which the plan does not declare`);
  }
  for (const model of args.models ?? []) {
    if (!keys.some((key) => anchorParts(key).modelId === model)) fail(`--model ${model} matches no plan anchor`);
  }
  const rows = [];
  for (const key of keys) {
    if (args.anchors && !args.anchors.includes(key)) continue;
    if ((args.models ?? []).length > 0 && !args.models.includes(anchorParts(key).modelId)) continue;
    const row = await classifyAnchor(key, plan.anchors[key], { models, backend: args.backend, hubs, current, captured, bounds, declaredLanes, declaredProviders, sdxlRoutes, wanMlxSealed });
    if (row.status === "other_backend" && !args.anchors) continue;
    // An unperformed route-revision comparison is reported on the row it did not happen for, so a
    // run without an inference checkout cannot silently look like a run that proved the engine agrees.
    if (row.routeCheck && ["runnable", "weights_missing"].includes(row.status)) {
      row.reason = row.reason ? `${row.reason}; ${row.routeCheck}` : row.routeCheck;
    }
    if (row.status === "runnable" && args.skipCurrent && row.current) {
      row.status = "current";
      row.reason = "anchor is current at the pinned inference revision (--skip-current)";
    }
    rows.push(row);
  }
  return { plan, rows, campaign, campaignDir, campaignPrefix: campaignDir, hubs };
}

function table(rows, columns) {
  const widths = columns.map((column) => Math.max(column.length, ...rows.map((row) => String(row[column] ?? "").length)));
  const line = (cells) => cells.map((cell, index) => String(cell ?? "").padEnd(widths[index])).join("  ").trimEnd();
  return [line(columns), line(widths.map((width) => "-".repeat(width))), ...rows.map((row) => line(columns.map((column) => row[column])))].join("\n");
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const { rows, campaign, campaignDir, campaignPrefix, hubs } = await planRun(args);

  if (args.list) {
    process.stdout.write(`${table(rows, ["key", "status", "reason"])}\n`);
    return;
  }

  const inferencePin = await compiledInferencePin();
  const preflight = [];
  const inferenceRepo = path.resolve(args.inferenceRepo);
  const inferenceHead = await git(["rev-parse", "HEAD"], inferenceRepo).catch((error) => { preflight.push(`inference repo: ${error.message}`); return null; });
  if (inferenceHead && inferenceHead !== inferencePin) {
    preflight.push(`inference checkout is at ${inferenceHead.slice(0, 12)} but the adapter compiled INFERENCE_PIN ${inferencePin.slice(0, 12)}; git -C ${inferenceRepo} checkout ${inferencePin}`);
  }
  if (!(await gitIsClean(ROOT))) preflight.push("SceneWorks checkout is dirty; commit or stash first (the harness counts untracked paths)");
  if (inferenceHead && !(await gitIsClean(inferenceRepo))) preflight.push("inference checkout is dirty");
  await mkdir(args.workDir, { recursive: true });
  await assertOutsideRepo(args.workDir, ROOT);
  if (args.adapter) {
    const [binary] = providerCommand(args.adapter);
    try { await stat(binary); } catch { if (!args.adapter.trimStart().startsWith("[")) preflight.push(`adapter binary not found at ${args.adapter}`); }
  }
  if (args.backend === "candle" && !process.env.CUDA_VISIBLE_DEVICES) {
    preflight.push("CUDA_VISIBLE_DEVICES is unset; the CUDA capture host pins one visible device (the runbook keeps GPU 1 visible)");
  }
  const branch = await git(["branch", "--show-current"]);
  if (args.commit && (branch === "main" || branch === "")) preflight.push(`refusing to commit measurements onto ${branch || "a detached HEAD"}; check out a story branch`);

  process.stdout.write(`campaign ${campaign} → ${campaignDir}\nhub roots: ${hubs.join(", ")}\n`);
  process.stdout.write(`${table(rows, ["key", "status", "reason"])}\n\n`);
  const runnable = rows.filter((row) => row.status === "runnable");
  if (args.dryRun) {
    for (const row of runnable) {
      process.stdout.write(`${row.key}\n  sourceCapture=${row.sourceCapture} physical=${row.physical} ltx25=${Boolean(row.ltx25SnapshotRoot)}\n`);
      for (const root of row.roots) process.stdout.write(`  ${root.label}: ${root.path}\n`);
      for (const [name, value] of Object.entries(row.env)) process.stdout.write(`  ${name}=${value}\n`);
    }
    if (preflight.length > 0) process.stdout.write(`\npreflight would refuse:\n- ${preflight.join("\n- ")}\n`);
    process.stdout.write(`\ndry run: ${runnable.length} anchor(s) would be captured, nothing executed\n`);
    return;
  }
  if (preflight.length > 0) fail(`preflight:\n- ${preflight.join("\n- ")}`);
  if (runnable.length === 0) {
    process.stdout.write("nothing runnable on this host for this backend\n");
    return;
  }

  for (const sub of ["logs", "captures", "raw"]) await mkdir(path.join(args.workDir, sub), { recursive: true });
  const state = { commits: [], halt: null, stopRequested: false, metalRefusedLast: null };
  const onInterrupt = () => {
    if (state.stopRequested) { process.stderr.write("\nsecond interrupt: exiting now; the adapter in flight is NOT killed\n"); process.exit(130); }
    state.stopRequested = true;
    process.stderr.write("\ninterrupt: finishing the anchor in flight, then stopping (press again to exit immediately)\n");
  };
  process.on("SIGINT", onInterrupt);
  process.on("SIGTERM", onInterrupt);

  const results = [];
  const summaryPath = path.join(args.workDir, `summary-${stamp()}.json`);
  const writeSummary = () => writeFile(summaryPath, JSON.stringify({ campaign, backend: args.backend, inferencePin, results, commits: state.commits, rows }, null, 2));
  for (const row of runnable) {
    if (state.halt) break;
    if (state.stopRequested) { results.push({ key: row.key, status: "not_started", reason: "interrupted" }); continue; }
    process.stdout.write(`\n=== ${row.key} (${results.length + 1}/${runnable.length}) ${new Date().toISOString()}\n`);
    const result = await measureAnchor(row, { args, inferencePin, campaignDir, campaignPrefix, workDir: args.workDir, state });
    results.push(result);
    process.stdout.write(`--- ${row.key}: ${result.status}${result.reason ? ` (${result.reason})` : ""} in ${result.seconds}s\n`);
    await writeSummary();
  }
  await writeSummary();
  process.stdout.write(`\n${table(results, ["key", "status", "seconds", "reason"])}\n`);
  process.stdout.write(`\ncommits: ${state.commits.length ? state.commits.join(" ") : "none"}\nsummary: ${summaryPath}\n`);
  if (state.halt) fail(state.halt);
  const failed = results.filter((result) => !["committed", "committed_exceeded", "captured", "exceeded"].includes(result.status));
  if (failed.length > 0) process.exitCode = 2;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    process.stderr.write(`${error.message}\n`);
    process.exit(1);
  });
}
