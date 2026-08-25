#!/usr/bin/env node
/**
 * TIER INTEGRITY checker + audit generator (sc-15799).
 *
 * The invariant, stated normatively in `docs/memory-strategy-contract.md` ("Tier integrity"):
 *
 *     No component is resident above the user's selected quant tier unless a DECLARED, MEASURED
 *     exception says otherwise.
 *
 * This script is the enforcement half. It reads the declared exceptions from
 * `config/tier-integrity.jsonc`, reconciles them against what the catalog and the worker actually do,
 * and generates the audit table the story requires (`docs/generated/tier-integrity.{json,md}`).
 *
 * What it enforces:
 *
 *  1. Every declared exception is WELL-FORMED: a real catalog entry, a known component/cause, a
 *     `residentTier` STRICTLY above every tier it claims to apply to (an "exception" that is at or below
 *     the selected tier is not an exception, it is a mistake), and — where the entry publishes per-tier
 *     download variants — only tiers that are actually hosted.
 *  2. Every exception carries EVIDENCE: a positive legacy scalar, or exact resident bytes for every
 *     applicable tier plus its isolation method, triage decision, and re-checked cause.
 *  3. `unmeasured` is forbidden unconditionally. sc-16015 emptied the temporary sc-15799 amnesty; the
 *     exported empty set and committed zero count make that completed migration mechanically visible.
 *  4. `mlx.denseTextEncoderTier` — the one above-tier mechanism the RUNTIME reads — must be declared in
 *     the manifest AND matched by an exception row. Since sc-15799 deleted the hardcoded
 *     `DENSE_TE_TIER_MODELS` worker registry, the flag is the only way to obtain the carve-out, so a
 *     provider-local exception the shared decision cannot see is now structurally impossible.
 *  5. The Krea control branch's declared `branchTierByBaseTier` agrees with the ledger: every base tier
 *     whose branch tier is ABOVE it must have a `controlBranch` exception row, and every base tier whose
 *     branch tier equals it must NOT.
 *  6. ANTI-DRIFT on the resolved Lens carve-out: the ledger and MLX fit gate must carry the same
 *     `sc-16014-resolution: rehosted-q4-q8` marker, and the ledger must contain no stale Lens text-
 *     encoder exception. This preserves the cross-source binding after the q4/q8 rehost eliminated the
 *     exception while the bf16-only MXFP4 expansion remained a fit-gate fact.
 *  7. `backends` is LOAD-BEARING, not a comment (sc-15799 review), and REQUIRED (sc-17395). The same
 *     declared tier can yield different residency on the two lanes — `sensenova_u1_8b::visionTower` is
 *     bf16 under mlx-gen and f32 under candle-gen, because the candle port widens the conv kernels — so
 *     the uniqueness key is **(model, component, LANE)**, not (model, component): a component whose
 *     residency genuinely differs per backend is split into one row per lane, and two rows may never
 *     claim the same lane. Lane names are validated, every declared lane must be one the ROUTER serves
 *     the model on, and the lane is PUBLISHED in the generated markdown so a per-backend claim is
 *     visible to a reader.
 *
 *     sc-15799 left `backends` optional, with omission meaning "both" — and the lane cross-check ran
 *     only when the key was PRESENT, so the default claim was the one nobody validated. Since every
 *     model in this ledger is routed on both lanes, each omitting row was silently asserting that its
 *     residentTier and cost hold identically under two independently-written loaders. They repeatedly
 *     do not. The key is now mandatory: see [`routedLanes`] for why the lane oracle is the routing
 *     tables and not the manifest's optional per-backend tuning block.
 *  8. The ledger validates against its own JSON Schema (`packages/schemas/tier-integrity.schema.json`),
 *     so an unknown or misspelled key is an error rather than a silently ignored field. The schema was
 *     unenforced when it was written, which is how it drifted to draft-07 while every sibling requires
 *     2020-12; `scripts/check-scaffold.mjs` now pins its conventions and this script applies it.
 *  9. Every VAE row declares a structured, receipt-bound scope. `full-autoencoder` counts encoder +
 *     decoder for a route that constructs both; `decoder-only` excludes encoder tensors when no
 *     advertised route on that row's backend constructs them. The receipt key binds that scope to the
 *     selected element count, so prose cannot disguise a mixed convention (sc-17435).
 *
 * Usage:
 *   node scripts/check-tier-integrity.mjs            # regenerate docs/generated/tier-integrity.*
 *   node scripts/check-tier-integrity.mjs --check    # verify + fail on drift (CI)
 *   node scripts/check-tier-integrity.mjs --self-test  # prove the checks actually fail when violated
 */

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import {
  HEADER_MEASUREMENT_TERMS,
  VAE_RESIDENCY_SCOPES,
  VAE_SCOPES,
} from "./tier-integrity-measurement-receipts.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const LEDGER = "config/tier-integrity.jsonc";
const SCHEMA = "packages/schemas/tier-integrity.schema.json";
const MANIFEST = "config/manifests/builtin.models.jsonc";
const MLX_FIT_GATE = "crates/sceneworks-worker/src/mlx_fit_gate.rs";
const ROUTING_CATALOG = "crates/sceneworks-core/src/jobs_store/routing/catalog.rs";
const ROUTING_CANDLE = "crates/sceneworks-core/src/jobs_store/routing/candle.rs";
const ROUTING_MLX = "crates/sceneworks-core/src/jobs_store/routing/mlx.rs";
const OUTPUT_JSON = "docs/generated/tier-integrity.json";
const OUTPUT_MD = "docs/generated/tier-integrity.md";

/**
 * Fidelity ladder for "is this component resident ABOVE the selected tier?". Higher = more faithful.
 * `nvfp4` ranks EQUAL to `q4` and `int8-convrot` equal to `q8`: NVFP4 is a distinct numeric regime
 * (~4.5 effective bits over E2M1 elements, sc-11042) and int8-ConvRot is int8 with online rotation, so
 * claiming an ordering against their neighbours is not something this ladder is entitled to do. Mirrors
 * `gen_core::tier_integrity::fidelity_rank`.
 */
const FIDELITY = { f32: 4, bf16: 3, f16: 3, q8: 2, "int8-convrot": 2, q4: 1, nvfp4: 1 };

const COMPONENTS = new Set([
  "textEncoder",
  "vae",
  // The AUDIO autoencoder of a joint audio+video family (sc-19506). Added rather than folded into
  // `vae` because the uniqueness key is (model, component, LANE): MiniMax-H3 constructs BOTH
  // autoencoders on every render and holds them at DIFFERENT widths — the video VAE at the lane's
  // bf16 `self.dtype`, the audio VAE at a hard-coded f32 — so one summed `vae` row could only carry
  // one `residentTier` and would have to over- or under-declare a half.
  "audioVae",
  "visionTower",
  "controlBranch",
  "identityAdapters",
  "transformerHead",
]);

/** Components whose rows carry a `vaeScope` and a structured residency receipt. */
const SCOPED_AUTOENCODERS = new Set(["vae", "audioVae"]);

const CAUSES = new Set(["packing-exception", "backend-capability", "structural"]);

/**
 * The two generation lanes a row may claim in `backends`, which is REQUIRED — there is no default
 * (sc-17395; omission used to mean BOTH, and skipped the lane cross-check while doing it).
 *
 * This is not decoration. The SAME declared tier can produce DIFFERENT residency per lane, because the
 * two ports make independent dtype decisions: `sensenova_u1_8b`'s vision conv kernels and `fm_head` are
 * bf16 under `mlx-gen-sensenova` but **f32** under `candle-gen-sensenova`, which widens every dense leaf
 * it multiplies against an f32 activation (`candle-gen-sensenova/src/quant.rs::get_f32`, sc-14249). A
 * ledger that can only say one tier per component either over-declares one lane or under-declares the
 * other, so the uniqueness key is (model, component, LANE) — see [`validate`].
 */
const BACKENDS = ["mlx", "candle"];

/**
 * The temporary sc-15799 amnesty remains as an exported EMPTY set so tests and reviewers can see that
 * sc-16015 closed the bounded debt rather than deleting the guard's history.
 */
export const GRANDFATHERED_UNMEASURED = new Set();

/**
 * The committed size of the completed amnesty migration. This must remain zero.
 */
export const GRANDFATHERED_UNMEASURED_COUNT = 0;

/**
 * Shared marker for the resolved Lens q4/q8 tier-integrity exception. It appears in the ledger source
 * and in `mlx_fit_gate.rs`; the checker requires both so removing the rows cannot silently erase the
 * fit-gate's remaining bf16-only MXFP4 accounting, or let that accounting drift back into a q4/q8 row.
 */
const LENS_REHOST_RESOLUTION = "sc-16014-resolution: rehosted-q4-q8";

function parseJsonc(body) {
  return JSON.parse(stripJsoncComments(body));
}

/**
 * The lanes the router actually SERVES a model id on — the lane-existence oracle (sc-17395).
 *
 * ## Why this is not `entry.mlx` / `entry.candle`
 *
 * It used to be, and that was wrong. A manifest per-backend block is optional TUNING config
 * (`vramGbByTier`, `minMemoryGb`, `supportsSequentialOffload`), not a declaration that the lane
 * exists: only 44 of 86 catalog entries carry a `candle` block, and `anima_base` is generated by
 * `candle-gen-anima` — registered at `candle-gen-catalog/src/lib.rs` and `candle_routed: true` in the
 * table below — while carrying no `candle` block at all. Reading block presence as lane existence is a
 * FALSE POSITIVE, and it is how sc-16462 concluded chroma had no candle lane and deleted a real f32
 * T5 residency instead of splitting the row per lane.
 *
 * ## Why the parse is shaped like this
 *
 * Lane facts live in four places, and the previous single regex read one of them:
 *   - `IMAGE_MODEL_CAPS` / `VIDEO_MODEL_CAPS` positional bools. `\s*` spans newlines on purpose:
 *     rustfmt wraps rows past the width limit, and a `"([^"]+)",\s*(true|false)` pattern that assumed
 *     one line silently saw NO route for the four multi-line `sensenova_u1_8b_infographic_*` rows.
 *   - the bespoke `CANDLE_IMAGE_ROUTES` shape routes (`ModelMatch::Any` / `::Family`), which is the
 *     ONLY candle lane `instantid_realvisxl` has — it is `candle_routed: false` yet served by
 *     `instantid_candle_eligible`, so a caps-only read made a live lane look dead.
 *   - direct model-id slice predicates used by video routing. SCAIL-2 deliberately keeps every
 *     Candle column false in `VIDEO_MODEL_CAPS` because it is a distinct engine, then
 *     `video_model_has_candle_video_route` unions `CANDLE_SCAIL2_VIDEO_MODELS`. Ignoring that union
 *     made a real Candle lane look unrouted (sc-18828).
 *
 * `routingMlx` is parsed by the SAME loop but contributes NOTHING today: `mlx.rs` has no `ModelMatch`
 * table and no `is_*_family` predicate, gating instead on `MLX_ROUTED_MODELS`, which is derived from
 * `IMAGE_MODEL_CAPS` and therefore already read above. It is threaded through anyway so the two lanes
 * stay symmetric — an MLX bespoke route added later would otherwise be a SILENT false negative, the
 * failure class this function exists to remove. `check-tier-integrity.test.mjs` pins the zero
 * contribution, so if `mlx.rs` ever gains one this comment fails rather than quietly going stale.
 *
 * Returns `Map<modelId, Set<lane>>`. [`validate`] treats a model absent from this map as an ERROR
 * rather than as "no constraint": a lane oracle that silently resolves nothing is exactly the failure
 * this function replaces, so it fails CLOSED.
 */
export function routedLanes({ routingCatalog, routingCandle, routingMlx }) {
  const lanes = new Map();
  const add = (id, lane) => {
    if (!lanes.has(id)) lanes.set(id, new Set());
    lanes.get(id).add(lane);
  };
  for (const constructor of ["ModelCaps", "VideoModelCaps"]) {
    const pattern = new RegExp(
      `${constructor}::new\\(\\s*"([^"]+)"\\s*,\\s*(true|false)\\s*,\\s*(true|false)\\s*,`,
      "g",
    );
    for (const match of (routingCatalog ?? "").matchAll(pattern)) {
      if (match[2] === "true") add(match[1], "mlx");
      if (match[3] === "true") add(match[1], "candle");
    }
  }
  // Resolve direct `CONST.contains(&model)` additions in the canonical backend-existence predicate.
  // Derived capability-table constants need no special handling because the constructor loop above
  // already reads their source rows. This arm is for genuinely bespoke direct slices such as
  // `CANDLE_SCAIL2_VIDEO_MODELS`, and follows the function body rather than hard-coding that name so
  // a future sibling route is discovered by the same oracle.
  const directSlices = new Map();
  for (const match of (routingCatalog ?? "").matchAll(
    /(?:pub\(crate\)\s+)?const\s+([A-Z][A-Z0-9_]*)\s*:\s*&\[&str\]\s*=\s*&\[([^\]]*)\]/g,
  )) {
    directSlices.set(match[1], [...match[2].matchAll(/"([^"]+)"/g)].map((id) => id[1]));
  }
  const candleVideoRouteBody = (routingCatalog ?? "").match(
    /fn\s+video_model_has_candle_video_route\([^)]*\)[^{]*\{([\s\S]*?)\n\}/,
  )?.[1];
  for (const match of (candleVideoRouteBody ?? "").matchAll(/([A-Z][A-Z0-9_]*)\.contains\(&model\)/g)) {
    for (const id of directSlices.get(match[1]) ?? []) add(id, "candle");
  }
  // `ModelMatch::Family(is_x_family_candle_model)` names a predicate whose `matches!` arm lists the
  // ids; resolve it so an SDXL/SD3 family lane is not invisible just because it is spelled as a fn.
  const familyIds = new Map();
  for (const source of [routingCandle, routingMlx]) {
    for (const match of (source ?? "").matchAll(
      /fn (is_\w+)\(model: &str\) -> bool \{\s*matches!\(\s*model,([^)]*)\)/g,
    )) {
      familyIds.set(match[1], [...match[2].matchAll(/"([^"]+)"/g)].map((id) => id[1]));
    }
  }
  for (const [source, lane] of [
    [routingCandle, "candle"],
    [routingMlx, "mlx"],
  ]) {
    for (const match of (source ?? "").matchAll(/ModelMatch::Any\(&\[([^\]]*)\]\)/g)) {
      for (const id of match[1].matchAll(/"([^"]+)"/g)) add(id[1], lane);
    }
    for (const match of (source ?? "").matchAll(/ModelMatch::Family\((\w+)\)/g)) {
      for (const id of familyIds.get(match[1]) ?? []) add(id, lane);
    }
  }
  return lanes;
}

/** Tiers an entry actually hosts, from its per-tier download variants, or `null` when it has none. */
function hostedTiers(entry) {
  const variants = (entry.downloads ?? [])
    .map((download) => download.variant)
    .filter((variant) => typeof variant === "string" && variant in FIDELITY);
  return variants.length > 0 ? new Set(variants) : null;
}

/**
 * Minimal JSON-Schema check for the subset `packages/schemas/tier-integrity.schema.json` actually uses:
 * `type`, `enum`, `required`, `properties`, `additionalProperties: false`, `items`, `minItems`,
 * `minLength`, `exclusiveMinimum` and local `$ref` into `$defs`. Returns a list of `path: problem`
 * strings.
 *
 * Hand-rolled on purpose: this repo has **no** npm dependencies (`package.json` declares neither
 * `dependencies` nor `devDependencies`), so pulling in ajv to enforce one 100-line schema would be a
 * worse trade than 40 lines that cover it exactly. The schema was written for sc-15799 but nothing ever
 * applied it — which is how it drifted to draft-07 and a `sceneworks.dev` `$id` while every sibling
 * requires 2020-12 / `sceneworks.local`. Its real value is `additionalProperties: false`: a misspelled
 * key (`backend` for `backends`, `residentTiers` for `residentTier`) was silently ignored before, and a
 * silently ignored `backends` is exactly the class of defect the sc-15799 review found.
 */
function schemaErrors(schema, value, root = schema, at = "$") {
  const out = [];
  if (!schema || typeof schema !== "object") return out;
  if (typeof schema.$ref === "string") {
    const target = schema.$ref
      .replace(/^#\//, "")
      .split("/")
      .reduce((node, part) => (node ?? {})[part], root);
    return schemaErrors(target, value, root, at);
  }
  const isArray = Array.isArray(value);
  const actual = isArray ? "array" : value === null ? "null" : typeof value;
  if (schema.type && schema.type !== actual) {
    out.push(`${at}: expected ${schema.type}, got ${actual}`);
    return out;
  }
  if (Array.isArray(schema.enum) && !schema.enum.includes(value)) {
    out.push(`${at}: ${JSON.stringify(value)} is not one of ${JSON.stringify(schema.enum)}`);
  }
  if (schema.type === "string" && typeof schema.minLength === "number" && value.length < schema.minLength) {
    out.push(`${at}: string shorter than minLength ${schema.minLength}`);
  }
  if (schema.type === "number" && typeof schema.exclusiveMinimum === "number" && !(value > schema.exclusiveMinimum)) {
    out.push(`${at}: ${value} must be > ${schema.exclusiveMinimum}`);
  }
  if (isArray) {
    if (typeof schema.minItems === "number" && value.length < schema.minItems) {
      out.push(`${at}: fewer than minItems ${schema.minItems}`);
    }
    if (schema.items) {
      value.forEach((item, index) => out.push(...schemaErrors(schema.items, item, root, `${at}[${index}]`)));
    }
  }
  if (actual === "object") {
    for (const key of schema.required ?? []) {
      if (!(key in value)) out.push(`${at}: missing required property "${key}"`);
    }
    const properties = schema.properties ?? {};
    if (schema.additionalProperties === false) {
      for (const key of Object.keys(value)) {
        if (!(key in properties)) {
          out.push(`${at}: unknown property "${key}" (additionalProperties is false)`);
        }
      }
    }
    for (const [key, subSchema] of Object.entries(properties)) {
      if (key in value) out.push(...schemaErrors(subSchema, value[key], root, `${at}.${key}`));
    }
  }
  return out;
}

/**
 * Validate the ledger against its schema, the catalog and the worker. Returns
 * `{ errors, rows, unverifiedTiers }`; pure so `--self-test` can drive it with mutated inputs.
 */
export function validate({
  ledger,
  ledgerSource,
  manifest,
  mlxFitGate,
  routingCatalog,
  routingCandle,
  routingMlx,
  schema,
}) {
  const errors = [];
  const rows = [];
  const unverifiedTiers = [];
  const byId = new Map((manifest.models ?? []).map((entry) => [entry.id, entry]));
  const routedBackends = routedLanes({ routingCatalog, routingCandle, routingMlx });
  const exceptions = ledger.exceptions ?? [];

  if (exceptions.length === 0) {
    errors.push(
      `${LEDGER} declares no exceptions. An EMPTY ledger is not the same as a conformant catalog — ` +
        `it means the audit was lost. Restore it or delete the check.`,
    );
  }

  // (0) The ledger's own schema, which nothing applied before the sc-15799 review.
  if (!schema) {
    errors.push(
      `${SCHEMA} was not supplied to validate(). The schema is part of the gate, not IDE decoration — ` +
        `skipping it would make a misspelled key a silent no-op again.`,
    );
  } else {
    for (const problem of schemaErrors(schema, ledger)) {
      errors.push(`${LEDGER} violates ${SCHEMA} — ${problem}`);
    }
  }

  // Lanes already claimed per (model, component). The uniqueness key includes the LANE, so a component
  // whose residency differs per backend can be told truthfully as two rows — but an OVERLAP is still a
  // double declaration, which is why this tracks claimed lanes rather than comparing key strings (a
  // naive `model::component::backends` key would let a both-lanes row and an mlx-only row coexist).
  const claimedLanes = new Map();
  const usedMeasurementReceipts = new Set();
  const usedVaeScopeReceipts = new Set();
  for (const row of exceptions) {
    const where = `${row.model ?? "?"}/${row.component ?? "?"}`;
    const entry = byId.get(row.model);
    if (!entry) {
      errors.push(`${where}: no catalog entry with id "${row.model}" in ${MANIFEST}.`);
      continue;
    }
    const key = `${row.model}::${row.component}`;

    // `backends`: which lanes this row speaks for. REQUIRED, validated against the routed lane
    // inventory, and published (sc-15799 review, sc-17395).
    //
    // The key used to be optional, and omitting it both claimed BOTH lanes and skipped every check
    // below — so the one construct that needed validating most was the one that got none. It is now
    // mandatory, because "omit to claim both" is a claim no one was making deliberately: EVERY model in
    // this ledger hosts both lanes, so all 42 rows that omitted the key were silently asserting that
    // their `residentTier` and cost hold identically on mlx-gen AND candle-gen. That is exactly the
    // assertion the two ports keep falsifying — SenseNova's vision tower is bf16 on mlx and f32 on
    // candle; chroma's T5 is bf16 on mlx and f32 on candle — and it is unfalsifiable when it is the
    // DEFAULT. Spelling the lanes forces the author to have looked at both loaders.
    let lanes = BACKENDS;
    if (row.backends === undefined) {
      errors.push(
        `${where}: \`backends\` is REQUIRED. Naming the lanes is how a row states which loader its ` +
          `residentTier and cost were read from; defaulting to both let a single-lane measurement ` +
          `claim a lane nobody checked. Declare ["mlx"], ["candle"], or both — and if the two ports ` +
          `hold this component at different widths, split it into one row per lane.`,
      );
    } else if (!Array.isArray(row.backends) || row.backends.length === 0) {
      errors.push(
        `${where}: \`backends\` must name at least one lane (${BACKENDS.join(" / ")}).`,
      );
    } else {
      const unknown = row.backends.filter((lane) => !BACKENDS.includes(lane));
      if (unknown.length > 0) {
        errors.push(
          `${where}: unknown backend(s) ${unknown.map((lane) => `"${lane}"`).join(", ")} in ` +
            `\`backends\`. Known: ${BACKENDS.join(", ")}.`,
        );
      }
      lanes = row.backends.filter((lane) => BACKENDS.includes(lane));
    }

    // CROSS-CHECK, now UNCONDITIONAL: every lane this row speaks for must be one the router actually
    // serves the model on. Runs for the defaulted lane set too — while it ran only under an explicit
    // `backends`, a row could claim a non-existent lane simply by saying nothing.
    const routed = routedBackends.get(row.model);
    if (!routed || routed.size === 0) {
      // FAIL CLOSED. An unresolvable model is not "unconstrained": the previous oracle resolved
      // nothing for the four multi-line `sensenova_u1_8b_infographic_*` rows and passed them anyway
      // via the manifest-block fallback, so a parse regression was indistinguishable from a clean run.
      errors.push(
        `${where}: no lane could be resolved for "${row.model}" from ${ROUTING_CATALOG}, ` +
          `${ROUTING_CANDLE} or ${ROUTING_MLX}. Either the model is not routed on any backend (so it ` +
          `can host no residency at all), or the routing tables changed shape and this checker's ` +
          `parse no longer sees them. Both are defects; neither may pass silently.`,
      );
    } else {
      for (const lane of lanes) {
        if (!routed.has(lane)) {
          errors.push(
            `${where}: declares \`backends\` including "${lane}", but the router serves "${row.model}" ` +
              `only on ${[...routed].sort().join(", ")}. A per-backend claim about a lane the model ` +
              `does not run on describes nothing — fix the lane.`,
          );
        }
      }
    }

    const alreadyClaimed = claimedLanes.get(key);
    if (alreadyClaimed) {
      const overlap = lanes.filter((lane) => alreadyClaimed.has(lane));
      if (overlap.length > 0) {
        errors.push(
          `${where}: declared twice for ${overlap.map((lane) => `"${lane}"`).join(", ")}. The ` +
            `uniqueness key is (model, component, LANE): a component whose residency genuinely differs ` +
            `per backend is SPLIT into one row per lane, but two rows may never claim the same lane. ` +
            `Merge them, or narrow their \`backends\`.`,
        );
      }
      for (const lane of lanes) alreadyClaimed.add(lane);
    } else {
      claimedLanes.set(key, new Set(lanes));
    }

    if (!COMPONENTS.has(row.component)) {
      errors.push(
        `${where}: unknown component "${row.component}". Known: ${[...COMPONENTS].sort().join(", ")}.`,
      );
    }
    if (!CAUSES.has(row.cause)) {
      errors.push(`${where}: unknown cause "${row.cause}". Known: ${[...CAUSES].sort().join(", ")}.`);
    }
    if (typeof row.reason !== "string" || row.reason.trim().length < 20) {
      errors.push(`${where}: needs a one-sentence \`reason\` in the provider's own terms.`);
    }

    const residentRank = FIDELITY[row.residentTier];
    if (residentRank === undefined) {
      errors.push(`${where}: unknown residentTier "${row.residentTier}".`);
    }
    const applies = Array.isArray(row.appliesToTiers) ? row.appliesToTiers : [];
    if (applies.length === 0) {
      errors.push(`${where}: \`appliesToTiers\` must name at least one selected tier.`);
    }
    const hosted = hostedTiers(entry);
    for (const tier of applies) {
      const tierRank = FIDELITY[tier];
      if (tierRank === undefined) {
        errors.push(`${where}: unknown tier "${tier}" in appliesToTiers.`);
        continue;
      }
      if (residentRank !== undefined && residentRank <= tierRank) {
        errors.push(
          `${where}: residentTier "${row.residentTier}" is NOT above the selected tier "${tier}" ` +
            `(${residentRank} <= ${tierRank}). A component at or below the selected tier needs no ` +
            `exception — remove the tier, or fix residentTier.`,
        );
      }
      if (hosted && !hosted.has(tier)) {
        errors.push(
          `${where}: tier "${tier}" is not hosted by this entry (variants: ` +
            `${[...hosted].sort().join(", ")}). An exception for a tier nobody can select is dead.`,
        );
      }
    }
    if (!hosted) {
      unverifiedTiers.push(row.model);
    }

    const evidence = row.evidence ?? {};
    const receiptKey = `${row.model}::${row.component}::${lanes.join("+")}`;
    if (typeof evidence.source !== "string" || evidence.source.trim().length === 0) {
      errors.push(`${where}: evidence.source must cite where the fact is recorded in-tree.`);
    }
    if (SCOPED_AUTOENCODERS.has(row.component)) {
      const receiptScope = VAE_RESIDENCY_SCOPES[receiptKey];
      if (!receiptScope) {
        errors.push(`${where}: VAE row has no structured residency-scope receipt for ${receiptKey}.`);
      } else {
        usedVaeScopeReceipts.add(receiptKey);
        if (evidence.vaeScope !== receiptScope) {
          errors.push(
            `${where}: evidence.vaeScope=${JSON.stringify(evidence.vaeScope)} does not match the ` +
              `numeric receipt scope ${JSON.stringify(receiptScope)} for ${receiptKey}.`,
          );
        }
        const isolationMarker =
          receiptScope === VAE_SCOPES.FULL
            ? "full autoencoder (encoder + decoder)"
            : "decoder only; encoder tensors are excluded";
        if (
          typeof evidence.isolation !== "string" ||
          !evidence.isolation.includes(isolationMarker)
        ) {
          errors.push(
            `${where}: VAE evidence.isolation must say ${JSON.stringify(isolationMarker)} for the ` +
              `${JSON.stringify(receiptScope)} receipt scope.`,
          );
        }
      }
    }
    if (evidence.state === "measured") {
      const scalarCost = evidence.costGb ?? evidence.costBytes;
      const byTier = evidence.costBytesByTier;
      const hasScalar = typeof scalarCost === "number" && Number.isFinite(scalarCost) && scalarCost > 0;
      const hasByTier = byTier !== null && typeof byTier === "object" && !Array.isArray(byTier);
      if (!hasScalar && !hasByTier) {
        errors.push(
          `${where}: evidence.state "measured" requires a positive costGb/costBytes or an exact ` +
            `costBytesByTier map. An exception ` +
            `whose cost nobody knows is undeclared in the only way that matters.`,
        );
      }
      if (hasByTier) {
        const terms = HEADER_MEASUREMENT_TERMS[receiptKey];
        if (!terms) {
          errors.push(
            `${where}: exact header measurement has no independent receipt for ${receiptKey}. ` +
              `Record selected element counts and runtime widths in ` +
              `scripts/tier-integrity-measurement-receipts.mjs.`,
          );
        } else {
          usedMeasurementReceipts.add(receiptKey);
          const receiptBytes = terms.reduce(
            (sum, [elementCount, bytesPerElement]) => sum + elementCount * bytesPerElement,
            0,
          );
          for (const [tier, cost] of Object.entries(byTier)) {
            if (cost !== receiptBytes) {
              errors.push(
                `${where}: costBytesByTier.${tier}=${cost} does not reproduce the independent ` +
                  `safetensors-header receipt ${receiptKey} (${receiptBytes} bytes).`,
              );
            }
          }
        }
        const expectedTiers = [...new Set(applies)].sort();
        const measuredTiers = Object.keys(byTier).sort();
        if (JSON.stringify(measuredTiers) !== JSON.stringify(expectedTiers)) {
          errors.push(
            `${where}: costBytesByTier keys ${JSON.stringify(measuredTiers)} must exactly match ` +
              `appliesToTiers ${JSON.stringify(expectedTiers)}. Every applicable tier needs an isolated ` +
              `cost, and unrelated tiers must not be smuggled into the row.`,
          );
        }
        for (const [tier, cost] of Object.entries(byTier)) {
          if (typeof cost !== "number" || !Number.isFinite(cost) || cost <= 0) {
            errors.push(`${where}: costBytesByTier.${tier} must be a positive finite byte count.`);
          }
        }
        if (typeof evidence.isolation !== "string" || evidence.isolation.trim().length < 20) {
          errors.push(`${where}: an exact per-tier measurement must record how the component was isolated.`);
        }
        const artifact = evidence.source?.match(
          /isolated artifact: ([A-Za-z0-9._-]+\/[A-Za-z0-9._-]+)@([0-9a-f]{40}):/,
        );
        if (!artifact) {
          errors.push(
            `${where}: an exact per-tier measurement must cite an immutable ` +
              `"isolated artifact: owner/repo@<40-hex-revision>:path" source.`,
          );
        } else {
          const [, repo, revision] = artifact;
          const pinned = (entry.downloads ?? []).some(
            (download) => download.repo === repo && download.revision === revision,
          );
          if (!pinned) {
            errors.push(
              `${where}: measurement source ${repo}@${revision} is not one of this catalog entry's ` +
                `pinned downloads. The measured artifact and the shipping artifact must move together.`,
            );
          }
        }
        if (!["measure", "eliminate"].includes(evidence.triageDecision)) {
          errors.push(`${where}: a promoted row must record triageDecision "measure" or "eliminate".`);
        }
        if (evidence.triageDecision === "eliminate") {
          if (!/^sc-[1-9][0-9]*$/.test(evidence.eliminationStory ?? "")) {
            errors.push(`${where}: triageDecision "eliminate" must name its eliminationStory.`);
          }
        } else if (evidence.eliminationStory !== undefined) {
          errors.push(`${where}: eliminationStory is only valid when triageDecision is "eliminate".`);
        }
        if (evidence.reviewedCause !== row.cause) {
          errors.push(
            `${where}: reviewedCause must equal the re-checked row cause "${row.cause}", got ` +
              `${JSON.stringify(evidence.reviewedCause)}.`,
          );
        }
        if (typeof evidence.triageNote !== "string" || evidence.triageNote.trim().length < 20) {
          errors.push(`${where}: eliminate-vs-measure triage needs a substantive triageNote.`);
        }
      }
    } else if (evidence.state === "unmeasured") {
      errors.push(
        `${where}: UNMEASURED exceptions are forbidden after sc-16015. Above-tier residency must ship ` +
          `with an isolated measured cost, or be eliminated before the catalog entry ships.`,
      );
    } else {
      errors.push(`${where}: evidence.state must be "measured".`);
    }

    rows.push({
      model: row.model,
      component: row.component,
      residentTier: row.residentTier,
      appliesToTiers: applies,
      backends: lanes,
      cause: row.cause,
      reason: row.reason,
      evidenceState: evidence.state,
      costGb: evidence.costGb ?? null,
      costBytes: evidence.costBytes ?? null,
      costBytesByTier: evidence.costBytesByTier ?? null,
      vaeScope: evidence.vaeScope ?? null,
      isolation: evidence.isolation ?? null,
      triageDecision: evidence.triageDecision ?? null,
      reviewedCause: evidence.reviewedCause ?? null,
      triageNote: evidence.triageNote ?? null,
      source: evidence.source ?? null,
      note: evidence.note ?? null,
    });
  }

  for (const receiptKey of Object.keys(HEADER_MEASUREMENT_TERMS)) {
    if (!usedMeasurementReceipts.has(receiptKey)) {
      errors.push(
        `scripts/tier-integrity-measurement-receipts.mjs has stale receipt ${receiptKey} with no ` +
          `matching exact ledger measurement.`,
      );
    }
  }
  for (const receiptKey of Object.keys(VAE_RESIDENCY_SCOPES)) {
    if (!usedVaeScopeReceipts.has(receiptKey)) {
      errors.push(
        `scripts/tier-integrity-measurement-receipts.mjs has stale VAE scope receipt ${receiptKey} ` +
          `with no matching ledger row.`,
      );
    }
  }

  // (3b) The completed migration remains pinned at an empty set and a zero count.
  if (GRANDFATHERED_UNMEASURED.size !== 0 || GRANDFATHERED_UNMEASURED_COUNT !== 0) {
    errors.push(
      `The sc-15799 unmeasured amnesty is closed: GRANDFATHERED_UNMEASURED must stay empty and ` +
        `GRANDFATHERED_UNMEASURED_COUNT must stay 0 (got ${GRANDFATHERED_UNMEASURED.size} and ` +
        `${GRANDFATHERED_UNMEASURED_COUNT}).`,
    );
  }

  // (4) The runtime-visible mechanism must be declared in BOTH places.
  const declaredTextEncoder = new Set(
    exceptions.filter((row) => row.component === "textEncoder").map((row) => row.model),
  );
  for (const entry of manifest.models ?? []) {
    if (entry.mlx?.denseTextEncoderTier === true && !declaredTextEncoder.has(entry.id)) {
      errors.push(
        `${entry.id}: declares \`mlx.denseTextEncoderTier: true\` — which keeps its text encoder ` +
          `resident ABOVE every packed tier — but has no textEncoder exception in ${LEDGER}. That is ` +
          `an undeclared above-tier residency, which sc-15799 makes a defect.`,
      );
    }
  }

  // (4b) Structured component floors are a binding three-way declaration: manifest (shared tier
  // decision + UI), provider descriptor (load path), and this measured ledger. This checker owns the
  // manifest↔ledger half; compiled worker tests own descriptor↔manifest parity on each backend.
  for (const entry of manifest.models ?? []) {
    const floors = entry.precisionFloors ?? [];
    for (const floor of floors) {
      const where = `${entry.id}/${floor.component}`;
      const selectedRank = FIDELITY[floor.selectedTier];
      const residentRank = FIDELITY[floor.residentTier];
      if (selectedRank !== undefined && residentRank !== undefined && residentRank <= selectedRank) {
        errors.push(
          `${where}: precisionFloors residentTier "${floor.residentTier}" is not above selectedTier ` +
            `"${floor.selectedTier}". Remove a no-op declaration or correct the tiers.`,
        );
      }
      const matching = exceptions.filter(
        (row) =>
          row.model === entry.id &&
          row.component === floor.component &&
          row.residentTier === floor.residentTier &&
          (row.appliesToTiers ?? []).includes(floor.selectedTier),
      );
      // Coverage is measured against the lanes the ROUTER serves, not `entry[lane]`. The manifest
      // per-backend block was the oracle here too, and it under-demands: an entry like `anima_base`
      // has no `candle` block, so a floor it declared would never have required a candle-lane row —
      // the same false negative [`routedLanes`] exists to remove. `?? BACKENDS` is gone with it;
      // `backends` is required, so a row without one is already an error and must not silently
      // satisfy coverage for both lanes on its way out.
      const covered = new Set(matching.flatMap((row) => row.backends ?? []));
      for (const lane of [...(routedBackends.get(entry.id) ?? [])].sort()) {
        if (!covered.has(lane)) {
          errors.push(
            `${where}: manifest precisionFloors declares ${floor.selectedTier} → ` +
              `${floor.residentTier}, but ${LEDGER} has no matching exception for backend "${lane}".`,
          );
        }
      }
    }
  }
  for (const row of exceptions) {
    if (!new Set(["textEncoder", "transformerHead"]).has(row.component)) continue;
    if (row.residentTier !== "q8" || !(row.appliesToTiers ?? []).includes("q4")) continue;
    const entry = byId.get(row.model);
    const declared = (entry?.precisionFloors ?? []).some(
      (floor) =>
        floor.component === row.component &&
        floor.selectedTier === "q4" &&
        floor.residentTier === "q8",
    );
    if (!declared) {
      errors.push(
        `${row.model}/${row.component}: ${LEDGER} records a q4 → q8 component floor, but the catalog ` +
          `has no matching precisionFloors declaration for the shared tier decision.`,
      );
    }
  }

  // (5) Krea's declared branch tiers must agree with the ledger, in both directions.
  for (const entry of manifest.models ?? []) {
    const map = entry.candle?.control?.branchTierByBaseTier;
    if (!map) continue;
    const declared = exceptions.some(
      (row) => row.model === entry.id && row.component === "controlBranch",
    );
    const exceptionTiers = new Set(
      exceptions
        .filter((row) => row.model === entry.id && row.component === "controlBranch")
        .flatMap((row) => row.appliesToTiers ?? []),
    );
    for (const [baseTier, branchTier] of Object.entries(map)) {
      const baseRank = FIDELITY[baseTier];
      const branchRank = FIDELITY[branchTier];
      if (baseRank === undefined || branchRank === undefined) {
        errors.push(
          `${entry.id}: branchTierByBaseTier has an unknown tier ("${baseTier}" → "${branchTier}").`,
        );
        continue;
      }
      const above = branchRank > baseRank;
      if (above && !exceptionTiers.has(baseTier)) {
        errors.push(
          `${entry.id}: the control branch is packed to "${branchTier}" on a "${baseTier}" base — ` +
            `above the selected tier — but ${LEDGER} declares no controlBranch exception for ` +
            `"${baseTier}".`,
        );
      }
      if (!above && exceptionTiers.has(baseTier)) {
        errors.push(
          `${entry.id}: ${LEDGER} declares a controlBranch exception for "${baseTier}", but the ` +
            `branch tier there is "${branchTier}", which is not above it. Following the tier needs no ` +
            `exception.`,
        );
      }
    }
    if (!declared && Object.entries(map).some(([b, t]) => FIDELITY[t] > FIDELITY[b])) {
      errors.push(`${entry.id}: control branch sits above a base tier with no declared exception.`);
    }
    // COVERAGE. The loop above iterates DECLARED keys, so a hosted tier variant missing from the map is
    // silently unchecked — the branch could sit above it with nothing noticing. Require the map to name
    // every tier the entry actually hosts.
    const hostedForBranch = hostedTiers(entry);
    if (hostedForBranch) {
      for (const tier of [...hostedForBranch].sort()) {
        if (!(tier in map)) {
          errors.push(
            `${entry.id}: branchTierByBaseTier does not cover the hosted tier variant "${tier}". The ` +
              `checker only inspects declared keys, so an omitted tier is an UNCHECKED branch tier. ` +
              `Declare what the branch does on "${tier}" (int8-convrot follows to q8, like q8).`,
          );
        }
      }
    }
  }

  // (6) Anti-drift: q4/q8 are re-hosted packed affine, so there is no Lens text-encoder exception.
  // The shared source marker keeps that ledger decision bound to the fit gate, where the bf16-only
  // MXFP4 expansion and architecture-specific activation transient still belong.
  const staleLensRows = exceptions.filter(
    (row) =>
      ["lens", "lens_turbo"].includes(row.model) && row.component === "textEncoder",
  );
  if (staleLensRows.length > 0) {
    errors.push(
      `${LEDGER} still declares ${staleLensRows.length} Lens textEncoder exception row(s), but the ` +
        `shipped q4/q8 turnkeys are re-hosted MLX affine packs resident at the selected tier. Remove ` +
        `the stale row(s); bf16 MXFP4 materialization belongs only in ${MLX_FIT_GATE}.`,
    );
  }
  for (const [file, source] of [
    [LEDGER, ledgerSource],
    [MLX_FIT_GATE, mlxFitGate],
  ]) {
    if (typeof source !== "string" || !source.includes(LENS_REHOST_RESOLUTION)) {
      errors.push(
        `${file} no longer contains "${LENS_REHOST_RESOLUTION}". The resolved ledger state and the ` +
          `bf16-only fit-gate accounting must move together (sc-16014).`,
      );
    }
  }

  // Sorted on the full uniqueness key, LANE included — two rows can now share (model, component), so
  // without the third term the generated audit's row order would depend on ledger order.
  rows.sort(
    (a, b) =>
      a.model.localeCompare(b.model) ||
      a.component.localeCompare(b.component) ||
      a.backends.join(",").localeCompare(b.backends.join(",")),
  );
  return { errors, rows, unverifiedTiers: [...new Set(unverifiedTiers)].sort() };
}

function renderJson({ rows, unverifiedTiers }) {
  const measured = rows.filter((row) => row.evidenceState === "measured").length;
  return `${JSON.stringify(
    {
      generatedBy: "scripts/check-tier-integrity.mjs",
      story: "sc-15799",
      measurementStory: "sc-16015",
      invariant:
        "No component is resident above the user's selected quant tier unless a declared, measured exception says otherwise.",
      normativeHome: "docs/memory-strategy-contract.md#tier-integrity",
      totals: {
        exceptions: rows.length,
        measured,
        unmeasured: rows.length - measured,
        entriesWithoutHostedTierVariants: unverifiedTiers.length,
      },
      entriesWithoutHostedTierVariants: unverifiedTiers,
      exceptions: rows,
    },
    null,
    2,
  )}\n`;
}

function renderMarkdown({ rows, unverifiedTiers }) {
  const measured = rows.filter((row) => row.evidenceState === "measured");
  const unmeasured = rows.filter((row) => row.evidenceState !== "measured");
  const lines = [
    "# Tier integrity — above-tier residency audit",
    "",
    "<!-- GENERATED by scripts/check-tier-integrity.mjs from config/tier-integrity.jsonc. Do not edit. -->",
    "",
    "> **No component is resident above the user's selected quant tier unless a declared, measured",
    "> exception says otherwise.**",
    "",
    "Normative statement: [`docs/memory-strategy-contract.md`](../memory-strategy-contract.md) —",
    '"Tier integrity" (sc-15799). Executable rule: `gen_core::tier_integrity`.',
    "",
    `Declared exceptions: **${rows.length}** — ${measured.length} measured, ${unmeasured.length} unmeasured.`,
    "",
    "Above-tier residency is not rare. On a q4 or q8 tier the great majority of image entries keep a",
    "VAE, a text encoder, or both resident at f16, bf16, or f32. Every component that meets the declaration",
    "THRESHOLD stated in `config/tier-integrity.jsonc`'s header is declared below with the in-tree",
    "citation for the fact and the isolated cost. `scripts/check-tier-integrity.mjs` rejects every",
    "unmeasured exception unconditionally: sc-16015 emptied the temporary sc-15799 amnesty.",
    "",
    "**The same declared tier can yield different residency per backend**, so the `backends` column is",
    "part of each row's identity, not a footnote. The two ports make independent dtype decisions: the",
    "SenseNova-U1 vision conv kernels and `fm_head` are bf16 under `mlx-gen-sensenova` but **f32** under",
    "`candle-gen-sensenova`, which widens every dense leaf it multiplies against an f32 activation; the",
    "chroma T5-XXL is bf16 under `mlx-gen` but **f32** under `candle-gen-chroma`, which loads the bf16",
    "checkpoint into f32 containers. Such a component is declared as one row PER LANE. A row listing both",
    "lanes claims the residency holds identically on both — which is now an explicit signature, not a",
    "default: `backends` is REQUIRED (sc-17395).",
    "",
    "| model | component | backends | resident at | above tier on | cause | cost by selected tier (GiB) | evidence |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
  ];
  for (const row of rows) {
    const formatGiB = (bytes) => (bytes / 2 ** 30).toFixed(3).replace(/\.?0+$/, "");
    const cost = row.costBytesByTier
      ? Object.entries(row.costBytesByTier)
          .map(([tier, bytes]) => `${tier}: ${formatGiB(bytes)}`)
          .join("; ")
      : row.costBytes !== null
        ? formatGiB(row.costBytes)
        : row.costGb !== null
          ? row.costGb.toFixed(3).replace(/\.?0+$/, "")
          : "_missing_";
    lines.push(
      `| \`${row.model}\` | ${row.component} | ${row.backends.join(" + ")} | ${row.residentTier} | ${row.appliesToTiers.join(", ")} | ${row.cause} | ${cost} | ${row.source ?? ""} |`,
    );
  }
  lines.push("", "## Why each cause is a different problem", "");
  lines.push(
    "- **packing-exception** — a deliberate quality decision: the component is not packed because",
    "  packing it measurably or structurally hurts. Legitimate, but it must carry the measurement that",
    "  justifies it.",
    "- **backend-capability** — the backend cannot serve the on-disk format, so it upcasts. Packing",
    "  cannot fix this; it needs a compute path or a different published dtype (sc-16014).",
    "- **structural** — there is nothing quantizable (all-conv decoders, projections the loader folds",
    "  away). The cost is real but the exception is not a choice anyone made.",
    "",
  );
  if (unverifiedTiers.length > 0) {
    lines.push(
      "## Entries whose hosted tiers cannot be checked",
      "",
      "These entries publish no per-tier download variants (their tiers are produced on-device), so the",
      "checker cannot verify that `appliesToTiers` names tiers a user can actually select. Recorded",
      "rather than silently skipped:",
      "",
      ...unverifiedTiers.map((model) => `- \`${model}\``),
      "",
    );
  }
  return `${lines.join("\n")}\n`;
}

async function readInputs() {
  const [ledgerBody, manifestBody, mlxFitGate, routingCatalog, routingCandle, routingMlx, schemaBody] =
    await Promise.all([
      readFile(path.join(ROOT, LEDGER), "utf8"),
      readFile(path.join(ROOT, MANIFEST), "utf8"),
      readFile(path.join(ROOT, MLX_FIT_GATE), "utf8"),
      readFile(path.join(ROOT, ROUTING_CATALOG), "utf8"),
      readFile(path.join(ROOT, ROUTING_CANDLE), "utf8"),
      readFile(path.join(ROOT, ROUTING_MLX), "utf8"),
      readFile(path.join(ROOT, SCHEMA), "utf8"),
    ]);
  return {
    ledger: parseJsonc(ledgerBody),
    ledgerSource: ledgerBody,
    manifest: parseJsonc(manifestBody),
    mlxFitGate,
    routingCatalog,
    routingCandle,
    routingMlx,
    schema: JSON.parse(schemaBody),
  };
}

/**
 * MUTATION CHECK. A guard that passes because nothing violated it is a false green, so prove each rule
 * actually fires: perturb the real inputs so a component sits above the selected tier without a valid
 * declaration, and require the checker to fail with a message that names the problem.
 */
async function selfTest() {
  const inputs = await readInputs();
  const baseline = validate(inputs);
  const failures = [];
  // COUNTED, not hand-maintained. The reported total was stale in both directions before sc-17395
  // (it read "14" against 13 blocks), and a summary line that misreports its own coverage is the same
  // class of defect as a guard that does not run.
  const assertions = [];
  const expect = (label, errors, needle) => {
    assertions.push(label);
    if (errors.length === 0) {
      failures.push(`${label}: expected an error, got none`);
    } else if (!errors.some((error) => error.includes(needle))) {
      failures.push(`${label}: no error mentioned "${needle}". Got: ${errors.join(" | ")}`);
    }
  };
  if (baseline.errors.length > 0) {
    failures.push(`baseline must be clean, got: ${baseline.errors.join(" | ")}`);
  }

  const clone = () => JSON.parse(JSON.stringify(inputs.ledger));

  // 1. An undeclared dense text encoder: a new entry sets the runtime flag with no ledger row.
  {
    const manifest = JSON.parse(JSON.stringify(inputs.manifest));
    const victim = manifest.models.find((entry) => entry.id === "qwen_image");
    victim.mlx = { ...(victim.mlx ?? {}), denseTextEncoderTier: true };
    const ledger = clone();
    ledger.exceptions = ledger.exceptions.filter(
      (row) => !(row.model === "qwen_image" && row.component === "textEncoder"),
    );
    expect(
      "undeclared denseTextEncoderTier",
      validate({ ...inputs, manifest, ledger }).errors,
      "has no textEncoder exception",
    );
  }

  // 2. The Krea control branch sits above q4 with the exception row removed.
  {
    const ledger = clone();
    ledger.exceptions = ledger.exceptions.filter((row) => row.component !== "controlBranch");
    expect(
      "undeclared control-branch floor",
      validate({ ...inputs, ledger }).errors,
      "declares no controlBranch exception",
    );
  }

  // 3. A NEW model declares an above-tier component with no measurement — the ratchet.
  {
    const ledger = clone();
    ledger.exceptions.push({
      model: "z_image_turbo",
      component: "vae",
      residentTier: "bf16",
      appliesToTiers: ["q4"],
      cause: "packing-exception",
      reason: "A newly added model that quietly keeps its decoder dense on a packed tier.",
      evidence: { state: "unmeasured", source: "somewhere", owedBy: "sc-99999" },
    });
    expect(
      "unconditional measurement",
      validate({ ...inputs, ledger }).errors,
      "UNMEASURED exceptions are forbidden",
    );
  }

  // 3b. A promoted per-tier measurement cannot omit one of the tiers it claims to cover.
  {
    const ledger = clone();
    const row = ledger.exceptions.find(
      (item) => item.model === "qwen_image" && item.component === "vae",
    );
    delete row.evidence.costBytesByTier.q8;
    expect(
      "per-tier measurement coverage",
      validate({ ...inputs, ledger }).errors,
      "must exactly match appliesToTiers",
    );
  }

  // 3c. THE VAE SCOPE sc-17435 PINNED. Changing a decoder-only row to full-autoencoder without
  //     changing its receipt must fail structurally, not merely because a prose marker disappeared.
  {
    const ledger = clone();
    const row = ledger.exceptions.find(
      (item) =>
        item.model === "qwen_image" &&
        item.component === "vae" &&
        item.backends.join() === "candle",
    );
    row.evidence.vaeScope = VAE_SCOPES.FULL;
    expect(
      "receipt-bound VAE residency scope",
      validate({ ...inputs, ledger }).errors,
      "does not match the numeric receipt scope",
    );
  }

  // 3d. Restoring the old whole-file Qwen count under a decoder-only scope must fail arithmetic.
  {
    const ledger = clone();
    const row = ledger.exceptions.find(
      (item) =>
        item.model === "qwen_image" &&
        item.component === "vae" &&
        item.backends.join() === "candle",
    );
    row.evidence.costBytesByTier.q4 = 126892531 * 4;
    expect(
      "decoder-only VAE element count",
      validate({ ...inputs, ledger }).errors,
      "does not reproduce the independent safetensors-header receipt",
    );
  }

  // 3e. Restoring Mage's old whole-file scalar as a decoder byte count must fail arithmetic.
  {
    const ledger = clone();
    const row = ledger.exceptions.find(
      (item) => item.model === "mage_flow" && item.component === "vae",
    );
    row.evidence.costBytesByTier.q4 = 172478148 * 2;
    expect(
      "Mage decoder-only VAE element count",
      validate({ ...inputs, ledger }).errors,
      "does not reproduce the independent safetensors-header receipt",
    );
  }

  // 3f. FLUX.2 decode receipts include both live BN-stat vectors and exclude the unused counter.
  {
    const ledger = clone();
    const row = ledger.exceptions.find(
      (item) =>
        item.model === "lens" &&
        item.component === "vae" &&
        item.backends.join() === "candle",
    );
    row.evidence.costBytesByTier.q4 = 49620259 * 4;
    expect(
      "FLUX.2 decoder BN-stat element count",
      validate({ ...inputs, ledger }).errors,
      "does not reproduce the independent safetensors-header receipt",
    );
  }

  // 4. A "measured" row with no number.
  {
    const ledger = clone();
    const row = ledger.exceptions.find((item) => item.evidence.state === "measured");
    delete row.evidence.costGb;
    delete row.evidence.costBytes;
    expect("measured without a cost", validate({ ...inputs, ledger }).errors, "requires a positive");
  }

  // 5. An "exception" that is not above the tier it claims.
  {
    const ledger = clone();
    ledger.exceptions[0] = { ...ledger.exceptions[0], residentTier: "q4", appliesToTiers: ["q8"] };
    expect("not actually above tier", validate({ ...inputs, ledger }).errors, "is NOT above");
  }

  // 5b. A hosted tier variant missing from branchTierByBaseTier — an UNCHECKED branch tier, because the
  //     agreement loop above only walks declared keys.
  {
    const manifest = JSON.parse(JSON.stringify(inputs.manifest));
    const victim = manifest.models.find((entry) => entry.id === "krea_2_turbo");
    delete victim.candle.control.branchTierByBaseTier["int8-convrot"];
    expect(
      "branchTierByBaseTier coverage",
      validate({ ...inputs, manifest }).errors,
      'does not cover the hosted tier variant "int8-convrot"',
    );
  }

  // 6. The resolved Lens anti-drift binding.
  {
    expect(
      "Lens rehost anti-drift",
      validate({ ...inputs, mlxFitGate: "no resolution marker here" }).errors,
      LENS_REHOST_RESOLUTION,
    );
  }

  // 7. `backends` LANE OVERLAP — the uniqueness key is (model, component, lane), so a component may be
  //    split per lane, but two rows may never claim the same lane. Duplicating an mlx-only mage row
  //    verbatim must fail: a naive `model::component::backends` string key would have let it through.
  {
    const ledger = clone();
    const mageRow = ledger.exceptions.find(
      (item) => item.model === "mage_flow_base" && item.component === "transformerHead",
    );
    ledger.exceptions.push(JSON.parse(JSON.stringify(mageRow)));
    expect("lane overlap", validate({ ...inputs, ledger }).errors, 'declared twice for "mlx"');
  }

  // 7b. A row whose declared lane the ROUTER DOES NOT SERVE. `krea_realtime_14b` is MLX-only (no
  //     Candle provider or bespoke route), so a fabricated Candle claim describes a lane no job can
  //     reach. SCAIL-2 used to be this fixture, but its bespoke Candle lane is now part of the oracle.
  {
    const ledger = clone();
    const row = ledger.exceptions.find(
      (item) => item.model === "sana_1600m" && item.component === "vae",
    );
    row.model = "krea_realtime_14b";
    row.backends = ["candle"];
    expect("wrong lane", validate({ ...inputs, ledger }).errors, "only on mlx");
  }

  // 7c. THE HOLE sc-17395 CLOSED. `backends` was optional, and the lane cross-check above ran only
  //     when the key was present — so a row could claim BOTH lanes, including one that does not
  //     exist, purely by staying silent. Deleting the key must now be an error in its own right.
  {
    const ledger = clone();
    delete ledger.exceptions[0].backends;
    expect("omitted backends", validate({ ...inputs, ledger }).errors, "`backends` is REQUIRED");
  }

  // 7d. The same omission on a model with only ONE lane is the case that used to slip through
  //     ENTIRELY: no key, so no validation, so a silent claim on a lane the router does not serve.
  //     Both errors must fire — the missing key AND the impossible lane.
  {
    const ledger = clone();
    const row = ledger.exceptions.find(
      (item) => item.model === "sana_1600m" && item.component === "vae",
    );
    row.model = "krea_realtime_14b";
    delete row.backends;
    const { errors } = validate({ ...inputs, ledger });
    expect("omitted backends still lane-checked", errors, "only on mlx");
    expect("omitted backends on single-lane model", errors, "`backends` is REQUIRED");
  }

  // 7e. The lane oracle must FAIL CLOSED. The previous regex could not match rustfmt's wrapped
  //     `ModelCaps::new(` rows, so it resolved no lane for the four multi-line
  //     `sensenova_u1_8b_infographic_*` entries — and the manifest-block fallback passed them anyway,
  //     making a parse regression indistinguishable from a clean run. Emptying the routing sources
  //     must now be loud.
  {
    expect(
      "unresolvable lane oracle",
      validate({ ...inputs, routingCatalog: "", routingCandle: "", routingMlx: "" }).errors,
      "no lane could be resolved",
    );
  }

  // 7f. MUTATION CHECK ON THE PARSE ITSELF. A test that only asserts the oracle resolves something is
  //     a false green, so pin the multi-line form directly: `sensenova_u1_8b_infographic_v2` is
  //     declared across six lines, and the pre-sc-17395 single-line regex saw NO route for it.
  {
    const lanes = routedLanes(inputs);
    const multiLine = lanes.get("sensenova_u1_8b_infographic_v2");
    if (!multiLine || !multiLine.has("mlx") || !multiLine.has("candle")) {
      failures.push(
        "multi-line ModelCaps parse: sensenova_u1_8b_infographic_v2 must resolve to mlx+candle, got " +
          `${multiLine ? [...multiLine].join(",") : "nothing"}`,
      );
    }
    // And the bespoke-route half: instantid is `candle_routed: false` yet served by a shape route.
    const bespoke = lanes.get("instantid_realvisxl");
    if (!bespoke?.has("candle")) {
      failures.push(
        "bespoke route parse: instantid_realvisxl must resolve a candle lane from CANDLE_IMAGE_ROUTES",
      );
    }
  }

  // 8. The ledger's own SCHEMA is applied. A misspelled key (`backend` for `backends`) used to be a
  //    silent no-op — which is precisely how an inert `backends` field survived review.
  {
    const ledger = clone();
    ledger.exceptions[0] = { ...ledger.exceptions[0], backend: ["mlx"] };
    expect("schema unknown key", validate({ ...inputs, ledger }).errors, 'unknown property "backend"');
  }

  // 9. A provider-local floor cannot remain ledger-only: deleting the shared manifest declaration
  //    must make the tier decision, worker label, and UI visibly incomplete.
  {
    const manifest = JSON.parse(JSON.stringify(inputs.manifest));
    const victim = manifest.models.find((entry) => entry.id === "mage_flow_base");
    delete victim.precisionFloors;
    expect(
      "ledger floor missing from manifest",
      validate({ ...inputs, manifest }).errors,
      "has no matching precisionFloors declaration",
    );
  }

  // 10. A shared floor must cover every hosted backend. Regressing the transformer-head row to the
  //     historical MLX-only declaration must fail because Mage also ships a Candle lane.
  {
    const ledger = clone();
    const victim = ledger.exceptions.find(
      (row) => row.model === "mage_flow_base" && row.component === "transformerHead",
    );
    victim.backends = ["mlx"];
    expect(
      "manifest floor missing backend lane",
      validate({ ...inputs, ledger }).errors,
      'no matching exception for backend "candle"',
    );
  }

  if (failures.length > 0) {
    console.error("check-tier-integrity self-test FAILED:");
    for (const failure of failures) console.error(`  - ${failure}`);
    process.exitCode = 1;
    return;
  }
  console.log(
    `check-tier-integrity self-test passed (${assertions.length} mutations, ${baseline.rows.length} declared exceptions, ` +
      `${GRANDFATHERED_UNMEASURED.size} grandfathered pairs).`,
  );
}

async function main() {
  if (process.argv.includes("--self-test")) {
    await selfTest();
    return;
  }
  const inputs = await readInputs();
  const result = validate(inputs);
  if (result.errors.length > 0) {
    console.error("Tier-integrity violations (sc-15799):");
    for (const error of result.errors) console.error(`  - ${error}`);
    process.exitCode = 1;
    return;
  }
  const json = renderJson(result);
  const markdown = renderMarkdown(result);
  if (process.argv.includes("--check")) {
    const [currentJson, currentMd] = await Promise.all([
      readFile(path.join(ROOT, OUTPUT_JSON), "utf8").catch(() => ""),
      readFile(path.join(ROOT, OUTPUT_MD), "utf8").catch(() => ""),
    ]);
    if (currentJson !== json || currentMd !== markdown) {
      console.error(
        `${OUTPUT_JSON} / ${OUTPUT_MD} are stale. Run \`node scripts/check-tier-integrity.mjs\`.`,
      );
      process.exitCode = 1;
      return;
    }
    const measured = result.rows.filter((row) => row.evidenceState === "measured").length;
    console.log(
      `tier integrity OK: ${result.rows.length} declared exceptions (${measured} measured, ` +
        `${result.rows.length - measured} unmeasured).`,
    );
    return;
  }
  await Promise.all([
    writeFile(path.join(ROOT, OUTPUT_JSON), json),
    writeFile(path.join(ROOT, OUTPUT_MD), markdown),
  ]);
  console.log(`Wrote ${OUTPUT_JSON} and ${OUTPUT_MD} (${result.rows.length} exceptions).`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
