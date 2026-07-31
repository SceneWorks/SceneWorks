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
 *  2. Every exception carries EVIDENCE: `measured` needs a number and a source; `unmeasured` needs a
 *     source for the fact and a story that owes the measurement.
 *  3. The RATCHET, keyed per **(model, component)**. `unmeasured` is legal only for the exact pairs
 *     grandfathered below. A NEW model cannot ship an above-tier component without a measured cost, and
 *     neither can an already-amnestied model add an unmeasured row for a component it does not already
 *     declare. Keying on the model alone left ~155 free slots: any of the 31 amnestied ids could add an
 *     unmeasured row for any component it had not yet used, and two ids held amnesty they never used at
 *     all. The set's SIZE is asserted against a committed constant, so any growth is a visible two-place
 *     edit, and a grandfathered key with no matching ledger row is an ERROR — so a promotion to
 *     `measured` (or a deleted row) must delete its amnesty line rather than leaving a live slot behind.
 *     This is what makes AC "a check prevents a new model shipping a component above its tier without a
 *     declared exception" have teeth without pretending the existing debt was measured.
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
 *  7. `backends` is LOAD-BEARING, not a comment (sc-15799 review). The same declared tier can yield
 *     different residency on the two lanes — `sensenova_u1_8b::visionTower` is bf16 under mlx-gen and
 *     f32 under candle-gen, because the candle port widens the conv kernels — so the uniqueness key is
 *     **(model, component, LANE)**, not (model, component): a component whose residency genuinely
 *     differs per backend is split into one row per lane, and two rows may never claim the same lane.
 *     Lane names are validated, an explicitly declared lane must be one the catalog entry actually has
 *     a block for (a "candle only" row on an mlx-only entry describes a lane that does not exist), and
 *     the lane is PUBLISHED in the generated markdown so a per-backend claim is visible to a reader.
 *     Before this, `backends` was a defaulted passthrough that nothing validated and the audit dropped.
 *  8. The ledger validates against its own JSON Schema (`packages/schemas/tier-integrity.schema.json`),
 *     so an unknown or misspelled key is an error rather than a silently ignored field. The schema was
 *     unenforced when it was written, which is how it drifted to draft-07 while every sibling requires
 *     2020-12; `scripts/check-scaffold.mjs` now pins its conventions and this script applies it.
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

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const LEDGER = "config/tier-integrity.jsonc";
const SCHEMA = "packages/schemas/tier-integrity.schema.json";
const MANIFEST = "config/manifests/builtin.models.jsonc";
const MLX_FIT_GATE = "crates/sceneworks-worker/src/mlx_fit_gate.rs";
const OUTPUT_JSON = "docs/generated/tier-integrity.json";
const OUTPUT_MD = "docs/generated/tier-integrity.md";

/**
 * Fidelity ladder for "is this component resident ABOVE the selected tier?". Higher = more faithful.
 * `nvfp4` ranks EQUAL to `q4` and `int8-convrot` equal to `q8`: NVFP4 is a distinct numeric regime
 * (~4.5 effective bits over E2M1 elements, sc-11042) and int8-ConvRot is int8 with online rotation, so
 * claiming an ordering against their neighbours is not something this ladder is entitled to do. Mirrors
 * `gen_core::tier_integrity::fidelity_rank`.
 */
const FIDELITY = { f32: 4, bf16: 3, q8: 2, "int8-convrot": 2, q4: 1, nvfp4: 1 };

const COMPONENTS = new Set([
  "textEncoder",
  "vae",
  "visionTower",
  "controlBranch",
  "identityAdapters",
  "transformerHead",
]);

const CAUSES = new Set(["packing-exception", "backend-capability", "structural"]);

/**
 * The two generation lanes a row may claim in `backends`. Omitting `backends` claims BOTH.
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
 * `model::component` pairs GRANDFATHERED with an `unmeasured` cost by sc-15799 — the audit found the FACT
 * of an above-tier component in-tree but no isolated cost for it, and estimating one from a per-tier total
 * would be inventing evidence.
 *
 * This list lives HERE and not in the ledger on purpose: editing the ledger must not be able to widen the
 * amnesty. It is keyed per (model, component), not per model: keying per model handed every amnestied id a
 * free slot for every component it did not already declare, which is not a ratchet — `sdxl` +
 * `textEncoder` sailed through with `evidence.source: "nowhere in particular"`.
 *
 * It SHRINKS as each row is promoted to `measured` (sc-16015 empties it). It may grow only for an entry
 * the original audit MISSED — a pre-existing above-tier residency, never a new one — and only as a visible
 * two-place edit, because [`GRANDFATHERED_UNMEASURED_COUNT`] pins the size. A pair NOT on this list may
 * not declare an unmeasured exception at all.
 */
export const GRANDFATHERED_UNMEASURED = new Set([
  "anima_aesthetic::textEncoder",
  "anima_aesthetic::vae",
  "anima_base::textEncoder",
  "anima_base::vae",
  "anima_turbo::textEncoder",
  "anima_turbo::vae",
  "boogu_image::vae",
  "boogu_image_edit::vae",
  "boogu_image_turbo::vae",
  "chroma1_base::textEncoder",
  "chroma1_base::vae",
  "chroma1_flash::textEncoder",
  "chroma1_flash::vae",
  "chroma1_hd::textEncoder",
  "chroma1_hd::vae",
  "flux2_dev::vae",
  "flux2_klein_9b::textEncoder",
  "flux2_klein_9b::vae",
  "flux2_klein_9b_kv::textEncoder",
  "flux2_klein_9b_kv::vae",
  "ideogram_4::vae",
  "ideogram_4_turbo::vae",
  "illustrious_xl_v1::vae",
  "illustrious_xl_v2::vae",
  "kolors::vae",
  "krea_2_raw::vae",
  "krea_2_turbo::vae",
  "lens::vae",
  "lens_turbo::vae",
  "qwen_image::textEncoder",
  "qwen_image::vae",
  "qwen_image_edit_2511_lightning::textEncoder",
  "realvisxl::vae",
  "realvisxl_lightning::vae",
  "sana_1600m::vae",
  "sana_sprint_1600m::vae",
  "sd3_5_large::textEncoder",
  "sd3_5_large::vae",
  "sd3_5_large_turbo::textEncoder",
  "sd3_5_large_turbo::vae",
  "sd3_5_medium::textEncoder",
  "sd3_5_medium::vae",
  "sensenova_u1_8b::transformerHead",
  "sensenova_u1_8b::visionTower",
  "sensenova_u1_8b_fast::transformerHead",
  "sensenova_u1_8b_fast::visionTower",
  "sensenova_u1_8b_infographic_v2::transformerHead",
  "sensenova_u1_8b_infographic_v2::visionTower",
  "sensenova_u1_8b_infographic_v2_fast::transformerHead",
  "sensenova_u1_8b_infographic_v2_fast::visionTower",
  "sensenova_u1_8b_infographic_v3::transformerHead",
  "sensenova_u1_8b_infographic_v3::visionTower",
  "sensenova_u1_8b_infographic_v3_fast::transformerHead",
  "sensenova_u1_8b_infographic_v3_fast::visionTower",
]);

/**
 * The COMMITTED size of [`GRANDFATHERED_UNMEASURED`]. Asserted, so adding an amnesty line is a two-place
 * edit a reviewer cannot miss, and so the set cannot quietly grow to cover a row someone added. Nothing
 * else pinned the list's size before, which made "it may only ever SHRINK" advisory prose.
 *
 * 42 from sc-15799's own audit + 13 the sc-15799 review found it had MISSED (`krea_2_raw::vae` and the
 * twelve SenseNova-U1 rows) − the resolved `lens::textEncoder` row (sc-16014) = 54. sc-16015 drives it
 * to 0.
 */
export const GRANDFATHERED_UNMEASURED_COUNT = 54;

/**
 * Shared marker for the resolved Lens q4/q8 tier-integrity exception. It appears in the ledger source
 * and in `mlx_fit_gate.rs`; the checker requires both so removing the rows cannot silently erase the
 * fit-gate's remaining bf16-only MXFP4 accounting, or let that accounting drift back into a q4/q8 row.
 */
const LENS_REHOST_RESOLUTION = "sc-16014-resolution: rehosted-q4-q8";

function parseJsonc(body) {
  return JSON.parse(stripJsoncComments(body));
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
export function validate({ ledger, ledgerSource, manifest, mlxFitGate, schema }) {
  const errors = [];
  const rows = [];
  const unverifiedTiers = [];
  const byId = new Map((manifest.models ?? []).map((entry) => [entry.id, entry]));
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
  for (const row of exceptions) {
    const where = `${row.model ?? "?"}/${row.component ?? "?"}`;
    const entry = byId.get(row.model);
    if (!entry) {
      errors.push(`${where}: no catalog entry with id "${row.model}" in ${MANIFEST}.`);
      continue;
    }
    const key = `${row.model}::${row.component}`;

    // `backends`: which lanes this row speaks for. Validated, cross-checked and published (sc-15799
    // review) — it used to be a defaulted passthrough that nothing read.
    let lanes = BACKENDS;
    if (row.backends !== undefined) {
      if (!Array.isArray(row.backends) || row.backends.length === 0) {
        errors.push(
          `${where}: \`backends\`, when present, must name at least one lane (${BACKENDS.join(" / ")}). ` +
            `Omit the key entirely to claim both.`,
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
        // CROSS-CHECK: an explicit per-lane claim must be about a lane the entry actually has. A row
        // that says "candle" for an entry with no `candle` block is describing residency on a lane that
        // does not exist for it — undetectable while `backends` was inert.
        for (const lane of lanes) {
          if (!entry[lane]) {
            errors.push(
              `${where}: declares \`backends\` including "${lane}", but catalog entry "${row.model}" ` +
                `has no \`${lane}\` block, so it does not host that lane. A per-backend claim about a ` +
                `lane the entry does not have describes nothing — fix the lane, or drop the key to ` +
                `claim both.`,
            );
          }
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
            `(A row with no \`backends\` claims BOTH lanes.) Merge them, or narrow their \`backends\`.`,
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
    if (typeof evidence.source !== "string" || evidence.source.trim().length === 0) {
      errors.push(`${where}: evidence.source must cite where the fact is recorded in-tree.`);
    }
    if (evidence.state === "measured") {
      const cost = evidence.costGb ?? evidence.costBytes;
      if (typeof cost !== "number" || !Number.isFinite(cost) || cost <= 0) {
        errors.push(
          `${where}: evidence.state "measured" requires a positive costGb/costBytes. An exception ` +
            `whose cost nobody knows is undeclared in the only way that matters.`,
        );
      }
    } else if (evidence.state === "unmeasured") {
      if (!evidence.owedBy) {
        errors.push(`${where}: an unmeasured exception must name the story that owes the measurement.`);
      }
      if (!GRANDFATHERED_UNMEASURED.has(key)) {
        errors.push(
          `${where}: UNMEASURED exceptions are only permitted for the (model, component) PAIRS ` +
            `grandfathered by sc-15799. "${key}" is not one of them, so this component must ship a ` +
            `MEASURED cost — neither a new model nor an already-amnestied one may add above-tier ` +
            `residency on trust for a component it does not already declare. (If this is a genuine ` +
            `pre-existing residency the audit missed, that is a finding: measure it, or add the pair ` +
            `AND bump GRANDFATHERED_UNMEASURED_COUNT so the growth is reviewed.)`,
        );
      }
    } else {
      errors.push(`${where}: evidence.state must be "measured" or "unmeasured".`);
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
      source: evidence.source ?? null,
      owedBy: evidence.owedBy ?? null,
      note: evidence.note ?? null,
    });
  }

  // (3b) The amnesty list must not outlive the debt. Every grandfathered pair needs a live UNMEASURED
  // ledger row: once a row is promoted to `measured` (or deleted), its amnesty line is a free slot
  // somebody could later fill without review, so removing it is part of the promotion.
  const unmeasuredKeys = new Set(
    exceptions
      .filter((row) => (row.evidence ?? {}).state === "unmeasured")
      .map((row) => `${row.model}::${row.component}`),
  );
  for (const grandfathered of [...GRANDFATHERED_UNMEASURED].sort()) {
    if (!unmeasuredKeys.has(grandfathered)) {
      errors.push(
        `GRANDFATHERED_UNMEASURED still amnesties "${grandfathered}", but ${LEDGER} has no unmeasured ` +
          `exception for it. A promotion to \`measured\` (or a deleted row) must delete its amnesty ` +
          `line — otherwise the slot stays open for an unreviewed row. The list may only shrink.`,
      );
    }
  }
  if (GRANDFATHERED_UNMEASURED.size !== GRANDFATHERED_UNMEASURED_COUNT) {
    errors.push(
      `GRANDFATHERED_UNMEASURED has ${GRANDFATHERED_UNMEASURED.size} entries but ` +
        `GRANDFATHERED_UNMEASURED_COUNT is ${GRANDFATHERED_UNMEASURED_COUNT}. The size is committed on ` +
        `purpose: changing the amnesty must be a visible two-place edit, not a one-line addition.`,
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
    "VAE, a text encoder, or both resident at bf16 or f32. Every component that meets the declaration",
    "THRESHOLD stated in `config/tier-integrity.jsonc`'s header is declared below with the in-tree",
    "citation for the fact; the unmeasured rows have no isolated component cost in the tree yet and name",
    "the story that owes it. `scripts/check-tier-integrity.mjs` refuses an unmeasured exception for any",
    "`(model, component)` PAIR not grandfathered by sc-15799 — per pair, not per model, so an amnestied",
    "entry cannot add one for a component it does not already declare — pins the amnesty set's SIZE",
    "against a committed constant, and errors on a grandfathered pair whose row is gone. sc-16015 owns",
    "emptying it.",
    "",
    "**The same declared tier can yield different residency per backend**, so the `backends` column is",
    "part of each row's identity, not a footnote. The two ports make independent dtype decisions: the",
    "SenseNova-U1 vision conv kernels and `fm_head` are bf16 under `mlx-gen-sensenova` but **f32** under",
    "`candle-gen-sensenova`, which widens every dense leaf it multiplies against an f32 activation; the",
    "`mage_flow*` `transformerHead` q8 floor exists only in `mlx-gen-mage`, so a q4 candle render there",
    "really is uniformly q4. Such a component is declared as one row PER LANE. A row listing both lanes",
    "claims the residency holds identically on both.",
    "",
    "| model | component | backends | resident at | above tier on | cause | cost (GB) | evidence |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
  ];
  for (const row of rows) {
    const cost =
      row.evidenceState === "measured" && row.costGb !== null
        ? row.costGb.toFixed(3).replace(/\.?0+$/, "")
        : `_unmeasured_ (${row.owedBy ?? "no owner"})`;
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
  const [ledgerBody, manifestBody, mlxFitGate, schemaBody] = await Promise.all([
    readFile(path.join(ROOT, LEDGER), "utf8"),
    readFile(path.join(ROOT, MANIFEST), "utf8"),
    readFile(path.join(ROOT, MLX_FIT_GATE), "utf8"),
    readFile(path.join(ROOT, SCHEMA), "utf8"),
  ]);
  return {
    ledger: parseJsonc(ledgerBody),
    ledgerSource: ledgerBody,
    manifest: parseJsonc(manifestBody),
    mlxFitGate,
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
  const expect = (label, errors, needle) => {
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
    expect("ratchet", validate({ ...inputs, ledger }).errors, "not one of them");
  }

  // 3b. The RATCHET, per component: an ALREADY-AMNESTIED model adding an unmeasured row for a component
  //     it does not already declare. Keyed per MODEL, this passed with `errors: 0` — 31 amnestied ids x
  //     the components they had not declared left ~155 open slots.
  //
  //     The victim must be a model that genuinely HOLDS amnesty for a DIFFERENT component, or this
  //     mutation is just mutation 3 again. `boogu_image` is amnestied for `vae` and has no `textEncoder`
  //     row, so `boogu_image::textEncoder` is exactly the "already-amnestied, new component" case. (This
  //     used `sdxl`, which holds NO amnesty pair at all — so despite its label it only ever exercised
  //     the non-amnestied path, and reverting the ratchet to per-MODEL keying would not have failed it.
  //     Found by the sc-15799 review.)
  {
    const ledger = clone();
    ledger.exceptions.push({
      model: "boogu_image",
      component: "textEncoder",
      residentTier: "bf16",
      appliesToTiers: ["q4", "q8"],
      cause: "packing-exception",
      reason: "An amnestied entry quietly adding a component the audit never declared for it.",
      evidence: { state: "unmeasured", source: "nowhere in particular", owedBy: "sc-99999" },
    });
    expect(
      "per-component ratchet",
      validate({ ...inputs, ledger }).errors,
      'boogu_image::textEncoder" is not one of them',
    );
  }

  // 3c. An amnesty line that outlived its row: promoting a row to `measured` must delete its line, or the
  //     slot stays open for an unreviewed addition.
  {
    const ledger = clone();
    const row = ledger.exceptions.find(
      (item) => item.model === "krea_2_raw" && item.component === "vae",
    );
    row.evidence = { state: "measured", costGb: 0.5, source: "somewhere real" };
    expect(
      "orphaned amnesty line",
      validate({ ...inputs, ledger }).errors,
      'still amnesties "krea_2_raw::vae"',
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

  // 7b. A row whose declared lane the ENTRY DOES NOT HAVE. `sana_1600m` is mlx-only (no `candle` block),
  //     so claiming its dense VAE on candle describes residency on a lane that does not exist for it.
  //     Nothing caught this while `backends` was a defaulted passthrough.
  {
    const ledger = clone();
    const row = ledger.exceptions.find(
      (item) => item.model === "sana_1600m" && item.component === "vae",
    );
    row.backends = ["candle"];
    expect("wrong lane", validate({ ...inputs, ledger }).errors, "has no `candle` block");
  }

  // 8. The ledger's own SCHEMA is applied. A misspelled key (`backend` for `backends`) used to be a
  //    silent no-op — which is precisely how an inert `backends` field survived review.
  {
    const ledger = clone();
    ledger.exceptions[0] = { ...ledger.exceptions[0], backend: ["mlx"] };
    expect("schema unknown key", validate({ ...inputs, ledger }).errors, 'unknown property "backend"');
  }

  if (failures.length > 0) {
    console.error("check-tier-integrity self-test FAILED:");
    for (const failure of failures) console.error(`  - ${failure}`);
    process.exitCode = 1;
    return;
  }
  console.log(
    `check-tier-integrity self-test passed (12 mutations, ${baseline.rows.length} declared exceptions, ` +
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
        `${result.rows.length - measured} unmeasured and owned).`,
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
