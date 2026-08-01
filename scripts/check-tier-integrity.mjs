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
import { HEADER_MEASUREMENT_TERMS } from "./tier-integrity-measurement-receipts.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const LEDGER = "config/tier-integrity.jsonc";
const SCHEMA = "packages/schemas/tier-integrity.schema.json";
const MANIFEST = "config/manifests/builtin.models.jsonc";
const MLX_FIT_GATE = "crates/sceneworks-worker/src/mlx_fit_gate.rs";
const ROUTING_CATALOG = "crates/sceneworks-core/src/jobs_store/routing/catalog.rs";
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
export function validate({ ledger, ledgerSource, manifest, mlxFitGate, routingCatalog, schema }) {
  const errors = [];
  const rows = [];
  const unverifiedTiers = [];
  const byId = new Map((manifest.models ?? []).map((entry) => [entry.id, entry]));
  const routedBackends = new Map();
  for (const match of (routingCatalog ?? "").matchAll(
    /ModelCaps::new\("([^"]+)",\s*(true|false),\s*(true|false),/g,
  )) {
    routedBackends.set(match[1], new Set([
      ...(match[2] === "true" ? ["mlx"] : []),
      ...(match[3] === "true" ? ["candle"] : []),
    ]));
  }
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
          if (!entry[lane] && !routedBackends.get(row.model)?.has(lane)) {
            errors.push(
              `${where}: declares \`backends\` including "${lane}", but catalog entry "${row.model}" ` +
                `has neither a \`${lane}\` block nor a ${ROUTING_CATALOG} route, so it does not host ` +
                `that lane. A per-backend claim about a ` +
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
        const receiptKey = `${row.model}::${row.component}::${lanes.join("+")}`;
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
      const covered = new Set(matching.flatMap((row) => row.backends ?? BACKENDS));
      for (const lane of BACKENDS.filter((lane) => entry[lane])) {
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
    "`mage_flow*` `transformerHead` q8 floor exists only in `mlx-gen-mage`, so a q4 candle render there",
    "really is uniformly q4. Such a component is declared as one row PER LANE. A row listing both lanes",
    "claims the residency holds identically on both.",
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
  const [ledgerBody, manifestBody, mlxFitGate, routingCatalog, schemaBody] = await Promise.all([
    readFile(path.join(ROOT, LEDGER), "utf8"),
    readFile(path.join(ROOT, MANIFEST), "utf8"),
    readFile(path.join(ROOT, MLX_FIT_GATE), "utf8"),
    readFile(path.join(ROOT, ROUTING_CATALOG), "utf8"),
    readFile(path.join(ROOT, SCHEMA), "utf8"),
  ]);
  return {
    ledger: parseJsonc(ledgerBody),
    ledgerSource: ledgerBody,
    manifest: parseJsonc(manifestBody),
    mlxFitGate,
    routingCatalog,
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

  // 7b. A row whose declared lane the ENTRY DOES NOT HAVE. `z_image_edit` has neither a candle block
  //     nor a generic Candle ModelCaps route, so a fabricated candle claim describes no hosted lane.
  {
    const ledger = clone();
    const row = ledger.exceptions.find(
      (item) => item.model === "sana_1600m" && item.component === "vae",
    );
    row.model = "z_image_edit";
    row.backends = ["candle"];
    expect("wrong lane", validate({ ...inputs, ledger }).errors, "does not host that lane");
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
    `check-tier-integrity self-test passed (14 mutations, ${baseline.rows.length} declared exceptions, ` +
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
