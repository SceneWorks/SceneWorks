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
 *  3. The RATCHET. `unmeasured` is legal only for the entries grandfathered below. A NEW model cannot
 *     ship an above-tier component without a measured cost, and the grandfather list can only shrink.
 *     This is what makes AC "a check prevents a new model shipping a component above its tier without a
 *     declared exception" have teeth without pretending the existing debt was measured.
 *  4. `mlx.denseTextEncoderTier` — the one above-tier mechanism the RUNTIME reads — must be declared in
 *     the manifest AND matched by an exception row. Since sc-15799 deleted the hardcoded
 *     `DENSE_TE_TIER_MODELS` worker registry, the flag is the only way to obtain the carve-out, so a
 *     provider-local exception the shared decision cannot see is now structurally impossible.
 *  5. The Krea control branch's declared `branchTierByBaseTier` agrees with the ledger: every base tier
 *     whose branch tier is ABOVE it must have a `controlBranch` exception row, and every base tier whose
 *     branch tier equals it must NOT.
 *  6. ANTI-DRIFT on the lens-turbo carve-out: the numbers the ledger cites (on-disk / resident / upcast)
 *     must still appear in `crates/sceneworks-worker/src/mlx_fit_gate.rs`, whose doc table and hardcoded
 *     test constants are the other two copies. A repack that moved one and not the others used to turn
 *     that lane red for a reason unrelated to the invariant.
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
 * Entries GRANDFATHERED with an `unmeasured` cost by sc-15799 — the audit found the FACT of an
 * above-tier component in-tree but no isolated cost for it, and estimating one from a per-tier total
 * would be inventing evidence.
 *
 * This list lives HERE and not in the ledger on purpose: editing the ledger must not be able to widen
 * the amnesty. It may only ever SHRINK — sc-16015 empties it, and each promotion to `measured` removes
 * a line. A model NOT on this list may not declare an unmeasured exception at all, which is the ratchet
 * that stops the debt growing while the existing debt stays visible instead of being papered over.
 */
const GRANDFATHERED_UNMEASURED = new Set([
  "anima_aesthetic",
  "anima_base",
  "anima_turbo",
  "boogu_image",
  "boogu_image_edit",
  "boogu_image_turbo",
  "chroma1_base",
  "chroma1_flash",
  "chroma1_hd",
  "flux2_dev",
  "flux2_klein_9b",
  "flux2_klein_9b_kv",
  "ideogram_4",
  "ideogram_4_turbo",
  "illustrious_xl_v1",
  "illustrious_xl_v2",
  "kolors",
  "krea_2_turbo",
  "lens",
  "lens_turbo",
  "qwen_image",
  "qwen_image_edit_2511",
  "qwen_image_edit_2511_lightning",
  "realvisxl",
  "realvisxl_lightning",
  "sana_1600m",
  "sana_sprint_1600m",
  "sd3_5_large",
  "sd3_5_large_turbo",
  "sd3_5_medium",
  "sdxl",
]);

/**
 * The lens-turbo backend-capability numbers, which exist in THREE places: this ledger, the
 * `HEADROOM_GB` doc table in `mlx_fit_gate.rs`, and that file's hardcoded test constants. Any repack or
 * re-measure has to move all three together (sc-16014 owns the re-measure).
 */
const LENS_TURBO_ANTI_DRIFT = ["28.43", "45.67", "17.24"];

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
 * Validate the ledger against the catalog and the worker. Returns `{ errors, rows, unverifiedTiers }`;
 * pure so `--self-test` can drive it with mutated inputs.
 */
export function validate({ ledger, manifest, mlxFitGate }) {
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

  const seen = new Set();
  for (const row of exceptions) {
    const where = `${row.model ?? "?"}/${row.component ?? "?"}`;
    const entry = byId.get(row.model);
    if (!entry) {
      errors.push(`${where}: no catalog entry with id "${row.model}" in ${MANIFEST}.`);
      continue;
    }
    const key = `${row.model}::${row.component}`;
    if (seen.has(key)) {
      errors.push(`${where}: declared twice. Merge the rows — one component, one exception.`);
    }
    seen.add(key);

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
      if (!GRANDFATHERED_UNMEASURED.has(row.model)) {
        errors.push(
          `${where}: UNMEASURED exceptions are only permitted for the entries grandfathered by ` +
            `sc-15799. "${row.model}" is not one of them, so this component must ship a MEASURED ` +
            `cost — a new model may not add above-tier residency on trust. (If this is a genuine ` +
            `pre-existing entry the audit missed, that is a finding: measure it rather than widening ` +
            `the amnesty list.)`,
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
      backends: row.backends ?? ["mlx", "candle"],
      cause: row.cause,
      reason: row.reason,
      evidenceState: evidence.state,
      costGb: evidence.costGb ?? null,
      source: evidence.source ?? null,
      owedBy: evidence.owedBy ?? null,
      note: evidence.note ?? null,
    });
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
  }

  // (6) Anti-drift: the lens-turbo numbers must still exist where the fit gate reads them.
  for (const number of LENS_TURBO_ANTI_DRIFT) {
    if (!mlxFitGate.includes(number)) {
      errors.push(
        `${MLX_FIT_GATE} no longer contains "${number}", which ${LEDGER} cites for the lens-turbo ` +
          `mxfp4 upcast exception. That file holds the calibration doc table AND hardcoded test ` +
          `constants for the same measurement; a re-measure must move every copy together (sc-16014).`,
      );
    }
  }

  rows.sort(
    (a, b) => a.model.localeCompare(b.model) || a.component.localeCompare(b.component),
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
    "VAE, a text encoder, or both resident at bf16 or f32. Every one is declared below with the in-tree",
    "citation for the fact; the unmeasured rows have no isolated component cost in the tree yet and name",
    "the story that owes it. `scripts/check-tier-integrity.mjs` refuses an unmeasured exception from any",
    "entry not grandfathered by sc-15799, so this debt is bounded and can only shrink.",
    "",
    "| model | component | resident at | above tier on | cause | cost (GB) | evidence |",
    "| --- | --- | --- | --- | --- | --- | --- |",
  ];
  for (const row of rows) {
    const cost =
      row.evidenceState === "measured" && row.costGb !== null
        ? row.costGb.toFixed(3).replace(/\.?0+$/, "")
        : `_unmeasured_ (${row.owedBy ?? "no owner"})`;
    lines.push(
      `| \`${row.model}\` | ${row.component} | ${row.residentTier} | ${row.appliesToTiers.join(", ")} | ${row.cause} | ${cost} | ${row.source ?? ""} |`,
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
  const [ledgerBody, manifestBody, mlxFitGate] = await Promise.all([
    readFile(path.join(ROOT, LEDGER), "utf8"),
    readFile(path.join(ROOT, MANIFEST), "utf8"),
    readFile(path.join(ROOT, MLX_FIT_GATE), "utf8"),
  ]);
  return {
    ledger: parseJsonc(ledgerBody),
    manifest: parseJsonc(manifestBody),
    mlxFitGate,
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

  // 6. The lens-turbo anti-drift binding.
  {
    expect(
      "lens-turbo anti-drift",
      validate({ ...inputs, mlxFitGate: "no numbers here" }).errors,
      "no longer contains",
    );
  }

  if (failures.length > 0) {
    console.error("check-tier-integrity self-test FAILED:");
    for (const failure of failures) console.error(`  - ${failure}`);
    process.exitCode = 1;
    return;
  }
  console.log(
    `check-tier-integrity self-test passed (6 mutations, ${baseline.rows.length} declared exceptions).`,
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
