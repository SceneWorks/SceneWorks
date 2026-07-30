/**
 * `node --test` coverage for `scripts/check-tier-integrity.mjs` (sc-15799).
 *
 * The checker ships a `--self-test` flag that mutates the REAL inputs and requires each rule to fire.
 * That is the right shape for "does this guard have teeth", but it lives inside the file under test and
 * runs as one pass/fail, so a rule can be deleted along with its own mutation and nothing notices. Every
 * other checker in `npm run check` has a sibling `node --test` file; this is that file. It asserts the
 * things `--self-test` structurally cannot: that the ratchet's shape and committed size are what they
 * claim, that the real ledger is clean, and that the amnesty set contains exactly the unmeasured rows.
 */

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  GRANDFATHERED_UNMEASURED,
  GRANDFATHERED_UNMEASURED_COUNT,
  validate,
} from "./check-tier-integrity.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

async function realInputs() {
  const [ledger, manifest, mlxFitGate] = await Promise.all([
    readFile(path.join(ROOT, "config/tier-integrity.jsonc"), "utf8"),
    readFile(path.join(ROOT, "config/manifests/builtin.models.jsonc"), "utf8"),
    readFile(path.join(ROOT, "crates/sceneworks-worker/src/mlx_fit_gate.rs"), "utf8"),
  ]);
  return {
    ledger: JSON.parse(stripJsoncComments(ledger)),
    manifest: JSON.parse(stripJsoncComments(manifest)),
    mlxFitGate,
  };
}

const clone = (value) => JSON.parse(JSON.stringify(value));

test("the shipped ledger validates clean", async () => {
  const { errors } = validate(await realInputs());
  assert.deepEqual(errors, [], "the committed ledger must be conformant");
});

test("the amnesty set is keyed per (model, component), not per model", () => {
  for (const key of GRANDFATHERED_UNMEASURED) {
    assert.match(
      key,
      /^[a-z0-9_]+::[a-zA-Z]+$/,
      `"${key}" must be a \`model::component\` pair — a bare model id hands the entry a free slot for ` +
        `every component it has not yet declared, which is the bypass sc-15799's review reproduced`,
    );
  }
});

test("the committed amnesty size matches the set, so growth is a two-place edit", () => {
  assert.equal(
    GRANDFATHERED_UNMEASURED.size,
    GRANDFATHERED_UNMEASURED_COUNT,
    "adding an amnesty line must also bump the committed count",
  );
});

test("the amnesty set is exactly the ledger's unmeasured rows", async () => {
  const { ledger } = await realInputs();
  const unmeasured = new Set(
    ledger.exceptions
      .filter((row) => (row.evidence ?? {}).state === "unmeasured")
      .map((row) => `${row.model}::${row.component}`),
  );
  assert.deepEqual(
    [...GRANDFATHERED_UNMEASURED].sort(),
    [...unmeasured].sort(),
    "no amnesty line may lack a row (an open slot) and no unmeasured row may lack a line",
  );
});

test("an already-amnestied model cannot add an unmeasured row for a new component", async () => {
  const inputs = await realInputs();
  const ledger = clone(inputs.ledger);
  ledger.exceptions.push({
    model: "sdxl",
    component: "textEncoder",
    residentTier: "bf16",
    appliesToTiers: ["q4", "q8"],
    cause: "packing-exception",
    reason: "An amnestied entry quietly adding a component the audit never declared for it.",
    evidence: { state: "unmeasured", source: "nowhere in particular", owedBy: "sc-99999" },
  });
  const { errors } = validate({ ...inputs, ledger });
  assert.ok(
    errors.some((error) => error.includes('sdxl::textEncoder" is not one of them')),
    `expected the per-component ratchet to fire, got: ${errors.join(" | ")}`,
  );
});

test("a brand-new model cannot add an unmeasured row at all", async () => {
  const inputs = await realInputs();
  const ledger = clone(inputs.ledger);
  ledger.exceptions.push({
    model: "z_image_turbo",
    component: "vae",
    residentTier: "bf16",
    appliesToTiers: ["q4"],
    cause: "packing-exception",
    reason: "A newly added model that quietly keeps its decoder dense on a packed tier.",
    evidence: { state: "unmeasured", source: "somewhere", owedBy: "sc-99999" },
  });
  const { errors } = validate({ ...inputs, ledger });
  assert.ok(errors.some((error) => error.includes("is not one of them")));
});

test("promoting a row to measured without deleting its amnesty line is an error", async () => {
  const inputs = await realInputs();
  const ledger = clone(inputs.ledger);
  const row = ledger.exceptions.find(
    (item) => item.model === "qwen_image" && item.component === "vae",
  );
  row.evidence = { state: "measured", costGb: 0.25, source: "a real measurement" };
  const { errors } = validate({ ...inputs, ledger });
  assert.ok(
    errors.some((error) => error.includes('still amnesties "qwen_image::vae"')),
    `expected the orphaned-amnesty check to fire, got: ${errors.join(" | ")}`,
  );
});

test("branchTierByBaseTier must cover every hosted tier variant", async () => {
  const inputs = await realInputs();
  const manifest = clone(inputs.manifest);
  const krea = manifest.models.find((entry) => entry.id === "krea_2_turbo");
  delete krea.candle.control.branchTierByBaseTier.q8;
  const { errors } = validate({ ...inputs, manifest });
  assert.ok(
    errors.some((error) => error.includes('does not cover the hosted tier variant "q8"')),
    `expected the coverage check to fire, got: ${errors.join(" | ")}`,
  );
});

test("the mage norm_out.linear floor is declared for every mage entry that has the DiT", async () => {
  const { ledger, manifest } = await realInputs();
  const mageIds = manifest.models.filter((entry) => entry.id.startsWith("mage_flow")).map((e) => e.id);
  const declared = new Set(
    ledger.exceptions
      .filter((row) => row.component === "transformerHead")
      .map((row) => row.model),
  );
  for (const id of mageIds) {
    assert.ok(
      declared.has(id),
      `${id} routes through the same MageTransformer, whose \`floor_bits(FINAL_MOD_BASE, bits)\` takes ` +
        `no model argument — so its q8 head floor must be declared`,
    );
  }
});

test("krea_2_raw declares the same dense VAE as krea_2_turbo", async () => {
  const { ledger } = await realInputs();
  const vaeRows = new Set(
    ledger.exceptions.filter((row) => row.component === "vae").map((row) => row.model),
  );
  assert.ok(vaeRows.has("krea_2_turbo"));
  assert.ok(
    vaeRows.has("krea_2_raw"),
    "Raw shares the whole KreaPipeline and hosts the same three tiers, so it has the same dense f32 VAE",
  );
});

test("every SenseNova-U1 entry declares its dense heads", async () => {
  const { ledger, manifest } = await realInputs();
  const ids = manifest.models.filter((entry) => entry.id.startsWith("sensenova_u1")).map((e) => e.id);
  assert.ok(ids.length > 0, "the fixture must actually contain SenseNova entries");
  for (const component of ["transformerHead", "visionTower"]) {
    const declared = new Set(
      ledger.exceptions.filter((row) => row.component === component).map((row) => row.model),
    );
    for (const id of ids) {
      assert.ok(
        declared.has(id),
        `${id} keeps its ${component} dense bf16 in every tier per both lanes' quant.rs, so it needs a row`,
      );
    }
  }
});

test("the retracted branchPackSaveGb key is gone from the catalog", async () => {
  const { manifest } = await realInputs();
  for (const entry of manifest.models) {
    assert.equal(
      entry.candle?.control?.branchPackSaveGb,
      undefined,
      `${entry.id}: branchPackSaveGb published q8 -8.4 / q4 -10.2 GB as a WEIGHT-side delta for a ` +
        `control branch that is 6.6 GB in total. It is retracted; sc-16013 owes the re-measure.`,
    );
  }
});
