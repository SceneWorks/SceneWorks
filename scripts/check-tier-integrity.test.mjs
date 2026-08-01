/**
 * `node --test` coverage for `scripts/check-tier-integrity.mjs` (sc-15799).
 *
 * The checker ships a `--self-test` flag that mutates the REAL inputs and requires each rule to fire.
 * That is the right shape for "does this guard have teeth", but it lives inside the file under test and
 * runs as one pass/fail, so a rule can be deleted along with its own mutation and nothing notices. Every
 * other checker in `npm run check` has a sibling `node --test` file; this is that file. It asserts the
 * things `--self-test` structurally cannot: that the real ledger is clean, the completed amnesty is
 * pinned at zero, and every promoted row carries exact per-tier evidence and its audit trail.
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
  const [ledger, manifest, mlxFitGate, routingCatalog, schema] = await Promise.all([
    readFile(path.join(ROOT, "config/tier-integrity.jsonc"), "utf8"),
    readFile(path.join(ROOT, "config/manifests/builtin.models.jsonc"), "utf8"),
    readFile(path.join(ROOT, "crates/sceneworks-worker/src/mlx_fit_gate.rs"), "utf8"),
    readFile(path.join(ROOT, "crates/sceneworks-core/src/jobs_store/routing/catalog.rs"), "utf8"),
    readFile(path.join(ROOT, "packages/schemas/tier-integrity.schema.json"), "utf8"),
  ]);
  return {
    ledger: JSON.parse(stripJsoncComments(ledger)),
    ledgerSource: ledger,
    manifest: JSON.parse(stripJsoncComments(manifest)),
    mlxFitGate,
    routingCatalog,
    schema: JSON.parse(schema),
  };
}

const clone = (value) => JSON.parse(JSON.stringify(value));

test("the shipped ledger validates clean", async () => {
  const { errors } = validate(await realInputs());
  assert.deepEqual(errors, [], "the committed ledger must be conformant");
});

test("Lens q4/q8 use re-hosted packed turnkeys with no text-encoder exception", async () => {
  const { ledger, ledgerSource, manifest, mlxFitGate } = await realInputs();
  const resolution = "sc-16014-resolution: rehosted-q4-q8";
  assert.ok(ledgerSource.includes(resolution));
  assert.ok(mlxFitGate.includes(resolution));
  assert.deepEqual(
    ledger.exceptions.filter(
      (row) => ["lens", "lens_turbo"].includes(row.model) && row.component === "textEncoder",
    ),
    [],
  );
  for (const modelId of ["lens", "lens_turbo"]) {
    const model = manifest.models.find((entry) => entry.id === modelId);
    const expectedRepo =
      modelId === "lens" ? "SceneWorks/lens-mlx" : "SceneWorks/lens-turbo-mlx";
    assert.equal(model.mlx.standardTierLayout, true);
    assert.equal(model.mlx.denseTextEncoderTier, undefined);
    assert.deepEqual(
      model.downloads
        .filter((download) => ["q4", "q8"].includes(download.variant))
        .map(({ variant, repo }) => ({ variant, repo }))
        .sort((left, right) => left.variant.localeCompare(right.variant)),
      [
        { variant: "q4", repo: expectedRepo },
        { variant: "q8", repo: expectedRepo },
      ],
      `${modelId} must host both packed tiers instead of routing them through the bf16 encoder`,
    );
  }
});

test("the Lens resolution marker is required in both bound sources", async () => {
  const inputs = await realInputs();
  const resolution = "sc-16014-resolution: rehosted-q4-q8";
  for (const field of ["ledgerSource", "mlxFitGate"]) {
    const { errors } = validate({ ...inputs, [field]: "marker removed" });
    assert.ok(
      errors.some((error) => error.includes(resolution)),
      `${field} lost the shared decision without tripping anti-drift: ${errors.join(" | ")}`,
    );
  }
});

test("a Lens text-encoder exception cannot be reintroduced after the q4/q8 rehost", async () => {
  const inputs = await realInputs();
  const ledger = clone(inputs.ledger);
  ledger.exceptions.push({
    model: "lens_turbo",
    component: "textEncoder",
    residentTier: "bf16",
    appliesToTiers: ["q4", "q8"],
    cause: "backend-capability",
    reason: "A stale exception for the superseded MXFP4 source artifact.",
    evidence: { state: "measured", costGb: 17.24, source: "superseded bf16 measurement" },
  });
  const { errors } = validate({ ...inputs, ledger });
  assert.ok(
    errors.some((error) => error.includes("still declares 1 Lens textEncoder exception row")),
    `expected the resolved-row guard to fire, got: ${errors.join(" | ")}`,
  );
});

test("the temporary unmeasured amnesty is empty", () => {
  assert.deepEqual([...GRANDFATHERED_UNMEASURED], []);
  assert.equal(GRANDFATHERED_UNMEASURED_COUNT, 0);
});

test("the committed amnesty count remains zero", () => {
  assert.equal(
    GRANDFATHERED_UNMEASURED.size,
    GRANDFATHERED_UNMEASURED_COUNT,
    "the completed migration must not regain an amnesty slot",
  );
});

test("the ledger has no unmeasured rows", async () => {
  const { ledger } = await realInputs();
  const unmeasured = new Set(
    ledger.exceptions
      .filter((row) => (row.evidence ?? {}).state === "unmeasured")
      .map((row) => `${row.model}::${row.component}`),
  );
  assert.deepEqual(
    [...GRANDFATHERED_UNMEASURED].sort(),
    [...unmeasured].sort(),
    "sc-16015 removed both the temporary amnesty and every unmeasured row",
  );
});

test("an existing model cannot add an unmeasured row", async () => {
  const inputs = await realInputs();
  const ledger = clone(inputs.ledger);
  ledger.exceptions.push({
    model: "boogu_image",
    component: "textEncoder",
    residentTier: "bf16",
    appliesToTiers: ["q4", "q8"],
    cause: "packing-exception",
    reason: "An amnestied entry quietly adding a component the audit never declared for it.",
    evidence: { state: "unmeasured", source: "nowhere in particular", owedBy: "sc-99999" },
  });
  const { errors } = validate({ ...inputs, ledger });
  assert.ok(
    errors.some((error) => error.includes("UNMEASURED exceptions are forbidden")),
    `expected unconditional measured enforcement to fire, got: ${errors.join(" | ")}`,
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
  assert.ok(errors.some((error) => error.includes("UNMEASURED exceptions are forbidden")));
});

test("exact per-tier measurements cover every applicable tier", async () => {
  const inputs = await realInputs();
  const ledger = clone(inputs.ledger);
  const row = ledger.exceptions.find(
    (item) => item.model === "qwen_image" && item.component === "vae",
  );
  delete row.evidence.costBytesByTier.q8;
  const { errors } = validate({ ...inputs, ledger });
  assert.ok(
    errors.some((error) => error.includes("must exactly match appliesToTiers")),
    `expected per-tier coverage enforcement to fire, got: ${errors.join(" | ")}`,
  );
});

test("promoted measurements record isolation, triage, and the reviewed cause", async () => {
  const { ledger } = await realInputs();
  const promoted = ledger.exceptions.filter((row) => row.evidence.costBytesByTier);
  assert.ok(promoted.length > 0);
  for (const row of promoted) {
    assert.ok(
      ["measure", "eliminate"].includes(row.evidence.triageDecision),
      `${row.model}/${row.component}: triage`,
    );
    if (row.evidence.triageDecision === "eliminate") {
      assert.match(row.evidence.eliminationStory, /^sc-[1-9][0-9]*$/);
    } else {
      assert.equal(row.evidence.eliminationStory, undefined);
    }
    assert.equal(row.evidence.reviewedCause, row.cause, `${row.model}/${row.component}: cause review`);
    assert.ok(row.evidence.isolation.length >= 20, `${row.model}/${row.component}: isolation method`);
    assert.ok(row.evidence.triageNote.length >= 20, `${row.model}/${row.component}: triage note`);
  }
});

test("an elimination decision must name the story that owns the repack", async () => {
  const inputs = await realInputs();
  const ledger = clone(inputs.ledger);
  const row = ledger.exceptions.find(
    (item) => item.model === "chroma1_base" && item.component === "textEncoder",
  );
  delete row.evidence.eliminationStory;
  const { errors } = validate({ ...inputs, ledger });
  assert.ok(
    errors.some((error) => error.includes("must name its eliminationStory")),
    `expected elimination ownership enforcement to fire, got: ${errors.join(" | ")}`,
  );
});

test("only the six Chroma T5 and VAE pairs are scoped to sc-16462 elimination", async () => {
  const { ledger } = await realInputs();
  const actual = ledger.exceptions
    .filter((row) => row.evidence.triageDecision === "eliminate")
    .map((row) => `${row.model}::${row.component}::${row.evidence.eliminationStory}`)
    .sort();
  const expected = ["chroma1_base", "chroma1_flash", "chroma1_hd"]
    .flatMap((model) => ["textEncoder", "vae"].map((component) => `${model}::${component}::sc-16462`))
    .sort();
  assert.deepEqual(actual, expected);
});

test("a promoted measurement is bound to the catalog's pinned artifact revision", async () => {
  const inputs = await realInputs();
  const manifest = clone(inputs.manifest);
  const anima = manifest.models.find((entry) => entry.id === "anima_base");
  for (const download of anima.downloads) download.revision = "0".repeat(40);
  const { errors } = validate({ ...inputs, manifest });
  assert.ok(
    errors.some((error) => error.includes("measured artifact and the shipping artifact must move together")),
    `expected artifact-binding enforcement to fire, got: ${errors.join(" | ")}`,
  );
});

test("exact bytes reproduce the independent safetensors-header receipt", async () => {
  const inputs = await realInputs();
  const ledger = clone(inputs.ledger);
  const row = ledger.exceptions.find(
    (item) => item.model === "sdxl" && item.component === "vae" && item.backends?.includes("candle"),
  );
  row.evidence.costBytesByTier.q4 += 2;
  const { errors } = validate({ ...inputs, ledger });
  assert.ok(
    errors.some((error) => error.includes("does not reproduce the independent safetensors-header receipt")),
    `expected receipt arithmetic enforcement to fire, got: ${errors.join(" | ")}`,
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
        `${id} keeps its ${component} dense in every tier per both lanes' quant.rs, so it needs a row`,
      );
    }
  }
});

/**
 * The SenseNova dense heads are held at DIFFERENT precisions by the two lanes, so a single both-lanes
 * row is necessarily wrong in one direction. mlx-gen holds them at the checkpoint's bf16 store dtype;
 * candle-gen widens the two vision conv kernels (`vision.rs:38-47`) and `fm_head` (`fm.rs:15-25`) to
 * **f32** via `quant::get_f32` (sc-14249).
 *
 * The first revision declared `residentTier: "bf16"` with `backends` omitted — i.e. bf16 on BOTH lanes —
 * which under-declared candle. This pins the split so a merge back to one row fails here.
 */
test("SenseNova dense heads are declared per lane: candle f32, mlx bf16", async () => {
  const { ledger } = await realInputs();
  const senseNova = ledger.exceptions.filter((row) => row.model.startsWith("sensenova_u1"));
  const byLane = (model, component, lane) =>
    senseNova.filter(
      (row) =>
        row.model === model &&
        row.component === component &&
        Array.isArray(row.backends) &&
        row.backends.length === 1 &&
        row.backends[0] === lane,
    );
  const ids = [...new Set(senseNova.map((row) => row.model))];
  assert.equal(ids.length, 6, "all six SenseNova-U1 entries must be declared");
  for (const id of ids) {
    for (const component of ["transformerHead", "visionTower"]) {
      const candle = byLane(id, component, "candle");
      const mlx = byLane(id, component, "mlx");
      assert.equal(candle.length, 1, `${id}/${component}: exactly one candle-lane row`);
      assert.equal(mlx.length, 1, `${id}/${component}: exactly one mlx-lane row`);
      assert.equal(
        candle[0].residentTier,
        "f32",
        `${id}/${component}: candle-gen-sensenova widens this to f32 (get_f32, sc-14249), so declaring ` +
          `bf16 under-declares the candle lane`,
      );
      assert.equal(
        mlx[0].residentTier,
        "bf16",
        `${id}/${component}: mlx-gen holds this at the bf16 store dtype, so declaring f32 over-declares ` +
          `the mlx lane`,
      );
    }
  }
});

test("a declared backends lane must be one the catalog entry actually has", async () => {
  const { ledger, manifest, routingCatalog } = await realInputs();
  const byId = new Map(manifest.models.map((entry) => [entry.id, entry]));
  const routed = new Map(
    [...routingCatalog.matchAll(/ModelCaps::new\("([^"]+)",\s*(true|false),\s*(true|false),/g)].map(
      (match) => [match[1], { mlx: match[2] === "true", candle: match[3] === "true" }],
    ),
  );
  for (const row of ledger.exceptions) {
    if (!Array.isArray(row.backends)) continue;
    for (const lane of row.backends) {
      assert.ok(
        byId.get(row.model)?.[lane] || routed.get(row.model)?.[lane],
        `${row.model}/${row.component} claims the "${lane}" lane, but neither the manifest nor ` +
          `ModelCaps routes that backend — the claim describes nothing`,
      );
    }
  }
});

test("no two rows claim the same (model, component, lane)", async () => {
  const { ledger } = await realInputs();
  const claimed = new Map();
  for (const row of ledger.exceptions) {
    const key = `${row.model}::${row.component}`;
    const lanes = Array.isArray(row.backends) ? row.backends : ["mlx", "candle"];
    const already = claimed.get(key) ?? new Set();
    for (const lane of lanes) {
      assert.ok(
        !already.has(lane),
        `${key} is declared twice for the "${lane}" lane — the uniqueness key is (model, component, lane)`,
      );
      already.add(lane);
    }
    claimed.set(key, already);
  }
});

test("the generated audit publishes the backends column", async () => {
  const markdown = await readFile(path.join(ROOT, "docs/generated/tier-integrity.md"), "utf8");
  assert.match(
    markdown,
    /\| model \| component \| backends \|/,
    "the only per-backend claims in the ledger must be visible in the published audit; the first " +
      "revision's renderMarkdown dropped the column, making an mlx-only row shape-indistinguishable " +
      "from a both-lanes one",
  );
  assert.match(markdown, /\| candle \|/, "a candle-only row must render its lane");
  assert.match(markdown, /\| mlx \|/, "an mlx-only row must render its lane");
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
