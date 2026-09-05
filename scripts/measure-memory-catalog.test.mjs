import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, readFile, stat } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  ROOT,
  MATRIX_PATH,
  PLAN_PATH,
  PACKAGED_SOURCES_PATH,
  PROVIDER_FAMILIES,
  SDXL_COMPONENTS,
  anchorParts,
  anchorSlug,
  appendPackagedSource,
  capturedInCampaign,
  classifyAnchor,
  compiledInferencePin,
  familyFor,
  hubRoots,
  failureReason,
  extractSeedingNewAnchors,
  SEED_DIGEST,
  measureAnchor,
  parseArgs,
  providerCommand,
  planRun,
  readManifestModels,
  readPlan,
  snapshotPath,
  tierDownload,
  parseSdxlRoutes,
  readSdxlCandleRoutes,
  sdxlCandleRouteDrift,
  SDXL_ROUTES_PATH,
  SDXL_ROUTES_UNCHECKED,
} from "./measure-memory-catalog.mjs";
import { LTX25_LANE_PROVIDERS } from "./memory-calibration-harness.mjs";
import { parseBackendTierOverrides } from "./generate-memory-matrix.mjs";

const execFileAsync = promisify(execFile);
const REVISION = "0123456789abcdef0123456789abcdef01234567";
const UPSTREAM = "fedcba9876543210fedcba9876543210fedcba98";

function fakeModels() {
  return [
    {
      id: "qwen_image",
      downloads: [
        { repo: "SceneWorks/qwen-image-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] },
        { repo: "SceneWorks/qwen-image-mlx", revision: REVISION, variant: "q8", files: ["q8/*"] },
        { repo: "SceneWorks/other", revision: UPSTREAM, coRequisite: true, files: ["x"] },
      ],
    },
    {
      id: "minimax_h3",
      downloads: [
        { repo: "SceneWorks/minimax-h3-mlx", revision: REVISION, variant: "q4", files: ["q4/transformer/*"] },
        { repo: "MiniMaxAI/MiniMax-H3", revision: UPSTREAM, coRequisite: true, files: ["vae/*"] },
      ],
    },
    { id: "ltx_2_5", downloads: [{ repo: "SceneWorks/ltx-2.5-mlx", revision: REVISION, variant: "q4", files: ["distilled/q4/*"] }] },
    { id: "z_image_turbo", downloads: [{ repo: "SceneWorks/z-image-turbo-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] }] },
    // The base model is a distinct engine provider with its own artifact family.
    { id: "z_image", downloads: [{ repo: "SceneWorks/z-image-mlx", revision: UPSTREAM, variant: "q8", files: ["q8/*"] }] },
    // The edit model is a catalog alias for the Turbo provider driven in edit_image mode: it ships
    // the Turbo weights (worker engines.rs `z_image_edit → z_image_turbo`).
    { id: "z_image_edit", downloads: [{ repo: "SceneWorks/z-image-turbo-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] }] },
  ];
}

async function fakeHub(layout) {
  const hub = await mkdtemp(path.join(tmpdir(), "catalog-hub-"));
  for (const [repo, revision, ...rest] of layout) {
    await mkdir(snapshotPath(hub, repo, revision, ...rest), { recursive: true });
  }
  return hub;
}

test("anchor keys parse into their three parts and slug without separators", () => {
  assert.deepEqual(anchorParts("qwen_image:q4:mlx"), { modelId: "qwen_image", tier: "q4", backend: "mlx" });
  assert.equal(anchorSlug("qwen_image_edit_2511:bf16:candle"), "qwen-image-edit-2511-bf16-candle");
  assert.throws(() => anchorParts("qwen_image:fp8:mlx"), /not an anchor key/);
});

test("argument parsing requires the backend and the out-of-tree work dir, and --hf-cache repeats", () => {
  assert.throws(() => parseArgs(["--work-dir", "/x"]), /--backend must be/);
  assert.throws(() => parseArgs(["--backend", "mlx"]), /--work-dir is required/);
  assert.throws(() => parseArgs(["--backend", "mlx", "--work-dir", "/x", "--inference-repo", "/i"]), /--adapter is required/);
  assert.throws(() => parseArgs(["--backend", "mlx", "--work-dir", "/x", "--inference-repo", "/i", "--dry-run", "--campaign", "a/b"]), /one path segment/);
  const args = parseArgs([
    "--backend", "candle", "--work-dir", "/x", "--inference-repo", "/i", "--dry-run",
    "--hf-cache", "/a", "--hf-cache", "/b", "--anchors", "z_image_turbo:q4:candle,qwen_image:q4:candle",
    "--skip-current", "--no-commit",
  ]);
  assert.deepEqual(args.hfCache, ["/a", "/b"]);
  assert.deepEqual(args.anchors, ["z_image_turbo:q4:candle", "qwen_image:q4:candle"]);
  assert.equal(args.skipCurrent, true);
  assert.equal(args.commit, false);
  assert.equal(hubRoots(["/a"])[0], "/a");
  assert.equal(parseArgs(["--backend", "mlx", "--list"]).list, true);
  assert.deepEqual(providerCommand('["node", "fixture.mjs", "dir"]'), ["node", "fixture.mjs", "dir"]);
  assert.deepEqual(providerCommand("target/release/memory-mlx-adapter"), [path.resolve("target/release/memory-mlx-adapter")]);
  assert.throws(() => providerCommand("[1]"), /non-empty array of strings/);
});

test("the tier download is the non-corequisite entry for the tier, so its revision names the snapshot", () => {
  const models = fakeModels();
  assert.equal(tierDownload(models, "qwen_image", "SceneWorks/qwen-image-mlx", "q8").variant, "q8");
  // An upstream family with no per-tier entry falls back to any exact-revision download from it.
  assert.equal(tierDownload(models, "minimax_h3", "MiniMaxAI/MiniMax-H3", "q4").revision, UPSTREAM);
  assert.throws(() => tierDownload(models, "qwen_image", "SceneWorks/nope", "q4"), /no download from/);
  assert.throws(() => tierDownload(models, "absent", "SceneWorks/qwen-image-mlx", "q4"), /no model absent/);
});

test("classification: runnable anchors carry the adapter env family and the canonical tier root", async () => {
  const hub = await fakeHub([
    ["SceneWorks/qwen-image-mlx", REVISION, "q4"],
    ["SceneWorks/minimax-h3-mlx", REVISION, "q4"],
    ["MiniMaxAI/MiniMax-H3", UPSTREAM],
    ["SceneWorks/ltx-2.5-mlx", REVISION],
  ]);
  const context = { models: fakeModels(), backend: "mlx", hubs: [hub], current: new Map(), captured: new Map() };
  const qwen = await classifyAnchor("qwen_image:q4:mlx", { provider: "qwen_image" }, context);
  assert.equal(qwen.status, "runnable");
  assert.equal(qwen.physical, true, "the Qwen MLX arm needs the physical receipt session");
  assert.deepEqual(qwen.env, {
    SCENEWORKS_QWEN_IMAGE_REPOSITORY: "SceneWorks/qwen-image-mlx",
    SCENEWORKS_QWEN_IMAGE_REVISION: REVISION,
    SCENEWORKS_QWEN_IMAGE_ROOT: snapshotPath(hub, "SceneWorks/qwen-image-mlx", REVISION, "q4"),
  });

  const minimax = await classifyAnchor("minimax_h3:q4:mlx", { provider: "minimax_h3" }, context);
  assert.equal(minimax.status, "runnable");
  assert.equal(minimax.physical, false, "only the Qwen arm emits a sourceCapture; a raw-log pair would make the harness refuse the render");
  assert.equal(minimax.env.SCENEWORKS_MINIMAX_H3_UPSTREAM_ROOT, snapshotPath(hub, "MiniMaxAI/MiniMax-H3", UPSTREAM));
  assert.equal(minimax.env.SCENEWORKS_MINIMAX_H3_UPSTREAM_REVISION, UPSTREAM);

  const ltx = await classifyAnchor("ltx_2_5:q4:mlx", { provider: "ltx_2_5" }, context);
  assert.equal(ltx.status, "runnable");
  assert.equal(ltx.ltx25SnapshotRoot, snapshotPath(hub, "SceneWorks/ltx-2.5-mlx", REVISION));
  assert.deepEqual(ltx.env, {}, "the harness binds LTX-2.5 itself");

  const missingTier = await classifyAnchor("qwen_image:q8:mlx", { provider: "qwen_image" }, context);
  assert.equal(missingTier.status, "weights_missing");
  assert.match(missingTier.reason, /q8 on this host/);
});

test("the z-image family: the base model has its own env family, and the edit alias loads the Turbo artifact", async () => {
  const hub = await fakeHub([
    ["SceneWorks/z-image-mlx", UPSTREAM, "q8"],
    ["SceneWorks/z-image-turbo-mlx", REVISION, "q4"],
  ]);
  for (const backend of ["mlx", "candle"]) {
    const context = { models: fakeModels(), backend, hubs: [hub], current: new Map(), captured: new Map() };
    const base = await classifyAnchor(`z_image:q8:${backend}`, { provider: "z_image" }, context);
    assert.equal(base.status, "runnable", `${backend}: ${base.reason}`);
    assert.equal(base.physical, false);
    assert.deepEqual(base.env, {
      SCENEWORKS_Z_IMAGE_BASE_REPOSITORY: "SceneWorks/z-image-mlx",
      SCENEWORKS_Z_IMAGE_BASE_REVISION: UPSTREAM,
      SCENEWORKS_Z_IMAGE_BASE_ROOT: snapshotPath(hub, "SceneWorks/z-image-mlx", UPSTREAM, "q8"),
    }, "the base model must never be served from the Turbo family's env or artifact");
    const baseMissing = await classifyAnchor(`z_image:q4:${backend}`, { provider: "z_image" }, context);
    assert.equal(baseMissing.status, "weights_missing");
    assert.match(baseMissing.reason, /z-image-mlx@.*\/q4 on this host/);

    // The edit anchor's provider is the Turbo engine id; the tier root resolves through the
    // z_image_edit MANIFEST entry (which ships the Turbo weights), not through z_image_turbo's.
    const edit = await classifyAnchor(`z_image_edit:q4:${backend}`, { provider: "z_image_turbo", mode: "edit_image" }, context);
    assert.equal(edit.status, "runnable", `${backend}: ${edit.reason}`);
    assert.equal(edit.modelId, "z_image_edit");
    assert.deepEqual(edit.env, {
      SCENEWORKS_Z_IMAGE_REPOSITORY: "SceneWorks/z-image-turbo-mlx",
      SCENEWORKS_Z_IMAGE_REVISION: REVISION,
      SCENEWORKS_Z_IMAGE_ROOT: snapshotPath(hub, "SceneWorks/z-image-turbo-mlx", REVISION, "q4"),
    });
  }
});

// sc-22725: LTX-2.5 reaches the two lanes under two engine ids off ONE public snapshot. Both
// families must therefore derive the same snapshot root, on their own lane and on no other.
test("the LTX-2.5 family derives the same snapshot root on both lanes, under each lane's engine id", async () => {
  const hub = await fakeHub([["SceneWorks/ltx-2.5-mlx", REVISION, "distilled"]]);
  const expected = snapshotPath(hub, "SceneWorks/ltx-2.5-mlx", REVISION);
  for (const [backend, provider] of [["mlx", "ltx_2_5"], ["candle", "ltx_2_5_distilled"]]) {
    const context = { models: fakeModels(), backend, hubs: [hub], current: new Map(), captured: new Map() };
    for (const tier of ["q4", "q8", "bf16"]) {
      const row = await classifyAnchor(`ltx_2_5:${tier}:${backend}`, { provider }, context);
      assert.equal(row.status, "runnable", `${backend} ${tier}: ${row.reason}`);
      assert.equal(row.ltx25SnapshotRoot, expected, "the harness is handed the snapshot, not a tier root");
      assert.deepEqual(row.env, {}, "LTX-2.5 carries no adapter env family; the harness binds it");
      assert.equal(row.physical, false);
    }
    // The other lane's engine id must NOT be served here: an arm is per-family, not per-model.
    const crossed = await classifyAnchor(
      `ltx_2_5:q4:${backend}`,
      { provider: backend === "mlx" ? "ltx_2_5_distilled" : "ltx_2_5" },
      context,
    );
    assert.equal(crossed.status, "no_adapter_arm", `${backend} must not serve the other lane's engine id`);
  }
});

// sc-22725 review: the lane→engine-id mapping is written down TWICE — the harness refuses a plan row
// through `LTX25_LANE_PROVIDERS` (memory-calibration-harness.mjs), the catalog runner classifies one
// through `PROVIDER_FAMILIES[*].arms` — and nothing bound the two. Editing either alone produces a
// lane the harness will prepare but the runner will not schedule, or the reverse, with every test in
// both files still green. This is that binding.
test("LTX25_LANE_PROVIDERS and PROVIDER_FAMILIES agree on which lane serves which LTX-2.5 engine id", () => {
  for (const [backend, provider] of Object.entries(LTX25_LANE_PROVIDERS)) {
    const family = PROVIDER_FAMILIES[provider];
    assert.ok(family, `the harness routes ${backend} to ${provider}, which is not a provider family`);
    assert.equal(family.ltx25, true, `${provider} must be an LTX-2.5 family`);
    assert.deepEqual(
      family.arms,
      [backend],
      `${provider} must be served by exactly the lane the harness binds its snapshot for`,
    );
  }
  assert.deepEqual(
    Object.entries(PROVIDER_FAMILIES).filter(([, family]) => family.ltx25).map(([id]) => id).sort(),
    Object.values(LTX25_LANE_PROVIDERS).slice().sort(),
    "no family may claim the LTX-2.5 snapshot without a lane in LTX25_LANE_PROVIDERS to bind it",
  );
});

test("classification refuses what no adapter arm or the harness cannot serve, and skips what is done", async () => {
  const hub = await fakeHub([["SceneWorks/z-image-turbo-mlx", REVISION, "q4"]]);
  const base = { models: fakeModels(), backend: "candle", hubs: [hub], current: new Map(), captured: new Map() };
  const flux = await classifyAnchor("flux2_dev:q4:candle", { provider: "flux2_dev" }, base);
  assert.equal(flux.status, "no_adapter_arm");
  // `harness_unsupported` is the refusal for a provider whose adapter arm exists but whose
  // artifacts the harness cannot bind. No SHIPPED family is in that state since sc-22725 gave
  // LTX-2.5's candle engine id a real row, so the branch is driven through a synthetic family.
  const unsupported = await classifyAnchor("ltx_2_5:q4:candle", { provider: "ltx_2_5_distilled" }, {
    ...base,
    families: { ...PROVIDER_FAMILIES, ltx_2_5_distilled: { ltx25: true, repo: "SceneWorks/ltx-2.5-mlx", arms: ["candle"], harnessUnsupported: "a synthetic unbindable family" } },
  });
  assert.equal(unsupported.status, "harness_unsupported");
  assert.equal(unsupported.reason, "a synthetic unbindable family");
  assert.equal(
    Object.values(PROVIDER_FAMILIES).filter((family) => family.harnessUnsupported).length,
    0,
    "no shipped provider family is unbindable by the harness",
  );
  const other = await classifyAnchor("qwen_image:q4:mlx", { provider: "qwen_image" }, base);
  assert.equal(other.status, "other_backend");
  const done = await classifyAnchor("z_image_turbo:q4:candle", { provider: "z_image_turbo" }, {
    ...base, captured: new Map([["z_image_turbo:q4:candle", "docs/calibration/x/z-evidence.json"]]),
  });
  assert.equal(done.status, "already_captured");
  const current = await classifyAnchor("z_image_turbo:q4:candle", { provider: "z_image_turbo" }, {
    ...base, current: new Map([["z_image_turbo:q4:candle", true]]),
  });
  assert.equal(current.status, "runnable");
  assert.equal(current.current, true, "currency is reported; only --skip-current acts on it");
  const undeclared = await classifyAnchor("z_image_turbo:q4:candle", { provider: "z_image_turbo" }, {
    ...base, declaredLanes: new Set(["qwen_image:candle"]),
  });
  assert.equal(undeclared.status, "lane_undeclared", "--stamp-anchors would throw after the render");
  const declared = await classifyAnchor("z_image_turbo:q4:candle", { provider: "z_image_turbo" }, {
    ...base, declaredLanes: new Set(["z_image_turbo:candle"]),
  });
  assert.equal(declared.status, "runnable");
  // E1's second declaration: the provider's crate closure in config/inference-provider-closures.json.
  const providerUndeclared = await classifyAnchor("z_image_turbo:q4:candle", { provider: "z_image_turbo" }, {
    ...base, declaredProviders: new Set(["mlx:z_image_turbo"]),
  });
  assert.equal(providerUndeclared.status, "provider_undeclared");
  assert.match(providerUndeclared.reason, /inference-provider-closures\.json/);
  const providerDeclared = await classifyAnchor("z_image_turbo:q4:candle", { provider: "z_image_turbo" }, {
    ...base, declaredProviders: new Set(["candle:z_image_turbo"]),
  });
  assert.equal(providerDeclared.status, "runnable");
});

test("appending to PACKAGED_MEMORY_ANCHOR_SOURCES is idempotent and keeps the rustfmt tuple shape", async () => {
  const source = await readFile(path.join(ROOT, PACKAGED_SOURCES_PATH), "utf8");
  const relative = "docs/calibration/sc-99999/qwen-image-q4-mlx-evidence.json";
  const once = appendPackagedSource(source, relative);
  assert.equal(appendPackagedSource(once, relative), once);
  const expected = [
    "    (",
    `        "${relative}",`,
    `        include_str!("../../../${relative}"),`,
    "    ),",
    "];",
  ].join("\n");
  assert.ok(once.includes(expected), "new tuple is the last entry before the closing bracket");
  assert.equal(once.length - source.length, expected.length - "];".length);
  assert.throws(() => appendPackagedSource("const OTHER: &[u8] = &[];", relative), /no longer declares/);
});

test("a campaign directory's ingested bundles mark their anchors as already captured", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "catalog-campaign-"));
  const dir = "docs/calibration/sc-1";
  await mkdir(path.join(root, dir), { recursive: true });
  await writeFile(path.join(root, dir, "a-evidence.json"), JSON.stringify({
    records: [{ backend: "mlx", target: { modelId: "qwen_image", tier: "q4" } }],
  }));
  await writeFile(path.join(root, dir, "notes.json"), "{not json");
  const captured = await capturedInCampaign(root, dir);
  assert.deepEqual([...captured.entries()], [["qwen_image:q4:mlx", `${dir}/a-evidence.json`]]);
  assert.equal((await capturedInCampaign(root, "docs/calibration/absent")).size, 0);
});

test("every provider the committed plan declares is either served by a family row or refused by name", async () => {
  const plan = await readPlan();
  const models = await readManifestModels();
  for (const backend of ["mlx", "candle"]) {
    const { rows } = await planRun({ backend, anchors: null, campaign: "sc-catalog-test", hfCache: [], skipCurrent: false });
    const keys = Object.keys(plan.anchors).filter((key) => key.endsWith(`:${backend}`));
    assert.deepEqual(rows.map((row) => row.key), keys.sort(), `${backend}: one row per plan anchor`);
    for (const row of rows) {
      assert.ok(["runnable", "weights_missing", "no_adapter_arm", "harness_unsupported", "lane_undeclared", "provider_undeclared"].includes(row.status), `${row.key}: ${row.status}`);
      if (row.status === "lane_undeclared") assert.match(row.reason, /anchor-loader-closures\.json/);
      if (row.status === "provider_undeclared") assert.match(row.reason, /inference-provider-closures\.json/);
      // sc-22729: the family is resolved by the SAME rule classification uses — model-keyed when
      // the row declares the plan's provider, provider-keyed otherwise — so this coverage check
      // cannot drift from what `--list` actually did.
      const family = familyFor(row.modelId, row.provider);
      if (row.status === "no_adapter_arm") {
        assert.equal(family?.arms.includes(backend) ?? false, false);
      } else if (!["harness_unsupported", "lane_undeclared", "provider_undeclared"].includes(row.status)) {
        // A served provider must resolve a manifest download, or the classification could not name a root.
        tierDownload(models, row.modelId, family.repo, row.tier);
        assert.ok(row.roots.length > 0, `${row.key} names the root it would load`);
      }
    }
  }
  assert.match(await compiledInferencePin(), /^[0-9a-f]{40}$/);
});

/**
 * Every shipped tier of every routed model, as a `<modelId>:<tier>:<backend>` key.
 *
 * - SHIPPED tier: a non-corequisite manifest download whose `variant` is a numeric tier. The
 *   manifest (`config/manifests/builtin.models.jsonc`) is the only artifact that says what a user
 *   can download, so it is the only source for the tier axis.
 * - ROUTED lane: `models[].backends` in `docs/generated/memory-matrix.json`, which
 *   `generate-memory-matrix.mjs` derives from the worker's route resolvers
 *   (`crates/sceneworks-worker/src/memory_route_registry.rs`, the same `CANDLE_BESPOKE_REQUEST_PROVIDERS`
 *   and per-family engine tables the worker dispatches with). A lane that does not route the
 *   model is the ONLY exemption; a "structurally N/A" matrix cell is not one (epic 22723 E1).
 * - ROUTED tier: `parseBackendTierOverrides` — and NOTHING else. See `computeShippedTieredCells`.
 */

/**
 * The tier overrides that come from CODE, read out of the worker source the matrix generator itself
 * reads. `parseBackendTierOverrides` throws if the shape it parses is gone, so this cannot silently
 * degrade to "no overrides" and quietly widen the denominator either.
 */
async function codeDerivedTierOverrides() {
  return parseBackendTierOverrides(
    await readFile(path.join(ROOT, "crates/sceneworks-worker/src/image_jobs/instantid.rs"), "utf8"),
  );
}

async function computeShippedTieredCells() {
  const models = await readManifestModels();
  const matrix = JSON.parse(await readFile(path.join(ROOT, MATRIX_PATH), "utf8"));
  const routed = new Map(matrix.models.map((model) => [model.id, model]));
  const overrides = await codeDerivedTierOverrides();
  const cells = [];
  const dropped = [];
  for (const model of models) {
    const shipped = [...new Set(
      (model.downloads ?? [])
        .filter((download) => !download.coRequisite && ["q4", "q8", "bf16"].includes(download.variant))
        .map((download) => download.variant),
    )];
    if (shipped.length === 0) continue;
    const entry = routed.get(model.id);
    for (const backend of entry?.backends ?? []) {
      // sc-22729: a shipped tier is only a CELL on a lane whose CODE can load that tier at all.
      // `instantid_realvisxl` ships q4/q8/bf16 but its candle stack is dense-only and always loads
      // `bf16/` (`image_jobs/instantid.rs` instantid_memory_backend_keys / instantid_tier_subdir on
      // the non-macOS branch), so a q4 candle anchor could only ever measure bf16 weights and file
      // the peaks under a packed tier.
      //
      // The narrowing source is `parseBackendTierOverrides` — the SAME worker-source derivation the
      // matrix generator uses — and deliberately NOT the matrix's `axes.<backend>.tiers`. That list
      // falls back to `model.<backend>.vramGbByTier`, a MEASUREMENT declaration: a missing key there
      // means "no peak recorded yet", which is precisely the gap this set exists to count. Reading
      // it here let the manifest delete six real cells (flux_dev / flux_schnell / flux2_dev candle
      // bf16; sd3_5_large / sd3_5_large_turbo / sd3_5_medium candle q8) with no routing fact behind
      // it — see `the gap-set denominator is narrowed only by code-derived tier overrides`.
      const override = overrides.get(`${model.id}:${backend}`);
      for (const tier of shipped) {
        const cell = { modelId: model.id, tier, backend, key: `${model.id}:${tier}:${backend}` };
        if (override && !override.includes(tier)) dropped.push({ ...cell, override });
        else cells.push(cell);
      }
    }
  }
  return { cells, dropped };
}

/**
 * Both derivations are pure over the checked-in manifest, matrix and plan, and both are asked for by
 * more than one case — `measurabilityGaps()` alone is two full `planRun`s with per-anchor filesystem
 * probes. Memoized as module-level promises so the whole file pays for each exactly once.
 */
let shippedTieredCellsPromise;
function tieredCellUniverse() {
  shippedTieredCellsPromise ??= computeShippedTieredCells();
  return shippedTieredCellsPromise;
}
async function shippedTieredCells() {
  return (await tieredCellUniverse()).cells;
}

let measurabilityGapsPromise;
function measurabilityGaps() {
  measurabilityGapsPromise ??= computeMeasurabilityGaps();
  return measurabilityGapsPromise;
}

/** The measurability gap set: shipped cells `--list` does not classify runnable / weights_missing. */
async function computeMeasurabilityGaps() {
  const plan = await readPlan();
  const rows = new Map();
  for (const backend of ["mlx", "candle"]) {
    const run = await planRun({ backend, anchors: null, campaign: "sc-catalog-test", hfCache: [], skipCurrent: false });
    for (const row of run.rows) rows.set(row.key, row);
  }
  const gaps = [];
  for (const cell of await shippedTieredCells()) {
    const row = rows.get(cell.key);
    const status = row?.status ?? (plan.anchors[cell.key] ? "unclassified" : "no_plan_anchor");
    if (!["runnable", "weights_missing"].includes(status)) {
      gaps.push({ ...cell, status, reason: row?.reason ?? `${PLAN_PATH} declares no anchor ${cell.key}` });
    }
  }
  return gaps;
}

function gapReport(gaps) {
  const perModel = new Map();
  for (const gap of gaps) perModel.set(gap.modelId, (perModel.get(gap.modelId) ?? 0) + 1);
  return [
    `${gaps.length} shipped tier×lane cell(s) are not measurable (per model: ${[...perModel].map(([id, n]) => `${id}=${n}`).join(", ") || "none"})`,
    ...gaps.map((gap) => `  ${gap.key.padEnd(44)} ${gap.status.padEnd(20)} ${gap.reason}`),
  ].join("\n");
}

// sc-22729 review: the burndown DENOMINATOR. A cell leaves the universe only for a ROUTING fact
// read out of worker source — never for a manifest MEASUREMENT declaration.
//
// The hazard is concrete and was live: intersecting against the matrix's `axes.<backend>.tiers`
// (whose `tiersFor` falls back to `model.candle.vramGbByTier` keys) silently deleted six real cells
// whose only crime was carrying no recorded peak yet — the exact thing the gap set counts.
const MANIFEST_ONLY_DECLARED_CELLS = [
  "flux_dev:bf16:candle", "flux_schnell:bf16:candle", "flux2_dev:bf16:candle",
  "sd3_5_large:q8:candle", "sd3_5_large_turbo:q8:candle", "sd3_5_medium:q8:candle",
];

test("the gap-set denominator is narrowed only by code-derived tier overrides", async () => {
  const { cells, dropped } = await tieredCellUniverse();
  const overrides = await codeDerivedTierOverrides();
  // Every drop names an override key the WORKER SOURCE produced, and drops only tiers that key omits.
  for (const drop of dropped) {
    const key = `${drop.modelId}:${drop.backend}`;
    assert.ok(overrides.has(key), `${drop.key} was dropped with no code-derived override for ${key}`);
    assert.deepEqual(drop.override, overrides.get(key), drop.key);
    assert.ok(!drop.override.includes(drop.tier), `${drop.key} is inside its own override`);
  }
  // The only lane whose code narrows a shipped tier axis today.
  assert.deepEqual(
    dropped.map((drop) => drop.key).sort(),
    ["instantid_realvisxl:q4:candle", "instantid_realvisxl:q8:candle"],
    "a new drop must come with the worker source that justifies it",
  );

  // …and the six cells a manifest-declaration intersection would have deleted are all present.
  const universe = new Set(cells.map((cell) => cell.key));
  const manifest = await readManifestModels();
  const byId = new Map(manifest.map((model) => [model.id, model]));
  for (const key of MANIFEST_ONLY_DECLARED_CELLS) {
    const [modelId, tier] = key.split(":");
    assert.ok(universe.has(key), `${key} left the denominator with no routing fact behind it`);
    assert.ok(
      !Object.keys(byId.get(modelId)?.candle?.vramGbByTier ?? {}).includes(tier),
      `${key} no longer exercises the hazard: the manifest now declares a ${tier} candle peak for it`,
    );
  }
});

// Epic 22723 E1/E2: measurability is a SHAPE claim over the manifest and the plan — no weights, no
// GPU, no frozen count. `weights_missing` is measurable (the host merely lacks the snapshot);
// anything else names work: a missing plan anchor, a missing adapter arm, a missing closure
// declaration, or a harness that cannot bind the lane.
test("the z-image family is measurable on every shipped tier of every routed lane", async () => {
  const cells = (await shippedTieredCells()).filter((cell) => ["z_image", "z_image_edit", "z_image_turbo"].includes(cell.modelId));
  assert.ok(cells.length >= 3 * 3 * 2, "z_image / z_image_edit / z_image_turbo × three tiers × two lanes are all shipped and routed");
  const gaps = (await measurabilityGaps()).filter((gap) => ["z_image", "z_image_edit", "z_image_turbo"].includes(gap.modelId));
  assert.equal(gaps.length, 0, gapReport(gaps));
});

// sc-22725: the two families that already had one candle arm each — Qwen-Image (an ordinary tier
// root) and LTX-2.5 (a harness-bound snapshot under a second engine id) — on every shipped tier of
// every routed lane. Same shape claim as the z-image case above: manifest + plan + declarations.
test("qwen_image and ltx_2_5 are measurable on every shipped tier of every routed lane", async () => {
  const families = ["qwen_image", "ltx_2_5"];
  const cells = (await shippedTieredCells()).filter((cell) => families.includes(cell.modelId));
  assert.ok(cells.length >= 2 * 3 * 2, "both models × three tiers × two lanes are shipped and routed");
  const gaps = (await measurabilityGaps()).filter((gap) => families.includes(gap.modelId));
  assert.equal(gaps.length, 0, gapReport(gaps));
});

// sc-22729: the SDXL family — five catalog models the worker routes onto ONE engine id (`sdxl`)
// plus the bespoke `instantid` route — on every shipped tier of every lane that routes them.
//
// Every cell is DECLARED — a plan anchor and both closure declarations on both lanes. Two facts
// keep some of them from being capturable today, both engine-side rather than adapter gaps, and
// both asserted here so neither can silently widen:
//   * `instantid_realvisxl` q4/q8 on candle are not CELLS at all — the candle InstantID stack is
//     dense-only, which the worker's own `instantid.rs` says, so the universe never produces them.
//   * all three tiers of `illustrious_xl_v1`/`v2` on candle are cells the engine cannot SEAL at
//     inference c6d6a4db: `candle-gen-sdxl`'s `SDXL_ROUTES` pins revisions this repository no
//     longer ships. The classification is derived from both sources, so it clears itself when the
//     inference-side fix (`story/sc-22729-sdxl-route-revisions`) lands and the pin moves.
const SDXL_FAMILY = [
  "sdxl", "realvisxl", "realvisxl_lightning", "illustrious_xl_v1", "illustrious_xl_v2",
  "instantid_realvisxl",
];
/** Every candle cell the route drift blocks: both Illustrious models × all three shipped tiers. */
const SDXL_CANDLE_ROUTE_DRIFT = ["illustrious_xl_v1", "illustrious_xl_v2"]
  .flatMap((modelId) => ["q4", "q8", "bf16"].map((tier) => `${modelId}:${tier}:candle`));

test("the sdxl family is measurable on every shipped tier of every routed lane", async () => {
  const cells = (await shippedTieredCells()).filter((cell) => SDXL_FAMILY.includes(cell.modelId));
  const perModel = new Map();
  for (const cell of cells) perModel.set(cell.modelId, (perModel.get(cell.modelId) ?? 0) + 1);
  for (const modelId of SDXL_FAMILY) {
    assert.ok(perModel.get(modelId) > 0, `${modelId} ships no tiered cell at all`);
  }
  // The candle InstantID lane is bf16-only, so its packed tiers are not cells.
  assert.deepEqual(
    cells.filter((cell) => cell.modelId === "instantid_realvisxl" && cell.backend === "candle")
      .map((cell) => cell.tier),
    ["bf16"],
    "the candle InstantID stack is dense-only; a packed candle cell would measure bf16 weights",
  );
  // Every family cell is measurable except the Illustrious candle routes the engine cannot seal —
  // and with no inference checkout to read, there are none of those either, because the refusal is
  // DERIVED. Both directions are asserted so neither mode can quietly assert nothing.
  const gaps = (await measurabilityGaps()).filter((gap) => SDXL_FAMILY.includes(gap.modelId));
  assert.deepEqual(
    gaps.map((gap) => gap.key).sort(),
    (await readSdxlCandleRoutes()) ? [...SDXL_CANDLE_ROUTE_DRIFT].sort() : [],
    gapReport(gaps),
  );
  // …and those are refused for the engine's own reason, not as missing plan or adapter work.
  for (const gap of gaps) {
    assert.equal(gap.status, "harness_unsupported", `${gap.key}: ${gap.reason}`);
    assert.match(gap.reason, /candle-gen-sdxl pins route/, gap.key);
    assert.match(gap.reason, /path_has_snapshot/, gap.key);
    assert.doesNotMatch(gap.reason, /declares no anchor/, `${gap.key} IS planned`);
  }
});

// sc-22729 review: all 33 cells the family ships are DECLARED — a plan anchor plus both closure
// declarations on every routed lane. A cell an engine defect blocks today is still declared: the
// defect blocks CAPTURE, never DECLARATION (epic 22723 E1), and a dropped declaration would erase
// the only record that the cell is owed a measurement.
test("every sdxl-family cell carries a plan anchor and a loader-closure declaration on its lane", async () => {
  const plan = await readPlan();
  const closures = JSON.parse(await readFile(path.join(ROOT, "config/anchor-loader-closures.json"), "utf8"));
  const cells = (await shippedTieredCells()).filter((cell) => SDXL_FAMILY.includes(cell.modelId));
  // The SHAPE, not a total: the five `sdxl` members carry all three tiers on both lanes, and the
  // bespoke InstantID route carries three on MLX and only its dense tier on candle. (34 cells; the
  // story text's "33" predates the derivation and undercounts by one.)
  const shape = new Map();
  for (const cell of cells) {
    const key = `${cell.modelId}:${cell.backend}`;
    shape.set(key, [...(shape.get(key) ?? []), cell.tier].sort());
  }
  assert.deepEqual(
    Object.fromEntries([...shape].sort()),
    Object.fromEntries([
      ...["sdxl", "realvisxl", "realvisxl_lightning", "illustrious_xl_v1", "illustrious_xl_v2"]
        .flatMap((id) => ["candle", "mlx"].map((lane) => [`${id}:${lane}`, ["bf16", "q4", "q8"]])),
      ["instantid_realvisxl:candle", ["bf16"]],
      ["instantid_realvisxl:mlx", ["bf16", "q4", "q8"]],
    ].sort()),
  );
  for (const cell of cells) {
    assert.ok(plan.anchors[cell.key], `${PLAN_PATH} declares no anchor ${cell.key}`);
    assert.ok(
      closures.models[`${cell.modelId}:${cell.backend}`],
      `config/anchor-loader-closures.json declares no loader closure ${cell.modelId}:${cell.backend}`,
    );
  }
});

// sc-22729 review: the exclusion is DERIVED from both revisions, never written down. Two directions:
// the pinned tree really does disagree today (so the refusal is live and not a leftover), and an
// engine that agrees clears it with no edit to this repository.
// Needs the pinned inference source: the whole claim is about what the ENGINE declares. On CI the
// parity-scaffold job fetches it and sets INFERENCE_REPO, so a missing clone there is a failure and
// not a skip — the same rule `anchor-loader-closure.test.mjs` follows for the same reason.
const sdxlRoutesAvailable = (await readSdxlCandleRoutes()) !== null;
if (!sdxlRoutesAvailable && process.env.CI) {
  throw new Error(
    `no inference checkout supplies ${SDXL_ROUTES_PATH}. On CI this is a FAILURE, not a skip: ` +
      "check.yml's parity-scaffold job fetches the pinned revision and sets INFERENCE_REPO.",
  );
}
const skipWithoutRoutes = sdxlRoutesAvailable
  ? false
  : `no inference checkout supplies ${SDXL_ROUTES_PATH}`;

test("the illustrious candle refusal is derived from the engine's own route revision", { skip: skipWithoutRoutes }, async () => {
  const models = await readManifestModels();
  const routes = await readSdxlCandleRoutes();
  for (const modelId of ["sdxl", "realvisxl", "realvisxl_lightning"]) {
    assert.equal(sdxlCandleRouteDrift(modelId, "q4", routes, models), null, `${modelId} must not be excluded`);
  }
  for (const modelId of ["illustrious_xl_v1", "illustrious_xl_v2"]) {
    const route = routes.get(modelId);
    const shipped = tierDownload(models, modelId, route.repository, "q4").revision;
    assert.notEqual(route.revision, shipped, `${modelId}: the engine and the manifest agree — drop the exclusion`);
    for (const tier of ["q4", "q8", "bf16"]) {
      assert.match(sdxlCandleRouteDrift(modelId, tier, routes, models), /candle-gen-sdxl pins route/);
    }
    // The same model, against an engine that ships what the manifest ships: no refusal at all.
    const agreed = new Map(routes).set(modelId, { ...route, revision: shipped });
    assert.equal(sdxlCandleRouteDrift(modelId, "q4", agreed, models), null, `${modelId}: equal revisions must clear it`);
  }
  // A route the engine does not declare AT ALL is refused for that reason, not silently admitted.
  const without = new Map(routes);
  without.delete("illustrious_xl_v1");
  assert.match(sdxlCandleRouteDrift("illustrious_xl_v1", "q4", without, models), /declares no route/);
});

test("SDXL_ROUTES is parsed from the engine source, and an unreadable checkout refuses nothing", async () => {
  const routes = parseSdxlRoutes(`
pub const SDXL_ROUTES: &[SdxlRoute] = &[
    SdxlRoute { id: "a", repository: "Org/a", revision: "aa", edit: true, lightning: false },
    SdxlRoute { id: "b", repository: "Org/b", revision: "bb", edit: false, lightning: true },
];
`);
  assert.deepEqual([...routes.keys()], ["a", "b"]);
  assert.deepEqual(routes.get("b"), { repository: "Org/b", revision: "bb" });
  assert.throws(() => parseSdxlRoutes("// no table here"), /no longer declares a parsable SDXL_ROUTES/);
  assert.equal(await readSdxlCandleRoutes(path.join(ROOT, "no", "such", "checkout")), null);
  assert.equal(await readSdxlCandleRoutes(""), null, "no inference checkout is not a refusal");

  // With nothing to compare against, the cell classifies as it otherwise would and SAYS so.
  const plan = await readPlan();
  const key = "illustrious_xl_v1:q4:candle";
  const row = await classifyAnchor(key, plan.anchors[key], {
    models: await readManifestModels(),
    backend: "candle",
    hubs: [path.join(ROOT, "no", "such", "hub")],
    current: new Map(),
    captured: new Map(),
    declaredLanes: new Set(["illustrious_xl_v1:candle"]),
    declaredProviders: new Set(["candle:sdxl"]),
    sdxlRoutes: null,
  });
  assert.equal(row.status, "weights_missing");
  assert.equal(row.routeCheck, SDXL_ROUTES_UNCHECKED);
});

// sc-22729: the three caller-staged SDXL components are declared in TWO places — the catalog's
// `SDXL_COMPONENTS` and the candle adapter's own `SDXL_COMPONENTS` — and `candle-gen-sdxl`
// validates all three at exact upstream revisions. A rename on one side would leave a capture
// binding a component the engine never sees, so the two lists are proven equal here.
test("the staged SDXL component env vars agree between the catalog and the candle adapter", async () => {
  const source = await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/bin/candle.rs"), "utf8");
  const declared = [...source.matchAll(/"(SCENEWORKS_SDXL_COMPONENT_[A-Z0-9_]+)"/g)].map((match) => match[1]);
  assert.deepEqual(
    [...new Set(declared)].sort(),
    SDXL_COMPONENTS.map((component) => component.env).sort(),
    "candle.rs SDXL_COMPONENTS and the catalog's SDXL_COMPONENTS must name the same env vars",
  );
  // Every component repo is a real corequisite of every SDXL-family model, so `tierDownload`
  // resolves a revision for it rather than falling back to an unrelated download.
  const models = await readManifestModels();
  for (const modelId of SDXL_FAMILY) {
    for (const component of SDXL_COMPONENTS) {
      const download = tierDownload(models, modelId, component.repo, "q4");
      assert.match(download.revision, /^[0-9a-f]{40}$/, `${modelId}/${component.repo}`);
      assert.equal(download.coRequisite, true, `${modelId}/${component.repo} must be a corequisite`);
    }
  }
});

// The catalog-wide burndown. `todo` until the terminal story of epic 22723 (sc-22738) promotes it:
// node:test reports a failing todo without failing the run, so the gap set is printed on every
// `npm run check` while the other families are brought in, and the assertion itself is already
// the one that will gate. Remove the `todo` option to promote; do not add a count.
test("every shipped tiered model is measurable", { todo: "epic 22723 burndown; sc-22738 promotes this to a hard assertion" }, async () => {
  const gaps = await measurabilityGaps();
  assert.equal(gaps.length, 0, gapReport(gaps));
});

test("failure reasons name the thrown error, not the Node banner after it", () => {
  const stderr = [
    "file:///x/harness.mjs:376",
    "  throw new Error(`memory-strategy calibration: ${message}`);",
    "Error: memory-strategy calibration: imc-1.hardware.model must be a non-empty string",
    "    at fail (file:///x/harness.mjs:376:9)",
    "Node.js v24.15.0",
  ].join("\n");
  assert.equal(failureReason({ stderr, message: "node exited 1" }), "Error: memory-strategy calibration: imc-1.hardware.model must be a non-empty string");
  assert.equal(failureReason({ message: "plain" }), "plain");
});

// A hermetic checkout: stub harness + derivation scripts standing in for the real ones, so the
// per-anchor sequence (capture → check → ingest → PACKAGED list → extract → stamp → matrix →
// commit) and its rollback can be exercised without weights, a GPU or an inference clone.
async function stubCheckout() {
  const root = await mkdtemp(path.join(tmpdir(), "catalog-checkout-"));
  const git = (...args) => execFileAsync("git", args, { cwd: root });
  await mkdir(path.join(root, "scripts"), { recursive: true });
  await mkdir(path.join(root, "config"), { recursive: true });
  await mkdir(path.join(root, "docs/generated"), { recursive: true });
  await mkdir(path.join(root, path.dirname(PACKAGED_SOURCES_PATH)), { recursive: true });
  await writeFile(path.join(root, "scripts/memory-calibration-harness.mjs"), `
    import { writeFile, readFile } from "node:fs/promises";
    const [command, ...args] = process.argv.slice(2);
    const value = (flag) => args[args.indexOf(flag) + 1];
    if (command === "capture") {
      if (process.env.STUB_CAPTURE_FAILS) { console.error("Error: stub capture refused"); process.exit(1); }
      await writeFile(value("--output"), JSON.stringify({ records: [{ backend: "mlx", target: { modelId: "z_image_turbo", tier: "q4" }, env: process.env.SCENEWORKS_Z_IMAGE_ROOT ?? null }] }));
    } else if (command === "ingest") {
      await writeFile(value("--output"), await readFile(value("--input")));
    } else if (command !== "check") { process.exit(2); }
  `);
  // The stub extractor behaves like the real one on a new anchor id: it refuses until the store
  // carries the id, then writes the store with whatever key the store carried (the placeholder).
  await writeFile(path.join(root, "scripts/extract-memory-anchors.mjs"), `
    import { writeFile, readFile } from "node:fs/promises";
    const store = JSON.parse(await readFile("config/memory-anchors.json", "utf8"));
    const id = "z_image_turbo:mlx:q4:base:base:fp:imc-new";
    const seeded = (store.anchors ?? []).find((anchor) => anchor.id === id);
    if (!seeded) {
      console.error("Error: anchor " + id + " has no recorded loader-closure digest in config/memory-anchors.json. A newly extracted anchor must ...");
      process.exit(1);
    }
    await writeFile("config/memory-anchors.json", JSON.stringify({ anchors: [{ id, modelId: "z_image_turbo", backend: "mlx", source: { loaderClosureDigest: seeded.source.loaderClosureDigest } }] }, null, 2) + "\\n");
  `);
  await writeFile(path.join(root, "scripts/anchor-loader-closure.mjs"), `
    import { writeFile, readFile } from "node:fs/promises";
    if (process.env.STUB_FAIL_AT === "anchor-loader-closure.mjs") { console.error("Error: anchor-loader-closure.mjs stub failed"); process.exit(1); }
    const store = JSON.parse(await readFile("config/memory-anchors.json", "utf8"));
    for (const anchor of store.anchors) anchor.source.loaderClosureDigest = "a".repeat(64);
    await writeFile("config/memory-anchors.json", JSON.stringify(store, null, 2) + "\\n");
  `);
  await writeFile(path.join(root, "scripts/generate-memory-matrix.mjs"), `
    import { writeFile } from "node:fs/promises";
    await writeFile("docs/generated/memory-matrix.json", JSON.stringify({ at: Date.now() }));
    await writeFile("docs/generated/memory-matrix.md", "# matrix " + Date.now() + "\\n");
    if (process.env.STUB_STRAY) await writeFile("docs/generated/stray.json", "{}");
  `);
  await writeFile(path.join(root, "config/memory-anchors.json"), JSON.stringify({ anchors: [] }) + "\n");
  await writeFile(path.join(root, "docs/generated/memory-matrix.json"), "{}\n");
  await writeFile(path.join(root, "docs/generated/memory-matrix.md"), "# matrix\n");
  await writeFile(path.join(root, ".gitignore"), "*.log\n");
  await writeFile(path.join(root, PACKAGED_SOURCES_PATH), [
    "const PACKAGED_MEMORY_ANCHOR_SOURCES: &[(&str, &str)] = &[",
    "    (",
    '        "docs/generated/memory-calibration-evidence.json",',
    '        include_str!("../../../docs/generated/memory-calibration-evidence.json"),',
    "    ),",
    "];",
    "",
  ].join("\n"));
  await git("init", "--quiet", "-b", "story");
  await git("-c", "user.email=t@t", "-c", "user.name=t", "add", ".");
  await git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m", "seed");
  const workDir = await mkdtemp(path.join(tmpdir(), "catalog-work-"));
  for (const sub of ["logs", "captures", "raw"]) await mkdir(path.join(workDir, sub));
  return { root, workDir, git };
}

function stubContext({ root, workDir }, overrides = {}) {
  return {
    root, workDir, inferencePin: REVISION, campaignDir: "docs/calibration/sc-stub",
    campaignPrefix: "docs/calibration/sc-stub", state: { commits: [], halt: null },
    args: { adapter: JSON.stringify([process.execPath, "unused-adapter.mjs"]), inferenceRepo: root, commit: true, campaign: "sc-stub" },
    ...overrides,
  };
}

test("one anchor lands as one commit carrying the evidence, the packaged-source entry, and the regenerated files", async () => {
  const checkout = await stubCheckout();
  const tierRoot = await mkdtemp(path.join(tmpdir(), "catalog-tier-"));
  await writeFile(path.join(tierRoot, "w.safetensors"), "weights");
  process.env.GIT_AUTHOR_NAME = process.env.GIT_COMMITTER_NAME = "t";
  process.env.GIT_AUTHOR_EMAIL = process.env.GIT_COMMITTER_EMAIL = "t@t";
  const row = { key: "z_image_turbo:q4:mlx", physical: false, tierRoot, env: { SCENEWORKS_Z_IMAGE_ROOT: tierRoot } };
  const context = stubContext(checkout);
  const result = await measureAnchor(row, context);
  assert.equal(result.status, "committed", result.reason);
  assert.equal(context.state.commits.length, 1);
  const { stdout: status } = await checkout.git("status", "--porcelain");
  assert.equal(status, "", "the tree is clean again for the next capture");
  const { stdout: shown } = await checkout.git("show", "--stat", "--format=%s", "HEAD");
  assert.match(shown, /chore\(sc-stub\): measure z_image_turbo:q4:mlx memory anchor/);
  for (const file of ["docs/calibration/sc-stub/z-image-turbo-q4-mlx-evidence.json", PACKAGED_SOURCES_PATH, "config/memory-anchors.json", "docs/generated/memory-matrix.json"]) {
    assert.ok(shown.includes(file), `${file} is in the commit`);
  }
  assert.ok(shown.includes("docs/generated/memory-matrix.md"), "the regenerated matrix markdown is in the commit too");
  const rust = await readFile(path.join(checkout.root, PACKAGED_SOURCES_PATH), "utf8");
  assert.ok(rust.includes('"docs/calibration/sc-stub/z-image-turbo-q4-mlx-evidence.json"'));
  const store = JSON.parse(await readFile(path.join(checkout.root, "config/memory-anchors.json"), "utf8"));
  assert.equal(store.anchors[0].source.loaderClosureDigest, "a".repeat(64), "the seeded placeholder was replaced by the stamp before the commit");
  assert.notEqual(store.anchors[0].source.loaderClosureDigest, SEED_DIGEST);
  const evidence = JSON.parse(await readFile(path.join(checkout.root, "docs/calibration/sc-stub/z-image-turbo-q4-mlx-evidence.json"), "utf8"));
  assert.equal(evidence.records[0].env, tierRoot, "the derived adapter environment reached the provider");
  assert.ok((await stat(result.log)).size > 0, "the per-anchor log was written");
});

test("a --no-commit run captures and checks, then stops with a clean tree so the next anchor can still be captured", async () => {
  const checkout = await stubCheckout();
  const row = { key: "z_image_turbo:q4:mlx", physical: false, env: {} };
  const context = stubContext(checkout, { args: { ...stubContext(checkout).args, commit: false } });
  const result = await measureAnchor(row, context);
  assert.equal(result.status, "captured", result.reason);
  assert.equal(context.state.commits.length, 0);
  assert.equal((await checkout.git("status", "--porcelain")).stdout, "", "nothing ingested, nothing derived: the tree is clean for the next anchor");
  assert.ok((await stat(path.join(checkout.workDir, "captures", "z-image-turbo-q4-mlx.json"))).size > 0, "the raw bundle is retained");
  await assert.rejects(stat(path.join(checkout.root, "docs/calibration/sc-stub")), "no evidence directory was created");
});

test("a failed derivation step rolls the tree back to HEAD and keeps the raw capture; a failed capture just logs", async () => {
  const checkout = await stubCheckout();
  const row = { key: "z_image_turbo:q4:mlx", physical: false, env: {} };
  process.env.STUB_FAIL_AT = "anchor-loader-closure.mjs";
  try {
    const context = stubContext(checkout);
    const result = await measureAnchor(row, context);
    assert.equal(result.status, "ingest_failed");
    assert.match(result.reason, /anchor-loader-closure.mjs stub failed/);
    assert.equal(context.state.commits.length, 0);
    assert.equal(context.state.halt, null);
    assert.equal((await checkout.git("status", "--porcelain")).stdout, "", "rollback left a clean tree");
    assert.ok((await stat(path.join(checkout.workDir, "captures", "z-image-turbo-q4-mlx.json"))).size > 0);
  } finally {
    delete process.env.STUB_FAIL_AT;
  }
  process.env.STUB_STRAY = "1";
  try {
    const context = stubContext(checkout);
    const result = await measureAnchor(row, context);
    assert.equal(result.status, "ingest_failed");
    assert.match(result.reason, /paths this run does not own/);
    assert.equal(context.state.commits.length, 0);
    // A write the script does not own cannot be undone blindly, so the run halts — but everything
    // the script DOES own was unstaged and restored to HEAD first, even after `git add` ran.
    assert.match(context.state.halt, /still dirty after rollback/);
    const { stdout: status } = await checkout.git("status", "--porcelain");
    assert.deepEqual(status.trim().split("\n"), ["?? docs/generated/stray.json"], "only the stray file the stub wrote remains");
  } finally {
    delete process.env.STUB_STRAY;
    await execFileAsync("rm", ["-f", path.join(checkout.root, "docs/generated/stray.json")]);
  }
  process.env.STUB_CAPTURE_FAILS = "1";
  try {
    const result = await measureAnchor(row, stubContext(checkout));
    assert.equal(result.status, "capture_failed");
    assert.equal(result.reason, "Error: stub capture refused");
    assert.equal((await checkout.git("status", "--porcelain")).stdout, "");
  } finally {
    delete process.env.STUB_CAPTURE_FAILS;
  }
});

test("a physical anchor's gitignored .log receipt is copied beside the evidence and force-added", async () => {
  const checkout = await stubCheckout();
  const row = { key: "z_image_turbo:q4:mlx", physical: true, env: {} };
  const context = stubContext(checkout);
  // The harness would write the receipt under <rawLogDir>/<campaignDir>/; emulate that.
  const rawLogDir = path.join(checkout.workDir, "raw", "z-image-turbo-q4-mlx");
  await mkdir(path.join(rawLogDir, "docs/calibration/sc-stub"), { recursive: true });
  await writeFile(path.join(rawLogDir, "docs/calibration/sc-stub/session-1.log"), "receipt");
  const result = await measureAnchor(row, context);
  assert.equal(result.status, "committed", result.reason);
  const { stdout: shown } = await checkout.git("show", "--stat", "--format=%s", "HEAD");
  assert.ok(shown.includes("docs/calibration/sc-stub/session-1.log"), "the *.log receipt is committed despite the ignore rule");
  assert.equal((await checkout.git("status", "--porcelain")).stdout, "");
});

test("seeding retries extraction only for the new-anchor refusal and never re-seeds an id twice", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "catalog-seed-"));
  await mkdir(path.join(root, "config"), { recursive: true });
  await writeFile(path.join(root, "config/memory-anchors.json"), JSON.stringify({ anchors: [] }));
  const calls = [];
  const exec = async () => {
    calls.push(1);
    if (calls.length === 1) throw Object.assign(new Error("x"), { stderr: "Error: anchor a:b has no recorded loader-closure digest in config/memory-anchors.json" });
  };
  await extractSeedingNewAnchors(exec, root, null);
  assert.equal(calls.length, 2);
  const store = JSON.parse(await readFile(path.join(root, "config/memory-anchors.json"), "utf8"));
  assert.deepEqual(store.anchors, [{ id: "a:b", source: { loaderClosureDigest: SEED_DIGEST } }]);
  const stuck = async () => { throw Object.assign(new Error("x"), { stderr: "Error: anchor a:b has no recorded loader-closure digest" }); };
  await assert.rejects(() => extractSeedingNewAnchors(stuck, root, null), (error) => /no recorded loader-closure digest/.test(error.stderr));
  assert.equal(JSON.parse(await readFile(path.join(root, "config/memory-anchors.json"), "utf8")).anchors.length, 1, "an id already seeded is not seeded again");
  const other = async () => { throw Object.assign(new Error("boom"), { stderr: "Error: boom" }); };
  await assert.rejects(() => extractSeedingNewAnchors(other, root, null), /boom/);
});

test("--model selects every tier of one model and refuses a model the plan does not declare", async () => {
  const args = parseArgs(["--backend", "mlx", "--list", "--model", "sdxl", "--model", "qwen_image"]);
  assert.deepEqual(args.models, ["sdxl", "qwen_image"]);
  const { rows } = await planRun({ ...args, anchors: null, campaign: "sc-catalog-test", hfCache: [] });
  assert.deepEqual([...new Set(rows.map((row) => row.modelId))].sort(), ["qwen_image", "sdxl"]);
  assert.equal(rows.filter((row) => row.modelId === "sdxl").length, 3, "all three sdxl tiers");
  await assert.rejects(
    planRun({ ...args, models: ["not_a_model"], anchors: null, campaign: "sc-catalog-test", hfCache: [] }),
    /--model not_a_model matches no plan anchor/,
  );
});
