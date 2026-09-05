import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, readFile, stat } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { parseRouteRegistryLaneTiers } from "./generate-memory-matrix.mjs";
import {
  ROOT,
  ADAPTER_LIB_PATH,
  MATRIX_PATH,
  PLAN_PATH,
  PACKAGED_SOURCES_PATH,
  PROVIDER_FAMILIES,
  providerFamily,
  anchorParts,
  anchorSlug,
  appendPackagedSource,
  capturedInCampaign,
  classifyAnchor,
  compiledInferencePin,
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
} from "./measure-memory-catalog.mjs";
import { LTX25_LANE_PROVIDERS } from "./memory-calibration-harness.mjs";

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
    // sc-22727: two catalog models over ONE engine provider id, each with its own rehost.
    { id: "flux2_klein_9b", downloads: [{ repo: "SceneWorks/flux2-klein-9b-mlx", revision: REVISION, variant: "q8", files: ["q8/*"] }] },
    { id: "flux2_klein_9b_kv", downloads: [{ repo: "SceneWorks/flux2-klein-9b-kv-mlx", revision: UPSTREAM, variant: "q8", files: ["q8/*"] }] },
    // The two Qwen edit ids are ONE checkpoint on ONE rehost, routed to one engine provider; only
    // the Lightning id additionally loads the pinned distill LoRA, which no download ships.
    { id: "qwen_image_edit_2511", downloads: [{ repo: "SceneWorks/qwen-image-edit-2511-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] }] },
    { id: "qwen_image_edit_2511_lightning", downloads: [{ repo: "SceneWorks/qwen-image-edit-2511-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] }] },
    // The FLUX.1 family (sc-22726). `pulid_flux_dev` ships the SAME flux1-dev backbone downloads as
    // `flux_dev`; its identity stack is fetched on first use and is not a manifest download at all.
    { id: "flux_dev", downloads: [{ repo: "SceneWorks/flux1-dev-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] }] },
    { id: "pulid_flux_dev", downloads: [{ repo: "SceneWorks/flux1-dev-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] }] },
    // The SD3.5 family (sc-22730). Three DISTINCT engine providers, each with its own tiered
    // rehost — none is an alias for another's backbone, so each resolves its own env family.
    { id: "sd3_5_large", downloads: [{ repo: "SceneWorks/sd3.5-large-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] }] },
    { id: "sd3_5_large_turbo", downloads: [{ repo: "SceneWorks/sd3.5-large-turbo-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] }] },
    { id: "sd3_5_medium", downloads: [{ repo: "SceneWorks/sd3.5-medium-mlx", revision: UPSTREAM, variant: "q8", files: ["q8/*"] }] },
    // sc-22731. SANA is the one family whose two lanes load DIFFERENT repositories: the MLX lane
    // opens the per-tier SceneWorks turnkey, the Candle lane the upstream dense diffusers snapshot
    // at its root. Chroma1 is the ordinary shape — one per-tier turnkey serving both lanes — and is
    // here so the lane override is proven to be a per-family override and not a global change.
    {
      id: "sana_1600m",
      downloads: [
        { repo: "SceneWorks/Sana_1600M_1024px_mlx", revision: REVISION, variant: "q4", files: ["q4/*"], platforms: ["macos"] },
        { repo: "Efficient-Large-Model/Sana_1600M_1024px_diffusers", revision: UPSTREAM, variant: "bf16", files: [], platforms: ["windows", "linux"] },
      ],
    },
    { id: "chroma1_hd", downloads: [{ repo: "SceneWorks/chroma1-hd-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] }] },
    // The turnkey still family (sc-22732). Kolors and the two Lens models each ship every tier from
    // one repository; the two Ideogram models ship q4/q8 from the packed turnkey and bf16 from a
    // SECOND repository at a SECOND revision, which is the case the family's `tiers` override
    // exists for. Both Ideogram members carry both downloads, exactly as the manifest does.
    { id: "kolors", downloads: [{ repo: "SceneWorks/kolors-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] }] },
    { id: "lens", downloads: [{ repo: "SceneWorks/lens-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] }] },
    { id: "lens_turbo", downloads: [{ repo: "SceneWorks/lens-turbo-mlx", revision: UPSTREAM, variant: "q4", files: ["q4/*"] }] },
    {
      id: "ideogram_4",
      downloads: [
        { repo: "SceneWorks/ideogram-4-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] },
        { repo: "SceneWorks/ideogram-4", revision: UPSTREAM, variant: "bf16", files: ["bf16/*"] },
      ],
    },
    {
      id: "ideogram_4_turbo",
      downloads: [
        { repo: "SceneWorks/ideogram-4-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] },
        { repo: "SceneWorks/ideogram-4", revision: UPSTREAM, variant: "bf16", files: ["bf16/*"] },
      ],
    },
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

test("each SD3.5 member binds its OWN artifact family on both lanes", async () => {
  // The three members share an engine crate but NOT an artifact: serving a Medium plan from Large
  // weights would re-label Large's peaks as Medium's. A member whose snapshot is absent must be
  // `weights_missing` — never satisfied by a sibling's staged root.
  const hub = await fakeHub([
    ["SceneWorks/sd3.5-large-mlx", REVISION, "q4"],
    ["SceneWorks/sd3.5-large-turbo-mlx", REVISION, "q4"],
  ]);
  for (const backend of ["mlx", "candle"]) {
    const context = { models: fakeModels(), backend, hubs: [hub], current: new Map(), captured: new Map() };
    const large = await classifyAnchor(`sd3_5_large:q4:${backend}`, { provider: "sd3_5_large" }, context);
    assert.equal(large.status, "runnable", `${backend}: ${large.reason}`);
    assert.deepEqual(large.env, {
      SCENEWORKS_SD3_5_LARGE_REPOSITORY: "SceneWorks/sd3.5-large-mlx",
      SCENEWORKS_SD3_5_LARGE_REVISION: REVISION,
      SCENEWORKS_SD3_5_LARGE_ROOT: snapshotPath(hub, "SceneWorks/sd3.5-large-mlx", REVISION, "q4"),
    });

    const turbo = await classifyAnchor(`sd3_5_large_turbo:q4:${backend}`, { provider: "sd3_5_large_turbo" }, context);
    assert.equal(turbo.status, "runnable", `${backend}: ${turbo.reason}`);
    assert.deepEqual(turbo.env, {
      SCENEWORKS_SD3_5_LARGE_TURBO_REPOSITORY: "SceneWorks/sd3.5-large-turbo-mlx",
      SCENEWORKS_SD3_5_LARGE_TURBO_REVISION: REVISION,
      SCENEWORKS_SD3_5_LARGE_TURBO_ROOT: snapshotPath(hub, "SceneWorks/sd3.5-large-turbo-mlx", REVISION, "q4"),
    });
    // Turbo is its own env family, so a Large root can never stand in for it.
    assert.notDeepEqual(Object.keys(turbo.env), Object.keys(large.env));

    // Medium's snapshot is NOT staged on this fake hub: it must report missing under its OWN
    // repository, not fall back to a sibling's.
    const medium = await classifyAnchor(`sd3_5_medium:q8:${backend}`, { provider: "sd3_5_medium" }, context);
    assert.equal(medium.status, "weights_missing", `${backend}: ${medium.reason}`);
    assert.match(medium.reason, /SceneWorks\/sd3\.5-medium-mlx@[0-9a-f]{8}\/q8/);
  }
});

test("the flux2-klein family: two catalog models on one provider id resolve their OWN artifacts", async () => {
  const hub = await fakeHub([
    ["SceneWorks/flux2-klein-9b-mlx", REVISION, "q8"],
    ["SceneWorks/flux2-klein-9b-kv-mlx", UPSTREAM, "q8"],
  ]);
  for (const backend of ["mlx", "candle"]) {
    const context = { models: fakeModels(), backend, hubs: [hub], current: new Map(), captured: new Map() };
    const base = await classifyAnchor(`flux2_klein_9b:q8:${backend}`, { provider: "flux2_klein_9b" }, context);
    assert.equal(base.status, "runnable", `${backend}: ${base.reason}`);
    assert.deepEqual(base.env, {
      SCENEWORKS_FLUX2_KLEIN_REPOSITORY: "SceneWorks/flux2-klein-9b-mlx",
      SCENEWORKS_FLUX2_KLEIN_REVISION: REVISION,
      SCENEWORKS_FLUX2_KLEIN_ROOT: snapshotPath(hub, "SceneWorks/flux2-klein-9b-mlx", REVISION, "q8"),
    });
    // The KV model shares `provider` with the row above and MUST NOT share its artifact family:
    // a KV plan served from the base rehost would re-label the base checkpoint's peaks.
    const kv = await classifyAnchor(`flux2_klein_9b_kv:q8:${backend}`, { provider: "flux2_klein_9b" }, context);
    assert.equal(kv.status, "runnable", `${backend}: ${kv.reason}`);
    assert.equal(kv.provider, "flux2_klein_9b", "both models load through one engine provider id");
    assert.deepEqual(kv.env, {
      SCENEWORKS_FLUX2_KLEIN_KV_REPOSITORY: "SceneWorks/flux2-klein-9b-kv-mlx",
      SCENEWORKS_FLUX2_KLEIN_KV_REVISION: UPSTREAM,
      SCENEWORKS_FLUX2_KLEIN_KV_ROOT: snapshotPath(hub, "SceneWorks/flux2-klein-9b-kv-mlx", UPSTREAM, "q8"),
    });
    // And the per-model override never leaks back into the shared row.
    assert.equal(providerFamily("flux2_klein_9b", "flux2_klein_9b").env, "FLUX2_KLEIN");
    assert.equal(providerFamily("flux2_klein_9b", "flux2_klein_9b_kv").env, "FLUX2_KLEIN_KV");
    assert.equal(providerFamily("flux2_klein_9b", "flux2_klein_9b_kv").variants, undefined);
    const missing = await classifyAnchor(`flux2_klein_9b_kv:q4:${backend}`, { provider: "flux2_klein_9b" }, context);
    assert.equal(missing.status, "weights_missing");
    assert.match(missing.reason, /flux2-klein-9b-kv-mlx@.*\/q4 on this host/);
  }
});

// sc-22728: the Qwen edit family is two catalog ids on ONE engine provider and ONE rehost, and the
// Lightning id needs a SECOND root the manifest does not ship (the pinned distill LoRA). Both roots
// must be derived, on both lanes, and neither id may be served without its own.
test("the qwen edit family derives the tier root on both lanes, and Lightning also derives its pinned distill LoRA", async () => {
  const lora = PROVIDER_FAMILIES.qwen_image_edit.sideArtifact.qwen_image_edit_2511_lightning;
  const hub = await fakeHub([
    ["SceneWorks/qwen-image-edit-2511-mlx", REVISION, "q4"],
    [lora.repo, lora.revision],
  ]);
  for (const backend of ["mlx", "candle"]) {
    const context = { models: fakeModels(), backend, hubs: [hub], current: new Map(), captured: new Map() };
    const planned = { provider: "qwen_image_edit", mode: "edit_image" };
    const base = await classifyAnchor(`qwen_image_edit_2511:q4:${backend}`, planned, context);
    assert.equal(base.status, "runnable", `${backend}: ${base.reason}`);
    assert.equal(base.physical, false, "only the qwen_image MLX arm emits a sourceCapture");
    assert.deepEqual(base.env, {
      SCENEWORKS_QWEN_IMAGE_EDIT_REPOSITORY: "SceneWorks/qwen-image-edit-2511-mlx",
      SCENEWORKS_QWEN_IMAGE_EDIT_REVISION: REVISION,
      SCENEWORKS_QWEN_IMAGE_EDIT_ROOT: snapshotPath(hub, "SceneWorks/qwen-image-edit-2511-mlx", REVISION, "q4"),
    }, "the production id loads the tier root and nothing else");

    const lightning = await classifyAnchor(`qwen_image_edit_2511_lightning:q4:${backend}`, planned, context);
    assert.equal(lightning.status, "runnable", `${backend}: ${lightning.reason}`);
    assert.equal(lightning.env.SCENEWORKS_QWEN_IMAGE_EDIT_ROOT, base.env.SCENEWORKS_QWEN_IMAGE_EDIT_ROOT, "one shared checkpoint");
    assert.equal(lightning.env.SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_REPOSITORY, lora.repo);
    assert.equal(lightning.env.SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_REVISION, lora.revision);
    assert.equal(
      lightning.env.SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_ROOT,
      snapshotPath(hub, lora.repo, lora.revision),
      "the distill LoRA snapshot, at the revision the engine pins",
    );
  }
  // Without the distill snapshot the Lightning cell is `weights_missing` — measurable, but not
  // runnable on this host — while the production id stays runnable off the same tier root.
  const tierOnly = await fakeHub([["SceneWorks/qwen-image-edit-2511-mlx", REVISION, "q4"]]);
  const context = { models: fakeModels(), backend: "mlx", hubs: [tierOnly], current: new Map(), captured: new Map() };
  const planned = { provider: "qwen_image_edit", mode: "edit_image" };
  assert.equal((await classifyAnchor("qwen_image_edit_2511:q4:mlx", planned, context)).status, "runnable");
  const missing = await classifyAnchor("qwen_image_edit_2511_lightning:q4:mlx", planned, context);
  assert.equal(missing.status, "weights_missing");
  assert.match(missing.reason, /Qwen-Image-Edit-2511-Lightning@/);
});

// The catalog's `physical` flag and the harness's receipt predicate are two spellings of one rule,
// and sc-22728 added a family that LOOKS like Qwen and must not inherit it. Bind them: the harness
// requires a physical source session for `qwen_image` alone, so exactly that family may carry the
// flag — passing `--raw-log-dir` for any other arm makes the harness refuse the render outright.
test("only the family the harness demands a physical receipt for is marked physical", async () => {
  const harness = await readFile(path.join(ROOT, "scripts/memory-calibration-harness.mjs"), "utf8");
  const predicate = /function requiresPhysicalMlxProvenanceForCurrency\(record\) \{([\s\S]*?)\n\}/.exec(harness);
  assert.ok(predicate, "the harness still declares requiresPhysicalMlxProvenanceForCurrency");
  const named = [...predicate[1].matchAll(/record\.target\.modelId === "([a-z0-9_]+)"/g)].map((match) => match[1]);
  assert.deepEqual(named, ["qwen_image"], "the harness scopes the receipt to exactly one model id");
  assert.deepEqual(
    Object.entries(PROVIDER_FAMILIES).filter(([, family]) => family.physical).map(([id]) => id),
    named,
    "a family marked physical that the harness does not demand a receipt for would make every capture of it fail",
  );
});

// The Lightning distill LoRA is the one artifact in this table the MANIFEST does not ship, so the
// repository, revision and file name below are a COPY of the worker's own pinned constants. Nothing
// derives one from the other, and a drift would send the capture at a LoRA the Candle engine rejects
// by exact path after a 28-57 GB load. This is that binding, read from both lanes' worker source.
test("the pinned Lightning distill LoRA matches the worker constants on both lanes", async () => {
  const lora = PROVIDER_FAMILIES.qwen_image_edit.sideArtifact.qwen_image_edit_2511_lightning;
  for (const file of [
    "crates/sceneworks-worker/src/image_jobs/qwen.rs",
    "crates/sceneworks-worker/src/image_jobs/qwen_edit_candle.rs",
  ]) {
    const source = await readFile(path.join(ROOT, file), "utf8");
    for (const [label, value] of Object.entries(lora)) {
      if (label === "env") continue;
      assert.ok(
        source.includes(`"${value}"`),
        `${file} does not name the ${label} ${JSON.stringify(value)} this table pins`,
      );
    }
  }
  // And the adapters resolve the file name from the shared protocol constant rather than a literal.
  const protocolSource = await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/lib.rs"), "utf8");
  // rustfmt may wrap a long const onto its own line, so match the declaration and its literal
  // rather than one exact source line.
  for (const [name, value] of [
    ["QWEN_EDIT_LIGHTNING_REPOSITORY", lora.repo],
    ["QWEN_EDIT_LIGHTNING_FILE", lora.file],
  ]) {
    const declaration = new RegExp(`pub const ${name}: &str =\\s*"${value.replaceAll(".", "\\.")}"`);
    assert.match(protocolSource, declaration);
  }
});

// The env FAMILY names in `PROVIDER_FAMILIES` are the other half of the same hand-duplication: this
// table derives `SCENEWORKS_<env>_{REPOSITORY,REVISION,ROOT}` and exports them into the adapter, and
// each adapter arm reads those exact names back as its own source literals. Nothing bound the two,
// on ANY family — renaming one side (or adding a family whose arm never reads it) leaves every test
// in both files green and fails only mid-campaign, after the weights are staged. This is that
// binding, per family, per declared arm, including the `upstream` and `sideArtifact` sub-families.
test("every family's derived env names are read back by each arm it declares", async () => {
  const lanes = new Map();
  for (const backend of ["mlx", "candle"]) {
    lanes.set(
      backend,
      await readFile(path.join(ROOT, `crates/sceneworks-memory-adapter/src/bin/${backend}.rs`), "utf8"),
    );
  }
  let checked = 0;
  for (const [id, family] of Object.entries(PROVIDER_FAMILIES)) {
    // The LTX-2.5 families carry no env family at all: the harness binds their snapshot root
    // directly (`--ltx25-snapshot-root`), which `classifyAnchor` asserts by leaving `row.env` empty.
    const envFamilies = [
      family.env,
      family.upstream?.env,
      // sc-22727: a per-modelId variant is a derived env family too — the KV klein row exports
      // `SCENEWORKS_FLUX2_KLEIN_KV_*` and each declared arm must read those names back.
      ...Object.values(family.variants ?? {}).map((variant) => variant.env),
      ...Object.values(family.sideArtifact ?? {}).map((side) => side.env),
    ].filter(Boolean);
    if (envFamilies.length === 0) {
      assert.ok(family.ltx25, `${id} declares neither an env family nor the LTX-2.5 harness binding`);
      continue;
    }
    for (const arm of family.arms) {
      const source = lanes.get(arm);
      assert.ok(source, `${id} declares an unknown adapter arm ${arm}`);
      // Per LANE (sc-22731): a family may bind a DIFFERENT artifact on each arm, and then only that
      // arm's env family is the one that arm reads. The two SANA routes are the case — MLX opens
      // the packed `SceneWorks/*_mlx` turnkey through `SCENEWORKS_SANA_*`, Candle the upstream
      // dense diffusers snapshot through `SCENEWORKS_SANA_DENSE_*` — so asserting the shared
      // `family.env` against candle.rs would demand an env that binary must never read.
      const laneEnvFamilies = family.lanes?.[arm]
        ? [family.lanes[arm].env].filter(Boolean)
        : envFamilies;
      for (const env of laneEnvFamilies) {
        for (const suffix of ["REPOSITORY", "REVISION", "ROOT"]) {
          const name = `SCENEWORKS_${env}_${suffix}`;
          assert.ok(
            source.includes(`"${name}"`),
            `${arm}.rs never reads ${name}, which ${id} exports into it`,
          );
          checked += 1;
        }
      }
    }
  }
  assert.ok(checked >= 45, `expected the whole table to be covered, checked ${checked} names`);
});

test("the flux.1 family: the two base providers bind their own artifacts, and PuLID rides the dev backbone plus a staged identity bundle", async () => {
  const hub = await fakeHub([["SceneWorks/flux1-dev-mlx", REVISION, "q4"]]);
  const previous = process.env.SCENEWORKS_PULID_WEIGHTS;
  try {
    for (const backend of ["mlx", "candle"]) {
      // A FRESH bundle per lane: the staging steps below are cumulative, so a shared directory
      // would let the second lane skip straight past the two `weights_missing` cases.
      const bundle = await mkdtemp(path.join(tmpdir(), "catalog-pulid-"));
      const context = { models: fakeModels(), backend, hubs: [hub], current: new Map(), captured: new Map() };
      const dev = await classifyAnchor(`flux_dev:q4:${backend}`, { provider: "flux1_dev" }, context);
      assert.equal(dev.status, "runnable", `${backend}: ${dev.reason}`);
      assert.deepEqual(dev.env, {
        SCENEWORKS_FLUX1_DEV_REPOSITORY: "SceneWorks/flux1-dev-mlx",
        SCENEWORKS_FLUX1_DEV_REVISION: REVISION,
        SCENEWORKS_FLUX1_DEV_ROOT: snapshotPath(hub, "SceneWorks/flux1-dev-mlx", REVISION, "q4"),
      });

      // The identity stack is not a manifest download on either lane, so the anchor binds the
      // operator's staged bundle through the same env both worker lanes honour. Unset or
      // incomplete is `weights_missing` — never a "runnable" cell that cannot actually run.
      delete process.env.SCENEWORKS_PULID_WEIGHTS;
      const unstaged = await classifyAnchor(`pulid_flux_dev:q4:${backend}`, { provider: "pulid_flux" }, context);
      assert.equal(unstaged.status, "weights_missing");
      assert.match(unstaged.reason, /SCENEWORKS_PULID_WEIGHTS is unset/);

      process.env.SCENEWORKS_PULID_WEIGHTS = bundle;
      const partial = await classifyAnchor(`pulid_flux_dev:q4:${backend}`, { provider: "pulid_flux" }, context);
      assert.equal(partial.status, "weights_missing");
      assert.match(partial.reason, /is missing pulid_flux_v0\.9\.1\.safetensors/);

      for (const file of PROVIDER_FAMILIES.pulid_flux.bundle.files) {
        await writeFile(path.join(bundle, file), "weights");
      }
      const pulid = await classifyAnchor(`pulid_flux_dev:q4:${backend}`, { provider: "pulid_flux" }, context);
      assert.equal(pulid.status, "runnable", `${backend}: ${pulid.reason}`);
      assert.equal(pulid.physical, false);
      assert.deepEqual(pulid.env, {
        // The PuLID backbone IS the FLUX.1-dev artifact, so it shares that env family — the way
        // z_image_edit shares the Turbo family — and its tier root resolves through the
        // `pulid_flux_dev` MANIFEST entry, not through `flux_dev`'s.
        SCENEWORKS_FLUX1_DEV_REPOSITORY: "SceneWorks/flux1-dev-mlx",
        SCENEWORKS_FLUX1_DEV_REVISION: REVISION,
        SCENEWORKS_FLUX1_DEV_ROOT: snapshotPath(hub, "SceneWorks/flux1-dev-mlx", REVISION, "q4"),
        SCENEWORKS_PULID_WEIGHTS: bundle,
      });
    }
  } finally {
    if (previous === undefined) delete process.env.SCENEWORKS_PULID_WEIGHTS;
    else process.env.SCENEWORKS_PULID_WEIGHTS = previous;
  }
});

test("the sana family binds a different artifact per lane, and chroma1 binds one turnkey on both", async () => {
  const hub = await fakeHub([
    ["SceneWorks/Sana_1600M_1024px_mlx", REVISION, "q4"],
    ["Efficient-Large-Model/Sana_1600M_1024px_diffusers", UPSTREAM],
    ["SceneWorks/chroma1-hd-mlx", REVISION, "q4"],
  ]);
  const context = (backend) => ({ models: fakeModels(), backend, hubs: [hub], current: new Map(), captured: new Map() });

  // MLX: the packed turnkey, descended into the planned tier.
  const mlx = await classifyAnchor("sana_1600m:q4:mlx", { provider: "sana_1600m" }, context("mlx"));
  assert.equal(mlx.status, "runnable", mlx.reason);
  assert.deepEqual(mlx.env, {
    SCENEWORKS_SANA_REPOSITORY: "SceneWorks/Sana_1600M_1024px_mlx",
    SCENEWORKS_SANA_REVISION: REVISION,
    SCENEWORKS_SANA_ROOT: snapshotPath(hub, "SceneWorks/Sana_1600M_1024px_mlx", REVISION, "q4"),
  });

  // Candle: the upstream dense snapshot, at its ROOT — no tier component, because that is what
  // `resolve_weights_dir` hands `candle-gen-sana` and what its `validate_immutable_root` requires.
  const candle = await classifyAnchor("sana_1600m:bf16:candle", { provider: "sana_1600m" }, context("candle"));
  assert.equal(candle.status, "runnable", candle.reason);
  assert.deepEqual(candle.env, {
    SCENEWORKS_SANA_DENSE_REPOSITORY: "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    SCENEWORKS_SANA_DENSE_REVISION: UPSTREAM,
    SCENEWORKS_SANA_DENSE_ROOT: snapshotPath(hub, "Efficient-Large-Model/Sana_1600M_1024px_diffusers", UPSTREAM),
  });
  assert.equal(candle.roots.find((root) => root.label === "snapshot root").path, candle.env.SCENEWORKS_SANA_DENSE_ROOT);

  // ...and the override is per family: Chroma1 derives the same per-tier root on both lanes.
  for (const backend of ["mlx", "candle"]) {
    const chroma = await classifyAnchor(`chroma1_hd:q4:${backend}`, { provider: "chroma1_hd" }, context(backend));
    assert.equal(chroma.status, "runnable", `${backend}: ${chroma.reason}`);
    assert.deepEqual(chroma.env, {
      SCENEWORKS_CHROMA1_HD_REPOSITORY: "SceneWorks/chroma1-hd-mlx",
      SCENEWORKS_CHROMA1_HD_REVISION: REVISION,
      SCENEWORKS_CHROMA1_HD_ROOT: snapshotPath(hub, "SceneWorks/chroma1-hd-mlx", REVISION, "q4"),
    });
  }
});

// sc-22731 review guard: the repository literal each family names is duplicated in the adapter's
// lib.rs, and the env family label is duplicated in whichever adapter binary serves that lane.
// Nothing bound the three copies. This parses the Rust and asserts they agree, so editing one alone
// reds here instead of on a capture host with the weights already staged.
test("every sana/chroma env family and repository is the one the adapter binaries actually read", async () => {
  const lib = await readFile(path.join(ROOT, ADAPTER_LIB_PATH), "utf8");
  const repositories = new Map(
    [...lib.matchAll(/pub const ([A-Z0-9_]+_REPOSITORY): &str =\s*"([^"]+)";/g)].map((match) => [match[2], match[1]]),
  );
  const binaries = {
    mlx: await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/bin/mlx.rs"), "utf8"),
    candle: await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/bin/candle.rs"), "utf8"),
  };
  const family = (id) => PROVIDER_FAMILIES[id];
  const expected = new Set(
    ["sana_1600m", "sana_sprint_1600m", "chroma1_hd", "chroma1_base", "chroma1_flash"].flatMap((id) =>
      family(id).arms.map((backend) => `${id}:${backend}`),
    ),
  );
  const checked = new Set();
  for (const id of ["sana_1600m", "sana_sprint_1600m", "chroma1_hd", "chroma1_base", "chroma1_flash"]) {
    for (const backend of family(id).arms) {
      const artifact = family(id).lanes?.[backend] ?? family(id);
      assert.ok(
        repositories.has(artifact.repo),
        `${id}:${backend}: ${artifact.repo} is not a *_REPOSITORY const in ${ADAPTER_LIB_PATH}`,
      );
      for (const suffix of ["REPOSITORY", "REVISION", "ROOT"]) {
        const name = `SCENEWORKS_${artifact.env}_${suffix}`;
        assert.ok(
          binaries[backend].includes(`"${name}"`),
          `${id}:${backend}: the ${backend} adapter never reads ${name}`,
        );
      }
      checked.add(`${id}:${backend}`);
    }
  }
  // A SET, not a count (sc-22731 review): derived from each family's own declared `arms`, so a
  // route that loses a lane — or gains one — names ITSELF here instead of reporting "nine, not ten".
  assert.deepEqual([...checked].sort(), [...expected].sort());
  assert.ok(expected.size > 0, "the sweep must actually visit the two families");
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
  // sc-22727 gave `flux2_dev` a Candle arm, so the no-arm probe moved to a provider that still
  // has none: `sdxl` is declared `arms: ["mlx"]`.
  const sdxl = await classifyAnchor("sdxl:q4:candle", { provider: "sdxl" }, base);
  assert.equal(sdxl.status, "no_adapter_arm");
  assert.match(sdxl.reason, /candle adapter implements no provider arm for sdxl/);
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

// sc-22726 review: the five PuLID bundle file names were hand-duplicated between this runner and
// the adapter's lib.rs with nothing binding them. This parses the Rust constants — the per-file
// `pub const PULID_*_FILE: &str = "…";` declarations and the `PULID_IDENTITY_BUNDLE_FILES` array
// that orders them — and asserts the runner's list is that list, in that order, under that env var.
test("PROVIDER_FAMILIES.pulid_flux.bundle is the adapter's PULID_IDENTITY_BUNDLE_FILES, in order", async () => {
  const lib = await readFile(path.join(ROOT, ADAPTER_LIB_PATH), "utf8");
  const consts = new Map();
  for (const match of lib.matchAll(/pub const (PULID_[A-Z_]+_FILE): &str = "([^"]+)";/g)) {
    consts.set(match[1], match[2]);
  }
  const array = lib.match(/pub const PULID_IDENTITY_BUNDLE_FILES: \[&str; (\d+)\] = \[([\s\S]*?)\];/);
  assert.ok(array, "lib.rs declares PULID_IDENTITY_BUNDLE_FILES");
  const names = array[2].split(",").map((name) => name.trim()).filter(Boolean);
  assert.equal(names.length, Number(array[1]), "the array literal carries every declared entry");
  const files = names.map((name) => {
    assert.ok(consts.has(name), `${name} resolves to a PULID_*_FILE const`);
    return consts.get(name);
  });
  assert.deepEqual(PROVIDER_FAMILIES.pulid_flux.bundle.files, files);
  const env = lib.match(/pub const PULID_IDENTITY_BUNDLE_ENV: &str = "([^"]+)";/);
  assert.ok(env, "lib.rs declares PULID_IDENTITY_BUNDLE_ENV");
  assert.equal(PROVIDER_FAMILIES.pulid_flux.bundle.env, env[1]);
});

// sc-22726 review: the adapter canonicalizes the bundle path from the HARNESS's cwd, so a relative
// export must already be absolute by the time this runner probes or forwards it.
test("a relative SCENEWORKS_PULID_WEIGHTS is resolved to an absolute path before it is probed or forwarded", async () => {
  const hub = await fakeHub([["SceneWorks/flux1-dev-mlx", REVISION, "q4"]]);
  const previous = process.env.SCENEWORKS_PULID_WEIGHTS;
  const cwd = process.cwd();
  try {
    const bundle = await mkdtemp(path.join(tmpdir(), "catalog-pulid-relative-"));
    for (const file of PROVIDER_FAMILIES.pulid_flux.bundle.files) {
      await writeFile(path.join(bundle, file), "weights");
    }
    process.chdir(path.dirname(bundle));
    process.env.SCENEWORKS_PULID_WEIGHTS = path.basename(bundle);
    // `process.cwd()` is the real path (macOS `/var` -> `/private/var`), so the expectation is
    // built from it rather than from the tmpdir spelling.
    const expected = path.join(process.cwd(), path.basename(bundle));
    const context = { models: fakeModels(), backend: "mlx", hubs: [hub], current: new Map(), captured: new Map() };
    const pulid = await classifyAnchor("pulid_flux_dev:q4:mlx", { provider: "pulid_flux" }, context);
    assert.equal(pulid.status, "runnable", pulid.reason);
    assert.ok(path.isAbsolute(pulid.env.SCENEWORKS_PULID_WEIGHTS));
    assert.equal(pulid.env.SCENEWORKS_PULID_WEIGHTS, expected);
    assert.equal(pulid.roots.find((root) => root.label === "pulid bundle").path, expected);
  } finally {
    process.chdir(cwd);
    if (previous === undefined) delete process.env.SCENEWORKS_PULID_WEIGHTS;
    else process.env.SCENEWORKS_PULID_WEIGHTS = previous;
  }
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
      if (row.status === "no_adapter_arm") {
        assert.equal(PROVIDER_FAMILIES[row.provider]?.arms.includes(backend) ?? false, false);
      } else if (!["harness_unsupported", "lane_undeclared", "provider_undeclared"].includes(row.status)) {
        // A served provider must resolve a manifest download, or the classification could not name a root.
        tierDownload(models, row.modelId, providerFamily(row.provider, row.modelId).repo, row.tier);
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
 *   and per-family engine tables the worker dispatches with).
 *
 * **The tier axis is per LANE (sc-22731).** It used to be per model: every tier any download ships
 * was claimed on every routed lane. That over-claimed cells no lane can ever load — `sana_1600m`
 * routes on Candle, but its q4/q8 tiers are `platforms: ["macos"]` MLX turnkeys and the Candle arm
 * loads the upstream dense diffusers snapshot instead ("there is no packed q4/q8 tier off-Mac; the
 * worker resolves this repo's snapshot ROOT, never a tier subdir", and
 * `candle-gen-sana`'s `validate_load_spec` refuses any `quantize`, while
 * `crates/sceneworks-worker/src/memory_route_registry.rs` routes candle `sana_1600m` /
 * `sana_sprint_1600m` as `BF16_ONLY`). A cell with no artifact on the lane is not measurement work
 * that has been skipped; it is an unrouted (lane, tier), which is epic 22723 E1's ONE exemption.
 * A "structurally N/A" matrix cell is still not an exemption.
 *
 * The lane test is the same three-way rule `tiersFor` applies (sc-22731 review), and its registry
 * half is IMPORTED from the generator rather than respelled. The first version tested only
 * `download.platforms`, and that was wrong in both directions: it claimed `bernini`/`bernini_image`
 * had no candle tiers at all, although their only off-Mac download is one UNTIERED
 * `SceneWorks/bernini` bundle whose tier subdirs live inside it and whose Candle route rule declares
 * `BF16_Q4_Q8`.
 *
 * The denominator is deliberately NOT read off the matrix's published axes. It is the broader,
 * manifest-derived claim, and the difference is load-bearing in one direction: `flux_dev` and
 * `flux_schnell` ship an ungated bf16 download while their candle blocks declare only
 * `vramGbByTier: {q4, q8}`, so the matrix publishes no `flux_dev:bf16:candle` cell — and the
 * burndown must still ask about it, because "the matrix under-declares a lane that ships a tier" is
 * exactly the false green epic 22723 exists to catch. Narrowing this to the matrix would delete the
 * question instead of answering it.
 */
const LANE_PLATFORM = Object.freeze({ mlx: "macos", candle: "linux" });

/** Whether a manifest download is one this lane's host would ever fetch. */
function downloadServesLane(download, backend) {
  return !download.platforms || download.platforms.includes(LANE_PLATFORM[backend]);
}

let routeLaneTiersPromise;
/** The registry's per-lane tier FLOOR, from the generator's own parser — never a second spelling. */
function routeLaneTiers() {
  routeLaneTiersPromise ??= readFile(
    path.join(ROOT, "crates/sceneworks-worker/src/memory_route_registry.rs"),
    "utf8",
  ).then(parseRouteRegistryLaneTiers);
  return routeLaneTiersPromise;
}

async function computeShippedTieredCells() {
  const models = await readManifestModels();
  const matrix = JSON.parse(await readFile(path.join(ROOT, MATRIX_PATH), "utf8"));
  const routed = new Map(matrix.models.map((model) => [model.id, model.backends ?? []]));
  const laneTiers = await routeLaneTiers();
  const cells = [];
  for (const model of models) {
    const shipped = (model.downloads ?? []).filter(
      (download) => !download.coRequisite && ["q4", "q8", "bf16"].includes(download.variant),
    );
    if (shipped.length === 0) continue;
    for (const backend of routed.get(model.id) ?? []) {
      // An untiered download this lane's host fetches is a BUNDLE whose tiers live inside it, so it
      // serves every tier the model ships. `SceneWorks/bernini` is that repo; LTX-2.3's dense
      // co-requisite is another.
      const bundled = (model.downloads ?? []).some(
        (download) => typeof download.variant !== "string" && downloadServesLane(download, backend),
      );
      const floor = laneTiers.get(`${backend}:${model.id}`) ?? new Set();
      const tiers = [...new Set(
        shipped
          .filter(
            (download) =>
              bundled || downloadServesLane(download, backend) || floor.has(download.variant),
          )
          .map((download) => download.variant),
      )];
      for (const tier of tiers) cells.push({ modelId: model.id, tier, backend, key: `${model.id}:${tier}:${backend}` });
    }
  }
  return cells;
}

/**
 * Both derivations are pure over the checked-in manifest, matrix and plan, and both are asked for by
 * more than one case — `measurabilityGaps()` alone is two full `planRun`s with per-anchor filesystem
 * probes. Memoized as module-level promises so the whole file pays for each exactly once.
 */
let shippedTieredCellsPromise;
function shippedTieredCells() {
  shippedTieredCellsPromise ??= computeShippedTieredCells();
  return shippedTieredCellsPromise;
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

// sc-22731: the tier axis is per LANE, and the rule is the manifest's own `platforms` selection —
// not a hand-kept list of exempt cells. Asserted as a KEY SET on the one family that instantiates
// it today, plus the invariant that drives it, so a new platform-gated download is covered without
// editing this test and a cell can never be exempted by being forgotten.
test("a lane only claims the tiers whose downloads that lane's host would fetch", async () => {
  const models = await readManifestModels();
  const cells = await shippedTieredCells();
  const keys = new Set(cells.map((cell) => cell.key));

  // SANA: three MLX turnkey tiers (`platforms: ["macos"]`) and ONE dense Candle snapshot
  // (`platforms: ["windows", "linux"]`), so the two lanes claim different tier sets for one model.
  for (const modelId of ["sana_1600m", "sana_sprint_1600m"]) {
    assert.deepEqual(
      cells.filter((cell) => cell.modelId === modelId).map((cell) => cell.key).sort(),
      [`${modelId}:bf16:candle`, `${modelId}:bf16:mlx`, `${modelId}:q4:mlx`, `${modelId}:q8:mlx`].sort(),
      `${modelId}: the packed tiers are macOS turnkeys; Candle loads the upstream dense snapshot`,
    );
  }
  // Chroma's downloads carry no `platforms` key at all, so every tier is claimed on both lanes —
  // the case that proves the rule above is a filter and not a blanket narrowing.
  for (const modelId of ["chroma1_hd", "chroma1_base", "chroma1_flash"]) {
    assert.deepEqual(
      cells.filter((cell) => cell.modelId === modelId).map((cell) => cell.key).sort(),
      ["bf16", "q4", "q8"].flatMap((tier) => [`${modelId}:${tier}:candle`, `${modelId}:${tier}:mlx`]).sort(),
      `${modelId}: an ungated download ships to both lanes`,
    );
  }
  // And the invariant itself, over the whole catalog: a claimed cell is exactly one whose tier
  // something on this lane can open — a download the host fetches, an untiered bundle it fetches,
  // or a route rule that declares the tier — and an unclaimed (routed model, shipped tier) has none
  // of the three.
  const matrix = JSON.parse(await readFile(path.join(ROOT, MATRIX_PATH), "utf8"));
  const routed = new Map(matrix.models.map((model) => [model.id, model.backends ?? []]));
  const laneTiers = await routeLaneTiers();
  for (const model of models) {
    const shipped = (model.downloads ?? []).filter(
      (download) => !download.coRequisite && ["q4", "q8", "bf16"].includes(download.variant),
    );
    for (const backend of routed.get(model.id) ?? []) {
      const bundled = (model.downloads ?? []).some(
        (download) => typeof download.variant !== "string" && downloadServesLane(download, backend),
      );
      const floor = laneTiers.get(`${backend}:${model.id}`) ?? new Set();
      for (const tier of new Set(shipped.map((download) => download.variant))) {
        const serves =
          bundled ||
          floor.has(tier) ||
          shipped.some((download) => download.variant === tier && downloadServesLane(download, backend));
        assert.equal(
          keys.has(`${model.id}:${tier}:${backend}`),
          serves,
          `${model.id}:${tier}:${backend} is claimed iff something on that lane can open the tier`,
        );
      }
    }
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

// Epic 22723 E1/E2, story sc-22727: the whole FLUX.2 family, on every shipped tier of both lanes.
// `flux2_klein_9b_true_v2` is deliberately not here — it ships no tiered download (its manifest
// binds one converted single-file snapshot), so it is not a shipped TIERED cell at all.
test("the flux2 family is measurable on every shipped tier of every routed lane", async () => {
  const family = ["flux2_dev", "flux2_klein_9b", "flux2_klein_9b_kv"];
  const cells = (await shippedTieredCells()).filter((cell) => family.includes(cell.modelId));
  assert.ok(cells.length >= 3 * 3 * 2, "three models x three tiers x two lanes are all shipped and routed");
  const gaps = (await measurabilityGaps()).filter((gap) => family.includes(gap.modelId));
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

// sc-22728: the Qwen EDIT family — two catalog ids on one engine provider, one of which loads a
// built-in distill LoRA — on every shipped tier of every routed lane. Same shape claim as the two
// cases above: manifest tiers + matrix lanes + plan anchors + both closure declarations.
test("the qwen edit family is measurable on every shipped tier of every routed lane", async () => {
  const families = ["qwen_image_edit_2511", "qwen_image_edit_2511_lightning"];
  const cells = (await shippedTieredCells()).filter((cell) => families.includes(cell.modelId));
  assert.ok(cells.length >= 2 * 3 * 2, "both edit ids × three tiers × two lanes are shipped and routed");
  const gaps = (await measurabilityGaps()).filter((gap) => families.includes(gap.modelId));
  assert.equal(gaps.length, 0, gapReport(gaps));
});

// sc-22726. Same shape claim, for the FLUX.1 family: `flux_dev` and `flux_schnell` are the two base
// text-to-image providers of the shared FLUX.1 engine crates, and `pulid_flux_dev` is the identity
// route over the same FLUX.1-dev backbone — a REGISTRY route on mlx and a BESPOKE one on candle.
test("the flux.1 family is measurable on every shipped tier of every routed lane", async () => {
  const family = ["flux_dev", "flux_schnell", "pulid_flux_dev"];
  const cells = (await shippedTieredCells()).filter((cell) => family.includes(cell.modelId));
  assert.equal(cells.length, 3 * 3 * 2, "three models x three shipped tiers x two routed lanes");
  const gaps = (await measurabilityGaps()).filter((gap) => family.includes(gap.modelId));
  assert.equal(gaps.length, 0, gapReport(gaps));
});

// sc-22732. Same shape claim, for the turnkey still family: five catalog models over three engine
// crate pairs (`*-gen-kolors`, `*-gen-ideogram`, `*-gen-lens`), every one a plain reference-free
// text-to-image route whose engine id equals its catalog model id.
test("the kolors, ideogram and lens families are measurable on every shipped tier of every routed lane", async () => {
  const family = ["kolors", "ideogram_4", "ideogram_4_turbo", "lens", "lens_turbo"];
  const cells = (await shippedTieredCells()).filter((cell) => family.includes(cell.modelId));
  // The expected KEY SET, spelled out for the same reason the sc-22731 case below spells its own
  // out: a frozen `5 * 3 * 2` says only how MANY cells there should be, so a cell silently lost on
  // one member and gained on another still comes to 30 and this case stays green. Every member is
  // derivable from the manifest — none of these five carries a `platforms` key on any download, so
  // all three tiers are real on both routed lanes.
  const expected = family.flatMap((id) =>
    ["bf16", "q4", "q8"].flatMap((tier) => [`${id}:${tier}:candle`, `${id}:${tier}:mlx`]),
  );
  assert.deepEqual(cells.map((cell) => cell.key).sort(), expected.sort());
// sc-22730. Same shape claim, for the SD3.5 family: three DISTINCT base text-to-image providers of
// the shared SD3.5 engine crates, each with its OWN tiered rehost, on the registry path of BOTH
// lanes. `sd3_5_*` is the catalog model id AND the engine provider id, so no alias is involved.
test("the sd3.5 family is measurable on every shipped tier of every routed lane", async () => {
  const family = ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"];
  const cells = (await shippedTieredCells()).filter((cell) => family.includes(cell.modelId));
  assert.equal(cells.length, 3 * 3 * 2, "three models x three shipped tiers x two routed lanes");
  const gaps = (await measurabilityGaps()).filter((gap) => family.includes(gap.modelId));
  assert.equal(gaps.length, 0, gapReport(gaps));
});

// sc-22731. Same shape claim, for the SANA and Chroma1 families. Chroma1 is the ordinary case —
// three routes x three shipped tiers x two routed lanes. SANA is not: its packed tiers are
// `platforms: ["macos"]` turnkeys and the Candle lane has ONE dense cell per route, which is an
// unrouted (lane, tier) rather than a measurement that has been skipped.
test("the sana and chroma1 families are measurable on every shipped tier of every routed lane", async () => {
  const family = ["sana_1600m", "sana_sprint_1600m", "chroma1_hd", "chroma1_base", "chroma1_flash"];
  const cells = (await shippedTieredCells()).filter((cell) => family.includes(cell.modelId));
  // The expected KEY SET, spelled out (sc-22731 review). A frozen `2 * 4 + 3 * 3 * 2` said only how
  // MANY cells there should be: if the Candle SANA axis silently regained q4/q8 while some Chroma1
  // lane lost a tier, the count still came to 26 and this case stayed green. Every member below is
  // derivable from the manifest — SANA's packed tiers are `platforms: ["macos"]` turnkeys and its
  // only off-Mac download is the untiered dense diffusers snapshot, so its Candle lane has ONE cell
  // per route; Chroma1's downloads carry no `platforms` key at all, so every tier is real on both
  // lanes — and naming them is what makes a lost or gained cell say WHICH.
  const expected = [
    ...["sana_1600m", "sana_sprint_1600m"].flatMap((id) => [
      `${id}:bf16:candle`,
      `${id}:bf16:mlx`,
      `${id}:q4:mlx`,
      `${id}:q8:mlx`,
    ]),
    ...["chroma1_hd", "chroma1_base", "chroma1_flash"].flatMap((id) =>
      ["bf16", "q4", "q8"].flatMap((tier) => [`${id}:${tier}:candle`, `${id}:${tier}:mlx`]),
    ),
  ];
  assert.deepEqual(cells.map((cell) => cell.key).sort(), expected.sort());
  const gaps = (await measurabilityGaps()).filter((gap) => family.includes(gap.modelId));
  assert.equal(gaps.length, 0, gapReport(gaps));
});

// The Ideogram bf16 tier is the only shipped cell whose artifact is NOT the family's default
// repository, so the `tiers` override is exercised end to end rather than only asserted as data:
// the wrong binding would still classify `runnable`, and would only ever show up as a wrong
// repository and a wrong revision inside a captured record's loadability fingerprint.
test("the turnkey still family binds one artifact per member, and Ideogram's bf16 tier binds its own repository", async () => {
  const hub = await fakeHub([
    ["SceneWorks/kolors-mlx", REVISION, "q4"],
    ["SceneWorks/lens-mlx", REVISION, "q4"],
    ["SceneWorks/lens-turbo-mlx", UPSTREAM, "q4"],
    ["SceneWorks/ideogram-4-mlx", REVISION, "q4"],
    ["SceneWorks/ideogram-4", UPSTREAM, "bf16"],
  ]);
  for (const backend of ["mlx", "candle"]) {
    const context = { models: fakeModels(), backend, hubs: [hub], current: new Map(), captured: new Map() };

    const kolors = await classifyAnchor(`kolors:q4:${backend}`, { provider: "kolors" }, context);
    assert.equal(kolors.status, "runnable", `${backend}: ${kolors.reason}`);
    assert.deepEqual(kolors.env, {
      SCENEWORKS_KOLORS_REPOSITORY: "SceneWorks/kolors-mlx",
      SCENEWORKS_KOLORS_REVISION: REVISION,
      SCENEWORKS_KOLORS_ROOT: snapshotPath(hub, "SceneWorks/kolors-mlx", REVISION, "q4"),
    });

    // Base Lens and Lens-Turbo are separate repositories at separate revisions: a turbo plan bound
    // to the base family would re-label the base model's peaks as the distilled model's.
    const lens = await classifyAnchor(`lens:q4:${backend}`, { provider: "lens" }, context);
    assert.equal(lens.status, "runnable", `${backend}: ${lens.reason}`);
    assert.deepEqual(lens.env, {
      SCENEWORKS_LENS_REPOSITORY: "SceneWorks/lens-mlx",
      SCENEWORKS_LENS_REVISION: REVISION,
      SCENEWORKS_LENS_ROOT: snapshotPath(hub, "SceneWorks/lens-mlx", REVISION, "q4"),
    });
    const turbo = await classifyAnchor(`lens_turbo:q4:${backend}`, { provider: "lens_turbo" }, context);
    assert.equal(turbo.status, "runnable", `${backend}: ${turbo.reason}`);
    assert.deepEqual(turbo.env, {
      SCENEWORKS_LENS_TURBO_REPOSITORY: "SceneWorks/lens-turbo-mlx",
      SCENEWORKS_LENS_TURBO_REVISION: UPSTREAM,
      SCENEWORKS_LENS_TURBO_ROOT: snapshotPath(hub, "SceneWorks/lens-turbo-mlx", UPSTREAM, "q4"),
    });

    // Both Ideogram members: q4 from the packed turnkey, bf16 from the second repository.
    for (const [modelId, provider] of [["ideogram_4", "ideogram_4"], ["ideogram_4_turbo", "ideogram_4_turbo"]]) {
      const packed = await classifyAnchor(`${modelId}:q4:${backend}`, { provider }, context);
      assert.equal(packed.status, "runnable", `${backend} ${modelId}: ${packed.reason}`);
      assert.deepEqual(packed.env, {
        SCENEWORKS_IDEOGRAM_REPOSITORY: "SceneWorks/ideogram-4-mlx",
        SCENEWORKS_IDEOGRAM_REVISION: REVISION,
        SCENEWORKS_IDEOGRAM_ROOT: snapshotPath(hub, "SceneWorks/ideogram-4-mlx", REVISION, "q4"),
      });
      const dense = await classifyAnchor(`${modelId}:bf16:${backend}`, { provider }, context);
      assert.equal(dense.status, "runnable", `${backend} ${modelId}: ${dense.reason}`);
      assert.deepEqual(dense.env, {
        SCENEWORKS_IDEOGRAM_BF16_REPOSITORY: "SceneWorks/ideogram-4",
        SCENEWORKS_IDEOGRAM_BF16_REVISION: UPSTREAM,
        SCENEWORKS_IDEOGRAM_BF16_ROOT: snapshotPath(hub, "SceneWorks/ideogram-4", UPSTREAM, "bf16"),
      });
    }

    // A tier the host does not hold is `weights_missing`, and the reason names the repository the
    // tier actually ships from — the packed one for q8, never the bf16 repo.
    const missing = await classifyAnchor(`ideogram_4:q8:${backend}`, { provider: "ideogram_4" }, context);
    assert.equal(missing.status, "weights_missing");
    assert.match(missing.reason, /ideogram-4-mlx@.*\/q8 on this host/);
  }
});

// sc-22730. The plan's SD3.5 fingerprints are the one claim a capture cannot re-derive on a host
// with no weights, and they are duplicated across three artifacts: the plan, the engine (through
// inference PR 950) and — on the candle lane only — the shipped manifest's declared contract.
//
// The CANDLE half is a real cross-source binding, and it holds AT THE OLD PIN: the manifest already
// declares `sd35-<route>-candle-resident-staged-v1` for every tier of every member, so a plan row
// naming anything else would name an identity the shipped declaration never promises.
//
// The MLX half CANNOT be bound to a constant here: `mlx-gen-sd3` at the compiled pin publishes no
// production identity at all (that is exactly what inference PR 950 adds), so there is nothing for
// CI at the old pin to read. It is bound to the (route, TIER) shape the merged engine mints instead
// — `mlx-gen-sd3/src/memory_strategy.rs:269-288` at `d606395b5` keys the identity on the proven
// artifact tier, because the three tiers of one route are three different resident sets and one
// anchor cannot price all three. The byte-for-byte binding to the LOADED contract is enforced where
// it can be — inside the capture, by `run_sd3`, which refuses a provider whose published string
// differs from `sd3_calibration_fingerprint(arm, artifact.tier)`.
test("every planned sd3.5 fingerprint is one its lane's declaration can emit", async () => {
  const plan = await readPlan();
  const models = await readManifestModels();
  const slugs = { sd3_5_large: "large", sd3_5_large_turbo: "large-turbo", sd3_5_medium: "medium" };
  let seen = 0;
  for (const [key, entry] of Object.entries(plan.anchors)) {
    const { modelId, tier, backend } = anchorParts(key);
    if (!(modelId in slugs)) continue;
    seen += 1;
    if (backend === "candle") {
      const model = models.find((candidate) => candidate.id === modelId);
      const declared = new Set(
        (model?.candle?.memoryStrategyContract?.implementations ?? []).map((impl) => impl.fingerprint),
      );
      assert.ok(
        declared.has(entry.calibrationFingerprint),
        `${key}: plan names ${entry.calibrationFingerprint}, manifest declares ${[...declared].join(", ")}`,
      );
    } else {
      assert.equal(entry.calibrationFingerprint, `sd3-5-${slugs[modelId]}-${tier}-mlx-shared-ladder-v1`, key);
    }
  }
  assert.equal(seen, 3 * 3 * 2, "every SD3.5 cell is priced by the plan");
});

// sc-22726 left this binding open and sc-22730 closes it: the artifact repository of every family
// is written down TWICE — as `PROVIDER_FAMILIES[*].repo` here and as a `pub const *_REPOSITORY` in
// the adapter's lib.rs, which is what both arms validate the operator's env against
// (`validate_artifact_identity`). Editing either alone yields a runner that stages a root the
// adapter then refuses by name, with every test in both languages still green.
test("PROVIDER_FAMILIES repos are the adapter's *_REPOSITORY constants", async () => {
  const lib = await readFile(path.join(ROOT, ADAPTER_LIB_PATH), "utf8");
  const declared = new Set();
  for (const match of lib.matchAll(/pub const [A-Z0-9_]+_REPOSITORY: &str = "([^"]+)";/g)) {
    declared.add(match[1]);
  }
  assert.ok(declared.size > 0, "lib.rs declares repository constants");
  for (const [provider, family] of Object.entries(PROVIDER_FAMILIES)) {
    // LTX-2.5 is bound by the harness itself rather than by an adapter env family, so its repo
    // literal lives in this module (`LTX25_REPOSITORY`) and not in lib.rs.
    if (family.ltx25) continue;
    assert.ok(
      declared.has(family.repo),
      `${provider}: ${family.repo} is not declared as a *_REPOSITORY const in ${ADAPTER_LIB_PATH}`,
    );
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
