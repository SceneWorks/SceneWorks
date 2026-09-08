import assert from "node:assert/strict";
import { mkdtemp, mkdir, open, readdir, writeFile, readFile, stat } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  OUT_OF_MATRIX_CATALOG_ENTRIES,
  parseBackendTierOverrides,
  parseInternalCandleVideoRoutes,
  parseRouteRegistryLaneTiers,
} from "./generate-memory-matrix.mjs";
import { routedLanes } from "./check-tier-integrity.mjs";
import { adapterCapturableProviders } from "./stale-lane-report.mjs";
import {
  ROOT,
  ADAPTER_LIB_PATH,
  MATRIX_PATH,
  PLAN_PATH,
  PACKAGED_SOURCES_PATH,
  PROVIDER_FAMILIES,
  SDXL_COMPONENTS,
  MAGE_COMPONENTS,
  MAGE_COMPONENT_IDS,
  providerFamily,
  anchorParts,
  anchorSlug,
  appendPackagedSource,
  writePackagedSource,
  RUSTFMT_EDITION,
  capturedInCampaign,
  classifyAnchor,
  compiledInferencePin,
  familyFor,
  hubRoots,
  failureReason,
  FAILURE_REASON_LIMIT,
  extractSeedingNewAnchors,
  SEED_DIGEST,
  SENSENOVA_DISTILL_MERGED_MARKER,
  measureAnchor,
  parseArgs,
  providerCommand,
  planRun,
  readManifestModels,
  readPlan,
  snapshotPath,
  tierDownload,
  familyArtifact,
  resolveArtifactRoot,
  parseSdxlRoutes,
  readSdxlCandleRoutes,
  sdxlCandleRouteDrift,
  parseDeclaredStrategySupport,
  readDeclaredStrategySupport,
  SDXL_ROUTES_PATH,
  SDXL_ROUTES_UNCHECKED,
  parseWanMlxSealedProviders,
  readWanMlxSealedProviders,
  wanMlxSealGap,
  WAN_MLX_LOADER_PATH,
  WAN_MLX_SEAL_UNCHECKED,
  WATCHDOG,
  LTX_Q4_F305_CRASH_FOOTPRINT_BYTES,
  UNIFIED_RESERVE_BYTES,
  guardsCapture,
  probeAdapter,
  watchdogCeilings,
  watchdogGuard,
  watchdogHardStop,
  watchdogPeakFootprint,
  runtimeBudgetStopSeconds,
  RUNTIME_BUDGET_STOP_PATTERN,
  PROBE_BUDGET_MINUTES,
  WITNESSED_CAPTURE_SECONDS,
  probeBudgetMinutes,
  probeLane,
  runtimeBudgetExceededReason,
  METAL_REFUSAL,
  ARTIFACT_UNSUPPORTED,
  artifactUnsupported,
  MINIMAX_UPSTREAM_ROOT_FILES,
  MINIMAX_TEXT_ENCODER_CONFIG,
  MINIMAX_TIER_DIT_FILES,
  requiredFilesFor,
  QWEN_EDIT_LIGHTNING_LORA,
  LTX25_REPOSITORY,
  LTX_2_3_REPOSITORY,
  LTX25_DEV_REFINEMENT_LORA,
  LTX25_ENHANCER_DIR,
  INSTANTID_IDENTITY_BUNDLE_FILES,
  INSTANTID_CONTROLNET_WEIGHT_FILE,
  ANCHOR_STORE_PATH,
  ANCHOR_LOADER_CONFIG_PATH,
  DOCKERFILE_PATH,
  DOCKERFILE_EMBED_ANCHOR,
  insertEvidenceCopy,
  receiptDisposition,
  assertStageable,
  MAX_STAGED_FILE_BYTES,
  readExceededBounds,
  anchorDownloadTargets,
  hfDownloadArgv,
  fetchAnchorSnapshots,
  tierDownloadRows,
  manifestPlatform,
  HF_CLI_CANDIDATES,
} from "./measure-memory-catalog.mjs";
import {
  ANCHOR_LANE_DEFAULT_STRATEGY_PATH,
  ANCHOR_STRATEGY,
  LTX25_LANE_PROVIDERS,
  planAnchor,
} from "./memory-calibration-harness.mjs";

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
    // The reference entry is its own catalog model on the same engine id, and it stages the tier
    // tree at every tier (`transformer_ref/` ships only in the rehost) while still loading the
    // dense upstream root — so it resolves through its OWN manifest downloads.
    {
      id: "minimax_h3_ref",
      downloads: [
        { repo: "SceneWorks/minimax-h3-mlx", revision: REVISION, variant: "bf16", files: ["bf16/transformer_ref/*"] },
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
    // sc-22738: the InstantID route — the plain RealVisXL rehost plus the three caller-staged SDXL
    // components, and an identity stack that is no download at all but two staged directories.
    {
      id: "instantid_realvisxl",
      downloads: [
        { repo: "SceneWorks/realvisxl-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] },
        ...SDXL_COMPONENTS.map(({ repo }) => ({ repo, revision: UPSTREAM, coRequisite: true, files: ["*"] })),
      ],
    },
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
    // sc-22736: the first family whose ARTIFACT is per (lane, tier). The MLX rehost carries all
    // three tiers under one revision; the candle rehost carries the packed two; the candle bf16
    // leg is the UPSTREAM Diffusers checkpoint, which the manifest ships with NO revision and
    // whose weights sit at the snapshot root rather than under a `bf16/` subtree.
    {
      id: "wan_2_2",
      downloads: [
        { repo: "SceneWorks/wan2.2-ti2v-5b-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] },
        { repo: "SceneWorks/wan2.2-ti2v-5b-mlx", revision: REVISION, variant: "bf16", files: ["bf16/*"] },
        { repo: "SceneWorks/wan2.2-ti2v-5b-candle", revision: UPSTREAM, variant: "q4", files: ["q4/*"] },
        { repo: "Wan-AI/Wan2.2-TI2V-5B-Diffusers", variant: "bf16", files: [] },
      ],
    },
  ];
}

/**
 * Stage everything the pinned MiniMax-H3 loader opens for `tier`, in the two roots it opens them
 * in. Written from the same declarations `classifyAnchor` probes, so a file added to either
 * declaration is staged here with no edit — the tests that want a COMPLETE install say so by
 * calling this, and the one that wants a specific hole omits exactly that file.
 */
async function stageMinimaxFiles(hub, tier, { omit = [] } = {}) {
  const roots = [
    [snapshotPath(hub, "SceneWorks/minimax-h3-mlx", REVISION, tier),
      requiredFilesFor({ all: MINIMAX_TIER_DIT_FILES, q4: [MINIMAX_TEXT_ENCODER_CONFIG], q8: [MINIMAX_TEXT_ENCODER_CONFIG] }, tier)],
    [snapshotPath(hub, "MiniMaxAI/MiniMax-H3", UPSTREAM),
      requiredFilesFor({ all: MINIMAX_UPSTREAM_ROOT_FILES, bf16: [MINIMAX_TEXT_ENCODER_CONFIG] }, tier)],
  ];
  for (const [root, files] of roots) {
    for (const file of files) {
      if (omit.includes(file)) continue;
      await mkdir(path.join(root, path.dirname(file)), { recursive: true });
      await writeFile(path.join(root, file), "{}\n");
    }
  }
}

/**
 * The two artifacts the LTX-2.5 arms open BESIDE `<variant>/<tier>` (sc-22738), staged into the
 * snapshot root. `omit` names the one a test wants absent.
 */
async function stageLtx25SnapshotEntries(hub, { omit = [] } = {}) {
  const snapshot = snapshotPath(hub, LTX25_REPOSITORY, REVISION);
  if (!omit.includes(LTX25_DEV_REFINEMENT_LORA)) {
    await mkdir(path.join(snapshot, path.dirname(LTX25_DEV_REFINEMENT_LORA)), { recursive: true });
    await writeFile(path.join(snapshot, LTX25_DEV_REFINEMENT_LORA), "lora");
  }
  if (!omit.includes(LTX25_ENHANCER_DIR)) await mkdir(path.join(snapshot, LTX25_ENHANCER_DIR), { recursive: true });
}

/** The pinned Lightning distill LoRA file inside its already-staged snapshot (sc-22738). */
async function stageQwenLightningLora(hub, file = QWEN_EDIT_LIGHTNING_LORA.file) {
  const root = snapshotPath(hub, QWEN_EDIT_LIGHTNING_LORA.repo, QWEN_EDIT_LIGHTNING_LORA.revision);
  await mkdir(root, { recursive: true });
  await writeFile(path.join(root, file), "lora");
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

/**
 * sc-22736. Before the Wan 2.2 family every row rehosted all three tiers of both lanes in ONE
 * repository under a `<tier>/` subtree, so the family's `repo`/`env` was the whole answer. Wan
 * ships three shapes at once, and the worker selects between them the same way
 * (`crates/sceneworks-worker/src/video_jobs/candle.rs` `candle_wan_tier_repo_from_downloads`).
 *
 * Mutations this kills:
 * - dropping the `artifacts[backend][tier]` lookup (a candle bf16 plan would probe the packed
 *   rehost's non-existent `bf16/` subtree and report a stageable cell `weights_missing`);
 * - dropping `layout: "flat"` (the upstream Diffusers snapshot would be probed at `<root>/bf16`);
 * - resolving the unpinned upstream download through `tierDownload`'s any-revision fallback (which
 *   would name the PACKED rehost's revision for a dense checkpoint);
 * - reporting the unpinned-and-unstaged case with the ordinary "no repo@rev/tier" sentence, which
 *   sends an operator to fetch a revision the manifest never pinned.
 */
test("an artifact is resolved per (lane, tier), including the flat upstream leg the manifest leaves unpinned", async () => {
  const models = fakeModels();
  // A LOCAL family literal, not a `PROVIDER_FAMILIES` row: this test owns the mechanism, and the
  // mechanism must be provable before any family declares an arm that consumes it. The shape is
  // the one the Wan 2.2 manifest entries have — a per-lane rehost, packed tiers on the Candle
  // rehost, and the upstream unpinned Diffusers checkpoint as the Candle bf16 leg.
  const family = {
    env: "WAN_TI2V_5B",
    repo: "SceneWorks/wan2.2-ti2v-5b-mlx",
    arms: ["mlx", "candle"],
    artifacts: {
      candle: {
        q4: { repo: "SceneWorks/wan2.2-ti2v-5b-candle" },
        q8: { repo: "SceneWorks/wan2.2-ti2v-5b-candle" },
        bf16: { repo: "Wan-AI/Wan2.2-TI2V-5B-Diffusers", layout: "flat" },
      },
    },
  };
  assert.deepEqual(familyArtifact(family, "mlx", "bf16"), {
    env: "WAN_TI2V_5B", repo: "SceneWorks/wan2.2-ti2v-5b-mlx", layout: "tiered",
  });
  assert.deepEqual(familyArtifact(family, "candle", "q4"), {
    env: "WAN_TI2V_5B", repo: "SceneWorks/wan2.2-ti2v-5b-candle", layout: "tiered",
  });
  assert.deepEqual(familyArtifact(family, "candle", "bf16"), {
    env: "WAN_TI2V_5B", repo: "Wan-AI/Wan2.2-TI2V-5B-Diffusers", layout: "flat",
  });
  // A family with no `artifacts` block is unchanged: the row itself, tiered. Asserted against a
  // real shipped row so the override stays a per-family opt-in rather than a global change.
  assert.deepEqual(familyArtifact(PROVIDER_FAMILIES.qwen_image, "candle", "q8"), {
    env: PROVIDER_FAMILIES.qwen_image.env,
    repo: PROVIDER_FAMILIES.qwen_image.repo,
    layout: "tiered",
  });

  const hub = await fakeHub([
    ["SceneWorks/wan2.2-ti2v-5b-mlx", REVISION, "bf16"],
    ["SceneWorks/wan2.2-ti2v-5b-candle", UPSTREAM, "q4"],
    ["Wan-AI/Wan2.2-TI2V-5B-Diffusers", REVISION],
  ]);
  const mlx = await resolveArtifactRoot(models, "wan_2_2", "bf16", familyArtifact(family, "mlx", "bf16"), [hub]);
  assert.equal(mlx.root, snapshotPath(hub, "SceneWorks/wan2.2-ti2v-5b-mlx", REVISION, "bf16"));
  assert.equal(mlx.revision, REVISION);

  const packed = await resolveArtifactRoot(models, "wan_2_2", "q4", familyArtifact(family, "candle", "q4"), [hub]);
  assert.equal(packed.root, snapshotPath(hub, "SceneWorks/wan2.2-ti2v-5b-candle", UPSTREAM, "q4"));
  assert.equal(packed.revision, UPSTREAM, "the candle rehost's own revision, never the MLX one's");

  // The unpinned upstream leg: the revision is READ off whatever snapshot is staged, and the root
  // is the snapshot itself — there is no `bf16/` subtree in a Diffusers checkpoint.
  const flat = await resolveArtifactRoot(models, "wan_2_2", "bf16", familyArtifact(family, "candle", "bf16"), [hub]);
  assert.equal(flat.root, snapshotPath(hub, "Wan-AI/Wan2.2-TI2V-5B-Diffusers", REVISION));
  assert.equal(flat.revision, REVISION);

  const bare = await fakeHub([]);
  const missing = await resolveArtifactRoot(models, "wan_2_2", "bf16", familyArtifact(family, "candle", "bf16"), [bare]);
  assert.equal(missing.root, null);
  assert.match(missing.reason, /shipped without a pinned revision/);
  const missingTier = await resolveArtifactRoot(models, "wan_2_2", "q4", familyArtifact(family, "candle", "q4"), [bare]);
  assert.equal(missingTier.root, null);
  assert.match(missingTier.reason, /^no SceneWorks\/wan2\.2-ti2v-5b-candle@[0-9a-f]{8}\/q4 on this host$/);
});

test("classification: runnable anchors carry the adapter env family and the canonical tier root", async () => {
  const hub = await fakeHub([
    ["SceneWorks/qwen-image-mlx", REVISION, "q4"],
    ["SceneWorks/minimax-h3-mlx", REVISION, "q4"],
    ["MiniMaxAI/MiniMax-H3", UPSTREAM],
    ["SceneWorks/ltx-2.5-mlx", REVISION],
  ]);
  // sc-22738: a MiniMax cell is runnable only when the files the pinned loader opens are actually
  // there — the two DiT partitions and the packed text encoder in the tier root, and the six dense
  // documents in the upstream one. A bare directory tree is what this host really had, and what
  // `--list` used to call runnable.
  await stageMinimaxFiles(hub, "q4");
  await stageLtx25SnapshotEntries(hub);
  const context = { models: fakeModels(), backend: "mlx", hubs: [hub], current: new Map(), captured: new Map() };
  const qwen = await classifyAnchor("qwen_image:q4:mlx", { provider: "qwen_image" }, context);
  assert.equal(qwen.status, "runnable");
  assert.equal(qwen.physical, true, "the Qwen MLX arm needs the physical receipt session");
  assert.equal(qwen.sourceCapture, true, "and therefore the raw-log pair the receipt is written into");
  assert.deepEqual(qwen.env, {
    SCENEWORKS_QWEN_IMAGE_REPOSITORY: "SceneWorks/qwen-image-mlx",
    SCENEWORKS_QWEN_IMAGE_REVISION: REVISION,
    SCENEWORKS_QWEN_IMAGE_ROOT: snapshotPath(hub, "SceneWorks/qwen-image-mlx", REVISION, "q4"),
  });

  const minimax = await classifyAnchor("minimax_h3:q4:mlx", { provider: "minimax_h3" }, context);
  assert.equal(minimax.status, "runnable");
  assert.equal(minimax.physical, false);
  assert.equal(minimax.sourceCapture, false, "the MiniMax arm emits no sourceCapture; a raw-log pair would make the harness refuse the render");
  assert.equal(minimax.env.SCENEWORKS_MINIMAX_H3_UPSTREAM_ROOT, snapshotPath(hub, "MiniMaxAI/MiniMax-H3", UPSTREAM));
  assert.equal(minimax.env.SCENEWORKS_MINIMAX_H3_UPSTREAM_REVISION, UPSTREAM);

  const ltx = await classifyAnchor("ltx_2_5:q4:mlx", { provider: "ltx_2_5" }, context);
  assert.equal(ltx.status, "runnable");
  assert.equal(ltx.ltx25SnapshotRoot, snapshotPath(hub, "SceneWorks/ltx-2.5-mlx", REVISION));
  assert.deepEqual(ltx.env, {}, "the harness binds LTX-2.5's artifact roots itself");
  // sc-22738: NOT `physical` (the harness demands the currency receipt for qwen_image alone) but it
  // does emit a sourceCapture, so `measureAnchor` owes it the capture-dir pair.
  assert.equal(ltx.physical, false);
  assert.equal(ltx.sourceCapture, true, "mlx_ltx25.rs emits a physical_mlx sourceCapture on every run");
  const ltxCandle = await classifyAnchor("ltx_2_5:q4:candle", { provider: "ltx_2_5_distilled" }, {
    ...context, backend: "candle",
  });
  assert.equal(ltxCandle.status, "runnable", ltxCandle.reason);
  assert.equal(ltxCandle.sourceCapture, false, "the candle LTX-2.5 arm emits none; a raw-log pair would make the harness refuse it");

  const missingTier = await classifyAnchor("qwen_image:q8:mlx", { provider: "qwen_image" }, context);
  assert.equal(missingTier.status, "weights_missing");
  assert.match(missingTier.reason, /q8 on this host/);
});

// sc-22738. The campaign lost a booked `minimax_h3:bf16:mlx` capture to
// `read …/models--MiniMaxAI--MiniMax-H3/snapshots/<rev>/text_encoder: No such file or directory`.
// `--list` had called the cell runnable because the upstream probe asked only whether the snapshot
// DIRECTORY existed — and the dense tree staged on that host holds `vae/`, `audio_vae/`,
// `tokenizer/` and `FL2VA/` but no `text_encoder/`. The loader's reads are declared now, per tier,
// in both roots, so an incomplete mirror is refused by name before anything is scheduled.
test("a minimax cell whose snapshot is missing a file the loader opens is weights_missing, by name", async () => {
  const layout = [
    ["SceneWorks/minimax-h3-mlx", REVISION, "q4"],
    ["SceneWorks/minimax-h3-mlx", REVISION, "bf16"],
    ["MiniMaxAI/MiniMax-H3", UPSTREAM],
  ];
  const planned = { provider: "minimax_h3" };

  // The exact host condition: everything but the dense text encoder. bf16 takes its TE from the
  // upstream root, so bf16 is refused — and q4 takes its own from the tier root, so q4 is NOT.
  const holed = await fakeHub(layout);
  await stageMinimaxFiles(holed, "q4");
  await stageMinimaxFiles(holed, "bf16", { omit: [MINIMAX_TEXT_ENCODER_CONFIG] });
  const context = (hub) => ({ models: fakeModels(), backend: "mlx", hubs: [hub], current: new Map(), captured: new Map() });
  const refused = await classifyAnchor("minimax_h3:bf16:mlx", planned, context(holed));
  assert.equal(refused.status, "weights_missing", refused.reason);
  assert.match(refused.reason, /text_encoder\/config\.json/);
  assert.ok(
    refused.reason.includes(snapshotPath(holed, "MiniMaxAI/MiniMax-H3", UPSTREAM)),
    `the refusal names the root it probed: ${refused.reason}`,
  );
  const unaffected = await classifyAnchor("minimax_h3:q4:mlx", planned, context(holed));
  assert.equal(unaffected.status, "runnable", unaffected.reason);

  // The reference entry rides the same declarations — it stages the tier tree at every tier and
  // still loads the dense root — so its bf16 cell is refused for the same missing file.
  const ref = await classifyAnchor("minimax_h3_ref:bf16:mlx", planned, context(holed));
  assert.equal(ref.status, "weights_missing", ref.reason);
  assert.match(ref.reason, /text_encoder\/config\.json/);

  // Each declared file, one at a time: dropping ANY of them refuses the cell, and no single one of
  // them is load-bearing for the others. A blanket "the snapshot is there" probe passes all of these.
  for (const file of [...MINIMAX_UPSTREAM_ROOT_FILES, MINIMAX_TEXT_ENCODER_CONFIG]) {
    const hub = await fakeHub(layout);
    await stageMinimaxFiles(hub, "bf16", { omit: [file] });
    const row = await classifyAnchor("minimax_h3:bf16:mlx", planned, context(hub));
    assert.equal(row.status, "weights_missing", `${file}: ${row.reason}`);
    assert.ok(row.reason.includes(file), `${file}: ${row.reason}`);
  }
  for (const file of [...MINIMAX_TIER_DIT_FILES, MINIMAX_TEXT_ENCODER_CONFIG]) {
    const hub = await fakeHub(layout);
    await stageMinimaxFiles(hub, "q4", { omit: [file] });
    const row = await classifyAnchor("minimax_h3:q4:mlx", planned, context(hub));
    assert.equal(row.status, "weights_missing", `${file}: ${row.reason}`);
    assert.ok(row.reason.includes(file), `${file}: ${row.reason}`);
  }

  // And a COMPLETE install is still runnable on both tiers, with the same env family as before.
  const whole = await fakeHub(layout);
  await stageMinimaxFiles(whole, "q4");
  await stageMinimaxFiles(whole, "bf16");
  for (const tier of ["q4", "bf16"]) {
    const row = await classifyAnchor(`minimax_h3:${tier}:mlx`, planned, context(whole));
    assert.equal(row.status, "runnable", `${tier}: ${row.reason}`);
    assert.equal(row.env.SCENEWORKS_MINIMAX_H3_UPSTREAM_ROOT, snapshotPath(whole, "MiniMaxAI/MiniMax-H3", UPSTREAM));
    assert.equal(row.env.SCENEWORKS_MINIMAX_H3_ROOT, snapshotPath(whole, "SceneWorks/minimax-h3-mlx", REVISION, tier));
  }
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

// sc-22734 review. Nine `_fast` MLX cells hard-fail at capture — after the load, hours into a
// booked session — when a `_fast` tier root carries no `distill_merged.json`: both engines withhold
// the production calibration identity without it, so the harness sees an identity mismatch rather
// than a classification. `classifyAnchor` now reads the marker off the resolved tier root and
// refuses the cell by name before anything is scheduled.
test("a _fast sensenova tier root without the pre-merge marker is refused by name, not scheduled", async () => {
  const REPO = "SceneWorks/sensenova-u1-8b-fast-mlx";
  const models = [
    { id: "sensenova_u1_8b_fast", downloads: [{ repo: REPO, revision: REVISION, variant: "q4", files: ["q4/*"] }] },
    { id: "sensenova_u1_8b", downloads: [{ repo: "SceneWorks/sensenova-u1-8b-mlx", revision: REVISION, variant: "q4", files: ["q4/*"] }] },
  ];
  const hub = await fakeHub([
    [REPO, REVISION, "q4"],
    ["SceneWorks/sensenova-u1-8b-mlx", REVISION, "q4"],
  ]);
  const planned = { provider: "sensenova_u1_8b_fast" };
  for (const backend of ["mlx", "candle"]) {
    const context = { models, backend, hubs: [hub], current: new Map(), captured: new Map() };

    // The marker is absent: refused by NAME, and the reason cites the file and the root.
    const refused = await classifyAnchor(`sensenova_u1_8b_fast:q4:${backend}`, planned, context);
    assert.equal(refused.status, "weights_missing", `${backend}: ${refused.reason}`);
    assert.match(refused.reason, new RegExp(SENSENOVA_DISTILL_MERGED_MARKER.replace(".", "\\.")));
    assert.match(refused.reason, /calibration identity/);
    assert.ok(
      refused.reason.includes(snapshotPath(hub, REPO, REVISION, "q4")),
      `${backend}: the refusal names the tier root it probed`,
    );

    // The QUALITY route declares no marker requirement, so the same hub runs it — proving the
    // probe is scoped to the `_fast` rows and is not a new blanket requirement.
    const quality = await classifyAnchor(
      `sensenova_u1_8b:q4:${backend}`,
      { provider: "sensenova_u1_8b" },
      context,
    );
    assert.equal(quality.status, "runnable", `${backend}: ${quality.reason}`);
  }

  // Write the marker and the same cell becomes runnable on both lanes.
  await writeFile(path.join(snapshotPath(hub, REPO, REVISION, "q4"), SENSENOVA_DISTILL_MERGED_MARKER), "{}\n");
  for (const backend of ["mlx", "candle"]) {
    const context = { models, backend, hubs: [hub], current: new Map(), captured: new Map() };
    const runnable = await classifyAnchor(`sensenova_u1_8b_fast:q4:${backend}`, planned, context);
    assert.equal(runnable.status, "runnable", `${backend}: ${runnable.reason}`);
    assert.equal(runnable.env.SCENEWORKS_SENSENOVA_U1_8B_FAST_ROOT, snapshotPath(hub, REPO, REVISION, "q4"));
  }
});

// Every `_fast` family row declares the marker requirement, and no quality row does — derived from
// the table, so a seventh route added to either half is covered with no edit here.
test("the marker requirement is declared on exactly the _fast sensenova family rows", () => {
  const sensenova = Object.entries(PROVIDER_FAMILIES).filter(([, row]) =>
    typeof row.provider === "string" && row.provider.startsWith("sensenova_u1_8b"));
  assert.ok(sensenova.length > 0, "the table declares SenseNova rows");
  for (const [modelId, row] of sensenova) {
    const expected = row.provider === "sensenova_u1_8b_fast" ? [SENSENOVA_DISTILL_MERGED_MARKER] : undefined;
    assert.deepEqual(row.requiredTierFiles, expected, modelId);
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
  await stageQwenLightningLora(hub);
  for (const backend of ["mlx", "candle"]) {
    const context = { models: fakeModels(), backend, hubs: [hub], current: new Map(), captured: new Map() };
    const planned = { provider: "qwen_image_edit", mode: "edit_image" };
    const base = await classifyAnchor(`qwen_image_edit_2511:q4:${backend}`, planned, context);
    assert.equal(base.status, "runnable", `${backend}: ${base.reason}`);
    assert.equal(base.physical, false);
    assert.equal(base.sourceCapture, false, "the edit arm emits no sourceCapture, whatever it shares with qwen_image");
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

// sc-22738. The campaign lost all three `qwen_image_edit_2511_lightning:*:mlx` captures to
// `memory-strategy provider adapter: the Lightning distill LoRA is not at …-4steps-V1.0-bf16.safetensors`.
// `--list` had called them runnable because the side-artifact probe asked only whether the SNAPSHOT
// existed — and the snapshot staged on that host carries the 8-step distill, not the pinned 4-step
// one the arm joins. A present snapshot is not the pinned file.
test("a Lightning cell whose distill snapshot lacks the pinned LoRA file is weights_missing, by name", async () => {
  const lora = QWEN_EDIT_LIGHTNING_LORA;
  const planned = { provider: "qwen_image_edit", mode: "edit_image" };
  const layout = [
    ["SceneWorks/qwen-image-edit-2511-mlx", REVISION, "q4"],
    [lora.repo, lora.revision],
  ];
  // The exact host condition: the snapshot is there, holding the WRONG step count.
  const holed = await fakeHub(layout);
  await stageQwenLightningLora(holed, "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors");
  const context = (hub) => ({ models: fakeModels(), backend: "mlx", hubs: [hub], current: new Map(), captured: new Map() });
  const refused = await classifyAnchor("qwen_image_edit_2511_lightning:q4:mlx", planned, context(holed));
  assert.equal(refused.status, "weights_missing", refused.reason);
  assert.match(refused.reason, new RegExp(lora.file.replaceAll(".", "\\.")), "the reason names the file the arm opens");
  assert.match(refused.reason, new RegExp(snapshotPath(holed, lora.repo, lora.revision)), "and the root it looked in");
  // The base id shares the tier root and attaches no LoRA, so it is untouched by the refusal.
  assert.equal((await classifyAnchor("qwen_image_edit_2511:q4:mlx", planned, context(holed))).status, "runnable");
  // And with the pinned file staged, the same cell is runnable — the probe is the file, not the name.
  const complete = await fakeHub(layout);
  await stageQwenLightningLora(complete);
  const row = await classifyAnchor("qwen_image_edit_2511_lightning:q4:mlx", planned, context(complete));
  assert.equal(row.status, "runnable", row.reason);
});

// sc-22738. The same class of miss on the other two families whose arms open something the tier or
// snapshot probe never looked at: the LTX-2.5 stage-two refinement LoRA and stock enhancer, and the
// InstantID identity stack's staged files. A directory that exists but is half-staged is exactly as
// unloadable as an absent one, and calling either `runnable` books a capture that cannot open.
test("an LTX-2.5 snapshot missing what the arm opens beside the load root is weights_missing, by name", async () => {
  const context = (hub, backend) => ({ models: fakeModels(), backend, hubs: [hub], current: new Map(), captured: new Map() });
  // MLX needs both; Candle attaches the LoRA and never opens the enhancer.
  const noLora = await fakeHub([[LTX25_REPOSITORY, REVISION, "distilled"]]);
  await stageLtx25SnapshotEntries(noLora, { omit: [LTX25_DEV_REFINEMENT_LORA] });
  for (const [backend, provider] of [["mlx", "ltx_2_5"], ["candle", "ltx_2_5_distilled"]]) {
    const row = await classifyAnchor(`ltx_2_5:q4:${backend}`, { provider }, context(noLora, backend));
    assert.equal(row.status, "weights_missing", `${backend}: ${row.reason}`);
    assert.match(row.reason, /distilled_lora\/ltx-2\.5-22b-distilled-lora-450-bf16\.safetensors/);
  }
  const noEnhancer = await fakeHub([[LTX25_REPOSITORY, REVISION, "distilled"]]);
  await stageLtx25SnapshotEntries(noEnhancer, { omit: [LTX25_ENHANCER_DIR] });
  const mlx = await classifyAnchor("ltx_2_5:q4:mlx", { provider: "ltx_2_5" }, context(noEnhancer, "mlx"));
  assert.equal(mlx.status, "weights_missing", mlx.reason);
  assert.match(mlx.reason, /enhancer/);
  const candle = await classifyAnchor("ltx_2_5:q4:candle", { provider: "ltx_2_5_distilled" }, context(noEnhancer, "candle"));
  assert.equal(candle.status, "runnable", "the Candle arm never opens the enhancer, so it must not be required of it");
});

test("an InstantID cell whose staged identity stack is missing a file is weights_missing, by name", async () => {
  const hub = await fakeHub([
    ["SceneWorks/realvisxl-mlx", REVISION, "q4"],
    ...SDXL_COMPONENTS.map(({ repo }) => [repo, UPSTREAM]),
  ]);
  const weights = await mkdtemp(path.join(tmpdir(), "instantid-weights-"));
  const controlnet = await mkdtemp(path.join(tmpdir(), "instantid-controlnet-"));
  const previous = { ...process.env };
  process.env.SCENEWORKS_INSTANTID_WEIGHTS = weights;
  process.env.SCENEWORKS_INSTANTID_CONTROLNET = controlnet;
  try {
    const planned = { provider: "instantid" };
    const context = { models: fakeModels(), backend: "mlx", hubs: [hub], current: new Map(), captured: new Map() };
    // Two empty directories: what `--list` used to call a staged identity stack.
    const bare = await classifyAnchor("instantid_realvisxl:q4:mlx", planned, context);
    assert.equal(bare.status, "weights_missing", bare.reason);
    assert.match(bare.reason, new RegExp(INSTANTID_IDENTITY_BUNDLE_FILES[0].replaceAll(".", "\\.")));
    // The bundle complete, the IdentityNet directory still empty: refused on its own file.
    for (const file of INSTANTID_IDENTITY_BUNDLE_FILES) await writeFile(path.join(weights, file), "weights");
    const half = await classifyAnchor("instantid_realvisxl:q4:mlx", planned, context);
    assert.equal(half.status, "weights_missing", half.reason);
    assert.match(half.reason, new RegExp(INSTANTID_CONTROLNET_WEIGHT_FILE.replaceAll(".", "\\.")));
    await writeFile(path.join(controlnet, INSTANTID_CONTROLNET_WEIGHT_FILE), "weights");
    const row = await classifyAnchor("instantid_realvisxl:q4:mlx", planned, context);
    assert.equal(row.status, "runnable", row.reason);
    assert.equal(row.env.SCENEWORKS_INSTANTID_WEIGHTS, weights);
    assert.equal(row.env.SCENEWORKS_INSTANTID_CONTROLNET, controlnet);
  } finally {
    process.env = previous;
  }
});

// The three declarations above are the ADAPTER's own constants, not this script's opinion of them.
// A drift on either side would send a capture at an artifact the arm does not open (or refuse one it
// does), so each is read out of the Rust source and compared.
test("the extra artifacts declared per family are the ones the adapter source names", async () => {
  const lib = await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/lib.rs"), "utf8");
  const bundleFiles = /pub const INSTANTID_IDENTITY_BUNDLE_FILES: \[&str; 3\] = \[([\s\S]*?)\];/.exec(lib);
  assert.ok(bundleFiles, "lib.rs still declares INSTANTID_IDENTITY_BUNDLE_FILES");
  const named = [...bundleFiles[1].matchAll(/INSTANTID_([A-Z_]+)_FILE/g)].map(([, stem]) => {
    const value = new RegExp(`pub const INSTANTID_${stem}_FILE: &str = "([^"]+)"`).exec(lib);
    assert.ok(value, `lib.rs still declares INSTANTID_${stem}_FILE`);
    return value[1];
  });
  assert.deepEqual([...INSTANTID_IDENTITY_BUNDLE_FILES], named, "the staged bundle list is the adapter's own");
  assert.equal(
    /pub const INSTANTID_CONTROLNET_WEIGHT_FILE: &str = "([^"]+)"/.exec(lib)?.[1],
    INSTANTID_CONTROLNET_WEIGHT_FILE,
  );
  const instantid = PROVIDER_FAMILIES.instantid_realvisxl.stagedEnv;
  assert.deepEqual(instantid.map((entry) => entry.env), [
    /pub const INSTANTID_IDENTITY_BUNDLE_ENV: &str = "([^"]+)"/.exec(lib)?.[1],
    /pub const INSTANTID_CONTROLNET_ENV: &str = "([^"]+)"/.exec(lib)?.[1],
  ], "each staged directory is bound to the env var the adapter reads");
  assert.ok(instantid.every((entry) => (entry.files ?? []).length > 0), "and each declares what it must carry");

  const mlxLtx25 = await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/bin/mlx_ltx25.rs"), "utf8");
  const candle = await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/bin/candle.rs"), "utf8");
  assert.equal(/const DEV_ADAPTER: &str = "([^"]+)"/.exec(mlxLtx25)?.[1], LTX25_DEV_REFINEMENT_LORA);
  assert.equal(
    /const LTX25_DISTILL_LORA_RELATIVE_PATH: &str =\s*"([^"]+)"/.exec(candle)?.[1],
    LTX25_DEV_REFINEMENT_LORA,
    "both lanes attach the same file, so one declaration serves both families",
  );
  assert.match(mlxLtx25, new RegExp(`join\\("${LTX25_ENHANCER_DIR}"\\)`), "the MLX arm still opens the enhancer by that name");

  const mlx = await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/bin/mlx.rs"), "utf8");
  assert.equal(
    /pub const QWEN_EDIT_LIGHTNING_FILE: &str =\s*"([^"]+)"/.exec(lib)?.[1],
    QWEN_EDIT_LIGHTNING_LORA.file,
    "the declared distill LoRA is the file the arm joins",
  );
  assert.match(mlx, /join\(protocol::QWEN_EDIT_LIGHTNING_FILE\)/);
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

// sc-22738. `sourceCapture` is the SECOND half of that rule, and the half the campaign proved was
// missing: it says the arm emits a provider `sourceCapture`, which is what obliges the runner to
// pass `SCENEWORKS_MEMORY_CAPTURE_DIR`/`_SOURCE_PATH_PREFIX` and the harness's
// `--raw-log-dir`/`--source-path-prefix`. The coupling is enforced in BOTH directions
// (`capturePlannedCase`), so an over-declaration is as fatal as an under-declaration: this is a set
// equality, never a subset.
//
// The expected side is DERIVED from the adapter, not spelled here — the mistake being fixed was a
// hand-kept flag that said "qwen only" while `mlx_ltx25.rs` had emitted a `physical_mlx`
// sourceCapture since SC-18783. The walk starts at each `match provider` arm's own entry function
// and follows calls through the MLX bin and its `#[path]` sibling modules, so an arm that grows an
// emission (or loses one) moves this case with no edit here.
function stripRustCommentsForScan(source) {
  return source.replace(/\/\/[^\n]*/g, "").replace(/\/\*[\s\S]*?\*\//g, "");
}

/** Rust with every `#[cfg(test)] mod …{…}` removed — and nothing else: `#[cfg(test)] const X …;`
 *  carries no block, and slicing at the attribute would eat the provider-id consts after it. */
function stripRustTestModules(source) {
  const marker = "#[cfg(test)]";
  for (let from = 0; ;) {
    const at = source.indexOf(marker, from);
    if (at === -1) return source;
    if (!/^\s*(?:pub\s+)?mod\s+/.test(source.slice(at + marker.length))) {
      from = at + marker.length;
      continue;
    }
    const open = source.indexOf("{", at);
    let depth = 0;
    let i = open;
    for (; i < source.length; i += 1) {
      if (source[i] === "{") depth += 1;
      else if (source[i] === "}") { depth -= 1; if (!depth) break; }
    }
    source = source.slice(0, at) + source.slice(i + 1);
    from = at;
  }
}

/** `name -> body` for every top-level `fn` in one Rust source, keyed as the caller spells it. */
function rustFunctionBodies(source, qualify = (name) => name) {
  const bodies = new Map();
  const declaration = /^(?:pub(?:\([a-z]+\))?\s+)?(?:async\s+)?fn\s+([a-z_][A-Za-z0-9_]*)/gm;
  for (let match; (match = declaration.exec(source));) {
    const open = source.indexOf("{", match.index);
    if (open === -1) continue;
    let depth = 0;
    let i = open;
    for (; i < source.length; i += 1) {
      if (source[i] === "{") depth += 1;
      else if (source[i] === "}") { depth -= 1; if (!depth) break; }
    }
    bodies.set(qualify(match[1]), source.slice(open, i + 1));
  }
  return bodies;
}

const RUST_CALL = /\b((?:[a-z_][A-Za-z0-9_]*::)?[a-z_][A-Za-z0-9_]*)\s*\(/g;

test("every MLX adapter arm that emits a provider sourceCapture is declared, and no other", async () => {
  const binDir = path.join(ROOT, "crates/sceneworks-memory-adapter/src/bin");
  const members = (await readdir(binDir)).filter(
    (name) => name === "mlx.rs" || name.startsWith("mlx_"),
  ).sort();
  assert.ok(members.length >= 2, "expected the mlx bin plus its #[path] arm modules");
  const sources = new Map();
  for (const name of members) {
    sources.set(name, stripRustTestModules(stripRustCommentsForScan(
      await readFile(path.join(binDir, name), "utf8"),
    )));
  }
  // Every sibling is indexed under the module path the bin calls it by, so a NEW arm module that
  // emits is walked rather than silently unreachable.
  const bodies = new Map();
  for (const [name, source] of sources) {
    const module = name === "mlx.rs" ? null : name.slice(0, -3);
    for (const [fn, body] of rustFunctionBodies(source, (id) => (module ? `${module}::${id}` : id))) {
      bodies.set(fn, body);
    }
  }
  const emitters = [...sources].filter(([, source]) => source.includes('"sourceCapture"')).map(([name]) => name);
  assert.ok(emitters.length > 0, "no MLX source emits a sourceCapture at all; the scan reads nothing");

  const consts = new Map(
    [...sources.get("mlx.rs").matchAll(/\bconst\s+([A-Z][A-Z0-9_]*)\s*:\s*&str\s*=\s*"([^"]*)"\s*;/g)]
      .map((match) => [match[1], match[2]]),
  );
  const armsFor = new Map();
  const dispatch = /(?:^|\n)\s*(?:"([a-z0-9_]+)"|([A-Z][A-Z0-9_]*))\s*=>\s*((?:[a-z_][A-Za-z0-9_]*::)?[a-z_][A-Za-z0-9_]*)\(request\)/g;
  for (const match of sources.get("mlx.rs").matchAll(dispatch)) {
    const provider = match[1] ?? consts.get(match[2]);
    if (!provider || !bodies.has(match[3])) continue;
    if (!armsFor.has(provider)) armsFor.set(provider, new Set());
    armsFor.get(provider).add(match[3]);
  }
  // Not vacuous: the parse must find the whole shipped dispatch, not a handful of arms.
  assert.ok(armsFor.size >= 40, `the dispatch parse found only ${armsFor.size} provider arms`);

  const reaches = (entry) => {
    const seen = new Set();
    const stack = [entry];
    while (stack.length > 0) {
      const name = stack.pop();
      if (seen.has(name)) continue;
      seen.add(name);
      const body = bodies.get(name);
      if (body === undefined) continue;
      if (body.includes('"sourceCapture"')) return true;
      for (const call of body.matchAll(RUST_CALL)) if (bodies.has(call[1])) stack.push(call[1]);
    }
    return false;
  };
  const emitting = [...armsFor]
    .filter(([, entries]) => [...entries].some(reaches))
    .map(([provider]) => provider)
    .sort();

  const declared = Object.entries(PROVIDER_FAMILIES)
    .filter(([, family]) => family.sourceCapture)
    .map(([id, family]) => family.provider ?? id)
    .sort();
  assert.deepEqual(
    declared,
    emitting,
    "a declared arm that emits nothing makes the harness refuse the render, and an emitting arm " +
      "left undeclared dies on `required environment variable SCENEWORKS_MEMORY_CAPTURE_DIR is not set`",
  );

  // Every emitting arm must also declare the MLX lane, since that is the only lane this walk reads.
  for (const [id, family] of Object.entries(PROVIDER_FAMILIES)) {
    if (!family.sourceCapture) continue;
    assert.ok(family.arms.includes("mlx"), `${id}: sourceCapture is an MLX-arm fact`);
  }
  // `physical` is the narrower currency rule and cannot hold without the receipt being emitted.
  for (const [id, family] of Object.entries(PROVIDER_FAMILIES)) {
    if (family.physical) assert.equal(family.sourceCapture, true, `${id}: physical implies sourceCapture`);
  }
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
  // A lane's SOURCE SET, not just its bin: an arm may live in a `#[path]` sibling module
  // (`mlx_ltx25.rs`, `mlx_wan_scail2.rs`, `candle_wan_scail2.rs`), and reading only `<backend>.rs`
  // would report a real arm's env family as unread. The sibling files are exactly the ones the bin
  // pulls in, and the directory holds nothing else.
  const binDir = path.join(ROOT, "crates/sceneworks-memory-adapter/src/bin");
  const lanes = new Map();
  const binFiles = (await readdir(binDir)).filter((name) => name.endsWith(".rs"));
  for (const backend of ["mlx", "candle"]) {
    const members = binFiles.filter(
      (name) => name === `${backend}.rs` || name.startsWith(`${backend}_`),
    );
    assert.ok(members.length >= 2, `${backend}: expected the bin plus its arm modules`);
    lanes.set(
      backend,
      (await Promise.all(members.map((name) => readFile(path.join(binDir, name), "utf8")))).join(
        "\n",
      ),
    );
  }
  let checked = 0;
  for (const [id, family] of Object.entries(PROVIDER_FAMILIES)) {
    // The LTX-2.5 families carry no env family at all: the harness binds their snapshot root
    // directly (`--ltx25-snapshot-root`), which `classifyAnchor` asserts by leaving `row.env` empty.
    const envFamilies = [
      family.env,
      family.upstream?.env,
      // sc-22733: the shared component snapshot a split-layout family stages its text encoder and
      // VAE from is exported the same way and must be read back the same way.
      family.components?.env,
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
      // Per (LANE, TIER), through the one function that already knows the rule (sc-22736). A family
      // may bind a different artifact on each arm — the two SANA routes do, MLX opening the packed
      // `SceneWorks/*_mlx` turnkey through `SCENEWORKS_SANA_*` and Candle the upstream dense
      // diffusers snapshot through `SCENEWORKS_SANA_DENSE_*` — and, since the Wan 2.2 family, on
      // each (lane, TIER): its candle q4/q8 open a `SceneWorks/…-candle` rehost while its candle
      // bf16 opens the upstream `Wan-AI/…-Diffusers` checkpoint. Asserting the shared `family.env`
      // against candle.rs would demand env names that binary must never read, and asserting only
      // the lane override would miss the second family entirely.
      //
      // `familyArtifact` is the production resolver, so this cannot drift from what a capture
      // actually exports: a new override shape is covered the moment that function honours it.
      const laneEnvFamilies = [
        ...new Set(
          ["bf16", "q4", "q8"]
            .map((tier) => familyArtifact(family, arm, tier).env)
            .filter(Boolean),
        ),
        ...envFamilies.filter((env) => env !== family.env),
      ];
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
  await stageLtx25SnapshotEntries(hub);
  const expected = snapshotPath(hub, "SceneWorks/ltx-2.5-mlx", REVISION);
  for (const [backend, provider] of [["mlx", "ltx_2_5"], ["candle", "ltx_2_5_distilled"]]) {
    const context = { models: fakeModels(), backend, hubs: [hub], current: new Map(), captured: new Map() };
    for (const tier of ["q4", "q8", "bf16"]) {
      const row = await classifyAnchor(`ltx_2_5:${tier}:${backend}`, { provider }, context);
      assert.equal(row.status, "runnable", `${backend} ${tier}: ${row.reason}`);
      assert.equal(row.ltx25SnapshotRoot, expected, "the harness is handed the snapshot, not a tier root");
      assert.deepEqual(row.env, {}, "LTX-2.5 carries no adapter env family; the harness binds it");
      assert.equal(row.physical, false);
      // sc-22738: the MLX arm emits a `physical_mlx` sourceCapture on every run and the Candle arm
      // emits none, so the raw-log pair is owed to one lane and refused on the other.
      assert.equal(row.sourceCapture, backend === "mlx", `${backend} ${tier}`);
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
  // sc-22727 gave `flux2_dev` a Candle arm, sc-22729 gave the whole `sdxl` family one and sc-22737
  // gave MiniMax-H3 one, so the no-arm probe moved again — to `ltx_2_3`, whose Candle lane loads
  // under its OWN engine id (`ltx_2_3_distilled`) and so is declared `arms: ["mlx"]`. Asserted
  // rather than assumed, so this probe cannot silently stop asking its question the next time a
  // lane is added.
  assert.deepEqual(PROVIDER_FAMILIES.ltx_2_3.arms, ["mlx"], "the no-arm probe needs an mlx-only family");
  const noArm = await classifyAnchor("ltx_2_3:q4:candle", { provider: "ltx_2_3" }, base);
  assert.equal(noArm.status, "no_adapter_arm");
  assert.match(noArm.reason, /candle adapter implements no provider arm for ltx_2_3/);
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
  const tuple = [
    "    (",
    `        "${relative}",`,
    `        include_str!("../../../${relative}"),`,
    "    ),",
  ].join("\n");
  assert.ok(once.includes(`${tuple}\n`), "the tuple keeps its rustfmt shape");
  assert.equal(once.length - source.length, tuple.length + 1);
  // sc-22738: IN SORTED POSITION, because `memory_anchor.rs` asserts the compiled-in list stays
  // sorted. A blind append only happened to hold while every new corpus landed under
  // `docs/generated/`; a `docs/calibration/` one — which is where every campaign's corpora go —
  // sorts before all of those, and appending it reds `cargo test -p sceneworks-core` on a commit
  // the runner has already made.
  const paths = [...once.matchAll(/^ {8}"([^"]+)",$/gm)].map((match) => match[1]);
  assert.deepEqual(paths, [...paths].sort(), "the list is still sorted after the insert");
  assert.ok(paths.includes(relative));
  // A corpus that DOES sort last still lands last.
  const last = "zz/last-of-all.json";
  const lastPaths = [...appendPackagedSource(source, last).matchAll(/^ {8}"([^"]+)",$/gm)].map(
    (match) => match[1],
  );
  assert.equal(lastPaths.at(-1), last);
  assert.throws(() => appendPackagedSource("const OTHER: &[u8] = &[];", relative), /no longer declares/);
  // sc-22738: idempotence is decided INSIDE the list. A mention of the path ELSEWHERE in the file
  // — a doc comment, a test fixture that cites the corpus it exercises — used to read as "already
  // packaged", so the append silently did nothing and every store row derived from that corpus
  // then failed its own handshake. This story's adapter fixture names the corpus it packages, which
  // is exactly how the defect surfaced.
  const mentioned = `${source}\n// see ${JSON.stringify(relative)} for the retained evidence\n`;
  assert.ok(
    appendPackagedSource(mentioned, relative).includes(
      `include_str!("../../../${relative}")`,
    ),
    "a mention outside the list must not suppress the append",
  );
});

/** A throwaway checkout carrying just what `writePackagedSource` reads: the file and rustfmt's config. */
async function packagedSourceFixture() {
  const root = await mkdtemp(path.join(tmpdir(), "catalog-packaged-fmt-"));
  await mkdir(path.join(root, path.dirname(PACKAGED_SOURCES_PATH)), { recursive: true });
  const source = await readFile(path.join(ROOT, PACKAGED_SOURCES_PATH), "utf8");
  await writeFile(path.join(root, PACKAGED_SOURCES_PATH), source);
  await writeFile(
    path.join(root, "rustfmt.toml"),
    await readFile(path.join(ROOT, "rustfmt.toml"), "utf8"),
  );
  return { root, source, file: path.join(root, PACKAGED_SOURCES_PATH) };
}

// sc-22738: the one-line tuple the append writes fits `max_width` only while the corpus name is
// short. `flux2-dev-bf16-mlx-exceeded-evidence.json` makes the `include_str!` line 101 columns, so
// the runner committed a tree whose `cargo fmt --check` reds the `parity-rust` lane — after the
// push. Every shorter name in the campaign happened to fit, which is why it surfaced this late.
test("packaging a long-named corpus leaves memory_anchor.rs rustfmt-clean", async () => {
  const { root, source, file } = await packagedSourceFixture();
  const relative = "docs/calibration/sc-99999/flux2-dev-bf16-mlx-exceeded-evidence.json";
  assert.ok(
    `        include_str!("../../../${relative}"),`.length > 100,
    "the fixture path must be long enough to exceed max_width on one line",
  );
  assert.equal(await writePackagedSource(root, relative), true);
  // The gate itself, not a re-implementation of its width rule.
  await execFileAsync("rustfmt", ["--edition", RUSTFMT_EDITION, "--check", file]);
  const written = await readFile(file, "utf8");
  assert.ok(
    written.includes(
      ["        include_str!(", `            "../../../${relative}"`, "        ),"].join("\n"),
    ),
    "rustfmt wrapped the argument onto its own line",
  );
  // The mutation this guards: the append's own output — the format step skipped — is what red the
  // lane, so `rustfmt --check` must reject it.
  await writeFile(file, appendPackagedSource(source, relative));
  await assert.rejects(
    execFileAsync("rustfmt", ["--edition", RUSTFMT_EDITION, "--check", file]),
    "an unformatted append must not pass the parity lane's check",
  );
});

test("packaging a short-named corpus is byte-identical to the bare append", async () => {
  const { root, source, file } = await packagedSourceFixture();
  const relative = "docs/calibration/sc-99999/qwen-image-q4-mlx-evidence.json";
  assert.equal(await writePackagedSource(root, relative), true);
  assert.equal(await readFile(file, "utf8"), appendPackagedSource(source, relative));
  // Already packaged: nothing is rewritten, so nothing is reformatted either.
  assert.equal(await writePackagedSource(root, relative), false);
});

test("the edition the runner formats with is the one rustfmt.toml declares", async () => {
  const config = await readFile(path.join(ROOT, "rustfmt.toml"), "utf8");
  assert.equal(config.match(/^edition\s*=\s*"([^"]+)"$/m)?.[1], RUSTFMT_EDITION);
});

// sc-22738: the embed and the Docker COPY lines are ONE step. `platform-review-contracts.test.mjs`
// ("Rust Docker builders copy every production generated embed from sceneworks-core") requires each
// `include_str!` in `memory_anchor.rs` to be copied into BOTH builder stages, and packaging a corpus
// without them reds that suite on a commit the runner has already made.
test("a new evidence corpus is copied into both Rust builder stages, by campaign directory", async () => {
  const dockerfile = await readFile(path.join(ROOT, DOCKERFILE_PATH), "utf8");
  const relative = "docs/calibration/sc-22738/aaa-first-mlx-evidence.json";
  const line = "COPY docs/calibration/sc-22738/ ./docs/calibration/sc-22738/";
  // The campaign is already carried, so an ingest into it rewrites the Dockerfile not at all — the
  // property that keeps the layer count flat no matter how many anchors a campaign lands.
  assert.equal(dockerfile.split(`\n${line}\n`).length - 1, 2, "one line per builder stage");
  assert.equal(insertEvidenceCopy(dockerfile, relative), dockerfile, "an ingest adds no layer");
  // A campaign with no line yet gets exactly one per stage, beside the other calibration lines.
  const fresh = "docs/calibration/sc-99999/x-evidence.json";
  const freshLine = "COPY docs/calibration/sc-99999/ ./docs/calibration/sc-99999/";
  const once = insertEvidenceCopy(dockerfile, fresh);
  const lines = once.split("\n");
  assert.equal(lines.filter((text) => text === freshLine).length, 2, "one line per builder stage");
  assert.equal(insertEvidenceCopy(once, fresh), once, "idempotent");
  assert.equal(
    insertEvidenceCopy(once, "docs/calibration/sc-99999/y-evidence.json"),
    once,
    "a second corpus in that campaign adds nothing",
  );
  for (const index of lines.flatMap((text, at) => (text === freshLine ? [at] : []))) {
    assert.ok(lines[index - 1].startsWith("COPY docs/calibration/"), "it lands beside the other calibration corpora");
    assert.ok(lines[index + 1].startsWith("COPY "), "inserted inside the stage's COPY run, never after it");
  }
  assert.throws(
    () => insertEvidenceCopy(`FROM rust AS builder\n${DOCKERFILE_EMBED_ANCHOR}\nRUN cargo build\n`, relative),
    /not the two builder stages/,
    "a Dockerfile whose two stages cannot be found reds the run instead of committing an uncopied embed",
  );
  // Only a campaign directory collapses. `docs/generated/` also holds the churning
  // `memory-matrix.json`, so collapsing IT would rebuild the whole Rust graph on selector edits.
  assert.throws(
    () => insertEvidenceCopy(dockerfile, "docs/generated/memory-calibration-evidence.json"),
    /not a docs\/calibration/,
    "a non-campaign path is refused rather than collapsed into a directory COPY",
  );
});

// sc-22738. Docker's overlay driver refuses a stage past ~125 layers, and the failure is a
// `max depth exceeded` while PREPARING the build — no compile error, no bad line to find. The
// per-corpus COPY form reached 141 instructions in `builder` and 143 in `candle-builder` on this
// campaign alone. 100 is the ceiling this file's directory-COPY form must keep every stage under.
test("no Rust Dockerfile stage approaches Docker's overlay layer limit", async () => {
  const dockerfile = await readFile(path.join(ROOT, DOCKERFILE_PATH), "utf8");
  const stages = new Map();
  let stage = null;
  for (const line of dockerfile.split("\n")) {
    const from = /^FROM\s+(\S+)(?:\s+AS\s+(\S+))?/.exec(line);
    if (from) {
      stage = from[2] ?? from[1];
      if (!stages.has(stage)) stages.set(stage, 0);
      continue;
    }
    if (stage !== null && /^(RUN|COPY|ADD)\s/.test(line)) stages.set(stage, stages.get(stage) + 1);
  }
  assert.ok(stages.has("builder") && stages.has("candle-builder"), "both Rust builder stages parse");
  for (const [name, count] of stages) {
    assert.ok(count < 100, `stage ${name} declares ${count} layer instructions, over the 100 ceiling`);
  }
  // The mechanism that keeps it there: evidence is copied by campaign directory, never per corpus.
  const perCorpus = dockerfile
    .split("\n")
    .filter((line) => /^COPY docs\/calibration\/[^/ ]+\/[^ ]+ /.test(line));
  assert.deepEqual(perCorpus, [], "calibration evidence is copied by directory, one layer per campaign");
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

// sc-22738. A bound's bundle has an EMPTY `records` array and the matrix publishes only anchors,
// so neither of the two indexes `classifyAnchor` consulted could see a cell the host had already
// proved it cannot finish: `--list` said `runnable`, and the next `--skip-current` re-run would
// book the same guarded render again (76 minutes for the bernini bf16 stop). Currency is the same
// rule a record's is, so a bound that stales stops classifying and the cell is capturable again.
test("a CURRENT exceeded bound is a captured cell; a stale bound leaves the anchor runnable", async () => {
  const storeRoot = await mkdtemp(path.join(tmpdir(), "catalog-bounds-"));
  await mkdir(path.join(storeRoot, "config"), { recursive: true });
  const digest = "b".repeat(64);
  const bound = {
    modelId: "bernini", tier: "bf16", backend: "mlx", observedFootprintBytes: 97147294328,
    source: {
      path: "docs/calibration/sc-22738/bernini-bf16-mlx-exceeded-evidence.json",
      loaderClosureDigest: digest,
    },
  };
  const seed = async (rows) => {
    await writeFile(path.join(storeRoot, ANCHOR_STORE_PATH), JSON.stringify({ anchors: [], exceededBounds: rows }));
    await writeFile(path.join(storeRoot, ANCHOR_LOADER_CONFIG_PATH), JSON.stringify({ models: { "bernini:mlx": { digest } } }));
  };
  await seed([bound]);
  const currentBounds = await readExceededBounds(storeRoot);
  assert.deepEqual([...currentBounds.keys()], ["bernini:bf16:mlx"]);
  assert.equal(currentBounds.get("bernini:bf16:mlx").current, true);
  assert.equal(currentBounds.get("bernini:bf16:mlx").source, bound.source.path);
  await seed([{ ...bound, source: { ...bound.source, loaderClosureDigest: "c".repeat(64) } }]);
  assert.equal((await readExceededBounds(storeRoot)).get("bernini:bf16:mlx").current, false, "a bound measured under another loader closure is stale");
  await seed([{ ...bound, backend: "candle" }]);
  assert.equal(
    (await readExceededBounds(storeRoot)).get("bernini:bf16:candle").current, false,
    "the closure digest is looked up per (model, LANE); the mlx digest does not keep a candle bound current",
  );
  assert.equal((await readExceededBounds(await mkdtemp(path.join(tmpdir(), "catalog-empty-")))).size, 0);
  // sc-22738: a cell with a stale bound BESIDE a current one is exceeded_current whichever row sorts
  // last — the adapter's seam consults only the current bounds, and this index must agree with it.
  const stale = { ...bound, source: { ...bound.source, loaderClosureDigest: "c".repeat(64) } };
  for (const rows of [[bound, stale], [stale, bound]]) {
    await seed(rows);
    const indexed = await readExceededBounds(storeRoot);
    assert.equal(indexed.size, 1);
    assert.equal(indexed.get("bernini:bf16:mlx").current, true, `order ${rows.indexOf(bound)}: any current bound keeps the cell exceeded_current`);
  }
  await seed([stale, { ...stale, observedFootprintBytes: 1 }]);
  assert.equal((await readExceededBounds(storeRoot)).get("bernini:bf16:mlx").current, false, "two stale bounds stay stale");

  const hub = await fakeHub([["SceneWorks/z-image-turbo-mlx", REVISION, "q4"]]);
  const base = { models: fakeModels(), backend: "candle", hubs: [hub], current: new Map(), captured: new Map() };
  const key = "z_image_turbo:q4:candle";
  const bounded = await classifyAnchor(key, { provider: "z_image_turbo" }, {
    ...base,
    bounds: new Map([[key, { current: true, source: "docs/calibration/sc-1/z-exceeded-evidence.json" }]]),
  });
  assert.equal(bounded.status, "exceeded_current");
  assert.equal(bounded.current, true);
  assert.match(bounded.reason, /exceeded bound recorded at the pinned inference revision/);
  assert.match(bounded.reason, /docs\/calibration\/sc-1\/z-exceeded-evidence\.json/);
  const staleRow = await classifyAnchor(key, { provider: "z_image_turbo" }, {
    ...base,
    bounds: new Map([[key, { current: false, source: "docs/calibration/sc-1/z-exceeded-evidence.json" }]]),
  });
  assert.equal(staleRow.status, "runnable", "a stale bound classifies nothing");
  assert.equal(
    (await classifyAnchor(key, { provider: "z_image_turbo" }, base)).status, "runnable",
    "a cell with no bound at all is untouched",
  );
});

// The same question against the REAL committed store, through the walk that schedules the captures:
// a cell whose bound is current is never in the runnable set — with or without `--skip-current`,
// because there is nothing left for a re-run to establish.
test("the committed exceeded bound keeps its own cell out of the runnable set of a real --list", async () => {
  const bounds = await readExceededBounds();
  assert.ok(bounds.size > 0, "the shipped store carries at least one measured lower bound");
  for (const skipCurrent of [false, true]) {
    const { rows } = await planRun({ backend: "mlx", anchors: null, campaign: "sc-catalog-test", hfCache: [], skipCurrent, models: [] });
    for (const [key, bound] of bounds) {
      const row = rows.find((candidate) => candidate.key === key);
      if (!row || row.backend !== "mlx") continue;
      if (bound.current) {
        assert.equal(row.status, "exceeded_current", `${key} carries a current bound`);
        assert.equal(row.current, true);
      } else {
        assert.notEqual(row.status, "exceeded_current", `${key}'s bound is stale, so the cell is capturable again`);
      }
    }
    assert.ok(
      !rows.filter((row) => row.status === "runnable").some((row) => bounds.get(row.key)?.current),
      "no cell with a current bound is scheduled for a capture",
    );
  }
});

test("every provider the committed plan declares is either served by a family row or refused by name", async () => {
  const plan = await readPlan();
  const models = await readManifestModels();
  for (const backend of ["mlx", "candle"]) {
    const { rows } = await planRun({ backend, anchors: null, campaign: "sc-catalog-test", hfCache: [], skipCurrent: false });
    const keys = Object.keys(plan.anchors).filter((key) => key.endsWith(`:${backend}`));
    assert.deepEqual(rows.map((row) => row.key), keys.sort(), `${backend}: one row per plan anchor`);
    for (const row of rows) {
      // sc-22738: `exceeded_current` is the seventh — a cell whose measured lower bound is current
      // is answered by the store rather than by a root, so it resolves none (the branch below
      // excludes it for the same reason `lane_undeclared` is excluded).
      assert.ok(["runnable", "weights_missing", "no_adapter_arm", "harness_unsupported", "lane_undeclared", "provider_undeclared", "exceeded_current"].includes(row.status), `${row.key}: ${row.status}`);
      if (row.status === "lane_undeclared") assert.match(row.reason, /anchor-loader-closures\.json/);
      if (row.status === "provider_undeclared") assert.match(row.reason, /inference-provider-closures\.json/);
      // sc-22729/sc-22734: the family is resolved by the SAME rule classification uses — model-keyed when
      // the row declares the plan's provider, provider-keyed otherwise — so this coverage check
      // cannot drift from what `--list` actually did.
      const family = familyFor(row.modelId, row.provider);
      if (row.status === "no_adapter_arm") {
        assert.equal(family?.arms.includes(backend) ?? false, false);
      } else if (!["harness_unsupported", "lane_undeclared", "provider_undeclared", "exceeded_current"].includes(row.status)) {
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
 * - ROUTED lane: `routedLanes()` (scripts/check-tier-integrity.mjs) over the ROUTING CATALOG
 *   itself — `crates/sceneworks-core/src/jobs_store/routing/{catalog,candle,mlx}.rs`. A lane that
 *   does not route the model is the ONLY exemption; a "structurally N/A" matrix cell is not one
 *   (epic 22723 E1).
 *
 *   **Not `models[].backends` from `docs/generated/memory-matrix.json` (sc-22738).** That column is
 *   the same `routedLanes()` oracle, but restricted to the matrix's own universe, and the matrix
 *   subtracts `OUT_OF_MATRIX_CATALOG_ENTRIES` (`minimax_h3`, `minimax_h3_ref`) before emitting
 *   `models` — a SECOND exemption source, which E1 admits exactly one of. Reading the matrix let
 *   all twelve MiniMax-H3 cells leave the denominator for a fact about the GENERATOR's parser
 *   (`minimax_h3_engine_id` is a prefix predicate it cannot enumerate) rather than about routing:
 *   the catalog routes both entries on both lanes (`VideoModelCaps::new("minimax_h3", true, true,
 *   …)`), so deleting their twelve plan rows left this test green. Reading the routing catalog
 *   directly removes the second exemption instead of documenting it; `the denominator's routed-lane
 *   oracle is the routing catalog, not the matrix` asserts the two agree everywhere the matrix has
 *   an opinion, so this is a widening and never a divergence.
 * - ROUTED tier: the per-lane rule below, narrowed further by `parseBackendTierOverrides` — and
 *   NOTHING else. See `computeShippedTieredCells`.
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

/**
 * Whether this lane's host fetches an untiered TIER BUNDLE for `model` — one repository whose tier
 * subdirectories live inside it, so it serves every tier the model ships.
 *
 * **A co-requisite is not a bundle (sc-22737).** This test used to be `variant` alone, and that let
 * a model's SIDE artifact decide its tier axis: LTX-2.3's only untiered download is its dense Gemma
 * text encoder (`coRequisite: true`), which the Candle host does fetch — and which contains no tier
 * subdirectory at all. That claimed `ltx_2_3:bf16:candle`, a cell no lane can open: the manifest
 * ships LTX-2.3's `bf16` download as `platforms: ["macos"]`, and the worker's Candle tier resolver
 * (`crates/sceneworks-worker/src/video_jobs/candle.rs#candle_ltx_bundle_tier_across_revisions`)
 * returns `None` for `CandleLtxTier::Bf16` — there is no dense off-Mac tier to resolve.
 *
 * Excluding co-requisites is the same filter `shipped` already applies to the tiered rows, and it is
 * NARROW: across the whole catalog it changes exactly that one cell (every other model whose
 * untiered downloads are all co-requisites reaches its tiers through the downloads' own `platforms`
 * instead). `SceneWorks/bernini` — a real off-Mac bundle carrying q4/q8/bf16 subdirs and no
 * `coRequisite` flag — is unaffected, which is what keeps Bernini's six Candle cells claimed.
 */
function laneHasTierBundle(model, backend) {
  return (model.downloads ?? []).some(
    (download) =>
      typeof download.variant !== "string" &&
      !download.coRequisite &&
      downloadServesLane(download, backend),
  );
}

const ROUTING_SOURCES = Object.freeze({
  routingCatalog: "crates/sceneworks-core/src/jobs_store/routing/catalog.rs",
  routingCandle: "crates/sceneworks-core/src/jobs_store/routing/candle.rs",
  routingMlx: "crates/sceneworks-core/src/jobs_store/routing/mlx.rs",
});

let routedCatalogLanesPromise;
/**
 * `Map<modelId, Set<lane>>` — the lane-existence oracle, read off the ROUTING CATALOG (sc-22738).
 *
 * The same `routedLanes()` the matrix generator itself calls, applied to the whole catalog instead
 * of to the matrix's post-subtraction universe. Fails closed: `routedLanes` throws nothing but
 * returns no entry for an id nothing routes, and every consumer here treats an absent id as "no
 * lane", which is E1's one exemption and is asserted to be a routing fact by the case below.
 */
function routedCatalogLanes() {
  routedCatalogLanesPromise ??= (async () =>
    routedLanes(
      Object.fromEntries(
        await Promise.all(
          Object.entries(ROUTING_SOURCES).map(async ([key, relative]) => [
            key,
            await readFile(path.join(ROOT, relative), "utf8"),
          ]),
        ),
      ),
    ))();
  return routedCatalogLanesPromise;
}

/** The lanes the routing catalog serves `modelId` on, in a stable order. */
function lanesOf(routed, modelId) {
  const served = routed.get(modelId) ?? new Set();
  return ["mlx", "candle"].filter((backend) => served.has(backend));
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

async function computeShippedTieredCells(models = null) {
  models ??= await readManifestModels();
  const routed = await routedCatalogLanes();
  const laneTiers = await routeLaneTiers();
  const overrides = await codeDerivedTierOverrides();
  const cells = [];
  const dropped = [];
  for (const model of models) {
    const shipped = (model.downloads ?? []).filter(
      (download) => !download.coRequisite && ["q4", "q8", "bf16"].includes(download.variant),
    );
    if (shipped.length === 0) continue;
    for (const backend of lanesOf(routed, model.id)) {
      // An untiered NON-co-requisite download this lane's host fetches is a BUNDLE whose tiers live
      // inside it, so it serves every tier the model ships — `SceneWorks/bernini` is that repo. See
      // `laneHasTierBundle` for why a co-requisite is not one.
      const bundled = laneHasTierBundle(model, backend);
      const floor = laneTiers.get(`${backend}:${model.id}`) ?? new Set();
      const tiers = [...new Set(
        shipped
          .filter(
            (download) =>
              bundled || downloadServesLane(download, backend) || floor.has(download.variant),
          )
          .map((download) => download.variant),
      )];
      // sc-22729: a shipped tier that this lane CAN fetch is still only a CELL if the lane's CODE
      // can load that tier at all. `instantid_realvisxl` ships q4/q8/bf16 but its candle stack is
      // dense-only and always loads `bf16/` (`image_jobs/instantid.rs`
      // instantid_memory_backend_keys / instantid_tier_subdir on the non-macOS branch), so a q4
      // candle anchor could only ever measure bf16 weights and file the peaks under a packed tier.
      //
      // The narrowing source is `parseBackendTierOverrides` — the SAME worker-source derivation the
      // matrix generator uses — and deliberately NOT the matrix's `axes.<backend>.tiers`. That list
      // falls back to `model.<backend>.vramGbByTier`, a MEASUREMENT declaration: a missing key there
      // means "no peak recorded yet", which is precisely the gap this set exists to count. Reading
      // it here let the manifest delete six real cells (flux_dev / flux_schnell / flux2_dev candle
      // bf16; sd3_5_large / sd3_5_large_turbo / sd3_5_medium candle q8) with no routing fact behind
      // it — see `the gap-set denominator is narrowed only by code-derived tier overrides`.
      const override = overrides.get(`${model.id}:${backend}`);
      for (const tier of tiers) {
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
async function measurabilityGaps() {
  measurabilityGapsPromise ??= computeMeasurabilityGaps();
  return (await measurabilityGapsPromise).gaps;
}

/**
 * sc-22738: the gaps excused because the PINNED `mlx-gen-wan` loader seals no memory receipt for
 * the route, so its loaded provider publishes no memory-strategy contract and no change in this
 * repository can make the cell capturable (fixed engine-side in SceneWorks/inference#964).
 *
 * Not an allowlist: the excuse is re-derived from the same pinned loader `classifyAnchor` reads, so
 * it empties itself at the pin bump. `every shipped tiered model is measurable` asserts it stays
 * bounded to exactly that.
 */
async function pinnedLoaderGaps() {
  measurabilityGapsPromise ??= computeMeasurabilityGaps();
  return (await measurabilityGapsPromise).pinned;
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
    // sc-22738: `exceeded_current` joins the two. The gap set counts cells the campaign cannot
    // REACH — a missing arm, an undeclared lane, an unbindable artifact — and a cell whose current
    // measured lower bound is already in the store is the opposite of unreached: it has evidence,
    // and re-running it could only re-establish the same inequality. A bound that stales puts the
    // cell straight back into `runnable`, so the gap set still sees it the moment it is capturable.
    if (!["runnable", "weights_missing", "exceeded_current"].includes(status)) {
      gaps.push({ ...cell, status, provider: row?.provider ?? null, reason: row?.reason ?? `${PLAN_PATH} declares no anchor ${cell.key}` });
    }
  }
  const sealed = await readWanMlxSealedProviders();
  const pinned = sealed === null
    ? []
    : gaps.filter((gap) => gap.provider !== null && gap.reason === wanMlxSealGap(gap.provider, sealed));
  return { gaps: gaps.filter((gap) => !pinned.includes(gap)), pinned, sealed };
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

// sc-22738 (feature-end round 1). The denominator's LANE axis, and E1's one exemption.
//
// Three claims, all mechanical:
//
//  1. Everywhere the matrix has an opinion, its `models[].backends` column equals the routing
//     catalog's answer — so moving the denominator off the matrix (see `computeShippedTieredCells`)
//     widened the universe and changed nothing else. A drift here means the generator's own
//     `routedLanes` join has moved and one of the two readers is stale.
//  2. Every `OUT_OF_MATRIX_CATALOG_ENTRIES` id the catalog still ships is ROUTED on at least one
//     lane and IS in the denominator. The matrix subtracts them for a fact about its parser, and
//     that is not an E1 exemption; this is the assertion that makes the subtraction unable to
//     silence a cell.
//  3. The general rule: a tiered manifest model is missing from the denominator on EVERY lane only
//     when the routing catalog routes it nowhere. Nothing else may remove a model.
test("the denominator's routed-lane oracle is the routing catalog, not the matrix", async () => {
  const routed = await routedCatalogLanes();
  const matrix = JSON.parse(await readFile(path.join(ROOT, MATRIX_PATH), "utf8"));
  assert.ok(matrix.models.length > 0, "the matrix still publishes models");
  for (const model of matrix.models) {
    assert.deepEqual(
      model.backends ?? [],
      lanesOf(routed, model.id),
      `${model.id}: the matrix's routed lanes disagree with the routing catalog`,
    );
  }

  const models = await readManifestModels();
  const byId = new Map(models.map((model) => [model.id, model]));
  const universe = new Set((await shippedTieredCells()).map((cell) => cell.modelId));
  for (const [id, entry] of OUT_OF_MATRIX_CATALOG_ENTRIES) {
    if (!byId.has(id)) continue; // `assertOutOfMatrixEntriesAreStillUnroutable` owns the stale case.
    assert.ok(
      !matrix.models.some((model) => model.id === id),
      `${id} is subtracted from the matrix universe (epic ${entry.epic})`,
    );
    assert.ok(
      lanesOf(routed, id).length > 0,
      `${id} is subtracted from the matrix but the routing catalog routes it nowhere`,
    );
    assert.ok(
      universe.has(id),
      `${id} is routed and ships tiers, but the burndown denominator does not ask about it`,
    );
  }

  // The rule itself, over the whole catalog.
  const absent = [];
  for (const model of models) {
    const tiered = (model.downloads ?? []).some(
      (download) => !download.coRequisite && ["q4", "q8", "bf16"].includes(download.variant),
    );
    if (!tiered) continue;
    if (!universe.has(model.id) && lanesOf(routed, model.id).length > 0) absent.push(model.id);
  }
  assert.deepEqual(
    absent,
    [],
    "a tiered manifest model may leave the denominator on every lane ONLY because the routing " +
      "catalog routes it nowhere (epic 22723 E1)",
  );
});

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
  // The loop above is vacuous over an empty set; the worker source narrows at least one lane's
  // shipped tier axis today (`instantid_realvisxl` on candle), so it must have had something to
  // check. Which cells drop is the override parser's answer, not a list kept here.
  assert.ok(dropped.length > 0, "no shipped cell was dropped, so the override rule was never exercised");

  // …and the six cells a manifest-declaration intersection would have deleted are all present.
  const universe = new Set(cells.map((cell) => cell.key));
  for (const key of MANIFEST_ONLY_DECLARED_CELLS) {
    assert.ok(universe.has(key), `${key} left the denominator with no routing fact behind it`);
  }

  // The shape rule behind those six: the denominator is INDEPENDENT of every `vramGbByTier`
  // measurement declaration. Recomputed over a manifest with every lane's `vramGbByTier` stripped,
  // the universe must be the live one, cell for cell — a derivation that consulted the declaration
  // (as the matrix's `tiersFor` fallback does) would lose exactly the cells with no peak recorded.
  const stripped = (await readManifestModels()).map((model) => {
    const copy = { ...model };
    for (const backend of ["candle", "mlx"]) {
      if (copy[backend]?.vramGbByTier === undefined) continue;
      const { vramGbByTier: _dropped, ...rest } = copy[backend];
      copy[backend] = rest;
    }
    return copy;
  });
  const recomputed = await computeShippedTieredCells(stripped);
  assert.deepEqual(
    recomputed.cells.map((cell) => cell.key).sort(),
    cells.map((cell) => cell.key).sort(),
    "the denominator reads a vramGbByTier measurement declaration somewhere",
  );
  assert.deepEqual(
    recomputed.dropped.map((drop) => drop.key).sort(),
    dropped.map((drop) => drop.key).sort(),
    "the override drops read a vramGbByTier measurement declaration somewhere",
  );
});

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
  const routed = await routedCatalogLanes();
  const laneTiers = await routeLaneTiers();
  const overrides = await codeDerivedTierOverrides();
  for (const model of models) {
    const shipped = (model.downloads ?? []).filter(
      (download) => !download.coRequisite && ["q4", "q8", "bf16"].includes(download.variant),
    );
    for (const backend of lanesOf(routed, model.id)) {
      const bundled = laneHasTierBundle(model, backend);
      const floor = laneTiers.get(`${backend}:${model.id}`) ?? new Set();
      // sc-22729: and the lane's own code must be able to LOAD the tier — see
      // `codeDerivedTierOverrides`, the only other narrowing this denominator admits.
      const override = overrides.get(`${model.id}:${backend}`);
      for (const tier of new Set(shipped.map((download) => download.variant))) {
        const serves =
          (bundled ||
            floor.has(tier) ||
            shipped.some(
              (download) => download.variant === tier && downloadServesLane(download, backend),
            )) &&
          (!override || override.includes(tier));
        assert.equal(
          keys.has(`${model.id}:${tier}:${backend}`),
          serves,
          `${model.id}:${tier}:${backend} is claimed iff something on that lane can open the tier`,
        );
      }
    }
  }
});

// sc-22737. The ONE cell of this story's table that is deliberately not claimed, and the reason,
// asserted against the two sources that decide it rather than against a hand-kept exemption list.
//
// `ltx_2_3:bf16:candle` is an unrouted (lane, tier), which is epic 22723 E1's one exemption:
//
//   1. the manifest ships LTX-2.3's `bf16` download as `platforms: ["macos"]`, so no Candle host
//      ever fetches it, and LTX-2.3's only untiered download is a co-requisite rather than a tier
//      bundle (see `laneHasTierBundle`);
//   2. the worker resolves no dense off-Mac tier —
//      `video_jobs/candle.rs#candle_ltx_bundle_tier_across_revisions` returns `None` for
//      `CandleLtxTier::Bf16`, while its q4 and q8 arms return a subdirectory.
//
// Both halves are read here, so the exemption cannot outlive either reason: shipping a Candle-served
// bf16 download, or teaching the worker to resolve one, turns this case RED and demands the cell.
test("ltx_2_3 claims q4 and q8 on candle, and bf16 only on mlx", async () => {
  const models = await readManifestModels();
  const ltx = models.find((model) => model.id === "ltx_2_3");
  assert.ok(ltx, "the catalog still carries ltx_2_3");

  // (1) The manifest half.
  const bf16 = (ltx.downloads ?? []).filter(
    (download) => download.variant === "bf16" && !download.coRequisite,
  );
  assert.equal(bf16.length, 1, "ltx_2_3 ships exactly one bf16 row");
  assert.equal(
    downloadServesLane(bf16[0], "candle"),
    false,
    "ltx_2_3's bf16 download is macOS-only; a Candle-served one would make the cell real",
  );
  assert.equal(
    laneHasTierBundle(ltx, "candle"),
    false,
    "ltx_2_3's only untiered download is a co-requisite, not a tier bundle",
  );

  // (2) The worker half, read off the source that decides it.
  const worker = await readFile(
    path.join(ROOT, "crates/sceneworks-worker/src/video_jobs/candle.rs"),
    "utf8",
  );
  const resolver = worker.match(
    /fn candle_ltx_bundle_tier_across_revisions\([\s\S]*?\n\}/,
  )?.[0];
  assert.ok(resolver, "the Candle LTX tier resolver must still be findable");
  assert.match(
    resolver,
    /CandleLtxTier::Bf16 => return None/,
    "the Candle lane resolves no dense LTX-2.3 tier; if it does now, bf16:candle is a real cell",
  );
  for (const tier of ["q4", "q8"]) {
    assert.ok(resolver.includes(`"${tier}"`), `the Candle lane resolves the ${tier} tier subdir`);
  }

  // And therefore the claimed key set is exactly five cells.
  assert.deepEqual(
    (await shippedTieredCells())
      .filter((cell) => cell.modelId === "ltx_2_3")
      .map((cell) => cell.key)
      .sort(),
    [
      "ltx_2_3:bf16:mlx",
      "ltx_2_3:q4:candle",
      "ltx_2_3:q4:mlx",
      "ltx_2_3:q8:candle",
      "ltx_2_3:q8:mlx",
    ],
  );
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

// sc-22729: the SDXL family — five catalog models the worker routes onto ONE engine id (`sdxl`)
// plus the bespoke `instantid` route — on every shipped tier of every lane that routes them.
//
// Every cell is DECLARED — a plan anchor and both closure declarations on both lanes. Two facts
// keep some of them from being capturable, both engine-side rather than adapter gaps, and both
// asserted here so neither can silently widen:
//   * `instantid_realvisxl` q4/q8 on candle are not CELLS at all — the candle InstantID stack is
//     dense-only, which the worker's own `instantid.rs` says, so the universe never produces them.
//   * any candle cell whose `SDXL_ROUTES` revision is not the one this repository ships cannot be
//     SEALED by the engine. At inference c6d6a4db that was all three tiers of
//     `illustrious_xl_v1`/`v2`; `ea4118b6a` (sc-22729, inference-side) repinned those routes, so at
//     the pin this branch carries the set is EMPTY. Nothing about that set is written down here —
//     it is derived from the pinned engine source against the manifest on every run, so it
//     repopulates by itself the moment a pin reintroduces a divergence.
const SDXL_FAMILY = [
  "sdxl", "realvisxl", "realvisxl_lightning", "illustrious_xl_v1", "illustrious_xl_v2",
  "instantid_realvisxl",
];

/**
 * Every candle cell of `cells` the engine's own route table blocks at the pinned inference
 * checkout, derived exactly as `classifyAnchor` derives it: the `sdxlRoute` families only, on the
 * candle lane only, refused only when `sdxlCandleRouteDrift` says so. With no checkout to read,
 * nothing is refused — that is `SDXL_ROUTES_UNCHECKED`, not a gap.
 */
async function sdxlCandleRouteDriftCells(cells) {
  const routes = await readSdxlCandleRoutes();
  if (!routes) return [];
  const [plan, models] = [await readPlan(), await readManifestModels()];
  return cells
    .filter((cell) => cell.backend === "candle"
      && familyFor(cell.modelId, plan.anchors[cell.key]?.provider)?.sdxlRoute
      && sdxlCandleRouteDrift(cell.modelId, cell.tier, routes, models) !== null)
    .map((cell) => cell.key);
}

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
  // Every family cell is measurable except the candle routes the engine cannot seal — and with no
  // inference checkout to read, there are none of those either, because the refusal is DERIVED.
  // Both sides of the comparison are read, so neither mode can quietly assert nothing.
  const gaps = (await measurabilityGaps()).filter((gap) => SDXL_FAMILY.includes(gap.modelId));
  assert.deepEqual(
    gaps.map((gap) => gap.key).sort(),
    (await sdxlCandleRouteDriftCells(cells)).sort(),
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

// sc-22729 review: the exclusion is DERIVED from both revisions, never written down. Three
// directions, none of which depends on which way the pinned tree happens to sit: every route the
// engine declares is compared (at this pin they all agree, so nothing is excluded), a route one
// revision off refuses every shipped tier, and a route the engine drops is refused for THAT reason.
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

// sc-22736, pin 8a65db2a: the inference-side repair landed, so the LIVE answer for every routed
// model is now "no refusal". The claim this test makes was never "Illustrious drifts" — it was that
// the refusal TRACKS the two revisions in both directions. So the live tree is asserted to agree
// everywhere, and the refusing direction is exercised by SYNTHESIZING a disagreement on a real
// route rather than by depending on one model still being broken upstream.
test("the sdxl candle refusal is derived from the engine's own route revision", { skip: skipWithoutRoutes }, async () => {
  const models = await readManifestModels();
  const routes = await readSdxlCandleRoutes();
  assert.ok(routes.size >= 5, "the engine declares the whole routed family");
  for (const [modelId, route] of routes) {
    // An engine that ships what the manifest ships refuses nothing…
    const shipped = tierDownload(models, modelId, route.repository, "q4").revision;
    const agreed = new Map(routes).set(modelId, { ...route, revision: shipped });
    assert.equal(sdxlCandleRouteDrift(modelId, "q4", agreed, models), null, `${modelId}: equal revisions must clear it`);
    // …and one revision off refuses every shipped tier of that model, with no edit to this
    // repository either way. Both directions are synthetic, so the case proves the DERIVATION
    // rather than whichever way the pinned engine happens to sit — that is `the sdxl family is
    // measurable…`'s job, and it reads the same two sources.
    const drifted = new Map(routes).set(modelId, { ...route, revision: REVISION });
    for (const tier of ["q4", "q8", "bf16"]) {
      assert.match(sdxlCandleRouteDrift(modelId, tier, drifted, models), /candle-gen-sdxl pins route/, `${modelId}:${tier}`);
    }
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

// sc-22738. `wan_2_2_t2v_14b:{bf16,q4,q8}:mlx` classified `runnable` and every booked capture died
// AFTER the load (273 s for bf16, 114 s for q4) on "loaded wan2_2_t2v_14b exposed no memory-strategy
// contract". Nothing the classifier read could see it: the anchor loader closure names
// `i2v_memory_strategy.rs` as the memory-strategy entry point for BOTH A14B routes, and that file
// registers the two identically. The divergence is in the loader one file over.
test("a Wan MLX route whose pinned loader seals no receipt is not measurable, and an unreadable checkout refuses nothing", async () => {
  // The two loader shapes, verbatim in the part that matters: one seals, one does not.
  const source = `
pub const MODEL_ID_T2V_14B: &str = "wan2_2_t2v_14b";
pub const MODEL_ID_I2V_14B: &str = "wan2_2_i2v_14b";
pub fn load_t2v_14b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // A comment naming crate::i2v_memory_strategy::prepare(spec, MODEL_ID_T2V_14B) is prose.
    Ok(Box::new(Wan14b { i2v_memory: None }))
}
pub fn load_i2v_14b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let i2v_memory = Some(crate::i2v_memory_strategy::prepare(spec, MODEL_ID_I2V_14B)?);
    Ok(Box::new(Wan14b { i2v_memory }))
}
`;
  const sealed = parseWanMlxSealedProviders(source);
  assert.deepEqual([...sealed], ["wan2_2_i2v_14b"], "a commented-out seal is prose, not a seal");
  assert.equal(wanMlxSealGap("wan2_2_i2v_14b", sealed), null);
  assert.match(wanMlxSealGap("wan2_2_t2v_14b", sealed), /seals no memory receipt for wan2_2_t2v_14b/);
  // Fail-closed both ways: an unresolvable provider const, and a source that seals nothing at all,
  // THROW rather than reporting every Wan MLX route as unsealed on a parser failure.
  assert.throws(
    () => parseWanMlxSealedProviders("crate::i2v_memory_strategy::prepare(spec, MODEL_ID_MYSTERY)"),
    /declares no `const … : &str` value for/,
  );
  assert.throws(() => parseWanMlxSealedProviders("fn load_t2v_14b() {}"), /for no route at all/);

  // Live: the pinned checkout's own loader, and the classification it produces for the T2V cell.
  const live = await readWanMlxSealedProviders();
  assert.equal(await readWanMlxSealedProviders(path.join(ROOT, "no", "such", "checkout")), null);
  assert.equal(await readWanMlxSealedProviders(""), null, "no inference checkout is not a refusal");
  const plan = await readPlan();
  const key = "wan_2_2_t2v_14b:q4:mlx";
  const models = await readManifestModels();
  const context = (wanMlxSealed) => ({
    models,
    backend: "mlx",
    hubs: [path.join(ROOT, "no", "such", "hub")],
    current: new Map(),
    captured: new Map(),
    declaredLanes: new Set(["wan_2_2_t2v_14b:mlx", "wan_2_2_i2v_14b:mlx"]),
    declaredProviders: new Set(["mlx:wan2_2_t2v_14b", "mlx:wan2_2_i2v_14b"]),
    wanMlxSealed,
  });
  // With nothing to read, the cell classifies as it otherwise would and SAYS the check did not run.
  const unchecked = await classifyAnchor(key, plan.anchors[key], context(null));
  assert.equal(unchecked.routeCheck, WAN_MLX_SEAL_UNCHECKED);
  assert.notEqual(unchecked.status, "harness_unsupported");
  // An unsealed T2V loader is a refusal that names the missing contract, and it is SCOPED: the I2V
  // route on the same family marker, and the T2V route once its loader seals, stay measurable.
  const unsealed = new Set(["wan2_2_ti2v_5b", "wan2_2_i2v_14b"]);
  const refused = await classifyAnchor(key, plan.anchors[key], context(unsealed));
  assert.equal(refused.status, "harness_unsupported");
  assert.match(refused.reason, /exposed no memory-strategy contract/);
  const i2vKey = "wan_2_2_i2v_14b:q4:mlx";
  const sibling = await classifyAnchor(i2vKey, plan.anchors[i2vKey], context(unsealed));
  assert.notEqual(sibling.status, "harness_unsupported", "the refusal is per route, not per family");
  const fixed = await classifyAnchor(key, plan.anchors[key], context(new Set([...unsealed, "wan2_2_t2v_14b"])));
  assert.notEqual(fixed.status, "harness_unsupported", "a pin whose loader seals re-admits the cell");

  // And the fact itself at the pin this repository ships, so a bump that fixes it is visible here.
  if (live === null) {
    assert.ok(!process.env.CI, `on CI the pinned inference checkout must supply ${WAN_MLX_LOADER_PATH}`);
  } else {
    assert.ok(live.has("wan2_2_i2v_14b"), "the I2V loader has always sealed; a parser that reads it as unsealed is broken");
    assert.ok(live.has("wan2_2_ti2v_5b"), "the TI2V-5B loader has always sealed");
  }
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
  const gaps = (await measurabilityGaps()).filter((gap) => family.includes(gap.modelId));
  assert.equal(gaps.length, 0, gapReport(gaps));
});

// sc-22733. Same shape claim, for the Mage-Flow family: SIX registered engine providers (three
// text-to-image checkpoints and three instruction editors), each with its own tiered rehost, all
// sharing ONE text-encoder/VAE components snapshot. Key sets, never a frozen count.
test("the mage-flow family is measurable on every shipped tier of every routed lane", async () => {
  const family = [
    "mage_flow", "mage_flow_base", "mage_flow_turbo",
    "mage_flow_edit", "mage_flow_edit_base", "mage_flow_edit_turbo",
  ];
  const cells = (await shippedTieredCells()).filter((cell) => family.includes(cell.modelId));
  assert.equal(cells.length, 6 * 3 * 2, "six models x three shipped tiers x two routed lanes");
  const gaps = (await measurabilityGaps()).filter((gap) => family.includes(gap.modelId));
  assert.equal(gaps.length, 0, gapReport(gaps));
});

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

// sc-22735. The Krea family widening. `krea_2_raw` is the undistilled base of the SAME two engine
// crates Turbo rides, off its own tiered rehost, on BOTH lanes; `krea_realtime_14b` is the
// autoregressive VIDEO member, MLX-only. The lane axis is DERIVED from the matrix here rather than
// asserted as a count per model, so a routing change moves the expectation with it: the assertion
// is that every routed lane of every shipped tier is measurable, whichever lanes those are.
test("the krea family is measurable on every shipped tier of every routed lane", async () => {
  const family = ["krea_2_turbo", "krea_2_raw", "krea_realtime_14b"];
  const cells = (await shippedTieredCells()).filter((cell) => family.includes(cell.modelId));
  const lanes = new Map();
  for (const cell of cells) lanes.set(cell.modelId, (lanes.get(cell.modelId) ?? new Set()).add(cell.backend));
  assert.deepEqual(
    [...lanes].map(([id, backends]) => [id, [...backends].sort()]).sort(),
    [
      ["krea_2_raw", ["candle", "mlx"]],
      ["krea_2_turbo", ["candle", "mlx"]],
      // MLX-only by routing, not by omission: `mlx-gen-krea-realtime` is the only engine that
      // registers this provider and every shipped download is `platforms: ["macos"]`.
      ["krea_realtime_14b", ["mlx"]],
    ],
    "the matrix routes the two image members on both lanes and the video member on mlx alone",
  );
  assert.equal(cells.length, 3 + 3 * 2 + 3 * 2, "three shipped tiers per routed lane, per member");
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

// sc-22735. `krea_realtime_14b` declares ONE adapter arm, and the AC that scopes this story to MLX
// rests on three independent facts that could each move on their own. This case pins all three so
// the day any of them changes, the family table is what fails rather than a capture booked against
// an arm that does not exist:
//
//  * the CATALOG: every shipped download is macOS-only, so no non-Mac host can install a tier;
//  * the ROUTING: the memory matrix — derived from the worker's own route resolvers — lists `mlx`
//    and nothing else for this id;
//  * the TABLE: `PROVIDER_FAMILIES` declares the `mlx` arm alone, so `classifyAnchor` reports
//    `no_adapter_arm` rather than routing a candle plan row at an arm the candle adapter lacks.
//
// A manifest that ever shipped a windows/linux download for this id, or a route resolver that ever
// gave it a candle lane, reds this case rather than silently widening the measurable surface.
test("krea_realtime_14b is an MLX-only lane in the catalog, the routing and the family table", async () => {
  const models = await readManifestModels();
  const realtime = models.find((model) => model.id === "krea_realtime_14b");
  assert.ok(realtime, "the manifest still ships krea_realtime_14b");
  const platforms = [...new Set((realtime.downloads ?? []).flatMap((download) => download.platforms ?? []))].sort();
  assert.deepEqual(platforms, ["macos"], "every krea_realtime_14b download is macOS-only");
  const matrix = JSON.parse(await readFile(path.join(ROOT, MATRIX_PATH), "utf8"));
  const routed = matrix.models.find((model) => model.id === "krea_realtime_14b");
  assert.deepEqual(routed?.backends ?? [], ["mlx"], "the worker routes krea_realtime_14b on mlx alone");
  assert.deepEqual(PROVIDER_FAMILIES.krea_realtime_14b.arms, ["mlx"]);
  // And the refusal itself: a candle plan row for this provider is classified as a missing arm,
  // naming the backend and the provider, rather than being silently served by another family.
  const candle = await classifyAnchor("krea_realtime_14b:q4:candle", { provider: "krea_realtime_14b" }, {
    models, backend: "candle", hubs: [], current: new Map(), captured: new Map(),
  });
  assert.equal(candle.status, "no_adapter_arm");
  assert.match(candle.reason, /candle adapter implements no provider arm for krea_realtime_14b/);
});

// sc-22734. Same shape claim, for the SenseNova-U1 family: six catalog models on TWO engine ids
// (`sensenova_u1_8b` and its 8-step distill `sensenova_u1_8b_fast`), each shipping its own
// independently pinned tiered rehost, all six routed on both lanes.
const SENSENOVA_FAMILY = Object.freeze([
  "sensenova_u1_8b",
  "sensenova_u1_8b_infographic_v2",
  "sensenova_u1_8b_infographic_v3",
  "sensenova_u1_8b_fast",
  "sensenova_u1_8b_infographic_v2_fast",
  "sensenova_u1_8b_infographic_v3_fast",
]);

test("the sensenova family is measurable on every shipped tier of every routed lane", async () => {
  const family = SENSENOVA_FAMILY;
  const cells = (await shippedTieredCells()).filter((cell) => family.includes(cell.modelId));
  assert.equal(cells.length, family.length * 3 * 2, "six routes x three shipped tiers x two routed lanes");
  const gaps = (await measurabilityGaps()).filter((gap) => family.includes(gap.modelId));
  assert.equal(gaps.length, 0, gapReport(gaps));
});

// The JS table and the Rust adapters are two spellings of ONE binding. A drift between them sends a
// capture at the wrong artifact family — the infographic ids all ride a shared engine id, so a wrong
// env would silently load the base SenseNova rehost and re-label its peaks — and nothing downstream
// would notice, because the record would be well-formed. Bound here rather than trusted.
test("every sensenova env family and repository is the one the adapter binaries actually read", async () => {
  const lib = await readFile(path.join(ROOT, ADAPTER_LIB_PATH), "utf8");
  const repositories = new Map(
    [...lib.matchAll(/pub const ([A-Z0-9_]+_REPOSITORY): &str =\s*"([^"]+)";/g)].map((match) => [match[2], match[1]]),
  );
  const binaries = {
    mlx: await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/bin/mlx.rs"), "utf8"),
    candle: await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/bin/candle.rs"), "utf8"),
  };
  let checked = 0;
  for (const id of SENSENOVA_FAMILY) {
    const family = PROVIDER_FAMILIES[id];
    assert.ok(family, `${id} has no PROVIDER_FAMILIES row`);
    for (const backend of family.arms) {
      assert.ok(
        repositories.has(family.repo),
        `${id}:${backend}: ${family.repo} is not a *_REPOSITORY const in ${ADAPTER_LIB_PATH}`,
      );
      for (const suffix of ["REPOSITORY", "REVISION", "ROOT"]) {
        const name = `SCENEWORKS_${family.env}_${suffix}`;
        assert.ok(
          binaries[backend].includes(`"${name}"`),
          `${id}:${backend}: the ${backend} adapter never reads ${name}`,
        );
      }
      checked += 1;
    }
  }
  assert.equal(checked, SENSENOVA_FAMILY.length * 2, "six routes, two lanes each");
});

/**
 * sc-22734. The anchor composition is fixed per lane by the harness (`ANCHOR_STRATEGY`), and a plan
 * row may override it only where the provider CONTRACT refuses the lane default: SenseNova
 * classifies `StagedResidency` as `StructurallyNotApplicable` on both lanes, so
 * `contract.validate_selection` would reject the candle default rung before any weight is read.
 *
 * The override is therefore DERIVED, not curated: the manifest's own
 * `<lane>.memoryStrategyStructuralExemptions` is the architecture evidence, and the rule below
 * reads the whole plan against it — a row's effective rung must be the lane default UNLESS the
 * manifest exempts that default, in which case it must be `resident`. On the candle lane that is
 * exactly "resident if and only if `staged_residency` is exempted"; on MLX the default is already
 * resident, so an exemption moves nothing. No list of model ids and no count lives here: a newly
 * exempt model, or a withdrawn exemption, moves the requirement on its own.
 */
test("an anchor row plans the lane's default rung unless the manifest exempts it", async (t) => {
  const plan = await readPlan();
  const models = new Map((await readManifestModels()).map((model) => [model.id, model]));
  // sc-22736: the SECOND derived source. A provider whose contract simply does not IMPLEMENT the
  // lane default — Candle SCAIL-2 publishes `Resident` alone, `Missing` rather than structurally
  // inapplicable, so no manifest exemption is honest — is read off the checked-in engine
  // capability dump, which records every contract's `implementedRungs` per (tier, load shape) at
  // the pin. Absence from the dump is NOT evidence: a row that overrides a provider the dump does
  // not know falls through to the third source below, and is refused if that cannot speak either.
  const dumps = new Map();
  for (const backend of ["mlx", "candle"]) {
    const dump = JSON.parse(
      await readFile(path.join(ROOT, `config/engine-capabilities/capabilities.${backend}.json`), "utf8"),
    );
    dumps.set(backend, new Map(dump.memoryContracts.map((contract) => [contract.id, contract])));
  }
  const dumpRefusesDefault = (backend, anchor, tier, fallbackRung) => {
    const contract = dumps.get(backend).get(anchor.provider);
    if (!contract) return null;
    const loadShape = anchor.loadShape;
    const surfaces = contract.surfaces.filter(
      (surface) => surface.selector.tier === tier && surface.selector.loadShape === loadShape,
    );
    if (surfaces.length === 0) return null;
    return surfaces.every((surface) => !surface.implementedRungs.includes(fallbackRung));
  };
  // sc-22736, pin 8a65db2a: the THIRD derived source, consulted only where the dump has no surface
  // to offer. `candle-gen-scail2` registers a memory strategy WITHOUT a weights-free surface
  // resolver, so `memory_contract_surfaces()` — and therefore capabilities.candle.json — never sees
  // it; no regeneration at any pin would change that. Its `build_contract` does declare the support
  // per rung, and the anchor loader closure already names that file as this (model, lane)'s
  // memory-strategy entry point, derived at the pin and `--check`ed. See
  // `readDeclaredStrategySupport`: an unreadable `strategies:` shape THROWS, and a provider that
  // declares nothing yields null, which is refused below.
  const closures = JSON.parse(await readFile(path.join(ROOT, "config/anchor-loader-closures.json"), "utf8"));
  const declaredRefusesDefault = async (backend, modelId, fallbackRung) => {
    const support = await readDeclaredStrategySupport(modelId, backend, closures);
    return support === null ? null : support(fallbackRung) !== "Implemented";
  };
  let overridden = 0;
  let candleExempt = 0;
  let declaredEvidence = 0;
  for (const [key, anchor] of Object.entries(plan.anchors)) {
    const { modelId, tier, backend } = anchorParts(key);
    const model = models.get(modelId);
    // A plan row for a model the manifest does not ship cannot be judged against manifest evidence;
    // `validatePlan` already refuses an invented model id, so this only skips fixture rows.
    if (!model) continue;
    const exemptions = model[backend]?.memoryStrategyStructuralExemptions ?? {};
    const manifestExempt = Object.hasOwn(exemptions, "staged_residency");
    const fallbackForDump = ANCHOR_STRATEGY[backend].rung;
    const judged = !manifestExempt && fallbackForDump === "staged_residency";
    let contractRefusesDefault = judged ? dumpRefusesDefault(backend, anchor, tier, fallbackForDump) : false;
    if (judged && contractRefusesDefault === null) {
      const declared = await declaredRefusesDefault(backend, modelId, fallbackForDump);
      if (declared !== null) declaredEvidence += 1;
      contractRefusesDefault = declared;
    }
    // Two different absences. A reachable checkout that declares nothing is REFUSED below. No
    // checkout at all is a property of the RUN, not of the override: the module-level guard above
    // already turns that into a hard failure on CI (where check.yml supplies the clone), so locally
    // it degrades to a diagnostic rather than a red that says nothing about this repository.
    if (anchor.strategy && judged && contractRefusesDefault === null && !sdxlRoutesAvailable) {
      t.diagnostic(
        `${key}: not judged — no pinned inference checkout ($INFERENCE_REPO) to read the engine's ` +
          "own strategy declaration from. On CI this run would already have failed.",
      );
      continue;
    }
    if (anchor.strategy && judged) {
      assert.notEqual(
        contractRefusesDefault,
        null,
        `${key}: overrides the lane default, but config/engine-capabilities/capabilities.${backend}.json ` +
          `records no ${anchor.provider} contract surface at ${tier}/${anchor.loadShape} AND the engine's own ` +
          `memory-strategy entry point for ${modelId}:${backend} (config/anchor-loader-closures.json) declares ` +
          "no per-rung support that can be read at the pinned inference checkout. Set $INFERENCE_REPO to a " +
          "checkout of the pin (check.yml's parity-scaffold job does), or drop the override: an override with " +
          "no derived architecture evidence is refused, never assumed.",
      );
    }
    const exempt = manifestExempt || contractRefusesDefault === true;
    // The exemption is scoped to the overlays it names, and an anchor renders exactly one.
    if (manifestExempt) {
      assert.ok(
        (exemptions.staged_residency.overlays ?? []).includes(anchor.overlay),
        `${key}: the manifest exempts staged_residency only for overlays ` +
          `${JSON.stringify(exemptions.staged_residency.overlays)}, not ${JSON.stringify(anchor.overlay)}`,
      );
    }
    const fallback = ANCHOR_STRATEGY[backend];
    const expected = exempt && fallback.rung === "staged_residency"
      ? { rung: "resident", engagedRungs: ["resident"] }
      : { rung: fallback.rung, engagedRungs: [...fallback.engagedRungs] };
    const effective = anchor.strategy ?? fallback;
    assert.equal(
      effective.rung,
      expected.rung,
      `${key}: manifest ${backend}.memoryStrategyStructuralExemptions ${manifestExempt ? "declares" : "does not declare"} ` +
        `staged_residency and the derived contract evidence ${contractRefusesDefault ? "records the contract refusing" : "does not record the contract refusing"} ` +
        `it, so the anchor must plan rung ${JSON.stringify(expected.rung)}`,
    );
    assert.deepEqual([...effective.engagedRungs], expected.engagedRungs, `${key}: engaged rung set`);
    if (anchor.strategy) overridden += 1;
    if (exempt && backend === "candle") candleExempt += 1;
  }
  assert.ok(overridden > 0, "the plan exercises the override at least once");
  assert.ok(candleExempt > 0, "at least one candle row is exempted from the lane's default rung");
  // …and the third source is genuinely load-bearing today, not dead code kept warm: the SCAIL-2
  // candle rows have no dump surface and are legitimate only because the engine's own declaration
  // is read. A count, not a list — a provider that grows a dump surface simply lowers it.
  if (sdxlRoutesAvailable) {
    assert.ok(
      declaredEvidence > 0,
      "no override rested on the engine's own strategy declaration; if every provider now carries a " +
        "dump surface, delete readDeclaredStrategySupport rather than leaving an unexercised source",
    );
  }
});

/**
 * sc-22738. The composition a row plans and the CONTROLS it carries are one claim, and the
 * campaign's first full MLX walk proved they were never checked together: `planAnchor` emits
 * `strategy.parameters: {}` for every row, while `crates/…/bin/mlx_ltx25.rs` demanded
 * `attentionChunkSize` unconditionally, so all three `ltx_2_5:*:mlx` cells refused before a weight
 * was read — the arm still spoke epic 18755's ladder sweep, which sc-22505 replaced with one anchor
 * at the lane default rung.
 *
 * The law, both directions, derived from the manifest rather than a list: the request the harness
 * builds must carry EXACTLY the parameters the model's own lane block declares for the rungs its
 * planned composition engages, and nothing for a rung it does not engage. `resident` and
 * `staged_residency` declare no controls, so today every row is `{}` — and a row that ever plans a
 * parameterized rung must ship those parameters or this reds. The engine enforces the same
 * symmetry (`attention_chunk_size requires chunk_attention=true`), which is why volunteering a
 * control is as wrong as omitting one.
 */
test("a planned anchor carries exactly the controls its engaged rungs declare", async () => {
  const plan = await readPlan();
  const models = new Map((await readManifestModels()).map((model) => [model.id, model]));
  // The wire spells the component lowercase; the manifest spells the engine's enum variant.
  const wire = (parameter, value) =>
    parameter === "transformerWindowComponent" ? String(value).toLowerCase() : value;
  let checked = 0;
  for (const key of Object.keys(plan.anchors)) {
    const { modelId, backend } = anchorParts(key);
    const model = models.get(modelId);
    if (!model) continue;
    const declared = model[backend]?.memoryStrategyCapabilities ?? {};
    const planned = planAnchor(plan, key);
    const expected = {};
    for (const rung of planned.strategy.engagedRungs) {
      for (const [parameter, value] of Object.entries(declared[rung]?.parameters ?? {})) {
        expected[parameter] = wire(parameter, value);
      }
    }
    assert.deepEqual(
      planned.strategy.parameters,
      expected,
      `${key}: plans rung ${planned.strategy.rung} engaging ${JSON.stringify(planned.strategy.engagedRungs)}, ` +
        `so the manifest's ${backend}.memoryStrategyCapabilities requires exactly ${JSON.stringify(expected)}`,
    );
    checked += 1;
  }
  assert.ok(checked > 0, "no plan row was judged against the manifest's declared controls");
});

// sc-22736. The third source read three ways: the shipped shape yields per-rung support, a file
// with no declaration yields null (which the rule above refuses), and a declaration this parser
// cannot read THROWS. That last case is the one that matters — the cheap failure mode for a
// source-text derivation is to silently stop matching and re-admit every override.
test("the engine's strategy declaration is parsed per rung, and an unreadable shape is refused", async () => {
  const shipped = `
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: if strategy == MemoryStrategy::Resident {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                },
                parameters: MemoryParameterRanges::default(),
            })
            .collect(),
  `;
  const support = parseDeclaredStrategySupport(shipped, "fixture.rs");
  assert.equal(support("resident"), "Implemented");
  assert.equal(support("staged_residency"), "Missing");
  assert.equal(support("bounded_decode"), "Missing");
  // No declaration at all: null, never "implements everything".
  assert.equal(parseDeclaredStrategySupport("fn build_contract() {}", "fixture.rs"), null);
  // MUTATION: the same file after an upstream refactor this parser does not know. It must throw —
  // returning null here would make the override rule fall back to "no evidence" and, worse,
  // returning a default would admit it.
  assert.throws(
    () => parseDeclaredStrategySupport("    strategies: strategies(spec),\n", "fixture.rs"),
    /cannot\s+read/,
  );
  // And the live wiring: candle SCAIL-2's real declaration, when a pinned checkout is reachable.
  const closures = JSON.parse(await readFile(path.join(ROOT, "config/anchor-loader-closures.json"), "utf8"));
  const live = await readDeclaredStrategySupport("scail2_14b", "candle", closures);
  if (live === null) {
    assert.ok(!process.env.CI, "on CI the pinned inference checkout must supply the SCAIL-2 declaration");
  } else {
    assert.equal(live("resident"), "Implemented");
    assert.equal(live("staged_residency"), "Missing");
  }
  // A model the closure does not declare has no entry point to read, so it yields null.
  assert.equal(await readDeclaredStrategySupport("not_a_model", "candle", closures), null);
});

// sc-22737. The inline `if strategy == … { } else { }` above is ONE of the shapes shipped at the
// pin, and the story's own crate does not use it: `candle-gen-bernini` hoists the declaration into a
// file-local `fn strategies()` whose `support:` is a `match strategy { … }` with `|`-alternatives.
// The parser refused that, so the override rule could not say which rungs Bernini implements and the
// whole suite died on the refusal. These pin the shapes that actually ship, and every one of them is
// read out of a fixture that mirrors the real crate — not the crate itself, so a re-slice upstream
// reds `the engine's own declaration parses for every provider the override rule can reach` below
// rather than silently rewriting what this test claims.
test("the strategy declaration parses through a file-local helper and a match arm", () => {
  // `candle-gen-bernini/src/memory_strategy.rs`, verbatim in shape: a helper call, a match, and a
  // multi-variant `|` arm. Bernini implements resident and bounded decode; staged residency is
  // Missing, which is exactly why its candle anchor rows plan `resident`.
  const bernini = `
fn strategies() -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident | MemoryStrategy::BoundedDecode => {
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::StagedResidency => MemoryStrategySupport::Missing,
                MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency => {
                    MemoryStrategySupport::Missing
                }
            },
            parameters: MemoryParameterRanges::default(),
        })
        .collect()
}

fn build_contract() -> MemoryContract {
    MemoryContract {
        strategies: strategies(),
    }
}
`;
  const support = parseDeclaredStrategySupport(bernini, "bernini.rs");
  assert.equal(support("resident"), "Implemented");
  assert.equal(support("staged_residency"), "Missing");
  assert.equal(support("bounded_decode"), "Implemented");
  assert.equal(support("bounded_attention"), "Missing");
  assert.equal(support("bounded_transformer_residency"), "Missing");

  // MUTATION: mangle the shape the parser was taught — the helper is no longer a `fn` in this file,
  // so the call resolves to nothing. It must refuse BY NAME rather than fall back to "no evidence".
  assert.throws(
    () => parseDeclaredStrategySupport(bernini.replace("fn strategies()", "fn strategies_v2()"), "bernini.rs"),
    /bernini\.rs .*cannot\s+read/s,
  );
  // MUTATION: the match itself replaced by an expression this parser has not been taught.
  assert.throws(
    () =>
      parseDeclaredStrategySupport(
        bernini.replace(/support: match strategy \{[\s\S]*?\n            \},/, "support: support_for(strategy),"),
        "bernini.rs",
      ),
    /bernini\.rs .*cannot\s+read/s,
  );

  // Prose is not code: `candle-gen-minimax-h3` carries paragraphs of `//` commentary between the
  // arms, mentioning the very variants and supports the parser is scanning for. A scanner that reads
  // comments reads the wrong arm — and would report rung 1 Implemented when it is deliberately
  // under-declared Missing.
  const commented = `
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: match strategy {
                    MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
                    // MemoryStrategy::StagedResidency => MemoryStrategySupport::Implemented, one day
                    // Rung 1 is implemented in code but stays Missing here: no behavior seam.
                    _ => MemoryStrategySupport::Missing,
                },
                parameters: MemoryParameterRanges::default(),
            })
            .collect(),
  `;
  const minimax = parseDeclaredStrategySupport(commented, "minimax.rs");
  assert.equal(minimax("resident"), "Implemented");
  assert.equal(minimax("staged_residency"), "Missing");

  // A struct-variant support carries a payload and still names its variant.
  const notApplicable = parseDeclaredStrategySupport(
    `
        strategies: MemoryStrategy::ALL.into_iter().map(|strategy| MemoryStrategyCapability {
                strategy,
                support: match strategy {
                    MemoryStrategy::BoundedAttention => {
                        MemoryStrategySupport::StructurallyNotApplicable { reason: reason() }
                    }
                    _ => MemoryStrategySupport::Implemented,
                },
                parameters: MemoryParameterRanges::default(),
            }).collect(),
  `,
    "na.rs",
  );
  assert.equal(notApplicable("bounded_attention"), "StructurallyNotApplicable");
  assert.equal(notApplicable("resident"), "Implemented");

  // A rung the source text does not DECIDE is refused rather than guessed. `mlx-gen-minimax-h3`
  // gates rung 4 on `streamable`, and `candle-gen-sdxl` gates staged residency on the surface; both
  // are runtime values. Picking either side would state a rung's support as a fact.
  const guarded = parseDeclaredStrategySupport(
    `
        strategies: MemoryStrategy::ALL.into_iter().map(|strategy| MemoryStrategyCapability {
                strategy,
                support: match strategy {
                    MemoryStrategy::BoundedTransformerResidency if streamable => {
                        MemoryStrategySupport::Implemented
                    }
                    MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
                    _ => MemoryStrategySupport::Implemented,
                },
                parameters: MemoryParameterRanges::default(),
            }).collect(),
  `,
    "guarded.rs",
  );
  assert.equal(guarded("resident"), "Implemented");
  assert.throws(() => guarded("bounded_transformer_residency"), /runtime condition/);

  // The same for the if/else form with a compound condition (`candle-gen-sdxl`): the rungs the
  // condition decides statically still resolve, and only the gated one refuses.
  const surfaceGated = parseDeclaredStrategySupport(
    `
        strategies: MemoryStrategy::ALL.into_iter().map(|strategy| MemoryStrategyCapability {
                strategy,
                support: if strategy == MemoryStrategy::BoundedTransformerResidency
                    || (surface == SdxlSurface::Bespoke
                        && strategy == MemoryStrategy::StagedResidency)
                {
                    MemoryStrategySupport::Missing
                } else {
                    MemoryStrategySupport::Implemented
                },
                parameters: MemoryParameterRanges::default(),
            }).collect(),
  `,
    "sdxl.rs",
  );
  assert.equal(surfaceGated("bounded_transformer_residency"), "Missing");
  assert.equal(surfaceGated("resident"), "Implemented");
  assert.equal(surfaceGated("bounded_decode"), "Implemented");
  assert.throws(() => surfaceGated("staged_residency"), /runtime condition|not about the strategy/);
});

// The fixtures above are shapes; this is the pin. EVERY memory-strategy entry point the anchor
// loader closure names must parse at 8a65db2a — not just the handful the override rule happens to
// reach today, because which provider it reaches moves whenever a dump surface appears or a plan row
// changes, and a parse refusal is a hard failure at that moment rather than a diagnostic.
test("the engine's own declaration parses for every provider the override rule can reach", async (t) => {
  const closures = JSON.parse(await readFile(path.join(ROOT, "config/anchor-loader-closures.json"), "utf8"));
  if (!process.env.INFERENCE_REPO) {
    assert.ok(!process.env.CI, "on CI the pinned inference checkout must be reachable ($INFERENCE_REPO)");
    t.diagnostic("no pinned inference checkout ($INFERENCE_REPO); the per-provider parse was not run");
    return;
  }
  let read = 0;
  let declared = 0;
  for (const key of Object.keys(closures.models ?? {})) {
    const [modelId, backend] = key.split(":");
    // A refusal throws out of here with the offending path in its message, which is the whole point.
    const support = await readDeclaredStrategySupport(modelId, backend, closures);
    read += 1;
    if (support === null) continue;
    declared += 1;
    // Every rung must come back as a named support or as the DELIBERATE ambiguity refusal — never
    // as "this shape cannot be read", which is the failure mode that broke the suite. `candle-gen-
    // sdxl` and `mlx-gen-krea` genuinely gate a rung on a runtime value, and for those the dump is
    // the source that answers; what must not happen is the parser losing the shape entirely.
    for (const rung of ["resident", "staged_residency", "bounded_decode", "bounded_attention", "bounded_transformer_residency"]) {
      try {
        assert.match(support(rung), /^[A-Z]\w+$/, `${key}: ${rung} support`);
      } catch (error) {
        assert.doesNotMatch(
          error.message,
          /cannot\s+read/s,
          `${key}: ${rung} is refused as an unreadable shape rather than answered or named ambiguous`,
        );
        assert.match(error.message, /runtime condition|not about the strategy/, `${key}: ${rung}`);
      }
    }
  }
  assert.ok(read > 50, `the closure names ${read} (model, lane) pairs; expected the full catalog`);
  assert.ok(declared > 0, "no closure entry point declares a strategies: field at all");
});

// Each Mage variant binds TWO artifact triples: its OWN tiered rehost (never a sibling's — the six
// checkpoints are architecturally identical, so a crossed root would be caught by nothing else) and
// the ONE shared components snapshot. This asserts the derivation produces exactly that.
test("every mage-flow member binds its own rehost plus the one shared components snapshot", async () => {
  const models = await readManifestModels();
  const seenComponents = new Set();
  for (const [modelId, family] of Object.entries(PROVIDER_FAMILIES)) {
    if (!modelId.startsWith("mage_flow")) continue;
    assert.deepEqual(family.arms, ["mlx", "candle"], `${modelId} is routed on both lanes`);
    assert.equal(family.components, MAGE_COMPONENTS, `${modelId} shares the one components row`);
    // The variant repo is this member's own, and it is the repo the manifest ships the tiers from.
    const download = tierDownload(models, modelId, family.repo, "q4");
    assert.equal(download.variant, "q4", `${modelId} ships a q4 tier from ${family.repo}`);
    assert.ok(!download.coRequisite, `${modelId}'s tier download is the primary, not a co-requisite`);
    // The components repo is shared: same repo AND same revision for every member.
    const components = tierDownload(models, modelId, MAGE_COMPONENTS.repo, "q4");
    assert.ok(components.coRequisite, "the components rows are declared as co-requisites");
    seenComponents.add(components.revision);
    // No two members may claim the same variant rehost.
    const others = Object.entries(PROVIDER_FAMILIES).filter(
      ([id, other]) => id !== modelId && id.startsWith("mage_flow") && other.repo === family.repo,
    );
    assert.deepEqual(others, [], `${modelId} shares its variant rehost with ${others.map(([id]) => id).join(", ")}`);
  }
  assert.equal(seenComponents.size, 1, "every Mage entry co-requires the SAME components revision");
});

// The component directory names this script probes must be the ones the adapters actually stage, or
// a cell reports `runnable` while the load cannot open its text encoder.
test("MAGE_COMPONENT_IDS are the adapter's own component-id constants", async () => {
  const source = await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/lib.rs"), "utf8");
  const declared = [
    ["MAGE_COMPONENT_TEXT_ENCODER", "text_encoder"],
    ["MAGE_COMPONENT_VAE", "vae"],
  ];
  for (const [name, value] of declared) {
    assert.match(source, new RegExp(`pub const ${name}: &str =\\s*"${value}"`));
  }
  assert.deepEqual(MAGE_COMPONENT_IDS, declared.map(([, value]) => value));
  // And both adapter arms stage exactly those two components rather than composing a path.
  for (const backend of ["mlx", "candle"]) {
    const arm = await readFile(path.join(ROOT, `crates/sceneworks-memory-adapter/src/bin/${backend}.rs`), "utf8");
    for (const [name] of declared) {
      assert.ok(arm.includes(`protocol::${name}`), `${backend}.rs never stages ${name}`);
    }
  }
});

// sc-22736. Same shape claim, for the Wan 2.2 family and SCAIL-2 — the first families whose
// ARTIFACT is per (lane, TIER) rather than per lane. Each Wan route ships a `SceneWorks/…-mlx`
// rehost on macOS and a separate `SceneWorks/…-candle` rehost on Windows/Linux, and the candle
// rehosts carry `q4` and `q8` only: the candle dense leg is the upstream `Wan-AI/…-Diffusers`
// checkpoint, at the snapshot ROOT and with no pinned revision. SCAIL-2 is the opposite shape —
// ONE repository, all three tiers, both lanes — which is why every one of its 24 sibling cells is
// still claimed on both lanes. The KEY SET is spelled out for the sc-22731 reason: a count alone
// would stay green if one route silently lost a lane while another gained one.
test("the wan 2.2 family and scail-2 are measurable on every shipped tier of every routed lane", async () => {
  const family = ["wan_2_2", "wan_2_2_t2v_14b", "wan_2_2_i2v_14b", "scail2_14b"];
  const cells = (await shippedTieredCells()).filter((cell) => family.includes(cell.modelId));
  const expected = family.flatMap((id) =>
    ["bf16", "q4", "q8"].flatMap((tier) => [`${id}:${tier}:candle`, `${id}:${tier}:mlx`]),
  );
  assert.deepEqual(cells.map((cell) => cell.key).sort(), expected.sort());
  const gaps = (await measurabilityGaps()).filter((gap) => family.includes(gap.modelId));
  assert.equal(gaps.length, 0, gapReport(gaps));
});

// The Wan candle bf16 leg is the one cell in the whole table whose repository the manifest ships
// WITHOUT a revision, so `resolveArtifactRoot` reads the revision off whatever snapshot is staged
// instead of probing a pinned one. That is a real difference in how a root is bound, and it is
// asserted here rather than left to the sibling case above, which would report only "measurable".
test("the wan candle dense leg binds the upstream snapshot flat and unpinned", async () => {
  const models = await readManifestModels();
  for (const [id, repo] of [
    ["wan_2_2", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"],
    ["wan_2_2_t2v_14b", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"],
    ["wan_2_2_i2v_14b", "Wan-AI/Wan2.2-I2V-A14B-Diffusers"],
  ]) {
    const provider = id === "wan_2_2" ? "wan2_2_ti2v_5b" : id.replace("wan_2_2", "wan2_2");
    const artifact = familyArtifact(PROVIDER_FAMILIES[provider], "candle", "bf16");
    assert.equal(artifact.repo, repo, `${id}: the candle dense leg is the upstream checkpoint`);
    assert.equal(artifact.layout, "flat", `${id}: the upstream checkpoint has no tier subtree`);
    // ...and the manifest really does ship it unpinned, which is what makes the flat, host-read
    // binding necessary rather than a convenience.
    const download = (models.find((model) => model.id === id)?.downloads ?? []).find(
      (entry) => entry.repo === repo,
    );
    assert.ok(download, `${id} ships ${repo}`);
    assert.equal(download.revision, undefined, `${id}: ${repo} is shipped without a revision`);
    // The packed siblings are the opposite: a pinned, tier-suffixed SceneWorks rehost.
    for (const tier of ["q4", "q8"]) {
      const packed = familyArtifact(PROVIDER_FAMILIES[provider], "candle", tier);
      assert.match(packed.repo, /^SceneWorks\/.*-candle$/, `${id}:${tier}`);
      assert.equal(packed.layout, "tiered", `${id}:${tier}`);
    }
    // And the MLX lane opens its own rehost for every tier.
    for (const tier of ["bf16", "q4", "q8"]) {
      const mlx = familyArtifact(PROVIDER_FAMILIES[provider], "mlx", tier);
      assert.match(mlx.repo, /^SceneWorks\/.*-mlx$/, `${id}:${tier}:mlx`);
      assert.equal(mlx.layout, "tiered", `${id}:${tier}:mlx`);
    }
  }
});

// sc-22737. Bernini: ONE engine provider (`bernini`) serves the video entry `bernini` and the still
// entry `bernini_image` — twelve cells over both lanes. The MLX lane opens the per-tier
// `SceneWorks/bernini-mlx` rehost; the Candle lane opens the untiered `SceneWorks/bernini` bundle
// whose tier subdirs live inside it (see `laneHasTierBundle`). Bernini VIDEO is Candle-routed OFF
// THE MODEL ID, ahead of the generic `candle_video_engine_id` arm
// (`crates/sceneworks-worker/src/video_jobs/mod.rs#resolve_candle_video_route` →
// `CandleVideoRoute::Bernini` → `generate_candle_bernini`), which is why the generator — and so the
// routed-lane set `shippedTieredCells` reads — lists both lanes for it.
test("the bernini family is measurable on every shipped tier of every routed lane", async () => {
  const family = ["bernini", "bernini_image"];
  const cells = (await shippedTieredCells()).filter((cell) => family.includes(cell.modelId));
  const expected = family.flatMap((id) =>
    ["bf16", "q4", "q8"].flatMap((tier) => [`${id}:${tier}:candle`, `${id}:${tier}:mlx`]),
  );
  assert.deepEqual(cells.map((cell) => cell.key).sort(), expected.sort());
  const gaps = (await measurabilityGaps()).filter((gap) => family.includes(gap.modelId));
  assert.equal(gaps.length, 0, gapReport(gaps));
});

// `ltx_2_3 claims q4 and q8 on candle, and bf16 only on mlx` derives the five-cell key set from the
// manifest and the worker's Candle tier resolver; this is the other half of the claim — every one
// of those five cells is planned, armed and closed over.
test("ltx_2_3 is measurable on every shipped tier of every routed lane", async () => {
  const cells = (await shippedTieredCells()).filter((cell) => cell.modelId === "ltx_2_3");
  assert.equal(cells.length, 5, "the five-cell set the sibling case derives");
  const gaps = (await measurabilityGaps()).filter((gap) => gap.modelId === "ltx_2_3");
  assert.equal(gaps.length, 0, gapReport(gaps));
});

// sc-22737, re-derived by sc-22738. `generate-memory-matrix.mjs` subtracts `minimax_h3` /
// `minimax_h3_ref` from the MATRIX universe (`OUT_OF_MATRIX_CATALOG_ENTRIES` — the MLX resolver is a
// prefix predicate the generator cannot enumerate). That is a fact about the generator's parser, and
// E1 admits exactly one exemption — an unrouted lane — so the burndown denominator no longer reads
// the matrix at all and the twelve cells are IN it (`the denominator's routed-lane oracle is the
// routing catalog, not the matrix`). This case is the second, independent reading: epic 22723 E2
// names `--list` as the oracle, so the family is asked there directly, with both lanes DERIVED from
// the worker's dispatch through the same `parseInternalCandleVideoRoutes` read the generator makes
// — it throws if the Candle `CandleVideoRoute::MiniMaxH3` arm
// (`video_jobs/mod.rs#resolve_candle_video_route`) or the shared `minimax_h3_engine_id` resolver
// both lanes consult is gone — the tier axis from the manifest, and every (member, tier, lane) must
// be planned and classify runnable / weights_missing.
test("the minimax-h3 family is measurable on every shipped tier of every routed lane, through --list", async () => {
  const family = ["minimax_h3", "minimax_h3_ref"];
  // The subtraction is recorded, never silent — and it no longer removes the family from the
  // burndown: every one of the twelve cells is in the gap universe, so the sibling gap cases and
  // this one now ask the same question through two different derivations.
  assert.deepEqual([...OUT_OF_MATRIX_CATALOG_ENTRIES.keys()].sort(), family);
  const routed = parseInternalCandleVideoRoutes(
    await readFile(path.join(ROOT, "crates/sceneworks-worker/src/video_jobs/mod.rs"), "utf8"),
    await readFile(path.join(ROOT, "crates/sceneworks-worker/src/video_jobs/minimax_h3.rs"), "utf8"),
  );
  assert.deepEqual([...routed.keys()].sort(), family, "both entries ride the one engine id on both lanes");
  // The cell count is DERIVED — members × shipped tiers (manifest) × routed lanes (routing catalog)
  // — never a literal: a member losing a tier or a lane must move the expectation with it, and the
  // denominator's answer for the family is then compared against exactly that product.
  const models = await readManifestModels();
  const catalogLanes = await routedCatalogLanes();
  const expected = [];
  for (const id of family) {
    const tiers = [...new Set(
      (models.find((model) => model.id === id)?.downloads ?? [])
        .filter((download) => !download.coRequisite && ["q4", "q8", "bf16"].includes(download.variant))
        .map((download) => download.variant),
    )].sort();
    assert.deepEqual(tiers, ["bf16", "q4", "q8"], `${id} ships three tiers`);
    const lanes = lanesOf(catalogLanes, id);
    assert.deepEqual(lanes, ["mlx", "candle"], `${id} is routed on both lanes`);
    for (const tier of tiers) for (const backend of lanes) expected.push(`${id}:${tier}:${backend}`);
  }
  expected.sort();
  assert.equal(
    (await shippedTieredCells()).filter((cell) => family.includes(cell.modelId)).length,
    expected.length,
    "the denominator claims every (member, shipped tier, routed lane) cell of the family",
  );
  const plan = await readPlan();
  assert.deepEqual(
    Object.keys(plan.anchors).filter((key) => family.includes(anchorParts(key).modelId)).sort(),
    expected,
    "every (member, tier, lane) has exactly one plan anchor",
  );
  const rows = new Map();
  for (const backend of ["mlx", "candle"]) {
    const run = await planRun({ backend, anchors: null, campaign: "sc-catalog-test", hfCache: [], skipCurrent: false });
    for (const row of run.rows) rows.set(row.key, row);
  }
  const gaps = expected
    .filter((key) => !["runnable", "weights_missing"].includes(rows.get(key)?.status))
    .map((key) => `${key} ${rows.get(key)?.status ?? "unclassified"} ${rows.get(key)?.reason ?? ""}`);
  assert.deepEqual(gaps, [], "--list must classify every MiniMax-H3 cell as measurable");
});

// sc-22737. The plan's fingerprints for the three video families are the one claim a capture cannot
// re-derive on a host with no weights, so each row is bound here to the (route, tier, lane) shape
// its lane's engine mints — as sd3.5 and mage-flow are above — and the Rust adapters refuse a plan
// row naming anything else BEFORE the load (`ltx_calibration_fingerprint`,
// `minimax_calibration_fingerprint`, `bernini_calibration_fingerprint` in `mlx.rs`, and their
// candle twins):
//
// * Bernini — `bernini-image-<tier>-<lane>-dual-expert-ladder-v1`. The route token is the FULL
//   pipeline's (`mlx-gen-bernini` `calibration_route(FULL_ID)` → `image`; the renderer-only sibling
//   mints `renderer`), and both catalog entries load that pipeline, so the two entries share each
//   (tier, lane) identity and are told apart by their own (modelId, mode) key.
// * LTX-2.3 — MLX `sc-20772-ltx-2-3-<tier>-mlx-memory-ladder-v2`, except that the engine's
//   CALIBRATED tier keeps the retained bare key `sc-20772-ltx-2-3-mlx-memory-ladder-v2`; Candle
//   `sc-20772-ltx-2-3-candle-<tier>-i2v-v1` under the distilled engine id.
// * MiniMax-H3 — MLX `minimax-h3-<tier>-mlx-staged-joint-av-eager-abi3-v1` with the same bare
//   exception (`minimax-h3-mlx-staged-joint-av-eager-abi3-v1`); Candle
//   `minimax-h3-<tier>-candle-staged-joint-av-v1`. Both entries load ONE `LoadSpec`
//   (`video_jobs/minimax_h3.rs` stages the base `transformer` even for a ref2va job; the engine
//   resolves `transformer_ref` per render), so the loaded contract publishes one identity per
//   (tier, lane) and the two entries' rows must carry the SAME key.
//
// WHICH tier keeps the bare key is the engines' `CALIBRATED_TIER`, bound byte-for-byte by the
// adapter's own tests; here it is held as a SHAPE — exactly one bare MLX member per family, every
// other member tokened with its own tier — and as the literal tier in the case below when an
// inference checkout is reachable.
const LTX_MLX_BARE_KEY = "sc-20772-ltx-2-3-mlx-memory-ladder-v2";
const MINIMAX_MLX_BARE_KEY = "minimax-h3-mlx-staged-joint-av-eager-abi3-v1";

async function videoFamilyBareTiers() {
  const plan = await readPlan();
  const shipped = await shippedTieredCells();
  const rows = (predicate) => Object.entries(plan.anchors).filter(([key, row]) => predicate(anchorParts(key), row));
  const keys = (entries) => entries.map(([key]) => key).sort();
  const bare = { ltx_2_3: [], minimax_h3: [] };

  const bernini = rows(({ modelId }) => ["bernini", "bernini_image"].includes(modelId));
  assert.deepEqual(keys(bernini), shipped.filter((cell) => ["bernini", "bernini_image"].includes(cell.modelId)).map((cell) => cell.key).sort());
  for (const [key, row] of bernini) {
    const { modelId, tier, backend } = anchorParts(key);
    assert.equal(row.provider, "bernini", key);
    assert.equal(row.calibrationFingerprint, `bernini-image-${tier}-${backend}-dual-expert-ladder-v1`, key);
    assert.equal(row.mode, modelId === "bernini" ? "text_to_video" : "text_to_image", key);
    assert.equal(row.geometry.frames, modelId === "bernini" ? 49 : 1, key);
  }

  const ltx = rows(({ modelId }) => modelId === "ltx_2_3");
  assert.deepEqual(keys(ltx), shipped.filter((cell) => cell.modelId === "ltx_2_3").map((cell) => cell.key).sort());
  for (const [key, row] of ltx) {
    const { tier, backend } = anchorParts(key);
    if (backend === "candle") {
      assert.equal(row.provider, "ltx_2_3_distilled", key);
      assert.equal(row.calibrationFingerprint, `sc-20772-ltx-2-3-candle-${tier}-i2v-v1`, key);
    } else {
      assert.equal(row.provider, "ltx_2_3", key);
      if (row.calibrationFingerprint === LTX_MLX_BARE_KEY) bare.ltx_2_3.push(tier);
      else assert.equal(row.calibrationFingerprint, `sc-20772-ltx-2-3-${tier}-mlx-memory-ladder-v2`, key);
    }
  }
  assert.equal(bare.ltx_2_3.length, 1, "exactly one MLX LTX-2.3 tier keeps the engine's retained bare key");

  const minimax = rows(({ modelId }) => ["minimax_h3", "minimax_h3_ref"].includes(modelId));
  assert.equal(minimax.length, 12, "two members x three tiers x two lanes");
  for (const [key, row] of minimax) {
    const { modelId, tier, backend } = anchorParts(key);
    assert.equal(row.provider, "minimax_h3", key);
    assert.equal(row.mode, modelId === "minimax_h3" ? "text_to_video" : "reference_to_video", key);
    if (backend === "candle") {
      assert.equal(row.calibrationFingerprint, `minimax-h3-${tier}-candle-staged-joint-av-v1`, key);
    } else if (row.calibrationFingerprint === MINIMAX_MLX_BARE_KEY) {
      if (modelId === "minimax_h3") bare.minimax_h3.push(tier);
    } else {
      assert.equal(row.calibrationFingerprint, `minimax-h3-${tier}-mlx-staged-joint-av-eager-abi3-v1`, key);
    }
    // One loaded provider, one identity per (tier, lane): the reference entry's row carries the
    // base entry's key, never a partition-tokened one the contract cannot publish.
    const sibling = plan.anchors[`${modelId === "minimax_h3" ? "minimax_h3_ref" : "minimax_h3"}:${tier}:${backend}`];
    assert.ok(sibling, `${key} has its sibling entry planned`);
    assert.equal(sibling.calibrationFingerprint, row.calibrationFingerprint, `${key}: the two entries share the loaded identity`);
  }
  assert.equal(bare.minimax_h3.length, 1, "exactly one MLX MiniMax-H3 tier keeps the engine's retained bare key");
  return bare;
}

test("every planned bernini, ltx_2_3 and minimax-h3 fingerprint is one its lane's engine can mint", async () => {
  await videoFamilyBareTiers();
});

// The literal half of the bare-key claim, read off the pinned engines when a checkout is reachable
// (CI always supplies one — see `skipWithoutRoutes`): the one bare MLX row per family is the tier
// the engine's `CALIBRATED_TIER` names, so a re-tiered engine reds the plan here as well as in
// the adapter's own tests.
// sc-22738. `mlx.rs#krea_realtime_probes_admission` decides whether the Krea Realtime capture asks
// the provider's admission gate anything, and its answer must be the ENGINE's: production admits
// only the routes `mlx-gen-krea-realtime`'s registered `safety_check` names, and every other request
// mode is refused there by name (`crossed resident request mode`) before the shared budget check is
// ever reached. sc-22735 probed the anchor's own reference-free t2v surface anyway and all three
// `krea_realtime_14b:*:mlx` cells died on the first probe after a full 14B weight load.
//
// The Rust side cannot make this binding: the gate short-circuits on an unresolvable artifact
// receipt, so a weights-free probe rejects every mode for a reason that says nothing about routing.
// So both route lists are READ — the engine's out of its `safety_check` match, the arm's out of its
// `matches!` — and compared. An engine that grows or drops a route reds here rather than leaving the
// arm quietly skipping a surface production now admits.
test("the Krea Realtime arm's admitted routes are the pinned engine's own", { skip: skipWithoutRoutes }, async () => {
  const engine = await readFile(
    path.join(process.env.INFERENCE_REPO, "crates/media/mlx-gen/mlx-gen-krea-realtime/src/memory_strategy.rs"),
    "utf8",
  );
  const gate = /pub\(crate\) fn safety_check\(([\s\S]*?)\n\}\n/.exec(engine);
  assert.ok(gate, "the pinned crate still declares a registered safety_check");
  const routed = /match context\.mode\.as_key\(\) \{([\s\S]*?)\n {4}\};/.exec(gate[1]);
  assert.ok(routed, "safety_check no longer routes on the request mode; re-read this derivation");
  const engineRoutes = [...routed[1].matchAll(/^\s*"([a-z_]+)" =>/gm)].map((match) => match[1]).sort();
  assert.ok(engineRoutes.length > 0, "the engine's route arm list parsed empty");
  assert.ok(
    !engineRoutes.includes("text_to_video"),
    "the engine now admits the reference-free t2v surface; the arm must PROBE admission again "
      + "rather than record the three scenarios unexecuted",
  );

  const arm = await readFile(path.join(ROOT, ADAPTER_BIN_PATHS.mlx), "utf8");
  const predicate = /fn krea_realtime_probes_admission\([\s\S]*?\n\}/.exec(arm);
  assert.ok(predicate, "the arm still declares krea_realtime_probes_admission");
  const armRoutes = [...predicate[0].matchAll(/"([a-z_]+)"/g)].map((match) => match[1]).sort();
  assert.deepEqual(
    armRoutes,
    engineRoutes,
    "the arm's admitted-route mirror has drifted from the pinned engine's own safety_check",
  );
  assert.match(
    predicate[0],
    /reference_count > 0/,
    "a reference-carrying request must probe on any mode, so a new reference surface cannot "
      + "inherit the skip silently",
  );
});

// sc-22738. `mlx_wan_scail2.rs#probes_admission` decides whether the Wan 2.2 / SCAIL-2 capture asks
// the provider's admission gate anything, and its answer turns on ONE fact: neither engine can read
// any evidence identity SceneWorks mints, because both parse the run context's `evidence_revision`
// as a receipt they sealed themselves. `wan_2_2_i2v_14b:bf16:mlx` died on the FIRST probe after a
// 383-second load before the arm mirrored that.
//
// The Wan half of the arm asks the engine for its token (`wan_i2v_memory::RECEIPT_VERSION` is
// `pub`); the SCAIL-2 half cannot, because `mlx-gen-scail2` compares its token as a bare literal
// inside `validate_context_revision_shape` and publishes no const for it. Both are bound to the
// pinned engine source HERE so a rename reds rather than silently widening the arm's skip — and the
// worker is checked to mint neither, which is what makes the skip production's own decision.
test("the Wan/SCAIL-2 arm's receipt tokens are the pinned engines' own, and the worker mints neither", { skip: skipWithoutRoutes }, async () => {
  const arm = await readFile(
    path.join(ROOT, "crates/sceneworks-memory-adapter/src/bin/mlx_wan_scail2.rs"),
    "utf8",
  );

  const wan = await readFile(
    path.join(process.env.INFERENCE_REPO, "crates/contracts/gen-core/src/wan_i2v_memory.rs"),
    "utf8",
  );
  const wanToken = /pub const RECEIPT_VERSION: &str = "([a-z0-9-]+)";/.exec(wan)?.[1];
  assert.ok(wanToken, "gen-core's wan_i2v_memory still publishes RECEIPT_VERSION");
  assert.match(
    arm,
    /Some\(_\) => mlx_gen::gen_core::wan_i2v_memory::RECEIPT_VERSION,/,
    "the Wan half must ASK the engine for its receipt token, never restate it",
  );
  // …and the arm's prose, which is what a record reader sees, names the same shape.
  assert.ok(
    arm.includes(`${wanToken}:<mode>:fps<N>:<artifact>:<selection>:`),
    `the arm's admission blocker must name the engine's own receipt shape (${wanToken})`,
  );

  const scail2 = await readFile(
    path.join(process.env.INFERENCE_REPO, "crates/media/mlx-gen/mlx-gen-scail2/src/memory_strategy.rs"),
    "utf8",
  );
  const shape = /fn validate_context_revision_shape\(([\s\S]*?)\n\}\n/.exec(scail2);
  assert.ok(shape, "mlx-gen-scail2 still gates the context on its own receipt shape");
  const scail2Token = /parts\[0\] != "([a-z0-9-]+)"/.exec(shape[1])?.[1];
  assert.ok(scail2Token, "the SCAIL-2 receipt token parsed empty");
  const mirrored = /const SCAIL2_RECEIPT_TOKEN: &str = "([a-z0-9-]+)";/.exec(arm)?.[1];
  assert.equal(
    mirrored,
    scail2Token,
    "the arm's SCAIL-2 receipt-token mirror has drifted from the pinned engine's own gate",
  );

  // The other half of the same claim, and the reason the skip is production's decision rather than
  // the adapter's: SceneWorks mints NEITHER token, so the evidence identity the worker puts on a
  // video run context can never be one these gates read.
  const minted = await execFileAsync("git", [
    "grep", "-l", "-F", "-e", wanToken, "-e", scail2Token, "--",
    "crates/sceneworks-worker/src", "crates/sceneworks-core/src",
  ], { cwd: ROOT }).then((result) => result.stdout.trim(), () => "");
  assert.equal(
    minted,
    "",
    "the worker now spells an engine receipt token; it may mint a context these gates accept, so "
      + "the Wan/SCAIL-2 arm must probe admission again instead of recording the scenarios unexecuted",
  );
});

test("the bare MLX LTX-2.3 and MiniMax-H3 plan rows are the engines' calibrated tiers", { skip: skipWithoutRoutes }, async () => {
  const bare = await videoFamilyBareTiers();
  for (const [family, crate] of [
    ["ltx_2_3", "crates/media/mlx-gen/mlx-gen-ltx/src/memory_strategy.rs"],
    ["minimax_h3", "crates/media/mlx-gen/mlx-gen-minimax-h3/src/memory_strategy.rs"],
  ]) {
    const source = await readFile(path.join(process.env.INFERENCE_REPO, crate), "utf8");
    const calibrated = /pub const CALIBRATED_TIER: &str = "([a-z0-9]+)";/.exec(source)?.[1];
    assert.ok(calibrated, `${crate} declares CALIBRATED_TIER`);
    assert.deepEqual(bare[family], [calibrated], `${family}: the bare MLX row is the engine's calibrated tier`);
  }
});

// Every Mage plan row's `calibrationFingerprint` must be the string the PRODUCTION contract emits
// for that cell, and every row's `loadShape` the shape the WORKER loads that cell under. Both are
// per (lane, tier) tables, bound here as the adapters bind them (`mage_calibration_fingerprint` in
// both `mlx.rs` and `candle.rs`, which refuse a plan row naming anything else BEFORE the load):
//
// * MLX: `mlx-gen-mage` `production_calibration_fingerprint` (inference PR 953) —
//   `mage-flow-<route>-<tier>-mlx-shared-ladder-v1`, 18 cells; the registry-conformance family
//   (`mage-flow-mlx-registry-behavior-v1-<route>-<tier>`) and the retired single string
//   (`mage-flow-mlx-shared-ladder-2026-08-03-v1`) are never production identities.
// * Candle: `candle-gen-mage` `production_calibration_fingerprint` —
//   `mage-flow-cuda-<provider>-<tier>-shared-ladder-v3`, 18 cells.
// * Shape: the worker's MLX stream evaluates Mage `Applied + Deferred` on every tier (typed rules,
//   BTR declared on all three tiers), while its Candle stream matches the generated BTR row's own
//   `tiers` (`["bf16"]`) — deferred on bf16, refused (eager) on q4/q8. The worker's
//   `memory_route_registry` Mage tests drive both evaluators over the real manifest entries and pin
//   these rows to them.
//
// The cell set is derived (family x shipped tiers x routed lanes), never a frozen count.
test("every mage-flow plan row names its cell's production identity and the worker's load shape", async () => {
  const plan = await readPlan();
  const family = [
    "mage_flow", "mage_flow_base", "mage_flow_turbo",
    "mage_flow_edit", "mage_flow_edit_base", "mage_flow_edit_turbo",
  ];
  const expectedKeys = (await shippedTieredCells()).filter((cell) => family.includes(cell.modelId)).map((cell) => cell.key).sort();
  const mage = Object.entries(plan.anchors).filter(([, row]) => family.includes(row.provider));
  assert.deepEqual(mage.map(([key]) => key).sort(), expectedKeys, "the planned Mage cells are exactly the shipped ones");
  const route = (provider) => provider.replaceAll("_", "-");
  const identities = new Set();
  for (const [key, row] of mage) {
    const { modelId, tier, backend } = anchorParts(key);
    assert.equal(row.provider, modelId, `${key}: catalog id and engine provider id are equal on every Mage row`);
    const expected = backend === "mlx"
      ? `mage-flow-${route(row.provider)}-${tier}-mlx-shared-ladder-v1`
      : `mage-flow-cuda-${route(row.provider)}-${tier}-shared-ladder-v3`;
    assert.equal(row.calibrationFingerprint, expected, `${key} names a fingerprint ${backend} cannot emit`);
    assert.ok(!row.calibrationFingerprint.startsWith("mage-flow-mlx-registry-behavior-v1"), `${key} names a weights-free conformance string`);
    assert.notEqual(row.calibrationFingerprint, "mage-flow-mlx-shared-ladder-2026-08-03-v1", `${key} names the retired single string`);
    assert.ok(!identities.has(row.calibrationFingerprint), `${key} shares its identity with another cell`);
    identities.add(row.calibrationFingerprint);
    const shape = backend === "mlx" || tier === "bf16" ? "deferred_materialization" : "eager_materialization";
    assert.equal(row.loadShape, shape, `${key} must plan the shape the worker loads`);
    assert.equal(row.overlay, "none", `${key} carries no overlay`);
  }
  assert.equal(identities.size, expectedKeys.length, "one identity per cell");
});

// The manifest's six MLX Mage contract blocks carry the ENGINE'S weights-free conformance identity
// per row, as every declared MLX family does (FLUX.1: `flux-one-static-registry-behavior-v2-dev`).
// `mlx-gen-mage` publishes that identity per (route, tier), so the rows are split per tier and no
// row may carry a production string, the retired single string, or a multi-tier `tiers` list.
test("every mage-flow MLX manifest row declares its own per-tier registry-conformance identity", async () => {
  const models = await readManifestModels();
  for (const modelId of Object.keys(PROVIDER_FAMILIES).filter((id) => id.startsWith("mage_flow"))) {
    const contract = models.find((model) => model.id === modelId)?.mlx?.memoryStrategyContract;
    assert.ok(contract, `${modelId} declares an MLX memory contract`);
    assert.equal(contract.provider, modelId);
    const cells = new Set();
    for (const row of contract.implementations) {
      assert.deepEqual(Object.keys(row).filter((k) => k === "tiers"), ["tiers"], `${modelId}/${row.rung} declares tiers`);
      assert.equal(row.tiers.length, 1, `${modelId}/${row.rung} rows are split per tier, got ${JSON.stringify(row.tiers)}`);
      const [tier] = row.tiers;
      assert.equal(row.fingerprint, `mage-flow-mlx-registry-behavior-v1-${modelId.replaceAll("_", "-")}-${tier}`, `${modelId}/${row.rung}/${tier}`);
      cells.add(`${row.rung}:${tier}`);
    }
    assert.deepEqual(
      [...cells].sort(),
      ["bounded_attention", "bounded_decode", "bounded_transformer_residency", "resident", "staged_residency"]
        .flatMap((rung) => ["bf16", "q4", "q8"].map((tier) => `${rung}:${tier}`)).sort(),
      `${modelId} declares every rung on every shipped tier exactly once`,
    );
  }
});

// A Mage cell whose components row for THIS tier is not declared (or whose components repo has no
// rows at all) is `weights_missing` — planned and armed, merely not bindable on this host — and
// never a thrown error, which would abort the whole `--list` for every other cell (sc-22733 review).
test("a mage-flow cell with no components row for its tier is weights_missing, not an error", async () => {
  const family = PROVIDER_FAMILIES.mage_flow;
  const variant = (tier) => ({ repo: family.repo, revision: REVISION, variant: tier, files: [`${tier}/*`] });
  const component = (tier, componentId) => ({
    repo: MAGE_COMPONENTS.repo, revision: UPSTREAM, coRequisite: true, componentId, variant: tier,
    subdir: `${tier}/${componentId}`, files: [`${tier}/${componentId}/*`],
  });
  const models = [
    { id: "mage_flow", downloads: [variant("q4"), variant("bf16"), ...MAGE_COMPONENT_IDS.map((id) => component("q4", id))] },
    { id: "mage_flow_base", downloads: [{ ...variant("q4"), repo: PROVIDER_FAMILIES.mage_flow_base.repo }] },
  ];
  const hub = await fakeHub([
    [family.repo, REVISION, "q4"],
    [family.repo, REVISION, "bf16"],
    [PROVIDER_FAMILIES.mage_flow_base.repo, REVISION, "q4"],
    ...MAGE_COMPONENT_IDS.map((id) => [MAGE_COMPONENTS.repo, UPSTREAM, "q4", id]),
  ]);
  const context = { models, backend: "mlx", hubs: [hub], current: new Map(), captured: new Map() };
  const runnable = await classifyAnchor("mage_flow:q4:mlx", { provider: "mage_flow" }, context);
  assert.equal(runnable.status, "runnable", runnable.reason);
  assert.equal(runnable.env[`SCENEWORKS_${MAGE_COMPONENTS.env}_ROOT`], snapshotPath(hub, MAGE_COMPONENTS.repo, UPSTREAM));
  const noTierRow = await classifyAnchor("mage_flow:bf16:mlx", { provider: "mage_flow" }, context);
  assert.equal(noTierRow.status, "weights_missing");
  assert.match(noTierRow.reason, /declares no .*Mage-Flow-Components-mlx components row for tier bf16/);
  assert.ok(noTierRow.roots.some((root) => root.label === "components snapshot"), "the missing root is named");
  const noRows = await classifyAnchor("mage_flow_base:q4:mlx", { provider: "mage_flow_base" }, context);
  assert.equal(noRows.status, "weights_missing");
  assert.match(noRows.reason, /mage_flow_base declares no .* components row for tier q4/);
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

// sc-22732 review item 4: NOTHING cross-checked the plan's turnkey fingerprints against the shipped
// manifest, so the two could name different identities for the same cell with every test green. The
// sc-22730 case above is the precedent; this is the same claim for the five turnkey still members,
// and it is FAIL-CLOSED — there is no third outcome for a cell.
//
// The CANDLE half is a real cross-source binding: the plan's string must be one the model's own
// `candle.memoryStrategyContract` declares. A cell whose lane declares NOTHING is not silently
// skipped; it is an error UNLESS that backend block carries a `memoryDeclarationWithhold` whose
// reason NAMES this anchor key. `kolors:*:candle` is the one such cell: SC-20790 withholds the
// `staged_residency` declaration because no measured record prices the base staged cell yet, while
// the sc-22732 inference head does publish a per-(route, artifact tier) identity for it. Requiring
// the withhold to name the anchors is what keeps that pairing visible instead of prose.
//
// The MLX half cannot be bound to a manifest constant for every member: `mlx-gen-ideogram` has no
// memory strategy at the compiled pin (inference PR 954 adds it) and the lens blocks carry only the
// engine-derived region, whose fingerprints arrive with the next capability dump. It is bound to the
// (route, TIER) shape the merged engines mint — and WHERE the manifest does declare a fingerprint
// for a planned mlx cell (kolors), that string must agree too, so the half that CAN be bound is.
test("every planned turnkey-still fingerprint is one its lane's declaration can emit", async () => {
  const plan = await readPlan();
  const models = await readManifestModels();
  // (model id -> the shape its MLX identity takes). `kolors` q4 and `lens` q4 are the preserved
  // measured keys, so they are named rather than derived.
  const mlxIdentity = {
    kolors: (tier) => (tier === "q4" ? "kolors-mlx-chatglm3-sdxl-unet-ladder-v1" : `kolors-${tier}-mlx-shared-ladder-v1`),
    ideogram_4: (tier) => `ideogram-4-${tier}-mlx-shared-ladder-v1`,
    ideogram_4_turbo: (tier) => `ideogram-4-turbo-${tier}-mlx-shared-ladder-v1`,
    lens: (tier) => (tier === "q4" ? "lens-mlx-shared-ladder-2026-08-03-v1" : `lens-${tier}-mlx-shared-ladder-v1`),
    lens_turbo: (tier) => `lens-turbo-${tier}-mlx-shared-ladder-v1`,
  };
  const withheld = [];
  let seen = 0;
  for (const [key, entry] of Object.entries(plan.anchors)) {
    const { modelId, tier, backend } = anchorParts(key);
    if (!(modelId in mlxIdentity)) continue;
    seen += 1;
    const model = models.find((candidate) => candidate.id === modelId);
    const declared = new Set(
      (model?.[backend]?.memoryStrategyContract?.implementations ?? [])
        .filter((impl) => impl.fingerprint && (impl.tiers ?? []).includes(tier))
        .map((impl) => impl.fingerprint),
    );
    if (backend === "mlx") {
      assert.equal(entry.calibrationFingerprint, mlxIdentity[modelId](tier), key);
      // Bound where it CAN be bound: a declared mlx fingerprint for a planned cell must agree.
      if (declared.size > 0) {
        assert.ok(
          declared.has(entry.calibrationFingerprint),
          `${key}: plan names ${entry.calibrationFingerprint}, manifest declares ${[...declared].join(", ")}`,
        );
      }
      continue;
    }
    if (declared.size > 0) {
      assert.ok(
        declared.has(entry.calibrationFingerprint),
        `${key}: plan names ${entry.calibrationFingerprint}, manifest declares ${[...declared].join(", ")}`,
      );
      continue;
    }
    // Fail-closed: an undeclared candle cell needs a cited withhold that NAMES this anchor.
    const reason = model?.[backend]?.memoryDeclarationWithhold?.reason ?? "";
    assert.ok(
      reason.includes(key),
      `${key}: the candle lane declares no fingerprint for this tier and no memoryDeclarationWithhold names the anchor`,
    );
    withheld.push(key);
  }
  assert.equal(seen, 5 * 3 * 2, "every turnkey still cell is priced by the plan");
  // Stated as data: the withheld set is exactly the three Kolors candle cells, so a lane that
  // silently loses its declarations cannot hide behind a broadly-worded withhold.
  assert.deepEqual(withheld.sort(), ["kolors:bf16:candle", "kolors:q4:candle", "kolors:q8:candle"]);
});

// sc-22726 left this binding open and sc-22730 closes it: the artifact repository of every family
// is written down TWICE — as `PROVIDER_FAMILIES[*].repo` here and as a `pub const *_REPOSITORY` in
// the adapter's lib.rs, which is what both arms validate the operator's env against
// (`validate_artifact_identity`). Editing either alone yields a runner that stages a root the
// adapter then refuses by name, with every test in both languages still green.
test("PROVIDER_FAMILIES repos are the adapter's *_REPOSITORY constants", async () => {
  const lib = await readFile(path.join(ROOT, ADAPTER_LIB_PATH), "utf8");
  const declared = new Set();
  // `\s*` after the `=`, not a space (sc-22734): rustfmt wraps the initializer onto its own line
  // whenever the declaration exceeds the width, which four of the six SenseNova repository consts
  // do. A single-space pattern reads those four as UNDECLARED and fails a lane that is in fact
  // bound — the binding this case exists to check is the const's VALUE, not its line breaks.
  for (const match of lib.matchAll(/pub const [A-Z0-9_]+_REPOSITORY: &str =\s*"([^"]+)";/g)) {
    declared.add(match[1]);
  }
  assert.ok(declared.size > 0, "lib.rs declares repository constants");
  for (const [provider, family] of Object.entries(PROVIDER_FAMILIES)) {
    // LTX-2.5 is bound by the harness itself rather than by an adapter env family, so its repo
    // literal lives in this module (`LTX25_REPOSITORY`) and not in lib.rs.
    if (family.ltx25) continue;
    // Through `familyArtifact`, so the per-(lane, TIER) overrides are covered too (sc-22736): the
    // Wan candle q4/q8 rehost and its upstream dense leg are repositories an arm validates the
    // operator's env against exactly as it does `family.repo`, and reading only the top-level key
    // would have left six of them bound in one language alone.
    const repos = new Set([
      family.repo,
      ...(family.arms ?? []).flatMap((arm) =>
        ["bf16", "q4", "q8"].map((tier) => familyArtifact(family, arm, tier).repo),
      ),
    ]);
    for (const repo of repos) {
      assert.ok(
        declared.has(repo),
        `${provider}: ${repo} is not declared as a *_REPOSITORY const in ${ADAPTER_LIB_PATH}`,
      );
    }
  }
});
const ADAPTER_BIN_PATHS = Object.freeze({
  mlx: "crates/sceneworks-memory-adapter/src/bin/mlx.rs",
  candle: "crates/sceneworks-memory-adapter/src/bin/candle.rs",
});

let adapterArmsPromise;
/** `{mlx, candle}` -> the provider ids each adapter binary's dispatch admits, parsed from Rust. */
function adapterArms() {
  adapterArmsPromise ??= (async () =>
    Object.fromEntries(
      await Promise.all(
        Object.entries(ADAPTER_BIN_PATHS).map(async ([backend, relative]) => [
          backend,
          adapterCapturableProviders(
            await readFile(path.join(ROOT, relative), "utf8"),
            relative,
          ),
        ]),
      ),
    ))();
  return adapterArmsPromise;
}

// sc-22738 (feature-end round 1). `classifyAnchor`'s `no_adapter_arm` verdict is graded off
// `PROVIDER_FAMILIES[].arms` — a hand-kept table — and NOTHING tied that table to the Rust the arms
// live in. Deleting `KOLORS_ID => Ok(KOLORS_PLAIN_EXECUTION_PATH)` from `candle.rs` left this file,
// `stale-lane-report.test.mjs` and the whole `npm run check` green: the adapter could no longer
// serve `candle:kolors`, and the measurability oracle went on saying it could.
//
// So `arms` is DERIVED here, through `adapterCapturableProviders` — the stale-lane report's own
// parser, which reads the `match provider` blocks able to refuse with "five-rung calibration does
// not implement provider" and admits a provider only when EVERY such dispatch does. It throws if
// the dispatch shape moves, so this cannot degrade to an empty set and pass vacuously.
//
// Two directions, because either one alone is a false green:
//
//  * every declared `(family, arm)` is a provider that lane's adapter dispatches — a deleted Rust
//    arm reds instead of becoming a silent `runnable`;
//  * every PLAN row's `<backend>:<provider>` is dispatched too — a plan row can name a provider no
//    family row covers, and `--list` would classify it off a family lookup that never happened.
test("every declared adapter arm is a provider that lane's adapter really dispatches", async () => {
  const arms = await adapterArms();
  for (const backend of Object.keys(ADAPTER_BIN_PATHS)) {
    assert.ok(arms[backend].length > 0, `${backend}: the dispatch parse admits no provider at all`);
  }

  let checked = 0;
  for (const [key, family] of Object.entries(PROVIDER_FAMILIES)) {
    // Model-keyed rows (sc-22729/sc-22734) name the engine provider they ride; provider-keyed rows
    // ARE the provider id. `familyFor` resolves both the same way.
    const provider = family.provider ?? key;
    assert.ok((family.arms ?? []).length > 0, `${key} declares no arm at all`);
    for (const backend of family.arms) {
      assert.ok(
        arms[backend].includes(provider),
        `${key}: PROVIDER_FAMILIES declares a ${backend} arm, but ${ADAPTER_BIN_PATHS[backend]} ` +
          `dispatches no ${provider} provider — classifyAnchor would call this cell runnable`,
      );
      checked += 1;
    }
  }
  assert.ok(checked > 0, "no family arm was checked; this test guards nothing");

  const plan = await readPlan();
  const unbacked = [];
  for (const [key, row] of Object.entries(plan.anchors)) {
    const { backend } = anchorParts(key);
    if (!arms[backend].includes(row.provider)) unbacked.push(`${key} -> ${backend}:${row.provider}`);
  }
  assert.deepEqual(
    unbacked,
    [],
    "every plan anchor names a provider its lane's adapter dispatch admits",
  );
});

/** Every `SCENEWORKS_LTX_*` root `relative`'s LTX-2.3 arm refuses to run without. */
function requiredLtxEnv(source, relative) {
  const names = [
    ...new Set(
      [...source.matchAll(/required_env\(\s*"(SCENEWORKS_LTX_[A-Z0-9_]+)"\s*,?\s*\)/g)].map(
        (match) => match[1],
      ),
    ),
  ].sort();
  assert.ok(
    names.length > 0,
    `${relative}: the required_env parse found no SCENEWORKS_LTX_* root at all; the LTX arm's ` +
      "environment family moved and this derivation has stopped asking its question",
  );
  return names;
}

// sc-22738. The campaign lost ALL THREE `ltx_2_3:*:mlx` captures to
// `required environment variable SCENEWORKS_LTX_TEXT_ENCODER_ROOT is not set`, ~150 s into each
// booked run: `PROVIDER_FAMILIES.ltx_2_3` declared no `siblingRoots`, on a comment claiming the MLX
// arm read no text-encoder root. `mlx.rs#ltx_load_spec` has required it since sc-18808, so `--list`
// called every cell runnable and the adapter refused before a weight was opened.
//
// NOTHING here is a literal env name: the requirement is READ out of each lane's adapter binary, so
// an arm that grows (or drops) a `SCENEWORKS_LTX_*` root moves this case with no edit, and a row
// that stops binding one reds. The resolution asserted is PRODUCTION's own —
// `video_jobs/ltx.rs#bundled_ltx_gemma_dir` joins `gemma` to the selected tier dir's parent
// snapshot — so the value handed to the arm is the value the worker would resolve.
test("every SCENEWORKS_LTX_* root the adapters require is bound by the LTX-2.3 rows", async () => {
  const lanes = [
    ["mlx", "ltx_2_3"],
    ["candle", "ltx_2_3_distilled"],
  ];
  const required = Object.fromEntries(
    await Promise.all(
      lanes.map(async ([backend]) => [
        backend,
        requiredLtxEnv(
          await readFile(path.join(ROOT, ADAPTER_BIN_PATHS[backend]), "utf8"),
          ADAPTER_BIN_PATHS[backend],
        ),
      ]),
    ),
  );
  // Both arms open the same LTX-2.3 identity stack, so a root one lane requires and the other does
  // not is itself a finding — and it would leave one lane's rows short exactly the way the MLX row
  // was short.
  assert.deepEqual(required.mlx, required.candle, "the two LTX arms require the same root family");

  const models = [
    ...fakeModels(),
    {
      id: "ltx_2_3",
      downloads: [
        { repo: LTX_2_3_REPOSITORY, revision: REVISION, variant: "q4", files: ["q4/*"] },
      ],
    },
  ];
  const layout = [[LTX_2_3_REPOSITORY, REVISION, "q4"]];

  const staged = await fakeHub([...layout, [LTX_2_3_REPOSITORY, REVISION, "gemma"]]);
  const snapshot = snapshotPath(staged, LTX_2_3_REPOSITORY, REVISION);
  for (const [backend, provider] of lanes) {
    const row = await classifyAnchor(`ltx_2_3:q4:${backend}`, { provider }, {
      models, backend, hubs: [staged], current: new Map(), captured: new Map(),
    });
    assert.equal(row.status, "runnable", `${backend}: ${row.reason}`);
    for (const name of required[backend]) {
      assert.ok(
        Object.hasOwn(row.env, name),
        `${backend}: ${ADAPTER_BIN_PATHS[backend]} requires ${name}, but the ltx_2_3 row binds ` +
          `only ${Object.keys(row.env).sort().join(", ")} — the booked capture dies on it`,
      );
    }
    assert.equal(
      row.env.SCENEWORKS_LTX_TEXT_ENCODER_ROOT,
      path.join(snapshot, "gemma"),
      `${backend}: the text-encoder root must be the gemma sibling of the SELECTED tier's own ` +
        "snapshot, which is what bundled_ltx_gemma_dir resolves and what both arms snapshot-validate",
    );
    assert.equal(row.env.SCENEWORKS_LTX_ROOT, path.join(snapshot, "q4"), backend);
  }

  // And a host WITHOUT the sibling is `weights_missing` by name rather than runnable: the load
  // cannot open, so booking a capture there only reproduces the failure this case exists for.
  const holed = await fakeHub(layout);
  for (const [backend, provider] of lanes) {
    const row = await classifyAnchor(`ltx_2_3:q4:${backend}`, { provider }, {
      models, backend, hubs: [holed], current: new Map(), captured: new Map(),
    });
    assert.equal(row.status, "weights_missing", `${backend}: ${row.reason}`);
    assert.match(row.reason, /gemma\//, backend);
  }
});

// sc-22738 (feature-end round 1). The lane-default anchor composition has ONE spelling.
//
// `crates/sceneworks-worker/src/inference_runtime.rs` walks the same plan and asks each row's
// contract to `validate_selection` the rung it will be captured at. Its `lane_default_rung` used to
// respell `ANCHOR_STRATEGY` in Rust — mlx -> "resident", candle -> "staged_residency" — with nothing
// binding the two, so changing a default here left the Rust walk asserting the old rung and green.
// Both sides now read `config/anchor-lane-default-strategy.json`. This case holds that shape:
// `ANCHOR_STRATEGY` really is that file, and the Rust really reads it rather than a literal.
test("the lane default anchor composition is one declaration both lanes' walkers read", async () => {
  const declared = JSON.parse(
    await readFile(path.join(ROOT, ANCHOR_LANE_DEFAULT_STRATEGY_PATH), "utf8"),
  );
  assert.deepEqual(Object.keys(declared.lanes).sort(), ["candle", "mlx"]);
  for (const [backend, entry] of Object.entries(declared.lanes)) {
    assert.equal(ANCHOR_STRATEGY[backend].rung, entry.rung, backend);
    assert.deepEqual([...ANCHOR_STRATEGY[backend].engagedRungs], entry.engagedRungs, backend);
    assert.ok(entry.engagedRungs.includes(entry.rung), `${backend}: the default rung is engaged`);
  }

  const rust = await readFile(
    path.join(ROOT, "crates/sceneworks-worker/src/inference_runtime.rs"),
    "utf8",
  );
  assert.match(
    rust,
    /include_str!\(\s*"\.\.\/\.\.\/\.\.\/config\/anchor-lane-default-strategy\.json"\s*\)/,
    "inference_runtime.rs no longer reads the shared lane-default declaration",
  );
  // …and it does not carry a second spelling of the values. `"mlx" => "resident"` was the defect.
  const walk = /fn every_planned_lane_row_resolves_a_weights_free_contract_implementing_its_rung[\s\S]*?\n\}\n/.exec(rust);
  assert.ok(walk, "the planned-rung walk is still where the lane default is applied");
  assert.doesNotMatch(
    walk[0],
    /"(?:mlx|candle)"\s*=>\s*"(?:resident|staged_residency)"/,
    "the planned-rung walk respells a lane default instead of reading the shared declaration",
  );
});

// The catalog-wide burndown, and epic 22723's E1/E2 gate. PROMOTED to a hard assertion by sc-22738
// (it was `todo` while the families were brought in, so the gap set printed on every `npm run check`
// without failing it). It is a SHAPE claim and carries no count: `measurabilityGaps()` walks every
// `<modelId>:<tier>:<backend>` the routing catalog and the manifest between them claim, on BOTH
// lanes, and demands `--list` classify each one `runnable` or `weights_missing`. `weights_missing`
// is a host condition, never a gap (E5), so this needs no weights, no GPU and no inference
// checkout.
test("every shipped tiered model is measurable", async () => {
  const gaps = await measurabilityGaps();
  assert.equal(gaps.length, 0, gapReport(gaps));
  // sc-22738: a route the PINNED `mlx-gen-wan` loader does not seal a receipt for cannot be made
  // measurable by any change in this repository — the loaded provider publishes no memory-strategy
  // contract, so the capture arm refuses the cell after the load (fixed engine-side in
  // SceneWorks/inference#964; the cells come back the moment the pin carries it). `pinnedLoaderGaps`
  // re-derives that excuse from the same pinned loader `classifyAnchor` reads rather than listing
  // cells, so it empties itself at the pin bump — and it stays bounded to exactly that here.
  const sealed = await readWanMlxSealedProviders();
  for (const gap of await pinnedLoaderGaps()) {
    assert.equal(
      gap.reason,
      wanMlxSealGap(gap.provider, sealed),
      `${gap.key}: excused for something other than the pinned loader's own seal`,
    );
    assert.equal(gap.backend, "mlx", `${gap.key}: the loader seal is an MLX-lane fact`);
    assert.ok(!sealed.has(gap.provider), `${gap.key}: excused a cell whose loader DOES seal`);
  }
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

  // sc-22738. The harness rejects a failed provider with the adapter's WHOLE stderr, terminal error
  // LAST, and adapters print informational lines on the way out. Reading from the front made every
  // Mage anchor in the 09-06 campaign report the sc-22414 coherence tally as its failure while the
  // real refusal never reached a summary row.
  const mage = [
    "file:///x/harness.mjs:1962",
    "        : reject(new Error(msg));",
    "Error: memory-mlx-adapter exited 1: GPU-view coherence retries during this render: mlx_gen=0 mlx_llm=0 (sc-22414)",
    "a synchronized Mage-Flow lifecycle phase reported a zero active peak",
    "    at ChildProcess.<anonymous> (file:///x/harness.mjs:1962:11)",
    "",
    "Node.js v24.15.0",
  ].join("\n");
  assert.equal(
    failureReason({ stderr: mage, message: "node exited 1" }),
    "a synchronized Mage-Flow lifecycle phase reported a zero active peak",
  );

  // sc-22738, the other half. `protocol::fail` now writes the failure line FIRST and flushes its
  // deferred notes after it, so reading from the end put the sc-22414 coherence tally back in
  // every summary row — by construction this time. A `note:` line is information, not an outcome.
  const deferred = [
    "file:///x/harness.mjs:1962",
    "Error: memory-mlx-adapter exited 1: memory-strategy provider adapter: LTX-2.3 admission rejected an exact-fit calibrated budget",
    "memory-strategy provider adapter: note: GPU-view coherence retries during this render: mlx_gen=0 mlx_llm=0 (sc-22414)",
    "    at ChildProcess.<anonymous> (file:///x/harness.mjs:1962:11)",
    "Node.js v24.15.0",
  ].join("\n");
  assert.equal(
    failureReason({ stderr: deferred, message: "node exited 1" }),
    "Error: memory-mlx-adapter exited 1: memory-strategy provider adapter: LTX-2.3 admission rejected an exact-fit calibrated budget",
  );
  // The same note without the adapter banner, and several of them, still skipped.
  assert.equal(
    failureReason({ stderr: ["the real refusal", "note: first tally", "note: second tally"].join("\n") }),
    "the real refusal",
  );
  // A real failure that merely CONTAINS the word is an outcome and must still be quoted.
  assert.equal(
    failureReason({ stderr: "adapter refused: take note: the ceiling was breached" }),
    "adapter refused: take note: the ceiling was breached",
  );
  // Notes alone leave the runner with nothing better to say than the note.
  assert.equal(failureReason({ stderr: "note: only a tally" }), "note: only a tally");

  // Bounded by LENGTH alone: a `;` or a `)` inside a real adapter message is content, not a cut.
  const delimited = "Error: adapter refused; the wired ceiling (87044670532 bytes) was already breached";
  assert.equal(failureReason({ stderr: delimited }), delimited);
  assert.equal(failureReason({ stderr: "x\n" + "y".repeat(400) }).length, FAILURE_REASON_LIMIT);

  // The runner's OWN refusals arrive with no child stderr and wrap across lines; they stay whole.
  assert.equal(
    failureReason({ message: "post-steps changed paths this run does not own:\n?? docs/generated/stray.json" }),
    "post-steps changed paths this run does not own: ?? docs/generated/stray.json",
  );
  assert.equal(failureReason({ stderr: "   \n" }), "unknown failure");
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
      // sc-22738: the wedge. A probe that stops climbing and never finishes — the shape
      // \`scail2_14b:bf16:mlx\` held for 90 minutes at a flat 93.5 GB — so the only thing that can
      // end it is the guard's wall-clock budget.
      if (process.env.STUB_CAPTURE_HANGS) { await new Promise((resolve) => { setTimeout(resolve, 600_000); }); }
      // sc-22738: the adapter's exit-1 stderr AFTER protocol::fail was taught to defer its
      // informational notes -- outcome first, flush_deferred_notes after it. The last line on
      // stderr is therefore a note on every failed capture, by construction.
      if (process.env.STUB_CAPTURE_DEFERRED_NOTE) {
        console.error("memory-strategy provider adapter: LTX-2.3 admission rejected an exact-fit calibrated budget: ltx_2_3: incremental live demand 9 exceeds effective budget 7");
        console.error("memory-strategy provider adapter: note: GPU-view coherence retries during this render: mlx_gen=0 mlx_llm=0 (sc-22414)");
        process.exit(1);
      }
      // sc-22738: the adapter's exit-1 stderr on a process-scoped Metal refusal, verbatim from the
      // flux2_dev:bf16:mlx run of 2026-09-06 (paths shortened).
      // sc-22738: the adapter's exit-1 stderr when the PINNED ENGINE's production loader refuses
      // the shipped artifact, verbatim from the wan_2_2_i2v_14b:q4:mlx run of 2026-09-06 (path
      // shortened), preceded by the informational line the arm always prints first.
      if (process.env.STUB_CAPTURE_ARTIFACT_UNSUPPORTED) {
        console.error("GPU-view coherence retries during this render: mlx_gen=0 mlx_llm=0 (sc-22414)");
        console.error("load real wan2_2_i2v_14b q4 provider: unsupported: /hub/q4/high_noise_model.safetensors packed head.head lacks scales");
        process.exit(1);
      }
      if (process.env.STUB_CAPTURE_METAL_REFUSAL) {
        console.error('Error: memory-mlx-adapter exited 1: memory-strategy provider adapter: generate measured render: "[METAL] Command buffer execution failed: Ignored (for causing prior/excessive GPU errors) (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored). at /out/mlx-c-staged/mlx/c/transforms.cpp:73"');
        process.exit(1);
      }
      // sc-22738: the two MLX arms that emit a provider sourceCapture (mlx.rs qwen_source_capture,
      // mlx_ltx25.rs prepare_source_capture) required_env the capture dir UNCONDITIONALLY, before
      // the load -- they do not wait to be asked for a receipt. Both stderr lines below are
      // verbatim from the 2026-09-06 campaign, in order: the informational sc-22414 tally the
      // adapter always prints, then the refusal that actually killed all three ltx_2_5 anchors.
      const captureDir = process.env.SCENEWORKS_MEMORY_CAPTURE_DIR ?? null;
      const sourcePathPrefix = process.env.SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX ?? null;
      if (/^(ltx_2_5|qwen_image):/.test(value("--anchor") ?? "") && !captureDir) {
        console.error("GPU-view coherence retries during this render: mlx_gen=0 mlx_llm=0 (sc-22414)");
        console.error("memory-strategy provider adapter: required environment variable SCENEWORKS_MEMORY_CAPTURE_DIR is not set");
        process.exit(1);
      }
      await writeFile(value("--output"), JSON.stringify({ records: [{ backend: "mlx", target: { modelId: "z_image_turbo", tier: "q4" }, env: process.env.SCENEWORKS_Z_IMAGE_ROOT ?? null, captureDir, sourcePathPrefix, rawLogDir: value("--raw-log-dir") ?? null }] }));
    } else if (command === "record-exceeded") {
      const events = (await readFile(value("--watchdog-events"), "utf8")).trim().split("\\n").map(JSON.parse);
      const stop = events.find((event) => event.event === "hard_stop");
      let observed;
      let ceiling;
      let reason;
      if (stop) {
        // The real harness refuses any hard stop that is not a footprint stop, because only a
        // footprint stop states a lower bound (sc-22738). The stub refuses the same way, so a run
        // that reaches this command with a WALL-CLOCK stop fails loudly here instead of inventing
        // a bound out of a budget.
        const footprint = /^physical_footprint_at_or_above_(\\d+):observed_(\\d+)$/.exec(stop.reason);
        if (!footprint) { console.error("Error: stub record-exceeded: " + stop.reason + " is not a physical-footprint stop"); process.exit(1); }
        [, ceiling, observed] = footprint;
        reason = stop.reason;
      } else {
        // The refusal arm: no hard stop, so the witnesses the runner must have handed over are the
        // adapter's stderr and the wired limit its probe read. Missing either is a stub failure,
        // which is how this stub proves the runner really passed them.
        const stderr = await readFile(value("--provider-stderr") ?? "/nonexistent", "utf8");
        const wired = Number(value("--wired-limit-bytes"));
        if (!stderr.includes("00000004") || !stderr.includes("kIOGPUCommandBufferCallbackErrorSubmissionsIgnored")) {
          console.error("Error: stub record-exceeded was given no refusal to read"); process.exit(1);
        }
        if (!Number.isSafeInteger(wired) || wired <= 0) { console.error("Error: stub record-exceeded got no --wired-limit-bytes"); process.exit(1); }
        observed = Math.max(...events.filter((event) => event.event === "sample").map((event) => event.physicalFootprintBytes));
        ceiling = observed;
        reason = "metal_submissions_ignored:observed_" + observed + ":wired_limit_" + wired;
      }
      await writeFile(value("--output"), JSON.stringify({
        records: [],
        exceededBounds: [{
          id: "exc-stub", backend: "mlx", artifact: JSON.parse(value("--artifact")),
          target: { modelId: "z_image_turbo", tier: "q4" },
          observedFootprintBytes: Number(observed), ceilingBytes: Number(ceiling), reason,
        }],
      }));
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
  // The stub adapter answers the one action the runner itself sends — `probe`, for the watchdog
  // ceilings (sc-22738) — with a 128 GiB host whose wired ceiling is the one this Mac resolves
  // (87,044,670,532 bytes, below the SC-18946 footprint), overridable per test; a candle-shaped
  // probe (no wired limit) is selectable too.
  await writeFile(path.join(root, "stub-adapter.mjs"), `
    let input = "";
    for await (const chunk of process.stdin) input += chunk;
    const request = JSON.parse(input);
    if (request.action !== "probe") { console.error("Error: stub adapter only probes"); process.exit(2); }
    if (process.env.STUB_PROBE_FAILS) { console.error("Error: stub probe refused"); process.exit(1); }
    const hardware = { memoryBytes: Number(process.env.STUB_PROBE_MEMORY_BYTES ?? 137438953472) };
    if (!process.env.STUB_PROBE_NO_WIRED_LIMIT) hardware.wiredLimitBytes = Number(process.env.STUB_PROBE_WIRED_LIMIT_BYTES ?? Math.min(87044670532, hardware.memoryBytes));
    process.stdout.write(JSON.stringify({ hardware }));
  `);
  await writeFile(path.join(root, "config/memory-anchors.json"), JSON.stringify({ anchors: [] }) + "\n");
  await writeFile(path.join(root, "docs/generated/memory-matrix.json"), "{}\n");
  await writeFile(path.join(root, "docs/generated/memory-matrix.md"), "# matrix\n");
  await writeFile(path.join(root, ".gitignore"), "*.log\n");
  // sc-22738: the two Rust builder stages, in the shape `docker/rust.Dockerfile` really has — one
  // contiguous run of evidence `COPY` lines per stage, opened by the memory-calibration evidence
  // line. Every `include_str!` the ingest compiles in has to reach BOTH of them.
  await mkdir(path.join(root, "docker"), { recursive: true });
  await writeFile(path.join(root, DOCKERFILE_PATH), [
    "FROM rust:1-bookworm AS builder",
    "COPY config ./config",
    "# Generated calibration inputs embedded by sceneworks-core.",
    "COPY docs/generated/memory-calibration-evidence.json ./docs/generated/",
    "COPY docs/calibration/sc-18791/ltx25-mlx-evidence.seed.json ./docs/calibration/sc-18791/",
    "COPY docs/generated/qwen-candle-five-rung-sc-15817.json ./docs/generated/",
    "",
    "RUN cargo build --release",
    "",
    "FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 AS candle-builder",
    "COPY config ./config",
    "# Generated calibration inputs embedded by sceneworks-core (see the ordinary builder above).",
    "COPY docs/generated/memory-calibration-evidence.json ./docs/generated/",
    "COPY docs/calibration/sc-18791/ltx25-mlx-evidence.seed.json ./docs/calibration/sc-18791/",
    "COPY docs/generated/qwen-candle-five-rung-sc-15817.json ./docs/generated/",
    "",
    "RUN cargo build --release",
    "",
  ].join("\n"));
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
    args: { adapter: JSON.stringify([process.execPath, "stub-adapter.mjs"]), inferenceRepo: root, commit: true, campaign: "sc-stub" },
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

// sc-22738: the gap this story closes on the ORDINARY commit path too. Every campaign commit added
// an `include_str!` and left `docker/rust.Dockerfile` alone, so each one landed a tree that reds
// `platform-review-contracts.test.mjs`; PR #2759 and the bernini q4 seed both had to add the two
// COPY lines by hand afterwards. The assertion below is that suite's own predicate, applied to the
// tree this run produced.
test("an anchor commit carries the new corpus's COPY line in both Docker builder stages", async () => {
  const checkout = await stubCheckout();
  const tierRoot = await mkdtemp(path.join(tmpdir(), "catalog-tier-"));
  await writeFile(path.join(tierRoot, "w.safetensors"), "weights");
  process.env.GIT_AUTHOR_NAME = process.env.GIT_COMMITTER_NAME = "t";
  process.env.GIT_AUTHOR_EMAIL = process.env.GIT_COMMITTER_EMAIL = "t@t";
  const row = { key: "z_image_turbo:q4:mlx", physical: false, tierRoot, env: { SCENEWORKS_Z_IMAGE_ROOT: tierRoot } };
  const context = stubContext(checkout);
  assert.equal((await measureAnchor(row, context)).status, "committed");
  const dockerfile = await readFile(path.join(checkout.root, DOCKERFILE_PATH), "utf8");
  const rust = await readFile(path.join(checkout.root, PACKAGED_SOURCES_PATH), "utf8");
  const embeds = [...rust.matchAll(/include_str!\("\.\.\/\.\.\/\.\.\/(docs\/(?:generated|calibration)\/[^"\n]+)"\)/g)]
    .map((match) => match[1]);
  assert.ok(embeds.includes("docs/calibration/sc-stub/z-image-turbo-q4-mlx-evidence.json"));
  for (const embed of embeds) {
    // A calibration campaign reaches the builders as its DIRECTORY (sc-22738); `docs/generated/`
    // stays per file. Either way the predicate is the same: present in both builder stages.
    const directory = embed.slice(0, embed.lastIndexOf("/") + 1);
    const copy = embed.startsWith("docs/calibration/")
      ? `COPY ${directory} ./${directory}`
      : `COPY ${embed} ./${directory}`;
    assert.equal(dockerfile.split(copy).length - 1, 2, `${embed} must reach both builder contexts`);
  }
  const { stdout: shown } = await checkout.git("show", "--stat", "--format=%s", "HEAD");
  assert.ok(shown.includes(DOCKERFILE_PATH), "the Dockerfile moves in the SAME commit as the embed");
  assert.equal((await checkout.git("status", "--porcelain")).stdout, "", "nothing is left behind for a by-hand repair");
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
  // sc-22738: end to end, through the real child process, with the adapter's deferred note LAST.
  // Dropping the informational skip in `failureReason` reds this on the note line.
  process.env.STUB_CAPTURE_DEFERRED_NOTE = "1";
  try {
    const result = await measureAnchor(row, stubContext(checkout));
    assert.equal(result.status, "capture_failed");
    assert.match(result.reason, /rejected an exact-fit calibrated budget/);
    assert.doesNotMatch(result.reason, /coherence retries/, "the deferred note is not the outcome");
    assert.equal((await checkout.git("status", "--porcelain")).stdout, "");
  } finally {
    delete process.env.STUB_CAPTURE_DEFERRED_NOTE;
  }
});

test("a source-capture anchor's gitignored .log receipt is copied beside the evidence and force-added; its rendered outputs are not", async () => {
  const checkout = await stubCheckout();
  const row = { key: "z_image_turbo:q4:mlx", sourceCapture: true, env: {} };
  const context = stubContext(checkout);
  // The harness would write the receipts under <rawLogDir>/<campaignDir>/; emulate that, with the
  // rendered selected/reference outputs the physical MLX arms write beside them (sc-22738: the
  // LTX-2.5 A/V pair is 165–176 MB EACH and GitHub refused the three commits that carried it).
  const rawLogDir = path.join(checkout.workDir, "raw", "z-image-turbo-q4-mlx");
  const receipts = path.join(rawLogDir, "docs/calibration/sc-stub");
  await mkdir(receipts, { recursive: true });
  const session = "ims-0123456789abcdef0123";
  const rendered = [
    `implan-0123456789abcdef0123-selected_av-512x768-f145-${"a".repeat(64)}.avbin`,
    `implan-0123456789abcdef0123-reference_av-512x768-f145-${"a".repeat(64)}.avbin`,
    `implan-0123456789abcdef0123-selected_rgb-1024x1024-${"b".repeat(64)}.rgb`,
    "render.png", "latents.npy", "audio.wav",
  ];
  await writeFile(path.join(receipts, `${session}.log`), "receipt");
  await writeFile(path.join(receipts, `${session}.request.json`), "{}");
  for (const name of rendered) await writeFile(path.join(receipts, name), "rendered bytes");
  const result = await measureAnchor(row, context);
  assert.equal(result.status, "committed", result.reason);
  // `--name-only`: `--stat` abbreviates long paths, and every name below is a long one.
  const { stdout: shown } = await checkout.git("show", "--name-only", "--format=%s", "HEAD");
  assert.ok(shown.includes(`docs/calibration/sc-stub/${session}.log`), "the *.log receipt is committed despite the ignore rule");
  assert.ok(shown.includes(`docs/calibration/sc-stub/${session}.request.json`), "and the request receipt beside it");
  const { stdout: tracked } = await checkout.git("ls-files", "docs/calibration/sc-stub");
  const committed = await readdir(path.join(checkout.root, "docs/calibration/sc-stub"));
  for (const name of rendered) {
    assert.ok(!shown.includes(name), `${name} is not in the commit`);
    assert.ok(!tracked.includes(name), `${name} is not tracked`);
    assert.ok(!committed.includes(name), `${name} never entered the evidence directory`);
  }
  assert.equal((await checkout.git("status", "--porcelain")).stdout, "");
  const log = await readFile(result.log, "utf8");
  for (const name of rendered) assert.ok(log.includes(`excluded rendered output ${name}`), `the anchor log names the excluded ${name}`);
});

// sc-22738: the ceiling behind the rule above, as a refusal in its own right — a runner that somehow
// reaches `git add` with a render in hand stops before the commit exists, not at the push.
test("the ingest refuses to stage any file over 50 MB, rolls back, and names the file", async () => {
  assert.equal(MAX_STAGED_FILE_BYTES, 50 * 1024 * 1024);
  assert.equal(receiptDisposition("ims-0123456789abcdef0123.log"), "receipt");
  assert.equal(receiptDisposition("ims-0123456789abcdef0123.request.json"), "receipt");
  for (const name of [
    `implan-0123456789abcdef0123-selected_av-512x768-f145-${"a".repeat(64)}.avbin`,
    `implan-0123456789abcdef0123-reference_rgb-1024x1024-${"b".repeat(64)}.rgb`,
    "session.log", "ims-0123456789abcdef0123.png", "ims-0123456789abcdef0123.log.bak",
  ]) assert.equal(receiptDisposition(name), "rendered_output", name);

  const checkout = await stubCheckout();
  const row = { key: "z_image_turbo:q4:mlx", sourceCapture: true, env: {} };
  const context = stubContext(checkout);
  const receipts = path.join(checkout.workDir, "raw", "z-image-turbo-q4-mlx", "docs/calibration/sc-stub");
  await mkdir(receipts, { recursive: true });
  // A receipt-shaped name carrying one byte over the ceiling (sparse, so the test costs no disk).
  const oversized = path.join(receipts, "ims-0123456789abcdef0123.log");
  const handle = await open(oversized, "w");
  await handle.truncate(MAX_STAGED_FILE_BYTES + 1);
  await handle.close();
  assert.equal((await stat(oversized)).size, MAX_STAGED_FILE_BYTES + 1);
  await assert.rejects(
    assertStageable(path.dirname(receipts), ["sc-stub"]),
    /refusing to stage 1 file\(s\) over the 52428800-byte \(50 MB\) staging ceiling[\s\S]*sc-stub\/ims-0123456789abcdef0123\.log \(52428801 bytes\)/,
  );
  await assertStageable(path.dirname(receipts), ["sc-stub"], MAX_STAGED_FILE_BYTES + 1);

  const result = await measureAnchor(row, context);
  assert.equal(result.status, "ingest_failed");
  assert.match(result.reason, /refusing to stage 1 file\(s\) over the 52428800-byte \(50 MB\) staging ceiling/);
  assert.equal(context.state.commits.length, 0, "no commit was made");
  assert.equal(context.state.halt, null);
  assert.equal((await checkout.git("status", "--porcelain")).stdout, "", "rollback left a clean tree");
  const { stdout: tracked } = await checkout.git("ls-files", "docs/calibration/sc-stub");
  assert.equal(tracked, "", "nothing under the campaign directory was staged");
});

// sc-22738. The defect this story fixes, driven end to end: `ltx_2_5:*:mlx` classified runnable,
// was scheduled, and died inside the adapter on
// `required environment variable SCENEWORKS_MEMORY_CAPTURE_DIR is not set` — for bf16 after 883
// seconds, because the harness re-hashes the LTX-2.5 snapshot before the adapter is ever spawned.
// The row flag is what the runner reads, so drive `measureAnchor` with it set and with it dropped:
// the second half is the mutation, and it reproduces the exact failure the campaign hit.
test("an ltx_2_5 mlx anchor's capture carries the capture-dir pair its arm requires", async () => {
  const checkout = await stubCheckout();
  const key = "ltx_2_5:q4:mlx";
  const slug = anchorSlug(key);
  const context = stubContext(checkout);
  const rawLogDir = path.join(checkout.workDir, "raw", slug);
  await mkdir(path.join(rawLogDir, "docs/calibration/sc-stub"), { recursive: true });
  const result = await measureAnchor(
    { key, sourceCapture: true, physical: false, ltx25SnapshotRoot: "/snapshot", env: {} },
    context,
  );
  assert.equal(result.status, "committed", result.reason);
  const evidence = JSON.parse(await readFile(
    path.join(checkout.root, `docs/calibration/sc-stub/${slug}-evidence.json`),
    "utf8",
  ));
  assert.equal(evidence.records[0].captureDir, rawLogDir, "the arm was handed its raw-log directory");
  assert.equal(evidence.records[0].sourcePathPrefix, "docs/calibration/sc-stub", "and the campaign prefix it writes under");
  assert.equal(evidence.records[0].rawLogDir, rawLogDir, "and the harness was told to expect the receipt there");

  // MUTATION: the pre-fix runner, which set the pair for the qwen_image family alone.
  const dropped = await measureAnchor(
    { key, sourceCapture: false, physical: false, ltx25SnapshotRoot: "/snapshot", env: {} },
    stubContext(await stubCheckout()),
  );
  assert.equal(dropped.status, "capture_failed");
  assert.match(dropped.reason, /required environment variable SCENEWORKS_MEMORY_CAPTURE_DIR is not set/);
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

// ---------------------------------------------------------------------------------------------
// sc-22738: the three MLX LTX-2.3 cells are driven by THIS runner, through production admission
// ---------------------------------------------------------------------------------------------

// The campaign found the old shape the expensive way: three anchors, three model-load-free
// failures, one per tier — `SC-18946 row is missing required _measurementSafety; refusing before
// model load`. The arm routed the harness's one action (`run`) to `LtxRunAdmission::Ordinary`,
// whose whole body was `refuse_unsafe_ltx_capture`: a demand for a block the anchor-plan schema
// (`additionalProperties: false`) cannot carry, followed by an unconditional `Err`. A guard with no
// success path is not a safety check; it is a hard-coded "never measure", and E4 requires the
// record to measure what ships. PR #2745 answered by classifying the cells `harness_unsupported`
// and delegating them to the safety canary; this supersedes that. The cells are ordinary anchors:
// the arm admits them through the production budget and the runner contains them with the same
// footprint hard stop it puts around every Darwin capture.
test("the MLX LTX-2.3 cells are ordinary anchors: admitted by the production budget, classified runnable or weights_missing", async () => {
  const adapter = await readFile(path.join(ROOT, ADAPTER_BIN_PATHS.mlx), "utf8");
  const harness = await readFile(path.join(ROOT, "scripts/memory-calibration-harness.mjs"), "utf8");
  // 1. The generic runner still sends exactly one provider action for a capture...
  assert.deepEqual(
    [...harness.matchAll(/action: "(\w+)",\n\s*planned/g)].map((match) => match[1]),
    ["run"],
    "the harness's capture action moved; the assertions below read the wrong dispatch arm",
  );
  // 2. ...and the arm's ordinary path carries no refusal of its own between the plan and the load:
  //    the unconditional guard is gone, the Ordinary admission arm is empty, and what stands before
  //    the load is the production admission (`ltx_ordinary_admission`), which returns the budget's
  //    own refusal or proceeds.
  // Assembled so the adapter's own source-shape test literal cannot satisfy the search.
  assert.equal(adapter.includes(["fn", "refuse_unsafe_ltx_capture("].join(" ")), false, "the SC-19642 unconditional refusal is back");
  const arm = adapter.slice(adapter.indexOf("fn run_ltx_with_admission("));
  const preLoad = arm.slice(0, arm.indexOf(".load(LTX_PROVIDER, &spec)"));
  assert.match(preLoad, /LtxRunAdmission::Ordinary => \{\}/, "the Ordinary admission arm must be empty");
  assert.ok(preLoad.indexOf("ltx_ordinary_admission(") > preLoad.indexOf("LtxRunAdmission::Ordinary => {}"), "the production admission stands before the load");
  assert.doesNotMatch(preLoad, /_measurementSafety/, "the ordinary path must not demand a block the plan schema cannot carry");
  // 3. Every planned MLX LTX-2.3 cell classifies as an ordinary anchor on this lane. There are
  //    exactly the manifest's three tiers, and none is delegated anywhere.
  const plan = await readPlan();
  const keys = Object.keys(plan.anchors).filter((key) => key.startsWith("ltx_2_3:") && key.endsWith(":mlx")).sort();
  assert.deepEqual(keys, ["ltx_2_3:bf16:mlx", "ltx_2_3:q4:mlx", "ltx_2_3:q8:mlx"]);
  const { rows } = await planRun({ backend: "mlx", anchors: null, campaign: "sc-catalog-test", hfCache: [], skipCurrent: false });
  const byKey = new Map(rows.map((row) => [row.key, row]));
  for (const key of keys) {
    const row = byKey.get(key);
    assert.ok(["runnable", "weights_missing"].includes(row.status), `${key}: ${row.status} (${row.reason})`);
    assert.ok(guardsCapture(key, "darwin"), `${key} runs under the footprint guard`);
  }
  // 4. The plan carries no composition override for them: the lane default (`resident`) is what
  //    the worker's selector ships for this cell on a host that fits it, and the adapter's own
  //    plan-driven test proves the pinned contract admits it.
  for (const key of keys) assert.equal(plan.anchors[key].strategy, undefined, `${key} plans the lane default composition`);
});

// ---------------------------------------------------------------------------------------------
// sc-22738: physical containment — every Darwin capture runs under the footprint hard stop
// ---------------------------------------------------------------------------------------------

test("the watchdog guard's footprint hard stop is the incident and host RAM, never the wired limit", () => {
  // This Mac's probe: 128 GiB, wired ceiling 87,044,670,532.
  const hardware = { memoryBytes: 137_438_953_472, wiredLimitBytes: 87_044_670_532 };
  const guard = watchdogGuard({ hardware, eventFile: "/tmp/events.jsonl", budgetMinutes: 60 });
  assert.equal(guard[0], path.join(ROOT, WATCHDOG));
  const flag = (name) => Number(guard[guard.indexOf(name) + 1]);
  // The wired limit caps Metal buffers; the guard samples the kernel `phys_footprint`, which counts
  // non-Metal pages too. As a hard stop it killed `flux2_dev:bf16` at 87,140,069,928 bytes and
  // 660 s — an anchor this same host had already rendered to completion unguarded (sc-22738,
  // measured 2026-09-06). It is advisory; the kill line is the incident less the reserve.
  assert.equal(flag("--max-footprint-bytes"), 94_822_600_832, "the incident less the reserve binds, not the 87,044,670,532 wired ceiling");
  assert.ok(flag("--max-footprint-bytes") > 87_140_069_928, "the measured flux2_dev:bf16 peak is below the kill line");
  assert.equal(flag("--host-memory-bytes"), 137_438_953_472);
  assert.equal(flag("--min-memory-free-bytes"), UNIFIED_RESERVE_BYTES, "the whole-host floor is the absolute 2 GiB reserve, not the non-wired remainder");
  assert.equal(guard[guard.indexOf("--event-file") + 1], "/tmp/events.jsonl");
  // The footprint ceiling is below the incident, and the two ceilings together leave room for
  // the host's own baseline: the guarded group can reach its ceiling without the whole-host floor
  // firing first (sc-22738 review: `memoryBytes − wiredLimitBytes` as the floor turned the
  // per-group ceiling into a cap on total host use).
  assert.ok(flag("--max-footprint-bytes") < LTX_Q4_F305_CRASH_FOOTPRINT_BYTES);
  assert.ok(flag("--min-memory-free-bytes") + flag("--max-footprint-bytes") <= hardware.memoryBytes - 32 * 1024 * 1024 * 1024, "≥ 32 GiB of baseline room");
  // The generic guard speaks no attestation or phase protocol — those belong to the frozen canary
  // profiles.
  for (const absent of ["--require-child-attestation", "--require-provider-phases", "--provider-phase-profile"]) {
    assert.equal(guard.includes(absent), false, absent);
  }
  // It DOES set a wall-time ceiling since sc-22738 (2026-09-07): `scail2_14b:bf16:mlx` sat flat at
  // 93.5 GB under this 94.82 GB kill line for 90 minutes with the host swapping, because nothing
  // ever passed the `--max-runtime-seconds` the guard has always supported. Minutes in, seconds out.
  assert.equal(flag("--max-runtime-seconds"), 3_600);
  assert.equal(Number(watchdogGuard({ hardware, eventFile: "x", budgetMinutes: 150 })[guard.indexOf("--max-runtime-seconds") + 1]), 9_000);
  assert.equal(Number(watchdogGuard({ hardware, eventFile: "x", budgetMinutes: 0.05 })[guard.indexOf("--max-runtime-seconds") + 1]), 3);
  // REQUIRED, not defaulted: an omitted budget is refused rather than quietly spawning the
  // unbounded probe this story exists to make impossible.
  assert.throws(() => watchdogGuard({ hardware, eventFile: "x" }), /positive budgetMinutes/);
  for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, "60"]) {
    assert.throws(() => watchdogGuard({ hardware, eventFile: "x", budgetMinutes: bad }), /positive budgetMinutes/, String(bad));
  }
  // The ceiling is the SMALLER of the incident less the reserve and host RAM less the reserve, so
  // no host policy can raise the kill line to or above the incident — and the wired limit is inert
  // in the derivation: below it, above it, or absent, the ceiling on this host is the same.
  const incidentBound = LTX_Q4_F305_CRASH_FOOTPRINT_BYTES - UNIFIED_RESERVE_BYTES;
  assert.equal(incidentBound, 94_822_600_832);
  for (const wiredLimitBytes of [87_044_670_532, 103_079_215_104, 137_438_953_472, undefined]) {
    assert.deepEqual(watchdogCeilings({ memoryBytes: 137_438_953_472, wiredLimitBytes }), { maxFootprintBytes: incidentBound, minMemoryFreeBytes: UNIFIED_RESERVE_BYTES }, `wired limit ${wiredLimitBytes} is not a hard-stop term`);
  }
  assert.deepEqual(watchdogCeilings({ memoryBytes: 68_719_476_736 }), { maxFootprintBytes: 68_719_476_736 - UNIFIED_RESERVE_BYTES, minMemoryFreeBytes: UNIFIED_RESERVE_BYTES }, "a 64 GiB host is bound by its own RAM less the reserve");
  for (const ceilings of [
    watchdogCeilings({ memoryBytes: 137_438_953_472 }),
    watchdogCeilings({ memoryBytes: 68_719_476_736 }),
  ]) {
    assert.ok(ceilings.maxFootprintBytes < LTX_Q4_F305_CRASH_FOOTPRINT_BYTES);
    assert.ok(ceilings.maxFootprintBytes + ceilings.minMemoryFreeBytes <= 137_438_953_472);
  }
  assert.throws(() => watchdogGuard({ hardware: { memoryBytes: 1, wiredLimitBytes: 2 }, eventFile: "x", budgetMinutes: 60 }), /above host memory/);
  assert.throws(() => watchdogGuard({ hardware: { memoryBytes: 1, wiredLimitBytes: 0 }, eventFile: "x", budgetMinutes: 60 }), /wiredLimitBytes/);
  assert.throws(() => watchdogGuard({ hardware: { wiredLimitBytes: 2 }, eventFile: "x", budgetMinutes: 60 }), /memoryBytes/);
  assert.throws(() => watchdogGuard({ hardware: { memoryBytes: UNIFIED_RESERVE_BYTES }, eventFile: "x", budgetMinutes: 60 }), /no positive footprint ceiling/);
  // Guarded set: EVERY capture on Darwin — a candle capture on a Mac draws from the same unified
  // pool the sampler measures — and none elsewhere, where the footprint sampler does not exist.
  assert.equal(guardsCapture("z_image_turbo:q4:mlx", "darwin"), true);
  assert.equal(guardsCapture("z_image_turbo:q4:candle", "darwin"), true);
  assert.equal(guardsCapture("z_image_turbo:q4:mlx", "linux"), false);
  assert.equal(guardsCapture("z_image_turbo:q4:candle", "linux"), false);
  assert.throws(() => guardsCapture("not-an-anchor", "darwin"), /not an anchor key/);
});

test("the runner's incident footprint and unified reserve are the Rust declarations, not second literals", async () => {
  const adapter = await readFile(path.join(ROOT, "crates/sceneworks-memory-adapter/src/bin/mlx.rs"), "utf8");
  const crash = /const LTX_Q4_F305_CRASH_FOOTPRINT_BYTES: u64 = ([\d_]+);/.exec(adapter);
  assert.ok(crash, "the adapter declares the incident footprint");
  assert.equal(LTX_Q4_F305_CRASH_FOOTPRINT_BYTES, Number(crash[1].replaceAll("_", "")));
  assert.equal(LTX_Q4_F305_CRASH_FOOTPRINT_BYTES, 96_970_084_480);
  const core = await readFile(path.join(ROOT, "crates/sceneworks-core/src/memory_anchor.rs"), "utf8");
  const reserve = /pub const LEGACY_UNIFIED_FALLBACK_RESERVE_GB: f64 = ([\d.]+);/.exec(core);
  assert.ok(reserve, "sceneworks-core declares the unified reserve the worker and the adapter read");
  assert.equal(UNIFIED_RESERVE_BYTES, Number(reserve[1]) * 1024 * 1024 * 1024);
  // And the worker READS it rather than restating it.
  const fitGate = await readFile(path.join(ROOT, "crates/sceneworks-worker/src/fit_gate.rs"), "utf8");
  assert.match(fitGate, /LEGACY_UNIFIED_FALLBACK_RESERVE_GB: f64 =\s*sceneworks_core::memory_anchor::LEGACY_UNIFIED_FALLBACK_RESERVE_GB;/);
  const mlxFitGate = await readFile(path.join(ROOT, "crates/sceneworks-worker/src/mlx_fit_gate.rs"), "utf8");
  assert.match(mlxFitGate, /const HEADROOM_GB: f64 = sceneworks_core::memory_anchor::MLX_GENERIC_HEADROOM_GB;/);
});

test("every Darwin anchor capture runs inside the footprint watchdog with the derived ceilings", async () => {
  const checkout = await stubCheckout();
  const context = stubContext(checkout, { args: { ...stubContext(checkout).args, commit: false } });
  const hardware = await probeAdapter([process.execPath, "stub-adapter.mjs"], { cwd: checkout.root });
  assert.deepEqual(hardware, { memoryBytes: 137438953472, wiredLimitBytes: 87044670532 });
  const mlx = await measureAnchor({ key: "z_image_turbo:q4:mlx", physical: false, env: {} }, context);
  const mlxLog = await readFile(mlx.log, "utf8");
  if (process.platform === "darwin") {
    assert.equal(mlx.status, "captured", mlx.reason);
    assert.match(mlxLog, /\/usr\/bin\/python3 \S+\/scripts\/memory-calibration-watchdog\.py --max-footprint-bytes 94822600832 --host-memory-bytes 137438953472 --min-memory-free-bytes 2147483648 /);
    const events = (await readFile(path.join(checkout.workDir, "logs", "z-image-turbo-q4-mlx-watchdog.jsonl"), "utf8")).trim().split("\n").map(JSON.parse);
    assert.ok(events.some((event) => event.event === "started"), "the guard owned the capture's process group");
    assert.ok(events.some((event) => event.event === "sample" && Number.isSafeInteger(event.physicalFootprintBytes)), "the guard sampled the capture's physical footprint");
    assert.ok(!events.some((event) => event.event === "hard_stop"), "a stub capture stays under the ceiling");
    // A refused probe is a refused capture: no ceiling, no run.
    process.env.STUB_PROBE_FAILS = "1";
    try {
      const refused = await measureAnchor({ key: "z_image_turbo:q4:mlx", physical: false, env: {} }, context);
      assert.equal(refused.status, "capture_failed");
      assert.match(refused.reason, /stub probe refused/);
    } finally {
      delete process.env.STUB_PROBE_FAILS;
    }
    // A candle-shaped probe (no wired limit) is refused for an MLX anchor: the MLX adapter always
    // resolves one, so its absence is a broken probe, not a smaller bound set.
    process.env.STUB_PROBE_NO_WIRED_LIMIT = "1";
    try {
      const refused = await measureAnchor({ key: "z_image_turbo:q4:mlx", physical: false, env: {} }, context);
      assert.equal(refused.status, "capture_failed");
      assert.match(refused.reason, /no positive hardware\.wiredLimitBytes/);
      // The same probe guards a candle anchor from the host figure alone: the ceiling is the
      // incident less the reserve.
      const candle = await measureAnchor({ key: "z_image_turbo:q4:candle", physical: false, env: {} }, context);
      assert.equal(candle.status, "captured", candle.reason);
      assert.match(await readFile(candle.log, "utf8"), /memory-calibration-watchdog\.py --max-footprint-bytes 94822600832 --host-memory-bytes 137438953472 --min-memory-free-bytes 2147483648 /);
      const candleEvents = (await readFile(path.join(checkout.workDir, "logs", "z-image-turbo-q4-candle-watchdog.jsonl"), "utf8")).trim().split("\n").map(JSON.parse);
      assert.ok(candleEvents.some((event) => event.event === "started"), "the candle capture ran under the guard too");
    } finally {
      delete process.env.STUB_PROBE_NO_WIRED_LIMIT;
    }
  } else {
    assert.doesNotMatch(mlxLog, /memory-calibration-watchdog/, "the footprint sampler is Darwin-only");
    const candle = await measureAnchor({ key: "z_image_turbo:q4:candle", physical: false, env: {} }, context);
    assert.equal(candle.status, "captured", candle.reason);
    assert.doesNotMatch(await readFile(candle.log, "utf8"), /memory-calibration-watchdog/);
  }
});

test("a footprint hard stop is COMMITTED as a measured lower bound, through the same ingest path a capture takes", { skip: process.platform !== "darwin" && "the footprint sampler is Darwin-only" }, async () => {
  const checkout = await stubCheckout();
  const tierRoot = await mkdtemp(path.join(tmpdir(), "catalog-tier-"));
  await writeFile(path.join(tierRoot, "w.safetensors"), "weights");
  process.env.GIT_AUTHOR_NAME = process.env.GIT_COMMITTER_NAME = "t";
  process.env.GIT_AUTHOR_EMAIL = process.env.GIT_COMMITTER_EMAIL = "t@t";
  process.env.STUB_PROBE_MEMORY_BYTES = String(2 * 1024 * 1024 * 1024 + 1);
  try {
    const row = {
      key: "z_image_turbo:q4:mlx", physical: false, tierRoot,
      env: { SCENEWORKS_Z_IMAGE_ROOT: tierRoot },
      artifact: { repository: "SceneWorks/z-image-turbo-mlx", resolvedRevision: REVISION, variant: "q4" },
    };
    const context = stubContext(checkout);
    const stopped = await measureAnchor(row, context);
    // THE defect this story fixes: the run reached a footprint the host could not carry, and until
    // now that fact was dropped on the floor while production kept admitting the same request.
    assert.equal(stopped.status, "committed_exceeded", stopped.reason);
    assert.equal(context.state.commits.length, 1, "the bound is one commit, like an anchor");
    const { stdout: status } = await checkout.git("status", "--porcelain");
    assert.equal(status, "", "the tree is clean again for the next capture");
    const { stdout: shown } = await checkout.git("show", "--stat", "--format=%s%n%b", "HEAD");
    assert.match(shown, /chore\(sc-stub\): record the z_image_turbo:q4:mlx footprint hard stop as a measured bound/);
    assert.match(shown, /watchdog hard stop: physical_footprint_at_or_above_1:observed_\d+/);
    for (const file of [
      "docs/calibration/sc-stub/z-image-turbo-q4-mlx-exceeded-evidence.json",
      PACKAGED_SOURCES_PATH, "config/memory-anchors.json", "docs/generated/memory-matrix.json",
    ]) {
      assert.ok(shown.includes(file), `${file} is in the commit`);
    }
    const rust = await readFile(path.join(checkout.root, PACKAGED_SOURCES_PATH), "utf8");
    assert.ok(
      rust.includes('"docs/calibration/sc-stub/z-image-turbo-q4-mlx-exceeded-evidence.json"'),
      "the bound's corpus is compiled into the Rust loader, exactly as an anchor's is",
    );
    // ...and its campaign directory is copied into both Docker builder stages by the same step
    // (sc-22738), so a hard-stop commit is not a tree that reds `platform-review-contracts.test.mjs`
    // until someone repairs it.
    const dockerfile = await readFile(path.join(checkout.root, DOCKERFILE_PATH), "utf8");
    assert.equal(
      dockerfile.split("COPY docs/calibration/sc-stub/ ./docs/calibration/sc-stub/").length - 1,
      2,
    );
    assert.ok(shown.includes(DOCKERFILE_PATH), "the Dockerfile is in the bound's commit");
    const bundle = JSON.parse(await readFile(
      path.join(checkout.root, "docs/calibration/sc-stub/z-image-turbo-q4-mlx-exceeded-evidence.json"), "utf8",
    ));
    assert.equal(bundle.exceededBounds.length, 1);
    assert.equal(bundle.exceededBounds[0].ceilingBytes, 1);
    assert.ok(bundle.exceededBounds[0].observedFootprintBytes > 0);
    assert.deepEqual(
      bundle.exceededBounds[0].artifact,
      { repository: "SceneWorks/z-image-turbo-mlx", resolvedRevision: REVISION, variant: "q4", inventorySha256: bundle.exceededBounds[0].artifact.inventorySha256 },
      "the bound names the weights the guarded render had open",
    );
    assert.match(bundle.exceededBounds[0].artifact.inventorySha256, /^[0-9a-f]{64}$/);
  } finally {
    delete process.env.STUB_PROBE_MEMORY_BYTES;
  }
});

test("a footprint hard stop with no artifact binding is still a capture_failed, never a bound over unnamed weights", { skip: process.platform !== "darwin" && "the footprint sampler is Darwin-only" }, async () => {
  const checkout = await stubCheckout();
  process.env.STUB_PROBE_MEMORY_BYTES = String(2 * 1024 * 1024 * 1024 + 1);
  try {
    const stopped = await measureAnchor(
      { key: "z_image_turbo:q4:mlx", physical: false, env: {} },
      stubContext(checkout),
    );
    assert.equal(stopped.status, "capture_failed");
    assert.match(stopped.reason, /no artifact binding for this row to bound it against/);
    const { stdout: status } = await checkout.git("status", "--porcelain");
    assert.equal(status, "", "nothing was committed");
  } finally {
    delete process.env.STUB_PROBE_MEMORY_BYTES;
  }
});

test("a footprint hard stop on a no-commit run surfaces as `exceeded` naming the stop, with the tree left clean", { skip: process.platform !== "darwin" && "the footprint sampler is Darwin-only" }, async () => {
  const checkout = await stubCheckout();
  const context = stubContext(checkout, { args: { ...stubContext(checkout).args, commit: false } });
  // A ceiling every process is already above: the guard's initial observation of the held group
  // fires before the child is released, exactly the path a runaway load takes at the ceiling. The
  // ceiling is host RAM less the 2 GiB reserve, so a probe reporting one byte more than the reserve
  // drives it to 1 — the wired limit cannot do this any more, and is not a hard-stop term.
  process.env.STUB_PROBE_MEMORY_BYTES = String(2 * 1024 * 1024 * 1024 + 1);
  try {
    const stopped = await measureAnchor({
      key: "z_image_turbo:q4:mlx", physical: false, env: {},
      artifact: { repository: "SceneWorks/z-image-turbo-mlx", resolvedRevision: REVISION, variant: "q4" },
    }, context);
    // sc-22738: the stop is no longer a dropped capture. On a NO-COMMIT run there is nowhere to
    // put the evidence — ingesting would dirty the tree — so it is named and carried out as its
    // own outcome rather than as a failure; the committing run below turns the same stop into a
    // packaged bound.
    assert.equal(stopped.status, "exceeded");
    assert.match(stopped.reason, /^watchdog hard stop: physical_footprint_at_or_above_1:observed_\d+$/);
    const log = await readFile(stopped.log, "utf8");
    assert.match(log, /memory-calibration-watchdog\.py --max-footprint-bytes 1 --host-memory-bytes 2147483649 --min-memory-free-bytes 2147483648 /);
    const eventFile = path.join(checkout.workDir, "logs", "z-image-turbo-q4-mlx-watchdog.jsonl");
    const events = (await readFile(eventFile, "utf8")).trim().split("\n").map(JSON.parse);
    const hardStop = events.find((event) => event.event === "hard_stop");
    assert.ok(hardStop, "the guard recorded the hard stop");
    assert.match(hardStop.reason, /^physical_footprint_at_or_above_1:observed_\d+$/);
    assert.ok(events.some((event) => event.event === "terminated"), "the guard terminated the group");
    assert.equal(await watchdogHardStop(eventFile), `watchdog hard stop: ${hardStop.reason}`);
    assert.equal(await watchdogHardStop(path.join(checkout.workDir, "logs", "never-written.jsonl")), null);
    const { stdout: status } = await checkout.git("status", "--porcelain");
    assert.equal(status, "", "a stopped capture leaves the checkout clean for the next anchor");
  } finally {
    delete process.env.STUB_PROBE_MEMORY_BYTES;
  }
});

// ---------------------------------------------------------------------------------------------
// sc-22738: the per-probe WALL-CLOCK budget, and the one hard stop that measures nothing.
// ---------------------------------------------------------------------------------------------

test("every probe carries a wall-clock budget: per lane by default, derived from the plan, overridable", async () => {
  // The table, and the flag that overrides it.
  assert.deepEqual(PROBE_BUDGET_MINUTES, { video: 270, image: 60 });
  assert.equal(probeBudgetMinutes({ videoLane: true }), 270);
  assert.equal(probeBudgetMinutes({ videoLane: false }), 60);
  // sc-22738 (2026-09-08): each lane's budget rests on the longest COMPLETED capture it has
  // witnessed, with the margin the table's own doc comment states. The video witness is the
  // 8,709 s `scail2_14b:bf16:mlx` capture that a 150-minute budget cleared by 3% and a 90-minute
  // campaign then stopped three times over; a budget that no longer clears its witness by that
  // margin is the defect this pins, whichever entry drifts.
  assert.deepEqual(WITNESSED_CAPTURE_SECONDS, { video: 8_709, image: 847 });
  assert.ok(PROBE_BUDGET_MINUTES.video * 60 >= WITNESSED_CAPTURE_SECONDS.video * 1.8, "video budget clears its witness by 1.8x");
  assert.ok(PROBE_BUDGET_MINUTES.image * 60 >= WITNESSED_CAPTURE_SECONDS.image * 4, "image budget clears its witness by 4x");
  assert.equal(probeLane({ videoLane: true }), "video");
  assert.equal(probeLane({ videoLane: false }), "image");
  assert.equal(probeBudgetMinutes({ videoLane: true }, 12), 12, "the flag overrides both lanes");
  assert.equal(probeBudgetMinutes({ videoLane: false }, 0.5), 0.5);
  assert.throws(() => probeBudgetMinutes({ videoLane: false }, 0), /positive number of minutes/);
  assert.equal(parseArgs(["--backend", "mlx", "--list"]).probeBudgetMinutes, null, "unset means the lane default");
  assert.equal(parseArgs(["--backend", "mlx", "--list", "--probe-budget-minutes", "45"]).probeBudgetMinutes, 45);
  for (const bad of ["0", "-3", "abc"]) {
    assert.throws(() => parseArgs(["--backend", "mlx", "--list", "--probe-budget-minutes", bad]), /positive number of minutes/, bad);
  }
  // The LANE is derived from the plan, never listed here: a video anchor is one whose planned
  // geometry renders more than one frame. Both video and image rows must exist for the check to
  // mean anything, and `bernini` (the video entry) versus `bernini_image` (the still one) is the
  // pair that makes the derivation worth having — one family, two lanes, two budgets.
  const plan = await readPlan();
  const { rows } = await planRun({ backend: "mlx", anchors: null, campaign: "sc-catalog-test", hfCache: [], skipCurrent: false });
  let video = 0;
  for (const row of rows) {
    const frames = plan.anchors[row.key].geometry?.frames ?? 1;
    assert.equal(row.videoLane, frames > 1, `${row.key} renders ${frames} frame(s)`);
    assert.equal(probeBudgetMinutes(row), frames > 1 ? 270 : 60, row.key);
    if (frames > 1) video += 1;
  }
  assert.ok(video > 0 && video < rows.length, `${video} of ${rows.length} mlx rows are video anchors`);
  const byKey = new Map(rows.map((row) => [row.key, row]));
  assert.equal(byKey.get("bernini:q8:mlx").videoLane, true);
  assert.equal(byKey.get("bernini_image:q8:mlx").videoLane, false);
});

test("a runtime stop is recognised by its spelling and reports the guard's sampled peak", async () => {
  assert.equal(runtimeBudgetStopSeconds("watchdog hard stop: runtime_at_or_above_9000.0s"), 9000);
  assert.equal(runtimeBudgetStopSeconds("watchdog hard stop: runtime_at_or_above_3s"), 3);
  // Everything that is NOT a wall-clock stop keeps its own arm — above all the footprint stop,
  // which DOES state a bound and must still reach `record-exceeded`.
  for (const other of [
    null, "", "watchdog hard stop: physical_footprint_at_or_above_94822600832:observed_94822600833",
    "watchdog hard stop: telemetry_lost:TimeoutError:x", "watchdog hard stop: monitor_signal_SIGTERM",
    "watchdog hard stop: runtime_at_or_above_9000",
  ]) {
    assert.equal(runtimeBudgetStopSeconds(other), null, String(other));
  }
  assert.ok(RUNTIME_BUDGET_STOP_PATTERN.test("watchdog hard stop: runtime_at_or_above_150.5s"));
  // The peak is read for the operator, so it never throws on the failure path.
  const dir = await mkdtemp(path.join(tmpdir(), "catalog-peak-"));
  const file = path.join(dir, "events.jsonl");
  await writeFile(file, [
    JSON.stringify({ event: "started" }),
    JSON.stringify({ event: "sample", physicalFootprintBytes: 3 }),
    JSON.stringify({ event: "sample", physicalFootprintBytes: 93_500_000_000 }),
    JSON.stringify({ event: "sample", physicalFootprintBytes: 12 }),
    "{ truncated",
    "",
  ].join("\n"));
  assert.deepEqual(await watchdogPeakFootprint(file), { peakBytes: 93_500_000_000, samples: 3 }, "the PEAK, not the last sample");
  await writeFile(path.join(dir, "empty.jsonl"), "");
  assert.equal(await watchdogPeakFootprint(path.join(dir, "empty.jsonl")), null);
  assert.equal(await watchdogPeakFootprint(path.join(dir, "absent.jsonl")), null);
});

test("a runtime stop's reason names the lane's longest completed capture, and a budget below the lane default as a shortfall rather than a stall", () => {
  const peak = { peakBytes: 68_564_154_584, samples: 2354 };
  const ceiling = "94822600832-byte ceiling";
  // The campaign of 2026-09-07: `scail2_14b:q4:mlx` under `--probe-budget-minutes 90`, flat at
  // 68.56 GB for 2,354 samples. This is the sentence that was missing when it was read as a hang.
  const shortfall = runtimeBudgetExceededReason({ row: { key: "scail2_14b:q4:mlx", videoLane: true }, budgetMinutes: 90, budgetSeconds: 5400, peak, ceiling });
  assert.equal(
    shortfall,
    "runtime_budget_exceeded: the 90-minute probe budget (5400s) elapsed; peak physical footprint 68564154584 bytes "
      + "over 2354 sample(s) against the 94822600832-byte ceiling. Nothing was measured, so no bound was recorded and "
      + "this cell stays runnable. The longest completed video capture on record ran 8709s, and this run was launched "
      + "with --probe-budget-minutes 90, below the video lane's 270-minute default: this stop is a budget shortfall, "
      + "not evidence of a stall. Re-run it at the default budget or larger before reading anything into the flat footprint.",
  );
  // At the lane default (or above it) the stop stands on its own: the witness is still named, the
  // shortfall sentence is not.
  for (const budgetMinutes of [270, 300]) {
    const reason = runtimeBudgetExceededReason({ row: { key: "scail2_14b:q4:mlx", videoLane: true }, budgetMinutes, budgetSeconds: budgetMinutes * 60, peak, ceiling });
    assert.match(reason, new RegExp(`^runtime_budget_exceeded: the ${budgetMinutes}-minute probe budget \\(${budgetMinutes * 60}s\\) elapsed; `), reason);
    assert.match(reason, /stays runnable\. The longest completed video capture on record ran 8709s\.$/, reason);
    assert.doesNotMatch(reason, /shortfall|not evidence of a stall/, reason);
  }
  // The image lane reads its own witness and its own default, and an unsampled peak is said so.
  const image = runtimeBudgetExceededReason({ row: { key: "sdxl:q4:mlx", videoLane: false }, budgetMinutes: 30, budgetSeconds: 1800, peak: null, ceiling: "guard's ceiling" });
  assert.match(image, /peak physical footprint not sampled against the guard's ceiling\./, image);
  assert.match(image, /The longest completed image capture on record ran 847s, and this run was launched with --probe-budget-minutes 30, below the image lane's 60-minute default/, image);
  const imageAtDefault = runtimeBudgetExceededReason({ row: { key: "sdxl:q4:mlx", videoLane: false }, budgetMinutes: 60, budgetSeconds: 3600, peak: null, ceiling: "guard's ceiling" });
  assert.match(imageAtDefault, /ran 847s\.$/, imageAtDefault);
  assert.doesNotMatch(imageAtDefault, /shortfall/, imageAtDefault);
});

test("a probe that reaches its wall-clock budget is a capture_failed, never a memory bound, and the cell stays runnable", { skip: process.platform !== "darwin" && "the footprint sampler is Darwin-only" }, async () => {
  const checkout = await stubCheckout();
  const tierRoot = await mkdtemp(path.join(tmpdir(), "catalog-tier-"));
  await writeFile(path.join(tierRoot, "w.safetensors"), "weights");
  process.env.GIT_AUTHOR_NAME = process.env.GIT_COMMITTER_NAME = "t";
  process.env.GIT_AUTHOR_EMAIL = process.env.GIT_COMMITTER_EMAIL = "t@t";
  // The wedge: the capture never finishes and its footprint never reaches the ceiling (the probe
  // reports a 128 GiB host, so the kill line is the incident's 94,822,600,832 bytes and a Node
  // process cannot approach it). Only the budget can end this run — 3 seconds of it.
  process.env.STUB_CAPTURE_HANGS = "1";
  try {
    const context = stubContext(checkout, {
      args: { ...stubContext(checkout).args, probeBudgetMinutes: 0.05 },
    });
    const stopped = await measureAnchor({
      key: "z_image_turbo:q4:mlx", physical: false, tierRoot, env: { SCENEWORKS_Z_IMAGE_ROOT: tierRoot },
      // Fully bound, so nothing but the runtime arm itself can be what stops a bound being written:
      // the "no artifact binding" refusal is not the guard under test here.
      artifact: { repository: "SceneWorks/z-image-turbo-mlx", resolvedRevision: REVISION, variant: "q4" },
    }, context);
    // A runtime stop measured NOTHING. It is a failed capture, not an exceedance: recording the
    // footprint the probe happened to be sitting at would state a lower bound the run never
    // established, and production would refuse hosts on it.
    assert.equal(stopped.status, "capture_failed", stopped.reason);
    assert.match(stopped.reason, /^runtime_budget_exceeded: the 0\.05-minute probe budget \(3s\) elapsed;/);
    // The line an operator reads has to carry both figures, or there is nothing to judge with.
    assert.match(stopped.reason, /peak physical footprint \d+ bytes over \d+ sample\(s\) against the 94822600832-byte ceiling/);
    assert.match(stopped.reason, /no bound was recorded and this cell stays runnable/);
    // sc-22738 (2026-09-08): 0.05 minutes is below the image lane's default, and the reason says
    // so — a stop under a shortened budget must not read as a wedged engine.
    assert.match(stopped.reason, /The longest completed image capture on record ran 847s, and this run was launched with --probe-budget-minutes 0\.05, below the image lane's 60-minute default: this stop is a budget shortfall, not evidence of a stall\./);
    assert.equal(context.state.commits.length, 0, "nothing was committed");
    const { stdout: status } = await checkout.git("status", "--porcelain");
    assert.equal(status, "", "the tree is clean for the next anchor");
    // NO BOUND, by every route one could have arrived through.
    await assert.rejects(
      stat(path.join(checkout.root, "docs/calibration/sc-stub/z-image-turbo-q4-mlx-exceeded-evidence.json")),
      "no bound bundle was written",
    );
    const store = JSON.parse(await readFile(path.join(checkout.root, ANCHOR_STORE_PATH), "utf8"));
    assert.deepEqual(store.exceededBounds ?? [], [], "the anchor store carries no bound for this cell");
    assert.deepEqual(store.anchors ?? [], [], "and no measurement either");
    // ...which is exactly why the cell is still schedulable: `--list` reads the store, and the
    // store never heard about this attempt.
    assert.equal((await readExceededBounds(checkout.root)).size, 0);
    // The guard's own evidence is KEPT in the work dir for the operator, on the SAME stop path a
    // footprint stop takes: samples, the hard stop, the escalation and the terminated group.
    const eventFile = path.join(checkout.workDir, "logs", "z-image-turbo-q4-mlx-watchdog.jsonl");
    const events = (await readFile(eventFile, "utf8")).trim().split("\n").map(JSON.parse);
    const hardStop = events.find((event) => event.event === "hard_stop");
    assert.ok(hardStop, "the guard recorded the stop");
    assert.equal(hardStop.reason, "runtime_at_or_above_3.0s");
    assert.equal(runtimeBudgetStopSeconds(await watchdogHardStop(eventFile)), 3, "the runner reads the guard's own spelling");
    assert.ok(events.some((event) => event.event === "sample"), "the samples behind the reported peak are kept");
    assert.ok(events.some((event) => event.event === "terminated"), "the group was terminated, not left running");
    assert.ok((await stat(stopped.log)).size > 0, "the per-anchor log is kept too");
    const log = await readFile(stopped.log, "utf8");
    // The plumbing, end to end: minutes on the runner, seconds on the guard.
    assert.match(log, /memory-calibration-watchdog\.py .*--max-runtime-seconds 3 /);
  } finally {
    delete process.env.STUB_CAPTURE_HANGS;
  }
});

// ---------------------------------------------------------------------------------------------
// sc-22738: a PROCESS-SCOPED Metal refusal, the second way a run ends having measured a bound.
// ---------------------------------------------------------------------------------------------

test("a Metal submissions-ignored refusal is COMMITTED as a measured bound at the wired limit", { skip: process.platform !== "darwin" && "the footprint sampler is Darwin-only" }, async () => {
  const checkout = await stubCheckout();
  const tierRoot = await mkdtemp(path.join(tmpdir(), "catalog-tier-"));
  await writeFile(path.join(tierRoot, "w.safetensors"), "weights");
  process.env.GIT_AUTHOR_NAME = process.env.GIT_COMMITTER_NAME = "t";
  process.env.GIT_AUTHOR_EMAIL = process.env.GIT_COMMITTER_EMAIL = "t@t";
  // NO hard stop: the guard's ceiling stays where the incident puts it, so the only thing that ends
  // this capture is the adapter's own refusal — exactly the flux2_dev:bf16:mlx shape.
  process.env.STUB_CAPTURE_METAL_REFUSAL = "1";
  try {
    const row = {
      key: "z_image_turbo:q4:mlx", physical: false, tierRoot,
      env: { SCENEWORKS_Z_IMAGE_ROOT: tierRoot },
      artifact: { repository: "SceneWorks/z-image-turbo-mlx", resolvedRevision: REVISION, variant: "q4" },
    };
    const context = stubContext(checkout);
    const refused = await measureAnchor(row, context);
    // THE defect: a refusal the guard never saw used to be an ordinary `capture_failed`, so the
    // store kept no trace and production kept admitting the request Metal had just refused.
    assert.equal(refused.status, "committed_exceeded", refused.reason);
    assert.equal(context.state.commits.length, 1, "the refusal is one commit, like an anchor");
    assert.equal(context.state.halt, null, "one refusal is process-scoped and does not stop the walk");
    assert.equal((await checkout.git("status", "--porcelain")).stdout, "");
    const { stdout: shown } = await checkout.git("show", "--stat", "--format=%s%n%b", "HEAD");
    assert.match(shown, /chore\(sc-stub\): record the z_image_turbo:q4:mlx Metal refusal as a measured bound/);
    assert.ok(shown.includes(METAL_REFUSAL), "the commit body names the refusal, not a watchdog stop");
    const bundle = JSON.parse(await readFile(
      path.join(checkout.root, "docs/calibration/sc-stub/z-image-turbo-q4-mlx-exceeded-evidence.json"), "utf8",
    ));
    const [bound] = bundle.exceededBounds;
    // The figures the runner's witnesses produced: the sampler's PEAK, and the wired limit its own
    // probe read (87,044,670,532 on the stub, this host's figure).
    assert.match(bound.reason, /^metal_submissions_ignored:observed_\d+:wired_limit_87044670532$/);
    assert.ok(bound.observedFootprintBytes > 0);
    assert.equal(
      bound.ceilingBytes, bound.observedFootprintBytes,
      "no guard fired, so the highest line the run is witnessed to have crossed is its own peak",
    );
    assert.equal(Number(/observed_(\d+)/.exec(bound.reason)[1]), bound.observedFootprintBytes);
    // The refusal's witness is written where the harness can re-read and hash it.
    const stderr = await readFile(path.join(checkout.workDir, "logs", "z-image-turbo-q4-mlx-provider-stderr.txt"), "utf8");
    assert.match(stderr, /00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored/);
  } finally {
    delete process.env.STUB_CAPTURE_METAL_REFUSAL;
  }
});

test("artifactUnsupported matches the production loader's refusal line and NOTHING else", () => {
  const wan = [
    "GPU-view coherence retries during this render: mlx_gen=0 mlx_llm=0 (sc-22414)",
    "load real wan2_2_i2v_14b q4 provider: unsupported: /hub/q4/high_noise_model.safetensors packed head.head lacks scales",
  ].join("\n");
  assert.deepEqual(artifactUnsupported(wan), {
    provider: "wan2_2_i2v_14b",
    tier: "q4",
    reason: "/hub/q4/high_noise_model.safetensors packed head.head lacks scales",
  });
  // The TI2V-5B shape, all three tiers, and the reason is carried whole.
  for (const tier of ["bf16", "q4", "q8"]) {
    const ti2v = `load real wan2_2_ti2v_5b ${tier} provider: unsupported: wan2_2_ti2v_5b: config.json is not the complete canonical dense Wan2.2 TI2V-5B configuration`;
    assert.deepEqual(artifactUnsupported(ti2v), {
      provider: "wan2_2_ti2v_5b",
      tier,
      reason: "wan2_2_ti2v_5b: config.json is not the complete canonical dense Wan2.2 TI2V-5B configuration",
    });
  }
  // NOT this: only `Unsupported` is the engine declining a surface it understands. Anything else
  // that went wrong during a load stays an ordinary failure.
  assert.equal(artifactUnsupported("load real wan2_2_ti2v_5b bf16 provider: No such file or directory (os error 2)"), null);
  assert.equal(artifactUnsupported("Error: memory-strategy provider adapter: mint the wan2_2_i2v_14b request receipt: unsupported: x"), null);
  assert.equal(artifactUnsupported("[METAL] Command buffer execution failed: Ignored (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)"), null);
  assert.equal(artifactUnsupported(""), null);
  assert.equal(artifactUnsupported(undefined), null);
});

test("the runbook's outcome table carries the pinned-artifact refusal and its two open rehosts", async () => {
  const runbook = await readFile(path.join(ROOT, "docs/calibration-runbook.md"), "utf8");
  assert.match(runbook, /\| `artifact_unsupported` \|/, "the outcome table names the status");
  assert.ok(runbook.includes(ARTIFACT_UNSUPPORTED), "the runbook quotes the constant, so the two cannot drift");
  // The note for the artifact owner: repo, revision, and the file/key each refusal names.
  for (const cited of [
    "SceneWorks/wan2.2-i2v-a14b-mlx",
    "c6c78617",
    "head.head",
    "SceneWorks/wan2.2-ti2v-5b-mlx",
    "bb1b0552",
    "max_area",
  ]) {
    assert.ok(runbook.includes(cited), `the open-rehost note must name ${cited}`);
  }
});

test("a pinned-artifact loader refusal is its OWN outcome: named, uncommitted, and not a walk failure", { skip: process.platform !== "darwin" && "the footprint sampler is Darwin-only" }, async () => {
  const checkout = await stubCheckout();
  const tierRoot = await mkdtemp(path.join(tmpdir(), "catalog-tier-"));
  await writeFile(path.join(tierRoot, "w.safetensors"), "weights");
  process.env.STUB_CAPTURE_ARTIFACT_UNSUPPORTED = "1";
  try {
    const context = stubContext(checkout);
    const refused = await measureAnchor({
      key: "z_image_turbo:q4:mlx", physical: false, tierRoot,
      env: { SCENEWORKS_Z_IMAGE_ROOT: tierRoot },
      // An artifact binding is present, which is precisely what would have made a Metal refusal a
      // BOUND: this outcome must be chosen anyway, because nothing was measured.
      artifact: { repository: "SceneWorks/z-image-turbo-mlx", resolvedRevision: REVISION, variant: "q4" },
    }, context);
    assert.equal(refused.status, "artifact_unsupported", refused.reason);
    // The row carries the ENGINE's own sentence, not a generic bucket and not the sc-22414 line
    // that `failureReason` would have quoted before the classification existed.
    assert.equal(
      refused.reason,
      "wan2_2_i2v_14b q4: /hub/q4/high_noise_model.safetensors packed head.head lacks scales",
    );
    assert.equal(context.state.commits.length, 0, "a load that never completed records nothing");
    assert.equal(context.state.halt, null);
    assert.equal(context.state.metalRefusedLast, null, "this is not a Metal refusal and must not arm the wedged-host counter");
    assert.equal((await checkout.git("status", "--porcelain")).stdout, "", "the store and matrix are untouched");
  } finally {
    delete process.env.STUB_CAPTURE_ARTIFACT_UNSUPPORTED;
  }
});

test("a capture that fails WITHOUT the refusal signature is still a capture_failed, never a bound", { skip: process.platform !== "darwin" && "the footprint sampler is Darwin-only" }, async () => {
  const checkout = await stubCheckout();
  process.env.STUB_CAPTURE_FAILS = "1";
  try {
    const context = stubContext(checkout);
    const failed = await measureAnchor({
      key: "z_image_turbo:q4:mlx", physical: false, env: {},
      artifact: { repository: "SceneWorks/z-image-turbo-mlx", resolvedRevision: REVISION, variant: "q4" },
    }, context);
    // The discrimination this story turns on, stated as its own case: an artifact-bound row that
    // fails is NOT thereby a bound. Only the signature makes it one.
    assert.equal(failed.status, "capture_failed");
    assert.equal(failed.reason, "Error: stub capture refused");
    assert.equal(context.state.commits.length, 0);
    assert.equal(context.state.metalRefusedLast, null, "a non-refusal clears the consecutive-refusal flag");
    assert.equal((await checkout.git("status", "--porcelain")).stdout, "");
  } finally {
    delete process.env.STUB_CAPTURE_FAILS;
  }
});

test("two consecutive Metal refusals are a WEDGED HOST: the walk halts naming the reboot instead of recording a second bound", { skip: process.platform !== "darwin" && "the footprint sampler is Darwin-only" }, async () => {
  const checkout = await stubCheckout();
  const tierRoot = await mkdtemp(path.join(tmpdir(), "catalog-tier-"));
  await writeFile(path.join(tierRoot, "w.safetensors"), "weights");
  process.env.GIT_AUTHOR_NAME = process.env.GIT_COMMITTER_NAME = "t";
  process.env.GIT_AUTHOR_EMAIL = process.env.GIT_COMMITTER_EMAIL = "t@t";
  process.env.STUB_CAPTURE_METAL_REFUSAL = "1";
  try {
    const context = stubContext(checkout);
    const artifact = { repository: "SceneWorks/z-image-turbo-mlx", resolvedRevision: REVISION, variant: "q4" };
    const first = await measureAnchor(
      { key: "z_image_turbo:q4:mlx", physical: false, tierRoot, env: {}, artifact }, context,
    );
    assert.equal(first.status, "committed_exceeded", first.reason);
    assert.equal(context.state.metalRefusedLast, "z_image_turbo:q4:mlx");
    const second = await measureAnchor(
      { key: "z_image_turbo:bf16:mlx", physical: false, tierRoot, env: {}, artifact }, context,
    );
    // `SubmissionsIgnored` has two scopes. The first refusal was the process's; a second in a row
    // is the GPU's, and every remaining anchor would otherwise "measure" a bound about the driver.
    assert.equal(second.status, "capture_failed", second.reason);
    assert.match(second.reason, /halting the walk/);
    assert.equal(context.state.commits.length, 1, "no second bound was recorded");
    assert.ok(context.state.halt, "the walk is halted");
    assert.match(context.state.halt, /REBOOT/i, "the halt names the reboot the host needs");
    assert.match(context.state.halt, /z_image_turbo:q4:mlx and then z_image_turbo:bf16:mlx/);
    assert.equal((await checkout.git("status", "--porcelain")).stdout, "");
  } finally {
    delete process.env.STUB_CAPTURE_METAL_REFUSAL;
  }
});

test("a refusal on a no-commit run surfaces as `exceeded` naming the Metal refusal, tree left clean", { skip: process.platform !== "darwin" && "the footprint sampler is Darwin-only" }, async () => {
  const checkout = await stubCheckout();
  const context = stubContext(checkout, { args: { ...stubContext(checkout).args, commit: false } });
  process.env.STUB_CAPTURE_METAL_REFUSAL = "1";
  try {
    const refused = await measureAnchor({
      key: "z_image_turbo:q4:mlx", physical: false, env: {},
      artifact: { repository: "SceneWorks/z-image-turbo-mlx", resolvedRevision: REVISION, variant: "q4" },
    }, context);
    assert.equal(refused.status, "exceeded");
    assert.equal(refused.reason, METAL_REFUSAL);
    assert.equal((await checkout.git("status", "--porcelain")).stdout, "");
  } finally {
    delete process.env.STUB_CAPTURE_METAL_REFUSAL;
  }
});

// ---------------------------------------------------------------------------------------------
// sc-22738: `--download-missing` fetches the pinned snapshots a weights_missing anchor needs
// ---------------------------------------------------------------------------------------------
//
// The campaign runs on boxes that do not hold the whole catalog, and downloading a missing snapshot
// on the Windows box beats copying it off the Mac's SSD. These tests drive the fetch through a STUB
// `hf` on PATH that writes the hub layout the real CLI writes, so they prove the wiring — which
// anchors are fetched, where they land, what argv the CLI is handed, and what a failure does — with
// no network and no weights.

// The stub is a shebang script, which the runner resolves off PATH only where a shebang is honoured;
// spawn() on Windows does no PATHEXT resolution, so the three PATH-driven tests below are macOS/Linux.
// Nothing lane-specific is being tested in them — the wiring is the same on either box — and the
// pure functions (`anchorDownloadTargets`, `hfDownloadArgv`, `tierDownloadRows`) run everywhere.
const skipWithoutShebang = process.platform === "win32" && "the stub CLI is a shebang script on PATH";

/** A stub `hf` on PATH: appends its argv to $STUB_HF_LOG and creates the snapshot it was asked for. */
async function stubHuggingFaceCli({ fail = false } = {}) {
  const dir = await mkdtemp(path.join(tmpdir(), "catalog-hf-cli-"));
  const log = path.join(dir, "invocations.txt");
  const body = fail
    ? 'process.stderr.write("stub refusal: 401 Client Error\\n");\nprocess.exit(1);'
    : [
      'const repo = argv[1];',
      'const revision = argv[argv.indexOf("--revision") + 1];',
      'const cacheDir = argv[argv.indexOf("--cache-dir") + 1];',
      'const includes = argv.flatMap((a, i) => (a === "--include" ? [argv[i + 1]] : []));',
      'const snapshot = path.join(cacheDir, "models--" + repo.split("/").join("--"), "snapshots", revision);',
      // The real CLI materialises whatever the globs select; the stub materialises each glob's own
      // directory prefix, which is what `classifyAnchor` probes for.
      'for (const glob of includes.length > 0 ? includes : ["."]) {',
      '  const directory = glob.includes("/") ? glob.slice(0, glob.lastIndexOf("/")) : ".";',
      '  fs.mkdirSync(path.join(snapshot, directory), { recursive: true });',
      '}',
      'fs.writeFileSync(path.join(snapshot, "config.json"), "{}");',
      // `refs/` is written exactly as the hub CLI writes it: not at all for a 40-hex revision.
      'fs.mkdirSync(path.join(cacheDir, "models--" + repo.split("/").join("--"), "blobs"), { recursive: true });',
    ].join("\n");
  const script = [
    "#!/usr/bin/env node",
    'const fs = require("node:fs");',
    'const path = require("node:path");',
    "const argv = process.argv.slice(2);",
    'fs.appendFileSync(process.env.STUB_HF_LOG, JSON.stringify(argv) + "\\n");',
    body,
    "",
  ].join("\n");
  const binary = path.join(dir, HF_CLI_CANDIDATES[0]);
  await writeFile(binary, script, { mode: 0o755 });
  return {
    dir,
    log,
    invocations: async () => (await readFile(log, "utf8").catch(() => ""))
      .split("\n").filter(Boolean).map((line) => JSON.parse(line)),
  };
}

/** Run `body` with the stub CLI first on PATH and $STUB_HF_LOG pointed at its log. */
async function withStubCli(stub, body) {
  const originalPath = process.env.PATH;
  const originalLog = process.env.STUB_HF_LOG;
  process.env.PATH = `${stub.dir}${path.delimiter}${originalPath}`;
  process.env.STUB_HF_LOG = stub.log;
  try {
    return await body();
  } finally {
    process.env.PATH = originalPath;
    if (originalLog === undefined) delete process.env.STUB_HF_LOG;
    else process.env.STUB_HF_LOG = originalLog;
  }
}

/**
 * A checkout-shaped directory `planRun` can read: one plan anchor per key, a manifest carrying the
 * families' own repositories, and the two closure declarations the classifier consults.
 */
async function fakePlanRoot(anchors, models) {
  const root = await mkdtemp(path.join(tmpdir(), "catalog-root-"));
  await mkdir(path.join(root, "config", "manifests"), { recursive: true });
  await writeFile(path.join(root, "config", "memory-calibration-plan.json"), JSON.stringify({ anchors }));
  await writeFile(path.join(root, "config", "manifests", "builtin.models.jsonc"), JSON.stringify({ models }));
  const lanes = Object.keys(anchors).map((key) => {
    const parts = anchorParts(key);
    return `${parts.modelId}:${parts.backend}`;
  });
  const providers = Object.entries(anchors).map(([key, planned]) => `${anchorParts(key).backend}:${planned.provider}`);
  await writeFile(
    path.join(root, "config", "anchor-loader-closures.json"),
    JSON.stringify({ models: Object.fromEntries(lanes.map((lane) => [lane, { digest: "d".repeat(64) }])) }),
  );
  await writeFile(
    path.join(root, "config", "inference-provider-closures.json"),
    JSON.stringify({ providers: Object.fromEntries(providers.map((provider) => [provider, {}])) }),
  );
  return root;
}

const DOWNLOAD_REVISION = "abcdef0123456789abcdef0123456789abcdef01";

function downloadFixture() {
  return {
    anchors: {
      "qwen_image:q4:mlx": { provider: "qwen_image" },
      "z_image_turbo:q4:mlx": { provider: "z_image_turbo" },
    },
    models: [
      {
        id: "qwen_image",
        downloads: [{
          repo: PROVIDER_FAMILIES.qwen_image.repo, revision: DOWNLOAD_REVISION, variant: "q4",
          files: ["q4/*"], estimatedSizeBytes: 4321,
        }],
      },
      {
        id: "z_image_turbo",
        downloads: [{
          repo: PROVIDER_FAMILIES.z_image_turbo.repo, revision: DOWNLOAD_REVISION, variant: "q4",
          files: ["q4/*"],
        }],
      },
    ],
  };
}

test("a download target names the family's repository, the PINNED revision and the manifest's globs", () => {
  const { models } = downloadFixture();
  const row = { key: "qwen_image:q4:mlx", modelId: "qwen_image", tier: "q4", backend: "mlx", provider: "qwen_image" };
  const { targets, unfetchable } = anchorDownloadTargets(row, models, { platform: "macos" });
  assert.deepEqual(unfetchable, []);
  assert.equal(targets.length, 1);
  assert.equal(targets[0].repo, PROVIDER_FAMILIES.qwen_image.repo);
  assert.equal(targets[0].revision, DOWNLOAD_REVISION);
  assert.deepEqual(targets[0].include, ["q4/*"]);
  assert.equal(targets[0].estimatedBytes, 4321);

  // THE MUTATION TARGET. `--revision` is the pinned revision, always: a fetch that resolved `main`
  // would land whatever the branch points at rather than the artifact the anchor prices — and
  // pinning a manifest download removes `refs/main` from the mirror in the first place.
  const argv = hfDownloadArgv(targets[0], "/hub");
  assert.deepEqual(argv, [
    "download", PROVIDER_FAMILIES.qwen_image.repo, "--revision", DOWNLOAD_REVISION,
    "--include", "q4/*", "--cache-dir", "/hub",
  ]);
  assert.equal(argv[argv.indexOf("--revision") + 1], DOWNLOAD_REVISION, "the pinned revision is passed, never main");
  assert.ok(!argv.includes("main"));
});

test("no anchor can be fetched off a branch: every target the shipped plan produces carries a 40-hex revision", async () => {
  const models = await readManifestModels();
  const plan = await readPlan();
  let unpinned = 0;
  for (const [key, planned] of Object.entries(plan.anchors)) {
    const parts = anchorParts(key);
    const { targets, unfetchable } = anchorDownloadTargets({ key, ...parts, provider: planned.provider }, models);
    for (const target of targets) {
      assert.match(target.revision, /^[0-9a-f]{40}$/, `${key} would fetch ${target.repo}@${target.revision}`);
    }
    if (unfetchable.some((note) => /shipped without a pinned revision/.test(note))) unpinned += 1;
  }
  // The upstream Wan 2.2 / SVD Diffusers checkpoints are the shipped unpinned case; they are
  // REPORTED rather than resolved off a branch, which is the whole reason the refusal exists.
  assert.ok(unpinned > 0, "the unpinned-revision refusal is reachable on the shipped plan");
});

test("--download-missing fetches ONLY the weights_missing anchors, into the first hub root, and re-classifies them", { skip: skipWithoutShebang }, async () => {
  const { anchors, models } = downloadFixture();
  const root = await fakePlanRoot(anchors, models);
  const first = await fakeHub([[PROVIDER_FAMILIES.z_image_turbo.repo, DOWNLOAD_REVISION, "q4"]]);
  const second = await mkdtemp(path.join(tmpdir(), "catalog-hub-second-"));
  const stub = await stubHuggingFaceCli();
  const args = {
    ...parseArgs(["--backend", "mlx", "--list", "--download-missing"]),
    campaign: "sc-dl-test", hfCache: [first, second],
  };

  const before = await planRun({ ...args, downloadMissing: false }, root);
  assert.equal(before.rows.find((row) => row.key === "qwen_image:q4:mlx").status, "weights_missing");
  assert.equal(before.rows.find((row) => row.key === "z_image_turbo:q4:mlx").status, "runnable");

  const { rows } = await withStubCli(stub, () => planRun(args, root));
  const fetched = rows.find((row) => row.key === "qwen_image:q4:mlx");
  assert.equal(fetched.status, "runnable", fetched.reason);
  // It landed in the FIRST --hf-cache root, in hub layout, and the tier root is what the row binds.
  assert.equal(fetched.tierRoot, snapshotPath(first, PROVIDER_FAMILIES.qwen_image.repo, DOWNLOAD_REVISION, "q4"));
  assert.ok((await stat(snapshotPath(first, PROVIDER_FAMILIES.qwen_image.repo, DOWNLOAD_REVISION, "q4"))).isDirectory());
  assert.deepEqual(await readdir(second), [], "nothing is written into a later --hf-cache root");

  // The anchor that was ALREADY present was never fetched: exactly one invocation, for the other repo.
  const invocations = await stub.invocations();
  assert.equal(invocations.length, 1, JSON.stringify(invocations));
  assert.equal(invocations[0][1], PROVIDER_FAMILIES.qwen_image.repo);
  assert.equal(invocations[0][invocations[0].indexOf("--revision") + 1], DOWNLOAD_REVISION);
  assert.equal(invocations[0][invocations[0].indexOf("--cache-dir") + 1], first);
  assert.equal(rows.find((row) => row.key === "z_image_turbo:q4:mlx").status, "runnable");
});

test("a failed fetch leaves the anchor weights_missing naming the reason, and never stops the walk", { skip: skipWithoutShebang }, async () => {
  const { anchors, models } = downloadFixture();
  const root = await fakePlanRoot(anchors, models);
  const hub = await fakeHub([[PROVIDER_FAMILIES.z_image_turbo.repo, DOWNLOAD_REVISION, "q4"]]);
  const stub = await stubHuggingFaceCli({ fail: true });
  const args = {
    ...parseArgs(["--backend", "mlx", "--list", "--download-missing"]),
    campaign: "sc-dl-test", hfCache: [hub],
  };

  const { rows } = await withStubCli(stub, () => planRun(args, root));
  const refused = rows.find((row) => row.key === "qwen_image:q4:mlx");
  assert.equal(refused.status, "weights_missing");
  assert.match(refused.reason, /download failed:/);
  assert.match(refused.reason, new RegExp(`${PROVIDER_FAMILIES.qwen_image.repo}@${DOWNLOAD_REVISION.slice(0, 8)}`));
  // The walk keeps going: the other anchor is classified exactly as it was.
  assert.equal(rows.find((row) => row.key === "z_image_turbo:q4:mlx").status, "runnable");
});

test("--dry-run --download-missing prints what it would fetch and fetches nothing", { skip: skipWithoutShebang }, async () => {
  const { anchors, models } = downloadFixture();
  const root = await fakePlanRoot(anchors, models);
  const hub = await mkdtemp(path.join(tmpdir(), "catalog-hub-dry-"));
  const stub = await stubHuggingFaceCli();
  const lines = [];
  const row = { key: "qwen_image:q4:mlx", modelId: "qwen_image", tier: "q4", backend: "mlx", provider: "qwen_image" };
  const outcome = await withStubCli(stub, () => fetchAnchorSnapshots(row, models, {
    cacheRoot: hub, dryRun: true, platform: "macos", log: (line) => lines.push(line),
  }));
  assert.equal(outcome.failed, null);
  assert.equal(outcome.fetched.length, 1);
  assert.match(lines.join("\n"), /would fetch 1 glob\(s\): q4\/\*.*~4321 bytes/);
  assert.deepEqual(await stub.invocations(), [], "a dry run runs no CLI at all");
  assert.deepEqual(await readdir(hub), [], "a dry run writes nothing into the hub root");

  // And through planRun: the dry run leaves the classification alone.
  const args = {
    ...parseArgs(["--backend", "mlx", "--list", "--download-missing"]),
    dryRun: true, campaign: "sc-dl-test", hfCache: [hub],
  };
  const { rows } = await withStubCli(stub, () => planRun(args, root));
  assert.equal(rows.find((entry) => entry.key === "qwen_image:q4:mlx").status, "weights_missing");
  assert.deepEqual(await stub.invocations(), []);
});

test("download rows are narrowed to this host's platform, and fall back rather than resolving to no globs", async () => {
  assert.equal(manifestPlatform("darwin"), "macos");
  assert.equal(manifestPlatform("win32"), "windows");
  assert.equal(manifestPlatform("linux"), "linux");
  const models = await readManifestModels();
  // MiniMax-H3's dense upstream ships the whole DiT on windows/linux and only the text encoder and
  // the VAEs on macOS; the narrowing is what keeps an MLX box from fetching the windows leg.
  const macos = tierDownloadRows(models, "minimax_h3", "MiniMaxAI/MiniMax-H3", "bf16", "macos");
  const windows = tierDownloadRows(models, "minimax_h3", "MiniMaxAI/MiniMax-H3", "bf16", "windows");
  assert.ok(macos.length > 0 && windows.length > 0);
  assert.ok(macos.every((download) => !download.platforms || download.platforms.includes("macos")));
  assert.ok(windows.some((download) => (download.files ?? []).some((glob) => glob === "transformer/*")));
  assert.ok(!macos.some((download) => (download.files ?? []).some((glob) => glob === "transformer/*")));
  // A repository whose rows are all scoped elsewhere falls back to the UNFILTERED set rather than
  // to "no globs", which downstream would read as "fetch the whole repository".
  const unfiltered = (models.find((entry) => entry.id === "minimax_h3").downloads ?? []).filter(
    (download) => download.repo === "MiniMaxAI/MiniMax-H3" && (download.variant === undefined || download.variant === "bf16"),
  );
  assert.deepEqual(tierDownloadRows(models, "minimax_h3", "MiniMaxAI/MiniMax-H3", "bf16", "plan9"), unfiltered);
  assert.ok(unfiltered.length > macos.length, "the fallback is wider than one platform's rows");
});
