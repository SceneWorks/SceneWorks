#[cfg(target_os = "macos")]
use super::ltx::extract_clip_frames;
#[allow(unused_imports)]
use super::prelude::*;
#[cfg(target_os = "macos")]
use super::wan::{generate_video_using, local_mlx_dir, VideoGenInput};

// ---------------------------------------------------------------------------
// Real MLX Krea Realtime 14B generation (macOS, via mlx-gen-krea-realtime, epic 8431 / sc-8443 S10):
// an autoregressive, self-forcing, real-time video denoiser whose SHIPPED checkpoint is Wan 2.1 T2V
// 14B weight-for-weight (the S1 audit fact the engine crate is built on). What makes it distinct is
// its inference REGIME — a short per-frame-block few-step Self-Forcing schedule + a rolling causal KV
// cache — not its weights, so the DiT / z16 VAE / UMT5 text encoder / RoPE / schedulers are all reused
// from stock Wan inside the engine.
//
// Surface (descriptor `pipeline::descriptor`, sc-8440 S7 + sc-15203 S19): CFG-OFF video (no
// negative-prompt / guidance / true-cfg axis — the AR few-step denoise runs a single batch-1 forward
// per step), sampler `self_forcing`, `supported_quants = [Q4, Q8]` (bf16 = their absence),
// `supports_lora`/`supports_lokr = true` (sc-15015 S14 wired the dense forward-time residual path;
// this module's S15 passthrough is what fills it), `mac_only = true`. It advertises two conditioning
// shapes the worker maps its own reference inputs onto here:
//   * i2v — a `Reference` still VAE-encoded to WARM the AR KV cache from clean context. The engine's
//     `run` reads only the reference IMAGE (`Conditioning::Reference { image, .. }`), never its
//     `strength` — the cache-warm has no strength analog — so **i2v `strength` is a NO-OP**, faithful
//     to the sibling `bernini`/reference paths which also emit `strength: None`. Documented + pinned
//     below so a future edit can't silently start honoring it.
//   * v2v — a `VideoClip` source that drives a strength-CONTROLLED AR init (`FewStepSchedule::for_
//     strength`): a lower strength preserves more of the source. So **v2v DOES honor `strength`**
//     (`videoConditioningStrength`, default 1.0 — the same knob + default the Wan/LTX v2v paths use).
//
// This is the non-gated worker DISPATCH wiring (routing + VideoRequest→GenerationRequest mapping +
// heartbeat-funnel routing). The real-weight watchable-clip e2e needs the real ~28 GB DiT and is the
// GATED S13 validation; the MLX rehost of the turnkey snapshot is the gated S2 remainder (sc-8435).
// LoRA passthrough landed in S15 ([`resolve_krea_realtime_adapters`]). Mac-only: the off-Mac (candle) lane
// has NO Krea Realtime engine, so a non-mac job fails loud in `run_video_generate_job` (mirroring the
// `wan_2_2_vace_fun_14b` mac-only guard) rather than silently routing to a different backend / the stub.
// ---------------------------------------------------------------------------

/// The SceneWorks/model catalog id for Krea Realtime 14B — identical to the engine registry id
/// (`mlx_gen_krea_realtime::MODEL_ID`), so the SceneWorks id maps to the engine 1:1 (no `_distilled`-
/// style split). Named once here so the route, the raw-settings, and the mac-only guard agree.
#[cfg(target_os = "macos")]
pub(super) const KREA_REALTIME_MODEL_ID: &str = "krea_realtime_14b";

/// Adapter id recorded on a real MLX Krea Realtime asset (mirrors the `mlx_*` convention).
#[cfg(target_os = "macos")]
pub(super) const KREA_REALTIME_ADAPTER: &str = "mlx_krea_realtime";

/// SceneWorks Krea Realtime model id → mlx-gen registry id, or `None` if `model` is not Krea Realtime.
/// The id IS the engine id (`KREA_REALTIME_MODEL_ID`); no earlier predicate in `resolve_video_route`
/// can match it (it is not a Wan/LTX/SVD/Bernini/SCAIL-2/Mochi id), so appending the krea arm leaves
/// every pre-existing route byte-identical.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_engine_id(model: &str) -> Option<&'static str> {
    (model == KREA_REALTIME_MODEL_ID).then_some(KREA_REALTIME_MODEL_ID)
}

/// Whether the linked Krea Realtime engine can serve this request now (a resolvable, LOADABLE tier).
/// `mac_only` is implicit — this fn is macOS-only, and a non-mac job never reaches the route (it is
/// rejected loudly in `run_video_generate_job`). Routed by weight availability like the other MLX
/// engines: an unresolved snapshot falls to `VideoRoute::Stub`, whose fail-loud gate
/// (`ensure_video_engine_weights`) then surfaces the resolver's precise error instead of a procedural
/// fake clip.
///
/// It asks the FULL [`resolve_krea_realtime_tier_dir_and_quant`], not just
/// [`resolve_krea_realtime_model_dir`] (sc-15258): every published Krea file lives under a `q4/`/`q8/`/
/// `bf16/` tier prefix, so the turnkey snapshot ROOT resolving is not evidence that anything is
/// loadable — a repo dir holding only `README.md` (or a torn tier) would otherwise route to the engine
/// and die inside the loader with a message about a missing `dit.safetensors`.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_available(request: &VideoRequest, settings: &Settings) -> bool {
    resolve_krea_realtime_tier_dir_and_quant(settings, request).is_ok()
}

/// The turnkey SceneWorks Krea Realtime MLX repo (sc-8435 S2b): the converted Krea DiT at three
/// precisions plus the stock Wan `t5_encoder`/`vae`/`tokenizer` companions, laid out as the epic-8506
/// quant matrix — self-contained `q4/` (default) + `q8/` + `bf16/` tier subdirs, NOT a flat root.
/// `resolve_krea_realtime_model_dir` only ever LOOKS UP an already-downloaded snapshot in the local
/// cache; the on-demand fetch of an explicitly-picked non-default tier is
/// [`ensure_krea_realtime_tier_present`].
#[cfg(target_os = "macos")]
pub(super) const KREA_REALTIME_REPO: &str = "SceneWorks/krea-realtime-14b-mlx";

/// Pinned revision for [`KREA_REALTIME_REPO`] (mirrors [`SCAIL2_REVISION`](super::scail2)). The repo is
/// a hard-coded const — no manifest/payload override reaches the on-demand `q8/` + `bf16/` fetches — so
/// pulling the mutable `main` branch would let an upstream re-push silently swap a checkpoint we load.
/// This is the S2b rehost commit, and it MUST equal the `revision` every `krea_realtime_14b` download
/// entry in `builtin.models.jsonc` pins, or the on-demand fetch would materialize a SECOND snapshot dir
/// beside the one the Model Manager installed
/// (`krea_realtime_worker_tier_table_matches_the_shipped_manifest` pins the equality). The native
/// downloader still verifies each file's own hash on download.
#[cfg(target_os = "macos")]
pub(super) const KREA_REALTIME_REVISION: &str = "e68e9a3d98187fdf6936838ffcf6df5aa48d6626";

/// The files that make a **packed** (`q4/`, `q8/`) Krea Realtime tier subdir COMPLETE: the single-file
/// packed DiT plus the stock dense Wan companions the engine opens by name
/// (`t2v::load_text_encoder`/`load_vae`) and the `config.json` carrying the quantization manifest.
#[cfg(target_os = "macos")]
const KREA_REALTIME_PACKED_TIER_FILES: &[&str] = &[
    "config.json",
    "dit.safetensors",
    "t5_encoder.safetensors",
    "tokenizer.json",
    "vae.safetensors",
];

/// The files that make the **`bf16/`** tier COMPLETE. Structurally different from the packed tiers:
/// its DiT is a SEVEN-shard `transformer/` directory one level deeper, because a single 28.58 GB bf16
/// blob is above the practical single-file limit the rehost emits. The engine's
/// `t2v::open_transformer_weights` probes both layouts (`dit.safetensors` OR a `transformer/` dir), so
/// this is a hosting shape rather than a second load path — but the *directory* handed to it must be
/// the tier dir either way. Every shard is listed individually so a torn download (6 of 7 shards) reads
/// INCOMPLETE and falls through, instead of resolving a tier that then dies inside the loader.
#[cfg(target_os = "macos")]
const KREA_REALTIME_BF16_TIER_FILES: &[&str] = &[
    "config.json",
    "t5_encoder.safetensors",
    "tokenizer.json",
    "vae.safetensors",
    "transformer/dit-00001-of-00007.safetensors",
    "transformer/dit-00002-of-00007.safetensors",
    "transformer/dit-00003-of-00007.safetensors",
    "transformer/dit-00004-of-00007.safetensors",
    "transformer/dit-00005-of-00007.safetensors",
    "transformer/dit-00006-of-00007.safetensors",
    "transformer/dit-00007-of-00007.safetensors",
];

/// The published Krea Realtime quant matrix: each tier subdir keyed to the files that make it COMPLETE
/// (mirrors [`SCAIL2_TIER_FILES`](super::scail2), but per-tier because the `bf16/` DiT is sharded a
/// level deeper while `q4/`/`q8/` ship a flat `dit.safetensors`). Kept in lockstep with the
/// `krea_realtime_14b` manifest `downloads[].files` — same repo, same revision, same paths — so the
/// worker's completeness predicate cannot drift from what the Model Manager actually installs.
#[cfg(target_os = "macos")]
pub(super) const KREA_REALTIME_TIER_FILES: &[(&str, &[&str])] = &[
    ("q4", KREA_REALTIME_PACKED_TIER_FILES),
    ("q8", KREA_REALTIME_PACKED_TIER_FILES),
    ("bf16", KREA_REALTIME_BF16_TIER_FILES),
];

/// The files that make `tier` COMPLETE, or `None` for a tier this repo does not publish. Deliberately
/// an `Option` rather than an empty-slice fallback: `[].iter().all(..)` is `true`, so an unknown tier
/// would otherwise read COMPLETE from an empty (or absent) directory.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_tier_files(tier: &str) -> Option<&'static [&'static str]> {
    KREA_REALTIME_TIER_FILES
        .iter()
        .find(|(name, _)| *name == tier)
        .map(|(_, files)| *files)
}

/// Map a Krea Realtime model id to its `(quant-matrix repo, pinned revision)` for the on-demand tier
/// fetch, or `None` for any other id (mirrors [`scail2_tier_repo`](super::scail2)). Keyed here so the
/// whole tier-resolve/fetch path routes on `krea_realtime_tier_repo(..).is_some()`.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_tier_repo(model: &str) -> Option<(&'static str, &'static str)> {
    (model == KREA_REALTIME_MODEL_ID).then_some((KREA_REALTIME_REPO, KREA_REALTIME_REVISION))
}

/// The Krea Realtime tier search order for a request — the preferred tier first, then the fallbacks,
/// derived from the SAME [`resolve_mlx_dense_quant`] decision that would pick a load-time quant
/// (sc-15258). `mlxQuantize <= 0` ⇒ `bf16`-first; `>= 5` ⇒ `q8`-first; no explicit pick OR an explicit
/// `q4` ⇒ **`q4`-first** — the dense-video Q4 default (sc-10750), which reaches the video lane here
/// because this order is what the resolver consults, not the image lane's `manifest.mlx.quantize`.
///
/// **`bf16` is the last fallback in EVERY order, never omitted** — the one deliberate divergence from
/// [`scail2_tier_order`](super::scail2), and the whole point of the story. SCAIL-2 can leave `bf16` out
/// of the default order because a repo with no matching tier still has a legacy FLAT root to fall back
/// on; the Krea rehost has no flat root at all. Dropping `bf16` from the default order would mean a
/// user who deliberately installed ONLY the `bf16/` tier gets "no tier found" and then either a hard
/// failure or — far worse — a root fallback that quantizes their dense weights to Q4 in memory. That is
/// the silent creative-tier switch the "quant tier is a CREATIVE choice, never switch it" convention
/// forbids. Listing it LAST keeps the other half of the guarantee: a default job never prefers the
/// ~40 GB dense tier while a leaner one is installed, and the tier is only ever *fetched* on an
/// explicit pick ([`ensure_krea_realtime_tier_present`]).
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_tier_order(request: &VideoRequest) -> &'static [&'static str] {
    match resolve_mlx_dense_quant(request) {
        None => &["bf16", "q8", "q4"],
        Some(Quant::Q8) => &["q8", "q4", "bf16"],
        // No explicit pick OR an explicit `q4` ⇒ q4-first (the sc-10750 dense-video default).
        _ => &["q4", "q8", "bf16"],
    }
}

/// The declared files of `tier` that are NOT present under `root/<tier>`, as repo-relative paths.
/// Empty ⇒ the tier is complete. An unknown tier reports ITSELF missing rather than reporting nothing,
/// so it can never read as complete through the empty-vec path.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_missing_tier_files(root: &Path, tier: &str) -> Vec<String> {
    let Some(files) = krea_realtime_tier_files(tier) else {
        return vec![format!("{tier}/ (not a published tier)")];
    };
    let dir = root.join(tier);
    files
        .iter()
        .filter(|file| !dir.join(file).is_file())
        .map(|file| format!("{tier}/{file}"))
        .collect()
}

/// Whether `root/<tier>` is a COMPLETE self-contained Krea Realtime tier snapshot (all of
/// [`krea_realtime_tier_files`]). A partially-downloaded tier — including a `bf16/` missing one of its
/// seven DiT shards — fails this, so [`krea_realtime_tier_subdir`] falls through to another complete
/// tier rather than half-loading. Unknown tiers are never complete.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_tier_is_complete(root: &Path, tier: &str) -> bool {
    krea_realtime_missing_tier_files(root, tier).is_empty()
}

/// Descend a resolved Krea Realtime repo `root` into the tier subdir this request should load: the
/// first COMPLETE tier in [`krea_realtime_tier_order`], as `(tier name, dir)`. `None` when no tier
/// subdir is complete — a locally converted FLAT snapshot (or nothing usable at all), which the caller
/// distinguishes.
///
/// The NAME rides along because the caller records it on the asset: the tier that actually loaded is
/// not always the one the request asked for (the order falls back), and the dir's last path component
/// is not a substitute — a flat snapshot has no tier component at all.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_tier_subdir(
    root: &Path,
    request: &VideoRequest,
) -> Option<(&'static str, PathBuf)> {
    krea_realtime_tier_order(request)
        .iter()
        .find(|tier| krea_realtime_tier_is_complete(root, tier))
        .map(|tier| (*tier, root.join(tier)))
}

/// Whether `dir` itself carries a Krea Realtime DiT — i.e. it is a FLAT snapshot rather than a
/// tier-matrix root. It asks the same two questions the engine's `t2v::open_transformer_weights` asks
/// (a single-file `dit.safetensors` OR a sharded `transformer/` dir), so the two agree about what
/// "loadable" means — but strictly TIGHTER, not identical: the engine probes the DiT with `exists()`
/// while this uses `is_file()`, so a *directory* named `dit.safetensors` would satisfy the engine's
/// check and not this one. That asymmetry runs in the safe direction (this can only refuse things the
/// engine would fail on anyway, never admit something it would reject) and matches the per-file
/// `is_file()` the tier-completeness check uses.
///
/// Only a locally converted snapshot (the `SCENEWORKS_MLX_KREA_REALTIME_DIR` override or the
/// app-managed convert output) is ever flat; the published repo root holds only tier subdirs, so it
/// correctly fails this and the resolver errors instead of handing the engine a dir with no weights in
/// it.
#[cfg(target_os = "macos")]
fn krea_realtime_dir_has_transformer(dir: &Path) -> bool {
    dir.join("dit.safetensors").is_file() || dir.join("transformer").is_dir()
}

/// Everything a Krea Realtime load must settle before the engine is asked for anything, resolved as
/// ONE unit by [`resolve_krea_realtime_tier_dir_and_quant`] (sc-15258). The three travel together on
/// purpose: splitting them back into independent lookups is exactly how the tier and the quantization
/// come apart, which is the silent creative-tier switch this story exists to prevent.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct KreaRealtimeLoad {
    /// The directory handed to the engine — an installed tier subdir, or a flat snapshot root.
    pub(super) dir: PathBuf,
    /// The load-time quant. Always `None` for a resolved tier (the tier on disk IS the quantization).
    pub(super) quant: Option<Quant>,
    /// The precision the DiT will actually run at (`q4` / `q8` / `bf16`) — recorded on the asset.
    pub(super) tier: &'static str,
}

/// Resolve the Krea Realtime model dir, load-time quant and tier label for a generation — the ONE
/// decision that picks them together (sc-15258), which is what makes the creative-tier choice
/// unswitchable:
///
/// * A COMPLETE tier subdir wins and loads with **`quant = None`**. The tier on disk IS the
///   quantization: `q4/`/`q8/` are pre-packed (the engine recovers `bits`/`group_size` from the packed
///   shapes and cross-checks `config.json`), and `bf16/` is dense. `mlxQuantize` selects WHICH tier,
///   never a load-time requant — so a deliberately-installed `bf16/` is served dense even under the Q4
///   default, and a packed tier is never asked to re-quantize (which the engine would reject loudly
///   anyway, `load::resolve_load_time_quant`).
/// * A FLAT snapshot (a local convert / env override with a `dit.safetensors` or `transformer/` at its
///   root — the published repo is never flat) keeps the load-time path: the root plus
///   [`resolve_mlx_dense_quant`], i.e. **Q4 by default** (sc-10750's dense-video convention), which the
///   engine applies in memory after the build.
/// * Otherwise a LOUD error naming the tiers. There is deliberately no "just hand it the root" fallback:
///   every published file lives under a tier prefix, so a root fallback could only fail deeper inside
///   the loader with a less actionable message.
///
/// `tier` names the precision that will actually run in both branches — the resolved subdir's own name,
/// or (flat) the tier the load-time quant packs to. It is the request's ASKED-FOR tier only when those
/// coincide, which is the point: the asset records what loaded.
#[cfg(target_os = "macos")]
pub(super) fn resolve_krea_realtime_tier_dir_and_quant(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<KreaRealtimeLoad> {
    let root = resolve_krea_realtime_model_dir(settings)?;
    if let Some((tier, dir)) = krea_realtime_tier_subdir(&root, request) {
        return Ok(KreaRealtimeLoad {
            dir,
            quant: None,
            tier,
        });
    }
    if krea_realtime_dir_has_transformer(&root) {
        let quant = resolve_mlx_dense_quant(request);
        return Ok(KreaRealtimeLoad {
            dir: root,
            quant,
            tier: krea_realtime_quant_tier(quant),
        });
    }
    Err(WorkerError::InvalidPayload(format!(
        "krea_realtime_14b: no complete MLX tier under {}. Install the model (or one of its q4 / q8 / \
         bf16 quality tiers) via the Model Manager — a partially-downloaded tier is skipped, so \
         re-running the download repairs it.",
        root.display(),
    )))
}

/// The tier label a LOAD-TIME quant packs a dense snapshot to (mirrors `mochi_raw_settings`'s mapping):
/// `Q4` ⇒ `q4`, `Q8` ⇒ `q8`, no quant ⇒ the dense `bf16`. Only the flat-snapshot branch needs this — a
/// resolved tier subdir already knows its own name.
#[cfg(target_os = "macos")]
fn krea_realtime_quant_tier(quant: Option<Quant>) -> &'static str {
    match quant {
        Some(Quant::Q4) => "q4",
        Some(Quant::Q8) => "q8",
        _ => "bf16",
    }
}

/// The tier an explicit `mlxQuantize` pick would FETCH on demand, or `None` for a default / `q4` job —
/// `q4` ships with the base install, so nothing to fetch. Split out from
/// [`ensure_krea_realtime_tier_present`] so the "which tier does this job want?" decision is directly
/// testable without a network or an HF cache (the sibling `mochi_wants_q8` / `mochi_wants_bf16` seam).
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_on_demand_tier(request: &VideoRequest) -> Option<&'static str> {
    match resolve_mlx_dense_quant(request) {
        None => Some("bf16"),
        Some(Quant::Q8) => Some("q8"),
        _ => None,
    }
}

/// The repo-relative paths the on-demand fetch asks for — the tier's EXPLICIT file list, each prefixed
/// with `<tier>/`, matching the manifest rather than a `<tier>/*` glob. `None` for an unpublished tier.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_tier_fetch_files(tier: &str) -> Option<Vec<String>> {
    Some(
        krea_realtime_tier_files(tier)?
            .iter()
            .map(|file| format!("{tier}/{file}"))
            .collect(),
    )
}

/// On-demand fetch of a non-default Krea Realtime tier subdir (mirrors
/// [`ensure_scail2_tier_present`](super::scail2)). The macOS default install is the lean `q4/` tier; a
/// job that explicitly opts into a heavier one ([`krea_realtime_on_demand_tier`]) pulls just that
/// subdir from the FIXED [`KREA_REALTIME_REVISION`] the first time it is requested, so
/// [`krea_realtime_tier_subdir`] can resolve it instead of silently falling back to a leaner tier — the
/// creative-tier switch again, now on the fetch side. No-op for a non-Krea model, a `q4`/default job,
/// an already-complete tier, or a repo that isn't downloaded at all (an env-override / local-convert
/// install has no repo to fetch from, and resolve surfaces its own clear error).
///
/// **Then it VERIFIES, because the fetch itself makes no per-file promise.**
/// `ensure_hf_files_cached` does not run the sc-12283 `unmatched_patterns` per-pattern hard-fail — that
/// lives in the user-facing model-download job (`model_jobs::run_model_download_job`) — so a pattern
/// matching nothing remotely is silently a no-op here. Left alone, an explicitly-picked tier whose
/// remote copy is missing a shard would download the other ten files, read INCOMPLETE, and fall back to
/// a leaner tier: precisely the silent downgrade this function exists to prevent. So after the fetch it
/// re-checks completeness on disk and fails loud, NAMING the missing files. That is a stronger check
/// than the per-pattern one (it tests what actually landed, not what the remote listing matched), and
/// it is scoped to this lane rather than changing `ensure_hf_files_cached` for the Wan / SCAIL-2 /
/// Bernini / Mochi callers, whose documented contract is to fall back when a tier is unpublished.
#[cfg(target_os = "macos")]
pub(super) async fn ensure_krea_realtime_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    let Some((repo, revision)) = krea_realtime_tier_repo(&request.model) else {
        return Ok(());
    };
    let Some(tier) = krea_realtime_on_demand_tier(request) else {
        return Ok(());
    };
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, repo) else {
        return Ok(());
    };
    if krea_realtime_tier_is_complete(&root, tier) {
        return Ok(());
    }
    let Some(files) = krea_realtime_tier_fetch_files(tier) else {
        return Ok(());
    };
    crate::model_jobs::ensure_hf_files_cached(api, settings, job, repo, revision, &files).await?;
    // Re-resolve: the fetch may have materialized a NEW snapshot dir for the pinned revision.
    let root = huggingface_snapshot_dir(&settings.data_dir, repo).unwrap_or(root);
    let missing = krea_realtime_missing_tier_files(&root, tier);
    if !missing.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "krea_realtime_14b: the {tier} tier is still incomplete after downloading it from {repo} \
             — missing {}. Falling back to a different tier would silently change the quality tier \
             you picked, so the job stops here.",
            missing.join(", "),
        )));
    }
    Ok(())
}

/// Resolve the Krea Realtime MLX snapshot ROOT: env override (`SCENEWORKS_MLX_KREA_REALTIME_DIR`) →
/// app-managed `<data>/models/mlx/krea_realtime` → the turnkey `SceneWorks/krea-realtime-14b-mlx`
/// snapshot if already downloaded (mirrors `resolve_bernini_model_dir`). Errors clearly if none is
/// present (no stub fallback).
///
/// This is the repo/override root, NOT the directory the engine loads: the published layout is a quant
/// matrix, so callers go through [`resolve_krea_realtime_tier_dir_and_quant`], which descends into the
/// installed tier.
#[cfg(target_os = "macos")]
pub(super) fn resolve_krea_realtime_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Some(dir) = local_mlx_dir(
        settings,
        "SCENEWORKS_MLX_KREA_REALTIME_DIR",
        "krea_realtime",
    ) {
        return Ok(dir);
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, KREA_REALTIME_REPO) {
        return Ok(dir);
    }
    Err(WorkerError::InvalidPayload(format!(
        "krea_realtime_14b: no MLX weights found. Download the turnkey {KREA_REALTIME_REPO} snapshot \
         via the Model Manager, set $SCENEWORKS_MLX_KREA_REALTIME_DIR, or place a converted snapshot \
         at {}.",
        settings
            .data_dir
            .join("models")
            .join("mlx")
            .join("krea_realtime")
            .display(),
    )))
}

// ---------------------------------------------------------------------------
// Wan-family LoRA passthrough (sc-15017, S15)
// ---------------------------------------------------------------------------

/// Build the adapter specs for a Krea Realtime generation. Krea Realtime is a **single dense**
/// transformer — no Lightning distill pair, no MoE high/low experts — so every user LoRA/LoKr is
/// applied shared (`moe_expert: None`) through the same [`resolve_dense_adapters`] the Wan-VACE and
/// SCAIL-2 arms use (sc-8830). Verbatim reuse: the confinement of the payload path to an app-managed
/// root, the directory→`.safetensors` resolve, the `classify_adapter` LoKr tag and the
/// [`MAX_JOB_LORAS`] cap are all that lane's, not a second copy.
///
/// The engine installs each file as a **forward-time residual** over the built DiT
/// (`t2v::load_transformer` → `apply_adapters_strict`), *after* any quantization — which is what
/// makes this tier-agnostic: a Wan style LoRA applies identically over the packed Q4/Q8 bases and the
/// dense bf16 one (the additive-on-packed property epic 10043 / sc-10578 established for the Wan
/// family). The checkpoint IS Wan 2.1 T2V 14B weight-for-weight, so a Wan-family file's
/// `blocks.{i}.self_attn/cross_attn/ffn` stems resolve against the Krea DiT's own module names.
#[cfg(target_os = "macos")]
pub(super) fn resolve_krea_realtime_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
    resolve_dense_adapters(settings, request, MAX_JOB_LORAS)
}

/// A LoRA tensor key the Krea strict installer will NOT apply: the ComfyUI / lightx2v **diff-patch**
/// family, whose deltas are whole dense `<module>.diff` weight / `<module>.diff_b` bias tensors
/// rather than `lora_down`/`lora_up` low-rank factors.
///
/// Krea installs through `apply_adapters_strict`, not the `_with_diff_patch` variant, so these keys
/// are dropped **silently** by the engine — they are not "unmatched targets" (which hard-error), they
/// are simply not the shape that pass looks for. The lightx2v Wan-2.1-14B step-distill file carries
/// 647 of them (200 `.diff` + 447 `.diff_b`) alongside the 406 low-rank targets that DO install, so a
/// user selecting it gets a genuine but PARTIAL application. Switching Krea to the diff-patch variant
/// is deliberately not free — on the default packed Q4 tier every dense per-block fold would skip,
/// making the same file render differently per tier — so it is tracked as its own decision
/// (**sc-15326**) and this module's job is only to make sure the partiality is not silent.
#[cfg(target_os = "macos")]
fn is_diff_patch_key(key: &str) -> bool {
    key.ends_with(".diff") || key.ends_with(".diff_b")
}

/// Split a safetensors header's key list into `(diff-patch keys, tensor keys total)`, ignoring the
/// non-tensor `__metadata__` entry. Pure over the key names so the classification is testable without
/// writing a file (the reader below is the only part that touches disk).
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_diff_patch_key_counts<'a>(
    keys: impl IntoIterator<Item = &'a str>,
) -> (usize, usize) {
    let mut diff_patch = 0;
    let mut total = 0;
    for key in keys {
        if key == "__metadata__" {
            continue;
        }
        total += 1;
        if is_diff_patch_key(key) {
            diff_patch += 1;
        }
    }
    (diff_patch, total)
}

/// How much of each selected LoRA the Krea install path will actually apply — one entry per file that
/// carries diff-patch keys, as `(file name, dropped key count, total tensor keys)`. Empty for the
/// ordinary case (a plain Wan style LoRA is 100% low-rank and applies in full).
///
/// Recorded so a partial application is visible in TWO places rather than nowhere: a `warn`-level
/// structured event in the app's Logs, and the asset's own `rawSettings` (durable provenance a user
/// reads back off the clip, beside `kreaRealtimeTier`). Without it the only signal is the engine's
/// "installed N LoRA target residual(s)" line, which would read like a clean success.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct KreaRealtimeLoraCoverage {
    pub(super) partial: Vec<(String, usize, usize)>,
}

#[cfg(target_os = "macos")]
impl KreaRealtimeLoraCoverage {
    /// The `kreaRealtimeLoraPartial` value for the asset stamp, or `None` when every selected LoRA
    /// applies in full (the common case — the key is then absent rather than an empty array, so the
    /// sidecar of an ordinary render is byte-identical to what it was before this existed).
    fn stamp(&self) -> Option<Value> {
        if self.partial.is_empty() {
            return None;
        }
        Some(Value::Array(
            self.partial
                .iter()
                .map(|(file, dropped, total)| {
                    json!({ "file": file, "unappliedKeys": dropped, "totalKeys": total })
                })
                .collect(),
        ))
    }

    /// Emit the `warn` event for each partially-applied file. Called once, before the load.
    fn warn(&self) {
        for (file, dropped, total) in &self.partial {
            tracing::warn!(
                event = "krea_realtime_lora_partially_applied",
                model = KREA_REALTIME_MODEL_ID,
                lora = %file,
                unapplied_keys = *dropped,
                total_keys = *total,
                "Krea Realtime applies low-rank LoRA factors only: {dropped} of {total} tensors in \
                 {file} are ComfyUI/lightx2v `.diff`/`.diff_b` diff-patch deltas and are NOT applied \
                 (sc-15326). The clip reflects a PARTIAL application of this LoRA."
            );
        }
    }
}

/// Read each resolved adapter's safetensors header and record the files the strict installer will
/// only partially apply. A header that cannot be read is skipped rather than failed here: the file
/// was already opened by `classify_adapter` during resolve, and the engine's own load surfaces a real
/// read failure with a better message than a provenance scan could.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_lora_coverage(adapters: &[AdapterSpec]) -> KreaRealtimeLoraCoverage {
    let mut partial = Vec::new();
    for adapter in adapters {
        let Ok(header) = sceneworks_core::lora_family::read_safetensors_header(&adapter.path)
        else {
            continue;
        };
        let Some(map) = header.as_object() else {
            continue;
        };
        let (dropped, total) = krea_realtime_diff_patch_key_counts(map.keys().map(String::as_str));
        if dropped > 0 {
            partial.push((adapter_file_label(&adapter.path), dropped, total));
        }
    }
    KreaRealtimeLoraCoverage { partial }
}

/// The adapter's file name for user-facing text. Deliberately the last path component, not the full
/// path: the resolved path points inside the app-managed LoRA root, which is noise in a log line and
/// on an asset sidecar.
#[cfg(target_os = "macos")]
fn adapter_file_label(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Rewrite an engine load failure caused by the strict adapter installer into something a user can
/// act on, naming the LoRAs that were selected. Any other failure — and every non-`Msg` variant,
/// notably the typed `Canceled` the funnel matches on — passes through untouched.
///
/// The engine is correct to hard-error: `apply_adapters_strict` refuses to install a file whose
/// targets do not all exist, because the alternative is a render that silently omits part of the
/// adapter and still reports success. But the raw text is an engine-internal module list behind a
/// generic "video load failed", which tells a user nothing about what to do. The one class that
/// reaches real users is an **I2V-family** Wan LoRA (`cross_attn.k_img` / `v_img`): those modules do
/// not exist on a T2V backbone at any surface width — no widening fixes it, the LoRA is genuinely for
/// a different model — so the message says exactly that.
#[cfg(target_os = "macos")]
pub(super) fn annotate_krea_realtime_lora_error(
    error: gen_core::Error,
    loras: &[String],
) -> gen_core::Error {
    let gen_core::Error::Msg(message) = &error else {
        return error;
    };
    if loras.is_empty() || !message.contains("adapters:") {
        return error;
    }
    let selected = loras.join(", ");
    let hint = if message.contains("k_img") || message.contains("v_img") {
        "This looks like an image-to-video (I2V) Wan LoRA: it targets the image cross-attention \
         projections (`cross_attn.k_img` / `v_img`), which a text-to-video backbone like Krea \
         Realtime 14B does not have. Pick a Wan T2V style / motion LoRA instead, or deselect it."
    } else {
        "Krea Realtime 14B accepts Wan-family (Wan 2.1 T2V 14B) LoRAs — its DiT is that model \
         weight-for-weight. A LoRA trained for a different architecture has targets this model does \
         not have, so it is refused rather than half-applied. Deselect it, or pick a Wan T2V LoRA."
    };
    gen_core::Error::Msg(format!(
        "Krea Realtime 14B could not apply the selected LoRA(s) ({selected}). {hint}\n\nEngine \
         detail: {message}"
    ))
}

/// The Krea Realtime video task a request resolves to, from its supplied media: a source clip → `v2v`,
/// else a reference still → `i2v`, else `t2v`. Routed on the MEDIA (matching how the engine's own `run`
/// routes on the conditioning it is handed), not the SceneWorks `mode` string, so the mapping is faithful
/// regardless of which mode label the caller used. Recorded on the asset for observability.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_video_task(request: &VideoRequest) -> &'static str {
    if request.source_clip_asset_id.is_some() {
        "v2v"
    } else if request.source_asset_id.is_some() {
        "i2v"
    } else {
        "t2v"
    }
}

/// Raw-settings recorded on a real MLX Krea Realtime asset (mirrors `bernini_raw_settings`).
///
/// `kreaRealtimeTier` names the tier that actually **LOADED**, not the one the request asked for — the
/// `mochiTier` convention (sc-15258). The two genuinely diverge now that this lane resolves tiers: an
/// explicit `mlxQuantize: 8` with only `q4/` on disk falls back to q4, and a DEFAULT job with only
/// `bf16/` installed loads bf16 dense. Without this stamp the sidecar would still carry the request's
/// verbatim `advanced.mlxQuantize`, so an asset could claim `8` for a clip rendered at q4 — provenance
/// drift on a field a user reads to reproduce a result.
///
/// `kreaRealtimeLoraPartial` (sc-15017) is present ONLY when a selected LoRA was partially applied —
/// see [`KreaRealtimeLoraCoverage`]. Absent for every ordinary render, so the sidecar of a clip with
/// no LoRA (or a fully-applied one) is unchanged.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_raw_settings(
    request: &VideoRequest,
    tier: &str,
    coverage: &KreaRealtimeLoraCoverage,
) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    // The engine task the supplied media resolved to (lineage / observability).
    raw.insert(
        "kreaRealtimeTask".to_owned(),
        Value::String(krea_realtime_video_task(request).to_owned()),
    );
    // The precision the DiT actually ran at — authoritative over the request's `mlxQuantize`.
    raw.insert(
        "kreaRealtimeTier".to_owned(),
        Value::String(tier.to_owned()),
    );
    // Only when something was actually dropped — a clean render's sidecar gains no key.
    if let Some(partial) = coverage.stamp() {
        raw.insert("kreaRealtimeLoraPartial".to_owned(), partial);
    }
    Value::Object(raw)
}

/// Build the Krea Realtime conditioning from the (already loaded) reference media — the PURE i2v/v2v
/// shape decision the engine's `run` mirrors. A source `clip` → one `VideoClip` whose `strength` is
/// HONORED (v2v drives a strength-controlled AR init). Else a reference `still` → one `Reference` with
/// `strength: None` — i2v warms the AR KV cache from the still and the engine reads only the image, so
/// **the i2v strength is a NO-OP** (faithful to the sibling `bernini`/reference paths). Else empty
/// (t2v). Clip takes precedence over a still, exactly as the engine's `run` prefers `VideoClip`.
///
/// Kept pure (loaded media in, conditioning out) so the invariant this story exists to wire — i2v drops
/// strength, v2v carries it — is unit-testable without a GPU, weights, or on-disk assets.
#[cfg(target_os = "macos")]
pub(super) fn krea_realtime_conditioning(
    still: Option<Image>,
    clip: Option<Vec<Image>>,
    v2v_strength: f32,
) -> Vec<Conditioning> {
    if let Some(frames) = clip {
        // v2v: the source clip drives the strength-controlled AR init. `frame_idx` is inert for Krea
        // (its `run` reads only `frames` + `strength`); carried at 0 for the shared `VideoClip` contract.
        return vec![Conditioning::VideoClip {
            frames,
            frame_idx: 0,
            strength: v2v_strength,
        }];
    }
    if let Some(image) = still {
        // i2v: the still warms the AR KV cache. `strength` is a NO-OP for the cache-warm (the engine
        // reads only the image), so it is dropped to `None` — do NOT wire a strength here.
        return vec![Conditioning::Reference {
            image,
            strength: None,
        }];
    }
    Vec::new()
}

/// Resolve a Krea Realtime request's supplied media into the engine conditioning: a source clip
/// (`sourceClipAssetId`) → v2v `VideoClip`, else a reference still (`sourceAssetId`) → i2v `Reference`,
/// else t2v (empty). The clip decodes to the output frame count via the shared `extract_clip_frames`;
/// the still loads via the shared `load_reference_image`. The shape + strength decision itself lives in
/// the pure [`krea_realtime_conditioning`].
#[cfg(target_os = "macos")]
pub(super) async fn resolve_krea_realtime_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    // v2v takes precedence over i2v (matches the engine's `run`): a source clip → v2v.
    if let Some(clip_id) = request.source_clip_asset_id.as_deref() {
        let frames = extract_clip_frames(
            api,
            settings,
            job,
            &request.project_id,
            project_path,
            clip_id,
            request.width,
            request.height,
            wan_frame_count(request.raw_frame_count()),
        )
        .await?;
        // v2v HONORS strength (`videoConditioningStrength`, default 1.0 — the Wan/LTX v2v convention).
        let strength = advanced::f32(&request.advanced, "videoConditioningStrength", 1.0);
        return Ok(krea_realtime_conditioning(None, Some(frames), strength));
    }
    // i2v: a reference still (the I2V conditioning source, like SVD's `sourceAssetId`).
    if let Some(ref_id) = request.source_asset_id.as_deref() {
        let still = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            ref_id,
            project_path,
        )?;
        // i2v strength is a NO-OP (the pure helper drops it); the `1.0` here is never read.
        return Ok(krea_realtime_conditioning(Some(still), None, 1.0));
    }
    // t2v: no conditioning.
    Ok(Vec::new())
}

/// Real MLX Krea Realtime generation (epic 8431 / sc-8443 S10): build the `VideoGenInput` and run the
/// shared `generate_video` heartbeat funnel. The supplied media resolves to the engine conditioning
/// ([`resolve_krea_realtime_conditioning`]) — empty for t2v, a `Reference` still for i2v (strength
/// no-op), a `VideoClip` source for v2v (strength honored). CFG off: no negative prompt / guidance
/// (the engine advertises none), `steps` from the advanced `steps` knob (else the engine default).
/// The model dir, the load-time `quant` and the tier label come from the single
/// [`resolve_krea_realtime_tier_dir_and_quant`] decision (sc-15258): the installed tier subdir with
/// `quant = None`, or a flat local snapshot with the Q4-default load-time quant. Frame count uses the
/// Wan 1-mod-4 stride coercion — the DiT + z16 VAE ARE stock Wan 2.1, so the Wan stride is exactly
/// right.
///
/// Returns the decoded clip plus the `rawSettings` to record on the asset (the `generate_mochi` shape):
/// the arm owns them because two of the fields — the tier that actually LOADED and any partial LoRA
/// application — are only knowable inside it, and re-deriving them at the call site would mean
/// re-resolving the tier and re-reading every adapter header.
///
/// Runs through the `generate_video` → `generate_video_using` funnel so a long/multi-minute AR job
/// drives `heartbeat(...)` and is NOT marked `interrupted` by the ~90 s API stale-sweep, and so the
/// funnel's per-step cancel poll trips the engine's `CancelFlag` promptly (the S8 half of this story's
/// DoD).
///
/// **LoRA passthrough (sc-15017, S15).** User-selected LoRAs ride `LoadSpec::adapters` via
/// [`resolve_krea_realtime_adapters`] — the same shared dense resolver the Wan-VACE / SCAIL-2 arms
/// use — and the engine installs them as forward-time residuals. Two honesty seams travel with it:
/// a file the engine cannot fully match hard-errors, rewritten here into an actionable message
/// ([`annotate_krea_realtime_lora_error`]); and a file carrying diff-patch deltas installs only
/// PARTIALLY, which is warned about and stamped rather than passed off as a clean apply
/// ([`KreaRealtimeLoraCoverage`]).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_krea_realtime(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    generate_krea_realtime_using(
        api,
        settings,
        job,
        request,
        project_path,
        engine_id,
        backend,
        crate::inference_runtime::load,
    )
    .await
}

/// [`generate_krea_realtime`] with the engine loader supplied by the caller (mirrors
/// `generate_mochi_using`, sc-12318). With the loader threaded in, a test can drive this arm against a
/// stub `Generator` and assert on the `GenerationRequest` that actually reached the engine (prompt /
/// steps / seed / frame count / fps and the resolved conditioning) without weights or a GPU.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_krea_realtime_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<(DecodedVideo, Value)> {
    let conditioning =
        resolve_krea_realtime_conditioning(api, settings, job, request, project_path).await?;
    // Wan-family LoRA passthrough (sc-15017). Resolved BEFORE the tier fetch so a bad selection (an
    // over-cap count, a path outside the app-managed root, a missing file) fails immediately instead
    // of after a multi-GB download.
    let adapters = resolve_krea_realtime_adapters(settings, request)?;
    // Files the strict installer will only partially apply — warned now, stamped on the asset below.
    let coverage = krea_realtime_lora_coverage(&adapters);
    coverage.warn();
    let lora_labels: Vec<String> = adapters
        .iter()
        .map(|adapter| adapter_file_label(&adapter.path))
        .collect();
    // Krea Realtime quant matrix (sc-15258, epic 8506 shape): an explicitly picked q8/bf16 tier is
    // fetched on demand before resolving (no-op for a default job or an already-present tier), then the
    // tier subdir AND the load-time quant come out of ONE decision — a resolved tier loads exactly as
    // it sits on disk (`quant = None`), a flat local snapshot keeps the Q4-default load-time quant.
    ensure_krea_realtime_tier_present(api, settings, job, request).await?;
    let load = resolve_krea_realtime_tier_dir_and_quant(settings, request)?;
    let tier = load.tier;
    // Wrap the caller's loader so a strict-adapter refusal reaches the user as guidance instead of an
    // engine-internal module list behind "video load failed". Non-adapter failures — and the typed
    // `Canceled` the funnel matches on — pass through untouched.
    let load_generator = move |engine: &str, spec: &LoadSpec| {
        load_generator(engine, spec)
            .map_err(|error| annotate_krea_realtime_lora_error(error, &lora_labels))
    };
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir: load.dir,
        quant: load.quant,
        conditioning,
        prompt: request.prompt.clone(),
        // CFG off: the engine advertises no negative-prompt / guidance axis, so leave both unset.
        negative_prompt: None,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps: super::wan::advanced_opt_u32(request, "steps"),
        seed: resolve_video_seed(request) as u64,
        // No `video_mode`: the engine routes purely on the supplied conditioning, not a task string.
        // User LoRAs ride here → `video_load_spec` → `LoadSpec::adapters` (sc-15017).
        adapters,
        ..VideoGenInput::default()
    };
    let decoded = generate_video_using(
        api,
        settings,
        job,
        backend,
        &request.advanced,
        input,
        load_generator,
    )
    .await?;
    Ok((
        decoded,
        krea_realtime_raw_settings(request, tier, &coverage),
    ))
}
