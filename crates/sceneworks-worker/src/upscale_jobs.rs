//! Image and dataset upscaling orchestration.
//!
//! Real-ESRGAN x2/x4 execution and numeric tiling belong to the pinned runtime's CV
//! engine. The worker retains model resolution, its factor/path cache, job progress,
//! persistence and cancellation. Both CoreML (Mac) and CUDA (off-Mac) use the
//! shared instrumented ORT loader and preserve the existing CPU fallback.
//!
//! SeedVR2 and Iris continue through their existing generator/runtime interfaces.
//! Model resolution remains cache-only: no weights download during rendering.

// `HashMap`/`Mutex`/`OnceLock` back the Real-ESRGAN per-factor session cache (the `ort` path,
// macOS + the off-Mac candle lane).
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::asset_media::resolve_asset_media;
use crate::cv::upscale::Upscaler;
use crate::generator_cache::with_cached_generator;
use crate::single_child_asset::{write_single_child_asset, SingleChildAssetSpec};
use crate::util::resolve_env_file_pin;
use gen_core::{
    CancelFlag, Conditioning, GenerationOutput, GenerationRequest, Image as GenImage, LoadSpec,
    WeightsSource,
};
use image::RgbImage;
use serde_json::{json, Value};

use crate::{
    heartbeat, parse_dataset_items, progress_payload, safe_project_path, task_join_error,
    update_job, ApiClient, DatasetItemPolicy, Settings, WorkerError, WorkerResult,
};
use sceneworks_core::contracts::{JobSnapshot, JobStatus, JsonObject, ProgressStage, WorkerStatus};
use sceneworks_core::project_store::ProjectStore;

/// SeedVR2's engine-enforced per-side output cap (the registry descriptor's `max_size`; the engine
/// errors `"size … outside supported range 16..=4096"` past it). Tighter than the generic
/// 8192-pixel CV output limit, so a >2048px source at ×2 (or >1024 at ×4) must be rejected with a
/// clear, actionable message here rather than failing deep in the engine as a raw "Image upscale failed"
/// (sc-8228 secondary gap).
const SEEDVR2_MAX_OUTPUT_DIMENSION: u32 = 4096;
const CANCEL_MESSAGE: &str = "Image upscale canceled by user.";

/// SceneWorks-owned HuggingFace repo hosting the pre-exported ONNX (reproducible from
/// `scripts/spikes/sc3489_export_reference.py`). Public; declared by the `real_esrgan` catalog
/// entry and installed from the Model Manager (sc-17633) — NOT downloaded on first use any more.
/// The same export feeds the off-Mac CUDA EP (sc-5499). Overridable via the manifest `onnx`
/// resource or the env pin `SCENEWORKS_REALESRGAN_X{2,4}_ONNX`.
const ONNX_REPO: &str = "SceneWorks/real-esrgan-onnx";
/// Pinned Real-ESRGAN ONNX revision (sc-9682, F-077 follow-up). Even though
/// `SceneWorks/real-esrgan-onnx` is a first-party repo, fetching the mutable `main`
/// branch means a re-push (or a compromised token) could silently swap the ONNX graph
/// we load. Pin the exact commit for defense-in-depth, mirroring the SeedVR2 pin
/// (sc-8879). The `real_esrgan` catalog entry declares this SAME revision (sc-17633), so the
/// Model Manager installs exactly the snapshot [`resolve_onnx`] then reads; the integrity check
/// against each file's `lfs.oid` happens there, in the download job, not at load time.
const ONNX_REVISION: &str = "09f741bac80a246b407da3ee902bf5f3291b602f";

fn onnx_file(factor: u8) -> String {
    format!("real_esrgan_x{factor}.onnx")
}

// ---------------------------------------------------------------------------
// pure tiling math (ported from upscalers.py; unit-tested without weights) — Real-ESRGAN
// ---------------------------------------------------------------------------

static UPSCALERS: OnceLock<Mutex<HashMap<(u8, PathBuf), Upscaler>>> = OnceLock::new();

fn upscaler_cache_key(factor: u8, onnx_path: &Path) -> (u8, PathBuf) {
    (factor, onnx_path.to_path_buf())
}

fn validate_upscale_target_dimensions(width: u32, height: u32) -> WorkerResult<()> {
    crate::cv::upscale::validate_upscale_target_dimensions(width, height).map_err(Into::into)
}

/// Blocking upscale: load+cache the factor's session (amortising the CoreML/CUDA graph
/// compile across a batch), run it. All `ort` objects live inside this closure and
/// never cross an await (mirrors `pose_jobs::detect_batch`). Returns the upscaled image plus the
/// session's actual execution device (`coreml`/`cuda`/`cpu`) so the caller can surface whether the
/// hardware EP was used or it fell back to CPU (sc-8923, mirrors `pose_jobs` reporting
/// `detector.device`).
fn upscale_blocking(
    onnx_path: PathBuf,
    factor: u8,
    img: RgbImage,
    cancel: CancelFlag,
) -> WorkerResult<(RgbImage, &'static str)> {
    use std::collections::hash_map::Entry;
    let cell = UPSCALERS.get_or_init(|| Mutex::new(HashMap::new()));
    // Recover from a poisoned lock rather than panicking every subsequent job: if a
    // prior upscale panicked mid-run holding this lock, take the inner guard and
    // clear the possibly-corrupt cached sessions so the match below reloads a fresh
    // one for this factor (sc-4277 / F-MLXW-13).
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        guard.clear();
        guard
    });
    let upscaler = match guard.entry(upscaler_cache_key(factor, &onnx_path)) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => e.insert(Upscaler::load(&onnx_path, &mut crate::cv::ModelIoLoader)?),
    };
    let device = upscaler.device;
    let out = upscaler.upscale(&img, factor as usize, &cancel)?;
    Ok((out, device))
}

// ---------------------------------------------------------------------------
// ONNX weight provisioning (cache-only since sc-17633; mirrors Python resolution order)
// ---------------------------------------------------------------------------

/// What a set-but-INCOMPLETE value means, which differs per key because the two keys did not
/// historically mean the same thing. See [`SEEDVR2_DIR_PINS`] for the full story.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SeedVr2Pin {
    /// The key NAMES a checkpoint, so incomplete is an operator error and fails loudly (sc-8911).
    NamedCheckpoint,
    /// The key named a STAGING DESTINATION the worker used to create and download into, so empty
    /// is a legal state for it and incomplete falls through to the next candidate.
    StagingDir,
}

/// Resolve one SeedVR2 checkpoint-dir env pin. Unset → `Ok(None)`. Set and holding both checkpoint
/// files → `Ok(Some(dir))`. Testable without env mutation via the explicit raw value.
///
/// Set but INCOMPLETE splits on `kind`, and the split is the whole point (sc-17632 review):
///
/// * [`SeedVr2Pin::NamedCheckpoint`] → an `InvalidPayload` error naming the key and the files that
///   are actually missing. This is sc-8911's rule, kept verbatim: an operator who points the worker
///   at a specific checkpoint must not silently get a different one, and an error they can read and
///   unset beats loading weights they did not ask for. The message tells them how to fix it.
/// * [`SeedVr2Pin::StagingDir`] → a loud `warn!` and `Ok(None)`, so resolution continues to the HF
///   cache. Erroring here would be a regression invented by this story rather than a rule anyone
///   chose: the key used to be a *download destination*, so an empty one is exactly the state it
///   was in before first use, and hard-failing on it would let a stale developer export make a
///   perfectly good installed model unreachable.
fn resolve_seedvr2_dir_pin(
    key: &str,
    kind: SeedVr2Pin,
    value: Option<std::ffi::OsString>,
) -> WorkerResult<Option<PathBuf>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let dir = PathBuf::from(&value);
    let missing: Vec<&str> = [SEEDVR2_DIT_FILE, SEEDVR2_VAE_FILE]
        .into_iter()
        .filter(|file| !dir.join(file).exists())
        .collect();
    if missing.is_empty() {
        return Ok(Some(dir));
    }
    // Name the files that are ACTUALLY absent rather than "X and/or Y": with two files, "and/or"
    // makes the operator check both when we already know which one is wrong.
    let missing = missing.join(" and ");
    match kind {
        SeedVr2Pin::NamedCheckpoint => Err(WorkerError::InvalidPayload(format!(
            "{key} is set to {} but that directory is missing {missing}. Point it at a complete \
             checkpoint dir, or unset it to use the copy installed from the Model Manager.",
            dir.display()
        ))),
        SeedVr2Pin::StagingDir => {
            tracing::warn!(
                pin = key,
                dir = %dir.display(),
                %missing,
                "ignoring {key}: it names an incomplete SeedVR2 checkpoint dir, falling back to \
                 the installed copy. Unset it, or populate it, to silence this."
            );
            Ok(None)
        }
    }
}

/// Resolve the ONNX for `factor`, **cache-only** (sc-17633, epic 17625). Order:
///
/// 1. the explicit env pins `SCENEWORKS_REALESRGAN_X{factor}_ONNX`, then `SCENEWORKS_REALESRGAN_ONNX`;
/// 2. the **HF cache**, where the `real_esrgan` Model Manager install puts it — the repo/file come
///    from the job's `resources.imageUpscalers.real-esrgan.x{factor}.onnx`, falling back to
///    [`ONNX_REPO`];
/// 3. the legacy `<data_dir>/cache/upscale/` copy, read-only.
///
/// There is no download half. This used to fetch ~67 MB per factor mid-upscale via
/// `ensure_hf_cached_file` into `<data_dir>/cache/upscale/` — a destination the Models screen can
/// neither size nor delete (6.9 GiB observed on the audit machine). The manifest now declares the
/// ONNX artifacts, so the weights arrive from the Model Manager like every other model, and a miss
/// here is an actionable install error rather than a surprise transfer.
///
/// Step 3 is the migration path (AC10 of sc-17598): an existing install has these bytes under
/// `cache/upscale/` already, so it keeps working and does not re-download. Nothing writes there any
/// more, so that copy only drains.
fn resolve_onnx(settings: &Settings, factor: u8, manifest_entry: &Value) -> WorkerResult<PathBuf> {
    for key in [
        format!("SCENEWORKS_REALESRGAN_X{factor}_ONNX"),
        "SCENEWORKS_REALESRGAN_ONNX".to_owned(),
    ] {
        // A set-but-missing env pin is an operator error: fail loudly instead of silently
        // falling through and loading different weights than the operator asked for (sc-8911).
        if let Some(path) = resolve_env_file_pin(
            &key,
            std::env::var_os(&key),
            &format!("the local Real-ESRGAN x{factor} ONNX export"),
        )? {
            return Ok(path);
        }
    }

    // manifest resource: resources.imageUpscalers.real-esrgan.x{factor}.onnx -> {repo,file}
    let (repo, file) = manifest_onnx_resource(manifest_entry, factor)
        .unwrap_or_else(|| (ONNX_REPO.to_owned(), onnx_file(factor)));
    // Pin the exact commit for the default first-party repo so `main` moving under us can't swap
    // the ONNX graph (sc-9682) — the same pin the catalog entry declares, so install and load agree.
    // A manifest-supplied override repo may carry its own revision layout, so only pin the default.
    let revision = if repo == ONNX_REPO {
        ONNX_REVISION
    } else {
        "main"
    };
    if let Some(path) =
        crate::downloads::resolve_hf_component_file(settings, &repo, revision, &file)
    {
        return Ok(path);
    }

    // Legacy pre-migration destination, read-only.
    let legacy = settings
        .data_dir
        .join("cache")
        .join("upscale")
        .join(onnx_file(factor));
    if legacy.exists() {
        return Ok(legacy);
    }

    Err(WorkerError::InvalidPayload(format!(
        "The Real-ESRGAN x{factor} upscaler weights are not installed ({repo}/{file}). Install \
         \"Real-ESRGAN Upscaler\" from the Model Manager, or set \
         SCENEWORKS_REALESRGAN_X{factor}_ONNX to a local export."
    )))
}

/// Pull a `{repo,file}` ONNX resource out of a job's `modelManifestEntry` if present:
/// `resources.imageUpscalers.real-esrgan.x{factor}.onnx`.
fn manifest_onnx_resource(manifest_entry: &Value, factor: u8) -> Option<(String, String)> {
    let onnx = manifest_entry
        .get("resources")?
        .get("imageUpscalers")
        .or_else(|| manifest_entry.get("resources")?.get("upscalers"))?
        .get("real-esrgan")?
        .get(format!("x{factor}"))?
        .get("onnx")?;
    let repo = onnx.get("repo")?.as_str()?.to_owned();
    let file = onnx
        .get("file")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| onnx_file(factor));
    Some((repo, file))
}

// ---------------------------------------------------------------------------
// SeedVR2 native-MLX upscaler (epic 4811 / sc-4815)
// ---------------------------------------------------------------------------

/// Upstream HuggingFace repo holding the raw SeedVR2 ComfyUI checkpoint. The `mlx-gen-seedvr2` /
/// `candle-gen-seedvr2` registries load it directly (converting to the backend layout in-memory, no
/// Python). Public; declared by the `seedvr2_upscaler` catalog entry and installed from the Model
/// Manager (sc-17632) — NOT downloaded on first use any more. Overridable via
/// `SCENEWORKS_SEEDVR2_CHECKPOINT` / `SCENEWORKS_SEEDVR2_DIR`.
///
/// This is the ONE repo both SeedVR2 lanes read: the image upscaler here and the `video_upscale`
/// job in [`crate::video_jobs`]. Until sc-17632 each lane owned a private copy of these constants
/// and downloaded the same ~7.3 GB twice, into two different `<data_dir>/cache` subtrees.
pub(crate) const SEEDVR2_REPO: &str = "numz/SeedVR2_comfyUI";
/// Pinned SeedVR2 checkpoint revision (sc-8879). `numz/SeedVR2_comfyUI` is a third-party
/// mirror; fetching the mutable `main` branch means an upstream re-push would silently
/// change the weights we load. Pin the exact commit that carries the 3B fp16 DiT + VAE
/// (`seedvr2_ema_3b_fp16.safetensors` / `ema_vae_fp16.safetensors`).
///
/// The `seedvr2_upscaler` catalog entry declares this SAME revision (sc-17632), so the Model
/// Manager installs exactly the snapshot [`resolve_seedvr2_checkpoint_dir`] then reads; the
/// integrity check against each file's `lfs.oid` happens there, in the download job, not at load
/// time. `manifest_declares_the_same_seedvr2_pin_the_loader_reads` fails if the two drift.
pub(crate) const SEEDVR2_REVISION: &str = "09ced71023636e9bc8cdf9cdecfb2625d1e691e8";
/// The exact filenames `Seedvr2Pipeline::load` expects in the checkpoint dir (3B fp16 DiT + VAE).
/// They are byte-identical to the upstream filenames, which is why nothing renames them: the
/// checkpoint dir the engine is handed is the HF snapshot dir itself.
const SEEDVR2_DIT_FILE: &str = "seedvr2_ema_3b_fp16.safetensors";
const SEEDVR2_VAE_FILE: &str = "ema_vae_fp16.safetensors";

/// Operator overrides naming a local SeedVR2 checkpoint dir, in precedence order, each with what a
/// set-but-INCOMPLETE value means for that key.
///
/// # One knob per lane, now read by both lanes
///
/// The two keys are historical siblings, not synonyms someone added for convenience.
/// `SCENEWORKS_SEEDVR2_CHECKPOINT` was the **image** upscaler's and has always been a *pin*: it
/// named a checkpoint, and an incomplete one was a loud error (sc-8911). `SCENEWORKS_SEEDVR2_DIR`
/// was the **video** upscaler's and was a **staging destination** — the pre-sc-17632 code created
/// that directory and downloaded the DiT + VAE *into* it, so an empty one was the normal state
/// before first use. Each key was completely INERT on the other lane.
///
/// Collapsing the two lanes onto one resolver (sc-17632) means **both keys are now read on both
/// lanes**, and there is nothing left to stage into — every candidate here is resolve-only. The
/// consequence a reader needs to know: an export that used to affect only video upscaling now
/// influences image upscaling too, and vice versa.
///
/// # Why the strictness differs
///
/// Because the widening is exactly where a hard error would do new damage. Keeping
/// `SCENEWORKS_SEEDVR2_DIR` strict would mean a stale developer export — legally empty under its
/// old meaning — could make a correctly installed model unreachable on a lane that export never
/// touched. So it warns and falls through, while `SCENEWORKS_SEEDVR2_CHECKPOINT` keeps sc-8911's
/// hard failure: it names a checkpoint, so incomplete means the operator got it wrong, and the
/// error says which files are missing and that unsetting it uses the installed copy.
///
/// The residual, stated rather than hidden: an incomplete `SCENEWORKS_SEEDVR2_CHECKPOINT` now fails
/// the **video** lane too, which it did not before. That is sc-8911's deliberate rule reaching a
/// lane that loads the same model from the same repo — not a new rule — and it is loud and
/// self-describing when it fires.
const SEEDVR2_DIR_PINS: &[(&str, SeedVr2Pin)] = &[
    ("SCENEWORKS_SEEDVR2_CHECKPOINT", SeedVr2Pin::NamedCheckpoint),
    ("SCENEWORKS_SEEDVR2_DIR", SeedVr2Pin::StagingDir),
];

/// Pre-migration checkpoint roots under `<data_dir>`, read-only (AC10 of sc-17598).
///
/// Before sc-17632 the two SeedVR2 lanes each fetched the SAME ~7.3 GB repo into its OWN cache
/// subtree — the image upscaler into `cache/upscale/seedvr2/`, the video upscaler into
/// `cache/seedvr2-mlx/` — so an install that used both engines paid for the checkpoint twice, in a
/// tree the Models screen can neither size nor delete. Both are read here so an existing install of
/// EITHER lane satisfies BOTH and the upgrade re-downloads nothing. Nothing WRITES to them any
/// more, so they only drain; reclaiming them is tracked on sc-17636 (S11).
const SEEDVR2_LEGACY_DIRS: &[&str] = &["cache/upscale/seedvr2", "cache/seedvr2-mlx"];

/// Round up to the nearest multiple of 16 (the SeedVR2 VAE /8 · patch /2 constraint the registry
/// validates), floored at 16. `factor × src` is usually already a multiple of 16; odd sizes round.
fn round_to_16(v: u32) -> u32 {
    (v.div_ceil(16)).max(1) * 16
}

/// Resolve the raw SeedVR2 checkpoint dir — the dir holding the canonical DiT + VAE filenames the
/// engine loads — **cache-only** (sc-17632, epic 17625). Order mirrors [`resolve_onnx`]:
///
/// 1. the explicit dir pins [`SEEDVR2_DIR_PINS`];
/// 2. the **HF cache**, where the `seedvr2_upscaler` Model Manager install puts it — the pinned
///    `snapshots/<sha>/` of [`SEEDVR2_REPO`] at [`SEEDVR2_REVISION`];
/// 3. the legacy [`SEEDVR2_LEGACY_DIRS`] copies, read-only.
///
/// Both files must be present **from the same root** — a torn pair is not a resolution, exactly as
/// in [`crate::person_segment_sam3_common::resolve_segmenter_weights`].
///
/// There is no download half. This used to fetch ~7.3 GB mid-render via `ensure_hf_cached_file`,
/// and the `video_upscale` lane fetched the identical repo again into a second destination. Both
/// lanes now call THIS function, so the checkpoint is installed once, into the HF cache, and a miss
/// is an actionable install error rather than a surprise 7.3 GB transfer inside a render.
///
/// # No rename, and nothing may canonicalize this path
///
/// The engine consumes the returned **directory** (`WeightsSource::Dir` → `Seedvr2Pipeline::load`),
/// and an HF snapshot dir holds relative SYMLINKS into extensionless blobs. The symlink names keep
/// their `.safetensors` extension, and the safetensors loader dispatches on the extension, so this
/// works only as long as nothing calls `canonicalize()` on the path (see the mlx-rs
/// extension-dispatch note). Do not add one. The deleted pre-migration code copied each file to a
/// "canonical" name under `cache/`; that rename was already a no-op — upstream's filenames ARE the
/// canonical ones — and its only live branch was a manifest override that no SceneWorks surface
/// could populate.
pub(crate) fn resolve_seedvr2_checkpoint_dir(settings: &Settings) -> WorkerResult<Option<PathBuf>> {
    let pair_in = |dir: PathBuf| -> Option<PathBuf> {
        (dir.join(SEEDVR2_DIT_FILE).exists() && dir.join(SEEDVR2_VAE_FILE).exists()).then_some(dir)
    };
    // An operator dir pin wins over anything installed. A set-but-incomplete one is either a loud
    // error or a warned fall-through depending on which key carried it — see `SEEDVR2_DIR_PINS`.
    for (key, kind) in SEEDVR2_DIR_PINS {
        if let Some(dir) = resolve_seedvr2_dir_pin(key, *kind, std::env::var_os(key))? {
            return Ok(Some(dir));
        }
    }
    // The installed location: whatever the `seedvr2_upscaler` Model Manager entry cached. Ask for
    // ONE file, take its parent — the pinned `snapshots/<sha>/` dir the engine is handed — then
    // require the other beside it, so the pair is whole PER ROOT exactly as in the legacy loop
    // below. Deriving the dir from a resolved file rather than rebuilding the cache path keeps the
    // `is_pinned_hf_revision` branch decision in one place; `.parent()` is lexical, and nothing on
    // this path may `canonicalize` (see above).
    if let Some(dit) = crate::downloads::resolve_hf_component_file(
        settings,
        SEEDVR2_REPO,
        SEEDVR2_REVISION,
        SEEDVR2_DIT_FILE,
    ) {
        if let Some(dir) = dit.parent().map(Path::to_path_buf).and_then(&pair_in) {
            return Ok(Some(dir));
        }
    }
    // Legacy pre-migration roots, read-only (AC10). Whole-pair per root, so a half-populated legacy
    // dir falls through to the next candidate instead of handing the engine a torn checkpoint.
    for sub in SEEDVR2_LEGACY_DIRS {
        if let Some(dir) = pair_in(settings.data_dir.join(sub)) {
            return Ok(Some(dir));
        }
    }
    Ok(None)
}

/// The SeedVR2 checkpoint dir, or an actionable install error (sc-17632 / S10, sc-17635).
///
/// Taking `&Settings` and **no `DownloadContext`** is the enforcement, not an aesthetic choice:
/// neither SeedVR2 lane can reach the network any more, by construction rather than by convention
/// (epic 17625).
pub(crate) fn require_seedvr2_checkpoint_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    resolve_seedvr2_checkpoint_dir(settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "The SeedVR2 upscaler is not installed. Install \"SeedVR2 Upscaler\" from the Model \
             Manager, or point SCENEWORKS_SEEDVR2_CHECKPOINT at a local {SEEDVR2_REPO} checkpoint \
             dir."
        ))
    })
}

/// Run a SeedVR2 image upscale: the LR `source` (native resolution) → a `round_to_16(factor×)`
/// super-resolved RGB image. Goes through the shared single-resident generator cache
/// (`with_cached_generator`) so loading SeedVR2 evicts any cached image-gen engine (the engine's
/// image path has no spatial tiling — keeping one resident model bounds peak memory). `softness`
/// (0..1) is the optional `--softness` pre-blur; `seed` makes the generative result reproducible.
async fn run_seedvr2_upscale(
    dir: PathBuf,
    source: RgbImage,
    factor: u8,
    softness: f32,
    seed: u64,
    cancel: CancelFlag,
) -> WorkerResult<RgbImage> {
    let (src_w, src_h) = (source.width(), source.height());
    let target_w = round_to_16(src_w.saturating_mul(u32::from(factor)));
    let target_h = round_to_16(src_h.saturating_mul(u32::from(factor)));
    validate_upscale_target_dimensions(target_w, target_h)?;
    // SeedVR2 hard-caps each output side at 4096 (engine `max_size`); pre-validate so an oversized
    // target surfaces a clear, actionable message instead of a raw engine failure (sc-8228).
    if target_w > SEEDVR2_MAX_OUTPUT_DIMENSION || target_h > SEEDVR2_MAX_OUTPUT_DIMENSION {
        return Err(WorkerError::InvalidPayload(format!(
            "SeedVR2 supports up to {SEEDVR2_MAX_OUTPUT_DIMENSION}px per side; a {factor}× upscale of \
             {src_w}x{src_h} would be {target_w}x{target_h}. Use a smaller factor or source image, or \
             pick the Real-ESRGAN upscaler for larger targets."
        )));
    }
    let image = GenImage {
        width: src_w,
        height: src_h,
        pixels: source.as_raw().clone(),
    };

    with_cached_generator(
        "seedvr2",
        LoadSpec::new(WeightsSource::Dir(dir)),
        "SeedVR2 engine load",
        move |generator| {
            let request = GenerationRequest {
                width: target_w,
                height: target_h,
                count: 1,
                seed: Some(seed),
                softness: Some(softness),
                conditioning: vec![Conditioning::Reference {
                    image,
                    strength: None,
                }],
                cancel: cancel.clone(),
                ..Default::default()
            };
            let output = generator
                .generate(&request, &mut |_progress| {})
                .map_err(|error| match error {
                    gen_core::Error::Canceled => WorkerError::Canceled(error.to_string()),
                    other => WorkerError::Engine(format!("SeedVR2 upscale failed: {other}")),
                })?;
            let image = match output {
                GenerationOutput::Images(mut images) if !images.is_empty() => images.remove(0),
                GenerationOutput::Images(_) => {
                    return Err(WorkerError::Engine("SeedVR2 produced no image".to_owned()))
                }
                other => {
                    return Err(WorkerError::Engine(format!(
                        "SeedVR2 returned non-image output: {other:?}"
                    )))
                }
            };
            RgbImage::from_raw(image.width, image.height, image.pixels)
                .ok_or_else(|| WorkerError::Engine("SeedVR2 image buffer size mismatch".to_owned()))
        },
    )
    .await
}

/// Upscale one in-memory RGB image with the selected engine, returning the upscaled image.
/// Shared by the standalone `image_upscale` job and the Image Studio inline "Upscale" toggle
/// (sc-8091): Real-ESRGAN runs via the cached `ort` session on a blocking thread; SeedVR2 runs
/// in-process through the registry generator. `manifest_entry` may be `Value::Null` (the inline
/// path carries the *generation* model's manifest, not an upscaler one) — the Real-ESRGAN resolver
/// then falls back to the default HF repo. `engine_id` is the canonical id (`seedvr2`, else
/// Real-ESRGAN); the caller normalises before passing it in.
///
/// Takes no `reqwest::Client`: since sc-17633 (Real-ESRGAN) and sc-17632 (SeedVR2) both weight
/// resolvers are cache-only, so this whole lane is unable to reach the network (epic 17625).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn upscale_image_in_memory(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    manifest_entry: &Value,
    engine_id: &str,
    factor: u8,
    softness: f32,
    seed: u64,
    source: RgbImage,
    cancel: &CancelFlag,
) -> WorkerResult<RgbImage> {
    // Both branches run their long compute under `run_upscale_with_heartbeat` so the worker keeps
    // emitting `Busy` heartbeats (every 5-15s) during model load + diffusion. Without it, an inline
    // SeedVR2 upscale (model load + one-step diffusion, easily > the API's 90s worker-timeout) goes
    // silent and the stale-sweep marks the in-flight image-generate job `interrupted` mid-flight —
    // then the worker's terminal post is rejected (409) and the job looks stuck (sc-8186). The
    // standalone `image_upscale` job already wraps the same compute this way.
    match engine_id {
        "seedvr2" => {
            let dir = require_seedvr2_checkpoint_dir(settings)?;
            let task_cancel = cancel.clone();
            run_upscale_with_heartbeat(
                api,
                settings,
                &job.id,
                cancel.clone(),
                crate::load_progress::spawn(async move {
                    run_seedvr2_upscale(dir, source, factor, softness, seed, task_cancel).await
                }),
            )
            .await
        }
        _ => {
            let onnx = resolve_onnx(settings, factor, manifest_entry)?;
            let task_cancel = cancel.clone();
            // This in-memory path feeds inline generation and does not build a device-reporting
            // result, so the session device is discarded here (surfaced on the standalone
            // `image_upscale` job's rawAdapterSettings instead, sc-8923).
            let (out, _device) = run_upscale_with_heartbeat(
                api,
                settings,
                &job.id,
                cancel.clone(),
                crate::load_progress::spawn_blocking(move || {
                    upscale_blocking(onnx, factor, source, task_cancel)
                }),
            )
            .await?;
            Ok(out)
        }
    }
}

/// Upscale keepalive: the shared [`crate::run_blocking_with_heartbeat`] (sc-8390) with the upscale's
/// one extra behavior — the instant a cancel is observed it posts an intermediate `Running`/"Canceling
/// image upscale." update so the UI acknowledges the cancel while the long diffusion finishes rather
/// than appearing frozen until it flips terminal. That extra post is now the shared helper's
/// `on_cancel_acknowledged` hook (sc-8928), so the sc-8390-critical select/heartbeat/teardown loop
/// lives in exactly ONE place instead of a maintained-in-sync copy.
async fn run_upscale_with_heartbeat<R>(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    cancel: CancelFlag,
    task: tokio::task::JoinHandle<WorkerResult<R>>,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    crate::run_blocking_with_heartbeat(
        api,
        settings,
        job_id,
        Some(cancel),
        CANCEL_MESSAGE,
        "upscale task",
        Some(|| async move {
            crate::progress::update_job_cancelling(
                api,
                job_id,
                progress_payload(
                    JobStatus::Running,
                    ProgressStage::Running,
                    0.45,
                    "Canceling image upscale.",
                    None,
                    None,
                    None,
                ),
            )
            .await
            .map(|_| ())
        }),
        task,
    )
    .await
}

// ---------------------------------------------------------------------------
// job handler
// ---------------------------------------------------------------------------

/// Resolve the requested image-upscale factor, rejecting anything other than the two supported
/// scales (2x/4x — the only `real_esrgan_x{factor}.onnx` exports and SeedVR2 targets). Mirrors the
/// video path's [`resolve_video_upscale_factor`](crate::video_jobs) (F-118 / sc-8920): an
/// unsupported factor now fails loudly instead of being silently coerced to 2x, which produced a
/// quietly-different output at a different resolution.
fn resolve_image_upscale_factor(factor: u64) -> WorkerResult<u8> {
    match factor {
        2 | 4 => Ok(factor as u8),
        other => Err(WorkerError::InvalidPayload(format!(
            "Image upscale supports only factor 2 or 4 (got {other})."
        ))),
    }
}

pub(crate) async fn run_image_upscale_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    // Iris restoration (sc-25683) is its own sized-by-scale engine with its own job body; every
    // other engine keeps the path below unchanged.
    let requested_engine = job
        .payload
        .get("engine")
        .and_then(Value::as_str)
        .unwrap_or("real-esrgan");
    if sceneworks_core::image_restoration::is_iris_engine(requested_engine) {
        return run_iris_restore_job_using(
            api,
            settings,
            job,
            crate::iris_restore::registry_transform_loader,
        )
        .await;
    }
    // `scale` / `inputSizing` / `colorFix` are Iris controls; another engine refuses them by name
    // rather than ignoring them.
    sceneworks_core::image_restoration::reject_iris_only_keys(&job.payload, requested_engine)
        .map_err(WorkerError::InvalidPayload)?;
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.12,
            "Loading source image.",
            None,
            None,
            None,
        ),
    )
    .await?;

    let payload = &job.payload;
    let source_asset_id = payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Upscale jobs require a source image asset.".to_owned())
        })?
        .to_owned();
    // Reject an unsupported factor early rather than silently coercing it to 2x (F-118 / sc-8920,
    // mirroring the video path). A missing factor still defaults to 2x.
    let factor: u8 =
        resolve_image_upscale_factor(payload.get("factor").and_then(Value::as_u64).unwrap_or(2))?;
    let engine = crate::payload::trimmed_string(payload.get("engine"))
        .unwrap_or("real-esrgan")
        .to_lowercase();
    // Canonical engine id. The Rust upscaler serves Real-ESRGAN (`ort`: CoreML on Mac / CUDA off-Mac,
    // sc-3489 / sc-5499) and SeedVR2 (native MLX / candle). `aura-sr` was dropped on Mac (sc-3668) and
    // as an offered engine off-Mac (sc-5499). The routing oracle (`upscale_job_is_mlx_eligible` /
    // `upscale_job_is_candle_eligible`) already refuses anything else, so this match is a defensive guard.
    let engine_id = match engine.as_str() {
        "real-esrgan" | "realesrgan" | "real_esrgan" => "real-esrgan",
        "seedvr2" => "seedvr2",
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "This upscaler supports engine=real-esrgan or engine=seedvr2 (got {other}); aura-sr is dropped (sc-3668 / sc-5499)."
            )));
        }
    };
    // SeedVR2-only knobs (ignored by Real-ESRGAN): the `--softness` pre-blur (0..1) and a seed for
    // reproducible generative output. Both default to 0 and are read leniently from the payload.
    let softness = payload
        .get("softness")
        .and_then(Value::as_f64)
        .map(|v| v.clamp(0.0, 1.0) as f32)
        .unwrap_or(0.0);
    let seed = payload.get("seed").and_then(Value::as_u64).unwrap_or(0);
    let manifest_entry = payload
        .get("modelManifestEntry")
        .cloned()
        .unwrap_or(Value::Null);

    // resolve the source asset against its originating project
    let project_id = payload
        .get("projectId")
        .and_then(Value::as_str)
        .or(job.project_id.as_deref())
        .ok_or_else(|| WorkerError::InvalidPayload("Upscale jobs require a projectId.".to_owned()))?
        .to_owned();
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store
        .get_project(&project_id)
        .map_err(|e| WorkerError::InvalidPayload(format!("project not found: {e}")))?;
    let project_path = PathBuf::from(project.path);
    let source = resolve_asset_media(&store, &project_id, &source_asset_id, &project_path)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Source image asset not found or missing: {source_asset_id}."
            ))
        })?;
    let source_path = source.path;
    let source_display = source.display_name;

    // Decode the source off the async runtime thread (sc-8909 / F-107): the full read + decode is
    // blocking and would otherwise stall the heartbeat before the upscale even starts.
    // The source may carry native transparency (sc-24111). Neither upscale engine can be handed a
    // fourth channel — Real-ESRGAN's ONNX export is `(1, 3, H, W)` and SeedVR2 returns three — so
    // the alpha plane is split off here and re-attached at the write, resampled to the upscaled
    // geometry. `None` for an opaque source, which leaves that lane exactly as it was.
    let (source_image, source_alpha) = {
        let source_path = source_path.clone();
        crate::load_progress::spawn_blocking(move || {
            crate::image_decode::decode_image_any(&source_path)
                .map_err(|e| {
                    WorkerError::InvalidPayload(format!("Source image could not be loaded: {e}"))
                })
                .map(|decoded| crate::image_jobs::split_alpha(&decoded))
        })
        .await
        .map_err(|e| task_join_error("upscale source decode task", e))??
    };
    let (src_w, src_h) = (source_image.width(), source_image.height());

    // Real-ESRGAN reports its actual `ort` execution device (coreml/cuda/cpu) for observability;
    // SeedVR2 is a generative pass with no such session, so it reports `None` (sc-8923).
    let (upscaled, esrgan_device): (RgbImage, Option<&'static str>) = if engine_id == "seedvr2" {
        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Running,
                ProgressStage::Downloading,
                0.25,
                crate::load_progress::PREPARING_MODEL,
                None,
                None,
                None,
            ),
        )
        .await?;
        let dir = require_seedvr2_checkpoint_dir(settings)?;

        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Running,
                ProgressStage::Running,
                0.45,
                &format!("Upscaling {factor}x with SeedVR2."),
                None,
                None,
                None,
            ),
        )
        .await?;
        let cancel = CancelFlag::new();
        // Move `source_image` into the future — the `else` branch already consumes it and it's
        // unused after this `if`, so the clone (a full RGB buffer copy) was needless (sc-8927).
        let out = run_upscale_with_heartbeat(
            api,
            settings,
            &job.id,
            cancel.clone(),
            crate::load_progress::spawn(async move {
                run_seedvr2_upscale(dir, source_image, factor, softness, seed, cancel).await
            }),
        )
        .await?;
        (out, None)
    } else {
        // Real-ESRGAN via `ort` — the CoreML EP on macOS (sc-3489), the CUDA EP off-Mac on the
        // candle GPU-worker lane (sc-5499), both with a CPU fallback. The routing oracle
        // (`upscale_job_is_candle_eligible` / `upscale_job_is_mlx_eligible`) admits Real-ESRGAN on
        // both lanes; `engine=aura-sr` is refused before reaching here (defensive `engine_id` guard
        // above) and remains queued.
        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Running,
                ProgressStage::Downloading,
                0.25,
                crate::load_progress::PREPARING_MODEL,
                None,
                None,
                None,
            ),
        )
        .await?;
        let onnx_path = resolve_onnx(settings, factor, &manifest_entry)?;

        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Running,
                ProgressStage::Running,
                0.45,
                &format!("Upscaling {factor}x with Real-ESRGAN."),
                None,
                None,
                None,
            ),
        )
        .await?;
        let cancel = CancelFlag::new();
        let (out, device) = run_upscale_with_heartbeat(
            api,
            settings,
            &job.id,
            cancel.clone(),
            crate::load_progress::spawn_blocking(move || {
                upscale_blocking(onnx_path, factor, source_image, cancel)
            }),
        )
        .await?;
        (out, Some(device))
    };
    let (out_w, out_h) = (upscaled.width(), upscaled.height());

    let source_name = payload
        .get("displayName")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or(source_display)
        .unwrap_or_else(|| "Image".to_owned());
    let mut upscale_settings = json!({
        "enabled": true,
        "engine": engine_id,
        "factor": factor,
        "sourceWidth": src_w,
        "sourceHeight": src_h,
        "width": out_w,
        "height": out_h,
    });
    if engine_id == "seedvr2" {
        // SeedVR2 is a generative one-step upscaler: record the detail/softness knob and the seed so
        // the result is reproducible + the UI can surface what produced it.
        upscale_settings["softness"] = json!(softness);
        upscale_settings["seed"] = json!(seed);
    }
    if let Some(device) = esrgan_device {
        // Report the Real-ESRGAN `ort` session's actual execution device (coreml/cuda/cpu) so an
        // upscale result reveals whether the hardware EP ran or it fell back to CPU (sc-8923).
        upscale_settings["device"] = json!(device);
    }
    // The sanitized workflow this PNG carries (epic 15945, sc-15948). This is a DISTINCT job with
    // its own payload, so — per the story's rule for a derived pass — the envelope describes THIS
    // pass and inherits nothing from whatever generated the source image. `engine_id` and `factor`
    // are the APPLIED values (canonicalized and validated above), not the requested ones, and
    // `softness` only travels for the engine that has the knob.
    let workflow = crate::image_jobs::workflow_source(settings, &job.payload).and_then(|payload| {
        crate::image_jobs::standalone_upscale_workflow_share(
            &payload,
            engine_id,
            factor,
            (engine_id == "seedvr2").then_some(softness),
            seed as i64,
            src_w,
            src_h,
        )
    });
    // Re-attach the plane the engine could not see. `write_single_child_asset` keys its encode on
    // the variant it is handed, so an RGBA source lands as an RGBA PNG with its workflow chunk.
    let written = match crate::image_jobs::reattach_alpha(upscaled, source_alpha.as_ref()) {
        crate::image_jobs::GeneratedPixels::Rgb(rgb) => image::DynamicImage::ImageRgb8(rgb),
        crate::image_jobs::GeneratedPixels::Rgba(rgba) => image::DynamicImage::ImageRgba8(rgba),
    };
    let result = write_single_child_asset(
        &project_path,
        written,
        SingleChildAssetSpec {
            filename_stem: &format!("upscaled_x{factor}"),
            mode: "image_upscale",
            model: engine_id,
            adapter: engine_id,
            encode_label: "upscale encode task",
            workflow,
        },
        |write| {
            json!({
                "assetId": write.asset_id,
                "mediaPath": write.media_path,
                "mimeType": "image/png",
                "type": "image",
                "width": out_w,
                "height": out_h,
                "normalizedWidth": out_w,
                "normalizedHeight": out_h,
                "count": 1,
                "seed": seed,
                "displayName": format!("{source_name} ({factor}x upscaled)"),
                "createdAt": write.created_at,
                "mode": "image_upscale",
                "model": engine_id,
                "adapter": engine_id,
                "prompt": "",
                "negativePrompt": "",
                "loras": [],
                "stylePreset": "",
                "sourceAssetId": source_asset_id,
                "rawAdapterSettings": { "upscale": upscale_settings },
                "parents": [source_asset_id],
                "extra": {
                    "isUpscaled": true,
                    "upscaledFromAssetId": source_asset_id,
                    "factor": factor,
                    "engine": engine_id,
                },
            })
        },
    )
    .await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Image upscale complete.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Iris-3B restoration (sc-25683): the `iris` engine of the standalone image_upscale job and of the
// Image Studio inline post-processing pass. See `crate::iris_restore` for the plan / install /
// admission halves; this is the job wiring both surfaces share.
// ---------------------------------------------------------------------------

const IRIS_CANCEL_MESSAGE: &str = "Iris restoration canceled by user.";

/// Resolve the install, plan the request against the INSTALLED export's geometry, admit it, then run
/// it through the cached transform while forwarding per-tile progress (built by `progress_for`) and
/// keeping the heartbeat + cancel poll alive. Shared by the standalone job and the inline pass, so
/// both refuse, plan and progress identically. Returns no image on cancel or refusal.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn restore_with_progress(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    loader: crate::iris_restore::TransformLoader,
    source: RgbImage,
    params: sceneworks_core::image_restoration::IrisRestoreParams,
    trained_model: Option<String>,
    progress_for: impl Fn(u32, u32) -> sceneworks_core::contracts::ProgressRequest + Send + 'static,
) -> WorkerResult<crate::iris_restore::RestoreOutcome> {
    use crate::iris_restore::{
        admission_error, live_total_gb, plan_restore, resolve_restore_export_dir,
        restore_image_using, RestoreLane,
    };
    // A trained restoration model (sc-25688) replaces the release export; its task is checked and
    // it plans with its own export geometry.
    let (export_dir, resources) = match trained_model.as_deref() {
        Some(model) => {
            let dir = crate::iris_task_training::trained_task_export_dir(
                &settings.data_dir,
                model,
                gen_core_iris::IrisTask::Restoration,
            )
            .map_err(|detail| WorkerError::InvalidPayload(format!("Iris restoration: {detail}")))?;
            let resources = gen_core_iris::restoration::RestorationResources::from_dir(
                &dir,
                sceneworks_core::image_restoration::IRIS_RESTORE_MODEL_ID,
            )
            .map_err(|error| WorkerError::InvalidPayload(format!("Iris restoration: {error}")))?;
            (dir, resources)
        }
        None => resolve_restore_export_dir(settings)?,
    };
    let geometry = resources.settings.geometry(resources.config());
    let plan = plan_restore(source.width(), source.height(), &params, geometry)
        .map_err(WorkerError::InvalidPayload)?;
    if let Some(refusal) = admission_error(
        RestoreLane::this_build(),
        &plan,
        live_total_gb(settings).await,
    ) {
        return Err(refusal);
    }

    // Per-tile progress crosses from the transform cache thread to this task through a channel; the
    // forwarder posts it and is drained BEFORE this returns, so no tile update can land after the
    // caller's terminal post.
    let (tile_tx, mut tile_rx) = tokio::sync::mpsc::unbounded_channel::<(u32, u32)>();
    let forward_api = api.clone();
    let forward_job = job_id.to_owned();
    let cancel = CancelFlag::new();
    let task_cancel = cancel.clone();
    // Each per-tile post answers with the job snapshot; a user cancel recorded there trips the
    // engine's flag at once, so the restore stops before the NEXT tile instead of waiting for the
    // heartbeat's cancel poll (5-15 s — longer than a whole warm restore, which would otherwise
    // publish an asset the user had canceled).
    let forward_cancel = cancel.clone();
    let forwarder = crate::load_progress::spawn(async move {
        while let Some((current, total)) = tile_rx.recv().await {
            // A failed progress post must not abort the restore; the heartbeat path owns liveness.
            if let Ok(snapshot) =
                update_job(&forward_api, &forward_job, progress_for(current, total)).await
            {
                if snapshot.cancel_requested {
                    forward_cancel.cancel();
                }
            }
        }
    });
    let observed_cancel = cancel.clone();
    let outcome = crate::run_blocking_with_heartbeat(
        api,
        settings,
        job_id,
        Some(cancel),
        IRIS_CANCEL_MESSAGE,
        "Iris restoration task",
        Some(|| async move {
            crate::progress::update_job_cancelling(
                api,
                job_id,
                progress_payload(
                    JobStatus::Running,
                    ProgressStage::Running,
                    0.45,
                    "Canceling Iris restoration after the current tile.",
                    None,
                    None,
                    None,
                ),
            )
            .await
            .map(|_| ())
        }),
        crate::load_progress::spawn(async move {
            restore_image_using(
                loader,
                export_dir,
                plan,
                source,
                params,
                task_cancel,
                move |current, total| {
                    let _ = tile_tx.send((current, total));
                },
            )
            .await
        }),
    )
    .await;
    // The tile sender lives in the finished (or abandoned) transform job; once it is dropped the
    // forwarder drains and exits. Bound the wait so an abandoned job can never hang the worker.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(10), forwarder).await;
    // A cancel the engine's flag recorded after its last checkpoint still means the user canceled:
    // return no image. A cancel recorded on the job AFTER the last per-tile reply never reaches this
    // flag (the heartbeat poll has ended too) — each caller re-checks the job right before it writes
    // its asset, which closes that window.
    if outcome.is_ok() && observed_cancel.is_cancelled() {
        crate::progress::mark_job_canceled(api, job_id, IRIS_CANCEL_MESSAGE).await?;
        return Err(WorkerError::Canceled(IRIS_CANCEL_MESSAGE.to_owned()));
    }
    outcome
}

/// The payload keys of OTHER engines that the Iris engine does not honour. A standalone Iris job
/// that names one is refused by name (E3) — `factor` in particular, because Iris sizes its output by
/// `scale` and silently ignoring a factor would return a different size than the caller asked for.
const IRIS_REFUSED_KEYS: [&str; 3] = ["factor", "softness", "seed"];

/// The standalone `image_upscale` job with `engine: "iris"` (sc-25683): restore one source asset into
/// a NEW lineage-preserving asset (the source is never touched), from the task's own installed
/// `upscaler/` export. `loader` is the transform loader seam (production:
/// [`crate::iris_restore::registry_transform_loader`]; tests inject a stub).
pub(crate) async fn run_iris_restore_job_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    loader: crate::iris_restore::TransformLoader,
) -> WorkerResult<()> {
    use sceneworks_core::image_restoration::{parse_iris_restore_params, IRIS_RESTORE_ENGINE};
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    let payload = &job.payload;
    if let Some(key) = IRIS_REFUSED_KEYS
        .iter()
        .find(|key| payload.get(**key).is_some_and(|value| !value.is_null()))
    {
        return Err(WorkerError::InvalidPayload(format!(
            "`{key}` is not a control of the Iris restoration engine (it sizes its output with \
             `scale`, and is deterministic with no softness)."
        )));
    }
    let params = parse_iris_restore_params(payload).map_err(WorkerError::InvalidPayload)?;
    let trained_model = sceneworks_core::image_restoration::parse_iris_restore_model(payload)
        .map_err(WorkerError::InvalidPayload)?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.12,
            "Loading source image.",
            None,
            None,
            None,
        ),
    )
    .await?;

    let source_asset_id = payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Restoration jobs require a source image asset.".to_owned())
        })?
        .to_owned();
    let project_id = payload
        .get("projectId")
        .and_then(Value::as_str)
        .or(job.project_id.as_deref())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Restoration jobs require a projectId.".to_owned())
        })?
        .to_owned();
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store
        .get_project(&project_id)
        .map_err(|e| WorkerError::InvalidPayload(format!("project not found: {e}")))?;
    let project_path = PathBuf::from(project.path);
    let source = resolve_asset_media(&store, &project_id, &source_asset_id, &project_path)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Source image asset not found or missing: {source_asset_id}."
            ))
        })?;
    let source_path = source.path;
    let source_display = source.display_name;
    // Upright pixels first (upstream `exif_transpose`s before restoring), then split any alpha plane
    // off: the transform takes RGB and the plane is re-attached at the output geometry.
    let (source_image, source_alpha) = crate::load_progress::spawn_blocking(move || {
        crate::iris_restore::decode_oriented(&source_path)
            .map_err(|e| {
                WorkerError::InvalidPayload(format!("Source image could not be loaded: {e}"))
            })
            .map(|decoded| crate::image_jobs::split_alpha(&decoded))
    })
    .await
    .map_err(|e| task_join_error("restoration source decode task", e))??;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.3,
            "Restoring with Iris.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let outcome = restore_with_progress(
        api,
        settings,
        &job.id,
        loader,
        source_image,
        params,
        trained_model.clone(),
        |current, total| {
            progress_payload(
                JobStatus::Running,
                ProgressStage::Running,
                0.3 + 0.6 * f64::from(current) / f64::from(total.max(1)),
                &format!("Restoring tile {current}/{total} with Iris."),
                None,
                None,
                None,
            )
        },
    )
    .await?;
    let plan = outcome.plan;
    let (out_w, out_h) = (outcome.image.width(), outcome.image.height());
    let record = crate::iris_restore::applied_settings_record(&plan, trained_model.as_deref());

    let source_name = payload
        .get("displayName")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or(source_display)
        .unwrap_or_else(|| "Image".to_owned());
    let workflow = crate::image_jobs::workflow_source(settings, &job.payload).and_then(|payload| {
        crate::image_jobs::standalone_restore_workflow_share(
            &payload,
            &plan,
            trained_model.as_deref(),
        )
    });
    let written = match crate::image_jobs::reattach_alpha(outcome.image, source_alpha.as_ref()) {
        crate::image_jobs::GeneratedPixels::Rgb(rgb) => image::DynamicImage::ImageRgb8(rgb),
        crate::image_jobs::GeneratedPixels::Rgba(rgba) => image::DynamicImage::ImageRgba8(rgba),
    };
    // A trained restoration model (sc-25688) is the asset's model, like a trained depth model.
    let model = trained_model
        .as_deref()
        .unwrap_or(crate::iris_restore::IRIS_RESTORE_MODEL_ID);
    let display_name = format!(
        "{source_name} ({})",
        crate::iris_restore::display_suffix(&plan)
    );
    let plan_view = crate::iris_restore::RestorePlanView::from_plan(&plan);
    // A cancel requested after the last per-tile reply is seen by no flag; re-check the job before
    // publishing so a canceled restore never writes (and then completes with) an asset.
    crate::progress::check_cancel(api, &job.id, IRIS_CANCEL_MESSAGE).await?;
    let result = write_single_child_asset(
        &project_path,
        written,
        SingleChildAssetSpec {
            filename_stem: "restored_iris",
            mode: "image_upscale",
            model,
            adapter: IRIS_RESTORE_ENGINE,
            encode_label: "restoration encode task",
            workflow,
        },
        |write| {
            json!({
                "assetId": write.asset_id,
                "mediaPath": write.media_path,
                "mimeType": "image/png",
                "type": "image",
                "width": out_w,
                "height": out_h,
                "normalizedWidth": out_w,
                "normalizedHeight": out_h,
                "count": 1,
                "seed": 0,
                "displayName": display_name,
                "createdAt": write.created_at,
                "mode": "image_upscale",
                "model": model,
                "adapter": IRIS_RESTORE_ENGINE,
                "prompt": "",
                "negativePrompt": "",
                "loras": [],
                "stylePreset": "",
                "sourceAssetId": source_asset_id,
                "rawAdapterSettings": { "upscale": record },
                "parents": [source_asset_id],
                "extra": {
                    "isUpscaled": true,
                    "upscaledFromAssetId": source_asset_id,
                    "engine": IRIS_RESTORE_ENGINE,
                    "scale": plan.scale,
                    "restoration": plan_view,
                },
            })
        },
    )
    .await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Iris restoration complete.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Dataset Doctor one-tap upscale (sc-6539): Real-ESRGAN over flagged low-resolution items, then
// re-point each item at the upscaled bytes via the API. Reuses the engine above (resolve_onnx +
// upscale_blocking). The payload-parse and repoint-body build are pure (unit-tested); the only
// session-unverifiable leg is the literal upscale_blocking call.
// ---------------------------------------------------------------------------

/// One dataset item to upscale, parsed from the job payload.
#[derive(Debug, Clone, PartialEq)]
struct DatasetUpscaleItem {
    item_id: String,
    image_path: PathBuf,
}

/// Parse the `items` array of a `dataset_upscale` job into typed entries. Each entry needs a
/// non-empty `itemId` + `imagePath`; malformed entries are skipped. Every `imagePath` is confined
/// to the payload's app-managed `datasetRoot` via [`resolve_dataset_item_path`] (sc-8842 / F-040):
/// the `imagePath` is client-supplied and reaches an on-disk read, so an unconfined path would be an
/// arbitrary-file-read/exfiltration primitive — a path escaping the dataset root fails the job.
fn parse_dataset_upscale_items(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<Vec<DatasetUpscaleItem>> {
    parse_dataset_items(
        settings,
        payload,
        "Dataset upscale",
        DatasetItemPolicy::SkipMalformed,
        |item| {
            Ok(DatasetUpscaleItem {
                item_id: item.item_id.to_owned(),
                image_path: item.image_path()?,
            })
        },
    )
}

/// Build the `/repoint` request body from `(item_id, project-relative source path)` results (pure).
/// `assetId` is null: the dataset fix re-points to the upscaled bytes directly (the original library
/// asset/upload is left untouched), so no child library asset is minted.
fn dataset_repoint_body(records: &[(String, String)]) -> Value {
    json!({
        "items": records
            .iter()
            .map(|(item_id, source_path)| {
                json!({ "itemId": item_id, "assetId": Value::Null, "sourcePath": source_path })
            })
            .collect::<Vec<_>>(),
    })
}

pub(crate) async fn run_dataset_upscale_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    let payload = &job.payload;
    // Reject an unsupported factor early rather than silently coercing it to 2x (F-118 / sc-8920,
    // mirroring the video path). A missing factor still defaults to 2x.
    let factor: u8 =
        resolve_image_upscale_factor(payload.get("factor").and_then(Value::as_u64).unwrap_or(2))?;
    let project_id = payload
        .get("projectId")
        .and_then(Value::as_str)
        .or(job.project_id.as_deref())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Dataset upscale jobs require a projectId.".to_owned())
        })?
        .to_owned();
    let dataset_id = payload
        .get("datasetId")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Dataset upscale jobs require a datasetId.".to_owned())
        })?
        .to_owned();
    let items = parse_dataset_upscale_items(settings, payload)?;
    if items.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Dataset upscale job has no items to upscale.".to_owned(),
        ));
    }

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::LoadingModel,
            ProgressStage::LoadingModel,
            0.1,
            crate::load_progress::PREPARING_MODEL,
            None,
            None,
            None,
        ),
    )
    .await?;
    let onnx_path = resolve_onnx(settings, factor, &Value::Null)?;

    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store
        .get_project(&project_id)
        .map_err(|e| WorkerError::InvalidPayload(format!("project not found: {e}")))?;
    let project_path = PathBuf::from(project.path);
    let derived_dir_rel = format!("training/datasets/{dataset_id}/upscaled");

    let total = items.len();
    let mut records: Vec<(String, String)> = Vec::with_capacity(total);
    for (index, item) in items.iter().enumerate() {
        let fraction = 0.15 + 0.75 * (index as f64 / total as f64);
        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Running,
                ProgressStage::Running,
                fraction,
                &format!("Upscaling image {} of {total}.", index + 1),
                None,
                None,
                None,
            ),
        )
        .await?;

        let onnx = onnx_path.clone();
        let cancel = CancelFlag::new();
        let cancel_run = cancel.clone();
        // Decode the source inside the blocking task (sc-8909 / F-107) so the per-item read + decode
        // never stalls the async runtime thread across the dataset loop.
        let item_image_path = item.image_path.clone();
        // Per-item device is uniform across the batch (same cached session) and not recorded per
        // item, so it's discarded here (sc-8923).
        let (upscaled, _device) = run_upscale_with_heartbeat(
            api,
            settings,
            &job.id,
            cancel,
            crate::load_progress::spawn_blocking(move || {
                // Deliberately RGB, unlike the library-asset lane above (sc-24111). This writes a
                // TRAINING dataset derivative, and the trainers consume three channels — a
                // four-channel derivative would be a format the consumer cannot read. A dataset
                // item that was transparent is flattened here, by the dataset contract rather than
                // by an oversight; the user-facing asset it was imported from keeps its alpha.
                let source_image = crate::image_decode::decode_image_any(&item_image_path)
                    .map_err(|e| {
                        WorkerError::InvalidPayload(format!("Image could not be loaded: {e}"))
                    })?
                    .to_rgb8();
                upscale_blocking(onnx, factor, source_image, cancel_run)
            }),
        )
        .await?;

        // sc-8842 / F-040: `dataset_id` and `item_id` are client-supplied and here reach a
        // filesystem write (`save_with_format` overwrites), so route the project-relative output
        // through `safe_project_path` — a `/`, `..`, or absolute component is rejected before the
        // write instead of traversing out of the project tree.
        let rel = format!("{derived_dir_rel}/{}.png", item.item_id);
        let abs = safe_project_path(&project_path, &rel)?;
        if let Some(parent) = abs.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        // Encode off the async runtime thread (sc-8909 / F-107).
        crate::load_progress::spawn_blocking(move || {
            upscaled
                .save_with_format(&abs, image::ImageFormat::Png)
                .map_err(|e| WorkerError::Io(std::io::Error::other(e)))
        })
        .await
        .map_err(|e| task_join_error("dataset upscale encode task", e))??;
        records.push((item.item_id.clone(), rel));
    }

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.94,
            "Updating dataset.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let _: Value = api
        .post_json(
            &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/repoint"),
            &dataset_repoint_body(&records),
        )
        .await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!("Upscaled {total} image(s)."),
            None,
            None,
            None,
        ),
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests;
