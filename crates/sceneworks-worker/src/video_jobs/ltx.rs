#[allow(unused_imports)]
use super::prelude::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use super::vace::{
    build_vace_conditioning, load_source_video_frames, replacement_mode_from,
    replacement_status_value, resolve_character_references, FRAME_PAD_COLOR,
};
#[cfg(target_os = "macos")]
use super::wan::{generate_video, VideoGenInput};

// ---------------------------------------------------------------------------
// Real MLX LTX-2.3 generation (macOS, via mlx-gen-ltx, sc-3035): T2V/I2V with
// SYNCHRONIZED AUDIO (the 2-stage distilled A/V pipeline; CFG forced 1.0). One
// engine model `ltx_2_3` serves both `ltx_2_3` + `ltx_2_3_eros` (the checkpoint dir
// selects quant via split_model.json). The Gemma-3 text encoder is resolved by the
// engine ($LTX_GEMMA_DIR / the HF cache). Audio rides the sc-3033 WAV→AAC mux path.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX LTX asset.
#[cfg(target_os = "macos")]
pub(super) const LTX_ADAPTER: &str = "mlx_ltx";

/// SceneWorks LTX model id → mlx-gen registry id (one engine model serves both), or
/// `None` if not an LTX family id.
#[cfg(target_os = "macos")]
pub(super) fn ltx_engine_id(model: &str) -> Option<&'static str> {
    matches!(model, "ltx_2_3" | "ltx_2_3_eros").then_some("ltx_2_3")
}

/// Whether the linked LTX engine can serve this request now (resolvable weights).
#[cfg(target_os = "macos")]
pub(super) fn ltx_available(request: &VideoRequest, settings: &Settings) -> bool {
    ltx_engine_id(&request.model).is_some() && resolve_ltx_model_dir(settings, request).is_ok()
}

/// The turnkey SceneWorks LTX-2.3 MLX bundle (sc-5608, epic 5594; replaces the third-party
/// `notapalindrome/ltx23-mlx-av-q4` + `mlx-community/gemma-3-12b-it-bf16` mirrors). One repo with
/// the LTX `q4/` (default) + `q8/` (opt-in) checkpoint subdirs — each the full audio+I2V component
/// set — plus the bundled `gemma/` text encoder the engine reads via `$LTX_GEMMA_DIR`.
#[cfg(target_os = "macos")]
pub(super) const LTX_BUNDLE_REPO: &str = "SceneWorks/ltx-2.3-mlx";
/// Pinned revision for the fixed [`LTX_BUNDLE_REPO`] (sc-9879, F-077 follow-up). The bundle repo is a
/// hard-coded const (no manifest/payload override reaches the on-demand `q8/*` + `bf16/*` fetches), so
/// pulling the mutable `main` branch would let an upstream re-push silently swap a checkpoint we load.
/// Pin the exact commit for defense-in-depth (mirrors the SeedVR2/Real-ESRGAN pins, sc-8879/sc-9682).
/// The native downloader still verifies each file's own hash on download. Bumped in sc-13870 to the
/// packed-q4 + Gemma revision validated by the candle training and inference round-trip.
#[cfg(target_os = "macos")]
pub(super) const LTX_BUNDLE_REVISION: &str = "254989c3ca7ee691187647f350b112c0c448789d";

/// Whether `dir` is a converted LTX snapshot **complete for the current engine** — it must
/// carry the audio `vocoder` + I2V `vae_encoder` + single `upsampler`/`vae_decoder` the
/// engine `load()` reads. Older conversions (`spatial_/temporal_upscaler_*`, no vocoder)
/// fail this, so a stale local dir is skipped in favour of the turnkey snapshot.
#[cfg(target_os = "macos")]
pub(super) fn ltx_dir_is_complete(dir: &Path) -> bool {
    [
        "connector.safetensors",
        "transformer.safetensors",
        "upsampler.safetensors",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
        "audio_vae.safetensors",
        "vocoder.safetensors",
    ]
    .iter()
    .all(|file| dir.join(file).is_file())
}

/// Whether `dir` is a complete Gemma-3 text-encoder snapshot the LTX engine can load: parseable,
/// non-empty config and tokenizer JSON, plus a structurally valid single safetensors file or every
/// structurally valid, safely-relative shard mapped by a non-empty index. The API readiness gate uses
/// this same core predicate, so it cannot admit a snapshot the worker rejects. Used so runtime option
/// discovery and eros provisioning never accept filename-only placeholders, path escapes, or
/// half-downloaded snapshots.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn ltx_gemma_dir_is_complete(dir: &Path) -> bool {
    sceneworks_core::safetensors::gemma_text_encoder_dir_is_complete(dir)
}

/// Parse `advanced.mlxQuantize` (int or numeric string) → the requested bit width, if present.
#[cfg(target_os = "macos")]
fn ltx_quant_bits(request: &VideoRequest) -> Option<i64> {
    request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
}

/// Whether the request opts into the higher-quality Q8 LTX checkpoint (`advanced.mlxQuantize: 8`,
/// accepted as int or string). The default is Q4 (sc-5608).
#[cfg(target_os = "macos")]
pub(super) fn ltx_wants_q8(request: &VideoRequest) -> bool {
    ltx_quant_bits(request)
        .map(|bits| bits >= 8)
        .unwrap_or(false)
}

/// Whether the request opts into the dense **bf16** LTX checkpoint (`advanced.mlxQuantize <= 0`,
/// int or string) — the ~47 GB power-user tier (sc-8513, epic 8506). Never the default: absent ⇒ Q4,
/// so the big bf16 bundle is a deliberate opt-in (mirrors [`resolve_mlx_dense_quant`]'s `<= 0` rule).
#[cfg(target_os = "macos")]
pub(super) fn ltx_wants_bf16(request: &VideoRequest) -> bool {
    ltx_quant_bits(request)
        .map(|bits| bits <= 0)
        .unwrap_or(false)
}

/// The SceneWorks LTX bundle tier search order for a request — preferred tier first, then the
/// always-smaller fallback tiers so a bundle missing the preferred subdir still loads (sc-8513):
/// `mlxQuantize <= 0` ⇒ `bf16`, `>= 8` ⇒ `q8`, and — with an explicit `1..=4` OR NO explicit
/// `mlxQuantize` — the **`q4`** default (q4-first). bf16 stays OUT of the default order, so a default
/// job never loads the huge dense tier by accident.
///
/// The video lane keeps the pre-sc-10726 q4-first default (sc-10859), NOT epic 10721's app-wide Q8 —
/// see [`wan_tier_order`] for the rationale (no MLX video Q8 lever ⇒ a silent Q8 only risks a
/// video-runtime OOM). The no-explicit-pick and explicit-`q4` cases now share one q4-first arm.
#[cfg(target_os = "macos")]
pub(super) fn ltx_bundle_tier_order(request: &VideoRequest) -> &'static [&'static str] {
    if ltx_wants_bf16(request) {
        &["bf16", "q8", "q4"]
    } else if ltx_wants_q8(request) {
        &["q8", "q4"]
    } else {
        // No explicit pick OR an explicit `1..=4` ⇒ q4-first (sc-10859 video carve-out).
        &["q4", "q8"]
    }
}

/// Pick the engine-complete `bf16/`/`q8/`/`q4/` checkpoint subdir of a SceneWorks LTX bundle `root`,
/// trying `order` (preferred tier first, sc-5608/sc-8513). Returns the first **complete**
/// ([`ltx_dir_is_complete`]) subdir — so a partially-downloaded bundle falls through rather than
/// half-loading — or `None`.
#[cfg(target_os = "macos")]
pub(super) fn ltx_bundle_subdir(root: &Path, order: &[&str]) -> Option<PathBuf> {
    order
        .iter()
        .map(|sub| root.join(sub))
        .find(|dir| ltx_dir_is_complete(dir))
}

/// Resolve the converted LTX MLX snapshot dir. Env override (`SCENEWORKS_MLX_LTX_DIR` /
/// `…_EROS_DIR`) → `<data>/models/mlx/<candidate>` → (base only) the turnkey SceneWorks bundle
/// [`LTX_BUNDLE_REPO`], descending into its `q4/`/`q8/` subdir. Only a dir **complete for the
/// current engine** ([`ltx_dir_is_complete`]) counts, so a stale local conversion is skipped. For
/// the base model the Q4 checkpoint is the default (`mlxQuantize: 8` prefers the Q8 one); the engine
/// reads the actual bits from `split_model.json`, so this only picks *which* dir to load.
#[cfg(target_os = "macos")]
pub(super) fn resolve_ltx_model_dir(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<PathBuf> {
    let eros = request.model == "ltx_2_3_eros";
    let env = if eros {
        "SCENEWORKS_MLX_LTX_EROS_DIR"
    } else {
        "SCENEWORKS_MLX_LTX_DIR"
    };
    if let Ok(override_dir) = std::env::var(env) {
        let path = PathBuf::from(override_dir.trim());
        if ltx_dir_is_complete(&path) {
            return Ok(path);
        }
    }
    let wants_bf16 = ltx_wants_bf16(request);
    let wants_q8 = ltx_wants_q8(request);
    let candidates: &[&str] = if eros {
        &["ltx_2_3_eros"]
    } else if wants_bf16 {
        // No local bf16 conversion id exists (install-time convert only emits Q4/Q8), so don't let a
        // local quantized dir shadow the dense turnkey tier — fall straight through to the bundle's
        // bf16/ subdir below.
        &[]
    } else if wants_q8 {
        &["ltx_2_3_base_q8", "ltx_2_3_base_q4", "ltx_2_3"]
    } else {
        &["ltx_2_3_base_q4", "ltx_2_3_base_q8", "ltx_2_3"]
    };
    for id in candidates {
        let dir = settings.data_dir.join("models").join("mlx").join(id);
        if ltx_dir_is_complete(&dir) {
            return Ok(dir);
        }
    }
    // Turnkey SceneWorks bundle for the base model (sc-5608): one repo with `bf16/` + `q8/` + `q4/`
    // LTX subdirs (+ a bundled `gemma/` the engine reads via $LTX_GEMMA_DIR). Pick the preferred tier
    // subdir; the engine reads the actual bits from split_model.json, so this only selects which to
    // load.
    if !eros {
        if let Some(root) = huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO) {
            if let Some(dir) = ltx_bundle_subdir(&root, ltx_bundle_tier_order(request)) {
                return Ok(dir);
            }
        }
    }
    Err(WorkerError::InvalidPayload(format!(
        "{}: no complete converted LTX MLX weights found under {} (expected one of {candidates:?} \
         with the audio vocoder + i2v vae_encoder; or the turnkey {LTX_BUNDLE_REPO} q4/ or q8/ \
         subdir; or set ${env})",
        request.model,
        settings.data_dir.join("models").join("mlx").display(),
    )))
}

/// The complete Gemma-3 text encoder managed with a resolved LTX dir, if present (sc-5608).
///
/// Normally the SceneWorks bundle ships it beside the selected `q4/`/`q8/` checkpoint dir as
/// `<snapshot>/gemma`. Hugging Face can retain different filtered downloads in different snapshot
/// revisions, though: a tier may resolve from `snapshots/<tier-rev>/q8` while the co-requisite lives
/// at `snapshots/<gemma-rev>/gemma`. After checking the selected snapshot, scan sibling revisions so
/// that valid managed layout still resolves one `LoadSpec::text_encoder` (sc-14377). A local/legacy
/// conversion is not under a `snapshots/` directory and therefore keeps the old sibling-only
/// behavior.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn bundled_ltx_gemma_dir(model_dir: &Path) -> Option<PathBuf> {
    let selected_snapshot = model_dir.parent()?;
    let gemma = selected_snapshot.join("gemma");
    if ltx_gemma_dir_is_complete(&gemma) {
        return Some(gemma);
    }

    let snapshots = selected_snapshot.parent()?;
    if snapshots.file_name().and_then(|name| name.to_str()) != Some("snapshots") {
        return None;
    }

    let mut revisions: Vec<PathBuf> = std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path != selected_snapshot)
        .collect();
    revisions.sort();
    revisions
        .into_iter()
        .map(|snapshot| snapshot.join("gemma"))
        .find(|candidate| ltx_gemma_dir_is_complete(candidate))
}

/// The operator `$LTX_GEMMA_DIR` override as an existence-checked gemma snapshot path (pure over the
/// raw env value so it is unit-testable without mutating the process-global env). As of sc-13664 the MLX
/// LTX provider no longer reads this env itself — `LoadSpec::text_encoder` is the ONLY (now required) TE
/// source — so the worker must resolve the override HERE and thread it onto the spec, rather than
/// returning `None` and deferring to the deleted `$LTX_GEMMA_DIR` / HF-cache fallback. A set-but-incomplete
/// override yields `None` so a good bundled / cache gemma still wins (a bad override never shadows it).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn ltx_gemma_override_path(raw: Option<std::ffi::OsString>) -> Option<PathBuf> {
    let dir = PathBuf::from(raw?);
    ltx_gemma_dir_is_complete(&dir).then_some(dir)
}

/// [`ltx_gemma_override_path`] applied to the live `$LTX_GEMMA_DIR`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn ltx_gemma_env_override() -> Option<PathBuf> {
    ltx_gemma_override_path(std::env::var_os("LTX_GEMMA_DIR"))
}

/// Resolve the Gemma-3 text encoder bundled beside the resolved LTX dir (sc-5608), returning it so the
/// caller can thread it onto `LoadSpec::text_encoder` (sc-8827) — a fresh install is self-contained (no
/// separate `mlx-community/gemma` download) without mutating the process-global `$LTX_GEMMA_DIR` at job
/// time on the multithreaded runtime (the old `set_var` seam was unsound, F-025). Resolution order: an
/// operator `$LTX_GEMMA_DIR` override (existence-checked, [`ltx_gemma_env_override`]) — threaded onto the
/// spec because the MLX provider dropped its own env read (sc-13664; the spec override already wins,
/// sc-8827) — then the bundled `<parent>/gemma` sibling ([`bundled_ltx_gemma_dir`]). `None` only when
/// neither resolves (a legacy local conversion with no co-located gemma), where the load now surfaces the
/// provider's required-`LoadSpec::text_encoder` error rather than a silent HF-cache scan.
///
/// `pub(crate)` so the LoRA trainer path reuses it (sc-9989): training resolves the TE identically to
/// inference, so a self-contained install trains without a separate `mlx-community/gemma` download.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn resolve_bundled_ltx_gemma_dir(model_dir: &Path) -> Option<PathBuf> {
    ltx_gemma_env_override().or_else(|| bundled_ltx_gemma_dir(model_dir))
}

/// Resolve the Gemma-3 text encoder for an **eros** generation. Unlike the base model — whose turnkey
/// bundle ships `gemma/` beside its checkpoint ([`resolve_bundled_ltx_gemma_dir`]) — the eros install
/// is a bare local conversion under `models/mlx/ltx_2_3_eros/` with no bundled TE, so gemma is
/// provisioned separately ([`ensure_ltx_gemma_present`]) and resolved here: a `models/mlx/gemma`
/// sibling of the checkpoint (the `<parent>/gemma` convention [`bundled_ltx_gemma_dir`] already uses)
/// first, then the fetched [`LTX_BUNDLE_REPO`] snapshot's `gemma/`. Resolution order: an operator
/// `$LTX_GEMMA_DIR` override (existence-checked, [`ltx_gemma_env_override`]) — threaded onto the spec
/// because the MLX provider dropped its own env read (sc-13664) — then the complete `<parent>/gemma`
/// sibling, then the bundle snapshot's `gemma/`. `None` only when nothing complete is on disk (the
/// provider then surfaces its required-`LoadSpec::text_encoder` error) — a partial dir never wins.
#[cfg(target_os = "macos")]
pub(super) fn resolve_ltx_eros_gemma_dir(settings: &Settings, model_dir: &Path) -> Option<PathBuf> {
    if let Some(env_dir) = ltx_gemma_env_override() {
        // Operator override rides the spec so it survives the sc-13664 provider env-read deletion.
        return Some(env_dir);
    }
    if let Some(sibling) = bundled_ltx_gemma_dir(model_dir) {
        if ltx_gemma_dir_is_complete(&sibling) {
            return Some(sibling);
        }
    }
    let bundle_gemma = huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO)?.join("gemma");
    ltx_gemma_dir_is_complete(&bundle_gemma).then_some(bundle_gemma)
}

/// The amoral 4-bit Gemma prompt-enhancer snapshot the OPT-IN `useUncensoredEnhancer` path loads
/// (sc-2845; the reference `--use-uncensored-enhancer`). A standalone mlx_lm checkpoint, distinct from
/// the always-on Gemma TE. Not a first-class SceneWorks download (it is an uncataloged power-user
/// opt-in — a manifest catalog entry with a pinned revision + license posture is a product decision,
/// tracked separately); the worker resolves it only if the operator has staged it.
#[cfg(target_os = "macos")]
const LTX_UNCENSORED_GEMMA_REPO: &str = "TheCluster/amoral-gemma-3-12B-v2-mlx-4bit";

/// Stable request selector for the shipped Gemma text-encoder backbone. The field is omitted by the
/// web client for this default, so old jobs and recipes retain identical behavior.
#[cfg(target_os = "macos")]
pub const DEFAULT_TEXT_ENCODER_ID: &str = "default";
/// Stable request selector for LTX's operator-staged, alternate prompt-enhancement Gemma.
#[cfg(target_os = "macos")]
pub const AMORAL_TEXT_ENCODER_ID: &str = "ltx_amoral_gemma_3_12b";

/// A selectable prompt text encoder surfaced by the model catalog. This is capability metadata only:
/// it contains no repo, revision, files, or download action, so operator-staged models remain outside
/// SceneWorks' distribution catalog.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TextEncoderOption {
    pub id: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    pub is_default: bool,
}

#[cfg(target_os = "macos")]
pub(super) fn ltx_text_encoder_options(alternate_staged: bool) -> Vec<TextEncoderOption> {
    let mut options = vec![TextEncoderOption {
        id: DEFAULT_TEXT_ENCODER_ID,
        label: "Shipped Gemma 3 12B (default)",
        description: "Uses the Gemma text encoder installed with LTX.",
        is_default: true,
    }];
    if alternate_staged {
        options.push(TextEncoderOption {
            id: AMORAL_TEXT_ENCODER_ID,
            label: "Amoral Gemma 3 12B (operator staged)",
            description:
                "Uses the complete alternate Gemma snapshot already staged by the operator.",
            is_default: false,
        });
    }
    options
}

/// Enumerate the complete text encoders this runtime can offer for a model adapter. The UI consumes
/// this generic list instead of hard-coding LTX ids. LTX is currently the only provider with a
/// swappable encoder, and that provider is MLX-only; future adapters can extend this registry without
/// changing Video Studio.
pub fn text_encoder_options_for_adapter(adapter: &str, data_dir: &Path) -> Vec<TextEncoderOption> {
    #[cfg(target_os = "macos")]
    {
        if adapter == "ltx_video" {
            return ltx_text_encoder_options(
                resolve_ltx_uncensored_enhancer_dir(data_dir).is_some(),
            );
        }
    }
    let _ = (adapter, data_dir);
    Vec::new()
}

/// Resolve the optional amoral 4-bit Gemma **enhancer** snapshot for a `useUncensoredEnhancer` LTX job
/// (sc-2845), to be staged in `LoadSpec::components["uncensored_enhancer"]`. Mirrors the resolution the
/// MLX LTX provider deleted at sc-13664 (moved into the worker): an operator `$LTX_UNCENSORED_GEMMA_DIR`
/// override (existence-checked) wins, else the HF-cache snapshot for [`LTX_UNCENSORED_GEMMA_REPO`] if
/// already on disk. `None` when neither resolves — the opt-in enhancer then surfaces the provider's
/// actionable "provision the amoral snapshot" error rather than silently degrading. Reuses
/// [`ltx_gemma_dir_is_complete`] (an mlx_lm gemma snapshot is `config.json` + shards, same shape as the
/// TE), so a partial/half-downloaded dir never wins.
#[cfg(target_os = "macos")]
fn resolve_ltx_uncensored_enhancer_dir(data_dir: &Path) -> Option<PathBuf> {
    if let Some(raw) = std::env::var_os("LTX_UNCENSORED_GEMMA_DIR") {
        let dir = PathBuf::from(raw);
        if ltx_gemma_dir_is_complete(&dir) {
            return Some(dir);
        }
    }
    let snapshot = huggingface_snapshot_dir(data_dir, LTX_UNCENSORED_GEMMA_REPO)?;
    ltx_gemma_dir_is_complete(&snapshot).then_some(snapshot)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LtxTextEncoderSelection {
    Default,
    Amoral,
}

#[cfg(target_os = "macos")]
pub(super) fn selected_ltx_text_encoder(
    advanced: &JsonObject,
) -> WorkerResult<LtxTextEncoderSelection> {
    let legacy = match advanced.get("useUncensoredEnhancer") {
        None => None,
        Some(Value::Bool(value)) => Some(*value),
        Some(_) => {
            return Err(WorkerError::InvalidPayload(
                "advanced.useUncensoredEnhancer must be a boolean when present".to_owned(),
            ));
        }
    };
    let selected = match advanced.get("textEncoderModel") {
        None => {
            return Ok(if legacy == Some(true) {
                LtxTextEncoderSelection::Amoral
            } else {
                LtxTextEncoderSelection::Default
            });
        }
        Some(Value::String(id)) if id == DEFAULT_TEXT_ENCODER_ID => {
            LtxTextEncoderSelection::Default
        }
        Some(Value::String(id)) if id == AMORAL_TEXT_ENCODER_ID => LtxTextEncoderSelection::Amoral,
        Some(Value::String(id)) => Err(WorkerError::InvalidPayload(format!(
            "unsupported LTX text encoder {id:?}; choose {DEFAULT_TEXT_ENCODER_ID:?} or an \
             installed option returned by GET /api/v1/models"
        )))?,
        Some(_) => {
            return Err(WorkerError::InvalidPayload(
                "advanced.textEncoderModel must be a string option id returned by GET /api/v1/models"
                    .to_owned(),
            ));
        }
    };
    if legacy.is_some_and(|legacy| legacy != (selected == LtxTextEncoderSelection::Amoral)) {
        return Err(WorkerError::InvalidPayload(format!(
            "advanced.textEncoderModel conflicts with advanced.useUncensoredEnhancer={}",
            legacy.unwrap_or_default()
        )));
    }
    Ok(selected)
}

#[cfg(target_os = "macos")]
pub(super) fn resolve_selected_ltx_text_encoder(
    advanced: &JsonObject,
    staged_alternate: Option<PathBuf>,
) -> WorkerResult<(bool, Option<PathBuf>)> {
    let selection = selected_ltx_text_encoder(advanced)?;
    let use_alternate = selection == LtxTextEncoderSelection::Amoral;
    if use_alternate && !advanced::bool(advanced, "enhancePrompt") {
        return Err(WorkerError::InvalidPayload(format!(
            "text encoder {AMORAL_TEXT_ENCODER_ID:?} is selected but prompt enhancement is off; \
             enable advanced.enhancePrompt or choose the default encoder"
        )));
    }
    let dir = if use_alternate {
        Some(staged_alternate.ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "text encoder {AMORAL_TEXT_ENCODER_ID:?} is selected but its complete weights are \
                 not staged; set $LTX_UNCENSORED_GEMMA_DIR to a complete snapshot or pre-stage \
                 TheCluster/amoral-gemma-3-12B-v2-mlx-4bit in the worker's Hugging Face cache"
            ))
        })?)
    } else {
        None
    };
    Ok((use_alternate, dir))
}

/// On-demand fetch of the bundle's `q8/` subdir (sc-5679). The macOS default download is lean
/// (`q4/` + `gemma/`); when a job opts into Q8 ([`ltx_wants_q8`]) and the bundle's `q8/` isn't already
/// complete, pull just `q8/*` from [`LTX_BUNDLE_REPO`] into the HF cache so [`resolve_ltx_model_dir`]
/// can load it. Base model only (eros has its own single-dir conversion). No-op when Q8 isn't
/// requested, the bundle snapshot isn't downloaded yet (resolve surfaces the clear "download the
/// bundle" error), or `q8/` is already present. Fails loud on a real download error — fast, before
/// any compute; a `q8/` tier that isn't published yet stays absent so resolve falls back to Q4.
/// Mirrors the eros [`ensure_ltx_upscaler_cached`] on-demand fetch.
#[cfg(target_os = "macos")]
pub(super) async fn ensure_ltx_q8_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    if request.model == "ltx_2_3_eros" || !ltx_wants_q8(request) {
        return Ok(());
    }
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO) else {
        return Ok(());
    };
    if ltx_dir_is_complete(&root.join("q8")) {
        return Ok(());
    }
    let files = vec!["q8/*".to_owned()];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        LTX_BUNDLE_REPO,
        LTX_BUNDLE_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// Fetch the SceneWorks LTX bundle's dense `bf16/` subdir on demand (sc-8513, epic 8506). The macOS
/// default download is lean (`q4/` + `gemma/`); a bf16 job ([`ltx_wants_bf16`]) pulls the ~47 GB
/// `bf16/*` from the FIXED [`LTX_BUNDLE_REVISION`] the first time it is requested. No-op for eros, for
/// non-bf16 jobs, or when `bf16/` is already complete. Mirrors [`ensure_ltx_q8_present`].
#[cfg(target_os = "macos")]
async fn ensure_ltx_bf16_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    if request.model == "ltx_2_3_eros" || !ltx_wants_bf16(request) {
        return Ok(());
    }
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO) else {
        return Ok(());
    };
    if ltx_dir_is_complete(&root.join("bf16")) {
        return Ok(());
    }
    let files = vec!["bf16/*".to_owned()];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        LTX_BUNDLE_REPO,
        LTX_BUNDLE_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// Ensure the Gemma-3 text encoder an **eros** generation needs is on disk (the eros gate over
/// [`ensure_ltx_bundle_gemma_present`]). No-op for the base model, which bundles gemma with its
/// turnkey checkpoint. Called just before resolving the LTX text encoder so an eros job that was
/// installed before install-time provisioning existed still self-heals on first generation.
#[cfg(target_os = "macos")]
async fn ensure_ltx_gemma_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    if request.model != "ltx_2_3_eros" {
        return Ok(());
    }
    ensure_ltx_bundle_gemma_present(api, settings, job).await
}

/// Ensure the eros Gemma-3 text encoder is on disk, fetching the bundle's `gemma/` on demand. The
/// eros install produces a bare converted checkpoint under `models/mlx/ltx_2_3_eros/` with no bundled
/// TE (unlike the base turnkey bundle, which ships `gemma/` beside its `q4/`), so without this an
/// eros generation dead-ends on "gemma snapshot not found". Pulls just `gemma/*` (~24 GB) from the
/// FIXED [`LTX_BUNDLE_REVISION`] — the same SceneWorks re-host the base model uses, so no separate
/// `mlx-community/gemma-3-12b-it-bf16` download. No-op when an operator `$LTX_GEMMA_DIR` is set, when
/// a local `models/mlx/gemma` sibling is already complete, or when the bundle snapshot's `gemma/` is
/// already complete. `pub(crate)` so the eros convert job provisions gemma at install time
/// ([`crate::model_jobs::run_model_convert_job`]) — the generation path is the self-healing backstop.
/// Mirrors [`ensure_ltx_q8_present`].
#[cfg(target_os = "macos")]
pub(crate) async fn ensure_ltx_bundle_gemma_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    if std::env::var_os("LTX_GEMMA_DIR").is_some() {
        return Ok(());
    }
    // A complete `<data>/models/mlx/gemma` sibling already satisfies the eros resolver — nothing to
    // fetch (also short-circuits an operator who provisioned gemma there by hand).
    let local_sibling = settings.data_dir.join("models").join("mlx").join("gemma");
    if ltx_gemma_dir_is_complete(&local_sibling) {
        return Ok(());
    }
    if let Some(root) = huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO) {
        if ltx_gemma_dir_is_complete(&root.join("gemma")) {
            return Ok(());
        }
    }
    let files = vec!["gemma/*".to_owned()];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        LTX_BUNDLE_REPO,
        LTX_BUNDLE_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// LoRAs for an LTX generation: the manifest-declared auto distill LoRA (when present) followed by
/// the user LoRAs (sc-3035). A model that declares `mlx.autoDistillLora` (10Eros) is NOT
/// pre-distilled, so its distill LoRA must be injected at runtime with per-pass strengths or its
/// video degrades to noise — see [`resolve_ltx_distill_adapter`]. Every user LoRA applies at a
/// uniform per-pass strength (`pass_scales` left `None` → the engine uses `scale` on every distilled
/// stage). peft LoKr allowed (engine residual), LyCORIS rejected.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_ltx_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
    let mut specs = Vec::with_capacity(request.loras.len() + 1);
    // The auto distill LoRA is the model's base recipe (per-pass 1.0/0.4); user LoRAs stack on top.
    if let Some(distill) = resolve_ltx_distill_adapter(settings, request)? {
        specs.push(distill);
    }
    specs.extend(resolve_ltx_user_adapters(settings, request)?);
    Ok(specs)
}

/// Resolve user-selected LTX video LoRAs for both native providers. MLX and Candle consume the same
/// PEFT attention-projection files and strengths. Candle has one distilled denoise pass, so this
/// shared resolver deliberately excludes the MLX-only two-pass 10Eros distill recipe above.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_ltx_user_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let mut specs = Vec::with_capacity(request.loras.len());
    for lora in &request.loras {
        let path = lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        let file = resolve_lora_file(
            settings,
            path,
            crate::image_jobs::declared_adapter_file(lora),
        )?;
        let kind = classify_adapter(&file)?;
        specs.push(AdapterSpec::new(file, lora_scale(lora), kind));
    }
    Ok(specs)
}

/// The auto-injected per-pass distill LoRA for an LTX model that declares `mlx.autoDistillLora`
/// (`ltx_2_3_eros` today), or `None` when the model declares none or the user opted out via
/// `advanced.useDistillLora = false`. 10Eros's base checkpoint is not pre-distilled, so without this
/// LoRA its MLX video collapses to noise a few frames in (the manifest documents that exact symptom).
///
/// The LoRA is the `coRequisite` `resources.distilledLora` (the cond_safe variant, sc-9696), so it
/// installs alongside the checkpoint and is resolved from the HF cache here. Strengths come from
/// `mlx.autoDistillLora` (`stage1Strength` full first pass / `stage2Strength` reduced spatial-upscale
/// pass — TenStrip's guidance for rank<=72 cond_safe LoRAs), overridable via
/// `advanced.distillStage1Strength` / `distillStage2Strength`, and applied as
/// `pass_scales = [stage1, stage2]` (the engine's LTX per-pass feature, sc-2687). Declared-but-missing
/// fails with an actionable error rather than silently producing noise.
///
/// Ported from the deleted Python `MlxVideoAdapter` (b821d74e): the injection was lost when video
/// generation moved to the Rust worker in the sc-3037 cutover, which is why 10Eros regressed to noise.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_ltx_distill_adapter(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Option<AdapterSpec>> {
    let Some(auto) = request
        .model_manifest_entry
        .get("autoDistillLora")
        .or_else(|| {
            request
                .model_manifest_entry
                .get("mlx")
                .and_then(Value::as_object)
                .and_then(|mlx| mlx.get("autoDistillLora"))
        })
        .and_then(Value::as_object)
    else {
        return Ok(None);
    };
    // Opt-out knob (default on): the distill LoRA is a required runtime component for these models.
    let enabled = request
        .advanced
        .get("useDistillLora")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if !enabled {
        return Ok(None);
    }
    let stage1 = advanced::f32(
        &request.advanced,
        "distillStage1Strength",
        auto.get("stage1Strength")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32,
    );
    let stage2 = advanced::f32(
        &request.advanced,
        "distillStage2Strength",
        auto.get("stage2Strength")
            .and_then(Value::as_f64)
            .unwrap_or(0.4) as f32,
    );
    // The distill LoRA repo/file live in `resources.distilledLora` (the recommended cond_safe variant).
    let (repo, file) = request
        .model_manifest_entry
        .get("resources")
        .and_then(Value::as_object)
        .and_then(|res| res.get("distilledLora"))
        .and_then(Value::as_object)
        .and_then(|d| {
            Some((
                d.get("repo").and_then(Value::as_str)?,
                d.get("file").and_then(Value::as_str)?,
            ))
        })
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "This model declares mlx.autoDistillLora but resources.distilledLora is missing its \
                 repo/file, so the distill LoRA cannot be resolved."
                    .to_owned(),
            )
        })?;
    // The LoRA is a download co-requisite (sc-9696), so it is expected in the HF cache. Fail with an
    // actionable message if it is absent rather than silently degrading the output to noise.
    let path = huggingface_snapshot_dir(&settings.data_dir, repo)
        .map(|dir| dir.join(file))
        .filter(|candidate| candidate.is_file())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "The required distill LoRA for this model is not installed ({repo}/{file}). \
                 Re-download the model to fetch its co-requisite distill LoRA."
            ))
        })?;
    let kind = classify_adapter(&path)?;
    Ok(Some(
        AdapterSpec::new(path, stage1, kind).with_pass_scales(vec![stage1, stage2]),
    ))
}

/// Optional I2V conditioning for LTX: a `source_asset_id` → a single `Reference` image
/// (image→video); absent → pure text→video. `first_last_frame` → two `Keyframe`s (sc-3055).
/// (Audio is produced either way.)
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_ltx_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    if request.mode == "first_last_frame" {
        return resolve_keyframe_conditioning(settings, request, project_path);
    }
    if request.mode != "image_to_video" {
        return if request.source_asset_id.is_some() {
            Err(WorkerError::InvalidPayload(format!(
                "{} does not accept sourceAssetId on the {} mode.",
                request.model, request.mode
            )))
        } else {
            Ok(Vec::new())
        };
    }
    match request.source_asset_id.as_deref() {
        Some(asset_id) => {
            let image = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                asset_id,
                project_path,
            )?;
            // Pre-fit the starting image to the output W×H by the chosen crop/pad mode
            // (sc-6139) — without this the engine resizes it internally = stretch. Reuses
            // the image-edit lane's helper; a pre-fit-to-exact-dims reference is a no-op
            // for any further internal resize.
            let image = crate::image_jobs::fit_engine_image(
                image,
                request.width,
                request.height,
                &request.fit_mode,
            )?;
            Ok(vec![Conditioning::Reference {
                image,
                strength: None,
            }])
        }
        None => Err(WorkerError::InvalidPayload(
            "image_to_video requires a source image (sourceAssetId).".to_owned(),
        )),
    }
}

/// Read an `advanced` boolean flag (JSON bool), default `false` (Python `bool(.get(k))`).
/// First/last-frame conditioning (sc-3055 cutover): two [`Conditioning::Keyframe`]s — the source
/// image pinned at latent frame 0 and the last-frame image at latent frame `-1` (the engine's
/// Python-style negative-from-end index, so the worker needs no latent-frame math; the engine
/// bounds-checks it). Mirrors the torch `_ltx_conditioning_images` first_last_frame path: first @
/// `imageConditioningStrength`, last @ `lastFrameConditioningStrength` (both default 1.0 = fully
/// pinned). Shared by LTX (`ltx_2_3`) and Wan TI2V-5B (`wan_2_2`), the engines whose providers
/// advertise `Keyframe`. `imageFrameIndex` (default 0) is forwarded as the first keyframe's latent
/// index — for the universal FLF case (0) latent 0 == output 0.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_keyframe_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    let first_id = request.source_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "first_last_frame requires a source image (sourceAssetId).".to_owned(),
        )
    })?;
    let last_id = request.last_frame_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "first_last_frame requires a last-frame image (lastFrameAssetId).".to_owned(),
        )
    })?;
    let first = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        first_id,
        project_path,
    )?;
    let last = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        last_id,
        project_path,
    )?;
    // Fit both keyframes to the output W×H by the chosen crop/pad mode (sc-6139) so a
    // square first/last frame letterboxes (pad) or fills+trims (crop) into an off-aspect
    // clip instead of the engine stretching each internally.
    let first = crate::image_jobs::fit_engine_image(
        first,
        request.width,
        request.height,
        &request.fit_mode,
    )?;
    let last = crate::image_jobs::fit_engine_image(
        last,
        request.width,
        request.height,
        &request.fit_mode,
    )?;
    Ok(vec![
        Conditioning::Keyframe {
            image: first,
            frame_idx: advanced::i32(&request.advanced, "imageFrameIndex", 0),
            strength: advanced::f32(&request.advanced, "imageConditioningStrength", 1.0),
        },
        Conditioning::Keyframe {
            image: last,
            frame_idx: -1,
            strength: advanced::f32(&request.advanced, "lastFrameConditioningStrength", 1.0),
        },
    ])
}

/// Whether the job's LoRA set includes an IC-LoRA — the in-context conditioning adapter the
/// LTX extend/bridge keyframe-append path needs (without it the appended clip tokens are inert,
/// per the engine `apply_ltx_adapters` seam). Port of the torch `lora_looks_like_ic_lora`
/// (lora_adapters.py): an explicit `icLora`/`isIcLora` flag, a `conditioningRole: ic_lora`, or an
/// "ic-lora" / "ltx-2-3-ic-" marker anywhere in the id / name / path / file list. The IC-LoRA is a
/// user-installed LoRA flowing through `request.loras` (not an auto-provisioned fixed repo), so it
/// rides the existing [`resolve_ltx_adapters`] seam with no new adapter-loading code.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn loras_contain_ic_lora(loras: &[Value]) -> bool {
    loras.iter().any(lora_looks_like_ic_lora)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn lora_looks_like_ic_lora(lora: &Value) -> bool {
    sceneworks_core::video_request::lora_looks_like_ltx_ic_lora(lora)
}

/// The torch `lora_looks_like_ic_lora` text test (already `_`→`-` normalised + lowercased).
/// Build the in-context [`Conditioning::VideoClip`] set for extend_clip / video_bridge (sc-3522).
/// Source-of-truth = torch `_ltx_video_conditioning` (video_adapters.py) + the engine consumer
/// `runtime_macos::providers::ltx::build_clips`: each source clip's frames are appended as IC-LoRA in-context tokens
/// at an output **latent** frame index, with a `1 − strength` denoise mask.
/// - **extend_clip** → one clip pinned at latent frame `0`, strength `videoConditioningStrength`.
/// - **video_bridge** → a left clip at `0` (strength `videoConditioningStrength`) + a right clip at
///   latent frame `-1` (the engine's negative-from-end index, `lf + idx`, so the worker needs no
///   latent-frame math), strength `bridgeRightVideoConditioningStrength`.
///
/// Both strengths default to `1.0` (fully pinned), mirroring the torch `_advanced_float` defaults.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn build_video_clip_conditioning(
    request: &VideoRequest,
    left_frames: Vec<Image>,
    right_frames: Option<Vec<Image>>,
) -> WorkerResult<Vec<Conditioning>> {
    let mut conditioning = vec![Conditioning::VideoClip {
        frames: left_frames,
        frame_idx: 0,
        strength: advanced::f32(&request.advanced, "videoConditioningStrength", 1.0),
    }];
    if request.mode == "video_bridge" {
        let right = right_frames.ok_or_else(|| {
            WorkerError::InvalidPayload(
                "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                    .to_owned(),
            )
        })?;
        conditioning.push(Conditioning::VideoClip {
            frames: right,
            frame_idx: -1,
            strength: advanced::f32(
                &request.advanced,
                "bridgeRightVideoConditioningStrength",
                1.0,
            ),
        });
    }
    Ok(conditioning)
}

/// Resolve an asset id to its on-disk media file path (the source clip mp4), mirroring the asset
/// lookup in [`load_reference_image`] but returning the path for ffmpeg frame extraction (the
/// Rust equivalent of the torch `source_asset_media_path`).
///
/// Ungated (compiled in every feature/target config): besides the macOS/candle video conditioning
/// callers, the cross-platform `audio_jobs` path resolves the voice-clone reference and the
/// audio-edit source clip through this same project-scoped guard (sc-13410 / sc-13411), so gating
/// it to the video lanes broke the default Linux `parity` build with an unresolved reference
/// (sc-13523). The body depends only on cross-platform helpers (`ProjectStore`,
/// `safe_project_path`), so it compiles everywhere.
pub(crate) fn resolve_clip_media_path(
    settings: &Settings,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> WorkerResult<PathBuf> {
    let asset = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_asset(project_id, asset_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("source clip asset {asset_id}: {error}"))
        })?;
    let rel = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("source clip asset {asset_id} has no media path"))
        })?;
    // file.path is sidecar-sourced (user-editable on disk), so guard it through
    // safe_project_path instead of a bare join so a poisoned sidecar can't escape
    // the project to read an arbitrary file as the source clip (sc-4278 / F-MLXW-14).
    let path = crate::safe_project_path(project_path, rel)?;
    if !path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "source clip file is missing for asset {asset_id}: {}",
            path.display()
        )));
    }
    Ok(path)
}

/// Decode the first `count` frames of a source clip into [`Image`]s for in-context conditioning.
/// Mirrors the torch reference `decode_video_by_frame(starting_frame=0, frame_cap=num_frames)` /
/// `video_preprocess` (ltx_pipelines): **sequential** frames from the start at the clip's native
/// cadence (no fps resample), fit to the output `width`×`height` by contain+pad (letterbox,
/// `FRAME_PAD_COLOR`) so a clip whose aspect differs from the output is not distorted (sc-6229;
/// the engine `build_clips` LANCZOS-downsizes each frame to stage-1 half-res, so this only bounds
/// memory). `count` is the
/// generation's snapped frame count (`8k+1`); a clip shorter than `count` yields fewer frames,
/// which the engine VAE encode accepts. Extracted via the shared [`run_ffmpeg`] (binary
/// resolution + heartbeat/cancel), then loaded off the async runtime.
///
/// Shared by the macOS MLX Bernini conditioning and the candle Bernini VIDEO lane (sc-10997): both
/// resolve arbitrary source-clip asset ids (mv2v supplies several via `sourceClipAssetIds`, ads2v
/// appends the reference video) into planner `VideoClip` conditioning, so the per-asset-id loader is
/// gated to both lanes (unlike `load_source_video_frames`, which reads only the request's single
/// `sourceClipAssetId`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
pub(super) async fn extract_clip_frames(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_id: &str,
    project_path: &Path,
    asset_id: &str,
    width: u32,
    height: u32,
    count: u32,
) -> WorkerResult<Vec<Image>> {
    let clip_path = resolve_clip_media_path(settings, project_id, asset_id, project_path)?;
    let frames_dir = project_path
        .join("assets")
        .join(".cond_clips")
        .join(Uuid::new_v4().simple().to_string());
    tokio::fs::create_dir_all(&frames_dir).await?;
    let pattern = frames_dir.join("frame_%05d.png");
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let result = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            clip_path.display().to_string(),
            // Contain+pad (letterbox) to the output dims so a clip whose aspect differs from the
            // requested W×H is not stretched (sc-6229); reuses the `FRAME_PAD_COLOR` recipe. The
            // engine re-resizes to stage-1 half-res, so this only bounds the extracted footprint.
            "-vf".to_owned(),
            format!(
                "scale={width}:{height}:force_original_aspect_ratio=decrease,\
                 pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={FRAME_PAD_COLOR},format=rgb24"
            ),
            // First `count` decoded frames, sequential from the start at native cadence.
            "-frames:v".to_owned(),
            count.to_string(),
            "-start_number".to_owned(),
            "0".to_owned(),
            pattern.display().to_string(),
        ],
        Some(ctx),
    )
    .await;
    // Load the extracted PNGs (sorted by frame index) into `Image`s, off the async runtime.
    let load = async {
        result?;
        let dir = frames_dir.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
            let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
                .collect();
            paths.sort();
            let mut frames = Vec::with_capacity(paths.len());
            for path in paths {
                let decoded = crate::image_decode::decode_image_any(&path)
                    .map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "conditioning frame {}: {error}",
                            path.display()
                        ))
                    })?
                    .to_rgb8();
                frames.push(Image {
                    width: decoded.width(),
                    height: decoded.height(),
                    pixels: decoded.into_raw(),
                });
            }
            Ok(frames)
        })
        .await
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?
    };
    let frames = load.await;
    // Best-effort cleanup of the scratch frame dir regardless of outcome.
    let _ = tokio::fs::remove_dir_all(&frames_dir).await;
    let frames = frames?;
    if frames.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "source clip {asset_id} produced no decodable frames for conditioning"
        )));
    }
    Ok(frames)
}

/// Resolve extend_clip / video_bridge into the in-context [`Conditioning::VideoClip`] set (sc-3522).
/// Requires an installed IC-LoRA (the keyframe-append adapter) — mirrors the torch gate
/// (`_uses_ic_lora_pipeline` + the "requires at least one installed LTX-compatible LoRA" error),
/// since without it the appended clip tokens are inert. Then decodes each source clip's first
/// `num_frames` frames and builds the clips. `num_frames` is the generation's snapped frame count,
/// the same value [`generate_ltx`] passes to the engine.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn resolve_video_clip_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    if !loras_contain_ic_lora(&request.loras) {
        return Err(WorkerError::InvalidPayload(format!(
            "{} requires an installed IC-LoRA (in-context conditioning adapter) — add an \
             LTX IC-LoRA to the selected preset; without it the source-clip conditioning is inert.",
            request.mode.replace('_', " ")
        )));
    }
    let left_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} requires a source clip (sourceClipAssetId).",
            request.mode.replace('_', " ")
        ))
    })?;
    let num_frames = ltx_frame_count(request.raw_frame_count());
    let left_frames = extract_clip_frames(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        left_id,
        request.width,
        request.height,
        num_frames,
    )
    .await?;
    let right_frames = if request.mode == "video_bridge" {
        let right_id = request
            .bridge_right_clip_asset_id
            .as_deref()
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
        Some(
            extract_clip_frames(
                api,
                settings,
                job,
                &request.project_id,
                project_path,
                right_id,
                request.width,
                request.height,
                num_frames,
            )
            .await?,
        )
    } else {
        None
    };
    build_video_clip_conditioning(request, left_frames, right_frames)
}

/// Resolve native LTX/Eros person replacement into the provider's `ControlClip` plus character
/// references. The selected IC-LoRA is mandatory because it teaches the keyframe-append tokens used
/// by this mode; without it the source clip would be accepted but its replacement conditioning would
/// be inert. This helper is backend-neutral and preserves the selected LTX model instead of silently
/// substituting the unrelated Wan-VACE checkpoint.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn resolve_ltx_replace_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    adapter: &'static str,
) -> WorkerResult<(Vec<Conditioning>, Value)> {
    if !loras_contain_ic_lora(&request.loras) {
        return Err(WorkerError::InvalidPayload(
            "LTX replace person requires a selected LTX IC-LoRA; without it the native control-clip conditioning is inert."
                .to_owned(),
        ));
    }
    let track_id = request.person_track_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a person track (personTrackId).".to_owned(),
        )
    })?;
    let track = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_person_track(&request.project_id, track_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("person track {track_id}: {error}"))
        })?;
    let frame_count = ltx_frame_count(request.raw_frame_count()) as usize;
    let frames =
        load_source_video_frames(api, settings, job, request, project_path, frame_count).await?;
    let (masks, mask_mode) = crate::person_replace::person_track_masks(
        project_path,
        &track,
        request.width,
        request.height,
        frames.len(),
    )?;
    let references = resolve_character_references(settings, request, project_path)?;
    let reference_count = references.len();
    let frame_total = frames.len();
    let masking_strength = advanced::f32(&request.advanced, "maskingStrength", 1.0);
    let conditioning = build_vace_conditioning(
        frames,
        masks,
        references,
        masking_strength,
        replacement_mode_from(&request.replacement_mode),
    )?;
    let status = replacement_status_value(
        &track,
        track_id,
        mask_mode,
        masking_strength,
        reference_count,
        frame_total,
        adapter,
    );
    Ok((conditioning, status))
}

/// Raw-settings recorded on a real MLX LTX asset (`advanced` knobs + real-inference markers).
#[cfg(target_os = "macos")]
pub(super) fn ltx_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    Value::Object(raw)
}

/// Resolve an LTX request into a [`VideoGenInput`] and run it (sc-3035). Distilled 2-stage
/// → no negative prompt / guidance (CFG 1.0); quant is checkpoint-driven (`None`); frames
/// snap to `8k+1`; `advanced.noAudio` → `video_mode = "no_audio"` (full A/V denoise, audio
/// decode skipped); prompt-enhance + per-pass LoRA flow through.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_ltx(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Option<Value>)> {
    // Validate and resolve an explicit encoder choice before any clip decoding, model download, or
    // engine setup. A bad request must remain a cheap, actionable InvalidPayload.
    let (use_uncensored_enhancer, uncensored_enhancer_dir) = resolve_selected_ltx_text_encoder(
        &request.advanced,
        resolve_ltx_uncensored_enhancer_dir(&settings.data_dir),
    )?;
    let enhance_prompt = advanced::bool(&request.advanced, "enhancePrompt");
    let video_mode = advanced::bool(&request.advanced, "noAudio").then(|| "no_audio".to_owned());
    let enhance_max_tokens = request
        .advanced
        .get("enhanceMaxTokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let enhance_temperature = request
        .advanced
        .get("enhanceTemperature")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);
    // extend_clip / video_bridge build in-context VideoClip conditioning from decoded source
    // clips (async ffmpeg extraction); every other mode resolves keyframe/reference conditioning
    // synchronously from images.
    let (conditioning, replacement_status) = match request.mode.as_str() {
        "extend_clip" | "video_bridge" => (
            resolve_video_clip_conditioning(api, settings, job, request, project_path).await?,
            None,
        ),
        "replace_person" => {
            let (conditioning, status) = resolve_ltx_replace_conditioning(
                api,
                settings,
                job,
                request,
                project_path,
                LTX_ADAPTER,
            )
            .await?;
            (conditioning, Some(status))
        }
        _ => (
            resolve_ltx_conditioning(settings, request, project_path)?,
            None,
        ),
    };
    // The macOS default download is lean (q4 + gemma); a Q8 / bf16 job fetches the bundle's q8/ or
    // bf16/ on demand before resolving (sc-5679 / sc-8513). No-op unless that tier is requested and
    // its subdir is absent.
    ensure_ltx_q8_present(api, settings, job, request).await?;
    ensure_ltx_bf16_present(api, settings, job, request).await?;
    // The eros install ships no bundled TE, so provision the bundle's `gemma/` on demand (no-op for
    // the base model, which bundles gemma with its checkpoint). Self-heals installs that predate this.
    ensure_ltx_gemma_present(api, settings, job, request).await?;
    let model_dir = resolve_ltx_model_dir(settings, request)?;
    // Thread the Gemma-3 text encoder onto the LoadSpec (sc-8827, was `$LTX_GEMMA_DIR`). Base: the
    // SceneWorks bundle subdir's sibling `gemma/`. Eros: the separately-provisioned gemma
    // ([`ensure_ltx_gemma_present`]) — a `models/mlx/gemma` sibling or the bundle snapshot's `gemma/`.
    // `None` ⇒ the engine falls back to the HF-cache gemma snapshot.
    let text_encoder_dir = if request.model == "ltx_2_3_eros" {
        resolve_ltx_eros_gemma_dir(settings, &model_dir)
    } else {
        resolve_bundled_ltx_gemma_dir(&model_dir)
    };
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant: None,
        adapters: resolve_ltx_adapters(settings, request)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt: None,
        width: request.width,
        height: request.height,
        frames: ltx_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps: None,
        guidance: None,
        seed: resolve_video_seed(request) as u64,
        control_scale: None,
        video_mode,
        enhance_prompt,
        use_uncensored_enhancer,
        enhance_max_tokens,
        enhance_temperature,
        text_encoder_dir,
        uncensored_enhancer_dir,
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    Ok((decoded, replacement_status))
}
