#[allow(unused_imports)]
use super::prelude::*;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::{
    mochi::{
        ensure_mochi_bf16_present, ensure_mochi_q8_present, mochi_precheck_dir, mochi_tier_quant,
        mochi_vram_precheck, resolve_mochi_model_dir, validate_mochi_mode, MOCHI_REPO,
    },
    scail2::{scail2_engine_video_mode, scail2_raw_settings, SCAIL2_REPO, SCAIL2_REVISION},
    svd::{svd_f32, svd_i32, svd_raw_settings, svd_steps, SVD_REPO},
    vace::{
        build_extend_bridge_vace_conditioning, build_vace_conditioning, extend_anchor_frames,
        load_clip_anchor_frames, load_source_video_frames, replacement_mode_from,
        replacement_status_value, resolve_character_references,
    },
    wan::{
        advanced_opt_f32, advanced_opt_u32, ensure_wan_lightning_present, generate_video,
        generate_video_using, resolve_scail2_adapters, resolve_wan_adapters, scail2_sampling,
        wan_lightning_on, wan_sampling, ClipFramePosition, ComfyuiWanExperts, VideoGenInput,
    },
};

// ---------------------------------------------------------------------------
// Candle (Windows/CUDA) video lane (sc-5097, epic 5095). The candle wan/ltx providers serve a narrow
// **txt2video-only** first slice (no image/VACE conditioning, audio, LoRA, or quant). This is the
// video sibling of the candle image lane (image_jobs.rs `generate_candle_stream`): it builds a
// `VideoGenInput` and drives the SAME neutral streaming harness (`generate_video` →
// `run_loaded_video_generation` → the registry-resolved candle generator), reusing the shared
// encode/mux/poster path. Reached only when `backend_candle_enabled` (default off).
// ---------------------------------------------------------------------------

/// Per-asset adapter ids for the candle video engines (`candle_<family>`), the candle siblings of
/// the MLX `mlx_wan` / `mlx_ltx` labels (sc-5097).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const CANDLE_WAN_ADAPTER: &str = "candle_wan";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_LTX_ADAPTER: &str = "candle_ltx";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_SVD_ADAPTER: &str = "candle_svd";
/// Adapter id recorded on a candle Mochi 1 asset (epic 1788 / sc-11992). Distinct from the MLX
/// [`MOCHI_ADAPTER`] so the sidecar records which backend rendered the clip, matching the
/// `candle_wan`/`mlx_wan` and `candle_ltx`/`mlx_ltx` pairs.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const CANDLE_MOCHI_ADAPTER: &str = "candle_mochi";

/// Default HuggingFace repos the candle video providers load (overridable via the manifest `repo`).
/// The candle wan providers read a Wan2.2 diffusers snapshot — the TI2V-5B, or the T2V-A14B /
/// I2V-A14B 14B MoE (sc-5175); base LTX reads the shared SceneWorks packed q4 + sibling Gemma
/// turnkey (sc-13870).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const CANDLE_WAN_5B_REPO: &str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_T2V_14B_REPO: &str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_I2V_14B_REPO: &str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_LTX_REPO: &str = "SceneWorks/ltx-2.3-mlx";
// The `ltx_2_3_eros` weights repo (sc-5495): a full dense LTX-2.3 fine-tune (the candle provider
// loads its bf16 single-file checkpoint like the base; same architecture, same Gemma encoder).
// The pinned checkpoint version is manifest-driven (`downloads[].files` / `mlx.convertSourceFile`),
// so only that one file is fetched even though the repo also ships older + fp8 variants.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_LTX_EROS_REPO: &str = "TenStrip/LTX2.3-10Eros";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_LTX_GEMMA_REPO: &str = "google/gemma-3-12b-it";

/// Per-asset adapter id for the candle Wan-VACE controllable-video lane (sc-5494) — the candle sibling
/// of the MLX `mlx_wan_vace` label.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const CANDLE_WAN_VACE_ADAPTER: &str = "candle_wan_vace";

/// The diffusers Wan2.1-VACE-14B snapshot the candle `wan_vace` provider reads (`transformer/` +
/// `text_encoder/` + `vae/` + `tokenizer/`). Overridable via `SCENEWORKS_CANDLE_WAN_VACE_DIR`. The 14B
/// repo matches the provider's `WanVaceConfig::vace_14b` dims (dim 5120, 40 layers).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_VACE_REPO: &str = "Wan-AI/Wan2.1-VACE-14B-diffusers";

/// SceneWorks video model id → candle registry engine id, or `None` for an id the candle video lane
/// does not serve. Note ltx maps to `ltx_2_3_distilled` (the candle provider's id), not the MLX
/// `ltx_2_3`. Covers the base txt2video ids (5B + ltx) plus the Wan2.2 **14B** dual-expert MoE pair
/// (sc-5174 / sc-5175): `wan_2_2_t2v_14b` (text→video) and `wan_2_2_i2v_14b` (image→video), plus `svd`
/// (image→video, sc-5493 / epic 5481). `ltx_2_3_eros` (sc-5495) maps to the same `ltx_2_3_distilled`
/// engine as the base — it's a full dense LTX-2.3 fine-tune — differing only in the weights repo.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_video_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "wan_2_2" => Some("wan2_2_ti2v_5b"),
        "wan_2_2_t2v_14b" => Some("wan2_2_t2v_14b"),
        "wan_2_2_i2v_14b" => Some("wan2_2_i2v_14b"),
        // Base + eros both load the one candle LTX-2.3 engine (the eros merge is a full dense LTX-2.3
        // checkpoint, sc-5495); they differ only in the weights repo (see `candle_video_repo`).
        "ltx_2_3" | "ltx_2_3_eros" => Some("ltx_2_3_distilled"),
        // SVD-XT image→video (sc-5493 / epic 5481): the candle-gen-svd provider's `svd_xt` engine.
        "svd" => Some("svd_xt"),
        // Mochi 1 (epic 1788 / sc-11992). The sceneworks id IS the engine id: `candle-gen-mochi`
        // registers the SAME `MODEL_ID = "mochi_1"` as `mlx-gen-mochi` (no `_distilled`-style split),
        // and its descriptor is `mac_only: false` — the off-Mac lane is real and CUDA-validated on
        // Blackwell (sc-11990), ingesting the same hosted mlx-affine tiers.
        //
        // Load-bearing: without this arm `is_candle_video_engine` is false, `resolve_candle_video_route`
        // never reaches the generic arm, and a Windows Mochi job falls to `CandleVideoRoute::Stub` —
        // handing the user a PROCEDURAL FAKE VIDEO instead of an error. B1 already routed Windows
        // (`candle_video_routed = true`), so that promise must be served here.
        "mochi_1" => Some("mochi_1"),
        _ => None,
    }
}

/// Whether `model` is served by the candle video lane (sc-5097).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn is_candle_video_engine(model: &str) -> bool {
    candle_video_engine_id(model).is_some()
}

/// The adapter id recorded on a candle video asset. Every engine that is NOT wan MUST have an
/// explicit arm: the `_` fall-through is the Wan default, so a missing arm silently stamps a
/// different model's provenance onto the asset sidecar + telemetry (sc-11992 — `mochi_1` landed in
/// `_` and was labelled a Wan adapter).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_video_adapter_label(engine_id: &str) -> &'static str {
    match engine_id {
        "ltx_2_3_distilled" => CANDLE_LTX_ADAPTER,
        "svd_xt" => CANDLE_SVD_ADAPTER,
        "mochi_1" => CANDLE_MOCHI_ADAPTER,
        _ => CANDLE_WAN_ADAPTER,
    }
}

/// The candle default weights repo for a video engine id (the per-variant Wan2.2 diffusers snapshot,
/// or the LTX-2.3 checkpoint). Used when the manifest entry omits `repo`. Like
/// [`candle_video_adapter_label`], the `_` arm is the Wan default — a non-wan engine without an
/// explicit arm inherits Wan's repo (sc-11992).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_video_default_repo(engine_id: &str) -> &'static str {
    match engine_id {
        "ltx_2_3_distilled" => CANDLE_LTX_REPO,
        "svd_xt" => SVD_REPO,
        "wan2_2_t2v_14b" => CANDLE_WAN_T2V_14B_REPO,
        "wan2_2_i2v_14b" => CANDLE_WAN_I2V_14B_REPO,
        // Mochi 1 (epic 1788 / sc-11992): ONE repo serves BOTH backends — candle ingests the same
        // mlx-affine tiers via A6's `.scales`-detect seam, and `SceneWorks/mochi-1-candle` was never
        // published. No manifest entry carries a top-level `repo`, so this default is what the candle
        // lane actually resolves.
        "mochi_1" => MOCHI_REPO,
        // `wan2_2_ti2v_5b` (and any other wan id) → the 5B TI2V snapshot.
        _ => CANDLE_WAN_5B_REPO,
    }
}

/// The candle weights repo for a video engine: the manifest `repo` wins, else — for the Wan
/// quant-matrix models whose per-tier candle repos live in `downloads[]` with no top-level `repo`
/// (`SceneWorks/wan2.2-*-candle`, sc-10027) — the platform-appropriate candle tier repo matching the
/// requested tier (default q4), else `ltx_2_3_eros`'s own fine-tune repo, else the candle default repo.
///
/// Without the `downloads[]` resolution the Windows/Linux Wan-14B lane fell back to the DENSE
/// `Wan-AI/*-Diffusers` default — a different (bf16, ~72 GB) repo that the packed-tier install never
/// fetches — so a candle Wan-14B job errored "snapshot not found" even with the q4 tier present (sc-10539).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_video_repo(request: &VideoRequest, engine_id: &str) -> String {
    if let Some(repo) = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return repo.to_owned();
    }
    if let Some(repo) = candle_wan_tier_repo_from_downloads(request, engine_id) {
        return repo;
    }
    if request.model == "ltx_2_3_eros" {
        CANDLE_LTX_EROS_REPO.to_owned()
    } else {
        candle_video_default_repo(engine_id).to_owned()
    }
}

/// The candle Wan tier repo from the manifest `downloads[]` for THIS platform (sc-10539). The Wan
/// quant-matrix (sc-10027) hosts each candle tier as a per-`variant` download entry — `q4`/`q8` in the
/// packed `SceneWorks/wan2.2-*-candle` repo, `bf16` in the dense `Wan-AI/*-Diffusers` repo — rather than
/// a single top-level `repo`, so `candle_video_repo` must consult them. Picks the repo for the highest-
/// preference tier present for this OS (default **q8-first** when the manifest lists it for the platform,
/// clamping to q4 otherwise — epic 10721 / sc-10726 — mirroring [`candle_wan_tier_subdir`] so the
/// resolved repo is the one whose tier subdir the loader then selects). `None` for non-Wan engines or a
/// manifest without matching platform downloads.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_wan_tier_repo_from_downloads(request: &VideoRequest, engine_id: &str) -> Option<String> {
    if !engine_id.starts_with("wan2_2") {
        return None;
    }
    let downloads = request.model_manifest_entry.get("downloads")?.as_array()?;
    let platform = if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    };
    let order: &[&str] = match candle_wan_quant_bits(request) {
        None => &["q8", "q4"],
        Some(bits) if bits <= 0 => &["bf16", "q8", "q4"],
        Some(bits) if bits >= 8 => &["q8", "q4"],
        _ => &["q4", "q8"],
    };
    order.iter().find_map(|&tier| {
        downloads.iter().find_map(|download| {
            if download.get("variant").and_then(Value::as_str) != Some(tier) {
                return None;
            }
            let on_platform = match download.get("platforms").and_then(Value::as_array) {
                Some(platforms) => platforms
                    .iter()
                    .any(|value| value.as_str() == Some(platform)),
                None => true,
            };
            if !on_platform {
                return None;
            }
            download
                .get("repo")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
    })
}

/// Resolve the candle weights snapshot dir for `repo`. Errors loudly (no procedural-stub fallback)
/// when the snapshot is absent, so a missing model surfaces a re-download error.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_snapshot_dir(settings: &Settings, repo: &str) -> WorkerResult<PathBuf> {
    huggingface_snapshot_dir(&settings.data_dir, repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "candle video weights snapshot not found for {repo}"
        ))
    })
}

/// (sc-10027) The `advanced.mlxQuantize` bits for a candle wan tier-select — a number or numeric string;
/// `None` when unset.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_wan_quant_bits(request: &VideoRequest) -> Option<i64> {
    request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
}

/// (sc-10027) Whether `dir` is a complete candle wan tier — a diffusers-layout snapshot with the DiT
/// transformer(s), the T5 encoder, the VAE and the tokenizer. The A14B MoE carries a second expert
/// (`transformer_2/`); the TI2V-5B is a single transformer.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_wan_tier_complete(dir: &Path, a14b: bool) -> bool {
    let has = |sub: &str| dir.join(sub).is_dir();
    has("transformer")
        && has("text_encoder")
        && has("vae")
        && has("tokenizer")
        && (!a14b || has("transformer_2"))
}

/// (sc-12402) The `candle.vramGbByTier` manifest key for the tier [`candle_wan_tier_subdir`] actually
/// resolved — derived from the RESOLVED quant marker, never re-derived from the request.
///
/// That is the sc-12090 lesson, which cost a false reject on the image lane: re-deriving the tier from
/// `advanced.mlxQuantize` sizes the tier the request ASKED for, while the disk-probing resolver may
/// have clamped to a different one (q8 default → q4 when only q4 is installed). Keying off the marker
/// the resolver returned means the gate and the loader agree on which tier ran by construction.
///
/// `None` ⇒ `bf16`, and that one mapping covers BOTH bf16 shapes: an explicit `bf16/` tier subdir, and
/// the flat dense `Wan-AI/*-Diffusers` fallback that [`candle_wan_tier_subdir`] declines (the caller
/// pairs that snapshot with a `None` marker). Wan advertises only `Quant::{Q4, Q8}`
/// (`supported_quants`, wan14b.rs), so no other variant can reach here; any future one would read
/// `bf16` — the HEAVIEST row, i.e. conservative.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_wan_tier_key(quant: Option<Quant>) -> &'static str {
    match quant {
        Some(Quant::Q4) => "q4",
        Some(Quant::Q8) => "q8",
        _ => "bf16",
    }
}

/// Resolve the base LTX packed-q4 turnkey tier shared by generation and training. The checkpoint is
/// already packed, so the returned load quant is deliberately `None`: `LoadSpec::quantize` means
/// on-the-fly quantization to the Candle LTX provider and must never be set for this tier. Eros remains
/// on its dense standalone checkpoint, so the SceneWorks model id is part of the guard.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_ltx_tier_subdir(
    root: &Path,
    engine_id: &str,
    model: &str,
) -> Option<(PathBuf, Option<Quant>)> {
    if engine_id != "ltx_2_3_distilled" || model != "ltx_2_3" {
        return None;
    }
    let q4 = root.join("q4");
    (q4.join("transformer.safetensors").is_file() && q4.join("quantize_config.json").is_file())
        .then_some((q4, None))
}

/// (sc-10027) Resolve the candle wan quant tier subdir (`q4`/`q8`/`bf16`) + its quant marker under a
/// `SceneWorks/wan2.2-*-candle` snapshot `root`, per `advanced.mlxQuantize` (default **q8** when
/// installed, clamping to q4 — epic 10721 / sc-10726 — falling back through the tier order), or `None`
/// for a non-wan engine or a flat repo with no tier subdirs (e.g. the
/// dense `Wan-AI/*-Diffusers` fallback, which loads as-is). A resolved subdir **is** the diffusers-layout
/// snapshot the sc-10025 packed-detect seam loads — the quant is baked into the tier, so the `Quant`
/// returned is a tier-select marker (`spec.quantize` is a no-op on the candle wan load). Candle analog of
/// the macOS `wan_tier_subdir` / `resolve_wan_tier_dir_and_quant` (sc-9079).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_wan_tier_subdir(
    root: &Path,
    engine_id: &str,
    request: &VideoRequest,
) -> Option<(PathBuf, Option<Quant>)> {
    if !engine_id.starts_with("wan2_2") {
        return None;
    }
    // Tier preference by requested bits (mirrors the macOS `wan_tier_order`): no explicit pick → q8
    // (the app-wide default, clamped to the installed tier, so q4-only installs still resolve q4);
    // explicit ≤4 → q4. `bf16` stays out of the default order (never auto-loaded on a default job).
    let order: &[&str] = match candle_wan_quant_bits(request) {
        None => &["q8", "q4"],
        Some(b) if b <= 0 => &["bf16", "q8", "q4"],
        Some(b) if b >= 8 => &["q8", "q4"],
        _ => &["q4", "q8"],
    };
    let a14b = engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b";
    order.iter().find_map(|&tier| {
        let dir = root.join(tier);
        candle_wan_tier_complete(&dir, a14b).then(|| {
            let quant = match tier {
                "q4" => Some(Quant::Q4),
                "q8" => Some(Quant::Q8),
                _ => None, // bf16
            };
            (dir, quant)
        })
    })
}

/// Resolve the Gemma-3-12B encoder snapshot dir for the candle LTX provider (sc-8827). Returns the
/// HF-cache snapshot path so the caller can thread it onto `LoadSpec::text_encoder` — no more mutating
/// the process-global `$LTX_GEMMA_DIR` at job time on the multithreaded runtime (the old `set_var`
/// seam was unsound, F-025). An explicit complete operator `$LTX_GEMMA_DIR` wins and is threaded on
/// the spec; the provider no longer reads process-global env itself. Best-effort: if neither source
/// resolves, the provider emits its required-`LoadSpec::text_encoder` error.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_ltx_gemma_dir(settings: &Settings) -> Option<PathBuf> {
    super::ltx::ltx_gemma_override_path(std::env::var_os("LTX_GEMMA_DIR"))
        .or_else(|| huggingface_snapshot_dir(&settings.data_dir, CANDLE_LTX_GEMMA_REPO))
}

/// Raw-settings recorded on a candle video asset (mirrors `wan_raw_settings`, trimmed to the
/// txt2video surface): the request `advanced` knobs plus the real-inference markers.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_video_raw_settings(request: &VideoRequest, repo: &str) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("fps".to_owned(), json!(request.fps));
    Value::Object(raw)
}

/// Per-request conditioning for a candle video generation. The Wan2.2 **14B I2V** engine
/// (sc-5174 / sc-5175) and **SVD-XT** (sc-5493) are conditioned: each requires a source image, loaded
/// to a single [`Conditioning::Reference`] — for Wan, the channel-concat first frame the provider
/// VAE-encodes into its `y` (`in_dim=36`); for SVD, the CLIP-encoded + noise-aug VAE-encoded driving
/// frame. The candle analog of the MLX i2v / SVD conditioning ([`resolve_wan_conditioning`]).
/// Every other candle video engine (5B, T2V-14B, ltx) is txt2video-only, so this returns an empty set.
/// The router's `video_request_candle_eligible` already guarantees the i2v shape carries a source and
/// the txt2video ids do not, but the source is required here too so a mis-routed job fails clearly.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn resolve_candle_video_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &str,
) -> WorkerResult<Vec<Conditioning>> {
    // The Wan2.2 14B I2V engine (sc-5175) and SVD-XT (sc-5493) condition on a single source image; every
    // other candle video engine (5B, T2V-14B, ltx) is txt2video-only (empty conditioning).
    if engine_id != "wan2_2_i2v_14b" && engine_id != "svd_xt" {
        return Ok(Vec::new());
    }
    let asset_id = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{engine_id}: image-to-video requires a source image (sourceAssetId)."
            ))
        })?;
    let image = crate::image_jobs::load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    // Pre-fit the source to the output W×H by the chosen crop/pad mode (sc-6139), the
    // same as the macOS LTX/Wan paths, so the Windows/CUDA candle I2V engine conditions
    // on an undistorted frame instead of an internal stretch.
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

/// The SVD pre-flight's gated result (sc-14492). The frame count used by the engine is obtainable only
/// by passing the CUDA VRAM check, so the generation arm cannot accidentally bypass admission while
/// still compiling.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SvdVramPreflight {
    pub(super) frames: u32,
    pub(super) decode_chunk_size: u32,
    pub(super) steps: u32,
}

/// Refuse an SVD burst that exceeds the profile validated on this physical VRAM class before resolving
/// the checkpoint or loading any component. `budget` is resolved by the caller so the decision stays
/// deterministic under the `SCENEWORKS_CUDA_VRAM_CAP_GB` hardware-emulation lane.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn svd_vram_preflight(
    request: &VideoRequest,
    gpu_id: &str,
    budget: Option<crate::vram_gate::VramBudget>,
) -> WorkerResult<SvdVramPreflight> {
    let frames = svd_i32(request, "numFrames", "numFrames", 25, 1, 25) as u32;
    let decode_chunk_size = svd_i32(request, "decodeChunkSize", "decodeChunkSize", 8, 1, 64) as u32;
    let steps = svd_steps(request);
    match crate::vram_gate::svd_fit_error(
        frames,
        decode_chunk_size,
        steps,
        request.width,
        request.height,
        gpu_id,
        budget,
    ) {
        Some(error) => Err(error),
        None => Ok(SvdVramPreflight {
            frames,
            decode_chunk_size,
            steps,
        }),
    }
}

/// Build the exact SVD input consumed by the production arm after admission. Keeping this in one
/// helper makes the measured fit gate inseparable from the sequential residency policy it was
/// measured under; the SVD arm returns before the generic Wan/LTX input builder below.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_svd_input(
    engine_id: &'static str,
    model_dir: PathBuf,
    conditioning: Vec<Conditioning>,
    request: &VideoRequest,
    preflight: SvdVramPreflight,
) -> VideoGenInput {
    VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        conditioning,
        width: request.width,
        height: request.height,
        frames: preflight.frames,
        fps: request.fps,
        steps: Some(preflight.steps),
        seed: resolve_video_seed(request) as u64,
        motion_bucket_id: Some(
            svd_i32(request, "motionBucketId", "motionBucketId", 127, 1, 255) as f32,
        ),
        noise_aug_strength: Some(svd_f32(
            request,
            "noiseAugStrength",
            "noiseAugStrength",
            0.02,
        )),
        decode_chunk_size: Some(preflight.decode_chunk_size),
        conditioning_fps: Some(svd_i32(request, "conditioningFps", "condFps", 7, 1, 30) as u32),
        offload_policy: candle_video_offload_policy(engine_id),
        ..VideoGenInput::default()
    }
}

/// The candle Mochi pre-flight's gated result (sc-12306): the tier's baked-in quant marker, obtainable
/// ONLY by passing the VRAM fit gate.
///
/// Bundling the marker into the gated return is deliberate, and mirrors the MLX lane's [`MochiPreflight`]
/// — which adopted the shape after a review mutation that deleted a free-standing `mochi_fit_check(...)?`
/// call still compiled and still rendered, silently un-gating the lane. With the marker only obtainable
/// here, the generation arm cannot reach a quant on a path that skipped the gate.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MochiVramPreflight {
    pub(super) quant: Option<Quant>,
}

/// Live pre-flight Mochi VRAM admission check for the candle lane (sc-12306) — the seam
/// [`generate_candle_video`] calls before the load + 64-step denoise.
///
/// Sums the on-disk bytes the load will hold resident via the SHARED [`crate::mlx_fit_gate::mochi_resident_bytes`]
/// (the tier dir's AsymmDiT plus the `text_encoder/` + `vae/` siblings from its parent): despite the
/// module name that scan describes the hosted repo layout, which is one repo serving both lanes, and
/// summing only the tier dir would miss the ~9.7 GiB T5-XXL + VAE — over half the resident footprint.
///
/// `budget` arrives resolved so this stays free of the GPU probe and is unit-testable without CUDA. No
/// budget signal ⇒ admits. `Err` is the actionable pre-denoise rejection.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn mochi_vram_preflight(
    model_label: &str,
    tier_dir: &Path,
    frames: u32,
    width: u32,
    height: u32,
    gpu_id: &str,
    budget: Option<crate::vram_gate::VramBudget>,
) -> WorkerResult<MochiVramPreflight> {
    match crate::vram_gate::mochi_fit_error(
        model_label,
        crate::mlx_fit_gate::mochi_resident_bytes(tier_dir),
        frames,
        width,
        height,
        gpu_id,
        budget,
    ) {
        Some(error) => Err(error),
        None => Ok(MochiVramPreflight {
            quant: mochi_tier_quant(tier_dir),
        }),
    }
}

/// The residency policy the candle load takes for `engine_id` (sc-12631, epic sc-12732). The A14B
/// T2V/I2V render with [`OffloadPolicy::Sequential`]: their two 14B MoE experts cannot be co-resident on
/// a target card, so the engine stages TE → high-expert → low-expert → VAE one-resident-at-a-time
/// (`candle-gen-wan` `render_sequential`). That is the residency the model's measured
/// `candle.vramGbByTier` peak was taken under (22.1 GiB @ q4 / 1280×720, vs a resident load that OOMs a
/// 96 GB card), so [`crate::vram_gate::wan_video_fit_error`]'s admission number is only truthful when the
/// load actually takes this path — [`video_load_spec`] threads it onto the `LoadSpec`, and
/// `apply_residency_policy` never downgrades a `Sequential` set here.
///
/// The dense TI2V-5B now renders [`OffloadPolicy::Sequential`] too (sc-13175). It has a single dense
/// transformer (no expert swap), but its resident peak was dominated by the UMT5 TE + z48 VAE held
/// alongside the DiT for the whole run, so the engine flushes TE/VAE off-GPU around the denoise
/// (sc-12757, `render_sequential`) and the model is now sized by its MEASURED sequential
/// `candle.vramGbByTier` peak (see the `wan_2_2` candle block) instead of the ~46 GiB RESIDENT peak
/// sc-12631 (PR #1598) shipped — so a ~24 GB card can run the 5B where the resident gate needed ~48.
///
/// SVD-XT joins that same contract in sc-14625: the provider stages image encoder + source VAE
/// encode → UNet → VAE decode, so selecting `Sequential` here is what activates its proven
/// one-phase-at-a-time residency and memory-aware decode. LTX still has no offload lifecycle and
/// remains resident.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_video_offload_policy(engine_id: &str) -> OffloadPolicy {
    match engine_id {
        "svd_xt" | "wan2_2_ti2v_5b" | "wan2_2_t2v_14b" | "wan2_2_i2v_14b" => {
            OffloadPolicy::Sequential
        }
        _ => OffloadPolicy::Resident,
    }
}

/// Live pre-flight Wan VRAM admission check for the candle lane — the non-Mochi half of this lane's
/// gate, and the seam [`generate_candle_video`] calls before the load + denoise.
///
/// Budgets on the tier's **MEASURED peak** (`candle.vramGbByTier[tier_key]`, sc-12402) when the
/// manifest carries one, and falls back to the sc-12344 on-disk weights FLOOR when it does not — see
/// [`crate::vram_gate::wan_video_fit_error`], which owns that choice and the reason a measurement
/// REPLACES the floor rather than composing with it.
///
/// The measured peak matters because the floor is wrong in BOTH directions, from one cause: on-disk
/// bytes are not the loaded set, because `candle-gen-wan` casts dtypes on load (`wan14b.rs:49-52` —
/// experts fp32→bf16 HALVE on the dense tier, the UMT5 TE bf16→f32 DOUBLES on every tier). The packed
/// tiers under-count by ~9-11 GiB (which is why a card that cannot fit the job is admitted and then
/// OOMs — sc-12402's story) and the dense tier over-counts by ~44 GiB.
///
/// `ltx`/`svd` carry no `candle` block and read `0` weights, so both paths return `None` in this
/// Wan-specific helper — their on-disk bytes are not their loaded set either, so any byte-derived floor
/// would wall-reject working cards (recorded on `vram_gate::wan_weight_components`; sc-12397). SVD is
/// independently protected by [`svd_vram_preflight`]'s frame-aware hardware-class gate (sc-14492).
///
/// **Takes and returns `model_dir` rather than borrowing it**, so the gate cannot be deleted without
/// breaking the build — the same "unskippable by construction" property [`MochiVramPreflight`] gets from
/// bundling its quant marker, and the shape `mlx_fit_gate::apply_residency_policy(spec, engine_id) ->
/// WorkerResult<LoadSpec>` already uses at the MLX cache seam. A free-standing `check(&dir)?` would
/// still compile after a review mutation deleted it, silently un-gating the lane.
///
/// `budget` arrives resolved so this stays free of the GPU probe and is unit-testable without CUDA. No
/// budget signal (or an exempt engine) ⇒ admits. `Err` is the actionable pre-load rejection.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn wan_vram_preflight(
    engine_id: &str,
    manifest_entry: &JsonObject,
    tier_key: &str,
    model_dir: PathBuf,
    gpu_id: &str,
    budget: Option<crate::vram_gate::VramBudget>,
) -> WorkerResult<PathBuf> {
    wan_vram_preflight_with_adapter_bytes(
        engine_id,
        manifest_entry,
        tier_key,
        model_dir,
        0,
        gpu_id,
        budget,
    )
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn wan_vram_preflight_with_adapter_bytes(
    engine_id: &str,
    manifest_entry: &JsonObject,
    tier_key: &str,
    model_dir: PathBuf,
    adapter_bytes: u64,
    gpu_id: &str,
    budget: Option<crate::vram_gate::VramBudget>,
) -> WorkerResult<PathBuf> {
    match crate::vram_gate::wan_video_fit_error_with_adapter_bytes(
        engine_id,
        manifest_entry,
        tier_key,
        crate::vram_gate::wan_weight_bytes(engine_id, &model_dir),
        adapter_bytes,
        gpu_id,
        budget,
    ) {
        Some(error) => Err(error),
        None => Ok(model_dir),
    }
}

/// Independently resident USER adapter bytes for the Candle Wan load. The calibrated A14B rows
/// already contain the built-in Lightning pair, so callers pass only the user tail. Shared MoE
/// factors are resident once; high/low expert-specific factors alternate and contribute the larger
/// side to the sequential peak.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn wan_user_adapter_resident_bytes(
    adapters: &[AdapterSpec],
    is_moe: bool,
    tier_key: &str,
) -> WorkerResult<u64> {
    if tier_key == "bf16" || adapters.is_empty() {
        return Ok(0);
    }
    let additive_bytes = |stack: &[AdapterSpec]| {
        gen_core::adapter_stack_resident_bytes(stack, gen_core::AdapterResidencyMode::Additive)
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "Wan cannot determine the resident size of the requested adapter stack."
                        .to_owned(),
                )
            })
    };
    if !is_moe {
        return additive_bytes(adapters);
    }
    let shared: Vec<_> = adapters
        .iter()
        .filter(|adapter| adapter.moe_expert.is_none())
        .cloned()
        .collect();
    let high: Vec<_> = adapters
        .iter()
        .filter(|adapter| adapter.moe_expert == Some(gen_core::MoeExpert::High))
        .cloned()
        .collect();
    let low: Vec<_> = adapters
        .iter()
        .filter(|adapter| adapter.moe_expert == Some(gen_core::MoeExpert::Low))
        .cloned()
        .collect();
    Ok(additive_bytes(&shared)?.saturating_add(additive_bytes(&high)?.max(additive_bytes(&low)?)))
}

/// The candle video lane's live VRAM budget: the real `nvidia-smi` reading, the
/// `SCENEWORKS_CUDA_VRAM_CAP_GB` small-card emulation folded over it, then this process's reclaimable
/// cudarc pool added back (sc-11023).
///
/// The reclaimable fold IS correct here: video routes through `generator_cache::with_cached_generator`
/// (the `comfyui` in-place MoE is the one uncached exception, and it is not Mochi), so the single
/// exclusive cache slot evicts its occupant BEFORE the incoming load. This budget is read while that
/// occupant is still resident, so raw `free` still counts the model it is about to replace; the fold
/// predicts the `free` the imminent evict will produce (whether that VRAM returns to the driver — a full
/// generator drop does, GPU-measured sc-13960 — or is reused in-process). Matches `generate_candle_stream`
/// (image_jobs/base.rs). The bespoke edit/control lanes reached the same result differently: they load
/// off the UNcached `start_gen_stream`, so they gate raw free and only EVICT-then-reclaim when it flips a
/// downtier/reject (sc-13960 `gate_with_evict_reclaim`) — before sc-13960 they omitted the fold entirely.
///
/// The reverse direction is deliberately NOT wired: an admitted Mochi job does not call
/// [`crate::vram_gate::note_loaded_peak`], so it contributes nothing to the reclaimable high-water the
/// image lane reads. That keeps today's behavior (the video lane has never recorded a peak) rather than
/// guessing: Mochi's predicted peak is dominated by a TRANSIENT decode, not by resident weights, and
/// publishing ~81 GB as "reclaimable" on the strength of a derived floor would relax later image gates
/// on a number nothing has measured. Under-reporting the pool only ever fails conservative (a spurious
/// reject, never an OOM). Revisit once B5/sc-11995 backfills real `footprint.peakMemoryBytes`.
///
/// **Called once per GATE, not once per job** (sc-12373): the pre-download check and the pre-load check
/// each take a fresh reading. They are minutes apart across a ~13–20 GiB download, and free VRAM is not
/// stable across that window — another process (or another worker) can take or release a card in the
/// meantime, so caching the first reading would gate the load against a stale number. The cost is one
/// extra `nvidia-smi` per Mochi job, which is noise next to the download it exists to avoid.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn candle_video_vram_budget(settings: &Settings) -> Option<crate::vram_gate::VramBudget> {
    let budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    budget.map(|budget| {
        crate::vram_gate::with_reclaimable(
            budget,
            crate::vram_gate::reclaimable_pool_gb(&settings.gpu_id),
        )
    })
}

/// The SCAIL model directory and its one-shot cold-load admission travel together. Carrying both in
/// this plan makes the contract structural: each production arm must attach the plan to its
/// [`VideoGenInput`], while the shared generator cache decides atomically whether the exact key is a
/// warm hit (bypass) or a cold miss (evict, fresh post-evict probe, gate, load).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) struct Scail2ColdLoadPlan {
    pub(super) model_dir: PathBuf,
    pub(super) admission: crate::generator_cache::GeneratorColdLoadAdmission,
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn scail2_cold_load_plan(
    manifest_entry: &JsonObject,
    model_dir: PathBuf,
    gpu_id: &str,
) -> Scail2ColdLoadPlan {
    let manifest_entry = manifest_entry.clone();
    let gpu_id = gpu_id.to_owned();
    let admission = crate::generator_cache::GeneratorColdLoadAdmission::new(move || {
        // This executes on the dedicated cache OS thread, after a different resident has been
        // dropped. The helper owns a private bounded runtime, so it cannot deadlock the async worker
        // that is awaiting this cache job, and it deliberately bypasses the heartbeat's stale cache.
        let budget = crate::vram_gate::apply_vram_cap(
            crate::gpu::nvidia_vram_budget_gb_fresh_blocking(&gpu_id),
            crate::vram_gate::cuda_vram_cap_gb(),
        );
        match crate::vram_gate::scail2_video_fit_error(&manifest_entry, &gpu_id, budget) {
            Some(error) => Err(error),
            None => Ok(()),
        }
    });
    Scail2ColdLoadPlan {
        model_dir,
        admission,
    }
}

/// Windows/CUDA candle video path (sc-5097 txt2video; sc-5175 adds the Wan2.2 14B MoE T2V + I2V).
/// Resolves the engine + weights, provisions the LTX Gemma encoder, resolves any i2v source-image
/// conditioning, builds a `VideoGenInput`, and runs it through the shared [`generate_video`] streaming
/// driver. Returns the decoded clip + the candle adapter label.
///
/// Admission is family-specific and happens before load: Mochi's frame-dependent decode gate
/// (sc-12306), SVD's 32 GB / long-burst gate (sc-14492), and the Wan gate — the tier's MEASURED
/// `candle.vramGbByTier` peak where the manifest carries one (sc-12402), else sc-12344's on-disk
/// weights floor. LTX remains exempt because its on-disk bytes are not its loaded set and no measured
/// peak exists; a byte-derived floor would wall-reject working cards.
///
/// Mochi keeps its own gate rather than joining the Wan one: its AsymmVAE decode is UNTILED, so its peak
/// grows linearly in clip length and no per-tier constant can express it (sc-12306). Wan's decode is
/// budget-TILED (`auto_tiling_budgeted_wan22`) and its peak is owned by the denoise, so a per-tier
/// constant at the model's default geometry is the honest shape there — the manifest schema's
/// "video = default frames".
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) async fn generate_candle_video(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, &'static str, Value)> {
    generate_candle_video_using(
        api,
        settings,
        job,
        request,
        project_path,
        backend,
        crate::inference_runtime::load,
    )
    .await
}

/// [`generate_candle_video`] with the engine loader supplied by the caller (sc-12318) — the candle
/// sibling of [`generate_mochi_using`], and for the same reason.
///
/// [`video_frame_count`] is a large part of the pre-load exposure here: swapping it for
/// `wan_frame_count` puts every non-Wan family (Mochi's `6k+1`, LTX's `8k+1`) off its engine's lattice,
/// which `validate_request` hard-rejects. `generate_candle_video_using_*` pins that call at the caller.
/// (Mochi's fit gate reads the coerced count too, but Wan's weights floor is frame-blind and LTX is
/// exempt, so the lattice remains this arm's own pin for those families.)
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) async fn generate_candle_video_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<(DecodedVideo, &'static str, Value)> {
    let engine_id = candle_video_engine_id(&request.model).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{} is not a candle video engine", request.model))
    })?;
    let adapter = candle_video_adapter_label(engine_id);
    let repo = candle_video_repo(request, engine_id);
    // Mochi (epic 1788 / sc-11992) resolves its tier dir through the SHARED resolver both lanes use —
    // `SceneWorks/mochi-1-mlx` serves candle too (A6's `.scales`-detect seam ingests the mlx-affine
    // tiers 1:1), so the tier-dir semantics, the on-demand q8/bf16 fetch and the shared-parent
    // co-requisite are identical off-Mac. It must NOT fall through to the wan tier-select below: the
    // Mochi tier layout is `<root>/{q4|q8|bf16}/transformer/` with the T5/VAE/tokenizer as siblings of
    // the tier dir, which `candle_wan_tier_subdir` does not understand.
    //
    // Keyed off the RESOLVED engine id, mirroring the `is_ltx` binding below — the id is already
    // resolved through `candle_video_engine_id`, so re-deriving the family from the model string here
    // would be a second, drift-prone source of truth.
    let is_mochi = engine_id == "mochi_1";
    if is_mochi {
        validate_mochi_mode(request)?;
        // Refuse a job that cannot possibly fit BEFORE paying for its tier download (sc-12373, the
        // candle half of sc-12322). Runs on a weights FLOOR — the already-present shared
        // co-requisites — so it only ever refuses when the full gate below would refuse too.
        //
        // MUST stay ahead of the fetches: that ordering IS the story. Below this point the user has
        // already paid ~13-20 GiB for an answer that was knowable here. Pinned by
        // `generate_candle_video_using_refuses_before_paying_for_the_tier_download`.
        mochi_vram_precheck(
            &request.model,
            engine_id,
            mochi_precheck_dir(settings, request).as_deref(),
            request.raw_frame_count(),
            request.width,
            request.height,
            &settings.gpu_id,
            candle_video_vram_budget(settings).await,
        )?;
        ensure_mochi_q8_present(api, settings, job, request).await?;
        ensure_mochi_bf16_present(api, settings, job, request).await?;
    }
    // sc-14492: gate SVD before even resolving its already-installed snapshot. A full/default
    // 25-frame burst OOMed twice on real 32 GB hardware, while the reduced 8-frame / chunk-1 / 12-step
    // recipe completed. The returned recipe fields are the only ones the SVD generation arm can
    // consume, making this check load-bearing.
    let svd_preflight = if engine_id == "svd_xt" {
        Some(svd_vram_preflight(
            request,
            &settings.gpu_id,
            candle_video_vram_budget(settings).await,
        )?)
    } else {
        None
    };
    let snapshot_dir = if is_mochi {
        // Resolve-or-error; never a stub (the candle generic arm has no stub fallback once
        // `candle_video_engine_id` resolves the id).
        resolve_mochi_model_dir(settings, request)?
    } else {
        candle_video_snapshot_dir(settings, &repo)?
    };
    // Coerce the requested frame count onto the engine's temporal stride — the ONE shared ladder both
    // lanes use (sc-11992), so the candle stride can never drift from the MLX one. Computed HERE, above
    // the tier binding, because Mochi's fit gate (sc-12306) needs the coerced count: the decode peak is
    // linear in frames, so gating on the raw request would size the check against a length that never
    // renders. (The SVD arm below returns before this is read; it derives its own model-fixed burst.)
    let frames = video_frame_count(&request.model, request.raw_frame_count());
    // Packed-tier select: base LTX QLoRA/inference shares the turnkey `q4/` tier on every native
    // backend; Wan quant-matrix repos select q4/q8/bf16 below.
    // q4/q8/bf16 subdirs — resolve the one matching `advanced.mlxQuantize` (default q4) and load from it
    // (the packed-detect seam reads the baked-in quant). A flat/dense repo (no subdirs, e.g. the
    // `Wan-AI/*-Diffusers` fallback) stays as-is with no quant marker.
    let is_ltx = engine_id == "ltx_2_3_distilled";
    let (mut model_dir, wan_quant) = if is_mochi {
        // `resolve_mochi_model_dir` already returned the TIER dir. The VRAM fit gate (sc-12306) runs
        // here, and the quant marker comes back OUT of it — see `mochi_vram_preflight` for why the
        // marker is bundled into the gated return rather than read alongside a free-standing check.
        let MochiVramPreflight { quant } = mochi_vram_preflight(
            engine_id,
            &snapshot_dir,
            frames,
            request.width,
            request.height,
            &settings.gpu_id,
            candle_video_vram_budget(settings).await,
        )?;
        (snapshot_dir, quant)
    } else if let Some((dir, quant)) =
        candle_ltx_tier_subdir(&snapshot_dir, engine_id, &request.model)
    {
        (dir, quant)
    } else {
        let (dir, quant) = match candle_wan_tier_subdir(&snapshot_dir, engine_id, request) {
            Some((tier_dir, quant)) => (tier_dir, quant),
            None => (snapshot_dir, None),
        };
        // The Wan VRAM fit gate: the tier's MEASURED peak (sc-12402) when the manifest carries one,
        // else the sc-12344 weights floor. The non-Mochi half of this lane's admission check. Runs on
        // the RESOLVED tier (so it sizes the tier that will actually load, not the manifest's default —
        // the sc-12090 lesson; `candle_wan_tier_key` derives the manifest key from the resolver's own
        // marker for exactly that reason) and BEFORE `ensure_wan_lightning_present` below, so a
        // card that cannot fit the job is refused without first paying for the Lightning fetch. A no-op
        // for `ltx`/`svd`, which are exempt from THIS Wan gate — SVD already passed its bespoke
        // frame-aware hardware-class preflight above; see `vram_gate::wan_weight_components`.
        let dir = wan_vram_preflight(
            engine_id,
            &request.model_manifest_entry,
            candle_wan_tier_key(quant),
            dir,
            &settings.gpu_id,
            candle_video_vram_budget(settings).await,
        )?;
        (dir, quant)
    };
    // ltx needs the separate Gemma-3-12B encoder (its only conditioning input). Resolve its snapshot
    // dir here and thread it onto the LoadSpec below (sc-8827) instead of mutating `$LTX_GEMMA_DIR`.
    let ltx_gemma_dir = if is_ltx {
        super::ltx::resolve_bundled_ltx_gemma_dir(&model_dir)
            .or_else(|| resolve_ltx_gemma_dir(settings))
    } else {
        None
    };
    // Wan 14B I2V conditions on a source image (`Conditioning::Reference`); every other candle video
    // engine is txt2video-only (empty conditioning).
    let conditioning =
        resolve_candle_video_conditioning(settings, request, project_path, engine_id)?;

    // SVD-XT (sc-5493): image→video only — no prompt / negative / guidance (the engine uses its
    // frame-wise CFG ramp), a model-fixed burst (≤25 frames), the user `fps` as the playback cadence,
    // and the motion micro-conditioning knobs (motion_bucket_id / noise_aug_strength / decode_chunk /
    // conditioning_fps). Mirrors the MLX `generate_svd`; the conditioning is the source `Reference`
    // resolved above.
    if engine_id == "svd_xt" {
        let input = candle_svd_input(
            engine_id,
            model_dir,
            conditioning,
            request,
            svd_preflight.expect("svd_xt must pass its VRAM preflight before model resolution"),
        );
        let mut raw_settings = svd_raw_settings(request);
        if let Value::Object(map) = &mut raw_settings {
            map.insert("repo".to_owned(), Value::String(repo.clone()));
        }
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
        return Ok((decoded, adapter, raw_settings));
    }

    let is_wan = engine_id == "wan2_2_ti2v_5b"
        || engine_id == "wan2_2_t2v_14b"
        || engine_id == "wan2_2_i2v_14b";

    // Wan Lightning toggle + adapters (sc-10138): self-heal the A14B Lightning distill pair when the
    // toggle is on, then resolve the adapter specs (Lightning + user LoRAs, per-expert on the MoE). The
    // candle Wan engine applies these additively on a packed q4/q8 tier (candle-gen sc-10094/10095) or
    // folds them on a dense tier. `Vec::new()` for the non-Wan (ltx) engine.
    let adapters = if is_wan {
        ensure_wan_lightning_present(api, settings, job, request, engine_id).await?;
        resolve_wan_adapters(settings, request, engine_id)?
    } else {
        Vec::new()
    };
    if is_wan {
        let built_in = if wan_lightning_on(engine_id, request)
            && matches!(engine_id, "wan2_2_t2v_14b" | "wan2_2_i2v_14b")
        {
            2
        } else {
            0
        };
        let tier_key = candle_wan_tier_key(wan_quant);
        let user_adapter_bytes = wan_user_adapter_resident_bytes(
            adapters.get(built_in..).unwrap_or(&[]),
            matches!(engine_id, "wan2_2_t2v_14b" | "wan2_2_i2v_14b"),
            tier_key,
        )?;
        if user_adapter_bytes > 0 {
            // Final pre-load gate: keep the early base-only check above so a hopeless card still avoids
            // downloads, then re-read live VRAM after adapters resolve and charge their exact residual.
            model_dir = wan_vram_preflight_with_adapter_bytes(
                engine_id,
                &request.model_manifest_entry,
                tier_key,
                model_dir,
                user_adapter_bytes,
                &settings.gpu_id,
                candle_video_vram_budget(settings).await,
            )?;
        }
    }

    // Descriptor-narrowed sampling surface: wan (5B + 14B) takes guidance + a negative prompt; the
    // distilled ltx takes neither (single-stage, no CFG). Wan uses the Lightning-aware recipe
    // ([`wan_sampling`]: 4-step/CFG-off when the toggle is on, else native multi-step + CFG);
    // ltx keeps its own step default and no CFG.
    //
    // Mochi needs its OWN arm (sc-11992) — it is not distilled, so it takes true CFG (negative prompt
    // + guidance), but it is also not a Wan model: falling through to `wan_sampling` would hit
    // that function's dense-5B tail and force `WAN5B_INTERIM_STEPS` (20) on it, silently
    // overriding the AsymmDiT's own 64-step default with a Wan tuning constant. `None` ⇒ the engine's
    // DEFAULT_STEPS (64) / DEFAULT_GUIDANCE (4.5) stand.
    let (steps, guidance, negative_prompt) = if is_ltx {
        (advanced_opt_u32(request, "steps"), None, None)
    } else if is_mochi {
        (
            advanced_opt_u32(request, "steps"),
            advanced_opt_f32(request, "guidanceScale"),
            non_empty_negative_prompt(request),
        )
    } else {
        let (steps, guidance) = wan_sampling(engine_id, request);
        (steps, guidance, non_empty_negative_prompt(request))
    };
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        // Wan quant-matrix tier marker (sc-10027): `Some(Q4/Q8)` when a packed candle tier subdir was
        // resolved, else `None` (bf16 tier / dense repo / ltx). A no-op on the candle wan load (the
        // packed-detect seam reads the tier's baked-in quant), carried for the LoadSpec + asset record.
        quant: wan_quant,
        adapters,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames,
        fps: request.fps,
        steps,
        guidance,
        seed: resolve_video_seed(request) as u64,
        // ltx's Gemma-3 encoder dir rides the LoadSpec (sc-8827); `None` for wan (bundled TE).
        text_encoder_dir: ltx_gemma_dir,
        // The candle A14B renders with sequential component offload + MoE expert-swap (sc-12631): the
        // residency the measured `candle.vramGbByTier` peak was taken under, so the gate's admission
        // number is only truthful when the load actually takes it. The 5B and SVD-XT also take their
        // own sequential lifecycles; only LTX remains resident.
        offload_policy: candle_video_offload_policy(engine_id),
        ..VideoGenInput::default()
    };
    let raw_settings = candle_video_raw_settings(request, &repo);
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
    Ok((decoded, adapter, raw_settings))
}

// ---------------------------------------------------------------------------
// Candle (Windows/CUDA) in-place ComfyUI Wan2.2 base generation (epic 10451 Phase 2c, sc-10671): the
// video sibling of the z-image/qwen ComfyUI base image lanes. Reads a user's two ComfyUI Wan A14B
// expert files in place (native-Wan keys + companion scaled-fp8), remapped + dequant'd off-Mac via
// `runtime_cuda::providers::wan::load_from_comfyui_experts`. The UMT5 TE + VAE are read in place too when the tree
// carries them (sc-10909, folded into `components[]` by the API); the tokenizer (and either component
// when absent) comes from a resident `SceneWorks/wan2.2-*-candle` snapshot tier. T2V only for now
// (I2V's channel-concat reference conditioning is a follow-up); the model id is an `external_base_*`
// catalog row.
// ---------------------------------------------------------------------------

/// The candle Wan2.2 T2V-A14B engine id the ComfyUI base experts load into.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const WAN_COMFYUI_T2V_ENGINE: &str = "wan2_2_t2v_14b";

/// The engine label recorded on candle ComfyUI Wan assets + telemetry.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const WAN_COMFYUI_CANDLE_ADAPTER: &str = "candle_wan_comfyui";

/// The resident Wan2.2 T2V-A14B snapshot repo supplying the dense UMT5 TE / VAE / tokenizer (the
/// experts are read from the ComfyUI tree). Any complete tier subdir serves — the TE/VAE stay dense
/// across tiers; only the transformer (which we don't use here) is quantized.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const WAN_COMFYUI_SNAPSHOT_REPO: &str = "SceneWorks/wan2.2-t2v-a14b-candle";

/// Tier subdirs probed for the dense TE/VAE/tokenizer (first fully-present tree wins).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const WAN_COMFYUI_SNAPSHOT_TIERS: &[&str] = &["q8", "q4", "bf16"];

/// Wan T2V DiT `patch_embedding` in-channels (16 latent). I2V is channel-concat (36) and needs the
/// reference-conditioning lane, so this slice serves only T2V experts.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const WAN_T2V_IN_CHANNELS: u64 = 16;

/// The two in-place ComfyUI expert files + the resident snapshot tier, plus the optional in-place UMT5
/// TE / VAE files (sc-10909). The snapshot tier always supplies the tokenizer (and the TE/VAE when their
/// files are absent).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) struct ComfyuiWanPaths {
    high: PathBuf,
    low: PathBuf,
    /// In-place UMT5 TE (`text_encoder` component), confined; `None` ⇒ snapshot `text_encoder/`.
    te: Option<PathBuf>,
    /// In-place Wan VAE (`vae` component), confined; `None` ⇒ snapshot `vae/`.
    vae: Option<PathBuf>,
    snapshot_dir: PathBuf,
}

#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
impl ComfyuiWanPaths {
    pub(super) fn test_fixture(
        high: PathBuf,
        low: PathBuf,
        te: Option<PathBuf>,
        vae: Option<PathBuf>,
        snapshot_dir: PathBuf,
    ) -> Self {
        Self {
            high,
            low,
            te,
            vae,
            snapshot_dir,
        }
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn comfyui_wan_admission_bytes(paths: &ComfyuiWanPaths) -> u64 {
    let file_bytes = |path: &Path| path.metadata().map(|m| m.len()).unwrap_or(0);
    file_bytes(&paths.high)
        .saturating_add(file_bytes(&paths.low))
        .saturating_add(match &paths.te {
            Some(path) => file_bytes(path),
            None => {
                crate::mlx_fit_gate::sum_safetensors_bytes(&paths.snapshot_dir.join("text_encoder"))
            }
        })
        .saturating_add(match &paths.vae {
            Some(path) => file_bytes(path),
            None => crate::mlx_fit_gate::sum_safetensors_bytes(&paths.snapshot_dir.join("vae")),
        })
}

/// Conservative admission for external ComfyUI A14B. These user-supplied experts have no builtin
/// manifest tier/measurement, so the only authoritative pre-load signal is their actual file set.
/// Sequential expert swapping is explicit at the load seam below, but a 24–32 GB card must not admit
/// a pair whose on-disk weights alone already exceed its live budget.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn comfyui_wan_vram_preflight(
    paths: ComfyuiWanPaths,
    gpu_id: &str,
    budget: Option<crate::vram_gate::VramBudget>,
) -> WorkerResult<ComfyuiWanPaths> {
    let Some(budget) = budget else {
        return Ok(paths);
    };
    let bytes = comfyui_wan_admission_bytes(&paths);
    if bytes == 0 {
        return Ok(paths);
    }
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const CUDA_HEADROOM_GB: f64 = 2.0;
    let needed = bytes as f64 / GIB + CUDA_HEADROOM_GB;
    if budget.free_gb + f64::EPSILON < needed {
        return Err(WorkerError::InvalidPayload(format!(
            "ComfyUI Wan2.2 A14B needs at least ~{} GB of VRAM admission budget for its supplied \
             experts and supporting weights, but GPU {gpu_id} has ~{} GB available. Use smaller \
             quantized ComfyUI experts, close other GPU workloads, or run on a larger GPU.",
            needed.ceil() as u64,
            budget.free_gb.round() as u64,
        )));
    }
    Ok(paths)
}

/// Peek a safetensors header (8-byte length + JSON) and return the `patch_embedding.weight`
/// in-channels (`shape[1]`) — the T2V(16)/I2V(36) discriminator. `None` on any read/parse miss.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn wan_expert_in_channels(path: &Path) -> Option<u64> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).ok()?;
    let header_len = u64::from_le_bytes(len) as usize;
    // Guard against a corrupt/huge length before allocating (headers are KB-scale).
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        return None;
    }
    let mut buf = vec![0u8; header_len];
    file.read_exact(&mut buf).ok()?;
    let header: Value = serde_json::from_slice(&buf).ok()?;
    header
        .get("patch_embedding.weight")?
        .get("shape")?
        .as_array()?
        .get(1)?
        .as_u64()
}

/// Resolve the ComfyUI Wan expert paths + a resident snapshot tier from the forwarded `external_base_*`
/// row. `Ok(None)` (router falls through) when this is not a runnable ComfyUI Wan T2V job: wrong family,
/// not usable, missing an expert component, no resident Wan snapshot, or an I2V expert (36-channel — the
/// reference-conditioning lane is deferred). Each expert path is confined by
/// `normalize_app_managed_model_path` (the sc-10668-widened external-roots allow-list); the snapshot dir
/// is a fixed-repo/cache path (never payload-derived), so it needs no confinement.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_wan_comfyui_paths(
    request: &VideoRequest,
    settings: &Settings,
) -> WorkerResult<Option<ComfyuiWanPaths>> {
    let entry = &request.model_manifest_entry;
    if entry.get("family").and_then(Value::as_str) != Some("wan-video") {
        return Ok(None);
    }
    if entry.get("usable").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    let Some(components) = entry.get("components").and_then(Value::as_array) else {
        return Ok(None);
    };
    let path_for = |role: &str| -> Option<&str> {
        components
            .iter()
            .find(|component| component.get("role").and_then(Value::as_str) == Some(role))
            .and_then(|component| component.get("path").and_then(Value::as_str))
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };
    let (Some(high), Some(low)) = (path_for("transformer_high"), path_for("transformer_low"))
    else {
        return Ok(None);
    };
    let Some(snapshot_root) =
        huggingface_snapshot_dir(&settings.data_dir, WAN_COMFYUI_SNAPSHOT_REPO)
    else {
        return Ok(None);
    };
    let Some(snapshot_dir) = WAN_COMFYUI_SNAPSHOT_TIERS
        .iter()
        .map(|tier| snapshot_root.join(tier))
        .find(|dir| {
            dir.join("text_encoder").is_dir()
                && dir.join("vae").is_dir()
                && dir.join("tokenizer").join("tokenizer.json").is_file()
        })
    else {
        return Ok(None);
    };
    let high = crate::paths::normalize_app_managed_model_path(
        settings,
        high,
        "ComfyUI Wan high-noise expert",
    )?;
    let low = crate::paths::normalize_app_managed_model_path(
        settings,
        low,
        "ComfyUI Wan low-noise expert",
    )?;
    // T2V only: an I2V expert (36 in-channels) would load into the wrong config; decline it here (the
    // channel-concat reference lane is a follow-up) rather than surface a shape error at generate.
    if wan_expert_in_channels(&high) != Some(WAN_T2V_IN_CHANNELS) {
        return Ok(None);
    }
    // Optional in-place UMT5 TE + Wan VAE (sc-10909): the API folds them into `components[]` as
    // `text_encoder` / `vae` when the tree carries them; each is confined like the experts. Absent ⇒
    // `None` ⇒ the resident snapshot tier supplies that component (the row is complete either way).
    let te = path_for("text_encoder")
        .map(|te| {
            crate::paths::normalize_app_managed_model_path(settings, te, "ComfyUI Wan UMT5 encoder")
        })
        .transpose()?;
    let vae = path_for("vae")
        .map(|vae| crate::paths::normalize_app_managed_model_path(settings, vae, "ComfyUI Wan VAE"))
        .transpose()?;
    Ok(Some(ComfyuiWanPaths {
        high,
        low,
        te,
        vae,
        snapshot_dir,
    }))
}

/// True when this is a candle-runnable in-place ComfyUI Wan2.2 T2V job: an `external_base_*` model whose
/// forwarded row is a usable wan-video with both expert components + a resident snapshot. Mirrors the
/// image comfyui availability predicates.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn wan_comfyui_available(request: &VideoRequest, settings: &Settings) -> bool {
    request.model.starts_with("external_base_")
        && matches!(resolve_wan_comfyui_paths(request, settings), Ok(Some(_)))
}

/// Real candle in-place ComfyUI Wan2.2 T2V generation: resolve + confine the two expert paths and the
/// snapshot tier, then drive the shared [`generate_video`] funnel with `input.comfyui` set (the bespoke
/// uncached `load_from_comfyui_experts` path). Non-distilled base — native multi-step + per-expert CFG
/// (guidance `None` ⇒ the engine's per-expert defaults); no Lightning distill (the base lane folds no
/// adapters).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) async fn generate_candle_wan_comfyui(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, &'static str, Value)> {
    let _ = project_path;
    let paths = resolve_wan_comfyui_paths(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(
            "ComfyUI Wan components could not be resolved (family/usable/experts/snapshot)"
                .to_owned(),
        )
    })?;
    // Admission happens before `generate_video` reaches the uncached external-expert load.
    let paths = comfyui_wan_vram_preflight(
        paths,
        &settings.gpu_id,
        candle_video_vram_budget(settings).await,
    )?;
    let input = VideoGenInput {
        engine_id: WAN_COMFYUI_T2V_ENGINE,
        // The snapshot tier supplies the tokenizer (and the metrics `model_dir`, and the UMT5 TE / VAE
        // when their tree files are absent); the experts — and the TE/VAE when present — are read in
        // place via `comfyui` below (sc-10909).
        model_dir: paths.snapshot_dir.clone(),
        prompt: request.prompt.clone(),
        negative_prompt: non_empty_negative_prompt(request),
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        // Non-Lightning base: honor a requested step count, else the engine default; per-expert CFG
        // defaults (guidance `None`). No adapters — the ComfyUI base lane does not fold LoRAs.
        steps: advanced_opt_u32(request, "steps"),
        guidance: None,
        seed: resolve_video_seed(request) as u64,
        conditioning: Vec::new(),
        offload_policy: candle_video_offload_policy(WAN_COMFYUI_T2V_ENGINE),
        comfyui: Some(ComfyuiWanExperts::new(
            paths.high, paths.low, paths.te, paths.vae, false,
        )),
        ..VideoGenInput::default()
    };
    let raw_settings = candle_video_raw_settings(request, WAN_COMFYUI_SNAPSHOT_REPO);
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    Ok((decoded, WAN_COMFYUI_CANDLE_ADAPTER, raw_settings))
}

// ---------------------------------------------------------------------------
// Candle (Windows/CUDA) SCAIL-2 generation (sc-6837, epic 6563): the off-Mac sibling of the macOS
// `generate_scail2` / `generate_scail2_replace` (epic 5439). Same end-to-end shape — a reference
// character image + a driving video → an animated clip (`animate_character`), or cross-identity
// `replace_person` (engine `replace_flag`) over the saved YOLO11 → ByteTrack → SAM3 person track. The
// worker paints the color-coded masks from the candle SAM3 segmenter (`person_segment_sam3_candle`);
// the painters (`scail2_masks`) are shared with the MLX lane. A distinct candle engine, NOT VACE — no
// torch fallback (`crate::inference_runtime::load("scail2_14b")` resolves the `candle_gen_scail2` provider, sc-6836).
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real candle SCAIL-2 asset (the candle sibling of `mlx_scail2`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const CANDLE_SCAIL2_ADAPTER: &str = "candle_scail2";

/// Resolve candle SCAIL-2 weights from an explicit override, the exact Model Manager-installed
/// `SceneWorks/scail2-mlx` bf16 tier, or the legacy app-managed candle component tree. Both native
/// engines consume the same six bf16 files; q4/q8 stay MLX-only because their DiT tensors use MLX
/// packing. Every candidate is classified by the pinned provider's own fail-closed layout predicate.
/// When none is complete, a missing checkpoint surfaces a clear repair error
/// instead of degrading to a stub (a character animation / replacement must never silently produce
/// meaningless output). The provider's `load` then validates each subdir and reports the precise gap.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_scail2_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(dir) = std::env::var("SCENEWORKS_CANDLE_SCAIL2_DIR") {
        let path = PathBuf::from(dir.trim());
        if !dir.trim().is_empty() {
            runtime_cuda::providers::scail2::snapshot_layout(&path).map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "scail2 (candle): $SCENEWORKS_CANDLE_SCAIL2_DIR points to an incomplete snapshot at {}: {error}",
                    path.display()
                ))
            })?;
            return Ok(path);
        }
    }

    resolve_managed_candle_scail2_model_dir(settings)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn resolve_managed_candle_scail2_model_dir(
    settings: &Settings,
) -> WorkerResult<PathBuf> {
    if let Some(snapshot) = crate::model_jobs::huggingface_pinned_snapshot_dir(
        &settings.data_dir,
        SCAIL2_REPO,
        SCAIL2_REVISION,
    ) {
        let bf16 = snapshot.join("bf16");
        if runtime_cuda::providers::scail2::snapshot_layout(&bf16).is_ok() {
            return Ok(bf16);
        }
    }

    // Retain the pre-Model-Manager manual component layout as a compatibility fallback.
    let managed = settings
        .data_dir
        .join("models")
        .join("candle")
        .join("scail2");
    if runtime_cuda::providers::scail2::snapshot_layout(&managed).is_ok() {
        return Ok(managed);
    }
    Err(WorkerError::InvalidPayload(format!(
        "scail2 (candle): the shared SCAIL-2 bf16 package is not installed or is incomplete. \
         Install the bf16 tier from Model Manager ({SCAIL2_REPO} at {SCAIL2_REVISION}), repair that \
         download, or set $SCENEWORKS_CANDLE_SCAIL2_DIR to a complete shared/legacy snapshot. \
         Legacy fallback checked: {}.",
        managed.display(),
    )))
}

/// `true` if any resolved adapter is a lightx2v diff-patch ("lightning") LoRA — the engine's own
/// detector (a file carrying full-rank `.diff`/`.diff_b` tensors), so the recipe keys off the actual
/// format, not a catalog id or filename. A file that can't be read is treated as non-lightning (the
/// engine surfaces the real load error downstream). The candle sibling of the macOS
/// `scail2_adapters_have_lightning`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_scail2_adapters_have_lightning(adapters: &[AdapterSpec]) -> bool {
    adapters
        .iter()
        .any(|a| runtime_cuda::providers::scail2::has_diff_patch_keys(&a.path).unwrap_or(false))
}

/// Resolve a candle SCAIL-2 `animate_character` request into the engine conditioning — the candle
/// sibling of the macOS `resolve_scail2_conditioning`. Loads the reference character image + the
/// driving clip, segments both with the candle SAM3 PCS segmenter (every person → a palette color,
/// left-to-right), paints the color-coded masks (animation: reference bg white, driving bg black), and
/// assembles `Reference` + `Mask` + `ControlClip`. Segmentation + painting run on the blocking pool.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn resolve_candle_scail2_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    // The character: a reference image (referenceAssetIds first, else the i2v sourceAssetId).
    let ref_id = request
        .reference_asset_ids
        .first()
        .map(String::as_str)
        .or(request.source_asset_id.as_deref())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "scail2 animate_character requires a reference character image (referenceAssetIds \
                 or sourceAssetId)."
                    .into(),
            )
        })?;
    let reference = crate::image_jobs::load_reference_image(
        &settings.data_dir,
        &request.project_id,
        ref_id,
        project_path,
    )?;

    // The driving video → frames at the output size (the engine re-resizes internally). Reuses the
    // candle Wan-VACE source-clip loader (`load_source_video_frames`, which reads `sourceClipAssetId`
    // and aspect-fits to W×H) — the candle-lane sibling of the macOS path's `extract_clip_frames`
    // (that helper is macOS-only).
    if request
        .source_clip_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .is_none()
    {
        return Err(WorkerError::InvalidPayload(
            "scail2 animate_character requires a driving video (sourceClipAssetId).".into(),
        ));
    }
    let driving = load_source_video_frames(
        api,
        settings,
        job,
        request,
        project_path,
        wan_frame_count(request.raw_frame_count()) as usize,
    )
    .await?;

    // SAM3 segmenter weights, resolved from the Model Manager install (sc-17629), shared by both
    // segmentation passes.
    let (sam_model, sam_tokenizer) =
        crate::person_segment_sam3_candle::require_segmenter_weights(settings)?;

    // Segment + paint via the shared orchestrator (sc-8830). Animation keeps the reference's world
    // (ref bg white) and drops the driving world (driving bg black); the candle SAM3 module is the
    // off-Mac twin, whose per-frame propagate contract (sc-8972) observes the tripped cancel flag
    // between frames.
    let (rm, rt) = (sam_model.clone(), sam_tokenizer.clone());
    assemble_scail2_animate_conditioning(
        api,
        settings,
        &job.id,
        move |flag| {
            let masks = crate::person_segment_sam3_candle::segment_all_persons_in_memory(
                &rm,
                &rt,
                std::slice::from_ref(&reference),
                Some(flag),
                None,
            )?;
            let mask =
                crate::scail2_masks::paint_reference_mask(&masks, crate::scail2_masks::BG_WHITE)?;
            Ok((reference, mask))
        },
        move |flag| {
            let masks = crate::person_segment_sam3_candle::segment_all_persons_in_memory(
                &sam_model,
                &sam_tokenizer,
                &driving,
                Some(flag),
                Some(Box::new(|frame, total| {
                    tracing::debug!(event = "scail2_sam3_propagate_progress", frame, total);
                })),
            )?;
            let masks =
                crate::scail2_masks::paint_driving_masks(&masks, crate::scail2_masks::BG_BLACK)?;
            Ok((driving, masks))
        },
    )
    .await
}

/// Real candle SCAIL-2 character animation (sc-6837 + sc-6838): build the `VideoGenInput` and run the
/// shared `generate_video` path. `animate_character` → engine task `animation`; the source media becomes
/// the SAM3-painted conditioning. Inference LoRA / LoKr / LoHa + the Bias-Aware DPO LoRA + the lightx2v
/// lightning diff-patch resolve from `request.loras` and merge into the dense DiT (sc-6838); a lightning
/// LoRA also applies the step-distill recipe (8 steps / CFG-off / shift 1). Frame count uses the Wan
/// 1-mod-4 stride (the renderer is Wan2.1); the engine stitches > 81-frame clips into overlapping segments.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) async fn generate_candle_scail2(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, &'static str, Value)> {
    let Scail2ColdLoadPlan {
        model_dir,
        admission,
    } = scail2_cold_load_plan(
        &request.model_manifest_entry,
        resolve_candle_scail2_model_dir(settings)?,
        &settings.gpu_id,
    );
    let negative_prompt = non_empty_negative_prompt(request);
    let conditioning =
        resolve_candle_scail2_conditioning(api, settings, job, request, project_path).await?;
    // Inference adapters (DPO / lightning / user LoRA) + the lightning step-distill recipe.
    let adapters = resolve_scail2_adapters(settings, request)?;
    let lightning = candle_scail2_adapters_have_lightning(&adapters);
    let (steps, guidance, scheduler_shift) = scail2_sampling(request, lightning);
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        adapters,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps,
        guidance,
        scheduler_shift,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(scail2_engine_video_mode(&request.mode).to_owned()),
        cold_load_admission: Some(admission),
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    Ok((
        decoded,
        CANDLE_SCAIL2_ADAPTER,
        scail2_raw_settings(request, lightning),
    ))
}

/// Resolve a candle SCAIL-2 `replace_person` request into cross-identity replacement conditioning — the
/// candle sibling of the macOS `resolve_scail2_replace_conditioning` (sc-5452). Reuses the masks
/// SceneWorks already computed: the saved person track (YOLO11 → ByteTrack → SAM3, corrections applied)
/// supplies the per-frame driving masks; the character's approved reference is the identity. Driving
/// frames load exactly as the candle Wan-VACE backend loads them (`load_source_video_frames`) so the
/// resampled track masks stay frame-aligned 1:1. Replacement keeps the driving clip's world (driving
/// mask bg white, reference mask bg black); `video_mode = "replacement"` flips the engine `replace_flag`.
/// SCAIL-2 replaces the whole tracked person, so face_only/full_person + maskingStrength are inert.
/// Returns the conditioning plus the honest `replacementStatus`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn resolve_candle_scail2_replace_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<(Vec<Conditioning>, Value)> {
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

    // Driving frames + their per-frame binary person masks — the same source the candle Wan-VACE
    // backend consumes, loaded identically so the resampled masks align 1:1 with the frames.
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let driving =
        load_source_video_frames(api, settings, job, request, project_path, frame_count).await?;
    let frame_total = driving.len();
    let (binary_masks, mask_mode) = crate::person_replace::person_track_masks(
        project_path,
        &track,
        request.width,
        request.height,
        frame_total,
    )?;
    // The tracked person → blue (person 0); replacement keeps the driving's world → white bg.
    let driving_masks = crate::scail2_masks::paint_track_driving_masks(
        &binary_masks,
        crate::scail2_masks::BG_WHITE,
    );

    // The character identity: the first approved reference image (multi-ref = the engine-contract
    // extension, sc-5583 on the MLX side).
    let references = resolve_character_references(settings, request, project_path)?;
    let reference_count = references.len();
    let reference = references.into_iter().next().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Replace Person requires at least one approved character reference image.".to_owned(),
        )
    })?;

    // The reference color mask: a candle SAM3 pass on the reference image → the primary person painted
    // blue on a black background (replacement discards the reference's surrounding world).
    let (sam_model, sam_tokenizer) =
        crate::person_segment_sam3_candle::require_segmenter_weights(settings)?;
    // Heartbeat keepalive + user cancel across the cold SAM3 parse + single-frame propagate
    // (sc-8390 / sc-8807), via the shared blocking-segment helper (sc-8830); the engine's per-frame
    // propagate contract (sc-8972) observes the tripped flag between frames, beyond the coarse seams
    // (cold parse / model build).
    let ref_mask = scail2_segment_blocking(
        api,
        settings,
        &job.id,
        "scail2 reference segment task",
        move |flag| {
            let masks = crate::person_segment_sam3_candle::segment_all_persons_in_memory(
                &sam_model,
                &sam_tokenizer,
                std::slice::from_ref(&reference),
                Some(flag),
                None,
            )?;
            let mask =
                crate::scail2_masks::paint_reference_mask(&masks, crate::scail2_masks::BG_BLACK)?;
            Ok((mask, reference))
        },
    )
    .await?;
    let (ref_mask, reference) = ref_mask;

    let conditioning = vec![
        Conditioning::Reference {
            image: reference,
            strength: None,
        },
        Conditioning::Mask { image: ref_mask },
        Conditioning::ControlClip {
            frames: driving,
            mask: driving_masks,
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        },
    ];
    let status = replacement_status_value(
        &track,
        track_id,
        mask_mode,
        1.0,
        reference_count,
        frame_total,
        CANDLE_SCAIL2_ADAPTER,
    );
    Ok((conditioning, status))
}

/// Real candle SCAIL-2 cross-identity replacement (sc-6837): the candle sibling of the MLX
/// `generate_scail2_replace`. Builds the replacement conditioning from the saved person track +
/// character reference and runs the shared `generate_video` path with `video_mode = "replacement"`
/// (engine `replace_flag = true`). Returns the decoded video + the honest `replacementStatus`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) async fn generate_candle_scail2_replace(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    let Scail2ColdLoadPlan {
        model_dir,
        admission,
    } = scail2_cold_load_plan(
        &request.model_manifest_entry,
        resolve_candle_scail2_model_dir(settings)?,
        &settings.gpu_id,
    );
    let negative_prompt = non_empty_negative_prompt(request);
    let (conditioning, status) =
        resolve_candle_scail2_replace_conditioning(api, settings, job, request, project_path)
            .await?;
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some("replacement".to_owned()),
        cold_load_admission: Some(admission),
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    Ok((decoded, status))
}

/// Resolve the candle Wan-VACE diffusers snapshot dir (sc-5494): `SCENEWORKS_CANDLE_WAN_VACE_DIR`
/// override (when it holds a `transformer/config.json`), else the HF [`CANDLE_WAN_VACE_REPO`] snapshot.
/// Errors loudly when absent (no stub fallback — a missing VACE checkpoint surfaces a re-download error).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_wan_vace_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(dir) = std::env::var("SCENEWORKS_CANDLE_WAN_VACE_DIR") {
        let path = PathBuf::from(dir.trim());
        if path.join("transformer").join("config.json").is_file() {
            return Ok(path);
        }
    }
    candle_video_snapshot_dir(settings, CANDLE_WAN_VACE_REPO)
}

/// Windows/CUDA candle Wan-VACE `replace_person` (sc-5494): the candle sibling of the MLX
/// [`generate_wan_vace`]. Resolves the diffusers VACE snapshot, extracts the source-clip control frames,
/// builds the per-frame person mask from the saved track + the character references, and runs the
/// `wan_vace` engine. Person detect/track/segment stays upstream (the masks are pre-saved); the
/// conditioning builders are shared with the MLX path. No quant / LoRA (the candle VACE provider rejects
/// them). Returns the decoded clip + the honest `replacementStatus`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) async fn generate_candle_wan_vace(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    let model_dir = resolve_candle_wan_vace_model_dir(settings)?;
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

    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
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
    let negative_prompt = non_empty_negative_prompt(request);
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "wan_vace",
        model_dir,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: frame_count as u32,
        fps: request.fps,
        steps: advanced_opt_u32(request, "steps"),
        guidance: advanced_opt_f32(request, "guidanceScale"),
        seed: resolve_video_seed(request) as u64,
        control_scale: Some(advanced::f32(&request.advanced, "conditioningScale", 1.0)),
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    let status = replacement_status_value(
        &track,
        track_id,
        mask_mode,
        masking_strength,
        reference_count,
        frame_total,
        CANDLE_WAN_VACE_ADAPTER,
    );
    Ok((decoded, status))
}

/// Windows/CUDA candle Wan-VACE `extend_clip` / `video_bridge` (sc-5494): the candle sibling of the MLX
/// [`generate_wan_vace_extend_bridge`]. Loads the real source-clip anchor frames (the left clip's tail
/// for extend; both clips' boundaries for bridge), builds the source-at-kept-positions + generated-span
/// ControlClip, and runs the `wan_vace` engine. No reference images, no quant / LoRA.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) async fn generate_candle_wan_vace_extend_bridge(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let model_dir = resolve_candle_wan_vace_model_dir(settings)?;
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let anchor = extend_anchor_frames(request, frame_count);
    let left_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} requires a source clip (sourceClipAssetId).",
            request.mode.replace('_', " ")
        ))
    })?;
    let left_anchor = load_clip_anchor_frames(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        left_id,
        request.width,
        request.height,
        anchor,
        ClipFramePosition::Last,
    )
    .await?;
    let right_anchor = if request.mode == "video_bridge" {
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
            load_clip_anchor_frames(
                api,
                settings,
                job,
                &request.project_id,
                project_path,
                right_id,
                request.width,
                request.height,
                anchor,
                ClipFramePosition::First,
            )
            .await?,
        )
    } else {
        None
    };
    let conditioning = build_extend_bridge_vace_conditioning(
        request,
        request.width,
        request.height,
        frame_count,
        left_anchor,
        right_anchor,
    )?;
    let negative_prompt = non_empty_negative_prompt(request);
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "wan_vace",
        model_dir,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: frame_count as u32,
        fps: request.fps,
        steps: advanced_opt_u32(request, "steps"),
        guidance: advanced_opt_f32(request, "guidanceScale"),
        seed: resolve_video_seed(request) as u64,
        control_scale: Some(advanced::f32(&request.advanced, "conditioningScale", 1.0)),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}

#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod adapter_fit_tests {
    use super::wan_user_adapter_resident_bytes;
    use gen_core::{AdapterKind, AdapterSpec, MoeExpert};
    use std::path::PathBuf;

    fn adapter(path: PathBuf, expert: Option<MoeExpert>) -> AdapterSpec {
        AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: expert,
        }
    }

    #[test]
    fn wan_user_adapter_bytes_follow_dense_single_and_moe_peak_semantics() {
        let root = tempfile::tempdir().unwrap();
        let shared = root.path().join("shared.safetensors");
        let high = root.path().join("high.safetensors");
        let low = root.path().join("low.safetensors");
        std::fs::write(&shared, vec![0_u8; 100]).unwrap();
        std::fs::write(&high, vec![0_u8; 300]).unwrap();
        std::fs::write(&low, vec![0_u8; 200]).unwrap();
        let stack = vec![
            adapter(shared, None),
            adapter(high, Some(MoeExpert::High)),
            adapter(low, Some(MoeExpert::Low)),
        ];
        assert_eq!(
            wan_user_adapter_resident_bytes(&stack, true, "q4").unwrap(),
            400,
            "shared is counted once and the sequential expert peak takes max(high, low)"
        );
        assert_eq!(
            wan_user_adapter_resident_bytes(&stack, false, "q4").unwrap(),
            600,
            "the single-expert 5B load retains the full stack"
        );
        assert_eq!(
            wan_user_adapter_resident_bytes(&stack, true, "bf16").unwrap(),
            0
        );

        let missing = vec![adapter(root.path().join("missing.safetensors"), None)];
        assert!(wan_user_adapter_resident_bytes(&missing, true, "q4").is_err());
    }
}
