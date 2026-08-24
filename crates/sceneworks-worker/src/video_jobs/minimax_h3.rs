#[allow(unused_imports)]
use super::prelude::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use super::wan::{advanced_opt_f32, advanced_opt_u32, generate_video_using, VideoGenInput};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::video_request::{classify_reference_set, ReferenceSetVerdict};

// ---------------------------------------------------------------------------
// MiniMax-H3 / Hailuo 3.0 (epic 17137, sc-19508): a joint audio+video family that emits video AND
// synchronized stereo audio in ONE denoise pass. MLX runs on macOS; off-Mac Candle dispatch is
// available for direct/replay execution while user routing remains deliberately disabled.
//
// Four facts shape this whole block, and every one of them DEVIATES from the Mochi/Krea template it
// otherwise mirrors:
//
//  1. **ONE engine id, TWO catalog entries.** `mlx-gen-minimax-h3` registers a single
//     `MODEL_ID = "minimax_h3"`. `minimax_h3` and `minimax_h3_ref` are not two providers — they are
//     two DiT *partition directories* of one provider (`MINIMAX_H3_PARTITIONS`, sc-19078), and the
//     provider picks between them from the CONDITIONING, not from a request flag. sc-19508's own
//     description said "the `transformer_ref` provider"; there is no such provider, and writing a
//     second engine id would have produced an unknown-model load error.
//
//  2. **The partition is derived, so the SHAPE is the safety property.** Upstream's
//     `MiniMaxH3Task::resolve(has_keyframes, has_references)` maps (false,false) ⇒ t2va
//     (`transformer/`), (true,false) ⇒ fl2va (`transformer/`), (false,true) ⇒ ref2va
//     (`transformer_ref/`), and refuses (true,true). The two checkpoints ship the SAME
//     `config.json` and the SAME 638 tensor names — only the values differ — so a request routed at
//     the wrong one RUNS, produces plausible video, and is wrong. That is why
//     [`minimax_h3_validate_partition`] exists: it is the only thing standing between a catalog id
//     and a silently-wrong checkpoint.
//
//  3. **The load probes BOTH partitions**, so each entry DOWNLOADS both (sc-19573).
//     `mlx-gen-minimax-h3::load` requires `{dit}/config.json` AND its sibling
//     `{dit}/../transformer_ref/config.json`, because ref2va is a first-class task of the engine
//     rather than an optional extra. Until sc-19573 the manifest shipped the two partitions as the
//     PRIMARY downloads of two separate catalog entries, so installing either alone produced an
//     install that could not load at all. Each entry now declares the sibling partition as a
//     per-tier `coRequisite`, which is what a co-requisite means — a pre-download weights floor. The
//     +18.78 GB (q4) that buys is not optional and never was; the pre-sc-19573 catalog comment
//     arguing it should stay optional was reasoning from the premise that one partition could load
//     on its own, which the engine has never allowed. [`resolve_minimax_h3_load`] still names which
//     half is missing, because a torn download remains reachable.
//
//  4. **No guidance, no negative prompt.** The checkpoint is guidance-distilled: there is no
//     unconditional branch anywhere in it. The manifest declares
//     `video.supportsGuidance = false` / `supportsNegativePrompt = false` and the engine REJECTS
//     both fields, so this arm passes `None` for each rather than forwarding a knob the engine
//     would refuse.
//
// Installed layout — the DiT partitions and (since sc-19120) the text encoder are the TIERED
// components, and they live in a DIFFERENT repo from the rest of the shared floor:
//
//     <tier root>/{q4,q8}/{transformer,transformer_ref,text_encoder}/ <- SceneWorks/minimax-h3-mlx
//     <tier root>/bf16/{transformer,transformer_ref}/                 <- SceneWorks/minimax-h3-mlx
//     <base root>/{tokenizer,vae,audio_vae,FL2VA/audio_vae}/          <- MiniMaxAI/MiniMax-H3 (coReq)
//     <base root>/text_encoder/                                       <- MiniMaxAI (bf16 tier ONLY)
//
// so `spec.weights` is the BASE root and the tier's `transformer/` is staged as the
// [`MINIMAX_H3_DIT_COMPONENT`]. Handing the loader the tier root instead would fail on
// `tokenizer/`, and handing it the base root alone would load the unquantized upstream DiT.
//
// The TEXT ENCODER is the one shared component whose root moves with the tier (sc-19120): q4 and q8
// read the packed encoder beside the DiT tiers, bf16 reads the dense upstream one. That is why the
// probe list and the root resolution live together in
// `sceneworks_core::mlx_tier_completeness` — see [`minimax_h3_text_encoder_dir`].
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX MiniMax-H3 asset.
#[cfg(target_os = "macos")]
pub(super) const MINIMAX_H3_ADAPTER: &str = "mlx_minimax_h3";
/// Adapter id recorded on a real off-Mac Candle MiniMax-H3 asset. It lives with the deliberately
/// non-user-routed executor rather than the generic Candle route table, so capability-fact parsing
/// continues to represent only user-reachable Candle lanes.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const CANDLE_MINIMAX_H3_ADAPTER: &str = "candle_minimax_h3";

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_MINIMAX_H3_REPO: &str = "MiniMaxAI/MiniMax-H3";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_MINIMAX_H3_TIER_REPO: &str = "SceneWorks/minimax-h3-mlx";

/// The ONE registry id both catalog entries load. See fact 1 above — the partition split is a
/// directory, not a provider.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const MINIMAX_H3_ENGINE_ID: &str = "minimax_h3";

/// The staged-component key `mlx-gen-minimax-h3::model::DIT_COMPONENT` reads the tiered DiT from.
/// Named `"transformer"` even when the request will denoise on `transformer_ref/`: the provider
/// resolves the reference partition as this directory's SIBLING, so the base partition is always
/// what is staged.
#[cfg(target_os = "macos")]
pub(super) const MINIMAX_H3_DIT_COMPONENT: &str = "transformer";

// The staged-component key `mlx-gen-minimax-h3::resolve_text_encoder_dir` reads the PACKED, per-tier
// text encoder from (sc-19120) is `mlx_tier_completeness::MINIMAX_H3_TEXT_ENCODER_DIR` — the same
// constant `minimax_h3_text_encoder_dir` builds the root out of, so the key staged and the directory
// probed cannot disagree. Absent ⇒ the provider falls back to `<spec.weights>/text_encoder`, the
// dense upstream bf16 Qwen3-VL-32B.

/// The SceneWorks rehost carrying the pre-quantized DiT tiers (both partitions, all three tiers).
#[cfg(target_os = "macos")]
pub(super) const MINIMAX_H3_TIER_REPO: &str = "SceneWorks/minimax-h3-mlx";

/// The upstream snapshot carrying every SHARED component — the dense Qwen3-VL-32B text encoder, the
/// tokenizer, the video VAE, the audio VAE and the `FL2VA/` audio-VAE constructor documents. These
/// arrive as `coRequisite` downloads and are NOT tiered.
#[cfg(target_os = "macos")]
pub(super) const MINIMAX_H3_BASE_REPO: &str = "MiniMaxAI/MiniMax-H3";

/// Operator override for the tier root (the dir holding `q4/`/`q8/`/`bf16/`).
#[cfg(target_os = "macos")]
pub(super) const MINIMAX_H3_TIER_DIR_ENV: &str = "SCENEWORKS_MLX_MINIMAX_H3_TIER_DIR";

/// Operator override for the shared/base snapshot root (the dir holding `text_encoder/`).
#[cfg(target_os = "macos")]
pub(super) const MINIMAX_H3_BASE_DIR_ENV: &str = "SCENEWORKS_MLX_MINIMAX_H3_BASE_DIR";

/// SceneWorks MiniMax-H3 model id → the gen-core registry id, or `None` outside the family.
///
/// BOTH catalog ids resolve to the SAME engine id on purpose (fact 1). The partition they own is
/// carried by the conditioning shape, validated by [`minimax_h3_validate_partition`], not by a
/// second registry id.
///
/// Keyed off `sceneworks_core::video_request::is_minimax_h3_model` rather than a retyped id list so
/// this cannot drift from the predicate every other MiniMax-H3 gate in the app already uses.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn minimax_h3_engine_id(model: &str) -> Option<&'static str> {
    sceneworks_core::video_request::is_minimax_h3_model(model).then_some(MINIMAX_H3_ENGINE_ID)
}

/// Whether the linked inference bundle actually REGISTERS the MiniMax-H3 engine.
///
/// This is the sc-19508 replacement for sc-17159's hard-coded "not in the pinned inference
/// revision" string, and the reason the whole arm below can be written before sc-18650 moves the
/// pin. The engine is reached through `media().load(id, spec)` — a **runtime lookup keyed on a
/// string** — so nothing in this module needs the provider to be importable at compile time. What
/// it does need is for the id to be PRESENT in the registry, which is exactly what this reads.
///
/// Deriving it beat asserting it, and sc-19721 is the proof: at `014134e3` the descriptor was
/// absent and this refusal fired, and when the pin moved to `75d66db5` the descriptor appeared and
/// the arm went live with **no code change here**. A hard-coded revision string would have gone
/// stale instead — which is exactly what happened to every prose citation of the old pin around
/// it.
///
/// The same `media_descriptor(...).is_none()` idiom already gates the not-in-this-bundle branches
/// of `mlx_fit_gate`, so this is the established way to ask the question.
#[cfg(target_os = "macos")]
pub(super) fn minimax_h3_engine_is_registered() -> bool {
    crate::inference_runtime::media_descriptor(MINIMAX_H3_ENGINE_ID).is_some()
}

/// The tier search order — the requested tier first, then the always-SMALLER fallbacks so a
/// partially-installed model still loads. Mirrors [`mochi_tier_order`] including the video lane's
/// q4-first default (sc-10859): absent `advanced.mlxQuantize` ⇒ q4.
///
/// The default matters more here than anywhere else in the app. The dense bf16 DiT is 66 GB and the
/// text encoder alone measures 53 GB, so a silent bf16 default would put a default job over the
/// budget of every Mac this family targets.
#[cfg(target_os = "macos")]
pub(super) fn minimax_h3_tier_order(request: &VideoRequest) -> &'static [&'static str] {
    match resolve_mlx_dense_quant(request) {
        None => &["bf16", "q8", "q4"],
        Some(Quant::Q8) => &["q8", "q4"],
        _ => &["q4", "q8"],
    }
}

/// Whether `dir` is a loadable MiniMax-H3 TIER dir — **both** partitions present.
///
/// Deliberately stricter than either half of `mlx_tier_completeness`. Those two predicates are
/// independent BY DESIGN (a user can legitimately own a complete `minimax_h3_ref` tier and no
/// `transformer/` at all, and the Model Manager must report each entry honestly), but the ENGINE's
/// `load` probes `transformer/config.json` AND `transformer_ref/config.json` on every load
/// regardless of task, because ref2va is a first-class task rather than an optional extra. A tier
/// with one partition therefore reports "installed" in the Model Manager and cannot load — see
/// [`resolve_minimax_h3_load`], which names the missing half rather than letting the engine fail on
/// a path the user never chose.
///
/// Built ON the two shipped predicates rather than re-globbing the shard layout, so the
/// sharded-index rules sc-19078 established stay in one place.
#[cfg(target_os = "macos")]
pub(super) fn minimax_h3_tier_dir_is_complete(dir: &Path) -> bool {
    sceneworks_core::mlx_tier_completeness::minimax_h3_tier_complete(dir)
        && sceneworks_core::mlx_tier_completeness::minimax_h3_ref_tier_complete(dir)
}

// sc-19558 x sc-19573 — THE SHARED PROBE LIST LIVES IN `sceneworks_core::mlx_tier_completeness`,
// AND IT PROBES FILES.
//
// This module used to carry its own list (`MINIMAX_H3_SHARED_PROBES` /
// `MINIMAX_H3_UPSTREAM_TEXT_ENCODER_PROBE`) alongside a `has_packed_text_encoder` flag. sc-19573
// moved the list to core so the manifest's download-coverage guard and the runtime floor read ONE
// list; sc-19558 established that the list has to name FILES, because an interrupted co-requisite
// download leaves the directory present and empty and a dir-level floor certifies it as complete.
// Both survive: core keeps the tier-aware root resolution and gains the file-level probes plus
// `minimax_h3_missing_shared_probes`, which is what lets the refusal below name what is absent.
//
// The `has_packed_text_encoder` flag is gone because `minimax_h3_text_encoder_dir` subsumes it: the
// text encoder's ROOT moves with the tier rather than its necessity, so there is nothing left to
// make conditional.
/// Everything a MiniMax-H3 load needs, resolved as ONE unit — the two roots are in different repos
/// and neither is usable alone, so producing them separately is how they drift.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MiniMaxH3Load {
    /// The upstream snapshot root → `spec.weights`. Carries every shared component.
    pub(super) root: PathBuf,
    /// `<tier>/transformer` → the staged [`MINIMAX_H3_DIT_COMPONENT`]. Always the BASE partition,
    /// even for a ref2va job: the provider resolves `transformer_ref/` as this dir's sibling.
    pub(super) dit_dir: PathBuf,
    /// The PACKED per-tier text encoder (sc-19120), or `None` at bf16 where the dense upstream
    /// `<root>/text_encoder` is what the provider falls back to.
    ///
    /// `Some` rides `LoadSpec::components["text_encoder"]` (via
    /// [`VideoGenInput::text_encoder_component_dir`]), which is the seam the engine actually reads:
    /// `mlx-gen-minimax-h3::resolve_text_encoder_dir` matches `spec.components.get("text_encoder")`
    /// and otherwise falls back to `<weights>/text_encoder`. `LoadSpec::text_encoder` — the LTX
    /// external-Gemma field — has ZERO references in that engine crate, so staging there would be a
    /// silent no-op: on q4/q8 the fallback resolves a path those tiers correctly lack and the load
    /// hard-errors inside the engine on the missing `text_encoder/config.json`.
    ///
    /// Without this staging the 18.7 GB (q4) / 33.7 GB (q8) packed encoder sc-19120 added to the
    /// manifest is downloaded and never opened, and the render loads the 53 GB dense upstream
    /// conditioner instead — the declared-but-unreachable shape rather than a saving.
    pub(super) text_encoder_dir: Option<PathBuf>,
    /// The tier's baked-in quant, ASSERTED against the staged dir by the provider rather than
    /// applied — nothing quantizes at load. `None` for the dense bf16 tier.
    pub(super) quant: Option<Quant>,
    /// The tier that will actually run, for the asset record.
    pub(super) tier: &'static str,
}

/// The `Quant` a tier name asserts. bf16 is dense ⇒ `None`.
#[cfg(target_os = "macos")]
fn minimax_h3_tier_quant(tier: &str) -> Option<Quant> {
    match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        _ => None,
    }
}

/// Resolve the two roots a MiniMax-H3 load needs, or fail with the actionable reason.
///
/// Resolve-or-error, never a fallback to "hand it the root": every published DiT file lives under a
/// `<tier>/<partition>/` prefix, so a root fallback could only fail deeper inside the loader with a
/// less actionable message. This is the resolver sc-19508 substitutes for sc-17159's unconditional
/// refusal — the guard does not disappear, it becomes a real weights check.
#[cfg(target_os = "macos")]
pub(super) fn resolve_minimax_h3_load(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<MiniMaxH3Load> {
    let order = minimax_h3_tier_order(request);
    let tier_root = match std::env::var(MINIMAX_H3_TIER_DIR_ENV) {
        Ok(dir) => PathBuf::from(dir.trim()),
        Err(_) => huggingface_snapshot_dir(&settings.data_dir, MINIMAX_H3_TIER_REPO).ok_or_else(
            || {
                WorkerError::InvalidPayload(format!(
                    "{}: MiniMax-H3 is not downloaded (no {MINIMAX_H3_TIER_REPO} snapshot under \
                     {}). Install it from the Model Manager, or set ${MINIMAX_H3_TIER_DIR_ENV} to a \
                     tier root.",
                    request.model,
                    settings.data_dir.display()
                ))
            },
        )?,
    };
    let base_root = match std::env::var(MINIMAX_H3_BASE_DIR_ENV) {
        Ok(dir) => PathBuf::from(dir.trim()),
        Err(_) => {
            huggingface_snapshot_dir(&settings.data_dir, MINIMAX_H3_BASE_REPO).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{}: the shared MiniMax-H3 components are not installed (no \
                     {MINIMAX_H3_BASE_REPO} snapshot under {}). They install alongside whichever \
                     tier you pick — re-download MiniMax-H3 in the Model Manager, or set \
                     ${MINIMAX_H3_BASE_DIR_ENV}.",
                    request.model,
                    settings.data_dir.display()
                ))
            })?
        }
    };
    let Some(tier) = order
        .iter()
        .copied()
        .find(|tier| minimax_h3_tier_dir_is_complete(&tier_root.join(tier)))
    else {
        // Say WHICH half is missing. The two partitions are separate downloads owned by separate
        // catalog entries, so "no complete tier" alone would send a user who installed one of them
        // back to re-download the one they already have.
        let half_installed = order.iter().copied().find(|tier| {
            let dir = tier_root.join(tier);
            sceneworks_core::mlx_tier_completeness::minimax_h3_tier_complete(&dir)
                || sceneworks_core::mlx_tier_completeness::minimax_h3_ref_tier_complete(&dir)
        });
        if let Some(tier) = half_installed {
            return Err(WorkerError::InvalidPayload(format!(
                "{}: the {tier} tier under {} carries only one of the two MiniMax-H3 DiT \
                 partitions. The engine loads `transformer/` and `transformer_ref/` together — \
                 they are separate downloads (MiniMax-H3 and MiniMax-H3 Reference in the Model \
                 Manager), and both are required before either can render.",
                request.model,
                tier_root.display()
            )));
        }
        return Err(WorkerError::InvalidPayload(format!(
            "{}: no complete MiniMax-H3 tier found under {} (looked for {order:?}). Download the \
             tier you selected in the Model Manager — a partially-downloaded tier is skipped, so \
             re-running the download repairs it.",
            request.model,
            tier_root.display()
        )));
    };
    // The shared floor is checked AFTER the tier (sc-19573), because one of its paths — the text
    // encoder — is only addressable once the tier is known (sc-19120 packs it per tier; see
    // `minimax_h3_text_encoder_dir`). Checking it first is what made every q4/q8 install fail with a
    // shared-floor error naming `text_encoder/` under the upstream snapshot, a path those tiers are
    // correct not to have.
    let text_encoder_dir = sceneworks_core::mlx_tier_completeness::minimax_h3_text_encoder_dir(
        &tier_root, &base_root, tier,
    );
    // ENUMERATED, not restated (sc-19558). The probes are FILES, so the check can say which one is
    // absent — and it must, because the pre-sc-19558 message listed everything the floor expects
    // whether one path was missing or all eight, which sends a user to re-download components they
    // already have. Single-sourced from `mlx_tier_completeness` either way: that is what keeps this
    // message honest when the engine's probe list changes.
    let missing = sceneworks_core::mlx_tier_completeness::minimax_h3_missing_shared_probes(
        &base_root,
        &text_encoder_dir,
    );
    if !missing.is_empty() {
        // JOINED INTO PROSE rather than interpolated with `{:?}`, which put Rust debug literals in
        // front of a user in the Model Manager.
        let missing = missing
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(WorkerError::InvalidPayload(format!(
            "{}: the shared MiniMax-H3 text encoder / tokenizer / VAEs are incomplete for the \
             {tier} tier — missing {missing}. These are separate co-requisite downloads from the \
             quality tiers — re-download MiniMax-H3 in the Model Manager.",
            request.model,
        )));
    }
    Ok(MiniMaxH3Load {
        root: base_root,
        dit_dir: tier_root.join(tier).join(MINIMAX_H3_DIT_COMPONENT),
        // `None` at bf16 leaves the provider on its own `<root>/text_encoder` fallback, which is
        // where the dense upstream component actually is — passing it explicitly would be the same
        // path spelled twice. Not `.filter(is_complete)`: the floor above already refused a torn
        // packed encoder BY NAME, so there is nothing left to silently fall back from (sc-19558 —
        // falling back would put a q4 render on the 53 GB dense conditioner, which is the sc-19506
        // defect wearing a different hat).
        text_encoder_dir: (tier != "bf16").then_some(text_encoder_dir),
        quant: minimax_h3_tier_quant(tier),
        tier,
    })
}

// `minimax_h3_packed_te_is_complete` used to live here (sc-19506): `config.json` + a safetensors
// index, the pair `resolve_text_encoder_dir` → `map_shards` actually reads, so a directory holding
// only `config.json` — the sc-19517 failure shape — is rejected rather than half-loaded at the
// provider. It is now `mlx_tier_completeness::MINIMAX_H3_TEXT_ENCODER_PROBED_FILES`, checked by the
// shared floor above for WHICHEVER root the tier resolves to (packed at q4/q8, dense upstream at
// bf16) rather than only for the packed one. That is the strictly larger check: the dense encoder
// was previously never probed beyond `is_dir()`.

/// Refuse a request whose CONDITIONING SHAPE does not match the catalog entry it was submitted
/// against — the guard that keeps a job off the wrong checkpoint.
///
/// This is the sc-19508 acceptance criterion "a test that would FAIL if the wrong partition's
/// provider id were resolved", expressed against the mechanism that actually exists. There is no
/// second provider id to resolve wrongly (fact 1); the partition is chosen by
/// `MiniMaxH3Task::resolve(has_keyframes, has_references)` inside the engine. So the way to land on
/// the wrong 18.78 GB DiT is to send a shape that resolves to the other task — and because the two
/// checkpoints are structurally identical, that produces a render rather than an error.
///
/// The rules, each mirroring an engine-side one:
/// * `minimax_h3` (base, `transformer/`) takes keyframes and NO references. A reference of any kind
///   would resolve ref2va and denoise on `transformer_ref/`.
/// * `minimax_h3_ref` (`transformer_ref/`) takes references and NO keyframes.
/// * Keyframes AND references together are refused upstream by `MiniMaxH3Task::resolve`; refused
///   here too, so the user gets a SceneWorks-worded reason before anything loads.
/// * An audio-only reference set is refused: an audio reference never reaches the reference
///   conditioner (upstream `before_encoder.py` raises on `set(kinds) == {"audio"}`), so it would
///   leave the visual stream unconditioned. sc-19574 lifted that rule into
///   [`classify_reference_set`] so the API and the MCP tool refuse the identical shape — this is no
///   longer the only layer holding the opinion, it is the last one.
///
/// Not a duplicate of the API's `reference_limit_error` (sc-17160): that bounds HOW MANY references
/// each entry accepts (the base entry declares 0/0/0). This bounds the SHAPE, and it is the layer a
/// job replayed from an older row or produced by any future non-HTTP path passes through.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn minimax_h3_validate_partition(request: &VideoRequest) -> WorkerResult<()> {
    let has_keyframes = request.source_asset_id.is_some() || request.last_frame_asset_id.is_some();
    let image_refs = request.reference_asset_ids.len();
    let clip_refs = request.source_clip_asset_ids.len();
    let audio_refs = request.reference_audio_asset_ids.len();
    let has_references = image_refs + clip_refs + audio_refs > 0;

    if has_keyframes && has_references {
        return Err(WorkerError::InvalidPayload(format!(
            "{}: a request carries both keyframes and references, which are different MiniMax-H3 \
             tasks on different checkpoints — first/last-frame conditioning pins a literal frame \
             of the generated clip, references condition on unpositioned media. Send one or the \
             other.",
            request.model
        )));
    }
    let is_reference_partition = request.model == "minimax_h3_ref";
    if is_reference_partition {
        // sc-19574: both refusals now come off the SHARED verdict
        // (`sceneworks_core::video_request::classify_reference_set`) that the API's
        // `validate_video_job` and the MCP tool's `reference` arm also read, so the three layers
        // cannot drift back into disagreeing about which sets are legal. Only the wording is the
        // worker's — an engine-side reason names the checkpoint, which a user-facing 400 should not.
        match classify_reference_set(image_refs, clip_refs, audio_refs) {
            ReferenceSetVerdict::Conditionable => {}
            ReferenceSetVerdict::Empty => {
                return Err(WorkerError::InvalidPayload(format!(
                    "{}: reference-to-video requires at least one reference (referenceAssetIds, \
                     sourceClipAssetIds or referenceAudioAssetIds). A request with none would \
                     denoise on the base MiniMax-H3 checkpoint, which is not the one this model \
                     loads.",
                    request.model
                )));
            }
            ReferenceSetVerdict::AudioOnly => {
                return Err(WorkerError::InvalidPayload(format!(
                    "{}: an audio-only reference set leaves the visual stream unconditioned — an \
                     audio reference never reaches the reference conditioner, so at least one image \
                     or video reference must accompany the audio.",
                    request.model
                )));
            }
        }
        // 🔴 THE BLANKET VIDEO-REFERENCE REFUSAL IS GONE (sc-18650). It said "video references are
        // not renderable in this build", and both of its successive premises are now discharged:
        // sc-19721's pin bump made `Conditioning::ReferenceVideo { frames, fps, audio }` real and
        // MiniMax-H3's descriptor advertise `ConditioningKind::ReferenceVideo`, and this story
        // built the missing SceneWorks half — [`resolve_minimax_h3_clip_conditioning`] decodes a
        // source clip into the frames, the rate they carry and the clip's own soundtrack.
        //
        // Nothing about the SHAPE of a clip-bearing set is refusable here: the caps are the API's
        // (`reference_limit_error`) and the engine's (`Ref2VaReferences::new`), the audio-only and
        // empty shapes are already decided by the shared verdict above, and everything else a clip
        // can get wrong — too few frames to fill one conditioner group, an out-of-range aspect, an
        // undecodable file — is knowable only from the ASSET. Those refusals therefore live in the
        // decode, which runs before any weight is read; putting a placeholder for them here would
        // be a second opinion with no evidence behind it.
        //
        // Still `ReferenceVideo` BY NAME and never downgraded to `Conditioning::VideoClip`: the
        // engine deliberately does NOT advertise `VideoClip` (it has no in-context clip mechanism),
        // so that substitution would be refused as `Unsupported` after the load — or, worse,
        // silently accepted by some future provider as a completely different mechanism.
    } else if has_references {
        return Err(WorkerError::InvalidPayload(format!(
            "{}: this MiniMax-H3 entry loads the base checkpoint, which has no reference \
             conditioning. Submit reference-to-video against minimax_h3_ref, which loads the \
             reference DiT partition.",
            request.model
        )));
    }
    Ok(())
}

/// Whether the linked MiniMax-H3 engine can serve this request NOW: a MiniMax-H3 id, an engine
/// actually present in the linked bundle, a conditioning shape that matches the entry's partition,
/// and a resolvable tier + shared-component pair.
///
/// Every one of the four is folded in here rather than checked in the dispatch arm so a request
/// that fails any of them lands on [`VideoRoute::Stub`], where `ensure_video_engine_weights`
/// re-runs the same checks and surfaces the precise reason — instead of a procedural fake clip.
#[cfg(target_os = "macos")]
pub(super) fn minimax_h3_available(request: &VideoRequest, settings: &Settings) -> bool {
    minimax_h3_engine_id(&request.model).is_some()
        && minimax_h3_engine_is_registered()
        && minimax_h3_validate_partition(request).is_ok()
        && resolve_minimax_h3_load(settings, request).is_ok()
}

/// The fail-loud gate for the MiniMax-H3 family, called from
/// [`super::wan::ensure_video_engine_weights`].
///
/// **This is what sc-17159's unconditional refusal became.** That arm said, in a hard-coded string,
/// "the MiniMax-H3 MLX engine is not in the pinned inference revision yet (sc-18650)". It could not
/// say anything else, because at that commit there was no arm to fall through to. Now there is, so
/// the refusal is derived in three layers, each naming its own cause:
///
/// 1. the engine is not in the linked bundle (today's pin) — READ from the registry, not asserted;
/// 2. the conditioning shape does not match the entry's DiT partition;
/// 3. the weights are unprovisioned or torn.
///
/// The order is deliberate. Before sc-19721's pin bump every MiniMax-H3 job stopped at (1) with
/// the same honest reason a user got before; at the current pin (`75d66db5`, the first revision
/// that registers `mlx_gen_minimax_h3` — see `jobs_store/routing/mlx.rs`) the descriptor is
/// present, (1) passes with no code change here, and an unprovisioned install stops at (3) with
/// the resolver's precise error — which is exactly the substitution sc-19508 asked for
/// ("an unprovisioned install must still fail loudly").
#[cfg(target_os = "macos")]
pub(super) fn ensure_minimax_h3_renderable(
    request: &VideoRequest,
    settings: &Settings,
) -> WorkerResult<()> {
    if minimax_h3_engine_id(&request.model).is_none() {
        return Ok(());
    }
    if !minimax_h3_engine_is_registered() {
        return Err(WorkerError::Engine(format!(
            "{} cannot render in this build: the MiniMax-H3 MLX engine is not in the pinned \
             inference revision yet (sc-18650). No output was produced.",
            request.model
        )));
    }
    minimax_h3_validate_partition(request)?;
    resolve_minimax_h3_load(settings, request)?;
    Ok(())
}

/// Build the adapter specs for a MiniMax-H3 generation (sc-18726).
///
/// MiniMax-H3 is a single **dense** DiT — no MoE experts, no Lightning distill pair to prepend — so
/// this is the shared [`super::resolve_dense_adapters`] the Wan-VACE and SCAIL-2 arms use, with the
/// same `MAX_JOB_LORAS` cap and the same confinement of the resolved path.
///
/// 🔴 Until this existed the arm passed `adapters: Vec::new()` unconditionally, so a selected
/// MiniMax-H3 LoRA — turbo or otherwise — was accepted by the API's family gate, round-tripped
/// through the payload, shown as selected in the picker, and then **silently dropped** on the way to
/// the engine. That is the whole reason the recipe below is not enough on its own: switching the
/// schedule to 4 steps without the distilled residual actually loading renders the base checkpoint
/// at 4 steps, which sc-18729 measured as visibly undercooked (motion 1.117 against 5.612).
///
/// The engine (`mlx-gen-minimax-h3`, `supports_lora: true`) resolves each file's rank from the factor
/// shapes and its alpha from the file — per-target `.alpha`, then PEFT metadata, then the bare
/// top-level `__metadata__["alpha"]` string, then `DEFAULT_LORA_ALPHA = 8`, never the rank. Nothing
/// about that fold belongs on this side: `defaultWeight` is the runtime `lora_scale` MULTIPLIER and
/// the catalog leaves it at 1.0 so the file's own alpha/rank fold lands unmodified.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_minimax_h3_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
    super::resolve_dense_adapters(settings, request, MAX_JOB_LORAS)
}

/// The MiniMax-H3 sampling recipe `(steps, video scheduler shift)` for this request, plus the turbo
/// variant that produced it (`None` ⇒ the base regime).
///
/// The twin of [`super::wan::scail2_sampling`], and it diverges from it in exactly the three ways
/// sc-18726 identified:
///
/// 1. **The recipe is per VARIANT, read from the catalog.** SCAIL-2 has one lightning recipe and can
///    sniff it from the file format; MiniMax-H3's four published adapters carry three different
///    `(NFE, video shift)` pairs in a byte-identical format, so the discriminator is the catalog id
///    and the values live in `builtin.loras.jsonc` (`sampling`). Resolution goes through
///    [`sceneworks_core::minimax_h3_turbo::resolve_turbo_recipe`] — the ONE resolver both entry
///    points (the studio's turbo control and the generic LoRA picker) reach, because both do the
///    same thing: put a catalog id in `request.loras`.
///
/// 2. **Guidance is not part of the recipe.** SCAIL-2's lightning invariants include CFG-off, because
///    SCAIL-2 has CFG to turn off. This checkpoint is guidance-distilled — there is no unconditional
///    branch in it, the manifest declares `video.supportsGuidance: false`, and the engine REJECTS a
///    guidance scale — so the turbo LoRA changes the schedule and nothing else.
///
/// 3. **The audio shift is recorded, not forwarded.** See
///    [`sceneworks_core::minimax_h3_turbo`]: the engine holds audio at its own `AUDIO_SIGMA_SHIFT`,
///    every published variant trains at 3.0, and the core guard goes red if a catalog variant ever
///    declares otherwise.
///
/// An explicit `advanced.steps` wins over the variant's own count, exactly as `scail2_sampling`
/// lets it — upstream's spec table lists the 8-step file as "8 / 4", so a caller who knows the
/// checkpoint can run it shorter. The shift is NOT overridable the same way: `advanced.schedulerShift`
/// still reaches the base regime (it is the pre-existing knob this arm already read), but under a
/// turbo variant the trained shift wins, because a distilled checkpoint sampled at a shift it was
/// not distilled for is the off-distribution render the recipe exists to prevent.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn minimax_h3_sampling(
    request: &VideoRequest,
) -> WorkerResult<(
    Option<u32>,
    Option<f32>,
    Option<&'static sceneworks_core::minimax_h3_turbo::TurboRecipe>,
)> {
    let recipe =
        sceneworks_core::minimax_h3_turbo::resolve_turbo_recipe(&request.model, &request.loras)
            .map_err(WorkerError::InvalidPayload)?;
    let Some(recipe) = recipe else {
        // Base regime — byte-identical to what this arm did before sc-18726: `None` ⇒ the engine's
        // own DEFAULT_STEPS (50) and VIDEO_SIGMA_SHIFT (12.0), with the pre-existing `advanced`
        // overrides still honored.
        return Ok((
            advanced_opt_u32(request, "steps"),
            advanced_opt_f32(request, "schedulerShift"),
            None,
        ));
    };
    Ok((
        advanced_opt_u32(request, "steps").or(Some(recipe.steps)),
        Some(recipe.video_shift),
        Some(recipe),
    ))
}

/// The `rawSettings` recorded on a real MiniMax-H3 asset.
///
/// `minimaxH3Tier` names the tier that actually LOADED, not the one the request asked for — the
/// `mochiTier` / `kreaRealtimeTier` convention (sc-15258). They diverge whenever a requested tier
/// is not installed and the order falls back.
///
/// `minimaxH3Task` records which of the three tasks — and therefore which of the two 18.78 GB DiT
/// partitions — actually denoised. It is the one field that makes a wrong-checkpoint render
/// visible after the fact: the output of a t2va job on `transformer_ref/` is plausible video, so
/// the asset record is the only place the truth can live.
///
/// The `effective*` block is the sc-18726 half — the `wan_raw_settings` / `scail2_raw_settings`
/// convention, and the only place the schedule a render ACTUALLY ran is inspectable. `minimaxH3Turbo`
/// names the variant (or `"off"`), `effectiveSteps` is the model-evaluation count that reached the
/// engine, `effectiveSchedulerShift` the video shift, and `effectiveAudioSchedulerShift` the audio
/// one.
///
/// `effectiveAudioSchedulerShift` is recorded from the DECLARED recipe (or, in the base regime, from
/// the engine constant it is pinned to) rather than from a request field, because there is no request
/// field: the engine holds audio at `AUDIO_SIGMA_SHIFT` on every path. Writing the number anyway is
/// how the asset states the WHOLE schedule instead of the video half of it, and
/// `sceneworks_core::minimax_h3_turbo` carries the guard that the declared and rendered values cannot
/// silently diverge.
///
/// `effectiveSteps` / `effectiveSchedulerShift` record `null` when the arm passed `None` — "the
/// engine picked its own default" is a fact about the render, and mirroring the engine's 50 / 12.0
/// here would create a second place for those constants to drift. `scail2_raw_settings` omits its
/// `effective*` keys off-recipe for the same reason it never has to answer that question; this arm
/// answers it explicitly so `minimaxH3Turbo: "off"` and a null step count read as one statement.
///
/// `minimaxH3ReferenceClips` / `minimaxH3ReferenceClipSoundtracks` are read off the CONDITIONING
/// that was actually assembled, for the same reason `task` is (sc-18650). Whether a video
/// reference's own soundtrack was conditioned on is a real fork in the request — it consumes an
/// `<Audio j>` label and adds audio rows on the shared rotary clock — and it is decided from the
/// source file rather than from anything the caller sent, so it is invisible in both the payload
/// and the output. A clip supplied silently without audio and a silent clip are the same render and
/// should read as the same record; a clip whose soundtrack WAS used is a different one.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn minimax_h3_raw_settings(
    request: &VideoRequest,
    tier: &str,
    task: &str,
    conditioning: &[Conditioning],
    steps: Option<u32>,
    scheduler_shift: Option<f32>,
    turbo: Option<&sceneworks_core::minimax_h3_turbo::TurboRecipe>,
) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert("minimaxH3Tier".to_owned(), Value::String(tier.to_owned()));
    raw.insert("minimaxH3Task".to_owned(), Value::String(task.to_owned()));
    let clips = conditioning
        .iter()
        .filter(|c| matches!(c, Conditioning::ReferenceVideo { .. }))
        .count();
    let clip_soundtracks = conditioning
        .iter()
        .filter(|c| matches!(c, Conditioning::ReferenceVideo { audio: Some(_), .. }))
        .count();
    raw.insert("minimaxH3ReferenceClips".to_owned(), json!(clips));
    raw.insert(
        "minimaxH3ReferenceClipSoundtracks".to_owned(),
        json!(clip_soundtracks),
    );
    raw.insert(
        "minimaxH3Turbo".to_owned(),
        Value::String(
            turbo
                .map(|recipe| recipe.lora_id.clone())
                .unwrap_or_else(|| "off".to_owned()),
        ),
    );
    raw.insert("effectiveSteps".to_owned(), json!(steps));
    raw.insert("effectiveSchedulerShift".to_owned(), json!(scheduler_shift));
    raw.insert(
        "effectiveAudioSchedulerShift".to_owned(),
        json!(turbo
            .map(|recipe| recipe.audio_shift)
            .unwrap_or(sceneworks_core::minimax_h3_turbo::ENGINE_AUDIO_SIGMA_SHIFT)),
    );
    Value::Object(raw)
}

// ---------------------------------------------------------------------------
// Reference CLIPS (sc-18650): `sourceClipAssetIds` -> `Conditioning::ReferenceVideo`.
//
// Five engine facts decide every line of the block below. All five are read off the pinned
// `mlx-gen-minimax-h3` (`crates/media/mlx-gen/mlx-gen-minimax-h3/src/reference.rs` and
// `model.rs::generate_ref2va`), not inferred from the shape of the payload:
//
//  1. **A reference does NOT bind the generated geometry.** `normalize_reference_clip` puts each
//     clip on the canvas ITS OWN aspect ratio resolves to (short edge 768, area cap 768·1344,
//     stride-32 snap) — never on the request's canvas. So this decode must NOT reuse the
//     `scale=W:H:force_original_aspect_ratio=decrease,pad=…` letterbox recipe the Wan-VACE control
//     video (`vace::load_source_video_frames`) and the LTX in-context clips
//     (`ltx::extract_clip_frames`) use: those pad to the OUTPUT dimensions, and padding a reference
//     bakes letterbox bars into media the model reads as content. Aspect is preserved and nothing
//     is padded; the engine finishes the fit.
//
//  2. **The engine resamples onto 24 fps by holding whole frames** — `ffmpeg`'s `fps` filter, which
//     the engine's own docs name as the mechanism the released model was conditioned through — and
//     that resample is the IDENTITY when the clip already carries 24 fps. So the decode asks ffmpeg
//     for `fps=24` and declares 24, rather than declaring a probed source rate and letting the
//     engine redo it. Three things follow, and each is why this direction was chosen:
//       * **VFR sources are correct.** ffmpeg resamples off real timestamps; the engine's
//         whole-frame hold assumes a constant rate, so a VFR clip declared at its *average* rate
//         would be conditioned on at the wrong speed with nothing to raise about it — the exact
//         failure `VideoReference::fps` exists to prevent.
//       * **No rate probe is needed at all**, so there is no sidecar `file.fps` to be stale or
//         absent (the seedvr2 upscale path defaults that field to 24 when missing — harmless for an
//         output rate, load-bearing for an input one).
//       * **The extracted frame count is bounded by the render**, not by the source: a 60 fps
//         10-minute clip yields the same `frames` PNGs a 24 fps 6-second one does. Declaring the
//         source rate instead would decode 36 000 frames for the engine to throw away.
//     The `fps` this arm declares stays TRUE data, not a default: it is the rate the frames handed
//     over actually carry, because this decode is what put them on it.
//
//  3. **The engine truncates the resampled clip to the generated frame count** and then encodes the
//     soundtrack whole. So the decode caps frames at the request's own `17n + 5` count AND caps the
//     soundtrack at the matching span — otherwise a 30 s soundtrack would be conditioned against
//     6 s of picture.
//
//  4. **The conditioner reads the clip at 2 fps in groups of 2**, so a clip that normalizes to
//     fewer than [`MINIMAX_H3_MIN_REFERENCE_CLIP_FRAMES`] frames is rejected — by the engine, but
//     only inside `encode_prompt_ref2va`, i.e. AFTER the 66.7 GB conditioner is mapped. It is
//     knowable from the decoded frames alone, so it is refused here instead.
//
//  5. **The engine ships no audio resampler.** `Ref2VaReferences::new` rejects any soundtrack that
//     is not already at the audio VAE's 32 kHz, at the engine boundary — "the caller resamples".
//     This is the caller.
// ---------------------------------------------------------------------------

/// The rate a reference clip is handed to the engine at — MiniMax-H3's own 24 fps, which is also
/// the only output rate it generates at.
///
/// Read off the SHARED `MINIMAX_H3_FPS` rather than retyped: the engine resamples every reference
/// onto `crate::denoise::MINIMAX_H3_FPS`, and the two being the same number is not a coincidence to
/// be maintained in two places.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const MINIMAX_H3_REFERENCE_CLIP_FPS: u32 =
    sceneworks_core::video_request::MINIMAX_H3_FPS;

/// The short edge the engine lays every reference clip's canvas on
/// (`mlx-gen-minimax-h3::pipeline::CANVAS_SHORT_EDGE`).
///
/// Mirrored here only as a DOWNWARD bound on what is decoded: the extraction never scales a clip
/// UP to it (the engine's `resolve_canvas_size` does its own upscale, and doing it twice only
/// costs), and never pads. Its whole job is to keep a 4K source from being materialized at 25 MB a
/// frame for a canvas that is about to throw 90% of it away.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const MINIMAX_H3_CANVAS_SHORT_EDGE: u32 = 768;

/// The audio VAE's rate (`mlx-gen-minimax-h3::audio_config::AUDIO_SAMPLE_RATE`). The engine has no
/// resampler and rejects anything else at its boundary, so the extraction targets it exactly.
///
/// An ALIAS of the ungated [`super::reference_audio::REFERENCE_AUDIO_SAMPLE_RATE`], not a second
/// literal: a video reference's own soundtrack (extracted here) and a standalone audio reference
/// (extracted by `super::reference_audio`, which cannot see this backend-gated module) are checked
/// against the SAME engine constant, so they can never drift apart.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const MINIMAX_H3_AUDIO_SAMPLE_RATE: u32 =
    super::reference_audio::REFERENCE_AUDIO_SAMPLE_RATE;

/// Channels a reference soundtrack is extracted at — stereo, the shape the joint model both emits
/// and was conditioned on. The engine accepts any positive channel count, so this is a choice
/// rather than a constraint, and downmixing to mono would discard the stereo image of the very clip
/// the caller supplied as the reference.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const MINIMAX_H3_REFERENCE_AUDIO_CHANNELS: u32 = 2;

/// Fewest 24 fps frames a reference clip may normalize to — **13**.
///
/// DERIVED, not picked. The conditioner samples the normalized clip at `VIDEO_SAMPLE_FPS = 2` and
/// merges the sampled frames in groups of `VISION_TEMPORAL_PATCH = 2`, so the stride is
/// `24 / 2 = 12` and a clip must reach the second sampled frame: `round(1 · 12) + 1 = 13` frames,
/// 0.5417 s. `sample_video_condition_frames` computes the identical bound and refuses below it,
/// because a merged group repeating a single frame is not a shape the released model ever saw.
///
/// It is re-derived here from its three inputs rather than written as a literal so a fine-tune that
/// moved either rate would move this with it.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const MINIMAX_H3_MIN_REFERENCE_CLIP_FRAMES: usize = {
    // `VIDEO_SAMPLE_FPS = 2.0`, `VISION_TEMPORAL_PATCH = 2`.
    let sample_fps = 2usize;
    let temporal_patch = 2usize;
    let stride = MINIMAX_H3_REFERENCE_CLIP_FPS as usize / sample_fps;
    (temporal_patch - 1) * stride + 1
};

/// The area budget the engine fits every canvas inside
/// (`mlx-gen-minimax-h3::pipeline::CANVAS_MAX_PIXELS`, `768 · 1344 = 1_032_192`).
///
/// The SHORT edge alone does not bound the picture: at 16:9 a 768-short-edge canvas is 768 x 1365,
/// already over budget, and the engine's `resolve_canvas_size` scales BOTH edges by
/// `sqrt(max_pixels / area)` once the area exceeds this. Mirrored here for the same reason
/// [`MINIMAX_H3_CANVAS_SHORT_EDGE`] is — as a downward bound on what is materialized, so a
/// cinemascope source is not decoded to a PNG sequence the engine is about to shrink anyway.
/// Tied to the pinned engine's own const by
/// `minimax_h3_reference_clip_filter_never_pads_or_upscales`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const MINIMAX_H3_CANVAS_MAX_PIXELS: u32 = MINIMAX_H3_CANVAS_SHORT_EDGE * 1344;

/// The widest aspect ratio, either way up, that MiniMax-H3 reads a reference at —
/// `keyframe::MAX_ASPECT_RATIO`, i.e. 1:4 to 4:1. Checked per decoded clip so a panoramic source is
/// refused from the asset instead of from inside `Ref2VaReferences::new` after the decode has
/// already been paid for.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const MINIMAX_H3_MAX_REFERENCE_ASPECT: f64 = 4.0;

/// The `-vf` chain a reference clip is extracted through.
///
/// `fps` FIRST, then `scale` — the resample is the cheaper filter and running it first means the
/// scaler only ever touches frames that survive. The scale expression is deliberately NOT one of
/// the `force_original_aspect_ratio` forms used elsewhere in this crate:
///
/// * it caps the SHORT edge (whichever that is) at `short_edge`, where
///   `force_original_aspect_ratio=decrease` caps the LONG one — which would hand the engine a clip
///   below its canvas for it to upscale back, losing detail for nothing;
/// * it ALSO caps the area at `max_pixels`, because the short edge alone does not bound the
///   picture. Both bounds are the engine's, and they are combined the way its own
///   `resolve_canvas_size` combines them: lay the short edge at `short_edge`, then, if the implied
///   area is over budget, scale by `sqrt(max_pixels / area)`. Solved for the constrained axis that
///   is `min(short, sqrt(max_pixels · short_side / long_side))` — the same number, spelled as one
///   expression. Past 1.75:1 (`max_pixels / short_edge²`) the area term is the binding one, so a
///   2.39:1 source is no longer materialized ~36% over the canvas the engine will refit it to;
/// * it never scales UP (`min(…)`), because the engine's own canvas resolver upscales a small
///   source anyway and doing it here first would only interpose a second resampler;
/// * it pads NOTHING. See fact 1: a reference does not bind the canvas, so letterbox bars added
///   here would be conditioned on as picture.
///
/// `-1` (not `-2`) on the free axis: the output is a PNG sequence with no chroma subsampling to
/// satisfy, so the aspect ratio is preserved to the nearest whole pixel rather than the nearest
/// even one. The engine snaps to a 32 multiple regardless — which is also why the `sqrt` result is
/// left unsnapped here: this is a materialization bound, not a canvas decision.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn minimax_h3_reference_clip_filter(
    fps: u32,
    short_edge: u32,
    max_pixels: u32,
) -> String {
    // Landscape constrains the height, portrait the width; the free axis rides `-1`.
    let landscape_h = format!("min(min(ih,{short_edge}),sqrt({max_pixels}*ih/iw))");
    let portrait_w = format!("min(min(iw,{short_edge}),sqrt({max_pixels}*iw/ih))");
    format!(
        "fps={fps},scale='if(gte(iw,ih),-1,{portrait_w})':'if(gte(iw,ih),{landscape_h},-1)',format=rgb24"
    )
}

/// Whether the ffmpeg run whose stderr this is read an INPUT that carries an audio stream.
///
/// Parsed off the extraction's own stderr rather than from a separate probe pass: ffmpeg always
/// prints the full input stream listing, so the answer is free once the frames have been pulled and
/// there is no second spawn (and no `ffprobe`, which the desktop bundle does not ship — see
/// `media_jobs::probe_source_frame_count`).
///
/// Scoped to the INPUT section on purpose. `Stream #…: Audio:` also appears under `Output #…` when
/// a command writes audio, and a parser that read the whole buffer would answer a different
/// question the moment this chain grows an audio output.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn minimax_h3_clip_input_has_audio(stderr: &str) -> bool {
    stderr
        .lines()
        .skip_while(|line| !line.trim_start().starts_with("Input #"))
        .take_while(|line| !line.trim_start().starts_with("Output #"))
        .any(|line| line.contains("Stream #") && line.contains(": Audio:"))
}

/// Refuse a decoded reference clip whose SHAPE the engine cannot condition on — from the frames
/// alone, before a single weight is read.
///
/// Both rules are the engine's own, and both would otherwise fire late: the frame floor from inside
/// `encode_prompt_ref2va` with the 66.7 GB conditioner already mapped, the aspect bound from
/// `Ref2VaReferences::new` after every clip in the request has been decoded. Neither is a NEW gate —
/// they are the same two refusals, moved to the first layer that can see the evidence, and worded
/// for someone holding the asset rather than the checkpoint.
///
/// What is deliberately NOT refused here: a clip LONGER than the render. The engine truncates it,
/// that truncation is defined behaviour, and the extraction below has already stopped pulling
/// frames at the render's own count — so "too long" is not an error shape, it is the normal one.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn minimax_h3_validate_reference_clip(
    model: &str,
    asset_id: &str,
    frames: &[Image],
) -> WorkerResult<()> {
    let Some(first) = frames.first() else {
        return Err(WorkerError::InvalidPayload(format!(
            "{model}: source clip {asset_id} produced no decodable frames, so there is nothing to \
             condition on. No output was produced."
        )));
    };
    if frames.len() < MINIMAX_H3_MIN_REFERENCE_CLIP_FRAMES {
        let seconds =
            MINIMAX_H3_MIN_REFERENCE_CLIP_FRAMES as f64 / f64::from(MINIMAX_H3_REFERENCE_CLIP_FPS);
        return Err(WorkerError::InvalidPayload(format!(
            "{model}: source clip {asset_id} is {} frames long at {} fps, but a reference clip is \
             read by the conditioner in merged pairs and must run at least \
             {MINIMAX_H3_MIN_REFERENCE_CLIP_FRAMES} frames ({seconds:.2} s). Supply a longer clip, \
             or send a still from it as a reference image instead. No output was produced.",
            frames.len(),
            MINIMAX_H3_REFERENCE_CLIP_FPS,
        )));
    }
    if first.width == 0 || first.height == 0 {
        return Err(WorkerError::InvalidPayload(format!(
            "{model}: source clip {asset_id} decoded to {}x{} frames. No output was produced.",
            first.width, first.height
        )));
    }
    let (w, h) = (f64::from(first.width), f64::from(first.height));
    if w > MINIMAX_H3_MAX_REFERENCE_ASPECT * h || h > MINIMAX_H3_MAX_REFERENCE_ASPECT * w {
        return Err(WorkerError::InvalidPayload(format!(
            "{model}: source clip {asset_id} is {}x{}, outside the 1:4 to 4:1 range MiniMax-H3 \
             reads a reference at. Crop it to a less extreme aspect ratio. No output was produced.",
            first.width, first.height
        )));
    }
    Ok(())
}

/// Trim one `sourceClipAssetIds` entry, refusing a blank one before it becomes a path component.
///
/// **Defense-in-depth behind the parser, not a reachable payload refusal.** `VideoRequest`'s
/// `string_list` already trims every entry and DROPS the blank ones, so an HTTP payload cannot
/// carry a blank id this far — `parses_mv2v_and_ads2v_multi_source_fields` in
/// `sceneworks-core::video_request` pins that. It is kept because this arm is also reachable from
/// producers that never touch that parser (a job row replayed from an older schema, a hand-built
/// `VideoRequest`), where an empty id would otherwise resolve to the project directory itself
/// rather than to a clip. Split out of [`resolve_minimax_h3_clip_conditioning`] so the refusal is
/// callable — and therefore testable — without an `ApiClient`, a `Settings` and a `JobSnapshot`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn minimax_h3_clip_asset_id<'a>(
    model: &str,
    asset_id: &'a str,
) -> WorkerResult<&'a str> {
    let asset_id = asset_id.trim();
    if asset_id.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "{model}: sourceClipAssetIds must not contain blank ids."
        )));
    }
    Ok(asset_id)
}

/// Decode `request.source_clip_asset_ids` into ordered [`Conditioning::ReferenceVideo`] entries.
///
/// One ffmpeg pass per clip for the picture, plus one more ONLY when that pass's stderr says the
/// source actually carries audio. Both go through the shared [`run_ffmpeg`]/
/// [`crate::media_jobs::run_ffmpeg_capture_stderr`] runners, so the job keeps heart-beating and
/// stays cancellable while a long clip is walked — the same lifecycle every other clip reader in
/// this crate gets, rather than a private `Command`.
///
/// # The soundtrack is carried, not dropped
///
/// A video reference's own soundtrack rides `VideoReference::audio` and is conditioned on as THAT
/// reference's own — rotary-aligned with its video rows, sharing their origin, consuming an
/// `<Audio j>` label but NOT one of the three standalone audio-reference slots. Dropping it would
/// silently substitute "condition on motion alone" for the request the caller made, on media they
/// supplied; re-sending it as a separate `ReferenceAudio` would substitute a standalone reference
/// with its own rotary slot. Neither is the same request. It is recorded in `rawSettings` because
/// it is otherwise invisible in the output.
///
/// `frames` is the render's own `17n + 5` count, so a clip is never decoded past the point the
/// engine would truncate it, and the soundtrack is cut to the matching span.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn resolve_minimax_h3_clip_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    frames: u32,
) -> WorkerResult<Vec<Conditioning>> {
    let mut conditioning = Vec::with_capacity(request.source_clip_asset_ids.len());
    for (index, asset_id) in request.source_clip_asset_ids.iter().enumerate() {
        let asset_id = minimax_h3_clip_asset_id(&request.model, asset_id)?;
        let clip_path = super::ltx::resolve_clip_media_path(
            settings,
            &request.project_id,
            asset_id,
            project_path,
        )?;
        // Sanitize the job id before it becomes a path component (F-111), exactly as the
        // person-track and seedvr2 scratch dirs do; the per-clip index keeps three concurrent
        // references in one job from sharing a directory.
        let work_dir = std::env::temp_dir().join(format!(
            "sw-minimax-h3-ref-{}-{index}",
            safe_download_dir(&job.id)
        ));
        let _ = tokio::fs::remove_dir_all(&work_dir).await;
        tokio::fs::create_dir_all(&work_dir).await?;
        let decoded = minimax_h3_decode_reference_clip(
            api, settings, job, request, asset_id, &clip_path, &work_dir, frames,
        )
        .await;
        let _ = tokio::fs::remove_dir_all(&work_dir).await;
        conditioning.push(decoded?);
    }
    Ok(conditioning)
}

/// One clip: extract, load, validate, and pair with its soundtrack. Split out of
/// [`resolve_minimax_h3_clip_conditioning`] only so the caller can drop the scratch directory on
/// EVERY exit — including the refusals, which are the paths most likely to leave a multi-hundred-MB
/// PNG sequence behind.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn minimax_h3_decode_reference_clip(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    asset_id: &str,
    clip_path: &Path,
    work_dir: &Path,
    frames: u32,
) -> WorkerResult<Conditioning> {
    let pattern = work_dir.join("ref_%05d.png");
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let stderr = crate::media_jobs::run_ffmpeg_capture_stderr(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            clip_path.display().to_string(),
            "-map".to_owned(),
            "0:v:0".to_owned(),
            "-vf".to_owned(),
            minimax_h3_reference_clip_filter(
                MINIMAX_H3_REFERENCE_CLIP_FPS,
                MINIMAX_H3_CANVAS_SHORT_EDGE,
                MINIMAX_H3_CANVAS_MAX_PIXELS,
            ),
            // Never more picture than the engine would keep (fact 3).
            "-frames:v".to_owned(),
            frames.to_string(),
            "-start_number".to_owned(),
            "0".to_owned(),
            pattern.display().to_string(),
        ],
        Some(ctx),
    )
    .await?;

    let decoded = minimax_h3_load_reference_frames(work_dir.to_path_buf()).await?;
    minimax_h3_validate_reference_clip(&request.model, asset_id, &decoded)?;

    let audio = if minimax_h3_clip_input_has_audio(&stderr) {
        let wav = work_dir.join("ref_audio.wav");
        let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
        run_ffmpeg(
            vec![
                "ffmpeg".to_owned(),
                "-nostdin".to_owned(),
                "-y".to_owned(),
                "-i".to_owned(),
                clip_path.display().to_string(),
                "-map".to_owned(),
                "0:a:0".to_owned(),
                "-vn".to_owned(),
                // The engine ships no resampler and refuses anything but its VAE's rate (fact 5).
                "-ar".to_owned(),
                MINIMAX_H3_AUDIO_SAMPLE_RATE.to_string(),
                "-ac".to_owned(),
                MINIMAX_H3_REFERENCE_AUDIO_CHANNELS.to_string(),
                // `read_wav_pcm16` reads PCM s16 only.
                "-c:a".to_owned(),
                "pcm_s16le".to_owned(),
                // Cut to the span of the picture that was actually kept — the engine encodes the
                // whole track, so an untrimmed soundtrack would be conditioned against a clip it is
                // five times longer than (fact 3). The shared `picture_bound_seconds` is the same
                // frames -> seconds bound the seedvr2 mux uses.
                "-t".to_owned(),
                picture_bound_seconds(decoded.len(), MINIMAX_H3_REFERENCE_CLIP_FPS),
                wav.display().to_string(),
            ],
            Some(ctx),
        )
        .await?;
        Some(crate::audio_jobs::read_wav_pcm16(&wav)?)
    } else {
        None
    };

    Ok(Conditioning::ReferenceVideo {
        frames: decoded,
        // TRUE data, not a default: `-vf fps=…` above put the frames on exactly this rate, so the
        // engine's resample is the identity rather than a second opinion (fact 2).
        fps: MINIMAX_H3_REFERENCE_CLIP_FPS as f32,
        audio,
    })
}

/// Load an extracted `ref_%05d.png` sequence into engine [`Image`]s, in frame order, off the async
/// runtime.
///
/// **`paths.sort()` is what makes the order the clip's order**, and the zero-padded `%05d` pattern
/// is what makes the lexical sort the numeric one. Frame order is the one property of a motion
/// reference that is completely invisible in the rendered output when it is wrong — a shuffled
/// reference still renders a plausible clip — which is why it is asserted directly by
/// `minimax_h3_reference_clip_frames_keep_source_order`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn minimax_h3_load_reference_frames(
    work_dir: PathBuf,
) -> WorkerResult<Vec<Image>> {
    tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&work_dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("png"))
            .collect();
        paths.sort();
        let mut frames = Vec::with_capacity(paths.len());
        for path in paths {
            let decoded = crate::image_decode::decode_image_any(&path)
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "reference clip frame {}: {error}",
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
    .map_err(|error| task_join_error("reference clip frame decode task", error))?
}

/// The engine task a resolved conditioning list denoises as — the SceneWorks-side mirror of
/// `MiniMaxH3Task::resolve`, derived from the SAME input the engine derives it from (the
/// conditioning itself, not the mode string or the model id).
///
/// Reading the conditioning rather than the request is what makes this honest: if the assembly
/// below ever drops a reference or a keyframe, the recorded task changes with it instead of
/// continuing to claim what the payload asked for.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn minimax_h3_task(conditioning: &[Conditioning]) -> &'static str {
    let has_keyframes = conditioning
        .iter()
        .any(|c| matches!(c, Conditioning::Keyframe { .. }));
    // 🔴 `ReferenceVideo` belongs in this list and was missing from it while the variant could not
    // be built (sc-18650). A clip-only Ref2VA request would otherwise have recorded `t2va` in
    // `rawSettings` for a render that denoised on `transformer_ref/` — the one number in the record
    // whose whole job is to say which checkpoint ran.
    let has_references = conditioning.iter().any(|c| {
        matches!(
            c,
            Conditioning::Reference { .. }
                | Conditioning::ReferenceVideo { .. }
                | Conditioning::ReferenceAudio { .. }
        )
    });
    match (has_keyframes, has_references) {
        (true, _) => "fl2va",
        (false, true) => "ref2va",
        (false, false) => "t2va",
    }
}

/// Build the MiniMax-H3 conditioning from the (already loaded) media — the PURE shape decision.
///
/// Kept pure (loaded media in, conditioning out) so the two things that are invisible in the output
/// when wrong — the keyframe ANCHOR INDEX and the reference ORDER — are unit-testable without a
/// GPU, weights, or on-disk assets.
///
/// **Anchor indices.** `frame_idx: 0` is the first frame; the last frame is `-1`, NOT
/// `frames - 1`. The engine accepts either, but `-1` is independent of the frame count this arm
/// computed — so a disagreement between our lattice coercion and the engine's can never silently
/// turn a last-frame anchor into a rejected mid-clip one. `strength: 1.0` fully pins the frame,
/// which is the only meaning first/last-frame conditioning has here (the engine pins anchor rows at
/// its own conditioning timestep and does not read this value).
///
/// **Reference order is semantic, not incidental.** The engine labels references `<Picture i>` /
/// `<Video k>` / `<Audio j>` in list order and advances one shared rotary clock across them, so
/// re-ordering the list changes the render. Images, then clips, then audio — and that grouping is
/// THIS function's choice, not the engine's. It is the only order the payload can express
/// (`referenceAssetIds`, `sourceClipAssetIds` and `referenceAudioAssetIds` are three separate lists
/// with no interleaving), so the worker has to pick one; the engine's `request_references`
/// (`mlx-gen-minimax-h3/src/model.rs:637` at pin `f17c82544`) then walks the conditioning LIST in
/// the order it arrives, matching each entry on its variant and pushing it straight onto one
/// `Vec<Ref2VaReference>`. It does NOT re-group by kind — there is no images-then-clips-then-audio
/// pass on that side to agree with. So whatever order this builds is the order that is labelled and
/// rotary-clocked, which is exactly why the grouping is stated here rather than left incidental.
/// The caller's order WITHIN each list is preserved exactly.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn minimax_h3_conditioning(
    first_frame: Option<Image>,
    last_frame: Option<Image>,
    reference_images: Vec<Image>,
    reference_clips: Vec<Conditioning>,
    reference_audio: Vec<Conditioning>,
) -> Vec<Conditioning> {
    let mut conditioning = Vec::new();
    if let Some(image) = first_frame {
        conditioning.push(Conditioning::Keyframe {
            image,
            frame_idx: 0,
            strength: 1.0,
        });
    }
    if let Some(image) = last_frame {
        conditioning.push(Conditioning::Keyframe {
            image,
            frame_idx: -1,
            strength: 1.0,
        });
    }
    for image in reference_images {
        conditioning.push(Conditioning::Reference {
            image,
            strength: None,
        });
    }
    conditioning.extend(reference_clips);
    conditioning.extend(reference_audio);
    conditioning
}

/// Resolve a MiniMax-H3 request's supplied media into the engine conditioning.
///
/// `sourceAssetId` → the FIRST keyframe, `lastFrameAssetId` → the LAST — fl2va takes 0, 1 or 2 of
/// them, so `image_to_video` (first only), a last-frame-only payload, and `first_last_frame` (both)
/// are one code path rather than three modes. `referenceAssetIds` → ordered image references,
/// `sourceClipAssetIds` → ordered video references, `referenceAudioAssetIds` → ordered audio
/// references, which together are the Ref2VA set.
///
/// The audio references go through the SHARED `resolve_reference_audio_conditioning` (sc-17160) —
/// the same function `run_video_generate_job` already calls to prove they resolve before the job is
/// marked Running. That call discards its value with a comment naming this arm as the consumer;
/// this is that consumer, and it deliberately re-uses the function rather than re-implementing the
/// project-scoped lookup and the `safe_project_path` guard.
///
/// Async since sc-18650: the clip AND audio references decode through ffmpeg (the audio ones to
/// satisfy `Ref2VaReferences::check_audio`'s rate gate, fact 5 above), and the shared runner that
/// owns the heartbeat and the cancel poll is async. Only the still-image half stays synchronous.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn resolve_minimax_h3_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    frames: u32,
) -> WorkerResult<Vec<Conditioning>> {
    let load = |asset_id: &str| {
        load_reference_image(
            &settings.data_dir,
            &request.project_id,
            asset_id,
            project_path,
        )
    };
    let first_frame = request.source_asset_id.as_deref().map(load).transpose()?;
    let last_frame = request
        .last_frame_asset_id
        .as_deref()
        .map(load)
        .transpose()?;
    let mut reference_images = Vec::with_capacity(request.reference_asset_ids.len());
    for asset_id in &request.reference_asset_ids {
        reference_images.push(load(asset_id)?);
    }
    // The still references are cheap, local reads and the audio references are one short ffmpeg
    // transcode each; a clip is a whole PNG sequence. So the clips are resolved LAST — a payload
    // that names a missing reference image or an undecodable soundtrack is refused before a single
    // frame is extracted.
    let reference_audio =
        super::resolve_reference_audio_conditioning(api, settings, job, request, project_path)
            .await?;
    let reference_clips =
        resolve_minimax_h3_clip_conditioning(api, settings, job, request, project_path, frames)
            .await?;
    Ok(minimax_h3_conditioning(
        first_frame,
        last_frame,
        reference_images,
        reference_clips,
        reference_audio,
    ))
}

/// Real MLX MiniMax-H3 generation (epic 17137 / sc-19508): build the [`VideoGenInput`] and run the
/// shared [`generate_video`](super::wan::generate_video) heartbeat funnel.
///
/// Returns the decoded clip — video AND its synchronized stereo soundtrack, which the funnel
/// already carries through `DecodedVideo::audio` from `GenerationOutput::Video { audio, .. }` —
/// plus the `rawSettings` to record. The arm owns the raw settings because the tier that actually
/// LOADED and the task that actually DENOISED are only knowable inside it.
#[cfg(target_os = "macos")]
pub(super) async fn generate_minimax_h3(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    generate_minimax_h3_using(
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

/// [`generate_minimax_h3`] with the engine loader supplied by the caller (the
/// `generate_mochi_using` / `generate_krea_realtime_using` seam, sc-12318).
///
/// With the loader threaded in, a test drives this whole arm against a stub `Generator` and asserts
/// on the `GenerationRequest` that actually reached the engine — the conditioning, the coerced
/// frame count, the fps, the seed, and the ABSENCE of the guidance/negative-prompt fields the
/// checkpoint rejects — with no weights, no GPU, and no registered MiniMax-H3 engine. That last
/// part is why this seam matters more here than anywhere else: it is the only way any of this arm
/// is covered before sc-18650 moves the pin.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_minimax_h3_using(
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
    // Shape before media: a request aimed at the wrong DiT partition must be refused before it
    // costs an asset decode, and long before it costs a 53 GB text-encoder load.
    minimax_h3_validate_partition(request)?;
    // Resolved ONCE and passed to both the conditioning resolve and the engine input: a reference
    // clip is truncated to the generated frame count, so the two must be the same number. Deriving
    // it twice from `request` would be one edit away from conditioning a 145-frame render on a
    // 209-frame reference.
    let frames = video_frame_count(&request.model, request.raw_frame_count());
    let conditioning =
        resolve_minimax_h3_conditioning(api, settings, job, request, project_path, frames).await?;
    let load = resolve_minimax_h3_load(settings, request)?;
    let task = minimax_h3_task(&conditioning);
    // The turbo recipe (sc-18726) is resolved BEFORE the adapters so a contradictory pair of
    // accelerators is refused by name rather than after a LoRA file has been opened and classified.
    let (steps, scheduler_shift, turbo) = minimax_h3_sampling(request)?;
    let adapters = resolve_minimax_h3_adapters(settings, request)?;
    let raw_settings = minimax_h3_raw_settings(
        request,
        load.tier,
        task,
        &conditioning,
        steps,
        scheduler_shift,
        turbo,
    );
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        // `spec.weights` is the UPSTREAM root (shared components); the tier's DiT rides
        // `components["transformer"]`. See fact 4 in the module header.
        model_dir: load.root,
        dit_component_dir: Some(load.dit_dir),
        // The packed q4/q8 encoder (sc-19120), staged as `components["text_encoder"]` — the only
        // seam `mlx-gen-minimax-h3::resolve_text_encoder_dir` reads. `None` at bf16 ⇒ the engine's
        // own `<root>/text_encoder` fallback, which is where the dense upstream component is.
        //
        // NOT `text_encoder_dir`: that field lands in `LoadSpec::text_encoder`, which this engine
        // never reads, so a q4/q8 job would fall back to a `<root>/text_encoder` those tiers do not
        // have and hard-error inside the engine.
        text_encoder_component_dir: load.text_encoder_dir,
        quant: load.quant,
        // sc-18726. Was an unconditional `Vec::new()`: every selected MiniMax-H3 LoRA was dropped
        // between the payload and the engine, so the turbo adapters sc-18725 catalogued could be
        // picked and never loaded. See [`resolve_minimax_h3_adapters`].
        adapters,
        conditioning,
        prompt: request.prompt.clone(),
        // Guidance-distilled: the checkpoint has no unconditional branch and the engine REJECTS
        // both fields. The manifest declares both axes false so the Video Studio hides the
        // controls; passing `None` here is the worker half of that same claim.
        negative_prompt: None,
        guidance: None,
        width: request.width,
        height: request.height,
        // The `17n + 5` lattice (sc-17158), dispatched BY MODEL through the shared ladder. NOT the
        // Wan `4k + 1` stride: the two lattices are not nested, so the Wan coercion would hand the
        // engine an off-lattice count it hard-rejects (the engine never refits).
        frames,
        fps: request.fps,
        // MODEL EVALUATIONS, not sigma grid points — the engine appends the terminal σ = 0 itself
        // (`JointSchedule::with_shifts(evaluations + 1, ..)`), so this number is passed through with
        // no ±1 anywhere on this side. `None` ⇒ the engine's own DEFAULT_STEPS (50); a selected turbo
        // variant supplies its own 4 or 8 ([`minimax_h3_sampling`]).
        steps,
        // The video sigma shift (sc-18729). `None` ⇒ the engine's own VIDEO_SIGMA_SHIFT; the turbo
        // 4-step recipe wants a lower one, so it is a real per-request axis rather than a constant.
        scheduler_shift,
        seed: resolve_video_seed(request) as u64,
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
    Ok((decoded, raw_settings))
}

/// The exact tier selected by the product's shared `mlxQuantize` request field. MiniMax-H3's
/// manifest default is q4; non-positive values explicitly select the dense upstream bf16 route.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_minimax_h3_tier(request: &VideoRequest) -> (&'static str, Option<Quant>) {
    let bits = request.advanced.get("mlxQuantize").and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    });
    match bits {
        Some(bits) if bits <= 0 => ("bf16", None),
        Some(bits) if bits <= 4 => ("q4", Some(Quant::Q4)),
        Some(_) => ("q8", Some(Quant::Q8)),
        None => ("q4", Some(Quant::Q4)),
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_minimax_h3_snapshot_dir(settings: &Settings, repo: &str) -> WorkerResult<PathBuf> {
    huggingface_snapshot_dir(&settings.data_dir, repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "candle MiniMax-H3 weights snapshot not found for {repo}"
        ))
    })
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_minimax_h3_component_dir(root: &Path, component: &str) -> WorkerResult<PathBuf> {
    let dir = root.join(component);
    if dir.is_dir() && dir.join("config.json").is_file() {
        Ok(dir)
    } else {
        Err(WorkerError::InvalidPayload(format!(
            "candle MiniMax-H3 installed snapshot is incomplete: missing {component}/config.json under {}",
            root.display()
        )))
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CandleMiniMaxH3Load {
    pub(super) root: PathBuf,
    pub(super) dit_dir: Option<PathBuf>,
    pub(super) text_encoder_dir: Option<PathBuf>,
    pub(super) quant: Option<Quant>,
    pub(super) tier: &'static str,
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn resolve_candle_minimax_h3_load(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<CandleMiniMaxH3Load> {
    let (tier, quant) = candle_minimax_h3_tier(request);
    let root = candle_minimax_h3_snapshot_dir(settings, CANDLE_MINIMAX_H3_REPO)?;
    let is_reference = request.model == "minimax_h3_ref";
    let tier_root = (quant.is_some() || is_reference)
        .then(|| candle_minimax_h3_snapshot_dir(settings, CANDLE_MINIMAX_H3_TIER_REPO))
        .transpose()?
        .map(|root| root.join(tier));
    let dit_dir = match (tier_root.as_deref(), is_reference, quant.is_some()) {
        (Some(tier_root), true, _) => {
            // The provider's `transformer` component is always the base partition. Ref2VA derives
            // `transformer_ref` as its sibling, so validate both halves but stage only the base.
            let transformer = candle_minimax_h3_component_dir(tier_root, "transformer")?;
            candle_minimax_h3_component_dir(tier_root, "transformer_ref")?;
            Some(transformer)
        }
        (Some(tier_root), false, true) => {
            Some(candle_minimax_h3_component_dir(tier_root, "transformer")?)
        }
        _ => None,
    };
    let text_encoder_dir = if quant.is_some() {
        Some(candle_minimax_h3_component_dir(
            tier_root.as_deref().expect("packed tier root"),
            "text_encoder",
        )?)
    } else {
        candle_minimax_h3_component_dir(&root, "text_encoder")?;
        None
    };
    Ok(CandleMiniMaxH3Load {
        root,
        dit_dir,
        text_encoder_dir,
        quant,
        tier,
    })
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) async fn generate_candle_minimax_h3(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    minimax_h3_validate_partition(request)?;
    let frames = video_frame_count(&request.model, request.raw_frame_count());
    let conditioning =
        resolve_minimax_h3_conditioning(api, settings, job, request, project_path, frames).await?;
    let (steps, scheduler_shift, turbo) = minimax_h3_sampling(request)?;
    let adapters = resolve_minimax_h3_adapters(settings, request)?;
    let load = resolve_candle_minimax_h3_load(settings, request)?;
    let adapter_bytes =
        gen_core::adapter_stack_resident_bytes(&adapters, gen_core::AdapterResidencyMode::Additive)
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "MiniMax-H3 cannot determine the resident size of the requested adapter stack."
                        .to_owned(),
                )
            })?;
    if let Some(error) = crate::vram_gate::minimax_h3_fit_error(
        &request.model_manifest_entry,
        load.tier,
        adapter_bytes,
        &settings.gpu_id,
        super::candle::candle_video_vram_budget(settings).await,
    ) {
        return Err(error);
    }
    let task = minimax_h3_task(&conditioning);
    let mut raw_settings = minimax_h3_raw_settings(
        request,
        load.tier,
        task,
        &conditioning,
        steps,
        scheduler_shift,
        turbo,
    );
    if let Value::Object(raw) = &mut raw_settings {
        raw.insert(
            "repo".to_owned(),
            Value::String(CANDLE_MINIMAX_H3_REPO.to_owned()),
        );
    }
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir: load.root,
        quant: load.quant,
        dit_component_dir: load.dit_dir,
        text_encoder_component_dir: load.text_encoder_dir,
        adapters,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt: None,
        guidance: None,
        width: request.width,
        height: request.height,
        frames,
        fps: request.fps,
        steps,
        scheduler_shift,
        seed: resolve_video_seed(request) as u64,
        offload_policy: OffloadPolicy::Sequential,
        ..VideoGenInput::default()
    };
    let decoded = generate_video_using(
        api,
        settings,
        job,
        backend,
        &request.advanced,
        input,
        crate::inference_runtime::load,
    )
    .await?;
    Ok((decoded, raw_settings))
}
