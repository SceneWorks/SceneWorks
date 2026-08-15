#[allow(unused_imports)]
use super::prelude::*;
#[cfg(target_os = "macos")]
use super::wan::{advanced_opt_f32, advanced_opt_u32, generate_video_using, VideoGenInput};
#[cfg(target_os = "macos")]
use sceneworks_core::video_request::{classify_reference_set, ReferenceSetVerdict};

// ---------------------------------------------------------------------------
// MiniMax-H3 / Hailuo 3.0 (epic 17137, sc-19508): a joint audio+video family that emits video AND
// synchronized stereo audio in ONE denoise pass. macOS/MLX only (`mac_only: true`, no candle
// engine).
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
//  3. **The load probes BOTH partitions.** `mlx-gen-minimax-h3::load` requires
//     `{dit}/config.json` AND its sibling `{dit}/../transformer_ref/config.json`, because ref2va is
//     a first-class task of the engine rather than an optional extra. The manifest ships the two
//     partitions as SEPARATE downloads (`minimax_h3` pulls `q4/transformer/*`, `minimax_h3_ref`
//     pulls `q4/transformer_ref/*`), so an install of either entry alone is not loadable —
//     [`resolve_minimax_h3_load`] requires both and says which half is missing.
//
//  4. **No guidance, no negative prompt.** The checkpoint is guidance-distilled: there is no
//     unconditional branch anywhere in it. The manifest declares
//     `video.supportsGuidance = false` / `supportsNegativePrompt = false` and the engine REJECTS
//     both fields, so this arm passes `None` for each rather than forwarding a knob the engine
//     would refuse.
//
// Installed layout — the DiT is the ONE tiered component, and it lives in a DIFFERENT repo from
// every shared component:
//
//     <tier root>/{q4,q8,bf16}/{transformer,transformer_ref}/   <- SceneWorks/minimax-h3-mlx
//     <base root>/{text_encoder,tokenizer,vae,audio_vae,FL2VA}/ <- MiniMaxAI/MiniMax-H3 (coRequisite)
//
// so `spec.weights` is the BASE root and the tier's `transformer/` is staged as the
// [`MINIMAX_H3_DIT_COMPONENT`]. Handing the loader the tier root instead would fail on
// `text_encoder/`, and handing it the base root alone would load the unquantized upstream DiT.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX MiniMax-H3 asset.
#[cfg(target_os = "macos")]
pub(super) const MINIMAX_H3_ADAPTER: &str = "mlx_minimax_h3";

/// The ONE registry id both catalog entries load. See fact 1 above — the partition split is a
/// directory, not a provider.
#[cfg(target_os = "macos")]
pub(super) const MINIMAX_H3_ENGINE_ID: &str = "minimax_h3";

/// The staged-component key `mlx-gen-minimax-h3::model::DIT_COMPONENT` reads the tiered DiT from.
/// Named `"transformer"` even when the request will denoise on `transformer_ref/`: the provider
/// resolves the reference partition as this directory's SIBLING, so the base partition is always
/// what is staged.
#[cfg(target_os = "macos")]
pub(super) const MINIMAX_H3_DIT_COMPONENT: &str = "transformer";

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
#[cfg(target_os = "macos")]
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
/// Deriving it beats asserting it: at the current pin (`014134e3`) the descriptor is absent and the
/// refusal fires; the moment sc-18650 lands a bundle that registers `minimax_h3`, the descriptor
/// appears and this arm goes live with **no code change**. A hard-coded revision string could only
/// have gone stale.
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

/// Whether `root` carries the SHARED components the loader probes under `spec.weights`. Every one
/// of these is a `coRequisite` download from a DIFFERENT repo than the tiers, so a user can
/// genuinely have a complete `q4/` and no text encoder.
///
/// The probe list mirrors `mlx-gen-minimax-h3::load`'s own: `vae/`, `audio_vae/`, `text_encoder/`,
/// `tokenizer/` and the `FL2VA/audio_vae/` constructor documents (the repackaged root config
/// carries none of the audio VAE's constructor arguments).
#[cfg(target_os = "macos")]
pub(super) fn minimax_h3_shared_is_complete(root: &Path) -> bool {
    ["text_encoder", "tokenizer", "vae", "audio_vae"]
        .iter()
        .all(|component| root.join(component).is_dir())
        && root.join("FL2VA").join("audio_vae").is_dir()
}

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
    if !minimax_h3_shared_is_complete(&base_root) {
        return Err(WorkerError::InvalidPayload(format!(
            "{}: the shared MiniMax-H3 text encoder / tokenizer / VAEs are incomplete (expected \
             text_encoder/, tokenizer/, vae/, audio_vae/ and FL2VA/audio_vae/ under {}). These are \
             a separate co-requisite download from the quality tiers — re-download MiniMax-H3 in \
             the Model Manager.",
            request.model,
            base_root.display()
        )));
    }
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
    Ok(MiniMaxH3Load {
        root: base_root,
        dit_dir: tier_root.join(tier).join(MINIMAX_H3_DIT_COMPONENT),
        quant: minimax_h3_tier_quant(tier),
        tier,
    })
}

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
#[cfg(target_os = "macos")]
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
        // `Conditioning::ReferenceVideo` — the ONLY variant that carries a reference clip's own
        // frame rate — arrives with the sc-18650 pin bump. It is genuinely absent from the pinned
        // gen-core, so this shape cannot be built here yet. Refuse it by name rather than
        // downgrading it to `Conditioning::VideoClip`: the engine deliberately does NOT advertise
        // `VideoClip` (it has no in-context clip mechanism), so that substitution would be refused
        // as `Unsupported` after the load — or, worse, silently accepted by some future provider as
        // a completely different mechanism. Tracked as the sc-19508 follow-up.
        if clip_refs > 0 {
            return Err(WorkerError::InvalidPayload(format!(
                "{}: video references are not renderable in this build — the conditioning variant \
                 that carries a reference clip's own frame rate arrives with the inference pin \
                 bump (sc-18650). Image and audio references render now; remove the \
                 sourceClipAssetIds entries, or wait for the pin. No output was produced.",
                request.model
            )));
        }
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
/// The order is deliberate: at the current pin every MiniMax-H3 job stops at (1) with the same
/// honest reason a user got before, and an unprovisioned install after the pin bump stops at (3)
/// with the resolver's precise error — which is exactly the substitution sc-19508 asked for
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
#[cfg(target_os = "macos")]
pub(super) fn minimax_h3_raw_settings(request: &VideoRequest, tier: &str, task: &str) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert("minimaxH3Tier".to_owned(), Value::String(tier.to_owned()));
    raw.insert("minimaxH3Task".to_owned(), Value::String(task.to_owned()));
    Value::Object(raw)
}

/// The engine task a resolved conditioning list denoises as — the SceneWorks-side mirror of
/// `MiniMaxH3Task::resolve`, derived from the SAME input the engine derives it from (the
/// conditioning itself, not the mode string or the model id).
///
/// Reading the conditioning rather than the request is what makes this honest: if the assembly
/// below ever drops a reference or a keyframe, the recorded task changes with it instead of
/// continuing to claim what the payload asked for.
#[cfg(target_os = "macos")]
pub(super) fn minimax_h3_task(conditioning: &[Conditioning]) -> &'static str {
    let has_keyframes = conditioning
        .iter()
        .any(|c| matches!(c, Conditioning::Keyframe { .. }));
    let has_references = conditioning.iter().any(|c| {
        matches!(
            c,
            Conditioning::Reference { .. } | Conditioning::ReferenceAudio { .. }
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
/// `<Audio j>` in list order and advances one shared rotary clock across them, so re-ordering the
/// list changes the render. Images precede audio because that is the only order the payload can
/// express — `referenceAssetIds` and `referenceAudioAssetIds` are two separate lists with no
/// interleaving — and the caller's order WITHIN each list is preserved exactly.
#[cfg(target_os = "macos")]
pub(super) fn minimax_h3_conditioning(
    first_frame: Option<Image>,
    last_frame: Option<Image>,
    reference_images: Vec<Image>,
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
    conditioning.extend(reference_audio);
    conditioning
}

/// Resolve a MiniMax-H3 request's supplied media into the engine conditioning.
///
/// `sourceAssetId` → the FIRST keyframe, `lastFrameAssetId` → the LAST — fl2va takes 0, 1 or 2 of
/// them, so `image_to_video` (first only), a last-frame-only payload, and `first_last_frame` (both)
/// are one code path rather than three modes. `referenceAssetIds` → ordered image references and
/// `referenceAudioAssetIds` → ordered audio references, which together are the Ref2VA set.
///
/// The audio references go through the SHARED `resolve_reference_audio_conditioning` (sc-17160) —
/// the same function `run_video_generate_job` already calls to prove they resolve before the job is
/// marked Running. That call discards its value with a comment naming this arm as the consumer;
/// this is that consumer, and it deliberately re-uses the function rather than re-implementing the
/// project-scoped lookup and the `safe_project_path` guard.
#[cfg(target_os = "macos")]
pub(super) fn resolve_minimax_h3_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
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
    let reference_audio =
        super::resolve_reference_audio_conditioning(settings, request, project_path)?;
    Ok(minimax_h3_conditioning(
        first_frame,
        last_frame,
        reference_images,
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
    let conditioning = resolve_minimax_h3_conditioning(settings, request, project_path)?;
    let load = resolve_minimax_h3_load(settings, request)?;
    let task = minimax_h3_task(&conditioning);
    let raw_settings = minimax_h3_raw_settings(request, load.tier, task);
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        // `spec.weights` is the UPSTREAM root (shared components); the tier's DiT rides
        // `components["transformer"]`. See fact 4 in the module header.
        model_dir: load.root,
        dit_component_dir: Some(load.dit_dir),
        quant: load.quant,
        adapters: Vec::new(),
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
        frames: video_frame_count(&request.model, request.raw_frame_count()),
        fps: request.fps,
        // `None` ⇒ the engine's own DEFAULT_STEPS (50).
        steps: advanced_opt_u32(request, "steps"),
        // The video sigma shift (sc-18729). `None` ⇒ the engine's own VIDEO_SIGMA_SHIFT; the turbo
        // 4-step recipe wants a lower one, so it is a real per-request axis rather than a constant.
        scheduler_shift: advanced_opt_f32(request, "schedulerShift"),
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
