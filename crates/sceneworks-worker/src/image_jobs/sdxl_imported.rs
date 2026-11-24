use super::adapters::resolve_prepared_adapters;
use super::adapters::PreparedAdapters;
use super::checkpoint_plan::checkpoint_plan_bespoke_primary_pin;
use super::checkpoint_plan::request_is_checkpoint_plan_backed;
use super::conditioning::load_reference_image;
use super::conditioning::non_empty;
#[cfg(test)]
use super::conditioning::pose_entries;
use super::events::consume_gen_events;
use super::geometry::fit_engine_image;
use super::krea_imported::imported_dit_file;
use super::krea_imported::imported_model_quant;
use super::mlx_model;
use super::pid::resolve_pid_weights;
use super::render::drive_hires_passes;
#[cfg(target_os = "macos")]
use super::render::hires_first_pass_context;
use super::render::HiresFixPlan;
use super::render::WorkerExecutionReport;
use super::resolve_seed;
use super::routing::PreparedFileDispatch;
use super::sampling::normalize_sampling_knob;
use super::sampling::read_resolved_image_guidance_method;
use super::sampling::read_resolved_image_sampling_knobs;
use super::sampling::resolved_option_or_manifest_f32;
use super::sampling::resolved_option_or_manifest_u32;
#[cfg(target_os = "macos")]
use super::sdxl::SDXL_EDIT_STRENGTH;
#[cfg(target_os = "macos")]
use super::sdxl::SDXL_INPAINT_STRENGTH;
use super::stream::drive_gen_items;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::stream::start_cached_gen_stream_after_cold_admission_with_placement;
#[cfg(target_os = "macos")]
use super::stream::start_cached_gen_stream_with_request_state;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::stream::ColdLoadAdmission;
use super::stream::GenEvent;
use super::CancelFlag;
use super::Conditioning;
use super::GenerationOutput;
use super::GenerationRequest;
use super::Generator;
use super::Image;
use super::ImageRequest;
use super::LoadSpec;
use super::Progress;
use super::PromptEnhance;
use super::WeightsSource;
use crate::ApiClient;
use crate::JobSnapshot;
use crate::Path;
use crate::PathBuf;
use crate::Settings;
use crate::Uuid;
use crate::Value;
use crate::WorkerError;
use crate::WorkerResult;
use serde_json::json;
// Shared MLX/Candle in-place loader for a fused SDXL LDM/A1111 checkpoint (sc-14024).
// The checkpoint supplies UNet + CLIP-L + OpenCLIP-bigG + VAE. MLX borrows only tokenizer assets
// from the installed SDXL turnkey; candle stages its existing tokenizer + fp16-fix VAE components.

#[cfg(target_os = "macos")]
pub(super) const SDXL_IMPORTED_ENGINE: &str = "mlx_sdxl_imported";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const SDXL_IMPORTED_ENGINE: &str = "candle_sdxl_imported";

const SDXL_IMPORTED_DEFAULT_STEPS: u32 = 30;
const SDXL_IMPORTED_DEFAULT_GUIDANCE: f32 = 7.0;
// On macOS these shared defaults are supplied by the advanced SDXL route included in the same
// module. The Windows Candle build does not include that MLX-only file, so keep its imported SDXL
// conditioning defaults explicit and byte-identical here.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const SDXL_EDIT_STRENGTH: f32 = 0.6;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) const SDXL_INPAINT_STRENGTH: f32 = 0.85;
pub(super) const SDXL_CLIP_L_REPO: &str = "openai/clip-vit-large-patch14";
pub(super) const SDXL_CLIP_L_REVISION: &str = "32bd64288804d66eefd0ccbe215aa642df71cc41";
// Compiled on every platform: `SDXL_IMPORTED_COMPONENT_SOURCES` is now the ONE table both the
// bespoke candle lane and the backend-agnostic plan route read, so its cells cannot be gated to one
// backend without splitting it in two.
const SDXL_CLIP_BIGG_REPO: &str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k";
pub(super) const SDXL_CLIP_BIGG_REVISION: &str = "743c27bd53dfe508a0ade0f50698f99b39d03bec";
const SDXL_VAE_REPO: &str = "madebyollin/sdxl-vae-fp16-fix";
const SDXL_VAE_REVISION: &str = "207b116dae70ace3637169f1ddd2434b91b3a8cd";

pub(super) struct PreparedSdxlImportedSources {
    checkpoint: gen_core::PinnedWeightsFile,
    adapters: PreparedAdapters,
}

pub(super) fn resolve_imported_sdxl_pin(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<gen_core::PinnedWeightsFile>> {
    // A plan-backed entry (`importPlan.checkpointId`, epic 20398) belongs to the plan-driven
    // route; this bespoke lane never also claims it, so one entry has exactly one owner. Explicit
    // rather than left to arm ordering in the resolver (sc-20634 review): ordering is not a claim.
    if request_is_checkpoint_plan_backed(request) {
        return Ok(None);
    }
    if request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
        != Some("sdxl")
    {
        return Ok(None);
    }
    // A plan-backed entry whose request the plan route does NOT claim (edit, LoRA, Hires.fix,
    // reference-guided init) still runs here, on the plan's verified FUSED layer rather than a
    // directory scan — the same inputs the plan route would hand the same provider, so a fixed seed
    // renders equal on either route, and a LINKED fused checkpoint reaches those shapes at all
    // (sc-20644 SDXL row).
    if let Some(pin) = checkpoint_plan_bespoke_primary_pin(request, settings, "sdxl")? {
        return Ok(Some(pin));
    }
    // A builtin SDXL engine id (in `MODEL_TABLE`) loads from its snapshot turnkey via the normal MLX
    // lane — never through the single-file entrypoint. Checked AFTER the plan pin, mirroring
    // `resolve_imported_krea_dit_pin`: a plan-backed entry must be offered its verified layer before
    // a builtin-id test can send it down the turnkey path (feature-end round 1, ordering minor).
    if mlx_model(&request.model).is_some() {
        return Ok(None);
    }
    let Some(raw_path) = request
        .advanced_text("modelPath")
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .or_else(|| {
            request
                .model_manifest_entry
                .get("modelPath")
                .and_then(crate::payload::trimmed_value_str)
        })
        .or_else(|| {
            request
                .model_manifest_entry
                .get("paths")
                .and_then(|paths| paths.get("model"))
                .and_then(crate::payload::trimmed_value_str)
        })
    else {
        return Ok(None);
    };
    let confined = crate::paths::normalize_app_managed_model_path(
        settings,
        raw_path,
        "Imported SDXL checkpoint",
    )?;
    let candidate = if confined.is_dir() {
        // Select inside the already-confined target, then reconstruct the caller-visible child and
        // re-confine/pin that exact lexical entry. A child symlink must never inherit trust from its
        // parent directory.
        imported_dit_file(&confined)
            .and_then(|file| file.file_name().map(|name| Path::new(raw_path).join(name)))
    } else if imported_dit_file(Path::new(raw_path)).is_some() {
        Some(PathBuf::from(raw_path))
    } else {
        None
    };
    candidate
        .map(|file| {
            crate::paths::pin_app_managed_model_file(settings, &file, "Imported SDXL checkpoint")
        })
        .transpose()
}

#[cfg(test)]
pub(super) fn sdxl_imported_request_shape_available(request: &ImageRequest) -> bool {
    sdxl_imported_request_shape_available_with_pose_count(request, pose_entries(request).len())
}

pub(super) fn sdxl_imported_request_shape_available_resolved(
    execution: &super::conditioning::ResolvedImageExecution,
) -> bool {
    sdxl_imported_request_shape_available_with_pose_count(execution, execution.pose_source_count())
}

fn sdxl_imported_request_shape_available_with_pose_count(
    request: &ImageRequest,
    pose_count: usize,
) -> bool {
    if sceneworks_core::jobs_store::imported_control_intent_is_material_resolved(
        request
            .resolved_advanced
            .as_ref()
            .expect("image options must be resolved"),
    ) {
        return false;
    }
    let operation = if request.mode == "edit_image" {
        gen_core::ImportedModelOperation::Edit
    } else {
        gen_core::ImportedModelOperation::Generate
    };
    let Some(descriptor) = crate::inference_runtime::imported_model_descriptor(
        "sdxl",
        gen_core::ImportedModelSource::FusedCheckpoint,
        operation,
    ) else {
        return false;
    };
    let caps = &descriptor.capabilities;
    if imported_model_quant(request, &descriptor, "Imported SDXL").is_err() {
        return false;
    }
    let has_phases = request
        .resolved_advanced
        .as_ref()
        .expect("image options must be resolved")
        .phases
        .as_ref()
        .is_some_and(|phases| !phases.value.is_empty());
    if pose_count > 0
        || !request.reference_asset_ids.is_empty()
        || request.character_id.is_some()
        || request.character_look_id.is_some()
        || has_phases
        || (!request.loras.is_empty() && !(caps.supports_lora || caps.supports_lokr))
    {
        return false;
    }
    let has_reference = non_empty(&request.reference_asset_id);
    let has_source = non_empty(&request.source_asset_id);
    if request.mode == "edit_image" {
        if !(has_source || has_reference)
            || !caps
                .conditioning
                .contains(&gen_core::ConditioningKind::Reference)
        {
            return false;
        }
        if non_empty(&request.mask_asset_id)
            && !caps
                .conditioning
                .contains(&gen_core::ConditioningKind::Mask)
        {
            return false;
        }
    } else if has_source
        || non_empty(&request.mask_asset_id)
        || (has_reference
            && !caps
                .conditioning
                .contains(&gen_core::ConditioningKind::Reference))
    {
        return false;
    }
    true
}

#[cfg(test)]
pub(super) fn resolve_imported_sdxl_file(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    Ok(resolve_imported_sdxl_pin(request, settings)?.map(|pin| pin.loader_path().to_path_buf()))
}

#[cfg(test)]
pub(super) fn sdxl_imported_available(request: &ImageRequest, settings: &Settings) -> bool {
    sdxl_imported_request_shape_available(request)
        && matches!(resolve_imported_sdxl_file(request, settings), Ok(Some(_)))
}

/// Resolve imported SDXL img2img/edit/inpaint/outpaint conditioning. The exact registered provider
/// owns the capability claim; this helper only turns SceneWorks asset ids into that contract.
fn resolve_imported_sdxl_conditioning(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<(Vec<Conditioning>, &'static str, Option<f32>)> {
    let (width, height) = (request.width, request.height);
    if request.mode != "edit_image" {
        let Some(asset_id) = request
            .reference_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
        else {
            return Ok((Vec::new(), "text_to_image", None));
        };
        let image = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            asset_id,
            project_path,
        )?;
        let image = fit_engine_image(image, width, height, &request.fit_mode)?;
        let strength = super::sampling::resolved_f32_clamped(
            request,
            "strength",
            SDXL_EDIT_STRENGTH,
            0.0..=1.0,
        );
        return Ok((
            vec![Conditioning::Reference {
                image,
                strength: Some(strength),
            }],
            "img2img",
            Some(strength),
        ));
    }

    let source_id = request
        .source_asset_id
        .as_deref()
        .or(request.reference_asset_id.as_deref())
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Imported SDXL edit requires a source image".to_owned())
        })?;
    let source = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        source_id,
        project_path,
    )?;
    let outpaint = request.fit_mode == "outpaint";
    let masked = non_empty(&request.mask_asset_id);
    let (src_w, src_h) = (source.width, source.height);
    let source = fit_engine_image(source, width, height, &request.fit_mode)?;
    let strength = super::sampling::resolved_f32_clamped(
        request,
        "strength",
        if masked || outpaint {
            SDXL_INPAINT_STRENGTH
        } else {
            SDXL_EDIT_STRENGTH
        },
        0.0..=1.0,
    );
    let mut conditioning = vec![Conditioning::Reference {
        image: source,
        strength: Some(strength),
    }];
    if masked || outpaint {
        let mut mask = if outpaint {
            gen_core::imageops::outpaint_border_mask(src_w, src_h, width, height)
        } else {
            let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
            let mask = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                mask_id,
                project_path,
            )?;
            fit_engine_image(mask, width, height, &request.fit_mode)?
        };
        if outpaint && masked {
            let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
            let user_mask = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                mask_id,
                project_path,
            )?;
            let user_mask = fit_engine_image(user_mask, width, height, "pad")?;
            mask = gen_core::imageops::union_masks(&mask, &user_mask).map_err(|error| {
                crate::generation_error("Imported SDXL outpaint mask union failed", error)
            })?;
        }
        conditioning.push(Conditioning::Mask { image: mask });
    }
    Ok((
        conditioning,
        if outpaint {
            "outpaint"
        } else if masked {
            "inpaint"
        } else {
            "edit"
        },
        Some(strength),
    ))
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn prepare_sdxl_imported_sources(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PreparedSdxlImportedSources>> {
    prepare_sdxl_imported_sources_with_shape(
        request,
        settings,
        sdxl_imported_request_shape_available(request),
    )
}

pub(super) fn prepare_sdxl_imported_sources_resolved(
    execution: &super::conditioning::ResolvedImageExecution,
    settings: &Settings,
) -> WorkerResult<Option<PreparedSdxlImportedSources>> {
    prepare_sdxl_imported_sources_with_shape(
        execution,
        settings,
        sdxl_imported_request_shape_available_resolved(execution),
    )
}

fn prepare_sdxl_imported_sources_with_shape(
    request: &ImageRequest,
    settings: &Settings,
    shape_available: bool,
) -> WorkerResult<Option<PreparedSdxlImportedSources>> {
    if !shape_available {
        return Ok(None);
    }
    let Some(checkpoint) = resolve_imported_sdxl_pin(request, settings)? else {
        return Ok(None);
    };
    Ok(Some(PreparedSdxlImportedSources {
        checkpoint,
        adapters: resolve_prepared_adapters(request, settings)?,
    }))
}

async fn stage_sdxl_component_file(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    repo: &str,
    revision: &str,
    file: &str,
    destination: &Path,
) -> WorkerResult<()> {
    let client = crate::downloads::streaming_download_client();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "SDXL generation canceled while staging shared components.",
        fresh_download: false,
    };
    crate::downloads::ensure_hf_cached_file_atomically(&context, repo, revision, file, destination)
        .await?;
    Ok(())
}

#[cfg(target_os = "macos")]
async fn stage_sdxl_tokenizer(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PathBuf> {
    let root = settings
        .data_dir
        .join("cache")
        .join("sdxl-imported-components");
    for file in ["vocab.json", "merges.txt"] {
        stage_sdxl_component_file(
            api,
            settings,
            job,
            SDXL_CLIP_L_REPO,
            SDXL_CLIP_L_REVISION,
            file,
            &root.join("tokenizer").join(file),
        )
        .await?;
    }
    Ok(root)
}

/// The `ldm_tokenizer` directory a fused SDXL checkpoint pairs with, resolved **cache-only** for the
/// plan-driven route: a root whose `tokenizer/` subdir holds CLIP-L's `vocab.json` + `merges.txt`
/// (`mlx-gen-sdxl`'s `loader::load_tokenizer` joins `tokenizer` onto whatever directory it is
/// handed).
///
/// Lives in THIS file, not in the plan route, for two reasons: the staging root and the pinned
/// CLIP-L revision are this lane's, so both routes hand the loader one directory rather than two
/// copies; and `<data>/cache/sdxl-imported-components` is a destination epic 17625's guard already
/// attributes to `image_jobs/sdxl_imported.rs` — a second file naming the same literal would count
/// as a second weights destination against a ratchet that only goes down, for no new destination.
///
/// This never DOWNLOADS. It mirrors two ~1 MB model-agnostic assets that are already in the
/// app-managed Hugging Face cache, and refuses with an actionable install-it-from-the-Model-Manager
/// diagnostic when they are not — during PLANNING, not inside a loader (epic 17625 AC9 / E7).
///
/// Without it the plan route CLAIMED a plan-backed fused SDXL text-to-image request — the SDXL
/// adapter binds `FusedCheckpoint`/`Generate` on this backend — and then refused it with
/// `[checkpoint-plan:missing-component] … 'ldm_tokenizer'`, so the family had no servable shape at
/// all on the universal route (sc-20644 SDXL row).
pub(super) fn resolve_sdxl_ldm_tokenizer_root_cache_only(
    settings: &Settings,
    checkpoint_id: &str,
) -> WorkerResult<PathBuf> {
    let root = settings
        .data_dir
        .join("cache")
        .join("sdxl-imported-components");
    let tokenizer_dir = root.join("tokenizer");
    for file in ["vocab.json", "merges.txt"] {
        mirror_cached_sdxl_component_file(
            settings,
            checkpoint_id,
            SDXL_CLIP_L_REPO,
            SDXL_CLIP_L_REVISION,
            file,
            &tokenizer_dir,
        )?;
    }
    Ok(root)
}

/// Mirror ONE already-installed Hugging Face asset into the SDXL staging root, cache-only.
///
/// The single copy-and-rename both cache-only resolvers use, so the two of them cannot drift in how
/// they refuse a missing install or in how they avoid publishing a half-written file.
pub(super) fn mirror_cached_sdxl_component_file(
    settings: &Settings,
    checkpoint_id: &str,
    repo: &str,
    revision: &str,
    file: &str,
    destination_dir: &Path,
) -> WorkerResult<()> {
    mirror_cached_sdxl_component_file_with_copy(
        settings,
        checkpoint_id,
        repo,
        revision,
        file,
        destination_dir,
        |source, staging| std::fs::copy(source, staging).map(|_| ()),
    )
}

pub(super) fn mirror_cached_sdxl_component_file_with_copy(
    settings: &Settings,
    checkpoint_id: &str,
    repo: &str,
    revision: &str,
    file: &str,
    destination_dir: &Path,
    copy_source: impl FnOnce(&Path, &Path) -> std::io::Result<()>,
) -> WorkerResult<()> {
    let destination = destination_dir.join(file);
    let staging_failed = |path: &Path, error: std::io::Error| {
        WorkerError::Engine(format!(
            "Checkpoint plan source preparation failed: {} ({error})",
            path.display()
        ))
    };
    let cached = crate::downloads::resolve_hf_component_file(settings, repo, revision, file)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "[checkpoint-plan:missing-component] checkpoint {checkpoint_id:?} (sdxl family) \
                 requires the fused-checkpoint tokenizer assets, and {repo}/{file} is not \
                 installed — install it from the Model Manager, then run this checkpoint again. A \
                 fused SDXL checkpoint carries its own UNet, encoders and VAE; only the \
                 model-agnostic CLIP tokenizer vocabulary comes from outside it."
            ))
        })?;

    // Resolve first so a missing app-managed source reports the existing actionable
    // diagnostic without creating cache state. From this point through validation
    // and publication, one target-scoped lock serializes every process and every
    // async task in a worker. A PID-only staging name is not task-unique: concurrent
    // jobs in one process could otherwise copy into the same inode and let one job
    // publish it while the other was still writing (sc-21689 review).
    let _target_lock = crate::downloads::DownloadLock::acquire(&destination)?;
    if destination.is_file()
        && sdxl_component_files_match(&cached, &destination)
            .map_err(|error| staging_failed(&destination, error))?
    {
        return Ok(());
    }
    std::fs::create_dir_all(destination_dir)
        .map_err(|error| staging_failed(destination_dir, error))?;
    // Keep task-local staging collision-free even though the lock is the primary
    // correctness boundary. This also makes stale scratch from a killed worker
    // incapable of becoming a later task's live write target.
    let staging = destination_dir.join(format!("{file}.{}.partial", Uuid::new_v4().simple()));
    copy_source(&cached, &staging)
        // Windows' FlushFileBuffers requires a write-capable handle. File::open is
        // read-only there and returns AccessDenied from sync_all even though the
        // copy itself succeeded.
        .and_then(|_| std::fs::OpenOptions::new().write(true).open(&staging))
        .and_then(|file| file.sync_all())
        .and_then(|_| {
            sdxl_component_files_match(&cached, &staging).and_then(|matches| {
                if matches {
                    Ok(())
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "staged component does not match its pinned cache source",
                    ))
                }
            })
        })
        .and_then(|_| promote_cached_sdxl_component(&staging, &destination))
        .map_err(|error| {
            let _ = std::fs::remove_file(&staging);
            staging_failed(&destination, error)
        })?;
    Ok(())
}

pub(super) fn promote_cached_sdxl_component(
    staging: &Path,
    destination: &Path,
) -> std::io::Result<()> {
    // Readers see either the prior complete component or the newly copied one. The
    // tempfile's safe platform implementation atomically replaces an existing
    // destination on both Unix and Windows. Never unlink the visible destination
    // before the staged file takes its place.
    tempfile::TempPath::try_from_path(staging)?
        .persist(destination)
        .map_err(|error| error.error)
}

/// Compare the staged copy with its resolved pinned-HF source without buffering a
/// component in memory. This protects cache-only restart paths: a torn or corrupt
/// destination is copied again instead of being accepted just because it exists.
fn sdxl_component_files_match(source: &Path, destination: &Path) -> std::io::Result<bool> {
    use std::io::Read;

    if std::fs::metadata(source)?.len() != std::fs::metadata(destination)?.len() {
        return Ok(false);
    }
    let mut source = std::fs::File::open(source)?;
    let mut destination = std::fs::File::open(destination)?;
    let mut source_buffer = [0_u8; 64 * 1024];
    let mut destination_buffer = [0_u8; 64 * 1024];
    loop {
        let source_read = source.read(&mut source_buffer)?;
        let destination_read = destination.read(&mut destination_buffer)?;
        if source_read != destination_read
            || source_buffer[..source_read] != destination_buffer[..destination_read]
        {
            return Ok(false);
        }
        if source_read == 0 {
            return Ok(true);
        }
    }
}

/// Where each SDXL component id's bytes come from: `(component id, repo, revision, file, subdir)`.
///
/// Catalog data only. WHICH of these the imported fused route actually needs is the registered
/// provider's `required_components`, never this table: a fused LDM checkpoint carries its own VAE,
/// so the candle binding declares `LDM_REQUIRED_COMPONENTS` — the two model-agnostic tokenizers —
/// while the ordinary snapshot route additionally requires the fp16-fix VAE. Staging all three
/// regardless downloaded ~335 MB from `madebyollin/sdxl-vae-fp16-fix` on every imported render and
/// handed the loader a component it drops: `SdxlComponents::from_spec` makes `vae_fp16_fix` optional
/// for a fused source and `load_components` takes the VAE from `ldm.vae`. A pure dead download,
/// removed by asking the descriptor (sc-20644 SDXL row).
///
/// Read by BOTH SDXL routes: the bespoke candle lane stages from it over the network
/// ([`attach_imported_sdxl_components`]), and the universal plan route mirrors from it cache-only
/// ([`resolve_sdxl_imported_component_cache_only`]). One table, so the two routes can never
/// disagree about which repo or pinned revision a component id means (sc-20651).
const SDXL_IMPORTED_COMPONENT_SOURCES: &[(&str, &str, &str, &str, &str)] = &[
    (
        "tokenizer_clip_l",
        SDXL_CLIP_L_REPO,
        SDXL_CLIP_L_REVISION,
        "tokenizer.json",
        "clip-l",
    ),
    (
        "tokenizer_clip_bigg",
        SDXL_CLIP_BIGG_REPO,
        SDXL_CLIP_BIGG_REVISION,
        "tokenizer.json",
        "clip-bigg",
    ),
    (
        "vae_fp16_fix",
        SDXL_VAE_REPO,
        SDXL_VAE_REVISION,
        "diffusion_pytorch_model.safetensors",
        "vae-fp16-fix",
    ),
];

/// Stage exactly the components the RESOLVED provider declares, in its declared order.
///
/// A declared id this build cannot source refuses by name rather than being skipped: skipping would
/// hand the loader an incomplete spec and turn a staging gap into a mid-load failure (E7).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn attach_imported_sdxl_components(
    spec: LoadSpec,
    descriptor: &gen_core::ModelDescriptor,
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<LoadSpec> {
    let root = settings
        .data_dir
        .join("cache")
        .join("sdxl-imported-components");
    let mut spec = spec;
    for component in descriptor.required_components {
        let Some((_, repo, revision, file, subdir)) = SDXL_IMPORTED_COMPONENT_SOURCES
            .iter()
            .find(|(id, ..)| id == component)
        else {
            return Err(WorkerError::InvalidPayload(format!(
                "The registered imported SDXL provider {:?} declares required component \
                 {component:?}, which this build cannot stage",
                descriptor.id
            )));
        };
        let destination_dir = root.join(subdir);
        stage_sdxl_component_file(
            api,
            settings,
            job,
            repo,
            revision,
            file,
            &destination_dir.join(file),
        )
        .await?;
        spec = spec.with_component(*component, WeightsSource::Dir(destination_dir));
    }
    Ok(spec)
}

/// Resolve a declared SDXL component from the same pinned cache and staging layout as the
/// bespoke lane. The table names the entire input of each component (one file) while the
/// provider continues to receive a directory. The file token lets the shared plan account for
/// and validate that exact member without scanning unrelated staging files.
pub(super) fn resolve_sdxl_imported_component_cache_only(
    settings: &Settings,
    family: &str,
    component: &str,
    checkpoint_id: &str,
) -> WorkerResult<Option<(PathBuf, gen_core::PinnedWeightsFile)>> {
    use sceneworks_core::safe_path::{confine_under, Confinement};

    if family != "sdxl" {
        return Ok(None);
    }
    let Some((_, repo, revision, file, subdir)) = SDXL_IMPORTED_COMPONENT_SOURCES
        .iter()
        .find(|(id, ..)| *id == component)
    else {
        return Ok(None);
    };
    let data_dir = crate::paths::normalized_data_dir(settings)?;
    let relative = format!("cache/sdxl-imported-components/{subdir}/{file}");
    let destination = data_dir.join(&relative);
    let destination_dir = destination.parent().expect("staged component has a parent");
    let invalid_destination = |reason: &str| {
        WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:unpriceable-component] checkpoint {checkpoint_id:?} ({family} family) cannot use the staged path for component {component:?}: {reason}"
        ))
    };
    let confined_destination = || {
        confine_under(&data_dir, &relative, Confinement::SymlinkSafe)
            .map_err(|error| invalid_destination(&error.to_string()))
    };
    confined_destination()?;
    mirror_cached_sdxl_component_file(
        settings,
        checkpoint_id,
        repo,
        revision,
        file,
        destination_dir,
    )?;
    let expected_target = confined_destination()?;
    let pin = gen_core::PinnedWeightsFile::pin(&destination)
        .map_err(|error| invalid_destination(&error.to_string()))?;
    if pin.canonical_target_path() != expected_target {
        return Err(invalid_destination("staged target changed while pinning"));
    }
    Ok(Some((destination_dir.to_path_buf(), pin)))
}

pub(super) fn validate_sdxl_imported_hires_conditioning(
    conditioning: &[Conditioning],
    hires_fix: Option<HiresFixPlan>,
) -> WorkerResult<()> {
    if !conditioning.is_empty() && hires_fix.is_some() {
        return Err(WorkerError::InvalidPayload(
            "Imported SDXL img2img/edit conditioning cannot be combined with Hires.fix".to_owned(),
        ));
    }
    Ok(())
}

pub(super) fn validate_sdxl_imported_hires_scheduler(
    engine_id: &str,
    scheduler: Option<&str>,
    hires_fix: Option<HiresFixPlan>,
) -> WorkerResult<()> {
    if hires_fix.is_some() && scheduler == Some("discrete") {
        return Err(WorkerError::InvalidPayload(format!(
            "{engine_id}: Imported SDXL Hires.fix is not supported with the \"discrete\" scheduler; select a curated scheduler instead"
        )));
    }
    Ok(())
}

pub(super) fn resolve_sdxl_imported_hires(
    request: &ImageRequest,
    first_pass_steps: u32,
    first_pass_guidance: f32,
) -> Option<HiresFixPlan> {
    if request.hires_fix.is_disabled() {
        return None;
    }
    let upscale_by = request.hires_fix.effective_upscale_by();
    let denoising_strength = request.hires_fix.effective_denoising_strength();
    let cfg = request
        .hires_fix
        .effective_cfg_scale(Some(first_pass_guidance));
    Some(HiresFixPlan {
        width: sceneworks_core::image_request::hires_fix_target_dimension(
            request.width,
            upscale_by,
        ),
        height: sceneworks_core::image_request::hires_fix_target_dimension(
            request.height,
            upscale_by,
        ),
        steps: request.hires_fix.effective_steps(first_pass_steps),
        guidance: Some(cfg.unwrap_or_default()),
        true_cfg: None,
        provider_reference_strength: denoising_strength,
    })
}

#[cfg(target_os = "macos")]
fn sdxl_imported_reference_count(conditioning: &[Conditioning]) -> u32 {
    conditioning.iter().fold(0_u32, |count, item| {
        let increment = match item {
            Conditioning::Reference { .. }
            | Conditioning::ReferenceRgba { .. }
            | Conditioning::Mask { .. } => 1,
            Conditioning::MultiReference { images } => {
                u32::try_from(images.len()).unwrap_or(u32::MAX)
            }
            _ => 0,
        };
        count.saturating_add(increment)
    })
}

fn loaded_sdxl_spec(
    spec: &LoadSpec,
    placement: crate::generator_cache::ExecutionPolicy,
) -> Option<LoadSpec> {
    if crate::generator_cache::ExecutionPolicy::from_load_spec(spec) == placement {
        return None;
    }
    let mut spec = spec.clone();
    spec.offload_policy = placement.offload_policy;
    spec.load_shape = placement.load_shape;
    spec.load_shape_declaration_result = placement.load_shape_declaration_result;
    Some(spec)
}

pub(super) struct SdxlHiresMemory {
    #[cfg(target_os = "macos")]
    output_size: (u32, u32),
    #[cfg(target_os = "macos")]
    reference_count: u32,
    workspace_bytes: u64,
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    host_bytes: u64,
}

type SdxlImportedPassOutput = (u32, u32, Vec<u8>, gen_core::generator::ExecutionIdentity);
type SdxlImportedOutput = (u32, u32, Vec<u8>, WorkerExecutionReport);

fn sdxl_imported_pass_request(
    prompt: &str,
    seed: i64,
    prototype: &GenerationRequest,
    conditioning: Vec<Conditioning>,
) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.to_owned(),
        seed: Some(seed as u64),
        conditioning,
        ..prototype.clone()
    }
}

fn render_sdxl_imported_pass(
    generator: &dyn Generator,
    mut request: GenerationRequest,
    plan: Option<&gen_core::RenderPlan>,
    memory_context: Option<&gen_core::MemoryRunContext>,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<SdxlImportedPassOutput> {
    let completed = match plan {
        Some(plan) => crate::memory_strategy::generate_with_scope_execution_report_planned(
            generator,
            &mut request,
            plan,
            memory_context,
            on_progress,
        ),
        None => crate::memory_strategy::generate_with_scope_execution_report(
            generator,
            &mut request,
            memory_context,
            on_progress,
        ),
    }
    .map_err(|error| {
        crate::generation_error("SDXL imported checkpoint generation failed", error)
    })?;
    match completed.generation.output {
        Some(GenerationOutput::Images(mut images)) if images.len() == 1 => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("SDXL imported checkpoint produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels, completed.execution))
        }
        Some(GenerationOutput::Images(images)) => Err(WorkerError::Engine(format!(
            "SDXL imported checkpoint pass returned {} images, expected one",
            images.len()
        ))),
        _ => Err(WorkerError::Engine(
            "SDXL imported checkpoint returned non-image output".to_owned(),
        )),
    }
}

pub(super) struct PreparedSdxlHires {
    requests: [GenerationRequest; 2],
    plan: gen_core::MultiPassRenderPlan,
    spec: std::sync::Arc<LoadSpec>,
    hires: HiresFixPlan,
    enhance: PromptEnhance,
}

impl PreparedSdxlHires {
    pub(super) fn prepare(
        registry: &gen_core::ProviderRegistry,
        engine_id: &str,
        spec: std::sync::Arc<LoadSpec>,
        first: GenerationRequest,
        hires: HiresFixPlan,
        enhance: PromptEnhance,
    ) -> WorkerResult<Self> {
        let mut first = first;
        // Enhancement belongs only to the final persisted refinement pass.
        first.enhance_prompt = false;
        first.enhance_temperature = None;
        first.enhance_max_tokens = None;
        first.prompt_enhancement = gen_core::PromptEnhancementSink::default();

        validate_sdxl_imported_hires_scheduler(engine_id, first.scheduler.as_deref(), Some(hires))?;

        let mut refine = GenerationRequest {
            width: hires.width,
            height: hires.height,
            steps: Some(hires.steps),
            guidance: hires.guidance,
            use_pid: false,
            conditioning: Vec::new(),
            low_res_start: None,
            source_image: None,
            strength: None,
            ..first.clone()
        };
        enhance.apply(&mut refine, gen_core::PromptEnhancementSink::default());
        let plan = super::render::compile_hires_multi_pass_plan(
            registry, engine_id, &spec, &first, &refine, &hires,
        )?
        .ok_or_else(|| {
            WorkerError::Engine(format!(
                "{engine_id}: Hires.fix route has no registered pre-load render planner"
            ))
        })?;
        Ok(Self {
            requests: [first, refine],
            plan,
            spec,
            hires,
            enhance,
        })
    }

    pub(super) fn passes(&self) -> (&gen_core::RenderPlan, &gen_core::RenderPlan) {
        (&self.plan.passes()[0], &self.plan.passes()[1])
    }

    pub(super) fn memory(&self) -> WorkerResult<SdxlHiresMemory> {
        let (first, refine) = self.passes();
        let base = first.output_size();
        let target = refine.output_size();
        let domain = if cfg!(target_os = "macos") {
            gen_core::MemoryBudgetDomain::UnifiedMemory
        } else {
            gen_core::MemoryBudgetDomain::DeviceVram
        };
        let enhance_workspace_bytes = if self.enhance.enabled {
            128 * 1024 * 1024
        } else {
            0
        };
        let pricing = crate::execution_planner::price_eng2_hires_fix_request_live_set_plan(
            "sdxl",
            domain,
            base.0,
            base.1,
            first.executed_steps() as u32,
            target.0,
            target.1,
            refine.executed_steps() as u32,
            1,
            &[],
            None,
            self.enhance.enabled,
            self.requests[0].use_pid,
            0,
            0,
            0,
            0,
            enhance_workspace_bytes,
            0,
            0,
            0,
            0,
        )?;
        Ok(SdxlHiresMemory {
            #[cfg(target_os = "macos")]
            output_size: target,
            #[cfg(target_os = "macos")]
            reference_count: refine.reference_count(),
            workspace_bytes: if domain == gen_core::MemoryBudgetDomain::UnifiedMemory {
                pricing.transition_peak_bytes
            } else {
                pricing.priced_domain_peak.request_peak_bytes
            },
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            host_bytes: crate::execution_planner::krea_hires_host_workspace_bytes(base, target),
        })
    }

    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn generate_mlx(
        self,
        generator: &dyn Generator,
        loaded_policy: crate::generator_cache::ExecutionPolicy,
        evaluation: &crate::mlx_fit_gate::MlxRequestEvaluation,
        preview: gen_core::PreviewSink,
        prompt_enhancement: gen_core::PromptEnhancementSink,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> WorkerResult<SdxlImportedOutput> {
        let (base, _) = self.passes();
        let (width, height) = base.output_size();
        let mut base_context =
            hires_first_pass_context(&evaluation.context, width, height, base.reference_count());
        base_context.use_pid = self.requests[0].use_pid;
        self.generate(
            generator,
            loaded_policy,
            Some(evaluation.memory),
            [Some(&base_context), Some(&evaluation.context)],
            preview,
            prompt_enhancement,
            cancel,
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate(
        self,
        generator: &dyn Generator,
        loaded_policy: crate::generator_cache::ExecutionPolicy,
        request_memory: Option<gen_core::GenerationMemory>,
        contexts: [Option<&gen_core::MemoryRunContext>; 2],
        preview: gen_core::PreviewSink,
        prompt_enhancement: gen_core::PromptEnhancementSink,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> WorkerResult<SdxlImportedOutput> {
        let Self {
            requests,
            plan,
            spec,
            hires,
            enhance,
        } = self;
        let plan = if let Some(loaded_spec) = loaded_sdxl_spec(&spec, loaded_policy) {
            plan.rebind_load_placement(&loaded_spec).map_err(|error| {
                crate::generation_error("SDXL Hires.fix placement changed", error)
            })?
        } else {
            plan
        };
        let first_plan = &plan.passes()[0];
        let refine_plan = &plan.passes()[1];
        let mut requests = requests;
        for request in &mut requests {
            request.memory = request_memory;
            request.preview = preview.clone();
            request.cancel = cancel.clone();
        }
        let [first, mut refine] = requests;
        let [first_context, refine_context] = contexts;
        let steps = first.steps.unwrap_or(0);
        let (base_execution, (width, height, pixels, refine_execution)) = drive_hires_passes(
            steps,
            &hires,
            Some(refine_plan),
            cancel,
            on_progress,
            |progress| {
                let (width, height, pixels, execution) = render_sdxl_imported_pass(
                    generator,
                    first,
                    Some(first_plan),
                    first_context,
                    progress,
                )?;
                Ok((
                    Image {
                        width,
                        height,
                        pixels,
                    },
                    execution,
                ))
            },
            |reference, progress| {
                let strength = refine_plan
                    .produced_reference()
                    .and_then(|reference| reference.strength);
                refine.conditioning = vec![Conditioning::Reference {
                    image: reference,
                    strength,
                }];
                enhance.apply(&mut refine, prompt_enhancement);
                render_sdxl_imported_pass(
                    generator,
                    refine,
                    Some(refine_plan),
                    refine_context,
                    progress,
                )
            },
        )?;
        let report = super::render::completed_hires_report(vec![
            ("base", WorkerExecutionReport::Provider(base_execution)),
            ("refine", WorkerExecutionReport::Provider(refine_execution)),
        ])?;
        Ok((width, height, pixels, report))
    }
}

pub(super) enum SdxlImportedItem {
    Ordinary(String),
    Hires(Box<PreparedSdxlHires>),
}

#[allow(clippy::too_many_arguments)]
fn sdxl_imported_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    seed: i64,
    prototype: &GenerationRequest,
    conditioning: &[Conditioning],
    memory_evaluation: Option<&crate::mlx_fit_gate::MlxRequestEvaluation>,
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<SdxlImportedOutput> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        seed: Some(seed as u64),
        conditioning: conditioning.to_vec(),
        memory: memory_evaluation.map(|evaluation| evaluation.memory),
        preview,
        cancel: cancel.clone(),
        ..prototype.clone()
    };
    let (width, height, pixels, execution) = render_sdxl_imported_pass(
        generator,
        request,
        None,
        memory_evaluation.map(|evaluation| &evaluation.context),
        on_progress,
    )?;
    Ok((
        width,
        height,
        pixels,
        WorkerExecutionReport::Provider(execution),
    ))
}

/// Request identity handed to MLX admission for the imported SDXL assembly. Hires admission names
/// the heavier final refinement (one generated-image reference); its first pass receives a derived
/// base-size context immediately before execution.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdxl_imported_memory_inputs(
    request: &ImageRequest,
    spec: &LoadSpec,
    conditioning: &[Conditioning],
    hires: Option<&SdxlHiresMemory>,
    adapter_count: usize,
    use_pid: bool,
    guidance: f32,
    negative_prompt: Option<&str>,
) -> crate::mlx_fit_gate::MlxRequestInputs {
    let request_reference_count = sdxl_imported_reference_count(conditioning);
    let reference_count = hires.map_or(request_reference_count, |hires| hires.reference_count);
    let (width, height) = hires.map_or((request.width, request.height), |hires| hires.output_size);
    let has_mask = conditioning
        .iter()
        .any(|item| matches!(item, Conditioning::Mask { .. }));
    let mut overlays = Vec::new();
    if request_reference_count > 0 {
        overlays.push(format!("references:{request_reference_count}"));
    }
    if has_mask {
        overlays.push("mask".to_owned());
    }
    if adapter_count > 0 {
        overlays.push(format!("adapters:{adapter_count}"));
    }
    if use_pid {
        overlays.push("pid".to_owned());
    }
    crate::mlx_fit_gate::MlxRequestInputs {
        native_batch: 1,
        width,
        height,
        count: request.count,
        mode: request.mode.clone(),
        overlay: (!overlays.is_empty()).then(|| overlays.join("+")),
        adapter_count,
        has_reference: reference_count > 0,
        reference_count,
        use_pid: if hires.is_some() { false } else { use_pid },
        has_phases: false,
        conditioning_windows: Some(crate::mlx_fit_gate::clip_windows_for_spec(
            "sdxl",
            spec,
            &request.prompt,
            &request.negative_prompt,
        )),
        negative_branch: crate::execution_planner::evaluates_negative_branch(
            negative_prompt.unwrap_or_default(),
            None,
            Some(guidance),
        ),
        prefix_kv_cache_eligible: false,
        reference_resources: None,
        pag_workspace_bytes: 0,
    }
}

/// Exact request-state driver installed into the imported-SDXL MLX cache callback.
///
/// Keeping the evaluator injectable makes the cache/admission handoff testable without loading a
/// multi-gigabyte checkpoint, while production passes [`crate::mlx_fit_gate::evaluate_request`]
/// directly. One evaluation is performed for every sequential image item. Hires remains one
/// admitted request with two planned provider passes, each with its own geometry and scope.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) fn drive_sdxl_imported_mlx_items<E>(
    generator: &dyn Generator,
    memory_plan: &crate::mlx_fit_gate::MlxRequestPlan,
    memory_inputs: &crate::mlx_fit_gate::MlxRequestInputs,
    initial_cache_state: gen_core::MemoryCacheState,
    loaded_policy: crate::generator_cache::ExecutionPolicy,
    warm_policy: crate::execution_planner::WarmPolicyProposal,
    external_committed_bytes: u64,
    work: Vec<(i64, SdxlImportedItem)>,
    prototype: GenerationRequest,
    conditioning: Vec<Conditioning>,
    tx: tokio::sync::mpsc::Sender<GenEvent>,
    cancel: CancelFlag,
    source_context: Option<&super::stream::ImportedCheckpointPublisher>,
    mut evaluate_request: E,
) -> WorkerResult<()>
where
    E: FnMut(
        &dyn Generator,
        &crate::mlx_fit_gate::MlxRequestPlan,
        &crate::mlx_fit_gate::MlxRequestInputs,
        gen_core::MemoryCacheState,
        gen_core::OffloadPolicy,
        crate::execution_planner::WarmPolicyProposal,
        u64,
    ) -> WorkerResult<crate::mlx_fit_gate::MlxRequestEvaluation>,
{
    let mut cache_state = initial_cache_state;
    let mut warm_policy = crate::execution_planner::WarmPolicyOnce::new(warm_policy);
    let facts_tx = tx.clone();
    drive_gen_items(
        tx,
        work,
        move |index, (seed, item), preview, on_progress| {
            if cancel.is_cancelled() {
                return Ok(None);
            }
            let memory_evaluation = evaluate_request(
                generator,
                memory_plan,
                memory_inputs,
                cache_state,
                loaded_policy.offload_policy,
                warm_policy.take(),
                external_committed_bytes,
            )?;
            cache_state = gen_core::MemoryCacheState::Warm;
            let _request_memory_limit = memory_evaluation
                .process_limit_bytes
                .and_then(crate::generator_cache::apply_request_gpu_memory_limit);
            let generated = match item {
                SdxlImportedItem::Hires(render) => render.generate_mlx(
                    generator,
                    loaded_policy,
                    &memory_evaluation,
                    preview,
                    gen_core::PromptEnhancementSink::default(),
                    &cancel,
                    on_progress,
                ),
                SdxlImportedItem::Ordinary(prompt) => sdxl_imported_generate_one(
                    generator,
                    &prompt,
                    seed,
                    &prototype,
                    &conditioning,
                    Some(&memory_evaluation),
                    preview,
                    &cancel,
                    on_progress,
                ),
            };
            let (out_width, out_height, pixels, report) = match generated {
                Ok(image) => image,
                // A poisoned process keeps its fatal class even when cancellation raced it.
                Err(error @ WorkerError::ProcessPoisoned(_)) => return Err(error),
                Err(_) if cancel.is_cancelled() => return Ok(None),
                Err(error) => return Err(error),
            };
            let mut facts = crate::JsonObject::new();
            report.insert_image_facts(&mut facts)?;
            let _ = facts_tx.blocking_send(GenEvent::ImageFacts { index, facts });
            if let Some(checkpoint) = source_context {
                checkpoint.publish(&facts_tx, generator);
            }
            Ok(Some((seed, out_width, out_height, pixels)))
        },
    )
}

fn pin_weights_source_file(
    source: WeightsSource,
    pins: &mut Vec<gen_core::PinnedWeightsFile>,
    label: &str,
) -> WorkerResult<WeightsSource> {
    match source {
        WeightsSource::File(path) => {
            let pin = gen_core::PinnedWeightsFile::pin(&path)
                .map_err(|error| crate::classify_engine_error(label, error))?;
            let loader_path = pin.loader_path().to_path_buf();
            pins.push(pin);
            Ok(WeightsSource::File(loader_path))
        }
        dir @ WeightsSource::Dir(_) => Ok(dir),
    }
}

pub(super) fn prepare_sdxl_imported_load_spec(
    checkpoint_pin: gen_core::PinnedWeightsFile,
    prepared_adapters: PreparedAdapters,
    quant: Option<gen_core::Quant>,
    pid_weights: Option<gen_core::PidWeights>,
) -> WorkerResult<LoadSpec> {
    let mut spec = LoadSpec::new(WeightsSource::File(
        checkpoint_pin.loader_path().to_path_buf(),
    ));
    if !prepared_adapters.specs.is_empty() {
        spec = spec.with_adapters(prepared_adapters.specs);
    }
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    let mut extra_pins = Vec::new();
    if let Some(pid) = pid_weights {
        let checkpoint = pin_weights_source_file(
            pid.checkpoint,
            &mut extra_pins,
            "SDXL imported PiD source preparation failed",
        )?;
        let gemma = pin_weights_source_file(
            pid.gemma,
            &mut extra_pins,
            "SDXL imported PiD source preparation failed",
        )?;
        spec = spec.with_pid(checkpoint, gemma);
    }
    crate::paths::prepare_load_spec_with_file_pins(
        &mut spec,
        std::iter::once(checkpoint_pin)
            .chain(prepared_adapters.pins)
            .chain(extra_pins),
        "SDXL imported source preparation failed",
    )?;
    Ok(spec)
}

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn sdxl_imported_candle_floor_companion_dirs(spec: &LoadSpec) -> Vec<&Path> {
    let mut companions: Vec<&Path> = spec
        .components
        .values()
        .filter_map(|source| match source {
            WeightsSource::Dir(dir) => Some(dir.as_path()),
            WeightsSource::File(_) => None,
        })
        .collect();
    if let Some(pid) = &spec.pid {
        if let WeightsSource::Dir(dir) = &pid.checkpoint {
            companions.push(dir.as_path());
        }
        if let WeightsSource::Dir(dir) = &pid.gemma {
            companions.push(dir.as_path());
        }
    }
    companions
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn prepare_sdxl_imported_candle_floor(
    model: &str,
    settings: &Settings,
    spec: &LoadSpec,
    hires_workspace_bytes: u64,
) -> WorkerResult<super::base_admission::CachedCandleBaseFloorAdmission> {
    let companions = sdxl_imported_candle_floor_companion_dirs(spec);
    Ok(super::base_admission::prepare_cached_candle_base_floor(
        model,
        "SDXL imported",
        settings,
        spec,
        &companions,
    )?
    .with_hires_workspace_bytes(hires_workspace_bytes))
}

pub(super) async fn generate_sdxl_imported_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    dispatch: PreparedFileDispatch<'_, PreparedSdxlImportedSources>,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let PreparedFileDispatch { plan, sources } = dispatch;
    let request = &plan.request;
    let PreparedSdxlImportedSources {
        checkpoint: file_pin,
        adapters: prepared_adapters,
    } = sources;
    let checkpoint =
        super::stream::ImportedCheckpointPublisher::new(&request.model_manifest_entry, &file_pin);
    let adapter_count = prepared_adapters.specs.len();
    let steps =
        resolved_option_or_manifest_u32(request, "steps", SDXL_IMPORTED_DEFAULT_STEPS, 1..=100);
    let guidance = resolved_option_or_manifest_f32(
        request,
        "guidanceScale",
        SDXL_IMPORTED_DEFAULT_GUIDANCE,
        0.0..=30.0,
    );
    let (sampler, scheduler, scheduler_shift) = read_resolved_image_sampling_knobs(request);
    let operation = if request.mode == "edit_image" {
        gen_core::ImportedModelOperation::Edit
    } else {
        gen_core::ImportedModelOperation::Generate
    };
    let descriptor = crate::inference_runtime::imported_model_descriptor(
        "sdxl",
        gen_core::ImportedModelSource::FusedCheckpoint,
        operation,
    )
    .ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "The active backend has no registered imported SDXL {operation:?} provider"
        ))
    })?;
    let caps = &descriptor.capabilities;
    let sampler = normalize_sampling_knob(
        sampler,
        &caps.samplers,
        "sampler",
        &request.model,
        &job.id,
        backend,
    );
    let scheduler = normalize_sampling_knob(
        scheduler,
        &caps.schedulers,
        "scheduler",
        &request.model,
        &job.id,
        backend,
    );
    let guidance_method = normalize_sampling_knob(
        read_resolved_image_guidance_method(request),
        &caps.supported_guidance_methods,
        "guidanceMethod",
        &request.model,
        &job.id,
        backend,
    );
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, "sdxl")?;
    let use_pid = pid_weights.is_some();
    let negative_prompt =
        (!request.negative_prompt.trim().is_empty()).then(|| request.negative_prompt.clone());
    let (width, height) = (request.width, request.height);
    let (conditioning, mode_tag, strength) =
        resolve_imported_sdxl_conditioning(request, settings, project_path)?;
    let hires_fix = resolve_sdxl_imported_hires(request, steps, guidance);
    validate_sdxl_imported_hires_conditioning(&conditioning, hires_fix)?;
    validate_sdxl_imported_hires_scheduler(descriptor.id, scheduler.as_deref(), hires_fix)?;
    let (quant, quant_bits) = imported_model_quant(request, &descriptor, "Imported SDXL")?;
    let enhance = PromptEnhance::default();
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let mut raw_settings = request.effective_advanced_wire();
    raw_settings.insert("realModelInference".to_owned(), Value::Bool(true));
    raw_settings.insert("numInferenceSteps".to_owned(), json!(steps));
    raw_settings.insert("guidanceScale".to_owned(), json!(guidance));
    raw_settings.insert("engine".to_owned(), Value::String(descriptor.id.to_owned()));
    raw_settings.insert(
        "importedCheckpoint".to_owned(),
        Value::String(request.model.clone()),
    );
    raw_settings.insert("mode".to_owned(), Value::String(mode_tag.to_owned()));
    raw_settings.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map_or(Value::Null, Value::from),
    );
    if let Some(strength) = strength {
        raw_settings.insert("strength".to_owned(), json!(strength));
    }
    if request.hires_fix.enabled {
        raw_settings.insert(
            "hiresFix".to_owned(),
            serde_json::to_value(&request.hires_fix).expect("HiresFixRequest is serializable"),
        );
    }

    let spec = prepare_sdxl_imported_load_spec(file_pin, prepared_adapters, quant, pid_weights)?;
    #[cfg(target_os = "macos")]
    let spec = spec.with_component(
        "ldm_tokenizer",
        WeightsSource::Dir(stage_sdxl_tokenizer(api, settings, job).await?),
    );
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let spec = attach_imported_sdxl_components(spec, &descriptor, api, settings, job).await?;

    let prototype = GenerationRequest {
        negative_prompt,
        width,
        height,
        count: 1,
        steps: Some(steps),
        guidance: Some(guidance),
        sampler,
        scheduler,
        scheduler_shift,
        guidance_method,
        use_pid,
        ..Default::default()
    };
    let work: Vec<(i64, SdxlImportedItem)> = if let Some(hires) = hires_fix {
        let shared_spec = std::sync::Arc::new(spec.clone());
        work.into_iter()
            .map(|(seed, prompt)| {
                let first = sdxl_imported_pass_request(&prompt, seed, &prototype, Vec::new());
                let render = PreparedSdxlHires::prepare(
                    crate::inference_runtime::media(),
                    descriptor.id,
                    shared_spec.clone(),
                    first,
                    hires,
                    enhance.clone(),
                )?;
                Ok((seed, SdxlImportedItem::Hires(Box::new(render))))
            })
            .collect::<WorkerResult<_>>()?
    } else {
        work.into_iter()
            .map(|(seed, prompt)| (seed, SdxlImportedItem::Ordinary(prompt)))
            .collect()
    };
    let first_hires = work.iter().find_map(|(_, item)| match item {
        SdxlImportedItem::Hires(render) => Some(render),
        SdxlImportedItem::Ordinary(_) => None,
    });
    let hires_memory = first_hires.map(|render| render.memory()).transpose()?;

    #[cfg(target_os = "macos")]
    let memory_plan = crate::mlx_fit_gate::MlxRequestPlan::try_for_spec_and_manifest(
        descriptor.id,
        &request.model,
        &spec,
        Some(&request.model_manifest_entry),
        None,
    )?
    .with_request_workspace(
        hires_memory
            .as_ref()
            .map_or(0, |hires| hires.workspace_bytes),
    );
    #[cfg(target_os = "macos")]
    let memory_inputs = sdxl_imported_memory_inputs(
        request,
        &spec,
        &conditioning,
        hires_memory.as_ref(),
        adapter_count,
        use_pid,
        guidance,
        prototype.negative_prompt.as_deref(),
    );
    #[cfg(target_os = "macos")]
    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state(
        job.id.clone(),
        descriptor.id,
        adapter_count,
        spec,
        "SDXL imported checkpoint load failed".to_owned(),
        move |model,
              initial_cache_state,
              loaded_policy,
              warm_policy,
              external_committed_bytes,
              tx,
              cancel| {
            drive_sdxl_imported_mlx_items(
                model,
                &memory_plan,
                &memory_inputs,
                initial_cache_state,
                loaded_policy,
                warm_policy,
                external_committed_bytes,
                work,
                prototype,
                conditioning,
                tx,
                cancel,
                Some(&checkpoint),
                crate::mlx_fit_gate::evaluate_request,
            )
        },
    );

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let hires_workspace_bytes = if let Some(hires) = &hires_memory {
        if let Some(available) = crate::yue2_admission::host_available_bytes()
            .await
            .filter(|&available| available < hires.host_bytes)
        {
            return Err(WorkerError::InvalidPayload(format!(
                "SDXL Hires.fix needs {} host bytes to resize its reference, but only {available} are available",
                hires.host_bytes
            )));
        }
        hires.workspace_bytes
    } else {
        0
    };
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let cold_admission =
        prepare_sdxl_imported_candle_floor(&request.model, settings, &spec, hires_workspace_bytes)?;
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let incoming_reclaimable_weight_bytes = cold_admission.reclaimable_weight_bytes();
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let workspace_runtime = tokio::runtime::Handle::current();
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let gpu_id = settings.gpu_id.clone();
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let (cancel, rx, blocking) = start_cached_gen_stream_after_cold_admission_with_placement(
        job.id.clone(),
        descriptor.id,
        adapter_count,
        spec,
        "SDXL imported checkpoint load failed".to_owned(),
        ColdLoadAdmission::new(
            incoming_reclaimable_weight_bytes,
            move |resident_reclaimable_weight_bytes| {
                cold_admission.admit(resident_reclaimable_weight_bytes)
            },
        ),
        move |model, cache_state, loaded_policy, tx, cancel| {
            if !cancel.is_cancelled()
                && hires_workspace_bytes > 0
                && cache_state == gen_core::MemoryCacheState::Warm
            {
                workspace_runtime.block_on(
                    super::krea_multiphase::admit_warm_candle_krea_workspace(
                        "SDXL Hires.fix",
                        &gpu_id,
                        hires_workspace_bytes,
                    ),
                )?;
            }
            let facts_tx = tx.clone();
            drive_gen_items(
                tx,
                work,
                move |index, (seed, item), preview, on_progress| {
                    if cancel.is_cancelled() {
                        return Ok(None);
                    }
                    let generated = match item {
                        SdxlImportedItem::Hires(render) => render.generate(
                            model,
                            loaded_policy,
                            None,
                            [None, None],
                            preview,
                            gen_core::PromptEnhancementSink::default(),
                            &cancel,
                            on_progress,
                        ),
                        SdxlImportedItem::Ordinary(prompt) => sdxl_imported_generate_one(
                            model,
                            &prompt,
                            seed,
                            &prototype,
                            &conditioning,
                            None,
                            preview,
                            &cancel,
                            on_progress,
                        ),
                    };
                    let (out_width, out_height, pixels, report) = match generated {
                        Ok(image) => image,
                        // A poisoned process keeps its fatal class even when cancellation raced it.
                        Err(error @ WorkerError::ProcessPoisoned(_)) => return Err(error),
                        Err(_) if cancel.is_cancelled() => return Ok(None),
                        Err(error) => return Err(error),
                    };
                    let mut facts = crate::JsonObject::new();
                    report.insert_image_facts(&mut facts)?;
                    let _ = facts_tx.blocking_send(GenEvent::ImageFacts { index, facts });
                    checkpoint.publish(&facts_tx, model);
                    Ok(Some((seed, out_width, out_height, pixels)))
                },
            )
        },
    );

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        SDXL_IMPORTED_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
