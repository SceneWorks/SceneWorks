#[cfg(target_os = "macos")]
use super::artifact_resolution::is_flux_model;
#[cfg(target_os = "macos")]
use super::artifact_resolution::resolve_flux_ip_adapter_dir;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::resolve_weights_dir;
#[cfg(target_os = "macos")]
use super::artifact_resolution::FLUX_IP_SCALE;
#[cfg(target_os = "macos")]
use super::artifact_resolution::FLUX_IP_TRUE_CFG;
use super::geometry::fit_engine_image;
use super::instantid::ensure_face_stack_dir;
#[cfg(target_os = "macos")]
use super::kolors::resolve_kolors_edit_init;
#[cfg(target_os = "macos")]
use super::kolors::resolve_kolors_ip_adapter_dir;
#[cfg(target_os = "macos")]
use super::kolors::resolve_kolors_ip_reference;
#[cfg(not(target_os = "macos"))]
use super::resolve_seed;
use super::split_alpha;
#[cfg(target_os = "macos")]
use super::zimage::CHARACTER_ANGLE_SET_ORDER;
use super::Conditioning;
use super::Image;
use super::ImageRequest;
use crate::advanced;
use crate::ApiClient;
use crate::JobSnapshot;
use crate::Path;
use crate::PathBuf;
use crate::ProjectStore;
use crate::Settings;
use crate::Value;
use crate::WorkerError;
use crate::WorkerResult;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_conditioned_angle_set(request: &ImageRequest) -> bool {
    request.mode == "character_image"
        && advanced::flag(&request.advanced, "angleSet")
        && pose_entries(request).is_empty()
}

/// Seeds/prompts for a Candle reference-edit lane. Angle sets use the canonical 11 prompts with one
/// shared seed; every other request preserves the ordinary count/per-image seed contract.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn candle_conditioned_edit_work(request: &ImageRequest) -> Vec<(i64, String)> {
    if candle_conditioned_angle_set(request) {
        let seed = resolve_seed(request, 0);
        return sceneworks_core::angle_kps::BUILTIN_ANGLE_SET_ORDER
            .iter()
            .map(|angle| (seed, augment_prompt_for_angle(&request.prompt, angle)))
            .collect();
    }
    (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect()
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn sensenova_edit_candle_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !request.reference_asset_ids.is_empty() {
        return request
            .reference_asset_ids
            .iter()
            .take(5)
            .cloned()
            .collect();
    }
    if let Some(id) = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        return vec![id.to_owned()];
    }
    if request.mode == "edit_image" {
        if let Some(id) = request
            .source_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
        {
            return vec![id.to_owned()];
        }
    }
    Vec::new()
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn resolve_sensenova_candle_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Vec<Image>> {
    let ids = sensenova_edit_candle_reference_ids(request);
    if ids.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "SenseNova edit requires at least one reference image".to_owned(),
        ));
    }
    ids.iter()
        .map(|id| {
            let image =
                load_reference_image(&settings.data_dir, &request.project_id, id, project_path)?;
            if request.fit_mode == "stretch" {
                Ok(image)
            } else {
                fit_engine_image(image, request.width, request.height, &request.fit_mode)
            }
        })
        .collect()
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn resolve_sensenova_candle_true_cfg(request: &ImageRequest) -> f32 {
    let default = if request.mode == "character_image" {
        1.5
    } else {
        1.0
    };
    request
        .advanced
        .get("trueCfgScale")
        .or_else(|| request.advanced.get("imageGuidanceScale"))
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(default)
        .clamp(1.0, 10.0)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn resolve_candle_kolors_edit_init(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    let Some(source_id) = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Ok(None);
    };
    let source = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        source_id,
        project_path,
    )?;
    let image = if request.fit_mode == "stretch" {
        source
    } else {
        fit_engine_image(source, request.width, request.height, &request.fit_mode)?
    };
    Ok(Some((
        image,
        advanced::f32_clamped(&request.advanced, "strength", 0.6, 0.05..=1.0),
    )))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn is_sensenova_candle_model(model: &str) -> bool {
    sceneworks_core::mlx_tier_completeness::SENSENOVA_MODELS.contains(&model)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn sensenova_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_sensenova_candle_model(&request.model)
        && matches!(request.mode.as_str(), "edit_image" | "character_image")
        && !sensenova_edit_candle_reference_ids(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn kolors_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "kolors"
        && request.mode == "edit_image"
        && request
            .source_asset_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty())
        && request.reference_asset_id.is_none()
        && request.reference_asset_ids.is_empty()
        && request.mask_asset_id.is_none()
        && pose_entries(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// How a native edit job batches its iterations (sc-8946 (F-144): renamed from `Flux2Grouping` and
/// moved here from flux2.rs — it is the SHARED grouping for the FLUX.2 / Qwen-Edit / SenseNova-U1 edit
/// lanes, not FLUX.2-specific, so a reader auditing Qwen/SenseNova grouping finds it in base.rs).
#[cfg(target_os = "macos")]
pub(super) enum EditGrouping {
    /// `count` independent images (per-image seeds), the plain reference/edit path.
    Plain,
    /// The 11-angle Character-Studio set: shared seed, per-angle prompt augment.
    Angles,
    /// The best-effort pose tier: `n` poses, shared seed, `[skeleton, reference]` sets.
    Poses(usize),
}

/// Decide the grouping for a native edit job (parity with the `Mlx*Adapter` decision: pose set >
/// angle set > plain, all gated to `character_image` mode — an `edit_image` job is never grouped).
/// The caller only reaches this with a reference present, so `is_character_image` reduces to the mode
/// check. Shared by the FLUX.2 / Qwen-Edit / SenseNova-U1 edit lanes (sc-8946 moved it from flux2.rs).
#[cfg(target_os = "macos")]
pub(super) fn edit_grouping(request: &ImageRequest) -> EditGrouping {
    if request.mode != "character_image" {
        return EditGrouping::Plain;
    }
    let poses = pose_entries(request).len();
    if poses > 0 {
        return EditGrouping::Poses(poses);
    }
    if advanced::flag(&request.advanced, "angleSet") {
        return EditGrouping::Angles;
    }
    EditGrouping::Plain
}

/// Upper bound on reference images for a multi-reference edit (sc-6211). Even with the engine's
/// sequence-gated activation chunking (sc-6266), the FLUX.2-dev edit stays activation-bound: 4
/// references at 1024² peak ~93 GB and 5 would exceed the 96 GB floor (measured). The per-machine
/// FLUX.2's provider safety contract rejects over-budget combinations with an actionable message;
/// this caps absurd inputs (and bounds the DiT sequence) before that.
#[cfg(target_os = "macos")]
pub(super) const MAX_EDIT_REFERENCES: usize = 4;

/// Reference asset ids for a native edit (sc-8946 moved it from flux2.rs — shared by the FLUX.2 /
/// SenseNova-via-grouping lanes, not FLUX.2-specific). The FLUX.2-dev multi-image picker (sc-6211)
/// sends the plural `referenceAssetIds` — take all of them in order, capped at [`MAX_EDIT_REFERENCES`].
/// With no plural list it falls back to the single-reference flows: the character `referenceAssetId`,
/// else the Image-Edit `sourceAssetId` (edit_image mode). Mirrors the Python
/// `ref_id = referenceAssetId or (sourceAssetId if edit_image)`, plus the new multi-reference set.
#[cfg(target_os = "macos")]
pub(super) fn edit_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !request.reference_asset_ids.is_empty() {
        // Parsed list is already trimmed + non-empty (sceneworks-core `string_list`).
        return request
            .reference_asset_ids
            .iter()
            .take(MAX_EDIT_REFERENCES)
            .cloned()
            .collect();
    }
    if let Some(id) = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        return vec![id.to_owned()];
    }
    if request.mode == "edit_image" {
        if let Some(id) = request
            .source_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
        {
            return vec![id.to_owned()];
        }
    }
    Vec::new()
}

#[cfg(target_os = "macos")]
pub(super) fn grouped_edit_image_count(request: &ImageRequest) -> u32 {
    match edit_grouping(request) {
        EditGrouping::Angles => CHARACTER_ANGLE_SET_ORDER.len() as u32,
        EditGrouping::Poses(count) => count as u32,
        EditGrouping::Plain => request.count,
    }
}

/// Resolve a reference/source asset id to an in-memory RGB8 image (the engine VAE-encodes + resizes
/// it). Uses the indexed `ProjectStore::get_asset` → `file.path`. Shared by the MLX image/video
/// conditioning paths and the candle video i2v conditioning (sc-5175), so it lives here (both lanes)
/// rather than in a macOS-only include.
#[cfg(native_lane)]
pub(crate) fn load_reference_image(
    data_dir: &Path,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> WorkerResult<Image> {
    // Every caller that existed before sc-24113 keeps upstream-parity truncation, byte-for-byte.
    load_reference_image_with(
        data_dir,
        project_id,
        asset_id,
        project_path,
        FlattenPolicy::Truncate,
    )
}

/// How a reference that CARRIES an alpha channel is reduced to the 3-channel `gen_core::Image`
/// every RGB-only engine takes (sc-24113).
///
/// A PARITY knob, not a quality one, and both answers are correct — for different upstream
/// pipelines:
///
/// * [`Truncate`](FlattenPolicy::Truncate) drops the fourth byte, which is what
///   `DynamicImage::to_rgb8()` does and what upstream `PIL.Image.convert("RGB")` does. Every edit
///   model SceneWorks shipped before 2.1 — SDXL inpaint, FLUX.2 edit, Kolors IP-adapter,
///   `qwen_image_edit_2511` — is compared against a reference implementation that truncates, so
///   this is the only answer that keeps them at parity, and it is the default for that reason.
/// * [`OverWhite`](FlattenPolicy::OverWhite) composites straight (un-premultiplied) alpha over an
///   opaque white backdrop, matching the S4 contract's `RgbaImage::to_rgb_over_white()`.
///   Qwen-Image 2.1 feeds its VISION tower the composited copy, so its reference path wants this.
///
/// ⚠️ The two are IDENTICAL on an opaque image (`A=255`) and differ MAXIMALLY on a transparent one,
/// because 2.1's alpha is straight: `A=0` does NOT zero RGB, so a transparent pixel keeps whatever
/// colour it was authored with. A blanket change of this default would therefore move every
/// existing model silently rather than loudly — which is exactly what the first cut of sc-24113 did.
///
/// Qwen-Image 2.1 does not flatten here at all: an alpha-carrying 2.1 reference travels UN-FLATTENED
/// as `Conditioning::ReferenceRgba` (its RGB via `Truncate`, which is the straight colour, plus its
/// own alpha plane), and the engine composites the vision tower's copy itself.
#[cfg(native_lane)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum FlattenPolicy {
    /// Drop the alpha byte — upstream `convert("RGB")` semantics. Every pre-2.1 lane.
    #[default]
    Truncate,
    /// Composite straight alpha over opaque white — the S4 `to_rgb_over_white()` semantics.
    ///
    /// Constructed only by tests: the one lane whose engine consumes a white composite
    /// (`qwen_image_2_1`'s vision tower) builds that copy engine-side from the un-flattened
    /// `Conditioning::ReferenceRgba`, so no production caller flattens over white. Kept so the
    /// parity decision stays stated and tested beside the default it contrasts with.
    #[allow(dead_code)]
    OverWhite,
}

/// [`load_reference_image`] with an explicit [`FlattenPolicy`].
#[cfg(native_lane)]
pub(crate) fn load_reference_image_with(
    data_dir: &Path,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
    flatten: FlattenPolicy,
) -> WorkerResult<Image> {
    let asset = ProjectStore::new(data_dir.to_path_buf(), "worker")
        .get_asset(project_id, asset_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("reference asset {asset_id}: {error}"))
        })?;
    let rel = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("reference asset {asset_id} has no media path"))
        })?;
    // The asset's file.path comes from an on-disk sidecar the user can edit, so
    // route it through safe_project_path (rejects `..`/absolute components) rather
    // than a bare join — matching the media-jobs reads and keeping a poisoned
    // sidecar from reading an arbitrary file as the reference (sc-4278 / F-MLXW-14).
    let path = crate::safe_project_path(project_path, rel)?;
    let decoded = crate::image_decode::decode_image_any(&path).map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "reference asset {asset_id}: could not read image {}: {error}",
            path.display()
        ))
    })?;
    // sc-24113: how an alpha-carrying reference is flattened is the CALLER's choice, and the
    // default is upstream-parity truncation. See [`FlattenPolicy`] for why both answers are
    // correct and why the default must not move: `to_rgb8()` drops the fourth byte exactly as
    // `PIL.Image.convert("RGB")` does, which is what every pre-2.1 edit model is compared against.
    //
    // `OverWhite` composites straight alpha over an opaque white backdrop, matching the S4
    // contract's `RgbaImage::to_rgb_over_white()`. It matters only for an image that actually
    // carries alpha, and only where the transparency is real: `A=255` makes the two identical, and
    // `A=0` makes them maximally different because straight alpha leaves the hidden RGB intact.
    //
    // A non-alpha source takes `to_rgb8()` under either policy — the same call, the same bytes —
    // so nothing here can perturb an ordinary opaque reference.
    let rgb = match (flatten, decoded.color().has_alpha()) {
        (FlattenPolicy::OverWhite, true) => {
            let rgba = decoded.to_rgba8();
            let (width, height) = (rgba.width(), rgba.height());
            image::RgbImage::from_fn(width, height, |x, y| {
                let [r, g, b, a] = rgba.get_pixel(x, y).0;
                let over_white = |channel: u8| {
                    // Straight alpha over an opaque white backdrop, rounded half-up:
                    //   out = channel * a/255 + 255 * (1 - a/255)
                    let blended = channel as u32 * a as u32 + 255 * (255 - a as u32);
                    ((blended + 127) / 255) as u8
                };
                image::Rgb([over_white(r), over_white(g), over_white(b)])
            })
        }
        _ => decoded.to_rgb8(),
    };
    Ok(Image {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    })
}

/// The alpha plane of the same asset [`load_reference_image`] loads, or `None` when it has none
/// (sc-24111).
///
/// A deliberate second read rather than a widening of `load_reference_image`. That function's
/// return type is `gen_core::Image`, whose `pixels` is a flat 3-channel buffer — the engine-side
/// contract, which the inference half of this story owns and which this PR does not touch. The
/// product-side lanes that refine or rescale an image and then write it back as an asset still
/// have to preserve the channel, so the plane travels beside the engine image instead of inside
/// it, and is re-attached after the pass by `image_jobs::reattach_alpha`.
///
/// Reads the same path through the same `safe_project_path` confinement, so a poisoned sidecar
/// cannot reach a different file here than it does there.
#[cfg(native_lane)]
pub(crate) fn load_reference_alpha(
    data_dir: &Path,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> WorkerResult<Option<image::GrayImage>> {
    let asset = ProjectStore::new(data_dir.to_path_buf(), "worker")
        .get_asset(project_id, asset_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("reference asset {asset_id}: {error}"))
        })?;
    let rel = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("reference asset {asset_id} has no media path"))
        })?;
    let path = crate::safe_project_path(project_path, rel)?;
    let decoded = crate::image_decode::decode_image_any(&path).map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "reference asset {asset_id}: could not read image {}: {error}",
            path.display()
        ))
    })?;
    Ok(split_alpha(&decoded).1)
}

/// The clamped identity img2img-init strength for a strict-pose set, or `None` for the pose-only tier.
/// `Some(strength)` iff `advanced.referenceStrength > 0` AND a non-empty `referenceAssetId` is present;
/// `strength` is the user value clamped to `[0.05, 1.0]`, carrying the mflux `image_strength`
/// convention **verbatim** (higher strength → later denoise start → output stays closer to the init).
///
/// sc-8946 (F-144): this was duplicated line-for-line as `zimage_identity_strength` (Z-Image, sc-3146)
/// and `flux2_identity_strength` (FLUX.2-dev control) — an identity-gate change had to be made twice.
/// The gate + clamp are IDENTICAL across the two lanes (both mirror `MlxZImageAdapter`), so it lives
/// here once. Pure (request only), so the parity-sensitive gate + clamp stay unit-testable without I/O.
#[cfg(target_os = "macos")]
pub(super) fn identity_strength(request: &ImageRequest) -> Option<f32> {
    let strength = request
        .advanced
        .get("referenceStrength")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .filter(|strength| *strength > 0.0)?;
    let has_asset = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .is_some_and(|id| !id.is_empty());
    has_asset.then(|| (strength as f32).clamp(0.05, 1.0))
}

/// Resolve the optional identity img2img-init for a strict-pose set: `Some((image, strength))` when
/// [`identity_strength`] engages (decoding `referenceAssetId` via [`load_reference_image`]), else
/// `None` (the default pose-only tier). The reference is shared across the whole pose set — identity is
/// constant; only the per-pose skeleton changes.
///
/// sc-8946 (F-144): the shared body of the former `resolve_identity_init` /
/// `resolve_flux2_identity_init` (both line-for-line copies). The Z-Image strict-pose stream and the
/// FLUX.2-dev control stream both call this.
#[cfg(target_os = "macos")]
pub(super) fn resolve_identity_init(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    let Some(strength) = identity_strength(request) else {
        return Ok(None);
    };
    let asset_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .expect("identity_strength guarantees a non-empty referenceAssetId");
    let image = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    Ok(Some((image, strength)))
}

/// Resolve the **Krea 2 Turbo img2img** init (epic 8588 slice A, sc-8591): `Some((reference, strength))`
/// from a `referenceAssetId` + `advanced.strength`, or `None` when no reference asset is supplied (the
/// lane then falls back to plain txt2img). `strength` is the full-range 0.0–1.0 reference-fidelity
/// slider (default 0.5); the worker does NOT clamp beyond `[0, 1]` — the usable band is model-specific
/// (A0/sc-8589 mapped Krea Turbo's sweet spot at ~0.35–0.65, but that is guidance, not a hard clamp).
/// The single `Conditioning::Reference` this produces is routed by the engine to
/// `generate_turbo_img2img` (sc-10135), whose `preprocess_init_image` LANCZOS-resizes the reference to
/// the output W×H — so, like Z-Image's [`resolve_identity_init`], the reference is fed raw (the
/// `edit_image`-only [`should_fit_edit_source`] crop/pad-fit never applies to Krea Turbo's
/// reference-guided img2img surface).
///
/// Available to the candle lane too (sc-10134): the candle `generate_candle_stream` calls this to
/// resolve the img2img init off-Mac for every model the candle router admits on a generic img2img
/// lane, feeding the same `(image, strength)` into `generate_one`'s `reference` →
/// `Conditioning::Reference` → the engine's img2img entrypoint. That arm is model-agnostic; the
/// authoritative list of the ids the router actually wires lives beside the `generate_candle_stream`
/// call site, and this resolver serves all of them uniformly through [`model_supports_img2img`].
///
/// NOT every `ui.img2img` model, despite the flag being the gate here: FLUX.2 Klein
/// (`flux2_klein_9b` / `_kv` / `_true_v2`) also declares `ui.img2img`, but
/// `candle_reserves_reference_only_modes` hands its whole reference surface to the bespoke
/// `Flux2Edit` route, and `generate_candle_stream` fails closed on a Klein reference-bearing mode
/// before reaching this resolver — so Klein never takes the generic init.
#[cfg(native_lane)]
pub(super) fn resolve_img2img_init_generic(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    let Some(asset_id) = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Ok(None);
    };
    let image = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    let strength = advanced::f32_clamped(&request.advanced, "strength", 0.5, 0.0..=1.0);
    Ok(Some((image, strength)))
}

/// Whether the model opts into plain-t2i img2img (reference-guided latent-init) via the catalog —
/// the SAME `ui.img2img` manifest flag the web reads to show the "Image reference" tile (epic 8588
/// A4, sc-10195/sc-10189). Manifest-flag-driven rather than an ever-growing model-string match, so a
/// new text-only model gains img2img by flipping its manifest flag + landing its mlx-gen entrypoint —
/// no worker change. Mirrors the existing `uses_standard_tier_layout`/`is_dense_te_tier` pattern.
///
/// This gates the GENERIC img2img arm in [`resolve_generic_lane_conditioning`], which sits AFTER the
/// model-specific reference arms (z-image identity-init, FLUX IP-Adapter, Kolors, Ideogram edit) so
/// those bespoke surfaces keep precedence; the generic arm then catches Krea + SD3.5 + any future
/// `ui.img2img` model uniformly.
///
/// Available to the candle lane too (sc-10134): `generate_candle_stream` gates its GENERIC img2img
/// resolve on this same manifest flag off-Mac, each engine owning its img2img entrypoint. The flag is
/// necessary but not sufficient there — a model reaches that arm only if the candle router also wires
/// it an img2img lane (the wired id list is stated at the `generate_candle_stream` call site). FLUX.2
/// Klein declares `ui.img2img` yet has no such lane: its reference surface is reserved for the bespoke
/// `Flux2Edit` route, so it never reaches [`resolve_img2img_init_generic`] off-Mac.
#[cfg(native_lane)]
pub(super) fn model_supports_img2img(request: &ImageRequest) -> bool {
    request
        .model_manifest_entry
        .get("ui")
        .and_then(|ui| ui.get("img2img"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Pure worker-side gate shared by the MLX and Candle generic conditioning resolvers. Scheduler
/// routing rejects unsupported plural/edit/control shapes before dispatch; this keeps the worker
/// defensive about the fields it consumes so neither backend silently treats an edit or a missing
/// reference as img2img.
#[cfg(native_lane)]
pub(super) fn uses_generic_img2img(request: &ImageRequest, has_reference: bool) -> bool {
    model_supports_img2img(request) && has_reference && request.mode != "edit_image"
}

/// Resolve the Krea "text style" tap-reweight gain (sc-11878; gate fixed in sc-12008). Set ONLY when
/// the model's manifest declares the `ui.textStyleGain` slider — Krea/Qwen-Image-family only, so every
/// other family self-gates to `None`. The user value comes from `advanced.textStyleGain`, clamped to
/// the GPU-validated `[0.25, 1.75]`; the manifest slider object is not a scalar, so the `1.0` default
/// (a byte-exact engine no-op) applies when the user leaves it at default.
///
/// NOTE (sc-12008): `model_manifest_entry` is the FULL model entry (`resolve_model_manifest_entry`),
/// so the slider lives at `.ui.textStyleGain`, NOT the top level. Reading `.get("textStyleGain")`
/// directly is always `None` and silently disables the feature — the original end-to-end bug. This is
/// the single seam for both the MLX (`generate_stream`) and candle (`generate_candle_stream`) lanes so
/// the two can't drift apart again.
#[cfg(native_lane)]
pub(super) fn resolve_text_style_gain(request: &ImageRequest) -> Option<f32> {
    request
        .model_manifest_entry
        .get("ui")
        .and_then(|ui| ui.get("textStyleGain"))
        .is_some()
        .then(|| advanced::f32_clamped(&request.advanced, "textStyleGain", 1.0, 0.25..=1.75))
}

/// Whether a Z-Image t2i request should take the generic `ui.img2img` reference-guided latent-init
/// rather than the Character Studio identity-init (sc-3619). Z-Image already owns a bespoke reference
/// arm in [`resolve_generic_lane_conditioning`] (keyed on `referenceStrength`), so it never reaches the
/// generic img2img arm below — this predicate re-introduces the generic path INSIDE that arm.
///
/// Identity-init keeps precedence (it also drives face-likeness scoring), so the generic img2img fires
/// ONLY when identity-init doesn't engage (no `referenceStrength`) yet the model opts into `ui.img2img`
/// and a reference is present — i.e. the Image Studio "Image reference" tile (`referenceAssetId` +
/// `advanced.strength`). The two surfaces are mutually exclusive by mode (character_image vs
/// text_to_image); encoding the precedence purely keeps it unit-testable without image I/O (epic 8588
/// A4.5, sc-10193). Base `z_image` is NOT in the z-image arm, so it reaches the generic arm directly and
/// never consults this predicate.
#[cfg(target_os = "macos")]
pub(super) fn zimage_uses_generic_img2img(request: &ImageRequest, has_reference: bool) -> bool {
    identity_strength(request).is_none() && has_reference && model_supports_img2img(request)
}

/// The source identity reference (decoded image + its asset id) a strict-control pose set scores its
/// finished poses against (epic 4406, sc-4410), or `None` when the job carries no identity reference.
///
/// A strict-control pose-library job locks the body pose via ControlNet but optionally carries a
/// character identity `referenceAssetId` (the same one the opt-in img2img init uses). When present, the
/// pose set is part of Character Studio and each finished pose is scored against that source identity
/// face through the shared [`crate::face_likeness`] seam — independent of whether the img2img init is
/// engaged (scoring observes the FINAL pose; the init only seeds it). `None` (a bare pose set with no
/// identity reference) ⇒ no scorer ⇒ the `faceLikeness` field is omitted from each sidecar — there is no
/// identity to compare against, which is honest, not an error.
///
/// Decoding is non-fatal: a reference that fails to load logs and yields `None` (scores omitted, the
/// set still renders) — scoring NEVER aborts a generation (the sc-4407 non-fatal AC). The source image
/// is decoded here ONCE and handed to the per-job scorer, which embeds it ONCE (the caching AC).
///
/// Lives in `base.rs` (compiled under BOTH the macOS pose lanes and the off-Mac candle-control lanes)
/// rather than the macOS-only `strict_control.rs` include, so the not-macOS candle strict-pose siblings
/// (`zimage_control` / `qwen_control` / `kolors_control` / `flux2_control_candle`) can resolve it.
#[cfg(native_lane)]
pub(crate) fn resolve_control_identity_source(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> Option<(Image, String)> {
    let asset_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())?
        .to_owned();
    match load_reference_image(
        &settings.data_dir,
        &request.project_id,
        &asset_id,
        project_path,
    ) {
        Ok(image) => Some((image, asset_id)),
        Err(error) => {
            tracing::warn!(
                error = %error,
                "strict-control identity reference decode failed; likeness scores omitted \
                 (generation continues)"
            );
            None
        }
    }
}

/// The source identity reference (decoded image + its asset id) an Image Studio "With Character"
/// (`character_image`) generation scores each finished image against (epic 4406, sc-4411), or `None`
/// when the job is not a plain character image with a reference face.
///
/// This is the GENERAL With-Character case — a regular `character_image` generation against a
/// `referenceAssetId` (the character reference shown in the Image Studio reference thumbnail), NOT an
/// angle set (`advanced.angleSet`) and NOT a pose-library set (`advanced.poses`). Those two are ALSO
/// `character_image` jobs but are already scored by sc-4409 (angles) / sc-4410 (poses) through the same
/// shared seam; this resolver deliberately returns `None` for them so the plain case never
/// double-attaches or conflicts on an angle/pose job. The gate is therefore:
/// `mode == "character_image"` AND a non-empty `referenceAssetId` AND no angle/pose grouping.
///
/// The asset id returned is the CURRENT job's `referenceAssetId`, so changing the reference asset
/// changes the source the score is computed against (an explicit sc-4411 acceptance criterion) — the
/// source is never cached across jobs or hardcoded.
///
/// Decoding is non-fatal (the sc-4407 contract): a reference that fails to load logs and yields `None`
/// (scores omitted, the generation still runs) — scoring NEVER aborts a generation. The source image is
/// decoded here ONCE and handed to the per-job scorer, which embeds it ONCE (the caching AC).
///
/// Lives in `base.rs` (compiled under BOTH the macOS routes and the off-Mac candle-control lanes) so the
/// candle siblings can resolve it identically.
#[cfg(native_lane)]
pub(crate) fn resolve_character_image_likeness_source(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> Option<(Image, String)> {
    if !character_image_likeness_requested(request) {
        return None;
    }
    resolve_control_identity_source(request, settings, project_path)
}

#[cfg(native_lane)]
pub(super) fn character_image_likeness_requested(request: &ImageRequest) -> bool {
    request.mode == "character_image"
        && request
            .reference_asset_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty())
        // Angle / pose sets are already scored by sc-4409 / sc-4410 through the shared seam.
        && pose_entries(request).is_empty()
        && !advanced::flag(&request.advanced, "angleSet")
}

/// img2img (Remix) strength for a plain Ideogram 4 edit with no mask — mirrors the sdxl/z-image 0.6
/// edit default and the engine's `DEFAULT_IMG2IMG_STRENGTH`. Shared by the macOS MLX edit path and the
/// candle in-lane edit (sc-6598), so it compiles off-Mac under `backend-candle` too.
#[cfg(native_lane)]
const IDEOGRAM_EDIT_STRENGTH: f32 = 0.6;
/// Heavier img2img strength for masked inpaint / outpaint (regenerate the painted region) — mirrors
/// the sdxl 0.85 inpaint default and the engine's `DEFAULT_INPAINT_STRENGTH`.
#[cfg(native_lane)]
const IDEOGRAM_INPAINT_STRENGTH: f32 = 0.85;

/// Upper bound on reference images for a Boogu instruction edit (sc-7645). The DiT's
/// `image_index_embedding` carries 5 per-image index slots (OmniGen2 lineage), so `N ∈ [1, 5]`
/// references can be packed into one edit (e.g. subject-from-A composed into scene-from-B); a plural
/// picker beyond that is capped here. Matches the `mlx-gen-boogu` / `candle-gen-boogu` `MAX_EDIT_REFS`.
#[cfg(native_lane)]
pub(super) const BOOGU_MAX_EDIT_REFERENCES: usize = 5;

/// One `Conditioning::Reference` (a single reference) or one `Conditioning::MultiReference` (2–5) from
/// the resolved Boogu edit references (cloned per output). Empty references → empty (T2I fallback).
/// The single case stays a `Reference` so it is byte-identical to the pre-sc-7645 single-reference
/// path. Mirrors `flux2.rs::build_edit_conditioning`. The per-reference strength is inert for Boogu
/// (the edit is structural), so `None` is used. Not cfg-gated — called from the un-gated [`generate_one`].
pub(super) fn build_reference_conditioning(references: &[Image]) -> Vec<Conditioning> {
    match references {
        [] => Vec::new(),
        [single] => vec![Conditioning::Reference {
            image: single.clone(),
            strength: None,
        }],
        many => vec![Conditioning::MultiReference {
            images: many.to_vec(),
        }],
    }
}

/// The ordered reference list where entries may carry their own alpha (S4 contract, sc-24111).
///
/// With no alpha anywhere this IS [`build_reference_conditioning`], byte for byte. Otherwise every
/// entry is emitted individually and in order — `Conditioning::Reference` for an RGB entry,
/// `Conditioning::ReferenceRgba` for one that carries alpha — because a `MultiReference` holds only
/// RGB images and flattening an alpha-carrying reference would send a DIFFERENT request (the VAE
/// encodes all four channels; the vision tower's white composite is engine-internal). The engine
/// numbers `Reference` / `ReferenceRgba` / `MultiReference` entries in one ordered sequence, so the
/// per-entry form keeps every reference's slot. Strength is `None`: upstream's condition images
/// have none, and a provider without img2img strength requires it unset.
pub(super) fn build_ordered_reference_conditioning(
    references: &[Image],
    alpha: &[Option<image::GrayImage>],
) -> Vec<Conditioning> {
    if !alpha.iter().any(Option::is_some) {
        return build_reference_conditioning(references);
    }
    references
        .iter()
        .enumerate()
        .map(
            |(slot, image)| match alpha.get(slot).and_then(Option::as_ref) {
                None => Conditioning::Reference {
                    image: image.clone(),
                    strength: None,
                },
                Some(plane) => Conditioning::ReferenceRgba {
                    image: rgba_reference(image, plane),
                    strength: None,
                },
            },
        )
        .collect()
}

/// Interleave an RGB engine image with its (same-geometry) alpha plane into a straight-alpha
/// `RgbaImage`. The plane is decoded from the same asset as the image, so a geometry mismatch here
/// is a programming error; it is resampled defensively rather than misaligned.
fn rgba_reference(image: &Image, plane: &image::GrayImage) -> gen_core::RgbaImage {
    let resized;
    let plane = if plane.dimensions() == (image.width, image.height) {
        plane
    } else {
        resized = image::imageops::resize(
            plane,
            image.width,
            image.height,
            image::imageops::FilterType::Triangle,
        );
        &resized
    };
    let mut pixels = Vec::with_capacity(image.pixels.len() / 3 * 4);
    for (rgb, a) in image
        .pixels
        .as_chunks::<3>()
        .0
        .iter()
        .zip(plane.as_raw().iter())
    {
        pixels.extend_from_slice(rgb);
        pixels.push(*a);
    }
    gen_core::RgbaImage {
        width: image.width,
        height: image.height,
        pixels,
    }
}

/// The inverse of [`rgba_reference`]: split a straight-alpha `RgbaImage` into the lane's RGB engine
/// image and its alpha plane, byte for byte.
fn split_rgba_reference(image: &gen_core::RgbaImage) -> WorkerResult<(Image, image::GrayImage)> {
    let mut rgb = Vec::with_capacity(image.pixels.len() / 4 * 3);
    let mut plane = Vec::with_capacity(image.pixels.len() / 4);
    for pixel in image.pixels.as_chunks::<4>().0 {
        rgb.extend_from_slice(&pixel[..3]);
        plane.push(pixel[3]);
    }
    let plane = image::GrayImage::from_raw(image.width, image.height, plane).ok_or_else(|| {
        WorkerError::InvalidPayload("RGBA reference buffer size mismatch".to_owned())
    })?;
    Ok((
        Image {
            width: image.width,
            height: image.height,
            pixels: rgb,
        },
        plane,
    ))
}

/// Reference asset ids for a Boogu instruction edit, in order. The multi-image picker sends the plural
/// `referenceAssetIds` — take all of them, capped at [`BOOGU_MAX_EDIT_REFERENCES`]; with no plural list
/// it falls back to the single Image-Edit `sourceAssetId` (`edit_image` mode). Mirrors
/// `edit_reference_ids`.
#[cfg(native_lane)]
pub(super) fn boogu_edit_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !request.reference_asset_ids.is_empty() {
        // The parsed list is already trimmed + non-empty (sceneworks-core `string_list`).
        return request
            .reference_asset_ids
            .iter()
            .take(BOOGU_MAX_EDIT_REFERENCES)
            .cloned()
            .collect();
    }
    if let Some(id) = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        return vec![id.to_owned()];
    }
    Vec::new()
}

/// Mage Edit always conditions on the required primary source first, followed by the optional
/// `referenceAssetIds` exactly in client order.
#[cfg(native_lane)]
pub(super) fn mage_edit_reference_ids(request: &ImageRequest) -> Vec<String> {
    if request.mode != "edit_image" {
        return Vec::new();
    }
    let Some(source) = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Vec::new();
    };
    let mut ids = Vec::with_capacity(1 + request.reference_asset_ids.len());
    ids.push(source.to_owned());
    ids.extend(request.reference_asset_ids.iter().cloned());
    ids
}

#[cfg(native_lane)]
pub(super) fn is_mage_edit_model(model: &str) -> bool {
    matches!(
        model,
        "mage_flow_edit_base" | "mage_flow_edit" | "mage_flow_edit_turbo"
    )
}

/// One resolved Qwen-Image 2.1 condition image, with the alpha plane it arrived with (sc-24110).
///
/// The plane travels BESIDE the engine image rather than inside it because `gen_core::Image` is a
/// flat 3-channel buffer — the same reason sc-24111 carries it beside the render on the way out.
/// Which of the two conditioning kinds an entry becomes is decided from `alpha` alone, so the
/// classification is a property of the ASSET, never of its ordinal position.
#[cfg(native_lane)]
pub(crate) struct QwenImage21Reference {
    pub(crate) image: Image,
    /// `Some` iff the source asset carried an alpha channel. An RGBA reference is a DIFFERENT
    /// request from the same picture flattened over white (S4 contract), so this is never dropped
    /// silently — see [`build_qwen_image_2_1_conditioning`].
    pub(crate) alpha: Option<image::GrayImage>,
}

/// Is this the model whose edit route is ONE ordered conditioning list (sc-24110)?
///
/// Deliberately the two 2.1 engine ids (base and Turbo, sc-25907 — one request contract) rather
/// than a family predicate: the 2512-weights `qwen_image_edit*` ids are a different engine, a
/// different latent space and a different edit contract, and they keep the `ImageRoute::QwenEdit`
/// / `CandleImageRoute::QwenEdit` lanes they have always had.
#[cfg(native_lane)]
pub(super) fn is_qwen_image_2_1_edit(request: &ImageRequest) -> bool {
    sceneworks_core::jobs_store::is_qwen_image_2_1_model(&request.model)
        && !qwen_image_2_1_reference_ids(request).is_empty()
}

/// The ORDERED condition-image asset ids of a Qwen-Image 2.1 edit, in the order the engine numbers
/// them. The worker-side twin of `sceneworks_core::jobs_store::qwen_image_2_1_reference_ids`, over
/// the parsed [`ImageRequest`] instead of the raw payload — and pinned against it by
/// `qwen_image_2_1_worker_and_router_agree_on_the_reference_order`, because a disagreement between
/// the two would mean the API validated one list and the worker rendered another.
///
/// Order: `sourceAssetId`, then `maskAssetId` (an ORDINARY reference — 2.1 has no mask tensor),
/// then `referenceAssetIds` in submitted order, then the singular `referenceAssetId`.
///
/// **MODE-INDEPENDENT**, matching the router exactly. A payload with no `mode` and a
/// `sourceAssetId`, or `mode: "image_generation"` with a `referenceAssetIds` list, is claimed by
/// the router as a conditioned request — so if this function consulted the mode it would return
/// empty for a job the router already admitted, and the references would be silently dropped into
/// a plain text-to-image render. There is no mode axis in the upstream contract at all: an empty
/// ordered list IS text-to-image and a non-empty one IS the edit call.
///
/// **DEDUPED by asset id, keeping the first occurrence.** The web's `editReferenceIds` leads
/// `referenceAssetIds` with the working image while `buildEditJobBody` also sets `sourceAssetId`,
/// so the ordinary Image-Editor payload names the same asset twice. Sending it twice is not a
/// harmless duplicate here: every entry occupies one of the ten slots and gets its own number in
/// the prompt template, so a duplicate silently costs a slot AND renumbers every reference after
/// it. First-occurrence wins because the earlier slot is the one the prompt refers to.
///
/// **No `.take(N)`.** Every other edit lane silently truncates an over-long set; here the API
/// refuses it with a 400 that names the cap, so a list that reaches the worker is already inside
/// the cap and truncating would only hide a routing defect.
#[cfg(native_lane)]
pub(super) fn qwen_image_2_1_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !sceneworks_core::jobs_store::is_qwen_image_2_1_model(&request.model) {
        return Vec::new();
    }
    let scalar = |value: &Option<String>| -> Option<String> {
        value
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
    };
    let mut ids = Vec::with_capacity(2 + request.reference_asset_ids.len());
    ids.extend(scalar(&request.source_asset_id));
    ids.extend(scalar(&request.mask_asset_id));
    ids.extend(request.reference_asset_ids.iter().cloned());
    ids.extend(scalar(&request.reference_asset_id));
    let mut seen = std::collections::HashSet::with_capacity(ids.len());
    ids.retain(|id| seen.insert(id.clone()));
    ids
}

/// Resolve those ids into engine images at their NATIVE geometry, each with the alpha plane its
/// asset carried (same geometry as the RGB it belongs to).
///
/// Unlike the other registry editors, NOTHING here fits a reference to the output W×H (sc-24114).
/// Upstream never crops or letterboxes a condition image: the engine fits each reference itself,
/// aspect-preserved, onto its ~1024²-area 32-px grid (S3 contract), and that one resize feeds both
/// the vision tower and the VAE. A worker-side `fit_engine_image` to the OUTPUT aspect ran first
/// and so centre-cropped (default `fitMode: "crop"`) or black-letterboxed every portrait reference
/// in a landscape request before the engine ever saw it — a different request from upstream's.
#[cfg(native_lane)]
pub(super) fn resolve_qwen_image_2_1_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Vec<QwenImage21Reference>> {
    let ids = qwen_image_2_1_reference_ids(request);
    // The ceiling comes from the job's own resolved manifest entry — the SAME `limits` key the
    // enqueue gate read — so the worker cannot hold a second opinion about what "too many" means,
    // and a manifest edit moves both. Absent means no cap, exactly as it does at enqueue.
    if let Some(cap) =
        sceneworks_core::video_request::image_max_reference_assets(&request.model_manifest_entry)
    {
        if ids.len() > cap {
            // Unreachable through the API, which refuses this at enqueue. Stated anyway because the
            // worker is also driven directly by tests and by jobs stored before the gate existed,
            // and because the alternative every sibling lane chose — a silent `.take(N)` — renders
            // a DIFFERENT request than the one asked for and reports success.
            return Err(WorkerError::InvalidPayload(format!(
                "qwen_image_2_1: {} reference images were supplied; upstream composes at most {cap}",
                ids.len()
            )));
        }
    }
    let mut references = Vec::with_capacity(ids.len());
    for id in &ids {
        // `FlattenPolicy::Truncate`, named EXPLICITLY rather than taken from the default
        // (sc-24110 answering the note on [`FlattenPolicy`]).
        //
        // Truncation drops the fourth byte and keeps the STRAIGHT RGB, which is exactly the colour
        // half of `Conditioning::ReferenceRgba`: an alpha-carrying reference travels UN-FLATTENED,
        // RGB + its own alpha plane, and the engine composites the vision-tower copy over white
        // itself (S4 contract). `OverWhite` would bake that composite into the colour the VAE
        // encodes — a different request. On an opaque reference the two policies are identical.
        let source = load_reference_image_with(
            &settings.data_dir,
            &request.project_id,
            id,
            project_path,
            FlattenPolicy::Truncate,
        )?;
        // sc-24111's lane, reused verbatim: the same asset, read through the same
        // `safe_project_path` confinement, for its alpha plane alone. Neither half is resampled,
        // so the plane is aligned with the RGB by construction (both are the decoded asset).
        let alpha =
            load_reference_alpha(&settings.data_dir, &request.project_id, id, project_path)?;
        references.push(QwenImage21Reference {
            image: source,
            alpha,
        });
    }
    Ok(references)
}

/// The ONE ordered conditioning list a Qwen-Image 2.1 edit sends, built from the resolved
/// references (sc-24110).
///
/// * No references → empty (the text-to-image call).
/// * All-RGB → exactly what every other registry editor sends: one `Conditioning::Reference` for a
///   single image, one `Conditioning::MultiReference` for many. Both kinds flatten into the same
///   ordered list engine-side, so the single case stays byte-identical to the one-reference path.
/// * **Never `Conditioning::Mask`.** The engine does not declare that kind and refuses it by name;
///   a mask asset is an ordinary ordered reference (see [`qwen_image_2_1_reference_ids`]). Such a
///   carrier reaches this route only from a DIRECT API CALLER OR WORKFLOW REPLAY — the Image
///   Editor gates its mask tool on `image_inpaint`, which this model does not declare.
/// * Strength is always `None` — upstream's condition images have no strength, and the engine
///   refuses anything but an unset-or-1.0 value.
///
/// * An alpha-carrying reference → `Conditioning::ReferenceRgba`, UN-flattened (S4 contract): the
///   VAE encodes all four channels, so flattening it would send a DIFFERENT request. A list with
///   any such entry is emitted per entry, in order ([`build_ordered_reference_conditioning`]).
#[cfg(native_lane)]
pub(super) fn build_qwen_image_2_1_conditioning(
    references: &[QwenImage21Reference],
) -> WorkerResult<Vec<Conditioning>> {
    let images: Vec<Image> = references.iter().map(|entry| entry.image.clone()).collect();
    let alpha: Vec<Option<image::GrayImage>> =
        references.iter().map(|entry| entry.alpha.clone()).collect();
    let conditioning = build_ordered_reference_conditioning(&images, &alpha);
    // The per-entry classification is `qwen_alpha::reference_conditioning_kind` (sc-24113), so
    // both halves of the epic name the carrier the same way. Asserted rather than assumed: the
    // builder above and the classifier must never disagree about what an entry became.
    if conditioning.len() == references.len() {
        for (entry, built) in references.iter().zip(&conditioning) {
            let built_kind = match built {
                Conditioning::ReferenceRgba { .. } => {
                    crate::qwen_alpha::CONDITIONING_REFERENCE_RGBA
                }
                _ => "Reference",
            };
            if built_kind != crate::qwen_alpha::reference_conditioning_kind(entry.alpha.is_some()) {
                return Err(WorkerError::Engine(format!(
                    "qwen_image_2_1: reference conditioning disagrees with its classification \
                     ({built_kind})"
                )));
            }
        }
    }
    Ok(conditioning)
}

/// The ordered condition images a Qwen-Image 2.1 edit contributes to the generic lane's
/// `edit_refs` slot, with each one's alpha plane (`None` for an opaque asset) — resolved,
/// validated, and in request order.
///
/// [`build_qwen_image_2_1_conditioning`] is the ONE source of truth for what this route sends: its
/// answer is unpacked here into the lane's `edit_refs` slot (plus each entry's alpha plane), and
/// `build_lane_conditioning_with_alpha` re-wraps those into exactly that answer (`Reference` for
/// one opaque image, `MultiReference` for many, per-entry `Reference` / `ReferenceRgba` once any
/// entry carries alpha — pinned by `qwen_image_2_1_lane_conditioning_matches_the_dedicated_builder`
/// and the live-path tests). So the builder's shape is what reaches the engine, and a builder that
/// ever emitted anything else — a `Mask` above all — is refused here rather than silently dropped.
/// Running it here also means a refusal happens before any weights are loaded, next to the asset
/// reads that produced it, rather than as a render failure.
#[cfg(native_lane)]
pub(super) fn resolve_qwen_image_2_1_edit_images(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<(Vec<Image>, Vec<Option<image::GrayImage>>)> {
    let references = resolve_qwen_image_2_1_edit(request, settings, project_path)?;
    let mut images = Vec::with_capacity(references.len());
    let mut alpha = Vec::with_capacity(references.len());
    for conditioning in build_qwen_image_2_1_conditioning(&references)? {
        match conditioning {
            Conditioning::Reference {
                image,
                strength: None,
            } => {
                images.push(image);
                alpha.push(None);
            }
            Conditioning::MultiReference { images: ordered } => {
                alpha.extend(ordered.iter().map(|_| None));
                images.extend(ordered);
            }
            // S4: an alpha-carrying reference, split back into the lane's RGB slot + its plane.
            Conditioning::ReferenceRgba {
                image,
                strength: None,
            } => {
                let (rgb, plane) = split_rgba_reference(&image)?;
                images.push(rgb);
                alpha.push(Some(plane));
            }
            other => {
                return Err(WorkerError::InvalidPayload(format!(
                    "qwen_image_2_1: the ordered conditioning list may carry only strength-less \
                     reference images, but the builder produced {other:?}"
                )))
            }
        }
    }
    Ok((images, alpha))
}

/// Everything the generic MLX lane conditions one render on, resolved once per job: the
/// per-family slots from [`resolve_generic_lane_conditioning`] plus the registry editors' ordered
/// `edit_refs` (Boogu, Mage, Qwen-Image 2.1).
///
/// Split out of [`generate_stream`] (sc-24110) so the conditioning a job REALLY sends is testable
/// without weights: [`generate_one`] assembles `build_lane_conditioning(identity_init, &edit_refs,
/// mask)` from exactly this tuple, so a test that resolves it for a real on-disk payload is looking
/// at the live path, not at a restatement of it.
#[cfg(target_os = "macos")]
#[allow(clippy::type_complexity)]
pub(super) fn resolve_generic_lane_inputs(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
    has_reference: bool,
) -> WorkerResult<(LaneConditioning, Vec<Image>, Vec<Option<image::GrayImage>>)> {
    // Per-family reference conditioning (Z-Image identity/edit-init, FLUX.1/Kolors IP-Adapter,
    // Kolors img2img, Ideogram edit + mask), resolved once — same predicate order + per-family
    // values as the historical inline 5-way match, table-ized into one resolver (sc-8828, F-026).
    // The strict-pose ControlNet / edit tiers divert earlier in `resolve_image_route`.
    let lane = resolve_generic_lane_conditioning(request, settings, project_path, has_reference)?;
    // Registry instruction edits: Boogu resolves 1..5 sources; Mage resolves its required primary
    // source followed by every optional reference in client order. Both thread through
    // `generate_one` as `Reference` (one) / `MultiReference` (many), never the img2img-init slot.
    // The alpha plane of each `edit_refs` entry — only Qwen-Image 2.1 ever fills it (S4 contract).
    let mut edit_ref_alpha: Vec<Option<image::GrayImage>> = Vec::new();
    let edit_refs: Vec<Image> = if request.model == "boogu_image_edit" {
        resolve_boogu_edit(request, settings, project_path)?
    } else if is_mage_edit_model(&request.model) {
        resolve_mage_edit(request, settings, project_path)?
    } else if is_qwen_image_2_1_edit(request) {
        // Qwen-Image 2.1 reference / local editing (sc-24110): ONE ordered list of 1..=10 condition
        // images, source → mask → submitted references. No `ImageRoute` variant of its own — the id
        // is in MODEL_TABLE, so an edit lands on the generic `Mlx` arm exactly as Mage Edit does,
        // and the ordering + the never-a-Mask guarantee live in the resolver.
        let (images, alpha) = resolve_qwen_image_2_1_edit_images(request, settings, project_path)?;
        edit_ref_alpha = alpha;
        images
    } else {
        Vec::new()
    };
    // The never-a-Mask / never-a-strength guarantee, on the path that actually runs.
    guard_qwen_image_2_1_lane_slots(
        request,
        lane.identity_init.as_ref(),
        lane.ideogram_edit_mask.as_ref(),
    )?;
    Ok((lane, edit_refs, edit_ref_alpha))
}

/// Refuse a Qwen-Image 2.1 render whose generic-lane slots carry anything but the ordered
/// reference list (sc-24110).
///
/// The never-a-Mask guarantee has to hold on the path that actually RUNS. What `generate_one`
/// sends is [`build_lane_conditioning`]`(identity_init, &edit_refs, edit_mask)` — so
/// [`build_qwen_image_2_1_conditioning`] can state the contract perfectly and still be bypassed if
/// either of the other two slots is ever populated for this model. Today neither is
/// (`resolve_generic_lane_conditioning` is a per-family table and 2.1 is in none of its arms, and
/// the candle lane's `edit_reference` is likewise family-keyed), which is exactly the problem: it
/// holds by accident of a table this model is absent from, and adding an arm for it would silently
/// start sending the engine a `Conditioning::Mask` it refuses by name, or an img2img-init
/// `Reference` carrying a strength it also refuses.
///
/// So the invariant is ASSERTED at the seam rather than inferred. Erroring is the right answer
/// over quietly clearing the slots: a populated slot means some caller believes this model has an
/// img2img or inpaint surface, and it does not — upstream's pipeline takes only an ordered list of
/// condition images (S3 contract).
#[cfg(native_lane)]
pub(super) fn guard_qwen_image_2_1_lane_slots(
    request: &ImageRequest,
    identity_init: Option<&(Image, f32)>,
    edit_mask: Option<&Image>,
) -> WorkerResult<()> {
    if !sceneworks_core::jobs_store::is_qwen_image_2_1_model(&request.model) {
        return Ok(());
    }
    if edit_mask.is_some() {
        return Err(WorkerError::InvalidPayload(
            "qwen_image_2_1: an inpaint mask reached the generic lane's mask slot, which would be \
             sent as Conditioning::Mask — a carrier this engine does not declare and refuses by \
             name. 2.1 has no mask tensor and performs no inpainting: draw the annotation into the \
             reference, or pass the mask as an ordinary ordered reference the prompt names."
                .to_owned(),
        ));
    }
    if identity_init.is_some() {
        return Err(WorkerError::InvalidPayload(
            "qwen_image_2_1: an img2img-init reference reached the generic lane's single-reference \
             slot, which would be sent with a strength. Upstream's condition images have no \
             strength and this engine refuses one; every 2.1 reference travels in the ordered \
             conditioning list instead."
                .to_owned(),
        ));
    }
    Ok(())
}

/// Resolve the Boogu instruction-edit sources: the `N ∈ [1, 5]` reference images (plural
/// `referenceAssetIds`, else the single `sourceAssetId` — [`boogu_edit_reference_ids`]), each fit to the
/// output W×H (so it satisfies the engine's multiple-of-16 guard and aligns to the target aspect).
/// Returns the references in order; **empty** when not an edit / no source. The engine treats one
/// reference as `Conditioning::Reference` and 2–5 as `Conditioning::MultiReference` (each read by the
/// Qwen3-VL vision tower + VAE-encoded into the DiT spatial sequence). The per-reference strength is
/// inert for Boogu (the edit is structural — the engine ignores `Conditioning::Reference.strength`). No
/// mask / outpaint path (the descriptor accepts only `Reference` / `MultiReference`).
///
/// Shared by the macOS MLX `generate_stream` and the off-Mac candle `generate_candle_stream` (sc-7524):
/// Boogu is the same engine family for T2I and edit on both backends (the registered `boogu_image_edit`
/// resolves the source `Reference`(s) in-lane, like Ideogram), so both lanes resolve the edit sources the
/// same way. Its deps (`load_reference_image`, `fit_engine_image`) already compile off-Mac under
/// `backend-candle` (the Ideogram edit path uses them too).
#[cfg(native_lane)]
pub(super) fn resolve_boogu_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Vec<Image>> {
    if request.mode != "edit_image" {
        return Ok(Vec::new());
    }
    let ids = boogu_edit_reference_ids(request);
    let mut references = Vec::with_capacity(ids.len());
    for id in &ids {
        let source =
            load_reference_image(&settings.data_dir, &request.project_id, id, project_path)?;
        let source = fit_engine_image(source, request.width, request.height, &request.fit_mode)?;
        references.push(source);
    }
    Ok(references)
}

#[cfg(native_lane)]
pub(super) fn resolve_mage_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Vec<Image>> {
    let ids = mage_edit_reference_ids(request);
    let mut references = Vec::with_capacity(ids.len());
    for id in &ids {
        let source =
            load_reference_image(&settings.data_dir, &request.project_id, id, project_path)?;
        references.push(fit_engine_image(
            source,
            request.width,
            request.height,
            &request.fit_mode,
        )?);
    }
    Ok(references)
}

/// Resolve the Ideogram 4 `edit_image` conditioning (sc-6303) into the base MLX path's
/// `(source, strength, optional-mask)` shape (→ the engine's `Conditioning::Reference` +
/// `Conditioning::Mask`). Three sub-shapes, mirroring the sdxl edit classification:
///   * **img2img / Remix** — `sourceAssetId`, no mask: pre-fit the source to the output W×H
///     (crop/pad, never stretch) → `(source, 0.6, None)`.
///   * **masked inpaint** — `+ maskAssetId`: the mask fit with the same geometry → `(source, 0.85,
///     Some(mask))` (white = repaint).
///   * **outpaint** — `fit_mode == "outpaint"`: contain-pad the source onto the canvas and generate
///     the border via [`gen_core::imageops::outpaint_border_mask`] (using the ORIGINAL source dims so
///     it lines up), unioning any user mask (white wins).
///
/// `None` when not an edit job or no source asset (the caller falls back to plain txt2img).
///
/// Shared by the macOS MLX `generate_stream` and the off-Mac candle `generate_candle_stream`
/// (sc-6598): Ideogram is the same engine for T2I and edit on both backends, so both lanes resolve the
/// edit conditioning the same way. Its deps (`load_reference_image`, `fit_engine_image`, `non_empty`,
/// the `gen_core::imageops` mask helpers) are all already compiled off-Mac under `backend-candle`.
#[cfg(native_lane)]
pub(super) fn resolve_ideogram_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32, Option<Image>)>> {
    if request.mode != "edit_image" {
        return Ok(None);
    }
    let Some(asset_id) = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Ok(None);
    };
    let source = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    let is_outpaint = request.fit_mode == "outpaint";
    let has_user_mask = non_empty(&request.mask_asset_id);
    let strength = advanced::f32_clamped(
        &request.advanced,
        "strength",
        if is_outpaint || has_user_mask {
            IDEOGRAM_INPAINT_STRENGTH
        } else {
            IDEOGRAM_EDIT_STRENGTH
        },
        0.05..=1.0,
    );

    if is_outpaint {
        // Pad the source onto the target canvas (contain) and regenerate the border. The border mask
        // uses the ORIGINAL source dims so it lines up with the padded canvas (same contain geometry
        // as `fit_engine_image`'s "outpaint"/pad). Any user mask unions into the border (white wins).
        let (src_w, src_h) = (source.width, source.height);
        let canvas = fit_engine_image(source, request.width, request.height, "outpaint")?;
        let mut mask =
            gen_core::imageops::outpaint_border_mask(src_w, src_h, request.width, request.height);
        if has_user_mask {
            let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
            let user_mask = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                mask_id,
                project_path,
            )?;
            let user_mask = fit_engine_image(user_mask, request.width, request.height, "pad")?;
            mask = gen_core::imageops::union_masks(&mask, &user_mask).map_err(|error| {
                WorkerError::Engine(format!("ideogram outpaint mask union failed: {error}"))
            })?;
        }
        return Ok(Some((canvas, strength, Some(mask))));
    }

    // img2img / inpaint: pre-fit the source to the output W×H so an off-aspect edit doesn't stretch.
    let source = fit_engine_image(source, request.width, request.height, &request.fit_mode)?;
    let mask = if has_user_mask {
        let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
        let user_mask = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            mask_id,
            project_path,
        )?;
        // Align the mask to the source with the SAME fit geometry.
        Some(fit_engine_image(
            user_mask,
            request.width,
            request.height,
            &request.fit_mode,
        )?)
    } else {
        None
    };
    Ok(Some((source, strength, mask)))
}

// ---------------------------------------------------------------------------
// Shared pose + angle-prompt helpers. Used by the macOS Z-Image strict-pose control path
// (`zimage.rs`) AND the InstantID lane (`instantid.rs`) on BOTH backends — the candle InstantID
// provider (sc-5491) needs them off-Mac, so they live here in the shared include rather than in the
// macOS-only `zimage.rs` (same reason `load_reference_image` does). All `include!`d image-job files
// share one module, so moving these here keeps them visible to `zimage.rs` on macOS unchanged.
// ---------------------------------------------------------------------------

/// True for a present, non-blank optional asset id (the conditioning-asset presence test shared by
/// the SDXL advanced sub-mode, PuLID, and InstantID gates). Moved here from the macOS-only `sdxl.rs`
/// so the candle InstantID lane (sc-5491) can use it off-Mac.
#[cfg(native_lane)]
pub(super) fn non_empty(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(|id| !id.trim().is_empty())
}

/// The object-shaped `advanced.poses` entries (the strict-pose tier; empty otherwise).
#[cfg(native_lane)]
pub(super) fn pose_entries(request: &ImageRequest) -> Vec<&Value> {
    request
        .advanced
        .get("poses")
        .and_then(Value::as_array)
        .map(|poses| poses.iter().filter(|pose| pose.is_object()).collect())
        .unwrap_or_default()
}

/// A pose's parsed keypoints, ready for [`crate::openpose_skeleton::draw_wholebody`].
// Shared strict-control preprocessing reads the whole body on both native backends.
#[cfg(native_lane)]
pub(super) struct PoseInput {
    pub(super) keypoints: Vec<crate::openpose_skeleton::Keypoint>,
    pub(super) hands: Option<Vec<crate::openpose_skeleton::Hand>>,
    pub(super) face: Option<Vec<crate::openpose_skeleton::Keypoint>>,
}

#[cfg(native_lane)]
pub(super) fn parse_poses(request: &ImageRequest) -> Vec<PoseInput> {
    use crate::openpose_skeleton::{normalize_face, normalize_hands, normalize_keypoints};
    pose_entries(request)
        .into_iter()
        .map(|entry| PoseInput {
            keypoints: entry
                .get("keypoints")
                .map(normalize_keypoints)
                .unwrap_or_else(|| vec![None; 18]),
            hands: entry.get("hands").and_then(normalize_hands),
            face: entry.get("face").and_then(normalize_face),
        })
        .collect()
}

/// Stage the antelopev2 face stack for identity-likeness scoring (epic 4406), collapsing the
/// warn-and-`None` staging block duplicated across every scored image lane (F-024, sc-8826). When
/// `should_stage` is false the stack is never fetched and this is `None`; when true it downloads the
/// shared InstantID bundle (a no-op if already cached) and returns its dir. Staging is **non-fatal**:
/// a download failure logs `warn_message` and yields `None`, so the scorer is simply skipped and the
/// generation still renders (no scores). `warn_message` is the per-lane phrasing (e.g. the
/// `character_image` edit streams vs. the `pose-set` control lanes) so the log line is unchanged from
/// the hand-written blocks this replaces.
#[cfg(native_lane)]
pub(super) async fn stage_likeness(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    should_stage: bool,
    warn_message: &'static str,
) -> Option<PathBuf> {
    if !should_stage {
        return None;
    }
    match ensure_face_stack_dir(api, settings, job).await {
        Ok(dir) => Some(dir),
        Err(error) => {
            tracing::warn!(error = %error, "{warn_message}");
            None
        }
    }
}

/// The per-angle continuation clause appended to the user's prompt (parity with
/// `character_studio_angles.ANGLE_PROMPT_AUGMENTS`). Unknown angle → empty.
#[cfg(native_lane)]
pub(super) fn angle_prompt_augment(angle: &str) -> &'static str {
    match angle {
        "front" => {
            "frontal portrait, looking directly at the camera, head and shoulders, neutral expression"
        }
        "three_quarter_left" => {
            "three-quarter left profile, head turned slightly to the left, three-quarter view"
        }
        "three_quarter_right" => {
            "three-quarter right profile, head turned slightly to the right, three-quarter view"
        }
        "left_profile" => {
            "full left profile, head turned 90 degrees to the left, side view of the head"
        }
        "right_profile" => {
            "full right profile, head turned 90 degrees to the right, side view of the head"
        }
        "up" => "looking up, head tilted slightly upward toward the sky",
        "down" => "looking down, head tilted slightly downward toward the floor",
        "up_left" => {
            "looking up and to the left, head tilted slightly upward and turned slightly to the left"
        }
        "up_right" => {
            "looking up and to the right, head tilted slightly upward and turned slightly to the right"
        }
        "down_left" => {
            "looking down and to the left, head tilted slightly downward and turned slightly to the left"
        }
        "down_right" => {
            "looking down and to the right, head tilted slightly downward and turned slightly to the right"
        }
        _ => "",
    }
}

/// Strip the user's base prompt for augmentation: trim whitespace, then trailing
/// `,`/`.`/`;` — exactly Python's `(base or "").strip().rstrip(",.;")` (which can
/// leave a trailing space, e.g. `"a . "` → `"a "`).
#[cfg(native_lane)]
pub(super) fn strip_base_prompt(base: &str) -> &str {
    base.trim().trim_end_matches([',', '.', ';'])
}

/// Append the per-angle clause to the user's base prompt (parity with
/// `augment_prompt_for_angle`). Empty base + unknown angle → empty string.
#[cfg(native_lane)]
pub(super) fn augment_prompt_for_angle(base: &str, angle: &str) -> String {
    let augment = angle_prompt_augment(angle);
    let base = strip_base_prompt(base);
    if !base.is_empty() && !augment.is_empty() {
        format!("{base}, {augment}")
    } else if !augment.is_empty() {
        augment.to_owned()
    } else {
        base.to_owned()
    }
}

/// True when an Image-Edit source should be fitted to the requested output geometry before it is
/// handed to an img2img provider. This lives in the cross-backend base lane because both the MLX
/// edit routes and the Candle Z-Image alias consume it.
pub(super) fn should_fit_edit_source(request: &ImageRequest) -> bool {
    let has_source = request
        .source_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    let no_reference = !request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    request.mode == "edit_image" && has_source && no_reference && request.fit_mode != "stretch"
}

/// Resolve the Z-Image Turbo provider's shared img2img init. Both backends map `z_image_edit` to
/// the registered `z_image_turbo` generator, so this resolver must live in the cross-backend lane.
pub(super) fn resolve_zimage_edit_init(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    if request.mode != "edit_image" {
        return Ok(None);
    }
    let Some(asset_id) = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Ok(None);
    };
    let source = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    let image = if should_fit_edit_source(request) {
        fit_engine_image(source, request.width, request.height, &request.fit_mode)?
    } else {
        source
    };
    let strength = advanced::f32_clamped(&request.advanced, "strength", 0.6, 0.05..=1.0);
    Ok(Some((image, strength)))
}

/// Per-family reference conditioning for the generic MLX lane (`generate_stream`), resolved once
/// (constant across the generation set). Bundles the four values the family dispatch produces so the
/// caller does one `resolve_generic_lane_conditioning(..)` call instead of the inline 5-way match —
/// the historical place per-family drift bugs land (sc-8828, F-026). The families served:
///  • Z-Image reference-identity img2img-init / edit-init (sc-3619 / epic 3529),
///  • FLUX.1 XLabs IP-Adapter (epic 3621 — schnell + dev; `strength = ipAdapterScale`, real CFG via
///    `trueCfgScale` on dev),
///  • Kolors img2img (sc-4765) + IP-Adapter-Plus reference (sc-4767), and
///  • Ideogram 4 img2img (Remix) + mask inpaint/outpaint (sc-6303).
/// Every other family (plain t2i, Boogu multi-reference) resolves to the all-`None`/`Vec::new` default.
#[cfg(target_os = "macos")]
#[derive(Default)]
pub(super) struct LaneConditioning {
    /// Single img2img-init / IP-Adapter reference image + strength (Z-Image, FLUX.1 IP, Kolors,
    /// Ideogram) fed to the engine as `Conditioning::Reference`. `None` for plain t2i.
    pub(super) identity_init: Option<(Image, f32)>,
    /// FLUX.1 / Kolors IP-Adapter weights directory threaded into the [`load_spec`]. `None` unless the
    /// IP-Adapter reference path is active.
    pub(super) flux_ip_dir: Option<PathBuf>,
    /// FLUX.1-dev reference path's real-CFG scale (`trueCfgScale`). `None` for distilled/guidance-scalar
    /// families; the caller folds it into the effective `true_cfg` alongside the true-CFG-family scale.
    pub(super) flux_true_cfg: Option<f32>,
    /// Ideogram 4 inpaint/outpaint mask (white = repaint) threaded to `generate_one` as
    /// `Conditioning::Mask`. `None` for every other family / plain img2img.
    pub(super) ideogram_edit_mask: Option<Image>,
}

/// Resolve the [`LaneConditioning`] for `request` on the generic MLX lane. Mirrors the historical
/// inline family dispatch EXACTLY — same predicate order, same per-family values — so routing is
/// byte-identical (sc-8828). The strict-pose ControlNet / edit tiers divert earlier (in
/// `resolve_image_route`), so only the reference/identity/img2img families reach here.
#[cfg(target_os = "macos")]
pub(super) fn resolve_generic_lane_conditioning(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
    has_reference: bool,
) -> WorkerResult<LaneConditioning> {
    if matches!(request.model.as_str(), "z_image_turbo" | "z_image_edit") {
        // Z-Image base path: `edit_image` → img2img-edit (sourceAssetId + strength, epic 3529);
        // otherwise the identity-init reference (referenceAssetId + referenceStrength, sc-3619).
        // Both feed the engine's single `Reference` conditioning; only the source + strength
        // keying differs. The strict-pose ControlNet tier diverts earlier (zimage_control_available).
        // Character Studio identity-init (referenceStrength, sc-3619) is the primary reference surface
        // and keeps precedence — it also drives face-likeness scoring. When it doesn't engage (the Image
        // Studio "Image reference" tile sends referenceAssetId + advanced.strength with NO
        // referenceStrength), the generic `ui.img2img` reference-guided latent-init (epic 8588 A4.5,
        // sc-10193) takes over so the slider actually reaches the engine. The two surfaces are mutually
        // exclusive by mode (character_image vs text_to_image), so this never double-drives; both produce
        // the same single `Conditioning::Reference`.
        let init = if request.mode == "edit_image" {
            resolve_zimage_edit_init(request, settings, project_path)?
        } else if zimage_uses_generic_img2img(request, has_reference) {
            resolve_img2img_init_generic(request, settings, project_path)?
        } else {
            resolve_identity_init(request, settings, project_path)?
        };
        Ok(LaneConditioning {
            identity_init: init,
            ..Default::default()
        })
    } else if is_flux_model(&request.model) && has_reference && request.mode != "edit_image" {
        let reference_id = request
            .reference_asset_id
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .to_owned();
        let image = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            &reference_id,
            project_path,
        )?;
        let scale = advanced::f32_clamped(
            &request.advanced,
            "ipAdapterScale",
            FLUX_IP_SCALE,
            0.0..=1.0,
        );
        let ip_dir = resolve_flux_ip_adapter_dir(settings)?;
        // Real CFG only on dev (schnell is distilled — no CFG).
        let true_cfg = (request.model == "flux_dev").then(|| {
            advanced::f32_clamped(
                &request.advanced,
                "trueCfgScale",
                FLUX_IP_TRUE_CFG,
                1.0..=10.0,
            )
        });
        Ok(LaneConditioning {
            identity_init: Some((image, scale)),
            flux_ip_dir: Some(ip_dir),
            flux_true_cfg: true_cfg,
            ideogram_edit_mask: None,
        })
    } else if request.model == "kolors" && request.mode == "edit_image" {
        // Kolors img2img (sc-4765): `sourceAssetId` + `strength` → the engine's `Reference`
        // (img2img init, no IP-Adapter loaded). Kolors carries CFG through `guidance` + negative
        // prompt (resolved above), not `true_cfg`.
        let init = resolve_kolors_edit_init(request, settings, project_path)?;
        Ok(LaneConditioning {
            identity_init: init,
            ..Default::default()
        })
    } else if request.model == "kolors" && has_reference {
        // Kolors IP-Adapter-Plus reference (sc-4767): `referenceAssetId` → the IP image prompt at
        // `ipAdapterScale`. `with_ip_adapter` makes the engine treat the `Reference` as the image
        // prompt (decoupled cross-attn) rather than an img2img init.
        let (image, scale) = resolve_kolors_ip_reference(request, settings, project_path)?;
        let ip_dir = resolve_kolors_ip_adapter_dir(settings)?;
        Ok(LaneConditioning {
            identity_init: Some((image, scale)),
            flux_ip_dir: Some(ip_dir),
            flux_true_cfg: None,
            ideogram_edit_mask: None,
        })
    } else if matches!(request.model.as_str(), "ideogram_4" | "ideogram_4_turbo")
        && request.mode == "edit_image"
    {
        // Ideogram 4 img2img (Remix) + mask inpaint / outpaint (Edit), sc-6303: `sourceAssetId` →
        // the engine's `Reference` (img2img init); a `maskAssetId` (inpaint) or `fit_mode ==
        // "outpaint"` adds a `Conditioning::Mask` (white = repaint), threaded via `ideogram_edit_mask`.
        // Works in both quality (`ideogram_4`) and turbo (same base + TurboTime LoRA). No IP-Adapter.
        match resolve_ideogram_edit(request, settings, project_path)? {
            Some((source, strength, mask)) => Ok(LaneConditioning {
                identity_init: Some((source, strength)),
                flux_ip_dir: None,
                flux_true_cfg: None,
                ideogram_edit_mask: mask,
            }),
            None => Ok(LaneConditioning::default()),
        }
    } else if uses_generic_img2img(request, has_reference) {
        // Generic plain-t2i img2img latent-init for any `ui.img2img` model (epic 8588 A4, sc-10189):
        // a `referenceAssetId` + `advanced.strength` seeds the denoise from the VAE-encoded reference,
        // which the engine routes to that model's img2img entrypoint via the single
        // `Conditioning::Reference`. Reached today by Krea 2 Turbo (sc-8591 #666), SD3.5
        // large/turbo/medium (sc-10189 #667), SANA base/Sprint (sc-18475), base `z_image` (sc-10265),
        // and Boogu / Ideogram 4 in a plain-t2i shape (Ideogram's bespoke arm above keys on
        // `edit_image`; Boogu's instruction editor is the separate `boogu_image_edit` id resolved
        // through `edit_refs`, so neither intercepts a t2i reference). A new text-only
        // model joins by flipping `ui.img2img` and landing its engine entrypoint. Sits after the
        // model-specific reference arms (z-image/flux/kolors/ideogram) so their bespoke surfaces keep
        // precedence; `z_image_turbo` reaches this same resolver from inside the z-image arm via
        // [`zimage_uses_generic_img2img`].
        Ok(LaneConditioning {
            identity_init: resolve_img2img_init_generic(request, settings, project_path)?,
            ..Default::default()
        })
    } else {
        // Multi-reference instruction editors resolve their ordered references into `edit_refs` below;
        // they use the `MultiReference`-capable path, not the single `identity_init` reference.
        Ok(LaneConditioning::default())
    }
}

#[cfg(native_lane)]
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/kolors_edit_worker_contract_tests.rs"]
mod kolors_edit_worker_contract_tests;

#[cfg(native_lane)]
#[cfg(all(test, native_lane))]
#[path = "tests/text_style_gain_gate_tests.rs"]
mod text_style_gain_gate_tests;
