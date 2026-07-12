// `EditGrouping` / `edit_grouping` moved to base.rs (sc-8946, F-144): they are the SHARED grouping for
// the FLUX.2 / Qwen-Edit / SenseNova-U1 edit lanes, not FLUX.2-specific, so navigation lands in base.rs.

/// The per-iteration plan shared by every native edit stream (FLUX.2 / Qwen-Edit / SenseNova-U1,
/// F-024 sc-8826): the `(seeds, prompts, pose_inputs)` grouping expansion, the
/// `angleSet`/`poseLibrary` raw-settings stamping, and the identity-likeness gate. Built once by
/// [`plan_edit_batch`] and consumed by all three lanes so a change to the plain-With-Character gate
/// (sc-4411) lands in exactly one place.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct EditBatch {
    /// Per-iteration seed (shared across an angle/pose set, per-image on the plain path).
    seeds: Vec<i64>,
    /// Per-iteration prompt (angle/pose augment on the set paths, the base prompt on plain).
    prompts: Vec<String>,
    /// Full `PoseInput`s (keypoints + hands + face) for the pose tier; `None` for angles/plain.
    pose_inputs: Option<Vec<PoseInput>>,
    /// The lane's base raw settings with `angleSet`/`poseLibrary` stamped per grouping.
    raw_settings: JsonObject,
    /// Whether this generation is identity-likeness scored (epic 4406): a Character-Studio angle
    /// set, a pose-library set, OR a plain With-Character `character_image` job (sc-4411). Drives
    /// the [`stage_likeness`] call each lane makes with `references[0]` as the scored source.
    score_likeness: bool,
}

/// Build the [`EditBatch`] for a native edit stream from its resolved `grouping` and its
/// lane-specific `raw_settings` (the per-lane `*_edit_raw_settings` output — the ONLY per-stream
/// input, so the grouping/stamping/gating logic stays identical across FLUX.2 / Qwen / SenseNova).
///
/// Grouping expansion (parity with the `Mlx*Adapter` decision):
/// - `Poses(n)`: shared seed, one `augment_prompt_for_pose` prompt per pose, full `PoseInput`s kept
///   so whole-body poses thread hand/face articulation into the skeleton (sc-6702 / sc-6599).
/// - `Angles`: shared seed so noise-derived attributes (hair, lighting) stay constant across angles
///   — only the head pose changes (sc-2050 InstantID strategy) — with a per-angle prompt augment.
/// - `Plain`: `count` independent per-image seeds, the base prompt each.
///
/// Likeness gate (epic 4406, sc-4409 angles / sc-4410 poses / sc-4411 plain With-Character): the
/// generator-agnostic post-pass applies to EVERY character_image generation. `character_set` covers
/// the angle + pose sets; `plain_with_character` is a `character_image` job (so NOT an `edit_image`,
/// whose `Plain` grouping also lands here but carries a `sourceAssetId`, not an identity reference)
/// whose `Plain` grouping is the general subject-variation case — for that lane the scored reference
/// is `references[0]`, which for a `character_image` job IS the `referenceAssetId` (first in the
/// lane's `*_edit_reference_ids`). `score_likeness` covers all three.
#[cfg(target_os = "macos")]
fn plan_edit_batch(
    request: &ImageRequest,
    grouping: &EditGrouping,
    mut raw_settings: JsonObject,
) -> EditBatch {
    let set_seed = resolve_seed(request, 0);
    let (seeds, prompts, pose_inputs): (Vec<i64>, Vec<String>, Option<Vec<PoseInput>>) =
        match grouping {
            EditGrouping::Poses(count) => {
                let poses = parse_poses(request);
                let prompts = vec![augment_prompt_for_pose(&request.prompt); *count];
                (vec![set_seed; *count], prompts, Some(poses))
            }
            EditGrouping::Angles => {
                let prompts = CHARACTER_ANGLE_SET_ORDER
                    .iter()
                    .map(|angle| augment_prompt_for_angle(&request.prompt, angle))
                    .collect();
                (
                    vec![set_seed; CHARACTER_ANGLE_SET_ORDER.len()],
                    prompts,
                    None,
                )
            }
            EditGrouping::Plain => {
                let count = request.count as usize;
                let seeds = (0..count).map(|index| resolve_seed(request, index)).collect();
                (seeds, vec![request.prompt.clone(); count], None)
            }
        };

    match grouping {
        EditGrouping::Angles => {
            raw_settings.insert("angleSet".to_owned(), Value::Bool(true));
        }
        EditGrouping::Poses(_) => {
            raw_settings.insert("poseLibrary".to_owned(), Value::Bool(true));
        }
        EditGrouping::Plain => {}
    }

    let character_set = matches!(grouping, EditGrouping::Angles | EditGrouping::Poses(_));
    let plain_with_character =
        matches!(grouping, EditGrouping::Plain) && request.mode == "character_image";
    let score_likeness = character_set || plain_with_character;

    EditBatch {
        seeds,
        prompts,
        pose_inputs,
        raw_settings,
        score_likeness,
    }
}

/// True when an Image-Edit *source* (`sourceAssetId`) should be pre-fitted to W×H: `edit_image`
/// mode, a source asset, no character `referenceAssetId`, and a non-`stretch` fit mode. Used by the
/// img2img-init edit resolvers (`zimage`/`kolors` `resolve_*_edit_init`) that fit only the edit
/// source. The character-reference / multi-reference edit path is fitted by [`fit_edit_references`]
/// instead — which, unlike this gate, does NOT exclude the character reference (sc-8253).
fn should_fit_edit_source(request: &ImageRequest) -> bool {
    let has_source = request
        .source_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    // No character referenceAssetId (absent or empty).
    let no_reference = !request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    request.mode == "edit_image" && has_source && no_reference && request.fit_mode != "stretch"
}

/// Pre-fit every resolved edit reference — the character `referenceAssetId`, the multi-image
/// `referenceAssetIds`, or the Image-Edit `sourceAssetId` alike — to the conditioning `width`×
/// `height` before it reaches the engine, unless the fit mode is `stretch` (which keeps the legacy
/// non-aspect resize). Mirrors the candle edit lane (`load_flux2_edit_references`), which fits every
/// reference the same way.
///
/// Unlike [`should_fit_edit_source`] — the edit-*source* gate that deliberately excludes character
/// references — this fits the character reference too. sc-8253: a non-square character reference was
/// previously left at native aspect and squished into the square (e.g. 1024²) latent by the
/// klein/qwen/sensenova edit engines, distorting face geometry and losing identity (measured on
/// klein: a 2400×1744 landscape reference → ArcFace 0.47 vs a square crop of the same photo → 0.60,
/// ≈ −0.13). Crop/pad-fitting the reference to the output aspect (cover+center-crop for `crop`,
/// letterbox for `pad`/`outpaint`) lets the engine's 1:1 latent mapping preserve the face.
fn fit_edit_references(
    references: Vec<Image>,
    request: &ImageRequest,
    width: u32,
    height: u32,
) -> WorkerResult<Vec<Image>> {
    if request.fit_mode == "stretch" {
        return Ok(references);
    }
    references
        .into_iter()
        .map(|reference| fit_engine_image(reference, width, height, &request.fit_mode))
        .collect()
}

// `contain_box` / `fit_rgb` / `fit_engine_image` moved to image_jobs/base.rs (sc-6231) so they
// resolve on the candle lane too (video_jobs.rs + the candle edit handlers call `fit_engine_image`).

// ---------------------------------------------------------------------------
// FLUX.2-klein edit / reference (macOS, sc-3029): the `flux2_klein_9b_edit` and
// `flux2_klein_9b_kv_edit` variants. FLUX.2-klein is MLX-only (no torch), so this
// is where its edit/reference jobs run. One output per requested count, each
// conditioned on the shared reference image(s); the -kv variant auto-engages the
// reference-K/V cache (~2.4× edit speedup).
// ---------------------------------------------------------------------------

/// The engine edit-variant id for a FLUX.2 SceneWorks model, or `None` if the model
/// has no edit variant. The base 9b + true_v2 share `flux2_klein_9b_edit`; the -kv
/// distill uses `flux2_klein_9b_kv_edit` (reference-K/V cache); dev uses the
/// `flux2_dev_edit` variant (sc-5919) — the same dev snapshot, edit conditioning via
/// the DiT token concat (Reference / MultiReference), embedded guidance, no -kv cache.
fn flux2_edit_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "flux2_klein_9b" | "flux2_klein_9b_true_v2" => Some("flux2_klein_9b_edit"),
        "flux2_klein_9b_kv" => Some("flux2_klein_9b_kv_edit"),
        "flux2_dev" => Some("flux2_dev_edit"),
        _ => None,
    }
}

// `MAX_EDIT_REFERENCES` / `edit_reference_ids` moved to base.rs (sc-8946, F-144): shared by the
// FLUX.2 / SenseNova-via-grouping edit lanes, so they live with the other shared edit helpers.

/// True when this is a FLUX.2 edit job (a flux2 edit-capable model + ≥1 reference)
/// whose base weights resolve — routed to the edit variant rather than txt2img.
fn flux2_edit_available(request: &ImageRequest, settings: &Settings) -> bool {
    flux2_edit_engine_id(&request.model).is_some()
        && !edit_reference_ids(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// One `Reference` (single) or one `MultiReference` (N) edit conditioning from the
/// resolved reference images (cloned per output).
fn build_edit_conditioning(references: &[Image]) -> Vec<Conditioning> {
    if references.len() == 1 {
        vec![Conditioning::Reference {
            image: references[0].clone(),
            strength: None,
        }]
    } else {
        vec![Conditioning::MultiReference {
            images: references.to_vec(),
        }]
    }
}

/// Realism-safe default image-guidance scale for the klein/dev edit identity lever (sc-8273 A/B:
/// ≥2.0 over-smooths skin / "clay"; 1.5 holds identity with natural texture).
const DEFAULT_EDIT_IMAGE_GUIDANCE: f32 = 1.5;

/// Image-guidance scale (the "Identity strength" lever) for the FLUX.2 klein/dev EDIT path
/// (sc-8278), threaded onto `GenerationRequest.image_guidance`. The engine extrapolates the
/// reference condition (`v = v_img0 + s·(v_ref − v_img0)`) so a strong scene prompt no longer
/// drowns the reference identity (sc-8234: ArcFace 0.60 → 0.38 without it). Reads the shared
/// identity-strength slider value `advanced.ipAdapterScale` (the "Identity strength" knob the web
/// already sends for character refs; klein's catalog entry sets its range to ~1.0–2.5 / default
/// 1.5). `≤1.0` = off. Defaults to the realism-safe validated value 1.5 (sc-8273) when a character
/// reference is present and the knob is unspecified; `None` outside `character_image` mode or with
/// no reference (off = byte-identical render).
fn flux2_edit_image_guidance(request: &ImageRequest) -> Option<f32> {
    if request.mode != "character_image" {
        return None;
    }
    let has_reference = request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty())
        || !request.reference_asset_ids.is_empty();
    if !has_reference {
        return None;
    }
    let scale = request
        .advanced
        .get("ipAdapterScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(DEFAULT_EDIT_IMAGE_GUIDANCE)
        .clamp(1.0, 2.5);
    (scale > 1.0).then_some(scale)
}

/// Estimated peak unified-memory footprint (GB) of a FLUX.2-dev edit at `width`×`height` with
/// `reference_count` reference images — the input to the multi-reference memory guard. The dev edit is
/// activation-bound: the DiT attends over the target latent plus every reference latent, each
/// ≈⌈W/16⌉·⌈H/16⌉ tokens (VAE ×8, patch ×2), so the peak scales with the total sequence length.
/// Re-anchored for sc-6211 on the **chunked** worker-layer measurements (Q4 packed, `/usr/bin/time
/// -l` peak on a 128 GB Mac, with the sc-6266 engine sequence-gated activation chunking ON): a
/// two-reference 1024² edit ~81 GB and a four-reference 1024² edit ~93 GB — a linear-in-tokens fit
/// over those two chunked edit points (`BASE + PER_TOKEN·(1 + refs)·tokens_per_image`). The chunked
/// slope (~0.0015 GB/token) is ~3.8× gentler than the pre-chunking sc-5923 fit (0.005615), which is
/// why the two-reference edit now fits under 96. Only used on the multi-reference branch
/// (`reference_count >= 2`); txt2img and single-reference are covered directly by the declared
/// `minMemoryGb`.
fn flux2_dev_edit_peak_gb(reference_count: usize, width: u32, height: u32) -> f64 {
    const BASE_GB: f64 = 62.9;
    const PER_TOKEN_GB: f64 = 0.001_489; // (93 − 81) GB / (20480 − 12288) tokens, sc-6211 (chunked).
    let tokens_per_image = (f64::from(width) / 16.0).ceil() * (f64::from(height) / 16.0).ceil();
    let total_tokens = (1.0 + reference_count as f64) * tokens_per_image;
    BASE_GB + PER_TOKEN_GB * total_tokens
}

/// Prevent a silent OOM on a FLUX.2-dev **multi-reference** edit. With the sc-6266 engine activation
/// chunking a two-reference 1024² edit now peaks ~81 GB (sc-6211) and fits the declared `minMemoryGb`
/// (96); but the edit stays activation-bound, so more references / higher resolution still grow the
/// footprint (three-reference 1024² ~87 GB, four ~93 GB, five+ over 96). When the estimated peak plus
/// a fixed runtime/OS headroom exceeds the machine's unified memory, reject with an actionable message
/// instead of being SIGKILL'd mid-render. `reference_count < 2` and a failed RAM probe (`available_gb
/// == None`) short-circuit to `Ok`, so the guard never touches txt2img / single-reference and never
/// blocks a machine that can actually fit the edit. (Pre-sc-6266 this rejected the two-reference edit
/// outright on a 96 GB Mac — the ~104 GB un-chunked peak; the re-anchored estimate now passes it.)
fn flux2_dev_edit_memory_guard(
    reference_count: usize,
    width: u32,
    height: u32,
    available_gb: Option<f64>,
) -> WorkerResult<()> {
    if reference_count < 2 {
        return Ok(());
    }
    let Some(available_gb) = available_gb else {
        return Ok(());
    };
    // Headroom for the OS + other apps + MLX Metal transient allocations on top of the (accurate,
    // chunked) estimate. 12 GB passes the canonical two-reference 1024² edit (~81 GB) on a 96 GB Mac
    // while rejecting the genuinely-too-tight three+/high-resolution combinations.
    const HEADROOM_GB: f64 = 12.0;
    let needed_gb = flux2_dev_edit_peak_gb(reference_count, width, height);
    if available_gb + f64::EPSILON < needed_gb + HEADROOM_GB {
        return Err(WorkerError::InvalidPayload(format!(
            "FLUX.2-dev multi-reference edit at {width}×{height} with {reference_count} reference \
             images needs ~{needed} GB of unified memory (with headroom) but this machine has \
             ~{available} GB. Lower the output resolution, use a single reference image, or run on \
             a Mac with more memory.",
            needed = needed_gb.round() as i64,
            available = available_gb.round() as i64,
        )));
    }
    Ok(())
}

/// Generate one FLUX.2 edit image conditioned on `conditioning` (the reference set).
/// Distilled klein: guidance 1.0, no negative prompt.
#[allow(clippy::too_many_arguments)]
fn flux2_edit_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    image_guidance: Option<f32>,
    conditioning: Vec<Conditioning>,
    enhance: &PromptEnhance,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let mut request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        image_guidance,
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    enhance.apply(&mut request);
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::Engine(format!("edit generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("edit generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "edit generator returned non-image output".to_owned(),
        )),
    }
}

fn flux2_edit_raw_settings(
    request: &ImageRequest,
    repo: &str,
    engine_id: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    reference_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert(
        "guidanceScale".to_owned(),
        guidance.map(|value| json!(value)).unwrap_or(Value::Null),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("editEngine".to_owned(), Value::String(engine_id.to_owned()));
    raw.insert("referenceCount".to_owned(), json!(reference_count));
    raw
}

/// Real FLUX.2 edit generation: load the edit variant once, then `count` outputs each
/// conditioned on the shared reference set. Mirrors [`generate_stream`]'s blocking-
/// thread + streamed-events shape and reuses [`consume_gen_events`].
async fn generate_flux2_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let model = mlx_model(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not an MLX-backed model".to_owned()))?;
    let engine_id = flux2_edit_engine_id(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not a FLUX.2 edit model".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("FLUX.2 weights not found".to_owned()))?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    // Identity strength (sc-8278): map the UI `referenceStrength` slider onto the engine's
    // image-guidance CFG so a strong prompt doesn't drop the reference identity (sc-8234).
    let image_guidance = flux2_edit_image_guidance(request);
    let adapters = resolve_adapters(request, settings)?;
    let repo = model_repo(request, &model);
    let adapter_label = model.adapter_label();

    // Resolve the reference image(s) on the async side (decode → Send Image moved in).
    let reference_ids = edit_reference_ids(request);
    let mut references = Vec::with_capacity(reference_ids.len());
    for id in &reference_ids {
        references.push(load_reference_image(
            &settings.data_dir,
            &request.project_id,
            id,
            project_path,
        )?);
    }
    if references.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "FLUX.2 edit requires a reference image".to_owned(),
        ));
    }
    // sc-3030 / sc-8253 fit_image: pre-fit every reference (the Image-Edit source AND the
    // Character-Studio character reference) to the output W×H (crop / pad / outpaint→pad) so an
    // off-aspect reference isn't squished into the square latent. `stretch` keeps the legacy resize.
    references = fit_edit_references(references, request, request.width, request.height)?;

    // sc-6124: guard the activation-bound FLUX.2-dev *multi-reference* edit against a silent OOM on
    // machines below its real requirement. The reachable surface (txt2img + single-reference edit)
    // fits the declared `minMemoryGb` (96); a second reference adds ~4096 latent tokens to the DiT
    // stream and pushes the 1024² peak to ~104 GB (sc-5923), over the floor. Single-reference and
    // txt2img short-circuit, so this is inert until a multi-image edit picker feeds ≥2 references.
    if engine_id == "flux2_dev_edit" {
        flux2_dev_edit_memory_guard(
            references.len(),
            request.width,
            request.height,
            crate::gpu::total_unified_memory_gb().await,
        )?;
    }

    // sc-3030 per-iteration grouping: a Character-Studio angle set (11 shared-seed,
    // per-angle prompt) / best-effort pose tier (one per pose, shared seed, each a
    // `[skeleton, reference]` set) / else the plain per-image reference path. The grouping
    // expansion, angleSet/poseLibrary stamping, and the identity-likeness gate (incl. the sc-4411
    // plain-With-Character case) are the shared `plan_edit_batch` builder (F-024 sc-8826). The
    // whole-body pose PoseInputs thread their hand/face articulation into the skeleton below
    // (sc-6702). Identity-likeness scoring (epic 4406, sc-4409 angles / sc-4410 poses / sc-4411
    // plain With-Character) applies to EVERY character_image generation on this FLUX.2 edit lane,
    // which produces the FINAL image directly (no face-restore pass), so scoring the generated
    // image scores what the user sees.
    let grouping = edit_grouping(request);
    let EditBatch {
        seeds,
        prompts,
        pose_inputs,
        raw_settings,
        score_likeness,
    } = plan_edit_batch(
        request,
        &grouping,
        flux2_edit_raw_settings(
            request,
            &repo,
            engine_id,
            steps,
            quant_bits,
            guidance,
            references.len(),
        ),
    );
    let total = seeds.len();

    // Stage the antelopev2 face stack (same bundle InstantID uses; a no-op if already cached) and
    // capture the source identity reference + its asset id; the `!Send` scorer is built ONCE inside
    // the generator-worker closure below and reused across all outputs. Staging is non-fatal (a
    // failure → no scorer → scores omitted, the generation still renders). The scored reference is
    // `references[0]` — for a character_image job that IS the `referenceAssetId` (first in
    // `edit_reference_ids`).
    let face_stack_dir = stage_likeness(
        api,
        settings,
        job,
        score_likeness,
        "character_image face-stack staging failed; likeness scores omitted",
    )
    .await;
    let likeness_source = (score_likeness && face_stack_dir.is_some()).then(|| references[0].clone());
    let likeness_source_ref = reference_ids.first().cloned();

    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let adapter_count = adapters.len();
    // sc-6135: FLUX.2-dev caption upsampling — image-conditioned on the reference for the edit path.
    // Gated to dev by the engine + the manifest `ui.promptEnhance` toggle; off for klein.
    let enhance = PromptEnhance::from_advanced(&request.advanced);
    let spec = load_spec(weights_dir, quant, adapters, None);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("{engine_id} load failed"),
        move |generator, tx, cancel| {
            // Build the per-job identity-likeness scorer ONCE here (on the generator-worker thread
            // where the `!Send` face stack is allowed), embedding the source identity face a single
            // time and reusing it across every angle / pose (sc-4409/sc-4410 caching AC). `None` ⇒ not
            // a character set, or non-fatal construction failure ⇒ scores omitted; the set still renders.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some(source)) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            drive_gen_items_scored(
                tx,
                seeds.into_iter().zip(prompts),
                move |index, (seed, prompt), on_progress| {
                    // Pose tier: pair this pose's DWPose whole-body skeleton (body + hands
                    // 21x2 + face 68 when the pose carries them — sc-6702) with the reference
                    // as a `[skeleton, reference]` multi-image set; else the plain reference
                    // set. A real-weight A/B confirmed the hand/gesture detail transfers
                    // (wave → open palm vs fist; point → raised finger vs loose hand) with
                    // identity intact; body-only poses render identically to the old path
                    // (draw_wholebody with no hands/face == draw_bodypose).
                    let conditioning = match &pose_inputs {
                        Some(poses) => {
                            let pose = &poses[index];
                            let skeleton = crate::openpose_skeleton::draw_wholebody(
                                width,
                                height,
                                &pose.keypoints,
                                pose.hands.as_deref(),
                                pose.face.as_deref(),
                                stickwidth,
                            );
                            vec![Conditioning::MultiReference {
                                images: vec![
                                    Image {
                                        width,
                                        height,
                                        pixels: skeleton.into_raw(),
                                    },
                                    references[0].clone(),
                                ],
                            }]
                        }
                        None => build_edit_conditioning(&references),
                    };
                    let (out_w, out_h, pixels) = flux2_edit_generate_one(
                        generator,
                        &prompt,
                        width,
                        height,
                        seed,
                        steps,
                        guidance,
                        image_guidance,
                        conditioning,
                        &enhance,
                        &cancel,
                        on_progress,
                    )?;
                    // Score this finished image against the cached source embedding (sc-4409 angles /
                    // sc-4410 poses / sc-4411 plain With-Character). The Image build + pixel clone is paid
                    // ONLY when a scorer exists (a character_image generation) — a plain `edit_image` job
                    // has no scorer, so this is a no-op with no clone. Profile / up / down / full-body →
                    // honest detected:false N/A; `None` scorer ⇒ field omitted.
                    let face_likeness = scorer.as_ref().and_then(|scorer| {
                        crate::face_likeness::score_generated_image(
                            Some(scorer),
                            &Image {
                                width: out_w,
                                height: out_h,
                                pixels: pixels.clone(),
                            },
                            likeness_source_ref.as_deref(),
                        )
                    });
                    Ok(Some((seed, out_w, out_h, pixels, face_likeness)))
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
        adapter_label,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// FLUX.2-dev strict-pose Fun-Controlnet-Union (macOS, sc-6055 / engine sc-2292):
// the `flux2_dev_control` registry generator — a VACE ControlNet on the dev base.
// One image per library pose, each conditioned on a DWPose skeleton fed to the
// `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` branch (TRUE pose lock, not the
// best-effort `[skeleton, reference]` tier above). FLUX.2 is MLX-only, so this is
// the only strict-pose path for dev (no candle sibling). Mirrors the Z-Image MLX
// control path (`generate_zimage_control_stream`).
// ---------------------------------------------------------------------------

/// The engine registry id for the FLUX.2-dev Fun-Controlnet-Union variant (sc-2292).
const FLUX2_DEV_CONTROL_ENGINE_ID: &str = "flux2_dev_control";
/// Default Fun-Controlnet-Union control-weights filename — the `-2602` CFG-distilled variant (the
/// recommended one — the previous version lost CFG distillation after control training). The default
/// *repo* is the shared strict-control table (single source of truth — `STRICT_CONTROL_ENGINES`).
const FLUX2_CONTROL_FILE: &str = "FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors";
/// Pinned revision for the default `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` control-weights repo
/// (sc-9879, F-077 follow-up). Fetching the mutable `main` branch means a re-push (or a compromised token)
/// could silently swap the ControlNet checkpoint we load; pin the exact commit for defense-in-depth
/// (mirrors sc-8879/sc-9682). Applied ONLY to the default table repo — a manifest `controlWeights.repo`
/// override carries its own revision layout, so it keeps `main`. HF's tree API still reports the file's
/// `lfs.oid`, which `ensure_hf_cached_file` verifies the downloaded content against.
const FLUX2_CONTROL_REVISION: &str = "b3dcd7836a0e926248dac3ccba8fc0853495764b";
/// The asset `adapter` id recorded on FLUX.2-dev strict-pose assets (the dev base MLX label).
const FLUX2_CONTROL_ADAPTER_LABEL: &str = "mlx_flux2";

/// True when this is a FLUX.2-dev strict-pose job (`flux2_dev` + ≥1 pose, not edit mode) whose base
/// weights resolve — routed to the Fun-Controlnet-Union control path rather than the best-effort edit
/// pose tier or plain txt2img. Gated to `flux2_dev` (klein has no control checkpoint). Control-weights
/// presence is NOT part of the gate: they are fetched on first use in the stream (a missing checkpoint
/// downloads, then errors loudly only on a real failure — never silently drops the poses).
fn flux2_dev_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "flux2_dev"
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// The (repo, filename) of the FLUX.2-dev control weights — `advanced.controlWeights.{repo,filename}`
/// overrides, else the `-2602` Fun-Controlnet-Union default (parity with the Z-Image resolver).
/// The payload filename must be a plain component (sc-8821 / F-019).
fn flux2_control_repo_file(request: &ImageRequest) -> WorkerResult<(String, String)> {
    let cw = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object);
    let pick = |key: &str, default: &str| {
        cw.and_then(|m| m.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or(default)
            .to_owned()
    };
    Ok((
        // Default repo from the shared strict-control table (single source of truth); the file stays
        // engine-specific.
        pick(
            "repo",
            strict_control_default_repo(FLUX2_DEV_CONTROL_ENGINE_ID),
        ),
        safe_weight_filename(
            &pick("filename", FLUX2_CONTROL_FILE),
            "advanced.controlWeights.filename",
        )?,
    ))
}

/// Resolve the Fun-Controlnet-Union checkpoint the engine loads, downloading on first use. Order: an
/// env-pinned file (`SCENEWORKS_CONTROLNET_FLUX2`) → a whole-repo HF cache snapshot → download into the
/// app cache. Mirrors the candle `ensure_zimage_control_weights` / `ensure_kolors_control_weights`. The
/// 8.2 GB control checkpoint is lazy-fetched only on the first pose job (vs bloating the base download).
async fn ensure_flux2_control_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    let (repo, file) = flux2_control_repo_file(request)?;
    if let Ok(p) = std::env::var("SCENEWORKS_CONTROLNET_FLUX2") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, &repo) {
        let f = snapshot.join(&file);
        if f.is_file() {
            return Ok(f);
        }
    }
    let client = crate::downloads::streaming_download_client();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "FLUX.2-dev strict-pose generation canceled while fetching control weights.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-flux2")
        .join(&file);
    // Pin the exact commit for the default table control repo so `main` moving under us can't swap the
    // ControlNet checkpoint (sc-9879). A manifest `controlWeights.repo` override may carry its own
    // revision layout, so only pin when we're on the default repo.
    let revision = if repo == strict_control_default_repo(FLUX2_DEV_CONTROL_ENGINE_ID) {
        FLUX2_CONTROL_REVISION
    } else {
        "main"
    };
    crate::downloads::ensure_hf_cached_file(&context, &repo, revision, &file, &dst).await
}

/// Pose ControlNet lock strength for FLUX.2-dev: `advanced.controlScale` (default 0.75, clamp [0,2]).
/// The Fun-Controlnet-Union README recommends 0.65–0.80 for the dev branch, so the default sits at the
/// mid-point (Z-Image's strict-pose default is 0.9; the dev branch over-locks above ~0.8).
fn flux2_control_scale(request: &ImageRequest) -> f32 {
    advanced::f32_clamped(&request.advanced, "controlScale", 0.75, 0.0..=2.0)
}

/// Generate one FLUX.2-dev strict-pose image: the pre-built `conditioning` (the required `Control` plus an
/// optional identity `Reference`, assembled by the shared [`build_control_conditioning`] driver) drives the
/// Fun-Controlnet-Union branch. dev is guidance-distilled (embedded scalar) — `guidance` rides the
/// transformer's guidance embedder (no true-CFG).
#[allow(clippy::too_many_arguments)]
fn flux2_control_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    conditioning: Vec<Conditioning>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::Engine(format!("FLUX.2-dev control generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("FLUX.2-dev control generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "FLUX.2-dev control generator returned non-image output".to_owned(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn flux2_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert(
        "guidanceScale".to_owned(),
        guidance.map(|value| json!(value)).unwrap_or(Value::Null),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(FLUX2_DEV_CONTROL_ENGINE_ID.to_owned()),
    );
    raw
}

// sc-8946 (F-144): the FLUX.2-dev identity gate (`flux2_identity_strength`) and its init resolver
// (`resolve_identity_init`) were line-for-line copies of the Z-Image pair. Both now share the
// single [`identity_strength`] / [`resolve_identity_init`] in base.rs — the FLUX.2-dev control stream
// calls those directly.

/// Build the FLUX.2-dev control LoadSpec: the base dev snapshot + the Fun-Controlnet-Union overlay
/// (+ quant + adapters). The dev base loads manifest-aware (a pre-quantized Q4 snapshot loads packed);
/// the bf16 control overlay loads dense and quantizes in place under `with_quant`.
fn flux2_control_spec(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir))
        .with_control(WeightsSource::File(control_weights));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    spec
}

#[cfg(all(target_os = "macos", test))]
fn flux2_control_load(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> WorkerResult<Box<dyn Generator>> {
    let spec = flux2_control_spec(weights_dir, control_weights, quant, adapters);
    load_control_engine(FLUX2_DEV_CONTROL_ENGINE_ID, &spec)
}

/// Real FLUX.2-dev strict-pose generation: one image per pose, each conditioned on a DWPose skeleton
/// locked by the Fun-Controlnet-Union branch (sc-6055; engine sc-2292). Mirrors
/// [`generate_zimage_control_stream`] — the control checkpoint is fetched on first use, then the dev
/// control engine loads once on the blocking thread and renders one image per pose (shared seed so
/// only the pose changes across the set). dev keeps its embedded guidance (no CFG).
async fn generate_flux2_dev_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    // Optional identity img2img-init (opt-in, off by default — `referenceStrength`-gated), shared
    // across the pose set. `None` → the pose-only tier (the validated sc-2292 default).
    let identity_init = resolve_identity_init(request, settings, project_path)?;

    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("FLUX.2-dev weights not found".to_owned()))?;
    let control_weights = ensure_flux2_control_weights(api, settings, job, request).await?;
    let (quant, quant_bits) = resolve_quant(request);
    let model = mlx_model("flux2_dev")
        .ok_or_else(|| WorkerError::InvalidPayload("flux2_dev model row missing".to_owned()))?;
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    let control_scale = flux2_control_scale(request);
    let adapters = resolve_adapters(request, settings)?;
    let repo = model_repo(request, &model);
    // Shared strict-control driver: validate the requested ControlKind against the engine's
    // supported_kinds (flux2_dev_control = {Pose, Canny, Depth}) + resolve an optional user-supplied
    // control-map passthrough. The current (pose-only) job sets no `controlMode`, so `kind == Pose` and the
    // skeleton preprocessor runs exactly as before.
    let control_kind = requested_control_kind(request)?;
    validate_control_kind(FLUX2_DEV_CONTROL_ENGINE_ID, &control_kind)?;
    let user_control = resolve_user_control_map(request, settings, project_path)?;
    // sc-8248 source threading: for canny/depth WITHOUT a user-supplied control map, the control map is
    // auto-derived from the input image (canny edges / Depth-Anything-V2). The pose tier never needs a
    // source (the skeleton is synthetic).
    let control_source = resolve_control_source(request, settings, project_path)?;
    // Auto depth-estimator weights: provisioned only when this is a depth job WITHOUT a user-supplied
    // depth map (the passthrough short-circuits estimation). Shared across the set; fetched once on the
    // first depth job (sc-8242).
    let depth_weights_dir = if control_kind == ControlKind::Depth && user_control.is_none() {
        Some(ensure_depth_estimator_dir(api, settings, job).await?)
    } else {
        None
    };
    let poses = parse_poses(request);
    let count = poses.len();
    let raw_settings = flux2_control_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance,
        control_scale,
        count,
    );
    // Strict pose shares one seed across the set so noise-derived attributes (hair, wardrobe,
    // lighting) stay constant while only the pose changes (Z-Image parity).
    let seed = resolve_seed(request, 0);

    // Identity-likeness scoring (epic 4406, sc-4410): a FLUX.2-dev strict-control pose set is a
    // Character-Studio pose-library job; when it carries a character identity `referenceAssetId`, score
    // every finished pose against that source identity face through the SHARED seam. Source decode +
    // face-stack staging are non-fatal (missing reference / failure → no scorer → scores omitted, the
    // set still renders). The `!Send` scorer is built ONCE in the closure (source embedded once, reused
    // across all poses).
    let likeness_source = resolve_control_identity_source(request, settings, project_path);
    let face_stack_dir = stage_likeness(
        api,
        settings,
        job,
        likeness_source.is_some(),
        "pose-set face-stack staging failed; likeness scores omitted",
    )
    .await;

    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let adapter_count = adapters.len();
    let spec = flux2_control_spec(weights_dir, control_weights, quant, adapters);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        FLUX2_DEV_CONTROL_ENGINE_ID,
        adapter_count,
        spec,
        "FLUX.2-dev control load failed".to_owned(),
        move |generator, tx, cancel| {
            let identity_init = identity_init.as_ref();
            let user_control = user_control.as_ref();
            let control_source = control_source.as_ref();
            let depth_weights_dir = depth_weights_dir.as_deref();
            // Per-job identity-likeness scorer built ONCE; source embedded once, reused across every
            // pose (sc-4410). `None` ⇒ no identity reference / non-fatal construction failure.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some((source, _))) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            let likeness_source_ref = likeness_source.as_ref().map(|(_, id)| id.clone());
            drive_gen_items_scored(tx, poses, move |_index, pose, on_progress| {
                let control = preprocess_control_entry(
                    &control_kind,
                    user_control,
                    Some(&pose),
                    control_source,
                    width,
                    height,
                    stickwidth,
                    depth_weights_dir,
                )?;
                let conditioning = build_control_conditioning(
                    control,
                    control_kind.clone(),
                    control_scale,
                    identity_init,
                );
                let (out_w, out_h, pixels) = flux2_control_generate_one(
                    generator,
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    conditioning,
                    &cancel,
                    on_progress,
                )?;
                // Score this finished pose against the cached source embedding (sc-4410). The strict-
                // control lane produces the FINAL image directly (no face-restore pass), so this scores
                // what the user sees. Clone paid ONLY when a scorer exists; a full-body / turned pose
                // with no reliable frontal face → honest detected:false N/A.
                let face_likeness = scorer.as_ref().and_then(|scorer| {
                    crate::face_likeness::score_generated_image(
                        Some(scorer),
                        &Image {
                            width: out_w,
                            height: out_h,
                            pixels: pixels.clone(),
                        },
                        likeness_source_ref.as_deref(),
                    )
                });
                Ok(Some((seed, out_w, out_h, pixels, face_likeness)))
            })
        },
    );

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        FLUX2_CONTROL_ADAPTER_LABEL,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// Qwen-Image strict-pose ControlNet (macOS, epic 3401 / sc-3575): the InstantX
// `Qwen-Image-ControlNet-Union` variant registered in mlx-gen as `qwen_image_control`.
// One image per library pose, shared seed, true CFG + character LoRA on the base Qwen model.
// ---------------------------------------------------------------------------
