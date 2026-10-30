//! One owned-asset resolution for both native FLUX.3 generation lanes.
//! User payloads supply only opaque asset ids and optional output sides. The decoded carriers,
//! geometry, identity and resource facts below are worker-owned and never deserialized from JSON.

use std::path::Path;

use gen_core::flux3::references::ReferenceOutputSize;
use gen_core::{Conditioning, MemoryReferenceResourceFacts};
use gen_core_flux3::{layouts::LayoutCompile, references::ReferencePlan};
use sceneworks_core::image_request::ImageRequest;
use sceneworks_core::project_store::ProjectStore;
use sha2::{Digest, Sha256};

use crate::{WorkerError, WorkerResult};

pub(crate) const FACT_KEY: &str = "flux3ReferenceEditing";

/// The native lanes share this ingress: reserved provenance is never caller-authored.
pub(crate) fn caller_raw_settings(
    advanced: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut raw = advanced.clone();
    raw.remove(FACT_KEY);
    raw.remove("flux3Layout");
    raw
}

/// The shared image-event persistence seam independently scrubs legacy/replayed settings, then
/// admits only the current render's worker-authored facts. No-reference renders leave no record.
pub(crate) fn merge_image_facts(
    raw: &mut serde_json::Map<String, serde_json::Value>,
    trusted_facts: Option<serde_json::Map<String, serde_json::Value>>,
) {
    raw.remove(FACT_KEY);
    raw.remove("flux3Layout");
    if let Some(facts) = trusted_facts {
        raw.extend(facts);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OwnedReferenceBinding {
    /// Zero-based slot identity: an asset id alone cannot distinguish duplicate attachments.
    pub(crate) ordinal: usize,
    pub(crate) asset_id: String,
    pub(crate) oriented_size: (u32, u32),
    /// Hash of the actual decoded mode, oriented dimensions and original pixel carrier. Hidden
    /// RGB and alpha are retained, so same-size content replacement cannot reuse a stale binding.
    pub(crate) content_sha256: String,
}

pub(crate) struct ResolvedReferences {
    pub(crate) conditioning: Vec<Conditioning>,
    pub(crate) bindings: Vec<OwnedReferenceBinding>,
    pub(crate) output_size: ReferenceOutputSize,
    pub(crate) plan: ReferencePlan,
    pub(crate) resources: MemoryReferenceResourceFacts,
}

/// The actual reference-render seam shared by MLX and Candle. The provider sees exactly the owned
/// carriers and facts priced before its load; a missing, stale, or substituted context fails closed.
pub(crate) fn render_reference_request(
    generator: &dyn gen_core::Generator,
    request: &mut gen_core::GenerationRequest,
    references: &ResolvedReferences,
    context: Option<&gen_core::MemoryRunContext>,
    on_progress: &mut dyn FnMut(gen_core::Progress),
) -> gen_core::Result<gen_core::GenerationReport> {
    let context = context
        .filter(|context| {
            context.reference_resources.as_ref() == Some(&references.resources)
                && usize::try_from(context.geometry.reference_count).ok()
                    == Some(references.conditioning.len())
                && (context.geometry.width, context.geometry.height)
                    == references.plan.output.canvas
                && (request.width, request.height) == references.plan.output.canvas
        })
        .ok_or_else(|| {
            gen_core::Error::Unsupported(
                "FLUX.3 render must retain the same admitted reference resource facts and canvas"
                    .to_owned(),
            )
        })?;
    request.conditioning = references.conditioning.clone();
    request.flux3_reference_output = Some(references.output_size);
    crate::memory_strategy::generate_with_scope_report(
        generator,
        request,
        Some(context),
        on_progress,
    )
}

impl ResolvedReferences {
    /// Record the actual output, authored optional sides and the source-to-conditioning transforms.
    /// This is an output report, never a trusted input snapshot or a memory hint on replay.
    pub(crate) fn image_fact(&self, actual_size: (u32, u32)) -> serde_json::Value {
        serde_json::json!({
            "schemaVersion": 1,
            "coordinateSpaces": {
                "sourceSize": "exif_oriented_source_pixels",
                "fittedSize": "resized_source_pixels",
                "sourceRegion": "resized_source_pixels",
                "canvasRegion": "prepared_reference_pixels",
                "outputCrop": "output_canvas_pixels",
            },
            "rectangleConvention": "xyxy_exclusive",
            "references": self.bindings,
            "geometry": self.plan.geometry.iter().map(|geometry| serde_json::json!({
                "sourceSize": geometry.source_size,
                "fittedSize": geometry.fitted_size,
                "sourceRegion": geometry.source_region,
                "canvasRegion": geometry.canvas_region,
            })).collect::<Vec<_>>(),
            "preparedCanvases": self.resources.prepared_canvases,
            "conditioningTokens": self.resources.conditioning_tokens,
            "output": {
                "requestedWidth": self.output_size.width,
                "requestedHeight": self.output_size.height,
                "canvas": self.plan.output.canvas,
                "crop": self.plan.output.crop,
                "actualSize": actual_size,
            },
        })
    }
}

/// Reuse the pure family policy at the live worker boundary without letting tolerant parsing or
/// later request edits erase a malformed carrier. Numeric canvas normalization is legitimate;
/// the original optional sides and ordered attachment identities remain immutable.
pub(crate) fn validated_request(
    request: &ImageRequest,
) -> WorkerResult<Option<&sceneworks_core::image_request::flux3::ReferenceRequest>> {
    if !sceneworks_core::image_request::flux3::is_model(&request.model) {
        return Ok(None);
    }
    let authored = request
        .flux3_reference_request
        .as_ref()
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "FLUX.3 request is missing its validated reference intent".to_owned(),
            )
        })?
        .as_ref()
        .map_err(|error| WorkerError::InvalidPayload(error.clone()))?;
    let live = serde_json::json!({
        "model": request.model,
        "mode": request.mode,
        "sourceAssetId": request.source_asset_id,
        "referenceAssetIds": request.reference_asset_ids,
        "referenceAssetId": request.reference_asset_id,
        "maskAssetId": request.mask_asset_id,
        "width": authored.width,
        "height": authored.height,
        "advanced": request.advanced,
    });
    let live =
        sceneworks_core::image_request::flux3::parse(live.as_object().expect("object literal"))
            .map_err(WorkerError::InvalidPayload)?;
    if live.asset_ids != authored.asset_ids
        || live.layout != authored.layout
        || live.reference_ids != authored.reference_ids
    {
        return Err(WorkerError::InvalidPayload("FLUX.3 reference attachment slots changed after validation; rebuild the request from the authored payload".to_owned()));
    }
    Ok(Some(authored))
}

/// Lossless authored layout and the exact locked rows resolved before either native lane loads.
pub(crate) struct ResolvedLayout {
    layout: gen_core::flux3::layouts::Layout,
    bindings: Vec<gen_core::flux3::layouts::ReferenceBinding>,
    fact: serde_json::Value,
}

impl ResolvedLayout {
    pub(crate) fn apply(&self, request: &mut gen_core::GenerationRequest) {
        request.flux3_layout = Some(self.layout.clone());
        request.flux3_layout_bindings = self.bindings.clone();
    }
    pub(crate) fn image_fact(&self) -> serde_json::Value {
        self.fact.clone()
    }
}

pub(crate) fn resolve_owned_layout(
    request: &ImageRequest,
    references: Option<&ResolvedReferences>,
) -> WorkerResult<Option<ResolvedLayout>> {
    let Some(authored) = validated_request(request)? else {
        return Ok(None);
    };
    let Some(document) = &authored.layout else {
        return Ok(None);
    };
    let invalid = |error: gen_core::Error| WorkerError::InvalidPayload(error.to_string());
    let layout = gen_core::flux3::layouts::Layout::from_json(document.clone()).map_err(invalid)?;
    let bindings = references
        .map(|refs| {
            refs.bindings
                .iter()
                .map(|binding| {
                    gen_core::flux3::layouts::ReferenceBinding::oriented(
                        authored
                            .reference_ids
                            .as_ref()
                            .map(|ids| ids[binding.ordinal].clone())
                            .unwrap_or_else(|| format!("ref_image_{}", binding.ordinal)),
                        binding.oriented_size,
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let output = references.map(|refs| refs.plan.output.clone()).unwrap_or(
        gen_core_flux3::references::OutputGeometry {
            canvas: (request.width, request.height),
            crop: None,
        },
    );
    let compiled = layout
        .compile(&bindings, references.map(|r| &r.plan), &output)
        .map_err(invalid)?;
    compiled.caption(&request.prompt).map_err(invalid)?;
    let fact = serde_json::json!({
        "schemaVersion":1,
        "upstreamRevision":gen_core_flux3::UPSTREAM_CODE_REVISION,
        "semantics":"caption_only_soft_spatial_guidance",
        "coordinateConvention":"yxyx",
        "authoring":layout.to_json(),
        "lockedAnnotations":compiled.rows_json(),
        "referenceInstanceIds":authored.reference_ids,
        "references":references.map(|r| &r.bindings),
    });
    Ok(Some(ResolvedLayout {
        layout,
        bindings,
        fact,
    }))
}

/// Run before either lane's fit, admission, cache lookup or generator load. The same immutable
/// carriers and plan then travel into the render; no second path resolves or preprocesses assets.
pub(crate) fn resolve_owned_references(
    request: &ImageRequest,
    data_dir: &Path,
    project_path: &Path,
) -> WorkerResult<Option<ResolvedReferences>> {
    let Some(authored) = validated_request(request)? else {
        return Ok(None);
    };
    if authored.asset_ids.is_empty() {
        return Ok(None);
    }
    let store = ProjectStore::new(data_dir.to_path_buf(), "worker");
    let project_root = std::fs::canonicalize(project_path).map_err(|error| {
        WorkerError::InvalidPayload(format!("FLUX.3 reference project is unavailable: {error}"))
    })?;
    let mut conditioning = Vec::with_capacity(authored.asset_ids.len());
    let mut bindings = Vec::with_capacity(authored.asset_ids.len());
    let mut dimensions = Vec::with_capacity(authored.asset_ids.len());
    // The raw parser has already validated EVERY carrier/count/strength before any file is opened.
    for (ordinal, asset_id) in authored.asset_ids.iter().enumerate() {
        let invalid = |detail: String| {
            WorkerError::InvalidPayload(format!(
                "FLUX.3 reference {} ({asset_id}): {detail}",
                ordinal + 1
            ))
        };
        let asset = store
            .get_asset(&request.project_id, asset_id)
            .map_err(|error| invalid(format!("not an owned project asset: {error}")))?;
        let mime = asset
            .pointer("/file/mimeType")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if asset.get("type").and_then(serde_json::Value::as_str) != Some("image")
            || !mime.starts_with("image/")
            || mime == "image/svg+xml"
        {
            return Err(invalid("must name an owned raster image".to_owned()));
        }
        let path = store
            .resolve_asset_media_path(&request.project_id, asset_id)
            .map_err(|error| invalid(error.to_string()))?;
        let path = std::fs::canonicalize(path)
            .map_err(|error| invalid(format!("media is missing or unreadable: {error}")))?;
        if !path.starts_with(&project_root) || !path.is_file() {
            return Err(invalid(
                "media must be a file inside the owning project".to_owned(),
            ));
        }
        // Reuse the frozen EXIF/source-mode decoder. AE's RGB fork and PUP's independent white
        // composite remain native responsibilities; never flatten alpha or resize at this seam.
        let image = gen_core_flux3::references::decode_reference_file(&path)
            .map_err(|error| invalid(error.to_string()))?;
        let (kind, width, height, pixels) = match &image {
            Conditioning::Reference {
                image,
                strength: None,
            } => (0_u8, image.width, image.height, image.pixels.as_slice()),
            Conditioning::ReferenceRgba {
                image,
                strength: None,
            } => (1_u8, image.width, image.height, image.pixels.as_slice()),
            _ => {
                return Err(invalid(
                    "decoder returned an unsupported reference carrier".to_owned(),
                ))
            }
        };
        let mut digest = Sha256::new();
        digest.update([kind]);
        digest.update(width.to_le_bytes());
        digest.update(height.to_le_bytes());
        digest.update(pixels);
        bindings.push(OwnedReferenceBinding {
            ordinal,
            asset_id: asset_id.clone(),
            oriented_size: (width, height),
            content_sha256: format!("{:x}", digest.finalize()),
        });
        dimensions.push((width, height));
        conditioning.push(image);
    }
    let output_size = ReferenceOutputSize {
        width: authored.width,
        height: authored.height,
    };
    let plan = gen_core_flux3::references::plan(&dimensions, output_size)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    let resources = plan
        .memory_resources()
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    Ok(Some(ResolvedReferences {
        conditioning,
        bindings,
        output_size,
        plan,
        resources,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::project_store::UploadAsset;
    use serde_json::json;

    struct Fixture {
        data: tempfile::TempDir,
        project_id: String,
        project_path: std::path::PathBuf,
        id: String,
    }

    impl Fixture {
        fn new() -> Self {
            let data = tempfile::tempdir().unwrap();
            let store = ProjectStore::new(data.path().to_path_buf(), "worker");
            let project = store.create_project("FLUX3 ordered references").unwrap();
            let source = data.path().join("rgba.png");
            image::RgbaImage::from_pixel(83, 67, image::Rgba([201, 77, 13, 0]))
                .save(&source)
                .unwrap();
            let asset = store
                .import_asset(
                    &project.id,
                    UploadAsset {
                        filename: "rgba.png".to_owned(),
                        content_type: Some("image/png".to_owned()),
                        source_path: source,
                        source_asset_id: None,
                        provenance: None,
                    },
                )
                .unwrap();
            Self {
                data,
                project_id: project.id,
                project_path: project.path.into(),
                id: asset["id"].as_str().unwrap().to_owned(),
            }
        }

        fn request(&self, extra: serde_json::Value) -> ImageRequest {
            let mut payload = json!({"projectId":self.project_id,"model":"flux3_klein","mode":"edit_image","referenceAssetId":self.id}).as_object().unwrap().clone();
            payload.extend(extra.as_object().unwrap().clone());
            ImageRequest::from_payload(&payload)
        }

        fn resolve(&self, request: &ImageRequest) -> ResolvedReferences {
            resolve_owned_references(request, self.data.path(), &self.project_path)
                .unwrap()
                .unwrap()
        }
    }

    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[test]
    fn real_generic_render_surface_transports_placement_document_to_generator() {
        struct Capture(gen_core::ModelDescriptor);
        impl gen_core::Generator for Capture {
            fn descriptor(&self) -> &gen_core::ModelDescriptor {
                &self.0
            }
            fn validate(&self, _: &gen_core::GenerationRequest) -> gen_core::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                req: &gen_core::GenerationRequest,
                _: &mut dyn FnMut(gen_core::Progress),
            ) -> gen_core::Result<gen_core::GenerationOutput> {
                let input = gen_core_flux3::references::upsampler_input(req)?;
                let outcome = gen_core_flux3::pup::upsample(
                    None,
                    &input,
                    &Default::default(),
                    &req.cancel,
                    &mut |_| panic!("bypass"),
                )?;
                assert_eq!(outcome.caption,"scene [{\"id\": \"placed\", \"bbox\": [100, 200, 700, 800], \"desc\": \"cat\"}]");
                Ok(gen_core::GenerationOutput::Images(vec![gen_core::Image {
                    width: 64,
                    height: 64,
                    pixels: vec![31; 64 * 64 * 3],
                }]))
            }
        }
        let payload = json!({"model":"flux3_klein","prompt":"scene","width":64,"height":64,"advanced":{"flux3_layout":{"version":1,"boxes":[{"id":"placed","desc":"cat","bbox":[0.1,0.2,0.7,0.8]}]}}});
        let request = ImageRequest::from_payload(payload.as_object().unwrap());
        let layout = resolve_owned_layout(&request, None).unwrap().unwrap();
        let generator = Capture(gen_core::ModelDescriptor {
            id: "flux3_klein",
            family: "flux3",
            backend: "candle",
            modality: gen_core::Modality::Image,
            capabilities: Default::default(),
            encoder_contract: None,
            required_components: &[],
            control_kinds: None,
            denoiser_output_latent_space: None,
        });
        let surface = super::super::LaneRgbaSurface {
            flux3_layout: Some(&layout),
            ..Default::default()
        };
        let images = super::super::render_on_surface(
            &generator,
            "scene",
            64,
            64,
            7,
            16,
            None,
            None,
            None,
            &[],
            None,
            None,
            None,
            None,
            None,
            None,
            false,
            None,
            None,
            None,
            &super::super::PromptEnhance::default(),
            Default::default(),
            Default::default(),
            &surface,
            &Default::default(),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].2, vec![31; 64 * 64 * 3]);
    }

    #[test]
    fn layout_payload_binds_duplicate_slots_and_reaches_native_caption_and_asset_recipe() {
        let fixture = Fixture::new();
        let document = json!({"version":1,"boxes":[{"id":"cat","desc":"remove","from":"ref_image_0","reference_id":"second","src_bbox":[0.123456789012345,0.2,0.7,0.8],"tgt_bbox":null}]});
        let request=fixture.request(json!({"referenceAssetId":null,"referenceAssetIds":[fixture.id,fixture.id],"advanced":{"flux3_layout":document,"flux3_reference_ids":["first","second"],"flux3Layout":{"forged":true}}}));
        let references = fixture.resolve(&request);
        let layout = resolve_owned_layout(&request, Some(&references))
            .unwrap()
            .unwrap();
        let mut native = gen_core::GenerationRequest {
            prompt: request.prompt.clone(),
            width: references.plan.output.canvas.0,
            height: references.plan.output.canvas.1,
            conditioning: references.conditioning.clone(),
            flux3_reference_output: Some(references.output_size),
            ..Default::default()
        };
        layout.apply(&mut native);
        let input = gen_core_flux3::references::upsampler_input(&native).unwrap();
        assert_eq!(
            input.boxes.as_ref().unwrap()[0].from.as_deref(),
            Some("ref_image_1")
        );
        let outcome = gen_core_flux3::pup::upsample(
            None,
            &input,
            &Default::default(),
            &native.cancel,
            &mut |_| panic!("bypass must not infer"),
        )
        .unwrap();
        assert!(outcome
            .caption
            .contains("\"src_bbox\": [123, 200, 700, 800]"));
        assert!(outcome.caption.contains("\"tgt_bbox\": null"));
        assert_eq!(native.flux3_layout.as_ref().unwrap().to_json(), &document);
        let (width, height) = references.plan.output.canvas;
        for adapter in ["mlx_flux3", "candle_flux3"] {
            let mut raw = caller_raw_settings(&request.advanced);
            assert!(!raw.contains_key("flux3Layout"));
            let mut facts = serde_json::Map::new();
            facts.insert(FACT_KEY.into(), references.image_fact((width, height)));
            facts.insert("flux3Layout".into(), layout.image_fact());
            merge_image_facts(&mut raw, Some(facts));
            let plan = super::super::ImagePlan::with_count_and_adapter(&request, 1, adapter, None);
            let fact = super::super::write_image_asset(
                &plan,
                0,
                77,
                width,
                height,
                vec![103; (width * height * 3) as usize],
                adapter,
                raw,
                &fixture.project_path,
            )
            .unwrap();
            let store = ProjectStore::new(fixture.data.path().to_path_buf(), "worker");
            store
                .persist_generated_asset(
                    &fixture.project_id,
                    "job-layout-roundtrip",
                    &plan.genset_id,
                    &serde_json::Value::Object(fact.clone()),
                )
                .unwrap();
            let path = fixture
                .project_path
                .join("recipes")
                .join(format!("{}.recipe.json", fact["assetId"].as_str().unwrap()));
            let recipe: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            assert_eq!(recipe["rawAdapterSettings"]["flux3_layout"], document);
            // The real worker persists ordered source identity in its trusted facts, NOT a
            // top-level request referenceAssetIds array. Replay must consume this producer shape.
            let raw = recipe["rawAdapterSettings"].as_object().unwrap();
            assert!(!raw.contains_key("referenceAssetIds"));
            assert_eq!(raw["flux3_reference_ids"], json!(["first", "second"]));
            assert_eq!(
                raw[FACT_KEY]["references"],
                serde_json::to_value(&references.bindings).unwrap()
            );
            assert_eq!(raw[FACT_KEY]["references"][0]["assetId"], fixture.id);
            assert_eq!(raw[FACT_KEY]["references"][1]["assetId"], fixture.id);
            assert_eq!(
                raw["flux3Layout"]["referenceInstanceIds"],
                raw["flux3_reference_ids"]
            );
            assert_eq!(
                recipe["rawAdapterSettings"]["flux3Layout"]["authoring"],
                document
            );
            assert_eq!(
                recipe["rawAdapterSettings"]["flux3Layout"]["references"][1]["ordinal"],
                1
            );
            assert_ne!(
                recipe["rawAdapterSettings"]["flux3Layout"]["references"][1]["contentSha256"],
                serde_json::Value::Null
            );
        }
        let mut tampered = request.clone();
        tampered
            .advanced
            .insert("flux3_reference_ids".into(), json!(["second", "first"]));
        assert!(resolve_owned_layout(&tampered, Some(&references)).is_err());
    }

    #[test]
    fn reference_asset_recipes_roundtrip_authored_options_and_actual_pixels_on_both_lanes() {
        let fixture = Fixture::new();
        let store = ProjectStore::new(fixture.data.path().to_path_buf(), "worker");
        for model in ["flux3_klein", "flux3_klein_base"] {
            for adapter in ["mlx_flux3", "candle_flux3"] {
                for (sides, actual) in [
                    (json!({}), (83, 67)),
                    (json!({"width":16}), (16, 80)),
                    (json!({"height":48}), (96, 48)),
                    (json!({"width":16,"height":48}), (16, 48)),
                ] {
                    let mut extra = sides.as_object().unwrap().clone();
                    extra.insert("model".to_owned(), json!(model));
                    extra.insert("modelManifestEntry".to_owned(), json!({"id":model,"family":"flux3","limits":{"minDimension":64,"maxDimension":4096,"requiresDimensionsMultipleOf":16}}));
                    extra.insert("advanced".to_owned(), json!({"flux3ReferenceEditing":{"references":[{"contentSha256":"forged"}],"output":{"requestedWidth":2048}}}));
                    let request = fixture.request(serde_json::Value::Object(extra));
                    let resolved = fixture.resolve(&request);
                    let genuine = resolved.image_fact(actual);
                    let mut raw = caller_raw_settings(&request.advanced);
                    assert!(
                        !raw.contains_key(FACT_KEY),
                        "{adapter}: caller provenance must be scrubbed at ingress"
                    );
                    // Independently protect the event boundary from legacy/replayed raw settings.
                    raw.insert(
                        FACT_KEY.to_owned(),
                        json!({"references":[{"contentSha256":"forged"}]}),
                    );
                    let mut facts = serde_json::Map::new();
                    facts.insert(FACT_KEY.to_owned(), genuine.clone());
                    merge_image_facts(&mut raw, Some(facts));
                    let plan =
                        super::super::ImagePlan::with_count_and_adapter(&request, 1, adapter, None);
                    let fact = super::super::write_image_asset(
                        &plan,
                        0,
                        71,
                        actual.0,
                        actual.1,
                        vec![137; (actual.0 * actual.1 * 3) as usize],
                        adapter,
                        raw,
                        &fixture.project_path,
                    )
                    .unwrap();
                    assert_eq!(
                        fact["normalizedWidth"], actual.0,
                        "{model}/{adapter}/{sides}"
                    );
                    assert_eq!(
                        fact["normalizedHeight"], actual.1,
                        "{model}/{adapter}/{sides}"
                    );
                    let fact_value = serde_json::Value::Object(fact.clone());
                    store
                        .persist_generated_asset(
                            &fixture.project_id,
                            "job-reference-roundtrip",
                            &plan.genset_id,
                            &fact_value,
                        )
                        .unwrap();
                    let asset_id = fact["assetId"].as_str().unwrap();
                    let asset = store.get_asset(&fixture.project_id, asset_id).unwrap();
                    assert_eq!(asset["file"]["width"], actual.0);
                    assert_eq!(asset["file"]["height"], actual.1);
                    assert_eq!(asset["recipe"]["normalizedSettings"]["width"], actual.0);
                    assert_eq!(asset["recipe"]["normalizedSettings"]["height"], actual.1);
                    let recorded = &asset["recipe"]["rawAdapterSettings"][FACT_KEY];
                    assert_eq!(recorded, &genuine);
                    assert_eq!(
                        recorded["output"]["requestedWidth"],
                        sides
                            .get("width")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null)
                    );
                    assert_eq!(
                        recorded["output"]["requestedHeight"],
                        sides
                            .get("height")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null)
                    );
                    assert_eq!(
                        recorded["output"]["actualSize"],
                        json!([actual.0, actual.1])
                    );
                    let png = image::open(
                        fixture
                            .project_path
                            .join(fact["mediaPath"].as_str().unwrap()),
                    )
                    .unwrap();
                    assert_eq!((png.width(), png.height()), actual);
                    let standalone: serde_json::Value = serde_json::from_slice(
                        &std::fs::read(
                            fixture
                                .project_path
                                .join("recipes")
                                .join(format!("{asset_id}.recipe.json")),
                        )
                        .unwrap(),
                    )
                    .unwrap();
                    assert_eq!(
                        standalone["normalizedSettings"],
                        asset["recipe"]["normalizedSettings"]
                    );
                    assert_eq!(standalone["rawAdapterSettings"][FACT_KEY], genuine);
                }
            }
        }
    }

    #[test]
    fn forged_reference_provenance_cannot_reach_a_plain_image_asset_recipe() {
        let fixture = Fixture::new();
        let store = ProjectStore::new(fixture.data.path().to_path_buf(), "worker");
        for model in ["flux3_klein", "flux3_klein_base"] {
            for adapter in ["mlx_flux3", "candle_flux3"] {
                let payload = json!({"projectId":fixture.project_id,"model":model,"mode":"text_to_image","prompt":"plain","count":1,"width":64,"height":64,
                    "advanced":{"flux3ReferenceEditing":{"references":[{"contentSha256":"forged"}],"geometry":[[2048,2048]]},"sampler":"euler"}});
                let request = ImageRequest::from_payload(payload.as_object().unwrap());
                assert!(resolve_owned_references(
                    &request,
                    fixture.data.path(),
                    &fixture.project_path
                )
                .unwrap()
                .is_none());
                let ingress = caller_raw_settings(&request.advanced);
                assert!(!ingress.contains_key(FACT_KEY));
                assert_eq!(ingress["sampler"], "euler");
                let mut legacy_raw = request.advanced.clone();
                merge_image_facts(&mut legacy_raw, None);
                let plan =
                    super::super::ImagePlan::with_count_and_adapter(&request, 1, adapter, None);
                let fact = super::super::write_image_asset(
                    &plan,
                    0,
                    72,
                    64,
                    64,
                    vec![77; 64 * 64 * 3],
                    adapter,
                    legacy_raw,
                    &fixture.project_path,
                )
                .unwrap();
                store
                    .persist_generated_asset(
                        &fixture.project_id,
                        "job-plain-roundtrip",
                        &plan.genset_id,
                        &serde_json::Value::Object(fact.clone()),
                    )
                    .unwrap();
                let asset = store
                    .get_asset(&fixture.project_id, fact["assetId"].as_str().unwrap())
                    .unwrap();
                assert!(
                    asset["recipe"]["rawAdapterSettings"]
                        .get(FACT_KEY)
                        .is_none(),
                    "{model}/{adapter}: {asset}"
                );
            }
        }
    }

    #[test]
    fn owned_rgba_auto_intent_and_ordered_memory_facts_are_actual_not_metadata_hints() {
        let fixture = Fixture::new();
        for model in ["flux3_klein", "flux3_klein_base"] {
            let one = fixture.resolve(&fixture.request(json!({"model":model,"advanced":{"conditioningTokens":1,"referenceCanvases":[[16,16]]}})));
            assert_eq!(one.plan.output.canvas, (96, 80));
            assert_eq!(one.plan.output.crop, Some([6, 6, 89, 73]));
            assert_eq!(one.resources.conditioning_tokens, 30);
            assert_eq!(one.resources.prepared_canvases, [(96, 80)]);
            let fact = one.image_fact((83, 67));
            assert_eq!(fact["output"]["actualSize"], json!([83, 67]));
            assert_eq!(fact["output"]["canvas"], json!([96, 80]));
            assert_eq!(fact["output"]["crop"], json!([6, 6, 89, 73]));
            assert!(fact["output"]["requestedWidth"].is_null());
            assert!(fact["output"]["requestedHeight"].is_null());
            assert_eq!(fact["references"][0]["assetId"], fixture.id);
            assert_eq!(fact["geometry"][0]["sourceRegion"], json!([0, 0, 83, 67]));
            assert_eq!(
                (one.output_size.width, one.output_size.height),
                (None, None)
            );
            match &one.conditioning[0] {
                Conditioning::ReferenceRgba {
                    image,
                    strength: None,
                } => {
                    assert_eq!((image.width, image.height), (83, 67));
                    assert_eq!(&image.pixels[..4], &[201, 77, 13, 0]);
                }
                other => panic!("original RGBA carrier was flattened: {other:?}"),
            }
            let ten = fixture
                .resolve(&fixture.request(
                    json!({"model":model,"referenceAssetIds":vec![fixture.id.clone();9]}),
                ));
            assert_eq!(ten.conditioning.len(), 10);
            assert_eq!(ten.resources.prepared_canvases, vec![(80, 64); 10]);
            assert_eq!(ten.resources.conditioning_tokens, 200);
            assert_eq!(ten.plan.output.canvas, (80, 64));
            let fact = ten.image_fact((80, 64));
            assert_eq!(fact["references"].as_array().unwrap().len(), 10);
            assert_eq!(fact["references"][9]["ordinal"], 9);
            assert_eq!(fact["references"][9]["assetId"], fixture.id);
            assert_eq!(fact["geometry"][0]["sourceRegion"], json!([1, 1, 81, 65]));
            assert_eq!(fact["geometry"][0]["canvasRegion"], json!([0, 0, 80, 64]));
            for (ordinal, binding) in ten.bindings.iter().enumerate() {
                assert_eq!(binding.ordinal, ordinal);
                assert_eq!(binding.asset_id, fixture.id);
                assert_eq!(binding.oriented_size, (83, 67));
            }
            let explicit = fixture.resolve(&fixture.request(json!({"model":model,"width":1024})));
            assert_eq!(explicit.plan.output.canvas, (1024, 80));
            assert_eq!(explicit.plan.output.crop, None);
            assert_eq!(
                (explicit.output_size.width, explicit.output_size.height),
                (Some(1024), None)
            );
        }
    }

    #[test]
    fn actual_exif_oriented_geometry_precedes_admission_instead_of_recorded_asset_dimensions() {
        use image::ImageEncoder;
        let fixture = Fixture::new();
        let store = ProjectStore::new(fixture.data.path().to_path_buf(), "worker");
        let path = store
            .resolve_asset_media_path(&fixture.project_id, &fixture.id)
            .unwrap();
        let mut encoder = image::codecs::png::PngEncoder::new(std::fs::File::create(path).unwrap());
        // One little-endian TIFF IFD: Orientation=6 (clockwise 90 degrees). The indexed asset
        // still records the unrotated 83x67 upload; only decoding the owned file sees 67x83.
        encoder
            .set_exif_metadata(vec![
                0x49, 0x49, 0x2a, 0, 8, 0, 0, 0, 1, 0, 0x12, 1, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0,
                0, 0,
            ])
            .unwrap();
        let pixels =
            image::RgbaImage::from_pixel(83, 67, image::Rgba([201, 77, 13, 128])).into_raw();
        encoder
            .write_image(&pixels, 83, 67, image::ExtendedColorType::Rgba8)
            .unwrap();
        let resolved = fixture.resolve(&fixture.request(json!({})));
        assert_eq!(resolved.bindings[0].oriented_size, (67, 83));
        assert_eq!(resolved.plan.output.canvas, (80, 96));
        assert_eq!(resolved.resources.prepared_canvases, [(80, 96)]);
        assert_eq!(resolved.plan.output.crop, Some([6, 6, 73, 89]));
        let fact = resolved.image_fact((67, 83));
        assert_eq!(fact["output"]["actualSize"], json!([67, 83]));
        assert_eq!(fact["references"][0]["orientedSize"], json!([67, 83]));
    }

    #[test]
    fn live_reference_reordering_cannot_reuse_a_stale_validated_snapshot() {
        let payload = serde_json::json!({"model":"flux3_klein","mode":"edit_image","referenceAssetIds":["a","b"]});
        let mut request = ImageRequest::from_payload(payload.as_object().unwrap());
        request.reference_asset_ids.reverse();
        let absent = Path::new("this-project-does-not-exist");
        let error = resolve_owned_references(&request, absent, absent)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("attachment slots changed"), "{error}");
        let mut request = ImageRequest::from_payload(payload.as_object().unwrap());
        request
            .advanced
            .insert("referenceStrength".to_owned(), serde_json::json!(1.0));
        assert!(validated_request(&request)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));
    }

    #[test]
    fn invalid_count_and_raw_carriers_fail_before_any_project_or_media_read() {
        let nonexistent = Path::new("this-project-does-not-exist");
        for extra in [
            json!({"referenceAssetIds":vec!["x";11]}),
            json!({"referenceAssetIds":["x",null]}),
            json!({"referenceAssetId":"x","advanced":{"strength":1.0}}),
        ] {
            let mut payload = json!({"model":"flux3_klein","mode":"edit_image"})
                .as_object()
                .unwrap()
                .clone();
            payload.extend(extra.as_object().unwrap().clone());
            let error = resolve_owned_references(
                &ImageRequest::from_payload(&payload),
                nonexistent,
                nonexistent,
            )
            .err()
            .unwrap()
            .to_string();
            assert!(
                !error.contains("project is unavailable"),
                "raw validation ran after IO: {error}"
            );
        }
    }

    #[test]
    fn foreign_missing_corrupt_assets_fail_and_same_size_content_replacement_changes_identity() {
        let fixture = Fixture::new();
        let request = fixture.request(json!({}));
        let before = fixture.resolve(&request);
        let store = ProjectStore::new(fixture.data.path().to_path_buf(), "worker");
        let foreign = store.create_project("foreign").unwrap();
        let foreign_request = fixture.request(json!({"projectId":foreign.id}));
        assert!(resolve_owned_references(
            &foreign_request,
            fixture.data.path(),
            Path::new(&foreign.path)
        )
        .is_err());
        let path = store
            .resolve_asset_media_path(&fixture.project_id, &fixture.id)
            .unwrap();
        image::RgbaImage::from_pixel(83, 67, image::Rgba([201, 77, 13, 128]))
            .save(&path)
            .unwrap();
        let after = fixture.resolve(&request);
        assert_eq!(before.plan, after.plan);
        assert_ne!(
            before.bindings[0].content_sha256,
            after.bindings[0].content_sha256
        );
        std::fs::write(&path, b"not a valid raster image").unwrap();
        assert!(
            resolve_owned_references(&request, fixture.data.path(), &fixture.project_path).is_err()
        );
        std::fs::remove_file(path).unwrap();
        assert!(
            resolve_owned_references(&request, fixture.data.path(), &fixture.project_path).is_err()
        );
    }
}
