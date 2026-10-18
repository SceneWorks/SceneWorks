//! One owned-asset resolution for both native FLUX.3 generation lanes.
//! User payloads supply only opaque asset ids and optional output sides. The decoded carriers,
//! geometry, identity and resource facts below are worker-owned and never deserialized from JSON.

use std::path::Path;

use gen_core::flux3::references::{ReferenceOutputSize, ReferencePlan};
use gen_core::{Conditioning, MemoryReferenceResourceFacts};
use sceneworks_core::image_request::ImageRequest;
use sceneworks_core::project_store::ProjectStore;
use sha2::{Digest, Sha256};

use crate::{WorkerError, WorkerResult};

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
    if live.asset_ids != authored.asset_ids {
        return Err(WorkerError::InvalidPayload("FLUX.3 reference attachment slots changed after validation; rebuild the request from the authored payload".to_owned()));
    }
    Ok(Some(authored))
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
        let image = gen_core::flux3::references::decode_reference_file(&path)
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
    let plan = gen_core::flux3::references::plan(&dimensions, output_size)
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
