use std::path::Path;

use image::DynamicImage;
use sceneworks_core::workflow_png::write_workflow_chunk;
use sceneworks_core::workflow_share::WorkflowShare;
use serde_json::{json, Value};

use crate::{fresh_asset_id, now_rfc3339, task_join_error, JsonObject, WorkerError, WorkerResult};

pub(crate) struct SingleChildAssetWrite {
    pub asset_id: String,
    pub generation_set_id: String,
    pub created_at: String,
    pub media_path: String,
}

pub(crate) struct SingleChildAssetSpec<'a> {
    pub filename_stem: &'a str,
    pub mode: &'a str,
    pub model: &'a str,
    pub adapter: &'a str,
    pub encode_label: &'a str,
    /// The sanitized workflow to embed in the written PNG, or `None` to write the file exactly as
    /// this seam always has (epic 15945, sc-15948).
    ///
    /// Not defaulted, and deliberately not `Default`-derived: this is the FOURTH place the worker
    /// writes a generated PNG, and the reason the standalone upscale shipped with no chunk is that
    /// nobody had to decide. A new caller of this seam now has to say which it is, and to say why in
    /// the `None` case.
    pub workflow: Option<WorkflowShare>,
}

/// Persist one PNG child and build the common one-asset generation result. Upscale and smart-select
/// supply only their domain-specific fact fields; atomic temp-file encoding, ids, generation-set
/// metadata, and the API result envelope stay identical.
pub(crate) async fn write_single_child_asset<F>(
    project_path: &Path,
    image: DynamicImage,
    mut spec: SingleChildAssetSpec<'_>,
    build_fact: F,
) -> WorkerResult<JsonObject>
where
    F: FnOnce(&SingleChildAssetWrite) -> Value,
{
    let created_at = now_rfc3339();
    let generation_set_id = format!("genset_{}", uuid::Uuid::new_v4().simple());
    let asset_id = fresh_asset_id();
    let date = &created_at[..10];
    let suffix: String = asset_id.chars().skip(6).take(8).collect();
    let filename = format!("{date}_{}_{suffix}.png", spec.filename_stem);
    let media_path = format!("assets/images/{generation_set_id}/{filename}");
    let absolute_path = project_path.join(&media_path);
    if let Some(parent) = absolute_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let tmp_path = absolute_path.with_extension("tmp.png");
    let encode_tmp = tmp_path.clone();
    let workflow = spec.workflow.take();
    tokio::task::spawn_blocking(move || match workflow {
        // The embed lane (sc-15948). The conversion follows the buffer the caller produced rather
        // than forcing one: the upscale lane hands in `DynamicImage::ImageRgb8`, so `into_rgb8()`
        // is still a move and its files are byte-identical, while an alpha-carrying render
        // (sc-24111, Qwen Image 2.1's native transparency) is written as RGBA instead of being
        // silently flattened against whatever RGB sat under the transparent pixels.
        //
        // Keyed on the DECLARED colour type, not on whether any pixel is actually transparent —
        // see `workflow_png::WorkflowImage`. The `None` arm below has always preserved alpha
        // (`DynamicImage::save_with_format` writes the variant it holds), so before this the two
        // arms of one function disagreed about the channel count. The grayscale mask lane passes
        // `None` and keeps its L8 encoding untouched.
        Some(share) => if image.color().has_alpha() {
            write_workflow_chunk(&image.into_rgba8(), &encode_tmp, Some(&share))
        } else {
            write_workflow_chunk(&image.into_rgb8(), &encode_tmp, Some(&share))
        }
        .map_err(|error| WorkerError::Io(std::io::Error::other(error))),
        None => image
            .save_with_format(&encode_tmp, image::ImageFormat::Png)
            .map_err(|error| WorkerError::Io(std::io::Error::other(error))),
    })
    .await
    .map_err(|error| task_join_error(spec.encode_label, error))??;
    tokio::fs::rename(&tmp_path, &absolute_path)
        .await
        .inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp_path);
        })?;

    let write = SingleChildAssetWrite {
        asset_id,
        generation_set_id: generation_set_id.clone(),
        created_at: created_at.clone(),
        media_path,
    };
    let generation_set = json!({
        "id": generation_set_id,
        "mode": spec.mode,
        "model": spec.model,
        "prompt": "",
        "negativePrompt": "",
        "count": 1,
        "createdAt": created_at,
    });
    let mut result = JsonObject::new();
    result.insert(
        "generationSetId".to_owned(),
        Value::String(write.generation_set_id.clone()),
    );
    result.insert("expectedCount".to_owned(), json!(1));
    result.insert("adapter".to_owned(), Value::String(spec.adapter.to_owned()));
    result.insert("model".to_owned(), Value::String(spec.model.to_owned()));
    result.insert("generationSet".to_owned(), generation_set);
    result.insert(
        "assetWrites".to_owned(),
        Value::Array(vec![build_fact(&write)]),
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn writes_one_png_and_builds_the_shared_result_envelope() {
        let dir = tempfile::tempdir().expect("temp dir");
        let image =
            DynamicImage::ImageLuma8(image::GrayImage::from_pixel(1, 1, image::Luma([255])));
        let result = write_single_child_asset(
            dir.path(),
            image,
            SingleChildAssetSpec {
                filename_stem: "mask",
                mode: "image_segment",
                model: "sam3",
                adapter: "sam3",
                encode_label: "test encode",
                workflow: None,
            },
            |write| {
                json!({
                    "assetId": write.asset_id,
                    "mediaPath": write.media_path,
                    "createdAt": write.created_at,
                })
            },
        )
        .await
        .expect("child writes");

        assert_eq!(result["expectedCount"], json!(1));
        assert_eq!(result["adapter"], "sam3");
        assert_eq!(
            result["generationSetId"], result["generationSet"]["id"],
            "one id feeds result and generation set"
        );
        let relative = result["assetWrites"][0]["mediaPath"]
            .as_str()
            .expect("media path");
        assert!(dir.path().join(relative).is_file());
        assert_eq!(
            image::open(dir.path().join(relative))
                .expect("saved png decodes")
                .color(),
            image::ColorType::L8,
            "mask encoding remains grayscale"
        );
        assert_eq!(
            sceneworks_core::workflow_png::read_workflow_chunk_file(&dir.path().join(relative))
                .expect("the mask PNG is readable"),
            None,
            "a segmentation mask carries no generation recipe — see the `workflow: None` at the \
             smart-select call site"
        );
    }

    /// The FOURTH write seam embeds too (sc-15948).
    ///
    /// `write_single_child_asset` is where the standalone `image_upscale` job's PNG is written, and
    /// it wrote a bare `save_with_format` with no chunk — so the most-shared asset class in the app
    /// was the one with no recipe inside it. This drives the real seam and reads the envelope back
    /// out of the file on disk, because the AC is about what is IN the written PNG.
    #[tokio::test]
    async fn an_upscaled_png_carries_the_workflow_of_the_pass_that_produced_it() {
        let dir = tempfile::tempdir().expect("temp dir");
        // The payload `buildUpscaleJobBody` posts, plus the geometry the pass resolved.
        let payload = json!({
            "projectId": "project_7a10",
            "sourceAssetId": "asset_source_1",
            "factor": 2,
            "engine": "seedvr2",
            "displayName": "Lighthouse in fog",
            "softness": 0.25
        })
        .as_object()
        .cloned()
        .expect("object");
        let share = crate::image_jobs::standalone_upscale_workflow_share(
            &payload,
            "seedvr2",
            2,
            Some(0.25),
            4242,
            320,
            256,
        )
        .expect("a fixture envelope is far under the recording ceiling");

        let image =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(8, 8, image::Rgb([12, 34, 56])));
        let result = write_single_child_asset(
            dir.path(),
            image,
            SingleChildAssetSpec {
                filename_stem: "upscaled_x2",
                mode: "image_upscale",
                model: "seedvr2",
                adapter: "seedvr2",
                encode_label: "test encode",
                workflow: Some(share),
            },
            |write| json!({ "mediaPath": write.media_path }),
        )
        .await
        .expect("child writes");

        let media = dir.path().join(
            result["assetWrites"][0]["mediaPath"]
                .as_str()
                .expect("path"),
        );
        let embedded = sceneworks_core::workflow_png::read_workflow_chunk_file(&media)
            .expect("the written PNG is readable")
            .expect("the standalone upscale must embed its own workflow");

        assert_eq!(embedded.mode, "image_upscale");
        assert_eq!(
            embedded.model, "seedvr2",
            "the engine IS the model of this pass"
        );
        assert_eq!(embedded.seed, Some(4242));
        // Source geometry, not the written file's: the envelope is a recipe, and "this 320x256
        // image, upscaled 2x" is what reproduces it.
        assert_eq!((embedded.width, embedded.height), (Some(320), Some(256)));
        let upscale = embedded.upscale.as_ref().expect("the pass is recorded");
        assert!(upscale.enabled);
        assert_eq!(upscale.engine.as_deref(), Some("seedvr2"));
        assert_eq!(upscale.factor, Some(2));
        assert_eq!(upscale.softness, Some(0.25));
        // The source image rides as a SHAPE, never as the local asset id.
        assert_eq!(embedded.inputs.len(), 1);
        assert_eq!(embedded.inputs[0].kind, "source");
        let text = serde_json::to_string(&embedded).expect("serializes");
        for local in ["asset_source_1", "project_7a10", "Lighthouse in fog"] {
            assert!(!text.contains(local), "{local} leaked: {text}");
        }

        // And it is still an ordinary PNG that every decoder reads.
        let decoded = image::open(&media).expect("decodes");
        assert_eq!((decoded.width(), decoded.height()), (8, 8));
    }

    /// Real-ESRGAN has no softness control, so the envelope must not invent one.
    #[test]
    fn a_softness_less_engine_records_no_softness() {
        let payload = json!({ "sourceAssetId": "asset_source_1", "factor": 4 })
            .as_object()
            .cloned()
            .expect("object");
        let share = crate::image_jobs::standalone_upscale_workflow_share(
            &payload,
            "real-esrgan",
            4,
            None,
            0,
            512,
            512,
        )
        .expect("a fixture envelope is far under the recording ceiling");
        let upscale = share.upscale.as_ref().expect("the pass is recorded");
        assert_eq!(upscale.engine.as_deref(), Some("real-esrgan"));
        assert_eq!(upscale.factor, Some(4));
        assert_eq!(
            upscale.softness, None,
            "recording a softness on an engine with no such knob would be inventing a fact"
        );
    }

    /// An alpha-carrying render keeps its channel through the embed lane (sc-24111).
    ///
    /// This seam used to read `write_workflow_chunk(&image.into_rgb8(), ...)`, which meant the two
    /// arms of one function disagreed: the `None` arm called `DynamicImage::save_with_format` and
    /// preserved whatever variant it held, while the `Some` arm flattened unconditionally. So
    /// turning workflow embedding ON silently cost the alpha channel — and embedding is on by
    /// default. Both arms are driven here with the same RGBA buffer.
    #[tokio::test]
    async fn an_rgba_render_keeps_its_alpha_through_both_arms_of_the_write() {
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("tests")
            .join("fixtures")
            .join("alpha")
            .join("alpha-64.png");
        let source = image::open(&fixture)
            .unwrap_or_else(|error| {
                panic!("RGBA fixture at {} decodes: {error}", fixture.display())
            })
            .to_rgba8();
        let expected: std::collections::BTreeMap<u8, usize> =
            source.pixels().fold(Default::default(), |mut map, pixel| {
                *map.entry(pixel.0[3]).or_insert(0) += 1;
                map
            });
        assert!(
            expected.len() >= 8 && expected.contains_key(&0) && expected.contains_key(&255),
            "the committed fixture lost its soft/transparent/opaque structure"
        );

        let payload = json!({ "sourceAssetId": "asset_source_1", "factor": 2 })
            .as_object()
            .cloned()
            .expect("object");
        let share = crate::image_jobs::standalone_upscale_workflow_share(
            &payload,
            "real-esrgan",
            2,
            None,
            7,
            64,
            64,
        )
        .expect("a fixture envelope is far under the recording ceiling");

        for (label, workflow) in [("embed", Some(share)), ("opt-out", None)] {
            let dir = tempfile::tempdir().expect("temp dir");
            let result = write_single_child_asset(
                dir.path(),
                DynamicImage::ImageRgba8(source.clone()),
                SingleChildAssetSpec {
                    filename_stem: "transparent",
                    mode: "image_upscale",
                    model: "real-esrgan",
                    adapter: "real-esrgan",
                    encode_label: "test encode",
                    workflow,
                },
                |write| json!({ "mediaPath": write.media_path }),
            )
            .await
            .expect("child writes");

            let media = dir.path().join(
                result["assetWrites"][0]["mediaPath"]
                    .as_str()
                    .expect("path"),
            );
            let decoded = image::open(&media).expect("the written PNG decodes");
            assert_eq!(
                decoded.color(),
                image::ColorType::Rgba8,
                "the {label} arm dropped the alpha channel"
            );
            let decoded = decoded.to_rgba8();
            let actual: std::collections::BTreeMap<u8, usize> =
                decoded.pixels().fold(Default::default(), |mut map, pixel| {
                    *map.entry(pixel.0[3]).or_insert(0) += 1;
                    map
                });
            assert_eq!(
                actual, expected,
                "the {label} arm changed the alpha histogram"
            );
            assert_eq!(
                decoded.as_raw(),
                source.as_raw(),
                "the {label} arm changed the pixels"
            );
        }
    }

    /// The control: an RGB render must not grow a channel it never had.
    #[tokio::test]
    async fn an_rgb_render_is_still_written_as_rgb() {
        let dir = tempfile::tempdir().expect("temp dir");
        let payload = json!({ "sourceAssetId": "asset_source_1", "factor": 2 })
            .as_object()
            .cloned()
            .expect("object");
        let share = crate::image_jobs::standalone_upscale_workflow_share(
            &payload,
            "real-esrgan",
            2,
            None,
            7,
            8,
            8,
        )
        .expect("a fixture envelope is far under the recording ceiling");
        let source = image::RgbImage::from_fn(8, 8, |x, y| {
            image::Rgb([(x * 7) as u8, (y * 11) as u8, 200])
        });

        let result = write_single_child_asset(
            dir.path(),
            DynamicImage::ImageRgb8(source.clone()),
            SingleChildAssetSpec {
                filename_stem: "opaque",
                mode: "image_upscale",
                model: "real-esrgan",
                adapter: "real-esrgan",
                encode_label: "test encode",
                workflow: Some(share),
            },
            |write| json!({ "mediaPath": write.media_path }),
        )
        .await
        .expect("child writes");

        let media = dir.path().join(
            result["assetWrites"][0]["mediaPath"]
                .as_str()
                .expect("path"),
        );
        let decoded = image::open(&media).expect("decodes");
        assert_eq!(decoded.color(), image::ColorType::Rgb8);
        assert_eq!(decoded.to_rgb8().as_raw(), source.as_raw());
    }
}
