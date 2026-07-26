use std::path::Path;

use image::DynamicImage;
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
}

/// Persist one PNG child and build the common one-asset generation result. Upscale and smart-select
/// supply only their domain-specific fact fields; atomic temp-file encoding, ids, generation-set
/// metadata, and the API result envelope stay identical.
pub(crate) async fn write_single_child_asset<F>(
    project_path: &Path,
    image: DynamicImage,
    spec: SingleChildAssetSpec<'_>,
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
    tokio::task::spawn_blocking(move || {
        image
            .save_with_format(&encode_tmp, image::ImageFormat::Png)
            .map_err(|error| WorkerError::Io(std::io::Error::other(error)))
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
    }
}
