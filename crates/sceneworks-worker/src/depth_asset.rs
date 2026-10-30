//! Reading a saved depth-map asset (Iris-3B, sc-25682) as a CONSUMER — independent of the job that
//! produced it, of the Iris weights and of any UI.
//!
//! A depth asset's `file` is a raw float32 `H × W` `<f4` `.npy` (relative log depth, see
//! [`sceneworks_core::depth_map`]); its sidecar `depth` block records the value convention. The
//! depth-control path ([`crate::image_jobs`] strict control, `ControlKind::Depth`) accepts such an
//! asset as its control map: [`depth_asset_control_image`] decodes a COPY of the raw map and derives
//! the near-bright control image through the engine's own adapter
//! ([`gen_core_iris::depth::near_bright_control_image`]). The `.npy` on disk is only ever read.

use std::path::Path;

use gen_core_iris::depth::{near_bright_control_image, DepthMap, VALUE_CONVENTION};
use sceneworks_core::depth_map::{decode_npy_f32, DEPTH_ASSET_TYPE};
use sceneworks_core::project_store::ProjectStore;
use serde_json::Value;

use crate::{WorkerError, WorkerResult};

/// Whether an asset record is a depth map.
pub(crate) fn is_depth_asset(asset: &Value) -> bool {
    asset.get("type").and_then(Value::as_str) == Some(DEPTH_ASSET_TYPE)
}

/// Decode the raw map of a depth asset record. Refuses a map whose recorded value convention is not
/// one this adapter understands, rather than presenting an unknown quantity as near-bright depth.
pub(crate) fn read_depth_asset_map(asset: &Value, project_path: &Path) -> WorkerResult<DepthMap> {
    let asset_id = asset.get("id").and_then(Value::as_str).unwrap_or("?");
    let convention = asset
        .pointer("/depth/valueConvention")
        .and_then(Value::as_str);
    if convention != Some(VALUE_CONVENTION) {
        return Err(WorkerError::InvalidPayload(format!(
            "depth asset {asset_id} records value convention {convention:?}; the depth-control \
             adapter reads only '{VALUE_CONVENTION}'"
        )));
    }
    let rel = asset
        .pointer("/file/path")
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("depth asset {asset_id} has no media path"))
        })?;
    let path = crate::safe_project_path(project_path, rel)?;
    let bytes = std::fs::read(&path).map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "depth asset {asset_id}: cannot read {}: {error}",
            path.display()
        ))
    })?;
    let array = decode_npy_f32(&bytes)
        .map_err(|error| WorkerError::InvalidPayload(format!("depth asset {asset_id}: {error}")))?;
    Ok(DepthMap {
        width: array.width,
        height: array.height,
        values: array.values,
    })
}

/// `Some(control image)` when `asset_id` names a depth asset — the near-bright gray RGB8 image at the
/// map's size, drop-in for `ControlKind::Depth` — and `None` for any other asset (the caller then
/// decodes it as an ordinary control image).
pub(crate) fn depth_asset_control_image(
    data_dir: &Path,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> WorkerResult<Option<gen_core::Image>> {
    let asset = ProjectStore::new(data_dir.to_path_buf(), "worker")
        .get_asset(project_id, asset_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("control asset {asset_id}: {error}"))
        })?;
    if !is_depth_asset(&asset) {
        return Ok(None);
    }
    let map = read_depth_asset_map(&asset, project_path)?;
    Ok(Some(near_bright_control_image(&map)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::depth_map::encode_npy_f32;
    use serde_json::json;

    /// The separate-consumer contract: a depth asset saved by an earlier job is read by a FRESH store
    /// (nothing in memory from the producer — the after-restart shape), turned into a near-bright
    /// control image, and the raw `.npy` bytes are unchanged afterwards.
    #[test]
    fn a_saved_depth_asset_feeds_depth_control_and_stays_raw() {
        let data = tempfile::tempdir().unwrap();
        let project = {
            let store = ProjectStore::new(data.path().to_path_buf(), "producer");
            store.create_project("Depth consumer").unwrap()
        };
        let project_path = std::path::PathBuf::from(&project.path);
        // 4x2 map: near (−1) on the left, far (+1) on the right.
        let values = vec![-1.0, -0.5, 0.5, 1.0, -1.0, -0.5, 0.5, 1.0];
        let npy_rel = "assets/depth/genset_c/map.npy";
        std::fs::create_dir_all(project_path.join("assets/depth/genset_c")).unwrap();
        let raw = encode_npy_f32(4, 2, &values).unwrap();
        std::fs::write(project_path.join(npy_rel), &raw).unwrap();
        ProjectStore::new(data.path().to_path_buf(), "producer")
            .persist_generated_asset(
                &project.id,
                "job-depth",
                "genset_c",
                &json!({
                    "assetId": "asset_depthc",
                    "displayName": "Map (depth)",
                    "createdAt": "2026-10-11T00:00:00Z",
                    "type": "depth",
                    "mediaPath": npy_rel,
                    "mimeType": "application/x-npy",
                    "width": 4,
                    "height": 2,
                    "mode": "image_depth",
                    "model": "iris_3b_depth",
                    "depth": { "valueConvention": "relative_log_depth" },
                }),
            )
            .unwrap();

        let control =
            depth_asset_control_image(data.path(), &project.id, "asset_depthc", &project_path)
                .unwrap()
                .expect("a depth asset yields a control image");
        assert_eq!((control.width, control.height), (4, 2));
        let gray: Vec<u8> = control
            .pixels
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| p[0])
            .collect();
        assert!(control
            .pixels
            .as_chunks::<3>()
            .0
            .iter()
            .all(|p| p[0] == p[1] && p[1] == p[2]));
        assert_eq!(gray[0], 255, "nearest is brightest");
        assert_eq!(gray[3], 0, "farthest is darkest");
        assert!(gray[1] > gray[2], "near-bright ordering");
        assert_eq!(
            std::fs::read(project_path.join(npy_rel)).unwrap(),
            raw,
            "the raw map is never rewritten"
        );
    }

    #[test]
    fn non_depth_assets_and_unknown_conventions_are_not_reinterpreted() {
        let project = tempfile::tempdir().unwrap();
        let image = json!({ "id": "a", "type": "image", "file": { "path": "x.png" } });
        assert!(!is_depth_asset(&image));
        let metric = json!({
            "id": "d", "type": "depth", "file": { "path": "x.npy" },
            "depth": { "valueConvention": "metric_meters" },
        });
        let error = read_depth_asset_map(&metric, project.path()).unwrap_err();
        assert!(error.to_string().contains("metric_meters"), "{error}");
    }
}
