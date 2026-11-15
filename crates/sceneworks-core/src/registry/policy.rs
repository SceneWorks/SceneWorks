//! SceneWorks router admission is separate from a native engine's capability descriptor.
//! This product policy is the sole authored owner of the former image/video routing columns.
use super::spec::RoutePolicy;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PolicyDocument {
    pub schema_version: u32,
    pub image: Vec<ImagePolicyRow>,
    pub video: Vec<VideoPolicyRow>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImagePolicyRow {
    pub id: String,
    pub mlx_routed: bool,
    pub candle_routed: bool,
    pub candle_quant: bool,
    pub candle_lora: bool,
    pub candle_quant_lora: bool,
    #[serde(default)]
    pub legacy_alias: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VideoPolicyRow {
    pub id: String,
    pub video_mlx_routed: bool,
    pub candle_video_routed: bool,
    pub candle_video_i2v: bool,
    pub candle_video_vace: bool,
    #[serde(default)]
    pub legacy_alias: bool,
}

pub fn parse(contents: &str) -> Result<PolicyDocument, String> {
    let document: PolicyDocument = serde_json::from_str(contents).map_err(|e| e.to_string())?;
    if document.schema_version != 1 {
        return Err(format!(
            "unsupported route policy schema {}",
            document.schema_version
        ));
    }
    let mut ids = BTreeSet::new();
    for row in &document.image {
        if !valid_id(&row.id) || !ids.insert(&row.id) {
            return Err(format!(
                "invalid or duplicate image route policy {}",
                row.id
            ));
        }
        if !row.candle_routed && (row.candle_quant || row.candle_lora || row.candle_quant_lora) {
            return Err(format!(
                "{} has candle adapter/tier policy without a route",
                row.id
            ));
        }
    }
    for row in &document.video {
        if !valid_id(&row.id) || !ids.insert(&row.id) {
            return Err(format!(
                "invalid or duplicate video route policy {}",
                row.id
            ));
        }
        if !row.candle_video_routed && (row.candle_video_i2v || row.candle_video_vace) {
            return Err(format!(
                "{} has candle video mode policy without a route",
                row.id
            ));
        }
    }
    Ok(document)
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

impl PolicyDocument {
    pub fn for_model(&self, id: &str) -> Option<RoutePolicy> {
        if let Some(row) = self.image.iter().find(|row| row.id == id) {
            return Some(RoutePolicy::Image {
                mlx_routed: row.mlx_routed,
                candle_routed: row.candle_routed,
                candle_quant: row.candle_quant,
                candle_lora: row.candle_lora,
                candle_quant_lora: row.candle_quant_lora,
            });
        }
        self.video
            .iter()
            .find(|row| row.id == id)
            .map(|row| RoutePolicy::Video {
                video_mlx_routed: row.video_mlx_routed,
                candle_video_routed: row.candle_video_routed,
                candle_video_i2v: row.candle_video_i2v,
                candle_video_vace: row.candle_video_vace,
            })
    }

    /// Emit Rust constants so existing job predicates keep their exact static list contracts.
    /// The checked-in JSON, not the generated include, is the sole authored model-fact table.
    #[allow(dead_code)] // Build.rs uses this; the core test module only checks the parser.
    pub fn rust_constants(&self) -> String {
        let mut out = String::from("pub(crate) const IMAGE_MODEL_CAPS: &[ModelCaps] = &[\n");
        for row in &self.image {
            out.push_str(&format!(
                "    ModelCaps::new({:?}, {}, {}, {}, {}, {}),\n",
                row.id,
                row.mlx_routed,
                row.candle_routed,
                row.candle_quant,
                row.candle_lora,
                row.candle_quant_lora
            ));
        }
        out.push_str("];\npub(crate) const VIDEO_MODEL_CAPS: &[VideoModelCaps] = &[\n");
        for row in &self.video {
            out.push_str(&format!(
                "    VideoModelCaps::new({:?}, {}, {}, {}, {}),\n",
                row.id,
                row.video_mlx_routed,
                row.candle_video_routed,
                row.candle_video_i2v,
                row.candle_video_vace
            ));
        }
        out.push_str("];\n");
        out.push_str("pub(crate) const LEGACY_IMAGE_MLX_ROUTED: &[&str] = &[\n");
        for row in self
            .image
            .iter()
            .filter(|row| row.legacy_alias && row.mlx_routed)
        {
            out.push_str(&format!("    {:?},\n", row.id));
        }
        out.push_str("];\n");
        out.push_str("pub(crate) const LEGACY_VIDEO_MLX_ROUTED: &[&str] = &[\n");
        for row in self
            .video
            .iter()
            .filter(|row| row.legacy_alias && row.video_mlx_routed)
        {
            out.push_str(&format!("    {:?},\n", row.id));
        }
        out.push_str("];\n");
        out
    }
}
