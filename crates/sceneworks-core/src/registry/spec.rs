//! Build-time inputs and portable static facts. No inference backend is linked to read these.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSpec {
    pub product: ProductModel,
    pub routes: BTreeMap<String, GeneratorRoute>,
    pub evidence: BTreeMap<String, BackendEvidence>,
}

/// The shipping identity, distribution, product menus and copy. Unknown fields are rejected so
/// a native capability or a memory declaration cannot silently acquire product authority.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProductModel {
    pub id: String,
    pub name: String,
    pub family: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub adapter: String,
    pub capabilities: Vec<String>,
    pub downloads: Vec<ProductDownload>,
    pub paths: BTreeMap<String, String>,
    pub gated: bool,
    pub defaults: ProductDefaults,
    pub limits: ProductMenus,
    pub lora_compatibility: BTreeMap<String, Vec<String>>,
    pub ui: Map<String, Value>,
    // These three declared controls are absent from the current runtime export.
    pub image: ImageControls,
    pub mlx: ProductMlx,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProductDownload {
    pub provider: String,
    pub repo: String,
    pub revision: String,
    pub variant: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<bool>,
    pub files: Vec<String>,
    pub estimated_size_bytes: u64,
    pub footprint: BTreeMap<String, Option<u64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProductDefaults {
    pub resolution: String,
    pub steps: u32,
    pub guidance_scale: serde_json::Number,
    pub count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductMenus {
    pub resolutions: Vec<String>,
    pub count: Vec<u32>,
    pub samplers: Vec<String>,
    pub schedulers: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImageControls {
    pub supports_guidance: bool,
    pub supports_negative_prompt: bool,
    pub supports_scheduler_shift: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductMlx {
    pub quantize: u32,
}

// Field names here are those of the shipped native runtime descriptor (snake case).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneratorRoute {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub modality: String,
    pub conditioning: Vec<String>,
    pub supported_quants: Vec<String>,
    pub supports_lora: bool,
    pub supports_lokr: bool,
    pub supports_preview: bool,
    pub supports_prompt_enhancement: bool,
    pub samplers: Vec<String>,
    pub schedulers: Vec<String>,
    pub supported_guidance_methods: Vec<String>,
}

/// Preserve the calibration and memory-contract ABI payloads and their identity fields verbatim.
/// Consumers deserialize them with their existing schema. The store grants no product overrides.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BackendEvidence {
    pub min_memory_gb: u32,
    pub memory_strategy_contract: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibrations: Option<Vec<Map<String, Value>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_strategy_capabilities: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_sequential_offload: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_gb_by_tier: Option<BTreeMap<String, f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequential_peak_gb: Option<BTreeMap<String, f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_measured_pixels: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured: Option<bool>,
}
