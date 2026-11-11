//! Build-time inputs and portable static facts. No inference backend is linked to read these.
//! The v2 manifest schema owns product choices and copy; native descriptor dumps own executable
//! backend capabilities; the evidence document owns measured memory and calibration facts. The
//! build joins those authorities without promoting host installation or credentials into a spec.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSpec {
    pub product: ProductModel,
    /// SceneWorks router admission, separate from native engine descriptor breadth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<RoutePolicy>,
    /// Stage-1/2 generated preview feature export, joined at build time for every public model.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preview_by_backend: BTreeMap<String, bool>,
    /// The primary image generator on each backend, retained for existing consumers.
    pub routes: BTreeMap<String, GeneratorRoute>,
    /// Video modes can select several native engines on the same backend.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub video_routes: BTreeMap<String, Vec<VideoRoute>>,
    /// Native audio and utility providers have no image generator mapping.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider_routes: BTreeMap<String, Vec<ProviderRoute>>,
    pub evidence: BTreeMap<String, BackendEvidence>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub model_evidence: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub download_evidence: Vec<Map<String, Value>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RoutePolicy {
    Image {
        mlx_routed: bool,
        candle_routed: bool,
        candle_quant: bool,
        candle_lora: bool,
        candle_quant_lora: bool,
    },
    Video {
        video_mlx_routed: bool,
        candle_video_routed: bool,
        candle_video_i2v: bool,
        candle_video_vace: bool,
    },
}

impl RoutePolicy {
    pub fn mlx_routed(&self) -> bool {
        match self {
            Self::Image { mlx_routed, .. } => *mlx_routed,
            Self::Video {
                video_mlx_routed, ..
            } => *video_mlx_routed,
        }
    }

    pub fn candle_routed(&self) -> bool {
        match self {
            Self::Image { candle_routed, .. } => *candle_routed,
            Self::Video {
                candle_video_routed,
                ..
            } => *candle_video_routed,
        }
    }
}

/// Shipping product facts. The common identity is typed here; the schema-validated manifest owns
/// specialized image, video, audio, utility, and vector fields without a second hand mirror. The
/// flattened fields are not a second source: build.rs validates the complete v2 source document
/// before assembly, and callers consume this single compiled projection.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProductModel {
    pub id: String,
    pub name: String,
    pub family: String,
    #[serde(rename = "type")]
    pub model_type: String,
    #[serde(flatten)]
    pub fields: BTreeMap<String, Value>,
}

impl ProductModel {
    /// The checkpoint's `image.supportsHiresFix` fact: false for a checkpoint that cannot run
    /// Hires.fix's img2img second pass. Absent means true. Author choices never change it.
    pub fn supports_hires_fix(&self) -> bool {
        supports_hires_fix(self.fields.get("image"))
    }
}

/// `image.supportsHiresFix` of a product or manifest `image` block; absent means true.
pub fn supports_hires_fix(image: Option<&Value>) -> bool {
    image
        .and_then(|image| image.get("supportsHiresFix"))
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoRoute {
    pub mode: String,
    pub engines: Vec<GeneratorRoute>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderRoute {
    pub kind: String,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generator: Option<GeneratorRoute>,
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
#[serde(rename_all = "camelCase")]
pub struct BackendEvidence {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_memory_gb: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_strategy_contract: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibrations: Option<Vec<Map<String, Value>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_strategy_capabilities: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_sequential_offload: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_gb_by_tier: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequential_peak_gb: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_measured_pixels: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured: Option<bool>,
    /// Other evidence keys are validated against the v2 evidence document before assembly.
    #[serde(flatten)]
    pub fields: BTreeMap<String, Value>,
}
