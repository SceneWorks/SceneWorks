use serde::{Deserialize, Serialize};
use serde_json::Value;
pub type JsonObject = serde_json::Map<String, Value>;
pub type ExtraFields = std::collections::BTreeMap<String, Value>;
pub type ContractNumber = serde_json::Number;
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageUpscaleRequest {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_image_upscale_factor")]
    pub factor: u8,
    #[serde(default = "default_image_upscale_engine")]
    pub engine: String,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

impl Default for ImageUpscaleRequest {
    fn default() -> Self {
        Self {
            enabled: false,
            factor: default_image_upscale_factor(),
            engine: default_image_upscale_engine(),
            extra: ExtraFields::default(),
        }
    }
}

impl ImageUpscaleRequest {
    pub fn is_disabled(&self) -> bool {
        !self.enabled
    }

    /// SeedVR2-only input pre-blur (0..1), carried in the flattened extras. 0.0 ⇒ none.
    pub fn softness(&self) -> f32 {
        self.extra
            .get("softness")
            .and_then(Value::as_f64)
            .map(|value| value.clamp(0.0, 1.0) as f32)
            .unwrap_or(0.0)
    }

    /// Whether this block selects the Iris-3B restoration engine (sc-25683).
    pub fn is_iris(&self) -> bool {
        crate::image_restoration::is_iris_engine(&self.engine)
    }

    /// The Iris-only controls (`scale` / `inputSizing` / `colorFix`) carried in the flattened
    /// extras, as a JSON object for [`crate::image_restoration`]'s shared parse.
    pub fn iris_controls(&self) -> JsonObject {
        crate::image_restoration::IRIS_ONLY_KEYS
            .iter()
            .filter_map(|key| {
                self.extra
                    .get(*key)
                    .map(|value| ((*key).to_owned(), value.clone()))
            })
            .collect()
    }

    /// Shared API/worker upscale boundary. Disabled controls remain inert.
    pub fn validate(&self) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        if self.engine.trim().is_empty() {
            return Err("upscale.engine is required".into());
        }
        if self.is_iris() {
            if self.factor != default_image_upscale_factor() {
                return Err("upscale.factor is not a control of the Iris restoration engine; use upscale.scale".into());
            }
            if self.extra.contains_key("softness") {
                return Err(
                    "upscale.softness is not a control of the Iris restoration engine".into(),
                );
            }
            self.iris_params()
                .map_err(|error| format!("upscale.{error}"))?;
        } else {
            image_upscale_factor(u64::from(self.factor))
                .map_err(|_| "upscale.factor must be 2 or 4".to_owned())?;
            crate::image_restoration::reject_iris_only_keys(&self.iris_controls(), &self.engine)
                .map_err(|error| format!("upscale.{error}"))?;
        }
        Ok(())
    }

    /// The parsed Iris restoration controls (a typed refusal for a malformed one).
    pub fn iris_params(&self) -> Result<crate::image_restoration::IrisRestoreParams, String> {
        crate::image_restoration::parse_iris_restore_params(&self.iris_controls())
    }
}

/// The discrete scales supported by the non-Iris image upscalers.
pub fn image_upscale_factor(factor: u64) -> Result<u8, String> {
    match factor {
        2 | 4 => Ok(factor as u8),
        other => Err(format!(
            "Image upscale supports only factor 2 or 4 (got {other})."
        )),
    }
}

pub const fn default_image_upscale_factor() -> u8 {
    2
}

pub fn default_image_upscale_engine() -> String {
    "real-esrgan".to_owned()
}

/// Optional A1111-style Hires.fix pass for text-to-image generation.
///
/// The first pass renders at the requested job dimensions. The worker then resizes that image to
/// `upscale_by` times the original dimensions and uses it as the img2img initialization for a second
/// pass. `steps == 0` and `cfg_scale == None` mean "inherit the first-pass value".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HiresFixRequest {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub steps: u32,
    #[serde(default = "default_hires_denoising_strength")]
    pub denoising_strength: f64,
    #[serde(default = "default_hires_upscale_by")]
    pub upscale_by: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f64>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

impl Default for HiresFixRequest {
    fn default() -> Self {
        Self {
            enabled: false,
            steps: 0,
            denoising_strength: default_hires_denoising_strength(),
            upscale_by: default_hires_upscale_by(),
            cfg_scale: None,
            extra: ExtraFields::default(),
        }
    }
}

impl HiresFixRequest {
    pub fn is_disabled(&self) -> bool {
        !self.enabled
    }

    pub fn effective_steps(&self, first_pass_steps: u32) -> u32 {
        if self.steps == 0 {
            first_pass_steps
        } else {
            self.steps.clamp(1, 150)
        }
    }

    pub fn effective_denoising_strength(&self) -> f32 {
        self.denoising_strength.clamp(0.0, 1.0) as f32
    }

    pub fn effective_upscale_by(&self) -> f32 {
        self.upscale_by.clamp(1.0, 4.0) as f32
    }

    pub fn effective_cfg_scale(&self, first_pass_cfg: Option<f32>) -> Option<f32> {
        self.cfg_scale
            .filter(|value| value.is_finite())
            .map(|value| value.clamp(0.0, 60.0) as f32)
            .or(first_pass_cfg)
    }
}

pub const fn default_hires_denoising_strength() -> f64 {
    0.7
}

pub const fn default_hires_upscale_by() -> f64 {
    2.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoUpscaleRequest {
    /// The source video asset to upscale (required; the worker validates non-empty).
    #[serde(default)]
    pub source_asset_id: String,
    /// Integer scale factor applied to the source dimensions (2 or 4). Ignored when explicit
    /// `target_width`/`target_height` are supplied.
    #[serde(default = "default_video_upscale_factor")]
    pub factor: u8,
    /// Explicit target width override (snapped to ÷16). `None` ⇒ `factor × source_width`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_width: Option<u32>,
    /// Explicit target height override (snapped to ÷16). `None` ⇒ `factor × source_height`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_height: Option<u32>,
    /// Upscale engine. Only `seedvr2` is Mac-supported today (native MLX).
    #[serde(default = "default_video_upscale_engine")]
    pub engine: String,
    /// SeedVR2 model variant id. `seedvr2_3b` is the only wired variant (7B = sc-5197).
    #[serde(default = "default_video_upscale_model")]
    pub model: String,
    /// SeedVR2 input pre-blur (0.0 = none, the reference default). Higher smooths source artifacts.
    #[serde(default)]
    pub softness: f32,
    /// Optional reproducibility seed (SeedVR2 invents plausible detail; one-step but non-deterministic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Project the source asset belongs to. `None` ⇒ the worker falls back to the job's project id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Optional display name for the result asset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

impl Default for VideoUpscaleRequest {
    fn default() -> Self {
        Self {
            source_asset_id: String::new(),
            factor: default_video_upscale_factor(),
            target_width: None,
            target_height: None,
            engine: default_video_upscale_engine(),
            model: default_video_upscale_model(),
            softness: 0.0,
            seed: None,
            project_id: None,
            display_name: None,
            extra: ExtraFields::default(),
        }
    }
}

pub const fn default_video_upscale_factor() -> u8 {
    2
}

pub fn default_video_upscale_engine() -> String {
    "seedvr2".to_owned()
}

pub fn default_video_upscale_model() -> String {
    "seedvr2_3b".to_owned()
}
