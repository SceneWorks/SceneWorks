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
