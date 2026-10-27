//! FLUX 3 Image Klein package selection (sc-26241).
//!
//! The distilled `flux3_klein` route loads exactly one [`Flux3Tier`] package, every one of them a
//! distinct artifact rather than a load-time transform of another:
//!
//! * **bf16** — the dense package at the pinned snapshot root (`config.json` + `model.safetensors`),
//!   with the bundled Qwen3-VL encoder and autoencoder its config names. `quantize = None`.
//! * **q4 / q8** — native packages CONVERTED ON THIS MACHINE from that dense package (the catalog's
//!   `localDerivation` rows): Candle GGML off-Mac, MLX affine on macOS. `quantize = Some(Q4 | Q8)`.
//! * **fp8r / nvfp4** — the publisher packages at `variants/<tier>` inside the same snapshot,
//!   loaded AS SHIPPED by the Candle provider on CUDA past their device floor.
//!   `quantize = Some(Fp8Rowwise | Nvfp4)`.
//!
//! Every non-dense package names the encoder and autoencoder by repository-relative path; they live
//! at the snapshot root, so they ride as explicit `text_encoder` / `autoencoder` components.
//!
//! The tier is selected ONLY by the request ([`flux3_requested_tier`]): an absent tier is bf16,
//! never the bits-ladder `q8` default the generic resolvers fall back to. Nothing here downtiers,
//! dequantizes or substitutes another package; every shape it cannot honor is a typed refusal.

use sceneworks_core::contracts::JsonObject;
use sceneworks_core::image_request::{
    flux3_requested_tier, image_quant_tier_refusal, Flux3Tier, FLUX3_TIERED_MODEL_IDS,
};

use super::{WorkerError, WorkerResult};

/// The native image lane a request is about to run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ImageLane {
    Mlx,
    Candle,
}

impl ImageLane {
    /// The lane this worker build runs image jobs on.
    pub(super) const fn current() -> Self {
        if cfg!(target_os = "macos") {
            Self::Mlx
        } else {
            Self::Candle
        }
    }

    const fn serves(self, tier: Flux3Tier) -> bool {
        match self {
            Self::Mlx => tier.served_by_mlx(),
            Self::Candle => tier.served_by_candle(),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Mlx => "MLX",
            Self::Candle => "Candle",
        }
    }
}

/// Worker trust-boundary check of named precision tiers for EVERY image request (raw queue writes
/// and stored jobs never passed the API validator): a malformed or conflicting FLUX 3 tier, FP8R on
/// a model with no package, and a FLUX 3 package this lane does not serve are all refused by name.
pub(super) fn validate_image_quant_tier_request(
    model: &str,
    advanced: &JsonObject,
    lane: ImageLane,
) -> WorkerResult<()> {
    if let Some(reason) = image_quant_tier_refusal(model, advanced) {
        return Err(WorkerError::InvalidPayload(reason));
    }
    flux3_requested_tier_for_lane(model, advanced, lane).map(|_| ())
}

/// The FLUX 3 package a request names, or `None` for any other model. A package this lane does not
/// serve is refused: the MLX provider has bf16/q4/q8 only, the CUDA formats are Candle's.
pub(super) fn flux3_requested_tier_for_lane(
    model: &str,
    advanced: &JsonObject,
    lane: ImageLane,
) -> WorkerResult<Option<Flux3Tier>> {
    if !FLUX3_TIERED_MODEL_IDS.contains(&model) {
        return Ok(None);
    }
    let tier = flux3_requested_tier(advanced)
        .map_err(|reason| WorkerError::InvalidPayload(format!("{model}: {reason}")))?;
    if !lane.serves(tier) {
        return Err(WorkerError::InvalidPayload(format!(
            "{model}: the {} package is a CUDA format served by the Candle backend; the {} \
             backend serves bf16, q4 and q8. Run it on a Candle CUDA worker, or select one of those",
            tier.key(),
            lane.name()
        )));
    }
    Ok(Some(tier))
}

#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) use native::*;

#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod native {
    use std::path::{Component, Path, PathBuf};

    use gen_core::{LoadSpec, Quant, WeightsSource};
    use sceneworks_core::contracts::JsonObject;
    use sceneworks_core::image_request::Flux3Tier;
    use sceneworks_core::model_artifacts::artifact_selection::{
        derived_snapshot_state, local_derivation, local_derivation_snapshot_dir,
        DerivedSnapshotState,
    };
    use serde_json::Value;

    use super::{ImageLane, WorkerError, WorkerResult};

    /// `LoadSpec.quantize` for a FLUX 3 package.
    pub(in crate::image_jobs) const fn flux3_quant(tier: Flux3Tier) -> Option<Quant> {
        match tier {
            Flux3Tier::Bf16 => None,
            Flux3Tier::Q4 => Some(Quant::Q4),
            Flux3Tier::Q8 => Some(Quant::Q8),
            Flux3Tier::Fp8Rowwise => Some(Quant::Fp8Rowwise),
            Flux3Tier::Nvfp4 => Some(Quant::Nvfp4),
        }
    }

    /// The resolved, existence-checked load of one FLUX 3 package. The generic lanes consume it
    /// at their five seams: the weights directory, the gate/selector tier, the load quant + recipe
    /// bits, the (skipped) capability downtier, and the root encoder/autoencoder components.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct Flux3LoadPlan {
        pub(crate) tier: Flux3Tier,
        /// The package directory: the snapshot root for bf16, else the variant/derived package.
        pub(crate) weights_dir: PathBuf,
        /// The root encoder and autoencoder, attached as explicit components for every non-dense
        /// package (`None` for bf16, which reads them from its own root).
        pub(crate) components: Option<(PathBuf, PathBuf)>,
    }

    impl Flux3LoadPlan {
        pub(crate) const fn quant(&self) -> Option<Quant> {
            flux3_quant(self.tier)
        }

        /// The `mlxQuantize` bits the recipe records: q4/q8 only, never an alias for FP8R/NVFP4.
        pub(crate) const fn recipe_bits(&self) -> Option<i64> {
            self.tier.recipe_bits()
        }

        /// The `(quant, recipe bits)` pair the generic lanes thread through.
        pub(crate) const fn quant_and_bits(&self) -> (Option<Quant>, Option<i64>) {
            (self.quant(), self.recipe_bits())
        }

        /// Attach the root encoder and autoencoder the package config names, so the selector, fit
        /// gate and provider load all inspect the same artifact set. Identity for bf16.
        pub(crate) fn attach_components(&self, spec: LoadSpec) -> LoadSpec {
            match &self.components {
                None => spec,
                Some((text_encoder, autoencoder)) => spec
                    .with_component(
                        gen_core::flux3::TEXT_ENCODER_COMPONENT,
                        WeightsSource::Dir(text_encoder.clone()),
                    )
                    .with_component(
                        gen_core::flux3::AUTOENCODER_COMPONENT,
                        WeightsSource::File(autoencoder.clone()),
                    ),
            }
        }
    }

    /// Refuse a package the linked provider does not advertise, before any weight is touched.
    pub(in crate::image_jobs) fn require_advertised_flux3_tier(
        model: &str,
        tier: Flux3Tier,
        supported_quants: &[Quant],
    ) -> WorkerResult<()> {
        match flux3_quant(tier) {
            Some(quant) if !supported_quants.contains(&quant) => {
                Err(WorkerError::InvalidPayload(format!(
                    "{model}: the {} package needs a provider that advertises it, and this \
                     worker's linked FLUX 3 provider advertises {supported_quants:?} (FP8R and \
                     NVFP4 are served by CUDA builds only)",
                    tier.key()
                )))
            }
            _ => Ok(()),
        }
    }

    /// Refuse a publisher package on a device below its floor, using the inference rule
    /// (`DitFormat::check_device`: FP8R sm_89+, NVFP4 sm_120+). `compute_cap` is the probed
    /// `major.minor`; `None` (no CUDA probe) refuses every publisher package. Never a dense
    /// fallback. Candle-only: the MLX lane never resolves a publisher package.
    #[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
    pub(in crate::image_jobs) fn require_flux3_device_floor(
        model: &str,
        tier: Flux3Tier,
        compute_cap: Option<f32>,
    ) -> WorkerResult<()> {
        use gen_core::flux3::quantization::{DitFormat, QuantDevice};
        let format = match tier {
            Flux3Tier::Fp8Rowwise => DitFormat::Fp8Rowwise,
            Flux3Tier::Nvfp4 => DitFormat::Nvfp4,
            Flux3Tier::Bf16 | Flux3Tier::Q4 | Flux3Tier::Q8 => return Ok(()),
        };
        let device = match compute_cap {
            Some(cap) => {
                // `nvidia-smi` reports `major.minor` as one decimal (8.9, 12.0).
                let tenths = (cap * 10.0).round() as u32;
                QuantDevice::Cuda {
                    major: tenths / 10,
                    minor: tenths % 10,
                }
            }
            None => QuantDevice::Cpu,
        };
        format.check_device(device).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "{model}: the {} package cannot run on this worker's device: {error}",
                tier.key()
            ))
        })
    }

    /// Resolve the package of `tier` for a request on `lane`. `snapshot_root` is the pinned
    /// snapshot (the bf16 package); `data_dir` + `manifest_entry` locate a derived q4/q8 package.
    pub(in crate::image_jobs) fn plan_flux3_load(
        tier: Flux3Tier,
        lane: ImageLane,
        snapshot_root: &Path,
        data_dir: &Path,
        manifest_entry: &JsonObject,
    ) -> WorkerResult<Flux3LoadPlan> {
        if tier == Flux3Tier::Bf16 {
            return Ok(Flux3LoadPlan {
                tier,
                weights_dir: snapshot_root.to_path_buf(),
                components: None,
            });
        }
        let package_dir = if tier.is_publisher_variant() {
            snapshot_root.join("variants").join(tier.key())
        } else {
            flux3_derived_tier_dir(data_dir, manifest_entry, tier, lane_platform(lane))?
        };
        let package = resolve_flux3_package(tier, snapshot_root, &package_dir)?;
        Ok(Flux3LoadPlan {
            tier,
            weights_dir: package.0,
            components: Some((package.1, package.2)),
        })
    }

    /// The manifest platform a lane's derived packages are declared for.
    fn lane_platform(lane: ImageLane) -> &'static str {
        match lane {
            ImageLane::Mlx => "macos",
            ImageLane::Candle => {
                if cfg!(target_os = "windows") {
                    "windows"
                } else {
                    "linux"
                }
            }
        }
    }

    /// The verified derived package of `tier` on `platform`: the catalog row's
    /// `localDerivation` location, which must match its pinned weights. A tier that is not derived
    /// yet (or no longer verifies) is a typed "install the tier" refusal, never another package.
    pub(in crate::image_jobs) fn flux3_derived_tier_dir(
        data_dir: &Path,
        manifest_entry: &JsonObject,
        tier: Flux3Tier,
        platform: &str,
    ) -> WorkerResult<PathBuf> {
        let model_id = manifest_entry
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("flux3_klein");
        let rows: Vec<&Value> = manifest_entry
            .get("downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|row| {
                row.get("variant").and_then(Value::as_str) == Some(tier.key())
                    && row.get("localDerivation").is_some()
                    && row
                        .get("platforms")
                        .and_then(Value::as_array)
                        .is_none_or(|platforms| {
                            platforms.iter().any(|p| p.as_str() == Some(platform))
                        })
            })
            .collect();
        let [row] = rows.as_slice() else {
            if rows.is_empty() && model_id == "flux3_klein_base" {
                return Err(WorkerError::InvalidPayload(format!(
                    "{model_id}: the {} package is unavailable without verified base conversion \
                     output size and SHA-256 catalog pins. \
                     Distilled pins and dense fallback are not accepted.",
                    tier.key()
                )));
            }
            return Err(WorkerError::InvalidPayload(format!(
                "{model_id}: the catalog declares {} {} derivations for {platform}; exactly one \
                 is required",
                rows.len(),
                tier.key()
            )));
        };
        let derivation = local_derivation(row).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{model_id}: the {} derivation row is malformed",
                tier.key()
            ))
        })?;
        let dir = local_derivation_snapshot_dir(data_dir, model_id, tier.key(), &derivation)
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{model_id}: the {} derivation names no plain path",
                    tier.key()
                ))
            })?;
        match derived_snapshot_state(&dir, &derivation) {
            DerivedSnapshotState::Verified => Ok(dir),
            DerivedSnapshotState::Absent => Err(WorkerError::InvalidPayload(format!(
                "{model_id}: the {} package is not installed; install the {} tier (it is \
                 converted on this machine from the pinned original)",
                tier.key(),
                tier.key()
            ))),
            DerivedSnapshotState::Invalid(why) => Err(WorkerError::InvalidPayload(format!(
                "{model_id}: the installed {} package does not match its pinned conversion \
                 ({why}); reinstall the {} tier",
                tier.key(),
                tier.key()
            ))),
        }
    }

    /// Existence- and containment-check one non-dense package: `config.json` declares the tier's
    /// quantization marker, `model.safetensors` is present, and the encoder / autoencoder its
    /// config names resolve INSIDE the snapshot. Returns `(package, text encoder, autoencoder)`.
    ///
    /// Containment is checked on canonical paths: a publisher variant directory, and the encoder
    /// directory, must not be a link out of the snapshot (a Hub snapshot links each FILE to its
    /// blob, which is followed; a linked DIRECTORY would let the package escape and is refused).
    pub(in crate::image_jobs) fn resolve_flux3_package(
        tier: Flux3Tier,
        snapshot_root: &Path,
        package_dir: &Path,
    ) -> WorkerResult<(PathBuf, PathBuf, PathBuf)> {
        let incomplete = |what: &str, path: &Path| {
            WorkerError::InvalidPayload(format!(
                "flux3_klein: the {} package is incomplete — {what} {} is missing; reinstall the \
                 tier (it is never substituted by another)",
                tier.key(),
                path.display()
            ))
        };
        let canonical = |path: &Path, what: &str| {
            std::fs::canonicalize(path).map_err(|_| incomplete(what, path))
        };
        let root = canonical(snapshot_root, "the pinned snapshot")?;
        let package = canonical(package_dir, "the package directory")?;
        if tier.is_publisher_variant() && !package.starts_with(&root) {
            return Err(WorkerError::InvalidPayload(format!(
                "flux3_klein: the {} package {} resolves outside the pinned snapshot ({}); a \
                 linked package directory is refused",
                tier.key(),
                package_dir.display(),
                package.display()
            )));
        }
        for name in ["config.json", "model.safetensors"] {
            let allowed_root = if tier.is_publisher_variant() {
                snapshot_root
            } else {
                package_dir
            };
            crate::flux3_snapshot_member(allowed_root, &package.join(name), false)
                .map_err(WorkerError::InvalidPayload)?;
        }
        let config_path = package.join("config.json");
        if !config_path.is_file() {
            return Err(incomplete("the package config", &config_path));
        }
        let weights = package.join("model.safetensors");
        if !weights.is_file() {
            return Err(incomplete("the transformer", &weights));
        }
        let config: Value = std::fs::read(&config_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "flux3_klein: the {} package config {} is not readable JSON",
                    tier.key(),
                    config_path.display()
                ))
            })?;
        let marker = config.get("quantization");
        let marker_ok = if tier.is_publisher_variant() {
            marker.and_then(Value::as_str) == Some(tier.key())
        } else {
            // A native package names its exact storage object; the provider verifies the receipt.
            marker.is_some_and(Value::is_object)
        };
        if !marker_ok {
            return Err(WorkerError::InvalidPayload(format!(
                "flux3_klein: {} declares quantization {marker:?}, which is not the {} package",
                config_path.display(),
                tier.key()
            )));
        }
        let relative = |section: &str| -> WorkerResult<PathBuf> {
            config
                .get(section)
                .and_then(|section| section.get("path"))
                .and_then(Value::as_str)
                .filter(|path| is_plain_relative(path))
                .map(|path| root.join(path))
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "flux3_klein: {} has no repository-relative {section}.path",
                        config_path.display()
                    ))
                })
        };
        let text_encoder = relative("text_encoder")?;
        if !text_encoder.is_dir() {
            return Err(incomplete("the bundled text encoder", &text_encoder));
        }
        if !canonical(&text_encoder, "the bundled text encoder")?.starts_with(&root) {
            return Err(WorkerError::InvalidPayload(format!(
                "flux3_klein: the text encoder {} resolves outside the pinned snapshot",
                text_encoder.display()
            )));
        }
        let autoencoder = relative("vae")?;
        if !autoencoder.is_file() {
            return Err(incomplete("the autoencoder", &autoencoder));
        }
        crate::flux3_snapshot_member(snapshot_root, &autoencoder, false)
            .map_err(WorkerError::InvalidPayload)?;
        Ok((package, text_encoder, autoencoder))
    }

    /// A config path that stays inside the snapshot: relative, no parent or root components.
    fn is_plain_relative(path: &str) -> bool {
        let path = Path::new(path);
        !path.as_os_str().is_empty()
            && path
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
    }
}

/// The Candle lane's FLUX 3 load: the requested package, refused unless the linked provider
/// advertises it and this device clears its floor, resolved against the snapshot at
/// `snapshot_root`. `None` for every other model.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn candle_flux3_plan(
    request: &super::ImageRequest,
    settings: &super::Settings,
    snapshot_root: &std::path::Path,
    supported_quants: &[gen_core::Quant],
) -> WorkerResult<Option<Flux3LoadPlan>> {
    let Some(tier) =
        flux3_requested_tier_for_lane(&request.model, &request.advanced, ImageLane::Candle)?
    else {
        return Ok(None);
    };
    require_advertised_flux3_tier(&request.model, tier, supported_quants)?;
    require_flux3_device_floor(&request.model, tier, crate::gpu::cached_compute_cap())?;
    plan_flux3_load(
        tier,
        ImageLane::Candle,
        snapshot_root,
        &settings.data_dir,
        &request.model_manifest_entry,
    )
    .map(Some)
}

/// The MLX lane's FLUX 3 load (bf16 / affine q4 / q8). `None` for every other model.
#[cfg(target_os = "macos")]
pub(crate) fn mlx_flux3_plan(
    request: &super::ImageRequest,
    settings: &super::Settings,
    snapshot_root: &std::path::Path,
    supported_quants: &[gen_core::Quant],
) -> WorkerResult<Option<Flux3LoadPlan>> {
    let Some(tier) =
        flux3_requested_tier_for_lane(&request.model, &request.advanced, ImageLane::Mlx)?
    else {
        return Ok(None);
    };
    require_advertised_flux3_tier(&request.model, tier, supported_quants)?;
    plan_flux3_load(
        tier,
        ImageLane::Mlx,
        snapshot_root,
        &settings.data_dir,
        &request.model_manifest_entry,
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{LoadSpec, Quant, WeightsSource};
    use serde_json::{json, Value};
    use std::path::Path;

    fn advanced(value: Value) -> JsonObject {
        value.as_object().cloned().expect("object")
    }

    #[test]
    fn each_lane_resolves_only_the_packages_its_provider_serves() {
        for (value, tier) in [
            (json!({}), Flux3Tier::Bf16),
            (json!({"mlxQuantize": 0}), Flux3Tier::Bf16),
            (json!({"mlxQuantize": 4}), Flux3Tier::Q4),
            (
                json!({"mlxQuantize": 8, "mlxQuantizeExplicit": true}),
                Flux3Tier::Q8,
            ),
        ] {
            for lane in [ImageLane::Mlx, ImageLane::Candle] {
                assert_eq!(
                    flux3_requested_tier_for_lane("flux3_klein", &advanced(value.clone()), lane)
                        .unwrap(),
                    Some(tier),
                    "{value} on {lane:?}"
                );
            }
        }
        for value in [json!({"quantTier": "fp8r"}), json!({"quantTier": "nvfp4"})] {
            let tier = flux3_requested_tier_for_lane(
                "flux3_klein",
                &advanced(value.clone()),
                ImageLane::Candle,
            )
            .unwrap()
            .unwrap();
            assert!(tier.is_publisher_variant(), "{value}");
            let error = flux3_requested_tier_for_lane(
                "flux3_klein",
                &advanced(value.clone()),
                ImageLane::Mlx,
            )
            .expect_err("CUDA format on MLX");
            assert!(format!("{error:?}").contains("MLX"), "{error:?}");
        }
        assert_eq!(
            flux3_requested_tier_for_lane(
                "z_image",
                &advanced(json!({"mlxQuantize": 8})),
                ImageLane::Mlx
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn the_worker_boundary_refuses_malformed_tiers_and_fp8r_off_flux3() {
        let fp8r = advanced(json!({"quantTier": "fp8r"}));
        assert!(validate_image_quant_tier_request("flux3_klein", &fp8r, ImageLane::Candle).is_ok());
        assert!(validate_image_quant_tier_request("flux3_klein", &fp8r, ImageLane::Mlx).is_err());
        for model in ["z_image", "flux2_klein_9b"] {
            for lane in [ImageLane::Candle, ImageLane::Mlx] {
                let error = validate_image_quant_tier_request(model, &fp8r, lane)
                    .expect_err("non-flux3 fp8r refused");
                assert!(format!("{error:?}").contains(model), "{error:?}");
            }
        }
        for value in [
            json!({"mlxQuantize": 6}),
            json!({"quantTier": "fp8r", "mlxQuantize": 8}),
            json!({"convRot": true}),
        ] {
            assert!(
                validate_image_quant_tier_request(
                    "flux3_klein",
                    &advanced(value.clone()),
                    ImageLane::Candle
                )
                .is_err(),
                "{value}"
            );
        }
        // Requests without a FLUX 3 tier are untouched on both lanes.
        for lane in [ImageLane::Candle, ImageLane::Mlx] {
            assert!(validate_image_quant_tier_request(
                "z_image",
                &advanced(json!({"mlxQuantize": 8})),
                lane
            )
            .is_ok());
        }
    }

    #[test]
    fn every_package_maps_to_its_inference_quant_by_identity() {
        assert_eq!(flux3_quant(Flux3Tier::Bf16), None);
        assert_eq!(flux3_quant(Flux3Tier::Q4), Some(Quant::Q4));
        assert_eq!(flux3_quant(Flux3Tier::Q8), Some(Quant::Q8));
        assert_eq!(flux3_quant(Flux3Tier::Fp8Rowwise), Some(Quant::Fp8Rowwise));
        assert_eq!(flux3_quant(Flux3Tier::Nvfp4), Some(Quant::Nvfp4));
    }

    #[test]
    fn a_package_needs_a_provider_that_advertises_it() {
        let cuda = [Quant::Q4, Quant::Q8, Quant::Fp8Rowwise, Quant::Nvfp4];
        let cpu = [Quant::Q4, Quant::Q8];
        for tier in Flux3Tier::ALL {
            assert!(require_advertised_flux3_tier("flux3_klein", tier, &cuda).is_ok());
        }
        for tier in [Flux3Tier::Fp8Rowwise, Flux3Tier::Nvfp4] {
            let error = require_advertised_flux3_tier("flux3_klein", tier, &cpu)
                .expect_err("CPU build has no publisher kernels");
            assert!(format!("{error:?}").contains("CUDA"), "{error:?}");
        }
        assert!(require_advertised_flux3_tier("flux3_klein", Flux3Tier::Q4, &cpu).is_ok());
        assert!(require_advertised_flux3_tier("flux3_klein", Flux3Tier::Bf16, &[]).is_ok());
    }

    #[test]
    fn publisher_packages_refuse_a_device_below_the_inference_floor() {
        let ok = |tier, cap| require_flux3_device_floor("flux3_klein", tier, cap).is_ok();
        assert!(ok(Flux3Tier::Fp8Rowwise, Some(8.9)));
        assert!(ok(Flux3Tier::Fp8Rowwise, Some(12.0)));
        assert!(!ok(Flux3Tier::Fp8Rowwise, Some(8.6)));
        assert!(ok(Flux3Tier::Nvfp4, Some(12.0)));
        assert!(!ok(Flux3Tier::Nvfp4, Some(9.0)));
        assert!(!ok(Flux3Tier::Nvfp4, None));
        assert!(!ok(Flux3Tier::Fp8Rowwise, None));
        for tier in [Flux3Tier::Bf16, Flux3Tier::Q4, Flux3Tier::Q8] {
            assert!(ok(tier, None), "{tier:?} runs on every Candle device");
        }
    }

    fn write_root(root: &Path) {
        std::fs::create_dir_all(root.join("text_encoder")).unwrap();
        std::fs::write(root.join("ae.safetensors"), b"x").unwrap();
        std::fs::write(root.join("config.json"), b"{}").unwrap();
    }

    fn write_package(dir: &Path, marker: Value) {
        std::fs::create_dir_all(dir).unwrap();
        let config = json!({
            "quantization": marker,
            "text_encoder": {"path": "text_encoder"},
            "vae": {"path": "ae.safetensors"}
        });
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"x").unwrap();
    }

    fn canonical(path: &Path) -> std::path::PathBuf {
        std::fs::canonicalize(path).unwrap()
    }

    #[test]
    fn publisher_packages_load_from_their_variant_with_root_components() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        write_root(root);
        for tier in [Flux3Tier::Fp8Rowwise, Flux3Tier::Nvfp4] {
            write_package(&root.join("variants").join(tier.key()), json!(tier.key()));
            let plan = plan_flux3_load(
                tier,
                ImageLane::Candle,
                root,
                Path::new("unused"),
                &JsonObject::new(),
            )
            .expect("complete package");
            assert_eq!(
                plan.weights_dir,
                canonical(&root.join("variants").join(tier.key()))
            );
            let (te, ae) = plan.components.clone().expect("non-dense components");
            assert_eq!(te, canonical(root).join("text_encoder"));
            assert_eq!(ae, canonical(root).join("ae.safetensors"));
            let spec = plan.attach_components(
                LoadSpec::new(WeightsSource::Dir(plan.weights_dir.clone()))
                    .with_quant(plan.quant().unwrap()),
            );
            assert_eq!(spec.quantize, flux3_quant(tier));
            assert_eq!(
                spec.components.get(gen_core::flux3::TEXT_ENCODER_COMPONENT),
                Some(&WeightsSource::Dir(te))
            );
            assert_eq!(
                spec.components.get(gen_core::flux3::AUTOENCODER_COMPONENT),
                Some(&WeightsSource::File(ae))
            );
            assert_eq!(
                plan.recipe_bits(),
                None,
                "never aliased onto the bits ladder"
            );
        }
    }

    #[test]
    fn the_dense_package_is_the_snapshot_root_with_no_overrides() {
        let temp = tempfile::tempdir().unwrap();
        let plan = plan_flux3_load(
            Flux3Tier::Bf16,
            ImageLane::Mlx,
            temp.path(),
            Path::new("unused"),
            &JsonObject::new(),
        )
        .unwrap();
        assert_eq!(plan.weights_dir, temp.path());
        assert_eq!(plan.components, None);
        assert_eq!(plan.quant_and_bits(), (None, None));
        let spec = LoadSpec::new(WeightsSource::Dir(plan.weights_dir.clone()));
        assert!(plan.attach_components(spec).components.is_empty());
    }

    #[test]
    fn an_incomplete_or_foreign_publisher_package_is_refused_by_path() {
        let temp = tempfile::tempdir().unwrap();
        write_root(temp.path());
        let plan = |root: &Path| {
            plan_flux3_load(
                Flux3Tier::Fp8Rowwise,
                ImageLane::Candle,
                root,
                Path::new("unused"),
                &JsonObject::new(),
            )
        };
        assert!(format!("{:?}", plan(temp.path()).unwrap_err()).contains("fp8r"));
        // A dense config in the variant slot, or another tier's package, is not this package.
        write_package(&temp.path().join("variants/fp8r"), Value::Null);
        assert!(plan(temp.path()).is_err());
        write_package(&temp.path().join("variants/fp8r"), json!("nvfp4"));
        assert!(plan(temp.path()).is_err());
        // A config path escaping the snapshot is refused.
        let mut config: Value = serde_json::from_slice(
            &std::fs::read(temp.path().join("variants/fp8r/config.json")).unwrap(),
        )
        .unwrap();
        config["quantization"] = json!("fp8r");
        config["vae"]["path"] = json!("../ae.safetensors");
        std::fs::write(
            temp.path().join("variants/fp8r/config.json"),
            config.to_string(),
        )
        .unwrap();
        assert!(plan(temp.path()).is_err());
        // Missing root autoencoder.
        config["vae"]["path"] = json!("ae.safetensors");
        std::fs::write(
            temp.path().join("variants/fp8r/config.json"),
            config.to_string(),
        )
        .unwrap();
        std::fs::remove_file(temp.path().join("ae.safetensors")).unwrap();
        assert!(format!("{:?}", plan(temp.path()).unwrap_err()).contains("ae.safetensors"));
    }

    /// A linked variant DIRECTORY pointing out of the snapshot is refused on its canonical path.
    /// Skipped (with a note) where the host forbids creating directory links (unprivileged
    /// Windows); the Linux CI lanes always run it.
    #[test]
    fn a_variant_directory_linked_out_of_the_snapshot_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("snapshot");
        write_root(&root);
        let outside = temp.path().join("elsewhere");
        write_package(&outside, json!("fp8r"));
        std::fs::create_dir_all(root.join("variants")).unwrap();
        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(&outside, root.join("variants/fp8r"));
        #[cfg(windows)]
        let linked = std::os::windows::fs::symlink_dir(&outside, root.join("variants/fp8r"));
        if let Err(error) = linked {
            eprintln!(
                "directory links unavailable on this host ({error}); containment unexercised"
            );
            return;
        }
        let error = plan_flux3_load(
            Flux3Tier::Fp8Rowwise,
            ImageLane::Candle,
            &root,
            Path::new("unused"),
            &JsonObject::new(),
        )
        .expect_err("escaping package");
        assert!(
            format!("{error:?}").contains("outside the pinned snapshot"),
            "{error:?}"
        );
    }

    fn derived_manifest(sha: &str, bytes: u64) -> JsonObject {
        let row = |platforms: Value, conversion: &str| {
            json!({
                "variant": "q8",
                "platforms": platforms,
                "localDerivation": {
                    "fromVariant": "bf16",
                    "conversion": conversion,
                    "weightsFile": "model.safetensors",
                    "weightsBytes": bytes,
                    "weightsSha256": sha,
                    "reason": "test"
                }
            })
        };
        advanced(json!({
            "id": "flux3_klein",
            "downloads": [
                {"variant": "bf16"},
                row(json!(["windows", "linux"]), "flux3-candle-ggml-q8_0-v1"),
                row(json!(["macos"]), "flux3-mlx-affine-q8-g64-v1"),
            ]
        }))
    }

    #[test]
    fn flux3_base_native_tiers_use_platform_specific_pins_and_never_dense_fallback() {
        use sha2::{Digest, Sha256};
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("snapshot");
        write_root(&root);
        let data = temp.path().join("data");
        let base = crate::tests::builtin_model_entry("flux3_klein_base");
        for tier in [Flux3Tier::Q4, Flux3Tier::Q8] {
            for lane in [ImageLane::Candle, ImageLane::Mlx] {
                // The real catalog now pins genuine local base conversions (sc-26243): with no
                // converted package on disk the load is an actionable not-installed refusal,
                // never a dense fallback.
                let error = plan_flux3_load(tier, lane, &root, &data, base.as_object().unwrap())
                    .unwrap_err();
                let reason = format!("{error:?}");
                assert!(
                    reason.contains("not installed")
                        && reason.contains("converted on this machine")
                        && reason.contains(tier.key()),
                    "{reason}"
                );
                // Without those pins the tier still fails closed on the missing SHA-256 pins.
                let mut unpinned = base.clone();
                for row in unpinned["downloads"].as_array_mut().unwrap() {
                    row.as_object_mut().unwrap().remove("localDerivation");
                }
                let error =
                    plan_flux3_load(tier, lane, &root, &data, unpinned.as_object().unwrap())
                        .unwrap_err();
                let reason = format!("{error:?}");
                assert!(
                    reason.contains("SHA-256")
                        && !reason.contains("403")
                        && reason.contains(tier.key()),
                    "{reason}"
                );
                // SYNTHETIC one-byte pins exercise routing/readiness only. No assertion here
                // claims a real base conversion or supplies production catalog pins.
                let mut manifest = derived_manifest(&format!("{:x}", Sha256::digest(b"x")), 1);
                manifest.insert("id".into(), json!("flux3_klein_base"));
                for row in manifest
                    .get_mut("downloads")
                    .unwrap()
                    .as_array_mut()
                    .unwrap()
                {
                    if row["variant"] == "q8" {
                        row["variant"] = json!(tier.key());
                        let id = row["localDerivation"]["conversion"]
                            .as_str()
                            .unwrap()
                            .to_owned();
                        row["localDerivation"]["conversion"] = json!(if tier == Flux3Tier::Q4 {
                            id.replace("q8_0", "q4_1").replace("q8-", "q4-")
                        } else {
                            id
                        });
                    }
                }
                let conversion = match (lane, tier) {
                    (ImageLane::Candle, Flux3Tier::Q4) => "flux3-candle-ggml-q4_1-v1",
                    (ImageLane::Candle, _) => "flux3-candle-ggml-q8_0-v1",
                    (ImageLane::Mlx, Flux3Tier::Q4) => "flux3-mlx-affine-q4-g64-v1",
                    (ImageLane::Mlx, _) => "flux3-mlx-affine-q8-g64-v1",
                };
                let dir = data
                    .join("models/derived/flux3_klein_base")
                    .join(tier.key())
                    .join(conversion);
                assert!(plan_flux3_load(tier, lane, &root, &data, &manifest).is_err());
                write_package(&dir, json!({"format":conversion}));
                let plan = plan_flux3_load(tier, lane, &root, &data, &manifest).unwrap();
                assert_eq!(plan.weights_dir, canonical(&dir));
                assert_eq!(plan.quant(), super::native::flux3_quant(tier));
                assert!(plan.components.is_some());
                std::fs::write(dir.join("model.safetensors"), b"tampered").unwrap();
                assert!(plan_flux3_load(tier, lane, &root, &data, &manifest).is_err());
            }
        }
        for tier in [Flux3Tier::Fp8Rowwise, Flux3Tier::Nvfp4] {
            let dir = root.join("variants").join(tier.key());
            write_package(&dir, json!(tier.key()));
            let plan = plan_flux3_load(
                tier,
                ImageLane::Candle,
                &root,
                &data,
                base.as_object().unwrap(),
            )
            .unwrap();
            assert_eq!(plan.weights_dir, canonical(&dir));
            assert_eq!(plan.quant(), super::native::flux3_quant(tier));
        }
    }

    #[test]
    fn a_derived_package_loads_only_once_it_matches_its_platform_pins() {
        use sha2::{Digest, Sha256};
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("snapshot");
        write_root(&root);
        let data = temp.path().join("data");
        let weights = b"x";
        let sha = format!("{:x}", Sha256::digest(weights));
        let manifest = derived_manifest(&sha, weights.len() as u64);
        let candle_dir = data.join("models/derived/flux3_klein/q8/flux3-candle-ggml-q8_0-v1");
        // Not converted yet: an install instruction, never the dense package.
        let error = plan_flux3_load(Flux3Tier::Q8, ImageLane::Candle, &root, &data, &manifest)
            .expect_err("not installed");
        assert!(
            format!("{error:?}").contains("install the q8 tier"),
            "{error:?}"
        );
        write_package(&candle_dir, json!({"format": "ggml_q8_0"}));
        let plan = plan_flux3_load(Flux3Tier::Q8, ImageLane::Candle, &root, &data, &manifest)
            .expect("verified derivation");
        assert_eq!(plan.weights_dir, canonical(&candle_dir));
        assert_eq!(plan.quant_and_bits(), (Some(Quant::Q8), Some(8)));
        assert!(plan.components.is_some());
        // The MLX lane resolves ITS conversion, which is a different (absent) package.
        assert!(plan_flux3_load(Flux3Tier::Q8, ImageLane::Mlx, &root, &data, &manifest).is_err());
        // Bytes that no longer match the pins are refused, not loaded.
        std::fs::write(candle_dir.join("model.safetensors"), b"yy").unwrap();
        let error = plan_flux3_load(Flux3Tier::Q8, ImageLane::Candle, &root, &data, &manifest)
            .expect_err("tampered derivation");
        assert!(format!("{error:?}").contains("reinstall"), "{error:?}");
    }

    /// The text from `declaration` to the end of its body.
    fn function_body(source: &'static str, declaration: &str) -> &'static str {
        let body = &source[source.find(declaration).expect(declaration)..];
        let mut depth = 0_usize;
        for (offset, byte) in body.bytes().enumerate().skip(body.find('{').expect("body")) {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &body[..=offset];
                    }
                }
                _ => {}
            }
        }
        panic!("unbalanced body for {declaration}");
    }

    fn position(body: &str, needle: &str, lane: &str) -> usize {
        body.find(needle)
            .unwrap_or_else(|| panic!("{lane}: the FLUX 3 plan seam `{needle}` is gone"))
    }

    /// sc-26241: the generic Candle lane consumes the FLUX 3 plan at all five seams, in order.
    /// These lanes load real weights on a GPU, so their wiring is what can be asserted here: deleting
    /// any one seam (or moving it past the step that reads it) turns this red.
    #[test]
    fn the_candle_lane_consumes_every_flux3_plan_seam_in_order() {
        let body = function_body(
            include_str!("base.rs"),
            "async fn generate_candle_stream_using(",
        );
        let lane = "generate_candle_stream_using";
        let plan = position(
            body,
            "let flux3_plan = flux3_precision::candle_flux3_plan(",
            lane,
        );
        let dir = position(body, "weights_dir = plan.weights_dir.clone();", lane);
        let gate_key = position(body, "let mut tier = candle_resolved_tier_key(", lane);
        let tier = position(body, "tier = plan.tier.key();", lane);
        let evidence = position(body, "validate_candle_tier_memory_evidence(", lane);
        let downtier = position(
            body,
            "if convrot.is_none() && !explicit_pick && flux3_plan.is_none() {",
            lane,
        );
        let quant = position(body, "Some(plan) => plan.quant_and_bits(),", lane);
        let spec = position(body, "let mut shared_contract_spec = load_spec(", lane);
        let components = position(
            body,
            "shared_contract_spec = plan.attach_components(shared_contract_spec);",
            lane,
        );
        let encoder = position(body, "attach_manifest_text_encoder(", lane);
        assert!(
            plan < dir && dir < gate_key,
            "the package directory is the gated directory"
        );
        assert!(
            gate_key < tier && tier < evidence,
            "the plan tier replaces the bits key"
        );
        assert!(
            tier < downtier && downtier < quant,
            "a FLUX 3 package is never downtiered"
        );
        assert!(quant < spec && spec < components && components < encoder);
    }

    /// The MLX lane's three FLUX 3 seams: the package directory, the load quant + recipe bits and
    /// tier identity of every prepared candidate, and the root components.
    #[test]
    fn the_mlx_lane_consumes_every_flux3_plan_seam() {
        let body = function_body(include_str!("base.rs"), "async fn generate_stream_using(");
        let lane = "generate_stream_using";
        let plan = position(
            body,
            "let flux3_plan = flux3_precision::mlx_flux3_plan(",
            lane,
        );
        let dir = position(
            body,
            ".map_or(weights_dir, |plan| plan.weights_dir.clone());",
            lane,
        );
        let quant = position(body, "Some(plan) => plan.quant_and_bits(),", lane);
        let tier = position(body, ".map(|plan| plan.tier.key())", lane);
        let components = position(body, "spec = plan.attach_components(spec);", lane);
        let reconcile = position(body, ".map(|plan| plan.quant_and_bits())", lane);
        assert!(plan < dir && dir < quant && quant < tier && tier < components);
        assert!(
            components < reconcile,
            "the fixed pair skips tier reconciliation"
        );
    }
}
