//! FLUX 3 Image Klein precision selection (sc-26241).
//!
//! The distilled `flux3_klein` route loads exactly one of two weight sets, both inside its single
//! pinned whole-repository snapshot:
//!
//! * **bf16** — the dense package at the snapshot root (`config.json` + `model.safetensors`), with
//!   the bundled Qwen3-VL encoder and autoencoder its config names. `LoadSpec.quantize` is `None`.
//! * **fp8r** — the publisher FP8 rowwise package at `variants/fp8r`, loaded AS SHIPPED by the
//!   Candle provider on CUDA (`LoadSpec.quantize = Some(Quant::Fp8Rowwise)`). Its config names the
//!   same encoder and autoencoder by repository-relative path; they live at the snapshot root, not
//!   inside the variant directory, so they ride as explicit `text_encoder` / `autoencoder`
//!   components.
//!
//! Precision is selected ONLY by the request: an absent tier is bf16, never the bits-ladder `q8`
//! default the generic resolvers fall back to (the descriptor advertising q4/q8 would otherwise turn
//! a plain request into a re-quantization the dense package refuses). Nothing here downtiers,
//! dequantizes or substitutes another precision; every shape it cannot honor is a typed refusal.

// Only the trust-boundary check runs on every lane; the package resolution is the Candle lane's
// (the MLX lane refuses FP8R and loads FLUX 3 dense via `fixed_mlx_artifact_quant`).
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
use std::path::{Path, PathBuf};

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
use gen_core::{LoadSpec, Quant, WeightsSource};
use sceneworks_core::contracts::JsonObject;
use sceneworks_core::image_request::{
    fp8r_quant_tier_refusal, requests_fp8r_tier, FP8R_QUANT_TIER,
};
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
use serde_json::Value;

use super::{WorkerError, WorkerResult};

/// The distilled FLUX 3 catalog id this module governs.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) const FLUX3_KLEIN_MODEL_ID: &str = "flux3_klein";

/// The FP8R package directory, relative to the pinned snapshot root.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) const FLUX3_FP8R_VARIANT_DIR: &str = "variants/fp8r";

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
}

/// The weight set a `flux3_klein` request names.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Flux3Precision {
    Bf16,
    Fp8Rowwise,
}

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
impl Flux3Precision {
    /// The tier key the fit gate, shared memory selector and telemetry name this load by — the
    /// tier IDENTITY, so FP8R is never sized or labelled as q8.
    pub(super) const fn tier(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8Rowwise => FP8R_QUANT_TIER,
        }
    }

    /// `LoadSpec.quantize` for this precision.
    pub(super) const fn quant(self) -> Option<Quant> {
        match self {
            Self::Bf16 => None,
            Self::Fp8Rowwise => Some(Quant::Fp8Rowwise),
        }
    }
}

/// Worker trust-boundary check of the FP8R tier label for EVERY image request (raw queue writes
/// and stored jobs never passed the API validator). Refuses a model with no FP8R package, a second
/// precision carrier, and the MLX lane — no MLX descriptor advertises the tier.
pub(super) fn validate_image_fp8r_request(
    model: &str,
    advanced: &JsonObject,
    lane: ImageLane,
) -> WorkerResult<()> {
    if let Some(reason) = fp8r_quant_tier_refusal(model, advanced) {
        return Err(WorkerError::InvalidPayload(reason));
    }
    if requests_fp8r_tier(advanced) && lane == ImageLane::Mlx {
        return Err(WorkerError::InvalidPayload(format!(
            "{model}: advanced.quantTier \"{FP8R_QUANT_TIER}\" is a CUDA format served by the \
             Candle backend; the MLX backend has no FP8R tier. Run it on a Candle CUDA worker, or \
             re-submit without the tier for the dense bf16 package"
        )));
    }
    Ok(())
}

/// The precision a `flux3_klein` request names, or `None` for any other model. Every other
/// precision carrier (`mlxQuantize`, another named tier, ConvRot) is refused by name: the package
/// set is bf16 + fp8r, and routing admits nothing else, so reaching here with one is a raw or
/// stale payload that must not be reinterpreted.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn flux3_requested_precision(
    model: &str,
    advanced: &JsonObject,
) -> WorkerResult<Option<Flux3Precision>> {
    if model != FLUX3_KLEIN_MODEL_ID {
        return Ok(None);
    }
    if let Some(reason) = fp8r_quant_tier_refusal(model, advanced) {
        return Err(WorkerError::InvalidPayload(reason));
    }
    if requests_fp8r_tier(advanced) {
        return Ok(Some(Flux3Precision::Fp8Rowwise));
    }
    match advanced.get("quantTier") {
        None | Some(Value::Null) => {}
        Some(Value::String(tier))
            if tier.trim().is_empty() || tier.trim().eq_ignore_ascii_case("bf16") => {}
        Some(other) => {
            return Err(WorkerError::InvalidPayload(format!(
                "{model}: advanced.quantTier {other} is not a FLUX 3 Image Klein package; the \
                 pinned repository ships bf16 and \"{FP8R_QUANT_TIER}\" only"
            )))
        }
    }
    if let Some(bits) = advanced.get("mlxQuantize").filter(|bits| !bits.is_null()) {
        return Err(WorkerError::InvalidPayload(format!(
            "{model}: advanced.mlxQuantize {bits} is not a FLUX 3 Image Klein package; select \
             the dense bf16 package or advanced.quantTier \"{FP8R_QUANT_TIER}\""
        )));
    }
    if advanced
        .get("convRot")
        .is_some_and(|convrot| !matches!(convrot, Value::Null | Value::Bool(false)))
    {
        return Err(WorkerError::InvalidPayload(format!(
            "{model}: advanced.convRot is not a FLUX 3 Image Klein package"
        )));
    }
    Ok(Some(Flux3Precision::Bf16))
}

/// Refuse a precision the linked provider does not advertise, before any weight is touched. The
/// Candle FLUX 3 provider advertises `Fp8Rowwise` only in CUDA builds, so a CPU-only worker refuses
/// an FP8R request here by name instead of reaching a loader that would.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn require_advertised_flux3_precision(
    model: &str,
    precision: Flux3Precision,
    supported_quants: &[Quant],
) -> WorkerResult<()> {
    match precision.quant() {
        Some(quant) if !supported_quants.contains(&quant) => {
            Err(WorkerError::InvalidPayload(format!(
                "{model}: advanced.quantTier \"{tier}\" needs a provider that advertises it, and \
                 this worker's linked FLUX 3 provider advertises {supported_quants:?} (FP8R is \
                 served by CUDA builds only)",
                tier = precision.tier()
            )))
        }
        _ => Ok(()),
    }
}

/// The resolved, existence-checked resources of the FP8R package inside the pinned snapshot.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Flux3Fp8rPackage {
    /// `variants/fp8r` — the LoadSpec weights directory.
    pub(super) package_dir: PathBuf,
    /// The bundled Qwen3-VL encoder the variant config names, resolved against the snapshot root.
    pub(super) text_encoder_dir: PathBuf,
    /// The autoencoder file the variant config names, resolved against the snapshot root.
    pub(super) autoencoder_file: PathBuf,
}

/// Resolve the FP8R package from the snapshot `root`. The variant's `config.json` must declare
/// `"quantization": "fp8r"` and name its encoder / autoencoder by repository-relative path; every
/// missing piece is a typed refusal naming the path, never a fallback to the dense package.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn resolve_flux3_fp8r_package(root: &Path) -> WorkerResult<Flux3Fp8rPackage> {
    let package_dir = root.join(FLUX3_FP8R_VARIANT_DIR);
    let incomplete = |what: &str, path: &Path| {
        WorkerError::InvalidPayload(format!(
            "{FLUX3_KLEIN_MODEL_ID}: the FP8R package is incomplete — {what} {} is missing; \
             reinstall the whole pinned repository (the fp8r tier is never substituted by another)",
            path.display()
        ))
    };
    let config_path = package_dir.join("config.json");
    if !config_path.is_file() {
        return Err(incomplete("the variant config", &config_path));
    }
    let weights = package_dir.join("model.safetensors");
    if !weights.is_file() {
        return Err(incomplete("the FP8R transformer", &weights));
    }
    let config: Value = std::fs::read(&config_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{FLUX3_KLEIN_MODEL_ID}: the FP8R variant config {} is not readable JSON",
                config_path.display()
            ))
        })?;
    let quantization = config.get("quantization").and_then(Value::as_str);
    if quantization != Some(FP8R_QUANT_TIER) {
        return Err(WorkerError::InvalidPayload(format!(
            "{FLUX3_KLEIN_MODEL_ID}: {} declares quantization {quantization:?}, not \
             \"{FP8R_QUANT_TIER}\" — it is not the publisher FP8R package",
            config_path.display()
        )));
    }
    let relative = |section: &str| {
        config
            .get(section)
            .and_then(|section| section.get("path"))
            .and_then(Value::as_str)
            .filter(|path| is_plain_relative(path))
            .map(|path| root.join(path))
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{FLUX3_KLEIN_MODEL_ID}: {} has no repository-relative {section}.path",
                    config_path.display()
                ))
            })
    };
    let text_encoder_dir = relative("text_encoder")?;
    if !text_encoder_dir.is_dir() {
        return Err(incomplete("the bundled text encoder", &text_encoder_dir));
    }
    let autoencoder_file = relative("vae")?;
    if !autoencoder_file.is_file() {
        return Err(incomplete("the autoencoder", &autoencoder_file));
    }
    Ok(Flux3Fp8rPackage {
        package_dir,
        text_encoder_dir,
        autoencoder_file,
    })
}

/// Attach the root-level encoder and autoencoder the FP8R variant config names as the provider's
/// explicit `text_encoder` / `autoencoder` components, so the selector, fit gate and provider load
/// all inspect the same artifact set.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn attach_flux3_fp8r_components(spec: LoadSpec, package: &Flux3Fp8rPackage) -> LoadSpec {
    spec.with_component(
        gen_core::flux3::TEXT_ENCODER_COMPONENT,
        WeightsSource::Dir(package.text_encoder_dir.clone()),
    )
    .with_component(
        gen_core::flux3::AUTOENCODER_COMPONENT,
        WeightsSource::File(package.autoencoder_file.clone()),
    )
}

/// A config path that stays inside the snapshot: relative, no parent or root components.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
fn is_plain_relative(path: &str) -> bool {
    let path = Path::new(path);
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn advanced(value: Value) -> JsonObject {
        value.as_object().cloned().expect("object")
    }

    #[test]
    fn an_absent_tier_is_dense_bf16_never_the_q8_bits_default() {
        for value in [
            json!({}),
            json!({"quantTier": null, "mlxQuantize": null}),
            json!({"quantTier": "bf16"}),
            json!({"quantTier": " BF16 "}),
            json!({"convRot": false}),
        ] {
            let precision = flux3_requested_precision("flux3_klein", &advanced(value.clone()))
                .expect("dense request");
            assert_eq!(precision, Some(Flux3Precision::Bf16), "{value}");
            assert_eq!(precision.unwrap().quant(), None);
            assert_eq!(precision.unwrap().tier(), "bf16");
        }
    }

    #[test]
    fn fp8r_maps_to_the_inference_fp8_rowwise_tier_by_identity() {
        let precision =
            flux3_requested_precision("flux3_klein", &advanced(json!({"quantTier": "fp8r"})))
                .expect("fp8r request")
                .expect("flux3");
        assert_eq!(precision, Flux3Precision::Fp8Rowwise);
        assert_eq!(precision.quant(), Some(Quant::Fp8Rowwise));
        assert_eq!(precision.tier(), "fp8r");
        assert_ne!(precision.tier(), "q8");
    }

    #[test]
    fn every_other_flux3_precision_carrier_is_a_typed_refusal() {
        for value in [
            json!({"quantTier": "nvfp4"}),
            json!({"quantTier": "q8"}),
            json!({"quantTier": 8}),
            json!({"mlxQuantize": 8}),
            json!({"mlxQuantize": 0}),
            json!({"convRot": true}),
            json!({"quantTier": "fp8r", "mlxQuantize": 8}),
            json!({"quantTier": "fp8r", "convRot": true}),
        ] {
            let error = flux3_requested_precision("flux3_klein", &advanced(value.clone()))
                .expect_err("refused");
            assert!(
                matches!(error, WorkerError::InvalidPayload(_)),
                "{value}: {error:?}"
            );
        }
        // Other models are not this resolver's business.
        assert_eq!(
            flux3_requested_precision("z_image", &advanced(json!({"mlxQuantize": 8}))).unwrap(),
            None
        );
    }

    #[test]
    fn the_worker_boundary_refuses_fp8r_off_flux3_and_on_mlx() {
        let fp8r = advanced(json!({"quantTier": "fp8r"}));
        assert!(validate_image_fp8r_request("flux3_klein", &fp8r, ImageLane::Candle).is_ok());
        let mlx = validate_image_fp8r_request("flux3_klein", &fp8r, ImageLane::Mlx)
            .expect_err("MLX has no FP8R tier");
        assert!(format!("{mlx:?}").contains("MLX"), "{mlx:?}");
        for model in ["z_image", "flux2_klein_9b", "flux3_klein_base"] {
            for lane in [ImageLane::Candle, ImageLane::Mlx] {
                let error = validate_image_fp8r_request(model, &fp8r, lane)
                    .expect_err("non-flux3 fp8r refused");
                assert!(format!("{error:?}").contains(model), "{error:?}");
            }
        }
        // Requests without the label are untouched on both lanes.
        for lane in [ImageLane::Candle, ImageLane::Mlx] {
            assert!(validate_image_fp8r_request(
                "z_image",
                &advanced(json!({"mlxQuantize": 8})),
                lane
            )
            .is_ok());
        }
    }

    #[test]
    fn fp8r_needs_a_provider_that_advertises_it() {
        assert!(require_advertised_flux3_precision(
            "flux3_klein",
            Flux3Precision::Fp8Rowwise,
            &[Quant::Q4, Quant::Q8, Quant::Fp8Rowwise, Quant::Nvfp4],
        )
        .is_ok());
        // The CPU build's advertisement.
        let error = require_advertised_flux3_precision(
            "flux3_klein",
            Flux3Precision::Fp8Rowwise,
            &[Quant::Q4, Quant::Q8],
        )
        .expect_err("CPU build has no FP8R");
        assert!(format!("{error:?}").contains("CUDA"), "{error:?}");
        assert!(
            require_advertised_flux3_precision("flux3_klein", Flux3Precision::Bf16, &[]).is_ok()
        );
    }

    fn write_snapshot(root: &Path, config: Value) {
        let variant = root.join(FLUX3_FP8R_VARIANT_DIR);
        std::fs::create_dir_all(&variant).unwrap();
        std::fs::write(variant.join("config.json"), config.to_string()).unwrap();
        std::fs::write(variant.join("model.safetensors"), b"x").unwrap();
        std::fs::create_dir_all(root.join("text_encoder")).unwrap();
        std::fs::write(root.join("ae.safetensors"), b"x").unwrap();
    }

    fn upstream_config() -> Value {
        json!({
            "quantization": "fp8r",
            "text_encoder": {"path": "text_encoder"},
            "vae": {"path": "ae.safetensors"}
        })
    }

    #[test]
    fn the_fp8r_package_resolves_its_encoder_and_autoencoder_at_the_snapshot_root() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        write_snapshot(root, upstream_config());
        let package = resolve_flux3_fp8r_package(root).expect("complete package");
        assert_eq!(package.package_dir, root.join("variants").join("fp8r"));
        assert_eq!(package.text_encoder_dir, root.join("text_encoder"));
        assert_eq!(package.autoencoder_file, root.join("ae.safetensors"));
    }

    #[test]
    fn the_fp8r_load_spec_carries_the_root_encoder_and_autoencoder_components() {
        let temp = tempfile::tempdir().unwrap();
        write_snapshot(temp.path(), upstream_config());
        let package = resolve_flux3_fp8r_package(temp.path()).unwrap();
        let spec = attach_flux3_fp8r_components(
            LoadSpec::new(WeightsSource::Dir(package.package_dir.clone()))
                .with_quant(Flux3Precision::Fp8Rowwise.quant().expect("fp8r quant")),
            &package,
        );
        assert_eq!(
            spec.weights,
            WeightsSource::Dir(package.package_dir.clone())
        );
        assert_eq!(spec.quantize, Some(Quant::Fp8Rowwise));
        assert_eq!(
            spec.components.get(gen_core::flux3::TEXT_ENCODER_COMPONENT),
            Some(&WeightsSource::Dir(temp.path().join("text_encoder")))
        );
        assert_eq!(
            spec.components.get(gen_core::flux3::AUTOENCODER_COMPONENT),
            Some(&WeightsSource::File(temp.path().join("ae.safetensors")))
        );
    }

    #[test]
    fn an_incomplete_or_foreign_fp8r_package_is_refused_by_path() {
        // Missing variant entirely.
        let temp = tempfile::tempdir().unwrap();
        let error = resolve_flux3_fp8r_package(temp.path()).expect_err("no variant");
        assert!(format!("{error:?}").contains("config.json"), "{error:?}");

        // Missing root autoencoder.
        let temp = tempfile::tempdir().unwrap();
        write_snapshot(temp.path(), upstream_config());
        std::fs::remove_file(temp.path().join("ae.safetensors")).unwrap();
        let error = resolve_flux3_fp8r_package(temp.path()).expect_err("no autoencoder");
        assert!(format!("{error:?}").contains("ae.safetensors"), "{error:?}");

        // A dense config in the variant slot is not the FP8R package.
        let temp = tempfile::tempdir().unwrap();
        let mut config = upstream_config();
        config["quantization"] = Value::Null;
        write_snapshot(temp.path(), config);
        assert!(resolve_flux3_fp8r_package(temp.path()).is_err());

        // A config path escaping the snapshot is refused.
        let temp = tempfile::tempdir().unwrap();
        let mut config = upstream_config();
        config["vae"]["path"] = json!("../ae.safetensors");
        write_snapshot(temp.path(), config);
        assert!(resolve_flux3_fp8r_package(temp.path()).is_err());
    }
}
