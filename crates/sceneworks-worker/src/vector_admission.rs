//! Cross-backend StarVector static device admission (SC-22261).
//!
//! The exact safetensor sum is an unmeasured impossible-load lower bound. It prevents a provider
//! load on a device that cannot hold the immutable native weights without pretending to know the
//! activation, allocator, or runtime peak that belongs to the terminal campaign.
//!
//! StarVector-8B is cataloged and installable in this batch, but remains fail-closed until that
//! stationary terminal candidate exists. Flipping an 8B provider to `available: true` prematurely
//! still produces a typed refusal here; weight bytes alone can never promote it.

use serde_json::Value;

const SCHEMA_VERSION: u64 = 1;
const FLOOR_BASIS: &str = "exact_safetensors_bytes";
const MLX_DEVICE_CLASS: &str = "apple_unified_memory";
const CANDLE_DEVICE_CLASS: &str = "nvidia_dedicated_vram";
const TERMINAL_8B_MODEL_ID: &str = "starvector_8b";
const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[cfg_attr(target_os = "macos", allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VectorBackend {
    Mlx,
    Candle,
}

impl VectorBackend {
    fn manifest_key(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::Candle => "candle",
        }
    }

    fn device_class(self) -> &'static str {
        match self {
            Self::Mlx => MLX_DEVICE_CLASS,
            Self::Candle => CANDLE_DEVICE_CLASS,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VectorDeviceFacts {
    device_class: &'static str,
    total_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum VectorAdmissionRefusal {
    MissingContract {
        model: String,
        backend: VectorBackend,
    },
    InvalidContract {
        model: String,
        backend: VectorBackend,
        detail: &'static str,
    },
    WrongDeviceClass {
        model: String,
        backend: VectorBackend,
        declared: String,
        expected: &'static str,
    },
    DeviceMetadataUnavailable {
        model: String,
        backend: VectorBackend,
    },
    InsufficientDeviceMemory {
        model: String,
        backend: VectorBackend,
        required_bytes: u64,
        available_bytes: u64,
    },
    TerminalCandidatePending {
        model: String,
        backend: VectorBackend,
    },
}

impl VectorAdmissionRefusal {
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::MissingContract { .. } => "vector_device_contract_missing",
            Self::InvalidContract { .. } => "vector_device_contract_invalid",
            Self::WrongDeviceClass { .. } => "vector_device_class_mismatch",
            Self::DeviceMetadataUnavailable { .. } => "vector_device_metadata_unavailable",
            Self::InsufficientDeviceMemory { .. } => "vector_device_memory_insufficient",
            Self::TerminalCandidatePending { .. } => "vector_terminal_candidate_pending",
        }
    }
}

impl std::fmt::Display for VectorAdmissionRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let backend = match self {
            Self::MissingContract { backend, .. }
            | Self::InvalidContract { backend, .. }
            | Self::WrongDeviceClass { backend, .. }
            | Self::DeviceMetadataUnavailable { backend, .. }
            | Self::InsufficientDeviceMemory { backend, .. }
            | Self::TerminalCandidatePending { backend, .. } => backend.manifest_key(),
        };
        match self {
            Self::MissingContract { model, .. } => write!(
                formatter,
                "{}: {model} has no exact {backend} device-admission contract",
                self.code()
            ),
            Self::InvalidContract { model, detail, .. } => write!(
                formatter,
                "{}: {model} has an invalid {backend} device-admission contract ({detail})",
                self.code()
            ),
            Self::WrongDeviceClass {
                model,
                declared,
                expected,
                ..
            } => write!(
                formatter,
                "{}: {model} declares {declared:?} for {backend}; expected {expected:?}",
                self.code()
            ),
            Self::DeviceMetadataUnavailable { model, .. } => write!(
                formatter,
                "{}: cannot read the live {backend} device memory for {model}",
                self.code()
            ),
            Self::InsufficientDeviceMemory {
                model,
                required_bytes,
                available_bytes,
                ..
            } => write!(
                formatter,
                "{}: {model} needs at least {:.2} GiB just to hold its exact native weights, but the selected {backend} device reports {:.2} GiB total",
                self.code(),
                *required_bytes as f64 / BYTES_PER_GIB,
                *available_bytes as f64 / BYTES_PER_GIB
            ),
            Self::TerminalCandidatePending { model, .. } => write!(
                formatter,
                "{}: {model} is installed but {backend} dispatch remains disabled until the permanent-pin terminal candidate is accepted",
                self.code()
            ),
        }
    }
}

fn model_id(model: &Value) -> String {
    model
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("unknown_vector_model")
        .to_owned()
}

fn provider_is_available(model: &Value, backend: VectorBackend) -> bool {
    model
        .pointer(&format!(
            "/vector/providers/{}/available",
            backend.manifest_key()
        ))
        .and_then(Value::as_bool)
        == Some(true)
}

fn invalid(model: &str, backend: VectorBackend, detail: &'static str) -> VectorAdmissionRefusal {
    VectorAdmissionRefusal::InvalidContract {
        model: model.to_owned(),
        backend,
        detail,
    }
}

/// Pure admission decision over injected live facts. `total_bytes` is total memory in the selected
/// backend-specific pool, not current free memory: concurrency and cache reclamation remain owned
/// by their existing runtime guards.
fn vector_admission_refusal(
    model: &Value,
    backend: VectorBackend,
    facts: Option<&VectorDeviceFacts>,
) -> Option<VectorAdmissionRefusal> {
    // Preserve the manifest/provider availability refusal as the first authority. The defensive
    // 8B refusal below still prevents an accidental `available: true` edit from promoting it.
    if !provider_is_available(model, backend) {
        return None;
    }

    let id = model_id(model);
    let Some(contract) = model.pointer("/vector/deviceAdmission") else {
        return Some(VectorAdmissionRefusal::MissingContract { model: id, backend });
    };
    if contract.get("schemaVersion").and_then(Value::as_u64) != Some(SCHEMA_VERSION) {
        return Some(invalid(&id, backend, "schemaVersion must be 1"));
    }
    if contract.get("basis").and_then(Value::as_str) != Some(FLOOR_BASIS) {
        return Some(invalid(
            &id,
            backend,
            "basis must be exact_safetensors_bytes",
        ));
    }
    if contract.get("measured").and_then(Value::as_bool) != Some(false) {
        return Some(invalid(
            &id,
            backend,
            "the static weights floor must declare measured=false",
        ));
    }
    let Some(required_bytes) = contract
        .get("staticWeightFloorBytes")
        .and_then(Value::as_u64)
        .filter(|bytes| *bytes > 0)
    else {
        return Some(invalid(
            &id,
            backend,
            "staticWeightFloorBytes must be a positive integer",
        ));
    };
    let declared = contract
        .pointer(&format!("/deviceClasses/{}", backend.manifest_key()))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if declared != backend.device_class() {
        return Some(VectorAdmissionRefusal::WrongDeviceClass {
            model: id,
            backend,
            declared: declared.to_owned(),
            expected: backend.device_class(),
        });
    }
    if id == TERMINAL_8B_MODEL_ID {
        return Some(VectorAdmissionRefusal::TerminalCandidatePending { model: id, backend });
    }
    let Some(facts) = facts else {
        return Some(VectorAdmissionRefusal::DeviceMetadataUnavailable { model: id, backend });
    };
    if facts.device_class != backend.device_class() {
        return Some(VectorAdmissionRefusal::WrongDeviceClass {
            model: id,
            backend,
            declared: facts.device_class.to_owned(),
            expected: backend.device_class(),
        });
    }
    (facts.total_bytes < required_bytes).then_some(
        VectorAdmissionRefusal::InsufficientDeviceMemory {
            model: id,
            backend,
            required_bytes,
            available_bytes: facts.total_bytes,
        },
    )
}

#[cfg(target_os = "macos")]
async fn live_device_facts(_gpu_id: &str) -> Option<VectorDeviceFacts> {
    let total_bytes = crate::gpu::total_unified_memory_gb().await? * BYTES_PER_GIB;
    Some(VectorDeviceFacts {
        device_class: MLX_DEVICE_CLASS,
        total_bytes: total_bytes.round() as u64,
    })
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn live_device_facts(gpu_id: &str) -> Option<VectorDeviceFacts> {
    let budget = crate::gpu::nvidia_vram_budget_gb(gpu_id).await?;
    Some(VectorDeviceFacts {
        device_class: CANDLE_DEVICE_CLASS,
        total_bytes: (budget.total_gb * BYTES_PER_GIB).round() as u64,
    })
}

#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
async fn live_device_facts(_gpu_id: &str) -> Option<VectorDeviceFacts> {
    None
}

/// Read the active native lane's live device metadata and return a user-facing typed refusal.
/// The vector job calls this immediately before native provider resolution/load.
pub(crate) async fn vector_admission_error(model: &Value, gpu_id: &str) -> Option<String> {
    #[cfg(target_os = "macos")]
    let backend = VectorBackend::Mlx;

    #[cfg(not(target_os = "macos"))]
    let backend = VectorBackend::Candle;

    let facts = live_device_facts(gpu_id).await;
    vector_admission_refusal(model, backend, facts.as_ref()).map(|refusal| refusal.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const FLOOR_1B: u64 = 5_142_705_320;
    const FLOOR_8B: u64 = 15_014_294_040;

    fn model(id: &str, floor: u64, available: bool) -> Value {
        json!({
            "id": id,
            "vector": {
                "providers": {
                    "mlx": { "available": available },
                    "candle": { "available": available }
                },
                "deviceAdmission": {
                    "schemaVersion": 1,
                    "basis": "exact_safetensors_bytes",
                    "measured": false,
                    "staticWeightFloorBytes": floor,
                    "deviceClasses": {
                        "mlx": "apple_unified_memory",
                        "candle": "nvidia_dedicated_vram"
                    }
                }
            }
        })
    }

    fn facts(backend: VectorBackend, total_bytes: u64) -> VectorDeviceFacts {
        VectorDeviceFacts {
            device_class: backend.device_class(),
            total_bytes,
        }
    }

    #[test]
    fn one_b_static_floor_splits_both_device_classes_at_the_byte_boundary() {
        let model = model("starvector_1b", FLOOR_1B, true);
        for backend in [VectorBackend::Mlx, VectorBackend::Candle] {
            let exact = facts(backend, FLOOR_1B);
            assert_eq!(
                vector_admission_refusal(&model, backend, Some(&exact)),
                None,
                "the exact floor fits"
            );
            let short = facts(backend, FLOOR_1B - 1);
            assert!(matches!(
                vector_admission_refusal(&model, backend, Some(&short)),
                Some(VectorAdmissionRefusal::InsufficientDeviceMemory {
                    required_bytes: FLOOR_1B,
                    available_bytes,
                    ..
                }) if available_bytes == FLOOR_1B - 1
            ));
        }
    }

    #[test]
    fn eight_b_cannot_be_promoted_by_provider_availability_or_weight_bytes_alone() {
        let premature = model("starvector_8b", FLOOR_8B, true);
        for backend in [VectorBackend::Mlx, VectorBackend::Candle] {
            assert!(matches!(
                vector_admission_refusal(&premature, backend, Some(&facts(backend, u64::MAX)),),
                Some(VectorAdmissionRefusal::TerminalCandidatePending { .. })
            ));
        }

        let unavailable = model("starvector_8b", FLOOR_8B, false);
        assert_eq!(
            vector_admission_refusal(&unavailable, VectorBackend::Mlx, None),
            None,
            "the provider's typed unavailable reason remains authoritative"
        );
    }

    #[test]
    fn available_provider_fails_closed_on_missing_malformed_or_unprobed_contract() {
        let mut missing = model("starvector_1b", FLOOR_1B, true);
        missing["vector"]
            .as_object_mut()
            .unwrap()
            .remove("deviceAdmission");
        assert!(matches!(
            vector_admission_refusal(
                &missing,
                VectorBackend::Mlx,
                Some(&facts(VectorBackend::Mlx, u64::MAX)),
            ),
            Some(VectorAdmissionRefusal::MissingContract { .. })
        ));

        let mut measured = model("starvector_1b", FLOOR_1B, true);
        measured["vector"]["deviceAdmission"]["measured"] = json!(true);
        assert!(matches!(
            vector_admission_refusal(
                &measured,
                VectorBackend::Mlx,
                Some(&facts(VectorBackend::Mlx, u64::MAX)),
            ),
            Some(VectorAdmissionRefusal::InvalidContract { .. })
        ));

        let mut wrong_class = model("starvector_1b", FLOOR_1B, true);
        wrong_class["vector"]["deviceAdmission"]["deviceClasses"]["candle"] =
            json!("apple_unified_memory");
        assert!(matches!(
            vector_admission_refusal(
                &wrong_class,
                VectorBackend::Candle,
                Some(&facts(VectorBackend::Candle, u64::MAX)),
            ),
            Some(VectorAdmissionRefusal::WrongDeviceClass { .. })
        ));

        assert!(matches!(
            vector_admission_refusal(
                &model("starvector_1b", FLOOR_1B, true),
                VectorBackend::Candle,
                None,
            ),
            Some(VectorAdmissionRefusal::DeviceMetadataUnavailable { .. })
        ));
    }

    #[test]
    fn refusal_codes_are_stable_and_actionable() {
        let metadata = vector_admission_refusal(
            &model("starvector_1b", FLOOR_1B, true),
            VectorBackend::Mlx,
            None,
        )
        .expect("missing metadata refuses");
        assert_eq!(metadata.code(), "vector_device_metadata_unavailable");
        assert!(metadata.to_string().starts_with(metadata.code()));

        let terminal = vector_admission_refusal(
            &model("starvector_8b", FLOOR_8B, true),
            VectorBackend::Mlx,
            Some(&facts(VectorBackend::Mlx, u64::MAX)),
        )
        .expect("premature 8B provider refuses");
        assert_eq!(terminal.code(), "vector_terminal_candidate_pending");
        assert!(terminal.to_string().starts_with(terminal.code()));
    }
}
