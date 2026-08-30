//! Cross-backend StarVector static device admission (SC-22261).
//!
//! The exact safetensor sum is an unmeasured impossible-load lower bound. It prevents a provider
//! load on a device that cannot hold the immutable native weights without pretending to know the
//! activation, allocator, or runtime peak that belongs to the terminal campaign.
//!
//! StarVector-8B is cataloged and installable in this batch, but remains fail-closed until that
//! stationary terminal candidate exists. Flipping an 8B provider to `available: true` prematurely
//! still produces a typed refusal here; weight bytes alone can never promote it.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: u64 = 1;
const FLOOR_BASIS: &str = "exact_safetensors_bytes";
const MLX_DEVICE_CLASS: &str = "apple_unified_memory";
const CANDLE_DEVICE_CLASS: &str = "nvidia_dedicated_vram";
const TERMINAL_8B_MODEL_ID: &str = "starvector_8b";
const TERMINAL_8B_REPOSITORY: &str = "starvector/starvector-8b-im2svg";
const TERMINAL_8B_REVISION: &str = "518beea8dcb5f7a37c5911e92d1d62a76beee7f9";
const TERMINAL_8B_MLX_PROVIDER: &str = "mlx-starvector-8b";
const TERMINAL_8B_CANDLE_PROVIDER: &str = "candle-starvector-8b";
const TERMINAL_8B_FILES: &[&str] = &[
    "README.md",
    "added_tokens.json",
    "config.json",
    "merges.txt",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
];
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
    device_name: Option<String>,
    total_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TerminalSupportedDevice {
    device_class: &'static str,
    device_name: Option<String>,
    total_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TerminalCandidateError {
    Pending,
    Invalid(&'static str),
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
    UnsupportedTerminalDevice {
        model: String,
        backend: VectorBackend,
        device_name: Option<String>,
        total_bytes: u64,
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
            Self::UnsupportedTerminalDevice { .. } => "vector_terminal_device_unsupported",
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
            | Self::TerminalCandidatePending { backend, .. }
            | Self::UnsupportedTerminalDevice { backend, .. } => backend.manifest_key(),
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
            Self::UnsupportedTerminalDevice {
                model,
                device_name,
                total_bytes,
                ..
            } => write!(
                formatter,
                "{}: {model} has no accepted {backend} terminal candidate for device {:?} with {} raw bytes",
                self.code(),
                device_name.as_deref().unwrap_or("unreported"),
                total_bytes
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

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn has_exact_keys(value: &Value, expected: &[&str]) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.len() == expected.len() && expected.iter().all(|key| object.contains_key(*key))
}

fn normalized_production_path(value: &str) -> bool {
    if value.is_empty()
        || value.starts_with('/')
        || value.contains('\\')
        || value.contains("//")
        || value == "config/manifests/builtin.models.jsonc"
    {
        return false;
    }
    value
        .split('/')
        .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

fn production_closure_sha256(entries: &[Value]) -> Option<String> {
    let canonical_entries = entries
        .iter()
        .map(|entry| {
            let mut canonical = BTreeMap::new();
            canonical.insert("byteSize", entry.get("byteSize")?.clone());
            canonical.insert("path", entry.get("path")?.clone());
            canonical.insert("sha256", entry.get("sha256")?.clone());
            Some(canonical)
        })
        .collect::<Option<Vec<_>>>()?;
    let payload = BTreeMap::from([("entries", canonical_entries)]);
    let bytes = serde_json::to_vec(&payload).ok()?;
    Some(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_production_closure(value: &Value) -> Result<(), TerminalCandidateError> {
    if !has_exact_keys(value, &["schemaVersion", "sha256", "entries"])
        || value.get("schemaVersion").and_then(Value::as_u64) != Some(SCHEMA_VERSION)
    {
        return Err(TerminalCandidateError::Invalid(
            "terminal productionClosure header is malformed",
        ));
    }
    let aggregate = value
        .get("sha256")
        .and_then(Value::as_str)
        .filter(|value| is_lower_hex(value, 64))
        .ok_or(TerminalCandidateError::Invalid(
            "terminal productionClosure sha256 is malformed",
        ))?;
    let entries = value
        .get("entries")
        .and_then(Value::as_array)
        .filter(|entries| !entries.is_empty())
        .ok_or(TerminalCandidateError::Invalid(
            "terminal productionClosure entries are empty",
        ))?;
    let mut previous: Option<&str> = None;
    let mut paths = BTreeSet::new();
    for entry in entries {
        if !has_exact_keys(entry, &["path", "byteSize", "sha256"])
            || entry.get("byteSize").and_then(Value::as_u64).is_none()
        {
            return Err(TerminalCandidateError::Invalid(
                "terminal productionClosure entry is malformed",
            ));
        }
        let path = entry
            .get("path")
            .and_then(Value::as_str)
            .filter(|path| normalized_production_path(path))
            .ok_or(TerminalCandidateError::Invalid(
                "terminal productionClosure path is not normalized",
            ))?;
        let digest = entry
            .get("sha256")
            .and_then(Value::as_str)
            .filter(|value| is_lower_hex(value, 64))
            .ok_or(TerminalCandidateError::Invalid(
                "terminal productionClosure entry sha256 is malformed",
            ))?;
        if previous.is_some_and(|previous| previous.as_bytes() >= path.as_bytes())
            || !paths.insert(path)
            || !is_lower_hex(digest, 64)
        {
            return Err(TerminalCandidateError::Invalid(
                "terminal productionClosure entries are not unique and byte-sorted",
            ));
        }
        previous = Some(path);
    }
    if production_closure_sha256(entries).as_deref() != Some(aggregate) {
        return Err(TerminalCandidateError::Invalid(
            "terminal productionClosure aggregate does not match canonical entries",
        ));
    }
    Ok(())
}

fn candidate_model_is_exact(value: &Value) -> bool {
    has_exact_keys(value, &["repository", "revision", "files"])
        && value.get("repository").and_then(Value::as_str) == Some(TERMINAL_8B_REPOSITORY)
        && value.get("revision").and_then(Value::as_str) == Some(TERMINAL_8B_REVISION)
        && value
            .get("files")
            .and_then(Value::as_array)
            .is_some_and(|files| {
                files.len() == TERMINAL_8B_FILES.len()
                    && files
                        .iter()
                        .zip(TERMINAL_8B_FILES)
                        .all(|(actual, expected)| actual.as_str() == Some(expected))
            })
}

fn parse_supported_devices(
    value: &Value,
    backend: VectorBackend,
) -> Result<Vec<TerminalSupportedDevice>, TerminalCandidateError> {
    let Some(devices) = value.as_array().filter(|devices| !devices.is_empty()) else {
        return if value.is_null() {
            Err(TerminalCandidateError::Pending)
        } else {
            Err(TerminalCandidateError::Invalid(
                "terminal supportedDevices must contain at least one device",
            ))
        };
    };
    let mut parsed = Vec::with_capacity(devices.len());
    let mut identities = BTreeSet::new();
    for device in devices {
        let expected_keys: &[&str] = match backend {
            VectorBackend::Mlx => &["deviceClass", "totalBytes"],
            VectorBackend::Candle => &["deviceClass", "deviceName", "totalBytes"],
        };
        if !has_exact_keys(device, expected_keys)
            || device.get("deviceClass").and_then(Value::as_str) != Some(backend.device_class())
        {
            return Err(TerminalCandidateError::Invalid(
                "terminal supported device class or keys are malformed",
            ));
        }
        let total_bytes = device
            .get("totalBytes")
            .and_then(Value::as_u64)
            .filter(|bytes| *bytes > 0)
            .ok_or(TerminalCandidateError::Invalid(
                "terminal supported device totalBytes is malformed",
            ))?;
        let device_name = match backend {
            VectorBackend::Mlx => None,
            VectorBackend::Candle => Some(
                device
                    .get("deviceName")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .ok_or(TerminalCandidateError::Invalid(
                        "terminal Candle deviceName is malformed",
                    ))?
                    .to_owned(),
            ),
        };
        if !identities.insert((device_name.clone(), total_bytes)) {
            return Err(TerminalCandidateError::Invalid(
                "terminal supported device is duplicated",
            ));
        }
        parsed.push(TerminalSupportedDevice {
            device_class: backend.device_class(),
            device_name,
            total_bytes,
        });
    }
    Ok(parsed)
}

fn validate_terminal_candidate(
    contract: &Value,
    backend: VectorBackend,
) -> Result<Vec<TerminalSupportedDevice>, TerminalCandidateError> {
    let candidate = contract
        .get("terminalCandidate")
        .ok_or(TerminalCandidateError::Pending)?;
    if !has_exact_keys(
        candidate,
        &[
            "schemaVersion",
            "inferenceRevision",
            "corpusSha256",
            "model",
            "providers",
            "productionClosure",
            "supportedDevices",
        ],
    ) || candidate.get("schemaVersion").and_then(Value::as_u64) != Some(SCHEMA_VERSION)
    {
        return Err(TerminalCandidateError::Invalid(
            "terminalCandidate header or keys are malformed",
        ));
    }
    let Some(inference_revision) = candidate.get("inferenceRevision").and_then(Value::as_str)
    else {
        return if candidate
            .get("inferenceRevision")
            .is_some_and(Value::is_null)
        {
            Err(TerminalCandidateError::Pending)
        } else {
            Err(TerminalCandidateError::Invalid(
                "terminal inferenceRevision is malformed",
            ))
        };
    };
    if !is_lower_hex(inference_revision, 40)
        || inference_revision != crate::catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION
    {
        return Err(TerminalCandidateError::Invalid(
            "terminal inferenceRevision does not match the linked inference runtime",
        ));
    }
    let Some(corpus_sha256) = candidate.get("corpusSha256").and_then(Value::as_str) else {
        return if candidate.get("corpusSha256").is_some_and(Value::is_null) {
            Err(TerminalCandidateError::Pending)
        } else {
            Err(TerminalCandidateError::Invalid(
                "terminal corpusSha256 is malformed",
            ))
        };
    };
    if !is_lower_hex(corpus_sha256, 64) {
        return Err(TerminalCandidateError::Invalid(
            "terminal corpusSha256 is malformed",
        ));
    }
    if !candidate.get("model").is_some_and(candidate_model_is_exact) {
        return Err(TerminalCandidateError::Invalid(
            "terminal immutable model closure is not exact",
        ));
    }
    let providers = candidate
        .get("providers")
        .ok_or(TerminalCandidateError::Invalid(
            "terminal provider identities are missing",
        ))?;
    if !has_exact_keys(providers, &["mlx", "candle"])
        || providers.get("mlx").and_then(Value::as_str) != Some(TERMINAL_8B_MLX_PROVIDER)
        || providers.get("candle").and_then(Value::as_str) != Some(TERMINAL_8B_CANDLE_PROVIDER)
    {
        return Err(TerminalCandidateError::Invalid(
            "terminal provider identities are not exact",
        ));
    }
    let production_closure =
        candidate
            .get("productionClosure")
            .ok_or(TerminalCandidateError::Invalid(
                "terminal productionClosure is missing",
            ))?;
    if production_closure.is_null() {
        return Err(TerminalCandidateError::Pending);
    }
    validate_production_closure(production_closure)?;
    let supported = candidate
        .get("supportedDevices")
        .filter(|value| has_exact_keys(value, &["mlx", "candle"]))
        .ok_or(TerminalCandidateError::Invalid(
            "terminal supportedDevices keys are malformed",
        ))?;
    // A terminal candidate is atomic across both required backends: a missing sibling fact keeps
    // every 8B route closed rather than admitting one half of an incomplete campaign.
    let mlx = parse_supported_devices(&supported["mlx"], VectorBackend::Mlx)?;
    let candle = parse_supported_devices(&supported["candle"], VectorBackend::Candle)?;
    Ok(match backend {
        VectorBackend::Mlx => mlx,
        VectorBackend::Candle => candle,
    })
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
    let terminal_devices = if id == TERMINAL_8B_MODEL_ID {
        match validate_terminal_candidate(contract, backend) {
            Ok(devices) => Some(devices),
            Err(TerminalCandidateError::Pending) => {
                return Some(VectorAdmissionRefusal::TerminalCandidatePending {
                    model: id,
                    backend,
                });
            }
            Err(TerminalCandidateError::Invalid(detail)) => {
                return Some(invalid(&id, backend, detail));
            }
        }
    } else {
        None
    };
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
    if terminal_devices.is_some_and(|devices| {
        !devices.iter().any(|device| {
            device.device_class == facts.device_class
                && device.total_bytes == facts.total_bytes
                && device.device_name == facts.device_name
        })
    }) {
        return Some(VectorAdmissionRefusal::UnsupportedTerminalDevice {
            model: id,
            backend,
            device_name: facts.device_name.clone(),
            total_bytes: facts.total_bytes,
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
        device_name: None,
        total_bytes: total_bytes.round() as u64,
    })
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn live_device_facts(gpu_id: &str) -> Option<VectorDeviceFacts> {
    let budget = crate::gpu::nvidia_vram_budget_gb(gpu_id).await?;
    let display_name = crate::gpu::query_nvidia_gpus()
        .await
        .into_iter()
        .find(|gpu| gpu.id == gpu_id)?
        .name;
    let suffix = format!(" ({} MB)", (budget.total_gb * 1024.0).round() as u64);
    let device_name = display_name
        .strip_suffix(&suffix)
        .unwrap_or(&display_name)
        .to_owned();
    Some(VectorDeviceFacts {
        device_class: CANDLE_DEVICE_CLASS,
        device_name: Some(device_name),
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
            device_name: (backend == VectorBackend::Candle).then(|| "NVIDIA Test GPU".to_owned()),
            total_bytes,
        }
    }

    fn complete_terminal_candidate() -> Value {
        let entries = vec![json!({
            "path": "Cargo.toml",
            "byteSize": 1,
            "sha256": "a".repeat(64)
        })];
        json!({
            "schemaVersion": 1,
            "inferenceRevision": crate::catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION,
            "corpusSha256": "b".repeat(64),
            "model": {
                "repository": TERMINAL_8B_REPOSITORY,
                "revision": TERMINAL_8B_REVISION,
                "files": TERMINAL_8B_FILES
            },
            "providers": {
                "mlx": TERMINAL_8B_MLX_PROVIDER,
                "candle": TERMINAL_8B_CANDLE_PROVIDER
            },
            "productionClosure": {
                "schemaVersion": 1,
                "sha256": production_closure_sha256(&entries).expect("closure hashes"),
                "entries": entries
            },
            "supportedDevices": {
                "mlx": [{
                    "deviceClass": MLX_DEVICE_CLASS,
                    "totalBytes": 137_438_953_472u64
                }],
                "candle": [{
                    "deviceClass": CANDLE_DEVICE_CLASS,
                    "deviceName": "NVIDIA Test GPU",
                    "totalBytes": 51_539_607_552u64
                }]
            }
        })
    }

    fn complete_eight_b_model() -> Value {
        let mut value = model(TERMINAL_8B_MODEL_ID, FLOOR_8B, true);
        value["vector"]["deviceAdmission"]["terminalCandidate"] = complete_terminal_candidate();
        value
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
    fn complete_eight_b_candidate_admits_only_exact_measured_device_facts() {
        let model = complete_eight_b_model();
        let mlx = facts(VectorBackend::Mlx, 137_438_953_472);
        assert_eq!(
            vector_admission_refusal(&model, VectorBackend::Mlx, Some(&mlx)),
            None
        );
        let candle = facts(VectorBackend::Candle, 51_539_607_552);
        assert_eq!(
            vector_admission_refusal(&model, VectorBackend::Candle, Some(&candle)),
            None
        );

        let mut wrong_name = candle.clone();
        wrong_name.device_name = Some("NVIDIA Other GPU".to_owned());
        assert!(matches!(
            vector_admission_refusal(&model, VectorBackend::Candle, Some(&wrong_name)),
            Some(VectorAdmissionRefusal::UnsupportedTerminalDevice { .. })
        ));
        let wrong_total = facts(VectorBackend::Mlx, 137_438_953_471);
        assert!(matches!(
            vector_admission_refusal(&model, VectorBackend::Mlx, Some(&wrong_total)),
            Some(VectorAdmissionRefusal::UnsupportedTerminalDevice { .. })
        ));
    }

    #[test]
    fn eight_b_candidate_identity_and_closure_mutations_fail_closed() {
        let exact = complete_eight_b_model();
        let mlx = facts(VectorBackend::Mlx, 137_438_953_472);
        for mutate in [
            |value: &mut Value| {
                value["vector"]["deviceAdmission"]["terminalCandidate"]["inferenceRevision"] =
                    json!("0".repeat(40));
            },
            |value: &mut Value| {
                value["vector"]["deviceAdmission"]["terminalCandidate"]["model"]["repository"] =
                    json!("other/model");
            },
            |value: &mut Value| {
                value["vector"]["deviceAdmission"]["terminalCandidate"]["providers"]["mlx"] =
                    json!("mlx-other");
            },
            |value: &mut Value| {
                value["vector"]["deviceAdmission"]["terminalCandidate"]["productionClosure"]
                    ["sha256"] = json!("0".repeat(64));
            },
        ] {
            let mut changed = exact.clone();
            mutate(&mut changed);
            assert!(matches!(
                vector_admission_refusal(&changed, VectorBackend::Mlx, Some(&mlx)),
                Some(VectorAdmissionRefusal::InvalidContract { .. })
            ));
        }
    }

    #[test]
    fn eight_b_candidate_nulls_are_pending_not_permanent_exceptions() {
        for pointer in [
            "/vector/deviceAdmission/terminalCandidate/inferenceRevision",
            "/vector/deviceAdmission/terminalCandidate/corpusSha256",
            "/vector/deviceAdmission/terminalCandidate/productionClosure",
            "/vector/deviceAdmission/terminalCandidate/supportedDevices/candle",
        ] {
            let mut pending = complete_eight_b_model();
            *pending
                .pointer_mut(pointer)
                .expect("candidate field exists") = Value::Null;
            assert!(matches!(
                vector_admission_refusal(
                    &pending,
                    VectorBackend::Mlx,
                    Some(&facts(VectorBackend::Mlx, 137_438_953_472)),
                ),
                Some(VectorAdmissionRefusal::TerminalCandidatePending { .. })
            ));
        }
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
