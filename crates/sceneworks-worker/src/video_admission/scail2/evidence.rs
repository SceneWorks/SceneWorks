//! Compact projection of a completed physical capture. The approved index hash binds
//! the raw receipt, native axes and each independently reviewed upper bound together.
use gen_core::generator::scail2_fit::{Scail2FitAxes, Scail2FitBackend, Scail2FitWitness};
use serde::Deserialize;
use std::sync::OnceLock;

const INDEX: &[u8] = include_bytes!("../../../../../config/scail2-plural-memory.json");
const APPROVED_INDEX_SHA256: &str =
    "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Capture {
    platform: String,
    inference_revision: String,
    device: String,
    runtime: String,
    artifact_id: u64,
    receipt_sha256: String,
    guard_policy_sha256: String,
    accepted: bool,
    post_execution_verified: bool,
    guard_exit_code: i32,
    axes: Scail2FitAxes,
    prompt_tokens: Vec<u32>,
    bounds: PhysicalBounds,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, tag = "measurement", rename_all = "snake_case")]
enum PhysicalBounds {
    DarwinLifetimeFootprint {
        unified: Envelope,
        mlx_decode_budget_bytes: u64,
    },
    WindowsJobAndWddm {
        host: Envelope,
        device: Envelope,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, tag = "basis", rename_all = "snake_case")]
enum Envelope {
    GuardedCeiling {
        observed_peak_bytes: u64,
        guard_ceiling_bytes: u64,
        unsampled_overshoot_upper_bytes: u64,
    },
    /// A completed aggregate on this exact hardware, native policy and topology.
    /// The archived review explains which allocations coexist and the allowance for
    /// repeatability or sampling. The worker's live reserve is applied separately.
    MeasuredAggregate {
        observed_peak_bytes: u64,
        classified_peak_bytes: u64,
        co_live_extra_bytes: u64,
        uncertainty_allowance_bytes: u64,
        review_sha256: String,
    },
}

impl Envelope {
    fn price(&self, floor: u64) -> Result<u64, String> {
        let upper = match self {
            Self::GuardedCeiling {
                observed_peak_bytes,
                guard_ceiling_bytes,
                unsampled_overshoot_upper_bytes,
            } if *observed_peak_bytes > 0 && observed_peak_bytes <= guard_ceiling_bytes => {
                guard_ceiling_bytes.checked_add(*unsampled_overshoot_upper_bytes)
            }
            Self::MeasuredAggregate {
                observed_peak_bytes,
                classified_peak_bytes,
                co_live_extra_bytes,
                uncertainty_allowance_bytes,
                review_sha256,
            } if *observed_peak_bytes > 0
                && classified_peak_bytes >= observed_peak_bytes
                && valid_hex(review_sha256, 64) =>
            {
                classified_peak_bytes
                    .checked_add(*co_live_extra_bytes)
                    .and_then(|covered| covered.checked_add(*uncertainty_allowance_bytes))
            }
            _ => return Err("capture has no reviewed physical bound or measured aggregate".into()),
        };
        upper
            .filter(|upper| *upper >= floor)
            .ok_or_else(|| "physical price cannot cover its native floor".into())
    }
}

pub(super) struct QualifiedWitness {
    platform: String,
    inference_revision: String,
    pub(super) device: String,
    pub(super) runtime: String,
    pub(super) artifact_id: u64,
    pub(super) evidence_sha256: String,
    pub(super) witness: Scail2FitWitness,
}

impl QualifiedWitness {
    pub(super) fn covers(&self, platform: &str, revision: &str, count: usize) -> bool {
        self.platform == platform
            && self.inference_revision == revision
            && (2..=self.witness.axes.references as usize).contains(&count)
    }
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Capture {
    fn qualify(self) -> Result<QualifiedWitness, String> {
        if !self.accepted
            || !self.post_execution_verified
            || self.guard_exit_code != 0
            || self.artifact_id == 0
            || !valid_hex(&self.receipt_sha256, 64)
            || !valid_hex(&self.guard_policy_sha256, 64)
            || !valid_hex(&self.inference_revision, 40)
            || !valid_hex(&self.axes.artifact_sha256, 64)
            || self.device.trim().is_empty()
            || self.runtime.trim().is_empty()
            || !(2..=6).contains(&self.axes.references)
            || self.axes.windows.is_empty()
            || self.prompt_tokens.len()
                != if self.axes.classifier_free_guidance {
                    2
                } else {
                    1
                }
            || self.prompt_tokens.iter().any(|tokens| *tokens != 512)
        {
            return Err("incomplete SCAIL-2 physical capture or unverified native source".into());
        }
        let (host, device, unified, decode_cap) =
            match (self.platform.as_str(), self.axes.backend, self.bounds) {
                (
                    "macos",
                    Scail2FitBackend::Mlx,
                    PhysicalBounds::DarwinLifetimeFootprint {
                        unified,
                        mlx_decode_budget_bytes,
                    },
                ) if mlx_decode_budget_bytes > 0 => (
                    0,
                    0,
                    unified.price(self.axes.floor_unified_bytes)?,
                    Some(mlx_decode_budget_bytes),
                ),
                (
                    "windows",
                    Scail2FitBackend::Cuda,
                    PhysicalBounds::WindowsJobAndWddm { host, device },
                ) => (
                    host.price(self.axes.floor_host_bytes)?,
                    device.price(self.axes.floor_device_bytes)?,
                    0,
                    None,
                ),
                _ => {
                    return Err(
                        "capture has no verified physical bounds for this platform and provider"
                            .into(),
                    )
                }
            };
        Ok(QualifiedWitness {
            platform: self.platform,
            inference_revision: self.inference_revision,
            device: self.device,
            runtime: self.runtime,
            artifact_id: self.artifact_id,
            evidence_sha256: self.receipt_sha256,
            witness: Scail2FitWitness {
                axes: self.axes,
                text_tokens: 512,
                gross_host_bytes: host,
                gross_device_bytes: device,
                gross_unified_bytes: unified,
                mlx_decode_budget_bytes: decode_cap,
            },
        })
    }
}

pub(super) fn qualify_index(bytes: &[u8], approved: &str) -> Result<Vec<QualifiedWitness>, String> {
    if sceneworks_core::hashing::sha256_bytes(bytes) != approved {
        return Err("SCAIL-2 physical capture index differs from its approved digest".into());
    }
    let captures: Vec<Capture> = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    captures.into_iter().map(Capture::qualify).collect()
}

pub(super) fn packaged() -> crate::WorkerResult<&'static [QualifiedWitness]> {
    static ROWS: OnceLock<Result<Vec<QualifiedWitness>, String>> = OnceLock::new();
    ROWS.get_or_init(|| qualify_index(INDEX, APPROVED_INDEX_SHA256))
        .as_ref()
        .map(Vec::as_slice)
        .map_err(|e| {
            crate::WorkerError::InvalidPayload(format!(
                "invalid SCAIL-2 plural memory evidence: {e}"
            ))
        })
}
