//! Compact projection of a completed physical capture. The approved index hash binds
//! the raw receipt, native axes and each independently reviewed upper bound together.
use gen_core::generator::scail2_fit::{Scail2FitAxes, Scail2FitBackend, Scail2FitWitness};
use serde::Deserialize;
use std::sync::OnceLock;

const INDEX: &[u8] = include_bytes!("../../../../../config/scail2-plural-memory.json");
const APPROVED_INDEX_SHA256: &str =
    "693e2a0caa6bb7b2b678661deda0dda41230a4eb01f40ac48a2fcc04fba8a0db";

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
        // Identifies the measuring adapter in the signed-off package for product proof.
        // Deployed workers match its hardware class, driver and native execution axes.
        capture_adapter: CaptureAdapter,
        host: Envelope,
        device: Envelope,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureAdapter {
    cuda_uuid: String,
    luid: String,
    node_mask: u32,
}

impl CaptureAdapter {
    fn valid(&self) -> bool {
        let Some(uuid) = self.cuda_uuid.strip_prefix("GPU-") else {
            return false;
        };
        let Some((high, low)) = self
            .luid
            .strip_prefix("luid_0x")
            .and_then(|v| v.split_once("_0x"))
        else {
            return false;
        };
        uuid::Uuid::try_parse(uuid).is_ok_and(|id| id.hyphenated().to_string() == uuid)
            && valid_hex(high, 8)
            && valid_hex(low, 8)
            && (high != "00000000" || low != "00000000")
            && self.node_mask.is_power_of_two()
    }
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
            && permits_revision(
                &self.inference_revision,
                revision,
                self.witness.axes.backend,
            )
            && (2..=self.witness.axes.references as usize).contains(&count)
    }
}

/// A cross-pin qualification is limited to a reviewed pair of immutable revisions and one
/// native backend. The full source-tree delta covers inputs outside the provider's compile digest.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PinCoverage {
    capture_revision: String,
    target_revision: String,
    backend: Scail2FitBackend,
    provider_digest: String,
    shared_contract_digest: String,
    tree_diff_sha256: String,
}

impl PinCoverage {
    fn matches(&self, capture: &str, target: &str, backend: Scail2FitBackend) -> bool {
        self.capture_revision == capture
            && self.target_revision == target
            && self.backend == backend
    }

    fn valid(&self) -> bool {
        valid_hex(&self.capture_revision, 40)
            && valid_hex(&self.target_revision, 40)
            && self.capture_revision != self.target_revision
            && [
                &self.provider_digest,
                &self.shared_contract_digest,
                &self.tree_diff_sha256,
            ]
            .iter()
            .all(|digest| valid_hex(digest, 64))
    }
}

fn approved_pin_coverage(bytes: &[u8], approved: &str) -> Option<Vec<PinCoverage>> {
    if sceneworks_core::hashing::sha256_bytes(bytes) != approved {
        return None;
    }
    let rows: Vec<PinCoverage> = serde_json::from_slice(bytes).ok()?;
    rows.iter().all(PinCoverage::valid).then_some(rows)
}

fn permits_revision(capture: &str, target: &str, backend: Scail2FitBackend) -> bool {
    if capture == target {
        return true;
    }
    static APPROVALS: OnceLock<Vec<PinCoverage>> = OnceLock::new();
    APPROVALS
        .get_or_init(|| {
            const BYTES: &[u8] =
                include_bytes!("../../../../../config/scail2-memory-pin-coverage.json");
            // Approve this digest only after reviewing the entire Git delta and matching both
            // closure digests offline. Runtime uses the certified immutable pair.
            const APPROVED: &str =
                "b5ac978a0832ad15a6a39d5662ae0822631892c94afb9eb3ed5fecb5cbc290ae";
            approved_pin_coverage(BYTES, APPROVED).unwrap_or_default()
        })
        .iter()
        .any(|entry| entry.matches(capture, target, backend))
}

#[cfg(test)]
#[test]
fn scail_plural_pin_coverage_binds_source_target_backend_and_reviewed_digests() {
    let capture = "a".repeat(40);
    let target = "b".repeat(40);
    let unrelated = "c".repeat(40);
    let fixture = serde_json::json!([{"capture_revision":capture,"target_revision":target,"backend":"mlx",
        "provider_digest":"1".repeat(64),"shared_contract_digest":"2".repeat(64),"tree_diff_sha256":"3".repeat(64)}]);
    let bytes = serde_json::to_vec(&fixture).unwrap();
    let approved = sceneworks_core::hashing::sha256_bytes(&bytes);
    let rows = approved_pin_coverage(&bytes, &approved).unwrap();
    assert!(rows[0].matches(&capture, &target, Scail2FitBackend::Mlx));
    assert!(!rows[0].matches(&unrelated, &target, Scail2FitBackend::Mlx));
    assert!(!rows[0].matches(&capture, &unrelated, Scail2FitBackend::Mlx));
    assert!(!rows[0].matches(&capture, &target, Scail2FitBackend::Cuda));
    for key in [
        "provider_digest",
        "shared_contract_digest",
        "tree_diff_sha256",
    ] {
        let mut tampered = fixture.clone();
        tampered[0][key] = serde_json::json!("9".repeat(64));
        assert!(
            approved_pin_coverage(&serde_json::to_vec(&tampered).unwrap(), &approved).is_none()
        );
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
                    PhysicalBounds::WindowsJobAndWddm {
                        host,
                        device,
                        capture_adapter,
                    },
                ) if capture_adapter.valid() => (
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
