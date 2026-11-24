//! Compact projection of a completed physical capture. The approved index hash binds
//! the raw receipt, native axes and each independently reviewed upper bound together.
use gen_core::generator::scail2_fit::{Scail2FitAxes, Scail2FitBackend, Scail2FitWitness};
use serde::Deserialize;
use std::sync::OnceLock;

const INDEX: &[u8] = include_bytes!("../../../../../config/scail2-plural-memory.json");
const APPROVED_INDEX_SHA256: &str =
    "31f82fe81675f38120b057b714982df8238b87fedbcd74c8cb3d01a131c875fa";

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
    count_accepted: Option<bool>,
    recovery_certificate_sha256: Option<String>,
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
        // Provenance only: the physical card that ran the capture. Hosts match the hardware
        // class (device model, total device memory, driver) and native execution axes, so an
        // identical second card, or the same card after a reboot renumbers its LUID, qualifies.
        capture_adapter: CaptureAdapter,
        /// `nvidia-smi memory.total` (MiB) of the capture card.
        device_total_mib: u64,
        device_interval: Option<DeviceInterval>,
        host: Envelope,
        device: Envelope,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum DeviceInterval {
    LoadThroughCompletedRgb,
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

/// The hardware class a capture was measured on. Two physical cards of one class run the
/// same native allocations, so a measurement on either prices both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct HostClass {
    pub(super) device: String,
    pub(super) runtime: String,
    /// Total accelerator memory in MiB for a discrete CUDA device; `None` for unified memory.
    pub(super) device_total_mib: Option<u64>,
}

impl HostClass {
    pub(super) fn new(device: &str, runtime: &str, device_total_mib: Option<u64>) -> Option<Self> {
        let (device, runtime) = (device.trim(), runtime.trim());
        (!device.is_empty()
            && !runtime.is_empty()
            && !runtime.contains(['\n', ','])
            && device_total_mib != Some(0))
        .then(|| Self {
            device: device.to_owned(),
            runtime: runtime.to_owned(),
            device_total_mib,
        })
    }
}

pub(super) struct QualifiedWitness {
    platform: String,
    inference_revision: String,
    pub(super) class: HostClass,
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

    /// The capture's physical card identity is provenance and never selects a host.
    pub(super) fn matches_host(&self, host: &HostClass) -> bool {
        self.class == *host
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
                "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570";
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
    fn source_accepted(&self) -> bool {
        const ARCHIVED_SIX_ARTIFACT: u64 = 14_719_969_057;
        const ARCHIVED_SIX_RECEIPT: &str =
            "750313a3cdb89dd4a0e4e49b31935949301b309fe9410f2bf0ce47cb4628575f";
        const RECOVERY_CERTIFICATE: &str =
            "dcc7c1eb186b001c75f4a9df2d49dd06501e8064be7d1818adb2ea59031c093f";
        const FOUR_COUNT_ARTIFACT: u64 = 14_725_116_132;
        const FOUR_COUNT_RECEIPT: &str =
            "d423f72d816ac6145c7553623a26e99f67fd59db8e83ab228a89ede8698ce8af";
        const FIVE_COUNT_ARTIFACT: u64 = 14_735_122_440;
        const FIVE_COUNT_RECEIPT: &str =
            "d711f34b0c83b54b41b147921b9601fe999b74ccaec8823b6dceadcd6d9fc545";
        for (artifact, receipt, revision) in [
            (
                FOUR_COUNT_ARTIFACT,
                FOUR_COUNT_RECEIPT,
                "f5c9ff80acd00334fd668e4ecc8d56be3347affd",
            ),
            (
                FIVE_COUNT_ARTIFACT,
                FIVE_COUNT_RECEIPT,
                "4fdd13cce7f590be0aed3c52a73a9cea9e6d1b54",
            ),
        ] {
            if self.artifact_id != artifact && self.receipt_sha256 != receipt {
                continue;
            }
            let (certificate, interval) = match (artifact, self.axes.references) {
                (FOUR_COUNT_ARTIFACT, 2 | 4) | (FIVE_COUNT_ARTIFACT, 2 | 4 | 5) => (None, false),
                (FIVE_COUNT_ARTIFACT, 6) => (None, true),
                (FOUR_COUNT_ARTIFACT, 3) => (
                    Some("c18632894d2d40c7d2b0eea519657714d873a611ca9e6f0845b0809dab4a024d"),
                    true,
                ),
                (FOUR_COUNT_ARTIFACT, 5) => (
                    Some("bd8567c64d120f24e9821747d94b63faec3e5919930ce65dccb5a79f7df86fe7"),
                    true,
                ),
                (FIVE_COUNT_ARTIFACT, 3) => (
                    Some("ca4696364a2a4c951b76f58879e0d25b7e48fdd4d6df5d9e7b605c313cbfd02d"),
                    true,
                ),
                _ => return false,
            };
            return self.artifact_id == artifact
                && self.receipt_sha256 == receipt
                && self.inference_revision == revision
                && !self.accepted
                && self.count_accepted == Some(certificate.is_none())
                && self.recovery_certificate_sha256.as_deref() == certificate
                && self.platform == "windows"
                && matches!(&self.bounds, PhysicalBounds::WindowsJobAndWddm { capture_adapter, device: Envelope::MeasuredAggregate {..}, device_interval,.. }
                    if capture_adapter.cuda_uuid == "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0"
                        && capture_adapter.luid == "luid_0x00000000_0x0001f603" && capture_adapter.node_mask == 1
                        && device_interval.is_some() == interval);
        }
        if self.count_accepted.is_some() {
            return false;
        }
        match self.recovery_certificate_sha256.as_deref() {
            None => {
                self.accepted
                    && self.artifact_id != ARCHIVED_SIX_ARTIFACT
                    && self.receipt_sha256 != ARCHIVED_SIX_RECEIPT
            }
            Some(certificate) => {
                !self.accepted
                    && certificate == RECOVERY_CERTIFICATE
                    && self.platform == "windows"
                    && self.axes.references == 6
                    && self.artifact_id == ARCHIVED_SIX_ARTIFACT
                    && self.receipt_sha256 == ARCHIVED_SIX_RECEIPT
                    && self.inference_revision == "7f6c4cca3fc053030bd71f0bfce6886adb762eb6"
            }
        }
    }

    fn qualify(self) -> Result<QualifiedWitness, String> {
        if !self.source_accepted()
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
        let (host, device, unified, decode_cap, device_total_mib) =
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
                    None,
                ),
                (
                    "windows",
                    Scail2FitBackend::Cuda,
                    PhysicalBounds::WindowsJobAndWddm {
                        host,
                        device,
                        capture_adapter,
                        device_total_mib,
                        device_interval,
                    },
                ) if capture_adapter.valid()
                    && device_total_mib > 0
                    && (self.accepted
                        || self.count_accepted == Some(true)
                        || device_interval.is_some())
                    && (device_interval.is_none()
                        || matches!(device, Envelope::MeasuredAggregate { .. })) =>
                {
                    (
                        host.price(self.axes.floor_host_bytes)?,
                        device.price(self.axes.floor_device_bytes)?,
                        0,
                        None,
                        Some(device_total_mib),
                    )
                }
                _ => {
                    return Err(
                        "capture has no verified physical bounds for this platform and provider"
                            .into(),
                    )
                }
            };
        let class = HostClass::new(&self.device, &self.runtime, device_total_mib)
            .filter(|class| class.device == self.device && class.runtime == self.runtime)
            .ok_or("capture has no exact hardware class")?;
        Ok(QualifiedWitness {
            platform: self.platform,
            inference_revision: self.inference_revision,
            class,
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

#[test]
fn packaged_evidence_covers_every_plural_count_at_the_pinned_inference_revision() {
    let pin = sceneworks_core::inference_provenance::INFERENCE_REVISION;
    let rows = packaged().unwrap();
    for platform in ["macos", "windows"] {
        for count in 2..=6 {
            assert!(
                rows.iter().any(|r| r.covers(platform, pin, count)),
                "{platform} {count}"
            );
        }
    }
}

#[test]
fn packaged_cuda_evidence_matches_the_hardware_class_not_the_capture_card() {
    use serde_json::{json, Value};
    let pin = sceneworks_core::inference_provenance::INFERENCE_REVISION;
    let class = |device: &str, runtime: &str, mib| HostClass::new(device, runtime, mib).unwrap();
    let rtx = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition";
    let host = class(rtx, "596.36", Some(97_887));
    let priced = |rows: &[QualifiedWitness], host: &HostClass| {
        (2..=6)
            .flat_map(|count| {
                rows.iter()
                    .filter(move |r| r.covers("windows", pin, count) && r.matches_host(host))
                    .map(move |r| {
                        let w = &r.witness;
                        (
                            count,
                            r.artifact_id,
                            r.evidence_sha256.clone(),
                            w.axes.references,
                            w.gross_host_bytes,
                            w.gross_device_bytes,
                        )
                    })
            })
            .collect::<Vec<_>>()
    };
    let packaged = packaged().unwrap();
    let gpu0 = priced(packaged, &host);
    assert!((2..=6).all(|count| gpu0.iter().any(|q| q.0 == count)));
    // The same captures attributed to the second identical card (or a renumbered LUID after a
    // reboot) qualify and price every count identically on the same class of host.
    let mut moved = serde_json::from_slice::<Vec<Value>>(INDEX).unwrap();
    for row in moved.iter_mut().filter(|r| r["platform"] == "windows") {
        row["bounds"]["capture_adapter"] = json!({"cuda_uuid":"GPU-e4b79931-7be6-f216-460a-f5405cfafffe",
            "luid":"luid_0x00000000_0x0002a1c4", "node_mask":2});
    }
    let moved = serde_json::to_vec(&moved).unwrap();
    let moved = qualify_index(&moved, &sceneworks_core::hashing::sha256_bytes(&moved)).unwrap();
    assert_eq!(priced(&moved, &host), gpu0);
    // Every other class axis still refuses.
    for other in [
        class(
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "596.36",
            Some(97_887),
        ),
        class(rtx, "596.36", Some(48_943)),
        class(rtx, "591.44", Some(97_887)),
        class(rtx, "596.36", None),
    ] {
        assert!(priced(packaged, &other).is_empty(), "{other:?}");
    }
    // A CUDA capture without its card capacity has no hardware class.
    let mut rows = serde_json::from_slice::<Vec<Value>>(INDEX).unwrap();
    let windows = rows
        .iter()
        .position(|r| r["platform"] == "windows")
        .unwrap();
    for total in [None, Some(json!(0))] {
        let bounds = rows[windows]["bounds"].as_object_mut().unwrap();
        match &total {
            Some(value) => bounds.insert("device_total_mib".into(), value.clone()),
            None => bounds.remove("device_total_mib"),
        };
        let bytes = serde_json::to_vec(&rows).unwrap();
        assert!(qualify_index(&bytes, &sceneworks_core::hashing::sha256_bytes(&bytes)).is_err());
    }
}

#[test]
fn continuation_keeps_each_original_count_and_its_own_certificate() {
    use serde_json::{json, Value};
    let certificates = [
        "c18632894d2d40c7d2b0eea519657714d873a611ca9e6f0845b0809dab4a024d",
        "bd8567c64d120f24e9821747d94b63faec3e5919930ce65dccb5a79f7df86fe7",
    ];
    let original = serde_json::from_slice::<Vec<Value>>(INDEX)
        .unwrap()
        .remove(0);
    let mut row = original.clone();
    row["platform"] = json!("windows");
    row["axes"]["backend"] = json!("cuda");
    row["artifact_id"] = json!(14_725_116_132_u64);
    row["receipt_sha256"] =
        json!("d423f72d816ac6145c7553623a26e99f67fd59db8e83ab228a89ede8698ce8af");
    row["inference_revision"] = json!("f5c9ff80acd00334fd668e4ecc8d56be3347affd");
    row["accepted"] = json!(false);
    let measured = json!({"basis":"measured_aggregate", "observed_peak_bytes":1,"classified_peak_bytes":1,
                          "co_live_extra_bytes":0,"uncertainty_allowance_bytes":0,"review_sha256":"1".repeat(64)});
    row["bounds"] = json!({"measurement":"windows_job_and_wddm", "device":measured, "host":measured,
        "capture_adapter":{"cuda_uuid":"GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0",
                           "luid":"luid_0x00000000_0x0001f603", "node_mask":1}, "device_total_mib":97887});
    let accepted = |value: &Value| {
        serde_json::from_value::<Capture>(value.clone())
            .unwrap()
            .source_accepted()
    };
    let mut full = row.clone();
    full["artifact_id"] = json!(14_735_122_440_u64);
    full["inference_revision"] = json!("4fdd13cce7f590be0aed3c52a73a9cea9e6d1b54");
    full["receipt_sha256"] =
        json!("d711f34b0c83b54b41b147921b9601fe999b74ccaec8823b6dceadcd6d9fc545");
    for count in 2..=6 {
        let mut r = full.clone();
        r["axes"]["references"] = json!(count);
        r["count_accepted"] = json!(count != 3);
        if count == 3 {
            r["recovery_certificate_sha256"] =
                json!("ca4696364a2a4c951b76f58879e0d25b7e48fdd4d6df5d9e7b605c313cbfd02d");
        }
        if matches!(count, 3 | 6) {
            r["bounds"]["device_interval"] = json!("load_through_completed_rgb");
        }
        assert!(accepted(&r));
        let mut crossed = r.clone();
        crossed["recovery_certificate_sha256"] = json!(certificates[0]);
        assert!(!accepted(&crossed));
        let mut crossed = r.clone();
        crossed["artifact_id"] = row["artifact_id"].clone();
        assert!(!accepted(&crossed));
        let mut crossed = r.clone();
        crossed["count_accepted"] = json!(count == 3);
        assert!(!accepted(&crossed));
        if matches!(count, 3 | 6) {
            r["bounds"]
                .as_object_mut()
                .unwrap()
                .remove("device_interval");
        } else {
            r["bounds"]["device_interval"] = json!("load_through_completed_rgb");
        }
        assert!(!accepted(&r));
    }
    for count in 2..=5 {
        let mut r = row.clone();
        r["axes"]["references"] = json!(count);
        let ordinary = count % 2 == 0;
        r["count_accepted"] = json!(ordinary);
        if !ordinary {
            r["recovery_certificate_sha256"] = json!(certificates[(count - 3) / 2]);
            r["bounds"]["device_interval"] = json!("load_through_completed_rgb");
        }
        assert!(accepted(&r));
        for (name, crossed) in [
            ("artifact_id", json!(1)),
            ("receipt_sha256", json!("0".repeat(64))),
            ("inference_revision", json!("0".repeat(40))),
            ("count_accepted", json!(!ordinary)),
            ("accepted", json!(true)),
        ] {
            let mut invalid = r.clone();
            invalid[name] = crossed;
            assert!(!accepted(&invalid), "{count} {name}");
        }
        let mut crossed = r.clone();
        crossed["recovery_certificate_sha256"] = json!(if count == 3 {
            certificates[1]
        } else {
            certificates[0]
        });
        assert!(!accepted(&crossed));
        if !ordinary {
            r.as_object_mut()
                .unwrap()
                .remove("recovery_certificate_sha256");
            assert!(!accepted(&r));
            r["accepted"] = json!(true);
            r.as_object_mut().unwrap().remove("count_accepted");
            r["bounds"]
                .as_object_mut()
                .unwrap()
                .remove("device_interval");
            assert!(!accepted(&r));
        }
    }
}
