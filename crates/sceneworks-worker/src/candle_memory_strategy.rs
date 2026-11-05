//! Candle image-provider adoption of the shared request-scoped memory selector.
//!
//! Static capability declarations never authorize an optimized request at a MEASURED grade. Exact
//! authoritative records from the packaged evidence bundle are the only measured optimized
//! candidates. Since sc-18097 (epic 18093 R1b, the candle mirror of sc-18096) every other
//! implemented optimized rung of a CERTIFIED artifact additionally carries a synthesized
//! ESTIMATE candidate — priced per rung through the image derivation law from the cell's measured
//! memory anchor where one exists (sc-22664, epic 22657 E4), else from the raw manifest
//! `vramGbByTier`/`sequentialPeakGb` rows, never a promised unmeasured saving — graded by the
//! shared selector behind the candle estimate margin
//! (`crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD`; CUDA OOM is a recoverable `Err`, so the
//! margin is looser than MLX's). The ladder's operational reserve is charged ONCE, against the
//! selector budget (`crate::vram_gate::ladder_reserve_gb`: the measured idle baseline plus a named
//! margin), and never inside a candidate's peak. Any eligible measured candidate at the same rung
//! supersedes the estimate, so measured-current admission is byte-for-byte unchanged; an
//! UNCERTIFIED artifact gets no floors at all (the manifest rows describe the certified bytes,
//! not an imported checkpoint) and keeps its resident-estimate-only behavior.

#[cfg(test)]
use crate::memory_route_registry::SD35_REQUEST_EVIDENCE_REVISION;
pub(crate) use crate::memory_route_registry::{
    KOLORS_REQUEST_EVIDENCE_REVISION, PULID_FLUX_REQUEST_EVIDENCE_REVISION,
    SDXL_REQUEST_EVIDENCE_REVISION,
};
use gen_core::{
    GenerationMemory, LoadSpec, MemoryBackend, MemoryCacheState, MemoryConformanceState,
    MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey, MemoryGeometry, MemoryMode,
    MemoryNumericTier, MemoryParityContract, MemoryParityResult, MemoryRunContext, MemorySelection,
    MemoryStrategy, Precision, Quant,
};
use sceneworks_core::memory_anchor::sdxl_cuda_memory_endpoints::SdxlCudaMemoryEndpoints;
use sceneworks_core::memory_calibration::{
    Backend as CalibrationBackend, CalibrationBinding, EvidenceQuery, EvidenceVerdict,
    Geometry as CalibrationGeometry, LoadShapeKey, QualityResult, RequiredNullable, StrategyRung,
};
use serde_json::{Map as JsonObject, Value};
use sha2::Digest;

use crate::memory_strategy::{Budget, Candidate, RequestScope, Selection};
use crate::vram_gate::VramBudget;
use crate::{WorkerError, WorkerResult};

const BYTES_PER_GIB: f64 = crate::fit_gate::BYTES_PER_GIB;
const CHROMA_ADAPTER_OVERLAY_PREFIX: &str = "chroma.adapters.ordered-additive.sha256:";
const IDEOGRAM_ADAPTER_OVERLAY_PREFIX: &str = "ideogram.adapters.ordered-additive.sha256:";
const IDEOGRAM_TURBO_ADAPTER_PREFIX: &str = "ideogram.turbo-time.sha256:";
const IDEOGRAM_PID_PREFIX: &str = "ideogram.pid.flux2.sha256:";
const IDEOGRAM_PHYSICAL_RECEIPT_PREFIX: &str = "ideogram.physical.sha256:";
const SANA_PHYSICAL_RECEIPT_PREFIX: &str = "sana.candle.dense.physical.sha256:";
const SD35_PHYSICAL_RECEIPT_PREFIX: &str = "sd3.5.candle.physical.sha256:";
const SD35_ADAPTER_RECEIPT_PREFIX: &str = "sd3.5.adapters.ordered-additive.sha256:";

const KOLORS_PHYSICAL_RECEIPT_PREFIX: &str = "kolors.physical.sha256:";
const KOLORS_ADAPTER_RECEIPT_PREFIX: &str = "kolors.adapters.ordered.sha256:";
const KOLORS_IP_RECEIPT_PREFIX: &str = "kolors.ip.sha256:";
const KOLORS_CONTROL_RECEIPT_PREFIX: &str = "kolors.control.sha256:";
const KOLORS_PID_RECEIPT_PREFIX: &str = "kolors.pid.sdxl.sha256:";
const SDXL_OVERLAY_RECEIPT_DOMAIN: &str = "sdxl-candle-overlay-assembly-v1";
const SDXL_OVERLAY_RECEIPT_PREFIX: &str = "sdxl.overlay.ordered.sha256:";

fn is_chroma(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Chroma)
}

fn is_ideogram(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Ideogram)
}

fn is_sana(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Sana)
}

/// The FLUX 3 Image Klein provider ids, whose packages include the publisher FP8R/NVFP4 tiers.
fn is_flux3(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Flux3)
}

pub(crate) fn is_sd35(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Sd35)
}

fn is_sealed_kolors_bespoke(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::KolorsBespoke)
}

/// Every family whose admitted peak is priced from an exact provider receipt rather than a
/// manifest estimate. The executed load policy of such a family must equal the policy its receipt
/// admitted — see `image_jobs::base`'s cache-execution guard.
pub(crate) fn is_receipt_priced(engine_id: &str) -> bool {
    is_chroma(engine_id)
        || is_ideogram(engine_id)
        || is_sana(engine_id)
        || is_sd35(engine_id)
        || engine_id == "kolors"
        || is_sealed_kolors_bespoke(engine_id)
}

fn sd35_provider_overlay_identity(
    engine_id: &str,
    contract: &gen_core::MemoryProviderContract,
    declared_overlay: Option<&str>,
) -> WorkerResult<Option<String>> {
    let components = contract.resident_components();
    let adapters = components
        .iter()
        .filter(|component| component.id.starts_with(SD35_ADAPTER_RECEIPT_PREFIX))
        .collect::<Vec<_>>();
    let valid = adapters.iter().all(|component| {
        component.kind == gen_core::MemoryComponentKind::AdapterStack
            && component.resident_bytes > 0
            && component.id.len() == SD35_ADAPTER_RECEIPT_PREFIX.len() + 64
            && component.id[SD35_ADAPTER_RECEIPT_PREFIX.len()..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
    });
    if !valid || adapters.len() > 1 {
        return Err(WorkerError::InvalidPayload(format!(
            "{engine_id} returned malformed ordered additive-adapter facts"
        )));
    }
    match (declared_overlay, adapters.as_slice()) {
        (None, []) => Ok(None),
        (Some("lora"), [component]) => Ok(Some(component.id.clone())),
        (None, _) => Err(WorkerError::InvalidPayload(format!(
            "{engine_id} plain load crossed an adapter receipt"
        ))),
        (Some("lora"), _) => Err(WorkerError::InvalidPayload(format!(
            "{engine_id} adapter load lacks its exact provider receipt"
        ))),
        (Some(other), _) => Err(WorkerError::InvalidPayload(format!(
            "{engine_id} does not advertise overlay {other}"
        ))),
    }
}

fn validate_sd35_asset_facts(
    engine_id: &str,
    contract: &gen_core::MemoryProviderContract,
) -> WorkerResult<String> {
    let facts = contract.asset_facts;
    let physical = contract
        .resident_components()
        .iter()
        .filter(|component| component.id.starts_with(SD35_PHYSICAL_RECEIPT_PREFIX))
        .collect::<Vec<_>>();
    let complete = is_sd35(engine_id)
        && contract.provider_id == engine_id
        && facts.conditioning_bytes > 0
        && facts.transformer_bytes > 0
        && facts.decoder_bytes > 0
        && facts.base_bytes
            == facts
                .conditioning_bytes
                .saturating_add(facts.transformer_bytes)
                .saturating_add(facts.decoder_bytes)
        && contract.auxiliary_resident_bytes() == facts.overlay_bytes
        && matches!(physical.as_slice(), [component]
            if component.id.len() == SD35_PHYSICAL_RECEIPT_PREFIX.len() + 64
                && component.id[SD35_PHYSICAL_RECEIPT_PREFIX.len()..]
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
                && component.kind == gen_core::MemoryComponentKind::TransformerSubStack(
                    gen_core::TransformerComponent::Dit
                )
                && component.resident_bytes == facts.transformer_bytes
                && component.bounded_by == Some(MemoryStrategy::StagedResidency)
                && component.residency == gen_core::MemoryComponentResidency::WholeRender);
    if !complete {
        return Err(WorkerError::InvalidPayload(format!(
            "{engine_id} returned incomplete or crossed SD3.5 physical facts"
        )));
    }
    Ok(physical[0].id.clone())
}

pub(crate) fn sd35_physical_receipt_identity(
    engine_id: &str,
    contract: &gen_core::MemoryProviderContract,
) -> WorkerResult<String> {
    validate_sd35_asset_facts(engine_id, contract)
}

/// Resolve the public `lora` cell to the exact ordered adapter load identity sealed by the
/// provider. The public manifest intentionally groups LoRA and LoKr as one compatibility cell;
/// the provider handshake does not — it binds kind, order, scale, target, digest, and materialized
/// bytes into this receipt identity.
fn chroma_provider_overlay_identity(
    contract: &gen_core::MemoryProviderContract,
    declared_overlay: Option<&str>,
) -> WorkerResult<Option<String>> {
    let identities = contract
        .resident_components()
        .iter()
        .filter(|component| component.id.starts_with(CHROMA_ADAPTER_OVERLAY_PREFIX))
        .collect::<Vec<_>>();
    if identities.iter().any(|component| {
        component.kind != gen_core::MemoryComponentKind::AdapterStack
            || component.resident_bytes == 0
            || component.id.len() != CHROMA_ADAPTER_OVERLAY_PREFIX.len() + 64
            || !component.id[CHROMA_ADAPTER_OVERLAY_PREFIX.len()..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
    }) {
        return Err(WorkerError::InvalidPayload(
            "Chroma provider returned a malformed materialized adapter receipt".to_owned(),
        ));
    }
    match declared_overlay {
        None => {
            if identities.is_empty() {
                Ok(None)
            } else {
                Err(WorkerError::InvalidPayload(
                    "Chroma plain load crossed a provider adapter receipt".to_owned(),
                ))
            }
        }
        Some("lora") => match identities.as_slice() {
            [component] => Ok(Some(component.id.clone())),
            _ => Err(WorkerError::InvalidPayload(
                "Chroma adapter load is missing its singular exact provider receipt".to_owned(),
            )),
        },
        Some(other) => Err(WorkerError::InvalidPayload(format!(
            "Chroma does not advertise provider overlay {other}"
        ))),
    }
}

fn ideogram_provider_overlay_identity(
    engine_id: &str,
    contract: &gen_core::MemoryProviderContract,
    declared_overlay: Option<&str>,
) -> WorkerResult<Option<String>> {
    let components = contract.resident_components();
    let user = components
        .iter()
        .filter(|component| component.id.starts_with(IDEOGRAM_ADAPTER_OVERLAY_PREFIX))
        .collect::<Vec<_>>();
    let turbo = components
        .iter()
        .filter(|component| component.id.starts_with(IDEOGRAM_TURBO_ADAPTER_PREFIX))
        .collect::<Vec<_>>();
    let pid = components
        .iter()
        .filter(|component| component.id.starts_with(IDEOGRAM_PID_PREFIX))
        .collect::<Vec<_>>();
    let valid = |component: &&gen_core::MemoryResidentComponent, prefix: &str| {
        component.kind == gen_core::MemoryComponentKind::AdapterStack
            && component.resident_bytes > 0
            && component.id.len() == prefix.len() + 64
            && component.id[prefix.len()..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
    };
    if user
        .iter()
        .any(|component| !valid(component, IDEOGRAM_ADAPTER_OVERLAY_PREFIX))
        || turbo
            .iter()
            .any(|component| !valid(component, IDEOGRAM_TURBO_ADAPTER_PREFIX))
        || pid
            .iter()
            .any(|component| !valid(component, IDEOGRAM_PID_PREFIX))
        || (engine_id == "ideogram_4" && !turbo.is_empty())
        || (engine_id == "ideogram_4_turbo" && turbo.len() != 1)
    {
        return Err(WorkerError::InvalidPayload(
            "Ideogram provider returned malformed or crossed physical adapter receipts".to_owned(),
        ));
    }
    match declared_overlay {
        None if user.is_empty() => Ok(None),
        Some("lora") if user.len() == 1 => Ok(Some(user[0].id.clone())),
        None => Err(WorkerError::InvalidPayload(
            "Ideogram plain load crossed a provider user-adapter receipt".to_owned(),
        )),
        Some("lora") => Err(WorkerError::InvalidPayload(
            "Ideogram adapter load is missing its singular exact provider receipt".to_owned(),
        )),
        Some(other) => Err(WorkerError::InvalidPayload(format!(
            "Ideogram does not advertise provider overlay {other}"
        ))),
    }
}

fn validate_chroma_asset_facts(
    engine_id: &str,
    contract: &gen_core::MemoryProviderContract,
) -> WorkerResult<()> {
    let facts = contract.asset_facts;
    // Route identity is a conjunct of the receipt, exactly as in `validate_sana_asset_facts` and
    // `validate_sd35_asset_facts`. Without it a contract minted by another Chroma variant (or by
    // any provider at all) prices this request: the three Chroma turnkeys are different physical
    // tensor sets under one family name, so `provider_id` is the only fact that separates them.
    let complete = is_chroma(engine_id)
        && contract.provider_id == engine_id
        && facts.conditioning_bytes > 0
        && facts.transformer_bytes > 0
        && facts.decoder_bytes > 0
        && facts.base_bytes
            == facts
                .conditioning_bytes
                .saturating_add(facts.transformer_bytes)
                .saturating_add(facts.decoder_bytes)
        && contract.auxiliary_resident_bytes() == facts.overlay_bytes;
    if !complete {
        return Err(WorkerError::InvalidPayload(format!(
            "{engine_id} provider returned incomplete or crossed materialized Chroma asset facts"
        )));
    }
    Ok(())
}

fn validate_ideogram_asset_facts(
    engine_id: &str,
    contract: &gen_core::MemoryProviderContract,
    use_pid: bool,
    mode: &MemoryMode,
    has_phases: bool,
) -> WorkerResult<()> {
    let facts = contract.asset_facts;
    let pid_receipts = contract
        .resident_components()
        .iter()
        .filter(|component| component.id.starts_with(IDEOGRAM_PID_PREFIX))
        .count();
    let physical = contract
        .resident_components()
        .iter()
        .filter(|component| component.id.starts_with(IDEOGRAM_PHYSICAL_RECEIPT_PREFIX))
        .collect::<Vec<_>>();
    // The hires FINAL pass is the one request shape whose contract legitimately carries a PiD
    // receipt it will not decode through: `evaluate_shared_image` is called for it with
    // `use_pid && hires_fix.is_none()` (false) while the provider has already materialized the PiD
    // stack for the FIRST pass. That pass is identified by `has_phases`, which base.rs sets from
    // `hires_fix.is_some()`. Without the `has_phases` conjunct any single-pass image-to-image
    // request — an edit, a style variation — accepts a PiD-charged contract and is admitted
    // against a peak that includes bytes its own render never asked for.
    let pid_shape_matches = pid_receipts == usize::from(use_pid)
        || (!use_pid && has_phases && *mode == MemoryMode::ImageToImage && pid_receipts == 1);
    let complete = facts.conditioning_bytes > 0
        && facts.transformer_bytes > 0
        && facts.decoder_bytes > 0
        && facts.base_bytes
            == facts
                .conditioning_bytes
                .saturating_add(facts.transformer_bytes)
                .saturating_add(facts.decoder_bytes)
        && contract.auxiliary_resident_bytes() == facts.overlay_bytes
        && pid_shape_matches
        && matches!(physical.as_slice(), [component]
            if component.id.len() == IDEOGRAM_PHYSICAL_RECEIPT_PREFIX.len() + 64
                && component.id[IDEOGRAM_PHYSICAL_RECEIPT_PREFIX.len()..]
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
                && component.kind == gen_core::MemoryComponentKind::TransformerSubStack(
                    gen_core::TransformerComponent::Dit
                )
                && component.resident_bytes == facts.transformer_bytes
                && component.bounded_by == Some(MemoryStrategy::BoundedTransformerResidency))
        && matches!(engine_id, "ideogram_4" | "ideogram_4_turbo");
    if !complete {
        return Err(WorkerError::InvalidPayload(
            "Ideogram provider returned incomplete or crossed materialized asset facts".to_owned(),
        ));
    }
    Ok(())
}

fn validate_sana_asset_facts(
    engine_id: &str,
    contract: &gen_core::MemoryProviderContract,
) -> WorkerResult<String> {
    let expected_fingerprint = match engine_id {
        "sana_1600m" => "sana-candle-dense-base-full-ladder-v1",
        "sana_sprint_1600m" => "sana-candle-dense-sprint-full-ladder-v1",
        _ => {
            return Err(WorkerError::InvalidPayload(
                "unknown SANA receipt-priced provider identity".to_owned(),
            ))
        }
    };
    let facts = contract.asset_facts;
    let physical = contract
        .resident_components()
        .iter()
        .filter(|component| component.id.starts_with(SANA_PHYSICAL_RECEIPT_PREFIX))
        .collect::<Vec<_>>();
    let complete = contract.provider_id == engine_id
        && contract
            .calibration
            .as_ref()
            .is_some_and(|calibration| calibration.fingerprint == expected_fingerprint)
        && facts.conditioning_bytes > 0
        && facts.transformer_bytes > 0
        && facts.decoder_bytes > 0
        && facts.base_bytes
            == facts
                .conditioning_bytes
                .saturating_add(facts.transformer_bytes)
                .saturating_add(facts.decoder_bytes)
        && facts.overlay_bytes == 0
        && contract.auxiliary_resident_bytes() == 0
        && matches!(physical.as_slice(), [component]
            if component.id.len() == SANA_PHYSICAL_RECEIPT_PREFIX.len() + 64
                && component.id[SANA_PHYSICAL_RECEIPT_PREFIX.len()..]
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
                && component.kind == gen_core::MemoryComponentKind::TransformerSubStack(
                    gen_core::TransformerComponent::Dit
                )
                && component.resident_bytes == facts.transformer_bytes
                && component.bounded_by == Some(MemoryStrategy::StagedResidency)
                && component.residency == gen_core::MemoryComponentResidency::WholeRender);
    if !complete {
        return Err(WorkerError::InvalidPayload(format!(
            "{engine_id} provider returned incomplete or crossed SANA physical facts"
        )));
    }
    Ok(physical[0].id.clone())
}

pub(crate) fn sana_physical_receipt_identity(
    engine_id: &str,
    contract: &gen_core::MemoryProviderContract,
) -> WorkerResult<String> {
    validate_sana_asset_facts(engine_id, contract)
}

fn validate_kolors_asset_facts(
    engine_id: &str,
    contract: &gen_core::MemoryProviderContract,
    use_pid: bool,
) -> WorkerResult<String> {
    let facts = contract.asset_facts;
    let components = contract.resident_components();
    let receipt = components
        .iter()
        .filter(|component| component.id.starts_with(KOLORS_PHYSICAL_RECEIPT_PREFIX))
        .collect::<Vec<_>>();
    let prefixed = |prefix: &str| {
        components
            .iter()
            .filter(|component| component.id.starts_with(prefix))
            .collect::<Vec<_>>()
    };
    let adapters = prefixed(KOLORS_ADAPTER_RECEIPT_PREFIX);
    let ip = prefixed(KOLORS_IP_RECEIPT_PREFIX);
    let control = prefixed(KOLORS_CONTROL_RECEIPT_PREFIX);
    let pid = prefixed(KOLORS_PID_RECEIPT_PREFIX);
    let valid_id = |id: &str, prefix: &str| {
        id.len() == prefix.len() + 64
            && id[prefix.len()..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
    };
    let exact_route = match engine_id {
        "kolors" => ip.is_empty() && control.is_empty() && pid.len() == usize::from(use_pid),
        "candle_kolors_ipadapter" => ip.len() == 1 && control.is_empty() && pid.is_empty(),
        "candle_kolors_control" => {
            control.len() == 1 && ip.is_empty() && pid.len() == usize::from(use_pid)
        }
        _ => false,
    };
    let overlay_sum = adapters
        .iter()
        .chain(ip.iter())
        .chain(control.iter())
        .chain(pid.iter())
        .map(|component| component.resident_bytes)
        .sum::<u64>();
    let valid = facts.conditioning_bytes > 0
        && facts.transformer_bytes > 0
        && facts.decoder_bytes > 0
        && facts.base_bytes
            == facts
                .conditioning_bytes
                .saturating_add(facts.transformer_bytes)
                .saturating_add(facts.decoder_bytes)
        && facts.overlay_bytes == overlay_sum
        && exact_route
        && adapters.len() <= 1
        && matches!(receipt.as_slice(), [component]
            if valid_id(&component.id, KOLORS_PHYSICAL_RECEIPT_PREFIX)
                && component.kind == gen_core::MemoryComponentKind::TransformerSubStack(
                    gen_core::TransformerComponent::Dit
                )
                && component.resident_bytes == facts.transformer_bytes
                && component.bounded_by == Some(MemoryStrategy::StagedResidency)
                && component.residency == gen_core::MemoryComponentResidency::WholeRender)
        && adapters.iter().all(|component| {
            valid_id(&component.id, KOLORS_ADAPTER_RECEIPT_PREFIX)
                && component.kind == gen_core::MemoryComponentKind::AdapterStack
                && component.resident_bytes > 0
                && component.bounded_by == Some(MemoryStrategy::StagedResidency)
                && component.residency == gen_core::MemoryComponentResidency::WholeRender
        })
        && ip.iter().all(|component| {
            valid_id(&component.id, KOLORS_IP_RECEIPT_PREFIX)
                && component.kind == gen_core::MemoryComponentKind::IpAdapter
                && component.resident_bytes > 0
                && component.bounded_by == Some(MemoryStrategy::StagedResidency)
                && component.residency == gen_core::MemoryComponentResidency::WholeRender
        })
        && control.iter().all(|component| {
            valid_id(&component.id, KOLORS_CONTROL_RECEIPT_PREFIX)
                && component.kind == gen_core::MemoryComponentKind::ControlBranch
                && component.resident_bytes > 0
                && component.bounded_by == Some(MemoryStrategy::StagedResidency)
                && component.residency == gen_core::MemoryComponentResidency::WholeRender
        })
        && pid.iter().all(|component| {
            valid_id(&component.id, KOLORS_PID_RECEIPT_PREFIX)
                && component.kind == gen_core::MemoryComponentKind::AdapterStack
                && component.resident_bytes > 0
                && component.bounded_by == Some(MemoryStrategy::StagedResidency)
                && component.residency == gen_core::MemoryComponentResidency::WholeRender
        });
    if !valid {
        return Err(WorkerError::InvalidPayload(format!(
            "{engine_id} provider returned incomplete or crossed Kolors physical receipts"
        )));
    }
    Ok(receipt[0].id.clone())
}

pub(crate) fn kolors_overlay_receipt_identity(
    engine_id: &str,
    contract: &gen_core::MemoryProviderContract,
    use_pid: bool,
) -> WorkerResult<Option<String>> {
    validate_kolors_asset_facts(engine_id, contract, use_pid)?;
    let accepted: Vec<&str> = match engine_id {
        "kolors" => vec![
            KOLORS_PHYSICAL_RECEIPT_PREFIX,
            KOLORS_ADAPTER_RECEIPT_PREFIX,
            KOLORS_PID_RECEIPT_PREFIX,
        ],
        "candle_kolors_ipadapter" => {
            vec![
                KOLORS_PHYSICAL_RECEIPT_PREFIX,
                KOLORS_IP_RECEIPT_PREFIX,
                KOLORS_ADAPTER_RECEIPT_PREFIX,
            ]
        }
        "candle_kolors_control" => vec![
            KOLORS_PHYSICAL_RECEIPT_PREFIX,
            KOLORS_CONTROL_RECEIPT_PREFIX,
            KOLORS_ADAPTER_RECEIPT_PREFIX,
            KOLORS_PID_RECEIPT_PREFIX,
        ],
        _ => vec![],
    };
    let identities = contract
        .resident_components()
        .iter()
        .filter(|component| {
            accepted
                .iter()
                .any(|prefix| component.id.starts_with(prefix))
        })
        .map(|component| component.id.as_str())
        .collect::<Vec<_>>();
    if identities.is_empty() {
        Ok(None)
    } else {
        Ok(Some(identities.join("+")))
    }
}

pub(crate) fn ideogram_physical_receipt_identity(
    contract: &gen_core::MemoryProviderContract,
) -> WorkerResult<String> {
    let identities = contract
        .resident_components()
        .iter()
        .filter(|component| component.id.starts_with(IDEOGRAM_PHYSICAL_RECEIPT_PREFIX))
        .map(|component| component.id.clone())
        .collect::<Vec<_>>();
    match identities.as_slice() {
        [identity]
            if identity.len() == IDEOGRAM_PHYSICAL_RECEIPT_PREFIX.len() + 64
                && identity[IDEOGRAM_PHYSICAL_RECEIPT_PREFIX.len()..]
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()) =>
        {
            Ok(identity.clone())
        }
        _ => Err(WorkerError::InvalidPayload(
            "Ideogram provider omitted its singular canonical physical receipt".to_owned(),
        )),
    }
}

fn chroma_base_phase_floor_bytes(contract: &gen_core::MemoryProviderContract) -> u64 {
    let facts = contract.asset_facts;
    facts
        .conditioning_bytes
        .max(facts.transformer_bytes)
        .max(facts.decoder_bytes)
}

fn receipt_base_phase_floor_bytes(contract: &gen_core::MemoryProviderContract) -> u64 {
    chroma_base_phase_floor_bytes(contract)
}

/// The output pixel count the manifest's candle rows were MEASURED at: `candle.vramMeasuredPixels`,
/// or the rows' documented 1024x1024 when the block states none (every shipped candle block
/// states it; the fallback is the geometry the rows have always been read as).
pub(crate) const MANIFEST_ROW_DEFAULT_MEASURED_PIXELS: u64 = 1_048_576;

pub(crate) fn manifest_measured_pixels(manifest: &JsonObject<String, Value>) -> u64 {
    manifest
        .get("candle")
        .and_then(|value| value.get("vramMeasuredPixels"))
        .and_then(Value::as_u64)
        .filter(|pixels| *pixels > 0)
        .unwrap_or(MANIFEST_ROW_DEFAULT_MEASURED_PIXELS)
}

/// The geometry the derivation law prices for one request (sc-25690).
///
/// Normally the request's own. An entry that DECLARES its per-megapixel activation term
/// (`candle.imageActivationGbPerMegapixel` — Iris-3B) has every image of a native prompt batch and
/// every megapixel beyond the ONE image at 1024² its floors carry charged by
/// `lane_controls::declared_activation_overlay_bytes`, which reaches every candidate here as
/// `runtime_overlay_bytes`. The law scales an anchor's activation residue by the request's batch
/// and area too, so pricing such an entry at the request's own geometry would charge the same
/// activations twice: once declared, once derived. For a declaring entry the law therefore prices
/// one image at no more than the floors' area, and the declared term carries the rest.
///
/// The admitted `MemoryGeometry` the provider's request scope receives is NOT this value — it
/// stays the request's own, because the scope bounds the images a request may render by `batch`.
fn law_geometry(manifest: &JsonObject<String, Value>, geometry: MemoryGeometry) -> MemoryGeometry {
    if crate::image_jobs::declared_activation_gb_per_megapixel(manifest).is_none() {
        return geometry;
    }
    let floor_pixels = crate::image_jobs::DECLARED_ACTIVATION_FLOOR_PIXELS;
    let pixels = u64::from(geometry.width) * u64::from(geometry.height);
    let (width, height) = if pixels > floor_pixels {
        let edge = u32::try_from(floor_pixels.isqrt())
            .unwrap_or(u32::MAX)
            .max(1);
        (edge, edge)
    } else {
        (geometry.width, geometry.height)
    };
    MemoryGeometry {
        width,
        height,
        batch: 1,
        ..geometry
    }
}

fn scale_ideogram_hires_envelope(
    engine_id: &str,
    mode: &MemoryMode,
    manifest: &JsonObject<String, Value>,
    geometry: MemoryGeometry,
    bytes: u64,
) -> u64 {
    if !is_ideogram(engine_id) || *mode != MemoryMode::ImageToImage {
        return bytes;
    }
    let measured_pixels = manifest_measured_pixels(manifest);
    let request_pixels = u64::from(geometry.width).saturating_mul(u64::from(geometry.height));
    let numerator = request_pixels.max(measured_pixels);
    bytes
        .saturating_mul(numerator)
        .saturating_add(measured_pixels - 1)
        / measured_pixels
}

pub(crate) struct CandleMemoryEvaluation {
    pub memory: Option<GenerationMemory>,
    pub context: MemoryRunContext,
    pub predicted_peak_gb: f64,
    /// A literal endpoint's observed whole-run peak. Admission uncertainty and reserve are
    /// reported separately; an intermediate derived configuration carries no measured claim.
    pub measured_endpoint_peak_bytes: Option<u64>,
    /// The exact lower-peak staged candidate for a receipt-priced provider. A warm cache entry may
    /// already have been loaded under `Sequential` even when this request's least-cost cold
    /// selection is Resident. In that case the cache policy requires the tighter loaded shape; this
    /// pre-admitted sibling lets the caller preserve that shape without inventing a post-load rung.
    pub warm_staged: Option<CandleWarmStagedEvaluation>,
    /// What priced the selected candidate (sc-22664, E7): a measured record, an anchor
    /// derivation, or a floor.
    pub basis: crate::memory_strategy::CandidateBasis,
    /// The selected rung's three derived phase peaks (before the runtime overlay charge) when the
    /// law priced it; `None` for a measured record, the resident live estimate, or an unscaled
    /// row. `predicted_peak_gb` is their max plus the overlay — the number the selector graded.
    pub phase_peaks: Option<sceneworks_core::memory_anchor::AnchorDerivedPhases>,
    /// The selector's own admission figures, so telemetry agrees with the selector by
    /// construction rather than by re-derivation.
    pub admitted: AdmittedBudget,
    /// What the admission decided about the provider's prefix KV cache (epic 25899): `None` for
    /// every provider that prices none. A `Cached` plan is already written onto [`Self::memory`]
    /// and counted in `context.predicted_peak_bytes` / [`Self::predicted_peak_gb`].
    pub prefix_kv_cache: Option<crate::execution_planner::PrefixKvCachePlan>,
}

/// Stable telemetry spelling of a rung, matching the `image_memory_strategy_selected` event the
/// receipt-priced lanes already emit.
pub(crate) const fn strategy_label(strategy: MemoryStrategy) -> &'static str {
    match strategy {
        MemoryStrategy::Resident => "resident",
        MemoryStrategy::StagedResidency => "staged_residency",
        MemoryStrategy::BoundedDecode => "bounded_decode",
        MemoryStrategy::BoundedAttention => "bounded_attention",
        MemoryStrategy::BoundedTransformerResidency => "bounded_transformer_residency",
    }
}

/// The admission figures of one selected candle strategy (sc-22664).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AdmittedBudget {
    /// The admitted peak (raw peak plus the policy allowance), GiB.
    pub needed_gb: f64,
    /// The selector's effective budget: free minus the reserve, GiB.
    pub available_gb: f64,
    /// The operational reserve charged once against the budget (`vram_gate::ladder_reserve_gb`).
    pub reserve_gb: f64,
}

impl CandleMemoryEvaluation {
    /// The `image_memory_strategy_selected` telemetry payload for this selection (sc-22664, epic
    /// 22657 E7): the selected rung, its parameters, the basis that priced it, the three derived
    /// phase peaks where the law priced it, and the selector's own admission figures. Pure, so the
    /// AC test can assert the event carries the rung and phases the selector chose.
    pub(crate) fn selection_telemetry(&self, engine_id: &str, tier_key: &str) -> Value {
        let selection = self.context.selection;
        let parameters = selection.parameters;
        let geometry = self.context.geometry;
        serde_json::json!({
            "backend": "candle",
            "route": engine_id,
            "actualTier": tier_key,
            "mode": self.context.mode.as_key(),
            "geometry": {
                "width": geometry.width,
                "height": geometry.height,
                "batch": geometry.batch,
                "frames": geometry.frames,
            },
            "referenceCount": geometry.reference_count,
            "overlay": self.context.overlay.clone(),
            "strategy": strategy_label(selection.strategy),
            "parameters": {
                "stageResidency": self.memory.as_ref().is_some_and(|memory| memory.stage_residency),
                "decodeTileEdge": parameters.decode_tile_edge,
                "decodeOverlap": parameters.decode_overlap,
                "attentionChunkSize": parameters.attention_chunk_size,
                "transformerWindowSize": parameters.transformer_window_size,
            },
            "basis": self.basis.as_key(),
            "authority": match self.context.optimization_authority {
                gen_core::MemoryOptimizationAuthority::Resident => "resident",
                gen_core::MemoryOptimizationAuthority::Estimated => "estimated",
                gen_core::MemoryOptimizationAuthority::Calibrated => "calibrated",
            },
            "cacheState": match self.context.cache_state {
                MemoryCacheState::Cold => "cold",
                MemoryCacheState::Warm => "warm",
            },
            "predictedPeakBytes": self.context.predicted_peak_bytes,
            "measuredEndpointPeakBytes": self.measured_endpoint_peak_bytes,
            "phasePeakBytes": self.phase_peaks.map(|phases| serde_json::json!({
                "conditioning": phases.conditioning,
                "denoise": phases.denoise,
                "decode": phases.decode,
            })),
            "admittedPeakGb": self.admitted.needed_gb,
            "availableGb": self.admitted.available_gb,
            "reserveGb": self.admitted.reserve_gb,
            "evidenceRevision": self.context.evidence_revision.clone(),
            "prefixKvCache": self.prefix_kv_cache.map(|plan| serde_json::json!({
                "selected": plan.mode.label(),
                "pricedBytes": plan.priced_bytes,
            })),
        })
    }
}

pub(crate) struct CandleWarmStagedEvaluation {
    pub memory: GenerationMemory,
    pub context: MemoryRunContext,
}

/// Bind an already admitted Ideogram request to the cache entry that will actually execute it.
/// A warm entry loaded under Sequential is a lower-peak shape than a newly selected Resident
/// request, so execution must retain the exact staged sibling selected during pre-load admission.
pub(crate) fn bind_ideogram_cache_execution(
    context: &mut MemoryRunContext,
    memory: &mut Option<GenerationMemory>,
    warm_staged: &mut Option<CandleWarmStagedEvaluation>,
    cache_state: MemoryCacheState,
    loaded_offload_policy: gen_core::OffloadPolicy,
) -> WorkerResult<()> {
    if loaded_offload_policy == gen_core::OffloadPolicy::Sequential
        && context.selection.strategy == MemoryStrategy::Resident
    {
        let staged = warm_staged.take().ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Ideogram warm Sequential cache entry has no exact pre-admitted staged strategy"
                    .to_owned(),
            )
        })?;
        *memory = Some(staged.memory);
        *context = staged.context;
    }
    context.cache_state = cache_state;
    Ok(())
}

#[cfg(native_lane)]
fn declared_component_floors(engine_id: &str) -> &'static [gen_core::ComponentPrecisionFloor] {
    crate::inference_runtime::media_descriptor(engine_id)
        .map(|descriptor| descriptor.capabilities.component_precision_floors)
        .unwrap_or(&[])
}

#[cfg(not(native_lane))]
fn declared_component_floors(_: &str) -> &'static [gen_core::ComponentPrecisionFloor] {
    &[]
}

fn active_component_floors(
    engine_id: &str,
    selected: Option<Quant>,
) -> &'static [gen_core::ComponentPrecisionFloor] {
    sceneworks_core::memory_contracts::active_component_floors(
        declared_component_floors(engine_id),
        selected,
    )
}

fn numeric_tier(engine_id: &str, tier: &str) -> Option<MemoryNumericTier> {
    let quant = match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        // sc-26241: FLUX 3's publisher packages are real numeric tiers of its provider (its memory
        // contract prices each from the package's own tensor headers), so the selector binds them
        // by identity. Other families keep NVFP4 off the shared selector.
        "fp8r" if is_flux3(engine_id) => Some(Quant::Fp8Rowwise),
        "nvfp4" if is_flux3(engine_id) => Some(Quant::Nvfp4),
        _ => return None,
    };
    Some(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: active_component_floors(engine_id, quant),
    })
}

struct RequestModeBinding {
    mode: MemoryMode,
    /// Catalog/matrix axis used to find the entry-specific calibration binding.
    calibration_key: String,
    /// Typed gen-core key used by the exact request scope (`MemoryMode::as_key`).
    scope_key: String,
}

fn request_mode(engine_id: &str, mode: &str) -> RequestModeBinding {
    let (mode, calibration_key) = match mode {
        "image_generation" | "text_to_image" => {
            (MemoryMode::TextToImage, "text_to_image".to_owned())
        }
        "style_variations" => (
            MemoryMode::Other("style_variations".to_owned()),
            "style_variations".to_owned(),
        ),
        "character_image" => (
            MemoryMode::Other("character_image".to_owned()),
            "character_image".to_owned(),
        ),
        // FLUX.2 Klein's reference alias is the same typed Edit coordinate as edit_image. Keep the
        // catalog axis at edit_image so an exact matrix calibration can promote that cell, while
        // the gen-core selector and provider receive MemoryMode::Edit / "edit".
        "reference" | "image_to_image" if engine_id == "flux2_klein_9b" => {
            (MemoryMode::Edit, "edit_image".to_owned())
        }
        "image_to_image" => (MemoryMode::ImageToImage, "image_to_image".to_owned()),
        "edit" | "edit_image" => (MemoryMode::Edit, "edit_image".to_owned()),
        _ => (MemoryMode::Other(mode.to_owned()), mode.to_owned()),
    };
    let scope_key = mode.as_key().to_owned();
    RequestModeBinding {
        mode,
        calibration_key,
        scope_key,
    }
}

fn request_mode_with_provider_override(
    engine_id: &str,
    public_mode: &str,
    provider_mode: Option<&str>,
) -> RequestModeBinding {
    let mut binding = request_mode(engine_id, public_mode);
    if let Some(provider_mode) = provider_mode {
        let provider_binding = request_mode(engine_id, provider_mode);
        binding.mode = provider_binding.mode;
        binding.scope_key = provider_binding.scope_key;
    }
    binding
}

fn strategy(rung: StrategyRung) -> MemoryStrategy {
    match rung {
        StrategyRung::Resident => MemoryStrategy::Resident,
        StrategyRung::StagedResidency => MemoryStrategy::StagedResidency,
        StrategyRung::BoundedDecode => MemoryStrategy::BoundedDecode,
        StrategyRung::BoundedAttention => MemoryStrategy::BoundedAttention,
        StrategyRung::BoundedTransformerResidency => MemoryStrategy::BoundedTransformerResidency,
    }
}

/// Read one binding's exact strategy parameters against the composition it ENGAGES (sc-17728).
///
/// The Candle sibling of `mlx_fit_gate::parse_evidence_parameters`, with the same rule and the same
/// shared derivation: a rung the composition engages must name its parameters, and a rung it does
/// not engage must not. Keying this on the selected rung's ordinal made a provider that implements
/// rung 4 with rungs 2 and 3 `Missing` unable to bind evidence at all.
fn parse_parameters(
    engaged: &[StrategyRung],
    parameters: &JsonObject<String, Value>,
) -> Option<gen_core::MemoryStrategyParameters> {
    const KEYS: [&str; 6] = [
        "decodeTileEdge",
        "decodeOverlap",
        "attentionChunkSize",
        "transformerWindowSize",
        "transformerWindowComponent",
        "stageResidency",
    ];
    if parameters.keys().any(|key| !KEYS.contains(&key.as_str())) {
        return None;
    }
    let integer = |key: &str, minimum: u32| {
        parameters
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value >= minimum)
    };
    let required = crate::memory_strategy::required_numeric_parameters(engaged);
    if required
        .iter()
        .any(|(key, minimum)| integer(key, *minimum).is_none())
    {
        return None;
    }
    if KEYS[..4]
        .iter()
        .filter(|key| !required.iter().any(|(required, _)| required == *key))
        .any(|key| parameters.contains_key(*key))
    {
        return None;
    }
    let engages_transformer = engaged.contains(&StrategyRung::BoundedTransformerResidency);
    let transformer_window_component = match parameters.get("transformerWindowComponent") {
        None => None,
        Some(Value::String(value)) if engages_transformer => Some(match value.as_str() {
            "dit" => gen_core::TransformerComponent::Dit,
            "text_encoder" => gen_core::TransformerComponent::TextEncoder,
            "both" => gen_core::TransformerComponent::Both,
            _ => return None,
        }),
        Some(_) => return None,
    };
    // Staging is an independent, explicit composition choice on gen-core selections. A record
    // cannot add it without declaring the same engaged rung, disable a prerequisite, or turn a
    // numeric/string value into permission to evict components.
    let stage_residency = match parameters.get("stageResidency") {
        None => None,
        Some(Value::Bool(true)) if engaged.contains(&StrategyRung::StagedResidency) => Some(true),
        Some(_) => return None,
    };
    Some(gen_core::MemoryStrategyParameters {
        stage_residency,
        decode_tile_edge: integer("decodeTileEdge", 1),
        decode_overlap: integer("decodeOverlap", 0),
        attention_chunk_size: integer("attentionChunkSize", 1),
        transformer_window_size: integer("transformerWindowSize", 1),
        transformer_window_component,
    })
}

fn text<'a>(object: &'a JsonObject<String, Value>, key: &str) -> Option<&'a str> {
    object
        .get(key)?
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn binding(object: &JsonObject<String, Value>) -> Option<CalibrationBinding> {
    let abi = u32::try_from(object.get("abi")?.as_u64()?).ok()?;
    let load_shape = match object.get("loadShape").and_then(Value::as_str) {
        Some("eager_materialization") => LoadShapeKey::EagerMaterialization,
        Some("deferred_materialization") => LoadShapeKey::DeferredMaterialization,
        Some(_) => return None,
        None if abi >= 2 => return None,
        None => LoadShapeKey::EagerMaterialization,
    };
    Some(CalibrationBinding {
        abi,
        load_shape,
        fingerprint: text(object, "fingerprint")?.to_owned(),
        scene_works_revision: text(object, "sceneWorksRevision")?.to_owned(),
        matrix_source_revision: text(object, "matrixSourceRevision")?.to_owned(),
        inference_revision: text(object, "inferenceRevision")?.to_owned(),
        inference_closure_digest: text(object, "inferenceClosureDigest")?.to_owned(),
        artifact_repository: text(object, "artifactRepository")?.to_owned(),
        artifact_resolved_revision: text(object, "artifactResolvedRevision")?.to_owned(),
        artifact_variant: text(object, "artifactVariant")?.to_owned(),
        resolved_path_fingerprint: text(object, "resolvedPathFingerprint")?.to_owned(),
    })
}

fn rung(value: &str) -> Option<StrategyRung> {
    match value {
        "resident" => Some(StrategyRung::Resident),
        "staged_residency" => Some(StrategyRung::StagedResidency),
        "bounded_decode" => Some(StrategyRung::BoundedDecode),
        "bounded_attention" => Some(StrategyRung::BoundedAttention),
        "bounded_transformer_residency" => Some(StrategyRung::BoundedTransformerResidency),
        _ => None,
    }
}

fn evidence_provider(engine_id: &str) -> &str {
    engine_id.strip_suffix("_control").unwrap_or(engine_id)
}

fn binding_matches_request(
    item: &JsonObject<String, Value>,
    provider: &str,
    tier: &str,
    mode: &str,
    overlay: &str,
    geometry: MemoryGeometry,
) -> bool {
    if text(item, "provider") != Some(provider)
        || text(item, "tier") != Some(tier)
        || text(item, "mode") != Some(mode)
        || text(item, "overlay") != Some(overlay)
    {
        return false;
    }
    let Some(item_geometry) = item.get("geometry").and_then(Value::as_object) else {
        return false;
    };
    ["width", "height", "batch", "frames"]
        .into_iter()
        .zip([
            geometry.width,
            geometry.height,
            geometry.batch,
            geometry.frames,
        ])
        .all(|(key, expected)| {
            item_geometry.get(key).and_then(Value::as_u64) == Some(u64::from(expected))
        })
}

#[allow(clippy::too_many_arguments)]
fn verified_candidates(
    manifest: &JsonObject<String, Value>,
    model_id: &str,
    runtime_provider: &str,
    tier: &str,
    mode: &RequestModeBinding,
    overlay: &str,
    geometry: MemoryGeometry,
) -> WorkerResult<Vec<MemoryEvidence>> {
    let evidence_provider = evidence_provider(runtime_provider);
    let bundle = sceneworks_core::memory_calibration::load_packaged_bundle().map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "packaged memory-calibration evidence is invalid: {error}"
        ))
    })?;
    let Some(bindings) = manifest
        .get("candle")
        .and_then(|value| value.get("calibrations"))
        .and_then(Value::as_array)
    else {
        return Ok(Vec::new());
    };
    let calibration_geometry = CalibrationGeometry {
        width: geometry.width,
        height: geometry.height,
        batch: geometry.batch,
        frames: geometry.frames,
    };
    let mut candidates = Vec::new();
    for value in bindings {
        let Some(item) = value.as_object() else {
            continue;
        };
        if !binding_matches_request(
            item,
            evidence_provider,
            tier,
            &mode.calibration_key,
            overlay,
            geometry,
        ) {
            continue;
        }
        let Some(rung) = text(item, "rung").and_then(rung) else {
            continue;
        };
        let parameters = item
            .get("parameters")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let declared_engaged = match item.get("engagedRungs") {
            None => None,
            Some(Value::Array(values)) => {
                let Some(declared) = values
                    .iter()
                    .map(|value| {
                        value
                            .as_str()
                            .and_then(crate::memory_strategy::rung_from_key)
                    })
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                Some(declared)
            }
            Some(_) => continue,
        };
        let Ok(engaged_rungs) =
            crate::memory_strategy::engaged_composition(rung, declared_engaged.as_deref())
        else {
            continue;
        };
        let Some(selection_parameters) = parse_parameters(&engaged_rungs, &parameters) else {
            continue;
        };
        let Some(calibration) = binding(item) else {
            continue;
        };
        // sc-17774: there is no per-model compatibility hatch any more. `flux2_dev` used to be the
        // one provider that could reach a later runtime pin, via a hand-audited
        // `compatibleInferenceRevision` naming exactly one target revision — spent the moment the
        // pin moved once more, and available to no other model. Currency is now decided for every
        // provider identically, by `calibration.inference_closure_digest` inside `evidence_for`.
        // `inferenceRevision` is capture provenance and is never compared.
        let query = EvidenceQuery {
            backend: CalibrationBackend::Candle,
            model_id: model_id.to_owned(),
            provider: evidence_provider.to_owned(),
            tier: tier.to_owned(),
            mode: mode.calibration_key.clone(),
            overlay: overlay.to_owned(),
            transformer_variant: None,
            decoder: None,
            geometry: calibration_geometry,
            rung,
            parameters,
            calibration: calibration.clone(),
        };
        let EvidenceVerdict::Verified(record) = bundle.evidence_for(&query) else {
            continue;
        };
        let RequiredNullable::Value(predicted) = &record.predicted_peak_bytes else {
            continue;
        };
        if record.quality.result != Some(QualityResult::Passed) {
            continue;
        }
        let selected_strategy = strategy(rung);
        let load_shape = match record.load_shape {
            LoadShapeKey::EagerMaterialization => gen_core::LoadShape::EagerMaterialization,
            LoadShapeKey::DeferredMaterialization => gen_core::LoadShape::DeferredMaterialization,
        };
        let evidence_key = MemoryEvidenceKey {
            model_family: model_id.to_owned(),
            resolved_route: runtime_provider.to_owned(),
            backend: MemoryBackend::Candle,
            tier: numeric_tier(runtime_provider, tier).expect("validated numeric tier"),
            mode: mode.mode.clone(),
            reference_shape: if geometry.reference_count == 0 {
                gen_core::MemoryReferenceShape::None
            } else {
                gen_core::MemoryReferenceShape::Image
            },
            load_shape,
            overlay: (overlay != "none").then(|| overlay.to_owned()),
            geometry,
            frames_per_second: None,
            strategy: selected_strategy,
            engaged_composition: record
                .strategy
                .engaged_rungs
                .iter()
                .copied()
                .map(strategy)
                .collect(),
            parameters: selection_parameters,
        };
        candidates.push(MemoryEvidence {
            key: evidence_key,
            conformance: MemoryConformanceState::Verified,
            dimensions: MemoryEvidenceDimensions::VERIFIED,
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint,
            sceneworks_revision: calibration.scene_works_revision,
            inference_revision: calibration.inference_revision.clone(),
            harness_version: record.harness_version.clone(),
            predicted_peak_bytes: predicted.overall(),
            observed_peak_bytes: match &record.observed_memory {
                RequiredNullable::Value(observed) => Some(observed.overall_non_reclaimable_bytes()),
                RequiredNullable::Null => None,
            },
            parity: if record.quality.maximum_error_threshold == Some(0.0)
                && record.quality.mean_error_threshold == Some(0.0)
            {
                MemoryParityContract::Exact
            } else {
                MemoryParityContract::Tolerance {
                    metric: "maximum RGB8 channel error".to_owned(),
                    maximum_error: record.quality.maximum_error_threshold.unwrap_or_default(),
                }
            },
            parity_result: MemoryParityResult::Passed,
        });
    }
    Ok(candidates)
}

fn account_for_runtime_overlay_bytes(
    candidates: &mut [MemoryEvidence],
    runtime_overlay_bytes: u64,
) {
    for candidate in candidates {
        candidate.predicted_peak_bytes = candidate
            .predicted_peak_bytes
            .saturating_add(runtime_overlay_bytes);
    }
}

/// The smallest declared value for every numeric knob the engaged composition requires — the most
/// deeply bounding parameters the provider publishes, which keeps the true runtime transient as
/// far below the floor's unreduced peak as the provider allows (sc-18097, the candle mirror of the
/// MLX gate's helper from sc-18096). `None` when a required knob has no declared range: such a
/// selection cannot be validated, so no estimate candidate is synthesized for the rung.
fn estimate_floor_parameters(
    contract: &gen_core::MemoryProviderContract,
    engaged: &[MemoryStrategy],
) -> Option<gen_core::MemoryStrategyParameters> {
    sceneworks_core::memory_contracts::smallest_parameters(
        contract,
        engaged,
        sceneworks_core::memory_contracts::TransformerScopeParameters::ProviderOrder,
    )
}

/// Synthesize an estimate-floor candidate for every implemented optimized rung (sc-18097, epic
/// 18093 R1b — the candle mirror of `mlx_fit_gate::synthesize_estimate_ladder`'s floor arm).
///
/// Peak source per rung, from the manifest rows the legacy candle gate already trusts
/// (`crate::vram_gate::predicted_peak_gb` conventions) — never a tuned coefficient and never a
/// promised unmeasured saving:
///
/// * `StagedResidency` — the `candle.sequentialPeakGb` row plus [`crate::vram_gate::HEADROOM_GB`]
///   (`vram_gate::predicted_sequential_peak_gb`, the same padded row the legacy sequential-offload
///   gate compares — a measured working set for legacy adopters). Chroma and the other
///   receipt-priced families instead carry a structural floor: the provider receipt's largest
///   materialized phase plus headroom, maxed against the padded row, required to remain genuinely
///   below resident. Absent the row, other adopters use the resident estimate: staging stays
///   selectable but promises no unmeasured saving.
/// * Rungs 2–4 bound transients/residency the manifest has NOT measured for this cell. Since
///   sc-22664 they are priced per rung through the law (below); before it they took the STAGED
///   floor unreduced. (Where a measured record for a rung exists it is a `verified_candidates`
///   candidate and supersedes the estimate in the selector.) The staged row prices the staged
///   WORKING SET, so it is a sound floor only for a composition that actually engages
///   `StagedResidency` (sc-18253). A provider may
///   implement a deep rung whose engaged composition excludes staging
///   (`gen_core::MemoryProviderContract::engaged_composition`) — such a request runs whole-model
///   resident, so its floor clamps to the RESIDENT estimate instead: the candle mirror of the MLX
///   floor's `engaged.contains(&StagedResidency)` max-vs-sum split
///   (`mlx_fit_gate::estimate_floor_weights_bytes`), which keeps the rung selectable without ever
///   under-predicting a resident working set behind the estimate margin.
///
/// The candle estimate margin is NOT applied here — the selector owns margin widening
/// (`crate::memory_strategy::select_strategy`), exactly as it owns the sc-18095 stale widening.
///
/// sc-22509 (epic 22505) inserted one rung ABOVE the manifest rows: where a measured memory ANCHOR
/// exists for this `(model, tier, candle lane)` cell, the rung floor is derived analytically from it
/// (`sceneworks_core::memory_anchor`) instead of read from `candle.sequentialPeakGb` /
/// `candle.vramGbByTier`. The manifest rows are a single scalar measured at one geometry
/// (`vramMeasuredPixels`); the derived estimate is geometry-aware, so a request at a geometry the
/// campaign never measured is priced from the anchor plus architecture facts rather than being
/// held to a 1024x1024 row.
///
/// sc-22664 (epic 22657 E4) prices EVERY rung through the law, each from its own regime:
///
/// * With an anchor for the cell, each implemented rung builds a [`RequestRegime`] from the
///   parameters [`estimate_floor_parameters`] selects for its engaged composition — the decode
///   tile, the attention chunk (the candle providers publish it as the score-element budget
///   `CONSTRAINED_ATTN_SCORES_BUDGET`, the unit the law takes), the transformer window — and
///   [`MemoryAnchor::derive_phase_peaks`] returns its three phase peaks; the rung's estimate is
///   the max over phases plus the runtime overlay. Basis:
///   [`CandidateBasis::EstimateAnchorDerived`]. No deeper rung reuses the staged floor.
/// * Without an anchor, the CONTRACT-ONLY path: every staged composition takes the law over the
///   manifest staged row, by treating the row as one staged measurement of every phase at the
///   geometry the manifest MEASURED it at — `candle.vramMeasuredPixels`, never the request's
///   ([`floor_pseudo_anchor`], sc-22667) — and decomposing it against the contract's component
///   bytes. The law then scales the row's residue UP to the request geometry before a deeper
///   rung's ratios apply, so a request above the measured geometry prices above the row on
///   every rung, and a rung's tile/chunk fractions are taken of the request-sized residue rather
///   than of a residue the row never measured at that size. At the measured geometry the staged
///   rung is the row itself (every ratio is 1). The row is phase-blind, so it is every phase's
///   peak; a deeper rung's ratios move the phases they bound (decode under the tile, denoise
///   under the chunk and the window) and its reported phase peaks say so, but NO rung bounds
///   conditioning, so at the measured geometry the admission peak — the max over phases — stays
///   at the row. A floor cannot promise a saving only a measured anchor can show, and the ladder
///   invents none. With the default architecture facts (this pin — see
///   `crate::video_admission::architecture_facts_from_contract`) every rung ratio is inert; the
///   geometry scaling needs no fact. A staging-free composition keeps the resident clamp of
///   sc-18253. Basis: [`CandidateBasis::EstimateFloor`] — the row, not a measured anchor, is the
///   basis.
///
/// WHERE THE RESERVE IS PAID (sc-22664, the rule `crate::memory_strategy::ReserveCharge` states
/// once): a manifest-row floor — the staged row, scaled or not — carries
/// [`crate::vram_gate::HEADROOM_GB`] INSIDE its peak exactly as before sc-22664, and the selector
/// compares it against the unreserved pool; an anchor-derived candidate carries NO pad, because
/// the anchor measured every phase as a device delta above the pre-load residency, and pays the
/// operational reserve ([`crate::vram_gate::ladder_reserve_gb`]) on the budget side instead. The
/// receipt-priced families' floors are STRUCTURAL weights-plus-headroom floors sealed from the
/// provider receipt, in which the headroom is the modelled activation term, and are pad-carrying
/// the same way.
///
/// [`RequestRegime`]: sceneworks_core::memory_anchor::RequestRegime
/// [`MemoryAnchor::derive_phase_peaks`]: sceneworks_core::memory_anchor::MemoryAnchor::derive_phase_peaks
/// [`CandidateBasis::EstimateAnchorDerived`]: crate::memory_strategy::CandidateBasis::EstimateAnchorDerived
/// [`CandidateBasis::EstimateFloor`]: crate::memory_strategy::CandidateBasis::EstimateFloor
#[allow(clippy::too_many_arguments)]
fn synthesize_estimate_floors(
    engine_id: &str,
    model_id: &str,
    contract: &gen_core::MemoryProviderContract,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    tier: MemoryNumericTier,
    mode: &RequestModeBinding,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    resident_peak_bytes: u64,
    runtime_overlay_bytes: u64,
    request_evidence_revision: &str,
    anchors: CandleLadderAnchors<'_>,
) -> Vec<EstimateCandidate> {
    use sceneworks_core::memory_anchor::{ImageDeriveRequest, RequestRegime};

    let calibration = contract.calibration.as_ref();
    let receipt_priced = is_receipt_priced(engine_id);
    let staged_floor_bytes = if receipt_priced {
        let structural = receipt_base_phase_floor_bytes(contract)
            .saturating_add((crate::vram_gate::HEADROOM_GB * BYTES_PER_GIB).ceil() as u64);
        let declared = crate::vram_gate::predicted_sequential_peak_gb(manifest, tier_key)
            .map(|gb| (gb * BYTES_PER_GIB).ceil().clamp(0.0, u64::MAX as f64) as u64)
            .unwrap_or(0);
        let declared =
            scale_ideogram_hires_envelope(engine_id, &mode.mode, manifest, geometry, declared);
        contract
            .predicted_peak_from_base(declared.max(structural))
            .predicted_peak_bytes()
    } else {
        // The staged row plus the structural pad, exactly as before sc-22664: the pad is INSIDE a
        // manifest-row floor, so the selector compares such a floor against the unreserved pool
        // (`crate::memory_strategy::ReserveCharge`). Absent the row, the resident estimate.
        crate::vram_gate::predicted_sequential_peak_gb(manifest, tier_key)
            .map(|gb| {
                ((gb * BYTES_PER_GIB).ceil().clamp(0.0, u64::MAX as f64) as u64)
                    .saturating_add(runtime_overlay_bytes)
            })
            .unwrap_or(resident_peak_bytes)
    };
    let headroom_bytes = (crate::vram_gate::HEADROOM_GB * BYTES_PER_GIB).ceil() as u64;
    // The RAW staged row — what the manifest measured, without the pad or the request's overlay —
    // is what the law decomposes for the contract-only deeper rungs; both charges are folded back
    // over the derivation.
    let raw_staged_row_bytes = crate::vram_gate::measured_sequential_peak_gb(manifest, tier_key)
        .map(|gb| (gb * BYTES_PER_GIB).ceil().clamp(0.0, u64::MAX as f64) as u64);
    // Chroma claims a staged saving only when the receipt-derived largest phase plus exact
    // auxiliaries is genuinely below its receipt-derived co-resident floor. Equal or crossed rows
    // are not an optimized strategy and cannot become selectable through a tier label.
    if receipt_priced && staged_floor_bytes >= resident_peak_bytes {
        return Vec::new();
    }
    // NOTE (sc-22509, reopened by sc-22666): this lookup now MATCHES packaged anchors. Every
    // retained corpus is compiled in since epic 22657 E5, so `z_image_turbo:candle` (q4/q8/bf16,
    // sc-15859) and the qwen candle rows answer here through the shared-image ladder. The
    // per-model allow-list that used to keep them out is gone; identity and loader-closure
    // currency are the only guards left, and a cell with no anchor row falls through to
    // `floor_anchor` below -- the contract-only per-rung ladder (sc-22664), never a bare
    // manifest scalar repeated across every rung.
    // The geometry the LAW prices — the request's own, except where the manifest's declared
    // activation term already charges the batch and the area (see `law_geometry`).
    let law = law_geometry(manifest, geometry);
    let anchor = candle_image_anchor(
        anchors, engine_id, model_id, contract, tier_key, mode, overlay, law,
    );
    let components = crate::video_admission::anchor_component_bytes(contract.asset_facts);
    // The contract-only basis for a staged composition with no anchor: the staged row treated as
    // one staged measurement of every phase at the request geometry (see the doc comment). Only
    // for the shared ladder — a receipt-priced floor is a structural sum, not a measured working
    // set, and stays unscaled.
    let floor_anchor = if anchor.is_none() && !receipt_priced {
        raw_staged_row_bytes.map(|row| floor_pseudo_anchor(engine_id, manifest, row))
    } else {
        None
    };
    let mut synthesized = Vec::new();
    // Scratch-bounding rungs remain independently usable with resident weights. SDXL additionally
    // offers their explicit staged composition, so the selector can reach the measured fully
    // optimized endpoint without changing the provider's cost-order defaults or bespoke support.
    for (strategy, explicit_staging) in MemoryStrategy::ALL.into_iter().flat_map(|strategy| {
        std::iter::once((strategy, false)).chain(
            (engine_id == "sdxl"
                && model_id == "sdxl"
                && mode.mode == MemoryMode::TextToImage
                && geometry.reference_count == 0
                && geometry.batch == 1
                && matches!(overlay, None | Some("lora"))
                && matches!(
                    strategy,
                    MemoryStrategy::BoundedDecode | MemoryStrategy::BoundedAttention
                ))
            .then_some((strategy, true)),
        )
    }) {
        if strategy == MemoryStrategy::Resident {
            // The resident live estimate is already submitted on every request.
            continue;
        }
        if !matches!(
            contract.capability(strategy).map(|cap| &cap.support),
            Some(gen_core::MemoryStrategySupport::Implemented)
        ) {
            continue;
        }
        let engaged = contract.engaged_composition(strategy);
        let Some(mut parameters) = estimate_floor_parameters(contract, &engaged) else {
            continue;
        };
        parameters.stage_residency = explicit_staging.then_some(true);
        let selection = MemorySelection {
            strategy,
            parameters,
            tier,
        };
        if contract.validate_selection(&selection).is_err() {
            continue;
        }
        let engaged = contract.engaged_composition_for_selection(&selection);
        let staged = engaged.contains(&MemoryStrategy::StagedResidency);
        // The rung's own regime: its engaged composition and the parameters it was selected with.
        // Every rung is priced from THIS, never from the staged rung's number (sc-22664).
        let regime = request_regime(&engaged, &parameters);
        let request = |regime: RequestRegime| ImageDeriveRequest {
            width: law.width,
            height: law.height,
            batch: law.batch.max(1),
            conditioning_tokens: None,
            regime,
        };
        // The retained candle phase peaks are device-usage DELTAS above the process's pre-load
        // residency; that residency is the reserve the selector budget charges once
        // (`vram_gate::ladder_reserve_gb`), so nothing is added here beyond the request's own
        // runtime overlay.
        let derived = anchor.zip(regime).and_then(|(anchor, regime)| {
            anchor
                .derive_phase_peaks(&request(regime), components, anchors.facts)
                .map(|phases| (phases, anchor.id.as_str()))
        });
        let (predicted_peak_bytes, basis, phase_peaks) = match derived {
            Some((phases, anchor_id)) => {
                let bytes = phases.peak_bytes().saturating_add(runtime_overlay_bytes);
                tracing::info!(
                    route = engine_id,
                    backend = "candle",
                    ?strategy,
                    anchor = anchor_id,
                    conditioning_peak_bytes = phases.conditioning,
                    denoise_peak_bytes = phases.denoise,
                    decode_peak_bytes = phases.decode,
                    raw_peak_bytes = bytes,
                    "synthesized anchor-derived estimate candidate"
                );
                (
                    bytes,
                    crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                        lane: crate::memory_strategy::AnchorDerivationLane::Image,
                    },
                    Some(phases),
                )
            }
            None => {
                // sc-18253: the staged row is only a sound floor for a composition that engages
                // staging; a deep rung excluding `StagedResidency` runs whole-model resident and
                // clamps to the resident estimate (see the doc comment above). A staged
                // composition takes the law's ratios over the row (`floor_pseudo_anchor`); with
                // no ratio to apply — the staged rung itself, or a regime the row cannot price —
                // the row stands unscaled.
                let from_floor = if staged {
                    floor_anchor
                        .as_ref()
                        .zip(regime)
                        .and_then(|(floor_anchor, regime)| {
                            floor_anchor.derive_phase_peaks(
                                &request(regime),
                                components,
                                anchors.facts,
                            )
                        })
                } else {
                    None
                };
                // A law-scaled row carries the same structural pad and overlay the unscaled
                // staged floor does: it is still a manifest-row floor.
                let (bytes, phase_peaks) = match from_floor {
                    Some(phases) => (
                        phases
                            .peak_bytes()
                            .saturating_add(headroom_bytes)
                            .saturating_add(runtime_overlay_bytes),
                        Some(phases),
                    ),
                    None if staged => (staged_floor_bytes, None),
                    None => (resident_peak_bytes, None),
                };
                tracing::info!(
                    route = engine_id,
                    backend = "candle",
                    ?strategy,
                    raw_peak_bytes = bytes,
                    law_scaled = phase_peaks.is_some(),
                    "synthesized manifest-row floor estimate candidate"
                );
                (
                    bytes,
                    crate::memory_strategy::CandidateBasis::EstimateFloor,
                    phase_peaks,
                )
            }
        };
        synthesized.push(EstimateCandidate {
            selection,
            phase_peaks,
            basis,
            evidence: sceneworks_core::memory_contracts::unverified_estimate(
                MemoryEvidenceKey {
                    model_family: engine_id.to_owned(),
                    resolved_route: engine_id.to_owned(),
                    backend: MemoryBackend::Candle,
                    tier,
                    mode: mode.mode.clone(),
                    reference_shape: if geometry.reference_count == 0 {
                        gen_core::MemoryReferenceShape::None
                    } else {
                        gen_core::MemoryReferenceShape::Image
                    },
                    load_shape: contract.load_shape,
                    overlay: overlay.map(str::to_owned),
                    geometry,
                    frames_per_second: None,
                    strategy,
                    engaged_composition: engaged,
                    parameters,
                },
                calibration.map_or(gen_core::MEMORY_CALIBRATION_ABI, |item| item.abi),
                calibration.map_or_else(String::new, |item| item.fingerprint.clone()),
                request_evidence_revision.to_owned(),
                predicted_peak_bytes,
            ),
        });
    }
    synthesized
}

/// One synthesized estimate candidate of the candle ladder (sc-22664): the rung, its evidence
/// row, the basis that priced it, and — where the law priced it — the three derived phase peaks
/// the admission telemetry reports (E7).
#[derive(Clone, Debug)]
pub(crate) struct EstimateCandidate {
    pub selection: MemorySelection,
    pub evidence: MemoryEvidence,
    pub basis: crate::memory_strategy::CandidateBasis,
    /// The derived conditioning / denoise / decode peaks BEFORE the runtime overlay charge, when
    /// the candidate was priced through `MemoryAnchor::derive_phase_peaks` (from a measured anchor
    /// or from the manifest row as a pseudo-anchor). `None` for an unscaled row and for the
    /// resident clamp.
    pub phase_peaks: Option<sceneworks_core::memory_anchor::AnchorDerivedPhases>,
}

/// Price a validated endpoint at its literal coordinate, or derive the same executable
/// composition at another geometry. The capture measures memory, not numerical parity: it
/// supplies an estimate-scoped candidate with the normal uncertainty and a separate reserve.
fn sdxl_endpoint_candidate(
    contract: &gen_core::MemoryProviderContract,
    resident: &MemoryEvidence,
    anchor: &sceneworks_core::memory_anchor::MemoryAnchor,
    selection: MemorySelection,
    load_peak_bytes: u64,
) -> Option<EstimateCandidate> {
    use sceneworks_core::memory_anchor::{
        AnchorBackend, AnchorDerivedPhases, AnchorLoadShape, ImageDeriveRequest,
    };

    let geometry = resident.key.geometry;
    let load_shape = match anchor.load_shape {
        AnchorLoadShape::EagerMaterialization => gen_core::LoadShape::EagerMaterialization,
        AnchorLoadShape::DeferredMaterialization => gen_core::LoadShape::DeferredMaterialization,
    };
    if anchor.model_id != "sdxl"
        || anchor.provider != contract.provider_id
        || anchor.route != resident.key.resolved_route
        || anchor.backend != AnchorBackend::Candle
        || anchor.tier != "q4"
        || anchor.mode != resident.key.mode.as_key()
        || anchor.overlay.is_some()
        || resident.key.overlay.is_some()
        || anchor.reference_count != 0
        || geometry.reference_count != 0
        || geometry.batch != 1
        || geometry.frames != 1
        || load_shape != contract.load_shape
        || !contract
            .calibration
            .as_ref()
            .is_some_and(|identity| identity.fingerprint == anchor.source.calibration_fingerprint)
        || contract.validate_selection(&selection).is_err()
    {
        return None;
    }
    let engaged = contract.engaged_composition_for_selection(&selection);
    let regime = request_regime(&engaged, &selection.parameters)?;
    // A bounded endpoint can only price its own exact bounded composition. The pair loader binds
    // the actual numeric knobs; this check also prevents borrowing it for an unbounded request.
    if anchor.measured_regime.staged != regime.staged
        || anchor.measured_regime.decode_tiled != regime.decode_tile.is_some()
        || anchor.measured_regime.attention_chunked != regime.attention_chunk_scores.is_some()
        || anchor.measured_regime.transformer_windowed != regime.transformer_window.is_some()
    {
        return None;
    }
    let exact_geometry =
        geometry.width == anchor.geometry.width && geometry.height == anchor.geometry.height;
    let phases = if exact_geometry {
        AnchorDerivedPhases {
            conditioning: anchor.phase_active_peak_bytes.conditioning,
            denoise: anchor.phase_active_peak_bytes.denoise,
            decode: anchor.phase_active_peak_bytes.decode,
        }
    } else {
        anchor.derive_phase_peaks(
            &ImageDeriveRequest {
                width: geometry.width,
                height: geometry.height,
                batch: 1,
                conditioning_tokens: None,
                regime,
            },
            crate::video_admission::anchor_component_bytes(contract.asset_facts),
            crate::video_admission::architecture_facts_from_contract(contract),
        )?
    };
    let peak = if exact_geometry {
        anchor.overall_allocator_envelope_bytes
    } else {
        phases.peak_bytes().max(load_peak_bytes)
    };
    let mut evidence = resident.clone();
    evidence.key.strategy = selection.strategy;
    evidence.key.parameters = selection.parameters;
    evidence.key.engaged_composition = engaged;
    evidence.predicted_peak_bytes = peak;
    evidence.observed_peak_bytes = exact_geometry.then_some(peak);
    Some(EstimateCandidate {
        selection,
        evidence,
        basis: if exact_geometry {
            crate::memory_strategy::CandidateBasis::EstimateMeasuredEndpoint
        } else {
            crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                lane: crate::memory_strategy::AnchorDerivationLane::Image,
            }
        },
        phase_peaks: Some(phases),
    })
}

/// The anchor store and architecture facts the candle ladder prices from (sc-22664).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CandleLadderAnchors<'a> {
    pub store: Option<&'a sceneworks_core::memory_anchor::MemoryAnchorStore>,
    /// The facts the law scales its residues by. From the contract in production
    /// ([`architecture_facts_from_contract`], the one worker-edge seam shared with the MLX and
    /// video lanes, which translates the contract's own `architecture_facts` block axis by axis
    /// since sc-22667); a fixture may state the model's facts directly.
    ///
    /// [`architecture_facts_from_contract`]: crate::video_admission::architecture_facts_from_contract
    pub facts: sceneworks_core::memory_anchor::ArchitectureFacts,
}

impl CandleLadderAnchors<'static> {
    /// The production source: the packaged store -- catalog-wide and unscoped since sc-22666 --
    /// and the architecture facts the contract states.
    pub(crate) fn packaged(contract: &gen_core::MemoryProviderContract) -> Self {
        Self {
            store: sceneworks_core::memory_anchor::packaged_memory_anchors(),
            facts: crate::video_admission::architecture_facts_from_contract(contract),
        }
    }
}

/// The [`RequestRegime`] one ladder rung executes in: its engaged composition plus the parameters
/// [`estimate_floor_parameters`] selected for it. `None` when an engaged rung's parameter is
/// absent (the selection would already have failed `validate_selection`).
///
/// The attention chunk is translated as SCORE ELEMENTS: every candle provider publishes
/// `attention_chunk_sizes` as `CONSTRAINED_ATTN_SCORES_BUDGET` (or a per-model score budget of the
/// same unit), which is the quantity `RequestRegime::attention_chunk_scores` prices.
///
/// [`RequestRegime`]: sceneworks_core::memory_anchor::RequestRegime
pub(crate) fn request_regime(
    engaged: &[MemoryStrategy],
    parameters: &gen_core::MemoryStrategyParameters,
) -> Option<sceneworks_core::memory_anchor::RequestRegime> {
    // Candle has historically refused incomplete engaged parameters; retain that boundary.
    for (rung, complete) in [
        (
            MemoryStrategy::BoundedDecode,
            parameters.decode_tile_edge.is_some() && parameters.decode_overlap.is_some(),
        ),
        (
            MemoryStrategy::BoundedAttention,
            parameters.attention_chunk_size.is_some(),
        ),
        (
            MemoryStrategy::BoundedTransformerResidency,
            parameters.transformer_window_size.is_some(),
        ),
    ] {
        if engaged.contains(&rung) && !complete {
            return None;
        }
    }
    Some(sceneworks_core::memory_contracts::request_regime(
        engaged, parameters,
    ))
}

/// The manifest staged row as a pseudo-anchor (sc-22664, the contract-only path): one STAGED
/// measurement of every phase at the geometry the manifest MEASURED the row at, which is exactly
/// the claim `candle.sequentialPeakGb` makes — the largest single working set of the staged path
/// at `candle.vramMeasuredPixels` — stated phase-blind. Running it through
/// `MemoryAnchor::derive_phase_peaks` at that geometry returns the row itself for the staged rung
/// (every ratio is 1) and the law's ratios over the row for a deeper rung; at a larger request
/// geometry the law scales the row's residue up first, then applies the rung's ratios to the
/// request-sized residue. A row below the component set of some phase is outside the law's
/// domain and the derivation refuses it, in which case the caller keeps the row unscaled.
///
/// WHY THE MEASURED GEOMETRY, NOT THE REQUEST'S (sc-22667, epic 22657 feature-end round): the
/// row is geometry-blind, so stamping it as a measurement AT the request geometry made a 2048²
/// bounded rung take its tile/chunk fractions of a residue that was measured at 1024² — scaling
/// the row's phases DOWN, with no measurement behind the saving. Anchored at the measured
/// geometry the residue is scaled UP to 2048² before any fraction is taken, which is the erring-
/// large reading the epic requires. The measured pixel count is read as the largest square that
/// fits it (`isqrt`), so the anchor side of every geometry ratio is at or below the true measured
/// count and the scale-up is never understated.
///
/// It is a floor, not evidence: its identity fields are the request's, it cites no source, and the
/// candidate it prices carries `CandidateBasis::EstimateFloor`.
fn floor_pseudo_anchor(
    engine_id: &str,
    manifest: &JsonObject<String, Value>,
    staged_floor_bytes: u64,
) -> sceneworks_core::memory_anchor::MemoryAnchor {
    use sceneworks_core::memory_anchor::{
        AnchorBackend, AnchorGeometry, AnchorLoadShape, AnchorMeasuredRegime, AnchorPhaseBytes,
        AnchorSource, MemoryAnchor,
    };
    let measured_edge = u32::try_from(manifest_measured_pixels(manifest).isqrt())
        .unwrap_or(u32::MAX)
        .max(1);
    MemoryAnchor {
        id: format!("floor:{engine_id}:candle:sequentialPeakGb"),
        model_id: engine_id.to_owned(),
        model_family: String::new(),
        route: engine_id.to_owned(),
        provider: engine_id.to_owned(),
        backend: AnchorBackend::Candle,
        tier: String::new(),
        transformer_variant: None,
        decoder: None,
        mode: String::new(),
        overlay: None,
        reference_count: 0,
        load_shape: AnchorLoadShape::EagerMaterialization,
        measured_regime: AnchorMeasuredRegime {
            decode_tiled: false,
            transformer_windowed: false,
            staged: true,
            attention_chunked: false,
        },
        source: AnchorSource {
            path: String::new(),
            sha256: String::new(),
            record_id: String::new(),
            calibration_fingerprint: String::new(),
            loader_closure_digest: String::new(),
            currency_attestation: None,
        },
        geometry: AnchorGeometry {
            width: measured_edge,
            height: measured_edge,
            frames: 1,
            fps: None,
        },
        phase_active_peak_bytes: AnchorPhaseBytes {
            conditioning: staged_floor_bytes,
            denoise: staged_floor_bytes,
            decode: staged_floor_bytes,
        },
        phase_allocator_envelope_bytes: None,
        overall_allocator_envelope_bytes: staged_floor_bytes,
        underived_reason: None,
        component_bytes: None,
        // sc-22734: an ordinary staged candle cell, not one whose engine declares staged
        // residency structurally impossible, so it keeps the staged-only derivation law.
        staged_residency_structurally_not_applicable: false,
    }
}

/// The measured memory anchor for this candle image request, or `None` to keep the manifest-row
/// floor (sc-22509, epic 22505).
///
/// Identity is strict and fail-closed, the candle mirror of `video_admission::anchor_currency_
/// matches` and `vram_gate::krea_store_anchor`: the anchor was measured on ONE catalog model,
/// route, provider, tier, mode and materialization shape, overlay-free and reference-free (checked
/// on BOTH the request and the anchor row), and its loader closure must still be current
/// (sc-22511). Anything else keeps the established floor. The
/// measured GEOMETRY is deliberately NOT part of the guard — pricing a never-measured geometry from
/// one anchor is the entire point of the derivation — and neither is `model_family`, which has no
/// source in the record (see [`sceneworks_core::memory_anchor::MemoryAnchor::model_family`]).
#[allow(clippy::too_many_arguments)]
fn candle_image_anchor<'a>(
    anchors: CandleLadderAnchors<'a>,
    engine_id: &str,
    model_id: &str,
    contract: &gen_core::MemoryProviderContract,
    tier_key: &str,
    mode: &RequestModeBinding,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
) -> Option<&'a sceneworks_core::memory_anchor::MemoryAnchor> {
    // The anchors were measured overlay-free, reference-free and single-batch on a still image.
    if overlay.is_some() || geometry.reference_count != 0 || geometry.batch != 1 {
        return None;
    }
    // NO MODEL SCOPE (sc-22666, epic 22657 E5). A `CANDLE_ANCHOR_COEFFICIENT_MODELS` allow-list
    // (`["krea_2_turbo"]`) used to stand here, because the sc-22509 candle law was three per-pixel
    // slopes fitted on Krea Turbo and pricing another model's row through them would have been
    // borrowing empirics. The core law fits nothing since sc-22663 and this lane prices every rung
    // through it since sc-22664 -- it decomposes THIS anchor's own measured peaks against THIS
    // contract's component bytes and rescales the residues by architecture facts -- so a store
    // anchor for any `(model, tier)` cell prices its own cell and nobody else's. Nor is it a
    // packaging question any more: every retained corpus is compiled in, so a packaged row is
    // priced the day it lands, which is the point of packaging them.
    //
    // The identity conjuncts below (route, provider, mode, overlay, references, materialization
    // shape, loader-closure currency) are the whole guard now: per-row facts, not a model census.
    let anchor = anchors.store?.image_anchor_for(
        model_id,
        sceneworks_core::memory_anchor::AnchorBackend::Candle,
        tier_key,
    )?;
    // An anchor the extractor marked underived validates its own measured point but is not
    // published as a derivation basis (the memory matrix prints the cell `Anchored/underived`);
    // the worker agrees with the matrix and does not price from it. Lifting that per model is
    // sc-22666's packaging call, not this lane's.
    if anchor.underived_reason.is_some() {
        return None;
    }
    // `model_family` is deliberately NOT a conjunct (sc-22509 review): the source calibration
    // record carries no family field, so the store's copy is catalog-derived and unvalidated, and
    // keying on it would make an extractor's catalog read load-bearing for admission. `model_id` is
    // bound to the record and already selects the cell.
    //
    // The ANCHOR side of the overlay/reference guard, mirroring `vram_gate::krea_store_anchor` and
    // `video_admission::anchor_currency_matches`: the request being overlay-free is only half the
    // question — an overlay-measured or reference-bearing anchor row must not be borrowed for an
    // overlay-free request either.
    if anchor.route != engine_id
        || anchor.provider != contract.provider_id
        || anchor.mode != mode.mode.as_key()
        || anchor.overlay.is_some()
        || anchor.reference_count != 0
    {
        return None;
    }
    // Materialization shape decides whether the text encoder is still resident, so an anchor
    // measured under the other shape does not price this request.
    let anchor_load_shape = match anchor.load_shape {
        sceneworks_core::memory_anchor::AnchorLoadShape::EagerMaterialization => {
            gen_core::LoadShape::EagerMaterialization
        }
        sceneworks_core::memory_anchor::AnchorLoadShape::DeferredMaterialization => {
            gen_core::LoadShape::DeferredMaterialization
        }
    };
    if anchor_load_shape != contract.load_shape {
        return None;
    }
    // No currency conjunct (sc-22738): the model's loader-closure digest (sc-22511) is what the
    // probe tooling re-captures on, and the calibration ABI and fingerprint are provenance bound
    // to the source record by `validate_anchor`. None of them demotes the anchor here — it prices
    // this request whether or not the loader has moved since it was measured.
    Some(anchor)
}

fn memory_for_selection(
    contract: &gen_core::MemoryProviderContract,
    selection: MemorySelection,
) -> Option<GenerationMemory> {
    if selection.strategy == MemoryStrategy::Resident {
        None
    } else {
        contract.generation_memory(&selection)
    }
}

/// [`evaluate_shared_image`] with the provider's prefix-KV-cache surcharge priced in (epic 25899):
/// `prefix_kv_cache_surcharge` is `execution_planner::prefix_kv_cache_surcharge_bytes` for the
/// request, and it is added rung-for-rung (`execution_planner::plan_prefix_kv_cache`): the shed
/// ladder picks the rung exactly as before, and the cache is taken only when that rung also fits it.
/// `None` is exactly [`evaluate_shared_image`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_shared_image_priced(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    prefix_kv_cache_surcharge: Option<u64>,
    sdxl_endpoints: Option<&SdxlCudaMemoryEndpoints>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    evaluate_shared_image_priced_with_resources(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        geometry,
        has_reference,
        use_pid,
        has_phases,
        request_has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        prefix_kv_cache_surcharge,
        None,
        sdxl_endpoints,
    )
}

/// Reference-aware admission; facts originate only in the owned-asset resolver.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_shared_image_priced_with_resources(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    prefix_kv_cache_surcharge: Option<u64>,
    reference_resources: Option<&gen_core::MemoryReferenceResourceFacts>,
    sdxl_endpoints: Option<&SdxlCudaMemoryEndpoints>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    // Resolve the backend-owned bound before contract derivation or any fit decision. A provider
    // with references but no pricer is a typed refusal, never a zero-priced legacy fallthrough.
    let reference_workspace_bytes =
        crate::memory_strategy::reference_workspace_bytes(engine_id, spec, reference_resources)?;
    evaluate_shared_image_dispatch(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        geometry,
        has_reference,
        use_pid,
        has_phases,
        request_has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        prefix_kv_cache_surcharge,
        reference_resources,
        reference_workspace_bytes,
        sdxl_endpoints,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_shared_image(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    evaluate_shared_image_dispatch(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        geometry,
        has_reference,
        use_pid,
        has_phases,
        request_has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        None,
        None,
        0,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_shared_image_dispatch(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    prefix_kv_cache_surcharge: Option<u64>,
    reference_resources: Option<&gen_core::MemoryReferenceResourceFacts>,
    reference_workspace_bytes: u64,
    sdxl_endpoints: Option<&SdxlCudaMemoryEndpoints>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    let (contract_override, provider_overlay_override, provider_mode_override) =
        if spec.load_shape_declaration_result == gen_core::LoadShapeDeclarationResult::Eligible {
            let Some(mode) =
                crate::memory_route_registry::MemoryRouteMode::from_request(request_mode_value)
            else {
                return Ok(None);
            };
            let Some(contract) = crate::memory_route_registry::declared_candle_selector_contract(
                engine_id,
                Some(tier_key),
                Some(mode),
                manifest,
                spec,
                crate::memory_route_registry::MemoryRouteRequestContext {
                    mode,
                    reference_count: geometry.reference_count,
                    use_pid,
                    has_phases: request_has_phases,
                },
            ) else {
                return Ok(None);
            };
            (Some(contract), None, None)
        } else {
            let Some(mode) =
                crate::memory_route_registry::MemoryRouteMode::from_request(request_mode_value)
            else {
                return Ok(None);
            };
            match crate::memory_route_registry::declared_candle_request_strategy_contract(
            engine_id,
            Some(tier_key),
            manifest,
            spec,
            crate::memory_route_registry::MemoryRouteRequestContext {
                mode,
                reference_count: geometry.reference_count,
                use_pid,
                has_phases: request_has_phases,
            },
        ) {
            crate::memory_route_registry::DeclaredCandleStrategyContract::NoRelevantDeclaration => {
                (None, None, None)
            }
            crate::memory_route_registry::DeclaredCandleStrategyContract::Applied {
                contract,
                provider_overlay,
                provider_mode,
            } => (Some(*contract), Some(provider_overlay), Some(provider_mode)),
            crate::memory_route_registry::DeclaredCandleStrategyContract::Refused => {
                return Ok(None)
            }
        }
        };
    let provider_overlay = provider_overlay_override
        .as_ref()
        .map(|overlay| overlay.as_deref())
        .unwrap_or(overlay);
    let exact_sdxl_overlay = (engine_id == "sdxl")
        .then(|| sdxl_provider_overlay(spec))
        .flatten();
    let provider_overlay = exact_sdxl_overlay.as_deref().or(provider_overlay);
    evaluate_shared_image_inner_priced_with_resources(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        provider_overlay,
        geometry,
        has_reference,
        use_pid,
        has_phases,
        request_has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        provider_mode_override.as_deref(),
        contract_override,
        None,
        None,
        prefix_kv_cache_surcharge,
        reference_resources,
        reference_workspace_bytes,
        sdxl_endpoints,
    )
}

/// Exact ordered physical overlay identity shared by registered and bespoke SDXL admissions.
///
/// Every sibling receipt-priced family binds its overlay to the exact materialized artifact —
/// Kolors through `KOLORS_CONTROL_RECEIPT_PREFIX`/`KOLORS_IP_RECEIPT_PREFIX`, SD3.5 and Chroma
/// through their ordered additive-adapter receipts. SDXL cannot read those from its contract:
/// `candle_gen_sdxl::memory_strategy::build_contract` emits no `resident_components`, and the
/// physical digest `SdxlArtifactSeal::capture` computes is discarded before the contract is
/// returned. So the worker seals its own ordered twin here, over the ROLE and the resolved SOURCE
/// PATH of every overlay slot, at the same grade `gen_core::adapter_stack_identity` already
/// applies to the adapter slot (path + kind + scale, sha256, domain-separated).
///
/// The identity this replaces was `control:{N}` / `ip-adapter` / `pid` — cardinality and kind
/// only. Two different tile ControlNets, or two different IP-Adapter checkpoints, produced one
/// identity and could therefore borrow each other's admitted peak. Ordering is the LoadSpec's own
/// slot order (control, then `extra_controls` in order, then IP-Adapter, then the PiD checkpoint
/// and its Gemma text encoder), never a sorted set, because the load order is what the provider
/// materializes.
pub(crate) fn sdxl_provider_overlay(spec: &LoadSpec) -> Option<String> {
    let mut slots: Vec<(&str, &gen_core::WeightsSource)> = Vec::new();
    if let Some(control) = spec.control.as_ref() {
        slots.push(("control", control));
    }
    for extra in &spec.extra_controls {
        slots.push(("extra-control", extra));
    }
    if let Some(ip_adapter) = spec.ip_adapter.as_ref() {
        slots.push(("ip-adapter", ip_adapter));
    }
    if let Some(pid) = spec.pid.as_ref() {
        slots.push(("pid-checkpoint", &pid.checkpoint));
        slots.push(("pid-gemma", &pid.gemma));
    }
    let adapters = gen_core::adapter_stack_identity(&spec.adapters);
    if slots.is_empty() && adapters.is_none() {
        return None;
    }
    let mut digest = sha2::Sha256::new();
    digest.update(SDXL_OVERLAY_RECEIPT_DOMAIN.as_bytes());
    for (ordinal, (role, source)) in slots.iter().enumerate() {
        // The source KIND is part of the seal: a directory snapshot and a single-file checkpoint
        // are different physical loads even when one path is a prefix of the other.
        let (kind, path) = match source {
            gen_core::WeightsSource::Dir(path) => ("dir", path),
            gen_core::WeightsSource::File(path) => ("file", path),
        };
        let path = format!("{:?}", path.as_os_str());
        digest.update((ordinal as u64).to_le_bytes());
        digest.update((role.len() as u64).to_le_bytes());
        digest.update(role.as_bytes());
        digest.update((kind.len() as u64).to_le_bytes());
        digest.update(kind.as_bytes());
        digest.update((path.len() as u64).to_le_bytes());
        digest.update(path.as_bytes());
    }
    if let Some(adapters) = adapters.as_deref() {
        digest.update((adapters.len() as u64).to_le_bytes());
        digest.update(adapters.as_bytes());
    }
    Some(format!(
        "{SDXL_OVERLAY_RECEIPT_PREFIX}{:x}",
        digest.finalize()
    ))
}

/// Shared-selector entry point for a bespoke Candle provider whose exact contract is built from
/// caller-provisioned paths rather than the registered provider catalog.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_shared_bespoke_image(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    contract: gen_core::MemoryProviderContract,
    request_evidence_revision: &'static str,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    let expected_revision = match engine_id {
        "pulid_flux" => crate::memory_route_registry::PULID_FLUX_REQUEST_EVIDENCE_REVISION,
        "candle_kolors_ipadapter" | "candle_kolors_control" => {
            crate::memory_route_registry::KOLORS_REQUEST_EVIDENCE_REVISION
        }
        "sdxl" => crate::memory_route_registry::SDXL_REQUEST_EVIDENCE_REVISION,
        _ => {
            return Err(WorkerError::InvalidPayload(format!(
                "{engine_id} has no registered bespoke memory evidence authority"
            )))
        }
    };
    if request_evidence_revision != expected_revision {
        return Err(WorkerError::InvalidPayload(format!(
            "{engine_id} crossed bespoke memory evidence revision {request_evidence_revision}"
        )));
    }
    if is_sealed_kolors_bespoke(engine_id) {
        let Some(mode) =
            crate::memory_route_registry::MemoryRouteMode::from_request(request_mode_value)
        else {
            return Ok(None);
        };
        if !crate::memory_route_registry::declared_candle_bespoke_request(
            engine_id,
            Some(tier_key),
            request_mode_value,
            spec,
            crate::memory_route_registry::MemoryRouteRequestContext {
                mode,
                reference_count: geometry.reference_count,
                use_pid,
                has_phases,
            },
        ) {
            return Ok(None);
        }
    }
    let exact_provider_overlay = if engine_id == "sdxl" {
        sdxl_provider_overlay(spec)
    } else {
        overlay.map(str::to_owned)
    };
    evaluate_shared_image_inner(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        exact_provider_overlay.as_deref(),
        geometry,
        has_reference,
        use_pid,
        has_phases,
        has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        None,
        Some(contract),
        Some(request_evidence_revision),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_shared_image_inner(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    provider_overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    worker_multipass: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    provider_mode_override: Option<&str>,
    contract_override: Option<gen_core::MemoryProviderContract>,
    request_evidence_revision_override: Option<&'static str>,
    ladder_anchors_override: Option<CandleLadderAnchors<'_>>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    evaluate_shared_image_inner_priced(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        provider_overlay,
        geometry,
        has_reference,
        use_pid,
        worker_multipass,
        request_has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        provider_mode_override,
        contract_override,
        request_evidence_revision_override,
        ladder_anchors_override,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_shared_image_inner_priced(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    provider_overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    worker_multipass: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    provider_mode_override: Option<&str>,
    contract_override: Option<gen_core::MemoryProviderContract>,
    request_evidence_revision_override: Option<&'static str>,
    ladder_anchors_override: Option<CandleLadderAnchors<'_>>,
    prefix_kv_cache_surcharge: Option<u64>,
    sdxl_endpoints: Option<&SdxlCudaMemoryEndpoints>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    evaluate_shared_image_inner_priced_with_resources(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        provider_overlay,
        geometry,
        has_reference,
        use_pid,
        worker_multipass,
        request_has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        provider_mode_override,
        contract_override,
        request_evidence_revision_override,
        ladder_anchors_override,
        prefix_kv_cache_surcharge,
        None,
        0,
        sdxl_endpoints,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_shared_image_inner_priced_with_resources(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    provider_overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    worker_multipass: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    provider_mode_override: Option<&str>,
    contract_override: Option<gen_core::MemoryProviderContract>,
    request_evidence_revision_override: Option<&'static str>,
    ladder_anchors_override: Option<CandleLadderAnchors<'_>>,
    prefix_kv_cache_surcharge: Option<u64>,
    reference_resources: Option<&gen_core::MemoryReferenceResourceFacts>,
    reference_workspace_bytes: u64,
    sdxl_endpoints: Option<&SdxlCudaMemoryEndpoints>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    let Some(registered_revision) =
        crate::memory_route_registry::candle_gate_evidence_revision(engine_id)
    else {
        // Unregistered families retain resident/legacy admission and cannot acquire an optimized
        // request contract through the former declaration-revision catch-all.
        return Ok(None);
    };
    let request_evidence_revision =
        request_evidence_revision_override.unwrap_or(registered_revision);
    // Hires-fix is two independently shaped denoise passes. Only families whose generation harness
    // mints and carries one exact context per pass may use request-scoped optimized admission.
    if worker_multipass && !is_ideogram(engine_id) && !is_sana(engine_id) && engine_id != "sdxl" {
        return Ok(None);
    }
    let mode =
        request_mode_with_provider_override(engine_id, request_mode_value, provider_mode_override);
    let Some(tier) = numeric_tier(engine_id, tier_key) else {
        return Ok(None);
    };
    let Some(budget) = budget else {
        return Ok(None);
    };
    let receipt_priced = is_receipt_priced(engine_id);
    let resident_peak_gb = match predicted_peak_gb {
        Some(peak) => peak,
        // SANA's certified provider receipt prices the selected dense snapshot from its real
        // safetensor headers. It deliberately has no per-story measured manifest curve.
        None if (is_sana(engine_id) || is_sd35(engine_id)) && artifact_is_certified => 0.0,
        None => {
            if receipt_priced && artifact_is_certified {
                return Err(WorkerError::InvalidPayload(format!(
                    "{engine_id}/{tier_key} is missing its structural resident peak row"
                )));
            }
            return Ok(None);
        }
    };
    if receipt_priced && !artifact_is_certified {
        // Receipt-priced optimized contracts are tied to the immutable turnkey revision and physical tier.
        // An imported/custom root retains the historical resident path and receives no estimate.
        return Ok(None);
    }
    let contract = match contract_override {
        Some(contract) => contract,
        None => {
            let Some(contract) = crate::inference_runtime::media()
                .memory_strategy_contract(engine_id, spec)
                .map_err(|error| {
                    WorkerError::Engine(format!("{engine_id} memory contract failed: {error}"))
                })?
            else {
                return Ok(None);
            };
            contract
        }
    };
    let provider_overlay = if is_chroma(engine_id) {
        validate_chroma_asset_facts(engine_id, &contract)?;
        chroma_provider_overlay_identity(&contract, provider_overlay)?
    } else if is_ideogram(engine_id) {
        validate_ideogram_asset_facts(engine_id, &contract, use_pid, &mode.mode, worker_multipass)?;
        ideogram_provider_overlay_identity(engine_id, &contract, provider_overlay)?
    } else if is_sana(engine_id) {
        validate_sana_asset_facts(engine_id, &contract)?;
        if provider_overlay.is_some() {
            return Err(WorkerError::InvalidPayload(format!(
                "{engine_id} does not accept an adapter overlay"
            )));
        }
        None
    } else if is_sd35(engine_id) {
        validate_sd35_asset_facts(engine_id, &contract)?;
        sd35_provider_overlay_identity(engine_id, &contract, provider_overlay)?
    } else if engine_id == "kolors" {
        let exact = kolors_overlay_receipt_identity(engine_id, &contract, use_pid)?;
        let has_declared_adapters = provider_overlay.is_some();
        let has_receipted_adapters = contract
            .resident_components()
            .iter()
            .any(|component| component.id.starts_with(KOLORS_ADAPTER_RECEIPT_PREFIX));
        if has_declared_adapters != has_receipted_adapters {
            return Err(WorkerError::InvalidPayload(
                "kolors public ordered-adapter identity crossed its exact provider receipt"
                    .to_owned(),
            ));
        }
        exact
    } else if is_sealed_kolors_bespoke(engine_id) {
        let exact = kolors_overlay_receipt_identity(engine_id, &contract, use_pid)?;
        match (provider_overlay, exact) {
            (None, None) => None,
            (Some(_), Some(exact)) => Some(exact),
            _ => {
                return Err(WorkerError::InvalidPayload(format!(
                    "{engine_id} public overlay crossed its exact provider receipt"
                )))
            }
        }
    } else {
        provider_overlay.map(str::to_owned)
    };
    let provider_overlay = provider_overlay.as_deref();
    let calibration = contract.calibration.as_ref();
    let resident_selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: Default::default(),
        tier,
    };
    let mut resident = sceneworks_core::memory_contracts::unverified_estimate(
        MemoryEvidenceKey {
            model_family: engine_id.to_owned(),
            resolved_route: engine_id.to_owned(),
            backend: MemoryBackend::Candle,
            tier,
            mode: mode.mode.clone(),
            reference_shape: if geometry.reference_count == 0 {
                gen_core::MemoryReferenceShape::None
            } else {
                gen_core::MemoryReferenceShape::Image
            },
            load_shape: contract.load_shape,
            overlay: provider_overlay.map(str::to_owned),
            geometry,
            frames_per_second: None,
            strategy: MemoryStrategy::Resident,
            engaged_composition: contract.engaged_composition(MemoryStrategy::Resident),
            parameters: resident_selection.parameters,
        },
        calibration.map_or(gen_core::MEMORY_CALIBRATION_ABI, |item| item.abi),
        calibration.map_or_else(String::new, |item| item.fingerprint.clone()),
        request_evidence_revision.to_owned(),
        {
            let declared = scale_ideogram_hires_envelope(
                engine_id,
                &mode.mode,
                manifest,
                geometry,
                (resident_peak_gb * BYTES_PER_GIB).ceil() as u64,
            );
            if receipt_priced {
                let structural = contract
                    .asset_facts
                    .base_bytes
                    .saturating_add((crate::vram_gate::HEADROOM_GB * BYTES_PER_GIB).ceil() as u64);
                contract
                    .predicted_peak_from_base(declared.max(structural))
                    .predicted_peak_bytes()
            } else {
                // The caller's resident peak is `vram_gate::predicted_peak_gb`: the measured
                // `vramGbByTier` row plus `HEADROOM_GB` (or `minMemoryGb`, padded by the manifest
                // itself). The pad stays INSIDE this candidate and it is compared against the
                // unreserved pool (`crate::memory_strategy::ReserveCharge`, sc-22664).
                declared
            }
        },
    );
    let exact_overlay = overlay.unwrap_or("none");
    // Matrix/evidence overlays remain cells of the advertised base provider (`z_image` or
    // `z_image_turbo`), while the runtime registers the strict-control implementation under a
    // dedicated `_control` id. Query the packaged evidence by its catalog identity, then bind the
    // returned candidate to the exact runtime route expected by the provider contract.
    let mut verified = if artifact_is_certified {
        verified_candidates(
            manifest,
            model_id,
            engine_id,
            tier_key,
            &mode,
            exact_overlay,
            geometry,
        )?
    } else {
        Vec::new()
    };
    // The provider-owned calibration identity answers whether the measured memory semantics
    // changed, and it has to match: FLUX.2-dev's caption-upsample lifecycle intentionally rotated
    // v2 -> v3 while retaining the old historical records, and a v2 record would feed an obsolete
    // prompt-conditioning peak into the live v3 contract. This is a provider DECLARATION, not
    // measurement currency — the closure digest a record was captured under is deliberately not
    // read here (sc-22738).
    let mut current_verified = Vec::with_capacity(verified.len());
    if let Some(current_calibration) = calibration {
        for candidate in verified {
            if candidate.calibration_fingerprint == current_calibration.fingerprint {
                current_verified.push(candidate);
            }
        }
    }
    verified = current_verified;
    // Packaged bindings are looked up by the public catalog/matrix overlay above. Once admitted,
    // however, every candidate submitted to the provider selector must carry the exact provider
    // evidence identity declared by the route. Krea adapters are load identity, not a provider
    // `MemoryRunContext` overlay, so their public `lora` cell is normalized to `None` here.
    for candidate in &mut verified {
        candidate.key.overlay = provider_overlay.map(str::to_owned);
    }
    // Calibration records describe the certified overlay fixture. User-provided adapters can be
    // larger, so every optimized candidate must reserve the bytes for the actual request before
    // the common selector performs its fit check. Legacy resident estimates already include the
    // caller's source-byte charge and must not be adjusted twice. Chroma instead ignores that raw
    // value and uses the provider receipt's materialized adapter + PiD aggregate here and above.
    let runtime_overlay_bytes = if receipt_priced {
        contract.asset_facts.overlay_bytes
    } else {
        runtime_overlay_bytes
    };
    account_for_runtime_overlay_bytes(&mut verified, runtime_overlay_bytes);
    let sdxl_endpoints = sdxl_endpoints.filter(|pair| {
        engine_id == "sdxl"
            && model_id == "sdxl"
            && tier_key == "q4"
            && artifact_is_certified
            && mode.mode == MemoryMode::TextToImage
            && !has_reference
            && !use_pid
            && !worker_multipass
            && geometry.batch == 1
            && geometry.frames == 1
            && geometry.reference_count == 0
            && provider_overlay.is_none()
            && runtime_overlay_bytes == 0
            && spec.resolved_route.as_deref() == Some("sdxl")
            && calibration.is_some_and(|identity| {
                identity.fingerprint == pair.resident.source.calibration_fingerprint
            })
            && manifest
                .get("downloads")
                .and_then(Value::as_array)
                .is_some_and(|downloads| {
                    downloads.iter().any(|download| {
                        download.get("repo").and_then(Value::as_str)
                            == Some(pair.artifact.repository.as_str())
                            && download.get("revision").and_then(Value::as_str)
                                == Some(pair.artifact.resolved_revision.as_str())
                            && download.get("variant").and_then(Value::as_str)
                                == Some(pair.artifact.variant.as_str())
                    })
                })
    });
    let resident_endpoint = sdxl_endpoints.and_then(|pair| {
        sdxl_endpoint_candidate(
            &contract,
            &resident,
            &pair.resident,
            resident_selection,
            pair.resident_load_peak_bytes,
        )
    });
    let resident_basis = resident_endpoint.as_ref().map_or(
        crate::memory_strategy::CandidateBasis::Measured,
        |candidate| candidate.basis,
    );
    let resident_has_manifest_pad = resident_endpoint.is_none();
    if let Some(candidate) = &resident_endpoint {
        resident = candidate.evidence.clone();
    }
    if let Some(exact) = verified.first() {
        resident
            .inference_revision
            .clone_from(&exact.inference_revision);
    }
    // sc-18097: estimate-floor candidates for every implemented optimized rung, so an unmeasured
    // cell can still engage the ladder behind the candle estimate margin instead of freezing to
    // resident-or-legacy behavior. Where an exact measured record exists the selector's
    // measured-supersedes-estimate rule keeps admission byte-for-byte unchanged.
    //
    // Gated on `artifact_is_certified`, the same conjunct that gates the packaged records above
    // (sc-18097 review, major finding). The floors are read from the SHIPPED manifest's
    // `sequentialPeakGb`/`vramGbByTier` rows, which describe the certified artifact. An imported
    // or community checkpoint on a supported route is different bytes: those rows do not describe
    // it, so extending its ladder from them would be a static capability declaration authorizing
    // an optimized request — exactly what this module's header forbids — and CUDA OOM being
    // recoverable does not make a wrong prediction safe. An uncertified artifact therefore keeps
    // its resident-estimate-only behavior, byte-for-byte as before this story. The sibling control
    // lane gates its floors the same way (`krea_control_fit.rs`, `runtime_verified`).
    // Reuse the existing image derivation law for intermediate configurations. The endpoint pair
    // has its own source-validated store, leaving the general catalog's unique-anchor policy intact.
    let endpoint_anchor_store =
        sdxl_endpoints.map(|pair| sceneworks_core::memory_anchor::MemoryAnchorStore {
            schema_version: sceneworks_core::memory_anchor::MEMORY_ANCHOR_SCHEMA_VERSION,
            anchors: vec![pair.resident.clone()],
            analytic_only: Vec::new(),
            component_deltas: Vec::new(),
            exceeded_bounds: Vec::new(),
            // The endpoint pair is a measured capacity capture, not an owned-process accounting
            // observation; the ladder store carries no operating envelopes.
            observed_operating_envelopes: Vec::new(),
        });
    let endpoint_anchors = endpoint_anchor_store
        .as_ref()
        .map(|store| CandleLadderAnchors {
            store: Some(store),
            facts: crate::video_admission::architecture_facts_from_contract(&contract),
        });
    let mut synthesized = if artifact_is_certified {
        synthesize_estimate_floors(
            engine_id,
            model_id,
            &contract,
            manifest,
            tier_key,
            tier,
            &mode,
            provider_overlay,
            geometry,
            if resident_endpoint.is_some() {
                (resident_peak_gb * BYTES_PER_GIB).ceil() as u64
            } else {
                resident.predicted_peak_bytes
            },
            runtime_overlay_bytes,
            request_evidence_revision,
            endpoint_anchors
                .or(ladder_anchors_override)
                .unwrap_or_else(|| CandleLadderAnchors::packaged(&contract)),
        )
    } else {
        Vec::new()
    };
    // This is a separate request demand, NOT an adapter overlay or a text-token hint. Charge
    // every complete candidate exactly once, after receipt overlay normalization and phase-law
    // synthesis but BEFORE selector grading. Keeping the source evidence objects charged also
    // keeps selected-context and warm-alternative peak reporting identical to the demand graded.
    account_for_runtime_overlay_bytes(
        std::slice::from_mut(&mut resident),
        reference_workspace_bytes,
    );
    account_for_runtime_overlay_bytes(&mut verified, reference_workspace_bytes);
    for candidate in &mut synthesized {
        account_for_runtime_overlay_bytes(
            std::slice::from_mut(&mut candidate.evidence),
            reference_workspace_bytes,
        );
    }
    if let Some(pair) = sdxl_endpoints {
        for candidate in &mut synthesized {
            if matches!(
                candidate.basis,
                crate::memory_strategy::CandidateBasis::EstimateAnchorDerived { .. }
            ) {
                let load_peak = if contract
                    .engages_selection(&candidate.selection, MemoryStrategy::StagedResidency)
                {
                    pair.sequential_load_peak_bytes
                } else {
                    pair.resident_load_peak_bytes
                };
                candidate.evidence.predicted_peak_bytes =
                    candidate.evidence.predicted_peak_bytes.max(load_peak);
            }
        }
        let engaged = [
            StrategyRung::Resident,
            StrategyRung::StagedResidency,
            StrategyRung::BoundedDecode,
            StrategyRung::BoundedAttention,
        ];
        if let Some(parameters) = parse_parameters(&engaged, &pair.sequential_parameters) {
            let selection = MemorySelection {
                strategy: MemoryStrategy::BoundedAttention,
                parameters,
                tier,
            };
            if let Some(endpoint) = sdxl_endpoint_candidate(
                &contract,
                &resident,
                &pair.sequential,
                selection,
                pair.sequential_load_peak_bytes,
            ) {
                // A prediction of this same exact composition must not supersede its observed
                // endpoint. Other compositions retain their independently derived estimates.
                synthesized.retain(|candidate| candidate.selection != endpoint.selection);
                synthesized.push(endpoint);
            }
        }
    }
    let capacity = verified.len() + synthesized.len() + 1;
    let mut selections = Vec::with_capacity(capacity);
    let mut evidence = Vec::with_capacity(capacity);
    // Index-aligned basis axis (sc-18097): the synthesized floors carry their estimate basis;
    // the resident live estimate and every packaged record stay `Measured`.
    let mut candidate_bases = Vec::with_capacity(capacity);
    selections.push(resident_selection);
    evidence.push(&resident);
    candidate_bases.push(resident_basis);
    for item in &verified {
        selections.push(MemorySelection {
            strategy: item.key.strategy,
            parameters: item.key.parameters,
            tier: item.key.tier,
        });
        evidence.push(item);
        candidate_bases.push(crate::memory_strategy::CandidateBasis::Measured);
    }
    for candidate in &synthesized {
        selections.push(candidate.selection);
        evidence.push(&candidate.evidence);
        candidate_bases.push(candidate.basis);
    }
    debug_assert_eq!(evidence.len(), candidate_bases.len());
    let candidates = selections
        .iter()
        .zip(evidence)
        .zip(&candidate_bases)
        .map(|((selection, evidence), basis)| Candidate {
            selection: *selection,
            evidence,
            basis: *basis,
            // sc-22508: the candle floors here are manifest `sequentialPeakGb`/`vramGbByTier`
            // rows, not a weights+headroom split this lane can decompose, so no activation term
            // is declared and the selector charges the backend's whole-peak accounting residual
            // (`CANDLE_RECAPTURE_SPREAD`). Declaring a split is sc-22509's anchor work.
            unmodeled_activation_bytes: None,
        })
        .collect::<Vec<_>>();
    let request_scope = RequestScope {
        resolved_route: engine_id,
        backend: "candle",
        tier,
        mode: &mode.scope_key,
        overlay: provider_overlay,
        geometry,
    };
    // sc-22664: the operational reserve — `vram_gate::ladder_reserve_gb` of the caller's RAW probe,
    // handed in explicitly so a reclaimable-credited `budget` can never derive it — on the selector
    // budget, charged per candidate by the one rule `crate::memory_strategy::ReserveCharge`
    // states: an anchor-derived candidate (a reserve-free device delta) compares against the pool
    // minus the reserve; the resident live estimate, a manifest-row floor and a receipt-priced
    // structural floor already carry `HEADROOM_GB` inside their peaks and compare against the
    // unreserved pool, so no candidate pays twice.
    let selector_budget = Some(Budget {
        available_gb: budget.free_gb,
        reclaimable_gb: 0.0,
        total_gb: budget.total_gb,
        reserved_headroom_gb: reserve_gb,
    });
    let pad_carrying = |candidate: &Candidate<'_>| {
        (resident_has_manifest_pad && std::ptr::eq(candidate.evidence, &resident))
            || candidate.basis == crate::memory_strategy::CandidateBasis::EstimateFloor
    };
    let reserve_charge = crate::memory_strategy::ReserveCharge::ExceptPadCarrying(&pad_carrying);
    // Epic 25899: the provider's prefix KV cache, added rung-for-rung. The ladder runs first at the
    // shed demand — exactly the pre-cache admission, which picks the rung — and the cache is taken
    // only if that SAME rung (`only_rung`) also fits with the surcharge on its peak, under the same
    // budget and pad rule. The cache never moves a request to another rung. `None` runs the ladder
    // once, unchanged.
    let run_ladder = |extra_bytes: u64, only_rung: Option<MemorySelection>| -> Selection {
        let on_rung =
            |candidate: &&Candidate<'_>| only_rung.is_none_or(|rung| candidate.selection == rung);
        if extra_bytes == 0 && only_rung.is_none() {
            return crate::memory_strategy::select_strategy_charging(
                request_scope,
                &contract,
                selector_budget,
                &candidates,
                reserve_charge,
            );
        }
        let kept = candidates.iter().filter(on_rung).collect::<Vec<_>>();
        let surcharged = kept
            .iter()
            .map(|candidate| {
                let mut evidence = candidate.evidence.clone();
                evidence.predicted_peak_bytes =
                    evidence.predicted_peak_bytes.saturating_add(extra_bytes);
                evidence
            })
            .collect::<Vec<_>>();
        let surcharged_candidates = kept
            .iter()
            .zip(&surcharged)
            .map(|(candidate, evidence)| Candidate {
                evidence,
                ..**candidate
            })
            .collect::<Vec<_>>();
        // The pad rule follows each candidate from the original list: the resident live estimate
        // and every estimate floor carry their pad.
        let carries_pad = kept
            .iter()
            .map(|candidate| pad_carrying(candidate))
            .collect::<Vec<_>>();
        let surcharged_pad_carrying = |candidate: &Candidate<'_>| {
            surcharged
                .iter()
                .position(|evidence| std::ptr::eq(candidate.evidence, evidence))
                .is_some_and(|index| carries_pad[index])
        };
        crate::memory_strategy::select_strategy_charging(
            request_scope,
            &contract,
            selector_budget,
            &surcharged_candidates,
            crate::memory_strategy::ReserveCharge::ExceptPadCarrying(&surcharged_pad_carrying),
        )
    };
    let (selected, prefix_kv_cache) = crate::execution_planner::plan_prefix_kv_cache(
        prefix_kv_cache_surcharge,
        run_ladder(0, None),
        |selection| matches!(selection, Selection::Selected { .. }),
        |shed, priced_bytes| {
            let Selection::Selected {
                selection: rung, ..
            } = shed
            else {
                return None;
            };
            let cached = run_ladder(priced_bytes, Some(*rung));
            matches!(&cached, Selection::Selected { selection, .. } if selection == rung)
                .then_some(cached)
        },
    );
    let prefix_kv_cache_extra = prefix_kv_cache.map_or(0, |plan| plan.admitted_surcharge_bytes());
    let Selection::Selected {
        selection,
        needed_gb,
        available_gb,
    } = selected
    else {
        if receipt_priced {
            return Err(WorkerError::InvalidPayload(format!(
                "{engine_id} has no exact resident or staged strategy that fits the sealed provider receipt"
            )));
        }
        return Ok(None);
    };
    let evidence_for_selection = |selection: MemorySelection| {
        let matches_selection = |item: &MemoryEvidence| {
            item.key.strategy == selection.strategy
                && item.key.parameters == selection.parameters
                && item.key.tier == selection.tier
        };
        if selection.strategy == MemoryStrategy::Resident {
            Some((
                &resident,
                resident_basis.is_estimate(),
                resident_endpoint.as_ref(),
            ))
        } else if let Some(item) = verified.iter().find(|item| matches_selection(item)) {
            Some((item, false, None))
        } else {
            synthesized
                .iter()
                .find(|candidate| matches_selection(&candidate.evidence))
                // sc-18097: synthesized floors are estimate-scoped, never calibrated evidence.
                .map(|candidate| (&candidate.evidence, true, Some(candidate)))
        }
    };
    let (selected_evidence, estimate_scoped, selected_estimate) = evidence_for_selection(selection)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{engine_id} selected a memory strategy without exact packaged evidence"
            ))
        })?;
    let to_bytes = |gb: f64| (gb * BYTES_PER_GIB).round().clamp(0.0, u64::MAX as f64) as u64;
    let context_for = |selection: MemorySelection,
                       selected_evidence: &MemoryEvidence,
                       estimate_scoped: bool|
     -> MemoryRunContext {
        MemoryRunContext {
            selection,
            optimization_authority: if selection.strategy == MemoryStrategy::Resident {
                gen_core::MemoryOptimizationAuthority::Resident
            } else if estimate_scoped {
                gen_core::MemoryOptimizationAuthority::Estimated
            } else {
                gen_core::MemoryOptimizationAuthority::Calibrated
            },
            calibration_abi: selected_evidence.calibration_abi,
            calibration_fingerprint: selected_evidence.calibration_fingerprint.clone(),
            mode: mode.mode.clone(),
            load_shape: contract.load_shape,
            has_reference,
            use_pid,
            has_phases: request_has_phases,
            geometry,
            reference_resources: reference_resources.cloned(),
            overlay: provider_overlay.map(str::to_owned),
            budget: gen_core::MemoryBudget {
                total_bytes: to_bytes(budget.total_gb),
                committed_bytes: to_bytes((budget.total_gb - budget.free_gb).max(0.0)),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: to_bytes(reserve_gb),
            },
            predicted_peak_bytes: selected_evidence.predicted_peak_bytes,
            cache_state,
            evidence_revision: if (selection.strategy == MemoryStrategy::Resident
                && selected_evidence.conformance != MemoryConformanceState::Verified)
                || estimate_scoped
            {
                request_evidence_revision.to_owned()
            } else {
                selected_evidence.harness_version.clone()
            },
        }
    };
    let mut selected_context = context_for(selection, selected_evidence, estimate_scoped);
    // The demand the admission approved: a cached selection was graded at shed + surcharge.
    selected_context.predicted_peak_bytes = selected_context
        .predicted_peak_bytes
        .saturating_add(prefix_kv_cache_extra);

    // A loaded Sequential cache entry is a tighter peak shape than a new Resident request. Keep an
    // exact, independently selected staged sibling so the cache callback can honor that already
    // materialized shape. This is not an admission bypass: the same authoritative request scope,
    // evidence set, estimate margin, and budget select it here before the load/cache access.
    let warm_staged = if is_ideogram(engine_id) {
        let staged_candidates = candidates
            .iter()
            .filter(|candidate| {
                contract.engages(
                    candidate.selection.strategy,
                    MemoryStrategy::StagedResidency,
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        match crate::memory_strategy::select_strategy_charging(
            request_scope,
            &contract,
            selector_budget,
            &staged_candidates,
            reserve_charge,
        ) {
            Selection::Selected {
                selection: staged_selection,
                ..
            } => {
                let (evidence, estimate_scoped, _) = evidence_for_selection(staged_selection)
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "{engine_id} selected a warm staged strategy without exact evidence"
                        ))
                    })?;
                contract.generation_memory(&staged_selection).map(|memory| {
                    CandleWarmStagedEvaluation {
                        memory,
                        context: context_for(staged_selection, evidence, estimate_scoped),
                    }
                })
            }
            Selection::Reject { .. } | Selection::Unverified { .. } => None,
        }
    } else {
        None
    };
    let mut memory = memory_for_selection(&contract, selection);
    if let Some(plan) = prefix_kv_cache.filter(|plan| plan.mode.is_cached()) {
        // A Resident selection has no block; an admitted cached selection needs one to live in.
        plan.write_onto(memory.get_or_insert_with(GenerationMemory::default));
    }
    Ok(Some(CandleMemoryEvaluation {
        memory,
        prefix_kv_cache,
        measured_endpoint_peak_bytes: (selected_estimate.is_some_and(|candidate| {
            candidate.basis == crate::memory_strategy::CandidateBasis::EstimateMeasuredEndpoint
        }))
        .then_some(selected_evidence.observed_peak_bytes)
        .flatten(),
        predicted_peak_gb: selected_evidence
            .predicted_peak_bytes
            .saturating_add(prefix_kv_cache_extra) as f64
            / BYTES_PER_GIB,
        context: selected_context,
        warm_staged,
        basis: selected_estimate.map_or_else(
            || {
                if selection.strategy == MemoryStrategy::Resident {
                    resident_basis
                } else {
                    crate::memory_strategy::CandidateBasis::Measured
                }
            },
            |candidate| candidate.basis,
        ),
        phase_peaks: selected_estimate.and_then(|candidate| candidate.phase_peaks),
        admitted: AdmittedBudget {
            needed_gb,
            available_gb,
            reserve_gb,
        },
    }))
}

#[cfg(test)]
mod tests;
