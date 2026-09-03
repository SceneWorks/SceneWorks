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

use gen_core::{
    GenerationMemory, LoadSpec, MemoryBackend, MemoryCacheState, MemoryConformanceState,
    MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey, MemoryEvidenceVerdict,
    MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryParityContract, MemoryParityResult,
    MemoryRunContext, MemorySelection, MemoryStrategy, Precision, Quant,
};
use sceneworks_core::memory_calibration::{
    Backend as CalibrationBackend, BundleLoad, CalibrationBinding, EvidenceQuery, EvidenceVerdict,
    Geometry as CalibrationGeometry, LoadShapeKey, QualityResult, RequiredNullable, StrategyRung,
};
use serde_json::{Map as JsonObject, Value};
use sha2::Digest;

use crate::memory_strategy::{Budget, Candidate, RequestScope, Selection};
use crate::vram_gate::VramBudget;
use crate::{WorkerError, WorkerResult};

const Z_IMAGE_REQUEST_EVIDENCE_REVISION: &str = "sc-15815-candle-z-image-request-scope-v1";
const QWEN_IMAGE_REQUEST_EVIDENCE_REVISION: &str = "sc-15817-candle-qwen-image-request-scope-v1";
const FLUX1_REQUEST_EVIDENCE_REVISION: &str = "sc-15823-candle-flux1-request-scope-v1";
const FLUX2_DEV_REQUEST_EVIDENCE_REVISION: &str = "sc-15833-candle-flux2-dev-request-scope-v1";
const FLUX2_KLEIN_REQUEST_EVIDENCE_REVISION: &str = "sc-15831-candle-flux2-klein-request-scope-v1";
const MAGE_FLOW_REQUEST_EVIDENCE_REVISION: &str = "sc-15813-candle-mage-flow-request-scope-v1";
const CHROMA_REQUEST_EVIDENCE_REVISION: &str = "sc-20788-candle-chroma-staged-residency-v2";
const IDEOGRAM_REQUEST_EVIDENCE_REVISION: &str =
    "sc-20789-candle-ideogram-request-scoped-staged-residency-v1";
pub(crate) const PULID_FLUX_REQUEST_EVIDENCE_REVISION: &str =
    "sc-15839-candle-pulid-flux-request-scope-v1";
const DECLARATION_REQUEST_EVIDENCE_REVISION: &str = "sc-18456-candle-declaration-request-scope-v1";
pub(crate) const SDXL_REQUEST_EVIDENCE_REVISION: &str = "sdxl-candle-request-contract-v1";
const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const CHROMA_ADAPTER_OVERLAY_PREFIX: &str = "chroma.adapters.ordered-additive.sha256:";
const IDEOGRAM_ADAPTER_OVERLAY_PREFIX: &str = "ideogram.adapters.ordered-additive.sha256:";
const IDEOGRAM_TURBO_ADAPTER_PREFIX: &str = "ideogram.turbo-time.sha256:";
const IDEOGRAM_PID_PREFIX: &str = "ideogram.pid.flux2.sha256:";
const IDEOGRAM_PHYSICAL_RECEIPT_PREFIX: &str = "ideogram.physical.sha256:";
const SANA_PHYSICAL_RECEIPT_PREFIX: &str = "sana.candle.dense.physical.sha256:";
const SD35_PHYSICAL_RECEIPT_PREFIX: &str = "sd3.5.candle.physical.sha256:";
const SD35_ADAPTER_RECEIPT_PREFIX: &str = "sd3.5.adapters.ordered-additive.sha256:";
pub(crate) const KOLORS_REQUEST_EVIDENCE_REVISION: &str = "kolors-candle-request-contract-v1";
const SANA_REQUEST_EVIDENCE_REVISION: &str = "sana-candle-dense-request-contract-v1";
const SD35_REQUEST_EVIDENCE_REVISION: &str = "sd3.5-candle-request-contract-v1";
const KOLORS_PHYSICAL_RECEIPT_PREFIX: &str = "kolors.physical.sha256:";
const KOLORS_ADAPTER_RECEIPT_PREFIX: &str = "kolors.adapters.ordered.sha256:";
const KOLORS_IP_RECEIPT_PREFIX: &str = "kolors.ip.sha256:";
const KOLORS_CONTROL_RECEIPT_PREFIX: &str = "kolors.control.sha256:";
const KOLORS_PID_RECEIPT_PREFIX: &str = "kolors.pid.sdxl.sha256:";
const SDXL_OVERLAY_RECEIPT_DOMAIN: &str = "sdxl-candle-overlay-assembly-v1";
const SDXL_OVERLAY_RECEIPT_PREFIX: &str = "sdxl.overlay.ordered.sha256:";

fn is_chroma(engine_id: &str) -> bool {
    matches!(engine_id, "chroma1_hd" | "chroma1_base" | "chroma1_flash")
}

fn is_ideogram(engine_id: &str) -> bool {
    matches!(engine_id, "ideogram_4" | "ideogram_4_turbo")
}

fn is_sana(engine_id: &str) -> bool {
    matches!(engine_id, "sana_1600m" | "sana_sprint_1600m")
}

pub(crate) fn is_sd35(engine_id: &str) -> bool {
    matches!(
        engine_id,
        "sd3_5_large" | "sd3_5_large_turbo" | "sd3_5_medium"
    )
}

fn is_sealed_kolors_bespoke(engine_id: &str) -> bool {
    matches!(
        engine_id,
        "candle_kolors_ipadapter" | "candle_kolors_control"
    )
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
            "phasePeakBytes": self.phase_peaks.map(|phases| serde_json::json!({
                "conditioning": phases.conditioning,
                "denoise": phases.denoise,
                "decode": phases.decode,
            })),
            "admittedPeakGb": self.admitted.needed_gb,
            "availableGb": self.admitted.available_gb,
            "reserveGb": self.admitted.reserve_gb,
            "evidenceRevision": self.context.evidence_revision.clone(),
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

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn declared_component_floors(engine_id: &str) -> &'static [gen_core::ComponentPrecisionFloor] {
    crate::inference_runtime::media_descriptor(engine_id)
        .map(|descriptor| descriptor.capabilities.component_precision_floors)
        .unwrap_or(&[])
}

#[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
fn declared_component_floors(_: &str) -> &'static [gen_core::ComponentPrecisionFloor] {
    &[]
}

fn active_component_floors(
    engine_id: &str,
    selected: Option<Quant>,
) -> &'static [gen_core::ComponentPrecisionFloor] {
    let declared = declared_component_floors(engine_id);
    match selected {
        Some(selected)
            if !declared.is_empty() && declared.iter().all(|floor| floor.applies_to(selected)) =>
        {
            declared
        }
        _ => &[],
    }
}

fn numeric_tier(engine_id: &str, tier: &str) -> Option<MemoryNumericTier> {
    let quant = match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
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
    const KEYS: [&str; 5] = [
        "decodeTileEdge",
        "decodeOverlap",
        "attentionChunkSize",
        "transformerWindowSize",
        "transformerWindowComponent",
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
    Some(gen_core::MemoryStrategyParameters {
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
    closure_digests: &mut Vec<String>,
) -> WorkerResult<Vec<MemoryEvidence>> {
    let evidence_provider = evidence_provider(runtime_provider);
    let loaded = sceneworks_core::memory_calibration::load_packaged_bundle().map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "packaged memory-calibration evidence is invalid: {error}"
        ))
    })?;
    let BundleLoad::Ready(bundle) = loaded else {
        return Ok(Vec::new());
    };
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
        // Index-aligned with `candidates`: `MemoryEvidenceKey` is a gen-core type without `Hash`,
        // and the two vectors are pushed together in this one loop.
        closure_digests.push(calibration.inference_closure_digest.clone());
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
    let smallest = |strategy: MemoryStrategy,
                    pick: fn(&gen_core::MemoryParameterRanges) -> &Vec<u32>|
     -> Option<Option<u32>> {
        if !engaged.contains(&strategy) {
            return Some(None);
        }
        pick(&contract.capability(strategy)?.parameters)
            .iter()
            .copied()
            .min()
            .map(Some)
    };
    let transformer_window_component =
        if engaged.contains(&MemoryStrategy::BoundedTransformerResidency) {
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)?
                .parameters
                .transformer_window_components
                .first()
                .copied()
        } else {
            None
        };
    Some(gen_core::MemoryStrategyParameters {
        decode_tile_edge: smallest(MemoryStrategy::BoundedDecode, |ranges| {
            &ranges.decode_tile_edges
        })?,
        decode_overlap: smallest(MemoryStrategy::BoundedDecode, |ranges| {
            &ranges.decode_overlaps
        })?,
        attention_chunk_size: smallest(MemoryStrategy::BoundedAttention, |ranges| {
            &ranges.attention_chunk_sizes
        })?,
        transformer_window_size: smallest(MemoryStrategy::BoundedTransformerResidency, |ranges| {
            &ranges.transformer_window_sizes
        })?,
        transformer_window_component,
    })
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
    let anchor = candle_image_anchor(
        anchors, engine_id, model_id, contract, tier_key, mode, overlay, geometry,
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
    for strategy in MemoryStrategy::ALL {
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
        let Some(parameters) = estimate_floor_parameters(contract, &engaged) else {
            continue;
        };
        let selection = MemorySelection {
            strategy,
            parameters,
            tier,
        };
        if contract.validate_selection(&selection).is_err() {
            continue;
        }
        let staged = engaged.contains(&MemoryStrategy::StagedResidency);
        // The rung's own regime: its engaged composition and the parameters it was selected with.
        // Every rung is priced from THIS, never from the staged rung's number (sc-22664).
        let regime = request_regime(&engaged, &parameters);
        let request = |regime: RequestRegime| ImageDeriveRequest {
            width: geometry.width,
            height: geometry.height,
            batch: geometry.batch.max(1),
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
            evidence: MemoryEvidence {
                key: MemoryEvidenceKey {
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
                conformance: MemoryConformanceState::ImplementedUnverified,
                dimensions: MemoryEvidenceDimensions {
                    static_implementation: MemoryEvidenceVerdict::Satisfied,
                    declared_calibration: MemoryEvidenceVerdict::Missing,
                    historical_verification: MemoryEvidenceVerdict::Missing,
                    current_environment_verification: MemoryEvidenceVerdict::Missing,
                    canonical_route_loadability: MemoryEvidenceVerdict::Unverified,
                    exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
                },
                calibration_abi: calibration
                    .map_or(gen_core::MEMORY_CALIBRATION_ABI, |item| item.abi),
                calibration_fingerprint: calibration
                    .map_or_else(String::new, |item| item.fingerprint.clone()),
                sceneworks_revision: request_evidence_revision.to_owned(),
                inference_revision: crate::catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION
                    .to_owned(),
                harness_version: String::new(),
                predicted_peak_bytes,
                observed_peak_bytes: None,
                parity: MemoryParityContract::Exact,
                parity_result: MemoryParityResult::NotRun,
            },
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
    use sceneworks_core::memory_anchor::{DecodeTile, RequestRegime};
    let engages = |strategy: MemoryStrategy| engaged.contains(&strategy);
    Some(RequestRegime {
        staged: engages(MemoryStrategy::StagedResidency),
        decode_tile: if engages(MemoryStrategy::BoundedDecode) {
            Some(DecodeTile {
                edge: parameters.decode_tile_edge?,
                overlap: parameters.decode_overlap?,
            })
        } else {
            None
        },
        attention_chunk_scores: if engages(MemoryStrategy::BoundedAttention) {
            Some(u64::from(parameters.attention_chunk_size?))
        } else {
            None
        },
        transformer_window: if engages(MemoryStrategy::BoundedTransformerResidency) {
            Some(parameters.transformer_window_size?)
        } else {
            None
        },
    })
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
    // Currency (sc-22511, epic 22505 E9): the model's OWN loader closure, and nothing else. The
    // calibration ABI and fingerprint are deliberately NOT asked here — since sc-22511 they are
    // provenance, bound to the source record by `validate_anchor` so the anchor cannot misattribute
    // its origin, but a new campaign no longer demotes evidence whose loader never moved. This is
    // the same seam `video_admission::anchor_currency_matches` and `vram_gate::krea_store_anchor`
    // grade on, so no two lanes can disagree about whether an anchor is live.
    //
    // It grades EVERY store since sc-22666: the per-store `AnchorStoreScope` split existed to hold
    // the model allow-list, and with that gone a caller-supplied row is graded on exactly the
    // conjuncts a packaged one is. `config/anchor-loader-closures.json` declares a closure for
    // every packaged (model, lane), so a fixture row states its own model's digest or reads stale
    // -- which is the truth about it.
    crate::video_admission::anchor_currency_matches(anchor).then_some(anchor)
}

fn memory_for_selection(
    contract: &gen_core::MemoryProviderContract,
    selection: MemorySelection,
) -> Option<GenerationMemory> {
    (selection.strategy != MemoryStrategy::Resident).then(|| GenerationMemory {
        stage_residency: contract.engages(selection.strategy, MemoryStrategy::StagedResidency),
        tile_vae_decode: contract.engages(selection.strategy, MemoryStrategy::BoundedDecode),
        decode_tile_edge: selection.parameters.decode_tile_edge,
        decode_overlap: selection.parameters.decode_overlap,
        chunk_attention: contract.engages(selection.strategy, MemoryStrategy::BoundedAttention),
        attention_chunk_size: selection.parameters.attention_chunk_size,
        stream_transformer_blocks: contract.engages(
            selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        ),
        transformer_window_size: selection.parameters.transformer_window_size,
        transformer_window_component: selection.parameters.transformer_window_component,
        ..Default::default()
    })
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
    evaluate_shared_image_inner(
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
        "pulid_flux" => PULID_FLUX_REQUEST_EVIDENCE_REVISION,
        "candle_kolors_ipadapter" | "candle_kolors_control" => KOLORS_REQUEST_EVIDENCE_REVISION,
        "sdxl" => SDXL_REQUEST_EVIDENCE_REVISION,
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
    let request_evidence_revision = request_evidence_revision_override.unwrap_or(match engine_id {
        "z_image" | "z_image_turbo" | "z_image_control" | "z_image_turbo_control" => {
            Z_IMAGE_REQUEST_EVIDENCE_REVISION
        }
        "qwen_image" | "qwen_image_edit" => QWEN_IMAGE_REQUEST_EVIDENCE_REVISION,
        "flux1_schnell" | "flux1_dev" => FLUX1_REQUEST_EVIDENCE_REVISION,
        "flux2_dev" => FLUX2_DEV_REQUEST_EVIDENCE_REVISION,
        "flux2_klein_9b" => FLUX2_KLEIN_REQUEST_EVIDENCE_REVISION,
        "mage_flow_base"
        | "mage_flow"
        | "mage_flow_turbo"
        | "mage_flow_edit_base"
        | "mage_flow_edit"
        | "mage_flow_edit_turbo" => MAGE_FLOW_REQUEST_EVIDENCE_REVISION,
        "chroma1_hd" | "chroma1_base" | "chroma1_flash" => CHROMA_REQUEST_EVIDENCE_REVISION,
        "ideogram_4" | "ideogram_4_turbo" => IDEOGRAM_REQUEST_EVIDENCE_REVISION,
        "kolors" => KOLORS_REQUEST_EVIDENCE_REVISION,
        "sana_1600m" | "sana_sprint_1600m" => SANA_REQUEST_EVIDENCE_REVISION,
        "sd3_5_large" | "sd3_5_large_turbo" | "sd3_5_medium" => SD35_REQUEST_EVIDENCE_REVISION,
        "sdxl" => SDXL_REQUEST_EVIDENCE_REVISION,
        _ => DECLARATION_REQUEST_EVIDENCE_REVISION,
    });
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
    let mut resident = MemoryEvidence {
        key: MemoryEvidenceKey {
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
        conformance: MemoryConformanceState::ImplementedUnverified,
        dimensions: MemoryEvidenceDimensions {
            static_implementation: MemoryEvidenceVerdict::Satisfied,
            declared_calibration: MemoryEvidenceVerdict::Missing,
            historical_verification: MemoryEvidenceVerdict::Missing,
            current_environment_verification: MemoryEvidenceVerdict::Missing,
            canonical_route_loadability: MemoryEvidenceVerdict::Unverified,
            exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
        },
        calibration_abi: calibration.map_or(gen_core::MEMORY_CALIBRATION_ABI, |item| item.abi),
        calibration_fingerprint: calibration
            .map_or_else(String::new, |item| item.fingerprint.clone()),
        sceneworks_revision: request_evidence_revision.to_owned(),
        inference_revision: crate::catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION.to_owned(),
        harness_version: String::new(),
        predicted_peak_bytes: {
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
        observed_peak_bytes: None,
        parity: MemoryParityContract::Exact,
        parity_result: MemoryParityResult::NotRun,
    };
    let exact_overlay = overlay.unwrap_or("none");
    // Matrix/evidence overlays remain cells of the advertised base provider (`z_image` or
    // `z_image_turbo`), while the runtime registers the strict-control implementation under a
    // dedicated `_control` id. Query the packaged evidence by its catalog identity, then bind the
    // returned candidate to the exact runtime route expected by the provider contract.
    let mut closure_digests: Vec<String> = Vec::new();
    let mut verified = if artifact_is_certified {
        verified_candidates(
            manifest,
            model_id,
            engine_id,
            tier_key,
            &mode,
            exact_overlay,
            geometry,
            &mut closure_digests,
        )?
    } else {
        Vec::new()
    };
    // A closure digest answers whether the compiled provider code changed; the provider-owned
    // calibration identity answers whether the measured memory semantics changed. Both have to
    // match. FLUX.2-dev's caption-upsample lifecycle intentionally rotated v2 -> v3 while retaining
    // the old historical records, so accepting closure-stale evidence merely because the selector
    // can widen it would feed an obsolete prompt-conditioning peak into the live v3 contract.
    let mut current_verified = Vec::with_capacity(verified.len());
    let mut current_closure_digests = Vec::with_capacity(closure_digests.len());
    if let Some(current_calibration) = calibration {
        for (candidate, closure_digest) in verified.into_iter().zip(closure_digests) {
            if candidate.calibration_fingerprint == current_calibration.fingerprint {
                current_verified.push(candidate);
                current_closure_digests.push(closure_digest);
            }
        }
    }
    verified = current_verified;
    closure_digests = current_closure_digests;
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
    let synthesized = if artifact_is_certified {
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
            resident.predicted_peak_bytes,
            runtime_overlay_bytes,
            request_evidence_revision,
            ladder_anchors_override.unwrap_or_else(|| CandleLadderAnchors::packaged(&contract)),
        )
    } else {
        Vec::new()
    };
    let capacity = verified.len() + synthesized.len() + 1;
    let mut selections = Vec::with_capacity(capacity);
    let mut evidence = Vec::with_capacity(capacity);
    // The resident candidate is a live estimate, not a calibrated record, so it carries the live
    // digest and is never staled by this gate.
    let live_closure_digest = sceneworks_core::memory_calibration::packaged_closure_digest(
        "candle",
        evidence_provider(engine_id),
    )
    .unwrap_or_default();
    let mut candidate_digests = Vec::with_capacity(capacity);
    // Index-aligned basis axis (sc-18097): the synthesized floors carry their estimate basis;
    // the resident live estimate and every packaged record stay `Measured`.
    let mut candidate_bases = Vec::with_capacity(capacity);
    selections.push(resident_selection);
    evidence.push(&resident);
    candidate_digests.push(live_closure_digest.clone());
    candidate_bases.push(crate::memory_strategy::CandidateBasis::Measured);
    for (index, item) in verified.iter().enumerate() {
        candidate_digests.push(closure_digests.get(index).cloned().unwrap_or_default());
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
        // A floor — manifest-row or anchor-derived — is a declaration under the LIVE closure, not a
        // calibrated record; there is nothing there for currency to invalidate.
        candidate_digests.push(live_closure_digest.clone());
        candidate_bases.push(candidate.basis);
    }
    debug_assert_eq!(evidence.len(), candidate_bases.len());
    let candidates = selections
        .iter()
        .zip(evidence)
        .zip(candidate_digests.iter().zip(&candidate_bases))
        .map(
            |((selection, evidence), (closure_digest, basis))| Candidate {
                selection: *selection,
                evidence,
                closure_digest,
                basis: *basis,
                // sc-22508: the candle floors here are manifest `sequentialPeakGb`/`vramGbByTier`
                // rows, not a weights+headroom split this lane can decompose, so no activation term
                // is declared and the selector charges the backend's whole-peak accounting residual
                // (`CANDLE_RECAPTURE_SPREAD`). Declaring a split is sc-22509's anchor work.
                unmodeled_activation_bytes: None,
            },
        )
        .collect::<Vec<_>>();
    let request_scope = RequestScope {
        resolved_route: engine_id,
        backend: "candle",
        tier,
        mode: &mode.scope_key,
        overlay: provider_overlay,
        geometry,
        // sc-17774: one mechanism, same as every other lane. `unwrap_or_default` fails closed.
        expected_closure_digest: &live_closure_digest,
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
        std::ptr::eq(candidate.evidence, &resident)
            || candidate.basis == crate::memory_strategy::CandidateBasis::EstimateFloor
    };
    let reserve_charge = crate::memory_strategy::ReserveCharge::ExceptPadCarrying(&pad_carrying);
    let selected = crate::memory_strategy::select_strategy_charging(
        request_scope,
        &contract,
        selector_budget,
        &candidates,
        reserve_charge,
    );
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
            Some((&resident, false, None))
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
            overlay: provider_overlay.map(str::to_owned),
            budget: gen_core::MemoryBudget {
                total_bytes: to_bytes(budget.total_gb),
                committed_bytes: to_bytes((budget.total_gb - budget.free_gb).max(0.0)),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: to_bytes(reserve_gb),
            },
            predicted_peak_bytes: selected_evidence.predicted_peak_bytes,
            cache_state,
            evidence_revision: if selection.strategy == MemoryStrategy::Resident || estimate_scoped
            {
                request_evidence_revision.to_owned()
            } else {
                selected_evidence.harness_version.clone()
            },
        }
    };
    let selected_context = context_for(selection, selected_evidence, estimate_scoped);

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
    Ok(Some(CandleMemoryEvaluation {
        memory: memory_for_selection(&contract, selection),
        predicted_peak_gb: selected_evidence.predicted_peak_bytes as f64 / BYTES_PER_GIB,
        context: selected_context,
        warm_staged,
        basis: selected_estimate.map_or(
            crate::memory_strategy::CandidateBasis::Measured,
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
mod tests {
    use super::*;
    use gen_core::WeightsSource;
    use serde_json::json;
    use std::path::PathBuf;

    /// sc-17728, the Candle sibling of the MLX fit gate's coverage. The required parameter set
    /// follows the ENGAGED composition, so a provider that implements rung 4 with rungs 2 and 3
    /// `Missing` reads cleanly, while an engaged rung must still name every parameter it owns and a
    /// withheld rung's parameters stay unnameable.
    #[test]
    fn candle_parameters_follow_the_engaged_composition_not_the_rung_ordinal() {
        let withheld = [
            StrategyRung::Resident,
            StrategyRung::StagedResidency,
            StrategyRung::BoundedTransformerResidency,
        ];
        let cumulative = crate::memory_strategy::default_engaged_composition(
            StrategyRung::BoundedTransformerResidency,
        );
        let object = |value: Value| value.as_object().expect("parameter object").clone();

        let parsed = parse_parameters(
            &withheld,
            &object(json!({ "transformerWindowSize": 1, "transformerWindowComponent": "both" })),
        )
        .expect("a rung-4 composition that withholds rungs 2 and 3 must read");
        assert_eq!(parsed.transformer_window_size, Some(1));
        assert_eq!(parsed.decode_tile_edge, None);
        assert_eq!(parsed.attention_chunk_size, None);

        // Naming a withheld rung's parameter, and omitting an engaged rung's, both fail closed.
        assert!(parse_parameters(
            &withheld,
            &object(
                json!({ "transformerWindowSize": 1, "decodeTileEdge": 512, "decodeOverlap": 64 })
            ),
        )
        .is_none());
        assert!(parse_parameters(
            &withheld,
            &object(json!({ "transformerWindowComponent": "dit" }))
        )
        .is_none());

        // The cumulative default is unchanged for a provider with the whole ladder implemented.
        assert!(
            parse_parameters(&cumulative, &object(json!({ "transformerWindowSize": 1 })),)
                .is_none()
        );
        assert!(parse_parameters(
            &cumulative,
            &object(json!({
                "decodeTileEdge": 512,
                "decodeOverlap": 64,
                "attentionChunkSize": 256,
                "transformerWindowSize": 1,
            })),
        )
        .is_some());
    }

    /// The reserve a production caller hands the ladder for `budget`: `vram_gate::ladder_reserve_gb`
    /// of the RAW probe (sc-22664 review, D2). The fixtures below probe no card, so their budget
    /// IS the raw probe.
    fn reserve_for(budget: Option<VramBudget>) -> f64 {
        budget.map_or(0.0, crate::vram_gate::ladder_reserve_gb)
    }

    fn gib(value: u64) -> u64 {
        value.saturating_mul(BYTES_PER_GIB as u64)
    }

    fn ideogram_cache_context(strategy: MemoryStrategy) -> MemoryRunContext {
        MemoryRunContext {
            selection: MemorySelection {
                strategy,
                parameters: Default::default(),
                tier: numeric_tier("ideogram_4", "q4").expect("q4 tier"),
            },
            optimization_authority: gen_core::MemoryOptimizationAuthority::Estimated,
            calibration_abi: 0,
            calibration_fingerprint: String::new(),
            mode: MemoryMode::TextToImage,
            load_shape: gen_core::LoadShape::EagerMaterialization,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: gen_core::MemoryBudget {
                total_bytes: gib(48),
                committed_bytes: gib(4),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: gib(2),
            },
            predicted_peak_bytes: gib(if strategy == MemoryStrategy::Resident {
                38
            } else {
                34
            }),
            cache_state: MemoryCacheState::Cold,
            evidence_revision: IDEOGRAM_REQUEST_EVIDENCE_REVISION.to_owned(),
        }
    }

    #[test]
    fn ideogram_cache_binding_reports_actual_state_and_preserves_a_loaded_staged_shape() {
        let mut context = ideogram_cache_context(MemoryStrategy::Resident);
        let staged_memory = GenerationMemory {
            stage_residency: true,
            ..Default::default()
        };
        let mut memory = None;
        let mut staged = Some(CandleWarmStagedEvaluation {
            memory: staged_memory,
            context: ideogram_cache_context(MemoryStrategy::StagedResidency),
        });
        bind_ideogram_cache_execution(
            &mut context,
            &mut memory,
            &mut staged,
            MemoryCacheState::Warm,
            gen_core::OffloadPolicy::Sequential,
        )
        .expect("warm staged cache binding");
        assert_eq!(context.selection.strategy, MemoryStrategy::StagedResidency);
        assert_eq!(context.cache_state, MemoryCacheState::Warm);
        assert_eq!(memory, Some(staged_memory));
        assert!(staged.is_none(), "the exact fallback is once-only");

        let mut cold_context = ideogram_cache_context(MemoryStrategy::Resident);
        let mut cold_memory = None;
        let mut no_fallback = None;
        bind_ideogram_cache_execution(
            &mut cold_context,
            &mut cold_memory,
            &mut no_fallback,
            MemoryCacheState::Cold,
            gen_core::OffloadPolicy::Resident,
        )
        .expect("cold resident binding");
        assert_eq!(cold_context.cache_state, MemoryCacheState::Cold);
        assert_eq!(cold_context.selection.strategy, MemoryStrategy::Resident);
        assert_eq!(cold_memory, None);
    }

    #[test]
    fn ideogram_warm_sequential_binding_fails_closed_without_exact_staged_evidence() {
        let mut context = ideogram_cache_context(MemoryStrategy::Resident);
        let mut memory = None;
        let mut missing = None;
        let error = bind_ideogram_cache_execution(
            &mut context,
            &mut memory,
            &mut missing,
            MemoryCacheState::Warm,
            gen_core::OffloadPolicy::Sequential,
        )
        .expect_err("missing warm staged evidence must refuse");
        assert!(error.to_string().contains("no exact pre-admitted staged"));
        assert_eq!(context.selection.strategy, MemoryStrategy::Resident);
        assert_eq!(context.cache_state, MemoryCacheState::Cold);
    }

    fn chroma_probe_contract(
        adapter_identity: Option<&str>,
        adapter_bytes: u64,
        pid_bytes: u64,
    ) -> gen_core::MemoryProviderContract {
        let mut contract = gen_core::MemoryProviderContract::compatibility_default(
            "chroma1_base",
            gen_core::MemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: true,
                host_to_device_block_materialization: true,
                block_materialization: gen_core::MemoryWindowMaterialization::DeviceFormatTransfer,
            },
        );
        contract.strategies = MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| gen_core::MemoryStrategyCapability {
                strategy,
                support: if matches!(
                    strategy,
                    MemoryStrategy::Resident | MemoryStrategy::StagedResidency
                ) {
                    gen_core::MemoryStrategySupport::Implemented
                } else {
                    gen_core::MemoryStrategySupport::Missing
                },
                parameters: Default::default(),
            })
            .collect();
        contract.lifecycle = gen_core::MemoryLifecycleCapabilities {
            phases: vec![
                gen_core::MemoryPhase::Conditioning,
                gen_core::MemoryPhase::Denoise,
                gen_core::MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: false,
            attention_chunking: false,
            transformer_window_materialization: false,
        };
        let mut resident_components = Vec::new();
        if let Some(identity) = adapter_identity {
            resident_components.push(gen_core::MemoryResidentComponent {
                id: identity.to_owned(),
                kind: gen_core::MemoryComponentKind::AdapterStack,
                resident_bytes: adapter_bytes,
                bounded_by: Some(MemoryStrategy::StagedResidency),
                residency: gen_core::MemoryComponentResidency::WholeRender,
            });
        }
        if pid_bytes > 0 {
            resident_components.push(gen_core::MemoryResidentComponent {
                id: "chroma.pid.flux-student-and-gemma".to_owned(),
                kind: gen_core::MemoryComponentKind::AdapterStack,
                resident_bytes: pid_bytes,
                bounded_by: Some(MemoryStrategy::StagedResidency),
                residency: gen_core::MemoryComponentResidency::WholeRender,
            });
        }
        contract.formula = gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: contract.lifecycle.phases.clone(),
            variables: vec![
                gen_core::MemoryFormulaVariable::AssetBytes,
                gen_core::MemoryFormulaVariable::PixelCount,
                gen_core::MemoryFormulaVariable::BatchCount,
                gen_core::MemoryFormulaVariable::ConditioningTokenCount,
                gen_core::MemoryFormulaVariable::OverlayBytes,
            ],
            resident_components,
        };
        contract.asset_facts = gen_core::MemoryAssetFacts {
            conditioning_bytes: gib(6),
            transformer_bytes: gib(10),
            decoder_bytes: gib(2),
            base_bytes: gib(18),
            overlay_bytes: adapter_bytes.saturating_add(pid_bytes),
        };
        contract
    }

    fn sana_probe_contract(provider: &str) -> gen_core::MemoryProviderContract {
        let mut contract = chroma_probe_contract(None, 0, 0);
        contract.provider_id = provider.to_owned();
        contract.load_shape = gen_core::LoadShape::DeferredMaterialization;
        contract.strategies = MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| gen_core::MemoryStrategyCapability {
                strategy,
                support: gen_core::MemoryStrategySupport::Implemented,
                parameters: match strategy {
                    MemoryStrategy::BoundedDecode => gen_core::MemoryParameterRanges {
                        decode_tile_edges: vec![512],
                        decode_overlaps: vec![128],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedAttention => gen_core::MemoryParameterRanges {
                        attention_chunk_sizes: vec![1_048_576],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedTransformerResidency => {
                        gen_core::MemoryParameterRanges {
                            transformer_window_sizes: vec![1],
                            transformer_window_components: vec![
                                gen_core::TransformerComponent::Dit,
                            ],
                            ..Default::default()
                        }
                    }
                    _ => Default::default(),
                },
            })
            .collect();
        contract.additional_prerequisites = [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ]
        .into_iter()
        .map(|strategy| {
            (
                strategy,
                gen_core::MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            )
        })
        .collect();
        contract.lifecycle.decode_tiling = true;
        contract.lifecycle.attention_chunking = true;
        contract.lifecycle.transformer_window_materialization = true;
        contract.calibration = Some(gen_core::MemoryCalibrationIdentity::new(
            if provider == "sana_1600m" {
                "sana-candle-dense-base-full-ladder-v1"
            } else {
                "sana-candle-dense-sprint-full-ladder-v1"
            },
            gen_core::LoadShape::DeferredMaterialization,
        ));
        contract.formula = gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: contract.lifecycle.phases.clone(),
            variables: vec![
                gen_core::MemoryFormulaVariable::AssetBytes,
                gen_core::MemoryFormulaVariable::PixelCount,
                gen_core::MemoryFormulaVariable::BatchCount,
                gen_core::MemoryFormulaVariable::ConditioningTokenCount,
                gen_core::MemoryFormulaVariable::DecodeTileArea,
                gen_core::MemoryFormulaVariable::AttentionChunkSize,
                gen_core::MemoryFormulaVariable::TransformerWindowSize,
            ],
            resident_components: vec![gen_core::MemoryResidentComponent {
                id: format!("{SANA_PHYSICAL_RECEIPT_PREFIX}{}", "a".repeat(64)),
                kind: gen_core::MemoryComponentKind::TransformerSubStack(
                    gen_core::TransformerComponent::Dit,
                ),
                resident_bytes: gib(10),
                bounded_by: Some(MemoryStrategy::StagedResidency),
                residency: gen_core::MemoryComponentResidency::WholeRender,
            }],
        };
        contract
    }

    fn sd35_probe_contract(
        provider: &str,
        adapter_identity: Option<&str>,
    ) -> gen_core::MemoryProviderContract {
        let adapter_bytes = adapter_identity.map_or(0, |_| gib(1));
        let mut contract = chroma_probe_contract(None, 0, 0);
        contract.provider_id = provider.to_owned();
        contract.strategies = MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| gen_core::MemoryStrategyCapability {
                strategy,
                support: if matches!(
                    strategy,
                    MemoryStrategy::Resident | MemoryStrategy::StagedResidency
                ) {
                    gen_core::MemoryStrategySupport::Implemented
                } else {
                    gen_core::MemoryStrategySupport::Missing
                },
                parameters: Default::default(),
            })
            .collect();
        let mut resident_components = vec![gen_core::MemoryResidentComponent {
            id: format!("{SD35_PHYSICAL_RECEIPT_PREFIX}{}", "a".repeat(64)),
            kind: gen_core::MemoryComponentKind::TransformerSubStack(
                gen_core::TransformerComponent::Dit,
            ),
            resident_bytes: gib(10),
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: gen_core::MemoryComponentResidency::WholeRender,
        }];
        if let Some(identity) = adapter_identity {
            resident_components.push(gen_core::MemoryResidentComponent {
                id: identity.to_owned(),
                kind: gen_core::MemoryComponentKind::AdapterStack,
                resident_bytes: adapter_bytes,
                bounded_by: Some(MemoryStrategy::StagedResidency),
                residency: gen_core::MemoryComponentResidency::WholeRender,
            });
        }
        contract.formula = gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: contract.lifecycle.phases.clone(),
            variables: vec![
                gen_core::MemoryFormulaVariable::AssetBytes,
                gen_core::MemoryFormulaVariable::PixelCount,
                gen_core::MemoryFormulaVariable::BatchCount,
                gen_core::MemoryFormulaVariable::ConditioningTokenCount,
                gen_core::MemoryFormulaVariable::OverlayBytes,
            ],
            resident_components,
        };
        contract.asset_facts = gen_core::MemoryAssetFacts {
            conditioning_bytes: gib(6),
            transformer_bytes: gib(10),
            decoder_bytes: gib(2),
            base_bytes: gib(18),
            overlay_bytes: adapter_bytes,
        };
        contract
    }

    fn chroma_structural_manifest() -> JsonObject<String, Value> {
        json!({
            "candle": {
                "vramGbByTier": { "q4": 18.0 },
                "sequentialPeakGb": { "q4": 11.0 },
                "measured": false
            }
        })
        .as_object()
        .expect("Chroma structural manifest")
        .clone()
    }

    /// Physical bytes of one Kolors turnkey tier, as (conditioning, unet/DiT, decoder) GiB.
    ///
    /// The three tiers are physically different tensor sets — a q4 UNet is not a bf16 UNet with a
    /// smaller label — so a fixture that returns identical facts for all three makes the tier axis
    /// of `kolors_bespoke_staged_surface_covers_tiers_geometries_modes_pid_and_cache_states`
    /// inert: every tier iteration prices the same bytes and no tier-keyed mutation can be killed.
    /// The ChatGLM text encoder is f32 in every tier (it is not quantized), so only the UNet and
    /// the VAE move.
    fn kolors_tier_gib(tier: &str) -> (u64, u64, u64) {
        match tier {
            "q4" => (6, 5, 1),
            "q8" => (6, 10, 2),
            "bf16" => (6, 20, 4),
            other => unreachable!("Kolors probe fixture has no physical tier {other}"),
        }
    }

    fn kolors_bespoke_probe_contract(
        provider: &str,
        tier: &str,
        use_pid: bool,
    ) -> gen_core::MemoryProviderContract {
        let (conditioning_gib, transformer_gib, decoder_gib) = kolors_tier_gib(tier);
        let mut contract = chroma_probe_contract(None, 0, 0);
        contract.provider_id = provider.to_owned();
        contract.calibration = Some(gen_core::MemoryCalibrationIdentity::new(
            "kolors-candle-staged-chatglm-unet-f32-vae-v1",
            gen_core::LoadShape::EagerMaterialization,
        ));
        let overlay_prefix = if provider == "candle_kolors_ipadapter" {
            KOLORS_IP_RECEIPT_PREFIX
        } else {
            KOLORS_CONTROL_RECEIPT_PREFIX
        };
        let overlay_kind = if provider == "candle_kolors_ipadapter" {
            gen_core::MemoryComponentKind::IpAdapter
        } else {
            gen_core::MemoryComponentKind::ControlBranch
        };
        let mut components = vec![
            gen_core::MemoryResidentComponent {
                id: format!("{KOLORS_PHYSICAL_RECEIPT_PREFIX}{}", "a".repeat(64)),
                kind: gen_core::MemoryComponentKind::TransformerSubStack(
                    gen_core::TransformerComponent::Dit,
                ),
                resident_bytes: gib(transformer_gib),
                bounded_by: Some(MemoryStrategy::StagedResidency),
                residency: gen_core::MemoryComponentResidency::WholeRender,
            },
            gen_core::MemoryResidentComponent {
                id: format!("{overlay_prefix}{}", "b".repeat(64)),
                kind: overlay_kind,
                resident_bytes: gib(2),
                bounded_by: Some(MemoryStrategy::StagedResidency),
                residency: gen_core::MemoryComponentResidency::WholeRender,
            },
        ];
        if use_pid {
            components.push(gen_core::MemoryResidentComponent {
                id: format!("{KOLORS_PID_RECEIPT_PREFIX}{}", "c".repeat(64)),
                kind: gen_core::MemoryComponentKind::AdapterStack,
                resident_bytes: gib(3),
                bounded_by: Some(MemoryStrategy::StagedResidency),
                residency: gen_core::MemoryComponentResidency::WholeRender,
            });
        }
        contract.formula = gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: contract.lifecycle.phases.clone(),
            variables: vec![
                gen_core::MemoryFormulaVariable::AssetBytes,
                gen_core::MemoryFormulaVariable::PixelCount,
                gen_core::MemoryFormulaVariable::BatchCount,
                gen_core::MemoryFormulaVariable::ConditioningTokenCount,
                gen_core::MemoryFormulaVariable::OverlayBytes,
            ],
            resident_components: components,
        };
        contract.asset_facts = gen_core::MemoryAssetFacts {
            conditioning_bytes: gib(conditioning_gib),
            transformer_bytes: gib(transformer_gib),
            decoder_bytes: gib(decoder_gib),
            base_bytes: gib(conditioning_gib + transformer_gib + decoder_gib),
            overlay_bytes: gib(if use_pid { 5 } else { 2 }),
        };
        contract
    }

    fn kolors_registered_probe_contract(
        tier: &str,
        with_adapter: bool,
        use_pid: bool,
    ) -> gen_core::MemoryProviderContract {
        let mut contract = kolors_bespoke_probe_contract("candle_kolors_control", tier, use_pid);
        contract.provider_id = "kolors".to_owned();
        let mut components = contract.resident_components().to_vec();
        components.retain(|component| !component.id.starts_with(KOLORS_CONTROL_RECEIPT_PREFIX));
        if with_adapter {
            components.push(gen_core::MemoryResidentComponent {
                id: format!("{KOLORS_ADAPTER_RECEIPT_PREFIX}{}", "d".repeat(64)),
                kind: gen_core::MemoryComponentKind::AdapterStack,
                resident_bytes: gib(1),
                bounded_by: Some(MemoryStrategy::StagedResidency),
                residency: gen_core::MemoryComponentResidency::WholeRender,
            });
        }
        contract.formula = gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: contract.lifecycle.phases.clone(),
            variables: vec![
                gen_core::MemoryFormulaVariable::AssetBytes,
                gen_core::MemoryFormulaVariable::PixelCount,
                gen_core::MemoryFormulaVariable::BatchCount,
                gen_core::MemoryFormulaVariable::ConditioningTokenCount,
                gen_core::MemoryFormulaVariable::OverlayBytes,
            ],
            resident_components: components,
        };
        contract.asset_facts.overlay_bytes = gib(u64::from(with_adapter) + 3 * u64::from(use_pid));
        contract
    }

    #[test]
    fn registered_kolors_is_receipt_priced_and_binds_exact_assembly_identity() {
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 30.0 },
                "sequentialPeakGb": { "q4": 15.0 },
                "measured": true
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("kolors-q4")));
        let contract = kolors_registered_probe_contract("q4", true, true);
        // The budget that separates the two rungs, recomputed from the fixture rather than typed.
        // Staged floor: `sequentialPeakGb` 15 + 2 `HEADROOM_GB` = 17, plus the receipt's exact
        // auxiliaries (adapter 1 + PiD 3) = 21 GiB, widened by the 4% candle estimate margin to
        // 21.84. Resident: the declared 30 GiB peak plus the same 4 GiB of auxiliaries = 34 GiB,
        // and it is CURRENT evidence, so it is never widened. The selector subtracts a 2 GiB
        // reserve, so a 25 GiB card leaves 23 GiB effective — above the staged floor and below the
        // resident peak. At the 22 GiB this test was written with, NEITHER rung fits and the
        // evaluation fails closed, which is what made the staged assertions unreachable.
        let free_gb = 25.0;
        let evaluation = evaluate_shared_image_inner(
            "kolors",
            "kolors",
            &spec,
            true,
            &manifest,
            "q4",
            "image_generation",
            Some("lora"),
            Some("lora"),
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            false,
            true,
            false,
            false,
            Some(VramBudget {
                free_gb,
                total_gb: 48.0,
            }),
            reserve_for(Some(VramBudget {
                free_gb,
                total_gb: 48.0,
            })),
            Some(30.0),
            0,
            MemoryCacheState::Cold,
            None,
            Some(contract.clone()),
            None,
            None,
        )
        .unwrap()
        .expect("registered Kolors staged selection");
        assert_eq!(
            evaluation.context.evidence_revision,
            KOLORS_REQUEST_EVIDENCE_REVISION
        );
        let assembly = evaluation.context.overlay.unwrap();
        assert!(assembly.contains(KOLORS_PHYSICAL_RECEIPT_PREFIX));
        assert!(assembly.contains(KOLORS_ADAPTER_RECEIPT_PREFIX));
        assert!(assembly.contains(KOLORS_PID_RECEIPT_PREFIX));

        let crossed_pid = evaluate_shared_image_inner(
            "kolors",
            "kolors",
            &spec,
            true,
            &manifest,
            "q4",
            "image_generation",
            None,
            None,
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            false,
            true,
            false,
            false,
            Some(VramBudget {
                free_gb,
                total_gb: 48.0,
            }),
            reserve_for(Some(VramBudget {
                free_gb,
                total_gb: 48.0,
            })),
            Some(30.0),
            0,
            MemoryCacheState::Cold,
            None,
            Some(kolors_registered_probe_contract("q4", false, false)),
            None,
            None,
        );
        // Message-checked, not merely `is_err`: at a budget where nothing fits, EVERY arm of this
        // test errors, and a bare `is_err` would pass without the crossing ever being detected.
        assert!(crossed_pid
            .err()
            .expect("a PiD request against a PiD-free receipt must fail")
            .to_string()
            .contains("crossed Kolors physical receipts"));

        let crossed_adapters = evaluate_shared_image_inner(
            "kolors",
            "kolors",
            &spec,
            true,
            &manifest,
            "q4",
            "image_generation",
            Some("lora"),
            Some("lora"),
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            false,
            false,
            false,
            false,
            Some(VramBudget {
                free_gb,
                total_gb: 48.0,
            }),
            reserve_for(Some(VramBudget {
                free_gb,
                total_gb: 48.0,
            })),
            Some(30.0),
            0,
            MemoryCacheState::Cold,
            None,
            Some(kolors_registered_probe_contract("q4", false, false)),
            None,
            None,
        );
        assert!(crossed_adapters
            .err()
            .expect("a declared public adapter overlay with no adapter receipt must fail")
            .to_string()
            .contains("public ordered-adapter identity crossed its exact provider receipt"));
    }

    #[test]
    fn kolors_bespoke_selector_is_request_authoritative_and_route_exact() {
        let manifest = json!({
            "candle": { "vramGbByTier": { "q4": 30.0 }, "measured": true }
        })
        .as_object()
        .unwrap()
        .clone();
        let ip_spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("kolors-q4")))
            .with_ip_adapter(WeightsSource::Dir(PathBuf::from("kolors-ip")));
        let budget = Some(VramBudget {
            free_gb: 22.0,
            total_gb: 48.0,
        });
        let evaluation = evaluate_shared_bespoke_image(
            "candle_kolors_ipadapter",
            "kolors",
            &ip_spec,
            true,
            &manifest,
            "q4",
            "character_image",
            Some("identity"),
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 1,
            },
            true,
            false,
            false,
            budget,
            reserve_for(budget),
            Some(30.0),
            0,
            MemoryCacheState::Cold,
            kolors_bespoke_probe_contract("candle_kolors_ipadapter", "q4", false),
            KOLORS_REQUEST_EVIDENCE_REVISION,
        )
        .unwrap()
        .expect("exact staged candidate");
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::StagedResidency
        );
        assert_eq!(
            evaluation.context.evidence_revision,
            KOLORS_REQUEST_EVIDENCE_REVISION
        );
        assert!(evaluation
            .context
            .overlay
            .as_deref()
            .unwrap()
            .contains(KOLORS_IP_RECEIPT_PREFIX));

        let control_spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("kolors-q4")))
            .with_control(WeightsSource::File(PathBuf::from(
                "kolors-control.safetensors",
            )));
        let crossed = evaluate_shared_bespoke_image(
            "candle_kolors_control",
            "kolors",
            &control_spec,
            true,
            &manifest,
            "q4",
            "text_to_image",
            Some("control"),
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            false,
            false,
            false,
            budget,
            reserve_for(budget),
            Some(30.0),
            0,
            MemoryCacheState::Cold,
            kolors_bespoke_probe_contract("candle_kolors_ipadapter", "q4", false),
            KOLORS_REQUEST_EVIDENCE_REVISION,
        );
        assert!(
            crossed.is_err(),
            "IP evidence must not authorize ControlNet"
        );
        assert!(evaluate_shared_bespoke_image(
            "candle_kolors_ipadapter",
            "kolors",
            &ip_spec,
            true,
            &manifest,
            "q4",
            "character_image",
            Some("identity"),
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 1,
            },
            true,
            false,
            false,
            budget,
            reserve_for(budget),
            Some(30.0),
            0,
            MemoryCacheState::Cold,
            kolors_bespoke_probe_contract("candle_kolors_ipadapter", "q4", false),
            "crossed-kolors-revision",
        )
        .is_err());
    }

    #[test]
    fn kolors_bespoke_staged_surface_covers_tiers_geometries_modes_pid_and_cache_states() {
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 16.0, "q8": 22.0, "bf16": 34.0 },
                "measured": true
            }
        })
        .as_object()
        .unwrap()
        .clone();
        // The declared resident row and the provider receipt describe the SAME physical tier. Both
        // move together, so a tier is a real axis of this coverage rather than a label on one set
        // of bytes: see `kolors_tier_gib`.
        let tier_resident_gb = |tier: &str| {
            let (conditioning, transformer, decoder) = kolors_tier_gib(tier);
            (conditioning + transformer + decoder) as f64 + 4.0
        };
        // The budget that makes STAGED the selection, recomputed PER ARM from that arm's own
        // receipt rather than fixed for the whole sweep. `select_strategy` walks
        // `MemoryStrategy::ALL` in order and returns the FIRST rung whose admitted peak fits;
        // Resident is first. A single roomy card (the 40 GiB this test was written with) clears
        // every tier's resident peak, so every cell below selected Resident and no staged surface
        // was covered at all.
        //
        // This manifest declares no `sequentialPeakGb`, so the staged floor is the receipt's
        // largest single phase + `HEADROOM_GB`, plus the exact auxiliaries, widened by the 4%
        // candle estimate margin (an estimate floor is never current evidence). The resident peak
        // is the declared row — clamped up to the receipt's own base + headroom — plus the same
        // auxiliaries, and IS current evidence, so it is never widened. The selector subtracts a
        // 2 GiB reserve from the card before comparing. The assertion inside keeps the arm honest:
        // if a fixture change ever lifts the staged floor past the resident peak, this fails loudly
        // instead of quietly grading Resident.
        let headroom_bytes = (crate::vram_gate::HEADROOM_GB * BYTES_PER_GIB).ceil() as u64;
        let staged_budget = |contract: &gen_core::MemoryProviderContract, tier: &str| {
            let facts = contract.asset_facts;
            let peak_gb = |base: u64| {
                contract
                    .predicted_peak_from_base(base)
                    .predicted_peak_bytes() as f64
                    / BYTES_PER_GIB
            };
            let staged_gb = peak_gb(
                facts
                    .conditioning_bytes
                    .max(facts.transformer_bytes)
                    .max(facts.decoder_bytes)
                    .saturating_add(headroom_bytes),
            );
            let resident_gb = peak_gb(
                ((tier_resident_gb(tier) * BYTES_PER_GIB).ceil() as u64)
                    .max(facts.base_bytes.saturating_add(headroom_bytes)),
            );
            let free_gb = staged_gb * (1.0 + crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD)
                + crate::vram_gate::HEADROOM_GB
                + 0.5;
            assert!(
                free_gb - crate::vram_gate::HEADROOM_GB < resident_gb,
                "{tier}: a budget of {free_gb} also clears the resident peak {resident_gb}; the \
                 staged assertion below would be vacuous"
            );
            Some(VramBudget {
                free_gb,
                total_gb: 96.0,
            })
        };
        let geometries = [(768, 768), (1024, 1024), (1280, 768), (768, 1280)];
        let mut ip_peaks_by_tier: Vec<(&str, u64)> = Vec::new();
        for tier in ["q4", "q8", "bf16"] {
            let base_spec =
                LoadSpec::new(WeightsSource::Dir(PathBuf::from(format!("kolors-{tier}"))));
            let ip_spec = base_spec
                .clone()
                .with_ip_adapter(WeightsSource::Dir(PathBuf::from("kolors-ip")));
            let control_spec = base_spec
                .clone()
                .with_control(WeightsSource::File(PathBuf::from(
                    "kolors-control.safetensors",
                )));
            let control_pid_spec = control_spec.clone().with_pid(
                WeightsSource::File(PathBuf::from("pid.safetensors")),
                WeightsSource::Dir(PathBuf::from("gemma")),
            );
            for (width, height) in geometries {
                for cache_state in [MemoryCacheState::Cold, MemoryCacheState::Warm] {
                    let ip_contract =
                        kolors_bespoke_probe_contract("candle_kolors_ipadapter", tier, false);
                    let ip = evaluate_shared_bespoke_image(
                        "candle_kolors_ipadapter",
                        "kolors",
                        &ip_spec,
                        true,
                        &manifest,
                        tier,
                        "character_image",
                        Some("identity"),
                        MemoryGeometry {
                            width,
                            height,
                            batch: 1,
                            frames: 1,
                            reference_count: 1,
                        },
                        true,
                        false,
                        false,
                        staged_budget(&ip_contract, tier),
                        reserve_for(staged_budget(&ip_contract, tier)),
                        Some(tier_resident_gb(tier)),
                        0,
                        cache_state,
                        ip_contract.clone(),
                        KOLORS_REQUEST_EVIDENCE_REVISION,
                    )
                    .unwrap()
                    .expect("IP staged candidate");
                    assert_eq!(
                        ip.context.selection.strategy,
                        MemoryStrategy::StagedResidency
                    );
                    if (width, height) == (1024, 1024) && cache_state == MemoryCacheState::Cold {
                        ip_peaks_by_tier.push((tier, ip.context.predicted_peak_bytes));
                    }

                    for (mode, use_pid) in [
                        ("text_to_image", false),
                        ("style_variations", false),
                        ("character_image", false),
                        ("text_to_image", true),
                    ] {
                        let control_contract =
                            kolors_bespoke_probe_contract("candle_kolors_control", tier, use_pid);
                        let control = evaluate_shared_bespoke_image(
                            "candle_kolors_control",
                            "kolors",
                            if use_pid {
                                &control_pid_spec
                            } else {
                                &control_spec
                            },
                            true,
                            &manifest,
                            tier,
                            mode,
                            Some("control"),
                            MemoryGeometry {
                                width,
                                height,
                                batch: 1,
                                frames: 1,
                                reference_count: 0,
                            },
                            false,
                            use_pid,
                            false,
                            staged_budget(&control_contract, tier),
                            reserve_for(staged_budget(&control_contract, tier)),
                            Some(tier_resident_gb(tier)),
                            0,
                            cache_state,
                            control_contract.clone(),
                            KOLORS_REQUEST_EVIDENCE_REVISION,
                        )
                        .unwrap()
                        .expect("Control staged candidate");
                        assert_eq!(
                            control.context.selection.strategy,
                            MemoryStrategy::StagedResidency
                        );
                    }
                }
            }
        }
        // The control arm for every staged assertion above: the SAME cell on a roomy card selects
        // Resident. Without it, a staged verdict could be an artifact of the fixture rather than
        // of the budget, and a change that made staged unconditionally selectable would pass.
        let roomy = evaluate_shared_bespoke_image(
            "candle_kolors_ipadapter",
            "kolors",
            &LoadSpec::new(WeightsSource::Dir(PathBuf::from("kolors-q4")))
                .with_ip_adapter(WeightsSource::Dir(PathBuf::from("kolors-ip"))),
            true,
            &manifest,
            "q4",
            "character_image",
            Some("identity"),
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 1,
            },
            true,
            false,
            false,
            Some(VramBudget {
                free_gb: 96.0,
                total_gb: 96.0,
            }),
            reserve_for(Some(VramBudget {
                free_gb: 96.0,
                total_gb: 96.0,
            })),
            Some(tier_resident_gb("q4")),
            0,
            MemoryCacheState::Cold,
            kolors_bespoke_probe_contract("candle_kolors_ipadapter", "q4", false),
            KOLORS_REQUEST_EVIDENCE_REVISION,
        )
        .unwrap()
        .expect("a roomy card still admits");
        assert_eq!(
            roomy.context.selection.strategy,
            MemoryStrategy::Resident,
            "the staged verdicts above must be the BUDGET's doing, not the fixture's"
        );

        // The tier axis has to REACH the admitted number. Before sc-20799 the probe contract
        // returned identical bytes for q4, q8 and bf16 and the declared row was 63.7 for all
        // three, so every tier iteration above admitted the same peak and no tier-keyed mutation
        // could be killed. Assert the SHAPE — three tiers, strictly ordered by their physical
        // size — never a byte literal, which would freeze the fixture into a golden.
        assert_eq!(
            ip_peaks_by_tier
                .iter()
                .map(|(tier, _)| *tier)
                .collect::<Vec<_>>(),
            vec!["q4", "q8", "bf16"]
        );
        assert!(
            ip_peaks_by_tier
                .windows(2)
                .all(|pair| pair[0].1 < pair[1].1),
            "staged admission must price each Kolors tier's own bytes, got {ip_peaks_by_tier:?}"
        );
    }

    fn ideogram_probe_contract(
        provider: &str,
        user_identity: Option<&str>,
        include_turbo: bool,
        include_pid: bool,
    ) -> gen_core::MemoryProviderContract {
        let mut contract = chroma_probe_contract(None, 0, 0);
        contract.provider_id = provider.to_owned();
        contract.strategies = MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| gen_core::MemoryStrategyCapability {
                strategy,
                support: gen_core::MemoryStrategySupport::Implemented,
                parameters: match strategy {
                    MemoryStrategy::BoundedDecode => gen_core::MemoryParameterRanges {
                        decode_tile_edges: vec![512],
                        decode_overlaps: vec![64],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedAttention => gen_core::MemoryParameterRanges {
                        attention_chunk_sizes: vec![64 * 1024 * 1024],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedTransformerResidency => {
                        gen_core::MemoryParameterRanges {
                            transformer_window_sizes: vec![4],
                            transformer_window_components: vec![
                                gen_core::TransformerComponent::Dit,
                            ],
                            ..Default::default()
                        }
                    }
                    _ => Default::default(),
                },
            })
            .collect();
        contract.lifecycle.decode_tiling = true;
        contract.lifecycle.attention_chunking = true;
        contract.lifecycle.transformer_window_materialization = true;
        contract.additional_prerequisites = [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ]
        .into_iter()
        .map(|strategy| {
            (
                strategy,
                gen_core::MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            )
        })
        .collect();
        let mut components = vec![gen_core::MemoryResidentComponent {
            id: format!("{IDEOGRAM_PHYSICAL_RECEIPT_PREFIX}{}", "e".repeat(64)),
            kind: gen_core::MemoryComponentKind::TransformerSubStack(
                gen_core::TransformerComponent::Dit,
            ),
            resident_bytes: contract.asset_facts.transformer_bytes,
            bounded_by: Some(MemoryStrategy::BoundedTransformerResidency),
            residency: gen_core::MemoryComponentResidency::WholeRender,
        }];
        if let Some(identity) = user_identity {
            components.push(gen_core::MemoryResidentComponent {
                id: identity.to_owned(),
                kind: gen_core::MemoryComponentKind::AdapterStack,
                resident_bytes: gib(2),
                bounded_by: Some(MemoryStrategy::StagedResidency),
                residency: gen_core::MemoryComponentResidency::WholeRender,
            });
        }
        if include_turbo {
            components.push(gen_core::MemoryResidentComponent {
                id: format!("{IDEOGRAM_TURBO_ADAPTER_PREFIX}{}", "b".repeat(64)),
                kind: gen_core::MemoryComponentKind::AdapterStack,
                resident_bytes: gib(1),
                bounded_by: Some(MemoryStrategy::StagedResidency),
                residency: gen_core::MemoryComponentResidency::WholeRender,
            });
        }
        if include_pid {
            components.push(gen_core::MemoryResidentComponent {
                id: format!("{IDEOGRAM_PID_PREFIX}{}", "c".repeat(64)),
                kind: gen_core::MemoryComponentKind::AdapterStack,
                resident_bytes: gib(3),
                bounded_by: Some(MemoryStrategy::StagedResidency),
                residency: gen_core::MemoryComponentResidency::WholeRender,
            });
        }
        let overlay_bytes = components
            .iter()
            .filter(|component| component.kind.is_auxiliary())
            .map(|component| component.resident_bytes)
            .sum();
        contract.formula = gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: contract.lifecycle.phases.clone(),
            variables: vec![
                gen_core::MemoryFormulaVariable::AssetBytes,
                gen_core::MemoryFormulaVariable::PixelCount,
                gen_core::MemoryFormulaVariable::BatchCount,
                gen_core::MemoryFormulaVariable::ConditioningTokenCount,
                gen_core::MemoryFormulaVariable::OverlayBytes,
                gen_core::MemoryFormulaVariable::DecodeTileArea,
                gen_core::MemoryFormulaVariable::AttentionChunkSize,
                gen_core::MemoryFormulaVariable::TransformerWindowSize,
            ],
            resident_components: components,
        };
        contract.asset_facts.overlay_bytes = overlay_bytes;
        contract
    }

    #[test]
    fn chroma_structural_rows_are_complete_differentiated_and_unmeasured() {
        let source = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, source)| *source)
            .expect("embedded model manifest");
        let stripped = sceneworks_core::jsonc::strip_jsonc_comments(source);
        let root: Value = serde_json::from_str(&stripped).expect("model manifest parses");
        for model_id in ["chroma1_hd", "chroma1_base", "chroma1_flash"] {
            let model = root["models"]
                .as_array()
                .expect("models array")
                .iter()
                .find(|model| model["id"] == model_id)
                .expect("Chroma model");
            let candle = &model["candle"];
            assert_eq!(candle["measured"], false);
            for tier in ["q4", "q8", "bf16"] {
                let resident = candle["vramGbByTier"][tier]
                    .as_f64()
                    .expect("resident structural row");
                let staged = candle["sequentialPeakGb"][tier]
                    .as_f64()
                    .expect("staged structural row");
                assert!(staged > 0.0 && staged < resident, "{model_id}/{tier}");
            }
            assert!(candle["vramGbByTier"]["q4"] != candle["vramGbByTier"]["q8"]);
            assert!(candle["vramGbByTier"]["q8"] != candle["vramGbByTier"]["bf16"]);
        }
    }

    #[test]
    fn chroma_selects_receipt_priced_staged_and_refuses_crossed_or_no_fit_requests() {
        let identity = format!("{CHROMA_ADAPTER_OVERLAY_PREFIX}{}", "a".repeat(64));
        let contract = chroma_probe_contract(Some(&identity), gib(2), gib(3));
        let manifest = chroma_structural_manifest();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("sealed-chroma-q4")))
            .with_resolved_route("chroma1_base");
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let evaluate = |free_gb: f64, provider_overlay: Option<&str>, contract| {
            evaluate_shared_image_inner(
                "chroma1_base",
                "chroma1_base",
                &spec,
                true,
                &manifest,
                "q4",
                "text_to_image",
                Some("lora"),
                provider_overlay,
                geometry,
                false,
                true,
                false,
                false,
                Some(VramBudget {
                    free_gb,
                    total_gb: 96.0,
                }),
                reserve_for(Some(VramBudget {
                    free_gb,
                    total_gb: 96.0,
                })),
                crate::vram_gate::predicted_peak_gb(&manifest, "q4"),
                gib(80),
                MemoryCacheState::Cold,
                None,
                Some(contract),
                Some(CHROMA_REQUEST_EVIDENCE_REVISION),
                None,
            )
        };

        // The raw staged estimate is 11 + 2 headroom + 2 adapter + 3 PiD = 18 GiB, widened by the
        // candle recapture spread. That floor CARRIES its pad, so it pays no reserve on the budget
        // side (`memory_strategy::ReserveCharge`, sc-22664 D1/D4): the admission threshold is the
        // widened floor itself, against the unreserved pool. 22 GiB free fits staged while the
        // 25 GiB resident receipt does not. The deliberately bogus 80 GiB generic adapter input
        // must not affect this receipt-priced result.
        let staged = evaluate(22.0, Some("lora"), contract.clone())
            .expect("exact Chroma evaluation")
            .expect("staged strategy fits");
        assert_eq!(
            staged.context.selection.strategy,
            MemoryStrategy::StagedResidency
        );
        assert_eq!(staged.context.overlay.as_deref(), Some(identity.as_str()));
        assert_eq!(staged.context.predicted_peak_bytes, gib(18));
        assert_eq!(
            staged.context.optimization_authority,
            gen_core::MemoryOptimizationAuthority::Estimated
        );
        let staged_threshold_gb = staged.admitted.needed_gb;
        assert!(
            (staged_threshold_gb
                - 18.0 * (1.0 + crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD))
                .abs()
                < 1e-6,
            "{staged_threshold_gb}"
        );
        assert_eq!(
            staged.admitted.available_gb, 22.0,
            "a pad-carrying structural floor compares against the UNRESERVED pool"
        );
        assert!(staged.admitted.reserve_gb > 0.0);

        // The single charge, straddled: the widened floor fits one hundredth above itself and
        // fails closed one hundredth below. MUTATION: charging the reserve against this floor as
        // well (`ReserveCharge::EveryCandidate` in `evaluate_shared_image_inner`) refuses the
        // upper arm.
        let just_fits = evaluate(staged_threshold_gb + 0.01, Some("lora"), contract.clone())
            .expect("exact Chroma evaluation")
            .expect("the widened floor fits at its own threshold");
        assert_eq!(
            just_fits.context.selection.strategy,
            MemoryStrategy::StagedResidency
        );
        let no_fit = evaluate(staged_threshold_gb - 0.01, Some("lora"), contract.clone())
            .err()
            .expect("a budget below the widened exact staged floor must fail closed");
        assert!(no_fit.to_string().contains("no exact resident or staged"));

        let crossed = evaluate(22.0, Some("lora"), chroma_probe_contract(None, 0, gib(3)))
            .err()
            .expect("a public adapter cell without its exact provider receipt must fail");
        assert!(crossed
            .to_string()
            .contains("singular exact provider receipt"));

        let stale = evaluate(22.0, Some("lora"), contract.clone())
            .expect("the declared public cell resolves the receipt internally")
            .expect("staged still fits");
        assert_ne!(stale.context.overlay.as_deref(), Some("lora"));

        let mut crossed_facts = contract;
        crossed_facts.asset_facts.overlay_bytes = gib(4);
        let crossed = evaluate(22.0, Some("lora"), crossed_facts)
            .err()
            .expect("typed adapter/PiD bytes must equal the aggregate footprint");
        assert!(crossed
            .to_string()
            .contains("materialized Chroma asset facts"));
    }

    #[test]
    fn sana_dense_receipts_admit_t2i_i2i_and_hires_but_refuse_crossed_or_no_fit() {
        let manifest = json!({ "candle": {} })
            .as_object()
            .expect("SANA structural manifest")
            .clone();
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        for provider in ["sana_1600m", "sana_sprint_1600m"] {
            let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("sealed-sana-dense")))
                .with_resolved_route(provider);
            let evaluate = |mode: &str,
                            reference_count: u32,
                            multipass: bool,
                            request_has_phases: bool,
                            free_gb: f64,
                            contract| {
                evaluate_shared_image_inner(
                    provider,
                    provider,
                    &spec,
                    true,
                    &manifest,
                    "bf16",
                    mode,
                    None,
                    None,
                    MemoryGeometry {
                        reference_count,
                        ..geometry
                    },
                    reference_count == 1,
                    false,
                    multipass,
                    request_has_phases,
                    Some(VramBudget {
                        free_gb,
                        total_gb: 48.0,
                    }),
                    reserve_for(Some(VramBudget {
                        free_gb,
                        total_gb: 48.0,
                    })),
                    None,
                    0,
                    MemoryCacheState::Cold,
                    None,
                    Some(contract),
                    Some(SANA_REQUEST_EVIDENCE_REVISION),
                    None,
                )
            };

            for (mode, references, multipass, request_has_phases) in [
                ("text_to_image", 0, false, false),
                ("image_to_image", 1, false, false),
                ("image_to_image", 1, true, true),
            ] {
                let evaluation = evaluate(
                    mode,
                    references,
                    multipass,
                    request_has_phases,
                    15.0,
                    sana_probe_contract(provider),
                )
                .expect("exact SANA receipt evaluation")
                .expect("staged phase floor fits");
                assert_eq!(
                    evaluation.context.selection.strategy,
                    MemoryStrategy::StagedResidency
                );
                assert_eq!(evaluation.context.geometry.reference_count, references);
                assert_eq!(evaluation.context.has_phases, request_has_phases);
                assert_eq!(
                    evaluation.context.evidence_revision,
                    SANA_REQUEST_EVIDENCE_REVISION
                );
            }

            // The receipt-derived staged floor carries its pad and pays no reserve on the budget
            // side (`memory_strategy::ReserveCharge`, sc-22664 D1/D4): its widened peak is the
            // threshold, against the unreserved pool. MUTATION: charging the reserve against the
            // floor as well refuses the `just_fits` arm.
            let staged = evaluate(
                "text_to_image",
                0,
                false,
                false,
                15.0,
                sana_probe_contract(provider),
            )
            .expect("exact SANA receipt evaluation")
            .expect("staged phase floor fits");
            let threshold_gb = staged.admitted.needed_gb;
            assert_eq!(
                staged.admitted.available_gb, 15.0,
                "a pad-carrying structural floor compares against the UNRESERVED pool"
            );
            let just_fits = evaluate(
                "text_to_image",
                0,
                false,
                false,
                threshold_gb + 0.01,
                sana_probe_contract(provider),
            )
            .expect("exact SANA receipt evaluation")
            .expect("the widened floor fits at its own threshold");
            assert_eq!(
                just_fits.context.selection.strategy,
                MemoryStrategy::StagedResidency
            );
            let no_fit = match evaluate(
                "text_to_image",
                0,
                false,
                false,
                threshold_gb - 0.01,
                sana_probe_contract(provider),
            ) {
                Err(error) => error,
                Ok(_) => panic!("a budget below the receipt-derived staged floor must refuse"),
            };
            assert!(no_fit.to_string().contains("no exact resident or staged"));

            let mut crossed = sana_probe_contract(provider);
            crossed.calibration.as_mut().unwrap().fingerprint =
                "sana-candle-dense-crossed-full-ladder-v1".to_owned();
            let crossed = match evaluate("text_to_image", 0, false, false, 15.0, crossed) {
                Err(error) => error,
                Ok(_) => panic!("crossed Base/Sprint identity must refuse before selection"),
            };
            assert!(crossed.to_string().contains("crossed SANA physical facts"));
        }
    }

    #[test]
    fn sd35_exact_receipts_cover_every_route_tier_profile_and_mode() {
        let manifest = json!({ "candle": {} })
            .as_object()
            .expect("SD3.5 structural manifest")
            .clone();
        let base_geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let adapter_identity = format!("{SD35_ADAPTER_RECEIPT_PREFIX}{}", "b".repeat(64));

        for provider in ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"] {
            for tier in ["q4", "q8", "bf16"] {
                let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(format!(
                    "sealed-{provider}-{tier}"
                ))))
                .with_resolved_route(provider);
                for (public_overlay, provider_overlay, receipt) in [
                    (None, None, None),
                    (Some("lora"), Some("lora"), Some(adapter_identity.as_str())),
                ] {
                    for (mode, references, has_reference) in
                        [("text_to_image", 0, false), ("image_to_image", 1, true)]
                    {
                        let evaluate = |free_gb| {
                            evaluate_shared_image_inner(
                                provider,
                                provider,
                                &spec,
                                true,
                                &manifest,
                                tier,
                                mode,
                                public_overlay,
                                provider_overlay,
                                MemoryGeometry {
                                    reference_count: references,
                                    ..base_geometry
                                },
                                has_reference,
                                false,
                                false,
                                false,
                                Some(VramBudget {
                                    free_gb,
                                    total_gb: 48.0,
                                }),
                                reserve_for(Some(VramBudget {
                                    free_gb,
                                    total_gb: 48.0,
                                })),
                                None,
                                0,
                                MemoryCacheState::Cold,
                                None,
                                Some(sd35_probe_contract(provider, receipt)),
                                Some(SD35_REQUEST_EVIDENCE_REVISION),
                                None,
                            )
                        };
                        let staged = evaluate(16.0)
                            .expect("sealed SD3.5 receipt evaluates")
                            .expect("staged phase envelope fits");
                        assert_eq!(
                            staged.context.selection.strategy,
                            MemoryStrategy::StagedResidency,
                            "{provider}/{tier}/{public_overlay:?}/{mode}"
                        );
                        let expected_mode = match mode {
                            "text_to_image" => MemoryMode::TextToImage,
                            "image_to_image" => MemoryMode::ImageToImage,
                            _ => unreachable!("the fixture enumerates only SD3.5 public modes"),
                        };
                        assert_eq!(staged.context.mode, expected_mode);
                        assert_eq!(staged.context.has_reference, has_reference);
                        assert_eq!(staged.context.geometry.reference_count, references);
                        assert_eq!(
                            staged.context.overlay.as_deref(),
                            receipt,
                            "public lora labels must become exact ordered provider receipts"
                        );
                        assert_eq!(
                            staged.context.evidence_revision,
                            SD35_REQUEST_EVIDENCE_REVISION
                        );

                        let resident = evaluate(24.0)
                            .expect("sealed SD3.5 receipt evaluates")
                            .expect("resident envelope fits");
                        assert_eq!(
                            resident.context.selection.strategy,
                            MemoryStrategy::Resident
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sd35_receipts_refuse_crossed_physical_adapter_context_and_no_fit() {
        let manifest = json!({ "candle": {} })
            .as_object()
            .expect("SD3.5 structural manifest")
            .clone();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("sealed-sd35-q4")))
            .with_resolved_route("sd3_5_large");
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let adapter_identity = format!("{SD35_ADAPTER_RECEIPT_PREFIX}{}", "b".repeat(64));
        let evaluate = |free_gb: f64,
                        public_overlay: Option<&str>,
                        provider_overlay: Option<&str>,
                        contract| {
            evaluate_shared_image_inner(
                "sd3_5_large",
                "sd3_5_large",
                &spec,
                true,
                &manifest,
                "q4",
                "text_to_image",
                public_overlay,
                provider_overlay,
                geometry,
                false,
                false,
                false,
                false,
                Some(VramBudget {
                    free_gb,
                    total_gb: 48.0,
                }),
                reserve_for(Some(VramBudget {
                    free_gb,
                    total_gb: 48.0,
                })),
                None,
                0,
                MemoryCacheState::Cold,
                None,
                Some(contract),
                Some(SD35_REQUEST_EVIDENCE_REVISION),
                None,
            )
        };

        // The staged envelope carries its pad and pays no reserve on the budget side
        // (`memory_strategy::ReserveCharge`, sc-22664 D1/D4): its widened peak is the threshold.
        let staged = evaluate(16.0, None, None, sd35_probe_contract("sd3_5_large", None))
            .expect("sealed SD3.5 receipt evaluates")
            .expect("staged phase envelope fits");
        let staged_threshold_gb = staged.admitted.needed_gb;
        let no_fit = match evaluate(
            staged_threshold_gb - 0.01,
            None,
            None,
            sd35_probe_contract("sd3_5_large", None),
        ) {
            Err(error) => error,
            Ok(_) => panic!("a budget below the exact staged envelope must refuse"),
        };
        assert!(no_fit.to_string().contains("no exact resident or staged"));

        let crossed_route =
            match evaluate(24.0, None, None, sd35_probe_contract("sd3_5_medium", None)) {
                Err(error) => error,
                Ok(_) => panic!("crossed provider identity must refuse"),
            };
        assert!(crossed_route
            .to_string()
            .contains("incomplete or crossed SD3.5 physical facts"));

        let mut crossed_physical = sd35_probe_contract("sd3_5_large", None);
        if let gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
            resident_components,
            ..
        } = &mut crossed_physical.formula
        {
            resident_components[0].id = format!("other.physical.sha256:{}", "a".repeat(64));
        }
        let crossed_physical = match evaluate(24.0, None, None, crossed_physical) {
            Err(error) => error,
            Ok(_) => panic!("crossed physical receipt must refuse"),
        };
        assert!(crossed_physical
            .to_string()
            .contains("incomplete or crossed SD3.5 physical facts"));

        let missing_adapter = match evaluate(
            24.0,
            Some("lora"),
            Some("lora"),
            sd35_probe_contract("sd3_5_large", None),
        ) {
            Err(error) => error,
            Ok(_) => panic!("missing ordered adapter receipt must refuse"),
        };
        assert!(missing_adapter
            .to_string()
            .contains("lacks its exact provider receipt"));

        let crossed_plain = match evaluate(
            24.0,
            None,
            None,
            sd35_probe_contract("sd3_5_large", Some(&adapter_identity)),
        ) {
            Err(error) => error,
            Ok(_) => panic!("plain request crossed with adapter facts must refuse"),
        };
        assert!(crossed_plain
            .to_string()
            .contains("plain load crossed an adapter receipt"));

        let unsupported_overlay = match evaluate(
            24.0,
            Some("lokr"),
            Some("lokr"),
            sd35_probe_contract("sd3_5_large", Some(&adapter_identity)),
        ) {
            Err(error) => error,
            Ok(_) => panic!("unsupported public overlay must refuse"),
        };
        assert!(unsupported_overlay
            .to_string()
            .contains("does not advertise overlay lokr"));

        assert!(evaluate_shared_image_inner(
            "sd3_5_large",
            "sd3_5_large",
            &spec,
            true,
            &manifest,
            "q4",
            "image_to_image",
            None,
            None,
            MemoryGeometry {
                reference_count: 1,
                ..geometry
            },
            true,
            false,
            true,
            true,
            Some(VramBudget {
                free_gb: 24.0,
                total_gb: 48.0,
            }),
            reserve_for(Some(VramBudget {
                free_gb: 24.0,
                total_gb: 48.0,
            })),
            None,
            0,
            MemoryCacheState::Cold,
            None,
            Some(sd35_probe_contract("sd3_5_large", None)),
            Some(SD35_REQUEST_EVIDENCE_REVISION),
            None,
        )
        .expect("unsupported Hires context returns no candidate")
        .is_none());
    }

    #[test]
    fn sd35_manifest_declares_only_exact_resident_and_staged_cells() {
        let source = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, source)| *source)
            .expect("embedded model manifest");
        let stripped = sceneworks_core::jsonc::strip_jsonc_comments(source);
        let root: Value = serde_json::from_str(&stripped).expect("model manifest parses");
        for provider in ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"] {
            let model = root["models"]
                .as_array()
                .expect("models array")
                .iter()
                .find(|model| model["id"] == provider)
                .expect("SD3.5 model");
            let contract = &model["candle"]["memoryStrategyContract"];
            assert_eq!(contract["provider"], provider);
            assert_eq!(contract["exhaustive"], true);
            let rows = contract["implementations"]
                .as_array()
                .expect("SD3.5 implementations");
            assert_eq!(rows.len(), 4);
            for row in rows {
                assert!(matches!(
                    row["rung"].as_str(),
                    Some("resident" | "staged_residency")
                ));
                assert_eq!(row["tiers"], json!(["q4", "q8", "bf16"]));
                assert_eq!(row["modes"], json!(["text_to_image", "image_to_image"]));
                assert_eq!(row["sourceKinds"], json!(["dir"]));
                assert_eq!(row["pid"], Value::Null);
                let contexts = row["requestContexts"]
                    .as_array()
                    .expect("exact request contexts");
                assert_eq!(contexts.len(), 2);
                assert_eq!(contexts[0]["referenceCounts"], json!([0]));
                assert_eq!(contexts[1]["referenceCounts"], json!([1]));
                assert_eq!(contexts[0]["hasPhases"], false);
                assert_eq!(contexts[1]["hasPhases"], false);
                if row["rung"] == "staged_residency" {
                    assert_eq!(row["requiredOffloadPolicy"], "sequential");
                } else {
                    assert_eq!(row["requiredOffloadPolicy"], Value::Null);
                }
            }
        }
    }

    #[test]
    fn ideogram_selects_exact_receipt_priced_staged_and_refuses_crossed_auxiliaries() {
        let identity = format!("{IDEOGRAM_ADAPTER_OVERLAY_PREFIX}{}", "a".repeat(64));
        let contract = ideogram_probe_contract("ideogram_4_turbo", Some(&identity), true, true);
        let manifest = chroma_structural_manifest();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("sealed-ideogram-q4")))
            .with_resolved_route("ideogram_4_turbo");
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let evaluate = |free_gb: f64, contract| {
            evaluate_shared_image_inner(
                "ideogram_4_turbo",
                "ideogram_4_turbo",
                &spec,
                true,
                &manifest,
                "q4",
                "text_to_image",
                Some("lora"),
                Some("lora"),
                geometry,
                false,
                true,
                false,
                false,
                Some(VramBudget {
                    free_gb,
                    total_gb: 96.0,
                }),
                reserve_for(Some(VramBudget {
                    free_gb,
                    total_gb: 96.0,
                })),
                crate::vram_gate::predicted_peak_gb(&manifest, "q4"),
                gib(80),
                MemoryCacheState::Warm,
                Some("text_to_image"),
                Some(contract),
                Some(IDEOGRAM_REQUEST_EVIDENCE_REVISION),
                None,
            )
        };

        let staged = evaluate(23.0, contract.clone())
            .expect("exact Ideogram evaluation")
            .expect("staged strategy fits");
        assert_eq!(
            staged.context.selection.strategy,
            MemoryStrategy::StagedResidency
        );
        assert_eq!(staged.context.overlay.as_deref(), Some(identity.as_str()));
        assert_eq!(staged.context.cache_state, MemoryCacheState::Warm);
        assert_eq!(
            staged.context.optimization_authority,
            gen_core::MemoryOptimizationAuthority::Estimated
        );

        let hires = evaluate_shared_image_inner(
            "ideogram_4_turbo",
            "ideogram_4_turbo",
            &spec,
            true,
            &manifest,
            "q4",
            "image_to_image",
            Some("lora"),
            Some("lora"),
            MemoryGeometry {
                width: 2048,
                height: 2048,
                batch: 1,
                frames: 1,
                reference_count: 1,
            },
            true,
            true,
            true,
            false,
            Some(VramBudget {
                free_gb: 64.0,
                total_gb: 128.0,
            }),
            reserve_for(Some(VramBudget {
                free_gb: 64.0,
                total_gb: 128.0,
            })),
            crate::vram_gate::predicted_peak_gb(&manifest, "q4"),
            gib(80),
            MemoryCacheState::Cold,
            Some("image_to_image"),
            Some(contract.clone()),
            Some(IDEOGRAM_REQUEST_EVIDENCE_REVISION),
            None,
        )
        .expect("Hires structural estimate")
        .expect("scaled Hires staged strategy fits");
        assert_eq!(hires.context.mode, MemoryMode::ImageToImage);
        assert_eq!(
            hires.context.selection.strategy,
            MemoryStrategy::StagedResidency
        );
        // Derived, not pinned. The declared staged row is `sequentialPeakGb` 11 + the 2 GiB
        // `HEADROOM_GB` that `predicted_sequential_peak_gb` already folds in = 13 GiB; the
        // Hires envelope scales it by request pixels over `vramMeasuredPixels` (2048² / 1024² = 4)
        // = 52 GiB; and the receipt's exact auxiliaries — user adapter 2 + TurboTime 1 + PiD 3 —
        // add 6 GiB on top. Writing 50 here (the same sum with the headroom forgotten) is the slip
        // this derivation exists to prevent.
        let staged_row_gib = 11 + crate::vram_gate::HEADROOM_GB as u64;
        let hires_scale = (2048 * 2048) / 1_048_576;
        let auxiliaries_gib = 2 + 1 + 3;
        assert_eq!(
            hires.context.predicted_peak_bytes,
            gib(staged_row_gib * hires_scale + auxiliaries_gib)
        );

        // The receipt-derived staged floor carries its pad and pays no reserve on the budget side
        // (`memory_strategy::ReserveCharge`, sc-22664 D1/D4): it is compared against the
        // unreserved pool and a budget one hundredth under its widened peak refuses.
        assert_eq!(staged.admitted.available_gb, 23.0);
        assert!(evaluate(staged.admitted.needed_gb - 0.01, contract.clone()).is_err());
        assert!(evaluate(
            23.0,
            ideogram_probe_contract("ideogram_4_turbo", Some(&identity), false, true),
        )
        .err()
        .expect("missing mandatory TurboTime must fail")
        .to_string()
        .contains("physical adapter receipts"));
        assert!(evaluate(
            23.0,
            ideogram_probe_contract("ideogram_4_turbo", Some(&identity), true, false),
        )
        .err()
        .expect("missing PiD receipt must fail")
        .to_string()
        .contains("materialized asset facts"));
        let crossed = format!("{CHROMA_ADAPTER_OVERLAY_PREFIX}{}", "d".repeat(64));
        assert!(evaluate(
            23.0,
            ideogram_probe_contract("ideogram_4_turbo", Some(&crossed), true, true),
        )
        .err()
        .expect("crossed user adapter receipt must fail")
        .to_string()
        .contains("singular exact provider receipt"));
    }

    #[test]
    fn ideogram_all_optimized_rungs_produce_the_exact_execution_controls() {
        let contract = ideogram_probe_contract("ideogram_4", None, false, false);
        let tier = numeric_tier("ideogram_4", "q4").expect("q4 tier");
        let selection = |strategy, parameters| MemorySelection {
            strategy,
            parameters,
            tier,
        };

        let staged = memory_for_selection(
            &contract,
            selection(MemoryStrategy::StagedResidency, Default::default()),
        )
        .expect("staged controls");
        assert!(staged.stage_residency);
        assert!(!staged.tile_vae_decode);
        assert!(!staged.chunk_attention);
        assert!(!staged.stream_transformer_blocks);

        let decode = memory_for_selection(
            &contract,
            selection(
                MemoryStrategy::BoundedDecode,
                gen_core::MemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    ..Default::default()
                },
            ),
        )
        .expect("decode controls");
        assert!(decode.stage_residency && decode.tile_vae_decode);
        assert_eq!(decode.decode_tile_edge, Some(512));
        assert_eq!(decode.decode_overlap, Some(64));

        let attention = memory_for_selection(
            &contract,
            selection(
                MemoryStrategy::BoundedAttention,
                gen_core::MemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    attention_chunk_size: Some(64 * 1024 * 1024),
                    ..Default::default()
                },
            ),
        )
        .expect("attention controls");
        assert!(attention.stage_residency && attention.tile_vae_decode);
        assert!(attention.chunk_attention);
        assert_eq!(attention.attention_chunk_size, Some(64 * 1024 * 1024));

        let transformer = memory_for_selection(
            &contract,
            selection(
                MemoryStrategy::BoundedTransformerResidency,
                gen_core::MemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    attention_chunk_size: Some(64 * 1024 * 1024),
                    transformer_window_size: Some(4),
                    transformer_window_component: Some(gen_core::TransformerComponent::Dit),
                },
            ),
        )
        .expect("transformer controls");
        assert!(transformer.stage_residency && transformer.tile_vae_decode);
        assert!(transformer.chunk_attention && transformer.stream_transformer_blocks);
        assert_eq!(transformer.transformer_window_size, Some(4));
        assert_eq!(
            transformer.transformer_window_component,
            Some(gen_core::TransformerComponent::Dit)
        );
    }

    #[test]
    fn ideogram_pid_receipt_only_allows_the_native_hires_refinement_coordinate() {
        let contract = ideogram_probe_contract("ideogram_4", None, false, true);
        validate_ideogram_asset_facts(
            "ideogram_4",
            &contract,
            true,
            &MemoryMode::TextToImage,
            true,
        )
        .expect("the PiD first pass binds its receipt");
        validate_ideogram_asset_facts(
            "ideogram_4",
            &contract,
            false,
            &MemoryMode::ImageToImage,
            true,
        )
        .expect("the native Hires refinement retains the load's charged PiD receipt");
        assert!(validate_ideogram_asset_facts(
            "ideogram_4",
            &contract,
            false,
            &MemoryMode::TextToImage,
            true,
        )
        .is_err());
        // sc-20799: the tolerance is the HIRES FINAL pass, identified by `worker_multipass`, not
        // "any image-to-image request". A single-pass edit or style variation asks for no PiD
        // decode, so a PiD-charged contract is bytes it never requested and must be refused.
        assert!(
            validate_ideogram_asset_facts(
                "ideogram_4",
                &contract,
                false,
                &MemoryMode::ImageToImage,
                false,
            )
            .is_err(),
            "a single-pass image-to-image request must not borrow a PiD-charged contract"
        );
        // The tolerance still has to be reachable in both hires directions, and an unrelated mode
        // must not acquire it just by being multipass.
        assert!(validate_ideogram_asset_facts(
            "ideogram_4",
            &contract,
            false,
            &MemoryMode::Edit,
            true,
        )
        .is_err());
        assert!(ideogram_physical_receipt_identity(&contract)
            .expect("physical receipt")
            .starts_with(IDEOGRAM_PHYSICAL_RECEIPT_PREFIX));
    }

    /// sc-20799: Chroma's receipt is bound to its ROUTE, exactly as SANA's and SD3.5's are. The
    /// three Chroma turnkeys are different physical tensor sets under one family name, so without
    /// `provider_id == engine_id` a contract minted by `chroma1_base` prices a `chroma1_hd`
    /// request.
    #[test]
    fn chroma_asset_facts_bind_the_exact_route_identity() {
        let contract = chroma_probe_contract(None, 0, 0);
        assert_eq!(contract.provider_id, "chroma1_base");
        validate_chroma_asset_facts("chroma1_base", &contract)
            .expect("the minting route prices its own request");
        for crossed in ["chroma1_hd", "chroma1_flash"] {
            assert!(
                validate_chroma_asset_facts(crossed, &contract).is_err(),
                "{crossed} must not price a request from another Chroma turnkey's contract"
            );
        }
        assert!(
            validate_chroma_asset_facts("sana_1600m", &contract).is_err(),
            "a non-Chroma route must not reach the Chroma receipt at all"
        );
    }

    /// sc-20799: SDXL's overlay identity used to be `control:{N}` / `ip-adapter` / `pid` —
    /// cardinality and kind only — so two different ControlNets or IP-Adapters shared one
    /// admission identity. Assert the SHAPE (distinct payloads are distinct, equal payloads are
    /// stable, order matters), never a digest literal: the seal covers absolute paths, which are
    /// machine-dependent and must never be frozen into a golden.
    #[test]
    fn sdxl_overlay_identity_separates_distinct_physical_payloads() {
        let base = LoadSpec::new(WeightsSource::Dir(PathBuf::from("sdxl-q4")));
        assert_eq!(sdxl_provider_overlay(&base), None);

        let control_a = base
            .clone()
            .with_control(WeightsSource::File(PathBuf::from("tile-a.safetensors")));
        let control_b = base
            .clone()
            .with_control(WeightsSource::File(PathBuf::from("tile-b.safetensors")));
        let a = sdxl_provider_overlay(&control_a).expect("control overlay");
        let b = sdxl_provider_overlay(&control_b).expect("control overlay");
        assert!(a.starts_with(SDXL_OVERLAY_RECEIPT_PREFIX));
        assert_ne!(a, b, "two different tile ControlNets are two identities");
        assert_eq!(
            a,
            sdxl_provider_overlay(&control_a).expect("control overlay"),
            "the same assembly must seal to the same identity"
        );

        let ip_a = base
            .clone()
            .with_ip_adapter(WeightsSource::File(PathBuf::from("ip-a.safetensors")));
        let ip_b = base
            .clone()
            .with_ip_adapter(WeightsSource::File(PathBuf::from("ip-b.safetensors")));
        assert_ne!(
            sdxl_provider_overlay(&ip_a),
            sdxl_provider_overlay(&ip_b),
            "two different IP-Adapter checkpoints are two identities"
        );
        assert_ne!(
            sdxl_provider_overlay(&ip_a),
            sdxl_provider_overlay(&control_a),
            "an IP-Adapter slot and a control slot are different roles"
        );

        // The source KIND is sealed: a directory snapshot is not the same load as a single file.
        let control_dir = base
            .clone()
            .with_control(WeightsSource::Dir(PathBuf::from("tile-a.safetensors")));
        assert_ne!(sdxl_provider_overlay(&control_dir), a.clone().into());

        // PiD contributes both of its sources, and the slot ORDER is load order.
        let pid = control_a.clone().with_pid(
            WeightsSource::File(PathBuf::from("pid.safetensors")),
            WeightsSource::Dir(PathBuf::from("gemma")),
        );
        let pid_swapped = control_a.clone().with_pid(
            WeightsSource::File(PathBuf::from("gemma")),
            WeightsSource::Dir(PathBuf::from("pid.safetensors")),
        );
        assert_ne!(sdxl_provider_overlay(&pid), Some(a));
        assert_ne!(
            sdxl_provider_overlay(&pid),
            sdxl_provider_overlay(&pid_swapped)
        );
    }

    #[test]
    fn strict_control_uses_the_advertised_base_provider_evidence_cell() {
        assert_eq!(evidence_provider("z_image_control"), "z_image");
        assert_eq!(evidence_provider("z_image_turbo_control"), "z_image_turbo");
        assert_eq!(evidence_provider("z_image"), "z_image");
    }

    #[test]
    fn flux2_klein_reference_modes_preserve_catalog_and_typed_keys() {
        let cases = [
            ("edit_image", MemoryMode::Edit, "edit_image", "edit"),
            ("reference", MemoryMode::Edit, "edit_image", "edit"),
            ("image_to_image", MemoryMode::Edit, "edit_image", "edit"),
            (
                "character_image",
                MemoryMode::Other("character_image".to_owned()),
                "character_image",
                "character_image",
            ),
            (
                "style_variations",
                MemoryMode::Other("style_variations".to_owned()),
                "style_variations",
                "style_variations",
            ),
        ];
        for (request, expected_mode, calibration_key, scope_key) in cases {
            let binding = request_mode("flux2_klein_9b", request);
            assert_eq!(binding.mode, expected_mode, "request={request}");
            assert_eq!(
                binding.calibration_key, calibration_key,
                "request={request}"
            );
            assert_eq!(binding.scope_key, scope_key, "request={request}");
            assert_eq!(
                binding.mode.as_key(),
                binding.scope_key,
                "request={request}"
            );
        }

        let generic = request_mode("z_image", "image_to_image");
        assert_eq!(generic.mode, MemoryMode::ImageToImage);
        assert_eq!(generic.calibration_key, "image_to_image");
        assert_eq!(generic.scope_key, "image_to_image");
    }

    #[test]
    fn declaration_provider_mode_keeps_the_public_matrix_coordinate() {
        let raw_reference = request_mode_with_provider_override(
            "krea_2_raw",
            "text_to_image",
            Some("image_to_image"),
        );
        assert_eq!(raw_reference.mode, MemoryMode::ImageToImage);
        assert_eq!(raw_reference.scope_key, "image_to_image");
        assert_eq!(raw_reference.calibration_key, "text_to_image");

        let ordinary = request_mode_with_provider_override(
            "krea_2_raw",
            "text_to_image",
            Some("text_to_image"),
        );
        assert_eq!(ordinary.mode, MemoryMode::TextToImage);
        assert_eq!(ordinary.scope_key, "text_to_image");
        assert_eq!(ordinary.calibration_key, "text_to_image");

        let qwen_character = request_mode_with_provider_override(
            "qwen_image_edit",
            "character_image",
            Some("character_image"),
        );
        assert_eq!(
            qwen_character.mode,
            MemoryMode::Other("character_image".to_owned())
        );
        assert_eq!(qwen_character.scope_key, "character_image");
        assert_eq!(qwen_character.calibration_key, "character_image");
    }

    #[test]
    fn declaration_provider_overlay_binds_staged_selection_and_request_context() {
        let mut contract = gen_core::MemoryProviderContract::compatibility_default(
            "krea_2_edit",
            gen_core::MemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: true,
                host_to_device_block_materialization: true,
                block_materialization: gen_core::MemoryWindowMaterialization::DeviceFormatTransfer,
            },
        );
        for capability in &mut contract.strategies {
            capability.support = if matches!(
                capability.strategy,
                MemoryStrategy::Resident | MemoryStrategy::StagedResidency
            ) {
                gen_core::MemoryStrategySupport::Implemented
            } else {
                gen_core::MemoryStrategySupport::Missing
            };
        }
        // Declaring StagedResidency Implemented is not free: gen-core's `conformance_errors()`
        // requires the lifecycle that rung actually needs — the three phases plus synchronized
        // release of completed ones — because staging IS phase-scoped residency. Without them the
        // contract is non-conformant, and `select_strategy`'s first gate rejects the WHOLE request
        // as `Unverified { Invalid }` before any candidate is considered, which reads as "nothing
        // fit" rather than "the fixture declared a rung it did not equip".
        contract.lifecycle = gen_core::MemoryLifecycleCapabilities {
            phases: vec![
                gen_core::MemoryPhase::Conditioning,
                gen_core::MemoryPhase::Denoise,
                gen_core::MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            ..contract.lifecycle
        };
        assert!(
            contract.conformance_errors().is_empty(),
            "the staged fixture must be a shape a real provider could declare: {:?}",
            contract.conformance_errors()
        );
        // `sequentialPeakGb` is read PER TIER by `vram_gate::predicted_sequential_peak_gb`
        // (`sequential.get(tier_key)`), so a bare scalar looks up as `None` and the staged floor
        // silently falls back to the RESIDENT peak — which can never fit where resident does not,
        // making this test's premise unsatisfiable by construction. Same map shape the sibling
        // `eligible_lens_selector_contract_...` fixture uses.
        const STAGED_ROW_GB: f64 = 2.5;
        const RESIDENT_PEAK_GB: f64 = 16.0;
        // `evaluate_shared_image_inner` hands the selector a fixed 2 GiB reserved headroom, and
        // `Budget::effective_gb` subtracts it, so the staged floor competes against
        // `free_gb - SELECTOR_RESERVE_GB`.
        const SELECTOR_RESERVE_GB: f64 = 2.0;
        let manifest = json!({ "candle": { "sequentialPeakGb": { "q4": STAGED_ROW_GB } } })
            .as_object()
            .expect("manifest object")
            .clone();
        // Derived through the production formula rather than guessed: the staged row is padded by
        // the allocator reserve and then widened by the candle estimate margin before the fit check.
        // The previous literals (4.0 row, 8.0 free) missed by 0.24 GiB even with the map shape, and
        // a margin re-derivation would silently move the window again.
        let staged_floor_gb = STAGED_ROW_GB + crate::vram_gate::HEADROOM_GB;
        let widened_staged_gb =
            staged_floor_gb * (1.0 + crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD);
        let free_gb = widened_staged_gb + SELECTOR_RESERVE_GB + 0.3;
        assert!(
            free_gb - SELECTOR_RESERVE_GB < RESIDENT_PEAK_GB,
            "the budget must stay below the resident estimate, or 'staged fits where resident does \
             not' proves nothing"
        );
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("krea-q4")));
        let evaluation = evaluate_shared_image_inner(
            "krea_2_edit",
            "krea_2_raw",
            &spec,
            true,
            &manifest,
            "q4",
            "edit_image",
            Some("lora"),
            None,
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 1,
            },
            true,
            false,
            false,
            false,
            Some(VramBudget {
                free_gb,
                total_gb: 32.0,
            }),
            reserve_for(Some(VramBudget {
                free_gb,
                total_gb: 32.0,
            })),
            Some(RESIDENT_PEAK_GB),
            0,
            MemoryCacheState::Cold,
            Some("edit_image"),
            Some(contract),
            None,
            None,
        )
        .expect("selector evaluation")
        .expect("staged estimate fits where resident does not");
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::StagedResidency
        );
        assert_eq!(evaluation.context.overlay, None);
        assert!(
            evaluation
                .memory
                .expect("staged selection configures generation memory")
                .stage_residency
        );
    }

    #[test]
    fn mage_routes_preserve_text_to_image_and_edit_scope_keys() {
        for engine_id in ["mage_flow_base", "mage_flow", "mage_flow_turbo"] {
            let binding = request_mode(engine_id, "image_generation");
            assert_eq!(binding.mode, MemoryMode::TextToImage, "engine={engine_id}");
            assert_eq!(binding.calibration_key, "text_to_image");
            assert_eq!(binding.scope_key, "text_to_image");
        }

        for engine_id in [
            "mage_flow_edit_base",
            "mage_flow_edit",
            "mage_flow_edit_turbo",
        ] {
            let binding = request_mode(engine_id, "edit_image");
            assert_eq!(binding.mode, MemoryMode::Edit, "engine={engine_id}");
            assert_eq!(binding.calibration_key, "edit_image");
            assert_eq!(binding.scope_key, "edit");
        }
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn mage_numeric_tiers_bind_only_the_active_q4_precision_floors() {
        for engine_id in [
            "mage_flow_base",
            "mage_flow",
            "mage_flow_turbo",
            "mage_flow_edit_base",
            "mage_flow_edit",
            "mage_flow_edit_turbo",
        ] {
            let declared = crate::inference_runtime::media_descriptor(engine_id)
                .unwrap_or_else(|| panic!("missing registered Mage descriptor {engine_id}"))
                .capabilities
                .component_precision_floors;
            assert!(!declared.is_empty(), "engine={engine_id}");
            assert!(
                declared.iter().all(|floor| floor.applies_to(Quant::Q4)),
                "engine={engine_id}"
            );
            assert_eq!(
                numeric_tier(engine_id, "q4")
                    .expect("q4 tier")
                    .component_precision_floors,
                declared,
                "engine={engine_id}"
            );
            for tier in ["q8", "bf16"] {
                assert!(
                    numeric_tier(engine_id, tier)
                        .unwrap_or_else(|| panic!("missing {tier} tier"))
                        .component_precision_floors
                        .is_empty(),
                    "engine={engine_id} tier={tier}"
                );
            }
        }

        for engine_id in [
            "z_image",
            "z_image_turbo",
            "z_image_control",
            "z_image_turbo_control",
            "qwen_image",
            "qwen_image_edit",
            "flux1_schnell",
            "flux1_dev",
            "flux2_dev",
            "flux2_klein_9b",
        ] {
            for tier in ["q4", "q8", "bf16"] {
                assert!(
                    numeric_tier(engine_id, tier)
                        .unwrap_or_else(|| panic!("missing {tier} tier"))
                        .component_precision_floors
                        .is_empty(),
                    "engine={engine_id} tier={tier}"
                );
            }
        }
    }

    #[test]
    fn mage_manifests_declare_unverified_capabilities_without_calibrations() {
        let source = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, source)| *source)
            .expect("embedded model manifest");
        let stripped = sceneworks_core::jsonc::strip_jsonc_comments(source);
        let root: Value = serde_json::from_str(&stripped).expect("model manifest parses");
        let models = root["models"].as_array().expect("models array");

        for model_id in [
            "mage_flow_edit_base",
            "mage_flow_edit",
            "mage_flow_edit_turbo",
            "mage_flow_base",
            "mage_flow",
            "mage_flow_turbo",
        ] {
            let candle = models
                .iter()
                .find(|model| model["id"] == model_id)
                .unwrap_or_else(|| panic!("missing Mage model {model_id}"))["candle"]
                .as_object()
                .expect("Candle manifest block");
            assert_eq!(candle["supportsSequentialOffload"], true);
            assert_eq!(candle["measured"], false);
            assert!(candle.get("calibrations").is_none());

            let capabilities = candle["memoryStrategyCapabilities"]
                .as_object()
                .expect("memory strategy capabilities");
            assert_eq!(capabilities.len(), 2);
            assert!(
                !capabilities.contains_key("bounded_decode"),
                "{model_id}: CoD full-field normalization makes bounded decode structurally inapplicable"
            );
            let bounded_decode_exemption =
                &candle["memoryStrategyStructuralExemptions"]["bounded_decode"];
            assert_eq!(
                bounded_decode_exemption["overlays"],
                json!(["none", "lora"])
            );
            assert_eq!(
                bounded_decode_exemption["evidence"]
                    .as_array()
                    .expect("bounded decode structural evidence")
                    .len(),
                2
            );
            assert_eq!(
                capabilities["bounded_attention"]["parameters"],
                json!({ "attentionChunkSize": 67_108_864 })
            );
            assert_eq!(
                capabilities["bounded_transformer_residency"]["parameters"],
                json!({ "transformerWindowSize": 1, "transformerWindowComponent": "Dit" })
            );
            assert!(capabilities
                .values()
                .all(|capability| capability["overlays"] == json!(["none"])));
        }
    }

    #[test]
    fn character_identity_and_control_bindings_require_the_canonical_exact_keys() {
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 1,
        };
        let binding = |overlay| {
            json!({
                "provider": "flux1_dev",
                "tier": "q4",
                "mode": "character_image",
                "overlay": overlay,
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
            })
            .as_object()
            .expect("binding object")
            .clone()
        };

        for overlay in ["identity", "control"] {
            let exact = binding(overlay);
            assert!(binding_matches_request(
                &exact,
                "flux1_dev",
                "q4",
                "character_image",
                overlay,
                geometry,
            ));
            assert!(!binding_matches_request(
                &exact,
                "flux1_dev",
                "q4",
                "image_to_image",
                overlay,
                geometry,
            ));
        }

        let identity = binding("identity");
        assert!(!binding_matches_request(
            &identity,
            "flux1_dev",
            "q4",
            "character_image",
            "ip_adapter",
            geometry,
        ));
    }

    #[test]
    fn flux1_base_bindings_are_current_selectable_and_overlay_free() {
        let source = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, source)| *source)
            .expect("embedded model manifest");
        let stripped = sceneworks_core::jsonc::strip_jsonc_comments(source);
        let root: Value = serde_json::from_str(&stripped).expect("model manifest parses");
        let models = root["models"].as_array().expect("models array");
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };

        for (model_id, provider) in [("flux_schnell", "flux1_schnell"), ("flux_dev", "flux1_dev")] {
            let model = models
                .iter()
                .find(|model| model["id"] == model_id)
                .expect("FLUX model");
            let manifest = model.as_object().expect("model object");
            let bindings = manifest["candle"]["calibrations"]
                .as_array()
                .expect("calibration bindings");
            assert_eq!(bindings.len(), 5);
            assert!(bindings.iter().all(|binding| binding["overlay"] == "none"));

            let candidates = verified_candidates(
                manifest,
                model_id,
                provider,
                "q4",
                &request_mode(provider, "text_to_image"),
                "none",
                geometry,
                &mut Vec::new(),
            )
            .expect("packaged FLUX evidence");
            assert_eq!(candidates.len(), 5);
            assert_eq!(
                candidates
                    .iter()
                    .map(|candidate| candidate.key.strategy)
                    .collect::<Vec<_>>(),
                vec![
                    MemoryStrategy::Resident,
                    MemoryStrategy::StagedResidency,
                    MemoryStrategy::BoundedDecode,
                    MemoryStrategy::BoundedAttention,
                    MemoryStrategy::BoundedTransformerResidency,
                ]
            );
            assert!(matches!(candidates[0].parity, MemoryParityContract::Exact));
            assert!(matches!(candidates[1].parity, MemoryParityContract::Exact));
            assert!(candidates[2..].iter().all(|candidate| matches!(
                candidate.parity,
                MemoryParityContract::Tolerance { .. }
            )));

            for overlay in ["identity", "control"] {
                assert!(verified_candidates(
                    manifest,
                    model_id,
                    provider,
                    "q4",
                    &request_mode(provider, "character_image"),
                    overlay,
                    MemoryGeometry {
                        reference_count: 1,
                        ..geometry
                    },
                    &mut Vec::new(),
                )
                .expect("uncertified overlay query")
                .is_empty());
            }
        }
    }

    #[test]
    fn flux2_dev_v2_bindings_stay_historical_but_cannot_enter_the_v3_selector() {
        let source = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, source)| *source)
            .expect("embedded model manifest");
        let stripped = sceneworks_core::jsonc::strip_jsonc_comments(source);
        let root: Value = serde_json::from_str(&stripped).expect("model manifest parses");
        let model = root["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|model| model["id"] == "flux2_dev")
            .expect("FLUX.2-dev model");
        let manifest = model.as_object().expect("FLUX.2-dev model object");
        let bindings = manifest["candle"]["calibrations"]
            .as_array()
            .expect("FLUX.2-dev calibration bindings");
        assert_eq!(bindings.len(), 5);
        assert!(bindings.iter().all(|binding| {
            binding["provider"] == "flux2_dev"
                && binding["tier"] == "q4"
                && binding["mode"] == "text_to_image"
                && binding["overlay"] == "none"
                && binding["inferenceRevision"] == "5ffd7612e7de4e76b6db00a7148ed3d9c15b4c0d"
                // sc-17774: `compatibleInferenceRevision` is gone — it was flux2_dev's one-shot
                // hand-audited hatch. The binding now carries the closure digest it was captured
                // under, which is what currency compares.
                && binding["inferenceClosureDigest"].is_string()
        }));

        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let candidates = verified_candidates(
            manifest,
            "flux2_dev",
            "flux2_dev",
            "q4",
            &request_mode("flux2_dev", "text_to_image"),
            "none",
            geometry,
            &mut Vec::new(),
        )
        .expect("packaged FLUX.2-dev evidence");
        assert_eq!(candidates.len(), 5);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.predicted_peak_bytes)
                .collect::<Vec<_>>(),
            vec![
                47_700_000_000,
                44_300_000_000,
                34_700_000_000,
                25_200_000_000,
                14_300_000_000
            ]
        );
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.observed_peak_bytes)
                .collect::<Vec<_>>(),
            vec![
                Some(44_911_779_404),
                Some(34_070_566_540),
                Some(30_325_300_672),
                Some(23_724_359_104),
                Some(13_537_107_920),
            ]
        );
        assert!(candidates.iter().all(|candidate| {
            candidate
                .observed_peak_bytes
                .is_some_and(|active| candidate.predicted_peak_bytes > active)
        }));
        let mut unaudited_manifest = manifest.clone();
        for binding in unaudited_manifest["candle"]["calibrations"]
            .as_array_mut()
            .expect("mutable FLUX.2-dev calibration bindings")
        {
            // sc-17774: as above — the deleted hatch cannot make a binding unaudited any more, so
            // move the mutation onto the closure digest currency actually compares.
            binding["inferenceClosureDigest"] = json!("a".repeat(64));
        }
        assert!(verified_candidates(
            &unaudited_manifest,
            "flux2_dev",
            "flux2_dev",
            "q4",
            &request_mode("flux2_dev", "text_to_image"),
            "none",
            geometry,
            &mut Vec::new(),
        )
        .expect("unaudited compatibility query")
        .is_empty());
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.key.strategy)
                .collect::<Vec<_>>(),
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ]
        );
        assert!(verified_candidates(
            manifest,
            "flux2_dev",
            "flux2_dev",
            "q4",
            &request_mode("flux2_dev", "text_to_image"),
            "control",
            geometry,
            &mut Vec::new(),
        )
        .expect("uncertified FLUX.2-dev control query")
        .is_empty());

        // The five exact historical bindings remain queryable for auditability. Admission is a
        // separate step: their v2 calibration identity no longer describes the live v3
        // caption-upsample lifecycle, so none of these ladder rungs may reach the selector.
        let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("flux2-dev-q4-fixture")))
            .with_quant(Quant::Q4);
        spec.load_shape = gen_core::LoadShape::DeferredMaterialization;
        let live_predicted_peak = crate::vram_gate::predicted_peak_gb(manifest, "q4")
            .expect("measured FLUX.2-dev q4 high-water");
        let evaluate = |free_gb: f64| {
            evaluate_shared_image(
                "flux2_dev",
                "flux2_dev",
                &spec,
                true,
                manifest,
                "q4",
                "text_to_image",
                None,
                geometry,
                false,
                false,
                false,
                false,
                Some(VramBudget {
                    free_gb,
                    total_gb: 96.0,
                }),
                reserve_for(Some(VramBudget {
                    free_gb,
                    total_gb: 96.0,
                })),
                Some(live_predicted_peak),
                0,
                MemoryCacheState::Cold,
            )
            .expect("FLUX.2-dev safe-device-peak evaluation")
        };
        for budget_gb in [16.0, 24.0] {
            assert!(
                evaluate(budget_gb).is_none(),
                "the obsolete v2 ladder must not reopen live v3 FLUX.2-dev on a {budget_gb} GB budget"
            );
        }

        assert!(
            candidates.iter().all(|candidate| candidate
                .calibration_fingerprint
                .ends_with("device-format-blocks-v2")),
            "the retained packaged rows must remain truthfully labeled as v2 history"
        );
        let live = evaluate(96.0).expect("live v3 FLUX.2-dev resident route");
        assert_eq!(live.context.selection.strategy, MemoryStrategy::Resident);
        assert_eq!(
            live.context.calibration_fingerprint,
            "flux2-dev-cuda-caption-upsample-staged-host-full-edge-decode-bounded-attention-device-format-blocks-v3"
        );
    }

    /// sc-18097 headline (epic 18093 R1b): an UNMEASURED provider cell — no packaged calibration
    /// records at all — under a small emulated VRAM budget (`SCENEWORKS_CUDA_VRAM_CAP_GB`
    /// scenario, driven through the same pure seam the cap feeds) engages the ladder by
    /// estimate-floor instead of freezing to resident-or-nothing, and refuses below the widened
    /// floors.
    ///
    /// Floor arithmetic (sc-22664: no headroom inside a candidate; the reserve is charged once on
    /// the budget): resident estimate 8.0 GiB caller-predicted = the 6.0 manifest row plus the
    /// legacy headroom, so the ladder's resident candidate is the raw 6.0; staged floor = the raw
    /// `sequentialPeakGb` row 2.5 GiB, widened by the candle recapture spread to 2.55. A 7 GiB
    /// budget on a 96 GiB card (foreign residency above the slack, so the reserve is the legacy 2
    /// GiB ceiling → 5 GiB effective) admits exactly the staged floor.
    #[test]
    fn unmeasured_provider_under_a_small_budget_engages_the_estimate_floor_ladder() {
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 6.0 },
                "sequentialPeakGb": { "q4": 2.5 },
                "supportsSequentialOffload": true
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("missing-z-image-q4")));
        let evaluate_as = |free_gb: f64, artifact_is_certified: bool| {
            evaluate_shared_image(
                "z_image_turbo",
                "z_image_turbo",
                &spec,
                artifact_is_certified,
                &manifest,
                "q4",
                "text_to_image",
                None,
                MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
                false,
                false,
                false,
                false,
                Some(VramBudget {
                    free_gb,
                    total_gb: 96.0,
                }),
                reserve_for(Some(VramBudget {
                    free_gb,
                    total_gb: 96.0,
                })),
                Some(8.0),
                0,
                MemoryCacheState::Cold,
            )
            .expect("unmeasured z-image evaluation")
        };
        let evaluate = |free_gb: f64| evaluate_as(free_gb, true);

        let evaluation = evaluate(7.0).expect(
            "an implemented rung's estimate floor must admit where the resident estimate cannot",
        );
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::StagedResidency,
            "the cheapest fitting estimate rung must win"
        );
        // The floor is the manifest staged row plus its structural pad, exactly as before
        // sc-22664 (the selector owns the widening; the pad-carrying floor pays no reserve on the
        // budget side — `memory_strategy::ReserveCharge`).
        assert!(
            (evaluation.predicted_peak_gb - (2.5 + crate::vram_gate::HEADROOM_GB)).abs() < 1e-6
        );
        assert_eq!(
            evaluation.admitted.available_gb, 7.0,
            "a pad-carrying floor compares against the UNRESERVED pool"
        );
        let memory = evaluation
            .memory
            .expect("optimized selection carries memory");
        assert!(memory.stage_residency);
        assert!(!memory.tile_vae_decode);
        assert_eq!(
            evaluation.context.optimization_authority,
            gen_core::MemoryOptimizationAuthority::Estimated,
            "a structural floor must never impersonate measured calibration"
        );
        // Estimate admissions are legacy-scoped in telemetry, exactly like the resident estimate.
        assert_eq!(
            evaluation.context.evidence_revision,
            Z_IMAGE_REQUEST_EVIDENCE_REVISION
        );

        // Margin mutation arm: at 4.52 GiB free the padded staged floor (2.5 + 2 = 4.5) fits but
        // the widened one (4.59) does not — the selector rejects, this lane falls back to the
        // established legacy gates (`None`), and a zeroed estimate margin would admit instead and
        // flip this arm red.
        assert!(
            evaluate(4.52).is_none(),
            "an estimate must be graded at its WIDENED peak, not its raw floor"
        );

        // Control: the resident estimate itself still admits on a roomy card without engaging any
        // rung — the floors extend the ladder downward, they do not perturb the fast path.
        let roomy = evaluate(64.0).expect("resident admits on a roomy card");
        assert_eq!(roomy.context.selection.strategy, MemoryStrategy::Resident);
        assert!(roomy.memory.is_none());
        assert_eq!(
            roomy.context.optimization_authority,
            gen_core::MemoryOptimizationAuthority::Resident,
            "a live resident estimate must not be labeled as calibrated evidence"
        );

        // sc-18097 review (major): the floors are gated on artifact certification, the SAME
        // conjunct that gates the packaged records. The manifest rows describe the certified
        // bytes; an imported/community checkpoint on this route is different bytes, so at the
        // exact budget where the certified artifact engages a floor rung, the uncertified one
        // gets no floors, its resident estimate does not fit, and the lane hands back to the
        // established gates (`None`). Removing the certification conjunct flips this red.
        assert!(
            evaluate_as(7.0, false).is_none(),
            "an uncertified artifact must not engage a rung off the certified manifest's rows"
        );
        // …and it keeps its pre-sc-18097 resident behavior where the resident estimate DOES fit,
        // so the gate withholds the floors rather than refusing the artifact.
        let uncertified_roomy =
            evaluate_as(64.0, false).expect("an uncertified artifact still admits resident");
        assert_eq!(
            uncertified_roomy.context.selection.strategy,
            MemoryStrategy::Resident
        );
    }

    /// A synthetic provider contract with every rung implemented and rungs 2-4 pinned to either
    /// side of the staging composition split (sc-18253). `bind_deep_rungs_to_staging` mirrors the
    /// shipped z-image contract's `additional_prerequisites` edges; without them the gen-core
    /// default composition excludes `StagedResidency` from every deep rung. `staged_implemented`
    /// false models the finding's contract — a provider implementing deep rungs with no staging
    /// rung at all.
    fn composition_probe_contract(
        staged_implemented: bool,
        bind_deep_rungs_to_staging: bool,
    ) -> gen_core::MemoryProviderContract {
        let mut contract = gen_core::MemoryProviderContract::compatibility_default(
            "z_image_turbo",
            gen_core::MemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: true,
                host_to_device_block_materialization: true,
                block_materialization: gen_core::MemoryWindowMaterialization::DeviceFormatTransfer,
            },
        );
        contract.strategies = MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| gen_core::MemoryStrategyCapability {
                strategy,
                support: if strategy == MemoryStrategy::StagedResidency && !staged_implemented {
                    gen_core::MemoryStrategySupport::Missing
                } else {
                    gen_core::MemoryStrategySupport::Implemented
                },
                parameters: match strategy {
                    MemoryStrategy::BoundedDecode => gen_core::MemoryParameterRanges {
                        decode_tile_edges: vec![512],
                        decode_overlaps: vec![128],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedAttention => gen_core::MemoryParameterRanges {
                        attention_chunk_sizes: vec![1024],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedTransformerResidency => {
                        gen_core::MemoryParameterRanges {
                            transformer_window_sizes: vec![1],
                            ..Default::default()
                        }
                    }
                    _ => Default::default(),
                },
            })
            .collect();
        contract.lifecycle = gen_core::MemoryLifecycleCapabilities {
            phases: vec![
                gen_core::MemoryPhase::Conditioning,
                gen_core::MemoryPhase::Denoise,
                gen_core::MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: true,
        };
        // Rung 4's SHARED prerequisite is the deferred load shape, not staged residency — which is
        // exactly why a rung-4-without-staging contract is a legal shape.
        contract.load_shape = gen_core::LoadShape::DeferredMaterialization;
        contract.formula = gen_core::MemoryFormulaKind::AssetBytesPlusHeadroom;
        contract.calibration = Some(gen_core::MemoryCalibrationIdentity::new(
            "sc-18253-composition-probe-v1",
            gen_core::LoadShape::DeferredMaterialization,
        ));
        if bind_deep_rungs_to_staging {
            contract.additional_prerequisites = [
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ]
            .into_iter()
            .map(|strategy| {
                (
                    strategy,
                    gen_core::MemoryStrategyPrerequisite::Rung {
                        rung: MemoryStrategy::StagedResidency,
                        scope: gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
                    },
                )
            })
            .collect();
        }
        contract
    }

    /// No anchor store at all: the contract-only path, under the default facts.
    fn no_anchors() -> CandleLadderAnchors<'static> {
        CandleLadderAnchors {
            store: None,
            facts: sceneworks_core::memory_anchor::ArchitectureFacts::default(),
        }
    }

    // -------------------------------------------------------------------------------------
    // sc-22664 (epic 22657 E4/E7) fixture: the sc-15859 Z-Image-Turbo q4 candle record.
    // -------------------------------------------------------------------------------------

    /// Z-Image-Turbo q4 component bytes on the candle lane (the `SceneWorks/z-image-turbo-mlx` q4
    /// tier the retained record names): text encoder 2.26 GB, transformer 3.47 GB, VAE 0.16 GB —
    /// the same figures `sceneworks_core::memory_anchor`'s own AC fixture binds to the packaged
    /// tier size.
    const Z_IMAGE_Q4_COMPONENTS: sceneworks_core::memory_anchor::ComponentBytes =
        sceneworks_core::memory_anchor::ComponentBytes {
            conditioning: 2_260_000_000,
            transformer: 3_470_000_000,
            decoder: 160_000_000,
        };

    /// Z-Image architecture facts as the candle provider publishes them off its loader presets
    /// (`candle-gen-z-image::memory_strategy::architecture_facts`: `DitConfig::z_image_turbo()`,
    /// `VaeConfig::z_image()`): 30 heads of 128, 30 blocks, patch 2, 16 latent channels, x8 VAE,
    /// bf16 activations, and NO temporal scale — Z-Image ships the FLUX.1 image VAE, so the axis is
    /// structurally absent and declared absent rather than `1`. This is the gen-core block on the
    /// fixture contract ([`Z_IMAGE_CONTRACT_FACTS`]) as `architecture_facts_from_contract` states
    /// it (sc-22667); the fixtures that pass it explicitly and the production seam must agree, and
    /// the headline test asserts that they do.
    const Z_IMAGE_FACTS: sceneworks_core::memory_anchor::ArchitectureFacts =
        sceneworks_core::memory_anchor::ArchitectureFacts {
            attention_heads: Some(30),
            head_dim: Some(128),
            transformer_blocks: Some(30),
            patch_size: Some(2),
            latent_channels: Some(16),
            vae_spatial_scale: Some(8),
            vae_temporal_scale: None,
            activation_dtype_width: Some(2),
        };

    /// The same facts as the provider contract carries them (sc-22667): what
    /// `candle-gen-z-image` publishes for a resolved snapshot, restated on the fixture contract so
    /// the production seam (`CandleLadderAnchors::packaged`) reads REAL facts off it.
    const Z_IMAGE_CONTRACT_FACTS: gen_core::MemoryArchitectureFacts =
        gen_core::MemoryArchitectureFacts {
            attention_heads: Some(30),
            head_dim: Some(128),
            transformer_blocks: Some(30),
            patch_size: Some(2),
            latent_channels: Some(16),
            vae_spatial_scale: Some(8),
            vae_temporal_scale: None,
            activation_dtype_width: Some(2),
        };

    /// The staged phase peaks of the retained sc-15859 q4 record (1024x1024, deferred
    /// materialization, `staged_residency` alone engaged): cond 3.10 / denoise 8.05 / decode
    /// 11.74 GB, byte-exact.
    const Z_IMAGE_Q4_STAGED_PEAKS: sceneworks_core::memory_anchor::AnchorPhaseBytes =
        sceneworks_core::memory_anchor::AnchorPhaseBytes {
            conditioning: 3_097_493_504,
            denoise: 8_050_966_528,
            decode: 11_741_954_048,
        };

    /// The fully engaged 1024x1024 composition's derived peaks from that record with the Z-Image
    /// facts (`memory_anchor::z_image_q4_rungs_price_from_the_staged_anchor_*` states the
    /// arithmetic): one resident block plus the non-score denoise residue plus the 64 Mi x 2 B
    /// chunk; and the decode residue split into the blender floor and the 3/8 host-transfer band.
    const Z_IMAGE_Q4_WINDOWED_PEAKS: sceneworks_core::memory_anchor::AnchorDerivedPhases =
        sceneworks_core::memory_anchor::AnchorDerivedPhases {
            conditioning: 3_097_493_504,
            denoise: 115_666_667 + 3_306_946_688 + 134_217_728,
            decode: 4_509_786_368,
        };

    fn z_image_q4_anchor() -> sceneworks_core::memory_anchor::MemoryAnchor {
        use sceneworks_core::memory_anchor::{
            AnchorBackend, AnchorGeometry, AnchorLoadShape, AnchorMeasuredRegime, AnchorSource,
            MemoryAnchor,
        };
        MemoryAnchor {
            id: "z_image_turbo:candle:q4:sc-15859".to_owned(),
            model_id: "z_image_turbo".to_owned(),
            model_family: "z_image".to_owned(),
            route: "z_image_turbo".to_owned(),
            provider: "z_image_turbo".to_owned(),
            backend: AnchorBackend::Candle,
            tier: "q4".to_owned(),
            transformer_variant: None,
            decoder: None,
            mode: "text_to_image".to_owned(),
            overlay: None,
            reference_count: 0,
            load_shape: AnchorLoadShape::DeferredMaterialization,
            measured_regime: AnchorMeasuredRegime {
                decode_tiled: false,
                transformer_windowed: false,
                staged: true,
                attention_chunked: false,
            },
            source: AnchorSource {
                path: "docs/calibration/sc-15859/z-image-turbo-q4-candle-anchor.json".to_owned(),
                sha256: String::new(),
                record_id: String::new(),
                calibration_fingerprint: "sc-18253-composition-probe-v1".to_owned(),
                // The model's OWN live loader-closure declaration, READ rather than frozen as a
                // literal (sc-22666): since the per-store scope split went with the model
                // allow-list, `candle_image_anchor` grades a fixture row's currency exactly as it
                // grades a packaged one, and a literal here would be a pin-coupled golden.
                loader_closure_digest:
                    sceneworks_core::memory_anchor::packaged_anchor_loader_closures()
                        .and_then(|closures| {
                            closures.digest_for("z_image_turbo", AnchorBackend::Candle)
                        })
                        .expect("z_image_turbo:candle must declare a loader closure")
                        .to_owned(),
            },
            geometry: AnchorGeometry {
                width: 1024,
                height: 1024,
                frames: 1,
                fps: None,
            },
            phase_active_peak_bytes: Z_IMAGE_Q4_STAGED_PEAKS,
            phase_allocator_envelope_bytes: None,
            overall_allocator_envelope_bytes: Z_IMAGE_Q4_STAGED_PEAKS.decode,
            underived_reason: None,
            component_bytes: None,
        }
    }

    fn z_image_q4_store() -> sceneworks_core::memory_anchor::MemoryAnchorStore {
        sceneworks_core::memory_anchor::MemoryAnchorStore {
            schema_version: sceneworks_core::memory_anchor::MEMORY_ANCHOR_SCHEMA_VERSION,
            anchors: vec![z_image_q4_anchor()],
            analytic_only: Vec::new(),
            component_deltas: Vec::new(),
        }
    }

    /// The PACKAGED store with its `z_image_turbo` candle rows re-stamped at the loader-closure
    /// digest the pin currently declares, so they grade as current — the same construction, and
    /// the same rationale, as `vram_gate::tests::krea_live_anchor_store`: whether a given pin
    /// leaves the shipped rows' digest current is a property of the pin, reported (never gated)
    /// by `sceneworks-core`'s `packaged_anchor_currency_is_reported_not_gated`, and not of the
    /// derivation graded here. Only the digest is touched; every measured byte is the packaged
    /// corpus's.
    fn z_image_live_anchor_store() -> sceneworks_core::memory_anchor::MemoryAnchorStore {
        use sceneworks_core::memory_anchor::AnchorBackend;

        let store = sceneworks_core::memory_anchor::packaged_memory_anchors()
            .expect("the packaged anchor store")
            .clone();
        let digest = sceneworks_core::memory_anchor::packaged_anchor_loader_closures()
            .and_then(|closures| closures.digest_for("z_image_turbo", AnchorBackend::Candle))
            .expect("z_image_turbo:candle must declare a loader closure")
            .to_owned();
        let anchors = store
            .anchors
            .into_iter()
            .map(|mut anchor| {
                if anchor.model_id == "z_image_turbo" && anchor.backend == AnchorBackend::Candle {
                    anchor.source.loader_closure_digest = digest.clone();
                }
                anchor
            })
            .collect();
        sceneworks_core::memory_anchor::MemoryAnchorStore { anchors, ..store }
    }

    /// The Z-Image-Turbo q4 contract shape the ladder grades: every rung implemented and bound to
    /// staging, the published parameters (`bounded_decode` 512/128, `bounded_attention` 64 Mi
    /// scores, transformer window 1), deferred materialization like the record, the q4
    /// component bytes as its asset facts, and the provider's architecture facts
    /// ([`Z_IMAGE_CONTRACT_FACTS`], sc-22667) as its architecture block.
    fn z_image_fixture_contract() -> gen_core::MemoryProviderContract {
        let mut contract = composition_probe_contract(true, true);
        for capability in &mut contract.strategies {
            if capability.strategy == MemoryStrategy::BoundedAttention {
                capability.parameters.attention_chunk_sizes = vec![64 * 1024 * 1024];
            }
        }
        contract.asset_facts = gen_core::MemoryAssetFacts {
            base_bytes: Z_IMAGE_Q4_COMPONENTS.total(),
            conditioning_bytes: Z_IMAGE_Q4_COMPONENTS.conditioning,
            transformer_bytes: Z_IMAGE_Q4_COMPONENTS.transformer,
            decoder_bytes: Z_IMAGE_Q4_COMPONENTS.decoder,
            overlay_bytes: 0,
        };
        contract.architecture_facts = Z_IMAGE_CONTRACT_FACTS;
        contract
    }

    /// The shipped `z_image_turbo` candle rows: resident q4 18.4 GiB, staged q4 5.7 GiB.
    fn z_image_fixture_manifest() -> JsonObject<String, Value> {
        json!({
            "candle": {
                "vramGbByTier": { "q4": 18.4 },
                "vramMeasuredPixels": 1_048_576,
                "sequentialPeakGb": { "q4": 5.7 },
                "supportsSequentialOffload": true
            }
        })
        .as_object()
        .unwrap()
        .clone()
    }

    const Z_IMAGE_FIXTURE_GEOMETRY: MemoryGeometry = MemoryGeometry {
        width: 1024,
        height: 1024,
        batch: 1,
        frames: 1,
        reference_count: 0,
    };

    fn z_image_ladder_anchors(
        store: &sceneworks_core::memory_anchor::MemoryAnchorStore,
    ) -> CandleLadderAnchors<'_> {
        CandleLadderAnchors {
            store: Some(store),
            facts: Z_IMAGE_FACTS,
        }
    }

    fn z_image_fixture_floors(
        anchors: CandleLadderAnchors<'_>,
        contract: &gen_core::MemoryProviderContract,
    ) -> Vec<EstimateCandidate> {
        synthesize_estimate_floors(
            "z_image_turbo",
            "z_image_turbo",
            contract,
            &z_image_fixture_manifest(),
            "q4",
            numeric_tier("z_image_turbo", "q4").expect("q4 tier"),
            &request_mode("z_image_turbo", "text_to_image"),
            None,
            Z_IMAGE_FIXTURE_GEOMETRY,
            (18.4 * BYTES_PER_GIB) as u64,
            0,
            Z_IMAGE_REQUEST_EVIDENCE_REVISION,
            anchors,
        )
    }

    /// The whole shared-image entry point, as `generate_candle_stream` reaches it for
    /// `z_image_turbo`, on a simulated card.
    fn evaluate_z_image_fixture(
        budget: VramBudget,
        store: &sceneworks_core::memory_anchor::MemoryAnchorStore,
    ) -> Option<CandleMemoryEvaluation> {
        evaluate_z_image_fixture_with(
            budget,
            reserve_for(Some(budget)),
            Some(z_image_ladder_anchors(store)),
        )
    }

    /// [`evaluate_z_image_fixture`] with the reserve and the anchor source spelled out: `None`
    /// anchors is the PRODUCTION source (`CandleLadderAnchors::packaged`), and `reserve_gb` is
    /// what the caller derived from its raw probe.
    fn evaluate_z_image_fixture_with(
        budget: VramBudget,
        reserve_gb: f64,
        anchors: Option<CandleLadderAnchors<'_>>,
    ) -> Option<CandleMemoryEvaluation> {
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("missing-z-image-q4")));
        evaluate_shared_image_inner(
            "z_image_turbo",
            "z_image_turbo",
            &spec,
            true,
            &z_image_fixture_manifest(),
            "q4",
            "text_to_image",
            None,
            None,
            Z_IMAGE_FIXTURE_GEOMETRY,
            false,
            false,
            false,
            false,
            Some(budget),
            reserve_gb,
            // What `generate_candle_stream` passes: `vram_gate::predicted_peak_gb`, the resident
            // row with the legacy headroom folded in.
            Some(18.4 + crate::vram_gate::HEADROOM_GB),
            0,
            MemoryCacheState::Cold,
            None,
            Some(z_image_fixture_contract()),
            None,
            anchors,
        )
        .expect("the fixture ladder evaluates")
    }

    fn rung_of(candidates: &[EstimateCandidate], strategy: MemoryStrategy) -> &EstimateCandidate {
        candidates
            .iter()
            .find(|candidate| candidate.selection.strategy == strategy)
            .unwrap_or_else(|| panic!("{strategy:?} must be synthesized"))
    }

    /// sc-22664 AC 1 (E4 + E7): `z_image_turbo` q4 text_to_image at 1024x1024, priced per rung
    /// from the sc-15859 staged anchor with the Z-Image facts, on a simulated 8 GB card (total
    /// 8.0, free 7.3) is SELECTED at a bounded rung — the fully engaged composition, whose derived
    /// peak (4.51 GB) is the first that fits — and the telemetry carries that rung and its three
    /// derived phase peaks, agreeing with the selector byte for byte.
    ///
    /// The reserve is the measured idle baseline (0.7 GB) plus its named margin, charged ONCE
    /// against the budget. MUTATION: re-introducing the double charge (adding `HEADROOM_GB` back
    /// into the candidates, or `reserved_headroom_gb: 2.0` on the budget while the floors carry
    /// it) lifts the admitted peak to 6.6 GB against a 5.3-6.35 GB budget and this arm reds.
    #[test]
    fn an_eight_gb_card_admits_z_image_q4_at_a_bounded_rung_and_the_telemetry_names_it() {
        let store = z_image_q4_store();
        let budget = VramBudget {
            free_gb: 7.3,
            total_gb: 8.0,
        };
        let evaluation =
            evaluate_z_image_fixture(budget, &store).expect("the 8 GB card admits a bounded rung");

        let selection = evaluation.context.selection;
        assert!(selection.strategy.is_optimized(), "{selection:?}");
        assert_eq!(
            selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
            "the fully engaged composition is the first rung whose derived peak fits"
        );
        assert_eq!(
            selection.parameters,
            gen_core::MemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(128),
                attention_chunk_size: Some(64 * 1024 * 1024),
                transformer_window_size: Some(1),
                transformer_window_component: None,
            }
        );
        assert_eq!(
            evaluation.basis,
            crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                lane: crate::memory_strategy::AnchorDerivationLane::Image,
            }
        );
        assert_eq!(
            evaluation.context.optimization_authority,
            gen_core::MemoryOptimizationAuthority::Estimated
        );
        let phases = evaluation
            .phase_peaks
            .expect("a law-priced selection carries its three phase peaks");
        assert_eq!(phases, Z_IMAGE_Q4_WINDOWED_PEAKS);
        // The peak the selector graded IS the max of the three phases: telemetry and selector
        // agree by construction (E7).
        assert_eq!(evaluation.context.predicted_peak_bytes, phases.peak_bytes());
        assert_eq!(evaluation.context.predicted_peak_bytes, 4_509_786_368);
        let memory = evaluation
            .memory
            .expect("an optimized selection carries request memory");
        assert!(memory.stage_residency && memory.tile_vae_decode && memory.chunk_attention);
        assert!(memory.stream_transformer_blocks);

        // The reserve: the probed idle baseline (0.7) is below the measured pre-load residency the
        // retained record carries, so that residency floors it (D3), plus the named margin, once.
        // An anchor-derived candidate is a reserve-free device delta, so its effective budget is
        // free minus exactly that, and the admitted peak is the raw derived peak widened by the
        // image-lane recapture spread — no headroom anywhere inside it.
        let reserve_gb = crate::vram_gate::ladder_reserve_gb(budget);
        assert!(
            (reserve_gb
                - (crate::vram_gate::MEASURED_PRELOAD_RESIDENCY_GB
                    + crate::vram_gate::LADDER_RESERVE_MARGIN_GB))
                .abs()
                < 1e-9
        );
        assert!(reserve_gb > 0.7 + crate::vram_gate::LADDER_RESERVE_MARGIN_GB);
        assert_eq!(evaluation.admitted.reserve_gb, reserve_gb);
        assert!((evaluation.admitted.available_gb - (7.3 - reserve_gb)).abs() < 1e-9);
        let admitted_bytes = (4_509_786_368f64
            * (1.0 + crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD))
            .ceil();
        assert!((evaluation.admitted.needed_gb - admitted_bytes / BYTES_PER_GIB).abs() < 1e-6);
        assert!(evaluation.admitted.needed_gb <= evaluation.admitted.available_gb);
        // …and the double charge would not have fit: the same derived peak with the legacy 2 GiB
        // headroom folded into the candidate AND the fixed 2 GiB reserve on the budget (what this
        // lane did before sc-22664) is 6.48 GiB admitted against 5.3 GiB effective. That is what
        // kept this card out; the mutation arms of the story restore exactly that and red here.
        let double_charged_bytes = (4_509_786_368f64
            + crate::vram_gate::HEADROOM_GB * BYTES_PER_GIB)
            * (1.0 + crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD);
        assert!(
            double_charged_bytes / BYTES_PER_GIB > 7.3 - crate::vram_gate::HEADROOM_GB,
            "the pre-sc-22664 accounting must not fit this card, or the fixture proves nothing"
        );
        assert_eq!(
            evaluation.context.budget.reserved_headroom_bytes,
            (reserve_gb * BYTES_PER_GIB).round() as u64
        );

        // E7: the event names the rung and its three derived phase peaks.
        let telemetry = evaluation.selection_telemetry("z_image_turbo", "q4");
        assert_eq!(telemetry["strategy"], "bounded_transformer_residency");
        assert_eq!(telemetry["basis"], "anchor_derived");
        assert_eq!(telemetry["authority"], "estimated");
        assert_eq!(
            telemetry["phasePeakBytes"],
            json!({
                "conditioning": Z_IMAGE_Q4_WINDOWED_PEAKS.conditioning,
                "denoise": Z_IMAGE_Q4_WINDOWED_PEAKS.denoise,
                "decode": Z_IMAGE_Q4_WINDOWED_PEAKS.decode,
            })
        );
        assert_eq!(telemetry["predictedPeakBytes"], 4_509_786_368u64);
        assert_eq!(telemetry["parameters"]["transformerWindowSize"], 1);
        assert_eq!(
            telemetry["parameters"]["attentionChunkSize"],
            64 * 1024 * 1024
        );
        assert_eq!(telemetry["reserveGb"], reserve_gb);
        assert_eq!(telemetry["availableGb"], evaluation.admitted.available_gb);
        assert_eq!(telemetry["admittedPeakGb"], evaluation.admitted.needed_gb);
    }

    /// sc-22664 AC 2 (E4): each deeper rung's admitted peak is STRICTLY below the staged one for
    /// this request — the staged rung is the anchor's own decode peak, the tiled rung moves
    /// decode, the chunked rung moves denoise, the windowed rung moves it again — and every rung
    /// carries the anchor-derived basis with its own three phase peaks. MUTATION: restoring the
    /// staged-floor reuse for deeper rungs prices all four at 11.74 GB and this arm reds.
    #[test]
    fn each_deeper_z_image_rung_prices_strictly_below_the_staged_one() {
        let store = z_image_q4_store();
        let contract = z_image_fixture_contract();
        let candidates = z_image_fixture_floors(z_image_ladder_anchors(&store), &contract);
        let staged = rung_of(&candidates, MemoryStrategy::StagedResidency);
        let tiled = rung_of(&candidates, MemoryStrategy::BoundedDecode);
        let chunked = rung_of(&candidates, MemoryStrategy::BoundedAttention);
        let windowed = rung_of(&candidates, MemoryStrategy::BoundedTransformerResidency);
        for candidate in [staged, tiled, chunked, windowed] {
            assert_eq!(
                candidate.basis,
                crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                    lane: crate::memory_strategy::AnchorDerivationLane::Image,
                },
                "{:?}",
                candidate.selection.strategy
            );
            let phases = candidate.phase_peaks.expect("law-priced");
            assert_eq!(candidate.evidence.predicted_peak_bytes, phases.peak_bytes());
        }
        // The staged rung at the anchor's own geometry and composition IS the anchor.
        assert_eq!(
            staged.phase_peaks.unwrap(),
            sceneworks_core::memory_anchor::AnchorDerivedPhases {
                conditioning: Z_IMAGE_Q4_STAGED_PEAKS.conditioning,
                denoise: Z_IMAGE_Q4_STAGED_PEAKS.denoise,
                decode: Z_IMAGE_Q4_STAGED_PEAKS.decode,
            }
        );
        let peak = |candidate: &EstimateCandidate| candidate.evidence.predicted_peak_bytes;
        assert_eq!(peak(staged), Z_IMAGE_Q4_STAGED_PEAKS.decode);
        // Strictly below the staged rung, every one of them…
        assert!(
            peak(tiled) < peak(staged),
            "{} vs {}",
            peak(tiled),
            peak(staged)
        );
        assert!(peak(chunked) < peak(staged));
        assert!(peak(windowed) < peak(staged));
        // …and strictly in order, because each rung's bound bites the phase that binds it.
        assert!(peak(chunked) < peak(tiled));
        assert!(peak(windowed) < peak(chunked));
        assert_eq!(windowed.phase_peaks.unwrap(), Z_IMAGE_Q4_WINDOWED_PEAKS);
        // The tiled rung's binding phase is still denoise (unchanged from the anchor); the
        // chunked rung's is the chunked denoise; the windowed rung's is decode.
        assert_eq!(peak(tiled), Z_IMAGE_Q4_STAGED_PEAKS.denoise);
        assert_eq!(
            peak(chunked),
            Z_IMAGE_Q4_COMPONENTS.transformer + 3_306_946_688 + 134_217_728
        );
        assert_eq!(peak(windowed), Z_IMAGE_Q4_WINDOWED_PEAKS.decode);
    }

    /// sc-22664 AC 3 (E4): with the same fixture, a budget below the deepest rung's admitted
    /// estimate returns `Selection::Reject` naming the needed and available figures, with the
    /// operational reserve charged exactly once: `available` is free minus the reserve and
    /// nothing else, `needed` is the deepest rung's widened derived peak with no headroom inside
    /// it. The end-to-end entry point then hands back to the legacy gates (`None`) rather than
    /// admitting.
    #[test]
    fn a_budget_below_the_deepest_z_image_rung_rejects_naming_needed_and_available() {
        let store = z_image_q4_store();
        let contract = z_image_fixture_contract();
        // A 6 GB card, total 6.0, free 5.0: idle baseline 1.0 + the margin leaves 3.75 GiB,
        // below the windowed rung's 4.20 GiB x 1.02 = 4.28 GiB admitted peak.
        let budget = VramBudget {
            free_gb: 5.0,
            total_gb: 6.0,
        };
        let reserve_gb = crate::vram_gate::ladder_reserve_gb(budget);
        assert!(reserve_gb < crate::vram_gate::HEADROOM_GB);

        let candidates = z_image_fixture_floors(z_image_ladder_anchors(&store), &contract);
        let live_closure_digest = sceneworks_core::memory_calibration::packaged_closure_digest(
            "candle",
            evidence_provider("z_image_turbo"),
        )
        .unwrap_or_default();
        // The resident live estimate the entry point submits alongside the floors: the raw
        // `vramGbByTier` row (18.4 GiB), shaped like the synthesized evidence.
        let resident_selection = MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: Default::default(),
            tier: numeric_tier("z_image_turbo", "q4").expect("q4 tier"),
        };
        let mut resident = rung_of(&candidates, MemoryStrategy::StagedResidency)
            .evidence
            .clone();
        resident.key.strategy = MemoryStrategy::Resident;
        resident.key.parameters = resident_selection.parameters;
        resident.key.engaged_composition = contract.engaged_composition(MemoryStrategy::Resident);
        resident.predicted_peak_bytes = (18.4 * BYTES_PER_GIB) as u64;
        let mut selector_candidates = vec![Candidate {
            selection: resident_selection,
            evidence: &resident,
            closure_digest: &live_closure_digest,
            basis: crate::memory_strategy::CandidateBasis::Measured,
            unmodeled_activation_bytes: None,
        }];
        selector_candidates.extend(candidates.iter().map(|candidate| Candidate {
            selection: candidate.selection,
            evidence: &candidate.evidence,
            closure_digest: &live_closure_digest,
            basis: candidate.basis,
            unmodeled_activation_bytes: None,
        }));
        let selected = crate::memory_strategy::select_strategy(
            RequestScope {
                resolved_route: "z_image_turbo",
                backend: "candle",
                tier: numeric_tier("z_image_turbo", "q4").expect("q4 tier"),
                mode: &request_mode("z_image_turbo", "text_to_image").scope_key,
                overlay: None,
                geometry: Z_IMAGE_FIXTURE_GEOMETRY,
                expected_closure_digest: &live_closure_digest,
            },
            &contract,
            Some(Budget {
                available_gb: budget.free_gb,
                reclaimable_gb: 0.0,
                total_gb: budget.total_gb,
                reserved_headroom_gb: reserve_gb,
            }),
            &selector_candidates,
        );
        let Selection::Reject {
            needed_gb,
            available_gb,
        } = selected
        else {
            panic!("a budget below the deepest rung must reject, got {selected:?}");
        };
        // The reserve, once: free minus the reserve, not free minus the reserve minus a headroom.
        assert!(
            (available_gb - (5.0 - reserve_gb)).abs() < 1e-9,
            "{available_gb}"
        );
        // Needed names the deepest rung's widened DERIVED peak, with no headroom inside it.
        let deepest = rung_of(&candidates, MemoryStrategy::BoundedTransformerResidency);
        let expected_needed = (deepest.evidence.predicted_peak_bytes as f64
            * (1.0 + crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD))
            .ceil()
            / BYTES_PER_GIB;
        assert!((needed_gb - expected_needed).abs() < 1e-6, "{needed_gb}");
        assert!(needed_gb > available_gb);
        assert!(
            needed_gb < available_gb + crate::vram_gate::HEADROOM_GB,
            "the reject must be decided by the derived peak against the single reserve, not by a \
             second headroom charge"
        );

        // End to end: the shared entry point admits nothing and hands back to the legacy gates.
        assert!(evaluate_z_image_fixture(budget, &store).is_none());
    }

    /// The selector, driven exactly as `evaluate_shared_image_inner` drives it for the Z-Image
    /// fixture: the resident live estimate (the caller's padded resident row) plus the
    /// synthesized floors, the reserve charged per `ReserveCharge::ExceptPadCarrying`.
    fn select_z_image_fixture(
        candidates: &[EstimateCandidate],
        contract: &gen_core::MemoryProviderContract,
        budget: VramBudget,
        reserve_gb: f64,
        resident_peak_bytes: u64,
    ) -> Selection {
        let live_closure_digest = sceneworks_core::memory_calibration::packaged_closure_digest(
            "candle",
            evidence_provider("z_image_turbo"),
        )
        .unwrap_or_default();
        let resident_selection = MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: Default::default(),
            tier: numeric_tier("z_image_turbo", "q4").expect("q4 tier"),
        };
        let mut resident = rung_of(candidates, MemoryStrategy::StagedResidency)
            .evidence
            .clone();
        resident.key.strategy = MemoryStrategy::Resident;
        resident.key.parameters = resident_selection.parameters;
        resident.key.engaged_composition = contract.engaged_composition(MemoryStrategy::Resident);
        resident.predicted_peak_bytes = resident_peak_bytes;
        let mut selector_candidates = vec![Candidate {
            selection: resident_selection,
            evidence: &resident,
            closure_digest: &live_closure_digest,
            basis: crate::memory_strategy::CandidateBasis::Measured,
            unmodeled_activation_bytes: None,
        }];
        selector_candidates.extend(candidates.iter().map(|candidate| Candidate {
            selection: candidate.selection,
            evidence: &candidate.evidence,
            closure_digest: &live_closure_digest,
            basis: candidate.basis,
            unmodeled_activation_bytes: None,
        }));
        let pad_carrying = |candidate: &Candidate<'_>| {
            std::ptr::eq(candidate.evidence, &resident)
                || candidate.basis == crate::memory_strategy::CandidateBasis::EstimateFloor
        };
        crate::memory_strategy::select_strategy_charging(
            RequestScope {
                resolved_route: "z_image_turbo",
                backend: "candle",
                tier: numeric_tier("z_image_turbo", "q4").expect("q4 tier"),
                mode: &request_mode("z_image_turbo", "text_to_image").scope_key,
                overlay: None,
                geometry: Z_IMAGE_FIXTURE_GEOMETRY,
                expected_closure_digest: &live_closure_digest,
            },
            contract,
            Some(Budget {
                available_gb: budget.free_gb,
                reclaimable_gb: 0.0,
                total_gb: budget.total_gb,
                reserved_headroom_gb: reserve_gb,
            }),
            &selector_candidates,
            crate::memory_strategy::ReserveCharge::ExceptPadCarrying(&pad_carrying),
        )
    }

    /// The epic's headline acceptance test (sc-22667, epic 22657 E4/E7), on the PRODUCTION path:
    /// the same `z_image_turbo` q4 request on the same 8 GB card (total 8.0, free 7.3), priced
    /// through the production anchor source (`CandleLadderAnchors::packaged`) — the packaged
    /// sc-15859 anchor AND the architecture facts read off the contract through
    /// `architecture_facts_from_contract` — is SELECTED at rung 4 (bounded transformer residency)
    /// at ≈4.51 GB, and the `image_memory_strategy_selected` telemetry names that rung and its
    /// three derived phase peaks in agreement with the selector.
    ///
    /// THE TWO HALVES OF THE UNLOCK, and why only the second could flip this test. sc-22666
    /// packaged the anchor: from then on the production source priced this cell from its own
    /// measured record, but with `ArchitectureFacts::default()` the law had no architecture to
    /// shrink a deeper rung by, so all four rungs priced at the anchor's measured staged decode
    /// peak (10.93 GiB) and the card REJECTED (the previous form of this test pinned exactly
    /// that). sc-22667 wires the facts: the block count makes rung 4's transformer share one
    /// thirtieth, the heads/patch/scale/width facts split the score tensor out of the denoise
    /// residue and price the 64 Mi-score chunk in its place, and the activation width prices the
    /// decode tile's blender floor — so rung 4's decode (4.51 GB) is the first peak that fits.
    /// Nothing in the ladder's accounting moved; the admission is attributable to the facts alone.
    ///
    /// THE TOLERANCE, justified. The expected phases are not restated: they are re-derived here
    /// from the packaged anchor with the law at rung 4's regime, and the selector and telemetry
    /// must equal them BYTE FOR BYTE (that is the E7 agreement). The headline figure is then
    /// asserted within 0.5 % of 4.51 GB (≈ ±22 MB) so a re-extraction of the same retained record
    /// that moves a rounding digit does not red the epic's number, while every other outcome the
    /// ladder could produce is GBs away: the chunked rung's 6.91 GB denoise, the staged rung's
    /// 8.05 GB, the unbounded decode's 11.74 GB, the padded manifest row's 8.27 GB.
    ///
    /// MUTATIONS: returning `ArchitectureFacts::default()` from `architecture_facts_from_contract`
    /// (the pre-sc-22667 seam) prices every rung at 10.93 GiB and reds the facts, rung and figure
    /// arms; dropping any one translated axis (`transformer_blocks: None`,
    /// `activation_dtype_width: None`) lifts rung 4 above the card and reds the selection; pricing
    /// a rung from the manifest row instead of the anchor (dropping the store from
    /// `CandleLadderAnchors::packaged`) puts every rung back on `EstimateFloor` and reds the basis
    /// arm.
    #[test]
    fn the_production_anchor_source_admits_z_image_q4_on_eight_gb_at_rung_four_from_the_contracts_facts(
    ) {
        use sceneworks_core::memory_anchor::{
            AnchorBackend, ArchitectureFacts, ImageDeriveRequest,
        };

        let contract = z_image_fixture_contract();
        let budget = VramBudget {
            free_gb: 7.3,
            total_gb: 8.0,
        };
        let reserve_gb = crate::vram_gate::ladder_reserve_gb(budget);
        let packaged = CandleLadderAnchors::packaged(&contract);
        // The facts flow: the production seam reads the contract's block, and it is the model's
        // real architecture — the same facts the explicit fixtures grade the law with.
        assert_eq!(
            packaged.facts, Z_IMAGE_FACTS,
            "the production seam must state the contract's architecture facts"
        );
        assert_ne!(packaged.facts, ArchitectureFacts::default());
        assert_eq!(packaged.facts.transformer_blocks, Some(30));
        // The anchor is the PACKAGED sc-15859 record (sc-22666) …
        assert!(
            packaged
                .store
                .expect("the packaged anchor store must load")
                .image_anchor_for("z_image_turbo", AnchorBackend::Candle, "q4")
                .is_some(),
            "sc-22666 packages the sc-15859 z_image_turbo candle corpus"
        );
        // … re-stamped at the loader-closure digest the pin currently DECLARES, exactly as
        // `vram_gate::tests::krea_live_anchor_store` does and for the same reason: whether
        // today's pin happens to leave the shipped row's digest current is a property of the pin
        // (an inference bump that touches a model's loader closure stales its packaged rows until
        // the anchors are re-extracted — the epic's terminal regeneration), graded honestly by
        // `sceneworks-core`'s `packaged_anchor_currency_is_reported_not_gated`, and not of the
        // derivation this test grades. The FACTS stay the production seam's.
        let live = z_image_live_anchor_store();
        let anchors = CandleLadderAnchors {
            store: Some(&live),
            facts: packaged.facts,
        };

        // The expected phases are DERIVED from the packaged anchor with the law at rung 4's own
        // regime, never restated: a re-capture that moves the measurement moves this expectation
        // with it.
        let anchor = live
            .image_anchor_for("z_image_turbo", AnchorBackend::Candle, "q4")
            .expect("the re-stamped store keeps the packaged z_image_turbo q4 row");
        let candidates = z_image_fixture_floors(anchors, &contract);
        assert!(!candidates.is_empty());
        let rung_4 = rung_of(&candidates, MemoryStrategy::BoundedTransformerResidency);
        let engaged = contract.engaged_composition_for_selection(&rung_4.selection);
        assert!(engaged.contains(&MemoryStrategy::StagedResidency));
        assert!(engaged.contains(&MemoryStrategy::BoundedDecode));
        assert!(engaged.contains(&MemoryStrategy::BoundedAttention));
        let expected = anchor
            .derive_phase_peaks(
                &ImageDeriveRequest::new(
                    Z_IMAGE_FIXTURE_GEOMETRY.width,
                    Z_IMAGE_FIXTURE_GEOMETRY.height,
                    request_regime(&engaged, &rung_4.selection.parameters)
                        .expect("rung 4's parameters translate to a regime"),
                ),
                crate::video_admission::anchor_component_bytes(contract.asset_facts),
                packaged.facts,
            )
            .expect("the packaged anchor prices rung 4 at its own geometry");
        for candidate in &candidates {
            assert_eq!(
                candidate.basis,
                crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                    lane: crate::memory_strategy::AnchorDerivationLane::Image,
                },
                "{:?}: the packaged sc-15859 anchor prices this cell since sc-22666",
                candidate.selection.strategy
            );
        }
        // The candidate the selector grades carries exactly the law's phases, and its peak is
        // their max (E7, at the candidate).
        assert_eq!(rung_4.phase_peaks, Some(expected));
        assert_eq!(rung_4.evidence.predicted_peak_bytes, expected.peak_bytes());
        // With the facts the deeper rungs price BELOW the staged one — the shrink the previous
        // form of this test proved impossible without them.
        let staged = rung_of(&candidates, MemoryStrategy::StagedResidency);
        assert!(
            rung_4.evidence.predicted_peak_bytes < staged.evidence.predicted_peak_bytes,
            "rung 4 ({}) must price below the staged rung ({})",
            rung_4.evidence.predicted_peak_bytes,
            staged.evidence.predicted_peak_bytes
        );
        // The headline figure: rung 4 at ≈4.51 GB (its tiled decode; see the doc for the band).
        let headline_bytes = 4.51e9;
        assert!(
            (expected.peak_bytes() as f64 - headline_bytes).abs() <= 0.005 * headline_bytes,
            "rung 4 must price at ≈4.51 GB, got {} bytes ({:?})",
            expected.peak_bytes(),
            expected
        );
        assert_eq!(
            expected.decode,
            expected.peak_bytes(),
            "decode is rung 4's binding phase"
        );

        // Selector level: SELECTED at rung 4, the first rung whose widened peak fits the
        // reserve-charged budget.
        let selected = select_z_image_fixture(
            &candidates,
            &contract,
            budget,
            reserve_gb,
            ((18.4 + crate::vram_gate::HEADROOM_GB) * BYTES_PER_GIB).ceil() as u64,
        );
        let Selection::Selected {
            selection,
            needed_gb,
            available_gb,
        } = selected
        else {
            panic!("an 8 GB card must admit rung 4 from the contract's facts, got {selected:?}");
        };
        assert_eq!(selection, rung_4.selection);
        assert_eq!(
            selection.strategy,
            MemoryStrategy::BoundedTransformerResidency
        );
        let expected_needed = (expected.peak_bytes() as f64
            * (1.0 + crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD))
            .ceil()
            / BYTES_PER_GIB;
        assert!((needed_gb - expected_needed).abs() < 1e-6, "{needed_gb}");
        // An anchor-derived candidate carries no structural pad, so it pays the reserve, once.
        assert!(
            (available_gb - (budget.free_gb - reserve_gb)).abs() < 1e-9,
            "{available_gb}"
        );
        assert!(needed_gb <= available_gb);

        // End to end through the shared-image entry point with the same anchors (the packaged
        // corpus re-stamped current, the production seam's facts): admitted at rung 4, and the
        // telemetry names the rung and the three derived phase peaks the selector graded — byte
        // for byte (E7).
        let evaluation = evaluate_z_image_fixture_with(budget, reserve_gb, Some(anchors))
            .expect("the 8 GB card is admitted at a bounded rung from the contract's facts");
        assert_eq!(evaluation.context.selection, rung_4.selection);
        assert_eq!(evaluation.phase_peaks, Some(expected));
        assert_eq!(
            evaluation.context.predicted_peak_bytes,
            expected.peak_bytes()
        );
        assert_eq!(
            evaluation.basis,
            crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                lane: crate::memory_strategy::AnchorDerivationLane::Image,
            }
        );
        let telemetry = evaluation.selection_telemetry("z_image_turbo", "q4");
        assert_eq!(telemetry["strategy"], "bounded_transformer_residency");
        assert_eq!(telemetry["basis"], "anchor_derived");
        assert_eq!(telemetry["parameters"]["transformerWindowSize"], 1);
        assert_eq!(
            telemetry["phasePeakBytes"],
            json!({
                "conditioning": expected.conditioning,
                "denoise": expected.denoise,
                "decode": expected.decode,
            })
        );
        assert_eq!(telemetry["predictedPeakBytes"], expected.peak_bytes());
        assert_eq!(telemetry["admittedPeakGb"], needed_gb);
        assert_eq!(telemetry["availableGb"], available_gb);
    }

    /// sc-22664 review D2: the reserve is derived from the RAW probe, never from the
    /// reclaimable-credited budget the ladder is handed. A warm 8 GB card whose resident model
    /// leaves 3.3 GiB free is credited to 7.8 GiB for the imminent evict; the reserve is still the
    /// raw probe's — 4.7 GiB of residency, capped at the legacy slack — and not the credited
    /// budget's 0.2 GiB idle (the measured floor plus the margin). MUTATION: deriving the reserve
    /// inside the ladder from `budget` (`ladder_reserve_gb(budget)` in place of the explicit
    /// parameter) reports the credited figure and reds this.
    #[test]
    fn the_reserve_is_derived_from_the_raw_probe_not_the_credited_budget() {
        let store = z_image_q4_store();
        let raw = VramBudget {
            free_gb: 3.3,
            total_gb: 8.0,
        };
        let credited = crate::vram_gate::with_reclaimable(raw, 4.5);
        assert!((credited.free_gb - 7.8).abs() < 1e-9);
        let raw_reserve_gb = crate::vram_gate::ladder_reserve_gb(raw);
        assert_eq!(raw_reserve_gb, crate::vram_gate::HEADROOM_GB);
        let credited_reserve_gb = crate::vram_gate::ladder_reserve_gb(credited);
        assert!(
            (credited_reserve_gb
                - (crate::vram_gate::MEASURED_PRELOAD_RESIDENCY_GB
                    + crate::vram_gate::LADDER_RESERVE_MARGIN_GB))
                .abs()
                < 1e-9
        );
        assert!(credited_reserve_gb < raw_reserve_gb);

        let evaluation = evaluate_z_image_fixture_with(
            credited,
            raw_reserve_gb,
            Some(z_image_ladder_anchors(&store)),
        )
        .expect("the credited card admits the windowed rung");
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency
        );
        assert_eq!(evaluation.admitted.reserve_gb, raw_reserve_gb);
        assert!(
            (evaluation.admitted.available_gb - (7.8 - raw_reserve_gb)).abs() < 1e-9,
            "{}",
            evaluation.admitted.available_gb
        );
        assert_eq!(
            evaluation.context.budget.reserved_headroom_bytes,
            (raw_reserve_gb * BYTES_PER_GIB).round() as u64
        );
        assert_eq!(
            evaluation.selection_telemetry("z_image_turbo", "q4")["reserveGb"],
            raw_reserve_gb
        );
    }

    /// sc-22664 review D4: a receipt-priced family near idle. SD3.5 large q4 on a 24 GB card
    /// with 23 GiB free: the reserve is 1.0 idle + the margin, and the structural resident floor
    /// (18 GiB of weights + 2 GiB headroom) fits — against the UNRESERVED pool, because a
    /// structural floor carries its pad. The single charge is proven by straddling both rungs: one
    /// hundredth above the widened resident floor selects Resident, one hundredth below drops to
    /// Staged; one hundredth above the widened staged floor still selects Staged, one hundredth
    /// below refuses. MUTATION: `ReserveCharge::EveryCandidate` in `evaluate_shared_image_inner`
    /// (charging the reserve against the floors too) drops the above-resident arm to Staged and
    /// refuses the above-staged arm — red.
    #[test]
    fn sd35_structural_floors_carry_their_pad_and_pay_no_reserve_near_idle() {
        let manifest = json!({ "candle": {} })
            .as_object()
            .expect("SD3.5 structural manifest")
            .clone();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("sealed-sd35-q4")))
            .with_resolved_route("sd3_5_large");
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let evaluate = |total_gb: f64, free_gb: f64| {
            let budget = VramBudget { free_gb, total_gb };
            evaluate_shared_image_inner(
                "sd3_5_large",
                "sd3_5_large",
                &spec,
                true,
                &manifest,
                "q4",
                "text_to_image",
                None,
                None,
                geometry,
                false,
                false,
                false,
                false,
                Some(budget),
                reserve_for(Some(budget)),
                None,
                0,
                MemoryCacheState::Cold,
                None,
                Some(sd35_probe_contract("sd3_5_large", None)),
                Some(SD35_REQUEST_EVIDENCE_REVISION),
                None,
            )
        };
        let reserve_gb = crate::vram_gate::ladder_reserve_gb(VramBudget {
            free_gb: 23.0,
            total_gb: 24.0,
        });
        assert!((reserve_gb - (1.0 + crate::vram_gate::LADDER_RESERVE_MARGIN_GB)).abs() < 1e-9);

        let near_idle = evaluate(24.0, 23.0)
            .expect("sealed SD3.5 receipt evaluates")
            .expect("the resident envelope fits a near-idle 24 GB card");
        assert_eq!(
            near_idle.context.selection.strategy,
            MemoryStrategy::Resident
        );
        assert_eq!(near_idle.admitted.reserve_gb, reserve_gb);
        assert_eq!(
            near_idle.admitted.available_gb, 23.0,
            "a structural floor carries its pad and compares against the UNRESERVED pool"
        );
        let resident_threshold_gb = near_idle.admitted.needed_gb;
        // 18 GiB of weights + 2 GiB headroom, graded as the live resident estimate (a measured-
        // current subject carries no allowance).
        assert_eq!(resident_threshold_gb, 20.0);

        let above_resident = evaluate(24.0, resident_threshold_gb + 0.01)
            .expect("sealed SD3.5 receipt evaluates")
            .expect("the widened resident floor fits at its own threshold");
        assert_eq!(
            above_resident.context.selection.strategy,
            MemoryStrategy::Resident
        );
        let below_resident = evaluate(24.0, resident_threshold_gb - 0.01)
            .expect("sealed SD3.5 receipt evaluates")
            .expect("the staged envelope still fits just under the resident one");
        assert_eq!(
            below_resident.context.selection.strategy,
            MemoryStrategy::StagedResidency
        );
        let staged_threshold_gb = below_resident.admitted.needed_gb;
        // The 10 GiB DiT + 2 GiB headroom staged envelope, widened by the recapture spread.
        assert!(
            (staged_threshold_gb
                - 12.0 * (1.0 + crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD))
                .abs()
                < 1e-6,
            "{staged_threshold_gb}"
        );
        assert!(staged_threshold_gb < resident_threshold_gb);
        assert_eq!(
            below_resident.admitted.available_gb,
            resident_threshold_gb - 0.01
        );

        let above_staged = evaluate(24.0, staged_threshold_gb + 0.01)
            .expect("sealed SD3.5 receipt evaluates")
            .expect("the widened staged floor fits at its own threshold");
        assert_eq!(
            above_staged.context.selection.strategy,
            MemoryStrategy::StagedResidency
        );
        let no_fit = match evaluate(24.0, staged_threshold_gb - 0.01) {
            Err(error) => error,
            Ok(_) => panic!("a budget below the widened staged floor must refuse"),
        };
        assert!(no_fit.to_string().contains("no exact resident or staged"));
    }

    /// The contract-only path (sc-22664): with NO anchor for the cell, the staged rung is the
    /// manifest staged row unscaled, and each deeper staged composition is the law's ratios over
    /// that row — the row decomposed against the contract's component bytes and scaled by the
    /// rung's own regime — so the ladder still prices per rung instead of reusing the row. Every
    /// candidate carries the floor basis, because the row is the basis. Under the default facts
    /// (this pin) the ratios are inert and the deeper rungs sit AT the row, which
    /// `deep_rung_floors_follow_the_engaged_composition_not_the_rung_ordinal` pins.
    #[test]
    fn without_an_anchor_the_deeper_rungs_take_the_laws_ratios_over_the_manifest_row() {
        let contract = z_image_fixture_contract();
        let candidates = z_image_fixture_floors(
            CandleLadderAnchors {
                store: None,
                facts: Z_IMAGE_FACTS,
            },
            &contract,
        );
        let staged_row_bytes = (5.7 * BYTES_PER_GIB).ceil() as u64;
        // A manifest-row floor carries the structural pad INSIDE its peak (sc-22664 D1): the law
        // decomposes the raw row, and the pad is folded back over every rung's derivation.
        let headroom_bytes = (crate::vram_gate::HEADROOM_GB * BYTES_PER_GIB).ceil() as u64;
        let padded_row_bytes =
            ((5.7 + crate::vram_gate::HEADROOM_GB) * BYTES_PER_GIB).ceil() as u64;
        assert_eq!(padded_row_bytes, staged_row_bytes + headroom_bytes);
        let staged = rung_of(&candidates, MemoryStrategy::StagedResidency);
        let tiled = rung_of(&candidates, MemoryStrategy::BoundedDecode);
        let chunked = rung_of(&candidates, MemoryStrategy::BoundedAttention);
        let windowed = rung_of(&candidates, MemoryStrategy::BoundedTransformerResidency);
        for candidate in [staged, tiled, chunked, windowed] {
            assert_eq!(
                candidate.basis,
                crate::memory_strategy::CandidateBasis::EstimateFloor,
                "{:?}: the manifest row is the basis, not a measured anchor",
                candidate.selection.strategy
            );
        }
        let peak = |candidate: &EstimateCandidate| candidate.evidence.predicted_peak_bytes;
        assert_eq!(
            peak(staged),
            padded_row_bytes,
            "the staged rung is the padded row, unscaled"
        );
        let phases = |candidate: &EstimateCandidate| candidate.phase_peaks.expect("law-scaled");
        // The row is phase-blind, so it is read as every phase's peak…
        assert_eq!(
            phases(staged),
            sceneworks_core::memory_anchor::AnchorDerivedPhases {
                conditioning: staged_row_bytes,
                denoise: staged_row_bytes,
                decode: staged_row_bytes,
            }
        );
        // …and each deeper rung's ratio bites the phase it bounds: the tile moves decode, the
        // chunk moves denoise, the window moves denoise again.
        assert!(phases(tiled).decode < phases(staged).decode);
        assert_eq!(phases(tiled).denoise, phases(staged).denoise);
        assert!(phases(chunked).denoise < phases(tiled).denoise);
        assert!(phases(windowed).denoise < phases(chunked).denoise);
        assert_eq!(phases(windowed).decode, phases(tiled).decode);
        // No rung bounds CONDITIONING, and the row prices it at the whole row, so the admission
        // peak of every contract-only rung stays AT the row: a phase-blind floor cannot promise a
        // saving the anchor would show, and the ladder does not invent one. (The measured anchor
        // is what moves admission — `each_deeper_z_image_rung_prices_strictly_below_the_staged_one`.)
        for candidate in [tiled, chunked, windowed] {
            assert_eq!(phases(candidate).conditioning, staged_row_bytes);
            assert_eq!(
                peak(candidate),
                padded_row_bytes,
                "{:?}: the law-scaled row is still a manifest-row floor and carries the pad",
                candidate.selection.strategy
            );
        }
        // Never below the components each phase keeps resident.
        let windowed_phases = phases(windowed);
        assert!(windowed_phases.conditioning >= Z_IMAGE_Q4_COMPONENTS.conditioning);
        assert!(windowed_phases.denoise >= Z_IMAGE_Q4_COMPONENTS.transformer.div_ceil(30));
        assert!(windowed_phases.decode >= Z_IMAGE_Q4_COMPONENTS.decoder);

        // sc-22666: "no anchor" is a property of the CELL, not of the store being absent. A store
        // that exists and carries rows — as the packaged one now does for every retained corpus —
        // but holds nothing for THIS cell must fall through to the same contract-only per-rung
        // ladder, not to a bare row repeated. The store below is the real z_image q4 anchor
        // relabelled onto another model, so the lookup misses on `model_id` alone.
        let mut foreign = z_image_q4_anchor();
        foreign.model_id = "krea_2_turbo".to_owned();
        foreign.route = "krea_2_turbo".to_owned();
        let foreign_store = sceneworks_core::memory_anchor::MemoryAnchorStore {
            schema_version: sceneworks_core::memory_anchor::MEMORY_ANCHOR_SCHEMA_VERSION,
            anchors: vec![foreign],
            analytic_only: Vec::new(),
            component_deltas: Vec::new(),
        };
        let absent_cell = z_image_fixture_floors(
            CandleLadderAnchors {
                store: Some(&foreign_store),
                facts: Z_IMAGE_FACTS,
            },
            &contract,
        );
        assert_eq!(absent_cell.len(), candidates.len());
        for (with_rows, without_store) in absent_cell.iter().zip(candidates.iter()) {
            assert_eq!(
                with_rows.selection.strategy,
                without_store.selection.strategy
            );
            assert_eq!(
                with_rows.basis, without_store.basis,
                "{:?}: a cell absent from a populated store is priced exactly as one with no \
                 store at all",
                with_rows.selection.strategy
            );
            assert_eq!(
                with_rows.phase_peaks, without_store.phase_peaks,
                "{:?}: the contract-only per-rung ladder, not a bare manifest scalar",
                with_rows.selection.strategy
            );
        }
    }

    /// The contract-only floors at an arbitrary geometry and manifest — the same call as
    /// `z_image_fixture_floors`, with the two inputs the sc-22667 pseudo-anchor test varies.
    fn z_image_fixture_floors_at(
        anchors: CandleLadderAnchors<'_>,
        contract: &gen_core::MemoryProviderContract,
        manifest: &JsonObject<String, Value>,
        geometry: MemoryGeometry,
    ) -> Vec<EstimateCandidate> {
        synthesize_estimate_floors(
            "z_image_turbo",
            "z_image_turbo",
            contract,
            manifest,
            "q4",
            numeric_tier("z_image_turbo", "q4").expect("q4 tier"),
            &request_mode("z_image_turbo", "text_to_image"),
            None,
            geometry,
            (18.4 * BYTES_PER_GIB) as u64,
            0,
            Z_IMAGE_REQUEST_EVIDENCE_REVISION,
            anchors,
        )
    }

    /// sc-22667 (epic 22657 feature-end round, E3/E5): the manifest staged row is a measurement
    /// at `candle.vramMeasuredPixels`, geometry-blind, so the pseudo-anchor the contract-only
    /// path builds from it is pinned at THAT geometry and the law scales its residue UP to the
    /// request before any rung fraction is taken. Graded at 2048x2048 with the Z-Image facts
    /// (the ratios are live) on a manifest measured at 1024x1024, with no anchor for the cell:
    ///
    /// * every rung prices at or above the pre-epic padded row (`sequentialPeakGb + HEADROOM`),
    ///   and the staged rung STRICTLY above it — a 2048² request is not the 1024² row;
    /// * every rung's phase peaks at 2048² are at or above the same rung's at 1024²
    ///   (monotone in geometry): a bounded rung's tile/chunk fraction is taken of the
    ///   request-sized residue, never of the measured one;
    /// * a manifest that states no `vramMeasuredPixels` prices identically to one stating the
    ///   documented 1024² default.
    ///
    /// MUTATION: pinning `floor_pseudo_anchor`'s geometry at the REQUEST geometry (the pre-fix
    /// code) makes the staged rung EQUAL the padded row at 2048² (first block reds) and scales
    /// the windowed rung's decode at 2048² BELOW its own 1024² decode — the on-device tile is a
    /// smaller fraction of a larger image (second block reds). Changing the fallback constant
    /// reds the third block.
    #[test]
    fn the_contract_only_pseudo_anchor_is_pinned_at_the_manifests_measured_geometry() {
        let contract = z_image_fixture_contract();
        let manifest = z_image_fixture_manifest();
        assert_eq!(
            manifest_measured_pixels(&manifest),
            1_048_576,
            "the fixture manifest measures its rows at 1024x1024"
        );
        let anchors = CandleLadderAnchors {
            store: None,
            facts: Z_IMAGE_FACTS,
        };
        let at = |edge: u32| MemoryGeometry {
            width: edge,
            height: edge,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let measured = z_image_fixture_floors_at(anchors, &contract, &manifest, at(1024));
        let large = z_image_fixture_floors_at(anchors, &contract, &manifest, at(2048));
        let padded_row_bytes =
            ((5.7 + crate::vram_gate::HEADROOM_GB) * BYTES_PER_GIB).ceil() as u64;
        let peak = |candidate: &EstimateCandidate| candidate.evidence.predicted_peak_bytes;
        let phases = |candidate: &EstimateCandidate| candidate.phase_peaks.expect("law-scaled");

        // 1. Every rung at 2048² is at or above the pre-epic padded row; the staged rung is
        //    strictly above it, because the row was measured at a quarter of the pixels.
        let staged_large = rung_of(&large, MemoryStrategy::StagedResidency);
        assert!(
            peak(staged_large) > padded_row_bytes,
            "the staged rung at 2048x2048 must scale ABOVE the 1024x1024 row ({}), got {}",
            padded_row_bytes,
            peak(staged_large)
        );
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            let candidate = rung_of(&large, strategy);
            assert_eq!(
                candidate.basis,
                crate::memory_strategy::CandidateBasis::EstimateFloor
            );
            assert!(
                peak(candidate) >= padded_row_bytes,
                "{strategy:?}: a 2048x2048 rung must never price below the padded row the \
                 pre-epic ladder charged, got {} against {}",
                peak(candidate),
                padded_row_bytes
            );
            // 2. Monotone in geometry, phase by phase: the fraction a bounded rung takes is of
            //    the request-sized residue.
            let small = phases(rung_of(&measured, strategy));
            let big = phases(candidate);
            assert!(
                big.conditioning >= small.conditioning
                    && big.denoise >= small.denoise
                    && big.decode >= small.decode,
                "{strategy:?}: every phase at 2048x2048 must be at or above the same rung at \
                 1024x1024, got {big:?} against {small:?}"
            );
        }
        // The windowed rung is where the pre-fix arithmetic went below the measured decode: its
        // 2048² decode is the tile blender floor plus a 3/16 band of a residue that the request-
        // pinned anchor never scaled up.
        let windowed_small = phases(rung_of(
            &measured,
            MemoryStrategy::BoundedTransformerResidency,
        ));
        let windowed_large = phases(rung_of(&large, MemoryStrategy::BoundedTransformerResidency));
        assert!(
            windowed_large.decode > windowed_small.decode,
            "the windowed rung's decode must GROW with the request ({} at 1024², {} at 2048²)",
            windowed_small.decode,
            windowed_large.decode
        );

        // 3. A manifest with no `vramMeasuredPixels` is read at the rows' documented 1024².
        let mut unstated = manifest.clone();
        unstated["candle"]
            .as_object_mut()
            .expect("candle block")
            .remove("vramMeasuredPixels");
        assert_eq!(
            manifest_measured_pixels(&unstated),
            MANIFEST_ROW_DEFAULT_MEASURED_PIXELS
        );
        let fallback = z_image_fixture_floors_at(anchors, &contract, &unstated, at(2048));
        for (stated, unstated) in large.iter().zip(fallback.iter()) {
            assert_eq!(stated.selection, unstated.selection);
            assert_eq!(
                stated.phase_peaks, unstated.phase_peaks,
                "{:?}: the unstated measured geometry falls back to 1024x1024",
                stated.selection.strategy
            );
        }

        // 4. (sc-22667 round-2 review, a) The DEFAULT facts — a contract stating none — where the
        //    law has no token ratio and no score split and scales the whole residue by pixels.
        //    This is the arm the advertised mutation reds unconditionally, and it pins the
        //    measured-geometry anchoring as ARITHMETIC rather than as an ordering: at 1024² the
        //    pseudo-anchor IS the request geometry, so the staged rung's three phases are the raw
        //    row itself; at 2048² each phase is the rung's resident set plus the row's residue
        //    over that set scaled by the pixel ratio — x1 for the prompt-shaped conditioning, x16
        //    for denoise (the factless law takes the worst-scaling quadratic term for growth) and
        //    x4 for the pixel-shaped decode. Pinned at the request geometry both sides read x1 and
        //    the 2048² phases collapse onto the row.
        let blind = CandleLadderAnchors {
            store: None,
            facts: sceneworks_core::memory_anchor::ArchitectureFacts::default(),
        };
        let measured_blind = z_image_fixture_floors_at(blind, &contract, &manifest, at(1024));
        let large_blind = z_image_fixture_floors_at(blind, &contract, &manifest, at(2048));
        let row = phases(rung_of(&measured_blind, MemoryStrategy::StagedResidency));
        assert_eq!(
            (row.conditioning, row.denoise),
            (row.decode, row.decode),
            "at the measured geometry the staged rung's three phases are the row itself"
        );
        let row_bytes = row.decode;
        assert!(row_bytes > Z_IMAGE_Q4_COMPONENTS.transformer);
        let components = Z_IMAGE_Q4_COMPONENTS;
        assert_eq!(
            phases(rung_of(&large_blind, MemoryStrategy::StagedResidency)),
            sceneworks_core::memory_anchor::AnchorDerivedPhases {
                conditioning: row_bytes,
                denoise: components.transformer + (row_bytes - components.transformer) * 16,
                decode: components.decoder + (row_bytes - components.decoder) * 4,
            },
            "the 2048x2048 staged rung is the row's residue over each phase's resident set, scaled \
             by the pixel ratio from the MEASURED geometry"
        );
        let staged_large_blind = rung_of(&large_blind, MemoryStrategy::StagedResidency);
        assert!(peak(staged_large_blind) > padded_row_bytes);
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            let candidate = rung_of(&large_blind, strategy);
            assert_eq!(
                candidate.basis,
                crate::memory_strategy::CandidateBasis::EstimateFloor
            );
            assert!(
                peak(candidate) >= padded_row_bytes,
                "{strategy:?} (default facts): a 2048x2048 rung must never price below the \
                 padded row, got {} against {}",
                peak(candidate),
                padded_row_bytes
            );
            let small = phases(rung_of(&measured_blind, strategy));
            let big = phases(candidate);
            assert!(
                big.conditioning >= small.conditioning
                    && big.denoise >= small.denoise
                    && big.decode >= small.decode,
                "{strategy:?} (default facts): every phase at 2048x2048 must be at or above the \
                 same rung at 1024x1024, got {big:?} against {small:?}"
            );
            // Without facts no rung fraction is live, so every rung is the staged rung: the
            // erring-large reading, and the reason the default-facts arm is where a request-
            // pinned anchor is most visible (nothing else moves the phases).
            assert_eq!(
                big,
                phases(staged_large_blind),
                "{strategy:?} (default facts): no ratio is live, so the rung is the staged one"
            );
        }
    }

    #[test]
    fn eligible_lens_selector_contract_can_select_sequential_without_mutating_the_load_spec() {
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 6.0 },
                "sequentialPeakGb": { "q4": 2.5 }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("missing-lens-q4")))
            .with_quant(Quant::Q4)
            .with_resolved_route("lens")
            .with_eligible_load_shape_declaration();
        let mut contract = composition_probe_contract(true, true);
        contract.provider_id = "lens".to_owned();
        let evaluation = evaluate_shared_image_inner(
            "lens",
            "lens",
            &spec,
            true,
            &manifest,
            "q4",
            "text_to_image",
            None,
            None,
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            false,
            false,
            false,
            false,
            Some(VramBudget {
                free_gb: 7.0,
                total_gb: 96.0,
            }),
            reserve_for(Some(VramBudget {
                free_gb: 7.0,
                total_gb: 96.0,
            })),
            Some(8.0),
            0,
            MemoryCacheState::Cold,
            None,
            Some(contract),
            None,
            None,
        )
        .expect("weights-free Lens selector")
        .expect("the staged estimate fits while resident does not");
        assert_ne!(
            evaluation.context.selection.strategy,
            MemoryStrategy::Resident
        );
        assert!(
            evaluation
                .memory
                .expect("optimized selection carries request memory")
                .stage_residency
        );
        assert_eq!(
            spec.load_shape_declaration_result,
            gen_core::LoadShapeDeclarationResult::Eligible
        );
        assert_eq!(spec.load_shape, gen_core::LoadShape::EagerMaterialization);
    }

    /// sc-18253: the staged manifest row prices the staged WORKING SET, so it is only a sound
    /// floor for a composition that actually engages `StagedResidency`. The gen-core engagement
    /// mechanism permits a provider to implement a deep rung whose engaged composition excludes
    /// staging — such a request runs whole-model resident, so its floor must clamp to the
    /// RESIDENT estimate (the candle mirror of the MLX floor's
    /// `engaged.contains(&StagedResidency)` max-vs-sum split) instead of under-predicting a whole
    /// resident working set behind only the estimate margin.
    ///
    /// Both mutation directions are pinned:
    ///  * deleting the composition check (every deep rung takes the staged row again) flips the
    ///    staging-free contract's deep-rung assertions red;
    ///  * inverting it (clamping every optimized floor to the resident row) flips the staged
    ///    rung's own assertion and the staging-bound contract's deep-rung assertions red.
    #[test]
    fn deep_rung_floors_follow_the_engaged_composition_not_the_rung_ordinal() {
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 6.0 },
                "sequentialPeakGb": { "q4": 2.5 },
                "supportsSequentialOffload": true
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let resident_peak_bytes = (8.0 * BYTES_PER_GIB) as u64;
        // The staged row plus its structural pad (sc-22664 D1): a manifest-row floor carries the
        // pad and is compared against the unreserved pool.
        let staged_floor_bytes = ((2.5 + crate::vram_gate::HEADROOM_GB) * BYTES_PER_GIB) as u64;
        assert_ne!(
            resident_peak_bytes, staged_floor_bytes,
            "the two floor sources must be distinguishable for the assertions below to bite"
        );
        let floors = |contract: &gen_core::MemoryProviderContract| {
            synthesize_estimate_floors(
                "z_image_turbo",
                "z_image_turbo",
                contract,
                &manifest,
                "q4",
                numeric_tier("z_image_turbo", "q4").expect("q4 tier"),
                &request_mode("z_image_turbo", "text_to_image"),
                None,
                geometry,
                resident_peak_bytes,
                0,
                Z_IMAGE_REQUEST_EVIDENCE_REVISION,
                no_anchors(),
            )
        };
        let floor_of = |synthesized: &[EstimateCandidate], strategy: MemoryStrategy| {
            synthesized
                .iter()
                .find(|candidate| candidate.selection.strategy == strategy)
                .map(|candidate| candidate.evidence.predicted_peak_bytes)
        };
        const DEEP_RUNGS: [MemoryStrategy; 3] = [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ];

        // Staging-free deep rungs (the gen-core default composition — no staged prerequisite
        // edges): the staged row must not price them.
        let synthesized = floors(&composition_probe_contract(true, false));
        assert_eq!(
            floor_of(&synthesized, MemoryStrategy::StagedResidency),
            Some(staged_floor_bytes),
            "the staged rung itself still takes the staged working-set row"
        );
        for deep in DEEP_RUNGS {
            assert_eq!(
                floor_of(&synthesized, deep),
                Some(resident_peak_bytes),
                "a {deep:?} composition that excludes staging runs whole-model resident and must \
                 clamp to the resident estimate, not the staged working-set row"
            );
        }

        // Control: a contract that binds every deep rung to staging (the shipped z-image shape)
        // keeps the staged working-set floor on those rungs — clamping regardless of composition
        // would flip these red.
        let synthesized = floors(&composition_probe_contract(true, true));
        for rung in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                floor_of(&synthesized, rung),
                Some(staged_floor_bytes),
                "a {rung:?} composition that engages staging keeps the staged working-set floor"
            );
        }
    }

    /// sc-22509 (epic 22505): where a measured anchor exists for the cell, the estimate floor is
    /// DERIVED from it and carries the `EstimateAnchorDerived` basis, instead of reading the
    /// hand-maintained `candle.sequentialPeakGb` row. The differential control in the same test —
    /// the identical call with no anchor store — pins that the manifest row is what the derivation
    /// displaces, so a derivation that silently reproduced the row could not pass both halves.
    #[test]
    fn an_anchored_cell_takes_the_derived_floor_and_an_unanchored_one_keeps_the_manifest_row() {
        use sceneworks_core::memory_anchor::{
            AnchorBackend, AnchorGeometry, AnchorLoadShape, AnchorMeasuredRegime, AnchorPhaseBytes,
            AnchorSource, ArchitectureFacts, ImageDeriveRequest, MemoryAnchor, MemoryAnchorStore,
            RequestRegime, MEMORY_ANCHOR_SCHEMA_VERSION,
        };
        let manifest = json!({
            "family": "krea_2",
            "candle": {
                "vramGbByTier": { "q4": 6.0 },
                "sequentialPeakGb": { "q4": 2.5 },
                "supportsSequentialOffload": true
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let geometry = MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let resident_peak_bytes = (8.0 * BYTES_PER_GIB) as u64;
        // The staged row plus its structural pad (sc-22664 D1): the manifest-row floor carries the
        // pad; only the anchor-derived candidate is reserve-free.
        let staged_floor_bytes = ((2.5 + crate::vram_gate::HEADROOM_GB) * BYTES_PER_GIB) as u64;
        let mut contract = composition_probe_contract(true, true);
        contract.provider_id = "krea_2_turbo".to_owned();
        contract.load_shape = gen_core::LoadShape::EagerMaterialization;
        contract.calibration = Some(gen_core::MemoryCalibrationIdentity {
            abi: gen_core::MEMORY_CALIBRATION_ABI,
            fingerprint: "anchor-seam-v1".to_owned(),
            load_shape: gen_core::LoadShape::EagerMaterialization,
        });
        // A hand-built store: this exercises the SELECTOR seam, not the store's own extraction
        // handshake (`load_memory_anchors`), which `sceneworks-core` owns and tests against the
        // retained evidence.
        //
        // Currency is the packaged loader closure since sc-22511, and `is_current` compares against
        // the PACKAGED declarations rather than anything injectable — so the control arm's digest
        // is READ from those declarations rather than frozen as a literal. A literal would be a
        // pin-coupled golden that reds on the next inference bump for no behavioural reason, and
        // would silently stop discriminating if the declaration were dropped.
        let live_loader_closure_digest =
            sceneworks_core::memory_anchor::packaged_anchor_loader_closures()
                .and_then(|closures| closures.digest_for("krea_2_turbo", AnchorBackend::Candle))
                .expect("krea_2_turbo:candle must declare a loader closure")
                .to_owned();
        let anchor = MemoryAnchor {
            id: "krea_2_turbo:candle:q4".to_owned(),
            model_id: "krea_2_turbo".to_owned(),
            model_family: "krea_2".to_owned(),
            route: "krea_2_turbo".to_owned(),
            provider: "krea_2_turbo".to_owned(),
            backend: AnchorBackend::Candle,
            tier: "q4".to_owned(),
            transformer_variant: None,
            decoder: None,
            mode: "text_to_image".to_owned(),
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
                path: "docs/generated/krea-candle-five-rung-sc-11045.json".to_owned(),
                sha256: String::new(),
                record_id: String::new(),
                calibration_fingerprint: "anchor-seam-v1".to_owned(),
                loader_closure_digest: live_loader_closure_digest.clone(),
            },
            geometry: AnchorGeometry {
                width: 1024,
                height: 1024,
                frames: 1,
                fps: None,
            },
            phase_active_peak_bytes: AnchorPhaseBytes {
                conditioning: 1_000_000_000,
                denoise: 3_000_000_000,
                decode: 4_000_000_000,
            },
            phase_allocator_envelope_bytes: None,
            overall_allocator_envelope_bytes: 4_000_000_000,
            underived_reason: None,
            component_bytes: None,
        };
        // Under the DEFAULT architecture facts (this pin — `architecture_facts_from_contract`)
        // every ratio the law could apply is inert, so each rung's derivation is the anchor's own
        // staged peak, and NO headroom is folded in (sc-22664: the selector budget charges the
        // reserve once). The staged-regime derivation is the number every rung must carry.
        let expected_derived = anchor
            .derive_phase_peaks(
                &ImageDeriveRequest::new(geometry.width, geometry.height, RequestRegime::staged()),
                crate::video_admission::anchor_component_bytes(contract.asset_facts),
                ArchitectureFacts::default(),
            )
            .expect("the anchor prices its own geometry")
            .peak_bytes();
        assert_eq!(expected_derived, 4_000_000_000);
        assert_ne!(
            expected_derived, staged_floor_bytes,
            "the two floor sources must be distinguishable for the assertions below to bite"
        );
        let store = MemoryAnchorStore {
            schema_version: MEMORY_ANCHOR_SCHEMA_VERSION,
            anchors: vec![anchor],
            // This fixture exercises the SELECTOR seam; the analytic-only half of the store
            // (sc-22510) is not read by it.
            analytic_only: Vec::new(),
            component_deltas: Vec::new(),
        };
        // Every guard of `candle_image_anchor` -- identity and loader-closure currency -- applies
        // to whatever store it is handed since sc-22666 (the per-store scope split went with the
        // model allow-list), so the injected store exercises all of them.
        let floors = |anchors: Option<&MemoryAnchorStore>| {
            synthesize_estimate_floors(
                "krea_2_turbo",
                "krea_2_turbo",
                &contract,
                &manifest,
                "q4",
                numeric_tier("krea_2_turbo", "q4").expect("q4 tier"),
                &request_mode("krea_2_turbo", "text_to_image"),
                None,
                geometry,
                resident_peak_bytes,
                0,
                Z_IMAGE_REQUEST_EVIDENCE_REVISION,
                CandleLadderAnchors {
                    store: anchors,
                    facts: ArchitectureFacts::default(),
                },
            )
        };

        // Every identity conjunct of `candle_image_anchor`, mutated one at a time on this same
        // hand-built store: each must knock the derivation back to the manifest-row floor. The
        // unmutated arm below is the control that keeps these from being vacuous.
        for (label, mutate) in [
            (
                "an overlay-measured anchor row",
                Box::new(|anchor: &mut MemoryAnchor| anchor.overlay = Some("lora".to_owned()))
                    as Box<dyn Fn(&mut MemoryAnchor)>,
            ),
            (
                "a reference-bearing anchor row",
                Box::new(|anchor: &mut MemoryAnchor| anchor.reference_count = 1),
            ),
            (
                "another route",
                Box::new(|anchor: &mut MemoryAnchor| anchor.route = "krea_2".to_owned()),
            ),
            (
                "another provider",
                Box::new(|anchor: &mut MemoryAnchor| anchor.provider = "someone_else".to_owned()),
            ),
            (
                "another mode",
                Box::new(|anchor: &mut MemoryAnchor| anchor.mode = "edit_image".to_owned()),
            ),
            (
                "another materialization shape",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.load_shape = AnchorLoadShape::DeferredMaterialization;
                }),
            ),
            (
                // THE currency conjunct since sc-22511: the model's own loader closure moved, so
                // the evidence no longer describes the code that will run. A rotated CALIBRATION
                // FINGERPRINT is deliberately absent from this list — see the arm below.
                "a moved loader closure",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.source.loader_closure_digest = "f".repeat(64);
                }),
            ),
        ] {
            let mut mutated = store.anchors[0].clone();
            mutate(&mut mutated);
            let mutated_store = MemoryAnchorStore {
                schema_version: MEMORY_ANCHOR_SCHEMA_VERSION,
                analytic_only: Vec::new(),
                component_deltas: Vec::new(),
                anchors: vec![mutated],
            };
            for candidate in floors(Some(&mutated_store)) {
                assert_eq!(
                    candidate.basis,
                    crate::memory_strategy::CandidateBasis::EstimateFloor,
                    "{label} must not be borrowed for this request ({:?})",
                    candidate.selection.strategy
                );
                assert_eq!(candidate.evidence.predicted_peak_bytes, staged_floor_bytes);
            }
        }
        // The INVERSE of the moved-closure arm, and the whole claim of sc-22511 (epic 22505 E9):
        // the calibration campaign is PROVENANCE, not currency. A rotated fingerprint AND an ABI
        // the runtime no longer speaks must both leave the derivation live, because neither one
        // says anything about whether the code that loads this model has moved. Before sc-22511
        // both of these demoted to the manifest-row floor; asserting the old behaviour here would
        // re-key currency onto the campaign through the test suite.
        {
            let mut rotated = store.anchors[0].clone();
            rotated.source.calibration_fingerprint = "anchor-seam-v2".to_owned();
            let rotated_store = MemoryAnchorStore {
                schema_version: MEMORY_ANCHOR_SCHEMA_VERSION,
                analytic_only: Vec::new(),
                component_deltas: Vec::new(),
                anchors: vec![rotated],
            };
            for candidate in floors(Some(&rotated_store)) {
                assert_eq!(
                    candidate.basis,
                    crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                        lane: crate::memory_strategy::AnchorDerivationLane::Image,
                    },
                    "a rotated calibration campaign is provenance and must not demote the anchor"
                );
                assert_eq!(candidate.evidence.predicted_peak_bytes, expected_derived);
            }

            let mut drifted = composition_probe_contract(true, true);
            drifted.provider_id = "krea_2_turbo".to_owned();
            drifted.load_shape = gen_core::LoadShape::EagerMaterialization;
            drifted.calibration = Some(gen_core::MemoryCalibrationIdentity {
                abi: gen_core::MEMORY_CALIBRATION_ABI + 1,
                fingerprint: "anchor-seam-v1".to_owned(),
                load_shape: gen_core::LoadShape::EagerMaterialization,
            });
            let drifted_floors = synthesize_estimate_floors(
                "krea_2_turbo",
                "krea_2_turbo",
                &drifted,
                &manifest,
                "q4",
                numeric_tier("krea_2_turbo", "q4").expect("q4 tier"),
                &request_mode("krea_2_turbo", "text_to_image"),
                None,
                geometry,
                resident_peak_bytes,
                0,
                Z_IMAGE_REQUEST_EVIDENCE_REVISION,
                CandleLadderAnchors {
                    store: Some(&store),
                    facts: ArchitectureFacts::default(),
                },
            );
            assert!(!drifted_floors.is_empty());
            for candidate in &drifted_floors {
                assert_eq!(
                    candidate.basis,
                    crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                        lane: crate::memory_strategy::AnchorDerivationLane::Image,
                    },
                    "a calibration ABI is provenance too and must not demote the anchor ({:?})",
                    candidate.selection.strategy
                );
            }
        }
        // …and `model_family` is deliberately NOT a conjunct: it has no source in the calibration
        // record, so keying on it would make an unvalidated field load-bearing for admission.
        {
            let mut relabelled = store.anchors[0].clone();
            relabelled.model_family = "not_the_catalog_family".to_owned();
            let relabelled_store = MemoryAnchorStore {
                schema_version: MEMORY_ANCHOR_SCHEMA_VERSION,
                analytic_only: Vec::new(),
                component_deltas: Vec::new(),
                anchors: vec![relabelled],
            };
            for candidate in floors(Some(&relabelled_store)) {
                assert_eq!(
                    candidate.basis,
                    crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                        lane: crate::memory_strategy::AnchorDerivationLane::Image,
                    },
                    "model_family must not gate the derivation"
                );
                assert_eq!(candidate.evidence.predicted_peak_bytes, expected_derived);
            }
        }
        // NO MODEL SCOPE (sc-22666, epic 22657 E5): `model_id` no longer gates the derivation.
        // A `CANDLE_ANCHOR_COEFFICIENT_MODELS` allow-list used to refuse exactly this row, because
        // the lane priced cells with per-pixel slopes fitted on Krea Turbo and another model would
        // have been priced with borrowed empirics. The law fits nothing since sc-22663 and every
        // retained corpus is packaged since sc-22666, so the catalog-wide store answers for
        // whichever cell it measured. The row below is the control with only `(model_id, route)`
        // moved -- the provider stays the contract's, so every other conjunct still passes -- and
        // it carries THAT model's own live loader-closure digest, so currency is satisfied and the
        // removed allow-list is the only thing that could have refused it.
        {
            let mut foreign = store.anchors[0].clone();
            foreign.model_id = "qwen_image".to_owned();
            foreign.route = "qwen_image".to_owned();
            foreign.source.loader_closure_digest =
                sceneworks_core::memory_anchor::packaged_anchor_loader_closures()
                    .and_then(|closures| closures.digest_for("qwen_image", AnchorBackend::Candle))
                    .expect("qwen_image:candle must declare a loader closure")
                    .to_owned();
            let foreign_store = MemoryAnchorStore {
                schema_version: MEMORY_ANCHOR_SCHEMA_VERSION,
                analytic_only: Vec::new(),
                component_deltas: Vec::new(),
                anchors: vec![foreign],
            };
            let foreign_floors = synthesize_estimate_floors(
                "qwen_image",
                "qwen_image",
                &contract,
                &manifest,
                "q4",
                numeric_tier("krea_2_turbo", "q4").expect("q4 tier"),
                &request_mode("krea_2_turbo", "text_to_image"),
                None,
                geometry,
                resident_peak_bytes,
                0,
                Z_IMAGE_REQUEST_EVIDENCE_REVISION,
                CandleLadderAnchors {
                    store: Some(&foreign_store),
                    facts: ArchitectureFacts::default(),
                },
            );
            assert!(!foreign_floors.is_empty());
            for candidate in &foreign_floors {
                assert_eq!(
                    candidate.basis,
                    crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                        lane: crate::memory_strategy::AnchorDerivationLane::Image,
                    },
                    "the anchor store is catalog-wide: a packaged model's own row must price \
                     its own cell ({:?})",
                    candidate.selection.strategy
                );
            }
        }

        let anchored = floors(Some(&store));
        assert!(
            !anchored.is_empty(),
            "the probe contract must implement optimized rungs"
        );
        for candidate in &anchored {
            assert_eq!(
                candidate.basis,
                crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                    lane: crate::memory_strategy::AnchorDerivationLane::Image,
                },
                "{:?} must be graded as an anchor derivation",
                candidate.selection.strategy
            );
            assert_eq!(
                candidate.evidence.predicted_peak_bytes, expected_derived,
                "{:?} must be priced by the derivation, not the manifest row",
                candidate.selection.strategy
            );
            let phases = candidate
                .phase_peaks
                .expect("a law-priced candidate reports its three phase peaks");
            assert_eq!(phases.peak_bytes(), expected_derived);
        }

        // Differential control: the identical call with no anchor keeps the manifest-row floor
        // and the floor basis. Under the default facts the contract-only path's ratios are all
        // inert, so every rung carries the raw staged row.
        let unanchored = floors(None);
        assert_eq!(unanchored.len(), anchored.len());
        for candidate in &unanchored {
            assert_eq!(
                candidate.basis,
                crate::memory_strategy::CandidateBasis::EstimateFloor,
                "{:?} must fall back to the manifest-row floor",
                candidate.selection.strategy
            );
            assert_eq!(
                candidate.evidence.predicted_peak_bytes, staged_floor_bytes,
                "{:?} must keep the staged manifest row",
                candidate.selection.strategy
            );
        }
    }

    /// The end-to-end arm of sc-18253: the exact contract the finding names — deep rungs
    /// implemented, no staging rung at all — must NOT admit a request on the staged row. Its
    /// floors clamp to the resident estimate, nothing fits below the resident peak, and the lane
    /// hands back to the established legacy gates (`None`). Before the composition check the
    /// staging-free `BoundedDecode` floor took the staged row and ADMITTED here at a
    /// whole-model-resident working set behind only the 4% estimate margin — deleting the check
    /// flips this arm red.
    #[test]
    fn a_staging_free_ladder_never_admits_on_the_staged_row() {
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 6.0 },
                "sequentialPeakGb": { "q4": 2.5 },
                "supportsSequentialOffload": true
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("missing-z-image-q4")));
        // A budget where the WIDENED staged-row floor fits but the resident estimate (8.0 GiB)
        // does not — recomputed from the policy margin, never a frozen literal.
        let staged_row_floor_gb = 2.5 + crate::vram_gate::HEADROOM_GB;
        let free_gb = staged_row_floor_gb
            * (1.0 + crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD)
            + crate::vram_gate::HEADROOM_GB
            + 0.3;
        assert!(
            free_gb - crate::vram_gate::HEADROOM_GB < 8.0,
            "the budget must stay below the resident estimate to discriminate"
        );
        // Driven through `evaluate_shared_image_inner`, which is the entry point `z_image_turbo`
        // actually reaches in production and the one the sibling request-axis probe already uses.
        // The bespoke wrapper is now an allowlist over the four registered bespoke authorities
        // (`pulid_flux`, the two Kolors routes, `sdxl`); `z_image_turbo` is not one of them, so it
        // refuses there before the ladder runs. The inner call preserves everything this arm
        // grades: the same probe contract is injected, and the evidence revision resolves to
        // Z-Image's own rather than being overridden.
        let evaluation = evaluate_shared_image_inner(
            "z_image_turbo",
            "z_image_turbo",
            &spec,
            true,
            &manifest,
            "q4",
            "text_to_image",
            None,
            None,
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            false,
            false,
            false,
            false,
            Some(VramBudget {
                free_gb,
                total_gb: 96.0,
            }),
            reserve_for(Some(VramBudget {
                free_gb,
                total_gb: 96.0,
            })),
            Some(8.0),
            0,
            MemoryCacheState::Cold,
            None,
            Some(composition_probe_contract(false, false)),
            None,
            None,
        )
        .expect("staging-free bespoke evaluation");
        assert!(
            evaluation.is_none(),
            "a staging-free deep rung must be graded at the resident clamp, not admitted on the \
             staged working-set row"
        );
    }

    /// An uncertified (imported / community) FLUX identity artifact never reaches an optimized
    /// rung — not through packaged records, and since sc-18097 not through estimate floors either.
    ///
    /// The budget is DELIBERATELY tight and the manifest populated (sc-18097 review): on a roomy
    /// card with an empty manifest the resident estimate wins for everyone, so the assertion could
    /// not tell a withheld floor from a floor that simply never competed. Here the certified
    /// control admits a floor rung at exactly the budget where the uncertified artifact must not —
    /// removing the certification conjunct from the floor guard turns the second arm red.
    #[test]
    fn uncertified_flux_identity_artifacts_fall_back_to_resident() {
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("alternate-flux-q4")))
            .with_ip_adapter(WeightsSource::File(PathBuf::from(
                "alternate-ip-adapter.safetensors",
            )))
            .with_component(
                "flux_ip_image_encoder",
                WeightsSource::Dir(PathBuf::from("alternate-image-encoder")),
            )
            .with_offload_policy(gen_core::OffloadPolicy::Sequential);
        // Staged floor = `sequentialPeakGb` 2.5 + 2.0 headroom = 4.5 GiB, widened by the 4% candle
        // estimate margin to 4.68; the resident estimate is 8.0. A 7 GiB budget (5.0 effective)
        // therefore separates "floor available" from "resident only".
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 6.0 },
                "sequentialPeakGb": { "q4": 2.5 },
                "supportsSequentialOffload": true
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let evaluate = |artifact_is_certified: bool, free_gb: f64| {
            evaluate_shared_image(
                "flux1_dev",
                "flux_dev",
                &spec,
                artifact_is_certified,
                &manifest,
                "q4",
                "character_image",
                Some("identity"),
                MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    reference_count: 1,
                },
                true,
                false,
                false,
                false,
                Some(VramBudget {
                    free_gb,
                    total_gb: 32.0,
                }),
                reserve_for(Some(VramBudget {
                    free_gb,
                    total_gb: 32.0,
                })),
                Some(8.0),
                0,
                MemoryCacheState::Cold,
            )
            .expect("FLUX identity evaluation")
        };

        // Roomy card: resident for both, with the typed scope preserved.
        let evaluation = evaluate(false, 32.0).expect("resident fallback");
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::Resident
        );
        assert_eq!(
            evaluation.context.mode,
            MemoryMode::Other("character_image".to_owned())
        );
        assert_eq!(evaluation.context.overlay.as_deref(), Some("identity"));
        assert!(evaluation.memory.is_none());

        // The discriminating pair at 7 GiB free: certified engages the staged floor…
        let certified = evaluate(true, 7.0)
            .expect("a certified artifact's estimate floor must admit at this budget");
        assert_eq!(
            certified.context.selection.strategy,
            MemoryStrategy::StagedResidency,
            "the control arm must actually reach a rung, or the assertion below proves nothing"
        );
        // …and uncertified gets no floors at all, so the lane hands back to the established gates.
        assert!(
            evaluate(false, 7.0).is_none(),
            "an uncertified artifact must not engage a rung off the certified manifest's rows"
        );
    }

    #[test]
    fn edit_alias_enters_turbo_contract_as_an_edit_reference_scope() {
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": 8.0 },
                "supportsSequentialOffload": true,
                "memoryStrategyCapabilities": {
                    "bounded_decode": {
                        "parameters": { "decodeTileEdge": 512, "decodeOverlap": 128 },
                        "overlays": ["none", "lora"]
                    }
                }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("missing-z-image-q4")));
        let evaluation = evaluate_shared_image(
            "z_image_turbo",
            "z_image_edit",
            &spec,
            true,
            &manifest,
            "q4",
            "edit_image",
            None,
            MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 1,
            },
            true,
            false,
            false,
            false,
            Some(VramBudget {
                free_gb: 32.0,
                total_gb: 32.0,
            }),
            reserve_for(Some(VramBudget {
                free_gb: 32.0,
                total_gb: 32.0,
            })),
            Some(8.0),
            0,
            MemoryCacheState::Cold,
        )
        .unwrap()
        .expect("resident alias selection");
        assert_eq!(
            evaluation.context.selection.strategy,
            MemoryStrategy::Resident
        );
        assert_eq!(evaluation.context.mode, MemoryMode::Edit);
        assert!(evaluation.context.has_reference);
        assert_eq!(
            evaluation.context.calibration_fingerprint,
            "z-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v2"
        );
        assert!(evaluation.memory.is_none());
    }

    #[test]
    fn hires_fix_does_not_reuse_one_geometry_scope_for_both_passes() {
        let manifest = JsonObject::new();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("missing-z-image-q4")));
        let evaluation = evaluate_shared_image(
            "z_image",
            "z_image",
            &spec,
            true,
            &manifest,
            "q4",
            "text_to_image",
            None,
            MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            false,
            false,
            true,
            false,
            Some(VramBudget {
                free_gb: 32.0,
                total_gb: 32.0,
            }),
            reserve_for(Some(VramBudget {
                free_gb: 32.0,
                total_gb: 32.0,
            })),
            Some(8.0),
            0,
            MemoryCacheState::Cold,
        )
        .unwrap();
        assert!(evaluation.is_none());

        // The route MUST be the probe contract's own provider (`z_image_turbo`), the same pairing
        // the sibling `evaluate_shared_bespoke_image` probe uses. `candidate_exclusion` fails closed
        // on `request.resolved_route != contract.provider_id` with a structural `Invalid`, so a
        // mismatched pair excludes every candidate and returns `None` — which would have made the
        // two `is_none()` assertions below pass for a reason that has nothing to do with the
        // Hires.fix / phases axes they exist to probe.
        let evaluate_axes = |worker_multipass: bool, request_has_phases: bool| {
            evaluate_shared_image_inner(
                "z_image_turbo",
                "z_image_turbo",
                &spec,
                true,
                &manifest,
                "q4",
                "text_to_image",
                None,
                None,
                MemoryGeometry {
                    width: 512,
                    height: 512,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
                false,
                false,
                worker_multipass,
                request_has_phases,
                Some(VramBudget {
                    free_gb: 32.0,
                    total_gb: 32.0,
                }),
                reserve_for(Some(VramBudget {
                    free_gb: 32.0,
                    total_gb: 32.0,
                })),
                Some(8.0),
                0,
                MemoryCacheState::Cold,
                None,
                Some(composition_probe_contract(true, false)),
                None,
                None,
            )
            .expect("request-axis evaluation")
        };
        // Control arm first, or the two `is_none()` assertions below prove nothing: this fixture must
        // actually be selectable when neither axis is set.
        let plain = evaluate_axes(false, false)
            .expect("the control arm must reach a selection, or the None assertions are vacuous");
        assert!(!plain.context.has_phases);
        assert!(
            evaluate_axes(true, false).is_none(),
            "Hires.fix alone must remain outside one-scope declaration authority"
        );
        assert!(
            evaluate_axes(true, true).is_none(),
            "Hires.fix plus GenerationRequest phases cannot collapse into one scope"
        );
        let phases = evaluate_axes(false, true).expect("actual request phases stay selectable");
        assert!(phases.context.has_phases);
    }

    #[test]
    fn optimized_candidates_reserve_the_actual_runtime_adapter_bytes() {
        let mut evidence = MemoryEvidence {
            key: MemoryEvidenceKey {
                model_family: "z_image".to_owned(),
                resolved_route: "z_image".to_owned(),
                backend: gen_core::MemoryBackend::Candle,
                tier: numeric_tier("z_image", "q4").expect("q4 is a supported numeric tier"),
                load_shape: gen_core::LoadShape::EagerMaterialization,
                mode: gen_core::MemoryMode::TextToImage,
                reference_shape: gen_core::MemoryReferenceShape::None,
                overlay: Some("lora".to_owned()),
                geometry: MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    // Text-to-image fixture: no reference images (sc-17054).
                    reference_count: 0,
                },
                frames_per_second: None,
                strategy: MemoryStrategy::BoundedTransformerResidency,
                engaged_composition: vec![MemoryStrategy::BoundedTransformerResidency],
                parameters: Default::default(),
            },
            conformance: MemoryConformanceState::Verified,
            dimensions: MemoryEvidenceDimensions::VERIFIED,
            calibration_abi: 1,
            calibration_fingerprint: "fixture".to_owned(),
            sceneworks_revision: "fixture".to_owned(),
            inference_revision: "fixture".to_owned(),
            harness_version: "fixture".to_owned(),
            predicted_peak_bytes: 8 * 1024,
            observed_peak_bytes: Some(8 * 1024),
            parity: MemoryParityContract::Exact,
            parity_result: MemoryParityResult::Passed,
        };

        account_for_runtime_overlay_bytes(std::slice::from_mut(&mut evidence), 512);

        assert_eq!(evidence.predicted_peak_bytes, 8 * 1024 + 512);
    }

    // -------------------------------------------------------------------------------------
    // sc-22667 (epic 22657 E6): the ONE falsification pass of the image derivation law on the
    // candle five-rung batch path — z-image q4 and krea q4, derived vs measured per rung per phase.
    // -------------------------------------------------------------------------------------

    /// The committed measurement: `memory-candle-adapter run_batch` at inference a5f643ae on GPU 1
    /// (one model load per tier, five rungs in canonical order), plus the staged rung alone in a
    /// fresh process (`coldStagedControl`) and the LOADED provider contract's component bytes,
    /// architecture facts and published rung parameters (`providerContract`), captured by the
    /// same binary in the same process as the measurement.
    const SC_22667_FALSIFICATION_FIXTURE: &str = include_str!(
        "../../../docs/calibration/sc-22657/candle-five-rung-falsification-sc-22667.json"
    );

    /// Over-prediction bound per cell (derived / measured per phase), every rung, every phase.
    /// Chosen from the fixture, not carried over from the prior art's 2.5: the binding cell is
    /// z-image's window-1 DECODE at 3.93x (derived 4.51 GB over a measured 1.15 GB), the next
    /// Krea's window-1 decode at 2.27x. Both are the same mechanism — the anchor's staged decode
    /// still holds the DiT (z-image's tiled decode measures 4.50 GB two rungs earlier, 3.36 GB
    /// above the window rung's 1.15, i.e. the q4 DiT), so the decode residue the law tiles
    /// carries a transformer the window rung does not, and the 3/8 host-transfer band
    /// (`DecodeTile::chunk_pixels`) keeps 3/8 of it. Every other cell sits at or below 1.47x.
    /// Raising this past the data keeps the test green; lowering it under 3.93 turns it red on
    /// that one cell.
    const SC_22667_OVER_PREDICTION_BOUND: f64 = 4.0;

    /// The under side a re-measure is allowed to sit at against the raw law: the lane's same-cell
    /// recapture spread, which admission charges on top of every image-lane anchor derivation
    /// (`ladder_margin_policy::CANDLE_RECAPTURE_SPREAD`). In the fixture the cold staged control
    /// lands between -3.6% (z-image conditioning, 2.989 GB against the anchor's 3.097) and +0.04%
    /// (z-image decode, 11.747 against 11.742), and the warm batch positions (rungs 4 and 5, after
    /// three generations in the same process) carry a conditioning phase up to +1.0% over the
    /// anchor (z-image 3.127 against 3.097).
    const SC_22667_RECAPTURE_BOUND: f64 =
        1.0 + crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD;

    /// FINDING (law): Krea's `bounded_decode` and `bounded_attention` rungs measure their tiled
    /// DECODE at 9.239 GB against a derived 8.706 (measured / derived 1.061 on both cells). The
    /// prior art skipped Krea's tiled decode phase outright ("the decode phase itself is asserted
    /// where the anchor's composition holds"); this pins it instead — the under-scaling must not
    /// deepen — and the rung's admission PEAK (its denoise, 15.10 GB derived over 14.91 measured)
    /// is asserted to bracket the measured overall like every other rung's.
    const SC_22667_KREA_TILED_DECODE_UNDERSCALE_BOUND: f64 = 1.07;

    /// Why a pinned cell sits under the raw law. Each class is asserted by its own witness, so a
    /// cell cannot be pinned without saying what it is.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Sc22667Under {
        /// A re-measure inside the recapture spread: a conditioning phase in a warm batch
        /// position (rungs 4 and 5), or the cold same-cell recapture of the staged anchor itself
        /// (z-image decode, 11.747 GB against the anchor's 11.742).
        WithinRecaptureSpread,
        /// z-image q4 window-1 DENOISE: derived 0.378 GB against a measured 2.993 (7.9x under).
        /// The contract's `transformer_bytes` (7.31 GB) is the q4 `transformer/` DIRECTORY, and
        /// gen-core's `safetensors_dir_bytes` (weightsmeta.rs at a5f643ae) recurses into the
        /// `.candle-device-format-v1/` cache it holds — 3.85 GB of `*.q4_1.safetensors` blocks on
        /// top of the 3.47 GB `model.safetensors`. Under that DiT the anchor's staged denoise
        /// residue (8.05 − 7.31 = 0.74 GB) is smaller than the full score tensor the law separates
        /// out (30 heads x 4608² x 2 B = 1.27 GB), the non-score residue collapses to zero, and the
        /// window rung prices at one block plus the chunk. WITNESS: the resident rung holds every
        /// component in every phase, and its measured conditioning (7.19 GB) is BELOW the
        /// contract's component total (9.99 GB) — a contract that over-reports. Krea, whose
        /// `loaded_asset_facts` prices the DiT from its own file, passes the same witness.
        /// Upstream (inference gen-core / candle-gen-z-image), not this repo; the pin flips red
        /// the day the facts are fixed so it is removed rather than forgotten.
        ContractOverReportsTransformer,
        /// See `SC_22667_KREA_TILED_DECODE_UNDERSCALE_BOUND`.
        KreaTiledDecodeUnderscaled,
    }

    /// The cells where the RAW law (no admission margin) prices under the measurement, as
    /// `(model, rung, phase, why)`, in grading order. See the headline test's FINDINGS. A cell
    /// that leaves this list (fixed) or joins it (regressed) fails the test either way.
    const SC_22667_RAW_UNDER_PREDICTIONS: &[(&str, &str, &str, Sc22667Under)] = &[
        (
            "z_image_turbo",
            "staged_residency",
            "decode",
            Sc22667Under::WithinRecaptureSpread,
        ),
        (
            "z_image_turbo",
            "bounded_attention",
            "conditioning",
            Sc22667Under::WithinRecaptureSpread,
        ),
        (
            "z_image_turbo",
            "bounded_transformer_residency",
            "conditioning",
            Sc22667Under::WithinRecaptureSpread,
        ),
        (
            "z_image_turbo",
            "bounded_transformer_residency",
            "denoise",
            Sc22667Under::ContractOverReportsTransformer,
        ),
        (
            "krea_2_turbo",
            "bounded_decode",
            "decode",
            Sc22667Under::KreaTiledDecodeUnderscaled,
        ),
        (
            "krea_2_turbo",
            "bounded_attention",
            "conditioning",
            Sc22667Under::WithinRecaptureSpread,
        ),
        (
            "krea_2_turbo",
            "bounded_attention",
            "decode",
            Sc22667Under::KreaTiledDecodeUnderscaled,
        ),
    ];

    fn sc_22667_strategy(key: &str) -> MemoryStrategy {
        match key {
            "resident" => MemoryStrategy::Resident,
            "staged_residency" => MemoryStrategy::StagedResidency,
            "bounded_decode" => MemoryStrategy::BoundedDecode,
            "bounded_attention" => MemoryStrategy::BoundedAttention,
            "bounded_transformer_residency" => MemoryStrategy::BoundedTransformerResidency,
            other => panic!("unknown rung key {other:?} in the sc-22667 fixture"),
        }
    }

    fn sc_22667_strategy_key(strategy: MemoryStrategy) -> &'static str {
        match strategy {
            MemoryStrategy::Resident => "resident",
            MemoryStrategy::StagedResidency => "staged_residency",
            MemoryStrategy::BoundedDecode => "bounded_decode",
            MemoryStrategy::BoundedAttention => "bounded_attention",
            MemoryStrategy::BoundedTransformerResidency => "bounded_transformer_residency",
        }
    }

    fn sc_22667_u32s(value: &Value) -> Vec<u32> {
        value
            .as_array()
            .expect("an array of integers")
            .iter()
            .map(|entry| u32::try_from(entry.as_u64().expect("an integer")).expect("a u32"))
            .collect()
    }

    fn sc_22667_opt_u32(value: &Value) -> Option<u32> {
        value
            .as_u64()
            .map(|entry| u32::try_from(entry).expect("a u32"))
    }

    /// One measured rung of a fragment: `[conditioning, denoise, decode]` device deltas in bytes.
    fn sc_22667_measured(fragment: &Value) -> [u64; 3] {
        let phase = |name: &str| {
            fragment["observedMemory"][name]["activeBytes"]
                .as_u64()
                .unwrap_or_else(|| panic!("observedMemory.{name}.activeBytes"))
        };
        [phase("conditioning"), phase("denoise"), phase("decode")]
    }

    /// The loaded provider contract, rebuilt from what the adapter recorded off the pinned
    /// provider in the measuring process: its published rung ranges, load shape, calibration
    /// identity, asset facts and architecture facts. The twin is checked against the recording —
    /// every rung's engaged composition must reproduce the one the provider itself measured under
    /// — so the derived side prices the SAME composition the measured side ran.
    fn sc_22667_contract(tier: &Value) -> gen_core::MemoryProviderContract {
        let recorded = &tier["providerContract"];
        let provider_id = recorded["providerId"].as_str().expect("providerId");
        let mut contract = gen_core::MemoryProviderContract::compatibility_default(
            provider_id,
            gen_core::MemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: true,
                host_to_device_block_materialization: true,
                block_materialization: gen_core::MemoryWindowMaterialization::DeviceFormatTransfer,
            },
        );
        contract.strategies = recorded["strategies"]
            .as_array()
            .expect("providerContract.strategies")
            .iter()
            .map(|capability| {
                let ranges = &capability["parameters"];
                assert_eq!(
                    capability["support"].as_str(),
                    Some("Implemented"),
                    "{provider_id}: the fixture measured every rung, so every rung is implemented"
                );
                gen_core::MemoryStrategyCapability {
                    strategy: sc_22667_strategy(capability["strategy"].as_str().expect("rung")),
                    support: gen_core::MemoryStrategySupport::Implemented,
                    parameters: gen_core::MemoryParameterRanges {
                        decode_tile_edges: sc_22667_u32s(&ranges["decodeTileEdges"]),
                        decode_overlaps: sc_22667_u32s(&ranges["decodeOverlaps"]),
                        attention_chunk_sizes: sc_22667_u32s(&ranges["attentionChunkSizes"]),
                        transformer_window_sizes: sc_22667_u32s(&ranges["transformerWindowSizes"]),
                        ..Default::default()
                    },
                }
            })
            .collect();
        assert_eq!(
            recorded["loadShape"].as_str(),
            Some("deferred_materialization")
        );
        contract.load_shape = gen_core::LoadShape::DeferredMaterialization;
        contract.calibration = Some(gen_core::MemoryCalibrationIdentity::new(
            recorded["calibrationFingerprint"]
                .as_str()
                .expect("calibrationFingerprint"),
            gen_core::LoadShape::DeferredMaterialization,
        ));
        contract.lifecycle = gen_core::MemoryLifecycleCapabilities {
            phases: vec![
                gen_core::MemoryPhase::Conditioning,
                gen_core::MemoryPhase::Denoise,
                gen_core::MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: true,
        };
        // Both providers bind their deeper rungs to staging in the same request (candle-gen-z-image
        // `memory_strategy.rs`, candle-gen-krea `lib.rs` at the pin); the composition check below
        // is what holds this twin to the recording.
        contract.additional_prerequisites = [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ]
        .into_iter()
        .map(|strategy| {
            (
                strategy,
                gen_core::MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            )
        })
        .collect();
        let assets = &recorded["assetFacts"];
        let bytes = |name: &str| {
            assets[name]
                .as_u64()
                .unwrap_or_else(|| panic!("assetFacts.{name}"))
        };
        contract.asset_facts = gen_core::MemoryAssetFacts {
            base_bytes: bytes("baseBytes"),
            conditioning_bytes: bytes("conditioningBytes"),
            transformer_bytes: bytes("transformerBytes"),
            decoder_bytes: bytes("decoderBytes"),
            overlay_bytes: bytes("overlayBytes"),
        };
        let facts = &recorded["architectureFacts"];
        contract.architecture_facts = gen_core::MemoryArchitectureFacts {
            attention_heads: sc_22667_opt_u32(&facts["attentionHeads"]),
            head_dim: sc_22667_opt_u32(&facts["headDim"]),
            transformer_blocks: sc_22667_opt_u32(&facts["transformerBlocks"]),
            patch_size: sc_22667_opt_u32(&facts["patchSize"]),
            latent_channels: sc_22667_opt_u32(&facts["latentChannels"]),
            vae_spatial_scale: sc_22667_opt_u32(&facts["vaeSpatialScale"]),
            vae_temporal_scale: sc_22667_opt_u32(&facts["vaeTemporalScale"]),
            activation_dtype_width: sc_22667_opt_u32(&facts["activationDtypeWidth"]),
        };
        for capability in recorded["strategies"].as_array().unwrap() {
            let strategy = sc_22667_strategy(capability["strategy"].as_str().unwrap());
            let engaged: Vec<&str> = contract
                .engaged_composition(strategy)
                .into_iter()
                .map(sc_22667_strategy_key)
                .collect();
            let recorded: Vec<&str> = capability["engagedRungs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|rung| rung.as_str().unwrap())
                .collect();
            assert_eq!(
                engaged, recorded,
                "{provider_id} {strategy:?}: the rebuilt contract must compose the rung exactly as \
                 the pinned provider measured it"
            );
        }
        contract
    }

    /// The packaged store with `model_id`'s candle rows re-stamped at the loader-closure digest
    /// the pin currently declares — the construction and rationale of `z_image_live_anchor_store`
    /// and `vram_gate::tests::krea_live_anchor_store`, for either model.
    fn sc_22667_live_anchor_store(
        model_id: &str,
    ) -> sceneworks_core::memory_anchor::MemoryAnchorStore {
        use sceneworks_core::memory_anchor::AnchorBackend;

        let store = sceneworks_core::memory_anchor::packaged_memory_anchors()
            .expect("the packaged anchor store")
            .clone();
        let digest = sceneworks_core::memory_anchor::packaged_anchor_loader_closures()
            .and_then(|closures| closures.digest_for(model_id, AnchorBackend::Candle))
            .unwrap_or_else(|| panic!("{model_id}:candle must declare a loader closure"))
            .to_owned();
        let anchors = store
            .anchors
            .into_iter()
            .map(|mut anchor| {
                if anchor.model_id == model_id && anchor.backend == AnchorBackend::Candle {
                    anchor.source.loader_closure_digest = digest.clone();
                }
                anchor
            })
            .collect();
        sceneworks_core::memory_anchor::MemoryAnchorStore { anchors, ..store }
    }

    /// The derived side of the falsification, per rung: `[conditioning, denoise, decode]` bytes
    /// priced through the PRODUCTION seam — `synthesize_estimate_floors` with the packaged anchor
    /// store, `architecture_facts_from_contract` and `anchor_component_bytes` on the recorded
    /// contract, `estimate_floor_parameters` picking the rung parameters — so every deeper rung
    /// is what the worker would submit for this cell. The resident rung, which the ladder never
    /// synthesizes (its live estimate is submitted on every request), is priced through the same
    /// law under `RequestRegime::resident()` from the same anchor, components and facts.
    fn sc_22667_derived(
        tier: &Value,
        contract: &gen_core::MemoryProviderContract,
        measured_overall: [u64; 2],
    ) -> Vec<(MemoryStrategy, [u64; 3])> {
        use sceneworks_core::memory_anchor::{AnchorBackend, ImageDeriveRequest};

        let model_id = tier["provider"].as_str().expect("provider");
        let geometry = MemoryGeometry {
            width: tier["geometry"]["width"].as_u64().unwrap() as u32,
            height: tier["geometry"]["height"].as_u64().unwrap() as u32,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let store = sc_22667_live_anchor_store(model_id);
        let anchors = CandleLadderAnchors {
            store: Some(&store),
            facts: crate::video_admission::architecture_facts_from_contract(contract),
        };
        // The manifest rows only feed the floors the ladder falls back to WITHOUT an anchor; with
        // the packaged anchor present every rung's phase peaks come from the law. They are stated
        // as the measured resident / staged overall peaks so the fixture carries no invented row.
        let [resident_overall, staged_overall] = measured_overall;
        let manifest = json!({
            "candle": {
                "vramGbByTier": { "q4": resident_overall as f64 / BYTES_PER_GIB },
                "vramMeasuredPixels": u64::from(geometry.width) * u64::from(geometry.height),
                "sequentialPeakGb": { "q4": staged_overall as f64 / BYTES_PER_GIB },
                "supportsSequentialOffload": true
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let floors = synthesize_estimate_floors(
            model_id,
            model_id,
            contract,
            &manifest,
            "q4",
            numeric_tier(model_id, "q4").expect("q4 tier"),
            &request_mode(model_id, "text_to_image"),
            None,
            geometry,
            resident_overall,
            0,
            "sc-22667-e6-falsification",
            anchors,
        );
        let anchor = store
            .image_anchor_for(model_id, AnchorBackend::Candle, "q4")
            .unwrap_or_else(|| panic!("{model_id}:candle:q4 must be packaged"));
        let components = crate::video_admission::anchor_component_bytes(contract.asset_facts);
        let resident_regime = request_regime(
            &[MemoryStrategy::Resident],
            &gen_core::MemoryStrategyParameters::default(),
        )
        .expect("the resident regime");
        let resident = anchor
            .derive_phase_peaks(
                &ImageDeriveRequest {
                    width: geometry.width,
                    height: geometry.height,
                    batch: 1,
                    conditioning_tokens: None,
                    regime: resident_regime,
                },
                components,
                anchors.facts,
            )
            .expect("the resident rung derives");
        let mut derived = vec![(
            MemoryStrategy::Resident,
            [resident.conditioning, resident.denoise, resident.decode],
        )];
        for strategy in MemoryStrategy::ALL.into_iter().filter(|s| s.is_optimized()) {
            let candidate = rung_of(&floors, strategy);
            assert!(
                matches!(
                    candidate.basis,
                    crate::memory_strategy::CandidateBasis::EstimateAnchorDerived {
                        lane: crate::memory_strategy::AnchorDerivationLane::Image
                    }
                ),
                "{model_id} {strategy:?}: the ladder must price this rung through the law, got {:?}",
                candidate.basis
            );
            let phases = candidate
                .phase_peaks
                .unwrap_or_else(|| panic!("{model_id} {strategy:?} carries its derived phases"));
            derived.push((
                strategy,
                [phases.conditioning, phases.denoise, phases.decode],
            ));
        }
        derived
    }

    /// E6, falsified once. For both tiers, every rung, every phase: the law's over-prediction is
    /// bounded by `SC_22667_OVER_PREDICTION_BOUND`; every rung's derived admission PEAK brackets
    /// its measured overall; and the raw law prices under the measurement in EXACTLY the pinned
    /// cells (`SC_22667_RAW_UNDER_PREDICTIONS`), each held to its class's witness. The staged rung
    /// is graded against its cold single-process control (the shape the packaged anchor was
    /// captured in); the batch's own staged row is graded as the carry-over it is.
    ///
    /// FINDING (batch path): rung 2 (`staged_residency`) runs second in a process whose
    /// `resident` rung just left the deferred-materialization DiT on the device, so its
    /// CONDITIONING phase carries that residency — z-image 6.95 GB and Krea 12.29 GB in the batch
    /// against 2.99 / 3.69 GB cold (Krea's staged DECODE likewise, 24.47 vs 22.35 GB). The law
    /// prices a request's own residency, not the previous request's leftovers, and the packaged
    /// anchors were measured cold, so the batch staged row is graded against the cold control and
    /// its carry-over is asserted to be exactly that: above the recapture spread, at most the
    /// DiT + VAE the resident rung could have left resident (plus the spread) — Krea's 8.60 GB
    /// carry sits 80 MB above its 8.52 GB DiT alone.
    #[test]
    fn sc_22667_candle_five_rung_batch_is_bracketed_by_the_derivation_law() {
        let fixture: Value =
            serde_json::from_str(SC_22667_FALSIFICATION_FIXTURE).expect("the fixture parses");
        assert_eq!(
            fixture["inferencePin"].as_str(),
            Some("a5f643ae58a4ed81e6be8280afbf29750da5ffe2")
        );
        let phases = ["conditioning", "denoise", "decode"];
        let mut graded_cells = 0usize;
        let mut raw_under: Vec<(&str, &str, &str)> = Vec::new();
        for tier in fixture["tiers"].as_array().expect("tiers") {
            let model_id = tier["provider"].as_str().unwrap();
            let fragments = tier["batch"]["fragments"]
                .as_array()
                .expect("five fragments");
            assert_eq!(
                fragments.len(),
                5,
                "{model_id}: one canonical five-rung batch"
            );
            assert_eq!(tier["batch"]["modelLoads"].as_u64(), Some(1));
            let measured: Vec<(MemoryStrategy, [u64; 3])> = fragments
                .iter()
                .map(|fragment| {
                    (
                        sc_22667_strategy(fragment["strategy"]["rung"].as_str().unwrap()),
                        sc_22667_measured(fragment),
                    )
                })
                .collect();
            let rungs: Vec<MemoryStrategy> = measured.iter().map(|(s, _)| *s).collect();
            assert_eq!(
                rungs,
                MemoryStrategy::ALL.to_vec(),
                "{model_id}: canonical rung order"
            );
            let cold_staged = sc_22667_measured(&tier["coldStagedControl"]["fragment"]);
            assert_eq!(
                tier["coldStagedControl"]["fragment"]["strategy"]["rung"].as_str(),
                Some("staged_residency")
            );
            let contract = sc_22667_contract(tier);
            let overall = |bytes: [u64; 3]| bytes.into_iter().max().unwrap();
            let derived = sc_22667_derived(
                tier,
                &contract,
                [overall(measured[0].1), overall(measured[1].1)],
            );
            let measured_resident_conditioning = measured[0].1[0];
            for ((strategy, measured_bytes), (derived_strategy, derived_bytes)) in
                measured.iter().zip(&derived)
            {
                assert_eq!(strategy, derived_strategy);
                let rung = sc_22667_strategy_key(*strategy);
                let batch_staged = *strategy == MemoryStrategy::StagedResidency;
                let graded = if batch_staged {
                    cold_staged
                } else {
                    *measured_bytes
                };
                // Admission compares the rung's PEAK: it brackets the measured overall everywhere
                // — the same-cell staged recapture inside the spread admission charges (z-image
                // decode re-measures 5 MB over its own anchor), every other rung raw.
                let peak_bound = if batch_staged {
                    SC_22667_RECAPTURE_BOUND
                } else {
                    1.0
                };
                assert!(
                    overall(*derived_bytes) as f64 * peak_bound >= overall(graded) as f64,
                    "{model_id} {rung}: derived peak {} under-predicts the measured overall {}",
                    overall(*derived_bytes),
                    overall(graded)
                );
                for (phase, (measured_phase, derived_phase)) in
                    phases.iter().zip(graded.iter().zip(derived_bytes))
                {
                    let ratio = *derived_phase as f64 / *measured_phase as f64;
                    eprintln!(
                        "sc-22667 {model_id} {rung} {phase}: measured {:.3} GB derived {:.3} GB \
                         ratio {ratio:.3}{}",
                        *measured_phase as f64 / 1e9,
                        *derived_phase as f64 / 1e9,
                        if batch_staged { " (cold control)" } else { "" }
                    );
                    assert!(
                        ratio <= SC_22667_OVER_PREDICTION_BOUND,
                        "{model_id} {rung} {phase}: derived {derived_phase} is {ratio:.3}x the \
                         measured {measured_phase}, over the {SC_22667_OVER_PREDICTION_BOUND}x bound"
                    );
                    if derived_phase < measured_phase {
                        raw_under.push((model_id, rung, phase));
                        let class = SC_22667_RAW_UNDER_PREDICTIONS
                            .iter()
                            .find(|(m, r, p, _)| (*m, *r, *p) == (model_id, rung, *phase))
                            .map(|(_, _, _, class)| *class)
                            .unwrap_or_else(|| {
                                panic!(
                                    "{model_id} {rung} {phase}: derived {derived_phase} \
                                     under-predicts the measured {measured_phase} and is not a \
                                     pinned finding"
                                )
                            });
                        match class {
                            Sc22667Under::WithinRecaptureSpread => assert!(
                                (*derived_phase as f64) * SC_22667_RECAPTURE_BOUND
                                    >= *measured_phase as f64,
                                "{model_id} {rung} {phase}: the warm re-measure {measured_phase} \
                                 lies outside the recapture spread of the derived {derived_phase}"
                            ),
                            Sc22667Under::ContractOverReportsTransformer => assert!(
                                contract.asset_facts.base_bytes > measured_resident_conditioning,
                                "{model_id}: the contract's component total {} no longer exceeds \
                                 the measured resident conditioning {} — the asset facts were \
                                 fixed upstream; drop this pin",
                                contract.asset_facts.base_bytes,
                                measured_resident_conditioning
                            ),
                            Sc22667Under::KreaTiledDecodeUnderscaled => assert!(
                                (*measured_phase as f64)
                                    <= (*derived_phase as f64)
                                        * SC_22667_KREA_TILED_DECODE_UNDERSCALE_BOUND,
                                "{model_id} {rung} {phase}: the tiled decode under-scaling \
                                 deepened — measured {measured_phase} over derived {derived_phase}"
                            ),
                        }
                    }
                    graded_cells += 1;
                }
            }
            // Krea passes the over-report witness (component total 11.79 GB under its measured
            // resident conditioning 12.73); z-image fails it (9.99 GB over 7.19) — see
            // `Sc22667Under::ContractOverReportsTransformer`.
            let over_reports = contract.asset_facts.base_bytes > measured_resident_conditioning;
            assert_eq!(
                over_reports,
                model_id == "z_image_turbo",
                "{model_id}: the resident-conditioning witness of an over-reporting contract moved"
            );
            // The batch staged row's carry-over, pinned as the finding it is.
            let batch_staged = measured[1].1;
            let carry = batch_staged[0].saturating_sub(cold_staged[0]) as f64;
            let spread =
                cold_staged[0] as f64 * crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD;
            // What the resident rung can leave behind for the next request's conditioning phase:
            // the DiT and the VAE. The text encoder is the staged rung's own conditioning-phase
            // residency and is priced by the anchor already.
            let leftover = (contract.asset_facts.transformer_bytes
                + contract.asset_facts.decoder_bytes) as f64;
            assert!(
                carry > spread,
                "{model_id}: the batch staged conditioning {} no longer carries the resident rung's \
                 residency over the cold {} — retire this finding and grade the batch row directly",
                batch_staged[0],
                cold_staged[0]
            );
            assert!(
                carry <= leftover + spread,
                "{model_id}: the batch staged conditioning carries {carry} bytes over the cold \
                 control, more than the {leftover} bytes of DiT + VAE the resident rung could \
                 have left resident"
            );
        }
        assert_eq!(
            graded_cells,
            2 * 5 * 3,
            "two tiers x five rungs x three phases"
        );
        let pinned: Vec<(&str, &str, &str)> = SC_22667_RAW_UNDER_PREDICTIONS
            .iter()
            .map(|(m, r, p, _)| (*m, *r, *p))
            .collect();
        assert_eq!(
            raw_under, pinned,
            "the set of cells the raw law prices under the measurement changed"
        );
    }
}
