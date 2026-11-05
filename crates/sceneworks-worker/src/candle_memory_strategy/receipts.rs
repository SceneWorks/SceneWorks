//! candle_memory_strategy receipts owner.
use super::*;

pub(super) fn is_chroma(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Chroma)
}

pub(super) fn is_ideogram(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Ideogram)
}

pub(super) fn is_sana(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Sana)
}

/// The FLUX 3 Image Klein provider ids, whose packages include the publisher FP8R/NVFP4 tiers.
pub(super) fn is_flux3(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Flux3)
}

pub(crate) fn is_sd35(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Sd35)
}

pub(super) fn is_sealed_kolors_bespoke(engine_id: &str) -> bool {
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

pub(super) fn sd35_provider_overlay_identity(
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

pub(super) fn validate_sd35_asset_facts(
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
pub(super) fn chroma_provider_overlay_identity(
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

pub(super) fn ideogram_provider_overlay_identity(
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

pub(super) fn validate_chroma_asset_facts(
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

pub(super) fn validate_ideogram_asset_facts(
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

pub(super) fn validate_sana_asset_facts(
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

pub(super) fn validate_kolors_asset_facts(
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

pub(super) fn chroma_base_phase_floor_bytes(contract: &gen_core::MemoryProviderContract) -> u64 {
    let facts = contract.asset_facts;
    facts
        .conditioning_bytes
        .max(facts.transformer_bytes)
        .max(facts.decoder_bytes)
}

pub(super) fn receipt_base_phase_floor_bytes(contract: &gen_core::MemoryProviderContract) -> u64 {
    chroma_base_phase_floor_bytes(contract)
}
