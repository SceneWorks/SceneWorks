//! CANDLE turnkey capture owner.
use super::*;

pub(super) fn turnkey_candle_member(provider_id: &str) -> Option<TurnkeyCandleMember> {
    TURNKEY_CANDLE_MEMBERS
        .iter()
        .copied()
        .find(|member| member.provider_id == provider_id)
}

/// The production calibration identity the loaded turnkey generator publishes for one
/// `(member, tier)` cell on this lane — the tables `candle-gen-kolors::memory_strategy`,
/// `candle-gen-ideogram::memory_strategy` and `candle-gen-lens` `production_calibration_fingerprint`
/// mint (inference PR `story/sc-22732-epic-22723-memory-anchor-measurability`). The one measured
/// key, `lens_turbo` q4, is preserved byte-for-byte; every other cell is
/// `kolors-candle-kolors-<tier>-staged-chatglm-unet-f32-vae-v1`,
/// `ideogram4-candle-request-scoped-staged-residency-v1-<base|turbo>-<tier>` or
/// `lens-<base|turbo>-<tier>-candle-cuda-shared-ladder-v1`.
///
/// Written here as literals rather than read off the engines: the binary is `compile_error!` on
/// macOS and the candle crates are reachable only through `runtime_cuda`, so the plan/arm binding
/// has to be provable on the CPU-only host that writes the arm. At inference `c6d6a4db`
/// `candle-gen-lens` publishes the preserved key for every tier of BOTH routes,
/// `candle-gen-kolors` publishes `kolors-candle-staged-chatglm-unet-f32-vae-v1` for every tier and
/// `candle-gen-ideogram` publishes nothing; the five-rung capture refuses a loaded contract whose
/// identity differs from this table (`contractFingerprintMismatch`), so the two copies cannot
/// drift unnoticed once the epic's pin bump lands.
/// FAIL-CLOSED on BOTH axes (sc-22732 review). The tier arms used to bind `tier` as a free
/// variable, so any string at all — `"q2"`, `"nvfp4"`, a typo — minted a plausible-looking identity
/// this family's engines never publish, and `validate_turnkey_identity` would then accept a plan row
/// naming it. Only the three tiers the turnkey family ships are nameable; everything else is `None`,
/// which every caller turns into a refusal.
pub(super) fn turnkey_calibration_fingerprint(provider_id: &str, tier: &str) -> Option<String> {
    if !matches!(tier, "bf16" | "q4" | "q8") {
        return None;
    }
    let identity = match (provider_id, tier) {
        (LENS_TURBO_ID, "q4") => {
            "lens-candle-cuda-shared-ladder-device-format-blocks-v1".to_owned()
        }
        (KOLORS_ID, tier) => format!("kolors-candle-kolors-{tier}-staged-chatglm-unet-f32-vae-v1"),
        (IDEOGRAM_ID, tier) => {
            format!("ideogram4-candle-request-scoped-staged-residency-v1-base-{tier}")
        }
        (IDEOGRAM_TURBO_ID, tier) => {
            format!("ideogram4-candle-request-scoped-staged-residency-v1-turbo-{tier}")
        }
        (LENS_ID, tier) => format!("lens-base-{tier}-candle-cuda-shared-ladder-v1"),
        (LENS_TURBO_ID, tier) => format!("lens-turbo-{tier}-candle-cuda-shared-ladder-v1"),
        // sc-24112: `candle-gen-qwen-image-2-1`'s `memory_strategy::MEMORY_CALIBRATION_FINGERPRINT`,
        // ONE derived identity for every tier (the load shape is its separate typed axis).
        (QWEN_IMAGE_2_1_ID, _) => QWEN_IMAGE_2_1_CANDLE_FINGERPRINT.to_owned(),
        // sc-25907: the Turbo id's OWN single identity — it prices a different checkpoint, so a
        // base record can never satisfy a Turbo cell's handshake (E10).
        (QWEN_IMAGE_2_1_TURBO_ID, _) => QWEN_IMAGE_2_1_TURBO_CANDLE_FINGERPRINT.to_owned(),
        _ => return None,
    };
    Some(identity)
}

/// The weights-free conformance identities the three engines publish for a registry-behaviour
/// contract that loaded no weights (`kolors-candle-registry-behavior-v1` and
/// `lens-candle-registry-behavior-v1` with their `-<route>-…` suffixes; `candle-gen-ideogram`'s
/// weights-free paths publish no identity, and the `…-static-v1-<route>` literals the manifest used
/// to carry exist in no engine), plus the pre-sc-22732 Kolors constant that named every tier at
/// once. A plan row naming any of these could never be satisfied by a production load at the
/// inference head this table binds. Test-only: the sole caller is the plan-identity conformance
/// test.
#[cfg(test)]
pub(super) fn is_turnkey_weights_free_fingerprint(fingerprint: &str) -> bool {
    fingerprint.starts_with(runtime_cuda::providers::kolors::memory_strategy::STATIC_BEHAVIOR_FINGERPRINT)
        || fingerprint.starts_with(runtime_cuda::providers::lens::STATIC_BEHAVIOR_FINGERPRINT)
        // Retired plan namespace: this string is not an engine identity at the pinned source.
        || fingerprint.starts_with("ideogram4-candle-request-scoped-staged-residency-static-v1")
        // Retired pre-sc-22732 all-tier spelling, also absent from the current engine exports.
        || fingerprint == "kolors-candle-staged-chatglm-unet-f32-vae-v1"
}

/// The plan row must name the production identity this cell's loaded generator publishes —
/// checked against the weights-free table BEFORE any environment or weight work, so a row still
/// carrying a conformance string fails in milliseconds rather than after a multi-gigabyte load.
pub(super) fn validate_turnkey_identity(
    request: &Value,
    provider_id: &str,
    tier: &str,
) -> Result<(), String> {
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?;
    let expected_fingerprint = turnkey_calibration_fingerprint(provider_id, tier)
        .ok_or_else(|| format!("no turnkey member {provider_id} at tier {tier}"))?;
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, the {provider_id} \
             {tier} production identity is {expected_fingerprint}"
        ));
    }
    Ok(())
}

/// The fixture slug one turnkey still member's Candle fixtures carry, or `None` for a provider that
/// is not a member (sc-22732). Written as a lookup rather than folded into
/// [`validate_flux_one_fixture`] so a FLUX.1 fixture can never satisfy a turnkey plan by accident:
/// the two families bind different prefixes at different seeds.
pub(super) fn turnkey_fixture_slug(provider: &str) -> Option<&'static str> {
    match provider {
        KOLORS_ID => Some("kolors"),
        IDEOGRAM_ID => Some("ideogram-4"),
        IDEOGRAM_TURBO_ID => Some("ideogram-4-turbo"),
        LENS_ID => Some("lens"),
        LENS_TURBO_ID => Some("lens-turbo"),
        QWEN_IMAGE_2_1_ID => Some("qwen-image-2-1"),
        QWEN_IMAGE_2_1_TURBO_ID => Some("qwen-image-2-1-turbo"),
        _ => None,
    }
}

/// The turnkey fixture binds the member, the tier, the geometry edge, the seed and the step count.
/// Every member rides the shared five-rung reference render, so the seed is [`FIVE_RUNG_SEED`] and
/// the prefix is the five-rung one — a fixture naming the PuLID/FLUX.1 seed is refused, because the
/// record's fixture is the one claim about the render nothing downstream can re-derive.
pub(super) fn validate_turnkey_fixture(
    request: &Value,
    slug: &str,
    tier: &str,
) -> Result<(), String> {
    let fixture = protocol::planned_fixture(request)?;
    let (width, _) = protocol::target_geometry(request)?;
    let prefix = format!("{FIVE_RUNG_FIXTURE_PREFIX}{slug}-{tier}-{width}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse turnkey fixture seed {seed:?}: {error}"))?;
    if seed != FIVE_RUNG_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the seed the five-rung reference renders \
             at ({FIVE_RUNG_SEED})"
        ));
    }
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse turnkey fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(())
}
