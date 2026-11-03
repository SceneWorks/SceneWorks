//! CANDLE flux one capture owner.
use super::*;

/// The FLUX.1 fixture binds the member, the tier, the geometry edge, the seed and the step count
/// — the MLX arm's `validate_flux_one_fixture`, on this lane's spellings. The base providers ride
/// the five-rung reference path (`fresh-five-rung-flux1-<route>-…-seed16402-step2`, at
/// [`FIVE_RUNG_SEED`]); PuLID is bespoke (`pulid-flux-face-sc16956-candle-…-seed22726-step2`, at
/// [`FLUX1_SEED`]). A fixture naming the other seed is refused: the record's fixture is the one
/// claim about the render that nothing downstream can re-derive.
pub(super) fn validate_flux_one_fixture(
    request: &Value,
    provider: &str,
    tier: &str,
) -> Result<(), String> {
    let (prefix, seed) = match provider {
        FLUX1_DEV_ID => ("fresh-five-rung-flux1-dev", FIVE_RUNG_SEED),
        FLUX1_SCHNELL_ID => ("fresh-five-rung-flux1-schnell", FIVE_RUNG_SEED),
        PULID_FLUX_ID => ("pulid-flux-face-sc16956-candle", FLUX1_SEED),
        other => {
            return Err(format!(
                "the Candle FLUX.1 fixture binding does not implement provider {other:?}"
            ))
        }
    };
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let (width, _) = protocol::target_geometry(request)?;
    let prefix = format!("{prefix}-{tier}-{width}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (planned_seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let planned_seed = planned_seed
        .parse::<u64>()
        .map_err(|error| format!("parse FLUX.1 fixture seed {planned_seed:?}: {error}"))?;
    if planned_seed != seed {
        return Err(format!(
            "planned.fixture seed {planned_seed} does not match the seed {provider} renders at \
             ({seed})"
        ));
    }
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse FLUX.1 fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(())
}

/// The tier a FLUX.1 base snapshot actually declares (`transformer/config.json`), read through
/// `candle-gen-flux`'s own resolver, compared against the PLANNED tier. The root suffix proved
/// the plan and the export agree on the directory NAME; this proves the weights inside agree
/// too — the worker's doctrine (`image_jobs/pulid_candle.rs`: "directory basenames are never
/// tier evidence; the packed transformer config is authoritative"), which the PuLID arm already
/// applied and the base arm did not.
pub(super) fn validate_flux_one_snapshot_tier(
    spec: &LoadSpec,
    provider_id: &str,
    tier: &str,
) -> Result<(), String> {
    let resolved =
        runtime_cuda::providers::flux::memory_strategy::resolved_numeric_tier(spec, provider_id)
            .map_err(|error| format!("resolve {provider_id} numeric tier: {error}"))?;
    let expected = numeric_tier(tier)?;
    if (resolved.precision, resolved.quant) != (expected.precision, expected.quant) {
        return Err(format!(
            "planned tier {tier} does not match the tier the {provider_id} snapshot declares \
             (precision={:?}, quant={:?})",
            resolved.precision, resolved.quant
        ));
    }
    Ok(())
}
