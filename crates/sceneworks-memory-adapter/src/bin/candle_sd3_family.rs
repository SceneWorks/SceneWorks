//! CANDLE sd3 capture owner.
use super::*;

/// The SD3.5 fixture binds the member, the tier, the geometry edge, the seed and the step count —
/// the MLX arm's `planned_sd3_seed`, on this lane's spellings.
///
/// All three members ride the shared five-rung reference path, so they render at [`FIVE_RUNG_SEED`]
/// and their fixtures must say so: `fresh-five-rung-sd3-5-<route>-<tier>-<edge>-seed16402-step2`.
/// A fixture naming the MLX arm's own seed is refused — the record's fixture is the one claim about
/// the render that nothing downstream can re-derive.
pub(super) fn validate_sd35_fixture(
    request: &Value,
    provider: &str,
    tier: &str,
) -> Result<(), String> {
    let slug = SD3_5_FIXTURE_SLUGS
        .iter()
        .find_map(|(id, slug)| (*id == provider).then_some(*slug))
        .ok_or_else(|| {
            format!("the Candle SD3.5 fixture binding does not implement provider {provider:?}")
        })?;
    let fixture = protocol::planned_fixture(request)?;
    let (width, _) = protocol::target_geometry(request)?;
    let prefix = format!("{FIVE_RUNG_FIXTURE_PREFIX}{slug}-{tier}-{width}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (planned_seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let planned_seed = planned_seed
        .parse::<u64>()
        .map_err(|error| format!("parse SD3.5 fixture seed {planned_seed:?}: {error}"))?;
    if planned_seed != FIVE_RUNG_SEED {
        return Err(format!(
            "planned.fixture seed {planned_seed} does not match the seed {provider} renders at \
             on this lane ({FIVE_RUNG_SEED})"
        ));
    }
    if steps != "2" {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(())
}
