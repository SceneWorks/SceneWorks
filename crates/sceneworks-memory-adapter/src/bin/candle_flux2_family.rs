//! CANDLE flux2 capture owner.
use super::*;

/// Which FLUX.2 member the plan asks for, or `None` when the plan is not a FLUX.2 one at all.
/// A FLUX.2 provider with a model id no member serves is an ERROR, not a `None`: the KV plan must
/// never be satisfied by the base klein artifact, which shares the provider id.
pub(super) fn flux2_arm(request: &Value) -> Result<Option<Flux2Arm>, String> {
    let provider = planned_provider(request)?;
    if !FLUX2_ARMS.iter().any(|arm| arm.provider == provider) {
        return Ok(None);
    }
    let model_id = protocol::planned_model_id(request)?;
    FLUX2_ARMS
        .into_iter()
        .find(|arm| arm.provider == provider && arm.model_id == model_id)
        .map(Some)
        .ok_or_else(|| {
            format!(
                "the Candle FLUX.2 arm does not implement provider {provider:?} for model \
                 {model_id:?}"
            )
        })
}
