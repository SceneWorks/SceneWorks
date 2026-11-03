use super::*;

/// The production Candle pricing site, against the LINKED registry: a single-pass 2.1 request
/// is priced at exactly the provider's own `prefix_kv_cache_demand_bytes`, a multi-pass one and
/// a provider with no cache at nothing.
///
/// *Mutations that red this:* returning `None` from the site (no surcharge for Turbo); dropping
/// the single-pass gate (the hires request is priced); passing another geometry, reference
/// count or branch flag through (the byte equality).
#[test]
fn the_candle_lane_prices_the_linked_turbo_cache_only_for_a_single_pass_request() {
    let expected =
        runtime_cuda::providers::qwen_image_2_1::memory_strategy::prefix_kv_cache_demand_bytes(
            1024, 768, 2, true,
        );
    for engine_id in ["qwen_image_2_1", "qwen_image_2_1_turbo"] {
        assert_eq!(
            candle_prefix_kv_cache_surcharge(engine_id, 1024, 768, 2, true, true),
            Some(expected),
            "{engine_id}"
        );
        assert_eq!(
            candle_prefix_kv_cache_surcharge(engine_id, 1024, 768, 2, true, false),
            None,
            "{engine_id}: a multi-pass request is priced no cache"
        );
    }
    assert_eq!(
        candle_prefix_kv_cache_surcharge("qwen_image", 1024, 768, 2, true, true),
        None,
        "a provider with no cache is priced nothing"
    );
}
