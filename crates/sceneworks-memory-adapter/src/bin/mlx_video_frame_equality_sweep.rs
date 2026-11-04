use super::*;

/// The two LTX arms keep a bare equality because their decode is CAUSAL at ×8 and every frame
/// count they can plan sits on the `1 + 8k` lattice the engine refuses to leave. Both halves
/// are asserted: a pin that made the LTX VAE non-causal, or a plan lattice that drifted off
/// `1 + 8k`, breaks the equality and reds here.
#[test]
fn the_arms_that_keep_a_bare_frame_equality_render_exactly_what_they_request() {
    for (provider, frames) in [
        (LTX_PROVIDER, LTX_CAMPAIGN_ENTRY_FRAMES),
        (LTX_PROVIDER, LTX_CANARY_FRAMES),
        (LTX25_PROVIDER, mlx_ltx25::BASE_FRAMES),
        (LTX25_PROVIDER, mlx_ltx25::MAX_FRAMES),
    ] {
        let vae = runtime_macos::vae_tiling(provider)
            .unwrap_or_else(|| panic!("{provider} publishes a VAE geometry"));
        assert!(vae.causal_temporal, "{provider}");
        assert_eq!(
            protocol::vae_decoded_frame_count(frames, vae.temporal_scale, vae.causal_temporal)
                .unwrap(),
            frames,
            "{provider} at {frames} frames"
        );
    }
}

/// Krea Realtime is the counter-example that proves the sweep is not vacuous: it decodes
/// through the SAME non-causal z16 VAE as Bernini and SCAIL-2, so its raw decode over-delivers
/// by one temporal stride minus one — and its arm still keeps a bare equality, because
/// `t2v.rs:348-353` drains that leading excess before returning. Applying the decoded-depth
/// rule to this arm would refuse a correct render.
///
/// Mutation that fails this: making the sweep apply the rule uniformly.
#[test]
fn the_krea_realtime_equality_rests_on_the_engines_trim_not_on_arithmetic() {
    let vae = runtime_macos::vae_tiling(KREA_REALTIME_PROVIDER)
        .expect("krea realtime publishes a VAE geometry");
    assert!(!vae.causal_temporal);
    let frames = 77;
    assert_ne!(
        protocol::vae_decoded_frame_count(frames, vae.temporal_scale, vae.causal_temporal).unwrap(),
        frames
    );
}

/// MiniMax-H3 publishes no VAE geometry on either resolver, so the shared rule has nothing to
/// resolve and its arm's equality is deliberately left alone. Pinned so a pin that STARTS
/// publishing one forces this sweep to be re-read rather than silently leaving the arm out.
#[test]
fn minimax_h3_publishes_no_vae_geometry_for_the_rule_to_resolve() {
    assert_eq!(runtime_macos::vae_tiling(MINIMAX_PROVIDER), None);
    assert_eq!(
        runtime_macos::vae_tiling_unmodelled_reason(MINIMAX_PROVIDER),
        None
    );
}
