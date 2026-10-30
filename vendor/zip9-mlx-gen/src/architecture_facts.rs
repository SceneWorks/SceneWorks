//! Shared narrowing helpers for the provider-declared [`gen_core::MemoryArchitectureFacts`].
//!
//! Every MLX provider crate mirrors the reference component `config.json` of its family as Rust
//! constants — `mlx_gen_sdxl::config::UNetConfig::sdxl_base()`,
//! `mlx_gen_z_image::transformer::ZImageTransformerConfig::turbo()`, and so on. Some crates parse
//! the snapshot's own `config.json` over that preset at load (Krea 2, LTX, MiniMax-H3, Mage, Wan);
//! the rest treat the preset as authoritative and validate a snapshot against it. Either way those
//! constants are the config a provider "already parses", and the architecture axes are published
//! from them. A consequence worth stating plainly: an MLX contract's architecture facts are
//! available *before* any snapshot is materialized, which is what lets the weights-free catalog
//! surface satisfy the E2 gate.
//!
//! This module owns only the narrowing arithmetic, so a fabricated or zeroed axis cannot appear in
//! one family and not another.

/// The materialized snapshot directory a contract is being built for, or `None`.
///
/// An MLX provider mirrors its family's reference `config.json` as crate constants, so its axes are
/// publishable before any asset exists — that is what lets the weights-free catalog surface satisfy
/// the E2 gate. But the *loader* overlays a materialized snapshot's own config over that preset, so
/// on the materialized path the preset is a guess, not the geometry. This is the gate a provider's
/// `architecture_facts` uses to tell the two paths apart: when it yields a root, the provider
/// re-runs its own snapshot parse and publishes what the snapshot actually says; when it yields
/// `None` there is nothing to read and the preset is the honest answer.
///
/// The rule is narrow: the spec's weights must be a [`gen_core::WeightsSource::Dir`] **whose path
/// exists on disk as a directory**. It does not compare the path against the registry's sentinel
/// constant, so it fails closed for a renamed sentinel — the registry's
/// `/__sceneworks_memory_contract_surface__` is a path nobody creates, and any other unmaterialized
/// path behaves identically. A single-file import is not a component snapshot either.
pub fn materialized_root(spec: &gen_core::LoadSpec) -> Option<&std::path::Path> {
    match &spec.weights {
        gen_core::WeightsSource::Dir(root) if root.is_dir() => Some(root.as_path()),
        _ => None,
    }
}

/// Bytes per element of a bf16 or f16 activation.
pub const HALF_ACTIVATION_WIDTH: u32 = 2;

/// Bytes per element of an f32 activation.
pub const FLOAT32_ACTIVATION_WIDTH: u32 = 4;

/// Narrow a config integer to a declared architecture axis.
///
/// Zero, negative, and out-of-`u32`-range values are declined rather than published: no axis has a
/// legitimate zero, and `Some(0)` would silently zero any activation estimate that multiplied by
/// it (epic SC-22657, E2).
pub fn axis<T: TryInto<u32>>(value: T) -> Option<u32> {
    let value = value.try_into().ok()?;
    (value != 0).then_some(value)
}

/// Per-head channel width of a uniform-head attention stack.
///
/// Published only when the head count divides the hidden dimension exactly. A non-uniform stack
/// has no single head width, and rounding the quotient would invent one.
pub fn head_dim<H: TryInto<u32>, N: TryInto<u32>>(
    hidden_dim: H,
    attention_heads: N,
) -> Option<u32> {
    let hidden = axis(hidden_dim)?;
    let heads = axis(attention_heads)?;
    (hidden % heads == 0).then(|| hidden / heads)
}

/// Pixels per latent unit for a VAE that halves both spatial axes `downsamples` times.
///
/// A count no `u32` scale can express declines the axis instead of clamping the shift into a
/// fabricated one.
pub fn vae_spatial_scale_from_downsamples(downsamples: usize) -> Option<u32> {
    let downsamples = u32::try_from(downsamples).ok()?;
    (downsamples <= 5).then(|| 1_u32 << downsamples)
}

/// Pixels per latent unit for a VAE declaring `stages` `block_out_channels` entries.
///
/// Each stage after the first halves both spatial axes, so the four-stage image autoencoder gives
/// the x8 scale. A zero stage count has no autoencoder to describe and declines the axis.
pub fn vae_spatial_scale_from_stages(stages: usize) -> Option<u32> {
    vae_spatial_scale_from_downsamples(stages.checked_sub(1)?)
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_zero_or_negative_config_value_declines_its_axis() {
        assert_eq!(super::axis(30_i32), Some(30));
        assert_eq!(super::axis(0_i32), None);
        assert_eq!(super::axis(-4_i32), None);
        assert_eq!(super::axis(16_usize), Some(16));
    }

    #[test]
    fn head_width_is_published_only_for_a_uniform_head_stack() {
        assert_eq!(super::head_dim(3072_i32, 24_i32), Some(128));
        assert_eq!(super::head_dim(3840_i32, 7_i32), None);
        assert_eq!(super::head_dim(3072_i32, 0_i32), None);
    }

    #[test]
    fn vae_spatial_scale_follows_the_declared_stage_count() {
        assert_eq!(super::vae_spatial_scale_from_stages(4), Some(8));
        assert_eq!(super::vae_spatial_scale_from_stages(1), Some(1));
        assert_eq!(super::vae_spatial_scale_from_stages(0), None);
        // Seven stages => six halvings => x64, past the scale the axis can honestly express.
        assert_eq!(super::vae_spatial_scale_from_stages(7), None);
        assert_eq!(super::vae_spatial_scale_from_downsamples(3), Some(8));
        assert_eq!(super::vae_spatial_scale_from_downsamples(0), Some(1));
        assert_eq!(super::vae_spatial_scale_from_downsamples(6), None);
    }

    #[test]
    fn a_materialized_root_is_only_an_existing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(dir.path().to_path_buf()));
        assert_eq!(super::materialized_root(&spec), Some(dir.path()));

        // The registry's weights-free surface: a `Dir` nobody ever creates.
        let sentinel = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ));
        assert_eq!(super::materialized_root(&sentinel), None);
    }
}
