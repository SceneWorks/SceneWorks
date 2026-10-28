//! # candle-audio-kokoro
//!
//! **Kokoro-82M** text-to-speech provider for the SceneWorks Candle audio lane — the first
//! real audio [`gen_core::Generator`] (sc-12836, epic sc-12833). One candle implementation
//! serves `runtime-cpu`, `runtime-cuda`, and `runtime-macos` through the audio composition
//! root (`candle-audio-catalog`), per `docs/architecture/audio-backend-strategy.md`.
//!
//! ## The port
//!
//! Kokoro (hexgrad/Kokoro-82M, Apache-2.0 weights + reference code) is a StyleTTS2-derived
//! architecture: a PLBERT (ALBERT) phoneme encoder, a prosody predictor (duration / F0 /
//! energy heads over BiLSTM + AdaIN residual stacks), a text encoder, and an iSTFT-Net-style
//! vocoder (harmonic-plus-noise source, AdaIN Snake resblocks, magnitude/phase head +
//! inverse STFT). This crate is a faithful component port onto the workspace's pinned candle
//! revision — module by module against the reference `model.py` / `modules.py` /
//! `istftnet.py`:
//!
//! - [`albert`] — the shared-layer ALBERT encoder,
//! - [`text_encoder`] / [`predictor`] — the StyleTTS2 text/prosody stacks,
//! - [`decoder`] — the styled decoder + vocoder head (harmonic source and the tiny n_fft=20
//!   STFT pair run as host `f32` DSP),
//! - [`weights`] — the five-section pickle checkpoint (old-style weight-norm resolved at
//!   load) and the voice style-vector packs,
//! - [`g2p`] — pure-Rust phonemization (misaki-rs lexicons + the exact US/GB post-processing
//!   Kokoro was trained with; **no espeak, no Python**),
//! - [`pipeline`] / [`model`] — the assembled synthesis pipeline and the
//!   [`gen_core::Generator`] adapter registered under **`kokoro_82m`**,
//! - [`prepare`] — the audio-lane snapshot-preparation accommodation (a validated
//!   passthrough; Kokoro snapshots carry no tokenizer.json for the LLM preparer to demand).
//!
//! Weights are supplied as an explicit passed-in snapshot on the `gen_core::LoadSpec`:
//! `hexgrad/Kokoro-82M`, staged locally and never self-fetched (epic 13657). The
//! `HUB_REPO`@`HUB_REVISION` pin is retained as the provenance record of that checkpoint.

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod albert;
pub mod config;
pub mod decoder;
pub mod g2p;
pub mod model;
pub mod nn;
pub mod pipeline;
pub mod predictor;
pub mod prepare;
pub mod text_encoder;
pub mod weights;

pub use config::KokoroConfig;
pub use g2p::{EnglishVariant, KokoroG2p};
pub use model::{
    descriptor, load, DEFAULT_VOICE, HUB_REPO, HUB_REVISION, LANGUAGES, MODEL_ID, REGISTRATION,
    VOICES,
};
pub use pipeline::KokoroPipeline;

pub use model::{COMPONENT_KEY, COMPONENT_LICENSE};

/// This crate's schema-3 component licence rows — one per loaded artifact (sc-16663). The audio
/// catalog concatenates every provider crate's slice into the licence table it ships.
pub const COMPONENT_LICENSES: &[gen_core::ComponentLicense] = &[model::COMPONENT_LICENSE];

/// The provider→component mapping this crate contributes. A provider's effective terms are
/// **derived** from these components by [`gen_core::provider_terms`] — never hand-authored — so
/// they cannot drift from the component rows they summarize.
pub const PROVIDER_COMPONENTS: &[gen_core::ProviderComponents] = &[gen_core::ProviderComponents {
    provider_id: model::MODEL_ID,
    components: &[model::COMPONENT_KEY],
}];

/// Add the Kokoro generator to an explicit audio registry builder (catalog composition).
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_generator(model::REGISTRATION)
}

/// Build the complete explicit Kokoro provider catalog (this crate's own surface).
pub fn provider_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    register_providers(gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_resolves_through_an_explicit_registry() {
        let registry = provider_registry().unwrap();
        let ids: Vec<String> = registry
            .generators()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert_eq!(ids, ["kokoro_82m"]);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
    }
}
