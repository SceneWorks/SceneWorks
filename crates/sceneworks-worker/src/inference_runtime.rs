//! SceneWorks' explicit inference composition root.
//!
//! The platform bundle owns the provider list. This module owns the single process-wide catalog
//! value and exposes the narrow loading/introspection seams used by the worker. The non-native
//! desktop build deliberately gets empty registries instead of linking a tensor backend.

use std::sync::OnceLock;

// Used only by the macOS-gated `textllms()` introspection seam below.
#[cfg(all(test, target_os = "macos"))]
use gen_core::core_llm::TextLlmRegistration;
use gen_core::core_llm::{LoadSpec as TextLoadSpec, ModelRequirements, TextLlm, TextLlmRegistry};
use gen_core::{AudioTransform, Generator, LoadSpec, ProviderRegistry};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::{Captioner, ImageEmbedder, ModelRegistration, TextEmbedder, Trainer};

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda as platform_runtime;
#[cfg(target_os = "macos")]
use runtime_macos as platform_runtime;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn catalog() -> &'static platform_runtime::RuntimeCatalog {
    static CATALOG: OnceLock<platform_runtime::RuntimeCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        platform_runtime::catalog().unwrap_or_else(|error| {
            panic!("the compile-time inference bundle must form a valid runtime catalog: {error}")
        })
    })
}

pub(crate) fn media() -> &'static ProviderRegistry {
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        catalog().media()
    }

    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        static EMPTY: OnceLock<ProviderRegistry> = OnceLock::new();
        EMPTY.get_or_init(|| {
            gen_core::ProviderRegistryBuilder::new()
                .build()
                .expect("an empty media registry is valid")
        })
    }
}

/// The runtime's dedicated **candle audio** provider registry (SceneWorks Audio Studio, epic 13400 /
/// sc-13404), or `None` when this build ships no audio lane. Audio is candle-native on every platform
/// and rides a separate registry from [`media`] (the mlx media graph on macOS): the `runtime-macos`
/// bundle carries it default-on (`default = ["media", "audio"]`, sc-12835), so the macOS GPU worker
/// links it without any feature wiring here. The non-native desktop build has no catalog at all, so
/// it returns `None` (an audio job never routes there — the capability is never advertised).
pub(crate) fn audio() -> Option<&'static ProviderRegistry> {
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        catalog().audio()
    }

    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        None
    }
}

/// Load an audio [`Generator`] by id from the runtime's candle audio registry (sc-13404). Errors
/// clearly when this build ships no audio lane, mirroring how [`load`] resolves a media generator —
/// the audio worker turns this into a loud job failure rather than a silent no-op.
pub(crate) fn load_audio(id: &str, spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    audio()
        .ok_or_else(|| {
            gen_core::Error::Msg(
                "no audio lane is linked in this runtime build (the candle audio registry is \
                 unavailable)"
                    .to_owned(),
            )
        })?
        .load(id, spec)
}

/// Load an audio [`AudioTransform`] by id from the runtime's candle audio registry — the
/// non-prompt audio→audio lane (OpenVoice V2 tone-color voice conversion, sc-13411 C4). The audio
/// twin of [`load_audio`]: errors clearly when this build ships no audio lane so the voice-clone job
/// turns it into a loud job failure rather than a silent no-op.
pub(crate) fn load_audio_transform(
    id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<Box<dyn AudioTransform>> {
    audio()
        .ok_or_else(|| {
            gen_core::Error::Msg(
                "no audio lane is linked in this runtime build (the candle audio registry is \
                 unavailable)"
                    .to_owned(),
            )
        })?
        .load_audio_transform(id, spec)
}

pub(crate) fn text() -> &'static TextLlmRegistry {
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        catalog().text()
    }

    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        static EMPTY: OnceLock<TextLlmRegistry> = OnceLock::new();
        EMPTY.get_or_init(|| {
            gen_core::core_llm::TextLlmRegistryBuilder::new()
                .build()
                .expect("an empty text registry is valid")
        })
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn generators() -> impl ExactSizeIterator<Item = &'static ModelRegistration> {
    media().generators()
}

// Only the macOS prompt-refine tests iterate the TextLlm registry; on the Windows/candle build
// nothing calls this, so gate it to match its callers and stay warning-clean under -D warnings.
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn textllms() -> impl ExactSizeIterator<Item = &'static TextLlmRegistration> {
    text().registrations()
}

pub(crate) fn load(id: &str, spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    media().load(id, spec)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn load_trainer(id: &str, spec: &LoadSpec) -> gen_core::Result<Box<dyn Trainer>> {
    media().load_trainer(id, spec)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn load_captioner(id: &str, spec: &LoadSpec) -> gen_core::Result<Box<dyn Captioner>> {
    media().load_captioner(id, spec)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn load_image_embedder(
    id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<Box<dyn ImageEmbedder>> {
    media().load_image_embedder(id, spec)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn load_text_embedder(
    id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<Box<dyn TextEmbedder>> {
    media().load_text_embedder(id, spec)
}

pub(crate) fn load_for_model_with(
    spec: &TextLoadSpec,
    requirements: &ModelRequirements,
) -> gen_core::core_llm::Result<Box<dyn TextLlm>> {
    text().load_for_model_with(spec, requirements)
}

#[cfg(test)]
mod tests {
    #[test]
    fn composition_is_available_without_loading_weights() {
        let media_count = super::media().generators().len();
        let text_count = super::text().registrations().len();

        #[cfg(target_os = "macos")]
        {
            assert!(media_count > 50);
            assert_eq!(text_count, 2);
        }

        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        {
            assert!(media_count > 40);
            assert_eq!(text_count, 2);
        }

        #[cfg(not(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        )))]
        {
            assert_eq!(media_count, 0);
            assert_eq!(text_count, 0);
        }
    }
}
