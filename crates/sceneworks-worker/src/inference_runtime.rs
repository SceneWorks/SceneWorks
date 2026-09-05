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
use gen_core::{AudioTransform, Generator, LoadSpec, ProviderRegistry, VoiceEmbedder};
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

/// The linked inference bundle's complete weights-free capability snapshot.
///
/// This is the source for the checked-in parity descriptor artifact. Returning JSON keeps the
/// worker-side artifact schema identical to `runtime-catalog::RuntimeCatalogSnapshot::to_json`
/// without defining a second copy of the inference contract in SceneWorks.
pub(crate) fn capability_snapshot_json() -> Option<serde_json::Value> {
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        Some(catalog().snapshot().to_json())
    }

    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        None
    }
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

/// Fresh, weights-free provider-owned memory-contract surfaces for generated capability facts.
/// Kept separate from the process catalog so each platform dumper can construct its contract-only
/// inventory without loading model weights.
pub(crate) fn memory_contract_surface_registry() -> gen_core::Result<ProviderRegistry> {
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        platform_runtime::memory_contract_surface_registry()
    }

    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        gen_core::ProviderRegistryBuilder::new().build()
    }
}

/// A registered trainer descriptor without loading model weights. Used by the training dry-run and
/// real-run shared preflight so both paths validate the active backend's exact network surface.
pub(crate) fn trainer_descriptor(id: &str) -> Option<gen_core::TrainerDescriptor> {
    media()
        .trainers()
        .map(|registration| (registration.descriptor)())
        .find(|descriptor| descriptor.id == id)
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

/// True when `id` names an audio **Generator** whose advertised [`Capabilities`] consume
/// [`gen_core::ConditioningKind::ReferenceAudio`] — i.e. a native clone-TTS provider (Chatterbox's
/// `chatterbox_tts`) that renders a full cloned-voice WAV from a script + reference clip in a single
/// [`Generator::generate`] call (sc-13412). This is the capability gate the Voice Clone job routes
/// on: the moment such a generator is linked into the audio catalog it lights up the native
/// single-call path; otherwise the two-call Kokoro→OpenVoice conversion chain remains the fallback.
///
/// Deliberately checks the GENERATOR registry, not the audio-transform registry: OpenVoice V2 is an
/// [`AudioTransform`] that also advertises `ReferenceAudio`, but it re-timbres existing speech and so
/// cannot render from text on its own — only a text→waveform generator can. Weights-free: reads the
/// registration's `descriptor` alone (no model load). Returns `false` on a build with no audio lane.
pub(crate) fn audio_generator_clones_from_reference(id: &str) -> bool {
    audio().is_some_and(|registry| {
        registry.generators().any(|registration| {
            let descriptor = (registration.descriptor)();
            descriptor.id == id
                && descriptor
                    .capabilities
                    .conditioning
                    .contains(&gen_core::ConditioningKind::ReferenceAudio)
        })
    })
}

/// The audio **Generator** descriptor for `id` (epic 13657, sc-13679), or `None` when `id` is
/// unknown or this build ships no audio lane. Weights-free — reads the registration's `descriptor`
/// alone (like [`audio_generator_clones_from_reference`]), so the worker can read a model's
/// [`gen_core::ModelDescriptor::required_components`] and stage its coRequisite-provisioned components
/// BEFORE building the `LoadSpec` it loads with.
pub(crate) fn audio_descriptor(id: &str) -> Option<gen_core::ModelDescriptor> {
    audio()?
        .generators()
        .map(|registration| (registration.descriptor)())
        .find(|descriptor| descriptor.id == id)
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

/// Load an audio **voice embedder** (a speaker-encoder, e.g. `chatterbox_ve`) by id from the runtime's
/// candle audio registry (sc-13517). It maps a reference audio clip to a raw 256-d speaker-identity
/// vector — the identity twin of `load_image_embedder`. Errors clearly when this build ships no audio
/// lane so the register-a-voice flow turns it into a loud failure rather than a silent no-op. Unlike
/// the clone Generator's `Conditioning::ReferenceAudio` path (which the provider re-embeds internally),
/// this seam exposes the embedding directly for the saved-voice registry's near-duplicate detection.
pub(crate) fn load_voice_embedder(
    id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<Box<dyn VoiceEmbedder>> {
    audio()
        .ok_or_else(|| {
            gen_core::Error::Msg(
                "no audio lane is linked in this runtime build (the candle audio registry is \
                 unavailable)"
                    .to_owned(),
            )
        })?
        .load_voice_embedder(id, spec)
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

/// The media (image/video) **Generator** descriptor for `id` (epic 13657, sc-13679), or `None` when
/// `id` is unknown. The image/video twin of [`audio_descriptor`]: weights-free registry introspection
/// so the generation harness can resolve a model's coRequisite-provisioned components before building
/// its `LoadSpec`. Dormant until an image provider advertises `required_components` (SDXL, sc-13682);
/// every current media descriptor advertises `&[]`, so the seam that reads this is a no-op today.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn media_descriptor(id: &str) -> Option<gen_core::ModelDescriptor> {
    media()
        .generators()
        .map(|registration| (registration.descriptor)())
        .find(|descriptor| descriptor.id == id)
}

#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
std::thread_local! {
    static TEST_MEDIA_ENCODER_CONTRACT: std::cell::Cell<
        Option<(&'static [&'static str], gen_core::EncoderContract)>
    > = const { std::cell::Cell::new(None) };
}

/// Scoped compact contract override for tests that exercise the production sealed-source path.
/// Production builds cannot install an override, and the guard restores a nested predecessor.
#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
pub(crate) struct TestMediaEncoderContractGuard {
    previous: Option<(&'static [&'static str], gen_core::EncoderContract)>,
}

#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
impl Drop for TestMediaEncoderContractGuard {
    fn drop(&mut self) {
        TEST_MEDIA_ENCODER_CONTRACT.set(self.previous);
    }
}

#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
pub(crate) fn scoped_test_media_encoder_contract(
    ids: &'static [&'static str],
    contract: gen_core::EncoderContract,
) -> TestMediaEncoderContractGuard {
    TestMediaEncoderContractGuard {
        previous: TEST_MEDIA_ENCODER_CONTRACT.replace(Some((ids, contract))),
    }
}

/// Resolve the provider-owned text-encoder contract for an ordinary generator or an explicitly
/// registered bespoke/composed route. Missing is fail-closed: consumers must never infer a sibling
/// base id or hardcode a family contract.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn media_encoder_contract(id: &str) -> Option<gen_core::EncoderContract> {
    let production_contract = media().provider_encoder_contract(id)?;
    #[cfg(test)]
    if let Some((ids, contract)) = TEST_MEDIA_ENCODER_CONTRACT.get() {
        if ids.contains(&id) {
            return Some(contract);
        }
    }
    Some(production_contract)
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

/// Resolve one validated imported source shape and operation to the ordinary provider descriptor
/// that will load it. Missing is an explicit unsupported answer; callers must not union sibling
/// routes by family.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn imported_model_descriptor(
    family: &str,
    source: gen_core::ImportedModelSource,
    operation: gen_core::ImportedModelOperation,
) -> Option<gen_core::ModelDescriptor> {
    media().imported_model_descriptor(family, source, operation)
}

/// The portable checkpoint-adapter authority for one PLAN family spelling (epic 20398, sc-20644).
///
/// Keyed on [`CheckpointAdapterRegistration::compatibility_projection`], not on the adapter's own
/// portable `family`: a compiled `ImportPlanV1` carries the inspector's family token
/// (`crate::checkpoint_inspector::normalize_family` — `"mage-flow"`, `"z-image"`, `"krea_2"`), which
/// is exactly the projection spelling, and it is the same key
/// [`ProviderRegistry::imported_model_descriptor`] resolves routes under. Looking the adapter up by
/// the portable `family` instead would silently miss `mage_flow` vs `mage-flow` and hand the plan
/// route a "no adapter" answer for a family this backend really does bind.
///
/// This is what makes a family's eligible backends, dialect source shapes, component topology and
/// capability policy PLAN truth rather than a per-lane constant in this repository (E2/E5).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn checkpoint_adapter(
    plan_family: &str,
) -> Option<&'static gen_core::CheckpointAdapterRegistration> {
    media()
        .checkpoint_adapters()
        .find(|adapter| adapter.compatibility_projection.family == plan_family)
}

/// The [`gen_core::CheckpointBackend`] this worker build actually binds providers for.
///
/// A single value per build, not a runtime probe: the MLX and Candle catalogs are compiled in by
/// mutually exclusive cfg, so "which backend am I" is a compile-time fact. Used to check a family's
/// declared [`CheckpointAdapterRegistration::eligible_backends`] before any load is attempted, so an
/// explicitly ineligible family (Z-Image on MLX, Mage-Flow on Candle) refuses during planning with
/// the adapter's own truth instead of failing inside a loader.
#[cfg(target_os = "macos")]
pub(crate) const CHECKPOINT_BACKEND: gen_core::CheckpointBackend = gen_core::CheckpointBackend::Mlx;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) const CHECKPOINT_BACKEND: gen_core::CheckpointBackend =
    gen_core::CheckpointBackend::Candle;

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

/// Every plan row on `lane` must resolve through the shipped registry to a weights-free provider
/// contract that implements the rung the plan will select for it (sc-18408 item (d); sc-22736 for
/// the rung). Derived from the plan rather than a hand-maintained provider list: adding a planned
/// lane without registering its contract, or planning a rung the provider's contract refuses
/// (Candle SCAIL-2 declares `Resident` alone, so a `staged_residency` plan would fail
/// `validate_selection` on every attempt — the 3-cell blocker the sc-22736 review found), must
/// fail here before the calibration adapter reaches a physical capture.
///
/// Provider-owned contract fixtures avoid filesystem-shaped test doubles where providers expose
/// them — or, for a route with no optimized surface, the typed resident-only witness, which is the
/// same weights-free builder on the other of the two mutually exclusive seams (the registry
/// builder refuses both at once); SDXL and FLUX.2-dev intentionally fall back to their normal
/// registrations, whose contract builders are themselves weights-free.
///
/// The planned rung is exactly what `memory-calibration-harness.mjs` `planAnchor` sends: the row's
/// own `rung` when it states one, else the lane default from `ANCHOR_STRATEGY` (mlx `resident`,
/// candle `staged_residency`). A change to either side that this does not follow is a red here.
#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
pub(crate) fn every_planned_lane_row_resolves_a_weights_free_contract_implementing_its_rung(
    lane: &str,
    expected_backend: gen_core::MemoryBackend,
) {
    use gen_core::{MemorySelection, MemoryStrategy, WeightsSource};
    let plan: serde_json::Value =
        serde_json::from_str(include_str!("../../../config/memory-calibration-plan.json"))
            .expect("memory calibration plan parses");
    // sc-22514: the plan is an ANCHOR plan — an object keyed `<modelId>:<tier>:<backend>` with
    // exactly one entry per cell — so the tier and the lane come out of the KEY and everything
    // else out of the entry.
    let anchors = plan["anchors"].as_object().expect("plan anchors object");
    let registry = media();
    let lane_default_rung = match lane {
        "mlx" => "resident",
        "candle" => "staged_residency",
        other => panic!("unknown lane {other}"),
    };
    let mut checked = 0_usize;
    let mut overridden = 0_usize;
    let mut forced = 0_usize;

    for (key, row) in anchors.iter() {
        let coordinates: Vec<&str> = key.split(':').collect();
        let [_model_id, tier, backend] = coordinates.as_slice() else {
            panic!("anchor key {key} must be <modelId>:<tier>:<backend>")
        };
        if *backend != lane {
            continue;
        }
        let provider = row["provider"].as_str().expect("anchor provider");
        let mode = row["mode"].as_str().expect("anchor mode");
        let overlay = row["overlay"].as_str().expect("anchor overlay");
        let load_shape = match row["loadShape"].as_str().expect("anchor loadShape") {
            "eager_materialization" => gen_core::LoadShape::EagerMaterialization,
            "deferred_materialization" => gen_core::LoadShape::DeferredMaterialization,
            other => {
                panic!("planned {lane} lane {provider}/{mode} names unknown load shape {other}")
            }
        };

        let mut spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from("fixture")))
            .with_load_shape(load_shape);
        let quant = match *tier {
            "q4" => Some(gen_core::Quant::Q4),
            "q8" => Some(gen_core::Quant::Q8),
            "bf16" => None,
            other => panic!("planned {lane} lane {provider}/{mode} names unknown tier {other}"),
        };
        if let Some(quant) = quant {
            spec = spec.with_quant(quant);
        }
        match overlay {
            "none" => {}
            "control:1" => {
                spec = spec.with_control(WeightsSource::File(std::path::PathBuf::from(
                    "fixture-control",
                )));
            }
            // sc-22728: production represents a `lora` overlay as an adapter stack on the
            // LoadSpec — the built-in Qwen edit Lightning distill is one LoRA at scale 1.0
            // ahead of any user adapters (`image_jobs/qwen.rs`), which is exactly the shape
            // the memory adapter's Lightning arm builds.
            "lora" => {
                spec = spec.with_adapters(vec![gen_core::AdapterSpec::new(
                    std::path::PathBuf::from("fixture-lora.safetensors"),
                    1.0,
                    gen_core::AdapterKind::Lora,
                )]);
            }
            // sc-22726: production represents the PuLID identity overlay as the typed
            // `LoadSpec::identity` seam — the adapter checkpoint, the EVA tower, and the
            // directory the three face models are read out of by name
            // (`image_jobs/pulid.rs::pulid_identity_weights`). All three slots are required by
            // the loader, so a fixture must fill all three or the contract builder refuses.
            "identity" => {
                spec.identity = Some(gen_core::IdentityWeights {
                    encoder: Some(WeightsSource::File(std::path::PathBuf::from(
                        "fixture-identity-encoder",
                    ))),
                    eva: Some(WeightsSource::File(std::path::PathBuf::from(
                        "fixture-identity-eva",
                    ))),
                    face_dir: Some(WeightsSource::Dir(std::path::PathBuf::from(
                        "fixture-identity-face",
                    ))),
                });
            }
            other => panic!(
                "planned {lane} lane {provider}/{mode} names unmapped overlay {other}; teach the \
                 generic contract guard how production represents it"
            ),
        }

        // sc-22729: a BESPOKE provider registers no `ModelDescriptor` and no memory-strategy
        // registration at all (`mlx_gen_catalog::BESPOKE_UTILITY_CRATES` lists `instantid`), so
        // there is no weights-free REGISTRY contract for this guard to resolve — the adapter arm
        // calls the crate's own `InstantId::load_with_memory_context`, which needs resolved paths,
        // not a fixture `LoadSpec`. Skipping silently would let a genuinely registered provider
        // take the same exit and lose its coverage, so the absence is ASSERTED here rather than
        // assumed, and the lane is not counted toward `checked`.
        const BESPOKE_PROVIDERS: [(&str, &str); 1] = [("mlx", "instantid")];
        if BESPOKE_PROVIDERS.contains(&(lane, provider)) {
            assert!(
                registry
                    .memory_strategy_registrations()
                    .all(|registration| registration.provider_id != provider),
                "planned {lane} lane {provider}/{mode} IS registered in the shipped runtime \
                 registry, so it must resolve a weights-free contract like every other lane \
                 instead of taking the bespoke exit"
            );
            continue;
        }

        let registration = registry
            .memory_strategy_registrations()
            .find(|registration| registration.provider_id == provider)
            .unwrap_or_else(|| {
                panic!(
                    "planned {lane} lane {provider}/{mode} has no memory-strategy registration in \
                     the shipped runtime registry"
                )
            });
        // sc-22736: a provider publishes its weights-free contract on exactly ONE of two seams,
        // and the registry builder refuses both at once ("has both a contract-surface fixture and
        // a resident-only witness"). A route with no optimized surface — SCAIL-2 on both lanes —
        // publishes the typed RESIDENT-ONLY WITNESS instead of a fixture, and it is the same
        // weights-free builder; reading only the fixture seam skipped it and fell through to the
        // normal registration, whose builder opens the artifact and fails with the snapshot's own
        // io error.
        let weights_free = registry
            .memory_contract_fixture_registrations()
            .find(|fixture| fixture.provider_id == provider)
            .map(|fixture| fixture.contract)
            .or_else(|| {
                registry
                    .resident_only_memory_contract_registrations()
                    .find(|witness| witness.provider_id == provider)
                    .map(|witness| witness.contract)
            });
        let contract = match weights_free {
            Some(build) => build(&spec),
            None => (registration.contract)(&spec),
        }
        .unwrap_or_else(|error| {
            panic!(
                "planned {lane} lane {provider}/{mode} cannot build a weights-free memory \
                 contract: {error}"
            )
        });

        assert_eq!(
            contract.provider_id, provider,
            "planned {lane} lane {provider}/{mode} resolved another provider's contract"
        );
        assert_eq!(
            contract.backend.backend_kind(),
            expected_backend,
            "planned {lane} lane {provider}/{mode} resolved a contract on another lane"
        );
        assert_eq!(
            contract.load_shape, load_shape,
            "planned {lane} lane {provider}/{mode} contract does not preserve its load shape"
        );
        let calibration = contract.calibration.as_ref().unwrap_or_else(|| {
            panic!(
                "planned {lane} lane {provider}/{mode} resolves only an uncalibratable \
                 compatibility contract"
            )
        });
        assert_eq!(
            calibration.load_shape, load_shape,
            "planned {lane} lane {provider}/{mode} calibration identity does not preserve its \
             load shape"
        );

        // The planned rung — the row's own `strategy.rung` (sc-22734's single-composition
        // override), else the lane default `planAnchor` applies — must be one this contract
        // EXECUTES. `validate_selection` is the production refusal the adapter hits after the load;
        // asking it here, weights-free, is what makes every planned cell non-vacuous rather than
        // three of them unrunnable by construction.
        let override_rung = row["strategy"]["rung"].as_str();
        let planned_rung = override_rung.unwrap_or(lane_default_rung);
        let as_strategy = |rung: &str| match rung {
            "resident" => MemoryStrategy::Resident,
            "staged_residency" => MemoryStrategy::StagedResidency,
            other => panic!("{key}: plans a rung an anchor may not plan: {other}"),
        };
        let selection_for = |strategy: MemoryStrategy| MemorySelection {
            strategy,
            parameters: gen_core::MemoryStrategyParameters::default(),
            tier: gen_core::MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant,
                component_precision_floors: &[],
            },
        };
        contract
            .validate_selection(&selection_for(as_strategy(planned_rung)))
            .unwrap_or_else(|error| {
                panic!(
                "{key}: the plan selects {planned_rung} but the {provider} contract refuses it — \
                 the anchor would fail at `contract.validate_selection` on every attempt: {error}"
            )
            });
        // An override is DERIVED, never picked (sc-22734, sc-22736): a row may move off the lane
        // default only because the contract refuses that default — SenseNova classifies
        // `StagedResidency` structurally not applicable, Candle SCAIL-2 implements `Resident`
        // alone — so an override that names a rung other than the default is legitimate exactly
        // when the default itself fails `validate_selection`. Same rung as the default (the MLX
        // SenseNova rows) moves nothing and proves nothing.
        if override_rung.is_some() {
            overridden += 1;
        }
        if planned_rung != lane_default_rung {
            let default_refused = contract
                .validate_selection(&selection_for(as_strategy(lane_default_rung)))
                .is_err();
            assert!(
                default_refused,
                "{key}: overrides the {lane} lane default {lane_default_rung} with {planned_rung}, \
                 but the {provider} contract would have executed the default — an override must be \
                 forced by the contract, not chosen"
            );
            forced += 1;
        }
        checked += 1;
    }

    assert!(
        checked > 0,
        "the shipped plan must contain at least one {lane} anchor"
    );
    // The mechanism is exercised, not merely tolerated: the candle lane carries at least one
    // contract-forced override (SenseNova's six and SCAIL-2's three today); on MLX the default is
    // already `resident`, so any override there is a no-op and nothing is forced.
    if lane == "candle" {
        assert!(
            forced > 0,
            "candle: the plan must carry at least one contract-forced rung override"
        );
    } else {
        assert_eq!(
            forced, 0,
            "{lane}: no override can move off a resident default"
        );
    }
    assert!(
        overridden > 0,
        "{lane}: the plan exercises the strategy override at least once"
    );
}

#[cfg(test)]
mod tests {
    /// The candle half of sc-18408 item (d) / sc-22736: every candle plan row resolves a
    /// weights-free contract that implements its planned rung — including the three SCAIL-2 cells
    /// whose provider declares `Resident` alone and which the plan therefore anchors resident.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn every_planned_candle_lane_resolves_a_weights_free_provider_contract() {
        super::every_planned_lane_row_resolves_a_weights_free_contract_implementing_its_rung(
            "candle",
            gen_core::MemoryBackend::Candle,
        );
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn test_encoder_contract_override_cannot_advertise_a_missing_route() {
        let contract = super::media_encoder_contract("qwen_image")
            .expect("the platform catalog registers Qwen Image");
        let _guard = super::scoped_test_media_encoder_contract(&["missing-test-route"], contract);

        assert_eq!(super::media_encoder_contract("missing-test-route"), None);
    }

    #[test]
    fn composition_is_available_without_loading_weights() {
        let media_count = super::media().generators().len();

        // The admitted text-LLM roster is an inference-pin fact: assert the exact id set rather
        // than a bare count so the next roster move names itself in the failure output.
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        let text_ids: Vec<String> = {
            let mut ids: Vec<String> = super::text()
                .registrations()
                .map(|registration| (registration.descriptor)().id)
                .collect();
            ids.sort();
            ids
        };

        #[cfg(target_os = "macos")]
        {
            assert!(media_count > 50);
            assert_eq!(
                text_ids,
                [
                    "mlx-joycaption",
                    "mlx-llama",
                    "mlx-starvector-1b",
                    "mlx-starvector-8b",
                ]
            );
        }

        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        {
            assert!(media_count > 40);
            assert_eq!(
                text_ids,
                [
                    "candle-llama",
                    "candle-llava",
                    "candle-starvector-1b",
                    "candle-starvector-8b",
                ]
            );
        }

        #[cfg(not(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        )))]
        {
            assert_eq!(media_count, 0);
            assert_eq!(super::text().registrations().len(), 0);
        }
    }
}
