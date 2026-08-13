//! **Stage 1** of the engine-capability pipeline (sc-16965, epic 16948): the engine-linked
//! dumper that turns the *linked* provider registry into a checked-in, engine-keyed facts file.
//!
//! # Why a dumper and not a serve-time lookup
//!
//! `Capabilities::supports_preview` (gen-core, sc-16677) is the only weights-free way to tell
//! "this route cannot live-preview" from "it can, but no frame has arrived yet". But the process
//! that serves `GET /api/v1/models` is **not** always the process that links an engine:
//!
//! - **Desktop** builds `sceneworks-rust-api --features embed-web,backend-candle` — registry
//!   in-process.
//! - **Docker / RunPod** builds `sceneworks-rust-api --features embed-web` with **no**
//!   `backend-candle`, while the *worker* image is built with it. Separate images.
//!
//! [`crate::inference_runtime::media`] returns an **empty** `ProviderRegistry` unless
//! `target_os = "macos"` or `feature = "backend-candle"`, so a serve-time derivation would answer
//! truthfully on desktop and report "nothing supports preview" on every server — a *wrong* answer,
//! not a missing one, in the topology least exercised locally. `sceneworks-core`, which owns the
//! manifests, has no engine dependency at all and can never derive.
//!
//! # Two stages
//!
//! 1. **This module** walks `registry.generators()` weights-free and writes one checked-in facts
//!    file *per backend* — `capabilities.candle.json` from a `backend-candle` lane,
//!    `capabilities.mlx.json` from the macOS lane. One file per backend means two lanes never
//!    rewrite the same file. These are the analogue of `documents/style.txt`: a checked-in source.
//!    The **audio** registry is dumped alongside them into `config/engine-capabilities/audio/` —
//!    see [the audio section](#the-audio-registry-is-a-second-dump-not-a-second-half-of-the-first).
//! 2. `apps/web/scripts/generate-preview-support.mjs` derives the served catalog block from those
//!    files, joined to SceneWorks model ids through [`crate::engines::MODEL_TABLE`], and
//!    `apps/web/src/data/previewSupportCatalog.test.js` re-derives it as a drift guard.
//!
//! Stage 2 depends only on checked-in files, so its guard runs on **every PR**. Stage 1 needs a
//! linked registry, which exists only on macOS and `backend-candle` lanes — so *writing* a facts
//! file stays a manual step run on the matching box.
//!
//! **Both files are nevertheless VERIFIED on every PR** that can invalidate them. `macos-mlx.yml`
//! re-dumps to a scratch dir and diffs `capabilities.mlx.json` against it (sc-17119); the
//! self-hosted CUDA lane `windows-candle.yml` does the same for `capabilities.candle.json` under
//! `shell: cmd` (sc-17592). Each lane watches its OWN dump by exact path — not the directory
//! (sc-17665), which would wake each self-hosted box for the other backend's file — so the one edit
//! these guards exist to catch — a **restamp**, where the `inferenceRevision` line is rewritten and
//! the engine list is left stale — cannot slip through by touching that file alone. Without those steps
//! every remaining guard reads this file as a *source* and is satisfied by editing one line. (The
//! other candle lanes still verify nothing: `desktop-windows.yml` builds the sidecar on main/release
//! only, and `server-candle-linux.yml` is `workflow_dispatch`-only.)
//!
//! **Nor was the dispatch-only era an unguarded gap.** sc-16951's `candle-gen-catalog` bidirectional
//! test runs in the *inference* repo's CI on every PR, so descriptor-level truth is guarded
//! continuously upstream. SceneWorks only needs to re-dump when the descriptors can have moved —
//! i.e. at an **inference pin bump**, where `scripts/bump-inference.mjs` fails closed on a stale
//! facts file, beside the licence re-scan in the same generated-artifact cascade.
//!
//! # The audio registry is a second dump, not a second half of the first
//!
//! [`crate::inference_runtime::audio`] is a **separate** `ProviderRegistry` (epic 13400 / sc-13404):
//! audio is candle-native on every platform and rides its own lane rather than the mlx media graph,
//! which is why both runtime bundles carry it `default = ["media", "audio"]`. Until sc-17593 nothing
//! dumped it, so `supports_preview` was absent from the derived catalog for **every** audio route on
//! **every** platform — reported as "unknown", which renders exactly like "not wired".
//!
//! It gets its own file under `config/engine-capabilities/audio/` rather than being folded into the
//! per-backend media file, for three reasons:
//!
//! 1. **Folding is unsound, not merely untidy.** `candle-audio-catalog::AUDIO_BACKEND` is `"candle"`
//!    on every platform, so the macOS lane — media `mlx`, audio `candle` — would have to write a
//!    *second* file named `capabilities.candle.json` and clobber the candle lane's **media** dump.
//!    One file per backend exists precisely so two lanes never rewrite the same file. Filing macOS's
//!    audio facts under `mlx` instead would be a plain lie: those descriptors say `backend: candle`.
//! 2. **The two registries' engine ids are independent namespaces joined by different rules.** Media
//!    engine ids reach SceneWorks model ids through [`crate::engines::MODEL_TABLE`]; audio engine ids
//!    **are** SceneWorks model ids (`kokoro_82m`, `chatterbox_tts`, `acestep_v15_turbo`, …), which is
//!    why [`crate::audio_jobs`] passes the payload's `model` straight into
//!    [`crate::inference_runtime::load_audio`]. Nothing prevents the two namespaces from colliding;
//!    one file would need a per-entry discriminator to keep the joins apart anyway.
//! 3. **A subdirectory, not a longer file name.** Stage 2 discovers media facts in three places
//!    (`bump-inference.mjs`, `generate-preview-support.mjs`, and the vitest guard's
//!    `import.meta.glob`). `readdirSync` and that glob are both single-level, so a subdirectory is
//!    invisible to all three *without editing the media discovery at all*; a sibling
//!    `capabilities.candle.audio.json` would be picked up by the glob's `*`, which does cross a dot.
//!
//! Media files deliberately carry no registry discriminator: `capabilities.mlx.json` can only be
//! re-dumped on a Mac, so the audio lane must remain distinguishable without rewriting that file.
//! Optional descriptor facts may still extend each engine row when both backend-owned dumps are
//! regenerated from the corresponding linked registries. The audio file instead carries its own
//! [`AudioCapabilityFacts::registry`] discriminator, so a file can never be mistaken for the other
//! kind regardless of where it sits.
//!
//! Only **generators** carry [`gen_core::Capabilities`], so only they are dumped. The audio lane's
//! other provider kinds — `openvoice_v2` (an `AudioTransform`), `chatterbox_ve` (a `VoiceEmbedder`),
//! Whisper (a `Transcriber`), CLAP (an `AudioEmbedder`) — have their own descriptor types with no
//! `supports_preview` at all, so they stay **unknown**: the registry genuinely has no opinion, and
//! inventing `false` would be asserting something never measured.
//!
//! # Running it
//!
//! ```text
//! cargo run -p sceneworks-worker --bin dump-engine-capabilities \
//!     --no-default-features --features backend-candle     # Windows / Linux CUDA → capabilities.candle.json
//! cargo run -p sceneworks-worker --bin dump-engine-capabilities   # macOS → capabilities.mlx.json
//! ```
//!
//! Either invocation also writes `audio/capabilities.candle.json`. Both lanes link the *same* audio
//! catalog — `candle-audio-catalog` gates nothing on the platform, its `metal`/`cuda` features only
//! forward the compute backend — so the two lanes produce byte-identical audio facts and neither
//! owns the file exclusively. That is not left as an argument: **both** restamp-verification lanes
//! diff this one file against their own fresh dump (`macos-mlx.yml` and `windows-candle.yml`), so
//! the day the two lanes stop agreeing, one of them says so.
//!
//! # The vacuous-green trap
//!
//! [`facts_from_descriptors`] **refuses** to produce anything from an empty descriptor list. On a
//! lane without engines the dumper would otherwise emit a valid, empty, entirely wrong facts file,
//! and stage 2 would happily derive "no route supports preview" from it — the same failure class as
//! a candle CI smoke that proves PASS rather than kill. The assertion is on the pure function, not
//! on the I/O wrapper, so it is unit-testable on **every** lane including the ones with no engines.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Every backend SceneWorks can link a **media** registry for, and therefore every backend stage 2
/// requires a checked-in facts file for (sc-17119).
///
/// Scoped to [`crate::inference_runtime::media`]. [`crate::inference_runtime::audio`] is a separate
/// registry with its own declared set, [`SCENEWORKS_AUDIO_BACKENDS`] — and it needs one, because this
/// list can never speak for it: audio is candle-native everywhere, so `candle` appearing here is
/// satisfied by the *media* dump alone and says nothing about whether the audio registry was ever
/// walked. Backend-level coverage was the wrong granularity, which is exactly how the audio gap
/// survived sc-17119 (sc-17593).
///
/// Without this list, "which backends are covered?" could only be answered by *the files on disk* —
/// so a backend that was **never dumped** passed every guard forever, silently. That is not
/// hypothetical: `capabilities.mlx.json` was absent for four consecutive pins beginning with the one
/// sc-16965 itself shipped on (`d4802320`, `bf06bb56`, `5b6d6aa0`, `5f973a73`), and the vitest drift
/// guard and the generator — which both glob this directory — were green throughout, as
/// `verifyEngineCapabilityFacts` would have been had anything invoked it. Absence rendered as
/// "unknown", which renders exactly as before — i.e. it looked fine.
///
/// This is the **union across both lanes**, so no single build can verify it whole: a macOS build
/// links only `mlx`, a `backend-candle` build only `candle` (see [`crate::inference_runtime::media`],
/// whose two cfg arms are the definition this list mirrors). What each lane *can* enforce is that
/// the backend it actually links is declared here — see `the_linked_backend_is_declared`. Between
/// the two lanes every entry is covered by a real registry.
///
/// Sorted and unique; stage 2 compares it against the sorted set of dumped files.
pub const SCENEWORKS_BACKENDS: &[&str] = &["candle", "mlx"];

/// Every backend the **audio** registry ([`crate::inference_runtime::audio`]) can register under,
/// and therefore every backend stage 2 requires an `audio/capabilities.<backend>.json` for (sc-17593).
///
/// One entry, on purpose. `candle-audio-catalog::AUDIO_BACKEND` is `"candle"` on every platform —
/// the audio lane is candle-native even on macOS, where the media graph is mlx — so unlike
/// [`SCENEWORKS_BACKENDS`] this is not a union across mutually exclusive lanes. **Both** lanes link
/// the same audio catalog and can dump the whole of it, so `the_linked_audio_backend_is_declared`
/// verifies this list completely on either box, and the two lanes' dumps are byte-identical.
///
/// It is a separate const rather than a reuse of [`SCENEWORKS_BACKENDS`] because the two lists answer
/// different questions. `SCENEWORKS_BACKENDS` containing `candle` is satisfied by the media dump
/// alone; without this const, "was the audio registry ever walked?" has no checkable answer — which
/// is the hole sc-17119's backend-level coverage passed straight over.
pub const SCENEWORKS_AUDIO_BACKENDS: &[&str] = &["candle"];

/// A latent grid's spatial pixel compression, serialized without backend-specific types.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpatialCompressionFact {
    pub height: u16,
    pub width: u16,
}

/// Whether the decoder seam receives the native latent grid or a patch-packed one.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    tag = "kind"
)]
pub enum LatentPatchLayoutFact {
    Unpacked,
    Packed { patch_height: u8, patch_width: u8 },
}

/// Temporal pixel-to-latent law. Kept explicit so a still-image z16 space never compares equal to
/// Wan's numerically normalized but causally compressed z16 video space.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum LatentTemporalLawFact {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "causal-4x")]
    Causal4x,
    #[serde(rename = "causal-6x")]
    Causal6x,
    #[serde(rename = "causal-8x")]
    Causal8x,
}

/// Stable normalization identity at the denoiser-to-decoder seam.
///
/// Fixed per-channel vectors carry a string hash because a 64-bit JSON number cannot be compared
/// exactly by JavaScript. Learned vectors deliberately carry no content hash, preserving gen-core's
/// fail-closed rule instead of claiming two independently loaded checkpoints are compatible.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    tag = "kind"
)]
pub enum LatentNormalizationFact {
    Identity,
    Affine {
        scale_bits: u32,
        shift_bits: u32,
    },
    PerChannel {
        identity: String,
        channels: u16,
        content_hash: String,
    },
    LearnedPerChannel {
        identity: String,
    },
}

/// Complete backend-neutral identity of a tensor at a denoiser-to-decoder boundary.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LatentSpaceFact {
    pub channels: u16,
    pub spatial_compression: SpatialCompressionFact,
    pub patch_layout: LatentPatchLayoutFact,
    pub temporal_law: LatentTemporalLawFact,
    pub normalization: LatentNormalizationFact,
}

impl From<&gen_core::LatentSpace> for LatentSpaceFact {
    fn from(space: &gen_core::LatentSpace) -> Self {
        let patch_layout = match space.patch_layout {
            gen_core::LatentPatchLayout::Unpacked => LatentPatchLayoutFact::Unpacked,
            gen_core::LatentPatchLayout::Packed {
                patch_height,
                patch_width,
            } => LatentPatchLayoutFact::Packed {
                patch_height,
                patch_width,
            },
        };
        let temporal_law = match space.temporal_law {
            gen_core::LatentTemporalLaw::None => LatentTemporalLawFact::None,
            gen_core::LatentTemporalLaw::Causal4x => LatentTemporalLawFact::Causal4x,
            gen_core::LatentTemporalLaw::Causal6x => LatentTemporalLawFact::Causal6x,
            gen_core::LatentTemporalLaw::Causal8x => LatentTemporalLawFact::Causal8x,
        };
        let normalization = match space.normalization {
            gen_core::LatentNormalization::Identity => LatentNormalizationFact::Identity,
            gen_core::LatentNormalization::Affine {
                scale_bits,
                shift_bits,
            } => LatentNormalizationFact::Affine {
                scale_bits,
                shift_bits,
            },
            gen_core::LatentNormalization::PerChannel(stats) => {
                LatentNormalizationFact::PerChannel {
                    identity: stats.identity.to_owned(),
                    channels: stats.channels,
                    content_hash: format!("fnv1a64:{:016x}", stats.content_hash),
                }
            }
            gen_core::LatentNormalization::LearnedPerChannel { identity } => {
                LatentNormalizationFact::LearnedPerChannel {
                    identity: identity.to_owned(),
                }
            }
        };
        Self {
            channels: space.channels,
            spatial_compression: SpatialCompressionFact {
                height: space.spatial_compression.height,
                width: space.spatial_compression.width,
            },
            patch_layout,
            temporal_law,
            normalization,
        }
    }
}

/// One engine's weights-free capability facts, as written to a per-backend facts file.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineFact {
    /// The gen-core registry id (`ModelDescriptor::id`) — the join key stage 2 resolves
    /// `MODEL_TABLE.engine_id` against.
    pub id: String,
    /// `"image"` | `"video"` | `"both"` | `"audio"` (`ModelDescriptor::modality`).
    pub modality: String,
    /// `Capabilities::supports_preview` — whether this engine emits `PreviewSink` frames.
    pub supports_preview: bool,
    /// Exact denoiser output contract. Missing remains unknown and therefore incompatible with
    /// every decoder; no consumer may infer it from the engine id or family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub denoiser_output_latent_space: Option<LatentSpaceFact>,
}

/// One provider-owned route for an exact imported source shape and request operation. These rows
/// are emitted from the linked inference registry; an absent row is an explicit backend refusal.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportedProviderFact {
    pub family: String,
    pub source: String,
    pub operation: String,
    pub provider_id: String,
    pub conditioning: Vec<String>,
    pub supports_lora: bool,
    pub supports_lokr: bool,
    pub supported_quants: Vec<String>,
    pub supports_kv_cache: bool,
    pub supports_sequential_offload: bool,
    /// Every route is resolved and loaded through the ordinary runtime registry/cache seam.
    pub registry_cached: bool,
}

/// Provenance stamped onto every facts file so a stale dump is detectable without rebuilding.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FactsProvenance {
    /// The inference revision the descriptors were read from — the same constant
    /// `scripts/bump-inference.mjs` repins, so a bump makes every facts file stale by construction
    /// and `verifyEngineCapabilityFacts` in that script says so.
    pub inference_revision: String,
    /// How to reproduce this file.
    pub dumper: String,
}

/// One backend's complete, weights-free engine facts.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineCapabilityFacts {
    /// `"mlx"` | `"candle"` — `ModelDescriptor::backend`.
    pub backend: String,
    pub generated_from: FactsProvenance,
    /// Every generator this backend registered, sorted by id so the file is byte-stable across runs.
    pub engines: Vec<EngineFact>,
    /// Exact provider-declared import routes. Consumers must match both `source` and `operation`;
    /// family-wide unioning would advertise shapes the selected loader cannot validate.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imports: Vec<ImportedProviderFact>,
}

impl EngineCapabilityFacts {
    /// `capabilities.<backend>.json` — the file name this backend owns. One file per backend is
    /// what keeps the macOS and candle lanes from overwriting each other's dump.
    pub fn file_name(&self) -> String {
        format!("capabilities.{}.json", self.backend)
    }
}

/// The `registry` discriminator every audio facts file carries.
///
/// Media facts files carry **no** `registry` key — deliberately, because that discriminator is
/// owned by the audio lane. So "has
/// `registry: audio`" is the test for an audio file, and its absence means media.
pub const AUDIO_REGISTRY_LABEL: &str = "audio";

/// One backend's complete, weights-free **audio** engine facts (sc-17593).
///
/// Structurally [`EngineCapabilityFacts`] plus the [`AUDIO_REGISTRY_LABEL`] discriminator, so a file
/// that ends up in the wrong directory is still self-describing. A distinct type rather than a flag
/// on the media struct for exactly that reason: the two registry namespaces must stay distinct.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioCapabilityFacts {
    /// Always [`AUDIO_REGISTRY_LABEL`]. Serialized first so the discriminator is the first line of
    /// the file a human opens.
    pub registry: String,
    /// `"candle"` on every platform — `candle-audio-catalog::AUDIO_BACKEND`.
    pub backend: String,
    pub generated_from: FactsProvenance,
    /// Every audio **generator** this backend registered, sorted by id. The lane's non-generator
    /// providers (`AudioTransform`, `VoiceEmbedder`, `Transcriber`, `AudioEmbedder`) have descriptor
    /// types with no `supports_preview`, so they are absent — i.e. unknown, which is the truth.
    pub engines: Vec<EngineFact>,
}

impl AudioCapabilityFacts {
    /// `capabilities.<backend>.json` — the name inside the `audio/` subdirectory, so it is
    /// deliberately identical to the media file's. What separates them is the directory, and
    /// failing that, [`AudioCapabilityFacts::registry`].
    pub fn file_name(&self) -> String {
        format!("capabilities.{}.json", self.backend)
    }
}

fn modality_label(modality: &gen_core::Modality) -> &'static str {
    // Exhaustive on purpose: a new gen-core modality must break this build loudly at the next pin
    // bump rather than be silently flattened into a catch-all label.
    match modality {
        gen_core::Modality::Image => "image",
        gen_core::Modality::Video => "video",
        gen_core::Modality::Both => "both",
        gen_core::Modality::Audio => "audio",
    }
}

/// Group descriptors into one [`EngineCapabilityFacts`] per backend.
///
/// Pure — no registry, no filesystem — so the empty-registry refusal below is exercised on lanes
/// that have no engines at all, which is exactly where the mistake it prevents would be made.
///
/// # Errors
///
/// - the descriptor list is **empty** (the vacuous-green trap: an empty registry must never be
///   mistaken for "no engine supports preview");
/// - a backend registered the same engine id twice, which would make the stage-2 join ambiguous.
pub fn facts_from_descriptors(
    descriptors: &[gen_core::ModelDescriptor],
    inference_revision: &str,
    dumper: &str,
) -> Result<Vec<EngineCapabilityFacts>, String> {
    if descriptors.is_empty() {
        return Err(
            "refusing to write engine-capability facts: the provider registry is EMPTY. This lane \
             links no engines, so the dump would be a valid, empty, entirely wrong facts file that \
             stage 2 would read as \"no route supports live preview\". Re-run on a lane that links \
             a registry: macOS (mlx), or off-Mac with --no-default-features --features \
             backend-candle."
                .to_owned(),
        );
    }

    let mut out = Vec::new();
    for (backend, engines) in group_by_backend(descriptors)? {
        out.push(EngineCapabilityFacts {
            backend,
            generated_from: FactsProvenance {
                inference_revision: inference_revision.to_owned(),
                dumper: dumper.to_owned(),
            },
            engines,
            imports: Vec::new(),
        });
    }
    Ok(out)
}

fn imported_source_label(source: gen_core::ImportedModelSource) -> &'static str {
    match source {
        gen_core::ImportedModelSource::TransformerFile => "transformer_file",
        gen_core::ImportedModelSource::FusedCheckpoint => "fused_checkpoint",
        gen_core::ImportedModelSource::TransformerDirectory => "transformer_directory",
        gen_core::ImportedModelSource::ComfyUiTree => "comfy_ui_tree",
    }
}

fn imported_operation_label(operation: gen_core::ImportedModelOperation) -> &'static str {
    match operation {
        gen_core::ImportedModelOperation::Generate => "generate",
        gen_core::ImportedModelOperation::Edit => "edit",
        gen_core::ImportedModelOperation::Pose => "pose",
        gen_core::ImportedModelOperation::MultiPhase => "multi_phase",
    }
}

fn conditioning_label(kind: gen_core::ConditioningKind) -> &'static str {
    match kind {
        gen_core::ConditioningKind::Reference => "reference",
        gen_core::ConditioningKind::ReferenceAudio => "reference_audio",
        gen_core::ConditioningKind::AudioEdit => "audio_edit",
        gen_core::ConditioningKind::AudioEditRegions => "audio_edit_regions",
        gen_core::ConditioningKind::VoiceEmbedding => "voice_embedding",
        gen_core::ConditioningKind::MultiReference => "multi_reference",
        gen_core::ConditioningKind::ReduxRefs => "redux_refs",
        gen_core::ConditioningKind::Control => "control",
        gen_core::ConditioningKind::Depth => "depth",
        gen_core::ConditioningKind::Mask => "mask",
        gen_core::ConditioningKind::Keyframe => "keyframe",
        gen_core::ConditioningKind::VideoClip => "video_clip",
        gen_core::ConditioningKind::ControlClip => "control_clip",
        gen_core::ConditioningKind::VideoSync => "video_sync",
        gen_core::ConditioningKind::ConversationHistory => "conversation_history",
    }
}

fn quant_label(quant: gen_core::Quant) -> &'static str {
    match quant {
        gen_core::Quant::Q4 => "q4",
        gen_core::Quant::Q8 => "q8",
        gen_core::Quant::Nvfp4 => "nvfp4",
    }
}

fn append_imported_fact(
    facts: &mut [EngineCapabilityFacts],
    route: &gen_core::ImportedModelRegistration,
    descriptor: gen_core::ModelDescriptor,
) {
    let backend = facts
        .iter_mut()
        .find(|facts| facts.backend == descriptor.backend)
        .expect("descriptor backend was grouped above");
    backend.imports.push(ImportedProviderFact {
        family: route.family.to_owned(),
        source: imported_source_label(route.source).to_owned(),
        operation: imported_operation_label(route.operation).to_owned(),
        provider_id: route.provider_id.to_owned(),
        conditioning: descriptor
            .capabilities
            .conditioning
            .iter()
            .copied()
            .map(conditioning_label)
            .map(str::to_owned)
            .collect(),
        supports_lora: descriptor.capabilities.supports_lora,
        supports_lokr: descriptor.capabilities.supports_lokr,
        supported_quants: descriptor
            .capabilities
            .supported_quants
            .iter()
            .copied()
            .map(quant_label)
            .map(str::to_owned)
            .collect(),
        supports_kv_cache: descriptor.capabilities.supports_kv_cache,
        supports_sequential_offload: descriptor.capabilities.supports_sequential_offload,
        registry_cached: true,
    });
}

fn sort_imported_facts(facts: &mut [EngineCapabilityFacts]) {
    for backend in facts {
        backend.imports.sort_by(|left, right| {
            (
                &left.family,
                &left.source,
                &left.operation,
                &left.provider_id,
            )
                .cmp(&(
                    &right.family,
                    &right.source,
                    &right.operation,
                    &right.provider_id,
                ))
        });
    }
}

/// Test-facing pure derivation from registry parts. Production uses [`facts_from_registry`] so the
/// exact gen-core resolution seam owns provider lookup and structural capability withdrawal.
pub fn facts_from_registry_parts(
    descriptors: &[gen_core::ModelDescriptor],
    imports: &[gen_core::ImportedModelRegistration],
    inference_revision: &str,
    dumper: &str,
) -> Result<Vec<EngineCapabilityFacts>, String> {
    let mut facts = facts_from_descriptors(descriptors, inference_revision, dumper)?;
    for route in imports {
        let mut descriptor = descriptors
            .iter()
            .find(|descriptor| descriptor.id == route.provider_id)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "imported route {}/{:?}/{:?} targets missing provider {}",
                    route.family, route.source, route.operation, route.provider_id
                )
            })?;
        if !route.inherit_adapters {
            descriptor.capabilities.supports_lora = false;
            descriptor.capabilities.supports_lokr = false;
        }
        append_imported_fact(&mut facts, route, descriptor);
    }
    sort_imported_facts(&mut facts);
    Ok(facts)
}

/// Derive facts directly from the provider registry. `imported_model_descriptor` performs exact
/// family/source/operation resolution and applies `inherit_adapters`; the serializer never
/// re-implements either decision.
pub fn facts_from_registry(
    registry: &gen_core::ProviderRegistry,
    inference_revision: &str,
    dumper: &str,
) -> Result<Vec<EngineCapabilityFacts>, String> {
    let descriptors: Vec<gen_core::ModelDescriptor> = registry
        .generators()
        .map(|registration| (registration.descriptor)())
        .collect();
    let mut facts = facts_from_descriptors(&descriptors, inference_revision, dumper)?;
    for route in registry.imported_models() {
        let descriptor = registry
            .imported_model_descriptor(route.family, route.source, route.operation)
            .ok_or_else(|| {
                format!(
                    "registry refused its own imported route {}/{:?}/{:?} targeting {}",
                    route.family, route.source, route.operation, route.provider_id
                )
            })?;
        append_imported_fact(&mut facts, route, descriptor);
    }
    sort_imported_facts(&mut facts);
    Ok(facts)
}

/// Bucket descriptors per backend, each bucket sorted by id and proven duplicate-free.
///
/// Shared by the media and audio dumps because both need exactly this: the file must be byte-stable
/// across runs, and a repeated id would make the stage-2 join resolve to whichever row won — a
/// silently wrong answer for every SceneWorks id mapping onto it.
fn group_by_backend(
    descriptors: &[gen_core::ModelDescriptor],
) -> Result<BTreeMap<String, Vec<EngineFact>>, String> {
    let mut by_backend: BTreeMap<String, Vec<EngineFact>> = BTreeMap::new();
    for descriptor in descriptors {
        by_backend
            .entry(descriptor.backend.to_owned())
            .or_default()
            .push(EngineFact {
                id: descriptor.id.to_owned(),
                modality: modality_label(&descriptor.modality).to_owned(),
                supports_preview: descriptor.capabilities.supports_preview,
                denoiser_output_latent_space: descriptor
                    .denoiser_output_latent_space
                    .map(LatentSpaceFact::from),
            });
    }
    for (backend, engines) in &mut by_backend {
        engines.sort_by(|left, right| left.id.cmp(&right.id));
        if let Some(duplicate) = engines
            .windows(2)
            .find(|pair| pair[0].id == pair[1].id)
            .map(|pair| pair[0].id.clone())
        {
            return Err(format!(
                "backend {backend:?} registered engine id {duplicate:?} more than once; the \
                 stage-2 join would be ambiguous (through MODEL_TABLE.engine_id for a media dump, \
                 through the id itself for an audio one)"
            ));
        }
    }
    Ok(by_backend)
}

/// Group **audio** descriptors into one [`AudioCapabilityFacts`] per backend.
///
/// The audio twin of [`facts_from_descriptors`], pure for the same reason: the empty refusal is the
/// vacuous-green trap, and it has to be exercisable on a lane that links no audio at all.
///
/// # Errors
///
/// - the descriptor list is **empty** — an audio registry that registered nothing would derive as
///   "no audio route supports live preview", a confident wrong answer rather than a missing one;
/// - a backend registered the same engine id twice.
pub fn audio_facts_from_descriptors(
    descriptors: &[gen_core::ModelDescriptor],
    inference_revision: &str,
    dumper: &str,
) -> Result<Vec<AudioCapabilityFacts>, String> {
    if descriptors.is_empty() {
        return Err(
            "refusing to write audio engine-capability facts: the audio provider registry is \
             EMPTY. Stage 2 would read that as \"no audio route supports live preview\" — a \
             confident wrong answer, not a missing one. Both runtime bundles carry the audio lane \
             default-on (default = [\"media\", \"audio\"]), so an empty registry here means the \
             bundle was built with --no-default-features and no `audio`."
                .to_owned(),
        );
    }

    let mut out = Vec::new();
    for (backend, engines) in group_by_backend(descriptors)? {
        out.push(AudioCapabilityFacts {
            registry: AUDIO_REGISTRY_LABEL.to_owned(),
            backend,
            generated_from: FactsProvenance {
                inference_revision: inference_revision.to_owned(),
                dumper: dumper.to_owned(),
            },
            engines,
        });
    }
    Ok(out)
}

/// How this build's **media** facts file is reproduced, named in the file itself.
///
/// Platform-branched, and correctly so: each media file is owned by exactly one lane, so naming that
/// lane's invocation tells the reader the only command that can regenerate the file in front of them.
fn dumper_invocation() -> &'static str {
    if cfg!(target_os = "macos") {
        "cargo run -p sceneworks-worker --bin dump-engine-capabilities"
    } else {
        "cargo run -p sceneworks-worker --bin dump-engine-capabilities --no-default-features \
         --features backend-candle"
    }
}

/// How the **audio** facts file is reproduced — deliberately NOT platform-branched.
///
/// The audio file is the one file both lanes write, because `AUDIO_BACKEND` is `candle` everywhere.
/// Stamping [`dumper_invocation`] into it would make the two lanes disagree on one line of a file
/// they share, so re-dumping at each pin bump — the documented procedure — would flip that line back
/// and forth forever, and no guard would object because nothing reads `dumper`. That is exactly the
/// "two lanes rewrite one file" failure the per-backend media naming exists to prevent, so the audio
/// provenance names both invocations instead of whichever box happened to run last.
const AUDIO_DUMPER_INVOCATION: &str = "cargo run -p sceneworks-worker --bin \
                                       dump-engine-capabilities [--no-default-features --features \
                                       backend-candle]";

/// Walk the linked provider registry weights-free and group it per backend.
///
/// Reads only each registration's `descriptor` closure — no model load, no weights on disk — the
/// same introspection `mlx-gen-catalog` and [`crate::engines::registry_capabilities`] do.
pub fn collect_engine_capability_facts() -> Result<Vec<EngineCapabilityFacts>, String> {
    let registry = crate::inference_runtime::media();
    facts_from_registry(
        registry,
        crate::catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION,
        dumper_invocation(),
    )
}

/// Walk the linked **audio** provider registry weights-free and group it per backend (sc-17593).
///
/// Refuses when this build links no audio lane at all rather than skipping it silently: a skip would
/// leave stage 2 with nothing to require, which is precisely the shape of the hole this story
/// closes. Both shipped bundles carry `default = ["media", "audio"]`, so the only way to reach the
/// refusal is to have deliberately built without the audio feature.
pub fn collect_audio_capability_facts() -> Result<Vec<AudioCapabilityFacts>, String> {
    let registry = crate::inference_runtime::audio().ok_or_else(|| {
        "refusing to dump audio engine-capability facts: this build links NO audio registry. \
         Skipping it silently is how `supports_preview` went undumped for every audio route on \
         every platform in the first place. Build with the audio lane on — both runtime bundles \
         carry it default-on — or dump on a lane that does."
            .to_owned()
    })?;
    let descriptors: Vec<gen_core::ModelDescriptor> = registry
        .generators()
        .map(|registration| (registration.descriptor)())
        .collect();
    audio_facts_from_descriptors(
        &descriptors,
        crate::catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION,
        AUDIO_DUMPER_INVOCATION,
    )
}

/// Serialize one backend's facts as the exact bytes checked in (pretty JSON + trailing newline),
/// matching `apps/web/scripts/generate-styles.mjs`'s `JSON.stringify(value, null, 2) + "\n"` so a
/// JS-side rewrite and a Rust-side dump agree byte-for-byte.
pub fn facts_json(facts: &EngineCapabilityFacts) -> Result<String, String> {
    let mut json = serde_json::to_string_pretty(facts)
        .map_err(|error| format!("engine capability facts do not serialize: {error}"))?;
    json.push('\n');
    Ok(json)
}

/// [`facts_json`] for the audio dump — same bytes-on-disk contract.
pub fn audio_facts_json(facts: &AudioCapabilityFacts) -> Result<String, String> {
    let mut json = serde_json::to_string_pretty(facts)
        .map_err(|error| format!("audio engine capability facts do not serialize: {error}"))?;
    json.push('\n');
    Ok(json)
}

/// The checked-in facts directory, resolved from this crate's manifest dir so the dumper works from
/// any cwd: `<repo>/config/engine-capabilities`.
///
/// Under `config/`, which `docker/rust.Dockerfile` `COPY config ./config`s wholesale — so nothing
/// here needs a Dockerfile line, unlike an `include_str!` reaching outside a crate.
pub fn default_facts_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("engine-capabilities")
}

/// The subdirectory of the facts root that holds the audio dump.
///
/// A subdirectory rather than a longer file name so that stage 2's three media-discovery sites —
/// `readdirSync` in two scripts and the vitest `import.meta.glob` — cannot pick it up: all three are
/// single-level, while the glob's `*` would happily match `capabilities.candle.audio.json`.
pub const AUDIO_FACTS_SUBDIR: &str = "audio";

/// `<facts root>/audio` — where [`dump_audio_to`] writes.
pub fn audio_facts_dir(root: &Path) -> PathBuf {
    root.join(AUDIO_FACTS_SUBDIR)
}

/// Dump this build's facts files into `dir`, returning the paths written.
///
/// Writes **nothing** when [`collect_engine_capability_facts`] refuses, so a lane with no engines
/// cannot leave a wrong file behind.
pub fn dump_to(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let facts = collect_engine_capability_facts()?;
    std::fs::create_dir_all(dir)
        .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
    let mut written = Vec::with_capacity(facts.len());
    for entry in &facts {
        let path = dir.join(entry.file_name());
        std::fs::write(&path, facts_json(entry)?)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        written.push(path);
    }
    Ok(written)
}

/// Dump this build's **audio** facts files into `<root>/audio`, returning the paths written.
///
/// Writes nothing when [`collect_audio_capability_facts`] refuses, for the same reason [`dump_to`]
/// does: a valid-looking wrong file is worse than no file, because stage 2 reads it confidently.
pub fn dump_audio_to(root: &Path) -> Result<Vec<PathBuf>, String> {
    let facts = collect_audio_capability_facts()?;
    let dir = audio_facts_dir(root);
    std::fs::create_dir_all(&dir)
        .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
    let mut written = Vec::with_capacity(facts.len());
    for entry in &facts {
        let path = dir.join(entry.file_name());
        std::fs::write(&path, audio_facts_json(entry)?)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(
        id: &'static str,
        backend: &'static str,
        supports_preview: bool,
    ) -> gen_core::ModelDescriptor {
        descriptor_with_modality(id, backend, supports_preview, gen_core::Modality::Image)
    }

    fn descriptor_with_modality(
        id: &'static str,
        backend: &'static str,
        supports_preview: bool,
        modality: gen_core::Modality,
    ) -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            denoiser_output_latent_space: None,
            id,
            family: id,
            backend,
            modality,
            capabilities: gen_core::Capabilities {
                supports_preview,
                ..Default::default()
            },
            encoder_contract: None,
            required_components: &[],
            control_kinds: None,
        }
    }

    fn audio_descriptor(id: &'static str, supports_preview: bool) -> gen_core::ModelDescriptor {
        descriptor_with_modality(id, "candle", supports_preview, gen_core::Modality::Audio)
    }

    // The vacuous-green trap (epic 16948 decision, activity-16974). An empty registry is the
    // *normal* state on the plain Linux / non-candle lane, so a dumper that happily wrote an empty
    // file there would ship "no route supports live preview" as fact. Assert the refusal on the
    // pure function so this test runs on EVERY lane — including the ones with no engines, which is
    // precisely where the vacuous file would be produced.
    #[test]
    fn refuses_to_dump_an_empty_registry() {
        let error = facts_from_descriptors(&[], "d48023204cd3a4f3f8eb060f79803dccaddcb482", "dump")
            .expect_err("an empty descriptor list must not produce facts");
        assert!(
            error.contains("EMPTY"),
            "the refusal must name the empty registry so the operator knows the lane is wrong, got: {error}"
        );
        assert!(
            error.contains("backend-candle"),
            "the refusal must name a lane that CAN dump, got: {error}"
        );
    }

    #[test]
    fn groups_by_backend_and_sorts_engines_by_id() {
        let facts = facts_from_descriptors(
            &[
                descriptor("z_image_turbo", "candle", false),
                descriptor("krea_2_turbo", "candle", true),
                descriptor("krea_2_turbo", "mlx", true),
            ],
            "d48023204cd3a4f3f8eb060f79803dccaddcb482",
            "dump",
        )
        .expect("non-empty descriptors produce facts");

        assert_eq!(
            facts.iter().map(|f| f.backend.as_str()).collect::<Vec<_>>(),
            ["candle", "mlx"],
            "one facts file per backend, in a stable order"
        );
        assert_eq!(
            facts[0]
                .engines
                .iter()
                .map(|e| e.id.as_str())
                .collect::<Vec<_>>(),
            ["krea_2_turbo", "z_image_turbo"],
            "engines are sorted by id so the checked-in file is byte-stable"
        );
        assert!(facts[0].engines[0].supports_preview);
        assert!(!facts[0].engines[1].supports_preview);
        assert_eq!(facts[0].file_name(), "capabilities.candle.json");
        assert_eq!(facts[1].file_name(), "capabilities.mlx.json");
    }

    #[test]
    fn imported_facts_match_exact_shape_and_apply_structural_adapter_refusal() {
        let mut provider = descriptor("mage_flow_base", "mlx", true);
        provider.family = "mage-flow";
        provider.capabilities.supports_lora = true;
        provider.capabilities.supports_lokr = true;
        provider.capabilities.supported_quants = &[gen_core::Quant::Q4];
        provider.capabilities.supports_kv_cache = true;
        let route = gen_core::ImportedModelRegistration {
            family: "mage-flow",
            source: gen_core::ImportedModelSource::TransformerDirectory,
            operation: gen_core::ImportedModelOperation::Generate,
            provider_id: "mage_flow_base",
            required_components: Some(&["mage_text_encoder", "mage_vae"]),
            inherit_adapters: false,
        };

        let facts = facts_from_registry_parts(&[provider], &[route], "revision", "dump")
            .expect("exact imported facts");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].imports.len(), 1);
        let imported = &facts[0].imports[0];
        assert_eq!(imported.family, "mage-flow");
        assert_eq!(imported.source, "transformer_directory");
        assert_eq!(imported.operation, "generate");
        assert_eq!(imported.provider_id, "mage_flow_base");
        assert!(!imported.supports_lora);
        assert!(!imported.supports_lokr);
        assert_eq!(imported.supported_quants, ["q4"]);
        assert!(imported.supports_kv_cache);
        assert!(imported.registry_cached);
    }

    // The same engine id twice in one backend would make the stage-2 join through
    // MODEL_TABLE.engine_id resolve to whichever row won — a silently wrong answer for every
    // SceneWorks id mapping onto it.
    #[test]
    fn rejects_a_duplicate_engine_id_within_one_backend() {
        let error = facts_from_descriptors(
            &[
                descriptor("krea_2_turbo", "candle", true),
                descriptor("krea_2_turbo", "candle", false),
            ],
            "d48023204cd3a4f3f8eb060f79803dccaddcb482",
            "dump",
        )
        .expect_err("a duplicate engine id must be refused");
        assert!(error.contains("krea_2_turbo"), "got: {error}");
    }

    // Stage 2 compares SCENEWORKS_BACKENDS against the sorted list of dumped files, so an unsorted
    // or duplicated entry would fail the coverage check for a reason that has nothing to do with a
    // missing dump.
    #[test]
    fn sceneworks_backends_is_sorted_and_unique() {
        let mut sorted = SCENEWORKS_BACKENDS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted, SCENEWORKS_BACKENDS,
            "SCENEWORKS_BACKENDS must be sorted and duplicate-free"
        );
        assert!(
            !SCENEWORKS_BACKENDS.is_empty(),
            "an empty backend list would make the stage-2 coverage assertion vacuous — the exact \
             class of hole it exists to close"
        );
    }

    // The half of SCENEWORKS_BACKENDS this lane can actually measure. The list is a union across
    // two mutually exclusive lanes, so neither build sees it whole; what each build proves is that
    // the backend IT links is declared. An undeclared backend would ship with no stage-2 coverage
    // assertion at all — a third engine could be added, never dumped, and stay invisible exactly
    // the way `mlx` did.
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[test]
    fn the_linked_backend_is_declared() {
        let facts = collect_engine_capability_facts()
            .expect("this lane links a media registry, so collecting facts must succeed");
        assert!(
            !facts.is_empty(),
            "a lane that links a registry must produce at least one backend's facts"
        );
        for entry in &facts {
            assert!(
                SCENEWORKS_BACKENDS.contains(&entry.backend.as_str()),
                "this build links backend {:?}, which SCENEWORKS_BACKENDS ({:?}) does not declare. \
                 Stage 2 requires one dumped facts file per DECLARED backend, so an undeclared one \
                 is covered by nothing. Add it to the list and dump {} on this lane.",
                entry.backend,
                SCENEWORKS_BACKENDS,
                entry.file_name(),
            );
        }
        let imported = facts
            .iter()
            .flat_map(|entry| entry.imports.iter())
            .collect::<Vec<_>>();
        assert!(
            !imported.is_empty(),
            "the linked registry must publish its exact imported-source routes"
        );
        if facts.iter().any(|entry| entry.backend == "mlx") {
            assert!(imported.iter().any(|route| {
                route.family == "mage-flow"
                    && route.source == "transformer_directory"
                    && route.operation == "generate"
                    && !route.supports_lora
                    && !route.supports_lokr
            }));
        }
    }

    // sc-17593. The audio twin of `refuses_to_dump_an_empty_registry`, and the same trap: an audio
    // registry that registered nothing derives as "no audio route supports live preview".
    #[test]
    fn refuses_to_dump_an_empty_audio_registry() {
        let error =
            audio_facts_from_descriptors(&[], "a4f409ae8ce73eda2ee8117b89b5f479666606b8", "dump")
                .expect_err("an empty audio descriptor list must not produce facts");
        assert!(
            error.contains("EMPTY"),
            "the refusal must name the empty registry, got: {error}"
        );
        assert!(
            error.contains("audio"),
            "the refusal must say which registry is empty, got: {error}"
        );
    }

    #[test]
    fn audio_facts_carry_the_registry_discriminator_and_sort_by_id() {
        let facts = audio_facts_from_descriptors(
            &[
                audio_descriptor("moss_sfx_v2", false),
                audio_descriptor("chatterbox_tts", false),
            ],
            "a4f409ae8ce73eda2ee8117b89b5f479666606b8",
            "dump",
        )
        .expect("non-empty audio descriptors produce facts");

        assert_eq!(
            facts.len(),
            1,
            "AUDIO_BACKEND is one value on every platform"
        );
        assert_eq!(facts[0].registry, AUDIO_REGISTRY_LABEL);
        assert_eq!(facts[0].backend, "candle");
        assert_eq!(
            facts[0]
                .engines
                .iter()
                .map(|engine| engine.id.as_str())
                .collect::<Vec<_>>(),
            ["chatterbox_tts", "moss_sfx_v2"],
            "sorted by id so the checked-in file is byte-stable"
        );
        assert_eq!(facts[0].file_name(), "capabilities.candle.json");
    }

    // The discriminator is what keeps a misplaced file from being read as the other kind, so assert
    // it is actually IN the bytes rather than only on the struct. Media files carry no `registry`
    // key at all; their optional engine-level descriptor facts do not change that namespace rule.
    #[test]
    fn audio_json_is_discriminated_and_media_json_has_no_registry_key() {
        let audio = audio_facts_from_descriptors(
            &[audio_descriptor("kokoro_82m", false)],
            "a4f409ae8ce73eda2ee8117b89b5f479666606b8",
            "dump",
        )
        .expect("audio facts");
        let json: serde_json::Value =
            serde_json::from_str(&audio_facts_json(&audio[0]).expect("serializes"))
                .expect("parses");
        assert_eq!(json["registry"], AUDIO_REGISTRY_LABEL);
        assert_eq!(json["backend"], "candle");
        assert_eq!(json["engines"][0]["modality"], "audio");
        assert_eq!(json["engines"][0]["supportsPreview"], false);
        assert!(json["engines"][0]
            .get("denoiserOutputLatentSpace")
            .is_none());

        let media = facts_from_descriptors(
            &[descriptor("krea_2_turbo", "candle", true)],
            "a4f409ae8ce73eda2ee8117b89b5f479666606b8",
            "dump",
        )
        .expect("media facts");
        let media_json: serde_json::Value =
            serde_json::from_str(&facts_json(&media[0]).expect("serializes")).expect("parses");
        assert!(
            media_json.get("registry").is_none(),
            "media facts must remain distinguishable from the audio registry"
        );
    }

    #[test]
    fn rejects_a_duplicate_audio_engine_id() {
        let error = audio_facts_from_descriptors(
            &[
                audio_descriptor("kokoro_82m", false),
                audio_descriptor("kokoro_82m", true),
            ],
            "a4f409ae8ce73eda2ee8117b89b5f479666606b8",
            "dump",
        )
        .expect_err("a duplicate audio engine id must be refused");
        assert!(error.contains("kokoro_82m"), "got: {error}");
    }

    #[test]
    fn sceneworks_audio_backends_is_sorted_and_unique() {
        let mut sorted = SCENEWORKS_AUDIO_BACKENDS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted, SCENEWORKS_AUDIO_BACKENDS,
            "SCENEWORKS_AUDIO_BACKENDS must be sorted and duplicate-free"
        );
        assert!(
            !SCENEWORKS_AUDIO_BACKENDS.is_empty(),
            "an empty audio backend list would make the stage-2 audio coverage assertion vacuous"
        );
    }

    // Unlike SCENEWORKS_BACKENDS, this list is NOT a union across mutually exclusive lanes: audio is
    // candle-native everywhere, so either lane links the whole audio catalog and can verify the list
    // completely. That is also why both lanes' audio dumps are byte-identical.
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[test]
    fn the_linked_audio_backend_is_declared() {
        let facts = collect_audio_capability_facts()
            .expect("this lane links the audio registry, so collecting audio facts must succeed");
        assert!(
            !facts.is_empty(),
            "a lane that links the audio registry must produce at least one backend's facts"
        );
        let mut linked: Vec<&str> = facts.iter().map(|entry| entry.backend.as_str()).collect();
        linked.sort_unstable();
        assert_eq!(
            linked, SCENEWORKS_AUDIO_BACKENDS,
            "the audio registry is platform-independent, so the backends it links must be EXACTLY \
             SCENEWORKS_AUDIO_BACKENDS — not merely a subset. A backend missing here means the \
             const over-declares and stage 2 asks for a dump nothing can produce; an extra one \
             means it under-declares and that backend's dump is required by nothing."
        );
    }

    // The measurement sc-17593 asked for before choosing a shape, pinned so it cannot quietly stop
    // being true: at the live pin EVERY audio generator reports `supports_preview: false`, which is
    // why this story is a completeness fix (unknown -> false) rather than a user-visible one. If a
    // future pin lights one up, this test goes red and the follow-up work is a UI question.
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[test]
    fn no_audio_generator_advertises_live_preview_at_this_pin() {
        let facts = collect_audio_capability_facts().expect("audio facts");
        let advertising: Vec<&str> = facts
            .iter()
            .flat_map(|entry| entry.engines.iter())
            .filter(|engine| engine.supports_preview)
            .map(|engine| engine.id.as_str())
            .collect();
        assert!(
            advertising.is_empty(),
            "an audio generator now advertises live preview ({advertising:?}). That is a real \
             capability change, not a test to relax: the derived catalog will start answering \
             `true` for an audio route, and `previewSupport.js`'s PREVIEW_JOB_TYPES gate (image \
             jobs only) means the UI still cannot show it. Decide what the audio card does first."
        );
    }

    // The audio file is the ONE file both lanes write, so every byte of it has to be
    // lane-independent or the documented "re-dump on each lane at every pin bump" procedure makes
    // it oscillate in git history — silently, since nothing reads `dumper`. The engine list already
    // is (candle-audio-catalog gates no registration on the platform); provenance is the field that
    // could quietly stop being.
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[test]
    fn the_audio_dump_provenance_does_not_vary_by_lane() {
        let facts = collect_audio_capability_facts().expect("audio facts");
        for entry in &facts {
            assert_eq!(
                entry.generated_from.dumper, AUDIO_DUMPER_INVOCATION,
                "the audio dump must stamp the lane-neutral invocation, not this build's own — \
                 otherwise macOS and candle write different bytes into the same file"
            );
            assert_ne!(
                entry.generated_from.dumper,
                dumper_invocation(),
                "AUDIO_DUMPER_INVOCATION has collapsed into the platform-branched media one"
            );
        }
        // Names both lanes, so a reader of the file knows either box regenerates it.
        assert!(AUDIO_DUMPER_INVOCATION.contains("backend-candle"));
        assert!(AUDIO_DUMPER_INVOCATION.contains('['));
    }

    #[test]
    fn serializes_as_camel_case_json_with_a_trailing_newline() {
        let facts = facts_from_descriptors(
            &[descriptor("krea_2_turbo", "candle", true)],
            "d48023204cd3a4f3f8eb060f79803dccaddcb482",
            "cargo run …",
        )
        .expect("facts");
        let json = facts_json(&facts[0]).expect("serializes");
        assert!(json.ends_with("}\n"), "one trailing newline, got: {json:?}");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parses");
        assert_eq!(parsed["backend"], "candle");
        assert_eq!(
            parsed["generatedFrom"]["inferenceRevision"],
            "d48023204cd3a4f3f8eb060f79803dccaddcb482"
        );
        assert_eq!(parsed["engines"][0]["id"], "krea_2_turbo");
        assert_eq!(parsed["engines"][0]["supportsPreview"], true);
        assert_eq!(parsed["engines"][0]["modality"], "image");
    }

    #[test]
    fn latent_space_facts_are_exact_fail_closed_and_round_trip() {
        let mut qwen = descriptor("qwen", "mlx", true);
        qwen.denoiser_output_latent_space = Some(&gen_core::QWEN_KREA_Z16_LATENT_SPACE);
        let mut wan = descriptor("wan", "mlx", false);
        wan.denoiser_output_latent_space = Some(&gen_core::WAN_Z16_VIDEO_LATENT_SPACE);
        let mut flux2 = descriptor("flux2", "mlx", true);
        flux2.denoiser_output_latent_space = Some(&gen_core::FLUX2_PACKED_LATENT_SPACE);
        let mut mochi = descriptor("mochi", "mlx", false);
        mochi.denoiser_output_latent_space = Some(&gen_core::MOCHI_VIDEO_LATENT_SPACE);
        let mut ltx = descriptor("ltx", "mlx", false);
        ltx.denoiser_output_latent_space = Some(&gen_core::LTX_VIDEO_LATENT_SPACE);
        let unknown = descriptor("unknown", "mlx", false);

        let facts = facts_from_descriptors(
            &[qwen, wan, flux2, mochi, ltx, unknown],
            "d48023204cd3a4f3f8eb060f79803dccaddcb482",
            "cargo run …",
        )
        .expect("facts");
        let json = facts_json(&facts[0]).expect("serializes");
        let round_trip: EngineCapabilityFacts =
            serde_json::from_str(&json).expect("typed facts deserialize");
        assert_eq!(round_trip, facts[0]);

        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parses");
        let engine = |id: &str| {
            parsed["engines"]
                .as_array()
                .expect("engine array")
                .iter()
                .find(|engine| engine["id"] == id)
                .unwrap_or_else(|| panic!("missing {id}"))
        };
        let qwen = &engine("qwen")["denoiserOutputLatentSpace"];
        let wan = &engine("wan")["denoiserOutputLatentSpace"];
        assert_eq!(qwen["channels"], 16);
        assert_eq!(qwen["patchLayout"]["kind"], "unpacked");
        assert_eq!(qwen["temporalLaw"], "none");
        assert_eq!(wan["temporalLaw"], "causal-4x");
        assert_eq!(
            engine("mochi")["denoiserOutputLatentSpace"]["temporalLaw"],
            "causal-6x"
        );
        assert_eq!(
            engine("ltx")["denoiserOutputLatentSpace"]["temporalLaw"],
            "causal-8x"
        );
        assert_eq!(qwen["normalization"], wan["normalization"]);
        assert!(qwen["normalization"]["contentHash"]
            .as_str()
            .expect("string hash")
            .starts_with("fnv1a64:"));
        assert_eq!(
            engine("flux2")["denoiserOutputLatentSpace"]["patchLayout"]["kind"],
            "packed"
        );
        assert_eq!(
            engine("flux2")["denoiserOutputLatentSpace"]["normalization"]["kind"],
            "learnedPerChannel"
        );
        assert!(engine("unknown").get("denoiserOutputLatentSpace").is_none());
    }
}
