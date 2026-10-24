//! `KokoroGenerator` — the [`gen_core::Generator`] implementation for **Kokoro-82M TTS** on the
//! candle audio lane (sc-12836), plus its [`descriptor`]/[`load`] entry points and the explicit
//! registration constant wired into `candle-audio-catalog` under the id `"kokoro_82m"` — the
//! first real audio provider (epic sc-12833).
//!
//! ## Snapshot layout
//!
//! [`load`] expects a `hexgrad/Kokoro-82M`-shaped snapshot directory:
//!
//! ```text
//!   config.json          → hyperparameters + phoneme vocab
//!   kokoro-v1_0.pth      → the five-section torch checkpoint (see crate::weights)
//!   voices/<voice>.pt    → per-voice style-vector packs (resolved lazily per request)
//! ```
//!
//! A `LoadSpec` points at exactly that layout: the snapshot is staged locally and passed in, never
//! self-fetched (epic 13657). The `HUB_REPO`@`HUB_REVISION` pin records its provenance.
//!
//! ## Request mapping
//!
//! `prompt` is the script text. The [`gen_core::AudioParams`] sub-block supplies `voice`
//! (default `af_heart`; the leading `a`/`b` selects the American/British G2P variant),
//! `language` (`en` family only), `sample_rate` (native 24 000 Hz only), and
//! `target_duration` — honored by deriving a speed factor from the duration head's natural
//! estimate (clamped to 0.5–2.0×, the model's usable range). Progress is 5 stage steps;
//! cancellation is checked before generate, at every stage boundary, AND inside the
//! dominant-cost decoder/vocoder stage (`crate::decoder::CancelProbe`), returning the typed
//! [`gen_core::Error::Canceled`]. Determinism: same request + seed ⇒ byte-identical samples.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_audio::gen_core::{
    self, AudioTrack, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, Progress, WeightsSource,
};
use candle_audio::{AudioError, Result as AudioResult};

use crate::decoder::SAMPLE_RATE;
use crate::g2p::{EnglishVariant, KokoroG2p};
use crate::pipeline::{KokoroPipeline, SAMPLES_PER_FRAME, STAGES};
use crate::weights::VoicePack;

/// Registry id (the SceneWorks worker routes `payload.model` to this exact id).
pub const MODEL_ID: &str = "kokoro_82m";

/// Hub pin: `hexgrad/Kokoro-82M` at an immutable commit SHA (F-029; Apache-2.0 weights).
pub const HUB_REPO: &str = "hexgrad/Kokoro-82M";
pub const HUB_REVISION: &str = "f3ff3571791e39611d31c381e3a41a3af07b4987";

/// Stable component key for the pinned Kokoro-82M checkpoint — what `PROVIDER_COMPONENTS`
/// resolves through, and the licence manifest's unique row key.
pub const COMPONENT_KEY: &str = "kokoro_82m";

/// The schema-3 licence row for the pinned Kokoro-82M checkpoint (sc-16663).
///
/// **Disclosure only.** The row records what the upstream declares so a consumer can show it to a
/// user; nothing here decides whether any use is permitted. `declared` and `gated` were read from
/// the `hexgrad/Kokoro-82M` model card on `retrieved`, and `family` normalizes that declaration onto
/// [`candle_audio::gen_core::license::families::APACHE_2_0`].
pub const COMPONENT_LICENSE: candle_audio::gen_core::ComponentLicense =
    candle_audio::gen_core::ComponentLicense {
        component: COMPONENT_KEY,
        source_url: "https://huggingface.co/hexgrad/Kokoro-82M",
        gated: false,
        declared: "apache-2.0",
        family: "apache-2-0",
        attribution: Some("Kokoro-82M © hexgrad — licensed under Apache-2.0"),
        retrieved: "2026-08-02",
    };

/// The advertised voice surface: every English voice the pinned snapshot ships (leading
/// `a` = American English, `b` = British English — the prefix selects the G2P variant).
/// Non-English voice packs exist upstream but are NOT advertised until their language
/// front-ends land (languages = `en` family only at this release).
pub const VOICES: &[&str] = &[
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
];

/// Default voice when the request supplies none (the upstream showcase voice).
pub const DEFAULT_VOICE: &str = "af_heart";

/// Advertised language codes (all resolve to the English front-end; `en-gb` only steers the
/// default variant when the voice itself is ambiguous — voices carry their own variant).
pub const LANGUAGES: &[&str] = &["en", "en-us", "en-gb"];

/// Longest clip advertised (seconds). The 512-token context bounds a single synthesis well
/// above this; 30 s keeps single-request latency and memory predictable.
pub const MAX_DURATION_SECS: f32 = 30.0;

/// Kokoro's identity + capabilities — constructible without weights (registry introspection).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "kokoro",
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            // Pure audio: no width/height. The descriptor sweep exempts Audio from the size floor
            // (sc-13314) and `validate_request_audio` skips the range, so these stay at the natural
            // unused 0 rather than a nominal placeholder bound.
            min_size: 0,
            // One clip per request (GenerationOutput::Audio carries a single track).
            max_count: 1,
            audio_sample_rates: vec![SAMPLE_RATE],
            max_audio_duration_secs: Some(MAX_DURATION_SECS),
            audio_voices: VOICES.to_vec(),
            audio_languages: LANGUAGES.to_vec(),
            ..Default::default()
        },
    }
}

/// Capability-driven request validation, factored out for weightless unit tests. Shared floor
/// checks ([`Capabilities::validate_request_audio`]) plus Kokoro's own: non-empty prompt.
pub(crate) fn validate_request(
    desc: &ModelDescriptor,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    let id = desc.id;
    if req.prompt.trim().is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt (the script text) must not be empty"
        )));
    }
    // Pure audio: width/height are unused, so the descriptor advertises no size bounds (sc-13314)
    // and the audio floor skips the size range entirely.
    let caps = &desc.capabilities;
    caps.validate_request_audio(id, req)
}

/// A loaded (lazy) Kokoro generator. Heavy state — the pipeline, per-voice style packs, and
/// the G2P engines (embedded-lexicon parses) — is built on first use and cached; `load` does
/// no file I/O beyond argument checks (the sibling providers' lazy-load discipline).
pub struct KokoroGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    pipeline: Mutex<Option<Arc<KokoroPipeline>>>,
    voices: Mutex<HashMap<String, Arc<VoicePack>>>,
    g2p_us: Mutex<Option<Arc<KokoroG2p>>>,
    g2p_gb: Mutex<Option<Arc<KokoroG2p>>>,
}

impl KokoroGenerator {
    fn pipeline(&self) -> gen_core::Result<Arc<KokoroPipeline>> {
        let mut guard = lock_recover(&self.pipeline);
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        let device = candle_audio::default_device()?;
        let built = Arc::new(KokoroPipeline::from_snapshot(&self.root, &device)?);
        *guard = Some(built.clone());
        Ok(built)
    }

    fn voice_pack(&self, voice: &str) -> gen_core::Result<Arc<VoicePack>> {
        let mut guard = lock_recover(&self.voices);
        if let Some(v) = guard.get(voice) {
            return Ok(v.clone());
        }
        let path = self.root.join("voices").join(format!("{voice}.pt"));
        if !path.is_file() {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: voice pack {} missing from the snapshot (the passed-in snapshot must \
                 contain every advertised voice)",
                path.display()
            )));
        }
        let pack = Arc::new(VoicePack::from_file(&path)?);
        guard.insert(voice.to_string(), pack.clone());
        Ok(pack)
    }

    fn g2p(&self, variant: EnglishVariant) -> Arc<KokoroG2p> {
        let cell = match variant {
            EnglishVariant::American => &self.g2p_us,
            EnglishVariant::British => &self.g2p_gb,
        };
        let mut guard = lock_recover(cell);
        if let Some(g) = guard.as_ref() {
            return g.clone();
        }
        let built = Arc::new(KokoroG2p::new(variant));
        *guard = Some(built.clone());
        built
    }
}

/// Recover a poisoned mutex (a prior panic mid-build leaves `None`/stale state, which the
/// lazy builders tolerate) — the audio twin of `candle_gen::lock_recover`.
fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Which English variant a voice speaks (its name prefix: `b…` = British, else American).
fn voice_variant(voice: &str, language: Option<&str>) -> EnglishVariant {
    if voice.starts_with('b') {
        EnglishVariant::British
    } else if voice.starts_with('a') {
        EnglishVariant::American
    } else if language == Some("en-gb") {
        EnglishVariant::British
    } else {
        EnglishVariant::American
    }
}

/// Construct the (lazy) Kokoro generator from a [`LoadSpec`]. `spec.weights` must be a
/// snapshot directory (see module docs); adapters/quantization/control overlays are rejected —
/// refusing is more honest than silently dropping.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID} expects a snapshot directory (config.json + {} + voices/), not a \
                 single file",
                crate::pipeline::CHECKPOINT_FILE
            )));
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support on-the-fly quantization"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support LoRA/LoKr adapters"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support control/IP-adapter overlays"
        )));
    }
    Ok(Box::new(KokoroGenerator {
        descriptor: descriptor(),
        root,
        pipeline: Mutex::new(None),
        voices: Mutex::new(HashMap::new()),
        g2p_us: Mutex::new(None),
        g2p_gb: Mutex::new(None),
    }))
}

impl Generator for KokoroGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        // Pre-generate cancellation seam (sc-11128 class): consult the flag before ANY heavy
        // work — G2P engine construction, checkpoint load, and synthesis all come after this.
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let audio = req.audio.clone().unwrap_or_default();
        let voice = audio.voice.as_deref().unwrap_or(DEFAULT_VOICE);
        let variant = voice_variant(voice, audio.language.as_deref());

        // G2P + tokenization (cheap; before any tensor work).
        let phonemes = self.g2p(variant).phonemize(&req.prompt)?;
        let pipeline = self.pipeline()?;
        let tokens = pipeline.config.phonemes_to_ids(&phonemes);
        let pack = self.voice_pack(voice)?;
        let ref_s = pack.ref_s(tokens.len()).to_vec();

        // target_duration → speed factor from the duration head's natural estimate, clamped
        // to the model's usable range.
        let speed = match audio.target_duration {
            Some(target) if target > 0.0 => {
                let raw = pipeline.raw_durations(&tokens, &ref_s)?;
                let natural = pipeline.natural_duration_secs(&raw);
                (natural / target).clamp(0.5, 2.0)
            }
            _ => 1.0,
        };

        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let cancel = req.cancel.clone();
        let mut stage = |current: u32| -> AudioResult<()> {
            // The vocoder is the terminal decode phase (Progress contract: Decoding exactly
            // once) — announce it before the last stage runs.
            if current == STAGES - 1 {
                on_progress(Progress::Decoding);
            }
            on_progress(Progress::Step {
                current,
                total: STAGES,
            });
            if cancel.is_cancelled() {
                return Err(AudioError::Canceled);
            }
            Ok(())
        };
        // The probe reaches inside the dominant stage-5 decoder/vocoder so a cancel lands
        // promptly mid-synthesis, not only at stage boundaries.
        let probe = req.cancel.clone();
        let samples =
            pipeline.synthesize(&tokens, &ref_s, speed, seed, &mut stage, &move || {
                probe.is_cancelled()
            })?;

        // Guard the advertised duration cap at the output too (the floor already bounds
        // target_duration; a runaway duration head must not exceed the advertisement wildly).
        let secs = samples.len() as f32 / SAMPLE_RATE as f32;
        if secs > 4.0 * MAX_DURATION_SECS {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: predicted {secs:.1}s exceeds the sane bound — refusing"
            )));
        }
        Ok(GenerationOutput::Audio(AudioTrack {
            samples,
            sample_rate: SAMPLE_RATE,
            channels: 1,
            ..Default::default()
        }))
    }
}

// Explicit catalog registration for `kokoro_82m` (composed by `candle-audio-catalog`).
candle_audio::register_generators! {
    pub const REGISTRATION = descriptor => load
}

/// Rough upper bound on synthesized seconds for a token count (`max_dur` frames per token) —
/// exposed for consumers sizing buffers; not a validation gate.
pub fn max_seconds_for_tokens(n_tokens: usize, max_dur: usize) -> f32 {
    (n_tokens * max_dur * SAMPLES_PER_FRAME) as f32 / SAMPLE_RATE as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{AudioParams, CancelFlag};

    fn audio_req(audio: AudioParams) -> GenerationRequest {
        GenerationRequest {
            prompt: "Hello there.".into(),
            audio: Some(audio),
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_advertises_the_audio_surface() {
        let d = descriptor();
        assert_eq!(d.id, "kokoro_82m");
        assert_eq!(d.backend, "candle");
        assert!(matches!(d.modality, Modality::Audio));
        assert_eq!(d.capabilities.audio_sample_rates, [24_000]);
        assert!(d.capabilities.audio_voices.contains(&"af_heart"));
        assert!(d.capabilities.audio_voices.contains(&"bm_george"));
        assert_eq!(d.capabilities.max_count, 1);
        // Every advertised voice is an English pack (languages = en family only).
        assert!(VOICES
            .iter()
            .all(|v| v.starts_with('a') || v.starts_with('b')));
    }

    #[test]
    fn validate_gates_voice_language_and_sample_rate() {
        let d = descriptor();
        // In-surface request passes.
        assert!(validate_request(
            &d,
            &audio_req(AudioParams {
                voice: Some("af_heart".into()),
                language: Some("en".into()),
                sample_rate: Some(24_000),
                ..Default::default()
            })
        )
        .is_ok());
        // Unadvertised voice / language / sample rate are rejected.
        for bad in [
            AudioParams {
                voice: Some("zf_xiaobei".into()),
                ..Default::default()
            },
            AudioParams {
                language: Some("ja".into()),
                ..Default::default()
            },
            AudioParams {
                sample_rate: Some(44_100),
                ..Default::default()
            },
            AudioParams {
                target_duration: Some(MAX_DURATION_SECS + 1.0),
                ..Default::default()
            },
        ] {
            assert!(validate_request(&d, &audio_req(bad)).is_err());
        }
        // Empty prompt is rejected.
        let mut r = audio_req(AudioParams::default());
        r.prompt = "  ".into();
        assert!(validate_request(&d, &r).is_err());
    }

    #[test]
    fn voice_prefix_selects_the_g2p_variant() {
        assert_eq!(voice_variant("af_heart", None), EnglishVariant::American);
        assert_eq!(voice_variant("bm_george", None), EnglishVariant::British);
        assert_eq!(
            voice_variant("bf_alice", Some("en")),
            EnglishVariant::British
        );
    }

    #[test]
    fn load_rejects_unsupported_spec_shapes() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let spec = LoadSpec::new(WeightsSource::File(dir.join("x.pth")));
        assert!(load(&spec).is_err());
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
        spec.quantize = Some(gen_core::Quant::Q4);
        assert!(matches!(load(&spec), Err(gen_core::Error::Unsupported(_))));
    }

    #[test]
    fn pre_tripped_cancel_returns_typed_canceled_before_any_heavy_work() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let g = load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let flag = CancelFlag::new();
        flag.cancel();
        let req = GenerationRequest {
            prompt: "hi".into(),
            cancel: flag,
            ..Default::default()
        };
        // The pre-generate seam fires before G2P/checkpoint work — typed Canceled, even though
        // this snapshot dir has no weights at all.
        let err = g.generate(&req, &mut |_| {}).unwrap_err();
        assert!(matches!(err, gen_core::Error::Canceled));
    }

    #[test]
    fn generate_on_a_missing_snapshot_fails_cleanly() {
        // A generator over an empty dir: generate must error (no weights), never panic.
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let g = load(&LoadSpec::new(WeightsSource::Dir(dir))).unwrap();
        let req = GenerationRequest {
            prompt: "hi".into(),
            ..Default::default()
        };
        let err = g.generate(&req, &mut |_| {}).unwrap_err();
        assert!(!matches!(err, gen_core::Error::Canceled));
    }
}
