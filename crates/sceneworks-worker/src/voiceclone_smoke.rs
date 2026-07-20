//! Local real-weight smoke for the **Voice Clone** two-call chain (epic 13400 / sc-13411, C4).
//! `#[ignore]`d — run by hand on an Apple-Silicon Mac. It drives the exact runtime seams the
//! worker's voice-clone job (`audio_jobs::run_voice_clone_synthesis`) uses, minus the API/job plumbing:
//!
//!   1. **Base TTS** — `runtime_macos::catalog().audio().load("kokoro_82m")` speaks the script in a
//!      default voice (the "content" clip).
//!   2. **Reference voice** — a SECOND real Kokoro clip in a distinctly different voice
//!      (`am_michael`, male) stands in for a user-supplied reference-voice sample. Any real clip
//!      works; a Kokoro clip in another voice gives a controlled, reproducible target timbre.
//!   3. **OpenVoice V2 conversion** — `load_audio_transform("openvoice_v2")` +
//!      `AudioTransform::apply` with the base clip as the source and the reference clip as
//!      `target_reference`, `strength` overriding the posterior-sampling temperature τ. This is the
//!      real tone-color transfer.
//!   4. **Chatterbox-VE evidence** — `load_voice_embedder("chatterbox_ve")` embeds the base,
//!      reference, and converted clips; the converted clip's speaker embedding must move measurably
//!      TOWARD the reference (cosine(converted, ref) > cosine(base, ref)). That is the objective,
//!      ear-free proof that the timbre moved toward the target — a true audible match still needs a
//!      listen, so the WAVs are written out for that.
//!
//! Setup — the three snapshots install through the ordinary Audio Studio model download; this smoke
//! resolves them from the HF hub cache (override with the env vars if they live elsewhere):
//! ```text
//! cargo test -p sceneworks-worker --release voiceclone_chain_smoke -- --ignored --nocapture
//! # overrides: VOICECLONE_KOKORO_DIR / VOICECLONE_OPENVOICE_DIR (the converter/ dir) /
//! #            VOICECLONE_CHATTERBOX_DIR, VOICECLONE_STRENGTH (τ, default 0.3),
//! #            VOICECLONE_OUT_DIR (default /tmp/voiceclone_smoke)
//! ```

use std::path::{Path, PathBuf};

use gen_core::{
    AudioParams, AudioTransformRequest, CancelFlag, Conditioning, GenerationOutput,
    GenerationRequest, LoadSpec, Progress, WeightsSource,
};

use super::smoke_support::env_or;
use crate::video_jobs::{write_wav_pcm16, AudioTrack};

/// The script the base TTS speaks (and the converted clip preserves).
const SCRIPT: &str = "SceneWorks turns a short reference clip into a cloned voiceover.";
/// The reference-voice clip's script — content is irrelevant to tone-color extraction; a different
/// sentence keeps it honest that the converter transfers TIMBRE, not content.
const REFERENCE_SCRIPT: &str = "This is my natural speaking voice, captured for cloning.";
/// The base (source/content) voice — Kokoro's default female voice.
const BASE_VOICE: &str = "af_heart";
/// The reference (target) voice — a distinctly different male voice, so the tone-color move is large
/// and the Chatterbox-VE evidence is unambiguous.
const REFERENCE_VOICE: &str = "am_michael";

/// The native clone smoke's script (sc-13412) — the words rendered in the reference voice.
const NATIVE_CLONE_SCRIPT: &str =
    "Hello there. This sentence is spoken in a cloned voice by the synthesizer.";
/// The reference voice the native clone must track — a female Kokoro voice.
const REFERENCE_VOICE_FEMALE: &str = "af_heart";
/// A DIFFERENT (male) control voice the clone must be measurably farther from than the reference.
const CONTROL_VOICE_MALE: &str = "am_michael";

/// Find the single HF-hub snapshot dir for `models--<owner>--<repo>` under `~/.cache/huggingface`,
/// or `None` if the cache is laid out elsewhere (then the caller falls back to the env override).
fn hub_snapshot(owner_repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let cache_key = format!("models--{}", owner_repo.replace('/', "--"));
    let snapshots = Path::new(&home)
        .join(".cache/huggingface/hub")
        .join(cache_key)
        .join("snapshots");
    let mut entries = std::fs::read_dir(&snapshots).ok()?;
    entries
        .find_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
}

/// Resolve a model dir from an env override or the HF hub cache, panicking with a clear message if
/// neither resolves — this is a manual smoke, so a missing snapshot should say exactly what to install.
fn resolve_dir(env_key: &str, owner_repo: &str, subdir: Option<&str>) -> PathBuf {
    let base = match std::env::var(env_key).ok().filter(|v| !v.trim().is_empty()) {
        Some(v) => PathBuf::from(v.trim()),
        None => hub_snapshot(owner_repo).unwrap_or_else(|| {
            panic!(
                "{env_key} unset and no HF hub snapshot for {owner_repo} — install the model via the \
                 Audio Studio (or `hf download {owner_repo}`) or set {env_key}"
            )
        }),
    };
    let dir = match subdir {
        Some(sub) if base.join(sub).is_dir() => base.join(sub),
        _ => base,
    };
    assert!(
        dir.is_dir(),
        "{env_key}: {} is not a directory",
        dir.display()
    );
    dir
}

/// Generate one real Kokoro clip (the base TTS call), returning the produced `gen_core::AudioTrack`.
fn kokoro_clip(dir: &Path, script: &str, voice: &str) -> gen_core::AudioTrack {
    let audio = runtime_macos::catalog()
        .expect("runtime catalog")
        .audio()
        .expect("audio lane")
        .load(
            "kokoro_82m",
            &LoadSpec::new(WeightsSource::Dir(dir.to_path_buf())),
        )
        .expect("load kokoro");
    let req = GenerationRequest {
        prompt: script.to_owned(),
        audio: Some(AudioParams {
            voice: Some(voice.to_owned()),
            language: Some("en-us".to_owned()),
            ..Default::default()
        }),
        cancel: CancelFlag::new(),
        ..Default::default()
    };
    let mut on_progress = |_p: Progress| {};
    match audio
        .generate(&req, &mut on_progress)
        .expect("kokoro generate")
    {
        GenerationOutput::Audio(track) => track,
        _ => panic!("kokoro returned non-audio output"),
    }
}

/// Peak absolute sample amplitude — the "is it silent?" check.
fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()))
}

/// Cosine similarity of two raw embedding vectors (both must be the same length).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Write a `gen_core::AudioTrack` out as a PCM-16 WAV for a manual listen.
fn dump_wav(track: &gen_core::AudioTrack, path: &Path) {
    let wav = AudioTrack {
        samples: track.samples.clone(),
        sample_rate: track.sample_rate.max(1),
        channels: track.channels.max(1),
    };
    write_wav_pcm16(&wav, path).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

#[test]
#[ignore = "real-weight audio smoke: run by hand on an Apple-Silicon Mac"]
fn voiceclone_chain_smoke() {
    let kokoro_dir = resolve_dir("VOICECLONE_KOKORO_DIR", "hexgrad/Kokoro-82M", None);
    // OpenVoice's converter files live under `converter/` in the snapshot.
    let openvoice_dir = resolve_dir(
        "VOICECLONE_OPENVOICE_DIR",
        "myshell-ai/OpenVoiceV2",
        Some("converter"),
    );
    let chatterbox_dir = resolve_dir("VOICECLONE_CHATTERBOX_DIR", "ResembleAI/chatterbox", None);
    let out_dir = PathBuf::from(env_or("VOICECLONE_OUT_DIR", "/tmp/voiceclone_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let tau: f32 = env_or("VOICECLONE_STRENGTH", "0.3")
        .parse()
        .expect("τ float");

    // ── Call 1: base TTS (the content clip) ──────────────────────────────────────────────────────
    let base = kokoro_clip(&kokoro_dir, SCRIPT, BASE_VOICE);
    assert!(!base.samples.is_empty(), "base clip is empty");
    let base_peak = peak(&base.samples);
    assert!(base_peak > 0.0, "base clip is silent");
    dump_wav(&base, &out_dir.join("base_kokoro.wav"));

    // A separate real clip in a DIFFERENT voice stands in for the user's reference-voice sample.
    let reference = kokoro_clip(&kokoro_dir, REFERENCE_SCRIPT, REFERENCE_VOICE);
    assert!(peak(&reference.samples) > 0.0, "reference clip is silent");
    dump_wav(&reference, &out_dir.join("reference.wav"));

    // ── Call 2: OpenVoice V2 tone-color conversion ───────────────────────────────────────────────
    let converter = runtime_macos::catalog()
        .expect("runtime catalog")
        .audio()
        .expect("audio lane")
        .load_audio_transform(
            "openvoice_v2",
            &LoadSpec::new(WeightsSource::Dir(openvoice_dir.clone())),
        )
        .expect("load openvoice_v2 transform");
    let req = AudioTransformRequest {
        audio: base.clone(),
        target_reference: Some(reference.clone()),
        strength: Some(tau),
        seed: Some(7),
        ..Default::default()
    };
    let mut on_progress = |_p: Progress| {};
    let converted = converter
        .apply(&req, &mut on_progress)
        .expect("openvoice apply")
        .into_iter()
        .next()
        .expect("openvoice produced a track");
    assert!(!converted.samples.is_empty(), "converted clip is empty");
    let conv_peak = peak(&converted.samples);
    assert!(conv_peak > 0.0, "converted clip is silent");
    dump_wav(&converted, &out_dir.join("converted.wav"));

    // The converted clip must NOT be a copy of the base clip: OpenVoice emits at its native 22.05 kHz
    // (Kokoro is 24 kHz), so the rate alone already differs — but assert the WAVEFORM moved too.
    assert_eq!(converted.sample_rate, 22_050, "openvoice native rate");
    assert_eq!(converted.channels, 1, "openvoice mono output");

    // ── Chatterbox-VE evidence: did the tone color move toward the reference? ─────────────────────
    let embedder = runtime_macos::catalog()
        .expect("runtime catalog")
        .audio()
        .expect("audio lane")
        .load_voice_embedder(
            "chatterbox_ve",
            // Chatterbox-VE loads the single `ve.safetensors`, not a snapshot dir.
            &LoadSpec::new(WeightsSource::File(chatterbox_dir.join("ve.safetensors"))),
        )
        .expect("load chatterbox_ve embedder");
    let emb_base = embedder.embed(&base).expect("embed base");
    let emb_ref = embedder.embed(&reference).expect("embed reference");
    let emb_conv = embedder.embed(&converted).expect("embed converted");

    let sim_base_ref = cosine(&emb_base, &emb_ref);
    let sim_conv_ref = cosine(&emb_conv, &emb_ref);
    let sim_conv_base = cosine(&emb_conv, &emb_base);

    eprintln!("[voiceclone-smoke] out dir: {}", out_dir.display());
    eprintln!(
        "[voiceclone-smoke] base:      {} samples @ {} Hz, peak {:.4}",
        base.samples.len(),
        base.sample_rate,
        base_peak
    );
    eprintln!(
        "[voiceclone-smoke] reference: {} samples @ {} Hz",
        reference.samples.len(),
        reference.sample_rate
    );
    eprintln!(
        "[voiceclone-smoke] converted: {} samples @ {} Hz, peak {:.4} (τ={tau})",
        converted.samples.len(),
        converted.sample_rate,
        conv_peak
    );
    eprintln!(
        "[voiceclone-smoke] Chatterbox-VE cosine — base↔ref {sim_base_ref:.4} | \
         converted↔ref {sim_conv_ref:.4} | converted↔base {sim_conv_base:.4}"
    );

    // The core acceptance evidence: the converted clip's speaker identity is closer to the reference
    // than the base clip's is — the tone color moved toward the target voice.
    assert!(
        sim_conv_ref > sim_base_ref,
        "tone color did not move toward the reference: converted↔ref {sim_conv_ref:.4} \
         must exceed base↔ref {sim_base_ref:.4}"
    );
    eprintln!(
        "[voiceclone-smoke] PASS — tone color moved toward the reference by \
         {:.4} cosine (listen: {})",
        sim_conv_ref - sim_base_ref,
        out_dir.join("converted.wav").display()
    );
}

/// Build the exact `GenerationRequest` the worker's native clone path
/// ([`crate::audio_jobs::run_native_voice_clone_synthesis`]) sends: the script as the prompt and the
/// reference clip as the SOLE voice conditioning (`Conditioning::ReferenceAudio`). Kept as a tiny
/// shared builder so this smoke proves the request the worker actually issues, not a bespoke one.
fn native_clone_request(
    script: &str,
    reference: gen_core::AudioTrack,
    seed: u64,
) -> GenerationRequest {
    GenerationRequest {
        prompt: script.to_owned(),
        seed: Some(seed),
        conditioning: vec![gen_core::Conditioning::ReferenceAudio {
            audio: reference,
            strength: None,
        }],
        cancel: CancelFlag::new(),
        ..Default::default()
    }
}

/// Local real-weight smoke for the **native cloned-voice TTS** path (sc-13412 / epic 12833 E1).
/// `#[ignore]`d — run by hand on an Apple-Silicon Mac. It drives the exact runtime seam the worker's
/// native Voice Clone path uses (`catalog().audio().load("chatterbox_tts")` +
/// `Generator::generate` with a single `Conditioning::ReferenceAudio`), minus the API/job plumbing,
/// and proves the DoD: a real cloned WAV whose Chatterbox-VE speaker embedding is materially closer to
/// the REFERENCE voice than to a different CONTROL voice.
///
///   1. **Reference + control clips** — two real Kokoro clips in distinctly different voices
///      (`af_heart` female reference, `am_michael` male control) stand in for a user reference sample
///      and an unrelated voice. The clone is rendered from the reference only.
///   2. **Native clone** — `chatterbox_tts` renders the script in the reference voice in ONE call
///      (T3 speech-token LM → S3Gen token→waveform + HiFTNet + PerTh watermark). 24 kHz mono.
///   3. **Chatterbox-VE evidence** — embed the clone, the reference, and the control; the clone's
///      speaker identity must be closer to the reference than to the control:
///      cosine(clone, ref) > cosine(clone, control). That is the objective, ear-free proof that the
///      clone tracks the reference voice — a true audible match still needs a listen, so all three
///      WAVs are written out.
///
/// Setup — one snapshot dir holding the Chatterbox files the generator + embedder load
/// (`t3_cfg.safetensors` + `s3gen.safetensors` + `tokenizer.json` + `ve.safetensors`), plus a Kokoro
/// snapshot for the reference/control clips; the PerTh watermarker resolves off the HF hub (online) or
/// from `PERTH_SNAPSHOT`:
/// ```text
/// cargo test -p sceneworks-worker --release native_voiceclone_chatterbox_smoke -- --ignored --nocapture
/// # overrides: VOICECLONE_KOKORO_DIR / VOICECLONE_CHATTERBOX_DIR / VOICECLONE_OUT_DIR
/// #            (default /tmp/voiceclone_smoke), PERTH_SNAPSHOT (offline watermarker override)
/// ```
#[test]
#[ignore = "real-weight audio smoke: run by hand on an Apple-Silicon Mac"]
fn native_voiceclone_chatterbox_smoke() {
    let kokoro_dir = resolve_dir("VOICECLONE_KOKORO_DIR", "hexgrad/Kokoro-82M", None);
    // The chatterbox_tts generator loads its T3 + S3Gen + tokenizer from a snapshot DIR; the
    // Chatterbox-VE embedder loads the single `ve.safetensors` from that same dir.
    let chatterbox_dir = resolve_dir("VOICECLONE_CHATTERBOX_DIR", "ResembleAI/chatterbox", None);
    let out_dir = PathBuf::from(env_or("VOICECLONE_OUT_DIR", "/tmp/voiceclone_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let seed: u64 = 20260720;

    // ── Reference + control clips (distinct voices) ──────────────────────────────────────────────
    let reference = kokoro_clip(
        &kokoro_dir,
        "The quick brown fox jumps over the lazy dog near the river bank at first light.",
        REFERENCE_VOICE_FEMALE,
    );
    assert!(peak(&reference.samples) > 0.0, "reference clip is silent");
    dump_wav(&reference, &out_dir.join("native_reference.wav"));
    let control = kokoro_clip(
        &kokoro_dir,
        "The quick brown fox jumps over the lazy dog near the river bank at first light.",
        CONTROL_VOICE_MALE,
    );
    assert!(peak(&control.samples) > 0.0, "control clip is silent");
    dump_wav(&control, &out_dir.join("native_control.wav"));

    // ── The single native clone call ─────────────────────────────────────────────────────────────
    let generator = runtime_macos::catalog()
        .expect("runtime catalog")
        .audio()
        .expect("audio lane")
        .load(
            "chatterbox_tts",
            &LoadSpec::new(WeightsSource::Dir(chatterbox_dir.clone())),
        )
        .expect("load chatterbox_tts generator");
    let mut on_progress = |_p: Progress| {};
    let clone = match generator
        .generate(
            &native_clone_request(NATIVE_CLONE_SCRIPT, reference.clone(), seed),
            &mut on_progress,
        )
        .expect("native clone generate")
    {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected audio, got {other:?}"),
    };
    assert!(!clone.samples.is_empty(), "clone is empty");
    let clone_peak = peak(&clone.samples);
    assert!(
        clone_peak > 1e-2,
        "clone is (near-)silent: peak {clone_peak:.6}"
    );
    assert_eq!(clone.sample_rate, 24_000, "clone must be 24 kHz");
    assert_eq!(clone.channels, 1, "clone must be mono");
    dump_wav(&clone, &out_dir.join("native_clone.wav"));

    // ── Chatterbox-VE evidence: does the clone track the reference? ──────────────────────────────
    let embedder = runtime_macos::catalog()
        .expect("runtime catalog")
        .audio()
        .expect("audio lane")
        .load_voice_embedder(
            "chatterbox_ve",
            &LoadSpec::new(WeightsSource::File(chatterbox_dir.join("ve.safetensors"))),
        )
        .expect("load chatterbox_ve embedder");
    let emb_clone = embedder.embed(&clone).expect("embed clone");
    let emb_ref = embedder.embed(&reference).expect("embed reference");
    let emb_control = embedder.embed(&control).expect("embed control");

    let sim_clone_ref = cosine(&emb_clone, &emb_ref);
    let sim_clone_control = cosine(&emb_clone, &emb_control);

    eprintln!("[native-clone-smoke] out dir: {}", out_dir.display());
    eprintln!(
        "[native-clone-smoke] clone: {} samples @ {} Hz, peak {clone_peak:.4}",
        clone.samples.len(),
        clone.sample_rate
    );
    eprintln!(
        "[native-clone-smoke] Chatterbox-VE cosine — clone↔ref {sim_clone_ref:.4} | \
         clone↔control {sim_clone_control:.4}  (margin {:.4})",
        sim_clone_ref - sim_clone_control
    );

    // The DoD evidence: the clone's speaker identity is closer to the reference than to a DIFFERENT
    // control voice — the clone tracks the reference the request supplied.
    assert!(
        sim_clone_ref > sim_clone_control,
        "clone did not track the reference: clone↔ref {sim_clone_ref:.4} must exceed \
         clone↔control {sim_clone_control:.4}"
    );
    eprintln!(
        "[native-clone-smoke] PASS — clone tracks the reference by {:.4} cosine (listen: {})",
        sim_clone_ref - sim_clone_control,
        out_dir.join("native_clone.wav").display()
    );
}

/// sc-13517 END-TO-END DoD: prove the "register a voice" feature drives a real cloned voiceover.
///
///   1. **Register (embed + store)** — embed a reference clip through the PUBLIC
///      `voice_register::embed_reference_clip` (the exact seam the rust-api calls on register), then
///      persist the saved voice through `SavedVoiceStore` — a real 256-d Chatterbox-VE embedding lands
///      in a real project.db.
///   2. **Near-duplicate consumer** — re-register the SAME clip: the store flags it as a near-duplicate
///      (cosine ≥ threshold). Register a spectrally-DISTINCT clip: NOT flagged. This is the embedding's
///      genuine consumer.
///   3. **Render through the saved voice** — load `chatterbox_tts` and render the script with the saved
///      voice's reference clip as `Conditioning::ReferenceAudio` (the referenceAudioAssetId → clone path
///      the Studio drives when a saved voice is picked). Writes a real, audible cloned WAV.
///
/// Writes `saved_voice_reference.wav`, `distinct_reference.wav`, `saved_voice_cloned_voiceover.wav`, and
/// `summary.json` to `$SAVED_VOICE_DOD_OUT` (default `/tmp/saved_voice_dod`). `#[ignore]`d real-weight
/// smoke — run by hand on an Apple-Silicon Mac with the Chatterbox Clone-TTS model installed.
#[test]
#[ignore = "real-weight end-to-end DoD: needs the cached ResembleAI/chatterbox model"]
fn saved_voice_register_render_dod() {
    use sceneworks_core::voice_store::{
        SavedVoiceCreateInput, SavedVoiceStore, DEFAULT_VOICE_DEDUP_THRESHOLD,
    };

    // The register embed path resolves ve.safetensors from the HF cache; point it at the OS hub cache.
    if std::env::var_os("HF_HOME").is_none() {
        if let Some(home) = sceneworks_core::hf_home::os_huggingface_home() {
            std::env::set_var("HF_HOME", home);
        }
    }
    let data_dir = std::env::temp_dir();
    let chatterbox_dir = resolve_dir("VOICECLONE_CHATTERBOX_DIR", "ResembleAI/chatterbox", None);

    let out_dir = PathBuf::from(env_or("SAVED_VOICE_DOD_OUT", "/tmp/saved_voice_dod"));
    std::fs::create_dir_all(&out_dir).expect("out dir");
    let project_dir = out_dir.join("project");
    std::fs::create_dir_all(&project_dir).expect("project dir");

    // The library audio assets a saved voice points to.
    let reference = synthetic_reference();
    let reference_wav = out_dir.join("saved_voice_reference.wav");
    dump_wav(&reference, &reference_wav);
    let distinct = distinct_reference();
    let distinct_wav = out_dir.join("distinct_reference.wav");
    dump_wav(&distinct, &distinct_wav);

    // ── (a) REGISTER: real embed via the API's embed path + persist through the store ──
    let emb_reference =
        crate::voice_register::embed_reference_clip(&data_dir, &reference_wav).expect("embed ref");
    assert_eq!(
        emb_reference.len(),
        256,
        "Chatterbox-VE embedding is 256-dim"
    );
    let store = SavedVoiceStore::new(&project_dir);
    let (narrator, dup_first) = store
        .create_saved_voice(
            "project_dod",
            SavedVoiceCreateInput {
                name: "Narrator".to_owned(),
                reference_audio_asset_id: "asset_reference".to_owned(),
                embedding: emb_reference.clone(),
            },
            DEFAULT_VOICE_DEDUP_THRESHOLD,
        )
        .expect("register Narrator");
    assert!(dup_first.is_none(), "first registration can't duplicate");
    let narrator_id = narrator["id"].as_str().expect("voice id").to_owned();

    // ── (b) DEDUP consumer: the SAME clip warns; a DISTINCT clip does not ──
    let emb_reference_again =
        crate::voice_register::embed_reference_clip(&data_dir, &reference_wav)
            .expect("re-embed ref");
    let (_dup_voice, dup_hit) = store
        .create_saved_voice(
            "project_dod",
            SavedVoiceCreateInput {
                name: "Narrator (again)".to_owned(),
                reference_audio_asset_id: "asset_reference_dup".to_owned(),
                embedding: emb_reference_again,
            },
            DEFAULT_VOICE_DEDUP_THRESHOLD,
        )
        .expect("register duplicate");
    let dup_hit = dup_hit.expect("re-registering the same clip must flag a near-duplicate");
    assert_eq!(dup_hit.name, "Narrator");

    let emb_distinct = crate::voice_register::embed_reference_clip(&data_dir, &distinct_wav)
        .expect("embed distinct");
    let (_distinct_voice, distinct_hit) = store
        .create_saved_voice(
            "project_dod",
            SavedVoiceCreateInput {
                name: "Villain".to_owned(),
                reference_audio_asset_id: "asset_distinct".to_owned(),
                embedding: emb_distinct.clone(),
            },
            DEFAULT_VOICE_DEDUP_THRESHOLD,
        )
        .expect("register distinct");
    assert!(
        distinct_hit.is_none(),
        "a distinct speaker must NOT be flagged"
    );
    assert_eq!(
        store.list_saved_voices("project_dod").expect("list").len(),
        3
    );

    // ── (c) RENDER: the saved voice's reference clip → chatterbox_tts clone (the Studio's path) ──
    let seed = 12_345u64;
    let generator = runtime_macos::catalog()
        .expect("runtime catalog")
        .audio()
        .expect("audio lane")
        .load(
            "chatterbox_tts",
            &LoadSpec::new(WeightsSource::Dir(chatterbox_dir.clone())),
        )
        .expect("load chatterbox_tts generator");
    let mut on_progress = |_p: Progress| {};
    let clone = match generator
        .generate(
            &native_clone_request(NATIVE_CLONE_SCRIPT, reference.clone(), seed),
            &mut on_progress,
        )
        .expect("clone generate")
    {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected audio, got {other:?}"),
    };
    let clone_peak = peak(&clone.samples);
    assert!(clone_peak > 1e-2, "cloned voiceover is (near-)silent");
    assert_eq!(clone.sample_rate, 24_000, "clone must be 24 kHz");
    let clone_wav = out_dir.join("saved_voice_cloned_voiceover.wav");
    dump_wav(&clone, &clone_wav);

    // Objective evidence the rendered clone tracks the saved voice's reference (not the distinct one).
    let emb_clone =
        crate::voice_register::embed_reference_clip(&data_dir, &clone_wav).expect("embed clone");
    let sim_clone_reference = cosine(&emb_clone, &emb_reference);
    let sim_clone_distinct = cosine(&emb_clone, &emb_distinct);

    let summary = serde_json::json!({
        "story": "sc-13517",
        "register": {
            "narratorId": narrator_id,
            "embeddingDim": emb_reference.len(),
            "firstRegistrationDuplicate": null,
        },
        "dedupConsumer": {
            "threshold": DEFAULT_VOICE_DEDUP_THRESHOLD,
            "sameClipReRegisterFlagged": { "id": dup_hit.id, "name": dup_hit.name, "similarity": dup_hit.similarity },
            "distinctClipFlagged": false,
        },
        "clonedVoiceover": {
            "wav": clone_wav.display().to_string(),
            "samples": clone.samples.len(),
            "sampleRate": clone.sample_rate,
            "peak": clone_peak,
            "cosineCloneReference": sim_clone_reference,
            "cosineCloneDistinct": sim_clone_distinct,
        },
        "artifacts": {
            "referenceWav": reference_wav.display().to_string(),
            "distinctWav": distinct_wav.display().to_string(),
            "clonedWav": clone_wav.display().to_string(),
        },
    });
    let summary_path = out_dir.join("summary.json");
    std::fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .expect("write summary");

    eprintln!("[saved-voice-dod] out dir: {}", out_dir.display());
    eprintln!(
        "[saved-voice-dod] dedup: same-clip re-register flagged '{}' @ {:.4}; distinct NOT flagged",
        dup_hit.name, dup_hit.similarity
    );
    eprintln!(
        "[saved-voice-dod] cloned voiceover: {} samples @ {} Hz peak {clone_peak:.4} → {}",
        clone.samples.len(),
        clone.sample_rate,
        clone_wav.display()
    );
    eprintln!(
        "[saved-voice-dod] clone tracks reference: cosine clone↔ref {sim_clone_reference:.4} vs clone↔distinct {sim_clone_distinct:.4}"
    );
    assert!(
        sim_clone_reference > sim_clone_distinct,
        "the cloned voiceover must track the saved voice's reference more than a distinct clip"
    );
}

// ── sc-13541: offline install-parity DoD ─────────────────────────────────────────────────────────
// The chatterbox_tts provider resolves two companion weights at generate() time via pinned-SHA
// `hf_get_pinned` fetches (candle_audio_chatterbox_ve + candle-audio-chatterbox/src/perth.rs at the
// SceneWorks-pinned inference commit). The manifest co-requisites (sc-13541) pin these EXACT SHAs.
const CHATTERBOX_REPO: &str = "ResembleAI/chatterbox";
const VE_REVISION: &str = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18";
const PERTH_REPO: &str = "SceneWorks/perth-implicit";
const PERTH_REVISION: &str = "80b60f9caead09b8d3b512bda0b24038f28c08ec";

/// Materialize one HF snapshot (`repo` @ `revision`, filtered to `files`) into the hub cache via the
/// worker's REAL model-download executor — `HuggingFaceSnapshot::resolve` + `download_snapshot_into_cache`,
/// the exact seam `run_model_download_job` drives for a Models-screen install. Returns the resolved
/// commit SHA. Proving the pinned snapshots land here proves the install materializes them (sc-13541).
async fn install_snapshot(
    settings: &crate::Settings,
    repo: &str,
    revision: &str,
    files: &[&str],
) -> String {
    use crate::downloads::{
        download_snapshot_into_cache, DownloadContext, DownloadProgress, HuggingFaceSnapshot,
    };
    let client = crate::downloads::streaming_download_client();
    let api = crate::ApiClient::new(settings);
    let repo_dir = sceneworks_core::hf_home::huggingface_repo_cache_path(&settings.data_dir, repo)
        .expect("resolve hub cache path");
    let file_patterns: Vec<String> = files.iter().map(|f| (*f).to_owned()).collect();
    let snapshot = HuggingFaceSnapshot::resolve(&client, settings, repo, revision, &file_patterns)
        .await
        .expect("resolve HF snapshot listing");
    assert!(
        !snapshot.files.is_empty(),
        "{repo}@{revision} resolved zero files for {file_patterns:?}"
    );
    let context = DownloadContext {
        api: &api,
        client: &client,
        settings,
        job_id: "sc-13541-offline-parity",
        cancel_message: "canceled",
        fresh_download: false,
    };
    // A report interval longer than any real transfer so the in-loop progress POST never fires: this
    // harness has no job API (the executor's heartbeat/progress are best-effort and only reached via
    // the interval tick), and cancel polls already swallow transport errors. The download itself only
    // talks to the HF hub, which is exactly what we are exercising.
    let mut progress = DownloadProgress::new(
        repo,
        0,
        snapshot.total_bytes(),
        std::time::Duration::from_secs(86_400),
    );
    download_snapshot_into_cache(&context, &repo_dir, revision, &snapshot, &mut progress)
        .await
        .unwrap_or_else(|e| panic!("materialize {repo}@{revision}: {e}"))
}

/// A short, non-trivial synthetic speech-like reference clip (a vibrato'd fundamental plus two
/// harmonics and light noise), 24 kHz mono — enough content for the provider to derive a speaker
/// embedding + prompt tokens without pulling in a second TTS model. Clone quality is irrelevant to
/// this DoD; the point is that generate() completes with no hub fetch.
fn synthetic_reference() -> gen_core::AudioTrack {
    let sample_rate = 24_000u32;
    let secs = 4.0f32;
    let n = (sample_rate as f32 * secs) as usize;
    let mut samples = Vec::with_capacity(n);
    let mut seed = 0x2b41_53c7u32;
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let vibrato = 1.0 + 0.02 * (2.0 * std::f32::consts::PI * 5.0 * t).sin();
        let f0 = 165.0 * vibrato;
        let mut s = 0.6 * (2.0 * std::f32::consts::PI * f0 * t).sin();
        s += 0.25 * (2.0 * std::f32::consts::PI * 2.0 * f0 * t).sin();
        s += 0.15 * (2.0 * std::f32::consts::PI * 3.0 * f0 * t).sin();
        // Cheap LCG noise for a little broadband energy.
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
        s += 0.05 * noise;
        // Gentle amplitude envelope so it is not a steady tone.
        let env = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * 0.5 * t).cos();
        samples.push(0.8 * env * s);
    }
    gen_core::AudioTrack {
        samples,
        sample_rate,
        channels: 1,
        stems: Vec::new(),
    }
}

/// A spectrally-distinct synthetic clip (higher fundamental, different harmonic mix + envelope) so its
/// Chatterbox-VE embedding differs from [`synthetic_reference`]'s — used by the sc-13517 embed-path
/// smoke to prove the speaker vector discriminates.
fn distinct_reference() -> gen_core::AudioTrack {
    let sample_rate = 24_000u32;
    let secs = 4.0f32;
    let n = (sample_rate as f32 * secs) as usize;
    let mut samples = Vec::with_capacity(n);
    let mut seed = 0x7f4a_1122u32;
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let vibrato = 1.0 + 0.03 * (2.0 * std::f32::consts::PI * 7.0 * t).sin();
        let f0 = 320.0 * vibrato;
        let mut s = 0.5 * (2.0 * std::f32::consts::PI * f0 * t).sin();
        s += 0.35 * (2.0 * std::f32::consts::PI * 1.5 * f0 * t).sin();
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
        s += 0.12 * noise;
        let env = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * 0.9 * t).cos();
        samples.push(0.8 * env * s);
    }
    gen_core::AudioTrack {
        samples,
        sample_rate,
        channels: 1,
        stems: Vec::new(),
    }
}

/// sc-13517 embed-path smoke: the PUBLIC `voice_register::embed_reference_clip` (what the rust-api
/// calls to register a voice) resolves the cached Chatterbox-VE weights, decodes a WAV, and returns a
/// 256-d speaker vector. Deterministic: the SAME clip self-embeds to cosine ~1.0, while a
/// spectrally-distinct clip scores lower — objective evidence the embedding (the registry's dedup
/// identity) discriminates. Run by hand on a Mac with the Chatterbox Clone-TTS model installed.
#[test]
#[ignore = "real-weight audio smoke: needs the cached ResembleAI/chatterbox ve.safetensors"]
fn voice_register_embed_path_smoke() {
    // Point HF cache resolution at the OS hub cache so embed_reference_clip's data_dir-based
    // resolution finds the installed snapshot regardless of the (unused) data dir passed below.
    if std::env::var_os("HF_HOME").is_none() {
        if let Some(home) = sceneworks_core::hf_home::os_huggingface_home() {
            std::env::set_var("HF_HOME", home);
        }
    }
    let data_dir = std::env::temp_dir();

    let tmp = std::env::temp_dir().join(format!("sw-voice-register-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("temp dir");
    let ref_a = tmp.join("ref_a.wav");
    let ref_b = tmp.join("ref_b.wav");
    dump_wav(&synthetic_reference(), &ref_a);
    dump_wav(&distinct_reference(), &ref_b);

    let emb_a1 =
        crate::voice_register::embed_reference_clip(&data_dir, &ref_a).expect("embed clip a");
    let emb_a2 =
        crate::voice_register::embed_reference_clip(&data_dir, &ref_a).expect("re-embed clip a");
    let emb_b =
        crate::voice_register::embed_reference_clip(&data_dir, &ref_b).expect("embed clip b");

    assert_eq!(emb_a1.len(), 256, "Chatterbox-VE embedding is 256-dim");
    let self_sim = cosine(&emb_a1, &emb_a2);
    let cross_sim = cosine(&emb_a1, &emb_b);
    eprintln!("voice_register embed smoke: self={self_sim:.4} cross={cross_sim:.4}");
    assert!(
        self_sim > 0.999,
        "the same clip must self-embed to ~1.0 (got {self_sim})"
    );
    assert!(
        cross_sim < self_sim,
        "a spectrally-distinct clip must score below the self-similarity (self {self_sim}, cross {cross_sim})"
    );
}

/// Sorted names directly under `dir` (empty when `dir` is absent) — a cheap fingerprint for asserting
/// that a directory was neither added to nor removed from.
fn dir_entry_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Fingerprint of the machine's REAL HF cache's chatterbox + perth entries (`snapshots/` and `refs/`
/// listings). The offline DoD smoke captures this before/after to prove it never reads into or writes
/// the real `~/.cache/huggingface` — everything must stay inside the isolated temp cache.
fn real_cache_fingerprint(chatterbox_dir: &Path, perth_dir: &Path) -> Vec<Vec<String>> {
    vec![
        dir_entry_names(&chatterbox_dir.join("snapshots")),
        dir_entry_names(&chatterbox_dir.join("refs")),
        dir_entry_names(&perth_dir.join("snapshots")),
        dir_entry_names(&perth_dir.join("refs")),
    ]
}

/// sc-13541 (epic 13400 / E1 follow-up) — **offline install-parity DoD**. `#[ignore]`d; run by hand
/// on an Apple-Silicon Mac (downloads ~3.2 GB into an ISOLATED temp HF cache):
/// ```text
/// cargo test -p sceneworks-worker chatterbox_offline_install_parity_smoke -- --ignored --nocapture
/// ```
/// Proves the acceptance criterion end to end AND that it is DISCRIMINATING — it would fail if the
/// co-requisite install were skipped, which the first cut did NOT: that version resolved off the
/// machine's real `~/.cache/huggingface` (which already held ve/perth), so it passed even though the
/// install under test was never actually exercised.
///
/// Two facts — verified against the pinned inference rev and hf-hub 0.4.3 — drive the design:
///   * The runtime resolver is `candle_audio::hub::hf_get_pinned` → hf-hub `Api::new()` →
///     `Cache::default()`, which is HARD-CODED to `dirs::home_dir()/.cache/huggingface/hub` and reads
///     NEITHER `HF_HOME` NOR `HF_ENDPOINT` (only `ApiBuilder::from_env()` reads those). On unix
///     `dirs::home_dir()` is `$HOME`, so the ONLY lever that redirects the resolver's cache is `$HOME`.
///     This test points `$HOME` at a fresh temp root ⇒ the resolver reads `<temp>/.cache/huggingface/
///     hub` and the real `~/.cache/huggingface` is never touched (asserted). `HF_HOME` is set to
///     `<temp>/.cache/huggingface` — exactly what the desktop injects (sc-1904) — so the SceneWorks
///     downloader (`HF_HOME/hub`) writes the SAME hub the resolver reads: the DEFAULT shipped alignment.
///   * hf-hub 0.4.3 has no `HF_HUB_OFFLINE` and `Api::new()` ignores `HF_ENDPOINT`, so the old
///     black-hole `HF_ENDPOINT` was INERT. But both hf-hub agents are built with ureq
///     `try_proxy_from_env(true)`, so a black-hole `ALL_PROXY`/`HTTPS_PROXY` DOES route (and fail)
///     every resolver `download()`, while a cache HIT short-circuits before the agent is constructed.
///     That makes "offline" deterministic and independent of the box's real connectivity.
///
///   1. **Isolate** — `$HOME` + `HF_HOME` at a fresh temp root; clear every other HF cache + proxy
///      var. Assert the isolated hub starts empty and differs from the real `~/.cache/huggingface`.
///   2. **Install** — run the REAL download executor (online) for the primary snapshot (`main`) PLUS
///      the two pinned companion co-requisites: `ve.safetensors` @ 5bb1f6ee… and
///      `perth_implicit.safetensors` @ 80b60f9c…. Assert each pinned `snapshots/<sha>/…` + `refs/<sha>`
///      materialized where `hf_get_pinned` reads (the hf-hub layout `download_snapshot_into_cache`
///      writes: `refs/<rev>` → commit, `snapshots/<commit>/<file>`).
///   3. **Go offline, HARD** — black-hole `ALL_PROXY`/`HTTPS_PROXY`/`HTTP_PROXY` (any resolver fetch
///      now fails through a dead proxy). Load `chatterbox_tts` from the installed snapshot dir and
///      `generate()` one clone from a `Conditioning::ReferenceAudio` → MUST SUCCEED: ve
///      (embed_reference) and perth (always-on watermark) resolve cache-first with zero network. A
///      success with the dead proxy set IS the "no runtime hub fetch" proof.
///   4. **Discriminate** — with the dead proxy still set, delete ONLY the PerTh snapshot, RELOAD the
///      generator (its lazy ve/perth caches reset) and re-render → MUST HARD-FAIL (perth cache-miss →
///      dead proxy). Then delete the ve snapshot + `refs/<sha>` and re-render → MUST HARD-FAIL (ve
///      cache-miss). Deleting from the ISOLATED cache breaking the render is the proof the resolver
///      reads THAT cache, not the real one. The 3.2 GB primary snapshot is reused — nothing re-downloads.
#[test]
#[ignore = "real-weight offline-install-parity DoD: downloads ~3.2 GB into an ISOLATED temp HF cache; run by hand on an Apple-Silicon Mac"]
fn chatterbox_offline_install_parity_smoke() {
    // ── Step 1: isolate the cache the RUNTIME RESOLVER actually reads ─────────────────────────────
    // The resolver is hf_get_pinned -> hf-hub Api::new() -> Cache::default() = dirs::home_dir()/.cache/
    // huggingface/hub, which reads NEITHER HF_HOME NOR HF_ENDPOINT (only ApiBuilder::from_env() does)
    // and on unix takes home_dir() from $HOME. So $HOME is the only lever that moves the resolver's
    // cache. Point it — and, for the downloader, HF_HOME=$HOME/.cache/huggingface, the desktop's
    // sc-1904 default — at a fresh temp root so BOTH sides share one empty, isolated hub.
    let home_root = tempfile::tempdir().expect("temp HOME root");
    let data_dir = tempfile::tempdir().expect("temp data dir");

    // Capture the machine's REAL hub BEFORE the override so we can prove it is never touched.
    let real_home = std::env::var("HOME").expect("HOME must be set (we isolate against it)");
    let real_hub = PathBuf::from(&real_home)
        .join(".cache")
        .join("huggingface")
        .join("hub");
    let real_chatterbox = real_hub.join("models--ResembleAI--chatterbox");
    let real_perth = real_hub.join("models--SceneWorks--perth-implicit");
    let real_before = real_cache_fingerprint(&real_chatterbox, &real_perth);
    assert_ne!(
        PathBuf::from(&real_home),
        home_root.path(),
        "the temp HOME must differ from the real home"
    );

    let hf_home = home_root.path().join(".cache").join("huggingface");
    let hub = hf_home.join("hub");
    // Pin the env for the whole test; EnvVars restores it on drop (Drop runs even on a panicking
    // assert, so a leaked $HOME can't poison sibling tests). Every HF cache var except HF_HOME is
    // cleared so neither the downloader nor the resolver can diverge to another location; proxy vars
    // are cleared now (install must reach the real hub) and set to a black hole in step 3.
    let _env = crate::test_env::EnvVars::set(&[
        ("HOME", home_root.path().to_str().expect("utf-8 HOME")),
        ("HF_HOME", hf_home.to_str().expect("utf-8 HF_HOME")),
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HUB_OFFLINE", ""),
        ("HF_ENDPOINT", ""),
        ("PERTH_SNAPSHOT", ""),
        ("ALL_PROXY", ""),
        ("all_proxy", ""),
        ("HTTPS_PROXY", ""),
        ("https_proxy", ""),
        ("HTTP_PROXY", ""),
        ("http_proxy", ""),
        ("NO_PROXY", ""),
        ("no_proxy", ""),
    ]);
    assert!(
        !hub.join("models--ResembleAI--chatterbox").exists()
            && !hub.join("models--SceneWorks--perth-implicit").exists(),
        "the DoD requires an EMPTY isolated HF cache to start"
    );

    let settings = crate::Settings {
        api_url: "http://127.0.0.1:1".to_owned(),
        access_token: None,
        data_dir: data_dir.path().to_path_buf(),
        config_dir: data_dir.path().join("config"),
        worker_id: "sc-13541".to_owned(),
        gpu_id: "gpu-0".to_owned(),
        is_child_worker: false,
        poll_seconds: 1,
        heartbeat_seconds: 5,
        shutdown_timeout_seconds: 1,
        huggingface_base_url: crate::DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
        huggingface_token: std::env::var("HF_TOKEN").ok(),
        credentials: Vec::new(),
        max_lora_url_bytes: crate::DEFAULT_MAX_LORA_URL_BYTES,
        max_model_url_bytes: crate::DEFAULT_MAX_MODEL_URL_BYTES,
        allow_private_lora_urls: false,
        utility_workers: 1,
        backend_mlx_enabled: true,
        backend_candle_enabled: false,
        gpu_memory_limit_bytes: 0,
        external_model_roots: Vec::new(),
    };

    // ── Step 2: install (the exact executor a Models-screen install runs), into the ISOLATED cache ─
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let main_commit = rt.block_on(install_snapshot(
        &settings,
        CHATTERBOX_REPO,
        "main",
        &["t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"],
    ));
    rt.block_on(install_snapshot(
        &settings,
        CHATTERBOX_REPO,
        VE_REVISION,
        &["ve.safetensors"],
    ));
    rt.block_on(install_snapshot(
        &settings,
        PERTH_REPO,
        PERTH_REVISION,
        &["perth_implicit.safetensors"],
    ));

    // ── Assert the pinned companion snapshots landed where hf_get_pinned reads (the hf-hub layout
    //    download_snapshot_into_cache writes: refs/<rev> -> commit + snapshots/<commit>/<file>) ────
    let chatterbox_dir = hub.join("models--ResembleAI--chatterbox");
    let perth_dir = hub.join("models--SceneWorks--perth-implicit");
    let ve_snapshot = chatterbox_dir
        .join("snapshots")
        .join(VE_REVISION)
        .join("ve.safetensors");
    let ve_ref = chatterbox_dir.join("refs").join(VE_REVISION);
    let perth_snapshot = perth_dir
        .join("snapshots")
        .join(PERTH_REVISION)
        .join("perth_implicit.safetensors");
    let perth_ref = perth_dir.join("refs").join(PERTH_REVISION);
    assert!(
        ve_snapshot.exists(),
        "ve.safetensors must materialize at the pinned snapshot: {}",
        ve_snapshot.display()
    );
    assert!(ve_ref.exists(), "ve refs/<sha> pointer must exist");
    assert!(
        perth_snapshot.exists(),
        "perth_implicit.safetensors must materialize at the pinned snapshot: {}",
        perth_snapshot.display()
    );
    assert!(perth_ref.exists(), "perth refs/<sha> pointer must exist");
    eprintln!(
        "[offline-parity] installed pinned snapshots into the ISOLATED hub {}:\n  {}\n  {}",
        hub.display(),
        ve_snapshot.display(),
        perth_snapshot.display()
    );

    // ── Step 3: go offline, HARD — a black-hole proxy fails EVERY resolver download() (both hf-hub
    //    agents are built with ureq try_proxy_from_env), while a cache HIT never constructs the agent.
    //    Set AFTER install (which needed the real hub). Api::new() ignores HF_ENDPOINT, so the proxy —
    //    not an endpoint — is the lever that actually blocks the resolver. ────────────────────────
    std::env::set_var("ALL_PROXY", "http://127.0.0.1:1");
    std::env::set_var("HTTPS_PROXY", "http://127.0.0.1:1");
    std::env::set_var("HTTP_PROXY", "http://127.0.0.1:1");

    let main_snapshot_dir = chatterbox_dir.join("snapshots").join(&main_commit);
    assert!(
        main_snapshot_dir.join("s3gen.safetensors").exists(),
        "primary chatterbox snapshot dir must hold the generator weights"
    );

    // One offline render from the installed primary snapshot, RELOADING the generator each call: the
    // provider caches its ve/perth embedders lazily in a Mutex on first use, so a second generate() on
    // the SAME generator would reuse them and never re-touch the cache — the discrimination below MUST
    // reload to force re-resolution. The primary weights load from the local DIR (no hub), so a reload
    // never re-downloads the 3.2 GB primary.
    let render_offline = || -> Result<gen_core::AudioTrack, String> {
        let request = GenerationRequest {
            prompt: "This cloned voice was rendered fully offline from a fresh install.".to_owned(),
            seed: Some(13541),
            conditioning: vec![Conditioning::ReferenceAudio {
                audio: synthetic_reference(),
                strength: None,
            }],
            cancel: CancelFlag::new(),
            ..Default::default()
        };
        let mut on_progress = |_p: Progress| {};
        let output = runtime_macos::catalog()
            .map_err(|e| format!("runtime catalog: {e}"))?
            .audio()
            .ok_or_else(|| "audio lane missing from the runtime catalog".to_owned())?
            .load(
                "chatterbox_tts",
                &LoadSpec::new(WeightsSource::Dir(main_snapshot_dir.clone())),
            )
            .map_err(|e| format!("load chatterbox_tts: {e}"))?
            .generate(&request, &mut on_progress)
            .map_err(|e| format!("generate: {e}"))?;
        match output {
            GenerationOutput::Audio(track) => Ok(track),
            other => Err(format!("expected audio, got {other:?}")),
        }
    };

    // ── Step 4a: the positive proof — ve + perth resolve cache-first, zero network ───────────────
    let clone = render_offline().unwrap_or_else(|e| {
        panic!(
            "offline native clone generate MUST succeed from the installed isolated cache \
             (ve + perth resolve cache-first with zero network; the dead proxy proves no fetch): {e}"
        )
    });
    assert!(!clone.samples.is_empty(), "offline clone is empty");
    assert!(
        peak(&clone.samples) > 1e-3,
        "offline clone is (near-)silent: peak {:.6}",
        peak(&clone.samples)
    );
    let out_dir = PathBuf::from(env_or("VOICECLONE_OUT_DIR", "/tmp/voiceclone_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    dump_wav(&clone, &out_dir.join("offline_parity_clone.wav"));
    eprintln!(
        "[offline-parity] step 4a PASS — offline render succeeded through a black-hole proxy: {} samples @ {} Hz (listen: {})",
        clone.samples.len(),
        clone.sample_rate,
        out_dir.join("offline_parity_clone.wav").display()
    );

    // ── Step 4b: discrimination — prove the INSTALL (not the real ~/.cache) is load-bearing ──────
    // #1: remove ONLY the PerTh snapshot (ve stays). A reload + render must reach the always-on
    // watermark, cache-miss PerTh, and hard-fail through the dead proxy.
    std::fs::remove_dir_all(&perth_dir).expect("remove the isolated PerTh model dir");
    let perth_err = render_offline().expect_err(
        "with the PerTh snapshot deleted from the ISOLATED cache the offline render MUST hard-fail — \
         a success would mean the resolver read some OTHER cache (e.g. the real ~/.cache): a false pass",
    );
    assert!(
        perth_err.contains("PerTh"),
        "the discrimination failure must be the missing PerTh weight resolving through the dead proxy, got: {perth_err}"
    );
    eprintln!(
        "[offline-parity] step 4b#1 PASS — PerTh removed ⇒ hard-fail as required: {perth_err}"
    );

    // #2: also remove the ve snapshot pointer + its refs/<sha>. A reload + render now cache-misses ve
    // at embed_reference (the first companion the clone path touches) and hard-fails through the proxy.
    std::fs::remove_file(&ve_snapshot).expect("remove the isolated ve.safetensors pointer");
    std::fs::remove_file(&ve_ref).expect("remove the isolated ve refs/<sha>");
    let ve_err = render_offline().expect_err(
        "with the ve snapshot deleted from the ISOLATED cache the offline render MUST hard-fail",
    );
    assert!(
        ve_err.contains("chatterbox_ve"),
        "the discrimination failure must be the missing ve embedder resolving through the dead proxy, got: {ve_err}"
    );
    eprintln!("[offline-parity] step 4b#2 PASS — ve removed ⇒ hard-fail as required: {ve_err}");

    // ── The machine's real ~/.cache/huggingface must be neither read into nor written by any of the
    //    above: install targeted HF_HOME=<temp>/hub and the resolver's Cache::default() targeted
    //    $HOME=<temp>. Its chatterbox/perth entries must match the pre-test fingerprint exactly. ───
    let real_after = real_cache_fingerprint(&real_chatterbox, &real_perth);
    assert_eq!(
        real_before, real_after,
        "the real ~/.cache/huggingface was modified — the test must be fully isolated under $HOME/HF_HOME=<temp>"
    );
    eprintln!(
        "[offline-parity] PASS — install→offline render is self-contained in the isolated cache, is \
         DISCRIMINATING (removing ve or perth hard-fails), and left the real {} untouched.",
        real_hub.display()
    );
}
