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

/// sc-13541 (epic 13400 / E1 follow-up) — **offline install-parity DoD**. `#[ignore]`d; run by hand
/// on an Apple-Silicon Mac (downloads ~3.2 GB into an ISOLATED temp HF cache):
/// ```text
/// cargo test -p sceneworks-worker chatterbox_offline_install_parity_smoke -- --ignored --nocapture
/// ```
/// Proves the acceptance criterion end to end: install `chatterbox_tts` on a machine with an EMPTY
/// HF cache, go offline, render a Voice Clone — it succeeds with NO runtime hub fetch.
///
///   1. **Empty cache** — point `HF_HOME` at a fresh temp dir and clear every other HF cache var so
///      both the SceneWorks downloader and the runtime's hf-hub resolver share one empty hub cache.
///   2. **Install** — run the REAL download executor for the three primary files (`main`) PLUS the two
///      pinned companion co-requisites: `ve.safetensors` @ 5bb1f6ee… and `perth_implicit.safetensors`
///      @ 80b60f9c…. Assert each pinned snapshot (`snapshots/<sha>/…`) + its `refs/<sha>` pointer is
///      materialized exactly where `hf_get_pinned` reads.
///   3. **Go offline, hard** — set `HF_HUB_OFFLINE=1` AND point `HF_ENDPOINT` at a black hole
///      (`127.0.0.1:1`). Cache-first `hf_get_pinned` returns the cached file without a fetch; if it
///      instead tried the hub (the pre-fix behavior when ve/perth were absent) it would hit the black
///      hole and hard-fail. So a successful render IS the "no hub fetch" proof.
///   4. **Render** — load `chatterbox_tts` from its snapshot dir and `generate()` one clone from a
///      `Conditioning::ReferenceAudio`. generate() resolves ve (embed_reference) + perth (always-on
///      watermark) internally; success ⇒ both resolved offline from the installed cache.
#[test]
#[ignore = "real-weight offline-install-parity DoD: downloads ~3.2 GB into a temp cache; run by hand on an Apple-Silicon Mac"]
fn chatterbox_offline_install_parity_smoke() {
    let cache_home = tempfile::tempdir().expect("temp HF home");
    let data_dir = tempfile::tempdir().expect("temp data dir");
    // One shared, EMPTY hub cache for both the downloader and the runtime resolver: HF_HOME/hub.
    std::env::set_var("HF_HOME", cache_home.path());
    std::env::remove_var("HF_HUB_CACHE");
    std::env::remove_var("HUGGINGFACE_HUB_CACHE");
    std::env::remove_var("HF_HUB_OFFLINE");
    std::env::remove_var("HF_ENDPOINT");
    std::env::remove_var("PERTH_SNAPSHOT");
    let hub = cache_home.path().join("hub");
    assert!(
        !hub.join("models--ResembleAI--chatterbox").exists()
            && !hub.join("models--SceneWorks--perth-implicit").exists(),
        "the DoD requires an EMPTY HF cache to start"
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

    // ── Step 2: install (the exact executor a Models-screen install runs) ────────────────────────
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

    // ── Assert the pinned companion snapshots landed where hf_get_pinned reads ───────────────────
    let ve_snapshot = hub
        .join("models--ResembleAI--chatterbox")
        .join("snapshots")
        .join(VE_REVISION)
        .join("ve.safetensors");
    let ve_ref = hub
        .join("models--ResembleAI--chatterbox")
        .join("refs")
        .join(VE_REVISION);
    let perth_snapshot = hub
        .join("models--SceneWorks--perth-implicit")
        .join("snapshots")
        .join(PERTH_REVISION)
        .join("perth_implicit.safetensors");
    let perth_ref = hub
        .join("models--SceneWorks--perth-implicit")
        .join("refs")
        .join(PERTH_REVISION);
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
        "[offline-parity] installed pinned snapshots:\n  {}\n  {}",
        ve_snapshot.display(),
        perth_snapshot.display()
    );

    // ── Step 3: go offline, HARD — any attempted hub fetch now fails loudly ──────────────────────
    std::env::set_var("HF_HUB_OFFLINE", "1");
    std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:1");

    // ── Step 4: the single native clone call (resolves ve + perth internally, offline) ───────────
    let main_snapshot_dir = hub
        .join("models--ResembleAI--chatterbox")
        .join("snapshots")
        .join(&main_commit);
    assert!(
        main_snapshot_dir.join("s3gen.safetensors").exists(),
        "primary chatterbox snapshot dir must hold the generator weights"
    );
    let generator = runtime_macos::catalog()
        .expect("runtime catalog")
        .audio()
        .expect("audio lane")
        .load(
            "chatterbox_tts",
            &LoadSpec::new(WeightsSource::Dir(main_snapshot_dir.clone())),
        )
        .expect("load chatterbox_tts generator (offline, from the installed snapshot)");
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
    let clone = match generator
        .generate(&request, &mut on_progress)
        .expect("offline native clone generate (ve + perth must resolve from the installed cache)")
    {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected audio, got {other:?}"),
    };
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
        "[offline-parity] PASS — offline render succeeded with a black-hole HF endpoint: {} samples @ {} Hz (listen: {})",
        clone.samples.len(),
        clone.sample_rate,
        out_dir.join("offline_parity_clone.wav").display()
    );
}
