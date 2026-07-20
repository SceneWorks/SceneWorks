//! rust-api jobs tests (split from tests.rs, sc-11217 F-030).
use super::support::*;

/// F-003 / sc-11159: a path-traversal `model` id is rejected at the POST boundary for BOTH
/// the image and video enqueue lanes (before any job is created), closing the remote
/// arbitrary-write primitive the worker filename builders would otherwise expose.
#[tokio::test]
async fn image_and_video_jobs_reject_path_unsafe_model_id() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    for (endpoint, mode) in [
        ("/api/v1/image/jobs", "text_to_image"),
        ("/api/v1/video/jobs", "text_to_video"),
    ] {
        for evil in ["../../../../etc/passwd", "..\\..\\evil", "/abs/pwn", "a/b"] {
            let (status, body) = request(
                app.clone(),
                "POST",
                endpoint,
                json!({
                    "projectId": "project-1",
                    "mode": mode,
                    "prompt": "a fox",
                    "model": evil,
                }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{endpoint} {evil:?}");
            assert!(
                body["detail"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("model must be a plain model id"),
                "{endpoint} {evil:?}: unexpected error {body}"
            );
        }
    }
}

/// sc-12305: the generic `POST /api/v1/jobs` enqueues `type` + payload verbatim — no
/// manifest resolution — so a generation job through that door carries no
/// `modelManifestEntry` and silently renders off-bucket (see the
/// `mochi_without_manifest_entry_*` test in `video_request.rs` for the exact geometry).
/// Every job type whose typed route injects an entry must be rejected here, pointed at
/// that route. Covers image as well as video: `image_request.rs` reads the entry the same way.
#[tokio::test]
async fn generic_jobs_route_rejects_generation_types_with_their_typed_route() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    for (job_type, typed_route) in [
        ("image_generate", "/api/v1/image/jobs"),
        ("image_edit", "/api/v1/image/jobs"),
        ("video_generate", "/api/v1/video/jobs"),
        ("video_extend", "/api/v1/video/jobs"),
        ("video_bridge", "/api/v1/video/jobs"),
        ("person_replace", "/api/v1/video/jobs"),
        // Audio Studio (sc-13404): the audio route injects the model's manifest entry too, so an
        // `audio_generate` job enqueued raw through the generic route must be rejected the same way.
        ("audio_generate", "/api/v1/audio/jobs"),
    ] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": job_type,
                "projectId": "project-1",
                "requestedGpu": "auto",
                "payload": { "model": "mochi_1", "prompt": "a fox", "width": 848, "height": 480 },
            }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{job_type} must be rejected"
        );
        let detail = body["detail"].as_str().unwrap_or_default();
        assert!(
            detail.contains(typed_route),
            "{job_type}: error must name {typed_route}, got {body}"
        );
    }
}

/// The other half of the guard: the job types the generic route legitimately serves keep
/// working. `image_upscale` / `image_detail` are the real web callers (batch ops), and
/// neither has a typed door — so the sc-12305 rejection must not touch them.
#[tokio::test]
async fn generic_jobs_route_still_serves_non_generation_types() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    for job_type in ["image_upscale", "image_detail"] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": job_type,
                "requestedGpu": "auto",
                "payload": { "sourceAssetId": "asset-1" },
            }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::CREATED,
            "{job_type} must still enqueue: {body}"
        );
    }
}

#[test]
fn serialize_job_lora_carries_network_type_to_payload() {
    // A trained LoKr adapter records networkType (epic 2193); the generation
    // payload must carry it so the worker can route LoKr off the MLX backend
    // without opening the file.
    let lora = json!({
        "id": "char",
        "family": "sdxl",
        "networkType": "lokr",
        "source": { "provider": "training" },
    });
    let payload = serialize_job_lora(&lora, &json!({}), "char");
    assert_eq!(
        payload.get("networkType").and_then(Value::as_str),
        Some("lokr")
    );

    // A plain LoRA without the field stays absent/null (treated as lora downstream).
    let plain = serialize_job_lora(&json!({ "id": "x", "family": "sdxl" }), &json!({}), "x");
    assert!(plain.get("networkType").map(Value::is_null).unwrap_or(true));
}

#[test]
fn person_readiness_reflects_live_worker_capabilities() {
    let workers = vec![
        readiness_worker(
            "gpu",
            WorkerStatus::Idle,
            vec![
                WorkerCapability::PersonDetect,
                WorkerCapability::PersonTrack,
                WorkerCapability::PersonReplace,
            ],
        ),
        readiness_worker(
            "cpu",
            WorkerStatus::Idle,
            vec![
                WorkerCapability::PersonDetectPreview,
                WorkerCapability::PersonTrackPreview,
            ],
        ),
        // Segment capability exists only on an offline worker -> not ready.
        readiness_worker(
            "dead",
            WorkerStatus::Offline,
            vec![WorkerCapability::PersonSegment],
        ),
    ];
    let readiness = person_readiness_from_workers(&workers);
    assert_eq!(readiness["detect"]["ready"], json!(true));
    assert_eq!(readiness["detect"]["capability"], json!("person_detect"));
    assert_eq!(readiness["track"]["ready"], json!(true));
    assert_eq!(readiness["replace"]["ready"], json!(true));
    assert_eq!(readiness["detectPreview"]["ready"], json!(true));
    assert_eq!(readiness["segment"]["ready"], json!(false));
}

#[tokio::test]
async fn create_image_job_rejects_over_length_negative_prompt() {
    // sc-8884 (F-082): negativePrompt now shares the prompt char cap.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, error) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
            "negativePrompt": "n".repeat(4001),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("negativePrompt")));
}

#[tokio::test]
async fn create_image_job_rejects_oversized_advanced_object() {
    // sc-8884 (F-082): the free-form `advanced` bag is bounded by serialized size.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, error) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
            "advanced": { "blob": "a".repeat(64 * 1024 + 1) },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("advanced")));
}

#[tokio::test]
async fn create_video_job_rejects_over_length_negative_prompt() {
    // sc-8884 (F-082): the negative-prompt cap is shared by the video validator too.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, error) = request(
        app,
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_video",
            "prompt": "a drone shot",
            "negativePrompt": "n".repeat(4001),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("negativePrompt")));
}

/// Seed a minimal audio + non-audio manifest into a test config dir so
/// `resolve_model_manifest_entry` resolves a real `type: audio` entry for the audio route.
fn write_audio_manifest(config_dir: &std::path::Path) {
    std::fs::create_dir_all(config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "kokoro_82m",
              "name": "Kokoro 82M",
              "family": "kokoro",
              "type": "audio",
              "audio": {
                "voices": [{ "id": "af_heart", "language": "en-US" }, { "id": "bm_george", "language": "en-GB" }],
                "languages": ["en-US", "en-GB"],
                "sampleRates": [24000],
                "maxDurationSecs": 30
              },
              "downloads": [
                { "provider": "huggingface", "repo": "hexgrad/Kokoro-82M", "files": ["config.json", "kokoro-v1_0.pth", "voices/*"] }
              ],
              "paths": { "model": "${HF_CACHE}/hexgrad/Kokoro-82M" },
              "ui": { "label": "Kokoro 82M" }
            },
            {
              "id": "moss_sfx_v2",
              "name": "MOSS SoundEffect v2 (SFX)",
              "family": "moss_soundeffect",
              "type": "audio",
              "audio": {
                "languages": ["en", "zh"],
                "sampleRates": [48000],
                "maxDurationSecs": 30
              },
              "downloads": [
                { "provider": "huggingface", "repo": "OpenMOSS-Team/MOSS-SoundEffect-v2.0", "files": ["model_index.json"] }
              ],
              "paths": { "model": "${HF_CACHE}/OpenMOSS-Team/MOSS-SoundEffect-v2.0" },
              "ui": { "label": "MOSS SoundEffect v2" }
            },
            {
              "id": "acestep_v15_turbo",
              "name": "ACE-Step v1.5 XL Turbo (Music)",
              "family": "acestep",
              "type": "audio",
              "audio": {
                "languages": ["en", "zh"],
                "sampleRates": [48000],
                "maxDurationSecs": 600,
                "editModes": ["inpaint", "repaint", "extend"],
                "conditioning": ["AudioEdit"]
              },
              "downloads": [
                { "provider": "huggingface", "repo": "ACE-Step/acestep-v15-xl-turbo-diffusers", "files": ["model_index.json"] }
              ],
              "paths": { "model": "${HF_CACHE}/ACE-Step/acestep-v15-xl-turbo-diffusers" },
              "ui": { "label": "ACE-Step v1.5 XL Turbo" }
            },
            {
              "id": "openvoice_v2",
              "name": "OpenVoice V2 (Voice Conversion)",
              "family": "openvoice",
              "type": "audio",
              "audio": {
                "sampleRates": [22050],
                "conditioning": ["ReferenceAudio"]
              },
              "downloads": [
                { "provider": "huggingface", "repo": "myshell-ai/OpenVoiceV2", "files": ["converter/config.json", "converter/checkpoint.pth"] }
              ],
              "paths": { "model": "${HF_CACHE}/myshell-ai/OpenVoiceV2" },
              "ui": { "label": "OpenVoice V2" }
            },
            {
              "id": "moss_ttsd_v05",
              "name": "MOSS-TTSD v0.5 (Multi-Speaker Dialogue)",
              "family": "moss_ttsd",
              "type": "audio",
              "audio": {
                "languages": ["zh", "en"],
                "sampleRates": [24000],
                "maxDurationSecs": 300,
                "supportsMultiSpeaker": true,
                "maxSpeakers": 2,
                "supportsStreaming": false
              },
              "downloads": [
                { "provider": "huggingface", "repo": "OpenMOSS-Team/MOSS-TTSD-v0.5", "files": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"] },
                { "provider": "huggingface", "repo": "OpenMOSS-Team/XY_Tokenizer_TTSD_V0", "revision": "c83433728e698ed0698e88cb5096bc221fb8f8c5", "coRequisite": true, "files": ["xy_tokenizer.ckpt"] }
              ],
              "paths": { "model": "${HF_CACHE}/OpenMOSS-Team/MOSS-TTSD-v0.5" },
              "ui": { "label": "MOSS-TTSD v0.5 (Multi-Speaker)" }
            },
            {
              "id": "not-audio-img",
              "name": "Not Audio",
              "family": "z_image",
              "type": "image",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/not-audio", "files": ["*.safetensors"] }
              ],
              "paths": {},
              "ui": { "label": "Not Audio" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
}

/// The audio job path (sc-13404): a well-formed `POST /api/v1/audio/jobs` maps the request into an
/// `audio_generate` job whose payload carries the audio knobs (voice / language / targetDurationSecs
/// / seed) verbatim and the resolved `type: audio` manifest entry — the audio twin of how the video
/// route injects `modelManifestEntry`.
#[tokio::test]
async fn create_audio_job_maps_request_to_audio_generate_payload() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Audio Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "kokoro_82m",
            "prompt": "Hello from SceneWorks audio.",
            "voice": "bm_george",
            "language": "en-GB",
            "targetDurationSecs": 4.0,
            "seed": 7,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "audio_generate");
    let payload = &job["payload"];
    assert_eq!(payload["model"], "kokoro_82m");
    assert_eq!(payload["prompt"], "Hello from SceneWorks audio.");
    assert_eq!(payload["voice"], "bm_george");
    assert_eq!(payload["language"], "en-GB");
    assert_eq!(payload["targetDurationSecs"], 4.0);
    assert_eq!(payload["seed"], 7);
    // The resolved manifest entry must travel with the job (the worker resolves weights from it),
    // and it must be THIS model's `type: audio` entry — not `{}` (which would slip a non-audio job
    // through) — carrying the HF download repo the worker resolves the Kokoro snapshot from.
    let entry = &payload["modelManifestEntry"];
    assert_eq!(entry["id"], "kokoro_82m");
    assert_eq!(entry["type"], "audio");
    assert_eq!(entry["downloads"][0]["repo"], "hexgrad/Kokoro-82M");
    // requestedGpu is stripped from the payload (it rides the job envelope), mirroring the video route.
    assert!(payload.get("requestedGpu").is_none());
}

/// The multi-speaker path (MOSS-TTSD, sc-13676): a well-formed dialogue `POST /api/v1/audio/jobs`
/// carries the segmented `script` (each turn's text + speaker) through to the worker payload verbatim,
/// with an EMPTY prompt accepted (the script carries the text), and injects the resolved MOSS-TTSD
/// `type: audio` manifest entry so the worker resolves both the AR + codec snapshots.
#[tokio::test]
async fn create_audio_job_maps_the_multi_speaker_script() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Dialogue Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "moss_ttsd_v05",
            // Empty prompt is accepted because a non-empty multi-speaker script carries the text.
            "prompt": "",
            "script": [
                { "text": "Hello, how are you today?", "speaker": "S1" },
                { "text": "I'm doing great, thanks for asking!", "speaker": "S2" },
            ],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "audio_generate");
    let payload = &job["payload"];
    assert_eq!(payload["model"], "moss_ttsd_v05");
    // The segmented dialogue travels verbatim to the worker as `script` (camelCase round-trip).
    let script = payload["script"]
        .as_array()
        .expect("script array in payload");
    assert_eq!(script.len(), 2);
    assert_eq!(script[0]["text"], "Hello, how are you today?");
    assert_eq!(script[0]["speaker"], "S1");
    assert_eq!(script[1]["speaker"], "S2");
    // The MOSS-TTSD manifest entry travels so the worker resolves the AR + codec co-requisite.
    let entry = &payload["modelManifestEntry"];
    assert_eq!(entry["id"], "moss_ttsd_v05");
    assert_eq!(entry["type"], "audio");
    assert_eq!(entry["audio"]["supportsMultiSpeaker"], true);
    assert_eq!(entry["audio"]["maxSpeakers"], 2);
}

/// A multi-speaker script naming >1 turn but with a whitespace-only prompt AND no script is still
/// rejected — the relaxed prompt guard only accepts an empty prompt when a real script is present.
#[tokio::test]
async fn create_audio_job_rejects_empty_prompt_and_empty_script() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Dialogue Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // Empty prompt + empty script array → rejected (a script with no segments is not content).
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({ "projectId": project_id, "model": "moss_ttsd_v05", "prompt": "", "script": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    // A script segment with empty text → rejected.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "moss_ttsd_v05",
            "prompt": "",
            "script": [{ "text": "   ", "speaker": "S1" }],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

/// The Sound FX path (MOSS-SoundEffect, sc-13409): a well-formed SFX `POST /api/v1/audio/jobs`
/// carries the diffusion sampling knobs (`guidance` = CFG scale, `steps`) through to the worker
/// payload verbatim alongside the shared audio knobs, and injects the resolved MOSS `type: audio`
/// manifest entry — so the worker can map guidance/steps onto the top-level GenerationRequest. No
/// voice is sent (MOSS advertises no voice surface).
#[tokio::test]
async fn create_audio_job_maps_sfx_sampling_knobs() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "SFX Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "moss_sfx_v2",
            "prompt": "a heavy wooden door creaking open",
            "language": "en",
            "targetDurationSecs": 3.0,
            "guidance": 6.5,
            "steps": 60,
            "seed": 11,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "audio_generate");
    let payload = &job["payload"];
    assert_eq!(payload["model"], "moss_sfx_v2");
    assert_eq!(payload["prompt"], "a heavy wooden door creaking open");
    assert_eq!(payload["language"], "en");
    assert_eq!(payload["targetDurationSecs"], 3.0);
    // The SFX sampling knobs ride through to the worker payload (which maps them onto the top-level
    // GenerationRequest's guidance/steps — not AudioParams).
    assert_eq!(payload["guidance"], 6.5);
    assert_eq!(payload["steps"], 60);
    assert_eq!(payload["seed"], 11);
    // No voice on an SFX request; the MOSS manifest entry travels for weight resolution.
    assert!(payload.get("voice").is_none());
    let entry = &payload["modelManifestEntry"];
    assert_eq!(entry["id"], "moss_sfx_v2");
    assert_eq!(entry["type"], "audio");
    assert_eq!(
        entry["downloads"][0]["repo"],
        "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
    );
}

/// The Voice Clone path (OpenVoice V2 conversion chain, sc-13411 C4): a `POST /api/v1/audio/jobs` that
/// names a converter model + `referenceAudioAssetId` carries the reference + match strength through to
/// the worker payload AND injects a SECOND manifest entry — the base TTS model (`baseModelManifestEntry`)
/// — so the worker resolves both snapshots. The converter's own entry rides as `modelManifestEntry`.
#[tokio::test]
async fn create_audio_job_injects_base_model_entry_for_voice_clone() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Voice Clone Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "openvoice_v2",
            "prompt": "Clone this into my reference voice.",
            "referenceAudioAssetId": "ref-voice-1",
            "matchStrength": 0.5,
            "seed": 3,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "audio_generate");
    let payload = &job["payload"];
    assert_eq!(payload["model"], "openvoice_v2");
    assert_eq!(payload["referenceAudioAssetId"], "ref-voice-1");
    assert_eq!(payload["matchStrength"], 0.5);
    // The selected converter's entry rides as modelManifestEntry (weights resolution).
    assert_eq!(payload["modelManifestEntry"]["id"], "openvoice_v2");
    assert_eq!(
        payload["modelManifestEntry"]["downloads"][0]["repo"],
        "myshell-ai/OpenVoiceV2"
    );
    // ...and the base TTS (Kokoro) entry is injected so the worker resolves the base generator too.
    let base = &payload["baseModelManifestEntry"];
    assert_eq!(base["id"], "kokoro_82m");
    assert_eq!(base["type"], "audio");
    assert_eq!(base["downloads"][0]["repo"], "hexgrad/Kokoro-82M");
}

/// A non-voice-clone audio job (no reference) carries NO `baseModelManifestEntry` — the base-model
/// resolution is scoped to the voice-clone path so ordinary Speech/SFX/Music jobs are untouched.
#[tokio::test]
async fn create_audio_job_omits_base_model_entry_without_a_reference() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Plain Audio Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "kokoro_82m",
            "prompt": "Just plain speech.",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert!(job["payload"].get("baseModelManifestEntry").is_none());
}

/// The Voice Clone match-strength floor (sc-13411 C4): the API blanket-bounds `matchStrength` to
/// 0..=1 (the converter re-checks it), so an out-of-range value is a 400 rather than reaching the worker.
#[tokio::test]
async fn create_audio_job_rejects_out_of_range_match_strength() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Bad Strength Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, _body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "openvoice_v2",
            "prompt": "over-driven strength",
            "referenceAudioAssetId": "ref-voice-1",
            "matchStrength": 1.5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// The audio validator rejects out-of-blanket sampling knobs up front (sc-13409) — the API blanket
/// (guidance 0..=100, steps 1..=10000); the per-model range (MOSS: guidance 1..=20, steps ≤1000) is
/// the generator's own `validate` at generate time.
#[tokio::test]
async fn create_audio_job_rejects_nonsense_sampling_knobs() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "SFX Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "moss_sfx_v2",
            "prompt": "thunder",
            "steps": 0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("steps")));
}

/// The Music path (ACE-Step, sc-13410): a well-formed music `POST /api/v1/audio/jobs` carries the
/// describe-the-music sub-block (bpm / musicalKey / lyrics) + steps AND the extend/edit source band
/// (sourceAudioAssetId / editMode / editRegion* / editStrength) through to the worker payload verbatim,
/// alongside the resolved ACE-Step `type: audio` manifest entry — so the worker maps them onto
/// AudioParams + a Conditioning::AudioEdit.
#[tokio::test]
async fn create_audio_job_maps_music_and_edit_fields() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Music Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "gentle lofi piano loop",
            "language": "en",
            "targetDurationSecs": 8.0,
            "steps": 8,
            "bpm": 92.0,
            "musicalKey": "C minor",
            "lyrics": "[verse] la la la",
            "sourceAudioAssetId": "audio-src-1",
            "editMode": "extend",
            "editRegionEndSecs": 20.0,
            "editStrength": 0.5,
            "seed": 5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "audio_generate");
    let payload = &job["payload"];
    assert_eq!(payload["model"], "acestep_v15_turbo");
    // Describe-the-music sub-block (rides AudioParams music fields).
    assert_eq!(payload["bpm"], 92.0);
    assert_eq!(payload["musicalKey"], "C minor");
    assert_eq!(payload["lyrics"], "[verse] la la la");
    assert_eq!(payload["steps"], 8);
    // Extend/edit source band (rides a Conditioning::AudioEdit).
    assert_eq!(payload["sourceAudioAssetId"], "audio-src-1");
    assert_eq!(payload["editMode"], "extend");
    assert_eq!(payload["editRegionEndSecs"], 20.0);
    assert_eq!(payload["editStrength"], 0.5);
    let entry = &payload["modelManifestEntry"];
    assert_eq!(entry["id"], "acestep_v15_turbo");
    assert_eq!(entry["type"], "audio");
    assert_eq!(
        entry["downloads"][0]["repo"],
        "ACE-Step/acestep-v15-xl-turbo-diffusers"
    );
}

/// A half-specified edit (a source track without an edit mode, or vice versa) is malformed — one names
/// WHAT to edit, the other HOW — so the validator rejects it up front rather than silently dropping the
/// edit (sc-13410).
#[tokio::test]
async fn create_audio_job_rejects_half_specified_edit() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Music Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "gentle lofi piano loop",
            // A source with no editMode → rejected as a malformed pair.
            "sourceAudioAssetId": "audio-src-1",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("editMode")));
}

/// The Music validator sanity-bounds the describe-the-music + edit fields up front (sc-13410): a
/// non-positive BPM, an unknown edit mode, and a mis-ordered region are all 400s before the job is
/// enqueued (the per-model gate — mode ∈ advertised editModes, region inside the clip — is the
/// generator's own `validate`).
#[tokio::test]
async fn create_audio_job_rejects_nonsense_music_fields() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Music Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // A non-positive BPM is rejected.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "loop",
            "bpm": 0.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"].as_str().is_some_and(|d| d.contains("bpm")));

    // An unknown edit mode token is rejected up front.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "loop",
            "sourceAudioAssetId": "audio-src-1",
            "editMode": "morph",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|d| d.contains("editMode")));

    // A region whose end is at/below its start is rejected.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "loop",
            "sourceAudioAssetId": "audio-src-1",
            "editMode": "inpaint",
            "editRegionStartSecs": 5.0,
            "editRegionEndSecs": 2.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|d| d.contains("editRegionEndSecs")));
}

/// The editMode token is case-insensitive at the API, matching the worker's own case handling
/// (`edit_mode.map(|m| m.to_lowercase())` at deserialize, then `parse_audio_edit_mode`). A mixed-case
/// KNOWN token (`"Extend"`) is accepted and forwarded verbatim (the worker lowercases it), while a
/// mixed-case UNKNOWN token (`"Morph"`) is still rejected — lowercasing widens casing, not the mode
/// set (sc-13410, worker-parity fix).
#[tokio::test]
async fn create_audio_job_accepts_case_insensitive_edit_mode() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Music Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // A mixed-case KNOWN token is accepted (parity with the worker) and forwarded verbatim.
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "gentle lofi piano loop",
            "sourceAudioAssetId": "audio-src-1",
            "editMode": "Extend",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["editMode"], "Extend");

    // A mixed-case UNKNOWN token is still rejected — casing widened, the mode set did not.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "loop",
            "sourceAudioAssetId": "audio-src-1",
            "editMode": "Morph",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|d| d.contains("editMode")));
}

/// The audio route is a door for `type: audio` models only: an image/video model posted here is
/// rejected up front rather than failing deep in the worker's audio lane (sc-13404).
#[tokio::test]
async fn create_audio_job_rejects_non_audio_model() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Audio Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // A path-safe but non-audio model id resolves to a `type: image` entry → rejected.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "not-audio-img",
            "prompt": "this is not audio",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("audio")));

    // An unknown model resolves to `{}` (no type) and is rejected for the same reason.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "definitely-not-real",
            "prompt": "hello",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// The audio validator bounds the script prompt exactly as the image/video validators do (sc-13404).
#[tokio::test]
async fn create_audio_job_rejects_empty_prompt() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Audio Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "kokoro_82m",
            "prompt": "   ",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("prompt")));
}

// The queue-lifecycle tests below drive `POST /api/v1/jobs` — claim, cancel, retry,
// progress, clear, stale sweep. They need *a* GPU-routed claimable job and do not care
// which; they use `image_detail` (a real caller of this route — the web batch ops post it)
// rather than `image_generate`, which sc-12305 moved behind its typed route so the model's
// manifest entry is always resolved. The worker `capabilities` match by job type
// (`required_capability`), so the two move together.

#[tokio::test]
async fn worker_can_register_claim_and_complete_job_through_http() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist over hills" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let (status, claimed) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(claimed["job"]["id"], created["id"]);
    assert_eq!(claimed["job"]["status"], "preparing");

    let job_id = created["id"].as_str().expect("job id is string");
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Done",
            "workerId": "worker-1",
            "result": { "assetIds": ["asset-1"] }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["status"], "completed");
    assert_eq!(completed["result"], json!({ "assetIds": ["asset-1"] }));

    let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["completed"], 1);
    assert_eq!(queue["workers"][0]["status"], "idle");
}

#[tokio::test]
async fn progress_ticks_only_republish_queue_on_status_change() {
    // sc-4203 (F-API-5): a pure progress tick (status unchanged) must not trigger the
    // full queue-summary recompute + queue.updated broadcast; a status transition must.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");

    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let job_id = created["id"].as_str().expect("job id is string").to_owned();
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    // Move the job into `running` (a transition from `preparing`).
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({ "status": "running", "stage": "running", "progress": 0.2, "message": "step", "workerId": "worker-1" }),
    )
    .await;

    // Subscribe AFTER the transition so we only observe the next ticks' events.
    let mut events = state.events.subscribe();

    // A pure progress tick (running -> running): job.updated, but NOT queue.updated.
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({ "status": "running", "stage": "running", "progress": 0.6, "message": "step", "workerId": "worker-1" }),
    )
    .await;
    let tick_events = drain_event_names(&mut events).await;
    assert!(
        tick_events.iter().any(|name| name == "job.updated"),
        "a progress tick still emits job.updated: {tick_events:?}"
    );
    assert!(
        !tick_events.iter().any(|name| name == "queue.updated"),
        "a pure progress tick must not republish the queue: {tick_events:?}"
    );

    // A status transition (running -> completed) republishes the queue.
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({ "status": "completed", "stage": "completed", "progress": 1, "message": "done", "workerId": "worker-1" }),
    )
    .await;
    let done_events = drain_event_names(&mut events).await;
    assert!(
        done_events.iter().any(|name| name == "queue.updated"),
        "a status transition must republish the queue: {done_events:?}"
    );
}

#[tokio::test]
async fn canceling_queued_job_finishes_without_worker_acknowledgement() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist over hills" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let job_id = created["id"].as_str().expect("job id is string");
    let (status, canceled) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/cancel"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(canceled["status"], "canceled");
    assert_eq!(canceled["stage"], "canceled");
    assert_eq!(canceled["progress"], 1.0);
    assert_eq!(canceled["cancelRequested"], true);
    assert_eq!(canceled["message"], "Canceled before a worker started.");
    assert!(canceled["canceledAt"].is_string());
    assert!(canceled["completedAt"].is_string());
    assert_eq!(canceled["workerId"], Value::Null);

    let (status, queue) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["canceled"], 1);
    assert_eq!(queue["counts"]["queued"], 0);

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, claimed) = request(
        app,
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(claimed["job"], Value::Null);
}

#[tokio::test]
async fn clear_jobs_soft_hides_terminal_items_from_the_queue() {
    // sc-12231 / issue #1556: POST /api/v1/jobs/clear drops every terminal job from
    // the queue list + counts, returns the cleared ids, and leaves active jobs alone.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // A queued job we cancel (terminal) + a queued job left active.
    let (_, terminal) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "done" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let terminal_id = terminal["id"].as_str().expect("job id").to_owned();
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{terminal_id}/cancel"),
        Value::Null,
    )
    .await;

    let (_, active) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "wait" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let active_id = active["id"].as_str().expect("job id").to_owned();

    // Both are listed before clearing.
    let (_, before) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(before.as_array().expect("jobs array").len(), 2);

    // Clear (empty body == all projects). Reports the one terminal job by id.
    let (status, cleared) = request(app.clone(), "POST", "/api/v1/jobs/clear", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cleared["cleared"], 1);
    assert_eq!(cleared["clearedIds"], json!([terminal_id]));

    // The queue now lists only the still-active job.
    let (_, after) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    let ids: Vec<&str> = after
        .as_array()
        .expect("jobs array")
        .iter()
        .filter_map(|job| job["id"].as_str())
        .collect();
    assert_eq!(ids, vec![active_id.as_str()]);

    // Status counts drop the canceled job; the queued one remains.
    let (_, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(queue["counts"]["canceled"], 0);
    assert_eq!(queue["counts"]["queued"], 1);
}

#[tokio::test]
async fn clear_jobs_scopes_to_the_requested_project() {
    // sc-12231: the clear honors the body's projectId so clearing one workspace's
    // completed items never touches another's.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let mut ids = Vec::new();
    for project in ["project-a", "project-b"] {
        let (_, job) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "image_detail",
                "projectId": project,
                "projectName": project,
                "payload": { "prompt": "done" },
                "requestedGpu": "auto"
            }),
        )
        .await;
        let id = job["id"].as_str().expect("job id").to_owned();
        request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{id}/cancel"),
            Value::Null,
        )
        .await;
        ids.push(id);
    }

    // Clear only project-a.
    let (status, cleared) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/clear",
        json!({ "projectId": "project-a" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cleared["clearedIds"], json!([ids[0]]));

    // project-a is empty; project-b's canceled job is untouched.
    let (_, a_jobs) = request(
        app.clone(),
        "GET",
        "/api/v1/jobs?projectId=project-a",
        Value::Null,
    )
    .await;
    assert!(a_jobs.as_array().expect("jobs array").is_empty());
    let (_, b_jobs) = request(app, "GET", "/api/v1/jobs?projectId=project-b", Value::Null).await;
    let b_ids: Vec<&str> = b_jobs
        .as_array()
        .expect("jobs array")
        .iter()
        .filter_map(|job| job["id"].as_str())
        .collect();
    assert_eq!(b_ids, vec![ids[1].as_str()]);
}

#[tokio::test]
async fn cancel_pending_jobs_cancels_every_queued_item_but_not_active_ones() {
    // sc-13448: POST /api/v1/jobs/cancel-pending flips every pending (queued) job to
    // terminal `canceled` in one call, returns the updated snapshots, and leaves a
    // worker-owned (active) job running.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // Register a worker + create three jobs, then claim one so it is active.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let mut ids = Vec::new();
    for prompt in ["one", "two", "three"] {
        let (_, job) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "image_detail",
                "projectId": "project-1",
                "projectName": "Project 1",
                "payload": { "prompt": prompt },
                "requestedGpu": "auto"
            }),
        )
        .await;
        ids.push(job["id"].as_str().expect("job id").to_owned());
    }

    // Claim the oldest job — it becomes active (Preparing) and worker-owned.
    let (_, claimed) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    let active_id = claimed["job"]["id"]
        .as_str()
        .expect("claimed id")
        .to_owned();
    assert_eq!(claimed["job"]["status"], "preparing");

    // Cancel all pending (empty body == all projects). The two still-queued jobs are
    // canceled; the active job is not.
    let (status, canceled) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/cancel-pending",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(canceled["canceled"], 2);
    let canceled_jobs = canceled["jobs"].as_array().expect("jobs array");
    assert_eq!(canceled_jobs.len(), 2);
    for job in canceled_jobs {
        assert_eq!(job["status"], "canceled");
        assert_eq!(job["stage"], "canceled");
        assert_eq!(job["cancelRequested"], true);
        assert_eq!(job["message"], "Canceled before a worker started.");
        assert!(job["canceledAt"].is_string());
        assert_ne!(job["id"], json!(active_id));
    }

    // The active job is untouched: still preparing, no cancel requested.
    let (_, active) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{active_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(active["status"], "preparing");
    assert_eq!(active["cancelRequested"], false);

    // Status counts: two canceled, none queued, the active one still in flight.
    let (_, queue) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(queue["counts"]["canceled"], 2);
    assert_eq!(queue["counts"]["queued"], 0);

    // A second sweep cancels nothing — no pending jobs remain.
    let (status, again) = request(app, "POST", "/api/v1/jobs/cancel-pending", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(again["canceled"], 0);
}

#[tokio::test]
async fn clear_single_job_soft_hides_only_that_terminal_job() {
    // sc-12231 / issue #1556: POST /api/v1/jobs/:id/clear (the per-card ×) drops one
    // terminal job from the queue and leaves its siblings alone.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // Two queued jobs; cancel one so it is terminal, leave the other active.
    let mut ids = Vec::new();
    for prompt in ["done", "wait"] {
        let (_, job) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "image_detail",
                "projectId": "project-1",
                "projectName": "Project 1",
                "payload": { "prompt": prompt },
                "requestedGpu": "auto"
            }),
        )
        .await;
        ids.push(job["id"].as_str().expect("job id").to_owned());
    }
    let (terminal_id, active_id) = (ids[0].clone(), ids[1].clone());
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{terminal_id}/cancel"),
        Value::Null,
    )
    .await;

    // Clear just the terminal one.
    let (status, cleared) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{terminal_id}/clear"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cleared["id"], terminal_id);

    // Only the still-active job remains in the queue.
    let (_, after) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    let remaining: Vec<&str> = after
        .as_array()
        .expect("jobs array")
        .iter()
        .filter_map(|job| job["id"].as_str())
        .collect();
    assert_eq!(remaining, vec![active_id.as_str()]);
}

#[tokio::test]
async fn clear_single_job_rejects_a_non_terminal_job() {
    // sc-12231: clearing an active (queued) job is a 400 — the × only appears on
    // terminal cards, and the server refuses to soft-hide a live job.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (_, job) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "wait" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let job_id = job["id"].as_str().expect("job id").to_owned();

    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/clear"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // The job is untouched — still listed.
    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(jobs.as_array().expect("jobs array").len(), 1);
}

#[tokio::test]
async fn image_job_route_threads_upscale_contract_when_enabled() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "z_image_turbo",
              "name": "Z-Image-Turbo",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image_diffusers",
              "capabilities": ["text_to_image"],
              "downloads": [],
              "paths": {},
              "resources": {
                "imageUpscalers": {
                  "real-esrgan": {
                    "x2": { "repo": "nateraw/real-esrgan", "file": "RealESRGAN_x2plus.pth" },
                    "x4": { "repo": "nateraw/real-esrgan", "file": "RealESRGAN_x4plus.pth" }
                  }
                }
              },
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": [], "types": [] },
              "ui": {}
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let base_request = json!({
        "projectId": "project-1",
        "mode": "text_to_image",
        "prompt": "mist over hills",
        "count": 1,
        "seed": 123
    });
    let (status, base_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        base_request.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(base_job["payload"].get("upscale").is_none());
    assert_eq!(
        base_job["payload"]["modelManifestEntry"]["resources"]["imageUpscalers"]["real-esrgan"]
            ["x4"]["file"],
        json!("RealESRGAN_x4plus.pth")
    );

    let mut disabled_request = base_request.clone();
    disabled_request["upscale"] = json!({ "enabled": false, "factor": 4, "engine": "real-esrgan" });
    let (status, disabled_job) =
        request(app.clone(), "POST", "/api/v1/image/jobs", disabled_request).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(disabled_job["payload"], base_job["payload"]);

    let mut enabled_request = base_request;
    enabled_request["upscale"] = json!({ "enabled": true, "factor": 4, "engine": "real-esrgan" });
    let (status, enabled_job) =
        request(app.clone(), "POST", "/api/v1/image/jobs", enabled_request).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        enabled_job["payload"]["upscale"],
        json!({ "enabled": true, "factor": 4, "engine": "real-esrgan" })
    );

    let (status, error) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
            "seed": 123,
            "upscale": { "enabled": true, "factor": 3, "engine": "real-esrgan" }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["detail"], "upscale.factor must be 2 or 4");
}

#[tokio::test]
async fn image_job_route_threads_reference_asset_ids() {
    // sc-6358 / sc-6107 regression guard: the multi-reference edit picker sends a top-level
    // `referenceAssetIds` array. The typed ImageJobRequest must carry the plural list through to the
    // job payload — without the field, serde drops the unknown key on deserialize and `to_json_object`
    // never forwards it, so the worker's `flux2_edit_reference_ids` never sees the references and the
    // FLUX.2 multi-reference edit silently no-ops (the original sc-6211 defect).
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, edit_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "edit_image",
            "prompt": "in the style of the references",
            "sourceAssetId": "work-scratch",
            "referenceAssetIds": ["work-scratch", "ref-a", "ref-b"],
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(edit_job["type"], "image_edit");
    assert_eq!(
        edit_job["payload"]["referenceAssetIds"],
        json!(["work-scratch", "ref-a", "ref-b"])
    );

    // A request that doesn't send the list still serializes a present (empty) array — the worker's
    // `string_list` treats missing/empty identically, so this never surprises a single-reference edit.
    let (status, plain_job) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "edit_image",
            "prompt": "make it dusk",
            "sourceAssetId": "asset-1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(plain_job["payload"]["referenceAssetIds"], json!([]));
}

#[tokio::test]
async fn ideogram_plain_text_job_returns_immediately_in_pending_caption() {
    // sc-9120: a direct/headless plain-text Ideogram 4 job returns 201 IMMEDIATELY in the non-claimable
    // `pending_caption` status — the POST no longer waits on the magic-prompt expansion at all. A
    // background watcher then runs the same separate expansion the web runs, rewrites the prompt to the
    // rich JSON caption, and promotes the job to `queued`, so the worker only ever sees it once queued.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // The POST is NOT spawned/awaited concurrently — it must return on its own, promptly.
    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "a red fox in a snowy forest",
            "model": "ideogram_4",
            "count": 1,
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["type"], "image_generate");
    assert_eq!(
        image_job["status"], "pending_caption",
        "the POST must return immediately in pending_caption, not wait on the caption"
    );
    // Still the ORIGINAL prompt at this point — the rewrite happens on the async promotion.
    assert_eq!(
        image_job["payload"]["prompt"],
        "a red fox in a snowy forest"
    );
    let image_job_id = image_job["id"].as_str().expect("job id").to_owned();

    // The background watcher enqueues the magic-prompt expansion carrying the plain prompt, the
    // magic_prompt task, and the derived aspect ratio.
    let refine_id = wait_for_prompt_refine_job(&app).await;
    let (status, refine_job) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{refine_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(refine_job["type"], "prompt_refine");
    assert_eq!(refine_job["payload"]["task"], "magic_prompt");
    assert_eq!(
        refine_job["payload"]["prompt"],
        "a red fox in a snowy forest"
    );
    assert_eq!(refine_job["payload"]["aspectRatio"], "1:1");

    // Complete the expansion with a rich caption. The unclaimed job has no owner, so the progress
    // report omits a workerId (matching the store's `(None, None)` ownership rule).
    let caption = r#"{"high_level_description": "a red fox", "compositional_deconstruction": {"background": "a snowy forest at golden hour", "elements": []}}"#;
    complete_prompt_refine_job(&app, &refine_id, json!({ "refinedPrompt": caption })).await;

    // The watcher now promotes the image job to `queued` with the rich caption as its prompt.
    let promoted = wait_for_job_out_of_pending_caption(&app, &image_job_id).await;
    assert_eq!(promoted["status"], "queued");
    assert_eq!(promoted["payload"]["model"], "ideogram_4");
    assert_eq!(promoted["payload"]["prompt"], caption);
}

#[tokio::test]
async fn ideogram_plain_text_job_degrades_to_original_prompt_when_expansion_fails() {
    // sc-9120 graceful degradation: if the magic_prompt expansion fails (e.g. the refiner is not
    // staged), the background watcher still promotes the image job to `queued` with the ORIGINAL prompt
    // — the worker's format-guard + placeholder reseed net (sc-6501) remains the fallback, so the job
    // is never stranded in pending_caption and a render is always produced.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "a red fox in a snowy forest",
            "model": "ideogram_4",
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["status"], "pending_caption");
    let image_job_id = image_job["id"].as_str().expect("job id").to_owned();

    let refine_id = wait_for_prompt_refine_job(&app).await;
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{refine_id}/progress"),
        json!({
            "status": "failed",
            "stage": "failed",
            "progress": 0,
            "message": "Expansion failed.",
            "error": "prompt-refine model not staged"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let promoted = wait_for_job_out_of_pending_caption(&app, &image_job_id).await;
    assert_eq!(promoted["status"], "queued");
    assert_eq!(promoted["type"], "image_generate");
    assert_eq!(promoted["payload"]["prompt"], "a red fox in a snowy forest");
}

#[tokio::test]
async fn ideogram_plain_text_job_degrades_on_invalid_caption_after_bounded_resamples() {
    // sc-9120: the expansion runs in a BACKGROUND task (no HTTP connection held), so a completed-but-
    // invalid caption may be re-sampled a small, bounded number of times (MAX_CAPTION_ATTEMPTS). When
    // every attempt returns prose (not a caption), the watcher degrades the image job to `queued` with
    // the ORIGINAL prompt (the worker's reseed net recovers it). The re-sample budget is small and
    // bounded, so an impatient client's retries can't stack unbounded magic-prompt jobs.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "a red fox in a snowy forest",
            "model": "ideogram_4",
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["status"], "pending_caption");
    let image_job_id = image_job["id"].as_str().expect("job id").to_owned();

    // Feed every re-sample a prose (non-caption) reply until the watcher exhausts its budget and
    // degrades. Completing each refine job unblocks the next attempt (a fresh refine job).
    let mut previous: Option<String> = None;
    loop {
        let job =
            wait_for_job_out_of_pending_caption_or_refine(&app, &image_job_id, previous.as_deref())
                .await;
        match job {
            PendingOrRefine::Promoted(promoted) => {
                // Degraded to the original prompt once the budget was exhausted.
                assert_eq!(promoted["status"], "queued");
                assert_eq!(promoted["payload"]["prompt"], "a red fox in a snowy forest");
                break;
            }
            PendingOrRefine::Refine(refine_id) => {
                complete_prompt_refine_job(
                    &app,
                    &refine_id,
                    json!({ "refinedPrompt": "just a fox, nothing structured" }),
                )
                .await;
                previous = Some(refine_id);
            }
        }
    }
}

#[tokio::test]
async fn pending_caption_ideogram_job_is_cancelable() {
    // sc-9120: a pending_caption job must be cancelable — it goes straight to `canceled` (no worker to
    // acknowledge), and a subsequent caption promotion does NOT resurrect it (the guarded UPDATE only
    // matches a still-pending row).
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "a red fox in a snowy forest",
            "model": "ideogram_4",
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["status"], "pending_caption");
    let image_job_id = image_job["id"].as_str().expect("job id").to_owned();

    // Cancel while still pending — it terminates immediately.
    let (status, canceled) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{image_job_id}/cancel"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(canceled["status"], "canceled");

    // Even if the background watcher's expansion later completes and it tries to promote, the canceled
    // job must NOT flip back to queued. Complete the refine job it enqueued and confirm the image job
    // stays canceled.
    let refine_id = wait_for_prompt_refine_job(&app).await;
    let caption =
        r#"{"compositional_deconstruction": {"background": "a snowy forest", "elements": []}}"#;
    complete_prompt_refine_job(&app, &refine_id, json!({ "refinedPrompt": caption })).await;
    // Give the watcher a moment to attempt (and no-op) the promotion.
    for _ in 0..25 {
        let (_, job) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/jobs/{image_job_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(
            job["status"], "canceled",
            "a canceled pending_caption job must never be resurrected by a late promotion"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn concurrent_ideogram_captions_share_one_refine_job() {
    // sc-9120: two identical plain-text Ideogram jobs (an impatient client re-POSTing) must reuse ONE
    // in-flight magic-prompt refine job rather than stacking a fresh one each time. Both image jobs land
    // in pending_caption; the second caption watcher reuses the first's refine job.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let post = |app: axum::Router| async move {
        request(
            app,
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": "project-1",
                "mode": "text_to_image",
                "prompt": "a red fox in a snowy forest",
                "model": "ideogram_4",
                "seed": 7
            }),
        )
        .await
    };

    let (status_a, job_a) = post(app.clone()).await;
    assert_eq!(status_a, StatusCode::CREATED);
    assert_eq!(job_a["status"], "pending_caption");
    // Wait for the first refine job to be in flight before the second POST, so the reuse path is hit
    // deterministically.
    let refine_id = wait_for_prompt_refine_job(&app).await;

    let (status_b, job_b) = post(app.clone()).await;
    assert_eq!(status_b, StatusCode::CREATED);
    assert_eq!(job_b["status"], "pending_caption");
    assert_ne!(job_a["id"], job_b["id"], "two distinct image jobs");

    // Let the second watcher run its reuse lookup, then assert exactly one refine job exists.
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    let refine_ids: Vec<String> = jobs
        .as_array()
        .expect("jobs is an array")
        .iter()
        .filter(|job| job["type"] == "prompt_refine")
        .filter_map(|job| job["id"].as_str().map(str::to_owned))
        .collect();
    assert_eq!(
        refine_ids,
        vec![refine_id.clone()],
        "the second identical caption must reuse the in-flight refine job, not stack a new one"
    );

    // Completing the shared refine job promotes BOTH image jobs to queued with the rich caption.
    let caption =
        r#"{"compositional_deconstruction": {"background": "a snowy forest", "elements": []}}"#;
    complete_prompt_refine_job(&app, &refine_id, json!({ "refinedPrompt": caption })).await;
    let a = wait_for_job_out_of_pending_caption(&app, job_a["id"].as_str().unwrap()).await;
    let b = wait_for_job_out_of_pending_caption(&app, job_b["id"].as_str().unwrap()).await;
    assert_eq!(a["status"], "queued");
    assert_eq!(a["payload"]["prompt"], caption);
    assert_eq!(b["status"], "queued");
    assert_eq!(b["payload"]["prompt"], caption);
}

#[tokio::test]
async fn ideogram_caption_prompt_dispatches_without_expansion() {
    // sc-6519: an already-structured caption (the normal web submit) is never re-expanded — no
    // magic_prompt job is enqueued and the job dispatches immediately with the caption unchanged.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let caption =
        r#"{"compositional_deconstruction": {"background": "a beach at sunset", "elements": []}}"#;
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": caption,
            "model": "ideogram_4",
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["payload"]["prompt"], caption);

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert!(
        jobs.as_array()
            .expect("jobs is an array")
            .iter()
            .all(|job| job["type"] != "prompt_refine"),
        "an already-structured caption must not enqueue a magic_prompt job"
    );
}

#[tokio::test]
async fn ideogram_edit_job_skips_auto_caption() {
    // sc-6519: an Ideogram 4 EDIT job conditions on a source image and its prompt is an edit
    // instruction, not a scene to caption — the auto-caption must not rewrite it, and no magic_prompt
    // job is enqueued.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "edit_image",
            "prompt": "make the sky purple",
            "model": "ideogram_4",
            "sourceAssetId": "asset-1",
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "image_edit");
    assert_eq!(job["payload"]["prompt"], "make the sky purple");

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert!(
        jobs.as_array()
            .expect("jobs is an array")
            .iter()
            .all(|job| job["type"] != "prompt_refine"),
        "an Ideogram edit job must not enqueue a magic_prompt job"
    );
}

#[tokio::test]
async fn non_ideogram_image_job_skips_auto_caption() {
    // sc-6519: the auto-caption is gated to the Ideogram 4 models — a plain prompt for any other
    // model dispatches immediately, unchanged, with no magic_prompt expansion job.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "model": "flux_dev",
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["payload"]["prompt"], "mist over hills");

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert!(
        jobs.as_array()
            .expect("jobs is an array")
            .iter()
            .all(|job| job["type"] != "prompt_refine"),
        "a non-Ideogram job must not enqueue a magic_prompt job"
    );
}

#[tokio::test]
async fn image_caption_refine_job_resolves_asset_to_confined_image_path() {
    // epic 8102 / sc-8108: the reference-image → JSON caption flow POSTs `task: "image_caption"` with a
    // project `sourceAssetId` (+ `projectId`) and the vision model's repo. The handler resolves the
    // asset to an absolute on-disk `imagePath` (inside the project dir), forwards the model verbatim,
    // and enqueues a `prompt_refine` job carrying that image-caption payload (no text prompt required).
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Caption Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Reference.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();
    let rel_path = asset["file"]["path"]
        .as_str()
        .expect("file path")
        .to_owned();

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({
            "task": "image_caption",
            "sourceAssetId": asset_id,
            "projectId": project_id,
            "model": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "prompt_refine");
    assert_eq!(job["payload"]["task"], "image_caption");
    assert_eq!(
        job["payload"]["model"],
        "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
    );
    // No text prompt is required for an image-caption job.
    assert!(job["payload"].get("prompt").is_none());
    // The resolved imagePath is the asset's absolute on-disk location inside the project dir.
    // Compare as paths: the handler joins component-wise (native separators), while `rel_path`
    // keeps the asset record's literal `/`, so a string comparison breaks on Windows (sc-8967).
    let expected = project_path.join(&rel_path);
    assert_eq!(
        std::path::Path::new(job["payload"]["imagePath"].as_str().unwrap()),
        expected
    );
}

#[tokio::test]
async fn image_caption_refine_job_requires_source_asset_and_project() {
    // sc-8108: the image-caption task is driven by a reference asset, so it must reject a request that
    // omits the `sourceAssetId` or `projectId` rather than enqueue an unresolvable job.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({ "task": "image_caption", "projectId": "project-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({ "task": "image_caption", "sourceAssetId": "asset-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn image_describe_refine_job_resolves_asset_and_forwards_caption_style() {
    // epic 8203 / sc-8206: the reference-image → plain-text DESCRIBE flow POSTs `task: "image_describe"`
    // with a project `sourceAssetId` (+ `projectId`), the vision model's repo, and an optional
    // `captionStyle`. The handler resolves the asset to a confined on-disk `imagePath` (same path as the
    // caption flow) and forwards `model` + `captionStyle` verbatim, with no text prompt required.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Describe Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Reference.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();
    let rel_path = asset["file"]["path"]
        .as_str()
        .expect("file path")
        .to_owned();

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({
            "task": "image_describe",
            "sourceAssetId": asset_id,
            "projectId": project_id,
            "model": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
            "captionStyle": "tags"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "prompt_refine");
    assert_eq!(job["payload"]["task"], "image_describe");
    assert_eq!(job["payload"]["captionStyle"], "tags");
    assert!(job["payload"].get("prompt").is_none());
    // Path (not string) comparison: separator-agnostic on Windows (sc-8967, same as the caption test).
    let expected = project_path.join(&rel_path);
    assert_eq!(
        std::path::Path::new(job["payload"]["imagePath"].as_str().unwrap()),
        expected
    );
}

#[tokio::test]
async fn image_describe_refine_job_requires_source_asset_and_project() {
    // sc-8206: the describe task is image-driven like image_caption, so it must reject a request missing
    // the `sourceAssetId` or `projectId`.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({ "task": "image_describe", "projectId": "project-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({ "task": "image_describe", "sourceAssetId": "asset-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn mood_board_refine_job_resolves_multiple_assets_to_image_paths() {
    // epic 8588 / sc-8595: a "mood board" describe POSTs `sourceAssetIds` (plural). The handler resolves
    // EACH id to a confined on-disk path, in order, and forwards them as the worker's `imagePaths` array
    // (no scalar `imagePath`), so the vision model synthesizes ONE prompt from the shared aesthetic.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Mood Board Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let mut asset_ids = Vec::new();
    let mut rel_paths = Vec::new();
    for name in ["First.png", "Second.png"] {
        let (status, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            name,
            "image/png",
            b"png-bytes",
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        asset_ids.push(asset["id"].as_str().expect("asset id").to_owned());
        rel_paths.push(
            asset["file"]["path"]
                .as_str()
                .expect("file path")
                .to_owned(),
        );
    }

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({
            "task": "image_describe",
            "sourceAssetIds": asset_ids,
            "projectId": project_id,
            "model": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["payload"]["task"], "image_describe");
    // The plural array is emitted; the scalar single-image key is NOT.
    assert!(job["payload"].get("imagePath").is_none());
    let paths = job["payload"]["imagePaths"]
        .as_array()
        .expect("imagePaths array");
    assert_eq!(paths.len(), 2, "both references resolved");
    for (path, rel) in paths.iter().zip(rel_paths.iter()) {
        // Path (not string) comparison: separator-agnostic on Windows (sc-8967).
        let expected = project_path.join(rel);
        assert_eq!(std::path::Path::new(path.as_str().unwrap()), expected);
    }
}

#[tokio::test]
async fn mood_board_refine_job_rejects_more_than_the_cap() {
    // sc-8595: the server-side ceiling (MAX_MOOD_BOARD_IMAGES) is authoritative — a board over the cap is
    // rejected with 400 before any asset resolution, so a runaway list cannot exhaust the vision runtime.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let too_many: Vec<String> = (0..(crate::prompts::MAX_MOOD_BOARD_IMAGES + 1))
        .map(|i| format!("asset-{i}"))
        .collect();
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({
            "task": "image_describe",
            "sourceAssetIds": too_many,
            "projectId": "project-1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn image_and_video_job_routes_normalize_payloads() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "projectName": "Project 1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["type"], "image_generate");
    assert_eq!(image_job["projectId"], "project-1");
    assert!(image_job["payload"].get("requestedGpu").is_none());
    assert_eq!(image_job["payload"]["seed"], Value::Null);
    assert_eq!(image_job["payload"]["seeds"].as_array().unwrap().len(), 2);

    let (status, edit_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "edit_image",
            "prompt": "make it dusk",
            "sourceAssetId": "asset-1",
            "seed": 42
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(edit_job["type"], "image_edit");
    assert!(edit_job["payload"].get("seeds").is_none());

    let (status, wide_seed_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": " ",
            "mode": "text_to_image",
            "prompt": "space project id stays Python-compatible",
            "seed": -42
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(wide_seed_job["payload"]["projectId"], " ");
    assert_eq!(wide_seed_job["payload"]["seed"], -42);

    let (status, video_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "mode": "replace_person",
            "prompt": "hero walks through rain",
            "sourceClipAssetId": "asset-video",
            "personTrackId": "track-1",
            "characterId": "character-1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(video_job["type"], "person_replace");
    assert!(video_job["payload"].get("requestedGpu").is_none());

    let (status, integer_duration_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_video",
            "prompt": "integer duration stays an integer",
            "duration": 6
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(integer_duration_job["payload"]["duration"], 6);

    let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["queued"], 5);
}

#[tokio::test]
async fn bernini_video_modes_validate_required_media() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Bernini" }),
    )
    .await;

    // video_to_video without a source clip is rejected.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "video_to_video",
            "prompt": "make it golden hour"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // reference_to_video without reference images is rejected.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_to_video",
            "prompt": "the subject dances"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // reference_video_to_video needs BOTH a source clip and references.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_video_to_video",
            "prompt": "swap the subject",
            "referenceAssetIds": ["ref-1"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Blank reference ids are rejected.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_to_video",
            "prompt": "the subject dances",
            "referenceAssetIds": ["  "]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Reference id lists are bounded before the worker has to encode them.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_to_video",
            "prompt": "the subject dances",
            "referenceAssetIds": ["ref-1", "ref-2", "ref-3", "ref-4", "ref-5", "ref-6", "ref-7", "ref-8", "ref-9"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // A complete video_to_video request creates a base video_generate job that
    // carries the source clip.
    let (status, v2v_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "video_to_video",
            "prompt": "make it golden hour",
            "sourceClipAssetId": "clip-a"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(v2v_job["type"], "video_generate");
    assert_eq!(v2v_job["payload"]["mode"], "video_to_video");
    assert_eq!(v2v_job["payload"]["sourceClipAssetId"], "clip-a");

    // A complete reference_video_to_video request carries both the clip and the refs.
    let (status, rv2v_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_video_to_video",
            "prompt": "swap the subject",
            "sourceClipAssetId": "clip-a",
            "referenceAssetIds": ["ref-1", "ref-2"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(rv2v_job["type"], "video_generate");
    assert_eq!(rv2v_job["payload"]["referenceAssetIds"][0], "ref-1");
    assert_eq!(rv2v_job["payload"]["referenceAssetIds"][1], "ref-2");

    // multi_video_to_video (sc-5425) needs at least two source clips.
    for clips in [json!([]), json!(["clip-a"])] {
        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": "project-1",
                "model": "bernini",
                "mode": "multi_video_to_video",
                "prompt": "blend the takes",
                "sourceClipAssetIds": clips
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // Blank source-clip ids are rejected.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "multi_video_to_video",
            "prompt": "blend the takes",
            "sourceClipAssetIds": ["clip-a", "  "]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Source clip lists are bounded before worker-side video conditioning.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "multi_video_to_video",
            "prompt": "blend the takes",
            "sourceClipAssetIds": ["clip-1", "clip-2", "clip-3", "clip-4", "clip-5", "clip-6", "clip-7", "clip-8", "clip-9"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // A complete multi_video_to_video request carries the clip array.
    let (status, mv2v_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "multi_video_to_video",
            "prompt": "blend the takes",
            "sourceClipAssetIds": ["clip-a", "clip-b"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(mv2v_job["type"], "video_generate");
    assert_eq!(mv2v_job["payload"]["sourceClipAssetIds"][0], "clip-a");
    assert_eq!(mv2v_job["payload"]["sourceClipAssetIds"][1], "clip-b");

    // ads2v (sc-5425) needs a source clip, a reference video, AND >=1 reference image.
    let ads2v_incomplete = [
        json!({ "referenceClipAssetId": "clip-ref", "referenceAssetIds": ["ref-1"] }),
        json!({ "sourceClipAssetId": "clip-src", "referenceAssetIds": ["ref-1"] }),
        json!({ "sourceClipAssetId": "clip-src", "referenceClipAssetId": "clip-ref" }),
    ];
    for extra in ads2v_incomplete {
        let mut body = json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "ads2v",
            "prompt": "drive the edit with the reference clip"
        });
        let object = body.as_object_mut().unwrap();
        for (key, value) in extra.as_object().unwrap() {
            object.insert(key.clone(), value.clone());
        }
        let (status, _) = request(app.clone(), "POST", "/api/v1/video/jobs", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // A complete ads2v request carries the source clip, reference video, and references.
    let (status, ads2v_job) = request(
        app,
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "ads2v",
            "prompt": "drive the edit with the reference clip",
            "sourceClipAssetId": "clip-src",
            "referenceClipAssetId": "clip-ref",
            "referenceAssetIds": ["ref-1"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(ads2v_job["type"], "video_generate");
    assert_eq!(ads2v_job["payload"]["sourceClipAssetId"], "clip-src");
    assert_eq!(ads2v_job["payload"]["referenceClipAssetId"], "clip-ref");
    assert_eq!(ads2v_job["payload"]["referenceAssetIds"][0], "ref-1");
}

#[tokio::test]
async fn person_tracking_routes_match_contracts() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Tracking Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());
    std::fs::write(
        project_path.join("person-tracks/track_1.sceneworks.person-track.json"),
        serde_json::to_string_pretty(&json!({
            "schemaVersion": 1,
            "id": "track_1",
            "projectId": project_id,
            "name": "Hero",
            "createdAt": "2026-05-17T00:00:00Z",
            "sourceAssetId": "asset-video",
            "representativeFrameAssetId": "asset-frame",
            "frames": [],
            "status": {}
        }))
        .expect("json"),
    )
    .expect("track sidecar writes");

    let (status, tracks) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/person-tracks"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(tracks[0]["id"], "track_1");
    assert_eq!(
        tracks[0]["path"],
        "person-tracks/track_1.sceneworks.person-track.json"
    );

    let (status, track) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/person-tracks/track_1"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(track["name"], "Hero");

    let (status, detection_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/person-tracks/detections"),
        json!({ "sourceAssetId": "asset-video", "sourceTimestamp": 1.25 }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(detection_job["type"], "person_detect");
    assert_eq!(detection_job["payload"]["sourceTimestamp"], 1.25);
    assert!(detection_job["projectName"]
        .as_str()
        .is_some_and(|value| value.starts_with("tracking")));

    let detection = json!({
        "id": "person_1",
        "box": { "x": 0.3, "y": 0.2, "width": 0.2, "height": 0.6 }
    });
    let (status, track_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/person-tracks/jobs"),
        json!({
            "sourceAssetId": "asset-video",
            "representativeFrameAssetId": "asset-frame",
            "detection": detection,
            "trackName": "Hero"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(track_job["type"], "person_track");
    assert_eq!(track_job["payload"]["trackName"], "Hero");

    for invalid_path in [
        format!("/api/v1/projects/{project_id}/person-tracks/%2E%2E"),
        format!("/api/v1/projects/{project_id}/person-tracks/%2E%2E%2Fescape"),
        format!("/api/v1/projects/{project_id}/person-tracks/track~bad"),
    ] {
        let (status, error) = request(app.clone(), "GET", &invalid_path, Value::Null).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error["detail"], "Invalid person track ID");
    }

    let (status, queue) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["queued"], 2);
}

#[tokio::test]
async fn generation_job_routes_reject_incompatible_loras() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z-Image",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "edit_image", "character_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": { "families": ["z-image"] },
                  "ui": {}
                },
                {
                  "id": "ltx_2_3",
                  "name": "LTX",
                  "family": "ltx-video",
                  "type": "video",
                  "adapter": "ltx_video",
                  "capabilities": ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": { "families": ["ltx-video"] },
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "qwen_style",
                  "name": "Qwen Style",
                  "family": "qwen-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["qwen-image"] },
                  "source": { "provider": "local", "path": "loras/qwen.safetensors" }
                }
              ]
            }
            "#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "bad_qwen",
                  "name": "Bad Qwen",
                  "workflow": "text_to_image",
                  "model": "z_image_turbo",
                  "loras": [{ "id": "qwen_style" }]
                }
              ]
            }
            "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("qwen.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Compatibility" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let (status, image_error) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "mist",
            "model": "z_image_turbo",
            "loras": [{ "id": "qwen_style" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        image_error["detail"],
        "LoRA qwen_style is not compatible with model z_image_turbo"
    );

    let (status, unknown_model_error) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "mist",
            "model": "missing_model",
            "loras": [{ "id": "qwen_style" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        unknown_model_error["detail"],
        "Model missing_model not found; cannot verify LoRA compatibility"
    );

    let (status, preset_error) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "mist",
            "model": "z_image_turbo",
            "recipePresetId": "bad_qwen"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        preset_error["detail"],
        "LoRA qwen_style is not compatible with model z_image_turbo"
    );

    for (mode, extra) in [
        ("image_to_video", json!({ "sourceAssetId": "asset-image" })),
        ("text_to_video", json!({})),
        (
            "first_last_frame",
            json!({ "sourceAssetId": "asset-first", "lastFrameAssetId": "asset-last" }),
        ),
        ("extend_clip", json!({ "sourceClipAssetId": "asset-video" })),
        (
            "video_bridge",
            json!({ "sourceClipAssetId": "asset-left", "bridgeRightClipAssetId": "asset-right" }),
        ),
        (
            "replace_person",
            json!({ "sourceClipAssetId": "asset-video", "personTrackId": "track-1", "characterId": "character-1" }),
        ),
    ] {
        let mut payload = json!({
            "projectId": project_id,
            "mode": mode,
            "prompt": "motion",
            "model": "ltx_2_3",
            "loras": [{ "id": "qwen_style" }]
        });
        payload
            .as_object_mut()
            .expect("video payload object")
            .extend(extra.as_object().expect("extra payload object").clone());
        let (status, video_error) =
            request(app.clone(), "POST", "/api/v1/video/jobs", payload).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{mode}");
        assert_eq!(
            video_error["detail"],
            "LoRA qwen_style is not compatible with model ltx_2_3"
        );
    }

    let (_, character) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters"),
        json!({ "name": "Mira", "type": "person" }),
    )
    .await;
    let character_id = character["id"].as_str().expect("character id");
    let character_lora = temp_dir
        .path()
        .join("data/loras/character-qwen.safetensors");
    write_test_safetensors(&character_lora);
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/loras"),
        json!({
            "name": "Character Qwen",
            "sourcePath": character_lora.display().to_string(),
            "compatibility": { "families": ["qwen-image"] }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let (status, character_error) = request(
        app,
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/test-jobs"),
        json!({ "prompt": "portrait", "model": "z_image_turbo" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(character_error["detail"]
        .as_str()
        .unwrap()
        .contains("is not compatible with model z_image_turbo"));
}

#[tokio::test]
async fn video_jobs_expand_recipe_presets_server_side() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "vid-model",
              "name": "Vid Model",
              "family": "wan-video",
              "type": "video",
              "adapter": "wan_video",
              "capabilities": ["text_to_video", "image_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["wan-video"] },
              "ui": { "label": "Vid" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "loras": [
            {
              "id": "motion-lora",
              "name": "Motion LoRA",
              "family": "wan-video",
              "triggerWords": ["motion"],
              "compatibility": { "families": ["wan-video"] },
              "source": { "provider": "local", "path": "loras/motion.safetensors" }
            }
          ]
        }
        "#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "dream_motion",
              "name": "Dream Motion",
              "workflow": "text_to_video",
              "model": "vid-model",
              "defaults": { "duration": 8, "fps": 30, "resolution": "1280x720", "quality": "best", "negativePrompt": "jitter" },
              "prompt": { "prefix": "cinematic", "suffix": "smooth camera motion" },
              "loras": [{ "id": "motion-lora", "weight": 0.5 }]
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("motion.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Video Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, video_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "model": "vid-model",
            // Client render settings that DIFFER from the preset's declared
            // defaults — the studio seeds the form from the preset but the user
            // is free to override, so these submitted values must win.
            "duration": 10,
            "fps": 24,
            "width": 640,
            "height": 640,
            "quality": "fast",
            "negativePrompt": "client jitter",
            "recipePresetId": "dream_motion"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    // Prompt prefix/suffix are folded in server-side around the raw client
    // prompt — the regression that motivated this path.
    assert_eq!(
        video_job["payload"]["prompt"],
        "cinematic, a fox runs, smooth camera motion"
    );
    // Render settings are client-owned and overrideable: the submitted values
    // win over the preset's declared defaults (8 / 30 / 1280x720 / best / jitter).
    assert_eq!(video_job["payload"]["duration"], 10);
    assert_eq!(video_job["payload"]["fps"], 24);
    assert_eq!(video_job["payload"]["width"], 640);
    assert_eq!(video_job["payload"]["height"], 640);
    assert_eq!(video_job["payload"]["quality"], "fast");
    assert_eq!(video_job["payload"]["negativePrompt"], "client jitter");
    // Preset LoRA merged in (client sent none) and stamped under advanced.
    assert_eq!(video_job["payload"]["loras"][0]["id"], "motion-lora");
    assert_eq!(
        video_job["payload"]["advanced"]["recipePresetId"],
        "dream_motion"
    );

    // sc-10520: submitting the job stamped lastUsedAt into the usage side store, and it
    // surfaces on the catalog read even though dream_motion is a read-only BUILTIN preset
    // (its own manifest can't be rewritten). The store lives beside the manifests.
    assert!(
        config_dir.join("recipe-preset-usage.json").is_file(),
        "job submit should create the recipe-preset usage store"
    );
    let (status, presets) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/recipe-presets?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let dream = presets
        .as_array()
        .expect("presets list")
        .iter()
        .find(|preset| preset["id"] == "dream_motion")
        .expect("dream_motion present");
    assert_eq!(dream["scope"], "builtin");
    assert!(
        dream["lastUsedAt"]
            .as_str()
            .is_some_and(|value| !value.is_empty()),
        "builtin preset should carry lastUsedAt after use, got {:?}",
        dream["lastUsedAt"]
    );
}

#[tokio::test]
async fn preset_overridden_video_model_carries_its_own_manifest_entry() {
    // sc-12300: a client that OMITS `model` (MCP's submit_video_job documents its `model`
    // param as "Omit for the server default") gets `default_video_model()` from serde, which
    // is exactly the gate `apply_recipe_preset_to_video_payload` uses to let the preset's
    // model win. The preset then overwrites job_payload["model"] — but the manifest entry
    // used to be resolved from the DTO's pre-override `payload.model`, so the job was
    // enqueued carrying the OVERRIDDEN model id alongside the DEFAULT model's entry.
    //
    // Both halves of that mismatch are asserted, because they fail in opposite ways:
    //   - `repo`   — the LOUD failure: the worker reaches for the wrong model's weights.
    //   - `limits.requiresDimensionsMultipleOf` — the SILENT one: sceneworks-core's
    //     normalized_dimensions honors this (sc-11993). A 16-multiple model handed a
    //     32-declaring entry silently renders off-bucket (Mochi's native 848x480 -> 832x480).
    // Pinning only `repo` would let the silent geometry bug regress undetected.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    // `default-vid` is BUILT FROM default_video_model() rather than hardcoding its id. That
    // literal is what the preset-override gate compares against, so a hardcoded fixture would
    // quietly stop modelling *the default's* entry if the default ever changed: the pre-fix
    // failure would degrade from "carries the DEFAULT's entry" to "carries {}" — still red,
    // but no longer demonstrating the documented defect. Its 32 / owner/default mirror the
    // real default video manifest; `preset-vid` mirrors mochi_1's 16 / distinct repo.
    let default_video_model = crate::defaults::default_video_model();
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "__DEFAULT_VIDEO_MODEL__",
              "name": "Default Vid",
              "family": "ltx-video",
              "type": "video",
              "adapter": "ltx_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/default-vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": { "requiresDimensionsMultipleOf": 32 },
              "ui": { "label": "Default Vid" }
            },
            {
              "id": "preset-vid",
              "name": "Preset Vid",
              "family": "mochi",
              "type": "video",
              "adapter": "mochi_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/preset-vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": { "requiresDimensionsMultipleOf": 16 },
              "ui": { "label": "Preset Vid" }
            }
          ]
        }
        "#
        .replace("__DEFAULT_VIDEO_MODEL__", &default_video_model),
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "preset_override",
              "name": "Preset Override",
              "workflow": "text_to_video",
              "model": "preset-vid",
              "defaults": {},
              "prompt": { "prefix": "cinematic", "suffix": "smooth" }
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Preset Override Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, video_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            // `model` deliberately OMITTED — this is the trigger. Sending it explicitly
            // closes the gate and the bug never fires.
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    // The preset's model won over the omitted-model default...
    assert_eq!(video_job["payload"]["model"], "preset-vid");
    // ...and the entry travelling with it must describe THAT model, not the default's.
    let entry = &video_job["payload"]["modelManifestEntry"];
    assert_eq!(
        entry["id"], "preset-vid",
        "manifest entry should be resolved from the post-override model"
    );
    assert_eq!(
        entry["downloads"][0]["repo"], "owner/preset-vid",
        "wrong repo => the worker fetches the wrong model's weights (loud failure)"
    );
    assert_eq!(
        entry["limits"]["requiresDimensionsMultipleOf"], 16,
        "wrong dimension floor => silently renders off-bucket geometry (sc-11993)"
    );
}

#[tokio::test]
async fn preset_overridden_image_model_carries_its_own_manifest_entry() {
    // sc-12300: create_image_job has the identical ordering shape as create_video_job —
    // apply_recipe_preset_to_image_payload may overwrite job_payload["model"] (gated on the
    // omitted-model default_image_model()), and the entry was likewise resolved from the
    // pre-override DTO. Same defect, same function family, so it is pinned the same way.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    // Built FROM default_image_model() — the id the override gate compares against — for the
    // same durability reason as the video fixture above.
    let default_image_model = crate::defaults::default_image_model();
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "__DEFAULT_IMAGE_MODEL__",
              "name": "Default Img",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/default-img", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": { "requiresDimensionsMultipleOf": 32 },
              "ui": { "label": "Default Img" }
            },
            {
              "id": "preset-img",
              "name": "Preset Img",
              "family": "flux",
              "type": "image",
              "adapter": "flux",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/preset-img", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": { "requiresDimensionsMultipleOf": 16 },
              "ui": { "label": "Preset Img" }
            }
          ]
        }
        "#
        .replace("__DEFAULT_IMAGE_MODEL__", &default_image_model),
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "img_override",
              "name": "Img Override",
              "workflow": "text_to_image",
              "model": "preset-img",
              "defaults": {},
              "prompt": { "prefix": "cinematic", "suffix": "smooth" }
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Img Override Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox runs",
            // `model` deliberately OMITTED — the trigger.
            "recipePresetId": "img_override"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["payload"]["model"], "preset-img");
    let entry = &image_job["payload"]["modelManifestEntry"];
    assert_eq!(
        entry["id"], "preset-img",
        "manifest entry should be resolved from the post-override model"
    );
    assert_eq!(
        entry["downloads"][0]["repo"], "owner/preset-img",
        "wrong repo => the worker fetches the wrong model's weights (loud failure)"
    );
    assert_eq!(
        entry["limits"]["requiresDimensionsMultipleOf"], 16,
        "wrong dimension floor => silently renders off-bucket geometry"
    );
}

#[tokio::test]
async fn preset_image_job_builds_each_catalog_once() {
    // sc-8819 (F-017): a preset-backed POST /image/jobs fans out into recipe_preset_catalog,
    // merge_preset_loras_into_payload, and validate_job_lora_compatibility. Before the fix
    // each re-assembled model_catalog/lora_catalog from scratch, re-running the whole
    // per-model/per-LoRA filesystem install-state probe sweep 2× each. The request-scoped
    // JobCatalogSnapshot threads one snapshot through those seams so each catalog is built
    // exactly once per job-create — assert that here.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "img-model",
              "name": "Img Model",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/img", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["z-image"] },
              "ui": { "label": "Img" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "loras": [
            {
              "id": "style-lora",
              "name": "Style LoRA",
              "family": "z-image",
              "triggerWords": ["style"],
              "compatibility": { "families": ["z-image"] },
              "source": { "provider": "local", "path": "loras/style.safetensors" }
            }
          ]
        }
        "#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "dream_style",
              "name": "Dream Style",
              "workflow": "text_to_image",
              "model": "img-model",
              "defaults": { "count": 1, "resolution": "1024x1024", "negativePrompt": "blur" },
              "prompt": { "prefix": "cinematic", "suffix": "high detail" },
              "loras": [{ "id": "style-lora", "weight": 0.5 }]
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("style.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Image Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // Reset the probe counters immediately before the job-create so project setup /
    // seeding above doesn't count against it.
    crate::test_reset_catalog_build_counters();
    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "recipePresetId": "dream_style"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "job created: {image_job:?}");

    // Correctness preserved: preset prompt folded in, preset LoRA merged, and it validated.
    assert_eq!(
        image_job["payload"]["prompt"],
        "cinematic, a fox, high detail"
    );
    assert_eq!(image_job["payload"]["loras"][0]["id"], "style-lora");
    assert_eq!(
        image_job["payload"]["advanced"]["recipePresetId"],
        "dream_style"
    );

    // The heart of sc-8819: each catalog's full FS-probe assembly ran exactly once for the
    // whole request, not 2–3× as before the snapshot was threaded through.
    assert_eq!(
        crate::test_model_catalog_builds(),
        1,
        "model catalog should be assembled once per preset job-create"
    );
    assert_eq!(
        crate::test_lora_catalog_builds(),
        1,
        "lora catalog should be assembled once per preset job-create"
    );
}

#[tokio::test]
async fn preset_image_job_skips_server_lora_merge_when_client_resolved() {
    // The web studio seeds a selected preset's LoRAs into the visible picker and sends them in
    // `loras`, so it — not the server — is authoritative for which preset LoRAs apply. When it
    // also sends presetLorasResolvedClientSide, the server must NOT re-merge the preset's LoRAs:
    // that is what lets a user REMOVE a preset LoRA (send it absent) and have the removal stick,
    // instead of the server silently adding it back. Headless clients that omit the flag keep the
    // server-side merge (covered by preset_image_job_builds_each_catalog_once).
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "img-model",
              "name": "Img Model",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/img", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["z-image"] },
              "ui": { "label": "Img" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "loras": [
            {
              "id": "style-lora",
              "name": "Style LoRA",
              "family": "z-image",
              "triggerWords": ["style"],
              "compatibility": { "families": ["z-image"] },
              "source": { "provider": "local", "path": "loras/style.safetensors" }
            }
          ]
        }
        "#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "dream_style",
              "name": "Dream Style",
              "workflow": "text_to_image",
              "model": "img-model",
              "defaults": { "count": 1, "resolution": "1024x1024" },
              "prompt": { "prefix": "cinematic" },
              "loras": [{ "id": "style-lora", "weight": 0.5 }]
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("style.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Client Resolved Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // The client selected the preset (recipePresetId) but removed its only LoRA in the picker, so
    // it sends an empty `loras` plus the client-resolved flag.
    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "recipePresetId": "dream_style",
            "presetLorasResolvedClientSide": true,
            "loras": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "job created: {image_job:?}");

    // The preset prompt is still folded in and the preset id is stamped, but the server left the
    // client's (empty) `loras` untouched — style-lora was NOT re-added.
    assert_eq!(image_job["payload"]["prompt"], "cinematic, a fox");
    assert_eq!(
        image_job["payload"]["advanced"]["recipePresetId"],
        "dream_style"
    );
    assert_eq!(
        image_job["payload"]["loras"],
        json!([]),
        "client-resolved preset LoRAs must not be re-merged by the server"
    );
}

#[tokio::test]
async fn preset_image_job_skips_server_prompt_fold_when_client_resolved() {
    // General presets stack (epic 11949): the studio composes the full preset-stack prompt
    // client-side because the server only knows how to fold ONE recipePresetId's prefix/suffix.
    // When the studio sends presetPromptResolvedClientSide, the server must take `prompt` verbatim
    // and NOT re-apply preset_prompt — otherwise the base preset's prefix would be folded twice.
    // Headless clients that omit the flag keep the server-side fold (asserted below as baseline).
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "img-model",
              "name": "Img Model",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/img", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["z-image"] },
              "ui": { "label": "Img" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "dream_style",
              "name": "Dream Style",
              "workflow": "text_to_image",
              "model": "img-model",
              "defaults": { "count": 1, "resolution": "1024x1024" },
              "prompt": { "prefix": "cinematic" }
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Client Prompt Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // Baseline: no flag → server folds the preset prefix into the prompt (today's behavior).
    let (status, folded) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "recipePresetId": "dream_style"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "baseline job created: {folded:?}"
    );
    assert_eq!(folded["payload"]["prompt"], "cinematic, a fox");

    // Client-authoritative: the studio already composed "cinematic, a fox" and sends the flag,
    // so the server must NOT fold the prefix again (would yield "cinematic, cinematic, a fox").
    let (status, verbatim) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "cinematic, a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "recipePresetId": "dream_style",
            "presetPromptResolvedClientSide": true
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "client-resolved job created: {verbatim:?}"
    );
    assert_eq!(
        verbatim["payload"]["prompt"], "cinematic, a fox",
        "client-resolved prompt must be taken verbatim, not re-folded"
    );
    // The preset id is still stamped for usage tracking.
    assert_eq!(
        verbatim["payload"]["advanced"]["recipePresetId"],
        "dream_style"
    );
}

#[tokio::test]
async fn generation_routes_reject_invalid_payloads() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({ "projectId": "project-1", "prompt": "x".repeat(4001) }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _) = request(
        app,
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "mode": "image_to_video",
            "prompt": "missing source image"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[test]
fn frame_extract_rejects_non_finite_playhead() {
    let result = crate::validate_frame_extract(&crate::FrameExtractRequest {
        playhead_seconds: f64::NAN,
        intended_use: "reuse".to_owned(),
        requested_gpu: "auto".to_owned(),
    });

    assert!(result.is_err());
}

#[test]
fn image_dimension_cap_covers_sensenova_buckets() {
    // Raised so SenseNova-U1's true trained buckets (largest 3456) pass.
    assert_eq!(crate::MAX_IMAGE_DIMENSION, 4096);
    assert!(crate::validate_dimension(2720, "width", crate::MAX_IMAGE_DIMENSION).is_ok());
    assert!(crate::validate_dimension(3456, "height", crate::MAX_IMAGE_DIMENSION).is_ok());
    assert!(crate::validate_dimension(4096, "width", crate::MAX_IMAGE_DIMENSION).is_ok());
    assert!(crate::validate_dimension(4097, "width", crate::MAX_IMAGE_DIMENSION).is_err());
    assert!(crate::validate_dimension(255, "width", crate::MAX_IMAGE_DIMENSION).is_err());
}

#[test]
fn vqa_job_validation_requires_question_and_asset() {
    let base = crate::VqaJobRequest {
        project_id: "project-1".to_owned(),
        project_name: None,
        source_asset_id: "asset_1".to_owned(),
        question: "What is in this image?".to_owned(),
        model: "sensenova_u1_8b".to_owned(),
        max_new_tokens: 256,
        requested_gpu: "auto".to_owned(),
        advanced: serde_json::Map::new(),
    };
    assert!(crate::validate_vqa_job(&base).is_ok());

    // The UI's length presets are all valid.
    for tokens in [256u32, 512, 1024] {
        let mut request = base.clone();
        request.max_new_tokens = tokens;
        assert!(crate::validate_vqa_job(&request).is_ok());
    }

    let mut blank_question = base.clone();
    blank_question.question = "   ".to_owned();
    assert!(crate::validate_vqa_job(&blank_question).is_err());

    let mut missing_asset = base.clone();
    missing_asset.source_asset_id = String::new();
    assert!(crate::validate_vqa_job(&missing_asset).is_err());

    let mut missing_project = base.clone();
    missing_project.project_id = String::new();
    assert!(crate::validate_vqa_job(&missing_project).is_err());

    let mut too_many_tokens = base.clone();
    too_many_tokens.max_new_tokens = 4096;
    assert!(crate::validate_vqa_job(&too_many_tokens).is_err());
}

#[test]
fn interleave_job_validation_bounds_prompt_images_and_assets() {
    let base = crate::InterleaveJobRequest {
        project_id: "project-1".to_owned(),
        project_name: None,
        prompt: "A short illustrated guide to brewing tea".to_owned(),
        source_asset_ids: Vec::new(),
        model: "sensenova_u1_8b".to_owned(),
        max_images: 6,
        width: 1024,
        height: 1024,
        seed: None,
        requested_gpu: "auto".to_owned(),
        advanced: serde_json::Map::new(),
    };
    assert!(crate::validate_interleave_job(&base).is_ok());

    // Optional input images (it2i) are allowed.
    let mut with_sources = base.clone();
    with_sources.source_asset_ids = vec!["asset_1".to_owned(), "asset_2".to_owned()];
    assert!(crate::validate_interleave_job(&with_sources).is_ok());

    let mut blank_prompt = base.clone();
    blank_prompt.prompt = "   ".to_owned();
    assert!(crate::validate_interleave_job(&blank_prompt).is_err());

    let mut missing_project = base.clone();
    missing_project.project_id = String::new();
    assert!(crate::validate_interleave_job(&missing_project).is_err());

    let mut zero_images = base.clone();
    zero_images.max_images = 0;
    assert!(crate::validate_interleave_job(&zero_images).is_err());

    let mut too_many_images = base.clone();
    too_many_images.max_images = 11;
    assert!(crate::validate_interleave_job(&too_many_images).is_err());

    let mut blank_asset = base.clone();
    blank_asset.source_asset_ids = vec!["  ".to_owned()];
    assert!(crate::validate_interleave_job(&blank_asset).is_err());

    let mut tiny = base.clone();
    tiny.width = 64;
    assert!(crate::validate_interleave_job(&tiny).is_err());
}

#[tokio::test]
async fn worker_heartbeat_interrupts_previous_active_job_through_http() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": null,
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({ "type": "image_detail", "payload": {}, "requestedGpu": "auto" }),
    )
    .await;
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;

    let job_id = created["id"].as_str().expect("job id is string");
    // The owning worker reports at least one heartbeat for the job, so a
    // later idle heartbeat is a genuine restart (not a claim race) and must
    // reclaim the now-orphaned active job.
    request(
        app.clone(),
        "POST",
        "/api/v1/workers/worker-1/heartbeat",
        json!({ "status": "busy", "currentJobId": job_id, "loadedModels": [] }),
    )
    .await;

    let (status, worker) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/worker-1/heartbeat",
        json!({ "status": "idle", "currentJobId": null, "loadedModels": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(worker["currentJobId"], Value::Null);

    let (status, job) = request(app, "GET", &format!("/api/v1/jobs/{job_id}"), Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(job["status"], "interrupted");
    assert_eq!(job["workerId"], Value::Null);
}

#[tokio::test]
async fn stale_sweep_broadcasts_job_updated_for_interrupted_jobs() {
    // sc-8186: the heartbeat stale-sweep marks an in-flight job `interrupted` in the DB, but
    // (unlike a worker-reported terminal status) emitted no per-job event — so a live client's
    // job card, driven by `job.updated`, showed its last running state forever. The sweep must
    // now broadcast `job.updated` for each job it interrupts.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    // Smallest timeout the store honors (clamped to >=1s); we sleep just past it to go stale.
    settings.worker_timeout_seconds = 1;
    let (app, state) = create_app_with_state(settings).expect("app creates");

    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": null,
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({ "type": "image_detail", "payload": {}, "requestedGpu": "auto" }),
    )
    .await;
    let job_id = created["id"].as_str().expect("job id is string").to_owned();
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;

    // Let the worker's last_seen age past the (1s) timeout so the next sweep interrupts its job,
    // then subscribe so we only observe the sweep's events. last_seen is stored at second
    // granularity and the cutoff is `now - 1s`, so we sleep just over 2s to clear the boundary.
    tokio::time::sleep(Duration::from_millis(2_100)).await;
    let mut events = state.events.subscribe();

    // Any endpoint that runs `queue_summary_snapshot` triggers the sweep; GET /queue is the
    // canonical one.
    let (status, _) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);

    let sweep_events = drain_event_names(&mut events).await;
    assert!(
        sweep_events.iter().any(|name| name == "job.updated"),
        "the stale-sweep must broadcast job.updated for the interrupted job: {sweep_events:?}"
    );

    let (status, job) = request(app, "GET", &format!("/api/v1/jobs/{job_id}"), Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(job["status"], "interrupted");
}

#[tokio::test]
async fn claim_sweeps_stale_jobs_once_and_still_refreshes_the_queue() {
    // sc-8889 / F-087: claim_job runs mark_stale_workers_interrupted in its own
    // transaction, then refreshes the queue via publish_queue_skip_sweep — which
    // no longer sweeps a SECOND time. This pins that dropping the duplicate sweep
    // did not regress the claim path: a claim still reaps a stale worker's job to
    // `interrupted` and still emits queue.updated.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.worker_timeout_seconds = 1;
    let (app, state) = create_app_with_state(settings).expect("app creates");

    // worker-1 claims a job, then goes stale.
    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": null,
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({ "type": "image_detail", "payload": {}, "requestedGpu": "auto" }),
    )
    .await;
    let stale_job_id = created["id"].as_str().expect("job id is string").to_owned();
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;

    // A fresh worker registers plus a second queued job so worker-2's claim
    // actually returns work (response.is_some -> the queue refresh fires). Age
    // worker-1 past the 1s timeout so the next claim's sweep reaps its job.
    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-2",
            "gpuId": "gpu-1",
            "gpuName": null,
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({ "type": "image_detail", "payload": {}, "requestedGpu": "auto" }),
    )
    .await;
    tokio::time::sleep(Duration::from_millis(2_100)).await;
    let mut events = state.events.subscribe();

    // worker-2 claims the second job. claim_job runs its own stale sweep
    // (interrupting worker-1's job) and refreshes the queue via the skip-sweep
    // path — without sweeping a second time.
    let (status, claim) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-2" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        claim["job"].is_object(),
        "worker-2 claims the second queued job: {claim}"
    );

    let claim_events = drain_event_names(&mut events).await;
    assert!(
        claim_events.iter().any(|name| name == "queue.updated"),
        "a claim that returns work still refreshes the queue: {claim_events:?}"
    );

    // The stale job was reaped exactly by the claim's own single sweep.
    let (status, job) = request(
        app,
        "GET",
        &format!("/api/v1/jobs/{stale_job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(job["status"], "interrupted");
    assert_eq!(job["workerId"], Value::Null);
}

/// sc-12297: `limits.hardMaxDuration` is enforced at enqueue, and — the part that makes WHERE it
/// lives load-bearing — against the POST-PRESET model's cap.
///
/// The fixture is built so the two plausible homes for this check disagree:
///   * default video model — cap 15 (generous)
///   * `preset-vid`        — cap  5 (strict)
///
/// The request omits `model` (so the preset's model wins, per sc-12300) and asks for 10s. Gating
/// on the DTO's `payload.model` — i.e. inside `validate_video_job`, the intuitive home, which runs
/// BEFORE `apply_recipe_preset_to_video_payload` — reads the DEFAULT's cap of 15, admits 10s, and
/// enqueues a job the strict model can't render. Only a gate placed after preset expansion AND
/// manifest resolution sees the 5 that actually applies. That is the whole reason this check is not
/// in `validate_video_job`, and this test is what pins it there.
#[tokio::test]
async fn video_duration_past_the_post_preset_models_hard_cap_is_rejected() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let default_video_model = crate::defaults::default_video_model();
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "__DEFAULT_VIDEO_MODEL__",
              "name": "Default Vid",
              "family": "ltx-video",
              "type": "video",
              "adapter": "ltx_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/default-vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": { "hardMaxDuration": 15 },
              "ui": { "label": "Default Vid" }
            },
            {
              "id": "preset-vid",
              "name": "Preset Vid",
              "family": "mochi",
              "type": "video",
              "adapter": "mochi_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/preset-vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": { "hardMaxDuration": 5 },
              "ui": { "label": "Preset Vid" }
            }
          ]
        }
        "#
        .replace("__DEFAULT_VIDEO_MODEL__", &default_video_model),
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "preset_override",
              "name": "Preset Override",
              "workflow": "text_to_video",
              "model": "preset-vid",
              "defaults": {},
              "prompt": { "prefix": "cinematic", "suffix": "smooth" }
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Duration Cap Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // 10s: legal for the default (15) but past the preset-resolved model's 5. `model` omitted so
    // the preset's model wins — the sc-12300 shape.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "duration": 10,
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "10s past preset-vid's 5s cap must be refused at enqueue, not silently clamped: {body}"
    );
    let detail = body["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains("preset-vid"),
        "names the model whose cap applied — NOT the default's: {detail}"
    );
    assert!(detail.contains("5s"), "states the cap: {detail}");
    assert!(detail.contains("10s"), "states what was asked: {detail}");

    // At-cap admits: 5s is exactly the cap, and the bound is `>`. This is what keeps the
    // assertion above from passing for a gate that simply rejects everything.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "duration": 5,
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "5s is at the cap: {body}");
    assert_eq!(body["payload"]["model"], "preset-vid");

    // ...and the SAME 10s request against the default model (cap 15) is admitted, proving the
    // rejection above came from the per-model cap rather than a blanket duration bound.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "duration": 10,
            "model": default_video_model
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "10s is within the default model's 15s cap: {body}"
    );
}

/// sc-12347: `limits.fps` is enforced at enqueue against the POST-PRESET model's menu, and an
/// OMITTED fps resolves to that model's declared `defaults.fps` rather than a blanket.
///
/// The fixture makes both halves load-bearing by having the two models' menus disagree:
///   * default video model — `[24, 25, 30]`, default 25 (permissive)
///   * `preset-vid`        — `[30]`, default 30 (strict, the mochi_1 shape)
///
/// `fps: 25` is the discriminator. It is on the default's menu and off `preset-vid`'s, so a gate
/// reading the DTO's stale `payload.model` — i.e. inside `validate_video_job`, before
/// `apply_recipe_preset_to_video_payload` — admits it and enqueues a job the strict model does not
/// advertise. 25 is also the blanket the DTO used to default to, which is why the omitted-fps case
/// below is the regression this story nearly shipped rather than a nicety.
#[tokio::test]
async fn video_fps_outside_the_post_preset_models_menu_is_rejected() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let default_video_model = crate::defaults::default_video_model();
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "__DEFAULT_VIDEO_MODEL__",
              "name": "Default Vid",
              "family": "ltx-video",
              "type": "video",
              "adapter": "ltx_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/default-vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": { "fps": 25 },
              "limits": { "fps": [24, 25, 30] },
              "ui": { "label": "Default Vid" }
            },
            {
              "id": "preset-vid",
              "name": "Preset Vid",
              "family": "mochi",
              "type": "video",
              "adapter": "mochi_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/preset-vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": { "fps": 30 },
              "limits": { "fps": [30] },
              "ui": { "label": "Preset Vid" }
            }
          ]
        }
        "#
        .replace("__DEFAULT_VIDEO_MODEL__", &default_video_model),
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "preset_override",
              "name": "Preset Override",
              "workflow": "text_to_video",
              "model": "preset-vid",
              "defaults": {},
              "prompt": { "prefix": "cinematic", "suffix": "smooth" }
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Fps Menu Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // 25fps: on the DEFAULT's menu, off the preset-resolved model's. `model` omitted so the preset's
    // model wins — the sc-12300 shape. A gate reading the stale DTO model admits this.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "fps": 25,
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "25fps is off preset-vid's [30] menu and must be refused at enqueue, not snapped: {body}"
    );
    let detail = body["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains("preset-vid"),
        "names the model whose menu applied — NOT the default's: {detail}"
    );
    assert!(
        detail.contains("30 fps"),
        "states what is allowed: {detail}"
    );
    assert!(detail.contains("25 fps"), "states what was asked: {detail}");

    // THE REGRESSION THIS STORY NEARLY SHIPPED: omitting fps must be ADMITTED, and must resolve to
    // the post-preset model's declared 30 — not the blanket 25 (which the assertion above proves is
    // rejected), and not the DEFAULT model's 25. Both wrong answers are 25, so this pins the
    // resolution AND that it is keyed off the post-preset model.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "a request naming no fps must not be refused by the menu: {body}"
    );
    assert_eq!(body["payload"]["model"], "preset-vid");
    assert_eq!(
        body["payload"]["fps"], 30,
        "the enqueued payload records the model's declared rate, not the blanket 25: {body}"
    );

    // An advertised rate admits — keeps the rejection above from passing for a gate that refuses
    // everything.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "fps": 30,
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "30 is what preset-vid advertises: {body}"
    );
    assert_eq!(body["payload"]["fps"], 30);

    // ...and the SAME 25fps request against the default model IS admitted, proving the rejection
    // came from the per-model menu rather than a blanket fps bound.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "fps": 25,
            "model": default_video_model
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "25 is on the default model's [24, 25, 30] menu: {body}"
    );
    assert_eq!(body["payload"]["fps"], 25);

    // The blanket payload-sanity bound still applies to a NAMED fps, and still comes from
    // `validate_video_job` — a different message than the per-model menu's.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "fps": 240,
            "model": default_video_model
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "240 is past the sanity bound: {body}"
    );
    assert!(body["detail"]
        .as_str()
        .unwrap_or_default()
        .contains("fps must be between 1 and 60"));
}

/// sc-12400 — the regression sc-12297 shipped: a request that names **no duration at all** was
/// rejected for "asking for 6s", on 7 of the 10 shipped video models.
///
/// Against the REAL manifest, because the whole bug is that the DTO's blanket 6.0 was never compared
/// to the caps sc-12297 began enforcing — a fixture would let me pick a cap that hides it.
///
/// This is the exact payload the MCP `generate_video` tool sends when its caller names only a model
/// and a prompt (`server.rs` omits every `None` optional), so it is the natural non-UI call, not an
/// edge case. Before the fix: `400 mochi_1 renders clips up to 5s, but this request asks for 6s` —
/// naming a value the caller never set, with a lever for a field they never touched.
#[tokio::test]
async fn a_video_request_naming_no_duration_is_admitted_at_the_models_own_default() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    sceneworks_core::builtin_manifests::seed_builtin_manifests(
        &settings.config_dir,
        sceneworks_core::builtin_manifests::SeedMode::Overwrite,
    )
    .expect("builtin manifests seed");

    let app = create_app(settings).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Duration Default Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // Every shipped video model whose cap is UNDER the old blanket 6.0 — i.e. every model this
    // regression bricked — plus ltx_2_3, whose cap of 15 admitted 6.0 and which therefore stayed
    // green throughout. Listing the survivor alongside the victims is what makes this a per-model
    // assertion rather than "the route works".
    for (model, want_duration) in [
        ("bernini", 5.0),
        ("scail2_14b", 5.0),
        ("wan_2_2_t2v_14b", 5.0),
        ("wan_2_2_i2v_14b", 5.0),
        ("wan_2_2_vace_fun_14b", 5.0),
        ("svd", 4.0),
        ("ltx_2_3", 6.0),
    ] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": project_id,
                "mode": "text_to_video",
                "prompt": "a fox runs",
                "model": model
            }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::CREATED,
            "{model}: a request naming no duration must be admitted, not 400'd for a value it \
             never set: {body}"
        );
        assert_eq!(
            body["payload"]["duration"], want_duration,
            "{model}: the enqueued payload records the model's declared defaults.duration, not the \
             blanket 6.0: {body}"
        );
    }

    // A NAMED over-cap duration is still refused — the cap is intact; only the phantom 6.0 is gone.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "model": "bernini",
            "duration": 30
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "sc-12297's cap must still refuse a duration the caller actually asked for: {body}"
    );
    assert!(body["detail"]
        .as_str()
        .unwrap_or_default()
        .contains("asks for 30s"));

    // The GEOMETRY axis of the same bug — the third dead `defaults.*` key. No 400 to observe here
    // (dimensions coerce), so the observable is the enqueued size itself: `bernini` must take its
    // declared 848x480 native bucket (stride 16), NOT the blanket 768x512 it never advertises.
    // `wan_2_2_t2v_14b` (1280x720) and `ltx_2_3` (768x512) come along to prove this reads the
    // per-model value rather than one hardcoded pair.
    for (model, want_w, want_h) in [
        ("bernini", 848, 480),
        // True 720p since sc-12308 (#1581) lifted the A14B area cap to its real 921,600.
        ("wan_2_2_t2v_14b", 1280, 720),
        ("ltx_2_3", 768, 512),
    ] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": project_id,
                "mode": "text_to_video",
                "prompt": "a fox runs",
                "model": model
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{model}: {body}");
        assert_eq!(
            (
                body["payload"]["width"].as_u64(),
                body["payload"]["height"].as_u64()
            ),
            (Some(want_w), Some(want_h)),
            "{model}: a size-less request must enqueue the model's declared defaults.resolution: \
             {body}"
        );
    }

    // A NAMED size is still honored verbatim — resolution fills a gap, it does not override.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "model": "bernini",
            "width": 640,
            "height": 384
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(
        (
            body["payload"]["width"].as_u64(),
            body["payload"]["height"].as_u64()
        ),
        (Some(640), Some(384)),
        "a caller's own size must not be replaced by the model's default: {body}"
    );
}

/// sc-12400 (image half): a request that names no size renders the MODEL's declared
/// `defaults.resolution`, not a blanket 1024 square.
///
/// The image twin of `a_video_request_naming_no_duration_is_admitted_at_the_models_own_default`,
/// and against the REAL manifest for the same reason: the bug is precisely that the DTO's blanket
/// was never compared to what each model declares, so a fixture would let me pick the values that
/// hide it.
///
/// No 400 to observe on this axis — geometry coerces — so the observable IS the enqueued size. The
/// expectations are transcribed from the manifest, never derived from `default_resolution`: routing
/// them through the function under test is what made the video half's first cut a tautology that
/// survived deleting the feature.
#[tokio::test]
async fn an_image_request_naming_no_size_renders_the_models_own_declared_resolution() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    sceneworks_core::builtin_manifests::seed_builtin_manifests(
        &settings.config_dir,
        sceneworks_core::builtin_manifests::SeedMode::Overwrite,
    )
    .expect("builtin manifests seed");

    let app = create_app(settings).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Image Default Resolution Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // A matrix that pins each axis INDEPENDENTLY. `defaults.resolution` and `defaults.count` are
    // read by separate code, so a fixture where the two always agree cannot tell a working pair
    // from one reader quietly carrying the other.
    //
    // Every id here is verified to exist. The first cut used `flux1_schnell`, which is NOT a model
    // (it is `flux_schnell`): it resolved to `{}`, took the unknown-model blanket, and passed —
    // green, while testing nothing about a declared default. **A typo'd id is a silent no-op in
    // this test by construction**, which is exactly why the control row must be a REAL model whose
    // declarations happen to equal the blankets.
    for (model, want_w, want_h, want_count) in [
        // BOTH axes discriminate. 2048x2048 + count 1: the blankets rendered this at HALF
        // resolution and FOUR times over, on the text/infographic family where pixels are the
        // entire point — and honoring only the size made the bare call 4x MORE expensive.
        ("sensenova_u1_8b", 2048, 2048, 1),
        ("sensenova_u1_8b_fast", 2048, 2048, 1),
        // COUNT discriminates, resolution coincides — red if the count reader is dropped, green if
        // only the size reader works.
        ("z_image", 1024, 1024, 1),
        // RESOLUTION discriminates, count coincides — the mirror image.
        ("chroma1_flash", 768, 768, 4),
        // NEITHER discriminates: a real model whose declarations equal both blankets. Keeps the
        // rows above from passing for a reader that returns some other model's values.
        ("z_image_turbo", 1024, 1024, 4),
    ] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": project_id,
                "mode": "text_to_image",
                "prompt": "a fox",
                "model": model
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{model}: {body}");
        assert_eq!(
            (
                body["payload"]["width"].as_u64(),
                body["payload"]["height"].as_u64()
            ),
            (Some(want_w), Some(want_h)),
            "{model}: a size-less request must enqueue the model's declared defaults.resolution: \
             {body}"
        );
        assert_eq!(
            body["payload"]["count"].as_u64(),
            Some(want_count),
            "{model}: a count-less request must enqueue the model's declared defaults.count, not \
             the blanket 4: {body}"
        );
        // The seed batch is generated from the RESOLVED count, so it must agree — otherwise a
        // count-1 model would still carry four seeds and the payload would contradict itself.
        assert_eq!(
            body["payload"]["seeds"].as_array().map(Vec::len),
            Some(want_count as usize),
            "{model}: one seed per image actually rendered: {body}"
        );
    }

    // A NAMED size is honored verbatim — resolution fills a gap, it never overrides.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "sensenova_u1_8b",
            "width": 512,
            "height": 512
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(
        (
            body["payload"]["width"].as_u64(),
            body["payload"]["height"].as_u64()
        ),
        (Some(512), Some(512)),
        "a caller's own size must not be replaced by the model's default: {body}"
    );
}

// --- sc-13134: server-side Style fold (headless/MCP parity with the web composer) -----------

/// Write the minimal builtin manifests a Style-fold image job needs into `<config>/manifests`:
/// a single image model, empty user models, and a small Style catalog carrying one group with one
/// sub-style. Returns nothing — the caller builds the app from the same temp dir.
fn write_style_test_manifests(config_dir: &std::path::Path) {
    let manifests = config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).expect("manifest dir creates");
    std::fs::write(
        manifests.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "img-model",
              "name": "Img Model",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/img", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["z-image"] },
              "ui": { "label": "Img" }
            },
            {
              "id": "tag-model",
              "name": "Tag Model",
              "family": "anima",
              "type": "image",
              "adapter": "anima",
              "captionStyle": "tags",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/tag", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["anima"] },
              "ui": { "label": "Tag" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        manifests.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    // A tiny Style catalog shaped exactly like the shipped builtin.styles.jsonc: one group
    // (its `description` is the group's "overall" style text) with one sub-style (`prompt`).
    std::fs::write(
        manifests.join("builtin.styles.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "source": "documents/style.txt",
          "promptTemplate": "Subject: {subject}\nStyle: {style}",
          "groups": [
            {
              "id": "test-anime",
              "name": "Test Anime",
              "description": "broad test anime look",
              "styles": [
                { "id": "test-ghibli", "name": "Test Ghibli", "prompt": "gentle hand-painted" }
              ]
            }
          ]
        }
        "#,
    )
    .expect("builtin styles writes");
}

#[tokio::test]
async fn styled_image_job_folds_style_server_side_from_style_id() {
    // Headless/MCP parity (sc-13134): a client that sends a `styleId` + a RAW prompt gets the
    // exact `Subject:`/`Style:` composition the web composer produces — including the
    // directive-collision splice (a user `Setting:` line stays a top-level sibling, the free text
    // folds into Subject). This is the whole point of the story: the same styled prompt whether
    // the fold happens on the web or on the server.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_style_test_manifests(&temp_dir.path().join("config"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Styled Job Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // Sub-style id → its `prompt` ("gentle hand-painted"); "a fox" becomes the leading Subject and
    // the user's `Setting:` directive is kept as a trailing sibling — byte-identical to
    // composeStyledPrompt.
    let (status, styled) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox\nSetting: snowy field",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "styleId": "test-ghibli"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "styled job created: {styled:?}"
    );
    assert_eq!(
        styled["payload"]["prompt"],
        "Subject: a fox\nStyle: gentle hand-painted\nSetting: snowy field",
        "server-side fold must match the web composer output"
    );

    // A GROUP id resolves to that group's `description` (the "overall" style), sc-13171 parity.
    let (status, group_styled) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "styleId": "test-anime"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{group_styled:?}");
    assert_eq!(
        group_styled["payload"]["prompt"],
        "Subject: a fox\nStyle: broad test anime look"
    );

    // An unknown styleId is a clean 400, not a silent no-op.
    let (status, err) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "styleId": "no-such-style"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{err:?}");
}

#[tokio::test]
async fn styled_image_job_rejects_a_style_on_a_tag_convention_model() {
    // Booru-tag models (`captionStyle: "tags"`) take comma-separated tags, so the prose Style catalog
    // does not apply. The studio hides the axis; a headless/MCP caller that sends a styleId anyway
    // must get a named 400 rather than a silently prose-wrapped prompt.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_style_test_manifests(&temp_dir.path().join("config"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Tag Model Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let body = json!({
        "projectId": project_id,
        "mode": "text_to_image",
        "prompt": "1girl, solo, pink hair",
        "model": "tag-model",
        "count": 1,
        "width": 1024,
        "height": 1024,
        "styleId": "test-ghibli"
    });
    let (status, err) = request(app.clone(), "POST", "/api/v1/image/jobs", body.clone()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{err:?}");
    let message = err["detail"].as_str().unwrap_or_default().to_owned();
    assert!(
        message.contains("tag-model") && message.contains("tag-style"),
        "the rejection must name the model and the reason: {message}"
    );

    // The gate is a MODEL capability, not a fold detail: it fires even when the caller claims the
    // prompt was already composed client-side (the flag that otherwise short-circuits the fold).
    let mut claimed = body;
    claimed["presetPromptResolvedClientSide"] = json!(true);
    let (status, err) = request(app.clone(), "POST", "/api/v1/image/jobs", claimed).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{err:?}");

    // Same model with NO styleId is untouched — the gate rejects the axis, not the model.
    let (status, plain) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project["id"].as_str().expect("project id"),
            "mode": "text_to_image",
            "prompt": "1girl, solo, pink hair",
            "model": "tag-model",
            "count": 1,
            "width": 1024,
            "height": 1024
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{plain:?}");
    assert_eq!(plain["payload"]["prompt"], "1girl, solo, pink hair");
}

#[tokio::test]
async fn styled_image_job_skips_fold_when_prompt_resolved_client_side() {
    // The web app composes the styled prompt CLIENT-side and sends it verbatim plus
    // presetPromptResolvedClientSide (mirroring the preset skip). The server must take the prompt
    // as-is and NOT re-fold — even when a `styleId` rides along (the studio records it for replay).
    // Otherwise a web-submitted "Subject: …\nStyle: …" would be double-wrapped.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_style_test_manifests(&temp_dir.path().join("config"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Client Styled Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let already_composed = "Subject: a fox\nStyle: gentle hand-painted";
    let (status, verbatim) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": already_composed,
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            // styleId-with-flag: the web records the picked id but has already composed the prompt.
            "styleId": "test-ghibli",
            "presetPromptResolvedClientSide": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{verbatim:?}");
    assert_eq!(
        verbatim["payload"]["prompt"], already_composed,
        "a client-composed styled prompt must be taken verbatim, never double-folded"
    );
}

#[tokio::test]
async fn styles_endpoint_serves_the_builtin_catalog() {
    // GET /api/v1/styles gives headless/MCP clients the same catalog the web reads: the grouped
    // Style picker data they need to choose a styleId.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_style_test_manifests(&temp_dir.path().join("config"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, catalog) = request(app.clone(), "GET", "/api/v1/styles", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{catalog:?}");
    assert_eq!(catalog["schemaVersion"], 1);
    assert_eq!(catalog["groups"][0]["id"], "test-anime");
    assert_eq!(catalog["groups"][0]["styles"][0]["id"], "test-ghibli");
    assert_eq!(
        catalog["groups"][0]["styles"][0]["prompt"],
        "gentle hand-painted"
    );
}
