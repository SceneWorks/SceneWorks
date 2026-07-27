//! Builtin model / LoRA / recipe-preset catalogs the app reads from
//! `config_dir/manifests`.
//!
//! The repo's `config/` provides these for the server stack (Compose
//! bind-mounts it) and for a repo checkout, but the desktop wrapper and a
//! directly-launched `sceneworks-rust-api` binary have no such directory — and
//! without them Model Manager is empty and model->file resolution for the native
//! adapters breaks. Embed the canonical repo copies at compile time so a
//! populated catalog can be made an invariant regardless of how the app is
//! launched. Both launchers (`apps/desktop`, `apps/rust-api`) seed from this one
//! source instead of carrying their own copy.
//!
//! NOTE: the `include_str!`s below resolve `config/manifests/*.jsonc` relative to
//! the workspace root, so any build that compiles `sceneworks-core` must have
//! that directory present. The desktop and a plain checkout always do; the
//! `docker/rust.Dockerfile` builder stage `COPY config`s it in for this reason.

use std::path::Path;
use std::sync::OnceLock;

/// `(file name, embedded contents)` for each builtin manifest, embedded at
/// compile time from the canonical repo copies under `config/manifests/`.
pub const BUILTIN_MANIFESTS: &[(&str, &str)] = &[
    (
        "builtin.models.jsonc",
        include_str!("../../../config/manifests/builtin.models.jsonc"),
    ),
    (
        "builtin.loras.jsonc",
        include_str!("../../../config/manifests/builtin.loras.jsonc"),
    ),
    (
        "builtin.recipe-presets.jsonc",
        include_str!("../../../config/manifests/builtin.recipe-presets.jsonc"),
    ),
    (
        // The Style catalog served at GET /api/v1/styles and folded server-side into a prompt
        // carrying a styleId (sc-13134). A mechanical derivation of documents/style.txt — never
        // hand-edited; regenerate via `npm run gen:styles` (apps/web).
        "builtin.styles.jsonc",
        include_str!("../../../config/manifests/builtin.styles.jsonc"),
    ),
    (
        "builtin.control_overlays.jsonc",
        include_str!("../../../config/manifests/builtin.control_overlays.jsonc"),
    ),
];

/// The immutable Hugging Face location of the Chatterbox voice-encoder weights.
///
/// This is derived from the shipped model manifest so downloads and runtime resolution have one pin
/// authority. The standalone `chatterbox_ve` declaration is authoritative; the helper also rejects a
/// catalog where `chatterbox_tts`'s primary or `voice_embedding` co-requisite has drifted away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatterboxVePin {
    pub repo: String,
    pub revision: String,
}

static CHATTERBOX_VE_PIN: OnceLock<Result<ChatterboxVePin, String>> = OnceLock::new();

/// Resolve and validate the Chatterbox voice-encoder pin in the embedded builtin catalog.
pub fn chatterbox_ve_pin() -> Result<&'static ChatterboxVePin, &'static str> {
    CHATTERBOX_VE_PIN
        .get_or_init(|| {
            let contents = BUILTIN_MANIFESTS
                .iter()
                .find(|(name, _)| *name == "builtin.models.jsonc")
                .map(|(_, contents)| *contents)
                .ok_or_else(|| "builtin.models.jsonc is not embedded".to_owned())?;
            parse_chatterbox_ve_pin(contents)
        })
        .as_ref()
        .map_err(String::as_str)
}

fn parse_chatterbox_ve_pin(contents: &str) -> Result<ChatterboxVePin, String> {
    fn downloads<'a>(
        model: &'a serde_json::Value,
        id: &str,
    ) -> Result<&'a Vec<serde_json::Value>, String> {
        model["downloads"]
            .as_array()
            .ok_or_else(|| format!("{id} has no downloads array"))
    }

    fn unique_download<'a>(
        candidates: Vec<&'a serde_json::Value>,
        description: &str,
    ) -> Result<&'a serde_json::Value, String> {
        match candidates.as_slice() {
            [download] => Ok(*download),
            [] => Err(format!("missing {description} download")),
            _ => Err(format!("multiple {description} downloads are ambiguous")),
        }
    }

    let stripped = crate::jsonc::strip_jsonc_comments(contents);
    let manifest: serde_json::Value = serde_json::from_str(&stripped)
        .map_err(|error| format!("builtin.models.jsonc is malformed: {error}"))?;
    let models = manifest["models"]
        .as_array()
        .ok_or_else(|| "builtin.models.jsonc has no models array".to_owned())?;

    let unique_model = |id: &str| -> Result<&serde_json::Value, String> {
        let matches = models
            .iter()
            .filter(|model| model["id"].as_str() == Some(id))
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [model] => Ok(*model),
            [] => Err(format!("builtin.models.jsonc is missing model {id}")),
            _ => Err(format!(
                "builtin.models.jsonc contains multiple models named {id}"
            )),
        }
    };
    let ve = unique_model("chatterbox_ve")?;
    let ve_download = unique_download(
        downloads(ve, "chatterbox_ve")?
            .iter()
            .filter(|download| {
                download["provider"].as_str() == Some("huggingface")
                    && download["files"]
                        .as_array()
                        .is_some_and(|files| files.iter().any(|file| file == "ve.safetensors"))
            })
            .collect(),
        "chatterbox_ve ve.safetensors",
    )?;
    let repo = ve_download["repo"]
        .as_str()
        .ok_or_else(|| "chatterbox_ve ve.safetensors download has no repo".to_owned())?;
    let revision = ve_download["revision"]
        .as_str()
        .ok_or_else(|| "chatterbox_ve ve.safetensors download has no revision".to_owned())?;
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "chatterbox_ve revision must be a full 40-character lowercase-hex commit SHA, got {revision:?}"
        ));
    }

    let tts = unique_model("chatterbox_tts")?;
    let tts_downloads = downloads(tts, "chatterbox_tts")?;
    let tts_primary = unique_download(
        tts_downloads
            .iter()
            .filter(|download| {
                download["provider"].as_str() == Some("huggingface")
                    && download["coRequisite"].as_bool() != Some(true)
            })
            .collect(),
        "chatterbox_tts primary",
    )?;
    let tts_voice_embedding = unique_download(
        tts_downloads
            .iter()
            .filter(|download| {
                download["provider"].as_str() == Some("huggingface")
                    && download["coRequisite"].as_bool() == Some(true)
                    && download["componentId"].as_str() == Some("voice_embedding")
                    && download["files"]
                        .as_array()
                        .is_some_and(|files| files.iter().any(|file| file == "ve.safetensors"))
            })
            .collect(),
        "chatterbox_tts voice_embedding co-requisite",
    )?;
    for (description, download) in [
        ("chatterbox_tts primary", tts_primary),
        (
            "chatterbox_tts voice_embedding co-requisite",
            tts_voice_embedding,
        ),
    ] {
        if download["repo"].as_str() != Some(repo)
            || download["revision"].as_str() != Some(revision)
        {
            return Err(format!(
                "{description} must share chatterbox_ve's repo {repo:?} and revision {revision:?}"
            ));
        }
    }

    Ok(ChatterboxVePin {
        repo: repo.to_owned(),
        revision: revision.to_owned(),
    })
}

/// How an existing manifest file is treated when seeding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedMode {
    /// Overwrite an existing manifest. The desktop seeds this way on every launch
    /// so the builtin catalog tracks the app version; user customizations live in
    /// the separate `user.*.jsonc` files, which seeding never touches.
    Overwrite,
    /// Only write a manifest that is missing. The API seeds this way when its config
    /// dir is EXPLICIT (`SCENEWORKS_CONFIG_DIR` set — a repo checkout or a Compose bind
    /// mount) so that dir stays authoritative: it fills gaps but never clobbers a copy
    /// the operator is editing (and never dirties a checked-out `config/`). When the API
    /// falls back to the platform-default app-owned dir it seeds `Overwrite` instead, so a
    /// directly-launched binary refreshes its builtin catalog on launch rather than serving
    /// a stale seed after an upgrade (sc-10212; see `seed_mode_for_config_dir` in rust-api).
    IfMissing,
}

/// Write the builtin manifests into `config_dir/manifests` according to `mode`.
///
/// Each file is written through [`store_util::atomic_write`], the house
/// atomic-write primitive: it stages into a uniquely-named temp in the same
/// directory, `sync_all`s the temp (and best-effort the parent dir) so the bytes
/// are durable *before* the rename, then renames into place. That closes the two
/// windows a bare temp+rename left open — a power loss after the rename leaving a
/// zero-length `builtin.*.jsonc` (sc-8949), and two processes seeding concurrently
/// colliding on a shared deterministic temp name (sc-1633). A crash therefore
/// cannot leave a truncated manifest that parses to an empty/broken catalog and
/// then gets skipped by a later `IfMissing` seeding.
///
/// Returns an error — annotated with which manifest failed — if any required
/// manifest can't be installed, so callers can abort startup rather than serving
/// an empty catalog.
pub fn seed_builtin_manifests(config_dir: &Path, mode: SeedMode) -> std::io::Result<()> {
    let dir = config_dir.join("manifests");
    std::fs::create_dir_all(&dir).map_err(|error| {
        std::io::Error::new(error.kind(), format!("create {}: {error}", dir.display()))
    })?;
    for &(name, contents) in BUILTIN_MANIFESTS {
        let target = dir.join(name);
        if mode == SeedMode::IfMissing && target.exists() {
            continue;
        }
        crate::store_util::atomic_write(&target, contents.as_bytes())
            .map_err(|error| std::io::Error::other(format!("install {name}: {error}")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn embedded(name: &str) -> &'static str {
        BUILTIN_MANIFESTS
            .iter()
            .find(|(file, _)| *file == name)
            .map(|(_, contents)| *contents)
            .expect("manifest present in BUILTIN_MANIFESTS")
    }

    fn chatterbox_catalog(
        ve_revision: &str,
        tts_primary_revision: &str,
        tts_component_revision: &str,
    ) -> String {
        json!({
            "models": [
                {
                    "id": "chatterbox_ve",
                    "downloads": [{
                        "provider": "huggingface",
                        "repo": "ResembleAI/chatterbox",
                        "revision": ve_revision,
                        "files": ["ve.safetensors"]
                    }]
                },
                {
                    "id": "chatterbox_tts",
                    "downloads": [
                        {
                            "provider": "huggingface",
                            "repo": "ResembleAI/chatterbox",
                            "revision": tts_primary_revision,
                            "files": ["t3_cfg.safetensors"]
                        },
                        {
                            "provider": "huggingface",
                            "repo": "ResembleAI/chatterbox",
                            "revision": tts_component_revision,
                            "coRequisite": true,
                            "componentId": "voice_embedding",
                            "files": ["ve.safetensors"]
                        }
                    ]
                }
            ]
        })
        .to_string()
    }

    #[test]
    fn shipped_chatterbox_downloads_share_one_manifest_authoritative_pin() {
        let pin = chatterbox_ve_pin().expect("shipped Chatterbox pin is valid");
        assert_eq!(pin.repo, "ResembleAI/chatterbox");
        assert_eq!(pin.revision.len(), 40);
        assert!(pin.revision.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn chatterbox_pin_rejects_malformed_and_divergent_revisions() {
        let sha = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18";
        let malformed = chatterbox_catalog("main", sha, sha);
        assert!(parse_chatterbox_ve_pin(&malformed)
            .unwrap_err()
            .contains("full 40-character lowercase-hex"));

        let other = "1111111111111111111111111111111111111111";
        let divergent = chatterbox_catalog(sha, other, sha);
        assert!(parse_chatterbox_ve_pin(&divergent)
            .unwrap_err()
            .contains("must share chatterbox_ve's repo"));
    }

    #[test]
    fn chatterbox_pin_rejects_ambiguous_voice_encoder_downloads() {
        let sha = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18";
        let mut catalog: serde_json::Value =
            serde_json::from_str(&chatterbox_catalog(sha, sha, sha)).unwrap();
        let duplicate = catalog["models"][0]["downloads"][0].clone();
        catalog["models"][0]["downloads"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        assert!(parse_chatterbox_ve_pin(&catalog.to_string())
            .unwrap_err()
            .contains("ambiguous"));
    }

    #[test]
    fn no_builtin_manifest_has_a_duplicate_key() {
        // Guard against the silent last-key-wins class (sc-10199): serde_json accepts a
        // duplicate object key without error and keeps only the last value, so a future
        // "add a field that already exists in another block" edit could drop data with no
        // parse failure — exactly how the img2img `ui` flag was lost (sc-10198, #1249).
        // Every shipped manifest, comments stripped, must be free of duplicate keys.
        for (name, contents) in BUILTIN_MANIFESTS {
            let stripped = crate::jsonc::strip_jsonc_comments(contents);
            crate::jsonc::reject_duplicate_keys(&stripped)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
        }
    }

    #[test]
    fn styles_manifest_parses_to_a_populated_catalog() {
        // The Style catalog the API serves + folds (sc-13134) is embedded here; a broken/empty
        // seed would leave GET /api/v1/styles and the server-side fold silently non-functional.
        // The JS drift guard (styleCatalog.test.js) proves it derives from style.txt; this is the
        // Rust-side backstop that the embedded copy parses and carries the shipped groups.
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.styles.jsonc"));
        let catalog: serde_json::Value =
            serde_json::from_str(&stripped).expect("styles manifest parses as JSON");
        assert_eq!(
            catalog
                .get("schemaVersion")
                .and_then(serde_json::Value::as_i64),
            Some(1)
        );
        let groups = catalog
            .get("groups")
            .and_then(serde_json::Value::as_array)
            .expect("styles manifest carries a groups array");
        assert_eq!(groups.len(), 8, "the eight authored top-level groups ship");
        let total_styles: usize = groups
            .iter()
            .filter_map(|group| group.get("styles").and_then(serde_json::Value::as_array))
            .map(Vec::len)
            .sum();
        assert_eq!(total_styles, 278, "the shipped sub-style count");
    }

    #[test]
    fn every_builtin_model_prompt_guide_exists_in_the_web_app() {
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.models.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.models.jsonc parses as JSON");
        let models = manifest["models"]
            .as_array()
            .expect("builtin.models.jsonc has a models array");
        let prompt_guides_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/web/public/prompt-guides");
        let mut checked = 0;

        for model in models {
            let Some(guide_path) = model["ui"]["promptGuide"]["path"].as_str() else {
                continue;
            };
            let relative_path = guide_path
                .strip_prefix("/prompt-guides/")
                .unwrap_or_else(|| panic!("{guide_path} is not rooted under /prompt-guides/"));
            let model_id = model["id"].as_str().unwrap_or("<missing model id>");

            // Both production consumers fetch this URL. A missing file silently empties the
            // guide modal and makes prompt refinement proceed without its declared guide text.
            assert!(
                prompt_guides_dir.join(relative_path).is_file(),
                "{model_id} ui.promptGuide.path does not resolve to a web asset: {guide_path}"
            );
            checked += 1;
        }

        assert!(
            checked > 0,
            "builtin models declare at least one prompt guide"
        );
    }

    #[test]
    fn krea_2_raw_declares_no_default_negative_and_low_guidance_with_raw_guide() {
        // sc-14203 (partially revises sc-13881): Raw's defaults fix. On-device render-validation showed the
        // "soft/over-warm" heat was driven by the guidance default (Krea's nominal 3.5 ≡ standard-CFG 4.5)
        // AND the sc-13881 S1 negative was a CO-CAUSE — guidance 1.0 WITH the S1 negative still rendered hot,
        // while guidance ~1.0 + an EMPTY negative renders natural. Source-of-truth manifest facts now:
        //   1. Raw seeds NO default negative (the sc-13881 seeding is removed — a fresh job starts empty).
        //      The negative-prompt capability is unchanged; only the default is gone.
        //   2. Raw's default guidance is 1.0 (down from 3.5).
        //   3. Raw points at its OWN guide krea-2-raw.md (the shared krea-2.md is Turbo-specific — "no
        //      negative", "~8 steps", "CFG-off" — all wrong for Raw); Turbo keeps krea-2.md.
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.models.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.models.jsonc parses as JSON");
        let models = manifest["models"]
            .as_array()
            .expect("builtin.models.jsonc has a models array");
        let find = |id: &str| {
            models
                .iter()
                .find(|model| model["id"].as_str() == Some(id))
                .unwrap_or_else(|| panic!("{id} present in the builtin models catalog"))
        };

        let raw = find("krea_2_raw");
        // No default negative is seeded (sc-14203). Discriminating: this fails if a default negative — the
        // sc-13881 string or any other — is re-added under ui.defaultNegativePrompt.
        assert!(
            raw["ui"]["defaultNegativePrompt"].is_null(),
            "krea_2_raw declares no default negative (sc-14203 dropped the sc-13881 seeding), got {:?}",
            raw["ui"]["defaultNegativePrompt"]
        );
        // Default guidance is 1.0, not the old 3.5 (pins the non-default so it discriminates a revert).
        assert_eq!(
            raw["defaults"]["guidanceScale"].as_f64(),
            Some(1.0),
            "krea_2_raw defaults to guidance 1.0 (sc-14203)"
        );
        assert_eq!(
            raw["ui"]["promptGuide"]["path"].as_str(),
            Some("/prompt-guides/krea-2-raw.md"),
            "krea_2_raw points at the Raw-specific prompt guide"
        );

        // Turbo is CFG-free — it must NOT carry a default negative, and it keeps the Turbo guide.
        let turbo = find("krea_2_turbo");
        assert!(
            turbo["ui"]["defaultNegativePrompt"].is_null(),
            "krea_2_turbo (CFG-free) declares no default negative"
        );
        assert_eq!(
            turbo["ui"]["promptGuide"]["path"].as_str(),
            Some("/prompt-guides/krea-2.md"),
            "krea_2_turbo keeps the shared Turbo prompt guide"
        );
    }

    #[test]
    fn ships_the_seeded_audio_models_with_populated_capability_blocks() {
        // sc-13402 (epic 13400) + sc-13412 + sc-13675 + sc-13676: the shipped catalog the API serves
        // carries the live audio providers as first-class `type: "audio"` entries, each with a populated
        // `audio` capability sub-block, and `audio` parses as a first-class ModelKind (not
        // the Unknown fallback) so the type is accepted end to end.
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.models.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.models.jsonc parses as JSON");
        let models = manifest["models"]
            .as_array()
            .expect("builtin.models.jsonc has a models array");

        let audio_ids = [
            "kokoro_82m",
            "moss_sfx_v2",
            "acestep_v15_turbo",
            "openvoice_v2",
            "chatterbox_ve",
            // Native cloned-voice TTS generator (sc-13412): script + reference clip → cloned WAV in
            // one call, with both VoiceEmbedding and ReferenceAudio conditioning advertised.
            "chatterbox_tts",
            // Streaming TTS (sc-13675): the audio lane's first `supportsStreaming` provider.
            "moss_tts_realtime",
            // Multi-speaker dialogue TTS (sc-13676): the audio lane's first `supportsMultiSpeaker`
            // provider (max_speakers = 2), the 8th audio model.
            "moss_ttsd_v05",
        ];
        for id in audio_ids {
            let entry = models
                .iter()
                .find(|m| m["id"].as_str() == Some(id))
                .unwrap_or_else(|| panic!("seeded audio model {id} missing from the catalog"));
            let ty = entry["type"].as_str().unwrap_or_default();
            assert_eq!(ty, "audio", "{id} must be type:audio");
            // `audio` is a first-class ModelKind, not degraded to Unknown().
            let kind: crate::contracts::ModelKind =
                serde_json::from_value(entry["type"].clone()).expect("type deserializes");
            assert_eq!(
                kind,
                crate::contracts::ModelKind::Audio,
                "{id}: `audio` must parse as ModelKind::Audio, not Unknown"
            );
            let audio = entry["audio"]
                .as_object()
                .unwrap_or_else(|| panic!("{id} must carry a populated `audio` block"));
            assert!(!audio.is_empty(), "{id}.audio must not be empty");
            // Installable/downloadable like image/video models.
            assert!(
                entry["downloads"][0]["repo"].as_str().is_some(),
                "{id} must define a download repo"
            );
        }

        // Kokoro is the recommended Speech model and advertises its 28 shipped voices.
        let kokoro = models
            .iter()
            .find(|m| m["id"].as_str() == Some("kokoro_82m"))
            .expect("kokoro present");
        assert_eq!(kokoro["recommended"].as_bool(), Some(true));
        assert_eq!(
            kokoro["audio"]["voices"].as_array().map(Vec::len),
            Some(28),
            "Kokoro advertises its 28 shipped English voices"
        );

        // MOSS-TTS-Realtime (sc-13675) is the audio lane's first STREAMING model: it advertises
        // `audio.supportsStreaming: true` (mirroring the backend Capabilities), ships NO fixed voice
        // bank (it serves Speech via the streaming signal, not a voice list), and declares the
        // MOSS-Audio-Tokenizer codec as a pinned-revision co-requisite so an offline install is
        // self-contained. No other seeded audio model advertises streaming, so this pins the surface.
        let moss_tts = models
            .iter()
            .find(|m| m["id"].as_str() == Some("moss_tts_realtime"))
            .expect("moss_tts_realtime present");
        assert_eq!(
            moss_tts["audio"]["supportsStreaming"].as_bool(),
            Some(true),
            "moss_tts_realtime must advertise audio.supportsStreaming: true"
        );
        assert!(
            moss_tts["audio"]["voices"].as_array().is_none(),
            "moss_tts_realtime ships no fixed voice bank"
        );
        let codec = moss_tts["downloads"]
            .as_array()
            .expect("moss_tts_realtime downloads array")
            .iter()
            .find(|d| d["coRequisite"].as_bool() == Some(true))
            .expect("moss_tts_realtime declares the MOSS-Audio-Tokenizer codec co-requisite");
        assert_eq!(
            codec["repo"].as_str(),
            Some("OpenMOSS-Team/MOSS-Audio-Tokenizer"),
            "the co-requisite is the MOSS-Audio-Tokenizer codec"
        );
        assert_eq!(
            codec["revision"].as_str().map(str::len),
            Some(40),
            "the codec co-requisite pins a full 40-hex commit SHA (hf_get_pinned reads snapshots/<sha>/)"
        );
        assert_eq!(
            codec["componentId"].as_str(),
            Some("codec"),
            "the codec co-requisite is tagged componentId: \"codec\" (sc-13681) so the worker's generic \
             resolve_co_requisites seam stages it under the descriptor's required_components: [\"codec\"]"
        );

        // MOSS-TTSD v0.5 (sc-13676) is the audio lane's first MULTI-SPEAKER model: it advertises
        // `audio.supportsMultiSpeaker: true` + `audio.maxSpeakers: 2` (mirroring the backend
        // Capabilities), ships NO fixed voice bank (it maps opaque [S1]/[S2] turn labels itself), does
        // NOT stream, and declares the XY_Tokenizer codec as a pinned-revision co-requisite so an
        // offline install is self-contained. No other seeded audio model advertises multi-speaker, so
        // this pins the surface.
        let moss_ttsd = models
            .iter()
            .find(|m| m["id"].as_str() == Some("moss_ttsd_v05"))
            .expect("moss_ttsd_v05 present");
        assert_eq!(
            moss_ttsd["audio"]["supportsMultiSpeaker"].as_bool(),
            Some(true),
            "moss_ttsd_v05 must advertise audio.supportsMultiSpeaker: true"
        );
        assert_eq!(
            moss_ttsd["audio"]["maxSpeakers"].as_u64(),
            Some(2),
            "moss_ttsd_v05 advertises max_speakers = 2 (matching the backend Capabilities)"
        );
        assert!(
            moss_ttsd["audio"]["voices"].as_array().is_none(),
            "moss_ttsd_v05 ships no fixed voice bank (opaque [S1]/[S2] labels)"
        );
        assert_ne!(
            moss_ttsd["audio"]["supportsStreaming"].as_bool(),
            Some(true),
            "moss_ttsd_v05 is one-shot, not streaming"
        );
        let ttsd_codec = moss_ttsd["downloads"]
            .as_array()
            .expect("moss_ttsd_v05 downloads array")
            .iter()
            .find(|d| d["coRequisite"].as_bool() == Some(true))
            .expect("moss_ttsd_v05 declares the XY_Tokenizer codec co-requisite");
        assert_eq!(
            ttsd_codec["repo"].as_str(),
            Some("OpenMOSS-Team/XY_Tokenizer_TTSD_V0"),
            "the co-requisite is the XY_Tokenizer codec"
        );
        assert_eq!(
            ttsd_codec["revision"].as_str().map(str::len),
            Some(40),
            "the codec co-requisite pins a full 40-hex commit SHA (hf_get_pinned reads snapshots/<sha>/)"
        );
        assert_eq!(
            ttsd_codec["componentId"].as_str(),
            Some("codec"),
            "the codec co-requisite is tagged componentId: \"codec\" (sc-13681) so the worker's generic \
             resolve_co_requisites seam stages it under the descriptor's required_components: [\"codec\"]"
        );

        for model in [
            "kokoro_82m",
            "moss_sfx_v2",
            "acestep_v15_turbo",
            "openvoice_v2",
            "chatterbox_ve",
            "chatterbox_tts",
            // MOSS-TTSD is multi-speaker, not streaming — it belongs on the streaming-negative side.
            "moss_ttsd_v05",
        ] {
            let entry = models
                .iter()
                .find(|m| m["id"].as_str() == Some(model))
                .unwrap_or_else(|| panic!("{model} present"));
            assert_ne!(
                entry["audio"]["supportsStreaming"].as_bool(),
                Some(true),
                "{model} must NOT advertise streaming — only moss_tts_realtime does"
            );
        }

        // Multi-speaker is exclusive to MOSS-TTSD across the seeded set (sc-13676) — the mirror of the
        // streaming-negative loop, so the capability that reveals the segmented-script editor can never
        // silently leak onto a single-voice model.
        for model in [
            "kokoro_82m",
            "moss_sfx_v2",
            "acestep_v15_turbo",
            "openvoice_v2",
            "chatterbox_ve",
            "chatterbox_tts",
            "moss_tts_realtime",
        ] {
            let entry = models
                .iter()
                .find(|m| m["id"].as_str() == Some(model))
                .unwrap_or_else(|| panic!("{model} present"));
            assert_ne!(
                entry["audio"]["supportsMultiSpeaker"].as_bool(),
                Some(true),
                "{model} must NOT advertise multi-speaker — only moss_ttsd_v05 does"
            );
        }
    }

    /// A full 40-char lowercase-hex commit SHA — the only revision shape the F-029 pin
    /// authority accepts (`^[0-9a-f]{40}$`, mirrored from model-manifest.schema.json).
    fn is_full_sha_revision(revision: &str) -> bool {
        revision.len() == 40
            && revision
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    /// `(model_id, repo)` co-requisite download pairs whose F-029 pin migration is
    /// still IN FLIGHT under sc-13591. Each is a KNOWN, tracked gap: the immutable
    /// commit SHA lives in the sc-13591 inventory but is applied by a later
    /// per-family story, not sc-13659 (which is schema + plumbing + enforcement only
    /// and must not add real pins). A brand-new co-requisite may NOT join this list —
    /// pin its `revision` instead. Kept in lockstep with the identical Python audit
    /// allowlist in tests/test_builtin_manifest_audit.py.
    const COREQUISITE_REVISION_MIGRATION_PENDING: &[(&str, &str)] = &[
        // ("ltx_2_3", "SceneWorks/ltx-2.3-mlx") pinned in sc-13683 (the gemma coRequisite now carries the
        // full 40-hex LTX_BUNDLE_REVISION), so its migration row was removed here + in the Python twin.
        (
            "ltx_2_3_eros",
            "TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments",
        ),
        ("wan_2_2_t2v_14b", "lightx2v/Wan2.2-Lightning"),
        ("wan_2_2_i2v_14b", "lightx2v/Wan2.2-Lightning"),
    ];

    /// Every `(model_id, repo)` co-requisite pair in the live manifest that is NOT
    /// pinned to a full 40-hex commit SHA. Shared by the enforcement test and its
    /// self-cleaning allowlist audit so both read the same signal.
    fn corequisite_revision_gaps(models: &[serde_json::Value]) -> Vec<(String, String)> {
        let mut gaps = Vec::new();
        for model in models {
            let id = model["id"].as_str().unwrap_or_default();
            let Some(downloads) = model["downloads"].as_array() else {
                continue;
            };
            for download in downloads {
                if download["coRequisite"].as_bool() != Some(true) {
                    continue;
                }
                let pinned = download["revision"]
                    .as_str()
                    .is_some_and(is_full_sha_revision);
                if !pinned {
                    let repo = download["repo"].as_str().unwrap_or_default();
                    gaps.push((id.to_owned(), repo.to_owned()));
                }
            }
        }
        gaps
    }

    #[test]
    fn corequisite_downloads_pin_a_full_sha_revision() {
        // F-029 (sc-13659): a coRequisite: true download is a FETCH-ALL companion the runtime
        // resolves offline via a pinned-SHA `hf_get_pinned` reading `snapshots/<sha>/`. Leaving it
        // on `main` lands the wrong snapshot and hard-fails offline, so every co-requisite MUST pin a
        // full 40-hex commit — the Rust-side backstop to the identical Python manifest audit. The
        // only tolerated gaps are the sc-13591 pins still being migrated by later stories.
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.models.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.models.jsonc parses as JSON");
        let models = manifest["models"]
            .as_array()
            .expect("builtin.models.jsonc has a models array");

        let allowlist: std::collections::HashSet<(&str, &str)> =
            COREQUISITE_REVISION_MIGRATION_PENDING
                .iter()
                .copied()
                .collect();
        let unexpected: Vec<(String, String)> = corequisite_revision_gaps(models)
            .into_iter()
            .filter(|(id, repo)| !allowlist.contains(&(id.as_str(), repo.as_str())))
            .collect();
        assert!(
            unexpected.is_empty(),
            "co-requisite downloads must pin a 40-hex commit SHA (F-029, sc-13659); \
             these are unpinned and NOT tracked for the sc-13591 migration: {unexpected:?}"
        );
    }

    #[test]
    fn corequisite_revision_migration_allowlist_has_no_stale_entries() {
        // Self-cleaning guard: the moment a later sc-13591 story pins one of these companions (or
        // removes the entry), its allowlist row stops matching an actual gap and MUST be deleted —
        // otherwise the allowlist would silently keep excusing a co-requisite that is already
        // compliant, masking a future regression. This asserts every allowlisted pair still names a
        // live, unpinned co-requisite. (This is why a test asserting a default value is a false green:
        // the allowlist must be forced to shrink, not linger.)
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.models.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.models.jsonc parses as JSON");
        let models = manifest["models"]
            .as_array()
            .expect("builtin.models.jsonc has a models array");

        let gaps: std::collections::HashSet<(String, String)> =
            corequisite_revision_gaps(models).into_iter().collect();
        let stale: Vec<&(&str, &str)> = COREQUISITE_REVISION_MIGRATION_PENDING
            .iter()
            .filter(|(id, repo)| !gaps.contains(&((*id).to_owned(), (*repo).to_owned())))
            .collect();
        assert!(
            stale.is_empty(),
            "stale F-029 migration allowlist entries (now pinned or removed) must be deleted from \
             COREQUISITE_REVISION_MIGRATION_PENDING: {stale:?}"
        );
    }

    #[test]
    fn model_download_revision_is_a_typed_round_tripping_field() {
        use crate::contracts::ModelDownload;

        // sc-13659: `revision` is a first-class typed field on ModelDownload, not an `extra` bag key,
        // so the F-029 pin round-trips through the contract type. A pinned entry deserializes into the
        // typed field (leaving `extra` free of it) and re-serializes the same key; an entry with no
        // revision keeps it `None` and serializes no `revision` key (main-branch default preserved).
        let sha = "80b60f9caead09b8d3b512bda0b24038f28c08ec";
        let pinned: ModelDownload = serde_json::from_value(serde_json::json!({
            "provider": "huggingface",
            "repo": "SceneWorks/perth-implicit",
            "files": ["perth_implicit.safetensors"],
            "revision": sha,
            "coRequisite": true,
            "componentId": "perth",
        }))
        .expect("pinned co-requisite deserializes");
        assert_eq!(pinned.revision.as_deref(), Some(sha));
        assert!(
            !pinned.extra.contains_key("revision"),
            "revision must land in the typed field, not the extra bag"
        );
        assert_eq!(
            pinned.extra.get("coRequisite"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            serde_json::to_value(&pinned).expect("re-serialize")["revision"],
            serde_json::json!(sha)
        );
        // sc-13679: `componentId` is likewise a first-class typed field (the repo→component mapping the
        // co-requisite provisioning seam reads), so it round-trips through the typed slot, not `extra`.
        assert_eq!(pinned.component_id.as_deref(), Some("perth"));
        assert!(
            !pinned.extra.contains_key("componentId"),
            "componentId must land in the typed field, not the extra bag"
        );
        assert_eq!(
            serde_json::to_value(&pinned).expect("re-serialize")["componentId"],
            serde_json::json!("perth")
        );

        let unpinned: ModelDownload = serde_json::from_value(serde_json::json!({
            "provider": "huggingface",
            "repo": "black-forest-labs/FLUX.1-dev",
            "files": [],
        }))
        .expect("unpinned entry deserializes");
        assert_eq!(unpinned.revision, None);
        assert_eq!(unpinned.component_id, None);
        assert!(
            serde_json::to_value(&unpinned)
                .expect("re-serialize")
                .get("revision")
                .is_none(),
            "an unpinned download must not serialize a revision key (main-branch default)"
        );
        assert!(
            serde_json::to_value(&unpinned)
                .expect("re-serialize")
                .get("componentId")
                .is_none(),
            "a non-component download must not serialize a componentId key"
        );
    }

    #[test]
    fn seeds_every_manifest_into_a_fresh_dir() {
        let temp = tempfile::tempdir().expect("temp dir");
        seed_builtin_manifests(temp.path(), SeedMode::IfMissing).expect("seeding succeeds");

        let dir = temp.path().join("manifests");
        for (name, contents) in BUILTIN_MANIFESTS {
            let written = std::fs::read_to_string(dir.join(name)).expect("manifest written");
            assert_eq!(&written, contents, "{name} matches the embedded copy");
        }
        // No temp files left behind by the atomic write.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .expect("read manifests dir")
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "no .tmp files remain after seeding");
    }

    #[test]
    fn if_missing_never_clobbers_an_existing_manifest() {
        let temp = tempfile::tempdir().expect("temp dir");
        let dir = temp.path().join("manifests");
        std::fs::create_dir_all(&dir).expect("create manifests dir");
        let edited = dir.join("builtin.models.jsonc");
        std::fs::write(&edited, "{ \"models\": [] } // operator edit").expect("seed existing");

        seed_builtin_manifests(temp.path(), SeedMode::IfMissing).expect("seeding succeeds");

        // The operator's copy is left untouched...
        assert_eq!(
            std::fs::read_to_string(&edited).expect("read existing"),
            "{ \"models\": [] } // operator edit"
        );
        // ...while the genuinely-missing manifests are still filled in.
        assert_eq!(
            std::fs::read_to_string(dir.join("builtin.loras.jsonc")).expect("loras written"),
            embedded("builtin.loras.jsonc")
        );
    }

    #[test]
    fn overwrite_repairs_a_truncated_manifest_and_leaves_no_temp() {
        // Simulate the crash the old temp+rename path could leave behind: a
        // zero-length `builtin.*.jsonc`. Overwrite seeding must replace it with the
        // full embedded copy and leave no atomic-write temp files behind.
        let temp = tempfile::tempdir().expect("temp dir");
        let dir = temp.path().join("manifests");
        std::fs::create_dir_all(&dir).expect("create manifests dir");
        let truncated = dir.join("builtin.models.jsonc");
        std::fs::write(&truncated, b"").expect("seed zero-length manifest");

        seed_builtin_manifests(temp.path(), SeedMode::Overwrite).expect("seeding succeeds");

        assert_eq!(
            std::fs::read_to_string(&truncated).expect("read repaired"),
            embedded("builtin.models.jsonc"),
            "overwrite repairs the truncated manifest to the full embedded copy"
        );
        // atomic_write stages into `*.<token>.tmp` and renames it away; none survive.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .expect("read manifests dir")
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "no atomic-write temp files remain");
    }

    #[test]
    fn overwrite_replaces_an_existing_manifest() {
        let temp = tempfile::tempdir().expect("temp dir");
        let dir = temp.path().join("manifests");
        std::fs::create_dir_all(&dir).expect("create manifests dir");
        let stale = dir.join("builtin.models.jsonc");
        std::fs::write(&stale, "stale").expect("seed stale");

        seed_builtin_manifests(temp.path(), SeedMode::Overwrite).expect("seeding succeeds");

        assert_eq!(
            std::fs::read_to_string(&stale).expect("read replaced"),
            embedded("builtin.models.jsonc"),
            "overwrite refreshes the builtin manifest to the embedded copy"
        );
    }

    #[test]
    fn krea_turbo_accelerator_lora_is_registered_and_sha_pinned() {
        // sc-13882 (epic 13879 S2): the Krea 2 turbo LoRA registers as a weight-load-only builtin
        // accelerator. Pin every NON-default field so this discriminates a real registration from an
        // empty/renamed one: repo + exact file, a FULL 40-hex commit SHA revision (a floating `main`
        // would silently drift the accelerator weights under the frozen Raw DiT), family `krea_2`
        // (what surfaces it in the Raw picker), and `role: accelerator` (the sampling-regime marker
        // S3 routes on).
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.loras.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.loras.jsonc parses as JSON");
        let loras = manifest["loras"]
            .as_array()
            .expect("builtin.loras.jsonc has a loras array");
        let lora = loras
            .iter()
            .find(|l| l["id"] == serde_json::json!("krea2_turbo_accel"))
            .expect("krea2_turbo_accel is registered in builtin.loras.jsonc");

        assert_eq!(lora["family"], serde_json::json!("krea_2"));
        assert_eq!(lora["role"], serde_json::json!("accelerator"));
        assert_eq!(
            lora["compatibility"]["families"],
            serde_json::json!(["krea_2"])
        );
        assert_eq!(lora["source"]["provider"], serde_json::json!("huggingface"));
        assert_eq!(
            lora["source"]["repo"],
            serde_json::json!("Comfy-Org/Krea-2")
        );
        assert_eq!(
            lora["source"]["file"],
            serde_json::json!("loras/krea2_turbo_lora_rank_64_bf16.safetensors")
        );
        let revision = lora["source"]["revision"]
            .as_str()
            .expect("the accelerator LoRA pins a source.revision");
        assert!(
            is_full_sha_revision(revision),
            "the accelerator LoRA must pin a full 40-hex commit SHA (not a floating branch); got \
             {revision:?}"
        );
        assert_eq!(revision, "952f49d49653cb42e7d6cf7cbfad74738073ec7d");
    }

    #[test]
    fn krea_2_raw_advertises_the_acceleration_lora_compat_type() {
        // sc-13882: Raw must advertise "acceleration" as a compatible LoRA type so the turbo adapter
        // is offered under Raw. Assert the NEW value is present AND the pre-existing trainable tiers
        // survive the edit (a replacement, not an addition, would be the likely regression).
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.models.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.models.jsonc parses as JSON");
        let models = manifest["models"]
            .as_array()
            .expect("builtin.models.jsonc has a models array");
        let raw = models
            .iter()
            .find(|m| m["id"] == serde_json::json!("krea_2_raw"))
            .expect("krea_2_raw is present");
        let types = raw["loraCompatibility"]["types"]
            .as_array()
            .expect("krea_2_raw declares loraCompatibility.types");
        assert!(
            types.contains(&serde_json::json!("acceleration")),
            "krea_2_raw must advertise the acceleration compat type (sc-13882); got {types:?}"
        );
        assert!(
            types.contains(&serde_json::json!("character"))
                && types.contains(&serde_json::json!("style")),
            "the trainable character/style tiers must survive the edit; got {types:?}"
        );
        // The family match that actually surfaces the LoRA in the Raw picker is unchanged.
        assert_eq!(
            raw["loraCompatibility"]["families"],
            serde_json::json!(["krea_2"])
        );
    }

    #[test]
    fn mage_flow_family_ships_six_variants_as_prequantized_per_tier_artifacts() {
        // sc-14980 / sc-14979 (epic 14034), superseding the sc-14059 load-time-quant invariant.
        //
        // Mage's q4/q8/bf16 are no longer on-the-fly quantization over ONE dense snapshot; they are
        // physically distinct `<tier>/` artifacts on our mirrors, with the text encoder and VAE —
        // bit-identical across all six variants — hosted once as per-tier co-requisites. The old
        // test pinned the exact opposite (identical repo+revision+files across tiers, and
        // `standardTierLayout` absent), so it is REPLACED rather than relaxed. What it protected is
        // preserved and inverted here, plus the two new invariants the split layout depends on:
        //
        //   1. Per-tier delete honesty (closes sc-14046's delegated acceptance). Each tier now
        //      declares its OWN `<tier>/*` predicate, so the overlap-safe delete in apps/rust-api
        //      reclaims that tier's real bytes while the other tiers' predicates keep theirs. The
        //      DISTINCT-predicate invariant is now the load-bearing one; identical predicates would
        //      silently return per-tier delete to reclaiming zero bytes.
        //   2. Shared components are never stranded. text_encoder/vae are `coRequisite` rows, which
        //      per-variant and per-tier delete skip structurally, so deleting one variant or tier
        //      can never remove weights another installed variant still needs.
        //   3. `mlx.standardTierLayout` MUST be set — it is what routes the worker into
        //      `standard_tier_subdir` so a tier request loads `<tier>/` instead of the snapshot root.
        //   4. Every co-requisite carries `componentId` + `subdir` + `variant`: the id matches the
        //      `mlx_mage` descriptor's `required_components`, the subdir addresses the component dir
        //      inside the shared mirror, and the variant scopes the fetch to one tier.
        //
        // It also proves the P6 model guide (apps/web/public/prompt-guides/mage-flow.md) is wired to
        // every variant.
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.models.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.models.jsonc parses as JSON");
        let models = manifest["models"]
            .as_array()
            .expect("builtin.models.jsonc has a models array");
        let find = |id: &str| {
            models
                .iter()
                .find(|model| model["id"].as_str() == Some(id))
                .unwrap_or_else(|| panic!("{id} present in the builtin models catalog"))
        };

        // Every Mage-Flow row in the catalog, by family — so a dropped or extra variant fails here.
        let mage_ids: Vec<&str> = models
            .iter()
            .filter(|model| model["family"].as_str() == Some("mage-flow"))
            .map(|model| model["id"].as_str().unwrap_or_default())
            .collect();
        let mut sorted_ids = mage_ids.clone();
        sorted_ids.sort_unstable();
        assert_eq!(
            sorted_ids,
            vec![
                "mage_flow",
                "mage_flow_base",
                "mage_flow_edit",
                "mage_flow_edit_base",
                "mage_flow_edit_turbo",
                "mage_flow_turbo",
            ],
            "the six Mage-Flow gen+edit variants ship in the catalog"
        );

        // (id, capability, steps, guidance) — the per-variant defaults from the epic Ground-Truth
        // Reference (Base 30/5, RL 20/5, Turbo 4/1(off), Edit-Base 30/5, Edit 30/5, Edit-Turbo 4/1).
        // These are manifest DATA (not code defaults), so pinning them discriminates a manifest edit.
        let expected: &[(&str, &str, u64, f64)] = &[
            ("mage_flow_base", "text_to_image", 30, 5.0),
            ("mage_flow", "text_to_image", 20, 5.0),
            ("mage_flow_turbo", "text_to_image", 4, 1.0),
            ("mage_flow_edit_base", "edit_image", 30, 5.0),
            ("mage_flow_edit", "edit_image", 30, 5.0),
            ("mage_flow_edit_turbo", "edit_image", 4, 1.0),
        ];

        // The one shared components mirror, pinned. Every variant must agree on it — a second
        // components repo would silently double the shared-weight cost the split layout exists to
        // avoid.
        const COMPONENTS_REPO: &str = "SceneWorks/Mage-Flow-Components-mlx";
        const TIERS: [&str; 3] = ["q4", "q8", "bf16"];

        for (id, capability, steps, guidance) in expected {
            let model = find(id);
            assert_eq!(
                model["type"].as_str(),
                Some("image"),
                "{id} is an image model"
            );
            assert_eq!(
                model["adapter"].as_str(),
                Some("mlx_mage"),
                "{id} routes to the mlx_mage adapter"
            );
            assert!(
                model["capabilities"]
                    .as_array()
                    .is_some_and(|caps| caps.iter().any(|c| c.as_str() == Some(capability))),
                "{id} advertises the {capability} capability"
            );
            assert_eq!(
                model["defaults"]["steps"].as_u64(),
                Some(*steps),
                "{id} default steps"
            );
            assert_eq!(
                model["defaults"]["guidanceScale"].as_f64(),
                Some(*guidance),
                "{id} default guidance (Turbo/Edit-Turbo are CFG-off at 1.0)"
            );
            assert_eq!(
                model["ui"]["promptGuide"]["path"].as_str(),
                Some("/prompt-guides/mage-flow.md"),
                "{id} points at the P6 Mage-Flow model guide"
            );

            // The physical-tier markers. `quantize` still names the default tier, but it now selects
            // WHICH subdir is loaded, and `standardTierLayout` is what makes the worker descend.
            assert_eq!(
                model["mlx"]["quantize"].as_u64(),
                Some(4),
                "{id} declares q4 as the default tier"
            );
            assert_eq!(
                model["mlx"]["standardTierLayout"].as_bool(),
                Some(true),
                "{id} MUST declare standardTierLayout — without it the worker loads the snapshot \
                 ROOT instead of the requested `<tier>/` subdir, silently serving the dense flat \
                 weights for every tier and making per-tier delete meaningless again"
            );

            let downloads = model["downloads"]
                .as_array()
                .unwrap_or_else(|| panic!("{id} has a downloads array"));
            let tier_rows: Vec<&serde_json::Value> = downloads
                .iter()
                .filter(|d| d.get("coRequisite").and_then(serde_json::Value::as_bool) != Some(true))
                .collect();
            let co_reqs: Vec<&serde_json::Value> = downloads
                .iter()
                .filter(|d| d.get("coRequisite").and_then(serde_json::Value::as_bool) == Some(true))
                .collect();
            assert_eq!(
                tier_rows.len(),
                3,
                "{id} declares exactly the q4/q8/bf16 tiers"
            );
            assert_eq!(
                co_reqs.len(),
                6,
                "{id} declares the shared text_encoder + vae at each of the three tiers"
            );

            let tier_of = |variant: &str| {
                *tier_rows
                    .iter()
                    .find(|d| {
                        d["variant"]
                            .as_str()
                            .is_some_and(|v| v.eq_ignore_ascii_case(variant))
                    })
                    .unwrap_or_else(|| panic!("{id} declares a {variant} tier"))
            };
            let (q4, q8, bf16) = (tier_of("q4"), tier_of("q8"), tier_of("bf16"));
            assert_eq!(
                q4["default"].as_bool(),
                Some(true),
                "{id} q4 is the default install tier"
            );
            assert!(
                q8["default"].as_bool() != Some(true) && bf16["default"].as_bool() != Some(true),
                "{id} marks only one default tier"
            );

            // THE inverted invariant. Under load-time quant the three tiers had to share one file
            // predicate so a per-tier delete could not corrupt the shared snapshot. Now they must
            // NOT: each tier owns a disjoint `<tier>/*` subtree, which is exactly what lets a tier
            // delete reclaim real bytes while the siblings survive. Identical predicates here would
            // mean the re-host silently regressed to zero-byte tier deletes.
            for (name, tier) in [("q4", q4), ("q8", q8), ("bf16", bf16)] {
                let files: Vec<&str> = tier["files"]
                    .as_array()
                    .unwrap_or_else(|| panic!("{id} {name} declares a files predicate"))
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .collect();
                assert_eq!(
                    files,
                    vec![format!("{name}/*")],
                    "{id} {name} must fetch ONLY its own tier subtree"
                );
                // Fetching the flat dense tree would defeat the whole re-host: the tier download
                // would pull all 17.5 GB again. The dense root stays HOSTED for existing installs,
                // but no tier may reference it.
                for legacy in [
                    "transformer/*",
                    "text_encoder/*",
                    "vae/*",
                    "model_index.json",
                ] {
                    assert!(
                        !files.contains(&legacy),
                        "{id} {name} must not fetch the flat dense {legacy}"
                    );
                }
            }
            assert_ne!(
                q4["files"], q8["files"],
                "{id} q4 and q8 must own DISJOINT file predicates (physical per-tier reclaim)"
            );
            assert_ne!(
                q4["files"], bf16["files"],
                "{id} q4 and bf16 must own DISJOINT file predicates (physical per-tier reclaim)"
            );

            // The mirror lives under our own HF org (never microsoft/*), pinned to an immutable SHA,
            // and all three tiers come from the SAME variant repo (only the subdir differs).
            let repo = q4["repo"].as_str().unwrap_or_default();
            assert!(
                repo.starts_with("SceneWorks/Mage-Flow"),
                "{id} pulls from the SceneWorks org mirror, not upstream; got {repo:?}"
            );
            for (name, tier) in [("q8", q8), ("bf16", bf16)] {
                assert_eq!(
                    tier["repo"].as_str(),
                    Some(repo),
                    "{id} {name} ships in the same variant mirror as q4"
                );
            }
            for (name, tier) in [("q4", q4), ("q8", q8), ("bf16", bf16)] {
                assert!(
                    tier["revision"].as_str().is_some_and(is_full_sha_revision),
                    "{id} {name} pins a full 40-hex mirror revision"
                );
                // A tier that reported the dense snapshot's size would mis-quote the download by ~3x
                // and defeat the point of the split, so require a plausible per-tier size.
                let bytes = tier["estimatedSizeBytes"].as_u64().unwrap_or_default();
                assert!(
                    (1..12_000_000_000).contains(&bytes),
                    "{id} {name} must declare its OWN tier size, not the 17.5 GB dense snapshot; \
                     got {bytes}"
                );
            }
            // Physically distinct means physically SMALLER as the tier drops.
            let size = |tier: &serde_json::Value| tier["estimatedSizeBytes"].as_u64().unwrap_or(0);
            assert!(
                size(q4) < size(q8) && size(q8) < size(bf16),
                "{id} tier sizes must be strictly ordered q4 < q8 < bf16 — equal sizes would mean \
                 the tiers are not physically distinct artifacts"
            );

            // ---- shared components (sc-14979) --------------------------------------------------
            for component in ["text_encoder", "vae"] {
                for tier in TIERS {
                    let row = co_reqs
                        .iter()
                        .find(|d| {
                            d["componentId"].as_str() == Some(component)
                                && d["variant"]
                                    .as_str()
                                    .is_some_and(|v| v.eq_ignore_ascii_case(tier))
                        })
                        .unwrap_or_else(|| {
                            panic!("{id} declares the shared {component} co-requisite for {tier}")
                        });
                    assert_eq!(
                        row["repo"].as_str(),
                        Some(COMPONENTS_REPO),
                        "{id} {component}/{tier} resolves from the ONE shared components mirror — a \
                         per-variant copy would restore the 105 GB six-variant install"
                    );
                    assert!(
                        row["revision"].as_str().is_some_and(is_full_sha_revision),
                        "{id} {component}/{tier} pins a full 40-hex revision"
                    );
                    assert_eq!(
                        row["subdir"].as_str(),
                        Some(format!("{tier}/{component}").as_str()),
                        "{id} {component}/{tier} must address its exact component dir — without \
                         `subdir` the whole components snapshot is staged and the engine is handed \
                         the wrong directory"
                    );
                    let files: Vec<&str> = row["files"]
                        .as_array()
                        .unwrap_or_else(|| panic!("{id} {component}/{tier} declares files"))
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .collect();
                    assert_eq!(
                        files,
                        vec![format!("{tier}/{component}/*")],
                        "{id} {component}/{tier} fetches only its own tier's component subtree"
                    );
                }
            }
            // The component ids must be exactly what the mlx_mage descriptor advertises as
            // `required_components`; a typo resolves to no row and fails the job at load.
            let mut ids: Vec<&str> = co_reqs
                .iter()
                .filter_map(|d| d["componentId"].as_str())
                .collect();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(
                ids,
                vec!["text_encoder", "vae"],
                "{id} co-requisites provision exactly the two components mlx_mage requires"
            );
        }
    }

    /// sc-8444 (epic 8431) — Krea Realtime 14B's three download tiers must describe the REAL
    /// `SceneWorks/krea-realtime-14b-mlx` layout, file for file and byte for byte.
    ///
    /// The expectation below is the remote tree as published (read off the HF tree API at authoring
    /// time), transcribed here rather than derived from the manifest — that independence is what
    /// makes this a check instead of a tautology.
    ///
    /// Two failures it exists to catch, both of which produce a SILENT bad install rather than an
    /// error:
    ///
    /// 1. **The bf16 DiT is one level deeper** (`bf16/transformer/dit-0000N-of-00007`) while q4/q8
    ///    keep a flat `dit.safetensors`. A `files: ["bf16/*"]` glob would still DOWNLOAD those
    ///    shards — the worker's `pattern_matches` uses the Rust `glob` crate with default
    ///    `MatchOptions`, where `*` crosses `/` (`scripts/check-download-patterns.mjs` documents the
    ///    same semantics) — but rust-api's cache health (`snapshot_contains_pattern`) treats a glob
    ///    as satisfied by ANY single matching file, and this repo ships no `model_index.json`, so
    ///    the per-component augmentation is a no-op for it. A tier that lost six of its seven shards
    ///    would read `installed` and then die at load. Explicit paths make each file its own
    ///    presence check, so this pins that they stay explicit AND complete.
    /// 2. **`estimatedSizeBytes` must be the tier's real total.** It drives the pre-download size
    ///    the user is shown and the free-space check, and nothing else verifies it.
    #[test]
    fn krea_realtime_tiers_match_the_published_rehost_layout() {
        // The dense companions repeat per tier by design (each tier dir is a self-contained
        // `from_snapshot` tree) and are byte-identical across all three.
        const T5: (&str, u64) = ("t5_encoder.safetensors", 11_361_845_504);
        const VAE: (&str, u64) = ("vae.safetensors", 507_591_212);
        const TOKENIZER: (&str, u64) = ("tokenizer.json", 16_837_417);
        // The 7 sharded bf16 DiT parts, in order.
        const BF16_DIT_SHARDS: [u64; 7] = [
            4_269_218_524,
            4_269_187_488,
            4_269_187_543,
            4_269_204_920,
            4_216_748_302,
            4_216_748_199,
            3_066_800_251,
        ];

        let packed_tier = |dit_bytes: u64| {
            vec![
                ("config.json".to_owned(), 1_557_u64),
                ("dit.safetensors".to_owned(), dit_bytes),
                (T5.0.to_owned(), T5.1),
                (TOKENIZER.0.to_owned(), TOKENIZER.1),
                (VAE.0.to_owned(), VAE.1),
            ]
        };
        let bf16_tier = {
            let mut files = vec![
                ("config.json".to_owned(), 1_496_u64),
                (T5.0.to_owned(), T5.1),
                (TOKENIZER.0.to_owned(), TOKENIZER.1),
                (VAE.0.to_owned(), VAE.1),
            ];
            for (index, bytes) in BF16_DIT_SHARDS.into_iter().enumerate() {
                files.push((
                    format!("transformer/dit-0000{}-of-00007.safetensors", index + 1),
                    bytes,
                ));
            }
            files
        };
        /// One published tier: its variant key, whether it is the default install tier, and its
        /// `(path-within-tier, bytes)` contents.
        struct Tier {
            variant: &'static str,
            is_default: bool,
            files: Vec<(String, u64)>,
        }
        let tiers = [
            Tier {
                variant: "q4",
                is_default: true,
                files: packed_tier(8_378_982_809),
            },
            Tier {
                variant: "q8",
                is_default: false,
                files: packed_tier(15_404_443_762),
            },
            Tier {
                variant: "bf16",
                is_default: false,
                files: bf16_tier,
            },
        ];

        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.models.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.models.jsonc parses as JSON");
        let model = manifest["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|model| model["id"].as_str() == Some("krea_realtime_14b"))
            .expect("krea_realtime_14b present in the builtin models catalog");

        assert_eq!(model["type"].as_str(), Some("video"));
        assert_eq!(
            model["family"].as_str(),
            Some("krea-realtime"),
            "its OWN family — not wan-video; the Wan-LoRA relation lives in \
             extra_compatible_lora_families instead (sc-8444)"
        );
        assert_eq!(
            model["loraCompatibility"]["families"],
            serde_json::json!(["krea-realtime"])
        );

        let downloads = model["downloads"].as_array().expect("downloads array");
        assert_eq!(
            downloads.len(),
            3,
            "exactly the three shipped tiers — no co-requisites, every tier is self-contained"
        );

        for tier in &tiers {
            let (variant, is_default, expected_files) =
                (tier.variant, tier.is_default, &tier.files);
            let entry = downloads
                .iter()
                .find(|d| d["variant"].as_str() == Some(variant))
                .unwrap_or_else(|| panic!("krea_realtime_14b declares a {variant} tier"));

            assert_eq!(entry["provider"].as_str(), Some("huggingface"));
            assert_eq!(
                entry["repo"].as_str(),
                Some("SceneWorks/krea-realtime-14b-mlx"),
                "{variant}: all tiers come from the one turnkey repo"
            );
            assert_eq!(
                entry["revision"].as_str(),
                Some("e68e9a3d98187fdf6936838ffcf6df5aa48d6626"),
                "{variant}: pinned to the published immutable commit"
            );
            assert_eq!(
                entry["platforms"],
                serde_json::json!(["macos"]),
                "{variant}: macOS-only — there is no candle Krea Realtime engine"
            );
            assert_eq!(
                entry["default"].as_bool() == Some(true),
                is_default,
                "{variant}: exactly q4 is the default install tier"
            );

            let mut declared: Vec<&str> = entry["files"]
                .as_array()
                .unwrap_or_else(|| panic!("{variant} declares a files list"))
                .iter()
                .filter_map(serde_json::Value::as_str)
                .collect();
            // Explicit paths, never a glob: a glob is satisfied by one surviving file in the
            // cache-health check, which is how a torn no-`model_index` tier reads `installed`.
            for pattern in &declared {
                assert!(
                    !pattern.contains(['*', '?', '[', ']']),
                    "{variant}: `{pattern}` is a glob — declare explicit paths so every file is \
                     its own presence check and its own download-time claim (sc-12283)"
                );
            }
            let mut want: Vec<String> = expected_files
                .iter()
                .map(|(name, _)| format!("{variant}/{name}"))
                .collect();
            declared.sort_unstable();
            want.sort_unstable();
            assert_eq!(
                declared,
                want.iter().map(String::as_str).collect::<Vec<_>>(),
                "{variant}: declared files must be exactly the published tier contents — in \
                 particular every one of bf16's seven `transformer/` DiT shards"
            );

            let total: u64 = expected_files.iter().map(|(_, bytes)| bytes).sum();
            assert_eq!(
                entry["estimatedSizeBytes"].as_u64(),
                Some(total),
                "{variant}: estimatedSizeBytes must be the tier's real byte total"
            );
            assert_eq!(
                entry["footprint"]["diskSizeBytes"].as_u64(),
                Some(total),
                "{variant}: footprint.diskSizeBytes must agree with estimatedSizeBytes"
            );
            // Memory footprints are MEASURED (sc-8516 / sc-8446); a derived guess must not be
            // parked here, where the RAM-suggestion surface would read it as a measurement.
            assert!(
                entry["footprint"]["residentMemoryBytes"].is_null()
                    && entry["footprint"]["peakMemoryBytes"].is_null(),
                "{variant}: memory footprints stay null until measured on-device (sc-8446)"
            );
        }

        // The three tiers own DISJOINT file sets, so a per-tier delete reclaims that tier's bytes.
        assert_ne!(downloads[0]["files"], downloads[1]["files"]);
        assert_ne!(downloads[0]["files"], downloads[2]["files"]);
        assert_ne!(downloads[1]["files"], downloads[2]["files"]);

        // The turnkey repo is named consistently across the surfaces the worker resolves.
        assert_eq!(
            model["mlx"]["repo"].as_str(),
            Some("SceneWorks/krea-realtime-14b-mlx")
        );
        assert_eq!(
            model["paths"]["model"].as_str(),
            Some("${HF_CACHE}/SceneWorks/krea-realtime-14b-mlx")
        );
        // Q4 is the declared default tier for this dense 14B video engine (sc-10750), matching the
        // `default: true` download above. (The video lane does not READ this key yet — sc-15258.)
        assert_eq!(model["mlx"]["quantize"].as_u64(), Some(4));
    }
}
