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
    /// Overwrite an existing manifest unconditionally. The desktop seeds this way on
    /// every launch so the builtin catalog tracks the app version; user customizations
    /// live in the separate `user.*.jsonc` files, which seeding never touches.
    Overwrite,
    /// Keep the on-disk builtin manifests in sync with the binary's embedded copy,
    /// rewriting a file only when it is missing OR its bytes have drifted from the
    /// embedded copy. The API seeds this way when its config dir is EXPLICIT
    /// (`SCENEWORKS_CONFIG_DIR` set — a repo checkout or a Compose/RunPod bind mount /
    /// persistent volume).
    ///
    /// The `builtin.*.jsonc` files are app-owned: nothing edits them at runtime (every
    /// install/hide/customization writes `user.*.jsonc` or a project manifest instead),
    /// so the on-disk copy is purely a materialized cache of the embedded bytes. A copy
    /// that no longer matches the running binary is therefore always stale — e.g. a
    /// persisted seed left untouched across a binary upgrade, the exact failure that hid
    /// the sc-10193 img2img flag AND the Krea Turbo memory-ladder curves (a directly
    /// launched API kept serving a months-old `builtin.models.jsonc`) — and must be
    /// refreshed. A byte-identical file is left untouched so a matching repo checkout is
    /// never dirtied and the API's mtime-keyed manifest cache is not needlessly busted.
    /// Operator customizations belong in `user.*.jsonc`, which this never writes.
    /// (sc-10212; see `seed_mode_for_config_dir` in rust-api.)
    SyncFromEmbedded,
    /// Fill only genuinely-missing manifests; never touch a file that already exists.
    /// This is the OPT-IN escape hatch for a fully operator-provisioned config dir — a
    /// deployment that intentionally ships its OWN `builtin.*.jsonc` (e.g. a Compose bind
    /// mount of a customized catalog, or the contract-snapshot test harness) and wants it
    /// used verbatim. Unlike [`SyncFromEmbedded`] it does NOT self-heal drift: the operator
    /// owns these files and is responsible for keeping them current, so a stale copy is
    /// preserved rather than refreshed. Never selected by default — the API reaches it only
    /// when the operator sets the explicit opt-in env (see `seed_mode_for_config_dir`).
    IfMissing,
}

/// Whether the seed must (re)write `target` for this `embedded` copy under `mode`.
///
/// `Overwrite` always writes. `SyncFromEmbedded` writes only when the on-disk copy is
/// absent or has drifted from the embedded bytes, so an up-to-date file (a matching repo
/// checkout, or an already-current seed) is left untouched while a stale seed left by an
/// older binary is refreshed in place; an unreadable file counts as drifted → rewrite.
/// `IfMissing` writes only when the file is absent, preserving any operator-provided copy.
/// Pure so the decision is unit-tested without touching the filesystem.
fn manifest_needs_write(existing: Option<&[u8]>, embedded: &[u8], mode: SeedMode) -> bool {
    match mode {
        SeedMode::Overwrite => true,
        SeedMode::SyncFromEmbedded => existing != Some(embedded),
        SeedMode::IfMissing => existing.is_none(),
    }
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
/// cannot leave a truncated manifest that parses to an empty/broken catalog: under
/// [`SeedMode::SyncFromEmbedded`] a truncated/drifted file no longer matches the
/// embedded copy and is rewritten on the next seed rather than skipped.
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
        let existing = std::fs::read(&target).ok();
        if !manifest_needs_write(existing.as_deref(), contents.as_bytes(), mode) {
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

    /// A fresh Eros install must carry the external Gemma encoder required by both native backends.
    #[test]
    fn eros_fresh_install_provisions_the_shared_gemma_encoder() {
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.models.jsonc"));
        let manifest: serde_json::Value = serde_json::from_str(&stripped).expect("manifest parses");
        let eros = manifest["models"]
            .as_array()
            .expect("models")
            .iter()
            .find(|model| model["id"] == "ltx_2_3_eros")
            .expect("Eros model");
        let gemma = eros["downloads"]
            .as_array()
            .expect("downloads")
            .iter()
            .find(|download| {
                download["repo"] == "SceneWorks/ltx-2.3-mlx" && download["coRequisite"] == true
            })
            .expect("Eros must install the shared Gemma bundle on a clean machine");
        assert_eq!(gemma["files"], serde_json::json!(["gemma/*"]));
        assert_eq!(gemma["revision"].as_str().map(str::len), Some(40));
        for platform in ["macos", "windows", "linux"] {
            assert!(
                gemma["platforms"]
                    .as_array()
                    .is_some_and(|values| values.iter().any(|value| value == platform)),
                "Gemma co-requisite must provision {platform}"
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
        seed_builtin_manifests(temp.path(), SeedMode::SyncFromEmbedded).expect("seeding succeeds");

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
    fn manifest_needs_write_refreshes_missing_or_drifted_only_when_syncing() {
        let embedded = b"embedded-bytes";
        // Overwrite always rewrites, regardless of what is on disk.
        assert!(manifest_needs_write(None, embedded, SeedMode::Overwrite));
        assert!(manifest_needs_write(
            Some(embedded),
            embedded,
            SeedMode::Overwrite
        ));
        // SyncFromEmbedded: missing or drifted (incl. an unreadable file → None) is rewritten;
        // a byte-identical file is left untouched so a matching checkout is not dirtied.
        assert!(manifest_needs_write(
            None,
            embedded,
            SeedMode::SyncFromEmbedded
        ));
        assert!(manifest_needs_write(
            Some(b"stale-seed"),
            embedded,
            SeedMode::SyncFromEmbedded
        ));
        assert!(!manifest_needs_write(
            Some(embedded),
            embedded,
            SeedMode::SyncFromEmbedded
        ));
        // IfMissing: fill only a genuinely-absent file; preserve any provided copy (even if it
        // drifts from the embedded bytes — the operator owns it).
        assert!(manifest_needs_write(None, embedded, SeedMode::IfMissing));
        assert!(!manifest_needs_write(
            Some(b"operator-provided"),
            embedded,
            SeedMode::IfMissing
        ));
        assert!(!manifest_needs_write(
            Some(embedded),
            embedded,
            SeedMode::IfMissing
        ));
    }

    #[test]
    fn if_missing_preserves_a_provided_manifest_and_fills_only_the_gaps() {
        let temp = tempfile::tempdir().expect("temp dir");
        let dir = temp.path().join("manifests");
        std::fs::create_dir_all(&dir).expect("create manifests dir");
        // A fully operator-provisioned config dir intentionally ships its OWN builtin.models.jsonc
        // (the sc-15504 opt-out via SCENEWORKS_OWN_MANIFESTS; also how the contract-snapshot harness
        // injects a synthetic catalog). IfMissing must use it verbatim, drift from the embedded copy
        // and all.
        let provided = dir.join("builtin.models.jsonc");
        let provided_body = "{ \"models\": [ { \"id\": \"operator-model\" } ] }";
        std::fs::write(&provided, provided_body).expect("seed provided");

        seed_builtin_manifests(temp.path(), SeedMode::IfMissing).expect("seeding succeeds");

        assert_eq!(
            std::fs::read_to_string(&provided).expect("read provided"),
            provided_body,
            "IfMissing never overwrites an operator-provided manifest"
        );
        // ...while genuinely-missing manifests are still filled from the embedded copy.
        assert_eq!(
            std::fs::read_to_string(dir.join("builtin.styles.jsonc")).expect("styles written"),
            embedded("builtin.styles.jsonc")
        );
    }

    #[test]
    fn sync_refreshes_a_drifted_manifest_but_leaves_an_identical_one_untouched() {
        let temp = tempfile::tempdir().expect("temp dir");
        let dir = temp.path().join("manifests");
        std::fs::create_dir_all(&dir).expect("create manifests dir");

        // A stale seed left by an older binary (e.g. a persisted RunPod volume without the Krea
        // Turbo `turboFit` curves) — its bytes differ from the embedded copy.
        let drifted = dir.join("builtin.models.jsonc");
        std::fs::write(&drifted, "{ \"models\": [] } // months-old seed").expect("seed drifted");
        // An already-current copy, byte-identical to the embedded manifest.
        let current = dir.join("builtin.loras.jsonc");
        std::fs::write(&current, embedded("builtin.loras.jsonc")).expect("seed current");
        let current_mtime = std::fs::metadata(&current)
            .and_then(|meta| meta.modified())
            .expect("current mtime");

        seed_builtin_manifests(temp.path(), SeedMode::SyncFromEmbedded).expect("seeding succeeds");

        // The drifted manifest is refreshed to the embedded copy — the whole point of the fix.
        assert_eq!(
            std::fs::read_to_string(&drifted).expect("read refreshed"),
            embedded("builtin.models.jsonc"),
            "SyncFromEmbedded refreshes a manifest that drifted from the running binary"
        );
        // The byte-identical manifest is left untouched: same content, and the seed did not rewrite
        // it (mtime unchanged), so a matching checkout is never dirtied and the mtime-keyed cache holds.
        assert_eq!(
            std::fs::read_to_string(&current).expect("read current"),
            embedded("builtin.loras.jsonc")
        );
        assert_eq!(
            std::fs::metadata(&current)
                .and_then(|meta| meta.modified())
                .expect("current mtime after"),
            current_mtime,
            "an already-current manifest is not rewritten"
        );
        // Genuinely-missing manifests are still filled in.
        assert_eq!(
            std::fs::read_to_string(dir.join("builtin.styles.jsonc")).expect("styles written"),
            embedded("builtin.styles.jsonc")
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

    /// The four DIFFUSERS turbo files published by `lightx2v/Minimax-h3-Turbo`, paired with the
    /// catalog id that must carry each. Transcribed from the real published headers at revision
    /// `5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1` (sc-18724 verified all four).
    const MINIMAX_H3_TURBO_FILES: [(&str, &str); 4] = [
        (
            "minimax_h3_turbo_4step_768p",
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
        ),
        (
            "minimax_h3_turbo_8step",
            "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
        ),
        (
            "minimax_h3_turbo_4step_v01",
            "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
        ),
        (
            "minimax_h3_ref2v_turbo_4step",
            "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
        ),
    ];

    fn builtin_loras() -> Vec<serde_json::Value> {
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.loras.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.loras.jsonc parses as JSON");
        manifest["loras"]
            .as_array()
            .expect("builtin.loras.jsonc has a loras array")
            .clone()
    }

    #[test]
    fn minimax_h3_turbo_loras_are_registered_and_sha_pinned() {
        // sc-18725 (epic 17137). All FOUR published diffusers checkpoints register as weight-load-only
        // accelerators. Pin every non-default field so this discriminates a real registration from an
        // empty or renamed one, and assert the COUNT so a fifth entry (or a dropped fourth — the
        // story's own table listed only three until sc-18724 found the ref2v file) has to be
        // deliberate.
        let loras = builtin_loras();
        let registered: Vec<&serde_json::Value> = loras
            .iter()
            .filter(|lora| lora["family"] == serde_json::json!("minimax-h3"))
            .collect();
        assert_eq!(
            registered.len(),
            MINIMAX_H3_TURBO_FILES.len(),
            "expected exactly {} minimax-h3 catalog LoRAs; got {:?}",
            MINIMAX_H3_TURBO_FILES.len(),
            registered
                .iter()
                .map(|lora| lora["id"].clone())
                .collect::<Vec<_>>()
        );

        for (id, file) in MINIMAX_H3_TURBO_FILES {
            let lora = loras
                .iter()
                .find(|lora| lora["id"] == serde_json::json!(id))
                .unwrap_or_else(|| panic!("{id} is registered in builtin.loras.jsonc"));

            assert_eq!(lora["family"], serde_json::json!("minimax-h3"), "{id}");
            assert_eq!(lora["role"], serde_json::json!("accelerator"), "{id}");
            assert_eq!(
                lora["compatibility"]["families"],
                serde_json::json!(["minimax-h3"]),
                "{id}"
            );
            // The runtime `lora_scale` multiplier, NOT alpha. Anything but 1.0 would silently rescale
            // the file's own alpha/rank fold (sc-18724).
            assert_eq!(lora["defaultWeight"], serde_json::json!(1.0), "{id}");
            assert_eq!(
                lora["source"]["provider"],
                serde_json::json!("huggingface"),
                "{id}"
            );
            assert_eq!(
                lora["source"]["repo"],
                serde_json::json!("lightx2v/Minimax-h3-Turbo"),
                "{id}"
            );
            assert_eq!(lora["source"]["file"], serde_json::json!(file), "{id}");
            let revision = lora["source"]["revision"]
                .as_str()
                .unwrap_or_else(|| panic!("{id} pins a source.revision"));
            assert!(
                is_full_sha_revision(revision),
                "{id} must pin a full 40-hex commit SHA (a floating `main` would drift the \
                 accelerator weights under a pinned DiT); got {revision:?}"
            );
            assert_eq!(revision, "5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1", "{id}");
        }
    }

    #[test]
    fn no_minimax_h3_lora_names_a_comfyui_export() {
        // 🔴 sc-18724 / sc-18725. `lightx2v/Minimax-h3-Turbo` publishes seven files: four diffusers
        // and three `_comfyui_` twins. The ComfyUI exports fuse q/k/v into `attn.qkv_proj` and swap
        // the SwiGLU halves in `mlp.fc1`, so folding one is shape-valid and numerically WRONG — the
        // sc-18740 class that shipped green at cosine 0.73-0.78. The engine refuses them by design,
        // which makes a `_comfyui_` filename here a HARD ERROR at install rather than a degradation.
        //
        // Deliberately a SUBSTRING scan over every registered file rather than a re-assertion of the
        // four expected names. The sibling test above pins those names AND the count, so it catches
        // both a swap and a bare fifth entry — but it checks only the names listed in
        // `MINIMAX_H3_TURBO_FILES`, so the moment that table is legitimately extended to cover a new
        // file, it vouches for whatever name was written into it. This scan keeps failing on any
        // `_comfyui_` file that ever appears under this family, which is the property that matters.
        for lora in builtin_loras() {
            if lora["family"] != serde_json::json!("minimax-h3") {
                continue;
            }
            let id = lora["id"].as_str().unwrap_or("<no id>");
            let file = lora["source"]["file"].as_str().unwrap_or_default();
            assert!(
                !file.contains("_comfyui_"),
                "{id} names the ComfyUI export {file:?}; the engine REFUSES that key space \
                 (fused qkv_proj + swapped SwiGLU halves). Use the diffusers twin."
            );
        }
    }

    /// sc-19563 — **every MiniMax-H3 turbo entry declares the ONE partition it is distilled for**,
    /// and the ref2v one is the odd one out.
    ///
    /// This reads the embedded manifest, so it is the guard on the declaration itself; the gate that
    /// reads it lives in `apps/rust-api/src/loras.rs::validate_lora_specs_for_model` and is proved
    /// reachable from a real submission by
    /// `jobs::cross_selecting_a_minimax_h3_partition_lora_is_refused_by_the_video_job_route`.
    ///
    /// The `assert_ne!` at the end is the half that matters. Family membership cannot express this
    /// pairing — both partitions are one architecture and both declare `family: minimax-h3` — so a
    /// table that gave all four the same `modelIds` would be exactly as wrong as declaring none, and
    /// would sail past a test that only checked the key was present.
    #[test]
    fn every_minimax_h3_turbo_lora_declares_its_partition() {
        // The expected pairing, spelled out rather than derived from the filename, so a manifest
        // edit that pointed the ref2v adapter at the fl2v partition reds here.
        const EXPECTED: [(&str, &str); 4] = [
            ("minimax_h3_turbo_4step_768p", "minimax_h3"),
            ("minimax_h3_turbo_8step", "minimax_h3"),
            ("minimax_h3_turbo_4step_v01", "minimax_h3"),
            ("minimax_h3_ref2v_turbo_4step", "minimax_h3_ref"),
        ];
        let mut seen = 0;
        for lora in builtin_loras() {
            if lora["family"] != serde_json::json!("minimax-h3") {
                continue;
            }
            let id = lora["id"].as_str().expect("a LoRA entry has an id");
            let want = EXPECTED
                .iter()
                .find(|(known, _)| *known == id)
                .unwrap_or_else(|| panic!("unlisted minimax-h3 LoRA {id}; add it to EXPECTED"))
                .1;
            let declared: Vec<&str> = lora["modelIds"]
                .as_array()
                .unwrap_or_else(|| {
                    panic!(
                        "{id} declares no `modelIds`. Both H3 partitions share \
                         `family: minimax-h3`, so without this the adapter attaches to either and \
                         folds CLEANLY at the wrong quality — a mismatch, not a failure (sc-19563)."
                    )
                })
                .iter()
                .filter_map(serde_json::Value::as_str)
                .collect();
            assert_eq!(declared, vec![want], "{id} names the wrong partition");
            seen += 1;
        }
        assert_eq!(seen, EXPECTED.len(), "every listed entry must be present");
        // The two partitions really are addressed differently — a uniform table would be as wrong
        // as no table at all, and would pass a mere presence check.
        assert_ne!(
            EXPECTED[0].1, EXPECTED[3].1,
            "the fl2v and ref2v adapters must name DIFFERENT partitions"
        );
    }

    #[test]
    fn no_minimax_h3_lora_declares_an_alpha() {
        // 🔴 sc-18724 / sc-18725. Alpha differs PER FILE inside this one repo — 128, 8, 8, and absent
        // — so a family-wide (or even per-entry) manifest alpha would be wrong for three of the four.
        // The engine resolves it from the file itself: rank from the factor shapes, alpha by
        // precedence, never the rank. A manifest that pinned one would re-introduce exactly the 16x /
        // 128x overshoot sc-18724 exists to close, and it would do so SILENTLY.
        //
        // The `lora-manifest.schema.json` item is `additionalProperties: false` and has no alpha key,
        // so today this is unrepresentable — this test is the guard on that staying true, since
        // relaxing the schema is a one-line edit and the failure it enables is invisible in output.
        const ALPHA_SPELLINGS: [&str; 6] = [
            "alpha",
            "loraAlpha",
            "lora_alpha",
            "networkAlpha",
            "network_alpha",
            "scale",
        ];
        for lora in builtin_loras() {
            if lora["family"] != serde_json::json!("minimax-h3") {
                continue;
            }
            let id = lora["id"].as_str().unwrap_or("<no id>");
            let object = lora.as_object().expect("a LoRA entry is an object");
            for spelling in ALPHA_SPELLINGS {
                assert!(
                    !object.contains_key(spelling),
                    "{id} declares `{spelling}`. Alpha is PER FILE (128 / 8 / 8 / absent) and is \
                     resolved by the engine from the checkpoint; declaring one here is wrong for \
                     three of the four files."
                );
                assert!(
                    lora["source"].get(spelling).is_none(),
                    "{id} declares `source.{spelling}`; see above."
                );
                assert!(
                    lora["compatibility"].get(spelling).is_none(),
                    "{id} declares `compatibility.{spelling}`; see above."
                );
            }
        }
    }

    /// Every path a MiniMax-H3 load OPENS, expressed as a snapshot-relative path prefix, for `tier`.
    ///
    /// Read entirely from `mlx_tier_completeness` — the same constants
    /// `minimax_h3_shared_is_complete` and `resolve_minimax_h3_load` gate on at job time. A second
    /// hand-copied list here would drift from the runtime check exactly the way the catalog drifted
    /// from it before sc-19573, which is the whole defect this guard exists to prevent.
    fn minimax_h3_probed_paths(tier: &str) -> Vec<String> {
        use crate::mlx_tier_completeness as tc;
        let mut paths = Vec::new();
        // Both DiT partitions, under the TIER root. Not one of them — the engine opens
        // `transformer/config.json` and `transformer_ref/config.json` on every load.
        for (_, partition) in tc::MINIMAX_H3_PARTITIONS {
            paths.push(format!("{tier}/{partition}"));
        }
        // The text encoder, at the root this tier actually reads it from (sc-19120). Both roots are
        // EMPTY here so the resolver yields the snapshot-relative remainder — `q4/text_encoder` for a
        // packed tier, plain `text_encoder` for bf16 — which is the form a manifest `files` pattern
        // is written in. Passing a real root would produce an absolute path no pattern can match, and
        // passing the tier as the root would double it.
        paths.push(
            tc::minimax_h3_text_encoder_dir(
                std::path::Path::new(""),
                std::path::Path::new(""),
                tier,
            )
            .to_string_lossy()
            .into_owned(),
        );
        for component in tc::MINIMAX_H3_SHARED_PROBED_DIRS {
            paths.push(component.to_owned());
        }
        for file in tc::MINIMAX_H3_AUDIO_VAE_CONFIG_FILES {
            paths.push(format!("{}/{file}", tc::MINIMAX_H3_AUDIO_VAE_CONFIG_DIR));
        }
        paths
    }

    /// The snapshot-relative paths `entry`'s downloads FETCH for `tier` — every non-co-requisite row
    /// whose `variant` matches, plus every co-requisite row that applies (tier-agnostic ones always
    /// apply; `variant`-scoped ones only for their own tier), exactly as
    /// `model_co_requisite_downloads_for_variant` selects them at install time.
    fn minimax_h3_declared_files(entry: &serde_json::Value, tier: &str) -> Vec<String> {
        entry["downloads"]
            .as_array()
            .expect("downloads array")
            .iter()
            .filter(|download| {
                match download["variant"].as_str() {
                    Some(variant) => variant.eq_ignore_ascii_case(tier),
                    // A tier-agnostic row applies to every tier — but only co-requisites are ever
                    // tier-agnostic here; a primary without a variant would be a different shape.
                    None => download["coRequisite"].as_bool() == Some(true),
                }
            })
            .flat_map(|download| {
                download["files"]
                    .as_array()
                    .expect("files array")
                    .iter()
                    .map(|file| file.as_str().expect("file pattern is a string").to_owned())
            })
            .collect()
    }

    /// The probed paths `entry` does NOT fetch for `tier`. Empty ⇒ the install this entry produces
    /// carries every path the loader opens.
    fn minimax_h3_unfetched_paths(entry: &serde_json::Value, tier: &str) -> Vec<String> {
        let declared = minimax_h3_declared_files(entry, tier);
        minimax_h3_probed_paths(tier)
            .into_iter()
            .filter(|probed| {
                // A pattern covers a probed path when it names the path itself (`FL2VA/audio_vae/…`)
                // or fetches into it (`q4/transformer/*`). Prefix-matched on a trailing `/` so
                // `q4/transformer/*` cannot be read as covering `q4/transformer_ref` — the exact
                // confusion `minimax_h3_ref` being a PREFIX EXTENSION of `minimax_h3` invites.
                !declared
                    .iter()
                    .any(|pattern| pattern == probed || pattern.starts_with(&format!("{probed}/")))
            })
            .collect()
    }

    #[test]
    fn minimax_h3_downloads_cover_every_path_the_loader_probes() {
        // sc-19573. A `coRequisite` is a PRE-DOWNLOAD WEIGHTS FLOOR, so an entry that omits a path
        // the loader opens is declaring an install that cannot load. Before this guard the catalog
        // omitted four: the sibling DiT partition (at every tier — the engine opens both), the
        // `tokenizer/` directory, and the three `FL2VA/audio_vae/` constructor documents.
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.models.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.models.jsonc parses as JSON");
        let models = manifest["models"].as_array().expect("models array");
        let entry = |id: &str| {
            models
                .iter()
                .find(|model| model["id"] == serde_json::json!(id))
                .unwrap_or_else(|| panic!("{id} is present"))
                .clone()
        };

        for id in ["minimax_h3", "minimax_h3_ref"] {
            let model = entry(id);
            for tier in ["q4", "q8", "bf16"] {
                assert!(
                    minimax_h3_unfetched_paths(&model, tier).is_empty(),
                    "{id} @ {tier}: installing this entry does not fetch {:?}, which \
                     mlx-gen-minimax-h3::load opens — the install would fail at load",
                    minimax_h3_unfetched_paths(&model, tier)
                );
            }
        }

        // MUTATION, one probed path at a time: removing the file PATTERNS that fetch a single path
        // must make THAT path — and only that path — the reported gap. Per pattern rather than per
        // row, because the three `FL2VA/audio_vae/` documents share one row: deleting the row would
        // move three paths at once and prove only that the guard notices a big hole. Deleting
        // everything at once would prove even less.
        for id in ["minimax_h3", "minimax_h3_ref"] {
            for tier in ["q4", "q8", "bf16"] {
                for probed in minimax_h3_probed_paths(tier) {
                    let mut mutated = entry(id);
                    let mut removed = 0usize;
                    for download in mutated["downloads"]
                        .as_array_mut()
                        .expect("downloads array")
                    {
                        let files = download["files"].as_array_mut().expect("files array");
                        let before = files.len();
                        files.retain(|file| {
                            let file = file.as_str().expect("file pattern");
                            !(file == probed || file.starts_with(&format!("{probed}/")))
                        });
                        removed += before - files.len();
                    }
                    assert!(
                        removed > 0,
                        "{id} @ {tier}: nothing fetched {probed}, so the mutation is vacuous"
                    );
                    assert_eq!(
                        minimax_h3_unfetched_paths(&mutated, tier),
                        vec![probed.clone()],
                        "{id} @ {tier}: removing the patterns that fetch {probed} must report \
                         exactly that path as unfetched"
                    );
                }
            }
        }
    }

    #[test]
    fn minimax_h3_sibling_partition_is_a_per_tier_co_requisite_of_both_entries() {
        // The half of sc-19573 a path-coverage check alone cannot express: the sibling partition must
        // arrive as a `coRequisite` (which `install_state_for` GATES on and the download job queues
        // alongside the primary), not merely as some row that happens to match. A `variant` is
        // required on each so a q4 user is not handed the 66 GB bf16 sibling, and so
        // `install_state_for`'s tier-scoped aggregate can judge one tier's pair at a time.
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.models.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.models.jsonc parses as JSON");
        let models = manifest["models"].as_array().expect("models array");
        let partitions = crate::mlx_tier_completeness::MINIMAX_H3_PARTITIONS;

        for (id, own) in partitions {
            let sibling = partitions
                .iter()
                .find(|(other, _)| *other != id)
                .map(|(_, partition)| *partition)
                .expect("the pair has two members");
            assert_ne!(
                own, sibling,
                "{id}: the sibling must be the OTHER partition"
            );
            let model = models
                .iter()
                .find(|model| model["id"] == serde_json::json!(id))
                .unwrap_or_else(|| panic!("{id} is present"));
            let downloads = model["downloads"].as_array().expect("downloads array");

            for tier in ["q4", "q8", "bf16"] {
                let row = downloads
                    .iter()
                    .find(|download| {
                        download["files"] == serde_json::json!([format!("{tier}/{sibling}/*")])
                    })
                    .unwrap_or_else(|| {
                        panic!("{id} @ {tier}: no row fetches the sibling {sibling} partition")
                    });
                assert_eq!(
                    row["coRequisite"].as_bool(),
                    Some(true),
                    "{id} @ {tier}: the sibling partition must be a coRequisite — a plain row is \
                     neither queued alongside the primary nor gated on by install state"
                );
                assert!(
                    row["variant"]
                        .as_str()
                        .is_some_and(|variant| variant.eq_ignore_ascii_case(tier)),
                    "{id} @ {tier}: the sibling co-requisite must be variant-scoped to its tier"
                );
                assert!(
                    row["required"].as_str() != Some("soft"),
                    "{id} @ {tier}: the sibling partition is a HARD dependency — the engine opens \
                     both partitions on every load, so there is no usable-without-it state for a \
                     soft co-requisite to preserve"
                );
                // A co-requisite must NOT also be the entry's own primary — that would make the
                // pair's install state depend on a row the download job never queues as a tier.
                assert!(
                    row["default"].as_bool() != Some(true),
                    "{id} @ {tier}: the sibling co-requisite must never be the default tier"
                );
            }
        }
    }

    #[test]
    fn both_minimax_h3_partitions_advertise_the_minimax_h3_lora_family() {
        // sc-18725. `loraCompatibility.families` is the LOAD gate the API
        // (`validate_lora_specs_for_model`) and the web picker (`loraMatchesModel`) both read. Both
        // partitions must declare it: the gate is per-model-id, and the ref2v adapter is distilled
        // for `minimax_h3_ref` specifically, so omitting it there would leave that adapter
        // unselectable on the only entry it belongs to.
        //
        // Also pins the ABSENCE of "acceleration" from `types`. That is the single token
        // `modelSupportsMultiPhase` reads to open Image Studio's multi-phase editor, whose worker gate
        // is `request.model == KREA_RAW_MODEL_ID` — advertising it from a video model would surface a
        // lane no worker serves. Carrying `role: accelerator` LoRAs does NOT imply that compat type.
        let stripped = crate::jsonc::strip_jsonc_comments(embedded("builtin.models.jsonc"));
        let manifest: serde_json::Value =
            serde_json::from_str(&stripped).expect("builtin.models.jsonc parses as JSON");
        let models = manifest["models"]
            .as_array()
            .expect("builtin.models.jsonc has a models array");

        for id in ["minimax_h3", "minimax_h3_ref"] {
            let model = models
                .iter()
                .find(|model| model["id"] == serde_json::json!(id))
                .unwrap_or_else(|| panic!("{id} is present"));
            assert_eq!(
                model["loraCompatibility"]["families"],
                serde_json::json!(["minimax-h3"]),
                "{id} must advertise the minimax-h3 LoRA family"
            );
            let types = model["loraCompatibility"]["types"]
                .as_array()
                .unwrap_or_else(|| panic!("{id} declares loraCompatibility.types"));
            assert!(
                !types.contains(&serde_json::json!("acceleration")),
                "{id} must NOT advertise the acceleration compat type: it opens the image \
                 multi-phase editor, which only krea_2_raw has a worker lane for. Got {types:?}"
            );
        }
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

        let mut prev_peak: Option<u64> = None;
        let mut q4_baseline: Option<(u64, u64, u64)> = None;
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
            // Memory footprints are grounded in an on-device MEASUREMENT (sc-8446 ran the Q4 tier at
            // 832x480 / 81 frames on an M5 Max); the Q8/bf16 rows are that measurement plus the exact
            // DiT byte-size difference, which is the only term that changes between tiers. They are no
            // longer null — but they still must not be arbitrary, so the relationships that make them
            // a derivation rather than a guess are pinned here.
            let resident = entry["footprint"]["residentMemoryBytes"]
                .as_u64()
                .unwrap_or_else(|| panic!("{variant}: residentMemoryBytes is measured (sc-8446)"));
            let peak = entry["footprint"]["peakMemoryBytes"]
                .as_u64()
                .unwrap_or_else(|| panic!("{variant}: peakMemoryBytes is measured (sc-8446)"));
            assert!(
                peak > resident,
                "{variant}: peak ({peak}) must exceed the steady-state resident ({resident}) — the \
                 terminal VAE decode is what sets the peak"
            );
            // Every tier carries the same ~7.14 GiB bf16 KV cache and the same companions, so the
            // resident figure must sit above the tier's own DiT bytes and below its disk total.
            let dit_bytes = expected_files
                .iter()
                .find(|(name, _)| {
                    name.ends_with("dit.safetensors") || name.contains("transformer/")
                })
                .map(|_| {
                    expected_files
                        .iter()
                        .filter(|(name, _)| {
                            name.ends_with("dit.safetensors") || name.contains("transformer/")
                        })
                        .map(|(_, b)| b)
                        .sum::<u64>()
                })
                .expect("every tier ships a DiT");
            assert!(
                resident > dit_bytes,
                "{variant}: resident ({resident}) must exceed the DiT alone ({dit_bytes}) — the KV \
                 cache and VAE are resident too"
            );
            // And the ladder must be monotonic in tier size, which is what makes bf16 the tier that
            // sets `mlx.minMemoryGb`.
            if let Some(prev) = prev_peak {
                assert!(
                    peak > prev,
                    "{variant}: the memory ladder must increase with tier size (got {peak} after \
                     {prev})"
                );
            }
            prev_peak = Some(peak);

            // THE DERIVATION ITSELF. Only Q4 was run on-device; Q8/bf16 are that measurement plus the
            // exact DiT byte difference, because the DiT is the only term that changes between tiers
            // (the KV cache holds bf16 activations and the companions are byte-identical). Monotonicity
            // alone does not pin that — any increasing triple passes it — so assert the actual
            // relationship: every tier's delta from Q4 must equal its DiT delta from Q4. A 64 MiB
            // tolerance absorbs the rounding in the committed values without admitting a guess.
            match q4_baseline {
                None => q4_baseline = Some((resident, peak, dit_bytes)),
                Some((q4_resident, q4_peak, q4_dit)) => {
                    let want = dit_bytes as i128 - q4_dit as i128;
                    const TOL: i128 = 64 * 1024 * 1024;
                    for (label, got) in [
                        ("resident", resident as i128 - q4_resident as i128),
                        ("peak", peak as i128 - q4_peak as i128),
                    ] {
                        assert!(
                            (got - want).abs() <= TOL,
                            "{variant}: {label} is {got} B above Q4 but its DiT is only {want} B \
                             bigger — the documented derivation is measured-Q4 + the exact DiT delta, \
                             so this value is not that derivation (drift {} B, tolerance {TOL} B)",
                            (got - want).abs()
                        );
                    }
                }
            }
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
