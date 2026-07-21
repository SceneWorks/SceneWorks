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

    fn embedded(name: &str) -> &'static str {
        BUILTIN_MANIFESTS
            .iter()
            .find(|(file, _)| *file == name)
            .map(|(_, contents)| *contents)
            .expect("manifest present in BUILTIN_MANIFESTS")
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
        (
            "qwen_image",
            "SceneWorks/qwen-image-2512-fun-controlnet-union",
        ),
        ("ltx_2_3", "SceneWorks/ltx-2.3-mlx"),
        (
            "ltx_2_3_eros",
            "TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments",
        ),
        ("wan_2_2_t2v_14b", "lightx2v/Wan2.2-Lightning"),
        ("wan_2_2_i2v_14b", "lightx2v/Wan2.2-Lightning"),
        ("pid_qwenimage", "SceneWorks/gemma-2-2b-it"),
        ("pid_flux", "SceneWorks/gemma-2-2b-it"),
        ("pid_flux2", "SceneWorks/gemma-2-2b-it"),
        ("pid_sdxl", "SceneWorks/gemma-2-2b-it"),
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

        let unpinned: ModelDownload = serde_json::from_value(serde_json::json!({
            "provider": "huggingface",
            "repo": "black-forest-labs/FLUX.1-dev",
            "files": [],
        }))
        .expect("unpinned entry deserializes");
        assert_eq!(unpinned.revision, None);
        assert!(
            serde_json::to_value(&unpinned)
                .expect("re-serialize")
                .get("revision")
                .is_none(),
            "an unpinned download must not serialize a revision key (main-branch default)"
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
}
