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
        // sc-13402 (epic 13400): the shipped catalog the API serves carries the five live
        // audio providers as first-class `type: "audio"` entries, each with a populated
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
