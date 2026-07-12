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
