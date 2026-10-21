# Tauri updater 2.11.0: zip 9 dependency backport

Owner: SceneWorks maintainers. This is the published updater crate with its
original Apache-2.0 / MIT licenses, source, permissions, and Cargo metadata.

- Archive: https://static.crates.io/crates/tauri-plugin-updater/tauri-plugin-updater-2.11.0.crate
- SHA-256: `b28d8cabdeb0564f03ae261963de4bc3d98321cd3d213e76a81b7d344e5df606`
- Original upstream revision: `6aa2854f314481a459be1189b02c65a2450789ab`
- Upstream zip 9 migration: https://github.com/tauri-apps/plugins-workspace/commit/ba98bfadc4c868c9a7e1a226c83af528e1f11b89

The sole production change is `Cargo.toml`'s Windows `zip` requirement from
`4` to `9`, matching the upstream migration. No Rust or JavaScript source,
updater signature verification, archive handling, features, or permissions
change. `Cargo.toml.orig` remains the original published manifest. The published
crate's independent `Cargo.lock` and Cargo's cache marker are omitted; SceneWorks'
workspace lockfile controls the build. This file records the local patch.

RUSTSEC-2026-0367, RUSTSEC-2026-0368, and RUSTSEC-2026-0369 affect zip versions
before 9. The latest published stable updater (2.14.0 on 2026-10-17) still
requires zip 4; upstream has merged the zip 9 migration but has not published it.
Vendoring the version SceneWorks already uses avoids unrelated updater changes.
The root `[patch.crates-io]` selects this copy on every platform; Windows is the
platform that compiles its zip dependency.

Validate with `cargo deny --locked check advisories` and desktop Rust tests and
lint on Linux/macOS/Windows. `apps/desktop/tests/zip_archive_safety.rs` exercises
normal extraction, privileged permission stripping, and symlink confinement
against the same zip dependency (the last two cases require Unix). Preserve
the source diff and checksum when refreshing this directory. Remove the patch
when a published updater containing the zip 9 migration is adopted and its
platform tests pass. Do not add advisory exemptions for the old zip versions.
