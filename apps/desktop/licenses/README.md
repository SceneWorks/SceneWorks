# Bundled third-party notices

`manifest.json` and the license files below it are the authoritative corpus for
SceneWorks' third-party notices. The desktop web build imports this corpus
directly, so **About → Licenses** is available inside every signed bundle
without a filesystem or network dependency.

Tauri's `bundle.licenseFile` is intentionally not used for this corpus. In
Tauri 2 it is a single package-license file included only in the bundle formats
that support one; it is not a multi-component notice facility and it does not
drive SceneWorks' About screen. SceneWorks' own package license is already
declared as `AGPL-3.0-or-later`. Adding `licenseFile` would therefore create a
second, platform-dependent copy without replacing the in-app notices.

`scripts/check-license-coverage.mjs` verifies that every manifest document is
wired into the web build. It also checks
`config/inference-third-party-source.json`, the audited inventory of ported or
compile-time-embedded source/data in the separately versioned inference
repository. The inventory revision must match SceneWorks' Cargo pins. When
bumping inference, audit its `NOTICE`, `LICENSE-*`, and production
`include_str!`/`include_bytes!` sites, update the inventory, and add every real
case to this corpus.
