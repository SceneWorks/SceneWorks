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
case to this corpus. `artifacts` and `includeSites` describe only the exact
pinned revision; `prospectiveDisclosures` records notices intentionally staged
for a later revision without falsely claiming that source is already shipped.
The inventory digest makes additions, deletions, and changed dispositions fail
until the complete audit is intentionally regenerated.

Two inventories back that audit, and they fail in different directions.
`config/inference-provenance-candidates.tsv` is produced by a marker regex over
doc comments — a heuristic, and one with a proven hole: `mlx-gen-krea-realtime`
described itself honestly as a port in words the regex did not know, matched
nothing, and shipped invisible to the audit (sc-15138). So
`config/inference-crate-prefixes.txt` lists every production-Rust **crate** in
the pinned revision without reading a byte of source, and every prefix in it
must be classified — covered by a `portedSourceAreas` entry, or given an
explicit `crateDispositions` decision (`first-party-original` or
`architecture-reimplementation-existing-terms`) with evidence. An unclassified
crate FAILS the check; there is no silent default (sc-15191). Regenerate both
with:

```
node scripts/scan-inference-provenance.mjs --repo <inference> \
  --write config/inference-provenance-candidates.tsv \
  --write-crates config/inference-crate-prefixes.txt
```

The executable guard also pins the packaging path: `api:build:embedded` must
build the web app with the raw notice imports and compile `apps/web/dist` into
the Rust API sidecar, while Tauri must package that sidecar. A production web
build verifies that both full notices survive bundling. A signed installer was
not constructed for this license-only check; release CI remains the final
signed-bundle smoke.
