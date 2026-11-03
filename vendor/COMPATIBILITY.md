# Desktop vendor review — 2026-11-03 (SW-6822 / F-185)

The production overrides remain GLib 0.18.5 with the string-array iterator safety
backport and updater 2.11.0 with its Windows zip 9 requirement. No package versions,
locked dependencies, signatures, permissions or historical source digests changed.

## Upstream compatibility

- The [official GLib release index](https://crates.io/api/v1/crates/glib) still lists
  **0.18.5** as the latest non-yanked 0.18 release (latest overall: 0.22.10).
  The [published 0.18.5 archive](https://static.crates.io/crates/glib/glib-0.18.5.crate)
  has SHA-256 `233daaf6e83ae6a12a52055f568f9d7cf4671dabb78ff9560ab6da230ce00ee5`
  and still passes an immutable out-pointer in `VariantStrIter::impl_get`.
  Our two-line upstream backport remains necessary for the GTK3 `glib ^0.18`
  dependency graph. A 0.22 upgrade does not satisfy that requirement or preserve
  its Rust types. Keep the existing optimized Linux regression and removal criteria
  in [GLib's provenance](glib-0.18.5/VENDORED.md).
- The [official updater release index](https://crates.io/api/v1/crates/tauri-plugin-updater)
  lists **2.14.1**, published 2026-10-28. Its
  [published archive](https://static.crates.io/crates/tauri-plugin-updater/tauri-plugin-updater-2.14.1.crate)
  has SHA-256 `d957abbcb3f9f6c76d22a3a0f012f30880640d0bccb5634d2b12a9937e4c872b`.
  Its Windows Cargo dependency already requires **zip 9**, matching the
  [upstream migration](https://github.com/tauri-apps/plugins-workspace/commit/ba98bfadc4c868c9a7e1a226c83af528e1f11b89).
  Waiting for a hypothetical 2.15 is unnecessary. However, 2.14.1 also changes
  commands, configuration, errors, library/installer code and the generated JS;
  `src/updater.rs` grows from 60,917 to 110,210 bytes. This review does not adopt
  those unrelated updater changes. The locked 2.11.0 published archive still requires
  zip 4, so its local zip 9 backport remains necessary. Adopting 2.14.1 deliberately
  with the updater's platform acceptance would allow removal of that override.
- The [official zip index](https://crates.io/api/v1/crates/zip) lists 9.3.0 on this
  review date. The workspace remains locked to its audited 9.0.5; a larger version
  number alone establishes neither a required upgrade nor installer compatibility.
  The [official RustSec zip advisories](https://github.com/RustSec/advisory-db/tree/main/crates/zip)
  mark 2026-0367/0368/0369 and the newer 2026-0429 extraction-allocation advisory as
  patched by `>= 9.0.0`; the locked version satisfies those ranges. The earlier
  2025-0168 canonicalization advisory is patched by `>= 2.3.0`.

The current updater tree, `CURRENT_UPDATER` closure and all retired `zip9-*` trees
remain byte-identical. The latter are historical source evidence consumed by the
source guards, archived receipts and retrieval tests, not active build overrides.
Do not restamp their manifests or digests to make pruning pass.

## Removal boundary

Removed only `glib-0.18.5/Gir_GObject.toml` (3,640 bytes; SHA-256
`02eca96b8fc6fe0e7b36407b91e4e91a5e1662bcdef9b4bbe4ab34744182e8a7`).
It configures a binding generator, has no build script/import/include or Cargo
target consumer, and the generated Rust sources already live in `src/`. Recover
the original from the checked archive above or Git commit
`6a923ccaa2cee63cddd5c22e7f148c085d68f434` at the same path.

Retained the other suggested candidates after inspecting actual consumers:

- `Gir.toml` is read by `glib/tests/check_gir.rs`.
- GLib's benchmark and test trees are explicit or automatically discovered Cargo
  targets, including compile-fail fixtures. Their supported surface stays intact.
- Updater's `api-iife.js` is read by `build.rs` through `global_api_script_path`;
  its permissions are also build inputs. Removing these would break the plugin.
- GLib's README is included into Rust documentation. Cargo metadata, licenses,
  copyright notices, security policy and provenance remain available.

Validation for this review uses metadata, source/importer closure checks and
offline archive/HTTP/filesystem fixtures. No model execution is involved.
