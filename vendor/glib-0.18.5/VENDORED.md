# glib 0.18.5 safety backport (sc-17186)

Owner: SceneWorks maintainers. This is the published, MIT-licensed glib 0.18.5
crate, retained with its LICENSE, COPYRIGHT and original Cargo metadata.

- Archive: https://static.crates.io/crates/glib/glib-0.18.5.crate
- SHA-256: `233daaf6e83ae6a12a52055f568f9d7cf4671dabb78ff9560ab6da230ce00ee5`
- Original upstream revision: `42b9caf98e03ded086362d9653ca58fe94dc8658`
- Fix: https://github.com/gtk-rs/gtk-rs-core/pull/1343
- Merge: https://github.com/gtk-rs/gtk-rs-core/commit/05dff0ee696f9bcd8617cd48c4b812d046d440cb

The only production change is the upstream two-line `VariantStrIter::impl_get`
correction in `src/variant_iter.rs`: declare the C string out-pointer mutable and
pass `&mut p` to `g_variant_get_child`. No APIs, ABI bindings, features, versions,
or dependency requirements were changed. The root `[patch.crates-io]` applies it
to the GTK3/Tauri graph. The crates.io index checked on 2026-09-29 still has 0.18.5
as the latest 0.18 release; replacing it directly with 0.20 is not compatible with
GTK3's `glib ^0.18` types.

Run `node scripts/check-glib-variant-iter.mjs` on Linux with Rust, Node and GLib
headers installed. It verifies the desktop dependency path, builds the locked
workspace glib in release mode, and links an optimized public-API regression
against that exact artifact. Coverage includes forward/reverse iteration,
interleaved ends, skips, UTF-8, empty strings and exhaustion. For compatibility,
build `cargo build --locked -p sceneworks-desktop` with GTK3, WebKitGTK 4.1 and
AppIndicator development libraries, then launch the real shell under a display.
A typecheck, placeholder sidecar or failed setup is not full product startup.

## Advisory status and maintenance

The package intentionally remains version **0.18.5**. Version-based scanners can
still report GHSA-wrw7-89jp-8q8g / RUSTSEC-2024-0429. This is a source backport,
not an upstream fixed release or proof that a scanner finding has closed. The
historical 2026-08-03 Dependabot #9 `tolerable_risk` dismissal remains recorded in
sc-17186; it is superseded as a remediation strategy by this patch. Actual
product reachability of Variant string-array iteration remains unproven.

Keep the archive checksum, source diff and optimized test together during any
refresh. Remove this override only when a compatible upstream release includes
the fix, or when the desktop deliberately migrates its entire GTK binding stack;
re-run the regression and Linux desktop build/start smoke on that change. Do not
rename this crate to a falsely fixed 0.20 version or suppress a finding solely
because the registry source disappeared from Cargo.lock.
