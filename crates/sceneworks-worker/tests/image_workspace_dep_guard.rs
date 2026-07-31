//! Structural guard for the workspace-wide `image` (sc-15052) and `png` (sc-15947) dependencies.
//!
//! Cargo unifies features for a shared dependency across workspace members, so a member's own
//! feature list never describes what it compiles with — every member builds against the union.
//! For `image` that divergence is SILENT and behavioural: its features decide which formats
//! decode, so a crate quietly gains a capability it never declared, and `cargo test -p <member>`
//! exercises a configuration that never ships. sc-15028 was exactly that: two transcode tests
//! asserted BMP could not decode, passed under `-p`, and failed in the workspace.
//!
//! sc-15052 closed it by hoisting `image` into `[workspace.dependencies]` with the union feature
//! set, so the declaration and the union are the same object. This guard keeps them that way, in
//! the spirit of the sibling `candle_kernels_patch_guard.rs` (also a root-`Cargo.toml` structural
//! guard living here). Three independent halves:
//!
//! 1. **Declaration** — parse the manifests: the root declares `image` with the expected features,
//!    and no member re-declares `image` OR `png` with a version/feature list of its own.
//! 2. **Compilation** — probe `ImageFormat::reading_enabled()`, which is a `cfg!(feature = ..)` in
//!    `image` itself, so it reports what this build ACTUALLY compiled. It asserts the enabled and
//!    the disabled formats, so it fails whether the set narrows or silently widens — and because
//!    it runs under `-p sceneworks-worker` and `--workspace` alike, a scope divergence like
//!    sc-15028's cannot come back without one of the two scopes going red.
//! 3. **Resolution** — read `Cargo.lock` and assert the `png` our crates resolve is the SAME
//!    package `image` resolved. sc-15947 writes PNG text chunks with `png` directly while `image`
//!    keeps encoding the pixels, and a skew there is invisible to both halves above: two `png`
//!    copies compile fine, declare identical features, and produce one file written by two
//!    encoders. The tree already carries a second `png` (0.17, via `tauri-codegen`/`ico`), so this
//!    is not hypothetical — resolving toward that one is a one-character edit away.
//!
//! Parse-only + a `cfg!` read: no GPU, no I/O beyond the manifests and the lockfile, so every CI
//! lane runs it.

use std::path::Path;

/// The feature union declared in `[workspace.dependencies]`. Adding a format is a deliberate
/// workspace-wide act: update the root `Cargo.toml` and this list together.
const EXPECTED_FEATURES: &[&str] = &["bmp", "gif", "jpeg", "png", "tiff", "webp"];

/// The members sc-15052 hoisted `image` for, as paths in the root `[workspace] members`.
const EXPECTED_IMAGE_MEMBERS: &[&str] = &[
    "apps/rust-api",
    "crates/sceneworks-core",
    "crates/sceneworks-image-quality",
    "crates/sceneworks-mcp",
    "crates/sceneworks-worker",
];

/// The members that inherit the workspace `png` (sc-15947). Only the crate that owns the workflow
/// text-chunk codec needs it; everything else goes through `image`.
const EXPECTED_PNG_MEMBERS: &[&str] = &["crates/sceneworks-core"];

// ---------------------------------------------------------------------------------------------
// Minimal TOML reading. Deliberately dependency-free (same reasoning as candle_kernels_patch_guard:
// a build guard should not need a parser in the graph). It handles what these manifests use —
// section headers, `key = value`, and inline tables/arrays spanning lines. It does not handle
// escaped quotes inside strings, which none of the workspace manifests contain.
// ---------------------------------------------------------------------------------------------

/// Drop a trailing `#` comment that is outside a string literal.
fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    for (i, b) in line.bytes().enumerate() {
        match b {
            b'"' => in_str = !in_str,
            b'#' if !in_str => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Net bracket/brace nesting introduced by a fragment, ignoring string contents.
fn depth_delta(s: &str) -> i32 {
    let mut in_str = false;
    let mut depth = 0;
    for b in s.bytes() {
        match b {
            b'"' => in_str = !in_str,
            b'{' | b'[' if !in_str => depth += 1,
            b'}' | b']' if !in_str => depth -= 1,
            _ => {}
        }
    }
    depth
}

/// `(section, key, value)` for every assignment in a manifest, with inline tables and arrays
/// re-joined onto one logical line so a multi-line `features = [ .. ]` reads as a single value.
fn manifest_entries(manifest: &str) -> Vec<(String, String, String)> {
    let mut entries = Vec::new();
    let mut section = String::new();
    let mut lines = manifest.lines();
    while let Some(raw) = lines.next() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        // A logical line starting with `[` and ending with `]` is a section header — a bare array
        // never starts one, because array values always follow a `key =`. Checking both ends is
        // what keeps `[target.'cfg(target_os = "macos")'.dependencies]` from parsing as an
        // assignment on its embedded `=`.
        if line.starts_with('[') && line.ends_with(']') {
            section = line
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim()
                .to_string();
            continue;
        }
        let Some((key, first)) = line.split_once('=') else {
            continue;
        };
        let mut value = first.trim().to_string();
        let mut depth = depth_delta(&value);
        while depth > 0 {
            let Some(next) = lines.next() else { break };
            let next = strip_comment(next).trim();
            depth += depth_delta(next);
            value.push(' ');
            value.push_str(next);
        }
        entries.push((section.clone(), key.trim().to_string(), value));
    }
    entries
}

/// Every double-quoted literal in a fragment, in order.
fn quoted_strings(s: &str) -> Vec<String> {
    s.split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect()
}

/// Split an inline table's body on its top-level commas.
fn split_top_level(inner: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_str = false;
    let mut depth = 0;
    for c in inner.chars() {
        match c {
            '"' => in_str = !in_str,
            '{' | '[' if !in_str => depth += 1,
            '}' | ']' if !in_str => depth -= 1,
            ',' if !in_str && depth == 0 => {
                parts.push(std::mem::take(&mut current));
                continue;
            }
            _ => {}
        }
        current.push(c);
    }
    parts.push(current);
    parts
        .into_iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// Look up one key inside an inline table value such as `{ version = "0.25", .. }`.
fn inline_field(value: &str, field: &str) -> Option<String> {
    let inner = value.trim().trim_start_matches('{').trim_end_matches('}');
    split_top_level(inner).into_iter().find_map(|part| {
        let (k, v) = part.split_once('=')?;
        (k.trim() == field).then(|| v.trim().to_string())
    })
}

// ---------------------------------------------------------------------------------------------
// The checks. Separated from file I/O so the red paths below are testable and this cannot rot
// into a guard that passes because its scan found nothing.
// ---------------------------------------------------------------------------------------------

/// The root must declare `image` once, with default features off and exactly the union.
fn check_root(root: &str) -> Result<(), String> {
    let value = manifest_entries(root)
        .into_iter()
        .find(|(section, key, _)| section == "workspace.dependencies" && key == "image")
        .map(|(_, _, value)| value)
        .ok_or(
            "root Cargo.toml has no `image` in [workspace.dependencies]: the hoist from sc-15052 \
             was undone, so each member's feature list is a mirror that will drift from the union \
             Cargo actually builds",
        )?;
    match inline_field(&value, "default-features").as_deref() {
        Some("false") => {}
        other => {
            return Err(format!(
                "workspace `image` must set `default-features = false` (found {other:?}): the \
                 feature list is meant to be exhaustive, and the defaults enable codecs no member \
                 asked for"
            ))
        }
    }
    let mut features = inline_field(&value, "features")
        .map(|list| quoted_strings(&list))
        .ok_or("workspace `image` declares no `features` list")?;
    // Compared as sets: reordering the root list is cosmetic and must not read as a union change.
    features.sort();
    let mut expected: Vec<String> = EXPECTED_FEATURES.iter().map(|f| f.to_string()).collect();
    expected.sort();
    if features != expected {
        return Err(format!(
            "workspace `image` features {features:?} != the union this guard pins {expected:?}. \
             Every member compiles against this list — if the change is intended, update \
             EXPECTED_FEATURES and the compiled-format test together."
        ));
    }
    Ok(())
}

/// The root must declare `png` once (sc-15947). Unlike `image` it has no feature list to pin —
/// `png` ships no default features and we enable none — so what matters is that the requirement
/// exists in exactly one place. Which VERSION it resolves to is checked against `Cargo.lock` by
/// [`check_lock_png_is_the_one_image_uses`], because the hazard is skew from `image`, not the
/// literal string.
fn check_root_png(root: &str) -> Result<(), String> {
    let value = manifest_entries(root)
        .into_iter()
        .find(|(section, key, _)| section == "workspace.dependencies" && key == "png")
        .map(|(_, _, value)| value)
        .ok_or(
            "root Cargo.toml has no `png` in [workspace.dependencies]: sc-15947 declares it there \
             so the crate that writes PNG text chunks cannot resolve a different `png` than the \
             one `image` encodes pixels with",
        )?;
    // A bare `png = "0.18"` or an inline table with a `version` — either is a requirement. What is
    // rejected is a declaration with no version at all, which would resolve to whatever anything
    // else in the graph happened to want.
    let has_version = value.trim().starts_with('"') || inline_field(&value, "version").is_some();
    if !has_version {
        return Err(format!(
            "workspace `png` is declared as `{value}` with no version requirement, so the \
             resolved copy is whatever the rest of the graph settles on (sc-15947)."
        ));
    }
    Ok(())
}

/// Fields that re-specify the dependency instead of inheriting it. Any of these on a member's
/// `image` or `png` is the drift this guard exists to stop.
const FORBIDDEN_FIELDS: &[&str] = &["default-features", "features", "version"];

fn mirror_error(dependency: &str, member: &str, section: &str, field: &str) -> String {
    format!(
        "{member}/Cargo.toml declares `{dependency}` in [{section}] with its own `{field}`. That \
         is a mirror of the workspace declaration, not the declaration itself — Cargo unifies the \
         graph anyway, so a per-crate `cargo test -p` would stop being representative of what \
         ships (sc-15028 / sc-15052 / sc-15947). Use `{dependency} = {{ workspace = true }}`."
    )
}

/// Whether `member` declares `dependency`, erroring if it declares it as anything but
/// workspace-inherited.
///
/// Cargo accepts three spellings and this checks all of them, because a guard that only knows the
/// one currently in the tree just redirects the next regression through the other two:
/// `image = { .. }`, the dotted `image.features = [..]`, and the `[dependencies.image]` sub-table.
fn check_member(dependency: &str, member: &str, manifest: &str) -> Result<bool, String> {
    let entries = manifest_entries(manifest);
    let dotted_prefix = format!("{dependency}.");
    let sub_table_suffix = format!("dependencies.{dependency}");
    let mut declares = false;

    // `image = { workspace = true }` form, in any dependency table (including target-gated ones).
    for (section, _key, value) in entries
        .iter()
        .filter(|(section, key, _)| key == dependency && section.ends_with("dependencies"))
    {
        declares = true;
        for field in FORBIDDEN_FIELDS {
            if inline_field(value, field).is_some() {
                return Err(mirror_error(dependency, member, section, field));
            }
        }
        if inline_field(value, "workspace").as_deref() != Some("true") {
            return Err(format!(
                "{member}/Cargo.toml declares `{dependency}` in [{section}] as `{value}` rather \
                 than `{{ workspace = true }}` (sc-15052 / sc-15947)."
            ));
        }
    }

    // Dotted-key form: `image.workspace = true`, `image.features = [..]`. This is the likeliest
    // way the hoist gets undone by hand — it is already this repo's house style for
    // `version.workspace` / `edition.workspace` / `rust-version.workspace` in every member.
    let dotted: Vec<(&str, &str, &str)> = entries
        .iter()
        .filter(|(section, key, _)| {
            section.ends_with("dependencies") && key.starts_with(&dotted_prefix)
        })
        .map(|(section, key, value)| {
            (
                section.as_str(),
                key.trim_start_matches(&dotted_prefix),
                value.as_str(),
            )
        })
        .collect();
    for (section, field, value) in &dotted {
        declares = true;
        if FORBIDDEN_FIELDS.contains(field) {
            return Err(mirror_error(dependency, member, section, field));
        }
        if *field == "workspace" && value.trim() != "true" {
            return Err(format!(
                "{member}/Cargo.toml sets `{dependency}.workspace = {value}` in [{section}]; it \
                 must be `true` (sc-15052 / sc-15947)."
            ));
        }
    }
    if let Some((section, ..)) = dotted.first() {
        if !dotted.iter().any(|(_, field, _)| *field == "workspace") {
            return Err(format!(
                "{member}/Cargo.toml declares `{dependency}.*` keys in [{section}] without \
                 `{dependency}.workspace = true`, so it does not inherit the workspace \
                 declaration (sc-15052 / sc-15947)."
            ));
        }
    }

    // `[dependencies.image]` sub-table form — same rule, different spelling.
    for (section, key, value) in entries
        .iter()
        .filter(|(section, _, _)| section.ends_with(&sub_table_suffix))
    {
        declares = true;
        if FORBIDDEN_FIELDS.contains(&key.as_str()) {
            return Err(mirror_error(dependency, member, section, key));
        }
        if key != "workspace" || value.trim() != "true" {
            return Err(format!(
                "{member}/Cargo.toml declares `{dependency}` as a [{section}] sub-table with \
                 `{key} = {value}`: only `workspace = true` may appear there (sc-15052 / \
                 sc-15947)."
            ));
        }
    }

    Ok(declares)
}

// ---------------------------------------------------------------------------------------------
// Resolution: the `png` we write chunks with must BE the `png` `image` encodes with.
// ---------------------------------------------------------------------------------------------

/// One `[[package]]` block from a `Cargo.lock`.
struct LockPackage {
    name: String,
    version: String,
    dependencies: Vec<String>,
}

/// Parse a `Cargo.lock` into its package blocks. Same dependency-free reasoning as the manifest
/// reader above; the lockfile subset in play is `[[package]]`, `name`, `version` and a
/// `dependencies` array of `"name"` / `"name version"` strings.
fn lock_packages(lock: &str) -> Vec<LockPackage> {
    let mut packages: Vec<LockPackage> = Vec::new();
    let mut lines = lock.lines().peekable();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line == "[[package]]" {
            packages.push(LockPackage {
                name: String::new(),
                version: String::new(),
                dependencies: Vec::new(),
            });
            continue;
        }
        let Some(current) = packages.last_mut() else {
            continue;
        };
        if let Some(rest) = line.strip_prefix("name = ") {
            current.name = quoted_strings(rest).into_iter().next().unwrap_or_default();
        } else if let Some(rest) = line.strip_prefix("version = ") {
            current.version = quoted_strings(rest).into_iter().next().unwrap_or_default();
        } else if line.starts_with("dependencies = [") {
            // Single-line `dependencies = []` and the usual multi-line form alike.
            current.dependencies.extend(quoted_strings(line));
            while !line.ends_with(']') {
                let Some(next) = lines.next() else { break };
                let next = next.trim();
                current.dependencies.extend(quoted_strings(next));
                if next.ends_with(']') {
                    break;
                }
            }
        }
    }
    packages
}

/// The version of `dependency` that `consumer` resolves to, per the lockfile.
///
/// Cargo writes a dependency entry as a bare `"name"` when only one version of it is in the graph
/// and as `"name version"` when several are. Both are handled: without an explicit version there
/// must be exactly one candidate, and that is the answer.
fn resolved_version(
    packages: &[LockPackage],
    consumer: &str,
    dependency: &str,
) -> Result<String, String> {
    let package = packages
        .iter()
        .find(|package| package.name == consumer)
        .ok_or_else(|| format!("Cargo.lock has no `{consumer}` package"))?;
    let entry = package
        .dependencies
        .iter()
        .find(|entry| {
            entry
                .split_whitespace()
                .next()
                .is_some_and(|name| name == dependency)
        })
        .ok_or_else(|| format!("Cargo.lock says `{consumer}` does not depend on `{dependency}`"))?;
    if let Some(version) = entry.split_whitespace().nth(1) {
        return Ok(version.to_owned());
    }
    let candidates: Vec<&LockPackage> = packages
        .iter()
        .filter(|package| package.name == dependency)
        .collect();
    match candidates.as_slice() {
        [only] => Ok(only.version.clone()),
        found => Err(format!(
            "`{consumer}` depends on an unversioned `{dependency}` entry but Cargo.lock holds \
             {} copies of it",
            found.len()
        )),
    }
}

/// `image` and every member that writes PNG text chunks must resolve the SAME `png`.
fn check_lock_png_is_the_one_image_uses(lock: &str, members: &[&str]) -> Result<(), String> {
    let packages = lock_packages(lock);
    let image_png = resolved_version(&packages, "image", "png")?;
    for member in members {
        // Member paths are `crates/sceneworks-core`; the lock keys on the package name.
        let crate_name = member.rsplit('/').next().unwrap_or(member);
        let member_png = resolved_version(&packages, crate_name, "png")?;
        if member_png != image_png {
            return Err(format!(
                "`{crate_name}` resolves png {member_png} but `image` resolves png {image_png}. \
                 Two copies of the PNG codec are in the graph, so the crate that writes the iTXt \
                 workflow chunk is not the crate that encodes the pixels next to it — a skew that \
                 compiles cleanly and shows up only in the file (sc-15947). Align the root \
                 [workspace.dependencies] `png` requirement with whatever `image` pulls in."
            ));
        }
    }
    Ok(())
}

/// Members listed in the root `[workspace]`.
fn workspace_members(root: &str) -> Vec<String> {
    manifest_entries(root)
        .into_iter()
        .find(|(section, key, _)| section == "workspace" && key == "members")
        .map(|(_, _, value)| quoted_strings(&value))
        .unwrap_or_default()
}

fn read_workspace_file(relative: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The set of members inheriting `dependency`, with every member's manifest checked on the way.
fn members_inheriting(dependency: &str, root: &str) -> Vec<String> {
    let members = workspace_members(root);
    assert!(
        !members.is_empty(),
        "parsed no members out of the root [workspace]: the scan is broken, not the manifests"
    );
    let mut declaring = Vec::new();
    for member in &members {
        let manifest = read_workspace_file(&format!("{member}/Cargo.toml"));
        match check_member(dependency, member, &manifest) {
            Ok(true) => declaring.push(member.clone()),
            Ok(false) => {}
            Err(msg) => panic!("{msg}"),
        }
    }
    declaring.sort();
    declaring
}

#[test]
fn image_is_declared_once_for_the_whole_workspace() {
    let root = read_workspace_file("Cargo.toml");
    if let Err(msg) = check_root(&root) {
        panic!("{msg}");
    }

    // Pin the identity, not a count: without this the guard would go green if `image` vanished from
    // every member — a pass that proves nothing — and a count alone would also accept four
    // DIFFERENT members inheriting it.
    assert_eq!(
        members_inheriting("image", &root),
        EXPECTED_IMAGE_MEMBERS,
        "the set of members inheriting the workspace `image` changed. If a member is missing here \
         but still uses `image`, the likely cause is a hand-written re-declaration this scan did \
         not recognise as inheritance — check its Cargo.toml before touching this list. Only edit \
         EXPECTED_IMAGE_MEMBERS when a member genuinely gained or dropped the dependency."
    );
}

#[test]
fn png_is_declared_once_for_the_whole_workspace() {
    let root = read_workspace_file("Cargo.toml");
    if let Err(msg) = check_root_png(&root) {
        panic!("{msg}");
    }
    assert_eq!(
        members_inheriting("png", &root),
        EXPECTED_PNG_MEMBERS,
        "the set of members inheriting the workspace `png` changed (sc-15947). A member that needs \
         PNG text chunks should inherit it and be added here; a member that re-declares its own \
         version is the skew this guard exists to stop."
    );
}

#[test]
fn the_resolved_png_is_the_one_image_encodes_with() {
    // The half neither the manifests nor `cfg!` can see. Reads the committed lockfile, so it fails
    // in review rather than after someone notices a shared image behaving oddly.
    let lock = read_workspace_file("Cargo.lock");
    if let Err(msg) = check_lock_png_is_the_one_image_uses(&lock, EXPECTED_PNG_MEMBERS) {
        panic!("{msg}");
    }
}

/// The compiled half: what `image` was ACTUALLY built with in this scope.
///
/// `reading_enabled()` is a `cfg!(feature = ..)` inside `image`, so this reads the real feature
/// set rather than the manifest's claim about it. Dds and Pcx are excluded: they return a hardcoded
/// `false`, so asserting them would pass with the dependency ripped out entirely.
#[test]
fn compiled_image_formats_match_the_workspace_declaration() {
    use image::ImageFormat;

    let enabled = [
        (ImageFormat::Bmp, "bmp"),
        (ImageFormat::Gif, "gif"),
        (ImageFormat::Jpeg, "jpeg"),
        (ImageFormat::Png, "png"),
        (ImageFormat::Tiff, "tiff"),
        (ImageFormat::WebP, "webp"),
    ];
    assert_eq!(
        enabled.len(),
        EXPECTED_FEATURES.len(),
        "this probe and EXPECTED_FEATURES have drifted apart"
    );
    for (format, feature) in enabled {
        assert!(
            format.reading_enabled(),
            "`image` was compiled WITHOUT `{feature}`, but [workspace.dependencies] declares it. \
             This scope decodes fewer formats than the shipped build — the sc-15028 trap."
        );
    }

    for (format, feature) in [
        (ImageFormat::Avif, "avif"),
        (ImageFormat::Farbfeld, "ff"),
        (ImageFormat::Hdr, "hdr"),
        (ImageFormat::Ico, "ico"),
        (ImageFormat::OpenExr, "exr"),
        (ImageFormat::Pnm, "pnm"),
        (ImageFormat::Qoi, "qoi"),
        (ImageFormat::Tga, "tga"),
    ] {
        assert!(
            !format.reading_enabled(),
            "`image` was compiled WITH `{feature}`, which [workspace.dependencies] does not \
             declare — `default-features = false` was dropped, or a dependency turned it on. \
             Code and tests can come to depend on a format nobody declared."
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Red-path coverage: each canned manifest is a failure this guard exists to catch, so a
// regression in the parser cannot quietly turn the real tests above into false greens.
// ---------------------------------------------------------------------------------------------

const GOOD_ROOT: &str = r#"
[workspace]
members = [
    "apps/rust-api",
    "crates/sceneworks-core",
]

[workspace.dependencies]
trash = "5"
# A comment mentioning image = { version = "9" } must not be parsed.
image = { version = "0.25", default-features = false, features = [
    "bmp",
    "gif",
    "jpeg",
    "png",
    "tiff",
    "webp",
] }
png = "0.18"
"#;

const GOOD_MEMBER: &str = r#"
[package]
name = "demo"

[dependencies]
imageproc = { version = "0.25", default-features = false }
image = { workspace = true }
image_hasher = "3"
png = { workspace = true }
png-decoder = "0.1"

[target.'cfg(target_os = "macos")'.dependencies]
mlx-rs = { git = "https://example.invalid" }
"#;

/// A lockfile in the shape Cargo writes when SEVERAL versions of a crate are in the graph, which
/// is this workspace's situation: `image` pulls png 0.18 and the Tauri codegen chain pulls 0.17.
const GOOD_LOCK: &str = r#"
[[package]]
name = "image"
version = "0.25.5"
dependencies = [
 "bytemuck",
 "png 0.18.1",
]

[[package]]
name = "png"
version = "0.17.16"

[[package]]
name = "png"
version = "0.18.1"
dependencies = [
 "bitflags",
]

[[package]]
name = "sceneworks-core"
version = "0.8.1"
dependencies = [
 "image",
 "png 0.18.1",
 "serde",
]

[[package]]
name = "tauri-codegen"
version = "2.4.0"
dependencies = [
 "png 0.17.16",
]
"#;

#[test]
fn guard_accepts_the_hoisted_shape() {
    assert_eq!(check_root(GOOD_ROOT), Ok(()));
    assert_eq!(check_root_png(GOOD_ROOT), Ok(()));
    assert_eq!(check_member("image", "demo", GOOD_MEMBER), Ok(true));
    assert_eq!(check_member("png", "demo", GOOD_MEMBER), Ok(true));
}

#[test]
fn guard_parses_members_and_ignores_lookalike_keys() {
    assert_eq!(
        workspace_members(GOOD_ROOT),
        vec!["apps/rust-api", "crates/sceneworks-core"]
    );
    // `imageproc` / `image_hasher` must not be mistaken for `image`, nor `png-decoder` for `png` —
    // that would make the member check fire on crates that never declare either.
    let no_image = GOOD_MEMBER.replace("image = { workspace = true }\n", "");
    assert_eq!(check_member("image", "demo", &no_image), Ok(false));
    let no_png = GOOD_MEMBER.replace("png = { workspace = true }\n", "");
    assert_eq!(check_member("png", "demo", &no_png), Ok(false));
}

#[test]
fn guard_rejects_a_dropped_hoist() {
    // Renames the real declaration only — GOOD_ROOT's decoy comment still spells
    // `image = { version = "9" }`, so this also proves commented-out text cannot satisfy the
    // lookup. (Targeting `image = {` instead would rewrite the comment and leave the real
    // declaration standing, which is how this test first failed.)
    let root = GOOD_ROOT.replacen(
        "image = { version = \"0.25\"",
        "unrelated = { version = \"0.25\"",
        1,
    );
    assert!(root.contains("image = { version = \"9\" }"), "decoy lost");
    let err = check_root(&root).unwrap_err();
    assert!(
        err.contains("no `image` in [workspace.dependencies]"),
        "{err}"
    );
}

#[test]
fn guard_rejects_default_features_creeping_back_on() {
    let root = GOOD_ROOT.replacen("default-features = false", "default-features = true", 1);
    let err = check_root(&root).unwrap_err();
    assert!(err.contains("default-features = false"), "{err}");
}

#[test]
fn guard_rejects_a_narrowed_union() {
    let root = GOOD_ROOT.replacen("    \"bmp\",\n", "", 1);
    let err = check_root(&root).unwrap_err();
    assert!(err.contains("!= the union"), "{err}");
}

#[test]
fn guard_rejects_a_member_re_declaring_its_own_features() {
    // The sc-15028 shape verbatim: a member pinning a narrower list than the workspace union.
    let member = GOOD_MEMBER.replacen(
        "image = { workspace = true }",
        r#"image = { version = "0.25", default-features = false, features = ["png", "jpeg", "webp"] }"#,
        1,
    );
    let err = check_member("image", "demo", &member).unwrap_err();
    assert!(err.contains("its own `default-features`"), "{err}");
}

#[test]
fn guard_rejects_a_member_re_declaring_a_bare_version() {
    let member = GOOD_MEMBER.replacen("image = { workspace = true }", r#"image = "0.25""#, 1);
    let err = check_member("image", "demo", &member).unwrap_err();
    assert!(err.contains("rather than"), "{err}");
}

#[test]
fn guard_rejects_a_target_gated_re_declaration() {
    // A dependency table other than plain [dependencies] must not be a way around the rule.
    let member = GOOD_MEMBER.replacen(
        r#"mlx-rs = { git = "https://example.invalid" }"#,
        r#"image = { version = "0.25", features = ["tga"] }"#,
        1,
    );
    let err = check_member("image", "demo", &member).unwrap_err();
    assert!(err.contains("its own `features`"), "{err}");
    assert!(err.contains("cfg(target_os = \"macos\")"), "{err}");
}

#[test]
fn guard_rejects_the_dotted_key_form() {
    // The spelling a developer reaches for by muscle memory, since every member already writes
    // `version.workspace = true`. Reviewed as the live hole in the first cut of this guard: it
    // reproduced the sc-15028 divergence (`-p sceneworks-core` back to jpeg alone) while the
    // declaration check saw nothing.
    let member = GOOD_MEMBER.replacen(
        "image = { workspace = true }",
        "image.version = \"0.25\"\nimage.default-features = false\nimage.features = [\"jpeg\"]",
        1,
    );
    let err = check_member("image", "demo", &member).unwrap_err();
    assert!(err.contains("its own `version`"), "{err}");
    assert!(err.contains("[dependencies]"), "{err}");
}

#[test]
fn guard_accepts_the_dotted_inheritance_form() {
    // `image.workspace = true` is legal and equivalent — it must pass, or the guard would be
    // pushing people away from a correct spelling.
    let member = GOOD_MEMBER.replacen("image = { workspace = true }", "image.workspace = true", 1);
    assert_eq!(check_member("image", "demo", &member), Ok(true));
}

#[test]
fn guard_rejects_dotted_keys_without_inheritance() {
    // No forbidden field, but no `workspace = true` either — still not inheriting the union.
    let member = GOOD_MEMBER.replacen("image = { workspace = true }", "image.optional = true", 1);
    let err = check_member("image", "demo", &member).unwrap_err();
    assert!(err.contains("without `image.workspace = true`"), "{err}");
}

#[test]
fn guard_accepts_the_sub_table_inheritance_form() {
    let member = GOOD_MEMBER.replacen("image = { workspace = true }", "", 1)
        + "\n[dependencies.image]\nworkspace = true\n";
    assert_eq!(check_member("image", "demo", &member), Ok(true));
}

#[test]
fn guard_rejects_the_sub_table_form() {
    let member = GOOD_MEMBER.replacen("image = { workspace = true }", "", 1)
        + "\n[dependencies.image]\nversion = \"0.25\"\n";
    let err = check_member("image", "demo", &member).unwrap_err();
    assert!(err.contains("[dependencies.image]"), "{err}");
    assert!(err.contains("its own `version`"), "{err}");
}

// ---------------------------------------------------------------------------------------------
// Red paths for the sc-15947 `png` half. The member check is the same code, so what needs its own
// coverage is the root declaration and the lockfile resolution.
// ---------------------------------------------------------------------------------------------

#[test]
fn guard_rejects_a_dropped_png_declaration() {
    let root = GOOD_ROOT.replacen("png = \"0.18\"", "", 1);
    let err = check_root_png(&root).unwrap_err();
    assert!(
        err.contains("no `png` in [workspace.dependencies]"),
        "{err}"
    );
}

#[test]
fn guard_rejects_a_png_declaration_with_no_version() {
    // `png = { git = .. }` or a bare `png = {}` resolves to whatever else in the graph wants, which
    // is the skew this guard exists to prevent — expressed as a declaration rather than as a
    // member re-declaration.
    let root = GOOD_ROOT.replacen("png = \"0.18\"", "png = { optional = true }", 1);
    let err = check_root_png(&root).unwrap_err();
    assert!(err.contains("no version requirement"), "{err}");
    // The inline-table form WITH a version is fine — the guard must not push people to a bare
    // string when they legitimately need another field.
    let root = GOOD_ROOT.replacen(
        "png = \"0.18\"",
        "png = { version = \"0.18\", optional = true }",
        1,
    );
    assert_eq!(check_root_png(&root), Ok(()));
}

#[test]
fn guard_rejects_a_member_re_declaring_png() {
    // The sc-15947 shape: someone adds `png` to a second crate by hand and pins it, and Cargo
    // happily resolves the 0.17 already in the tree.
    let member = GOOD_MEMBER.replacen("png = { workspace = true }", r#"png = "0.17""#, 1);
    let err = check_member("png", "demo", &member).unwrap_err();
    assert!(err.contains("rather than"), "{err}");

    let member = GOOD_MEMBER.replacen(
        "png = { workspace = true }",
        "png.version = \"0.17\"\npng.workspace = true",
        1,
    );
    let err = check_member("png", "demo", &member).unwrap_err();
    assert!(err.contains("its own `version`"), "{err}");
}

#[test]
fn guard_accepts_a_lock_where_png_agrees_with_image() {
    assert_eq!(
        check_lock_png_is_the_one_image_uses(GOOD_LOCK, &["crates/sceneworks-core"]),
        Ok(())
    );
}

#[test]
fn guard_rejects_a_lock_where_png_skews_from_image() {
    // The exact regression: our crate resolves the 0.17 the Tauri chain brought in while `image`
    // still encodes with 0.18. Compiles, same features, two encoders in one file.
    let lock = GOOD_LOCK.replacen(
        " \"png 0.18.1\",\n \"serde\",",
        " \"png 0.17.16\",\n \"serde\",",
        1,
    );
    assert!(
        lock.contains("\"png 0.17.16\",\n \"serde\""),
        "the swap missed"
    );
    let err = check_lock_png_is_the_one_image_uses(&lock, &["crates/sceneworks-core"]).unwrap_err();
    assert!(err.contains("resolves png 0.17.16"), "{err}");
    assert!(err.contains("`image` resolves png 0.18.1"), "{err}");
}

#[test]
fn guard_rejects_a_lock_where_our_crate_dropped_png_entirely() {
    // A pass that proves nothing is the failure mode a lockfile scan invites: if the dependency
    // simply is not there, "no skew found" must be an error and not a green.
    let lock = GOOD_LOCK.replacen(" \"png 0.18.1\",\n \"serde\",", " \"serde\",", 1);
    let err = check_lock_png_is_the_one_image_uses(&lock, &["crates/sceneworks-core"]).unwrap_err();
    assert!(err.contains("does not depend on `png`"), "{err}");
}

#[test]
fn guard_resolves_an_unversioned_lock_entry() {
    // When only ONE copy of a crate is in the graph Cargo writes the dependency as a bare name. If
    // the 0.17 chain ever leaves the tree the real lockfile takes that shape, and the scan must
    // keep working rather than start reporting a missing dependency.
    let lock = r#"
[[package]]
name = "image"
version = "0.25.5"
dependencies = [
 "png",
]

[[package]]
name = "png"
version = "0.18.1"

[[package]]
name = "sceneworks-core"
version = "0.8.1"
dependencies = [
 "png",
]
"#;
    assert_eq!(
        check_lock_png_is_the_one_image_uses(lock, &["crates/sceneworks-core"]),
        Ok(())
    );
}
