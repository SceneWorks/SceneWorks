//! Structural guard for the workspace-wide `image` dependency (sc-15052).
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
//! guard living here). Two independent halves:
//!
//! 1. **Declaration** — parse the manifests: the root declares `image` with the expected features,
//!    and no member re-declares it with a version/feature list of its own.
//! 2. **Compilation** — probe `ImageFormat::reading_enabled()`, which is a `cfg!(feature = ..)` in
//!    `image` itself, so it reports what this build ACTUALLY compiled. It asserts the enabled and
//!    the disabled formats, so it fails whether the set narrows or silently widens — and because
//!    it runs under `-p sceneworks-worker` and `--workspace` alike, a scope divergence like
//!    sc-15028's cannot come back without one of the two scopes going red.
//!
//! Parse-only + a `cfg!` read: no GPU, no I/O beyond the manifests, so every CI lane runs it.

use std::path::Path;

/// The feature union declared in `[workspace.dependencies]`. Adding a format is a deliberate
/// workspace-wide act: update the root `Cargo.toml` and this list together.
const EXPECTED_FEATURES: &[&str] = &["bmp", "gif", "jpeg", "png", "tiff", "webp"];

/// The members sc-15052 hoisted `image` for, as paths in the root `[workspace] members`.
const EXPECTED_IMAGE_MEMBERS: &[&str] = &[
    "apps/rust-api",
    "crates/sceneworks-core",
    "crates/sceneworks-image-quality",
    "crates/sceneworks-worker",
];

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

/// Fields that re-specify the dependency instead of inheriting it. Any of these on a member's
/// `image` is the drift this guard exists to stop.
const FORBIDDEN_FIELDS: &[&str] = &["default-features", "features", "version"];

fn mirror_error(member: &str, section: &str, field: &str) -> String {
    format!(
        "{member}/Cargo.toml declares `image` in [{section}] with its own `{field}`. That is a \
         mirror of the workspace union, not the union itself — Cargo will still build this crate \
         against the union, so a per-crate `cargo test -p` would stop being representative of what \
         ships (sc-15028 / sc-15052). Use `image = {{ workspace = true }}`."
    )
}

/// Whether `member` declares `image`, erroring if it declares it as anything but workspace-inherited.
///
/// Cargo accepts three spellings and this checks all of them, because a guard that only knows the
/// one currently in the tree just redirects the next regression through the other two:
/// `image = { .. }`, the dotted `image.features = [..]`, and the `[dependencies.image]` sub-table.
fn check_member(member: &str, manifest: &str) -> Result<bool, String> {
    let entries = manifest_entries(manifest);
    let mut declares = false;

    // `image = { workspace = true }` form, in any dependency table (including target-gated ones).
    for (section, _key, value) in entries
        .iter()
        .filter(|(section, key, _)| key == "image" && section.ends_with("dependencies"))
    {
        declares = true;
        for field in FORBIDDEN_FIELDS {
            if inline_field(value, field).is_some() {
                return Err(mirror_error(member, section, field));
            }
        }
        if inline_field(value, "workspace").as_deref() != Some("true") {
            return Err(format!(
                "{member}/Cargo.toml declares `image` in [{section}] as `{value}` rather than \
                 `{{ workspace = true }}` (sc-15052)."
            ));
        }
    }

    // Dotted-key form: `image.workspace = true`, `image.features = [..]`. This is the likeliest
    // way the hoist gets undone by hand — it is already this repo's house style for
    // `version.workspace` / `edition.workspace` / `rust-version.workspace` in every member.
    let dotted: Vec<(&str, &str, &str)> = entries
        .iter()
        .filter(|(section, key, _)| section.ends_with("dependencies") && key.starts_with("image."))
        .map(|(section, key, value)| {
            (
                section.as_str(),
                key.trim_start_matches("image."),
                value.as_str(),
            )
        })
        .collect();
    for (section, field, value) in &dotted {
        declares = true;
        if FORBIDDEN_FIELDS.contains(field) {
            return Err(mirror_error(member, section, field));
        }
        if *field == "workspace" && value.trim() != "true" {
            return Err(format!(
                "{member}/Cargo.toml sets `image.workspace = {value}` in [{section}]; it must be \
                 `true` (sc-15052)."
            ));
        }
    }
    if let Some((section, ..)) = dotted.first() {
        if !dotted.iter().any(|(_, field, _)| *field == "workspace") {
            return Err(format!(
                "{member}/Cargo.toml declares `image.*` keys in [{section}] without \
                 `image.workspace = true`, so it does not inherit the workspace union (sc-15052)."
            ));
        }
    }

    // `[dependencies.image]` sub-table form — same rule, different spelling.
    for (section, key, value) in entries
        .iter()
        .filter(|(section, _, _)| section.ends_with("dependencies.image"))
    {
        declares = true;
        if FORBIDDEN_FIELDS.contains(&key.as_str()) {
            return Err(mirror_error(member, section, key));
        }
        if key != "workspace" || value.trim() != "true" {
            return Err(format!(
                "{member}/Cargo.toml declares `image` as a [{section}] sub-table with \
                 `{key} = {value}`: only `workspace = true` may appear there (sc-15052)."
            ));
        }
    }

    Ok(declares)
}

/// Members listed in the root `[workspace]`.
fn workspace_members(root: &str) -> Vec<String> {
    manifest_entries(root)
        .into_iter()
        .find(|(section, key, _)| section == "workspace" && key == "members")
        .map(|(_, _, value)| quoted_strings(&value))
        .unwrap_or_default()
}

#[test]
fn image_is_declared_once_for_the_whole_workspace() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let read = |path: std::path::PathBuf| {
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    };

    let root = read(workspace.join("Cargo.toml"));
    if let Err(msg) = check_root(&root) {
        panic!("{msg}");
    }

    let members = workspace_members(&root);
    assert!(
        !members.is_empty(),
        "parsed no members out of the root [workspace]: the scan is broken, not the manifests"
    );

    let mut declaring = Vec::new();
    for member in &members {
        let manifest = read(workspace.join(member).join("Cargo.toml"));
        match check_member(member, &manifest) {
            Ok(true) => declaring.push(member.clone()),
            Ok(false) => {}
            Err(msg) => panic!("{msg}"),
        }
    }
    declaring.sort();
    // Pin the identity, not a count: without this the guard would go green if `image` vanished from
    // every member — a pass that proves nothing — and a count alone would also accept four
    // DIFFERENT members inheriting it.
    assert_eq!(
        declaring, EXPECTED_IMAGE_MEMBERS,
        "the set of members inheriting the workspace `image` changed. If a member is missing here \
         but still uses `image`, the likely cause is a hand-written re-declaration this scan did \
         not recognise as inheritance — check its Cargo.toml before touching this list. Only edit \
         EXPECTED_IMAGE_MEMBERS when a member genuinely gained or dropped the dependency."
    );
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
"#;

const GOOD_MEMBER: &str = r#"
[package]
name = "demo"

[dependencies]
imageproc = { version = "0.25", default-features = false }
image = { workspace = true }
image_hasher = "3"

[target.'cfg(target_os = "macos")'.dependencies]
mlx-rs = { git = "https://example.invalid" }
"#;

#[test]
fn guard_accepts_the_hoisted_shape() {
    assert_eq!(check_root(GOOD_ROOT), Ok(()));
    assert_eq!(check_member("demo", GOOD_MEMBER), Ok(true));
}

#[test]
fn guard_parses_members_and_ignores_lookalike_keys() {
    assert_eq!(
        workspace_members(GOOD_ROOT),
        vec!["apps/rust-api", "crates/sceneworks-core"]
    );
    // `imageproc` / `image_hasher` must not be mistaken for `image` — that would make the member
    // check fire on crates that never declare it.
    let no_image = GOOD_MEMBER.replace("image = { workspace = true }\n", "");
    assert_eq!(check_member("demo", &no_image), Ok(false));
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
    let err = check_member("demo", &member).unwrap_err();
    assert!(err.contains("its own `default-features`"), "{err}");
}

#[test]
fn guard_rejects_a_member_re_declaring_a_bare_version() {
    let member = GOOD_MEMBER.replacen("image = { workspace = true }", r#"image = "0.25""#, 1);
    let err = check_member("demo", &member).unwrap_err();
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
    let err = check_member("demo", &member).unwrap_err();
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
    let err = check_member("demo", &member).unwrap_err();
    assert!(err.contains("its own `version`"), "{err}");
    assert!(err.contains("[dependencies]"), "{err}");
}

#[test]
fn guard_accepts_the_dotted_inheritance_form() {
    // `image.workspace = true` is legal and equivalent — it must pass, or the guard would be
    // pushing people away from a correct spelling.
    let member = GOOD_MEMBER.replacen("image = { workspace = true }", "image.workspace = true", 1);
    assert_eq!(check_member("demo", &member), Ok(true));
}

#[test]
fn guard_rejects_dotted_keys_without_inheritance() {
    // No forbidden field, but no `workspace = true` either — still not inheriting the union.
    let member = GOOD_MEMBER.replacen("image = { workspace = true }", "image.optional = true", 1);
    let err = check_member("demo", &member).unwrap_err();
    assert!(err.contains("without `image.workspace = true`"), "{err}");
}

#[test]
fn guard_accepts_the_sub_table_inheritance_form() {
    let member = GOOD_MEMBER.replacen("image = { workspace = true }", "", 1)
        + "\n[dependencies.image]\nworkspace = true\n";
    assert_eq!(check_member("demo", &member), Ok(true));
}

#[test]
fn guard_rejects_the_sub_table_form() {
    let member = GOOD_MEMBER.replacen("image = { workspace = true }", "", 1)
        + "\n[dependencies.image]\nversion = \"0.25\"\n";
    let err = check_member("demo", &member).unwrap_err();
    assert!(err.contains("[dependencies.image]"), "{err}");
    assert!(err.contains("its own `version`"), "{err}");
}
