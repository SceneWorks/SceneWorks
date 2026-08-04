//! Source-level drift guard for the **bespoke candle preview wiring** (epic 16948, sc-16962).
//!
//! ## The defect this exists to make impossible
//!
//! Most candle image lanes reach their engine through `dyn Generator`, so the registry threads the
//! job's live [`gen_core::PreviewSink`] onto `GenerationRequest.preview` once, for everyone. A handful
//! of lanes do not: `Krea2ControlRequest`, `QwenFunControlRequest` and `QwenEditRequest` are invoked
//! BY NAME, so each carries its own `preview` field and each worker call site must set it.
//!
//! Those call sites are exhaustive struct literals with no `..Default::default()`. When upstream adds
//! the field the build breaks, and the cheapest way to make it green is `preview: Default::default()`
//! — which compiles, ships, and emits nothing. It happened: the `8ffa211a` pin bump (sc-17054) landed
//! exactly that on the Krea control lane, so Krea rendered with a fully-wired engine and a fully-wired
//! UI and the user still saw no developing image. Nothing was red.
//!
//! ## Why a source guard rather than a compiled test
//!
//! Every one of these lanes lives under `#[cfg(all(not(target_os = "macos"), feature =
//! "backend-candle"))]`, and both candle CI lanes (`desktop-windows.yml`, `server-candle-linux.yml`)
//! are `workflow_dispatch`-only. A compiled `#[test]` over them would not run on an ordinary PR — the
//! very situation in which the regression would land. This module is plain `#[cfg(test)]`: it reads
//! the lanes' source text, so it runs on macOS, on the Linux candle-off parity lane, and on Windows,
//! on every PR. Precedent is [`crate::architecture_tests`], whose comment stripper it reuses.
//!
//! ## What it asserts
//!
//! 1. Every bespoke request literal listed in [`WIRED_LANES`] binds `preview` to a live expression.
//! 2. **No** file under `src/image_jobs/` binds `preview` to a `::default()` expression — a sweep, not
//!    a list, so the eight remaining Tier 1 family stories (sc-16953…sc-16959) are covered the day
//!    their upstream field appears rather than needing this guard amended.
//! 3. No bespoke `*Request` literal hides `preview` behind a struct-update (`..Default::default()`)
//!    entry. This is the *same* defect as 2 wearing the spelling a developer actually reaches for
//!    first: when an exhaustive literal gains one field upstream, `..Default::default()` is the
//!    shortest way to make the build green, and it produces an identical inert sink. Rule 2 alone
//!    cannot see it — the entry's field name parses as `..Default`, which never equals `preview`.
//! 4. The worker closures that receive the live sink actually bind it, instead of discarding it as
//!    `_preview`.

use std::path::{Path, PathBuf};

use crate::architecture_tests::code_without_comments_or_literals;

/// The bespoke candle request literals whose upstream struct carries a `preview` field at the pinned
/// inference revision, and the worker file that builds each.
///
/// All three request types remain present at inference `5b6d6aa`; sc-16950/sc-16952 originally
/// introduced their preview fields.
const WIRED_LANES: &[(&str, &str)] = &[
    ("Krea2ControlRequest", "krea_control_candle.rs"),
    ("QwenFunControlRequest", "qwen_control.rs"),
    ("QwenEditRequest", "qwen_edit_candle.rs"),
];

/// Files that thread the sink from a `drive_gen_items*` closure into a bespoke provider. Each must
/// BIND the closure's preview parameter; `_preview` means the sink is being dropped on the floor.
const SINK_CONSUMERS: &[&str] = &["candle_strict_control.rs", "qwen_edit_candle.rs"];

/// `*Request` types under `src/image_jobs/` that are **not** bespoke by-name provider requests, and
/// are therefore exempt from the struct-update rule in
/// [`no_bespoke_candle_request_hides_its_preview_behind_a_rest_init`].
///
/// * `gen_core::GenerationRequest` is the *registry-driven* request. The registry threads the job's
///   sink onto it once for every family that renders through `dyn Generator` (`base.rs` sets
///   `preview` explicitly alongside its `..Default::default()`), which is precisely why the bespoke
///   lanes are the ones that need guarding. The rest-inits that omit `preview` are deliberate:
///   `detail.rs`'s per-tile refinement pass renders tiles, not the image the card shows.
/// * `ImageRequest` is SceneWorks' own JSON job payload. It carries no `preview` field and no engine
///   ever reads one off it; the `fn request(..) -> ImageRequest` test helpers in the lane files are
///   all it names.
///
/// This list is deliberately tiny and deliberately by name. The regression being prevented is a
/// developer reaching for `..Default::default()` at a call site, not a developer editing this list.
const NON_BESPOKE_REQUESTS: &[&str] = &["GenerationRequest", "ImageRequest"];

fn image_jobs_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/image_jobs")
}

/// Every `.rs` file directly under `src/image_jobs/`, as `(file name, stripped source)`, sorted.
///
/// Read from disk rather than enumerated with `include_str!` so a NEW candle lane is swept the moment
/// it is added. A hardcoded list is how the previous inventory-shaped guard in this repo went stale.
fn image_jobs_sources() -> Vec<(String, String)> {
    let dir = image_jobs_dir();
    let mut sources: Vec<(String, String)> = std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("read {}: {error}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .map(|path| {
            let name = path
                .file_name()
                .expect("file name")
                .to_string_lossy()
                .into_owned();
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            (name, code_without_comments_or_literals(&source))
        })
        .collect();
    sources.sort();
    assert!(
        sources.len() > 20,
        "expected the candle/MLX image lanes under {}, found {} files",
        dir.display(),
        sources.len()
    );
    sources
}

/// Collapse runs of whitespace so an expression can be compared regardless of rustfmt's line breaks.
fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `value` is one of the inert-sink spellings. `Default::default()`, `PreviewSink::default()`
/// and `gen_core::PreviewSink::default()` are all the same defect wearing different clothes, so match
/// the suffix rather than an allow-list of paths.
fn is_default_expression(value: &str) -> bool {
    let value = collapse(value).replace(' ', "");
    value.ends_with("::default()") || value == "Default::default()"
}

/// The field name a top-level struct-literal entry binds. `preview: sink.clone()` and the shorthand
/// `preview` both yield `preview`; a struct-update entry yields the nonsense `..Default`, which is
/// exactly why [`is_rest_init`] has to exist alongside this.
fn field_name(field: &str) -> &str {
    field
        .split_once(':')
        .map(|(name, _)| name)
        .unwrap_or(field)
        .trim()
}

/// The value a top-level struct-literal entry binds. Field shorthand (`preview,`) forwards the
/// binding in scope, so the entry is its own value.
fn field_value(field: &str) -> &str {
    field
        .split_once(':')
        .map(|(_, value)| value)
        .unwrap_or(field)
        .trim()
}

/// Whether `field` is a struct-update entry (`..Default::default()`, `..Foo::default()`, `..base`)
/// rather than a named field.
///
/// Any spelling counts, not just `::default()`. A rest-init of any kind silently supplies every
/// field the literal does not name, so `..some_prebuilt_request` hides an inert `preview` exactly as
/// thoroughly as `..Default::default()` does — and reads more innocent.
fn is_rest_init(field: &str) -> bool {
    field.trim_start().starts_with("..")
}

/// Brace blocks in `code` that name a **bespoke** provider request — a `*Request` type the worker
/// invokes by name, so the registry cannot thread the job's sink onto it.
fn bespoke_request_blocks(code: &str) -> Vec<(String, Vec<String>)> {
    brace_blocks(code)
        .into_iter()
        .filter(|(name, _)| {
            name.ends_with("Request") && !NON_BESPOKE_REQUESTS.contains(&name.as_str())
        })
        .collect()
}

/// Bespoke request literals in `code` that name no `preview` field and fill the gap with a
/// struct-update entry, rendered for a failure message.
///
/// The judgment lives here rather than inline in the test so
/// [`preview_binding_parser_recognises_the_shapes_it_must_judge`] can exercise the same code path on
/// hand-written snippets. A rule that is only ever run over a clean tree proves nothing.
fn rest_init_offenders(code: &str) -> Vec<String> {
    let mut offenders = Vec::new();
    for (literal, fields) in bespoke_request_blocks(code) {
        if fields.iter().any(|field| field_name(field) == "preview") {
            continue;
        }
        for field in fields.iter().filter(|field| is_rest_init(field)) {
            offenders.push(format!("{literal} {{ {} }}", collapse(field)));
        }
    }
    offenders
}

/// Every `TypeName { .. }` brace block in `code`, as `(type name, top-level field entries)`.
///
/// Scans complete identifier tokens rather than searching for a substring: `Krea2ControlRequest` does
/// NOT contain a token boundary before `Request`, so a substring search for `"Request"` matches
/// nothing and a sweep built on one silently passes forever. Tracks paren/bracket depth alongside
/// brace depth so a comma inside `foo(a, b)` does not split a field, and a nested literal's fields are
/// attributed to the nested literal rather than to the outer one.
///
/// Deliberately loose about what a "struct literal" is — `impl Foo {` and match patterns come back
/// too. Callers filter by name, and neither shape can carry a `preview:` field bound to a default.
fn brace_blocks(code: &str) -> Vec<(String, Vec<String>)> {
    let bytes = code.as_bytes();
    let mut blocks = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        // Identifier tokens only, anchored at a real token boundary.
        if !(bytes[index].is_ascii_uppercase()) {
            index += 1;
            continue;
        }
        let preceded_by_ident = index
            .checked_sub(1)
            .map(|i| bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric())
            .unwrap_or(false);
        if preceded_by_ident {
            index += 1;
            continue;
        }
        let mut end = index;
        while end < bytes.len() && (bytes[end] == b'_' || bytes[end].is_ascii_alphanumeric()) {
            end += 1;
        }
        let name = code[index..end].to_owned();
        let mut cursor = end;
        while cursor < bytes.len() && (bytes[cursor] as char).is_whitespace() {
            cursor += 1;
        }
        if cursor >= bytes.len() || bytes[cursor] != b'{' {
            index = end;
            continue;
        }

        let mut braces = 1i32;
        let mut nesting = 0i32;
        let mut fields: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut scan = cursor + 1;
        while scan < bytes.len() {
            let byte = bytes[scan] as char;
            match byte {
                '{' => braces += 1,
                '}' => {
                    braces -= 1;
                    if braces == 0 {
                        break;
                    }
                }
                '(' | '[' => nesting += 1,
                ')' | ']' => nesting -= 1,
                ',' if braces == 1 && nesting == 0 => {
                    fields.push(current.trim().to_owned());
                    current.clear();
                    scan += 1;
                    continue;
                }
                _ => {}
            }
            current.push(byte);
            scan += 1;
        }
        if !current.trim().is_empty() {
            fields.push(current.trim().to_owned());
        }
        blocks.push((name, fields.into_iter().filter(|f| !f.is_empty()).collect()));
        // Continue INSIDE the block so nested literals are found too.
        index = cursor + 1;
    }
    blocks
}

/// Top-level field entries of every `literal { .. }` block in `code`.
fn struct_literal_fields(code: &str, literal: &str) -> Vec<Vec<String>> {
    brace_blocks(code)
        .into_iter()
        .filter(|(name, _)| name == literal)
        .map(|(_, fields)| fields)
        .collect()
}

/// The `preview` field entries of every `literal` struct literal in `code`, as written.
fn preview_bindings(code: &str, literal: &str) -> Vec<String> {
    struct_literal_fields(code, literal)
        .into_iter()
        .flatten()
        .filter(|field| field_name(field) == "preview")
        .collect()
}

fn source_for(file: &str) -> String {
    let sources = image_jobs_sources();
    sources
        .into_iter()
        .find(|(name, _)| name == file)
        .unwrap_or_else(|| panic!("src/image_jobs/{file} must exist"))
        .1
}

/// The headline guard: each bespoke request literal feeds `preview` a live expression.
///
/// Reverting any one of the three call sites to `preview: Default::default()` — the change that
/// compiles, ships, and silently emits nothing — fails here, on every platform, on every PR.
#[test]
fn bespoke_candle_requests_bind_a_live_preview_sink() {
    for (literal, file) in WIRED_LANES {
        let code = source_for(file);
        let bindings = preview_bindings(&code, literal);
        assert_eq!(
            bindings.len(),
            1,
            "src/image_jobs/{file}: expected exactly one `{literal}` literal binding `preview`, \
             found {}. This lane is invoked by name, so the registry cannot thread the sink for it.",
            bindings.len()
        );
        let binding = &bindings[0];
        let value = binding
            .split_once(':')
            .map(|(_, value)| value.trim().to_owned())
            // Field shorthand (`preview,`) forwards the binding in scope, which is the live sink.
            .unwrap_or_else(|| "preview".to_owned());
        assert!(
            !value.is_empty(),
            "src/image_jobs/{file}: `{literal}.preview` has no value"
        );
        assert!(
            !is_default_expression(&value),
            "src/image_jobs/{file}: `{literal}.preview` is bound to the INERT default \
             (`{value}`). That compiles and ships a lane that emits no live preview at all — the \
             sc-17054 regression this guard exists for. Pass the job's live sink, which reaches this \
             lane as the third parameter of the `drive_gen_items*` closure."
        );
    }
}

/// The sweep: nothing under `src/image_jobs/` may default a `preview` field, including lanes not yet
/// listed in [`WIRED_LANES`]. When sc-16953…sc-16959 add the field to their own request structs, the
/// compile break must be fixed by wiring, not by defaulting — and this catches the difference without
/// anyone remembering to amend a list.
#[test]
fn no_candle_image_lane_defaults_its_preview_sink() {
    let mut offenders = Vec::new();
    let mut swept = 0usize;
    for (file, code) in image_jobs_sources() {
        for (literal, fields) in brace_blocks(&code) {
            for field in fields {
                // Field shorthand (`preview,`) is its own value — the binding in scope, never a default.
                if field_name(&field) != "preview" {
                    continue;
                }
                swept += 1;
                let value = field_value(&field);
                if is_default_expression(value) {
                    offenders.push(format!(
                        "src/image_jobs/{file}: {literal} {{ preview: {} }}",
                        collapse(value)
                    ));
                }
            }
        }
    }
    // A sweep that inspected nothing is indistinguishable from a clean sweep. Each wired lane
    // contributes exactly one `preview` field binding, so fewer means the scan broke, not the lanes.
    assert!(
        swept >= WIRED_LANES.len(),
        "the preview sweep inspected only {swept} `preview` bindings under src/image_jobs/, expected \
         at least {} — the source scan is broken, not the lanes",
        WIRED_LANES.len()
    );
    assert!(
        offenders.is_empty(),
        "a candle image lane binds its preview sink to the inert default, so that lane renders with \
         no live preview:\n  {}\nPass the live sink from the `drive_gen_items*` closure instead.",
        offenders.join("\n  ")
    );
}

/// The other half of the sweep: a bespoke request may not fill `preview` from a struct-update entry.
///
/// [`no_candle_image_lane_defaults_its_preview_sink`] only inspects fields the literal NAMES, so it
/// reads `Flux2ControlRequest { ..Default::default(), prompt }` as clean — the entry's field name
/// parses as `..Default`, never `preview`. That is not a hypothetical spelling: when sc-16953…
/// sc-16959 add `preview` to their own request structs upstream, these exhaustive literals stop
/// compiling, and `..Default::default()` is the shortest edit that makes them compile again. It ships
/// an inert sink identical to `preview: Default::default()`, and nothing else here would be red.
///
/// The three already-wired lanes are covered by the `bindings.len() == 1` count in
/// [`bespoke_candle_requests_bind_a_live_preview_sink`] — reverting one of THOSE to a rest-init drops
/// the count to zero. This test covers the twelve lanes that are not wired yet, which is the entire
/// population the sweep was built for.
#[test]
fn no_bespoke_candle_request_hides_its_preview_behind_a_rest_init() {
    let mut offenders = Vec::new();
    let mut inspected = 0usize;
    for (file, code) in image_jobs_sources() {
        // `tests.rs` is the crate's test corpus, not a shipping lane. Its provider requests are
        // fixtures whose preview is legitimately inert — `InstantIdRequest { ..InstantIdRequest::
        // default() }` in the real-weight angle harness, for one — and no user ever sees a card for
        // them. The lanes a family story actually edits all live in their own files.
        if file == "tests.rs" {
            continue;
        }
        inspected += bespoke_request_blocks(&code).len();
        offenders.extend(
            rest_init_offenders(&code)
                .into_iter()
                .map(|offender| format!("src/image_jobs/{file}: {offender}")),
        );
    }
    // A sweep that inspected nothing is indistinguishable from a clean sweep. Each wired lane's
    // literal is itself a bespoke `*Request` block, so fewer than that many means the scan broke.
    assert!(
        inspected >= WIRED_LANES.len(),
        "the rest-init sweep inspected only {inspected} bespoke `*Request` blocks under \
         src/image_jobs/, expected at least {} — the source scan is broken, not the lanes",
        WIRED_LANES.len()
    );
    assert!(
        offenders.is_empty(),
        "a bespoke candle request fills `preview` from a struct-update entry instead of naming it, \
         so that lane ships an inert sink and renders with no live preview:\n  {}\nWhen an \
         exhaustive literal stops compiling because upstream added `preview`, the fix is to pass the \
         live sink from the `drive_gen_items*` closure — not to absorb the new field into a default.",
        offenders.join("\n  ")
    );
}

/// The upstream half: the worker closures that RECEIVE the live sink must bind it. A `_preview`
/// parameter in one of these files means the sink is discarded before any request is built, which no
/// amount of correctness at the struct literal can repair.
#[test]
fn candle_preview_consumers_do_not_discard_their_sink() {
    for file in SINK_CONSUMERS {
        let code = source_for(file);
        assert!(
            !code.contains("_preview,"),
            "src/image_jobs/{file}: a `drive_gen_items*` closure still binds `_preview` and drops \
             the job's live sink. Bind `preview` and forward it to the bespoke request."
        );
        assert!(
            code.contains("preview"),
            "src/image_jobs/{file}: expected this lane to thread a preview sink"
        );
    }
}

/// Self-test the parser rather than trusting it: a guard that silently matches nothing is worse than
/// no guard. Covers the shorthand form, the `::default()` family, nested literals, and commas inside
/// call arguments.
#[test]
fn preview_binding_parser_recognises_the_shapes_it_must_judge() {
    let live = "let r = QwenEditRequest { prompt, memory: pick(a, b), preview };";
    assert_eq!(preview_bindings(live, "QwenEditRequest"), vec!["preview"]);

    let cloned = "Krea2ControlRequest { seed, preview: preview.clone() }";
    let bindings = preview_bindings(cloned, "Krea2ControlRequest");
    assert_eq!(bindings.len(), 1);
    assert!(!is_default_expression(
        bindings[0].split_once(':').unwrap().1
    ));

    for inert in [
        "Default::default()",
        "PreviewSink::default()",
        "gen_core::PreviewSink::default()",
    ] {
        assert!(is_default_expression(inert), "{inert} must read as inert");
    }
    assert!(!is_default_expression("preview.clone()"));
    assert!(!is_default_expression("preview"));

    // A nested literal's own fields must not be attributed to the outer one.
    let nested = "Outer { inner: QwenFunControlRequest { preview: preview.clone() }, other: 1 }";
    assert_eq!(
        preview_bindings(nested, "QwenFunControlRequest"),
        vec!["preview: preview.clone()"]
    );

    // A longer identifier that merely contains the name is not the literal.
    assert!(
        preview_bindings("MyQwenEditRequestBuilder { preview: x }", "QwenEditRequest").is_empty()
    );

    // The sweep walks blocks by NAME, so it must see the literal even though `Krea2ControlRequest`
    // has no token boundary before `Request`. A substring-based scan finds nothing here, passes
    // forever, and is exactly the false green this self-test exists to prevent.
    let swept: Vec<String> = brace_blocks("Krea2ControlRequest { preview: Default::default() }")
        .into_iter()
        .flat_map(|(name, fields)| fields.into_iter().map(move |f| format!("{name}::{f}")))
        .collect();
    assert_eq!(
        swept,
        vec!["Krea2ControlRequest::preview: Default::default()"]
    );

    // Struct-update init. `..Default::default()` arrives as a top-level entry whose field NAME parses
    // as `..Default`, so every named-field check reads this literal as clean while it ships the same
    // inert sink as `preview: Default::default()`. This exact snippet was mutated into an unlisted
    // lane during review and all four guards stayed green.
    let rest_init = "Flux2ControlRequest { ..Default::default(), prompt: p.clone() }";
    assert!(
        preview_bindings(rest_init, "Flux2ControlRequest").is_empty(),
        "the named-field parse must NOT see a preview binding here — that blindness is the hole"
    );
    assert_eq!(field_name("..Default::default()"), "..Default");
    assert!(is_rest_init("..Default::default()"));
    assert!(is_rest_init("..Flux2ControlRequest::default()"));
    // Any struct-update spelling hides the field, not just the `::default()` family.
    assert!(is_rest_init("..prebuilt"));
    assert!(!is_rest_init("preview: preview.clone()"));
    assert_eq!(
        rest_init_offenders(rest_init),
        vec!["Flux2ControlRequest { ..Default::default() }"],
        "an unlisted bespoke lane that absorbs the new `preview` field into a rest-init must be an \
         offender"
    );

    // Naming `preview` next to a rest-init is the CORRECT shape — the other fields are genuinely
    // defaulted and the sink is genuinely live — so it must not be flagged.
    assert!(rest_init_offenders(
        "Flux2ControlRequest { preview: preview.clone(), ..Default::default() }"
    )
    .is_empty());
    // The registry-driven request threads its sink through `base.rs`; the tile-refinement pass in
    // `detail.rs` omits it on purpose. Neither is a bespoke by-name lane.
    assert!(rest_init_offenders("GenerationRequest { prompt, ..Default::default() }").is_empty());
    assert!(rest_init_offenders("ImageRequest { model, ..Default::default() }").is_empty());
    // A rest-init with no `preview` in a non-request literal is none of this guard's business.
    assert!(rest_init_offenders("Sampling { steps, ..Default::default() }").is_empty());
}
