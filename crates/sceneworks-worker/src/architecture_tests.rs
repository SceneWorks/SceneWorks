//! Executable drift guards for `ARCHITECTURE.md`.
//!
//! The capability matrix is checked against source text rather than a second hand-maintained Rust
//! list: adding a `JobType` or omitting an explicit worker dispatch must make this test fail until
//! the documentation and dispatcher are reconciled.

use std::collections::{BTreeMap, BTreeSet};

const CONTRACTS: &str = include_str!("../../sceneworks-core/src/contracts.rs");
const WORKER: &str = include_str!("lib.rs");
const ARCHITECTURE: &str = include_str!("../ARCHITECTURE.md");
const FACE_ANALYSIS_JOBS: &str = include_str!("face_analysis_jobs.rs");

fn known_job_types() -> BTreeMap<String, String> {
    let body = CONTRACTS
        .split_once("pub enum JobType {")
        .expect("contracts.rs must declare JobType")
        .1
        .split_once("\n    }")
        .expect("JobType declaration must have a closing brace")
        .0;

    body.lines()
        .filter_map(|line| {
            let (variant, wire) = line.split_once("=>")?;
            let wire = wire.trim().strip_prefix('"')?.split('"').next()?;
            Some((variant.trim().to_owned(), wire.to_owned()))
        })
        .collect()
}

fn documented_job_types() -> Vec<String> {
    let matrix = ARCHITECTURE
        .split_once("<!-- job-matrix:start -->")
        .expect("architecture must mark the job matrix start")
        .1
        .split_once("<!-- job-matrix:end -->")
        .expect("architecture must mark the job matrix end")
        .0;

    matrix
        .lines()
        .filter_map(|line| {
            let value = line.strip_prefix("| `")?;
            Some(value.split('`').next()?.to_owned())
        })
        .collect()
}

/// The end of a `'x'` / `'\n'` / `'\''` / `'\u{1F600}'` char literal starting at `start`, or `None`
/// when the quote opens a lifetime (`'a`, `'_`, `'static`).
///
/// Without this, a `'"'` char literal — `downloads.rs`, `session_log.rs`, `jsonc.rs` and four other
/// scanned files contain one — flips the string-literal parity of everything that follows it in the
/// file. The consequence is not a false positive but a silent BLIND SPOT: the tail of the file gets
/// absorbed into a phantom string literal and stops being scanned at all.
///
/// # The two ways to get this wrong, both of which shipped
///
/// * **Escapes.** The closing quote of `'\''` is the byte at `start + 3`; the byte at `start + 2` is
///   the *escaped* quote. Searching from `start + 2` ends the literal one byte early and orphans a
///   `'`, which then eats the opening `"` of the next string literal and inverts parity for the rest
///   of the file. `media_jobs.rs:3087` (`.replace('\'', "'\\''")`) put its whole tail out of reach of
///   both gates that way — strictly worse than the stripper this replaced.
/// * **Lifetimes.** "a quote somewhere in the next five bytes" reads `Foo<'a, 'b>` as a char literal
///   spanning `'a, '`, which can again leave the cursor on a `"`. A plain char literal is exactly one
///   code point wide, so the closing quote must sit *immediately* after it — required here, never
///   searched for.
pub(crate) fn char_literal_end(bytes: &[u8], start: usize) -> Option<usize> {
    if bytes.get(start) != Some(&b'\'') {
        return None;
    }
    if bytes.get(start + 1) == Some(&b'\\') {
        // The escaped byte is at `start + 2` and is never the terminator. Escapes run from `'\n'`
        // (4 bytes) to `'\u{1F600}'` (11).
        return (start + 3..=start + 11)
            .find(|index| bytes.get(*index) == Some(&b'\''))
            .map(|index| index + 1);
    }
    let width = match *bytes.get(start + 1)? {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => return None,
    };
    (bytes.get(start + 1 + width) == Some(&b'\'')).then_some(start + 2 + width)
}

/// A raw string literal (`r"…"`, `r#"…"#`, `br#"…"#`) whose `r`/`b` prefix starts at `start`, as
/// `(number of hashes, index just past the whole literal)`.
///
/// A raw string can contain unescaped `"` — `ideogram_caption.rs` has `r#"a "quoted" fox"#` — which
/// the plain-string walk closes early on, flipping parity exactly as a `'"'` char literal does.
pub(crate) fn raw_string_parts(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let mut index = start;
    if bytes.get(index) == Some(&b'b') {
        index += 1;
    }
    if bytes.get(index) != Some(&b'r') {
        return None;
    }
    index += 1;
    let hash_start = index;
    while bytes.get(index) == Some(&b'#') {
        index += 1;
    }
    let hashes = index - hash_start;
    if bytes.get(index) != Some(&b'"') {
        return None;
    }
    index += 1;
    while index < bytes.len() {
        if bytes[index] == b'"'
            && bytes[index + 1..]
                .iter()
                .take(hashes)
                .filter(|byte| **byte == b'#')
                .count()
                == hashes
        {
            return Some((hashes, index + 1 + hashes));
        }
        index += 1;
    }
    Some((hashes, bytes.len()))
}

/// Whether the byte at `index` could START an `r"…"` / `b"…"` prefix rather than continue an
/// identifier (`for r in …` must not read `r` as a raw-string prefix, but `let x = r"…"` must).
fn starts_a_token(bytes: &[u8], index: usize) -> bool {
    index
        .checked_sub(1)
        .map(|previous| !(bytes[previous] == b'_' || bytes[previous].is_ascii_alphanumeric()))
        .unwrap_or(true)
}

/// Remove comments — and nothing else — before inspecting Rust syntax. Understands nested block
/// comments and steps over quoted strings, raw strings and char literals, so a `//` inside a URL
/// literal is not mistaken for the start of a comment and a `'"'` does not open a phantom string.
///
/// String literal CONTENTS survive, which is what separates this from
/// [`code_without_comments_or_literals`]. [`crate::job_time_download_guard`] (sc-17637) needs both:
/// its download-reachability sweep must not let a doc-comment mention of `ensure_hf_cached_file`
/// count as a call, while its `<data_dir>/cache` sweep is looking for `.join("cache")` — a
/// destination that only exists *as* a string literal, and which the blanking pass below erases.
pub(crate) fn code_without_comments(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        // Char literals and raw strings first: both can carry a `"` that must not open a string.
        if bytes[index] == b'\'' {
            if let Some(end) = char_literal_end(bytes, index) {
                output.extend_from_slice(&bytes[index..end.min(bytes.len())]);
                index = end.min(bytes.len());
                continue;
            }
        }
        if (bytes[index] == b'r' || bytes[index] == b'b') && starts_a_token(bytes, index) {
            if let Some((_, end)) = raw_string_parts(bytes, index) {
                let end = end.min(bytes.len());
                output.extend_from_slice(&bytes[index..end]);
                index = end;
                continue;
            }
        }
        match (bytes[index], bytes.get(index + 1).copied()) {
            (b'/', Some(b'/')) => {
                index += 2;
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            (b'/', Some(b'*')) => {
                index += 2;
                let mut depth = 1_u32;
                while index < bytes.len() && depth > 0 {
                    match (bytes[index], bytes.get(index + 1).copied()) {
                        (b'/', Some(b'*')) => {
                            depth += 1;
                            index += 2;
                        }
                        (b'*', Some(b'/')) => {
                            depth -= 1;
                            index += 2;
                        }
                        _ => index += 1,
                    }
                }
            }
            (b'"', _) => {
                output.push(b'"');
                index += 1;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' => {
                            output.push(b'\\');
                            if let Some(escaped) = bytes.get(index + 1).copied() {
                                output.push(escaped);
                            }
                            index += 2;
                        }
                        b'"' => {
                            output.push(b'"');
                            index += 1;
                            break;
                        }
                        byte => {
                            output.push(byte);
                            index += 1;
                        }
                    }
                }
            }
            (byte, _) => {
                output.push(byte);
                index += 1;
            }
        }
    }

    String::from_utf8(output).expect("Rust source outside comments remains UTF-8")
}

/// Remove comments and literal contents before inspecting Rust syntax. This is intentionally small
/// (the production source remains compiled by rustc), but it understands nested block comments and
/// escaped quoted strings so a `JobType::Variant` mention outside code cannot satisfy the guard.
///
/// Shared with [`crate::candle_preview_wiring_tests`] (sc-16962), whose guard reads the candle image
/// lanes' own source: without comment stripping, a doc comment that NAMES the forbidden
/// `preview: Default::default()` would trip the guard it documents. [`crate::job_time_download_guard`]
/// (sc-17637) uses it for the same reason — ~25 doc comments name `ensure_hf_cached_file` in files
/// that never call it.
pub(crate) fn code_without_comments_or_literals(source: &str) -> String {
    let comment_free = code_without_comments(source);
    let bytes = comment_free.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        // A char literal is kept verbatim — callers look for `'{'` — but it must be *stepped over*,
        // or the `"` in a `'"'` literal opens a phantom string that blanks the rest of the file.
        if bytes[index] == b'\'' {
            if let Some(end) = char_literal_end(bytes, index) {
                let end = end.min(bytes.len());
                output.extend_from_slice(&bytes[index..end]);
                index = end;
                continue;
            }
        }
        // A raw string keeps its delimiter shape and loses its contents, like a plain one.
        if (bytes[index] == b'r' || bytes[index] == b'b') && starts_a_token(bytes, index) {
            if let Some((hashes, end)) = raw_string_parts(bytes, index) {
                if bytes[index] == b'b' {
                    output.push(b'b');
                }
                output.push(b'r');
                output.extend(std::iter::repeat(b'#').take(hashes));
                output.extend_from_slice(b"\"\"");
                output.extend(std::iter::repeat(b'#').take(hashes));
                index = end.min(bytes.len());
                continue;
            }
        }
        match bytes[index] {
            b'"' => {
                output.push(b'"');
                index += 1;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' => index += 2,
                        b'"' => {
                            output.push(b'"');
                            index += 1;
                            break;
                        }
                        _ => index += 1,
                    }
                }
            }
            byte => {
                output.push(byte);
                index += 1;
            }
        }
    }

    String::from_utf8(output).expect("Rust source outside comments and literals remains UTF-8")
}

#[test]
fn catalog_face_adapter_uses_each_backends_actual_score_field() {
    let code = code_without_comments_or_literals(FACE_ANALYSIS_JOBS);
    let mlx_arm = code
        .split_once("CatalogFaceBackend::Mlx(analysis) => {")
        .expect("catalog face adapter must keep an MLX conversion arm")
        .1
        .split_once("CatalogFaceBackend::Candle(analysis) =>")
        .expect("MLX conversion must precede the candle conversion")
        .0;
    assert!(
        mlx_arm.contains("confidence: face.score"),
        "mlx-gen-face::Detection exposes `score`, not gen-core's `det_score`"
    );
    assert!(
        !mlx_arm.contains("face.det_score"),
        "the MLX conversion must not use the candle result field"
    );

    let candle_arm = code
        .split_once("CatalogFaceBackend::Candle(analysis) =>")
        .expect("catalog face adapter must keep a candle conversion arm")
        .1
        .split_once("Ok((image.width, image.height, detections))")
        .expect("candle conversion must remain inside CatalogFaceDetector::detect")
        .0;
    assert!(
        candle_arm.contains("confidence: face.det_score"),
        "gen-core's candle face-analysis result exposes `det_score`"
    );
    assert!(
        candle_arm.contains(".detect(&image)"),
        "catalog face counting must use the candle detection-only path"
    );
    assert!(
        !candle_arm.contains(".analyze(&image)"),
        "catalog face counting must not compute unused ArcFace embeddings"
    );
}

/// Extract only `JobType` variants occurring in top-level patterns of
/// `run_utility_job`'s `match job.job_type`. Mentions in comments, strings, helper calls, or arm
/// bodies are deliberately ignored.
fn explicit_dispatch_variants(source: &str) -> BTreeSet<String> {
    let code = code_without_comments_or_literals(source);
    let function = code
        .split_once("async fn run_utility_job(")
        .expect("worker must declare run_utility_job")
        .1;
    let match_body = function
        .split_once("match job.job_type {")
        .expect("run_utility_job must match job.job_type")
        .1;

    let bytes = match_body.as_bytes();
    let mut variants = BTreeSet::new();
    let mut arm_start = 0;
    let mut braces = 0_u32;
    let mut parentheses = 0_u32;
    let mut brackets = 0_u32;
    let mut index = 0;

    while index < bytes.len() {
        match bytes[index] {
            b'{' => braces += 1,
            b'}' if braces == 0 => break,
            b'}' => braces -= 1,
            b'(' => parentheses += 1,
            b')' => parentheses -= 1,
            b'[' => brackets += 1,
            b']' => brackets -= 1,
            b'=' if bytes.get(index + 1) == Some(&b'>')
                && braces == 0
                && parentheses == 0
                && brackets == 0 =>
            {
                let pattern = &match_body[arm_start..index];
                let mut rest = pattern;
                while let Some((_, after_prefix)) = rest.split_once("JobType::") {
                    let variant: String = after_prefix
                        .chars()
                        .take_while(|character| {
                            character.is_ascii_alphanumeric() || *character == '_'
                        })
                        .collect();
                    if !variant.is_empty() {
                        variants.insert(variant);
                    }
                    rest = after_prefix;
                }
                index += 1;
            }
            b',' if braces == 0 && parentheses == 0 && brackets == 0 => arm_start = index + 1,
            _ => {}
        }
        index += 1;
    }

    variants
}

#[test]
fn architecture_matrix_covers_every_known_job_type_and_dispatch_arm() {
    let known = known_job_types();
    let rows = documented_job_types();
    let documented: BTreeSet<_> = rows.iter().cloned().collect();
    let expected: BTreeSet<_> = known.values().cloned().collect();

    assert_eq!(
        rows.len(),
        documented.len(),
        "ARCHITECTURE.md has duplicate JobType rows"
    );
    assert_eq!(
        documented, expected,
        "ARCHITECTURE.md must have exactly one row for every known JobType"
    );

    let dispatched = explicit_dispatch_variants(WORKER);
    let expected_variants: BTreeSet<_> = known.keys().cloned().collect();
    assert_eq!(
        dispatched, expected_variants,
        "run_utility_job must have an explicit match arm for every known JobType"
    );

    assert!(
        !ARCHITECTURE.contains("Proof (file:line)"),
        "architecture anchors must name durable modules/functions, not line numbers"
    );
}

#[test]
fn dispatch_guard_rejects_removed_arm_even_when_a_comment_keeps_the_variant_name() {
    let mutated = WORKER.replacen(
        "JobType::PromptRefine =>",
        "// JobType::PromptRefine => removed arm\n            _removed_prompt_refine =>",
        1,
    );
    assert_ne!(mutated, WORKER, "mutation target must exist");
    assert!(
        !explicit_dispatch_variants(&mutated).contains("PromptRefine"),
        "a comment containing JobType::PromptRefine must not masquerade as a dispatch arm"
    );
}

#[test]
fn real_image_job_modules_import_the_json_macro_they_invoke() {
    let image_jobs = include_str!("image_jobs.rs");
    let module_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/image_jobs");

    for module in image_jobs.lines().filter_map(|line| {
        line.trim()
            .strip_prefix("mod ")
            .and_then(|line| line.strip_suffix(';'))
    }) {
        let path = module_dir.join(format!("{module}.rs"));
        if !path.is_file() {
            continue;
        }
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let code = code_without_comments_or_literals(&source);
        if !code.contains("json!") {
            continue;
        }
        let imports_json = code.split(';').any(|statement| {
            statement.trim_start().starts_with("use serde_json")
                && statement
                    .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                    .any(|word| word == "json")
        });
        assert!(
            imports_json,
            "{} invokes json! but does not explicitly import serde_json::json",
            path.display()
        );
    }
}

// ---------------------------------------------------------------------------
// epic 20398 / sc-20651 — "no bespoke imported route or family allow-list outside adapters"
// ---------------------------------------------------------------------------
//
// ## What the epic requires
//
// E2: adding a checkpoint family must need "one registered adapter and fixtures — no API,
// scheduler, worker or catalog allow-list edits". The epic acceptance matrix turns that into an
// AUDIT obligation: no bespoke imported route and no family allow-list may live outside the
// adapters.
//
// ## Why an inventory rather than a flat prohibition
//
// The prohibition is not true today and writing it as one would produce either a red tree or a
// vacuous scan. Nine family allow-lists exist at this commit, every one of them with a reason that
// is a fact about a real deployment or a real crate graph rather than about laziness. So the guard
// is shaped the way `crate::candle_preview_wiring_tests` is: an explicit inventory of what is
// known and recorded, a directory-wide sweep for everything that is not, a non-vacuity floor so a
// broken scanner cannot read as a clean tree, and an actionable message.
//
// Both directions are asserted, which is what makes the inventory shrink visibly:
//
// * an allow-list the sweep finds that is NOT recorded is an offender — a NEW family gate;
// * a recorded entry the sweep does NOT find is also an offender — the lane went away and its
//   exception must go with it.
//
// ## The limit of this audit, stated rather than hidden
//
// The sweep recognises a family allow-list by the family TOKENS it names, from an explicit
// vocabulary (`CHECKPOINT_PLAN_FAMILY_TOKENS`, the `checkpoint_inspector::normalize_family`
// outputs). A gate for a family that vocabulary does not yet contain is invisible to it. That hole
// is closed as far as source text can close it by
// `the_family_token_vocabulary_covers_every_family_the_live_tables_name`, which fails the moment a
// new family reaches any of the three live tables without reaching the vocabulary.
//
// A second hole is not closable from source text at all, and is stated rather than papered over: a
// constant naming exactly ONE token, and that token an AMBIGUOUS one, is not judged a family
// allow-list ([`FAMILY_TOKENS_THAT_ARE_ALSO_BUILTIN_MODEL_IDS`]). `sdxl` is that token, and a
// genuine single-family `sdxl` gate is byte-identical to a builtin-model-id list holding the one
// word `sdxl` — the sweep sees a set of one ambiguous token in both cases and cannot tell them
// apart. It resolves that tie toward NOT an offender, so an SDXL-only family gate would pass this
// audit unrecorded. Widening the judgment the other way is the worse trade (it would record ten
// non-offenders and bury the four that matter), and no amount of token analysis distinguishes the
// two: only a reader can. Any second vocabulary token puts the constant back in the sweep's view.

/// The plan-family vocabulary: every value `sceneworks_core::checkpoint_inspector::normalize_family`
/// can return.
///
/// Spelled here rather than imported because `normalize_family` is private to the inspector and its
/// outputs are a `match` arm's right-hand side, not a list. Kept honest by
/// [`the_family_token_vocabulary_covers_every_family_the_live_tables_name`].
const CHECKPOINT_PLAN_FAMILY_TOKENS: &[&str] = &[
    "sdxl",
    "qwen-image",
    "flux2",
    "flux",
    "wan-video",
    "z-image",
    "mage-flow",
    "krea_2",
    "ltx-video",
    "ideogram",
    "anima",
];

/// Family tokens that are ALSO a builtin model id, and are therefore ambiguous on their own.
///
/// `sdxl` is the only one: the family the inspector detects and the builtin SDXL model id are the
/// same string. Ten constants in the swept tree are pure builtin-model-id lists whose sole overlap
/// with the vocabulary is that word — `SDXL_CONTROL_MODELS`, `CANDLE_POSE_MODELS`, `IMAGE_MODEL_CAPS`
/// and friends. None of them is a family gate, and recording ten non-offenders as exceptions would
/// bury the four that matter.
///
/// So a constant naming exactly ONE token, and that token an ambiguous one, is not judged a family
/// allow-list. Naming two or more tokens, or naming a single unambiguous token (`mage-flow`,
/// `krea_2`, `wan-video`, …), is.
const FAMILY_TOKENS_THAT_ARE_ALSO_BUILTIN_MODEL_IDS: &[&str] = &["sdxl"];

/// Directory roots swept for family allow-lists, relative to this crate's manifest directory.
///
/// The three surfaces E2 names, whole: the worker crate (its image and video lanes, its scheduler
/// and dispatch, and the one-time catalog migration), the catalog/routing crate, and the API.
///
/// Read from disk, recursively, rather than enumerated with `include_str!` so a NEW file — or a new
/// MODULE, which is how a family gate would most plausibly arrive — is swept the day it is added. A
/// hardcoded file list is how an inventory-shaped guard goes stale.
const FAMILY_ALLOW_LIST_SCAN_ROOTS: &[&str] =
    &["src", "../sceneworks-core/src", "../../apps/rust-api/src"];

/// The one swept file exempt from the sweep: this one.
///
/// [`CHECKPOINT_PLAN_FAMILY_TOKENS`] is the audit's own vocabulary — the instrument it judges by,
/// not a gate any code consults — and [`RECORDED_FAMILY_ALLOW_LISTS`] is the inventory itself.
/// Recording the audit as an exception to the audit would be a circular row that can never shrink.
const FAMILY_ALLOW_LIST_SCAN_EXEMPT: &[&str] = &["src/architecture_tests.rs"];

/// Every family allow-list that exists at this commit, with the recorded reason it is retained.
///
/// `(path relative to a scan root's spec, constant name, reason)`. sc-20651 established each reason
/// by reading the code, not by asking the constant's own doc comment.
///
/// **`apps/rust-api/src` contributes ZERO rows, and that is the positive half of the audit.** The
/// API declares no family allow-list at all, so E2's "no API allow-list edits" is not an aspiration
/// here — it is the swept state, and one added row reds this test.
const RECORDED_FAMILY_ALLOW_LISTS: &[(&str, &str, &str)] = &[
    (
        "src/image_jobs/checkpoint_plan.rs",
        "CHECKPOINT_PLAN_BESPOKE_PLAN_SOURCED_FAMILIES",
        "MUST OUTLIVE the lanes. Its sole reader is `checkpoint_plan_shape_has_other_lane`; \
         deleting it collapses `checkpoint_plan_unservable_shape` into `checkpoint_plan_unservable` \
         and flips every LINKED checkpoint whose shape the plan route does not serve (LoRA, \
         reference, Hires.fix) from DECLINE to hard refuse.",
    ),
    (
        "src/image_jobs/checkpoint_plan.rs",
        "CHECKPOINT_PLAN_FAMILY_COMPONENT_RESOLVERS",
        "Catalog data, not family truth: the adapter's `component_topology` says WHICH components \
         are required; this says where THIS app installed them. Mage-Flow's fine-tunes live at \
         `<data>/models/finetunes/<loraId>`, outside `CheckpointPlanStore::installs_root()`.",
    ),
    (
        "src/image_jobs/checkpoint_plan.rs",
        "CHECKPOINT_PLAN_RESIDENT_BASE_TIERS",
        "Catalog data, not family truth: the adapter declares the `base-snapshot` dependency and \
         which families satisfy it; this is only where the app installed that tier.",
    ),
    (
        "src/mlx_fit_gate.rs",
        "RESIDENT_ONLY_AUDIT_FAMILIES",
        "Not an import gate and not production: a `#[cfg(test)]`, macOS-only SCOPE for the \
         Resident-only estimate-band audit, over shipped MANIFEST `family` values — a different \
         vocabulary that merely overlaps ours on `flux` and `ideogram`. It routes no checkpoint and \
         admits no import; widening it changes which manifest entries a memory audit measures.",
    ),
    (
        "../sceneworks-core/src/base_weights.rs",
        "IMPORT_SUPPORTED_FAMILIES",
        "NOT DERIVABLE at this commit. The adapter registry is \
         `gen_core::ProviderRegistry::checkpoint_adapters`, and neither `sceneworks-core` nor \
         `apps/rust-api` — the crate that owns this constant and the process that serves the import \
         endpoint — depends on `sceneworks-gen-core`. The registry is also cfg-gated to \
         macOS-or-candle: on the candle-off build `inference_runtime::media()` is EMPTY, so a \
         derived list would refuse every import there. And the registry binds at least six families \
         against this list's four (`flux2`, `krea_2`, `mage-flow`, `sdxl`), so deriving would \
         silently WIDEN what import accepts.",
    ),
    (
        "../sceneworks-core/src/checkpoint_inspector.rs",
        "MULTI_EXPERT_FAMILIES",
        "Checkpoint truth the adapter cannot carry: it is what makes `refine_multi_expert_role` \
         provably ADDITIVE, so no other family's already-compiled plan can change. An adapter is \
         resolved from the family this list helps determine, so deriving it from one is circular.",
    ),
    (
        "../sceneworks-core/src/jobs_store/routing/catalog.rs",
        "MLX_ROUTED_FAMILIES",
        "Catalog routing authority in a crate with no adapter-registry dependency (see \
         IMPORT_SUPPORTED_FAMILIES). Pinned against the import gate by \
         `mlx_routed_families_agree_with_import_supported_families`, so the two cannot drift.",
    ),
    (
        "../sceneworks-core/src/jobs_store/routing/catalog.rs",
        "CANDLE_ROUTED_FAMILIES",
        "The candle twin of MLX_ROUTED_FAMILIES, same reason, same parity assertion.",
    ),
    (
        "../sceneworks-core/src/jobs_store/routing/catalog.rs",
        "EXPECTED_MLX_ROUTED_FAMILIES",
        "A `#[cfg(test)]` fixture that pins MLX_ROUTED_FAMILIES by value. Not a production gate; it \
         exists so a silent edit to the production list is red.",
    ),
    (
        "../sceneworks-core/src/jobs_store/routing/catalog.rs",
        "EXPECTED_CANDLE_ROUTED_FAMILIES",
        "The candle twin of EXPECTED_MLX_ROUTED_FAMILIES.",
    ),
];

/// The live tables whose family membership [`CHECKPOINT_PLAN_FAMILY_TOKENS`] must cover.
const LIVE_FAMILY_TABLES: &[(&str, &str)] = &[
    (
        "src/image_jobs/checkpoint_plan.rs",
        "CHECKPOINT_PLAN_BESPOKE_PLAN_SOURCED_FAMILIES",
    ),
    (
        "../sceneworks-core/src/base_weights.rs",
        "IMPORT_SUPPORTED_FAMILIES",
    ),
    (
        "../sceneworks-core/src/checkpoint_inspector.rs",
        "MULTI_EXPERT_FAMILIES",
    ),
];

/// Every `const NAME: TYPE = &[ .. ];` (or `= [ .. ]`) in `code`, as
/// `(constant name, the bracketed body, the body's byte range in `code`)`.
///
/// The `[` is looked for AFTER the `=`, never in the type: `const HEX: &[u8; 16] = b"0..";` would
/// otherwise be read as a slice literal whose "body" is the type's length. Brackets inside string
/// literals are skipped, so a constant carrying `"a[b"` does not truncate the scan.
///
/// Deliberately loose about what it accepts — an `impl` associated const and a `#[cfg(test)]` one
/// both come back. Callers judge by the tokens the body NAMES, and a non-family constant names none.
fn slice_const_bodies(code: &str) -> Vec<(String, String, std::ops::Range<usize>)> {
    let bytes = code.as_bytes();
    let mut found = Vec::new();
    let mut index = 0usize;
    while let Some(offset) = code[index..].find("const ") {
        let at = index + offset;
        index = at + "const ".len();
        if !starts_a_token(bytes, at) {
            continue;
        }
        let mut cursor = index;
        while cursor < bytes.len() && (bytes[cursor] as char).is_whitespace() {
            cursor += 1;
        }
        let name_start = cursor;
        while cursor < bytes.len()
            && (bytes[cursor] == b'_' || bytes[cursor].is_ascii_alphanumeric())
        {
            cursor += 1;
        }
        if cursor == name_start {
            continue;
        }
        let name = code[name_start..cursor].to_owned();
        // The `=` must precede the declaration's `;`, and the `[` must follow the `=` (modulo
        // whitespace and one `&`) rather than living in the type.
        let Some(equals) = code[cursor..].find('=').map(|offset| cursor + offset) else {
            continue;
        };
        if code[cursor..equals].contains(';') {
            continue;
        }
        let mut open = equals + 1;
        while open < bytes.len() && ((bytes[open] as char).is_whitespace() || bytes[open] == b'&') {
            open += 1;
        }
        if open >= bytes.len() || bytes[open] != b'[' {
            index = equals + 1;
            continue;
        }
        let Some(close) = matching_bracket(bytes, open) else {
            continue;
        };
        found.push((name, code[open + 1..close].to_owned(), (open + 1)..close));
        index = close + 1;
    }
    found
}

/// The index of the `]` closing the `[` at `open`, skipping brackets inside string, raw-string and
/// char literals.
///
/// The char-literal and raw-string steps are the same quote-parity defence
/// [`string_literals`] needs, for the same reason: a `'"'` anywhere between the brackets otherwise
/// opens a phantom string, the closing `]` is never found, and the constant silently stops being
/// inspected at all.
fn matching_bracket(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut index = open;
    while index < bytes.len() {
        if bytes[index] == b'\'' {
            if let Some(end) = char_literal_end(bytes, index) {
                index = end.min(bytes.len()).max(index + 1);
                continue;
            }
        }
        if (bytes[index] == b'r' || bytes[index] == b'b') && starts_a_token(bytes, index) {
            if let Some((_, end)) = raw_string_parts(bytes, index) {
                index = end.min(bytes.len()).max(index + 1);
                continue;
            }
        }
        match bytes[index] {
            b'"' => {
                index += 1;
                while index < bytes.len() && bytes[index] != b'"' {
                    index += if bytes[index] == b'\\' { 2 } else { 1 };
                }
            }
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
        index += 1;
    }
    None
}

/// Every plain double-quoted string literal in `code`, as
/// `(contents, byte offset of the opening quote)`.
///
/// `code` must already have been through [`code_without_comments`], so a family name written in a
/// doc comment cannot be mistaken for a gate.
///
/// Char literals and raw strings are STEPPED OVER rather than parsed, for the reason
/// [`char_literal_end`] documents at length: a `'"'` char literal — seven swept files contain one —
/// otherwise opens a phantom string that swallows the rest of the file, and a raw string's
/// unescaped inner `"` closes early. Either one inverts quote parity, and the failure is silent
/// blindness rather than a red test. Neither shape can hold a family gate, so skipping is enough.
fn string_literals(code: &str) -> Vec<(String, usize)> {
    let bytes = code.as_bytes();
    let mut literals = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'\'' {
            if let Some(end) = char_literal_end(bytes, index) {
                index = end.min(bytes.len()).max(index + 1);
                continue;
            }
        }
        if (bytes[index] == b'r' || bytes[index] == b'b') && starts_a_token(bytes, index) {
            if let Some((_, end)) = raw_string_parts(bytes, index) {
                index = end.min(bytes.len()).max(index + 1);
                continue;
            }
        }
        if bytes[index] != b'"' {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        let content_start = index;
        while index < bytes.len() && bytes[index] != b'"' {
            index += if bytes[index] == b'\\' { 2 } else { 1 };
        }
        let end = index.min(bytes.len());
        if !code.is_char_boundary(content_start) || !code.is_char_boundary(end) {
            break;
        }
        literals.push((code[content_start..end].to_owned(), start));
        index = end + 1;
    }
    literals
}

/// The family tokens `body` names, as exact literal values.
fn family_tokens_named(body: &str) -> BTreeSet<String> {
    string_literals(body)
        .into_iter()
        .map(|(value, _)| value)
        .filter(|value| CHECKPOINT_PLAN_FAMILY_TOKENS.contains(&value.as_str()))
        .collect()
}

/// Whether a constant naming `tokens` is a family ALLOW-LIST rather than a model-id list that
/// happens to share a word with the vocabulary. See
/// [`FAMILY_TOKENS_THAT_ARE_ALSO_BUILTIN_MODEL_IDS`].
fn is_family_allow_list(tokens: &BTreeSet<String>) -> bool {
    match tokens.len() {
        0 => false,
        1 => !FAMILY_TOKENS_THAT_ARE_ALSO_BUILTIN_MODEL_IDS
            .contains(&tokens.iter().next().expect("one token").as_str()),
        _ => true,
    }
}

/// Every `.rs` file under the scan roots, as `(display path, comment-stripped source)`.
///
/// The display path is the scan root's own spec plus the path below it, so it is identical on every
/// machine — never an absolute path, which would make the inventory machine-dependent.
fn family_allow_list_sources() -> Vec<(String, String)> {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources = Vec::new();
    let mut exempted: Vec<String> = Vec::new();
    for spec in FAMILY_ALLOW_LIST_SCAN_ROOTS {
        let root = manifest.join(spec);
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir)
                .unwrap_or_else(|error| panic!("read {}: {error}", dir.display()));
            for entry in entries {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().is_none_or(|ext| ext != "rs") {
                    continue;
                }
                let below = path
                    .strip_prefix(&root)
                    .expect("scanned path is below its root")
                    .to_string_lossy()
                    .replace('\\', "/");
                let display = format!("{spec}/{below}");
                if FAMILY_ALLOW_LIST_SCAN_EXEMPT.contains(&display.as_str()) {
                    exempted.push(display);
                    continue;
                }
                let source = std::fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
                sources.push((display, code_without_comments(&source)));
            }
        }
    }
    sources.sort();
    exempted.sort();
    // Every exemption must have matched a real file. A misspelled exempt path skips nothing while
    // reading as a deliberate skip, and — worse — a path that stops existing leaves an exemption
    // standing that would silently absorb a future file of that name.
    assert_eq!(
        exempted,
        FAMILY_ALLOW_LIST_SCAN_EXEMPT
            .iter()
            .map(|path| (*path).to_owned())
            .collect::<Vec<_>>(),
        "every FAMILY_ALLOW_LIST_SCAN_EXEMPT path must name a file the sweep actually reached"
    );
    assert!(
        sources.len() > 100,
        "expected the worker lanes, the catalog crate and the API under \
         {FAMILY_ALLOW_LIST_SCAN_ROOTS:?}, found {} files — the scan is broken, not the tree",
        sources.len()
    );
    sources
}

/// The bracketed body of one named constant in one swept file.
fn recorded_const_body(file: &str, name: &str) -> String {
    let (_, code) = family_allow_list_sources()
        .into_iter()
        .find(|(path, _)| path == file)
        .unwrap_or_else(|| panic!("{file} must be a swept file"));
    slice_const_bodies(&code)
        .into_iter()
        .find(|(candidate, _, _)| candidate == name)
        .unwrap_or_else(|| panic!("{file} must declare `const {name}`"))
        .1
}

/// The headline audit: no family allow-list exists outside the recorded exception inventory, and no
/// recorded exception outlives the lane it was recorded for.
///
/// Failing mutation: add `const FAKE_IMPORT_FAMILIES: &[&str] = &["krea_2", "flux2"];` to any file
/// under a scan root. It is not in the inventory, so it is an offender and this test fails.
#[test]
fn no_family_allow_list_exists_outside_the_recorded_exception_inventory() {
    let recorded: BTreeSet<(&str, &str)> = RECORDED_FAMILY_ALLOW_LISTS
        .iter()
        .map(|(file, name, _)| (*file, *name))
        .collect();
    assert_eq!(
        recorded.len(),
        RECORDED_FAMILY_ALLOW_LISTS.len(),
        "the exception inventory names the same (file, constant) twice"
    );

    let mut found: BTreeSet<(String, String)> = BTreeSet::new();
    let mut unrecorded: Vec<String> = Vec::new();
    let mut inspected = 0usize;
    for (file, code) in family_allow_list_sources() {
        for (name, body, _) in slice_const_bodies(&code) {
            inspected += 1;
            let tokens = family_tokens_named(&body);
            if !is_family_allow_list(&tokens) {
                continue;
            }
            found.insert((file.clone(), name.clone()));
            if !recorded.contains(&(file.as_str(), name.as_str())) {
                unrecorded.push(format!(
                    "{file}: const {name} — names {:?}",
                    tokens.into_iter().collect::<Vec<_>>()
                ));
            }
        }
    }

    // A sweep that inspected nothing is indistinguishable from a clean sweep. Every recorded
    // exception is itself a slice constant, so fewer than that many means the SCANNER broke rather
    // than the tree becoming clean.
    assert!(
        inspected >= RECORDED_FAMILY_ALLOW_LISTS.len(),
        "the family allow-list sweep inspected only {inspected} slice constants across \
         {FAMILY_ALLOW_LIST_SCAN_ROOTS:?}, expected at least {} — the source scan is broken, not \
         the tree",
        RECORDED_FAMILY_ALLOW_LISTS.len()
    );
    assert!(
        unrecorded.is_empty(),
        "a family allow-list lives outside the checkpoint adapters and is not in the sc-20651 \
         exception inventory (epic 20398 E2 — adding a family must need one registered adapter and \
         fixtures, no API / scheduler / worker / catalog allow-list edits):\n  {}\nEither route the \
         decision through `gen_core::CheckpointAdapterRegistration`, or add a row to \
         `RECORDED_FAMILY_ALLOW_LISTS` stating why it cannot be.",
        unrecorded.join("\n  ")
    );

    // The other direction, which is what makes the inventory SHRINK rather than accumulate: a
    // recorded exception whose constant is gone must be removed from the inventory with it.
    let stale: Vec<String> = RECORDED_FAMILY_ALLOW_LISTS
        .iter()
        .filter(|(file, name, _)| !found.contains(&((*file).to_owned(), (*name).to_owned())))
        .map(|(file, name, _)| format!("{file}: const {name}"))
        .collect();
    assert!(
        stale.is_empty(),
        "the sc-20651 exception inventory records a family allow-list that no longer exists as \
         one:\n  {}\nDelete the row — the inventory is meant to shrink as lanes go away.",
        stale.join("\n  ")
    );
}

/// The plan-driven route is adapter-mediated: `checkpoint_plan.rs` names a family only inside its
/// three recorded catalog-data tables, never in a branch.
///
/// This is the "no bespoke imported route" half, and it is the half the allow-list sweep cannot see:
/// `if family == "flux2" { .. }` declares no constant. The route resolves its adapter, its provider,
/// its component topology and its capability policy from
/// `gen_core::CheckpointAdapterRegistration`, so a family literal anywhere else in this file is by
/// construction a bespoke route.
///
/// Failing mutation: add `if resolved.family() == "flux2" { .. }` anywhere in `checkpoint_plan.rs`
/// outside the three tables.
#[test]
fn the_checkpoint_plan_route_names_a_family_only_inside_its_recorded_tables() {
    let file = "src/image_jobs/checkpoint_plan.rs";
    let (_, code) = family_allow_list_sources()
        .into_iter()
        .find(|(path, _)| path == file)
        .expect("the plan route must be swept");

    let tables: Vec<std::ops::Range<usize>> = slice_const_bodies(&code)
        .into_iter()
        .filter(|(name, _, _)| {
            RECORDED_FAMILY_ALLOW_LISTS
                .iter()
                .any(|(recorded_file, recorded_name, _)| {
                    *recorded_file == file && recorded_name == name
                })
        })
        .map(|(_, _, range)| range)
        .collect();
    // Fixture check: all three of this file's recorded tables were located. Without it a scanner
    // that found zero tables would leave every literal "outside a table", and one that swallowed
    // the whole file would leave every literal inside one — opposite failures, both silent.
    assert_eq!(
        tables.len(),
        3,
        "expected the plan route's three recorded family tables, located {}",
        tables.len()
    );

    let mut outside = 0usize;
    let mut offenders: Vec<String> = Vec::new();
    for (value, at) in string_literals(&code) {
        if tables.iter().any(|table| table.contains(&at)) {
            continue;
        }
        outside += 1;
        if CHECKPOINT_PLAN_FAMILY_TOKENS.contains(&value.as_str()) {
            offenders.push(format!("{value:?}"));
        }
    }
    // Non-vacuity: the route is full of component ids, refusal codes and diagnostics, so a scanner
    // that saw only a handful of literals outside the tables is broken.
    assert!(
        outside >= 20,
        "the plan-route literal scan saw only {outside} literals outside the family tables — the \
         scan is broken, not the route"
    );
    assert!(
        offenders.is_empty(),
        "the plan-driven checkpoint route names a family outside its recorded catalog-data tables: \
         {}\nThe route is adapter-mediated (epic 20398 E2): resolve the behaviour from \
         `gen_core::CheckpointAdapterRegistration` — its `eligible_backends`, `component_topology`, \
         `base_compatibility` or per-operation capability policy — rather than branching on the \
         family name.",
        offenders.join(", ")
    );
}

/// The vocabulary the sweep judges by must cover every family the live tables actually name.
///
/// Without this, adding a family with a token the vocabulary does not carry would make it invisible
/// to [`no_family_allow_list_exists_outside_the_recorded_exception_inventory`] — the sweep would
/// keep passing while the very thing it exists to catch walked past it.
#[test]
fn the_family_token_vocabulary_covers_every_family_the_live_tables_name() {
    let vocabulary: BTreeSet<&str> = CHECKPOINT_PLAN_FAMILY_TOKENS.iter().copied().collect();
    for (file, name) in LIVE_FAMILY_TABLES {
        let body = recorded_const_body(file, name);
        let named: Vec<String> = string_literals(&body)
            .into_iter()
            .map(|(value, _)| value)
            .filter(|value| !value.is_empty())
            .collect();
        assert!(
            !named.is_empty(),
            "{file}: const {name} names no string at all — the table parse is broken"
        );
        for value in named {
            assert!(
                vocabulary.contains(value.as_str()),
                "{file}: const {name} names the family {value:?}, which \
                 CHECKPOINT_PLAN_FAMILY_TOKENS does not carry. Add it there, or the family \
                 allow-list sweep is blind to every gate written for it."
            );
        }
    }
}

/// Self-test the scanner rather than trusting it: a guard that silently matches nothing is worse
/// than no guard. Covers the type-vs-value `[` ambiguity, brackets inside literals, the
/// builtin-model-id collision, and the judgment that decides offender from non-offender.
#[test]
fn the_family_allow_list_scanner_recognises_the_shapes_it_must_judge() {
    // The `[` in the TYPE must never be read as the value. Without the after-`=` rule this returns
    // a "body" of `u8; 16` and the real value is never inspected.
    let byte_const = "const HEX: &[u8; 16] = b\"0123456789abcdef\";";
    assert!(
        slice_const_bodies(byte_const).is_empty(),
        "a byte-string constant declares no slice literal"
    );

    let families = "const F: &[&str] = &[\"krea_2\", \"mage-flow\"];";
    let parsed = slice_const_bodies(families);
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].0, "F");
    assert_eq!(
        family_tokens_named(&parsed[0].1),
        ["krea_2".to_owned(), "mage-flow".to_owned()]
            .into_iter()
            .collect()
    );
    assert!(is_family_allow_list(&family_tokens_named(&parsed[0].1)));
    // And the reported range really points at the body, so the plan-route test can mask it.
    assert_eq!(&families[parsed[0].2.clone()], parsed[0].1);

    // A bracket inside a literal must not close the body early.
    let bracketed = "const B: &[&str] = &[\"a]b\", \"wan-video\"];";
    let parsed = slice_const_bodies(bracketed);
    assert_eq!(parsed.len(), 1);
    assert!(is_family_allow_list(&family_tokens_named(&parsed[0].1)));

    // The `sdxl` collision: a builtin-model-id list whose only overlap is that word is NOT a family
    // allow-list, but the same list plus one real family token is.
    let model_ids = "const M: &[&str] = &[\"sdxl\", \"realvisxl\", \"illustrious_xl_v1\"];";
    let tokens = family_tokens_named(&slice_const_bodies(model_ids)[0].1);
    assert_eq!(tokens, ["sdxl".to_owned()].into_iter().collect());
    assert!(
        !is_family_allow_list(&tokens),
        "a builtin-model-id list must not be judged a family gate"
    );
    let widened = "const M: &[&str] = &[\"sdxl\", \"krea_2\"];";
    assert!(is_family_allow_list(&family_tokens_named(
        &slice_const_bodies(widened)[0].1
    )));
    // A single UNAMBIGUOUS token is a family gate on its own — `CHECKPOINT_PLAN_RESIDENT_BASE_TIERS`
    // and `MULTI_EXPERT_FAMILIES` are each exactly that shape.
    let single = "const S: &[(&str, fn())] = &[(\"mage-flow\", resolve)];";
    assert!(is_family_allow_list(&family_tokens_named(
        &slice_const_bodies(single)[0].1
    )));

    // A constant naming no family token is none of this audit's business.
    let unrelated = "const T: &[&str] = &[\"bf16\", \"q8\", \"q4\"];";
    assert!(!is_family_allow_list(&family_tokens_named(
        &slice_const_bodies(unrelated)[0].1
    )));

    // A comment naming a family cannot create an offender, and cannot hide a deleted one: the sweep
    // runs over `code_without_comments`.
    let commented = "// const C: &[&str] = &[\"krea_2\", \"flux2\"];\nconst D: u8 = 1;";
    assert!(slice_const_bodies(&code_without_comments(commented)).is_empty());

    // Quote parity. A `'"'` char literal ahead of a family gate must not swallow it, and a raw
    // string's unescaped inner quote must not invert everything after it. Both failures are SILENT
    // — the scan simply stops seeing gates — which is why they are asserted rather than assumed.
    let after_char_literal =
        "fn q(c: char) -> bool { c == '\"' }\nconst G: &[&str] = &[\"krea_2\", \"flux2\"];";
    let parsed = slice_const_bodies(after_char_literal);
    assert_eq!(
        parsed.len(),
        1,
        "a `'\"'` char literal must not hide the gate that follows it"
    );
    assert!(is_family_allow_list(&family_tokens_named(&parsed[0].1)));

    let after_raw_string =
        "const R: &str = r#\"a \"quoted\" fox\"#;\nconst H: &[&str] = &[\"mage-flow\"];";
    let parsed = slice_const_bodies(after_raw_string);
    let gate = parsed
        .iter()
        .find(|(name, _, _)| name == "H")
        .expect("a raw string's inner quotes must not hide the gate that follows it");
    assert!(is_family_allow_list(&family_tokens_named(&gate.1)));

    // `string_literals` needs the same defence in its OWN right, and it needs it asserted directly:
    // it is what `the_checkpoint_plan_route_names_a_family_only_inside_its_recorded_tables` runs
    // over a whole production file. The two cases above go through `matching_bracket`, so they stay
    // green even with this function's guard removed — a guard nothing can falsify is not a guard.
    //
    // Without the char-literal step, the `'"'` below opens a phantom literal that runs to the next
    // quote, `"krea_2"` is consumed as ordinary text, and the family gate after it is invisible.
    let literals = string_literals("fn q(c: char) -> bool { c == '\"' }\nlet f = \"krea_2\";");
    assert!(
        literals.iter().any(|(value, _)| value == "krea_2"),
        "a `'\"'` char literal must not swallow the family literal that follows it; found \
         {literals:?}"
    );
    // The raw-string half of the same claim.
    let literals =
        string_literals("const R: &str = r#\"a \"quoted\" fox\"#;\nlet f = \"mage-flow\";");
    assert!(
        literals.iter().any(|(value, _)| value == "mage-flow"),
        "a raw string's inner quotes must not swallow the family literal that follows it; found \
         {literals:?}"
    );
}

/// sc-21534 — the pre-loader source guard (sc-19708) can re-verify a multi-GB resolved bundle,
/// which outlasts the API's 90s stale-worker timeout. `run_utility_job` must therefore await the
/// guard's blocking task through the `heartbeat_while_blocking` keepalive (whose behavior is
/// pinned by `heartbeat_while_blocking_keeps_worker_live_through_a_long_pass`), never bare —
/// a bare `.await` is exactly how the Krea bf16 lost-heartbeat incident swept a healthy worker.
/// Source-structure assertion in the same style as the dispatch-matrix test above: behavior lives
/// in the wrapper's own test; this pins the WIRING at the one admission call site.
#[test]
fn the_source_guard_await_is_wrapped_in_the_heartbeat_keepalive() {
    let code = code_without_comments_or_literals(WORKER);
    let function = code
        .split_once("async fn run_utility_job(")
        .expect("worker must declare run_utility_job")
        .1;
    let after_guard_spawn = function
        .split_once("RuntimeSourceGuard::begin(")
        .expect("run_utility_job must admit every job through the source guard")
        .1;
    let between_guard_and_dispatch = after_guard_spawn
        .split_once("match job.job_type {")
        .expect("the guard must run before the job-type dispatch")
        .0;
    let after_wrapper = between_guard_and_dispatch
        .split_once("heartbeat_while_blocking(")
        .expect(
            "the source-guard blocking task must be awaited via heartbeat_while_blocking, not \
             bare — a guard pass longer than the stale-worker timeout would get the worker swept \
             mid-verify",
        )
        .1;
    // The wrapper must be wrapping THE GUARD's handle, not some other blocking task that later
    // lands in the same window while the guard goes back to a bare await.
    assert!(
        after_wrapper
            .split_once(')')
            .expect("the keepalive call must close its argument list")
            .0
            .contains("guard_task"),
        "heartbeat_while_blocking must receive the source guard's own JoinHandle (guard_task)"
    );
}
